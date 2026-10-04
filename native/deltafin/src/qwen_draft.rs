//! Raw-completion translation policy for untrusted Qwen proposals.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::draft::DraftSource;
use crate::error::{DeltafinError, Result};
use crate::packfile::{Digest, digest_bytes};
use crate::qwen_provider::{NativeQwen, NativeQwenGeneration};
use crate::tokenizer::K3Tokenizer;
use crate::sys::fs::{self as sys_fs, Open};

const TOKENIZER_BYTES: u64 = 7_031_645;
const TOKENIZER_SHA256: Digest = [
    0xc0, 0x38, 0x21, 0x17, 0xea, 0x32, 0x9c, 0xdf, 0x09, 0x70, 0x41, 0x13, 0x2f, 0x6d, 0x73, 0x59,
    0x24, 0xb6, 0x97, 0x92, 0x4d, 0x6f, 0x6f, 0xc3, 0x94, 0x57, 0x13, 0xe9, 0x6c, 0xe8, 0x75, 0x39,
];
const CONFIDENCE_THRESHOLD: f32 = 0.3;
const MAXIMUM_ASSISTANT_TOKENS: usize = 20;
const TRANSLATION_SLACK: usize = 2;

/// One untrusted Qwen proposal plus the reason it stopped early.
///
/// An empty confidence-stopped proposal is economically different from a
/// broken or empty assistant: it says that a wide target verifier is unlikely
/// to pay at this position, while a previously qualified request may safely
/// try again at the next position. The full K3 verifier remains the only
/// authority for every token in `token_ids`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QwenDraftProposal {
    token_ids: Box<[u32]>,
    raw_token_ids: Box<[u32]>,
    confidence_stopped: bool,
    minimum_confidence: Option<f32>,
}

impl QwenDraftProposal {
    fn empty() -> Self {
        Self {
            token_ids: Box::new([]),
            raw_token_ids: Box::new([]),
            confidence_stopped: false,
            minimum_confidence: None,
        }
    }

    fn new(
        token_ids: Box<[u32]>,
        raw_token_ids: Box<[u32]>,
        confidence_stopped: bool,
        minimum_confidence: Option<f32>,
    ) -> Self {
        Self {
            token_ids,
            raw_token_ids,
            confidence_stopped,
            minimum_confidence,
        }
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn confidence_stopped(&self) -> bool {
        self.confidence_stopped
    }

    pub fn raw_token_ids(&self) -> &[u32] {
        if self.raw_token_ids.is_empty() {
            &self.token_ids
        } else {
            &self.raw_token_ids
        }
    }

    pub fn minimum_confidence(&self) -> Option<f32> {
        self.minimum_confidence
    }

    pub fn into_token_ids(self) -> Box<[u32]> {
        self.token_ids
    }
}

const HYBRID_WIDE_CONFIDENCE_MARGIN: f32 = 0.02;
const ADAPTIVE_RAW_OVERRIDE_MINIMUM: usize = 3;

/// Result of the cheap assistant's first pass through the adaptive hybrid.
///
/// A complete retained proposal needs no second opinion. A confidence-cut
/// proposal may expose the same complete raw candidate after at least three
/// retained target tokens; that candidate is still untrusted and is safe to
/// submit because full K3 verifies every position. Only an early/incomplete
/// candidate asks the caller to materialize and run the larger assistant.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AdaptiveQwenProbeSelection {
    Selected {
        proposal: QwenDraftProposal,
        raw_override: bool,
    },
    NeedsWide(QwenDraftProposal),
}

pub(crate) fn select_adaptive_qwen_probe(
    probe: QwenDraftProposal,
    maximum: usize,
    raw_override_allowed: bool,
) -> AdaptiveQwenProbeSelection {
    if maximum == 0 || probe.token_ids().len() == maximum {
        return AdaptiveQwenProbeSelection::Selected {
            proposal: probe,
            raw_override: false,
        };
    }
    let raw = probe.raw_token_ids();
    if raw_override_allowed
        && raw.len() == maximum
        && probe.token_ids().len() >= ADAPTIVE_RAW_OVERRIDE_MINIMUM.min(maximum)
    {
        let raw = raw.to_vec().into_boxed_slice();
        return AdaptiveQwenProbeSelection::Selected {
            proposal: QwenDraftProposal::new(raw.clone(), raw, false, probe.minimum_confidence()),
            raw_override: true,
        };
    }
    AdaptiveQwenProbeSelection::NeedsWide(probe)
}

/// Reproduce the established probe-plus-wide selection policy without giving
/// either assistant authority over output. Complete agreement between their
/// raw translated drafts bypasses an isolated confidence false-negative;
/// otherwise the wider model wins only by a measured confidence margin.
pub fn select_hybrid_qwen_proposal(
    probe: QwenDraftProposal,
    wide: QwenDraftProposal,
    maximum: usize,
) -> QwenDraftProposal {
    let probe_raw = probe.raw_token_ids();
    let wide_raw = wide.raw_token_ids();
    if !probe_raw.is_empty() && probe_raw.len() == maximum && probe_raw == wide_raw {
        let consensus = probe_raw.to_vec().into_boxed_slice();
        let minimum_confidence = match (probe.minimum_confidence(), wide.minimum_confidence()) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left @ Some(_), None) | (None, left @ Some(_)) => left,
            (None, None) => None,
        };
        return QwenDraftProposal::new(consensus.clone(), consensus, false, minimum_confidence);
    }
    let score = |proposal: &QwenDraftProposal| proposal.minimum_confidence().unwrap_or(-1.0);
    if score(&wide) > score(&probe) + HYBRID_WIDE_CONFIDENCE_MARGIN {
        wide
    } else {
        probe
    }
}

pub trait TargetTextCodec: Send + Sync {
    fn decode_target(&self, tokens: &[u32]) -> Result<String>;
    fn encode_target_allow_special(&self, text: &str) -> Result<Vec<u32>>;
}

impl TargetTextCodec for K3Tokenizer {
    fn decode_target(&self, tokens: &[u32]) -> Result<String> {
        self.decode(tokens)
    }

    fn encode_target_allow_special(&self, text: &str) -> Result<Vec<u32>> {
        self.encode(text, true)
    }
}

pub trait AssistantTextCodec: Send + Sync {
    fn encode_raw(&self, text: &str) -> Result<Vec<u32>>;
    fn decode_raw(&self, tokens: &[u32]) -> Result<String>;
}

#[derive(Clone)]
pub struct QwenTokenizer {
    inner: Arc<Tokenizer>,
}

impl QwenTokenizer {
    pub fn load(model_directory: &Path) -> Result<Self> {
        let path = model_directory.join("tokenizer.json");
        let mut file = Open::new()
            .read(true)
            .no_follow()
            .open(&path)
            .map_err(|error| DeltafinError::new(format!("open pinned Qwen tokenizer: {error}")))?;
        let metadata = sys_fs::fstat(&file)
            .map_err(|error| DeltafinError::new(format!("stat pinned Qwen tokenizer: {error}")))?;
        if !metadata.is_file() || metadata.len() != TOKENIZER_BYTES {
            return Err(DeltafinError::new(
                "pinned Qwen tokenizer is not the exact regular file",
            ));
        }
        let mut bytes = Vec::with_capacity(TOKENIZER_BYTES as usize);
        file.read_to_end(&mut bytes)
            .map_err(|error| DeltafinError::new(format!("read pinned Qwen tokenizer: {error}")))?;
        let final_metadata = sys_fs::fstat(&file).map_err(|error| {
            DeltafinError::new(format!("restat pinned Qwen tokenizer: {error}"))
        })?;
        if metadata.len() != final_metadata.len()
            || metadata.id() != final_metadata.id()
            || digest_bytes(&bytes) != TOKENIZER_SHA256
        {
            return Err(DeltafinError::new(
                "Qwen tokenizer changed or differs from its inert pin",
            ));
        }
        let inner = Tokenizer::from_bytes(bytes)
            .map_err(|error| DeltafinError::new(format!("parse Qwen tokenizer: {error}")))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

impl AssistantTextCodec for QwenTokenizer {
    fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| DeltafinError::new(format!("encode Qwen completion: {error}")))
    }

    fn decode_raw(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, false)
            .map_err(|error| DeltafinError::new(format!("decode Qwen completion: {error}")))
    }
}

pub trait QwenGenerator {
    fn generate(&mut self, input: &[u32], maximum_new: usize) -> Result<NativeQwenGeneration>;
}

impl QwenGenerator for NativeQwen {
    fn generate(&mut self, input: &[u32], maximum_new: usize) -> Result<NativeQwenGeneration> {
        NativeQwen::generate(self, input, maximum_new)
    }
}

pub struct QwenDraftController<G, T, A> {
    generator: G,
    target: Arc<T>,
    assistant: A,
}

impl<G, T, A> QwenDraftController<G, T, A>
where
    G: QwenGenerator,
    T: TargetTextCodec,
    A: AssistantTextCodec,
{
    pub fn new(generator: G, target: Arc<T>, assistant: A) -> Self {
        Self {
            generator,
            target,
            assistant,
        }
    }

    fn translate_once(
        &mut self,
        target_history: &[u32],
        requested: usize,
        budget: usize,
    ) -> Result<(Box<[u32]>, Box<[u32]>, bool, bool, Option<f32>)> {
        let canonical = self.target.decode_target(target_history)?;
        if self.target.encode_target_allow_special(&canonical)? != target_history {
            return Err(DeltafinError::new(
                "K3 history does not survive the canonical raw-completion round trip",
            ));
        }
        let assistant_prefix = self.assistant.encode_raw(&canonical)?;
        if assistant_prefix.is_empty() {
            return Ok((Box::new([]), Box::new([]), false, false, None));
        }
        let generated = self.generator.generate(&assistant_prefix, budget)?;
        if generated.token_ids.len() != generated.probabilities.len() {
            return Err(DeltafinError::new("Qwen token/confidence rows disagree"));
        }
        let use_confidence = requested > 2;
        let accepted = if use_confidence {
            generated
                .probabilities
                .iter()
                .position(|&probability| probability < CONFIDENCE_THRESHOLD)
                .unwrap_or(generated.token_ids.len())
        } else {
            generated.token_ids.len()
        };
        let confidence_stopped = accepted < generated.token_ids.len();
        let minimum_confidence = use_confidence
            .then(|| generated.probabilities.iter().copied().reduce(f32::min))
            .flatten();
        let hit_budget = generated.token_ids.len() >= budget;
        let mut raw_complete = assistant_prefix;
        raw_complete.extend_from_slice(&generated.token_ids);
        let raw_text = self.assistant.decode_raw(&raw_complete)?;
        let raw_translated = self.target.encode_target_allow_special(&raw_text)?;
        let translated = if confidence_stopped {
            raw_complete.truncate(raw_complete.len() - generated.token_ids.len() + accepted);
            let retained_text = self.assistant.decode_raw(&raw_complete)?;
            self.target.encode_target_allow_special(&retained_text)?
        } else {
            raw_translated.clone()
        };
        if !raw_translated.starts_with(target_history) || !translated.starts_with(target_history) {
            return Err(DeltafinError::new(
                "assistant text did not preserve the exact K3 token prefix",
            ));
        }
        let take_suffix = |tokens: &[u32]| {
            tokens[target_history.len()..]
                .iter()
                .copied()
                .take(requested)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        Ok((
            take_suffix(&translated),
            take_suffix(&raw_translated),
            confidence_stopped,
            hit_budget,
            minimum_confidence,
        ))
    }

    pub fn propose_with_outcome(
        &mut self,
        target_history: &[u32],
        maximum: usize,
    ) -> Result<QwenDraftProposal> {
        if maximum == 0 || target_history.is_empty() {
            return Ok(QwenDraftProposal::empty());
        }
        let budget = maximum
            .saturating_add(TRANSLATION_SLACK)
            .clamp(3, MAXIMUM_ASSISTANT_TOKENS);
        let (draft, raw_draft, confidence_stopped, hit_budget, minimum_confidence) =
            self.translate_once(target_history, maximum, budget)?;
        if draft.len() >= maximum
            || confidence_stopped
            || !hit_budget
            || budget == MAXIMUM_ASSISTANT_TOKENS
        {
            return Ok(QwenDraftProposal::new(
                draft,
                raw_draft,
                confidence_stopped,
                minimum_confidence,
            ));
        }
        let (retry, retry_raw, retry_confidence_stopped, _, retry_minimum_confidence) =
            self.translate_once(target_history, maximum, MAXIMUM_ASSISTANT_TOKENS)?;
        Ok(QwenDraftProposal::new(
            retry,
            retry_raw,
            retry_confidence_stopped,
            retry_minimum_confidence,
        ))
    }

    pub fn propose_strict(&mut self, target_history: &[u32], maximum: usize) -> Result<Box<[u32]>> {
        self.propose_with_outcome(target_history, maximum)
            .map(QwenDraftProposal::into_token_ids)
    }
}

/// Converts every assistant admission, tokenizer, or provider failure to an
/// empty proposal and permanently disables this source. Full K3 remains live.
pub struct FailSoftQwenDraft<G, T, A> {
    controller: QwenDraftController<G, T, A>,
    enabled: bool,
    last_error: Option<String>,
}

impl<G, T, A> FailSoftQwenDraft<G, T, A>
where
    G: QwenGenerator,
    T: TargetTextCodec,
    A: AssistantTextCodec,
{
    pub fn new(controller: QwenDraftController<G, T, A>) -> Self {
        Self {
            controller,
            enabled: true,
            last_error: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn propose_with_outcome(
        &mut self,
        target_history: &[u32],
        maximum: usize,
    ) -> Result<QwenDraftProposal> {
        if !self.enabled {
            return Ok(QwenDraftProposal::empty());
        }
        match self
            .controller
            .propose_with_outcome(target_history, maximum)
        {
            Ok(proposal) => Ok(proposal),
            Err(error) => {
                self.enabled = false;
                self.last_error = Some(error.to_string());
                Ok(QwenDraftProposal::empty())
            }
        }
    }
}

impl<G, T, A> DraftSource for FailSoftQwenDraft<G, T, A>
where
    G: QwenGenerator,
    T: TargetTextCodec,
    A: AssistantTextCodec,
{
    fn propose(&mut self, target_history: &[u32], maximum: usize) -> Result<Box<[u32]>> {
        self.propose_with_outcome(target_history, maximum)
            .map(QwenDraftProposal::into_token_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::platform::Device;
    #[cfg(target_os = "macos")]
    use crate::provider::NativeProviderSession;
    #[cfg(target_os = "macos")]
    use crate::qwen_checkpoint::QwenCheckpoint;
    use crate::qwen_checkpoint::QwenVariant;

    #[derive(Default)]
    struct IdentityCodec;

    impl TargetTextCodec for IdentityCodec {
        fn decode_target(&self, tokens: &[u32]) -> Result<String> {
            String::from_utf8(tokens.iter().map(|&token| token as u8).collect())
                .map_err(|error| DeltafinError::new(error.to_string()))
        }

        fn encode_target_allow_special(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }
    }

    impl AssistantTextCodec for IdentityCodec {
        fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode_raw(&self, tokens: &[u32]) -> Result<String> {
            String::from_utf8(tokens.iter().map(|&token| token as u8).collect())
                .map_err(|error| DeltafinError::new(error.to_string()))
        }
    }

    struct DroppingTargetCodec;

    impl TargetTextCodec for DroppingTargetCodec {
        fn decode_target(&self, tokens: &[u32]) -> Result<String> {
            IdentityCodec.decode_target(tokens)
        }

        fn encode_target_allow_special(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text
                .bytes()
                .filter(|byte| *byte != b'_')
                .map(u32::from)
                .collect())
        }
    }

    struct FakeGenerator {
        rows: Vec<NativeQwenGeneration>,
        budgets: Vec<usize>,
        inputs: Vec<Vec<u32>>,
        fail: bool,
    }

    impl QwenGenerator for FakeGenerator {
        fn generate(&mut self, input: &[u32], maximum_new: usize) -> Result<NativeQwenGeneration> {
            self.budgets.push(maximum_new);
            self.inputs.push(input.to_vec());
            if self.fail {
                return Err(DeltafinError::new("synthetic assistant failure"));
            }
            Ok(self.rows.remove(0))
        }
    }

    fn generation(text: &str, probabilities: &[f32]) -> NativeQwenGeneration {
        NativeQwenGeneration {
            token_ids: text
                .bytes()
                .map(u32::from)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            probabilities: probabilities.into(),
        }
    }

    #[test]
    fn translation_is_suffix_only_and_retries_at_twenty() {
        let generator = FakeGenerator {
            rows: vec![
                generation("x____", &[0.9, 0.9, 0.9, 0.9, 0.9]),
                generation("xyz", &[0.9, 0.9, 0.9]),
            ],
            budgets: Vec::new(),
            inputs: Vec::new(),
            fail: false,
        };
        let mut source =
            QwenDraftController::new(generator, Arc::new(DroppingTargetCodec), IdentityCodec);
        assert_eq!(
            &*source
                .propose_strict(
                    b"ab"
                        .iter()
                        .map(|&x| u32::from(x))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    3
                )
                .unwrap(),
            b"xyz"
                .iter()
                .map(|&x| u32::from(x))
                .collect::<Vec<_>>()
                .as_slice()
        );
        assert_eq!(source.generator.budgets, vec![5, 20]);
    }

    #[test]
    fn confidence_stops_without_retry_and_never_returns_low_row() {
        let generator = FakeGenerator {
            rows: vec![generation("xyz", &[0.9, 0.2, 0.9])],
            budgets: Vec::new(),
            inputs: Vec::new(),
            fail: false,
        };
        let mut source =
            QwenDraftController::new(generator, Arc::new(IdentityCodec), IdentityCodec);
        let proposal = source.propose_with_outcome(&[b'a' as u32], 4).unwrap();
        assert_eq!(proposal.token_ids(), &[b'x' as u32]);
        assert!(proposal.confidence_stopped());
        assert_eq!(source.generator.budgets, vec![6]);
    }

    #[test]
    fn hybrid_consensus_restores_complete_raw_draft_but_divergence_keeps_confidence_policy() {
        let truncated = QwenDraftProposal::new(
            vec![10, 11].into_boxed_slice(),
            vec![10, 11, 12, 13].into_boxed_slice(),
            true,
            Some(0.2),
        );
        let agreeing = QwenDraftProposal::new(
            vec![10, 11, 12].into_boxed_slice(),
            vec![10, 11, 12, 13].into_boxed_slice(),
            true,
            Some(0.25),
        );
        let consensus = select_hybrid_qwen_proposal(truncated.clone(), agreeing, 4);
        assert_eq!(consensus.token_ids(), &[10, 11, 12, 13]);
        assert!(!consensus.confidence_stopped());
        assert_eq!(consensus.minimum_confidence(), Some(0.25));

        let stronger_wide = QwenDraftProposal::new(
            vec![20, 21].into_boxed_slice(),
            vec![20, 21, 22, 23].into_boxed_slice(),
            true,
            Some(0.31),
        );
        let selected = select_hybrid_qwen_proposal(truncated, stronger_wide, 4);
        assert_eq!(selected.token_ids(), &[20, 21]);
        assert!(selected.confidence_stopped());
    }

    #[test]
    fn adaptive_probe_skips_wide_for_complete_or_bounded_raw_candidates() {
        let complete = QwenDraftProposal::new(
            vec![10, 11, 12, 13].into_boxed_slice(),
            vec![10, 11, 12, 13].into_boxed_slice(),
            false,
            Some(0.8),
        );
        assert_eq!(
            select_adaptive_qwen_probe(complete.clone(), 4, true),
            AdaptiveQwenProbeSelection::Selected {
                proposal: complete,
                raw_override: false,
            }
        );

        let confidence_cut = QwenDraftProposal::new(
            vec![20, 21, 22].into_boxed_slice(),
            vec![20, 21, 22, 23].into_boxed_slice(),
            true,
            Some(0.2),
        );
        assert_eq!(
            select_adaptive_qwen_probe(confidence_cut.clone(), 4, true),
            AdaptiveQwenProbeSelection::Selected {
                proposal: QwenDraftProposal::new(
                    vec![20, 21, 22, 23].into_boxed_slice(),
                    vec![20, 21, 22, 23].into_boxed_slice(),
                    false,
                    Some(0.2),
                ),
                raw_override: true,
            }
        );
        assert_eq!(
            select_adaptive_qwen_probe(confidence_cut, 4, false),
            AdaptiveQwenProbeSelection::NeedsWide(QwenDraftProposal::new(
                vec![20, 21, 22].into_boxed_slice(),
                vec![20, 21, 22, 23].into_boxed_slice(),
                true,
                Some(0.2),
            ))
        );
    }

    #[test]
    fn adaptive_probe_requires_wide_for_early_uncertainty() {
        let uncertain = QwenDraftProposal::new(
            vec![30, 31].into_boxed_slice(),
            vec![30, 31, 32, 33].into_boxed_slice(),
            true,
            Some(0.1),
        );
        assert_eq!(
            select_adaptive_qwen_probe(uncertain.clone(), 4, true),
            AdaptiveQwenProbeSelection::NeedsWide(uncertain)
        );
    }

    #[test]
    fn confidence_can_skip_one_position_without_disabling_the_source() {
        let generator = FakeGenerator {
            rows: vec![generation("x", &[0.2]), generation("y", &[0.9])],
            budgets: Vec::new(),
            inputs: Vec::new(),
            fail: false,
        };
        let controller =
            QwenDraftController::new(generator, Arc::new(IdentityCodec), IdentityCodec);
        let mut source = FailSoftQwenDraft::new(controller);
        let skipped = source.propose_with_outcome(&[97], 4).unwrap();
        assert!(skipped.token_ids().is_empty());
        assert!(skipped.confidence_stopped());
        assert!(source.is_enabled());
        let next = source.propose_with_outcome(&[97, 98], 4).unwrap();
        assert_eq!(next.token_ids(), &[b'y' as u32]);
        assert!(source.is_enabled());
    }

    #[test]
    fn each_proposal_resends_complete_history_without_cross_call_state() {
        let generator = FakeGenerator {
            rows: vec![generation("x", &[0.9]), generation("y", &[0.9])],
            budgets: Vec::new(),
            inputs: Vec::new(),
            fail: false,
        };
        let mut source =
            QwenDraftController::new(generator, Arc::new(IdentityCodec), IdentityCodec);
        assert_eq!(&*source.propose_strict(&[97], 1).unwrap(), &[120]);
        assert_eq!(&*source.propose_strict(&[97, 98], 1).unwrap(), &[121]);
        assert_eq!(source.generator.inputs, vec![vec![97], vec![97, 98]]);
    }

    #[test]
    fn provider_failure_disables_only_the_proposal_source() {
        let generator = FakeGenerator {
            rows: vec![],
            budgets: vec![],
            inputs: vec![],
            fail: true,
        };
        let controller =
            QwenDraftController::new(generator, Arc::new(IdentityCodec), IdentityCodec);
        let mut source = FailSoftQwenDraft::new(controller);
        assert!(source.propose(&[97], 3).unwrap().is_empty());
        assert!(!source.is_enabled());
        assert!(
            source
                .last_error()
                .unwrap()
                .contains("synthetic assistant failure")
        );
        assert!(source.propose(&[97], 3).unwrap().is_empty());
    }

    #[test]
    fn installed_inert_tokenizers_parse_and_preserve_raw_text() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for variant in [QwenVariant::Probe06B, QwenVariant::Wide17B] {
            let directory = root.join(variant.directory());
            if !directory.is_dir() {
                continue;
            }
            let tokenizer = QwenTokenizer::load(&directory).unwrap();
            let text = "raw completion <|endoftext|> café\n";
            let tokens = tokenizer.encode_raw(text).unwrap();
            assert_eq!(tokenizer.decode_raw(&tokens).unwrap(), text);
        }
    }

    #[test]
    fn qwen_tokenizer_matches_frozen_corpus() {
        // Frozen with tokenizers 0.22.2's Oniguruma backend. The cases cover
        // every alternative in Qwen's split expression, NFC normalization,
        // ByteLevel encoding, and added-token recognition so dependency or
        // tokenizer backend changes remain conditional on exact token parity.
        let corpus: &[(&str, &[u32])] = &[
            ("", &[]),
            (
                "raw completion <|endoftext|> café\n",
                &[1041, 9755, 220, 151643, 51950, 198],
            ),
            ("Hello, world!", &[9707, 11, 1879, 0]),
            (
                "I’M I'm we're they've I'll I'd",
                &[
                    40, 527, 44, 358, 2776, 582, 2299, 807, 3003, 358, 3278, 358, 4172,
                ],
            ),
            ("a   b    ", &[64, 256, 293, 257]),
            (
                "one\r\ntwo\n\nthree\t end",
                &[603, 319, 19789, 271, 27856, 197, 835],
            ),
            (
                "中文 Ελληνικά русский हिन्दी العربية",
                &[
                    104811, 7851, 243, 33486, 33486, 41424, 33269, 29762, 67337, 74134, 18108,
                    43055, 126302, 84310, 42311, 101, 30484, 99, 43647, 129071,
                ],
            ),
            (
                "emoji 👩🏽‍💻🚀 — café cafe\u{301}",
                &[
                    37523, 61804, 102, 145375, 378, 235, 145851, 145836, 1959, 51950, 51950,
                ],
            ),
            (
                "0 12 345 6789 1000000",
                &[
                    15, 220, 16, 17, 220, 18, 19, 20, 220, 21, 22, 23, 24, 220, 16, 15, 15, 15, 15,
                    15, 15,
                ],
            ),
            (
                "<|im_start|>assistant\n<|im_end|>\n",
                &[151644, 77091, 198, 151645, 198],
            ),
            (
                "fn main() { println!(\"hello\"); } // comment\n",
                &[
                    8822, 1887, 368, 314, 13751, 17223, 14990, 5038, 335, 442, 3980, 198,
                ],
            ),
        ];

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let Some(directory) = [QwenVariant::Probe06B, QwenVariant::Wide17B]
            .into_iter()
            .map(|variant| root.join(variant.directory()))
            .find(|directory| directory.is_dir())
        else {
            return;
        };
        let tokenizer = QwenTokenizer::load(&directory).unwrap();
        for &(text, expected) in corpus {
            assert_eq!(tokenizer.encode_raw(text).unwrap(), expected, "{text:?}");
        }
    }

    /// Physical, opt-in canary for the adaptive native raw-completion proposal
    /// path. This opens only the tokenizer plus the small optional Qwen; K3
    /// weights remain unopened and every returned ID is still merely a draft.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires the optional pinned Qwen 0.6B checkpoint and an MPS host"]
    fn installed_adaptive_qwen_skips_wide_for_the_frozen_paris_schedule() {
        let _device = crate::provider::exclusive_mps_device();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let target = Arc::new(K3Tokenizer::load_from_root(&root).expect("load K3 tokenizer"));
        let probe_checkpoint = QwenCheckpoint::open(&root, QwenVariant::Probe06B)
            .expect("open installed pinned Qwen 0.6B checkpoint");
        let probe_tokenizer =
            QwenTokenizer::load(probe_checkpoint.root()).expect("load Qwen 0.6B tokenizer");
        let session = NativeProviderSession::target(Device::Mps).expect("create MPS provider");
        let probe_model =
            NativeQwen::bind_with_context_capacity(&session, &probe_checkpoint, 4_389)
                .expect("bind native Qwen 0.6B");
        let mut probe = QwenDraftController::new(probe_model, target, probe_tokenizer);
        let cases: &[(&[u32], usize, &[u32])] = &[
            (&[1_008, 10_484, 318, 15_383, 387, 17_374], 2, &[13, 646]),
            (
                &[1_008, 10_484, 318, 15_383, 387, 17_374, 13, 646, 606],
                8,
                &[142_957, 37_092, 387, 7_081, 306, 17_374, 13, 646],
            ),
            (
                &[
                    1_008, 10_484, 318, 15_383, 387, 17_374, 13, 646, 606, 142_957, 37_092, 387,
                    7_081, 306, 17_374, 13, 646, 14_715,
                ],
                3,
                &[91_527, 16_575, 387],
            ),
        ];
        let wide_invocations = 0_u64;
        let mut wide_skips = 0_u64;
        let mut raw_override_selections = 0_u64;
        let mut proposal_seconds = Vec::with_capacity(cases.len());
        for &(history, maximum, expected) in cases {
            let proposal_started = std::time::Instant::now();
            let probe_proposal = probe
                .propose_with_outcome(history, maximum)
                .expect("translate native Qwen probe proposal");
            let proposal = if maximum <= 2 {
                probe_proposal
            } else {
                match select_adaptive_qwen_probe(probe_proposal, maximum, true) {
                    AdaptiveQwenProbeSelection::Selected {
                        proposal,
                        raw_override,
                    } => {
                        wide_skips += 1;
                        raw_override_selections += u64::from(raw_override);
                        proposal
                    }
                    AdaptiveQwenProbeSelection::NeedsWide(_) => {
                        panic!("the frozen Paris schedule unexpectedly requested Qwen 1.7B")
                    }
                }
            };
            proposal_seconds.push(proposal_started.elapsed().as_secs_f64());
            eprintln!(
                "native adaptive Qwen history={} maximum={maximum} proposal={:?} raw={:?} confidence_stopped={}",
                history.len(),
                proposal.token_ids(),
                proposal.raw_token_ids(),
                proposal.confidence_stopped(),
            );
            assert_eq!(proposal.token_ids(), expected);
            assert!(!proposal.confidence_stopped());
        }
        eprintln!(
            "native adaptive Qwen telemetry: wide_invocations={wide_invocations} wide_skips={wide_skips} raw_override_selections={raw_override_selections} proposal_seconds={proposal_seconds:?} total_proposal_seconds={:.6}",
            proposal_seconds.iter().sum::<f64>(),
        );
        assert_eq!(wide_invocations, 0);
        assert_eq!(wide_skips, 2);
        assert_eq!(raw_override_selections, 1);
    }
}
