//! Transactional, target-authoritative DSpark scheduling.
//!
//! This module deliberately has no tensor, device, checkpoint, tokenizer, or
//! HTTP dependencies.  A provider implements [`DraftBackend`]; this controller
//! owns only exact token ledgers, opaque cache snapshots, proposal leases, and
//! admission economics.  DSpark candidates are always untrusted.  The caller
//! may submit them to the full K3 verifier, but only target-certified tokens
//! may be passed back to [`DSparkRuntime::resolve`].
//!
//! A target-cache hit is an independent fact.  Missing, stale, corrupt, or
//! uneconomic DSpark state disables this optional controller and can never ask
//! the target to re-prefill.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::time::{Duration, Instant};

pub const STATE_FORMAT: &str = "deltafin-dspark-state-v1";
pub const DEFAULT_VOCAB_SIZE: u32 = 163_840;
pub const PROBE_DRAFTS: u8 = 2;
pub const MAX_DRAFTS: u8 = 7;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Mode {
    Off,
    On,
    Auto,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ErrorKind {
    Configuration,
    Protocol,
    State,
    Backend,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RuntimeError {
    kind: ErrorKind,
    message: String,
}

impl RuntimeError {
    fn configuration(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Configuration,
            message: message.into(),
        }
    }

    fn protocol(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Protocol,
            message: message.into(),
        }
    }

    fn state(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::State,
            message: message.into(),
        }
    }

    fn backend(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Backend,
            message: message.into(),
        }
    }

    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for RuntimeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl Error for RuntimeError {}

pub type Result<T> = std::result::Result<T, RuntimeError>;

/// A provider error is optional by construction.  The release hint is only a
/// residency-policy signal; it never changes target execution or acceptance.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BackendFailure {
    message: String,
    release_optional_drafter: bool,
}

impl BackendFailure {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            release_optional_drafter: false,
        }
    }

    pub fn releasable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            release_optional_drafter: true,
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub const fn release_optional_drafter(&self) -> bool {
        self.release_optional_drafter
    }
}

impl Display for BackendFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl Error for BackendFailure {}

/// Exact identity of every property that can change draft-cache meaning.
///
/// Private fields prevent callers from constructing an incomplete identity.
/// The checkpoint SHA is binary so malformed or ambiguously cased hex cannot
/// enter state pairing.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ModelIdentity {
    adapter: String,
    weights_sha256: [u8; 32],
    trained_target_revision: String,
    runtime_target_revision: String,
    tokenizer_identity: String,
    cache_geometry: String,
    numerical_mode: String,
    device: String,
}

impl ModelIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adapter: impl Into<String>,
        weights_sha256: [u8; 32],
        trained_target_revision: impl Into<String>,
        runtime_target_revision: impl Into<String>,
        tokenizer_identity: impl Into<String>,
        cache_geometry: impl Into<String>,
        numerical_mode: impl Into<String>,
        device: impl Into<String>,
    ) -> Result<Self> {
        let identity = Self {
            adapter: adapter.into(),
            weights_sha256,
            trained_target_revision: trained_target_revision.into(),
            runtime_target_revision: runtime_target_revision.into(),
            tokenizer_identity: tokenizer_identity.into(),
            cache_geometry: cache_geometry.into(),
            numerical_mode: numerical_mode.into(),
            device: device.into(),
        };
        for (label, value) in [
            ("adapter", identity.adapter.as_str()),
            (
                "trained target revision",
                identity.trained_target_revision.as_str(),
            ),
            (
                "runtime target revision",
                identity.runtime_target_revision.as_str(),
            ),
            ("tokenizer identity", identity.tokenizer_identity.as_str()),
            ("cache geometry", identity.cache_geometry.as_str()),
            ("numerical mode", identity.numerical_mode.as_str()),
            ("device", identity.device.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(RuntimeError::configuration(format!(
                    "DSpark {label} must not be empty"
                )));
            }
        }
        Ok(identity)
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    pub const fn weights_sha256(&self) -> &[u8; 32] {
        &self.weights_sha256
    }

    pub fn trained_target_revision(&self) -> &str {
        &self.trained_target_revision
    }

    pub fn runtime_target_revision(&self) -> &str {
        &self.runtime_target_revision
    }

    pub fn tokenizer_identity(&self) -> &str {
        &self.tokenizer_identity
    }

    pub fn cache_geometry(&self) -> &str {
        &self.cache_geometry
    }

    pub fn numerical_mode(&self) -> &str {
        &self.numerical_mode
    }

    pub fn device(&self) -> &str {
        &self.device
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BackendProposal {
    token_ids: Box<[u32]>,
}

impl BackendProposal {
    pub fn new(token_ids: impl Into<Box<[u32]>>) -> Self {
        Self {
            token_ids: token_ids.into(),
        }
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

/// Provider boundary for model-free controller tests and compiled backends.
///
/// `state_token_count` counts committed, target-derived rows only.  Proposal
/// query rows are ephemeral and must not alter it.  Snapshots should be cheap
/// opaque handles; cloning one must not copy tensor payloads.
pub trait DraftBackend {
    type Snapshot: Clone;
    type TargetContext: ?Sized;

    fn reset_state(&mut self) -> std::result::Result<(), BackendFailure>;
    fn snapshot_state(&mut self) -> std::result::Result<Self::Snapshot, BackendFailure>;
    fn restore_state(
        &mut self,
        snapshot: &Self::Snapshot,
    ) -> std::result::Result<(), BackendFailure>;
    fn state_token_count(&mut self) -> std::result::Result<usize, BackendFailure>;
    fn model_identity(&mut self) -> std::result::Result<ModelIdentity, BackendFailure>;
    fn propose(
        &mut self,
        pending_token_id: u32,
        max_drafts: u8,
    ) -> std::result::Result<BackendProposal, BackendFailure>;
    /// Append the committed target rows in `target_context`. Row `i` is the
    /// position that consumed `committed_input_ids[i]`; a backend that pairs
    /// each row with its successor token (EAGLE-3) reads the IDs, while one
    /// that consumes rows alone (DSpark) needs only their count.
    fn advance_target_state(
        &mut self,
        target_context: &Self::TargetContext,
        committed_input_ids: &[u32],
    ) -> std::result::Result<(), BackendFailure>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeConfig {
    pub vocab_size: u32,
    pub probe_drafts: u8,
    pub max_drafts: u8,
    pub max_context_tokens: Option<usize>,
    pub min_auto_speedup: f64,
    /// Whether one verified proposal with no accepted draft ends proposals
    /// for the request. Right for a drafter whose first guess is rarely
    /// right (DSpark); a chain drafter that agrees with K3 ~90% of the time
    /// (EAGLE-3) instead narrows to the probe width and keeps going.
    pub disable_on_miss: bool,
    /// Automatic-mode verified steps observed before a negative cumulative
    /// credit, a losing probe or a loss streak may disable proposals. One
    /// early miss is noise for a drafter that usually wins.
    pub economic_grace_steps: u32,
    /// Consecutive clearly-losing automatic steps that disable proposals.
    pub loss_streak_limit: u8,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            vocab_size: DEFAULT_VOCAB_SIZE,
            probe_drafts: PROBE_DRAFTS,
            max_drafts: MAX_DRAFTS,
            max_context_tokens: Some(8_192),
            min_auto_speedup: 0.03,
            disable_on_miss: true,
            economic_grace_steps: 0,
            loss_streak_limit: 2,
        }
    }
}

impl RuntimeConfig {
    fn validate(self) -> Result<Self> {
        if self.vocab_size == 0 {
            return Err(RuntimeError::configuration(
                "DSpark vocabulary size must be positive",
            ));
        }
        if self.probe_drafts == 0
            || self.probe_drafts > self.max_drafts
            || self.max_drafts > MAX_DRAFTS
        {
            return Err(RuntimeError::configuration(
                "DSpark widths must satisfy 1 <= probe <= max <= 7",
            ));
        }
        if self.max_context_tokens == Some(0) {
            return Err(RuntimeError::configuration(
                "DSpark max context must be positive",
            ));
        }
        if !self.min_auto_speedup.is_finite() || !(0.0..1.0).contains(&self.min_auto_speedup) {
            return Err(RuntimeError::configuration(
                "DSpark minimum automatic speedup must be finite in [0, 1)",
            ));
        }
        if self.loss_streak_limit == 0 {
            return Err(RuntimeError::configuration(
                "DSpark loss-streak limit must be positive",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct BoundaryId(BoundaryIdValue);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum BoundaryIdValue {
    Numeric(u64),
    Text(String),
}

impl BoundaryId {
    pub const fn numeric(value: u64) -> Self {
        Self(BoundaryIdValue::Numeric(value))
    }

    pub fn text(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty() {
            return Err(RuntimeError::protocol(
                "target boundary ID must not be empty",
            ));
        }
        Ok(Self(BoundaryIdValue::Text(value)))
    }

    fn unpaired() -> Self {
        Self(BoundaryIdValue::Text(String::new()))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TargetCache {
    Miss,
    Hit {
        cached_tokens: usize,
        boundary_id: BoundaryId,
    },
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct StateSignature {
    format: &'static str,
    model: ModelIdentity,
    token_count: usize,
}

#[derive(Debug, Clone)]
struct StateSnapshot<S: Clone> {
    signature: StateSignature,
    opaque: S,
}

#[derive(Debug, Clone)]
struct Boundary<S: Clone> {
    token_ids: Box<[u32]>,
    state: StateSnapshot<S>,
    target_boundary_id: BoundaryId,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BoundaryStageToken {
    lease_id: u64,
    generation: u64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DraftLease {
    lease_id: u64,
    enabled: bool,
    target_cache_hit: bool,
    draft_cache_hit: bool,
    cached_tokens: usize,
    reason: Option<String>,
}

impl DraftLease {
    pub const fn lease_id(&self) -> u64 {
        self.lease_id
    }

    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    pub const fn target_cache_hit(&self) -> bool {
        self.target_cache_hit
    }

    pub const fn draft_cache_hit(&self) -> bool {
        self.draft_cache_hit
    }

    pub const fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// This is intentionally constant: optional draft state has no authority
    /// to invalidate or replay the target's retained cache.
    pub const fn requires_target_reprefill(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DraftProposal {
    lease_id: u64,
    proposal_id: u64,
    token_ids: Box<[u32]>,
    generated_token_ids: Box<[u32]>,
    probe: bool,
    base_state: StateSignature,
    pending_token_id: u32,
    seconds: f64,
}

impl DraftProposal {
    pub const fn proposal_id(&self) -> u64 {
        self.proposal_id
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn generated_token_ids(&self) -> &[u32] {
        &self.generated_token_ids
    }

    pub const fn is_probe(&self) -> bool {
        self.probe
    }

    pub const fn pending_token_id(&self) -> u32 {
        self.pending_token_id
    }

    pub const fn seconds(&self) -> f64 {
        self.seconds
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DisableCategory {
    Economics,
    AcceptanceEconomics,
    OptionalBackend,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AdmissionDecision {
    proposals_enabled: bool,
    state_aligned: bool,
    qualified: bool,
    next_width: u8,
    reason: Option<String>,
    disable_category: Option<DisableCategory>,
}

impl AdmissionDecision {
    pub const fn proposals_enabled(&self) -> bool {
        self.proposals_enabled
    }

    pub const fn state_aligned(&self) -> bool {
        self.state_aligned
    }

    pub const fn qualified(&self) -> bool {
        self.qualified
    }

    pub const fn next_width(&self) -> u8 {
        self.next_width
    }

    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    pub const fn disable_category(&self) -> Option<DisableCategory> {
        self.disable_category
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Metrics {
    pub sessions: u64,
    pub enabled_sessions: u64,
    pub disabled_sessions: u64,
    pub runtime_disables: u64,
    pub context_limit_disables: u64,
    pub target_cache_hits: u64,
    pub draft_cache_hits: u64,
    pub missing_draft_state: u64,
    pub proposals: u64,
    pub proposal_failures: u64,
    pub generated_drafts: u64,
    pub submitted_drafts: u64,
    pub accepted_drafts: u64,
    pub emitted_tokens: u64,
    pub full_matches: u64,
    pub partial_matches: u64,
    pub misses: u64,
    pub probe_qualifications: u64,
    pub state_failures: u64,
    pub proposal_aborts: u64,
    pub request_aborts: u64,
    pub boundaries_staged: u64,
    pub boundaries_published: u64,
    pub boundary_aborts: u64,
    pub obsolete_boundaries_discarded: u64,
    pub baseline_passes: u64,
    pub baseline_seconds: f64,
    pub verifier_seconds: f64,
    pub verified_step_seconds: f64,
    pub economic_steps: u64,
    pub economic_expected_seconds: f64,
    pub economic_credit_seconds: f64,
    pub economic_disables: u64,
    /// Kept as an explicit audited zero rather than inferred from logs.
    pub target_reprefill_requests: u64,
}

struct ProposalTransaction<S: Clone> {
    proposal: DraftProposal,
    base: StateSnapshot<S>,
}

struct ActiveRequest<S: Clone> {
    lease: DraftLease,
    origin: Option<StateSnapshot<S>>,
    proposals_enabled: bool,
    state_aligned: bool,
    retain_state_after_disable: bool,
    qualified: bool,
    width: u8,
    reason: Option<String>,
    disable_category: Option<DisableCategory>,
    ledger: Vec<u32>,
    prompt_boundary: Option<Boundary<S>>,
    transaction: Option<ProposalTransaction<S>>,
    baseline_seconds_per_token: Option<f64>,
    last_verifier_seconds: Option<f64>,
    economic_credit_seconds: f64,
    economic_loss_streak: u8,
    economic_steps: u32,
    last_resolved: Option<(u64, usize, usize)>,
}

struct StagedBoundary<S: Clone> {
    token: BoundaryStageToken,
    boundary: Boundary<S>,
}

/// One-request-at-a-time exact-token DSpark state/admission controller.
pub struct DSparkRuntime<B: DraftBackend> {
    mode: Mode,
    config: RuntimeConfig,
    backend: Option<B>,
    next_id: u64,
    generation: u64,
    active: Option<ActiveRequest<B::Snapshot>>,
    published: Option<Boundary<B::Snapshot>>,
    staged: Option<StagedBoundary<B::Snapshot>>,
    metrics: Metrics,
}

impl<B: DraftBackend> DSparkRuntime<B> {
    pub fn new(mode: Mode, backend: Option<B>, config: RuntimeConfig) -> Result<Self> {
        Ok(Self {
            mode,
            config: config.validate()?,
            backend,
            next_id: 1,
            generation: 0,
            active: None,
            published: None,
            staged: None,
            metrics: Metrics::default(),
        })
    }

    pub const fn mode(&self) -> Mode {
        self.mode
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn backend(&self) -> Option<&B> {
        self.backend.as_ref()
    }

    pub fn current_ledger(&self, lease: &DraftLease) -> Result<&[u32]> {
        Ok(&self.active_for(lease)?.ledger)
    }

    pub fn published_token_ids(&self) -> Option<&[u32]> {
        self.published
            .as_ref()
            .map(|boundary| boundary.token_ids.as_ref())
    }

    pub fn published_target_boundary_id(&self) -> Option<&BoundaryId> {
        self.published
            .as_ref()
            .map(|boundary| &boundary.target_boundary_id)
    }

    fn next_identifier(&mut self) -> Result<u64> {
        let value = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| RuntimeError::state("DSpark identifier space exhausted"))?;
        Ok(value)
    }

    fn validate_tokens(&self, values: &[u32], label: &str) -> Result<Box<[u32]>> {
        if values.is_empty() {
            return Err(RuntimeError::protocol(format!("{label} must not be empty")));
        }
        for (index, &token) in values.iter().enumerate() {
            if token >= self.config.vocab_size {
                return Err(RuntimeError::protocol(format!(
                    "{label} token {index} is outside vocabulary: {token}"
                )));
            }
        }
        Ok(values.into())
    }

    fn active_for(&self, lease: &DraftLease) -> Result<&ActiveRequest<B::Snapshot>> {
        match self.active.as_ref() {
            Some(active) if active.lease.lease_id == lease.lease_id => Ok(active),
            _ => Err(RuntimeError::protocol("DSpark lease is not active")),
        }
    }

    fn active_for_mut(&mut self, lease: &DraftLease) -> Result<&mut ActiveRequest<B::Snapshot>> {
        match self.active.as_mut() {
            Some(active) if active.lease.lease_id == lease.lease_id => Ok(active),
            _ => Err(RuntimeError::protocol("DSpark lease is not active")),
        }
    }

    fn current_signature(&mut self) -> Result<StateSignature> {
        let backend = self
            .backend
            .as_mut()
            .ok_or_else(|| RuntimeError::state("no DSpark backend is installed"))?;
        let token_count = backend.state_token_count().map_err(|error| {
            RuntimeError::backend(format!("DSpark state token count is unreadable: {error}"))
        })?;
        let model = backend.model_identity().map_err(|error| {
            RuntimeError::backend(format!("DSpark model identity is unreadable: {error}"))
        })?;
        Ok(StateSignature {
            format: STATE_FORMAT,
            model,
            token_count,
        })
    }

    fn capture(&mut self) -> Result<StateSnapshot<B::Snapshot>> {
        let before = self.current_signature()?;
        let opaque = self
            .backend
            .as_mut()
            .ok_or_else(|| RuntimeError::state("no DSpark backend is installed"))?
            .snapshot_state()
            .map_err(|error| {
                RuntimeError::backend(format!("DSpark state snapshot failed: {error}"))
            })?;
        let after = self.current_signature()?;
        if after != before {
            return Err(RuntimeError::state("DSpark snapshot mutated live state"));
        }
        Ok(StateSnapshot {
            signature: before,
            opaque,
        })
    }

    fn restore(&mut self, snapshot: &StateSnapshot<B::Snapshot>) -> Result<()> {
        self.backend
            .as_mut()
            .ok_or_else(|| RuntimeError::state("no DSpark backend is installed"))?
            .restore_state(&snapshot.opaque)
            .map_err(|error| {
                RuntimeError::backend(format!("DSpark state restore failed: {error}"))
            })?;
        let restored = self.current_signature()?;
        if restored != snapshot.signature {
            return Err(RuntimeError::state(
                "restored DSpark state signature does not match snapshot",
            ));
        }
        Ok(())
    }

    fn reset_and_capture(&mut self) -> Result<StateSnapshot<B::Snapshot>> {
        self.backend
            .as_mut()
            .ok_or_else(|| RuntimeError::state("no DSpark backend is installed"))?
            .reset_state()
            .map_err(|error| {
                RuntimeError::backend(format!("DSpark state reset failed: {error}"))
            })?;
        let signature = self.current_signature()?;
        if signature.token_count != 0 {
            return Err(RuntimeError::state(
                "reset DSpark state is not at token boundary zero",
            ));
        }
        self.capture()
    }

    fn disable_active(
        &mut self,
        reason: impl Into<String>,
        state_aligned: Option<bool>,
        category: Option<DisableCategory>,
    ) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        if active.proposals_enabled {
            self.metrics.runtime_disables += 1;
        }
        active.proposals_enabled = false;
        if let Some(aligned) = state_aligned {
            active.state_aligned = aligned;
        }
        active.reason = Some(reason.into());
        active.disable_category = category;
    }

    fn release_obsolete_publication(&mut self, reset_backend: bool) {
        if self.published.take().is_some() {
            self.metrics.obsolete_boundaries_discarded += 1;
        }
        if !reset_backend {
            return;
        }
        let Some(backend) = self.backend.as_mut() else {
            return;
        };
        let already_empty = backend.state_token_count().is_ok_and(|count| count == 0);
        if !already_empty && backend.reset_state().is_err() {
            self.metrics.state_failures += 1;
        }
    }

    /// Starts a request while preserving the target cache decision verbatim.
    pub fn begin_request(
        &mut self,
        target_prompt_ids: &[u32],
        target_cache: TargetCache,
        retain_state_after_disable: bool,
    ) -> Result<DraftLease> {
        if self.active.is_some() {
            return Err(RuntimeError::protocol("another DSpark request is active"));
        }
        let prompt = self.validate_tokens(target_prompt_ids, "target prompt")?;
        self.metrics.sessions += 1;
        let target_cache_hit = matches!(target_cache, TargetCache::Hit { .. });
        if target_cache_hit {
            self.metrics.target_cache_hits += 1;
        }
        let lease_id = self.next_identifier()?;
        let mut reason = None;
        let mut draft_cache_hit = false;
        let mut origin = None;
        let mut enabled = self.mode != Mode::Off && self.backend.is_some();
        let mut aligned = enabled;
        let mut cached_tokens = 0;

        if self.mode == Mode::Off {
            reason = Some("DSpark is off".to_owned());
            enabled = false;
            aligned = false;
        } else if self.backend.is_none() {
            reason = Some("no DSpark backend is installed".to_owned());
            enabled = false;
            aligned = false;
        } else if self
            .config
            .max_context_tokens
            .is_some_and(|limit| prompt.len() > limit)
        {
            reason = Some("DSpark prompt exceeds the safe auxiliary-state limit".to_owned());
            enabled = false;
            aligned = false;
            self.metrics.context_limit_disables += 1;
            self.release_obsolete_publication(true);
        } else {
            match target_cache {
                TargetCache::Miss => {
                    self.release_obsolete_publication(false);
                    match self.reset_and_capture() {
                        Ok(snapshot) => origin = Some(snapshot),
                        Err(error) => {
                            reason = Some(error.to_string());
                            enabled = false;
                            aligned = false;
                            self.metrics.state_failures += 1;
                        }
                    }
                }
                TargetCache::Hit {
                    cached_tokens: target_cached_tokens,
                    boundary_id,
                } => {
                    let boundary = self.published.take();
                    let prefix_matches = boundary.as_ref().is_some_and(|boundary| {
                        target_cached_tokens > 0
                            && target_cached_tokens < prompt.len()
                            && boundary.token_ids.len() == target_cached_tokens
                            && prompt[..target_cached_tokens] == *boundary.token_ids
                            && boundary.target_boundary_id == boundary_id
                    });
                    if !prefix_matches {
                        reason = Some(
                            "target cache hit has no matching DSpark boundary; retaining target reuse without drafting"
                                .to_owned(),
                        );
                        enabled = false;
                        aligned = false;
                        self.metrics.missing_draft_state += 1;
                        if boundary.is_some() {
                            self.metrics.obsolete_boundaries_discarded += 1;
                        }
                        self.release_obsolete_publication(true);
                    } else {
                        let boundary = boundary.expect("prefix match requires boundary");
                        let restored = self.current_signature().and_then(|current| {
                            if current.model != boundary.state.signature.model {
                                Err(RuntimeError::state(
                                    "published DSpark model identity changed",
                                ))
                            } else {
                                self.restore(&boundary.state)
                            }
                        });
                        match restored {
                            Ok(()) => {
                                cached_tokens = target_cached_tokens;
                                origin = Some(boundary.state);
                                draft_cache_hit = true;
                                self.metrics.draft_cache_hits += 1;
                            }
                            Err(error) => {
                                reason = Some(error.to_string());
                                enabled = false;
                                aligned = false;
                                self.metrics.missing_draft_state += 1;
                                self.metrics.state_failures += 1;
                                self.release_obsolete_publication(true);
                            }
                        }
                    }
                }
            }
        }

        let lease = DraftLease {
            lease_id,
            enabled,
            target_cache_hit,
            draft_cache_hit,
            cached_tokens,
            reason: reason.clone(),
        };
        self.active = Some(ActiveRequest {
            lease: lease.clone(),
            origin,
            proposals_enabled: enabled,
            state_aligned: aligned,
            retain_state_after_disable,
            qualified: false,
            width: self.config.probe_drafts,
            reason,
            disable_category: None,
            ledger: if draft_cache_hit {
                prompt[..cached_tokens].to_vec()
            } else {
                Vec::new()
            },
            prompt_boundary: None,
            transaction: None,
            baseline_seconds_per_token: None,
            last_verifier_seconds: None,
            economic_credit_seconds: 0.0,
            economic_loss_streak: 0,
            economic_steps: 0,
            last_resolved: None,
        });
        if enabled {
            self.metrics.enabled_sessions += 1;
        } else {
            self.metrics.disabled_sessions += 1;
        }
        debug_assert_eq!(self.metrics.target_reprefill_requests, 0);
        Ok(lease)
    }

    pub fn tracks_target_rows(&self, lease: &DraftLease) -> Result<bool> {
        Ok(self.active_for(lease)?.state_aligned)
    }

    pub fn needs_target_baseline(&self, lease: &DraftLease) -> Result<bool> {
        let active = self.active_for(lease)?;
        Ok(self.mode == Mode::Auto
            && active.proposals_enabled
            && active.state_aligned
            && active.baseline_seconds_per_token.is_none())
    }

    pub fn record_target_baseline(
        &mut self,
        lease: &DraftLease,
        seconds: Duration,
        emitted_tokens: usize,
    ) -> Result<AdmissionDecision> {
        let seconds = positive_seconds(seconds, "target baseline")?;
        if emitted_tokens == 0 {
            return Err(RuntimeError::protocol(
                "target baseline emitted-token count must be positive",
            ));
        }
        let per_token = seconds / emitted_tokens as f64;
        let active = self.active_for_mut(lease)?;
        active.baseline_seconds_per_token = Some(per_token);
        self.metrics.baseline_passes += 1;
        self.metrics.baseline_seconds += seconds;
        self.decision(lease)
    }

    pub fn record_verified_step(
        &mut self,
        lease: &DraftLease,
        proposal: &DraftProposal,
        accepted_drafts: usize,
        emitted_tokens: usize,
        seconds: Duration,
    ) -> Result<AdmissionDecision> {
        let seconds = positive_seconds(seconds, "verified step")?;
        if proposal.lease_id != lease.lease_id {
            return Err(RuntimeError::protocol(
                "verified-step proposal belongs to another request",
            ));
        }
        if accepted_drafts > proposal.token_ids.len() || emitted_tokens == 0 {
            return Err(RuntimeError::protocol("verified-step counts are invalid"));
        }
        {
            let active = self.active_for(lease)?;
            if active.last_resolved != Some((proposal.proposal_id, accepted_drafts, emitted_tokens))
            {
                return Err(RuntimeError::protocol(
                    "verified-step timing does not match the last resolved target step",
                ));
            }
        }
        self.metrics.verified_step_seconds += seconds;
        let auto_mode = self.mode == Mode::Auto;
        let grace_steps = self.config.economic_grace_steps;
        let (baseline, expected, credit, should_evaluate, in_grace) = {
            let active = self.active_for_mut(lease)?;
            active.last_resolved = None;
            let baseline = active.baseline_seconds_per_token;
            let expected = baseline.map(|value| value * emitted_tokens as f64);
            let credit = expected.map(|value| value - seconds);
            if let Some(value) = credit {
                active.economic_credit_seconds += value;
                active.economic_steps = active.economic_steps.saturating_add(1);
            }
            (
                baseline,
                expected,
                credit,
                auto_mode && active.proposals_enabled,
                active.economic_steps <= grace_steps,
            )
        };
        if let (Some(expected), Some(credit)) = (expected, credit) {
            self.metrics.economic_steps += 1;
            self.metrics.economic_expected_seconds += expected;
            self.metrics.economic_credit_seconds += credit;
        }
        if !should_evaluate {
            return self.decision(lease);
        }
        let Some(baseline) = baseline else {
            self.metrics.economic_disables += 1;
            let retain = self.active_for(lease)?.retain_state_after_disable;
            self.disable_active(
                "automatic DSpark has no measured target baseline",
                Some(retain),
                Some(DisableCategory::Economics),
            );
            return self.decision(lease);
        };
        let expected = expected.expect("baseline supplies expected seconds");
        let credit = credit.expect("baseline supplies credit");
        let full_match = accepted_drafts == proposal.token_ids.len();
        let unqualified_probe = {
            let active = self.active_for(lease)?;
            proposal.probe && full_match && !active.qualified
        };
        if unqualified_probe {
            if credit >= expected * self.config.min_auto_speedup {
                let max_width = self.config.max_drafts;
                let active = self.active_for_mut(lease)?;
                active.qualified = true;
                active.width = max_width;
                self.metrics.probe_qualifications += 1;
            } else if in_grace {
                // Stay at the probe width and probe again.
            } else {
                self.metrics.economic_disables += 1;
                let retain = self.active_for(lease)?.retain_state_after_disable;
                self.disable_active(
                    "automatic DSpark probe did not beat its live target baseline",
                    Some(retain),
                    Some(DisableCategory::Economics),
                );
            }
            return self.decision(lease);
        }
        let loss_streak_limit = self.config.loss_streak_limit;
        let disable = {
            let active = self.active_for_mut(lease)?;
            if credit < -baseline {
                active.economic_loss_streak = active.economic_loss_streak.saturating_add(1);
            } else if credit >= 0.0 {
                active.economic_loss_streak = 0;
            }
            !in_grace
                && (active.economic_credit_seconds < 0.0
                    || active.economic_loss_streak >= loss_streak_limit)
        };
        if disable {
            self.metrics.economic_disables += 1;
            let retain = self.active_for(lease)?.retain_state_after_disable;
            self.disable_active(
                "automatic DSpark no longer repays its measured target cost",
                Some(retain),
                Some(DisableCategory::Economics),
            );
        }
        self.decision(lease)
    }

    fn advance(
        &mut self,
        lease: &DraftLease,
        target_context: &B::TargetContext,
        committed_input_ids: &[u32],
    ) -> Result<bool> {
        let committed = self.validate_tokens(committed_input_ids, "committed target inputs")?;
        if !self.active_for(lease)?.state_aligned {
            return Ok(false);
        }
        let ledger_len = self.active_for(lease)?.ledger.len();
        if self.config.max_context_tokens.is_some_and(|limit| {
            ledger_len
                .checked_add(committed.len())
                .is_none_or(|value| value > limit)
        }) {
            self.release_obsolete_publication(false);
            {
                let active = self.active_for_mut(lease)?;
                active.origin = None;
                active.prompt_boundary = None;
                active.ledger.clear();
            }
            if self
                .backend
                .as_mut()
                .is_some_and(|backend| backend.reset_state().is_err())
            {
                self.metrics.state_failures += 1;
            }
            self.metrics.context_limit_disables += 1;
            self.disable_active(
                "DSpark safe context limit reached; full K3 continues without drafting",
                Some(false),
                None,
            );
            return Ok(false);
        }
        let before = match self.capture() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.metrics.state_failures += 1;
                self.disable_active(error.to_string(), Some(false), None);
                return Ok(false);
            }
        };
        if before.signature.token_count != ledger_len {
            self.metrics.state_failures += 1;
            self.disable_active(
                "DSpark model state disagrees with its exact token ledger",
                Some(false),
                None,
            );
            return Ok(false);
        }
        let advanced = self
            .backend
            .as_mut()
            .expect("aligned request always has a backend")
            .advance_target_state(target_context, &committed);
        let after = if advanced.is_ok() {
            self.current_signature()
        } else {
            Err(RuntimeError::backend(format!(
                "DSpark target-row advance failed: {}",
                advanced.expect_err("checked error")
            )))
        };
        let valid = after.and_then(|after| {
            if after.model != before.signature.model {
                Err(RuntimeError::state(
                    "advancing target rows changed DSpark model identity",
                ))
            } else if after.token_count
                != before
                    .signature
                    .token_count
                    .checked_add(committed.len())
                    .ok_or_else(|| RuntimeError::state("DSpark token count overflowed"))?
            {
                Err(RuntimeError::state(
                    "DSpark target-row advance ended at the wrong token count",
                ))
            } else {
                Ok(())
            }
        });
        if let Err(error) = valid {
            let _ = self.restore(&before);
            self.metrics.state_failures += 1;
            self.disable_active(error.to_string(), Some(false), None);
            return Ok(false);
        }
        self.active_for_mut(lease)?
            .ledger
            .extend_from_slice(&committed);
        Ok(true)
    }

    pub fn commit_target_rows(
        &mut self,
        lease: &DraftLease,
        target_context: &B::TargetContext,
        committed_input_ids: &[u32],
    ) -> Result<bool> {
        if self.staged.is_some() {
            return Err(RuntimeError::protocol(
                "cannot append target rows while a boundary is staged",
            ));
        }
        if self.active_for(lease)?.transaction.is_some() {
            return Err(RuntimeError::protocol(
                "cannot append target rows during a draft transaction",
            ));
        }
        self.advance(lease, target_context, committed_input_ids)
    }

    pub fn capture_prompt_boundary(
        &mut self,
        lease: &DraftLease,
        token_ids: &[u32],
    ) -> Result<bool> {
        if self.staged.is_some() {
            return Err(RuntimeError::protocol(
                "cannot capture a prompt boundary while another boundary is staged",
            ));
        }
        let tokens = self.validate_tokens(token_ids, "prompt boundary")?;
        if !self.active_for(lease)?.state_aligned {
            return Ok(false);
        }
        if self.active_for(lease)?.transaction.is_some() {
            return Err(RuntimeError::protocol(
                "cannot capture prompt boundary during a proposal",
            ));
        }
        if self.active_for(lease)?.ledger.as_slice() != tokens.as_ref() {
            self.metrics.state_failures += 1;
            self.disable_active(
                "DSpark prompt boundary does not match its exact token ledger",
                Some(false),
                None,
            );
            return Ok(false);
        }
        let state = match self.capture() {
            Ok(state) if state.signature.token_count == tokens.len() => state,
            Ok(_) => {
                self.metrics.state_failures += 1;
                self.disable_active(
                    "DSpark prompt boundary token count is misaligned",
                    Some(false),
                    None,
                );
                return Ok(false);
            }
            Err(error) => {
                self.metrics.state_failures += 1;
                self.disable_active(error.to_string(), Some(false), None);
                return Ok(false);
            }
        };
        let active = self.active_for_mut(lease)?;
        active.prompt_boundary = Some(Boundary {
            token_ids: tokens,
            state,
            // Not publishable until paired with a target generation.
            target_boundary_id: BoundaryId::unpaired(),
        });
        active.origin = None;
        Ok(true)
    }

    pub fn propose(
        &mut self,
        lease: &DraftLease,
        pending_token_id: u32,
        max_verify_drafts: Option<u8>,
    ) -> Result<Option<DraftProposal>> {
        if self.staged.is_some() {
            return Err(RuntimeError::protocol(
                "cannot propose while a target boundary is staged",
            ));
        }
        if pending_token_id >= self.config.vocab_size {
            return Err(RuntimeError::protocol(
                "pending target token is outside the vocabulary",
            ));
        }
        {
            let active = self.active_for(lease)?;
            if !active.proposals_enabled || !active.state_aligned {
                return Ok(None);
            }
            if active.transaction.is_some() {
                return Err(RuntimeError::protocol(
                    "a DSpark proposal is already pending",
                ));
            }
            if self.mode == Mode::Auto && active.last_resolved.is_some() {
                return Err(RuntimeError::protocol(
                    "record verified-step economics before the next automatic proposal",
                ));
            }
        }
        let width = {
            let active = self.active_for(lease)?;
            match max_verify_drafts {
                Some(0) => {
                    return Err(RuntimeError::protocol("max verify drafts must be positive"));
                }
                Some(limit) => active.width.min(limit),
                None => active.width,
            }
        };
        let signature = match self.current_signature() {
            Ok(signature) => signature,
            Err(error) => {
                self.metrics.state_failures += 1;
                self.disable_active(error.to_string(), Some(false), None);
                return Ok(None);
            }
        };
        if signature.token_count != self.active_for(lease)?.ledger.len() {
            self.metrics.state_failures += 1;
            self.disable_active(
                "DSpark context must end immediately before the pending token",
                Some(false),
                None,
            );
            return Ok(None);
        }
        let base = match self.capture() {
            Ok(base) => base,
            Err(error) => {
                self.metrics.state_failures += 1;
                self.disable_active(error.to_string(), Some(false), None);
                return Ok(None);
            }
        };
        let started = Instant::now();
        let raw = self
            .backend
            .as_mut()
            .expect("enabled request always has a backend")
            .propose(pending_token_id, width);
        let seconds = started.elapsed().as_secs_f64();
        let generated = match raw {
            Ok(raw) => raw.token_ids,
            Err(error) => {
                let restored = self.restore(&base).is_ok();
                self.metrics.proposal_failures += 1;
                let category = error
                    .release_optional_drafter()
                    .then_some(DisableCategory::OptionalBackend);
                let retain = restored && self.active_for(lease)?.retain_state_after_disable;
                self.disable_active(
                    format!("DSpark proposal failed: {error}"),
                    Some(retain),
                    category,
                );
                return Ok(None);
            }
        };
        let proposal_state = self.current_signature();
        if proposal_state.as_ref() != Ok(&base.signature) {
            let restored = self.restore(&base).is_ok();
            self.metrics.proposal_failures += 1;
            self.metrics.state_failures += 1;
            let retain = restored && self.active_for(lease)?.retain_state_after_disable;
            self.disable_active(
                "DSpark proposal mutated committed state",
                Some(retain),
                None,
            );
            return Ok(None);
        }
        if generated.is_empty()
            || generated.len() > self.config.max_drafts as usize
            || generated.len() < width as usize
            || generated
                .iter()
                .any(|&token| token >= self.config.vocab_size)
        {
            let restored = self.restore(&base).is_ok();
            self.metrics.proposal_failures += 1;
            let retain = restored && self.active_for(lease)?.retain_state_after_disable;
            self.disable_active(
                "DSpark proposal violated its token or width contract",
                Some(retain),
                None,
            );
            return Ok(None);
        }
        let proposal_id = self.next_identifier()?;
        let probe = !self.active_for(lease)?.qualified;
        let submitted: Box<[u32]> = generated[..width as usize].into();
        let proposal = DraftProposal {
            lease_id: lease.lease_id,
            proposal_id,
            token_ids: submitted,
            generated_token_ids: generated,
            probe,
            base_state: base.signature.clone(),
            pending_token_id,
            seconds,
        };
        self.active_for_mut(lease)?.transaction = Some(ProposalTransaction {
            proposal: proposal.clone(),
            base,
        });
        self.metrics.proposals += 1;
        self.metrics.generated_drafts += proposal.generated_token_ids.len() as u64;
        self.metrics.submitted_drafts += proposal.token_ids.len() as u64;
        Ok(Some(proposal))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        &mut self,
        lease: &DraftLease,
        proposal: &DraftProposal,
        accepted_drafts: usize,
        emitted_token_ids: &[u32],
        target_context: &B::TargetContext,
        verifier_seconds: Duration,
    ) -> Result<AdmissionDecision> {
        let transaction_matches =
            self.active_for(lease)?
                .transaction
                .as_ref()
                .is_some_and(|transaction| {
                    transaction.proposal.proposal_id == proposal.proposal_id
                        && proposal.lease_id == lease.lease_id
                        && transaction.proposal.base_state == proposal.base_state
                });
        if !transaction_matches {
            return Err(RuntimeError::protocol(
                "proposal does not own the active DSpark transaction",
            ));
        }
        if accepted_drafts > proposal.token_ids.len() {
            return Err(RuntimeError::protocol("accepted draft count is invalid"));
        }
        let emitted = self.validate_tokens(emitted_token_ids, "target-emitted tokens")?;
        if accepted_drafts > emitted.len() {
            return Err(RuntimeError::protocol(
                "accepted drafts exceed emitted target tokens",
            ));
        }
        if emitted.len() > accepted_drafts + 1 {
            return Err(RuntimeError::protocol(
                "target emitted more than the accepted draft prefix plus one authoritative token",
            ));
        }
        if emitted[..accepted_drafts] != proposal.token_ids[..accepted_drafts] {
            return Err(RuntimeError::protocol(
                "accepted target tokens do not match the proposal prefix",
            ));
        }
        if accepted_drafts < proposal.token_ids.len()
            && emitted.len() == accepted_drafts + 1
            && emitted[accepted_drafts] == proposal.token_ids[accepted_drafts]
        {
            return Err(RuntimeError::protocol(
                "accepted draft count is not the longest target-matching prefix",
            ));
        }
        let verifier_seconds = nonnegative_seconds(verifier_seconds, "verifier")?;
        let transaction = self
            .active_for_mut(lease)?
            .transaction
            .take()
            .expect("validated transaction exists");
        if let Err(error) = self.restore(&transaction.base) {
            self.metrics.state_failures += 1;
            self.disable_active(error.to_string(), Some(false), None);
            return self.decision(lease);
        }
        let mut committed_inputs = Vec::with_capacity(emitted.len());
        committed_inputs.push(proposal.pending_token_id);
        committed_inputs.extend_from_slice(&emitted[..emitted.len() - 1]);
        if !self.advance(lease, target_context, &committed_inputs)? {
            return self.decision(lease);
        }
        self.metrics.accepted_drafts += accepted_drafts as u64;
        self.metrics.emitted_tokens += emitted.len() as u64;
        self.metrics.verifier_seconds += verifier_seconds;
        {
            let active = self.active_for_mut(lease)?;
            active.last_verifier_seconds = Some(verifier_seconds);
            active.last_resolved = Some((proposal.proposal_id, accepted_drafts, emitted.len()));
        }
        if accepted_drafts == proposal.token_ids.len() {
            self.metrics.full_matches += 1;
            if self.mode != Mode::Auto
                && !self.active_for(lease)?.qualified
                && proposal.token_ids.len() >= self.config.probe_drafts as usize
            {
                self.active_for_mut(lease)?.qualified = true;
                self.metrics.probe_qualifications += 1;
            }
            let qualified = self.active_for(lease)?.qualified;
            let width = if qualified {
                self.config.max_drafts
            } else {
                self.config.probe_drafts
            };
            self.active_for_mut(lease)?.width = width;
        } else if accepted_drafts == 0 {
            self.metrics.misses += 1;
            if self.config.disable_on_miss {
                let retain = self.active_for(lease)?.retain_state_after_disable;
                self.disable_active(
                    "DSpark target-verification miss",
                    Some(retain),
                    Some(DisableCategory::AcceptanceEconomics),
                );
            } else {
                let probe = self.config.probe_drafts;
                self.active_for_mut(lease)?.width = probe;
            }
        } else {
            self.metrics.partial_matches += 1;
            let doubled = accepted_drafts.saturating_mul(2);
            let width = usize::from(self.config.probe_drafts)
                .max(usize::from(self.config.max_drafts).min(doubled));
            self.active_for_mut(lease)?.width = width as u8;
        }
        self.decision(lease)
    }

    pub fn abort_proposal(
        &mut self,
        lease: &DraftLease,
        proposal: Option<&DraftProposal>,
    ) -> Result<bool> {
        let Some(transaction) = self.active_for(lease)?.transaction.as_ref() else {
            return Ok(false);
        };
        if proposal.is_some_and(|proposal| proposal.proposal_id != transaction.proposal.proposal_id)
        {
            return Err(RuntimeError::protocol(
                "proposal does not own the active DSpark transaction",
            ));
        }
        let transaction = self
            .active_for_mut(lease)?
            .transaction
            .take()
            .expect("transaction was just checked");
        if let Err(error) = self.restore(&transaction.base) {
            self.metrics.state_failures += 1;
            self.disable_active(error.to_string(), Some(false), None);
        }
        self.metrics.proposal_aborts += 1;
        Ok(true)
    }

    /// Stops optional proposals while optionally preserving a separately
    /// proven target-row alignment for final-boundary reuse.
    pub fn disable_proposals(
        &mut self,
        lease: &DraftLease,
        reason: impl Into<String>,
        state_aligned: Option<bool>,
    ) -> Result<AdmissionDecision> {
        if self.active_for(lease)?.transaction.is_some() {
            return Err(RuntimeError::protocol(
                "abort the active proposal before disabling proposals",
            ));
        }
        self.disable_active(reason, state_aligned, None);
        self.decision(lease)
    }

    /// Whether the optional backend may be evicted after a terminal provider
    /// failure or an automatic acceptance/economic loss.  This is only a
    /// policy signal; the controller never owns or releases target residency.
    pub fn eviction_recommended(&self, lease: &DraftLease) -> Result<bool> {
        let active = self.active_for(lease)?;
        Ok(!active.proposals_enabled
            && active.transaction.is_none()
            && self.backend.is_some()
            && (active.disable_category == Some(DisableCategory::OptionalBackend)
                || (self.mode == Mode::Auto
                    && matches!(
                        active.disable_category,
                        Some(
                            DisableCategory::Economics
                                | DisableCategory::AcceptanceEconomics
                                | DisableCategory::OptionalBackend
                        )
                    ))))
    }

    pub fn stage_final_boundary(
        &mut self,
        lease: &DraftLease,
        token_ids: &[u32],
        target_boundary_id: BoundaryId,
    ) -> Result<Option<BoundaryStageToken>> {
        if self.staged.is_some() {
            return Err(RuntimeError::protocol(
                "a DSpark final boundary is already staged",
            ));
        }
        let tokens = self.validate_tokens(token_ids, "final boundary")?;
        let active = self.active_for(lease)?;
        if active.transaction.is_some() {
            return Err(RuntimeError::protocol(
                "cannot stage boundary during a draft transaction",
            ));
        }
        if !active.state_aligned {
            return Ok(None);
        }
        if active.ledger.as_slice() != tokens.as_ref() {
            self.metrics.state_failures += 1;
            self.disable_active(
                "DSpark final boundary does not match its exact token ledger",
                Some(false),
                None,
            );
            return Ok(None);
        }
        let state = match self.capture() {
            Ok(state) if state.signature.token_count == tokens.len() => state,
            Ok(_) => {
                self.metrics.state_failures += 1;
                self.disable_active(
                    "DSpark final boundary token count is misaligned",
                    Some(false),
                    None,
                );
                return Ok(None);
            }
            Err(error) => {
                self.metrics.state_failures += 1;
                self.disable_active(error.to_string(), Some(false), None);
                return Ok(None);
            }
        };
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| RuntimeError::state("DSpark boundary generation exhausted"))?;
        let token = BoundaryStageToken {
            lease_id: lease.lease_id,
            generation: self.generation,
        };
        self.staged = Some(StagedBoundary {
            token,
            boundary: Boundary {
                token_ids: tokens,
                state,
                target_boundary_id,
            },
        });
        self.metrics.boundaries_staged += 1;
        Ok(Some(token))
    }

    pub fn stage_prompt_fallback(
        &mut self,
        lease: &DraftLease,
        target_boundary_id: BoundaryId,
    ) -> Result<Option<BoundaryStageToken>> {
        if self.staged.is_some() {
            return Err(RuntimeError::protocol(
                "a DSpark final boundary is already staged",
            ));
        }
        if self.active_for(lease)?.transaction.is_some() {
            return Err(RuntimeError::protocol(
                "cannot stage boundary during a draft transaction",
            ));
        }
        let Some(prompt) = self.active_for(lease)?.prompt_boundary.clone() else {
            return Ok(None);
        };
        if let Err(error) = self.restore(&prompt.state) {
            self.metrics.state_failures += 1;
            self.disable_active(error.to_string(), Some(false), None);
            return Ok(None);
        }
        self.active_for_mut(lease)?.ledger = prompt.token_ids.to_vec();
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| RuntimeError::state("DSpark boundary generation exhausted"))?;
        let token = BoundaryStageToken {
            lease_id: lease.lease_id,
            generation: self.generation,
        };
        self.staged = Some(StagedBoundary {
            token,
            boundary: Boundary {
                token_ids: prompt.token_ids,
                state: prompt.state,
                target_boundary_id,
            },
        });
        self.metrics.boundaries_staged += 1;
        Ok(Some(token))
    }

    pub fn commit_staged(&mut self, token: BoundaryStageToken) -> Result<()> {
        let staged = self
            .staged
            .take()
            .ok_or_else(|| RuntimeError::protocol("no DSpark final boundary is staged"))?;
        if staged.token != token {
            self.staged = Some(staged);
            return Err(RuntimeError::protocol("stale DSpark boundary stage"));
        }
        if self
            .active
            .as_ref()
            .is_none_or(|active| active.lease.lease_id != token.lease_id)
        {
            self.staged = Some(staged);
            return Err(RuntimeError::protocol(
                "staged DSpark boundary has no active request",
            ));
        }
        self.published = Some(staged.boundary);
        self.active = None;
        self.metrics.boundaries_published += 1;
        Ok(())
    }

    pub fn abort_staged(&mut self, token: Option<BoundaryStageToken>) -> Result<bool> {
        let Some(staged) = self.staged.as_ref() else {
            return Ok(false);
        };
        if token.is_some_and(|token| token != staged.token) {
            return Err(RuntimeError::protocol("stale DSpark boundary stage"));
        }
        self.staged = None;
        self.metrics.boundary_aborts += 1;
        Ok(true)
    }

    pub fn finish_request(&mut self, lease: &DraftLease) -> Result<()> {
        if self.active_for(lease)?.transaction.is_some() {
            self.abort_proposal(lease, None)?;
        }
        self.abort_staged(None)?;
        self.active = None;
        Ok(())
    }

    pub fn abort_request(&mut self, lease: &DraftLease) -> Result<()> {
        if self.active_for(lease)?.transaction.is_some() {
            self.abort_proposal(lease, None)?;
        }
        self.abort_staged(None)?;
        let fallback = {
            let active = self.active_for(lease)?;
            active
                .prompt_boundary
                .as_ref()
                .map(|boundary| boundary.state.clone())
                .or_else(|| active.origin.clone())
        };
        if self.active_for(lease)?.state_aligned
            && fallback
                .as_ref()
                .is_some_and(|state| self.restore(state).is_err())
        {
            self.metrics.state_failures += 1;
        }
        self.active = None;
        self.metrics.request_aborts += 1;
        Ok(())
    }

    fn decision(&self, lease: &DraftLease) -> Result<AdmissionDecision> {
        let active = self.active_for(lease)?;
        Ok(AdmissionDecision {
            proposals_enabled: active.proposals_enabled,
            state_aligned: active.state_aligned,
            qualified: active.qualified,
            next_width: active.width,
            reason: active.reason.clone(),
            disable_category: active.disable_category,
        })
    }
}

fn positive_seconds(duration: Duration, label: &str) -> Result<f64> {
    let seconds = duration.as_secs_f64();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(RuntimeError::protocol(format!(
            "{label} seconds must be finite and positive"
        )));
    }
    Ok(seconds)
}

fn nonnegative_seconds(duration: Duration, label: &str) -> Result<f64> {
    let seconds = duration.as_secs_f64();
    if !seconds.is_finite() {
        return Err(RuntimeError::protocol(format!(
            "{label} seconds must be finite and non-negative"
        )));
    }
    Ok(seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct MockSnapshot {
        count: usize,
        query_epoch: u64,
    }

    #[derive(Debug)]
    struct MockBackend {
        identity: ModelIdentity,
        count: usize,
        query_epoch: u64,
        proposal: Vec<u32>,
        advances: Vec<usize>,
        advanced_ids: Vec<u32>,
        reset_calls: usize,
        snapshot_mutates: bool,
        restore_wrong_count: bool,
        advance_wrong_count: bool,
        proposal_mutates_committed_count: bool,
        proposal_failure: Option<BackendFailure>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                identity: identity(1),
                count: 0,
                query_epoch: 0,
                proposal: vec![10, 11, 12, 13, 14, 15, 16],
                advances: Vec::new(),
                advanced_ids: Vec::new(),
                reset_calls: 0,
                snapshot_mutates: false,
                restore_wrong_count: false,
                advance_wrong_count: false,
                proposal_mutates_committed_count: false,
                proposal_failure: None,
            }
        }
    }

    impl DraftBackend for MockBackend {
        type Snapshot = MockSnapshot;
        type TargetContext = [u32];

        fn reset_state(&mut self) -> std::result::Result<(), BackendFailure> {
            self.count = 0;
            self.query_epoch = 0;
            self.reset_calls += 1;
            Ok(())
        }

        fn snapshot_state(&mut self) -> std::result::Result<Self::Snapshot, BackendFailure> {
            let snapshot = MockSnapshot {
                count: self.count,
                query_epoch: self.query_epoch,
            };
            if self.snapshot_mutates {
                self.count += 1;
            }
            Ok(snapshot)
        }

        fn restore_state(
            &mut self,
            snapshot: &Self::Snapshot,
        ) -> std::result::Result<(), BackendFailure> {
            self.count = snapshot.count + usize::from(self.restore_wrong_count);
            self.query_epoch = snapshot.query_epoch;
            Ok(())
        }

        fn state_token_count(&mut self) -> std::result::Result<usize, BackendFailure> {
            Ok(self.count)
        }

        fn model_identity(&mut self) -> std::result::Result<ModelIdentity, BackendFailure> {
            Ok(self.identity.clone())
        }

        fn propose(
            &mut self,
            _pending_token_id: u32,
            _max_drafts: u8,
        ) -> std::result::Result<BackendProposal, BackendFailure> {
            if let Some(error) = self.proposal_failure.take() {
                return Err(error);
            }
            self.query_epoch += 1;
            if self.proposal_mutates_committed_count {
                self.count += 1;
            }
            Ok(BackendProposal::new(self.proposal.clone()))
        }

        fn advance_target_state(
            &mut self,
            target_context: &Self::TargetContext,
            committed_input_ids: &[u32],
        ) -> std::result::Result<(), BackendFailure> {
            if target_context.len() != committed_input_ids.len() {
                return Err(BackendFailure::new("target context row mismatch"));
            }
            self.advances.push(committed_input_ids.len());
            self.advanced_ids.extend_from_slice(committed_input_ids);
            self.count += committed_input_ids.len() + usize::from(self.advance_wrong_count);
            Ok(())
        }
    }

    fn identity(seed: u8) -> ModelIdentity {
        ModelIdentity::new(
            "deltafin-dspark",
            [seed; 32],
            "trained-k3",
            "runtime-k3",
            "tokenizer-sha",
            "5x576xbf16-cow-v1",
            "bf16-dense",
            "mps:0",
        )
        .unwrap()
    }

    fn runtime(mode: Mode) -> DSparkRuntime<MockBackend> {
        DSparkRuntime::new(mode, Some(MockBackend::new()), RuntimeConfig::default()).unwrap()
    }

    fn begin_and_prefill(runtime: &mut DSparkRuntime<MockBackend>, prompt: &[u32]) -> DraftLease {
        let lease = runtime
            .begin_request(prompt, TargetCache::Miss, true)
            .unwrap();
        assert!(lease.enabled());
        assert!(runtime.commit_target_rows(&lease, prompt, prompt).unwrap());
        lease
    }

    #[test]
    fn configuration_and_identity_are_fail_closed() {
        let bad = RuntimeConfig {
            probe_drafts: 0,
            ..RuntimeConfig::default()
        };
        let error = match DSparkRuntime::<MockBackend>::new(Mode::Auto, None, bad) {
            Ok(_) => panic!("invalid configuration was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::Configuration);
        assert!(ModelIdentity::new("", [0; 32], "a", "b", "c", "d", "e", "f").is_err());
    }

    #[test]
    fn exact_boundary_pair_resumes_without_target_reprefill() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1, 2, 3]);
        let stage = runtime
            .stage_final_boundary(&lease, &[1, 2, 3], BoundaryId::numeric(44))
            .unwrap()
            .unwrap();
        runtime.commit_staged(stage).unwrap();
        assert_eq!(runtime.published_token_ids(), Some(&[1, 2, 3][..]));

        let lease = runtime
            .begin_request(
                &[1, 2, 3, 4],
                TargetCache::Hit {
                    cached_tokens: 3,
                    boundary_id: BoundaryId::numeric(44),
                },
                true,
            )
            .unwrap();
        assert!(lease.target_cache_hit());
        assert!(lease.draft_cache_hit());
        assert_eq!(lease.cached_tokens(), 3);
        assert!(!lease.requires_target_reprefill());
        assert_eq!(runtime.current_ledger(&lease).unwrap(), &[1, 2, 3]);
        assert_eq!(runtime.metrics().target_reprefill_requests, 0);
    }

    #[test]
    fn missing_or_wrong_pair_preserves_target_cache_hit_and_disables_only_draft() {
        let mut runtime = runtime(Mode::On);
        let lease = runtime
            .begin_request(
                &[1, 2],
                TargetCache::Hit {
                    cached_tokens: 1,
                    boundary_id: BoundaryId::numeric(9),
                },
                true,
            )
            .unwrap();
        assert!(lease.target_cache_hit());
        assert!(!lease.draft_cache_hit());
        assert!(!lease.enabled());
        assert!(!lease.requires_target_reprefill());
        assert_eq!(runtime.metrics().target_reprefill_requests, 0);

        runtime.finish_request(&lease).unwrap();
        let fresh = begin_and_prefill(&mut runtime, &[5, 6]);
        let stage = runtime
            .stage_final_boundary(&fresh, &[5, 6], BoundaryId::numeric(10))
            .unwrap()
            .unwrap();
        runtime.commit_staged(stage).unwrap();
        let wrong = runtime
            .begin_request(
                &[5, 6, 7],
                TargetCache::Hit {
                    cached_tokens: 2,
                    boundary_id: BoundaryId::numeric(11),
                },
                true,
            )
            .unwrap();
        assert!(wrong.target_cache_hit());
        assert!(!wrong.enabled());
        assert_eq!(runtime.metrics().target_reprefill_requests, 0);
    }

    #[test]
    fn changed_model_identity_rejects_published_cache_pair() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1, 2]);
        let stage = runtime
            .stage_final_boundary(&lease, &[1, 2], BoundaryId::numeric(3))
            .unwrap()
            .unwrap();
        runtime.commit_staged(stage).unwrap();
        runtime.backend.as_mut().unwrap().identity = identity(2);
        let lease = runtime
            .begin_request(
                &[1, 2, 3],
                TargetCache::Hit {
                    cached_tokens: 2,
                    boundary_id: BoundaryId::numeric(3),
                },
                true,
            )
            .unwrap();
        assert!(!lease.enabled());
        assert!(lease.target_cache_hit());
        assert!(!lease.requires_target_reprefill());
    }

    #[test]
    fn explicit_mode_uses_two_then_seven_and_commits_only_target_rows() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1, 2]);
        let proposal = runtime.propose(&lease, 3, None).unwrap().unwrap();
        assert_eq!(proposal.token_ids(), &[10, 11]);
        assert!(proposal.is_probe());
        assert_eq!(runtime.backend().unwrap().query_epoch, 1);

        let decision = runtime
            .resolve(
                &lease,
                &proposal,
                2,
                &[10, 11, 99],
                &[100, 101, 102],
                Duration::from_millis(10),
            )
            .unwrap();
        assert!(decision.qualified());
        assert_eq!(decision.next_width(), 7);
        assert_eq!(runtime.current_ledger(&lease).unwrap(), &[1, 2, 3, 10, 11]);
        assert_eq!(runtime.backend().unwrap().count, 5);
        assert_eq!(runtime.backend().unwrap().query_epoch, 0);
        // The authoritative bonus token remains pending and is not cached.
        assert!(!runtime.current_ledger(&lease).unwrap().contains(&99));
        let next = runtime.propose(&lease, 99, None).unwrap().unwrap();
        assert_eq!(next.token_ids().len(), 7);
    }

    #[test]
    fn auto_probe_qualifies_only_after_live_economic_win() {
        let mut runtime = runtime(Mode::Auto);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        assert!(runtime.needs_target_baseline(&lease).unwrap());
        runtime
            .record_target_baseline(&lease, Duration::from_secs(10), 1)
            .unwrap();
        let proposal = runtime.propose(&lease, 2, None).unwrap().unwrap();
        runtime
            .resolve(
                &lease,
                &proposal,
                2,
                &[10, 11, 12],
                &[1, 2, 3],
                Duration::from_secs(8),
            )
            .unwrap();
        let decision = runtime
            .record_verified_step(&lease, &proposal, 2, 3, Duration::from_secs(20))
            .unwrap();
        assert!(decision.qualified());
        assert_eq!(decision.next_width(), 7);
        assert_eq!(runtime.metrics().probe_qualifications, 1);
    }

    #[test]
    fn auto_probe_that_misses_speed_floor_disables_but_keeps_alignment() {
        let mut runtime = runtime(Mode::Auto);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        runtime
            .record_target_baseline(&lease, Duration::from_secs(10), 1)
            .unwrap();
        let proposal = runtime.propose(&lease, 2, None).unwrap().unwrap();
        runtime
            .resolve(
                &lease,
                &proposal,
                2,
                &[10, 11, 12],
                &[1, 2, 3],
                Duration::ZERO,
            )
            .unwrap();
        let decision = runtime
            .record_verified_step(&lease, &proposal, 2, 3, Duration::from_secs(30))
            .unwrap();
        assert!(!decision.proposals_enabled());
        assert!(decision.state_aligned());
        assert_eq!(
            decision.disable_category(),
            Some(DisableCategory::Economics)
        );
        assert!(runtime.tracks_target_rows(&lease).unwrap());
    }

    #[test]
    fn cumulative_auto_loss_disables_after_qualification() {
        let mut runtime = runtime(Mode::Auto);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        runtime
            .record_target_baseline(&lease, Duration::from_secs(10), 1)
            .unwrap();
        let probe = runtime.propose(&lease, 2, None).unwrap().unwrap();
        runtime
            .resolve(&lease, &probe, 2, &[10, 11, 12], &[1, 2, 3], Duration::ZERO)
            .unwrap();
        runtime
            .record_verified_step(&lease, &probe, 2, 3, Duration::from_secs(20))
            .unwrap();
        let wide = runtime.propose(&lease, 12, None).unwrap().unwrap();
        runtime
            .resolve(
                &lease,
                &wide,
                7,
                &[10, 11, 12, 13, 14, 15, 16, 17],
                &[0; 8],
                Duration::ZERO,
            )
            .unwrap();
        let decision = runtime
            .record_verified_step(&lease, &wide, 7, 8, Duration::from_secs(100))
            .unwrap();
        assert!(!decision.proposals_enabled());
        assert_eq!(runtime.metrics().economic_disables, 1);
    }

    fn chain_runtime(mode: Mode) -> DSparkRuntime<MockBackend> {
        DSparkRuntime::new(
            mode,
            Some(MockBackend::new()),
            RuntimeConfig {
                probe_drafts: 3,
                disable_on_miss: false,
                economic_grace_steps: 4,
                loss_streak_limit: 4,
                ..RuntimeConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn chain_drafter_miss_narrows_to_probe_and_advances_committed_ids() {
        let mut runtime = chain_runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1, 2]);
        let probe = runtime.propose(&lease, 3, None).unwrap().unwrap();
        assert_eq!(probe.token_ids(), &[10, 11, 12]);
        runtime
            .resolve(&lease, &probe, 3, &[10, 11, 12, 13], &[0; 4], Duration::ZERO)
            .unwrap();
        // The old pending token and the accepted drafts are the inputs the
        // committed rows consumed; the bonus token stays pending.
        assert_eq!(
            runtime.backend().unwrap().advanced_ids,
            vec![1, 2, 3, 10, 11, 12]
        );
        let wide = runtime.propose(&lease, 13, None).unwrap().unwrap();
        assert_eq!(wide.token_ids().len(), 7);
        let decision = runtime
            .resolve(&lease, &wide, 0, &[77], &[0], Duration::ZERO)
            .unwrap();
        assert!(decision.proposals_enabled());
        assert!(decision.state_aligned());
        assert_eq!(decision.next_width(), 3);
        assert_eq!(runtime.metrics().misses, 1);
        assert_eq!(
            runtime.backend().unwrap().advanced_ids,
            vec![1, 2, 3, 10, 11, 12, 13]
        );
        let next = runtime.propose(&lease, 77, None).unwrap().unwrap();
        assert_eq!(next.token_ids().len(), 3);
    }

    #[test]
    fn auto_grace_rides_out_early_losses_then_judges_cumulative_credit() {
        let mut runtime = chain_runtime(Mode::Auto);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        runtime
            .record_target_baseline(&lease, Duration::from_secs(10), 1)
            .unwrap();
        // Step 1: the probe misses outright, a clear loss inside the grace.
        let probe = runtime.propose(&lease, 2, None).unwrap().unwrap();
        runtime
            .resolve(&lease, &probe, 0, &[50], &[0], Duration::ZERO)
            .unwrap();
        let decision = runtime
            .record_verified_step(&lease, &probe, 0, 1, Duration::from_secs(25))
            .unwrap();
        assert!(decision.proposals_enabled());
        assert!(!decision.qualified());
        // Step 2: a full probe that beats the baseline qualifies (+20 s).
        let probe = runtime.propose(&lease, 50, None).unwrap().unwrap();
        assert_eq!(probe.token_ids().len(), 3);
        runtime
            .resolve(&lease, &probe, 3, &[10, 11, 12, 60], &[0; 4], Duration::ZERO)
            .unwrap();
        let decision = runtime
            .record_verified_step(&lease, &probe, 3, 4, Duration::from_secs(20))
            .unwrap();
        assert!(decision.qualified());
        assert_eq!(decision.next_width(), 7);
        // Steps 3-5 miss at a 20 s loss each. Cumulative credit goes negative
        // at step 3, but only step 5, past the four-step grace, may act on it.
        let mut pending = 60;
        for step in 3..=5_u32 {
            let proposal = runtime.propose(&lease, pending, None).unwrap().unwrap();
            pending = 70 + step;
            runtime
                .resolve(&lease, &proposal, 0, &[pending], &[0], Duration::ZERO)
                .unwrap();
            let decision = runtime
                .record_verified_step(&lease, &proposal, 0, 1, Duration::from_secs(30))
                .unwrap();
            assert_eq!(decision.proposals_enabled(), step < 5, "step {step}");
        }
        assert_eq!(runtime.metrics().economic_disables, 1);
    }

    #[test]
    fn target_prefix_mismatch_is_protocol_error_and_abort_restores_base() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        let proposal = runtime.propose(&lease, 2, None).unwrap().unwrap();
        let error = runtime
            .resolve(&lease, &proposal, 1, &[999], &[0], Duration::ZERO)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Protocol);
        assert!(runtime.abort_proposal(&lease, Some(&proposal)).unwrap());
        assert_eq!(runtime.backend().unwrap().query_epoch, 0);
        assert_eq!(runtime.current_ledger(&lease).unwrap(), &[1]);
    }

    #[test]
    fn verifier_must_report_the_longest_matching_prefix() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        let proposal = runtime.propose(&lease, 2, None).unwrap().unwrap();
        let error = runtime
            .resolve(&lease, &proposal, 1, &[10, 11], &[0, 0], Duration::ZERO)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Protocol);
        assert!(error.message().contains("longest"));
        assert!(runtime.abort_proposal(&lease, Some(&proposal)).unwrap());
    }

    #[test]
    fn proposal_committed_state_mutation_is_detected_and_rolled_back() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        runtime
            .backend
            .as_mut()
            .unwrap()
            .proposal_mutates_committed_count = true;
        assert!(runtime.propose(&lease, 2, None).unwrap().is_none());
        assert_eq!(runtime.backend().unwrap().count, 1);
        assert!(runtime.tracks_target_rows(&lease).unwrap());
        assert_eq!(runtime.metrics().proposal_failures, 1);
    }

    #[test]
    fn snapshot_and_advance_mismatches_disable_state_alignment() {
        let mut broken_runtime = runtime(Mode::On);
        broken_runtime.backend.as_mut().unwrap().snapshot_mutates = true;
        let lease = broken_runtime
            .begin_request(&[1], TargetCache::Miss, true)
            .unwrap();
        assert!(!lease.enabled());
        assert_eq!(broken_runtime.metrics().state_failures, 1);
        broken_runtime.finish_request(&lease).unwrap();

        let mut runtime = runtime(Mode::On);
        let lease = runtime
            .begin_request(&[1], TargetCache::Miss, true)
            .unwrap();
        runtime.backend.as_mut().unwrap().advance_wrong_count = true;
        assert!(!runtime.commit_target_rows(&lease, &[0], &[1]).unwrap());
        assert!(!runtime.tracks_target_rows(&lease).unwrap());
        assert_eq!(runtime.backend().unwrap().count, 0);
    }

    #[test]
    fn proposal_backend_failure_restores_and_disables_safely() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        runtime.backend.as_mut().unwrap().proposal_failure =
            Some(BackendFailure::releasable("provider lost device"));
        assert!(runtime.propose(&lease, 2, None).unwrap().is_none());
        let decision = runtime.decision(&lease).unwrap();
        assert!(!decision.proposals_enabled());
        assert!(decision.state_aligned());
        assert_eq!(
            decision.disable_category(),
            Some(DisableCategory::OptionalBackend)
        );
        assert_eq!(runtime.backend().unwrap().count, 1);
        assert!(runtime.eviction_recommended(&lease).unwrap());
    }

    #[test]
    fn partial_match_reduces_width_and_zero_match_disables_candidates_only() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        let probe = runtime.propose(&lease, 2, None).unwrap().unwrap();
        runtime
            .resolve(&lease, &probe, 2, &[10, 11, 12], &[0, 0, 0], Duration::ZERO)
            .unwrap();
        let wide = runtime.propose(&lease, 12, None).unwrap().unwrap();
        let decision = runtime
            .resolve(&lease, &wide, 2, &[10, 11, 90], &[0, 0, 0], Duration::ZERO)
            .unwrap();
        assert_eq!(decision.next_width(), 4);
        assert!(decision.proposals_enabled());
        let narrower = runtime.propose(&lease, 90, None).unwrap().unwrap();
        assert_eq!(narrower.token_ids().len(), 4);
        let decision = runtime
            .resolve(&lease, &narrower, 0, &[91], &[0], Duration::ZERO)
            .unwrap();
        assert!(!decision.proposals_enabled());
        assert!(decision.state_aligned());
        assert_eq!(
            decision.disable_category(),
            Some(DisableCategory::AcceptanceEconomics)
        );
        assert!(runtime.tracks_target_rows(&lease).unwrap());
    }

    #[test]
    fn auto_without_baseline_fails_closed_after_verified_probe() {
        let mut runtime = runtime(Mode::Auto);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        let proposal = runtime.propose(&lease, 2, None).unwrap().unwrap();
        runtime
            .resolve(
                &lease,
                &proposal,
                2,
                &[10, 11, 12],
                &[0, 0, 0],
                Duration::ZERO,
            )
            .unwrap();
        let decision = runtime
            .record_verified_step(&lease, &proposal, 2, 3, Duration::from_secs(1))
            .unwrap();
        assert!(!decision.proposals_enabled());
        assert!(decision.state_aligned());
        assert_eq!(
            decision.disable_category(),
            Some(DisableCategory::Economics)
        );
        assert!(runtime.eviction_recommended(&lease).unwrap());
    }

    #[test]
    fn failed_restore_on_abort_invalidates_draft_alignment_only() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        let proposal = runtime.propose(&lease, 2, None).unwrap().unwrap();
        runtime.backend.as_mut().unwrap().restore_wrong_count = true;
        assert!(runtime.abort_proposal(&lease, Some(&proposal)).unwrap());
        assert!(!runtime.tracks_target_rows(&lease).unwrap());
        assert!(!runtime.decision(&lease).unwrap().proposals_enabled());
        assert_eq!(runtime.metrics().target_reprefill_requests, 0);
    }

    #[test]
    fn ledger_mismatch_cannot_be_published_or_used_as_prompt_fallback() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1, 2]);
        assert!(!runtime.capture_prompt_boundary(&lease, &[1, 3]).unwrap());
        assert!(!runtime.tracks_target_rows(&lease).unwrap());
        assert!(
            runtime
                .stage_final_boundary(&lease, &[1, 2], BoundaryId::numeric(2))
                .unwrap()
                .is_none()
        );
        assert_eq!(runtime.metrics().target_reprefill_requests, 0);
    }

    #[test]
    fn context_limit_drops_only_optional_state() {
        let config = RuntimeConfig {
            max_context_tokens: Some(3),
            ..RuntimeConfig::default()
        };
        let mut runtime = DSparkRuntime::new(Mode::On, Some(MockBackend::new()), config).unwrap();
        let lease = begin_and_prefill(&mut runtime, &[1, 2, 3]);
        assert!(!runtime.commit_target_rows(&lease, &[0], &[4]).unwrap());
        assert!(!runtime.tracks_target_rows(&lease).unwrap());
        assert_eq!(runtime.backend().unwrap().count, 0);
        assert_eq!(runtime.metrics().context_limit_disables, 1);
        assert_eq!(runtime.metrics().target_reprefill_requests, 0);
    }

    #[test]
    fn prompt_fallback_and_abort_restore_exact_request_boundaries() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1, 2]);
        assert!(runtime.capture_prompt_boundary(&lease, &[1, 2]).unwrap());
        runtime.commit_target_rows(&lease, &[0], &[3]).unwrap();
        let stage = runtime
            .stage_prompt_fallback(&lease, BoundaryId::numeric(70))
            .unwrap()
            .unwrap();
        assert_eq!(runtime.current_ledger(&lease).unwrap(), &[1, 2]);
        runtime.abort_staged(Some(stage)).unwrap();
        runtime.abort_request(&lease).unwrap();
        assert_eq!(runtime.backend().unwrap().count, 2);
    }

    #[test]
    fn stale_leases_proposals_and_stages_cannot_cross_transactions() {
        let mut runtime = runtime(Mode::On);
        let lease = begin_and_prefill(&mut runtime, &[1]);
        assert!(
            runtime
                .begin_request(&[2], TargetCache::Miss, true)
                .is_err()
        );
        let proposal = runtime.propose(&lease, 2, None).unwrap().unwrap();
        let mut forged = proposal.clone();
        forged.proposal_id += 1;
        assert!(runtime.abort_proposal(&lease, Some(&forged)).is_err());
        runtime.abort_proposal(&lease, Some(&proposal)).unwrap();
        let stage = runtime
            .stage_final_boundary(&lease, &[1], BoundaryId::numeric(8))
            .unwrap()
            .unwrap();
        assert!(runtime.commit_target_rows(&lease, &[0], &[2]).is_err());
        assert!(runtime.propose(&lease, 2, None).is_err());
        let stale = BoundaryStageToken {
            generation: stage.generation + 1,
            ..stage
        };
        assert!(runtime.commit_staged(stale).is_err());
        runtime.commit_staged(stage).unwrap();
    }

    #[test]
    fn off_or_missing_backend_is_a_target_only_fallback() {
        let mut off = DSparkRuntime::new(
            Mode::Off,
            Some(MockBackend::new()),
            RuntimeConfig::default(),
        )
        .unwrap();
        let lease = off.begin_request(&[1], TargetCache::Miss, true).unwrap();
        assert!(!lease.enabled());
        assert!(!lease.requires_target_reprefill());
        assert!(!off.tracks_target_rows(&lease).unwrap());

        let mut missing =
            DSparkRuntime::<MockBackend>::new(Mode::Auto, None, RuntimeConfig::default()).unwrap();
        let lease = missing
            .begin_request(&[1], TargetCache::Miss, true)
            .unwrap();
        assert!(!lease.enabled());
        assert!(lease.reason().unwrap().contains("no DSpark backend"));
        assert_eq!(missing.metrics().target_reprefill_requests, 0);
    }
}
