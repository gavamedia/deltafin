//! One-process native target bootstrap and ownership boundary.
//!
//! Construction performs every cold discovery step exactly once, validates the
//! complete K3 contract before publishing the engine, and retains the bounded
//! resources needed by later requests.  It deliberately does not embed or
//! spawn CPython. The target path is one exact Rust-owned transaction loop;
//! dropping an unpublished provider sequence cancels every staged cache.

use std::ffi::OsStr;
use std::fmt::{self, Display, Formatter};
use std::io::{self, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};

use crate::chat::{ChatOptions, encode_chat};
use crate::config::{
    DSparkRequest, Eagle3Request, ExpertBackendRequest, ExpertScale4Request, QwenRequest,
    RuntimeConfig, RuntimeSurface, SpineRequest,
};
use crate::decode::{DecodeArena, StopReason};
use crate::draft::{DraftSource, NgramDraftSource};
use crate::dspark_checkpoint::{DSparkCheckpoint, DSparkConfig, OFFICIAL_PARAMETER_COUNT};
use crate::draft_backend::NativeDraftBackend;
use crate::dspark_provider::NativeDSparkBackend;
use crate::dspark_runtime::{
    BoundaryId, BoundaryStageToken, DSparkRuntime, DraftLease,
    RuntimeConfig as DSparkRuntimeConfig, TargetCache,
};
use crate::eagle3_provider::{self, Eagle3Checkpoint, NativeEagle3Backend};
use crate::embedding::{
    BF16_BYTES, EmbeddingArena, EmbeddingSpec, EmbeddingTable, k3_embedding_path,
};
use crate::error::{DeltafinError, Result};
use crate::experts::{
    ExpertStorageLayout, ExpertUnionReadBatch, ExpertUnionReadTicket, K3_EXPERT_BASE_UNION_MAX,
    K3_EXPERT_SOURCE_BYTES, K3_EXPERT_TOP_K, K3_EXPERT_UNION_MAX, RawExpertCorpus,
};
use crate::inventory::PINNED_INVENTORY_SHA256;
use crate::model::ModelSpec;
use crate::openai::{
    AuthoritativeTarget, ClientPresence, FinishReason, StreamGenerationError, StreamPublication,
    TargetDelta, TargetDeltaSink, TargetOutput, TargetPrompt, TargetRequest, TargetStreamSummary,
    TokenUsage,
};
use crate::output::IncrementalUtf8Decoder;
use crate::expert_heat::{
    EXPERT_HEAT_HOST_RESERVE_BYTES, ExpertHeat, ExpertPinTier, PageAlignedSpan,
    resolve_auto_pin_budget,
};
use crate::pilot_gate::{ExpertPrefetchPlan, PilotGate, PilotGateReport};
use crate::platform::{
    Device, DeviceRequest, DeviceSelection, DeviceSelectionPolicy, ProviderInventory,
    apple_cpu_family,
};
use crate::program::{
    GlobalSpinePlan, LayerSpinePlan, PackedSpineCatalog, SourceLayout, SpineRepresentation,
    TargetProgram, WeightDType, WeightStorage,
};
use crate::provider::{
    CUDA_EXPERT_CACHE_MAX_EXPERTS, CudaExpertCachePolicy, NativeProvider, NativeProviderInventory,
    NativeProviderMemorySnapshot, NativeProviderSession, ProviderTensor,
    TARGET_PILOT_RESERVE_BYTES, TargetExpertBackend, TargetGlobalGroup, TargetSequence,
    TargetSequenceCommit, TargetSequenceLayerPrepare, TargetSequenceMailbox, TargetSequenceMode,
    TargetSequenceStats, TargetStateBoundary, TargetStateBranch,
};
use crate::quality::ResidentWeightAuthority;
use crate::qwen_checkpoint::{QwenCheckpoint, QwenVariant};
use crate::qwen_draft::{
    AdaptiveQwenProbeSelection, FailSoftQwenDraft, QwenDraftController, QwenDraftProposal,
    QwenTokenizer, select_adaptive_qwen_probe, select_hybrid_qwen_proposal,
};
use crate::qwen_provider::NativeQwen;
use crate::residency::{
    FixedCosts, HostMemory, ProviderMemory, ResidencyOverride, ResidencyPolicy, ResidencySelection,
    ResidencyStop, probe_host_memory, select_resident_prefix,
};
use crate::router_trace::{ROUTER_TRACE_HOST_RESERVE_BYTES, RouterTrace, RouterTraceMode};
use crate::run_events::RunEventLog;
use crate::run_interrupt::InterruptSource;
use crate::spine_runtime::SpinePipeline;
use crate::storage_homes::StorageHomes;
use crate::storage::{
    BufferLengths, BufferRetireHook, CachePolicy, LOOSE_SPINE_DESCRIPTOR_RESERVE, Reader,
    prepare_persistent_descriptor_capacity,
};
use crate::tokenizer::K3Tokenizer;

const SPINE_READER_LIMIT: usize = 4;
const QUALIFIED_SPINE_READER_LIMIT: usize = 6;
const QUALIFIED_PHYSICAL_BYTES: u64 = 64 * (1_u64 << 30);
const QUALIFIED_RECOMMENDED_BYTES: u64 = 55_662_788_608;
const QUALIFIED_MAX_BUFFER_BYTES: u64 = 41_747_087_360;
const QUALIFIED_RESOURCE_SLOP_BYTES: u64 = 1_u64 << 30;
const AUTO_SPINE_HOST_RESERVE_MIN_BYTES: u64 = 20 * (1_u64 << 30);
const AUTO_SPINE_DEVICE_RESERVE_BYTES: u64 = 12 * (1_u64 << 30);
const AUTO_SPINE_RESIDENT_PERCENT: u64 = 3;
const AUTO_SPINE_RESIDENT_MAX_BYTES: u64 = 2_100_000_000;
const SPINE_ARENA_SLOTS: usize = 2;
const EXPERT_READER_LIMIT: usize = 4;
// Authoritative expert misses are read and consumed synchronously. One
// reusable union slot prevents a second multi-gigabyte demand slab from
// becoming resident; scheduling-only one-expert tickets use the separate
// arena below.
const EXPERT_ARENA_SLOTS: usize = 1;
const FULL_COMMIT_EXPERT_UNION_MAX: usize = (MAX_EXACT_DRAFTS + 1) * K3_EXPERT_TOP_K;
// Two explicit generations of up to thirty-two one-expert speculative leases
// plus one deliberately empty arena slot.  Generation N may still be borrowed
// by the current Metal tile while generation N+1 is submitted before that
// synchronous kernel. `ReadPriority::Prefetch` may never consume a Reader's
// final slot, so the extra slot preserves the Reader's demand-first invariant.
// The complete 64-live-slot cost is charged by `native_fixed_costs` below.
// How many freshly arrived experts an early drain waits for before handing a
// wave to the provider. Each offer costs one provider call and one GPU
// submission, so a per-expert offer spends more on submission than it saves in
// overlap. Sized to the demand reader's worker count: experts land in
// roughly that many at a time.
const EXPERT_EARLY_DRAIN_WAVE: usize = EXPERT_READER_LIMIT;
const EXPERT_PREFETCH_MAX_EXPERTS: usize = 2 * K3_EXPERT_TOP_K;
const EXPERT_PREFETCH_GENERATIONS: usize = 2;
const EXPERT_PREFETCH_LIVE_SLOTS: usize = EXPERT_PREFETCH_GENERATIONS * EXPERT_PREFETCH_MAX_EXPERTS;
const EXPERT_PREFETCH_ARENA_SLOTS: usize = EXPERT_PREFETCH_LIVE_SLOTS + 1;
/// The provider route ABI currently admits at most 64 positions.  Keeping the
/// exact embedding reader at the same bound prevents a second, larger staging
/// allocation from appearing when layer-major prefill is connected.
const EMBEDDING_ARENA_ROWS: usize = 64;
const K3_KDA_CONV_ELEMENTS_PER_LAYER: u64 = 147_456;
const K3_KDA_RECURRENT_ELEMENTS_PER_LAYER: u64 = 1_572_864;
const K3_KDA_PROJECTION: u64 = 12_288;
const K3_KDA_HEADS: u64 = 96;
const K3_KDA_CONVOLUTION_WIDTH: u64 = 4;
const K3_MLA_INITIAL_CAPACITY: u64 = 16;
const K3_MLA_STORAGE_BUDGET_PER_LAYER_BYTES: u64 = 512 * 1024 * 1024;
// Do not let retained spine layers consume every byte that the live cache
// admission gate will need immediately afterward. Reserve the complete staged
// generation and allocator scratch required to reach an ordinary 256-token
// conversation. Longer histories still use fresh live admission and may trade
// future residency for memory pressure in a later provider implementation.
const K3_MLA_STARTUP_RESERVE_TOKENS: u64 = 256;
const TARGET_SEQUENCE_MAX_POSITIONS: u64 = 64;
const K3_LAYER_DERIVED_RESIDUAL_BYTES: u64 = 2 * 7_168 * 4;
const K3_TAIL_DERIVED_RESIDUAL_BYTES: u64 = 7_168 * 4;
const TARGET_EXPERT_TILE_MAX_ROWS: usize = 16;
const K3_EXPERT_COUNT: u16 = 896;
// Shared with K3_METAL_MOE_EMBEDDED_SOURCE_V1 in tools/metal_moe_abi.h. This is
// an ABI selector, not a repository-relative path: production binaries carry
// the reviewed precompiled metallib and never need a loose .metal file beside
// the model or invoke Metal's source compiler while serving.
const K3_METAL_EMBEDDED_SOURCE_V1: &str = "deltafin:embedded-metal-moe-mxfp4:v1";
// Retain the former override name only as a fail-closed tripwire. Product
// binaries never admit loose MSL, including debug builds; the native xtask
// owns the isolated source-compiler test variant.
const K3_METAL_DEVELOPMENT_SOURCE_ENV: &str = "K3_METAL_SRC";
const MAX_NATIVE_CPU_EXPERT_THREADS: usize = 32;
const MAX_EXACT_DRAFTS: usize = 8;
const SERVER_TARGET_REUSE_MIN_TOKENS: usize = 117;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NativeGeneration {
    /// Every full-K3-authoritative token, including EOS when K3 emitted it.
    pub token_ids: Box<[u32]>,
    pub stop: StopReason,
    /// True when at least one non-EOS token reached the output sink.
    pub wrote_text: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum NativeTargetReadiness {
    /// Structural/runtime gates pass, but this exact device, resident-spine,
    /// and expert backend combination has no recorded physical real-weight
    /// sequence-parity evidence yet.
    StructurallyReadyUnverified,
    /// The original-BF16 spine plus raw-v1 Metal expert path completed exact
    /// 17-token sequence parity against the independent frozen target on a
    /// physical Apple MPS device, traversing the complete K3 target.
    VerifiedOriginalBf16MpsRawMetal,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum NativeEngineLifecycle {
    Ready,
    Running,
    Poisoned,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
enum NativeStreamBoundary {
    #[default]
    None,
    AwaitingPublication,
    PublicationFailed,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum NativeStreamResolution {
    None,
    Preserve,
    Discard,
    Poison,
}

/// Provider-backed operations required before cross-request target KV reuse
/// can be enabled honestly.
///
/// Token-prefix metadata is not a capability. `RequestBranchV1` means the
/// provider can atomically branch its complete committed KDA/MLA session,
/// route every subsequent target-sequence commit into that private branch,
/// and either publish the branch or restore the parent byte-for-byte. The
/// expected position passed at branch creation protects the engine/provider
/// one-token-lag boundary from stale metadata.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
enum TargetStateTransactionCapability {
    ResetOnly,
    RequestBranchV1,
}

trait TargetStateCapabilityProvider {
    fn target_state_transaction_capability(&self) -> TargetStateTransactionCapability;
}

/// Future provider implementations must implement this complete transaction;
/// exposing only a cache length, prefix hash, or reset operation is
/// insufficient and must remain `ResetOnly`.
#[allow(dead_code)]
trait TransactionalTargetStateProvider: TargetStateCapabilityProvider {
    type RequestBranch;

    /// Begin one exclusive child of the currently published target state.
    /// Existing target-sequence APIs must address the child until resolution.
    fn begin_target_state_branch(
        &mut self,
        expected_committed_positions: u64,
    ) -> Result<Self::RequestBranch>;

    /// Atomically replace the published parent after the whole response is
    /// externally complete.
    fn publish_target_state_branch(&mut self, branch: Self::RequestBranch) -> Result<()>;

    /// Discard every child mutation and restore the exact parent state.
    fn discard_target_state_branch(&mut self, branch: Self::RequestBranch) -> Result<()>;
}

impl TargetStateCapabilityProvider for NativeProviderSession {
    fn target_state_transaction_capability(&self) -> TargetStateTransactionCapability {
        TargetStateTransactionCapability::RequestBranchV1
    }
}

impl TransactionalTargetStateProvider for NativeProviderSession {
    type RequestBranch = TargetStateBranch;

    fn begin_target_state_branch(
        &mut self,
        expected_committed_positions: u64,
    ) -> Result<Self::RequestBranch> {
        let boundary = self.inspect_target_state()?;
        if boundary.committed_positions != expected_committed_positions {
            return Err(DeltafinError::new(
                "provider target boundary differs from the exact prefix plan",
            ));
        }
        NativeProviderSession::begin_target_state_branch(self, boundary)
    }

    fn publish_target_state_branch(&mut self, branch: Self::RequestBranch) -> Result<()> {
        NativeProviderSession::publish_target_state_branch(self, branch).map(|_| ())
    }

    fn discard_target_state_branch(&mut self, branch: Self::RequestBranch) -> Result<()> {
        NativeProviderSession::discard_target_state_branch(self, branch).map(|_| ())
    }
}

impl NativeStreamBoundary {
    fn resolve(&mut self, publication: StreamPublication) -> NativeStreamResolution {
        let boundary = std::mem::take(self);
        match (boundary, publication) {
            (Self::None, _) => NativeStreamResolution::None,
            (Self::AwaitingPublication, StreamPublication::Complete) => {
                NativeStreamResolution::Preserve
            }
            (Self::AwaitingPublication | Self::PublicationFailed, StreamPublication::Aborted) => {
                NativeStreamResolution::Discard
            }
            (Self::PublicationFailed, StreamPublication::Complete) => {
                NativeStreamResolution::Poison
            }
        }
    }
}

impl NativeEngineLifecycle {
    fn begin(&mut self) -> Result<()> {
        match self {
            Self::Ready => {
                *self = Self::Running;
                Ok(())
            }
            Self::Running => Err(DeltafinError::new(
                "native target engine is already executing a request",
            )),
            Self::Poisoned => Err(DeltafinError::new(
                "native target engine is poisoned after a failed request; bootstrap a fresh engine",
            )),
        }
    }

    fn publish(&mut self) -> Result<()> {
        if *self != Self::Running {
            *self = Self::Poisoned;
            return Err(DeltafinError::new(
                "native target engine can publish a request only from its running state",
            ));
        }
        *self = Self::Ready;
        Ok(())
    }

    fn poison(&mut self) {
        *self = Self::Poisoned;
    }
}

impl Display for NativeTargetReadiness {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::StructurallyReadyUnverified => formatter.write_str(
                "native target structure ready (all 16 routed experts; physical real-weight sequence parity not yet recorded for this backend/representation)",
            ),
            Self::VerifiedOriginalBf16MpsRawMetal => formatter.write_str(
                "native target verified (physical MPS, original-BF16 spine, raw-v1 Metal experts, all 16 routed experts; exact 17-token full-target parity)",
            ),
        }
    }
}

impl NativeTargetReadiness {
    const fn permits_generation(self) -> bool {
        matches!(
            self,
            Self::StructurallyReadyUnverified | Self::VerifiedOriginalBf16MpsRawMetal
        )
    }
}

fn target_readiness(
    device: Device,
    representation: SpineRepresentation,
    expert_backend: ResolvedExpertBackend,
    expert_storage: ExpertStorageLayout,
) -> NativeTargetReadiness {
    if device == Device::Mps
        && representation == SpineRepresentation::OriginalBf16
        && expert_backend == ResolvedExpertBackend::Metal
        && expert_storage == ExpertStorageLayout::RawV1
    {
        NativeTargetReadiness::VerifiedOriginalBf16MpsRawMetal
    } else {
        NativeTargetReadiness::StructurallyReadyUnverified
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpinePlanSource {
    AuthenticatedPacks,
    LooseDeferredFiles,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ResolvedExpertBackend {
    Cpu,
    Metal,
    /// Runtime CUDA KAT may fail soft to the exact compiled CPU expert path.
    CudaAuto,
    /// Explicit CUDA request; any runtime qualification failure is fatal.
    Cuda,
}

impl Display for ResolvedExpertBackend {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Metal => formatter.write_str("metal"),
            Self::CudaAuto => formatter.write_str("cuda(auto)"),
            Self::Cuda => formatter.write_str("cuda"),
        }
    }
}

impl Display for SpinePlanSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthenticatedPacks => formatter.write_str("authenticated-packs"),
            Self::LooseDeferredFiles => formatter.write_str("loose-deferred"),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QwenRuntimeState {
    Off,
    /// No request this run can serve is routed to Qwen (see
    /// [`qwen_usable_by_run`]), so it is neither loaded nor reserved.
    UnusedByRun,
    IneligibleDevice,
    NotInstalled,
    MemoryRejected,
    Probe06B,
    Wide17B,
    FailedSoft,
}

impl Display for QwenRuntimeState {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Off => "off",
            Self::UnusedByRun => "unused-by-run",
            Self::IneligibleDevice => "ineligible-device",
            Self::NotInstalled => "not-installed",
            Self::MemoryRejected => "memory-rejected",
            Self::Probe06B => "qwen3-0.6b",
            Self::Wide17B => "qwen3-1.7b",
            Self::FailedSoft => "failed-soft",
        })
    }
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct QwenTelemetry {
    pub proposals: u64,
    pub proposed_tokens: u64,
    pub accepted_tokens: u64,
    pub fallbacks: u64,
    pub verify_memory_rejections: u64,
    pub wide_loads: u64,
    pub wide_invocations: u64,
    pub wide_skips: u64,
    pub wide_failures: u64,
    pub raw_override_selections: u64,
    pub raw_override_failures: u64,
}

#[derive(Debug)]
enum CompiledSpine {
    Packed(PackedSpineCatalog),
    Loose(Box<[LayerSpinePlan]>),
}

impl CompiledSpine {
    fn source(&self) -> SpinePlanSource {
        match self {
            Self::Packed(_) => SpinePlanSource::AuthenticatedPacks,
            Self::Loose(_) => SpinePlanSource::LooseDeferredFiles,
        }
    }

    fn layers(&self) -> &[LayerSpinePlan] {
        match self {
            Self::Packed(catalog) => catalog.layers(),
            Self::Loose(layers) => layers,
        }
    }

    fn persistent_source_count(&self) -> usize {
        self.layers()
            .iter()
            .map(|layer| layer.read_plan().persistent_source_count())
            .sum()
    }

    fn opened_persistent_source_count(&self) -> usize {
        self.layers()
            .iter()
            .map(|layer| layer.read_plan().opened_persistent_source_count())
            .sum()
    }
}

type NativeQwenDraft = FailSoftQwenDraft<NativeQwen, K3Tokenizer, QwenTokenizer>;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct QwenPlan {
    state: QwenRuntimeState,
    initial: Option<QwenVariant>,
    wide_lazy: bool,
    reserved_provider_bytes: u64,
    reserved_verify_bytes: u64,
    reserved_verify_positions: u64,
    context_capacity: usize,
}

impl QwenPlan {
    const fn inactive(state: QwenRuntimeState) -> Self {
        Self {
            state,
            initial: None,
            wide_lazy: false,
            reserved_provider_bytes: 0,
            reserved_verify_bytes: 0,
            reserved_verify_positions: 0,
            context_capacity: 0,
        }
    }
}

struct QwenRuntime {
    state: QwenRuntimeState,
    source: Option<NativeQwenDraft>,
    wide_source: Option<NativeQwenDraft>,
    target: Arc<K3Tokenizer>,
    model_root: PathBuf,
    wide_lazy: bool,
    reserved_provider_bytes: u64,
    reserved_verify_bytes: u64,
    reserved_verify_positions: u64,
    context_capacity: usize,
    telemetry: QwenTelemetry,
    last_error: Option<String>,
    raw_override_allowed: bool,
    last_raw_override: bool,
    last_submitted_drafts: usize,
}

impl QwenRuntime {
    fn from_plan(
        plan: QwenPlan,
        provider: &NativeProviderSession,
        model_root: &Path,
        target: Arc<K3Tokenizer>,
    ) -> Self {
        let mut runtime = Self {
            state: plan.state,
            source: None,
            wide_source: None,
            target,
            model_root: model_root.to_path_buf(),
            wide_lazy: plan.wide_lazy,
            reserved_provider_bytes: plan.reserved_provider_bytes,
            reserved_verify_bytes: plan.reserved_verify_bytes,
            reserved_verify_positions: plan.reserved_verify_positions,
            context_capacity: plan.context_capacity,
            telemetry: QwenTelemetry::default(),
            last_error: None,
            raw_override_allowed: true,
            last_raw_override: false,
            last_submitted_drafts: 0,
        };
        let Some(variant) = plan.initial else {
            return runtime;
        };
        match load_qwen_source(
            provider,
            model_root,
            Arc::clone(&runtime.target),
            variant,
            runtime.context_capacity,
        ) {
            Ok(source) => {
                runtime.state = qwen_variant_state(variant);
                runtime.source = Some(source);
            }
            Err(error) => runtime.fail_soft(error),
        }
        runtime
    }

    fn fail_soft(&mut self, error: DeltafinError) {
        self.source = None;
        self.wide_source = None;
        self.state = QwenRuntimeState::FailedSoft;
        self.last_error = Some(error.to_string());
        self.last_raw_override = false;
        self.last_submitted_drafts = 0;
    }

    fn begin_request(&mut self) {
        self.raw_override_allowed = true;
        self.last_raw_override = false;
        self.last_submitted_drafts = 0;
    }

    fn propose(
        &mut self,
        provider: &NativeProviderSession,
        target_history: &[u32],
        maximum: usize,
    ) -> QwenDraftProposal {
        self.last_raw_override = false;
        self.last_submitted_drafts = 0;
        if maximum == 0 {
            return QwenDraftProposal::default();
        }
        let Some(source) = self.source.as_mut() else {
            self.telemetry.fallbacks = self.telemetry.fallbacks.saturating_add(1);
            return QwenDraftProposal::default();
        };
        self.telemetry.proposals = self.telemetry.proposals.saturating_add(1);
        let probe = source
            .propose_with_outcome(target_history, maximum)
            .unwrap_or_default();
        let probe_error = (!source.is_enabled()).then(|| {
            source
                .last_error()
                .unwrap_or("native Qwen proposal source disabled")
                .to_owned()
        });
        if let Some(detail) = probe_error {
            self.fail_soft(DeltafinError::new(detail));
            self.telemetry.fallbacks = self.telemetry.fallbacks.saturating_add(1);
            return QwenDraftProposal::default();
        }
        let selected = if maximum <= 2 || !self.wide_lazy {
            probe
        } else {
            match select_adaptive_qwen_probe(probe, maximum, self.raw_override_allowed) {
                AdaptiveQwenProbeSelection::Selected {
                    proposal,
                    raw_override,
                } => {
                    self.telemetry.wide_skips = self.telemetry.wide_skips.saturating_add(1);
                    if raw_override {
                        self.telemetry.raw_override_selections =
                            self.telemetry.raw_override_selections.saturating_add(1);
                        self.last_raw_override = true;
                    }
                    proposal
                }
                AdaptiveQwenProbeSelection::NeedsWide(probe) => {
                    if self.wide_source.is_none() {
                        match load_qwen_source(
                            provider,
                            &self.model_root,
                            Arc::clone(&self.target),
                            QwenVariant::Wide17B,
                            self.context_capacity,
                        ) {
                            Ok(source) => {
                                self.wide_source = Some(source);
                                self.telemetry.wide_loads =
                                    self.telemetry.wide_loads.saturating_add(1);
                            }
                            Err(error) => {
                                self.wide_lazy = false;
                                self.telemetry.wide_failures =
                                    self.telemetry.wide_failures.saturating_add(1);
                                self.last_error = Some(error.to_string());
                            }
                        }
                    }
                    if let Some(wide) = self.wide_source.as_mut() {
                        self.telemetry.wide_invocations =
                            self.telemetry.wide_invocations.saturating_add(1);
                        let proposal = wide
                            .propose_with_outcome(target_history, maximum)
                            .unwrap_or_default();
                        let wide_error = (!wide.is_enabled()).then(|| {
                            wide.last_error()
                                .unwrap_or("native wide Qwen proposal source disabled")
                                .to_owned()
                        });
                        if let Some(detail) = wide_error {
                            self.telemetry.wide_failures =
                                self.telemetry.wide_failures.saturating_add(1);
                            self.last_error = Some(detail);
                            self.wide_source = None;
                            self.wide_lazy = false;
                            probe
                        } else {
                            select_hybrid_qwen_proposal(probe, proposal, maximum)
                        }
                    } else {
                        probe
                    }
                }
            }
        };
        self.last_submitted_drafts = selected.token_ids().len();
        self.telemetry.proposed_tokens = self
            .telemetry
            .proposed_tokens
            .saturating_add(selected.token_ids().len() as u64);
        if selected.token_ids().is_empty() {
            self.telemetry.fallbacks = self.telemetry.fallbacks.saturating_add(1);
        }
        selected
    }

    fn record_verified(&mut self, accepted: usize) {
        self.telemetry.accepted_tokens = self
            .telemetry
            .accepted_tokens
            .saturating_add(accepted as u64);
        if self.last_raw_override && accepted < self.last_submitted_drafts {
            self.raw_override_allowed = false;
            self.telemetry.raw_override_failures =
                self.telemetry.raw_override_failures.saturating_add(1);
        }
        self.last_raw_override = false;
        self.last_submitted_drafts = 0;
    }

    fn is_available(&self) -> bool {
        self.source.is_some()
    }
}

/// Request-local economic control for raw-completion Qwen proposals.
///
/// The probe begins at two tokens so the 0.6B assistant can establish that a
/// wide verifier is useful before the 1.7B checkpoint is loaded. A full probe
/// acceptance widens to the admitted maximum; misses shrink or disable only
/// this request. None of these states can commit a token or mutate target
/// history—the exact K3 verifier remains the sole authority.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct QwenRequestPolicy {
    active: bool,
    qualified: bool,
    consecutive_misses: usize,
    probe_width: usize,
    maximum_width: usize,
    current_width: usize,
}

impl QwenRequestPolicy {
    fn new(available: bool, maximum_width: usize) -> Self {
        let maximum_width = maximum_width.min(MAX_EXACT_DRAFTS);
        let probe_width = maximum_width.min(2);
        Self {
            active: available && probe_width != 0,
            qualified: false,
            consecutive_misses: 0,
            probe_width,
            maximum_width,
            current_width: probe_width,
        }
    }

    fn proposal_width(self, available_budget: usize) -> usize {
        if self.active {
            self.current_width.min(available_budget)
        } else {
            0
        }
    }

    fn record_empty(&mut self, confidence_stopped: bool) {
        if !confidence_stopped {
            self.active = false;
        }
    }

    fn record_verified(&mut self, accepted: usize, proposed: usize) {
        if !self.active || proposed == 0 {
            return;
        }
        if accepted > proposed {
            self.active = false;
            return;
        }
        if accepted == 0 {
            self.consecutive_misses = self.consecutive_misses.saturating_add(1);
        } else {
            self.consecutive_misses = 0;
        }
        if !self.qualified {
            if accepted == proposed && accepted >= 2 {
                self.qualified = true;
                self.current_width = self.maximum_width;
            } else if accepted == 0 {
                self.active = false;
            } else {
                self.current_width = self.probe_width;
            }
            return;
        }
        if accepted == proposed {
            self.current_width = self.maximum_width;
        } else if accepted == 0 {
            if self.consecutive_misses >= 2 {
                self.active = false;
            } else {
                self.current_width = self.probe_width;
            }
        } else {
            self.current_width = self
                .probe_width
                .max(self.maximum_width.min(accepted.saturating_mul(2)));
        }
    }
}

fn qwen_variant_state(variant: QwenVariant) -> QwenRuntimeState {
    match variant {
        QwenVariant::Probe06B => QwenRuntimeState::Probe06B,
        QwenVariant::Wide17B => QwenRuntimeState::Wide17B,
    }
}

fn load_qwen_source(
    provider: &NativeProviderSession,
    model_root: &Path,
    target: Arc<K3Tokenizer>,
    variant: QwenVariant,
    context_capacity: usize,
) -> Result<NativeQwenDraft> {
    let checkpoint = QwenCheckpoint::open(model_root, variant)?;
    let tokenizer = QwenTokenizer::load(checkpoint.root())?;
    let model = NativeQwen::bind_with_context_capacity(provider, &checkpoint, context_capacity)?;
    Ok(FailSoftQwenDraft::new(QwenDraftController::new(
        model, target, tokenizer,
    )))
}

/// A borrow-only startup report.  Producing it performs no provider discovery,
/// filesystem walk, or allocation; all values come from the one engine owner.
#[derive(Debug, Clone, Copy)]
pub struct NativeEngineStatus<'a> {
    pub model_root: &'a Path,
    pub device: Device,
    pub device_selection_policy: DeviceSelectionPolicy,
    pub expert_backend: ResolvedExpertBackend,
    pub expert_storage: ExpertStorageLayout,
    pub libtorch_version: &'a str,
    pub spine_source: SpinePlanSource,
    pub spine_representation: SpineRepresentation,
    pub spine_layers: usize,
    pub global_transfer_groups: usize,
    pub resident_prefix_layers: usize,
    pub resident_prefix_bytes: u64,
    pub transient_layer_bytes: u64,
    pub context_growth: ContextGrowthBudget,
    pub verify_snapshots: VerifySnapshotBudget,
    pub spine_reader_workers: usize,
    pub spine_fd_cache_descriptors: usize,
    pub spine_fd_cache_opened: usize,
    pub expert_reader_workers: usize,
    pub expert_prefetch_reader_workers: usize,
    pub router_trace_enabled: bool,
    pub expert_heat_recording: bool,
    pub expert_heat_weight: f64,
    pub expert_pin_candidates: usize,
    pub expert_pin_charged_bytes: u64,
    pub expert_pin_resident_experts: u64,
    pub expert_pin_resident_bytes: u64,
    pub expert_pin_hits: u64,
    pub expert_pin_served_bytes: u64,
    pub expert_cpu_threads: usize,
    pub speculative_max_drafts: usize,
    pub qwen_state: QwenRuntimeState,
    pub qwen_reserved_provider_bytes: u64,
    pub qwen_reserved_verify_bytes: u64,
    pub qwen_reserved_verify_positions: u64,
    pub qwen_context_capacity: usize,
    pub qwen_telemetry: QwenTelemetry,
    pub qwen_last_error: Option<&'a str>,
    pub lazy_expert_files_missing: usize,
    pub readiness: NativeTargetReadiness,
}

/// Provider-memory contract for exact MLA context growth.
///
/// `initial_provider_bytes` is charged before resident-layer selection.
/// Every later geometric capacity increase must be admitted against a fresh
/// provider/host snapshot using `bytes_per_capacity_token`; bootstrap never
/// labels that future context memory as available for pinned layers.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ContextGrowthBudget {
    /// Complete storage for one additional capacity position across all 24
    /// MLA layers.
    pub bytes_per_capacity_token: u64,
    /// Complete storage for one capacity position in one MLA layer. Keeping
    /// this explicit lets admission include the provider's intermediate
    /// geometric reallocations instead of hiding them in an average.
    pub bytes_per_layer_capacity_token: u64,
    pub mla_layers: u64,
    pub initial_capacity_tokens: u64,
    pub initial_provider_bytes: u64,
    /// Architectural checkpoint contract. This is not a claim that the
    /// current expanded cache representation can physically retain it.
    pub model_max_context_tokens: u64,
    /// Fail-closed bound of the current exact expanded MLA representation.
    /// A future reviewed latent cache may raise this without changing K3.
    pub admitted_expanded_context_tokens: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ContextGrowthAdmission {
    pub committed_capacity_tokens: u64,
    pub next_capacity_tokens: u64,
    /// Storage already owned by all 24 MLA caches.
    pub committed_provider_bytes: u64,
    /// Final new storage that must coexist with the committed storage until
    /// the whole-model transaction commits. Live admission also includes the
    /// bounded geometric-growth scratch below.
    pub staged_provider_bytes: u64,
    /// One layer grows through intermediate geometric capacities before its
    /// final allocation. Allocators may retain those freed blocks for reuse by
    /// later layers, so admission conservatively charges one complete copy of
    /// every intermediate staged capacity.
    pub growth_scratch_provider_bytes: u64,
    /// Old committed plus final staged storage plus growth scratch.
    pub transaction_peak_provider_bytes: u64,
}

impl ContextGrowthBudget {
    /// Headroom withheld from resident-layer selection for early exact cache
    /// growth. The committed initial generation is charged separately; this
    /// is only the additional generation and allocator scratch which must be
    /// simultaneously available during growth.
    pub fn startup_growth_reserve(self) -> Result<(u64, u64)> {
        let target = K3_MLA_STARTUP_RESERVE_TOKENS
            .min(self.admitted_expanded_context_tokens)
            .max(self.initial_capacity_tokens);
        if target <= self.initial_capacity_tokens {
            return Ok((target, 0));
        }
        let admission = self.admission(self.initial_capacity_tokens, target)?;
        let bytes = admission
            .staged_provider_bytes
            .checked_add(admission.growth_scratch_provider_bytes)
            .ok_or_else(|| DeltafinError::new("MLA startup growth reserve overflows u64"))?;
        Ok((target, bytes))
    }

    pub fn admission(
        self,
        committed_capacity_tokens: u64,
        needed_tokens: u64,
    ) -> Result<ContextGrowthAdmission> {
        if committed_capacity_tokens > self.admitted_expanded_context_tokens
            || needed_tokens <= committed_capacity_tokens
            || needed_tokens > self.admitted_expanded_context_tokens
        {
            return Err(DeltafinError::new(
                "MLA context growth request is outside the current exact-expanded provider bound",
            ));
        }
        if self.mla_layers == 0
            || self
                .bytes_per_layer_capacity_token
                .checked_mul(self.mla_layers)
                != Some(self.bytes_per_capacity_token)
        {
            return Err(DeltafinError::new(
                "MLA context-growth budget has inconsistent per-layer storage",
            ));
        }

        // provider_target_sequence.cpp appends one row at a time. Therefore a
        // 64-row chunk does not jump directly from capacity 0 to ceil(64*1.25)
        // (80): it follows 0->16->24->36->54->81. Simulate that exact sequence
        // or Rust would understate both the committed capacity and live peak.
        let mut next_capacity_tokens = committed_capacity_tokens;
        let mut intermediate_staged_capacity_tokens = 0_u64;
        while next_capacity_tokens < needed_tokens {
            let previous_capacity = next_capacity_tokens;
            let grown = if previous_capacity == 0 {
                self.initial_capacity_tokens
            } else {
                // Mirrors provider_mla.cpp's ceil(capacity * 1.5).
                previous_capacity
                    .checked_mul(3)
                    .and_then(|value| value.checked_add(1))
                    .map(|value| value / 2)
                    .ok_or_else(|| DeltafinError::new("MLA geometric capacity overflows u64"))?
            };
            if previous_capacity > committed_capacity_tokens {
                intermediate_staged_capacity_tokens = intermediate_staged_capacity_tokens
                    .checked_add(previous_capacity)
                    .ok_or_else(|| {
                        DeltafinError::new("MLA intermediate capacity scratch overflows u64")
                    })?;
            }
            next_capacity_tokens = grown.min(self.admitted_expanded_context_tokens);
            if next_capacity_tokens <= previous_capacity {
                return Err(DeltafinError::new(
                    "MLA context growth made no progress inside its admitted bound",
                ));
            }
        }
        let committed_provider_bytes = self
            .bytes_per_capacity_token
            .checked_mul(committed_capacity_tokens)
            .ok_or_else(|| DeltafinError::new("committed MLA storage overflows u64"))?;
        let staged_provider_bytes = self
            .bytes_per_capacity_token
            .checked_mul(next_capacity_tokens)
            .ok_or_else(|| DeltafinError::new("staged MLA storage overflows u64"))?;
        let growth_scratch_provider_bytes = self
            .bytes_per_layer_capacity_token
            .checked_mul(intermediate_staged_capacity_tokens)
            .ok_or_else(|| DeltafinError::new("MLA growth scratch overflows u64"))?;
        let transaction_peak_provider_bytes = committed_provider_bytes
            .checked_add(staged_provider_bytes)
            .and_then(|value| value.checked_add(growth_scratch_provider_bytes))
            .ok_or_else(|| DeltafinError::new("MLA growth transaction peak overflows u64"))?;
        Ok(ContextGrowthAdmission {
            committed_capacity_tokens,
            next_capacity_tokens,
            committed_provider_bytes,
            staged_provider_bytes,
            growth_scratch_provider_bytes,
            transaction_peak_provider_bytes,
        })
    }
}

/// Exact KDA snapshot cost of wide target verification.
///
/// Ordinary decode/prefill reserves one committed and one staged generation.
/// Verify must keep one staged boundary per candidate row so an accepted
/// prefix can commit without replay. The caller must admit this peak against a
/// fresh memory snapshot; failure selects ordinary exact target decode, never a
/// shorter approximate verification or a target-quality change.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct VerifySnapshotBudget {
    pub bytes_per_kda_generation: u64,
    /// What a multi-row verify keeps per row so a commit can replay any
    /// accepted prefix's KDA state: the recurrence inputs, not the states.
    pub replay_bytes_per_position: u64,
    /// The record's fixed part: each convolution source's history window.
    pub replay_fixed_bytes: u64,
    pub max_positions: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct VerifySnapshotAdmission {
    pub positions: u64,
    pub committed_provider_bytes: u64,
    pub staged_boundary_provider_bytes: u64,
    pub transaction_peak_provider_bytes: u64,
    /// Extra over the already-fixed ordinary committed+T1-staged reserve.
    pub additional_over_decode_reserve_bytes: u64,
}

impl VerifySnapshotBudget {
    pub fn admission(self, positions: u64) -> Result<VerifySnapshotAdmission> {
        if positions == 0 || positions > self.max_positions {
            return Err(DeltafinError::new(format!(
                "target verification width must be in 1..={}",
                self.max_positions
            )));
        }
        let committed_provider_bytes = self.bytes_per_kda_generation;
        // Every verify stages the state after all of its rows. More than one
        // row also keeps a prefix record (provider_target_sequence.h); the
        // replayed state of a shorter commit replaces the full one layer by
        // layer, so no second generation is ever live.
        let record_bytes = if positions > 1 {
            self.replay_bytes_per_position
                .checked_mul(positions)
                .and_then(|bytes| bytes.checked_add(self.replay_fixed_bytes))
                .ok_or_else(|| DeltafinError::new("verify KDA record overflows u64"))?
        } else {
            0
        };
        let staged_boundary_provider_bytes = self
            .bytes_per_kda_generation
            .checked_add(record_bytes)
            .ok_or_else(|| DeltafinError::new("verify KDA snapshots overflow u64"))?;
        let transaction_peak_provider_bytes = committed_provider_bytes
            .checked_add(staged_boundary_provider_bytes)
            .ok_or_else(|| DeltafinError::new("verify transaction peak overflows u64"))?;
        let ordinary_decode_reserve = self
            .bytes_per_kda_generation
            .checked_mul(2)
            .ok_or_else(|| DeltafinError::new("ordinary KDA reserve overflows u64"))?;
        Ok(VerifySnapshotAdmission {
            positions,
            committed_provider_bytes,
            staged_boundary_provider_bytes,
            transaction_peak_provider_bytes,
            additional_over_decode_reserve_bytes: transaction_peak_provider_bytes
                .saturating_sub(ordinary_decode_reserve),
        })
    }
}

/// Persistent owner of the native target runtime.
///
/// Every large or performance-sensitive resource has one owner here: one
/// provider session, one layer-stream reader, one expert reader, one exact
/// embedding descriptor/arena, and one compiled manifest.  Request handling
/// must borrow these resources; it must not rediscover the model or create a
/// replacement worker pool.
pub struct NativeTargetEngine {
    model_root: PathBuf,
    model: ModelSpec,
    program: TargetProgram,
    inventory: NativeProviderInventory,
    device: Device,
    device_selection_policy: DeviceSelectionPolicy,
    expert_backend: ResolvedExpertBackend,
    expert_cpu_threads: usize,
    ngram_drafter: NgramDraftSource,
    /// The optional proposal slot: EAGLE-3.1 when installed, else DSpark.
    dspark: DSparkRuntime<NativeDraftBackend>,
    qwen: QwenRuntime,
    speculative_max_drafts: usize,
    complete_expert_union: bool,
    complete_expert_union_reserved_capacity: Option<usize>,
    metal_expert_wrapper_retention: bool,
    /// Whether a decode layer starts each missing expert's matmul as its own
    /// bytes land instead of after the layer's whole miss set. Scheduling
    /// only; the reduction stays in route order either way.
    expert_early_drain: bool,
    tokenizer: Arc<K3Tokenizer>,
    output_decoder: IncrementalUtf8Decoder,
    embedding: EmbeddingTable,
    embedding_arena: EmbeddingArena,
    experts: RawExpertCorpus,
    /// Drives holding copies of model files; shared by every model reader.
    storage_homes: Option<Arc<StorageHomes>>,
    expert_reader: Reader,
    expert_prefetch_reader: Option<Reader>,
    pilot_gate: Option<PilotGate>,
    expert_heat: ExpertHeat,
    expert_pin_tier: Option<ExpertPinTier>,
    router_trace: RouterTrace,
    global_plans: Box<[GlobalSpinePlan]>,
    spine: CompiledSpine,
    provider: NativeProviderSession,
    spine_pipeline: SpinePipeline,
    residency: ResidencySelection,
    transient_layer_bytes: u64,
    context_growth: ContextGrowthBudget,
    committed_context_tokens: u64,
    mla_capacity_tokens: u64,
    /// A bounded request's whole-context ceiling, staged at request start and
    /// consumed by its first chunk's admission; see
    /// [`context_pre_admission_envelope`]. Advisory: a declined envelope
    /// falls back to ordinary incremental growth.
    pending_context_envelope: Option<u64>,
    verify_snapshots: VerifySnapshotBudget,
    /// Verify bytes the startup plan withheld for the slot drafter.
    slot_verify_reserved_bytes: u64,
    metal_source_selector: Option<String>,
    /// Configured chat thinking depth; `None` defers to the template's `max`.
    reasoning_effort: Option<String>,
    readiness: NativeTargetReadiness,
    lifecycle: NativeEngineLifecycle,
    stream_boundary: NativeStreamBoundary,
    target_state_capability: TargetStateTransactionCapability,
    target_reuse_identity: TargetReuseIdentity,
    published_target_boundary: Option<PublishedTargetBoundary>,
    pending_target_publication: Option<PendingTargetPublication>,
    next_target_boundary_id: u64,
}

impl Drop for NativeTargetEngine {
    fn drop(&mut self) {
        // Persist any expert-heat deltas the periodic cadence has not merged
        // yet. Best-effort by contract: teardown proceeds identically when
        // the file or its lock is unavailable.
        self.expert_heat.flush_best_effort();
        // Drop bodies run before fields. This explicit ordering lets the
        // pipeline synchronously abort a borrowed CPU/CUDA/Metal source while
        // the provider session and its completion primitives are still live.
        // If native completion cannot be proven, retain both the reader lease
        // (inside the poisoned pipeline) and the complete provider session
        // until process exit rather than freeing either side of an unknown
        // asynchronous use.
        // Expert calls are synchronous, so no dispatch can still be using an
        // arena at this exclusive teardown boundary. Clear the process-global
        // no-copy wrapper cache before either reader field can release pages.
        // Each arena also owns the same hook as a final fail-safe.
        let metal_cache_failed = self.expert_backend == ResolvedExpertBackend::Metal
            && self.provider.flush_metal_expert_cache().is_err();
        if self.spine_pipeline.teardown(&self.provider).is_err() || metal_cache_failed {
            self.provider.suppress_destroy_after_unproven_source_use();
        }
    }
}

fn select_target_for_spine(
    inventory: &NativeProviderInventory,
    request: DeviceRequest,
    representation: SpineRepresentation,
    apple_family: Option<u64>,
) -> Result<DeviceSelection> {
    let original_bf16 = representation == SpineRepresentation::OriginalBf16;
    let mut eligible = inventory.providers;
    if original_bf16 && !inventory.cuda_exact_bf16_compiled {
        match request {
            DeviceRequest::Cuda(index) => {
                return Err(DeltafinError::new(format!(
                    "CUDA device {index} was requested for original-BF16 K3, but this Deltafin binary was built without its exact RAW_BF16 CUDA kernel; install the matching audited NVCC toolkit and rebuild/upgrade Deltafin, or explicitly use --spine int8. Deltafin will not expand the checkpoint weights to resident FP32"
                )));
            }
            DeviceRequest::Auto => {
                // CUDA remains a valid target for the separately requested
                // int8 spine.  Mask it only for this original-BF16 selection
                // so Auto fails safely to another exact backend rather than
                // selecting CUDA and discovering the missing kernel at bind.
                eligible.cuda_devices = 0;
            }
            DeviceRequest::Cpu | DeviceRequest::Mps => {}
        }
    }
    eligible.select_target(request, original_bf16, apple_family)
}

impl NativeTargetEngine {
    pub fn bootstrap(config: &RuntimeConfig) -> Result<Self> {
        let representation = match config.spine {
            // The default resident spine is the measured row-int8 conversion;
            // the original checkpoint remains selectable explicitly.
            SpineRequest::Auto | SpineRequest::Int8 => SpineRepresentation::QuantizedInt8,
            SpineRequest::Bf16 => SpineRepresentation::OriginalBf16,
        };
        let expected_authority = match representation {
            SpineRepresentation::OriginalBf16 => ResidentWeightAuthority::OriginalBf16,
            SpineRepresentation::QuantizedInt8 => ResidentWeightAuthority::QuantizedInt8,
        };
        if config.quality.resident_weights != expected_authority {
            return Err(DeltafinError::new(
                "resident-spine request disagrees with the fail-closed quality policy",
            ));
        }

        // Validate accelerator and expert-backend requests before touching the
        // model tree. A misspelled/incompatible explicit backend should not
        // trigger tens of thousands of manifest entries or any weight I/O.
        let inventory = NativeProvider::inventory()?;
        let device_selection = select_target_for_spine(
            &inventory,
            config.device,
            representation,
            apple_cpu_family(),
        )?;
        let device = device_selection.device;
        let expert_backend = resolve_expert_backend(
            config.expert_backend,
            device,
            inventory.providers,
            inventory.cuda_moe_compiled,
        )?;
        let expert_cpu_threads = configured_expert_cpu_threads()?;
        let speculative_max_drafts = configured_speculative_max_drafts()?;
        let complete_expert_union = configured_complete_expert_union()?;

        // Resolve the root once. All subordinate paths derive from this stable
        // absolute spelling instead of repeating current-directory lookups.
        let model_root = crate::sys::fs::canonicalize(&config.model_root).map_err(|error| {
            DeltafinError::new(format!(
                "resolve model root {}: {error}",
                config.model_root.display()
            ))
        })?;
        if !model_root.is_dir() {
            return Err(DeltafinError::new(format!(
                "model root is not a directory: {}",
                model_root.display()
            )));
        }
        let router_trace_path = if config.router_trace_mode == RouterTraceMode::Off {
            None
        } else {
            Some(match config.router_trace_path.as_deref() {
                Some(path) if path.is_absolute() => path.to_path_buf(),
                Some(path) => model_root.join(path),
                None => model_root.join("k3-meta/router_trace.jsonl"),
            })
        };
        let router_trace =
            RouterTrace::open(config.router_trace_mode, router_trace_path.as_deref())?;

        let model = ModelSpec::load_from_root(&model_root)?;
        let program = TargetProgram::compile_with_representation(&model, representation)?;
        let layout = SourceLayout::under(&model_root);
        if representation == SpineRepresentation::QuantizedInt8 && !layout.int8_tensors.is_dir() {
            return Err(DeltafinError::new(format!(
                "the default row-int8 resident spine is not installed at {}; run `deltafin convert-spine-int8` (setup prepares it automatically), or select the original weights explicitly with --spine bf16",
                layout.int8_tensors.display(),
            )));
        }
        validate_embedding_contract(&model_root, &model, &program, &layout)?;

        // The exact table is one persistent descriptor. Only bounded rows are
        // staged; the 2.35 GB table is never made provider-resident or q8.
        let embedding = EmbeddingTable::open_k3(&model_root)?;
        let embedding_arena = EmbeddingArena::new(EMBEDDING_ARENA_ROWS)?;
        let tokenizer = Arc::new(K3Tokenizer::load_from_root(&model_root)?);
        let output_decoder = IncrementalUtf8Decoder::new();

        let provider = NativeProviderSession::target(device)?;
        let cache_capabilities = provider.memory_snapshot(false)?;
        let cache_host = probe_host_memory();
        let source_layer_bytes = program.int8_stream_layer_bytes()?;
        let spine_cache = configured_spine_cache_plan(
            config.spine_stream_nocache,
            config.spine_resident_bytes,
            cfg!(target_os = "macos"),
            device,
            representation,
            cache_host.physical_bytes,
            cache_capabilities.recommended_bytes,
            &source_layer_bytes,
        );
        let automatic_streaming = spine_cache.stream_nocache
            && config.spine_stream_nocache.is_none()
            && representation == SpineRepresentation::QuantizedInt8
            && device == Device::Mps;
        let max_buffer_length = metal_max_buffer_length(device);
        let available_cpus = std::thread::available_parallelism().map_or(1, usize::from);
        let pack_directory = model_root.join(program.representation().pack_directory_name());
        let loose_spine =
            spine_source_intent(&pack_directory)? == SpinePlanSource::LooseDeferredFiles;
        // Extra drives holding copies of the experts and/or the int8 spine.
        // Every model reader shares one set so all storage traffic counts
        // against the same physical devices.
        let storage_homes = StorageHomes::open(&model_root, &config.storage_homes)?;
        if let Some(homes) = &storage_homes {
            for (index, drive) in homes.stats().iter().enumerate() {
                eprintln!(
                    "[native] storage drive {index}: {}{}",
                    drive.root.display(),
                    if index == 0 {
                        " (model root)".to_string()
                    } else {
                        format!(" ({} copies)", drive.mirrored_files)
                    },
                );
            }
        }
        let storage_drives = storage_homes.as_ref().map_or(1, |homes| homes.device_count());
        let request_fd_cache = resolve_loose_spine_fd_cache(
            config.spine_fd_cache,
            loose_spine,
            // A persistent descriptor pins each spine file to whichever drive
            // opened it first; with storage homes every read chooses afresh.
            automatic_streaming && storage_homes.is_none(),
            qualified_spine_resource_tuple(
                cfg!(target_os = "macos"),
                cache_host.physical_bytes,
                available_cpus,
                cache_capabilities.recommended_bytes,
                max_buffer_length,
            ),
        );
        let persistent_loose_descriptors = if loose_spine && request_fd_cache {
            match prepare_persistent_descriptor_capacity(
                program.source_components(),
                LOOSE_SPINE_DESCRIPTOR_RESERVE,
            ) {
                Ok(()) => true,
                Err(error) if config.spine_fd_cache == Some(true) => {
                    return Err(DeltafinError::new(format!(
                        "K3_SPINE_FDCACHE=1 cannot admit the complete immutable loose-spine roster: {error}"
                    )));
                }
                Err(_) => false,
            }
        } else {
            false
        };

        // Globals are compiled once and kept separate from layer packs so one
        // temporary startup slab can be released immediately after each
        // immutable provider bind.
        let global_plans = program.global_loose_read_plans_default(&layout)?;
        let spine_candidate = compile_spine(
            &program,
            &model_root,
            &layout,
            &spine_cache.policies,
            persistent_loose_descriptors,
        );
        let spine = match spine_candidate {
            Ok(spine) => spine,
            Err(_) if persistent_loose_descriptors && config.spine_fd_cache.is_none() => {
                // Automatic admission is transactional: a descriptor race or
                // roster drift drops every partial reservation and rebuilds
                // the same immutable manifests with batch-local descriptors.
                compile_spine(&program, &model_root, &layout, &spine_cache.policies, false)?
            }
            Err(error) if persistent_loose_descriptors && config.spine_fd_cache == Some(true) => {
                return Err(DeltafinError::new(format!(
                    "K3_SPINE_FDCACHE=1 failed while publishing the complete loose-spine descriptor roster: {error}"
                )));
            }
            Err(error) => return Err(error),
        };

        let target_state_capability = provider.target_state_transaction_capability();
        let metal_source_selector = resolve_metal_source(expert_backend)?;
        // Compact storage is never inferred from a filename. The selected
        // Metal bridge must first initialize with the production shader and
        // advertise its complete descriptor suite; only then may a complete,
        // authenticated scale4 manifest replace the explicit raw layout.
        let experts = open_expert_corpus(
            config.expert_scale4,
            &model_root,
            expert_backend,
            &provider,
            metal_source_selector.as_deref(),
            expert_stream_cache_policy(config.expert_stream_nocache, cfg!(target_os = "macos")),
        )?;
        // Cross-run learned expert residency. The heat histogram observes the
        // authoritative routes of every backend; a byte budget freezes a
        // candidate roster whose spans promote lazily on their first
        // authoritative read. The automatic default sizes the budget to
        // exactly the qualifying roster, capped by live free memory minus a
        // reserved headroom — thin history or a tight host resolves to off.
        // CUDA keeps its provider-owned device cache, so the host tier is
        // limited to the scattered-span backends.
        let expert_heat = ExpertHeat::open(&model_root, config.expert_heat);
        let expert_pin_budget_bytes = match config.expert_pin_bytes {
            Some(bytes) => bytes,
            None => resolve_auto_pin_budget(
                expert_heat.snapshot(),
                experts.layout().expert_span_bytes(),
                probe_host_memory().available_bytes,
            ),
        };
        // The tier is sized and charged only after residency and the Qwen
        // drafter are chosen (see the admission below), so a pin budget can
        // never silently push out a drafter or a spine that fits completely.
        let requested_pin_tier = (expert_pin_budget_bytes > 0
            && matches!(
                expert_backend,
                ResolvedExpertBackend::Cpu | ResolvedExpertBackend::Metal
            ))
        .then(|| {
            ExpertPinTier::plan(
                expert_heat.snapshot(),
                experts.layout().expert_span_bytes(),
                expert_pin_budget_bytes,
            )
        })
        .flatten();
        let expert_reader_workers =
            configured_expert_reader_workers(config.expert_read_threads, storage_drives);
        let metal_expert_retire_hook: Option<BufferRetireHook> =
            (expert_backend == ResolvedExpertBackend::Metal).then(|| {
                let session = provider.lease();
                Arc::new(move || session.flush_metal_expert_cache()) as BufferRetireHook
            });
        let metal_expert_wrapper_retention = metal_expert_retire_hook.is_some();
        let mut expert_reader = Reader::with_arena_capacity_and_retire_hook(
            expert_reader_workers,
            EXPERT_ARENA_SLOTS,
            metal_expert_retire_hook.clone(),
        )?;
        expert_reader.set_storage_homes(storage_homes.clone());
        // The nine-row verifier's compact expert slab is a startup resource,
        // not a decode-time arena growth. Reserve it while the demand Reader
        // is still empty: no Metal wrapper can alias the slot, so admission
        // cannot invoke the retire hook or flush process-global Metal state.
        // A failed live-memory proof keeps the exact <=64-expert tiled path.
        // The later residency snapshot observes this live allocation and the
        // ordinary 64-expert fixed reserve remains conservatively charged.
        let complete_expert_union_reserved_capacity = reserve_complete_expert_union_at_startup(
            complete_expert_union,
            target_expert_backend(expert_backend),
            &expert_reader,
            experts.layout(),
            probe_host_memory(),
        );
        // A partial installation cannot safely speculate because a prediction
        // must never trigger a remote fetch. CUDA uses its separate provider-
        // owned cache plan and requires one contiguous miss slab, so the
        // scattered host-span path is intentionally limited to CPU/Metal.
        let expert_prefetch_enabled = experts.lazy_missing_files() == 0
            && matches!(
                expert_backend,
                ResolvedExpertBackend::Cpu | ResolvedExpertBackend::Metal
            );
        let expert_prefetch_reader = expert_prefetch_enabled
            .then(|| {
                Reader::with_arena_capacity_and_retire_hook(
                    expert_reader_workers,
                    EXPERT_PREFETCH_ARENA_SLOTS,
                    metal_expert_retire_hook.clone(),
                )
                .map(|mut reader| {
                    reader.set_storage_homes(storage_homes.clone());
                    reader
                })
            })
            .transpose()?;
        // The gate governs only speculative reads, so it exists exactly when
        // the speculative reader does; `K3_PILOT_GATE=off` keeps the legacy
        // ungoverned scheduler with no governor in the loop.
        let pilot_gate = expert_prefetch_enabled
            .then(|| {
                PilotGate::new(
                    config.pilot_gate,
                    config.pilot_gate_threshold,
                    config.pilot_gate_warmup,
                )
            })
            .flatten();

        // Ordinary Reader slabs allocate lazily; the optional compact wide
        // expert slab above is already reflected in the live host snapshot.
        // Keep charging the ordinary component-wise high-water marks and all
        // future global provider allocations before admitting residency.
        let context_growth = context_growth_budget(&model)?;
        let verify_snapshots = verify_snapshot_budget(&model)?;
        let expert_prefetch_bytes = if expert_prefetch_enabled {
            u64::try_from(experts.layout().expert_span_bytes())
                .ok()
                .and_then(|bytes| bytes.checked_mul(EXPERT_PREFETCH_LIVE_SLOTS as u64))
                .ok_or_else(|| DeltafinError::new("expert prefetch arena budget overflows u64"))?
        } else {
            0
        };
        let execution_arena_reserve = fp32_spine_execution_arena_reserve(&program, device)?;
        // Compact int8/original-BF16 accelerator storage expands only the
        // current layer into one serially reused FP32 matrix arena. Automatic
        // server reuse retains one complete KDA parent, and optional DSpark /
        // PILOT storage is likewise fixed before any retained layer is chosen.
        let eagle3_context = eagle3_context(context_growth);
        let slot_model_reserve =
            draft_slot_provider_reserve(config, &model_root, device, eagle3_context)?;
        // A slot drafter verifies through ordinary prefix commits. Withhold
        // the widest verify's prefix record now, as Qwen withholds its
        // verifier: a plan that spends every spare byte on pins and spine
        // otherwise refuses even that ~200 MiB at run time.
        let slot_verify_reserved_bytes = if slot_model_reserve == 0 {
            0
        } else {
            slot_verify_reserve(verify_snapshots, speculative_max_drafts)?
        };
        let dspark_reserve = slot_model_reserve
            .checked_add(slot_verify_reserved_bytes)
            .ok_or_else(|| DeltafinError::new("draft slot reserve overflows u64"))?;
        let fixed_provider_addition = execution_arena_reserve
            .checked_add(verify_snapshots.bytes_per_kda_generation)
            .and_then(|bytes| bytes.checked_add(dspark_reserve))
            .and_then(|bytes| {
                bytes.checked_add(if expert_prefetch_enabled {
                    TARGET_PILOT_RESERVE_BYTES
                } else {
                    0
                })
            })
            .ok_or_else(|| DeltafinError::new("native fixed provider reserve overflows u64"))?;
        let fixed_host_addition = if router_trace.enabled() {
            ROUTER_TRACE_HOST_RESERVE_BYTES
        } else {
            0
        };
        let fixed_host_addition = fixed_host_addition
            .checked_add(if expert_heat.recording() {
                EXPERT_HEAT_HOST_RESERVE_BYTES
            } else {
                0
            })
            .ok_or_else(|| {
                DeltafinError::new("expert-heat host reserve overflows fixed host costs")
            })?;
        let fixed = fixed_costs(
            &program,
            &global_plans,
            &spine,
            EMBEDDING_ARENA_ROWS,
            expert_prefetch_bytes,
            context_growth,
            verify_snapshots,
        )?;
        let fixed = FixedCosts {
            host_bytes: fixed
                .host_bytes
                .checked_add(fixed_host_addition)
                .ok_or_else(|| DeltafinError::new("native fixed host reserve overflows u64"))?,
            provider_bytes: fixed
                .provider_bytes
                .checked_add(fixed_provider_addition)
                .ok_or_else(|| DeltafinError::new("native fixed provider reserve overflows u64"))?,
        };
        let provider_layer_bytes = provider_layer_costs(&program)?;
        let host_memory = probe_host_memory();
        let provider_memory_snapshot = provider.memory_snapshot(false)?;
        let provider_memory = provider_memory(provider_memory_snapshot);
        // The CUDA expert cache is budgeted from the same snapshot that
        // residency consumes, then charged as a fixed provider cost so the
        // resident spine prefix and the cache can never double-book VRAM.
        // Streaming-critical experts outrank resident spine layers.
        let cuda_expert_cache = plan_cuda_expert_cache_budget(
            device,
            provider_memory_snapshot,
            fixed.provider_bytes,
            provider_layer_bytes.iter().copied().max().unwrap_or(0),
        );
        let fixed = match &cuda_expert_cache {
            Some(budget) => FixedCosts {
                host_bytes: fixed.host_bytes,
                provider_bytes: fixed
                    .provider_bytes
                    .checked_add(budget.charged_provider_bytes)
                    .ok_or_else(|| {
                        DeltafinError::new(
                            "CUDA expert-cache reserve overflows fixed provider costs",
                        )
                    })?,
            },
            None => fixed,
        };
        let qwen_context_capacity = qwen_context_capacity(context_growth)?;
        let discovered_qwen = if qwen_usable_by_run(config) {
            discover_qwen_plan(
                config.qwen,
                &model_root,
                device,
                qwen_context_capacity,
                verify_snapshots,
                speculative_max_drafts,
            )
        } else {
            QwenPlan::inactive(QwenRuntimeState::UnusedByRun)
        };
        let residency_override = ResidencyOverride {
            requested_layers: config.provider_resident_layers,
            requested_provider_bytes: None,
        };
        let plan_selector = PlanSelector {
            host_memory,
            provider_memory,
            provider_layer_bytes: &provider_layer_bytes,
            residency_override,
            discovered_qwen,
            device,
        };
        let select_plan = |fixed: FixedCosts| plan_selector.select(fixed);
        let (qwen_plan, residency, transient_layer_bytes) = select_plan(fixed)?;
        // Learned-pin admission, by measured marginal value:
        // - the Qwen drafter outranks pins (up to 2x on draftable text versus
        //   +15-19% for pins, which are near-neutral once drafting runs);
        // - a spine that fits completely outranks pins: streaming it back in
        //   costs every pass (-23% on a 128 GB host when pins displaced 60
        //   layers), and so does a prefix the operator explicitly requested;
        // - pins outrank a spine that only fits partly (64 GB M1 Max: +15-19%
        //   for a ~1.9 GB roster against +0.23% for 1.7 GB more spine).
        // The tier gets the largest whole-expert roster whose charge keeps
        // the protected parts of the pin-free plan intact.
        let (expert_pin_tier, qwen_plan, residency, transient_layer_bytes) = match requested_pin_tier {
            None => (None, qwen_plan, residency, transient_layer_bytes),
            Some(requested) => {
                let span_bytes = experts.layout().expert_span_bytes() as u64;
                let keep_spine = pin_tier_keeps_resident_prefix(residency.stop);
                let admitted = admit_pin_count(
                    requested.candidate_count(),
                    span_bytes,
                    fixed,
                    &select_plan,
                    qwen_plan,
                    &residency,
                    keep_spine,
                );
                if admitted < requested.candidate_count() {
                    let protected = match (keep_spine, qwen_plan.initial.is_some()) {
                        (true, true) => "the resident spine prefix and the Qwen plan keep priority",
                        (true, false) => "the resident spine prefix keeps priority",
                        (false, true) => "the Qwen plan keeps priority",
                        (false, false) => "the live memory proof still holds",
                    };
                    eprintln!(
                        "[native] expert pin tier: memory admission kept {} of {} candidate experts ({:.2} of {:.2} GiB) so that {protected}",
                        admitted,
                        requested.candidate_count(),
                        (admitted as u64 * span_bytes) as f64 / f64::from(1_u32 << 30),
                        requested.charged_host_bytes() as f64 / f64::from(1_u32 << 30),
                    );
                }
                let tier = (admitted != 0)
                    .then(|| {
                        ExpertPinTier::plan(
                            expert_heat.snapshot(),
                            span_bytes as usize,
                            admitted as u64 * span_bytes,
                        )
                    })
                    .flatten();
                match tier {
                    Some(tier) => {
                        let host_bytes = fixed
                            .host_bytes
                            .checked_add(tier.charged_host_bytes())
                            .ok_or_else(|| {
                                DeltafinError::new("expert-pin host reserve overflows fixed host costs")
                            })?;
                        let (qwen_plan, residency, transient_layer_bytes) = select_plan(FixedCosts {
                            host_bytes,
                            provider_bytes: fixed.provider_bytes,
                        })?;
                        (Some(tier), qwen_plan, residency, transient_layer_bytes)
                    }
                    None => (None, qwen_plan, residency, transient_layer_bytes),
                }
            }
        };
        let spine_reader_workers = configured_spine_reader_workers(
            config.spine_read_threads,
            // The measured six-reader tuple belongs to automatic streaming.
            // An explicit cache override keeps the portable reader default
            // unless its independent thread override was also supplied.
            automatic_streaming,
            host_memory,
            provider_memory_snapshot,
            max_buffer_length,
        );
        // Each extra drive needs its own queue depth to stream its share.
        let spine_reader_workers = if config.spine_read_threads.is_none() && storage_drives > 1 {
            (spine_reader_workers + 2 * (storage_drives - 1))
                .min(crate::config::MAX_SPINE_READ_THREADS)
        } else {
            spine_reader_workers
        };
        let mut spine_pipeline = SpinePipeline::with_resident_prefix(
            spine_reader_workers,
            SPINE_ARENA_SLOTS,
            u32::try_from(residency.resident_layers).map_err(|_| {
                DeltafinError::new("resident layer prefix does not fit the spine ABI")
            })?,
        )?;
        spine_pipeline.set_storage_homes(storage_homes.clone());

        // Freeze the session's expert-cache budget before anything can run a
        // target sequence or availability probe (both freeze the policy). An
        // explicit zero is still configured: it makes capacity-starved
        // sessions deterministically stream instead of letting the provider
        // re-budget against VRAM that residency has already promised away.
        if let Some(budget) = &cuda_expert_cache {
            provider.configure_cuda_expert_cache(budget.policy)?;
            eprintln!(
                "[native] CUDA expert cache: {} experts ({:.2} GiB), device reserve {:.2} GiB, resident spine prefix {} layers",
                budget.policy.capacity_experts,
                budget.charged_provider_bytes as f64 / (1_u64 << 30) as f64,
                budget.policy.reserve_bytes as f64 / (1_u64 << 30) as f64,
                residency.resident_layers,
            );
        }

        // Admit PILOT only after its complete provider-memory reserve has
        // participated in residency selection, and before the first global or
        // layer bind. Its predictions remain scheduling-only; the independent
        // authoritative K3 mailbox still decides all executed experts.
        if expert_prefetch_enabled {
            let admission = provider.enable_target_pilot()?;
            if admission.reserve_bytes != TARGET_PILOT_RESERVE_BYTES {
                return Err(DeltafinError::new(
                    "target PILOT admission differs from its charged memory reserve",
                ));
            }
        }

        // Bind both immutable global groups only after residency selection:
        // the preceding fixed-cost calculation already reserved their future
        // provider storage, while this short-lived reader never overlaps the
        // persistent layer/expert arena high-water phase.
        bind_target_globals_once(&provider, &global_plans)?;
        let dspark =
            build_native_draft_slot(config, &provider, &model_root, device, eagle3_context)?;
        let qwen =
            QwenRuntime::from_plan(qwen_plan, &provider, &model_root, Arc::clone(&tokenizer));
        let target_reuse_identity = TargetReuseIdentity {
            model_inventory: PINNED_INVENTORY_SHA256,
            device,
            spine: program.representation(),
            expert_backend,
            expert_storage: experts.layout(),
            expert_cpu_threads,
            provider_abi: 1,
            transaction_contract: 1,
        };
        let readiness = target_readiness(
            device,
            program.representation(),
            expert_backend,
            experts.layout(),
        );

        let engine = Self {
            model_root,
            model,
            program,
            inventory,
            device,
            device_selection_policy: device_selection.policy,
            expert_backend,
            expert_cpu_threads,
            ngram_drafter: NgramDraftSource::default(),
            dspark,
            qwen,
            speculative_max_drafts,
            complete_expert_union,
            complete_expert_union_reserved_capacity,
            metal_expert_wrapper_retention,
            expert_early_drain: config.expert_early_drain,
            tokenizer,
            output_decoder,
            embedding,
            embedding_arena,
            experts,
            expert_reader,
            expert_prefetch_reader,
            storage_homes,
            pilot_gate,
            expert_heat,
            expert_pin_tier,
            router_trace,
            global_plans,
            spine,
            provider,
            spine_pipeline,
            residency,
            transient_layer_bytes,
            context_growth,
            committed_context_tokens: 0,
            mla_capacity_tokens: 0,
            pending_context_envelope: None,
            verify_snapshots,
            slot_verify_reserved_bytes,
            metal_source_selector,
            reasoning_effort: config.reasoning_effort.clone(),
            readiness,
            lifecycle: NativeEngineLifecycle::Ready,
            stream_boundary: NativeStreamBoundary::None,
            target_state_capability,
            target_reuse_identity,
            published_target_boundary: None,
            pending_target_publication: None,
            next_target_boundary_id: 1,
        };
        engine.validate_owned_state()?;
        Ok(engine)
    }

    /// Snapshot of the PILOT gate's scored recall and admission telemetry.
    /// `None` when speculative prefetch is uninstalled or `K3_PILOT_GATE=off`.
    pub fn pilot_gate_report(&self) -> Option<PilotGateReport> {
        self.pilot_gate.as_ref().map(PilotGate::report)
    }

    pub fn status(&self) -> NativeEngineStatus<'_> {
        NativeEngineStatus {
            model_root: &self.model_root,
            device: self.device,
            device_selection_policy: self.device_selection_policy,
            expert_backend: self.expert_backend,
            expert_storage: self.experts.layout(),
            libtorch_version: &self.inventory.libtorch_version,
            spine_source: self.spine.source(),
            spine_representation: self.program.representation(),
            spine_layers: self.spine.layers().len(),
            global_transfer_groups: self.global_plans.len(),
            resident_prefix_layers: self.residency.resident_layers,
            resident_prefix_bytes: self.residency.resident_provider_bytes,
            transient_layer_bytes: self.transient_layer_bytes,
            context_growth: self.context_growth,
            verify_snapshots: self.verify_snapshots,
            spine_reader_workers: self.spine_pipeline.workers(),
            spine_fd_cache_descriptors: self.spine.persistent_source_count(),
            spine_fd_cache_opened: self.spine.opened_persistent_source_count(),
            expert_reader_workers: self.expert_reader.workers(),
            expert_prefetch_reader_workers: self
                .expert_prefetch_reader
                .as_ref()
                .map_or(0, Reader::workers),
            router_trace_enabled: self.router_trace.enabled(),
            expert_heat_recording: self.expert_heat.recording(),
            expert_heat_weight: self.expert_heat.live_weight(),
            expert_pin_candidates: self
                .expert_pin_tier
                .as_ref()
                .map_or(0, ExpertPinTier::candidate_count),
            expert_pin_charged_bytes: self
                .expert_pin_tier
                .as_ref()
                .map_or(0, ExpertPinTier::charged_host_bytes),
            expert_pin_resident_experts: self
                .expert_pin_tier
                .as_ref()
                .map_or(0, ExpertPinTier::resident_experts),
            expert_pin_resident_bytes: self
                .expert_pin_tier
                .as_ref()
                .map_or(0, ExpertPinTier::resident_bytes),
            expert_pin_hits: self.expert_pin_tier.as_ref().map_or(0, ExpertPinTier::hits),
            expert_pin_served_bytes: self.expert_pin_tier.as_ref().map_or(0, |tier| {
                tier.hits().saturating_mul(tier.span_bytes() as u64)
            }),
            expert_cpu_threads: self.expert_cpu_threads,
            speculative_max_drafts: self.speculative_max_drafts,
            qwen_state: self.qwen.state,
            qwen_reserved_provider_bytes: self.qwen.reserved_provider_bytes,
            qwen_reserved_verify_bytes: self.qwen.reserved_verify_bytes,
            qwen_reserved_verify_positions: self.qwen.reserved_verify_positions,
            qwen_context_capacity: self.qwen.context_capacity,
            qwen_telemetry: self.qwen.telemetry,
            qwen_last_error: self.qwen.last_error.as_deref(),
            lazy_expert_files_missing: self.experts.lazy_missing_files(),
            readiness: self.readiness,
        }
    }

    fn validate_owned_state(&self) -> Result<()> {
        if self.provider.device() != self.device {
            return Err(DeltafinError::new(
                "native provider session device differs from the selected device",
            ));
        }
        if self.model.num_hidden_layers != self.program.layers.len()
            || self.program.layers.len() != self.spine.layers().len()
        {
            return Err(DeltafinError::new(
                "native engine model, program, and spine layer counts differ",
            ));
        }
        if self.tokenizer.vocab_size() != self.model.vocab_size {
            return Err(DeltafinError::new(
                "native tokenizer vocabulary differs from the target model",
            ));
        }
        if self.embedding.spec().rows() as usize != self.model.vocab_size
            || self.embedding.spec().columns() as usize != self.model.hidden_size
            || self.embedding_arena.max_rows() != EMBEDDING_ARENA_ROWS
        {
            return Err(DeltafinError::new(
                "native embedding owner differs from the target model contract",
            ));
        }
        if self.experts.source_count() != crate::experts::K3_EXPERT_RAW_FILES {
            return Err(DeltafinError::new("native expert namespace is incomplete"));
        }
        if self.experts.layout() == ExpertStorageLayout::Scale4V2
            && (self.expert_backend != ResolvedExpertBackend::Metal
                || self.metal_source_selector.is_none())
        {
            return Err(DeltafinError::new(
                "native scale4-v2 expert corpus lacks its qualified Metal backend",
            ));
        }
        if self.expert_pin_tier.is_some()
            && !matches!(
                self.expert_backend,
                ResolvedExpertBackend::Cpu | ResolvedExpertBackend::Metal
            )
        {
            return Err(DeltafinError::new(
                "the learned expert pin tier is limited to the scattered-span backends",
            ));
        }
        let prefetch_expected = self.experts.lazy_missing_files() == 0
            && matches!(
                self.expert_backend,
                ResolvedExpertBackend::Cpu | ResolvedExpertBackend::Metal
            );
        if self.expert_prefetch_reader.is_some() != prefetch_expected
            || self
                .expert_prefetch_reader
                .as_ref()
                .is_some_and(|reader| reader.workers() != self.expert_reader.workers())
        {
            return Err(DeltafinError::new(
                "native expert-prefetch owner differs from its fail-closed device/install policy",
            ));
        }
        if self.pilot_gate.is_some() && self.expert_prefetch_reader.is_none() {
            return Err(DeltafinError::new(
                "native PILOT gate exists without the speculative reader it governs",
            ));
        }
        if self.global_plans.len() != crate::program::K3_GLOBAL_TRANSFER_GROUPS {
            return Err(DeltafinError::new(
                "native global transfer plan count differs from the target roster",
            ));
        }
        if self.spine_pipeline.resident_prefix_target() as usize != self.residency.resident_layers {
            return Err(DeltafinError::new(
                "native spine pipeline differs from the selected resident prefix",
            ));
        }
        if self.qwen.is_available() {
            let (positions, bytes) =
                qwen_verify_reserve(self.verify_snapshots, self.speculative_max_drafts)?;
            if self.qwen.reserved_provider_bytes == 0
                || self.qwen.reserved_verify_positions != positions
                || self.qwen.reserved_verify_bytes != bytes
            {
                return Err(DeltafinError::new(
                    "native Qwen source lacks its exact target-verifier startup reserve",
                ));
            }
        }
        if self.expert_cpu_threads == 0
            || self.expert_cpu_threads > MAX_NATIVE_CPU_EXPERT_THREADS
            || self.speculative_max_drafts == 0
            || self.speculative_max_drafts > MAX_EXACT_DRAFTS
            || self.committed_context_tokens != 0
            || self.mla_capacity_tokens != 0
            || self.lifecycle != NativeEngineLifecycle::Ready
            || self.stream_boundary != NativeStreamBoundary::None
            || self.target_state_capability != self.provider.target_state_transaction_capability()
            || self.published_target_boundary.is_some()
            || self.pending_target_publication.is_some()
            || self.next_target_boundary_id == 0
        {
            return Err(DeltafinError::new(
                "native engine request lifecycle or CPU expert policy is invalid at bootstrap",
            ));
        }
        // These allocation-free observations also assert that both persistent
        // stream state owners were constructed, not temporary probes.
        let _ = self.output_decoder.pending_bytes();
        let _ = self.embedding_arena.allocated_bytes();
        Ok(())
    }

    fn encode_prompt(&self, config: &RuntimeConfig) -> Result<Vec<u32>> {
        if !config.chat {
            return self.tokenizer.encode_ordinary(&config.prompt);
        }
        let mut message = Map::new();
        message.insert("role".into(), Value::String("user".into()));
        message.insert("content".into(), Value::String(config.prompt.clone()));
        let mut options = ChatOptions::default();
        if let Some(effort) = config.reasoning_effort.as_deref() {
            options.thinking_effort = Some(effort);
        }
        encode_chat(&self.tokenizer, &[Value::Object(message)], &options)
    }

    fn reset_for_fresh_request(&mut self) -> Result<()> {
        if self.committed_context_tokens == 0 && self.mla_capacity_tokens == 0 {
            return Ok(());
        }
        if self.spine_pipeline.pending_layer().is_some() || self.output_decoder.pending_bytes() != 0
        {
            return Err(DeltafinError::new(
                "native engine cannot reset while unpublished I/O or UTF-8 state remains",
            ));
        }
        self.provider.reset_target_state()?;
        self.committed_context_tokens = 0;
        self.mla_capacity_tokens = 0;
        Ok(())
    }

    fn reset_target_reuse_to_fresh(&mut self) -> Result<()> {
        self.published_target_boundary = None;
        if self.pending_target_publication.is_some() {
            return Err(DeltafinError::new(
                "native target reuse cannot reset with an unresolved publication",
            ));
        }
        self.reset_for_fresh_request()
    }

    fn begin_target_reuse_request(
        &mut self,
        prompt: &[u32],
        reusable: bool,
    ) -> Result<(usize, Option<TargetStateBranch>, TargetCache)> {
        let (published, plan) = consume_target_reuse_slot(
            &mut self.published_target_boundary,
            self.target_state_capability,
            self.target_reuse_identity,
            prompt,
        );
        if reusable {
            if let (
                Some(boundary),
                TargetReusePlan::Reuse {
                    expected_committed_positions,
                    expected_cache_generation,
                    replay,
                },
            ) = (published.as_ref(), plan)
            {
                let expected = TargetStateBoundary {
                    committed_positions: expected_committed_positions as u64,
                    cache_generation: expected_cache_generation,
                };
                let live = self.provider.inspect_target_state();
                if live.as_ref() == Ok(&expected) {
                    if let Ok(branch) = self.provider.begin_target_state_branch(expected) {
                        self.committed_context_tokens = expected.committed_positions;
                        // The provider may retain spare geometric capacity. A
                        // logical lower bound is conservative for live growth
                        // admission and never authorizes extra context.
                        self.mla_capacity_tokens = expected.committed_positions;
                        return Ok((
                            replay.start,
                            Some(branch),
                            TargetCache::Hit {
                                cached_tokens: expected_committed_positions,
                                boundary_id: boundary.boundary_id.clone(),
                            },
                        ));
                    }
                }
            }
        }
        // The one retained slot is consumed on every miss, truncation,
        // divergence, identity change, raw completion, or provider mismatch.
        // No longest-common-prefix rollback is attempted for recurrent KDA.
        self.reset_target_reuse_to_fresh()?;
        Ok((0, None, TargetCache::Miss))
    }

    fn next_target_boundary(&mut self) -> Result<BoundaryId> {
        if self.next_target_boundary_id == u64::MAX {
            return Err(DeltafinError::new(
                "native target boundary identifier space is exhausted",
            ));
        }
        let identifier = BoundaryId::numeric(self.next_target_boundary_id);
        self.next_target_boundary_id += 1;
        Ok(identifier)
    }

    fn publish_prompt_parent_and_begin_decode_branch(
        &mut self,
        prompt: &[u32],
        pending_token: u32,
        prefill_branch: Option<TargetStateBranch>,
    ) -> Result<(PublishedTargetBoundary, TargetStateBranch)> {
        let state = if let Some(branch) = prefill_branch {
            self.provider.publish_target_state_branch(branch)?
        } else {
            self.provider.inspect_target_state()?
        };
        if state.committed_positions != prompt.len() as u64 {
            return Err(DeltafinError::new(
                "native prompt boundary differs from its full-K3 committed prefix",
            ));
        }
        let mut logical_tokens = Vec::with_capacity(prompt.len().saturating_add(1));
        logical_tokens.extend_from_slice(prompt);
        logical_tokens.push(pending_token);
        let prompt_boundary = PublishedTargetBoundary {
            identity: self.target_reuse_identity,
            logical_tokens: logical_tokens.into_boxed_slice(),
            committed_positions: prompt.len(),
            cache_generation: state.cache_generation,
            pending_token,
            boundary_id: self.next_target_boundary()?,
        };
        let branch = self.provider.begin_target_state_branch(state)?;
        Ok((prompt_boundary, branch))
    }

    fn settle_target_publication(&mut self, publication: StreamPublication) {
        let Some(pending) = self.pending_target_publication.take() else {
            return;
        };
        let use_final =
            publication == StreamPublication::Complete && pending.final_tokens.is_some();
        let target = if use_final {
            self.provider
                .publish_target_state_branch(pending.branch)
                .ok()
                .and_then(|state| {
                    let tokens = pending.final_tokens?;
                    if state.committed_positions as usize + 1 != tokens.len() {
                        return None;
                    }
                    let pending_token = *tokens.last()?;
                    Some(PublishedTargetBoundary {
                        identity: self.target_reuse_identity,
                        logical_tokens: tokens,
                        committed_positions: state.committed_positions as usize,
                        cache_generation: state.cache_generation,
                        pending_token,
                        boundary_id: self.next_target_boundary().ok()?,
                    })
                })
        } else {
            self.provider
                .discard_target_state_branch(pending.branch)
                .ok()
                .and_then(|state| {
                    (state.committed_positions as usize == pending.prompt.committed_positions)
                        .then_some(PublishedTargetBoundary {
                            cache_generation: state.cache_generation,
                            ..pending.prompt
                        })
                })
        };

        let Some(target) = target.filter(|boundary| {
            boundary.committed_positions >= SERVER_TARGET_REUSE_MIN_TOKENS
                && boundary.committed_positions
                    <= self.context_growth.admitted_expanded_context_tokens as usize
        }) else {
            self.published_target_boundary = None;
            if let Some(lease) = pending.dspark_lease.as_ref() {
                let _ = self.dspark.abort_request(lease);
            }
            if self.provider.reset_target_state().is_ok() {
                self.committed_context_tokens = 0;
                self.mla_capacity_tokens = 0;
            } else {
                // Publication cleanup is optional for the completed response,
                // but an unprovable provider branch must never serve another.
                self.lifecycle.poison();
            }
            return;
        };

        // Target publication is authoritative and happens first. DSpark may
        // pair only the exact committed input tape and boundary identifier;
        // any optional failure drops/reset draft state while retaining target.
        self.committed_context_tokens = target.committed_positions as u64;
        self.mla_capacity_tokens = self.committed_context_tokens;
        self.published_target_boundary = Some(target.clone());
        if let Some(lease) = pending.dspark_lease.as_ref() {
            let staged: Option<BoundaryStageToken> = if use_final {
                self.dspark
                    .stage_final_boundary(
                        lease,
                        &target.logical_tokens[..target.committed_positions],
                        target.boundary_id.clone(),
                    )
                    .ok()
                    .flatten()
            } else {
                self.dspark
                    .stage_prompt_fallback(lease, target.boundary_id.clone())
                    .ok()
                    .flatten()
            };
            if let Some(stage) = staged {
                if self.dspark.commit_staged(stage).is_ok() {
                    return;
                }
            }
            let _ = self.dspark.abort_staged(None);
            let _ = self.dspark.abort_request(lease);
        }
    }

    fn execute_target_chunk(
        &mut self,
        token_ids: &[u32],
        collect_stats: bool,
        verbose_layer_profile: bool,
        collect_profile: bool,
        capture_dspark: bool,
    ) -> Result<CompletedTargetChunk> {
        let prepared = self.prepare_target_chunk(
            token_ids,
            TargetSequenceMode::Prefill,
            collect_stats,
            verbose_layer_profile,
            collect_profile,
            capture_dspark,
        )?;
        self.commit_target_chunk(prepared, token_ids.len())
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_target_chunk(
        &mut self,
        token_ids: &[u32],
        mode: TargetSequenceMode,
        collect_stats: bool,
        verbose_layer_profile: bool,
        collect_profile: bool,
        capture_dspark: bool,
    ) -> Result<PreparedTargetChunk> {
        self.prepare_target_chunk_with_commit_policy(
            token_ids,
            mode,
            collect_stats,
            verbose_layer_profile,
            collect_profile,
            capture_dspark,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_full_commit_verify_chunk(
        &mut self,
        token_ids: &[u32],
        collect_stats: bool,
        verbose_layer_profile: bool,
        collect_profile: bool,
        capture_dspark: bool,
    ) -> Result<PreparedTargetChunk> {
        self.prepare_target_chunk_with_commit_policy(
            token_ids,
            TargetSequenceMode::Verify,
            collect_stats,
            verbose_layer_profile,
            collect_profile,
            capture_dspark,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_target_chunk_with_commit_policy(
        &mut self,
        token_ids: &[u32],
        mode: TargetSequenceMode,
        collect_stats: bool,
        verbose_layer_profile: bool,
        collect_profile: bool,
        capture_dspark: bool,
        full_commit_only: bool,
    ) -> Result<PreparedTargetChunk> {
        if token_ids.is_empty() || token_ids.len() > TARGET_SEQUENCE_MAX_POSITIONS as usize {
            return Err(DeltafinError::new(format!(
                "native target chunks need 1..={TARGET_SEQUENCE_MAX_POSITIONS} positions"
            )));
        }
        let staged_session_positions = self
            .committed_context_tokens
            .checked_add(token_ids.len() as u64)
            .ok_or_else(|| DeltafinError::new("native committed context length overflows u64"))?;
        // The first chunk of a bounded request proves the whole request's
        // context envelope instead of just its own rows: every later growth
        // boundary then passes on the committed-capacity fast path with no
        // live memory proof, so pressure arriving mid-run cannot fail a
        // request that was admissible when it began. Declining is ordinary —
        // the impatient proof neither trims nor waits — and falls straight
        // back to the incremental growth this call always performed.
        let envelope = self
            .pending_context_envelope
            .take()
            .filter(|&tokens| tokens > staged_session_positions);
        let next_mla_capacity = match envelope {
            Some(tokens) => {
                let prior_capacity = self.mla_capacity_tokens;
                match self.admit_context_growth(tokens, false) {
                    Ok(capacity) => {
                        if capacity > prior_capacity {
                            eprintln!(
                                "[native] pre-admitted this request's {tokens}-token context envelope (exact MLA capacity {capacity} tokens; later growth in this request needs no mid-run memory proof)"
                            );
                        }
                        capacity
                    }
                    Err(declined) => {
                        eprintln!(
                            "[native] context envelope pre-admission ({tokens} tokens) declined; keeping incremental growth: {declined}"
                        );
                        self.admit_context_chunk(staged_session_positions)?
                    }
                }
            }
            None => self.admit_context_chunk(staged_session_positions)?,
        };
        let capture = capture_dspark
            .then(|| self.dspark.backend().map(NativeDraftBackend::capture))
            .flatten();
        let mut sequence = {
            let embedding = self
                .embedding
                .read_rows(token_ids, &mut self.embedding_arena)?;
            if full_commit_only {
                if mode != TargetSequenceMode::Verify {
                    return Err(DeltafinError::new(
                        "full-commit-only target chunks are valid only for exact verification",
                    ));
                }
                self.provider
                    .begin_target_sequence_bf16_verify_full_commit_only(
                        embedding.bytes(),
                        token_ids.len(),
                        capture,
                    )?
            } else if let Some(capture) = capture {
                self.provider.begin_target_sequence_bf16_capturing(
                    embedding.bytes(),
                    token_ids.len(),
                    mode,
                    capture,
                )?
            } else {
                self.provider.begin_target_sequence_bf16(
                    embedding.bytes(),
                    token_ids.len(),
                    mode,
                )?
            }
        };
        let metal_source_selector = self.metal_source_selector.as_deref();
        let complete_expert_union = prepare_complete_expert_union(
            self.complete_expert_union,
            self.complete_expert_union_reserved_capacity,
            mode,
            full_commit_only,
            token_ids.len(),
            target_expert_backend(self.expert_backend),
            &self.expert_reader,
            self.experts.layout(),
        );
        let result = execute_target_sequence(
            &mut sequence,
            &self.provider,
            &mut self.spine_pipeline,
            self.spine.layers(),
            &self.experts,
            &self.expert_reader,
            self.expert_prefetch_reader.as_ref(),
            self.pilot_gate.as_mut(),
            &self.expert_heat,
            self.expert_pin_tier.as_ref(),
            &mut self.router_trace,
            target_expert_backend(self.expert_backend),
            self.expert_cpu_threads,
            metal_source_selector,
            self.metal_expert_wrapper_retention,
            self.expert_early_drain,
            complete_expert_union,
            collect_stats,
            verbose_layer_profile,
            collect_profile,
        );
        let (predictions, stats, profile) = match result {
            Ok(result) => result,
            Err(error) => return Err(cancel_after_error(sequence, error)),
        };
        // One completed 93-layer pass of authoritative routes is the heat
        // histogram's weight unit; the periodic merge stays off the per-layer
        // path and out of the sequence transaction entirely.
        if self.expert_heat.note_pass_committed() {
            self.expert_heat.flush_best_effort();
        }
        let expected_predictions = match mode {
            TargetSequenceMode::Prefill => 1,
            TargetSequenceMode::Verify => token_ids.len(),
        };
        if predictions.len() != expected_predictions {
            return Err(cancel_after_error(
                sequence,
                DeltafinError::new(
                    "native target returned a prediction count inconsistent with its sequence mode",
                ),
            ));
        }
        let dspark_rows = capture
            .is_some()
            .then(|| sequence.dspark_target_rows().ok())
            .flatten();
        Ok(PreparedTargetChunk {
            sequence,
            predictions,
            stats,
            profile,
            dspark_rows,
            dspark_capture_requested: capture_dspark,
            input_positions: token_ids.len(),
            next_mla_capacity,
            mode,
        })
    }

    fn commit_target_chunk(
        &mut self,
        prepared: PreparedTargetChunk,
        positions: usize,
    ) -> Result<CompletedTargetChunk> {
        if positions == 0
            || positions > prepared.input_positions
            || (prepared.mode == TargetSequenceMode::Prefill
                && positions != prepared.input_positions)
        {
            return Err(DeltafinError::new(
                "native target commit prefix is invalid for its prepared sequence",
            ));
        }
        let expected_session_positions = self
            .committed_context_tokens
            .checked_add(positions as u64)
            .ok_or_else(|| DeltafinError::new("native committed context length overflows u64"))?;
        // `commit_prefix` consumes the sequence. Its implementation retains
        // the RAII handle until native commit succeeds, so a failed commit
        // still cancels the unpublished cache generation while unwinding.
        let commit = if positions == prepared.input_positions {
            prepared.sequence.commit_all()?
        } else {
            prepared.sequence.commit_prefix(positions)?
        };
        if commit.committed_positions != positions as u64
            || commit.session_committed_positions != expected_session_positions
        {
            return Err(DeltafinError::new(
                "native target committed a different context length than its exact input chunk",
            ));
        }
        self.committed_context_tokens = commit.session_committed_positions;
        self.mla_capacity_tokens = prepared.next_mla_capacity;
        Ok(CompletedTargetChunk {
            predictions: prepared.predictions,
            stats: prepared.stats,
            profile: prepared.profile,
            commit,
            dspark_rows: prepared.dspark_rows,
            dspark_capture_requested: prepared.dspark_capture_requested,
        })
    }

    fn admit_context_chunk(&self, needed_tokens: u64) -> Result<u64> {
        self.admit_context_growth(needed_tokens, true)
    }

    /// `patient = false` performs at most two live proofs (one before and one
    /// after shedding this process's own advisory storage) and never trims,
    /// sleeps, or waits — for opportunistic envelope pre-admission, where a
    /// refusal is an ordinary fallback to incremental growth rather than a
    /// failure worth stalling a request over.
    fn admit_context_growth(&self, needed_tokens: u64, patient: bool) -> Result<u64> {
        if needed_tokens <= self.committed_context_tokens
            || self.committed_context_tokens > self.mla_capacity_tokens
            || needed_tokens > self.context_growth.admitted_expanded_context_tokens
        {
            return Err(DeltafinError::new(
                "native context request is outside its committed exact MLA state",
            ));
        }
        if needed_tokens <= self.mla_capacity_tokens {
            return Ok(self.mla_capacity_tokens);
        }
        let admission = self
            .context_growth
            .admission(self.mla_capacity_tokens, needed_tokens)?;
        let before = self.provider.memory_snapshot(false)?;
        let first =
            admit_live_context_growth(probe_host_memory(), provider_memory(before), admission);
        let Err(mut last_error) = first else {
            return Ok(admission.next_capacity_tokens);
        };
        // Before treating this as someone else's memory pressure, shed our
        // own advisory storage: the speculative-prefetch arena retains up to
        // ~1.1 GiB of lazily grown slabs that exist only to accelerate
        // optional reads, and measured admission margins here were 0-340 MiB.
        // Every recorded admission failure was the full-speculation arm; the
        // suppressed arm, which materializes no slabs, never failed once.
        // The slabs re-grow lazily on the next speculative read, so exact
        // context growth must always outrank them.
        if let Some(reader) = self.expert_prefetch_reader.as_ref() {
            match reader.release_cached_slabs() {
                Ok(0) => {}
                Ok(released) => {
                    eprintln!(
                        "[native] released {:.2} MiB of speculative prefetch slabs ahead of exact context growth {}->{} tokens",
                        released as f64 / (1_u64 << 20) as f64,
                        admission.committed_capacity_tokens,
                        admission.next_capacity_tokens,
                    );
                    match admit_live_context_growth(
                        probe_host_memory(),
                        provider_memory(self.provider.memory_snapshot(false)?),
                        admission,
                    ) {
                        Ok(()) => return Ok(admission.next_capacity_tokens),
                        Err(still_refused) => last_error = still_refused,
                    }
                }
                Err(error) => eprintln!(
                    "[native] speculative prefetch slabs stayed resident during context growth: {error}"
                ),
            }
        }
        if !patient {
            return Err(last_error);
        }
        // Live pressure at a growth boundary is often another process's
        // transient spike, not this process's fault. Shed what is optional
        // (unused accelerator cache), then wait bounded intervals for the
        // host snapshot to recover before failing closed. Generation pauses
        // here rather than dying; committed exact state is untouched while
        // waiting, and the policy refusing unsafe growth stays authoritative.
        //
        // K3_MEMORY_PATIENCE_SECONDS extends the wait for hosts that are
        // deliberately shared with interactive work: generation stalls at the
        // boundary until headroom returns instead of failing the run. The
        // window stays bounded (24h cap) so an abandoned process still exits.
        const PRESSURE_RETRY_DELAY: Duration = Duration::from_secs(15);
        const PRESSURE_HEARTBEAT_EVERY: u32 = 8;
        let patience_seconds = std::env::var("K3_MEMORY_PATIENCE_SECONDS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .map(|value| value.clamp(15, 86_400))
            .unwrap_or(105);
        let pressure_retries =
            u32::try_from((patience_seconds / PRESSURE_RETRY_DELAY.as_secs()).max(1))
                .unwrap_or(u32::MAX);
        let mut last_reserved: Option<(Option<u64>, Option<u64>)> = None;
        for attempt in 1..=pressure_retries {
            let trim = self.device != Device::Cpu;
            let before_retry = self.provider.memory_snapshot(false)?;
            let after = if trim {
                self.provider.memory_snapshot(true).map_err(|trim_error| {
                    DeltafinError::new(format!(
                        "{last_error}; native accelerator cache recovery failed: {trim_error}"
                    ))
                })?
            } else {
                before_retry
            };
            let host = if trim {
                account_trimmed_unified_memory(probe_host_memory(), before_retry, after)
            } else {
                probe_host_memory()
            };
            last_reserved = Some((before_retry.reserved_bytes, after.reserved_bytes));
            match admit_live_context_growth(host, provider_memory(after), admission) {
                Ok(()) => {
                    if attempt > 1 {
                        eprintln!(
                            "[native] live memory pressure cleared after {} deferred attempt(s); resuming exact context growth {}->{} tokens",
                            attempt - 1,
                            admission.committed_capacity_tokens,
                            admission.next_capacity_tokens,
                        );
                    }
                    return Ok(admission.next_capacity_tokens);
                }
                Err(retry_error) => {
                    last_error = retry_error;
                    if attempt == pressure_retries {
                        break;
                    }
                    // Log the first deferral, then heartbeat every ~2 minutes
                    // so an hour-long stall is visible without log spam.
                    if attempt == 1 || attempt % PRESSURE_HEARTBEAT_EVERY == 0 {
                        eprintln!(
                            "[native] live memory pressure: deferring exact context growth {}->{} tokens (waited {}s of up to {}s; retrying every {}s)",
                            admission.committed_capacity_tokens,
                            admission.next_capacity_tokens,
                            u64::from(attempt - 1) * PRESSURE_RETRY_DELAY.as_secs(),
                            patience_seconds,
                            PRESSURE_RETRY_DELAY.as_secs(),
                        );
                    }
                    std::thread::sleep(PRESSURE_RETRY_DELAY);
                }
            }
        }
        let (reserved_before, reserved_after) = last_reserved.unwrap_or((None, None));
        Err(DeltafinError::new(format!(
            "{last_error}; admission still failed after releasing unused accelerator cache (reserved before={}, after={}) and waiting {}s for live memory pressure to clear",
            optional_mib(reserved_before),
            optional_mib(reserved_after),
            u64::from(pressure_retries - 1) * PRESSURE_RETRY_DELAY.as_secs(),
        )))
    }

    fn admit_verify_width(&self, positions: usize, full_commit_only: bool) -> Result<bool> {
        let positions = u64::try_from(positions)
            .map_err(|_| DeltafinError::new("target verify width does not fit u64"))?;
        let admission = self.verify_snapshots.admission(positions)?;
        // Qwen enters the provider's reviewed full-commit-only transaction.
        // That transaction deliberately retains no per-row KDA boundaries:
        // it stages exactly one final generation and reruns the accepted
        // prefix after a mismatch. The ordinary committed+staged fixed cost
        // already reserves that generation, so charging `positions - 1`
        // additional snapshots here would reject or displace useful resident
        // layers for storage the provider never allocates.
        if full_commit_only {
            return Ok(true);
        }
        // Subtract only verifier bytes already withheld by the selected
        // startup plan: the slot drafter's prefix record, when one is
        // loaded. Qwen and the slot never verify at once, so the larger
        // reserve covers either. Anything beyond it (an n-gram verify with
        // no slot drafter) is charged against a fresh live snapshot.
        let unreserved_bytes = verify_live_admission_bytes(
            admission,
            self.qwen
                .reserved_verify_bytes
                .max(self.slot_verify_reserved_bytes),
        );
        if unreserved_bytes == 0 {
            return Ok(true);
        }
        let selection = select_resident_prefix(
            probe_host_memory(),
            provider_memory(self.provider.memory_snapshot(false)?),
            &[],
            FixedCosts {
                host_bytes: 0,
                provider_bytes: unreserved_bytes,
            },
            ResidencyOverride::default(),
            ResidencyPolicy::default(),
        );
        Ok(selection.stop == ResidencyStop::AllLayersFit)
    }

    fn validate_pending_target_lag(&self, decode: &DecodeArena, pending_token: u32) -> Result<()> {
        let logical_tokens = u64::try_from(decode.history().len())
            .map_err(|_| DeltafinError::new("native decode history length does not fit u64"))?;
        let expected_logical_tokens =
            self.committed_context_tokens
                .checked_add(1)
                .ok_or_else(|| {
                    DeltafinError::new("native target pending-token length overflows u64")
                })?;
        if logical_tokens != expected_logical_tokens
            || decode.history().last().copied() != Some(pending_token)
        {
            return Err(DeltafinError::new(format!(
                "native target/cache boundary lost its one-token lag (logical={logical_tokens}, provider={}, pending={pending_token})",
                self.committed_context_tokens,
            )));
        }
        Ok(())
    }

    fn dspark_tracks_rows(&self, lease: Option<&DraftLease>) -> bool {
        lease.is_some_and(|lease| self.dspark.tracks_target_rows(lease).unwrap_or(false))
    }

    fn advance_dspark_rows(
        &mut self,
        lease: Option<&DraftLease>,
        completed: &CompletedTargetChunk,
        committed_input_ids: &[u32],
    ) {
        let Some(lease) = lease else {
            return;
        };
        if !completed.dspark_capture_requested {
            return;
        }
        let advanced = completed.dspark_rows.as_ref().is_some_and(|rows| {
            self.dspark
                .commit_target_rows(lease, rows, committed_input_ids)
                .unwrap_or(false)
        });
        if !advanced {
            let _ = self.dspark.disable_proposals(
                lease,
                "provider-owned DSpark target capture/advance failed",
                Some(false),
            );
        }
    }

    fn write_token<W: Write>(&mut self, output: &mut W, token_id: u32) -> Result<()> {
        let text = self.output_decoder.push_token(&self.tokenizer, token_id)?;
        output
            .write_all(text.as_bytes())
            .map_err(|error| output_error("write generated token", error))?;
        output
            .flush()
            .map_err(|error| output_error("flush generated token", error))
    }

    fn finish_output<W: Write>(&mut self, output: &mut W) -> Result<()> {
        let trailing = self.output_decoder.finish();
        output
            .write_all(trailing.as_bytes())
            .map_err(|error| output_error("write final UTF-8 suffix", error))?;
        output
            .flush()
            .map_err(|error| output_error("flush generated output", error))
    }
}

fn resolve_expert_backend(
    request: ExpertBackendRequest,
    device: Device,
    inventory: ProviderInventory,
    cuda_moe_compiled: bool,
) -> Result<ResolvedExpertBackend> {
    match request {
        ExpertBackendRequest::Auto => match device {
            Device::Cpu => Ok(ResolvedExpertBackend::Cpu),
            Device::Mps => Ok(ResolvedExpertBackend::Metal),
            Device::Cuda(_) if cuda_moe_compiled => Ok(ResolvedExpertBackend::CudaAuto),
            Device::Cuda(_) => Ok(ResolvedExpertBackend::Cpu),
        },
        ExpertBackendRequest::Cpu => Ok(ResolvedExpertBackend::Cpu),
        ExpertBackendRequest::Metal if device == Device::Mps && inventory.mps => {
            Ok(ResolvedExpertBackend::Metal)
        }
        ExpertBackendRequest::Metal => Err(DeltafinError::new(
            "the Metal expert backend requires the selected target device to be an available MPS device",
        )),
        ExpertBackendRequest::Cuda if matches!(device, Device::Cuda(_)) && cuda_moe_compiled => {
            Ok(ResolvedExpertBackend::Cuda)
        }
        ExpertBackendRequest::Cuda if !cuda_moe_compiled => Err(DeltafinError::new(
            "the selected native provider was built without the CUDA MXFP4 expert adapter; install an NVCC-enabled build or use expert-backend auto/cpu",
        )),
        ExpertBackendRequest::Cuda => Err(DeltafinError::new(
            "the CUDA expert backend requires the selected target device to be CUDA",
        )),
    }
}

/// Context the EAGLE-3.1 drafter tracks: K3's admitted context, bounded.
fn eagle3_context(context_growth: ContextGrowthBudget) -> usize {
    usize::try_from(context_growth.admitted_expanded_context_tokens)
        .unwrap_or(usize::MAX)
        .min(eagle3_provider::MAXIMUM_CONTEXT)
}

/// Whether this run can use the proposal slot at all: chat and server
/// surfaces under automatic mode, any surface when it is forced on.
fn draft_slot_wanted(config: &RuntimeConfig, device: Device) -> bool {
    match config.dspark {
        DSparkRequest::Off => false,
        // Raw direct completions route automatically to Qwen, never the
        // slot. Loading a slot drafter there consumed roughly 6.8 GiB of
        // unified provider memory and displaced exact BF16 spine residency
        // even though no request could create a lease. Chat and server
        // surfaces retain the automatic path; explicit `on` remains an
        // override for raw experimentation.
        DSparkRequest::Auto => {
            (config.chat || config.surface == RuntimeSurface::Server)
                && !matches!(device, Device::Cpu)
        }
        DSparkRequest::On => true,
    }
}

/// EAGLE-3.1 takes the slot whenever it is installed and not switched off.
/// When it is selected DSpark is neither reserved nor loaded, so a broken
/// EAGLE install leaves the slot empty rather than over its reservation.
fn eagle3_selected(config: &RuntimeConfig, model_root: &Path, device: Device) -> bool {
    draft_slot_wanted(config, device)
        && match config.eagle3 {
            Eagle3Request::Off => false,
            Eagle3Request::Auto => model_root.join(eagle3_provider::DIRECTORY).is_dir(),
            Eagle3Request::On => true,
        }
}

fn eagle3_runtime_config(config: &RuntimeConfig, context: usize) -> DSparkRuntimeConfig {
    DSparkRuntimeConfig {
        vocab_size: 163_840,
        // Measured on real K3 states: 3.26 tokens per pass at three drafts,
        // 4.17 at five and 4.77 at seven, first draft right 89% of the time.
        probe_drafts: 3,
        max_drafts: eagle3_provider::MAXIMUM_DRAFTS,
        max_context_tokens: Some(context),
        min_auto_speedup: config.dspark_min_auto_speedup,
        disable_on_miss: false,
        economic_grace_steps: 4,
        loss_streak_limit: 4,
    }
}

fn build_native_draft_slot(
    config: &RuntimeConfig,
    provider: &NativeProviderSession,
    model_root: &Path,
    device: Device,
    eagle3_context: usize,
) -> Result<DSparkRuntime<NativeDraftBackend>> {
    if eagle3_selected(config, model_root, device) {
        let started = Instant::now();
        let loaded = (|| -> Result<NativeEagle3Backend> {
            let checkpoint =
                Eagle3Checkpoint::open(&model_root.join(eagle3_provider::DIRECTORY))?;
            NativeEagle3Backend::bind(provider, &checkpoint, model_root, device, eagle3_context)
        })();
        let backend = match (config.eagle3, loaded) {
            (_, Ok(backend)) => {
                eprintln!(
                    "[native] EAGLE-3.1 drafter loaded in {:.1}s ({eagle3_context}-token context)",
                    started.elapsed().as_secs_f64()
                );
                Some(NativeDraftBackend::Eagle3(backend))
            }
            (Eagle3Request::On, Err(error)) => return Err(error),
            (_, Err(error)) => {
                eprintln!(
                    "[native] optional EAGLE-3.1 drafter unavailable, drafting without it (K3_EAGLE3=off selects DSpark): {error}"
                );
                None
            }
        };
        return DSparkRuntime::new(
            config.dspark.runtime_mode(),
            backend,
            eagle3_runtime_config(config, eagle3_context),
        )
        .map_err(|error| DeltafinError::new(format!("configure native EAGLE-3 runtime: {error}")));
    }
    build_native_dspark(config, provider, model_root, device)
}

fn build_native_dspark(
    config: &RuntimeConfig,
    provider: &NativeProviderSession,
    model_root: &Path,
    device: Device,
) -> Result<DSparkRuntime<NativeDraftBackend>> {
    let runtime_config = DSparkRuntimeConfig {
        vocab_size: 163_840,
        probe_drafts: 2,
        max_drafts: 7,
        max_context_tokens: config.dspark_max_context,
        min_auto_speedup: config.dspark_min_auto_speedup,
        ..DSparkRuntimeConfig::default()
    };
    let directory = model_root.join("k3-draft-dspark");
    let eligible = dspark_eligible(config, &directory, device);
    let backend = if eligible {
        let loaded = (|| -> Result<NativeDSparkBackend> {
            let checkpoint_config = DSparkConfig::load_official(&directory)?;
            if checkpoint_config != DSparkConfig::OFFICIAL {
                return Err(DeltafinError::new(
                    "DSpark configuration differs from the pinned K3 proposal contract",
                ));
            }
            let checkpoint = DSparkCheckpoint::open_official(&directory)?;
            NativeDSparkBackend::bind(provider, &checkpoint, model_root, device)
        })();
        match (config.dspark, loaded) {
            (_, Ok(backend)) => Some(NativeDraftBackend::DSpark(backend)),
            (DSparkRequest::On, Err(error)) => return Err(error),
            (DSparkRequest::Auto, Err(error)) => {
                eprintln!("[native] optional DSpark unavailable: {error}");
                None
            }
            (DSparkRequest::Off, Err(_)) => None,
        }
    } else {
        None
    };
    DSparkRuntime::new(config.dspark.runtime_mode(), backend, runtime_config)
        .map_err(|error| DeltafinError::new(format!("configure native DSpark runtime: {error}")))
}

fn dspark_eligible(config: &RuntimeConfig, directory: &Path, device: Device) -> bool {
    draft_slot_wanted(config, device)
        && (config.dspark == DSparkRequest::On || directory.is_dir())
}

/// Provider memory the proposal slot holds for whichever model it loads.
fn draft_slot_provider_reserve(
    config: &RuntimeConfig,
    model_root: &Path,
    device: Device,
    eagle3_context: usize,
) -> Result<u64> {
    if eagle3_selected(config, model_root, device) {
        return eagle3_provider::provider_reserve(eagle3_context);
    }
    dspark_provider_reserve(config, model_root, device)
}

fn dspark_provider_reserve(
    config: &RuntimeConfig,
    model_root: &Path,
    device: Device,
) -> Result<u64> {
    if !dspark_eligible(config, &model_root.join("k3-draft-dspark"), device) {
        return Ok(0);
    }
    let shape = DSparkConfig::OFFICIAL;
    let embedding_parameters = shape
        .vocab_size
        .checked_mul(shape.hidden_size)
        .ok_or_else(|| DeltafinError::new("DSpark embedding size overflowed"))?;
    let owned_parameters = OFFICIAL_PARAMETER_COUNT
        .checked_sub(embedding_parameters)
        .ok_or_else(|| DeltafinError::new("DSpark owned parameter count underflowed"))?;
    let owned_bytes = owned_parameters
        .checked_mul(2)
        .ok_or_else(|| DeltafinError::new("DSpark owned storage overflowed"))?;
    let fused_context_bytes = shape
        .layers
        .checked_mul(shape.latent_cache_width())
        .and_then(|value| value.checked_mul(shape.hidden_size))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| DeltafinError::new("DSpark fused projection reserve overflowed"))?;
    let requested_context = config
        .dspark_max_context
        .map(|value| value as u64)
        .unwrap_or(shape.maximum_context)
        .min(shape.maximum_context);
    let cache_capacity = requested_context
        .max(8)
        .checked_next_power_of_two()
        .unwrap_or(shape.maximum_context)
        .min(shape.maximum_context);
    let cache_bytes = shape
        .layers
        .checked_mul(shape.latent_cache_width())
        .and_then(|value| value.checked_mul(cache_capacity))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| DeltafinError::new("DSpark cache snapshot reserve overflowed"))?;
    owned_bytes
        .checked_add(fused_context_bytes)
        .and_then(|value| value.checked_add(cache_bytes))
        .ok_or_else(|| DeltafinError::new("DSpark total provider reserve overflowed"))
}

fn qwen_context_capacity(context_growth: ContextGrowthBudget) -> Result<usize> {
    let target_capacity = usize::try_from(context_growth.admitted_expanded_context_tokens)
        .map_err(|_| DeltafinError::new("target context capacity exceeds usize"))?;
    let desired = target_capacity
        .checked_add(NativeQwen::maximum_new_tokens())
        .ok_or_else(|| DeltafinError::new("Qwen context capacity overflows usize"))?;
    let model_capacity = usize::try_from(QwenVariant::Probe06B.architecture().maximum_position)
        .map_err(|_| DeltafinError::new("Qwen model context capacity exceeds usize"))?;
    let admitted = desired.min(model_capacity);
    if admitted == 0 {
        return Err(DeltafinError::new("Qwen context capacity cannot be empty"));
    }
    Ok(admitted)
}

fn discover_qwen_plan(
    request: QwenRequest,
    model_root: &Path,
    device: Device,
    context_capacity: usize,
    verify_snapshots: VerifySnapshotBudget,
    speculative_max_drafts: usize,
) -> QwenPlan {
    if request == QwenRequest::Off {
        return QwenPlan::inactive(QwenRuntimeState::Off);
    }
    if request == QwenRequest::Auto && !matches!(device, Device::Mps) {
        return QwenPlan::inactive(QwenRuntimeState::IneligibleDevice);
    }
    let probe = QwenCheckpoint::open(model_root, QwenVariant::Probe06B).is_ok();
    let wide = QwenCheckpoint::open(model_root, QwenVariant::Wide17B).is_ok();
    let Ok((reserved_verify_positions, reserved_verify_bytes)) =
        qwen_verify_reserve(verify_snapshots, speculative_max_drafts)
    else {
        return QwenPlan::inactive(QwenRuntimeState::MemoryRejected);
    };
    qwen_candidate_plan(
        device,
        probe,
        wide,
        context_capacity,
        reserved_verify_positions,
        reserved_verify_bytes,
    )
}

fn qwen_candidate_plan(
    device: Device,
    probe: bool,
    wide: bool,
    context_capacity: usize,
    reserved_verify_positions: u64,
    reserved_verify_bytes: u64,
) -> QwenPlan {
    let initial = if probe {
        Some(QwenVariant::Probe06B)
    } else if wide {
        Some(QwenVariant::Wide17B)
    } else {
        None
    };
    let Some(initial) = initial else {
        return QwenPlan::inactive(QwenRuntimeState::NotInstalled);
    };
    let variants = if probe && wide {
        &[QwenVariant::Probe06B, QwenVariant::Wide17B][..]
    } else {
        std::slice::from_ref(&initial)
    };
    let reserved_provider_bytes = match qwen_provider_reserve(device, variants, context_capacity) {
        Ok(bytes) => bytes,
        Err(_) => return QwenPlan::inactive(QwenRuntimeState::MemoryRejected),
    };
    QwenPlan {
        state: qwen_variant_state(initial),
        initial: Some(initial),
        wide_lazy: probe && wide,
        reserved_provider_bytes,
        reserved_verify_bytes,
        reserved_verify_positions,
        context_capacity,
    }
}

fn qwen_probe_only_fallback(plan: QwenPlan, device: Device) -> Option<QwenPlan> {
    (plan.initial == Some(QwenVariant::Probe06B) && plan.wide_lazy)
        .then(|| {
            qwen_candidate_plan(
                device,
                true,
                false,
                plan.context_capacity,
                plan.reserved_verify_positions,
                plan.reserved_verify_bytes,
            )
        })
        .filter(|fallback| fallback.reserved_provider_bytes != 0)
}

fn qwen_verify_reserve(
    verify_snapshots: VerifySnapshotBudget,
    speculative_max_drafts: usize,
) -> Result<(u64, u64)> {
    let positions = u64::try_from(speculative_max_drafts)
        .ok()
        .and_then(|drafts| drafts.checked_add(1))
        .ok_or_else(|| DeltafinError::new("Qwen verifier width overflows u64"))?;
    // Validate the bounded width against the target-sequence contract, but do
    // not reserve ordinary per-row verifier snapshots. Qwen is always routed
    // through `prepare_full_commit_verify_chunk`: the compiled provider keeps
    // one final KDA generation and reports zero `verify_snapshot_bytes` for
    // that mode. The ordinary decode fixed cost already includes one staged
    // generation. A mismatch cancels it and reruns only the accepted prefix.
    let _ = verify_snapshots.admission(positions)?;
    Ok((positions, 0))
}

/// Prefix-record bytes the widest ordinary slot verify (all configured
/// drafts plus the pending row) needs beyond the decode reserve.
fn slot_verify_reserve(
    verify_snapshots: VerifySnapshotBudget,
    speculative_max_drafts: usize,
) -> Result<u64> {
    let positions = u64::try_from(speculative_max_drafts)
        .ok()
        .and_then(|drafts| drafts.checked_add(1))
        .ok_or_else(|| DeltafinError::new("slot verifier width overflows u64"))?;
    Ok(verify_snapshots
        .admission(positions)?
        .additional_over_decode_reserve_bytes)
}

fn verify_live_admission_bytes(
    admission: VerifySnapshotAdmission,
    startup_reserved_bytes: u64,
) -> u64 {
    admission
        .additional_over_decode_reserve_bytes
        .saturating_sub(startup_reserved_bytes)
}

fn qwen_fixed_costs(base: FixedCosts, plan: QwenPlan) -> Option<FixedCosts> {
    Some(FixedCosts {
        host_bytes: base.host_bytes,
        provider_bytes: base
            .provider_bytes
            .checked_add(plan.reserved_provider_bytes)?
            .checked_add(plan.reserved_verify_bytes)?,
    })
}

fn qwen_allowed_for_request(allow_dspark: bool) -> bool {
    !allow_dspark
}

/// Qwen serves raw completions only, and only while the proposal slot is not
/// forced on (`qwen_allowed_for_request`). A direct run has one request
/// shape, so a chat run, or any run with the slot forced on, can never route
/// a request to Qwen; reserving it there cost 1.7 GiB of residency for
/// nothing. A server takes both shapes over its lifetime and keeps it.
fn qwen_usable_by_run(config: &RuntimeConfig) -> bool {
    config.dspark != DSparkRequest::On
        && (config.surface == RuntimeSurface::Server || !config.chat)
}

fn qwen_provider_reserve(
    device: Device,
    variants: &[QwenVariant],
    context_capacity: usize,
) -> Result<u64> {
    if variants.is_empty() {
        return Ok(0);
    }
    if context_capacity == 0 {
        return Err(DeltafinError::new(
            "Qwen reserve context capacity cannot be empty",
        ));
    }
    let scalar_bytes = if matches!(device, Device::Cpu) { 4 } else { 2 };
    let model_bytes = variants.iter().try_fold(0_u64, |total, variant| {
        variant
            .parameter_count()
            .checked_mul(scalar_bytes)
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| DeltafinError::new("Qwen resident model reserve overflows u64"))
    })?;
    let context_capacity = u64::try_from(context_capacity)
        .map_err(|_| DeltafinError::new("Qwen reserve context capacity exceeds u64"))?;
    let cache_bytes = variants.iter().try_fold(0_u64, |maximum, variant| {
        let architecture = variant.architecture();
        if context_capacity > architecture.maximum_position {
            return Err(DeltafinError::new(
                "Qwen reserve exceeds a selected checkpoint's context contract",
            ));
        }
        let bytes = architecture
            .layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(context_capacity))
            .and_then(|value| value.checked_mul(architecture.key_value_heads))
            .and_then(|value| value.checked_mul(architecture.head_dim))
            .and_then(|value| value.checked_mul(scalar_bytes))
            .ok_or_else(|| DeltafinError::new("Qwen bounded KV reserve overflows u64"))?;
        Ok(maximum.max(bytes))
    })?;
    // Covers logits, attention scores, rotary values, and the largest live MLP
    // intermediates. Model and KV storage dominate this deliberately bounded
    // allowance by orders of magnitude.
    const GENERATION_SCRATCH_BYTES: u64 = 128 * 1024 * 1024;
    model_bytes
        .checked_add(cache_bytes)
        .and_then(|bytes| bytes.checked_add(GENERATION_SCRATCH_BYTES))
        .ok_or_else(|| DeltafinError::new("Qwen total provider reserve overflows u64"))
}

/// Everything the startup plan selection reads besides the fixed costs, so the
/// engine and the tests that probe pin admission run the same chain.
struct PlanSelector<'a> {
    host_memory: HostMemory,
    provider_memory: ProviderMemory,
    provider_layer_bytes: &'a [u64],
    residency_override: ResidencyOverride,
    discovered_qwen: QwenPlan,
    device: Device,
}

impl PlanSelector<'_> {
    /// One complete plan for a set of fixed costs: the resident spine prefix
    /// plus the Qwen chain (wide, then probe-only, then rejected). The chain
    /// is deterministic in `fixed`, which pin admission relies on when it
    /// re-runs it with candidate pin charges.
    fn select(&self, fixed: FixedCosts) -> Result<(QwenPlan, ResidencySelection, u64)> {
        let (baseline_residency, baseline_transient) = select_residency_with_transient(
            self.host_memory,
            self.provider_memory,
            self.provider_layer_bytes,
            fixed,
            self.residency_override,
        )?;
        let admit_qwen = |plan: QwenPlan| {
            let qwen_fixed = qwen_fixed_costs(fixed, plan)?;
            let (residency, transient_layer_bytes) = select_residency_with_transient(
                self.host_memory,
                self.provider_memory,
                self.provider_layer_bytes,
                qwen_fixed,
                self.residency_override,
            )
            .ok()?;
            qwen_residency_admitted(&residency).then_some((plan, residency, transient_layer_bytes))
        };
        let discovered_qwen = self.discovered_qwen;
        Ok(if discovered_qwen.reserved_provider_bytes != 0 {
            if let Some(admitted) = admit_qwen(discovered_qwen) {
                admitted
            } else if let Some(admitted) =
                qwen_probe_only_fallback(discovered_qwen, self.device).and_then(admit_qwen)
            {
                admitted
            } else {
                (
                    QwenPlan::inactive(QwenRuntimeState::MemoryRejected),
                    baseline_residency,
                    baseline_transient,
                )
            }
        } else {
            (discovered_qwen, baseline_residency, baseline_transient)
        })
    }
}

/// Whether the pin-free resident spine prefix outranks the pin tier.
///
/// A spine that fits completely does: streaming it back in costs every pass,
/// while the tier's gain is bounded by what routes happen to hit it. An
/// explicit residency request does too, because the operator asked for exactly
/// that prefix and an automatic tier must not quietly undercut it. Only a
/// prefix the memory budget itself cut short may give way to pins.
fn pin_tier_keeps_resident_prefix(stop: ResidencyStop) -> bool {
    matches!(
        stop,
        ResidencyStop::AllLayersFit
            | ResidencyStop::ExplicitLayerLimit
            | ResidencyStop::ExplicitByteLimit
    )
}

/// Whether a pin roster of `count` whole experts keeps the protected parts of
/// the pin-free `reference` plan: the same Qwen plan and a valid memory proof
/// always, and the same resident spine prefix when `keep_spine`.
fn pin_count_admitted<S>(
    count: usize,
    span_bytes: u64,
    base: FixedCosts,
    select_plan: &S,
    reference_qwen: QwenPlan,
    reference: &ResidencySelection,
    keep_spine: bool,
) -> bool
where
    S: Fn(FixedCosts) -> Result<(QwenPlan, ResidencySelection, u64)>,
{
    if count == 0 {
        return true;
    }
    let Some(host_bytes) = (count as u64)
        .checked_mul(span_bytes)
        .and_then(|charge| base.host_bytes.checked_add(charge))
    else {
        return false;
    };
    match select_plan(FixedCosts {
        host_bytes,
        provider_bytes: base.provider_bytes,
    }) {
        Ok((qwen, residency, _)) => {
            qwen == reference_qwen
                && qwen_residency_admitted(&residency)
                && (!keep_spine || residency.resident_layers == reference.resident_layers)
        }
        Err(_) => false,
    }
}

/// Largest whole-expert pin roster that [`pin_count_admitted`] accepts.
/// Charges grow with the count and a larger fixed cost never grows residency
/// or upgrades the Qwen chain, so the admitted counts form a prefix and binary
/// search finds its end (a test checks that claim against the real selector).
fn admit_pin_count<S>(
    eligible: usize,
    span_bytes: u64,
    base: FixedCosts,
    select_plan: &S,
    reference_qwen: QwenPlan,
    reference: &ResidencySelection,
    keep_spine: bool,
) -> usize
where
    S: Fn(FixedCosts) -> Result<(QwenPlan, ResidencySelection, u64)>,
{
    let admitted = |count: usize| {
        pin_count_admitted(
            count,
            span_bytes,
            base,
            select_plan,
            reference_qwen,
            reference,
            keep_spine,
        )
    };
    let (mut low, mut high) = (0_usize, eligible);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if admitted(middle) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn qwen_residency_admitted(selection: &ResidencySelection) -> bool {
    // Resident K3 layers are an I/O optimization, not model authority.  The
    // previous policy rejected Qwen whenever its safely reserved provider
    // storage displaced even one resident layer.  On a 64 GiB M1 Max that
    // made the installed drafter permanently `memory-rejected`, so every
    // accepted output token required another full 93-layer target pass.
    //
    // A valid residency selection has already charged both Qwen variants,
    // their maximum KV generation, the target caches, and the transient
    // streamed layer.  It is therefore safe to trade some retained layers for
    // fewer full-K3 verification passes.  Bad proposals can waste a pass but
    // cannot change output: K3 still verifies every token with all 16 routed
    // experts.  Reject only selections whose memory proof itself failed.
    !matches!(
        selection.stop,
        ResidencyStop::InvalidPolicy
            | ResidencyStop::HostBudgetUnknown
            | ResidencyStop::DeviceBudgetUnknown
            | ResidencyStop::FixedCostsExceedBudget
            | ResidencyStop::ArithmeticOverflow
    )
}

impl NativeTargetEngine {
    fn storage_bytes(&self) -> StorageBytes {
        StorageBytes {
            spine: self.spine_pipeline.bytes_read(),
            expert_demand: self.expert_reader.bytes_read(),
            expert_prefetch: self
                .expert_prefetch_reader
                .as_ref()
                .map_or(0, Reader::bytes_read),
        }
    }
}

impl NativeTargetEngine {
    /// Execute one independent exact target request into an arbitrary native
    /// byte sink. This is the shared compiled boundary for the CLI and the
    /// in-process OpenAI server; neither needs a Python bridge. The CLI
    /// passes its SIGINT flag as the interrupt source; the server passes the
    /// requesting client's socket liveness.
    #[expect(
        clippy::too_many_arguments,
        reason = "the native generation boundary keeps authority, publication, and cooperative-stop controls explicit"
    )]
    fn generate_tokens_to_with_interrupt<W: Write, I: InterruptSource>(
        &mut self,
        prompt: &[u32],
        maximum_new: Option<u64>,
        stats: bool,
        verbose_layer_profile: bool,
        allow_dspark: bool,
        reusable_target: bool,
        interrupt: &I,
        mut events: Option<&mut RunEventLog>,
        output: &mut W,
    ) -> Result<NativeGeneration> {
        if !self.readiness.permits_generation() {
            return Err(DeltafinError::new(
                "native target execution was requested before bootstrap became ready",
            ));
        }
        self.lifecycle.begin()?;
        let mut dspark_lease = None;
        let mut prefill_branch = None;
        let mut prefill_start = 0_usize;
        let mut reuse_transaction: Option<(PublishedTargetBoundary, TargetStateBranch)> = None;
        let mut result = (|| -> Result<NativeGeneration> {
            let started = Instant::now();
            // Structured evidence, explicit --stats diagnostics, and the
            // separate --layer-profile opt-in all need per-layer phase data
            // collected; ordinary CLI/server generation needs none of them,
            // so its target loop remains free of profiling clock reads.
            let collect_profile = stats || verbose_layer_profile || events.is_some();
            if self.output_decoder.pending_bytes() != 0 {
                return Err(DeltafinError::new(
                    "native output decoder retained bytes before a request",
                ));
            }

            let maximum_context =
                usize::try_from(self.context_growth.admitted_expanded_context_tokens)
                    .map_err(|_| DeltafinError::new("native context limit does not fit usize"))?;
            let maximum_new = maximum_new
                .map(|value| {
                    usize::try_from(value)
                        .map_err(|_| DeltafinError::new("--max-new does not fit this host"))
                })
                .transpose()?;
            let eos_token = u32::try_from(self.model.eos_token_id)
                .map_err(|_| DeltafinError::new("model EOS token does not fit u32"))?;
            // Stage the request's context ceiling for the first chunk's
            // admission. Assigned unconditionally (None for unbounded or
            // oversized requests) so a stale envelope from an earlier
            // request can never leak into this one.
            self.pending_context_envelope = context_pre_admission_envelope(
                prompt.len(),
                maximum_new,
                self.speculative_max_drafts,
                self.context_growth.admitted_expanded_context_tokens,
            );
            let mut decode = DecodeArena::new(
                prompt,
                maximum_context,
                maximum_new,
                eos_token,
                self.speculative_max_drafts,
            )?;
            let mut counters = NativeRunCounters {
                storage_at_start: self.storage_bytes(),
                ..NativeRunCounters::default()
            };
            if let Some(stop) = decode.stop_reason() {
                if let Some(events) = events.as_deref_mut() {
                    events.emit_prefill_done_with_profile(
                        elapsed_ns(started),
                        &[],
                        counters.target_profile_json(),
                    )?;
                }
                if stats {
                    print_run_stats(
                        &counters,
                        started,
                        self.storage_homes.as_deref(),
                        self.storage_bytes(),
                    );
                }
                self.finish_output(output)?;
                return Ok(NativeGeneration {
                    token_ids: Box::new([]),
                    stop,
                    wrote_text: false,
                });
            }
            if interrupt.requested() {
                if let Some(events) = events.as_deref_mut() {
                    events.emit_prefill_done_with_profile(
                        elapsed_ns(started),
                        &[],
                        counters.target_profile_json(),
                    )?;
                }
                if stats {
                    print_run_stats(
                        &counters,
                        started,
                        self.storage_homes.as_deref(),
                        self.storage_bytes(),
                    );
                }
                self.finish_output(output)?;
                return Ok(NativeGeneration {
                    token_ids: Box::new([]),
                    stop: StopReason::Interrupted,
                    wrote_text: false,
                });
            }

            let (start, branch, target_cache) =
                self.begin_target_reuse_request(prompt, reusable_target)?;
            prefill_start = start;
            prefill_branch = branch;
            dspark_lease = allow_dspark
                .then(|| {
                    self.dspark
                        .begin_request(prompt, target_cache, reusable_target)
                        .ok()
                })
                .flatten();

            // Prefill commits each bounded chunk atomically. The provider's
            // committed KDA/MLA state therefore becomes the exact prefix for the
            // next chunk, while at most 64 activation rows are live at once.
            let mut first = prefill_start;
            let mut prediction = None;
            let mut interrupted_after_prefill = false;
            while let Some(range) = next_prompt_chunk(prompt.len(), first)? {
                if interrupt.requested() {
                    if let Some(events) = events.as_deref_mut() {
                        // A completed prefix transaction contains no response
                        // token until the entire prompt has been consumed.
                        events.emit_prefill_done_with_profile(
                            elapsed_ns(started),
                            &[],
                            counters.target_profile_json(),
                        )?;
                    }
                    if stats {
                        print_run_stats(
                        &counters,
                        started,
                        self.storage_homes.as_deref(),
                        self.storage_bytes(),
                    );
                    }
                    self.finish_output(output)?;
                    return Ok(NativeGeneration {
                        token_ids: Box::new([]),
                        stop: StopReason::Interrupted,
                        wrote_text: false,
                    });
                }
                let capture_dspark = self.dspark_tracks_rows(dspark_lease.as_ref());
                let completed = self.execute_target_chunk(
                    &prompt[range.clone()],
                    stats,
                    verbose_layer_profile,
                    collect_profile,
                    capture_dspark,
                )?;
                if completed.predictions.len() != 1 {
                    return Err(DeltafinError::new(
                        "prefill target sequence did not return exactly one authoritative tail token",
                    ));
                }
                prediction = Some(completed.predictions[0]);
                counters.absorb(&completed);
                first = range.end;
                // Observe SIGINT only after the provider transaction returned.
                // Optional draft state from an interrupted transaction is not
                // advanced or published.
                interrupted_after_prefill = interrupt.requested();
                if !interrupted_after_prefill {
                    self.advance_dspark_rows(
                        dspark_lease.as_ref(),
                        &completed,
                        &prompt[range.clone()],
                    );
                }
            }
            let prediction = prediction.ok_or_else(|| {
                DeltafinError::new("native target prompt produced no bounded prefill chunk")
            })?;
            interrupted_after_prefill |= interrupt.requested();
            if !interrupted_after_prefill && let Some(lease) = dspark_lease.as_ref() {
                let _ = self.dspark.capture_prompt_boundary(lease, prompt);
            }
            interrupted_after_prefill |= interrupt.requested();
            if reusable_target && !interrupted_after_prefill {
                reuse_transaction = Some(self.publish_prompt_parent_and_begin_decode_branch(
                    prompt,
                    prediction,
                    prefill_branch.take(),
                )?);
            }

            let mut wrote_token = false;
            let mut generated = Vec::with_capacity(maximum_new.unwrap_or(16).min(1_024));
            let (initial_token, initial_stop) = {
                let committed = decode.commit_target(prediction)?;
                if committed.emitted.len() != 1 {
                    return Err(DeltafinError::new(
                        "initial native decode did not emit exactly one full-target token",
                    ));
                }
                (committed.emitted[0], committed.stop)
            };
            generated.push(initial_token);
            counters.generated_tokens = counters.generated_tokens.saturating_add(1);
            if let Some(events) = events.as_deref_mut() {
                events.emit_prefill_done_with_profile(
                    elapsed_ns(started),
                    &[initial_token],
                    counters.target_profile_json(),
                )?;
            }
            if initial_token != eos_token {
                self.write_token(output, initial_token)?;
                wrote_token = true;
            }
            if stats {
                print_run_stats(
                        &counters,
                        started,
                        self.storage_homes.as_deref(),
                        self.storage_bytes(),
                    );
            }

            let mut pending_token = initial_token;
            self.validate_pending_target_lag(&decode, pending_token)?;
            let mut final_stop = stop_after_transaction(
                initial_stop,
                interrupted_after_prefill || interrupt.requested(),
            );
            let mut decode_step = 0_u64;
            let mut qwen_policy = QwenRequestPolicy::new(
                qwen_allowed_for_request(allow_dspark) && self.qwen.is_available(),
                self.speculative_max_drafts,
            );
            self.qwen.begin_request();
            while final_stop.is_none() {
                if interrupt.requested() {
                    final_stop = Some(StopReason::Interrupted);
                    break;
                }
                // Do not even read the clock on the normal server/default
                // path. The optional branch is the complete disabled cost of
                // collecting transaction timing.
                let transaction_started = events.as_ref().map(|_| Instant::now());
                let remaining_context = maximum_context.saturating_sub(decode.history().len());
                let remaining_output = maximum_new
                    .map(|maximum| maximum.saturating_sub(decode.generated()))
                    .unwrap_or(remaining_context);
                let draft_budget = bounded_draft_budget(
                    self.speculative_max_drafts,
                    remaining_context,
                    remaining_output,
                );
                let needs_dspark_baseline = dspark_lease
                    .as_ref()
                    .is_some_and(|lease| self.dspark.needs_target_baseline(lease).unwrap_or(false));
                let mut dspark_proposal = if !needs_dspark_baseline && draft_budget != 0 {
                    dspark_lease.as_ref().and_then(|lease| {
                        self.dspark
                            .propose(lease, pending_token, u8::try_from(draft_budget.min(7)).ok())
                            .ok()
                            .flatten()
                    })
                } else {
                    None
                };
                let mut qwen_proposal_used = false;
                let mut qwen_proposed_drafts = 0_usize;
                let mut drafts = if let Some(proposal) = dspark_proposal.as_ref() {
                    proposal.token_ids().to_vec()
                } else if needs_dspark_baseline {
                    Vec::new()
                } else {
                    let qwen_width = qwen_policy.proposal_width(draft_budget);
                    let qwen_attempted = qwen_width != 0;
                    let qwen = if qwen_attempted {
                        self.qwen
                            .propose(&self.provider, decode.history(), qwen_width)
                    } else {
                        QwenDraftProposal::default()
                    };
                    if qwen.token_ids().is_empty() {
                        if qwen_attempted {
                            qwen_policy.record_empty(qwen.confidence_stopped());
                        }
                        if qwen_attempted && qwen.confidence_stopped() {
                            // A low-confidence wide row is evidence against
                            // paying for a verifier at this position. Decode
                            // one exact target row but retain request-local
                            // qualification for the next position.
                            Vec::new()
                        } else {
                            self.ngram_drafter
                                .propose(decode.history(), draft_budget)?
                                .into_vec()
                        }
                    } else {
                        qwen_proposal_used = true;
                        qwen.token_ids().to_vec()
                    }
                };
                // Stateless text/ngram sources have no transaction to abort;
                // an impossible suffix after EOS can be discarded eagerly.
                if dspark_proposal.is_none() {
                    truncate_after_first(&mut drafts, eos_token);
                }
                if qwen_proposal_used {
                    qwen_proposed_drafts = drafts.len();
                }

                let proposal_candidate_count = drafts.len();
                // An ordinary verify keeps one KDA boundary per row (453 MiB
                // each on K3) so it can commit any prefix. When live memory
                // cannot hold them, a slot proposal is verified as one
                // full-commit transaction instead, like Qwen's: a full match
                // commits every row and a mismatch reruns only the accepted
                // prefix, still exactly.
                let (use_verify, full_commit) = if drafts.is_empty() {
                    (false, false)
                } else if qwen_proposal_used {
                    (self.admit_verify_width(drafts.len().saturating_add(1), true)?, true)
                } else if self.admit_verify_width(drafts.len().saturating_add(1), false)? {
                    (true, false)
                } else if dspark_proposal.is_some() {
                    (self.admit_verify_width(drafts.len().saturating_add(1), true)?, true)
                } else {
                    (false, false)
                };
                let proposal_memory_rejected = proposal_candidate_count != 0 && !use_verify;
                if full_commit && !qwen_proposal_used {
                    counters.full_commit_slot_verifies =
                        counters.full_commit_slot_verifies.saturating_add(1);
                }
                if !use_verify {
                    if let (Some(lease), Some(proposal)) =
                        (dspark_lease.as_ref(), dspark_proposal.as_ref())
                    {
                        let _ = self.dspark.abort_proposal(lease, Some(proposal));
                    }
                    dspark_proposal = None;
                    if qwen_proposal_used {
                        self.qwen.telemetry.fallbacks =
                            self.qwen.telemetry.fallbacks.saturating_add(1);
                        if proposal_memory_rejected {
                            self.qwen.telemetry.verify_memory_rejections = self
                                .qwen
                                .telemetry
                                .verify_memory_rejections
                                .saturating_add(1);
                        }
                        qwen_proposal_used = false;
                    }
                    drafts.clear();
                }
                // Draft providers are scheduling-only. If SIGINT arrived while
                // one was proposing, abort that optional transaction and do
                // not start another full-target provider call.
                if interrupt.requested() {
                    if let (Some(lease), Some(proposal)) =
                        (dspark_lease.as_ref(), dspark_proposal.as_ref())
                    {
                        let _ = self.dspark.abort_proposal(lease, Some(proposal));
                    }
                    final_stop = Some(StopReason::Interrupted);
                    break;
                }
                let (emitted, stop, completed, accepted_drafts, interrupted_after_target) =
                    if use_verify {
                        // The provider is exactly one token behind DecodeArena:
                        // `pending_token` was authored by K3 but has not yet been
                        // fed through the target. Each following input is only an
                        // untrusted proposal. Verify returns the authoritative
                        // next-token prediction after every one of those rows.
                        let mut inputs = Vec::with_capacity(drafts.len().saturating_add(1));
                        inputs.push(pending_token);
                        inputs.extend_from_slice(&drafts);
                        let capture_dspark = self.dspark_tracks_rows(dspark_lease.as_ref());
                        let verifier_started = Instant::now();
                        let prepared = if full_commit {
                            self.prepare_full_commit_verify_chunk(
                                &inputs,
                                stats,
                                verbose_layer_profile,
                                collect_profile,
                                capture_dspark,
                            )?
                        } else {
                            self.prepare_target_chunk(
                                &inputs,
                                TargetSequenceMode::Verify,
                                stats,
                                verbose_layer_profile,
                                collect_profile,
                                capture_dspark,
                            )?
                        };
                        let plan = match decode.plan_verified(&drafts, &prepared.predictions) {
                            Ok(plan) => plan,
                            Err(error) => {
                                return Err(cancel_after_error(prepared.sequence, error));
                            }
                        };
                        let emitted = plan.emitted().to_vec();
                        let stop = plan.stop();
                        let accepted_drafts = plan.accepted_drafts();
                        if emitted.is_empty() {
                            return Err(cancel_after_error(
                                prepared.sequence,
                                DeltafinError::new(
                                    "full-target verification planned no authoritative token",
                                ),
                            ));
                        }

                        // A full-commit verify (Qwen, or a slot proposal whose
                        // per-row boundaries did not fit) is optimistic. A full
                        // match can publish the wide transaction directly. A
                        // mismatch first cancels every staged wide row, then
                        // reruns only old-pending + accepted drafts. The rerun
                        // must reproduce the saved full-K3 authoritative IDs
                        // before its complete sequence is allowed to commit.
                        let completed = if full_commit {
                            if interrupt.requested() {
                                prepared.sequence.cancel()?;
                                final_stop = Some(StopReason::Interrupted);
                                break;
                            }
                            if emitted.len() == inputs.len() {
                                self.commit_target_chunk(prepared, inputs.len())?
                            } else {
                                prepared.sequence.cancel()?;
                                if interrupt.requested() {
                                    final_stop = Some(StopReason::Interrupted);
                                    break;
                                }
                                let rerun_inputs = &inputs[..emitted.len()];
                                let rerun = self.prepare_full_commit_verify_chunk(
                                    rerun_inputs,
                                    stats,
                                    verbose_layer_profile,
                                    collect_profile,
                                    capture_dspark,
                                )?;
                                if rerun.predictions.as_ref() != emitted.as_slice() {
                                    return Err(cancel_after_error(
                                        rerun.sequence,
                                        DeltafinError::new(
                                            "full-target Qwen verifier rerun disagreed with the saved authoritative prefix",
                                        ),
                                    ));
                                }
                                if interrupt.requested() {
                                    rerun.sequence.cancel()?;
                                    final_stop = Some(StopReason::Interrupted);
                                    break;
                                }
                                self.commit_target_chunk(rerun, rerun_inputs.len())?
                            }
                        } else {
                            // Commit old pending + accepted draft rows. The
                            // newest correction/bonus is authoritative output
                            // but remains the one pending input for the next
                            // target transaction.
                            self.commit_target_chunk(prepared, emitted.len())?
                        };
                        let verified = decode.apply_verified(plan)?;
                        if verified.emitted != emitted.as_slice()
                            || verified.stop != stop
                            || verified.accepted_drafts != accepted_drafts
                        {
                            return Err(DeltafinError::new(
                                "applied decode plan disagreed with its immutable authoritative decision",
                            ));
                        }
                        let verifier_seconds = verifier_started.elapsed();
                        let interrupted_after_target = interrupt.requested();
                        if interrupted_after_target {
                            if let (Some(lease), Some(proposal)) =
                                (dspark_lease.as_ref(), dspark_proposal.as_ref())
                            {
                                let _ = self.dspark.abort_proposal(lease, Some(proposal));
                            }
                        } else if let (Some(lease), Some(proposal)) =
                            (dspark_lease.as_ref(), dspark_proposal.as_ref())
                        {
                            if let Some(rows) = completed.dspark_rows.as_ref() {
                                if self
                                    .dspark
                                    .resolve(
                                        lease,
                                        proposal,
                                        accepted_drafts,
                                        &emitted,
                                        rows,
                                        verifier_seconds,
                                    )
                                    .is_ok()
                                {
                                    let verified_step_seconds = Duration::from_secs_f64(
                                        proposal.seconds() + verifier_seconds.as_secs_f64(),
                                    );
                                    let _ = self.dspark.record_verified_step(
                                        lease,
                                        proposal,
                                        accepted_drafts,
                                        emitted.len(),
                                        verified_step_seconds,
                                    );
                                }
                            } else {
                                let _ = self.dspark.abort_proposal(lease, Some(proposal));
                                let _ = self.dspark.disable_proposals(
                                    lease,
                                    "provider-owned DSpark verification capture failed",
                                    Some(false),
                                );
                            }
                        } else {
                            self.advance_dspark_rows(
                                dspark_lease.as_ref(),
                                &completed,
                                &inputs[..emitted.len()],
                            );
                        }
                        (
                            emitted,
                            stop,
                            completed,
                            accepted_drafts,
                            interrupted_after_target,
                        )
                    } else {
                        // Memory pressure or an absent useful n-gram is only a
                        // scheduling decision. Fall back to one ordinary exact K3
                        // row; output semantics and sampling are unchanged.
                        let capture_dspark = self.dspark_tracks_rows(dspark_lease.as_ref());
                        let baseline_started = Instant::now();
                        let completed = self.execute_target_chunk(
                            &[pending_token],
                            stats,
                            verbose_layer_profile,
                            collect_profile,
                            capture_dspark,
                        )?;
                        if completed.predictions.len() != 1 {
                            return Err(DeltafinError::new(
                                "decode target sequence did not return exactly one authoritative tail token",
                            ));
                        }
                        let (emitted, stop) = {
                            let committed = decode.commit_target(completed.predictions[0])?;
                            if committed.emitted.len() != 1 {
                                return Err(DeltafinError::new(
                                    "ordinary native decode did not emit exactly one full-target token",
                                ));
                            }
                            (committed.emitted.to_vec(), committed.stop)
                        };
                        let baseline_seconds = baseline_started.elapsed();
                        let interrupted_after_target = interrupt.requested();
                        if !interrupted_after_target {
                            self.advance_dspark_rows(
                                dspark_lease.as_ref(),
                                &completed,
                                &[pending_token],
                            );
                        }
                        if needs_dspark_baseline && !interrupted_after_target {
                            if let Some(lease) = dspark_lease.as_ref() {
                                let _ =
                                    self.dspark
                                        .record_target_baseline(lease, baseline_seconds, 1);
                            }
                        }
                        (emitted, stop, completed, 0, interrupted_after_target)
                    };

                counters.absorb(&completed);
                if use_verify {
                    counters.verify_transactions = counters.verify_transactions.saturating_add(1);
                    counters.verified_draft_tokens = counters
                        .verified_draft_tokens
                        .saturating_add(drafts.len() as u64);
                    counters.accepted_draft_tokens = counters
                        .accepted_draft_tokens
                        .saturating_add(accepted_drafts as u64);
                    if qwen_proposal_used {
                        self.qwen.record_verified(accepted_drafts);
                        qwen_policy.record_verified(accepted_drafts, qwen_proposed_drafts);
                    }
                }
                if let (Some(events), Some(transaction_started)) =
                    (events.as_deref_mut(), transaction_started)
                {
                    events.emit_decode_step(
                        decode_step,
                        elapsed_ns(transaction_started),
                        &emitted,
                        proposal_candidate_count as u64,
                        drafts.len() as u64,
                        accepted_drafts as u64,
                        proposal_memory_rejected,
                    )?;
                    decode_step = decode_step.checked_add(1).ok_or_else(|| {
                        DeltafinError::new("native decode_step sequence overflowed u64")
                    })?;
                }
                for token_id in emitted.iter().copied() {
                    generated.push(token_id);
                    counters.generated_tokens = counters.generated_tokens.saturating_add(1);
                    // EOS is authoritative control state, not user-visible text.
                    if token_id != eos_token {
                        self.write_token(output, token_id)?;
                        wrote_token = true;
                    }
                }
                pending_token = *emitted.last().ok_or_else(|| {
                    DeltafinError::new("native target transaction lost its authoritative tail")
                })?;
                self.validate_pending_target_lag(&decode, pending_token)?;
                // Stop before another provider call while retaining every
                // token K3 just authored. A pending SIGINT owns the boundary
                // so optional state is never published under a cancellation.
                final_stop =
                    stop_after_transaction(stop, interrupted_after_target || interrupt.requested());
                if stats {
                    print_run_stats(
                        &counters,
                        started,
                        self.storage_homes.as_deref(),
                        self.storage_bytes(),
                    );
                }
            }
            self.finish_output(output)?;
            let mut final_stop = stop_after_transaction(final_stop, interrupt.requested())
                .ok_or_else(|| {
                    DeltafinError::new("native decode loop ended without an exact stop reason")
                })?;
            if final_stop != StopReason::Interrupted
                && let Some((prompt_boundary, branch)) = reuse_transaction.take()
            {
                let final_tokens = canonical_chat_target_boundary(
                    &self.tokenizer,
                    eos_token,
                    prompt,
                    &generated,
                    final_stop,
                )
                .then(|| {
                    let mut tokens = Vec::with_capacity(prompt.len() + generated.len());
                    tokens.extend_from_slice(prompt);
                    tokens.extend_from_slice(&generated);
                    tokens.into_boxed_slice()
                });
                self.pending_target_publication = Some(PendingTargetPublication {
                    branch,
                    prompt: prompt_boundary,
                    final_tokens,
                    dspark_lease: dspark_lease.take(),
                });
            }
            final_stop = stop_after_transaction(Some(final_stop), interrupt.requested())
                .expect("an existing terminal stop remains terminal");
            Ok(NativeGeneration {
                token_ids: generated.into_boxed_slice(),
                stop: final_stop,
                wrote_text: wrote_token,
            })
        })();

        if interrupt.requested()
            && let Ok(generation) = &mut result
        {
            generation.stop = StopReason::Interrupted;
        }
        let interrupted = result
            .as_ref()
            .is_ok_and(|generation| generation.stop == StopReason::Interrupted);
        if interrupted {
            // A run-only cancellation is a successful partial result, but it
            // must never become a cross-request target/draft cache boundary.
            // Explicitly resolve provider branches before lifecycle publish;
            // relying only on their best-effort Drop rollback would hide an
            // unsafe branch-discard failure.
            let abort_reuse = (|| -> Result<()> {
                if let Some((_prompt, branch)) = reuse_transaction.take() {
                    let restored = self.provider.discard_target_state_branch(branch)?;
                    self.committed_context_tokens = restored.committed_positions;
                    self.mla_capacity_tokens = restored.committed_positions;
                }
                if let Some(branch) = prefill_branch.take() {
                    let restored = self.provider.discard_target_state_branch(branch)?;
                    self.committed_context_tokens = restored.committed_positions;
                    self.mla_capacity_tokens = restored.committed_positions;
                }
                if let Some(pending) = self.pending_target_publication.take() {
                    let restored = self.provider.discard_target_state_branch(pending.branch)?;
                    self.committed_context_tokens = restored.committed_positions;
                    self.mla_capacity_tokens = restored.committed_positions;
                    if let Some(lease) = pending.dspark_lease.as_ref() {
                        let _ = self.dspark.abort_request(lease);
                    }
                }
                self.published_target_boundary = None;
                Ok(())
            })();
            if let Err(error) = abort_reuse {
                result = Err(DeltafinError::new(format!(
                    "abort interrupted native target reuse state: {error}"
                )));
            }
        }

        if let Some(lease) = dspark_lease.as_ref() {
            if result
                .as_ref()
                .is_ok_and(|generation| generation.stop != StopReason::Interrupted)
            {
                let _ = self.dspark.finish_request(lease);
            } else {
                let _ = self.dspark.abort_request(lease);
            }
        }
        let abandoned_layer = self.spine_pipeline.discard_pending();
        let pending_utf8_bytes = self.output_decoder.pending_bytes();
        match result {
            Ok(generation)
                if abandoned_layer.is_none()
                    && pending_utf8_bytes == 0
                    && self.committed_context_tokens <= self.mla_capacity_tokens =>
            {
                self.lifecycle.publish()?;
                Ok(generation)
            }
            Ok(_) => {
                self.output_decoder.reset();
                self.lifecycle.poison();
                Err(DeltafinError::new(format!(
                    "native request ended with unpublished runtime state (pending spine layer {:?}, UTF-8 bytes {}, context {}/{})",
                    abandoned_layer,
                    pending_utf8_bytes,
                    self.committed_context_tokens,
                    self.mla_capacity_tokens,
                )))
            }
            Err(error) => {
                self.output_decoder.reset();
                self.lifecycle.poison();
                if let Some(layer) = abandoned_layer {
                    Err(DeltafinError::new(format!(
                        "{error}; discarded unpublished spine read for layer {layer} and poisoned this native engine"
                    )))
                } else {
                    Err(error)
                }
            }
        }
    }
}

impl NativeTargetEngine {
    pub(crate) fn prefill_and_decode_interruptible<I: InterruptSource>(
        &mut self,
        config: &RuntimeConfig,
        interrupt: &I,
    ) -> Result<()> {
        self.prefill_and_decode_with(config, interrupt)
    }

    fn prefill_and_decode_with<I: InterruptSource>(
        &mut self,
        config: &RuntimeConfig,
        interrupt: &I,
    ) -> Result<()> {
        let mut events = config
            .events_jsonl
            .as_deref()
            .map(|path| RunEventLog::open(Some(path)))
            .transpose()?;
        let run_started = Instant::now();
        let prompt = match self.encode_prompt(config) {
            Ok(prompt) => prompt,
            Err(error) => {
                emit_terminal_run_error(events.as_mut(), "encode", &error, run_started);
                return Err(error);
            }
        };
        let maximum_new = config.max_new.unwrap_or_else(|| {
            self.context_growth
                .admitted_expanded_context_tokens
                .saturating_sub(prompt.len() as u64)
        });
        if let Some(events) = events.as_mut() {
            let event_eos_token_id = match u32::try_from(self.model.eos_token_id) {
                Ok(token_id) => token_id,
                Err(_) => {
                    let error = DeltafinError::new("model EOS token does not fit u32");
                    emit_terminal_run_error(Some(events), "run_start", &error, run_started);
                    return Err(error);
                }
            };
            let started = events.emit_run_start(
                &config.prompt,
                config.chat,
                maximum_new,
                &prompt,
                serde_json::json!({
                    "device": self.device.to_string(),
                    "device_selection_policy": self.device_selection_policy.name(),
                    "expert_backend": self.expert_backend.to_string(),
                    "eos_token_id": event_eos_token_id,
                    "spine": {
                        "requested": format!("{:?}", config.spine),
                        "loaded": self.program.representation().to_string(),
                    },
                    "dspark": {
                        "requested": format!("{:?}", config.dspark),
                        "eagle3_requested": format!("{:?}", config.eagle3),
                        "mode": format!("{:?}", self.dspark.mode()).to_ascii_lowercase(),
                        "backend_loaded": self.dspark.backend().is_some(),
                        "backend": self.dspark.backend().map(NativeDraftBackend::name),
                    },
                    "qwen": {
                        "requested": format!("{:?}", config.qwen),
                        "state": self.qwen.state.to_string(),
                        "source_loaded": self.qwen.source.is_some(),
                        "reserved_provider_bytes": self.qwen.reserved_provider_bytes,
                        "reserved_verify_bytes": self.qwen.reserved_verify_bytes,
                        "reserved_verify_positions": self.qwen.reserved_verify_positions,
                        "context_capacity": self.qwen.context_capacity,
                    },
                    // Stable top-level proofs consumed by native benchmark
                    // qualification. The richer objects above remain useful
                    // for diagnostics without making availability implicit.
                    "dspark_loaded": self.dspark.backend().is_some(),
                    "universal_draft_loaded": self.qwen.source.is_some(),
                    "speculative_max_drafts": self.speculative_max_drafts,
                }),
            );
            if let Err(error) = started {
                emit_terminal_run_error(Some(events), "run_start", &error, run_started);
                return Err(error);
            }
        }
        let stdout = io::stdout();
        let mut output = CliOutputWriter::new(stdout.lock(), config.chat);
        let generation = match self.generate_tokens_to_with_interrupt(
            &prompt,
            config.max_new,
            config.stats,
            config.layer_profile,
            config.chat || config.dspark == DSparkRequest::On,
            false,
            interrupt,
            events.as_mut(),
            &mut output,
        ) {
            Ok(generation) => generation,
            Err(error) => {
                emit_terminal_run_error(events.as_mut(), "generate", &error, run_started);
                return Err(error);
            }
        };
        let output_result = (|| -> Result<()> {
            output
                .finish(finish_reason_for_stop(generation.stop))
                .map_err(|error| output_error("finish structured output", error))?;
            if output.wrote_public_text() {
                output
                    .write_terminal_newline()
                    .map_err(|error| output_error("write output newline", error))?;
            }
            output
                .flush()
                .map_err(|error| output_error("flush generated output", error))
        })();
        if let Err(error) = output_result {
            emit_terminal_run_error(events.as_mut(), "output", &error, run_started);
            return Err(error);
        }
        if let Some(events) = events.as_mut() {
            let eos_token = match u32::try_from(self.model.eos_token_id) {
                Ok(eos_token) => eos_token,
                Err(_) => {
                    let error = DeltafinError::new("model EOS token does not fit u32");
                    emit_terminal_run_error(Some(events), "decode_completion", &error, run_started);
                    return Err(error);
                }
            };
            let completion_ids = completion_token_ids(&generation.token_ids, eos_token);
            let completion_text = match self.tokenizer.decode(completion_ids) {
                Ok(text) => text,
                Err(error) => {
                    emit_terminal_run_error(Some(events), "decode_completion", &error, run_started);
                    return Err(error);
                }
            };
            let telemetry = self.qwen.telemetry;
            let dspark_metrics = self.dspark.metrics();
            let universal_draft_failures = telemetry.wide_failures.saturating_add(
                if self.qwen.state == QwenRuntimeState::FailedSoft {
                    1
                } else {
                    0
                },
            );
            let ended = events.emit_run_end(
                run_status_name(generation.stop),
                elapsed_ns(run_started),
                &prompt,
                &generation.token_ids,
                completion_ids,
                stop_reason_name(generation.stop),
                completion_text,
                serde_json::json!({
                    "qwen_state": self.qwen.state.to_string(),
                    "qwen": {
                        "reserved_provider_bytes": self.qwen.reserved_provider_bytes,
                        "reserved_verify_bytes": self.qwen.reserved_verify_bytes,
                        "reserved_verify_positions": self.qwen.reserved_verify_positions,
                        "context_capacity": self.qwen.context_capacity,
                        "proposals": telemetry.proposals,
                        "proposed_tokens": telemetry.proposed_tokens,
                        "accepted_tokens": telemetry.accepted_tokens,
                        "fallbacks": telemetry.fallbacks,
                        "verify_memory_rejections": telemetry.verify_memory_rejections,
                        "wide_loads": telemetry.wide_loads,
                        "wide_invocations": telemetry.wide_invocations,
                        "wide_skips": telemetry.wide_skips,
                        "wide_failures": telemetry.wide_failures,
                        "raw_override_selections": telemetry.raw_override_selections,
                        "raw_override_failures": telemetry.raw_override_failures,
                    },
                    "universal_draft": {
                        "available": self.qwen.source.is_some(),
                        "state": self.qwen.state.to_string(),
                        "proposals": telemetry.proposals,
                        "proposed_tokens": telemetry.proposed_tokens,
                        "accepted_tokens": telemetry.accepted_tokens,
                        "fallbacks": telemetry.fallbacks,
                        "verify_memory_rejections": telemetry.verify_memory_rejections,
                        "failures": universal_draft_failures,
                        "wide_loads": telemetry.wide_loads,
                        "wide_invocations": telemetry.wide_invocations,
                        "wide_skips": telemetry.wide_skips,
                        "raw_override_selections": telemetry.raw_override_selections,
                        "raw_override_failures": telemetry.raw_override_failures,
                    },
                    "dspark": {
                        "mode": format!("{:?}", self.dspark.mode()).to_ascii_lowercase(),
                        "backend_loaded": self.dspark.backend().is_some(),
                        "backend": self.dspark.backend().map(NativeDraftBackend::name),
                        "available": self.dspark.backend().is_some(),
                        "sessions": dspark_metrics.sessions,
                        "enabled_sessions": dspark_metrics.enabled_sessions,
                        "disabled_sessions": dspark_metrics.disabled_sessions,
                        "runtime_disables": dspark_metrics.runtime_disables,
                        "context_limit_disables": dspark_metrics.context_limit_disables,
                        "proposals": dspark_metrics.proposals,
                        "proposal_failures": dspark_metrics.proposal_failures,
                        "generated_drafts": dspark_metrics.generated_drafts,
                        "submitted_drafts": dspark_metrics.submitted_drafts,
                        "accepted_drafts": dspark_metrics.accepted_drafts,
                        "emitted_tokens": dspark_metrics.emitted_tokens,
                        "full_matches": dspark_metrics.full_matches,
                        "partial_matches": dspark_metrics.partial_matches,
                        "misses": dspark_metrics.misses,
                        "probe_qualifications": dspark_metrics.probe_qualifications,
                        "state_failures": dspark_metrics.state_failures,
                        "proposal_aborts": dspark_metrics.proposal_aborts,
                        "request_aborts": dspark_metrics.request_aborts,
                        "baseline_passes": dspark_metrics.baseline_passes,
                        "baseline_seconds": dspark_metrics.baseline_seconds,
                        "verifier_seconds": dspark_metrics.verifier_seconds,
                        "verified_step_seconds": dspark_metrics.verified_step_seconds,
                        "economic_steps": dspark_metrics.economic_steps,
                        "economic_disables": dspark_metrics.economic_disables,
                        "target_reprefill_requests": dspark_metrics.target_reprefill_requests,
                    },
                    "lazy_expert_files_missing": self.experts.lazy_missing_files(),
                }),
            );
            if let Err(error) = ended {
                emit_terminal_run_error(Some(events), "run_end", &error, run_started);
                return Err(error);
            }
        }
        if generation.stop == StopReason::Interrupted {
            eprintln!("[stopped by Ctrl-C]");
        }
        Ok(())
    }
}

/// Cooperative cancellation source for one server generation: an HTTP client
/// that has provably disconnected is a standing request to stop spending
/// hours of compute on a response nobody will receive.  The probe is checked
/// at the same transaction boundaries as the CLI's SIGINT flag, so a
/// disconnect can never unwind provider-owned state mid-mutation.
struct ClientPresenceInterrupt<'a>(&'a ClientPresence);

impl InterruptSource for ClientPresenceInterrupt<'_> {
    fn requested(&self) -> bool {
        self.0.disconnected()
    }
}

/// A disconnect-interrupted generation is a *successful partial* result at
/// the engine layer (the engine returns to `Ready`), but it is not a
/// K3-certified response: publishing or memoizing it would replay truncated
/// text for future identical requests.  The adapter therefore converts it
/// into an error; the server's error path settles the target with `Aborted`,
/// which is a no-op because the interrupt tail already discarded every staged
/// reuse branch.
fn client_disconnect_error() -> DeltafinError {
    DeltafinError::new(
        "client disconnected during native generation; the partial result was discarded",
    )
}

impl AuthoritativeTarget for NativeTargetEngine {
    fn generate_target(
        &mut self,
        request: &TargetRequest,
        client: &ClientPresence,
    ) -> Result<TargetOutput> {
        let encoded = self.encode_target_request(request)?;
        let mut bytes = Vec::new();
        let allow_dspark =
            encoded.chat_request || self.dspark.mode() == crate::dspark_runtime::Mode::On;
        let generation = self.generate_tokens_to_with_interrupt(
            &encoded.prompt,
            Some(encoded.maximum_new),
            false,
            false,
            allow_dspark,
            encoded.chat_request,
            &ClientPresenceInterrupt(client),
            None,
            &mut bytes,
        )?;
        if generation.stop == StopReason::Interrupted {
            return Err(client_disconnect_error());
        }
        let raw_text = String::from_utf8(bytes).map_err(|_| {
            DeltafinError::new("native incremental output produced invalid UTF-8 at publication")
        })?;
        let (finish_reason, usage) = target_generation_metadata(&encoded.prompt, &generation)
            .inspect_err(|_error| {
                self.lifecycle.poison();
            })?;
        let mut output = if encoded.chat_request {
            let (reasoning, content) = split_chat_output(raw_text, finish_reason);
            let mut output = TargetOutput::target_verified(content, finish_reason);
            if let Some(reasoning) = reasoning {
                output = output.with_reasoning_content(reasoning);
            }
            output
        } else {
            TargetOutput::target_verified(raw_text, finish_reason)
        };
        output = output.with_usage(usage);
        Ok(output)
    }

    fn finish_target_response(&mut self, publication: StreamPublication) {
        self.settle_target_publication(publication);
    }

    fn generate_target_stream(
        &mut self,
        request: &TargetRequest,
        sink: &mut dyn TargetDeltaSink,
        client: &ClientPresence,
    ) -> std::result::Result<TargetStreamSummary, StreamGenerationError> {
        if self.stream_boundary != NativeStreamBoundary::None {
            return Err(StreamGenerationError::Target(DeltafinError::new(
                "native target stream has an unresolved response boundary",
            )));
        }
        let encoded = self.encode_target_request(request)?;
        let mut writer = TargetStreamWriter::new(sink, encoded.chat_request);
        let allow_dspark =
            encoded.chat_request || self.dspark.mode() == crate::dspark_runtime::Mode::On;
        let generated = self.generate_tokens_to_with_interrupt(
            &encoded.prompt,
            Some(encoded.maximum_new),
            false,
            false,
            allow_dspark,
            encoded.chat_request,
            &ClientPresenceInterrupt(client),
            None,
            &mut writer,
        );

        let generation = match generated {
            Ok(generation) => generation,
            Err(target_error) => {
                if let Some(publication_error) = writer.take_publication_error() {
                    // `generate_tokens_to` fail-closes on every `Write` error.
                    // This adapter alone can prove that this particular error
                    // originated in the client-owned delta sink, so defer the
                    // independent-boundary reset to `finish_target_stream`.
                    self.stream_boundary = NativeStreamBoundary::PublicationFailed;
                    return Err(StreamGenerationError::Publication(publication_error));
                }
                return Err(StreamGenerationError::Target(target_error));
            }
        };
        if generation.stop == StopReason::Interrupted {
            // Leave the stream boundary at `None`: the interrupt tail already
            // discarded staged state, so `finish_target_stream(Aborted)` must
            // resolve to a no-op rather than a second discard.
            return Err(StreamGenerationError::Target(client_disconnect_error()));
        }

        let (finish_reason, usage) = match target_generation_metadata(&encoded.prompt, &generation)
        {
            Ok(metadata) => metadata,
            Err(error) => {
                self.lifecycle.poison();
                return Err(StreamGenerationError::Target(error));
            }
        };
        if let Err(publication_error) = writer.finish(finish_reason) {
            self.stream_boundary = NativeStreamBoundary::PublicationFailed;
            return Err(StreamGenerationError::Publication(publication_error));
        }
        self.stream_boundary = NativeStreamBoundary::AwaitingPublication;
        Ok(TargetStreamSummary::target_verified(finish_reason).with_usage(usage))
    }

    fn finish_target_stream(&mut self, publication: StreamPublication) {
        match self.stream_boundary.resolve(publication) {
            NativeStreamResolution::None => {}
            NativeStreamResolution::Preserve => {
                self.settle_target_publication(StreamPublication::Complete);
                if self.lifecycle != NativeEngineLifecycle::Ready {
                    self.lifecycle.poison();
                }
            }
            NativeStreamResolution::Discard => {
                let abandoned_layer = self.spine_pipeline.discard_pending();
                self.output_decoder.reset();
                self.settle_target_publication(StreamPublication::Aborted);
                if abandoned_layer.is_none() {
                    self.lifecycle = NativeEngineLifecycle::Ready;
                } else {
                    self.lifecycle.poison();
                }
            }
            NativeStreamResolution::Poison => {
                self.settle_target_publication(StreamPublication::Aborted);
                self.lifecycle.poison();
            }
        }
    }
}

struct EncodedTargetRequest {
    prompt: Vec<u32>,
    maximum_new: u64,
    chat_request: bool,
}

impl NativeTargetEngine {
    fn encode_target_request(&self, request: &TargetRequest) -> Result<EncodedTargetRequest> {
        let (prompt, chat_request) = match &request.prompt {
            TargetPrompt::Completion(text) => (self.tokenizer.encode_ordinary(text)?, false),
            TargetPrompt::Chat(messages) => {
                let messages = messages
                    .iter()
                    .map(|message| {
                        let mut object = message.additional_fields.clone();
                        object.insert("role".into(), Value::String(message.role.clone()));
                        object.insert("content".into(), message.content.clone());
                        Value::Object(object)
                    })
                    .collect::<Vec<_>>();
                // Per-request effort wins; the engine's configured default is
                // the fallback; both absent leaves the template's own `max`.
                let mut options = ChatOptions::default();
                if let Some(effort) = request
                    .reasoning_effort
                    .as_deref()
                    .or(self.reasoning_effort.as_deref())
                {
                    options.thinking_effort = Some(effort);
                }
                (encode_chat(&self.tokenizer, &messages, &options)?, true)
            }
        };
        let maximum_new = u64::try_from(request.max_new_tokens)
            .map_err(|_| DeltafinError::new("server max-token request does not fit u64"))?;
        Ok(EncodedTargetRequest {
            prompt,
            maximum_new,
            chat_request,
        })
    }
}

fn target_generation_metadata(
    prompt: &[u32],
    generation: &NativeGeneration,
) -> Result<(FinishReason, TokenUsage)> {
    let finish_reason = finish_reason_for_stop(generation.stop);
    // Match the established server contract: EOS terminates generation but is
    // not part of the public completion token tape or its usage count.
    let completion_token_count =
        if generation.stop == StopReason::Eos {
            generation.token_ids.len().checked_sub(1).ok_or_else(|| {
                DeltafinError::new("EOS-terminated generation has no terminal token")
            })?
        } else {
            generation.token_ids.len()
        };
    let completion_tokens = u64::try_from(completion_token_count)
        .map_err(|_| DeltafinError::new("completion token count does not fit u64"))?;
    let prompt_tokens = u64::try_from(prompt.len())
        .map_err(|_| DeltafinError::new("prompt token count does not fit u64"))?;
    Ok((
        finish_reason,
        TokenUsage {
            prompt_tokens,
            completion_tokens,
        },
    ))
}

const fn finish_reason_for_stop(stop: StopReason) -> FinishReason {
    match stop {
        StopReason::Eos => FinishReason::Stop,
        StopReason::MaxNew | StopReason::ContextFull => FinishReason::Length,
        // The server adapter rejects interrupted generations before mapping a
        // finish reason, so this arm serves only the CLI's partial-result
        // reporting; retain a total mapping for the shared result type.
        StopReason::Interrupted => FinishReason::Stop,
    }
}

const THINK_RESPONSE_MARKER: &str = "<|close|>think<|sep|><|open|>response<|sep|>";
const THINK_CLOSE_MARKER: &str = "<|close|>think<|sep|>";
const RESPONSE_MARKER: &str = "<|open|>response<|sep|>";
const ASSISTANT_CLOSE_MARKER: &str = "<|close|>response<|sep|><|close|>message<|sep|>";
const THINK_OPEN_MARKER: &str = "<|open|>think<|sep|>";
const END_OF_MESSAGE_MARKER: &str = "<|end_of_msg|>";
const CHAT_CONTROL_MARKERS: &[&str] = &[
    ASSISTANT_CLOSE_MARKER,
    THINK_RESPONSE_MARKER,
    THINK_CLOSE_MARKER,
    RESPONSE_MARKER,
    THINK_OPEN_MARKER,
    END_OF_MESSAGE_MARKER,
];

fn canonical_chat_target_boundary(
    tokenizer: &K3Tokenizer,
    eos_token: u32,
    _prompt: &[u32],
    generated: &[u32],
    stop: StopReason,
) -> bool {
    if stop != StopReason::Eos || generated.last().copied() != Some(eos_token) {
        return false;
    }
    let payload = match tokenizer
        .decode_bytes(&generated[..generated.len().saturating_sub(1)])
        .and_then(|bytes| {
            String::from_utf8(bytes)
                .map_err(|_| DeltafinError::new("generated chat boundary is not valid UTF-8"))
        }) {
        Ok(payload) => payload,
        Err(_) => return false,
    };
    let Some(body) = payload.strip_suffix(ASSISTANT_CLOSE_MARKER) else {
        return false;
    };
    let Some((reasoning, content)) = body.split_once(THINK_RESPONSE_MARKER) else {
        return false;
    };
    if content.contains(THINK_RESPONSE_MARKER)
        || (!reasoning.is_empty() && reasoning.trim().is_empty())
    {
        return false;
    }
    let mut expected = Vec::with_capacity(generated.len());
    for (text, structural) in [
        (reasoning, false),
        (THINK_RESPONSE_MARKER, true),
        (content, false),
        (ASSISTANT_CLOSE_MARKER, true),
    ] {
        if tokenizer
            .encode_into(text, structural, &mut expected)
            .is_err()
        {
            return false;
        }
    }
    expected.push(eos_token);
    expected == generated
}

struct TargetStreamWriter<'a> {
    sink: &'a mut dyn TargetDeltaSink,
    output: TargetStreamOutput,
    publication_error: Option<io::Error>,
}

enum TargetStreamOutput {
    Completion,
    Chat(ChatStreamParser),
}

/// Terminal output adapter for the native CLI. Raw completion mode remains a
/// direct byte stream. Chat mode uses the same reviewed XTML response parser
/// as the OpenAI server and publishes only the assistant's public content;
/// reasoning and structural control markers never reach stdout.
struct CliOutputWriter<W> {
    sink: W,
    chat: Option<ChatStreamParser>,
    wrote_public_text: bool,
}

impl<W: Write> CliOutputWriter<W> {
    fn new(sink: W, chat: bool) -> Self {
        Self {
            sink,
            chat: chat.then(ChatStreamParser::default),
            wrote_public_text: false,
        }
    }

    fn publish(&mut self, deltas: Vec<ChatTextDelta>) -> io::Result<()> {
        for delta in deltas {
            if let ChatTextDelta::Content(text) = delta
                && !text.is_empty()
            {
                self.sink.write_all(text.as_bytes())?;
                self.wrote_public_text = true;
            }
        }
        Ok(())
    }

    fn finish(&mut self, finish_reason: FinishReason) -> io::Result<()> {
        if let Some(parser) = &mut self.chat {
            let deltas = parser.finish(finish_reason);
            self.publish(deltas)?;
        }
        self.sink.flush()
    }

    const fn wrote_public_text(&self) -> bool {
        self.wrote_public_text
    }

    fn write_terminal_newline(&mut self) -> io::Result<()> {
        self.sink.write_all(b"\n")
    }
}

impl<W: Write> Write for CliOutputWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.chat.is_none() {
            let written = self.sink.write(bytes)?;
            self.wrote_public_text |= written != 0;
            return Ok(written);
        }
        let text = std::str::from_utf8(bytes).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "native CLI output adapter received a non-UTF-8 write",
            )
        })?;
        let deltas = self.chat.as_mut().expect("checked chat parser").push(text);
        self.publish(deltas)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Do not force a partial structural marker out of the parser. Public
        // content already emitted by `write` may still be flushed normally.
        self.sink.flush()
    }
}

impl<'a> TargetStreamWriter<'a> {
    fn new(sink: &'a mut dyn TargetDeltaSink, chat_request: bool) -> Self {
        Self {
            sink,
            output: if chat_request {
                TargetStreamOutput::Chat(ChatStreamParser::default())
            } else {
                TargetStreamOutput::Completion
            },
            publication_error: None,
        }
    }

    fn publish(&mut self, deltas: Vec<ChatTextDelta>) -> io::Result<()> {
        for delta in deltas {
            let target_delta = match delta {
                ChatTextDelta::Reasoning(text) => TargetDelta::target_verified_reasoning(text),
                ChatTextDelta::Content(text) => TargetDelta::target_verified_content(text),
            };
            if let Err(error) = self.sink.publish_target_delta(target_delta) {
                let returned = io::Error::new(error.kind(), error.to_string());
                self.publication_error = Some(error);
                return Err(returned);
            }
        }
        Ok(())
    }

    fn finish(&mut self, finish_reason: FinishReason) -> io::Result<()> {
        let deltas = match &mut self.output {
            TargetStreamOutput::Completion => Vec::new(),
            TargetStreamOutput::Chat(parser) => parser.finish(finish_reason),
        };
        if let Err(error) = self.publish(deltas) {
            return Err(self.publication_error.take().unwrap_or(error));
        }
        Ok(())
    }

    fn take_publication_error(&mut self) -> Option<io::Error> {
        self.publication_error.take()
    }
}

impl Write for TargetStreamWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let text = std::str::from_utf8(bytes).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "native output adapter received a non-UTF-8 write",
            )
        })?;
        let deltas = match &mut self.output {
            TargetStreamOutput::Completion => (!text.is_empty())
                .then(|| ChatTextDelta::Content(text.to_owned()))
                .into_iter()
                .collect(),
            TargetStreamOutput::Chat(parser) => parser.push(text),
        };
        self.publish(deltas)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Each sink publication owns and flushes its complete SSE frame. A
        // parser-only flush must not expose a possible control-marker prefix.
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
enum ChatTextDelta {
    Reasoning(String),
    Content(String),
}

#[derive(Debug, Default)]
struct ChatStreamParser {
    phase: ChatParsePhase,
}

#[derive(Debug, Default)]
enum ChatParsePhase {
    #[default]
    AwaitingResponse,
    AwaitingResponseText(String),
    Response(String),
    Closed,
}

impl ChatStreamParser {
    fn push(&mut self, text: &str) -> Vec<ChatTextDelta> {
        let phase = std::mem::replace(&mut self.phase, ChatParsePhase::Closed);
        let mut buffered = match phase {
            ChatParsePhase::AwaitingResponse => String::new(),
            ChatParsePhase::AwaitingResponseText(buffered) => buffered,
            ChatParsePhase::Response(rolling) => {
                let (phase, deltas) = push_response_text(rolling, text, None);
                self.phase = phase;
                return deltas;
            }
            ChatParsePhase::Closed => {
                self.phase = ChatParsePhase::Closed;
                return Vec::new();
            }
        };
        buffered.push_str(text);
        let Some(marker_start) = buffered.find(RESPONSE_MARKER) else {
            self.phase = ChatParsePhase::AwaitingResponseText(buffered);
            return Vec::new();
        };

        let response_start = marker_start + RESPONSE_MARKER.len();
        let response = buffered[response_start..].to_owned();
        let mut reasoning = buffered[..marker_start].to_owned();
        if let Some(without_close) = reasoning.strip_suffix(THINK_CLOSE_MARKER) {
            reasoning.truncate(without_close.len());
        }
        reasoning = sanitize_chat_control_text(reasoning);
        let mut deltas = Vec::new();
        if !reasoning.is_empty() {
            deltas.push(ChatTextDelta::Reasoning(reasoning));
        }
        let (phase, mut response_deltas) = push_response_text(String::new(), &response, None);
        self.phase = phase;
        deltas.append(&mut response_deltas);
        deltas
    }

    fn finish(&mut self, finish_reason: FinishReason) -> Vec<ChatTextDelta> {
        let phase = std::mem::replace(&mut self.phase, ChatParsePhase::Closed);
        match phase {
            ChatParsePhase::AwaitingResponse => Vec::new(),
            ChatParsePhase::AwaitingResponseText(buffered) => {
                finish_without_response_marker(buffered, finish_reason)
            }
            ChatParsePhase::Response(rolling) => {
                push_response_text(rolling, "", Some(finish_reason)).1
            }
            ChatParsePhase::Closed => Vec::new(),
        }
    }
}

fn push_response_text(
    mut rolling: String,
    text: &str,
    finish_reason: Option<FinishReason>,
) -> (ChatParsePhase, Vec<ChatTextDelta>) {
    rolling.push_str(text);
    if let Some(finish_reason) = finish_reason {
        if finish_reason == FinishReason::Stop && rolling == ASSISTANT_CLOSE_MARKER {
            return (ChatParsePhase::Closed, Vec::new());
        }
        if finish_reason == FinishReason::Length && rolling == ASSISTANT_CLOSE_MARKER {
            return (
                ChatParsePhase::Closed,
                vec![ChatTextDelta::Content(rolling)],
            );
        }
        let content = sanitize_chat_control_text(rolling);
        return (
            ChatParsePhase::Closed,
            (!content.is_empty())
                .then_some(ChatTextDelta::Content(content))
                .into_iter()
                .collect(),
        );
    }

    let mut emitted = String::new();
    while let Some((marker_start, marker)) = first_chat_control_marker(&rolling) {
        emitted.push_str(&rolling[..marker_start]);
        if marker == ASSISTANT_CLOSE_MARKER {
            let marker_end = marker_start + marker.len();
            if marker_end == rolling.len() {
                // A close marker is structural only when EOS immediately
                // follows. Hold it until terminal state proves that. If K3
                // continues, the marker was ordinary answer text and must be
                // preserved rather than truncating everything after it.
                rolling = marker.to_owned();
                break;
            }
            emitted.push_str(marker);
            rolling = rolling[marker_end..].to_owned();
            continue;
        }
        rolling = rolling[marker_start + marker.len()..].to_owned();
    }

    if rolling == ASSISTANT_CLOSE_MARKER {
        return (
            ChatParsePhase::Response(rolling),
            (!emitted.is_empty())
                .then_some(ChatTextDelta::Content(emitted))
                .into_iter()
                .collect(),
        );
    }

    let retained = longest_control_prefix_suffix(&rolling, CHAT_CONTROL_MARKERS);
    let emitted_end = rolling.len() - retained;
    emitted.push_str(&rolling[..emitted_end]);
    let suffix = rolling[emitted_end..].to_owned();
    (
        ChatParsePhase::Response(suffix),
        (!emitted.is_empty())
            .then_some(ChatTextDelta::Content(emitted))
            .into_iter()
            .collect(),
    )
}

fn first_chat_control_marker(text: &str) -> Option<(usize, &'static str)> {
    CHAT_CONTROL_MARKERS
        .iter()
        .filter_map(|marker| text.find(marker).map(|position| (position, *marker)))
        .min_by_key(|(position, marker)| (*position, std::cmp::Reverse(marker.len())))
}

fn finish_without_response_marker(
    buffered: String,
    finish_reason: FinishReason,
) -> Vec<ChatTextDelta> {
    if finish_reason == FinishReason::Length {
        // K3's generation prompt has already opened the private `think`
        // channel. Until the complete think->response delimiter appears,
        // length-capped output is unfinished reasoning, not public answer
        // text. Keep this rule identical for JSON and live SSE publication.
        let reasoning = sanitize_chat_control_text(buffered);
        return (!reasoning.is_empty())
            .then_some(ChatTextDelta::Reasoning(reasoning))
            .into_iter()
            .collect();
    }

    if let Some(marker_start) = buffered.find(THINK_CLOSE_MARKER) {
        let reasoning = sanitize_chat_control_text(buffered[..marker_start].to_owned());
        let content = sanitize_chat_control_text(
            buffered[marker_start + THINK_CLOSE_MARKER.len()..].to_owned(),
        );
        let mut deltas = Vec::new();
        if !reasoning.is_empty() {
            deltas.push(ChatTextDelta::Reasoning(reasoning));
        }
        if !content.is_empty() {
            deltas.push(ChatTextDelta::Content(content));
        }
        return deltas;
    }

    // Without a complete channel marker there is no evidence that an
    // otherwise ordinary completion is private reasoning. Keep it public,
    // while still suppressing any complete or truncated K3 control suffix.
    let content = sanitize_chat_control_text(buffered);
    (!content.is_empty())
        .then_some(ChatTextDelta::Content(content))
        .into_iter()
        .collect()
}

fn sanitize_chat_control_text(mut text: String) -> String {
    for marker in CHAT_CONTROL_MARKERS {
        text = text.replace(marker, "");
    }
    let retained = longest_control_prefix_suffix(&text, CHAT_CONTROL_MARKERS);
    text.truncate(text.len() - retained);
    text
}

fn longest_control_prefix_suffix(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|marker| 1..marker.len())
        .filter(|&length| length <= text.len())
        .filter(|&length| {
            text.get(text.len() - length..)
                .is_some_and(|suffix| markers.iter().any(|marker| marker.starts_with(suffix)))
        })
        .max()
        .unwrap_or(0)
}

fn split_chat_output(text: String, finish: FinishReason) -> (Option<String>, String) {
    let mut parser = ChatStreamParser::default();
    let mut deltas = parser.push(&text);
    deltas.extend(parser.finish(finish));
    let mut reasoning = String::new();
    let mut content = String::new();
    for delta in deltas {
        match delta {
            ChatTextDelta::Reasoning(text) => reasoning.push_str(&text),
            ChatTextDelta::Content(text) => content.push_str(&text),
        }
    }
    ((!reasoning.is_empty()).then_some(reasoning), content)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct TargetReuseIdentity {
    model_inventory: [u8; 32],
    device: Device,
    spine: SpineRepresentation,
    expert_backend: ResolvedExpertBackend,
    expert_storage: ExpertStorageLayout,
    expert_cpu_threads: usize,
    provider_abi: u32,
    transaction_contract: u32,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PublishedTargetBoundary {
    identity: TargetReuseIdentity,
    logical_tokens: Box<[u32]>,
    committed_positions: usize,
    cache_generation: u64,
    pending_token: u32,
    boundary_id: BoundaryId,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum TargetReuseInvalidation {
    UnsupportedCapability,
    NoPublishedBoundary,
    InvalidPublishedBoundary,
    ModelOrConfigChanged,
    PromptTruncated,
    PromptDiverged,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum TargetReusePlan {
    Reset(TargetReuseInvalidation),
    Reuse {
        expected_committed_positions: usize,
        expected_cache_generation: u64,
        /// Starts with the prior full-K3 pending token, which has not yet been
        /// consumed by the provider, followed by any new prompt suffix.
        replay: Range<usize>,
    },
}

fn plan_target_reuse(
    capability: TargetStateTransactionCapability,
    boundary: Option<&PublishedTargetBoundary>,
    identity: TargetReuseIdentity,
    prompt: &[u32],
) -> TargetReusePlan {
    if capability != TargetStateTransactionCapability::RequestBranchV1 {
        return TargetReusePlan::Reset(TargetReuseInvalidation::UnsupportedCapability);
    }
    let Some(boundary) = boundary else {
        return TargetReusePlan::Reset(TargetReuseInvalidation::NoPublishedBoundary);
    };
    if boundary.logical_tokens.is_empty()
        || boundary.committed_positions.checked_add(1) != Some(boundary.logical_tokens.len())
        || boundary.logical_tokens.last().copied() != Some(boundary.pending_token)
        || boundary.cache_generation == 0
    {
        return TargetReusePlan::Reset(TargetReuseInvalidation::InvalidPublishedBoundary);
    }
    if boundary.identity != identity {
        return TargetReusePlan::Reset(TargetReuseInvalidation::ModelOrConfigChanged);
    }
    if prompt.len() <= boundary.logical_tokens.len() {
        return TargetReusePlan::Reset(TargetReuseInvalidation::PromptTruncated);
    }
    if !prompt.starts_with(&boundary.logical_tokens) {
        return TargetReusePlan::Reset(TargetReuseInvalidation::PromptDiverged);
    }
    TargetReusePlan::Reuse {
        expected_committed_positions: boundary.committed_positions,
        expected_cache_generation: boundary.cache_generation,
        replay: boundary.committed_positions..prompt.len(),
    }
}

fn consume_target_reuse_slot(
    slot: &mut Option<PublishedTargetBoundary>,
    capability: TargetStateTransactionCapability,
    identity: TargetReuseIdentity,
    prompt: &[u32],
) -> (Option<PublishedTargetBoundary>, TargetReusePlan) {
    let published = slot.take();
    let plan = plan_target_reuse(capability, published.as_ref(), identity, prompt);
    (published, plan)
}

struct PendingTargetPublication {
    branch: TargetStateBranch,
    prompt: PublishedTargetBoundary,
    final_tokens: Option<Box<[u32]>>,
    dspark_lease: Option<DraftLease>,
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn completion_token_ids(token_ids: &[u32], eos_token: u32) -> &[u32] {
    token_ids.strip_suffix(&[eos_token]).unwrap_or(token_ids)
}

const fn stop_reason_name(stop: StopReason) -> &'static str {
    match stop {
        StopReason::Eos => "eos",
        StopReason::MaxNew => "max_new",
        StopReason::ContextFull => "context_full",
        StopReason::Interrupted => "interrupted",
    }
}

const fn run_status_name(stop: StopReason) -> &'static str {
    match stop {
        StopReason::Interrupted => "interrupted",
        StopReason::Eos | StopReason::MaxNew | StopReason::ContextFull => "ok",
    }
}

fn emit_terminal_run_error(
    events: Option<&mut RunEventLog>,
    phase: &str,
    error: &DeltafinError,
    started: Instant,
) {
    if let Some(events) = events {
        let _ = events.emit_run_error(
            phase,
            "DeltafinError",
            &error.to_string(),
            elapsed_ns(started),
        );
    }
}

#[derive(Debug)]
struct PreparedTargetChunk {
    sequence: TargetSequence,
    predictions: Box<[u32]>,
    stats: Option<TargetSequenceStats>,
    profile: Option<TargetExecutionProfile>,
    dspark_rows: Option<ProviderTensor>,
    dspark_capture_requested: bool,
    input_positions: usize,
    next_mla_capacity: u64,
    mode: TargetSequenceMode,
}

#[derive(Debug)]
struct CompletedTargetChunk {
    predictions: Box<[u32]>,
    stats: Option<TargetSequenceStats>,
    profile: Option<TargetExecutionProfile>,
    commit: TargetSequenceCommit,
    dspark_rows: Option<ProviderTensor>,
    dspark_capture_requested: bool,
}

const TARGET_PROFILE_LAYER_COUNT: usize = 93;

/// Host-observed timing around existing synchronous boundaries. No field is a
/// GPU timestamp and collecting it never inserts a device synchronization.
/// `spine_read_active_ns` is informational overlap time and is deliberately
/// excluded from the additive phase sum; `spine_read_wait_ns` is the portion
/// that actually blocked the target thread.
#[derive(Debug, Clone, Copy, Default)]
struct TargetLayerPhaseProfile {
    passes: u64,
    spine_read_bytes: u64,
    spine_read_active_ns: u64,
    spine_read_wait_ns: u64,
    spine_prefetch_submit_ns: u64,
    spine_bind_upload_ns: u64,
    attention_resident_compute_ns: u64,
    authoritative_expert_read_prefetch_ns: u64,
    expert_kernel_ns: u64,
    source_fence_ns: u64,
    layer_total_ns: u64,
    /// Routed experts already resident in the provider's CUDA cache when the
    /// tile was planned. Zero on the CPU/Metal paths, which have no plan.
    expert_plan_hits: u64,
    /// Routed experts the plan required this pass to read and upload.
    expert_plan_misses: u64,
    /// Experts whose matmul was started early, while the rest of the layer's
    /// misses were still being read.
    expert_early_drained: u64,
}

impl TargetLayerPhaseProfile {
    fn absorb(&mut self, other: &Self) {
        self.passes = self.passes.saturating_add(other.passes);
        self.spine_read_bytes = self.spine_read_bytes.saturating_add(other.spine_read_bytes);
        self.spine_read_active_ns = self
            .spine_read_active_ns
            .saturating_add(other.spine_read_active_ns);
        self.spine_read_wait_ns = self
            .spine_read_wait_ns
            .saturating_add(other.spine_read_wait_ns);
        self.spine_prefetch_submit_ns = self
            .spine_prefetch_submit_ns
            .saturating_add(other.spine_prefetch_submit_ns);
        self.spine_bind_upload_ns = self
            .spine_bind_upload_ns
            .saturating_add(other.spine_bind_upload_ns);
        self.attention_resident_compute_ns = self
            .attention_resident_compute_ns
            .saturating_add(other.attention_resident_compute_ns);
        self.authoritative_expert_read_prefetch_ns = self
            .authoritative_expert_read_prefetch_ns
            .saturating_add(other.authoritative_expert_read_prefetch_ns);
        self.expert_kernel_ns = self.expert_kernel_ns.saturating_add(other.expert_kernel_ns);
        self.source_fence_ns = self.source_fence_ns.saturating_add(other.source_fence_ns);
        self.layer_total_ns = self.layer_total_ns.saturating_add(other.layer_total_ns);
        self.expert_plan_hits = self.expert_plan_hits.saturating_add(other.expert_plan_hits);
        self.expert_plan_misses = self
            .expert_plan_misses
            .saturating_add(other.expert_plan_misses);
        self.expert_early_drained = self
            .expert_early_drained
            .saturating_add(other.expert_early_drained);
    }

    fn attributed_ns(&self) -> u64 {
        self.spine_read_wait_ns
            .saturating_add(self.spine_prefetch_submit_ns)
            .saturating_add(self.spine_bind_upload_ns)
            .saturating_add(self.attention_resident_compute_ns)
            .saturating_add(self.authoritative_expert_read_prefetch_ns)
            .saturating_add(self.expert_kernel_ns)
            .saturating_add(self.source_fence_ns)
    }

    fn other_control_ns(&self) -> u64 {
        self.layer_total_ns.saturating_sub(self.attributed_ns())
    }

    fn json(&self, layer: usize) -> Value {
        serde_json::json!({
            "layer": layer,
            "passes": self.passes,
            "layer_total_ns": self.layer_total_ns,
            "spine_read_bytes": self.spine_read_bytes,
            "spine_read_active_ns": self.spine_read_active_ns,
            "spine_read_wait_ns": self.spine_read_wait_ns,
            "spine_prefetch_submit_ns": self.spine_prefetch_submit_ns,
            "spine_bind_upload_ns": self.spine_bind_upload_ns,
            "attention_resident_compute_ns": self.attention_resident_compute_ns,
            "authoritative_expert_read_prefetch_ns": self.authoritative_expert_read_prefetch_ns,
            "expert_kernel_ns": self.expert_kernel_ns,
            "source_fence_ns": self.source_fence_ns,
            "other_control_ns": self.other_control_ns(),
            "expert_plan_hits": self.expert_plan_hits,
            "expert_plan_misses": self.expert_plan_misses,
            "expert_early_drained": self.expert_early_drained,
        })
    }
}

#[derive(Debug, Clone)]
struct TargetExecutionProfile {
    chunks: u64,
    sequence_total_ns: u64,
    tail_head_sync_ns: u64,
    layers: [TargetLayerPhaseProfile; TARGET_PROFILE_LAYER_COUNT],
}

impl Default for TargetExecutionProfile {
    fn default() -> Self {
        Self {
            chunks: 0,
            sequence_total_ns: 0,
            tail_head_sync_ns: 0,
            layers: [TargetLayerPhaseProfile::default(); TARGET_PROFILE_LAYER_COUNT],
        }
    }
}

impl TargetExecutionProfile {
    fn absorb(&mut self, other: &Self) {
        self.chunks = self.chunks.saturating_add(other.chunks);
        self.sequence_total_ns = self
            .sequence_total_ns
            .saturating_add(other.sequence_total_ns);
        self.tail_head_sync_ns = self
            .tail_head_sync_ns
            .saturating_add(other.tail_head_sync_ns);
        for (target, source) in self.layers.iter_mut().zip(other.layers.iter()) {
            target.absorb(source);
        }
    }

    fn layer_totals(&self) -> TargetLayerPhaseProfile {
        let mut total = TargetLayerPhaseProfile::default();
        for layer in &self.layers {
            total.absorb(layer);
        }
        total
    }

    fn json(&self) -> Value {
        let totals = self.layer_totals();
        let sequence_control_ns = self
            .sequence_total_ns
            .saturating_sub(totals.layer_total_ns)
            .saturating_sub(self.tail_head_sync_ns);
        serde_json::json!({
            "schema": "deltafin.target_phase_profile.v1",
            "clock": "host_monotonic",
            "device_synchronizations_added": 0,
            "chunks": self.chunks,
            "totals": {
                "sequence_total_ns": self.sequence_total_ns,
                "layer_total_ns": totals.layer_total_ns,
                "spine_read_bytes": totals.spine_read_bytes,
                "spine_read_active_ns": totals.spine_read_active_ns,
                "spine_read_wait_ns": totals.spine_read_wait_ns,
                "spine_prefetch_submit_ns": totals.spine_prefetch_submit_ns,
                "spine_bind_upload_ns": totals.spine_bind_upload_ns,
                "attention_resident_compute_ns": totals.attention_resident_compute_ns,
                "authoritative_expert_read_prefetch_ns": totals.authoritative_expert_read_prefetch_ns,
                "expert_kernel_ns": totals.expert_kernel_ns,
                "source_fence_ns": totals.source_fence_ns,
                "layer_other_control_ns": totals.other_control_ns(),
                "tail_head_sync_ns": self.tail_head_sync_ns,
                "sequence_control_ns": sequence_control_ns,
                "expert_plan_hits": totals.expert_plan_hits,
                "expert_plan_misses": totals.expert_plan_misses,
            },
            "layers": self.layers.iter().enumerate()
                .filter(|(_, profile)| profile.passes != 0)
                .map(|(layer, profile)| profile.json(layer))
                .collect::<Vec<_>>(),
        })
    }
}

/// Bytes the model readers have physically read from storage.
#[derive(Debug, Default, Clone, Copy)]
struct StorageBytes {
    spine: u64,
    expert_demand: u64,
    expert_prefetch: u64,
}

impl StorageBytes {
    fn since(self, start: Self) -> Self {
        Self {
            spine: self.spine.saturating_sub(start.spine),
            expert_demand: self.expert_demand.saturating_sub(start.expert_demand),
            expert_prefetch: self.expert_prefetch.saturating_sub(start.expert_prefetch),
        }
    }
}

#[derive(Debug, Default)]
struct NativeRunCounters {
    /// Reader byte counts when this run started; `--stats` reports the
    /// difference, so a long-lived server reports per request.
    storage_at_start: StorageBytes,
    target_chunks: u64,
    committed_positions: u64,
    streamed_layer_passes: u64,
    expert_rows: u64,
    expert_tiles: u64,
    generated_tokens: u64,
    verify_transactions: u64,
    verified_draft_tokens: u64,
    accepted_draft_tokens: u64,
    /// Slot proposals verified as one full-commit transaction because their
    /// per-row KDA boundaries did not fit live memory.
    full_commit_slot_verifies: u64,
    target_profile: TargetExecutionProfile,
}

impl NativeRunCounters {
    fn absorb(&mut self, completed: &CompletedTargetChunk) {
        self.target_chunks = self.target_chunks.saturating_add(1);
        self.committed_positions = self
            .committed_positions
            .saturating_add(completed.commit.committed_positions);
        if let Some(stats) = completed.stats {
            self.streamed_layer_passes = self
                .streamed_layer_passes
                .saturating_add(stats.streamed_layer_passes);
            self.expert_rows = self.expert_rows.saturating_add(stats.expert_rows_completed);
            self.expert_tiles = self
                .expert_tiles
                .saturating_add(stats.expert_tiles_completed);
        }
        if let Some(profile) = completed.profile.as_ref() {
            self.target_profile.absorb(profile);
        }
    }

    fn target_profile_json(&self) -> Option<Value> {
        (self.target_profile.chunks != 0).then(|| self.target_profile.json())
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum CompleteExpertUnion {
    Disabled,
    Dynamic,
    Reserved(usize),
}

impl CompleteExpertUnion {
    const fn enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ExpertTilePlan {
    first_row: usize,
    row_count: usize,
    canonical_experts: Box<[u16]>,
}

impl ExpertTilePlan {
    fn expert_ids(&self) -> &[u16] {
        &self.canonical_experts
    }

    fn expert_count(&self) -> usize {
        self.canonical_experts.len()
    }
}

fn resolve_metal_source(backend: ResolvedExpertBackend) -> Result<Option<String>> {
    let development_source = std::env::var_os(K3_METAL_DEVELOPMENT_SOURCE_ENV);
    resolve_metal_source_with_policy(backend, development_source.as_deref())
}

pub(crate) fn reject_product_metal_source_override() -> Result<()> {
    if std::env::var_os(K3_METAL_DEVELOPMENT_SOURCE_ENV).is_some() {
        return Err(DeltafinError::new(format!(
            "{K3_METAL_DEVELOPMENT_SOURCE_ENV} is disabled in every Deltafin product build; the reviewed precompiled Metal library embedded in the native executable is mandatory"
        )));
    }
    Ok(())
}

fn resolve_metal_source_with_policy(
    backend: ResolvedExpertBackend,
    development_source: Option<&OsStr>,
) -> Result<Option<String>> {
    if development_source.is_some() {
        return Err(DeltafinError::new(format!(
            "{K3_METAL_DEVELOPMENT_SOURCE_ENV} is disabled in every Deltafin product build; the reviewed precompiled Metal library embedded in the native executable is mandatory"
        )));
    }
    if backend != ResolvedExpertBackend::Metal {
        return Ok(None);
    }
    Ok(Some(K3_METAL_EMBEDDED_SOURCE_V1.to_owned()))
}

fn admit_scale4_storage(
    request: ExpertScale4Request,
    backend: ResolvedExpertBackend,
    metal_qualified: bool,
) -> Result<bool> {
    match request {
        ExpertScale4Request::Off => Ok(false),
        ExpertScale4Request::Auto => Ok(backend == ResolvedExpertBackend::Metal && metal_qualified),
        ExpertScale4Request::Require
            if backend == ResolvedExpertBackend::Metal && metal_qualified =>
        {
            Ok(true)
        }
        ExpertScale4Request::Require if backend != ResolvedExpertBackend::Metal => {
            Err(DeltafinError::new(
                "K3_EXPERT_SCALE4=require needs the qualified Metal expert backend; native CPU and CUDA remain raw-v1",
            ))
        }
        ExpertScale4Request::Require => Err(DeltafinError::new(
            "K3_EXPERT_SCALE4=require needs the Metal scale4-v2 descriptor capability",
        )),
    }
}

/// Page-cache policy for streaming expert reads.
///
/// The 64 GiB reference host must keep purging: one pass streams far more
/// expert bytes than host RAM, and its F_NOCACHE pool design depends on the
/// page cache staying out of the way. Hosts with different RAM/VRAM shapes
/// (typically discrete-GPU workstations) instead leave eviction to the
/// kernel's LRU, so repeatedly routed experts are re-read from RAM rather
/// than disk — the behavior the mature Python pipeline relied on.
fn expert_stream_cache_policy(explicit_nocache: Option<bool>, is_macos: bool) -> CachePolicy {
    let nocache = explicit_nocache.unwrap_or(is_macos);
    if nocache {
        CachePolicy::Streaming
    } else {
        CachePolicy::Resident
    }
}

fn open_expert_corpus(
    request: ExpertScale4Request,
    model_root: &Path,
    backend: ResolvedExpertBackend,
    provider: &NativeProviderSession,
    metal_source_selector: Option<&str>,
    stream_cache_policy: CachePolicy,
) -> Result<RawExpertCorpus> {
    let metal_qualified = if request != ExpertScale4Request::Off
        && backend == ResolvedExpertBackend::Metal
    {
        let selector = metal_source_selector
            .ok_or_else(|| DeltafinError::new("qualified Metal experts need a source selector"))?;
        provider
            .metal_expert_layouts(selector)?
            .supports_scale4_v2()
    } else {
        false
    };
    if admit_scale4_storage(request, backend, metal_qualified)? {
        match RawExpertCorpus::open_auto(
            model_root,
            ExpertStorageLayout::Scale4V2,
            stream_cache_policy,
        ) {
            Ok(corpus) => return Ok(corpus),
            Err(error) if request == ExpertScale4Request::Auto => {
                eprintln!(
                    "[native] exact scale4-v2 expert storage unavailable ({error}); using raw-v1"
                );
            }
            Err(error) => {
                return Err(DeltafinError::new(format!(
                    "required complete authenticated scale4-v2 corpus is unavailable: {error}"
                )));
            }
        }
    }
    RawExpertCorpus::open_auto(model_root, ExpertStorageLayout::RawV1, stream_cache_policy)
}

fn target_global_group(group: u16) -> Result<TargetGlobalGroup> {
    match group {
        1 => Ok(TargetGlobalGroup::Tail),
        2 => Ok(TargetGlobalGroup::LanguageModelHead),
        _ => Err(DeltafinError::new(format!(
            "compiled target globals contain unknown group {group}"
        ))),
    }
}

fn bind_target_globals_once(
    provider: &NativeProviderSession,
    plans: &[GlobalSpinePlan],
) -> Result<()> {
    if plans.len() != crate::program::K3_GLOBAL_TRANSFER_GROUPS {
        return Err(DeltafinError::new(
            "exact target startup needs both immutable global transfer groups",
        ));
    }
    let reader = Reader::with_arena_capacity(bounded_worker_count(SPINE_READER_LIMIT), 1)?;
    for (index, plan) in plans.iter().enumerate() {
        let group = target_global_group(plan.group())?;
        let (buffers, _) = reader.read(plan.read_plan())?;
        let report = provider.bind_target_globals(group, plan.descriptors(), &buffers)?;
        let expected_ready = index + 1;
        if report.group != group || report.groups_ready != expected_ready {
            return Err(DeltafinError::new(format!(
                "target global group {} bound out of order (provider reports {} groups ready)",
                plan.group(),
                report.groups_ready
            )));
        }
    }
    Ok(())
}

fn target_expert_backend(backend: ResolvedExpertBackend) -> TargetExpertBackend {
    match backend {
        ResolvedExpertBackend::Cpu => TargetExpertBackend::Cpu,
        ResolvedExpertBackend::Metal => TargetExpertBackend::Metal,
        ResolvedExpertBackend::CudaAuto => TargetExpertBackend::Auto,
        ResolvedExpertBackend::Cuda => TargetExpertBackend::Cuda,
    }
}

fn configured_expert_cpu_threads() -> Result<usize> {
    let configured = std::env::var_os("K3_GEMV_THREADS")
        .map(|value| {
            value.into_string().map_err(|_| {
                DeltafinError::new("K3_GEMV_THREADS must be valid UTF-8 and a positive integer")
            })
        })
        .transpose()?;
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    resolve_expert_cpu_threads(configured.as_deref(), available, cfg!(target_os = "linux"))
}

fn configured_speculative_max_drafts() -> Result<usize> {
    let configured = std::env::var_os("K3_SPEC_DEPTH")
        .map(|raw| {
            raw.into_string().map_err(|_| {
                DeltafinError::new("K3_SPEC_DEPTH must be valid UTF-8 and an integer in 1..=8")
            })
        })
        .transpose()?;
    resolve_speculative_max_drafts(configured.as_deref())
}

fn configured_complete_expert_union() -> Result<bool> {
    let configured = std::env::var_os("K3_COMPLETE_EXPERT_UNION")
        .map(|raw| {
            raw.into_string().map_err(|_| {
                DeltafinError::new("K3_COMPLETE_EXPERT_UNION must be valid UTF-8 and 0 or 1")
            })
        })
        .transpose()?;
    resolve_complete_expert_union(configured.as_deref())
}

fn resolve_complete_expert_union(configured: Option<&str>) -> Result<bool> {
    match configured.map(str::trim) {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(_) => Err(DeltafinError::new(
            "K3_COMPLETE_EXPERT_UNION must be 0 or 1",
        )),
    }
}

fn resolve_speculative_max_drafts(configured: Option<&str>) -> Result<usize> {
    let Some(raw) = configured else {
        // Capacity is not a promise to verify eight rows every step. Each
        // proposal source starts narrow, widens only after request-local
        // evidence, and still passes the live cache/snapshot admission gate.
        return Ok(MAX_EXACT_DRAFTS);
    };
    let drafts = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| DeltafinError::new("K3_SPEC_DEPTH must be an integer in 1..=8"))?;
    if !(1..=MAX_EXACT_DRAFTS).contains(&drafts) {
        return Err(DeltafinError::new(
            "K3_SPEC_DEPTH must be an integer in 1..=8",
        ));
    }
    Ok(drafts)
}

fn resolve_expert_cpu_threads(
    configured: Option<&str>,
    available: usize,
    linux: bool,
) -> Result<usize> {
    if let Some(raw) = configured {
        let threads = raw.trim().parse::<usize>().map_err(|_| {
            DeltafinError::new("K3_GEMV_THREADS must be a positive integer in 1..=32")
        })?;
        if !(1..=MAX_NATIVE_CPU_EXPERT_THREADS).contains(&threads) {
            return Err(DeltafinError::new(
                "K3_GEMV_THREADS must be a positive integer in 1..=32",
            ));
        }
        return Ok(threads);
    }
    let measured_default = if linux { 8 } else { 4 };
    Ok(available.max(1).min(measured_default))
}

#[allow(clippy::too_many_arguments)]
fn execute_target_sequence(
    sequence: &mut TargetSequence,
    provider: &NativeProviderSession,
    spine_pipeline: &mut SpinePipeline,
    layers: &[LayerSpinePlan],
    experts: &RawExpertCorpus,
    expert_reader: &Reader,
    expert_prefetch_reader: Option<&Reader>,
    mut pilot_gate: Option<&mut PilotGate>,
    expert_heat: &ExpertHeat,
    expert_pin_tier: Option<&ExpertPinTier>,
    router_trace: &mut RouterTrace,
    expert_backend: TargetExpertBackend,
    cpu_threads: usize,
    metal_source_selector: Option<&str>,
    metal_expert_wrapper_retention: bool,
    expert_early_drain: bool,
    complete_expert_union: CompleteExpertUnion,
    collect_stats: bool,
    verbose_layer_profile: bool,
    collect_profile: bool,
) -> Result<(
    Box<[u32]>,
    Option<TargetSequenceStats>,
    Option<TargetExecutionProfile>,
)> {
    if layers.len() != 93 || sequence.next_layer() != 0 || sequence.waiting_for_experts() {
        return Err(DeltafinError::new(
            "native target sequence did not begin at the complete 93-layer boundary",
        ));
    }
    if metal_expert_wrapper_retention != (expert_backend == TargetExpertBackend::Metal) {
        return Err(DeltafinError::new(
            "Metal expert wrapper retention disagrees with its flush-before-retire arena lifecycle",
        ));
    }
    let sequence_started = collect_profile.then(Instant::now);
    let mut profile = collect_profile.then(TargetExecutionProfile::default);
    spine_pipeline.prime(&layers[0])?;
    let mut trace_pass = router_trace.begin_pass();
    // Layer 1 is the one routed layer PILOT can never see coming: its hint
    // would have to come from the dense layer 0, which produces no mailbox.
    // The governor's prev-token predictor covers it from the previous pass's
    // routes, overlapping layer 1's otherwise-cold expert reads with layer
    // 0's compute and both layers' spine binds. Same admission machinery,
    // same fail-soft ticket handling, same authoritative demand path.
    let mut pending_expert_prefetch = match (pilot_gate.as_deref_mut(), expert_prefetch_reader) {
        (Some(gate), Some(reader)) => gate
            .plan_sequence_start()
            .and_then(|plan| try_schedule_expert_prefetch(experts, reader, plan)),
        _ => None,
    };
    for (index, current) in layers.iter().enumerate() {
        let layer_started = collect_profile.then(Instant::now);
        let mut layer_profile = TargetLayerPhaseProfile::default();
        layer_profile.passes = u64::from(collect_profile);
        let next = layers.get(index + 1);
        let binding = if collect_profile {
            spine_pipeline.bind_current_profiled(provider, current, next)?
        } else {
            spine_pipeline.bind_current(provider, current, next)?
        };
        if collect_profile {
            debug_assert!(binding.profiled);
            layer_profile.spine_read_bytes = binding.read.bytes;
            layer_profile.spine_read_active_ns = duration_ns(binding.read.elapsed);
            layer_profile.spine_read_wait_ns = duration_ns(binding.read_wait);
            layer_profile.spine_prefetch_submit_ns = duration_ns(binding.next_prefetch_submit);
            layer_profile.spine_bind_upload_ns = duration_ns(binding.bind_upload);
        }
        if binding.layer != current.layer() || binding.generation == 0 {
            let stale = DeltafinError::new("spine pipeline returned a stale target-layer binding");
            if spine_pipeline.has_borrowed_source_use()
                && let Err(abort_error) = spine_pipeline.abort_active_borrowed_source_use(provider)
            {
                return Err(DeltafinError::new(format!(
                    "{stale}; stale borrowed spine abort also failed: {abort_error}"
                )));
            }
            return Err(stale);
        }
        let prepare_started = collect_profile.then(Instant::now);
        let prepared_layer = sequence.prepare_layer(current.layer(), binding.generation);
        if let Some(started) = prepare_started {
            layer_profile.attention_resident_compute_ns = elapsed_ns(started);
        }
        let layer_result = (|| -> Result<()> {
            match prepared_layer? {
                TargetSequenceLayerPrepare::DenseCompleted { next_layer }
                    if current.layer() == 0 && next_layer == 1 =>
                {
                    Ok(())
                }
                TargetSequenceLayerPrepare::ExpertsRequired(mailbox)
                    if current.layer() != 0
                        && mailbox.layer_index() == current.layer()
                        && mailbox.spine_generation() == binding.generation =>
                {
                    trace_pass.record_mailbox(&mailbox)?;
                    finish_expert_mailbox(
                        sequence,
                        &mailbox,
                        experts,
                        expert_reader,
                        expert_prefetch_reader,
                        &mut pending_expert_prefetch,
                        pilot_gate.as_deref_mut(),
                        expert_heat,
                        expert_pin_tier,
                        expert_backend,
                        cpu_threads,
                        metal_source_selector,
                        metal_expert_wrapper_retention,
                        expert_early_drain,
                        complete_expert_union,
                        collect_profile.then_some(&mut layer_profile),
                    )?;
                    Ok(())
                }
                _ => Err(DeltafinError::new(format!(
                    "target layer {} returned an impossible dense/expert state",
                    current.layer()
                ))),
            }
        })();
        if let Err(error) = layer_result {
            if spine_pipeline.has_borrowed_source_use()
                && let Err(abort_error) =
                    spine_pipeline.abort_borrowed_source_use(provider, binding.generation)
            {
                return Err(DeltafinError::new(format!(
                    "{error}; borrowed spine abort also failed: {abort_error}"
                )));
            }
            return Err(error);
        }
        let source_fence_started = collect_profile.then(Instant::now);
        if spine_pipeline.has_borrowed_source_use() {
            if let Err(seal_error) =
                spine_pipeline.seal_borrowed_source_use(provider, binding.generation)
            {
                let abort = spine_pipeline.abort_borrowed_source_use(provider, binding.generation);
                return Err(match abort {
                    Ok(()) => seal_error,
                    Err(abort_error) => DeltafinError::new(format!(
                        "{seal_error}; borrowed spine abort also failed: {abort_error}"
                    )),
                });
            }
            match spine_pipeline.try_reclaim_borrowed_source_use(provider, binding.generation) {
                Ok(true) => {}
                Ok(false) => {
                    let pending_error = DeltafinError::new(
                        "synchronous CPU spine source use was not reclaimable at the completed layer boundary",
                    );
                    let abort =
                        spine_pipeline.abort_borrowed_source_use(provider, binding.generation);
                    return Err(match abort {
                        Ok(()) => pending_error,
                        Err(abort_error) => DeltafinError::new(format!(
                            "{pending_error}; borrowed spine abort also failed: {abort_error}"
                        )),
                    });
                }
                Err(reclaim_error) => {
                    let abort =
                        spine_pipeline.abort_borrowed_source_use(provider, binding.generation);
                    return Err(match abort {
                        Ok(()) => reclaim_error,
                        Err(abort_error) => DeltafinError::new(format!(
                            "{reclaim_error}; borrowed spine abort also failed: {abort_error}"
                        )),
                    });
                }
            }
        }
        if let Some(started) = source_fence_started {
            layer_profile.source_fence_ns = elapsed_ns(started);
        }
        if let Some(started) = layer_started {
            layer_profile.layer_total_ns = elapsed_ns(started);
            profile.as_mut().expect("profile exists").layers[index].absorb(&layer_profile);
            // Deliberately independent of collect_stats: the cheap cumulative
            // summary line (print_run_stats) needs neither this nor per-layer
            // profile collection at all. This print is the verbose ~93
            // lines/token detail, opt-in via --layer-profile alone.
            if verbose_layer_profile {
                print_target_layer_profile(&layer_profile, index, layers.len());
            }
        }
    }
    if sequence.next_layer() != 93 || sequence.waiting_for_experts() {
        return Err(DeltafinError::new(
            "native target sequence ended before every K3 layer and expert row completed",
        ));
    }
    if let Some(prefetch) = pending_expert_prefetch.take() {
        prefetch.cancel_and_drain();
        return Err(DeltafinError::new(
            "native target ended with an expert prefetch beyond the complete layer roster",
        ));
    }
    let tail_started = collect_profile.then(Instant::now);
    let predictions = sequence.finish_tail()?;
    if let Some(started) = tail_started {
        profile.as_mut().expect("profile exists").tail_head_sync_ns = elapsed_ns(started);
    }
    if let Some(started) = sequence_started {
        let profile = profile.as_mut().expect("profile exists");
        profile.chunks = 1;
        profile.sequence_total_ns = elapsed_ns(started);
    }
    let stats = collect_stats.then(|| sequence.stats()).transpose()?;
    trace_pass.commit()?;
    Ok((predictions, stats, profile))
}

struct ExpertPrefetchSet {
    target_layer: u32,
    tickets: Vec<(u16, ExpertUnionReadTicket)>,
}

impl ExpertPrefetchSet {
    fn cancel_and_drain(self) {
        cancel_and_drain_prefetch_tickets(self.tickets);
    }
}

fn take_due_expert_prefetch(
    pending: &mut Option<ExpertPrefetchSet>,
    current_layer: u32,
) -> Option<ExpertPrefetchSet> {
    pending
        .as_ref()
        .is_some_and(|prefetch| prefetch.target_layer <= current_layer)
        .then(|| pending.take())
        .flatten()
}

fn cancel_and_drain_prefetch_tickets(tickets: Vec<(u16, ExpertUnionReadTicket)>) {
    // Claim every not-yet-started job before draining any active syscall. This
    // prevents an early drain from giving another worker time to begin a
    // loser, and prevents an arena slot from recycling under in-flight I/O.
    for (_, ticket) in &tickets {
        ticket.cancel_unclaimed();
    }
    for (_, ticket) in tickets {
        ticket.drain_cancelled();
    }
}

/// Cancel every queued loser before authoritative work is published, but do
/// not serialize that publication behind a loser which is already inside a
/// syscall. The submit closure owns the independent demand Reader admission;
/// draining afterward preserves every speculative arena lease until its final
/// writer has stopped without delaying the start of useful I/O.
fn cancel_submit_drain<T, R, Cancel, Submit, Drain>(
    tickets: Vec<T>,
    mut cancel: Cancel,
    submit: Submit,
    mut drain: Drain,
) -> Result<R>
where
    Cancel: FnMut(&T),
    Submit: FnOnce() -> Result<R>,
    Drain: FnMut(T),
{
    for ticket in &tickets {
        cancel(ticket);
    }
    let submitted = submit();
    for ticket in tickets {
        drain(ticket);
    }
    submitted
}

fn try_schedule_expert_prefetch(
    experts: &RawExpertCorpus,
    reader: &Reader,
    plan: ExpertPrefetchPlan,
) -> Option<ExpertPrefetchSet> {
    let expected = plan.expert_ids().len();
    if !(K3_EXPERT_TOP_K..=EXPERT_PREFETCH_MAX_EXPERTS).contains(&expected) {
        return None;
    }
    let mut tickets = Vec::with_capacity(expected);
    for &expert in plan.expert_ids() {
        let ticket =
            match experts.try_submit_local_prefetch_one(reader, plan.target_layer(), expert) {
                Ok(Some(ticket))
                    if ticket.layer() == plan.target_layer()
                        && ticket.expert_ids() == [expert]
                        && ticket.layout() == experts.layout() =>
                {
                    ticket
                }
                _ => {
                    cancel_and_drain_prefetch_tickets(tickets);
                    return None;
                }
            };
        tickets.push((expert, ticket));
    }
    if tickets.len() != expected {
        cancel_and_drain_prefetch_tickets(tickets);
        return None;
    }
    Some(ExpertPrefetchSet {
        target_layer: plan.target_layer(),
        tickets,
    })
}

enum ExpertTileLease {
    Contiguous(ExpertUnionReadBatch),
    Scattered {
        layout: ExpertStorageLayout,
        pinned: Vec<(u16, Arc<PageAlignedSpan>)>,
        hits: Vec<ExpertUnionReadBatch>,
        demand: Option<ExpertUnionReadBatch>,
    },
}

impl ExpertTileLease {
    #[allow(clippy::too_many_arguments)]
    fn finish(
        self,
        sequence: &mut TargetSequence,
        mailbox: &TargetSequenceMailbox,
        tile: &ExpertTilePlan,
        backend: TargetExpertBackend,
        cpu_threads: usize,
        metal_source_selector: Option<&str>,
        retain_metal_wrappers: bool,
    ) -> Result<()> {
        match self {
            Self::Contiguous(batch) => sequence.finish_expert_tile(
                mailbox,
                tile.first_row,
                tile.row_count,
                batch.expert_ids(),
                batch.buffers().other(),
                batch.layout(),
                backend,
                cpu_threads,
                metal_source_selector,
                retain_metal_wrappers,
            ),
            Self::Scattered {
                layout,
                pinned,
                hits,
                demand,
            } => {
                let mut hit_spans = Vec::with_capacity(hits.len());
                for batch in &hits {
                    let &[expert] = batch.expert_ids() else {
                        return Err(DeltafinError::new(
                            "expert-prefetch hit batch is not a single-expert span",
                        ));
                    };
                    hit_spans.push((expert, batch.buffers().other()));
                }
                let spans = assemble_expert_spans(
                    tile.expert_ids(),
                    &pinned,
                    &hit_spans,
                    demand
                        .as_ref()
                        .map(|batch| (batch.expert_ids(), batch.buffers().other())),
                    layout.expert_span_bytes(),
                )?;
                sequence.finish_expert_span_tile(
                    mailbox,
                    tile.first_row,
                    tile.row_count,
                    tile.expert_ids(),
                    &spans,
                    layout,
                    backend,
                    cpu_threads,
                    metal_source_selector,
                    retain_metal_wrappers,
                )
            }
        }
    }
}

/// Resolve one span per canonical tile expert from, in order of preference,
/// the permanent pin tier, single-expert prefetch-hit spans, and the
/// expert-major demand slab. Pure assembly: every byte source was already
/// authenticated by the read that produced it, and the caller's canonical ID
/// order is preserved exactly.
fn assemble_expert_spans<'a>(
    tile_expert_ids: &[u16],
    pinned: &'a [(u16, Arc<PageAlignedSpan>)],
    hits: &'a [(u16, &'a [u8])],
    demand: Option<(&'a [u16], &'a [u8])>,
    span_bytes: usize,
) -> Result<Vec<&'a [u8]>> {
    let mut spans = Vec::with_capacity(tile_expert_ids.len());
    for &expert in tile_expert_ids {
        if let Some((_, span)) = pinned.iter().find(|(pinned_id, _)| *pinned_id == expert) {
            if span.len() != span_bytes {
                return Err(DeltafinError::new(
                    "pinned expert span disagrees with the corpus span length",
                ));
            }
            spans.push(&span[..]);
            continue;
        }
        if let Some(&(_, span)) = hits.iter().find(|(hit_id, _)| *hit_id == expert) {
            spans.push(span);
            continue;
        }
        let (demand_ids, demand_bytes) = demand.ok_or_else(|| {
            DeltafinError::new("expert tile merge lost an authoritative demand miss")
        })?;
        spans.push(canonical_expert_span(
            demand_ids,
            demand_bytes,
            span_bytes,
            expert,
        )?);
    }
    Ok(spans)
}

fn canonical_expert_span<'a>(
    canonical_expert_ids: &[u16],
    expert_major_bytes: &'a [u8],
    span_bytes: usize,
    expert: u16,
) -> Result<&'a [u8]> {
    if span_bytes == 0
        || canonical_expert_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || canonical_expert_ids
            .len()
            .checked_mul(span_bytes)
            .is_none_or(|expected| expected != expert_major_bytes.len())
    {
        return Err(DeltafinError::new(
            "expert-major slab does not match its canonical span roster",
        ));
    }
    let index = canonical_expert_ids.binary_search(&expert).map_err(|_| {
        DeltafinError::new("expert-prefetch demand slab lacks an authoritative expert")
    })?;
    let start = index
        .checked_mul(span_bytes)
        .ok_or_else(|| DeltafinError::new("expert-prefetch span offset overflows usize"))?;
    let end = start
        .checked_add(span_bytes)
        .ok_or_else(|| DeltafinError::new("expert-prefetch span end overflows usize"))?;
    expert_major_bytes
        .get(start..end)
        .ok_or_else(|| DeltafinError::new("expert-prefetch demand span exceeds its checked slab"))
}

/// Feed one just-authenticated batch to the tier's sticky promotion. A no-op
/// without a tier; the tier itself narrows to unresolved candidates, so this
/// trails every successful authoritative read at negligible cost.
fn promote_from_batch(
    expert_pin_tier: Option<&ExpertPinTier>,
    layer: u32,
    batch: &ExpertUnionReadBatch,
) {
    if let Some(tier) = expert_pin_tier {
        tier.maybe_promote(layer, batch.expert_ids(), batch.buffers().other());
    }
}

/// Sink for expert spans that have become readable before the rest of the
/// layer has landed.
///
/// This exists purely to pull compute forward into the read window. It is
/// offered a batch at a time, in whatever order storage happens to deliver,
/// and is free to ignore any of it: nothing it does is authoritative, and the
/// layer's reduction still happens once, later, over every expert in the
/// router's order. `false` from `stage` means "stop offering" — the drain
/// then simply waits for the remaining reads as it always did.
trait ExpertDrain {
    fn stage(&mut self, arrived: &[(u16, &[u8])]) -> bool;
}

/// Makes one refusal terminal for the whole tile, so every offer site can
/// ignore the return value and still honor it.
struct LatchingDrain<'a> {
    inner: &'a mut dyn ExpertDrain,
    offering: bool,
}

impl<'a> LatchingDrain<'a> {
    fn new(inner: &'a mut dyn ExpertDrain) -> Self {
        Self {
            inner,
            offering: true,
        }
    }
}

impl ExpertDrain for LatchingDrain<'_> {
    fn stage(&mut self, arrived: &[(u16, &[u8])]) -> bool {
        if !self.offering {
            return false;
        }
        self.offering = self.inner.stage(arrived);
        self.offering
    }
}

/// The established batch-then-compute behavior: nothing is offered early, so
/// every expert's work waits for the layer's whole miss set.
#[cfg_attr(not(test), allow(dead_code))]
struct NoExpertDrain;

impl ExpertDrain for NoExpertDrain {
    fn stage(&mut self, _arrived: &[(u16, &[u8])]) -> bool {
        false
    }
}

/// Hands arrived expert spans to the provider's staging entry point for one
/// tile. Everything it does is a head start on the tile's own finish call,
/// which remains the sole authority for routes, weights, and the ordered
/// reduction — so a tile that stages nothing produces identical bytes.
struct TileExpertDrain<'a> {
    offering: bool,
    sequence: &'a mut TargetSequence,
    mailbox: &'a TargetSequenceMailbox,
    tile: &'a ExpertTilePlan,
    layout: ExpertStorageLayout,
    backend: TargetExpertBackend,
    cpu_threads: usize,
    metal_source_selector: Option<&'a str>,
    drained: u64,
}

impl<'a> TileExpertDrain<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        enabled: bool,
        sequence: &'a mut TargetSequence,
        mailbox: &'a TargetSequenceMailbox,
        tile: &'a ExpertTilePlan,
        layout: ExpertStorageLayout,
        backend: TargetExpertBackend,
        cpu_threads: usize,
        metal_source_selector: Option<&'a str>,
    ) -> Self {
        Self {
            offering: enabled,
            sequence,
            mailbox,
            tile,
            layout,
            backend,
            cpu_threads,
            metal_source_selector,
            drained: 0,
        }
    }

    const fn drained(&self) -> u64 {
        self.drained
    }
}

impl ExpertDrain for TileExpertDrain<'_> {
    fn stage(&mut self, arrived: &[(u16, &[u8])]) -> bool {
        if !self.offering || arrived.is_empty() {
            return false;
        }
        let staged = self.sequence.stage_expert_spans(
            self.mailbox,
            self.tile.first_row,
            self.tile.row_count,
            arrived,
            self.layout,
            self.backend,
            self.cpu_threads,
            self.metal_source_selector,
        );
        if staged {
            self.drained = self.drained.saturating_add(arrived.len() as u64);
        } else {
            // The provider declines for structural reasons — backend, tile
            // shape, an unqualified bridge — none of which change while this
            // tile is being read. Stop asking rather than pay for a refused
            // call per arrival.
            self.offering = false;
        }
        staged
    }
}

/// Offer tier-resident spans to the drain. They cost no I/O, so this is the
/// one batch that is always available before any read has finished.
fn stage_pinned_spans(pinned: &[(u16, Arc<PageAlignedSpan>)], drain: &mut dyn ExpertDrain) {
    if pinned.is_empty() {
        return;
    }
    let mut arrived: Vec<(u16, &[u8])> = pinned
        .iter()
        .map(|(expert, span)| (*expert, &span[..]))
        .collect();
    arrived.sort_unstable_by_key(|(expert, _)| *expert);
    drain.stage(&arrived);
}

/// Offer the settled speculative hits. Called only once every hit has been
/// waited on and validated, so a speculative failure can never reach here.
fn stage_hit_batches(hits: &[ExpertUnionReadBatch], drain: &mut dyn ExpertDrain) {
    let mut arrived: Vec<(u16, &[u8])> = Vec::with_capacity(hits.len());
    for batch in hits {
        let [expert] = batch.expert_ids() else {
            return;
        };
        arrived.push((*expert, batch.buffers().other()));
    }
    if arrived.is_empty() {
        return;
    }
    arrived.sort_unstable_by_key(|(expert, _)| *expert);
    drain.stage(&arrived);
}

/// Wait for one demand union, handing each expert to the drain as its own
/// bytes land instead of after the whole batch settles.
///
/// The batch this returns is exactly the batch the plain wait would have
/// returned; the drain only gets a look at parts of it early. When the reader
/// offers no partial view — an unsupported plan shape, or simply a batch that
/// finished before the first look — this degrades to the ordinary wait.
fn drain_expert_union(
    ticket: ExpertUnionReadTicket,
    drain: &mut dyn ExpertDrain,
) -> Result<ExpertUnionReadBatch> {
    let mut staged = 0_u64;
    let mut seen_reads = 0_u64;
    while !ticket.is_ready() {
        let arrived = ticket.arrived_experts() & !staged;
        // Hand over a wave at a time, not an expert at a time. Each offer
        // costs a provider call and a GPU submission, and experts land in
        // roughly reader-pool-sized groups anyway, so waiting for a group
        // keeps the fixed cost per staged expert small. The tail below the
        // threshold is deliberately left to the tile's own finish call: by
        // then there is no read left to overlap, and finishing it in one
        // batched dispatch beats staging it in several small ones.
        if arrived.count_ones() < EXPERT_EARLY_DRAIN_WAVE as u32 {
            // Park rather than spin: the reader workers own the device, and a
            // busy caller would only take cycles away from the reads it is
            // waiting for.
            seen_reads = ticket.wait_for_read_progress(seen_reads);
            continue;
        }
        let expert_ids = ticket.expert_ids();
        let mut batch: Vec<(u16, &[u8])> = Vec::with_capacity(expert_ids.len());
        for (slot, expert) in expert_ids.iter().enumerate() {
            if arrived & (1_u64 << slot) == 0 {
                continue;
            }
            // SAFETY: `arrived_experts` reported this slot's every read
            // complete, and the borrow ends before the ticket is waited on.
            let Some(span) = (unsafe { ticket.arrived_span(slot) }) else {
                continue;
            };
            batch.push((*expert, span));
        }
        staged |= arrived;
        if batch.is_empty() {
            continue;
        }
        if !drain.stage(&batch) {
            break;
        }
    }
    ticket.wait()
}

#[cfg_attr(not(test), allow(dead_code))]
fn read_expert_tile_with_prefetch(
    experts: &RawExpertCorpus,
    expert_reader: &Reader,
    expert_pin_tier: Option<&ExpertPinTier>,
    layer: u32,
    canonical_expert_ids: &[u16],
    prefetch: Option<ExpertPrefetchSet>,
) -> Result<ExpertTileLease> {
    read_expert_tile_draining(
        experts,
        expert_reader,
        expert_pin_tier,
        layer,
        canonical_expert_ids,
        prefetch,
        &mut NoExpertDrain,
    )
}

fn read_expert_tile_draining(
    experts: &RawExpertCorpus,
    expert_reader: &Reader,
    expert_pin_tier: Option<&ExpertPinTier>,
    layer: u32,
    canonical_expert_ids: &[u16],
    prefetch: Option<ExpertPrefetchSet>,
    offered: &mut dyn ExpertDrain,
) -> Result<ExpertTileLease> {
    let mut latched = LatchingDrain::new(offered);
    let drain: &mut dyn ExpertDrain = &mut latched;
    // Partition against the permanent tier before any submission: a resident
    // span was authenticated when it was promoted and needs no file I/O this
    // pass. `streamed` keeps canonical ascending order for the demand union.
    let mut pinned: Vec<(u16, Arc<PageAlignedSpan>)> = Vec::new();
    let streamed: Vec<u16> = match expert_pin_tier {
        Some(tier) => canonical_expert_ids
            .iter()
            .copied()
            .filter(|&expert| match tier.lookup(layer, expert) {
                Some(span) => {
                    pinned.push((expert, span));
                    false
                }
                None => true,
            })
            .collect(),
        None => canonical_expert_ids.to_vec(),
    };
    let read_streamed_demand = |lease_pinned: Vec<(u16, Arc<PageAlignedSpan>)>,
                                drain: &mut dyn ExpertDrain|
     -> Result<ExpertTileLease> {
        if lease_pinned.is_empty() {
            let ticket = experts.submit_union(expert_reader, layer, canonical_expert_ids)?;
            let batch = drain_expert_union(ticket, drain)?;
            promote_from_batch(expert_pin_tier, layer, &batch);
            return Ok(ExpertTileLease::Contiguous(batch));
        }
        let demand = if streamed.is_empty() {
            None
        } else {
            // Start the disk before anything else. Tier-resident spans need no
            // I/O at all, so their compute is the cheapest thing to put into
            // the window these reads are about to occupy — but only once the
            // reads are actually in flight.
            let ticket = experts.submit_union(expert_reader, layer, &streamed)?;
            stage_pinned_spans(&lease_pinned, drain);
            let batch = drain_expert_union(ticket, drain)?;
            promote_from_batch(expert_pin_tier, layer, &batch);
            Some(batch)
        };
        Ok(ExpertTileLease::Scattered {
            layout: experts.layout(),
            pinned: lease_pinned,
            hits: Vec::new(),
            demand,
        })
    };
    let Some(prefetch) = prefetch else {
        return read_streamed_demand(pinned, drain);
    };
    if prefetch.target_layer != layer {
        prefetch.cancel_and_drain();
        return read_streamed_demand(pinned, drain);
    }

    // A speculative ticket for a tier-resident expert is a loser: cancelling
    // it can still reclaim a read the filesystem has not started.
    let mut hit_tickets = Vec::with_capacity(prefetch.tickets.len());
    let mut loser_tickets = Vec::with_capacity(prefetch.tickets.len());
    for (expert, ticket) in prefetch.tickets {
        if streamed.binary_search(&expert).is_ok() {
            hit_tickets.push((expert, ticket));
        } else {
            loser_tickets.push((expert, ticket));
        }
    }

    // Cancel queued losers immediately, then submit known misses before
    // draining any loser already inside a syscall or waiting for speculative
    // hits. Demand and prefetch own independent bounded Reader arenas, so the
    // useful read can overlap the unavoidable tail of wrong speculation. This
    // changes scheduling only: a speculative error still forces one complete
    // checked demand retry below, and loser storage remains leased until its
    // final writer is drained.
    hit_tickets.sort_unstable_by_key(|(expert, _)| *expert);
    let mut provisional_misses = Vec::new();
    let demand_ticket = match cancel_submit_drain(
        loser_tickets,
        |(_, ticket)| ticket.cancel_unclaimed(),
        || {
            provisional_misses = streamed
                .iter()
                .copied()
                .filter(|expert| {
                    hit_tickets
                        .binary_search_by_key(expert, |(hit, _)| *hit)
                        .is_err()
                })
                .collect();
            if provisional_misses.is_empty() {
                Ok(None)
            } else {
                experts
                    .submit_union(expert_reader, layer, &provisional_misses)
                    .map(Some)
            }
        },
        |(_, ticket)| ticket.drain_cancelled(),
    ) {
        Ok(ticket) => ticket,
        Err(error) => {
            cancel_and_drain_prefetch_tickets(hit_tickets);
            return Err(error);
        }
    };

    // Tier-resident spans are already authenticated and need no read at all,
    // so their compute is the first thing worth pulling into the window the
    // demand reads are about to occupy.
    stage_pinned_spans(&pinned, drain);

    // A speculative read error is not an authoritative failure. Treat it as
    // a miss and retry the complete canonical union through the ordinary
    // checked demand path. Drain the already-submitted known-miss batch first
    // so its private arena cannot outlive this failed merge attempt.
    let mut hits = Vec::with_capacity(hit_tickets.len());
    let mut speculative_failure = false;
    for (expert, ticket) in hit_tickets {
        match ticket.wait() {
            Ok(batch)
                if batch.layer() == layer
                    && batch.expert_ids() == [expert]
                    && batch.layout() == experts.layout()
                    && batch.buffers().other().len() == experts.layout().expert_span_bytes() =>
            {
                hits.push(batch);
            }
            _ => speculative_failure = true,
        }
    }
    if speculative_failure {
        if let Some(ticket) = demand_ticket {
            let _ = ticket.wait();
        }
        // The retry reads the complete canonical union — including any
        // tier-resident experts — so this fallback stays byte-identical to
        // the established path. Anything already staged came from a
        // tier-resident span, which the retry re-reads to the same bytes, so
        // reusing or recomputing it is indistinguishable.
        let batch = experts.read_union(expert_reader, layer, canonical_expert_ids)?;
        promote_from_batch(expert_pin_tier, layer, &batch);
        return Ok(ExpertTileLease::Contiguous(batch));
    }

    hits.sort_unstable_by_key(|batch| batch.expert_ids()[0]);
    // Every speculative hit is now authenticated and settled, so these spans
    // can join the drain before the demand reads finish.
    stage_hit_batches(&hits, drain);
    let demand = match demand_ticket {
        Some(ticket) => {
            let batch = drain_expert_union(ticket, drain)?;
            if batch.layer() != layer
                || batch.expert_ids() != provisional_misses
                || batch.layout() != experts.layout()
            {
                return Err(DeltafinError::new(
                    "expert-prefetch demand read returned a stale layer, set, or layout",
                ));
            }
            Some(batch)
        }
        None => None,
    };
    for batch in &hits {
        promote_from_batch(expert_pin_tier, layer, batch);
    }
    if let Some(batch) = &demand {
        promote_from_batch(expert_pin_tier, layer, batch);
    }
    if hits.is_empty() && pinned.is_empty() {
        return demand.map(ExpertTileLease::Contiguous).ok_or_else(|| {
            DeltafinError::new("expert tile merge produced neither resident nor demand bytes")
        });
    }
    Ok(ExpertTileLease::Scattered {
        layout: experts.layout(),
        pinned,
        hits,
        demand,
    })
}

#[allow(clippy::too_many_arguments)]
fn finish_expert_mailbox(
    sequence: &mut TargetSequence,
    mailbox: &TargetSequenceMailbox,
    experts: &RawExpertCorpus,
    expert_reader: &Reader,
    expert_prefetch_reader: Option<&Reader>,
    pending_expert_prefetch: &mut Option<ExpertPrefetchSet>,
    mut pilot_gate: Option<&mut PilotGate>,
    expert_heat: &ExpertHeat,
    expert_pin_tier: Option<&ExpertPinTier>,
    backend: TargetExpertBackend,
    cpu_threads: usize,
    metal_source_selector: Option<&str>,
    metal_expert_wrapper_retention: bool,
    early_drain_enabled: bool,
    complete_expert_union: CompleteExpertUnion,
    mut profile: Option<&mut TargetLayerPhaseProfile>,
) -> Result<()> {
    // The authoritative routes are the free scoring signal: they settle any
    // outstanding lookahead prediction for this layer and reseed the
    // previous-token predictor, before a single expert byte is read.
    if let Some(gate) = pilot_gate.as_deref_mut() {
        gate.observe_routes(
            mailbox.layer_index(),
            (0..mailbox.position_count())
                .filter_map(|row| mailbox.route(row))
                .map(|route| route.ordered_experts()),
        );
    }
    // The same free signal feeds the cross-run heat histogram: presence per
    // mailbox, all backends, infallible, and strictly after the authoritative
    // routes were published.
    expert_heat.observe_layer_routes(
        mailbox.layer_index(),
        (0..mailbox.position_count())
            .filter_map(|row| mailbox.route(row))
            .map(|route| route.ordered_experts()),
    );
    let mut first_row = 0_usize;
    // The established Python/Metal verifier fetches one sorted unique expert
    // union for all candidate rows. Reproduce that topology only for exact
    // CPU/Metal verification and only after a fresh host-memory admission.
    // Prefill remains at the proven 64-expert bound, and CUDA retains its
    // provider-owned cache/miss planner.
    let mut admitted_complete_verifier_tile = if may_stage_complete_verifier_union(
        complete_expert_union.enabled(),
        sequence.mode(),
        backend,
        mailbox.position_count(),
    ) {
        let candidate =
            next_expert_tile(mailbox.position_count(), 0, K3_EXPERT_UNION_MAX, |row| {
                mailbox
                    .route(row)
                    .map(|route| *route.ordered_experts())
                    .ok_or_else(|| DeltafinError::new("target route mailbox lost a requested row"))
            })?;
        (candidate.row_count == mailbox.position_count()
            && admit_complete_verifier_union(
                complete_expert_union,
                expert_reader,
                experts.layout(),
                candidate.expert_count(),
            ))
        .then_some(candidate)
    } else {
        None
    };
    while first_row < mailbox.position_count() {
        let tile = match admitted_complete_verifier_tile.take() {
            Some(tile) if first_row == 0 => tile,
            _ => next_expert_tile(
                mailbox.position_count(),
                first_row,
                K3_EXPERT_BASE_UNION_MAX,
                |row| {
                    mailbox
                        .route(row)
                        .map(|route| *route.ordered_experts())
                        .ok_or_else(|| {
                            DeltafinError::new("target route mailbox lost a requested row")
                        })
                },
            )?,
        };
        if matches!(
            backend,
            TargetExpertBackend::Auto | TargetExpertBackend::Cuda
        ) {
            if expert_prefetch_reader.is_some() || pending_expert_prefetch.is_some() {
                return Err(DeltafinError::new(
                    "CUDA expert planning cannot share the scattered host-prefetch path",
                ));
            }
            if expert_pin_tier.is_some() {
                return Err(DeltafinError::new(
                    "CUDA expert planning cannot share the pinned host-span tier",
                ));
            }
            let planning_started = profile.as_ref().map(|_| Instant::now());
            let plan = sequence.plan_expert_tile(
                mailbox,
                tile.first_row,
                tile.row_count,
                tile.expert_ids(),
                backend,
                cpu_threads,
                metal_source_selector,
            )?;
            add_profile_elapsed(profile.as_deref_mut(), planning_started, |profile| {
                &mut profile.authoritative_expert_read_prefetch_ns
            });
            if let Some(profile) = profile.as_deref_mut() {
                let misses = plan.missing_experts().len() as u64;
                profile.expert_plan_misses = profile.expert_plan_misses.saturating_add(misses);
                profile.expert_plan_hits = profile
                    .expert_plan_hits
                    .saturating_add((tile.expert_ids().len() as u64).saturating_sub(misses));
            }
            let read_started = profile.as_ref().map(|_| Instant::now());
            let misses = if plan.missing_experts().is_empty() {
                None
            } else {
                let batch = experts.read_union(
                    expert_reader,
                    mailbox.layer_index(),
                    plan.missing_experts(),
                )?;
                if batch.layer() != mailbox.layer_index()
                    || batch.expert_ids() != plan.missing_experts()
                    || batch.layout() != ExpertStorageLayout::RawV1
                {
                    return Err(DeltafinError::new(
                        "CUDA expert-plan reader returned a stale layer, miss set, or layout",
                    ));
                }
                Some(batch)
            };
            add_profile_elapsed(profile.as_deref_mut(), read_started, |profile| {
                &mut profile.authoritative_expert_read_prefetch_ns
            });
            let kernel_started = profile.as_ref().map(|_| Instant::now());
            sequence.finish_planned_expert_tile(
                mailbox,
                plan,
                misses
                    .as_ref()
                    .map_or(&[][..], |batch| batch.buffers().other()),
            )?;
            add_profile_elapsed(profile.as_deref_mut(), kernel_started, |profile| {
                &mut profile.expert_kernel_ns
            });
            first_row += tile.row_count;
            continue;
        }
        let read_started = profile.as_ref().map(|_| Instant::now());
        // A prediction for L+1 must survive every remaining tile of L.  The
        // old unconditional `take()` handed that future generation to the
        // next same-layer tile, which correctly rejected and canceled it as a
        // layer mismatch.  Consume only a generation that is due now (or an
        // impossible stale generation, so the existing fail-safe drains it).
        let due_prefetch = take_due_expert_prefetch(pending_expert_prefetch, mailbox.layer_index());
        // Early drain: each expert's own matmul starts as its bytes land
        // rather than after the layer's whole miss set has been read. Purely a
        // head start — `lease.finish` below still runs the authoritative tile,
        // including the one reduction over every expert in route order.
        let mut early_drain = TileExpertDrain::new(
            early_drain_enabled,
            sequence,
            mailbox,
            &tile,
            experts.layout(),
            backend,
            cpu_threads,
            metal_source_selector,
        );
        let lease = read_expert_tile_draining(
            experts,
            expert_reader,
            expert_pin_tier,
            mailbox.layer_index(),
            tile.expert_ids(),
            due_prefetch,
            &mut early_drain,
        )?;
        let drained_experts = early_drain.drained();
        add_profile_elapsed(profile.as_deref_mut(), read_started, |profile| {
            &mut profile.authoritative_expert_read_prefetch_ns
        });
        if let Some(profile) = profile.as_deref_mut() {
            profile.expert_early_drained =
                profile.expert_early_drained.saturating_add(drained_experts);
        }
        // The provider materializes the optional scheduling hint only after
        // Rust has the real current-layer route and its authoritative bytes.
        // An ABI error is terminal because the sequence cancels itself; a
        // prediction miss is represented as a successful empty hint.
        let final_tile = first_row
            .checked_add(tile.row_count)
            .is_some_and(|end| end == mailbox.position_count());
        let hint_started = profile.as_ref().map(|_| Instant::now());
        let next_hint = if expert_prefetch_reader.is_some() && final_tile {
            sequence.take_prefetch_hint()?
        } else {
            None
        };
        add_profile_elapsed(profile.as_deref_mut(), hint_started, |profile| {
            &mut profile.authoritative_expert_read_prefetch_ns
        });
        // Match the proven scheduler only at L's final tile: once every
        // authoritative demand tile and the provider-owned hint have landed,
        // submit L+1 before running the last synchronous Metal kernel.  Doing
        // this on an earlier tile lets speculative disk reads contend with
        // still-unfinished authoritative tiles from L.  The second arena
        // generation guarantees that up to 32 current hit leases and 32 next
        // tickets cannot deadlock. Keep the new generation local until current
        // publication succeeds; on failure drain all speculative work before
        // returning.
        //
        // The gate sits strictly between the hint and the read submission: it
        // records every taken hint for scoring, then admits, redirects to the
        // better-scoring predictor, or suppresses this layer's speculative
        // reads. Without a governor the hint passes through unchanged.
        let submit_started = profile.as_ref().map(|_| Instant::now());
        let next_plan = if expert_prefetch_reader.is_some() && final_tile {
            let pilot_plan = next_hint
                .as_ref()
                .and_then(|hint| ExpertPrefetchPlan::new(hint.target_layer(), hint.expert_ids()));
            match pilot_gate.as_deref_mut() {
                Some(gate) => gate.admit(mailbox.layer_index(), pilot_plan),
                None => pilot_plan,
            }
        } else {
            None
        };
        let next_prefetch = match (
            pending_expert_prefetch.is_none(),
            expert_prefetch_reader,
            next_plan,
        ) {
            (true, Some(reader), Some(plan)) => try_schedule_expert_prefetch(experts, reader, plan),
            _ => None,
        };
        add_profile_elapsed(profile.as_deref_mut(), submit_started, |profile| {
            &mut profile.authoritative_expert_read_prefetch_ns
        });

        let kernel_started = profile.as_ref().map(|_| Instant::now());
        let finish_result = lease.finish(
            sequence,
            mailbox,
            &tile,
            backend,
            cpu_threads,
            metal_source_selector,
            metal_expert_wrapper_retention,
        );
        add_profile_elapsed(profile.as_deref_mut(), kernel_started, |profile| {
            &mut profile.expert_kernel_ns
        });
        if let Err(error) = finish_result {
            if let Some(prefetch) = next_prefetch {
                prefetch.cancel_and_drain();
            }
            if let Some(prefetch) = pending_expert_prefetch.take() {
                prefetch.cancel_and_drain();
            }
            return Err(error);
        }
        if let Some(prefetch) = next_prefetch {
            debug_assert!(pending_expert_prefetch.is_none());
            *pending_expert_prefetch = Some(prefetch);
        }
        first_row += tile.row_count;
    }
    Ok(())
}

fn add_profile_elapsed<F>(
    profile: Option<&mut TargetLayerPhaseProfile>,
    started: Option<Instant>,
    field: F,
) where
    F: FnOnce(&mut TargetLayerPhaseProfile) -> &mut u64,
{
    if let (Some(profile), Some(started)) = (profile, started) {
        let target = field(profile);
        *target = target.saturating_add(elapsed_ns(started));
    }
}

fn next_expert_tile<F>(
    position_count: usize,
    first_row: usize,
    expert_limit: usize,
    mut route_at: F,
) -> Result<ExpertTilePlan>
where
    F: FnMut(usize) -> Result<[u16; K3_EXPERT_TOP_K]>,
{
    if position_count == 0
        || position_count > TARGET_SEQUENCE_MAX_POSITIONS as usize
        || first_row >= position_count
        || !(K3_EXPERT_TOP_K..=K3_EXPERT_UNION_MAX).contains(&expert_limit)
    {
        return Err(DeltafinError::new(
            "expert tile cursor is outside the bounded target mailbox",
        ));
    }
    let first_route = canonical_route(route_at(first_row)?)?;
    let mut selected = [false; K3_EXPERT_COUNT as usize];
    for expert in first_route {
        selected[expert as usize] = true;
    }
    let mut expert_count = K3_EXPERT_TOP_K;
    let mut row_count = 1_usize;
    while row_count < TARGET_EXPERT_TILE_MAX_ROWS && first_row + row_count < position_count {
        let next = canonical_route(route_at(first_row + row_count)?)?;
        let additions = next
            .iter()
            .filter(|&&expert| !selected[expert as usize])
            .count();
        if expert_count + additions > expert_limit {
            break;
        }
        for expert in next {
            selected[expert as usize] = true;
        }
        expert_count += additions;
        row_count += 1;
    }
    let mut canonical_experts = Vec::with_capacity(expert_count);
    for (expert, &included) in selected.iter().enumerate() {
        if included {
            canonical_experts.push(expert as u16);
        }
    }
    if canonical_experts.len() != expert_count {
        return Err(DeltafinError::new(
            "expert tile union count disagrees with its canonical storage set",
        ));
    }
    Ok(ExpertTilePlan {
        first_row,
        row_count,
        canonical_experts: canonical_experts.into_boxed_slice(),
    })
}

/// Bound a complete verifier-row expert slab to the maximum exact draft width.
fn full_commit_union_upper_bound(positions: usize) -> Option<usize> {
    let experts = positions.checked_mul(K3_EXPERT_TOP_K)?;
    (experts > K3_EXPERT_BASE_UNION_MAX && experts <= FULL_COMMIT_EXPERT_UNION_MAX)
        .then_some(experts)
}

fn startup_complete_expert_union_capacity(
    enabled: bool,
    backend: TargetExpertBackend,
    layout: ExpertStorageLayout,
) -> Option<usize> {
    (enabled
        && matches!(
            backend,
            TargetExpertBackend::Cpu | TargetExpertBackend::Metal
        )
        && layout == ExpertStorageLayout::Scale4V2)
        .then_some(FULL_COMMIT_EXPERT_UNION_MAX)
}

/// Reserve the compact nine-row verifier slab before target execution begins.
///
/// Admission is fail-soft: unknown/pressured memory or allocation failure
/// leaves the ordinary exact <=64-expert tiling available. Because this runs
/// immediately after construction of an empty Reader, successful reservation
/// cannot retire an older slab or invalidate Metal no-copy wrappers.
fn reserve_complete_expert_union_at_startup(
    enabled: bool,
    backend: TargetExpertBackend,
    expert_reader: &Reader,
    layout: ExpertStorageLayout,
    host: HostMemory,
) -> Option<usize> {
    let expert_capacity = startup_complete_expert_union_capacity(enabled, backend, layout)?;
    if !admit_dynamic_complete_verifier_union(host, expert_reader, layout, expert_capacity) {
        return None;
    }
    let logical_bytes = layout.expert_span_bytes().checked_mul(expert_capacity)?;
    expert_reader
        .reserve_capacity(BufferLengths::new(0, 0, logical_bytes))
        .ok()?;
    Some(expert_capacity)
}

fn prepare_complete_expert_union(
    enabled: bool,
    startup_reserved_capacity: Option<usize>,
    mode: TargetSequenceMode,
    full_commit_only: bool,
    positions: usize,
    backend: TargetExpertBackend,
    expert_reader: &Reader,
    layout: ExpertStorageLayout,
) -> CompleteExpertUnion {
    if !enabled {
        return CompleteExpertUnion::Disabled;
    }
    if mode != TargetSequenceMode::Verify
        || !full_commit_only
        || !matches!(
            backend,
            TargetExpertBackend::Cpu | TargetExpertBackend::Metal
        )
    {
        return CompleteExpertUnion::Dynamic;
    }
    let Some(expert_capacity) = full_commit_union_upper_bound(positions) else {
        return if positions.saturating_mul(K3_EXPERT_TOP_K) <= K3_EXPERT_BASE_UNION_MAX {
            CompleteExpertUnion::Dynamic
        } else {
            CompleteExpertUnion::Disabled
        };
    };
    // Compact verifier slabs are admitted and allocated exactly once during
    // runtime construction. Never grow the expert Reader, retire a slab, or
    // flush Metal wrapper caches from this live target hot path.
    if layout == ExpertStorageLayout::Scale4V2 {
        return startup_reserved_capacity
            .filter(|&capacity| capacity >= expert_capacity)
            .map_or(CompleteExpertUnion::Disabled, CompleteExpertUnion::Reserved);
    }
    let Some(logical_bytes) = layout.expert_span_bytes().checked_mul(expert_capacity) else {
        return CompleteExpertUnion::Disabled;
    };
    let lengths = BufferLengths::new(0, 0, logical_bytes);
    let Ok(host_bytes) = expert_reader.replacement_admission_bytes(lengths) else {
        return CompleteExpertUnion::Disabled;
    };
    if host_bytes != 0 {
        let selection = select_resident_prefix(
            probe_host_memory(),
            ProviderMemory::Host,
            &[],
            FixedCosts {
                host_bytes,
                provider_bytes: 0,
            },
            ResidencyOverride::default(),
            ResidencyPolicy::default(),
        );
        if selection.stop != ResidencyStop::AllLayersFit {
            return CompleteExpertUnion::Disabled;
        }
    }
    if expert_reader.reserve_capacity(lengths).is_err() {
        return CompleteExpertUnion::Disabled;
    }
    CompleteExpertUnion::Reserved(expert_capacity)
}

fn admit_complete_verifier_union(
    admission: CompleteExpertUnion,
    expert_reader: &Reader,
    layout: ExpertStorageLayout,
    expert_count: usize,
) -> bool {
    if expert_count <= K3_EXPERT_BASE_UNION_MAX {
        return true;
    }
    if expert_count > K3_EXPERT_UNION_MAX {
        return false;
    }
    if let CompleteExpertUnion::Reserved(capacity) = admission {
        return expert_count <= capacity;
    }
    if admission == CompleteExpertUnion::Disabled {
        return false;
    }
    admit_dynamic_complete_verifier_union(probe_host_memory(), expert_reader, layout, expert_count)
}

fn admit_dynamic_complete_verifier_union(
    host: HostMemory,
    expert_reader: &Reader,
    layout: ExpertStorageLayout,
    expert_count: usize,
) -> bool {
    if expert_count <= K3_EXPERT_BASE_UNION_MAX {
        return true;
    }
    if expert_count > K3_EXPERT_UNION_MAX {
        return false;
    }
    let Some(logical_bytes) = layout.expert_span_bytes().checked_mul(expert_count) else {
        return false;
    };
    let Ok(host_bytes) =
        expert_reader.replacement_admission_bytes(BufferLengths::new(0, 0, logical_bytes))
    else {
        return false;
    };
    if host_bytes == 0 {
        return true;
    }
    let selection = select_resident_prefix(
        host,
        ProviderMemory::Host,
        &[],
        FixedCosts {
            host_bytes,
            provider_bytes: 0,
        },
        ResidencyOverride::default(),
        ResidencyPolicy::default(),
    );
    selection.stop == ResidencyStop::AllLayersFit
}

fn may_stage_complete_verifier_union(
    enabled: bool,
    mode: TargetSequenceMode,
    backend: TargetExpertBackend,
    positions: usize,
) -> bool {
    enabled
        && mode == TargetSequenceMode::Verify
        && matches!(
            backend,
            TargetExpertBackend::Cpu | TargetExpertBackend::Metal
        )
        && (1..=TARGET_EXPERT_TILE_MAX_ROWS).contains(&positions)
}

fn canonical_route(mut ordered_experts: [u16; K3_EXPERT_TOP_K]) -> Result<[u16; K3_EXPERT_TOP_K]> {
    ordered_experts.sort_unstable();
    for (index, &expert) in ordered_experts.iter().enumerate() {
        if expert >= K3_EXPERT_COUNT || (index != 0 && ordered_experts[index - 1] == expert) {
            return Err(DeltafinError::new(
                "target route must contain all 16 unique experts in 0..896",
            ));
        }
    }
    Ok(ordered_experts)
}

fn next_prompt_chunk(total: usize, first: usize) -> Result<Option<Range<usize>>> {
    if first > total {
        return Err(DeltafinError::new(
            "prompt chunk cursor advanced beyond the tokenized prompt",
        ));
    }
    if first == total {
        return Ok(None);
    }
    let end = first
        .saturating_add(TARGET_SEQUENCE_MAX_POSITIONS as usize)
        .min(total);
    Ok(Some(first..end))
}

/// The whole-request context ceiling worth proving once at request start, or
/// `None` when the request is unbounded or too large to reserve speculatively.
///
/// A bounded request's final context need is known before its first token:
/// prompt length plus `--max-new`, plus the widest speculative verify chunk
/// that can be staged beyond the committed prefix. Proving that ceiling while
/// the request is seconds old moves the memory-admission gate from "35
/// minutes in, at whatever the host looks like then" to "now, when the
/// request was admissible enough to start" — every later growth boundary
/// then takes the committed-capacity fast path with no live re-proof, so
/// pressure arriving mid-run (another application opening) can no longer
/// fail a generation that was admissible when it began.
///
/// The token bound keeps this a reservation of intent, not a land grab: a
/// server chat request arrives with the server-wide output ceiling (defaults
/// to a million tokens) rather than an expected length, and pre-admitting
/// gigabytes against a ceiling nobody will reach would displace resident
/// layers for nothing. Requests above the bound simply keep incremental
/// growth, exactly as before.
fn context_pre_admission_envelope(
    prompt_tokens: usize,
    maximum_new: Option<usize>,
    speculative_max_drafts: usize,
    admitted_expanded_context_tokens: u64,
) -> Option<u64> {
    const MAX_PRE_ADMISSION_TOKENS: u64 = 512;
    let maximum_new = u64::try_from(maximum_new?).ok()?;
    let envelope = u64::try_from(prompt_tokens)
        .ok()?
        .checked_add(maximum_new)?
        .checked_add(u64::try_from(speculative_max_drafts).ok()?)?
        .checked_add(1)?;
    (envelope <= MAX_PRE_ADMISSION_TOKENS.min(admitted_expanded_context_tokens))
        .then_some(envelope)
}

fn bounded_draft_budget(
    configured: usize,
    remaining_context: usize,
    remaining_output: usize,
) -> usize {
    configured
        .min(remaining_context.saturating_sub(1))
        .min(remaining_output.saturating_sub(1))
}

const fn stop_after_transaction(
    natural_stop: Option<StopReason>,
    interrupt_requested: bool,
) -> Option<StopReason> {
    if interrupt_requested {
        // SIGINT owns the boundary once observed, even when the same completed
        // transaction reached a natural limit. This keeps optional target and
        // draft publication fail-closed under every cancellation race.
        Some(StopReason::Interrupted)
    } else {
        natural_stop
    }
}

fn truncate_after_first(tokens: &mut Vec<u32>, terminal: u32) {
    if let Some(index) = tokens.iter().position(|&token| token == terminal) {
        tokens.truncate(index.saturating_add(1));
    }
}

fn cancel_after_error(sequence: TargetSequence, error: DeltafinError) -> DeltafinError {
    match sequence.cancel() {
        Ok(()) => error,
        Err(cancel_error) => DeltafinError::new(format!(
            "{error}; additionally failed to cancel the unpublished native target sequence: {cancel_error}"
        )),
    }
}

fn print_run_stats(
    counters: &NativeRunCounters,
    started: Instant,
    homes: Option<&StorageHomes>,
    storage: StorageBytes,
) {
    let elapsed = started.elapsed().as_secs_f64();
    let tokens_per_second = if elapsed > 0.0 {
        counters.generated_tokens as f64 / elapsed
    } else {
        0.0
    };
    let seconds_per_token = if counters.generated_tokens == 0 {
        0.0
    } else {
        elapsed / counters.generated_tokens as f64
    };
    eprintln!(
        "\n[stats] generated={} elapsed={elapsed:.3}s speed={tokens_per_second:.4} token/s ({seconds_per_token:.3} s/token) chunks={} committed_rows={} verify_tx={} (full-commit slot {}) drafts={}/{} layer_passes={} expert_rows={} expert_tiles={}",
        counters.generated_tokens,
        counters.target_chunks,
        counters.committed_positions,
        counters.verify_transactions,
        counters.full_commit_slot_verifies,
        counters.accepted_draft_tokens,
        counters.verified_draft_tokens,
        counters.streamed_layer_passes,
        counters.expert_rows,
        counters.expert_tiles,
    );
    let read = storage.since(counters.storage_at_start);
    let total = read.spine + read.expert_demand + read.expert_prefetch;
    eprintln!(
        "[stats] disk: spine {:.2} GB + experts {:.2} GB demand + {:.2} GB prefetch = {:.2} GB in {elapsed:.1}s ({:.2} GB/s, {:.2} GB/token)",
        read.spine as f64 / 1e9,
        read.expert_demand as f64 / 1e9,
        read.expert_prefetch as f64 / 1e9,
        total as f64 / 1e9,
        if elapsed > 0.0 { total as f64 / 1e9 / elapsed } else { 0.0 },
        if counters.generated_tokens == 0 {
            0.0
        } else {
            total as f64 / 1e9 / counters.generated_tokens as f64
        },
    );
    if let Some(homes) = homes {
        let drives = homes.stats();
        let total: u64 = drives.iter().map(|drive| drive.served_bytes).sum();
        for (index, drive) in drives.iter().enumerate() {
            eprintln!(
                "[stats] drive {index}: {:.2} GB served ({:.1}%) in {} reads, {} {:.2} GB/s, {} failed reads{}, {}",
                drive.served_bytes as f64 / 1e9,
                if total == 0 { 0.0 } else { 100.0 * drive.served_bytes as f64 / total as f64 },
                drive.reads,
                if drive.measured { "measured" } else { "assumed" },
                drive.rate_gbps,
                drive.failures,
                if drive.quarantined { " (sitting out)" } else { "" },
                drive.root.display(),
            );
        }
    }
    if let Some(emulator) = crate::storage_emulation::DeviceEmulator::global() {
        emulator.print_report();
    }
    if counters.target_profile.chunks != 0 {
        let totals = counters.target_profile.layer_totals();
        eprintln!(
            "[phases] chunks={} sequence={:.3}s read_wait={:.3}s bind_upload={:.3}s attention_resident={:.3}s expert_read_prefetch={:.3}s expert_kernel={:.3}s source_fence={:.3}s tail_head_sync={:.3}s layer_other={:.3}s",
            counters.target_profile.chunks,
            ns_seconds(counters.target_profile.sequence_total_ns),
            ns_seconds(totals.spine_read_wait_ns),
            ns_seconds(totals.spine_bind_upload_ns),
            ns_seconds(totals.attention_resident_compute_ns),
            ns_seconds(totals.authoritative_expert_read_prefetch_ns),
            ns_seconds(totals.expert_kernel_ns),
            ns_seconds(totals.source_fence_ns),
            ns_seconds(counters.target_profile.tail_head_sync_ns),
            ns_seconds(totals.other_control_ns()),
        );
        if totals.expert_plan_hits != 0 || totals.expert_plan_misses != 0 {
            eprintln!(
                "[phases] expert plans: {} cache hits, {} misses read+uploaded",
                totals.expert_plan_hits, totals.expert_plan_misses,
            );
        }
        if totals.expert_early_drained != 0 {
            eprintln!(
                "[phases] expert early drain: {} experts started before their layer's reads finished",
                totals.expert_early_drained,
            );
        }
    }
}

fn print_target_layer_profile(profile: &TargetLayerPhaseProfile, layer: usize, layer_count: usize) {
    eprintln!(
        "[phase layer {layer:>2}/{last}] total={:.3}s read_wait={:.3}s bind_upload={:.3}s attention_resident={:.3}s expert_read_prefetch={:.3}s expert_kernel={:.3}s fence={:.3}s other={:.3}s",
        ns_seconds(profile.layer_total_ns),
        ns_seconds(profile.spine_read_wait_ns),
        ns_seconds(profile.spine_bind_upload_ns),
        ns_seconds(profile.attention_resident_compute_ns),
        ns_seconds(profile.authoritative_expert_read_prefetch_ns),
        ns_seconds(profile.expert_kernel_ns),
        ns_seconds(profile.source_fence_ns),
        ns_seconds(profile.other_control_ns()),
        last = layer_count.saturating_sub(1),
    );
}

fn ns_seconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000_000_000.0
}

fn output_error(operation: &str, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation}: {error}"))
}

fn compile_spine(
    program: &TargetProgram,
    model_root: &Path,
    layout: &SourceLayout,
    cache_policies: &[CachePolicy],
    persistent_loose_descriptors: bool,
) -> Result<CompiledSpine> {
    let pack_directory = model_root.join(program.representation().pack_directory_name());
    if spine_source_intent(&pack_directory)? == SpinePlanSource::AuthenticatedPacks {
        // Presence is a commitment. A malformed or partial pack set is not a
        // reason to silently switch representations after the user prepared it.
        return program
            .open_packed_spine_with_cache_policies(
                model_root,
                Some(&pack_directory),
                cache_policies,
            )
            .map(CompiledSpine::Packed);
    }
    if persistent_loose_descriptors {
        program
            .loose_spine_read_plans_with_cache_policies_and_descriptors(
                layout,
                crate::program::DEFAULT_SPINE_CHUNK_BYTES,
                cache_policies,
                true,
            )
            .map(CompiledSpine::Loose)
    } else {
        program
            .loose_spine_read_plans_with_cache_policies(
                layout,
                crate::program::DEFAULT_SPINE_CHUNK_BYTES,
                cache_policies,
            )
            .map(CompiledSpine::Loose)
    }
}

fn spine_source_intent(pack_directory: &Path) -> Result<SpinePlanSource> {
    match std::fs::symlink_metadata(pack_directory) {
        Ok(_) => Ok(SpinePlanSource::AuthenticatedPacks),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(SpinePlanSource::LooseDeferredFiles)
        }
        Err(error) => Err(DeltafinError::new(format!(
            "inspect resident-spine pack directory {}: {error}",
            pack_directory.display()
        ))),
    }
}

fn validate_embedding_contract(
    model_root: &Path,
    model: &ModelSpec,
    program: &TargetProgram,
    layout: &SourceLayout,
) -> Result<()> {
    let weight = &program.embedding_weight;
    let source = weight.source_paths(layout);
    let expected_path = k3_embedding_path(model_root);
    let expected_shape = [model.vocab_size as u64, model.hidden_size as u64];
    if source.data != expected_path
        || source.scales.is_some()
        || weight.storage != WeightStorage::Raw(WeightDType::Bf16)
        || weight.shape.as_ref() != expected_shape
        || weight.expected_data_bytes()? != EmbeddingSpec::K3.table_bytes()
        || model.vocab_size != EmbeddingSpec::K3.rows() as usize
        || model.hidden_size != EmbeddingSpec::K3.columns() as usize
    {
        return Err(DeltafinError::new(
            "compiled embedding roster differs from the exact BF16 K3 row-table contract",
        ));
    }
    Ok(())
}

fn fixed_costs(
    program: &TargetProgram,
    global_plans: &[GlobalSpinePlan],
    spine: &CompiledSpine,
    embedding_rows: usize,
    expert_prefetch_bytes: u64,
    context_growth: ContextGrowthBudget,
    verify_snapshots: VerifySnapshotBudget,
) -> Result<FixedCosts> {
    let layer_slot = component_high_water(spine.layers())?;
    let spine_arenas = layer_slot
        .checked_mul(SPINE_ARENA_SLOTS as u64)
        .ok_or_else(|| DeltafinError::new("spine arena budget overflows u64"))?;
    let expert_slot = (K3_EXPERT_SOURCE_BYTES as u64)
        .checked_mul(K3_EXPERT_BASE_UNION_MAX as u64)
        .ok_or_else(|| DeltafinError::new("expert arena slot budget overflows u64"))?;
    let expert_arenas = expert_slot
        .checked_mul(EXPERT_ARENA_SLOTS as u64)
        .ok_or_else(|| DeltafinError::new("expert arena budget overflows u64"))?;
    let persistent_reader_arenas = spine_arenas
        .checked_add(expert_arenas)
        .and_then(|bytes| bytes.checked_add(expert_prefetch_bytes))
        .ok_or_else(|| DeltafinError::new("persistent reader arena budget overflows u64"))?;
    // Global startup binds use one temporary slab and release it before layer
    // and expert arenas reach their high-water marks. Budget the larger phase,
    // rather than falsely summing mutually exclusive peaks or ignoring the
    // 1.17 GiB vocabulary-head transfer.
    let global_transient =
        buffer_high_water(global_plans.iter().map(GlobalSpinePlan::buffer_lengths))?;
    // Multi-row reads retain both the sorted-unique storage and caller-order
    // output. Include conservative index scratch without relying on a private
    // struct layout from the embedding module.
    let embedding_row_bytes = (EmbeddingSpec::K3.columns() as u64)
        .checked_mul(BF16_BYTES as u64)
        .ok_or_else(|| DeltafinError::new("embedding row budget overflows u64"))?;
    let embedding_bytes = embedding_row_bytes
        .checked_mul(embedding_rows as u64)
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| bytes.checked_add((embedding_rows as u64).saturating_mul(32)))
        .ok_or_else(|| DeltafinError::new("embedding arena budget overflows u64"))?;
    let host_bytes = persistent_reader_arenas
        .max(global_transient)
        .checked_add(embedding_bytes)
        .ok_or_else(|| DeltafinError::new("native fixed host budget overflows u64"))?;
    // A complete target transaction retains the committed KDA generation and
    // stages the next generation for all 69 layers until global commit.
    let kda_committed_and_staged = verify_snapshots
        .admission(1)?
        .transaction_peak_provider_bytes;
    let (_, startup_growth_reserve_bytes) = context_growth.startup_growth_reserve()?;
    let provider_bytes = program
        .provider_global_bytes()?
        .checked_add(K3_TAIL_DERIVED_RESIDUAL_BYTES)
        .and_then(|bytes| bytes.checked_add(kda_committed_and_staged))
        .and_then(|bytes| bytes.checked_add(context_growth.initial_provider_bytes))
        .and_then(|bytes| bytes.checked_add(startup_growth_reserve_bytes))
        .ok_or_else(|| DeltafinError::new("native fixed provider budget overflows u64"))?;
    Ok(FixedCosts {
        host_bytes,
        provider_bytes,
    })
}

fn provider_layer_costs(program: &TargetProgram) -> Result<Box<[u64]>> {
    let mut costs = program.provider_layer_bytes()?;
    for payload_bytes in &mut costs {
        // The provider derives two exact fp32 residual-score rows for every
        // complete K3 layer binding. Keep this addition beside the provider-
        // facing bootstrap until TargetProgram itself reports derived
        // allocations; moving it there must remove this term.
        *payload_bytes = (*payload_bytes)
            .checked_add(K3_LAYER_DERIVED_RESIDUAL_BYTES)
            .ok_or_else(|| DeltafinError::new("derived layer residency overflows u64"))?;
    }
    Ok(costs)
}

fn fp32_spine_execution_arena_reserve(program: &TargetProgram, device: Device) -> Result<u64> {
    if device == Device::Mps {
        return program.fp32_spine_execution_arena_bytes();
    }
    Ok(0)
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct SpineCachePlan {
    stream_nocache: bool,
    policies: Box<[CachePolicy]>,
}

#[allow(clippy::too_many_arguments)]
fn configured_spine_cache_plan(
    explicit_stream_nocache: Option<bool>,
    explicit_resident_bytes: Option<u64>,
    is_macos: bool,
    device: Device,
    representation: SpineRepresentation,
    physical_bytes: Option<u64>,
    recommended_working_set_bytes: Option<u64>,
    layer_source_bytes: &[u64],
) -> SpineCachePlan {
    let all_resident = || SpineCachePlan {
        stream_nocache: false,
        policies: vec![CachePolicy::Resident; layer_source_bytes.len()].into_boxed_slice(),
    };
    let all_streaming = || SpineCachePlan {
        stream_nocache: true,
        policies: vec![CachePolicy::Streaming; layer_source_bytes.len()].into_boxed_slice(),
    };

    // The public two-tier policy applies only to the optional int8 spine. Its
    // resident tier is clean file cache, not provider-owned tensor residency;
    // the latter remains under the independent live safety proof below. Other
    // representations retain the native runtime's established stream policy.
    if representation != SpineRepresentation::QuantizedInt8 {
        return all_streaming();
    }

    let source_total = layer_source_bytes
        .iter()
        .try_fold(0_u64, |total, bytes| total.checked_add(*bytes));
    let automatic_budget = || {
        let (Some(physical), Some(spine)) = (physical_bytes, source_total) else {
            return None;
        };
        if physical == 0 || spine == 0 {
            return None;
        }
        let host_reserve = AUTO_SPINE_HOST_RESERVE_MIN_BYTES.max(physical / 4);
        let mut safe_cache_envelope = physical.saturating_sub(host_reserve);
        if let Some(recommended) = recommended_working_set_bytes.filter(|bytes| *bytes > 0) {
            safe_cache_envelope = safe_cache_envelope
                .min(recommended.saturating_sub(AUTO_SPINE_DEVICE_RESERVE_BYTES));
        }
        (spine > safe_cache_envelope).then(|| {
            physical
                .saturating_mul(AUTO_SPINE_RESIDENT_PERCENT)
                .saturating_div(100)
                .min(AUTO_SPINE_RESIDENT_MAX_BYTES)
        })
    };

    let automatic_resident_bytes = match explicit_stream_nocache {
        Some(false) => return all_resident(),
        Some(true) => None,
        None if !is_macos || device != Device::Mps => return all_streaming(),
        None => match automatic_budget() {
            Some(bytes) => Some(bytes),
            None => return all_resident(),
        },
    };

    let resident_budget = explicit_resident_bytes
        .or(automatic_resident_bytes)
        .unwrap_or(0);
    let mut resident_bytes = 0_u64;
    let policies = layer_source_bytes
        .iter()
        .map(|&bytes| {
            let admitted = bytes > 0
                && resident_bytes
                    .checked_add(bytes)
                    .is_some_and(|next| next <= resident_budget);
            if admitted {
                resident_bytes += bytes;
                CachePolicy::Resident
            } else {
                CachePolicy::Streaming
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    SpineCachePlan {
        stream_nocache: true,
        policies,
    }
}

fn select_residency_with_transient(
    host: HostMemory,
    provider: ProviderMemory,
    layer_bytes: &[u64],
    base_fixed: FixedCosts,
    request: ResidencyOverride,
) -> Result<(ResidencySelection, u64)> {
    if layer_bytes.is_empty() {
        return Ok((
            select_resident_prefix(
                host,
                provider,
                layer_bytes,
                base_fixed,
                request,
                ResidencyPolicy::default(),
            ),
            0,
        ));
    }

    let mut best: Option<(ResidencySelection, u64)> = None;
    // Evaluate every possible ordered prefix. Reserving the global maximum up
    // front can double-count layer zero and miss a feasible larger prefix;
    // the exact suffix maximum removes that circularity without guessing.
    for candidate_layers in 0..=layer_bytes.len() {
        let transient_layer_bytes = layer_bytes[candidate_layers..]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let Some(provider_bytes) = base_fixed.provider_bytes.checked_add(transient_layer_bytes)
        else {
            continue;
        };
        let selection = select_resident_prefix(
            host,
            provider,
            layer_bytes,
            FixedCosts {
                host_bytes: base_fixed.host_bytes,
                provider_bytes,
            },
            request,
            ResidencyPolicy::default(),
        );
        if selection.resident_layers >= candidate_layers {
            best = Some((selection, transient_layer_bytes));
        }
    }
    best.ok_or_else(|| DeltafinError::new("no safe transient-layer residency state exists"))
}

fn verify_snapshot_budget(model: &ModelSpec) -> Result<VerifySnapshotBudget> {
    k3_verify_snapshot_budget(model.kda_layers() as u64)
}

fn k3_verify_snapshot_budget(kda_layers: u64) -> Result<VerifySnapshotBudget> {
    let bytes_per_kda_generation = kda_layers
        .checked_mul(
            K3_KDA_CONV_ELEMENTS_PER_LAYER
                .checked_add(K3_KDA_RECURRENT_ELEMENTS_PER_LAYER)
                .ok_or_else(|| DeltafinError::new("KDA cache element budget overflows u64"))?,
        )
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| DeltafinError::new("KDA cache byte budget overflows u64"))?;
    // Per row and layer: decay, key and value rows, beta, and one column of
    // each of the three convolution sources, all fp32.
    let replay_bytes_per_position = kda_layers
        .checked_mul(6 * K3_KDA_PROJECTION + K3_KDA_HEADS)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| DeltafinError::new("KDA replay budget overflows u64"))?;
    let replay_fixed_bytes = kda_layers
        .checked_mul(3 * K3_KDA_PROJECTION * (K3_KDA_CONVOLUTION_WIDTH - 1))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| DeltafinError::new("KDA replay budget overflows u64"))?;
    Ok(VerifySnapshotBudget {
        bytes_per_kda_generation,
        replay_bytes_per_position,
        replay_fixed_bytes,
        max_positions: TARGET_SEQUENCE_MAX_POSITIONS,
    })
}

fn context_growth_budget(model: &ModelSpec) -> Result<ContextGrowthBudget> {
    let key_columns = (model.qk_nope_head_dim as u64)
        .checked_add(model.qk_rope_head_dim as u64)
        .ok_or_else(|| DeltafinError::new("MLA key width overflows u64"))?;
    let stored_columns = key_columns
        .checked_add(model.value_head_dim as u64)
        .ok_or_else(|| DeltafinError::new("MLA stored width overflows u64"))?;
    let bytes_per_layer_token = (model.num_attention_heads as u64)
        .checked_mul(stored_columns)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| DeltafinError::new("per-layer MLA storage budget overflows u64"))?;
    let mla_layers = model.mla_layers() as u64;
    let bytes_per_capacity_token = mla_layers
        .checked_mul(bytes_per_layer_token)
        .ok_or_else(|| DeltafinError::new("MLA context-growth budget overflows u64"))?;
    let initial_provider_bytes = bytes_per_capacity_token
        .checked_mul(K3_MLA_INITIAL_CAPACITY)
        .ok_or_else(|| DeltafinError::new("initial MLA cache budget overflows u64"))?;
    let model_max_context_tokens = u64::try_from(model.max_position_embeddings)
        .map_err(|_| DeltafinError::new("model context bound does not fit u64"))?;
    let admitted_expanded_context_tokens = (K3_MLA_STORAGE_BUDGET_PER_LAYER_BYTES
        / bytes_per_layer_token)
        .min(model_max_context_tokens);
    if admitted_expanded_context_tokens < K3_MLA_INITIAL_CAPACITY {
        return Err(DeltafinError::new(
            "expanded MLA storage budget cannot admit its initial capacity",
        ));
    }
    Ok(ContextGrowthBudget {
        bytes_per_capacity_token,
        bytes_per_layer_capacity_token: bytes_per_layer_token,
        mla_layers,
        initial_capacity_tokens: K3_MLA_INITIAL_CAPACITY,
        initial_provider_bytes,
        model_max_context_tokens,
        admitted_expanded_context_tokens,
    })
}

fn component_high_water(layers: &[LayerSpinePlan]) -> Result<u64> {
    buffer_high_water(layers.iter().map(LayerSpinePlan::buffer_lengths))
}

fn buffer_high_water(lengths: impl IntoIterator<Item = BufferLengths>) -> Result<u64> {
    let high_water = lengths
        .into_iter()
        .fold(BufferLengths::default(), |maximum, lengths| {
            BufferLengths::new(
                maximum.quantized.max(lengths.quantized),
                maximum.scales.max(lengths.scales),
                maximum.other.max(lengths.other),
            )
        });
    (high_water.quantized as u64)
        .checked_add(high_water.scales as u64)
        .and_then(|bytes| bytes.checked_add(high_water.other as u64))
        .ok_or_else(|| DeltafinError::new("spine component high-water mark overflows u64"))
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct CudaExpertCacheBudget {
    policy: CudaExpertCachePolicy,
    /// Bytes charged into residency's fixed provider costs so the resident
    /// spine prefix can only claim VRAM the expert cache will not use.
    charged_provider_bytes: u64,
}

/// Split discrete VRAM between the CUDA expert cache and every other
/// provider consumer before residency selection runs.
///
/// Expert misses sit on the routing-dependent critical path: they cannot be
/// read before the router runs, so every miss pays a demand disk read plus a
/// PCIe upload inside the layer. Spine streaming reads prefetch one layer
/// ahead and overlap compute. The cache therefore takes VRAM priority, and
/// resident spine layers receive only what remains — the same split the
/// measured Python v4 pipeline used. The envelope mirrors the residency
/// policy exactly (device reserve, then discrete utilization), so the two
/// budgets combined can never exceed what residency alone was allowed.
fn plan_cuda_expert_cache_budget(
    device: Device,
    snapshot: NativeProviderMemorySnapshot,
    fixed_provider_bytes: u64,
    peak_transient_layer_bytes: u64,
) -> Option<CudaExpertCacheBudget> {
    if !matches!(device, Device::Cuda(_)) {
        return None;
    }
    let policy = ResidencyPolicy::default();
    let zero = CudaExpertCacheBudget {
        policy: CudaExpertCachePolicy {
            capacity_experts: 0,
            reserve_bytes: 0,
        },
        charged_provider_bytes: 0,
    };
    let (Some(total), Some(available)) = (snapshot.total_bytes, snapshot.available_bytes) else {
        // Without trustworthy discrete accounting residency also selects
        // nothing; an explicit zero keeps the provider from budgeting later
        // against unattributed free VRAM and colliding with other consumers.
        return Some(zero);
    };
    let reserve = policy
        .device_reserve_floor_bytes
        .max(total.saturating_mul(u64::from(policy.device_reserve_permille)) / 1_000)
        .min(total);
    let raw = total
        .saturating_sub(reserve)
        .min(available.min(total).saturating_sub(reserve));
    let Some(envelope) = raw
        .checked_mul(u64::from(policy.discrete_utilization_permille))
        .map(|bytes| bytes / 1_000)
    else {
        return Some(zero);
    };
    let usable = envelope
        .saturating_sub(fixed_provider_bytes)
        .saturating_sub(peak_transient_layer_bytes);
    let span = crate::experts::K3_EXPERT_SOURCE_BYTES as u64;
    let capacity = u32::try_from((usable / span).min(u64::from(CUDA_EXPERT_CACHE_MAX_EXPERTS)))
        .expect("bounded CUDA expert-cache capacity fits u32");
    Some(CudaExpertCacheBudget {
        policy: CudaExpertCachePolicy {
            capacity_experts: capacity,
            reserve_bytes: reserve,
        },
        charged_provider_bytes: u64::from(capacity) * span,
    })
}

fn provider_memory(snapshot: NativeProviderMemorySnapshot) -> ProviderMemory {
    match snapshot.device {
        Device::Cpu => ProviderMemory::Host,
        Device::Mps => ProviderMemory::Unified {
            // Metal's recommendedMaxWorkingSetSize is advisory; LibTorch's
            // supported unified-memory allocator deliberately permits a
            // higher watermark. Treating the recommendation as a hard cap
            // dropped this M1 Max from six resident layers to five even
            // though the existing host envelope remained safe. The live host
            // snapshot is authoritative for unified memory; provider metrics
            // remain available for trim diagnostics without reducing useful
            // residency.
            recommended_working_set_bytes: None,
            available_working_set_bytes: None,
        },
        Device::Cuda(_) => ProviderMemory::Discrete {
            // Never infer VRAM from host RAM. A CUDA build which cannot expose
            // both values remains unknown and therefore fails closed.
            total_bytes: snapshot.total_bytes,
            available_bytes: snapshot.available_bytes,
        },
    }
}

fn optional_mib(bytes: Option<u64>) -> String {
    bytes.map_or_else(
        || "unknown".into(),
        |value| format!("{:.2} MiB", value as f64 / (1_u64 << 20) as f64),
    )
}

fn account_trimmed_unified_memory(
    mut host: HostMemory,
    before: NativeProviderMemorySnapshot,
    after: NativeProviderMemorySnapshot,
) -> HostMemory {
    if before.device != Device::Mps || after.device != Device::Mps || !after.cache_trimmed {
        return host;
    }
    let reclaimed = before
        .reserved_bytes
        .zip(after.reserved_bytes)
        .map_or(0, |(before, after)| before.saturating_sub(after));
    if reclaimed == 0 {
        return host;
    }
    // MPS driverAllocatedMemory includes allocator heaps and MPSGraph-owned
    // storage. A measured decrease after synchronize+emptyCache proves that
    // those unified pages are no longer provider-owned. Mach may temporarily
    // classify the pages as inactive anonymous rather than free/external;
    // add only this causal delta to the fresh snapshot, retain the ordinary
    // 18%/10-GiB host reserve, and never manufacture availability when the
    // host query itself failed.
    host.available_bytes = host.available_bytes.map(|available| {
        available
            .saturating_add(reclaimed)
            .min(host.physical_bytes.unwrap_or(u64::MAX))
    });
    host
}

fn admit_live_context_growth(
    host: HostMemory,
    provider: ProviderMemory,
    admission: ContextGrowthAdmission,
) -> Result<()> {
    // The committed MLA allocation is already reflected in this fresh memory
    // snapshot. Charge the complete new staged generation plus the bounded
    // intermediate reallocations which may remain in the provider allocator
    // while all 24 layers build their unpublished generation.
    let live_new_provider_bytes = admission
        .staged_provider_bytes
        .checked_add(admission.growth_scratch_provider_bytes)
        .ok_or_else(|| DeltafinError::new("live MLA admission bytes overflow u64"))?;
    let selection = select_resident_prefix(
        host,
        provider,
        &[],
        FixedCosts {
            host_bytes: 0,
            provider_bytes: live_new_provider_bytes,
        },
        ResidencyOverride::default(),
        ResidencyPolicy::default(),
    );
    if selection.stop != ResidencyStop::AllLayersFit {
        return Err(DeltafinError::new(format!(
            "live memory snapshot cannot safely admit MLA growth {}->{} tokens (new staged {:.2} MiB, growth scratch {:.2} MiB, old+new peak {:.2} MiB, host total={}, host available={}, host envelope={}, provider envelope={}, policy stop {:?})",
            admission.committed_capacity_tokens,
            admission.next_capacity_tokens,
            admission.staged_provider_bytes as f64 / (1_u64 << 20) as f64,
            admission.growth_scratch_provider_bytes as f64 / (1_u64 << 20) as f64,
            admission.transaction_peak_provider_bytes as f64 / (1_u64 << 20) as f64,
            optional_mib(host.effective_total_bytes()),
            optional_mib(host.effective_available_bytes()),
            optional_mib(selection.host_envelope_bytes),
            optional_mib(selection.provider_envelope_bytes),
            selection.stop,
        )));
    }
    Ok(())
}

fn bounded_worker_count(limit: usize) -> usize {
    std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, limit)
}

fn configured_expert_reader_workers(configured: Option<usize>, homes: usize) -> usize {
    // Each extra drive brings its own queue depth; give it its own workers
    // so a batch can keep every home busy at once.
    let automatic = if homes > 1 {
        (bounded_worker_count(EXPERT_READER_LIMIT) * homes)
            .min(crate::config::MAX_EXPERT_READ_THREADS)
    } else {
        bounded_worker_count(EXPERT_READER_LIMIT)
    };
    configured
        .unwrap_or(automatic)
        .clamp(1, crate::config::MAX_EXPERT_READ_THREADS)
}

fn resolve_loose_spine_fd_cache(
    configured: Option<bool>,
    loose_spine: bool,
    automatic_streaming: bool,
    qualified_resource_tuple: bool,
) -> bool {
    loose_spine && configured.unwrap_or(automatic_streaming && qualified_resource_tuple)
}

fn configured_spine_reader_workers(
    configured: Option<usize>,
    stream_nocache: bool,
    host: HostMemory,
    provider: NativeProviderMemorySnapshot,
    max_buffer_length: Option<u64>,
) -> usize {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    resolve_spine_reader_workers(
        configured,
        cfg!(target_os = "macos"),
        stream_nocache,
        host.physical_bytes,
        available,
        provider.recommended_bytes,
        max_buffer_length,
    )
}

fn resolve_spine_reader_workers(
    configured: Option<usize>,
    is_macos: bool,
    stream_nocache: bool,
    physical_bytes: Option<u64>,
    available_cpus: usize,
    recommended_working_set_bytes: Option<u64>,
    max_buffer_length_bytes: Option<u64>,
) -> usize {
    if let Some(configured) = configured {
        debug_assert!((1..=crate::config::MAX_SPINE_READ_THREADS).contains(&configured));
        return configured.clamp(1, crate::config::MAX_SPINE_READ_THREADS);
    }
    let measured_resource_tuple = stream_nocache
        && qualified_spine_resource_tuple(
            is_macos,
            physical_bytes,
            available_cpus,
            recommended_working_set_bytes,
            max_buffer_length_bytes,
        );
    let limit = if measured_resource_tuple {
        QUALIFIED_SPINE_READER_LIMIT
    } else {
        SPINE_READER_LIMIT
    };
    available_cpus.clamp(1, limit)
}

fn qualified_spine_resource_tuple(
    is_macos: bool,
    physical_bytes: Option<u64>,
    available_cpus: usize,
    recommended_working_set_bytes: Option<u64>,
    max_buffer_length_bytes: Option<u64>,
) -> bool {
    is_macos
        && available_cpus == 10
        && physical_bytes.is_some_and(|bytes| {
            bytes.abs_diff(QUALIFIED_PHYSICAL_BYTES) <= QUALIFIED_RESOURCE_SLOP_BYTES
        })
        && recommended_working_set_bytes.is_some_and(|bytes| {
            bytes.abs_diff(QUALIFIED_RECOMMENDED_BYTES) <= QUALIFIED_RESOURCE_SLOP_BYTES
        })
        && max_buffer_length_bytes.is_some_and(|bytes| {
            bytes.abs_diff(QUALIFIED_MAX_BUFFER_BYTES) <= QUALIFIED_RESOURCE_SLOP_BYTES
        })
}

#[cfg(target_os = "macos")]
fn metal_max_buffer_length(device: Device) -> Option<u64> {
    use std::ffi::{c_char, c_void};

    if device != Device::Mps {
        return None;
    }
    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }
    #[link(name = "objc")]
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        #[link_name = "objc_msgSend"]
        fn objc_msg_send_usize(receiver: *mut c_void, selector: *mut c_void) -> usize;
        fn objc_release(object: *mut c_void);
    }

    // SAFETY: the process already linked and initialized the selected MPS
    // provider. Metal returns one retained default-device object; the selector
    // takes no arguments and returns NSUInteger on supported macOS/aarch64.
    let metal_device = unsafe { MTLCreateSystemDefaultDevice() };
    if metal_device.is_null() {
        return None;
    }
    // SAFETY: the static selector spelling is NUL terminated, and the live
    // object conforms to MTLDevice. Release balances the Create ownership.
    let value = unsafe {
        let selector = sel_registerName(c"maxBufferLength".as_ptr());
        let value = if selector.is_null() {
            0
        } else {
            objc_msg_send_usize(metal_device, selector)
        };
        objc_release(metal_device);
        value
    };
    u64::try_from(value).ok().filter(|value| *value != 0)
}

#[cfg(not(target_os = "macos"))]
const fn metal_max_buffer_length(_device: Device) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::RunArgs;
    use crate::model::LayerKind;
    use crate::program::{
        SPINE_BUFFER_NONE, SPINE_BUFFER_OTHER, SPINE_ENCODING_RAW_BF16, SpineTensorDescriptorV1,
    };
    use crate::residency::{GIB, HostMemory};
    use crate::storage::{BufferKind, CachePolicy, Extent, ReadPlan};

    #[test]
    fn context_envelope_covers_bounded_requests_and_declines_the_rest() {
        // The benchmark shape that failed overnight: ~45-token prompt,
        // --max-new 150, 8 speculative drafts -> a 204-token ceiling, well
        // inside both the pre-admission bound and the expanded-context cap.
        assert_eq!(
            context_pre_admission_envelope(45, Some(150), 8, 4_369),
            Some(204)
        );
        // Slack covers the widest staged verify chunk beyond the prefix.
        assert_eq!(context_pre_admission_envelope(1, Some(1), 0, 4_369), Some(3));

        // Unbounded requests keep incremental growth: there is no envelope.
        assert_eq!(context_pre_admission_envelope(45, None, 8, 4_369), None);
        // A server-wide output ceiling is not an expected length; oversized
        // envelopes decline rather than reserving gigabytes nobody will use.
        assert_eq!(
            context_pre_admission_envelope(45, Some(1_000_000), 8, 4_369),
            None
        );
        assert_eq!(context_pre_admission_envelope(400, Some(150), 8, 4_369), None);
        // The expanded-context cap binds when it is smaller than the bound.
        assert_eq!(context_pre_admission_envelope(45, Some(150), 8, 128), None);
        // Exactly at the bound is still worth proving once.
        assert_eq!(
            context_pre_admission_envelope(303, Some(200), 8, 4_369),
            Some(512)
        );
        // Arithmetic overflow declines instead of wrapping.
        assert_eq!(
            context_pre_admission_envelope(usize::MAX, Some(usize::MAX), 8, u64::MAX),
            None
        );
    }

    #[test]
    fn target_phase_profile_aggregates_additive_waits_without_double_counting_overlap() {
        let mut first = TargetExecutionProfile::default();
        first.chunks = 1;
        first.sequence_total_ns = 1_000;
        first.tail_head_sync_ns = 100;
        first.layers[0] = TargetLayerPhaseProfile {
            passes: 1,
            spine_read_bytes: 64,
            spine_read_active_ns: 700,
            spine_read_wait_ns: 100,
            spine_prefetch_submit_ns: 10,
            spine_bind_upload_ns: 200,
            attention_resident_compute_ns: 150,
            authoritative_expert_read_prefetch_ns: 80,
            expert_kernel_ns: 200,
            source_fence_ns: 10,
            layer_total_ns: 800,
            expert_plan_hits: 12,
            expert_plan_misses: 4,
            expert_early_drained: 3,
        };
        // The 700ns active-read interval overlaps preceding work. Only the
        // 100ns caller wait contributes to the additive layer attribution.
        assert_eq!(first.layers[0].attributed_ns(), 750);
        assert_eq!(first.layers[0].other_control_ns(), 50);

        let mut aggregate = TargetExecutionProfile::default();
        aggregate.absorb(&first);
        aggregate.absorb(&first);
        let json = aggregate.json();
        assert_eq!(json["chunks"], 2);
        assert_eq!(json["totals"]["sequence_total_ns"], 2_000);
        assert_eq!(json["totals"]["spine_read_active_ns"], 1_400);
        assert_eq!(json["totals"]["spine_read_wait_ns"], 200);
        assert_eq!(json["totals"]["layer_other_control_ns"], 100);
        assert_eq!(json["totals"]["sequence_control_ns"], 200);
        assert_eq!(json["layers"].as_array().unwrap().len(), 1);
        assert_eq!(json["layers"][0]["passes"], 2);
        assert_eq!(json["totals"]["expert_plan_hits"], 24);
        assert_eq!(json["totals"]["expert_plan_misses"], 8);
        // Plan counters are informational and must never enter the additive
        // time attribution.
        assert_eq!(json["layers"][0]["expert_plan_hits"], 24);
    }

    #[test]
    fn cuda_expert_cache_budget_shares_one_envelope_with_residency() {
        let snapshot = |total: Option<u64>, available: Option<u64>| NativeProviderMemorySnapshot {
            device: Device::Cuda(0),
            active_bytes: None,
            reserved_bytes: None,
            recommended_bytes: None,
            total_bytes: total,
            available_bytes: available,
            cache_trimmed: false,
        };
        let gib = 1_u64 << 30;
        let span = crate::experts::K3_EXPERT_SOURCE_BYTES as u64;

        // Non-CUDA devices have no CUDA cache to budget.
        assert_eq!(
            plan_cuda_expert_cache_budget(
                Device::Cpu,
                snapshot(Some(32 * gib), Some(30 * gib)),
                0,
                0
            ),
            None,
        );

        // Unknown discrete accounting fails closed to an explicit zero, the
        // same outcome residency reaches for an unknown discrete envelope.
        let unknown =
            plan_cuda_expert_cache_budget(Device::Cuda(0), snapshot(None, Some(30 * gib)), 0, 0)
                .unwrap();
        assert_eq!(unknown.policy.capacity_experts, 0);
        assert_eq!(unknown.charged_provider_bytes, 0);

        // A 32 GiB card with 30 GiB free: reserve is max(2 GiB, 3.2 GiB),
        // envelope is 85% of (30 - 3.2) GiB, and the remainder after fixed
        // and transient charges converts to whole expert spans.
        let total = 32 * gib;
        let available = 30 * gib;
        let fixed = 2 * gib;
        let transient = gib;
        let budget = plan_cuda_expert_cache_budget(
            Device::Cuda(0),
            snapshot(Some(total), Some(available)),
            fixed,
            transient,
        )
        .unwrap();
        let reserve = (total / 10).max(2 * gib);
        let envelope = (available - reserve) * 850 / 1_000;
        let expected = ((envelope - fixed - transient) / span)
            .min(u64::from(CUDA_EXPERT_CACHE_MAX_EXPERTS)) as u32;
        assert_eq!(budget.policy.capacity_experts, expected);
        assert!(budget.policy.capacity_experts > 0);
        assert_eq!(budget.policy.reserve_bytes, reserve);
        assert_eq!(budget.charged_provider_bytes, u64::from(expected) * span,);
        // The charge plus fixed costs stays inside the shared envelope, so
        // residency cannot select spine layers the cache will also claim.
        assert!(budget.charged_provider_bytes + fixed + transient <= envelope);

        // Plentiful VRAM clamps at the provider ABI bound instead of growing
        // past one entry per routed edge.
        let large = plan_cuda_expert_cache_budget(
            Device::Cuda(0),
            snapshot(Some(96 * gib), Some(90 * gib)),
            0,
            0,
        )
        .unwrap();
        assert_eq!(large.policy.capacity_experts, CUDA_EXPERT_CACHE_MAX_EXPERTS);

        // A card whose free VRAM sits inside the device reserve configures an
        // explicit zero instead of leaving the provider free to budget later.
        let starved = plan_cuda_expert_cache_budget(
            Device::Cuda(0),
            snapshot(Some(3 * gib), Some(3 * gib / 2)),
            0,
            0,
        )
        .unwrap();
        assert_eq!(starved.policy.capacity_experts, 0);
        assert_eq!(starved.charged_provider_bytes, 0);

        // Fixed provider charges below the envelope shrink capacity expert
        // by expert rather than collapsing it.
        let squeezed = plan_cuda_expert_cache_budget(
            Device::Cuda(0),
            snapshot(Some(32 * gib), Some(30 * gib)),
            20 * gib,
            gib,
        )
        .unwrap();
        assert!(squeezed.policy.capacity_experts > 0);
        assert!(squeezed.policy.capacity_experts < budget.policy.capacity_experts);
    }

    #[test]
    fn original_bf16_cuda_selection_requires_the_compiled_exact_kernel() {
        let without_kernel = NativeProviderInventory {
            providers: ProviderInventory {
                mps: false,
                cuda_devices: 2,
            },
            cuda_moe_compiled: true,
            cuda_exact_bf16_compiled: false,
            libtorch_version: "test".to_owned(),
        };
        assert_eq!(
            select_target_for_spine(
                &without_kernel,
                DeviceRequest::Auto,
                SpineRepresentation::OriginalBf16,
                None,
            )
            .unwrap()
            .device,
            Device::Cpu,
        );
        let explicit = select_target_for_spine(
            &without_kernel,
            DeviceRequest::Cuda(1),
            SpineRepresentation::OriginalBf16,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(explicit.contains("exact RAW_BF16 CUDA kernel"));
        assert!(explicit.contains("NVCC"));
        assert!(explicit.contains("--spine int8"));
        assert!(explicit.contains("will not expand"));
        assert_eq!(
            select_target_for_spine(
                &without_kernel,
                DeviceRequest::Auto,
                SpineRepresentation::QuantizedInt8,
                None,
            )
            .unwrap()
            .device,
            Device::Cuda(0),
        );

        let with_kernel = NativeProviderInventory {
            cuda_exact_bf16_compiled: true,
            ..without_kernel
        };
        assert_eq!(
            select_target_for_spine(
                &with_kernel,
                DeviceRequest::Auto,
                SpineRepresentation::OriginalBf16,
                None,
            )
            .unwrap()
            .device,
            Device::Cuda(0),
        );
    }

    fn test_target_reuse_identity(seed: u8) -> TargetReuseIdentity {
        let mut model_inventory = [0_u8; 32];
        model_inventory[0] = seed;
        TargetReuseIdentity {
            model_inventory,
            device: Device::Cpu,
            spine: SpineRepresentation::OriginalBf16,
            expert_backend: ResolvedExpertBackend::Cpu,
            expert_storage: ExpertStorageLayout::RawV1,
            expert_cpu_threads: 1,
            provider_abi: 1,
            transaction_contract: 1,
        }
    }

    #[test]
    fn scale4_auto_and_require_policy_never_changes_cpu_or_cuda_layouts() {
        assert!(
            !admit_scale4_storage(ExpertScale4Request::Auto, ResolvedExpertBackend::Cpu, true,)
                .unwrap()
        );
        assert!(
            !admit_scale4_storage(ExpertScale4Request::Auto, ResolvedExpertBackend::Cuda, true,)
                .unwrap()
        );
        assert!(
            admit_scale4_storage(
                ExpertScale4Request::Require,
                ResolvedExpertBackend::Cpu,
                false,
            )
            .unwrap_err()
            .to_string()
            .contains("CPU and CUDA remain raw-v1")
        );
    }

    #[test]
    fn scale4_auto_needs_metal_capability_and_off_is_explicit_raw() {
        assert!(
            !admit_scale4_storage(ExpertScale4Request::Off, ResolvedExpertBackend::Metal, true,)
                .unwrap()
        );
        assert!(
            !admit_scale4_storage(
                ExpertScale4Request::Auto,
                ResolvedExpertBackend::Metal,
                false,
            )
            .unwrap()
        );
        assert!(
            admit_scale4_storage(
                ExpertScale4Request::Auto,
                ResolvedExpertBackend::Metal,
                true,
            )
            .unwrap()
        );
        assert!(
            admit_scale4_storage(
                ExpertScale4Request::Require,
                ResolvedExpertBackend::Metal,
                false,
            )
            .is_err()
        );
    }
    use std::fs::{self, File};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    struct FakeTargetBranch(u64);

    #[derive(Debug)]
    struct FakeTransactionalTargetState {
        committed_positions: u64,
        parent_positions: Option<u64>,
        active_branch: Option<FakeTargetBranch>,
        next_branch: u64,
    }

    impl FakeTransactionalTargetState {
        fn new(committed_positions: u64) -> Self {
            Self {
                committed_positions,
                parent_positions: None,
                active_branch: None,
                next_branch: 1,
            }
        }

        fn advance_private_branch(&mut self, committed_positions: u64) {
            assert!(self.active_branch.is_some());
            assert!(committed_positions >= self.committed_positions);
            self.committed_positions = committed_positions;
        }

        fn take_branch(&mut self, branch: FakeTargetBranch) -> Result<u64> {
            if self.active_branch != Some(branch) {
                return Err(DeltafinError::new(
                    "target state branch is stale or not active",
                ));
            }
            self.active_branch = None;
            self.parent_positions
                .take()
                .ok_or_else(|| DeltafinError::new("target state branch lost its parent"))
        }
    }

    impl TargetStateCapabilityProvider for FakeTransactionalTargetState {
        fn target_state_transaction_capability(&self) -> TargetStateTransactionCapability {
            TargetStateTransactionCapability::RequestBranchV1
        }
    }

    impl TransactionalTargetStateProvider for FakeTransactionalTargetState {
        type RequestBranch = FakeTargetBranch;

        fn begin_target_state_branch(
            &mut self,
            expected_committed_positions: u64,
        ) -> Result<Self::RequestBranch> {
            if self.active_branch.is_some() {
                return Err(DeltafinError::new(
                    "a target state branch is already active",
                ));
            }
            if self.committed_positions != expected_committed_positions {
                return Err(DeltafinError::new(
                    "target state branch position differs from the exact prefix plan",
                ));
            }
            let branch = FakeTargetBranch(self.next_branch);
            self.next_branch += 1;
            self.parent_positions = Some(self.committed_positions);
            self.active_branch = Some(branch);
            Ok(branch)
        }

        fn publish_target_state_branch(&mut self, branch: Self::RequestBranch) -> Result<()> {
            self.take_branch(branch)?;
            Ok(())
        }

        fn discard_target_state_branch(&mut self, branch: Self::RequestBranch) -> Result<()> {
            self.committed_positions = self.take_branch(branch)?;
            Ok(())
        }
    }

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new() -> Self {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-engine-bootstrap-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn production_metal_uses_the_versioned_embedded_source_selector() {
        assert_eq!(
            resolve_metal_source_with_policy(ResolvedExpertBackend::Metal, None).unwrap(),
            Some(K3_METAL_EMBEDDED_SOURCE_V1.to_owned())
        );
        assert_eq!(
            resolve_metal_source_with_policy(ResolvedExpertBackend::Cpu, None).unwrap(),
            None,
            "non-Metal platforms must never resolve or require MSL"
        );
    }

    #[test]
    fn explicit_metal_source_override_is_rejected_by_every_product_build() {
        for backend in [ResolvedExpertBackend::Metal, ResolvedExpertBackend::Cpu] {
            let error = resolve_metal_source_with_policy(
                backend,
                Some(OsStr::new("/not/a/product-input.metal")),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("disabled in every Deltafin product build"));
            assert!(error.contains("embedded"));
        }
    }

    fn synthetic_layer(index: u32, lengths: BufferLengths) -> LayerSpinePlan {
        let mut extents = Vec::new();
        for (kind, length) in [
            (BufferKind::Quantized, lengths.quantized),
            (BufferKind::Scales, lengths.scales),
            (BufferKind::Other, lengths.other),
        ] {
            if length != 0 {
                extents.push(Extent::zero(kind, 0, length));
            }
        }
        let read_plan = ReadPlan::open(extents, lengths, 0, CachePolicy::Streaming).unwrap();
        LayerSpinePlan {
            layer: index,
            kind: LayerKind::Kda,
            descriptors: vec![SpineTensorDescriptorV1 {
                slot: 1,
                encoding: SPINE_ENCODING_RAW_BF16,
                rank: 1,
                data_buffer: SPINE_BUFFER_OTHER,
                auxiliary_buffer: SPINE_BUFFER_NONE,
                reserved0: 0,
                shape: [1, 0, 0, 0, 0, 0, 0, 0],
                data_offset: 0,
                data_length: lengths.other as u64,
                auxiliary_offset: 0,
                auxiliary_length: 0,
                reserved: [0; 4],
            }]
            .into_boxed_slice(),
            buffer_lengths: lengths,
            read_plan,
        }
    }

    #[test]
    fn component_budget_uses_per_buffer_high_water_not_an_impossible_layer_sum() {
        let layers = vec![
            synthetic_layer(0, BufferLengths::new(100, 2, 3)),
            synthetic_layer(1, BufferLengths::new(4, 80, 7)),
            synthetic_layer(2, BufferLengths::new(8, 9, 60)),
        ];
        assert_eq!(component_high_water(&layers).unwrap(), 100 + 80 + 60);
    }

    #[test]
    fn cuda_residency_fails_closed_without_a_live_vram_budget() {
        let selection = select_resident_prefix(
            HostMemory {
                physical_bytes: Some(128 << 30),
                available_bytes: Some(100 << 30),
                cgroup_limit_bytes: None,
                cgroup_available_bytes: None,
                constraints_readable: true,
            },
            provider_memory(NativeProviderMemorySnapshot {
                device: Device::Cuda(0),
                active_bytes: None,
                reserved_bytes: None,
                recommended_bytes: None,
                total_bytes: None,
                available_bytes: None,
                cache_trimmed: false,
            }),
            &[1 << 30, 1 << 30],
            FixedCosts::default(),
            ResidencyOverride::default(),
            ResidencyPolicy::default(),
        );
        assert_eq!(selection.resident_layers, 0);
        assert_eq!(
            selection.stop,
            crate::residency::ResidencyStop::DeviceBudgetUnknown
        );
    }

    #[test]
    fn residency_evaluates_the_exact_suffix_transient_without_double_counting_layer_zero() {
        let gib = 1_u64 << 30;
        let host = HostMemory {
            physical_bytes: Some(20 * gib),
            available_bytes: Some(20 * gib),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        };
        let (selection, transient) = select_residency_with_transient(
            host,
            ProviderMemory::Host,
            &[6 * gib, 2 * gib],
            FixedCosts::default(),
            ResidencyOverride::default(),
        )
        .unwrap();
        assert_eq!(selection.resident_layers, 2);
        assert_eq!(selection.resident_provider_bytes, 8 * gib);
        assert_eq!(transient, 0);
    }

    #[test]
    fn provider_prefix_control_preserves_auto_zero_and_safety_clamping() {
        let gib = 1_u64 << 30;
        let host = HostMemory {
            physical_bytes: Some(20 * gib),
            available_bytes: Some(20 * gib),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        };
        let layer_bytes = [6 * gib, 2 * gib];

        let (automatic, automatic_transient) = select_residency_with_transient(
            host,
            ProviderMemory::Host,
            &layer_bytes,
            FixedCosts::default(),
            ResidencyOverride::default(),
        )
        .unwrap();
        assert_eq!(automatic.resident_layers, 2);
        assert_eq!(automatic_transient, 0);

        let zero_request = ResidencyOverride {
            requested_layers: Some(0),
            requested_provider_bytes: None,
        };
        let (zero, zero_transient) = select_residency_with_transient(
            host,
            ProviderMemory::Host,
            &layer_bytes,
            FixedCosts::default(),
            zero_request,
        )
        .unwrap();
        assert_eq!(zero.resident_layers, 0);
        assert_eq!(zero.resident_provider_bytes, 0);
        assert_eq!(zero.stop, ResidencyStop::ExplicitLayerLimit);
        assert_eq!(zero_transient, 6 * gib);

        let constrained_host = HostMemory {
            // The 10 GiB reserve leaves exactly 6 GiB for the mandatory
            // transient slot and no room for an additional retained layer.
            available_bytes: Some(16 * gib),
            ..host
        };
        let over_safe_limit = ResidencyOverride {
            requested_layers: Some(2),
            requested_provider_bytes: None,
        };
        let (clamped, transient) = select_residency_with_transient(
            constrained_host,
            ProviderMemory::Host,
            &layer_bytes,
            FixedCosts::default(),
            over_safe_limit,
        )
        .unwrap();
        assert_eq!(clamped.resident_layers, 0);
        assert!(clamped.override_clamped_by_safety);
        assert_eq!(transient, 6 * gib);
    }

    #[test]
    fn qwen_and_baseline_residency_branches_share_the_explicit_prefix_control() {
        let gib = 1_u64 << 30;
        let host = HostMemory {
            physical_bytes: Some(64 * gib),
            available_bytes: Some(64 * gib),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        };
        let provider = ProviderMemory::Unified {
            recommended_working_set_bytes: None,
            available_working_set_bytes: None,
        };
        let layer_bytes = [3 * gib, 3 * gib];
        let control = ResidencyOverride {
            requested_layers: Some(0),
            requested_provider_bytes: None,
        };
        let base = FixedCosts {
            host_bytes: 0,
            provider_bytes: 24 * gib,
        };
        let qwen = FixedCosts {
            host_bytes: 0,
            provider_bytes: 36 * gib,
        };

        for costs in [base, qwen] {
            let (selection, transient) =
                select_residency_with_transient(host, provider, &layer_bytes, costs, control)
                    .unwrap();
            assert_eq!(selection.resident_layers, 0);
            assert_eq!(selection.stop, ResidencyStop::ExplicitLayerLimit);
            assert_eq!(transient, 3 * gib);
        }
    }

    #[test]
    #[ignore = "requires an installed k3-meta fixture (deltafin setup); run with --include-ignored on a machine with K3 metadata present"]
    fn low_headroom_mps_int8_replays_the_two_layer_clean_file_tier() {
        const SPINE_SOURCE_BYTES: u64 = 54_397_786_304;
        let gib = 1_u64 << 30;
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = ModelSpec::load_from_root(&root).unwrap();
        let program =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        let layer_bytes = program.int8_stream_layer_bytes().unwrap();
        assert_eq!(layer_bytes.iter().sum::<u64>(), SPINE_SOURCE_BYTES);

        let plan = configured_spine_cache_plan(
            None,
            None,
            true,
            Device::Mps,
            SpineRepresentation::QuantizedInt8,
            Some(64 * gib),
            Some(55_662_788_608),
            &layer_bytes,
        );
        assert!(plan.stream_nocache);
        assert_eq!(
            plan.policies
                .iter()
                .filter(|policy| **policy == CachePolicy::Resident)
                .count(),
            2,
        );
        // This is intentionally a source-page policy. It does not fabricate a
        // provider ResidencyOverride or change K3 tensor authority.
        assert_eq!(plan.policies.len(), layer_bytes.len());
    }

    #[test]
    fn higher_memory_mps_int8_does_not_inherit_the_streaming_policy() {
        let gib = 1_u64 << 30;
        let plan = configured_spine_cache_plan(
            None,
            None,
            true,
            Device::Mps,
            SpineRepresentation::QuantizedInt8,
            Some(128 * gib),
            Some(112 * gib),
            &[54_397_786_304],
        );
        assert!(!plan.stream_nocache);
        assert_eq!(plan.policies.as_ref(), &[CachePolicy::Resident]);
    }

    #[test]
    fn automatic_streaming_policy_fails_closed_on_unknown_mps_capabilities() {
        let gib = 1_u64 << 30;
        let plan = configured_spine_cache_plan(
            None,
            None,
            true,
            Device::Mps,
            SpineRepresentation::QuantizedInt8,
            None,
            Some(52 * gib),
            &[54_397_786_304],
        );
        assert!(!plan.stream_nocache);
        assert_eq!(plan.policies.as_ref(), &[CachePolicy::Resident]);
    }

    #[test]
    fn non_darwin_cuda_and_bf16_keep_the_existing_native_stream_policy() {
        let gib = 1_u64 << 30;
        for plan in [
            configured_spine_cache_plan(
                None,
                None,
                false,
                Device::Mps,
                SpineRepresentation::QuantizedInt8,
                Some(64 * gib),
                Some(52 * gib),
                &[54_397_786_304],
            ),
            configured_spine_cache_plan(
                None,
                None,
                true,
                Device::Cuda(0),
                SpineRepresentation::QuantizedInt8,
                Some(64 * gib),
                Some(52 * gib),
                &[54_397_786_304],
            ),
            configured_spine_cache_plan(
                None,
                None,
                true,
                Device::Mps,
                SpineRepresentation::OriginalBf16,
                Some(64 * gib),
                Some(52 * gib),
                &[54_397_786_304],
            ),
        ] {
            assert!(plan.stream_nocache);
            assert_eq!(plan.policies.as_ref(), &[CachePolicy::Streaming]);
        }
    }

    #[test]
    fn explicit_streaming_and_resident_budget_remain_authoritative_cross_platform() {
        let plan = configured_spine_cache_plan(
            Some(true),
            Some(5),
            false,
            Device::Cuda(0),
            SpineRepresentation::QuantizedInt8,
            None,
            None,
            &[3, 2, 4],
        );
        assert!(plan.stream_nocache);
        assert_eq!(
            plan.policies.as_ref(),
            &[
                CachePolicy::Resident,
                CachePolicy::Resident,
                CachePolicy::Streaming,
            ]
        );

        let disabled = configured_spine_cache_plan(
            Some(false),
            Some(0),
            true,
            Device::Mps,
            SpineRepresentation::QuantizedInt8,
            Some(64 << 30),
            Some(52 << 30),
            &[54_397_786_304],
        );
        assert!(!disabled.stream_nocache);
        assert_eq!(disabled.policies.as_ref(), &[CachePolicy::Resident]);
    }

    #[test]
    #[ignore = "requires an installed k3-meta fixture (deltafin setup); run with --include-ignored on a machine with K3 metadata present"]
    fn compact_accelerator_spines_charge_one_shared_fp32_execution_arena() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = ModelSpec::load_from_root(&root).unwrap();
        let int8 =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        assert_eq!(
            fp32_spine_execution_arena_reserve(&int8, Device::Mps).unwrap(),
            4_680_974_336,
        );
        assert_eq!(
            fp32_spine_execution_arena_reserve(&int8, Device::Cpu).unwrap(),
            0,
        );
        assert_eq!(
            fp32_spine_execution_arena_reserve(&int8, Device::Cuda(0)).unwrap(),
            0,
        );

        // Original BF16 keeps compact checkpoint storage but expands one
        // current layer only on the independently qualified ATen/MPS path.
        // CUDA retains its existing direct raw-BF16 accounting until it has a
        // separate CUDA stream/throughput gate.
        let exact = TargetProgram::compile(&model).unwrap();
        assert_eq!(
            fp32_spine_execution_arena_reserve(&exact, Device::Mps).unwrap(),
            4_680_974_336,
        );
        assert_eq!(
            fp32_spine_execution_arena_reserve(&exact, Device::Cuda(0)).unwrap(),
            0,
        );
        assert_eq!(
            fp32_spine_execution_arena_reserve(&exact, Device::Cpu).unwrap(),
            0,
        );
    }

    #[test]
    fn explicit_expert_backends_must_match_a_linked_selected_accelerator() {
        let all = ProviderInventory {
            mps: true,
            cuda_devices: 2,
        };
        assert_eq!(
            resolve_expert_backend(ExpertBackendRequest::Auto, Device::Mps, all, true).unwrap(),
            ResolvedExpertBackend::Metal
        );
        assert_eq!(
            resolve_expert_backend(ExpertBackendRequest::Cpu, Device::Mps, all, true).unwrap(),
            ResolvedExpertBackend::Cpu
        );
        assert_eq!(
            resolve_expert_backend(ExpertBackendRequest::Auto, Device::Cuda(1), all, true).unwrap(),
            ResolvedExpertBackend::CudaAuto
        );
        assert_eq!(
            resolve_expert_backend(ExpertBackendRequest::Auto, Device::Cuda(1), all, false)
                .unwrap(),
            ResolvedExpertBackend::Cpu
        );
        assert!(
            resolve_expert_backend(ExpertBackendRequest::Metal, Device::Cpu, all, true).is_err()
        );
        assert_eq!(
            resolve_expert_backend(ExpertBackendRequest::Cuda, Device::Cuda(1), all, true).unwrap(),
            ResolvedExpertBackend::Cuda
        );
        assert!(
            resolve_expert_backend(ExpertBackendRequest::Cuda, Device::Cuda(1), all, false)
                .is_err()
        );
    }

    #[test]
    fn context_growth_exposes_old_plus_staged_transaction_peak() {
        let budget = ContextGrowthBudget {
            bytes_per_capacity_token: 100,
            bytes_per_layer_capacity_token: 25,
            mla_layers: 4,
            initial_capacity_tokens: 16,
            initial_provider_bytes: 1_600,
            model_max_context_tokens: 1_048_576,
            admitted_expanded_context_tokens: 4_369,
        };
        let first = budget.admission(0, 1).unwrap();
        assert_eq!(first.next_capacity_tokens, 16);
        assert_eq!(first.committed_provider_bytes, 0);
        assert_eq!(first.staged_provider_bytes, 1_600);
        assert_eq!(first.growth_scratch_provider_bytes, 0);
        assert_eq!(first.transaction_peak_provider_bytes, 1_600);

        let growth = budget.admission(16, 17).unwrap();
        assert_eq!(growth.next_capacity_tokens, 24);
        assert_eq!(growth.committed_provider_bytes, 1_600);
        assert_eq!(growth.staged_provider_bytes, 2_400);
        assert_eq!(growth.growth_scratch_provider_bytes, 0);
        assert_eq!(growth.transaction_peak_provider_bytes, 4_000);

        // The provider appends each row independently, so a 64-row initial
        // chunk follows 0->16->24->36->54->81 rather than jumping to 80.
        let wide = budget.admission(0, 64).unwrap();
        assert_eq!(wide.next_capacity_tokens, 81);
        assert_eq!(wide.staged_provider_bytes, 8_100);
        assert_eq!(wide.growth_scratch_provider_bytes, 3_250);
        assert_eq!(wide.transaction_peak_provider_bytes, 11_350);
        assert!(budget.admission(16, 16).is_err());
        assert!(budget.admission(4_369, 4_370).is_err());
        assert_eq!(budget.startup_growth_reserve().unwrap(), (256, 40_000));
    }

    #[test]
    fn client_presence_interrupt_mirrors_the_probe() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        assert!(!ClientPresenceInterrupt(&ClientPresence::assumed_present()).requested());
        let departed = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&departed);
        let client = ClientPresence::from_probe(Box::new(move || observed.load(Ordering::Relaxed)));
        let interrupt = ClientPresenceInterrupt(&client);
        assert!(!interrupt.requested());
        departed.store(true, Ordering::Relaxed);
        assert!(interrupt.requested());
    }

    #[test]
    fn lifecycle_returns_successes_to_ready_and_poisons_failures() {
        let mut success = NativeEngineLifecycle::Ready;
        success.begin().unwrap();
        assert_eq!(success, NativeEngineLifecycle::Running);
        success.publish().unwrap();
        assert_eq!(success, NativeEngineLifecycle::Ready);
        success.begin().unwrap();
        success.publish().unwrap();

        let mut failure = NativeEngineLifecycle::Ready;
        failure.begin().unwrap();
        failure.poison();
        assert_eq!(failure, NativeEngineLifecycle::Poisoned);
        assert!(failure.begin().is_err());

        let mut invalid_publish = NativeEngineLifecycle::Ready;
        assert!(invalid_publish.publish().is_err());
        assert_eq!(invalid_publish, NativeEngineLifecycle::Poisoned);
    }

    #[test]
    fn structured_event_helpers_preserve_eos_and_stop_reason() {
        assert_eq!(completion_token_ids(&[1, 2, 3], 3), &[1, 2]);
        assert_eq!(completion_token_ids(&[1, 2, 3], 4), &[1, 2, 3]);
        assert!(completion_token_ids(&[], 3).is_empty());
        assert_eq!(stop_reason_name(StopReason::Eos), "eos");
        assert_eq!(stop_reason_name(StopReason::MaxNew), "max_new");
        assert_eq!(stop_reason_name(StopReason::ContextFull), "context_full");
        assert_eq!(stop_reason_name(StopReason::Interrupted), "interrupted");
        assert_eq!(run_status_name(StopReason::Interrupted), "interrupted");
        assert_eq!(run_status_name(StopReason::Eos), "ok");
    }

    #[test]
    fn cooperative_stop_occurs_only_at_a_completed_transaction_boundary() {
        assert_eq!(
            stop_after_transaction(None, true),
            Some(StopReason::Interrupted)
        );
        assert_eq!(stop_after_transaction(None, false), None);
        assert_eq!(
            stop_after_transaction(Some(StopReason::Eos), true),
            Some(StopReason::Interrupted)
        );
        assert_eq!(
            stop_after_transaction(Some(StopReason::MaxNew), true),
            Some(StopReason::Interrupted)
        );
    }

    #[test]
    fn cpu_expert_thread_policy_matches_the_qualified_cross_platform_defaults() {
        assert_eq!(resolve_expert_cpu_threads(None, 16, false).unwrap(), 4);
        assert_eq!(resolve_expert_cpu_threads(None, 16, true).unwrap(), 8);
        assert_eq!(resolve_expert_cpu_threads(None, 2, false).unwrap(), 2);
        assert_eq!(resolve_expert_cpu_threads(None, 2, true).unwrap(), 2);
        assert_eq!(
            resolve_expert_cpu_threads(Some(" 12 "), 2, false).unwrap(),
            12
        );
        assert!(resolve_expert_cpu_threads(Some("0"), 64, true).is_err());
        assert!(resolve_expert_cpu_threads(Some("33"), 64, true).is_err());
        assert!(resolve_expert_cpu_threads(Some("many"), 64, true).is_err());
    }

    #[test]
    fn live_context_growth_uses_a_fresh_snapshot_and_charges_the_new_generation() {
        let gib = 1_u64 << 30;
        let admission = ContextGrowthAdmission {
            committed_capacity_tokens: 16,
            next_capacity_tokens: 24,
            committed_provider_bytes: 20 * gib,
            staged_provider_bytes: 2 * gib,
            growth_scratch_provider_bytes: gib,
            transaction_peak_provider_bytes: 23 * gib,
        };
        let healthy = HostMemory {
            physical_bytes: Some(64 * gib),
            available_bytes: Some(32 * gib),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        };
        // The old 20 GiB is already live and therefore already absent from the
        // fresh available-memory snapshot. Only the new complete generation is
        // charged again at admission.
        let unified = ProviderMemory::Unified {
            recommended_working_set_bytes: None,
            available_working_set_bytes: None,
        };
        admit_live_context_growth(healthy, unified, admission).unwrap();

        let pressured = HostMemory {
            available_bytes: Some(10 * gib),
            ..healthy
        };
        assert!(admit_live_context_growth(pressured, unified, admission).is_err());
        assert!(
            admit_live_context_growth(
                healthy,
                ProviderMemory::Discrete {
                    total_bytes: None,
                    available_bytes: None,
                },
                admission,
            )
            .is_err(),
            "CUDA growth must fail closed until the provider reports live VRAM"
        );
    }

    #[test]
    fn measured_mps_trim_delta_repairs_lagging_mach_availability_only() {
        let gib = 1_u64 << 30;
        let host = HostMemory {
            physical_bytes: Some(64 * gib),
            available_bytes: Some(8 * gib),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        };
        let snapshot = |reserved_bytes, cache_trimmed| NativeProviderMemorySnapshot {
            device: Device::Mps,
            active_bytes: Some(20 * gib),
            reserved_bytes: Some(reserved_bytes),
            recommended_bytes: Some(48 * gib),
            total_bytes: None,
            available_bytes: None,
            cache_trimmed,
        };
        let repaired = account_trimmed_unified_memory(
            host,
            snapshot(31 * gib, false),
            snapshot(26 * gib, true),
        );
        assert_eq!(repaired.available_bytes, Some(13 * gib));

        let unrelated = account_trimmed_unified_memory(
            host,
            NativeProviderMemorySnapshot {
                device: Device::Cuda(0),
                ..snapshot(31 * gib, false)
            },
            NativeProviderMemorySnapshot {
                device: Device::Cuda(0),
                ..snapshot(26 * gib, true)
            },
        );
        assert_eq!(unrelated, host);

        let unknown_host = HostMemory {
            available_bytes: None,
            ..host
        };
        assert_eq!(
            account_trimmed_unified_memory(
                unknown_host,
                snapshot(31 * gib, false),
                snapshot(26 * gib, true),
            )
            .available_bytes,
            None,
        );
    }

    #[test]
    fn verify_budget_charges_the_prefix_record_and_preserves_exact_fallback() {
        let budget = VerifySnapshotBudget {
            bytes_per_kda_generation: 100,
            replay_bytes_per_position: 3,
            replay_fixed_bytes: 5,
            max_positions: 64,
        };
        let ordinary = budget.admission(1).unwrap();
        assert_eq!(ordinary.committed_provider_bytes, 100);
        assert_eq!(ordinary.staged_boundary_provider_bytes, 100);
        assert_eq!(ordinary.transaction_peak_provider_bytes, 200);
        assert_eq!(ordinary.additional_over_decode_reserve_bytes, 0);

        // Seven rows stage one final generation plus a 5 + 7*3 byte record,
        // never a second generation.
        let wide = budget.admission(7).unwrap();
        assert_eq!(wide.staged_boundary_provider_bytes, 126);
        assert_eq!(wide.transaction_peak_provider_bytes, 226);
        assert_eq!(wide.additional_over_decode_reserve_bytes, 26);
        assert!(budget.admission(0).is_err());
        assert!(budget.admission(65).is_err());
    }

    #[test]
    fn slot_verify_reserve_covers_the_widest_configured_verify() {
        let budget = k3_verify_snapshot_budget(69).unwrap();
        let reserved = slot_verify_reserve(budget, 8).unwrap();
        for positions in 1..=9 {
            let admission = budget.admission(positions).unwrap();
            assert_eq!(verify_live_admission_bytes(admission, reserved), 0);
        }
        assert!(verify_live_admission_bytes(budget.admission(10).unwrap(), reserved) > 0);
        assert!(reserved < 256 << 20);
    }

    #[test]
    fn k3_verify_record_is_a_small_fraction_of_a_generation() {
        let budget = k3_verify_snapshot_budget(69).unwrap();
        assert_eq!(budget.bytes_per_kda_generation, 69 * 1_720_320 * 4);
        assert_eq!(budget.replay_bytes_per_position, 69 * 73_824 * 4);
        // An eight-row verify needs about 185 MiB beyond the decode
        // reserve, against 3.1 GiB when every row kept its own state.
        let wide = budget.admission(8).unwrap();
        assert!(wide.additional_over_decode_reserve_bytes < 200 << 20);
    }

    #[test]
    fn pack_directory_presence_is_a_transactional_commitment() {
        let root = TempRoot::new();
        let path = root.0.join("k3-resident-packs-bf16");
        assert_eq!(
            spine_source_intent(&path).unwrap(),
            SpinePlanSource::LooseDeferredFiles
        );
        fs::create_dir(&path).unwrap();
        assert_eq!(
            spine_source_intent(&path).unwrap(),
            SpinePlanSource::AuthenticatedPacks
        );

        let root = TempRoot::new();
        let malformed = root.0.join("k3-resident-packs-bf16");
        File::create(&malformed).unwrap();
        assert_eq!(
            spine_source_intent(&malformed).unwrap(),
            SpinePlanSource::AuthenticatedPacks
        );
        // `compile_spine` will route both present cases only through pack
        // admission, where a file or partial directory fails closed.
    }

    #[test]
    fn readiness_reports_only_the_physically_verified_configuration() {
        let verified = target_readiness(
            Device::Mps,
            SpineRepresentation::OriginalBf16,
            ResolvedExpertBackend::Metal,
            ExpertStorageLayout::RawV1,
        );
        assert_eq!(
            verified,
            NativeTargetReadiness::VerifiedOriginalBf16MpsRawMetal
        );
        assert!(verified.permits_generation());
        let message = verified.to_string();
        assert!(message.contains("physical MPS"));
        assert!(message.contains("original-BF16"));
        assert!(message.contains("raw-v1 Metal"));
        assert!(message.contains("exact 17-token"));
        assert!(message.contains("all 16 routed experts"));

        for unverified in [
            target_readiness(
                Device::Cpu,
                SpineRepresentation::OriginalBf16,
                ResolvedExpertBackend::Cpu,
                ExpertStorageLayout::RawV1,
            ),
            target_readiness(
                Device::Mps,
                SpineRepresentation::QuantizedInt8,
                ResolvedExpertBackend::Metal,
                ExpertStorageLayout::RawV1,
            ),
            target_readiness(
                Device::Mps,
                SpineRepresentation::OriginalBf16,
                ResolvedExpertBackend::Metal,
                ExpertStorageLayout::Scale4V2,
            ),
            target_readiness(
                Device::Cuda(0),
                SpineRepresentation::OriginalBf16,
                ResolvedExpertBackend::Cuda,
                ExpertStorageLayout::RawV1,
            ),
        ] {
            assert_eq!(
                unverified,
                NativeTargetReadiness::StructurallyReadyUnverified
            );
            assert!(unverified.permits_generation());
            assert!(unverified.to_string().contains("not yet recorded"));
        }
    }

    #[test]
    fn prompt_chunks_cover_every_row_once_with_the_provider_bound() {
        let first = next_prompt_chunk(130, 0).unwrap().unwrap();
        let second = next_prompt_chunk(130, first.end).unwrap().unwrap();
        let third = next_prompt_chunk(130, second.end).unwrap().unwrap();
        assert_eq!(first, 0..64);
        assert_eq!(second, 64..128);
        assert_eq!(third, 128..130);
        assert_eq!(next_prompt_chunk(130, third.end).unwrap(), None);
        assert!(next_prompt_chunk(130, 131).is_err());
    }

    #[test]
    fn speculative_budget_always_leaves_room_for_the_target_bonus() {
        assert_eq!(bounded_draft_budget(7, 1, 100), 0);
        assert_eq!(bounded_draft_budget(7, 100, 1), 0);
        assert_eq!(bounded_draft_budget(7, 2, 100), 1);
        assert_eq!(bounded_draft_budget(7, 100, 2), 1);
        assert_eq!(bounded_draft_budget(7, 8, 8), 7);
        assert_eq!(bounded_draft_budget(3, 100, 100), 3);
        assert_eq!(bounded_draft_budget(0, 100, 100), 0);
    }

    #[test]
    fn speculative_capacity_defaults_wide_but_remains_explicitly_bounded() {
        assert_eq!(resolve_speculative_max_drafts(None).unwrap(), 8);
        assert_eq!(resolve_speculative_max_drafts(Some("1")).unwrap(), 1);
        assert_eq!(resolve_speculative_max_drafts(Some(" 7 ")).unwrap(), 7);
        assert!(resolve_speculative_max_drafts(Some("0")).is_err());
        assert!(resolve_speculative_max_drafts(Some("9")).is_err());
        assert!(resolve_speculative_max_drafts(Some("wide")).is_err());
    }

    #[test]
    fn qwen_request_policy_probes_then_widens_and_adapts() {
        let mut policy = QwenRequestPolicy::new(true, 8);
        assert!(policy.active);
        assert!(!policy.qualified);
        assert_eq!(policy.proposal_width(8), 2);

        policy.record_verified(2, 2);
        assert!(policy.qualified);
        assert_eq!(policy.proposal_width(8), 8);

        policy.record_verified(3, 8);
        assert_eq!(policy.proposal_width(8), 6);
        policy.record_verified(6, 6);
        assert_eq!(policy.proposal_width(8), 8);

        policy.record_verified(0, 8);
        assert!(policy.active);
        assert_eq!(policy.proposal_width(8), 2);
        policy.record_verified(0, 2);
        assert!(!policy.active);
        assert_eq!(policy.proposal_width(8), 0);
    }

    #[test]
    fn qwen_request_policy_disables_bad_probes_but_not_confidence_skips() {
        let mut confidence_skip = QwenRequestPolicy::new(true, 8);
        confidence_skip.record_empty(true);
        assert!(confidence_skip.active);
        assert_eq!(confidence_skip.proposal_width(8), 2);

        let mut empty = QwenRequestPolicy::new(true, 8);
        empty.record_empty(false);
        assert!(!empty.active);

        let mut rejected = QwenRequestPolicy::new(true, 8);
        rejected.record_verified(0, 2);
        assert!(!rejected.active);

        let mut narrow = QwenRequestPolicy::new(true, 1);
        narrow.record_verified(1, 1);
        assert!(narrow.active);
        assert!(!narrow.qualified);
        assert_eq!(narrow.proposal_width(8), 1);

        let mut impossible = QwenRequestPolicy::new(true, 8);
        impossible.record_verified(3, 2);
        assert!(!impossible.active);
    }

    #[test]
    fn draft_rows_after_eos_are_never_sent_to_the_target() {
        let mut tokens = vec![11, 12, 99, 13, 14];
        truncate_after_first(&mut tokens, 99);
        assert_eq!(tokens, [11, 12, 99]);

        let mut absent = vec![1, 2, 3];
        truncate_after_first(&mut absent, 99);
        assert_eq!(absent, [1, 2, 3]);
    }

    #[test]
    fn native_server_splits_k3_reasoning_without_exposing_control_tokens() {
        let raw = format!("private{THINK_RESPONSE_MARKER}public{ASSISTANT_CLOSE_MARKER}");
        assert_eq!(
            split_chat_output(raw, FinishReason::Stop),
            (Some("private".into()), "public".into())
        );
        assert_eq!(
            split_chat_output("unfinished thought".into(), FinishReason::Length),
            (Some("unfinished thought".into()), String::new())
        );
        assert_eq!(
            split_chat_output("plain response".into(), FinishReason::Stop),
            (None, "plain response".into())
        );
    }

    #[test]
    fn native_server_usage_excludes_terminal_eos_like_the_established_server() {
        let eos = NativeGeneration {
            token_ids: vec![41, 42, 99].into_boxed_slice(),
            stop: StopReason::Eos,
            wrote_text: true,
        };
        let (finish, usage) = target_generation_metadata(&[1, 2, 3, 4], &eos).unwrap();
        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(usage.prompt_tokens, 4);
        assert_eq!(usage.completion_tokens, 2);

        let capped = NativeGeneration {
            token_ids: vec![41, 42, 43].into_boxed_slice(),
            stop: StopReason::MaxNew,
            wrote_text: true,
        };
        let (finish, usage) = target_generation_metadata(&[1, 2, 3, 4], &capped).unwrap();
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(usage.completion_tokens, 3);
    }

    #[test]
    fn length_capped_chat_never_relabels_unopened_response_as_public_content() {
        let partial_control = &THINK_RESPONSE_MARKER[..THINK_RESPONSE_MARKER.len() - 3];
        let raw = format!("unfinished private reasoning{partial_control}");
        for split in 0..=raw.len() {
            let mut parser = ChatStreamParser::default();
            let mut deltas = parser.push(&raw[..split]);
            deltas.extend(parser.push(&raw[split..]));
            deltas.extend(parser.finish(FinishReason::Length));
            assert_eq!(
                collect_chat_deltas(deltas),
                ("unfinished private reasoning".into(), String::new()),
                "split at byte {split}"
            );
        }

        assert_eq!(
            split_chat_output(raw, FinishReason::Length),
            (Some("unfinished private reasoning".into()), String::new())
        );
    }

    #[test]
    fn native_cli_chat_streams_only_public_content_across_marker_splits() {
        let raw = format!("private{THINK_RESPONSE_MARKER}public{ASSISTANT_CLOSE_MARKER}");
        for split in 0..=raw.len() {
            let mut bytes = Vec::new();
            {
                let mut writer = CliOutputWriter::new(&mut bytes, true);
                writer.write_all(&raw.as_bytes()[..split]).unwrap();
                writer.write_all(&raw.as_bytes()[split..]).unwrap();
                writer.finish(FinishReason::Stop).unwrap();
                assert!(writer.wrote_public_text());
                writer.write_terminal_newline().unwrap();
            }
            assert_eq!(String::from_utf8(bytes).unwrap(), "public\n");
        }

        let mut raw_bytes = Vec::new();
        let mut raw_writer = CliOutputWriter::new(&mut raw_bytes, false);
        raw_writer.write_all(b"ordinary completion").unwrap();
        raw_writer.finish(FinishReason::Stop).unwrap();
        assert_eq!(raw_bytes, b"ordinary completion");
    }

    fn collect_chat_deltas(deltas: impl IntoIterator<Item = ChatTextDelta>) -> (String, String) {
        let mut reasoning = String::new();
        let mut content = String::new();
        for delta in deltas {
            match delta {
                ChatTextDelta::Reasoning(text) => reasoning.push_str(&text),
                ChatTextDelta::Content(text) => content.push_str(&text),
            }
        }
        (reasoning, content)
    }

    #[test]
    fn chat_stream_parser_preserves_channels_at_every_marker_split() {
        let raw = format!("private{THINK_RESPONSE_MARKER}public{ASSISTANT_CLOSE_MARKER}");
        for split in 0..=raw.len() {
            let mut parser = ChatStreamParser::default();
            let mut deltas = parser.push(&raw[..split]);
            deltas.extend(parser.push(&raw[split..]));
            deltas.extend(parser.finish(FinishReason::Stop));
            assert_eq!(
                collect_chat_deltas(deltas),
                ("private".into(), "public".into()),
                "split at byte {split}"
            );
        }
    }

    #[test]
    fn chat_stream_parser_does_not_guess_an_unmarked_channel() {
        let mut parser = ChatStreamParser::default();
        assert!(parser.push("unknown no-marker completion").is_empty());
        assert_eq!(
            parser.finish(FinishReason::Stop),
            [ChatTextDelta::Content(
                "unknown no-marker completion".into()
            )]
        );
    }

    #[test]
    fn chat_stream_parser_rolls_a_bounded_closing_marker_candidate() {
        let mut parser = ChatStreamParser::default();
        let start = format!("thought{THINK_RESPONSE_MARKER}");
        assert_eq!(
            parser.push(&start),
            [ChatTextDelta::Reasoning("thought".into())]
        );
        for byte in ASSISTANT_CLOSE_MARKER.as_bytes() {
            parser.push(std::str::from_utf8(std::slice::from_ref(byte)).unwrap());
            match &parser.phase {
                ChatParsePhase::Response(rolling) => {
                    assert!(rolling.len() <= ASSISTANT_CLOSE_MARKER.len())
                }
                phase => panic!("unexpected parser phase: {phase:?}"),
            }
        }
        assert!(parser.finish(FinishReason::Stop).is_empty());
    }

    #[test]
    fn chat_stream_parser_preserves_a_close_marker_when_k3_continues() {
        let raw = format!(
            "thought{THINK_RESPONSE_MARKER}before{ASSISTANT_CLOSE_MARKER}after{ASSISTANT_CLOSE_MARKER}"
        );
        for split in 0..=raw.len() {
            let mut parser = ChatStreamParser::default();
            let mut deltas = parser.push(&raw[..split]);
            deltas.extend(parser.push(&raw[split..]));
            deltas.extend(parser.finish(FinishReason::Stop));
            assert_eq!(
                collect_chat_deltas(deltas),
                (
                    "thought".into(),
                    format!("before{ASSISTANT_CLOSE_MARKER}after")
                ),
                "split at byte {split}"
            );
        }
    }

    #[test]
    fn chat_stream_parser_suppresses_unexpected_controls_across_writes() {
        let raw = format!(
            "thought{THINK_RESPONSE_MARKER}answer{THINK_OPEN_MARKER}continued{ASSISTANT_CLOSE_MARKER}"
        );
        let mut parser = ChatStreamParser::default();
        let mut deltas = Vec::new();
        for byte in raw.as_bytes() {
            deltas.extend(parser.push(std::str::from_utf8(std::slice::from_ref(byte)).unwrap()));
        }
        deltas.extend(parser.finish(FinishReason::Stop));
        assert_eq!(
            collect_chat_deltas(deltas),
            ("thought".into(), "answercontinued".into())
        );

        let mut truncated = ChatStreamParser::default();
        assert!(truncated.push("plain<|open|>respon").is_empty());
        assert_eq!(
            truncated.finish(FinishReason::Stop),
            [ChatTextDelta::Content("plain".into())]
        );
    }

    #[test]
    fn stream_boundary_recovers_only_abortable_publication_states() {
        let mut successful = NativeStreamBoundary::AwaitingPublication;
        assert_eq!(
            successful.resolve(StreamPublication::Complete),
            NativeStreamResolution::Preserve
        );
        assert_eq!(successful, NativeStreamBoundary::None);

        let mut unpublished = NativeStreamBoundary::AwaitingPublication;
        assert_eq!(
            unpublished.resolve(StreamPublication::Aborted),
            NativeStreamResolution::Discard
        );

        let mut failed = NativeStreamBoundary::PublicationFailed;
        assert_eq!(
            failed.resolve(StreamPublication::Aborted),
            NativeStreamResolution::Discard
        );

        let mut invalid_complete = NativeStreamBoundary::PublicationFailed;
        assert_eq!(
            invalid_complete.resolve(StreamPublication::Complete),
            NativeStreamResolution::Poison
        );

        let mut unrelated = NativeStreamBoundary::None;
        assert_eq!(
            unrelated.resolve(StreamPublication::Aborted),
            NativeStreamResolution::None
        );
    }

    #[test]
    fn target_reuse_plan_requires_real_branching_and_exact_one_token_lag() {
        let identity = test_target_reuse_identity(17);
        let boundary = PublishedTargetBoundary {
            identity,
            logical_tokens: vec![10, 11, 12, 13].into_boxed_slice(),
            committed_positions: 3,
            cache_generation: 23,
            pending_token: 13,
            boundary_id: BoundaryId::numeric(1),
        };
        let continued = [10, 11, 12, 13, 20, 21];

        assert_eq!(
            plan_target_reuse(
                TargetStateTransactionCapability::ResetOnly,
                Some(&boundary),
                identity,
                &continued,
            ),
            TargetReusePlan::Reset(TargetReuseInvalidation::UnsupportedCapability),
            "matching metadata alone must never be reported as provider-backed KV reuse"
        );
        assert_eq!(
            plan_target_reuse(
                TargetStateTransactionCapability::RequestBranchV1,
                Some(&boundary),
                identity,
                &continued,
            ),
            TargetReusePlan::Reuse {
                expected_committed_positions: 3,
                expected_cache_generation: 23,
                replay: 3..6,
            },
            "replay starts with the K3-authored token still pending at the provider"
        );

        let corrupt = PublishedTargetBoundary {
            pending_token: 99,
            ..boundary.clone()
        };
        assert_eq!(
            plan_target_reuse(
                TargetStateTransactionCapability::RequestBranchV1,
                Some(&corrupt),
                identity,
                &continued,
            ),
            TargetReusePlan::Reset(TargetReuseInvalidation::InvalidPublishedBoundary)
        );
    }

    #[test]
    fn target_reuse_plan_invalidates_changed_truncated_and_diverged_prompts() {
        let identity = test_target_reuse_identity(31);
        let boundary = PublishedTargetBoundary {
            identity,
            logical_tokens: vec![1, 2, 3, 4].into_boxed_slice(),
            committed_positions: 3,
            cache_generation: 37,
            pending_token: 4,
            boundary_id: BoundaryId::numeric(2),
        };
        let capability = TargetStateTransactionCapability::RequestBranchV1;

        assert_eq!(
            plan_target_reuse(capability, None, identity, &[1, 2, 3, 4]),
            TargetReusePlan::Reset(TargetReuseInvalidation::NoPublishedBoundary)
        );
        assert_eq!(
            plan_target_reuse(
                capability,
                Some(&boundary),
                test_target_reuse_identity(99),
                &[1, 2, 3, 4],
            ),
            TargetReusePlan::Reset(TargetReuseInvalidation::ModelOrConfigChanged)
        );
        assert_eq!(
            plan_target_reuse(capability, Some(&boundary), identity, &[1, 2, 3]),
            TargetReusePlan::Reset(TargetReuseInvalidation::PromptTruncated)
        );
        assert_eq!(
            plan_target_reuse(capability, Some(&boundary), identity, &[1, 2, 3, 4]),
            TargetReusePlan::Reset(TargetReuseInvalidation::PromptTruncated),
            "reuse is an exclusive strict extension, never an equal-prompt alias"
        );
        assert_eq!(
            plan_target_reuse(capability, Some(&boundary), identity, &[1, 2, 9, 4, 5]),
            TargetReusePlan::Reset(TargetReuseInvalidation::PromptDiverged)
        );
    }

    #[test]
    fn target_reuse_slot_is_consumed_on_divergence() {
        let identity = test_target_reuse_identity(41);
        let mut slot = Some(PublishedTargetBoundary {
            identity,
            logical_tokens: vec![7, 8, 9].into_boxed_slice(),
            committed_positions: 2,
            cache_generation: 5,
            pending_token: 9,
            boundary_id: BoundaryId::numeric(3),
        });
        let (evicted, plan) = consume_target_reuse_slot(
            &mut slot,
            TargetStateTransactionCapability::RequestBranchV1,
            identity,
            &[7, 8, 10, 11],
        );
        assert!(evicted.is_some());
        assert!(slot.is_none(), "one-slot reuse must evict on every miss");
        assert_eq!(
            plan,
            TargetReusePlan::Reset(TargetReuseInvalidation::PromptDiverged)
        );
    }

    #[test]
    fn target_state_branch_serializes_and_resolves_publication_transactionally() {
        let mut provider = FakeTransactionalTargetState::new(8);
        let aborted = provider.begin_target_state_branch(8).unwrap();
        assert!(provider.begin_target_state_branch(8).is_err());
        provider.advance_private_branch(13);
        provider.discard_target_state_branch(aborted).unwrap();
        assert_eq!(provider.committed_positions, 8);
        assert!(provider.active_branch.is_none());

        assert!(provider.begin_target_state_branch(7).is_err());
        let complete = provider.begin_target_state_branch(8).unwrap();
        provider.advance_private_branch(15);
        provider.publish_target_state_branch(complete).unwrap();
        assert_eq!(provider.committed_positions, 15);
        assert!(provider.active_branch.is_none());
    }

    #[test]
    fn stream_writer_retains_client_failure_as_publication_error() {
        struct BrokenClient;

        impl TargetDeltaSink for BrokenClient {
            fn publish_target_delta(&mut self, _delta: TargetDelta) -> io::Result<()> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "client left"))
            }
        }

        let mut sink = BrokenClient;
        let mut writer = TargetStreamWriter::new(&mut sink, false);
        let error = writer.write_all(b"K3-certified").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        let publication = writer.take_publication_error().unwrap();
        assert_eq!(publication.kind(), io::ErrorKind::BrokenPipe);
        assert_eq!(publication.to_string(), "client left");
    }

    #[test]
    fn expert_tiles_merge_overlapping_routes_into_one_canonical_union() {
        let mut first_route = [0_u16; K3_EXPERT_TOP_K];
        for (index, expert) in first_route.iter_mut().enumerate() {
            *expert = (K3_EXPERT_TOP_K - 1 - index) as u16;
        }
        let mut second_route = first_route;
        second_route[0] = 16;
        let routes = [first_route, first_route, second_route, second_route];

        let first = next_expert_tile(routes.len(), 0, K3_EXPERT_BASE_UNION_MAX, |row| {
            Ok(routes[row])
        })
        .unwrap();
        assert_eq!(first.first_row, 0);
        assert_eq!(first.row_count, 4);
        assert_eq!(first.expert_count(), 17);
        assert_eq!(first.expert_ids(), &(0_u16..=16).collect::<Vec<_>>());
    }

    #[test]
    fn expert_tiles_are_bounded_and_reject_any_quality_reducing_route() {
        let route = core::array::from_fn(|index| index as u16);
        let routes = [route; 20];
        assert_eq!(
            next_expert_tile(routes.len(), 0, K3_EXPERT_BASE_UNION_MAX, |row| Ok(
                routes[row]
            ),)
            .unwrap()
            .row_count,
            TARGET_EXPERT_TILE_MAX_ROWS
        );
        assert_eq!(
            next_expert_tile(
                routes.len(),
                TARGET_EXPERT_TILE_MAX_ROWS,
                K3_EXPERT_BASE_UNION_MAX,
                |row| Ok(routes[row]),
            )
            .unwrap()
            .row_count,
            4
        );

        let mut duplicate = route;
        duplicate[15] = duplicate[14];
        assert!(canonical_route(duplicate).is_err());
        let mut out_of_range = route;
        out_of_range[15] = K3_EXPERT_COUNT;
        assert!(canonical_route(out_of_range).is_err());
    }

    #[test]
    fn expert_tiles_admit_exactly_sixty_four_experts_and_split_before_sixty_five() {
        let routes: [[u16; K3_EXPERT_TOP_K]; 5] = core::array::from_fn(|row| {
            core::array::from_fn(|slot| (row * K3_EXPERT_TOP_K + slot) as u16)
        });
        let first = next_expert_tile(routes.len(), 0, K3_EXPERT_BASE_UNION_MAX, |row| {
            Ok(routes[row])
        })
        .unwrap();
        assert_eq!(first.first_row, 0);
        assert_eq!(first.row_count, 4);
        assert_eq!(first.expert_count(), K3_EXPERT_BASE_UNION_MAX);
        assert_eq!(
            first.expert_ids(),
            &(0_u16..K3_EXPERT_BASE_UNION_MAX as u16).collect::<Vec<_>>()
        );

        let second = next_expert_tile(
            routes.len(),
            first.row_count,
            K3_EXPERT_BASE_UNION_MAX,
            |row| Ok(routes[row]),
        )
        .unwrap();
        assert_eq!(second.first_row, 4);
        assert_eq!(second.row_count, 1);
        assert_eq!(second.expert_count(), K3_EXPERT_TOP_K);
        assert_eq!(second.expert_ids(), &(64_u16..80).collect::<Vec<_>>());
        assert_eq!(second.first_row + second.row_count, routes.len());
    }

    #[test]
    fn expert_prefetch_keeps_two_bounded_generations_and_future_tiles_do_not_cancel_it() {
        assert_eq!(EXPERT_PREFETCH_MAX_EXPERTS, 32);
        assert_eq!(EXPERT_PREFETCH_GENERATIONS, 2);
        assert_eq!(EXPERT_PREFETCH_LIVE_SLOTS, 64);
        assert_eq!(EXPERT_PREFETCH_ARENA_SLOTS, 65);

        let mut pending = Some(ExpertPrefetchSet {
            target_layer: 18,
            tickets: Vec::new(),
        });
        assert!(take_due_expert_prefetch(&mut pending, 17).is_none());
        assert_eq!(
            pending.as_ref().map(|prefetch| prefetch.target_layer),
            Some(18)
        );

        let due = take_due_expert_prefetch(&mut pending, 18)
            .expect("the generation becomes due at its exact target layer");
        assert_eq!(due.target_layer, 18);
        assert!(pending.is_none());

        let mut stale = Some(ExpertPrefetchSet {
            target_layer: 18,
            tickets: Vec::new(),
        });
        assert!(take_due_expert_prefetch(&mut stale, 19).is_some());
        assert!(stale.is_none());
    }

    #[test]
    fn expert_prefetch_losers_cancel_before_demand_submit_and_drain_afterward() {
        let events = std::cell::RefCell::new(Vec::new());
        let submitted = cancel_submit_drain(
            vec![3_u16, 7_u16],
            |ticket| events.borrow_mut().push(format!("cancel-{ticket}")),
            || {
                events.borrow_mut().push("submit-demand".to_string());
                Ok(41_u32)
            },
            |ticket| events.borrow_mut().push(format!("drain-{ticket}")),
        )
        .unwrap();

        assert_eq!(submitted, 41);
        assert_eq!(
            events.into_inner(),
            [
                "cancel-3",
                "cancel-7",
                "submit-demand",
                "drain-3",
                "drain-7",
            ]
        );
    }

    #[test]
    fn expert_prefetch_losers_are_drained_when_demand_submission_fails() {
        let events = std::cell::RefCell::new(Vec::new());
        let error = cancel_submit_drain(
            vec![5_u16],
            |ticket| events.borrow_mut().push(format!("cancel-{ticket}")),
            || {
                events.borrow_mut().push("submit-demand".to_string());
                Err::<(), _>(DeltafinError::new("synthetic demand rejection"))
            },
            |ticket| events.borrow_mut().push(format!("drain-{ticket}")),
        )
        .unwrap_err();

        assert!(error.to_string().contains("synthetic demand rejection"));
        assert_eq!(
            events.into_inner(),
            ["cancel-5", "submit-demand", "drain-5"]
        );
    }

    #[test]
    fn complete_verifier_union_keeps_all_nine_rows_and_legacy_fallback_is_exact() {
        assert_eq!(full_commit_union_upper_bound(4), None);
        assert_eq!(full_commit_union_upper_bound(5), Some(80));
        assert_eq!(full_commit_union_upper_bound(9), Some(144));
        assert_eq!(full_commit_union_upper_bound(10), None);
        let routes: [[u16; K3_EXPERT_TOP_K]; 9] = core::array::from_fn(|row| {
            core::array::from_fn(|slot| (row * K3_EXPERT_TOP_K + slot) as u16)
        });
        let complete =
            next_expert_tile(routes.len(), 0, K3_EXPERT_UNION_MAX, |row| Ok(routes[row])).unwrap();
        assert_eq!(complete.first_row, 0);
        assert_eq!(complete.row_count, 9);
        assert_eq!(complete.expert_count(), 144);
        assert_eq!(complete.expert_ids(), &(0_u16..144_u16).collect::<Vec<_>>());

        let mut cursor = 0;
        let mut legacy_rows = Vec::new();
        while cursor < routes.len() {
            let tile = next_expert_tile(routes.len(), cursor, K3_EXPERT_BASE_UNION_MAX, |row| {
                Ok(routes[row])
            })
            .unwrap();
            for row in cursor..cursor + tile.row_count {
                assert!(
                    routes[row]
                        .iter()
                        .all(|expert| tile.expert_ids().binary_search(expert).is_ok())
                );
            }
            legacy_rows.push(tile.row_count);
            cursor += tile.row_count;
        }
        assert_eq!(legacy_rows, [4, 4, 1]);
    }

    #[test]
    fn complete_union_is_forward_safe_through_sixteen_rows_and_t1_is_unchanged() {
        let routes: [[u16; K3_EXPERT_TOP_K]; TARGET_EXPERT_TILE_MAX_ROWS] =
            core::array::from_fn(|row| {
                core::array::from_fn(|slot| (row * K3_EXPERT_TOP_K + slot) as u16)
            });
        let complete =
            next_expert_tile(routes.len(), 0, K3_EXPERT_UNION_MAX, |row| Ok(routes[row])).unwrap();
        assert_eq!(complete.row_count, TARGET_EXPERT_TILE_MAX_ROWS);
        assert_eq!(complete.expert_count(), K3_EXPERT_UNION_MAX);

        let t1 = next_expert_tile(1, 0, K3_EXPERT_BASE_UNION_MAX, |_| Ok(routes[0])).unwrap();
        assert_eq!(t1.row_count, 1);
        assert_eq!(t1.expert_ids(), &routes[0]);
    }

    #[test]
    fn complete_verifier_union_is_cpu_metal_only_and_memory_fail_soft() {
        assert!(may_stage_complete_verifier_union(
            true,
            TargetSequenceMode::Verify,
            TargetExpertBackend::Cpu,
            9,
        ));
        assert!(may_stage_complete_verifier_union(
            true,
            TargetSequenceMode::Verify,
            TargetExpertBackend::Metal,
            9,
        ));
        assert!(!may_stage_complete_verifier_union(
            false,
            TargetSequenceMode::Verify,
            TargetExpertBackend::Metal,
            9,
        ));
        assert!(!may_stage_complete_verifier_union(
            true,
            TargetSequenceMode::Prefill,
            TargetExpertBackend::Metal,
            9,
        ));
        assert!(!may_stage_complete_verifier_union(
            true,
            TargetSequenceMode::Verify,
            TargetExpertBackend::Auto,
            9,
        ));
        assert!(!may_stage_complete_verifier_union(
            true,
            TargetSequenceMode::Verify,
            TargetExpertBackend::Cuda,
            9,
        ));
        assert!(!may_stage_complete_verifier_union(
            true,
            TargetSequenceMode::Verify,
            TargetExpertBackend::Metal,
            TARGET_EXPERT_TILE_MAX_ROWS + 1,
        ));

        let gib = 1_u64 << 30;
        let healthy = HostMemory {
            physical_bytes: Some(64 * gib),
            available_bytes: Some(20 * gib),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        };
        let pressured = HostMemory {
            available_bytes: Some(12 * gib),
            ..healthy
        };
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        assert!(admit_dynamic_complete_verifier_union(
            healthy,
            &reader,
            ExpertStorageLayout::RawV1,
            144,
        ));
        assert!(admit_dynamic_complete_verifier_union(
            healthy,
            &reader,
            ExpertStorageLayout::Scale4V2,
            144,
        ));
        assert!(!admit_dynamic_complete_verifier_union(
            pressured,
            &reader,
            ExpertStorageLayout::RawV1,
            144,
        ));
        assert!(!admit_dynamic_complete_verifier_union(
            HostMemory::unknown(),
            &reader,
            ExpertStorageLayout::Scale4V2,
            144,
        ));
        assert!(admit_dynamic_complete_verifier_union(
            HostMemory::unknown(),
            &reader,
            ExpertStorageLayout::RawV1,
            K3_EXPERT_BASE_UNION_MAX,
        ));
        assert!(!admit_dynamic_complete_verifier_union(
            healthy,
            &reader,
            ExpertStorageLayout::RawV1,
            K3_EXPERT_UNION_MAX + 1,
        ));
    }

    #[test]
    fn compact_wide_union_is_a_cpu_metal_startup_reservation_only() {
        assert_eq!(
            startup_complete_expert_union_capacity(
                true,
                TargetExpertBackend::Cpu,
                ExpertStorageLayout::Scale4V2,
            ),
            Some(FULL_COMMIT_EXPERT_UNION_MAX),
        );
        assert_eq!(
            startup_complete_expert_union_capacity(
                true,
                TargetExpertBackend::Metal,
                ExpertStorageLayout::Scale4V2,
            ),
            Some(FULL_COMMIT_EXPERT_UNION_MAX),
        );
        assert_eq!(
            startup_complete_expert_union_capacity(
                false,
                TargetExpertBackend::Metal,
                ExpertStorageLayout::Scale4V2,
            ),
            None,
        );
        assert_eq!(
            startup_complete_expert_union_capacity(
                true,
                TargetExpertBackend::Cuda,
                ExpertStorageLayout::Scale4V2,
            ),
            None,
        );
        assert_eq!(
            startup_complete_expert_union_capacity(
                true,
                TargetExpertBackend::Metal,
                ExpertStorageLayout::RawV1,
            ),
            None,
        );
    }

    #[test]
    fn compact_wide_union_hot_prepare_never_grows_the_reader() {
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        let lengths = BufferLengths::new(
            0,
            0,
            ExpertStorageLayout::Scale4V2
                .expert_span_bytes()
                .checked_mul(FULL_COMMIT_EXPERT_UNION_MAX)
                .unwrap(),
        );
        let before = reader.replacement_admission_bytes(lengths).unwrap();
        assert_eq!(
            prepare_complete_expert_union(
                true,
                Some(FULL_COMMIT_EXPERT_UNION_MAX),
                TargetSequenceMode::Verify,
                true,
                MAX_EXACT_DRAFTS + 1,
                TargetExpertBackend::Metal,
                &reader,
                ExpertStorageLayout::Scale4V2,
            ),
            CompleteExpertUnion::Reserved(FULL_COMMIT_EXPERT_UNION_MAX),
        );
        assert_eq!(reader.replacement_admission_bytes(lengths).unwrap(), before);
        assert_eq!(
            prepare_complete_expert_union(
                true,
                None,
                TargetSequenceMode::Verify,
                true,
                MAX_EXACT_DRAFTS + 1,
                TargetExpertBackend::Metal,
                &reader,
                ExpertStorageLayout::Scale4V2,
            ),
            CompleteExpertUnion::Disabled,
        );
        assert_eq!(reader.replacement_admission_bytes(lengths).unwrap(), before);
    }

    #[test]
    fn complete_verifier_union_defaults_on_with_an_explicit_escape_hatch() {
        assert!(resolve_complete_expert_union(None).unwrap());
        assert!(!resolve_complete_expert_union(Some("0")).unwrap());
        assert!(resolve_complete_expert_union(Some("1")).unwrap());
        assert!(resolve_complete_expert_union(Some("yes")).is_err());
        assert!(resolve_complete_expert_union(Some("2")).is_err());
    }

    #[test]
    fn canonical_expert_span_borrows_the_exact_expert_major_storage() {
        let ids = [3_u16, 8, 21];
        let slab = [
            0x30_u8, 0x31, 0x32, 0x33, 0x80, 0x81, 0x82, 0x83, 0xa0, 0xa1, 0xa2, 0xa3,
        ];
        let selected = canonical_expert_span(&ids, &slab, 4, 8).unwrap();
        assert_eq!(selected, &[0x80, 0x81, 0x82, 0x83]);
        assert_eq!(selected.as_ptr(), slab[4..].as_ptr());
        assert!(canonical_expert_span(&ids, &slab, 4, 7).is_err());
        assert!(canonical_expert_span(&[8, 3, 21], &slab, 4, 8).is_err());
        assert!(canonical_expert_span(&ids, &slab[..11], 4, 8).is_err());
        assert!(canonical_expert_span(&ids, &slab, 0, 8).is_err());
    }

    #[test]
    fn target_global_groups_bind_in_the_compiled_roster_order() {
        assert_eq!(target_global_group(1).unwrap(), TargetGlobalGroup::Tail);
        assert_eq!(
            target_global_group(2).unwrap(),
            TargetGlobalGroup::LanguageModelHead
        );
        assert!(target_global_group(0).is_err());
        assert!(target_global_group(3).is_err());
    }

    #[test]
    fn worker_width_is_nonzero_and_bounded() {
        assert!((1..=SPINE_READER_LIMIT).contains(&bounded_worker_count(SPINE_READER_LIMIT)));
        assert_eq!(bounded_worker_count(1), 1);
    }

    #[test]
    fn spine_reader_replays_only_the_complete_physically_qualified_tuple() {
        let gib = 1_u64 << 30;
        assert_eq!(
            resolve_spine_reader_workers(
                None,
                true,
                true,
                Some(64 * gib),
                10,
                Some(55_662_788_608),
                Some(41_747_087_360),
            ),
            6,
        );
        // Each memory capability retains the Python policy's one-GiB slop.
        assert_eq!(
            resolve_spine_reader_workers(
                None,
                true,
                true,
                Some(64 * gib - gib),
                10,
                Some(55_662_788_608 + gib),
                Some(41_747_087_360 - gib),
            ),
            6,
        );
    }

    #[test]
    fn spine_reader_does_not_copy_the_measured_result_to_other_hosts() {
        let gib = 1_u64 << 30;
        let exact = (
            Some(64 * gib),
            10,
            Some(55_662_788_608),
            Some(41_747_087_360),
        );
        for (is_macos, streaming, physical, cpus, recommended, max_buffer) in [
            (false, true, exact.0, exact.1, exact.2, exact.3),
            (true, false, exact.0, exact.1, exact.2, exact.3),
            (true, true, Some(32 * gib), exact.1, exact.2, exact.3),
            (true, true, exact.0, 12, exact.2, exact.3),
            (true, true, exact.0, exact.1, None, exact.3),
            (true, true, exact.0, exact.1, exact.2, None),
            (
                true,
                true,
                exact.0,
                exact.1,
                Some(QUALIFIED_RECOMMENDED_BYTES + 2 * gib),
                exact.3,
            ),
            (
                true,
                true,
                exact.0,
                exact.1,
                exact.2,
                Some(QUALIFIED_MAX_BUFFER_BYTES + 2 * gib),
            ),
        ] {
            assert_eq!(
                resolve_spine_reader_workers(
                    None,
                    is_macos,
                    streaming,
                    physical,
                    cpus,
                    recommended,
                    max_buffer,
                ),
                4,
            );
        }
        assert_eq!(
            resolve_spine_reader_workers(None, false, true, None, 0, None, None),
            1,
        );
    }

    #[test]
    fn spine_reader_override_is_explicit_and_hard_bounded() {
        assert_eq!(crate::config::parse_spine_read_threads("1").unwrap(), 1);
        assert_eq!(crate::config::parse_spine_read_threads("16").unwrap(), 16);
        for rejected in ["", "0", "17", "-1", "six"] {
            assert!(crate::config::parse_spine_read_threads(rejected).is_err());
        }
        assert_eq!(
            resolve_spine_reader_workers(Some(12), false, false, None, 1, None, None),
            12,
        );
    }

    #[test]
    fn expert_reader_preserves_portable_default_and_bounded_override() {
        assert!((1..=EXPERT_READER_LIMIT).contains(&configured_expert_reader_workers(None, 1)));
        let single = configured_expert_reader_workers(None, 1);
        assert_eq!(
            configured_expert_reader_workers(None, 3),
            (single * 3).min(crate::config::MAX_EXPERT_READ_THREADS)
        );
        assert_eq!(configured_expert_reader_workers(Some(8), 1), 8);
        assert_eq!(configured_expert_reader_workers(Some(2), 4), 2, "explicit wins");
        assert_eq!(
            configured_expert_reader_workers(Some(usize::MAX), 1),
            crate::config::MAX_EXPERT_READ_THREADS
        );
    }

    #[test]
    fn loose_descriptor_cache_auto_replays_only_the_qualified_streaming_tuple() {
        assert!(resolve_loose_spine_fd_cache(None, true, true, true));
        assert!(!resolve_loose_spine_fd_cache(None, false, true, true));
        assert!(!resolve_loose_spine_fd_cache(None, true, false, true));
        assert!(!resolve_loose_spine_fd_cache(None, true, true, false));
        assert!(resolve_loose_spine_fd_cache(Some(true), true, false, false));
        assert!(!resolve_loose_spine_fd_cache(Some(false), true, true, true));
        assert!(!resolve_loose_spine_fd_cache(Some(true), false, true, true));
    }

    #[test]
    fn non_mps_devices_never_probe_a_metal_buffer_limit() {
        assert_eq!(metal_max_buffer_length(Device::Cpu), None);
        assert_eq!(metal_max_buffer_length(Device::Cuda(0)), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn selected_mps_host_exposes_a_bounded_metal_buffer_limit() {
        let bytes = metal_max_buffer_length(Device::Mps)
            .expect("an available system Metal device must report maxBufferLength");
        assert!(bytes >= 256 * (1_u64 << 20));
        assert!(bytes <= 1_u64 << 50);
    }

    #[test]
    fn qwen_policy_is_raw_only_and_keeps_dspark_as_chat_default() {
        assert!(qwen_allowed_for_request(false));
        assert!(!qwen_allowed_for_request(true));
        let missing = Path::new("/definitely/not/a/model/root");
        let verify = VerifySnapshotBudget {
            bytes_per_kda_generation: 100,
            replay_bytes_per_position: 3,
            replay_fixed_bytes: 5,
            max_positions: 64,
        };
        assert_eq!(
            discover_qwen_plan(QwenRequest::Off, missing, Device::Mps, 4_389, verify, 8).state,
            QwenRuntimeState::Off
        );
        assert_eq!(
            discover_qwen_plan(QwenRequest::Auto, missing, Device::Cpu, 4_389, verify, 8).state,
            QwenRuntimeState::IneligibleDevice
        );
        assert_eq!(
            discover_qwen_plan(QwenRequest::Auto, missing, Device::Mps, 4_389, verify, 8).state,
            QwenRuntimeState::NotInstalled
        );
    }

    #[test]
    fn dspark_auto_never_reserves_an_unusable_raw_direct_runtime() {
        let directory = Path::new(".");
        let mut config = RuntimeConfig::resolve(RunArgs::default(), |_| None).unwrap();
        assert_eq!(config.surface, RuntimeSurface::DirectRun);
        assert!(!config.chat);
        assert!(!dspark_eligible(&config, directory, Device::Mps));

        config.chat = true;
        assert!(dspark_eligible(&config, directory, Device::Mps));
        config.chat = false;
        config.surface = RuntimeSurface::Server;
        assert!(dspark_eligible(&config, directory, Device::Mps));
        assert!(!dspark_eligible(&config, directory, Device::Cpu));

        config.surface = RuntimeSurface::DirectRun;
        config.dspark = DSparkRequest::On;
        assert!(dspark_eligible(
            &config,
            Path::new("/definitely/not/a/dspark/checkpoint"),
            Device::Cpu,
        ));
    }

    #[test]
    fn eagle3_takes_the_slot_only_where_the_slot_is_wanted() {
        let root = std::env::temp_dir().join(format!("deltafin-eagle3-slot-{}", std::process::id()));
        std::fs::create_dir_all(root.join(eagle3_provider::DIRECTORY)).unwrap();
        let missing = Path::new("/definitely/not/a/model/root");
        let mut config = RuntimeConfig::resolve(RunArgs::default(), |_| None).unwrap();
        assert_eq!(config.eagle3, Eagle3Request::Auto);
        // A raw direct run never uses the slot, so nothing is loaded.
        assert!(!eagle3_selected(&config, &root, Device::Mps));

        config.chat = true;
        assert!(eagle3_selected(&config, &root, Device::Mps));
        assert!(!eagle3_selected(&config, &root, Device::Cpu));
        assert!(!eagle3_selected(&config, missing, Device::Mps));
        assert_eq!(
            draft_slot_provider_reserve(&config, &root, Device::Mps, 8_192).unwrap(),
            eagle3_provider::provider_reserve(8_192).unwrap()
        );
        // Without EAGLE the slot falls back to DSpark's own eligibility.
        assert_eq!(
            draft_slot_provider_reserve(&config, missing, Device::Mps, 8_192).unwrap(),
            0
        );

        config.eagle3 = Eagle3Request::Off;
        assert!(!eagle3_selected(&config, &root, Device::Mps));
        config.eagle3 = Eagle3Request::On;
        assert!(eagle3_selected(&config, missing, Device::Mps));
        config.dspark = DSparkRequest::Off;
        assert!(!eagle3_selected(&config, &root, Device::Mps));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn qwen_is_planned_only_where_a_request_can_reach_it() {
        let mut config = RuntimeConfig::resolve(RunArgs::default(), |_| None).unwrap();
        assert!(qwen_usable_by_run(&config));
        config.chat = true;
        assert!(!qwen_usable_by_run(&config));
        config.surface = RuntimeSurface::Server;
        assert!(qwen_usable_by_run(&config));
        config.dspark = DSparkRequest::On;
        assert!(!qwen_usable_by_run(&config));
        // Every surface that can reach Qwen is one where the per-request
        // gate can also allow it.
        config.dspark = DSparkRequest::Auto;
        config.chat = false;
        config.surface = RuntimeSurface::DirectRun;
        assert!(qwen_usable_by_run(&config));
        assert!(qwen_allowed_for_request(
            config.chat || config.dspark == DSparkRequest::On
        ));
    }

    #[test]
    fn eagle3_runtime_keeps_drafting_through_misses() {
        let config = RuntimeConfig::resolve(RunArgs::default(), |_| None).unwrap();
        let runtime = eagle3_runtime_config(&config, 4_096);
        assert!(!runtime.disable_on_miss);
        assert_eq!(runtime.max_drafts, eagle3_provider::MAXIMUM_DRAFTS);
        assert_eq!(runtime.max_context_tokens, Some(4_096));
        assert!(runtime.economic_grace_steps > 0);
        assert!(
            DSparkRuntime::<NativeDraftBackend>::new(
                crate::dspark_runtime::Mode::Auto,
                None,
                runtime
            )
            .is_ok()
        );
    }

    #[test]
    fn qwen_probe_first_plan_reserves_the_established_hybrid_pair() {
        let context_capacity = 4_389_usize;
        let both = qwen_candidate_plan(Device::Mps, true, true, context_capacity, 9, 800);
        assert_eq!(both.initial, Some(QwenVariant::Probe06B));
        assert!(both.wide_lazy);
        assert_eq!(both.state, QwenRuntimeState::Probe06B);
        assert_eq!(both.context_capacity, context_capacity);
        assert_eq!(both.reserved_verify_positions, 9);
        assert_eq!(both.reserved_verify_bytes, 800);
        assert_eq!(
            both.reserved_provider_bytes,
            qwen_provider_reserve(
                Device::Mps,
                &[QwenVariant::Probe06B, QwenVariant::Wide17B],
                context_capacity,
            )
            .unwrap()
        );

        let wide_only = qwen_candidate_plan(Device::Mps, false, true, context_capacity, 9, 800);
        assert_eq!(wide_only.initial, Some(QwenVariant::Wide17B));
        assert!(!wide_only.wide_lazy);
        assert!(wide_only.reserved_provider_bytes < both.reserved_provider_bytes);

        let probe_only = qwen_probe_only_fallback(both, Device::Mps).unwrap();
        assert_eq!(probe_only.initial, Some(QwenVariant::Probe06B));
        assert!(!probe_only.wide_lazy);
        assert_eq!(probe_only.context_capacity, context_capacity);
        assert_eq!(probe_only.reserved_verify_positions, 9);
        assert_eq!(probe_only.reserved_verify_bytes, 800);
        assert!(probe_only.reserved_provider_bytes < both.reserved_provider_bytes);
        assert!(qwen_probe_only_fallback(wide_only, Device::Mps).is_none());
    }

    #[test]
    fn qwen_reserve_covers_models_bounded_context_kv_and_scratch() {
        let context_capacity = 4_389_usize;
        let gpu_probe =
            qwen_provider_reserve(Device::Mps, &[QwenVariant::Probe06B], context_capacity).unwrap();
        let cpu_probe =
            qwen_provider_reserve(Device::Cpu, &[QwenVariant::Probe06B], context_capacity).unwrap();
        let architecture = QwenVariant::Probe06B.architecture();
        let gpu_model = QwenVariant::Probe06B.parameter_count() * 2;
        let gpu_kv = architecture.layers
            * 2
            * context_capacity as u64
            * architecture.key_value_heads
            * architecture.head_dim
            * 2;
        assert_eq!(gpu_probe, gpu_model + gpu_kv + 128 * 1024 * 1024);
        assert_eq!(
            cpu_probe,
            QwenVariant::Probe06B.parameter_count() * 4 + gpu_kv * 2 + 128 * 1024 * 1024
        );
        assert!(qwen_provider_reserve(Device::Mps, &[QwenVariant::Probe06B], 0).is_err());
        assert!(
            qwen_provider_reserve(
                Device::Mps,
                &[QwenVariant::Probe06B],
                architecture.maximum_position as usize + 1,
            )
            .is_err()
        );
    }

    #[test]
    fn qwen_capacity_tracks_the_exact_target_bound_plus_one_proposal() {
        let budget = ContextGrowthBudget {
            bytes_per_capacity_token: 100,
            bytes_per_layer_capacity_token: 25,
            mla_layers: 4,
            initial_capacity_tokens: 16,
            initial_provider_bytes: 1_600,
            model_max_context_tokens: 1_048_576,
            admitted_expanded_context_tokens: 4_369,
        };
        assert_eq!(qwen_context_capacity(budget).unwrap(), 4_389);
    }

    #[test]
    fn qwen_full_commit_validates_width_without_reserving_unallocated_snapshots() {
        let snapshots = VerifySnapshotBudget {
            bytes_per_kda_generation: 100,
            replay_bytes_per_position: 3,
            replay_fixed_bytes: 5,
            max_positions: 64,
        };
        assert_eq!(qwen_verify_reserve(snapshots, 2).unwrap(), (3, 0));
        assert_eq!(qwen_verify_reserve(snapshots, 8).unwrap(), (9, 0));
        assert!(qwen_verify_reserve(snapshots, 64).is_err());

        // Ordinary verifiers still charge their exact live admission (one
        // final generation plus a 5 + 9*3 byte prefix record); only the
        // provider-proven full-commit mode bypasses it.
        let widest = snapshots.admission(9).unwrap();
        assert_eq!(widest.additional_over_decode_reserve_bytes, 32);
        assert_eq!(verify_live_admission_bytes(widest, 800), 0);
        assert_eq!(verify_live_admission_bytes(widest, 20), 12);
        assert_eq!(
            verify_live_admission_bytes(snapshots.admission(3).unwrap(), 800),
            0
        );

        const GIB: u64 = 1 << 30;
        let host = HostMemory {
            physical_bytes: Some(64 * GIB),
            available_bytes: Some(64 * GIB),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        };
        let plan = QwenPlan {
            state: QwenRuntimeState::Probe06B,
            initial: Some(QwenVariant::Probe06B),
            wide_lazy: false,
            reserved_provider_bytes: 8 * GIB,
            reserved_verify_bytes: 0,
            reserved_verify_positions: 9,
            context_capacity: 4_389,
        };
        let base = FixedCosts {
            host_bytes: 0,
            provider_bytes: 40 * GIB,
        };
        let old_model_only = select_resident_prefix(
            host,
            ProviderMemory::Unified {
                recommended_working_set_bytes: None,
                available_working_set_bytes: None,
            },
            &[3 * GIB],
            FixedCosts {
                host_bytes: 0,
                provider_bytes: 48 * GIB,
            },
            ResidencyOverride::default(),
            ResidencyPolicy::default(),
        );
        assert_eq!(old_model_only.resident_layers, 1);

        let proof = qwen_fixed_costs(base, plan).unwrap();
        assert_eq!(proof.provider_bytes, 48 * GIB);
        let verifier_safe = select_resident_prefix(
            host,
            ProviderMemory::Unified {
                recommended_working_set_bytes: None,
                available_working_set_bytes: None,
            },
            &[3 * GIB],
            proof,
            ResidencyOverride::default(),
            ResidencyPolicy::default(),
        );
        assert_eq!(verifier_safe.resident_layers, 1);
    }

    #[test]
    fn qwen_admission_rejects_unknown_or_exceeded_fixed_budgets() {
        let selection = |stop| ResidencySelection {
            resident_layers: 0,
            resident_provider_bytes: 0,
            host_envelope_bytes: None,
            provider_envelope_bytes: None,
            stop,
            override_clamped_by_safety: false,
        };
        assert!(!qwen_residency_admitted(&selection(
            ResidencyStop::HostBudgetUnknown
        )));
        assert!(!qwen_residency_admitted(&selection(
            ResidencyStop::FixedCostsExceedBudget
        )));
        assert!(qwen_residency_admitted(&selection(
            ResidencyStop::NextLayerWouldExceedBudget
        )));
        assert!(qwen_residency_admitted(&selection(
            ResidencyStop::AllLayersFit
        )));
    }

    fn pin_snapshot_hot_at(
        slots: &[(u32, u16)],
        heat: f32,
        total_weight: f64,
    ) -> crate::expert_heat::ExpertHeatSnapshot {
        let mut heats = vec![0.0_f32; crate::experts::K3_EXPERT_RAW_FILES].into_boxed_slice();
        for &(layer, expert) in slots {
            heats[(layer - 1) as usize * crate::routing::K3_EXPERTS + expert as usize] = heat;
        }
        crate::expert_heat::ExpertHeatSnapshot {
            heats,
            total_weight,
        }
    }

    #[test]
    fn assemble_expert_spans_prefers_pinned_then_hits_then_demand() {
        let span_bytes = 8_usize;
        let pinned_bytes = vec![0xAA_u8; span_bytes];
        let pinned = vec![(7_u16, PageAlignedSpan::copy_from(&pinned_bytes).unwrap())];
        let hit_bytes = vec![0xBB_u8; span_bytes];
        let hits = vec![(11_u16, &hit_bytes[..])];
        let demand_ids = [3_u16, 20_u16];
        let demand_bytes: Vec<u8> = [0x33_u8; 8].into_iter().chain([0x44_u8; 8]).collect();
        let spans = assemble_expert_spans(
            &[3, 7, 11, 20],
            &pinned,
            &hits,
            Some((&demand_ids, &demand_bytes)),
            span_bytes,
        )
        .unwrap();
        assert_eq!(spans.len(), 4);
        assert_eq!(spans[0], &[0x33_u8; 8][..]);
        assert_eq!(spans[1], &pinned_bytes[..]);
        assert_eq!(spans[2], &hit_bytes[..]);
        assert_eq!(spans[3], &[0x44_u8; 8][..]);

        // All-pinned tiles need no demand slab at all.
        let alone = assemble_expert_spans(&[7], &pinned, &[], None, span_bytes).unwrap();
        assert_eq!(alone, vec![&pinned_bytes[..]]);

        // A tile expert with no source is an authoritative failure.
        assert!(assemble_expert_spans(&[7, 9], &pinned, &[], None, span_bytes).is_err());
        // A pinned span whose length disagrees with the corpus is refused.
        assert!(assemble_expert_spans(&[7], &pinned, &[], None, span_bytes + 1).is_err());
    }

    #[test]
    fn pin_tier_serves_second_route_from_ram_with_identical_bytes() {
        use crate::experts::{K3_EXPERT_SOURCE_BYTES, RawExpertCorpus};

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "deltafin-engine-pin-{}-{nonce}",
            std::process::id()
        ));
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
        let cache = root.join("k3-experts");
        std::fs::create_dir_all(&cache).unwrap();
        for (expert, fill) in [(3_u16, 0xA3_u8), (7, 0xA7), (11, 0xAB)] {
            std::fs::write(
                cache.join(format!("L1-E{expert}.bin")),
                vec![fill; K3_EXPERT_SOURCE_BYTES],
            )
            .unwrap();
        }
        let corpus = RawExpertCorpus::open_raw_v1(&root).unwrap();
        let reader = Reader::with_arena_capacity(2, 2).unwrap();
        let span_bytes = corpus.layout().expert_span_bytes();
        let tier = ExpertPinTier::plan(
            &pin_snapshot_hot_at(&[(1, 3), (1, 7)], 300.0, 512.0),
            span_bytes,
            2 * span_bytes as u64,
        )
        .unwrap();

        // The invariant baseline: the exact bytes a tier-free read produces.
        let baseline = corpus.read_union(&reader, 1, &[3, 7, 11]).unwrap();
        let expected = baseline.buffers().other().to_vec();
        drop(baseline);

        // First route: everything streams, candidates promote stickily.
        let first =
            read_expert_tile_with_prefetch(&corpus, &reader, Some(&tier), 1, &[3, 7, 11], None)
                .unwrap();
        assert!(matches!(first, ExpertTileLease::Contiguous(_)));
        drop(first);
        assert_eq!(tier.resident_experts(), 2);
        assert_eq!(tier.resident_bytes(), 2 * span_bytes as u64);

        // Second route: both candidates come from RAM; only expert 11 reads.
        let second =
            read_expert_tile_with_prefetch(&corpus, &reader, Some(&tier), 1, &[3, 7, 11], None)
                .unwrap();
        let ExpertTileLease::Scattered {
            layout,
            pinned,
            hits,
            demand,
        } = second
        else {
            panic!("tier-resident tile did not use the scattered lease");
        };
        assert_eq!(layout, corpus.layout());
        assert_eq!(
            pinned.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![3, 7]
        );
        assert!(hits.is_empty());
        let demand = demand.expect("expert 11 is not a candidate and must stream");
        assert_eq!(demand.expert_ids(), [11]);

        // Byte-identical to the tier-free union, span by span.
        let hit_spans: Vec<(u16, &[u8])> = Vec::new();
        let spans = assemble_expert_spans(
            &[3, 7, 11],
            &pinned,
            &hit_spans,
            Some((demand.expert_ids(), demand.buffers().other())),
            span_bytes,
        )
        .unwrap();
        for (index, span) in spans.iter().enumerate() {
            assert_eq!(
                *span,
                &expected[index * span_bytes..(index + 1) * span_bytes],
                "span {index} diverged from the tier-free read"
            );
        }
        assert_eq!(tier.hits(), 2);

        // An all-pinned tile performs no file I/O and still assembles.
        drop(demand);
        let third = read_expert_tile_with_prefetch(&corpus, &reader, Some(&tier), 1, &[3, 7], None)
            .unwrap();
        let ExpertTileLease::Scattered {
            pinned: third_pinned,
            hits: third_hits,
            demand: third_demand,
            ..
        } = third
        else {
            panic!("all-pinned tile did not use the scattered lease");
        };
        assert_eq!(third_pinned.len(), 2);
        assert!(third_hits.is_empty());
        assert!(third_demand.is_none());
        let spans =
            assemble_expert_spans(&[3, 7], &third_pinned, &hit_spans, None, span_bytes).unwrap();
        assert_eq!(spans[0], &expected[..span_bytes]);
        assert_eq!(spans[1], &expected[span_bytes..2 * span_bytes]);
        assert_eq!(tier.hits(), 4);
    }

    /// Records what an early drain was offered, and can refuse.
    struct RecordingDrain {
        offers: Vec<Vec<(u16, Vec<u8>)>>,
        accept: bool,
    }

    impl RecordingDrain {
        const fn new(accept: bool) -> Self {
            Self {
                offers: Vec::new(),
                accept,
            }
        }

        fn offered_experts(&self) -> Vec<u16> {
            let mut experts: Vec<u16> = self
                .offers
                .iter()
                .flat_map(|offer| offer.iter().map(|(expert, _)| *expert))
                .collect();
            experts.sort_unstable();
            experts
        }
    }

    impl ExpertDrain for RecordingDrain {
        fn stage(&mut self, arrived: &[(u16, &[u8])]) -> bool {
            self.offers.push(
                arrived
                    .iter()
                    .map(|(expert, span)| (*expert, span.to_vec()))
                    .collect(),
            );
            self.accept
        }
    }

    #[test]
    fn early_drain_offers_authoritative_bytes_and_stops_when_refused() {
        use crate::experts::{K3_EXPERT_SOURCE_BYTES, RawExpertCorpus};

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "deltafin-engine-drain-{}-{nonce}",
            std::process::id()
        ));
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
        let cache = root.join("k3-experts");
        std::fs::create_dir_all(&cache).unwrap();
        let experts: [u16; 4] = [2, 5, 9, 14];
        for expert in experts {
            std::fs::write(
                cache.join(format!("L1-E{expert}.bin")),
                vec![expert as u8; K3_EXPERT_SOURCE_BYTES],
            )
            .unwrap();
        }
        let corpus = RawExpertCorpus::open_raw_v1(&root).unwrap();
        let reader = Reader::with_arena_capacity(2, 2).unwrap();
        let span_bytes = corpus.layout().expert_span_bytes();
        let tier = ExpertPinTier::plan(
            &pin_snapshot_hot_at(&[(1, 2), (1, 5)], 300.0, 512.0),
            span_bytes,
            2 * span_bytes as u64,
        )
        .unwrap();

        // Promote the two candidates so the next tile has a pinned partition
        // to offer with no I/O at all.
        drop(
            read_expert_tile_draining(
                &corpus,
                &reader,
                Some(&tier),
                1,
                &experts,
                None,
                &mut NoExpertDrain,
            )
            .unwrap(),
        );
        assert_eq!(tier.resident_experts(), 2);

        let mut drain = RecordingDrain::new(true);
        let lease = read_expert_tile_draining(
            &corpus,
            &reader,
            Some(&tier),
            1,
            &experts,
            None,
            &mut drain,
        )
        .unwrap();

        // Whatever was offered must be exactly the bytes the authoritative
        // tile ends up using for that expert. An early drain that could hand
        // out different bytes than the finish call is the one way this
        // optimization could change output.
        let ExpertTileLease::Scattered {
            pinned,
            hits,
            demand,
            ..
        } = &lease
        else {
            panic!("tier-resident tile did not use the scattered lease");
        };
        let hit_spans: Vec<(u16, &[u8])> = hits
            .iter()
            .map(|batch| (batch.expert_ids()[0], batch.buffers().other()))
            .collect();
        let spans = assemble_expert_spans(
            &experts,
            pinned,
            &hit_spans,
            demand
                .as_ref()
                .map(|batch| (batch.expert_ids(), batch.buffers().other())),
            span_bytes,
        )
        .unwrap();
        for offer in &drain.offers {
            assert!(!offer.is_empty(), "an empty batch was offered");
            assert!(
                offer.windows(2).all(|pair| pair[0].0 < pair[1].0),
                "an offered batch was not canonical ascending"
            );
            for (expert, bytes) in offer {
                let index = experts.iter().position(|id| id == expert).unwrap();
                assert_eq!(
                    bytes.as_slice(),
                    spans[index],
                    "expert {expert} was offered bytes the tile did not use"
                );
            }
        }
        // Every expert is offered at most once: a repeat would be staged twice.
        let offered = drain.offered_experts();
        assert!(offered.windows(2).all(|pair| pair[0] < pair[1]));
        // The pinned pair costs no I/O, so it is always available to offer.
        assert!(offered.contains(&2) && offered.contains(&5));
        drop(lease);

        // A refusal is terminal for the tile: nothing further is offered.
        let mut refusing = RecordingDrain::new(false);
        drop(
            read_expert_tile_draining(
                &corpus,
                &reader,
                Some(&tier),
                1,
                &experts,
                None,
                &mut refusing,
            )
            .unwrap(),
        );
        assert_eq!(
            refusing.offers.len(),
            1,
            "a refused drain was offered another batch"
        );
    }

    fn pin_selection(resident_layers: usize, stop: ResidencyStop) -> ResidencySelection {
        ResidencySelection {
            resident_layers,
            resident_provider_bytes: 0,
            host_envelope_bytes: None,
            provider_envelope_bytes: None,
            stop,
            override_clamped_by_safety: false,
        }
    }

    const PIN_SPAN: u64 = 1 << 20;

    fn pin_base() -> FixedCosts {
        FixedCosts {
            host_bytes: 8 * PIN_SPAN,
            provider_bytes: 0,
        }
    }

    fn wide_qwen() -> QwenPlan {
        QwenPlan {
            state: QwenRuntimeState::NotInstalled,
            initial: None,
            wide_lazy: true,
            reserved_provider_bytes: 4 * PIN_SPAN,
            reserved_verify_bytes: 0,
            reserved_verify_positions: 0,
            context_capacity: 0,
        }
    }

    #[test]
    fn pins_take_only_headroom_from_a_spine_that_fits_completely() {
        let reference = pin_selection(93, ResidencyStop::AllLayersFit);
        let headroom = pin_base().host_bytes + 10 * PIN_SPAN;
        let select = |fixed: FixedCosts| -> Result<(QwenPlan, ResidencySelection, u64)> {
            let layers = if fixed.host_bytes <= headroom { 93 } else { 92 };
            let stop = if layers == 93 {
                ResidencyStop::AllLayersFit
            } else {
                ResidencyStop::NextLayerWouldExceedBudget
            };
            Ok((wide_qwen(), pin_selection(layers, stop), 0))
        };
        for eligible in [0_usize, 1, 10, 300] {
            assert_eq!(
                admit_pin_count(eligible, PIN_SPAN, pin_base(), &select, wide_qwen(), &reference, true),
                eligible.min(10),
                "eligible={eligible}"
            );
        }
    }

    #[test]
    fn pins_displace_a_partial_spine_until_the_memory_proof_fails() {
        let reference = pin_selection(7, ResidencyStop::NextLayerWouldExceedBudget);
        let select = |fixed: FixedCosts| -> Result<(QwenPlan, ResidencySelection, u64)> {
            let charge = (fixed.host_bytes - pin_base().host_bytes) / PIN_SPAN;
            if charge > 50 {
                return Ok((wide_qwen(), pin_selection(0, ResidencyStop::FixedCostsExceedBudget), 0));
            }
            let layers = 7_usize.saturating_sub(charge as usize / 8);
            Ok((wide_qwen(), pin_selection(layers, ResidencyStop::NextLayerWouldExceedBudget), 0))
        };
        // Partial spine: every pin up to the proof limit beats a resident layer.
        assert_eq!(
            admit_pin_count(300, PIN_SPAN, pin_base(), &select, wide_qwen(), &reference, false),
            50
        );
    }

    #[test]
    fn pins_never_push_out_the_qwen_drafter() {
        let reference = pin_selection(7, ResidencyStop::NextLayerWouldExceedBudget);
        let select = |fixed: FixedCosts| -> Result<(QwenPlan, ResidencySelection, u64)> {
            let charge = (fixed.host_bytes - pin_base().host_bytes) / PIN_SPAN;
            let qwen = if charge <= 3 {
                wide_qwen()
            } else {
                QwenPlan::inactive(QwenRuntimeState::MemoryRejected)
            };
            Ok((qwen, pin_selection(5, ResidencyStop::NextLayerWouldExceedBudget), 0))
        };
        assert_eq!(
            admit_pin_count(64, PIN_SPAN, pin_base(), &select, wide_qwen(), &reference, false),
            3
        );
    }

    #[test]
    fn pin_admission_fails_closed_on_errors_and_overflow() {
        let reference = pin_selection(93, ResidencyStop::AllLayersFit);
        let failing = |_: FixedCosts| -> Result<(QwenPlan, ResidencySelection, u64)> {
            Err(DeltafinError::new("probe failed"))
        };
        assert_eq!(
            admit_pin_count(16, PIN_SPAN, pin_base(), &failing, wide_qwen(), &reference, true),
            0
        );
        let accepting = |_: FixedCosts| -> Result<(QwenPlan, ResidencySelection, u64)> {
            Ok((wide_qwen(), pin_selection(93, ResidencyStop::AllLayersFit), 0))
        };
        let saturated = FixedCosts {
            host_bytes: u64::MAX - 1,
            provider_bytes: 0,
        };
        assert_eq!(
            admit_pin_count(16, PIN_SPAN, saturated, &accepting, wide_qwen(), &reference, true),
            0
        );
    }

    #[test]
    fn an_explicitly_requested_prefix_outranks_the_pin_tier() {
        for stop in [
            ResidencyStop::AllLayersFit,
            ResidencyStop::ExplicitLayerLimit,
            ResidencyStop::ExplicitByteLimit,
        ] {
            assert!(pin_tier_keeps_resident_prefix(stop), "{stop:?}");
        }
        // A prefix the memory budget itself cut short is the only one that
        // may give way, and an unusable selection protects nothing.
        for stop in [
            ResidencyStop::NextLayerWouldExceedBudget,
            ResidencyStop::FixedCostsExceedBudget,
            ResidencyStop::LayerCostUnknown,
            ResidencyStop::HostBudgetUnknown,
            ResidencyStop::DeviceBudgetUnknown,
            ResidencyStop::InvalidPolicy,
            ResidencyStop::ArithmeticOverflow,
        ] {
            assert!(!pin_tier_keeps_resident_prefix(stop), "{stop:?}");
        }
        // Forty of the ninety-three layers were requested by the operator;
        // the fortieth is lost to an eleventh expert, so only ten are kept.
        let reference = pin_selection(40, ResidencyStop::ExplicitLayerLimit);
        let headroom = pin_base().host_bytes + 10 * PIN_SPAN;
        let select = |fixed: FixedCosts| -> Result<(QwenPlan, ResidencySelection, u64)> {
            let (layers, stop) = if fixed.host_bytes <= headroom {
                (40, ResidencyStop::ExplicitLayerLimit)
            } else {
                (39, ResidencyStop::NextLayerWouldExceedBudget)
            };
            Ok((wide_qwen(), pin_selection(layers, stop), 0))
        };
        assert_eq!(
            admit_pin_count(
                300,
                PIN_SPAN,
                pin_base(),
                &select,
                wide_qwen(),
                &reference,
                pin_tier_keeps_resident_prefix(reference.stop),
            ),
            10
        );
    }

    fn real_host(total_gib: u64, available_gib: u64) -> HostMemory {
        HostMemory {
            physical_bytes: Some(total_gib * GIB),
            available_bytes: Some(available_gib * GIB),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        }
    }

    /// One K3 routed expert as the raw layout spans it.
    const REAL_EXPERT_SPAN: u64 = 17_547_264;

    #[test]
    fn a_big_pin_budget_cannot_displace_a_spine_that_fits_on_a_large_host() {
        // The host class of public issue #25 / PR #23: 128 GiB with about
        // 103 GiB free, an int8-sized spine that fits with some room to spare
        // and the wide Qwen drafter installed. Charging a 32 GB pin ceiling
        // before selection collapsed the resident prefix; admitting it after
        // selection must leave all 93 layers and the drafter alone.
        let layers = vec![GIB / 2; 93];
        let discovered = qwen_candidate_plan(Device::Mps, true, true, 8192, 8, 0);
        assert_ne!(discovered.reserved_provider_bytes, 0);
        let selector = PlanSelector {
            host_memory: real_host(128, 103),
            provider_memory: ProviderMemory::Unified {
                recommended_working_set_bytes: Some(96 * GIB),
                available_working_set_bytes: Some(96 * GIB),
            },
            provider_layer_bytes: &layers,
            residency_override: ResidencyOverride::default(),
            discovered_qwen: discovered,
            device: Device::Mps,
        };
        let fixed = FixedCosts {
            host_bytes: 20 * GIB,
            provider_bytes: 2 * GIB,
        };
        let (reference_qwen, reference, _) = selector.select(fixed).unwrap();
        assert_eq!(reference.stop, ResidencyStop::AllLayersFit);
        assert_eq!(reference.resident_layers, 93);
        assert_eq!(reference_qwen, discovered);
        let select = |fixed: FixedCosts| selector.select(fixed);
        for pin_gb in [8_u64, 16, 24, 32] {
            let eligible = (pin_gb * 1_000_000_000 / REAL_EXPERT_SPAN) as usize;
            let up_front = selector
                .select(FixedCosts {
                    host_bytes: fixed.host_bytes + eligible as u64 * REAL_EXPERT_SPAN,
                    ..fixed
                })
                .unwrap();
            if pin_gb >= 24 {
                assert!(
                    up_front.1.resident_layers < 93,
                    "pin {pin_gb} GB charged up front must reproduce the displacement"
                );
            }
            let admitted = admit_pin_count(
                eligible,
                REAL_EXPERT_SPAN,
                fixed,
                &select,
                reference_qwen,
                &reference,
                pin_tier_keeps_resident_prefix(reference.stop),
            );
            let charged = |count: usize| FixedCosts {
                host_bytes: fixed.host_bytes + count as u64 * REAL_EXPERT_SPAN,
                ..fixed
            };
            let (qwen, residency, _) = selector.select(charged(admitted)).unwrap();
            assert_eq!(residency.resident_layers, 93, "pin {pin_gb} GB");
            assert_eq!(qwen, reference_qwen, "pin {pin_gb} GB");
            if pin_gb >= 24 {
                assert!(admitted < eligible, "headroom is finite: pin {pin_gb} GB");
                assert!(admitted > 0, "some headroom remains: pin {pin_gb} GB");
            }
            if admitted < eligible {
                // Maximal: one more expert would cost a layer or the drafter.
                let (qwen, residency, _) = selector.select(charged(admitted + 1)).unwrap();
                assert!(
                    residency.resident_layers < 93 || qwen != reference_qwen,
                    "pin {pin_gb} GB left admissible headroom unused"
                );
            }
        }
    }

    #[test]
    fn pins_cannot_push_the_qwen_drafter_out_on_a_memory_tight_host() {
        // The incident behind the ordering: a 64 GiB host whose matured auto
        // roster (about 1,700 experts, 27 GiB) reserved up front left no room
        // for the drafter, which was then memory-rejected along with the
        // whole resident spine.
        let layers = vec![(GIB * 6) / 10; 93];
        let discovered = qwen_candidate_plan(Device::Mps, true, true, 8192, 8, 0);
        let selector = PlanSelector {
            host_memory: real_host(64, 50),
            provider_memory: ProviderMemory::Unified {
                recommended_working_set_bytes: Some(48 * GIB),
                available_working_set_bytes: Some(48 * GIB),
            },
            provider_layer_bytes: &layers,
            residency_override: ResidencyOverride::default(),
            discovered_qwen: discovered,
            device: Device::Mps,
        };
        let fixed = FixedCosts {
            host_bytes: 6 * GIB,
            provider_bytes: 3 * GIB,
        };
        let (reference_qwen, reference, _) = selector.select(fixed).unwrap();
        assert_eq!(reference_qwen, discovered, "the drafter fits without pins");
        assert!(
            (1..93).contains(&reference.resident_layers),
            "a partly resident spine: {}",
            reference.resident_layers
        );
        assert!(!pin_tier_keeps_resident_prefix(reference.stop));
        let roster = 1_700_usize;
        let charged = |count: usize| FixedCosts {
            host_bytes: fixed.host_bytes + count as u64 * REAL_EXPERT_SPAN,
            ..fixed
        };
        let up_front = selector.select(charged(roster)).unwrap();
        assert_ne!(
            up_front.0, reference_qwen,
            "charging the whole roster first must reproduce the lost drafter"
        );
        let admitted = admit_pin_count(
            roster,
            REAL_EXPERT_SPAN,
            fixed,
            &|fixed| selector.select(fixed),
            reference_qwen,
            &reference,
            false,
        );
        assert!(admitted > 0 && admitted < roster, "admitted {admitted}");
        let (qwen, residency, _) = selector.select(charged(admitted)).unwrap();
        assert_eq!(qwen, reference_qwen);
        assert!(qwen_residency_admitted(&residency));
        // A partly resident spine may give way to pins, never the drafter.
        assert!(residency.resident_layers <= reference.resident_layers);
        let (qwen, residency, _) = selector.select(charged(admitted + 1)).unwrap();
        assert!(
            qwen != reference_qwen || !qwen_residency_admitted(&residency),
            "one more expert still left the drafter and the proof intact"
        );
    }

    #[test]
    fn an_explicit_zero_layer_request_is_told_apart_from_an_over_budget_probe() {
        // Zero resident layers means two different things: the operator asked
        // for it, or fixed costs exceed the budget. Both report zero layers,
        // so only the memory proof can bound admission.
        let layers = vec![GIB; 93];
        let selector = PlanSelector {
            host_memory: real_host(64, 48),
            provider_memory: ProviderMemory::Host,
            provider_layer_bytes: &layers,
            residency_override: ResidencyOverride {
                requested_layers: Some(0),
                requested_provider_bytes: None,
            },
            discovered_qwen: QwenPlan::inactive(QwenRuntimeState::NotInstalled),
            device: Device::Cpu,
        };
        let fixed = FixedCosts {
            host_bytes: 10 * GIB,
            provider_bytes: 0,
        };
        let (qwen, reference, _) = selector.select(fixed).unwrap();
        assert_eq!(reference.resident_layers, 0);
        assert_eq!(reference.stop, ResidencyStop::ExplicitLayerLimit);
        let charged = |count: usize| FixedCosts {
            host_bytes: fixed.host_bytes + count as u64 * GIB,
            ..fixed
        };
        let admitted = admit_pin_count(
            100,
            GIB,
            fixed,
            &|fixed| selector.select(fixed),
            qwen,
            &reference,
            pin_tier_keeps_resident_prefix(reference.stop),
        );
        let expected = (0..=100)
            .rev()
            .find(|&count| {
                selector.select(charged(count)).unwrap().1.stop == ResidencyStop::ExplicitLayerLimit
            })
            .unwrap();
        assert!(expected > 0 && expected < 100, "expected {expected}");
        assert_eq!(admitted, expected);
        let over = selector.select(charged(expected + 1)).unwrap().1;
        assert_eq!(over.resident_layers, 0, "the same layer count as the request");
        assert_eq!(over.stop, ResidencyStop::FixedCostsExceedBudget);
    }

    #[test]
    fn admitted_pin_counts_form_a_prefix_under_the_real_planning_chain() {
        // Binary search is only sound if admission is monotone in the count.
        // Check that claim exhaustively against the real selector across
        // randomized hosts, spines, drafters and overrides.
        const SPAN: u64 = 128 << 20;
        const ELIGIBLE: usize = 128;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for scenario in 0..24 {
            let total = (16 + next() % 177) * GIB;
            let available = total / 100 * (30 + next() % 70);
            let layers: Vec<u64> = (0..93).map(|_| (200 + next() % 800) << 20).collect();
            let provider_memory = match next() % 3 {
                0 => ProviderMemory::Host,
                1 => ProviderMemory::Unified {
                    recommended_working_set_bytes: Some(total / 4 * 3),
                    available_working_set_bytes: Some(available),
                },
                _ => ProviderMemory::Unified {
                    recommended_working_set_bytes: None,
                    available_working_set_bytes: None,
                },
            };
            let residency_override = match next() % 4 {
                0 => ResidencyOverride {
                    requested_layers: Some(0),
                    requested_provider_bytes: None,
                },
                1 => ResidencyOverride {
                    requested_layers: Some((next() % 94) as usize),
                    requested_provider_bytes: None,
                },
                _ => ResidencyOverride::default(),
            };
            let discovered_qwen = if next() % 2 == 0 {
                qwen_candidate_plan(Device::Mps, true, true, 4096, 8, 0)
            } else {
                QwenPlan::inactive(QwenRuntimeState::NotInstalled)
            };
            let fixed = FixedCosts {
                host_bytes: (1 + next() % 30) * GIB,
                provider_bytes: (next() % 8) * GIB,
            };
            let selector = PlanSelector {
                host_memory: HostMemory {
                    physical_bytes: Some(total),
                    available_bytes: Some(available),
                    cgroup_limit_bytes: None,
                    cgroup_available_bytes: None,
                    constraints_readable: true,
                },
                provider_memory,
                provider_layer_bytes: &layers,
                residency_override,
                discovered_qwen,
                device: Device::Mps,
            };
            let select = |fixed: FixedCosts| selector.select(fixed);
            let Ok((reference_qwen, reference, _)) = select(fixed) else {
                continue;
            };
            for keep_spine in [false, true] {
                let admitted: Vec<bool> = (0..=ELIGIBLE)
                    .map(|count| {
                        pin_count_admitted(
                            count,
                            SPAN,
                            fixed,
                            &select,
                            reference_qwen,
                            &reference,
                            keep_spine,
                        )
                    })
                    .collect();
                assert!(admitted[0], "scenario {scenario}: zero pins always admit");
                let end = admitted.iter().position(|ok| !ok).unwrap_or(admitted.len());
                assert!(
                    admitted[end..].iter().all(|ok| !ok),
                    "scenario {scenario} keep_spine={keep_spine}: admission is not a prefix"
                );
                assert_eq!(
                    admit_pin_count(
                        ELIGIBLE,
                        SPAN,
                        fixed,
                        &select,
                        reference_qwen,
                        &reference,
                        keep_spine
                    ),
                    end - 1,
                    "scenario {scenario} keep_spine={keep_spine}"
                );
            }
        }
    }
}
