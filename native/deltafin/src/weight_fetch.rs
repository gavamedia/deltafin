//! Transactional, Python-free installation of K3's raw weight payloads.
//!
//! Planning consumes only the already-authenticated inventory. Downloads are
//! exact HTTPS ranges, staged in bounded chunks, and published without
//! replacing anything which raced into place. Routed experts are split only
//! after their complete coalesced source run has passed HTTP range validation.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::error::{DeltafinError, Result};
use crate::experts::{
    K3_EXPERT_PACKED_BYTES, K3_EXPERT_SCALE_BYTES, K3_EXPERT_SOURCE_BYTES, K3_EXPERTS_PER_LAYER,
    K3_MOE_LAYER_FIRST, K3_MOE_LAYER_LAST,
};
use crate::inventory::{InventoryDocument, K3Inventory, TensorRecord, safe_tensor_path};
use crate::k3_source;
use crate::sys::fs::{self as sys_fs, Open};
use crate::trusted_download::{
    ByteRange, NativeHttpsTransport, Request, ResponseMeta, TimeoutPolicy, Transport,
    fsync_directory, publish_hard_link, secure_create_new,
};

const USER_AGENT: &str = "deltafin-weight-fetch/1";
pub const DEFAULT_PARALLEL_TRANSFERS: usize = 8;
pub const MAX_PARALLEL_TRANSFERS: usize = 16;
const DEFAULT_EXPERTS_PER_RUN: usize = 8;
const MAX_EXPERTS_PER_RUN: usize = 32;
const TRANSFER_CHUNK_BYTES: u64 = 64 << 20;
/// Minimum user-available space which must remain after the conservative peak
/// allocation for any non-empty weight transfer. This preserves the legacy
/// bulk-fetch contract while applying it to every native entry point.
pub const DEFAULT_REMAINING_FREE_BYTES: u64 = 100_000_000_000;
const SPOTLIGHT_MARKER: &str = ".metadata_never_index";
const EXPERT_RECORDS: usize =
    (K3_MOE_LAYER_LAST as usize) * K3_EXPERTS_PER_LAYER * EXPERT_COMPONENTS.len();

const EXPERT_COMPONENTS: [ExpertComponent; 6] = [
    ExpertComponent::new("w1.weight_packed", K3_EXPERT_PACKED_BYTES, &[3_072, 1_792]),
    ExpertComponent::new("w1.weight_scale", K3_EXPERT_SCALE_BYTES, &[3_072, 112]),
    ExpertComponent::new("w2.weight_packed", K3_EXPERT_PACKED_BYTES, &[3_584, 1_536]),
    ExpertComponent::new("w2.weight_scale", K3_EXPERT_SCALE_BYTES, &[3_584, 96]),
    ExpertComponent::new("w3.weight_packed", K3_EXPERT_PACKED_BYTES, &[3_072, 1_792]),
    ExpertComponent::new("w3.weight_scale", K3_EXPERT_SCALE_BYTES, &[3_072, 112]),
];

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightSelection {
    ResidentSpine,
    ExpertPool,
    Full,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct FetchLimits {
    pub workers: usize,
    pub experts_per_run: usize,
}

impl Default for FetchLimits {
    fn default() -> Self {
        Self {
            workers: DEFAULT_PARALLEL_TRANSFERS,
            experts_per_run: DEFAULT_EXPERTS_PER_RUN,
        }
    }
}

impl FetchLimits {
    fn validate(self) -> Result<Self> {
        if !(1..=MAX_PARALLEL_TRANSFERS).contains(&self.workers) {
            return Err(DeltafinError::new(format!(
                "weight fetch workers must be in 1..={MAX_PARALLEL_TRANSFERS}"
            )));
        }
        if !(1..=MAX_EXPERTS_PER_RUN).contains(&self.experts_per_run) {
            return Err(DeltafinError::new(format!(
                "experts per request must be in 1..={MAX_EXPERTS_PER_RUN}"
            )));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct WeightFetchDryRun {
    pub resident_files_missing: usize,
    pub resident_files_reused: usize,
    pub expert_files_missing: usize,
    pub expert_files_reused: usize,
    pub transfer_requests: usize,
    pub bytes_missing: u64,
    pub bytes_reused: u64,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct WeightFetchProgress {
    pub files_completed: usize,
    pub files_reused: usize,
    pub requests_completed: usize,
    pub bytes_transferred: u64,
    pub total_files: usize,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ExpertFetchOutcome {
    pub planned: WeightFetchDryRun,
    pub progress: WeightFetchProgress,
}

pub trait ProgressSink: Sync {
    fn update(&self, progress: WeightFetchProgress);
}

impl<F> ProgressSink for F
where
    F: Fn(WeightFetchProgress) + Sync,
{
    fn update(&self, progress: WeightFetchProgress) {
        self(progress);
    }
}

#[derive(Debug, Clone)]
pub struct ResidentRangePlan {
    pub tensor_name: String,
    pub destination: PathBuf,
    pub shard: String,
    pub source_start: u64,
    pub source_end: u64,
    pub shard_bytes: u64,
}

impl ResidentRangePlan {
    pub fn size(&self) -> u64 {
        self.source_end - self.source_start + 1
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertSpanPlan {
    pub layer: u32,
    pub expert: u16,
    pub destination: PathBuf,
    pub source_start: u64,
}

#[derive(Debug, Clone)]
pub struct ExpertRunPlan {
    pub layer: u32,
    pub shard: String,
    pub source_start: u64,
    pub source_end: u64,
    pub shard_bytes: u64,
    pub experts: Vec<ExpertSpanPlan>,
    part: PathBuf,
}

impl ExpertRunPlan {
    pub fn size(&self) -> u64 {
        self.source_end - self.source_start + 1
    }
}

#[derive(Debug, Clone)]
pub struct WeightFetchPlan {
    pub selection: WeightSelection,
    pub resident: Vec<ResidentRangePlan>,
    pub expert_runs: Vec<ExpertRunPlan>,
    pub dry_run: WeightFetchDryRun,
    limits: FetchLimits,
    base_url: String,
    model_root: PathBuf,
    resident_directory: PathBuf,
    expert_directory: PathBuf,
}

/// Validated disk accounting for an immutable fetch plan.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct WeightFetchStorage {
    /// Bytes already held by safe resumable partials on the target volume.
    pub staged_bytes: u64,
    /// Permanent bytes still required after crediting resident-file partials.
    pub final_bytes_to_allocate: u64,
    /// Maximum coalesced expert-run growth that may coexist with final files.
    pub expert_run_temporary_bytes: u64,
    /// Bounded range chunks that may coexist with their appended destination.
    pub range_chunk_temporary_bytes: u64,
    /// Conservative peak additional allocation from the current filesystem
    /// state. Existing staged bytes are not counted twice.
    pub peak_additional_bytes: u64,
}

/// Fail-closed capacity result for one immutable, authenticated weight plan.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct WeightFetchCapacity {
    /// Exact final payload bytes absent when this plan was constructed.
    pub missing_payload_bytes: u64,
    /// Additional allocation at the conservative transfer/split peak.
    pub peak_additional_bytes: u64,
    /// User-available bytes reported by the target filesystem.
    pub available_bytes: u64,
    /// Free bytes reserved after the transfer peak.
    pub remaining_free_bytes: u64,
    /// Peak allocation plus the retained free-space floor.
    pub required_available_bytes: u64,
    /// Additional bytes needed before execution may begin.
    pub shortfall_bytes: u64,
}

impl WeightFetchCapacity {
    pub fn has_capacity(self) -> bool {
        self.shortfall_bytes == 0
    }

    pub fn require(self) -> Result<()> {
        if self.has_capacity() {
            return Ok(());
        }
        Err(DeltafinError::new(format!(
            "native weight fetch needs {} more bytes on the target volume \
             ({} peak additional bytes for {} exact missing payload bytes, plus {} bytes \
             which must remain free; {} bytes available). No network transfer was started",
            self.shortfall_bytes,
            self.peak_additional_bytes,
            self.missing_payload_bytes,
            self.remaining_free_bytes,
            self.available_bytes,
        )))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum HostPlatform {
    MacOs,
    Other,
}

impl HostPlatform {
    const fn current() -> Self {
        if cfg!(target_os = "macos") {
            Self::MacOs
        } else {
            Self::Other
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
struct PartialStorage {
    physical: u64,
    committed: u64,
}

/// Authenticated, reusable source catalog for lazy routed-expert admission.
///
/// Constructing the catalog proves the complete 92 x 896 x 6 tensor layout
/// once.  Per-layer demand then performs only bounded index lookups and local
/// file admission before creating a coalesced range plan.  The execution lock
/// prevents a demand request and a speculative prefetch in this process from
/// appending to the same resumable object concurrently; publication remains
/// no-replace safe against other processes.
#[derive(Debug, Clone)]
pub struct ExpertFetchCatalog {
    inner: Arc<ExpertFetchCatalogInner>,
}

#[derive(Debug)]
struct ExpertFetchCatalogInner {
    layouts: Vec<ExpertLayout>,
    shard_bytes: BTreeMap<String, u64>,
    limits: FetchLimits,
    base_url: String,
    model_root: PathBuf,
    resident_directory: PathBuf,
    expert_directory: PathBuf,
    execution_lock: Mutex<()>,
}

#[derive(Debug, Clone)]
enum WorkItem {
    Resident(ResidentRangePlan),
    Experts(ExpertRunPlan),
}

#[derive(Debug, Clone)]
struct ExpertLayout {
    layer: u32,
    expert: u16,
    shard: String,
    hlen: u64,
    data_start: u64,
}

#[derive(Debug, Clone, Copy)]
struct ExpertComponent {
    suffix: &'static str,
    bytes: usize,
    shape: &'static [u64],
}

impl ExpertComponent {
    const fn new(suffix: &'static str, bytes: usize, shape: &'static [u64]) -> Self {
        Self {
            suffix,
            bytes,
            shape,
        }
    }
}

/// Build a read-only installation plan from the pinned, authenticated inventory.
pub fn plan(
    root: &Path,
    inventory: &K3Inventory,
    selection: WeightSelection,
    limits: FetchLimits,
) -> Result<WeightFetchPlan> {
    let base_url = k3_source::base_url()?;
    plan_from_document(root, inventory.document(), selection, limits, base_url)
}

/// Build an expert-only plan for a validated subset of routed layer numbers.
///
/// The complete authenticated inventory is still structurally validated before
/// any subset is admitted. Layer selection narrows downloads only; it never
/// changes tensor coordinates or accepts a partial/untrusted inventory.
pub fn plan_expert_layers(
    root: &Path,
    inventory: &K3Inventory,
    layers: &[u32],
    limits: FetchLimits,
) -> Result<WeightFetchPlan> {
    let layers = validate_expert_layers(layers)?;
    let base_url = k3_source::base_url()?;
    plan_from_document_filtered(
        root,
        inventory.document(),
        WeightSelection::ExpertPool,
        limits,
        base_url,
        Some(&layers),
    )
}

/// Execute a previously constructed plan with bounded in-process HTTPS workers.
pub fn execute(plan: &WeightFetchPlan, progress: &dyn ProgressSink) -> Result<WeightFetchProgress> {
    execute_guarded_with(
        plan,
        progress,
        || NativeHttpsTransport,
        available_disk_bytes,
        HostPlatform::current(),
    )
}

impl WeightFetchPlan {
    /// Inspect resumable objects without following links and calculate the
    /// peak additional storage required to complete this exact plan.
    ///
    /// Resident partials become their final file by hard link and therefore
    /// reduce the permanent allocation still needed. A coalesced expert run,
    /// however, coexists with the split per-expert files and is accounted as
    /// temporary storage. Range chunks also briefly coexist with the bytes
    /// appended to their destination. The largest concurrently active set is
    /// bounded by the plan's worker count.
    pub fn storage_requirement(&self) -> Result<WeightFetchStorage> {
        let mut staged_bytes = 0_u64;
        let mut final_bytes_to_allocate = 0_u64;
        let mut expert_run_growth = Vec::new();
        let mut chunk_growth = Vec::new();

        for item in &self.resident {
            let directory = item.destination.parent().ok_or_else(|| {
                DeltafinError::new("resident destination has no parent directory")
            })?;
            let part = directory.join(format!(".{}.part", item.tensor_name));
            let staged = validated_partial_storage(&part, item.size())?;
            staged_bytes = checked_add(staged_bytes, staged.physical, "staged resident bytes")?;
            // Any incomplete trailing chunk is truncated before transfer. Its
            // physical bytes still reduce net growth from the current disk
            // state, while only the aligned committed prefix reduces the next
            // temporary chunk's size.
            let remaining = item.size() - staged.physical;
            final_bytes_to_allocate = checked_add(
                final_bytes_to_allocate,
                remaining,
                "remaining resident allocation",
            )?;
            let transfer_remaining = item.size() - staged.committed;
            if transfer_remaining != 0 {
                chunk_growth.push(transfer_remaining.min(TRANSFER_CHUNK_BYTES));
            }
        }

        let mut planned_expert_files = 0_u64;
        for run in &self.expert_runs {
            let staged = validated_partial_storage(&run.part, run.size())?;
            staged_bytes = checked_add(staged_bytes, staged.physical, "staged expert-run bytes")?;
            let remaining = run.size() - staged.physical;
            if remaining != 0 {
                expert_run_growth.push(remaining);
            }
            let transfer_remaining = run.size() - staged.committed;
            if transfer_remaining != 0 {
                chunk_growth.push(transfer_remaining.min(TRANSFER_CHUNK_BYTES));
            }
            let files = u64::try_from(run.experts.len())
                .map_err(|_| DeltafinError::new("expert plan count overflows u64"))?;
            planned_expert_files =
                checked_add(planned_expert_files, files, "planned expert file count")?;
        }
        let expert_final = planned_expert_files
            .checked_mul(K3_EXPERT_SOURCE_BYTES as u64)
            .ok_or_else(|| DeltafinError::new("expert final allocation overflowed"))?;
        final_bytes_to_allocate = checked_add(
            final_bytes_to_allocate,
            expert_final,
            "remaining final allocation",
        )?;

        // Every missing file appears exactly once in resident or expert work.
        // Keep this invariant explicit so future planning changes cannot make
        // the capacity calculation silently optimistic.
        let resident_planned: u64 = self.resident.iter().try_fold(0_u64, |total, item| {
            checked_add(total, item.size(), "planned resident bytes")
        })?;
        let planned_final = checked_add(
            resident_planned,
            expert_final,
            "planned final payload bytes",
        )?;
        if planned_final != self.dry_run.bytes_missing {
            return Err(DeltafinError::new(format!(
                "weight plan storage accounting describes {planned_final} missing bytes; dry plan reports {}",
                self.dry_run.bytes_missing
            )));
        }

        let expert_run_temporary_bytes = largest_sum(
            &mut expert_run_growth,
            self.limits.workers,
            "parallel expert-run allocation",
        )?;
        let range_chunk_temporary_bytes = largest_sum(
            &mut chunk_growth,
            self.limits.workers,
            "parallel range-chunk allocation",
        )?;
        let peak_additional_bytes = checked_add(
            checked_add(
                final_bytes_to_allocate,
                expert_run_temporary_bytes,
                "weight peak allocation",
            )?,
            range_chunk_temporary_bytes,
            "weight peak allocation",
        )?;
        Ok(WeightFetchStorage {
            staged_bytes,
            final_bytes_to_allocate,
            expert_run_temporary_bytes,
            range_chunk_temporary_bytes,
            peak_additional_bytes,
        })
    }
}

/// Probe and validate the capacity boundary without creating directories or
/// contacting the network.
pub fn inspect_capacity(plan: &WeightFetchPlan) -> Result<WeightFetchCapacity> {
    inspect_capacity_with(plan, available_disk_bytes)
}

fn inspect_capacity_with<F>(plan: &WeightFetchPlan, available: F) -> Result<WeightFetchCapacity>
where
    F: Fn(&Path) -> Result<u64>,
{
    let storage = plan.storage_requirement()?;
    if plan.dry_run.bytes_missing == 0 {
        return capacity_from_available(plan.dry_run.bytes_missing, storage, 0);
    }
    require_real_directory(&plan.model_root)?;
    capacity_from_available(
        plan.dry_run.bytes_missing,
        storage,
        available(&plan.model_root)?,
    )
}

fn capacity_from_available(
    missing_payload_bytes: u64,
    storage: WeightFetchStorage,
    available_bytes: u64,
) -> Result<WeightFetchCapacity> {
    let remaining_free_bytes = if missing_payload_bytes == 0 {
        0
    } else {
        DEFAULT_REMAINING_FREE_BYTES
    };
    if missing_payload_bytes == 0 && storage.peak_additional_bytes != 0 {
        return Err(DeltafinError::new(
            "completed weight fetch plan still requires additional allocation",
        ));
    }
    let required_available_bytes = storage
        .peak_additional_bytes
        .checked_add(remaining_free_bytes)
        .ok_or_else(|| DeltafinError::new("weight fetch capacity requirement overflowed"))?;
    Ok(WeightFetchCapacity {
        missing_payload_bytes,
        peak_additional_bytes: storage.peak_additional_bytes,
        available_bytes,
        remaining_free_bytes,
        required_available_bytes,
        shortfall_bytes: required_available_bytes.saturating_sub(available_bytes),
    })
}

/// Worst-case transient allocation before an authenticated inventory is
/// locally available. This lets one-shot setup apply the same fail-closed
/// capacity boundary before its first metadata write.
pub fn maximum_temporary_bytes(limits: FetchLimits) -> Result<u64> {
    let limits = limits.validate()?;
    let run = (limits.experts_per_run as u64)
        .checked_mul(K3_EXPERT_SOURCE_BYTES as u64)
        .ok_or_else(|| DeltafinError::new("maximum expert-run bytes overflowed"))?;
    let per_worker = run
        .checked_add(TRANSFER_CHUNK_BYTES)
        .ok_or_else(|| DeltafinError::new("maximum per-worker temporary bytes overflowed"))?;
    (limits.workers as u64)
        .checked_mul(per_worker)
        .ok_or_else(|| DeltafinError::new("maximum parallel temporary bytes overflowed"))
}

/// Worst-case parallel range-chunk allocation for a resident-only plan before
/// the authenticated inventory is locally available.
pub fn maximum_resident_temporary_bytes(limits: FetchLimits) -> Result<u64> {
    let limits = limits.validate()?;
    (limits.workers as u64)
        .checked_mul(TRANSFER_CHUNK_BYTES)
        .ok_or_else(|| DeltafinError::new("maximum resident temporary bytes overflowed"))
}

fn largest_sum(values: &mut [u64], maximum: usize, context: &str) -> Result<u64> {
    values.sort_unstable_by(|left, right| right.cmp(left));
    values
        .iter()
        .take(maximum)
        .try_fold(0_u64, |total, value| checked_add(total, *value, context))
}

fn validated_partial_storage(path: &Path, maximum: u64) -> Result<PartialStorage> {
    match sys_fs::lstat(path) {
        Ok(before) => {
            if before.file_type().is_symlink() || !before.is_file() || before.len() > maximum {
                return Err(DeltafinError::new(format!(
                    "unsafe resumable weight partial {}",
                    path.display()
                )));
            }
            let file = Open::new()
                .read(true)
                .no_follow()
                .open(path)
                .map_err(|error| io_error("open resumable partial for accounting", path, error))?;
            let opened = sys_fs::fstat(&file)
                .map_err(|error| io_error("stat resumable partial for accounting", path, error))?;
            if !opened.is_file()
                || (opened.dev(), opened.ino(), opened.len())
                    != (before.dev(), before.ino(), before.len())
            {
                return Err(DeltafinError::new(format!(
                    "resumable weight partial changed while accounting: {}",
                    path.display()
                )));
            }
            let physical = opened.len();
            let committed = if physical == maximum {
                physical
            } else {
                physical - physical % TRANSFER_CHUNK_BYTES
            };
            Ok(PartialStorage {
                physical,
                committed,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(PartialStorage::default()),
        Err(error) => Err(io_error(
            "inspect resumable partial for accounting",
            path,
            error,
        )),
    }
}

impl ExpertFetchCatalog {
    /// Authenticate the complete expert layout and retain its compact lookup
    /// table for every later routed layer.
    pub fn open(root: &Path, inventory: &K3Inventory, limits: FetchLimits) -> Result<Self> {
        let limits = limits.validate()?;
        require_real_directory(root)?;
        let expert_directory = root.join("k3-experts");
        inspect_optional_directory(&expert_directory)?;
        let document = inventory.document();
        let layouts = derive_all_experts(document)?;
        let shard_bytes = derive_shard_bytes(document)?;
        Self::from_authenticated_parts(root, layouts, shard_bytes, limits, k3_source::base_url()?)
    }

    fn from_authenticated_parts(
        root: &Path,
        layouts: Vec<ExpertLayout>,
        shard_bytes: BTreeMap<String, u64>,
        limits: FetchLimits,
        base_url: String,
    ) -> Result<Self> {
        let expected = (K3_MOE_LAYER_LAST as usize) * K3_EXPERTS_PER_LAYER;
        if layouts.len() != expected {
            return Err(DeltafinError::new(format!(
                "authenticated expert catalog has {} entries; expected {expected}",
                layouts.len()
            )));
        }
        for layer in K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST {
            for expert in 0..K3_EXPERTS_PER_LAYER {
                let index = expert_index(layer, expert as u16)?;
                let layout = layouts.get(index).ok_or_else(|| {
                    DeltafinError::new("authenticated expert catalog index is missing")
                })?;
                if layout.layer != layer || layout.expert != expert as u16 {
                    return Err(DeltafinError::new(format!(
                        "authenticated expert catalog is not canonical at L{layer}-E{expert}"
                    )));
                }
                if !shard_bytes.contains_key(&layout.shard) {
                    return Err(DeltafinError::new(format!(
                        "authenticated expert catalog shard {:?} has no length",
                        layout.shard
                    )));
                }
            }
        }
        Ok(Self {
            inner: Arc::new(ExpertFetchCatalogInner {
                layouts,
                shard_bytes,
                limits,
                base_url,
                model_root: root.to_path_buf(),
                resident_directory: root.join("k3-resident").join("tensors"),
                expert_directory: root.join("k3-experts"),
                execution_lock: Mutex::new(()),
            }),
        })
    }

    /// Plan one routed layer without touching the network. Duplicate expert
    /// IDs are collapsed because a canonical cache object is shared by every
    /// route occurrence, while callers retain their original ordered routing
    /// record for the authoritative reduction.
    pub fn plan_layer(&self, layer: u32, experts: &[u16]) -> Result<WeightFetchPlan> {
        if !(K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(&layer) {
            return Err(DeltafinError::new(format!(
                "routed expert layer must be in {K3_MOE_LAYER_FIRST}..={K3_MOE_LAYER_LAST}"
            )));
        }
        let mut requested = experts.to_vec();
        requested.sort_unstable();
        requested.dedup();
        if requested
            .iter()
            .any(|expert| *expert as usize >= K3_EXPERTS_PER_LAYER)
        {
            return Err(DeltafinError::new(format!(
                "routed expert ID must be below {K3_EXPERTS_PER_LAYER}"
            )));
        }

        let mut dry_run = WeightFetchDryRun::default();
        let mut missing = Vec::new();
        for expert in requested {
            let layout = &self.inner.layouts[expert_index(layer, expert)?];
            let destination = self
                .inner
                .expert_directory
                .join(format!("L{layer}-E{expert}.bin"));
            if exact_regular_file(&destination, K3_EXPERT_SOURCE_BYTES as u64)? {
                dry_run.expert_files_reused += 1;
                dry_run.bytes_reused = checked_add(
                    dry_run.bytes_reused,
                    K3_EXPERT_SOURCE_BYTES as u64,
                    "reused demand bytes",
                )?;
                continue;
            }
            let source_start = 8_u64
                .checked_add(layout.hlen)
                .and_then(|value| value.checked_add(layout.data_start))
                .ok_or_else(|| DeltafinError::new("expert demand source offset overflowed"))?;
            missing.push((layout, destination, source_start));
            dry_run.expert_files_missing += 1;
            dry_run.bytes_missing = checked_add(
                dry_run.bytes_missing,
                K3_EXPERT_SOURCE_BYTES as u64,
                "missing demand bytes",
            )?;
        }
        missing.sort_by_key(|entry| entry.2);
        let mut expert_runs = Vec::new();
        for group in group_adjacent(&missing, self.inner.limits.experts_per_run) {
            let first = group.first().expect("non-empty demand group");
            let last = group.last().expect("non-empty demand group");
            let source_end = last
                .2
                .checked_add(K3_EXPERT_SOURCE_BYTES as u64 - 1)
                .ok_or_else(|| DeltafinError::new("expert demand range overflowed"))?;
            let shard = first.0.shard.clone();
            let part = unique_part(
                &self.inner.expert_directory,
                &format!("L{layer}-{}-{source_end}.demand", first.2),
            )?;
            expert_runs.push(ExpertRunPlan {
                layer,
                shard: shard.clone(),
                source_start: first.2,
                source_end,
                shard_bytes: *self.inner.shard_bytes.get(&shard).ok_or_else(|| {
                    DeltafinError::new(format!(
                        "authenticated inventory shard {shard:?} has no length"
                    ))
                })?,
                experts: group
                    .iter()
                    .map(|(layout, destination, source_start)| ExpertSpanPlan {
                        layer: layout.layer,
                        expert: layout.expert,
                        destination: destination.clone(),
                        source_start: *source_start,
                    })
                    .collect(),
                part,
            });
        }
        dry_run.transfer_requests = expert_runs
            .iter()
            .map(|run| run.size().div_ceil(TRANSFER_CHUNK_BYTES) as usize)
            .sum();
        Ok(WeightFetchPlan {
            selection: WeightSelection::ExpertPool,
            resident: Vec::new(),
            expert_runs,
            dry_run,
            limits: self.inner.limits,
            base_url: self.inner.base_url.clone(),
            model_root: self.inner.model_root.clone(),
            resident_directory: self.inner.resident_directory.clone(),
            expert_directory: self.inner.expert_directory.clone(),
        })
    }

    /// Fetch and durably publish the missing members of one routed layer.
    /// Planning occurs inside the lock so a completed prefetch is observed as
    /// a cache hit instead of triggering a duplicate transfer.
    pub fn fetch_layer(
        &self,
        layer: u32,
        experts: &[u16],
        progress: &dyn ProgressSink,
    ) -> Result<WeightFetchProgress> {
        Ok(self
            .fetch_layer_detailed(layer, experts, progress)?
            .progress)
    }

    pub fn fetch_layer_detailed(
        &self,
        layer: u32,
        experts: &[u16],
        progress: &dyn ProgressSink,
    ) -> Result<ExpertFetchOutcome> {
        let _guard = self
            .inner
            .execution_lock
            .lock()
            .map_err(|_| DeltafinError::new("expert demand execution lock was poisoned"))?;
        let plan = self.plan_layer(layer, experts)?;
        let planned = plan.dry_run;
        let progress = execute(&plan, progress)?;
        Ok(ExpertFetchOutcome { planned, progress })
    }
}

fn expert_index(layer: u32, expert: u16) -> Result<usize> {
    if !(K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(&layer)
        || expert as usize >= K3_EXPERTS_PER_LAYER
    {
        return Err(DeltafinError::new("expert catalog index is out of range"));
    }
    (layer as usize - K3_MOE_LAYER_FIRST as usize)
        .checked_mul(K3_EXPERTS_PER_LAYER)
        .and_then(|value| value.checked_add(expert as usize))
        .ok_or_else(|| DeltafinError::new("expert catalog index overflowed"))
}

fn plan_from_document(
    root: &Path,
    document: &InventoryDocument,
    selection: WeightSelection,
    limits: FetchLimits,
    base_url: String,
) -> Result<WeightFetchPlan> {
    plan_from_document_filtered(root, document, selection, limits, base_url, None)
}

fn plan_from_document_filtered(
    root: &Path,
    document: &InventoryDocument,
    selection: WeightSelection,
    limits: FetchLimits,
    base_url: String,
    expert_layers: Option<&std::collections::BTreeSet<u32>>,
) -> Result<WeightFetchPlan> {
    let limits = limits.validate()?;
    require_real_directory(root)?;
    let resident_directory = root.join("k3-resident").join("tensors");
    let expert_directory = root.join("k3-experts");
    if selection != WeightSelection::ExpertPool {
        inspect_optional_directory(&resident_directory)?;
    }
    if selection != WeightSelection::ResidentSpine {
        inspect_optional_directory(&expert_directory)?;
    }
    let shard_bytes = derive_shard_bytes(document)?;
    let mut dry_run = WeightFetchDryRun::default();
    let mut resident = Vec::new();
    if selection != WeightSelection::ExpertPool {
        for (name, record) in document {
            if is_expert_record(name) {
                continue;
            }
            let size = record.offsets[1] - record.offsets[0];
            let destination = safe_tensor_path(&resident_directory, name)?;
            if exact_regular_file(&destination, size)? {
                dry_run.resident_files_reused += 1;
                dry_run.bytes_reused = checked_add(dry_run.bytes_reused, size, "reused bytes")?;
                continue;
            }
            let (source_start, source_end) = source_range(record, record.offsets)?;
            resident.push(ResidentRangePlan {
                tensor_name: name.clone(),
                destination,
                shard: record.shard.clone(),
                source_start,
                source_end,
                shard_bytes: *shard_bytes.get(&record.shard).ok_or_else(|| {
                    DeltafinError::new(format!("inventory shard {:?} has no length", record.shard))
                })?,
            });
            dry_run.resident_files_missing += 1;
            dry_run.bytes_missing = checked_add(dry_run.bytes_missing, size, "missing bytes")?;
        }
    }
    resident.sort_by(|left, right| {
        (&left.shard, left.source_start).cmp(&(&right.shard, right.source_start))
    });

    let mut expert_runs = Vec::new();
    if selection != WeightSelection::ResidentSpine {
        let layouts = derive_all_experts(document)?;
        for layer in K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST {
            if expert_layers.is_some_and(|selected| !selected.contains(&layer)) {
                continue;
            }
            let mut missing = Vec::new();
            for layout in layouts.iter().filter(|layout| layout.layer == layer) {
                let destination =
                    expert_directory.join(format!("L{}-E{}.bin", layout.layer, layout.expert));
                if exact_regular_file(&destination, K3_EXPERT_SOURCE_BYTES as u64)? {
                    dry_run.expert_files_reused += 1;
                    dry_run.bytes_reused = checked_add(
                        dry_run.bytes_reused,
                        K3_EXPERT_SOURCE_BYTES as u64,
                        "reused bytes",
                    )?;
                } else {
                    let source_start = 8_u64
                        .checked_add(layout.hlen)
                        .and_then(|value| value.checked_add(layout.data_start))
                        .ok_or_else(|| DeltafinError::new("expert source offset overflowed"))?;
                    missing.push((layout, destination, source_start));
                    dry_run.expert_files_missing += 1;
                    dry_run.bytes_missing = checked_add(
                        dry_run.bytes_missing,
                        K3_EXPERT_SOURCE_BYTES as u64,
                        "missing bytes",
                    )?;
                }
            }
            missing.sort_by_key(|entry| entry.2);
            for group in group_adjacent(&missing, limits.experts_per_run) {
                let first = group.first().expect("non-empty group");
                let last = group.last().expect("non-empty group");
                let source_end = last
                    .2
                    .checked_add(K3_EXPERT_SOURCE_BYTES as u64 - 1)
                    .ok_or_else(|| DeltafinError::new("expert run range overflowed"))?;
                let shard = first.0.shard.clone();
                let part =
                    expert_directory.join(format!(".L{layer}-{}-{}.run.part", first.2, source_end));
                expert_runs.push(ExpertRunPlan {
                    layer,
                    shard: shard.clone(),
                    source_start: first.2,
                    source_end,
                    shard_bytes: *shard_bytes.get(&shard).ok_or_else(|| {
                        DeltafinError::new(format!("inventory shard {shard:?} has no length"))
                    })?,
                    experts: group
                        .iter()
                        .map(|(layout, destination, source_start)| ExpertSpanPlan {
                            layer: layout.layer,
                            expert: layout.expert,
                            destination: destination.clone(),
                            source_start: *source_start,
                        })
                        .collect(),
                    part,
                });
            }
        }
    }
    dry_run.transfer_requests = resident
        .iter()
        .map(|item| item.size().div_ceil(TRANSFER_CHUNK_BYTES) as usize)
        .chain(
            expert_runs
                .iter()
                .map(|run| run.size().div_ceil(TRANSFER_CHUNK_BYTES) as usize),
        )
        .sum();
    Ok(WeightFetchPlan {
        selection,
        resident,
        expert_runs,
        dry_run,
        limits,
        base_url,
        model_root: root.to_path_buf(),
        resident_directory,
        expert_directory,
    })
}

fn validate_expert_layers(layers: &[u32]) -> Result<std::collections::BTreeSet<u32>> {
    if layers.is_empty() {
        return Err(DeltafinError::new(
            "expert layer selection must contain at least one layer",
        ));
    }
    let mut selected = std::collections::BTreeSet::new();
    for &layer in layers {
        if !(K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(&layer) {
            return Err(DeltafinError::new(format!(
                "expert layer {layer} is outside {K3_MOE_LAYER_FIRST}..={K3_MOE_LAYER_LAST}"
            )));
        }
        if !selected.insert(layer) {
            return Err(DeltafinError::new(format!(
                "expert layer {layer} was selected more than once"
            )));
        }
    }
    Ok(selected)
}

fn derive_shard_bytes(document: &InventoryDocument) -> Result<BTreeMap<String, u64>> {
    let mut shards = BTreeMap::<String, (u64, u64)>::new();
    for record in document.values() {
        let entry = shards
            .entry(record.shard.clone())
            .or_insert((record.hlen, 0));
        if entry.0 != record.hlen {
            return Err(DeltafinError::new(format!(
                "shard {:?} has inconsistent header lengths",
                record.shard
            )));
        }
        entry.1 = entry.1.max(record.offsets[1]);
    }
    shards
        .into_iter()
        .map(|(name, (hlen, data))| {
            let bytes = 8_u64
                .checked_add(hlen)
                .and_then(|value| value.checked_add(data))
                .ok_or_else(|| DeltafinError::new("shard byte length overflowed"))?;
            Ok((name, bytes))
        })
        .collect()
}

fn derive_all_experts(document: &InventoryDocument) -> Result<Vec<ExpertLayout>> {
    let actual = document
        .keys()
        .filter(|name| is_expert_record(name))
        .count();
    if actual != EXPERT_RECORDS {
        return Err(DeltafinError::new(format!(
            "authenticated inventory has {actual} expert records; expected {EXPERT_RECORDS}"
        )));
    }
    let mut layouts = Vec::with_capacity((K3_MOE_LAYER_LAST as usize) * K3_EXPERTS_PER_LAYER);
    for layer in K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST {
        let mut layer_layouts = Vec::with_capacity(K3_EXPERTS_PER_LAYER);
        for expert in 0..K3_EXPERTS_PER_LAYER {
            layer_layouts.push(derive_expert(document, layer, expert as u16)?);
        }
        layer_layouts.sort_by_key(|layout| layout.data_start);
        let shard = layer_layouts[0].shard.clone();
        let hlen = layer_layouts[0].hlen;
        for pair in layer_layouts.windows(2) {
            if pair[0].shard != shard
                || pair[1].shard != shard
                || pair[0].hlen != hlen
                || pair[1].hlen != hlen
                || pair[0].data_start + K3_EXPERT_SOURCE_BYTES as u64 != pair[1].data_start
            {
                return Err(DeltafinError::new(format!(
                    "layer {layer} expert spans are not one authenticated contiguous shard run"
                )));
            }
        }
        // The physical-order pass above proves contiguity. Store the retained
        // catalog in logical expert-ID order so every demand lookup is one
        // checked arithmetic index rather than an 82,432-entry search.
        layer_layouts.sort_by_key(|layout| layout.expert);
        layouts.extend(layer_layouts);
    }
    Ok(layouts)
}

fn derive_expert(document: &InventoryDocument, layer: u32, expert: u16) -> Result<ExpertLayout> {
    let prefix = format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.");
    let anchor_name = format!("{prefix}{}", EXPERT_COMPONENTS[0].suffix);
    let anchor = document.get(&anchor_name).ok_or_else(|| {
        DeltafinError::new(format!("authenticated inventory lacks {anchor_name:?}"))
    })?;
    let mut expected_start = anchor.offsets[0];
    for component in EXPERT_COMPONENTS {
        let name = format!("{prefix}{}", component.suffix);
        let record = document
            .get(&name)
            .ok_or_else(|| DeltafinError::new(format!("authenticated inventory lacks {name:?}")))?;
        if record.dtype != "U8"
            || record.shape != component.shape
            || record.shard != anchor.shard
            || record.hlen != anchor.hlen
            || record.offsets[0] != expected_start
            || record.offsets[1] - record.offsets[0] != component.bytes as u64
        {
            return Err(DeltafinError::new(format!(
                "expert L{layer}-E{expert} six-record layout is not canonical at {name:?}"
            )));
        }
        expected_start = record.offsets[1];
    }
    if expected_start - anchor.offsets[0] != K3_EXPERT_SOURCE_BYTES as u64 {
        return Err(DeltafinError::new(format!(
            "expert L{layer}-E{expert} is not exactly {K3_EXPERT_SOURCE_BYTES} contiguous bytes"
        )));
    }
    Ok(ExpertLayout {
        layer,
        expert,
        shard: anchor.shard.clone(),
        hlen: anchor.hlen,
        data_start: anchor.offsets[0],
    })
}

fn group_adjacent<'a>(
    entries: &'a [(&'a ExpertLayout, PathBuf, u64)],
    maximum: usize,
) -> Vec<&'a [(&'a ExpertLayout, PathBuf, u64)]> {
    let mut groups = Vec::new();
    let mut begin = 0;
    while begin < entries.len() {
        let mut end = begin + 1;
        while end < entries.len()
            && end - begin < maximum
            && entries[end - 1].0.shard == entries[end].0.shard
            && entries[end - 1].2 + K3_EXPERT_SOURCE_BYTES as u64 == entries[end].2
        {
            end += 1;
        }
        groups.push(&entries[begin..end]);
        begin = end;
    }
    groups
}

fn execute_guarded_with<T, F, D>(
    plan: &WeightFetchPlan,
    sink: &dyn ProgressSink,
    factory: F,
    available: D,
    platform: HostPlatform,
) -> Result<WeightFetchProgress>
where
    T: Transport,
    F: Fn() -> T + Sync,
    D: Fn(&Path) -> Result<u64>,
{
    inspect_capacity_with(plan, available)?.require()?;
    prepare_download_roots(plan, platform)?;
    execute_transfers(plan, sink, factory)
}

fn prepare_download_roots(plan: &WeightFetchPlan, platform: HostPlatform) -> Result<()> {
    // Do not let recursive child creation recreate a model root which was
    // replaced after planning/capacity admission.
    require_real_directory(&plan.model_root)?;
    if plan.selection != WeightSelection::ExpertPool {
        ensure_real_directory_tree(&plan.resident_directory)?;
        ensure_spotlight_marker(&plan.resident_directory, platform)?;
    }
    if plan.selection != WeightSelection::ResidentSpine {
        ensure_real_directory_tree(&plan.expert_directory)?;
        ensure_spotlight_marker(&plan.expert_directory, platform)?;
    }
    Ok(())
}

fn execute_transfers<T, F>(
    plan: &WeightFetchPlan,
    sink: &dyn ProgressSink,
    factory: F,
) -> Result<WeightFetchProgress>
where
    T: Transport,
    F: Fn() -> T + Sync,
{
    let mut work = Vec::with_capacity(plan.resident.len() + plan.expert_runs.len());
    work.extend(plan.resident.iter().cloned().map(WorkItem::Resident));
    work.extend(plan.expert_runs.iter().cloned().map(WorkItem::Experts));
    let state = ExecutionState::new(plan);
    sink.update(state.snapshot());
    let cursor = AtomicUsize::new(0);
    let stopped = AtomicBool::new(false);
    let failure = Mutex::new(None::<DeltafinError>);
    thread::scope(|scope| {
        for _ in 0..plan.limits.workers.min(work.len().max(1)) {
            let factory = &factory;
            let work = &work;
            let state = &state;
            let cursor = &cursor;
            let stopped = &stopped;
            let failure = &failure;
            scope.spawn(move || {
                let mut transport = factory();
                loop {
                    if stopped.load(Ordering::Acquire) {
                        break;
                    }
                    let index = cursor.fetch_add(1, Ordering::AcqRel);
                    let Some(item) = work.get(index) else { break };
                    let result = match item {
                        WorkItem::Resident(item) => {
                            install_resident(item, &plan.base_url, &mut transport, state, sink)
                        }
                        WorkItem::Experts(item) => {
                            install_expert_run(item, &plan.base_url, &mut transport, state, sink)
                        }
                    };
                    if let Err(error) = result {
                        stopped.store(true, Ordering::Release);
                        let mut slot = failure.lock().unwrap_or_else(|poison| poison.into_inner());
                        if slot.is_none() {
                            *slot = Some(error);
                        }
                        break;
                    }
                }
            });
        }
    });
    if let Some(error) = failure
        .into_inner()
        .unwrap_or_else(|poison| poison.into_inner())
    {
        return Err(error);
    }
    let complete = state.snapshot();
    sink.update(complete);
    Ok(complete)
}

struct ExecutionState {
    files_completed: AtomicUsize,
    files_reused: AtomicUsize,
    requests_completed: AtomicUsize,
    bytes_transferred: AtomicU64,
    total_files: usize,
    total_bytes: u64,
}

impl ExecutionState {
    fn new(plan: &WeightFetchPlan) -> Self {
        Self {
            files_completed: AtomicUsize::new(0),
            files_reused: AtomicUsize::new(
                plan.dry_run.resident_files_reused + plan.dry_run.expert_files_reused,
            ),
            requests_completed: AtomicUsize::new(0),
            bytes_transferred: AtomicU64::new(0),
            total_files: plan.dry_run.resident_files_missing
                + plan.dry_run.resident_files_reused
                + plan.dry_run.expert_files_missing
                + plan.dry_run.expert_files_reused,
            total_bytes: plan.dry_run.bytes_missing + plan.dry_run.bytes_reused,
        }
    }

    fn snapshot(&self) -> WeightFetchProgress {
        WeightFetchProgress {
            files_completed: self.files_completed.load(Ordering::Acquire),
            files_reused: self.files_reused.load(Ordering::Acquire),
            requests_completed: self.requests_completed.load(Ordering::Acquire),
            bytes_transferred: self.bytes_transferred.load(Ordering::Acquire),
            total_files: self.total_files,
            total_bytes: self.total_bytes,
        }
    }

    fn request(&self, bytes: u64, sink: &dyn ProgressSink) {
        self.requests_completed.fetch_add(1, Ordering::AcqRel);
        self.bytes_transferred.fetch_add(bytes, Ordering::AcqRel);
        sink.update(self.snapshot());
    }

    fn published(&self, sink: &dyn ProgressSink) {
        self.files_completed.fetch_add(1, Ordering::AcqRel);
        sink.update(self.snapshot());
    }

    fn reused(&self, sink: &dyn ProgressSink) {
        self.files_reused.fetch_add(1, Ordering::AcqRel);
        sink.update(self.snapshot());
    }
}

fn install_resident(
    item: &ResidentRangePlan,
    base: &str,
    transport: &mut dyn Transport,
    state: &ExecutionState,
    sink: &dyn ProgressSink,
) -> Result<()> {
    if exact_regular_file(&item.destination, item.size())? {
        state.reused(sink);
        return Ok(());
    }
    let directory = item.destination.parent().unwrap();
    let part = directory.join(format!(".{}.part", item.tensor_name));
    download_resumable(
        &k3_source::shard_url(base, shard_number(&item.shard)?)?,
        item.source_start,
        item.source_end,
        item.shard_bytes,
        &part,
        transport,
        state,
        sink,
    )?;
    publish_or_accept_race(&part, &item.destination, directory, item.size())?;
    state.published(sink);
    Ok(())
}

fn install_expert_run(
    run: &ExpertRunPlan,
    base: &str,
    transport: &mut dyn Transport,
    state: &ExecutionState,
    sink: &dyn ProgressSink,
) -> Result<()> {
    let directory = run.part.parent().unwrap();
    let mut all_present = true;
    for expert in &run.experts {
        if !exact_regular_file(&expert.destination, K3_EXPERT_SOURCE_BYTES as u64)? {
            all_present = false;
        }
    }
    if all_present {
        for _ in &run.experts {
            state.reused(sink);
        }
        return Ok(());
    }
    download_resumable(
        &k3_source::shard_url(base, shard_number(&run.shard)?)?,
        run.source_start,
        run.source_end,
        run.shard_bytes,
        &run.part,
        transport,
        state,
        sink,
    )?;
    split_expert_run(run, K3_EXPERT_SOURCE_BYTES as u64, directory, state, sink)
}

fn split_expert_run(
    run: &ExpertRunPlan,
    expert_bytes: u64,
    directory: &Path,
    state: &ExecutionState,
    sink: &dyn ProgressSink,
) -> Result<()> {
    let mut source = open_exact_regular(&run.part, run.size())?;
    for expert in &run.experts {
        if exact_regular_file(&expert.destination, expert_bytes)? {
            state.reused(sink);
            continue;
        }
        let offset = expert.source_start - run.source_start;
        source
            .seek(SeekFrom::Start(offset))
            .map_err(|error| io_error("seek validated expert run", &run.part, error))?;
        let temporary = unique_part(directory, &format!("L{}-E{}", expert.layer, expert.expert))?;
        let mut target = secure_create_new(&temporary, 0o600)?;
        let copied = std::io::copy(
            &mut Read::by_ref(&mut source).take(expert_bytes),
            &mut target,
        )
        .map_err(|error| io_error("split validated expert run", &temporary, error));
        let result = (|| {
            if copied? != expert_bytes {
                return Err(DeltafinError::new(
                    "validated expert run ended while splitting",
                ));
            }
            target
                .sync_all()
                .map_err(|error| io_error("fsync split expert", &temporary, error))?;
            publish_or_accept_race(&temporary, &expert.destination, directory, expert_bytes)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result?;
        state.published(sink);
    }
    fs::remove_file(&run.part)
        .map_err(|error| io_error("remove completely split expert run", &run.part, error))?;
    fsync_directory(directory)
}

#[allow(clippy::too_many_arguments)]
fn download_resumable(
    url: &str,
    source_start: u64,
    source_end: u64,
    source_total: u64,
    part: &Path,
    transport: &mut dyn Transport,
    state: &ExecutionState,
    sink: &dyn ProgressSink,
) -> Result<()> {
    let expected = source_end - source_start + 1;
    let mut output = open_resumable(part, expected)?;
    let mut committed = output
        .metadata()
        .map_err(|error| io_error("stat resumable weight partial", part, error))?
        .len();
    if committed != expected && committed % TRANSFER_CHUNK_BYTES != 0 {
        committed -= committed % TRANSFER_CHUNK_BYTES;
        output
            .set_len(committed)
            .map_err(|error| io_error("truncate interrupted weight chunk", part, error))?;
    }
    while committed < expected {
        let count = (expected - committed).min(TRANSFER_CHUNK_BYTES);
        let start = source_start + committed;
        let end = start + count - 1;
        let directory = part.parent().unwrap();
        let chunk_path = unique_part(directory, "range")?;
        let mut chunk = secure_create_new(&chunk_path, 0o600)?;
        let transfer = transport.transfer(
            &Request {
                url: url.to_owned(),
                range: Some(ByteRange {
                    start,
                    end: Some(end),
                }),
                user_agent: USER_AGENT,
                timeout: TimeoutPolicy::LargePayload,
            },
            &mut chunk,
            count,
        );
        let result = (|| {
            let meta = transfer?;
            validate_range_response(&meta, start, end, source_total)?;
            chunk
                .sync_all()
                .map_err(|error| io_error("fsync validated range chunk", &chunk_path, error))?;
            let mut reader = open_exact_regular(&chunk_path, count)?;
            let copied = std::io::copy(&mut reader, &mut output)
                .map_err(|error| io_error("append validated range chunk", part, error))?;
            if copied != count {
                return Err(DeltafinError::new(
                    "validated range chunk was short while appending",
                ));
            }
            output
                .sync_all()
                .map_err(|error| io_error("fsync resumable weight partial", part, error))?;
            Ok(())
        })();
        let _ = fs::remove_file(&chunk_path);
        result?;
        committed += count;
        state.request(count, sink);
    }
    let metadata = output
        .metadata()
        .map_err(|error| io_error("restat completed weight partial", part, error))?;
    if metadata.len() != expected {
        return Err(DeltafinError::new(format!(
            "completed weight partial has {} bytes; expected {expected}",
            metadata.len()
        )));
    }
    Ok(())
}

fn validate_range_response(meta: &ResponseMeta, start: u64, end: u64, total: u64) -> Result<()> {
    let expected = end - start + 1;
    if meta.status != 206 || meta.bytes != expected {
        return Err(DeltafinError::new(format!(
            "weight range returned status {} and {} bytes; expected 206 and {expected}",
            meta.status, meta.bytes
        )));
    }
    let content_length = meta
        .headers
        .get("content-length")
        .ok_or_else(|| DeltafinError::new("weight range lacks Content-Length"))?;
    if parse_decimal(content_length, "Content-Length")? != expected {
        return Err(DeltafinError::new(
            "weight range Content-Length is not exact",
        ));
    }
    let content_range = meta
        .headers
        .get("content-range")
        .ok_or_else(|| DeltafinError::new("weight range lacks Content-Range"))?;
    if content_range != &format!("bytes {start}-{end}/{total}") {
        return Err(DeltafinError::new(format!(
            "weight range Content-Range {content_range:?} is not exact"
        )));
    }
    Ok(())
}

fn source_range(record: &TensorRecord, offsets: [u64; 2]) -> Result<(u64, u64)> {
    let start = 8_u64
        .checked_add(record.hlen)
        .and_then(|value| value.checked_add(offsets[0]))
        .ok_or_else(|| DeltafinError::new("tensor source range overflowed"))?;
    let end = 8_u64
        .checked_add(record.hlen)
        .and_then(|value| value.checked_add(offsets[1]))
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| DeltafinError::new("tensor source range is empty or overflowed"))?;
    Ok((start, end))
}

fn is_expert_record(name: &str) -> bool {
    name.contains(".block_sparse_moe.experts.")
}

fn exact_regular_file(path: &Path, expected: u64) -> Result<bool> {
    match sys_fs::lstat(path) {
        Ok(before) => {
            if before.file_type().is_symlink() || !before.is_file() {
                return Err(DeltafinError::new(format!(
                    "refusing non-regular or symlink weight path {}",
                    path.display()
                )));
            }
            if before.len() != expected {
                return Err(DeltafinError::new(format!(
                    "refusing wrong-size weight path {}: got {}, expected {expected}",
                    path.display(),
                    before.len()
                )));
            }
            let file = Open::new()
                .read(true)
                .no_follow()
                .open(path)
                .map_err(|error| io_error("open existing weight without symlinks", path, error))?;
            let opened = sys_fs::fstat(&file)
                .map_err(|error| io_error("stat opened existing weight", path, error))?;
            if !opened.is_file()
                || (opened.dev(), opened.ino(), opened.len())
                    != (before.dev(), before.ino(), expected)
            {
                return Err(DeltafinError::new(format!(
                    "existing weight changed while opening: {}",
                    path.display()
                )));
            }
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io_error("inspect weight path", path, error)),
    }
}

fn open_resumable(path: &Path, expected: u64) -> Result<File> {
    match sys_fs::lstat(path) {
        Ok(before) => {
            if before.file_type().is_symlink() || !before.is_file() || before.len() > expected {
                return Err(DeltafinError::new(format!(
                    "unsafe resumable weight partial {}",
                    path.display()
                )));
            }
            let file = Open::new()
                .append(true)
                .no_follow()
                .open(path)
                .map_err(|error| io_error("open resumable weight partial", path, error))?;
            let opened = sys_fs::fstat(&file)
                .map_err(|error| io_error("stat opened weight partial", path, error))?;
            if !opened.is_file()
                || (opened.dev(), opened.ino(), opened.len())
                    != (before.dev(), before.ino(), before.len())
            {
                return Err(DeltafinError::new(format!(
                    "resumable weight partial changed while opening: {}",
                    path.display()
                )));
            }
            Ok(file)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            secure_create_new(path, 0o600)
        }
        Err(error) => Err(io_error("inspect resumable weight partial", path, error)),
    }
}

fn open_exact_regular(path: &Path, expected: u64) -> Result<File> {
    let before = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect validated weight file", path, error))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() != expected {
        return Err(DeltafinError::new(format!(
            "validated weight file is not an exact regular file: {}",
            path.display()
        )));
    }
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open validated weight file", path, error))?;
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat validated weight file", path, error))?;
    if (opened.dev(), opened.ino(), opened.len()) != (before.dev(), before.ino(), expected) {
        return Err(DeltafinError::new(format!(
            "validated weight file changed while opening: {}",
            path.display()
        )));
    }
    Ok(file)
}

fn publish_or_accept_race(
    part: &Path,
    destination: &Path,
    directory: &Path,
    expected: u64,
) -> Result<()> {
    match publish_hard_link(part, destination, directory) {
        Ok(()) => Ok(()),
        Err(publication) => {
            if exact_regular_file(destination, expected)? {
                fs::remove_file(part).map_err(|error| {
                    io_error("remove concurrently published partial", part, error)
                })?;
                fsync_directory(directory)
            } else {
                Err(publication)
            }
        }
    }
}

fn inspect_optional_directory(path: &Path) -> Result<()> {
    match sys_fs::lstat(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(DeltafinError::new(format!(
            "weight destination is not a real directory: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error("inspect weight destination", path, error)),
    }
}

fn require_real_directory(path: &Path) -> Result<()> {
    let metadata =
        sys_fs::lstat(path).map_err(|error| io_error("inspect model root", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DeltafinError::new(format!(
            "model root is not a real directory: {}",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_real_directory_tree(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && sys_fs::lstat(parent).is_err()
    {
        ensure_real_directory_tree(parent)?;
    }
    match sys_fs::lstat(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(DeltafinError::new(format!(
            "refusing unsafe weight directory {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(path)
                .map_err(|error| io_error("create weight directory", path, error))?;
            if let Some(parent) = path.parent() {
                fsync_directory(parent)?;
            }
            Ok(())
        }
        Err(error) => Err(io_error("inspect weight directory", path, error)),
    }
}

/// Return user-available bytes on the exact filesystem containing the model
/// root. `f_bavail` is intentionally used instead of `f_bfree`, so quotas and
/// blocks reserved from the current user cannot make the gate optimistic.
fn available_disk_bytes(root: &Path) -> Result<u64> {
    let before = sys_fs::lstat(root)
        .map_err(|error| io_error("inspect model root for capacity check", root, error))?;
    if before.file_type().is_symlink() || !before.is_dir() {
        return Err(DeltafinError::new(format!(
            "capacity target is not a real directory: {}",
            root.display()
        )));
    }
    let directory = Open::new()
        .read(true)
        .directory().no_follow()
        .open(root)
        .map_err(|error| io_error("open model root for capacity check", root, error))?;
    let opened = sys_fs::fstat(&directory)
        .map_err(|error| io_error("stat opened capacity target", root, error))?;
    if !opened.is_dir() || (opened.dev(), opened.ino()) != (before.dev(), before.ino()) {
        return Err(DeltafinError::new(format!(
            "model root changed during capacity check: {}",
            root.display()
        )));
    }
    sys_fs::available_space(&directory, root)
        .map_err(|error| io_error("query model-volume capacity", root, error))
}

/// Create or authenticate a directory-local Spotlight exclusion marker. The
/// operation is deliberately a no-op off macOS and never changes volume-wide
/// indexing policy.
fn ensure_spotlight_marker(directory: &Path, platform: HostPlatform) -> Result<()> {
    if platform != HostPlatform::MacOs {
        return Ok(());
    }
    let marker = directory.join(SPOTLIGHT_MARKER);
    match sys_fs::lstat(&marker) {
        Ok(before) => {
            if before.file_type().is_symlink() || !before.is_file() {
                return Err(DeltafinError::new(format!(
                    "unsafe Spotlight exclusion marker {}",
                    marker.display()
                )));
            }
            let file = Open::new()
                .read(true)
                .no_follow()
                .open(&marker)
                .map_err(|error| {
                    io_error(
                        "open Spotlight marker without following links",
                        &marker,
                        error,
                    )
                })?;
            let opened = sys_fs::fstat(&file)
                .map_err(|error| io_error("stat opened Spotlight marker", &marker, error))?;
            if !opened.is_file() || (opened.dev(), opened.ino()) != (before.dev(), before.ino()) {
                return Err(DeltafinError::new(format!(
                    "Spotlight exclusion marker changed while opening: {}",
                    marker.display()
                )));
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            secure_create_new(&marker, 0o600)?
                .sync_all()
                .map_err(|error| io_error("fsync Spotlight exclusion marker", &marker, error))?;
            fsync_directory(directory)
        }
        Err(error) => Err(io_error(
            "inspect Spotlight exclusion marker",
            &marker,
            error,
        )),
    }
}

/// Same marker, for callers outside this module that have no reason to know
/// about [`HostPlatform`]: gate on the actual current host automatically.
/// `directory` need not be a weight subdirectory — placing this at the top of
/// `model_root` excludes the whole install tree in one call, including
/// anything a caller adds later that this module was never told about.
pub(crate) fn ensure_spotlight_marker_here(directory: &Path) -> Result<()> {
    ensure_spotlight_marker(directory, HostPlatform::current())
}

fn unique_part(directory: &Path, label: &str) -> Result<PathBuf> {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    for _ in 0..128 {
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(".{label}.{}.{sequence}.part", std::process::id()));
        if sys_fs::lstat(&path).is_err() {
            return Ok(path);
        }
    }
    Err(DeltafinError::new(
        "could not allocate a unique weight partial name",
    ))
}

fn shard_number(name: &str) -> Result<usize> {
    let digits = name
        .strip_prefix("model-")
        .and_then(|rest| rest.strip_suffix("-of-000096.safetensors"))
        .ok_or_else(|| DeltafinError::new(format!("invalid inventory shard {name:?}")))?;
    digits
        .parse::<usize>()
        .map_err(|_| DeltafinError::new(format!("invalid inventory shard {name:?}")))
}

fn parse_decimal(value: &str, label: &str) -> Result<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(DeltafinError::new(format!("non-decimal {label}")));
    }
    value
        .parse()
        .map_err(|_| DeltafinError::new(format!("{label} overflows u64")))
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| DeltafinError::new(format!("{label} overflowed")))
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "deltafin-weight-fetch-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[derive(Clone)]
    struct FakeReply {
        status: u16,
        body: Vec<u8>,
        content_range: String,
        content_length: String,
    }

    #[derive(Clone)]
    struct FakeTransport {
        replies: std::sync::Arc<Mutex<VecDeque<FakeReply>>>,
        requests: std::sync::Arc<Mutex<Vec<Request>>>,
    }

    impl Transport for FakeTransport {
        fn transfer(
            &mut self,
            request: &Request,
            target: &mut dyn std::io::Write,
            maximum: u64,
        ) -> Result<ResponseMeta> {
            self.requests.lock().unwrap().push(request.clone());
            let reply = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| DeltafinError::new("no fake response"))?;
            if reply.body.len() as u64 > maximum {
                return Err(DeltafinError::new("fake body exceeds maximum"));
            }
            target
                .write_all(&reply.body)
                .map_err(|error| DeltafinError::new(error.to_string()))?;
            Ok(ResponseMeta {
                status: reply.status,
                headers: BTreeMap::from([
                    ("content-range".into(), reply.content_range),
                    ("content-length".into(), reply.content_length),
                ]),
                bytes: reply.body.len() as u64,
            })
        }
    }

    fn fake(body: Vec<u8>, start: u64, total: u64) -> FakeReply {
        let end = start + body.len() as u64 - 1;
        FakeReply {
            status: 206,
            content_range: format!("bytes {start}-{end}/{total}"),
            content_length: body.len().to_string(),
            body,
        }
    }

    fn state(bytes: u64) -> ExecutionState {
        ExecutionState {
            files_completed: AtomicUsize::new(0),
            files_reused: AtomicUsize::new(0),
            requests_completed: AtomicUsize::new(0),
            bytes_transferred: AtomicU64::new(0),
            total_files: 1,
            total_bytes: bytes,
        }
    }

    fn tiny_resident_plan(root: &Path) -> WeightFetchPlan {
        let document = InventoryDocument::from([(
            "language_model.model.norm.weight".into(),
            TensorRecord {
                dtype: "U8".into(),
                shape: vec![3],
                offsets: [0, 3],
                shard: "model-00001-of-000096.safetensors".into(),
                hlen: 8,
            },
        )]);
        plan_from_document(
            root,
            &document,
            WeightSelection::ResidentSpine,
            FetchLimits {
                workers: 1,
                experts_per_run: 1,
            },
            "https://example.invalid/k3/".into(),
        )
        .unwrap()
    }

    #[test]
    fn capacity_arithmetic_accepts_the_exact_boundary_and_fails_closed() {
        let storage = WeightFetchStorage {
            peak_additional_bytes: 23,
            ..WeightFetchStorage::default()
        };
        let required = DEFAULT_REMAINING_FREE_BYTES + 23;
        let exact = capacity_from_available(17, storage, required).unwrap();
        assert!(exact.has_capacity());
        assert_eq!(exact.required_available_bytes, required);
        assert_eq!(exact.shortfall_bytes, 0);

        let short = capacity_from_available(17, storage, required - 1).unwrap();
        assert!(!short.has_capacity());
        assert_eq!(short.shortfall_bytes, 1);
        assert!(short.require().is_err());

        let complete = capacity_from_available(0, WeightFetchStorage::default(), 0).unwrap();
        assert!(complete.has_capacity());
        assert_eq!(complete.remaining_free_bytes, 0);
        assert_eq!(complete.required_available_bytes, 0);
        assert!(
            capacity_from_available(
                0,
                WeightFetchStorage {
                    peak_additional_bytes: 1,
                    ..WeightFetchStorage::default()
                },
                u64::MAX,
            )
            .is_err()
        );
        assert!(
            capacity_from_available(
                1,
                WeightFetchStorage {
                    peak_additional_bytes: u64::MAX,
                    ..WeightFetchStorage::default()
                },
                u64::MAX,
            )
            .is_err()
        );
    }

    #[test]
    fn insufficient_capacity_refuses_before_transport_or_destination_creation() {
        let root = TestRoot::new();
        let plan = tiny_resident_plan(&root.0);
        let required = plan.storage_requirement().unwrap().peak_additional_bytes
            + DEFAULT_REMAINING_FREE_BYTES;
        let factories = AtomicUsize::new(0);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let result = execute_guarded_with(
            &plan,
            &|_| {},
            || {
                factories.fetch_add(1, Ordering::Relaxed);
                FakeTransport {
                    replies: Arc::new(Mutex::new(VecDeque::new())),
                    requests: requests.clone(),
                }
            },
            |_| Ok(required - 1),
            HostPlatform::Other,
        );
        let error = result.unwrap_err().to_string();
        assert!(error.contains("No network transfer was started"));
        assert_eq!(factories.load(Ordering::Relaxed), 0);
        assert!(requests.lock().unwrap().is_empty());
        assert!(!root.0.join("k3-resident").exists());
    }

    #[test]
    fn spotlight_marker_is_macos_only_idempotent_and_rejects_symlinks() {
        use crate::sys::fs::symlink;

        let root = TestRoot::new();
        let directory = root.0.join("weights");
        fs::create_dir(&directory).unwrap();
        let marker = directory.join(SPOTLIGHT_MARKER);

        ensure_spotlight_marker(&directory, HostPlatform::Other).unwrap();
        assert!(!marker.exists());

        ensure_spotlight_marker(&directory, HostPlatform::MacOs).unwrap();
        let metadata = fs::symlink_metadata(&marker).unwrap();
        assert!(metadata.is_file());
        assert!(!metadata.file_type().is_symlink());
        ensure_spotlight_marker(&directory, HostPlatform::MacOs).unwrap();

        fs::remove_file(&marker).unwrap();
        symlink(&root.0, &marker).unwrap();
        assert!(ensure_spotlight_marker(&directory, HostPlatform::MacOs).is_err());
        // Non-macOS is a genuine no-op and does not inspect or follow a marker.
        ensure_spotlight_marker(&directory, HostPlatform::Other).unwrap();
    }

    /// Unlike the test above, which only ever passes an explicit
    /// `HostPlatform`, this exercises the actual `HostPlatform::current()`
    /// resolution the install-time caller depends on -- so it asserts
    /// whichever behavior is correct for the host actually running the test,
    /// not a fixed expectation.
    #[test]
    fn ensure_spotlight_marker_here_matches_the_current_host() {
        let root = TestRoot::new();
        let directory = root.0.join("model-root");
        fs::create_dir(&directory).unwrap();
        let marker = directory.join(SPOTLIGHT_MARKER);

        ensure_spotlight_marker_here(&directory).unwrap();
        if cfg!(target_os = "macos") {
            let metadata = fs::symlink_metadata(&marker).unwrap();
            assert!(metadata.is_file());
            assert!(!metadata.file_type().is_symlink());
        } else {
            assert!(!marker.exists());
        }
        ensure_spotlight_marker_here(&directory).unwrap();
    }

    #[test]
    fn full_fetch_preflight_marks_both_download_roots_on_macos() {
        let root = TestRoot::new();
        let plan = WeightFetchPlan {
            selection: WeightSelection::Full,
            resident: Vec::new(),
            expert_runs: Vec::new(),
            dry_run: WeightFetchDryRun::default(),
            limits: FetchLimits {
                workers: 1,
                experts_per_run: 1,
            },
            base_url: "https://example.invalid/k3/".into(),
            model_root: root.0.clone(),
            resident_directory: root.0.join("k3-resident/tensors"),
            expert_directory: root.0.join("k3-experts"),
        };
        prepare_download_roots(&plan, HostPlatform::MacOs).unwrap();
        for directory in [&plan.resident_directory, &plan.expert_directory] {
            let marker = fs::symlink_metadata(directory.join(SPOTLIGHT_MARKER)).unwrap();
            assert!(marker.is_file());
            assert!(!marker.file_type().is_symlink());
        }
    }

    #[test]
    fn native_capacity_probe_uses_a_real_directory_descriptor() {
        use crate::sys::fs::symlink;

        let root = TestRoot::new();
        let _available = available_disk_bytes(&root.0).unwrap();
        let link = root.0.with_extension("link");
        symlink(&root.0, &link).unwrap();
        assert!(available_disk_bytes(&link).is_err());
        fs::remove_file(link).unwrap();
    }

    #[test]
    fn exact_range_validation_rejects_status_range_and_length_mismatches() {
        let response = || ResponseMeta {
            status: 206,
            headers: BTreeMap::from([
                ("content-range".into(), "bytes 10-12/20".into()),
                ("content-length".into(), "3".into()),
            ]),
            bytes: 3,
        };
        let good = response();
        validate_range_response(&good, 10, 12, 20).unwrap();
        let mut bad = response();
        bad.status = 200;
        assert!(validate_range_response(&bad, 10, 12, 20).is_err());
        let mut bad = response();
        bad.bytes = 2;
        assert!(validate_range_response(&bad, 10, 12, 20).is_err());
        let mut bad = response();
        bad.headers
            .insert("content-range".into(), "bytes 9-12/20".into());
        assert!(validate_range_response(&bad, 10, 12, 20).is_err());
        let mut bad = response();
        bad.headers.insert("content-length".into(), "4".into());
        assert!(validate_range_response(&bad, 10, 12, 20).is_err());
    }

    #[test]
    fn resumable_download_uses_existing_validated_chunk_prefix() {
        let root = TestRoot::new();
        let part = root.0.join("payload.part");
        File::create(&part)
            .unwrap()
            .set_len(TRANSFER_CHUNK_BYTES)
            .unwrap();
        let tail = vec![9_u8; 11];
        let start = 100 + TRANSFER_CHUNK_BYTES;
        let replies = std::sync::Arc::new(Mutex::new(VecDeque::from([fake(
            tail.clone(),
            start,
            start + 11,
        )])));
        let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
        let mut transport = FakeTransport {
            replies,
            requests: requests.clone(),
        };
        download_resumable(
            "https://example.invalid/shard",
            100,
            start + 10,
            start + 11,
            &part,
            &mut transport,
            &state(TRANSFER_CHUNK_BYTES + 11),
            &|_| {},
        )
        .unwrap();
        let request = &requests.lock().unwrap()[0];
        assert_eq!(request.range.as_ref().unwrap().start, start);
        assert_eq!(fs::metadata(part).unwrap().len(), TRANSFER_CHUNK_BYTES + 11);
    }

    #[test]
    fn bad_response_never_advances_or_publishes_the_resumable_partial() {
        let root = TestRoot::new();
        let part = root.0.join("payload.part");
        let mut reply = fake(vec![1, 2, 3], 10, 20);
        reply.status = 200;
        let mut transport = FakeTransport {
            replies: std::sync::Arc::new(Mutex::new(VecDeque::from([reply]))),
            requests: std::sync::Arc::new(Mutex::new(Vec::new())),
        };
        assert!(
            download_resumable(
                "https://example.invalid/shard",
                10,
                12,
                20,
                &part,
                &mut transport,
                &state(3),
                &|_| {},
            )
            .is_err()
        );
        assert_eq!(fs::metadata(part).unwrap().len(), 0);
    }

    #[test]
    fn resident_plan_executes_exact_fake_range_and_publishes_locally() {
        let root = TestRoot::new();
        let name = "language_model.model.norm.weight";
        let document = InventoryDocument::from([(
            name.into(),
            TensorRecord {
                dtype: "U8".into(),
                shape: vec![3],
                offsets: [0, 3],
                shard: "model-00001-of-000096.safetensors".into(),
                hlen: 8,
            },
        )]);
        let plan = plan_from_document(
            &root.0,
            &document,
            WeightSelection::ResidentSpine,
            FetchLimits {
                workers: 1,
                experts_per_run: 1,
            },
            "https://example.invalid/k3/".into(),
        )
        .unwrap();
        assert_eq!(plan.dry_run.resident_files_missing, 1);
        assert_eq!(plan.resident[0].source_start, 16);
        assert_eq!(plan.resident[0].source_end, 18);

        let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
        let transport = FakeTransport {
            replies: std::sync::Arc::new(Mutex::new(VecDeque::from([fake(vec![4, 5, 6], 16, 19)]))),
            requests: requests.clone(),
        };
        let progress = execute_guarded_with(
            &plan,
            &|_| {},
            || transport.clone(),
            |_| Ok(u64::MAX),
            HostPlatform::Other,
        )
        .unwrap();
        assert_eq!(progress.files_completed, 1);
        assert_eq!(progress.bytes_transferred, 3);
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert_eq!(
            fs::read(root.0.join("k3-resident/tensors").join(name)).unwrap(),
            [4, 5, 6]
        );
    }

    #[test]
    fn expert_six_record_layout_is_derived_from_w1_anchor() {
        let mut document = InventoryDocument::new();
        let prefix = "language_model.model.layers.1.block_sparse_moe.experts.7.";
        let mut offset = 400_u64;
        for component in EXPERT_COMPONENTS {
            let end = offset + component.bytes as u64;
            document.insert(
                format!("{prefix}{}", component.suffix),
                TensorRecord {
                    dtype: "U8".into(),
                    shape: component.shape.to_vec(),
                    offsets: [offset, end],
                    shard: "model-00002-of-000096.safetensors".into(),
                    hlen: 800,
                },
            );
            offset = end;
        }
        let layout = derive_expert(&document, 1, 7).unwrap();
        assert_eq!(layout.data_start, 400);
        document
            .get_mut(&format!("{prefix}w2.weight_scale"))
            .unwrap()
            .offsets[0] += 1;
        assert!(derive_expert(&document, 1, 7).is_err());
    }

    #[test]
    fn complete_validated_run_is_split_then_each_expert_is_published() {
        let root = TestRoot::new();
        let part = root.0.join("run.part");
        fs::write(&part, [1_u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let run = ExpertRunPlan {
            layer: 1,
            shard: "model-00002-of-000096.safetensors".into(),
            source_start: 100,
            source_end: 107,
            shard_bytes: 200,
            experts: vec![
                ExpertSpanPlan {
                    layer: 1,
                    expert: 0,
                    destination: root.0.join("L1-E0.bin"),
                    source_start: 100,
                },
                ExpertSpanPlan {
                    layer: 1,
                    expert: 1,
                    destination: root.0.join("L1-E1.bin"),
                    source_start: 104,
                },
            ],
            part: part.clone(),
        };
        let state = state(8);
        split_expert_run(&run, 4, &root.0, &state, &|_| {}).unwrap();
        assert_eq!(fs::read(root.0.join("L1-E0.bin")).unwrap(), [1, 2, 3, 4]);
        assert_eq!(fs::read(root.0.join("L1-E1.bin")).unwrap(), [5, 6, 7, 8]);
        assert!(!part.exists());
        assert_eq!(state.snapshot().files_completed, 2);
    }

    #[test]
    fn physically_adjacent_missing_experts_group_in_authenticated_order() {
        let layouts = [
            ExpertLayout {
                layer: 1,
                expert: 0,
                shard: "s".into(),
                hlen: 8,
                data_start: 0,
            },
            ExpertLayout {
                layer: 1,
                expert: 10,
                shard: "s".into(),
                hlen: 8,
                data_start: K3_EXPERT_SOURCE_BYTES as u64,
            },
            ExpertLayout {
                layer: 1,
                expert: 2,
                shard: "s".into(),
                hlen: 8,
                data_start: 3 * K3_EXPERT_SOURCE_BYTES as u64,
            },
        ];
        let entries = vec![
            (&layouts[0], PathBuf::from("a"), 100),
            (
                &layouts[1],
                PathBuf::from("b"),
                100 + K3_EXPERT_SOURCE_BYTES as u64,
            ),
            (
                &layouts[2],
                PathBuf::from("c"),
                100 + 3 * K3_EXPERT_SOURCE_BYTES as u64,
            ),
        ];
        let groups = group_adjacent(&entries, 8);
        assert_eq!(
            groups.iter().map(|group| group.len()).collect::<Vec<_>>(),
            [2, 1]
        );
        assert_eq!(groups[0][1].0.expert, 10);
    }

    fn synthetic_expert_catalog(root: &Path) -> ExpertFetchCatalog {
        let mut layouts = Vec::with_capacity(K3_MOE_LAYER_LAST as usize * K3_EXPERTS_PER_LAYER);
        let mut shard_bytes = BTreeMap::new();
        for layer in K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST {
            let shard = format!("model-{layer:05}-of-000096.safetensors");
            let hlen = 800_u64;
            for expert in 0..K3_EXPERTS_PER_LAYER {
                layouts.push(ExpertLayout {
                    layer,
                    expert: expert as u16,
                    shard: shard.clone(),
                    hlen,
                    data_start: expert as u64 * K3_EXPERT_SOURCE_BYTES as u64,
                });
            }
            shard_bytes.insert(
                shard,
                8 + hlen + K3_EXPERTS_PER_LAYER as u64 * K3_EXPERT_SOURCE_BYTES as u64,
            );
        }
        ExpertFetchCatalog::from_authenticated_parts(
            root,
            layouts,
            shard_bytes,
            FetchLimits {
                workers: 4,
                experts_per_run: 8,
            },
            "https://example.invalid/k3/".into(),
        )
        .unwrap()
    }

    #[test]
    fn demand_catalog_deduplicates_routes_and_coalesces_only_physical_neighbors() {
        let root = TestRoot::new();
        let catalog = synthetic_expert_catalog(&root.0);
        let plan = catalog.plan_layer(17, &[5, 3, 2, 2]).unwrap();
        assert_eq!(plan.dry_run.expert_files_missing, 3);
        assert_eq!(plan.dry_run.expert_files_reused, 0);
        assert_eq!(plan.expert_runs.len(), 2);
        assert_eq!(
            plan.expert_runs[0]
                .experts
                .iter()
                .map(|entry| entry.expert)
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(plan.expert_runs[1].experts[0].expert, 5);
        assert_eq!(plan.dry_run.transfer_requests, 2);
    }

    #[test]
    fn demand_catalog_rechecks_durable_cache_and_rejects_bad_coordinates() {
        let root = TestRoot::new();
        let cache = root.0.join("k3-experts");
        fs::create_dir(&cache).unwrap();
        File::create(cache.join("L17-E3.bin"))
            .unwrap()
            .set_len(K3_EXPERT_SOURCE_BYTES as u64)
            .unwrap();
        let catalog = synthetic_expert_catalog(&root.0);
        let plan = catalog.plan_layer(17, &[2, 3, 5]).unwrap();
        assert_eq!(plan.dry_run.expert_files_reused, 1);
        assert_eq!(plan.dry_run.expert_files_missing, 2);
        assert_eq!(plan.expert_runs.len(), 2);
        assert!(catalog.plan_layer(0, &[1]).is_err());
        assert!(catalog.plan_layer(93, &[1]).is_err());
        assert!(catalog.plan_layer(17, &[896]).is_err());
    }

    #[test]
    fn storage_requirement_credits_staged_bytes_but_keeps_split_and_chunk_peak() {
        let root = TestRoot::new();
        let resident_directory = root.0.join("k3-resident/tensors");
        let expert_directory = root.0.join("k3-experts");
        fs::create_dir_all(&resident_directory).unwrap();
        fs::create_dir(&expert_directory).unwrap();
        let resident_destination = resident_directory.join("tensor");
        File::create(resident_directory.join(".tensor.part"))
            .unwrap()
            .set_len(40)
            .unwrap();
        let run_part = expert_directory.join("run.part");
        File::create(&run_part).unwrap().set_len(50).unwrap();
        let expert_bytes = K3_EXPERT_SOURCE_BYTES as u64;
        let plan = WeightFetchPlan {
            selection: WeightSelection::Full,
            resident: vec![ResidentRangePlan {
                tensor_name: "tensor".into(),
                destination: resident_destination,
                shard: "model-00001-of-000096.safetensors".into(),
                source_start: 0,
                source_end: 99,
                shard_bytes: 100,
            }],
            expert_runs: vec![ExpertRunPlan {
                layer: 1,
                shard: "model-00002-of-000096.safetensors".into(),
                source_start: 100,
                source_end: 100 + expert_bytes - 1,
                shard_bytes: 100 + expert_bytes,
                experts: vec![ExpertSpanPlan {
                    layer: 1,
                    expert: 7,
                    destination: expert_directory.join("L1-E7.bin"),
                    source_start: 100,
                }],
                part: run_part,
            }],
            dry_run: WeightFetchDryRun {
                resident_files_missing: 1,
                expert_files_missing: 1,
                bytes_missing: 100 + expert_bytes,
                ..WeightFetchDryRun::default()
            },
            limits: FetchLimits {
                workers: 1,
                experts_per_run: 8,
            },
            base_url: "https://example.invalid/".into(),
            model_root: root.0.clone(),
            resident_directory,
            expert_directory,
        };
        let storage = plan.storage_requirement().unwrap();
        assert_eq!(storage.staged_bytes, 90);
        assert_eq!(storage.final_bytes_to_allocate, 60 + expert_bytes);
        assert_eq!(storage.expert_run_temporary_bytes, expert_bytes - 50);
        // Both tiny unaligned partials are discarded before their next range,
        // so the next chunk may span each complete object. With one worker the
        // larger expert object is the peak chunk.
        assert_eq!(storage.range_chunk_temporary_bytes, expert_bytes);
        assert_eq!(
            storage.peak_additional_bytes,
            60 + expert_bytes + (expert_bytes - 50) + expert_bytes
        );
    }

    #[test]
    fn fresh_capacity_bound_covers_parallel_maximum_runs_and_chunks() {
        let limits = FetchLimits {
            workers: 3,
            experts_per_run: 5,
        };
        assert_eq!(
            maximum_temporary_bytes(limits).unwrap(),
            3 * (5 * K3_EXPERT_SOURCE_BYTES as u64 + TRANSFER_CHUNK_BYTES)
        );
        assert_eq!(
            maximum_resident_temporary_bytes(limits).unwrap(),
            3 * TRANSFER_CHUNK_BYTES
        );
        assert!(
            maximum_temporary_bytes(FetchLimits {
                workers: 0,
                ..limits
            })
            .is_err()
        );
    }

    #[test]
    fn expert_layer_subsets_are_nonempty_unique_and_in_model_bounds() {
        assert_eq!(
            validate_expert_layers(&[1, 45, 92])
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            [1, 45, 92]
        );
        assert!(validate_expert_layers(&[]).is_err());
        assert!(validate_expert_layers(&[0]).is_err());
        assert!(validate_expert_layers(&[93]).is_err());
        assert!(validate_expert_layers(&[1, 1]).is_err());
    }

    #[test]
    fn exact_size_regular_files_are_reused_but_symlinks_fail_closed() {
        use crate::sys::fs::symlink;
        let root = TestRoot::new();
        let file = root.0.join("weight");
        fs::write(&file, [1, 2, 3]).unwrap();
        assert!(exact_regular_file(&file, 3).unwrap());
        let link = root.0.join("link");
        symlink(&file, &link).unwrap();
        assert!(exact_regular_file(&link, 3).is_err());
    }
}
