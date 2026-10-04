//! Full-quality, one-shot native K3 setup orchestration.
//!
//! This module owns no downloader. It composes the pinned transactional
//! metadata, DSpark, optional Qwen, and canonical-weight installers and puts a
//! fail-closed capacity boundary in front of them. Planning is read-only and
//! never contacts the network.

use std::path::{Path, PathBuf};

use crate::dspark_setup;
use crate::error::{DeltafinError, Result};
use crate::experts::{K3_EXPERT_RAW_FILES, K3_EXPERT_SOURCE_BYTES};
use crate::setup_k3;
use crate::setup_qwen;
use crate::sys::fs as sys_fs;
use crate::weight_fetch::{
    self, FetchLimits, MAX_PARALLEL_TRANSFERS, ProgressSink, WeightFetchDryRun,
    WeightFetchProgress, WeightFetchStorage, WeightSelection,
};

/// The safety margin retained by the legacy setup contract. It also bounds
/// temporary range chunks and expert-run splitting by a very wide margin.
pub(crate) const DEFAULT_HEADROOM_BYTES: u64 = 100_000_000_000;

/// Exact canonical expert payload implied by 92 layers x 896 experts.
pub(crate) const EXPERT_WEIGHT_BYTES: u64 =
    K3_EXPERT_RAW_FILES as u64 * K3_EXPERT_SOURCE_BYTES as u64;

/// Exact full payload from the immutable authenticated K3 inventory. The
/// inventory is re-derived and checked against this value once metadata is
/// available, so a changed pin cannot silently alter the capacity contract.
pub(crate) const FULL_WEIGHT_BYTES: u64 = 1_560_860_324_864;
pub(crate) const RESIDENT_WEIGHT_BYTES: u64 = FULL_WEIGHT_BYTES - EXPERT_WEIGHT_BYTES;

/// Requested installation policy. `Auto` resolves to a concrete mode during
/// the read-only capacity plan and is frozen for the subsequent transaction.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum SetupMode {
    Auto,
    Full,
    Stream,
}

impl SetupMode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Full => "full",
            Self::Stream => "streaming",
        }
    }

    fn weight_selection(self) -> Result<WeightSelection> {
        match self {
            Self::Full => Ok(WeightSelection::Full),
            Self::Stream => Ok(WeightSelection::ResidentSpine),
            Self::Auto => Err(DeltafinError::new(
                "automatic setup mode must resolve before weight planning",
            )),
        }
    }

    const fn weight_bytes(self) -> u64 {
        match self {
            Self::Full => FULL_WEIGHT_BYTES,
            Self::Stream => RESIDENT_WEIGHT_BYTES,
            Self::Auto => 0,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct SetupOptions {
    pub root: PathBuf,
    pub workers: usize,
    pub include_qwen: bool,
    pub mode: SetupMode,
}

impl SetupOptions {
    #[cfg(test)]
    pub(crate) fn full(root: PathBuf) -> Self {
        Self {
            root,
            workers: weight_fetch::DEFAULT_PARALLEL_TRANSFERS,
            include_qwen: false,
            mode: SetupMode::Full,
        }
    }

    fn limits(&self) -> Result<FetchLimits> {
        if !(1..=MAX_PARALLEL_TRANSFERS).contains(&self.workers) {
            return Err(DeltafinError::new(format!(
                "one-shot setup workers must be in 1..={MAX_PARALLEL_TRANSFERS}"
            )));
        }
        Ok(FetchLimits {
            workers: self.workers,
            ..FetchLimits::default()
        })
    }
}

/// Read-only description of all remaining work and its capacity boundary.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct SetupPlan {
    /// Concrete mode selected for this immutable plan; never `Auto`.
    pub mode: SetupMode,
    /// Present only when `Auto` fell back from a full installation.
    pub auto_full_shortfall_bytes: Option<u64>,
    pub metadata_bytes_missing: u64,
    pub dspark_bytes_missing: u64,
    pub weight_work: WeightFetchDryRun,
    /// Detailed storage accounting once the pinned inventory is authenticated.
    /// Fresh-root plans have no inventory-derived plan yet and therefore use
    /// `weight_peak_additional_bytes` with this field set to `None`.
    pub weight_storage: Option<WeightFetchStorage>,
    pub weight_peak_additional_bytes: u64,
    pub qwen_bytes_missing: u64,
    pub payload_bytes_missing: u64,
    pub peak_additional_bytes: u64,
    pub headroom_bytes: u64,
    pub required_available_bytes: u64,
    pub available_bytes: u64,
    pub shortfall_bytes: u64,
    pub authenticated_inventory: bool,
    pub includes_qwen: bool,
}

impl SetupPlan {
    pub(crate) fn is_complete(self) -> bool {
        self.payload_bytes_missing == 0
    }

    pub(crate) fn has_capacity(self) -> bool {
        self.shortfall_bytes == 0
    }

    pub(crate) fn require_capacity(self) -> Result<()> {
        if self.has_capacity() {
            return Ok(());
        }
        Err(DeltafinError::new(format!(
            "{} native K3 setup needs {} more bytes on the target volume \
             ({} peak additional bytes to publish {} missing payload bytes, plus {} bytes \
             of headroom; {} bytes available). No further network write was started at this \
             capacity boundary{}",
            self.mode.as_str(),
            self.shortfall_bytes,
            self.peak_additional_bytes,
            self.payload_bytes_missing,
            self.headroom_bytes,
            self.available_bytes,
            if self.mode == SetupMode::Full {
                "; use --stream to keep exact output quality while fetching routed experts on demand"
            } else {
                ""
            },
        )))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct SetupReport {
    pub initial_plan: SetupPlan,
    pub final_plan: SetupPlan,
    pub weight_progress: WeightFetchProgress,
}

#[derive(Debug, Clone, Copy)]
struct Requirements {
    metadata_bytes_missing: u64,
    dspark_bytes_missing: u64,
    weight_work: WeightFetchDryRun,
    weight_storage: Option<WeightFetchStorage>,
    weight_peak_additional_bytes: u64,
    qwen_bytes_missing: u64,
    authenticated_inventory: bool,
}

#[derive(Debug, Clone, Copy)]
struct CommonRequirements {
    metadata_bytes_missing: u64,
    dspark_bytes_missing: u64,
    qwen_bytes_missing: u64,
}

#[derive(Debug, Clone, Copy)]
struct WeightRequirements {
    work: WeightFetchDryRun,
    storage: Option<WeightFetchStorage>,
    peak_additional_bytes: u64,
    authenticated_inventory: bool,
}

/// Build a complete dry plan. This only audits local published files, derives
/// weight reuse from authenticated metadata when available, and reads free
/// space from the target filesystem. It performs no writes and no network I/O.
pub(crate) fn plan(options: &SetupOptions) -> Result<SetupPlan> {
    validate_root(&options.root)?;
    options.limits()?;
    let available = available_disk_bytes(&options.root)?;
    let common = inspect_common_requirements(options)?;
    match options.mode {
        SetupMode::Full | SetupMode::Stream => {
            let weights = inspect_weight_requirements(options, options.mode)?;
            calculate_plan(
                options,
                options.mode,
                combine_requirements(common, weights),
                available,
            )
        }
        SetupMode::Auto => {
            let full_weights = inspect_weight_requirements(options, SetupMode::Full)?;
            let full = calculate_plan(
                options,
                SetupMode::Full,
                combine_requirements(common, full_weights),
                available,
            )?;
            if full.has_capacity() {
                return Ok(full);
            }
            let stream_weights = inspect_weight_requirements(options, SetupMode::Stream)?;
            let mut stream = calculate_plan(
                options,
                SetupMode::Stream,
                combine_requirements(common, stream_weights),
                available,
            )?;
            stream.auto_full_shortfall_bytes = Some(full.shortfall_bytes);
            Ok(stream)
        }
    }
}

/// Audit that the selected one-shot installation is complete. This is the
/// check-only counterpart to `run` and also performs no writes or network I/O.
pub(crate) fn check(options: &SetupOptions) -> Result<SetupPlan> {
    let plan = plan(options)?;
    if plan.is_complete() {
        Ok(plan)
    } else {
        Err(DeltafinError::new(format!(
            "{} native K3 setup is incomplete: {} pinned payload bytes remain",
            plan.mode.as_str(),
            plan.payload_bytes_missing
        )))
    }
}

/// Offline-audit whichever complete default installation is actually present.
/// A complete full corpus wins; otherwise a complete streaming payload is
/// accepted. Common pinned metadata, DSpark, and optional-Qwen checks are
/// performed once, so falling back to streaming does not hash multi-gigabyte
/// draft checkpoints a second time. No writes or network requests occur.
pub(crate) fn audit_installed(options: &SetupOptions) -> Result<SetupPlan> {
    if options.mode != SetupMode::Auto {
        return Err(DeltafinError::new(
            "installed setup audit requires automatic full/stream selection",
        ));
    }
    validate_root(&options.root)?;
    options.limits()?;
    let available = available_disk_bytes(&options.root)?;
    let common = inspect_common_requirements(options)?;

    let full_weights = inspect_weight_requirements(options, SetupMode::Full)?;
    let full = calculate_plan(
        options,
        SetupMode::Full,
        combine_requirements(common, full_weights),
        available,
    )?;
    if full.is_complete() {
        return Ok(full);
    }

    let stream_weights = inspect_weight_requirements(options, SetupMode::Stream)?;
    let stream = calculate_plan(
        options,
        SetupMode::Stream,
        combine_requirements(common, stream_weights),
        available,
    )?;
    if stream.is_complete() {
        return Ok(stream);
    }

    Err(DeltafinError::new(format!(
        "native K3 installation is incomplete: full mode is missing {} pinned bytes and streaming mode is missing {} pinned bytes; the offline audit requires pinned K3 metadata, the selected canonical weight roster, and required default DSpark data (run `deltafin setup` to resume)",
        full.payload_bytes_missing, stream.payload_bytes_missing,
    )))
}

/// Execute the full-quality plan through the existing transactional installers.
/// Streaming changes storage/performance only: every routed expert remains an
/// exact pinned tensor and is admitted on demand. Qwen is included only when
/// `include_qwen` is explicitly true.
pub(crate) fn run(options: &SetupOptions, progress: &dyn ProgressSink) -> Result<SetupReport> {
    let initial_plan = plan(options)?;
    initial_plan.require_capacity()?;
    let selected_mode = initial_plan.mode;

    setup_k3::install_at(&options.root)?;
    // The inventory is now pinned locally, so this second boundary replaces
    // the conservative fresh-install total with exact authenticated reuse.
    // Freeze the initially reported auto decision for this transaction.
    plan_for_resolved_mode(options, selected_mode)?.require_capacity()?;

    dspark_setup::install_at(&options.root)?;
    if options.include_qwen {
        setup_qwen::install_at(&options.root)?;
    }

    // Re-read capacity immediately before the only multi-terabyte phase.
    plan_for_resolved_mode(options, selected_mode)?.require_capacity()?;
    let inventory = setup_k3::load_installed_inventory(&options.root)?;
    let weight_plan = weight_fetch::plan(
        &options.root,
        &inventory,
        selected_mode.weight_selection()?,
        options.limits()?,
    )?;
    validate_weight_total(selected_mode, weight_plan.dry_run)?;
    let weight_progress = weight_fetch::execute(&weight_plan, progress)?;

    let mut final_plan = plan_for_resolved_mode(options, selected_mode)?;
    if !final_plan.is_complete() {
        return Err(DeltafinError::new(format!(
            "{} native K3 setup ended with {} pinned payload bytes remaining",
            selected_mode.as_str(),
            final_plan.payload_bytes_missing,
        )));
    }
    final_plan.auto_full_shortfall_bytes = initial_plan.auto_full_shortfall_bytes;
    Ok(SetupReport {
        initial_plan,
        final_plan,
        weight_progress,
    })
}

fn plan_for_resolved_mode(options: &SetupOptions, mode: SetupMode) -> Result<SetupPlan> {
    if mode == SetupMode::Auto {
        return Err(DeltafinError::new(
            "automatic setup mode was not resolved before execution",
        ));
    }
    let common = inspect_common_requirements(options)?;
    let weights = inspect_weight_requirements(options, mode)?;
    let available = available_disk_bytes(&options.root)?;
    calculate_plan(
        options,
        mode,
        combine_requirements(common, weights),
        available,
    )
}

fn inspect_common_requirements(options: &SetupOptions) -> Result<CommonRequirements> {
    let metadata_total = setup_k3::exact_payload_bytes();
    let metadata_path = options.root.join("k3-meta");
    let metadata_bytes_missing = match sys_fs::lstat(&metadata_path) {
        Ok(_) => {
            setup_k3::load_installed_inventory(&options.root)?;
            0
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => metadata_total,
        Err(error) => {
            return Err(io_error(
                "inspect K3 metadata destination",
                &metadata_path,
                error,
            ));
        }
    };

    let dspark_total = dspark_setup::exact_install_bytes()?;
    let dspark_installed = dspark_setup::audited_installed_bytes(&options.root)?;
    let dspark_bytes_missing = checked_sub(dspark_total, dspark_installed, "DSpark bytes")?;

    let qwen_bytes_missing = if options.include_qwen {
        let total = setup_qwen::exact_install_bytes()?;
        let installed = setup_qwen::audited_installed_bytes(&options.root)?;
        checked_sub(total, installed, "Qwen bytes")?
    } else {
        0
    };

    Ok(CommonRequirements {
        metadata_bytes_missing,
        dspark_bytes_missing,
        qwen_bytes_missing,
    })
}

fn inspect_weight_requirements(
    options: &SetupOptions,
    mode: SetupMode,
) -> Result<WeightRequirements> {
    let expected = mode.weight_bytes();
    let metadata_path = options.root.join("k3-meta");
    match sys_fs::lstat(&metadata_path) {
        Ok(_) => {
            let inventory = setup_k3::load_installed_inventory(&options.root)?;
            let weight_plan = weight_fetch::plan(
                &options.root,
                &inventory,
                mode.weight_selection()?,
                options.limits()?,
            )?;
            validate_weight_total(mode, weight_plan.dry_run)?;
            let storage = weight_plan.storage_requirement()?;
            Ok(WeightRequirements {
                work: weight_plan.dry_run,
                storage: Some(storage),
                peak_additional_bytes: storage.peak_additional_bytes,
                authenticated_inventory: true,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let temporary = match mode {
                SetupMode::Full => weight_fetch::maximum_temporary_bytes(options.limits()?)?,
                SetupMode::Stream => {
                    weight_fetch::maximum_resident_temporary_bytes(options.limits()?)?
                }
                SetupMode::Auto => unreachable!("mode was resolved above"),
            };
            let peak = expected
                .checked_add(temporary)
                .ok_or_else(|| DeltafinError::new("fresh K3 weight peak allocation overflowed"))?;
            Ok(WeightRequirements {
                work: WeightFetchDryRun {
                    expert_files_missing: if mode == SetupMode::Full {
                        K3_EXPERT_RAW_FILES
                    } else {
                        0
                    },
                    bytes_missing: expected,
                    ..WeightFetchDryRun::default()
                },
                storage: None,
                peak_additional_bytes: peak,
                authenticated_inventory: false,
            })
        }
        Err(error) => Err(io_error(
            "inspect K3 metadata destination",
            &metadata_path,
            error,
        )),
    }
}

fn combine_requirements(common: CommonRequirements, weights: WeightRequirements) -> Requirements {
    Requirements {
        metadata_bytes_missing: common.metadata_bytes_missing,
        dspark_bytes_missing: common.dspark_bytes_missing,
        weight_work: weights.work,
        weight_storage: weights.storage,
        weight_peak_additional_bytes: weights.peak_additional_bytes,
        qwen_bytes_missing: common.qwen_bytes_missing,
        authenticated_inventory: weights.authenticated_inventory,
    }
}

fn calculate_plan(
    options: &SetupOptions,
    mode: SetupMode,
    requirements: Requirements,
    available_bytes: u64,
) -> Result<SetupPlan> {
    if mode == SetupMode::Auto {
        return Err(DeltafinError::new(
            "automatic setup mode must resolve before capacity calculation",
        ));
    }
    if requirements.authenticated_inventory != requirements.weight_storage.is_some() {
        return Err(DeltafinError::new(
            "detailed weight storage must come from an authenticated inventory plan",
        ));
    }
    if requirements.weight_storage.is_some_and(|storage| {
        storage.peak_additional_bytes != requirements.weight_peak_additional_bytes
    }) {
        return Err(DeltafinError::new(
            "detailed weight storage disagrees with the setup peak",
        ));
    }
    let payload_bytes_missing = checked_sum(
        [
            requirements.metadata_bytes_missing,
            requirements.dspark_bytes_missing,
            requirements.weight_work.bytes_missing,
            requirements.qwen_bytes_missing,
        ],
        "one-shot payload bytes",
    )?;
    let peak_additional_bytes = checked_sum(
        [
            requirements.metadata_bytes_missing,
            requirements.dspark_bytes_missing,
            requirements.weight_peak_additional_bytes,
            requirements.qwen_bytes_missing,
        ],
        "one-shot peak additional bytes",
    )?;
    if payload_bytes_missing == 0 && peak_additional_bytes != 0 {
        return Err(DeltafinError::new(
            "completed one-shot plan still reports additional allocation",
        ));
    }
    let required_available_bytes = if payload_bytes_missing == 0 {
        0
    } else {
        peak_additional_bytes
            .checked_add(DEFAULT_HEADROOM_BYTES)
            .ok_or_else(|| DeltafinError::new("one-shot capacity requirement overflowed"))?
    };
    Ok(SetupPlan {
        mode,
        auto_full_shortfall_bytes: None,
        metadata_bytes_missing: requirements.metadata_bytes_missing,
        dspark_bytes_missing: requirements.dspark_bytes_missing,
        weight_work: requirements.weight_work,
        weight_storage: requirements.weight_storage,
        weight_peak_additional_bytes: requirements.weight_peak_additional_bytes,
        qwen_bytes_missing: requirements.qwen_bytes_missing,
        payload_bytes_missing,
        peak_additional_bytes,
        headroom_bytes: DEFAULT_HEADROOM_BYTES,
        required_available_bytes,
        available_bytes,
        shortfall_bytes: required_available_bytes.saturating_sub(available_bytes),
        authenticated_inventory: requirements.authenticated_inventory,
        includes_qwen: options.include_qwen,
    })
}

fn validate_weight_total(mode: SetupMode, work: WeightFetchDryRun) -> Result<()> {
    let total = work
        .bytes_missing
        .checked_add(work.bytes_reused)
        .ok_or_else(|| DeltafinError::new("authenticated K3 weight total overflowed"))?;
    let expected = mode.weight_bytes();
    if total != expected {
        return Err(DeltafinError::new(format!(
            "authenticated K3 inventory describes {total} canonical weight bytes; \
             expected the pinned {}-mode total {expected}",
            mode.as_str(),
        )));
    }
    Ok(())
}

pub(crate) fn available_disk_bytes(root: &Path) -> Result<u64> {
    let canonical = crate::sys::fs::canonicalize(root)
        .map_err(|error| io_error("resolve model root for capacity check", root, error))?;
    let directory = crate::sys::fs::Open::new()
        .read(true)
        .directory()
        .open(&canonical)
        .map_err(|error| io_error("open model root for capacity check", &canonical, error))?;
    crate::sys::fs::available_space(&directory, &canonical)
        .map_err(|error| io_error("query model-volume capacity", &canonical, error))
}

fn validate_root(root: &Path) -> Result<()> {
    let metadata = sys_fs::lstat(root)
        .map_err(|error| io_error("inspect one-shot model root", root, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DeltafinError::new(format!(
            "one-shot model root is not a real directory: {}",
            root.display()
        )));
    }
    Ok(())
}

fn checked_sum<const N: usize>(values: [u64; N], context: &str) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| DeltafinError::new(format!("{context} overflowed")))
    })
}

fn checked_sub(total: u64, installed: u64, context: &str) -> Result<u64> {
    total
        .checked_sub(installed)
        .ok_or_else(|| DeltafinError::new(format!("{context} installed credit exceeds its pin")))
}

fn io_error(context: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{context} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "deltafin-one-shot-{}-{}",
                std::process::id(),
                TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn options(include_qwen: bool) -> SetupOptions {
        let mut options = SetupOptions::full(PathBuf::from("/unused"));
        options.include_qwen = include_qwen;
        options
    }

    fn requirements() -> Requirements {
        Requirements {
            metadata_bytes_missing: 11,
            dspark_bytes_missing: 13,
            weight_work: WeightFetchDryRun {
                bytes_missing: 17,
                bytes_reused: FULL_WEIGHT_BYTES - 17,
                ..WeightFetchDryRun::default()
            },
            weight_storage: Some(WeightFetchStorage {
                final_bytes_to_allocate: 17,
                expert_run_temporary_bytes: 3,
                range_chunk_temporary_bytes: 5,
                peak_additional_bytes: 25,
                ..WeightFetchStorage::default()
            }),
            weight_peak_additional_bytes: 25,
            qwen_bytes_missing: 0,
            authenticated_inventory: true,
        }
    }

    #[test]
    fn pinned_full_quality_totals_are_exact() {
        assert_eq!(K3_EXPERT_RAW_FILES, 92 * 896);
        assert_eq!(EXPERT_WEIGHT_BYTES, 1_446_456_066_048);
        assert_eq!(RESIDENT_WEIGHT_BYTES, 114_404_258_816);
        assert_eq!(
            RESIDENT_WEIGHT_BYTES + EXPERT_WEIGHT_BYTES,
            FULL_WEIGHT_BYTES
        );
        assert_eq!(setup_k3::exact_payload_bytes(), 111_389_904);
    }

    #[test]
    fn pure_plan_adds_every_default_payload_and_headroom() {
        let options = options(false);
        let expected_payload = 11 + 13 + 17;
        let expected_peak = 11 + 13 + 25;
        let plan = calculate_plan(
            &options,
            SetupMode::Full,
            requirements(),
            expected_peak + DEFAULT_HEADROOM_BYTES,
        )
        .unwrap();
        assert_eq!(plan.payload_bytes_missing, expected_payload);
        assert_eq!(plan.peak_additional_bytes, expected_peak);
        assert_eq!(
            plan.required_available_bytes,
            expected_peak + DEFAULT_HEADROOM_BYTES
        );
        assert!(plan.has_capacity());
        assert!(plan.authenticated_inventory);
        assert!(!plan.includes_qwen);
    }

    #[test]
    fn insufficient_explicit_full_plan_fails_closed_and_points_to_streaming() {
        let options = options(false);
        let plan = calculate_plan(&options, SetupMode::Full, requirements(), 1).unwrap();
        assert!(!plan.has_capacity());
        let message = plan.require_capacity().unwrap_err().to_string();
        assert!(message.contains("use --stream"));
        assert!(message.contains("No further network write was"));
    }

    #[test]
    fn qwen_is_counted_only_as_an_explicit_add_on() {
        let mut with_qwen = requirements();
        with_qwen.qwen_bytes_missing = 19;
        // Requirements come from inspection and therefore omit Qwen entirely
        // for the default path.
        let mut without_qwen_requirement = with_qwen;
        without_qwen_requirement.qwen_bytes_missing = 0;
        let default = calculate_plan(
            &options(false),
            SetupMode::Full,
            without_qwen_requirement,
            u64::MAX,
        )
        .unwrap();
        let optional =
            calculate_plan(&options(true), SetupMode::Full, with_qwen, u64::MAX).unwrap();
        assert_eq!(
            optional.payload_bytes_missing,
            default.payload_bytes_missing + 19
        );
        assert!(optional.includes_qwen);
    }

    #[test]
    fn fresh_root_plan_is_read_only_and_uses_exact_full_payload() {
        let root = TempRoot::new();
        let options = SetupOptions::full(root.0.clone());
        let plan = plan(&options).unwrap();
        let expected_payload = setup_k3::exact_payload_bytes()
            + dspark_setup::exact_install_bytes().unwrap()
            + FULL_WEIGHT_BYTES;
        assert_eq!(plan.metadata_bytes_missing, 111_389_904);
        assert_eq!(plan.weight_work.bytes_missing, FULL_WEIGHT_BYTES);
        assert_eq!(plan.mode, SetupMode::Full);
        assert_eq!(plan.auto_full_shortfall_bytes, None);
        assert!(plan.weight_storage.is_none());
        assert_eq!(plan.payload_bytes_missing, expected_payload);
        assert_eq!(
            plan.required_available_bytes,
            plan.peak_additional_bytes + DEFAULT_HEADROOM_BYTES
        );
        assert!(plan.peak_additional_bytes > plan.payload_bytes_missing);
        assert!(!plan.authenticated_inventory);
        assert!(!plan.includes_qwen);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }

    #[test]
    fn fresh_stream_plan_keeps_exact_weights_but_omits_the_eager_expert_pool() {
        let root = TempRoot::new();
        let mut options = SetupOptions::full(root.0.clone());
        options.mode = SetupMode::Stream;
        let plan = plan(&options).unwrap();
        let expected_payload = setup_k3::exact_payload_bytes()
            + dspark_setup::exact_install_bytes().unwrap()
            + RESIDENT_WEIGHT_BYTES;
        assert_eq!(plan.mode, SetupMode::Stream);
        assert_eq!(plan.weight_work.bytes_missing, RESIDENT_WEIGHT_BYTES);
        assert_eq!(plan.weight_work.expert_files_missing, 0);
        assert_eq!(plan.payload_bytes_missing, expected_payload);
        assert_eq!(plan.auto_full_shortfall_bytes, None);
        assert!(plan.peak_additional_bytes < FULL_WEIGHT_BYTES);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }

    #[test]
    fn installed_audit_is_read_only_and_rejects_both_incomplete_modes() {
        let root = TempRoot::new();
        let mut options = SetupOptions::full(root.0.clone());
        options.mode = SetupMode::Auto;

        let error = audit_installed(&options).unwrap_err().to_string();

        assert!(error.contains("full mode is missing"));
        assert!(error.contains("streaming mode is missing"));
        assert!(error.contains("required default DSpark"));
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }

    #[test]
    fn installed_audit_refuses_an_explicit_setup_mode() {
        let root = TempRoot::new();
        let options = SetupOptions::full(root.0.clone());

        let error = audit_installed(&options).unwrap_err().to_string();

        assert!(error.contains("requires automatic full/stream selection"));
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }

    #[test]
    fn automatic_plan_chooses_full_only_when_its_validated_capacity_bound_fits() {
        let root = TempRoot::new();
        let full_options = SetupOptions::full(root.0.clone());
        let full = plan(&full_options).unwrap();
        let mut automatic_options = full_options;
        automatic_options.mode = SetupMode::Auto;
        let automatic = plan(&automatic_options).unwrap();
        if full.has_capacity() {
            assert_eq!(automatic.mode, SetupMode::Full);
            assert_eq!(automatic.auto_full_shortfall_bytes, None);
        } else {
            assert_eq!(automatic.mode, SetupMode::Stream);
            // Available space may move by filesystem blocks between the two
            // independent statvfs snapshots; the auto plan must preserve its
            // own positive full-mode shortfall rather than match a stale one.
            assert!(
                automatic
                    .auto_full_shortfall_bytes
                    .is_some_and(|value| value > 0)
            );
            assert_eq!(automatic.weight_work.bytes_missing, RESIDENT_WEIGHT_BYTES);
        }
        assert_eq!(fs::read_dir(&automatic_options.root).unwrap().count(), 0);
    }

    #[test]
    fn completed_plan_needs_no_reserved_free_space() {
        let options = options(false);
        let complete = Requirements {
            metadata_bytes_missing: 0,
            dspark_bytes_missing: 0,
            weight_work: WeightFetchDryRun {
                bytes_reused: FULL_WEIGHT_BYTES,
                ..WeightFetchDryRun::default()
            },
            weight_storage: Some(WeightFetchStorage::default()),
            weight_peak_additional_bytes: 0,
            qwen_bytes_missing: 0,
            authenticated_inventory: true,
        };
        let plan = calculate_plan(&options, SetupMode::Full, complete, 0).unwrap();
        assert!(plan.is_complete());
        assert_eq!(plan.required_available_bytes, 0);
        assert!(plan.has_capacity());
    }
}
