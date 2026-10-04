//! Production, row-streaming conversion of K3 BF16 spine matrices to row-I8.
//!
//! This codec is deliberately **not weight exact**. It reproduces the original
//! NumPy conversion contract byte-for-byte for finite K3 weights: BF16 is
//! widened to F32, each row uses `absmax / 127` (with an all-zero scale of
//! one), values are rounded to nearest with ties to even and clamped to
//! `[-127, 127]`, and the F32 scale is finally stored as IEEE F16.
//!
//! Conversion is bounded to one BF16 source row, two I8 work/verification rows,
//! and fixed-size scale/digest state. Durable paired partials permit a crash to
//! resume at the last authenticated row. Scales are published first, quantized
//! data second (the activation edge used by the existing loader), and an
//! identity/digest receipt last.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use half::f16;
use serde::{Deserialize, Serialize};

use crate::error::{DeltafinError, Result};
use crate::inventory::{K3Inventory, PINNED_INVENTORY_SHA256, TensorRecord, safe_tensor_path};
use crate::packfile::{Digest, DigestState, digest_open_file};
use crate::trusted_download::{fsync_directory, publish_hard_link, secure_create_new};
use crate::sys::fs::{self as sys_fs, Open};

pub const IS_WEIGHT_EXACT: bool = false;
pub const RECEIPT_SCHEMA: &str = "deltafin-spine-row-i8-f16-v1";
pub const DEFAULT_CHECKPOINT_ROWS: usize = 128;
pub const MAX_CHECKPOINT_ROWS: usize = 1 << 20;

const SOURCE_DIRECTORY: &str = "k3-resident/tensors";
const OUTPUT_PARENT: &str = "k3-resident-int8";
const OUTPUT_DIRECTORY: &str = "k3-resident-int8/tensors";
const SPOTLIGHT_MARKER: &str = ".metadata_never_index";
const LOCK_NAME: &str = ".spine-int8-convert.lock";
const RECEIPT_SUFFIX: &str = ".row-i8.json";
const MAX_RECEIPT_BYTES: u64 = 16 << 10;
const MAX_ROW_BYTES: u64 = 256 << 20;
const EXCLUDED_NAME_PARTS: [&str; 9] = [
    "conv1d",
    "norm",
    "A_log",
    "dt_bias",
    "res_proj",
    "e_score_correction_bias",
    ".experts.",
    "vision",
    "mm_projector",
];

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConvertOptions {
    pub model_root: PathBuf,
    pub resume: bool,
    pub checkpoint_rows: usize,
}

impl ConvertOptions {
    pub fn under(model_root: impl AsRef<Path>) -> Self {
        Self {
            model_root: model_root.as_ref().to_path_buf(),
            resume: true,
            checkpoint_rows: DEFAULT_CHECKPOINT_ROWS,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConvertReport {
    /// Always false: row-I8/F16-scale conversion changes target weights.
    pub weight_exact: bool,
    pub already_complete: bool,
    pub target_tensors: usize,
    pub converted_tensors: usize,
    pub resumed_tensors: usize,
    pub source_bytes: u64,
    pub quantized_bytes: u64,
    pub scale_bytes: u64,
    pub output_directory: PathBuf,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProgressEvent<'a> {
    Inventory {
        tensors: usize,
        source_bytes: u64,
    },
    TensorStarted {
        index: usize,
        tensors: usize,
        name: &'a str,
        rows: u64,
        resumed_rows: u64,
    },
    RowsCommitted {
        index: usize,
        tensors: usize,
        name: &'a str,
        completed_rows: u64,
        rows: u64,
    },
    ScalesPublished {
        index: usize,
        tensors: usize,
        name: &'a str,
    },
    TensorPublished {
        index: usize,
        tensors: usize,
        name: &'a str,
        resumed: bool,
    },
    Complete {
        tensors: usize,
        converted_tensors: usize,
        resumed_tensors: usize,
        source_bytes: u64,
    },
}

/// A synchronous deterministic observer. Returning an error stops conversion
/// only after the most recently reported row checkpoint or publication edge
/// has been made durable, which is also useful for controlled interruption.
pub trait ProgressSink {
    fn record(&mut self, event: ProgressEvent<'_>) -> Result<()>;
}

#[derive(Debug, Default)]
pub struct NoProgress;

impl ProgressSink for NoProgress {
    fn record(&mut self, _event: ProgressEvent<'_>) -> Result<()> {
        Ok(())
    }
}

/// Convert every Python-compatible target selected from the authenticated K3
/// inventory. Missing resident tensors are skipped just as in the original
/// converter; an existing unsafe or malformed source is rejected.
pub fn convert_full(
    options: &ConvertOptions,
    progress: &mut dyn ProgressSink,
) -> Result<ConvertReport> {
    validate_options(options)?;
    require_little_endian()?;
    validate_real_directory(&options.model_root, "model root")?;
    let source_root = options.model_root.join(SOURCE_DIRECTORY);
    validate_real_directory(&source_root, "resident tensor directory")?;
    let inventory = K3Inventory::load_from_root(&options.model_root)?;
    let targets = select_targets(&inventory, &source_root)?;
    let output_parent = options.model_root.join(OUTPUT_PARENT);
    let output_root = options.model_root.join(OUTPUT_DIRECTORY);
    convert_targets(&targets, &output_parent, &output_root, options, progress)
}

pub fn convert_full_quiet(options: &ConvertOptions) -> Result<ConvertReport> {
    convert_full(options, &mut NoProgress)
}

#[derive(Debug, Clone)]
struct Target {
    name: String,
    source: PathBuf,
    source_identity: FileIdentity,
    rows: u64,
    columns: u64,
    source_bytes: u64,
    quantized_bytes: u64,
    scale_bytes: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileIdentity {
    fn from_metadata(metadata: &sys_fs::Stat) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PairReceipt {
    schema: String,
    version: u32,
    weight_exact: bool,
    inventory_sha256: String,
    tensor: String,
    rows: u64,
    columns: u64,
    source: FileIdentity,
    quantized: FileIdentity,
    scales: FileIdentity,
    quantized_sha256: String,
    scales_sha256: String,
}

#[derive(Debug, Clone, Copy)]
struct TensorOutcome {
    converted: bool,
    resumed: bool,
}

#[derive(Debug, Clone, Copy)]
struct PairDigests {
    quantized: Digest,
    scales: Digest,
}

fn validate_options(options: &ConvertOptions) -> Result<()> {
    if options.checkpoint_rows == 0 || options.checkpoint_rows > MAX_CHECKPOINT_ROWS {
        return Err(invalid(format!(
            "checkpoint_rows must be in 1..={MAX_CHECKPOINT_ROWS}"
        )));
    }
    Ok(())
}

fn require_little_endian() -> Result<()> {
    if cfg!(target_endian = "little") {
        Ok(())
    } else {
        Err(invalid(
            "the canonical K3 BF16/I8/F16 files require a little-endian target",
        ))
    }
}

fn select_targets(inventory: &K3Inventory, source_root: &Path) -> Result<Vec<Target>> {
    let mut targets = Vec::new();
    for (name, record) in inventory.iter() {
        if !is_python_target(name, record) {
            continue;
        }
        let source = safe_tensor_path(source_root, name)?;
        let metadata = match sys_fs::lstat(&source) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(io_error("inspect resident BF16 tensor", &source, error)),
        };
        targets.push(target_from_metadata(name, record, source, metadata)?);
    }
    Ok(targets)
}

fn is_python_target(name: &str, record: &TensorRecord) -> bool {
    record.dtype == "BF16"
        && record.shape.len() == 2
        && !EXCLUDED_NAME_PARTS
            .iter()
            .any(|excluded| name.contains(excluded))
}

fn target_from_metadata(
    name: &str,
    record: &TensorRecord,
    source: PathBuf,
    metadata: sys_fs::Stat,
) -> Result<Target> {
    let [rows, columns]: [u64; 2] = record
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| invalid(format!("target {name:?} is not rank two")))?;
    let quantized_bytes = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid(format!("target {name:?} element count overflowed")))?;
    let source_bytes = quantized_bytes
        .checked_mul(2)
        .ok_or_else(|| invalid(format!("target {name:?} BF16 extent overflowed")))?;
    let scale_bytes = rows
        .checked_mul(2)
        .ok_or_else(|| invalid(format!("target {name:?} scale extent overflowed")))?;
    let row_bytes = columns
        .checked_mul(2)
        .ok_or_else(|| invalid(format!("target {name:?} row extent overflowed")))?;
    if rows == 0 || columns == 0 {
        return Err(invalid(format!("target {name:?} has an empty dimension")));
    }
    if row_bytes > MAX_ROW_BYTES {
        return Err(invalid(format!(
            "target {name:?} has a {row_bytes}-byte row exceeding the streaming bound"
        )));
    }
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != source_bytes {
        return Err(invalid(format!(
            "{} must be a regular non-symlink {source_bytes}-byte BF16 tensor",
            source.display()
        )));
    }
    Ok(Target {
        name: name.to_owned(),
        source,
        source_identity: FileIdentity::from_metadata(&metadata),
        rows,
        columns,
        source_bytes,
        quantized_bytes,
        scale_bytes,
    })
}

fn convert_targets(
    targets: &[Target],
    output_parent: &Path,
    output_root: &Path,
    options: &ConvertOptions,
    progress: &mut dyn ProgressSink,
) -> Result<ConvertReport> {
    create_real_directory(output_parent)?;
    ensure_spotlight_marker(output_parent)?;
    create_real_directory(output_root)?;
    let _lock = ConversionLock::acquire(output_parent)?;
    let source_bytes = checked_sum(targets.iter().map(|target| target.source_bytes), "source")?;
    let quantized_bytes = checked_sum(
        targets.iter().map(|target| target.quantized_bytes),
        "quantized output",
    )?;
    let scale_bytes = checked_sum(targets.iter().map(|target| target.scale_bytes), "scale")?;
    progress.record(ProgressEvent::Inventory {
        tensors: targets.len(),
        source_bytes,
    })?;

    let mut converted_tensors = 0_usize;
    let mut resumed_tensors = 0_usize;
    for (offset, target) in targets.iter().enumerate() {
        let outcome = convert_one(
            target,
            output_root,
            options,
            offset + 1,
            targets.len(),
            progress,
        )?;
        converted_tensors += usize::from(outcome.converted);
        resumed_tensors += usize::from(outcome.resumed);
    }
    let report = ConvertReport {
        weight_exact: IS_WEIGHT_EXACT,
        already_complete: converted_tensors == 0,
        target_tensors: targets.len(),
        converted_tensors,
        resumed_tensors,
        source_bytes,
        quantized_bytes,
        scale_bytes,
        output_directory: output_root.to_path_buf(),
    };
    progress.record(ProgressEvent::Complete {
        tensors: report.target_tensors,
        converted_tensors: report.converted_tensors,
        resumed_tensors: report.resumed_tensors,
        source_bytes: report.source_bytes,
    })?;
    Ok(report)
}

fn convert_one(
    target: &Target,
    output_root: &Path,
    options: &ConvertOptions,
    index: usize,
    tensors: usize,
    progress: &mut dyn ProgressSink,
) -> Result<TensorOutcome> {
    let paths = PairPaths::new(output_root, &target.name);
    if path_exists(&paths.receipt)? {
        if !options.resume {
            return Err(invalid(format!(
                "completed tensor exists and resume is disabled: {}",
                paths.receipt.display()
            )));
        }
        verify_receipt(target, &paths)?;
        progress.record(ProgressEvent::TensorStarted {
            index,
            tensors,
            name: &target.name,
            rows: target.rows,
            resumed_rows: target.rows,
        })?;
        progress.record(ProgressEvent::TensorPublished {
            index,
            tensors,
            name: &target.name,
            resumed: true,
        })?;
        return Ok(TensorOutcome {
            converted: false,
            resumed: true,
        });
    }

    let data_final = path_exists(&paths.data_final)?;
    let scale_final = path_exists(&paths.scale_final)?;
    let data_part = path_exists(&paths.data_part)?;
    let scale_part = path_exists(&paths.scale_part)?;
    let had_state = data_final || scale_final || data_part || scale_part;
    if had_state && !options.resume {
        return Err(invalid(format!(
            "tensor state exists and resume is disabled: {}",
            target.name
        )));
    }
    if data_final && !scale_final {
        return Err(invalid(format!(
            "activated quantized tensor lacks its scale pair: {}",
            target.name
        )));
    }

    let digests = if scale_final {
        let data_path = if data_final {
            &paths.data_final
        } else {
            if !data_part {
                return Err(invalid(format!(
                    "published scales lack their complete quantized partial: {}",
                    target.name
                )));
            }
            &paths.data_part
        };
        progress.record(ProgressEvent::TensorStarted {
            index,
            tensors,
            name: &target.name,
            rows: target.rows,
            resumed_rows: target.rows,
        })?;
        let digests = verify_complete_pair(target, data_path, &paths.scale_final)?;
        if !data_final {
            publish_hard_link(&paths.data_part, &paths.data_final, output_root)?;
        }
        digests
    } else {
        let (digests, resumed_rows) =
            build_pair(target, &paths, options, index, tensors, progress)?;
        if resumed_rows != 0 && !had_state {
            return Err(invalid(
                "internal resume accounting disagrees with partial state",
            ));
        }
        publish_hard_link(&paths.scale_part, &paths.scale_final, output_root)?;
        progress.record(ProgressEvent::ScalesPublished {
            index,
            tensors,
            name: &target.name,
        })?;
        publish_hard_link(&paths.data_part, &paths.data_final, output_root)?;
        digests
    };

    remove_redundant_link(&paths.scale_part, &paths.scale_final, output_root)?;
    remove_redundant_link(&paths.data_part, &paths.data_final, output_root)?;
    let receipt = build_receipt(target, &paths, digests)?;
    publish_receipt(&paths, output_root, &receipt)?;
    require_path_identity(&target.source, target.source_identity, "BF16 source")?;
    require_path_identity(
        &paths.data_final,
        receipt.quantized,
        "published quantized output",
    )?;
    require_path_identity(&paths.scale_final, receipt.scales, "published scale output")?;
    progress.record(ProgressEvent::TensorPublished {
        index,
        tensors,
        name: &target.name,
        resumed: had_state,
    })?;
    Ok(TensorOutcome {
        converted: true,
        resumed: had_state,
    })
}

struct PairPaths {
    data_final: PathBuf,
    scale_final: PathBuf,
    receipt: PathBuf,
    data_part: PathBuf,
    scale_part: PathBuf,
    receipt_part: PathBuf,
}

impl PairPaths {
    fn new(root: &Path, name: &str) -> Self {
        Self {
            data_final: root.join(format!("{name}.i8")),
            scale_final: root.join(format!("{name}.sc")),
            receipt: root.join(format!("{name}{RECEIPT_SUFFIX}")),
            data_part: root.join(format!(".{name}.i8.part")),
            scale_part: root.join(format!(".{name}.sc.part")),
            receipt_part: root.join(format!(".{name}{RECEIPT_SUFFIX}.part")),
        }
    }
}

fn build_pair(
    target: &Target,
    paths: &PairPaths,
    options: &ConvertOptions,
    index: usize,
    tensors: usize,
    progress: &mut dyn ProgressSink,
) -> Result<(PairDigests, u64)> {
    let (mut source, source_identity) =
        open_exact(&target.source, target.source_bytes, "BF16 source")?;
    if source_identity != target.source_identity {
        return Err(invalid(format!(
            "BF16 source changed after inventory preflight: {}",
            target.source.display()
        )));
    }
    let (mut data, _) = open_or_create_partial(&paths.data_part, paths_root(paths))?;
    let (mut scales, _) = open_or_create_partial(&paths.scale_part, paths_root(paths))?;
    let row_data_bytes = usize::try_from(target.columns)
        .map_err(|_| invalid("row column count exceeds this host"))?;
    let row_source_bytes = row_data_bytes
        .checked_mul(2)
        .ok_or_else(|| invalid("source row size overflows usize"))?;
    let mut resumed_rows = normalize_partial_lengths(target, &data, &scales)?;
    let mut source_row = vec![0_u8; row_source_bytes];
    let mut expected_data = vec![0_u8; row_data_bytes];
    let mut observed_data = vec![0_u8; row_data_bytes];
    let mut observed_scale = [0_u8; 2];
    let mut data_digest = DigestState::new();
    let mut scale_digest = DigestState::new();

    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| io_error("seek BF16 source", &target.source, error))?;
    data.seek(SeekFrom::Start(0))
        .map_err(|error| io_error("seek quantized partial", &paths.data_part, error))?;
    scales
        .seek(SeekFrom::Start(0))
        .map_err(|error| io_error("seek scale partial", &paths.scale_part, error))?;
    let mut verified_rows = 0_u64;
    while verified_rows < resumed_rows {
        read_exact_retry(&mut source, &mut source_row, &target.source)?;
        let expected_scale = quantize_row(&source_row, &mut expected_data)?;
        read_exact_retry(&mut data, &mut observed_data, &paths.data_part)?;
        read_exact_retry(&mut scales, &mut observed_scale, &paths.scale_part)?;
        if observed_data != expected_data || observed_scale != expected_scale {
            break;
        }
        data_digest.update(&observed_data);
        scale_digest.update(&observed_scale);
        verified_rows += 1;
    }
    if verified_rows != resumed_rows {
        resumed_rows = verified_rows;
        truncate_pair(target, &data, &scales, resumed_rows)?;
    }
    seek_row(
        target,
        &mut source,
        &mut data,
        &mut scales,
        resumed_rows,
        paths,
    )?;
    progress.record(ProgressEvent::TensorStarted {
        index,
        tensors,
        name: &target.name,
        rows: target.rows,
        resumed_rows,
    })?;

    let checkpoint = options.checkpoint_rows as u64;
    for row in resumed_rows..target.rows {
        read_exact_retry(&mut source, &mut source_row, &target.source)?;
        let scale = quantize_row(&source_row, &mut expected_data)?;
        // Scale first, data second: after a crash the recoverable committed
        // prefix is always the minimum of the two complete extents.
        scales
            .write_all(&scale)
            .map_err(|error| io_error("write scale partial", &paths.scale_part, error))?;
        data.write_all(&expected_data)
            .map_err(|error| io_error("write quantized partial", &paths.data_part, error))?;
        scale_digest.update(&scale);
        data_digest.update(&expected_data);
        let completed = row + 1;
        if completed % checkpoint == 0 || completed == target.rows {
            scales
                .sync_data()
                .map_err(|error| io_error("checkpoint scale partial", &paths.scale_part, error))?;
            data.sync_data().map_err(|error| {
                io_error("checkpoint quantized partial", &paths.data_part, error)
            })?;
            progress.record(ProgressEvent::RowsCommitted {
                index,
                tensors,
                name: &target.name,
                completed_rows: completed,
                rows: target.rows,
            })?;
        }
    }
    data.sync_all()
        .map_err(|error| io_error("fsync quantized partial", &paths.data_part, error))?;
    scales
        .sync_all()
        .map_err(|error| io_error("fsync scale partial", &paths.scale_part, error))?;
    require_open_identity(&source, &target.source, target.source_identity)?;
    require_path_identity(&target.source, target.source_identity, "BF16 source")?;
    require_open_extent(&data, &paths.data_part, target.quantized_bytes)?;
    require_open_extent(&scales, &paths.scale_part, target.scale_bytes)?;
    require_open_path_identity(&data, &paths.data_part, "quantized partial")?;
    require_open_path_identity(&scales, &paths.scale_part, "scale partial")?;
    Ok((
        PairDigests {
            quantized: data_digest.finalize(),
            scales: scale_digest.finalize(),
        },
        resumed_rows,
    ))
}

fn paths_root(paths: &PairPaths) -> &Path {
    paths
        .data_part
        .parent()
        .expect("pair paths always have an output parent")
}

fn normalize_partial_lengths(target: &Target, data: &File, scales: &File) -> Result<u64> {
    let data_len = data
        .metadata()
        .map_err(|error| io_error("stat quantized partial", &target.source, error))?
        .len();
    let scale_len = scales
        .metadata()
        .map_err(|error| io_error("stat scale partial", &target.source, error))?
        .len();
    if data_len > target.quantized_bytes || scale_len > target.scale_bytes {
        return Err(invalid(format!(
            "partial output exceeds the authenticated shape for {}",
            target.name
        )));
    }
    let data_rows = data_len / target.columns;
    let scale_rows = scale_len / 2;
    let rows = data_rows.min(scale_rows).min(target.rows);
    truncate_pair(target, data, scales, rows)?;
    Ok(rows)
}

fn truncate_pair(target: &Target, data: &File, scales: &File, rows: u64) -> Result<()> {
    let data_bytes = rows
        .checked_mul(target.columns)
        .ok_or_else(|| invalid("partial quantized extent overflowed"))?;
    let scale_bytes = rows
        .checked_mul(2)
        .ok_or_else(|| invalid("partial scale extent overflowed"))?;
    data.set_len(data_bytes)
        .map_err(|error| io_error("truncate quantized partial", &target.source, error))?;
    scales
        .set_len(scale_bytes)
        .map_err(|error| io_error("truncate scale partial", &target.source, error))
}

fn seek_row(
    target: &Target,
    source: &mut File,
    data: &mut File,
    scales: &mut File,
    row: u64,
    paths: &PairPaths,
) -> Result<()> {
    let source_offset = row
        .checked_mul(target.columns)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| invalid("source resume offset overflowed"))?;
    let data_offset = row
        .checked_mul(target.columns)
        .ok_or_else(|| invalid("quantized resume offset overflowed"))?;
    let scale_offset = row
        .checked_mul(2)
        .ok_or_else(|| invalid("scale resume offset overflowed"))?;
    source
        .seek(SeekFrom::Start(source_offset))
        .map_err(|error| io_error("seek BF16 source", &target.source, error))?;
    data.seek(SeekFrom::Start(data_offset))
        .map_err(|error| io_error("seek quantized partial", &paths.data_part, error))?;
    scales
        .seek(SeekFrom::Start(scale_offset))
        .map_err(|error| io_error("seek scale partial", &paths.scale_part, error))?;
    Ok(())
}

fn verify_complete_pair(
    target: &Target,
    data_path: &Path,
    scale_path: &Path,
) -> Result<PairDigests> {
    let (mut source, source_identity) =
        open_exact(&target.source, target.source_bytes, "BF16 source")?;
    if source_identity != target.source_identity {
        return Err(invalid(format!(
            "BF16 source changed before pair recovery: {}",
            target.source.display()
        )));
    }
    let (mut data, data_identity) =
        open_exact(data_path, target.quantized_bytes, "quantized output")?;
    let (mut scales, scale_identity) = open_exact(scale_path, target.scale_bytes, "scale output")?;
    let row_data_bytes = usize::try_from(target.columns)
        .map_err(|_| invalid("row column count exceeds this host"))?;
    let row_source_bytes = row_data_bytes
        .checked_mul(2)
        .ok_or_else(|| invalid("source row size overflows usize"))?;
    let mut source_row = vec![0_u8; row_source_bytes];
    let mut expected_data = vec![0_u8; row_data_bytes];
    let mut observed_data = vec![0_u8; row_data_bytes];
    let mut observed_scale = [0_u8; 2];
    let mut data_digest = DigestState::new();
    let mut scale_digest = DigestState::new();
    for row in 0..target.rows {
        read_exact_retry(&mut source, &mut source_row, &target.source)?;
        let expected_scale = quantize_row(&source_row, &mut expected_data)?;
        read_exact_retry(&mut data, &mut observed_data, data_path)?;
        read_exact_retry(&mut scales, &mut observed_scale, scale_path)?;
        if observed_data != expected_data || observed_scale != expected_scale {
            return Err(invalid(format!(
                "published pair differs from BF16 source for {} at row {row}",
                target.name
            )));
        }
        data_digest.update(&observed_data);
        scale_digest.update(&observed_scale);
    }
    require_open_identity(&source, &target.source, target.source_identity)?;
    require_path_identity(&target.source, target.source_identity, "BF16 source")?;
    require_open_identity(&data, data_path, data_identity)?;
    require_path_identity(data_path, data_identity, "quantized output")?;
    require_open_identity(&scales, scale_path, scale_identity)?;
    require_path_identity(scale_path, scale_identity, "scale output")?;
    Ok(PairDigests {
        quantized: data_digest.finalize(),
        scales: scale_digest.finalize(),
    })
}

fn quantize_row(source_bf16: &[u8], quantized: &mut [u8]) -> Result<[u8; 2]> {
    if source_bf16.len() != quantized.len().saturating_mul(2) || quantized.is_empty() {
        return Err(invalid("BF16 row and quantized row extents disagree"));
    }
    let mut maximum = 0.0_f32;
    for bits in source_bf16.chunks_exact(2) {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([bits[0], bits[1]])) << 16);
        if !value.is_finite() {
            return Err(invalid("BF16 spine row contains a non-finite value"));
        }
        maximum = maximum.max(value.abs());
    }
    let scale = if maximum == 0.0 {
        1.0_f32
    } else {
        maximum / 127.0_f32
    };
    for (bits, output) in source_bf16.chunks_exact(2).zip(quantized.iter_mut()) {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([bits[0], bits[1]])) << 16);
        let rounded = (value / scale).round_ties_even().clamp(-127.0, 127.0);
        *output = (rounded as i8) as u8;
    }
    Ok(f16::from_f32(scale).to_bits().to_le_bytes())
}

fn build_receipt(target: &Target, paths: &PairPaths, digests: PairDigests) -> Result<PairReceipt> {
    let (quantized_file, quantized) = open_exact(
        &paths.data_final,
        target.quantized_bytes,
        "published quantized output",
    )?;
    let (scale_file, scales) = open_exact(
        &paths.scale_final,
        target.scale_bytes,
        "published scale output",
    )?;
    let observed_quantized = digest_open_file(&quantized_file, &paths.data_final)
        .map_err(|error| invalid(format!("hash published quantized output: {error}")))?;
    let observed_scales = digest_open_file(&scale_file, &paths.scale_final)
        .map_err(|error| invalid(format!("hash published scale output: {error}")))?;
    if observed_quantized != digests.quantized || observed_scales != digests.scales {
        return Err(invalid(format!(
            "published output digest changed before receipt for {}",
            target.name
        )));
    }
    require_open_identity(&quantized_file, &paths.data_final, quantized)?;
    require_open_identity(&scale_file, &paths.scale_final, scales)?;
    require_path_identity(&target.source, target.source_identity, "BF16 source")?;
    require_path_identity(&paths.data_final, quantized, "published quantized output")?;
    require_path_identity(&paths.scale_final, scales, "published scale output")?;
    Ok(PairReceipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        version: 1,
        weight_exact: IS_WEIGHT_EXACT,
        inventory_sha256: hex_digest(PINNED_INVENTORY_SHA256),
        tensor: target.name.clone(),
        rows: target.rows,
        columns: target.columns,
        source: target.source_identity,
        quantized,
        scales,
        quantized_sha256: hex_digest(digests.quantized),
        scales_sha256: hex_digest(digests.scales),
    })
}

fn publish_receipt(paths: &PairPaths, output_root: &Path, receipt: &PairReceipt) -> Result<()> {
    let payload = serde_json::to_vec(receipt)
        .map_err(|error| invalid(format!("serialize pair receipt: {error}")))?;
    let mut create = false;
    match sys_fs::lstat(&paths.receipt_part) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || !metadata.is_owner_only()
            {
                return Err(invalid(format!(
                    "unsafe receipt partial: {}",
                    paths.receipt_part.display()
                )));
            }
            let matches = if metadata.len() == payload.len() as u64 {
                let mut file = open_exact(
                    &paths.receipt_part,
                    payload.len() as u64,
                    "pair receipt partial",
                )?
                .0;
                let mut observed = Vec::with_capacity(payload.len());
                file.read_to_end(&mut observed).map_err(|error| {
                    io_error("read pair receipt partial", &paths.receipt_part, error)
                })?;
                observed == payload
            } else {
                false
            };
            if !matches {
                // This stable name is exclusively owned under ConversionLock.
                // A crash may have left any prefix; the already-authenticated
                // final pair is the authority for recreating it.
                fs::remove_file(&paths.receipt_part).map_err(|error| {
                    io_error(
                        "remove incomplete receipt partial",
                        &paths.receipt_part,
                        error,
                    )
                })?;
                fsync_directory(output_root)?;
                create = true;
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => create = true,
        Err(error) => {
            return Err(io_error(
                "inspect pair receipt partial",
                &paths.receipt_part,
                error,
            ));
        }
    }
    if create {
        let mut file = secure_create_new(&paths.receipt_part, 0o600)?;
        file.write_all(&payload)
            .map_err(|error| io_error("write pair receipt partial", &paths.receipt_part, error))?;
        file.sync_all()
            .map_err(|error| io_error("fsync pair receipt partial", &paths.receipt_part, error))?;
        drop(file);
        fsync_directory(output_root)?;
    }
    let (mut partial, partial_identity) = open_exact(
        &paths.receipt_part,
        payload.len() as u64,
        "pair receipt partial",
    )?;
    let mut observed = Vec::with_capacity(payload.len());
    partial
        .read_to_end(&mut observed)
        .map_err(|error| io_error("read pair receipt partial", &paths.receipt_part, error))?;
    if observed != payload {
        return Err(invalid(format!(
            "pair receipt partial changed before publication: {}",
            paths.receipt_part.display()
        )));
    }
    require_open_identity(&partial, &paths.receipt_part, partial_identity)?;
    publish_hard_link(&paths.receipt_part, &paths.receipt, output_root)?;
    let published_identity = FileIdentity::from_metadata(
        &sys_fs::fstat(&partial)
            .map_err(|error| io_error("restat published pair receipt", &paths.receipt, error))?,
    );
    require_path_identity(&paths.receipt, published_identity, "published pair receipt")
}

fn verify_receipt(target: &Target, paths: &PairPaths) -> Result<()> {
    let (mut receipt_file, receipt_identity) =
        open_bounded(&paths.receipt, MAX_RECEIPT_BYTES, "pair receipt")?;
    let mut raw = Vec::new();
    receipt_file
        .read_to_end(&mut raw)
        .map_err(|error| io_error("read pair receipt", &paths.receipt, error))?;
    let receipt: PairReceipt = serde_json::from_slice(&raw)
        .map_err(|error| invalid(format!("parse pair receipt: {error}")))?;
    if receipt.schema != RECEIPT_SCHEMA
        || receipt.version != 1
        || receipt.weight_exact != IS_WEIGHT_EXACT
        || receipt.inventory_sha256 != hex_digest(PINNED_INVENTORY_SHA256)
        || receipt.tensor != target.name
        || receipt.rows != target.rows
        || receipt.columns != target.columns
        || receipt.source != target.source_identity
    {
        return Err(invalid(format!(
            "pair receipt identity disagrees with authenticated target {}",
            target.name
        )));
    }
    require_path_identity(&target.source, receipt.source, "BF16 source")?;
    let (data, data_identity) = open_exact(
        &paths.data_final,
        target.quantized_bytes,
        "published quantized output",
    )?;
    let (scales, scale_identity) = open_exact(
        &paths.scale_final,
        target.scale_bytes,
        "published scale output",
    )?;
    if data_identity != receipt.quantized || scale_identity != receipt.scales {
        return Err(invalid(format!(
            "published pair identity changed after receipt for {}",
            target.name
        )));
    }
    let data_digest = digest_open_file(&data, &paths.data_final)
        .map_err(|error| invalid(format!("hash quantized output: {error}")))?;
    let scale_digest = digest_open_file(&scales, &paths.scale_final)
        .map_err(|error| invalid(format!("hash scale output: {error}")))?;
    if hex_digest(data_digest) != receipt.quantized_sha256
        || hex_digest(scale_digest) != receipt.scales_sha256
    {
        return Err(invalid(format!(
            "published pair digest changed after receipt for {}",
            target.name
        )));
    }
    require_open_identity(&data, &paths.data_final, receipt.quantized)?;
    require_open_identity(&scales, &paths.scale_final, receipt.scales)?;
    require_open_identity(&receipt_file, &paths.receipt, receipt_identity)?;
    require_path_identity(&target.source, receipt.source, "BF16 source")?;
    require_path_identity(
        &paths.data_final,
        receipt.quantized,
        "published quantized output",
    )?;
    require_path_identity(&paths.scale_final, receipt.scales, "published scale output")?;
    require_path_identity(&paths.receipt, receipt_identity, "pair receipt")?;
    Ok(())
}

fn open_or_create_partial(path: &Path, directory: &Path) -> Result<(File, bool)> {
    match sys_fs::lstat(path) {
        Ok(observed) => {
            if observed.file_type().is_symlink()
                || !observed.is_file()
                || !observed.is_owner_only()
            {
                return Err(invalid(format!(
                    "partial is not a private regular non-symlink file: {}",
                    path.display()
                )));
            }
            let file = Open::new()
                .read(true)
                .write(true)
                .no_follow()
                .open(path)
                .map_err(|error| {
                    io_error("open partial without following symlinks", path, error)
                })?;
            let opened = sys_fs::fstat(&file)
                .map_err(|error| io_error("stat opened partial", path, error))?;
            if !opened.is_owner_only()
                || FileIdentity::from_metadata(&opened) != FileIdentity::from_metadata(&observed)
            {
                return Err(invalid(format!(
                    "partial changed while opening: {}",
                    path.display()
                )));
            }
            Ok((file, true))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let file = Open::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o600)
                .no_follow()
                .open(path)
                .map_err(|error| io_error("securely create conversion partial", path, error))?;
            file.sync_all()
                .map_err(|error| io_error("fsync new conversion partial", path, error))?;
            fsync_directory(directory)?;
            Ok((file, false))
        }
        Err(error) => Err(io_error("inspect conversion partial", path, error)),
    }
}

fn open_exact(path: &Path, bytes: u64, label: &str) -> Result<(File, FileIdentity)> {
    let observed = sys_fs::lstat(path).map_err(|error| io_error("inspect", path, error))?;
    if observed.file_type().is_symlink() || !observed.is_file() || observed.len() != bytes {
        return Err(invalid(format!(
            "{label} {} must be a regular non-symlink {bytes}-byte file",
            path.display()
        )));
    }
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open without following symlinks", path, error))?;
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat opened file", path, error))?;
    let observed = FileIdentity::from_metadata(&observed);
    let opened = FileIdentity::from_metadata(&opened);
    if opened != observed {
        return Err(invalid(format!(
            "{label} changed while opening: {}",
            path.display()
        )));
    }
    Ok((file, opened))
}

fn open_bounded(path: &Path, maximum: u64, label: &str) -> Result<(File, FileIdentity)> {
    let observed = sys_fs::lstat(path).map_err(|error| io_error("inspect", path, error))?;
    if observed.file_type().is_symlink()
        || !observed.is_file()
        || observed.len() == 0
        || observed.len() > maximum
    {
        return Err(invalid(format!(
            "{label} {} must be a non-empty regular file no larger than {maximum} bytes",
            path.display()
        )));
    }
    open_exact(path, observed.len(), label)
}

fn require_open_identity(file: &File, path: &Path, expected: FileIdentity) -> Result<()> {
    let current = sys_fs::fstat(&file)
        .map_err(|error| io_error("restat opened file", path, error))?;
    if FileIdentity::from_metadata(&current) != expected {
        return Err(invalid(format!(
            "opened file identity changed during conversion: {}",
            path.display()
        )));
    }
    Ok(())
}

fn require_path_identity(path: &Path, expected: FileIdentity, label: &str) -> Result<()> {
    let metadata =
        sys_fs::lstat(path).map_err(|error| io_error("reinspect", path, error))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || FileIdentity::from_metadata(&metadata) != expected
    {
        return Err(invalid(format!(
            "{label} identity changed: {}",
            path.display()
        )));
    }
    Ok(())
}

fn require_open_extent(file: &File, path: &Path, expected: u64) -> Result<()> {
    let actual = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat completed partial", path, error))?
        .len();
    if actual != expected {
        return Err(invalid(format!(
            "completed partial {} has {actual} bytes; expected {expected}",
            path.display()
        )));
    }
    Ok(())
}

fn require_open_path_identity(file: &File, path: &Path, label: &str) -> Result<FileIdentity> {
    let metadata = sys_fs::fstat(&file)
        .map_err(|error| io_error("restat opened file", path, error))?;
    let identity = FileIdentity::from_metadata(&metadata);
    require_path_identity(path, identity, label)?;
    Ok(identity)
}

fn read_exact_retry(file: &mut File, target: &mut [u8], path: &Path) -> Result<()> {
    let mut filled = 0;
    while filled < target.len() {
        match file.read(&mut target[filled..]) {
            Ok(0) => {
                return Err(invalid(format!(
                    "short read from {} at {filled} of {} bytes",
                    path.display(),
                    target.len()
                )));
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_error("read conversion input", path, error)),
        }
    }
    Ok(())
}

fn remove_redundant_link(part: &Path, final_path: &Path, directory: &Path) -> Result<()> {
    let part_metadata = match sys_fs::lstat(part) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(io_error("inspect redundant partial link", part, error)),
    };
    let final_metadata = sys_fs::lstat(final_path)
        .map_err(|error| io_error("inspect published output", final_path, error))?;
    if part_metadata.file_type().is_symlink()
        || !part_metadata.is_file()
        || !final_metadata.is_file()
        || (part_metadata.dev(), part_metadata.ino())
            != (final_metadata.dev(), final_metadata.ino())
    {
        return Err(invalid(format!(
            "partial and published output are not the same regular file: {}",
            final_path.display()
        )));
    }
    fs::remove_file(part)
        .map_err(|error| io_error("remove redundant partial link", part, error))?;
    fsync_directory(directory)
}

fn create_real_directory(path: &Path) -> Result<()> {
    ensure_directory_without_links(path)?;
    validate_real_directory(path, "conversion output directory")?;
    fsync_directory(path)
}

fn ensure_directory_without_links(path: &Path) -> Result<()> {
    let selected = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    match sys_fs::lstat(selected) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => return Ok(()),
        Ok(_) => {
            return Err(invalid(format!(
                "directory component is not a real directory: {}",
                selected.display()
            )));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect directory component", selected, error)),
    }
    let parent = selected
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if parent == selected {
        return Err(invalid(format!(
            "cannot establish directory ancestor for {}",
            selected.display()
        )));
    }
    ensure_directory_without_links(parent)?;
    match fs::create_dir(selected) {
        Ok(()) => fsync_directory(parent)?,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io_error("create conversion directory", selected, error)),
    }
    validate_real_directory(selected, "conversion directory component")
}

fn ensure_spotlight_marker(directory: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let marker = directory.join(SPOTLIGHT_MARKER);
    match sys_fs::lstat(&marker) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(invalid(format!(
            "unsafe Spotlight marker {}",
            marker.display()
        ))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            secure_create_new(&marker, 0o600)?
                .sync_all()
                .map_err(|error| io_error("fsync Spotlight marker", &marker, error))?;
            fsync_directory(directory)
        }
        Err(error) => Err(io_error("inspect Spotlight marker", &marker, error)),
    }
}

fn validate_real_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = sys_fs::lstat(path).map_err(|error| io_error("inspect", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!("{label} is not a real directory")));
    }
    Ok(())
}

struct ConversionLock {
    _file: File,
}

impl ConversionLock {
    fn acquire(directory: &Path) -> Result<Self> {
        let path = directory.join(LOCK_NAME);
        let existed = match sys_fs::lstat(&path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    return Err(invalid("conversion lock is not a regular non-symlink file"));
                }
                true
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(error) => return Err(io_error("inspect conversion lock", &path, error)),
        };
        let file = Open::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .no_follow()
            .open(&path)
            .map_err(|error| io_error("open conversion lock", &path, error))?;
        let opened = sys_fs::fstat(&file)
            .map_err(|error| io_error("stat conversion lock", &path, error))?;
        let live = sys_fs::lstat(&path)
            .map_err(|error| io_error("reinspect conversion lock", &path, error))?;
        if !opened.is_file()
            || !opened.is_owner_only()
            || live.file_type().is_symlink()
            || !live.is_file()
            || (opened.dev(), opened.ino()) != (live.dev(), live.ino())
        {
            return Err(invalid("conversion lock is not a regular file"));
        }
        if !existed {
            file.sync_all()
                .map_err(|error| io_error("fsync conversion lock", &path, error))?;
            fsync_directory(directory)?;
        }
        let locked = sys_fs::try_lock_exclusive(&file)
            .map_err(|error| io_error("acquire exclusive conversion lock", &path, error))?;
        if !locked {
            return Err(io_error(
                "acquire exclusive conversion lock",
                &path,
                io::Error::from(io::ErrorKind::WouldBlock),
            ));
        }
        Ok(Self { _file: file })
    }
}


fn path_exists(path: &Path) -> Result<bool> {
    match sys_fs::lstat(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(invalid(format!(
            "refusing symlink conversion path {}",
            path.display()
        ))),
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io_error("inspect conversion path", path, error)),
    }
}

fn checked_sum(mut values: impl Iterator<Item = u64>, label: &str) -> Result<u64> {
    values.try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| invalid(format!("{label} byte total overflowed")))
    })
}

fn hex_digest(digest: Digest) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn invalid(message: impl Into<String>) -> DeltafinError {
    DeltafinError::new(format!(
        "native spine int8 conversion failed: {}",
        message.into()
    ))
}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "deltafin-spine-int8-{}-{}",
                std::process::id(),
                NEXT_TEST.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn raw_bf16(bits: &[u16]) -> Vec<u8> {
        bits.iter().flat_map(|bits| bits.to_le_bytes()).collect()
    }

    #[test]
    fn golden_numpy_bf16_row_parity_includes_ties_and_zero_scale() {
        let rows = [
            (
                [17150, 16128, 16320, 16416, 48896, 49088, 49184, 49918],
                [127_i8, 0, 2, 2, 0, -2, -2, -127],
                0x3c00_u16,
            ),
            ([0, 32768, 0, 0, 0, 0, 0, 0], [0_i8; 8], 0x3c00_u16),
            (
                [49152, 16256, 16128, 49024, 48896, 0, 16384, 49152],
                [-127_i8, 64, 32, -64, -32, 0, 127, -127],
                0x2408_u16,
            ),
        ];
        for (source, expected, scale) in rows {
            let mut output = [0_u8; 8];
            let observed_scale = quantize_row(&raw_bf16(&source), &mut output).unwrap();
            assert_eq!(
                output,
                expected.map(|value| value as u8),
                "quantized row differs from NumPy"
            );
            assert_eq!(u16::from_le_bytes(observed_scale), scale);
        }
    }

    #[test]
    fn deterministic_finite_bf16_corpus_matches_numpy_reference_digest() {
        const ROWS: usize = 257;
        const COLUMNS: usize = 259;
        const NUMPY_SHA256: &str =
            "fa09a36deccba67cd6e8f2f097e4d2d4adbb084974dffcb86a716b88769692f7";

        let mut state = 0xc0ff_ee11_u32;
        let mut source = vec![0_u8; COLUMNS * 2];
        let mut quantized = vec![0_u8; COLUMNS];
        let mut transcript = Vec::with_capacity(ROWS * (COLUMNS + 2));
        for _ in 0..ROWS {
            for (column, output) in source.chunks_exact_mut(2).enumerate() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let mut bits = (state >> 16) as u16;
                if bits & 0x7f80 == 0x7f80 {
                    // Preserve sign/mantissa while changing exponent 0xff to
                    // 0xfe so the corpus covers only the converter contract's
                    // finite BF16 domain.
                    bits ^= 0x0080;
                }
                output.copy_from_slice(&bits.to_le_bytes());
                debug_assert!(column < COLUMNS);
            }
            let scale = quantize_row(&source, &mut quantized).unwrap();
            transcript.extend_from_slice(&quantized);
            transcript.extend_from_slice(&scale);
        }
        assert_eq!(
            hex_digest(crate::packfile::digest_bytes(&transcript)),
            NUMPY_SHA256
        );
    }

    fn synthetic_target(root: &Path, name: &str, rows: u64, columns: u64) -> Target {
        let source = root.join(name);
        let mut file = File::create(&source).unwrap();
        for row in 0..rows {
            for column in 0..columns {
                let value = ((row * columns + column + 1) as f32)
                    * if column & 1 == 0 { 1.0 } else { -1.0 };
                file.write_all(&((value.to_bits() >> 16) as u16).to_le_bytes())
                    .unwrap();
            }
        }
        file.sync_all().unwrap();
        let metadata = sys_fs::lstat(&source).unwrap();
        Target {
            name: name.into(),
            source,
            source_identity: FileIdentity::from_metadata(&metadata),
            rows,
            columns,
            source_bytes: rows * columns * 2,
            quantized_bytes: rows * columns,
            scale_bytes: rows * 2,
        }
    }

    fn expected_outputs(target: &Target) -> (Vec<u8>, Vec<u8>) {
        let source = fs::read(&target.source).unwrap();
        let columns = usize::try_from(target.columns).unwrap();
        let mut quantized_row = vec![0_u8; columns];
        let mut quantized = Vec::with_capacity(usize::try_from(target.quantized_bytes).unwrap());
        let mut scales = Vec::with_capacity(usize::try_from(target.scale_bytes).unwrap());
        for source_row in source.chunks_exact(columns * 2) {
            let scale = quantize_row(source_row, &mut quantized_row).unwrap();
            quantized.extend_from_slice(&quantized_row);
            scales.extend_from_slice(&scale);
        }
        (quantized, scales)
    }

    struct InterruptRows {
        at: u64,
    }

    impl ProgressSink for InterruptRows {
        fn record(&mut self, event: ProgressEvent<'_>) -> Result<()> {
            if matches!(
                event,
                ProgressEvent::RowsCommitted { completed_rows, .. } if completed_rows == self.at
            ) {
                return Err(DeltafinError::new("controlled row interruption"));
            }
            Ok(())
        }
    }

    struct InterruptScales;

    impl ProgressSink for InterruptScales {
        fn record(&mut self, event: ProgressEvent<'_>) -> Result<()> {
            if matches!(event, ProgressEvent::ScalesPublished { .. }) {
                return Err(DeltafinError::new("controlled publication interruption"));
            }
            Ok(())
        }
    }

    #[derive(Default)]
    struct CaptureStart {
        resumed_rows: Option<u64>,
    }

    impl ProgressSink for CaptureStart {
        fn record(&mut self, event: ProgressEvent<'_>) -> Result<()> {
            if let ProgressEvent::TensorStarted { resumed_rows, .. } = event {
                self.resumed_rows = Some(resumed_rows);
            }
            Ok(())
        }
    }

    struct ReplaceSource {
        path: PathBuf,
        replaced: bool,
    }

    impl ProgressSink for ReplaceSource {
        fn record(&mut self, event: ProgressEvent<'_>) -> Result<()> {
            if !self.replaced
                && matches!(
                    event,
                    ProgressEvent::RowsCommitted {
                        completed_rows: 1,
                        ..
                    }
                )
            {
                let mut replacement = fs::read(&self.path).unwrap();
                let mantissa_byte = replacement.len() - 2;
                replacement[mantissa_byte] ^= 0x01;
                let replacement_path = self.path.with_extension("replacement");
                let mut file = File::create(&replacement_path).unwrap();
                file.write_all(&replacement).unwrap();
                file.sync_all().unwrap();
                drop(file);
                fs::rename(&replacement_path, &self.path).unwrap();
                self.replaced = true;
            }
            Ok(())
        }
    }

    fn test_options(root: &Path) -> ConvertOptions {
        ConvertOptions {
            model_root: root.to_path_buf(),
            resume: true,
            checkpoint_rows: 1,
        }
    }

    #[test]
    fn interruption_resumes_from_the_last_authenticated_row() {
        let directory = TestDirectory::new();
        let source = directory.0.join("source");
        let output_parent = directory.0.join("int8");
        let output = output_parent.join("tensors");
        fs::create_dir(&source).unwrap();
        let target = synthetic_target(&source, "weight", 4, 8);
        let options = test_options(&directory.0);
        let first = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut InterruptRows { at: 2 },
        );
        assert!(first.is_err());
        assert_eq!(
            fs::metadata(output.join(".weight.i8.part")).unwrap().len(),
            16
        );
        assert_eq!(
            fs::metadata(output.join(".weight.sc.part")).unwrap().len(),
            4
        );
        assert!(!output.join("weight.i8").exists());

        let report = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut NoProgress,
        )
        .unwrap();
        assert!(!report.weight_exact);
        assert_eq!(report.converted_tensors, 1);
        assert_eq!(report.resumed_tensors, 1);
        assert_eq!(fs::metadata(output.join("weight.i8")).unwrap().len(), 32);
        assert_eq!(fs::metadata(output.join("weight.sc")).unwrap().len(), 8);
        assert!(output.join(format!("weight{RECEIPT_SUFFIX}")).is_file());
    }

    #[test]
    fn interruption_between_scale_and_data_publication_resumes_the_pair() {
        let directory = TestDirectory::new();
        let source = directory.0.join("source");
        let output_parent = directory.0.join("int8");
        let output = output_parent.join("tensors");
        fs::create_dir(&source).unwrap();
        let target = synthetic_target(&source, "projection", 3, 8);
        let options = test_options(&directory.0);
        let first = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut InterruptScales,
        );
        assert!(first.is_err());
        assert!(output.join("projection.sc").is_file());
        assert!(output.join(".projection.i8.part").is_file());
        assert!(!output.join("projection.i8").exists());

        let report = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut NoProgress,
        )
        .unwrap();
        assert_eq!(report.converted_tensors, 1);
        assert_eq!(report.resumed_tensors, 1);
        assert!(output.join("projection.i8").is_file());
        assert!(output.join(format!("projection{RECEIPT_SUFFIX}")).is_file());

        let second = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut NoProgress,
        )
        .unwrap();
        assert!(second.already_complete);
        assert_eq!(second.resumed_tensors, 1);
    }

    #[test]
    fn corrupt_and_torn_partials_rewind_to_the_authenticated_prefix() {
        let directory = TestDirectory::new();
        let source = directory.0.join("source");
        let output_parent = directory.0.join("int8");
        let output = output_parent.join("tensors");
        fs::create_dir(&source).unwrap();
        let target = synthetic_target(&source, "corruptible", 4, 8);
        let expected = expected_outputs(&target);
        let options = test_options(&directory.0);

        let first = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut InterruptRows { at: 3 },
        );
        assert!(first.is_err());

        let data_part = output.join(".corruptible.i8.part");
        let mut data = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&data_part)
            .unwrap();
        data.seek(SeekFrom::Start(target.columns)).unwrap();
        let mut corrupted = [0_u8; 1];
        data.read_exact(&mut corrupted).unwrap();
        corrupted[0] ^= 0x5a;
        data.seek(SeekFrom::Start(target.columns)).unwrap();
        data.write_all(&corrupted).unwrap();
        data.sync_all().unwrap();

        let scale_part = output.join(".corruptible.sc.part");
        let scales = OpenOptions::new().write(true).open(&scale_part).unwrap();
        scales.set_len(5).unwrap();
        scales.sync_all().unwrap();

        let mut progress = CaptureStart::default();
        let report = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut progress,
        )
        .unwrap();
        assert_eq!(progress.resumed_rows, Some(1));
        assert_eq!(report.resumed_tensors, 1);
        assert_eq!(fs::read(output.join("corruptible.i8")).unwrap(), expected.0);
        assert_eq!(fs::read(output.join("corruptible.sc")).unwrap(), expected.1);
    }

    #[test]
    fn source_path_change_is_rejected_then_reauthenticated_on_resume() {
        let directory = TestDirectory::new();
        let source = directory.0.join("source");
        let output_parent = directory.0.join("int8");
        let output = output_parent.join("tensors");
        fs::create_dir(&source).unwrap();
        let target = synthetic_target(&source, "identity", 4, 8);
        let options = test_options(&directory.0);
        let mut replace = ReplaceSource {
            path: target.source.clone(),
            replaced: false,
        };

        let first = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut replace,
        )
        .unwrap_err();
        assert!(replace.replaced);
        assert!(first.to_string().contains("identity changed"));
        assert!(!output.join("identity.i8").exists());
        assert!(!output.join(format!("identity{RECEIPT_SUFFIX}")).exists());

        let mut refreshed = target.clone();
        refreshed.source_identity =
            FileIdentity::from_metadata(&sys_fs::lstat(&refreshed.source).unwrap());
        let expected = expected_outputs(&refreshed);
        let mut progress = CaptureStart::default();
        let report = convert_targets(
            std::slice::from_ref(&refreshed),
            &output_parent,
            &output,
            &options,
            &mut progress,
        )
        .unwrap();
        assert_eq!(progress.resumed_rows, Some(3));
        assert_eq!(report.resumed_tensors, 1);
        assert_eq!(fs::read(output.join("identity.i8")).unwrap(), expected.0);
        assert_eq!(fs::read(output.join("identity.sc")).unwrap(), expected.1);
    }

    #[test]
    fn receipt_rejects_a_same_length_replacement_of_a_published_file() {
        let directory = TestDirectory::new();
        let source = directory.0.join("source");
        let output_parent = directory.0.join("int8");
        let output = output_parent.join("tensors");
        fs::create_dir(&source).unwrap();
        let target = synthetic_target(&source, "tampered", 2, 8);
        let options = test_options(&directory.0);
        convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut NoProgress,
        )
        .unwrap();

        let final_path = output.join("tampered.i8");
        let mut replacement = fs::read(&final_path).unwrap();
        replacement[0] ^= 0x01;
        let replacement_path = output.join("replacement.i8");
        let mut file = File::create(&replacement_path).unwrap();
        file.write_all(&replacement).unwrap();
        file.sync_all().unwrap();
        drop(file);
        fs::rename(&replacement_path, &final_path).unwrap();

        let error = convert_targets(
            std::slice::from_ref(&target),
            &output_parent,
            &output,
            &options,
            &mut NoProgress,
        )
        .unwrap_err();
        assert!(error.to_string().contains("identity changed after receipt"));
    }
}
