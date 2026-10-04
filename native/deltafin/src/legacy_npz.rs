//! Exact, bounded migration of NumPy ZIP expert shards to Deltafin's raw span.
//!
//! This is the native replacement for `convert_npz_cache.py` and the legacy
//! `warm_expert_cache.py --convert-npz` path.  It accepts only the six canonical
//! uint8 NPY members, consumes each member through the ZIP reader to EOF (which
//! verifies its CRC32), and writes the identical C-order bytes expected by the
//! expert reader.  Publication is deliberately conservative: a unique partial
//! is fsync'd, read back and SHA-256 checked, atomically published without
//! replacing another file, directory-fsync'd, and only then may the NPZ source
//! be unlinked.

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use zip::{CompressionMethod, ZipArchive};

use crate::error::{DeltafinError, Result};
use crate::packfile::{Digest, DigestState, digest_open_file};
use crate::trusted_download::{fsync_directory, rename_noreplace, secure_create_new};
use crate::sys::fs::{self as sys_fs, Open};

pub const EXPERT_SPAN_BYTES: usize = 17_547_264;
pub const DEFAULT_CONVERSION_WORKERS: usize = 4;
pub const MAX_CONVERSION_WORKERS: usize = 16;

const MAX_ARCHIVE_BYTES: u64 = 64 << 20;
const MAX_NPY_HEADER_BYTES: usize = 64 << 10;
const MAX_NPY_PREFIX_BYTES: usize = 12;
const COPY_BUFFER_BYTES: usize = 1 << 20;
const TEMP_CREATE_ATTEMPTS: usize = 64;

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct MemberLayout {
    name: &'static str,
    offset: usize,
    bytes: usize,
    shape: (usize, usize),
}

const PACKED_BYTES: usize = 5_505_024;
const SCALE_BYTES: usize = 344_064;
const MEMBERS: [MemberLayout; 6] = [
    MemberLayout {
        name: "w1_p.npy",
        offset: 0,
        bytes: PACKED_BYTES,
        shape: (3072, 1792),
    },
    MemberLayout {
        name: "w1_s.npy",
        offset: PACKED_BYTES,
        bytes: SCALE_BYTES,
        shape: (3072, 112),
    },
    MemberLayout {
        name: "w2_p.npy",
        offset: PACKED_BYTES + SCALE_BYTES,
        bytes: PACKED_BYTES,
        shape: (3584, 1536),
    },
    MemberLayout {
        name: "w2_s.npy",
        offset: 2 * PACKED_BYTES + SCALE_BYTES,
        bytes: SCALE_BYTES,
        shape: (3584, 96),
    },
    MemberLayout {
        name: "w3_p.npy",
        offset: 2 * (PACKED_BYTES + SCALE_BYTES),
        bytes: PACKED_BYTES,
        shape: (3072, 1792),
    },
    MemberLayout {
        name: "w3_s.npy",
        offset: 3 * PACKED_BYTES + 2 * SCALE_BYTES,
        bytes: SCALE_BYTES,
        shape: (3072, 112),
    },
];

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ConversionOptions {
    pub workers: usize,
    /// Convert only the first N numerically sorted sources. `None` means all.
    pub limit: Option<usize>,
    pub keep_npz: bool,
    pub throttle: Duration,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            workers: DEFAULT_CONVERSION_WORKERS,
            limit: None,
            keep_npz: false,
            throttle: Duration::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ConversionReport {
    pub discovered_npz: usize,
    pub selected_npz: usize,
    pub converted_npz: usize,
    pub reused_raw: usize,
    pub deleted_npz: usize,
    pub bytes_converted: u64,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ConversionProgress {
    pub completed: usize,
    pub total: usize,
    pub converted: usize,
    pub reused: usize,
    pub failed: usize,
    pub bytes_converted: u64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct LegacyExpert {
    layer: u32,
    expert: u16,
    source_name: String,
    destination_name: String,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ConversionKind {
    Converted,
    Reused,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct OneConversion {
    kind: ConversionKind,
    deleted_npz: bool,
}

/// Convert all selected legacy cache entries with fixed worker and per-worker
/// memory bounds. Failures do not expose partial `.bin` files or remove their
/// `.npz`; independent jobs continue so a subsequent run resumes naturally.
pub fn convert_all(
    cache: &Path,
    options: ConversionOptions,
    progress_sink: &dyn Fn(ConversionProgress),
) -> Result<ConversionReport> {
    validate_options(options)?;
    validate_layout()?;
    inspect_cache_directory(cache)?;

    let mut jobs = scan(cache)?;
    let discovered_npz = jobs.len();
    if let Some(limit) = options.limit {
        jobs.truncate(limit.min(jobs.len()));
    }
    let selected_npz = jobs.len();
    if jobs.is_empty() {
        return Ok(ConversionReport {
            discovered_npz,
            selected_npz,
            ..ConversionReport::default()
        });
    }

    let next = AtomicUsize::new(0);
    let channel_capacity = options.workers.saturating_mul(2).max(1);
    let (sender, receiver) = mpsc::sync_channel(channel_capacity);
    let mut report = ConversionReport {
        discovered_npz,
        selected_npz,
        ..ConversionReport::default()
    };
    let mut failed = 0_usize;
    let mut first_failure = None::<String>;

    thread::scope(|scope| {
        for _ in 0..options.workers.min(jobs.len()) {
            let sender = sender.clone();
            let jobs = &jobs;
            let next = &next;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else {
                        break;
                    };
                    let result = convert_one(cache, job, options.keep_npz);
                    let succeeded = result.is_ok();
                    if sender.send((index, result)).is_err() {
                        break;
                    }
                    if succeeded && !options.throttle.is_zero() {
                        thread::sleep(options.throttle);
                    }
                }
            });
        }
        drop(sender);

        for completed in 1..=selected_npz {
            let (index, result) = receiver
                .recv()
                .expect("conversion workers retain a sender until every claimed job reports");
            match result {
                Ok(result) => {
                    match result.kind {
                        ConversionKind::Converted => {
                            report.converted_npz += 1;
                            report.bytes_converted = report
                                .bytes_converted
                                .checked_add(EXPERT_SPAN_BYTES as u64)
                                .expect("bounded expert count fits a u64 byte total");
                        }
                        ConversionKind::Reused => report.reused_raw += 1,
                    }
                    report.deleted_npz += usize::from(result.deleted_npz);
                }
                Err(error) => {
                    failed += 1;
                    if first_failure.is_none() {
                        first_failure = Some(format!("{}: {error}", jobs[index].source_name));
                    }
                }
            }
            progress_sink(ConversionProgress {
                completed,
                total: selected_npz,
                converted: report.converted_npz,
                reused: report.reused_raw,
                failed,
                bytes_converted: report.bytes_converted,
            });
        }
    });

    if failed != 0 {
        return Err(DeltafinError::new(format!(
            "native NPZ conversion failed for {failed}/{selected_npz} experts; {} converted and {} exact raw files reused; first failure: {}",
            report.converted_npz,
            report.reused_raw,
            first_failure.expect("a failed conversion records its first error"),
        )));
    }
    Ok(report)
}

fn validate_options(options: ConversionOptions) -> Result<()> {
    if !(1..=MAX_CONVERSION_WORKERS).contains(&options.workers) {
        return Err(DeltafinError::new(format!(
            "NPZ conversion workers must be in 1..={MAX_CONVERSION_WORKERS}"
        )));
    }
    Ok(())
}

fn validate_layout() -> Result<()> {
    let mut offset = 0_usize;
    for member in MEMBERS {
        if member.offset != offset {
            return Err(DeltafinError::new(
                "internal legacy NPZ layout is not contiguous",
            ));
        }
        let elements = member
            .shape
            .0
            .checked_mul(member.shape.1)
            .ok_or_else(|| DeltafinError::new("internal legacy NPZ shape overflowed"))?;
        if elements != member.bytes {
            return Err(DeltafinError::new(format!(
                "internal legacy NPZ shape drift for {}",
                member.name
            )));
        }
        offset = offset
            .checked_add(member.bytes)
            .ok_or_else(|| DeltafinError::new("internal legacy NPZ span overflowed"))?;
    }
    if offset != EXPERT_SPAN_BYTES {
        return Err(DeltafinError::new(format!(
            "internal legacy NPZ span {offset} != canonical {EXPERT_SPAN_BYTES}"
        )));
    }
    Ok(())
}

fn inspect_cache_directory(cache: &Path) -> Result<()> {
    let metadata = sys_fs::lstat(cache)
        .map_err(|error| io_error("inspect expert-cache directory", cache, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DeltafinError::new(format!(
            "expert cache is not a real directory: {}",
            cache.display()
        )));
    }
    let directory = Open::new()
        .read(true)
        .directory().no_follow()
        .open(cache)
        .map_err(|error| {
            io_error(
                "open expert-cache directory without following links",
                cache,
                error,
            )
        })?;
    let opened = sys_fs::fstat(&directory)
        .map_err(|error| io_error("stat opened expert-cache directory", cache, error))?;
    if !opened.is_dir() || opened.dev() != metadata.dev() || opened.ino() != metadata.ino() {
        return Err(DeltafinError::new(format!(
            "expert-cache directory changed while opening: {}",
            cache.display()
        )));
    }
    Ok(())
}

fn scan(cache: &Path) -> Result<Vec<LegacyExpert>> {
    let mut experts = Vec::new();
    let entries =
        fs::read_dir(cache).map_err(|error| io_error("scan legacy expert cache", cache, error))?;
    for entry in entries {
        let entry =
            entry.map_err(|error| io_error("read legacy expert-cache entry", cache, error))?;
        let Some(expert) = parse_source_name(&entry.file_name())? else {
            continue;
        };
        experts.push(expert);
    }
    experts.sort_unstable_by_key(|expert| (expert.layer, expert.expert));
    Ok(experts)
}

fn parse_source_name(name: &OsStr) -> Result<Option<LegacyExpert>> {
    let bytes = crate::sys::path::os_str_bytes(name);
    if !bytes.starts_with(b"L") || !bytes.ends_with(b".npz") {
        return Ok(None);
    }
    let core = &bytes[1..bytes.len() - 4];
    let Some(separator) = core.windows(2).position(|window| window == b"-E") else {
        return Ok(None);
    };
    let layer_bytes = &core[..separator];
    let expert_bytes = &core[separator + 2..];
    if layer_bytes.is_empty()
        || expert_bytes.is_empty()
        || !layer_bytes.iter().all(u8::is_ascii_digit)
        || !expert_bytes.iter().all(u8::is_ascii_digit)
    {
        return Ok(None);
    }
    let layer = parse_ascii_u32(layer_bytes, "legacy expert layer")?;
    let expert_u32 = parse_ascii_u32(expert_bytes, "legacy expert ID")?;
    if !(crate::experts::K3_MOE_LAYER_FIRST..=crate::experts::K3_MOE_LAYER_LAST).contains(&layer)
        || expert_u32 >= crate::experts::K3_EXPERTS_PER_LAYER as u32
    {
        return Err(DeltafinError::new(format!(
            "legacy expert filename is outside the K3 roster: {}",
            String::from_utf8_lossy(bytes)
        )));
    }
    let expert = u16::try_from(expert_u32)
        .map_err(|_| DeltafinError::new("legacy expert ID does not fit u16"))?;
    let canonical = format!("L{layer}-E{expert}.npz");
    if canonical.as_bytes() != bytes {
        return Err(DeltafinError::new(format!(
            "legacy expert filename is not canonical: {} (expected {canonical})",
            String::from_utf8_lossy(bytes)
        )));
    }
    Ok(Some(LegacyExpert {
        layer,
        expert,
        source_name: canonical,
        destination_name: format!("L{layer}-E{expert}.bin"),
    }))
}

fn parse_ascii_u32(bytes: &[u8], label: &str) -> Result<u32> {
    let mut value = 0_u32;
    for &byte in bytes {
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add(u32::from(byte - b'0')))
            .ok_or_else(|| DeltafinError::new(format!("{label} overflowed")))?;
    }
    Ok(value)
}

fn convert_one(cache: &Path, expert: &LegacyExpert, keep_npz: bool) -> Result<OneConversion> {
    let source_path = cache.join(&expert.source_name);
    let destination_path = cache.join(&expert.destination_name);
    let source_metadata = admit_source(&source_path)?;
    let source = open_source(&source_path, &source_metadata)?;
    cache_neutral(&source);
    let mut archive = ZipArchive::new(source).map_err(|error| {
        DeltafinError::new(format!(
            "open {} as a ZIP archive: {error}",
            source_path.display()
        ))
    })?;
    validate_archive(&mut archive, &source_path)?;

    let (temporary_path, mut temporary) = create_temporary(cache, &expert.destination_name)?;
    let mut temporary_guard = TemporaryGuard::new(temporary_path.clone());
    cache_neutral(&temporary);
    let mut expected_digest = DigestState::new();
    let mut written = 0_usize;
    let mut scratch = Vec::new();
    for layout in MEMBERS {
        if written != layout.offset {
            return Err(DeltafinError::new("internal NPZ output offset drifted"));
        }
        let mut member = archive.by_name(layout.name).map_err(|error| {
            DeltafinError::new(format!(
                "read ZIP member {} from {}: {error}",
                layout.name,
                source_path.display()
            ))
        })?;
        let maximum = layout
            .bytes
            .checked_add(MAX_NPY_PREFIX_BYTES + MAX_NPY_HEADER_BYTES)
            .ok_or_else(|| DeltafinError::new("NPY member byte limit overflowed"))?;
        let member_bytes = read_member_bounded(&mut member, maximum, &source_path, layout.name)?;
        let array = parse_npy(&member_bytes, layout, &source_path)?;
        if array.fortran_order {
            reorder_fortran_to_c(array.payload, layout.shape, &mut scratch)?;
            temporary.write_all(&scratch).map_err(|error| {
                io_error("write canonical expert temporary", &temporary_path, error)
            })?;
            expected_digest.update(&scratch);
        } else {
            temporary.write_all(array.payload).map_err(|error| {
                io_error("write canonical expert temporary", &temporary_path, error)
            })?;
            expected_digest.update(array.payload);
        }
        written = written
            .checked_add(layout.bytes)
            .ok_or_else(|| DeltafinError::new("canonical expert byte count overflowed"))?;
    }
    let source = archive.into_inner();
    let source_after = sys_fs::fstat(&source)
        .map_err(|error| io_error("restat opened legacy NPZ", &source_path, error))?;
    if !same_file(&source_metadata, &source_after) {
        return Err(DeltafinError::new(format!(
            "legacy NPZ changed while reading: {}",
            source_path.display()
        )));
    }
    if written != EXPERT_SPAN_BYTES {
        return Err(DeltafinError::new(format!(
            "converted expert length {written} != canonical {EXPERT_SPAN_BYTES}"
        )));
    }
    temporary
        .sync_all()
        .map_err(|error| io_error("fsync canonical expert temporary", &temporary_path, error))?;
    let temporary_metadata = sys_fs::fstat(&temporary)
        .map_err(|error| io_error("stat canonical expert temporary", &temporary_path, error))?;
    if !temporary_metadata.is_file() || temporary_metadata.len() != EXPERT_SPAN_BYTES as u64 {
        return Err(DeltafinError::new(format!(
            "{} is not an exact {EXPERT_SPAN_BYTES}-byte temporary",
            temporary_path.display()
        )));
    }
    drop_completed_cache(&temporary, EXPERT_SPAN_BYTES as u64);
    drop(temporary);

    let expected_digest = expected_digest.finalize();
    let temporary_verified = verify_exact_file(
        &temporary_path,
        EXPERT_SPAN_BYTES as u64,
        expected_digest,
        Some(&temporary_metadata),
        "converted expert readback",
    )?;

    let (kind, durable_bin) = match verify_existing_destination(&destination_path, expected_digest)?
    {
        Some(existing) => {
            remove_temporary(&mut temporary_guard)?;
            (ConversionKind::Reused, existing)
        }
        None => match rename_noreplace(&temporary_path, &destination_path) {
            Ok(()) => {
                temporary_guard.disarm();
                assert_path_identity(
                    &destination_path,
                    &temporary_verified.metadata,
                    "published expert",
                )?;
                (ConversionKind::Converted, temporary_verified)
            }
            Err(publish_error) => {
                match verify_existing_destination(&destination_path, expected_digest) {
                    Ok(Some(existing)) => {
                        remove_temporary(&mut temporary_guard)?;
                        (ConversionKind::Reused, existing)
                    }
                    Ok(None) => return Err(publish_error),
                    Err(existing_error) => {
                        return Err(DeltafinError::new(format!(
                            "{publish_error}; competing destination is not exact: {existing_error}"
                        )));
                    }
                }
            }
        },
    };

    durable_bin
        .file
        .sync_all()
        .map_err(|error| io_error("fsync verified raw expert", &destination_path, error))?;
    assert_path_identity(
        &destination_path,
        &durable_bin.metadata,
        "durable raw expert",
    )?;
    fsync_directory(cache)?;
    assert_path_identity(
        &destination_path,
        &durable_bin.metadata,
        "durable raw expert",
    )?;

    let deleted_npz = if keep_npz {
        false
    } else {
        assert_path_identity(&source_path, &source_after, "legacy NPZ before deletion")?;
        fs::remove_file(&source_path).map_err(|error| {
            io_error("remove durably converted legacy NPZ", &source_path, error)
        })?;
        fsync_directory(cache)?;
        true
    };
    drop_completed_cache(&source, source_after.len());
    drop_completed_cache(&durable_bin.file, EXPERT_SPAN_BYTES as u64);
    Ok(OneConversion { kind, deleted_npz })
}

fn admit_source(path: &Path) -> Result<sys_fs::Stat> {
    let metadata =
        sys_fs::lstat(path).map_err(|error| io_error("inspect legacy NPZ", path, error))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > MAX_ARCHIVE_BYTES
    {
        return Err(DeltafinError::new(format!(
            "legacy NPZ must be a regular non-symlink file no larger than {MAX_ARCHIVE_BYTES} bytes: {}",
            path.display()
        )));
    }
    Ok(metadata)
}

fn open_source(path: &Path, expected: &sys_fs::Stat) -> Result<File> {
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open legacy NPZ without following links", path, error))?;
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat opened legacy NPZ", path, error))?;
    if !same_file(expected, &opened) {
        return Err(DeltafinError::new(format!(
            "legacy NPZ changed while opening: {}",
            path.display()
        )));
    }
    Ok(file)
}

fn validate_archive(archive: &mut ZipArchive<File>, path: &Path) -> Result<()> {
    if archive.offset() != 0 || archive.len() != MEMBERS.len() {
        return Err(DeltafinError::new(format!(
            "{} must contain exactly the six canonical NPY members and no leading payload",
            path.display()
        )));
    }
    if archive.has_overlapping_files().map_err(|error| {
        DeltafinError::new(format!("inspect ZIP ranges in {}: {error}", path.display()))
    })? {
        return Err(DeltafinError::new(format!(
            "{} contains overlapping ZIP members",
            path.display()
        )));
    }
    let mut seen = [false; MEMBERS.len()];
    for index in 0..archive.len() {
        let member = archive.by_index(index).map_err(|error| {
            DeltafinError::new(format!(
                "inspect ZIP member {index} in {}: {error}",
                path.display()
            ))
        })?;
        let name = member.name_raw();
        let Some(expected_index) = MEMBERS
            .iter()
            .position(|expected| expected.name.as_bytes() == name)
        else {
            return Err(DeltafinError::new(format!(
                "{} contains unexpected ZIP member {:?}",
                path.display(),
                String::from_utf8_lossy(name)
            )));
        };
        if std::mem::replace(&mut seen[expected_index], true) {
            return Err(DeltafinError::new(format!(
                "{} contains duplicate ZIP member {}",
                path.display(),
                MEMBERS[expected_index].name
            )));
        }
        if member.encrypted()
            || !member.is_file()
            || member.enclosed_name().as_deref() != Some(Path::new(MEMBERS[expected_index].name))
            || !matches!(
                member.compression(),
                CompressionMethod::Stored | CompressionMethod::Deflated
            )
        {
            return Err(DeltafinError::new(format!(
                "{} contains an unsafe or unsupported ZIP member {}",
                path.display(),
                MEMBERS[expected_index].name
            )));
        }
        let maximum = MEMBERS[expected_index]
            .bytes
            .checked_add(MAX_NPY_PREFIX_BYTES + MAX_NPY_HEADER_BYTES)
            .ok_or_else(|| DeltafinError::new("NPY member byte limit overflowed"))?
            as u64;
        if member.size() > maximum || member.compressed_size() > MAX_ARCHIVE_BYTES {
            return Err(DeltafinError::new(format!(
                "{}:{} exceeds the bounded NPY member size",
                path.display(),
                MEMBERS[expected_index].name
            )));
        }
    }
    if seen.iter().any(|present| !present) {
        return Err(DeltafinError::new(format!(
            "{} is missing one or more canonical NPY members",
            path.display()
        )));
    }
    Ok(())
}

fn read_member_bounded<R: Read>(
    member: &mut R,
    maximum: usize,
    archive_path: &Path,
    member_name: &str,
) -> Result<Vec<u8>> {
    let limit = u64::try_from(maximum)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| DeltafinError::new("NPY member read limit overflowed"))?;
    let mut bytes = Vec::with_capacity(maximum.min(COPY_BUFFER_BYTES));
    member
        .take(limit)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            DeltafinError::new(format!(
                "read and CRC-check {member_name} from {}: {error}",
                archive_path.display()
            ))
        })?;
    if bytes.len() > maximum {
        return Err(DeltafinError::new(format!(
            "{}:{member_name} exceeds the bounded NPY member size",
            archive_path.display()
        )));
    }
    Ok(bytes)
}

#[derive(Debug, Clone, Copy)]
struct ParsedNpy<'a> {
    fortran_order: bool,
    payload: &'a [u8],
}

fn parse_npy<'a>(bytes: &'a [u8], layout: MemberLayout, path: &Path) -> Result<ParsedNpy<'a>> {
    let invalid = |message: &str| {
        DeltafinError::new(format!("{}:{}: {message}", path.display(), layout.name))
    };
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(invalid("missing NPY magic"));
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (prefix_bytes, header_bytes) = match (major, minor) {
        (1, 0) => {
            let length = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (10_usize, length)
        }
        (2 | 3, 0) => {
            if bytes.len() < 12 {
                return Err(invalid("truncated NPY v2/v3 prefix"));
            }
            let length = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
            (12_usize, length)
        }
        _ => return Err(invalid("unsupported NPY format version")),
    };
    if header_bytes == 0 || header_bytes > MAX_NPY_HEADER_BYTES {
        return Err(invalid("NPY header exceeds the safety bound"));
    }
    let payload_offset = prefix_bytes
        .checked_add(header_bytes)
        .ok_or_else(|| invalid("NPY header offset overflowed"))?;
    let expected_total = payload_offset
        .checked_add(layout.bytes)
        .ok_or_else(|| invalid("NPY payload length overflowed"))?;
    if bytes.len() != expected_total {
        return Err(invalid("NPY member has the wrong payload length"));
    }
    let header = &bytes[prefix_bytes..payload_offset];
    if !header.ends_with(b"\n") || !header.is_ascii() {
        return Err(invalid(
            "NPY header must be canonical ASCII ending in newline",
        ));
    }
    let parsed = HeaderParser::new(header)
        .parse()
        .map_err(|message| invalid(&message))?;
    if !matches!(parsed.descr.as_slice(), b"|u1" | b"<u1" | b">u1" | b"=u1") {
        return Err(invalid("NPY dtype is not uint8"));
    }
    if parsed.shape != layout.shape {
        return Err(invalid("NPY array has the wrong shape"));
    }
    Ok(ParsedNpy {
        fortran_order: parsed.fortran_order,
        payload: &bytes[payload_offset..],
    })
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ParsedHeader {
    descr: Vec<u8>,
    fortran_order: bool,
    shape: (usize, usize),
}

struct HeaderParser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> HeaderParser<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    fn parse(mut self) -> std::result::Result<ParsedHeader, String> {
        self.skip_whitespace();
        self.expect(b'{')?;
        let mut descr = None;
        let mut fortran_order = None;
        let mut shape = None;
        loop {
            self.skip_whitespace();
            if self.consume(b'}') {
                break;
            }
            let key = self.quoted()?.to_vec();
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            match key.as_slice() {
                b"descr" => {
                    if descr.is_some() {
                        return Err("duplicate descr field in NPY header".into());
                    }
                    descr = Some(self.quoted()?.to_vec());
                }
                b"fortran_order" => {
                    if fortran_order.is_some() {
                        return Err("duplicate fortran_order field in NPY header".into());
                    }
                    fortran_order = Some(self.boolean()?);
                }
                b"shape" => {
                    if shape.is_some() {
                        return Err("duplicate shape field in NPY header".into());
                    }
                    shape = Some(self.shape()?);
                }
                _ => return Err("unexpected field in NPY header".into()),
            }
            self.skip_whitespace();
            if self.consume(b',') {
                continue;
            }
            if self.peek() == Some(b'}') {
                continue;
            }
            return Err("expected ',' or '}' in NPY header".into());
        }
        self.skip_whitespace();
        if self.position != self.input.len() {
            return Err("trailing non-whitespace bytes in NPY header".into());
        }
        Ok(ParsedHeader {
            descr: descr.ok_or_else(|| "missing descr field in NPY header".to_owned())?,
            fortran_order: fortran_order
                .ok_or_else(|| "missing fortran_order field in NPY header".to_owned())?,
            shape: shape.ok_or_else(|| "missing shape field in NPY header".to_owned())?,
        })
    }

    fn quoted(&mut self) -> std::result::Result<&'a [u8], String> {
        let quote = self
            .peek()
            .filter(|byte| matches!(byte, b'\'' | b'"'))
            .ok_or_else(|| "expected quoted string in NPY header".to_owned())?;
        self.position += 1;
        let start = self.position;
        while let Some(byte) = self.peek() {
            if byte == quote {
                let value = &self.input[start..self.position];
                self.position += 1;
                return Ok(value);
            }
            if byte == b'\\' || !byte.is_ascii_graphic() {
                return Err("unsupported quoted string in NPY header".into());
            }
            self.position += 1;
        }
        Err("unterminated quoted string in NPY header".into())
    }

    fn boolean(&mut self) -> std::result::Result<bool, String> {
        if self.consume_slice(b"True") {
            Ok(true)
        } else if self.consume_slice(b"False") {
            Ok(false)
        } else {
            Err("expected Python boolean in NPY header".into())
        }
    }

    fn shape(&mut self) -> std::result::Result<(usize, usize), String> {
        self.expect(b'(')?;
        self.skip_whitespace();
        let rows = self.positive_integer()?;
        self.skip_whitespace();
        self.expect(b',')?;
        self.skip_whitespace();
        let columns = self.positive_integer()?;
        self.skip_whitespace();
        let _ = self.consume(b',');
        self.skip_whitespace();
        self.expect(b')')?;
        Ok((rows, columns))
    }

    fn positive_integer(&mut self) -> std::result::Result<usize, String> {
        let start = self.position;
        let mut value = 0_usize;
        while let Some(byte) = self.peek().filter(u8::is_ascii_digit) {
            value = value
                .checked_mul(10)
                .and_then(|value| value.checked_add(usize::from(byte - b'0')))
                .ok_or_else(|| "integer overflow in NPY header".to_owned())?;
            self.position += 1;
        }
        if self.position == start || value == 0 {
            return Err("expected positive integer in NPY shape".into());
        }
        Ok(value)
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.position += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> std::result::Result<(), String> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err(format!("expected {:?} in NPY header", expected as char))
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn consume_slice(&mut self, expected: &[u8]) -> bool {
        if self.input[self.position..].starts_with(expected) {
            self.position += expected.len();
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }
}

fn reorder_fortran_to_c(payload: &[u8], shape: (usize, usize), output: &mut Vec<u8>) -> Result<()> {
    let elements = shape
        .0
        .checked_mul(shape.1)
        .ok_or_else(|| DeltafinError::new("Fortran-order NPY shape overflowed"))?;
    if payload.len() != elements {
        return Err(DeltafinError::new(
            "Fortran-order NPY payload length drifted",
        ));
    }
    output.clear();
    output.resize(elements, 0);
    for row in 0..shape.0 {
        let c_row = row * shape.1;
        for column in 0..shape.1 {
            output[c_row + column] = payload[column * shape.0 + row];
        }
    }
    Ok(())
}

fn create_temporary(cache: &Path, destination_name: &str) -> Result<(PathBuf, File)> {
    for _ in 0..TEMP_CREATE_ATTEMPTS {
        let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = cache.join(format!(
            ".{destination_name}.deltafin-convtmp-{}-{nonce}",
            std::process::id()
        ));
        match secure_create_new(&path, 0o644) {
            Ok(file) => return Ok((path, file)),
            Err(_error) if path.exists() => continue,
            Err(error) => return Err(error),
        }
    }
    Err(DeltafinError::new(format!(
        "could not allocate a unique conversion temporary in {}",
        cache.display()
    )))
}

struct VerifiedFile {
    file: File,
    metadata: sys_fs::Stat,
}

fn verify_exact_file(
    path: &Path,
    expected_bytes: u64,
    expected_digest: Digest,
    expected_identity: Option<&sys_fs::Stat>,
    label: &str,
) -> Result<VerifiedFile> {
    let before = sys_fs::lstat(path)
        .map_err(|error| io_error(&format!("inspect {label}"), path, error))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() != expected_bytes {
        return Err(DeltafinError::new(format!(
            "{label} is not an exact regular {expected_bytes}-byte file: {}",
            path.display()
        )));
    }
    if expected_identity.is_some_and(|expected| !same_file(expected, &before)) {
        return Err(DeltafinError::new(format!(
            "{label} changed before readback: {}",
            path.display()
        )));
    }
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| {
            io_error(
                &format!("open {label} without following links"),
                path,
                error,
            )
        })?;
    cache_neutral(&file);
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error(&format!("stat opened {label}"), path, error))?;
    if !same_file(&before, &opened) {
        return Err(DeltafinError::new(format!(
            "{label} changed while opening: {}",
            path.display()
        )));
    }
    let digest = digest_open_file(&file, path)
        .map_err(|error| DeltafinError::new(format!("hash {label} {}: {error}", path.display())))?;
    let after = sys_fs::fstat(&file)
        .map_err(|error| io_error(&format!("restat {label}"), path, error))?;
    if !same_file(&opened, &after) || digest != expected_digest {
        return Err(DeltafinError::new(format!(
            "{label} failed exact SHA-256 readback: {}",
            path.display()
        )));
    }
    drop_completed_cache(&file, expected_bytes);
    Ok(VerifiedFile {
        file,
        metadata: after,
    })
}

fn verify_existing_destination(
    path: &Path,
    expected_digest: Digest,
) -> Result<Option<VerifiedFile>> {
    match sys_fs::lstat(path) {
        Ok(_) => verify_exact_file(
            path,
            EXPERT_SPAN_BYTES as u64,
            expected_digest,
            None,
            "existing raw expert",
        )
        .map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io_error("inspect existing raw expert", path, error)),
    }
}

fn assert_path_identity(path: &Path, expected: &sys_fs::Stat, label: &str) -> Result<()> {
    let actual = sys_fs::lstat(path)
        .map_err(|error| io_error(&format!("inspect {label}"), path, error))?;
    // A successful rename can legitimately advance ctime while preserving the
    // inode. Publication identity therefore uses the immutable descriptor
    // tuple, while source/readback checks above additionally pin timestamps.
    if actual.file_type().is_symlink() || !same_inode_and_length(expected, &actual) {
        return Err(DeltafinError::new(format!(
            "{label} path changed unexpectedly: {}",
            path.display()
        )));
    }
    Ok(())
}

struct TemporaryGuard {
    path: Option<PathBuf>,
}

impl TemporaryGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for TemporaryGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn remove_temporary(guard: &mut TemporaryGuard) -> Result<()> {
    let path = guard
        .path
        .as_ref()
        .expect("a live conversion temporary has a path");
    fs::remove_file(path)
        .map_err(|error| io_error("remove verified conversion temporary", path, error))?;
    guard.disarm();
    Ok(())
}

fn same_inode_and_length(left: &sys_fs::Stat, right: &sys_fs::Stat) -> bool {
    left.is_file()
        && right.is_file()
        && left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
}

fn same_file(left: &sys_fs::Stat, right: &sys_fs::Stat) -> bool {
    same_inode_and_length(left, right)
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
}

fn cache_neutral(file: &File) {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd as _;
        const F_NOCACHE: i32 = 48;
        // SAFETY: the descriptor is live and F_NOCACHE accepts an integer.
        let _ = unsafe { libc::fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
    }
    #[cfg(not(target_os = "macos"))]
    let _ = file;
}

fn drop_completed_cache(file: &File, bytes: u64) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd as _;
        if let Ok(length) = i64::try_from(bytes) {
            // SAFETY: the descriptor remains live for this best-effort advisory.
            let _ = unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, length, libc::POSIX_FADV_DONTNEED)
            };
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, bytes);
}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    static NEXT_TEST: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "deltafin-legacy-npz-{nonce}-{}-{}",
                std::process::id(),
                NEXT_TEST.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn npy_header(shape: (usize, usize), fortran: bool) -> Vec<u8> {
        let dictionary = format!(
            "{{'descr': '|u1', 'fortran_order': {}, 'shape': ({}, {}), }}",
            if fortran { "True" } else { "False" },
            shape.0,
            shape.1
        );
        let prefix = 10_usize;
        let padding = (64 - ((prefix + dictionary.len() + 1) % 64)) % 64;
        let header_length = dictionary.len() + padding + 1;
        let mut bytes = Vec::with_capacity(prefix + header_length);
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(header_length as u16).to_le_bytes());
        bytes.extend_from_slice(dictionary.as_bytes());
        bytes.resize(bytes.len() + padding, b' ');
        bytes.push(b'\n');
        bytes
    }

    fn logical_value(member: usize, row: usize, column: usize) -> u8 {
        ((member * 37 + row * 3 + column * 5) % 251) as u8
    }

    fn write_valid_npz(path: &Path, fortran_member: Option<usize>) {
        let file = File::create(path).unwrap();
        let mut writer = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        // Deliberately store the ZIP directory in reverse order: conversion
        // must use the canonical w1p,w1s,w2p,w2s,w3p,w3s layout, never archive
        // insertion order.
        for member_index in (0..MEMBERS.len()).rev() {
            let layout = &MEMBERS[member_index];
            writer.start_file(layout.name, options).unwrap();
            writer
                .write_all(&npy_header(
                    layout.shape,
                    fortran_member == Some(member_index),
                ))
                .unwrap();
            if fortran_member == Some(member_index) {
                let mut column_bytes = vec![0_u8; layout.shape.0];
                for column in 0..layout.shape.1 {
                    for (row, value) in column_bytes.iter_mut().enumerate() {
                        *value = logical_value(member_index, row, column);
                    }
                    writer.write_all(&column_bytes).unwrap();
                }
            } else {
                let mut row_bytes = vec![0_u8; layout.shape.1];
                for row in 0..layout.shape.0 {
                    for (column, value) in row_bytes.iter_mut().enumerate() {
                        *value = logical_value(member_index, row, column);
                    }
                    writer.write_all(&row_bytes).unwrap();
                }
            }
        }
        writer.finish().unwrap().sync_all().unwrap();
    }

    fn assert_raw_bytes(path: &Path) {
        let mut file = File::open(path).unwrap();
        for (member_index, layout) in MEMBERS.iter().enumerate() {
            let mut row_bytes = vec![0_u8; layout.shape.1];
            for row in 0..layout.shape.0 {
                file.read_exact(&mut row_bytes).unwrap();
                for (column, &value) in row_bytes.iter().enumerate() {
                    assert_eq!(value, logical_value(member_index, row, column));
                }
            }
        }
        let mut extra = [0_u8; 1];
        assert_eq!(file.read(&mut extra).unwrap(), 0);
    }

    fn flip_first_member_payload(path: &Path) -> u64 {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let member = archive.by_name(MEMBERS[0].name).unwrap();
        let offset = member.data_start() + (member.compressed_size() / 2).max(1);
        drop(member);
        let mut file = archive.into_inner();
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut byte = [0_u8; 1];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[byte[0] ^ 0xff]).unwrap();
        file.sync_all().unwrap();
        offset
    }

    #[test]
    fn exact_conversion_reorders_fortran_verifies_crc_and_resumes_safely() {
        let root = TestDirectory::new();
        let source = root.0.join("L1-E7.npz");
        let destination = root.0.join("L1-E7.bin");
        write_valid_npz(&source, Some(1));

        let options = ConversionOptions {
            workers: 1,
            keep_npz: true,
            ..ConversionOptions::default()
        };
        let report = convert_all(&root.0, options, &|_| {}).unwrap();
        assert_eq!(report.converted_npz, 1);
        assert_eq!(report.reused_raw, 0);
        assert_eq!(report.deleted_npz, 0);
        assert_eq!(
            fs::metadata(&destination).unwrap().len(),
            EXPERT_SPAN_BYTES as u64
        );
        assert_raw_bytes(&destination);

        let flipped_bin_offset = 123_456_u64;
        let mut bin = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&destination)
            .unwrap();
        bin.seek(SeekFrom::Start(flipped_bin_offset)).unwrap();
        let mut original = [0_u8; 1];
        bin.read_exact(&mut original).unwrap();
        bin.seek(SeekFrom::Start(flipped_bin_offset)).unwrap();
        bin.write_all(&[original[0] ^ 0xff]).unwrap();
        bin.sync_all().unwrap();
        drop(bin);
        assert!(convert_all(&root.0, options, &|_| {}).is_err());
        assert!(source.exists());
        let mut bin = OpenOptions::new().write(true).open(&destination).unwrap();
        bin.seek(SeekFrom::Start(flipped_bin_offset)).unwrap();
        bin.write_all(&original).unwrap();
        bin.sync_all().unwrap();

        let crc_offset = flip_first_member_payload(&source);
        assert!(convert_all(&root.0, options, &|_| {}).is_err());
        assert!(source.exists());
        let _ = flip_first_member_payload(&source);
        assert!(crc_offset > 0);

        let report = convert_all(
            &root.0,
            ConversionOptions {
                workers: 1,
                keep_npz: false,
                ..ConversionOptions::default()
            },
            &|_| {},
        )
        .unwrap();
        assert_eq!(report.converted_npz, 0);
        assert_eq!(report.reused_raw, 1);
        assert_eq!(report.deleted_npz, 1);
        assert!(!source.exists());
        assert_raw_bytes(&destination);
        assert!(fs::read_dir(&root.0).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("convtmp")
        }));
    }

    #[test]
    fn archive_shape_names_and_source_file_type_fail_closed() {
        let root = TestDirectory::new();
        let extra = root.0.join("L2-E3.npz");
        let file = File::create(&extra).unwrap();
        let mut writer = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        for layout in MEMBERS {
            writer.start_file(layout.name, options).unwrap();
            writer.write_all(b"not an npy").unwrap();
        }
        writer.start_file("extra.npy", options).unwrap();
        writer.write_all(b"extra").unwrap();
        writer.finish().unwrap();
        assert!(
            convert_all(
                &root.0,
                ConversionOptions {
                    workers: 1,
                    ..ConversionOptions::default()
                },
                &|_| {},
            )
            .is_err()
        );
        assert!(extra.exists());
        assert!(!root.0.join("L2-E3.bin").exists());

        fs::remove_file(&extra).unwrap();
        #[cfg(unix)]
        {
            let target = root.0.join("target");
            fs::write(&target, b"untouched").unwrap();
            crate::sys::fs::symlink(&target, root.0.join("L2-E3.npz")).unwrap();
            assert!(
                convert_all(
                    &root.0,
                    ConversionOptions {
                        workers: 1,
                        ..ConversionOptions::default()
                    },
                    &|_| {},
                )
                .is_err()
            );
            assert_eq!(fs::read(target).unwrap(), b"untouched");
        }
    }

    #[test]
    fn header_parser_is_exact_bounded_and_order_independent() {
        let parsed =
            HeaderParser::new(b"  {'shape': (3, 4), 'descr': '|u1', 'fortran_order': False, }\n")
                .parse()
                .unwrap();
        assert_eq!(
            parsed,
            ParsedHeader {
                descr: b"|u1".to_vec(),
                fortran_order: false,
                shape: (3, 4),
            }
        );
        assert!(
            HeaderParser::new(
                b"{'descr': '|u1', 'fortran_order': False, 'shape': (3, 4), 'x': 1}\n"
            )
            .parse()
            .is_err()
        );
        assert!(
            HeaderParser::new(
                b"{'descr': '|u1', 'descr': '|u1', 'fortran_order': False, 'shape': (3, 4)}\n"
            )
            .parse()
            .is_err()
        );
        assert!(parse_source_name(OsStr::new("L01-E7.npz")).is_err());
        assert!(
            parse_source_name(OsStr::new("notes.npz"))
                .unwrap()
                .is_none()
        );
        assert!(
            validate_options(ConversionOptions {
                workers: MAX_CONVERSION_WORKERS + 1,
                ..ConversionOptions::default()
            })
            .is_err()
        );
    }
}
