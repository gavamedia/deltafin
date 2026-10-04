//! Exact Rust codec for Deltafin's lossless K3 expert scale sidecars.
//!
//! `K3SC4V2` never quantizes an expert value. The packed E2M1 planes remain
//! byte-for-byte in the canonical raw expert and each E8M0 scale byte is stored
//! as an exact four-bit delta from a per-matrix base. This module deliberately
//! mirrors the already-deployed format instead of introducing a Rust-specific
//! variant, so existing sidecars and compiled kernels remain interoperable.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::error::{DeltafinError, Result};
use crate::packfile::{Digest, DigestState, digest_bytes};
use crate::sys::fs::{self as sys_fs, Open};

pub const MAGIC: [u8; 8] = *b"K3SC4V2\0";
pub const VERSION: u32 = 2;
pub const LAYOUT_ID: u32 = 2;
pub const HEADER_BYTES: usize = 8_192;
pub const SOURCE_BYTES: usize = 17_547_264;
pub const PACKED_BYTES: usize = 5_505_024;
pub const SCALE_BYTES: usize = 344_064;
pub const SCALE4_BYTES: usize = SCALE_BYTES / 2;
pub const BLOB_BYTES: usize = 17_039_360;
pub const FILE_BYTES: usize = HEADER_BYTES + 3 * SCALE4_BYTES;
pub const MANIFEST_NAME: &str = "scale4-manifest.json";
pub const MANIFEST_SCHEMA: &str = "deltafin-expert-scale4-v2";
pub const MANIFEST_VERSION: u32 = 2;

pub mod convert;
pub mod manifest;

const HEADER_PREFIX_END: usize = 120;
const TABLE_OFFSETS: [usize; 3] = [128, 192, 256];
const TABLE_BYTES: usize = 16 * 4;
const SIDECAR_TABLE_OFFSET: usize = 320;
const SIDECAR_TABLE_END: usize = SIDECAR_TABLE_OFFSET + 6 * 4;

/// Canonical raw component extents, alternating packed and scale planes.
pub const RAW_LAYOUT: [(u64, usize); 6] = [
    (0, PACKED_BYTES),
    (PACKED_BYTES as u64, SCALE_BYTES),
    ((PACKED_BYTES + SCALE_BYTES) as u64, PACKED_BYTES),
    ((2 * PACKED_BYTES + SCALE_BYTES) as u64, SCALE_BYTES),
    ((2 * (PACKED_BYTES + SCALE_BYTES)) as u64, PACKED_BYTES),
    ((3 * PACKED_BYTES + 2 * SCALE_BYTES) as u64, SCALE_BYTES),
];

/// Canonical assembled in-memory offsets and sizes: w1p, w1s4, w2p, w2s4,
/// w3p, w3s4. These twelve words are embedded in every header.
pub const COMPACT_LAYOUT: [(u32, u32); 6] = [
    (8_192, 5_505_024),
    (5_513_216, 172_032),
    (5_685_248, 5_505_024),
    (11_190_272, 172_032),
    (11_362_304, 5_505_024),
    (16_867_328, 172_032),
];

/// Canonical on-disk scale-plane offsets and sizes.
pub const SIDECAR_LAYOUT: [(u32, u32); 3] =
    [(8_192, 172_032), (180_224, 172_032), (352_256, 172_032)];

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Scale4Header {
    pub source_sha256: Digest,
    pub bases: [u8; 3],
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EncodedScalePlane {
    pub base: u8,
    pub packed: Box<[u8]>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SourceIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EncodedExpert {
    pub source: PathBuf,
    pub source_identity: SourceIdentity,
    pub source_sha256: Digest,
    pub record_sha256: Digest,
    pub bases: [u8; 3],
    pub record: Box<[u8]>,
}

/// Return the exact IEEE-754 float32 bit patterns for E8M0 bytes
/// `base..=base+15`. Requiring all 16 entries to be valid keeps native lookup
/// tables safe even when malformed payload data selects an unused nibble.
pub fn exponent_table(base: u8) -> Result<[u32; 16]> {
    if base > 240 {
        return Err(invalid(format!(
            "scale base {base} cannot populate a complete uint8 delta table"
        )));
    }
    let mut table = [0_u32; 16];
    for (delta, entry) in table.iter_mut().enumerate() {
        *entry = u32::from(base + delta as u8) << 23;
    }
    Ok(table)
}

/// Pack low-even/high-odd exact uint4 deltas. When `base` is absent, the
/// minimum source byte is the canonical base used by the Python-produced V2
/// corpus.
pub fn pack_scale4(values: &[u8], base: Option<u8>) -> Result<EncodedScalePlane> {
    let (&first, rest) = values
        .split_first()
        .ok_or_else(|| invalid("a scale stream may not be empty"))?;
    let mut minimum = first;
    let mut maximum = first;
    for &value in rest {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    let selected = base.unwrap_or(minimum);
    exponent_table(selected)?;
    if minimum < selected || maximum.saturating_sub(selected) > 15 {
        return Err(invalid(format!(
            "scale range {minimum}..{maximum} does not fit base {selected} plus uint4"
        )));
    }

    let mut packed = vec![0_u8; values.len().div_ceil(2)];
    for (index, &value) in values.iter().enumerate() {
        let delta = value - selected;
        if index & 1 == 0 {
            packed[index / 2] = delta;
        } else {
            packed[index / 2] |= delta << 4;
        }
    }
    Ok(EncodedScalePlane {
        base: selected,
        packed: packed.into_boxed_slice(),
    })
}

pub fn unpack_scale4(packed: &[u8], base: u8, count: usize) -> Result<Box<[u8]>> {
    if packed.len() != count.div_ceil(2) {
        return Err(invalid(format!(
            "scale4 extent {} cannot reconstruct {count} bytes",
            packed.len()
        )));
    }
    exponent_table(base)?;
    let mut values = vec![0_u8; count];
    for (index, value) in values.iter_mut().enumerate() {
        let byte = packed[index / 2];
        let delta = if index & 1 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        };
        *value = base
            .checked_add(delta)
            .ok_or_else(|| invalid("decoded scale exceeds uint8"))?;
    }
    Ok(values.into_boxed_slice())
}

pub fn build_header(source_sha256: Digest, bases: [u8; 3]) -> Result<Box<[u8; HEADER_BYTES]>> {
    let tables = [
        exponent_table(bases[0])?,
        exponent_table(bases[1])?,
        exponent_table(bases[2])?,
    ];
    let mut header = Box::new([0_u8; HEADER_BYTES]);
    header[..8].copy_from_slice(&MAGIC);
    put_u32(&mut header[..], 8, VERSION);
    put_u32(&mut header[..], 12, HEADER_BYTES as u32);
    put_u64(&mut header[..], 16, BLOB_BYTES as u64);
    put_u64(&mut header[..], 24, SOURCE_BYTES as u64);
    header[32..64].copy_from_slice(&source_sha256);
    header[64..67].copy_from_slice(&bases);

    let mut cursor = 72;
    for (offset, size) in COMPACT_LAYOUT {
        put_u32(&mut header[..], cursor, offset);
        put_u32(&mut header[..], cursor + 4, size);
        cursor += 8;
    }
    debug_assert_eq!(cursor, HEADER_PREFIX_END);
    for (offset, table) in TABLE_OFFSETS.into_iter().zip(tables) {
        for (slot, bits) in table.into_iter().enumerate() {
            put_u32(&mut header[..], offset + slot * 4, bits);
        }
    }
    cursor = SIDECAR_TABLE_OFFSET;
    for (offset, size) in SIDECAR_LAYOUT {
        put_u32(&mut header[..], cursor, offset);
        put_u32(&mut header[..], cursor + 4, size);
        cursor += 8;
    }
    debug_assert_eq!(cursor, SIDECAR_TABLE_END);
    Ok(header)
}

pub fn parse_header(header: &[u8]) -> Result<Scale4Header> {
    if header.len() != HEADER_BYTES {
        return Err(invalid(format!(
            "compact header length {} != {HEADER_BYTES}",
            header.len()
        )));
    }
    if header[..8] != MAGIC {
        return Err(invalid("bad scale4 magic"));
    }
    expect_u32(header, 8, VERSION, "format version")?;
    expect_u32(header, 12, HEADER_BYTES as u32, "header length")?;
    expect_u64(header, 16, BLOB_BYTES as u64, "blob length")?;
    expect_u64(header, 24, SOURCE_BYTES as u64, "source length")?;

    let mut source_sha256 = [0_u8; 32];
    source_sha256.copy_from_slice(&header[32..64]);
    let bases = [header[64], header[65], header[66]];
    expect_zeroes(&header[67..72], "compact-header alignment padding")?;

    let mut cursor = 72;
    for (expected_offset, expected_size) in COMPACT_LAYOUT {
        expect_u32(header, cursor, expected_offset, "compact payload offset")?;
        expect_u32(header, cursor + 4, expected_size, "compact payload size")?;
        cursor += 8;
    }
    expect_zeroes(
        &header[HEADER_PREFIX_END..TABLE_OFFSETS[0]],
        "reserved compact-header prefix",
    )?;
    for (matrix, (offset, base)) in TABLE_OFFSETS.into_iter().zip(bases).enumerate() {
        let expected = exponent_table(base)?;
        for (slot, bits) in expected.into_iter().enumerate() {
            expect_u32(
                header,
                offset + slot * 4,
                bits,
                match matrix {
                    0 => "w1 exponent table",
                    1 => "w2 exponent table",
                    _ => "w3 exponent table",
                },
            )?;
        }
    }
    expect_zeroes(
        &header[TABLE_OFFSETS[2] + TABLE_BYTES..SIDECAR_TABLE_OFFSET],
        "reserved compact-header table padding",
    )?;
    cursor = SIDECAR_TABLE_OFFSET;
    for (expected_offset, expected_size) in SIDECAR_LAYOUT {
        expect_u32(header, cursor, expected_offset, "sidecar payload offset")?;
        expect_u32(header, cursor + 4, expected_size, "sidecar payload size")?;
        cursor += 8;
    }
    expect_zeroes(
        &header[SIDECAR_TABLE_END..],
        "reserved compact-header bytes",
    )?;
    Ok(Scale4Header {
        source_sha256,
        bases,
    })
}

pub fn build_record(source_sha256: Digest, planes: [&EncodedScalePlane; 3]) -> Result<Box<[u8]>> {
    for plane in planes {
        if plane.packed.len() != SCALE4_BYTES {
            return Err(invalid(format!(
                "scale4 plane length {} != {SCALE4_BYTES}",
                plane.packed.len()
            )));
        }
    }
    let bases = [planes[0].base, planes[1].base, planes[2].base];
    let header = build_header(source_sha256, bases)?;
    let mut record = vec![0_u8; FILE_BYTES];
    record[..HEADER_BYTES].copy_from_slice(header.as_slice());
    for (plane, (offset, size)) in planes.into_iter().zip(SIDECAR_LAYOUT) {
        let offset = offset as usize;
        let size = size as usize;
        record[offset..offset + size].copy_from_slice(&plane.packed);
    }
    Ok(record.into_boxed_slice())
}

pub fn record_digest(record: &[u8]) -> Result<Digest> {
    if record.len() != FILE_BYTES {
        return Err(invalid(format!(
            "scale4 record length {} != {FILE_BYTES}",
            record.len()
        )));
    }
    parse_header(&record[..HEADER_BYTES])?;
    Ok(digest_bytes(record))
}

/// Read one canonical raw expert exactly once, authenticating all bytes while
/// retaining only its three scale planes. Packed E2M1 bytes are deliberately
/// not copied into the sidecar: native execution gathers them directly from
/// the immutable raw source and combines them with these lossless deltas.
pub fn encode_raw_expert(path: impl AsRef<Path>) -> Result<EncodedExpert> {
    let path = path.as_ref();
    let (mut source, before) = open_exact_source(path)?;
    cache_neutral(&source);
    let mut digest = DigestState::new();
    let mut scratch = vec![0_u8; 1 << 20];
    let mut scales: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::new());

    for (component, (_offset, bytes)) in RAW_LAYOUT.into_iter().enumerate() {
        let mut remaining = bytes;
        while remaining != 0 {
            let request = remaining.min(scratch.len());
            let read = read_retry(&mut source, &mut scratch[..request], path)?;
            if read == 0 {
                return Err(invalid(format!(
                    "raw expert {} ended with {remaining} bytes left in component {component}",
                    path.display()
                )));
            }
            digest.update(&scratch[..read]);
            if component & 1 == 1 {
                scales[component / 2].extend_from_slice(&scratch[..read]);
            }
            remaining -= read;
        }
    }
    if source
        .read(&mut scratch[..1])
        .map_err(|error| io_error("check raw expert extent", path, error))?
        != 0
    {
        return Err(invalid(format!(
            "raw expert {} contains trailing bytes",
            path.display()
        )));
    }
    let after = sys_fs::fstat(&source)
        .map_err(|error| io_error("restat raw expert", path, error))?;
    let after = source_identity(&after);
    if after != before {
        return Err(invalid(format!(
            "raw expert changed during its authenticated pass: {}",
            path.display()
        )));
    }
    drop_completed_cache(&source, SOURCE_BYTES);

    let [w1, w2, w3] = scales;
    if [w1.len(), w2.len(), w3.len()] != [SCALE_BYTES; 3] {
        return Err(invalid("raw expert scale extents are not canonical"));
    }
    let w1 = pack_scale4(&w1, None)?;
    let w2 = pack_scale4(&w2, None)?;
    let w3 = pack_scale4(&w3, None)?;
    let source_sha256 = digest.finalize();
    let record = build_record(source_sha256, [&w1, &w2, &w3])?;
    let record_sha256 = record_digest(&record)?;
    Ok(EncodedExpert {
        source: path.to_path_buf(),
        source_identity: before,
        source_sha256,
        record_sha256,
        bases: [w1.base, w2.base, w3.base],
        record,
    })
}

pub fn source_still_matches(encoded: &EncodedExpert) -> Result<()> {
    let metadata = sys_fs::lstat(&encoded.source)
        .map_err(|error| io_error("reinspect raw expert", &encoded.source, error))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || source_identity(&metadata) != encoded.source_identity
    {
        return Err(invalid(format!(
            "raw expert changed before sidecar publication: {}",
            encoded.source.display()
        )));
    }
    Ok(())
}

fn open_exact_source(path: &Path) -> Result<(File, SourceIdentity)> {
    let observed = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect raw expert", path, error))?;
    if observed.file_type().is_symlink()
        || !observed.is_file()
        || observed.len() != SOURCE_BYTES as u64
    {
        return Err(invalid(format!(
            "{} must be a regular non-symlink {SOURCE_BYTES}-byte raw expert",
            path.display()
        )));
    }
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open raw expert without following symlinks", path, error))?;
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat opened raw expert", path, error))?;
    let observed = source_identity(&observed);
    let opened = source_identity(&opened);
    if observed != opened {
        return Err(invalid(format!(
            "raw expert changed while opening: {}",
            path.display()
        )));
    }
    Ok((file, opened))
}

fn source_identity(metadata: &sys_fs::Stat) -> SourceIdentity {
    SourceIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

fn read_retry(file: &mut File, target: &mut [u8], path: &Path) -> Result<usize> {
    loop {
        match file.read(target) {
            Ok(read) => return Ok(read),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_error("read raw expert", path, error)),
        }
    }
}

#[cfg(target_os = "macos")]
fn cache_neutral(file: &File) {
    use std::os::fd::AsRawFd as _;
    const F_NOCACHE: i32 = 48;
    unsafe extern "C" {
        fn fcntl(fd: i32, command: i32, ...) -> i32;
    }
    // SAFETY: the descriptor is live and F_NOCACHE accepts an integer value.
    let _ = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
}

#[cfg(not(target_os = "macos"))]
fn cache_neutral(_file: &File) {}

#[cfg(target_os = "linux")]
fn drop_completed_cache(file: &File, bytes: usize) {
    use std::os::fd::AsRawFd as _;
    const POSIX_FADV_DONTNEED: i32 = 4;
    unsafe extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, length: i64, advice: i32) -> i32;
    }
    if let Ok(length) = i64::try_from(bytes) {
        // SAFETY: the descriptor is live; posix_fadvise borrows it and this is
        // an advisory, best-effort cache policy matching the mature converter.
        let _ = unsafe { posix_fadvise(file.as_raw_fd(), 0, length, POSIX_FADV_DONTNEED) };
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_completed_cache(_file: &File, _bytes: usize) {}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

fn expect_zeroes(bytes: &[u8], field: &str) -> Result<()> {
    if bytes.iter().any(|&byte| byte != 0) {
        return Err(invalid(format!("{field} must be zero")));
    }
    Ok(())
}

fn expect_u32(bytes: &[u8], offset: usize, expected: u32, field: &str) -> Result<()> {
    let actual = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    if actual != expected {
        return Err(invalid(format!("{field} {actual} != canonical {expected}")));
    }
    Ok(())
}

fn expect_u64(bytes: &[u8], offset: usize, expected: u64, field: &str) -> Result<()> {
    let actual = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    if actual != expected {
        return Err(invalid(format!("{field} {actual} != canonical {expected}")));
    }
    Ok(())
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn invalid(message: impl Into<String>) -> DeltafinError {
    DeltafinError::new(format!("invalid K3SC4V2 data: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-scale4-{nonce}-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn exact_nibble_order_roundtrips_even_and_odd_planes() {
        let even = [17_u8, 18, 31, 17, 22, 25];
        let encoded = pack_scale4(&even, None).unwrap();
        assert_eq!(encoded.base, 17);
        assert_eq!(encoded.packed.as_ref(), &[0x10, 0x0e, 0x85]);
        assert_eq!(
            unpack_scale4(&encoded.packed, encoded.base, even.len())
                .unwrap()
                .as_ref(),
            even
        );

        let odd = [240_u8, 255, 241];
        let encoded = pack_scale4(&odd, Some(240)).unwrap();
        assert_eq!(encoded.packed.as_ref(), &[0xf0, 0x01]);
        assert_eq!(
            unpack_scale4(&encoded.packed, encoded.base, odd.len())
                .unwrap()
                .as_ref(),
            odd
        );
    }

    #[test]
    fn exact_codec_rejects_lossy_ranges_and_invalid_bases() {
        assert!(pack_scale4(&[], None).is_err());
        assert!(pack_scale4(&[1, 17], None).is_err());
        assert!(pack_scale4(&[239, 240], Some(241)).is_err());
        assert!(unpack_scale4(&[0], 241, 1).is_err());
        assert!(unpack_scale4(&[0], 0, 3).is_err());
    }

    #[test]
    fn canonical_header_roundtrips_and_rejects_every_reserved_region() {
        let digest = std::array::from_fn(|index| index as u8);
        let header = build_header(digest, [7, 100, 240]).unwrap();
        assert_eq!(
            hex(digest_bytes(header.as_slice())),
            "826d7d4a29872614e6298e05be15f7637db041ce762a806ec606ccf573ee3f4b"
        );
        assert_eq!(
            parse_header(header.as_slice()).unwrap(),
            Scale4Header {
                source_sha256: digest,
                bases: [7, 100, 240],
            }
        );

        for offset in [67, 120, 127, 319, 344, HEADER_BYTES - 1] {
            let mut corrupt = header.clone();
            corrupt[offset] = 1;
            assert!(parse_header(corrupt.as_slice()).is_err(), "offset {offset}");
        }
        let mut corrupt_table = header.clone();
        corrupt_table[TABLE_OFFSETS[1] + 9] ^= 1;
        assert!(parse_header(corrupt_table.as_slice()).is_err());
    }

    #[test]
    fn record_builder_uses_the_exact_fixed_extents() {
        let values = vec![120_u8; SCALE_BYTES];
        let w1 = pack_scale4(&values, None).unwrap();
        let w2 = pack_scale4(&values, None).unwrap();
        let w3 = pack_scale4(&values, None).unwrap();
        let record = build_record([9; 32], [&w1, &w2, &w3]).unwrap();
        assert_eq!(record.len(), FILE_BYTES);
        assert_eq!(
            parse_header(&record[..HEADER_BYTES]).unwrap().bases,
            [120; 3]
        );
        assert_eq!(record_digest(&record).unwrap(), digest_bytes(&record));
        assert_eq!(
            hex(record_digest(&record).unwrap()),
            "8b50b1d82b5a7a0fd22f12826ad900e7d675ae4a89ff85921499b7e780f60f9f"
        );
    }

    #[test]
    fn one_pass_raw_encoder_matches_the_deployed_python_format() {
        let directory = TestDirectory::new();
        let path = directory.0.join("L1-E0.bin");
        let mut file = File::create(&path).unwrap();
        let bases = [100_u8, 20, 225];
        let multipliers = [1_usize, 3, 5];
        for matrix in 0..3 {
            let packed = (0..PACKED_BYTES)
                .map(|index| (((index + matrix * 13) * 17 + 3) & 255) as u8)
                .collect::<Vec<_>>();
            file.write_all(&packed).unwrap();
            let scales = (0..SCALE_BYTES)
                .map(|index| bases[matrix] + ((index * multipliers[matrix]) % 16) as u8)
                .collect::<Vec<_>>();
            file.write_all(&scales).unwrap();
        }
        file.sync_all().unwrap();
        drop(file);

        let encoded = encode_raw_expert(&path).unwrap();
        assert_eq!(encoded.bases, bases);
        assert_eq!(
            hex(encoded.source_sha256),
            "df98a3651d9bf2d60650c803589b2ab073f96b8a363411442b04fa4d943642dc"
        );
        assert_eq!(
            hex(encoded.record_sha256),
            "59cad0834d587aeebb18cae768265ccbb9e982fe8af2b7a496f4a887ebbb236a"
        );
        assert_eq!(encoded.record.len(), FILE_BYTES);
        source_still_matches(&encoded).unwrap();

        let mut replacement = OpenOptions::new().append(true).open(&path).unwrap();
        replacement.write_all(&[0]).unwrap();
        replacement.sync_all().unwrap();
        assert!(source_still_matches(&encoded).is_err());
    }

    fn hex(digest: Digest) -> String {
        let mut output = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            write!(output, "{byte:02x}").unwrap();
        }
        output
    }
}
