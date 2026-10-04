//! Deterministic, lossless per-layer resident-spine pack files.
//!
//! DFSP v1 replaces thousands of immutable tensor files with one file per
//! layer.  It does not quantize, dequantize, reinterpret, or otherwise alter
//! model bytes: every component is copied verbatim and carries an independent
//! SHA-256 digest.  The payload mirrors the native upload slab (quantized
//! bytes, row scales, then raw BF16/F32 tails), so reading a layer requires no
//! CPU repacking.
//!
//! The on-disk representation is explicitly little-endian and never casts an
//! untrusted byte slice to a Rust/C structure.  All offsets, lengths, counts,
//! names, shapes, section coverage, alignments, and hashes are validated before
//! a descriptor is exposed to the execution engine.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sys::fs::{FileExt, Open};

pub const MAGIC: [u8; 8] = *b"DFSPINE\0";
pub const VERSION_MAJOR: u16 = 1;
pub const VERSION_MINOR: u16 = 0;
pub const SUPERBLOCK_BYTES: usize = 4096;
pub const RECORD_BYTES: usize = 256;
pub const CHUNK_RECORD_BYTES: usize = 48;
pub const PAYLOAD_ALIGNMENT: u64 = 64 * 1024;
pub const COMPONENT_ALIGNMENT: u64 = 256;
pub const DEFAULT_CHUNK_BYTES: u32 = 8 * 1024 * 1024;
pub const DIGEST_BYTES: usize = 32;

const ENDIAN_TAG: u32 = 0x0102_0304;
const HASH_SHA256: u32 = 1;
const FLAG_ZERO_PADDING: u32 = 1 << 0;
const FLAG_CHUNK_HASHES: u32 = 1 << 1;
const FLAG_COMPONENT_HASHES: u32 = 1 << 2;
const REQUIRED_FLAGS: u32 = FLAG_ZERO_PADDING | FLAG_CHUNK_HASHES | FLAG_COMPONENT_HASHES;
const RECORD_FLAG_ROW_MAJOR: u32 = 1 << 0;
const RECORD_FLAG_IMMUTABLE: u32 = 1 << 1;

const HEADER_DIGEST_OFFSET: usize = 384;
const DIRECTORY_DIGEST_OFFSET: usize = 288;
const PAYLOAD_DIGEST_OFFSET: usize = 320;
const PACK_DIGEST_OFFSET: usize = 352;

pub type Digest = [u8; DIGEST_BYTES];

#[derive(Debug)]
pub enum PackError {
    Invalid(String),
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
}

impl PackError {
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    fn io(operation: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.into(),
            source,
        }
    }
}

impl Display for PackError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => message.fmt(formatter),
            Self::Io {
                operation,
                path,
                source,
            } => write!(formatter, "{operation} {}: {source}", path.display()),
        }
    }
}

impl Error for PackError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Invalid(_) => None,
            Self::Io { source, .. } => Some(source),
        }
    }
}

pub type Result<T> = std::result::Result<T, PackError>;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PackIdentity {
    pub model: Digest,
    pub source_inventory: Digest,
    pub layout_schema: Digest,
}

impl PackIdentity {
    pub const fn new(model: Digest, source_inventory: Digest, layout_schema: Digest) -> Self {
        Self {
            model,
            source_inventory,
            layout_schema,
        }
    }

    fn validate(self) -> Result<()> {
        if self.model == [0; DIGEST_BYTES] {
            return Err(PackError::invalid("model identity digest is all zero"));
        }
        if self.source_inventory == [0; DIGEST_BYTES] {
            return Err(PackError::invalid(
                "source-inventory identity digest is all zero",
            ));
        }
        if self.layout_schema == [0; DIGEST_BYTES] {
            return Err(PackError::invalid(
                "layout-schema identity digest is all zero",
            ));
        }
        Ok(())
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Codec {
    Raw = 0,
    RowI8F16Scale = 1,
}

impl Codec {
    fn parse(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::RowI8F16Scale),
            _ => Err(PackError::invalid(format!("unknown tensor codec {value}"))),
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DType {
    None = 0,
    U8 = 1,
    I8 = 2,
    F16 = 3,
    Bf16 = 4,
    F32 = 5,
}

impl DType {
    fn parse(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::U8),
            2 => Ok(Self::I8),
            3 => Ok(Self::F16),
            4 => Ok(Self::Bf16),
            5 => Ok(Self::F32),
            _ => Err(PackError::invalid(format!("unknown tensor dtype {value}"))),
        }
    }

    const fn byte_width(self) -> Option<u64> {
        match self {
            Self::None => None,
            Self::U8 | Self::I8 => Some(1),
            Self::F16 | Self::Bf16 => Some(2),
            Self::F32 => Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ComponentSource {
    Bytes(Arc<[u8]>),
    FileRange {
        path: PathBuf,
        offset: u64,
        length: u64,
    },
}

impl ComponentSource {
    pub fn bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(bytes.into().into())
    }

    pub fn file(path: impl Into<PathBuf>, offset: u64, length: u64) -> Self {
        Self::FileRange {
            path: path.into(),
            offset,
            length,
        }
    }

    pub fn len(&self) -> u64 {
        match self {
            Self::Bytes(bytes) => bytes.len() as u64,
            Self::FileRange { length, .. } => *length,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct BuildTensor {
    name: String,
    codec: Codec,
    logical_dtype: DType,
    shape: Vec<u64>,
    data: ComponentSource,
    auxiliary: Option<ComponentSource>,
    upload_group: u16,
    upload_order: u16,
}

impl BuildTensor {
    pub fn raw(
        name: impl Into<String>,
        dtype: DType,
        shape: impl Into<Vec<u64>>,
        bytes: ComponentSource,
        upload_group: u16,
        upload_order: u16,
    ) -> Self {
        Self {
            name: name.into(),
            codec: Codec::Raw,
            logical_dtype: dtype,
            shape: shape.into(),
            data: bytes,
            auxiliary: None,
            upload_group,
            upload_order,
        }
    }

    pub fn row_i8_f16_scale(
        name: impl Into<String>,
        logical_dtype: DType,
        shape: [u64; 2],
        quantized: ComponentSource,
        scales: ComponentSource,
        upload_group: u16,
        upload_order: u16,
    ) -> Self {
        Self {
            name: name.into(),
            codec: Codec::RowI8F16Scale,
            logical_dtype,
            shape: shape.into(),
            data: quantized,
            auxiliary: Some(scales),
            upload_group,
            upload_order,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ParseLimits {
    pub max_file_bytes: u64,
    pub max_payload_bytes: u64,
    pub max_metadata_bytes: u64,
    pub max_records: u32,
    pub max_name_bytes: u32,
    pub max_chunks: u32,
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            // The largest current layer is 1.18 GB.  Leave room for later
            // exact layouts without permitting attacker-controlled petabyte
            // allocations from a corrupt header.
            max_file_bytes: 8 * 1024 * 1024 * 1024,
            max_payload_bytes: 8 * 1024 * 1024 * 1024,
            max_metadata_bytes: 64 * 1024 * 1024,
            max_records: 4096,
            max_name_bytes: 1024,
            max_chunks: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackHeader {
    pub layer: u32,
    pub identity: PackIdentity,
    pub file_bytes: u64,
    pub payload_offset: u64,
    pub payload_bytes: u64,
    pub quantized_offset: u64,
    pub quantized_used: u64,
    pub quantized_span: u64,
    pub scales_offset: u64,
    pub scales_used: u64,
    pub scales_span: u64,
    pub raw_offset: u64,
    pub raw_used: u64,
    pub raw_span: u64,
    pub chunk_bytes: u32,
    pub directory_digest: Digest,
    pub payload_digest: Digest,
    pub pack_digest: Digest,
    records_offset: u64,
    records_bytes: u64,
    strings_offset: u64,
    strings_bytes: u64,
    chunks_offset: u64,
    chunks_bytes: u64,
    record_count: u32,
    chunk_count: u32,
    header_digest: Digest,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TensorRecord {
    pub name: String,
    pub name_digest: Digest,
    pub codec: Codec,
    pub logical_dtype: DType,
    pub data_dtype: DType,
    pub auxiliary_dtype: DType,
    pub rank: u8,
    pub scale_axis: i8,
    pub flags: u32,
    pub upload_group: u16,
    pub upload_order: u16,
    pub shape: [u64; 8],
    pub data_offset: u64,
    pub data_length: u64,
    pub auxiliary_offset: u64,
    pub auxiliary_length: u64,
    pub element_count: u64,
    pub rows: u32,
    pub columns: u32,
    pub data_digest: Digest,
    pub auxiliary_digest: Digest,
    name_offset: u32,
    name_length: u16,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ChunkRecord {
    pub payload_offset: u64,
    pub length: u32,
    pub digest: Digest,
}

/// A C-compatible, pointer-free upload description.  Offsets are relative to
/// the beginning of the host/device payload slab, not the beginning of the
/// pack file.  Names are resolved to provider slots once, outside the hot path.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct UploadDescriptorV1 {
    pub name_digest: Digest,
    pub record_index: u32,
    pub codec: u8,
    pub logical_dtype: u8,
    pub data_dtype: u8,
    pub auxiliary_dtype: u8,
    pub rank: u8,
    pub scale_axis: i8,
    pub flags: u16,
    pub upload_group: u16,
    pub upload_order: u16,
    pub reserved: u32,
    pub shape: [u64; 8],
    pub data_offset: u64,
    pub data_length: u64,
    pub auxiliary_offset: u64,
    pub auxiliary_length: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ReadExtent {
    pub file_offset: u64,
    pub destination_offset: u64,
    pub length: u32,
    pub expected_digest: Digest,
}

#[derive(Debug, Clone)]
pub struct PackFile {
    path: PathBuf,
    header: PackHeader,
    tensors: Vec<TensorRecord>,
    chunks: Vec<ChunkRecord>,
}

impl PackFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, ParseLimits::default())
    }

    pub fn open_for(
        path: impl AsRef<Path>,
        expected_layer: u32,
        expected_identity: PackIdentity,
    ) -> Result<Self> {
        let pack = Self::open(path)?;
        pack.require_identity(expected_layer, expected_identity)?;
        Ok(pack)
    }

    pub fn open_with_limits(path: impl AsRef<Path>, limits: ParseLimits) -> Result<Self> {
        validate_limits(limits)?;
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|error| PackError::io("open", &path, error))?;
        let actual_file_bytes = file
            .metadata()
            .map_err(|error| PackError::io("stat", &path, error))?
            .len();
        if actual_file_bytes > limits.max_file_bytes {
            return Err(PackError::invalid(format!(
                "pack is {actual_file_bytes} bytes, above the {}-byte parsing limit",
                limits.max_file_bytes
            )));
        }
        if actual_file_bytes < SUPERBLOCK_BYTES as u64 {
            return Err(PackError::invalid(format!(
                "pack is truncated: {actual_file_bytes} bytes is smaller than the {SUPERBLOCK_BYTES}-byte superblock"
            )));
        }

        let mut superblock = [0_u8; SUPERBLOCK_BYTES];
        read_exact_at(&mut file, 0, &mut superblock, &path)?;
        let header = decode_header(&superblock, actual_file_bytes, limits)?;

        let directory_len = usize_from_u64(
            header
                .payload_offset
                .checked_sub(SUPERBLOCK_BYTES as u64)
                .ok_or_else(|| PackError::invalid("payload precedes the record directory"))?,
            "directory length",
        )?;
        let mut directory = vec![0_u8; directory_len];
        read_exact_at(&mut file, SUPERBLOCK_BYTES as u64, &mut directory, &path)?;
        if sha256(&directory) != header.directory_digest {
            return Err(PackError::invalid("record-directory SHA-256 mismatch"));
        }

        let tensors = decode_records(&header, &directory, limits)?;
        let chunks = decode_chunks(&header, &directory, limits)?;
        validate_tensor_layout(&header, &tensors)?;

        Ok(Self {
            path,
            header,
            tensors,
            chunks,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &PackHeader {
        &self.header
    }

    pub fn tensors(&self) -> &[TensorRecord] {
        &self.tensors
    }

    pub fn chunks(&self) -> &[ChunkRecord] {
        &self.chunks
    }

    pub fn require_identity(&self, layer: u32, identity: PackIdentity) -> Result<()> {
        identity.validate()?;
        if self.header.layer != layer {
            return Err(PackError::invalid(format!(
                "pack layer {} does not match expected layer {layer}",
                self.header.layer
            )));
        }
        if self.header.identity != identity {
            return Err(PackError::invalid(
                "pack model/source/layout identity does not match the active runtime",
            ));
        }
        Ok(())
    }

    pub fn upload_descriptors(&self) -> Vec<UploadDescriptorV1> {
        self.tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| UploadDescriptorV1 {
                name_digest: tensor.name_digest,
                record_index: index as u32,
                codec: tensor.codec as u8,
                logical_dtype: tensor.logical_dtype as u8,
                data_dtype: tensor.data_dtype as u8,
                auxiliary_dtype: tensor.auxiliary_dtype as u8,
                rank: tensor.rank,
                scale_axis: tensor.scale_axis,
                flags: tensor.flags as u16,
                upload_group: tensor.upload_group,
                upload_order: tensor.upload_order,
                reserved: 0,
                shape: tensor.shape,
                data_offset: tensor.data_offset,
                data_length: tensor.data_length,
                auxiliary_offset: tensor.auxiliary_offset,
                auxiliary_length: tensor.auxiliary_length,
            })
            .collect()
    }

    pub fn read_extents(&self) -> Vec<ReadExtent> {
        self.chunks
            .iter()
            .map(|chunk| ReadExtent {
                file_offset: self.header.payload_offset + chunk.payload_offset,
                destination_offset: chunk.payload_offset,
                length: chunk.length,
                expected_digest: chunk.digest,
            })
            .collect()
    }

    /// Verify one payload chunk. This is suitable for a first-read bitset only
    /// while the scheduler retains the same immutable opened file generation.
    /// A bitset must be discarded if the pack is reopened or its file identity
    /// changes; a pathname alone is not a stable verification identity.
    pub fn verify_chunk(&self, index: usize) -> Result<()> {
        let chunk = self
            .chunks
            .get(index)
            .ok_or_else(|| PackError::invalid(format!("chunk index {index} is out of range")))?;
        let mut file =
            File::open(&self.path).map_err(|error| PackError::io("open", &self.path, error))?;
        let mut bytes = vec![0_u8; chunk.length as usize];
        read_exact_at(
            &mut file,
            self.header.payload_offset + chunk.payload_offset,
            &mut bytes,
            &self.path,
        )?;
        self.verify_chunk_data(index, &bytes)
    }

    /// Verify bytes that the I/O scheduler already read into its destination
    /// slab.  This avoids a second disk read on a chunk's first verified use.
    pub fn verify_chunk_data(&self, index: usize, bytes: &[u8]) -> Result<()> {
        let chunk = self
            .chunks
            .get(index)
            .ok_or_else(|| PackError::invalid(format!("chunk index {index} is out of range")))?;
        if bytes.len() != chunk.length as usize {
            return Err(PackError::invalid(format!(
                "payload chunk {index} supplied {} bytes, expected {}",
                bytes.len(),
                chunk.length
            )));
        }
        if sha256(bytes) != chunk.digest {
            return Err(PackError::invalid(format!(
                "payload chunk {index} SHA-256 mismatch"
            )));
        }
        Ok(())
    }

    /// Stream the payload once and verify its whole-file, per-chunk, and
    /// per-component digests.  Bytes outside declared components must be zero.
    pub fn verify_all(&self) -> Result<()> {
        // Open first and inspect that exact descriptor. A path-level `stat`
        // followed by `open` would permit a rename between the two calls and
        // could validate the size of a different inode than the one read.
        let mut file =
            File::open(&self.path).map_err(|error| PackError::io("open", &self.path, error))?;
        let actual = file
            .metadata()
            .map_err(|error| PackError::io("stat", &self.path, error))?
            .len();
        if actual != self.header.file_bytes {
            return Err(PackError::invalid(format!(
                "pack length changed after admission: {actual} != {}",
                self.header.file_bytes
            )));
        }

        let ranges = component_ranges(&self.tensors)?;
        let mut component_hashers = vec![Sha256::new(); ranges.len()];
        let mut payload_hasher = Sha256::new();
        let mut chunk_buffer = vec![0_u8; self.header.chunk_bytes as usize];
        let mut first_live_range = 0_usize;

        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let bytes = &mut chunk_buffer[..chunk.length as usize];
            read_exact_at(
                &mut file,
                self.header.payload_offset + chunk.payload_offset,
                bytes,
                &self.path,
            )?;
            let chunk_digest = sha256(bytes);
            if chunk_digest != chunk.digest {
                return Err(PackError::invalid(format!(
                    "payload chunk {chunk_index} SHA-256 mismatch"
                )));
            }
            payload_hasher.update(bytes);
            // Component ranges and chunks are both canonical and sorted. Do
            // not restart the range scan at tensor zero for every chunk: a
            // malicious maximum-size manifest could otherwise turn a linear
            // integrity pass into billions of range comparisons.
            while first_live_range < ranges.len()
                && ranges[first_live_range].end <= chunk.payload_offset
            {
                first_live_range += 1;
            }
            verify_chunk_coverage(
                chunk.payload_offset,
                bytes,
                &ranges[first_live_range..],
                &mut component_hashers[first_live_range..],
            )?;
        }

        if payload_hasher.finalize() != self.header.payload_digest {
            return Err(PackError::invalid("whole-payload SHA-256 mismatch"));
        }
        for (range, hasher) in ranges.iter().zip(component_hashers) {
            if hasher.finalize() != range.expected_digest {
                return Err(PackError::invalid(format!(
                    "{} {} component SHA-256 mismatch",
                    range.name, range.part
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct PackBuilder {
    layer: u32,
    identity: PackIdentity,
    chunk_bytes: u32,
    tensors: Vec<BuildTensor>,
}

impl PackBuilder {
    pub fn new(layer: u32, identity: PackIdentity) -> Result<Self> {
        identity.validate()?;
        Ok(Self {
            layer,
            identity,
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            tensors: Vec::new(),
        })
    }

    pub fn set_chunk_bytes(&mut self, chunk_bytes: u32) -> Result<()> {
        validate_chunk_bytes(chunk_bytes)?;
        self.chunk_bytes = chunk_bytes;
        Ok(())
    }

    pub fn push(&mut self, tensor: BuildTensor) {
        self.tensors.push(tensor);
    }

    /// Build, fsync, fully re-open and verify, then publish without replacing
    /// an existing destination.  `hard_link` gives no-clobber atomic visibility
    /// on the same filesystem; the private temporary name is removed after the
    /// directory entry is durable.
    pub fn write_atomic(&self, destination: impl AsRef<Path>) -> Result<PackFile> {
        let destination = destination.as_ref();
        if destination.exists() {
            return Err(PackError::invalid(format!(
                "refusing to replace existing pack {}",
                destination.display()
            )));
        }
        // `Path::parent()` returns `Some("")` for a bare relative name such
        // as `layer.dfsp`.  Treat that empty path as the current directory;
        // otherwise an otherwise valid relative destination is rejected by
        // the `is_dir` check below on Unix and Windows.
        let parent = destination_parent(destination);
        if !parent.is_dir() {
            return Err(PackError::invalid(format!(
                "pack parent directory does not exist: {}",
                parent.display()
            )));
        }
        let stem = destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("layer.dfsp");
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(".{stem}.tmp-{}-{sequence}", std::process::id()));

        let result = (|| {
            let mut output = OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&temporary)
                .map_err(|error| PackError::io("create", &temporary, error))?;
            self.write_file(&mut output, &temporary)?;
            output
                .sync_all()
                .map_err(|error| PackError::io("fsync", &temporary, error))?;
            drop(output);

            let staged = PackFile::open_for(&temporary, self.layer, self.identity)?;
            staged.verify_all()?;
            fs::hard_link(&temporary, destination)
                .map_err(|error| PackError::io("publish", destination, error))?;
            sync_directory(parent)?;
            let _ = fs::remove_file(&temporary);
            PackFile::open_for(destination, self.layer, self.identity)
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    fn write_file(&self, output: &mut File, output_path: &Path) -> Result<()> {
        validate_chunk_bytes(self.chunk_bytes)?;
        let limits = ParseLimits::default();
        let mut planned = plan_build(self, limits)?;
        let placeholder_bytes =
            usize_from_u64(planned.header.payload_offset, "metadata placeholder length")?;
        write_zeroes(
            output,
            placeholder_bytes,
            output_path,
            "write metadata placeholder",
        )?;
        output
            .seek(SeekFrom::Start(planned.header.payload_offset))
            .map_err(|error| PackError::io("seek payload", output_path, error))?;

        let mut payload = PayloadWriter::new(
            output,
            output_path,
            planned.header.payload_bytes,
            self.chunk_bytes,
        );

        write_component_section(
            &mut payload,
            &mut planned.tensors,
            SectionPart::Data,
            Codec::RowI8F16Scale,
            planned.header.quantized_offset,
            planned.header.quantized_span,
        )?;
        write_component_section(
            &mut payload,
            &mut planned.tensors,
            SectionPart::Auxiliary,
            Codec::RowI8F16Scale,
            planned.header.scales_offset,
            planned.header.scales_span,
        )?;
        write_component_section(
            &mut payload,
            &mut planned.tensors,
            SectionPart::Data,
            Codec::Raw,
            planned.header.raw_offset,
            planned.header.raw_span,
        )?;
        let payload_result = payload.finish()?;
        planned.header.payload_digest = payload_result.digest;
        planned.chunks = payload_result.chunks;

        let directory = encode_directory(&planned)?;
        planned.header.directory_digest = sha256(&directory);
        planned.header.pack_digest = compute_pack_digest(&planned.header);
        let mut superblock = encode_header(&planned.header);
        let header_digest =
            sha256_with_zeroed_range(&superblock, HEADER_DIGEST_OFFSET, DIGEST_BYTES);
        planned.header.header_digest = header_digest;
        superblock = encode_header(&planned.header);

        output
            .seek(SeekFrom::Start(0))
            .map_err(|error| PackError::io("seek superblock", output_path, error))?;
        output
            .write_all(&superblock)
            .map_err(|error| PackError::io("write superblock", output_path, error))?;
        output
            .write_all(&directory)
            .map_err(|error| PackError::io("write directory", output_path, error))?;
        output
            .set_len(planned.header.file_bytes)
            .map_err(|error| PackError::io("set pack length", output_path, error))?;
        Ok(())
    }
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct PlannedTensor {
    source: BuildTensor,
    record: TensorRecord,
}

#[derive(Debug)]
struct PlannedPack {
    header: PackHeader,
    tensors: Vec<PlannedTensor>,
    chunks: Vec<ChunkRecord>,
    strings: Vec<u8>,
}

fn plan_build(builder: &PackBuilder, limits: ParseLimits) -> Result<PlannedPack> {
    if builder.tensors.is_empty() {
        return Err(PackError::invalid(
            "a layer pack must contain at least one tensor",
        ));
    }
    if builder.tensors.len() > limits.max_records as usize {
        return Err(PackError::invalid(format!(
            "pack has {} tensors, above the {}-record limit",
            builder.tensors.len(),
            limits.max_records
        )));
    }

    let mut sources = builder.tensors.clone();
    sources.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    for pair in sources.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(PackError::invalid(format!(
                "duplicate tensor name {}",
                pair[0].name
            )));
        }
    }

    let mut strings = Vec::new();
    let mut tensors = Vec::with_capacity(sources.len());
    for source in sources {
        validate_build_tensor(builder.layer, &source, limits.max_name_bytes)?;
        let name_offset = u32_from_usize(strings.len(), "string-table offset")?;
        let name_length = u16::try_from(source.name.len())
            .map_err(|_| PackError::invalid("tensor name does not fit in u16"))?;
        strings.extend_from_slice(source.name.as_bytes());
        let (data_dtype, auxiliary_dtype, scale_axis, flags) = match source.codec {
            Codec::Raw => (source.logical_dtype, DType::None, -1, RECORD_FLAG_IMMUTABLE),
            Codec::RowI8F16Scale => (
                DType::I8,
                DType::F16,
                0,
                RECORD_FLAG_ROW_MAJOR | RECORD_FLAG_IMMUTABLE,
            ),
        };
        let mut shape = [0_u64; 8];
        shape[..source.shape.len()].copy_from_slice(&source.shape);
        let element_count = checked_product(&source.shape, "tensor element count")?;
        let rows = if source.shape.len() == 2 {
            u32::try_from(source.shape[0])
                .map_err(|_| PackError::invalid("tensor row count does not fit in u32"))?
        } else {
            0
        };
        let columns = if source.shape.len() == 2 {
            u32::try_from(source.shape[1])
                .map_err(|_| PackError::invalid("tensor column count does not fit in u32"))?
        } else {
            0
        };
        tensors.push(PlannedTensor {
            record: TensorRecord {
                name: source.name.clone(),
                name_digest: sha256(source.name.as_bytes()),
                codec: source.codec,
                logical_dtype: source.logical_dtype,
                data_dtype,
                auxiliary_dtype,
                rank: source.shape.len() as u8,
                scale_axis,
                flags,
                upload_group: source.upload_group,
                upload_order: source.upload_order,
                shape,
                data_offset: 0,
                data_length: source.data.len(),
                auxiliary_offset: 0,
                auxiliary_length: source.auxiliary.as_ref().map_or(0, ComponentSource::len),
                element_count,
                rows,
                columns,
                data_digest: [0; DIGEST_BYTES],
                auxiliary_digest: [0; DIGEST_BYTES],
                name_offset,
                name_length,
            },
            source,
        });
    }
    if strings.len() > limits.max_metadata_bytes as usize {
        return Err(PackError::invalid(
            "tensor-name table exceeds metadata limit",
        ));
    }

    let q_order = physical_order(&tensors, Codec::RowI8F16Scale);
    let raw_order = physical_order(&tensors, Codec::Raw);
    let (quantized_used, quantized_span) =
        assign_offsets(&mut tensors, &q_order, SectionPart::Data, 0)?;
    let scales_offset = quantized_span;
    let (scales_used, scales_span) = assign_offsets(
        &mut tensors,
        &q_order,
        SectionPart::Auxiliary,
        scales_offset,
    )?;
    let raw_offset = checked_add(scales_offset, scales_span, "raw-section offset")?;
    let (raw_used, raw_span) =
        assign_offsets(&mut tensors, &raw_order, SectionPart::Data, raw_offset)?;
    let payload_bytes = checked_add(raw_offset, raw_span, "payload length")?;
    if payload_bytes == 0 {
        return Err(PackError::invalid("layer payload is empty"));
    }
    if payload_bytes > limits.max_payload_bytes {
        return Err(PackError::invalid(format!(
            "payload is {payload_bytes} bytes, above the {}-byte limit",
            limits.max_payload_bytes
        )));
    }

    let record_count = u32_from_usize(tensors.len(), "record count")?;
    let records_offset = SUPERBLOCK_BYTES as u64;
    let records_bytes = checked_mul(record_count as u64, RECORD_BYTES as u64, "record table")?;
    let strings_offset = align_up(
        checked_add(records_offset, records_bytes, "string-table offset")?,
        8,
    )?;
    let strings_bytes = strings.len() as u64;
    let chunks_offset = align_up(
        checked_add(strings_offset, strings_bytes, "chunk-table offset")?,
        8,
    )?;
    let chunk_count_u64 = ceil_div(payload_bytes, builder.chunk_bytes as u64)?;
    let chunk_count = u32::try_from(chunk_count_u64)
        .map_err(|_| PackError::invalid("chunk count does not fit in u32"))?;
    if chunk_count > limits.max_chunks {
        return Err(PackError::invalid(format!(
            "pack needs {chunk_count} chunks, above the {}-chunk limit",
            limits.max_chunks
        )));
    }
    let chunks_bytes = checked_mul(chunk_count as u64, CHUNK_RECORD_BYTES as u64, "chunk table")?;
    let payload_offset = align_up(
        checked_add(chunks_offset, chunks_bytes, "payload offset")?,
        PAYLOAD_ALIGNMENT,
    )?;
    let metadata_bytes = payload_offset - SUPERBLOCK_BYTES as u64;
    if metadata_bytes > limits.max_metadata_bytes {
        return Err(PackError::invalid(format!(
            "metadata is {metadata_bytes} bytes, above the {}-byte limit",
            limits.max_metadata_bytes
        )));
    }
    let file_bytes = checked_add(payload_offset, payload_bytes, "pack file length")?;
    if file_bytes > limits.max_file_bytes {
        return Err(PackError::invalid(format!(
            "pack is {file_bytes} bytes, above the {}-byte limit",
            limits.max_file_bytes
        )));
    }

    Ok(PlannedPack {
        header: PackHeader {
            layer: builder.layer,
            identity: builder.identity,
            file_bytes,
            payload_offset,
            payload_bytes,
            quantized_offset: 0,
            quantized_used,
            quantized_span,
            scales_offset,
            scales_used,
            scales_span,
            raw_offset,
            raw_used,
            raw_span,
            chunk_bytes: builder.chunk_bytes,
            directory_digest: [0; DIGEST_BYTES],
            payload_digest: [0; DIGEST_BYTES],
            pack_digest: [0; DIGEST_BYTES],
            records_offset,
            records_bytes,
            strings_offset,
            strings_bytes,
            chunks_offset,
            chunks_bytes,
            record_count,
            chunk_count,
            header_digest: [0; DIGEST_BYTES],
        },
        tensors,
        chunks: Vec::new(),
        strings,
    })
}

fn validate_build_tensor(layer: u32, tensor: &BuildTensor, max_name_bytes: u32) -> Result<()> {
    validate_name(layer, &tensor.name, max_name_bytes)?;
    if tensor.shape.is_empty() || tensor.shape.len() > 8 {
        return Err(PackError::invalid(format!(
            "{} has rank {}; DFSP v1 supports ranks 1..=8",
            tensor.name,
            tensor.shape.len()
        )));
    }
    if tensor.shape.contains(&0) {
        return Err(PackError::invalid(format!(
            "{} has a zero-sized dimension",
            tensor.name
        )));
    }
    let elements = checked_product(&tensor.shape, "tensor shape")?;
    match tensor.codec {
        Codec::Raw => {
            let width = tensor.logical_dtype.byte_width().ok_or_else(|| {
                PackError::invalid(format!("{} raw tensor has no storage dtype", tensor.name))
            })?;
            let expected = checked_mul(elements, width, "raw tensor byte length")?;
            if tensor.data.len() != expected {
                return Err(PackError::invalid(format!(
                    "{} raw byte length {} does not equal shape/dtype length {expected}",
                    tensor.name,
                    tensor.data.len()
                )));
            }
            if tensor.auxiliary.is_some() {
                return Err(PackError::invalid(format!(
                    "{} raw tensor unexpectedly has auxiliary bytes",
                    tensor.name
                )));
            }
        }
        Codec::RowI8F16Scale => {
            if tensor.shape.len() != 2 {
                return Err(PackError::invalid(format!(
                    "{} row-int8 tensor must have rank 2",
                    tensor.name
                )));
            }
            if !matches!(tensor.logical_dtype, DType::Bf16 | DType::F32) {
                return Err(PackError::invalid(format!(
                    "{} row-int8 logical dtype must be BF16 or F32",
                    tensor.name
                )));
            }
            if tensor.data.len() != elements {
                return Err(PackError::invalid(format!(
                    "{} q8 byte length {} does not equal rows*columns {elements}",
                    tensor.name,
                    tensor.data.len()
                )));
            }
            let scales = tensor.auxiliary.as_ref().ok_or_else(|| {
                PackError::invalid(format!(
                    "{} row-int8 tensor has no scale bytes",
                    tensor.name
                ))
            })?;
            let expected_scales = checked_mul(tensor.shape[0], 2, "scale byte length")?;
            if scales.len() != expected_scales {
                return Err(PackError::invalid(format!(
                    "{} scale byte length {} does not equal rows*2 {expected_scales}",
                    tensor.name,
                    scales.len()
                )));
            }
        }
    }
    Ok(())
}

fn validate_name(layer: u32, name: &str, max_name_bytes: u32) -> Result<()> {
    if name.is_empty() || name.len() > max_name_bytes as usize {
        return Err(PackError::invalid(format!(
            "tensor name length {} is outside 1..={max_name_bytes}",
            name.len()
        )));
    }
    if !name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(PackError::invalid(format!(
            "tensor name contains a non-canonical byte: {name:?}"
        )));
    }
    let prefix = format!("language_model.model.layers.{layer}.");
    if !name.starts_with(&prefix) || name.len() == prefix.len() {
        return Err(PackError::invalid(format!(
            "tensor {name:?} does not belong to layer {layer}"
        )));
    }
    Ok(())
}

fn physical_order(tensors: &[PlannedTensor], codec: Codec) -> Vec<usize> {
    let mut indices: Vec<usize> = tensors
        .iter()
        .enumerate()
        .filter_map(|(index, tensor)| (tensor.record.codec == codec).then_some(index))
        .collect();
    indices.sort_by(|left, right| {
        let left = &tensors[*left].record;
        let right = &tensors[*right].record;
        (left.upload_group, left.upload_order, left.name.as_bytes()).cmp(&(
            right.upload_group,
            right.upload_order,
            right.name.as_bytes(),
        ))
    });
    indices
}

#[derive(Debug, Clone, Copy)]
enum SectionPart {
    Data,
    Auxiliary,
}

fn assign_offsets(
    tensors: &mut [PlannedTensor],
    order: &[usize],
    part: SectionPart,
    section_offset: u64,
) -> Result<(u64, u64)> {
    let mut cursor = section_offset;
    let mut used = 0_u64;
    for &index in order {
        cursor = align_up(cursor, COMPONENT_ALIGNMENT)?;
        let record = &mut tensors[index].record;
        let length = match part {
            SectionPart::Data => {
                record.data_offset = cursor;
                record.data_length
            }
            SectionPart::Auxiliary => {
                record.auxiliary_offset = cursor;
                record.auxiliary_length
            }
        };
        if length == 0 {
            return Err(PackError::invalid(format!(
                "{} contains an empty declared component",
                record.name
            )));
        }
        used = checked_add(used, length, "section used-byte count")?;
        cursor = checked_add(cursor, length, "component end")?;
    }
    if order.is_empty() {
        return Ok((0, 0));
    }
    let local_end = cursor
        .checked_sub(section_offset)
        .ok_or_else(|| PackError::invalid("section cursor underflow"))?;
    Ok((used, align_up(local_end, PAYLOAD_ALIGNMENT)?))
}

fn write_component_section(
    writer: &mut PayloadWriter<'_>,
    tensors: &mut [PlannedTensor],
    part: SectionPart,
    codec: Codec,
    section_offset: u64,
    section_span: u64,
) -> Result<()> {
    writer.pad_to(section_offset)?;
    let order = physical_order(tensors, codec);
    for index in order {
        let target = match part {
            SectionPart::Data => tensors[index].record.data_offset,
            SectionPart::Auxiliary => tensors[index].record.auxiliary_offset,
        };
        writer.pad_to(target)?;
        let digest = match part {
            SectionPart::Data => writer.write_source(&tensors[index].source.data)?,
            SectionPart::Auxiliary => {
                let source = tensors[index].source.auxiliary.as_ref().ok_or_else(|| {
                    PackError::invalid(format!(
                        "{} lost its auxiliary source during build",
                        tensors[index].record.name
                    ))
                })?;
                writer.write_source(source)?
            }
        };
        match part {
            SectionPart::Data => tensors[index].record.data_digest = digest,
            SectionPart::Auxiliary => tensors[index].record.auxiliary_digest = digest,
        }
    }
    writer.pad_to(checked_add(section_offset, section_span, "section end")?)
}

struct PayloadWriter<'a> {
    output: &'a mut File,
    output_path: &'a Path,
    position: u64,
    payload_bytes: u64,
    chunk_bytes: u32,
    whole: Sha256,
    chunk: Sha256,
    chunk_start: u64,
    chunk_length: u32,
    chunks: Vec<ChunkRecord>,
}

struct PayloadResult {
    digest: Digest,
    chunks: Vec<ChunkRecord>,
}

impl<'a> PayloadWriter<'a> {
    fn new(
        output: &'a mut File,
        output_path: &'a Path,
        payload_bytes: u64,
        chunk_bytes: u32,
    ) -> Self {
        Self {
            output,
            output_path,
            position: 0,
            payload_bytes,
            chunk_bytes,
            whole: Sha256::new(),
            chunk: Sha256::new(),
            chunk_start: 0,
            chunk_length: 0,
            chunks: Vec::new(),
        }
    }

    fn pad_to(&mut self, target: u64) -> Result<()> {
        if target < self.position {
            return Err(PackError::invalid(format!(
                "payload writer moved backwards from {} to {target}",
                self.position
            )));
        }
        let mut remaining = target - self.position;
        let zeroes = [0_u8; 64 * 1024];
        while remaining != 0 {
            let count = remaining.min(zeroes.len() as u64) as usize;
            self.write_bytes(&zeroes[..count])?;
            remaining -= count as u64;
        }
        Ok(())
    }

    fn write_source(&mut self, source: &ComponentSource) -> Result<Digest> {
        let mut component = Sha256::new();
        match source {
            ComponentSource::Bytes(bytes) => {
                component.update(bytes);
                self.write_bytes(bytes)?;
            }
            ComponentSource::FileRange {
                path,
                offset,
                length,
            } => {
                let mut input = File::open(path)
                    .map_err(|error| PackError::io("open component", path, error))?;
                let available = input
                    .metadata()
                    .map_err(|error| PackError::io("stat component", path, error))?
                    .len();
                let end = offset
                    .checked_add(*length)
                    .ok_or_else(|| PackError::invalid("component file range overflows u64"))?;
                if end > available {
                    return Err(PackError::invalid(format!(
                        "component range {offset}..{end} exceeds {}-byte file {}",
                        available,
                        path.display()
                    )));
                }
                input
                    .seek(SeekFrom::Start(*offset))
                    .map_err(|error| PackError::io("seek component", path, error))?;
                let mut remaining = *length;
                let mut buffer = vec![0_u8; 1024 * 1024];
                while remaining != 0 {
                    let wanted = remaining.min(buffer.len() as u64) as usize;
                    input
                        .read_exact(&mut buffer[..wanted])
                        .map_err(|error| PackError::io("read component", path, error))?;
                    component.update(&buffer[..wanted]);
                    self.write_bytes(&buffer[..wanted])?;
                    remaining -= wanted as u64;
                }
            }
        }
        Ok(component.finalize())
    }

    fn write_bytes(&mut self, mut bytes: &[u8]) -> Result<()> {
        let end = self
            .position
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| PackError::invalid("payload write position overflows u64"))?;
        if end > self.payload_bytes {
            return Err(PackError::invalid(format!(
                "payload writer exceeded declared {} bytes",
                self.payload_bytes
            )));
        }
        self.output
            .write_all(bytes)
            .map_err(|error| PackError::io("write payload", self.output_path, error))?;
        self.whole.update(bytes);
        while !bytes.is_empty() {
            let room = self.chunk_bytes - self.chunk_length;
            let take = (room as usize).min(bytes.len());
            self.chunk.update(&bytes[..take]);
            self.chunk_length += take as u32;
            self.position += take as u64;
            bytes = &bytes[take..];
            if self.chunk_length == self.chunk_bytes {
                self.finish_chunk();
            }
        }
        Ok(())
    }

    fn finish_chunk(&mut self) {
        if self.chunk_length == 0 {
            return;
        }
        let digest = std::mem::replace(&mut self.chunk, Sha256::new()).finalize();
        self.chunks.push(ChunkRecord {
            payload_offset: self.chunk_start,
            length: self.chunk_length,
            digest,
        });
        self.chunk_start = self.position;
        self.chunk_length = 0;
    }

    fn finish(mut self) -> Result<PayloadResult> {
        self.pad_to(self.payload_bytes)?;
        self.finish_chunk();
        Ok(PayloadResult {
            digest: self.whole.finalize(),
            chunks: self.chunks,
        })
    }
}

fn encode_directory(pack: &PlannedPack) -> Result<Vec<u8>> {
    if pack.chunks.len() != pack.header.chunk_count as usize {
        return Err(PackError::invalid(format!(
            "builder produced {} chunks, expected {}",
            pack.chunks.len(),
            pack.header.chunk_count
        )));
    }
    let directory_len = usize_from_u64(
        pack.header.payload_offset - SUPERBLOCK_BYTES as u64,
        "directory length",
    )?;
    let mut bytes = vec![0_u8; directory_len];
    let records_base = relative_directory_offset(pack.header.records_offset)?;
    for (index, tensor) in pack.tensors.iter().enumerate() {
        let start = records_base + index * RECORD_BYTES;
        encode_record(&tensor.record, &mut bytes[start..start + RECORD_BYTES]);
    }
    let strings_base = relative_directory_offset(pack.header.strings_offset)?;
    bytes[strings_base..strings_base + pack.strings.len()].copy_from_slice(&pack.strings);
    let chunks_base = relative_directory_offset(pack.header.chunks_offset)?;
    for (index, chunk) in pack.chunks.iter().enumerate() {
        let start = chunks_base + index * CHUNK_RECORD_BYTES;
        encode_chunk(chunk, &mut bytes[start..start + CHUNK_RECORD_BYTES]);
    }
    Ok(bytes)
}

fn encode_header(header: &PackHeader) -> [u8; SUPERBLOCK_BYTES] {
    let mut out = [0_u8; SUPERBLOCK_BYTES];
    out[0..8].copy_from_slice(&MAGIC);
    put_u16(&mut out, 8, VERSION_MAJOR);
    put_u16(&mut out, 10, VERSION_MINOR);
    put_u32(&mut out, 12, SUPERBLOCK_BYTES as u32);
    put_u32(&mut out, 16, ENDIAN_TAG);
    put_u32(&mut out, 20, REQUIRED_FLAGS);
    put_u32(&mut out, 24, header.layer);
    put_u32(&mut out, 28, RECORD_BYTES as u32);
    put_u32(&mut out, 32, header.record_count);
    put_u32(&mut out, 36, HASH_SHA256);
    put_u32(&mut out, 40, header.chunk_bytes);
    put_u32(&mut out, 44, header.chunk_count);
    put_u64(&mut out, 48, header.file_bytes);
    put_u64(&mut out, 56, header.records_offset);
    put_u64(&mut out, 64, header.records_bytes);
    put_u64(&mut out, 72, header.strings_offset);
    put_u64(&mut out, 80, header.strings_bytes);
    put_u64(&mut out, 88, header.chunks_offset);
    put_u64(&mut out, 96, header.chunks_bytes);
    put_u64(&mut out, 104, header.payload_offset);
    put_u64(&mut out, 112, header.payload_bytes);
    put_u64(&mut out, 120, header.quantized_offset);
    put_u64(&mut out, 128, header.quantized_used);
    put_u64(&mut out, 136, header.quantized_span);
    put_u64(&mut out, 144, header.scales_offset);
    put_u64(&mut out, 152, header.scales_used);
    put_u64(&mut out, 160, header.scales_span);
    put_u64(&mut out, 168, header.raw_offset);
    put_u64(&mut out, 176, header.raw_used);
    put_u64(&mut out, 184, header.raw_span);
    out[192..224].copy_from_slice(&header.identity.model);
    out[224..256].copy_from_slice(&header.identity.source_inventory);
    out[256..288].copy_from_slice(&header.identity.layout_schema);
    out[DIRECTORY_DIGEST_OFFSET..DIRECTORY_DIGEST_OFFSET + DIGEST_BYTES]
        .copy_from_slice(&header.directory_digest);
    out[PAYLOAD_DIGEST_OFFSET..PAYLOAD_DIGEST_OFFSET + DIGEST_BYTES]
        .copy_from_slice(&header.payload_digest);
    out[PACK_DIGEST_OFFSET..PACK_DIGEST_OFFSET + DIGEST_BYTES].copy_from_slice(&header.pack_digest);
    out[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + DIGEST_BYTES]
        .copy_from_slice(&header.header_digest);
    out
}

fn decode_header(
    bytes: &[u8; SUPERBLOCK_BYTES],
    actual_file_bytes: u64,
    limits: ParseLimits,
) -> Result<PackHeader> {
    if bytes[0..8] != MAGIC {
        return Err(PackError::invalid("invalid DFSP magic"));
    }
    if get_u16(bytes, 8) != VERSION_MAJOR || get_u16(bytes, 10) != VERSION_MINOR {
        return Err(PackError::invalid(format!(
            "unsupported DFSP version {}.{}",
            get_u16(bytes, 8),
            get_u16(bytes, 10)
        )));
    }
    if get_u32(bytes, 12) != SUPERBLOCK_BYTES as u32 {
        return Err(PackError::invalid("unexpected superblock size"));
    }
    if get_u32(bytes, 16) != ENDIAN_TAG {
        return Err(PackError::invalid("invalid little-endian marker"));
    }
    if get_u32(bytes, 20) != REQUIRED_FLAGS {
        return Err(PackError::invalid("unknown or missing DFSP feature flags"));
    }
    if get_u32(bytes, 28) != RECORD_BYTES as u32 {
        return Err(PackError::invalid("unexpected tensor-record size"));
    }
    if get_u32(bytes, 36) != HASH_SHA256 {
        return Err(PackError::invalid("unsupported DFSP hash algorithm"));
    }
    if bytes[416..].iter().any(|byte| *byte != 0) {
        return Err(PackError::invalid("non-zero reserved superblock bytes"));
    }
    let expected_header_digest = array_at::<DIGEST_BYTES>(bytes, HEADER_DIGEST_OFFSET)?;
    let observed_header_digest =
        sha256_with_zeroed_range(bytes, HEADER_DIGEST_OFFSET, DIGEST_BYTES);
    if observed_header_digest != expected_header_digest {
        return Err(PackError::invalid("superblock SHA-256 mismatch"));
    }

    let identity = PackIdentity {
        model: array_at::<DIGEST_BYTES>(bytes, 192)?,
        source_inventory: array_at::<DIGEST_BYTES>(bytes, 224)?,
        layout_schema: array_at::<DIGEST_BYTES>(bytes, 256)?,
    };
    identity.validate()?;
    let header = PackHeader {
        layer: get_u32(bytes, 24),
        identity,
        record_count: get_u32(bytes, 32),
        chunk_bytes: get_u32(bytes, 40),
        chunk_count: get_u32(bytes, 44),
        file_bytes: get_u64(bytes, 48),
        records_offset: get_u64(bytes, 56),
        records_bytes: get_u64(bytes, 64),
        strings_offset: get_u64(bytes, 72),
        strings_bytes: get_u64(bytes, 80),
        chunks_offset: get_u64(bytes, 88),
        chunks_bytes: get_u64(bytes, 96),
        payload_offset: get_u64(bytes, 104),
        payload_bytes: get_u64(bytes, 112),
        quantized_offset: get_u64(bytes, 120),
        quantized_used: get_u64(bytes, 128),
        quantized_span: get_u64(bytes, 136),
        scales_offset: get_u64(bytes, 144),
        scales_used: get_u64(bytes, 152),
        scales_span: get_u64(bytes, 160),
        raw_offset: get_u64(bytes, 168),
        raw_used: get_u64(bytes, 176),
        raw_span: get_u64(bytes, 184),
        directory_digest: array_at::<DIGEST_BYTES>(bytes, DIRECTORY_DIGEST_OFFSET)?,
        payload_digest: array_at::<DIGEST_BYTES>(bytes, PAYLOAD_DIGEST_OFFSET)?,
        pack_digest: array_at::<DIGEST_BYTES>(bytes, PACK_DIGEST_OFFSET)?,
        header_digest: expected_header_digest,
    };
    validate_header_layout(&header, actual_file_bytes, limits)?;
    if compute_pack_digest(&header) != header.pack_digest {
        return Err(PackError::invalid("pack-root SHA-256 mismatch"));
    }
    Ok(header)
}

fn validate_header_layout(
    header: &PackHeader,
    actual_file_bytes: u64,
    limits: ParseLimits,
) -> Result<()> {
    validate_chunk_bytes(header.chunk_bytes)?;
    if header.file_bytes != actual_file_bytes {
        return Err(PackError::invalid(format!(
            "declared file length {} does not match actual length {actual_file_bytes}",
            header.file_bytes
        )));
    }
    if header.file_bytes > limits.max_file_bytes || header.payload_bytes > limits.max_payload_bytes
    {
        return Err(PackError::invalid(
            "declared pack size exceeds parsing limits",
        ));
    }
    if header.record_count == 0 || header.record_count > limits.max_records {
        return Err(PackError::invalid(format!(
            "record count {} is outside 1..={} ",
            header.record_count, limits.max_records
        )));
    }
    if header.chunk_count == 0 || header.chunk_count > limits.max_chunks {
        return Err(PackError::invalid(format!(
            "chunk count {} is outside 1..={} ",
            header.chunk_count, limits.max_chunks
        )));
    }
    let expected_records_bytes = checked_mul(
        header.record_count as u64,
        RECORD_BYTES as u64,
        "record-table length",
    )?;
    if header.records_offset != SUPERBLOCK_BYTES as u64
        || header.records_bytes != expected_records_bytes
    {
        return Err(PackError::invalid("non-canonical record-table layout"));
    }
    let expected_strings_offset = align_up(
        checked_add(
            header.records_offset,
            header.records_bytes,
            "string-table offset",
        )?,
        8,
    )?;
    if header.strings_offset != expected_strings_offset {
        return Err(PackError::invalid("non-canonical string-table offset"));
    }
    let expected_chunks_offset = align_up(
        checked_add(
            header.strings_offset,
            header.strings_bytes,
            "chunk-table offset",
        )?,
        8,
    )?;
    if header.chunks_offset != expected_chunks_offset {
        return Err(PackError::invalid("non-canonical chunk-table offset"));
    }
    let expected_chunks_bytes = checked_mul(
        header.chunk_count as u64,
        CHUNK_RECORD_BYTES as u64,
        "chunk-table length",
    )?;
    if header.chunks_bytes != expected_chunks_bytes {
        return Err(PackError::invalid("non-canonical chunk-table length"));
    }
    let expected_payload_offset = align_up(
        checked_add(header.chunks_offset, header.chunks_bytes, "payload offset")?,
        PAYLOAD_ALIGNMENT,
    )?;
    if header.payload_offset != expected_payload_offset {
        return Err(PackError::invalid("non-canonical payload offset"));
    }
    let metadata_bytes = header.payload_offset - SUPERBLOCK_BYTES as u64;
    if metadata_bytes > limits.max_metadata_bytes {
        return Err(PackError::invalid("metadata exceeds parsing limit"));
    }
    if header.strings_bytes > limits.max_metadata_bytes {
        return Err(PackError::invalid("string table exceeds parsing limit"));
    }
    if checked_add(header.payload_offset, header.payload_bytes, "file length")? != header.file_bytes
    {
        return Err(PackError::invalid(
            "payload does not exactly cover file tail",
        ));
    }
    if header.quantized_offset != 0
        || header.quantized_span % PAYLOAD_ALIGNMENT != 0
        || header.scales_offset
            != checked_add(
                header.quantized_offset,
                header.quantized_span,
                "scale-section offset",
            )?
        || header.scales_span % PAYLOAD_ALIGNMENT != 0
        || header.raw_offset
            != checked_add(
                header.scales_offset,
                header.scales_span,
                "raw-section offset",
            )?
        || header.raw_span % PAYLOAD_ALIGNMENT != 0
        || checked_add(header.raw_offset, header.raw_span, "payload end")? != header.payload_bytes
    {
        return Err(PackError::invalid("non-canonical payload-section layout"));
    }
    for (name, used, span) in [
        ("quantized", header.quantized_used, header.quantized_span),
        ("scales", header.scales_used, header.scales_span),
        ("raw", header.raw_used, header.raw_span),
    ] {
        if used > span || (used == 0) != (span == 0) {
            return Err(PackError::invalid(format!(
                "invalid {name} section used/span values {used}/{span}"
            )));
        }
    }
    let expected_chunks = ceil_div(header.payload_bytes, header.chunk_bytes as u64)?;
    if expected_chunks != header.chunk_count as u64 {
        return Err(PackError::invalid(
            "chunk count does not cover payload exactly",
        ));
    }
    Ok(())
}

fn encode_record(record: &TensorRecord, out: &mut [u8]) {
    debug_assert_eq!(out.len(), RECORD_BYTES);
    out.fill(0);
    out[0..32].copy_from_slice(&record.name_digest);
    put_u32(out, 32, record.name_offset);
    put_u16(out, 36, record.name_length);
    out[38] = record.codec as u8;
    out[39] = record.logical_dtype as u8;
    out[40] = record.data_dtype as u8;
    out[41] = record.auxiliary_dtype as u8;
    out[42] = record.rank;
    out[43] = record.scale_axis as u8;
    put_u32(out, 44, record.flags);
    put_u16(out, 48, record.upload_group);
    put_u16(out, 50, record.upload_order);
    for (index, dimension) in record.shape.iter().enumerate() {
        put_u64(out, 56 + index * 8, *dimension);
    }
    put_u64(out, 120, record.data_offset);
    put_u64(out, 128, record.data_length);
    put_u64(out, 136, record.auxiliary_offset);
    put_u64(out, 144, record.auxiliary_length);
    put_u64(out, 152, record.element_count);
    put_u32(out, 160, record.rows);
    put_u32(out, 164, record.columns);
    out[168..200].copy_from_slice(&record.data_digest);
    out[200..232].copy_from_slice(&record.auxiliary_digest);
}

fn decode_records(
    header: &PackHeader,
    directory: &[u8],
    limits: ParseLimits,
) -> Result<Vec<TensorRecord>> {
    let records_base = relative_directory_offset(header.records_offset)?;
    let strings_base = relative_directory_offset(header.strings_offset)?;
    let strings_len = usize_from_u64(header.strings_bytes, "string-table length")?;
    let strings = checked_slice(directory, strings_base, strings_len, "string table")?;
    let mut expected_name_offset = 0_usize;
    let mut previous_name: Option<String> = None;
    let mut records = Vec::with_capacity(header.record_count as usize);

    for index in 0..header.record_count as usize {
        let start = records_base
            .checked_add(index * RECORD_BYTES)
            .ok_or_else(|| PackError::invalid("record offset overflows usize"))?;
        let bytes = checked_slice(directory, start, RECORD_BYTES, "tensor record")?;
        if bytes[52..56].iter().any(|byte| *byte != 0)
            || bytes[232..256].iter().any(|byte| *byte != 0)
        {
            return Err(PackError::invalid(format!(
                "tensor record {index} has non-zero reserved bytes"
            )));
        }
        let name_offset = get_u32(bytes, 32);
        let name_length = get_u16(bytes, 36);
        if name_length == 0 || name_length as u32 > limits.max_name_bytes {
            return Err(PackError::invalid(format!(
                "tensor record {index} has invalid name length {name_length}"
            )));
        }
        if name_offset as usize != expected_name_offset {
            return Err(PackError::invalid(format!(
                "tensor record {index} does not canonically cover the string table"
            )));
        }
        let name_bytes = checked_slice(
            strings,
            name_offset as usize,
            name_length as usize,
            "tensor name",
        )?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| PackError::invalid(format!("tensor record {index} name is not UTF-8")))?
            .to_owned();
        validate_name(header.layer, &name, limits.max_name_bytes)?;
        if let Some(previous) = &previous_name {
            if previous.as_bytes() >= name.as_bytes() {
                return Err(PackError::invalid(
                    "tensor records are not strictly sorted by canonical name",
                ));
            }
        }
        previous_name = Some(name.clone());
        expected_name_offset = expected_name_offset
            .checked_add(name_length as usize)
            .ok_or_else(|| PackError::invalid("string-table coverage overflows usize"))?;

        let name_digest = array_at::<DIGEST_BYTES>(bytes, 0)?;
        if sha256(name.as_bytes()) != name_digest {
            return Err(PackError::invalid(format!(
                "tensor {name} name digest mismatch"
            )));
        }
        let codec = Codec::parse(bytes[38])?;
        let logical_dtype = DType::parse(bytes[39])?;
        let data_dtype = DType::parse(bytes[40])?;
        let auxiliary_dtype = DType::parse(bytes[41])?;
        let rank = bytes[42];
        let scale_axis = bytes[43] as i8;
        let flags = get_u32(bytes, 44);
        let mut shape = [0_u64; 8];
        for (dimension, slot) in shape.iter_mut().enumerate() {
            *slot = get_u64(bytes, 56 + dimension * 8);
        }
        records.push(TensorRecord {
            name,
            name_digest,
            codec,
            logical_dtype,
            data_dtype,
            auxiliary_dtype,
            rank,
            scale_axis,
            flags,
            upload_group: get_u16(bytes, 48),
            upload_order: get_u16(bytes, 50),
            shape,
            data_offset: get_u64(bytes, 120),
            data_length: get_u64(bytes, 128),
            auxiliary_offset: get_u64(bytes, 136),
            auxiliary_length: get_u64(bytes, 144),
            element_count: get_u64(bytes, 152),
            rows: get_u32(bytes, 160),
            columns: get_u32(bytes, 164),
            data_digest: array_at::<DIGEST_BYTES>(bytes, 168)?,
            auxiliary_digest: array_at::<DIGEST_BYTES>(bytes, 200)?,
            name_offset,
            name_length,
        });
    }
    if expected_name_offset != strings.len() {
        return Err(PackError::invalid(
            "tensor records do not completely cover the string table",
        ));
    }
    validate_directory_zero_padding(header, directory)?;
    Ok(records)
}

fn validate_directory_zero_padding(header: &PackHeader, directory: &[u8]) -> Result<()> {
    let records_end = relative_directory_offset(header.records_offset + header.records_bytes)?;
    let strings_start = relative_directory_offset(header.strings_offset)?;
    require_zero(
        &directory[records_end..strings_start],
        "record/string padding",
    )?;
    let strings_end = relative_directory_offset(header.strings_offset + header.strings_bytes)?;
    let chunks_start = relative_directory_offset(header.chunks_offset)?;
    require_zero(
        &directory[strings_end..chunks_start],
        "string/chunk padding",
    )?;
    let chunks_end = relative_directory_offset(header.chunks_offset + header.chunks_bytes)?;
    require_zero(&directory[chunks_end..], "directory/payload padding")
}

fn encode_chunk(chunk: &ChunkRecord, out: &mut [u8]) {
    debug_assert_eq!(out.len(), CHUNK_RECORD_BYTES);
    out.fill(0);
    put_u64(out, 0, chunk.payload_offset);
    put_u32(out, 8, chunk.length);
    out[16..48].copy_from_slice(&chunk.digest);
}

fn decode_chunks(
    header: &PackHeader,
    directory: &[u8],
    limits: ParseLimits,
) -> Result<Vec<ChunkRecord>> {
    if header.chunk_count > limits.max_chunks {
        return Err(PackError::invalid("chunk count exceeds parsing limit"));
    }
    let base = relative_directory_offset(header.chunks_offset)?;
    let mut chunks = Vec::with_capacity(header.chunk_count as usize);
    let mut expected_offset = 0_u64;
    for index in 0..header.chunk_count as usize {
        let start = base
            .checked_add(index * CHUNK_RECORD_BYTES)
            .ok_or_else(|| PackError::invalid("chunk-record offset overflows usize"))?;
        let bytes = checked_slice(directory, start, CHUNK_RECORD_BYTES, "chunk record")?;
        if bytes[12..16].iter().any(|byte| *byte != 0) {
            return Err(PackError::invalid(format!(
                "chunk record {index} has non-zero reserved flags"
            )));
        }
        let payload_offset = get_u64(bytes, 0);
        let length = get_u32(bytes, 8);
        if payload_offset != expected_offset {
            return Err(PackError::invalid(format!(
                "chunk {index} starts at {payload_offset}, expected {expected_offset}"
            )));
        }
        let remaining = header.payload_bytes - expected_offset;
        let expected_length = remaining.min(header.chunk_bytes as u64) as u32;
        if length != expected_length || length == 0 {
            return Err(PackError::invalid(format!(
                "chunk {index} length {length} does not equal canonical length {expected_length}"
            )));
        }
        expected_offset = checked_add(expected_offset, length as u64, "chunk coverage")?;
        chunks.push(ChunkRecord {
            payload_offset,
            length,
            digest: array_at::<DIGEST_BYTES>(bytes, 16)?,
        });
    }
    if expected_offset != header.payload_bytes {
        return Err(PackError::invalid("chunk table does not cover payload"));
    }
    Ok(chunks)
}

fn validate_tensor_layout(header: &PackHeader, tensors: &[TensorRecord]) -> Result<()> {
    for tensor in tensors {
        validate_record_semantics(tensor)?;
    }
    let q_order = record_physical_order(tensors, Codec::RowI8F16Scale);
    let raw_order = record_physical_order(tensors, Codec::Raw);
    validate_section_records(
        header,
        tensors,
        &q_order,
        SectionPart::Data,
        header.quantized_offset,
        header.quantized_used,
        header.quantized_span,
        "quantized",
    )?;
    validate_section_records(
        header,
        tensors,
        &q_order,
        SectionPart::Auxiliary,
        header.scales_offset,
        header.scales_used,
        header.scales_span,
        "scales",
    )?;
    validate_section_records(
        header,
        tensors,
        &raw_order,
        SectionPart::Data,
        header.raw_offset,
        header.raw_used,
        header.raw_span,
        "raw",
    )?;
    Ok(())
}

fn validate_record_semantics(record: &TensorRecord) -> Result<()> {
    if record.rank == 0 || record.rank > 8 {
        return Err(PackError::invalid(format!(
            "{} has invalid rank {}",
            record.name, record.rank
        )));
    }
    let rank = record.rank as usize;
    if record.shape[..rank].contains(&0) || record.shape[rank..].iter().any(|dim| *dim != 0) {
        return Err(PackError::invalid(format!(
            "{} has non-canonical shape dimensions",
            record.name
        )));
    }
    let elements = checked_product(&record.shape[..rank], "record element count")?;
    if elements != record.element_count {
        return Err(PackError::invalid(format!(
            "{} element count {} does not match shape product {elements}",
            record.name, record.element_count
        )));
    }
    if record.data_length == 0 || record.data_offset % COMPONENT_ALIGNMENT != 0 {
        return Err(PackError::invalid(format!(
            "{} has an empty or misaligned data component",
            record.name
        )));
    }
    match record.codec {
        Codec::Raw => {
            let width = record.logical_dtype.byte_width().ok_or_else(|| {
                PackError::invalid(format!("{} raw tensor has no dtype width", record.name))
            })?;
            let expected = checked_mul(elements, width, "raw record bytes")?;
            let (expected_rows, expected_columns) = if rank == 2 {
                (
                    u32::try_from(record.shape[0])
                        .map_err(|_| PackError::invalid("raw rows do not fit in u32"))?,
                    u32::try_from(record.shape[1])
                        .map_err(|_| PackError::invalid("raw columns do not fit in u32"))?,
                )
            } else {
                (0, 0)
            };
            if record.data_dtype != record.logical_dtype
                || record.auxiliary_dtype != DType::None
                || record.scale_axis != -1
                || record.flags != RECORD_FLAG_IMMUTABLE
                || record.data_length != expected
                || record.auxiliary_offset != 0
                || record.auxiliary_length != 0
                || record.auxiliary_digest != [0; DIGEST_BYTES]
                || record.rows != expected_rows
                || record.columns != expected_columns
            {
                return Err(PackError::invalid(format!(
                    "{} has invalid RAW codec metadata",
                    record.name
                )));
            }
        }
        Codec::RowI8F16Scale => {
            let rows = u32::try_from(record.shape[0])
                .map_err(|_| PackError::invalid("q8 rows do not fit in u32"))?;
            let columns = u32::try_from(record.shape[1])
                .map_err(|_| PackError::invalid("q8 columns do not fit in u32"))?;
            if rank != 2
                || !matches!(record.logical_dtype, DType::Bf16 | DType::F32)
                || record.data_dtype != DType::I8
                || record.auxiliary_dtype != DType::F16
                || record.scale_axis != 0
                || record.flags != RECORD_FLAG_ROW_MAJOR | RECORD_FLAG_IMMUTABLE
                || record.data_length != elements
                || record.auxiliary_length != rows as u64 * 2
                || record.auxiliary_offset % COMPONENT_ALIGNMENT != 0
                || record.rows != rows
                || record.columns != columns
            {
                return Err(PackError::invalid(format!(
                    "{} has invalid row-I8/F16-scale metadata",
                    record.name
                )));
            }
        }
    }
    Ok(())
}

fn record_physical_order(tensors: &[TensorRecord], codec: Codec) -> Vec<usize> {
    let mut indices: Vec<usize> = tensors
        .iter()
        .enumerate()
        .filter_map(|(index, tensor)| (tensor.codec == codec).then_some(index))
        .collect();
    indices.sort_by(|left, right| {
        let left = &tensors[*left];
        let right = &tensors[*right];
        (left.upload_group, left.upload_order, left.name.as_bytes()).cmp(&(
            right.upload_group,
            right.upload_order,
            right.name.as_bytes(),
        ))
    });
    indices
}

#[allow(clippy::too_many_arguments)]
fn validate_section_records(
    header: &PackHeader,
    tensors: &[TensorRecord],
    order: &[usize],
    part: SectionPart,
    section_offset: u64,
    declared_used: u64,
    declared_span: u64,
    section_name: &str,
) -> Result<()> {
    let mut cursor = section_offset;
    let mut used = 0_u64;
    for &index in order {
        cursor = align_up(cursor, COMPONENT_ALIGNMENT)?;
        let record = &tensors[index];
        let (offset, length) = match part {
            SectionPart::Data => (record.data_offset, record.data_length),
            SectionPart::Auxiliary => (record.auxiliary_offset, record.auxiliary_length),
        };
        if offset != cursor {
            return Err(PackError::invalid(format!(
                "{} {section_name} component starts at {offset}, expected {cursor}",
                record.name
            )));
        }
        used = checked_add(used, length, "section component coverage")?;
        cursor = checked_add(cursor, length, "section component end")?;
        if cursor > header.payload_bytes {
            return Err(PackError::invalid(format!(
                "{} component exceeds payload",
                record.name
            )));
        }
    }
    let expected_span = if order.is_empty() {
        0
    } else {
        align_up(cursor - section_offset, PAYLOAD_ALIGNMENT)?
    };
    if used != declared_used || expected_span != declared_span {
        return Err(PackError::invalid(format!(
            "{section_name} component coverage {used}/{expected_span} does not match declared {declared_used}/{declared_span}"
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct ComponentRange<'a> {
    start: u64,
    end: u64,
    expected_digest: Digest,
    name: &'a str,
    part: &'static str,
}

fn component_ranges(tensors: &[TensorRecord]) -> Result<Vec<ComponentRange<'_>>> {
    let mut ranges = Vec::new();
    for tensor in tensors {
        ranges.push(ComponentRange {
            start: tensor.data_offset,
            end: checked_add(tensor.data_offset, tensor.data_length, "data range")?,
            expected_digest: tensor.data_digest,
            name: &tensor.name,
            part: "data",
        });
        if tensor.auxiliary_length != 0 {
            ranges.push(ComponentRange {
                start: tensor.auxiliary_offset,
                end: checked_add(
                    tensor.auxiliary_offset,
                    tensor.auxiliary_length,
                    "auxiliary range",
                )?,
                expected_digest: tensor.auxiliary_digest,
                name: &tensor.name,
                part: "auxiliary",
            });
        }
    }
    ranges.sort_by_key(|range| range.start);
    for pair in ranges.windows(2) {
        if pair[1].start < pair[0].end {
            return Err(PackError::invalid(format!(
                "component ranges overlap at byte {}",
                pair[1].start
            )));
        }
    }
    Ok(ranges)
}

fn verify_chunk_coverage(
    chunk_offset: u64,
    bytes: &[u8],
    ranges: &[ComponentRange<'_>],
    hashers: &mut [Sha256],
) -> Result<()> {
    let chunk_end = checked_add(chunk_offset, bytes.len() as u64, "chunk end")?;
    let mut cursor = chunk_offset;
    for (index, range) in ranges.iter().enumerate() {
        if range.end <= chunk_offset {
            continue;
        }
        if range.start >= chunk_end {
            break;
        }
        let intersection_start = range.start.max(chunk_offset);
        let intersection_end = range.end.min(chunk_end);
        if cursor < intersection_start {
            require_zero(
                payload_slice(bytes, chunk_offset, cursor, intersection_start)?,
                "payload alignment padding",
            )?;
        }
        let component = payload_slice(bytes, chunk_offset, intersection_start, intersection_end)?;
        hashers[index].update(component);
        cursor = cursor.max(intersection_end);
    }
    if cursor < chunk_end {
        require_zero(
            payload_slice(bytes, chunk_offset, cursor, chunk_end)?,
            "payload tail padding",
        )?;
    }
    Ok(())
}

fn payload_slice(bytes: &[u8], base: u64, start: u64, end: u64) -> Result<&[u8]> {
    let start = usize_from_u64(
        start
            .checked_sub(base)
            .ok_or_else(|| PackError::invalid("payload slice starts before chunk"))?,
        "payload slice start",
    )?;
    let end = usize_from_u64(
        end.checked_sub(base)
            .ok_or_else(|| PackError::invalid("payload slice ends before chunk"))?,
        "payload slice end",
    )?;
    checked_slice(bytes, start, end - start, "payload slice")
}

fn validate_limits(limits: ParseLimits) -> Result<()> {
    if limits.max_file_bytes < SUPERBLOCK_BYTES as u64
        || limits.max_payload_bytes == 0
        || limits.max_metadata_bytes == 0
        || limits.max_records == 0
        || limits.max_name_bytes == 0
        || limits.max_chunks == 0
    {
        return Err(PackError::invalid("parse limits must all be positive"));
    }
    Ok(())
}

fn validate_chunk_bytes(bytes: u32) -> Result<()> {
    if !(64 * 1024..=64 * 1024 * 1024).contains(&bytes) || !bytes.is_power_of_two() {
        return Err(PackError::invalid(format!(
            "chunk size {bytes} must be a power of two in 64 KiB..=64 MiB"
        )));
    }
    Ok(())
}

fn compute_pack_digest(header: &PackHeader) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"DFSP-PACK-V1\0");
    hasher.update(&header.layer.to_le_bytes());
    hasher.update(&header.identity.model);
    hasher.update(&header.identity.source_inventory);
    hasher.update(&header.identity.layout_schema);
    hasher.update(&header.directory_digest);
    hasher.update(&header.payload_digest);
    hasher.finalize()
}

fn read_exact_at(file: &mut File, offset: u64, bytes: &mut [u8], path: &Path) -> Result<()> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| PackError::io("seek", path, error))?;
    file.read_exact(bytes)
        .map_err(|error| PackError::io("read", path, error))
}

fn write_zeroes(
    file: &mut File,
    mut length: usize,
    path: &Path,
    operation: &'static str,
) -> Result<()> {
    let zeroes = [0_u8; 64 * 1024];
    while length != 0 {
        let count = length.min(zeroes.len());
        file.write_all(&zeroes[..count])
            .map_err(|error| PackError::io(operation, path, error))?;
        length -= count;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    let directory = Open::new()
        .read(true)
        .directory()
        .open(path)
        .map_err(|error| PackError::io("open directory for fsync", path, error))?;
    crate::sys::fs::sync_directory(&directory)
        .map_err(|error| PackError::io("fsync directory", path, error))
}

fn destination_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn relative_directory_offset(absolute: u64) -> Result<usize> {
    usize_from_u64(
        absolute
            .checked_sub(SUPERBLOCK_BYTES as u64)
            .ok_or_else(|| PackError::invalid("metadata offset precedes superblock"))?,
        "relative metadata offset",
    )
}

fn checked_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    length: usize,
    label: &str,
) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| PackError::invalid(format!("{label} range overflows usize")))?;
    bytes
        .get(offset..end)
        .ok_or_else(|| PackError::invalid(format!("{label} range is out of bounds")))
}

fn require_zero(bytes: &[u8], label: &str) -> Result<()> {
    if bytes.iter().any(|byte| *byte != 0) {
        return Err(PackError::invalid(format!("non-zero {label}")));
    }
    Ok(())
}

fn checked_product(values: &[u64], label: &str) -> Result<u64> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| PackError::invalid(format!("{label} overflows u64")))
    })
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| PackError::invalid(format!("{label} overflows u64")))
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| PackError::invalid(format!("{label} overflows u64")))
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(PackError::invalid(
            "alignment must be a non-zero power of two",
        ));
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| PackError::invalid("aligned offset overflows u64"))
}

fn ceil_div(value: u64, divisor: u64) -> Result<u64> {
    if divisor == 0 {
        return Err(PackError::invalid("division by zero"));
    }
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or_else(|| PackError::invalid("ceiling division overflows u64"))
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| PackError::invalid(format!("{label} does not fit in usize")))
}

fn u32_from_usize(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| PackError::invalid(format!("{label} does not fit in u32")))
}

fn array_at<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N]> {
    let slice = checked_slice(bytes, offset, N, "fixed-size field")?;
    let mut out = [0_u8; N];
    out.copy_from_slice(slice);
    Ok(out)
}

fn get_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn sha256_with_zeroed_range(bytes: &[u8], offset: usize, length: usize) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(&bytes[..offset]);
    let zeroes = [0_u8; DIGEST_BYTES];
    debug_assert!(length <= zeroes.len());
    hasher.update(&zeroes[..length]);
    hasher.update(&bytes[offset + length..]);
    hasher.finalize()
}

/// Hash a small canonical metadata buffer with the same SHA-256 implementation
/// used by DFSP records.  Exposing this avoids a second digest implementation
/// when the execution roster derives its layout identity.
pub fn digest_bytes(bytes: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize()
}

/// Incremental access to the same exact SHA-256 implementation used
/// by immutable packs. Large native converters use this while copying a source
/// exactly once, avoiding both a second multi-terabyte scan and a parallel
/// digest implementation.
#[derive(Debug, Clone)]
pub(crate) struct DigestState(Sha256);

impl DigestState {
    pub(crate) fn new() -> Self {
        Self(Sha256::new())
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    pub(crate) fn finalize(self) -> Digest {
        self.0.finalize()
    }
}

fn sha256(bytes: &[u8]) -> Digest {
    digest_bytes(bytes)
}

/// Stream a file into the DFSP SHA-256 implementation without retaining it in
/// memory.  This is used only while deriving immutable pack identities.
pub fn digest_file(path: impl AsRef<Path>) -> Result<Digest> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|error| PackError::io("open", path, error))?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut hasher = Sha256::new();
    loop {
        match file.read(&mut buffer) {
            Ok(0) => return Ok(hasher.finalize()),
            Ok(length) => hasher.update(&buffer[..length]),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(PackError::io("read", path, error)),
        }
    }
}

/// Hash one already-open, already-admitted descriptor without reopening its
/// path or disturbing its shared file cursor. Native checkpoint loaders use
/// this after `O_NOFOLLOW`/inode/length validation so integrity never races a
/// second pathname lookup.
pub(crate) fn digest_open_file(file: &File, path: &Path) -> Result<Digest> {
    let length = file
        .metadata()
        .map_err(|error| PackError::io("stat", path, error))?
        .len();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut offset = 0_u64;
    let mut hasher = Sha256::new();
    while offset < length {
        let remaining = usize::try_from((length - offset).min(buffer.len() as u64))
            .map_err(|_| PackError::invalid("open-file digest range exceeds usize"))?;
        match file.read_at(&mut buffer[..remaining], offset) {
            Ok(0) => {
                return Err(PackError::invalid(format!(
                    "{} ended while hashing at byte {offset} of {length}",
                    path.display()
                )));
            }
            Ok(read) => {
                hasher.update(&buffer[..read]);
                offset = offset
                    .checked_add(read as u64)
                    .ok_or_else(|| PackError::invalid("open-file digest offset overflowed"))?;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(PackError::io("read", path, error)),
        }
    }
    Ok(hasher.finalize())
}

/// Exact SHA-256 wrapper used for immutable-pack integrity. Apple platforms
/// delegate compression to the system CommonCrypto implementation; other
/// targets use RustCrypto's CPU-dispatched implementation. Both produce the
/// same standard digest bytes and preserve every on-disk format.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Clone)]
struct CommonCryptoSha256Context {
    count: [u32; 2],
    hash: [u32; 8],
    work: [u32; 16],
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn CC_SHA256_Init(context: *mut CommonCryptoSha256Context) -> i32;
    fn CC_SHA256_Update(
        context: *mut CommonCryptoSha256Context,
        input: *const std::ffi::c_void,
        length: u32,
    ) -> i32;
    fn CC_SHA256_Final(output: *mut u8, context: *mut CommonCryptoSha256Context) -> i32;
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct Sha256(CommonCryptoSha256Context);

#[cfg(target_os = "macos")]
impl Sha256 {
    fn new() -> Self {
        let mut context = CommonCryptoSha256Context {
            count: [0; 2],
            hash: [0; 8],
            work: [0; 16],
        };
        // SAFETY: `context` has the exact public CC_SHA256_CTX layout and is
        // exclusively borrowed for initialization.
        let initialized = unsafe { CC_SHA256_Init(&mut context) };
        assert_eq!(initialized, 1, "CommonCrypto SHA-256 initialization failed");
        Self(context)
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            let length = input.len().min(u32::MAX as usize);
            // SAFETY: the input slice is live for `length` bytes and the
            // context is exclusively borrowed for the duration of the call.
            let updated = unsafe {
                CC_SHA256_Update(
                    &mut self.0,
                    input.as_ptr().cast::<std::ffi::c_void>(),
                    length as u32,
                )
            };
            assert_eq!(updated, 1, "CommonCrypto SHA-256 update failed");
            input = &input[length..];
        }
    }

    fn finalize(mut self) -> Digest {
        let mut output = [0_u8; DIGEST_BYTES];
        // SAFETY: `output` has the required 32-byte capacity and `self.0` is
        // exclusively owned until finalization returns.
        let finalized = unsafe { CC_SHA256_Final(output.as_mut_ptr(), &mut self.0) };
        assert_eq!(finalized, 1, "CommonCrypto SHA-256 finalization failed");
        output
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone)]
struct Sha256(sha2::Sha256);

#[cfg(not(target_os = "macos"))]
impl Sha256 {
    fn new() -> Self {
        use sha2::Digest;
        Self(sha2::Sha256::new())
    }

    fn update(&mut self, input: &[u8]) {
        use sha2::Digest;
        self.0.update(input);
    }

    fn finalize(self) -> Digest {
        use sha2::Digest;
        self.0.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-dfsp-{label}-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn identity() -> PackIdentity {
        PackIdentity::new([1; 32], [2; 32], [3; 32])
    }

    fn tensor_name(layer: u32, suffix: &str) -> String {
        format!("language_model.model.layers.{layer}.{suffix}")
    }

    fn sample_builder(reverse: bool) -> PackBuilder {
        let mut builder = PackBuilder::new(7, identity()).unwrap();
        builder.set_chunk_bytes(64 * 1024).unwrap();
        let q = BuildTensor::row_i8_f16_scale(
            tensor_name(7, "self_attn.q_proj.weight"),
            DType::Bf16,
            [2, 3],
            ComponentSource::bytes(vec![0x80, 0xff, 0, 1, 2, 0x7f]),
            ComponentSource::bytes(vec![0x00, 0x3c, 0x00, 0x40]),
            1,
            0,
        );
        let raw = BuildTensor::raw(
            tensor_name(7, "input_layernorm.weight"),
            DType::Bf16,
            vec![3],
            ComponentSource::bytes(vec![0x00, 0x3f, 0x80, 0x3f, 0x00, 0x40]),
            2,
            0,
        );
        if reverse {
            builder.push(q);
            builder.push(raw);
        } else {
            builder.push(raw);
            builder.push(q);
        }
        builder
    }

    fn read_bytes(path: &Path) -> Vec<u8> {
        fs::read(path).unwrap()
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
    }

    fn hex(digest: Digest) -> String {
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn repair_outer_integrity(bytes: &mut [u8]) {
        let payload_offset = get_u64(bytes, 104) as usize;
        let directory_digest = sha256(&bytes[SUPERBLOCK_BYTES..payload_offset]);
        bytes[DIRECTORY_DIGEST_OFFSET..DIRECTORY_DIGEST_OFFSET + DIGEST_BYTES]
            .copy_from_slice(&directory_digest);

        let mut pack = Sha256::new();
        pack.update(b"DFSP-PACK-V1\0");
        pack.update(&get_u32(bytes, 24).to_le_bytes());
        pack.update(&bytes[192..224]);
        pack.update(&bytes[224..256]);
        pack.update(&bytes[256..288]);
        pack.update(&directory_digest);
        pack.update(&bytes[PAYLOAD_DIGEST_OFFSET..PAYLOAD_DIGEST_OFFSET + DIGEST_BYTES]);
        let pack_digest = pack.finalize();
        bytes[PACK_DIGEST_OFFSET..PACK_DIGEST_OFFSET + DIGEST_BYTES].copy_from_slice(&pack_digest);
        let header_digest = sha256_with_zeroed_range(
            &bytes[..SUPERBLOCK_BYTES],
            HEADER_DIGEST_OFFSET,
            DIGEST_BYTES,
        );
        bytes[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + DIGEST_BYTES]
            .copy_from_slice(&header_digest);
    }

    fn repair_payload_and_outer_integrity(bytes: &mut [u8]) {
        let payload_offset = get_u64(bytes, 104) as usize;
        let chunk_count = get_u32(bytes, 44) as usize;
        let chunks_offset = get_u64(bytes, 88) as usize;
        for index in 0..chunk_count {
            let record = chunks_offset + index * CHUNK_RECORD_BYTES;
            let offset = get_u64(bytes, record) as usize;
            let length = get_u32(bytes, record + 8) as usize;
            let digest = sha256(&bytes[payload_offset + offset..payload_offset + offset + length]);
            bytes[record + 16..record + 48].copy_from_slice(&digest);
        }
        let payload_digest = sha256(&bytes[payload_offset..]);
        bytes[PAYLOAD_DIGEST_OFFSET..PAYLOAD_DIGEST_OFFSET + DIGEST_BYTES]
            .copy_from_slice(&payload_digest);
        repair_outer_integrity(bytes);
    }

    #[test]
    fn sha256_matches_standard_vectors_and_incremental_updates() {
        assert_eq!(
            hex(sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut incremental = Sha256::new();
        incremental.update(b"a");
        incremental.update(b"b");
        incremental.update(b"c");
        assert_eq!(incremental.finalize(), sha256(b"abc"));
        assert_eq!(
            hex(sha256(&vec![b'a'; 1_000_000])),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn round_trip_is_lossless_and_independent_of_input_order() {
        let directory = TestDirectory::new("roundtrip");
        let first = directory.path("first.dfsp");
        let second = directory.path("second.dfsp");
        let pack = sample_builder(false).write_atomic(&first).unwrap();
        sample_builder(true).write_atomic(&second).unwrap();

        assert_eq!(read_bytes(&first), read_bytes(&second));
        assert_eq!(pack.tensors().len(), 2);
        assert_eq!(pack.header().payload_offset % PAYLOAD_ALIGNMENT, 0);
        assert_eq!(pack.header().file_bytes % PAYLOAD_ALIGNMENT, 0);
        pack.verify_all().unwrap();

        let q = pack
            .tensors()
            .iter()
            .find(|tensor| tensor.codec == Codec::RowI8F16Scale)
            .unwrap();
        let file = read_bytes(&first);
        let payload = pack.header().payload_offset as usize;
        assert_eq!(
            &file[payload + q.data_offset as usize
                ..payload + (q.data_offset + q.data_length) as usize],
            &[0x80, 0xff, 0, 1, 2, 0x7f]
        );
        assert_eq!(
            &file[payload + q.auxiliary_offset as usize
                ..payload + (q.auxiliary_offset + q.auxiliary_length) as usize],
            &[0x00, 0x3c, 0x00, 0x40]
        );
        assert_eq!(pack.upload_descriptors().len(), 2);
        assert_eq!(pack.read_extents().len(), 3);
        pack.verify_chunk(0).unwrap();
        let first_extent = pack.read_extents()[0];
        let start = first_extent.file_offset as usize;
        let end = start + first_extent.length as usize;
        pack.verify_chunk_data(0, &file[start..end]).unwrap();
        assert!(pack.verify_chunk_data(0, &file[start..end - 1]).is_err());
    }

    #[test]
    fn file_ranges_are_copied_verbatim() {
        let directory = TestDirectory::new("file-range");
        let source = directory.path("source.bin");
        fs::write(&source, [9, 8, 1, 2, 3, 4, 7]).unwrap();
        let mut builder = PackBuilder::new(2, identity()).unwrap();
        builder.set_chunk_bytes(64 * 1024).unwrap();
        builder.push(BuildTensor::raw(
            tensor_name(2, "input_layernorm.weight"),
            DType::F32,
            vec![1],
            ComponentSource::file(&source, 2, 4),
            0,
            0,
        ));
        let destination = directory.path("layer.dfsp");
        let pack = builder.write_atomic(&destination).unwrap();
        pack.verify_all().unwrap();
        let bytes = read_bytes(&destination);
        let record = &pack.tensors()[0];
        let start = (pack.header().payload_offset + record.data_offset) as usize;
        assert_eq!(&bytes[start..start + 4], &[1, 2, 3, 4]);
    }

    #[test]
    fn component_hashing_remains_exact_across_chunk_boundaries() {
        let directory = TestDirectory::new("cross-chunk");
        let destination = directory.path("layer.dfsp");
        let bytes: Vec<u8> = (0..70_000).map(|index| (index % 251) as u8).collect();
        let mut builder = PackBuilder::new(3, identity()).unwrap();
        builder.set_chunk_bytes(64 * 1024).unwrap();
        builder.push(BuildTensor::raw(
            tensor_name(3, "input_layernorm.weight"),
            DType::U8,
            vec![bytes.len() as u64],
            ComponentSource::bytes(bytes),
            0,
            0,
        ));

        let pack = builder.write_atomic(&destination).unwrap();
        assert_eq!(pack.chunks().len(), 2);
        pack.verify_all().unwrap();
    }

    #[test]
    fn builder_rejects_wrong_lengths_duplicate_names_and_wrong_layer() {
        let mut wrong_length = PackBuilder::new(1, identity()).unwrap();
        wrong_length.push(BuildTensor::row_i8_f16_scale(
            tensor_name(1, "self_attn.q_proj.weight"),
            DType::Bf16,
            [2, 2],
            ComponentSource::bytes(vec![1, 2, 3]),
            ComponentSource::bytes(vec![0; 4]),
            0,
            0,
        ));
        assert!(plan_build(&wrong_length, ParseLimits::default()).is_err());

        let raw = BuildTensor::raw(
            tensor_name(1, "input_layernorm.weight"),
            DType::Bf16,
            vec![1],
            ComponentSource::bytes(vec![0; 2]),
            0,
            0,
        );
        let mut duplicate = PackBuilder::new(1, identity()).unwrap();
        duplicate.push(raw.clone());
        duplicate.push(raw);
        assert!(
            plan_build(&duplicate, ParseLimits::default())
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );

        let mut wrong_layer = PackBuilder::new(1, identity()).unwrap();
        wrong_layer.push(BuildTensor::raw(
            tensor_name(2, "input_layernorm.weight"),
            DType::Bf16,
            vec![1],
            ComponentSource::bytes(vec![0; 2]),
            0,
            0,
        ));
        assert!(
            plan_build(&wrong_layer, ParseLimits::default())
                .unwrap_err()
                .to_string()
                .contains("does not belong")
        );
    }

    #[test]
    fn admission_rejects_header_directory_payload_and_length_corruption() {
        let directory = TestDirectory::new("corruption");
        let original = directory.path("original.dfsp");
        let pack = sample_builder(false).write_atomic(&original).unwrap();
        let original_bytes = read_bytes(&original);

        let header_path = directory.path("header.dfsp");
        let mut bytes = original_bytes.clone();
        bytes[20] ^= 1;
        write_bytes(&header_path, &bytes);
        assert!(PackFile::open(&header_path).is_err());

        let directory_path = directory.path("directory.dfsp");
        let mut bytes = original_bytes.clone();
        bytes[SUPERBLOCK_BYTES + 10] ^= 1;
        write_bytes(&directory_path, &bytes);
        assert!(
            PackFile::open(&directory_path)
                .unwrap_err()
                .to_string()
                .contains("directory")
        );

        let payload_path = directory.path("payload.dfsp");
        let mut bytes = original_bytes.clone();
        let data = (pack.header().payload_offset + pack.tensors()[0].data_offset) as usize;
        bytes[data] ^= 1;
        write_bytes(&payload_path, &bytes);
        let corrupted = PackFile::open(&payload_path).unwrap();
        assert!(
            corrupted
                .verify_all()
                .unwrap_err()
                .to_string()
                .contains("chunk")
        );

        let truncated_path = directory.path("truncated.dfsp");
        write_bytes(&truncated_path, &original_bytes[..original_bytes.len() - 1]);
        assert!(PackFile::open(&truncated_path).is_err());
    }

    #[test]
    fn identity_and_existing_destination_fail_closed() {
        let directory = TestDirectory::new("identity");
        let destination = directory.path("layer.dfsp");
        let first = sample_builder(false).write_atomic(&destination).unwrap();
        let wrong_identity = PackIdentity::new([9; 32], [2; 32], [3; 32]);
        assert!(first.require_identity(7, wrong_identity).is_err());
        assert!(first.require_identity(8, identity()).is_err());
        let before = read_bytes(&destination);
        assert!(sample_builder(false).write_atomic(&destination).is_err());
        assert_eq!(before, read_bytes(&destination));
    }

    #[test]
    fn concurrent_publishers_never_replace_the_winner() {
        let directory = TestDirectory::new("publish-race");
        let destination = directory.path("layer.dfsp");
        let first_destination = destination.clone();
        let second_destination = destination.clone();

        let first =
            std::thread::spawn(move || sample_builder(false).write_atomic(first_destination));
        let second =
            std::thread::spawn(move || sample_builder(true).write_atomic(second_destination));
        let results = [first.join().unwrap(), second.join().unwrap()];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);

        let admitted = PackFile::open_for(&destination, 7, identity()).unwrap();
        admitted.verify_all().unwrap();
    }

    #[test]
    fn rehashed_structural_malformations_still_fail_closed() {
        let directory = TestDirectory::new("structural");
        let original = directory.path("original.dfsp");
        sample_builder(false).write_atomic(&original).unwrap();
        let original_bytes = read_bytes(&original);
        let records_offset = get_u64(&original_bytes, 56) as usize;
        let record_count = get_u32(&original_bytes, 32) as usize;
        let q_record = (0..record_count)
            .find(|index| original_bytes[records_offset + index * RECORD_BYTES + 38] == 1)
            .unwrap();

        let bad_component = directory.path("bad-component.dfsp");
        let mut bytes = original_bytes.clone();
        let scales_offset = get_u64(&bytes, 144);
        put_u64(
            &mut bytes,
            records_offset + q_record * RECORD_BYTES + 120,
            scales_offset,
        );
        repair_outer_integrity(&mut bytes);
        write_bytes(&bad_component, &bytes);
        assert!(
            PackFile::open(&bad_component)
                .unwrap_err()
                .to_string()
                .contains("quantized component starts")
        );

        let bad_chunks = directory.path("bad-chunks.dfsp");
        let mut bytes = original_bytes;
        let chunks_offset = get_u64(&bytes, 88) as usize;
        put_u64(&mut bytes, chunks_offset, COMPONENT_ALIGNMENT);
        repair_outer_integrity(&mut bytes);
        write_bytes(&bad_chunks, &bytes);
        assert!(
            PackFile::open(&bad_chunks)
                .unwrap_err()
                .to_string()
                .contains("chunk 0 starts")
        );
    }

    #[test]
    fn zero_padding_is_required_even_when_all_outer_hashes_are_recomputed() {
        let directory = TestDirectory::new("padding");
        let original = directory.path("original.dfsp");
        let pack = sample_builder(false).write_atomic(&original).unwrap();
        let q = pack
            .tensors()
            .iter()
            .find(|tensor| tensor.codec == Codec::RowI8F16Scale)
            .unwrap();
        let mut bytes = read_bytes(&original);
        let padding_byte =
            (pack.header().payload_offset + q.data_offset + q.data_length + 1) as usize;
        bytes[padding_byte] = 0xa5;
        repair_payload_and_outer_integrity(&mut bytes);
        let malformed = directory.path("nonzero-padding.dfsp");
        write_bytes(&malformed, &bytes);
        let admitted = PackFile::open(&malformed).unwrap();
        assert!(
            admitted
                .verify_all()
                .unwrap_err()
                .to_string()
                .contains("padding")
        );
    }

    #[test]
    fn bounded_parser_rejects_excessive_claims_before_allocating() {
        let directory = TestDirectory::new("limits");
        let destination = directory.path("layer.dfsp");
        sample_builder(false).write_atomic(&destination).unwrap();
        let tiny = ParseLimits {
            max_file_bytes: 100,
            ..ParseLimits::default()
        };
        assert!(PackFile::open_with_limits(&destination, tiny).is_err());
    }

    #[test]
    fn rehashed_integer_overflow_claims_fail_before_directory_allocation() {
        let directory = TestDirectory::new("overflow");
        let original = directory.path("original.dfsp");
        sample_builder(false).write_atomic(&original).unwrap();
        let mut bytes = read_bytes(&original);

        // Make `strings_offset + strings_bytes` overflow while keeping the
        // superblock hash valid. Admission must return an error rather than
        // panic or derive a wrapped allocation length.
        put_u64(&mut bytes, 80, u64::MAX);
        let header_digest = sha256_with_zeroed_range(
            &bytes[..SUPERBLOCK_BYTES],
            HEADER_DIGEST_OFFSET,
            DIGEST_BYTES,
        );
        bytes[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + DIGEST_BYTES]
            .copy_from_slice(&header_digest);

        let malformed = directory.path("overflow.dfsp");
        write_bytes(&malformed, &bytes);
        assert!(
            PackFile::open_with_limits(
                &malformed,
                ParseLimits {
                    max_metadata_bytes: u64::MAX,
                    ..ParseLimits::default()
                }
            )
            .unwrap_err()
            .to_string()
            .contains("overflows")
        );
    }

    #[test]
    fn short_component_file_is_rejected_without_publishing() {
        let directory = TestDirectory::new("short-source");
        let source = directory.path("source.bin");
        fs::write(&source, [1, 2]).unwrap();
        let destination = directory.path("layer.dfsp");
        let mut builder = PackBuilder::new(0, identity()).unwrap();
        builder.push(BuildTensor::raw(
            tensor_name(0, "input_layernorm.weight"),
            DType::F32,
            vec![1],
            ComponentSource::file(&source, 0, 4),
            0,
            0,
        ));
        assert!(builder.write_atomic(&destination).is_err());
        assert!(!destination.exists());
    }

    #[test]
    fn bare_relative_destination_uses_current_directory() {
        assert_eq!(destination_parent(Path::new("layer.dfsp")), Path::new("."));
        assert_eq!(
            destination_parent(Path::new("packs/layer.dfsp")),
            Path::new("packs")
        );
    }
}
