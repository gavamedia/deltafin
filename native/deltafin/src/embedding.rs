//! Exact, Python-free row reads from K3's canonical BF16 token embedding.
//!
//! The authoritative embedding remains the unquantized 2.35 GB table used by
//! the Python `LazyEmbed` path.  This module owns one hardened descriptor and
//! reads only requested rows into a caller-owned reusable arena.  It performs
//! no dequantization, floating-point conversion, caching, or device transfer;
//! the returned bytes are the exact on-disk BF16 bit patterns.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::ptr;

use crate::error::{DeltafinError, Result};
use crate::sys::fs::{self as sys_fs, FileExt, Open};

pub const K3_EMBEDDING_RELATIVE_PATH: &str =
    "k3-resident/tensors/language_model.model.embed_tokens.weight";
pub const K3_EMBEDDING_ROWS: u32 = 163_840;
pub const K3_EMBEDDING_COLUMNS: u32 = 7_168;
pub const BF16_BYTES: usize = 2;
pub const K3_EMBEDDING_ROW_BYTES: usize = K3_EMBEDDING_COLUMNS as usize * BF16_BYTES;
pub const K3_EMBEDDING_TABLE_BYTES: u64 = K3_EMBEDDING_ROWS as u64 * K3_EMBEDDING_ROW_BYTES as u64;

/// Resolve the exact loose BF16 table used by the authoritative `LazyEmbed`.
pub fn k3_embedding_path(model_root: &Path) -> PathBuf {
    model_root.join(K3_EMBEDDING_RELATIVE_PATH)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct EmbeddingSpec {
    rows: u32,
    columns: u32,
    row_bytes: usize,
    table_bytes: u64,
}

impl EmbeddingSpec {
    pub const K3: Self = Self {
        rows: K3_EMBEDDING_ROWS,
        columns: K3_EMBEDDING_COLUMNS,
        row_bytes: K3_EMBEDDING_ROW_BYTES,
        table_bytes: K3_EMBEDDING_TABLE_BYTES,
    };

    pub fn new(rows: u32, columns: u32) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(DeltafinError::new(
                "BF16 embedding dimensions must both be nonzero",
            ));
        }
        let row_bytes = (columns as usize)
            .checked_mul(BF16_BYTES)
            .ok_or_else(|| DeltafinError::new("BF16 embedding row size overflow"))?;
        let table_bytes = (rows as u64)
            .checked_mul(row_bytes as u64)
            .ok_or_else(|| DeltafinError::new("BF16 embedding table size overflow"))?;
        Ok(Self {
            rows,
            columns,
            row_bytes,
            table_bytes,
        })
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.columns
    }

    pub const fn row_bytes(self) -> usize {
        self.row_bytes
    }

    pub const fn table_bytes(self) -> u64 {
        self.table_bytes
    }
}

/// A persistent descriptor for one exact raw-BF16 embedding table.
#[derive(Debug)]
pub struct EmbeddingTable {
    file: File,
    path: PathBuf,
    spec: EmbeddingSpec,
}

impl EmbeddingTable {
    pub fn open_k3(model_root: &Path) -> Result<Self> {
        Self::open_exact(k3_embedding_path(model_root), EmbeddingSpec::K3)
    }

    pub fn open_exact(path: impl AsRef<Path>, spec: EmbeddingSpec) -> Result<Self> {
        let path = path.as_ref();
        let file = Open::new()
            .read(true)
            .no_follow()
            .open(path)
            .map_err(|error| {
                io_error(
                    "open BF16 embedding without following symlinks",
                    path,
                    error,
                )
            })?;
        let table = Self {
            file,
            path: path.to_owned(),
            spec,
        };
        table.validate_open_descriptor()?;
        Ok(table)
    }

    pub const fn spec(&self) -> EmbeddingSpec {
        self.spec
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read rows into `arena` and return a borrow valid until that arena is
    /// reused.  The output is row-major BF16 in the caller's original order.
    pub fn read_rows<'arena>(
        &self,
        token_ids: &[u32],
        arena: &'arena mut EmbeddingArena,
    ) -> Result<EmbeddingBatch<'arena>> {
        arena.require_rows(token_ids.len())?;

        if token_ids.is_empty() {
            arena.clear_lengths();
            return Ok(EmbeddingBatch::new(
                &arena.output,
                0,
                self.spec.columns,
                0,
                0,
            ));
        }

        for &token_id in token_ids {
            self.checked_row_offset(token_id)?;
        }

        if token_ids.len() == 1 {
            let output_bytes = self.spec.row_bytes;
            resize_bytes(&mut arena.output, output_bytes, "embedding output arena")?;
            arena.indexed.clear();
            arena.unique_ids.clear();
            arena.order_to_unique.clear();
            arena.unique_bytes.clear();
            let offset = self.checked_row_offset(token_ids[0])?;
            read_exact_at(&self.file, &self.path, &mut arena.output, offset)?;
            return Ok(EmbeddingBatch::new(
                &arena.output,
                1,
                self.spec.columns,
                1,
                1,
            ));
        }

        prepare_scratch(arena, token_ids)?;
        let unique_bytes = arena
            .unique_ids
            .len()
            .checked_mul(self.spec.row_bytes)
            .ok_or_else(|| DeltafinError::new("unique embedding byte size overflow"))?;
        resize_bytes(
            &mut arena.unique_bytes,
            unique_bytes,
            "unique embedding row arena",
        )?;

        let mut physical_reads = 0_usize;
        let mut first_unique = 0_usize;
        while first_unique < arena.unique_ids.len() {
            let mut end_unique = first_unique + 1;
            while end_unique < arena.unique_ids.len()
                && arena.unique_ids[end_unique]
                    == arena.unique_ids[end_unique - 1]
                        .checked_add(1)
                        .expect("validated token IDs cannot overflow u32")
            {
                end_unique += 1;
            }

            let destination_start = first_unique
                .checked_mul(self.spec.row_bytes)
                .ok_or_else(|| DeltafinError::new("embedding destination offset overflow"))?;
            let destination_end = end_unique
                .checked_mul(self.spec.row_bytes)
                .ok_or_else(|| DeltafinError::new("embedding destination extent overflow"))?;
            let source_offset = self.checked_row_offset(arena.unique_ids[first_unique])?;
            read_exact_at(
                &self.file,
                &self.path,
                &mut arena.unique_bytes[destination_start..destination_end],
                source_offset,
            )?;
            physical_reads += 1;
            first_unique = end_unique;
        }

        let output_bytes = token_ids
            .len()
            .checked_mul(self.spec.row_bytes)
            .ok_or_else(|| DeltafinError::new("embedding output byte size overflow"))?;
        resize_bytes(&mut arena.output, output_bytes, "embedding output arena")?;
        for (output_row, &unique_row) in arena.order_to_unique.iter().enumerate() {
            let source_start = unique_row
                .checked_mul(self.spec.row_bytes)
                .ok_or_else(|| DeltafinError::new("embedding source offset overflow"))?;
            let destination_start = output_row
                .checked_mul(self.spec.row_bytes)
                .ok_or_else(|| DeltafinError::new("embedding output offset overflow"))?;
            arena.output[destination_start..destination_start + self.spec.row_bytes]
                .copy_from_slice(
                    &arena.unique_bytes[source_start..source_start + self.spec.row_bytes],
                );
        }

        Ok(EmbeddingBatch::new(
            &arena.output,
            token_ids.len(),
            self.spec.columns,
            arena.unique_ids.len(),
            physical_reads,
        ))
    }

    fn validate_open_descriptor(&self) -> Result<()> {
        // `File::metadata` is `fstat(2)` on the already-open descriptor.  This
        // validates the object actually read, not a racy path lookup.
        let metadata = sys_fs::fstat(&self.file)
            .map_err(|error| io_error("stat opened BF16 embedding", &self.path, error))?;
        if !metadata.is_file() {
            return Err(DeltafinError::new(format!(
                "BF16 embedding is not a regular file: {}",
                self.path.display(),
            )));
        }
        if metadata.len() != self.spec.table_bytes {
            return Err(DeltafinError::new(format!(
                "BF16 embedding {} is {} bytes; expected exact length {}",
                self.path.display(),
                metadata.len(),
                self.spec.table_bytes,
            )));
        }
        Ok(())
    }

    fn checked_row_offset(&self, token_id: u32) -> Result<u64> {
        if token_id >= self.spec.rows {
            return Err(DeltafinError::new(format!(
                "token ID {token_id} is outside the embedding table (0..{})",
                self.spec.rows,
            )));
        }
        (token_id as u64)
            .checked_mul(self.spec.row_bytes as u64)
            .ok_or_else(|| DeltafinError::new("embedding source offset overflow"))
    }

    #[cfg(all(test, unix))]
    fn descriptor(&self) -> i32 {
        use std::os::fd::AsRawFd;
        self.file.as_raw_fd()
    }
}

#[derive(Debug, Clone, Copy)]
struct IndexedRow {
    token_id: u32,
    original_index: usize,
}

/// Reusable caller-owned storage for embedding requests.
///
/// `max_rows` is an explicit memory bound. Vectors grow only to the largest
/// admitted request and then retain that allocation for later tokens.
#[derive(Debug)]
pub struct EmbeddingArena {
    max_rows: usize,
    output: Vec<u8>,
    unique_bytes: Vec<u8>,
    indexed: Vec<IndexedRow>,
    unique_ids: Vec<u32>,
    order_to_unique: Vec<usize>,
}

impl EmbeddingArena {
    pub fn new(max_rows: usize) -> Result<Self> {
        if max_rows > u32::MAX as usize {
            return Err(DeltafinError::new(
                "embedding arena row bound exceeds the provider ABI's u32 row count",
            ));
        }
        Ok(Self {
            max_rows,
            output: Vec::new(),
            unique_bytes: Vec::new(),
            indexed: Vec::new(),
            unique_ids: Vec::new(),
            order_to_unique: Vec::new(),
        })
    }

    pub const fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn allocated_bytes(&self) -> usize {
        self.output
            .capacity()
            .saturating_add(self.unique_bytes.capacity())
            .saturating_add(
                self.indexed
                    .capacity()
                    .saturating_mul(std::mem::size_of::<IndexedRow>()),
            )
            .saturating_add(
                self.unique_ids
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(
                self.order_to_unique
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
    }

    fn require_rows(&self, rows: usize) -> Result<()> {
        if rows > self.max_rows {
            return Err(DeltafinError::new(format!(
                "embedding request has {rows} rows; arena limit is {}",
                self.max_rows,
            )));
        }
        Ok(())
    }

    fn clear_lengths(&mut self) {
        self.output.clear();
        self.unique_bytes.clear();
        self.indexed.clear();
        self.unique_ids.clear();
        self.order_to_unique.clear();
    }
}

/// Borrowed exact-BF16 result in original token order.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingBatch<'arena> {
    bytes: &'arena [u8],
    rows: usize,
    columns: u32,
    unique_rows: usize,
    physical_reads: usize,
}

impl<'arena> EmbeddingBatch<'arena> {
    fn new(
        bytes: &'arena [u8],
        rows: usize,
        columns: u32,
        unique_rows: usize,
        physical_reads: usize,
    ) -> Self {
        Self {
            bytes,
            rows,
            columns,
            unique_rows,
            physical_reads,
        }
    }

    pub const fn bytes(self) -> &'arena [u8] {
        self.bytes
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.columns
    }

    pub const fn unique_rows(self) -> usize {
        self.unique_rows
    }

    pub const fn physical_reads(self) -> usize {
        self.physical_reads
    }

    /// Pointer/shape view for a future coarse provider call. The provider may
    /// borrow this memory only while both this batch and its arena stay live.
    pub fn provider_view(self) -> Bf16EmbeddingBatchViewV1 {
        Bf16EmbeddingBatchViewV1 {
            data: if self.bytes.is_empty() {
                ptr::null()
            } else {
                self.bytes.as_ptr()
            },
            byte_length: self.bytes.len() as u64,
            rows: self.rows as u32,
            columns: self.columns,
            row_stride_bytes: self.columns as u64 * BF16_BYTES as u64,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Bf16EmbeddingBatchViewV1 {
    pub data: *const u8,
    pub byte_length: u64,
    pub rows: u32,
    pub columns: u32,
    pub row_stride_bytes: u64,
}

fn prepare_scratch(arena: &mut EmbeddingArena, token_ids: &[u32]) -> Result<()> {
    reserve_items(
        &mut arena.indexed,
        token_ids.len(),
        "embedding sort scratch",
    )?;
    arena.indexed.clear();
    arena
        .indexed
        .extend(
            token_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(original_index, token_id)| IndexedRow {
                    token_id,
                    original_index,
                }),
        );
    arena
        .indexed
        .sort_unstable_by_key(|row| (row.token_id, row.original_index));

    reserve_items(
        &mut arena.unique_ids,
        token_ids.len(),
        "unique embedding ID scratch",
    )?;
    arena.unique_ids.clear();
    reserve_items(
        &mut arena.order_to_unique,
        token_ids.len(),
        "embedding order scratch",
    )?;
    arena.order_to_unique.clear();
    arena.order_to_unique.resize(token_ids.len(), 0);

    let mut previous = None;
    let mut unique_index = 0_usize;
    for row in &arena.indexed {
        if previous != Some(row.token_id) {
            arena.unique_ids.push(row.token_id);
            unique_index = arena.unique_ids.len() - 1;
            previous = Some(row.token_id);
        }
        arena.order_to_unique[row.original_index] = unique_index;
    }
    Ok(())
}

fn reserve_items<T>(values: &mut Vec<T>, wanted: usize, description: &str) -> Result<()> {
    if wanted > values.capacity() {
        values
            .try_reserve_exact(wanted - values.len())
            .map_err(|error| DeltafinError::new(format!("allocate {description}: {error}")))?;
    }
    Ok(())
}

fn resize_bytes(values: &mut Vec<u8>, wanted: usize, description: &str) -> Result<()> {
    reserve_items(values, wanted, description)?;
    values.resize(wanted, 0);
    Ok(())
}

fn read_exact_at(file: &File, path: &Path, destination: &mut [u8], offset: u64) -> Result<()> {
    let mut completed = 0_usize;
    while completed < destination.len() {
        let completed_u64 = u64::try_from(completed)
            .map_err(|_| DeltafinError::new("embedding read offset conversion overflow"))?;
        let position = offset
            .checked_add(completed_u64)
            .ok_or_else(|| DeltafinError::new("embedding read offset overflow"))?;
        match file.read_at(&mut destination[completed..], position) {
            Ok(0) => {
                return Err(DeltafinError::new(format!(
                    "short BF16 embedding read {completed}/{} at byte {offset}: {}",
                    destination.len(),
                    path.display(),
                )));
            }
            Ok(count) => completed += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(io_error("pread BF16 embedding", path, error)),
        }
    }
    Ok(())
}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use crate::sys::fs::symlink;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let serial = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-embedding-test-{}-{serial}",
                std::process::id(),
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn table_bytes(rows: u32, columns: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows as usize * columns as usize * BF16_BYTES);
        for row in 0..rows {
            for column in 0..columns {
                // Include arbitrary BF16 bit patterns rather than round-trip
                // through float, including sign/exponent-heavy values.
                let bits = ((row as u16).wrapping_mul(0x2111))
                    ^ ((column as u16).wrapping_mul(0x8f03))
                    ^ 0x7f81;
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
        }
        bytes
    }

    fn write_file(path: &Path, bytes: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
    }

    fn expected_rows(source: &[u8], row_bytes: usize, rows: &[u32]) -> Vec<u8> {
        let mut expected = Vec::with_capacity(rows.len() * row_bytes);
        for &row in rows {
            let start = row as usize * row_bytes;
            expected.extend_from_slice(&source[start..start + row_bytes]);
        }
        expected
    }

    #[test]
    fn k3_contract_matches_authoritative_lazy_embed_file() {
        assert_eq!(K3_EMBEDDING_ROW_BYTES, 14_336);
        assert_eq!(K3_EMBEDDING_TABLE_BYTES, 2_348_810_240);
        assert_eq!(EmbeddingSpec::K3.rows(), 163_840);
        assert_eq!(EmbeddingSpec::K3.columns(), 7_168);
        assert_eq!(
            k3_embedding_path(Path::new("/model")),
            Path::new("/model").join(K3_EMBEDDING_RELATIVE_PATH),
        );
    }

    #[test]
    fn batch_is_raw_bf16_exact_and_restores_duplicate_out_of_order_rows() {
        let directory = TestDirectory::new();
        let path = directory.path("embedding.bf16");
        let spec = EmbeddingSpec::new(7, 5).unwrap();
        let source = table_bytes(spec.rows(), spec.columns());
        write_file(&path, &source);
        let table = EmbeddingTable::open_exact(&path, spec).unwrap();
        let mut arena = EmbeddingArena::new(16).unwrap();
        let requested = [5, 2, 3, 0, 5, 2];

        let batch = table.read_rows(&requested, &mut arena).unwrap();
        assert_eq!(
            batch.bytes(),
            expected_rows(&source, spec.row_bytes(), &requested),
        );
        assert_eq!(batch.rows(), requested.len());
        assert_eq!(batch.columns(), spec.columns());
        assert_eq!(batch.unique_rows(), 4);
        assert_eq!(batch.physical_reads(), 3); // 0, 2..3, 5
        let view = batch.provider_view();
        assert_eq!(view.data, batch.bytes().as_ptr());
        assert_eq!(view.byte_length, batch.bytes().len() as u64);
        assert_eq!(view.rows, requested.len() as u32);
        assert_eq!(view.columns, spec.columns());
        assert_eq!(view.row_stride_bytes, spec.row_bytes() as u64);
    }

    #[test]
    fn t1_reads_directly_into_the_reusable_output_arena() {
        let directory = TestDirectory::new();
        let path = directory.path("embedding.bf16");
        let spec = EmbeddingSpec::new(4, 4).unwrap();
        let source = table_bytes(spec.rows(), spec.columns());
        write_file(&path, &source);
        let table = EmbeddingTable::open_exact(&path, spec).unwrap();
        let mut arena = EmbeddingArena::new(4).unwrap();

        let batch = table.read_rows(&[3], &mut arena).unwrap();
        assert_eq!(
            batch.bytes(),
            expected_rows(&source, spec.row_bytes(), &[3]),
        );
        assert_eq!(batch.unique_rows(), 1);
        assert_eq!(batch.physical_reads(), 1);
        assert!(arena.unique_bytes.is_empty());
        assert!(arena.indexed.is_empty());
    }

    #[test]
    fn arena_reuses_its_high_water_allocations_and_enforces_its_bound() {
        let directory = TestDirectory::new();
        let path = directory.path("embedding.bf16");
        let spec = EmbeddingSpec::new(8, 8).unwrap();
        write_file(&path, &table_bytes(spec.rows(), spec.columns()));
        let table = EmbeddingTable::open_exact(&path, spec).unwrap();
        let mut arena = EmbeddingArena::new(6).unwrap();

        let _ = table.read_rows(&[5, 2, 3, 0, 5, 2], &mut arena).unwrap();
        let allocation = arena.allocated_bytes();
        let output_pointer = arena.output.as_ptr();
        let unique_pointer = arena.unique_bytes.as_ptr();
        let _ = table.read_rows(&[3, 2, 3, 2], &mut arena).unwrap();
        assert_eq!(arena.allocated_bytes(), allocation);
        assert_eq!(arena.output.as_ptr(), output_pointer);
        assert_eq!(arena.unique_bytes.as_ptr(), unique_pointer);
        assert!(table.read_rows(&[0; 7], &mut arena).is_err());
    }

    #[test]
    fn rejects_short_oversized_and_detects_post_open_truncation_in_pread() {
        let directory = TestDirectory::new();
        let spec = EmbeddingSpec::new(4, 3).unwrap();
        let exact = table_bytes(spec.rows(), spec.columns());

        let short = directory.path("short.bf16");
        write_file(&short, &exact[..exact.len() - 1]);
        assert!(
            EmbeddingTable::open_exact(&short, spec)
                .unwrap_err()
                .to_string()
                .contains("expected exact length")
        );

        let oversized = directory.path("oversized.bf16");
        let mut extra = exact.clone();
        extra.push(0);
        write_file(&oversized, &extra);
        assert!(
            EmbeddingTable::open_exact(&oversized, spec)
                .unwrap_err()
                .to_string()
                .contains("expected exact length")
        );

        let live = directory.path("live.bf16");
        write_file(&live, &exact);
        let table = EmbeddingTable::open_exact(&live, spec).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&live)
            .unwrap()
            .set_len(exact.len() as u64 - 1)
            .unwrap();
        let mut arena = EmbeddingArena::new(1).unwrap();
        assert!(
            table
                .read_rows(&[spec.rows() - 1], &mut arena)
                .unwrap_err()
                .to_string()
                .contains("short BF16 embedding read")
        );
    }

    #[test]
    fn nofollow_rejects_a_symlink_table() {
        let directory = TestDirectory::new();
        let target = directory.path("target.bf16");
        let link = directory.path("link.bf16");
        let spec = EmbeddingSpec::new(2, 2).unwrap();
        write_file(&target, &table_bytes(spec.rows(), spec.columns()));
        symlink(&target, &link).unwrap();

        let error = EmbeddingTable::open_exact(&link, spec).unwrap_err();
        assert!(error.to_string().contains("without following symlinks"));
    }

    #[cfg(unix)]
    #[test]
    fn one_persistent_descriptor_is_reused_and_closed_without_a_read_leak() {
        unsafe extern "C" {
            fn fcntl(descriptor: i32, command: i32, ...) -> i32;
        }
        const F_GETFD: i32 = 1;

        let directory = TestDirectory::new();
        let path = directory.path("embedding.bf16");
        let spec = EmbeddingSpec::new(4, 4).unwrap();
        write_file(&path, &table_bytes(spec.rows(), spec.columns()));
        let table = EmbeddingTable::open_exact(&path, spec).unwrap();
        let descriptor = table.descriptor();
        let descriptor_flags = unsafe { fcntl(descriptor, F_GETFD) };
        assert!(descriptor_flags >= 0);
        assert_eq!(descriptor_flags & 1, 1, "descriptor must be CLOEXEC");

        let mut arena = EmbeddingArena::new(4).unwrap();
        for _ in 0..64 {
            let _ = table.read_rows(&[3, 1, 2, 1], &mut arena).unwrap();
            assert_eq!(table.descriptor(), descriptor);
        }
        drop(table);
        assert_eq!(unsafe { fcntl(descriptor, F_GETFD) }, -1);
    }

    #[test]
    fn checked_ids_empty_batches_and_arena_limits_fail_closed() {
        let directory = TestDirectory::new();
        let path = directory.path("embedding.bf16");
        let spec = EmbeddingSpec::new(3, 2).unwrap();
        write_file(&path, &table_bytes(spec.rows(), spec.columns()));
        let table = EmbeddingTable::open_exact(&path, spec).unwrap();
        let mut arena = EmbeddingArena::new(2).unwrap();

        let empty = table.read_rows(&[], &mut arena).unwrap();
        assert!(empty.bytes().is_empty());
        assert!(empty.provider_view().data.is_null());
        assert!(table.read_rows(&[3], &mut arena).is_err());
        assert!(table.read_rows(&[0, 1, 2], &mut arena).is_err());
    }
}
