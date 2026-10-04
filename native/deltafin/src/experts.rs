//! Fixed native schema and direct-read plans for K3's routed MXFP4 experts.
//!
//! Each legacy cache object is already a canonical 17,547,264-byte sequence of
//! six raw components. Rust reads that sequence once into one expert-major
//! span, which is the exact layout consumed by the established CPU, Metal, and
//! CUDA kernels. No Python `Future`, dictionary, NumPy view, per-component
//! allocation, or six-way restaging step is involved.

use std::fmt::{self, Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{DeltafinError, Result};
use crate::expert_scale4::manifest::{ManifestEntry, Scale4Manifest};
use crate::expert_scale4::{
    self, COMPACT_LAYOUT, FILE_BYTES as SCALE4_RECORD_BYTES, HEADER_BYTES as SCALE4_HEADER_BYTES,
    SIDECAR_LAYOUT,
};
use crate::inventory::K3Inventory;
use crate::storage::{
    BufferKind, BufferLengths, CachePolicy, DeferredExactCatalog, DeferredSourceIdentity,
    DeferredSourceLength, DeferredSourceName, Extent, LayerBuffers, ReadPlan, ReadPriority,
    ReadStats, ReadTicket, Reader, VectoredDestination,
};
use crate::weight_fetch::{ExpertFetchCatalog, FetchLimits};

pub const K3_MOE_LAYER_FIRST: u32 = 1;
pub const K3_MOE_LAYER_LAST: u32 = 92;
pub const K3_EXPERTS_PER_LAYER: usize = crate::routing::K3_EXPERTS;
pub const K3_EXPERT_TOP_K: usize = crate::routing::ROUTED_EXPERTS;
pub const K3_EXPERT_COMPONENTS: usize = 3;
pub const K3_EXPERT_PACKED_BYTES: usize = 5_505_024;
pub const K3_EXPERT_SCALE_BYTES: usize = 344_064;
pub const K3_EXPERT_SOURCE_BYTES: usize = 17_547_264;
pub const K3_SCALE4_BLOB_BYTES: usize = expert_scale4::BLOB_BYTES;
pub const K3_EXPERT_SOURCE_COMPONENTS: usize =
    (K3_MOE_LAYER_LAST as usize) * K3_EXPERTS_PER_LAYER * K3_EXPERT_COMPONENTS * 2;
pub const K3_EXPERT_RAW_FILES: usize = (K3_MOE_LAYER_LAST as usize) * K3_EXPERTS_PER_LAYER;
/// Structural expert-union capacity for one complete bounded verifier tile.
///
/// The current product verifier admits at most nine positions, so its actual
/// union cannot exceed 144 experts.  Keeping the storage contract aligned with
/// the provider's reviewed sixteen-row ceiling makes a later wider exact
/// verifier safe without changing this parser or silently dropping routes.
/// Reader slabs are still allocated for the *actual* canonical union only.
pub const K3_EXPERT_UNION_MAX: usize = 16 * K3_EXPERT_TOP_K;
/// Startup reserves the established four-row high-water.  Wider verifier
/// unions must pass a fresh live-memory admission before growing this arena.
pub const K3_EXPERT_BASE_UNION_MAX: usize = 64;
pub const K3_EXPERT_UNION_MAX_BYTES: usize = K3_EXPERT_UNION_MAX * K3_EXPERT_SOURCE_BYTES;
pub const K3_SCALE4_UNION_MAX_BYTES: usize = K3_EXPERT_UNION_MAX * K3_SCALE4_BLOB_BYTES;
pub const K3_EXPERT_BASE_UNION_BYTES: usize = K3_EXPERT_BASE_UNION_MAX * K3_EXPERT_SOURCE_BYTES;
pub const K3_SCALE4_BASE_UNION_BYTES: usize = K3_EXPERT_BASE_UNION_MAX * K3_SCALE4_BLOB_BYTES;
pub const DEFAULT_EXPERT_CHUNK_BYTES: usize = K3_EXPERT_SOURCE_BYTES;
pub const K3_EXPERT_BUFFER_KIND: BufferKind = BufferKind::Other;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertStorageLayout {
    RawV1,
    Scale4V2,
}

impl ExpertStorageLayout {
    pub const fn expert_span_bytes(self) -> usize {
        match self {
            Self::RawV1 => K3_EXPERT_SOURCE_BYTES,
            Self::Scale4V2 => K3_SCALE4_BLOB_BYTES,
        }
    }
}

impl Display for ExpertStorageLayout {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::RawV1 => "raw-v1",
            Self::Scale4V2 => "scale4-v2",
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ExpertMatrixDescriptorV1 {
    /// Byte offset from the batch's single expert-major raw buffer.
    pub packed_offset: u64,
    /// Byte offset from the same expert-major raw buffer.
    pub scale_offset: u64,
    pub rows: u32,
    pub columns: u32,
    pub packed_columns: u32,
    pub scale_columns: u32,
}

const MATRIX_SHAPES: [(u32, u32, u32, u32); K3_EXPERT_COMPONENTS] = [
    (3_072, 3_584, 1_792, 112),
    (3_584, 3_072, 1_536, 96),
    (3_072, 3_584, 1_792, 112),
];

const SOURCE_PACKED_OFFSETS: [u64; K3_EXPERT_COMPONENTS] = [
    0,
    (K3_EXPERT_PACKED_BYTES + K3_EXPERT_SCALE_BYTES) as u64,
    (2 * (K3_EXPERT_PACKED_BYTES + K3_EXPERT_SCALE_BYTES)) as u64,
];
const SOURCE_SCALE_OFFSETS: [u64; K3_EXPERT_COMPONENTS] = [
    K3_EXPERT_PACKED_BYTES as u64,
    (2 * K3_EXPERT_PACKED_BYTES + K3_EXPERT_SCALE_BYTES) as u64,
    (3 * K3_EXPERT_PACKED_BYTES + 2 * K3_EXPERT_SCALE_BYTES) as u64,
];

const EMPTY_MATRIX_DESCRIPTOR: ExpertMatrixDescriptorV1 = ExpertMatrixDescriptorV1 {
    packed_offset: 0,
    scale_offset: 0,
    rows: 0,
    columns: 0,
    packed_columns: 0,
    scale_columns: 0,
};

const fn decode_matrix_descriptors()
-> [[ExpertMatrixDescriptorV1; K3_EXPERT_COMPONENTS]; K3_EXPERT_TOP_K] {
    let mut descriptors = [[EMPTY_MATRIX_DESCRIPTOR; K3_EXPERT_COMPONENTS]; K3_EXPERT_TOP_K];
    let mut slot = 0;
    while slot < K3_EXPERT_TOP_K {
        let base = slot * K3_EXPERT_SOURCE_BYTES;
        let mut matrix = 0;
        while matrix < K3_EXPERT_COMPONENTS {
            let shape = MATRIX_SHAPES[matrix];
            descriptors[slot][matrix] = ExpertMatrixDescriptorV1 {
                packed_offset: (base + SOURCE_PACKED_OFFSETS[matrix] as usize) as u64,
                scale_offset: (base + SOURCE_SCALE_OFFSETS[matrix] as usize) as u64,
                rows: shape.0,
                columns: shape.1,
                packed_columns: shape.2,
                scale_columns: shape.3,
            };
            matrix += 1;
        }
        slot += 1;
    }
    descriptors
}

const fn decode_scale4_matrix_descriptors()
-> [[ExpertMatrixDescriptorV1; K3_EXPERT_COMPONENTS]; K3_EXPERT_TOP_K] {
    let mut descriptors = [[EMPTY_MATRIX_DESCRIPTOR; K3_EXPERT_COMPONENTS]; K3_EXPERT_TOP_K];
    let mut slot = 0;
    while slot < K3_EXPERT_TOP_K {
        let base = slot * K3_SCALE4_BLOB_BYTES;
        let mut matrix = 0;
        while matrix < K3_EXPERT_COMPONENTS {
            let shape = MATRIX_SHAPES[matrix];
            descriptors[slot][matrix] = ExpertMatrixDescriptorV1 {
                packed_offset: (base + COMPACT_LAYOUT[matrix * 2].0 as usize) as u64,
                scale_offset: (base + COMPACT_LAYOUT[matrix * 2 + 1].0 as usize) as u64,
                rows: shape.0,
                columns: shape.1,
                packed_columns: shape.2,
                scale_columns: shape.3,
            };
            matrix += 1;
        }
        slot += 1;
    }
    descriptors
}

/// The raw-v1 descriptor tape is invariant across every decode layer and
/// token. Keeping it in read-only storage removes 48 descriptor writes plus a
/// heap allocation from each of the 92 routed layers.
pub const K3_DECODE_MATRIX_DESCRIPTORS: [[ExpertMatrixDescriptorV1; K3_EXPERT_COMPONENTS];
    K3_EXPERT_TOP_K] = decode_matrix_descriptors();
pub const K3_SCALE4_DECODE_MATRIX_DESCRIPTORS: [[ExpertMatrixDescriptorV1; K3_EXPERT_COMPONENTS];
    K3_EXPERT_TOP_K] = decode_scale4_matrix_descriptors();

/// Session-owned raw-v1 expert namespace. All 82,432 safe relative names and
/// the `k3-experts` directory descriptor are compiled once at startup. Decode
/// submits only 16 integer source IDs to the persistent reader; wider sequence
/// unions retain the same exact-size and bounded-arena contracts.
#[derive(Debug)]
pub struct RawExpertCorpus {
    storage: ExpertCorpusStorage,
    model_root: PathBuf,
    lazy: Option<LazyExpertAdmission>,
    /// Page-cache treatment for every streaming expert read this corpus
    /// plans. The memory-tight reference host purges after each pass;
    /// large-RAM discrete-GPU hosts keep the kernel's file cache warm so
    /// repeated routed misses stop paying full disk latency.
    stream_cache_policy: CachePolicy,
}

#[derive(Debug)]
enum ExpertCorpusStorage {
    Raw(DeferredExactCatalog),
    Scale4 {
        manifest: Scale4Manifest,
        validation: Arc<Scale4ValidationCache>,
    },
}

const SCALE4_VALIDATION_WORDS: usize =
    (K3_EXPERTS_PER_LAYER + u64::BITS as usize - 1) / u64::BITS as usize;

#[derive(Debug, Clone, Copy)]
struct Scale4LayerValidation {
    corpus_identity: Option<DeferredSourceIdentity>,
    identity: Option<DeferredSourceIdentity>,
    validated: [u64; SCALE4_VALIDATION_WORDS],
}

impl Scale4LayerValidation {
    const EMPTY: Self = Self {
        corpus_identity: None,
        identity: None,
        validated: [0; SCALE4_VALIDATION_WORDS],
    };
}

#[derive(Debug)]
struct Scale4ValidationCache {
    layers: Mutex<[Scale4LayerValidation; K3_MOE_LAYER_LAST as usize + 1]>,
    #[cfg(test)]
    record_hashes: AtomicUsize,
}

impl Default for Scale4ValidationCache {
    fn default() -> Self {
        Self {
            layers: Mutex::new([Scale4LayerValidation::EMPTY; K3_MOE_LAYER_LAST as usize + 1]),
            #[cfg(test)]
            record_hashes: AtomicUsize::new(0),
        }
    }
}

impl Scale4ValidationCache {
    /// Return a pinned identity only after at least one selected record from
    /// that descriptor has authenticated successfully. Until then, concurrent
    /// plans keep doing their own live capture and must agree with the first
    /// identity pinned for this session.
    fn reusable_corpus_identity(&self, layer: u32) -> Option<DeferredSourceIdentity> {
        let layers = self.layers.lock().unwrap();
        let state = &layers[layer as usize];
        if state.corpus_identity == state.identity && state.validated.iter().any(|&word| word != 0)
        {
            state.corpus_identity
        } else {
            None
        }
    }

    fn admit_corpus_identity(&self, layer: u32, identity: DeferredSourceIdentity) -> Result<()> {
        let mut layers = self.layers.lock().unwrap();
        let state = &mut layers[layer as usize];
        match state.corpus_identity {
            None => {
                state.corpus_identity = Some(identity);
                Ok(())
            }
            Some(expected) if expected == identity => Ok(()),
            Some(_) => Err(DeltafinError::new(format!(
                "scale4 sidecar identity for layer {layer} changed after session pinning"
            ))),
        }
    }

    fn prepare(&self, layer: u32, identity: DeferredSourceIdentity) {
        let mut layers = self.layers.lock().unwrap();
        let state = &mut layers[layer as usize];
        if state.identity != Some(identity) {
            state.identity = Some(identity);
            state.validated = [0; SCALE4_VALIDATION_WORDS];
        }
    }

    fn needs_hash(&self, layer: u32, expert: u16, identity: DeferredSourceIdentity) -> bool {
        let layers = self.layers.lock().unwrap();
        let state = &layers[layer as usize];
        if state.identity != Some(identity) {
            return true;
        }
        let expert = expert as usize;
        state.validated[expert / u64::BITS as usize] & (1_u64 << (expert % u64::BITS as usize)) == 0
    }

    fn publish(
        &self,
        layer: u32,
        identity: DeferredSourceIdentity,
        entries: &[ManifestEntry],
    ) -> bool {
        let mut layers = self.layers.lock().unwrap();
        let state = &mut layers[layer as usize];
        if state.identity != Some(identity) {
            return false;
        }
        for entry in entries {
            let expert = entry.expert as usize;
            state.validated[expert / u64::BITS as usize] |= 1_u64 << (expert % u64::BITS as usize);
        }
        true
    }

    #[cfg(test)]
    fn record_hashes(&self) -> usize {
        self.record_hashes.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
struct Scale4BatchValidation {
    cache: Arc<Scale4ValidationCache>,
    layer: u32,
    identity: DeferredSourceIdentity,
    /// Records whose manifest digest was already checked inside the reader
    /// worker before this batch's identity and header validation boundary.
    reader_verified: Box<[u16]>,
}

#[derive(Debug)]
struct LazyExpertAdmission {
    catalog: ExpertFetchCatalog,
    remaining: AtomicUsize,
}

impl LazyExpertAdmission {
    fn ensure(&self, layer: u32, expert_ids: &[u16]) -> Result<()> {
        if self.remaining.load(Ordering::Acquire) == 0 {
            return Ok(());
        }
        let outcome = self
            .catalog
            .fetch_layer_detailed(layer, expert_ids, &|_| {})?;
        let admitted = outcome.planned.expert_files_missing;
        if admitted == 0 {
            return Ok(());
        }
        self.remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(admitted)
            })
            .map_err(|remaining| {
                DeltafinError::new(format!(
                    "lazy expert admission published {admitted} files with only {remaining} missing"
                ))
            })?;
        Ok(())
    }
}

impl RawExpertCorpus {
    pub fn open(model_root: &Path, layout: ExpertStorageLayout) -> Result<Self> {
        Self::open_with_cache_policy(model_root, layout, CachePolicy::Streaming)
    }

    pub fn open_with_cache_policy(
        model_root: &Path,
        layout: ExpertStorageLayout,
        stream_cache_policy: CachePolicy,
    ) -> Result<Self> {
        match layout {
            ExpertStorageLayout::RawV1 => {
                Self::open_raw_v1_with_cache_policy(model_root, stream_cache_policy)
            }
            ExpertStorageLayout::Scale4V2 => {
                Self::open_scale4_v2_with_cache_policy(model_root, stream_cache_policy)
            }
        }
    }

    /// Open the selected exact storage format and automatically attach the
    /// authenticated range-fetch catalog only when the local raw corpus is
    /// incomplete. A complete installation keeps the decode hot path free of
    /// duplicate metadata calls; a partial installation admits exactly the
    /// bounded routed union before its ordinary no-follow reader ticket.
    pub fn open_auto(
        model_root: &Path,
        layout: ExpertStorageLayout,
        stream_cache_policy: CachePolicy,
    ) -> Result<Self> {
        ensure_raw_cache_directory(model_root)?;
        let mut corpus = Self::open_with_cache_policy(model_root, layout, stream_cache_policy)?;
        let missing = raw_cache_missing_files(model_root)?;
        if missing != 0 {
            let inventory = K3Inventory::load_from_root(model_root)?;
            let catalog = ExpertFetchCatalog::open(model_root, &inventory, FetchLimits::default())?;
            corpus.lazy = Some(LazyExpertAdmission {
                catalog,
                remaining: AtomicUsize::new(missing),
            });
        }
        Ok(corpus)
    }

    pub fn open_raw_v1(model_root: &Path) -> Result<Self> {
        Self::open_raw_v1_with_cache_policy(model_root, CachePolicy::Streaming)
    }

    pub fn open_raw_v1_with_cache_policy(
        model_root: &Path,
        stream_cache_policy: CachePolicy,
    ) -> Result<Self> {
        let mut names = Vec::with_capacity(K3_EXPERT_RAW_FILES);
        for layer in K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST {
            for expert in 0..K3_EXPERTS_PER_LAYER as u16 {
                names.push(raw_expert_source_name(layer, expert)?);
            }
        }
        let catalog = DeferredExactCatalog::open(
            &model_root.join("k3-experts"),
            names,
            K3_EXPERT_SOURCE_BYTES as u64,
            stream_cache_policy,
        )?;
        debug_assert_eq!(catalog.source_count(), K3_EXPERT_RAW_FILES);
        Ok(Self {
            storage: ExpertCorpusStorage::Raw(catalog),
            model_root: model_root.to_path_buf(),
            lazy: None,
            stream_cache_policy,
        })
    }

    pub fn open_scale4_v2(model_root: &Path) -> Result<Self> {
        Self::open_scale4_v2_with_cache_policy(model_root, CachePolicy::Streaming)
    }

    pub fn open_scale4_v2_with_cache_policy(
        model_root: &Path,
        stream_cache_policy: CachePolicy,
    ) -> Result<Self> {
        let manifest = Scale4Manifest::load_full(model_root.join("k3-experts-scale4"))?;
        if manifest.entries().len() != K3_EXPERT_RAW_FILES {
            return Err(DeltafinError::new(
                "scale4 activation manifest does not cover the complete K3 expert corpus",
            ));
        }
        Ok(Self {
            storage: ExpertCorpusStorage::Scale4 {
                manifest,
                validation: Arc::new(Scale4ValidationCache::default()),
            },
            model_root: model_root.to_path_buf(),
            lazy: None,
            stream_cache_policy,
        })
    }

    pub fn submit_decode(
        &self,
        reader: &Reader,
        layer: u32,
        ascending_expert_ids: &[u16],
    ) -> Result<ExpertReadTicket> {
        self.submit_decode_with_priority(reader, layer, ascending_expert_ids, ReadPriority::Demand)
    }

    pub fn submit_decode_with_priority(
        &self,
        reader: &Reader,
        layer: u32,
        ascending_expert_ids: &[u16],
        priority: ReadPriority,
    ) -> Result<ExpertReadTicket> {
        let (expert_ids, source_indices) = decode_source_indices(layer, ascending_expert_ids)?;
        self.ensure_available(layer, &expert_ids)?;
        match &self.storage {
            ExpertCorpusStorage::Raw(catalog) => {
                let ticket = reader.submit_deferred_exact(
                    catalog,
                    &source_indices,
                    K3_EXPERT_BUFFER_KIND,
                    priority,
                )?;
                Ok(ExpertReadTicket {
                    layer,
                    expert_ids,
                    layout: ExpertStorageLayout::RawV1,
                    scale4_entries: None,
                    scale4_validation: None,
                    ticket,
                })
            }
            ExpertCorpusStorage::Scale4 {
                manifest,
                validation,
            } => {
                let plan = ExpertBatchPlan::open_scale4_manifest_for_corpus(
                    &self.model_root,
                    manifest,
                    validation,
                    layer,
                    &expert_ids,
                    0,
                    self.stream_cache_policy,
                )?;
                let identity = plan.scale4_identity.ok_or_else(|| {
                    DeltafinError::new("scale4 read plan lost its sidecar identity")
                })?;
                let scale4_entries = plan.scale4_entries.clone();
                let reader_verified = plan.scale4_reader_verified.clone();
                let ticket = reader.submit(plan.read_plan(), priority)?;
                Ok(ExpertReadTicket {
                    layer,
                    expert_ids,
                    layout: ExpertStorageLayout::Scale4V2,
                    scale4_entries,
                    scale4_validation: Some(Scale4BatchValidation {
                        cache: Arc::clone(validation),
                        layer,
                        identity,
                        reader_verified,
                    }),
                    ticket,
                })
            }
        }
    }

    pub fn read_decode(
        &self,
        reader: &Reader,
        layer: u32,
        ascending_expert_ids: &[u16],
    ) -> Result<ExpertReadBatch> {
        self.submit_decode(reader, layer, ascending_expert_ids)?
            .wait()
    }

    /// Submit one canonical raw-v1 expert union for a target-sequence tile.
    ///
    /// IDs must already be unique and strictly ascending because provider
    /// route-to-storage indices are defined in that canonical order. This API
    /// deliberately rejects duplicates rather than silently deduplicating or
    /// sorting them. Unions of up to 16 experts retain the catalog's inline
    /// request path; wider unions use the same bounded `Reader` arena with a
    /// deferred exact-size plan and one complete-file job per expert.
    pub fn submit_union(
        &self,
        reader: &Reader,
        layer: u32,
        canonical_expert_ids: &[u16],
    ) -> Result<ExpertUnionReadTicket> {
        self.submit_union_with_priority(reader, layer, canonical_expert_ids, ReadPriority::Demand)
    }

    pub fn submit_union_with_priority(
        &self,
        reader: &Reader,
        layer: u32,
        canonical_expert_ids: &[u16],
        priority: ReadPriority,
    ) -> Result<ExpertUnionReadTicket> {
        let selection = canonical_union_selection(layer, canonical_expert_ids)?;
        self.ensure_available(layer, selection.expert_ids())?;
        let (layout, expected_bytes, expected_jobs, scale4_entries, scale4_validation, ticket) =
            match &self.storage {
                ExpertCorpusStorage::Raw(catalog) => {
                    let expected_bytes =
                        checked_batch_bytes(selection.len(), K3_EXPERT_SOURCE_BYTES)?;
                    if expected_bytes > K3_EXPERT_UNION_MAX_BYTES {
                        return Err(DeltafinError::new(
                            "expert union exceeds its bounded raw-v1 arena contract",
                        ));
                    }
                    let ticket = if selection.len() <= K3_EXPERT_TOP_K {
                        reader.submit_deferred_exact(
                            catalog,
                            selection.source_indices(),
                            K3_EXPERT_BUFFER_KIND,
                            priority,
                        )?
                    } else {
                        // The inline catalog batch is intentionally fixed at the hot
                        // decode top-k. A wider sequence union still enters the Reader's
                        // bounded arena, opens every source only on a worker with
                        // O_NOFOLLOW, and validates the live descriptor's exact length.
                        let plan = ExpertBatchPlan::open_raw_cache_with_cache_policy(
                            &self.model_root,
                            layer,
                            selection.expert_ids(),
                            DEFAULT_EXPERT_CHUNK_BYTES,
                            self.stream_cache_policy,
                        )?;
                        reader.submit(plan.read_plan(), priority)?
                    };
                    (
                        ExpertStorageLayout::RawV1,
                        expected_bytes,
                        selection.len(),
                        None,
                        None,
                        ticket,
                    )
                }
                ExpertCorpusStorage::Scale4 {
                    manifest,
                    validation,
                } => {
                    let expected_bytes =
                        checked_batch_bytes(selection.len(), K3_SCALE4_BLOB_BYTES)?;
                    if expected_bytes > K3_SCALE4_UNION_MAX_BYTES {
                        return Err(DeltafinError::new(
                            "expert union exceeds its bounded scale4-v2 arena contract",
                        ));
                    }
                    let plan = ExpertBatchPlan::open_scale4_manifest_for_corpus(
                        &self.model_root,
                        manifest,
                        validation,
                        layer,
                        selection.expert_ids(),
                        0,
                        self.stream_cache_policy,
                    )?;
                    let identity = plan.scale4_identity.ok_or_else(|| {
                        DeltafinError::new("scale4 union plan lost its sidecar identity")
                    })?;
                    let expected_jobs = plan.read_plan().jobs();
                    let scale4_entries = plan.scale4_entries.clone();
                    let reader_verified = plan.scale4_reader_verified.clone();
                    let ticket = reader.submit(plan.read_plan(), priority)?;
                    (
                        ExpertStorageLayout::Scale4V2,
                        expected_bytes,
                        expected_jobs,
                        scale4_entries,
                        Some(Scale4BatchValidation {
                            cache: Arc::clone(validation),
                            layer,
                            identity,
                            reader_verified,
                        }),
                        ticket,
                    )
                }
            };
        Ok(ExpertUnionReadTicket {
            layer,
            expert_ids: selection.expert_ids().to_vec().into_boxed_slice(),
            layout,
            expected_bytes,
            expected_jobs,
            scale4_entries,
            scale4_validation,
            ticket,
        })
    }

    /// Submit one scheduling-only expert read without ever admitting a remote
    /// fetch. A partial lazy installation returns `None`; the authoritative
    /// demand path remains responsible for fetching and validating misses.
    /// Complete installations reuse the ordinary authenticated one-expert
    /// plan at prefetch priority.
    pub fn try_submit_local_prefetch_one(
        &self,
        reader: &Reader,
        layer: u32,
        expert: u16,
    ) -> Result<Option<ExpertUnionReadTicket>> {
        if self.lazy_missing_files() != 0 {
            return Ok(None);
        }
        self.submit_union_with_priority(reader, layer, &[expert], ReadPriority::Prefetch)
            .map(Some)
    }

    pub fn read_union(
        &self,
        reader: &Reader,
        layer: u32,
        canonical_expert_ids: &[u16],
    ) -> Result<ExpertUnionReadBatch> {
        self.submit_union(reader, layer, canonical_expert_ids)?
            .wait()
    }

    pub fn source_count(&self) -> usize {
        match &self.storage {
            ExpertCorpusStorage::Raw(catalog) => catalog.source_count(),
            ExpertCorpusStorage::Scale4 { manifest, .. } => manifest.entries().len(),
        }
    }

    pub const fn layout(&self) -> ExpertStorageLayout {
        match &self.storage {
            ExpertCorpusStorage::Raw(_) => ExpertStorageLayout::RawV1,
            ExpertCorpusStorage::Scale4 { .. } => ExpertStorageLayout::Scale4V2,
        }
    }

    pub fn lazy_missing_files(&self) -> usize {
        self.lazy
            .as_ref()
            .map_or(0, |lazy| lazy.remaining.load(Ordering::Acquire))
    }

    fn ensure_available(&self, layer: u32, expert_ids: &[u16]) -> Result<()> {
        match &self.lazy {
            Some(lazy) => lazy.ensure(layer, expert_ids),
            None => Ok(()),
        }
    }
}

fn ensure_raw_cache_directory(model_root: &Path) -> Result<()> {
    let root = model_root.join("k3-experts");
    match fs::symlink_metadata(&root) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(DeltafinError::new(format!(
            "raw expert cache is not a real directory: {}",
            root.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => match fs::create_dir(&root) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                ensure_raw_cache_directory(model_root)
            }
            Err(error) => Err(DeltafinError::new(format!(
                "create raw expert cache {}: {error}",
                root.display()
            ))),
        },
        Err(error) => Err(DeltafinError::new(format!(
            "inspect raw expert cache {}: {error}",
            root.display()
        ))),
    }
}

/// One startup-only directory census. It is deliberately a performance hint,
/// not a trust boundary: every actual read still opens O_NOFOLLOW and validates
/// the exact live descriptor. Canonical-looking corrupt entries fail closed;
/// partials and unrelated cache metadata are ignored.
fn raw_cache_missing_files(model_root: &Path) -> Result<usize> {
    let root = model_root.join("k3-experts");
    let mut present = vec![false; K3_EXPERT_RAW_FILES];
    let entries = fs::read_dir(&root).map_err(|error| {
        DeltafinError::new(format!("scan raw expert cache {}: {error}", root.display()))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            DeltafinError::new(format!("scan raw expert cache {}: {error}", root.display()))
        })?;
        let Some((layer, expert)) = parse_raw_expert_filename(&entry.file_name()) else {
            continue;
        };
        let file_type = entry.file_type().map_err(|error| {
            DeltafinError::new(format!("inspect raw expert cache entry: {error}"))
        })?;
        let metadata = entry.metadata().map_err(|error| {
            DeltafinError::new(format!(
                "inspect raw expert cache entry {}: {error}",
                entry.path().display()
            ))
        })?;
        if file_type.is_symlink()
            || !file_type.is_file()
            || !metadata.is_file()
            || metadata.len() != K3_EXPERT_SOURCE_BYTES as u64
        {
            return Err(DeltafinError::new(format!(
                "canonical raw expert cache entry is not an exact regular {}-byte file: {}",
                K3_EXPERT_SOURCE_BYTES,
                entry.path().display()
            )));
        }
        present[expert_source_index(layer, expert)?] = true;
    }
    Ok(present.into_iter().filter(|value| !*value).count())
}

pub(crate) fn parse_raw_expert_filename(name: &std::ffi::OsStr) -> Option<(u32, u16)> {
    let text = name.to_str()?;
    let body = text.strip_prefix('L')?.strip_suffix(".bin")?;
    let (layer_text, expert_text) = body.split_once("-E")?;
    if layer_text.is_empty()
        || expert_text.is_empty()
        || !layer_text.bytes().all(|byte| byte.is_ascii_digit())
        || !expert_text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let layer = layer_text.parse::<u32>().ok()?;
    let expert = expert_text.parse::<u16>().ok()?;
    if !(K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(&layer)
        || expert as usize >= K3_EXPERTS_PER_LAYER
        || text != format!("L{layer}-E{expert}.bin")
    {
        return None;
    }
    Some((layer, expert))
}

fn expert_source_index(layer: u32, expert: u16) -> Result<usize> {
    validate_layer(layer)?;
    if expert as usize >= K3_EXPERTS_PER_LAYER {
        return Err(DeltafinError::new("raw expert source ID is out of range"));
    }
    (layer as usize - K3_MOE_LAYER_FIRST as usize)
        .checked_mul(K3_EXPERTS_PER_LAYER)
        .and_then(|base| base.checked_add(expert as usize))
        .ok_or_else(|| DeltafinError::new("raw expert source index overflows usize"))
}

pub struct ExpertReadTicket {
    layer: u32,
    expert_ids: [u16; K3_EXPERT_TOP_K],
    layout: ExpertStorageLayout,
    scale4_entries: Option<Box<[ManifestEntry]>>,
    scale4_validation: Option<Scale4BatchValidation>,
    ticket: ReadTicket,
}

impl ExpertReadTicket {
    pub fn is_ready(&self) -> bool {
        self.ticket.is_ready()
    }

    pub fn wait(self) -> Result<ExpertReadBatch> {
        let (buffers, stats) = self.ticket.wait()?;
        validate_completed_layout(
            self.layout,
            &buffers,
            self.scale4_entries.as_deref(),
            self.scale4_validation.as_ref(),
        )?;
        Ok(ExpertReadBatch {
            layer: self.layer,
            expert_ids: self.expert_ids,
            layout: self.layout,
            buffers,
            stats,
        })
    }
}

/// One exact 16-expert, ascending-storage-order arena lease. The provider may
/// borrow `buffers().pointer(Other)` only while this value remains live.
pub struct ExpertReadBatch {
    layer: u32,
    expert_ids: [u16; K3_EXPERT_TOP_K],
    layout: ExpertStorageLayout,
    buffers: LayerBuffers,
    stats: ReadStats,
}

impl ExpertReadBatch {
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    pub fn expert_ids(&self) -> &[u16; K3_EXPERT_TOP_K] {
        &self.expert_ids
    }

    pub fn descriptors(
        &self,
    ) -> &[[ExpertMatrixDescriptorV1; K3_EXPERT_COMPONENTS]; K3_EXPERT_TOP_K] {
        match self.layout {
            ExpertStorageLayout::RawV1 => &K3_DECODE_MATRIX_DESCRIPTORS,
            ExpertStorageLayout::Scale4V2 => &K3_SCALE4_DECODE_MATRIX_DESCRIPTORS,
        }
    }

    pub const fn layout(&self) -> ExpertStorageLayout {
        self.layout
    }

    pub const fn buffers(&self) -> &LayerBuffers {
        &self.buffers
    }

    pub const fn stats(&self) -> ReadStats {
        self.stats
    }

    pub fn into_parts(self) -> ([u16; K3_EXPERT_TOP_K], LayerBuffers, ReadStats) {
        (self.expert_ids, self.buffers, self.stats)
    }
}

pub struct ExpertUnionReadTicket {
    layer: u32,
    expert_ids: Box<[u16]>,
    layout: ExpertStorageLayout,
    expected_bytes: usize,
    expected_jobs: usize,
    scale4_entries: Option<Box<[ManifestEntry]>>,
    scale4_validation: Option<Scale4BatchValidation>,
    ticket: ReadTicket,
}

impl ExpertUnionReadTicket {
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    pub const fn layout(&self) -> ExpertStorageLayout {
        self.layout
    }

    pub fn is_ready(&self) -> bool {
        self.ticket.is_ready()
    }

    /// Bitmask over `expert_ids()` of the experts whose complete span has
    /// already landed, while the rest of this union is still being read.
    ///
    /// Scale4 reads three raw planes plus one authenticated sidecar record per
    /// expert; a bit is set only when every one of that expert's reads
    /// finished, so a set bit always means one whole authenticated span.
    /// Zero means "no partial view", which is always a legal answer.
    pub fn arrived_experts(&self) -> u64 {
        if self.expert_ids.len() > u64::BITS as usize {
            return 0;
        }
        self.ticket.completed_destination_slots(
            K3_EXPERT_BUFFER_KIND,
            self.layout.expert_span_bytes(),
            self.expert_ids.len(),
        )
    }

    /// Borrow one arrived expert's span.
    ///
    /// # Safety
    /// `slot` must be set in a bitmask this ticket just returned from
    /// [`Self::arrived_experts`], and the slice must be dropped before the
    /// ticket is waited on or cancelled.
    pub unsafe fn arrived_span(&self, slot: usize) -> Option<&[u8]> {
        // SAFETY: forwarded contract -- the caller proved this slot complete
        // and keeps the borrow inside the ticket's lifetime.
        unsafe {
            self.ticket.destination_slot(
                K3_EXPERT_BUFFER_KIND,
                self.layout.expert_span_bytes(),
                slot,
            )
        }
    }

    /// Block until another of this union's reads finishes, or the whole union
    /// settles. `seen` is the value a previous call returned (zero to start).
    pub fn wait_for_read_progress(&self, seen: u64) -> u64 {
        self.ticket.wait_for_job_progress(seen)
    }

    pub fn completed_reads(&self) -> u64 {
        self.ticket.completed_jobs()
    }

    pub fn cancel_unclaimed(&self) {
        self.ticket.cancel_unclaimed();
    }

    pub fn drain_cancelled(self) {
        self.ticket.drain_cancelled();
    }

    pub fn wait(self) -> Result<ExpertUnionReadBatch> {
        let (buffers, stats) = self.ticket.wait()?;
        if !buffers.quantized().is_empty()
            || !buffers.scales().is_empty()
            || buffers.other().len() != self.expected_bytes
            || stats.bytes != self.expected_bytes as u64
            || stats.jobs != self.expected_jobs
        {
            return Err(DeltafinError::new(format!(
                "expert union read returned an invalid slab: layout={:?} ids={} bytes={} expected={} jobs={} expected_jobs={} stats_bytes={}",
                self.layout,
                self.expert_ids.len(),
                buffers.other().len(),
                self.expected_bytes,
                stats.jobs,
                self.expected_jobs,
                stats.bytes,
            )));
        }
        validate_completed_layout(
            self.layout,
            &buffers,
            self.scale4_entries.as_deref(),
            self.scale4_validation.as_ref(),
        )?;
        Ok(ExpertUnionReadBatch {
            layer: self.layer,
            expert_ids: self.expert_ids,
            layout: self.layout,
            buffers,
            stats,
        })
    }
}

/// One checked bounded canonical-storage-order arena lease. The raw
/// bytes are contiguous and expert-major: slot `i` occupies exactly
/// `K3_EXPERT_SOURCE_BYTES` bytes and belongs to `expert_ids()[i]`.
pub struct ExpertUnionReadBatch {
    layer: u32,
    expert_ids: Box<[u16]>,
    layout: ExpertStorageLayout,
    buffers: LayerBuffers,
    stats: ReadStats,
}

impl ExpertUnionReadBatch {
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    pub const fn layout(&self) -> ExpertStorageLayout {
        self.layout
    }

    pub const fn buffers(&self) -> &LayerBuffers {
        &self.buffers
    }

    pub const fn stats(&self) -> ReadStats {
        self.stats
    }

    pub fn into_parts(self) -> (Box<[u16]>, LayerBuffers, ReadStats) {
        (self.expert_ids, self.buffers, self.stats)
    }
}

#[derive(Debug)]
pub struct ExpertBatchPlan {
    layer: u32,
    expert_ids: Box<[u16]>,
    descriptors: Box<[[ExpertMatrixDescriptorV1; K3_EXPERT_COMPONENTS]]>,
    scale4_entries: Option<Box<[ManifestEntry]>>,
    scale4_identity: Option<DeferredSourceIdentity>,
    scale4_reader_verified: Box<[u16]>,
    read_plan: ReadPlan,
}

impl ExpertBatchPlan {
    /// Open the explicitly selected on-disk layout without format fallback.
    ///
    /// In particular, an activated scale4-v2 corpus must never be interpreted
    /// as raw-v1 or silently downgraded. Its header, manifest, per-record hash,
    /// exponent-table, and two-source gather contract require a distinct plan.
    pub fn open(
        model_root: &Path,
        layer: u32,
        routed_experts: &[u16],
        chunk_bytes: usize,
        layout: ExpertStorageLayout,
    ) -> Result<Self> {
        match layout {
            ExpertStorageLayout::RawV1 => {
                Self::open_raw_cache(model_root, layer, routed_experts, chunk_bytes)
            }
            ExpertStorageLayout::Scale4V2 => {
                let manifest = Scale4Manifest::load_full(model_root.join("k3-experts-scale4"))?;
                Self::open_scale4_manifest(
                    model_root,
                    &manifest,
                    layer,
                    routed_experts,
                    chunk_bytes,
                )
            }
        }
    }

    pub fn open_raw_cache(
        model_root: &Path,
        layer: u32,
        routed_experts: &[u16],
        chunk_bytes: usize,
    ) -> Result<Self> {
        Self::open_raw_cache_with_cache_policy(
            model_root,
            layer,
            routed_experts,
            chunk_bytes,
            CachePolicy::Streaming,
        )
    }

    pub fn open_raw_cache_with_cache_policy(
        model_root: &Path,
        layer: u32,
        routed_experts: &[u16],
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        // Loose routed experts are deliberately not opened here. The fixed
        // reader pool opens them concurrently with O_NOFOLLOW, validates the
        // exact canonical length on the live descriptor, reads one complete
        // expert-major span, and closes it. That removes both the serial
        // control-thread open/stat loop and the persistent-FD explosion that
        // the legacy one-file-per-expert corpus otherwise causes.
        validate_layer(layer)?;
        if routed_experts.is_empty() {
            return Err(DeltafinError::new(
                "an expert read plan needs at least one routed expert",
            ));
        }
        let expert_ids = canonical_expert_ids(routed_experts)?;
        if chunk_bytes != 0 && chunk_bytes < K3_EXPERT_SOURCE_BYTES {
            return Err(DeltafinError::new(format!(
                "raw expert reads must keep each canonical {}-byte span in one job; chunk size {chunk_bytes} is too small",
                K3_EXPERT_SOURCE_BYTES,
            )));
        }

        let raw_bytes = checked_batch_bytes(expert_ids.len(), K3_EXPERT_SOURCE_BYTES)?;
        let mut extents = Vec::with_capacity(expert_ids.len());
        let mut descriptors = Vec::with_capacity(expert_ids.len());
        let cache = model_root.join("k3-experts");
        for (batch_index, &expert) in expert_ids.iter().enumerate() {
            let path = expert_path(&cache, layer, expert);
            let expert_destination = batch_index
                .checked_mul(K3_EXPERT_SOURCE_BYTES)
                .ok_or_else(|| DeltafinError::new("expert raw offset overflows usize"))?;
            extents.push(Extent::new(
                &path,
                0,
                K3_EXPERT_BUFFER_KIND,
                expert_destination,
                K3_EXPERT_SOURCE_BYTES,
            ));
            let mut matrices = [ExpertMatrixDescriptorV1 {
                packed_offset: 0,
                scale_offset: 0,
                rows: 0,
                columns: 0,
                packed_columns: 0,
                scale_columns: 0,
            }; K3_EXPERT_COMPONENTS];
            for matrix in 0..K3_EXPERT_COMPONENTS {
                let packed_destination = expert_destination
                    .checked_add(SOURCE_PACKED_OFFSETS[matrix] as usize)
                    .ok_or_else(|| DeltafinError::new("expert packed offset overflows usize"))?;
                let scale_destination = expert_destination
                    .checked_add(SOURCE_SCALE_OFFSETS[matrix] as usize)
                    .ok_or_else(|| DeltafinError::new("expert scale offset overflows usize"))?;
                let (rows, columns, packed_columns, scale_columns) = MATRIX_SHAPES[matrix];
                matrices[matrix] = ExpertMatrixDescriptorV1 {
                    packed_offset: packed_destination as u64,
                    scale_offset: scale_destination as u64,
                    rows,
                    columns,
                    packed_columns,
                    scale_columns,
                };
            }
            descriptors.push(matrices);
        }
        let read_plan = ReadPlan::open_deferred_exact(
            extents,
            BufferLengths::new(0, 0, raw_bytes),
            chunk_bytes,
            cache_policy,
            K3_EXPERT_SOURCE_BYTES as u64,
        )?;
        Ok(Self {
            layer,
            expert_ids: expert_ids.into_boxed_slice(),
            descriptors: descriptors.into_boxed_slice(),
            scale4_entries: None,
            scale4_identity: None,
            scale4_reader_verified: Box::new([]),
            read_plan,
        })
    }

    fn open_scale4_manifest(
        model_root: &Path,
        manifest: &Scale4Manifest,
        layer: u32,
        routed_experts: &[u16],
        chunk_bytes: usize,
    ) -> Result<Self> {
        Self::open_scale4_manifest_with_identity(
            model_root,
            manifest,
            layer,
            routed_experts,
            chunk_bytes,
            None,
            None,
            CachePolicy::Streaming,
        )
    }

    fn open_scale4_manifest_for_corpus(
        model_root: &Path,
        manifest: &Scale4Manifest,
        validation: &Scale4ValidationCache,
        layer: u32,
        routed_experts: &[u16],
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        let reusable_identity = validation.reusable_corpus_identity(layer);
        let plan = Self::open_scale4_manifest_with_identity(
            model_root,
            manifest,
            layer,
            routed_experts,
            chunk_bytes,
            reusable_identity,
            Some(validation),
            cache_policy,
        )?;
        let identity = plan.scale4_identity.ok_or_else(|| {
            DeltafinError::new("scale4 corpus plan lost its sidecar identity contract")
        })?;
        validation.admit_corpus_identity(layer, identity)?;
        validation.prepare(layer, identity);
        Ok(plan)
    }

    #[allow(clippy::too_many_arguments)]
    fn open_scale4_manifest_with_identity(
        model_root: &Path,
        manifest: &Scale4Manifest,
        layer: u32,
        routed_experts: &[u16],
        chunk_bytes: usize,
        captured_identity: Option<DeferredSourceIdentity>,
        validation: Option<&Scale4ValidationCache>,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        validate_layer(layer)?;
        if routed_experts.is_empty() {
            return Err(DeltafinError::new(
                "a scale4 expert read plan needs at least one routed expert",
            ));
        }
        let expert_ids = canonical_expert_ids(routed_experts)?;
        let blob_bytes = checked_batch_bytes(expert_ids.len(), K3_SCALE4_BLOB_BYTES)?;
        let layer_extent = manifest.layer_extent(layer).ok_or_else(|| {
            DeltafinError::new(format!("scale4 manifest has no records for layer {layer}"))
        })?;
        let layer_bytes = layer_extent.file_bytes;
        debug_assert_eq!(
            layer_bytes,
            layer_extent.records as u64 * SCALE4_RECORD_BYTES as u64
        );

        let raw_root = model_root.join("k3-experts");
        let sidecar = manifest.root().join(format!("L{layer}.sc4"));
        let mut extents = Vec::with_capacity(expert_ids.len() * 4);
        let mut source_lengths = Vec::with_capacity(expert_ids.len() + 1);
        let mut descriptors = Vec::with_capacity(expert_ids.len());
        let mut entries = Vec::with_capacity(expert_ids.len());
        let mut reader_verified = Vec::with_capacity(expert_ids.len());
        let sidecar_source = match captured_identity {
            Some(identity) => {
                DeferredSourceLength::new_with_captured_identity(&sidecar, layer_bytes, identity)?
            }
            None => DeferredSourceLength::new_with_live_identity(&sidecar, layer_bytes)?,
        };
        let scale4_identity = sidecar_source.identity().ok_or_else(|| {
            DeltafinError::new("scale4 sidecar identity contract was not captured")
        })?;
        source_lengths.push(sidecar_source);
        for (slot, &expert) in expert_ids.iter().enumerate() {
            let entry = *manifest.entry(layer, expert).ok_or_else(|| {
                DeltafinError::new(format!(
                    "scale4 manifest has no entry for L{layer}-E{expert}"
                ))
            })?;
            let expert_destination = slot
                .checked_mul(K3_SCALE4_BLOB_BYTES)
                .ok_or_else(|| DeltafinError::new("scale4 expert offset overflows usize"))?;
            let raw = expert_path(&raw_root, layer, expert);

            source_lengths.push(DeferredSourceLength::new(
                &raw,
                K3_EXPERT_SOURCE_BYTES as u64,
            ));
            let mut sidecar_destinations = Vec::with_capacity(K3_EXPERT_COMPONENTS + 1);
            sidecar_destinations.push(VectoredDestination::new(
                K3_EXPERT_BUFFER_KIND,
                expert_destination,
                SCALE4_HEADER_BYTES,
            ));
            let mut sidecar_cursor = SCALE4_HEADER_BYTES as u64;

            let mut matrices = [EMPTY_MATRIX_DESCRIPTOR; K3_EXPERT_COMPONENTS];
            for matrix in 0..K3_EXPERT_COMPONENTS {
                let (raw_offset, raw_length) = expert_scale4::RAW_LAYOUT[matrix * 2];
                let (packed_offset, packed_length) = COMPACT_LAYOUT[matrix * 2];
                if raw_length != packed_length as usize {
                    return Err(DeltafinError::new(
                        "scale4 packed source and destination extents disagree",
                    ));
                }
                let (scale_source_offset, scale_length) = SIDECAR_LAYOUT[matrix];
                if u64::from(scale_source_offset) != sidecar_cursor {
                    return Err(DeltafinError::new(
                        "scale4 sidecar planes are not one contiguous canonical record",
                    ));
                }
                let (scale_offset, compact_scale_length) = COMPACT_LAYOUT[matrix * 2 + 1];
                if scale_length != compact_scale_length {
                    return Err(DeltafinError::new(
                        "scale4 sidecar and destination extents disagree",
                    ));
                }
                let packed_destination = expert_destination
                    .checked_add(packed_offset as usize)
                    .ok_or_else(|| DeltafinError::new("scale4 packed offset overflows usize"))?;
                let scale_destination = expert_destination
                    .checked_add(scale_offset as usize)
                    .ok_or_else(|| DeltafinError::new("scale4 scale offset overflows usize"))?;
                extents.push(Extent::new(
                    &raw,
                    raw_offset,
                    K3_EXPERT_BUFFER_KIND,
                    packed_destination,
                    raw_length,
                ));
                sidecar_destinations.push(VectoredDestination::new(
                    K3_EXPERT_BUFFER_KIND,
                    scale_destination,
                    scale_length as usize,
                ));
                sidecar_cursor = sidecar_cursor
                    .checked_add(u64::from(scale_length))
                    .ok_or_else(|| DeltafinError::new("scale4 sidecar length overflows u64"))?;
                let (rows, columns, packed_columns, scale_columns) = MATRIX_SHAPES[matrix];
                matrices[matrix] = ExpertMatrixDescriptorV1 {
                    packed_offset: packed_destination as u64,
                    scale_offset: scale_destination as u64,
                    rows,
                    columns,
                    packed_columns,
                    scale_columns,
                };
            }
            if sidecar_cursor != SCALE4_RECORD_BYTES as u64 {
                return Err(DeltafinError::new(
                    "scale4 sidecar planes do not cover one canonical record",
                ));
            }
            let verify_in_reader = validation
                .is_some_and(|validation| validation.needs_hash(layer, expert, scale4_identity));
            if verify_in_reader {
                extents.push(Extent::vectored_verified(
                    &sidecar,
                    entry.record_offset,
                    sidecar_destinations,
                    entry.record_sha256,
                )?);
                reader_verified.push(expert);
            } else {
                extents.push(Extent::vectored(
                    &sidecar,
                    entry.record_offset,
                    sidecar_destinations,
                )?);
            }
            entries.push(entry);
            descriptors.push(matrices);
        }
        let read_plan = ReadPlan::open_deferred_ranges(
            extents,
            source_lengths,
            BufferLengths::new(0, 0, blob_bytes),
            chunk_bytes,
            cache_policy,
        )?;
        Ok(Self {
            layer,
            expert_ids: expert_ids.into_boxed_slice(),
            descriptors: descriptors.into_boxed_slice(),
            scale4_entries: Some(entries.into_boxed_slice()),
            scale4_identity: Some(scale4_identity),
            scale4_reader_verified: reader_verified.into_boxed_slice(),
            read_plan,
        })
    }

    pub fn open_raw_cache_default(
        model_root: &Path,
        layer: u32,
        routed_experts: &[u16],
    ) -> Result<Self> {
        Self::open_raw_cache(
            model_root,
            layer,
            routed_experts,
            DEFAULT_EXPERT_CHUNK_BYTES,
        )
    }

    pub const fn layer(&self) -> u32 {
        self.layer
    }

    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    pub fn descriptors(&self) -> &[[ExpertMatrixDescriptorV1; K3_EXPERT_COMPONENTS]] {
        &self.descriptors
    }

    pub const fn read_plan(&self) -> &ReadPlan {
        &self.read_plan
    }
}

fn validate_completed_layout(
    layout: ExpertStorageLayout,
    buffers: &LayerBuffers,
    scale4_entries: Option<&[ManifestEntry]>,
    scale4_validation: Option<&Scale4BatchValidation>,
) -> Result<()> {
    match layout {
        ExpertStorageLayout::RawV1 => {
            if scale4_entries.is_some() || scale4_validation.is_some() {
                return Err(DeltafinError::new(
                    "raw-v1 read unexpectedly carries scale4 validation entries",
                ));
            }
            Ok(())
        }
        ExpertStorageLayout::Scale4V2 => {
            let entries = scale4_entries.ok_or_else(|| {
                DeltafinError::new("scale4 read has no activated manifest entries")
            })?;
            let expected_bytes = checked_batch_bytes(entries.len(), K3_SCALE4_BLOB_BYTES)?;
            if !buffers.quantized().is_empty()
                || !buffers.scales().is_empty()
                || buffers.other().len() != expected_bytes
            {
                return Err(DeltafinError::new(
                    "scale4 read returned a non-canonical expert-major slab",
                ));
            }
            if let Some(validation) = scale4_validation {
                if entries.iter().any(|entry| entry.layer != validation.layer) {
                    return Err(DeltafinError::new(
                        "scale4 validation cache layer disagrees with manifest entries",
                    ));
                }
                if validation
                    .reader_verified
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                    || validation.reader_verified.iter().any(|expert| {
                        entries
                            .binary_search_by_key(expert, |entry| entry.expert)
                            .is_err()
                    })
                {
                    return Err(DeltafinError::new(
                        "scale4 reader-verification roster is not a canonical manifest subset",
                    ));
                }
            }
            for (slot, entry) in entries.iter().enumerate() {
                let start = slot * K3_SCALE4_BLOB_BYTES;
                let blob = &buffers.other()[start..start + K3_SCALE4_BLOB_BYTES];
                let header = expert_scale4::parse_header(&blob[..SCALE4_HEADER_BYTES])?;
                if header.source_sha256 != entry.source_sha256 || header.bases != entry.bases {
                    return Err(DeltafinError::new(format!(
                        "scale4 header disagrees with activated manifest for L{}-E{}",
                        entry.layer, entry.expert
                    )));
                }
                let needs_hash = scale4_validation.is_none_or(|validation| {
                    validation
                        .cache
                        .needs_hash(validation.layer, entry.expert, validation.identity)
                });
                if needs_hash {
                    #[cfg(test)]
                    if let Some(validation) = scale4_validation {
                        validation
                            .cache
                            .record_hashes
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    let verified_by_reader = scale4_validation.is_some_and(|validation| {
                        validation
                            .reader_verified
                            .binary_search(&entry.expert)
                            .is_ok()
                    });
                    if !verified_by_reader {
                        // Direct plans retain this exact fallback. Production
                        // corpus misses are authenticated over the same four
                        // ordered extents in parallel reader workers before
                        // ReadTicket can publish the batch.
                        let mut record_digest = crate::packfile::DigestState::new();
                        record_digest.update(&blob[..SCALE4_HEADER_BYTES]);
                        for matrix in 0..K3_EXPERT_COMPONENTS {
                            let (offset, length) = COMPACT_LAYOUT[matrix * 2 + 1];
                            let offset = offset as usize;
                            let length = length as usize;
                            record_digest.update(&blob[offset..offset + length]);
                        }
                        if record_digest.finalize() != entry.record_sha256 {
                            return Err(DeltafinError::new(format!(
                                "scale4 record disagrees with activated manifest for L{}-E{}",
                                entry.layer, entry.expert
                            )));
                        }
                    }
                }
            }
            if let Some(validation) = scale4_validation {
                if !validation
                    .cache
                    .publish(validation.layer, validation.identity, entries)
                {
                    return Err(DeltafinError::new(
                        "scale4 sidecar identity changed while validating selected records",
                    ));
                }
            }
            Ok(())
        }
    }
}

fn raw_expert_source_name(layer: u32, expert: u16) -> Result<DeferredSourceName> {
    // Maximum canonical spelling is `L92-E895.bin` (12 bytes). Decimal
    // assembly into a stack buffer avoids 82,432 temporary String allocations
    // while constructing the session catalog.
    let mut bytes = [0_u8; 16];
    let mut cursor = 0;
    bytes[cursor] = b'L';
    cursor += 1;
    append_decimal(&mut bytes, &mut cursor, layer)?;
    bytes[cursor] = b'-';
    bytes[cursor + 1] = b'E';
    cursor += 2;
    append_decimal(&mut bytes, &mut cursor, u32::from(expert))?;
    bytes[cursor..cursor + 4].copy_from_slice(b".bin");
    cursor += 4;
    let name = std::str::from_utf8(&bytes[..cursor])
        .map_err(|_| DeltafinError::new("internal expert name is not UTF-8"))?;
    DeferredSourceName::new(name)
}

fn append_decimal(buffer: &mut [u8], cursor: &mut usize, mut value: u32) -> Result<()> {
    let mut reverse = [0_u8; 10];
    let mut digits = 0;
    loop {
        reverse[digits] = b'0' + (value % 10) as u8;
        digits += 1;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    let end = cursor
        .checked_add(digits)
        .ok_or_else(|| DeltafinError::new("expert name length overflows usize"))?;
    if end > buffer.len() {
        return Err(DeltafinError::new("expert name exceeds its stack buffer"));
    }
    for digit in reverse[..digits].iter().rev() {
        buffer[*cursor] = *digit;
        *cursor += 1;
    }
    Ok(())
}

struct CanonicalExpertUnion {
    expert_ids: Box<[u16]>,
    source_indices: Box<[u32]>,
}

impl CanonicalExpertUnion {
    fn len(&self) -> usize {
        self.expert_ids.len()
    }

    fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    fn source_indices(&self) -> &[u32] {
        &self.source_indices
    }
}

fn canonical_union_selection(
    layer: u32,
    canonical_expert_ids: &[u16],
) -> Result<CanonicalExpertUnion> {
    validate_layer(layer)?;
    if canonical_expert_ids.is_empty() || canonical_expert_ids.len() > K3_EXPERT_UNION_MAX {
        return Err(DeltafinError::new(format!(
            "expert union needs 1..={K3_EXPERT_UNION_MAX} experts; got {}",
            canonical_expert_ids.len()
        )));
    }
    let layer_base = (layer as usize - K3_MOE_LAYER_FIRST as usize)
        .checked_mul(K3_EXPERTS_PER_LAYER)
        .ok_or_else(|| DeltafinError::new("expert union layer offset overflows usize"))?;
    let mut expert_ids = Vec::with_capacity(canonical_expert_ids.len());
    let mut source_indices = Vec::with_capacity(canonical_expert_ids.len());
    let mut previous = None;
    for (slot, &expert) in canonical_expert_ids.iter().enumerate() {
        if usize::from(expert) >= K3_EXPERTS_PER_LAYER {
            return Err(DeltafinError::new(format!(
                "expert union ID {expert} is outside 0..{}",
                K3_EXPERTS_PER_LAYER - 1
            )));
        }
        if let Some(previous_expert) = previous
            && expert <= previous_expert
        {
            return Err(DeltafinError::new(format!(
                "expert union IDs must be unique and strictly ascending; slot {slot} has {expert} after {previous_expert}",
            )));
        }
        let source = layer_base
            .checked_add(usize::from(expert))
            .ok_or_else(|| DeltafinError::new("expert union source index overflows usize"))?;
        expert_ids.push(expert);
        source_indices.push(
            u32::try_from(source)
                .map_err(|_| DeltafinError::new("expert union source index exceeds u32"))?,
        );
        previous = Some(expert);
    }
    Ok(CanonicalExpertUnion {
        expert_ids: expert_ids.into_boxed_slice(),
        source_indices: source_indices.into_boxed_slice(),
    })
}

fn decode_source_indices(
    layer: u32,
    ascending_expert_ids: &[u16],
) -> Result<([u16; K3_EXPERT_TOP_K], [u32; K3_EXPERT_TOP_K])> {
    validate_layer(layer)?;
    if ascending_expert_ids.len() != K3_EXPERT_TOP_K {
        return Err(DeltafinError::new(format!(
            "decode expert read needs exactly {K3_EXPERT_TOP_K} experts; got {}",
            ascending_expert_ids.len()
        )));
    }
    let layer_base = (layer as usize - K3_MOE_LAYER_FIRST as usize)
        .checked_mul(K3_EXPERTS_PER_LAYER)
        .ok_or_else(|| DeltafinError::new("expert catalog layer offset overflows usize"))?;
    let mut expert_ids = [0_u16; K3_EXPERT_TOP_K];
    let mut source_indices = [0_u32; K3_EXPERT_TOP_K];
    let mut previous = None;
    for (slot, &expert) in ascending_expert_ids.iter().enumerate() {
        if usize::from(expert) >= K3_EXPERTS_PER_LAYER {
            return Err(DeltafinError::new(format!(
                "routed expert {expert} is outside 0..{}",
                K3_EXPERTS_PER_LAYER - 1
            )));
        }
        if let Some(previous_expert) = previous
            && expert <= previous_expert
        {
            return Err(DeltafinError::new(format!(
                "decode experts must be unique and strictly ascending; slot {slot} has {expert} after {previous_expert}",
            )));
        }
        let source = layer_base
            .checked_add(usize::from(expert))
            .ok_or_else(|| DeltafinError::new("expert catalog source index overflows usize"))?;
        expert_ids[slot] = expert;
        source_indices[slot] = u32::try_from(source)
            .map_err(|_| DeltafinError::new("expert catalog source index exceeds u32"))?;
        previous = Some(expert);
    }
    Ok((expert_ids, source_indices))
}

fn canonical_expert_ids(routed_experts: &[u16]) -> Result<Vec<u16>> {
    let mut seen = [false; K3_EXPERTS_PER_LAYER];
    for &expert in routed_experts {
        let index = usize::from(expert);
        if index >= K3_EXPERTS_PER_LAYER {
            return Err(DeltafinError::new(format!(
                "routed expert {expert} is outside 0..{}",
                K3_EXPERTS_PER_LAYER - 1
            )));
        }
        seen[index] = true;
    }

    // RouteArena::edge_to_slot and the established kimi_run fetch path both
    // define storage slots by ascending expert ID. First-occurrence order is
    // not interchangeable: an edge-to-slot index would then silently select
    // another expert's weights.
    Ok(seen
        .iter()
        .enumerate()
        .filter_map(|(expert, &selected)| selected.then_some(expert as u16))
        .collect())
}

fn checked_batch_bytes(experts: usize, expert_bytes: usize) -> Result<usize> {
    experts
        .checked_mul(expert_bytes)
        .ok_or_else(|| DeltafinError::new("expert batch buffer length overflows usize"))
}

fn validate_layer(layer: u32) -> Result<()> {
    if !(K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(&layer) {
        return Err(DeltafinError::new(format!(
            "MoE layer {layer} is outside {K3_MOE_LAYER_FIRST}..={K3_MOE_LAYER_LAST}"
        )));
    }
    Ok(())
}

fn expert_path(cache: &Path, layer: u32, expert: u16) -> PathBuf {
    cache.join(format!("L{layer}-E{expert}.bin"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage_homes::StorageHomes;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEST_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    struct TestModelRoot(PathBuf);

    impl TestModelRoot {
        fn with_experts(layer: u32, experts: impl IntoIterator<Item = u16>) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let serial = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "deltafin-experts-{}-{nonce}-{serial}",
                std::process::id()
            ));
            let cache = root.join("k3-experts");
            fs::create_dir_all(&cache).unwrap();
            for expert in experts {
                fs::File::create(expert_path(&cache, layer, expert))
                    .unwrap()
                    .set_len(K3_EXPERT_SOURCE_BYTES as u64)
                    .unwrap();
            }
            Self(root)
        }
    }

    impl Drop for TestModelRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn digest_text(digest: [u8; 32]) -> String {
        let mut text = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            write!(text, "{byte:02x}").unwrap();
        }
        text
    }

    fn write_synthetic_scale4(
        root: &Path,
        header_source_override: Option<[u8; 32]>,
    ) -> (Scale4Manifest, Vec<u8>, Vec<u8>) {
        write_synthetic_scale4_experts(root, header_source_override, 1)
    }

    fn write_synthetic_scale4_experts(
        root: &Path,
        header_source_override: Option<[u8; 32]>,
        expert_count: u16,
    ) -> (Scale4Manifest, Vec<u8>, Vec<u8>) {
        use crate::expert_scale4::manifest::{ManifestRow, manifest_bytes};
        use crate::packfile::digest_bytes;

        let raw_root = root.join("k3-experts");
        let side_root = root.join("k3-experts-scale4");
        fs::create_dir_all(&raw_root).unwrap();
        fs::create_dir_all(&side_root).unwrap();
        let mut raw = vec![0_u8; K3_EXPERT_SOURCE_BYTES];
        for (component, (offset, length)) in expert_scale4::RAW_LAYOUT.into_iter().enumerate() {
            raw[offset as usize..offset as usize + length].fill(0x11 * (component as u8 + 1));
        }
        let source_sha256 = digest_bytes(&raw);
        for expert in 0..expert_count {
            fs::write(expert_path(&raw_root, 1, expert), &raw).unwrap();
        }

        let bases = [16, 48, 80];
        let header =
            expert_scale4::build_header(header_source_override.unwrap_or(source_sha256), bases)
                .unwrap();
        let mut record = Vec::with_capacity(SCALE4_RECORD_BYTES);
        record.extend_from_slice(header.as_slice());
        for matrix in 0..K3_EXPERT_COMPONENTS {
            record.extend(std::iter::repeat_n(
                0x21 * (matrix as u8 + 1),
                expert_scale4::SCALE4_BYTES,
            ));
        }
        assert_eq!(record.len(), SCALE4_RECORD_BYTES);
        let record_sha256 = expert_scale4::record_digest(&record).unwrap();
        let mut layer_records = Vec::with_capacity(record.len() * expert_count as usize);
        for _ in 0..expert_count {
            layer_records.extend_from_slice(&record);
        }
        fs::write(side_root.join("L1.sc4"), layer_records).unwrap();

        let names: Vec<_> = (0..expert_count)
            .map(|expert| format!("L1-E{expert}.bin"))
            .collect();
        let manifest = manifest_bytes(
            (0..expert_count)
                .map(|expert| ManifestRow {
                    bases,
                    expert,
                    layer: 1,
                    record_sha256: digest_text(record_sha256),
                    source_sha256: digest_text(source_sha256),
                })
                .collect(),
            &names,
        )
        .unwrap();
        fs::write(side_root.join(expert_scale4::MANIFEST_NAME), manifest).unwrap();
        (
            Scale4Manifest::load_for_raw_names(&side_root, &names).unwrap(),
            raw,
            record,
        )
    }

    #[test]
    fn source_layout_and_full_corpus_count_are_exact() {
        assert_eq!(SOURCE_PACKED_OFFSETS, [0, 5_849_088, 11_698_176]);
        assert_eq!(SOURCE_SCALE_OFFSETS, [5_505_024, 11_354_112, 17_203_200]);
        assert_eq!(
            SOURCE_SCALE_OFFSETS[2] as usize + K3_EXPERT_SCALE_BYTES,
            K3_EXPERT_SOURCE_BYTES
        );
        assert_eq!(K3_EXPERT_SOURCE_COMPONENTS, 494_592);
        assert_eq!(K3_EXPERT_RAW_FILES, 82_432);
        assert_eq!(std::mem::size_of::<ExpertMatrixDescriptorV1>(), 32);
        assert_eq!(std::mem::align_of::<ExpertMatrixDescriptorV1>(), 8);
        assert_eq!(
            std::mem::offset_of!(ExpertMatrixDescriptorV1, packed_offset),
            0
        );
        assert_eq!(
            std::mem::offset_of!(ExpertMatrixDescriptorV1, scale_offset),
            8
        );
        assert_eq!(std::mem::offset_of!(ExpertMatrixDescriptorV1, rows), 16);
        assert_eq!(std::mem::offset_of!(ExpertMatrixDescriptorV1, columns), 20);
        assert_eq!(
            std::mem::offset_of!(ExpertMatrixDescriptorV1, packed_columns),
            24
        );
        assert_eq!(
            std::mem::offset_of!(ExpertMatrixDescriptorV1, scale_columns),
            28
        );
    }

    #[test]
    fn scale4_decode_descriptors_cover_all_16_experts_in_storage_order() {
        assert_eq!(K3_SCALE4_DECODE_MATRIX_DESCRIPTORS.len(), K3_EXPERT_TOP_K);
        for (slot, matrices) in K3_SCALE4_DECODE_MATRIX_DESCRIPTORS.iter().enumerate() {
            let expert_base = slot * K3_SCALE4_BLOB_BYTES;
            for (matrix, descriptor) in matrices.iter().enumerate() {
                let (rows, columns, packed_columns, scale_columns) = MATRIX_SHAPES[matrix];
                assert_eq!(
                    descriptor.packed_offset,
                    (expert_base + COMPACT_LAYOUT[matrix * 2].0 as usize) as u64
                );
                assert_eq!(
                    descriptor.scale_offset,
                    (expert_base + COMPACT_LAYOUT[matrix * 2 + 1].0 as usize) as u64
                );
                assert_eq!(
                    (
                        descriptor.rows,
                        descriptor.columns,
                        descriptor.packed_columns,
                        descriptor.scale_columns,
                    ),
                    (rows, columns, packed_columns, scale_columns)
                );
            }
            let final_scale = &matrices[K3_EXPERT_COMPONENTS - 1];
            let final_scale_bytes = COMPACT_LAYOUT[K3_EXPERT_COMPONENTS * 2 - 1].1 as u64;
            assert_eq!(
                final_scale.scale_offset + final_scale_bytes,
                ((slot + 1) * K3_SCALE4_BLOB_BYTES) as u64
            );
        }
    }

    #[test]
    fn decode_descriptor_tape_is_static_and_expert_major() {
        assert_eq!(K3_DECODE_MATRIX_DESCRIPTORS.len(), K3_EXPERT_TOP_K);
        for (slot, descriptors) in K3_DECODE_MATRIX_DESCRIPTORS.iter().enumerate() {
            let base = (slot * K3_EXPERT_SOURCE_BYTES) as u64;
            for matrix in 0..K3_EXPERT_COMPONENTS {
                let descriptor = descriptors[matrix];
                assert_eq!(
                    descriptor.packed_offset,
                    base + SOURCE_PACKED_OFFSETS[matrix]
                );
                assert_eq!(descriptor.scale_offset, base + SOURCE_SCALE_OFFSETS[matrix]);
                assert_eq!(
                    (
                        descriptor.rows,
                        descriptor.columns,
                        descriptor.packed_columns,
                        descriptor.scale_columns,
                    ),
                    MATRIX_SHAPES[matrix]
                );
            }
        }
    }

    #[test]
    fn raw_catalog_names_and_decode_indices_are_exact_without_sorting() {
        assert_eq!(raw_expert_source_name(1, 0).unwrap().as_str(), "L1-E0.bin");
        assert_eq!(
            raw_expert_source_name(92, 895).unwrap().as_str(),
            "L92-E895.bin"
        );

        let first: Vec<u16> = (0..K3_EXPERT_TOP_K as u16).collect();
        let (ids, indices) = decode_source_indices(1, &first).unwrap();
        assert_eq!(&ids, first.as_slice());
        assert_eq!(indices, std::array::from_fn(|index| index as u32));

        let last: Vec<u16> = (880..896).collect();
        let (_, indices) = decode_source_indices(92, &last).unwrap();
        let layer_base = 91 * K3_EXPERTS_PER_LAYER;
        assert_eq!(
            indices,
            std::array::from_fn(|index| (layer_base + 880 + index) as u32)
        );
    }

    #[test]
    fn lazy_cache_census_accepts_only_canonical_exact_expert_objects() {
        use std::ffi::OsStr;

        assert_eq!(
            parse_raw_expert_filename(OsStr::new("L1-E0.bin")),
            Some((1, 0))
        );
        assert_eq!(
            parse_raw_expert_filename(OsStr::new("L92-E895.bin")),
            Some((92, 895))
        );
        for invalid in [
            "L0-E0.bin",
            "L93-E0.bin",
            "L1-E896.bin",
            "L01-E0.bin",
            "L1-E00.bin",
            "L1-E0.bin.part",
            "notes.json",
        ] {
            assert_eq!(parse_raw_expert_filename(OsStr::new(invalid)), None);
        }

        let root = TestModelRoot::with_experts(17, [7]);
        fs::write(root.0.join("k3-experts/ignored.part"), [1, 2, 3]).unwrap();
        assert_eq!(
            raw_cache_missing_files(&root.0).unwrap(),
            K3_EXPERT_RAW_FILES - 1
        );
        fs::OpenOptions::new()
            .write(true)
            .open(root.0.join("k3-experts/L17-E7.bin"))
            .unwrap()
            .set_len(K3_EXPERT_SOURCE_BYTES as u64 - 1)
            .unwrap();
        assert!(raw_cache_missing_files(&root.0).is_err());
    }

    #[test]
    fn decode_fast_path_fails_closed_on_noncanonical_route_shape_or_order() {
        let ascending: Vec<u16> = (20..36).collect();
        assert!(decode_source_indices(0, &ascending).is_err());
        assert!(decode_source_indices(93, &ascending).is_err());
        assert!(decode_source_indices(1, &ascending[..15]).is_err());

        let mut duplicate = ascending.clone();
        duplicate[8] = duplicate[7];
        assert!(decode_source_indices(1, &duplicate).is_err());
        let mut descending = ascending.clone();
        descending.swap(8, 9);
        assert!(decode_source_indices(1, &descending).is_err());
        let mut out_of_range = ascending;
        out_of_range[15] = K3_EXPERTS_PER_LAYER as u16;
        assert!(decode_source_indices(1, &out_of_range).is_err());
    }

    #[test]
    fn expert_union_selection_is_bounded_canonical_and_catalog_exact() {
        let one = canonical_union_selection(1, &[895]).unwrap();
        assert_eq!(one.expert_ids(), &[895]);
        assert_eq!(one.source_indices(), &[895]);

        let current_verifier_maximum: Vec<u16> = (500..644).collect();
        let current = canonical_union_selection(92, &current_verifier_maximum).unwrap();
        assert_eq!(current.len(), 144);
        assert_eq!(current.expert_ids(), current_verifier_maximum.as_slice());

        let maximum: Vec<u16> = (0..K3_EXPERT_UNION_MAX as u16).collect();
        let selection = canonical_union_selection(92, &maximum).unwrap();
        assert_eq!(selection.len(), K3_EXPERT_UNION_MAX);
        assert_eq!(selection.expert_ids(), maximum.as_slice());
        let layer_base = 91 * K3_EXPERTS_PER_LAYER;
        assert_eq!(selection.source_indices()[0], layer_base as u32);
        assert_eq!(
            selection.source_indices()[K3_EXPERT_UNION_MAX - 1],
            (layer_base + K3_EXPERT_UNION_MAX - 1) as u32
        );
        assert_eq!(K3_EXPERT_BASE_UNION_BYTES, 1_123_024_896);
        assert_eq!(K3_EXPERT_UNION_MAX_BYTES, 4_492_099_584);
        assert_eq!(K3_SCALE4_BASE_UNION_BYTES, 1_090_519_040);
        assert_eq!(K3_SCALE4_UNION_MAX_BYTES, 4_362_076_160);

        assert!(canonical_union_selection(0, &[0]).is_err());
        assert!(canonical_union_selection(93, &[0]).is_err());
        assert!(canonical_union_selection(1, &[]).is_err());
        assert!(
            canonical_union_selection(
                1,
                &(0_u16..=K3_EXPERT_UNION_MAX as u16).collect::<Vec<_>>(),
            )
            .is_err()
        );
        assert!(canonical_union_selection(1, &[4, 4]).is_err());
        assert!(canonical_union_selection(1, &[5, 4]).is_err());
        assert!(canonical_union_selection(1, &[895, 896]).is_err());
    }

    #[test]
    fn one_expert_union_preserves_raw_bytes_and_reports_exact_lease() {
        use crate::sys::fs::FileExt;

        let root = TestModelRoot::with_experts(17, [7]);
        let source_path = expert_path(&root.0.join("k3-experts"), 17, 7);
        let source = fs::OpenOptions::new()
            .write(true)
            .open(source_path)
            .unwrap();
        let head = [0x19, 0x27, 0x35, 0x43];
        let tail = [0xa1, 0xb2, 0xc3, 0xd4];
        assert_eq!(source.write_at(&head, 0).unwrap(), head.len());
        assert_eq!(
            source
                .write_at(&tail, (K3_EXPERT_SOURCE_BYTES - tail.len()) as u64)
                .unwrap(),
            tail.len()
        );

        let corpus = RawExpertCorpus::open_raw_v1(&root.0).unwrap();
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        let batch = corpus.read_union(&reader, 17, &[7]).unwrap();
        assert_eq!(batch.layer(), 17);
        assert_eq!(batch.expert_ids(), &[7]);
        assert_eq!(batch.buffers().other().len(), K3_EXPERT_SOURCE_BYTES);
        assert_eq!(batch.stats().bytes, K3_EXPERT_SOURCE_BYTES as u64);
        assert_eq!(batch.stats().jobs, 1);
        assert_eq!(&batch.buffers().other()[..head.len()], &head);
        assert_eq!(
            &batch.buffers().other()[K3_EXPERT_SOURCE_BYTES - tail.len()..],
            &tail
        );
    }

    #[test]
    fn one_expert_union_rejects_a_wrong_sized_live_source() {
        let root = TestModelRoot::with_experts(17, [7]);
        fs::OpenOptions::new()
            .write(true)
            .open(expert_path(&root.0.join("k3-experts"), 17, 7))
            .unwrap()
            .set_len(K3_EXPERT_SOURCE_BYTES as u64 - 1)
            .unwrap();
        let corpus = RawExpertCorpus::open_raw_v1(&root.0).unwrap();
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        let error = match corpus.read_union(&reader, 17, &[7]) {
            Ok(_) => panic!("wrong-sized raw-v1 expert was unexpectedly accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 17547264"));
    }

    #[test]
    fn raw_corpus_compiles_all_names_once_and_absent_scale4_fails_closed() {
        let root = TestModelRoot::with_experts(17, []);
        let corpus = RawExpertCorpus::open(&root.0, ExpertStorageLayout::RawV1).unwrap();
        assert_eq!(corpus.source_count(), K3_EXPERT_RAW_FILES);
        let error = RawExpertCorpus::open(&root.0, ExpertStorageLayout::Scale4V2).unwrap_err();
        assert!(error.to_string().contains("scale4 manifest"));
    }

    #[test]
    #[ignore = "manual request-compilation microbenchmark"]
    fn benchmark_decode_request_compilation() {
        use std::hint::black_box;
        use std::time::Instant;

        let root = TestModelRoot::with_experts(17, []);
        let catalog_started = Instant::now();
        let corpus = RawExpertCorpus::open_raw_v1(&root.0).unwrap();
        let catalog_elapsed = catalog_started.elapsed();
        assert_eq!(corpus.source_count(), K3_EXPERT_RAW_FILES);
        let experts: Vec<u16> = (400..416).collect();

        const FAST_ITERATIONS: usize = 200_000;
        let fast_started = Instant::now();
        for _ in 0..FAST_ITERATIONS {
            black_box(decode_source_indices(47, black_box(&experts)).unwrap());
        }
        let fast_elapsed = fast_started.elapsed();

        const PLAN_ITERATIONS: usize = 2_000;
        let plan_started = Instant::now();
        for _ in 0..PLAN_ITERATIONS {
            black_box(
                ExpertBatchPlan::open_raw_cache_default(&root.0, 47, black_box(&experts)).unwrap(),
            );
        }
        let plan_elapsed = plan_started.elapsed();
        let fast_ns = fast_elapsed.as_nanos() / FAST_ITERATIONS as u128;
        let plan_ns = plan_elapsed.as_nanos() / PLAN_ITERATIONS as u128;
        eprintln!(
            "raw-v1 request compilation: catalog_startup={catalog_elapsed:?} integer_selection={fast_ns}ns legacy_plan={plan_ns}ns ratio={:.1}x",
            plan_ns as f64 / fast_ns.max(1) as f64,
        );
    }

    /// Real-corpus exactness gate for `K3_STORAGE_HOMES`: every MoE layer,
    /// both storage layouts, decode and wide-union paths, read once from the
    /// model root alone and once spread over a storage home, compared whole.
    #[test]
    #[ignore = "reads ~60 GB of the installed corpus; set DELTAFIN_TEST_STORAGE_HOME"]
    fn installed_corpus_reads_identical_bytes_through_storage_homes() {
        let Some(home) = std::env::var_os("DELTAFIN_TEST_STORAGE_HOME").map(PathBuf::from) else {
            return;
        };
        let root = crate::sys::fs::canonicalize(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
        let homes = StorageHomes::open(&root, &home_specs(&home)).unwrap().unwrap();
        let reader = Reader::with_arena_capacity(8, 2).unwrap();
        let mut homed_reader = Reader::with_arena_capacity(8, 2).unwrap();
        homed_reader.set_storage_homes(Some(Arc::clone(&homes)));
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut route = |count: usize| -> Vec<u16> {
            let mut ids = std::collections::BTreeSet::new();
            while ids.len() < count {
                ids.insert((next() % K3_EXPERTS_PER_LAYER as u64) as u16);
            }
            ids.into_iter().collect()
        };
        let mut compared = 0_u64;
        for layout in [ExpertStorageLayout::Scale4V2, ExpertStorageLayout::RawV1] {
            let corpus = RawExpertCorpus::open(&root, layout).unwrap();
            for layer in K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST {
                let decode = route(K3_EXPERT_TOP_K);
                let expected = corpus.read_decode(&reader, layer, &decode).unwrap();
                let actual = corpus.read_decode(&homed_reader, layer, &decode).unwrap();
                assert!(
                    expected.buffers().other() == actual.buffers().other(),
                    "{layout:?} decode layer {layer} differs through storage homes"
                );
                compared += actual.buffers().other().len() as u64;
                drop((expected, actual));
                if layer % 23 == 1 {
                    let union = route(40);
                    let expected = corpus.read_union(&reader, layer, &union).unwrap();
                    let actual = corpus.read_union(&homed_reader, layer, &union).unwrap();
                    assert!(
                        expected.buffers().other() == actual.buffers().other(),
                        "{layout:?} union layer {layer} differs through storage homes"
                    );
                    compared += actual.buffers().other().len() as u64;
                }
            }
        }
        let stats = homes.stats();
        eprintln!(
            "served GB per drive {:?}",
            stats
                .iter()
                .map(|device| device.served_bytes as f64 / 1e9)
                .collect::<Vec<_>>()
        );
        assert!(stats.iter().all(|device| device.served_bytes > 0));
        eprintln!("compared {:.1} GB byte-for-byte", compared as f64 / 1e9);
    }

    #[test]
    #[ignore = "reads one installed 281 MiB raw-v1 expert batch"]
    fn installed_decode_fast_path_is_byte_exact_and_storage_ordered() {
        use crate::sys::fs::FileExt;

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let experts: Vec<u16> = (0..K3_EXPERT_TOP_K as u16).collect();
        if experts
            .iter()
            .any(|&expert| !expert_path(&root.join("k3-experts"), 1, expert).is_file())
        {
            return;
        }
        let corpus = RawExpertCorpus::open_raw_v1(&root).unwrap();
        let reader = Reader::with_arena_capacity(6, 1).unwrap();
        let batch = corpus.read_decode(&reader, 1, &experts).unwrap();
        assert_eq!(batch.expert_ids().as_slice(), experts);
        assert_eq!(
            batch.buffers().other().len(),
            K3_EXPERT_TOP_K * K3_EXPERT_SOURCE_BYTES
        );
        assert_eq!(
            batch.stats().bytes,
            (K3_EXPERT_TOP_K * K3_EXPERT_SOURCE_BYTES) as u64
        );
        for (slot, &expert) in experts.iter().enumerate() {
            let file = fs::File::open(expert_path(&root.join("k3-experts"), 1, expert)).unwrap();
            for source_offset in [0, K3_EXPERT_SOURCE_BYTES - 32] {
                let mut expected = [0_u8; 32];
                assert_eq!(
                    file.read_at(&mut expected, source_offset as u64).unwrap(),
                    expected.len()
                );
                let batch_offset = slot * K3_EXPERT_SOURCE_BYTES + source_offset;
                assert_eq!(
                    &batch.buffers().other()[batch_offset..batch_offset + expected.len()],
                    &expected
                );
            }
        }
    }

    #[test]
    fn installed_route_builds_one_direct_native_plan_when_cache_is_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let candidate = root.join("k3-experts/L17-E24.bin");
        if !candidate.is_file() {
            return;
        }
        let plan = ExpertBatchPlan::open_raw_cache_default(&root, 17, &[24, 24]).unwrap();
        assert_eq!(plan.layer(), 17);
        assert_eq!(plan.expert_ids(), &[24]);
        assert_eq!(plan.descriptors().len(), 1);
        assert_eq!(plan.descriptors()[0][0].rows, 3_072);
        assert_eq!(plan.descriptors()[0][1].columns, 3_072);
        assert_eq!(plan.descriptors()[0][2].packed_offset, 11_698_176);
    }

    #[test]
    fn expert_slots_match_route_arenas_ascending_canonical_order() {
        assert_eq!(
            canonical_expert_ids(&[700, 4, 511, 4, 0, 700]).unwrap(),
            [0, 4, 511, 700]
        );
    }

    #[test]
    fn decode_batch_is_one_canonical_extent_and_handle_per_expert() {
        let root = TestModelRoot::with_experts(17, 0_u16..16);
        let routed = [15, 3, 12, 0, 14, 2, 11, 1, 13, 5, 10, 4, 9, 6, 8, 7];
        let plan = ExpertBatchPlan::open_raw_cache_default(&root.0, 17, &routed).unwrap();

        assert_eq!(plan.expert_ids(), &(0_u16..16).collect::<Vec<_>>());
        assert_eq!(plan.read_plan().source_count(), 16);
        assert_eq!(plan.read_plan().persistent_source_count(), 0);
        assert_eq!(plan.read_plan().jobs(), 16);
        assert_eq!(
            plan.read_plan().logical_bytes(),
            (16 * K3_EXPERT_SOURCE_BYTES) as u64
        );
        assert_eq!(
            plan.read_plan().buffer_len(K3_EXPERT_BUFFER_KIND),
            16 * K3_EXPERT_SOURCE_BYTES
        );
        assert_eq!(plan.read_plan().buffer_len(BufferKind::Quantized), 0);
        assert_eq!(plan.read_plan().buffer_len(BufferKind::Scales), 0);

        let second = &plan.descriptors()[1];
        assert_eq!(second[0].packed_offset, K3_EXPERT_SOURCE_BYTES as u64);
        assert_eq!(
            second[2].scale_offset,
            (K3_EXPERT_SOURCE_BYTES as u64) + SOURCE_SCALE_OFFSETS[2]
        );
    }

    #[test]
    fn rejects_invalid_layers_and_experts_before_opening_files() {
        let root = Path::new("/does-not-exist");
        assert!(ExpertBatchPlan::open_raw_cache_default(root, 0, &[1]).is_err());
        assert!(ExpertBatchPlan::open_raw_cache_default(root, 93, &[1]).is_err());
        assert!(ExpertBatchPlan::open_raw_cache_default(root, 1, &[896]).is_err());
        assert!(ExpertBatchPlan::open_raw_cache_default(root, 1, &[]).is_err());
        assert!(
            ExpertBatchPlan::open_raw_cache(root, 1, &[1], 16 * 1024 * 1024)
                .unwrap_err()
                .to_string()
                .contains("one job")
        );
    }

    #[test]
    fn scale4_selection_requires_an_activated_manifest_without_raw_fallback() {
        let error = ExpertBatchPlan::open(
            Path::new("/does-not-exist"),
            17,
            &[24],
            DEFAULT_EXPERT_CHUNK_BYTES,
            ExpertStorageLayout::Scale4V2,
        )
        .unwrap_err();
        assert!(error.to_string().contains("scale4 manifest"));
    }

    #[test]
    fn scale4_gather_is_exact_expert_major_without_whole_source_rereads() {
        let root = TestModelRoot::with_experts(1, []);
        let (manifest, raw, record) = write_synthetic_scale4(&root.0, None);
        let plan = ExpertBatchPlan::open_scale4_manifest(&root.0, &manifest, 1, &[0], 0).unwrap();
        assert_eq!(plan.expert_ids(), &[0]);
        assert_eq!(plan.read_plan().source_count(), 2);
        assert_eq!(plan.read_plan().persistent_source_count(), 0);
        // Three packed-plane reads plus one preadv that scatters the physically
        // contiguous header|w1s4|w2s4|w3s4 record into the interleaved native
        // blob. Post-read validation authenticates that canonical record.
        assert_eq!(plan.read_plan().jobs(), 4);
        assert_eq!(
            plan.read_plan().logical_bytes(),
            K3_SCALE4_BLOB_BYTES as u64
        );
        assert_eq!(
            plan.read_plan().buffer_len(K3_EXPERT_BUFFER_KIND),
            K3_SCALE4_BLOB_BYTES
        );

        let entries = plan.scale4_entries.clone().unwrap();
        let reader = Reader::with_arena_capacity(4, 1).unwrap();
        let (buffers, stats) = reader.read(plan.read_plan()).unwrap();
        validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &buffers,
            Some(&entries),
            None,
        )
        .unwrap();
        assert_eq!(stats.bytes, K3_SCALE4_BLOB_BYTES as u64);
        assert_eq!(stats.jobs, 4);
        let blob = buffers.other();
        assert_eq!(&blob[..SCALE4_HEADER_BYTES], &record[..SCALE4_HEADER_BYTES]);
        for matrix in 0..K3_EXPERT_COMPONENTS {
            let (raw_offset, raw_length) = expert_scale4::RAW_LAYOUT[matrix * 2];
            let (packed_offset, packed_length) = COMPACT_LAYOUT[matrix * 2];
            assert_eq!(raw_length, packed_length as usize);
            assert_eq!(
                &blob[packed_offset as usize..packed_offset as usize + raw_length],
                &raw[raw_offset as usize..raw_offset as usize + raw_length]
            );
            let (side_offset, side_length) = SIDECAR_LAYOUT[matrix];
            let (scale_offset, scale_length) = COMPACT_LAYOUT[matrix * 2 + 1];
            assert_eq!(side_length, scale_length);
            assert_eq!(
                &blob[scale_offset as usize..scale_offset as usize + scale_length as usize],
                &record[side_offset as usize..side_offset as usize + side_length as usize]
            );
        }
        assert_eq!(
            plan.descriptors()[0][0].packed_offset,
            COMPACT_LAYOUT[0].0 as u64
        );
        assert_eq!(
            plan.descriptors()[0][2].scale_offset,
            COMPACT_LAYOUT[5].0 as u64
        );
    }

    #[test]
    fn scale4_record_digest_cache_hits_and_invalidates_on_sidecar_replacement() {
        let root = TestModelRoot::with_experts(1, []);
        let (manifest, _, record) = write_synthetic_scale4(&root.0, None);
        let plan = ExpertBatchPlan::open_scale4_manifest(&root.0, &manifest, 1, &[0], 0).unwrap();
        let identity = plan.scale4_identity.unwrap();
        let entries = plan.scale4_entries.clone().unwrap();
        let cache = Arc::new(Scale4ValidationCache::default());
        cache.admit_corpus_identity(1, identity).unwrap();
        cache.prepare(1, identity);
        let validation = Scale4BatchValidation {
            cache: Arc::clone(&cache),
            layer: 1,
            identity,
            reader_verified: Box::new([]),
        };
        let reader = Reader::with_arena_capacity(4, 1).unwrap();
        let (buffers, _) = reader.read(plan.read_plan()).unwrap();

        validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &buffers,
            Some(&entries),
            Some(&validation),
        )
        .unwrap();
        assert_eq!(cache.record_hashes(), 1);
        assert!(!cache.needs_hash(1, 0, identity));

        validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &buffers,
            Some(&entries),
            Some(&validation),
        )
        .unwrap();
        assert_eq!(cache.record_hashes(), 1);
        drop(buffers);

        let mut corrupted = record;
        corrupted[SCALE4_HEADER_BYTES + 10] ^= 0xff;
        let sidecar = root.0.join("k3-experts-scale4/L1.sc4");
        let replacement = root.0.join("k3-experts-scale4/L1.replacement.sc4");
        fs::write(&replacement, corrupted).unwrap();
        fs::rename(replacement, sidecar).unwrap();

        let changed_plan =
            ExpertBatchPlan::open_scale4_manifest(&root.0, &manifest, 1, &[0], 0).unwrap();
        let changed_identity = changed_plan.scale4_identity.unwrap();
        assert_ne!(changed_identity, identity);
        let pin_error = cache
            .admit_corpus_identity(1, changed_identity)
            .unwrap_err();
        assert!(
            pin_error
                .to_string()
                .contains("changed after session pinning")
        );
        assert_eq!(cache.reusable_corpus_identity(1), Some(identity));
        // Exercise the identity-keyed digest cache independently of the
        // immutable corpus pin; production corpus plans never prepare a
        // replacement identity after admission fails.
        cache.prepare(1, changed_identity);
        assert!(cache.needs_hash(1, 0, changed_identity));
        let changed_validation = Scale4BatchValidation {
            cache: Arc::clone(&cache),
            layer: 1,
            identity: changed_identity,
            reader_verified: Box::new([]),
        };
        let changed_entries = changed_plan.scale4_entries.clone().unwrap();
        let (changed_buffers, _) = reader.read(changed_plan.read_plan()).unwrap();
        let error = validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &changed_buffers,
            Some(&changed_entries),
            Some(&changed_validation),
        )
        .unwrap_err();
        assert!(error.to_string().contains("record disagrees"));
        assert_eq!(cache.record_hashes(), 2);
        assert!(cache.needs_hash(1, 0, changed_identity));
    }

    #[test]
    fn scale4_corpus_reuses_authenticated_identity_without_live_recapture() {
        let root = TestModelRoot::with_experts(1, []);
        let (manifest, _, _) = write_synthetic_scale4(&root.0, None);
        let cache = Arc::new(Scale4ValidationCache::default());
        let first = ExpertBatchPlan::open_scale4_manifest_for_corpus(
            &root.0,
            &manifest,
            &cache,
            1,
            &[0],
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let identity = first.scale4_identity.unwrap();
        assert_eq!(&*first.scale4_reader_verified, &[0]);
        assert_eq!(cache.reusable_corpus_identity(1), None);
        let entries = first.scale4_entries.clone().unwrap();
        let validation = Scale4BatchValidation {
            cache: Arc::clone(&cache),
            layer: 1,
            identity,
            reader_verified: first.scale4_reader_verified.clone(),
        };
        let reader = Reader::with_arena_capacity(4, 1).unwrap();
        let (buffers, _) = reader.read(first.read_plan()).unwrap();
        validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &buffers,
            Some(&entries),
            Some(&validation),
        )
        .unwrap();
        drop(buffers);
        assert_eq!(cache.record_hashes(), 1);
        assert_eq!(cache.reusable_corpus_identity(1), Some(identity));

        let sidecar = root.0.join("k3-experts-scale4/L1.sc4");
        let displaced = root.0.join("k3-experts-scale4/L1.displaced.sc4");
        fs::rename(&sidecar, displaced).unwrap();

        // The corpus path uses the authenticated session identity and therefore
        // constructs this second plan without opening or statting the missing
        // pathname on the caller thread.
        let reused = ExpertBatchPlan::open_scale4_manifest_for_corpus(
            &root.0,
            &manifest,
            &cache,
            1,
            &[0],
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(reused.scale4_identity, Some(identity));
        assert!(reused.scale4_reader_verified.is_empty());

        // Direct plans deliberately retain live identity capture for tools and
        // tests that are not backed by a session corpus.
        let direct_error =
            ExpertBatchPlan::open_scale4_manifest(&root.0, &manifest, 1, &[0], 0).unwrap_err();
        assert!(
            direct_error
                .to_string()
                .contains("open deferred identity source")
        );

        let read_error = match reader.read(reused.read_plan()) {
            Ok(_) => panic!("identity-reusing corpus read accepted a missing sidecar"),
            Err(error) => error,
        };
        assert!(
            read_error
                .to_string()
                .contains("open deferred non-symlink source")
        );
    }

    fn home_specs(home: &Path) -> Vec<crate::storage_homes::StorageHomeSpec> {
        crate::storage_homes::parse_storage_homes(&home.display().to_string()).unwrap()
    }

    /// A storage home beside the model root holding copies of its experts.
    fn mirror_home(root: &Path) -> PathBuf {
        let home = root.join("mirror-home");
        let experts = home.join("k3-experts");
        fs::create_dir_all(&experts).unwrap();
        for entry in fs::read_dir(root.join("k3-experts")).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), experts.join(entry.file_name())).unwrap();
        }
        home
    }

    fn homed_reader(root: &Path, home: &Path, workers: usize, slots: usize) -> (Reader, Arc<StorageHomes>) {
        let homes = StorageHomes::open(root, &home_specs(home)).unwrap().unwrap();
        let mut reader = Reader::with_arena_capacity(workers, slots).unwrap();
        reader.set_storage_homes(Some(Arc::clone(&homes)));
        (reader, homes)
    }

    fn write_distinct_raw_experts(root: &Path, layer: u32, experts: std::ops::Range<u16>) {
        use crate::sys::fs::FileExt;
        let cache = root.join("k3-experts");
        for expert in experts {
            let file = fs::OpenOptions::new()
                .write(true)
                .open(expert_path(&cache, layer, expert))
                .unwrap();
            let marker = [expert as u8 ^ 0x5a, 0x11, expert as u8, 0xee];
            file.write_at(&marker, 0).unwrap();
            file.write_at(&marker, (K3_EXPERT_SOURCE_BYTES - marker.len()) as u64)
                .unwrap();
        }
    }

    fn raw_slots(buffers: &LayerBuffers, experts: usize) -> Vec<Vec<u8>> {
        (0..experts)
            .map(|slot| {
                let span = &buffers.other()[slot * K3_EXPERT_SOURCE_BYTES..][..K3_EXPERT_SOURCE_BYTES];
                [&span[..4], &span[K3_EXPERT_SOURCE_BYTES - 4..]].concat()
            })
            .collect()
    }

    #[test]
    fn raw_reads_spread_over_storage_homes_return_identical_bytes() {
        let root = TestModelRoot::with_experts(5, 0..40);
        write_distinct_raw_experts(&root.0, 5, 0..40);
        let home = mirror_home(&root.0);
        let corpus = RawExpertCorpus::open_raw_v1(&root.0).unwrap();
        let reader = Reader::with_arena_capacity(4, 2).unwrap();
        let decode_ids: Vec<u16> = (0..16).collect();
        let union_ids: Vec<u16> = (0..40).collect();
        let (expected_decode, expected_union) = {
            let decode = corpus.read_decode(&reader, 5, &decode_ids).unwrap();
            let decode = raw_slots(decode.buffers(), 16);
            let union = corpus.read_union(&reader, 5, &union_ids).unwrap();
            (decode, raw_slots(union.buffers(), 40))
        };

        let (homed, homes) = homed_reader(&root.0, &home, 4, 2);
        // Which drive serves a read depends on both drives' load at that
        // instant, which is thread timing. Pin it instead: with one drive held
        // busy every read takes the other, so each drive demonstrably serves
        // bytes and every read must come back identical from either.
        for busy_device in [0_u8, 1] {
            let _busy = homes.begin(busy_device, 1 << 40);
            let decode = corpus.read_decode(&homed, 5, &decode_ids).unwrap();
            assert_eq!(raw_slots(decode.buffers(), 16), expected_decode);
            drop(decode);
            // Narrow (inline catalog) and wide (deferred plan) unions both route.
            let narrow = corpus.read_union(&homed, 5, &decode_ids[..9]).unwrap();
            assert_eq!(raw_slots(narrow.buffers(), 9), expected_decode[..9]);
            drop(narrow);
            let union = corpus.read_union(&homed, 5, &union_ids).unwrap();
            assert_eq!(raw_slots(union.buffers(), 40), expected_union);
            drop(union);
        }
        let stats = homes.stats();
        assert!(stats[0].served_bytes > 0 && stats[1].served_bytes > 0, "{stats:?}");
        // Every read finished, so nothing stays charged in flight. Which
        // drive would win an idle tie is deliberately not asserted: more than
        // a window of traffic has flowed, so both drives now carry a measured
        // rate and the faster measurement rightly wins -- a timing outcome.
        assert_eq!(homes.outstanding_bytes(), vec![0, 0]);
    }

    #[test]
    fn storage_home_reads_really_come_from_the_extra_drive() {
        use crate::sys::fs::FileExt;
        let root = TestModelRoot::with_experts(6, 0..16);
        let home = mirror_home(&root.0);
        // Deliberately diverge every copy (tests only: a real home is
        // byte-identical) so the bytes reveal which drive served each slot.
        for expert in 0..16_u16 {
            let file = fs::OpenOptions::new()
                .write(true)
                .open(expert_path(&home.join("k3-experts"), 6, expert))
                .unwrap();
            file.write_at(&[0xff], 0).unwrap();
        }
        let corpus = RawExpertCorpus::open_raw_v1(&root.0).unwrap();
        let (reader, homes) = homed_reader(&root.0, &home, 4, 1);
        let ids: Vec<u16> = (0..16).collect();
        // Where a read goes is decided by drive load, so a test that only
        // runs the reads and counts is at the mercy of thread timing. Hold one
        // drive busy to pin the choice: every slot then reveals, by its marker
        // byte, which drive really served it.
        let from_copy = |busy_device: u8| {
            let _busy = homes.begin(busy_device, 1 << 40);
            let batch = corpus.read_decode(&reader, 6, &ids).unwrap();
            (0..16)
                .filter(|slot| batch.buffers().other()[slot * K3_EXPERT_SOURCE_BYTES] == 0xff)
                .count()
        };
        assert_eq!(from_copy(0), 16, "with the primary busy every read takes the copy");
        assert_eq!(from_copy(1), 0, "with the copy busy every read takes the primary");
    }

    #[test]
    fn a_failing_storage_home_falls_back_to_another_copy() {
        let root = TestModelRoot::with_experts(7, 0..16);
        write_distinct_raw_experts(&root.0, 7, 0..16);
        let home = mirror_home(&root.0);
        let corpus = RawExpertCorpus::open_raw_v1(&root.0).unwrap();
        let reader = Reader::with_arena_capacity(4, 1).unwrap();
        let ids: Vec<u16> = (0..16).collect();
        let expected = raw_slots(corpus.read_decode(&reader, 7, &ids).unwrap().buffers(), 16);
        let (homed, homes) = homed_reader(&root.0, &home, 4, 1);
        // The drive disappears after startup admitted it.
        fs::remove_dir_all(&home).unwrap();
        // Whether any read picks the vanished drive is a load (timing)
        // outcome, so make the primary look busy: the first choice for every
        // job is then the dead drive, and each must fall back to the copy.
        let batch = {
            let _primary_busy = homes.begin(0, 1 << 40);
            corpus.read_decode(&homed, 7, &ids).unwrap()
        };
        assert_eq!(raw_slots(batch.buffers(), 16), expected);
        drop(batch);
        let stats = homes.stats();
        assert!(stats[1].failures > 0 && stats[1].quarantined, "{stats:?}");
        // While quarantined it receives nothing more.
        let before = homes.stats()[1].failures;
        let again = corpus.read_decode(&homed, 7, &ids).unwrap();
        assert_eq!(raw_slots(again.buffers(), 16), expected);
        assert_eq!(homes.stats()[1].failures, before);
    }

    #[test]
    fn scale4_reads_spread_over_storage_homes_return_identical_bytes() {
        let root = TestModelRoot::with_experts(1, []);
        let _ = write_synthetic_scale4_experts(&root.0, None, 16);
        let home = mirror_home(&root.0);
        let names: Vec<_> = (0..16).map(|expert| format!("L1-E{expert}.bin")).collect();
        let corpus = RawExpertCorpus {
            storage: ExpertCorpusStorage::Scale4 {
                manifest: Scale4Manifest::load_for_raw_names(
                    root.0.join("k3-experts-scale4"),
                    &names,
                )
                .unwrap(),
                validation: Arc::new(Scale4ValidationCache::default()),
            },
            model_root: root.0.clone(),
            lazy: None,
            stream_cache_policy: CachePolicy::Streaming,
        };
        let reader = Reader::with_arena_capacity(4, 1).unwrap();
        let ids: Vec<u16> = (0..16).collect();
        let expected = corpus
            .read_decode(&reader, 1, &ids)
            .unwrap()
            .buffers()
            .other()
            .to_vec();
        let (homed, homes) = homed_reader(&root.0, &home, 4, 1);
        // The copy must demonstrably serve bytes; pin the choice with load
        // rather than hoping thread timing happens to spread the reads.
        let _primary_busy = homes.begin(0, 1 << 40);
        let batch = corpus.read_decode(&homed, 1, &ids).unwrap();
        assert!(batch.buffers().other() == &expected[..]);
        drop(batch);
        let union = corpus.read_union(&homed, 1, &ids[..5]).unwrap();
        assert!(union.buffers().other() == &expected[..5 * K3_SCALE4_BLOB_BYTES]);
        assert!(homes.stats()[1].served_bytes > 0);
    }

    #[test]
    fn scale4_corpus_mixed_hit_miss_verifies_only_the_missing_record() {
        let root = TestModelRoot::with_experts(1, []);
        let (manifest, _, _) = write_synthetic_scale4_experts(&root.0, None, 2);
        let cache = Arc::new(Scale4ValidationCache::default());
        let reader = Reader::with_arena_capacity(4, 1).unwrap();

        let first = ExpertBatchPlan::open_scale4_manifest_for_corpus(
            &root.0,
            &manifest,
            &cache,
            1,
            &[0],
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(&*first.scale4_reader_verified, &[0]);
        let identity = first.scale4_identity.unwrap();
        let first_entries = first.scale4_entries.clone().unwrap();
        let first_validation = Scale4BatchValidation {
            cache: Arc::clone(&cache),
            layer: 1,
            identity,
            reader_verified: first.scale4_reader_verified.clone(),
        };
        let (first_buffers, _) = reader.read(first.read_plan()).unwrap();
        validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &first_buffers,
            Some(&first_entries),
            Some(&first_validation),
        )
        .unwrap();
        drop(first_buffers);
        assert_eq!(cache.record_hashes(), 1);

        let mixed = ExpertBatchPlan::open_scale4_manifest_for_corpus(
            &root.0,
            &manifest,
            &cache,
            1,
            &[0, 1],
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(mixed.scale4_identity, Some(identity));
        assert_eq!(&*mixed.scale4_reader_verified, &[1]);
        let mixed_entries = mixed.scale4_entries.clone().unwrap();
        let mixed_validation = Scale4BatchValidation {
            cache: Arc::clone(&cache),
            layer: 1,
            identity,
            reader_verified: mixed.scale4_reader_verified.clone(),
        };
        let (mixed_buffers, _) = reader.read(mixed.read_plan()).unwrap();
        validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &mixed_buffers,
            Some(&mixed_entries),
            Some(&mixed_validation),
        )
        .unwrap();
        assert_eq!(cache.record_hashes(), 2);
        assert!(!cache.needs_hash(1, 0, identity));
        assert!(!cache.needs_hash(1, 1, identity));
    }

    #[test]
    fn scale4_gather_rejects_record_corruption_before_provider_publication() {
        use crate::sys::fs::FileExt;

        let root = TestModelRoot::with_experts(1, []);
        let (manifest, _, _) = write_synthetic_scale4(&root.0, None);
        let path = root.0.join("k3-experts-scale4/L1.sc4");
        let file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.write_at(&[0xff], (SCALE4_HEADER_BYTES + 10) as u64)
            .unwrap();
        file.sync_all().unwrap();
        let plan = ExpertBatchPlan::open_scale4_manifest(&root.0, &manifest, 1, &[0], 0).unwrap();
        let entries = plan.scale4_entries.clone().unwrap();
        let reader = Reader::with_arena_capacity(4, 1).unwrap();
        let (buffers, _) = reader.read(plan.read_plan()).unwrap();
        let error = validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &buffers,
            Some(&entries),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("record disagrees"));
    }

    #[test]
    fn scale4_corpus_miss_rejects_record_corruption_inside_reader_worker() {
        use crate::sys::fs::FileExt;

        let root = TestModelRoot::with_experts(1, []);
        let (manifest, _, _) = write_synthetic_scale4(&root.0, None);
        let path = root.0.join("k3-experts-scale4/L1.sc4");
        let file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.write_at(&[0xff], (SCALE4_HEADER_BYTES + 10) as u64)
            .unwrap();
        file.sync_all().unwrap();

        let cache = Scale4ValidationCache::default();
        let plan = ExpertBatchPlan::open_scale4_manifest_for_corpus(
            &root.0,
            &manifest,
            &cache,
            1,
            &[0],
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(&*plan.scale4_reader_verified, &[0]);
        let identity = plan.scale4_identity.unwrap();
        let error = match Reader::with_arena_capacity(4, 1)
            .unwrap()
            .read(plan.read_plan())
        {
            Ok(_) => panic!("corrupted scale4 cache miss escaped reader verification"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("failed SHA-256 verification"));
        assert!(cache.needs_hash(1, 0, identity));
        assert_eq!(cache.reusable_corpus_identity(1), None);
    }

    #[test]
    fn scale4_gather_retains_raw_v1_live_descriptor_length_contract() {
        let root = TestModelRoot::with_experts(1, []);
        let (manifest, _, _) = write_synthetic_scale4(&root.0, None);
        let raw = expert_path(&root.0.join("k3-experts"), 1, 0);
        fs::OpenOptions::new()
            .write(true)
            .open(&raw)
            .unwrap()
            .set_len((K3_EXPERT_SOURCE_BYTES - 1) as u64)
            .unwrap();
        let plan = ExpertBatchPlan::open_scale4_manifest(&root.0, &manifest, 1, &[0], 0).unwrap();
        let reader = Reader::with_arena_capacity(4, 1).unwrap();
        let error = match reader.read(plan.read_plan()) {
            Ok(_) => panic!("wrong-sized raw expert was accepted by scale4 gather"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length"));
    }

    #[test]
    fn scale4_gather_rejects_a_manifest_disagreeing_header_after_digest_checks() {
        let root = TestModelRoot::with_experts(1, []);
        let wrong_header_source = [0xa5; 32];
        let (manifest, _, _) = write_synthetic_scale4(&root.0, Some(wrong_header_source));
        let plan = ExpertBatchPlan::open_scale4_manifest(&root.0, &manifest, 1, &[0], 0).unwrap();
        let entries = plan.scale4_entries.clone().unwrap();
        let reader = Reader::with_arena_capacity(4, 1).unwrap();
        let (buffers, _) = reader.read(plan.read_plan()).unwrap();
        let error = validate_completed_layout(
            ExpertStorageLayout::Scale4V2,
            &buffers,
            Some(&entries),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("header disagrees"));
    }

    #[test]
    fn raw_cache_rejects_a_symlink_even_when_its_target_has_the_right_size() {
        use crate::sys::fs::symlink;

        let root = TestModelRoot::with_experts(17, [0]);
        let cache = root.0.join("k3-experts");
        symlink(expert_path(&cache, 17, 0), expert_path(&cache, 17, 1)).unwrap();
        let plan = ExpertBatchPlan::open_raw_cache_default(&root.0, 17, &[1]).unwrap();
        let reader = crate::storage::Reader::new(1).unwrap();
        let error = match reader.read(plan.read_plan()) {
            Ok(_) => panic!("raw expert symlink was unexpectedly followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
    }
}
