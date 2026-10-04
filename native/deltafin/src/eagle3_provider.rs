//! EAGLE-3.1 hidden-state chain drafter
//! (`lightseekorg/kimi-k3-eagle3.1-mla`): pinned checkpoint admission, Rust
//! ownership of its proposal-only C ABI, and the engine-facing
//! [`DraftBackend`].
//!
//! The drafter reads K3's within-block AttnRes stream after layers 1, 45 and
//! 89, which the target sequence captures on the device, so no activation
//! crosses into Rust. Its token embedding is byte-identical to K3's: the
//! engine's exact BF16 table supplies every input row and no second copy is
//! resident. Everything it proposes is untrusted; full K3 verifies each draft
//! before anything is emitted.

use std::ffi::c_char;
use std::fs::File;
use std::io::Read;
use std::mem::size_of;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sha2::{Digest as _, Sha256};

use crate::dspark_checkpoint::{FileIdentity, digest_from_hex, open_regular, strict_json};
use crate::dspark_provider::{
    BF16, ReadOnlyMap, ResourceV1, TensorV1, ffi_status, hex_digest, releasable_backend_failure,
    release, resource,
};
use crate::dspark_runtime::{BackendFailure, BackendProposal, DraftBackend, ModelIdentity};
use crate::embedding::{EmbeddingArena, EmbeddingTable};
use crate::error::{DeltafinError, Result};
use crate::inventory::PINNED_INVENTORY_SHA256;
use crate::packfile::digest_open_file;
use crate::platform::Device;
use crate::provider::{NativeProviderSession, ProviderTensor, SessionInner};

pub const DIRECTORY: &str = "k3-draft-eagle3";
pub const OFFICIAL_MODEL_ID: &str = "lightseekorg/kimi-k3-eagle3.1-mla";
pub const CHECKPOINT_BASENAME: &str = "model.safetensors";
pub const CONFIG_BASENAME: &str = "config.json";
pub const CHECKPOINT_BYTES: u64 = 6_031_253_592;
pub const HEADER_BYTES: u64 = 2_128;
pub const CONFIG_SHA256: &str = "f57ad7ca2e5023bf01fba58c96ac5cb6842c43111be3e1a964b5a4080b85ac8d";
pub const WEIGHTS_SHA256: &str = "af999f57eea206c5193feee3463034ce63db62ceba93ba87cc6de2c4201b595f";
/// BF16 bytes the provider owns: every tensor except the shared embedding.
pub const OWNED_BYTES: u64 = 3_682_441_216;
pub const MAXIMUM_DRAFTS: u8 = 7;
/// One committed position: the 512-wide latent and the 64-wide rotary key.
pub const CACHE_BYTES_PER_POSITION: u64 = (512 + 64) * 2;
/// Context the drafter tracks before it steps aside: its training length
/// was 46,000 tokens and its cache costs 1,152 bytes a position.
pub const MAXIMUM_CONTEXT: usize = 131_072;
/// Cache rows past the committed context: one proposal chain.
pub const CHAIN_HEADROOM: usize = 8;

const HIDDEN: usize = 7_168;
const VOCABULARY: u32 = 163_840;
const TENSOR_COUNT: usize = 19;
const ABI_VERSION: u32 = 1;
const ERROR_CAPACITY: usize = 2_048;
const SYNTHETIC: u32 = 1;
const STEP_BEGIN: u32 = 1;
const STEP_CONTINUE: u32 = 2;
const STEP_END: u32 = 3;
/// Rows one advance carries at most: a whole prefill chunk.
const MAXIMUM_ADVANCE_ROWS: usize = 64;
const EMBEDDING_NAME: &str = "embed_tokens.weight";
const EMBEDDING_SHAPE: [u64; 2] = [163_840, 7_168];
const MAXIMUM_CONFIG_BYTES: u64 = 64 * 1024;

/// The provider's fixed roster: (checkpoint name, ABI slot, shape, rank).
const ROSTER: [(&str, u32, [u64; 2], u32); TENSOR_COUNT] = [
    ("fc.weight", 1, [7_168, 21_504], 2),
    ("fc_norm.0.weight", 2, [7_168, 0], 1),
    ("fc_norm.1.weight", 3, [7_168, 0], 1),
    ("fc_norm.2.weight", 4, [7_168, 0], 1),
    ("layers.0.hidden_norm.weight", 5, [7_168, 0], 1),
    ("layers.0.input_layernorm.weight", 6, [7_168, 0], 1),
    ("layers.0.self_attn.q_a_proj.weight", 7, [1_536, 14_336], 2),
    ("layers.0.self_attn.q_a_layernorm.weight", 8, [1_536, 0], 1),
    ("layers.0.self_attn.q_b_proj.weight", 9, [12_288, 1_536], 2),
    (
        "layers.0.self_attn.kv_a_proj_with_mqa.weight",
        10,
        [576, 14_336],
        2,
    ),
    ("layers.0.self_attn.kv_a_layernorm.weight", 11, [512, 0], 1),
    ("layers.0.self_attn.kv_b_proj.weight", 12, [16_384, 512], 2),
    ("layers.0.self_attn.o_proj.weight", 13, [7_168, 8_192], 2),
    (
        "layers.0.post_attention_layernorm.weight",
        14,
        [7_168, 0],
        1,
    ),
    ("layers.0.mlp.gate_proj.weight", 15, [18_432, 7_168], 2),
    ("layers.0.mlp.up_proj.weight", 16, [18_432, 7_168], 2),
    ("layers.0.mlp.down_proj.weight", 17, [7_168, 18_432], 2),
    ("norm.weight", 18, [7_168, 0], 1),
    ("lm_head.weight", 19, [163_840, 7_168], 2),
];

#[derive(Debug, Clone)]
pub struct Eagle3Tensor {
    pub name: &'static str,
    pub slot: u32,
    pub rank: u32,
    pub shape: [u64; 2],
    /// Absolute byte range inside the admitted checkpoint descriptor.
    pub bytes: Range<u64>,
}

/// The pinned public checkpoint, admitted by exact config digest, exact
/// safetensors roster and, at bind, the full-file SHA-256.
#[derive(Debug)]
pub struct Eagle3Checkpoint {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    tensors: Box<[Eagle3Tensor]>,
}

impl Eagle3Checkpoint {
    pub fn open(directory: &Path) -> Result<Self> {
        let config_path = directory.join(CONFIG_BASENAME);
        let (config, config_identity) = open_regular(&config_path, None)?;
        let mut raw_config = Vec::new();
        (&config)
            .take(MAXIMUM_CONFIG_BYTES + 1)
            .read_to_end(&mut raw_config)
            .map_err(|error| {
                DeltafinError::new(format!("read {}: {error}", config_path.display()))
            })?;
        config_identity.validate(&config, &config_path)?;
        let config_digest: [u8; 32] = Sha256::digest(&raw_config).into();
        if raw_config.len() as u64 > MAXIMUM_CONFIG_BYTES
            || config_digest != digest_from_hex(CONFIG_SHA256)?
        {
            return Err(DeltafinError::new(format!(
                "EAGLE-3 config differs from the pinned {OFFICIAL_MODEL_ID} release"
            )));
        }

        let path = directory.join(CHECKPOINT_BASENAME);
        let (mut file, identity) = open_regular(&path, Some(CHECKPOINT_BYTES))?;
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix).map_err(|error| {
            DeltafinError::new(format!("read EAGLE-3 safetensors prefix: {error}"))
        })?;
        if u64::from_le_bytes(prefix) != HEADER_BYTES {
            return Err(DeltafinError::new(format!(
                "EAGLE-3 safetensors header is not the pinned {HEADER_BYTES} bytes"
            )));
        }
        let mut raw_header = vec![0_u8; HEADER_BYTES as usize];
        file.read_exact(&mut raw_header).map_err(|error| {
            DeltafinError::new(format!("read EAGLE-3 safetensors header: {error}"))
        })?;
        identity.validate(&file, &path)?;
        let tensors = admit_header(&raw_header)?;
        Ok(Self {
            path,
            file,
            identity,
            tensors,
        })
    }

    pub fn tensors(&self) -> &[Eagle3Tensor] {
        &self.tensors
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }

    pub(crate) fn validate_live_identity(&self) -> Result<()> {
        self.identity.validate(&self.file, &self.path)
    }

    pub fn verify_full_digest(&self) -> Result<()> {
        let digest = digest_open_file(&self.file, &self.path)
            .map_err(|error| DeltafinError::new(error.to_string()))?;
        self.identity.validate(&self.file, &self.path)?;
        if digest != digest_from_hex(WEIGHTS_SHA256)? {
            return Err(DeltafinError::new(format!(
                "EAGLE-3 checkpoint SHA-256 does not match the pinned {OFFICIAL_MODEL_ID} release"
            )));
        }
        Ok(())
    }
}

fn admit_header(raw_header: &[u8]) -> Result<Box<[Eagle3Tensor]>> {
    let header = strict_json(raw_header, "EAGLE-3 safetensors header")?;
    let object = header
        .as_object()
        .ok_or_else(|| DeltafinError::new("EAGLE-3 safetensors header is not an object"))?;
    if object.len() != ROSTER.len() + 2 || !object.contains_key("__metadata__") {
        return Err(DeltafinError::new(
            "EAGLE-3 safetensors header does not name exactly the pinned roster",
        ));
    }
    let data_start = 8 + HEADER_BYTES;
    let data_bytes = CHECKPOINT_BYTES - data_start;
    let locate = |name: &str, shape: &[u64]| -> Result<Range<u64>> {
        let entry = object
            .get(name)
            .ok_or_else(|| DeltafinError::new(format!("EAGLE-3 checkpoint has no {name}")))?;
        let dtype = entry.get("dtype").and_then(|value| value.as_str());
        let actual: Option<Vec<u64>> = entry
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| values.iter().filter_map(|value| value.as_u64()).collect());
        let offsets: Option<Vec<u64>> = entry
            .get("data_offsets")
            .and_then(|value| value.as_array())
            .map(|values| values.iter().filter_map(|value| value.as_u64()).collect());
        let elements: u64 = shape.iter().product();
        match (dtype, actual, offsets) {
            (Some("BF16"), Some(actual), Some(offsets))
                if actual == shape
                    && offsets.len() == 2
                    && offsets[0] <= offsets[1]
                    && offsets[1] <= data_bytes
                    && offsets[1] - offsets[0] == elements * 2 =>
            {
                Ok(data_start + offsets[0]..data_start + offsets[1])
            }
            _ => Err(DeltafinError::new(format!(
                "EAGLE-3 tensor {name} violates its pinned BF16 shape contract"
            ))),
        }
    };
    locate(EMBEDDING_NAME, &EMBEDDING_SHAPE)?;
    ROSTER
        .iter()
        .map(|&(name, slot, shape, rank)| {
            let bytes = locate(name, &shape[..rank as usize])?;
            Ok(Eagle3Tensor {
                name,
                slot,
                rank,
                shape,
                bytes,
            })
        })
        .collect()
}

/// Provider memory the drafter needs for `context` committed positions: its
/// owned weights, the cache, and a bounded number of live prefix snapshots.
pub fn provider_reserve(context: usize) -> Result<u64> {
    let positions = u64::try_from(context.saturating_add(CHAIN_HEADROOM))
        .map_err(|_| DeltafinError::new("EAGLE-3 context does not fit u64"))?;
    // The live cache plus the controller's base, origin, boundary and
    // in-flight snapshots, each at most one cache.
    let cache = positions
        .checked_mul(CACHE_BYTES_PER_POSITION)
        .and_then(|bytes| bytes.checked_mul(5))
        .ok_or_else(|| DeltafinError::new("EAGLE-3 cache reserve overflowed"))?;
    // One step's activations: a 64-row advance through the fused 2H input,
    // the 18,432-wide MLP and FP32 head logits for a proposal row.
    let workspace =
        64 * (2 * HIDDEN as u64 + 18_432 + 3 * HIDDEN as u64) * 4 + u64::from(VOCABULARY) * 4 * 2;
    OWNED_BYTES
        .checked_add(cache)
        .and_then(|bytes| bytes.checked_add(workspace))
        .ok_or_else(|| DeltafinError::new("EAGLE-3 provider reserve overflowed"))
}

#[repr(C)]
struct CreateV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    flags: u32,
    tensor_count: u32,
    tensors: *const TensorV1,
    max_positions: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct ReportV1 {
    struct_size: u32,
    abi_version: u32,
    model: u64,
    token_count: u64,
    cache_length: u64,
    max_positions: u64,
    flags: u32,
    proposing: u32,
    reserved: [u64; 3],
}

impl ReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            model: 0,
            token_count: 0,
            cache_length: 0,
            max_positions: 0,
            flags: 0,
            proposing: 0,
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct AdvanceV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    model: u64,
    target_rows: u64,
    rows: u64,
    expected_token_count: u64,
    input_embeddings_bf16: *const u8,
    input_embedding_bytes: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct StepV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    model: u64,
    phase: u32,
    reserved32: u32,
    embedding_bf16: *const u8,
    embedding_bytes: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct StepReportV1 {
    struct_size: u32,
    abi_version: u32,
    token_id: u32,
    drafts: u32,
    reserved: [u64; 4],
}

impl StepReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            token_id: 0,
            drafts: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct SnapshotReportV1 {
    struct_size: u32,
    abi_version: u32,
    snapshot: u64,
    token_count: u64,
    reserved: [u64; 4],
}

impl SnapshotReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            snapshot: 0,
            token_count: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct RestoreV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    model: u64,
    snapshot: u64,
    reserved: [u64; 4],
}

const _: [(); 80] = [(); size_of::<CreateV1>()];
const _: [(); 72] = [(); size_of::<ReportV1>()];
const _: [(); 96] = [(); size_of::<AdvanceV1>()];
const _: [(); 80] = [(); size_of::<StepV1>()];
const _: [(); 48] = [(); size_of::<StepReportV1>()];
const _: [(); 56] = [(); size_of::<SnapshotReportV1>()];
const _: [(); 64] = [(); size_of::<RestoreV1>()];

unsafe extern "C" {
    fn deltafin_provider_eagle3_create_v1(
        request: *const CreateV1,
        report: *mut ReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_eagle3_destroy_v1(
        request: *const ResourceV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_eagle3_advance_v1(
        request: *const AdvanceV1,
        report: *mut ReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_eagle3_step_v1(
        request: *const StepV1,
        report: *mut StepReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_eagle3_snapshot_v1(
        request: *const ResourceV1,
        report: *mut SnapshotReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_eagle3_restore_v1(
        request: *const RestoreV1,
        report: *mut ReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_eagle3_snapshot_destroy_v1(
        request: *const ResourceV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

struct ModelInner {
    session: Arc<SessionInner>,
    handle: u64,
    flags: u32,
    hidden: usize,
    tap_width: usize,
    vocabulary: u32,
    max_positions: u64,
    /// Committed positions the drafter has seen, including the pending row.
    token_count: Mutex<usize>,
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        release(
            self.session.handle,
            self.handle,
            deltafin_provider_eagle3_destroy_v1,
        );
    }
}

/// Provider-owned EAGLE-3.1 arithmetic and cache. It can only return
/// unverified candidate token IDs.
#[derive(Clone)]
pub struct NativeEagle3 {
    inner: Arc<ModelInner>,
}

impl NativeEagle3 {
    pub fn bind(
        session: &NativeProviderSession,
        checkpoint: &Eagle3Checkpoint,
        max_positions: usize,
    ) -> Result<Self> {
        checkpoint.verify_full_digest()?;
        let mapping = ReadOnlyMap::file(checkpoint.file())?;
        let descriptors = checkpoint
            .tensors()
            .iter()
            .map(|tensor| {
                Ok(TensorV1 {
                    slot: tensor.slot,
                    scalar_type: BF16,
                    rank: tensor.rank,
                    flags: 0,
                    shape: tensor.shape,
                    data: mapping
                        .pointer(tensor.bytes.start, tensor.bytes.end - tensor.bytes.start)?,
                    data_length: tensor.bytes.end - tensor.bytes.start,
                    reserved: [0; 2],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let model = Self::create(session, &descriptors, 0, max_positions)?;
        checkpoint.validate_live_identity()?;
        Ok(model)
    }

    fn create(
        session: &NativeProviderSession,
        descriptors: &[TensorV1],
        flags: u32,
        max_positions: usize,
    ) -> Result<Self> {
        let lease = session.lease();
        let max_positions = u64::try_from(max_positions)
            .map_err(|_| DeltafinError::new("EAGLE-3 capacity does not fit u64"))?;
        let request = CreateV1 {
            struct_size: size_of::<CreateV1>() as u32,
            abi_version: ABI_VERSION,
            session: lease.handle,
            flags,
            tensor_count: descriptors.len() as u32,
            tensors: descriptors.as_ptr(),
            max_positions,
            reserved: [0; 5],
        };
        let mut report = ReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: descriptor backing stays mapped through the synchronous
        // call; native code copies every tensor before it returns.
        let status = unsafe {
            deltafin_provider_eagle3_create_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "create native EAGLE-3", &error)?;
        if report.struct_size as usize != size_of::<ReportV1>()
            || report.abi_version != ABI_VERSION
            || report.model == 0
            || report.token_count != 0
            || report.cache_length != 0
            || report.max_positions != max_positions
            || report.flags != flags
            || report.proposing != 0
            || report.reserved != [0; 3]
        {
            if report.model != 0 {
                release(
                    lease.handle,
                    report.model,
                    deltafin_provider_eagle3_destroy_v1,
                );
            }
            return Err(DeltafinError::new(
                "native EAGLE-3 create report is invalid",
            ));
        }
        let synthetic = flags & SYNTHETIC != 0;
        let hidden = if synthetic { 8 } else { HIDDEN };
        Ok(Self {
            inner: Arc::new(ModelInner {
                session: lease,
                handle: report.model,
                flags,
                hidden,
                tap_width: 3 * hidden,
                vocabulary: if synthetic { 32 } else { VOCABULARY },
                max_positions,
                token_count: Mutex::new(0),
            }),
        })
    }

    pub fn token_count(&self) -> usize {
        *self
            .inner
            .token_count
            .lock()
            .expect("EAGLE-3 state mutex poisoned")
    }

    /// Advance by the leading `rows` of a provider-owned BF16 [T, 3H] capture.
    /// `input_embeddings_bf16` holds the exact embedding rows of the tokens
    /// those positions consumed, in order.
    pub fn advance(
        &self,
        target_rows: &ProviderTensor,
        rows: usize,
        input_embeddings_bf16: &[u8],
    ) -> Result<()> {
        let (available, columns) = target_rows.shape();
        if rows == 0 || rows > available || columns != self.inner.tap_width {
            return Err(DeltafinError::new(
                "EAGLE-3 target rows have the wrong [T,3H] geometry",
            ));
        }
        if input_embeddings_bf16.len() != rows * self.inner.hidden * 2 {
            return Err(DeltafinError::new(
                "EAGLE-3 input embeddings do not match the committed rows",
            ));
        }
        let handle = target_rows.handle_in_session(&self.inner.session)?;
        let mut token_count = self
            .inner
            .token_count
            .lock()
            .expect("EAGLE-3 state mutex poisoned");
        let expected = token_count
            .checked_add(rows)
            .ok_or_else(|| DeltafinError::new("EAGLE-3 token count overflowed"))?;
        let request = AdvanceV1 {
            struct_size: size_of::<AdvanceV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.session.handle,
            model: self.inner.handle,
            target_rows: handle,
            rows: rows as u64,
            expected_token_count: *token_count as u64,
            input_embeddings_bf16: input_embeddings_bf16.as_ptr(),
            input_embedding_bytes: input_embeddings_bf16.len() as u64,
            reserved: [0; 4],
        };
        let mut report = ReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the tensor and model Arcs keep both handles live, and the
        // embedding bytes stay valid for the synchronous native copy.
        let status = unsafe {
            deltafin_provider_eagle3_advance_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "advance native EAGLE-3", &error)?;
        self.validate_report(&report, expected)?;
        *token_count = expected;
        Ok(())
    }

    /// Greedy chain of `drafts` proposals continuing the pending row. The
    /// first input is the token K3 produced there; each later input is the
    /// previous draft. `embed` returns one exact BF16 embedding row. The
    /// chain's speculative cache rows are always discarded before return.
    pub fn propose(
        &self,
        pending_token_id: u32,
        drafts: usize,
        mut embed: impl FnMut(u32) -> Result<Vec<u8>>,
    ) -> Result<Vec<u32>> {
        if !(1..=usize::from(MAXIMUM_DRAFTS)).contains(&drafts) {
            return Err(DeltafinError::new("EAGLE-3 proposals carry 1..=7 drafts"));
        }
        let token_count = self
            .inner
            .token_count
            .lock()
            .expect("EAGLE-3 state mutex poisoned");
        if *token_count == 0 {
            return Err(DeltafinError::new(
                "EAGLE-3 proposal needs at least one committed row",
            ));
        }
        let mut tokens = Vec::with_capacity(drafts);
        let mut input = pending_token_id;
        let chain = (|| -> Result<()> {
            for index in 0..drafts {
                let embedding = embed(input)?;
                let phase = if index == 0 {
                    STEP_BEGIN
                } else {
                    STEP_CONTINUE
                };
                let token = self.step(phase, &embedding)?;
                tokens.push(token);
                input = token;
            }
            Ok(())
        })();
        let ended = self.step_end();
        drop(token_count);
        chain?;
        ended?;
        Ok(tokens)
    }

    fn step(&self, phase: u32, embedding: &[u8]) -> Result<u32> {
        if embedding.len() != self.inner.hidden * 2 {
            return Err(DeltafinError::new(
                "EAGLE-3 step needs exactly one BF16 embedding row",
            ));
        }
        let request = StepV1 {
            struct_size: size_of::<StepV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.session.handle,
            model: self.inner.handle,
            phase,
            reserved32: 0,
            embedding_bf16: embedding.as_ptr(),
            embedding_bytes: embedding.len() as u64,
            reserved: [0; 4],
        };
        let mut report = StepReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the embedding row stays valid for the synchronous copy.
        let status = unsafe {
            deltafin_provider_eagle3_step_v1(&request, &mut report, error.as_mut_ptr(), error.len())
        };
        ffi_status(status, "run native EAGLE-3 proposal step", &error)?;
        if report.struct_size as usize != size_of::<StepReportV1>()
            || report.abi_version != ABI_VERSION
            || report.drafts != 1
            || report.token_id >= self.inner.vocabulary
            || report.reserved != [0; 4]
        {
            return Err(DeltafinError::new(
                "native EAGLE-3 proposal step report is invalid",
            ));
        }
        Ok(report.token_id)
    }

    fn step_end(&self) -> Result<()> {
        let request = StepV1 {
            struct_size: size_of::<StepV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.session.handle,
            model: self.inner.handle,
            phase: STEP_END,
            reserved32: 0,
            embedding_bf16: std::ptr::null(),
            embedding_bytes: 0,
            reserved: [0; 4],
        };
        let mut report = StepReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: an END request carries no pointer.
        let status = unsafe {
            deltafin_provider_eagle3_step_v1(&request, &mut report, error.as_mut_ptr(), error.len())
        };
        ffi_status(status, "end native EAGLE-3 proposal chain", &error)
    }

    pub fn snapshot(&self) -> Result<NativeEagle3Snapshot> {
        let token_count = self
            .inner
            .token_count
            .lock()
            .expect("EAGLE-3 state mutex poisoned");
        let request = resource(self.inner.session.handle, self.inner.handle);
        let mut report = SnapshotReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/report/error buffers are valid for the call.
        let status = unsafe {
            deltafin_provider_eagle3_snapshot_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "snapshot native EAGLE-3", &error)?;
        if report.struct_size as usize != size_of::<SnapshotReportV1>()
            || report.abi_version != ABI_VERSION
            || report.snapshot == 0
            || report.token_count != *token_count as u64
            || report.reserved != [0; 4]
        {
            if report.snapshot != 0 {
                release(
                    self.inner.session.handle,
                    report.snapshot,
                    deltafin_provider_eagle3_snapshot_destroy_v1,
                );
            }
            return Err(DeltafinError::new(
                "native EAGLE-3 snapshot report is invalid",
            ));
        }
        Ok(NativeEagle3Snapshot {
            inner: Arc::new(SnapshotInner {
                model: Arc::clone(&self.inner),
                handle: report.snapshot,
                token_count: *token_count,
            }),
        })
    }

    pub fn restore(&self, snapshot: &NativeEagle3Snapshot) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &snapshot.inner.model) {
            return Err(DeltafinError::new(
                "EAGLE-3 snapshot belongs to another model",
            ));
        }
        let mut token_count = self
            .inner
            .token_count
            .lock()
            .expect("EAGLE-3 state mutex poisoned");
        let request = RestoreV1 {
            struct_size: size_of::<RestoreV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.session.handle,
            model: self.inner.handle,
            snapshot: snapshot.inner.handle,
            reserved: [0; 4],
        };
        let mut report = ReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: model and snapshot Arcs keep both native handles live.
        let status = unsafe {
            deltafin_provider_eagle3_restore_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "restore native EAGLE-3", &error)?;
        self.validate_report(&report, snapshot.inner.token_count)?;
        *token_count = snapshot.inner.token_count;
        Ok(())
    }

    fn validate_report(&self, report: &ReportV1, token_count: usize) -> Result<()> {
        if report.struct_size as usize != size_of::<ReportV1>()
            || report.abi_version != ABI_VERSION
            || report.model != self.inner.handle
            || report.token_count != token_count as u64
            || report.cache_length != token_count.saturating_sub(1) as u64
            || report.max_positions != self.inner.max_positions
            || report.flags != self.inner.flags
            || report.proposing != 0
            || report.reserved != [0; 3]
        {
            return Err(DeltafinError::new("native EAGLE-3 state report is invalid"));
        }
        Ok(())
    }
}

struct SnapshotInner {
    model: Arc<ModelInner>,
    handle: u64,
    token_count: usize,
}

impl Drop for SnapshotInner {
    fn drop(&mut self) {
        release(
            self.model.session.handle,
            self.handle,
            deltafin_provider_eagle3_snapshot_destroy_v1,
        );
    }
}

#[derive(Clone)]
pub struct NativeEagle3Snapshot {
    inner: Arc<SnapshotInner>,
}

impl NativeEagle3Snapshot {
    pub fn token_count(&self) -> usize {
        self.inner.token_count
    }
}

/// Engine-facing adapter for the transactional proposal controller. It owns
/// a second descriptor on K3's exact embedding file and a one-chunk arena;
/// the 2.35 GB table is never copied or made provider-resident.
pub(crate) struct NativeEagle3Backend {
    model: NativeEagle3,
    zero: NativeEagle3Snapshot,
    embedding: EmbeddingTable,
    embedding_arena: EmbeddingArena,
    identity: ModelIdentity,
}

impl NativeEagle3Backend {
    pub(crate) fn bind(
        session: &NativeProviderSession,
        checkpoint: &Eagle3Checkpoint,
        model_root: &Path,
        device: Device,
        context: usize,
    ) -> Result<Self> {
        let model =
            NativeEagle3::bind(session, checkpoint, context.saturating_add(CHAIN_HEADROOM))?;
        let zero = model.snapshot()?;
        if zero.token_count() != 0 {
            return Err(DeltafinError::new(
                "new native EAGLE-3 model did not begin at token boundary zero",
            ));
        }
        let embedding = EmbeddingTable::open_k3(model_root)?;
        let embedding_arena = EmbeddingArena::new(MAXIMUM_ADVANCE_ROWS)?;
        let identity = ModelIdentity::new(
            "deltafin-native-eagle3.1-v1",
            digest_from_hex(WEIGHTS_SHA256)?,
            "moonshotai/Kimi-K3 mxfp4",
            hex_digest(&PINNED_INVENTORY_SHA256),
            "deltafin-k3-tokenizer-contract-v1",
            format!("layers=1,hidden={HIDDEN},kv=512,rope=64,taps=1/45/89,max={context}"),
            "bf16-rms-fp32-yarn-adjacent-v1",
            device.to_string(),
        )
        .map_err(|error| DeltafinError::new(format!("build EAGLE-3 model identity: {error}")))?;
        Ok(Self {
            model,
            zero,
            embedding,
            embedding_arena,
            identity,
        })
    }
}

impl DraftBackend for NativeEagle3Backend {
    type Snapshot = NativeEagle3Snapshot;
    type TargetContext = ProviderTensor;

    fn reset_state(&mut self) -> std::result::Result<(), BackendFailure> {
        self.model
            .restore(&self.zero)
            .map_err(releasable_backend_failure)
    }

    fn snapshot_state(&mut self) -> std::result::Result<Self::Snapshot, BackendFailure> {
        self.model.snapshot().map_err(releasable_backend_failure)
    }

    fn restore_state(
        &mut self,
        snapshot: &Self::Snapshot,
    ) -> std::result::Result<(), BackendFailure> {
        self.model
            .restore(snapshot)
            .map_err(releasable_backend_failure)
    }

    fn state_token_count(&mut self) -> std::result::Result<usize, BackendFailure> {
        Ok(self.model.token_count())
    }

    fn model_identity(&mut self) -> std::result::Result<ModelIdentity, BackendFailure> {
        Ok(self.identity.clone())
    }

    fn propose(
        &mut self,
        pending_token_id: u32,
        max_drafts: u8,
    ) -> std::result::Result<BackendProposal, BackendFailure> {
        let embedding = &self.embedding;
        let arena = &mut self.embedding_arena;
        let tokens = self
            .model
            .propose(
                pending_token_id,
                usize::from(max_drafts.min(MAXIMUM_DRAFTS)),
                |token| Ok(embedding.read_rows(&[token], arena)?.bytes().to_vec()),
            )
            .map_err(releasable_backend_failure)?;
        Ok(BackendProposal::new(tokens))
    }

    fn advance_target_state(
        &mut self,
        target_context: &Self::TargetContext,
        committed_input_ids: &[u32],
    ) -> std::result::Result<(), BackendFailure> {
        let embeddings = self
            .embedding
            .read_rows(committed_input_ids, &mut self.embedding_arena)
            .map_err(releasable_backend_failure)?;
        self.model
            .advance(
                target_context,
                committed_input_ids.len(),
                embeddings.bytes(),
            )
            .map_err(releasable_backend_failure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shapes of the provider's small canary (H=8, I=12, two heads, q/kv
    /// rank 4, nope 2, rope 4, v 2, vocabulary 32).
    fn canary_shape(slot: u32) -> Vec<u64> {
        match slot {
            1 => vec![8, 24],
            2..=6 | 14 | 18 => vec![8],
            7 => vec![4, 16],
            8 | 11 => vec![4],
            9 => vec![12, 4],
            10 => vec![8, 16],
            12 => vec![8, 4],
            13 => vec![8, 4],
            15 | 16 => vec![12, 8],
            17 => vec![8, 12],
            19 => vec![32, 8],
            _ => unreachable!(),
        }
    }

    fn bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn canary_descriptors() -> (Vec<Vec<u16>>, Vec<TensorV1>) {
        let storage: Vec<Vec<u16>> = (1..=TENSOR_COUNT as u32)
            .map(|slot| {
                let shape = canary_shape(slot);
                let elements = shape.iter().product::<u64>() as usize;
                let norm = shape.len() == 1;
                (0..elements)
                    .map(|index| {
                        if norm {
                            bf16(1.0)
                        } else {
                            // Deterministic, slot-salted weights in [-0.5, 0.5].
                            let raw = (index as u32 * 7 + slot * 13) % 23;
                            bf16(raw as f32 / 22.0 - 0.5)
                        }
                    })
                    .collect()
            })
            .collect();
        let descriptors = storage
            .iter()
            .enumerate()
            .map(|(index, values)| {
                let slot = index as u32 + 1;
                let shape = canary_shape(slot);
                let mut fixed = [0; 2];
                fixed[..shape.len()].copy_from_slice(&shape);
                TensorV1 {
                    slot,
                    scalar_type: BF16,
                    rank: shape.len() as u32,
                    flags: 0,
                    shape: fixed,
                    data: values.as_ptr().cast(),
                    data_length: (values.len() * 2) as u64,
                    reserved: [0; 2],
                }
            })
            .collect();
        (storage, descriptors)
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| bf16(value).to_le_bytes())
            .collect()
    }

    fn embedding_row(token: u32) -> Vec<u8> {
        bytes(
            &(0..8)
                .map(|column| (((token * 5 + column * 3) % 11) as f32 - 5.0) / 5.0)
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn synthetic_wrapper_tracks_rows_snapshots_and_chains() {
        let session = NativeProviderSession::target(Device::Cpu).expect("CPU session");
        let (_storage, descriptors) = canary_descriptors();
        let model = NativeEagle3::create(&session, &descriptors, SYNTHETIC, 32).unwrap();
        assert_eq!(model.token_count(), 0);
        assert!(
            model
                .propose(1, 3, |token| Ok(embedding_row(token)))
                .is_err()
        );

        let taps: Vec<f32> = (0..4 * 24)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) / 9.0)
            .collect();
        let rows = session.upload_bf16(4, 24, &bytes(&taps)).unwrap();
        let inputs: Vec<u8> = [3_u32, 4, 5]
            .iter()
            .flat_map(|&token| embedding_row(token))
            .collect();
        model.advance(&rows, 3, &inputs).unwrap();
        assert_eq!(model.token_count(), 3);
        let base = model.snapshot().unwrap();

        let first = model
            .propose(6, 4, |token| Ok(embedding_row(token)))
            .unwrap();
        assert_eq!(first.len(), 4);
        assert!(first.iter().all(|&token| token < 32));
        // A proposal never changes committed state and repeats exactly.
        assert_eq!(model.token_count(), 3);
        let again = model
            .propose(6, 4, |token| Ok(embedding_row(token)))
            .unwrap();
        assert_eq!(first, again);
        // A failed embedding lookup still closes the chain.
        let mut calls = 0;
        assert!(
            model
                .propose(6, 4, |token| {
                    calls += 1;
                    if calls == 2 {
                        Err(DeltafinError::new("embedding unavailable"))
                    } else {
                        Ok(embedding_row(token))
                    }
                })
                .is_err()
        );
        assert_eq!(
            model
                .propose(6, 4, |token| Ok(embedding_row(token)))
                .unwrap(),
            first
        );

        // Advancing, then restoring, reproduces the original proposal.
        let more: Vec<u8> = [6_u32, 7]
            .iter()
            .flat_map(|&token| embedding_row(token))
            .collect();
        model.advance(&rows, 2, &more).unwrap();
        assert_eq!(model.token_count(), 5);
        model.restore(&base).unwrap();
        assert_eq!(model.token_count(), 3);
        assert_eq!(
            model
                .propose(6, 4, |token| Ok(embedding_row(token)))
                .unwrap(),
            first
        );

        // Geometry and contract violations are refused before native state moves.
        assert!(model.advance(&rows, 5, &more).is_err());
        assert!(model.advance(&rows, 2, &inputs).is_err());
        assert!(
            model
                .propose(6, 8, |token| Ok(embedding_row(token)))
                .is_err()
        );
        assert_eq!(model.token_count(), 3);
    }

    #[test]
    fn reserve_charges_owned_weights_cache_and_workspace() {
        let small = provider_reserve(8_192).unwrap();
        let large = provider_reserve(131_072).unwrap();
        assert!(small > OWNED_BYTES);
        assert!(small < OWNED_BYTES + (128 << 20));
        assert!(large > small);
        assert_eq!(
            large - small,
            (131_072 - 8_192) * CACHE_BYTES_PER_POSITION * 5
        );
    }

    #[test]
    fn installed_checkpoint_is_admitted_against_the_pinned_roster() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(DIRECTORY);
        if !directory.join(CHECKPOINT_BASENAME).is_file() {
            return;
        }
        let checkpoint = Eagle3Checkpoint::open(&directory).unwrap();
        let owned: u64 = checkpoint
            .tensors()
            .iter()
            .map(|tensor| tensor.bytes.end - tensor.bytes.start)
            .sum();
        assert_eq!(owned, OWNED_BYTES);
        assert_eq!(checkpoint.tensors().len(), TENSOR_COUNT);
        assert!(
            checkpoint
                .tensors()
                .iter()
                .enumerate()
                .all(|(index, tensor)| tensor.slot == index as u32 + 1)
        );
    }

    #[test]
    fn header_admission_refuses_a_changed_roster() {
        let mut header = serde_json::Map::new();
        header.insert(
            "__metadata__".into(),
            serde_json::json!({"torchspec_version": "0.1.0"}),
        );
        let mut offset = 0_u64;
        let mut push = |name: &str, shape: &[u64]| {
            let bytes = shape.iter().product::<u64>() * 2;
            header.insert(
                name.into(),
                serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        };
        push(EMBEDDING_NAME, &EMBEDDING_SHAPE);
        for (name, _, shape, rank) in ROSTER {
            push(name, &shape[..rank as usize]);
        }
        let admitted = serde_json::to_vec(&header).unwrap();
        assert_eq!(admit_header(&admitted).unwrap().len(), TENSOR_COUNT);

        let mut renamed = header.clone();
        let value = renamed.remove("norm.weight").unwrap();
        renamed.insert("final_norm.weight".into(), value);
        assert!(admit_header(&serde_json::to_vec(&renamed).unwrap()).is_err());

        let mut reshaped = header.clone();
        reshaped["fc.weight"]["shape"] = serde_json::json!([7_168, 14_336]);
        assert!(admit_header(&serde_json::to_vec(&reshaped).unwrap()).is_err());

        let mut retyped = header;
        retyped["lm_head.weight"]["dtype"] = serde_json::json!("F16");
        assert!(admit_header(&serde_json::to_vec(&retyped).unwrap()).is_err());
    }
}
