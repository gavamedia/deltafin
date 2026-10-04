//! Lifetime-safe Rust ownership for the fixed DSpark proposal-only C ABI.
//!
//! The admitted safetensors file is mapped read-only only for the synchronous
//! native bind. C++ copies every owned BF16 tensor before publishing a handle.
//! The shared K3 embedding is deliberately excluded; callers provide exactly
//! seven already-read BF16 embedding rows per proposal. The K3 LM head is
//! borrowed from the owning provider session.

use std::ffi::{CStr, c_char};
use std::fs::File;
use std::mem::size_of;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::dspark_checkpoint::{
    DSparkCheckpoint, DSparkConfig, OFFICIAL_WEIGHTS_SHA256, OWNED_TENSOR_COUNT,
    TRAINED_TARGET_REVISION, digest_from_hex,
};
use crate::dspark_runtime::{BackendFailure, BackendProposal, DraftBackend, ModelIdentity};
use crate::embedding::{EmbeddingArena, EmbeddingTable};
use crate::error::{DeltafinError, Result};
use crate::inventory::PINNED_INVENTORY_SHA256;
use crate::platform::Device;
use crate::provider::{NativeProviderSession, ProviderTensor, SessionInner};
use crate::sys::fs as sys_fs;

const ABI_VERSION: u32 = 1;
pub(crate) const BF16: u32 = 1;
const TENSOR_COUNT: usize = 67;
const QUERY_ROWS: usize = 7;
const HIDDEN: usize = 7_168;
const TARGET_CONTEXT: usize = 5 * HIDDEN;
const ERROR_CAPACITY: usize = 2_048;

/// One borrowed BF16 weight for a proposal-model create call; shared by the
/// DSpark and EAGLE-3 rosters (`DeltafinProviderDSparkTensorV1`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct TensorV1 {
    pub(crate) slot: u32,
    pub(crate) scalar_type: u32,
    pub(crate) rank: u32,
    pub(crate) flags: u32,
    pub(crate) shape: [u64; 2],
    pub(crate) data: *const u8,
    pub(crate) data_length: u64,
    pub(crate) reserved: [u64; 2],
}

#[repr(C)]
struct CreateV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    flags: u32,
    tensor_count: u32,
    tensors: *const TensorV1,
    synthetic_head_f32: *const f32,
    synthetic_head_elements: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct ReportV1 {
    struct_size: u32,
    abi_version: u32,
    model: u64,
    cache_length: u64,
    cache_generation: u64,
    tensor_count: u32,
    flags: u32,
    reserved: [u64; 3],
}

impl ReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            model: 0,
            cache_length: 0,
            cache_generation: 0,
            tensor_count: 0,
            flags: 0,
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
pub(crate) struct ResourceV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    resource: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 4],
}

pub(crate) fn resource(session: u64, handle: u64) -> ResourceV1 {
    ResourceV1 {
        struct_size: size_of::<ResourceV1>() as u32,
        abi_version: ABI_VERSION,
        session,
        resource: handle,
        flags: 0,
        reserved0: 0,
        reserved: [0; 4],
    }
}

#[repr(C)]
struct AppendV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    model: u64,
    target_context_bf16: *const u8,
    target_context_bytes: u64,
    positions: *const i64,
    rows: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct AppendTensorV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    model: u64,
    target_context: u64,
    expected_cache_length: u64,
    expected_cache_generation: u64,
    rows: u64,
    reserved: [u64; 3],
}

#[repr(C)]
struct SnapshotReportV1 {
    struct_size: u32,
    abi_version: u32,
    snapshot: u64,
    cache_length: u64,
    cache_generation: u64,
    reserved: [u64; 4],
}

impl SnapshotReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            snapshot: 0,
            cache_length: 0,
            cache_generation: 0,
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

#[repr(C)]
struct ProposeV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    model: u64,
    anchor_token_id: u32,
    score_rows: u32,
    query_embeddings_bf16: *const u8,
    query_embedding_bytes: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct ProposalReportV1 {
    struct_size: u32,
    abi_version: u32,
    score_rows: u32,
    flags: u32,
    anchor_position: u64,
    cache_generation: u64,
    token_ids: [u32; QUERY_ROWS],
    confidence_logits: [f32; QUERY_ROWS],
    reserved: [u64; 4],
}

impl ProposalReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            score_rows: 0,
            flags: 0,
            anchor_position: 0,
            cache_generation: 0,
            token_ids: [0; QUERY_ROWS],
            confidence_logits: [0.0; QUERY_ROWS],
            reserved: [0; 4],
        }
    }
}

const _: [(); 64] = [(); size_of::<TensorV1>()];
const _: [(); 88] = [(); size_of::<CreateV1>()];
const _: [(); 64] = [(); size_of::<ReportV1>()];
const _: [(); 88] = [(); size_of::<AppendV1>()];
const _: [(); 80] = [(); size_of::<AppendTensorV1>()];
const _: [(); 64] = [(); size_of::<SnapshotReportV1>()];
const _: [(); 64] = [(); size_of::<RestoreV1>()];
const _: [(); 80] = [(); size_of::<ProposeV1>()];
const _: [(); 120] = [(); size_of::<ProposalReportV1>()];

unsafe extern "C" {
    fn deltafin_provider_dspark_create_v1(
        request: *const CreateV1,
        report: *mut ReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_dspark_destroy_v1(
        request: *const ResourceV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_dspark_append_target_v1(
        request: *const AppendV1,
        report: *mut ReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_dspark_append_target_tensor_v1(
        request: *const AppendTensorV1,
        report: *mut ReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_dspark_snapshot_v1(
        request: *const ResourceV1,
        report: *mut SnapshotReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_dspark_restore_v1(
        request: *const RestoreV1,
        report: *mut ReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_dspark_snapshot_destroy_v1(
        request: *const ResourceV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_dspark_propose_v1(
        request: *const ProposeV1,
        report: *mut ProposalReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

pub(crate) struct ReadOnlyMap(sys_fs::ReadOnlyMap);

impl ReadOnlyMap {
    fn checkpoint(checkpoint: &DSparkCheckpoint) -> Result<Self> {
        Self::file(checkpoint.file())
    }

    /// Map an admitted proposal checkpoint read-only for one synchronous
    /// native bind.
    pub(crate) fn file(file: &File) -> Result<Self> {
        let length = usize::try_from(
            file.metadata()
                .map_err(|error| DeltafinError::new(format!("stat proposal checkpoint: {error}")))?
                .len(),
        )
        .map_err(|_| DeltafinError::new("proposal checkpoint length exceeds usize"))?;
        sys_fs::ReadOnlyMap::map(file, length)
            .map(Self)
            .map_err(|error| {
                DeltafinError::new(format!("map admitted proposal checkpoint: {error}"))
            })
    }

    pub(crate) fn pointer(&self, offset: u64, length: u64) -> Result<*const u8> {
        let offset = usize::try_from(offset)
            .map_err(|_| DeltafinError::new("DSpark tensor offset exceeds usize"))?;
        let length = usize::try_from(length)
            .map_err(|_| DeltafinError::new("DSpark tensor length exceeds usize"))?;
        if offset > self.0.len() || length > self.0.len() - offset {
            return Err(DeltafinError::new(
                "DSpark tensor range exceeds mapped checkpoint",
            ));
        }
        // SAFETY: the checked extent lies within this live mapping.
        Ok(unsafe { self.0.address().add(offset) })
    }
}

fn tensor_slot(name: &str) -> Option<u32> {
    match name {
        "context_proj.weight" => return Some(1),
        "context_norm.weight" => return Some(2),
        "final_norm.weight" => return Some(3),
        "markov_head.markov_w1.weight" => return Some(4),
        "markov_head.markov_w2.weight" => return Some(5),
        "confidence_head.proj.weight" => return Some(6),
        "confidence_head.proj.bias" => return Some(7),
        "embed_tokens.weight" => return None,
        _ => {}
    }
    let rest = name.strip_prefix("layers.")?;
    let (layer, suffix) = rest.split_once('.')?;
    let layer: u32 = layer.parse().ok()?;
    if layer >= 5 {
        return None;
    }
    let component = match suffix {
        "input_layernorm.weight" => 0,
        "post_attention_layernorm.weight" => 1,
        "self_attn.q_a_proj.weight" => 2,
        "self_attn.q_a_layernorm.weight" => 3,
        "self_attn.q_b_proj.weight" => 4,
        "self_attn.kv_a_proj_with_mqa.weight" => 5,
        "self_attn.kv_a_layernorm.weight" => 6,
        "self_attn.kv_b_proj.weight" => 7,
        "self_attn.o_proj.weight" => 8,
        "mlp.gate_proj.weight" => 9,
        "mlp.up_proj.weight" => 10,
        "mlp.down_proj.weight" => 11,
        _ => return None,
    };
    Some(8 + layer * 12 + component)
}

#[derive(Debug, Clone, Copy)]
struct State {
    length: usize,
    generation: u64,
}

struct ModelInner {
    session: Arc<SessionInner>,
    handle: u64,
    flags: u32,
    target_context_width: usize,
    state: Mutex<State>,
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        let request = resource(self.session.handle, self.handle);
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the session Arc outlives this drop and the request is valid.
        let _ = unsafe {
            deltafin_provider_dspark_destroy_v1(&request, error.as_mut_ptr(), error.len())
        };
    }
}

/// Provider-owned DSpark arithmetic and cache. It can only return unverified
/// candidate evidence and never exposes an acceptance or emission operation.
#[derive(Clone)]
pub struct NativeDSpark {
    inner: Arc<ModelInner>,
}

impl NativeDSpark {
    pub fn bind(session: &NativeProviderSession, checkpoint: &DSparkCheckpoint) -> Result<Self> {
        checkpoint.verify_full_digest()?;
        if checkpoint.tensors().len() != OWNED_TENSOR_COUNT + 1 {
            return Err(DeltafinError::new(
                "DSpark checkpoint roster changed after admission",
            ));
        }
        let mapping = ReadOnlyMap::checkpoint(checkpoint)?;
        let mut slots: [Option<TensorV1>; TENSOR_COUNT] = [None; TENSOR_COUNT];
        for tensor in checkpoint.tensors() {
            if !tensor.owned_by_dspark {
                if tensor.name != "embed_tokens.weight" {
                    return Err(DeltafinError::new(
                        "unexpected externally-owned DSpark checkpoint tensor",
                    ));
                }
                continue;
            }
            let slot = tensor_slot(&tensor.name).ok_or_else(|| {
                DeltafinError::new(format!(
                    "DSpark tensor has no fixed provider slot: {}",
                    tensor.name
                ))
            })?;
            let index = usize::try_from(slot - 1)
                .map_err(|_| DeltafinError::new("DSpark slot conversion failed"))?;
            if slots[index].is_some() || tensor.shape.len() > 2 {
                return Err(DeltafinError::new(
                    "DSpark tensor slot is duplicate or has unsupported rank",
                ));
            }
            let absolute = tensor.bytes.start;
            let byte_length = tensor.bytes.end - tensor.bytes.start;
            let mut shape = [0_u64; 2];
            shape[..tensor.shape.len()].copy_from_slice(&tensor.shape);
            slots[index] = Some(TensorV1 {
                slot,
                scalar_type: BF16,
                rank: tensor.shape.len() as u32,
                flags: 0,
                shape,
                data: mapping.pointer(absolute, byte_length)?,
                data_length: byte_length,
                reserved: [0; 2],
            });
        }
        let descriptors: Vec<_> = slots
            .into_iter()
            .map(|slot| {
                slot.ok_or_else(|| DeltafinError::new("DSpark provider roster is incomplete"))
            })
            .collect::<Result<_>>()?;
        let model = Self::create(session, &descriptors, ptr::null(), 0, 0)?;
        checkpoint.validate_live_identity()?;
        Ok(model)
    }

    fn create(
        session: &NativeProviderSession,
        descriptors: &[TensorV1],
        synthetic_head: *const f32,
        synthetic_head_elements: usize,
        flags: u32,
    ) -> Result<Self> {
        let lease = session.lease();
        let request = CreateV1 {
            struct_size: size_of::<CreateV1>() as u32,
            abi_version: ABI_VERSION,
            session: lease.handle,
            flags,
            tensor_count: descriptors.len() as u32,
            tensors: descriptors.as_ptr(),
            synthetic_head_f32: synthetic_head,
            synthetic_head_elements: synthetic_head_elements as u64,
            reserved: [0; 5],
        };
        let mut report = ReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: descriptor backing and optional head remain live through the
        // synchronous call; native code copies all tensor bytes before return.
        let status = unsafe {
            deltafin_provider_dspark_create_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "create native DSpark", &error)?;
        if report.struct_size as usize != size_of::<ReportV1>()
            || report.abi_version != ABI_VERSION
            || report.model == 0
            || report.tensor_count as usize != TENSOR_COUNT
            || report.cache_length != 0
            || report.cache_generation != 0
            || report.flags != flags
            || report.reserved != [0; 3]
        {
            if report.model != 0 {
                release(
                    lease.handle,
                    report.model,
                    deltafin_provider_dspark_destroy_v1,
                );
            }
            return Err(DeltafinError::new("native DSpark create report is invalid"));
        }
        Ok(Self {
            inner: Arc::new(ModelInner {
                session: lease,
                handle: report.model,
                flags,
                target_context_width: if flags == 0 { TARGET_CONTEXT } else { 40 },
                state: Mutex::new(State {
                    length: 0,
                    generation: 0,
                }),
            }),
        })
    }

    pub fn state_token_count(&self) -> usize {
        self.inner
            .state
            .lock()
            .expect("DSpark state mutex poisoned")
            .length
    }

    pub fn append_target_context_bf16(
        &self,
        combined_rows: &[u8],
        positions: &[i64],
    ) -> Result<()> {
        let expected = positions
            .len()
            .checked_mul(TARGET_CONTEXT)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| DeltafinError::new("DSpark target context size overflowed"))?;
        if positions.is_empty() || combined_rows.len() != expected {
            return Err(DeltafinError::new(
                "DSpark target context BF16 rows do not match positions",
            ));
        }
        let mut state = self
            .inner
            .state
            .lock()
            .expect("DSpark state mutex poisoned");
        let expected = State {
            length: state
                .length
                .checked_add(positions.len())
                .ok_or_else(|| DeltafinError::new("DSpark cache length overflowed"))?,
            generation: state
                .generation
                .checked_add(1)
                .ok_or_else(|| DeltafinError::new("DSpark cache generation is exhausted"))?,
        };
        let request = AppendV1 {
            struct_size: size_of::<AppendV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.session.handle,
            model: self.inner.handle,
            target_context_bf16: combined_rows.as_ptr(),
            target_context_bytes: combined_rows.len() as u64,
            positions: positions.as_ptr(),
            rows: positions.len() as u64,
            reserved: [0; 4],
        };
        let mut report = ReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: both borrowed slices remain valid through the synchronous call.
        let status = unsafe {
            deltafin_provider_dspark_append_target_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "append native DSpark target context", &error)?;
        Self::validate_state_report(&self.inner, &report, expected)?;
        *state = expected;
        Ok(())
    }

    /// Advance the proposal cache from a provider-owned BF16 [T,5*H]
    /// activation. The handle is consumed in place on the model's device;
    /// activation bytes never cross into Rust or return through host memory.
    pub fn append_target_context_tensor(&self, context: &ProviderTensor) -> Result<()> {
        self.append_target_context_tensor_prefix(context, context.shape().0)
    }

    /// Append only the authoritative prefix of a provider-owned verification
    /// tensor. Native code narrows the existing device tensor without copying.
    pub fn append_target_context_tensor_prefix(
        &self,
        context: &ProviderTensor,
        rows: usize,
    ) -> Result<()> {
        let (available_rows, columns) = context.shape();
        if rows == 0 || rows > available_rows || columns != self.inner.target_context_width {
            return Err(DeltafinError::new(
                "DSpark provider tensor prefix has the wrong [T,5*H] geometry",
            ));
        }
        let handle = context.handle_in_session(&self.inner.session)?;
        let mut state = self
            .inner
            .state
            .lock()
            .expect("DSpark state mutex poisoned");
        let expected = State {
            length: state
                .length
                .checked_add(rows)
                .ok_or_else(|| DeltafinError::new("DSpark cache length overflowed"))?,
            generation: state
                .generation
                .checked_add(1)
                .ok_or_else(|| DeltafinError::new("DSpark cache generation is exhausted"))?,
        };
        let request = AppendTensorV1 {
            struct_size: size_of::<AppendTensorV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.session.handle,
            model: self.inner.handle,
            target_context: handle,
            expected_cache_length: state.length as u64,
            expected_cache_generation: state.generation,
            rows: rows as u64,
            reserved: [0; 3],
        };
        let mut report = ReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request contains only provider-owned handles. The borrowed
        // tensor and model Arc remain live through this synchronous call.
        let status = unsafe {
            deltafin_provider_dspark_append_target_tensor_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "append native DSpark provider tensor", &error)?;
        Self::validate_state_report(&self.inner, &report, expected)?;
        *state = expected;
        Ok(())
    }

    pub fn snapshot(&self) -> Result<NativeDSparkSnapshot> {
        let state = self
            .inner
            .state
            .lock()
            .expect("DSpark state mutex poisoned");
        let request = resource(self.inner.session.handle, self.inner.handle);
        let mut report = SnapshotReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/report/error buffers are valid for the call.
        let status = unsafe {
            deltafin_provider_dspark_snapshot_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "snapshot native DSpark", &error)?;
        if report.struct_size as usize != size_of::<SnapshotReportV1>()
            || report.abi_version != ABI_VERSION
            || report.snapshot == 0
            || report.cache_length != state.length as u64
            || report.cache_generation != state.generation
            || report.reserved != [0; 4]
        {
            if report.snapshot != 0 {
                release(
                    self.inner.session.handle,
                    report.snapshot,
                    deltafin_provider_dspark_snapshot_destroy_v1,
                );
            }
            return Err(DeltafinError::new(
                "native DSpark snapshot report is invalid",
            ));
        }
        Ok(NativeDSparkSnapshot {
            inner: Arc::new(SnapshotInner {
                model: Arc::clone(&self.inner),
                handle: report.snapshot,
                length: state.length,
                generation: state.generation,
            }),
        })
    }

    pub fn restore(&self, snapshot: &NativeDSparkSnapshot) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &snapshot.inner.model) {
            return Err(DeltafinError::new(
                "DSpark snapshot belongs to another model",
            ));
        }
        let mut state = self
            .inner
            .state
            .lock()
            .expect("DSpark state mutex poisoned");
        let expected = State {
            length: snapshot.inner.length,
            generation: state
                .generation
                .max(snapshot.inner.generation)
                .checked_add(1)
                .ok_or_else(|| DeltafinError::new("DSpark cache generation is exhausted"))?,
        };
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
            deltafin_provider_dspark_restore_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "restore native DSpark", &error)?;
        Self::validate_state_report(&self.inner, &report, expected)?;
        *state = expected;
        Ok(())
    }

    pub fn propose(
        &self,
        anchor_token_id: u32,
        score_rows: usize,
        trained_query_embeddings_bf16: &[u8],
    ) -> Result<NativeDSparkProposal> {
        if !(1..=QUERY_ROWS).contains(&score_rows)
            || trained_query_embeddings_bf16.len() != QUERY_ROWS * HIDDEN * 2
        {
            return Err(DeltafinError::new(
                "DSpark proposal requires 1..7 scores and exactly seven BF16 embedding rows",
            ));
        }
        let state = self
            .inner
            .state
            .lock()
            .expect("DSpark state mutex poisoned");
        let request = ProposeV1 {
            struct_size: size_of::<ProposeV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.session.handle,
            model: self.inner.handle,
            anchor_token_id,
            score_rows: score_rows as u32,
            query_embeddings_bf16: trained_query_embeddings_bf16.as_ptr(),
            query_embedding_bytes: trained_query_embeddings_bf16.len() as u64,
            reserved: [0; 4],
        };
        let mut report = ProposalReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: embedding bytes remain valid for the synchronous native copy.
        let status = unsafe {
            deltafin_provider_dspark_propose_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "run native DSpark proposal", &error)?;
        if report.struct_size as usize != size_of::<ProposalReportV1>()
            || report.abi_version != ABI_VERSION
            || report.score_rows as usize != score_rows
            || report.flags != 0
            || report.anchor_position != state.length as u64
            || report.cache_generation != state.generation
            || report.reserved != [0; 4]
            || report.confidence_logits[..score_rows]
                .iter()
                .any(|value| !value.is_finite())
            || report.token_ids[score_rows..]
                .iter()
                .any(|&value| value != 0)
            || report.confidence_logits[score_rows..]
                .iter()
                .any(|&value| value != 0.0)
        {
            return Err(DeltafinError::new(
                "native DSpark proposal report is invalid",
            ));
        }
        Ok(NativeDSparkProposal {
            token_ids: report.token_ids[..score_rows].into(),
            confidence_logits: report.confidence_logits[..score_rows].into(),
            anchor_position: state.length,
            cache_generation: state.generation,
        })
    }

    fn validate_state_report(inner: &ModelInner, report: &ReportV1, expected: State) -> Result<()> {
        if report.struct_size as usize != size_of::<ReportV1>()
            || report.abi_version != ABI_VERSION
            || report.model != inner.handle
            || report.cache_length != expected.length as u64
            || report.cache_generation != expected.generation
            || report.tensor_count as usize != TENSOR_COUNT
            || report.flags != inner.flags
            || report.reserved != [0; 3]
        {
            return Err(DeltafinError::new("native DSpark state report is invalid"));
        }
        Ok(())
    }
}

struct SnapshotInner {
    model: Arc<ModelInner>,
    handle: u64,
    length: usize,
    generation: u64,
}

impl Drop for SnapshotInner {
    fn drop(&mut self) {
        release(
            self.model.session.handle,
            self.handle,
            deltafin_provider_dspark_snapshot_destroy_v1,
        );
    }
}

#[derive(Clone)]
pub struct NativeDSparkSnapshot {
    inner: Arc<SnapshotInner>,
}

impl NativeDSparkSnapshot {
    pub fn token_count(&self) -> usize {
        self.inner.length
    }
}

pub struct NativeDSparkProposal {
    token_ids: Box<[u32]>,
    confidence_logits: Box<[f32]>,
    anchor_position: usize,
    cache_generation: u64,
}

impl NativeDSparkProposal {
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn confidence_logits(&self) -> &[f32] {
        &self.confidence_logits
    }

    pub fn anchor_position(&self) -> usize {
        self.anchor_position
    }

    pub fn cache_generation(&self) -> u64 {
        self.cache_generation
    }
}

/// Engine-facing adapter for the model-free transactional DSpark controller.
/// It owns a second descriptor for the exact K3 embedding file, but only a
/// seven-row arena; the 2.35 GiB table is never copied or provider-resident.
pub(crate) struct NativeDSparkBackend {
    model: NativeDSpark,
    zero: NativeDSparkSnapshot,
    embedding: EmbeddingTable,
    embedding_arena: EmbeddingArena,
    identity: ModelIdentity,
}

impl NativeDSparkBackend {
    pub(crate) fn bind(
        session: &NativeProviderSession,
        checkpoint: &DSparkCheckpoint,
        model_root: &Path,
        device: Device,
    ) -> Result<Self> {
        let model = NativeDSpark::bind(session, checkpoint)?;
        let zero = model.snapshot()?;
        if zero.token_count() != 0 {
            return Err(DeltafinError::new(
                "new native DSpark model did not begin at token boundary zero",
            ));
        }
        let embedding = EmbeddingTable::open_k3(model_root)?;
        let embedding_arena = EmbeddingArena::new(QUERY_ROWS)?;
        let weights = digest_from_hex(OFFICIAL_WEIGHTS_SHA256)?;
        let runtime_revision = hex_digest(&PINNED_INVENTORY_SHA256);
        let identity = ModelIdentity::new(
            "deltafin-native-dspark-v1",
            weights,
            TRAINED_TARGET_REVISION,
            runtime_revision,
            "deltafin-k3-tokenizer-contract-v1",
            format!(
                "layers=5,hidden={},kv={},rope={},max={}",
                DSparkConfig::OFFICIAL.hidden_size,
                DSparkConfig::OFFICIAL.kv_lora_rank,
                DSparkConfig::OFFICIAL.qk_rope_head_dim,
                DSparkConfig::OFFICIAL.maximum_context,
            ),
            "bf16-rms-fp32-yarn-adjacent-v1",
            device.to_string(),
        )
        .map_err(|error| DeltafinError::new(format!("build DSpark model identity: {error}")))?;
        Ok(Self {
            model,
            zero,
            embedding,
            embedding_arena,
            identity,
        })
    }
}

impl DraftBackend for NativeDSparkBackend {
    type Snapshot = NativeDSparkSnapshot;
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
        Ok(self.model.state_token_count())
    }

    fn model_identity(&mut self) -> std::result::Result<ModelIdentity, BackendFailure> {
        Ok(self.identity.clone())
    }

    fn propose(
        &mut self,
        pending_token_id: u32,
        max_drafts: u8,
    ) -> std::result::Result<BackendProposal, BackendFailure> {
        let mut query = [DSparkConfig::OFFICIAL.mask_token_id; QUERY_ROWS];
        query[0] = pending_token_id;
        let embeddings = self
            .embedding
            .read_rows(&query, &mut self.embedding_arena)
            .map_err(releasable_backend_failure)?;
        let proposal = self
            .model
            .propose(
                pending_token_id,
                usize::from(max_drafts),
                embeddings.bytes(),
            )
            .map_err(releasable_backend_failure)?;
        Ok(BackendProposal::new(proposal.token_ids()))
    }

    fn advance_target_state(
        &mut self,
        target_context: &Self::TargetContext,
        committed_input_ids: &[u32],
    ) -> std::result::Result<(), BackendFailure> {
        // DSpark consumes the rows alone; the IDs are already in its ledger.
        self.model
            .append_target_context_tensor_prefix(target_context, committed_input_ids.len())
            .map_err(releasable_backend_failure)
    }
}

pub(crate) fn releasable_backend_failure(error: DeltafinError) -> BackendFailure {
    BackendFailure::releasable(error.to_string())
}

pub(crate) fn hex_digest(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for &byte in digest {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

pub(crate) type ReleaseFn = unsafe extern "C" fn(*const ResourceV1, *mut c_char, usize) -> i32;

pub(crate) fn release(session: u64, handle: u64, function: ReleaseFn) {
    let request = resource(session, handle);
    let mut error = [0 as c_char; ERROR_CAPACITY];
    // SAFETY: caller's Arc ordering keeps the owning session live.
    let _ = unsafe { function(&request, error.as_mut_ptr(), error.len()) };
}

pub(crate) fn ffi_status(status: i32, operation: &str, error: &[c_char]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: native ffi_guard always NUL-terminates the fixed error buffer.
    let detail = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    Err(DeltafinError::new(format!("{operation}: {detail}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::Device;

    fn synthetic_descriptors() -> (Vec<Vec<u16>>, Vec<TensorV1>) {
        let shape = |slot: u32| -> Vec<u64> {
            match slot {
                1 => return vec![8, 40],
                2 | 3 => return vec![8],
                4 | 5 => return vec![32, 4],
                6 => return vec![1, 12],
                7 => return vec![1],
                _ => {}
            }
            match (slot - 8) % 12 {
                0 | 1 => vec![8],
                2 => vec![4, 8],
                3 => vec![4],
                4 => vec![12, 4],
                5 => vec![8, 8],
                6 => vec![4],
                7 => vec![8, 4],
                8 => vec![8, 4],
                9 | 10 => vec![12, 8],
                11 => vec![8, 12],
                _ => unreachable!(),
            }
        };
        let mut storage = Vec::new();
        for slot in 1..=67 {
            let dimensions = shape(slot);
            let elements = dimensions.iter().product::<u64>() as usize;
            storage.push(vec![0x3b80_u16; elements]);
        }
        let descriptors = storage
            .iter()
            .enumerate()
            .map(|(index, values)| {
                let slot = index as u32 + 1;
                let dimensions = shape(slot);
                let mut fixed = [0; 2];
                fixed[..dimensions.len()].copy_from_slice(&dimensions);
                TensorV1 {
                    slot,
                    scalar_type: BF16,
                    rank: dimensions.len() as u32,
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

    #[test]
    fn synthetic_wrapper_owns_snapshots_and_proposal_evidence() {
        let session = NativeProviderSession::target(Device::Cpu).expect("CPU session");
        let (_storage, descriptors) = synthetic_descriptors();
        let head = vec![0.01_f32; 32 * 8];
        let model = NativeDSpark::create(&session, &descriptors, head.as_ptr(), head.len(), 1)
            .expect("synthetic DSpark");

        let context = vec![0x80_u8; 2 * 40 * 2];
        let provider_context = session
            .upload_bf16(2, 40, &context)
            .expect("provider-owned BF16 context");
        model
            .append_target_context_tensor(&provider_context)
            .expect("zero-hop tensor append");
        let snapshot = model.snapshot().expect("snapshot");
        assert_eq!(snapshot.token_count(), 2);
        assert_eq!(snapshot.inner.generation, 1);

        let embeddings = [0x80_u8; 7 * 8 * 2];
        let request = ProposeV1 {
            struct_size: size_of::<ProposeV1>() as u32,
            abi_version: ABI_VERSION,
            session: model.inner.session.handle,
            model: model.inner.handle,
            anchor_token_id: 4,
            score_rows: 2,
            query_embeddings_bf16: embeddings.as_ptr(),
            query_embedding_bytes: embeddings.len() as u64,
            reserved: [0; 4],
        };
        let mut proposal = ProposalReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            deltafin_provider_dspark_propose_v1(
                &request,
                &mut proposal,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "synthetic propose", &error).expect("proposal");
        assert_eq!(proposal.score_rows, 2);
        assert_eq!(proposal.anchor_position, 2);
        assert!(
            proposal.confidence_logits[..2]
                .iter()
                .all(|v| v.is_finite())
        );
        model.restore(&snapshot).expect("restore");
        assert_eq!(model.state_token_count(), 2);
        assert_eq!(model.inner.state.lock().expect("state").generation, 2);
    }
}
