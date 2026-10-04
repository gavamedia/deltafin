//! Lifetime-safe Rust ownership for the proposal-only Qwen provider ABI.

use std::ffi::{CStr, c_char};
use std::mem::size_of;
use std::sync::Arc;

use crate::error::{DeltafinError, Result};
use crate::provider::{NativeProviderSession, SessionInner};
use crate::qwen_checkpoint::{QwenCheckpoint, QwenVariant};
use crate::sys::fs as sys_fs;

const ABI_VERSION: u32 = 1;
const BF16: u32 = 1;
const TENSOR_COUNT: usize = 310;
const MAX_NEW: usize = 20;
const ERROR_CAPACITY: usize = 2_048;

#[repr(C)]
#[derive(Clone, Copy)]
struct TensorV1 {
    slot: u32,
    scalar_type: u32,
    rank: u32,
    flags: u32,
    shape: [u64; 2],
    data: *const u8,
    data_length: u64,
    reserved: [u64; 2],
}

#[repr(C)]
struct CreateV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    variant: u32,
    tensor_count: u32,
    tensors: *const TensorV1,
    reserved: [u64; 6],
}

#[repr(C)]
struct ReportV1 {
    struct_size: u32,
    abi_version: u32,
    model: u64,
    variant: u32,
    tensor_count: u32,
    reserved: [u64; 5],
}

impl ReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            model: 0,
            variant: 0,
            tensor_count: 0,
            reserved: [0; 5],
        }
    }
}

#[repr(C)]
struct ResourceV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    resource: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct GenerateV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    model: u64,
    input_token_ids: *const u32,
    input_token_count: u64,
    max_new_tokens: u32,
    flags: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct GenerationReportV1 {
    struct_size: u32,
    abi_version: u32,
    generated_token_count: u32,
    flags: u32,
    token_ids: [u32; MAX_NEW],
    probabilities: [f32; MAX_NEW],
    reserved: [u64; 4],
}

impl GenerationReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            generated_token_count: 0,
            flags: 0,
            token_ids: [0; MAX_NEW],
            probabilities: [0.0; MAX_NEW],
            reserved: [0; 4],
        }
    }
}

const _: [(); 64] = [(); size_of::<TensorV1>()];
const _: [(); 80] = [(); size_of::<CreateV1>()];
const _: [(); 64] = [(); size_of::<ReportV1>()];
const _: [(); 64] = [(); size_of::<ResourceV1>()];
const _: [(); 80] = [(); size_of::<GenerateV1>()];
const _: [(); 208] = [(); size_of::<GenerationReportV1>()];

unsafe extern "C" {
    fn deltafin_provider_qwen_create_v1(
        request: *const CreateV1,
        report: *mut ReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_qwen_destroy_v1(
        request: *const ResourceV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_qwen_generate_v1(
        request: *const GenerateV1,
        report: *mut GenerationReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

struct ReadOnlyMap(sys_fs::ReadOnlyMap);

impl ReadOnlyMap {
    fn checkpoint(checkpoint: &QwenCheckpoint) -> Result<Self> {
        let length = usize::try_from(
            checkpoint
                .file()
                .metadata()
                .map_err(|error| DeltafinError::new(format!("stat Qwen checkpoint: {error}")))?
                .len(),
        )
        .map_err(|_| DeltafinError::new("Qwen checkpoint length exceeds usize"))?;
        sys_fs::ReadOnlyMap::map(checkpoint.file(), length)
            .map(Self)
            .map_err(|error| DeltafinError::new(format!("map admitted Qwen checkpoint: {error}")))
    }

    fn pointer(&self, offset: u64, length: u64) -> Result<*const u8> {
        let offset = usize::try_from(offset)
            .map_err(|_| DeltafinError::new("Qwen tensor offset exceeds usize"))?;
        let length = usize::try_from(length)
            .map_err(|_| DeltafinError::new("Qwen tensor length exceeds usize"))?;
        if offset > self.0.len() || length > self.0.len() - offset {
            return Err(DeltafinError::new(
                "Qwen tensor range exceeds mapped checkpoint",
            ));
        }
        // SAFETY: the checked extent lies within this live mapping.
        Ok(unsafe { self.0.address().add(offset) })
    }
}

fn tensor_slot(name: &str) -> Option<u32> {
    if name == "model.embed_tokens.weight" {
        return Some(0);
    }
    if name == "model.norm.weight" {
        return Some(1);
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, suffix) = rest.split_once('.')?;
    let layer: u32 = layer.parse().ok()?;
    if layer >= 28 {
        return None;
    }
    let component = match suffix {
        "input_layernorm.weight" => 0,
        "post_attention_layernorm.weight" => 1,
        "self_attn.q_norm.weight" => 2,
        "self_attn.k_norm.weight" => 3,
        "self_attn.q_proj.weight" => 4,
        "self_attn.k_proj.weight" => 5,
        "self_attn.v_proj.weight" => 6,
        "self_attn.o_proj.weight" => 7,
        "mlp.gate_proj.weight" => 8,
        "mlp.up_proj.weight" => 9,
        "mlp.down_proj.weight" => 10,
        _ => return None,
    };
    Some(2 + layer * 11 + component)
}

#[derive(Debug)]
struct ModelInner {
    session: Arc<SessionInner>,
    handle: u64,
    variant: QwenVariant,
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        let request = ResourceV1 {
            struct_size: size_of::<ResourceV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            resource: self.handle,
            flags: 0,
            reserved0: 0,
            reserved: [0; 4],
        };
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the session Arc keeps the owning native session alive.
        let _ =
            unsafe { deltafin_provider_qwen_destroy_v1(&request, error.as_mut_ptr(), error.len()) };
    }
}

#[derive(Debug, Clone)]
pub struct NativeQwen {
    inner: Arc<ModelInner>,
    // Complete input-plus-proposal capacity whose KV storage participated in
    // the target engine's startup memory proof.  The C++ provider allocates
    // cache rows lazily for each stateless proposal, so checking this before
    // the FFI call prevents a long raw completion from silently exceeding the
    // bytes that residency admission charged.
    context_capacity: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeQwenGeneration {
    pub token_ids: Box<[u32]>,
    pub probabilities: Box<[f32]>,
}

impl NativeQwen {
    pub fn bind(session: &NativeProviderSession, checkpoint: &QwenCheckpoint) -> Result<Self> {
        Self::bind_with_context_capacity(
            session,
            checkpoint,
            usize::try_from(checkpoint.architecture().maximum_position)
                .map_err(|_| DeltafinError::new("Qwen context limit exceeds usize"))?,
        )
    }

    pub fn bind_with_context_capacity(
        session: &NativeProviderSession,
        checkpoint: &QwenCheckpoint,
        context_capacity: usize,
    ) -> Result<Self> {
        let model_capacity = usize::try_from(checkpoint.architecture().maximum_position)
            .map_err(|_| DeltafinError::new("Qwen context limit exceeds usize"))?;
        if context_capacity == 0 || context_capacity > model_capacity {
            return Err(DeltafinError::new(
                "Qwen admitted context capacity is outside its pinned model contract",
            ));
        }
        checkpoint.verify_full_digest()?;
        let mapping = ReadOnlyMap::checkpoint(checkpoint)?;
        let mut slots: [Option<TensorV1>; TENSOR_COUNT] = [None; TENSOR_COUNT];
        for tensor in checkpoint.tensors() {
            let slot = tensor_slot(&tensor.name)
                .ok_or_else(|| DeltafinError::new("Qwen tensor has no fixed provider slot"))?;
            let index = slot as usize;
            if slots[index].is_some() || tensor.shape.len() > 2 {
                return Err(DeltafinError::new("Qwen tensor slot/rank is invalid"));
            }
            let byte_length = tensor.bytes.end - tensor.bytes.start;
            let mut shape = [0_u64; 2];
            shape[..tensor.shape.len()].copy_from_slice(&tensor.shape);
            slots[index] = Some(TensorV1 {
                slot,
                scalar_type: BF16,
                rank: tensor.shape.len() as u32,
                flags: 0,
                shape,
                data: mapping.pointer(tensor.bytes.start, byte_length)?,
                data_length: byte_length,
                reserved: [0; 2],
            });
        }
        let descriptors: Vec<_> = slots
            .into_iter()
            .map(|slot| {
                slot.ok_or_else(|| DeltafinError::new("Qwen provider roster is incomplete"))
            })
            .collect::<Result<_>>()?;
        let lease = session.lease();
        let variant = match checkpoint.variant() {
            QwenVariant::Probe06B => 1,
            QwenVariant::Wide17B => 2,
        };
        let request = CreateV1 {
            struct_size: size_of::<CreateV1>() as u32,
            abi_version: ABI_VERSION,
            session: lease.handle,
            variant,
            tensor_count: TENSOR_COUNT as u32,
            tensors: descriptors.as_ptr(),
            reserved: [0; 6],
        };
        let mut report = ReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: native create synchronously copies every mapped tensor before return.
        let status = unsafe {
            deltafin_provider_qwen_create_v1(&request, &mut report, error.as_mut_ptr(), error.len())
        };
        ffi_status(status, "create native Qwen", &error)?;
        if report.struct_size as usize != size_of::<ReportV1>()
            || report.abi_version != ABI_VERSION
            || report.model == 0
            || report.variant != variant
            || report.tensor_count as usize != TENSOR_COUNT
            || report.reserved != [0; 5]
        {
            if report.model != 0 {
                destroy(lease.handle, report.model);
            }
            return Err(DeltafinError::new("native Qwen create report is invalid"));
        }
        let model = Self {
            inner: Arc::new(ModelInner {
                session: lease,
                handle: report.model,
                variant: checkpoint.variant(),
            }),
            context_capacity,
        };
        checkpoint.validate_live_identity()?;
        Ok(model)
    }

    pub const fn maximum_new_tokens() -> usize {
        MAX_NEW
    }

    pub fn variant(&self) -> QwenVariant {
        self.inner.variant
    }

    pub const fn context_capacity(&self) -> usize {
        self.context_capacity
    }

    pub fn generate(
        &self,
        input_ids: &[u32],
        maximum_new_tokens: usize,
    ) -> Result<NativeQwenGeneration> {
        if input_ids.is_empty() || !(1..=MAX_NEW).contains(&maximum_new_tokens) {
            return Err(DeltafinError::new("Qwen generation bounds are invalid"));
        }
        if !generation_fits_context(input_ids.len(), maximum_new_tokens, self.context_capacity) {
            return Err(DeltafinError::new(
                "Qwen proposal exceeds its memory-admitted context capacity",
            ));
        }
        let request = GenerateV1 {
            struct_size: size_of::<GenerateV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.session.handle,
            model: self.inner.handle,
            input_token_ids: input_ids.as_ptr(),
            input_token_count: input_ids.len() as u64,
            max_new_tokens: maximum_new_tokens as u32,
            flags: 0,
            reserved: [0; 4],
        };
        let mut report = GenerationReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the input slice and report buffers remain live for the synchronous call.
        let status = unsafe {
            deltafin_provider_qwen_generate_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ffi_status(status, "generate native Qwen proposal", &error)?;
        let count = report.generated_token_count as usize;
        if report.struct_size as usize != size_of::<GenerationReportV1>()
            || report.abi_version != ABI_VERSION
            || count > maximum_new_tokens
            || report.flags != 0
            || report.reserved != [0; 4]
            || report.probabilities[..count]
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(DeltafinError::new(
                "native Qwen generation report is invalid",
            ));
        }
        Ok(NativeQwenGeneration {
            token_ids: report.token_ids[..count].into(),
            probabilities: report.probabilities[..count].into(),
        })
    }
}

fn generation_fits_context(input: usize, generated: usize, capacity: usize) -> bool {
    input != 0
        && generated != 0
        && input
            .checked_add(generated)
            .is_some_and(|total| total <= capacity)
}

fn ffi_status(status: i32, operation: &str, error: &[c_char]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: native ffi_guard always NUL-terminates this error buffer.
    let detail = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    Err(DeltafinError::new(format!("{operation}: {detail}")))
}

fn destroy(session: u64, model: u64) {
    let request = ResourceV1 {
        struct_size: size_of::<ResourceV1>() as u32,
        abi_version: ABI_VERSION,
        session,
        resource: model,
        flags: 0,
        reserved0: 0,
        reserved: [0; 4],
    };
    let mut error = [0 as c_char; ERROR_CAPACITY];
    // SAFETY: the owning provider session lease remains live for this call.
    let _ = unsafe { deltafin_provider_qwen_destroy_v1(&request, error.as_mut_ptr(), error.len()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::platform::Device;
    #[cfg(target_os = "macos")]
    use crate::provider::NativeProviderSession;
    #[cfg(target_os = "macos")]
    use std::path::Path;

    #[cfg(target_os = "macos")]
    fn frozen_mps_case(name: &str) -> (Vec<u32>, usize, Vec<u32>, Vec<u32>) {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../provider_gate/qwen_hf_oracle_mps_f16.json"
        ))
        .expect("parse frozen Transformers Qwen oracle");
        assert_eq!(oracle["schema"], "deltafin.qwen-hf-oracle.v1");
        assert_eq!(oracle["transformers"], "4.56.2");
        assert_eq!(oracle["torch"], "2.13.0");
        assert_eq!(oracle["device"], "mps");
        assert_eq!(oracle["dtype"], "float16");
        assert_eq!(oracle["attention_implementation"], "sdpa");
        let row = oracle["cases"]
            .as_array()
            .expect("oracle cases array")
            .iter()
            .find(|row| row["name"].as_str() == Some(name))
            .expect("named oracle case");
        let integers = |field: &str| {
            row[field]
                .as_array()
                .expect("oracle integer array")
                .iter()
                .map(|value| {
                    u32::try_from(value.as_u64().expect("oracle nonnegative integer"))
                        .expect("oracle integer fits u32")
                })
                .collect::<Vec<_>>()
        };
        let probability_bits = row["probability_f32_bits"]
            .as_array()
            .expect("oracle probability bit array")
            .iter()
            .map(|value| {
                u32::from_str_radix(
                    value
                        .as_str()
                        .expect("oracle probability bit string")
                        .strip_prefix("0x")
                        .expect("oracle probability hex prefix"),
                    16,
                )
                .expect("oracle probability bits")
            })
            .collect();
        (
            integers("input_ids"),
            usize::try_from(
                row["maximum_new_tokens"]
                    .as_u64()
                    .expect("oracle generation budget"),
            )
            .expect("oracle generation budget fits usize"),
            integers("generated_ids"),
            probability_bits,
        )
    }

    #[test]
    fn abi_layout_and_complete_slot_formula_are_fixed() {
        assert_eq!(size_of::<TensorV1>(), 64);
        assert_eq!(size_of::<GenerationReportV1>(), 208);
        assert_eq!(tensor_slot("model.embed_tokens.weight"), Some(0));
        assert_eq!(tensor_slot("model.norm.weight"), Some(1));
        assert_eq!(
            tensor_slot("model.layers.0.input_layernorm.weight"),
            Some(2)
        );
        assert_eq!(
            tensor_slot("model.layers.27.mlp.down_proj.weight"),
            Some(309)
        );
        assert_eq!(tensor_slot("lm_head.weight"), None);
    }

    #[test]
    fn proposal_capacity_is_checked_before_provider_allocation() {
        assert!(generation_fits_context(4_000, 20, 4_020));
        assert!(!generation_fits_context(4_001, 20, 4_020));
        assert!(!generation_fits_context(usize::MAX, 1, usize::MAX));
        assert!(!generation_fits_context(0, 1, 1));
        assert!(!generation_fits_context(1, 0, 1));
    }

    /// Physical, opt-in parity canary for the reviewed Qwen provider.  This
    /// loads only the 0.6B proposal model; it never opens K3 weights.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires the optional pinned Qwen checkpoint and an MPS host"]
    fn installed_qwen06_mps_matches_frozen_transformers_sdpa_oracle() {
        let _device = crate::provider::exclusive_mps_device();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let checkpoint = QwenCheckpoint::open(&root, QwenVariant::Probe06B)
            .expect("open installed pinned Qwen 0.6B checkpoint");
        let session = NativeProviderSession::target(Device::Mps).expect("create MPS provider");
        let bind_started = std::time::Instant::now();
        let model = NativeQwen::bind_with_context_capacity(&session, &checkpoint, 4_389)
            .expect("bind native Qwen 0.6B");
        let bind_seconds = bind_started.elapsed().as_secs_f64();
        // These values come from the independent pinned Transformers 4.56.2
        // MPS/FP16/SDPA artifact; the provider under test did not create them.
        for name in [
            "paris",
            "single_token",
            "unicode_multiline_code",
            "long_batched_prefill",
        ] {
            let (input, budget, expected_ids, expected_probability_bits) = frozen_mps_case(name);
            let generation_started = std::time::Instant::now();
            let generation = model
                .generate(&input, budget)
                .expect("generate native Qwen proposal");
            let generation_seconds = generation_started.elapsed().as_secs_f64();
            eprintln!(
                "native Qwen case={name} ids={:?} probabilities={:?} bind_seconds={bind_seconds:.6} generation_seconds={generation_seconds:.6}",
                generation.token_ids, generation.probabilities,
            );
            assert_eq!(generation.token_ids.as_ref(), expected_ids.as_slice());
            assert_eq!(
                generation
                    .probabilities
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_probability_bits,
            );
        }
    }
}
