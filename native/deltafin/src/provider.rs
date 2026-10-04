//! Safe Rust ownership of Deltafin's versioned native provider ABI.
//!
//! The C++ side catches every exception and exposes only fixed-layout C data.
//! No PyObject, pybind handle, C++ container, tensor, or allocator ownership
//! crosses this boundary.

use std::ffi::{CStr, c_char};
use std::mem::{align_of, size_of};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::error::{DeltafinError, Result};
use crate::experts::{
    ExpertStorageLayout, K3_EXPERT_SOURCE_BYTES, K3_EXPERT_TOP_K, K3_SCALE4_BLOB_BYTES,
};
use crate::platform::{Device, ProviderInventory};
use crate::program::{
    SPINE_BUFFER_NONE, SPINE_BUFFER_OTHER, SPINE_ENCODING_RAW_BF16, SPINE_ENCODING_RAW_F32,
    SPINE_ENCODING_ROW_I8_F16_SCALE, SpineTensorDescriptorV1, WeightSlot,
};
#[cfg(test)]
use crate::program::{SPINE_BUFFER_QUANTIZED, SPINE_BUFFER_SCALES};
use crate::storage::{
    BufferKind, BufferLengths, CachePolicy, Extent, LayerBuffers, ReadPlan, Reader,
};

const ABI_VERSION: u32 = 1;
const DEVICE_CPU: u32 = 1;
const DEVICE_MPS: u32 = 2;
const DEVICE_CUDA: u32 = 3;
const CHECK_RMS_FP32: u32 = 1 << 0;
const CHECK_MATMUL_FP32: u32 = 1 << 1;
const CHECK_SOFTMAX_FP32: u32 = 1 << 2;
const CHECK_PACKED_INT8_FP32: u32 = 1 << 3;
const FEATURE_CUDA_MOE: u32 = 1 << 0;
const FEATURE_CUDA_EXACT_BF16: u32 = 1 << 1;
const KNOWN_PROVIDER_FEATURES: u32 = FEATURE_CUDA_MOE | FEATURE_CUDA_EXACT_BF16;
const REQUIRE_PACKED_INT8: u32 = 1 << 0;
const SESSION_SYNTHETIC_SPLIT: u32 = 1 << 0;
const SESSION_SYNTHETIC_KDA: u32 = 1 << 1;
const SESSION_SYNTHETIC_MLA: u32 = 1 << 2;
const MEMORY_TRIM_UNUSED: u32 = 1 << 0;
const MEMORY_ACTIVE_BYTES: u32 = 1 << 0;
const MEMORY_RESERVED_BYTES: u32 = 1 << 1;
const MEMORY_RECOMMENDED_BYTES: u32 = 1 << 2;
const MEMORY_TOTAL_BYTES: u32 = 1 << 3;
const MEMORY_AVAILABLE_BYTES: u32 = 1 << 4;
const MEMORY_ALL_FIELDS: u32 = MEMORY_ACTIVE_BYTES
    | MEMORY_RESERVED_BYTES
    | MEMORY_RECOMMENDED_BYTES
    | MEMORY_TOTAL_BYTES
    | MEMORY_AVAILABLE_BYTES;
const SPINE_COMPONENT_DATA: u32 = 1;
const SPINE_COMPONENT_AUXILIARY: u32 = 2;
const SPINE_SCALAR_I8: u32 = 1;
const SPINE_SCALAR_F32: u32 = 2;
const SPINE_SCALAR_BF16: u32 = 3;
const BIND_SPINE_RETAIN: u32 = 1 << 0;
const BIND_SPINE_ALLOW_BORROW: u32 = 1 << 1;
const SPINE_SOURCE_DETACHED: u32 = 1;
const SPINE_SOURCE_BORROWED: u32 = 2;
const SPINE_SOURCE_SEALED: u32 = 2;
const SPINE_SOURCE_RECLAIMED: u32 = 3;
const SPINE_SOURCE_ABORTED: u32 = 4;
const TARGET_GLOBAL_TAIL: u32 = 1;
const TARGET_GLOBAL_HEAD: u32 = 2;
const TARGET_DENSE_COMPLETE: u32 = 1;
const TARGET_EXPERTS_REQUIRED: u32 = 2;
const TARGET_STATE_ACTIVE: u32 = 1;
const TARGET_STATE_READY_FOR_TAIL: u32 = 3;
const TARGET_STATE_COMMITTED: u32 = 4;
const TARGET_EXPERT_AUTO: u32 = 0;
const TARGET_EXPERT_CPU: u32 = 1;
const TARGET_EXPERT_METAL: u32 = 2;
const TARGET_EXPERT_CUDA: u32 = 3;
const TARGET_EXPERT_RETAIN_METAL_WRAPPERS: u32 = 1 << 0;
const EXPERT_LAYOUT_RAW_V1: u32 = 1;
const EXPERT_LAYOUT_SCALE4_V2: u32 = 2;
const METAL_DESCRIPTOR_ABI_V1: u32 = 1;
const METAL_CAP_RAW_V1: u64 = 1 << EXPERT_LAYOUT_RAW_V1;
const METAL_CAP_SCALE4_V2: u64 = 1 << EXPERT_LAYOUT_SCALE4_V2;
const TARGET_SEQUENCE_PREFILL: u32 = 1;
const TARGET_SEQUENCE_VERIFY: u32 = 2;
const TARGET_SEQUENCE_CAPTURE_DSPARK: u32 = 1 << 0;
const TARGET_SEQUENCE_FULL_COMMIT_ONLY: u32 = 1 << 1;
const TARGET_SEQUENCE_CAPTURE_EAGLE3: u32 = 1 << 2;
const TARGET_SEQUENCE_STATE_ACTIVE: u32 = 1;
const TARGET_SEQUENCE_STATE_WAITING_FOR_EXPERTS: u32 = 2;
const TARGET_SEQUENCE_STATE_READY_FOR_TAIL: u32 = 3;
const TARGET_SEQUENCE_STATE_READY_TO_COMMIT: u32 = 4;
const TARGET_SEQUENCE_STATE_COMMITTED: u32 = 5;
const TARGET_SEQUENCE_STATE_CANCELLED: u32 = 6;
const TARGET_SEQUENCE_STATE_POISONED: u32 = 7;
pub(crate) const ROUTE_TOP_K: usize = 16;

/// The K3 rows a target sequence captures on the device for the proposal
/// model in the engine's draft slot.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ProposalCapture {
    /// DSpark's five post-layer hidden states, BF16 [T, 5H].
    DSpark,
    /// EAGLE-3's AttnRes stream after layers 1, 45 and 89, BF16 [T, 3H].
    Eagle3,
}

impl ProposalCapture {
    const fn flag(self) -> u32 {
        match self {
            Self::DSpark => TARGET_SEQUENCE_CAPTURE_DSPARK,
            Self::Eagle3 => TARGET_SEQUENCE_CAPTURE_EAGLE3,
        }
    }

    pub(crate) const fn columns(self) -> usize {
        match self {
            Self::DSpark => 5 * 7_168,
            Self::Eagle3 => 3 * 7_168,
        }
    }
}
pub(crate) const PILOT_MAX_PREFETCH: usize = 32;
const ROUTE_MAX_POSITIONS: usize = 64;
const ROUTE_MAX_EDGES: usize = ROUTE_TOP_K * ROUTE_MAX_POSITIONS;
const TARGET_SEQUENCE_MAX_EXPERTS: usize = 64;
const TARGET_SEQUENCE_MAX_EXPERTS_V2: usize = 16 * ROUTE_TOP_K;
const TARGET_SEQUENCE_MAX_TILE_ROWS: usize = 16;
const ERROR_CAPACITY: usize = 2048;

/// Complete provider-memory admission bound for the optional CPU/MPS
/// scheduling-only router roster. Sessions which never enable it retain zero
/// bytes for this feature.
pub const TARGET_PILOT_LAYER_CAPACITY: u32 = 92;
pub const TARGET_PILOT_RESERVE_BYTES: u64 = 594_169_856;

#[repr(C)]
struct InventoryV1 {
    struct_size: u32,
    abi_version: u32,
    cpu_available: u32,
    mps_available: u32,
    cuda_device_count: u32,
    provider_features: u32,
    reserved: [u32; 10],
    libtorch_version: [c_char; 32],
}

impl InventoryV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            cpu_available: 0,
            mps_available: 0,
            cuda_device_count: 0,
            provider_features: 0,
            reserved: [0; 10],
            libtorch_version: [0; 32],
        }
    }
}

#[repr(C)]
struct CanaryRequestV1 {
    struct_size: u32,
    abi_version: u32,
    requested_device: u32,
    device_index: u32,
    flags: u32,
    reserved0: u32,
    packed_rows: u64,
    packed_columns: u64,
    reserved: [u64; 6],
}

#[repr(C)]
struct CanaryReportV1 {
    struct_size: u32,
    abi_version: u32,
    selected_device: u32,
    device_index: u32,
    attempted_checks: u32,
    passed_checks: u32,
    required_passed: u32,
    reserved0: u32,
    packed_rows: u64,
    packed_columns: u64,
    reserved: [u64; 6],
    detail: [c_char; 1024],
}

impl CanaryReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            selected_device: 0,
            device_index: 0,
            attempted_checks: 0,
            passed_checks: 0,
            required_passed: 0,
            reserved0: 0,
            packed_rows: 0,
            packed_columns: 0,
            reserved: [0; 6],
            detail: [0; 1024],
        }
    }
}

#[repr(C)]
struct SessionRequestV1 {
    struct_size: u32,
    abi_version: u32,
    requested_device: u32,
    device_index: u32,
    flags: u32,
    max_route_positions: u32,
    synthetic_hidden_columns: u32,
    synthetic_experts: u32,
    reserved: [u64; 6],
}

#[repr(C)]
struct SessionReportV1 {
    struct_size: u32,
    abi_version: u32,
    selected_device: u32,
    device_index: u32,
    session: u64,
    max_route_positions: u32,
    flags: u32,
    reserved: [u64; 6],
}

#[repr(C)]
struct MemoryRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    actions: u32,
    reserved0: u32,
    reserved: [u64; 5],
}

#[repr(C)]
struct MemoryReportV1 {
    struct_size: u32,
    abi_version: u32,
    selected_device: u32,
    device_index: u32,
    available_fields: u32,
    performed_actions: u32,
    reserved0: u32,
    reserved1: u32,
    active_bytes: u64,
    reserved_bytes: u64,
    recommended_bytes: u64,
    total_bytes: u64,
    available_bytes: u64,
    reserved: [u64; 4],
}

impl MemoryReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            selected_device: 0,
            device_index: 0,
            available_fields: 0,
            performed_actions: 0,
            reserved0: 0,
            reserved1: 0,
            active_bytes: 0,
            reserved_bytes: 0,
            recommended_bytes: 0,
            total_bytes: 0,
            available_bytes: 0,
            reserved: [0; 4],
        }
    }
}

impl SessionReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            selected_device: 0,
            device_index: 0,
            session: 0,
            max_route_positions: 0,
            flags: 0,
            reserved: [0; 6],
        }
    }
}

#[repr(C)]
struct TargetPilotEnableReportV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    enabled: u32,
    layer_capacity: u32,
    reserve_bytes: u64,
    reserved: [u64; 4],
}

impl TargetPilotEnableReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            session: 0,
            enabled: 0,
            layer_capacity: 0,
            reserve_bytes: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct MetalExpertLayoutsRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    metal_shader_path: *const c_char,
    metal_shader_path_length: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct MetalExpertLayoutsReportV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    descriptor_abi: u32,
    flags: u32,
    layout_capabilities: u64,
    raw_span_bytes: u64,
    scale4_span_bytes: u64,
    reserved: [u64; 2],
}

impl MetalExpertLayoutsReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            session: 0,
            descriptor_abi: 0,
            flags: 0,
            layout_capabilities: 0,
            raw_span_bytes: 0,
            scale4_span_bytes: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
struct MetalExpertCacheStatsReportV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    calls: u64,
    zero_copy_wraps: u64,
    copies: u64,
    cache_entries: u64,
    bindless: u64,
    reserved: [u64; 2],
}

impl MetalExpertCacheStatsReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            session: 0,
            calls: 0,
            zero_copy_wraps: 0,
            copies: 0,
            cache_entries: 0,
            bindless: 0,
            reserved: [0; 2],
        }
    }
}

const CUDA_CACHE_CAPACITY_EXACT: u32 = 1;
const CUDA_CACHE_RESERVE_BYTES: u32 = 1;

/// Provider-side bound: one cache entry per routed expert edge across the 92
/// MoE layers (`92 * DELTAFIN_PROVIDER_ROUTE_TOP_K_V1`). The runtime rejects
/// larger requests, so the engine clamps before configuring.
pub const CUDA_EXPERT_CACHE_MAX_EXPERTS: u32 = 92 * 16;

/// Explicit, session-frozen CUDA expert-cache budget. Configuration must
/// happen before the first expert plan or availability probe; the provider
/// still clamps the requested capacity against live free VRAM minus
/// `reserve_bytes` when the budget is first exercised.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CudaExpertCachePolicy {
    pub capacity_experts: u32,
    pub reserve_bytes: u64,
}

#[repr(C)]
struct CudaCacheConfigureRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    capacity_mode: u32,
    reserve_mode: u32,
    capacity_experts: u64,
    reserve_value: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 2],
}

#[repr(C)]
struct CudaCacheConfigureReportV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    capacity_mode: u32,
    reserve_mode: u32,
    capacity_experts: u64,
    reserve_value: u64,
    configured: u32,
    flags: u32,
    reserved: [u64; 2],
}

impl CudaCacheConfigureReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            session: 0,
            capacity_mode: 0,
            reserve_mode: 0,
            capacity_experts: 0,
            reserve_value: 0,
            configured: 0,
            flags: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
struct TensorUploadF32V1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    rows: u64,
    columns: u64,
    data: *const f32,
    element_count: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
#[cfg(test)]
struct TensorUploadBf16V1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    rows: u64,
    columns: u64,
    data: *const u8,
    byte_length: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct TensorReportV1 {
    struct_size: u32,
    abi_version: u32,
    tensor: u64,
    rows: u64,
    columns: u64,
    reserved: [u64; 4],
}

impl TensorReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            tensor: 0,
            rows: 0,
            columns: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct TensorReadF32V1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    tensor: u64,
    destination: *mut f32,
    element_capacity: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct CacheCreateF32V1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    rows: u64,
    columns: u64,
    initial_data: *const f32,
    element_count: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct CacheReportV1 {
    struct_size: u32,
    abi_version: u32,
    cache: u64,
    rows: u64,
    columns: u64,
    version: u64,
    reserved: [u64; 3],
}

impl CacheReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            cache: 0,
            rows: 0,
            columns: 0,
            version: 0,
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct CacheReadF32V1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    cache: u64,
    destination: *mut f32,
    element_capacity: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct ResourceRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    resource: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 4],
}

impl ResourceRequestV1 {
    fn new(session: u64, resource: u64) -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: ABI_VERSION,
            session,
            resource,
            flags: 0,
            reserved0: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct PrepareLayerRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    hidden: u64,
    cache: u64,
    layer_index: u32,
    flags: u32,
    reserved: [u64; 5],
}

#[repr(C)]
#[derive(Debug)]
struct RouteMailboxV1 {
    struct_size: u32,
    abi_version: u32,
    ticket: u64,
    positions: u32,
    top_k: u32,
    edge_count: u32,
    flags: u32,
    hidden_columns: u64,
    cache_version: u64,
    reserved: [u64; 4],
    ordered_experts: [u16; ROUTE_MAX_EDGES],
    ordered_weight_bits: [u32; ROUTE_MAX_EDGES],
}

impl RouteMailboxV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            ticket: 0,
            positions: 0,
            top_k: 0,
            edge_count: 0,
            flags: 0,
            hidden_columns: 0,
            cache_version: 0,
            reserved: [0; 4],
            ordered_experts: [0; ROUTE_MAX_EDGES],
            ordered_weight_bits: [0; ROUTE_MAX_EDGES],
        }
    }
}

#[repr(C)]
struct FinishLayerRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    ticket: u64,
    expert_output: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 5],
}

#[repr(C)]
struct FinishLayerReportV1 {
    struct_size: u32,
    abi_version: u32,
    output: u64,
    positions: u64,
    hidden_columns: u64,
    committed_cache_version: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct BindSpineLayerRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    layer_index: u32,
    flags: u32,
    generation: u64,
    descriptors: *const SpineTensorDescriptorV1,
    descriptor_count: u64,
    quantized: *const u8,
    quantized_length: u64,
    scales: *const u8,
    scales_length: u64,
    other: *const u8,
    other_length: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct BindSpineLayerReportV1 {
    struct_size: u32,
    abi_version: u32,
    layer_index: u32,
    tensor_count: u32,
    generation: u64,
    quantized_tensor_count: u32,
    raw_tensor_count: u32,
    quantized_bytes: u64,
    scales_bytes: u64,
    other_bytes: u64,
    resident_storage_bytes: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct BindSpineLayerRequestV2 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    layer_index: u32,
    flags: u32,
    generation: u64,
    descriptors: *const SpineTensorDescriptorV1,
    descriptor_count: u64,
    quantized: *const u8,
    quantized_length: u64,
    quantized_allocation_length: u64,
    scales: *const u8,
    scales_length: u64,
    scales_allocation_length: u64,
    other: *const u8,
    other_length: u64,
    other_allocation_length: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct BindSpineLayerReportV2 {
    struct_size: u32,
    abi_version: u32,
    layer_index: u32,
    tensor_count: u32,
    generation: u64,
    quantized_tensor_count: u32,
    raw_tensor_count: u32,
    quantized_bytes: u64,
    scales_bytes: u64,
    other_bytes: u64,
    resident_storage_bytes: u64,
    source_use_kind: u32,
    borrowed_tensor_count: u32,
    source_use: u64,
    borrowed_source_bytes: u64,
    reserved: [u64; 3],
}

impl BindSpineLayerReportV2 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            layer_index: 0,
            tensor_count: 0,
            generation: 0,
            quantized_tensor_count: 0,
            raw_tensor_count: 0,
            quantized_bytes: 0,
            scales_bytes: 0,
            other_bytes: 0,
            resident_storage_bytes: 0,
            source_use_kind: 0,
            borrowed_tensor_count: 0,
            source_use: 0,
            borrowed_source_bytes: 0,
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct SpineSourceUseRequestV2 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    source_use: u64,
    generation: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct SpineSourceUseReportV2 {
    struct_size: u32,
    abi_version: u32,
    source_use: u64,
    generation: u64,
    state: u32,
    ready: u32,
    reserved: [u64; 4],
}

impl SpineSourceUseReportV2 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            source_use: 0,
            generation: 0,
            state: 0,
            ready: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct SpineTensorReadF32V1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    generation: u64,
    slot: u32,
    component: u32,
    destination: *mut f32,
    element_capacity: u64,
    flags: u32,
    layer_index: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct SpineTensorReadReportV1 {
    struct_size: u32,
    abi_version: u32,
    stored_scalar_type: u32,
    rank: u32,
    element_count: u64,
    shape: [u64; 8],
    reserved: [u64; 1],
}

impl SpineTensorReadReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            stored_scalar_type: 0,
            rank: 0,
            element_count: 0,
            shape: [0; 8],
            reserved: [0; 1],
        }
    }
}

#[repr(C)]
struct KdaCacheCreateV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    layer_index: u32,
    flags: u32,
    reserved: [u64; 5],
}

#[repr(C)]
struct KdaCacheReportV1 {
    struct_size: u32,
    abi_version: u32,
    cache: u64,
    layer_index: u32,
    flags: u32,
    version: u64,
    convolution_elements: u64,
    recurrent_elements: u64,
    reserved: [u64; 2],
}

impl KdaCacheReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            cache: 0,
            layer_index: 0,
            flags: 0,
            version: 0,
            convolution_elements: 0,
            recurrent_elements: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
struct KdaDecodeRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    hidden: u64,
    cache: u64,
    layer_index: u32,
    flags: u32,
    spine_generation: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct KdaDecodeReportV1 {
    struct_size: u32,
    abi_version: u32,
    output: u64,
    ticket: u64,
    cache_version: u64,
    spine_generation: u64,
    rows: u64,
    columns: u64,
    reserved: [u64; 3],
}

impl KdaDecodeReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            output: 0,
            ticket: 0,
            cache_version: 0,
            spine_generation: 0,
            rows: 0,
            columns: 0,
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct KdaCommitReportV1 {
    struct_size: u32,
    abi_version: u32,
    cache: u64,
    committed_version: u64,
    layer_index: u32,
    flags: u32,
    reserved: [u64; 4],
}

impl KdaCommitReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            cache: 0,
            committed_version: 0,
            layer_index: 0,
            flags: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct MlaCacheCreateV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    layer_index: u32,
    flags: u32,
    reserved: [u64; 5],
}

#[repr(C)]
struct MlaCacheReportV1 {
    struct_size: u32,
    abi_version: u32,
    cache: u64,
    layer_index: u32,
    flags: u32,
    version: u64,
    length: u64,
    capacity: u64,
    reserved: [u64; 2],
}

impl MlaCacheReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            cache: 0,
            layer_index: 0,
            flags: 0,
            version: 0,
            length: 0,
            capacity: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
struct MlaDecodeRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    hidden: u64,
    cache: u64,
    layer_index: u32,
    flags: u32,
    spine_generation: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct MlaDecodeReportV1 {
    struct_size: u32,
    abi_version: u32,
    output: u64,
    ticket: u64,
    cache_version: u64,
    spine_generation: u64,
    rows: u64,
    columns: u64,
    proposed_length: u64,
    proposed_capacity: u64,
    input_bundle_rows: u64,
    reserved: [u64; 2],
}

impl MlaDecodeReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            output: 0,
            ticket: 0,
            cache_version: 0,
            spine_generation: 0,
            rows: 0,
            columns: 0,
            proposed_length: 0,
            proposed_capacity: 0,
            input_bundle_rows: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
struct MlaCommitReportV1 {
    struct_size: u32,
    abi_version: u32,
    cache: u64,
    committed_version: u64,
    layer_index: u32,
    flags: u32,
    committed_length: u64,
    capacity: u64,
    reserved: [u64; 2],
}

#[repr(C)]
struct BindTargetGlobalsRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    group: u32,
    flags: u32,
    descriptors: *const SpineTensorDescriptorV1,
    descriptor_count: u64,
    quantized: *const u8,
    quantized_length: u64,
    scales: *const u8,
    scales_length: u64,
    other: *const u8,
    other_length: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct BindTargetGlobalsReportV1 {
    struct_size: u32,
    abi_version: u32,
    group: u32,
    tensor_count: u32,
    quantized_tensor_count: u32,
    raw_tensor_count: u32,
    quantized_bytes: u64,
    scales_bytes: u64,
    other_bytes: u64,
    resident_storage_bytes: u64,
    groups_ready: u32,
    flags: u32,
    reserved: [u64; 4],
}

impl BindTargetGlobalsReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            group: 0,
            tensor_count: 0,
            quantized_tensor_count: 0,
            raw_tensor_count: 0,
            quantized_bytes: 0,
            scales_bytes: 0,
            other_bytes: 0,
            resident_storage_bytes: 0,
            groups_ready: 0,
            flags: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct TargetBeginRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    hidden: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct TargetBeginBf16RequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    data: *const u8,
    byte_length: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct TargetBeginReportV1 {
    struct_size: u32,
    abi_version: u32,
    position: u64,
    next_layer: u32,
    state: u32,
    kda_cache_count: u32,
    mla_cache_count: u32,
    reserved: [u64; 4],
}

impl TargetBeginReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            position: 0,
            next_layer: 0,
            state: 0,
            kda_cache_count: 0,
            mla_cache_count: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct TargetPrepareRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    position: u64,
    spine_generation: u64,
    layer_index: u32,
    flags: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct TargetPrepareReportV1 {
    struct_size: u32,
    abi_version: u32,
    position: u64,
    spine_generation: u64,
    layer_index: u32,
    kind: u32,
    next_layer: u32,
    top_k: u32,
    ordered_experts: [u16; ROUTE_TOP_K],
    ordered_weight_bits: [u32; ROUTE_TOP_K],
    reserved: [u64; 3],
}

impl TargetPrepareReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            position: 0,
            spine_generation: 0,
            layer_index: 0,
            kind: 0,
            next_layer: 0,
            top_k: 0,
            ordered_experts: [0; ROUTE_TOP_K],
            ordered_weight_bits: [0; ROUTE_TOP_K],
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct TargetFinishExpertsRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    position: u64,
    spine_generation: u64,
    layer_index: u32,
    expert_backend: u32,
    cpu_threads: u32,
    expert_count: u32,
    expert_ids: [u16; ROUTE_TOP_K],
    expert_major_bytes: *const u8,
    expert_major_length: u64,
    metal_shader_path: *const c_char,
    metal_shader_path_length: u64,
    flags: u32,
    expert_layout: u32,
    expert_span_bytes: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct TargetFinishExpertsReportV1 {
    struct_size: u32,
    abi_version: u32,
    position: u64,
    completed_layer: u32,
    next_layer: u32,
    state: u32,
    flags: u32,
    reserved: [u64; 4],
}

impl TargetFinishExpertsReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            position: 0,
            completed_layer: 0,
            next_layer: 0,
            state: 0,
            flags: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct TargetGreedyReportV1 {
    struct_size: u32,
    abi_version: u32,
    position: u64,
    token_id: u32,
    state: u32,
    committed_positions: u64,
    reserved: [u64; 4],
}

impl TargetGreedyReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            position: 0,
            token_id: 0,
            state: 0,
            committed_positions: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct TargetSequenceBeginBf16RequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    data: *const u8,
    byte_length: u64,
    positions: u32,
    mode: u32,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct TargetSequenceBeginReportV1 {
    struct_size: u32,
    abi_version: u32,
    sequence: u64,
    positions: u32,
    mode: u32,
    next_layer: u32,
    state: u32,
    kda_cache_count: u32,
    mla_cache_count: u32,
    reserved: [u64; 3],
}

impl TargetSequenceBeginReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            sequence: 0,
            positions: 0,
            mode: 0,
            next_layer: 0,
            state: 0,
            kda_cache_count: 0,
            mla_cache_count: 0,
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct TargetSequencePrepareRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    flags: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct TargetSequencePrepareReportV1 {
    struct_size: u32,
    abi_version: u32,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    kind: u32,
    next_layer: u32,
    positions: u32,
    top_k: u32,
    flags: u32,
    ordered_experts: [u16; ROUTE_MAX_EDGES],
    ordered_weight_bits: [u32; ROUTE_MAX_EDGES],
    reserved: [u64; 4],
}

#[repr(C)]
struct TargetSequencePrefetchHintReportV1 {
    struct_size: u32,
    abi_version: u32,
    sequence: u64,
    source_layer: u32,
    target_layer: u32,
    expert_count: u32,
    flags: u32,
    expert_ids: [u16; PILOT_MAX_PREFETCH],
    reserved: [u64; 4],
}

impl TargetSequencePrefetchHintReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            sequence: 0,
            source_layer: 0,
            target_layer: 0,
            expert_count: 0,
            flags: 0,
            expert_ids: [0; PILOT_MAX_PREFETCH],
            reserved: [0; 4],
        }
    }
}

impl TargetSequencePrepareReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            sequence: 0,
            spine_generation: 0,
            layer_index: 0,
            kind: 0,
            next_layer: 0,
            positions: 0,
            top_k: 0,
            flags: 0,
            ordered_experts: [0; ROUTE_MAX_EDGES],
            ordered_weight_bits: [0; ROUTE_MAX_EDGES],
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct TargetSequenceFinishExpertsRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    row_count: u32,
    expert_backend: u32,
    cpu_threads: u32,
    expert_count: u32,
    flags: u32,
    expert_layout: u32,
    expert_ids: [u16; TARGET_SEQUENCE_MAX_EXPERTS],
    expert_major_bytes: *const u8,
    expert_major_length: u64,
    metal_shader_path: *const c_char,
    metal_shader_path_length: u64,
    expert_span_bytes: u64,
    reserved: [u64; 3],
}

#[repr(C)]
struct TargetSequenceFinishExpertSpansRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    row_count: u32,
    expert_backend: u32,
    cpu_threads: u32,
    expert_count: u32,
    flags: u32,
    expert_layout: u32,
    expert_ids: [u16; TARGET_SEQUENCE_MAX_EXPERTS],
    expert_span_pointers: [*const u8; TARGET_SEQUENCE_MAX_EXPERTS],
    metal_shader_path: *const c_char,
    metal_shader_path_length: u64,
    expert_span_bytes: u64,
    reserved: [u64; 4],
}

/// Expert early drain. Carries only the experts whose bytes have landed, so
/// the provider can start their independent work while the rest of the layer
/// is still being read. Advisory: a refusal is reported, never raised, and
/// anything accepted is still recomputed by the finish call unless the staged
/// state lines up exactly. The reduction stays entirely inside the finish
/// call, in the router's order.
#[repr(C)]
struct TargetSequenceStageExpertSpansRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    row_count: u32,
    expert_backend: u32,
    cpu_threads: u32,
    expert_count: u32,
    flags: u32,
    expert_layout: u32,
    expert_ids: [u16; TARGET_SEQUENCE_MAX_EXPERTS],
    expert_span_pointers: [*const u8; TARGET_SEQUENCE_MAX_EXPERTS],
    metal_shader_path: *const c_char,
    metal_shader_path_length: u64,
    expert_span_bytes: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Default)]
struct TargetSequenceStageExpertSpansReportV1 {
    struct_size: u32,
    abi_version: u32,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    expert_count: u32,
    staged: u32,
    reserved: [u64; 2],
}

/// Additive wide-union request. V1 remains the byte-for-byte hot ABI for up
/// to 64 experts; V2 replaces its fixed arrays with synchronously borrowed
/// pointers and is used only for exact 65..=256 CPU/Metal route unions.
#[repr(C)]
struct TargetSequenceFinishExpertsRequestV2 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    row_count: u32,
    expert_backend: u32,
    cpu_threads: u32,
    expert_count: u32,
    flags: u32,
    expert_layout: u32,
    expert_ids: *const u16,
    expert_ids_length: u64,
    expert_major_bytes: *const u8,
    expert_major_length: u64,
    expert_span_pointers: *const *const u8,
    expert_span_pointer_count: u64,
    metal_shader_path: *const c_char,
    metal_shader_path_length: u64,
    expert_span_bytes: u64,
    reserved: [u64; 8],
}

#[repr(C)]
struct TargetSequencePlanExpertsRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    row_count: u32,
    expert_backend: u32,
    cpu_threads: u32,
    expert_count: u32,
    flags: u32,
    expert_ids: [u16; TARGET_SEQUENCE_MAX_EXPERTS],
    metal_shader_path: *const c_char,
    metal_shader_path_length: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct TargetSequencePlanExpertsReportV1 {
    struct_size: u32,
    abi_version: u32,
    plan: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    row_count: u32,
    effective_backend: u32,
    missing_count: u32,
    cache_capacity_experts: u32,
    residency_enabled: u32,
    flags: u32,
    missing_experts: [u16; TARGET_SEQUENCE_MAX_EXPERTS],
    reserved: [u64; 3],
}

impl TargetSequencePlanExpertsReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            plan: 0,
            spine_generation: 0,
            layer_index: 0,
            first_row: 0,
            row_count: 0,
            effective_backend: 0,
            missing_count: 0,
            cache_capacity_experts: 0,
            residency_enabled: 0,
            flags: 0,
            missing_experts: [0; TARGET_SEQUENCE_MAX_EXPERTS],
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct TargetSequenceFinishPlannedExpertsRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    sequence: u64,
    plan: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    row_count: u32,
    missing_count: u32,
    flags: u32,
    missing_experts: [u16; TARGET_SEQUENCE_MAX_EXPERTS],
    expert_major_bytes: *const u8,
    expert_major_length: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct TargetSequenceFinishExpertsReportV1 {
    struct_size: u32,
    abi_version: u32,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: u32,
    row_count: u32,
    next_expert_row: u32,
    state: u32,
    flags: u32,
    reserved: [u64; 2],
}

impl TargetSequenceFinishExpertsReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            sequence: 0,
            spine_generation: 0,
            layer_index: 0,
            first_row: 0,
            row_count: 0,
            next_expert_row: 0,
            state: 0,
            flags: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
struct TargetSequenceTailReportV1 {
    struct_size: u32,
    abi_version: u32,
    sequence: u64,
    token_count: u32,
    state: u32,
    tail_rows: u32,
    tail_provider_dispatches: u32,
    token_ids: [u32; ROUTE_MAX_POSITIONS],
    reserved: [u64; 4],
}

impl TargetSequenceTailReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            sequence: 0,
            token_count: 0,
            state: 0,
            tail_rows: 0,
            tail_provider_dispatches: 0,
            token_ids: [0; ROUTE_MAX_POSITIONS],
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct TargetSequenceCommitRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    sequence: u64,
    positions: u32,
    flags: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct TargetSequenceCommitReportV1 {
    struct_size: u32,
    abi_version: u32,
    sequence: u64,
    committed_positions: u64,
    session_committed_positions: u64,
    state: u32,
    flags: u32,
    reserved: [u64; 3],
}

impl TargetSequenceCommitReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            sequence: 0,
            committed_positions: 0,
            session_committed_positions: 0,
            state: 0,
            flags: 0,
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct TargetStateReportV1 {
    struct_size: u32,
    abi_version: u32,
    committed_positions: u64,
    cache_generation: u64,
    active_branch: u64,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 2],
}

impl TargetStateReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            committed_positions: 0,
            cache_generation: 0,
            active_branch: 0,
            flags: 0,
            reserved0: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
struct TargetStateBranchRequestV1 {
    struct_size: u32,
    abi_version: u32,
    session: u64,
    expected_committed_positions: u64,
    expected_cache_generation: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct TargetSequenceStatsReportV1 {
    struct_size: u32,
    abi_version: u32,
    sequence: u64,
    positions: u64,
    streamed_layer_passes: u64,
    attention_rows: u64,
    expert_row_requests: u64,
    expert_rows_completed: u64,
    expert_tiles_completed: u64,
    tail_rows: u64,
    tail_provider_dispatches: u64,
    maximum_live_streamed_layers: u64,
    maximum_experts_per_request: u64,
    maximum_positions_per_expert_tile: u64,
    staged_kda_storage_bytes: u64,
    verify_snapshot_bytes: u64,
    projected_mla_storage_bytes: u64,
    additional_mla_storage_bytes: u64,
    mode: u32,
    state: u32,
    reserved: [u64; 2],
}

impl TargetSequenceStatsReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            sequence: 0,
            positions: 0,
            streamed_layer_passes: 0,
            attention_rows: 0,
            expert_row_requests: 0,
            expert_rows_completed: 0,
            expert_tiles_completed: 0,
            tail_rows: 0,
            tail_provider_dispatches: 0,
            maximum_live_streamed_layers: 0,
            maximum_experts_per_request: 0,
            maximum_positions_per_expert_tile: 0,
            staged_kda_storage_bytes: 0,
            verify_snapshot_bytes: 0,
            projected_mla_storage_bytes: 0,
            additional_mla_storage_bytes: 0,
            mode: 0,
            state: 0,
            reserved: [0; 2],
        }
    }
}

impl MlaCommitReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            cache: 0,
            committed_version: 0,
            layer_index: 0,
            flags: 0,
            committed_length: 0,
            capacity: 0,
            reserved: [0; 2],
        }
    }
}

impl FinishLayerReportV1 {
    fn request() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            output: 0,
            positions: 0,
            hidden_columns: 0,
            committed_cache_version: 0,
            reserved: [0; 5],
        }
    }
}

const _: [(); 80] = [(); size_of::<SessionRequestV1>()];
const _: [(); 80] = [(); size_of::<SessionReportV1>()];
const _: [(); 64] = [(); size_of::<MemoryRequestV1>()];
const _: [(); 104] = [(); size_of::<MemoryReportV1>()];
const _: [(); 32] = [(); std::mem::offset_of!(MemoryReportV1, active_bytes)];
const _: [(); 64] = [(); size_of::<TargetPilotEnableReportV1>()];
const _: [(); 24] = [(); std::mem::offset_of!(TargetPilotEnableReportV1, reserve_bytes)];
const _: [(); 32] = [(); std::mem::offset_of!(TargetPilotEnableReportV1, reserved)];
const _: [(); 80] = [(); size_of::<TensorUploadF32V1>()];
#[cfg(test)]
const _: [(); 80] = [(); size_of::<TensorUploadBf16V1>()];
const _: [(); 64] = [(); size_of::<TensorReportV1>()];
const _: [(); 72] = [(); size_of::<TensorReadF32V1>()];
const _: [(); 80] = [(); size_of::<CacheCreateF32V1>()];
const _: [(); 64] = [(); size_of::<CacheReportV1>()];
const _: [(); 72] = [(); size_of::<CacheReadF32V1>()];
const _: [(); 64] = [(); size_of::<ResourceRequestV1>()];
const _: [(); 80] = [(); size_of::<PrepareLayerRequestV1>()];
const _: [(); 6224] = [(); size_of::<RouteMailboxV1>()];
const _: [(); 80] = [(); size_of::<FinishLayerRequestV1>()];
const _: [(); 80] = [(); size_of::<FinishLayerReportV1>()];
const _: [(); 152] = [(); size_of::<SpineTensorDescriptorV1>()];
const _: [(); 128] = [(); size_of::<BindSpineLayerRequestV1>()];
const _: [(); 96] = [(); size_of::<BindSpineLayerReportV1>()];
const _: [(); 160] = [(); size_of::<BindSpineLayerRequestV2>()];
const _: [(); 112] = [(); size_of::<BindSpineLayerReportV2>()];
const _: [(); 64] = [(); size_of::<SpineSourceUseRequestV2>()];
const _: [(); 64] = [(); size_of::<SpineSourceUseReportV2>()];
const _: [(); 80] = [(); size_of::<SpineTensorReadF32V1>()];
const _: [(); 96] = [(); size_of::<SpineTensorReadReportV1>()];
const _: [(); 64] = [(); size_of::<KdaCacheCreateV1>()];
const _: [(); 64] = [(); size_of::<KdaCacheReportV1>()];
const _: [(); 80] = [(); size_of::<KdaDecodeRequestV1>()];
const _: [(); 80] = [(); size_of::<KdaDecodeReportV1>()];
const _: [(); 64] = [(); size_of::<KdaCommitReportV1>()];
const _: [(); 64] = [(); size_of::<MlaCacheCreateV1>()];
const _: [(); 64] = [(); size_of::<MlaCacheReportV1>()];
const _: [(); 80] = [(); size_of::<MlaDecodeRequestV1>()];
const _: [(); 96] = [(); size_of::<MlaDecodeReportV1>()];
const _: [(); 64] = [(); size_of::<MlaCommitReportV1>()];
const _: [(); 128] = [(); size_of::<BindTargetGlobalsRequestV1>()];
const _: [(); 96] = [(); size_of::<BindTargetGlobalsReportV1>()];
const _: [(); 64] = [(); size_of::<TargetBeginRequestV1>()];
const _: [(); 64] = [(); size_of::<TargetBeginBf16RequestV1>()];
const _: [(); 64] = [(); size_of::<TargetBeginReportV1>()];
const _: [(); 64] = [(); size_of::<MetalExpertLayoutsRequestV1>()];
const _: [(); 64] = [(); size_of::<MetalExpertLayoutsReportV1>()];
const _: [(); 72] = [(); size_of::<MetalExpertCacheStatsReportV1>()];
const _: [(); 64] = [(); size_of::<TargetPrepareRequestV1>()];
const _: [(); 160] = [(); size_of::<TargetPrepareReportV1>()];
const _: [(); 160] = [(); size_of::<TargetFinishExpertsRequestV1>()];
const _: [(); 64] = [(); size_of::<TargetFinishExpertsReportV1>()];
const _: [(); 64] = [(); size_of::<TargetGreedyReportV1>()];
const _: [(); 80] = [(); size_of::<TargetSequenceBeginBf16RequestV1>()];
const _: [(); 64] = [(); size_of::<TargetSequenceBeginReportV1>()];
const _: [(); 64] = [(); size_of::<TargetSequencePrepareRequestV1>()];
const _: [(); 6224] = [(); size_of::<TargetSequencePrepareReportV1>()];
const _: [(); 128] = [(); size_of::<TargetSequencePrefetchHintReportV1>()];
const _: [(); 256] = [(); size_of::<TargetSequenceFinishExpertsRequestV1>()];
const _: [(); 760] = [(); size_of::<TargetSequenceFinishExpertSpansRequestV1>()];
const _: [(); 200] = [(); size_of::<TargetSequenceFinishExpertsRequestV2>()];
const _: [(); 240] = [(); size_of::<TargetSequencePlanExpertsRequestV1>()];
const _: [(); 208] = [(); size_of::<TargetSequencePlanExpertsReportV1>()];
const _: [(); 240] = [(); size_of::<TargetSequenceFinishPlannedExpertsRequestV1>()];
const _: [(); 64] = [(); size_of::<TargetSequenceFinishExpertsReportV1>()];
const _: [(); 320] = [(); size_of::<TargetSequenceTailReportV1>()];
const _: [(); 64] = [(); size_of::<TargetSequenceCommitRequestV1>()];
const _: [(); 64] = [(); size_of::<TargetSequenceCommitReportV1>()];
const _: [(); 56] = [(); size_of::<TargetStateReportV1>()];
const _: [(); 64] = [(); size_of::<TargetStateBranchRequestV1>()];
const _: [(); 160] = [(); size_of::<TargetSequenceStatsReportV1>()];
const _: [(); 8] = [(); align_of::<TargetSequencePrepareReportV1>()];
const _: [(); 48] = [(); std::mem::offset_of!(TargetSequencePrepareReportV1, ordered_experts)];
const _: [(); 2096] =
    [(); std::mem::offset_of!(TargetSequencePrepareReportV1, ordered_weight_bits)];
const _: [(); 64] = [(); std::mem::offset_of!(TargetSequenceFinishExpertsRequestV1, expert_ids)];
const _: [(); 192] =
    [(); std::mem::offset_of!(TargetSequenceFinishExpertsRequestV1, expert_major_bytes)];
const _: [(); 192] = [(); std::mem::offset_of!(
    TargetSequenceFinishExpertSpansRequestV1,
    expert_span_pointers
)];
const _: [(); 64] = [(); std::mem::offset_of!(TargetSequenceFinishExpertsRequestV2, expert_ids)];
const _: [(); 80] =
    [(); std::mem::offset_of!(TargetSequenceFinishExpertsRequestV2, expert_major_bytes)];
const _: [(); 96] =
    [(); std::mem::offset_of!(TargetSequenceFinishExpertsRequestV2, expert_span_pointers)];
const _: [(); 136] = [(); std::mem::offset_of!(TargetSequenceFinishExpertsRequestV2, reserved)];
const _: [(); 60] = [(); std::mem::offset_of!(TargetSequencePlanExpertsRequestV1, expert_ids)];
const _: [(); 192] =
    [(); std::mem::offset_of!(TargetSequencePlanExpertsRequestV1, metal_shader_path)];
const _: [(); 56] = [(); std::mem::offset_of!(TargetSequencePlanExpertsReportV1, missing_experts)];
const _: [(); 60] =
    [(); std::mem::offset_of!(TargetSequenceFinishPlannedExpertsRequestV1, missing_experts)];
const _: [(); 192] = [(); std::mem::offset_of!(
    TargetSequenceFinishPlannedExpertsRequestV1,
    expert_major_bytes
)];
const _: [(); 24] = [(); std::mem::offset_of!(TargetSequenceFinishExpertsReportV1, layer_index)];
const _: [(); 64] = [(); size_of::<CudaCacheConfigureRequestV1>()];
const _: [(); 24] = [(); std::mem::offset_of!(CudaCacheConfigureRequestV1, capacity_experts)];
const _: [(); 40] = [(); std::mem::offset_of!(CudaCacheConfigureRequestV1, flags)];
const _: [(); 64] = [(); size_of::<CudaCacheConfigureReportV1>()];
const _: [(); 40] = [(); std::mem::offset_of!(CudaCacheConfigureReportV1, configured)];

unsafe extern "C" {
    #[cfg(all(test, target_os = "macos"))]
    fn k3_metal_embedded_library_cycle_test_v1(cycles: i32) -> i32;
    #[cfg(all(test, target_os = "macos"))]
    fn deltafin_route_mailbox_metallib_cycle_test_v1(cycles: i32) -> i32;
    fn deltafin_provider_abi_version() -> u32;
    fn deltafin_provider_inventory_v1(
        inventory: *mut InventoryV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_canary_v1(
        request: *const CanaryRequestV1,
        report: *mut CanaryReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_session_create_v1(
        request: *const SessionRequestV1,
        report: *mut SessionReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_memory_snapshot_v1(
        request: *const MemoryRequestV1,
        report: *mut MemoryReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_pilot_enable_v1(
        request: *const ResourceRequestV1,
        report: *mut TargetPilotEnableReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_session_destroy_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_metal_expert_layouts_v1(
        request: *const MetalExpertLayoutsRequestV1,
        report: *mut MetalExpertLayoutsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_metal_expert_cache_flush_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_metal_expert_cache_stats_v1(
        request: *const ResourceRequestV1,
        report: *mut MetalExpertCacheStatsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_cuda_cache_configure_v1(
        request: *const CudaCacheConfigureRequestV1,
        report: *mut CudaCacheConfigureReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_tensor_upload_f32_v1(
        request: *const TensorUploadF32V1,
        report: *mut TensorReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    #[cfg(test)]
    fn deltafin_provider_tensor_upload_bf16_v1(
        request: *const TensorUploadBf16V1,
        report: *mut TensorReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_tensor_read_f32_v1(
        request: *const TensorReadF32V1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_tensor_release_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_cache_create_f32_v1(
        request: *const CacheCreateF32V1,
        report: *mut CacheReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_cache_read_f32_v1(
        request: *const CacheReadF32V1,
        report: *mut CacheReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_cache_release_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_prepare_layer_v1(
        request: *const PrepareLayerRequestV1,
        mailbox: *mut RouteMailboxV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_finish_layer_v1(
        request: *const FinishLayerRequestV1,
        report: *mut FinishLayerReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_ticket_release_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_bind_spine_layer_v2(
        request: *const BindSpineLayerRequestV2,
        report: *mut BindSpineLayerReportV2,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_spine_source_use_seal_v2(
        request: *const SpineSourceUseRequestV2,
        report: *mut SpineSourceUseReportV2,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_spine_source_use_try_reclaim_v2(
        request: *const SpineSourceUseRequestV2,
        report: *mut SpineSourceUseReportV2,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_spine_source_use_abort_v2(
        request: *const SpineSourceUseRequestV2,
        report: *mut SpineSourceUseReportV2,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_spine_tensor_read_f32_v1(
        request: *const SpineTensorReadF32V1,
        report: *mut SpineTensorReadReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_kda_cache_create_v1(
        request: *const KdaCacheCreateV1,
        report: *mut KdaCacheReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_kda_cache_release_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_kda_decode_v1(
        request: *const KdaDecodeRequestV1,
        report: *mut KdaDecodeReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_kda_commit_v1(
        request: *const ResourceRequestV1,
        report: *mut KdaCommitReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_kda_ticket_release_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_mla_cache_create_v1(
        request: *const MlaCacheCreateV1,
        report: *mut MlaCacheReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_mla_cache_release_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_mla_decode_v1(
        request: *const MlaDecodeRequestV1,
        report: *mut MlaDecodeReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_mla_commit_v1(
        request: *const ResourceRequestV1,
        report: *mut MlaCommitReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_mla_ticket_release_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_bind_target_globals_v1(
        request: *const BindTargetGlobalsRequestV1,
        report: *mut BindTargetGlobalsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_begin_v1(
        request: *const TargetBeginRequestV1,
        report: *mut TargetBeginReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_begin_bf16_v1(
        request: *const TargetBeginBf16RequestV1,
        report: *mut TargetBeginReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_prepare_v1(
        request: *const TargetPrepareRequestV1,
        report: *mut TargetPrepareReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_finish_experts_v1(
        request: *const TargetFinishExpertsRequestV1,
        report: *mut TargetFinishExpertsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_finish_greedy_v1(
        request: *const ResourceRequestV1,
        report: *mut TargetGreedyReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_cancel_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_state_reset_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_state_inspect_v1(
        request: *const ResourceRequestV1,
        report: *mut TargetStateReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_state_branch_begin_v1(
        request: *const TargetStateBranchRequestV1,
        report: *mut TargetStateReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_state_branch_publish_v1(
        request: *const ResourceRequestV1,
        report: *mut TargetStateReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_state_branch_discard_v1(
        request: *const ResourceRequestV1,
        report: *mut TargetStateReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_begin_bf16_v1(
        request: *const TargetSequenceBeginBf16RequestV1,
        report: *mut TargetSequenceBeginReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_prepare_v1(
        request: *const TargetSequencePrepareRequestV1,
        report: *mut TargetSequencePrepareReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_take_prefetch_hint_v1(
        request: *const ResourceRequestV1,
        report: *mut TargetSequencePrefetchHintReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_finish_experts_v1(
        request: *const TargetSequenceFinishExpertsRequestV1,
        report: *mut TargetSequenceFinishExpertsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_finish_expert_spans_v1(
        request: *const TargetSequenceFinishExpertSpansRequestV1,
        report: *mut TargetSequenceFinishExpertsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_stage_expert_spans_v1(
        request: *const TargetSequenceStageExpertSpansRequestV1,
        report: *mut TargetSequenceStageExpertSpansReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_finish_experts_v2(
        request: *const TargetSequenceFinishExpertsRequestV2,
        report: *mut TargetSequenceFinishExpertsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_plan_experts_v1(
        request: *const TargetSequencePlanExpertsRequestV1,
        report: *mut TargetSequencePlanExpertsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_finish_planned_experts_v1(
        request: *const TargetSequenceFinishPlannedExpertsRequestV1,
        report: *mut TargetSequenceFinishExpertsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_moe_plan_release_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_finish_tail_v1(
        request: *const ResourceRequestV1,
        report: *mut TargetSequenceTailReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_dspark_rows_v1(
        request: *const ResourceRequestV1,
        report: *mut TensorReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_commit_v1(
        request: *const TargetSequenceCommitRequestV1,
        report: *mut TargetSequenceCommitReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_stats_v1(
        request: *const ResourceRequestV1,
        report: *mut TargetSequenceStatsReportV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn deltafin_provider_target_sequence_cancel_v1(
        request: *const ResourceRequestV1,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NativeProviderInventory {
    pub providers: ProviderInventory,
    pub cuda_moe_compiled: bool,
    pub cuda_exact_bf16_compiled: bool,
    pub libtorch_version: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CanaryReport {
    pub device: Device,
    pub packed_shape: (u64, u64),
    pub rms_fp32: bool,
    pub matmul_fp32: bool,
    pub softmax_fp32: bool,
    pub packed_int8_fp32: bool,
    pub required_passed: bool,
    pub detail: String,
}

impl CanaryReport {
    pub fn core_passed(&self) -> bool {
        self.rms_fp32 && self.matmul_fp32 && self.softmax_fp32
    }
}

pub struct NativeProvider;

impl NativeProvider {
    pub fn inventory() -> Result<NativeProviderInventory> {
        // SAFETY: this function has no arguments and returns an integer ABI
        // constant. The symbol is supplied by the statically linked bridge.
        let linked_version = unsafe { deltafin_provider_abi_version() };
        if linked_version != ABI_VERSION {
            return Err(DeltafinError::new(format!(
                "native provider ABI mismatch: Rust expects {ABI_VERSION}, linked bridge reports {linked_version}"
            )));
        }
        let mut inventory = InventoryV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: both buffers are valid and writable for their stated sizes;
        // the C ABI catches C++ exceptions and retains all native ownership.
        let status = unsafe {
            deltafin_provider_inventory_v1(&mut inventory, error.as_mut_ptr(), error.len())
        };
        if status != 0 {
            return Err(ffi_error("native provider inventory", &error));
        }
        if inventory.abi_version != ABI_VERSION
            || inventory.cpu_available != 1
            || inventory.provider_features & !KNOWN_PROVIDER_FEATURES != 0
            || inventory.reserved != [0; 10]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid ABI version or missing CPU provider",
            ));
        }
        let cuda_devices = u16::try_from(inventory.cuda_device_count).map_err(|_| {
            DeltafinError::new(
                "native provider reported more CUDA devices than Deltafin can address",
            )
        })?;
        Ok(NativeProviderInventory {
            providers: ProviderInventory {
                mps: inventory.mps_available != 0,
                cuda_devices,
            },
            cuda_moe_compiled: inventory.provider_features & FEATURE_CUDA_MOE != 0,
            cuda_exact_bf16_compiled: inventory.provider_features & FEATURE_CUDA_EXACT_BF16 != 0,
            libtorch_version: fixed_c_string(&inventory.libtorch_version),
        })
    }

    pub fn canary(
        device: Device,
        packed_shape: (u64, u64),
        require_packed_int8: bool,
    ) -> Result<CanaryReport> {
        let (requested_device, device_index) = match device {
            Device::Cpu => (DEVICE_CPU, 0),
            Device::Mps => (DEVICE_MPS, 0),
            Device::Cuda(index) => (DEVICE_CUDA, u32::from(index)),
        };
        let request = CanaryRequestV1 {
            struct_size: size_of::<CanaryRequestV1>() as u32,
            abi_version: ABI_VERSION,
            requested_device,
            device_index,
            flags: if require_packed_int8 {
                REQUIRE_PACKED_INT8
            } else {
                0
            },
            reserved0: 0,
            packed_rows: packed_shape.0,
            packed_columns: packed_shape.1,
            reserved: [0; 6],
        };
        let mut report = CanaryReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request is immutable for the call; report and error are
        // valid writable buffers. The ABI never retains their addresses.
        let status = unsafe {
            deltafin_provider_canary_v1(&request, &mut report, error.as_mut_ptr(), error.len())
        };
        if status != 0 {
            return Err(ffi_error("native provider canary", &error));
        }
        if report.abi_version != ABI_VERSION {
            return Err(DeltafinError::new(
                "native provider canary returned the wrong ABI version",
            ));
        }
        let selected = match (report.selected_device, report.device_index) {
            (DEVICE_CPU, 0) => Device::Cpu,
            (DEVICE_MPS, 0) => Device::Mps,
            (DEVICE_CUDA, index) => Device::Cuda(u16::try_from(index).map_err(|_| {
                DeltafinError::new("native provider returned an invalid CUDA device index")
            })?),
            _ => {
                return Err(DeltafinError::new(
                    "native provider returned an invalid selected device",
                ));
            }
        };
        if selected != device {
            return Err(DeltafinError::new(format!(
                "native provider selected {selected}, but Rust requested {device}"
            )));
        }
        let passed = report.passed_checks;
        let result = CanaryReport {
            device: selected,
            packed_shape: (report.packed_rows, report.packed_columns),
            rms_fp32: passed & CHECK_RMS_FP32 != 0,
            matmul_fp32: passed & CHECK_MATMUL_FP32 != 0,
            softmax_fp32: passed & CHECK_SOFTMAX_FP32 != 0,
            packed_int8_fp32: passed & CHECK_PACKED_INT8_FP32 != 0,
            required_passed: report.required_passed != 0,
            detail: fixed_c_string(&report.detail),
        };
        if !result.required_passed {
            return Err(DeltafinError::new(format!(
                "native provider canary failed on {}: {}",
                result.device, result.detail
            )));
        }
        Ok(result)
    }

    /// Exercises the real provider-owned split transaction without claiming
    /// that K3's layer weights/tape are linked yet. This is an ownership and
    /// ABI canary, never a target-model fallback.
    pub fn split_layer_canary(device: Device) -> Result<SplitLayerCanaryReport> {
        let session = NativeProviderSession::synthetic_split(device, 4, 32, 32)?;
        let hidden_values: Vec<f32> = (0..64).map(|index| (index as f32 - 17.0) / 64.0).collect();
        let initial_cache = vec![0.25_f32; hidden_values.len()];
        let expert_values = vec![1.0_f32; hidden_values.len()];
        let hidden = session.upload_f32(2, 32, &hidden_values)?;
        let mut cache = session.create_cache_f32(2, 32, Some(&initial_cache))?;
        let expert = session.upload_f32(2, 32, &expert_values)?;
        let prepared = session.prepare_layer(&hidden, &mut cache, 0)?;
        let route = prepared.route();
        let route_edges = route.ordered_experts().len();
        if route.positions() != 2
            || route.ordered_weight_bits().len() != route_edges
            || route.cache_version() != 0
        {
            return Err(DeltafinError::new(
                "split-layer canary returned an invalid route view",
            ));
        }
        let output = prepared.finish(&expert)?;
        let output_values = output.read_f32()?;
        let cache_values = cache.read_f32()?;

        for (index, actual) in output_values.iter().enumerate() {
            let expected = hidden_values[index] + initial_cache[index] + expert_values[index];
            if actual.to_bits() != expected.to_bits() {
                return Err(DeltafinError::new(format!(
                    "split-layer canary output differed at element {index}: {actual:?} != {expected:?}"
                )));
            }
        }
        for (index, actual) in cache_values.iter().enumerate() {
            let expected = hidden_values[index] + initial_cache[index];
            if actual.to_bits() != expected.to_bits() {
                return Err(DeltafinError::new(format!(
                    "split-layer canary cache differed at element {index}: {actual:?} != {expected:?}"
                )));
            }
        }
        Ok(SplitLayerCanaryReport {
            device: session.device(),
            positions: 2,
            route_edges,
            committed_cache_version: cache.version(),
        })
    }

    /// Qualify the one-call, provider-owned spine boundary on a real selected
    /// device using only tiny synthetic buffers. No model file is opened.
    pub fn spine_binding_canary(device: Device) -> Result<SpineBindingCanaryReport> {
        let read_plan = ReadPlan::open(
            vec![Extent::zero(BufferKind::Other, 0, 772)],
            BufferLengths::new(0, 0, 772),
            64,
            CachePolicy::Resident,
        )?;
        let reader = Reader::new(1)?;
        let (buffers, _) = reader.read(&read_plan)?;
        let descriptors = [
            SpineTensorDescriptorV1 {
                slot: 1,
                encoding: SPINE_ENCODING_RAW_BF16,
                rank: 1,
                data_buffer: SPINE_BUFFER_OTHER,
                auxiliary_buffer: SPINE_BUFFER_NONE,
                reserved0: 0,
                shape: [2, 0, 0, 0, 0, 0, 0, 0],
                data_offset: 0,
                data_length: 4,
                auxiliary_offset: 0,
                auxiliary_length: 0,
                reserved: [0; 4],
            },
            SpineTensorDescriptorV1 {
                slot: 7,
                encoding: SPINE_ENCODING_RAW_F32,
                rank: 1,
                data_buffer: SPINE_BUFFER_OTHER,
                auxiliary_buffer: SPINE_BUFFER_NONE,
                reserved0: 0,
                shape: [1, 0, 0, 0, 0, 0, 0, 0],
                data_offset: 256,
                data_length: 4,
                auxiliary_offset: 0,
                auxiliary_length: 0,
                reserved: [0; 4],
            },
            SpineTensorDescriptorV1 {
                slot: 13,
                encoding: SPINE_ENCODING_ROW_I8_F16_SCALE,
                rank: 2,
                // DFSP packs keep all authenticated components in one Other
                // slab; encoding, not slab identity, gives these bytes their
                // int8/fp16 meaning.
                data_buffer: SPINE_BUFFER_OTHER,
                auxiliary_buffer: SPINE_BUFFER_OTHER,
                reserved0: 0,
                shape: [2, 2, 0, 0, 0, 0, 0, 0],
                data_offset: 512,
                data_length: 4,
                auxiliary_offset: 768,
                auxiliary_length: 4,
                reserved: [0; 4],
            },
        ];
        let session = NativeProviderSession::target(device)?;
        let bound = session.bind_spine_layer(
            0,
            1,
            &descriptors,
            &buffers,
            SpineLayerRetention::Transient,
        )?;
        drop(buffers);
        let raw = session.read_spine_tensor_f32(0, 1, 1, SpineComponent::Data, 2)?;
        let quantized = session.read_spine_tensor_f32(0, 1, 13, SpineComponent::Data, 4)?;
        let scales = session.read_spine_tensor_f32(0, 1, 13, SpineComponent::Auxiliary, 2)?;
        if raw.stored_scalar != SpineStoredScalar::F32
            || quantized.stored_scalar != SpineStoredScalar::I8
            || scales.stored_scalar != SpineStoredScalar::F32
            || raw.values.iter().any(|&value| value != 0.0)
            || quantized.values.iter().any(|&value| value != 0.0)
            || scales.values.iter().any(|&value| value != 0.0)
        {
            return Err(DeltafinError::new(
                "native provider spine-binding canary changed storage type or value",
            ));
        }
        Ok(SpineBindingCanaryReport {
            device: session.device(),
            tensors: bound.tensor_count,
            quantized_tensors: bound.quantized_tensor_count,
            raw_tensors: bound.raw_tensor_count,
        })
    }

    /// Exercise the Rust→C ABI→ATen KDA ownership path with fixed compact
    /// zero weights. The explicit synthetic session flag is required on both
    /// sides, so these dimensions can never enter a target-model session.
    pub fn kda_transaction_canary(device: Device) -> Result<KdaTransactionCanaryReport> {
        let (descriptors, read_plan) = synthetic_kda_spine_plan()?;
        let reader = Reader::new(1)?;
        let (buffers, _) = reader.read(&read_plan)?;
        let session = NativeProviderSession::synthetic_kda_canary(device)?;
        let bound = session.bind_spine_layer(
            0,
            1,
            &descriptors,
            &buffers,
            SpineLayerRetention::Transient,
        )?;
        drop(buffers);
        if bound.tensor_count != 14 || bound.quantized_tensor_count != 8 {
            return Err(DeltafinError::new(
                "synthetic KDA spine binding returned the wrong tensor roster",
            ));
        }
        let hidden = session.upload_f32(1, 32, &[0.0_f32; 32])?;
        let mut cache = session.create_kda_cache(0)?;

        let canceled = session.decode_kda(&hidden, &mut cache, 0, 1)?;
        if canceled
            .output()
            .read_f32()?
            .iter()
            .any(|value| value.to_bits() != 0)
        {
            return Err(DeltafinError::new(
                "synthetic KDA canceled output was not exact zero",
            ));
        }
        canceled.cancel()?;
        if cache.version() != 0 {
            return Err(DeltafinError::new(
                "synthetic KDA cancellation changed the Rust cache version",
            ));
        }

        // The next decode validates native cache_version==0. If cancellation
        // had published state, this second call would fail before commit.
        let prepared = session.decode_kda(&hidden, &mut cache, 0, 1)?;
        let output = prepared.commit()?;
        if cache.version() != 1 || output.read_f32()?.iter().any(|value| value.to_bits() != 0) {
            return Err(DeltafinError::new(
                "synthetic KDA commit changed zero output or wrong cache version",
            ));
        }
        Ok(KdaTransactionCanaryReport {
            device: session.device(),
            canceled_version: 0,
            committed_version: cache.version(),
            convolution_elements: cache.convolution_elements(),
            recurrent_elements: cache.recurrent_elements(),
        })
    }

    /// Exercise exact one-position MLA, including cancel/commit ownership and
    /// geometric KV publication, through Rust→C ABI→ATen. MPS also exercises
    /// the bind-time same-input bundle; other devices deliberately report the
    /// reviewed three-call fallback.
    pub fn mla_transaction_canary(device: Device) -> Result<MlaTransactionCanaryReport> {
        let (descriptors, read_plan) = synthetic_mla_spine_plan()?;
        let reader = Reader::new(1)?;
        let (buffers, _) = reader.read(&read_plan)?;
        let session = NativeProviderSession::synthetic_mla_canary(device)?;
        const LAYER: u32 = 3;
        const GENERATION: u64 = 1;
        let bound = session.bind_spine_layer(
            LAYER,
            GENERATION,
            &descriptors,
            &buffers,
            SpineLayerRetention::Transient,
        )?;
        drop(buffers);
        if bound.tensor_count != 8
            || bound.quantized_tensor_count != 6
            || bound.raw_tensor_count != 2
        {
            return Err(DeltafinError::new(
                "synthetic MLA spine binding returned the wrong tensor roster",
            ));
        }
        let hidden = session.upload_f32(1, 32, &[0.0_f32; 32])?;
        let mut cache = session.create_mla_cache(LAYER)?;

        let canceled = session.decode_mla(&hidden, &mut cache, LAYER, GENERATION)?;
        let bundle_rows = canceled.input_bundle_rows();
        if canceled
            .output()
            .read_f32()?
            .iter()
            .any(|value| value.to_bits() != 0)
        {
            return Err(DeltafinError::new(
                "synthetic MLA canceled output was not exact zero",
            ));
        }
        canceled.cancel()?;
        if cache.version() != 0 || cache.length() != 0 || cache.capacity() != 0 {
            return Err(DeltafinError::new(
                "synthetic MLA cancellation published cache state",
            ));
        }

        let prepared = session.decode_mla(&hidden, &mut cache, LAYER, GENERATION)?;
        if prepared.input_bundle_rows() != bundle_rows {
            return Err(DeltafinError::new(
                "synthetic MLA retry changed its bundle/fallback path",
            ));
        }
        let output = prepared.commit()?;
        if cache.version() != 1
            || cache.length() != 1
            || cache.capacity() < 1
            || output.read_f32()?.iter().any(|value| value.to_bits() != 0)
        {
            return Err(DeltafinError::new(
                "synthetic MLA commit changed zero output or wrong cache state",
            ));
        }
        let expected_bundle_rows = if device == Device::Mps { 128 } else { 0 };
        if bundle_rows != expected_bundle_rows {
            return Err(DeltafinError::new(
                "synthetic MLA provider selected an unexpected bundle/fallback path",
            ));
        }
        Ok(MlaTransactionCanaryReport {
            device: session.device(),
            canceled_version: 0,
            committed_version: cache.version(),
            committed_length: cache.length(),
            capacity: cache.capacity(),
            input_bundle_rows: bundle_rows,
            production_bundle_rows: if bundle_rows == 0 { 0 } else { 14_400 },
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SplitLayerCanaryReport {
    pub device: Device,
    pub positions: usize,
    pub route_edges: usize,
    pub committed_cache_version: u64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpineBindingCanaryReport {
    pub device: Device,
    pub tensors: usize,
    pub quantized_tensors: usize,
    pub raw_tensors: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct KdaTransactionCanaryReport {
    pub device: Device,
    pub canceled_version: u64,
    pub committed_version: u64,
    pub convolution_elements: u64,
    pub recurrent_elements: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MlaTransactionCanaryReport {
    pub device: Device,
    pub canceled_version: u64,
    pub committed_version: u64,
    pub committed_length: u64,
    pub capacity: u64,
    pub input_bundle_rows: u64,
    pub production_bundle_rows: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BoundSpineLayerReport {
    pub layer_index: u32,
    pub generation: u64,
    pub tensor_count: usize,
    pub quantized_tensor_count: usize,
    pub raw_tensor_count: usize,
    pub quantized_bytes: u64,
    pub scales_bytes: u64,
    pub other_bytes: u64,
    pub resident_storage_bytes: u64,
    pub borrowed_tensor_count: usize,
    pub borrowed_source_bytes: u64,
    pub retention: SpineLayerRetention,
    pub source_use: SpineSourceUse,
}

pub(crate) struct OwnedBoundSpineLayer {
    pub(crate) binding: BoundSpineLayerReport,
    pub(crate) source_lease: LayerBuffers,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpineSourceUse {
    /// The provider owns every byte it can still access. The reader arena may
    /// recycle its source slabs immediately after bind returns.
    Detached,
    /// The provider may still access the source allocation. Rust must retain
    /// the arena lease until this token is sealed and reclaimed or aborted.
    Borrowed(SpineSourceUseToken),
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct SpineSourceUseToken {
    pub(crate) session_identity: u64,
    pub(crate) generation: u64,
    pub(crate) handle: u64,
}

impl SpineSourceUseToken {
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TargetGlobalGroup {
    Tail,
    LanguageModelHead,
}

impl TargetGlobalGroup {
    const fn abi_value(self) -> u32 {
        match self {
            Self::Tail => TARGET_GLOBAL_TAIL,
            Self::LanguageModelHead => TARGET_GLOBAL_HEAD,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BoundTargetGlobalsReport {
    pub group: TargetGlobalGroup,
    pub tensor_count: usize,
    pub quantized_tensor_count: usize,
    pub raw_tensor_count: usize,
    pub quantized_bytes: u64,
    pub scales_bytes: u64,
    pub other_bytes: u64,
    pub resident_storage_bytes: u64,
    pub groups_ready: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MetalExpertLayouts {
    pub descriptor_abi: u32,
    pub layout_capabilities: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct MetalExpertCacheStats {
    pub(crate) calls: u64,
    pub(crate) zero_copy_wraps: u64,
    pub(crate) copies: u64,
    pub(crate) cache_entries: u64,
    pub(crate) bindless: bool,
}

impl MetalExpertLayouts {
    pub const fn supports_scale4_v2(self) -> bool {
        self.descriptor_abi == METAL_DESCRIPTOR_ABI_V1
            && self.layout_capabilities & METAL_CAP_SCALE4_V2 != 0
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TargetExpertBackend {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl TargetExpertBackend {
    const fn abi_value(self) -> u32 {
        match self {
            Self::Auto => TARGET_EXPERT_AUTO,
            Self::Cpu => TARGET_EXPERT_CPU,
            Self::Metal => TARGET_EXPERT_METAL,
            Self::Cuda => TARGET_EXPERT_CUDA,
        }
    }
}

fn expert_layout_abi(layout: ExpertStorageLayout) -> (u32, u64) {
    match layout {
        ExpertStorageLayout::RawV1 => (EXPERT_LAYOUT_RAW_V1, K3_EXPERT_SOURCE_BYTES as u64),
        ExpertStorageLayout::Scale4V2 => (EXPERT_LAYOUT_SCALE4_V2, K3_SCALE4_BLOB_BYTES as u64),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetRoute {
    layer_index: u32,
    spine_generation: u64,
    ordered_experts: [u16; ROUTE_TOP_K],
    ordered_weight_bits: [u32; ROUTE_TOP_K],
}

impl TargetRoute {
    pub const fn layer_index(&self) -> u32 {
        self.layer_index
    }

    pub const fn spine_generation(&self) -> u64 {
        self.spine_generation
    }

    pub const fn ordered_experts(&self) -> &[u16; ROUTE_TOP_K] {
        &self.ordered_experts
    }

    pub const fn ordered_weight_bits(&self) -> &[u32; ROUTE_TOP_K] {
        &self.ordered_weight_bits
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetLayerPrepare {
    DenseCompleted { next_layer: u32 },
    ExpertsRequired(TargetRoute),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TargetSequenceMode {
    Prefill,
    Verify,
}

impl TargetSequenceMode {
    const fn abi_value(self) -> u32 {
        match self {
            Self::Prefill => TARGET_SEQUENCE_PREFILL,
            Self::Verify => TARGET_SEQUENCE_VERIFY,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TargetSequenceState {
    Active,
    WaitingForExperts,
    ReadyForTail,
    ReadyToCommit,
    Committed,
    Cancelled,
    Poisoned,
}

fn decode_target_sequence_state(value: u32) -> Result<TargetSequenceState> {
    match value {
        TARGET_SEQUENCE_STATE_ACTIVE => Ok(TargetSequenceState::Active),
        TARGET_SEQUENCE_STATE_WAITING_FOR_EXPERTS => Ok(TargetSequenceState::WaitingForExperts),
        TARGET_SEQUENCE_STATE_READY_FOR_TAIL => Ok(TargetSequenceState::ReadyForTail),
        TARGET_SEQUENCE_STATE_READY_TO_COMMIT => Ok(TargetSequenceState::ReadyToCommit),
        TARGET_SEQUENCE_STATE_COMMITTED => Ok(TargetSequenceState::Committed),
        TARGET_SEQUENCE_STATE_CANCELLED => Ok(TargetSequenceState::Cancelled),
        TARGET_SEQUENCE_STATE_POISONED => Ok(TargetSequenceState::Poisoned),
        _ => Err(DeltafinError::new(
            "native provider returned an invalid target-sequence state",
        )),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetSequenceMailbox {
    layer_index: u32,
    spine_generation: u64,
    routes: Box<[TargetRoute]>,
}

impl TargetSequenceMailbox {
    pub const fn layer_index(&self) -> u32 {
        self.layer_index
    }

    pub const fn spine_generation(&self) -> u64 {
        self.spine_generation
    }

    pub fn position_count(&self) -> usize {
        self.routes.len()
    }

    pub fn route(&self, row: usize) -> Option<&TargetRoute> {
        self.routes.get(row)
    }

    pub fn routes(&self) -> &[TargetRoute] {
        &self.routes
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetSequencePrefetchHint {
    source_layer: u32,
    target_layer: u32,
    expert_count: u8,
    expert_ids: [u16; PILOT_MAX_PREFETCH],
}

impl TargetSequencePrefetchHint {
    pub const fn source_layer(&self) -> u32 {
        self.source_layer
    }

    pub const fn target_layer(&self) -> u32 {
        self.target_layer
    }

    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids[..self.expert_count as usize]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetSequenceLayerPrepare {
    DenseCompleted { next_layer: u32 },
    ExpertsRequired(TargetSequenceMailbox),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetSequenceStats {
    pub positions: u64,
    pub streamed_layer_passes: u64,
    pub attention_rows: u64,
    pub expert_row_requests: u64,
    pub expert_rows_completed: u64,
    pub expert_tiles_completed: u64,
    pub tail_rows: u64,
    pub tail_provider_dispatches: u64,
    pub maximum_live_streamed_layers: u64,
    pub maximum_experts_per_request: u64,
    pub maximum_positions_per_expert_tile: u64,
    pub staged_kda_storage_bytes: u64,
    pub verify_snapshot_bytes: u64,
    pub projected_mla_storage_bytes: u64,
    pub additional_mla_storage_bytes: u64,
    pub mode: TargetSequenceMode,
    pub state: TargetSequenceState,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetSequenceCommit {
    pub committed_positions: u64,
    pub session_committed_positions: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpineLayerRetention {
    /// Replace the provider's one streamed layer slot at commit.
    Transient,
    /// Append this layer to the provider's immutable ordered prefix.
    Retained,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpineComponent {
    Data,
    Auxiliary,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpineStoredScalar {
    I8,
    Bf16,
    F32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpineTensorReadback {
    pub stored_scalar: SpineStoredScalar,
    pub shape: Box<[u64]>,
    pub values: Box<[f32]>,
}

#[derive(Debug)]
pub(crate) struct SessionInner {
    pub(crate) handle: u64,
    device: Device,
    flags: u32,
    max_route_positions: usize,
    hidden_columns: usize,
    experts: usize,
    /// A failed source-use abort means native accelerator work may still own
    /// session storage. In that exceptional state, leaking the bounded native
    /// session is safer than destroying tensors/events with unknown users.
    suppress_destroy: AtomicBool,
}

impl SessionInner {
    pub(crate) fn flush_metal_expert_cache(&self) -> Result<()> {
        if self.device != Device::Mps {
            return Err(DeltafinError::new(
                "Metal expert-cache flush requires an MPS provider session",
            ));
        }
        release_resource(
            deltafin_provider_metal_expert_cache_flush_v1,
            self.handle,
            0,
            "native Metal expert-cache flush",
        )
    }

    pub(crate) fn metal_expert_cache_stats(&self) -> Result<MetalExpertCacheStats> {
        if self.device != Device::Mps {
            return Err(DeltafinError::new(
                "Metal expert-cache stats require an MPS provider session",
            ));
        }
        let request = ResourceRequestV1::new(self.handle, 0);
        let mut report = MetalExpertCacheStatsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request is immutable and report/error remain writable
        // for this synchronous, session-retained ABI call.
        let status = unsafe {
            deltafin_provider_metal_expert_cache_stats_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native Metal expert-cache stats", &error));
        }
        if report.struct_size as usize != size_of::<MetalExpertCacheStatsReportV1>()
            || report.abi_version != ABI_VERSION
            || report.session != self.handle
            || report.bindless > 1
            || report.reserved != [0; 2]
        {
            return Err(DeltafinError::new(
                "native provider returned invalid Metal expert-cache stats",
            ));
        }
        Ok(MetalExpertCacheStats {
            calls: report.calls,
            zero_copy_wraps: report.zero_copy_wraps,
            copies: report.copies,
            cache_entries: report.cache_entries,
            bindless: report.bindless != 0,
        })
    }

    fn configure_cuda_expert_cache(&self, policy: CudaExpertCachePolicy) -> Result<()> {
        if !matches!(self.device, Device::Cuda(_)) {
            return Err(DeltafinError::new(
                "CUDA expert-cache configuration requires a CUDA provider session",
            ));
        }
        if policy.capacity_experts > CUDA_EXPERT_CACHE_MAX_EXPERTS {
            return Err(DeltafinError::new(format!(
                "CUDA expert-cache capacity {} exceeds the provider bound {}",
                policy.capacity_experts, CUDA_EXPERT_CACHE_MAX_EXPERTS,
            )));
        }
        let request = CudaCacheConfigureRequestV1 {
            struct_size: size_of::<CudaCacheConfigureRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.handle,
            capacity_mode: CUDA_CACHE_CAPACITY_EXACT,
            reserve_mode: CUDA_CACHE_RESERVE_BYTES,
            capacity_experts: u64::from(policy.capacity_experts),
            reserve_value: policy.reserve_bytes,
            flags: 0,
            reserved0: 0,
            reserved: [0; 2],
        };
        let mut report = CudaCacheConfigureReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request is immutable and report/error remain writable
        // for this synchronous, session-retained ABI call.
        let status = unsafe {
            deltafin_provider_cuda_cache_configure_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native CUDA expert-cache configure", &error));
        }
        if report.struct_size as usize != size_of::<CudaCacheConfigureReportV1>()
            || report.abi_version != ABI_VERSION
            || report.session != self.handle
            || report.capacity_mode != CUDA_CACHE_CAPACITY_EXACT
            || report.reserve_mode != CUDA_CACHE_RESERVE_BYTES
            || report.capacity_experts != u64::from(policy.capacity_experts)
            || report.reserve_value != policy.reserve_bytes
            || report.configured != 1
            || report.flags != 0
            || report.reserved != [0; 2]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid CUDA expert-cache configuration report",
            ));
        }
        Ok(())
    }
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        if self.suppress_destroy.load(Ordering::Acquire) {
            return;
        }
        let _ = release_resource(
            deltafin_provider_session_destroy_v1,
            self.handle,
            0,
            "native provider session destroy",
        );
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct TargetStateBoundary {
    pub(crate) committed_positions: u64,
    pub(crate) cache_generation: u64,
}

type TargetStateResolveFunction = unsafe extern "C" fn(
    *const ResourceRequestV1,
    *mut TargetStateReportV1,
    *mut c_char,
    usize,
) -> i32;

#[derive(Debug)]
pub(crate) struct TargetStateBranch {
    session: Arc<SessionInner>,
    handle: u64,
    parent: TargetStateBoundary,
}

impl Drop for TargetStateBranch {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        let request = ResourceRequestV1::new(self.session.handle, self.handle);
        let mut report = TargetStateReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: best-effort rollback while the Arc keeps the session live.
        let _ = unsafe {
            deltafin_provider_target_state_branch_discard_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        self.handle = 0;
    }
}

/// Owns one provider-side device context. Tensor, cache, and ticket objects
/// below contain only opaque integer IDs plus an `Arc` that keeps this session
/// alive; an ATen object never crosses into Rust.
///
/// Threading contract on MPS: the provider locks each session independently,
/// but ATen submits every MPS operation through one process-global Metal
/// command buffer that two threads may not encode into at once — one thread's
/// commit hits the other's open encoder and Metal aborts the process. Session
/// locking cannot close that gap, because the shared stream sits below the
/// session boundary and ATen also submits from allocator and completion
/// callbacks that never enter this ABI. So MPS work must be driven from one
/// thread at a time process-wide. The runtime satisfies this structurally (an
/// engine owns exactly one session, and the server admits one generation at a
/// time); tests must use [`exclusive_mps_device`].
#[derive(Debug)]
pub struct NativeProviderSession {
    inner: Arc<SessionInner>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct NativeProviderMemorySnapshot {
    pub device: Device,
    pub active_bytes: Option<u64>,
    /// Complete allocator/driver reservation, including inactive cached
    /// blocks when the provider exposes that distinction.
    pub reserved_bytes: Option<u64>,
    pub recommended_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub cache_trimmed: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetPilotAdmission {
    pub layer_capacity: u32,
    pub reserve_bytes: u64,
}

impl NativeProviderSession {
    pub(crate) fn lease(&self) -> Arc<SessionInner> {
        Arc::clone(&self.inner)
    }

    pub(crate) fn flush_metal_expert_cache(&self) -> Result<()> {
        self.inner.flush_metal_expert_cache()
    }

    #[cfg(test)]
    pub(crate) fn metal_expert_cache_stats(&self) -> Result<MetalExpertCacheStats> {
        self.inner.metal_expert_cache_stats()
    }

    /// Freeze the session's CUDA expert-cache budget before any expert plan
    /// or availability probe runs. The engine derives this policy from the
    /// same memory snapshot that sizes spine residency, so the two VRAM
    /// consumers can never double-book the device.
    pub fn configure_cuda_expert_cache(&self, policy: CudaExpertCachePolicy) -> Result<()> {
        self.inner.configure_cuda_expert_cache(policy)
    }

    /// Create a real-target provider context. The complete K3 execution tape
    /// is linked, but callers must still pass the separate real-weight parity,
    /// cancellation, memory, and performance gates before treating it as the
    /// production default.
    pub fn target(device: Device) -> Result<Self> {
        Self::create(device, 0, ROUTE_MAX_POSITIONS, 0, 0)
    }

    pub fn synthetic_split(
        device: Device,
        max_route_positions: usize,
        hidden_columns: usize,
        experts: usize,
    ) -> Result<Self> {
        Self::create(
            device,
            SESSION_SYNTHETIC_SPLIT,
            max_route_positions,
            hidden_columns,
            experts,
        )
    }

    fn synthetic_kda_canary(device: Device) -> Result<Self> {
        Self::create(device, SESSION_SYNTHETIC_KDA, 1, 0, 0)
    }

    fn synthetic_mla_canary(device: Device) -> Result<Self> {
        Self::create(device, SESSION_SYNTHETIC_MLA, 1, 0, 0)
    }

    fn create(
        device: Device,
        flags: u32,
        max_route_positions: usize,
        hidden_columns: usize,
        experts: usize,
    ) -> Result<Self> {
        let max_route_positions = u32::try_from(max_route_positions)
            .map_err(|_| DeltafinError::new("route-position count exceeds ABI u32"))?;
        let hidden_columns = u32::try_from(hidden_columns)
            .map_err(|_| DeltafinError::new("hidden width exceeds ABI u32"))?;
        let experts = u32::try_from(experts)
            .map_err(|_| DeltafinError::new("expert count exceeds ABI u32"))?;
        let (requested_device, device_index) = device_fields(device);
        let request = SessionRequestV1 {
            struct_size: size_of::<SessionRequestV1>() as u32,
            abi_version: ABI_VERSION,
            requested_device,
            device_index,
            flags,
            max_route_positions,
            synthetic_hidden_columns: hidden_columns,
            synthetic_experts: experts,
            reserved: [0; 6],
        };
        let mut report = SessionReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request is immutable; report/error are writable for their
        // declared sizes. The provider copies all configuration values and
        // returns only an opaque integer ID.
        let status = unsafe {
            deltafin_provider_session_create_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider session create", &error));
        }
        let selected = match decode_device(report.selected_device, report.device_index) {
            Ok(selected) => selected,
            Err(error) => {
                if report.session != 0 {
                    let _ = release_resource(
                        deltafin_provider_session_destroy_v1,
                        report.session,
                        0,
                        "invalid native provider session destroy",
                    );
                }
                return Err(error);
            }
        };
        if report.abi_version != ABI_VERSION
            || report.struct_size as usize != size_of::<SessionReportV1>()
            || report.session == 0
            || selected != device
            || report.flags != flags
            || report.max_route_positions != max_route_positions
            || report.reserved != [0; 6]
        {
            if report.session != 0 {
                let _ = release_resource(
                    deltafin_provider_session_destroy_v1,
                    report.session,
                    0,
                    "invalid native provider session destroy",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid session report",
            ));
        }
        Ok(Self {
            inner: Arc::new(SessionInner {
                handle: report.session,
                device: selected,
                flags,
                max_route_positions: max_route_positions as usize,
                hidden_columns: hidden_columns as usize,
                experts: experts as usize,
                suppress_destroy: AtomicBool::new(false),
            }),
        })
    }

    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// Query live provider memory. `trim_unused` is an exceptional pressure
    /// recovery operation: the native side first proves that no target/cache
    /// transaction is in flight, drains the selected accelerator, then frees
    /// only allocator blocks which no live tensor owns.
    pub fn memory_snapshot(&self, trim_unused: bool) -> Result<NativeProviderMemorySnapshot> {
        let actions = if trim_unused { MEMORY_TRIM_UNUSED } else { 0 };
        let request = MemoryRequestV1 {
            struct_size: size_of::<MemoryRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            actions,
            reserved0: 0,
            reserved: [0; 5],
        };
        let mut report = MemoryReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request is borrowed only for this call; report/error are
        // writable for their declared sizes, and the ABI retains no address.
        let status = unsafe {
            deltafin_provider_memory_snapshot_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider memory snapshot", &error));
        }
        let selected = decode_device(report.selected_device, report.device_index)?;
        if report.struct_size as usize != size_of::<MemoryReportV1>()
            || report.abi_version != ABI_VERSION
            || selected != self.inner.device
            || report.available_fields & !MEMORY_ALL_FIELDS != 0
            || report.performed_actions & !MEMORY_TRIM_UNUSED != 0
            || report.performed_actions != actions
            || report.reserved0 != 0
            || report.reserved1 != 0
            || report.reserved != [0; 4]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid memory report",
            ));
        }
        let field = |mask: u32, value: u64| -> Result<Option<u64>> {
            if report.available_fields & mask != 0 {
                Ok(Some(value))
            } else if value == 0 {
                Ok(None)
            } else {
                Err(DeltafinError::new(
                    "native provider populated an unavailable memory field",
                ))
            }
        };
        let snapshot = NativeProviderMemorySnapshot {
            device: selected,
            active_bytes: field(MEMORY_ACTIVE_BYTES, report.active_bytes)?,
            reserved_bytes: field(MEMORY_RESERVED_BYTES, report.reserved_bytes)?,
            recommended_bytes: field(MEMORY_RECOMMENDED_BYTES, report.recommended_bytes)?,
            total_bytes: field(MEMORY_TOTAL_BYTES, report.total_bytes)?,
            available_bytes: field(MEMORY_AVAILABLE_BYTES, report.available_bytes)?,
            cache_trimmed: report.performed_actions & MEMORY_TRIM_UNUSED != 0,
        };
        if let (Some(available), Some(total)) = (snapshot.available_bytes, snapshot.total_bytes)
            && available > total
        {
            return Err(DeltafinError::new(
                "native provider memory report has available bytes above total bytes",
            ));
        }
        if trim_unused && !snapshot.cache_trimmed {
            return Err(DeltafinError::new(
                "native provider did not perform the requested cache trim",
            ));
        }
        Ok(snapshot)
    }

    /// Admit the scheduling-only CPU/MPS router roster before any target or
    /// spine weight is bound. This call allocates no prediction output; it
    /// only permits later authoritative layer binds to publish detached,
    /// fail-soft scheduling clones. Those hints can schedule reads but can
    /// never supply route IDs, route weights, or model output.
    pub fn enable_target_pilot(&self) -> Result<TargetPilotAdmission> {
        let request = ResourceRequestV1::new(self.inner.handle, 0);
        let mut report = TargetPilotEnableReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request is immutable and report/error remain writable for
        // the synchronous ABI call. The provider retains none of them.
        let status = unsafe {
            deltafin_provider_target_pilot_enable_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider target PILOT admission", &error));
        }
        if report.struct_size as usize != size_of::<TargetPilotEnableReportV1>()
            || report.abi_version != ABI_VERSION
            || report.session != self.inner.handle
            || report.enabled != 1
            || report.layer_capacity != TARGET_PILOT_LAYER_CAPACITY
            || report.reserve_bytes != TARGET_PILOT_RESERVE_BYTES
            || report.reserved != [0; 4]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid target PILOT admission report",
            ));
        }
        Ok(TargetPilotAdmission {
            layer_capacity: report.layer_capacity,
            reserve_bytes: report.reserve_bytes,
        })
    }

    pub fn metal_expert_layouts(&self, metal_shader_path: &str) -> Result<MetalExpertLayouts> {
        if self.inner.device != Device::Mps {
            return Err(DeltafinError::new(
                "Metal expert-layout qualification requires the selected MPS provider",
            ));
        }
        let shader_bytes = metal_shader_path.as_bytes();
        if shader_bytes.is_empty() || shader_bytes.contains(&0) || shader_bytes.len() > 4096 {
            return Err(DeltafinError::new(
                "Metal expert-layout shader path must be nonempty UTF-8 without NUL and at most 4096 bytes",
            ));
        }
        let request = MetalExpertLayoutsRequestV1 {
            struct_size: size_of::<MetalExpertLayoutsRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            metal_shader_path: shader_bytes.as_ptr().cast(),
            metal_shader_path_length: shader_bytes.len() as u64,
            flags: 0,
            reserved0: 0,
            reserved: [0; 3],
        };
        let mut report = MetalExpertLayoutsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the shader bytes remain live through this synchronous call;
        // the provider retains neither the path nor either ABI object.
        let status = unsafe {
            deltafin_provider_metal_expert_layouts_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error(
                "native Metal expert-layout qualification",
                &error,
            ));
        }
        let recognized = METAL_CAP_RAW_V1 | METAL_CAP_SCALE4_V2;
        if report.struct_size as usize != size_of::<MetalExpertLayoutsReportV1>()
            || report.abi_version != ABI_VERSION
            || report.session != self.inner.handle
            || report.flags != 0
            || report.layout_capabilities & METAL_CAP_RAW_V1 == 0
            || report.layout_capabilities & !recognized != 0
            || report.raw_span_bytes != K3_EXPERT_SOURCE_BYTES as u64
            || report.scale4_span_bytes != K3_SCALE4_BLOB_BYTES as u64
            || report.reserved != [0; 2]
            || (report.layout_capabilities & METAL_CAP_SCALE4_V2 != 0
                && report.descriptor_abi != METAL_DESCRIPTOR_ABI_V1)
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid Metal expert-layout capability report",
            ));
        }
        Ok(MetalExpertLayouts {
            descriptor_abi: report.descriptor_abi,
            layout_capabilities: report.layout_capabilities,
        })
    }

    /// Resolve device/backend compatibility before the caller starts disk I/O.
    /// CUDA `Auto` remains deferred so the session-owned runtime KAT can select
    /// native CUDA or fail soft to the exact compiled CPU expert path.
    pub fn resolve_target_expert_backend(
        &self,
        requested: TargetExpertBackend,
    ) -> Result<TargetExpertBackend> {
        resolve_target_backend(self.device(), requested)
    }

    pub(crate) fn identity(&self) -> u64 {
        self.inner.handle
    }

    /// Fail closed after Rust can no longer prove that a borrowed host slab is
    /// unused. Every clone shares this flag, so the eventual final Arc drop
    /// retains the native session (and therefore its device fences/storage)
    /// until process exit instead of risking an asynchronous use-after-free.
    pub(crate) fn suppress_destroy_after_unproven_source_use(&self) {
        self.inner.suppress_destroy.store(true, Ordering::Release);
    }

    /// A malformed post-success borrow report whose token cannot be aborted
    /// leaves both native ownership and the host lease indeterminate. Publish
    /// session retention before leaking the lease, so even an unwind between
    /// those actions cannot destroy accelerator resources first.
    fn retain_unproven_source_lease<Lease>(&self, lease: Lease) {
        self.suppress_destroy_after_unproven_source_use();
        std::mem::forget(lease);
    }

    /// Clear only committed target attention state for a fresh conversation.
    /// Immutable globals, retained spine layers, provider qualifications, and
    /// worker ownership remain live. The provider rejects this operation if a
    /// target transaction is still unpublished.
    pub fn reset_target_state(&self) -> Result<()> {
        release_resource(
            deltafin_provider_target_state_reset_v1,
            self.inner.handle,
            0,
            "native provider target-state reset",
        )
    }

    pub(crate) fn inspect_target_state(&self) -> Result<TargetStateBoundary> {
        let request = ResourceRequestV1::new(self.inner.handle, 0);
        let mut report = TargetStateReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/report/error remain valid for the synchronous call.
        let status = unsafe {
            deltafin_provider_target_state_inspect_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("inspect native provider target state", &error));
        }
        validate_target_state_report(&report, None)?;
        if report.active_branch != 0 {
            return Err(DeltafinError::new(
                "native provider target state unexpectedly has an active branch",
            ));
        }
        Ok(TargetStateBoundary {
            committed_positions: report.committed_positions,
            cache_generation: report.cache_generation,
        })
    }

    pub(crate) fn begin_target_state_branch(
        &self,
        expected: TargetStateBoundary,
    ) -> Result<TargetStateBranch> {
        let request = TargetStateBranchRequestV1 {
            struct_size: size_of::<TargetStateBranchRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            expected_committed_positions: expected.committed_positions,
            expected_cache_generation: expected.cache_generation,
            reserved: [0; 4],
        };
        let mut report = TargetStateReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/report/error remain valid for the synchronous call.
        let status = unsafe {
            deltafin_provider_target_state_branch_begin_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error(
                "begin native provider target-state branch",
                &error,
            ));
        }
        validate_target_state_report(&report, None)?;
        if report.active_branch == 0
            || report.committed_positions != expected.committed_positions
            || report.cache_generation != expected.cache_generation
        {
            return Err(DeltafinError::new(
                "native provider began target-state branch at the wrong boundary",
            ));
        }
        Ok(TargetStateBranch {
            session: Arc::clone(&self.inner),
            handle: report.active_branch,
            parent: expected,
        })
    }

    pub(crate) fn publish_target_state_branch(
        &self,
        mut branch: TargetStateBranch,
    ) -> Result<TargetStateBoundary> {
        self.resolve_target_state_branch(
            &mut branch,
            deltafin_provider_target_state_branch_publish_v1,
            "publish native provider target-state branch",
            false,
        )
    }

    pub(crate) fn discard_target_state_branch(
        &self,
        mut branch: TargetStateBranch,
    ) -> Result<TargetStateBoundary> {
        self.resolve_target_state_branch(
            &mut branch,
            deltafin_provider_target_state_branch_discard_v1,
            "discard native provider target-state branch",
            true,
        )
    }

    fn resolve_target_state_branch(
        &self,
        branch: &mut TargetStateBranch,
        function: TargetStateResolveFunction,
        operation: &str,
        restoring_parent: bool,
    ) -> Result<TargetStateBoundary> {
        if branch.handle == 0 || !Arc::ptr_eq(&self.inner, &branch.session) {
            return Err(DeltafinError::new(
                "native target-state branch belongs to another provider session",
            ));
        }
        let request = ResourceRequestV1::new(self.inner.handle, branch.handle);
        let mut report = TargetStateReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/report/error remain valid for the synchronous call.
        let status = unsafe { function(&request, &mut report, error.as_mut_ptr(), error.len()) };
        if status != 0 {
            return Err(ffi_error(operation, &error));
        }
        validate_target_state_report(&report, Some(0))?;
        if restoring_parent && report.committed_positions != branch.parent.committed_positions {
            return Err(DeltafinError::new(
                "native target-state discard restored the wrong position boundary",
            ));
        }
        branch.handle = 0;
        Ok(TargetStateBoundary {
            committed_positions: report.committed_positions,
            cache_generation: report.cache_generation,
        })
    }

    pub fn upload_f32(
        &self,
        rows: usize,
        columns: usize,
        values: &[f32],
    ) -> Result<ProviderTensor> {
        let elements = checked_shape(rows, columns)?;
        if values.len() != elements {
            return Err(DeltafinError::new(format!(
                "tensor shape {rows}x{columns} needs {elements} values; got {}",
                values.len()
            )));
        }
        let request = TensorUploadF32V1 {
            struct_size: size_of::<TensorUploadF32V1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            rows: rows as u64,
            columns: columns as u64,
            data: values.as_ptr(),
            element_count: elements as u64,
            flags: 0,
            reserved0: 0,
            reserved: [0; 3],
        };
        let mut report = TensorReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: values remains live for the complete call. The provider
        // clones it synchronously and does not retain the pointer.
        let status = unsafe {
            deltafin_provider_tensor_upload_f32_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider tensor upload", &error));
        }
        if report.abi_version != ABI_VERSION
            || report.struct_size as usize != size_of::<TensorReportV1>()
            || report.tensor == 0
            || report.rows != rows as u64
            || report.columns != columns as u64
            || report.reserved != [0; 4]
        {
            if report.tensor != 0 {
                let _ = release_resource(
                    deltafin_provider_tensor_release_v1,
                    self.inner.handle,
                    report.tensor,
                    "invalid native provider tensor release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid tensor report",
            ));
        }
        Ok(ProviderTensor {
            session: Arc::clone(&self.inner),
            handle: report.tensor,
            rows,
            columns,
        })
    }

    #[cfg(test)]
    pub(crate) fn upload_bf16(
        &self,
        rows: usize,
        columns: usize,
        bytes: &[u8],
    ) -> Result<ProviderTensor> {
        let expected = checked_shape(rows, columns)?
            .checked_mul(2)
            .ok_or_else(|| DeltafinError::new("BF16 tensor byte length overflowed"))?;
        if bytes.len() != expected {
            return Err(DeltafinError::new(format!(
                "BF16 tensor shape {rows}x{columns} needs {expected} bytes; got {}",
                bytes.len()
            )));
        }
        let request = TensorUploadBf16V1 {
            struct_size: size_of::<TensorUploadBf16V1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            rows: rows as u64,
            columns: columns as u64,
            data: bytes.as_ptr(),
            byte_length: bytes.len() as u64,
            flags: 0,
            reserved0: 0,
            reserved: [0; 3],
        };
        let mut report = TensorReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: bytes remains live through the synchronous provider copy.
        let status = unsafe {
            deltafin_provider_tensor_upload_bf16_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider BF16 tensor upload", &error));
        }
        if report.abi_version != ABI_VERSION
            || report.struct_size as usize != size_of::<TensorReportV1>()
            || report.tensor == 0
            || report.rows != rows as u64
            || report.columns != columns as u64
            || report.reserved != [0; 4]
        {
            if report.tensor != 0 {
                let _ = release_resource(
                    deltafin_provider_tensor_release_v1,
                    self.inner.handle,
                    report.tensor,
                    "invalid native provider BF16 tensor release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid BF16 tensor report",
            ));
        }
        Ok(ProviderTensor {
            session: Arc::clone(&self.inner),
            handle: report.tensor,
            rows,
            columns,
        })
    }

    pub fn create_cache_f32(
        &self,
        rows: usize,
        columns: usize,
        initial: Option<&[f32]>,
    ) -> Result<ProviderCache> {
        let elements = checked_shape(rows, columns)?;
        if let Some(values) = initial
            && values.len() != elements
        {
            return Err(DeltafinError::new(format!(
                "cache shape {rows}x{columns} needs {elements} values; got {}",
                values.len()
            )));
        }
        let request = CacheCreateF32V1 {
            struct_size: size_of::<CacheCreateF32V1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            rows: rows as u64,
            columns: columns as u64,
            initial_data: initial.map_or(ptr::null(), |values| values.as_ptr()),
            element_count: initial.map_or(0, |values| values.len()) as u64,
            flags: 0,
            reserved0: 0,
            reserved: [0; 3],
        };
        let mut report = CacheReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: optional initial data remains live during the call and is
        // copied synchronously. Only an opaque cache ID is returned.
        let status = unsafe {
            deltafin_provider_cache_create_f32_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider cache create", &error));
        }
        if report.abi_version != ABI_VERSION
            || report.struct_size as usize != size_of::<CacheReportV1>()
            || report.cache == 0
            || report.rows != rows as u64
            || report.columns != columns as u64
            || report.version != 0
            || report.reserved != [0; 3]
        {
            if report.cache != 0 {
                let _ = release_resource(
                    deltafin_provider_cache_release_v1,
                    self.inner.handle,
                    report.cache,
                    "invalid native provider cache release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid cache report",
            ));
        }
        Ok(ProviderCache {
            session: Arc::clone(&self.inner),
            handle: report.cache,
            rows,
            columns,
            version: report.version,
        })
    }

    /// Allocate provider-owned state for one exact K3 KDA layer. The compact
    /// dimensions are reachable only in the explicitly synthetic doctor
    /// session; ordinary target sessions always allocate the released K3
    /// [1,96,128,128] recurrent contract.
    pub fn create_kda_cache(&self, layer_index: u32) -> Result<ProviderKdaCache> {
        let request = KdaCacheCreateV1 {
            struct_size: size_of::<KdaCacheCreateV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            layer_index,
            flags: 0,
            reserved: [0; 5],
        };
        let mut report = KdaCacheReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request is immutable and report/error are valid writable
        // buffers. The returned ID owns all device tensors inside C++.
        let status = unsafe {
            deltafin_provider_kda_cache_create_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider KDA cache create", &error));
        }
        let (expected_convolution, expected_recurrent) =
            if self.inner.flags & SESSION_SYNTHETIC_KDA != 0 {
                (12_288_u64, 32_768_u64)
            } else {
                (147_456_u64, 1_572_864_u64)
            };
        if report.struct_size as usize != size_of::<KdaCacheReportV1>()
            || report.abi_version != ABI_VERSION
            || report.cache == 0
            || report.layer_index != layer_index
            || report.flags != 0
            || report.version != 0
            || report.convolution_elements != expected_convolution
            || report.recurrent_elements != expected_recurrent
            || report.reserved != [0; 2]
        {
            if report.cache != 0 {
                let _ = release_resource(
                    deltafin_provider_kda_cache_release_v1,
                    self.inner.handle,
                    report.cache,
                    "invalid native provider KDA cache release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid KDA cache report",
            ));
        }
        Ok(ProviderKdaCache {
            session: Arc::clone(&self.inner),
            handle: report.cache,
            layer_index,
            version: 0,
            convolution_elements: report.convolution_elements,
            recurrent_elements: report.recurrent_elements,
            poisoned: false,
        })
    }

    /// Run one KDA attention position and stage, but do not publish, its four
    /// cache updates. Dropping the returned value cancels the ticket.
    pub fn decode_kda<'cache>(
        &self,
        hidden: &ProviderTensor,
        cache: &'cache mut ProviderKdaCache,
        layer_index: u32,
        spine_generation: u64,
    ) -> Result<PreparedKdaDecode<'cache>> {
        require_same_session(&self.inner, &hidden.session, "KDA hidden tensor")?;
        require_same_session(&self.inner, &cache.session, "KDA cache")?;
        if cache.poisoned {
            return Err(DeltafinError::new(
                "native provider KDA cache is poisoned by an invalid commit report",
            ));
        }
        if cache.layer_index != layer_index || hidden.rows != 1 {
            return Err(DeltafinError::new(
                "KDA hidden/cache/layer contract does not match one-position decode",
            ));
        }
        let expected_hidden = if self.inner.flags & SESSION_SYNTHETIC_KDA != 0 {
            32
        } else {
            7_168
        };
        if hidden.columns != expected_hidden || spine_generation == 0 {
            return Err(DeltafinError::new(
                "KDA hidden width or spine generation does not match the session",
            ));
        }
        let request = KdaDecodeRequestV1 {
            struct_size: size_of::<KdaDecodeRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            hidden: hidden.handle,
            cache: cache.handle,
            layer_index,
            flags: 0,
            spine_generation,
            reserved: [0; 4],
        };
        let mut report = KdaDecodeReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: all tensor/cache storage is provider-owned. The request is
        // immutable and report/error remain writable for the call.
        let status = unsafe {
            deltafin_provider_kda_decode_v1(&request, &mut report, error.as_mut_ptr(), error.len())
        };
        if status != 0 {
            return Err(ffi_error("native provider KDA decode", &error));
        }
        if report.struct_size as usize != size_of::<KdaDecodeReportV1>()
            || report.abi_version != ABI_VERSION
            || report.output == 0
            || report.ticket == 0
            || report.cache_version != cache.version
            || report.spine_generation != spine_generation
            || report.rows != 1
            || report.columns != expected_hidden as u64
            || report.reserved != [0; 3]
        {
            if report.ticket != 0 {
                let _ = release_resource(
                    deltafin_provider_kda_ticket_release_v1,
                    self.inner.handle,
                    report.ticket,
                    "invalid native provider KDA ticket release",
                );
            }
            if report.output != 0 {
                let _ = release_resource(
                    deltafin_provider_tensor_release_v1,
                    self.inner.handle,
                    report.output,
                    "invalid native provider KDA output release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid KDA decode report",
            ));
        }
        Ok(PreparedKdaDecode {
            session: Arc::clone(&self.inner),
            ticket: report.ticket,
            cache,
            output: Some(ProviderTensor {
                session: Arc::clone(&self.inner),
                handle: report.output,
                rows: 1,
                columns: expected_hidden,
            }),
        })
    }

    /// Allocate provider-owned geometric expanded KV storage for one exact K3
    /// MLA layer. The public ABI never selects the non-bit-exact absorbed
    /// compact research representation.
    pub fn create_mla_cache(&self, layer_index: u32) -> Result<ProviderMlaCache> {
        let request = MlaCacheCreateV1 {
            struct_size: size_of::<MlaCacheCreateV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            layer_index,
            flags: 0,
            reserved: [0; 5],
        };
        let mut report = MlaCacheReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request is immutable and report/error are valid writable
        // buffers. The returned handle owns only C++ tensor objects.
        let status = unsafe {
            deltafin_provider_mla_cache_create_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider MLA cache create", &error));
        }
        if report.struct_size as usize != size_of::<MlaCacheReportV1>()
            || report.abi_version != ABI_VERSION
            || report.cache == 0
            || report.layer_index != layer_index
            || report.flags != 0
            || report.version != 0
            || report.length != 0
            || report.capacity != 0
            || report.reserved != [0; 2]
        {
            if report.cache != 0 {
                let _ = release_resource(
                    deltafin_provider_mla_cache_release_v1,
                    self.inner.handle,
                    report.cache,
                    "invalid native provider MLA cache release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid MLA cache report",
            ));
        }
        Ok(ProviderMlaCache {
            session: Arc::clone(&self.inner),
            handle: report.cache,
            layer_index,
            version: 0,
            length: 0,
            capacity: 0,
            poisoned: false,
        })
    }

    /// Run one exact MLA decode position. The returned output and candidate
    /// cache growth remain speculative until `PreparedMlaDecode::commit`.
    pub fn decode_mla<'cache>(
        &self,
        hidden: &ProviderTensor,
        cache: &'cache mut ProviderMlaCache,
        layer_index: u32,
        spine_generation: u64,
    ) -> Result<PreparedMlaDecode<'cache>> {
        require_same_session(&self.inner, &hidden.session, "MLA hidden tensor")?;
        require_same_session(&self.inner, &cache.session, "MLA cache")?;
        if cache.poisoned {
            return Err(DeltafinError::new(
                "native provider MLA cache is poisoned by an invalid commit report",
            ));
        }
        if cache.layer_index != layer_index || hidden.rows != 1 {
            return Err(DeltafinError::new(
                "MLA hidden/cache/layer contract does not match one-position decode",
            ));
        }
        let synthetic = self.inner.flags & SESSION_SYNTHETIC_MLA != 0;
        let expected_hidden = if synthetic { 32 } else { 7_168 };
        if hidden.columns != expected_hidden || spine_generation == 0 {
            return Err(DeltafinError::new(
                "MLA hidden width or spine generation does not match the session",
            ));
        }
        let expected_length = cache
            .length
            .checked_add(1)
            .ok_or_else(|| DeltafinError::new("native provider MLA length overflowed in Rust"))?;
        let request = MlaDecodeRequestV1 {
            struct_size: size_of::<MlaDecodeRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            hidden: hidden.handle,
            cache: cache.handle,
            layer_index,
            flags: 0,
            spine_generation,
            reserved: [0; 4],
        };
        let mut report = MlaDecodeReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: all referenced tensor/cache state is provider-owned and the
        // report/error buffers remain writable for the complete call.
        let status = unsafe {
            deltafin_provider_mla_decode_v1(&request, &mut report, error.as_mut_ptr(), error.len())
        };
        if status != 0 {
            return Err(ffi_error("native provider MLA decode", &error));
        }
        let expected_bundle_rows = if self.inner.device == Device::Mps {
            if synthetic { 128 } else { 14_400 }
        } else {
            0
        };
        let max_context = if synthetic { 32 } else { 1_048_576 };
        if report.struct_size as usize != size_of::<MlaDecodeReportV1>()
            || report.abi_version != ABI_VERSION
            || report.output == 0
            || report.ticket == 0
            || report.cache_version != cache.version
            || report.spine_generation != spine_generation
            || report.rows != 1
            || report.columns != expected_hidden as u64
            || report.proposed_length != expected_length
            || report.proposed_capacity < report.proposed_length
            || report.proposed_capacity > max_context
            || report.input_bundle_rows != expected_bundle_rows
            || report.reserved != [0; 2]
        {
            if report.ticket != 0 {
                let _ = release_resource(
                    deltafin_provider_mla_ticket_release_v1,
                    self.inner.handle,
                    report.ticket,
                    "invalid native provider MLA ticket release",
                );
            }
            if report.output != 0 {
                let _ = release_resource(
                    deltafin_provider_tensor_release_v1,
                    self.inner.handle,
                    report.output,
                    "invalid native provider MLA output release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid MLA decode report",
            ));
        }
        Ok(PreparedMlaDecode {
            session: Arc::clone(&self.inner),
            ticket: report.ticket,
            cache,
            output: Some(ProviderTensor {
                session: Arc::clone(&self.inner),
                handle: report.output,
                rows: 1,
                columns: expected_hidden,
            }),
            proposed_length: report.proposed_length,
            proposed_capacity: report.proposed_capacity,
            input_bundle_rows: report.input_bundle_rows,
        })
    }

    /// Bind one complete resident-spine layer in a single native call.
    ///
    /// The C++ provider validates the full descriptor set before reading any
    /// slab payload and commits the new generation only after all selected-
    /// device preparation succeeds. V2 reports whether source ownership is
    /// detached or borrowed; the caller must keep `buffers` alive for every
    /// borrowed source-use token until the token is reclaimed or aborted.
    pub fn bind_spine_layer(
        &self,
        layer_index: u32,
        generation: u64,
        descriptors: &[SpineTensorDescriptorV1],
        buffers: &LayerBuffers,
        retention: SpineLayerRetention,
    ) -> Result<BoundSpineLayerReport> {
        let mut leak_source = false;
        let result = self.bind_spine_layer_inner(
            layer_index,
            generation,
            descriptors,
            buffers,
            retention,
            false,
            &mut leak_source,
        );
        debug_assert!(
            !leak_source,
            "detached bind may never borrow source storage"
        );
        result
    }

    /// Ownership-safe V2 bind used by the streaming pipeline. Borrowing is an
    /// explicit transient-only opt-in, and the source lease is owned before
    /// the FFI call. A malformed post-success report is synchronously aborted;
    /// if the provider cannot prove that abort, the bounded slab is leaked
    /// rather than recycled underneath an unknown native reader.
    pub(crate) fn bind_spine_layer_owned(
        &self,
        layer_index: u32,
        generation: u64,
        descriptors: &[SpineTensorDescriptorV1],
        buffers: LayerBuffers,
        retention: SpineLayerRetention,
    ) -> Result<OwnedBoundSpineLayer> {
        let allow_borrow = retention == SpineLayerRetention::Transient;
        let mut leak_source = false;
        let binding = match self.bind_spine_layer_inner(
            layer_index,
            generation,
            descriptors,
            &buffers,
            retention,
            allow_borrow,
            &mut leak_source,
        ) {
            Ok(binding) => binding,
            Err(error) => {
                if leak_source {
                    self.retain_unproven_source_lease(buffers);
                }
                return Err(error);
            }
        };
        Ok(OwnedBoundSpineLayer {
            binding,
            source_lease: buffers,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_spine_layer_inner(
        &self,
        layer_index: u32,
        generation: u64,
        descriptors: &[SpineTensorDescriptorV1],
        buffers: &LayerBuffers,
        retention: SpineLayerRetention,
        allow_borrow: bool,
        leak_source: &mut bool,
    ) -> Result<BoundSpineLayerReport> {
        *leak_source = false;
        if descriptors.is_empty() || descriptors.len() > 64 {
            return Err(DeltafinError::new(
                "native spine binding needs 1..64 descriptors",
            ));
        }
        let quantized = buffers.quantized();
        let scales = buffers.scales();
        let other = buffers.other();
        let allocations = buffers.allocation_lengths();
        if allocations.quantized < quantized.len()
            || allocations.scales < scales.len()
            || allocations.other < other.len()
        {
            return Err(DeltafinError::new(
                "native spine allocation envelope is shorter than its logical bytes",
            ));
        }
        let request = BindSpineLayerRequestV2 {
            struct_size: size_of::<BindSpineLayerRequestV2>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            layer_index,
            flags: match retention {
                SpineLayerRetention::Transient => 0,
                SpineLayerRetention::Retained => BIND_SPINE_RETAIN,
            } | if allow_borrow {
                BIND_SPINE_ALLOW_BORROW
            } else {
                0
            },
            generation,
            descriptors: descriptors.as_ptr(),
            descriptor_count: descriptors.len() as u64,
            quantized: buffers.pointer(BufferKind::Quantized),
            quantized_length: quantized.len() as u64,
            quantized_allocation_length: allocations.quantized as u64,
            scales: buffers.pointer(BufferKind::Scales),
            scales_length: scales.len() as u64,
            scales_allocation_length: allocations.scales as u64,
            other: buffers.pointer(BufferKind::Other),
            other_length: other.len() as u64,
            other_allocation_length: allocations.other as u64,
            reserved: [0; 5],
        };
        let mut report = BindSpineLayerReportV2::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: descriptors and all three immutable LayerBuffers allocations
        // stay live through this call. On success, the source-use disposition
        // precisely declares whether the caller must extend that lifetime.
        let status = unsafe {
            deltafin_provider_bind_spine_layer_v2(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider bind spine layer v2", &error));
        }

        let validation = (|| -> Result<BoundSpineLayerReport> {
            let source_use = match (report.source_use_kind, report.source_use) {
                (SPINE_SOURCE_DETACHED, 0) => SpineSourceUse::Detached,
                (SPINE_SOURCE_BORROWED, handle) if handle != 0 => {
                    SpineSourceUse::Borrowed(SpineSourceUseToken {
                        session_identity: self.identity(),
                        generation,
                        handle,
                    })
                }
                _ => {
                    return Err(DeltafinError::new(
                        "native provider returned an invalid spine source-use disposition",
                    ));
                }
            };

            let expected_quantized_count = descriptors
                .iter()
                .filter(|descriptor| descriptor.encoding == SPINE_ENCODING_ROW_I8_F16_SCALE)
                .count();
            let expected_raw_count = descriptors.len() - expected_quantized_count;
            let expected_quantized_bytes =
                descriptor_bytes(descriptors, SPINE_ENCODING_ROW_I8_F16_SCALE, |descriptor| {
                    descriptor.data_length
                })?;
            let expected_scales_bytes =
                descriptor_bytes(descriptors, SPINE_ENCODING_ROW_I8_F16_SCALE, |descriptor| {
                    descriptor.auxiliary_length
                })?;
            let expected_other_bytes = descriptors
                .iter()
                .filter(|descriptor| {
                    matches!(
                        descriptor.encoding,
                        SPINE_ENCODING_RAW_BF16 | SPINE_ENCODING_RAW_F32
                    )
                })
                .try_fold(0_u64, |total, descriptor| {
                    total.checked_add(descriptor.data_length).ok_or_else(|| {
                        DeltafinError::new("native spine raw-byte report overflowed")
                    })
                })?;
            let borrowed_descriptor = |descriptor: &&SpineTensorDescriptorV1| {
                is_large_bf16_projection_descriptor(descriptor)
            };
            let (expected_borrowed_tensor_count, expected_borrowed_source_bytes) = match source_use
            {
                SpineSourceUse::Detached => (0_usize, 0_u64),
                SpineSourceUse::Borrowed(_) => {
                    if !allow_borrow || retention != SpineLayerRetention::Transient {
                        return Err(DeltafinError::new(
                            "native provider borrowed spine storage outside the transient opt-in",
                        ));
                    }
                    let eligible = descriptors
                        .iter()
                        .filter(borrowed_descriptor)
                        .collect::<Vec<_>>();
                    let bytes = eligible.iter().try_fold(0_u64, |total, descriptor| {
                        total.checked_add(descriptor.data_length).ok_or_else(|| {
                            DeltafinError::new("native borrowed spine source bytes overflowed")
                        })
                    })?;
                    (eligible.len(), bytes)
                }
            };
            if matches!(source_use, SpineSourceUse::Borrowed(_))
                && expected_borrowed_tensor_count == 0
            {
                return Err(DeltafinError::new(
                    "native provider returned a borrowed source use without an eligible matrix",
                ));
            }
            let mut expected_resident_storage_bytes =
                descriptors.iter().try_fold(0_u64, |total, descriptor| {
                    let bytes = spine_descriptor_resident_bytes(
                        descriptor,
                        matches!(source_use, SpineSourceUse::Borrowed(_))
                            && is_large_bf16_projection_descriptor(descriptor),
                    )?;
                    total.checked_add(bytes).ok_or_else(|| {
                        DeltafinError::new("native spine provider-residency sum overflowed")
                    })
                })?;
            if (1_u32..=6).all(|slot| descriptors.iter().any(|descriptor| descriptor.slot == slot))
            {
                expected_resident_storage_bytes = expected_resident_storage_bytes
                    .checked_add((2 * 7_168 * size_of::<f32>()) as u64)
                    .ok_or_else(|| {
                        DeltafinError::new("native prepared residual score bytes overflowed")
                    })?;
            }
            if report.struct_size as usize != size_of::<BindSpineLayerReportV2>()
                || report.abi_version != ABI_VERSION
                || report.layer_index != layer_index
                || report.generation != generation
                || report.tensor_count as usize != descriptors.len()
                || report.quantized_tensor_count as usize != expected_quantized_count
                || report.raw_tensor_count as usize != expected_raw_count
                || report.quantized_bytes != expected_quantized_bytes
                || report.scales_bytes != expected_scales_bytes
                || report.other_bytes != expected_other_bytes
                || report.resident_storage_bytes != expected_resident_storage_bytes
                || report.borrowed_tensor_count as usize != expected_borrowed_tensor_count
                || report.borrowed_source_bytes != expected_borrowed_source_bytes
                || report.reserved != [0; 3]
            {
                return Err(DeltafinError::new(
                    "native provider returned an invalid bind-spine report",
                ));
            }
            Ok(BoundSpineLayerReport {
                layer_index: report.layer_index,
                generation: report.generation,
                tensor_count: report.tensor_count as usize,
                quantized_tensor_count: report.quantized_tensor_count as usize,
                raw_tensor_count: report.raw_tensor_count as usize,
                quantized_bytes: report.quantized_bytes,
                scales_bytes: report.scales_bytes,
                other_bytes: report.other_bytes,
                resident_storage_bytes: report.resident_storage_bytes,
                borrowed_tensor_count: report.borrowed_tensor_count as usize,
                borrowed_source_bytes: report.borrowed_source_bytes,
                retention,
                source_use,
            })
        })();

        if validation.is_err()
            && allow_borrow
            && (report.source_use_kind == SPINE_SOURCE_BORROWED || report.source_use != 0)
        {
            if report.source_use == 0 {
                *leak_source = true;
            } else {
                let token = SpineSourceUseToken {
                    session_identity: self.identity(),
                    generation,
                    handle: report.source_use,
                };
                if self.abort_spine_source_use(token).is_err() {
                    *leak_source = true;
                }
            }
        }
        validation
    }

    pub(crate) fn seal_spine_source_use(&self, token: SpineSourceUseToken) -> Result<()> {
        let report = self.spine_source_use_operation(
            token,
            deltafin_provider_spine_source_use_seal_v2,
            "native provider seal spine source use",
        )?;
        if report.state != SPINE_SOURCE_SEALED || report.ready != 0 {
            return Err(DeltafinError::new(
                "native provider returned an invalid sealed source-use state",
            ));
        }
        Ok(())
    }

    pub(crate) fn try_reclaim_spine_source_use(&self, token: SpineSourceUseToken) -> Result<bool> {
        let report = self.spine_source_use_operation(
            token,
            deltafin_provider_spine_source_use_try_reclaim_v2,
            "native provider reclaim spine source use",
        )?;
        match (report.state, report.ready) {
            (SPINE_SOURCE_SEALED, 0) => Ok(false),
            (SPINE_SOURCE_RECLAIMED, 1) => Ok(true),
            _ => Err(DeltafinError::new(
                "native provider returned an invalid reclaimed source-use state",
            )),
        }
    }

    pub(crate) fn abort_spine_source_use(&self, token: SpineSourceUseToken) -> Result<()> {
        let report = self.spine_source_use_operation(
            token,
            deltafin_provider_spine_source_use_abort_v2,
            "native provider abort spine source use",
        )?;
        if report.state != SPINE_SOURCE_ABORTED || report.ready != 1 {
            return Err(DeltafinError::new(
                "native provider returned an invalid aborted source-use state",
            ));
        }
        Ok(())
    }

    fn spine_source_use_operation(
        &self,
        token: SpineSourceUseToken,
        operation: SpineSourceUseOperationV2,
        name: &'static str,
    ) -> Result<SpineSourceUseReportV2> {
        if token.session_identity != self.identity() || token.generation == 0 || token.handle == 0 {
            return Err(DeltafinError::new(format!(
                "{name} rejected a stale or cross-session token"
            )));
        }
        let request = SpineSourceUseRequestV2 {
            struct_size: size_of::<SpineSourceUseRequestV2>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            source_use: token.handle,
            generation: token.generation,
            flags: 0,
            reserved0: 0,
            reserved: [0; 3],
        };
        let mut report = SpineSourceUseReportV2::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/report/error storage remains live for the complete
        // synchronous control call. No source slab pointer crosses this call.
        let status = unsafe { operation(&request, &mut report, error.as_mut_ptr(), error.len()) };
        if status != 0 {
            return Err(ffi_error(name, &error));
        }
        if report.struct_size as usize != size_of::<SpineSourceUseReportV2>()
            || report.abi_version != ABI_VERSION
            || report.source_use != token.handle
            || report.generation != token.generation
            || report.reserved != [0; 4]
        {
            return Err(DeltafinError::new(format!(
                "{name} returned an invalid source-use report"
            )));
        }
        Ok(report)
    }

    /// Bind one immutable target-global group. Group 1 is exactly slots
    /// 41..43; group 2 is exactly slot 44. Slot 40 is never accepted here.
    pub fn bind_target_globals(
        &self,
        group: TargetGlobalGroup,
        descriptors: &[SpineTensorDescriptorV1],
        buffers: &LayerBuffers,
    ) -> Result<BoundTargetGlobalsReport> {
        if descriptors.is_empty() || descriptors.len() > 3 {
            return Err(DeltafinError::new(
                "native target-global binding needs 1..3 descriptors",
            ));
        }
        let quantized = buffers.quantized();
        let scales = buffers.scales();
        let other = buffers.other();
        let request = BindTargetGlobalsRequestV1 {
            struct_size: size_of::<BindTargetGlobalsRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            group: group.abi_value(),
            flags: 0,
            descriptors: descriptors.as_ptr(),
            descriptor_count: descriptors.len() as u64,
            quantized: nonempty_slice_pointer(quantized),
            quantized_length: quantized.len() as u64,
            scales: nonempty_slice_pointer(scales),
            scales_length: scales.len() as u64,
            other: nonempty_slice_pointer(other),
            other_length: other.len() as u64,
            reserved: [0; 5],
        };
        let mut report = BindTargetGlobalsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: descriptor/slab slices remain immutable and live through
        // the call. Native code clones every byte before returning.
        let status = unsafe {
            deltafin_provider_bind_target_globals_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider bind target globals", &error));
        }
        let expected_quantized_count = descriptors
            .iter()
            .filter(|descriptor| descriptor.encoding == SPINE_ENCODING_ROW_I8_F16_SCALE)
            .count();
        let expected_raw_count = descriptors.len() - expected_quantized_count;
        let expected_quantized_bytes =
            descriptor_bytes(descriptors, SPINE_ENCODING_ROW_I8_F16_SCALE, |descriptor| {
                descriptor.data_length
            })?;
        let expected_scales_bytes =
            descriptor_bytes(descriptors, SPINE_ENCODING_ROW_I8_F16_SCALE, |descriptor| {
                descriptor.auxiliary_length
            })?;
        let expected_other_bytes = descriptors
            .iter()
            .filter(|descriptor| {
                matches!(
                    descriptor.encoding,
                    SPINE_ENCODING_RAW_BF16 | SPINE_ENCODING_RAW_F32
                )
            })
            .try_fold(0_u64, |total, descriptor| {
                total
                    .checked_add(descriptor.data_length)
                    .ok_or_else(|| DeltafinError::new("target-global raw bytes overflowed"))
            })?;
        let expected_resident_storage_bytes =
            descriptors.iter().try_fold(0_u64, |total, descriptor| {
                let bytes = spine_descriptor_resident_bytes(descriptor, false)?;
                total
                    .checked_add(bytes)
                    .ok_or_else(|| DeltafinError::new("target-global residency sum overflowed"))
            })?;
        if report.struct_size as usize != size_of::<BindTargetGlobalsReportV1>()
            || report.abi_version != ABI_VERSION
            || report.group != group.abi_value()
            || report.tensor_count as usize != descriptors.len()
            || report.quantized_tensor_count as usize != expected_quantized_count
            || report.raw_tensor_count as usize != expected_raw_count
            || report.quantized_bytes != expected_quantized_bytes
            || report.scales_bytes != expected_scales_bytes
            || report.other_bytes != expected_other_bytes
            || report.resident_storage_bytes != expected_resident_storage_bytes
            || !(1..=2).contains(&report.groups_ready)
            || report.flags != 0
            || report.reserved != [0; 4]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid target-global bind report",
            ));
        }
        Ok(BoundTargetGlobalsReport {
            group,
            tensor_count: report.tensor_count as usize,
            quantized_tensor_count: report.quantized_tensor_count as usize,
            raw_tensor_count: report.raw_tensor_count as usize,
            quantized_bytes: report.quantized_bytes,
            scales_bytes: report.scales_bytes,
            other_bytes: report.other_bytes,
            resident_storage_bytes: report.resident_storage_bytes,
            groups_ready: report.groups_ready as usize,
        })
    }

    /// Begin one exact target position from an existing provider-owned fp32
    /// [1,7168] row. The native session owns all 93 speculative cache updates.
    pub fn begin_target_position(&self, hidden: &ProviderTensor) -> Result<TargetPosition> {
        require_same_session(&self.inner, &hidden.session, "target hidden tensor")?;
        if hidden.shape() != (1, 7_168) {
            return Err(DeltafinError::new(
                "target begin needs one exact fp32 hidden row of width 7168",
            ));
        }
        let request = TargetBeginRequestV1 {
            struct_size: size_of::<TargetBeginRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            hidden: hidden.handle,
            flags: 0,
            reserved0: 0,
            reserved: [0; 4],
        };
        let mut report = TargetBeginReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request contains only provider-owned handles and native
        // code copies the ATen storage handle before returning.
        let status = unsafe {
            deltafin_provider_target_begin_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider target begin", &error));
        }
        target_position_from_report(&self.inner, report)
    }

    /// Begin directly from the exact 14,336 BF16 bytes returned by the row
    /// reader. Native code memcpy-aligns, promotes, and uploads them in one
    /// coarse call; it never retains this byte slice.
    pub fn begin_target_position_bf16(&self, row: &[u8]) -> Result<TargetPosition> {
        if row.len() != 7_168 * 2 {
            return Err(DeltafinError::new(format!(
                "target BF16 begin needs exactly {} bytes; got {}",
                7_168 * 2,
                row.len()
            )));
        }
        let request = TargetBeginBf16RequestV1 {
            struct_size: size_of::<TargetBeginBf16RequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            data: row.as_ptr(),
            byte_length: row.len() as u64,
            flags: 0,
            reserved0: 0,
            reserved: [0; 3],
        };
        let mut report = TargetBeginReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: row is a valid byte slice for byte_length. Native code first
        // memcpy's into aligned owned BF16 storage and retains no Rust pointer.
        let status = unsafe {
            deltafin_provider_target_begin_bf16_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider target BF16 begin", &error));
        }
        target_position_from_report(&self.inner, report)
    }

    /// Begin one exact layer-major target transaction from adjacent BF16
    /// embedding rows. Native code copies and promotes the complete slice
    /// before returning; no Rust pointer crosses the synchronous call.
    pub fn begin_target_sequence_bf16(
        &self,
        rows: &[u8],
        positions: usize,
        mode: TargetSequenceMode,
    ) -> Result<TargetSequence> {
        self.begin_target_sequence_bf16_options(rows, positions, mode, None, false)
    }

    pub(crate) fn begin_target_sequence_bf16_capturing(
        &self,
        rows: &[u8],
        positions: usize,
        mode: TargetSequenceMode,
        capture: ProposalCapture,
    ) -> Result<TargetSequence> {
        self.begin_target_sequence_bf16_options(rows, positions, mode, Some(capture), false)
    }

    /// Begin an exact Verify transaction whose speculative cache state can
    /// only be published by committing every proposed position. This narrow
    /// contract lets the compiled provider omit prefix-recovery state without
    /// weakening ordinary Verify transactions. A mismatch must cancel this
    /// sequence and rerun its accepted prefix as a new full-commit sequence.
    pub(crate) fn begin_target_sequence_bf16_verify_full_commit_only(
        &self,
        rows: &[u8],
        positions: usize,
        capture: Option<ProposalCapture>,
    ) -> Result<TargetSequence> {
        self.begin_target_sequence_bf16_options(
            rows,
            positions,
            TargetSequenceMode::Verify,
            capture,
            true,
        )
    }

    fn begin_target_sequence_bf16_options(
        &self,
        rows: &[u8],
        positions: usize,
        mode: TargetSequenceMode,
        capture: Option<ProposalCapture>,
        full_commit_only: bool,
    ) -> Result<TargetSequence> {
        if !(1..=ROUTE_MAX_POSITIONS).contains(&positions) {
            return Err(DeltafinError::new(
                "target sequence position count must be in 1..64",
            ));
        }
        let expected_bytes = positions
            .checked_mul(7_168 * 2)
            .ok_or_else(|| DeltafinError::new("target sequence BF16 size overflowed"))?;
        if rows.len() != expected_bytes {
            return Err(DeltafinError::new(format!(
                "target sequence needs {expected_bytes} BF16 bytes for {positions} rows; got {}",
                rows.len()
            )));
        }
        let request = TargetSequenceBeginBf16RequestV1 {
            struct_size: size_of::<TargetSequenceBeginBf16RequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            data: rows.as_ptr(),
            byte_length: rows.len() as u64,
            positions: positions as u32,
            mode: mode.abi_value(),
            flags: capture.map_or(0, ProposalCapture::flag)
                | (if full_commit_only {
                TARGET_SEQUENCE_FULL_COMMIT_ONLY
            } else {
                0
            }),
            reserved0: 0,
            reserved: [0; 4],
        };
        let mut report = TargetSequenceBeginReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: rows is valid for byte_length and native code copies it
        // before returning. Report/error remain writable for the call.
        let status = unsafe {
            deltafin_provider_target_sequence_begin_bf16_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error(
                "native provider target-sequence BF16 begin",
                &error,
            ));
        }
        target_sequence_from_report(
            &self.inner,
            report,
            positions,
            mode,
            capture,
            full_commit_only,
        )
    }

    /// Read back one provider-owned component for exact parity tests. Normal
    /// layer execution consumes the stored tensors in place and never calls
    /// this diagnostic method.
    pub fn read_spine_tensor_f32(
        &self,
        layer_index: u32,
        generation: u64,
        slot: u32,
        component: SpineComponent,
        expected_elements: usize,
    ) -> Result<SpineTensorReadback> {
        if expected_elements == 0 {
            return Err(DeltafinError::new(
                "native spine readback needs a positive element count",
            ));
        }
        let mut values = vec![0.0_f32; expected_elements];
        let request = SpineTensorReadF32V1 {
            struct_size: size_of::<SpineTensorReadF32V1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            generation,
            slot,
            component: match component {
                SpineComponent::Data => SPINE_COMPONENT_DATA,
                SpineComponent::Auxiliary => SPINE_COMPONENT_AUXILIARY,
            },
            destination: values.as_mut_ptr(),
            element_capacity: values.len() as u64,
            flags: 0,
            layer_index,
            reserved: [0; 3],
        };
        let mut report = SpineTensorReadReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: values/report/error are valid writable buffers throughout
        // the call and the provider retains none of their addresses.
        let status = unsafe {
            deltafin_provider_spine_tensor_read_f32_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider spine tensor read", &error));
        }
        let rank = report.rank as usize;
        if report.struct_size as usize != size_of::<SpineTensorReadReportV1>()
            || report.abi_version != ABI_VERSION
            || !(1..=8).contains(&rank)
            || report.element_count as usize != expected_elements
            || report.shape[rank..] != [0; 8][rank..]
            || report.reserved != [0; 1]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid spine readback report",
            ));
        }
        let stored_scalar = match report.stored_scalar_type {
            SPINE_SCALAR_I8 => SpineStoredScalar::I8,
            SPINE_SCALAR_BF16 => SpineStoredScalar::Bf16,
            SPINE_SCALAR_F32 => SpineStoredScalar::F32,
            _ => {
                return Err(DeltafinError::new(
                    "native provider returned an invalid stored spine scalar type",
                ));
            }
        };
        Ok(SpineTensorReadback {
            stored_scalar,
            shape: report.shape[..rank].into(),
            values: values.into_boxed_slice(),
        })
    }

    pub fn prepare_layer<'cache>(
        &self,
        hidden: &ProviderTensor,
        cache: &'cache mut ProviderCache,
        layer_index: u32,
    ) -> Result<PreparedLayer<'cache>> {
        require_same_session(&self.inner, &hidden.session, "hidden tensor")?;
        require_same_session(&self.inner, &cache.session, "cache")?;
        if hidden.rows != cache.rows || hidden.columns != cache.columns {
            return Err(DeltafinError::new(
                "split layer hidden and cache shapes differ",
            ));
        }
        if hidden.rows > self.inner.max_route_positions
            || hidden.columns != self.inner.hidden_columns
        {
            return Err(DeltafinError::new(
                "split layer hidden shape exceeds its session contract",
            ));
        }
        let request = PrepareLayerRequestV1 {
            struct_size: size_of::<PrepareLayerRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.inner.handle,
            hidden: hidden.handle,
            cache: cache.handle,
            layer_index,
            flags: 0,
            reserved: [0; 5],
        };
        let mut mailbox = Box::new(RouteMailboxV1::request());
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/mailbox/error are valid for the call. The provider
        // stores only its own tensor/cache state under the returned ticket ID.
        let status = unsafe {
            deltafin_provider_prepare_layer_v1(
                &request,
                mailbox.as_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider prepare layer", &error));
        }
        let route = validate_route_mailbox(mailbox, &self.inner, hidden, cache)?;
        let ticket = route.raw.ticket;
        Ok(PreparedLayer {
            session: Arc::clone(&self.inner),
            ticket,
            cache,
            route,
        })
    }
}

#[derive(Debug)]
pub struct ProviderTensor {
    session: Arc<SessionInner>,
    handle: u64,
    rows: usize,
    columns: usize,
}

impl ProviderTensor {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub(crate) fn handle_in_session(&self, session: &Arc<SessionInner>) -> Result<u64> {
        require_same_session(session, &self.session, "provider tensor")?;
        Ok(self.handle)
    }

    pub fn read_f32(&self) -> Result<Vec<f32>> {
        let mut destination = vec![0.0_f32; checked_shape(self.rows, self.columns)?];
        let request = TensorReadF32V1 {
            struct_size: size_of::<TensorReadF32V1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            tensor: self.handle,
            destination: destination.as_mut_ptr(),
            element_capacity: destination.len() as u64,
            flags: 0,
            reserved0: 0,
            reserved: [0; 3],
        };
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: destination is writable for element_capacity f32 values and
        // the provider retains no pointer after returning.
        let status = unsafe {
            deltafin_provider_tensor_read_f32_v1(&request, error.as_mut_ptr(), error.len())
        };
        if status != 0 {
            return Err(ffi_error("native provider tensor read", &error));
        }
        Ok(destination)
    }
}

impl Drop for ProviderTensor {
    fn drop(&mut self) {
        if self.handle != 0 {
            let _ = release_resource(
                deltafin_provider_tensor_release_v1,
                self.session.handle,
                self.handle,
                "native provider tensor release",
            );
            self.handle = 0;
        }
    }
}

fn target_position_from_report(
    session: &Arc<SessionInner>,
    report: TargetBeginReportV1,
) -> Result<TargetPosition> {
    if report.struct_size as usize != size_of::<TargetBeginReportV1>()
        || report.abi_version != ABI_VERSION
        || report.position == 0
        || report.next_layer != 0
        || report.state != TARGET_STATE_ACTIVE
        || report.kda_cache_count != 69
        || report.mla_cache_count != 24
        || report.reserved != [0; 4]
    {
        if report.position != 0 {
            let _ = release_resource(
                deltafin_provider_target_cancel_v1,
                session.handle,
                report.position,
                "invalid native target position cancel",
            );
        }
        return Err(DeltafinError::new(
            "native provider returned an invalid target-begin report",
        ));
    }
    Ok(TargetPosition {
        session: Arc::clone(session),
        handle: report.position,
        next_layer: 0,
        waiting: None,
    })
}

fn target_sequence_from_report(
    session: &Arc<SessionInner>,
    report: TargetSequenceBeginReportV1,
    positions: usize,
    mode: TargetSequenceMode,
    capture: Option<ProposalCapture>,
    full_commit_only: bool,
) -> Result<TargetSequence> {
    if report.struct_size as usize != size_of::<TargetSequenceBeginReportV1>()
        || report.abi_version != ABI_VERSION
        || report.sequence == 0
        || report.positions as usize != positions
        || report.mode != mode.abi_value()
        || report.next_layer != 0
        || report.state != TARGET_SEQUENCE_STATE_ACTIVE
        || report.kda_cache_count != 69
        || report.mla_cache_count != 24
        || report.reserved != [0; 3]
    {
        if report.sequence != 0 {
            let _ = release_resource(
                deltafin_provider_target_sequence_cancel_v1,
                session.handle,
                report.sequence,
                "invalid native target-sequence cancel",
            );
        }
        return Err(DeltafinError::new(
            "native provider returned an invalid target-sequence begin report",
        ));
    }
    Ok(TargetSequence {
        session: Arc::clone(session),
        handle: report.sequence,
        mode,
        position_count: positions,
        next_layer: 0,
        state: TargetSequenceState::Active,
        waiting: None,
        expert_plan: None,
        capture,
        full_commit_only,
    })
}

fn resolve_target_backend(
    device: Device,
    requested: TargetExpertBackend,
) -> Result<TargetExpertBackend> {
    match requested {
        TargetExpertBackend::Auto => Ok(match device {
            Device::Mps => TargetExpertBackend::Metal,
            Device::Cpu => TargetExpertBackend::Cpu,
            Device::Cuda(_) => TargetExpertBackend::Auto,
        }),
        TargetExpertBackend::Cpu => Ok(TargetExpertBackend::Cpu),
        TargetExpertBackend::Metal if device == Device::Mps => Ok(TargetExpertBackend::Metal),
        TargetExpertBackend::Metal => Err(DeltafinError::new(
            "Metal target experts require the selected MPS provider",
        )),
        TargetExpertBackend::Cuda if matches!(device, Device::Cuda(_)) => {
            Ok(TargetExpertBackend::Cuda)
        }
        TargetExpertBackend::Cuda => Err(DeltafinError::new(
            "CUDA target experts require the selected CUDA provider",
        )),
    }
}

/// RAII ownership of one unpublished K3 position. Dropping or explicitly
/// cancelling it rolls back every staged KDA/MLA cache update. Only a valid
/// greedy finish publishes all 93 cache updates together.
#[derive(Debug)]
pub struct TargetPosition {
    session: Arc<SessionInner>,
    handle: u64,
    next_layer: u32,
    waiting: Option<TargetRoute>,
}

impl TargetPosition {
    pub const fn next_layer(&self) -> u32 {
        self.next_layer
    }

    pub const fn waiting_for_experts(&self) -> bool {
        self.waiting.is_some()
    }

    pub fn prepare_layer(
        &mut self,
        layer_index: u32,
        spine_generation: u64,
    ) -> Result<TargetLayerPrepare> {
        if self.handle == 0 || self.waiting.is_some() {
            return Err(DeltafinError::new(
                "target position cannot prepare while closed or waiting for experts",
            ));
        }
        if layer_index != self.next_layer || layer_index >= 93 || spine_generation == 0 {
            return Err(DeltafinError::new(
                "target layer/generation is outside the active position order",
            ));
        }
        let request = TargetPrepareRequestV1 {
            struct_size: size_of::<TargetPrepareRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            position: self.handle,
            spine_generation,
            layer_index,
            flags: 0,
            reserved: [0; 3],
        };
        let mut report = TargetPrepareReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: all weight/cache/tensor objects are provider-owned. The
        // fixed report and error buffers are writable for the whole call.
        let status = unsafe {
            deltafin_provider_target_prepare_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider target prepare", &error));
        }
        let common_valid = report.struct_size as usize == size_of::<TargetPrepareReportV1>()
            && report.abi_version == ABI_VERSION
            && report.position == self.handle
            && report.spine_generation == spine_generation
            && report.layer_index == layer_index
            && report.reserved == [0; 3];
        if !common_valid {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native provider returned an invalid target-prepare report",
            ));
        }
        match report.kind {
            TARGET_DENSE_COMPLETE
                if layer_index == 0
                    && report.next_layer == 1
                    && report.top_k == 0
                    && report.ordered_experts == [0; ROUTE_TOP_K]
                    && report.ordered_weight_bits == [0; ROUTE_TOP_K] =>
            {
                self.next_layer = 1;
                Ok(TargetLayerPrepare::DenseCompleted { next_layer: 1 })
            }
            TARGET_EXPERTS_REQUIRED
                if layer_index != 0
                    && report.next_layer == layer_index
                    && report.top_k as usize == ROUTE_TOP_K =>
            {
                for (index, &expert) in report.ordered_experts.iter().enumerate() {
                    if expert >= 896 || report.ordered_experts[..index].contains(&expert) {
                        self.cancel_after_invalid_report();
                        return Err(DeltafinError::new(
                            "native target route contains an invalid or repeated expert",
                        ));
                    }
                    let weight = f32::from_bits(report.ordered_weight_bits[index]);
                    if !weight.is_finite() || weight < 0.0 {
                        self.cancel_after_invalid_report();
                        return Err(DeltafinError::new(
                            "native target route contains an invalid fp32 weight",
                        ));
                    }
                }
                let route = TargetRoute {
                    layer_index,
                    spine_generation,
                    ordered_experts: report.ordered_experts,
                    ordered_weight_bits: report.ordered_weight_bits,
                };
                self.waiting = Some(route);
                Ok(TargetLayerPrepare::ExpertsRequired(route))
            }
            _ => {
                self.cancel_after_invalid_report();
                Err(DeltafinError::new(
                    "native provider returned an invalid target-prepare state",
                ))
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "this exact-provider boundary mirrors the versioned native ABI without hidden defaults"
    )]
    pub fn finish_experts(
        &mut self,
        route: &TargetRoute,
        canonical_expert_ids: &[u16; ROUTE_TOP_K],
        expert_major_bytes: &[u8],
        expert_layout: ExpertStorageLayout,
        backend: TargetExpertBackend,
        cpu_threads: usize,
        metal_shader_path: Option<&str>,
    ) -> Result<()> {
        let Some(waiting) = self.waiting else {
            return Err(DeltafinError::new(
                "target position is not waiting for expert bytes",
            ));
        };
        if self.handle == 0 || waiting != *route {
            return Err(DeltafinError::new(
                "target expert finish received a stale route",
            ));
        }
        let backend = resolve_target_backend(self.session.device, backend)?;
        if expert_layout == ExpertStorageLayout::Scale4V2 && backend != TargetExpertBackend::Metal {
            return Err(DeltafinError::new(
                "scale4-v2 target experts require the selected Metal backend",
            ));
        }
        for (index, &expert) in canonical_expert_ids.iter().enumerate() {
            if expert >= 896 || (index != 0 && canonical_expert_ids[index - 1] >= expert) {
                return Err(DeltafinError::new(
                    "target expert IDs must be 16 unique ascending canonical IDs",
                ));
            }
        }
        let (expert_layout_abi, expert_span_bytes) = expert_layout_abi(expert_layout);
        let expected_bytes = usize::try_from(expert_span_bytes)
            .ok()
            .and_then(|span| span.checked_mul(ROUTE_TOP_K))
            .ok_or_else(|| DeltafinError::new("target expert byte count overflowed"))?;
        if expert_major_bytes.len() != expected_bytes {
            return Err(DeltafinError::new(format!(
                "target expert batch has {} bytes, expected {expected_bytes}",
                expert_major_bytes.len()
            )));
        }
        let cpu_threads = u32::try_from(cpu_threads)
            .map_err(|_| DeltafinError::new("target expert thread count exceeds ABI u32"))?;
        if cpu_threads == 0 || cpu_threads > 1024 {
            return Err(DeltafinError::new(
                "target expert thread count must be in 1..1024",
            ));
        }
        let shader_bytes = metal_shader_path
            .filter(|path| !path.is_empty())
            .map(str::as_bytes)
            .unwrap_or_default();
        if shader_bytes.contains(&0) {
            return Err(DeltafinError::new(
                "target Metal shader path contains an embedded NUL",
            ));
        }
        if shader_bytes.len() > 4096 {
            return Err(DeltafinError::new(
                "target Metal shader path exceeds 4096 bytes",
            ));
        }
        let request = TargetFinishExpertsRequestV1 {
            struct_size: size_of::<TargetFinishExpertsRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            position: self.handle,
            spine_generation: route.spine_generation,
            layer_index: route.layer_index,
            expert_backend: backend.abi_value(),
            cpu_threads,
            expert_count: ROUTE_TOP_K as u32,
            expert_ids: *canonical_expert_ids,
            expert_major_bytes: expert_major_bytes.as_ptr(),
            expert_major_length: expert_major_bytes.len() as u64,
            metal_shader_path: if shader_bytes.is_empty() {
                ptr::null()
            } else {
                shader_bytes.as_ptr().cast()
            },
            metal_shader_path_length: shader_bytes.len() as u64,
            flags: 0,
            expert_layout: expert_layout_abi,
            expert_span_bytes,
            reserved: [0; 4],
        };
        let mut report = TargetFinishExpertsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: both expert slices and optional shader bytes remain valid
        // through this synchronous call; native code retains no pointer.
        let status = unsafe {
            deltafin_provider_target_finish_experts_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider target finish experts", &error));
        }
        let expected_next = route.layer_index + 1;
        let expected_state = if expected_next == 93 {
            TARGET_STATE_READY_FOR_TAIL
        } else {
            TARGET_STATE_ACTIVE
        };
        if report.struct_size as usize != size_of::<TargetFinishExpertsReportV1>()
            || report.abi_version != ABI_VERSION
            || report.position != self.handle
            || report.completed_layer != route.layer_index
            || report.next_layer != expected_next
            || report.state != expected_state
            || report.flags != 0
            || report.reserved != [0; 4]
        {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native provider returned an invalid target expert-finish report",
            ));
        }
        self.waiting = None;
        self.next_layer = expected_next;
        Ok(())
    }

    pub fn finish_greedy(mut self) -> Result<u32> {
        if self.handle == 0 || self.waiting.is_some() || self.next_layer != 93 {
            return Err(DeltafinError::new(
                "target position is not ready for its greedy tail",
            ));
        }
        let request = ResourceRequestV1::new(self.session.handle, self.handle);
        let mut report = TargetGreedyReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request names the one provider-owned transaction and
        // report/error are writable for the complete call.
        let status = unsafe {
            deltafin_provider_target_finish_greedy_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider target greedy finish", &error));
        }
        let consumed_handle = self.handle;
        self.handle = 0;
        if report.struct_size as usize != size_of::<TargetGreedyReportV1>()
            || report.abi_version != ABI_VERSION
            || report.position != consumed_handle
            || report.state != TARGET_STATE_COMMITTED
            || report.committed_positions == 0
            || report.reserved != [0; 4]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid target greedy report",
            ));
        }
        Ok(report.token_id)
    }

    pub fn cancel(mut self) -> Result<()> {
        if self.handle == 0 {
            return Ok(());
        }
        let result = release_resource(
            deltafin_provider_target_cancel_v1,
            self.session.handle,
            self.handle,
            "native provider target cancel",
        );
        if result.is_ok() {
            self.handle = 0;
        }
        result
    }

    fn cancel_after_invalid_report(&mut self) {
        if self.handle == 0 {
            return;
        }
        let _ = release_resource(
            deltafin_provider_target_cancel_v1,
            self.session.handle,
            self.handle,
            "invalid native target position cancel",
        );
        self.handle = 0;
        self.waiting = None;
    }
}

impl Drop for TargetPosition {
    fn drop(&mut self) {
        if self.handle != 0 {
            let _ = release_resource(
                deltafin_provider_target_cancel_v1,
                self.session.handle,
                self.handle,
                "native provider target cancel",
            );
            self.handle = 0;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TargetSequenceWaiting {
    layer_index: u32,
    spine_generation: u64,
    next_row: usize,
}

enum TargetExpertTileStorage<'a> {
    Contiguous(&'a [u8]),
    Scattered(&'a [&'a [u8]]),
}

#[derive(Debug)]
struct TargetSequenceExpertPlanLease {
    handle: AtomicU64,
}

impl TargetSequenceExpertPlanLease {
    fn new(handle: u64) -> Self {
        Self {
            handle: AtomicU64::new(handle),
        }
    }

    fn handle(&self) -> u64 {
        self.handle.load(Ordering::Acquire)
    }

    fn consume(&self) -> u64 {
        self.handle.swap(0, Ordering::AcqRel)
    }
}

/// One authenticated, pre-I/O raw-v1 CUDA residency snapshot. The missing
/// IDs are the only expert spans a caller may read. This resource is
/// session-owned rather than sequence-owned, so it remains safely releasable
/// after its target sequence has been cancelled or dropped.
#[derive(Debug)]
pub struct TargetSequenceExpertPlan {
    session: Arc<SessionInner>,
    lease: Arc<TargetSequenceExpertPlanLease>,
    sequence: u64,
    spine_generation: u64,
    layer_index: u32,
    first_row: usize,
    row_count: usize,
    canonical_experts: Box<[u16]>,
    missing_experts: Box<[u16]>,
    effective_backend: TargetExpertBackend,
    cache_capacity_experts: usize,
    residency_enabled: bool,
}

impl TargetSequenceExpertPlan {
    pub fn missing_experts(&self) -> &[u16] {
        &self.missing_experts
    }

    pub const fn effective_backend(&self) -> TargetExpertBackend {
        self.effective_backend
    }

    pub const fn cache_capacity_experts(&self) -> usize {
        self.cache_capacity_experts
    }

    pub const fn residency_enabled(&self) -> bool {
        self.residency_enabled
    }

    pub const fn first_row(&self) -> usize {
        self.first_row
    }

    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn is_live(&self) -> bool {
        self.lease.handle() != 0
    }

    /// Explicit release is independent of target-sequence liveness. On an
    /// error the handle remains armed so Drop makes one final best-effort
    /// release attempt while its Arc still keeps the native session alive.
    pub fn release(self) -> Result<()> {
        let handle = self.lease.handle();
        if handle == 0 {
            return Ok(());
        }
        let result = release_resource(
            deltafin_provider_moe_plan_release_v1,
            self.session.handle,
            handle,
            "native provider expert-plan release",
        );
        if result.is_ok() {
            let _ = self.lease.consume();
        }
        result
    }
}

impl Drop for TargetSequenceExpertPlan {
    fn drop(&mut self) {
        let handle = self.lease.consume();
        if handle != 0 {
            let _ = release_resource(
                deltafin_provider_moe_plan_release_v1,
                self.session.handle,
                handle,
                "native provider expert-plan release",
            );
        }
    }
}

#[derive(Debug)]
struct ValidatedTargetSequenceExpertPlanReport {
    effective_backend: TargetExpertBackend,
    missing_experts: Box<[u16]>,
    cache_capacity_experts: usize,
    residency_enabled: bool,
}

fn target_sequence_route_union(
    mailbox: &TargetSequenceMailbox,
    first_row: usize,
    row_count: usize,
    maximum_experts: usize,
) -> Result<Box<[u16]>> {
    if row_count == 0
        || !(K3_EXPERT_TOP_K..=TARGET_SEQUENCE_MAX_EXPERTS_V2).contains(&maximum_experts)
        || first_row
            .checked_add(row_count)
            .is_none_or(|end| end > mailbox.position_count())
    {
        return Err(DeltafinError::new(
            "target-sequence route union is outside its mailbox rows",
        ));
    }
    let mut union = Vec::with_capacity(row_count * ROUTE_TOP_K);
    for route in &mailbox.routes[first_row..first_row + row_count] {
        if route.layer_index != mailbox.layer_index
            || route.spine_generation != mailbox.spine_generation
        {
            return Err(DeltafinError::new(
                "target-sequence mailbox route identity is inconsistent",
            ));
        }
        for (index, &expert) in route.ordered_experts.iter().enumerate() {
            if expert >= 896 || route.ordered_experts[..index].contains(&expert) {
                return Err(DeltafinError::new(
                    "target-sequence mailbox contains an invalid routed expert",
                ));
            }
            union.push(expert);
        }
    }
    union.sort_unstable();
    union.dedup();
    if union.is_empty() || union.len() > maximum_experts {
        return Err(DeltafinError::new(format!(
            "target-sequence route union exceeds its {maximum_experts}-expert ABI bound"
        )));
    }
    Ok(union.into_boxed_slice())
}

fn validate_target_sequence_expert_plan_report(
    report: &TargetSequencePlanExpertsReportV1,
    requested_backend: TargetExpertBackend,
    spine_generation: u64,
    layer_index: u32,
    first_row: usize,
    row_count: usize,
    canonical_experts: &[u16],
) -> Result<ValidatedTargetSequenceExpertPlanReport> {
    let missing_count = usize::try_from(report.missing_count)
        .map_err(|_| DeltafinError::new("native expert-plan miss count exceeds usize"))?;
    if report.struct_size as usize != size_of::<TargetSequencePlanExpertsReportV1>()
        || report.abi_version != ABI_VERSION
        || report.plan == 0
        || report.spine_generation != spine_generation
        || report.layer_index != layer_index
        || report.first_row as usize != first_row
        || report.row_count as usize != row_count
        || missing_count > TARGET_SEQUENCE_MAX_EXPERTS
        || report.flags != 0
        || report.reserved != [0; 3]
        || report.residency_enabled > 1
        || report.missing_experts[missing_count..]
            != [0; TARGET_SEQUENCE_MAX_EXPERTS][missing_count..]
    {
        return Err(DeltafinError::new(
            "native provider returned an invalid target-sequence expert-plan report",
        ));
    }
    let effective_backend = match report.effective_backend {
        TARGET_EXPERT_CPU => TargetExpertBackend::Cpu,
        TARGET_EXPERT_CUDA => TargetExpertBackend::Cuda,
        _ => {
            return Err(DeltafinError::new(
                "native expert plan did not freeze exact CPU or CUDA",
            ));
        }
    };
    if requested_backend == TargetExpertBackend::Cuda
        && effective_backend != TargetExpertBackend::Cuda
    {
        return Err(DeltafinError::new(
            "explicit CUDA expert planning was silently downgraded",
        ));
    }
    let missing = &report.missing_experts[..missing_count];
    for (index, &expert) in missing.iter().enumerate() {
        if expert >= 896
            || (index != 0 && missing[index - 1] >= expert)
            || canonical_experts.binary_search(&expert).is_err()
        {
            return Err(DeltafinError::new(
                "native expert-plan misses are not a canonical route subset",
            ));
        }
    }
    let cache_capacity_experts = usize::try_from(report.cache_capacity_experts)
        .map_err(|_| DeltafinError::new("native CUDA cache capacity exceeds usize"))?;
    let residency_enabled = report.residency_enabled != 0;
    match effective_backend {
        TargetExpertBackend::Cpu
            if cache_capacity_experts == 0
                && !residency_enabled
                && missing == canonical_experts => {}
        TargetExpertBackend::Cuda
            if residency_enabled == (cache_capacity_experts != 0)
                && (cache_capacity_experts != 0 || missing == canonical_experts) => {}
        _ => {
            return Err(DeltafinError::new(
                "native expert-plan cache state disagrees with its frozen backend",
            ));
        }
    }
    Ok(ValidatedTargetSequenceExpertPlanReport {
        effective_backend,
        missing_experts: missing.to_vec().into_boxed_slice(),
        cache_capacity_experts,
        residency_enabled,
    })
}

fn borrowed_bytes_pointer(bytes: &[u8]) -> *const u8 {
    if bytes.is_empty() {
        ptr::null()
    } else {
        bytes.as_ptr()
    }
}

fn validate_target_sequence_commit_prefix(
    mode: TargetSequenceMode,
    position_count: usize,
    full_commit_only: bool,
    positions: usize,
) -> Result<()> {
    if full_commit_only && positions != position_count {
        return Err(DeltafinError::new(
            "full-commit-only target sequence cannot commit a partial prefix",
        ));
    }
    if positions > position_count
        || (mode == TargetSequenceMode::Prefill && positions != position_count)
    {
        return Err(DeltafinError::new(
            "target-sequence commit prefix is invalid for its mode",
        ));
    }
    Ok(())
}

/// RAII ownership of one unpublished layer-major K3 transaction. The native
/// provider owns every activation and speculative cache state. Dropping this
/// value cancels the complete transaction; only commit_prefix publishes it.
#[derive(Debug)]
pub struct TargetSequence {
    session: Arc<SessionInner>,
    handle: u64,
    mode: TargetSequenceMode,
    position_count: usize,
    next_layer: u32,
    state: TargetSequenceState,
    waiting: Option<TargetSequenceWaiting>,
    expert_plan: Option<Arc<TargetSequenceExpertPlanLease>>,
    capture: Option<ProposalCapture>,
    full_commit_only: bool,
}

impl TargetSequence {
    pub const fn mode(&self) -> TargetSequenceMode {
        self.mode
    }

    pub const fn position_count(&self) -> usize {
        self.position_count
    }

    pub const fn next_layer(&self) -> u32 {
        self.next_layer
    }

    pub const fn state(&self) -> TargetSequenceState {
        self.state
    }

    pub const fn full_commit_only(&self) -> bool {
        self.full_commit_only
    }

    pub const fn waiting_for_experts(&self) -> bool {
        self.waiting.is_some()
    }

    fn has_live_expert_plan(&mut self) -> bool {
        let live = self
            .expert_plan
            .as_ref()
            .is_some_and(|lease| lease.handle() != 0);
        if !live {
            self.expert_plan = None;
        }
        live
    }

    pub fn prepare_layer(
        &mut self,
        layer_index: u32,
        spine_generation: u64,
    ) -> Result<TargetSequenceLayerPrepare> {
        if self.handle == 0 || self.state != TargetSequenceState::Active || self.waiting.is_some() {
            return Err(DeltafinError::new(
                "target sequence cannot prepare while closed or waiting for experts",
            ));
        }
        if layer_index != self.next_layer || layer_index >= 93 || spine_generation == 0 {
            return Err(DeltafinError::new(
                "target-sequence layer/generation is outside the active order",
            ));
        }
        let request = TargetSequencePrepareRequestV1 {
            struct_size: size_of::<TargetSequencePrepareRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            sequence: self.handle,
            spine_generation,
            layer_index,
            flags: 0,
            reserved: [0; 3],
        };
        let mut report = Box::new(TargetSequencePrepareReportV1::request());
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request contains only provider-owned handles. The fixed
        // report and error buffers remain writable throughout the call.
        let status = unsafe {
            deltafin_provider_target_sequence_prepare_v1(
                &request,
                report.as_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            let failure = ffi_error("native provider target-sequence prepare", &error);
            self.cancel_after_invalid_report();
            return Err(failure);
        }
        let common_valid = report.struct_size as usize
            == size_of::<TargetSequencePrepareReportV1>()
            && report.abi_version == ABI_VERSION
            && report.sequence == self.handle
            && report.spine_generation == spine_generation
            && report.layer_index == layer_index
            && report.positions as usize == self.position_count
            && report.flags == 0
            && report.reserved == [0; 4];
        if !common_valid {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native provider returned an invalid target-sequence prepare report",
            ));
        }
        let used_edges = self
            .position_count
            .checked_mul(ROUTE_TOP_K)
            .ok_or_else(|| DeltafinError::new("target-sequence route count overflowed"))?;
        match report.kind {
            TARGET_DENSE_COMPLETE
                if layer_index == 0
                    && report.next_layer == 1
                    && report.top_k == 0
                    && report.ordered_experts == [0; ROUTE_MAX_EDGES]
                    && report.ordered_weight_bits == [0; ROUTE_MAX_EDGES] =>
            {
                self.next_layer = 1;
                Ok(TargetSequenceLayerPrepare::DenseCompleted { next_layer: 1 })
            }
            TARGET_EXPERTS_REQUIRED
                if layer_index != 0
                    && report.next_layer == layer_index
                    && report.top_k as usize == ROUTE_TOP_K
                    && report.ordered_experts[used_edges..]
                        == [0; ROUTE_MAX_EDGES][used_edges..]
                    && report.ordered_weight_bits[used_edges..]
                        == [0; ROUTE_MAX_EDGES][used_edges..] =>
            {
                let mut routes = Vec::with_capacity(self.position_count);
                for row in 0..self.position_count {
                    let first = row * ROUTE_TOP_K;
                    let last = first + ROUTE_TOP_K;
                    let mut ordered_experts = [0_u16; ROUTE_TOP_K];
                    let mut ordered_weight_bits = [0_u32; ROUTE_TOP_K];
                    ordered_experts.copy_from_slice(&report.ordered_experts[first..last]);
                    ordered_weight_bits.copy_from_slice(&report.ordered_weight_bits[first..last]);
                    for (index, &expert) in ordered_experts.iter().enumerate() {
                        if expert >= 896 || ordered_experts[..index].contains(&expert) {
                            self.cancel_after_invalid_report();
                            return Err(DeltafinError::new(
                                "native target-sequence route contains an invalid or repeated expert",
                            ));
                        }
                        let weight = f32::from_bits(ordered_weight_bits[index]);
                        if !weight.is_finite() || weight < 0.0 {
                            self.cancel_after_invalid_report();
                            return Err(DeltafinError::new(
                                "native target-sequence route contains an invalid fp32 weight",
                            ));
                        }
                    }
                    routes.push(TargetRoute {
                        layer_index,
                        spine_generation,
                        ordered_experts,
                        ordered_weight_bits,
                    });
                }
                self.waiting = Some(TargetSequenceWaiting {
                    layer_index,
                    spine_generation,
                    next_row: 0,
                });
                self.state = TargetSequenceState::WaitingForExperts;
                Ok(TargetSequenceLayerPrepare::ExpertsRequired(
                    TargetSequenceMailbox {
                        layer_index,
                        spine_generation,
                        routes: routes.into_boxed_slice(),
                    },
                ))
            }
            _ => {
                self.cancel_after_invalid_report();
                Err(DeltafinError::new(
                    "native provider returned an invalid target-sequence prepare state",
                ))
            }
        }
    }

    /// Take at most one scheduling-only next-layer read hint. A normal
    /// predictor miss returns `None`; malformed provider output cancels the
    /// transaction rather than allowing advisory IDs to escape their bounds.
    pub fn take_prefetch_hint(&mut self) -> Result<Option<TargetSequencePrefetchHint>> {
        let Some(waiting) = self.waiting else {
            return Ok(None);
        };
        if self.handle == 0 || self.state != TargetSequenceState::WaitingForExperts {
            return Err(DeltafinError::new(
                "target sequence is not waiting at a prefetch-hint boundary",
            ));
        }
        let request = ResourceRequestV1::new(self.session.handle, self.handle);
        let mut report = TargetSequencePrefetchHintReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request contains only provider-owned handles and the
        // report/error buffers are writable for the synchronous call.
        let status = unsafe {
            deltafin_provider_target_sequence_take_prefetch_hint_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            // Prediction is optional. A provider-side fail-soft miss should be
            // encoded as an empty successful report; an ABI failure is not
            // safe to reinterpret and therefore cancels this transaction.
            let failure = ffi_error("native provider target prefetch hint", &error);
            self.cancel_after_invalid_report();
            return Err(failure);
        }
        if report.struct_size as usize != size_of::<TargetSequencePrefetchHintReportV1>()
            || report.abi_version != ABI_VERSION
            || report.sequence != self.handle
            || report.flags != 0
            || report.reserved != [0; 4]
            || report.expert_count as usize > PILOT_MAX_PREFETCH
        {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native provider returned an invalid prefetch-hint report",
            ));
        }
        let count = report.expert_count as usize;
        if count == 0 {
            if report.source_layer != 0
                || report.target_layer != 0
                || report.expert_ids != [0; PILOT_MAX_PREFETCH]
            {
                self.cancel_after_invalid_report();
                return Err(DeltafinError::new(
                    "empty native prefetch hint contains nonzero advisory state",
                ));
            }
            return Ok(None);
        }
        if !(ROUTE_TOP_K..=PILOT_MAX_PREFETCH).contains(&count)
            || report.source_layer != waiting.layer_index
            || report.target_layer != waiting.layer_index + 1
            || report.target_layer >= 93
            || report.expert_ids[count..] != [0; PILOT_MAX_PREFETCH][count..]
        {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native prefetch hint disagrees with the active next-layer boundary",
            ));
        }
        for (index, &expert) in report.expert_ids[..count].iter().enumerate() {
            if expert >= 896 || (index != 0 && report.expert_ids[index - 1] >= expert) {
                self.cancel_after_invalid_report();
                return Err(DeltafinError::new(
                    "native prefetch hint IDs are not canonical ascending experts",
                ));
            }
        }
        Ok(Some(TargetSequencePrefetchHint {
            source_layer: report.source_layer,
            target_layer: report.target_layer,
            expert_count: count as u8,
            expert_ids: report.expert_ids,
        }))
    }

    /// Freeze the raw-v1 CUDA/Auto backend and authenticate its exact cache
    /// misses before the caller opens any expert files. A CUDA cache miss is
    /// never inferred from byte length: `missing_experts()` is the complete
    /// and only ordered read list for the matching finish call.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_expert_tile(
        &mut self,
        mailbox: &TargetSequenceMailbox,
        first_row: usize,
        row_count: usize,
        canonical_expert_ids: &[u16],
        backend: TargetExpertBackend,
        cpu_threads: usize,
        metal_shader_path: Option<&str>,
    ) -> Result<TargetSequenceExpertPlan> {
        let Some(waiting) = self.waiting else {
            return Err(DeltafinError::new(
                "target sequence is not waiting at an expert-plan boundary",
            ));
        };
        if self.handle == 0
            || self.state != TargetSequenceState::WaitingForExperts
            || !matches!(self.session.device, Device::Cuda(_))
            || !matches!(
                backend,
                TargetExpertBackend::Auto | TargetExpertBackend::Cuda
            )
            || waiting.layer_index != mailbox.layer_index
            || waiting.spine_generation != mailbox.spine_generation
            || mailbox.position_count() != self.position_count
            || first_row != waiting.next_row
            || !(1..=TARGET_SEQUENCE_MAX_TILE_ROWS).contains(&row_count)
            || first_row
                .checked_add(row_count)
                .is_none_or(|end| end > self.position_count)
        {
            return Err(DeltafinError::new(
                "target expert planning requires the active CUDA row cursor and Auto/CUDA raw-v1 backend",
            ));
        }
        if self.has_live_expert_plan() {
            return Err(DeltafinError::new(
                "target sequence already owns a live expert plan",
            ));
        }
        let canonical = target_sequence_route_union(
            mailbox,
            first_row,
            row_count,
            TARGET_SEQUENCE_MAX_EXPERTS,
        )?;
        if canonical.as_ref() != canonical_expert_ids {
            return Err(DeltafinError::new(
                "target expert plan IDs are not the exact canonical route union",
            ));
        }
        let cpu_threads = u32::try_from(cpu_threads)
            .map_err(|_| DeltafinError::new("target expert thread count exceeds ABI u32"))?;
        if cpu_threads == 0 || cpu_threads > 1024 {
            return Err(DeltafinError::new(
                "target expert thread count must be in 1..1024",
            ));
        }
        let shader_bytes = metal_shader_path
            .filter(|path| !path.is_empty())
            .map(str::as_bytes)
            .unwrap_or_default();
        if shader_bytes.contains(&0) || shader_bytes.len() > 4096 {
            return Err(DeltafinError::new(
                "target Metal shader path is invalid or exceeds 4096 bytes",
            ));
        }
        let shader_pointer = if shader_bytes.is_empty() {
            ptr::null()
        } else {
            shader_bytes.as_ptr().cast()
        };
        let mut expert_ids = [0_u16; TARGET_SEQUENCE_MAX_EXPERTS];
        expert_ids[..canonical.len()].copy_from_slice(&canonical);
        let request = TargetSequencePlanExpertsRequestV1 {
            struct_size: size_of::<TargetSequencePlanExpertsRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            sequence: self.handle,
            spine_generation: waiting.spine_generation,
            layer_index: waiting.layer_index,
            first_row: first_row as u32,
            row_count: row_count as u32,
            expert_backend: backend.abi_value(),
            cpu_threads,
            expert_count: canonical.len() as u32,
            flags: 0,
            expert_ids,
            metal_shader_path: shader_pointer,
            metal_shader_path_length: shader_bytes.len() as u64,
            reserved: [0; 4],
        };
        let mut report = TargetSequencePlanExpertsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request contains provider handles plus borrowed shader
        // bytes that remain live for this synchronous planning call.
        let status = unsafe {
            deltafin_provider_target_sequence_plan_experts_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error(
                "native provider target-sequence expert plan",
                &error,
            ));
        }
        let validated = match validate_target_sequence_expert_plan_report(
            &report,
            backend,
            waiting.spine_generation,
            waiting.layer_index,
            first_row,
            row_count,
            &canonical,
        ) {
            Ok(validated) => validated,
            Err(failure) => {
                if report.plan != 0 {
                    let _ = release_resource(
                        deltafin_provider_moe_plan_release_v1,
                        self.session.handle,
                        report.plan,
                        "invalid native expert-plan release",
                    );
                }
                self.cancel_after_invalid_report();
                return Err(failure);
            }
        };
        let lease = Arc::new(TargetSequenceExpertPlanLease::new(report.plan));
        self.expert_plan = Some(Arc::clone(&lease));
        Ok(TargetSequenceExpertPlan {
            session: Arc::clone(&self.session),
            lease,
            sequence: self.handle,
            spine_generation: waiting.spine_generation,
            layer_index: waiting.layer_index,
            first_row,
            row_count,
            canonical_experts: canonical,
            missing_experts: validated.missing_experts,
            effective_backend: validated.effective_backend,
            cache_capacity_experts: validated.cache_capacity_experts,
            residency_enabled: validated.residency_enabled,
        })
    }

    /// Finish one planned raw-v1 tile using exactly one span per reported
    /// miss. The plan is consumed as soon as the native call is attempted,
    /// regardless of success; an unexpected pre-execution native rejection
    /// gets one best-effort explicit release before the sequence is cancelled.
    pub fn finish_planned_expert_tile(
        &mut self,
        mailbox: &TargetSequenceMailbox,
        plan: TargetSequenceExpertPlan,
        missing_expert_major_bytes: &[u8],
    ) -> Result<()> {
        let Some(waiting) = self.waiting else {
            return Err(DeltafinError::new(
                "target sequence is not waiting for planned expert bytes",
            ));
        };
        let plan_handle = plan.lease.handle();
        let owns_live_plan = self
            .expert_plan
            .as_ref()
            .is_some_and(|lease| Arc::ptr_eq(lease, &plan.lease) && lease.handle() == plan_handle);
        if self.handle == 0
            || plan_handle == 0
            || !owns_live_plan
            || !Arc::ptr_eq(&self.session, &plan.session)
            || plan.sequence != self.handle
            || self.state != TargetSequenceState::WaitingForExperts
            || waiting.layer_index != mailbox.layer_index
            || waiting.spine_generation != mailbox.spine_generation
            || mailbox.position_count() != self.position_count
            || plan.layer_index != waiting.layer_index
            || plan.spine_generation != waiting.spine_generation
            || plan.first_row != waiting.next_row
            || plan
                .first_row
                .checked_add(plan.row_count)
                .is_none_or(|end| end > self.position_count)
            || target_sequence_route_union(
                mailbox,
                plan.first_row,
                plan.row_count,
                TARGET_SEQUENCE_MAX_EXPERTS,
            )?
            .as_ref()
                != plan.canonical_experts.as_ref()
        {
            return Err(DeltafinError::new(
                "target expert plan is stale or belongs to another sequence/row union",
            ));
        }
        let expected_bytes = K3_EXPERT_SOURCE_BYTES
            .checked_mul(plan.missing_experts.len())
            .ok_or_else(|| DeltafinError::new("planned expert miss bytes overflowed usize"))?;
        if missing_expert_major_bytes.len() != expected_bytes {
            return Err(DeltafinError::new(format!(
                "planned expert miss slab has {} bytes, expected {expected_bytes}",
                missing_expert_major_bytes.len()
            )));
        }
        let mut missing_experts = [0_u16; TARGET_SEQUENCE_MAX_EXPERTS];
        missing_experts[..plan.missing_experts.len()].copy_from_slice(&plan.missing_experts);
        let request = TargetSequenceFinishPlannedExpertsRequestV1 {
            struct_size: size_of::<TargetSequenceFinishPlannedExpertsRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            sequence: self.handle,
            plan: plan_handle,
            spine_generation: plan.spine_generation,
            layer_index: plan.layer_index,
            first_row: plan.first_row as u32,
            row_count: plan.row_count as u32,
            missing_count: plan.missing_experts.len() as u32,
            flags: 0,
            missing_experts,
            expert_major_bytes: borrowed_bytes_pointer(missing_expert_major_bytes),
            expert_major_length: missing_expert_major_bytes.len() as u64,
            reserved: [0; 4],
        };
        let mut report = TargetSequenceFinishExpertsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the exact miss slab is borrowed only for this synchronous
        // call. An empty all-hit slab is represented by canonical null/zero.
        let status = unsafe {
            deltafin_provider_target_sequence_finish_planned_experts_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let consumed_handle = plan.lease.consume();
        self.expert_plan = None;
        if status != 0 {
            if consumed_handle != 0 {
                let _ = release_resource(
                    deltafin_provider_moe_plan_release_v1,
                    self.session.handle,
                    consumed_handle,
                    "failed native expert-plan cleanup",
                );
            }
            let failure = ffi_error(
                "native provider target-sequence finish planned experts",
                &error,
            );
            self.cancel_after_invalid_report();
            return Err(failure);
        }
        let next_row = plan.first_row + plan.row_count;
        let completed_layer = next_row == self.position_count;
        let expected_state = if completed_layer {
            if waiting.layer_index + 1 == 93 {
                TARGET_SEQUENCE_STATE_READY_FOR_TAIL
            } else {
                TARGET_SEQUENCE_STATE_ACTIVE
            }
        } else {
            TARGET_SEQUENCE_STATE_WAITING_FOR_EXPERTS
        };
        if report.struct_size as usize != size_of::<TargetSequenceFinishExpertsReportV1>()
            || report.abi_version != ABI_VERSION
            || report.sequence != self.handle
            || report.spine_generation != waiting.spine_generation
            || report.layer_index != waiting.layer_index
            || report.first_row as usize != plan.first_row
            || report.row_count as usize != plan.row_count
            || report.next_expert_row as usize != next_row
            || report.state != expected_state
            || report.flags != 0
            || report.reserved != [0; 2]
        {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native provider returned an invalid planned expert-finish report",
            ));
        }
        if completed_layer {
            self.waiting = None;
            self.next_layer = waiting.layer_index + 1;
            self.state = if self.next_layer == 93 {
                TargetSequenceState::ReadyForTail
            } else {
                TargetSequenceState::Active
            };
        } else {
            self.waiting = Some(TargetSequenceWaiting {
                next_row,
                ..waiting
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn finish_expert_tile(
        &mut self,
        mailbox: &TargetSequenceMailbox,
        first_row: usize,
        row_count: usize,
        canonical_expert_ids: &[u16],
        expert_major_bytes: &[u8],
        expert_layout: ExpertStorageLayout,
        backend: TargetExpertBackend,
        cpu_threads: usize,
        metal_shader_path: Option<&str>,
        retain_metal_wrappers: bool,
    ) -> Result<()> {
        self.finish_expert_tile_storage(
            mailbox,
            first_row,
            row_count,
            canonical_expert_ids,
            TargetExpertTileStorage::Contiguous(expert_major_bytes),
            expert_layout,
            backend,
            cpu_threads,
            metal_shader_path,
            retain_metal_wrappers,
        )
    }

    /// Start the independent per-expert work for experts whose bytes have
    /// landed, without waiting for the rest of this layer's reads.
    ///
    /// `arrived` pairs unique ascending expert IDs with one complete
    /// authenticated span each; every span must stay live until the layer's
    /// `finish_expert_span_tile`. Returns whether the provider took the offer.
    ///
    /// This buys scheduling only. The provider still owns the route, the fp32
    /// weights, and the single reduction over every edge in the router's
    /// order, all of which happen in the finish call — so a declined offer, a
    /// partial offer, and no offer at all produce identical output bytes.
    /// Errors are deliberately not raised: a staging failure is reported as
    /// `false` so it can never fail a generation.
    pub fn stage_expert_spans(
        &mut self,
        mailbox: &TargetSequenceMailbox,
        first_row: usize,
        row_count: usize,
        arrived: &[(u16, &[u8])],
        expert_layout: ExpertStorageLayout,
        backend: TargetExpertBackend,
        cpu_threads: usize,
        metal_shader_path: Option<&str>,
    ) -> bool {
        let Some(waiting) = self.waiting else {
            return false;
        };
        let (expert_layout_abi, expert_span_bytes) = expert_layout_abi(expert_layout);
        let Ok(expert_span_len) = usize::try_from(expert_span_bytes) else {
            return false;
        };
        if self.handle == 0
            || self.has_live_expert_plan()
            || self.state != TargetSequenceState::WaitingForExperts
            || waiting.layer_index != mailbox.layer_index
            || waiting.spine_generation != mailbox.spine_generation
            || mailbox.position_count() != self.position_count
            || first_row != waiting.next_row
            || row_count != 1
            || self.position_count != 1
            || arrived.is_empty()
            || arrived.len() > TARGET_SEQUENCE_MAX_EXPERTS
        {
            return false;
        }
        let mut expert_ids = [0_u16; TARGET_SEQUENCE_MAX_EXPERTS];
        let mut expert_span_pointers = [ptr::null(); TARGET_SEQUENCE_MAX_EXPERTS];
        for (index, (expert, span)) in arrived.iter().enumerate() {
            if *expert >= 896
                || span.len() != expert_span_len
                || (index != 0 && arrived[index - 1].0 >= *expert)
            {
                return false;
            }
            expert_ids[index] = *expert;
            expert_span_pointers[index] = span.as_ptr();
        }
        let Ok(cpu_threads) = u32::try_from(cpu_threads) else {
            return false;
        };
        if cpu_threads == 0 || cpu_threads > 1024 {
            return false;
        }
        let shader_bytes = metal_shader_path
            .filter(|path| !path.is_empty())
            .map(str::as_bytes)
            .unwrap_or_default();
        if shader_bytes.contains(&0) || shader_bytes.len() > 4096 {
            return false;
        }
        let shader_pointer = if shader_bytes.is_empty() {
            ptr::null()
        } else {
            shader_bytes.as_ptr().cast()
        };
        let request = TargetSequenceStageExpertSpansRequestV1 {
            struct_size: size_of::<TargetSequenceStageExpertSpansRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            sequence: self.handle,
            spine_generation: waiting.spine_generation,
            layer_index: waiting.layer_index,
            first_row: first_row as u32,
            row_count: row_count as u32,
            expert_backend: backend.abi_value(),
            cpu_threads,
            expert_count: arrived.len() as u32,
            flags: 0,
            expert_layout: expert_layout_abi,
            expert_ids,
            expert_span_pointers,
            metal_shader_path: shader_pointer,
            metal_shader_path_length: shader_bytes.len() as u64,
            expert_span_bytes,
            reserved: [0; 4],
        };
        let mut report = TargetSequenceStageExpertSpansReportV1 {
            struct_size: size_of::<TargetSequenceStageExpertSpansReportV1>() as u32,
            ..Default::default()
        };
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: every arrived span and the shader selector remain live for
        // this synchronous call, and the caller holds each span's lease until
        // the layer's finish call returns.
        let status = unsafe {
            deltafin_provider_target_sequence_stage_expert_spans_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status == 0 && report.staged == 1
    }

    /// Finish one authoritative tile from individually owned expert spans.
    /// This is the no-copy partial-prefetch reuse boundary: slot i must be one
    /// complete authenticated span for canonical_expert_ids[i]. The provider
    /// borrows every slice synchronously. `retain_metal_wrappers` is legal
    /// only for an arena whose owner flushes the provider cache before its
    /// storage can retire; false preserves the ordinary call-scoped lifetime.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_expert_span_tile(
        &mut self,
        mailbox: &TargetSequenceMailbox,
        first_row: usize,
        row_count: usize,
        canonical_expert_ids: &[u16],
        expert_spans: &[&[u8]],
        expert_layout: ExpertStorageLayout,
        backend: TargetExpertBackend,
        cpu_threads: usize,
        metal_shader_path: Option<&str>,
        retain_metal_wrappers: bool,
    ) -> Result<()> {
        self.finish_expert_tile_storage(
            mailbox,
            first_row,
            row_count,
            canonical_expert_ids,
            TargetExpertTileStorage::Scattered(expert_spans),
            expert_layout,
            backend,
            cpu_threads,
            metal_shader_path,
            retain_metal_wrappers,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_expert_tile_storage(
        &mut self,
        mailbox: &TargetSequenceMailbox,
        first_row: usize,
        row_count: usize,
        canonical_expert_ids: &[u16],
        storage: TargetExpertTileStorage<'_>,
        expert_layout: ExpertStorageLayout,
        backend: TargetExpertBackend,
        cpu_threads: usize,
        metal_shader_path: Option<&str>,
        retain_metal_wrappers: bool,
    ) -> Result<()> {
        if self.has_live_expert_plan() {
            return Err(DeltafinError::new(
                "target sequence owns a live expert plan; use its planned finish or release it first",
            ));
        }
        let Some(waiting) = self.waiting else {
            return Err(DeltafinError::new(
                "target sequence is not waiting for expert bytes",
            ));
        };
        if self.handle == 0
            || self.state != TargetSequenceState::WaitingForExperts
            || waiting.layer_index != mailbox.layer_index
            || waiting.spine_generation != mailbox.spine_generation
            || mailbox.position_count() != self.position_count
            || first_row != waiting.next_row
            || !(1..=TARGET_SEQUENCE_MAX_TILE_ROWS).contains(&row_count)
            || first_row
                .checked_add(row_count)
                .is_none_or(|end| end > self.position_count)
        {
            return Err(DeltafinError::new(
                "target-sequence expert tile is stale, out of order, or outside its row bound",
            ));
        }
        if canonical_expert_ids.is_empty()
            || canonical_expert_ids.len() > TARGET_SEQUENCE_MAX_EXPERTS_V2
        {
            return Err(DeltafinError::new(format!(
                "target-sequence expert tile needs 1..={TARGET_SEQUENCE_MAX_EXPERTS_V2} canonical experts"
            )));
        }
        for (index, &expert) in canonical_expert_ids.iter().enumerate() {
            if expert >= 896 || (index != 0 && canonical_expert_ids[index - 1] >= expert) {
                return Err(DeltafinError::new(
                    "target-sequence expert IDs must be unique canonical ascending IDs",
                ));
            }
        }
        let wide_request = canonical_expert_ids.len() > TARGET_SEQUENCE_MAX_EXPERTS;
        let route_union = target_sequence_route_union(
            mailbox,
            first_row,
            row_count,
            if wide_request {
                TARGET_SEQUENCE_MAX_EXPERTS_V2
            } else {
                TARGET_SEQUENCE_MAX_EXPERTS
            },
        )?;
        if route_union.as_ref() != canonical_expert_ids {
            return Err(DeltafinError::new(
                "target-sequence expert IDs are not the exact canonical active-row union",
            ));
        }
        let backend = resolve_target_backend(self.session.device, backend)?;
        if retain_metal_wrappers && backend != TargetExpertBackend::Metal {
            return Err(DeltafinError::new(
                "retained Metal expert wrappers require the selected Metal backend",
            ));
        }
        if wide_request && backend == TargetExpertBackend::Cuda {
            return Err(DeltafinError::new(
                "wide target-sequence expert unions are not a CUDA cache-plan input",
            ));
        }
        if expert_layout == ExpertStorageLayout::Scale4V2 && backend != TargetExpertBackend::Metal {
            return Err(DeltafinError::new(
                "scale4-v2 target-sequence experts require the selected Metal backend",
            ));
        }
        let (expert_layout_abi, expert_span_bytes) = expert_layout_abi(expert_layout);
        let expert_span_len = usize::try_from(expert_span_bytes)
            .map_err(|_| DeltafinError::new("target-sequence expert span exceeds usize"))?;
        match &storage {
            TargetExpertTileStorage::Contiguous(expert_major_bytes) => {
                let expected_bytes = expert_span_len
                    .checked_mul(canonical_expert_ids.len())
                    .ok_or_else(|| {
                        DeltafinError::new("target-sequence expert byte count overflowed")
                    })?;
                if expert_major_bytes.len() != expected_bytes {
                    return Err(DeltafinError::new(format!(
                        "target-sequence expert tile has {} bytes, expected {expected_bytes}",
                        expert_major_bytes.len()
                    )));
                }
            }
            TargetExpertTileStorage::Scattered(expert_spans) => {
                if backend == TargetExpertBackend::Cuda
                    || (backend == TargetExpertBackend::Auto
                        && matches!(self.session.device, Device::Cuda(_)))
                {
                    return Err(DeltafinError::new(
                        "scattered expert spans are not a CUDA cache-plan input",
                    ));
                }
                if expert_spans.len() != canonical_expert_ids.len() {
                    return Err(DeltafinError::new(
                        "scattered expert span count does not match canonical IDs",
                    ));
                }
                for (index, span) in expert_spans.iter().enumerate() {
                    if span.len() != expert_span_len {
                        return Err(DeltafinError::new(format!(
                            "scattered expert span {index} has {} bytes, expected {expert_span_len}",
                            span.len()
                        )));
                    }
                    if expert_spans[..index]
                        .iter()
                        .any(|prior| prior.as_ptr() == span.as_ptr())
                    {
                        return Err(DeltafinError::new(
                            "scattered expert spans must have distinct storage addresses",
                        ));
                    }
                }
            }
        }
        let cpu_threads = u32::try_from(cpu_threads)
            .map_err(|_| DeltafinError::new("target expert thread count exceeds ABI u32"))?;
        if cpu_threads == 0 || cpu_threads > 1024 {
            return Err(DeltafinError::new(
                "target expert thread count must be in 1..1024",
            ));
        }
        let shader_bytes = metal_shader_path
            .filter(|path| !path.is_empty())
            .map(str::as_bytes)
            .unwrap_or_default();
        if shader_bytes.contains(&0) || shader_bytes.len() > 4096 {
            return Err(DeltafinError::new(
                "target Metal shader path is invalid or exceeds 4096 bytes",
            ));
        }
        let mut report = TargetSequenceFinishExpertsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let shader_pointer = if shader_bytes.is_empty() {
            ptr::null()
        } else {
            shader_bytes.as_ptr().cast()
        };
        let expert_flags = if retain_metal_wrappers {
            TARGET_EXPERT_RETAIN_METAL_WRAPPERS
        } else {
            0
        };
        let status = if wide_request {
            let mut expert_span_pointers = [ptr::null(); TARGET_SEQUENCE_MAX_EXPERTS_V2];
            let (expert_major_bytes, expert_major_length, span_pointers, span_pointer_count) =
                match storage {
                    TargetExpertTileStorage::Contiguous(expert_major_bytes) => (
                        expert_major_bytes.as_ptr(),
                        expert_major_bytes.len() as u64,
                        ptr::null(),
                        0,
                    ),
                    TargetExpertTileStorage::Scattered(expert_spans) => {
                        for (destination, span) in
                            expert_span_pointers.iter_mut().zip(expert_spans.iter())
                        {
                            *destination = span.as_ptr();
                        }
                        (
                            ptr::null(),
                            0,
                            expert_span_pointers.as_ptr(),
                            expert_spans.len() as u64,
                        )
                    }
                };
            let request = TargetSequenceFinishExpertsRequestV2 {
                struct_size: size_of::<TargetSequenceFinishExpertsRequestV2>() as u32,
                abi_version: ABI_VERSION,
                session: self.session.handle,
                sequence: self.handle,
                spine_generation: waiting.spine_generation,
                layer_index: waiting.layer_index,
                first_row: first_row as u32,
                row_count: row_count as u32,
                expert_backend: backend.abi_value(),
                cpu_threads,
                expert_count: canonical_expert_ids.len() as u32,
                flags: expert_flags,
                expert_layout: expert_layout_abi,
                expert_ids: canonical_expert_ids.as_ptr(),
                expert_ids_length: canonical_expert_ids.len() as u64,
                expert_major_bytes,
                expert_major_length,
                expert_span_pointers: span_pointers,
                expert_span_pointer_count: span_pointer_count,
                metal_shader_path: shader_pointer,
                metal_shader_path_length: shader_bytes.len() as u64,
                expert_span_bytes,
                reserved: [0; 8],
            };
            // SAFETY: the canonical IDs, selected storage form, and shader
            // remain live for this synchronous V2 call. The explicit flag is
            // set only by an engine with flush-before-retire arena hooks.
            unsafe {
                deltafin_provider_target_sequence_finish_experts_v2(
                    &request,
                    &mut report,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        } else {
            let mut expert_ids = [0_u16; TARGET_SEQUENCE_MAX_EXPERTS];
            expert_ids[..canonical_expert_ids.len()].copy_from_slice(canonical_expert_ids);
            match storage {
                TargetExpertTileStorage::Contiguous(expert_major_bytes) => {
                    let request = TargetSequenceFinishExpertsRequestV1 {
                        struct_size: size_of::<TargetSequenceFinishExpertsRequestV1>() as u32,
                        abi_version: ABI_VERSION,
                        session: self.session.handle,
                        sequence: self.handle,
                        spine_generation: waiting.spine_generation,
                        layer_index: waiting.layer_index,
                        first_row: first_row as u32,
                        row_count: row_count as u32,
                        expert_backend: backend.abi_value(),
                        cpu_threads,
                        expert_count: canonical_expert_ids.len() as u32,
                        flags: expert_flags,
                        expert_layout: expert_layout_abi,
                        expert_ids,
                        expert_major_bytes: expert_major_bytes.as_ptr(),
                        expert_major_length: expert_major_bytes.len() as u64,
                        metal_shader_path: shader_pointer,
                        metal_shader_path_length: shader_bytes.len() as u64,
                        expert_span_bytes,
                        reserved: [0; 3],
                    };
                    // SAFETY: expert/shader slices remain live for this synchronous
                    // call. Retention is opt-in only for a hooked arena.
                    unsafe {
                        deltafin_provider_target_sequence_finish_experts_v1(
                            &request,
                            &mut report,
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    }
                }
                TargetExpertTileStorage::Scattered(expert_spans) => {
                    let mut expert_span_pointers = [ptr::null(); TARGET_SEQUENCE_MAX_EXPERTS];
                    for (destination, span) in
                        expert_span_pointers.iter_mut().zip(expert_spans.iter())
                    {
                        *destination = span.as_ptr();
                    }
                    let request = TargetSequenceFinishExpertSpansRequestV1 {
                        struct_size: size_of::<TargetSequenceFinishExpertSpansRequestV1>() as u32,
                        abi_version: ABI_VERSION,
                        session: self.session.handle,
                        sequence: self.handle,
                        spine_generation: waiting.spine_generation,
                        layer_index: waiting.layer_index,
                        first_row: first_row as u32,
                        row_count: row_count as u32,
                        expert_backend: backend.abi_value(),
                        cpu_threads,
                        expert_count: canonical_expert_ids.len() as u32,
                        flags: expert_flags,
                        expert_layout: expert_layout_abi,
                        expert_ids,
                        expert_span_pointers,
                        metal_shader_path: shader_pointer,
                        metal_shader_path_length: shader_bytes.len() as u64,
                        expert_span_bytes,
                        reserved: [0; 4],
                    };
                    // SAFETY: every expert/shader slice remains live for this
                    // synchronous call. Retention is opt-in only for hooked arenas.
                    unsafe {
                        deltafin_provider_target_sequence_finish_expert_spans_v1(
                            &request,
                            &mut report,
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    }
                }
            }
        };
        if status != 0 {
            let failure = ffi_error("native provider target-sequence finish experts", &error);
            self.cancel_after_invalid_report();
            return Err(failure);
        }
        let next_row = first_row + row_count;
        let completed_layer = next_row == self.position_count;
        let expected_state = if completed_layer {
            if waiting.layer_index + 1 == 93 {
                TARGET_SEQUENCE_STATE_READY_FOR_TAIL
            } else {
                TARGET_SEQUENCE_STATE_ACTIVE
            }
        } else {
            TARGET_SEQUENCE_STATE_WAITING_FOR_EXPERTS
        };
        if report.struct_size as usize != size_of::<TargetSequenceFinishExpertsReportV1>()
            || report.abi_version != ABI_VERSION
            || report.sequence != self.handle
            || report.spine_generation != waiting.spine_generation
            || report.layer_index != waiting.layer_index
            || report.first_row as usize != first_row
            || report.row_count as usize != row_count
            || report.next_expert_row as usize != next_row
            || report.state != expected_state
            || report.flags != 0
            || report.reserved != [0; 2]
        {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native provider returned an invalid target-sequence expert report",
            ));
        }
        if completed_layer {
            self.waiting = None;
            self.next_layer = waiting.layer_index + 1;
            self.state = if self.next_layer == 93 {
                TargetSequenceState::ReadyForTail
            } else {
                TargetSequenceState::Active
            };
        } else {
            self.waiting = Some(TargetSequenceWaiting {
                next_row,
                ..waiting
            });
        }
        Ok(())
    }

    pub fn finish_tail(&mut self) -> Result<Box<[u32]>> {
        if self.handle == 0
            || self.state != TargetSequenceState::ReadyForTail
            || self.waiting.is_some()
            || self.next_layer != 93
        {
            return Err(DeltafinError::new(
                "target sequence is not ready for its target tail",
            ));
        }
        let request = ResourceRequestV1::new(self.session.handle, self.handle);
        let mut report = TargetSequenceTailReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request names provider-owned state and report/error are
        // writable for the complete call.
        let status = unsafe {
            deltafin_provider_target_sequence_finish_tail_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            let failure = ffi_error("native provider target-sequence finish tail", &error);
            self.cancel_after_invalid_report();
            return Err(failure);
        }
        let expected_tokens = match self.mode {
            TargetSequenceMode::Prefill => 1,
            TargetSequenceMode::Verify => self.position_count,
        };
        if report.struct_size as usize != size_of::<TargetSequenceTailReportV1>()
            || report.abi_version != ABI_VERSION
            || report.sequence != self.handle
            || report.token_count as usize != expected_tokens
            || report.state != TARGET_SEQUENCE_STATE_READY_TO_COMMIT
            || report.tail_rows as usize != expected_tokens
            || report.tail_provider_dispatches != 1
            || report.token_ids[expected_tokens..] != [0; ROUTE_MAX_POSITIONS][expected_tokens..]
            || report.reserved != [0; 4]
        {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native provider returned an invalid target-sequence tail report",
            ));
        }
        self.state = TargetSequenceState::ReadyToCommit;
        Ok(report.token_ids[..expected_tokens]
            .to_vec()
            .into_boxed_slice())
    }

    /// Materialize the captured K3 rows for the proposal model (DSpark's
    /// five post-layer hidden states or EAGLE-3's three AttnRes taps) as one
    /// provider-owned BF16 tensor. No activation bytes cross into Rust; the
    /// returned opaque tensor is intended only for that model's cache
    /// adapter and remains independently releasable if target commit fails.
    pub fn dspark_target_rows(&mut self) -> Result<ProviderTensor> {
        let Some(capture) = self.capture.filter(|_| {
            self.handle != 0 && self.state == TargetSequenceState::ReadyToCommit
        }) else {
            return Err(DeltafinError::new(
                "target sequence has no completed proposal-model capture",
            ));
        };
        let request = ResourceRequestV1::new(self.session.handle, self.handle);
        let mut report = TensorReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request contains only provider-owned handles. The provider
        // returns a new opaque tensor resource and retains no Rust pointer.
        let status = unsafe {
            deltafin_provider_target_sequence_dspark_rows_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error(
                "native provider target-sequence DSpark capture",
                &error,
            ));
        }
        if report.struct_size as usize != size_of::<TensorReportV1>()
            || report.abi_version != ABI_VERSION
            || report.tensor == 0
            || report.rows as usize != self.position_count
            || report.columns as usize != capture.columns()
            || report.reserved != [0; 4]
        {
            if report.tensor != 0 {
                let _ = release_resource(
                    deltafin_provider_tensor_release_v1,
                    self.session.handle,
                    report.tensor,
                    "invalid target-sequence DSpark tensor release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid proposal-model target-row tensor",
            ));
        }
        Ok(ProviderTensor {
            session: Arc::clone(&self.session),
            handle: report.tensor,
            rows: self.position_count,
            columns: capture.columns(),
        })
    }

    pub fn stats(&mut self) -> Result<TargetSequenceStats> {
        if self.handle == 0 {
            return Err(DeltafinError::new("target sequence is closed"));
        }
        let request = ResourceRequestV1::new(self.session.handle, self.handle);
        let mut report = TargetSequenceStatsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request contains only provider-owned handles; report/error
        // remain writable for the complete call.
        let status = unsafe {
            deltafin_provider_target_sequence_stats_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            let failure = ffi_error("native provider target-sequence stats", &error);
            self.cancel_after_invalid_report();
            return Err(failure);
        }
        let mode = match report.mode {
            TARGET_SEQUENCE_PREFILL => TargetSequenceMode::Prefill,
            TARGET_SEQUENCE_VERIFY => TargetSequenceMode::Verify,
            _ => {
                self.cancel_after_invalid_report();
                return Err(DeltafinError::new(
                    "native provider returned an invalid target-sequence stats mode",
                ));
            }
        };
        let state = match decode_target_sequence_state(report.state) {
            Ok(state) => state,
            Err(error) => {
                self.cancel_after_invalid_report();
                return Err(error);
            }
        };
        if report.struct_size as usize != size_of::<TargetSequenceStatsReportV1>()
            || report.abi_version != ABI_VERSION
            || report.sequence != self.handle
            || report.positions as usize != self.position_count
            || mode != self.mode
            || state != self.state
            || report.maximum_live_streamed_layers > 1
            || report.maximum_experts_per_request > TARGET_SEQUENCE_MAX_EXPERTS_V2 as u64
            || report.maximum_positions_per_expert_tile > TARGET_SEQUENCE_MAX_TILE_ROWS as u64
            || report.reserved != [0; 2]
        {
            self.cancel_after_invalid_report();
            return Err(DeltafinError::new(
                "native provider returned an invalid target-sequence stats report",
            ));
        }
        Ok(TargetSequenceStats {
            positions: report.positions,
            streamed_layer_passes: report.streamed_layer_passes,
            attention_rows: report.attention_rows,
            expert_row_requests: report.expert_row_requests,
            expert_rows_completed: report.expert_rows_completed,
            expert_tiles_completed: report.expert_tiles_completed,
            tail_rows: report.tail_rows,
            tail_provider_dispatches: report.tail_provider_dispatches,
            maximum_live_streamed_layers: report.maximum_live_streamed_layers,
            maximum_experts_per_request: report.maximum_experts_per_request,
            maximum_positions_per_expert_tile: report.maximum_positions_per_expert_tile,
            staged_kda_storage_bytes: report.staged_kda_storage_bytes,
            verify_snapshot_bytes: report.verify_snapshot_bytes,
            projected_mla_storage_bytes: report.projected_mla_storage_bytes,
            additional_mla_storage_bytes: report.additional_mla_storage_bytes,
            mode,
            state,
        })
    }

    pub fn commit_all(self) -> Result<TargetSequenceCommit> {
        let positions = self.position_count;
        self.commit_prefix(positions)
    }

    pub fn commit_prefix(mut self, positions: usize) -> Result<TargetSequenceCommit> {
        if self.handle == 0 || self.state != TargetSequenceState::ReadyToCommit {
            return Err(DeltafinError::new("target sequence is not ready to commit"));
        }
        validate_target_sequence_commit_prefix(
            self.mode,
            self.position_count,
            self.full_commit_only,
            positions,
        )?;
        let positions_u32 = u32::try_from(positions)
            .map_err(|_| DeltafinError::new("target-sequence commit exceeds ABI u32"))?;
        let request = TargetSequenceCommitRequestV1 {
            struct_size: size_of::<TargetSequenceCommitRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            sequence: self.handle,
            positions: positions_u32,
            flags: 0,
            reserved: [0; 4],
        };
        let mut report = TargetSequenceCommitReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request contains only provider-owned handles and report/error
        // are writable. Successful commit consumes the native handle.
        let status = unsafe {
            deltafin_provider_target_sequence_commit_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider target-sequence commit", &error));
        }
        let consumed_handle = self.handle;
        self.handle = 0;
        self.state = TargetSequenceState::Committed;
        if report.struct_size as usize != size_of::<TargetSequenceCommitReportV1>()
            || report.abi_version != ABI_VERSION
            || report.sequence != consumed_handle
            || report.committed_positions != positions as u64
            || report.session_committed_positions < positions as u64
            || report.state != TARGET_SEQUENCE_STATE_COMMITTED
            || report.flags != 0
            || report.reserved != [0; 3]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid target-sequence commit report",
            ));
        }
        Ok(TargetSequenceCommit {
            committed_positions: report.committed_positions,
            session_committed_positions: report.session_committed_positions,
        })
    }

    pub fn cancel(mut self) -> Result<()> {
        if self.handle == 0 {
            return Ok(());
        }
        let result = release_resource(
            deltafin_provider_target_sequence_cancel_v1,
            self.session.handle,
            self.handle,
            "native provider target-sequence cancel",
        );
        if result.is_ok() {
            self.handle = 0;
            self.state = TargetSequenceState::Cancelled;
            self.waiting = None;
            self.expert_plan = None;
        }
        result
    }

    fn cancel_after_invalid_report(&mut self) {
        if self.handle == 0 {
            return;
        }
        let _ = release_resource(
            deltafin_provider_target_sequence_cancel_v1,
            self.session.handle,
            self.handle,
            "invalid native target-sequence cancel",
        );
        self.handle = 0;
        self.state = TargetSequenceState::Cancelled;
        self.waiting = None;
        self.expert_plan = None;
    }
}

impl Drop for TargetSequence {
    fn drop(&mut self) {
        if self.handle != 0 {
            let _ = release_resource(
                deltafin_provider_target_sequence_cancel_v1,
                self.session.handle,
                self.handle,
                "native provider target-sequence cancel",
            );
            self.handle = 0;
            self.state = TargetSequenceState::Cancelled;
            self.waiting = None;
            self.expert_plan = None;
        }
    }
}

#[derive(Debug)]
pub struct ProviderCache {
    session: Arc<SessionInner>,
    handle: u64,
    rows: usize,
    columns: usize,
    version: u64,
}

impl ProviderCache {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn read_f32(&self) -> Result<Vec<f32>> {
        let mut destination = vec![0.0_f32; checked_shape(self.rows, self.columns)?];
        let request = CacheReadF32V1 {
            struct_size: size_of::<CacheReadF32V1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            cache: self.handle,
            destination: destination.as_mut_ptr(),
            element_capacity: destination.len() as u64,
            flags: 0,
            reserved0: 0,
            reserved: [0; 3],
        };
        let mut report = CacheReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: destination/report/error are valid writable buffers. No
        // pointer or ownership crosses the completed call.
        let status = unsafe {
            deltafin_provider_cache_read_f32_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider cache read", &error));
        }
        if report.abi_version != ABI_VERSION
            || report.struct_size as usize != size_of::<CacheReportV1>()
            || report.cache != self.handle
            || report.rows != self.rows as u64
            || report.columns != self.columns as u64
            || report.version != self.version
            || report.reserved != [0; 3]
        {
            return Err(DeltafinError::new(
                "native provider returned an invalid cache read report",
            ));
        }
        Ok(destination)
    }
}

impl Drop for ProviderCache {
    fn drop(&mut self) {
        if self.handle != 0 {
            let _ = release_resource(
                deltafin_provider_cache_release_v1,
                self.session.handle,
                self.handle,
                "native provider cache release",
            );
            self.handle = 0;
        }
    }
}

#[derive(Debug)]
pub struct ProviderKdaCache {
    session: Arc<SessionInner>,
    handle: u64,
    layer_index: u32,
    version: u64,
    convolution_elements: u64,
    recurrent_elements: u64,
    poisoned: bool,
}

impl ProviderKdaCache {
    pub fn layer_index(&self) -> u32 {
        self.layer_index
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn convolution_elements(&self) -> u64 {
        self.convolution_elements
    }

    pub fn recurrent_elements(&self) -> u64 {
        self.recurrent_elements
    }
}

impl Drop for ProviderKdaCache {
    fn drop(&mut self) {
        if self.handle != 0 {
            let _ = release_resource(
                deltafin_provider_kda_cache_release_v1,
                self.session.handle,
                self.handle,
                "native provider KDA cache release",
            );
            self.handle = 0;
        }
    }
}

/// Owns one speculative KDA output and its staged cache state. A drop before
/// `commit` releases both resources and leaves the live cache unchanged.
#[derive(Debug)]
pub struct PreparedKdaDecode<'cache> {
    session: Arc<SessionInner>,
    ticket: u64,
    cache: &'cache mut ProviderKdaCache,
    output: Option<ProviderTensor>,
}

impl PreparedKdaDecode<'_> {
    pub fn output(&self) -> &ProviderTensor {
        self.output
            .as_ref()
            .expect("live prepared KDA decode must own its output")
    }

    pub fn cancel(mut self) -> Result<()> {
        let result = release_resource(
            deltafin_provider_kda_ticket_release_v1,
            self.session.handle,
            self.ticket,
            "native provider KDA ticket release",
        );
        if result.is_ok() {
            self.ticket = 0;
        }
        result
    }

    pub fn commit(mut self) -> Result<ProviderTensor> {
        let expected_version = self.cache.version.checked_add(1).ok_or_else(|| {
            DeltafinError::new("native provider KDA cache version overflowed in Rust")
        })?;
        let request = ResourceRequestV1::new(self.session.handle, self.ticket);
        let mut report = KdaCommitReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request identifies only provider-owned resources;
        // report/error are valid writable buffers for the complete call.
        let status = unsafe {
            deltafin_provider_kda_commit_v1(&request, &mut report, error.as_mut_ptr(), error.len())
        };
        if status != 0 {
            return Err(ffi_error("native provider KDA commit", &error));
        }
        // A successful native call consumes the ticket even if a corrupt
        // report prevents Rust from trusting the new version.
        self.ticket = 0;
        if report.struct_size as usize != size_of::<KdaCommitReportV1>()
            || report.abi_version != ABI_VERSION
            || report.cache != self.cache.handle
            || report.committed_version != expected_version
            || report.layer_index != self.cache.layer_index
            || report.flags != 0
            || report.reserved != [0; 4]
        {
            self.cache.poisoned = true;
            return Err(DeltafinError::new(
                "native provider returned an invalid KDA commit report",
            ));
        }
        self.cache.version = report.committed_version;
        Ok(self
            .output
            .take()
            .expect("live prepared KDA decode must own its output"))
    }
}

impl Drop for PreparedKdaDecode<'_> {
    fn drop(&mut self) {
        if self.ticket != 0 {
            let _ = release_resource(
                deltafin_provider_kda_ticket_release_v1,
                self.session.handle,
                self.ticket,
                "native provider KDA ticket release",
            );
            self.ticket = 0;
        }
    }
}

#[derive(Debug)]
pub struct ProviderMlaCache {
    session: Arc<SessionInner>,
    handle: u64,
    layer_index: u32,
    version: u64,
    length: u64,
    capacity: u64,
    poisoned: bool,
}

impl ProviderMlaCache {
    pub fn layer_index(&self) -> u32 {
        self.layer_index
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn length(&self) -> u64 {
        self.length
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

impl Drop for ProviderMlaCache {
    fn drop(&mut self) {
        if self.handle != 0 {
            let _ = release_resource(
                deltafin_provider_mla_cache_release_v1,
                self.session.handle,
                self.handle,
                "native provider MLA cache release",
            );
            self.handle = 0;
        }
    }
}

/// Owns one exact MLA output and the corresponding unpublished KV position.
/// Dropping it cancels the native ticket, including any uncommitted growth.
#[derive(Debug)]
pub struct PreparedMlaDecode<'cache> {
    session: Arc<SessionInner>,
    ticket: u64,
    cache: &'cache mut ProviderMlaCache,
    output: Option<ProviderTensor>,
    proposed_length: u64,
    proposed_capacity: u64,
    input_bundle_rows: u64,
}

impl PreparedMlaDecode<'_> {
    pub fn output(&self) -> &ProviderTensor {
        self.output
            .as_ref()
            .expect("live prepared MLA decode must own its output")
    }

    pub fn proposed_length(&self) -> u64 {
        self.proposed_length
    }

    pub fn proposed_capacity(&self) -> u64 {
        self.proposed_capacity
    }

    pub fn input_bundle_rows(&self) -> u64 {
        self.input_bundle_rows
    }

    pub fn cancel(mut self) -> Result<()> {
        let result = release_resource(
            deltafin_provider_mla_ticket_release_v1,
            self.session.handle,
            self.ticket,
            "native provider MLA ticket release",
        );
        if result.is_ok() {
            self.ticket = 0;
        }
        result
    }

    pub fn commit(mut self) -> Result<ProviderTensor> {
        let expected_version = self.cache.version.checked_add(1).ok_or_else(|| {
            DeltafinError::new("native provider MLA cache version overflowed in Rust")
        })?;
        let request = ResourceRequestV1::new(self.session.handle, self.ticket);
        let mut report = MlaCommitReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: the request names provider-owned cache/ticket resources and
        // report/error are valid writable buffers for the complete call.
        let status = unsafe {
            deltafin_provider_mla_commit_v1(&request, &mut report, error.as_mut_ptr(), error.len())
        };
        if status != 0 {
            return Err(ffi_error("native provider MLA commit", &error));
        }
        // Native commit consumes the ticket even if Rust rejects its report.
        self.ticket = 0;
        if report.struct_size as usize != size_of::<MlaCommitReportV1>()
            || report.abi_version != ABI_VERSION
            || report.cache != self.cache.handle
            || report.committed_version != expected_version
            || report.layer_index != self.cache.layer_index
            || report.flags != 0
            || report.committed_length != self.proposed_length
            || report.capacity != self.proposed_capacity
            || report.reserved != [0; 2]
        {
            self.cache.poisoned = true;
            return Err(DeltafinError::new(
                "native provider returned an invalid MLA commit report",
            ));
        }
        self.cache.version = report.committed_version;
        self.cache.length = report.committed_length;
        self.cache.capacity = report.capacity;
        Ok(self
            .output
            .take()
            .expect("live prepared MLA decode must own its output"))
    }
}

impl Drop for PreparedMlaDecode<'_> {
    fn drop(&mut self) {
        if self.ticket != 0 {
            let _ = release_resource(
                deltafin_provider_mla_ticket_release_v1,
                self.session.handle,
                self.ticket,
                "native provider MLA ticket release",
            );
            self.ticket = 0;
        }
    }
}

#[derive(Debug)]
pub struct NativeRouteMailbox {
    raw: Box<RouteMailboxV1>,
    positions: usize,
    edges: usize,
}

impl NativeRouteMailbox {
    pub fn positions(&self) -> usize {
        self.positions
    }

    pub fn ordered_experts(&self) -> &[u16] {
        &self.raw.ordered_experts[..self.edges]
    }

    pub fn ordered_weight_bits(&self) -> &[u32] {
        &self.raw.ordered_weight_bits[..self.edges]
    }

    pub fn cache_version(&self) -> u64 {
        self.raw.cache_version
    }
}

#[derive(Debug)]
pub struct PreparedLayer<'cache> {
    session: Arc<SessionInner>,
    ticket: u64,
    cache: &'cache mut ProviderCache,
    route: NativeRouteMailbox,
}

impl PreparedLayer<'_> {
    pub fn route(&self) -> &NativeRouteMailbox {
        &self.route
    }

    pub fn finish(mut self, expert_output: &ProviderTensor) -> Result<ProviderTensor> {
        require_same_session(&self.session, &expert_output.session, "expert output")?;
        if expert_output.shape() != self.cache.shape() {
            return Err(DeltafinError::new(
                "split layer expert output and cache shapes differ",
            ));
        }
        let request = FinishLayerRequestV1 {
            struct_size: size_of::<FinishLayerRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: self.session.handle,
            ticket: self.ticket,
            expert_output: expert_output.handle,
            flags: 0,
            reserved0: 0,
            reserved: [0; 5],
        };
        let mut report = FinishLayerReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: all resources are provider-owned and identified by opaque
        // IDs. The report/error buffers are valid for the call.
        let status = unsafe {
            deltafin_provider_finish_layer_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("native provider finish layer", &error));
        }
        // A successful native finish consumes the ticket and commits cache.
        self.ticket = 0;
        let expected_version = self.cache.version.checked_add(1).ok_or_else(|| {
            DeltafinError::new("native provider cache version overflowed in Rust")
        })?;
        if report.abi_version != ABI_VERSION
            || report.struct_size as usize != size_of::<FinishLayerReportV1>()
            || report.output == 0
            || report.positions != self.cache.rows as u64
            || report.hidden_columns != self.cache.columns as u64
            || report.committed_cache_version != expected_version
            || report.reserved != [0; 5]
        {
            if report.output != 0 {
                let _ = release_resource(
                    deltafin_provider_tensor_release_v1,
                    self.session.handle,
                    report.output,
                    "invalid native provider output release",
                );
            }
            return Err(DeltafinError::new(
                "native provider returned an invalid finish-layer report",
            ));
        }
        self.cache.version = report.committed_cache_version;
        Ok(ProviderTensor {
            session: Arc::clone(&self.session),
            handle: report.output,
            rows: report.positions as usize,
            columns: report.hidden_columns as usize,
        })
    }
}

impl Drop for PreparedLayer<'_> {
    fn drop(&mut self) {
        if self.ticket != 0 {
            let _ = release_resource(
                deltafin_provider_ticket_release_v1,
                self.session.handle,
                self.ticket,
                "native provider ticket release",
            );
            self.ticket = 0;
        }
    }
}

type ReleaseFunction = unsafe extern "C" fn(*const ResourceRequestV1, *mut c_char, usize) -> i32;
type SpineSourceUseOperationV2 = unsafe extern "C" fn(
    *const SpineSourceUseRequestV2,
    *mut SpineSourceUseReportV2,
    *mut c_char,
    usize,
) -> i32;

fn release_resource(
    function: ReleaseFunction,
    session: u64,
    resource: u64,
    operation: &str,
) -> Result<()> {
    let request = ResourceRequestV1::new(session, resource);
    let mut error = [0 as c_char; ERROR_CAPACITY];
    // SAFETY: request/error are valid for the call and the supplied function
    // has the exact release ABI declared above.
    let status = unsafe { function(&request, error.as_mut_ptr(), error.len()) };
    if status != 0 {
        return Err(ffi_error(operation, &error));
    }
    Ok(())
}

fn validate_target_state_report(
    report: &TargetStateReportV1,
    expected_active_branch: Option<u64>,
) -> Result<()> {
    if report.struct_size as usize != size_of::<TargetStateReportV1>()
        || report.abi_version != ABI_VERSION
        || report.flags != 0
        || report.reserved0 != 0
        || report.reserved != [0; 2]
        || expected_active_branch.is_some_and(|value| report.active_branch != value)
    {
        return Err(DeltafinError::new(
            "native provider returned an invalid target-state report",
        ));
    }
    Ok(())
}

fn checked_shape(rows: usize, columns: usize) -> Result<usize> {
    if rows == 0 || columns == 0 {
        return Err(DeltafinError::new(
            "provider tensor dimensions must be positive",
        ));
    }
    rows.checked_mul(columns)
        .ok_or_else(|| DeltafinError::new("provider tensor element count overflows usize"))
}

fn nonempty_slice_pointer(values: &[u8]) -> *const u8 {
    if values.is_empty() {
        ptr::null()
    } else {
        values.as_ptr()
    }
}

fn descriptor_bytes(
    descriptors: &[SpineTensorDescriptorV1],
    encoding: u32,
    length: impl Fn(&SpineTensorDescriptorV1) -> u64,
) -> Result<u64> {
    descriptors
        .iter()
        .filter(|descriptor| descriptor.encoding == encoding)
        .try_fold(0_u64, |total, descriptor| {
            total
                .checked_add(length(descriptor))
                .ok_or_else(|| DeltafinError::new("native spine byte report overflowed"))
        })
}

/// Exact-BF16 provider kernels consume large rank-two projection matrices
/// without expanding their resident storage. Norms/vectors and `[1, H]`
/// residual projections deliberately remain on the existing fp32 operator
/// path and therefore retain their promoted-size accounting.
const fn is_large_bf16_projection_descriptor(descriptor: &SpineTensorDescriptorV1) -> bool {
    descriptor.encoding == SPINE_ENCODING_RAW_BF16
        && descriptor.rank == 2
        && descriptor.shape[0] > 1
}

fn spine_descriptor_resident_bytes(
    descriptor: &SpineTensorDescriptorV1,
    borrowed_projection: bool,
) -> Result<u64> {
    if borrowed_projection && !is_large_bf16_projection_descriptor(descriptor) {
        return Err(DeltafinError::new(
            "only a large exact-BF16 projection may omit provider-owned residency",
        ));
    }
    match descriptor.encoding {
        SPINE_ENCODING_RAW_BF16 if borrowed_projection => Ok(0),
        SPINE_ENCODING_RAW_BF16 if is_large_bf16_projection_descriptor(descriptor) => {
            Ok(descriptor.data_length)
        }
        SPINE_ENCODING_RAW_BF16 => descriptor.data_length.checked_mul(2).ok_or_else(|| {
            DeltafinError::new("promoted BF16 vector provider residency overflowed")
        }),
        SPINE_ENCODING_RAW_F32 => Ok(descriptor.data_length),
        SPINE_ENCODING_ROW_I8_F16_SCALE => descriptor
            .auxiliary_length
            .checked_mul(2)
            .and_then(|scales| descriptor.data_length.checked_add(scales))
            .ok_or_else(|| DeltafinError::new("quantized spine provider residency overflowed")),
        _ => Err(DeltafinError::new(
            "spine provider residency saw an unknown encoding",
        )),
    }
}

fn reserve_canary_component(cursor: &mut usize, length: usize) -> Result<u64> {
    let aligned = cursor
        .checked_add(255)
        .map(|value| value & !255)
        .ok_or_else(|| DeltafinError::new("synthetic KDA spine alignment overflowed"))?;
    *cursor = aligned
        .checked_add(length)
        .ok_or_else(|| DeltafinError::new("synthetic KDA spine length overflowed"))?;
    Ok(aligned as u64)
}

fn canary_shape(dimensions: &[u64]) -> Result<([u64; 8], usize)> {
    if dimensions.is_empty() || dimensions.len() > 8 {
        return Err(DeltafinError::new(
            "synthetic KDA tensor rank is outside 1..8",
        ));
    }
    let mut shape = [0_u64; 8];
    shape[..dimensions.len()].copy_from_slice(dimensions);
    let elements = dimensions.iter().try_fold(1_usize, |total, &dimension| {
        let dimension = usize::try_from(dimension)
            .map_err(|_| DeltafinError::new("synthetic KDA dimension exceeds usize"))?;
        total
            .checked_mul(dimension)
            .ok_or_else(|| DeltafinError::new("synthetic KDA tensor size overflowed"))
    })?;
    Ok((shape, elements))
}

fn push_canary_raw(
    descriptors: &mut Vec<SpineTensorDescriptorV1>,
    cursor: &mut usize,
    slot: WeightSlot,
    dimensions: &[u64],
) -> Result<()> {
    let (shape, elements) = canary_shape(dimensions)?;
    let length = elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| DeltafinError::new("synthetic KDA raw bytes overflowed"))?;
    let offset = reserve_canary_component(cursor, length)?;
    descriptors.push(SpineTensorDescriptorV1 {
        slot: slot as u32,
        encoding: SPINE_ENCODING_RAW_F32,
        rank: dimensions.len() as u32,
        data_buffer: SPINE_BUFFER_OTHER,
        auxiliary_buffer: SPINE_BUFFER_NONE,
        reserved0: 0,
        shape,
        data_offset: offset,
        data_length: length as u64,
        auxiliary_offset: 0,
        auxiliary_length: 0,
        reserved: [0; 4],
    });
    Ok(())
}

fn push_canary_bf16(
    descriptors: &mut Vec<SpineTensorDescriptorV1>,
    cursor: &mut usize,
    slot: WeightSlot,
    dimensions: &[u64],
) -> Result<()> {
    let (shape, elements) = canary_shape(dimensions)?;
    let length = elements
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| DeltafinError::new("synthetic MLA bf16 bytes overflowed"))?;
    let offset = reserve_canary_component(cursor, length)?;
    descriptors.push(SpineTensorDescriptorV1 {
        slot: slot as u32,
        encoding: SPINE_ENCODING_RAW_BF16,
        rank: dimensions.len() as u32,
        data_buffer: SPINE_BUFFER_OTHER,
        auxiliary_buffer: SPINE_BUFFER_NONE,
        reserved0: 0,
        shape,
        data_offset: offset,
        data_length: length as u64,
        auxiliary_offset: 0,
        auxiliary_length: 0,
        reserved: [0; 4],
    });
    Ok(())
}

fn push_canary_q8(
    descriptors: &mut Vec<SpineTensorDescriptorV1>,
    cursor: &mut usize,
    slot: WeightSlot,
    rows: u64,
    columns: u64,
) -> Result<()> {
    let (shape, elements) = canary_shape(&[rows, columns])?;
    let scale_length = usize::try_from(rows)
        .ok()
        .and_then(|value| value.checked_mul(size_of::<u16>()))
        .ok_or_else(|| DeltafinError::new("synthetic KDA scale bytes overflowed"))?;
    let data_offset = reserve_canary_component(cursor, elements)?;
    let scale_offset = reserve_canary_component(cursor, scale_length)?;
    descriptors.push(SpineTensorDescriptorV1 {
        slot: slot as u32,
        encoding: SPINE_ENCODING_ROW_I8_F16_SCALE,
        rank: 2,
        data_buffer: SPINE_BUFFER_OTHER,
        auxiliary_buffer: SPINE_BUFFER_OTHER,
        reserved0: 0,
        shape,
        data_offset,
        data_length: elements as u64,
        auxiliary_offset: scale_offset,
        auxiliary_length: scale_length as u64,
        reserved: [0; 4],
    });
    Ok(())
}

fn synthetic_kda_spine_plan() -> Result<(Box<[SpineTensorDescriptorV1]>, ReadPlan)> {
    const HIDDEN: u64 = 32;
    const HEADS: u64 = 32;
    const HEAD_WIDTH: u64 = 32;
    const PROJECTION: u64 = HEADS * HEAD_WIDTH;
    let mut cursor = 0_usize;
    let mut descriptors = Vec::with_capacity(14);
    push_canary_raw(
        &mut descriptors,
        &mut cursor,
        WeightSlot::KdaALog,
        &[HEAD_WIDTH],
    )?;
    push_canary_raw(
        &mut descriptors,
        &mut cursor,
        WeightSlot::KdaDtBias,
        &[PROJECTION],
    )?;
    for slot in [
        WeightSlot::KdaQueryConvolution,
        WeightSlot::KdaKeyConvolution,
        WeightSlot::KdaValueConvolution,
    ] {
        push_canary_raw(&mut descriptors, &mut cursor, slot, &[PROJECTION, 1, 4])?;
    }
    push_canary_raw(
        &mut descriptors,
        &mut cursor,
        WeightSlot::KdaOutputNorm,
        &[HEAD_WIDTH],
    )?;
    for (slot, rows, columns) in [
        (WeightSlot::KdaQueryProjection, PROJECTION, HIDDEN),
        (WeightSlot::KdaKeyProjection, PROJECTION, HIDDEN),
        (WeightSlot::KdaValueProjection, PROJECTION, HIDDEN),
        (WeightSlot::KdaGateProjection, PROJECTION, HIDDEN),
        (WeightSlot::KdaFeatureAProjection, HEAD_WIDTH, HIDDEN),
        (WeightSlot::KdaFeatureBProjection, PROJECTION, HEAD_WIDTH),
        (WeightSlot::KdaBetaProjection, HEADS, HIDDEN),
        (WeightSlot::KdaOutputProjection, HIDDEN, PROJECTION),
    ] {
        push_canary_q8(&mut descriptors, &mut cursor, slot, rows, columns)?;
    }
    let plan = ReadPlan::open(
        vec![Extent::zero(BufferKind::Other, 0, cursor)],
        BufferLengths::new(0, 0, cursor),
        64 * 1024,
        CachePolicy::Resident,
    )?;
    Ok((descriptors.into_boxed_slice(), plan))
}

fn synthetic_mla_spine_plan() -> Result<(Box<[SpineTensorDescriptorV1]>, ReadPlan)> {
    const HIDDEN: u64 = 32;
    const HEADS: u64 = 2;
    const Q_LORA: u64 = 32;
    const KV_LORA: u64 = 32;
    const QK_NOPE: u64 = 16;
    const QK_ROPE: u64 = 32;
    const VALUE: u64 = 16;
    const QUERY_WIDTH: u64 = HEADS * (QK_NOPE + QK_ROPE);
    const VALUE_WIDTH: u64 = HEADS * VALUE;
    let mut cursor = 0_usize;
    let mut descriptors = Vec::with_capacity(8);
    push_canary_q8(
        &mut descriptors,
        &mut cursor,
        WeightSlot::MlaQueryAProjection,
        Q_LORA,
        HIDDEN,
    )?;
    push_canary_bf16(
        &mut descriptors,
        &mut cursor,
        WeightSlot::MlaQueryANorm,
        &[Q_LORA],
    )?;
    push_canary_q8(
        &mut descriptors,
        &mut cursor,
        WeightSlot::MlaQueryBProjection,
        QUERY_WIDTH,
        Q_LORA,
    )?;
    push_canary_q8(
        &mut descriptors,
        &mut cursor,
        WeightSlot::MlaKeyValueAProjection,
        KV_LORA + QK_ROPE,
        HIDDEN,
    )?;
    push_canary_bf16(
        &mut descriptors,
        &mut cursor,
        WeightSlot::MlaKeyValueANorm,
        &[KV_LORA],
    )?;
    push_canary_q8(
        &mut descriptors,
        &mut cursor,
        WeightSlot::MlaKeyValueBProjection,
        HEADS * (QK_NOPE + VALUE),
        KV_LORA,
    )?;
    push_canary_q8(
        &mut descriptors,
        &mut cursor,
        WeightSlot::MlaGateProjection,
        VALUE_WIDTH,
        HIDDEN,
    )?;
    push_canary_q8(
        &mut descriptors,
        &mut cursor,
        WeightSlot::MlaOutputProjection,
        HIDDEN,
        VALUE_WIDTH,
    )?;
    let plan = ReadPlan::open(
        vec![Extent::zero(BufferKind::Other, 0, cursor)],
        BufferLengths::new(0, 0, cursor),
        64 * 1024,
        CachePolicy::Resident,
    )?;
    Ok((descriptors.into_boxed_slice(), plan))
}

fn device_fields(device: Device) -> (u32, u32) {
    match device {
        Device::Cpu => (DEVICE_CPU, 0),
        Device::Mps => (DEVICE_MPS, 0),
        Device::Cuda(index) => (DEVICE_CUDA, u32::from(index)),
    }
}

fn decode_device(kind: u32, index: u32) -> Result<Device> {
    match (kind, index) {
        (DEVICE_CPU, 0) => Ok(Device::Cpu),
        (DEVICE_MPS, 0) => Ok(Device::Mps),
        (DEVICE_CUDA, index) => u16::try_from(index)
            .map(Device::Cuda)
            .map_err(|_| DeltafinError::new("provider returned an invalid CUDA device index")),
        _ => Err(DeltafinError::new(
            "provider returned an invalid selected device",
        )),
    }
}

fn require_same_session(
    expected: &Arc<SessionInner>,
    actual: &Arc<SessionInner>,
    resource: &str,
) -> Result<()> {
    if !Arc::ptr_eq(expected, actual) {
        return Err(DeltafinError::new(format!(
            "{resource} belongs to a different native provider session"
        )));
    }
    Ok(())
}

fn validate_route_mailbox(
    mailbox: Box<RouteMailboxV1>,
    session: &SessionInner,
    hidden: &ProviderTensor,
    cache: &ProviderCache,
) -> Result<NativeRouteMailbox> {
    let positions = mailbox.positions as usize;
    let edges = positions
        .checked_mul(ROUTE_TOP_K)
        .ok_or_else(|| DeltafinError::new("native route edge count overflowed"))?;
    if mailbox.abi_version != ABI_VERSION
        || mailbox.struct_size as usize != size_of::<RouteMailboxV1>()
        || mailbox.ticket == 0
        || positions != hidden.rows
        || positions > session.max_route_positions
        || mailbox.top_k as usize != ROUTE_TOP_K
        || mailbox.edge_count as usize != edges
        || edges > ROUTE_MAX_EDGES
        || mailbox.flags != 0
        || mailbox.hidden_columns != hidden.columns as u64
        || mailbox.cache_version != cache.version
        || mailbox.reserved != [0; 4]
    {
        if mailbox.ticket != 0 {
            let _ = release_resource(
                deltafin_provider_ticket_release_v1,
                session.handle,
                mailbox.ticket,
                "invalid native provider ticket release",
            );
        }
        return Err(DeltafinError::new(
            "native provider returned an invalid route mailbox header",
        ));
    }
    let ids = &mailbox.ordered_experts[..edges];
    let weights = &mailbox.ordered_weight_bits[..edges];
    for position in 0..positions {
        let row = &ids[position * ROUTE_TOP_K..(position + 1) * ROUTE_TOP_K];
        for (column, expert) in row.iter().copied().enumerate() {
            if usize::from(expert) >= session.experts || row[..column].contains(&expert) {
                let _ = release_resource(
                    deltafin_provider_ticket_release_v1,
                    session.handle,
                    mailbox.ticket,
                    "invalid native provider ticket release",
                );
                return Err(DeltafinError::new(format!(
                    "native route position {position} contains an invalid or repeated expert"
                )));
            }
        }
    }
    for (edge, bits) in weights.iter().copied().enumerate() {
        let weight = f32::from_bits(bits);
        if !weight.is_finite() || weight < 0.0 {
            let _ = release_resource(
                deltafin_provider_ticket_release_v1,
                session.handle,
                mailbox.ticket,
                "invalid native provider ticket release",
            );
            return Err(DeltafinError::new(format!(
                "native route edge {edge} contains an invalid fp32 weight"
            )));
        }
    }
    Ok(NativeRouteMailbox {
        raw: mailbox,
        positions,
        edges,
    })
}

fn fixed_c_string(value: &[c_char]) -> String {
    let end = value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len());
    let bytes: Vec<u8> = value[..end].iter().map(|byte| *byte as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn ffi_error(operation: &str, error: &[c_char]) -> DeltafinError {
    // The native function always NUL-terminates this buffer. Fall back to a
    // bounded conversion if a corrupt provider violates that contract.
    let message = if error.last() == Some(&0) {
        // SAFETY: the buffer is NUL-terminated and remains alive for this call.
        unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    } else {
        fixed_c_string(error)
    };
    DeltafinError::new(format!("{operation} failed: {message}"))
}

/// Serializes every test that drives work on the MPS device.
///
/// ATen submits all MPS work through one process-global Metal command buffer,
/// and two threads may not encode into it concurrently: one thread's commit
/// observes the other's still-open encoder and Metal aborts the whole process
/// with `commit command buffer with uncommitted encoder`. Locking provider
/// sessions individually cannot prevent this — the shared stream is below the
/// session boundary, and ATen also submits from allocator and completion
/// callbacks that never pass through this ABI.
///
/// The runtime satisfies the contract structurally: an engine owns exactly one
/// provider session, and the server admits one generation at a time. The test
/// harness does not — `cargo test` runs these tests on as many threads as the
/// host has cores — so every test that touches the device must hold this guard
/// for as long as its MPS session, tensors, or canaries live.
#[cfg(test)]
pub(crate) fn exclusive_mps_device() -> std::sync::MutexGuard<'static, ()> {
    static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A test that panicked while holding the guard has already been reported.
    // Poisoning it must not cascade into unrelated MPS tests.
    DEVICE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    static NEXT_TEMP_SOURCE: AtomicU64 = AtomicU64::new(1);

    struct CountedLease(Arc<AtomicUsize>);

    impl Drop for CountedLease {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn malformed_borrow_with_unprovable_abort_suppresses_destroy_before_lease_leak() {
        let inner = Arc::new(SessionInner {
            handle: 0,
            device: Device::Cpu,
            flags: 0,
            max_route_positions: 1,
            hidden_columns: 0,
            experts: 0,
            suppress_destroy: AtomicBool::new(false),
        });
        let observed = Arc::clone(&inner);
        let session = NativeProviderSession { inner };
        let drops = Arc::new(AtomicUsize::new(0));

        session.retain_unproven_source_lease(CountedLease(Arc::clone(&drops)));
        assert!(observed.suppress_destroy.load(Ordering::Acquire));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(session);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        // Dropping this final observation Arc exercises SessionInner::drop;
        // handle zero is never passed to FFI because suppression was already
        // published by the malformed-report cleanup helper.
        drop(observed);
    }

    #[test]
    fn exact_bf16_residency_predicate_excludes_vectors_and_one_row_scalars() {
        let descriptor = |rank: u32, shape: [u64; 8]| SpineTensorDescriptorV1 {
            slot: WeightSlot::LanguageModelHead as u32,
            encoding: SPINE_ENCODING_RAW_BF16,
            rank,
            data_buffer: SPINE_BUFFER_OTHER,
            auxiliary_buffer: SPINE_BUFFER_NONE,
            reserved0: 0,
            shape,
            data_offset: 0,
            data_length: 2,
            auxiliary_offset: 0,
            auxiliary_length: 0,
            reserved: [0; 4],
        };
        let projection = descriptor(2, [163_840, 7_168, 0, 0, 0, 0, 0, 0]);
        let vector = descriptor(1, [7_168, 0, 0, 0, 0, 0, 0, 0]);
        let one_row = descriptor(2, [1, 7_168, 0, 0, 0, 0, 0, 0]);
        assert!(is_large_bf16_projection_descriptor(&projection));
        assert!(!is_large_bf16_projection_descriptor(&vector));
        assert!(!is_large_bf16_projection_descriptor(&one_row));
        assert_eq!(
            spine_descriptor_resident_bytes(&projection, false).unwrap(),
            2
        );
        assert_eq!(
            spine_descriptor_resident_bytes(&projection, true).unwrap(),
            0
        );
        assert_eq!(spine_descriptor_resident_bytes(&vector, false).unwrap(), 4);
        assert_eq!(spine_descriptor_resident_bytes(&one_row, false).unwrap(), 4);
        assert!(spine_descriptor_resident_bytes(&vector, true).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn embedded_metallibs_survive_repeated_create_load_and_release() {
        let _device = exclusive_mps_device();
        let inventory = NativeProvider::inventory().unwrap();
        if !inventory.providers.mps {
            return;
        }
        // SAFETY: both test-only C functions accept one bounded integer and
        // retain no Rust pointer. Every iteration creates an owned dispatch
        // data copy, loads a library, and lets ARC release both before the next.
        unsafe {
            assert_eq!(k3_metal_embedded_library_cycle_test_v1(16), 0);
            assert_eq!(deltafin_route_mailbox_metallib_cycle_test_v1(16), 0);
        }
        let session = NativeProviderSession::target(Device::Mps).unwrap();
        let layouts = session
            .metal_expert_layouts("deltafin:embedded-metal-moe-mxfp4:v1")
            .unwrap();
        assert_eq!(
            layouts.layout_capabilities,
            METAL_CAP_RAW_V1 | METAL_CAP_SCALE4_V2
        );
    }

    #[test]
    fn non_mps_sessions_reject_metal_expert_cache_control_without_affecting_teardown() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        assert!(
            session
                .flush_metal_expert_cache()
                .unwrap_err()
                .to_string()
                .contains("MPS provider")
        );
        assert!(
            session
                .metal_expert_cache_stats()
                .unwrap_err()
                .to_string()
                .contains("MPS provider")
        );
        drop(session);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_expert_cache_flush_stats_and_multi_session_teardown_are_total() {
        let _device = exclusive_mps_device();
        let inventory = NativeProvider::inventory().unwrap();
        if !inventory.providers.mps {
            return;
        }
        let first = NativeProviderSession::target(Device::Mps).unwrap();
        first
            .metal_expert_layouts("deltafin:embedded-metal-moe-mxfp4:v1")
            .unwrap();
        first.flush_metal_expert_cache().unwrap();
        let before = first.metal_expert_cache_stats().unwrap();
        assert_eq!(before.cache_entries, 0);
        assert!(before.zero_copy_wraps >= before.cache_entries);

        // The bridge cache is process-global. A second session may flush it
        // without invalidating the first; session teardown repeats that same
        // idempotent safety boundary and must not deadlock or throw.
        let second = NativeProviderSession::target(Device::Mps).unwrap();
        second.flush_metal_expert_cache().unwrap();
        let after = second.metal_expert_cache_stats().unwrap();
        assert_eq!(after.cache_entries, 0);
        assert_eq!(after.calls, before.calls);
        assert_eq!(after.zero_copy_wraps, before.zero_copy_wraps);
        assert_eq!(after.copies, before.copies);
        drop(first);
        second.flush_metal_expert_cache().unwrap();
        assert_eq!(second.metal_expert_cache_stats().unwrap().cache_entries, 0);
        drop(second);
    }

    struct TempSource(PathBuf);

    impl Drop for TempSource {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn synthetic_spine_buffers() -> (TempSource, LayerBuffers, Vec<SpineTensorDescriptorV1>) {
        let nonce = NEXT_TEMP_SOURCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "deltafin-provider-spine-{}-{nonce}.bin",
            std::process::id()
        ));
        let mut bytes = vec![253_u8, 0, 7, 127];
        bytes.extend_from_slice(&0x3800_u16.to_le_bytes()); // fp16 0.5
        bytes.extend_from_slice(&0xc000_u16.to_le_bytes()); // fp16 -2.0
        bytes.extend_from_slice(&0x3f80_u16.to_le_bytes()); // bf16 1.0
        bytes.extend_from_slice(&0xc000_u16.to_le_bytes()); // bf16 -2.0
        bytes.extend_from_slice(&3.5_f32.to_le_bytes());
        fs::write(&path, bytes).unwrap();
        let source = TempSource(path.clone());
        let plan = ReadPlan::open(
            vec![
                Extent::new(&path, 0, BufferKind::Quantized, 0, 4),
                Extent::new(&path, 4, BufferKind::Scales, 0, 4),
                Extent::new(&path, 8, BufferKind::Other, 0, 4),
                Extent::zero(BufferKind::Other, 4, 252),
                Extent::new(&path, 12, BufferKind::Other, 256, 4),
            ],
            BufferLengths::new(4, 4, 260),
            64,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let (buffers, _) = reader.read(&plan).unwrap();
        let descriptor = |slot,
                          encoding,
                          rank,
                          shape,
                          data_buffer,
                          data_offset,
                          data_length,
                          auxiliary_buffer,
                          auxiliary_offset,
                          auxiliary_length| {
            SpineTensorDescriptorV1 {
                slot,
                encoding,
                rank,
                data_buffer,
                auxiliary_buffer,
                reserved0: 0,
                shape,
                data_offset,
                data_length,
                auxiliary_offset,
                auxiliary_length,
                reserved: [0; 4],
            }
        };
        let descriptors = vec![
            descriptor(
                1,
                SPINE_ENCODING_RAW_BF16,
                1,
                [2, 0, 0, 0, 0, 0, 0, 0],
                SPINE_BUFFER_OTHER,
                0,
                4,
                SPINE_BUFFER_NONE,
                0,
                0,
            ),
            descriptor(
                7,
                SPINE_ENCODING_RAW_F32,
                1,
                [1, 0, 0, 0, 0, 0, 0, 0],
                SPINE_BUFFER_OTHER,
                256,
                4,
                SPINE_BUFFER_NONE,
                0,
                0,
            ),
            descriptor(
                13,
                SPINE_ENCODING_ROW_I8_F16_SCALE,
                2,
                [2, 2, 0, 0, 0, 0, 0, 0],
                SPINE_BUFFER_QUANTIZED,
                0,
                4,
                SPINE_BUFFER_SCALES,
                0,
                4,
            ),
        ];
        (source, buffers, descriptors)
    }

    fn zero_target_tail_globals() -> (LayerBuffers, Vec<SpineTensorDescriptorV1>) {
        const ROW_BYTES: usize = 7_168 * 2;
        const TOTAL_BYTES: usize = ROW_BYTES * 3;
        let plan = ReadPlan::open(
            vec![Extent::zero(BufferKind::Other, 0, TOTAL_BYTES)],
            BufferLengths::new(0, 0, TOTAL_BYTES),
            64,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let (buffers, _) = reader.read(&plan).unwrap();
        let descriptor =
            |slot: u32, rank: u32, shape: [u64; 8], offset: usize| SpineTensorDescriptorV1 {
                slot,
                encoding: SPINE_ENCODING_RAW_BF16,
                rank,
                data_buffer: SPINE_BUFFER_OTHER,
                auxiliary_buffer: SPINE_BUFFER_NONE,
                reserved0: 0,
                shape,
                data_offset: offset as u64,
                data_length: ROW_BYTES as u64,
                auxiliary_offset: 0,
                auxiliary_length: 0,
                reserved: [0; 4],
            };
        (
            buffers,
            vec![
                descriptor(41, 1, [7_168, 0, 0, 0, 0, 0, 0, 0], 0),
                descriptor(42, 1, [7_168, 0, 0, 0, 0, 0, 0, 0], ROW_BYTES),
                descriptor(43, 2, [1, 7_168, 0, 0, 0, 0, 0, 0], ROW_BYTES * 2),
            ],
        )
    }

    #[test]
    fn target_sequence_abi_layout_and_offsets_are_fixed() {
        assert_eq!(TARGET_EXPERT_RETAIN_METAL_WRAPPERS, 1);
        assert_eq!(TARGET_SEQUENCE_CAPTURE_DSPARK, 1);
        assert_eq!(TARGET_SEQUENCE_FULL_COMMIT_ONLY, 2);
        assert_eq!(TARGET_SEQUENCE_CAPTURE_EAGLE3, 4);
        assert_eq!(ProposalCapture::DSpark.flag(), TARGET_SEQUENCE_CAPTURE_DSPARK);
        assert_eq!(ProposalCapture::Eagle3.flag(), TARGET_SEQUENCE_CAPTURE_EAGLE3);
        assert_eq!(ProposalCapture::Eagle3.columns(), 21_504);
        assert_eq!(
            TARGET_SEQUENCE_CAPTURE_DSPARK | TARGET_SEQUENCE_FULL_COMMIT_ONLY,
            3
        );
        assert_eq!(size_of::<MemoryRequestV1>(), 64);
        assert_eq!(size_of::<MemoryReportV1>(), 104);
        assert_eq!(std::mem::offset_of!(MemoryReportV1, active_bytes), 32);
        assert_eq!(size_of::<MetalExpertLayoutsRequestV1>(), 64);
        assert_eq!(size_of::<MetalExpertLayoutsReportV1>(), 64);
        assert_eq!(size_of::<TargetFinishExpertsRequestV1>(), 160);
        assert_eq!(size_of::<TargetSequenceBeginBf16RequestV1>(), 80);
        assert_eq!(size_of::<TargetSequenceBeginReportV1>(), 64);
        assert_eq!(size_of::<TargetSequencePrepareRequestV1>(), 64);
        assert_eq!(size_of::<TargetSequencePrepareReportV1>(), 6_224);
        assert_eq!(size_of::<TargetSequencePrefetchHintReportV1>(), 128);
        assert_eq!(size_of::<TargetSequenceFinishExpertsRequestV1>(), 256);
        assert_eq!(size_of::<TargetSequenceFinishExpertSpansRequestV1>(), 760);
        assert_eq!(size_of::<TargetSequenceFinishExpertsRequestV2>(), 200);
        assert_eq!(size_of::<TargetSequencePlanExpertsRequestV1>(), 240);
        assert_eq!(size_of::<TargetSequencePlanExpertsReportV1>(), 208);
        assert_eq!(
            size_of::<TargetSequenceFinishPlannedExpertsRequestV1>(),
            240
        );
        assert_eq!(size_of::<TargetSequenceFinishExpertsReportV1>(), 64);
        assert_eq!(size_of::<TargetSequenceTailReportV1>(), 320);
        assert_eq!(size_of::<TargetSequenceCommitRequestV1>(), 64);
        assert_eq!(size_of::<TargetSequenceCommitReportV1>(), 64);
        assert_eq!(size_of::<TargetSequenceStatsReportV1>(), 160);
        assert_eq!(align_of::<TargetSequencePrepareReportV1>(), 8);
        assert_eq!(
            std::mem::offset_of!(TargetSequencePrepareReportV1, ordered_experts),
            48
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequencePrepareReportV1, ordered_weight_bits),
            2_096
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertsRequestV1, expert_ids),
            64
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertsRequestV1, expert_major_bytes),
            192
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertsRequestV1, expert_span_bytes),
            224
        );
        assert_eq!(
            std::mem::offset_of!(
                TargetSequenceFinishExpertSpansRequestV1,
                expert_span_pointers
            ),
            192
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertSpansRequestV1, expert_span_bytes),
            720
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertsRequestV2, expert_ids),
            64
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertsRequestV2, expert_major_bytes),
            80
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertsRequestV2, expert_span_pointers),
            96
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertsRequestV2, reserved),
            136
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequencePlanExpertsRequestV1, expert_ids),
            60
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequencePlanExpertsRequestV1, metal_shader_path),
            192
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequencePlanExpertsReportV1, missing_experts),
            56
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishPlannedExpertsRequestV1, missing_experts),
            60
        );
        assert_eq!(
            std::mem::offset_of!(
                TargetSequenceFinishPlannedExpertsRequestV1,
                expert_major_bytes
            ),
            192
        );
        assert_eq!(
            std::mem::offset_of!(TargetSequenceFinishExpertsReportV1, layer_index),
            24
        );
    }

    #[test]
    fn cpu_memory_snapshot_is_native_bounded_and_refuses_accelerator_trim() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        assert_eq!(
            session.memory_snapshot(false).unwrap(),
            NativeProviderMemorySnapshot {
                device: Device::Cpu,
                active_bytes: None,
                reserved_bytes: None,
                recommended_bytes: None,
                total_bytes: None,
                available_bytes: None,
                cache_trimmed: false,
            }
        );
        let error = session.memory_snapshot(true).unwrap_err();
        assert!(error.to_string().contains("no accelerator cache"));
    }

    #[test]
    fn cuda_plan_report_validation_covers_cpu_cuda_and_zero_miss_all_hit() {
        let canonical = [2_u16, 7, 11];
        let mut cpu = TargetSequencePlanExpertsReportV1::request();
        cpu.abi_version = ABI_VERSION;
        cpu.plan = 91;
        cpu.spine_generation = 44;
        cpu.layer_index = 6;
        cpu.first_row = 3;
        cpu.row_count = 2;
        cpu.effective_backend = TARGET_EXPERT_CPU;
        cpu.missing_count = canonical.len() as u32;
        cpu.missing_experts[..canonical.len()].copy_from_slice(&canonical);
        let validated = validate_target_sequence_expert_plan_report(
            &cpu,
            TargetExpertBackend::Auto,
            44,
            6,
            3,
            2,
            &canonical,
        )
        .unwrap();
        assert_eq!(validated.effective_backend, TargetExpertBackend::Cpu);
        assert_eq!(validated.missing_experts.as_ref(), canonical);
        assert!(!validated.residency_enabled);

        let mut all_hit = TargetSequencePlanExpertsReportV1::request();
        all_hit.abi_version = ABI_VERSION;
        all_hit.plan = 92;
        all_hit.spine_generation = 44;
        all_hit.layer_index = 6;
        all_hit.first_row = 3;
        all_hit.row_count = 2;
        all_hit.effective_backend = TARGET_EXPERT_CUDA;
        all_hit.cache_capacity_experts = 12;
        all_hit.residency_enabled = 1;
        let validated = validate_target_sequence_expert_plan_report(
            &all_hit,
            TargetExpertBackend::Auto,
            44,
            6,
            3,
            2,
            &canonical,
        )
        .unwrap();
        assert_eq!(validated.effective_backend, TargetExpertBackend::Cuda);
        assert!(validated.missing_experts.is_empty());
        assert!(validated.residency_enabled);
        assert!(borrowed_bytes_pointer(&[]).is_null());

        assert!(
            validate_target_sequence_expert_plan_report(
                &cpu,
                TargetExpertBackend::Cuda,
                44,
                6,
                3,
                2,
                &canonical,
            )
            .unwrap_err()
            .to_string()
            .contains("silently downgraded")
        );

        let mut disabled_cache_without_misses = all_hit;
        disabled_cache_without_misses.cache_capacity_experts = 0;
        disabled_cache_without_misses.residency_enabled = 0;
        assert!(
            validate_target_sequence_expert_plan_report(
                &disabled_cache_without_misses,
                TargetExpertBackend::Auto,
                44,
                6,
                3,
                2,
                &canonical,
            )
            .unwrap_err()
            .to_string()
            .contains("cache state")
        );
    }

    #[test]
    fn cuda_plan_route_union_is_exact_canonical_and_bounded() {
        let first = std::array::from_fn(|index| index as u16);
        let second = std::array::from_fn(|index| (index + 8) as u16);
        let mailbox = TargetSequenceMailbox {
            layer_index: 5,
            spine_generation: 77,
            routes: vec![
                TargetRoute {
                    layer_index: 5,
                    spine_generation: 77,
                    ordered_experts: first,
                    ordered_weight_bits: [0; ROUTE_TOP_K],
                },
                TargetRoute {
                    layer_index: 5,
                    spine_generation: 77,
                    ordered_experts: second,
                    ordered_weight_bits: [0; ROUTE_TOP_K],
                },
            ]
            .into_boxed_slice(),
        };
        let union =
            target_sequence_route_union(&mailbox, 0, 2, TARGET_SEQUENCE_MAX_EXPERTS).unwrap();
        assert_eq!(union.as_ref(), (0_u16..24).collect::<Vec<_>>());
        assert!(
            target_sequence_route_union(&mailbox, 1, 2, TARGET_SEQUENCE_MAX_EXPERTS,)
                .unwrap_err()
                .to_string()
                .contains("outside")
        );
    }

    #[test]
    fn cuda_plan_raii_disarms_after_drop_and_failed_explicit_release() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let make_plan = |handle: u64| {
            let lease = Arc::new(TargetSequenceExpertPlanLease::new(handle));
            let plan = TargetSequenceExpertPlan {
                session: Arc::clone(&session.inner),
                lease: Arc::clone(&lease),
                sequence: 1,
                spine_generation: 2,
                layer_index: 1,
                first_row: 0,
                row_count: 1,
                canonical_experts: vec![7].into_boxed_slice(),
                missing_experts: vec![7].into_boxed_slice(),
                effective_backend: TargetExpertBackend::Cpu,
                cache_capacity_experts: 0,
                residency_enabled: false,
            };
            (plan, lease)
        };

        let (plan, lease) = make_plan(u64::MAX - 10);
        drop(plan);
        assert_eq!(lease.handle(), 0);

        let (plan, lease) = make_plan(u64::MAX - 11);
        assert!(plan.release().is_err());
        // Explicit release leaves the handle armed only long enough for the
        // consuming value's Drop to make its final independent attempt.
        assert_eq!(lease.handle(), 0);
    }

    #[test]
    fn cuda_plan_consume_on_attempt_marker_is_one_way() {
        let lease = TargetSequenceExpertPlanLease::new(1234);
        assert_eq!(lease.handle(), 1234);
        assert_eq!(lease.consume(), 1234);
        assert_eq!(lease.handle(), 0);
        assert_eq!(lease.consume(), 0);
    }

    #[test]
    fn cuda_plan_extern_rejects_non_cuda_before_sequence_lookup() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let mut request = TargetSequencePlanExpertsRequestV1 {
            struct_size: size_of::<TargetSequencePlanExpertsRequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: session.inner.handle,
            sequence: 999,
            spine_generation: 2,
            layer_index: 1,
            first_row: 0,
            row_count: 1,
            expert_backend: TARGET_EXPERT_AUTO,
            cpu_threads: 1,
            expert_count: 1,
            flags: 0,
            expert_ids: [0; TARGET_SEQUENCE_MAX_EXPERTS],
            metal_shader_path: ptr::null(),
            metal_shader_path_length: 0,
            reserved: [0; 4],
        };
        request.expert_ids[0] = 7;
        let mut report = TargetSequencePlanExpertsReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: fixed-layout request/report/error objects remain live.
        let status = unsafe {
            deltafin_provider_target_sequence_plan_experts_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_ne!(status, 0);
        assert!(
            ffi_error("Rust CUDA-plan ABI gate", &error)
                .to_string()
                .contains("selected CUDA")
        );
        assert_eq!(report.plan, 0);
    }

    #[test]
    fn expert_layout_ids_and_spans_are_never_inferred_from_byte_length() {
        assert_eq!(
            expert_layout_abi(ExpertStorageLayout::RawV1),
            (EXPERT_LAYOUT_RAW_V1, K3_EXPERT_SOURCE_BYTES as u64)
        );
        assert_eq!(
            expert_layout_abi(ExpertStorageLayout::Scale4V2),
            (EXPERT_LAYOUT_SCALE4_V2, K3_SCALE4_BLOB_BYTES as u64)
        );
        assert!(
            MetalExpertLayouts {
                descriptor_abi: METAL_DESCRIPTOR_ABI_V1,
                layout_capabilities: METAL_CAP_RAW_V1 | METAL_CAP_SCALE4_V2,
            }
            .supports_scale4_v2()
        );
        assert!(
            !MetalExpertLayouts {
                descriptor_abi: 0,
                layout_capabilities: METAL_CAP_RAW_V1 | METAL_CAP_SCALE4_V2,
            }
            .supports_scale4_v2()
        );
    }

    #[test]
    fn target_sequence_begin_is_bounded_and_reaches_the_versioned_abi() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        assert!(
            session
                .begin_target_sequence_bf16(&[], 0, TargetSequenceMode::Prefill)
                .unwrap_err()
                .to_string()
                .contains("1..64")
        );
        assert!(
            session
                .begin_target_sequence_bf16(&[], 65, TargetSequenceMode::Verify)
                .unwrap_err()
                .to_string()
                .contains("1..64")
        );
        assert!(
            session
                .begin_target_sequence_bf16(&[0; 31], 2, TargetSequenceMode::Verify)
                .unwrap_err()
                .to_string()
                .contains("BF16 bytes")
        );
        // A structurally valid request reaches C++ and fails before allocating
        // caches because this test intentionally did not bind the huge head.
        assert!(
            session
                .begin_target_sequence_bf16(&[0; 2 * 7_168 * 2], 2, TargetSequenceMode::Verify,)
                .unwrap_err()
                .to_string()
                .contains("global groups")
        );

        let rows = [0_u8; 7_168 * 2];
        let request = TargetSequenceBeginBf16RequestV1 {
            struct_size: size_of::<TargetSequenceBeginBf16RequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: session.inner.handle,
            data: rows.as_ptr(),
            byte_length: rows.len() as u64,
            positions: 1,
            mode: 99,
            flags: 0,
            reserved0: 0,
            reserved: [0; 4],
        };
        let mut report = TargetSequenceBeginReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/report/error point to live fixed-layout objects.
        let status = unsafe {
            deltafin_provider_target_sequence_begin_bf16_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_ne!(status, 0);
        assert!(
            ffi_error("target sequence test", &error)
                .to_string()
                .contains("mode")
        );
        assert_eq!(report.sequence, 0);

        let request = TargetSequenceBeginBf16RequestV1 {
            struct_size: size_of::<TargetSequenceBeginBf16RequestV1>() as u32,
            abi_version: ABI_VERSION,
            session: session.inner.handle,
            data: rows.as_ptr(),
            byte_length: rows.len() as u64,
            positions: 1,
            mode: TARGET_SEQUENCE_PREFILL,
            flags: TARGET_SEQUENCE_FULL_COMMIT_ONLY,
            reserved0: 0,
            reserved: [0; 4],
        };
        let mut report = TargetSequenceBeginReportV1::request();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: request/report/error point to live fixed-layout objects.
        let status = unsafe {
            deltafin_provider_target_sequence_begin_bf16_v1(
                &request,
                &mut report,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_ne!(status, 0);
        assert!(
            ffi_error("target sequence full-commit prefill test", &error)
                .to_string()
                .contains("full-commit-only requires verify mode")
        );
        assert_eq!(report.sequence, 0);

        // Every capture combination is accepted for Verify and reaches the
        // next independent admission gate. This session deliberately has no
        // huge immutable target globals, so no transaction or cache is
        // allocated.
        for capture in [
            None,
            Some(ProposalCapture::DSpark),
            Some(ProposalCapture::Eagle3),
        ] {
            let error = session
                .begin_target_sequence_bf16_verify_full_commit_only(&rows, 1, capture)
                .unwrap_err();
            assert!(error.to_string().contains("global groups"));
        }
    }

    #[test]
    fn full_commit_only_is_narrow_and_normal_verify_prefixes_are_unchanged() {
        assert!(
            validate_target_sequence_commit_prefix(TargetSequenceMode::Verify, 5, false, 0,)
                .is_ok()
        );
        assert!(
            validate_target_sequence_commit_prefix(TargetSequenceMode::Verify, 5, false, 3,)
                .is_ok()
        );
        assert!(
            validate_target_sequence_commit_prefix(TargetSequenceMode::Verify, 5, false, 5,)
                .is_ok()
        );

        let partial =
            validate_target_sequence_commit_prefix(TargetSequenceMode::Verify, 5, true, 3)
                .unwrap_err();
        assert!(
            partial
                .to_string()
                .contains("cannot commit a partial prefix")
        );
        assert!(
            validate_target_sequence_commit_prefix(TargetSequenceMode::Verify, 5, true, 5,).is_ok()
        );

        let prefill =
            validate_target_sequence_commit_prefix(TargetSequenceMode::Prefill, 5, false, 3)
                .unwrap_err();
        assert!(prefill.to_string().contains("invalid for its mode"));
        assert!(
            validate_target_sequence_commit_prefix(TargetSequenceMode::Prefill, 5, false, 5,)
                .is_ok()
        );
        assert!(
            validate_target_sequence_commit_prefix(TargetSequenceMode::Verify, 5, false, 6,)
                .is_err()
        );
    }

    #[test]
    fn rejected_full_commit_prefix_consumes_the_raii_value_without_poisoning_session() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let sequence = TargetSequence {
            session: Arc::clone(&session.inner),
            // A deliberately stale nonzero handle makes Drop exercise the
            // native cancellation call without owning any provider resource.
            handle: u64::MAX,
            mode: TargetSequenceMode::Verify,
            position_count: 4,
            next_layer: 93,
            state: TargetSequenceState::ReadyToCommit,
            waiting: None,
            expert_plan: None,
            capture: None,
            full_commit_only: true,
        };
        let error = sequence.commit_prefix(3).unwrap_err();
        assert!(error.to_string().contains("cannot commit a partial prefix"));

        // The consumed value's Drop path failed closed on the stale handle;
        // the same real session still admits the next independent request.
        let rows = [0_u8; 7_168 * 2];
        let error = session
            .begin_target_sequence_bf16_verify_full_commit_only(&rows, 1, None)
            .unwrap_err();
        assert!(error.to_string().contains("global groups"));
    }

    #[test]
    fn linked_abi_inventory_is_versioned_and_has_cpu() {
        let inventory = NativeProvider::inventory().unwrap();
        assert!(!inventory.libtorch_version.is_empty());
        assert_eq!(
            inventory
                .providers
                .select(crate::platform::DeviceRequest::Cpu)
                .unwrap(),
            Device::Cpu
        );
        if !cfg!(target_os = "linux") {
            assert!(!inventory.cuda_moe_compiled);
            assert!(!inventory.cuda_exact_bf16_compiled);
        }
    }

    #[test]
    fn target_pilot_admission_is_exact_one_shot_and_real_session_only() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let admission = session.enable_target_pilot().unwrap();
        assert_eq!(admission.layer_capacity, TARGET_PILOT_LAYER_CAPACITY);
        assert_eq!(admission.reserve_bytes, TARGET_PILOT_RESERVE_BYTES);
        assert!(
            session
                .enable_target_pilot()
                .unwrap_err()
                .to_string()
                .contains("exactly once")
        );

        let synthetic = NativeProviderSession::synthetic_split(Device::Cpu, 1, 32, 32).unwrap();
        assert!(
            synthetic
                .enable_target_pilot()
                .unwrap_err()
                .to_string()
                .contains("real CPU or MPS")
        );
    }

    #[test]
    fn cpu_runs_the_same_coarse_canary_boundary_as_accelerators() {
        let report = NativeProvider::canary(Device::Cpu, (32, 32), false).unwrap();
        assert!(report.core_passed());
        assert_eq!(report.packed_shape, (32, 32));
        let spine = NativeProvider::spine_binding_canary(Device::Cpu).unwrap();
        assert_eq!(spine.tensors, 3);
        assert_eq!(spine.quantized_tensors, 1);
        assert_eq!(spine.raw_tensors, 2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mps_f32_upload_owns_static_and_dropped_sources_before_return() {
        let _device = exclusive_mps_device();
        let inventory = NativeProvider::inventory().unwrap();
        if !inventory.providers.mps {
            return;
        }

        // A promoted constant can reside in the executable mapping. Passing
        // its borrowed address directly to MPS previously let Metal wire an
        // __TEXT page for I/O and SIGBUS the next Rust instruction/read.
        let session = NativeProviderSession::target(Device::Mps).unwrap();
        let static_tensor = session.upload_f32(1, 32, &[0.0_f32; 32]).unwrap();

        // Also exercise the ordinary caller-lifetime contract: the source is
        // destroyed and similarly-sized allocations churn the heap before a
        // device-to-host read forces consumption of the queued MPS upload.
        const ELEMENTS: usize = 8_192;
        let source: Vec<f32> = (0..ELEMENTS)
            .map(|index| ((index % 257) as f32 - 128.0) / 16.0)
            .collect();
        let tensor = session.upload_f32(1, ELEMENTS, &source).unwrap();
        drop(source);
        for seed in 0..32 {
            let churn = vec![-10_000.0_f32 - seed as f32; ELEMENTS];
            std::hint::black_box(&churn);
        }

        assert_eq!(static_tensor.read_f32().unwrap(), vec![0.0_f32; 32]);
        let actual = tensor.read_f32().unwrap();
        assert_eq!(actual.len(), ELEMENTS);
        for (index, value) in actual.iter().enumerate() {
            let expected = ((index % 257) as f32 - 128.0) / 16.0;
            assert_eq!(value.to_bits(), expected.to_bits(), "element {index}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mps_cache_trim_preserves_live_tensors_and_rejects_live_tickets() {
        let _device = exclusive_mps_device();
        let inventory = NativeProvider::inventory().unwrap();
        if !inventory.providers.mps {
            return;
        }

        let session = NativeProviderSession::target(Device::Mps).unwrap();
        let expected: Vec<f32> = (0..4_096)
            .map(|index| ((index % 127) as f32 - 63.0) / 8.0)
            .collect();
        let live = session.upload_f32(1, expected.len(), &expected).unwrap();
        let scratch = session.upload_f32(1, 65_536, &[1.0; 65_536]).unwrap();
        drop(scratch);
        let before = session.memory_snapshot(false).unwrap();
        assert!(before.active_bytes.is_some());
        assert!(before.reserved_bytes.is_some());
        assert!(before.recommended_bytes.is_some());
        let after = session.memory_snapshot(true).unwrap();
        assert!(after.cache_trimmed);
        let actual = live.read_f32().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "element {index}");
        }

        let split = NativeProviderSession::synthetic_split(Device::Mps, 1, 32, 32).unwrap();
        let hidden = split.upload_f32(1, 32, &[0.125; 32]).unwrap();
        let mut cache = split.create_cache_f32(1, 32, None).unwrap();
        let prepared = split.prepare_layer(&hidden, &mut cache, 0).unwrap();
        let error = split.memory_snapshot(true).unwrap_err();
        assert!(error.to_string().contains("quiescent transaction"));
        drop(prepared);
        assert!(split.memory_snapshot(true).unwrap().cache_trimmed);
        assert_eq!(hidden.read_f32().unwrap(), vec![0.125; 32]);
    }

    #[test]
    fn target_globals_are_exact_immutable_groups_and_never_accept_embedding_slot() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let (buffers, descriptors) = zero_target_tail_globals();
        let report = session
            .bind_target_globals(TargetGlobalGroup::Tail, &descriptors, &buffers)
            .unwrap();
        assert_eq!(report.group, TargetGlobalGroup::Tail);
        assert_eq!(report.tensor_count, 3);
        assert_eq!(report.raw_tensor_count, 3);
        assert_eq!(report.quantized_tensor_count, 0);
        assert_eq!(report.groups_ready, 1);
        assert_eq!(report.resident_storage_bytes, (3 * 7_168 * 4) as u64);

        let duplicate = session
            .bind_target_globals(TargetGlobalGroup::Tail, &descriptors, &buffers)
            .unwrap_err();
        assert!(duplicate.to_string().contains("immutable"));

        let mut embedding = descriptors.clone();
        embedding.truncate(1);
        embedding[0].slot = WeightSlot::TokenEmbedding as u32;
        let error = session
            .bind_target_globals(TargetGlobalGroup::LanguageModelHead, &embedding, &buffers)
            .unwrap_err();
        assert!(error.to_string().contains("slot outside"));
    }

    #[test]
    fn target_begin_fails_before_cache_allocation_until_both_global_groups_are_ready() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        assert!(
            session
                .begin_target_position_bf16(&[0_u8; 7_168 * 2])
                .unwrap_err()
                .to_string()
                .contains("global groups")
        );
        assert!(
            session
                .begin_target_position_bf16(&[0_u8; 31])
                .unwrap_err()
                .to_string()
                .contains("exactly")
        );

        let (buffers, descriptors) = zero_target_tail_globals();
        session
            .bind_target_globals(TargetGlobalGroup::Tail, &descriptors, &buffers)
            .unwrap();
        let hidden = session.upload_f32(1, 7_168, &[0.0; 7_168]).unwrap();
        assert!(
            session
                .begin_target_position(&hidden)
                .unwrap_err()
                .to_string()
                .contains("global groups")
        );
    }

    #[test]
    fn target_expert_backend_resolves_before_expert_io_and_fails_closed() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        assert_eq!(
            session
                .resolve_target_expert_backend(TargetExpertBackend::Auto)
                .unwrap(),
            TargetExpertBackend::Cpu
        );
        assert_eq!(
            session
                .resolve_target_expert_backend(TargetExpertBackend::Cpu)
                .unwrap(),
            TargetExpertBackend::Cpu
        );
        assert!(
            session
                .resolve_target_expert_backend(TargetExpertBackend::Metal)
                .unwrap_err()
                .to_string()
                .contains("MPS")
        );
        assert!(
            session
                .resolve_target_expert_backend(TargetExpertBackend::Cuda)
                .unwrap_err()
                .to_string()
                .contains("CUDA provider")
        );
    }

    #[test]
    fn split_layer_canary_owns_state_and_commits_exactly_once() {
        let report = NativeProvider::split_layer_canary(Device::Cpu).unwrap();
        assert_eq!(report.positions, 2);
        assert_eq!(report.route_edges, 2 * ROUTE_TOP_K);
        assert_eq!(report.committed_cache_version, 1);
    }

    #[test]
    fn kda_canary_owns_all_state_and_cancels_before_commit() {
        let report = NativeProvider::kda_transaction_canary(Device::Cpu).unwrap();
        assert_eq!(report.device, Device::Cpu);
        assert_eq!(report.canceled_version, 0);
        assert_eq!(report.committed_version, 1);
        assert_eq!(report.convolution_elements, 12_288);
        assert_eq!(report.recurrent_elements, 32_768);
    }

    #[test]
    fn mla_canary_cancels_growth_then_commits_one_position() {
        let report = NativeProvider::mla_transaction_canary(Device::Cpu).unwrap();
        assert_eq!(report.device, Device::Cpu);
        assert_eq!(report.canceled_version, 0);
        assert_eq!(report.committed_version, 1);
        assert_eq!(report.committed_length, 1);
        assert!(report.capacity >= 1);
        assert_eq!(report.input_bundle_rows, 0);
        assert_eq!(report.production_bundle_rows, 0);
    }

    #[test]
    fn dropping_prepare_ticket_discards_staged_cache_and_preserves_route_bits() {
        let session = NativeProviderSession::synthetic_split(Device::Cpu, 2, 32, 32).unwrap();
        let hidden_values: Vec<f32> = (0..64).map(|index| index as f32 / 32.0).collect();
        let initial_cache = vec![0.5_f32; 64];
        let hidden = session.upload_f32(2, 32, &hidden_values).unwrap();
        let mut cache = session
            .create_cache_f32(2, 32, Some(&initial_cache))
            .unwrap();

        let (first_positions, first_ids, first_weights, first_cache_version) = {
            let prepared = session.prepare_layer(&hidden, &mut cache, 7).unwrap();
            let route = prepared.route();
            let snapshot = (
                route.positions(),
                route.ordered_experts().to_vec(),
                route.ordered_weight_bits().to_vec(),
                route.cache_version(),
            );
            drop(prepared);
            snapshot
        };
        assert_eq!(cache.version(), 0);
        assert_eq!(cache.read_f32().unwrap(), initial_cache);

        let second = session.prepare_layer(&hidden, &mut cache, 7).unwrap();
        // Ticket IDs are intentionally monotonic and therefore differ after a
        // cancellation. Only the exact route payload and cache snapshot are
        // invariant across the retried prepare.
        assert_eq!(second.route().positions(), first_positions);
        assert_eq!(second.route().ordered_experts(), first_ids);
        assert_eq!(second.route().ordered_weight_bits(), first_weights);
        assert_eq!(second.route().cache_version(), first_cache_version);
        let zero_experts = session.upload_f32(2, 32, &[0.0; 64]).unwrap();
        let output = second.finish(&zero_experts).unwrap();
        assert_eq!(cache.version(), 1);
        let expected: Vec<f32> = hidden_values
            .iter()
            .zip(initial_cache.iter())
            .map(|(hidden, cached)| hidden + cached)
            .collect();
        assert_eq!(output.read_f32().unwrap(), expected);
        assert_eq!(cache.read_f32().unwrap(), expected);
    }

    #[test]
    fn fixed_mailbox_materializes_without_reordering_or_requantizing() {
        let session = NativeProviderSession::synthetic_split(Device::Cpu, 1, 32, 32).unwrap();
        let hidden = session.upload_f32(1, 32, &[0.125; 32]).unwrap();
        let mut cache = session.create_cache_f32(1, 32, None).unwrap();
        let prepared = session.prepare_layer(&hidden, &mut cache, 0).unwrap();
        let route = prepared.route();
        let mut arena = crate::routing::RouteArena::new(1).unwrap();
        let view = arena
            .materialize(
                route.positions(),
                route.ordered_experts(),
                route.ordered_weight_bits(),
            )
            .unwrap();
        assert_eq!(view.ordered_experts, route.ordered_experts());
        assert_eq!(view.ordered_weight_bits, route.ordered_weight_bits());
        assert_eq!(view.unique_experts.len(), ROUTE_TOP_K);
    }

    #[test]
    fn resources_from_different_sessions_are_rejected_before_ffi() {
        let first = NativeProviderSession::synthetic_split(Device::Cpu, 1, 32, 32).unwrap();
        let second = NativeProviderSession::synthetic_split(Device::Cpu, 1, 32, 32).unwrap();
        let hidden = first.upload_f32(1, 32, &[0.0; 32]).unwrap();
        let mut foreign_cache = second.create_cache_f32(1, 32, None).unwrap();
        let error = first
            .prepare_layer(&hidden, &mut foreign_cache, 0)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("different native provider session")
        );
    }

    #[test]
    fn cpu_spine_binding_owns_exact_converted_values_after_buffers_drop() {
        let (_source, buffers, descriptors) = synthetic_spine_buffers();
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let report = session
            .bind_spine_layer(0, 1, &descriptors, &buffers, SpineLayerRetention::Transient)
            .unwrap();
        assert_eq!(report.tensor_count, 3);
        assert_eq!(report.quantized_tensor_count, 1);
        assert_eq!(report.raw_tensor_count, 2);
        assert_eq!(report.quantized_bytes, 4);
        assert_eq!(report.scales_bytes, 4);
        assert_eq!(report.other_bytes, 8);
        assert_eq!(report.resident_storage_bytes, 24);
        assert_eq!(report.retention, SpineLayerRetention::Transient);
        assert_eq!(report.source_use, SpineSourceUse::Detached);
        drop(buffers);

        let bf16 = session
            .read_spine_tensor_f32(0, 1, 1, SpineComponent::Data, 2)
            .unwrap();
        assert_eq!(bf16.stored_scalar, SpineStoredScalar::F32);
        assert_eq!(&*bf16.shape, &[2]);
        assert_eq!(&*bf16.values, &[1.0, -2.0]);
        let f32_weight = session
            .read_spine_tensor_f32(0, 1, 7, SpineComponent::Data, 1)
            .unwrap();
        assert_eq!(&*f32_weight.values, &[3.5]);
        let quantized = session
            .read_spine_tensor_f32(0, 1, 13, SpineComponent::Data, 4)
            .unwrap();
        assert_eq!(quantized.stored_scalar, SpineStoredScalar::I8);
        assert_eq!(&*quantized.shape, &[2, 2]);
        assert_eq!(&*quantized.values, &[-3.0, 0.0, 7.0, 127.0]);
        let scales = session
            .read_spine_tensor_f32(0, 1, 13, SpineComponent::Auxiliary, 2)
            .unwrap();
        assert_eq!(scales.stored_scalar, SpineStoredScalar::F32);
        assert_eq!(&*scales.shape, &[2]);
        assert_eq!(&*scales.values, &[0.5, -2.0]);

        // A target session with resident weights must still fail closed until
        // the complete full-K3 attention/MoE execution tape is linked.
        let hidden = session.upload_f32(1, 1, &[0.0]).unwrap();
        let mut cache = session.create_cache_f32(1, 1, None).unwrap();
        let error = session.prepare_layer(&hidden, &mut cache, 0).unwrap_err();
        assert!(
            error.to_string().contains("session contract")
                || error.to_string().contains("no loaded K3 layer tape")
        );
    }

    #[test]
    fn detached_provider_has_no_reclaimable_source_use_handles() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let unknown = SpineSourceUseToken {
            session_identity: session.identity(),
            generation: 1,
            handle: 1,
        };
        let error = session.seal_spine_source_use(unknown).unwrap_err();
        assert!(error.to_string().contains("stale or unknown"));
    }

    #[test]
    fn invalid_spine_descriptors_never_replace_the_previous_generation() {
        let (_source, buffers, descriptors) = synthetic_spine_buffers();
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        session
            .bind_spine_layer(
                0,
                10,
                &descriptors,
                &buffers,
                SpineLayerRetention::Transient,
            )
            .unwrap();

        let mut invalid_cases = Vec::new();
        let mut invalid = descriptors.clone();
        invalid[0].data_offset = 512;
        invalid_cases.push(invalid);
        let mut invalid = descriptors.clone();
        invalid[0].rank = 0;
        invalid_cases.push(invalid);
        let mut invalid = descriptors.clone();
        invalid[0].encoding = 99;
        invalid_cases.push(invalid);
        let mut invalid = descriptors.clone();
        invalid[1].slot = invalid[0].slot;
        invalid_cases.push(invalid);
        let mut invalid = descriptors.clone();
        invalid[0].reserved[2] = 1;
        invalid_cases.push(invalid);

        for (index, invalid) in invalid_cases.iter().enumerate() {
            let error = session
                .bind_spine_layer(
                    1,
                    11 + index as u64,
                    invalid,
                    &buffers,
                    SpineLayerRetention::Transient,
                )
                .unwrap_err();
            assert!(error.to_string().contains("bind spine layer"));
            let still_loaded = session
                .read_spine_tensor_f32(0, 10, 7, SpineComponent::Data, 1)
                .unwrap();
            assert_eq!(&*still_loaded.values, &[3.5]);
        }

        let replaced = session
            .bind_spine_layer(
                1,
                100,
                &descriptors,
                &buffers,
                SpineLayerRetention::Transient,
            )
            .unwrap();
        assert_eq!(replaced.layer_index, 1);
        assert_eq!(replaced.generation, 100);
        assert!(
            session
                .read_spine_tensor_f32(0, 10, 7, SpineComponent::Data, 1)
                .unwrap_err()
                .to_string()
                .contains("generation is stale")
        );
        assert_eq!(
            &*session
                .read_spine_tensor_f32(1, 100, 7, SpineComponent::Data, 1)
                .unwrap()
                .values,
            &[3.5]
        );
    }

    #[test]
    fn retained_spine_prefix_survives_transient_churn_and_rejects_replacement() {
        let (_source, buffers, descriptors) = synthetic_spine_buffers();
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let first = session
            .bind_spine_layer(0, 1, &descriptors, &buffers, SpineLayerRetention::Retained)
            .unwrap();
        assert_eq!(first.retention, SpineLayerRetention::Retained);
        assert_eq!(first.source_use, SpineSourceUse::Detached);
        assert_eq!(first.resident_storage_bytes, 24);

        session
            .bind_spine_layer(1, 2, &descriptors, &buffers, SpineLayerRetention::Transient)
            .unwrap();
        session
            .bind_spine_layer(2, 3, &descriptors, &buffers, SpineLayerRetention::Transient)
            .unwrap();
        assert_eq!(
            &*session
                .read_spine_tensor_f32(0, 1, 7, SpineComponent::Data, 1)
                .unwrap()
                .values,
            &[3.5]
        );
        assert!(
            session
                .read_spine_tensor_f32(1, 1, 7, SpineComponent::Data, 1)
                .unwrap_err()
                .to_string()
                .contains("layer/generation")
        );

        // Retained layers are immutable. A rejected duplicate must not consume
        // generation 4, so the next ordered-prefix append can use it exactly.
        assert!(
            session
                .bind_spine_layer(0, 4, &descriptors, &buffers, SpineLayerRetention::Retained,)
                .unwrap_err()
                .to_string()
                .contains("ordered prefix")
        );
        session
            .bind_spine_layer(1, 4, &descriptors, &buffers, SpineLayerRetention::Retained)
            .unwrap();

        assert!(
            session
                .bind_spine_layer(1, 5, &descriptors, &buffers, SpineLayerRetention::Transient,)
                .unwrap_err()
                .to_string()
                .contains("retained")
        );
        session
            .bind_spine_layer(3, 5, &descriptors, &buffers, SpineLayerRetention::Transient)
            .unwrap();
        assert_eq!(
            &*session
                .read_spine_tensor_f32(0, 1, 7, SpineComponent::Data, 1)
                .unwrap()
                .values,
            &[3.5]
        );
        assert_eq!(
            &*session
                .read_spine_tensor_f32(1, 4, 7, SpineComponent::Data, 1)
                .unwrap()
                .values,
            &[3.5]
        );
    }

    #[test]
    fn real_target_state_can_reset_repeatedly_without_rebuilding_the_session() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        let identity = session.identity();
        session.reset_target_state().unwrap();
        session.reset_target_state().unwrap();
        assert_eq!(session.identity(), identity);

        let synthetic = NativeProviderSession::synthetic_split(Device::Cpu, 1, 8, 16).unwrap();
        assert!(synthetic.reset_target_state().is_err());
    }

    #[test]
    fn target_state_branch_is_exclusive_and_drop_rolls_back_generation() {
        let session = NativeProviderSession::target(Device::Cpu).unwrap();
        session.reset_target_state().unwrap();
        let parent = session.inspect_target_state().unwrap();
        assert_eq!(parent.committed_positions, 0);

        let branch = session.begin_target_state_branch(parent).unwrap();
        assert!(session.begin_target_state_branch(parent).is_err());
        assert!(session.inspect_target_state().is_err());
        drop(branch);

        let restored = session.inspect_target_state().unwrap();
        assert_eq!(restored.committed_positions, parent.committed_positions);
        assert!(restored.cache_generation > parent.cache_generation);

        let branch = session.begin_target_state_branch(restored).unwrap();
        let published = session.publish_target_state_branch(branch).unwrap();
        assert_eq!(published, restored);
        assert_eq!(session.inspect_target_state().unwrap(), published);
    }
}
