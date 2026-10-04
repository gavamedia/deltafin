#include "provider_moe.h"

#include "provider_cuda_moe.h"
#include "../../tools/metal_moe_abi.h"

#if defined(__APPLE__) && defined(DELTAFIN_HAVE_MPS_ROUTE_MAILBOX_V1)
#include "provider_route_mailbox.h"
#endif

#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/add.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/div.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/topk.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(DELTAFIN_HAVE_MXFP4_CPU_V1)
extern "C" {
void mxfp4_moe_layer(const std::uint8_t* const* packed,
                     const std::uint8_t* const* scales, const int* rows,
                     const int* columns, int matrix_count, const float* input,
                     float* output, int threads);
void mxfp4_gemv_batch(const std::uint8_t* const* packed,
                      const std::uint8_t* const* scales,
                      const float* const* inputs, float* const* outputs,
                      const int* rows, const int* columns, int matrix_count,
                      int threads);
int mxfp4_cpu_compatible(void);
}
#endif

#if defined(__APPLE__)
#include <dlfcn.h>
#if defined(DELTAFIN_HAVE_METAL_MOE_V1)
extern "C" {
int k3_metal_init(const char* source_path);
int k3_metal_moe_layer(const std::uint8_t* const* expert_blobs,
                       int expert_count, const float* weights,
                       const float* input, float* output);
int k3_metal_moe_positions(const std::uint8_t* const* expert_blobs,
                           int edge_count, const int* position_offsets,
                           int position_count, const float* weights,
                           const float* input, float* output);
int k3_metal_moe_positions_flat(const std::uint8_t* const* expert_blobs,
                                int edge_count,
                                const int* position_offsets,
                                int position_count, const float* weights,
                                const float* input, float* output);
int k3_metal_moe_positions_flat_desc_v1(
    const K3MetalExpertDescriptorV1* experts, int edge_count,
    const int* position_offsets, int position_count, const float* weights,
    const float* input, float* output);
void k3_metal_drop(const std::uint8_t* blob);
void k3_metal_flush(void);
void k3_metal_stats(long long* five);
const char* k3_metal_last_error(void);
}
#endif
#endif

namespace deltafin::provider_internal {
namespace {

constexpr float kSituBeta = 4.0F;
constexpr float kSituLinearBeta = 25.0F;
constexpr float kRmsEpsilon = 1.0e-5F;
constexpr std::uint32_t kFirstMoeLayer = 1;
constexpr std::uint32_t kLastMoeLayer = 92;
constexpr std::int64_t kMaximumPreparedPositions = 64;

void record_trace(MoeExecutionTrace* trace, const MoeExecutionStage stage) {
  if (trace != nullptr) {
    trace->record(stage);
  }
}

struct ExpertOffsets {
  std::uint64_t w1_packed = 0;
  std::uint64_t w1_scales = 0;
  std::uint64_t w2_packed = 0;
  std::uint64_t w2_scales = 0;
  std::uint64_t w3_packed = 0;
  std::uint64_t w3_scales = 0;
  std::uint64_t span = 0;
};

#if defined(__APPLE__)
struct MetalApi {
  using Init = int (*)(const char*);
  using Layer = int (*)(const std::uint8_t* const*, int, const float*,
                        const float*, float*);
  using Positions = int (*)(const std::uint8_t* const*, int, const int*, int,
                            const float*, const float*, float*);
  using DescriptorAbi = std::uint32_t (*)();
  using LayoutCapabilities = std::uint64_t (*)();
  using LayerDescriptor = int (*)(const K3MetalExpertDescriptorV1*, int,
                                  const float*, const float*, float*);
  using PositionsDescriptor = int (*)(const K3MetalExpertDescriptorV1*, int,
                                      const int*, int, const float*,
                                      const float*, float*);
  using Drop = void (*)(const std::uint8_t*);
  using Flush = void (*)();
  using Stats = void (*)(long long*);
  using LastError = const char* (*)();
  using StageBegin = std::uint64_t (*)(int, const float*);
  using StageEdges = int (*)(std::uint64_t, const K3MetalExpertDescriptorV1*,
                             const int*, int);
  using StageFinish = int (*)(std::uint64_t, const K3MetalExpertDescriptorV1*,
                              int, const float*, const float*, float*);
  using StageAbandon = void (*)(std::uint64_t);
  using StageStats = void (*)(long long*);
  Init init = nullptr;
  Layer layer = nullptr;
  Positions positions = nullptr;
  Positions positions_flat = nullptr;
  DescriptorAbi descriptor_abi = nullptr;
  LayoutCapabilities layout_capabilities = nullptr;
  LayerDescriptor layer_descriptor = nullptr;
  PositionsDescriptor positions_descriptor = nullptr;
  PositionsDescriptor positions_flat_descriptor = nullptr;
  Drop drop = nullptr;
  Flush flush = nullptr;
  Stats stats = nullptr;
  LastError last_error = nullptr;
  StageBegin stage_begin = nullptr;
  StageEdges stage_edges = nullptr;
  StageFinish stage_finish = nullptr;
  StageAbandon stage_abandon = nullptr;
  StageStats stage_stats = nullptr;
};

const MetalApi& metal_api() {
#if defined(DELTAFIN_HAVE_METAL_MOE_V1)
  // Direct references force the established Objective-C++ bridge object out
  // of the provider's static archive when Rust links the one executable.
  static const MetalApi api{
      &k3_metal_init,
      &k3_metal_moe_layer,
      &k3_metal_moe_positions,
      &k3_metal_moe_positions_flat,
      &k3_metal_descriptor_abi_version,
      &k3_metal_layout_capabilities,
      &k3_metal_moe_layer_desc_v1,
      &k3_metal_moe_positions_desc_v1,
      &k3_metal_moe_positions_flat_desc_v1,
      &k3_metal_drop,
      &k3_metal_flush,
      &k3_metal_stats,
      &k3_metal_last_error,
      &k3_metal_moe_stage_begin_v1,
      &k3_metal_moe_stage_edges_v1,
      &k3_metal_moe_stage_finish_v1,
      &k3_metal_moe_stage_abandon_v1,
      &k3_metal_moe_stage_stats_v1};
#else
  // Resolving from the process image avoids a required symbol on Linux and in
  // CPU-only macOS builds. A production Apple link that includes the reviewed
  // Objective-C++ bridge exposes these four C symbols from the same binary.
  static const MetalApi api{
      reinterpret_cast<MetalApi::Init>(dlsym(RTLD_DEFAULT, "k3_metal_init")),
      reinterpret_cast<MetalApi::Layer>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_layer")),
      reinterpret_cast<MetalApi::Positions>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_positions")),
      reinterpret_cast<MetalApi::Positions>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_positions_flat")),
      reinterpret_cast<MetalApi::DescriptorAbi>(
          dlsym(RTLD_DEFAULT, "k3_metal_descriptor_abi_version")),
      reinterpret_cast<MetalApi::LayoutCapabilities>(
          dlsym(RTLD_DEFAULT, "k3_metal_layout_capabilities")),
      reinterpret_cast<MetalApi::LayerDescriptor>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_layer_desc_v1")),
      reinterpret_cast<MetalApi::PositionsDescriptor>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_positions_desc_v1")),
      reinterpret_cast<MetalApi::PositionsDescriptor>(dlsym(
          RTLD_DEFAULT, "k3_metal_moe_positions_flat_desc_v1")),
      reinterpret_cast<MetalApi::Drop>(dlsym(RTLD_DEFAULT, "k3_metal_drop")),
      reinterpret_cast<MetalApi::Flush>(dlsym(RTLD_DEFAULT, "k3_metal_flush")),
      reinterpret_cast<MetalApi::Stats>(dlsym(RTLD_DEFAULT, "k3_metal_stats")),
      reinterpret_cast<MetalApi::LastError>(
          dlsym(RTLD_DEFAULT, "k3_metal_last_error")),
      reinterpret_cast<MetalApi::StageBegin>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_stage_begin_v1")),
      reinterpret_cast<MetalApi::StageEdges>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_stage_edges_v1")),
      reinterpret_cast<MetalApi::StageFinish>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_stage_finish_v1")),
      reinterpret_cast<MetalApi::StageAbandon>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_stage_abandon_v1")),
      reinterpret_cast<MetalApi::StageStats>(
          dlsym(RTLD_DEFAULT, "k3_metal_moe_stage_stats_v1"))};
#endif
  return api;
}
#endif

std::uint64_t checked_mul(const std::uint64_t left,
                          const std::uint64_t right, const char* name) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::invalid_argument(std::string(name) + " overflows uint64");
  }
  return left * right;
}

std::uint64_t checked_add(const std::uint64_t left,
                          const std::uint64_t right, const char* name) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::invalid_argument(std::string(name) + " overflows uint64");
  }
  return left + right;
}

std::uint64_t matrix_packed_bytes(const std::uint64_t rows,
                                  const std::uint64_t columns) {
  return checked_mul(rows, columns, "MXFP4 packed matrix elements") / 2;
}

std::uint64_t matrix_scale_bytes(const std::uint64_t rows,
                                 const std::uint64_t columns) {
  return checked_mul(rows, columns / 32, "MXFP4 scale matrix elements");
}

ExpertOffsets expert_offsets(const MoeGeometry& geometry) {
  if (geometry.routed_hidden == 0 || geometry.intermediate == 0 ||
      geometry.routed_hidden % 32 != 0 || geometry.intermediate % 32 != 0) {
    throw std::invalid_argument(
        "MXFP4 MoE dimensions must be positive multiples of 32");
  }
  const std::uint64_t w13_packed =
      matrix_packed_bytes(geometry.intermediate, geometry.routed_hidden);
  const std::uint64_t w13_scales =
      matrix_scale_bytes(geometry.intermediate, geometry.routed_hidden);
  const std::uint64_t w2_packed =
      matrix_packed_bytes(geometry.routed_hidden, geometry.intermediate);
  const std::uint64_t w2_scales =
      matrix_scale_bytes(geometry.routed_hidden, geometry.intermediate);
  ExpertOffsets offsets;
  offsets.w1_packed = 0;
  offsets.w1_scales = w13_packed;
  offsets.w2_packed = checked_add(w13_packed, w13_scales, "w2 offset");
  offsets.w2_scales =
      checked_add(offsets.w2_packed, w2_packed, "w2 scale offset");
  offsets.w3_packed =
      checked_add(offsets.w2_scales, w2_scales, "w3 offset");
  offsets.w3_scales =
      checked_add(offsets.w3_packed, w13_packed, "w3 scale offset");
  offsets.span = checked_add(offsets.w3_scales, w13_scales, "expert span");
  return offsets;
}

bool exact_k3_geometry(const MoeGeometry& geometry) {
  const MoeGeometry expected = k3_moe_geometry();
  return geometry.hidden == expected.hidden &&
      geometry.routed_hidden == expected.routed_hidden &&
      geometry.intermediate == expected.intermediate &&
      geometry.experts == expected.experts &&
      geometry.shared_intermediate == expected.shared_intermediate;
}

std::uint64_t required_expert_span(const MoeExpertLayout layout,
                                   const MoeGeometry& geometry) {
  switch (layout) {
    case MoeExpertLayout::RawV1:
      return expert_offsets(geometry).span;
    case MoeExpertLayout::Scale4V2:
      if (!exact_k3_geometry(geometry)) {
        throw std::invalid_argument(
            "scale4-v2 expert storage accepts only exact K3 geometry");
      }
      return K3_SCALE4_V2_EXPERT_SPAN;
  }
  throw std::invalid_argument("expert storage layout is unknown");
}

void validate_geometry(const MoeGeometry& geometry) {
  if (geometry.hidden == 0 || geometry.experts < kMoeRouteTopK ||
      geometry.shared_intermediate == 0) {
    throw std::invalid_argument("MoE geometry has an empty or undersized field");
  }
  static_cast<void>(expert_offsets(geometry));
#if !defined(DELTAFIN_PROVIDER_MOE_TESTING)
  const MoeGeometry expected = k3_moe_geometry();
  if (geometry.hidden != expected.hidden ||
      geometry.routed_hidden != expected.routed_hidden ||
      geometry.intermediate != expected.intermediate ||
      geometry.experts != expected.experts ||
      geometry.shared_intermediate != expected.shared_intermediate) {
    throw std::invalid_argument(
        "production routed-MoE provider accepts only the exact K3 geometry");
  }
#endif
}

void require_f32_tensor(const at::Tensor& tensor,
                        const at::IntArrayRef shape,
                        const at::Device& device, const char* name) {
  if (!tensor.defined() || tensor.scalar_type() != at::kFloat ||
      tensor.device() != device || !tensor.is_contiguous() ||
      tensor.sizes() != shape) {
    throw std::invalid_argument(std::string(name) +
                                " violates its contiguous fp32 shape/device contract");
  }
}

void require_row_int8_matrix(const MoeRowInt8Matrix& matrix,
                             const std::int64_t rows,
                             const std::int64_t columns,
                             const at::Device& device,
                             const bool packed_int8_qualified,
                             const char* name) {
  const bool has_q8 = matrix.quantized.defined() || matrix.row_scales.defined();
  const bool has_dense = matrix.dense_f32.defined();
  const bool has_original = matrix.original_bf16.defined();
  if (has_q8) {
    if (!matrix.quantized.defined() ||
        matrix.quantized.scalar_type() != at::kChar ||
        matrix.quantized.device() != device ||
        !matrix.quantized.is_contiguous() ||
        matrix.quantized.sizes() != at::IntArrayRef({rows, columns}) ||
        !matrix.row_scales.defined() ||
        matrix.row_scales.scalar_type() != at::kFloat ||
        matrix.row_scales.device() != device ||
        !matrix.row_scales.is_contiguous() ||
        matrix.row_scales.sizes() != at::IntArrayRef({rows})) {
      throw std::invalid_argument(
          std::string(name) + " violates its row-int8 shape/device contract");
    }
  }
  if (has_dense) {
    require_f32_tensor(matrix.dense_f32, {rows, columns}, device,
                       (std::string(name) + " dense weight").c_str());
  }
  if (has_original &&
      !original_bf16_matrix_matches(
          matrix.original_bf16, device, static_cast<std::size_t>(rows),
          static_cast<std::size_t>(columns))) {
    throw std::invalid_argument(std::string(name) +
                                " has an invalid original-BF16 carrier");
  }
  if ((packed_int8_qualified && !has_q8) ||
      (!packed_int8_qualified && (has_dense == has_original)) ||
      (packed_int8_qualified && (has_dense || has_original))) {
    throw std::invalid_argument(
        std::string(name) +
        " does not provide the selected row-int8 or original-BF16 weight");
  }
}

std::optional<MoeRowInt8Matrix>
adjacent_shared_gate_up(const MoeSpineT1& spine) {
  const MoeRowInt8Matrix& gate = spine.shared_gate;
  const MoeRowInt8Matrix& up = spine.shared_up;
  if (gate.original_bf16.defined() || up.original_bf16.defined()) {
    if (!gate.original_bf16.defined() || !up.original_bf16.defined() ||
        gate.quantized.defined() || gate.row_scales.defined() ||
        gate.dense_f32.defined() || up.quantized.defined() ||
        up.row_scales.defined() || up.dense_f32.defined()) {
      return std::nullopt;
    }
    const std::array<const OriginalBf16Matrix*, 2> matrices{
        &gate.original_bf16, &up.original_bf16};
    auto combined = adjacent_original_bf16_matrices(matrices);
    if (!combined.has_value()) {
      return std::nullopt;
    }
    return MoeRowInt8Matrix{at::Tensor(), at::Tensor(), at::Tensor(),
                            std::move(*combined)};
  }
  if (gate.dense_f32.defined() || up.dense_f32.defined()) {
    if (!gate.dense_f32.defined() || !up.dense_f32.defined() ||
        gate.quantized.defined() || gate.row_scales.defined() ||
        up.quantized.defined() || up.row_scales.defined() ||
        !gate.dense_f32.is_contiguous() || !up.dense_f32.is_contiguous() ||
        gate.dense_f32.dim() != 2 ||
        gate.dense_f32.sizes() != up.dense_f32.sizes() ||
        !gate.dense_f32.is_alias_of(up.dense_f32) ||
        up.dense_f32.storage_offset() !=
            gate.dense_f32.storage_offset() + gate.dense_f32.numel()) {
      return std::nullopt;
    }
    const std::int64_t rows = gate.dense_f32.size(0);
    const std::int64_t columns = gate.dense_f32.size(1);
    if (rows <= 0 || columns <= 0 ||
        rows > std::numeric_limits<std::int64_t>::max() / 2) {
      return std::nullopt;
    }
    return MoeRowInt8Matrix{
        at::Tensor(), at::Tensor(),
        gate.dense_f32.as_strided(
            {rows * 2, columns}, {columns, 1},
            gate.dense_f32.storage_offset()),
        {}};
  }
  if (!gate.quantized.defined() || !gate.row_scales.defined() ||
      !up.quantized.defined() || !up.row_scales.defined() ||
      !gate.quantized.is_contiguous() || !gate.row_scales.is_contiguous() ||
      !up.quantized.is_contiguous() || !up.row_scales.is_contiguous() ||
      gate.quantized.dim() != 2 || gate.row_scales.dim() != 1 ||
      gate.quantized.sizes() != up.quantized.sizes() ||
      gate.row_scales.sizes() != up.row_scales.sizes() ||
      !gate.quantized.is_alias_of(up.quantized) ||
      !gate.row_scales.is_alias_of(up.row_scales) ||
      up.quantized.storage_offset() !=
          gate.quantized.storage_offset() + gate.quantized.numel() ||
      up.row_scales.storage_offset() !=
          gate.row_scales.storage_offset() + gate.row_scales.numel()) {
    return std::nullopt;
  }
  const std::int64_t rows = gate.quantized.size(0);
  const std::int64_t columns = gate.quantized.size(1);
  if (rows <= 0 || columns <= 0 ||
      rows > std::numeric_limits<std::int64_t>::max() / 2) {
    return std::nullopt;
  }
  return MoeRowInt8Matrix{
      gate.quantized.as_strided({rows * 2, columns}, {columns, 1},
                                gate.quantized.storage_offset()),
      gate.row_scales.as_strided({rows * 2}, {1},
                                 gate.row_scales.storage_offset()),
      at::Tensor(),
      {},
  };
}

void validate_spine(const MoeSpineT1& spine, const at::Device& device) {
  validate_geometry(spine.geometry);
  if (spine.layer_index < kFirstMoeLayer ||
      spine.layer_index > kLastMoeLayer || spine.generation == 0) {
    throw std::invalid_argument(
        "MoE spine layer/generation is outside K3's loaded-layer contract");
  }
  const auto hidden = static_cast<std::int64_t>(spine.geometry.hidden);
  const auto routed = static_cast<std::int64_t>(spine.geometry.routed_hidden);
  const auto experts = static_cast<std::int64_t>(spine.geometry.experts);
  const auto shared =
      static_cast<std::int64_t>(spine.geometry.shared_intermediate);
  require_row_int8_matrix(spine.router, experts, hidden, device,
                          spine.packed_int8_qualified, "MoE router");
  require_f32_tensor(spine.router_correction_bias, {experts}, device,
                     "MoE router correction bias");
  require_row_int8_matrix(spine.routed_down, routed, hidden, device,
                          spine.packed_int8_qualified,
                          "MoE routed down projection");
  require_f32_tensor(spine.routed_norm, {routed}, device,
                     "MoE routed RMS norm");
  require_row_int8_matrix(spine.routed_up, hidden, routed, device,
                          spine.packed_int8_qualified,
                          "MoE routed up projection");
  require_row_int8_matrix(spine.shared_gate, shared, hidden, device,
                          spine.packed_int8_qualified,
                          "MoE shared gate projection");
  require_row_int8_matrix(spine.shared_up, shared, hidden, device,
                          spine.packed_int8_qualified,
                          "MoE shared up projection");
  require_row_int8_matrix(spine.shared_down, hidden, shared, device,
                          spine.packed_int8_qualified,
                          "MoE shared down projection");
  if (spine.shared_gate_up_qualified) {
    require_row_int8_matrix(spine.shared_gate_up, shared * 2, hidden, device,
                            spine.packed_int8_qualified,
                            "MoE shared gate/up bundle");
    const auto adjacent = adjacent_shared_gate_up(spine);
    const bool matching_q8 = adjacent.has_value() &&
        spine.packed_int8_qualified &&
        adjacent->quantized.is_alias_of(spine.shared_gate_up.quantized) &&
        adjacent->quantized.storage_offset() ==
            spine.shared_gate_up.quantized.storage_offset() &&
        adjacent->row_scales.storage_offset() ==
            spine.shared_gate_up.row_scales.storage_offset();
    const bool matching_original = adjacent.has_value() &&
        !spine.packed_int8_qualified &&
        adjacent->original_bf16.owned_storage ==
            spine.shared_gate_up.original_bf16.owned_storage &&
        adjacent->original_bf16.owned_element_offset ==
            spine.shared_gate_up.original_bf16.owned_element_offset &&
        adjacent->original_bf16.borrowed_cpu_data ==
            spine.shared_gate_up.original_bf16.borrowed_cpu_data;
    const bool matching_dense = adjacent.has_value() &&
        !spine.packed_int8_qualified &&
        adjacent->dense_f32.defined() &&
        spine.shared_gate_up.dense_f32.defined() &&
        adjacent->dense_f32.is_alias_of(spine.shared_gate_up.dense_f32) &&
        adjacent->dense_f32.storage_offset() ==
            spine.shared_gate_up.dense_f32.storage_offset();
    if (!matching_q8 && !matching_original && !matching_dense) {
      throw std::invalid_argument(
          "MoE shared gate/up bundle is not its zero-copy adjacent super-view");
    }
  } else if (spine.shared_gate_up.quantized.defined() ||
             spine.shared_gate_up.row_scales.defined() ||
             spine.shared_gate_up.dense_f32.defined() ||
             spine.shared_gate_up.original_bf16.defined()) {
    throw std::invalid_argument(
        "MoE shared gate/up payload was supplied without qualification");
  }
  if (spine.shared_gate_up_enabled && !spine.shared_gate_up_qualified) {
    throw std::invalid_argument(
        "MoE shared gate/up experiment was enabled without qualification");
  }
}

at::Tensor materialize_row_int8(const MoeRowInt8Matrix& matrix) {
  if (matrix.dense_f32.defined()) {
    return matrix.dense_f32;
  }
  if (matrix.original_bf16.defined()) {
    return materialize_original_bf16_f32(matrix.original_bf16);
  }
  return at::mul(matrix.quantized.to(at::kFloat),
                 matrix.row_scales.unsqueeze(1));
}

at::Tensor row_int8_linear(const at::Tensor& input,
                           const MoeRowInt8Matrix& matrix,
                           const bool packed_int8_qualified) {
  if (packed_int8_qualified) {
    // Qualification happens once at session/device admission. A provider that
    // later rejects this exact call must fail the layer transaction; silently
    // changing arithmetic after routing would make diagnosis impossible.
    return at::_weight_int8pack_mm(input, matrix.quantized,
                                   matrix.row_scales);
  }
  if (matrix.original_bf16.defined()) {
    return original_bf16_linear(input, matrix.original_bf16);
  }
  const at::Tensor dense = materialize_row_int8(matrix);
  return at::linear(input, dense, std::nullopt);
}

at::Tensor situ(const at::Tensor& gate, const at::Tensor& up) {
  // Keep the operation order identical to Kimi's SituAndMul.forward. Every
  // operand and intermediate is fp32; no fast-math approximation is enabled.
  const at::Tensor bounded_gate = at::mul(
      at::mul(at::tanh(at::div(gate, kSituBeta)), kSituBeta),
      at::sigmoid(gate));
  const at::Tensor bounded_up =
      at::mul(at::tanh(at::div(up, kSituLinearBeta)), kSituLinearBeta);
  return at::mul(bounded_gate, bounded_up);
}

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight) {
  const at::Tensor input_f32 = input.to(at::kFloat);
  const at::Tensor variance = at::mean(
      at::pow(input_f32, 2), std::vector<std::int64_t>{-1}, true);
  const at::Tensor normalized = at::mul(
      input_f32, at::rsqrt(at::add(variance, kRmsEpsilon)));
  return at::mul(weight, normalized);
}

at::Tensor shared_expert(const at::Tensor& identity,
                         const MoeSpineT1& spine) {
  at::Tensor shared_gate;
  at::Tensor shared_up;
  const MoeRowInt8Matrix* bundle = nullptr;
  if (spine.shared_gate_up_enabled) {
    bundle = &spine.shared_gate_up;
  }
  if (bundle != nullptr) {
    const at::Tensor gate_up = row_int8_linear(
        identity, *bundle, spine.packed_int8_qualified);
    const std::int64_t rows =
        static_cast<std::int64_t>(spine.geometry.shared_intermediate);
    shared_gate = gate_up.narrow(1, 0, rows);
    shared_up = gate_up.narrow(1, rows, rows);
  } else {
    shared_gate = row_int8_linear(
        identity, spine.shared_gate, spine.packed_int8_qualified);
    shared_up = row_int8_linear(
        identity, spine.shared_up, spine.packed_int8_qualified);
  }
  const at::Tensor shared_hidden = situ(shared_gate, shared_up);
  return row_int8_linear(shared_hidden, spine.shared_down,
                         spine.packed_int8_qualified);
}

void validate_route(const MoeRouteT1& route, const MoeGeometry& geometry) {
  for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
    const std::uint16_t expert = route.expert_ids[edge];
    if (expert >= geometry.experts) {
      throw std::invalid_argument("routed-MoE route contains an unknown expert");
    }
    if (std::find(route.expert_ids.begin(), route.expert_ids.begin() + edge,
                  expert) != route.expert_ids.begin() + edge) {
      throw std::invalid_argument("routed-MoE T=1 route repeats an expert");
    }
    const float weight = std::bit_cast<float>(route.weight_bits[edge]);
    if (!std::isfinite(weight) || weight < 0.0F) {
      throw std::invalid_argument(
          "routed-MoE route contains a non-finite or negative fp32 weight");
    }
  }
}

MoeRouteT1 materialize_route_t1(const at::Tensor& expert_ids,
                                const at::Tensor& weights,
                                const MoeGeometry& geometry) {
  MoeRouteT1 route;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_MPS_ROUTE_MAILBOX_V1)
  if (expert_ids.device().is_mps()) {
    static_assert(kRouteMailboxTopK == kMoeRouteTopK);
    RouteMailboxT1 snapshot;
    if (try_materialize_mps_route_t1(expert_ids, weights, snapshot)) {
      for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
        const std::int64_t expert = snapshot.expert_ids[edge];
        if (expert < 0 ||
            expert >= static_cast<std::int64_t>(geometry.experts)) {
          throw std::runtime_error(
              "provider router produced an invalid expert ID");
        }
        route.expert_ids[edge] = static_cast<std::uint16_t>(expert);
        route.weight_bits[edge] = snapshot.weight_bits[edge];
      }
      validate_route(route, geometry);
      return route;
    }
  }
#endif

  // Authoritative fallback on CPU, CUDA, non-Apple builds, nonqualifying
  // tensors, and every mailbox setup/runtime failure.
  const at::Tensor ids_cpu = expert_ids.to(at::kCPU).contiguous();
  const at::Tensor weights_cpu = weights.to(at::kCPU).contiguous();
  const auto* ids = ids_cpu.const_data_ptr<std::int64_t>();
  const auto* values = weights_cpu.const_data_ptr<float>();
  for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
    if (ids[edge] < 0 ||
        ids[edge] >= static_cast<std::int64_t>(geometry.experts)) {
      throw std::runtime_error("provider router produced an invalid expert ID");
    }
    route.expert_ids[edge] = static_cast<std::uint16_t>(ids[edge]);
    route.weight_bits[edge] = std::bit_cast<std::uint32_t>(values[edge]);
  }
  validate_route(route, geometry);
  return route;
}

std::vector<MoeRouteT1> materialize_route_positions(
    const at::Tensor& expert_ids, const at::Tensor& weights,
    const MoeGeometry& geometry) {
  if (!expert_ids.defined() || expert_ids.scalar_type() != at::kLong ||
      !expert_ids.is_contiguous() || expert_ids.dim() != 2 ||
      expert_ids.size(0) < 2 ||
      expert_ids.size(0) > kMaximumPreparedPositions ||
      expert_ids.size(1) != static_cast<std::int64_t>(kMoeRouteTopK) ||
      !weights.defined() || weights.scalar_type() != at::kFloat ||
      !weights.is_contiguous() || weights.sizes() != expert_ids.sizes() ||
      weights.device() != expert_ids.device()) {
    throw std::invalid_argument(
        "routed-MoE position routes require matching contiguous [2..64,16] IDs and fp32 weights");
  }

  const std::int64_t positions = expert_ids.size(0);
  std::vector<MoeRouteT1> routes(static_cast<std::size_t>(positions));
  if (expert_ids.device().is_cpu()) {
    const auto* ids = expert_ids.const_data_ptr<std::int64_t>();
    const auto* values = weights.const_data_ptr<float>();
    for (std::int64_t row = 0; row < positions; ++row) {
      MoeRouteT1& route = routes[static_cast<std::size_t>(row)];
      for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
        const std::size_t index =
            static_cast<std::size_t>(row) * kMoeRouteTopK + edge;
        if (ids[index] < 0 ||
            ids[index] >= static_cast<std::int64_t>(geometry.experts)) {
          throw std::runtime_error(
              "provider router produced an invalid expert ID");
        }
        route.expert_ids[edge] = static_cast<std::uint16_t>(ids[index]);
        route.weight_bits[edge] =
            std::bit_cast<std::uint32_t>(values[index]);
      }
      validate_route(route, geometry);
    }
    return routes;
  }

  /*
   * IDs are bounded by K3's 896 experts, so their fp32 representation is
   * exact. Packing IDs beside the already-fp32 weights creates one ordered
   * device-to-host boundary for the entire position set instead of two
   * transfers for every row. Weight bits are copied without conversion.
   */
  const at::Tensor packed =
      at::cat({expert_ids.to(at::kFloat), weights}, 1)
          .to(at::kCPU, at::kFloat)
          .contiguous();
  const auto* values = packed.const_data_ptr<float>();
  const std::size_t packed_width = kMoeRouteTopK * 2;
  for (std::int64_t row = 0; row < positions; ++row) {
    MoeRouteT1& route = routes[static_cast<std::size_t>(row)];
    const std::size_t base = static_cast<std::size_t>(row) * packed_width;
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      const float raw_id = values[base + edge];
      if (!std::isfinite(raw_id) || std::trunc(raw_id) != raw_id ||
          raw_id < 0.0F ||
          raw_id >= static_cast<float>(geometry.experts)) {
        throw std::runtime_error(
            "provider router produced an invalid expert ID");
      }
      route.expert_ids[edge] = static_cast<std::uint16_t>(raw_id);
      route.weight_bits[edge] = std::bit_cast<std::uint32_t>(
          values[base + kMoeRouteTopK + edge]);
    }
    validate_route(route, geometry);
  }
  return routes;
}

void validate_expert_storage(const std::span<const std::uint16_t> expert_ids,
                             const std::span<const std::uint8_t> bytes,
                             const MoeExpertLayout layout,
                             const std::uint64_t expert_span_bytes,
                             const MoeGeometry& geometry,
                             const std::size_t maximum_experts) {
  if (expert_ids.empty() || expert_ids.size() > maximum_experts) {
    throw std::invalid_argument(
        "expert tile is empty or exceeds its bounded unique-expert limit");
  }
  for (std::size_t index = 0; index < expert_ids.size(); ++index) {
    if (expert_ids[index] >= geometry.experts ||
        (index != 0 && expert_ids[index - 1] >= expert_ids[index])) {
      throw std::invalid_argument(
          "expert batch IDs must be unique, in range, and strictly ascending");
    }
  }
  const std::uint64_t span = required_expert_span(layout, geometry);
  if (expert_span_bytes != span) {
    throw std::invalid_argument(
        "expert span does not match its explicit storage layout");
  }
  const std::uint64_t expected_bytes = checked_mul(
      span, expert_ids.size(), "expert-major batch byte length");
  if (expected_bytes > std::numeric_limits<std::size_t>::max() ||
      bytes.size() != static_cast<std::size_t>(expected_bytes)) {
    throw std::invalid_argument(
        "expert-major buffer length does not match its canonical IDs");
  }
}

void validate_expert_span_pointers(
    const std::span<const std::uint16_t> expert_ids,
    const std::span<const std::uint8_t* const> expert_span_pointers,
    const MoeExpertLayout layout, const std::uint64_t expert_span_bytes,
    const MoeGeometry& geometry, const std::size_t maximum_experts) {
  if (expert_ids.empty() || expert_ids.size() > maximum_experts ||
      expert_span_pointers.size() != expert_ids.size()) {
    throw std::invalid_argument(
        "scattered expert tile has invalid ID/pointer bounds");
  }
  const std::uint64_t span = required_expert_span(layout, geometry);
  if (expert_span_bytes != span ||
      expert_span_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(
        "scattered expert span does not match its explicit storage layout");
  }
  for (std::size_t index = 0; index < expert_ids.size(); ++index) {
    if (expert_ids[index] >= geometry.experts ||
        (index != 0 && expert_ids[index - 1] >= expert_ids[index]) ||
        expert_span_pointers[index] == nullptr) {
      throw std::invalid_argument(
          "scattered expert IDs/pointers must be non-null, unique, in range, and canonical ascending");
    }
    if (std::find(expert_span_pointers.begin(),
                  expert_span_pointers.begin() +
                      static_cast<std::ptrdiff_t>(index),
                  expert_span_pointers[index]) !=
        expert_span_pointers.begin() +
            static_cast<std::ptrdiff_t>(index)) {
      throw std::invalid_argument(
          "scattered expert pointer slots must name distinct spans");
    }
  }
}

void validate_planned_cuda_misses(
    const CanonicalExpertPositionTileT1& experts,
    const MoeGeometry& geometry) {
  if (experts.layout != MoeExpertLayout::RawV1 ||
      experts.expert_span_bytes != geometry.expert_span_bytes() ||
      experts.expert_ids.size() > kMoePositionTileMaxExperts) {
    throw std::invalid_argument(
        "planned CUDA misses require bounded exact raw-v1 expert spans");
  }
  for (std::size_t index = 0; index < experts.expert_ids.size(); ++index) {
    if (experts.expert_ids[index] >= geometry.experts ||
        (index != 0 && experts.expert_ids[index - 1] >=
                           experts.expert_ids[index])) {
      throw std::invalid_argument(
          "planned CUDA miss IDs must be unique canonical ascending IDs");
    }
  }
  const std::uint64_t expected = checked_mul(
      experts.expert_span_bytes, experts.expert_ids.size(),
      "planned CUDA miss byte length");
  if (expected > std::numeric_limits<std::size_t>::max() ||
      experts.expert_major_bytes.size() != static_cast<std::size_t>(expected)) {
    throw std::invalid_argument(
        "planned CUDA miss bytes do not match their canonical IDs");
  }
}

std::array<const std::uint8_t*, kMoeRouteTopK> ordered_expert_pointers(
    const MoeRouteT1& route,
    const std::span<const std::uint16_t> expert_ids,
    const std::span<const std::uint8_t> bytes,
    const MoeGeometry& geometry, const std::uint64_t expert_span_bytes) {
  validate_route(route, geometry);

  std::array<const std::uint8_t*, kMoeRouteTopK> pointers = {};
  if (expert_span_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("expert span exceeds host size_t");
  }
  const std::size_t span = static_cast<std::size_t>(expert_span_bytes);
  for (std::size_t edge = 0; edge < route.expert_ids.size(); ++edge) {
    const std::uint16_t expert = route.expert_ids[edge];
    const auto found =
        std::lower_bound(expert_ids.begin(), expert_ids.end(), expert);
    if (found == expert_ids.end() || *found != expert) {
      throw std::invalid_argument(
          "expert-major batch is missing one of the ordered route experts");
    }
    const std::size_t slot =
        static_cast<std::size_t>(found - expert_ids.begin());
    pointers[edge] = bytes.data() + slot * span;
  }
  return pointers;
}

std::array<const std::uint8_t*, kMoeRouteTopK> ordered_expert_pointers(
    const MoeRouteT1& route,
    const std::span<const std::uint16_t> expert_ids,
    const std::span<const std::uint8_t* const> expert_span_pointers,
    const MoeGeometry& geometry) {
  validate_route(route, geometry);
  std::array<const std::uint8_t*, kMoeRouteTopK> pointers = {};
  for (std::size_t edge = 0; edge < route.expert_ids.size(); ++edge) {
    const std::uint16_t expert = route.expert_ids[edge];
    const auto found =
        std::lower_bound(expert_ids.begin(), expert_ids.end(), expert);
    if (found == expert_ids.end() || *found != expert) {
      throw std::invalid_argument(
          "scattered expert tile is missing one ordered route expert");
    }
    const std::size_t slot =
        static_cast<std::size_t>(found - expert_ids.begin());
    pointers[edge] = expert_span_pointers[slot];
  }
  return pointers;
}

void validate_backend_layout(const MoeExpertBackend backend,
                             const MoeExpertLayout layout) {
  switch (layout) {
    case MoeExpertLayout::RawV1:
      return;
    case MoeExpertLayout::Scale4V2:
      if (backend != MoeExpertBackend::MetalMxfp4) {
        throw std::invalid_argument(
            "scale4-v2 expert storage is qualified only for Metal; CPU and CUDA require raw-v1");
      }
      return;
  }
  throw std::invalid_argument("expert storage layout is unknown");
}

#if defined(__APPLE__)
std::uint32_t metal_layout_id(const MoeExpertLayout layout) {
  switch (layout) {
    case MoeExpertLayout::RawV1:
      return K3_LAYOUT_RAW_V1;
    case MoeExpertLayout::Scale4V2:
      return K3_LAYOUT_SCALE4_V2;
  }
  throw std::invalid_argument("Metal expert storage layout is unknown");
}

K3MetalExpertDescriptorV1 metal_descriptor(const MoeExpertLayout layout,
                                           const std::uint64_t span_bytes,
                                           const std::uint8_t* blob) {
  return K3MetalExpertDescriptorV1{
      .abi_version = K3_DESCRIPTOR_ABI_V1,
      .struct_bytes = sizeof(K3MetalExpertDescriptorV1),
      .layout_id = metal_layout_id(layout),
      .reserved = 0,
      .blob_bytes = span_bytes,
      .blob = blob};
}

bool metal_descriptor_suite_ready(const MetalApi& api,
                                  const MoeExpertLayout layout) {
  if (api.descriptor_abi == nullptr || api.layout_capabilities == nullptr ||
      api.descriptor_abi() != K3_DESCRIPTOR_ABI_V1) {
    return false;
  }
  const std::uint64_t capability = layout == MoeExpertLayout::Scale4V2
                                       ? K3_CAP_SCALE4_V2
                                       : K3_CAP_RAW_V1;
  return (api.layout_capabilities() & capability) != 0;
}
#endif

std::array<float, kMoeRouteTopK> route_weights(const MoeRouteT1& route) {
  std::array<float, kMoeRouteTopK> weights = {};
  for (std::size_t edge = 0; edge < weights.size(); ++edge) {
    weights[edge] = std::bit_cast<float>(route.weight_bits[edge]);
  }
  return weights;
}

at::Tensor execute_cpu(const PreparedMoeT1& prepared,
                       const std::array<const std::uint8_t*, kMoeRouteTopK>& blobs,
                       const MoeRunOptions& options,
                       const MoeGeometry& geometry) {
#if !defined(DELTAFIN_HAVE_MXFP4_CPU_V1)
  static_cast<void>(prepared);
  static_cast<void>(blobs);
  static_cast<void>(options);
  static_cast<void>(geometry);
  throw std::runtime_error(
      "CPU MXFP4 MoE kernel was not linked into this provider build");
#else
  if (mxfp4_cpu_compatible() == 0) {
    throw std::runtime_error(
        "x86-64 CPU MXFP4 requires SSSE3, AVX, and FMA3; AVX2 is optional "
        "and runtime-dispatched");
  }
  const ExpertOffsets offsets = expert_offsets(geometry);
  const auto input_device = prepared.routed_input.device();
  at::Tensor input = prepared.routed_input_cpu;
  if (!input.defined()) {
    if (!prepared.routed_input.device().is_cpu()) {
      record_trace(options.execution_trace,
                   MoeExecutionStage::RoutedInputHostMaterialization);
    }
    input = prepared.routed_input.to(at::kCPU, at::kFloat).contiguous();
  }
  if (input.sizes() != at::IntArrayRef(
                           {1, static_cast<std::int64_t>(geometry.routed_hidden)})) {
    throw std::invalid_argument(
        "routed-MoE CPU input does not have shape [1, routed_hidden]");
  }

  constexpr int matrix13_count = static_cast<int>(2 * kMoeRouteTopK);
  constexpr int matrix2_count = static_cast<int>(kMoeRouteTopK);
  std::array<const std::uint8_t*, matrix13_count> packed13 = {};
  std::array<const std::uint8_t*, matrix13_count> scales13 = {};
  std::array<int, matrix13_count> rows13 = {};
  std::array<int, matrix13_count> columns13 = {};
  std::array<const std::uint8_t*, matrix2_count> packed2 = {};
  std::array<const std::uint8_t*, matrix2_count> scales2 = {};
  std::array<int, matrix2_count> rows2 = {};
  std::array<int, matrix2_count> columns2 = {};
  for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
    const std::uint8_t* blob = blobs[edge];
    packed13[2 * edge] = blob + offsets.w1_packed;
    scales13[2 * edge] = blob + offsets.w1_scales;
    packed13[2 * edge + 1] = blob + offsets.w3_packed;
    scales13[2 * edge + 1] = blob + offsets.w3_scales;
    rows13[2 * edge] = static_cast<int>(geometry.intermediate);
    rows13[2 * edge + 1] = static_cast<int>(geometry.intermediate);
    columns13[2 * edge] = static_cast<int>(geometry.routed_hidden);
    columns13[2 * edge + 1] = static_cast<int>(geometry.routed_hidden);
    packed2[edge] = blob + offsets.w2_packed;
    scales2[edge] = blob + offsets.w2_scales;
    rows2[edge] = static_cast<int>(geometry.routed_hidden);
    columns2[edge] = static_cast<int>(geometry.intermediate);
  }

  at::Tensor gate_up = at::empty(
      {static_cast<std::int64_t>(kMoeRouteTopK), 2,
       static_cast<std::int64_t>(geometry.intermediate)},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const int threads = static_cast<int>(std::clamp<std::uint32_t>(
      options.cpu_threads == 0 ? 1 : options.cpu_threads, 1, 32));
  record_trace(options.execution_trace, MoeExecutionStage::CpuExpertDispatch);
  mxfp4_moe_layer(packed13.data(), scales13.data(), rows13.data(),
                   columns13.data(), matrix13_count,
                   input.const_data_ptr<float>(), gate_up.data_ptr<float>(),
                   threads);

  // The established fused C one-call path uses platform libm and is documented
  // as differing from Kimi's framework SiTU by roughly one ulp. Keep SiTU in
  // this provider-owned ATen fp32 island, then re-enter the exact GEMV kernel.
  at::Tensor activated = situ(gate_up.select(1, 0), gate_up.select(1, 1))
                             .contiguous();
  at::Tensor outputs = at::empty(
      {static_cast<std::int64_t>(kMoeRouteTopK),
       static_cast<std::int64_t>(geometry.routed_hidden)},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  std::array<const float*, matrix2_count> activation_rows = {};
  std::array<float*, matrix2_count> output_rows = {};
  for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
    activation_rows[edge] = activated[static_cast<std::int64_t>(edge)]
                                .const_data_ptr<float>();
    output_rows[edge] =
        outputs[static_cast<std::int64_t>(edge)].data_ptr<float>();
  }
  mxfp4_gemv_batch(packed2.data(), scales2.data(), activation_rows.data(),
                    output_rows.data(), rows2.data(), columns2.data(),
                    matrix2_count, threads);

  const auto weights = route_weights(prepared.route);
  at::Tensor combined = at::zeros(
      {1, static_cast<std::int64_t>(geometry.routed_hidden)},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
    // Ordered, in-place fp32 accumulation mirrors the accepted Python path;
    // no atomic or expert-major reordering participates in this reduction.
    outputs[static_cast<std::int64_t>(edge)].mul_(weights[edge]);
    combined[0].add_(outputs[static_cast<std::int64_t>(edge)]);
  }
  return combined.to(input_device, at::kFloat);
#endif
}

at::Tensor execute_metal(const PreparedMoeT1& prepared,
                         const std::array<const std::uint8_t*, kMoeRouteTopK>& blobs,
                         const MoeExpertLayout layout,
                         const std::uint64_t expert_span_bytes,
                         const MoeRunOptions& options,
                         const MoeGeometry& geometry) {
#if !defined(__APPLE__)
  static_cast<void>(prepared);
  static_cast<void>(blobs);
  static_cast<void>(layout);
  static_cast<void>(expert_span_bytes);
  static_cast<void>(options);
  static_cast<void>(geometry);
  throw std::runtime_error(
      "Metal MXFP4 MoE was requested on a non-Apple provider build");
#else
  if (geometry.hidden != k3_moe_geometry().hidden ||
      geometry.routed_hidden != k3_moe_geometry().routed_hidden ||
      geometry.intermediate != k3_moe_geometry().intermediate) {
    throw std::invalid_argument(
        "the established Metal MXFP4 kernel accepts only exact K3 dimensions");
  }
  const MetalApi& api = metal_api();
  if (api.init == nullptr || api.layer == nullptr || api.drop == nullptr ||
      api.flush == nullptr) {
    throw std::runtime_error(
        "Metal MXFP4 MoE bridge with cache-retirement support is not linked "
        "into this provider executable");
  }
  const char* shader =
      options.metal_shader_path.empty() ? nullptr
                                        : options.metal_shader_path.c_str();
  if (api.init(shader) != 0) {
    const char* detail =
        api.last_error == nullptr ? nullptr : api.last_error();
    throw std::runtime_error(std::string("Metal MXFP4 initialization failed: ") +
                             (detail == nullptr ? "no detail" : detail));
  }
  const auto input_device = prepared.routed_input.device();
  at::Tensor input = prepared.routed_input_cpu;
  if (!input.defined()) {
    record_trace(options.execution_trace,
                 MoeExecutionStage::RoutedInputHostMaterialization);
    input = prepared.routed_input.to(at::kCPU, at::kFloat).contiguous();
  }
  // Write directly into ATen-owned host storage.  The previous vector +
  // from_blob().clone() sequence performed two heap allocations and copied
  // the 14 KiB routed row for every MoE layer of every token.
  at::Tensor routed_output = at::empty(
      {1, static_cast<std::int64_t>(geometry.routed_hidden)},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const auto weights = route_weights(prepared.route);
  record_trace(options.execution_trace,
               MoeExecutionStage::MetalExpertRowDispatch);
  int status = 0;
  bool executed = false;
  if (options.metal_staged_token != 0) {
    // Complete a layer whose experts were partly computed as their bytes
    // landed. The bridge recomputes anything that never got staged and then
    // performs the single route-ordered reduction, so consuming the token is
    // equivalent to the established call in every observable way.
    if (api.stage_finish == nullptr ||
        !metal_descriptor_suite_ready(api, layout)) {
      if (api.stage_abandon != nullptr) {
        api.stage_abandon(options.metal_staged_token);
      }
    } else {
      std::array<K3MetalExpertDescriptorV1, kMoeRouteTopK> descriptors{};
      for (std::size_t edge = 0; edge < descriptors.size(); ++edge) {
        descriptors[edge] =
            metal_descriptor(layout, expert_span_bytes, blobs[edge]);
      }
      status = api.stage_finish(
          options.metal_staged_token, descriptors.data(),
          static_cast<int>(descriptors.size()), weights.data(),
          input.const_data_ptr<float>(), routed_output.data_ptr<float>());
      executed = true;
    }
  }
  if (executed) {
    // The staged finish already produced the layer's routed output.
  } else if (layout == MoeExpertLayout::RawV1) {
    status = api.layer(blobs.data(), static_cast<int>(blobs.size()),
                       weights.data(), input.const_data_ptr<float>(),
                       routed_output.data_ptr<float>());
  } else if (layout == MoeExpertLayout::Scale4V2) {
    if (api.descriptor_abi == nullptr || api.layout_capabilities == nullptr ||
        api.layer_descriptor == nullptr ||
        api.descriptor_abi() != K3_DESCRIPTOR_ABI_V1 ||
        (api.layout_capabilities() & K3_CAP_SCALE4_V2) == 0) {
      throw std::runtime_error(
          "Metal bridge did not qualify the scale4-v2 descriptor suite");
    }
    std::array<K3MetalExpertDescriptorV1, kMoeRouteTopK> descriptors{};
    for (std::size_t edge = 0; edge < descriptors.size(); ++edge) {
      descriptors[edge] = K3MetalExpertDescriptorV1{
          .abi_version = K3_DESCRIPTOR_ABI_V1,
          .struct_bytes = sizeof(K3MetalExpertDescriptorV1),
          .layout_id = K3_LAYOUT_SCALE4_V2,
          .reserved = 0,
          .blob_bytes = expert_span_bytes,
          .blob = blobs[edge]};
    }
    status = api.layer_descriptor(
        descriptors.data(), static_cast<int>(descriptors.size()),
        weights.data(), input.const_data_ptr<float>(),
        routed_output.data_ptr<float>());
  } else {
    throw std::invalid_argument("Metal expert storage layout is unknown");
  }
  if (status != 0) {
    // A failed call never earns cache residency. Drop only the wrappers this
    // attempt may have created before the borrowed lease can unwind.
    for (const std::uint8_t* blob : blobs) {
      api.drop(blob);
    }
    const char* detail =
        api.last_error == nullptr ? nullptr : api.last_error();
    throw std::runtime_error(std::string("Metal MXFP4 MoE failed: ") +
                             (detail == nullptr ? "no detail" : detail));
  }
  if (!options.metal_retain_expert_wrappers) {
    for (const std::uint8_t* blob : blobs) {
      api.drop(blob);
    }
  }
  return routed_output.to(input_device, at::kFloat);
#endif
}

std::vector<const std::uint8_t*> contiguous_expert_span_pointers(
    const CanonicalExpertPositionTileT1& experts) {
  if (experts.expert_span_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("expert span exceeds host size_t");
  }
  const std::size_t span =
      static_cast<std::size_t>(experts.expert_span_bytes);
  std::vector<const std::uint8_t*> pointers;
  pointers.reserve(experts.expert_ids.size());
  for (std::size_t index = 0; index < experts.expert_ids.size(); ++index) {
    pointers.push_back(experts.expert_major_bytes.data() + index * span);
  }
  return pointers;
}

at::Tensor execute_cpu_pointer_positions(
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const std::span<const std::uint16_t> expert_ids,
    const std::span<const std::uint8_t* const> expert_span_pointers,
    const MoeRunOptions& options, const MoeGeometry& geometry) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(prepared_rows.size());
  for (const PreparedMoeT1* prepared : prepared_rows) {
    const auto blobs = ordered_expert_pointers(prepared->route, expert_ids,
                                               expert_span_pointers, geometry);
    outputs.push_back(execute_cpu(*prepared, blobs, options, geometry));
  }
  return at::cat(outputs, 0).contiguous();
}

at::Tensor execute_cpu_positions(
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const CanonicalExpertPositionTileT1& experts,
    const MoeRunOptions& options, const MoeGeometry& geometry) {
  const std::vector<const std::uint8_t*> pointers =
      contiguous_expert_span_pointers(experts);
  return execute_cpu_pointer_positions(prepared_rows, experts.expert_ids,
                                       pointers, options, geometry);
}

at::Tensor execute_cuda_plan_cpu_fallback(
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const CanonicalExpertPositionTileT1& missing_experts,
    const MoeRunOptions& options, const MoeGeometry& geometry) {
  if (options.cuda_cache == nullptr || options.cuda_plan == 0 ||
      missing_experts.layout != MoeExpertLayout::RawV1 ||
      missing_experts.expert_span_bytes != geometry.expert_span_bytes()) {
    throw std::logic_error(
        "CUDA automatic fallback lacks its exact raw residency plan");
  }
  try {
    CudaMoeHostFallback fallback =
        options.cuda_cache->materialize_plan_for_cpu(options.cuda_plan);
    const std::size_t span = static_cast<std::size_t>(
        geometry.expert_span_bytes());
    const std::uint64_t expected_missing = checked_mul(
        span, missing_experts.expert_ids.size(),
        "CUDA fallback missing-expert bytes");
    if (expected_missing > std::numeric_limits<std::size_t>::max() ||
        missing_experts.expert_major_bytes.size() !=
            static_cast<std::size_t>(expected_missing)) {
      throw std::invalid_argument(
          "CUDA fallback missing bytes do not match their canonical IDs");
    }
    const std::uint64_t full_size = checked_mul(
        span, fallback.canonical_experts.size(),
        "CUDA fallback canonical expert bytes");
    if (full_size > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("CUDA fallback host slab exceeds size_t");
    }
    std::vector<std::uint8_t> full(static_cast<std::size_t>(full_size));
    for (std::size_t canonical = 0;
         canonical < fallback.canonical_experts.size(); ++canonical) {
      const std::uint16_t expert = fallback.canonical_experts[canonical];
      if (canonical != 0 &&
          fallback.canonical_experts[canonical - 1] >= expert) {
        throw std::invalid_argument(
            "CUDA fallback plan IDs are not canonical ascending IDs");
      }
      const auto resident = std::find_if(
          fallback.resident_experts.begin(), fallback.resident_experts.end(),
          [expert](const CudaMoeHostExpert& item) {
            return item.expert == expert;
          });
      const auto missing = std::lower_bound(
          missing_experts.expert_ids.begin(),
          missing_experts.expert_ids.end(), expert);
      const bool has_missing = missing != missing_experts.expert_ids.end() &&
          *missing == expert;
      if ((resident != fallback.resident_experts.end()) == has_missing) {
        throw std::invalid_argument(
            "CUDA fallback expert must come from exactly one authenticated source");
      }
      const std::uint8_t* source = nullptr;
      if (resident != fallback.resident_experts.end()) {
        if (!resident->bytes.defined() ||
            resident->bytes.device().type() != at::kCPU ||
            resident->bytes.scalar_type() != at::kByte ||
            !resident->bytes.is_contiguous() || resident->bytes.dim() != 1 ||
            resident->bytes.numel() != static_cast<std::int64_t>(span)) {
          throw std::runtime_error(
              "CUDA fallback resident expert is not one exact host byte span");
        }
        source = resident->bytes.const_data_ptr<std::uint8_t>();
      } else {
        const std::size_t missing_index = static_cast<std::size_t>(
            missing - missing_experts.expert_ids.begin());
        source = missing_experts.expert_major_bytes.data() +
            missing_index * span;
      }
      std::memcpy(full.data() + canonical * span, source, span);
    }
    MoeRunOptions cpu_options = options;
    cpu_options.expert_backend = MoeExpertBackend::CpuMxfp4;
    cpu_options.cuda_plan = 0;
    cpu_options.cuda_auto_fallback = false;
    const CanonicalExpertPositionTileT1 complete{
        .expert_ids = fallback.canonical_experts,
        .expert_major_bytes = full,
        .layout = MoeExpertLayout::RawV1,
        .expert_span_bytes = geometry.expert_span_bytes()};
    at::Tensor output =
        execute_cpu_positions(prepared_rows, complete, cpu_options, geometry);
    options.cuda_cache->cancel_plan(options.cuda_plan);
    return output;
  } catch (...) {
    options.cuda_cache->cancel_plan(options.cuda_plan);
    throw;
  }
}

at::Tensor execute_metal_pointer_positions(
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const std::span<const std::uint16_t> expert_ids,
    const std::span<const std::uint8_t* const> expert_span_pointers,
    const MoeExpertLayout expert_layout,
    const std::uint64_t expert_span_bytes,
    const MoeRunOptions& options, const MoeGeometry& geometry) {
#if !defined(__APPLE__)
  static_cast<void>(prepared_rows);
  static_cast<void>(expert_ids);
  static_cast<void>(expert_span_pointers);
  static_cast<void>(expert_layout);
  static_cast<void>(expert_span_bytes);
  static_cast<void>(options);
  static_cast<void>(geometry);
  throw std::runtime_error(
      "Metal MXFP4 position tile was requested on a non-Apple provider build");
#else
  if (geometry.hidden != k3_moe_geometry().hidden ||
      geometry.routed_hidden != k3_moe_geometry().routed_hidden ||
      geometry.intermediate != k3_moe_geometry().intermediate) {
    throw std::invalid_argument(
        "the Metal position ABI accepts only exact K3 dimensions");
  }
  const MetalApi& api = metal_api();
  if (api.init == nullptr || api.layer == nullptr || api.drop == nullptr ||
      api.flush == nullptr) {
    throw std::runtime_error(
        "Metal MXFP4 MoE bridge with cache-retirement support is not linked "
        "into this provider executable");
  }
  const char* shader = options.metal_shader_path.empty()
                           ? nullptr
                           : options.metal_shader_path.c_str();
  if (api.init(shader) != 0) {
    const char* detail = api.last_error == nullptr ? nullptr : api.last_error();
    throw std::runtime_error(std::string("Metal MXFP4 initialization failed: ") +
                             (detail == nullptr ? "no detail" : detail));
  }
  const bool position_abi_available =
      expert_layout == MoeExpertLayout::RawV1
      ? (api.positions_flat != nullptr || api.positions != nullptr)
      : (api.positions_flat_descriptor != nullptr ||
         api.positions_descriptor != nullptr);
  if (prepared_rows.size() == 1 || !options.metal_position_batch ||
      !position_abi_available) {
    // Preserve the established one-command-buffer-per-row Metal arithmetic,
    // but mirror the qualified Python provider's transfer boundary: move the
    // complete routed-input tile to host once and the complete output tile
    // back once. Calling execute_metal() here would perform both crossings for
    // every row at every layer.
    std::vector<at::Tensor> input_rows;
    input_rows.reserve(prepared_rows.size());
    const at::Device output_device =
        prepared_rows.front()->routed_input.device();
    const bool all_host_materialized = std::all_of(
        prepared_rows.begin(), prepared_rows.end(),
        [](const PreparedMoeT1* prepared) {
          return prepared->routed_input_cpu.defined();
        });
    if (options.metal_retain_position_outputs_cpu &&
        !all_host_materialized) {
      throw std::logic_error(
          "retained Metal position outputs require whole-layer host-staged inputs");
    }
    if (!all_host_materialized) {
      record_trace(options.execution_trace,
                   MoeExecutionStage::RoutedInputHostMaterialization);
    }
    for (const PreparedMoeT1* prepared : prepared_rows) {
      input_rows.push_back(all_host_materialized
                               ? prepared->routed_input_cpu
                               : prepared->routed_input);
    }
    at::Tensor inputs = at::cat(input_rows, 0);
    if (!all_host_materialized) {
      inputs = inputs.to(at::kCPU, at::kFloat);
    }
    inputs = inputs.contiguous();
    at::Tensor outputs = at::empty(
        {static_cast<std::int64_t>(prepared_rows.size()),
         static_cast<std::int64_t>(geometry.routed_hidden)},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    record_trace(options.execution_trace,
                 MoeExecutionStage::MetalExpertRowDispatch);

    try {
      for (std::size_t row = 0; row < prepared_rows.size(); ++row) {
        const PreparedMoeT1& prepared = *prepared_rows[row];
        const auto blobs = ordered_expert_pointers(
            prepared.route, expert_ids, expert_span_pointers, geometry);
        const auto weights = route_weights(prepared.route);
        int status = 0;
        bool executed = false;
        // A staged token belongs to exactly one row; a tile that turned out to
        // hold more rows than the staging assumed cannot consume it.
        if (options.metal_staged_token != 0 && prepared_rows.size() == 1) {
          if (api.stage_finish == nullptr ||
              !metal_descriptor_suite_ready(api, expert_layout)) {
            if (api.stage_abandon != nullptr) {
              api.stage_abandon(options.metal_staged_token);
            }
          } else {
            std::array<K3MetalExpertDescriptorV1, kMoeRouteTopK> descriptors{};
            for (std::size_t edge = 0; edge < descriptors.size(); ++edge) {
              descriptors[edge] = metal_descriptor(
                  expert_layout, expert_span_bytes, blobs[edge]);
            }
            status = api.stage_finish(
                options.metal_staged_token, descriptors.data(),
                static_cast<int>(descriptors.size()), weights.data(),
                inputs[static_cast<std::int64_t>(row)]
                    .const_data_ptr<float>(),
                outputs[static_cast<std::int64_t>(row)].data_ptr<float>());
            executed = true;
          }
        }
        if (executed) {
          // The staged finish already produced this row's routed output.
        } else if (expert_layout == MoeExpertLayout::RawV1) {
          status = api.layer(
              blobs.data(), static_cast<int>(blobs.size()), weights.data(),
              inputs[static_cast<std::int64_t>(row)].const_data_ptr<float>(),
              outputs[static_cast<std::int64_t>(row)].data_ptr<float>());
        } else if (expert_layout == MoeExpertLayout::Scale4V2) {
          if (api.descriptor_abi == nullptr ||
              api.layout_capabilities == nullptr ||
              api.layer_descriptor == nullptr ||
              api.descriptor_abi() != K3_DESCRIPTOR_ABI_V1 ||
              (api.layout_capabilities() & K3_CAP_SCALE4_V2) == 0) {
            throw std::runtime_error(
                "Metal bridge did not qualify the scale4-v2 descriptor suite");
          }
          std::array<K3MetalExpertDescriptorV1, kMoeRouteTopK> descriptors{};
          for (std::size_t edge = 0; edge < descriptors.size(); ++edge) {
            descriptors[edge] = K3MetalExpertDescriptorV1{
                .abi_version = K3_DESCRIPTOR_ABI_V1,
                .struct_bytes = sizeof(K3MetalExpertDescriptorV1),
                .layout_id = K3_LAYOUT_SCALE4_V2,
                .reserved = 0,
                .blob_bytes = expert_span_bytes,
                .blob = blobs[edge]};
          }
          status = api.layer_descriptor(
              descriptors.data(), static_cast<int>(descriptors.size()),
              weights.data(),
              inputs[static_cast<std::int64_t>(row)].const_data_ptr<float>(),
              outputs[static_cast<std::int64_t>(row)].data_ptr<float>());
        } else {
          throw std::invalid_argument("Metal expert storage layout is unknown");
        }
        if (status != 0) {
          const char* detail =
              api.last_error == nullptr ? nullptr : api.last_error();
          throw std::runtime_error(std::string("Metal MXFP4 MoE failed: ") +
                                   (detail == nullptr ? "no detail" : detail));
        }
      }
    } catch (...) {
      for (const std::uint8_t* pointer : expert_span_pointers) {
        api.drop(pointer);
      }
      throw;
    }
    if (!options.metal_retain_expert_wrappers) {
      for (const std::uint8_t* pointer : expert_span_pointers) {
        api.drop(pointer);
      }
    }
    // Retained successful wrappers alias stable arena pages. Rust opts into
    // that lifetime explicitly and flushes before growth, retirement, or
    // engine teardown. Default/direct callers preserve synchronous borrowing.
    return options.metal_retain_position_outputs_cpu
        ? outputs
        : outputs.to(output_device, at::kFloat);
  }

  std::vector<const std::uint8_t*> edge_blobs;
  std::vector<float> edge_weights;
  std::vector<at::Tensor> input_rows;
  std::array<int, kMoePositionTileMaxRows + 1> offsets{};
  edge_blobs.reserve(prepared_rows.size() * kMoeRouteTopK);
  edge_weights.reserve(prepared_rows.size() * kMoeRouteTopK);
  input_rows.reserve(prepared_rows.size());
  const at::Device output_device = prepared_rows.front()->routed_input.device();
  const bool all_host_materialized = std::all_of(
      prepared_rows.begin(), prepared_rows.end(),
      [](const PreparedMoeT1* prepared) {
        return prepared->routed_input_cpu.defined();
      });
  if (options.metal_retain_position_outputs_cpu &&
      !all_host_materialized) {
    throw std::logic_error(
        "retained Metal position outputs require whole-layer host-staged inputs");
  }
  if (!all_host_materialized) {
    record_trace(options.execution_trace,
                 MoeExecutionStage::RoutedInputHostMaterialization);
  }
  for (std::size_t row = 0; row < prepared_rows.size(); ++row) {
    const PreparedMoeT1& prepared = *prepared_rows[row];
    const auto blobs = ordered_expert_pointers(
        prepared.route, expert_ids, expert_span_pointers, geometry);
    const auto weights = route_weights(prepared.route);
    edge_blobs.insert(edge_blobs.end(), blobs.begin(), blobs.end());
    edge_weights.insert(edge_weights.end(), weights.begin(), weights.end());
    input_rows.push_back(all_host_materialized
                             ? prepared.routed_input_cpu
                             : prepared.routed_input);
    offsets[row + 1] = static_cast<int>(edge_blobs.size());
  }
  at::Tensor inputs = at::cat(input_rows, 0);
  if (!all_host_materialized) {
    inputs = inputs.to(at::kCPU, at::kFloat);
  }
  inputs = inputs.contiguous();
  at::Tensor outputs = at::empty(
      {static_cast<std::int64_t>(prepared_rows.size()),
       static_cast<std::int64_t>(geometry.routed_hidden)},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  record_trace(options.execution_trace,
               MoeExecutionStage::MetalExpertPositionDispatch);
  int status = 0;
  bool fall_back_to_rows = false;
  if (expert_layout == MoeExpertLayout::RawV1) {
    status = api.positions_flat == nullptr
        ? -6
        : api.positions_flat(
              edge_blobs.data(), static_cast<int>(edge_blobs.size()),
              offsets.data(), static_cast<int>(prepared_rows.size()),
              edge_weights.data(), inputs.const_data_ptr<float>(),
              outputs.data_ptr<float>());
    if (status == -6) {
      if (api.positions != nullptr) {
        status = api.positions(
            edge_blobs.data(), static_cast<int>(edge_blobs.size()),
            offsets.data(), static_cast<int>(prepared_rows.size()),
            edge_weights.data(), inputs.const_data_ptr<float>(),
            outputs.data_ptr<float>());
      } else {
        fall_back_to_rows = true;
      }
    }
  } else if (expert_layout == MoeExpertLayout::Scale4V2) {
    if (api.descriptor_abi == nullptr || api.layout_capabilities == nullptr ||
        (api.positions_flat_descriptor == nullptr &&
         api.positions_descriptor == nullptr) ||
        api.descriptor_abi() != K3_DESCRIPTOR_ABI_V1 ||
        (api.layout_capabilities() & K3_CAP_SCALE4_V2) == 0) {
      throw std::runtime_error(
          "Metal bridge did not qualify the scale4-v2 position descriptor suite");
    }
    std::vector<K3MetalExpertDescriptorV1> descriptors(edge_blobs.size());
    for (std::size_t edge = 0; edge < descriptors.size(); ++edge) {
      descriptors[edge] = K3MetalExpertDescriptorV1{
          .abi_version = K3_DESCRIPTOR_ABI_V1,
          .struct_bytes = sizeof(K3MetalExpertDescriptorV1),
          .layout_id = K3_LAYOUT_SCALE4_V2,
          .reserved = 0,
          .blob_bytes = expert_span_bytes,
          .blob = edge_blobs[edge]};
    }
    status = api.positions_flat_descriptor == nullptr
        ? -6
        : api.positions_flat_descriptor(
              descriptors.data(), static_cast<int>(descriptors.size()),
              offsets.data(), static_cast<int>(prepared_rows.size()),
              edge_weights.data(), inputs.const_data_ptr<float>(),
              outputs.data_ptr<float>());
    if (status == -6) {
      if (api.positions_descriptor != nullptr) {
        status = api.positions_descriptor(
            descriptors.data(), static_cast<int>(descriptors.size()),
            offsets.data(), static_cast<int>(prepared_rows.size()),
            edge_weights.data(), inputs.const_data_ptr<float>(),
            outputs.data_ptr<float>());
      } else {
        fall_back_to_rows = true;
      }
    }
  } else {
    throw std::invalid_argument("Metal expert storage layout is unknown");
  }

  if (fall_back_to_rows) {
    // -6 is the flat ABI's explicit capability miss and occurs before it
    // creates no-copy wrappers. Drop defensively anyway, then preserve the
    // established row-serial implementation without changing arithmetic.
    for (const std::uint8_t* pointer : expert_span_pointers) {
      api.drop(pointer);
    }
    MoeRunOptions row_options = options;
    row_options.metal_position_batch = false;
    return execute_metal_pointer_positions(
        prepared_rows, expert_ids, expert_span_pointers, expert_layout,
        expert_span_bytes, row_options, geometry);
  }

  if (status != 0) {
    for (const std::uint8_t* pointer : expert_span_pointers) {
      api.drop(pointer);
    }
    const char* detail = api.last_error == nullptr ? nullptr : api.last_error();
    throw std::runtime_error(
        std::string("Metal MXFP4 position batch failed: ") +
        (detail == nullptr ? "no detail" : detail));
  }
  if (!options.metal_retain_expert_wrappers) {
    for (const std::uint8_t* pointer : expert_span_pointers) {
      api.drop(pointer);
    }
  }
  return options.metal_retain_position_outputs_cpu
      ? outputs
      : outputs.to(output_device, at::kFloat);
#endif
}

at::Tensor execute_metal_positions(
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const CanonicalExpertPositionTileT1& experts,
    const MoeRunOptions& options, const MoeGeometry& geometry) {
  const std::vector<const std::uint8_t*> pointers =
      contiguous_expert_span_pointers(experts);
  return execute_metal_pointer_positions(
      prepared_rows, experts.expert_ids, pointers, experts.layout,
      experts.expert_span_bytes, options, geometry);
}

MoeExpertBackend select_backend(const at::Device& device,
                                const MoeExpertBackend requested,
                                CudaMoeExpertCache* const cuda_cache) {
  if (requested != MoeExpertBackend::Auto) {
    return requested;
  }
  if (device.is_mps()) {
#if defined(__APPLE__)
    if (metal_api().layer != nullptr && metal_api().drop != nullptr &&
        metal_api().flush != nullptr) {
      return MoeExpertBackend::MetalMxfp4;
    }
#endif
    return MoeExpertBackend::CpuMxfp4;
  }
  if (device.is_cuda()) {
    if (cuda_cache != nullptr && cuda_cache->available()) {
      return MoeExpertBackend::CudaMxfp4;
    }
    return MoeExpertBackend::CpuMxfp4;
  }
  if (device.is_cpu()) {
    return MoeExpertBackend::CpuMxfp4;
  }
  throw std::runtime_error("selected provider has no routed-MoE backend");
}

}  // namespace

std::uint64_t MoeGeometry::expert_span_bytes() const {
  validate_geometry(*this);
  return expert_offsets(*this).span;
}

MetalExpertLayoutCapabilities qualify_metal_expert_layouts(
    const std::string& metal_shader_path) {
#if !defined(__APPLE__)
  static_cast<void>(metal_shader_path);
  throw std::runtime_error(
      "Metal expert-layout qualification requires an Apple provider build");
#else
  const MetalApi& api = metal_api();
  if (api.init == nullptr || api.layer == nullptr || api.drop == nullptr ||
      api.flush == nullptr) {
    throw std::runtime_error(
        "Metal MXFP4 MoE bridge with cache-retirement support is not linked "
        "into this provider executable");
  }
  const char* shader = metal_shader_path.empty()
                           ? nullptr
                           : metal_shader_path.c_str();
  if (api.init(shader) != 0) {
    const char* detail = api.last_error == nullptr ? nullptr : api.last_error();
    throw std::runtime_error(std::string("Metal MXFP4 initialization failed: ") +
                             (detail == nullptr ? "no detail" : detail));
  }

  MetalExpertLayoutCapabilities result{
      .descriptor_abi = 0,
      .layout_capabilities = K3_CAP_RAW_V1,
      .raw_span_bytes = K3_RAW_V1_EXPERT_SPAN,
      .scale4_span_bytes = K3_SCALE4_V2_EXPERT_SPAN};
  // Raw-v1 keeps its legacy entry points and remains usable when an older
  // dynamically linked bridge lacks the additive position/descriptor suite.
  // Scale4 auto-admission requires that complete suite and therefore degrades
  // to a raw-only report instead of turning an optional capability miss into
  // a provider-startup failure.
  if (api.positions == nullptr || api.descriptor_abi == nullptr ||
      api.layout_capabilities == nullptr ||
      api.layer_descriptor == nullptr ||
      api.positions_descriptor == nullptr) {
    return result;
  }
  result.descriptor_abi = api.descriptor_abi();
  if (result.descriptor_abi != K3_DESCRIPTOR_ABI_V1) {
    return result;
  }
  const std::uint64_t advertised = api.layout_capabilities();
  if ((advertised & K3_CAP_RAW_V1) == 0) {
    throw std::runtime_error(
        "Metal descriptor capability report omitted raw-v1");
  }
  constexpr std::uint64_t recognized = K3_CAP_RAW_V1 | K3_CAP_SCALE4_V2;
  result.layout_capabilities = advertised & recognized;
  return result;
#endif
}

void flush_metal_expert_cache() {
#if !defined(__APPLE__)
  throw std::runtime_error(
      "Metal expert cache flush requires an Apple provider build");
#else
  const MetalApi& api = metal_api();
  if (api.flush == nullptr) {
    throw std::runtime_error(
        "Metal MXFP4 cache flush is not linked into this provider executable");
  }
  api.flush();
#endif
}

MetalExpertCacheStats metal_expert_cache_stats() {
#if !defined(__APPLE__)
  throw std::runtime_error(
      "Metal expert cache stats require an Apple provider build");
#else
  const MetalApi& api = metal_api();
  if (api.stats == nullptr) {
    throw std::runtime_error(
        "Metal MXFP4 cache stats are not linked into this provider executable");
  }
  std::array<long long, 5> raw{};
  api.stats(raw.data());
  if (std::any_of(raw.begin(), raw.end(), [](const long long value) {
        return value < 0;
      }) || raw[4] > 1) {
    throw std::runtime_error(
        "Metal MXFP4 cache returned invalid cumulative counters");
  }
  return MetalExpertCacheStats{
      .calls = static_cast<std::uint64_t>(raw[0]),
      .zero_copy_wraps = static_cast<std::uint64_t>(raw[1]),
      .copies = static_cast<std::uint64_t>(raw[2]),
      .cache_entries = static_cast<std::uint64_t>(raw[3]),
      .bindless = static_cast<std::uint64_t>(raw[4])};
#endif
}

bool moe_positions_select_metal(const at::Device& device,
                                const MoeRunOptions& options) {
  return select_backend(device, options.expert_backend, options.cuda_cache) ==
      MoeExpertBackend::MetalMxfp4;
}

std::uint64_t begin_staged_routed_moe_t1(const PreparedMoeT1& prepared,
                                         const at::Tensor& routed_input_cpu,
                                         const MoeRunOptions& options) {
#if !defined(__APPLE__)
  static_cast<void>(prepared);
  static_cast<void>(routed_input_cpu);
  static_cast<void>(options);
  return 0;
#else
  const MoeGeometry& geometry = prepared.geometry;
  if (geometry.hidden != k3_moe_geometry().hidden ||
      geometry.routed_hidden != k3_moe_geometry().routed_hidden ||
      geometry.intermediate != k3_moe_geometry().intermediate) {
    return 0;
  }
  if (select_backend(prepared.routed_input.device(), options.expert_backend,
                     options.cuda_cache) != MoeExpertBackend::MetalMxfp4) {
    return 0;
  }
  if (!routed_input_cpu.defined() ||
      routed_input_cpu.scalar_type() != at::kFloat ||
      !routed_input_cpu.device().is_cpu() ||
      !routed_input_cpu.is_contiguous() ||
      routed_input_cpu.sizes() != at::IntArrayRef(
          {1, static_cast<std::int64_t>(geometry.routed_hidden)})) {
    return 0;
  }
  const MetalApi& api = metal_api();
  if (api.init == nullptr || api.stage_begin == nullptr ||
      api.stage_edges == nullptr || api.stage_finish == nullptr ||
      api.stage_abandon == nullptr) {
    return 0;
  }
  const char* shader =
      options.metal_shader_path.empty() ? nullptr
                                        : options.metal_shader_path.c_str();
  if (api.init(shader) != 0) {
    return 0;
  }
  return api.stage_begin(static_cast<int>(kMoeRouteTopK),
                         routed_input_cpu.const_data_ptr<float>());
#endif
}

bool stage_routed_moe_edges_t1(
    const std::uint64_t token, const PreparedMoeT1& prepared,
    const std::span<const std::uint16_t> arrived_expert_ids,
    const std::span<const std::uint8_t* const> arrived_span_pointers,
    const MoeExpertLayout layout, const std::uint64_t expert_span_bytes,
    const MoeRunOptions& options) {
#if !defined(__APPLE__)
  static_cast<void>(token);
  static_cast<void>(prepared);
  static_cast<void>(arrived_expert_ids);
  static_cast<void>(arrived_span_pointers);
  static_cast<void>(layout);
  static_cast<void>(expert_span_bytes);
  static_cast<void>(options);
  return false;
#else
  if (token == 0 || arrived_expert_ids.empty() ||
      arrived_expert_ids.size() != arrived_span_pointers.size() ||
      arrived_expert_ids.size() > kMoeRouteTopK ||
      expert_span_bytes != required_expert_span(layout, prepared.geometry)) {
    return false;
  }
  const MetalApi& api = metal_api();
  if (api.stage_edges == nullptr || !metal_descriptor_suite_ready(api, layout)) {
    return false;
  }
  // Map each arrived expert onto the route edge that owns it. The edge index
  // is the router's own position for that expert, which is what fixes the
  // reduction order later -- arrival order only decides when the edge's
  // independent matmul runs, never where its result lands.
  std::array<K3MetalExpertDescriptorV1, kMoeRouteTopK> descriptors{};
  std::array<int, kMoeRouteTopK> edges{};
  std::size_t count = 0;
  for (std::size_t index = 0; index < arrived_expert_ids.size(); ++index) {
    const std::uint16_t expert = arrived_expert_ids[index];
    if (index != 0 && arrived_expert_ids[index - 1] >= expert) {
      return false;
    }
    const std::uint8_t* blob = arrived_span_pointers[index];
    if (blob == nullptr) {
      return false;
    }
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      if (prepared.route.expert_ids[edge] != expert) {
        continue;
      }
      if (count >= edges.size()) {
        return false;
      }
      descriptors[count] = metal_descriptor(layout, expert_span_bytes, blob);
      edges[count] = static_cast<int>(edge);
      ++count;
    }
  }
  if (count == 0) {
    return false;
  }
  static_cast<void>(options);
  return api.stage_edges(token, descriptors.data(), edges.data(),
                         static_cast<int>(count)) == 0;
#endif
}

void abandon_staged_routed_moe_t1(const std::uint64_t token) {
#if defined(__APPLE__)
  if (token == 0) {
    return;
  }
  const MetalApi& api = metal_api();
  if (api.stage_abandon != nullptr) {
    api.stage_abandon(token);
  }
#else
  static_cast<void>(token);
#endif
}

StagedMoeStats staged_moe_stats() {
  StagedMoeStats stats;
#if defined(__APPLE__)
  const MetalApi& api = metal_api();
  if (api.stage_stats != nullptr) {
    long long counters[4] = {0, 0, 0, 0};
    api.stage_stats(counters);
    stats.begins = static_cast<std::uint64_t>(std::max<long long>(counters[0], 0));
    stats.edges = static_cast<std::uint64_t>(std::max<long long>(counters[1], 0));
    stats.reused_finishes =
        static_cast<std::uint64_t>(std::max<long long>(counters[2], 0));
    stats.recomputed_finishes =
        static_cast<std::uint64_t>(std::max<long long>(counters[3], 0));
  }
#endif
  return stats;
}

bool qualify_moe_shared_gate_up(MoeSpineT1& spine) {
  std::optional<MoeRowInt8Matrix> bundle = adjacent_shared_gate_up(spine);
  if (!bundle.has_value()) {
    spine.shared_gate_up = MoeRowInt8Matrix{};
    spine.shared_gate_up_qualified = false;
    spine.shared_gate_up_enabled = false;
    return false;
  }
  spine.shared_gate_up = std::move(*bundle);
  spine.shared_gate_up_qualified = true;
  spine.shared_gate_up_enabled = false;
  return true;
}

PreparedMoeT1 prepare_moe_t1(const at::Tensor& hidden,
                             const MoeSpineT1& spine,
                             MoeExecutionTrace* execution_trace) {
  const c10::InferenceMode inference_guard;
  if (!hidden.defined() || hidden.scalar_type() != at::kFloat ||
      !hidden.is_contiguous() || hidden.dim() != 2 || hidden.size(0) != 1 ||
      hidden.size(1) != static_cast<std::int64_t>(spine.geometry.hidden)) {
    throw std::invalid_argument(
        "routed-MoE prepare requires contiguous fp32 hidden [1, 7168]");
  }
  validate_spine(spine, hidden.device());

  record_trace(execution_trace, MoeExecutionStage::Router);
  const at::Tensor logits =
      row_int8_linear(hidden, spine.router, spine.packed_int8_qualified);
  const at::Tensor scores = at::sigmoid(logits);
  const at::Tensor choice = at::add(scores, spine.router_correction_bias);
  const auto [ignored_values, topk_ids] = at::topk(
      choice, static_cast<std::int64_t>(kMoeRouteTopK), -1, true, false);
  static_cast<void>(ignored_values);
  const at::Tensor unnormalized = at::gather(scores, 1, topk_ids);
  const at::Tensor denominator = at::add(
      at::sum(unnormalized, std::vector<std::int64_t>{-1}, true), 1.0e-20);
  const at::Tensor weights = at::div(unnormalized, denominator);
  // Match KimiSparseMoeBlock scheduling: queue routed_down while route tensors
  // remain device-resident, then cross only the route host boundary here.
  record_trace(execution_trace, MoeExecutionStage::RoutedDown);
  at::Tensor routed_input = row_int8_linear(
      hidden, spine.routed_down, spine.packed_int8_qualified);

  PreparedMoeT1 prepared;
  prepared.layer_index = spine.layer_index;
  prepared.spine_generation = spine.generation;
  prepared.geometry = spine.geometry;
  record_trace(execution_trace, MoeExecutionStage::RouteMaterialization);
  prepared.route = materialize_route_t1(topk_ids, weights, spine.geometry);
  prepared.identity = hidden;
  prepared.routed_input = std::move(routed_input);
  // Host materialization belongs to expert execution, after Rust has fetched
  // and authenticated the route's expert bytes. Preparing it here would put a
  // second MPS synchronization directly on the demand-I/O critical path.
  return prepared;
}

PreparedMoePositionsT1 prepare_moe_positions_t1(
    const at::Tensor& hidden_rows, const MoeSpineT1& spine,
    MoeExecutionTrace* execution_trace) {
  const c10::InferenceMode inference_guard;
  if (!hidden_rows.defined() || hidden_rows.scalar_type() != at::kFloat ||
      !hidden_rows.is_contiguous() || hidden_rows.dim() != 2 ||
      hidden_rows.size(0) < 1 ||
      hidden_rows.size(0) > kMaximumPreparedPositions ||
      hidden_rows.size(1) !=
          static_cast<std::int64_t>(spine.geometry.hidden)) {
    throw std::invalid_argument(
        "routed-MoE position prepare requires contiguous fp32 hidden [1..64, 7168]");
  }
  if (hidden_rows.size(0) == 1) {
    PreparedMoePositionsT1 result;
    result.rows.push_back(
        prepare_moe_t1(hidden_rows, spine, execution_trace));
    result.router_dispatches = 1;
    result.routed_down_dispatches = 1;
    result.shared_dispatches = 0;
    result.route_materializations = 1;
    result.route_host_transfers = hidden_rows.device().is_cpu() ? 0 : 1;
    result.routed_input_host_transfers = 0;
    return result;
  }
  validate_spine(spine, hidden_rows.device());

  record_trace(execution_trace, MoeExecutionStage::Router);
  const at::Tensor logits = row_int8_linear(
      hidden_rows, spine.router, spine.packed_int8_qualified);
  const at::Tensor scores = at::sigmoid(logits);
  const at::Tensor choice = at::add(scores, spine.router_correction_bias);
  const auto [ignored_values, topk_ids] = at::topk(
      choice, static_cast<std::int64_t>(kMoeRouteTopK), -1, true, false);
  static_cast<void>(ignored_values);
  const at::Tensor unnormalized = at::gather(scores, 1, topk_ids);
  const at::Tensor denominator = at::add(
      at::sum(unnormalized, std::vector<std::int64_t>{-1}, true), 1.0e-20);
  const at::Tensor weights = at::div(unnormalized, denominator).contiguous();
  // Queue the full-T routed projection before route materialization forces a
  // host boundary, matching the live Python ordering.
  record_trace(execution_trace, MoeExecutionStage::RoutedDown);
  const at::Tensor routed_inputs = row_int8_linear(
      hidden_rows, spine.routed_down, spine.packed_int8_qualified)
                                       .contiguous();
  record_trace(execution_trace, MoeExecutionStage::RouteMaterialization);
  std::vector<MoeRouteT1> routes =
      materialize_route_positions(topk_ids.contiguous(), weights,
                                  spine.geometry);

  PreparedMoePositionsT1 result;
  const std::size_t positions =
      static_cast<std::size_t>(hidden_rows.size(0));
  result.rows.reserve(positions);
  for (std::size_t row = 0; row < positions; ++row) {
    const auto index = static_cast<std::int64_t>(row);
    result.rows.push_back(PreparedMoeT1{
        .layer_index = spine.layer_index,
        .spine_generation = spine.generation,
        .geometry = spine.geometry,
        .identity = hidden_rows.narrow(0, index, 1),
        .routed_input = routed_inputs.narrow(0, index, 1),
        .routed_input_cpu = at::Tensor(),
        .shared_output = at::Tensor(),
        .route = routes[row],
    });
  }
  result.routed_inputs = routed_inputs;
  result.router_dispatches = 1;
  result.routed_down_dispatches = 1;
  result.shared_dispatches = 0;
  result.route_materializations = 1;
  result.route_host_transfers = hidden_rows.device().is_cpu() ? 0 : 1;
  result.routed_input_host_transfers = 0;
  return result;
}

at::Tensor execute_routed_moe_t1(const PreparedMoeT1& prepared,
                                 const CanonicalExpertBatchT1& experts,
                                 const MoeRunOptions& options) {
  const c10::InferenceMode inference_guard;
  if (!prepared.routed_input.defined() ||
      prepared.routed_input.scalar_type() != at::kFloat ||
      !prepared.routed_input.is_contiguous() ||
      prepared.routed_input.dim() != 2 ||
      prepared.routed_input.size(0) != 1) {
    throw std::invalid_argument(
        "routed-MoE execute requires contiguous fp32 routed input [1, H]");
  }
  const MoeGeometry& geometry = prepared.geometry;
  validate_geometry(geometry);
  if (experts.expert_ids.size() != kMoeRouteTopK) {
    throw std::invalid_argument(
        "T=1 expert batch must contain exactly all 16 routed experts");
  }
  const MoeExpertBackend backend = select_backend(
      prepared.routed_input.device(), options.expert_backend,
      options.cuda_cache);
  validate_backend_layout(backend, experts.layout);
  validate_expert_storage(experts.expert_ids, experts.expert_major_bytes,
                          experts.layout, experts.expert_span_bytes, geometry,
                          kMoeRouteTopK);
  if (prepared.routed_input.size(1) !=
      static_cast<std::int64_t>(geometry.routed_hidden)) {
    throw std::invalid_argument(
        "routed-MoE execute width does not match its prepared geometry");
  }
  const auto blobs = ordered_expert_pointers(
      prepared.route, experts.expert_ids, experts.expert_major_bytes, geometry,
      experts.expert_span_bytes);
  record_trace(options.execution_trace,
               MoeExecutionStage::ExpertBytesBorrowed);
  switch (backend) {
    case MoeExpertBackend::CpuMxfp4:
      return execute_cpu(prepared, blobs, options, geometry);
    case MoeExpertBackend::MetalMxfp4:
      return execute_metal(prepared, blobs, experts.layout,
                           experts.expert_span_bytes, options, geometry);
    case MoeExpertBackend::CudaMxfp4:
      if (options.cuda_cache == nullptr) {
        throw std::runtime_error(
            "CUDA MXFP4 was requested without a session-owned expert cache");
      }
      record_trace(options.execution_trace,
                   MoeExecutionStage::CudaExpertDispatch);
      return options.cuda_cache->execute_t1(prepared, experts);
    case MoeExpertBackend::Auto:
      break;
  }
  throw std::logic_error("automatic routed-MoE backend remained unresolved");
}

at::Tensor execute_routed_moe_positions_t1(
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const CanonicalExpertPositionTileT1& experts,
    const MoeRunOptions& options) {
  const c10::InferenceMode inference_guard;
  if (prepared_rows.empty() ||
      prepared_rows.size() > kMoePositionTileMaxRows) {
    throw std::invalid_argument(
        "routed-MoE position tile must contain 1..16 rows");
  }
  const PreparedMoeT1* first = prepared_rows.front();
  if (first == nullptr) {
    throw std::invalid_argument("routed-MoE position tile has a null row");
  }
  const MoeGeometry& geometry = first->geometry;
  validate_geometry(geometry);
  const at::Device device = first->routed_input.device();
  const MoeExpertBackend backend =
      select_backend(device, options.expert_backend, options.cuda_cache);
  validate_backend_layout(backend, experts.layout);
  const bool planned_cuda = options.cuda_plan != 0;
  const bool scattered = !experts.expert_span_pointers.empty();
  const bool contiguous = !experts.expert_major_bytes.empty();
  if (planned_cuda) {
    if (scattered) {
      throw std::invalid_argument(
          "planned CUDA experts require the contiguous residency-plan storage form");
    }
    if (contiguous != !experts.expert_ids.empty()) {
      throw std::invalid_argument(
          "planned CUDA miss storage must be empty exactly when every expert is resident");
    }
    validate_planned_cuda_misses(experts, geometry);
  } else {
    if (scattered == contiguous) {
      throw std::invalid_argument(
          "expert tile must provide exactly one contiguous or scattered storage form");
    }
  }
  if (planned_cuda) {
    // The plan validator above intentionally permits an empty all-hit miss
    // set. No caller-owned byte storage exists in that qualified case.
  } else if (scattered) {
    if (backend == MoeExpertBackend::CudaMxfp4) {
      throw std::invalid_argument(
          "CUDA expert execution does not accept scattered host spans");
    }
    validate_expert_span_pointers(
        experts.expert_ids, experts.expert_span_pointers, experts.layout,
        experts.expert_span_bytes, geometry, kMoePositionTileMaxExperts);
  } else {
    validate_expert_storage(experts.expert_ids, experts.expert_major_bytes,
                            experts.layout, experts.expert_span_bytes, geometry,
                            kMoePositionTileMaxExperts);
  }
  for (const PreparedMoeT1* prepared : prepared_rows) {
    if (prepared == nullptr || !prepared->routed_input.defined() ||
        prepared->routed_input.scalar_type() != at::kFloat ||
        !prepared->routed_input.is_contiguous() ||
        prepared->routed_input.sizes() != at::IntArrayRef(
            {1, static_cast<std::int64_t>(geometry.routed_hidden)}) ||
        prepared->routed_input.device() != device ||
        prepared->layer_index != first->layer_index ||
        prepared->spine_generation != first->spine_generation ||
        prepared->geometry.hidden != geometry.hidden ||
        prepared->geometry.routed_hidden != geometry.routed_hidden ||
        prepared->geometry.intermediate != geometry.intermediate ||
        prepared->geometry.experts != geometry.experts ||
        prepared->geometry.shared_intermediate !=
            geometry.shared_intermediate) {
      throw std::invalid_argument(
          "routed-MoE position rows disagree on layer/generation/geometry/device");
    }
    validate_route(prepared->route, geometry);
    if (!planned_cuda) {
      // Without a provider-owned residency snapshot, every route edge must be
      // present in this caller-owned canonical byte tile.
      if (scattered) {
        static_cast<void>(ordered_expert_pointers(
            prepared->route, experts.expert_ids,
            experts.expert_span_pointers, geometry));
      } else {
        static_cast<void>(ordered_expert_pointers(
            prepared->route, experts.expert_ids, experts.expert_major_bytes,
            geometry, experts.expert_span_bytes));
      }
    }
  }

  record_trace(options.execution_trace,
               MoeExecutionStage::ExpertBytesBorrowed);
  switch (backend) {
    case MoeExpertBackend::CpuMxfp4:
      if (scattered) {
        return execute_cpu_pointer_positions(
            prepared_rows, experts.expert_ids, experts.expert_span_pointers,
            options, geometry);
      }
      return execute_cpu_positions(prepared_rows, experts, options, geometry);
    case MoeExpertBackend::MetalMxfp4:
      if (scattered) {
        return execute_metal_pointer_positions(
            prepared_rows, experts.expert_ids, experts.expert_span_pointers,
            experts.layout, experts.expert_span_bytes, options, geometry);
      }
      return execute_metal_positions(prepared_rows, experts, options, geometry);
    case MoeExpertBackend::CudaMxfp4:
      if (options.cuda_cache == nullptr) {
        throw std::runtime_error(
            "CUDA MXFP4 position execution lacks a session-owned expert cache");
      }
      if (planned_cuda) {
        try {
          record_trace(options.execution_trace,
                       MoeExecutionStage::CudaExpertDispatch);
          return options.cuda_cache->execute_positions_plan_t1(
              options.cuda_plan, prepared_rows, experts);
        } catch (const CudaMoeRecoverableError&) {
          if (!options.cuda_auto_fallback) {
            throw;
          }
          return execute_cuda_plan_cpu_fallback(
              prepared_rows, experts, options, geometry);
        }
      }
      record_trace(options.execution_trace,
                   MoeExecutionStage::CudaExpertDispatch);
      return options.cuda_cache->execute_positions_t1(prepared_rows, experts);
    case MoeExpertBackend::Auto:
      break;
  }
  throw std::logic_error(
      "automatic routed-MoE position backend remained unresolved");
}

at::Tensor complete_moe_t1(const PreparedMoeT1& prepared,
                           const at::Tensor& routed_output,
                           const MoeSpineT1& spine,
                           MoeExecutionTrace* execution_trace) {
  const c10::InferenceMode inference_guard;
  validate_spine(spine, prepared.identity.device());
  if (prepared.layer_index != spine.layer_index ||
      prepared.spine_generation != spine.generation ||
      prepared.geometry.hidden != spine.geometry.hidden ||
      prepared.geometry.routed_hidden != spine.geometry.routed_hidden ||
      prepared.geometry.intermediate != spine.geometry.intermediate ||
      prepared.geometry.experts != spine.geometry.experts ||
      prepared.geometry.shared_intermediate !=
          spine.geometry.shared_intermediate) {
    throw std::invalid_argument(
        "routed-MoE completion cannot consume a stale or different spine generation");
  }
  const auto routed = static_cast<std::int64_t>(spine.geometry.routed_hidden);
  if (!routed_output.defined() || routed_output.scalar_type() != at::kFloat ||
      routed_output.device() != prepared.identity.device() ||
      !routed_output.is_contiguous() ||
      routed_output.sizes() != at::IntArrayRef({1, routed})) {
    throw std::invalid_argument(
        "routed-MoE completion output violates [1, routed_hidden] fp32 contract");
  }

  record_trace(execution_trace, MoeExecutionStage::RoutedUp);
  const at::Tensor normalized =
      rms_norm(routed_output, spine.routed_norm);
  const at::Tensor routed_full = row_int8_linear(
      normalized, spine.routed_up, spine.packed_int8_qualified);
  record_trace(execution_trace, MoeExecutionStage::Shared);
  const at::Tensor shared_output = prepared.shared_output.defined()
                                       ? prepared.shared_output
                                       : shared_expert(prepared.identity, spine);
  const auto hidden = static_cast<std::int64_t>(spine.geometry.hidden);
  if (shared_output.scalar_type() != at::kFloat ||
      shared_output.device() != prepared.identity.device() ||
      shared_output.sizes() != at::IntArrayRef({1, hidden})) {
    throw std::runtime_error(
        "provider shared-expert output violated its fp32 shape/device contract");
  }
  record_trace(execution_trace, MoeExecutionStage::Merge);
  return at::add(routed_full, shared_output).contiguous();
}

at::Tensor complete_moe_positions_t1(
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const at::Tensor& routed_outputs, const MoeSpineT1& spine,
    MoeExecutionTrace* execution_trace) {
  const c10::InferenceMode inference_guard;
  if (prepared_rows.empty() ||
      prepared_rows.size() > kMaximumPreparedPositions) {
    throw std::invalid_argument(
        "routed-MoE position completion requires 1..64 rows");
  }
  if (prepared_rows.size() == 1) {
    if (prepared_rows.front() == nullptr) {
      throw std::invalid_argument(
          "routed-MoE position completion has a null row");
    }
    return complete_moe_t1(*prepared_rows.front(), routed_outputs, spine,
                           execution_trace);
  }

  const PreparedMoeT1* first = prepared_rows.front();
  if (first == nullptr) {
    throw std::invalid_argument(
        "routed-MoE position completion has a null first row");
  }
  validate_spine(spine, first->identity.device());
  const auto hidden = static_cast<std::int64_t>(spine.geometry.hidden);
  const auto routed = static_cast<std::int64_t>(spine.geometry.routed_hidden);
  const auto positions = static_cast<std::int64_t>(prepared_rows.size());
  if (!routed_outputs.defined() ||
      routed_outputs.scalar_type() != at::kFloat ||
      routed_outputs.device() != first->identity.device() ||
      !routed_outputs.is_contiguous() ||
      routed_outputs.sizes() != at::IntArrayRef({positions, routed})) {
    throw std::invalid_argument(
        "routed-MoE position completion output violates its [T,routed_hidden] fp32 contract");
  }

  std::vector<at::Tensor> identities;
  std::vector<at::Tensor> shared_rows;
  identities.reserve(prepared_rows.size());
  shared_rows.reserve(prepared_rows.size());
  bool has_shared = first->shared_output.defined();
  for (const PreparedMoeT1* prepared : prepared_rows) {
    if (prepared == nullptr || prepared->layer_index != spine.layer_index ||
        prepared->spine_generation != spine.generation ||
        prepared->geometry.hidden != spine.geometry.hidden ||
        prepared->geometry.routed_hidden != spine.geometry.routed_hidden ||
        prepared->geometry.intermediate != spine.geometry.intermediate ||
        prepared->geometry.experts != spine.geometry.experts ||
        prepared->geometry.shared_intermediate !=
            spine.geometry.shared_intermediate ||
        !prepared->identity.defined() ||
        prepared->identity.scalar_type() != at::kFloat ||
        prepared->identity.device() != first->identity.device() ||
        !prepared->identity.is_contiguous() ||
        prepared->identity.sizes() != at::IntArrayRef({1, hidden}) ||
        prepared->shared_output.defined() != has_shared) {
      throw std::invalid_argument(
          "routed-MoE position completion rows disagree on their spine/device contract");
    }
    identities.push_back(prepared->identity);
    if (has_shared) {
      if (prepared->shared_output.scalar_type() != at::kFloat ||
          prepared->shared_output.device() != first->identity.device() ||
          !prepared->shared_output.is_contiguous() ||
          prepared->shared_output.sizes() !=
              at::IntArrayRef({1, hidden})) {
        throw std::runtime_error(
            "provider shared-expert position output violated its fp32 shape/device contract");
      }
      shared_rows.push_back(prepared->shared_output);
    }
  }

  record_trace(execution_trace, MoeExecutionStage::RoutedUp);
  const at::Tensor normalized = rms_norm(routed_outputs, spine.routed_norm);
  const at::Tensor routed_full = row_int8_linear(
      normalized, spine.routed_up, spine.packed_int8_qualified);
  record_trace(execution_trace, MoeExecutionStage::Shared);
  const at::Tensor shared_output =
      has_shared
          ? at::cat(shared_rows, 0).contiguous()
          : shared_expert(at::cat(identities, 0).contiguous(), spine)
                .contiguous();
  if (shared_output.scalar_type() != at::kFloat ||
      shared_output.device() != first->identity.device() ||
      shared_output.sizes() != at::IntArrayRef({positions, hidden})) {
    throw std::runtime_error(
        "provider shared-expert position output violated its fp32 shape/device contract");
  }
  record_trace(execution_trace, MoeExecutionStage::Merge);
  return at::add(routed_full, shared_output).contiguous();
}

at::Tensor run_moe_t1(const at::Tensor& hidden, const MoeSpineT1& spine,
                      const CanonicalExpertBatchT1& experts,
                      const MoeRunOptions& options) {
  const PreparedMoeT1 prepared =
      prepare_moe_t1(hidden, spine, options.execution_trace);
  const MoeGeometry& geometry = spine.geometry;
  if (experts.expert_ids.size() != kMoeRouteTopK) {
    throw std::invalid_argument(
        "T=1 expert batch must contain exactly all 16 routed experts");
  }
  const MoeExpertBackend backend = select_backend(
      prepared.routed_input.device(), options.expert_backend,
      options.cuda_cache);
  validate_backend_layout(backend, experts.layout);
  validate_expert_storage(experts.expert_ids, experts.expert_major_bytes,
                          experts.layout, experts.expert_span_bytes, geometry,
                          kMoeRouteTopK);
  const auto blobs = ordered_expert_pointers(
      prepared.route, experts.expert_ids, experts.expert_major_bytes, geometry,
      experts.expert_span_bytes);
  record_trace(options.execution_trace,
               MoeExecutionStage::ExpertBytesBorrowed);
  at::Tensor routed;
  switch (backend) {
    case MoeExpertBackend::CpuMxfp4:
      routed = execute_cpu(prepared, blobs, options, geometry);
      break;
    case MoeExpertBackend::MetalMxfp4:
      routed = execute_metal(prepared, blobs, experts.layout,
                             experts.expert_span_bytes, options, geometry);
      break;
    case MoeExpertBackend::CudaMxfp4:
      if (options.cuda_cache == nullptr) {
        throw std::runtime_error(
            "CUDA MXFP4 was requested without a session-owned expert cache");
      }
      record_trace(options.execution_trace,
                   MoeExecutionStage::CudaExpertDispatch);
      routed = options.cuda_cache->execute_t1(prepared, experts);
      break;
    case MoeExpertBackend::Auto:
      throw std::logic_error("automatic routed-MoE backend remained unresolved");
  }
  return complete_moe_t1(prepared, routed, spine, options.execution_trace);
}

}  // namespace deltafin::provider_internal
