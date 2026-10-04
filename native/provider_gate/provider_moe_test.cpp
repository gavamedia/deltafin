#define DELTAFIN_PROVIDER_MOE_TESTING 1
#include "provider_moe.h"
#include "provider_cuda_moe.h"
#include "../../tools/metal_moe_abi.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/add.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/tanh.h>

#if defined(__APPLE__)
#include <torch/mps.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::CanonicalExpertBatchT1;
using deltafin::provider_internal::MoeGeometry;
using deltafin::provider_internal::MoeExecutionStage;
using deltafin::provider_internal::MoeExecutionTrace;
using deltafin::provider_internal::MoeExpertLayout;
using deltafin::provider_internal::MoeRowInt8Matrix;
using deltafin::provider_internal::MoeRunOptions;
using deltafin::provider_internal::MoeSpineT1;
using deltafin::provider_internal::PreparedMoeT1;
using deltafin::provider_internal::kMoeRouteTopK;

constexpr MoeGeometry kTestGeometry{32, 32, 32, 16, 64};

CanonicalExpertBatchT1 raw_batch(
    const std::span<const std::uint16_t> ids,
    const std::span<const std::uint8_t> bytes) {
  return CanonicalExpertBatchT1{
      .expert_ids = ids,
      .expert_major_bytes = bytes,
      .layout = MoeExpertLayout::RawV1,
      .expert_span_bytes = kTestGeometry.expert_span_bytes()};
}

struct MatrixLayout {
  std::size_t packed_offset;
  std::size_t scale_offset;
  std::size_t rows;
  std::size_t columns;
};

std::array<MatrixLayout, 3> layouts(const MoeGeometry& geometry) {
  const auto packed = [](const std::size_t rows, const std::size_t columns) {
    return rows * columns / 2;
  };
  const auto scales = [](const std::size_t rows, const std::size_t columns) {
    return rows * columns / 32;
  };
  const std::size_t p13 = packed(geometry.intermediate,
                                 geometry.routed_hidden);
  const std::size_t s13 = scales(geometry.intermediate,
                                 geometry.routed_hidden);
  const std::size_t p2 = packed(geometry.routed_hidden,
                                geometry.intermediate);
  const std::size_t s2 = scales(geometry.routed_hidden,
                                geometry.intermediate);
  return {{{0, p13, geometry.intermediate, geometry.routed_hidden},
           {p13 + s13, p13 + s13 + p2, geometry.routed_hidden,
            geometry.intermediate},
           {p13 + s13 + p2 + s2, p13 + s13 + p2 + s2 + p13,
            geometry.intermediate, geometry.routed_hidden}}};
}

float e2m1(const std::uint8_t code) {
  constexpr std::array<float, 8> magnitude =
      {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = magnitude[code & 7U];
  return (code & 8U) == 0 ? value : -value;
}

std::uint8_t test_code(const std::size_t expert, const std::size_t matrix,
                       const std::size_t row, const std::size_t column) {
  std::uint8_t magnitude = static_cast<std::uint8_t>(
      1 + ((expert * 3 + matrix * 5 + row * 7 + column * 11) % 4));
  if (((expert + matrix + row + column) & 1U) != 0) {
    magnitude = static_cast<std::uint8_t>(magnitude | 8U);
  }
  return magnitude;
}

void encode_expert(std::span<std::uint8_t> destination,
                   const std::size_t expert, const MoeGeometry& geometry) {
  const auto matrix_layouts = layouts(geometry);
  for (std::size_t matrix = 0; matrix < matrix_layouts.size(); ++matrix) {
    const MatrixLayout& layout = matrix_layouts[matrix];
    for (std::size_t row = 0; row < layout.rows; ++row) {
      for (std::size_t column = 0; column < layout.columns; column += 2) {
        const std::uint8_t low = test_code(expert, matrix, row, column);
        const std::uint8_t high = test_code(expert, matrix, row, column + 1);
        destination[layout.packed_offset + row * (layout.columns / 2) +
                    column / 2] =
            static_cast<std::uint8_t>(low | (high << 4));
      }
      for (std::size_t group = 0; group < layout.columns / 32; ++group) {
        destination[layout.scale_offset + row * (layout.columns / 32) + group] =
            static_cast<std::uint8_t>(123 +
                ((expert + matrix + row + group) % 4));
      }
    }
  }
}

at::Tensor decode_matrix(const std::uint8_t* expert,
                         const MatrixLayout& layout) {
  at::Tensor result = at::empty(
      {static_cast<std::int64_t>(layout.rows),
       static_cast<std::int64_t>(layout.columns)},
      at::TensorOptions().dtype(at::kFloat));
  auto values = result.accessor<float, 2>();
  for (std::size_t row = 0; row < layout.rows; ++row) {
    for (std::size_t column = 0; column < layout.columns; ++column) {
      const std::uint8_t packed =
          expert[layout.packed_offset + row * (layout.columns / 2) + column / 2];
      const std::uint8_t code = (column & 1U) == 0
                                    ? static_cast<std::uint8_t>(packed & 15U)
                                    : static_cast<std::uint8_t>(packed >> 4);
      const std::uint8_t exponent =
          expert[layout.scale_offset + row * (layout.columns / 32) + column / 32];
      values[static_cast<std::int64_t>(row)]
            [static_cast<std::int64_t>(column)] =
          std::ldexp(e2m1(code), static_cast<int>(exponent) - 127);
    }
  }
  return result;
}

MoeRowInt8Matrix make_row_int8(const std::int64_t rows,
                               const std::int64_t columns,
                               const std::int64_t seed) {
  at::Tensor quantized = at::empty(
      {rows, columns}, at::TensorOptions().dtype(at::kChar));
  at::Tensor scales =
      at::empty({rows}, at::TensorOptions().dtype(at::kFloat));
  auto q = quantized.accessor<std::int8_t, 2>();
  auto s = scales.accessor<float, 1>();
  for (std::int64_t row = 0; row < rows; ++row) {
    s[row] = static_cast<float>(1 + ((row + seed) % 3)) / 64.0F;
    for (std::int64_t column = 0; column < columns; ++column) {
      q[row][column] = static_cast<std::int8_t>(
          ((row * 3 + column * 5 + seed * 7) % 9) - 4);
    }
  }
  at::Tensor dense = at::mul(quantized.to(at::kFloat), scales.unsqueeze(1))
                         .contiguous();
  return {std::move(quantized), std::move(scales), std::move(dense), {}};
}

MoeSpineT1 make_spine() {
  MoeSpineT1 spine;
  spine.layer_index = 1;
  spine.generation = 1;
  spine.geometry = kTestGeometry;
  spine.packed_int8_qualified = false;
  spine.router = make_row_int8(16, 32, 1);
  spine.router_correction_bias =
      at::arange(16, at::TensorOptions().dtype(at::kFloat)).mul_(1.0F / 4096.0F);
  spine.routed_down = make_row_int8(32, 32, 2);
  spine.routed_norm =
      at::linspace(0.75, 1.25, 32, at::TensorOptions().dtype(at::kFloat));
  spine.routed_up = make_row_int8(32, 32, 3);
  spine.shared_gate = make_row_int8(64, 32, 4);
  spine.shared_up = make_row_int8(64, 32, 5);
  spine.shared_down = make_row_int8(32, 64, 6);
  return spine;
}

at::Tensor linear_reference(const at::Tensor& input,
                            const MoeRowInt8Matrix& weight) {
  return at::matmul(input, weight.dense_f32.transpose(0, 1));
}

at::Tensor situ_reference(const at::Tensor& gate, const at::Tensor& up) {
  const at::Tensor gate_term = at::mul(
      at::mul(at::tanh(gate / 4.0F), 4.0F), at::sigmoid(gate));
  const at::Tensor up_term = at::mul(at::tanh(up / 25.0F), 25.0F);
  return at::mul(gate_term, up_term);
}

at::Tensor routed_reference(const PreparedMoeT1& prepared,
                            std::span<const std::uint16_t> ids,
                            std::span<const std::uint8_t> bytes) {
  const auto matrix_layouts = layouts(kTestGeometry);
  const std::size_t span =
      static_cast<std::size_t>(kTestGeometry.expert_span_bytes());
  at::Tensor output = at::zeros({1, 32}, at::TensorOptions().dtype(at::kFloat));
  for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
    const auto found =
        std::find(ids.begin(), ids.end(), prepared.route.expert_ids[edge]);
    if (found == ids.end()) {
      throw std::runtime_error("test route expert is missing");
    }
    const std::size_t slot = static_cast<std::size_t>(found - ids.begin());
    const std::uint8_t* expert = bytes.data() + slot * span;
    const at::Tensor w1 = decode_matrix(expert, matrix_layouts[0]);
    const at::Tensor w2 = decode_matrix(expert, matrix_layouts[1]);
    const at::Tensor w3 = decode_matrix(expert, matrix_layouts[2]);
    const at::Tensor gate =
        at::matmul(prepared.routed_input, w1.transpose(0, 1));
    const at::Tensor up =
        at::matmul(prepared.routed_input, w3.transpose(0, 1));
    const at::Tensor hidden = situ_reference(gate, up);
    at::Tensor expert_output = at::matmul(hidden, w2.transpose(0, 1));
    const float weight =
        std::bit_cast<float>(prepared.route.weight_bits[edge]);
    expert_output.mul_(weight);
    output.add_(expert_output);
  }
  return output;
}

at::Tensor complete_reference(const PreparedMoeT1& prepared,
                              const at::Tensor& routed,
                              const MoeSpineT1& spine) {
  const at::Tensor variance =
      at::mean(at::pow(routed, 2), std::vector<std::int64_t>{-1}, true);
  const at::Tensor normalized = at::mul(
      spine.routed_norm,
      at::mul(routed, at::rsqrt(at::add(variance, 1.0e-5F))));
  const at::Tensor routed_full = linear_reference(normalized, spine.routed_up);
  const at::Tensor shared_gate =
      linear_reference(prepared.identity, spine.shared_gate);
  const at::Tensor shared_up =
      linear_reference(prepared.identity, spine.shared_up);
  const at::Tensor shared = linear_reference(
      situ_reference(shared_gate, shared_up), spine.shared_down);
  return at::add(routed_full, shared);
}

double require_close(const at::Tensor& actual, const at::Tensor& expected,
                     const double tolerance, const char* name) {
  const double error =
      at::max(at::abs(actual.to(at::kFloat) - expected.to(at::kFloat)))
          .item<double>();
  if (!(error <= tolerance)) {
    throw std::runtime_error(std::string(name) + " max_abs=" +
                             std::to_string(error));
  }
  return error;
}

void parity_test() {
  MoeSpineT1 spine = make_spine();
  at::Tensor hidden = at::linspace(
      -0.25, 0.375, 32, at::TensorOptions().dtype(at::kFloat));
  hidden = hidden.reshape({1, 32}).contiguous();
  const std::size_t span =
      static_cast<std::size_t>(kTestGeometry.expert_span_bytes());
  std::vector<std::uint16_t> ids(kMoeRouteTopK);
  std::vector<std::uint8_t> bytes(kMoeRouteTopK * span);
  for (std::size_t expert = 0; expert < kMoeRouteTopK; ++expert) {
    ids[expert] = static_cast<std::uint16_t>(expert);
    encode_expert(std::span<std::uint8_t>(bytes).subspan(expert * span, span),
                  expert, kTestGeometry);
  }

  const PreparedMoeT1 prepared =
      deltafin::provider_internal::prepare_moe_t1(hidden, spine);
  const at::Tensor expected_routed = routed_reference(prepared, ids, bytes);
  const at::Tensor expected = complete_reference(prepared, expected_routed, spine);
  MoeRunOptions options;
  options.expert_backend =
      deltafin::provider_internal::MoeExpertBackend::CpuMxfp4;
  options.cpu_threads = 3;
  const at::Tensor actual = deltafin::provider_internal::run_moe_t1(
      hidden, spine, raw_batch(ids, bytes), options);
  const double error =
      require_close(actual, expected, 2.0e-4, "full MoE T=1 parity");
  std::cout << "provider_moe.cpu_reference_max_abs=" << error << '\n';
}

void position_prepare_complete_parity_test() {
  MoeSpineT1 spine = make_spine();
  at::Tensor hidden_rows = at::linspace(
      -0.375, 0.625, 3 * 32,
      at::TensorOptions().dtype(at::kFloat));
  hidden_rows = hidden_rows.reshape({3, 32}).contiguous();

  auto prepared =
      deltafin::provider_internal::prepare_moe_positions_t1(hidden_rows,
                                                            spine);
  if (prepared.rows.size() != 3 || prepared.router_dispatches != 1 ||
      !prepared.routed_inputs.defined() ||
      prepared.routed_inputs.sizes() != at::IntArrayRef({3, 32}) ||
      prepared.routed_inputs.device() != hidden_rows.device() ||
      prepared.routed_down_dispatches != 1 ||
      prepared.shared_dispatches != 0 ||
      prepared.route_materializations != 1 ||
      prepared.route_host_transfers != 0 ||
      prepared.routed_input_host_transfers != 0) {
    throw std::runtime_error(
        "position MoE preparation counters are not batch-scaled");
  }
  const MoeRunOptions position_defaults;
  if (!position_defaults.metal_position_batch ||
      position_defaults.metal_retain_position_outputs_cpu ||
      position_defaults.metal_retain_expert_wrappers) {
    throw std::runtime_error(
        "native Metal defaults lost flat-first/call-scoped lifetime semantics");
  }

  // Independent live-shape oracle: the downloaded prompt/verify router and
  // routed-down projection both consume [T,H] once. Do not use the provider's
  // own preparation helper as its oracle.
  const at::Tensor live_logits = linear_reference(hidden_rows, spine.router);
  const at::Tensor live_scores = at::sigmoid(live_logits);
  const at::Tensor live_choice =
      live_scores + spine.router_correction_bias;
  const auto [ignored, live_ids] = at::topk(
      live_choice, static_cast<std::int64_t>(kMoeRouteTopK), -1, true,
      false);
  static_cast<void>(ignored);
  const at::Tensor live_selected = at::gather(live_scores, 1, live_ids);
  const at::Tensor live_weights = live_selected /
      (at::sum(live_selected, std::vector<std::int64_t>{-1}, true) +
       1.0e-20);
  const at::Tensor live_ids_cpu = live_ids.contiguous();
  const at::Tensor live_weights_cpu = live_weights.contiguous();
  const auto* id_values = live_ids_cpu.const_data_ptr<std::int64_t>();
  const auto* weight_values = live_weights_cpu.const_data_ptr<float>();
  for (std::size_t row = 0; row < prepared.rows.size(); ++row) {
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      const std::size_t index = row * kMoeRouteTopK + edge;
      if (prepared.rows[row].route.expert_ids[edge] !=
              static_cast<std::uint16_t>(id_values[index]) ||
          prepared.rows[row].route.weight_bits[edge] !=
              std::bit_cast<std::uint32_t>(weight_values[index])) {
        throw std::runtime_error(
            "position MoE preparation differs from live T-wide router");
      }
    }
  }
  if (!at::equal(prepared.rows[0].routed_input,
                 linear_reference(hidden_rows, spine.routed_down)
                     .narrow(0, 0, 1)) ||
      !at::equal(prepared.rows[1].routed_input,
                 linear_reference(hidden_rows, spine.routed_down)
                     .narrow(0, 1, 1)) ||
      !at::equal(prepared.rows[2].routed_input,
                 linear_reference(hidden_rows, spine.routed_down)
                     .narrow(0, 2, 1))) {
    throw std::runtime_error(
        "position MoE routed-down differs from live T-wide projection");
  }

  std::vector<PreparedMoeT1> sequential;
  sequential.reserve(3);
  double maximum_rowwise_weight_error = 0.0;
  double maximum_rowwise_input_error = 0.0;
  for (std::int64_t row = 0; row < 3; ++row) {
    sequential.push_back(deltafin::provider_internal::prepare_moe_t1(
        hidden_rows.narrow(0, row, 1), spine));
    if (prepared.rows[static_cast<std::size_t>(row)].route.expert_ids !=
        sequential.back().route.expert_ids) {
      throw std::runtime_error(
          "position MoE preparation changed a canonical route ID");
    }
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      maximum_rowwise_weight_error = std::max(
          maximum_rowwise_weight_error,
          std::abs(static_cast<double>(std::bit_cast<float>(
                       prepared.rows[static_cast<std::size_t>(row)]
                           .route.weight_bits[edge])) -
                   static_cast<double>(std::bit_cast<float>(
                       sequential.back().route.weight_bits[edge]))));
    }
    maximum_rowwise_input_error = std::max(
        maximum_rowwise_input_error,
        at::max(at::abs(
                    prepared.rows[static_cast<std::size_t>(row)].routed_input -
                    sequential.back().routed_input))
            .item<double>());
  }
  if (!std::isfinite(maximum_rowwise_weight_error) ||
      !std::isfinite(maximum_rowwise_input_error)) {
    throw std::runtime_error(
        "position MoE rowwise comparison became non-finite");
  }

  at::Tensor routed_outputs = at::linspace(
      -0.125, 0.25, 3 * 32,
      at::TensorOptions().dtype(at::kFloat));
  routed_outputs = routed_outputs.reshape({3, 32}).contiguous();
  std::array<const PreparedMoeT1*, 3> row_pointers{
      &prepared.rows[0], &prepared.rows[1], &prepared.rows[2]};
  const at::Tensor actual =
      deltafin::provider_internal::complete_moe_positions_t1(
          row_pointers, routed_outputs, spine);
  PreparedMoeT1 live_completion;
  live_completion.identity = hidden_rows;
  const at::Tensor expected =
      complete_reference(live_completion, routed_outputs, spine);
  if (!at::equal(actual, expected)) {
    const double error =
        at::max(at::abs(actual - expected)).item<double>();
    throw std::runtime_error(
        "position MoE completion changed sequential fp32 bits, max_abs=" +
        std::to_string(error));
  }

  auto one = deltafin::provider_internal::prepare_moe_positions_t1(
      hidden_rows.narrow(0, 0, 1).contiguous(), spine);
  const PreparedMoeT1 direct =
      deltafin::provider_internal::prepare_moe_t1(
          hidden_rows.narrow(0, 0, 1).contiguous(), spine);
  if (one.rows.size() != 1 ||
      one.rows[0].route.expert_ids != direct.route.expert_ids ||
      one.rows[0].route.weight_bits != direct.route.weight_bits ||
      !at::equal(one.rows[0].routed_input, direct.routed_input)) {
    throw std::runtime_error(
        "one-position MoE batch wrapper changed the established T=1 path");
  }
  const std::array<const PreparedMoeT1*, 1> one_pointer{&one.rows[0]};
  const at::Tensor one_routed = routed_outputs.narrow(0, 0, 1).contiguous();
  if (!at::equal(
          deltafin::provider_internal::complete_moe_positions_t1(
              one_pointer, one_routed, spine),
          deltafin::provider_internal::complete_moe_t1(
              direct, one_routed, spine))) {
    throw std::runtime_error(
        "one-position MoE completion wrapper changed the established T=1 path");
  }
  std::cout << "provider_moe.position_batch=PASS dispatches=3->1"
            << " rowwise_weight_max_abs="
            << maximum_rowwise_weight_error
            << " rowwise_input_max_abs=" << maximum_rowwise_input_error
            << '\n';
}

void near_tie_router_batch_test() {
  MoeSpineT1 spine = make_spine();
  spine.geometry.experts = 17;
  spine.router = make_row_int8(17, 32, 17);
  spine.router.dense_f32.zero_();
  spine.router_correction_bias = at::empty(
      {17}, at::TensorOptions().dtype(at::kFloat));
  float* bias = spine.router_correction_bias.data_ptr<float>();
  for (std::int64_t expert = 0; expert < 15; ++expert) {
    bias[expert] = 1.0F + static_cast<float>(expert) / 64.0F;
  }
  // Construct the one-ULP margin in the authoritative post-add choice, not
  // merely in the pre-add correction bias: adding sigmoid(0)=0.5 can round a
  // one-ULP bias distinction away at a different exponent.
  constexpr float cutoff_choice = 0.75F;
  const float winning_choice = std::nextafter(
      cutoff_choice, std::numeric_limits<float>::infinity());
  bias[15] = cutoff_choice - 0.5F;
  bias[16] = winning_choice - 0.5F;
  const float cutoff_margin = winning_choice - cutoff_choice;
  if (!(cutoff_margin > 0.0F && cutoff_margin <= 1.0e-7F)) {
    throw std::runtime_error("near-tie fixture lost its one-ULP cutoff");
  }

  at::Tensor hidden_rows = at::linspace(
      -0.25, 0.375, 3 * 32,
      at::TensorOptions().dtype(at::kFloat)).reshape({3, 32}).contiguous();
  const auto wide = deltafin::provider_internal::prepare_moe_positions_t1(
      hidden_rows, spine);
  for (std::int64_t row = 0; row < hidden_rows.size(0); ++row) {
    const PreparedMoeT1 one =
        deltafin::provider_internal::prepare_moe_t1(
            hidden_rows.narrow(0, row, 1).contiguous(), spine);
    const PreparedMoeT1& batch =
        wide.rows[static_cast<std::size_t>(row)];
    if (batch.route.expert_ids != one.route.expert_ids ||
        batch.route.weight_bits != one.route.weight_bits) {
      throw std::runtime_error(
          "T-wide router changed a one-ULP cutoff route");
    }
    const bool kept_sixteen =
        std::find(batch.route.expert_ids.begin(),
                  batch.route.expert_ids.end(), 16) !=
        batch.route.expert_ids.end();
    const bool kept_fifteen =
        std::find(batch.route.expert_ids.begin(),
                  batch.route.expert_ids.end(), 15) !=
        batch.route.expert_ids.end();
    if (!kept_sixteen || kept_fifteen) {
      throw std::runtime_error(
          "near-tie router selected the wrong side of its cutoff");
    }
  }
  std::cout << "provider_moe.near_tie_cutoff=PASS margin="
            << cutoff_margin << '\n';
}

void shared_gate_up_bundle_test() {
  MoeSpineT1 baseline = make_spine();
  baseline.packed_int8_qualified = true;
  // A qualified packed spine publishes exactly one authoritative weight
  // representation.  The ordinary fixture also carries dense reference
  // tensors, so discard those test-only fallbacks before exercising the
  // production packed path.
  for (MoeRowInt8Matrix* matrix :
       {&baseline.router, &baseline.routed_down, &baseline.routed_up,
        &baseline.shared_gate, &baseline.shared_up,
        &baseline.shared_down}) {
    matrix->dense_f32 = at::Tensor();
  }
  MoeSpineT1 bundled = baseline;
  at::Tensor quantized = at::cat(
      {baseline.shared_gate.quantized, baseline.shared_up.quantized}, 0)
                             .contiguous();
  at::Tensor scales = at::cat(
      {baseline.shared_gate.row_scales, baseline.shared_up.row_scales}, 0)
                          .contiguous();
  const std::int64_t rows = baseline.shared_gate.quantized.size(0);
  bundled.shared_gate.quantized = quantized.narrow(0, 0, rows);
  bundled.shared_gate.row_scales = scales.narrow(0, 0, rows);
  bundled.shared_up.quantized = quantized.narrow(0, rows, rows);
  bundled.shared_up.row_scales = scales.narrow(0, rows, rows);
  if (!deltafin::provider_internal::qualify_moe_shared_gate_up(bundled) ||
      !bundled.shared_gate_up.quantized.is_alias_of(quantized) ||
      !bundled.shared_gate_up.row_scales.is_alias_of(scales) ||
      bundled.shared_gate_up.quantized.numel() !=
          baseline.shared_gate.quantized.numel() +
              baseline.shared_up.quantized.numel() ||
      bundled.shared_gate_up.row_scales.numel() !=
          baseline.shared_gate.row_scales.numel() +
              baseline.shared_up.row_scales.numel() ||
      bundled.shared_gate_up_enabled) {
    throw std::runtime_error(
        "shared gate/up bundle was not zero-copy and default-off");
  }
  bundled.shared_gate_up_enabled = true;

  at::Tensor hidden = at::linspace(
      -0.25, 0.375, 32, at::TensorOptions().dtype(at::kFloat));
  hidden = hidden.reshape({1, 32}).contiguous();
  const PreparedMoeT1 prepared =
      deltafin::provider_internal::prepare_moe_t1(hidden, baseline);
  const at::Tensor routed = at::zeros({1, 32}, hidden.options());
  const at::Tensor expected = deltafin::provider_internal::complete_moe_t1(
      prepared, routed, baseline);
  const at::Tensor actual = deltafin::provider_internal::complete_moe_t1(
      prepared, routed, bundled);
  if (!at::equal(actual, expected)) {
    throw std::runtime_error(
        "one-dispatch shared gate/up bundle changed fp32 output bits");
  }

  constexpr int rounds = 200;
  const auto measure = [&](const MoeSpineT1& spine) {
    const auto started = std::chrono::steady_clock::now();
    at::Tensor last;
    for (int round = 0; round < rounds; ++round) {
      last = deltafin::provider_internal::complete_moe_t1(
          prepared, routed, spine);
    }
    static_cast<void>(last.sum().item<float>());
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - started)
        .count() / rounds;
  };
  static_cast<void>(measure(baseline));
  static_cast<void>(measure(bundled));
  const double baseline_us = measure(baseline);
  const double bundled_us = measure(bundled);

  MoeSpineT1 dense_baseline = make_spine();
  MoeSpineT1 dense_bundled = dense_baseline;
  at::Tensor dense_arena = at::cat(
      {dense_baseline.shared_gate.dense_f32,
       dense_baseline.shared_up.dense_f32}, 0).contiguous();
  const std::int64_t dense_rows =
      dense_baseline.shared_gate.dense_f32.size(0);
  dense_bundled.shared_gate.quantized = at::Tensor();
  dense_bundled.shared_gate.row_scales = at::Tensor();
  dense_bundled.shared_up.quantized = at::Tensor();
  dense_bundled.shared_up.row_scales = at::Tensor();
  dense_bundled.shared_gate.dense_f32 =
      dense_arena.narrow(0, 0, dense_rows);
  dense_bundled.shared_up.dense_f32 =
      dense_arena.narrow(0, dense_rows, dense_rows);
  if (!deltafin::provider_internal::qualify_moe_shared_gate_up(
          dense_bundled) ||
      !dense_bundled.shared_gate_up.dense_f32.is_alias_of(dense_arena) ||
      dense_bundled.shared_gate_up.dense_f32.storage_offset() != 0 ||
      dense_bundled.shared_gate_up_enabled) {
    throw std::runtime_error(
        "dense shared gate/up adjacency did not qualify zero-copy/default-off");
  }
  dense_bundled.shared_gate_up_enabled = true;
  const PreparedMoeT1 dense_prepared =
      deltafin::provider_internal::prepare_moe_t1(hidden, dense_baseline);
  const at::Tensor dense_expected =
      deltafin::provider_internal::complete_moe_t1(
          dense_prepared, routed, dense_baseline);
  const at::Tensor dense_actual =
      deltafin::provider_internal::complete_moe_t1(
          dense_prepared, routed, dense_bundled);
  if (!at::equal(dense_actual, dense_expected)) {
    throw std::runtime_error(
        "dense adjacent shared gate/up changed fp32 output bits");
  }
  std::cout << "provider_moe.shared_gate_up_us=" << baseline_us << "->"
            << bundled_us << " dispatches=2->1 dense=PASS\n";
}

#if defined(__APPLE__)
void move_matrix(MoeRowInt8Matrix& matrix, const at::Device& device) {
  matrix.quantized = matrix.quantized.to(device).contiguous();
  matrix.row_scales = matrix.row_scales.to(device).contiguous();
  matrix.dense_f32 = matrix.dense_f32.to(device).contiguous();
}
#endif

bool mps_parity_test() {
#if !defined(__APPLE__)
  return false;
#else
  if (!at::hasMPS()) {
    return false;
  }
  MoeSpineT1 cpu_spine = make_spine();
  at::Tensor hidden_cpu = at::linspace(
      -0.25, 0.375, 32, at::TensorOptions().dtype(at::kFloat));
  hidden_cpu = hidden_cpu.reshape({1, 32}).contiguous();
  const std::size_t span =
      static_cast<std::size_t>(kTestGeometry.expert_span_bytes());
  std::vector<std::uint16_t> ids(kMoeRouteTopK);
  std::vector<std::uint8_t> bytes(kMoeRouteTopK * span);
  for (std::size_t expert = 0; expert < kMoeRouteTopK; ++expert) {
    ids[expert] = static_cast<std::uint16_t>(expert);
    encode_expert(std::span<std::uint8_t>(bytes).subspan(expert * span, span),
                  expert, kTestGeometry);
  }
  MoeRunOptions options;
  options.expert_backend =
      deltafin::provider_internal::MoeExpertBackend::CpuMxfp4;
  options.cpu_threads = 3;
  const at::Tensor expected = deltafin::provider_internal::run_moe_t1(
      hidden_cpu, cpu_spine, raw_batch(ids, bytes), options);

  const at::Device mps(at::kMPS);
  MoeSpineT1 mps_spine = make_spine();
  move_matrix(mps_spine.router, mps);
  mps_spine.router_correction_bias =
      mps_spine.router_correction_bias.to(mps).contiguous();
  move_matrix(mps_spine.routed_down, mps);
  mps_spine.routed_norm = mps_spine.routed_norm.to(mps).contiguous();
  move_matrix(mps_spine.routed_up, mps);
  move_matrix(mps_spine.shared_gate, mps);
  move_matrix(mps_spine.shared_up, mps);
  move_matrix(mps_spine.shared_down, mps);
  const at::Tensor hidden_mps = hidden_cpu.to(mps).contiguous();
  MoeExecutionTrace trace;
  options.execution_trace = &trace;
  const PreparedMoeT1 staged =
      deltafin::provider_internal::prepare_moe_t1(hidden_mps, mps_spine,
                                                  &trace);
  if (staged.routed_input_cpu.defined() || staged.shared_output.defined()) {
    throw std::runtime_error(
        "MPS prepare crossed a deferred host/shared boundary");
  }
  const at::Tensor routed =
      deltafin::provider_internal::execute_routed_moe_t1(
          staged, raw_batch(ids, bytes), options);
  const at::Tensor actual = deltafin::provider_internal::complete_moe_t1(
                                staged, routed, mps_spine, &trace)
                                .to(at::kCPU);
  constexpr std::array<MoeExecutionStage, 9> kExpectedOrder{
      MoeExecutionStage::Router,
      MoeExecutionStage::RoutedDown,
      MoeExecutionStage::RouteMaterialization,
      MoeExecutionStage::ExpertBytesBorrowed,
      MoeExecutionStage::RoutedInputHostMaterialization,
      MoeExecutionStage::CpuExpertDispatch,
      MoeExecutionStage::RoutedUp,
      MoeExecutionStage::Shared,
      MoeExecutionStage::Merge,
  };
  if (trace.count != kExpectedOrder.size()) {
    throw std::runtime_error("MPS demand-I/O trace had wrong stage count");
  }
  for (std::size_t index = 0; index < kExpectedOrder.size(); ++index) {
    if (trace.stages[index] != kExpectedOrder[index]) {
      throw std::runtime_error(
          "MPS routed input crossed the host boundary before expert bytes");
    }
  }
  const double error =
      require_close(actual, expected, 2.0e-3, "MPS/CPU provider MoE parity");
  std::cout << "provider_moe.mps_cpu_max_abs=" << error << '\n';

  // Exercise the packed operation used by the actual non-CPU shared branch.
  // The tiny test geometry's 16-row router is intentionally below MPS's
  // packed N%32 admission rule, so entering through prepare_moe_t1 would fail
  // before reaching the qualifying 64-row shared projections.
  MoeSpineT1 bundled_spine = mps_spine;
  bundled_spine.packed_int8_qualified = true;
  bundled_spine.shared_gate.dense_f32 = at::Tensor();
  bundled_spine.shared_up.dense_f32 = at::Tensor();
  at::Tensor shared_quantized =
      at::cat({mps_spine.shared_gate.quantized,
               mps_spine.shared_up.quantized}, 0)
          .contiguous();
  at::Tensor shared_scales =
      at::cat({mps_spine.shared_gate.row_scales,
               mps_spine.shared_up.row_scales}, 0)
          .contiguous();
  const std::int64_t shared_rows =
      mps_spine.shared_gate.quantized.size(0);
  bundled_spine.shared_gate.quantized =
      shared_quantized.narrow(0, 0, shared_rows);
  bundled_spine.shared_gate.row_scales =
      shared_scales.narrow(0, 0, shared_rows);
  bundled_spine.shared_up.quantized =
      shared_quantized.narrow(0, shared_rows, shared_rows);
  bundled_spine.shared_up.row_scales =
      shared_scales.narrow(0, shared_rows, shared_rows);
  if (!deltafin::provider_internal::qualify_moe_shared_gate_up(
          bundled_spine)) {
    throw std::runtime_error(
        "MPS shared gate/up storage did not qualify as an adjacent super-view");
  }
  bundled_spine.shared_gate_up_enabled = true;
  std::array<at::Tensor, 2> separate_shared;
  at::Tensor bundled_shared;
  const auto run_separate_shared = [&] {
    separate_shared[0] = at::_weight_int8pack_mm(
        hidden_mps, bundled_spine.shared_gate.quantized,
        bundled_spine.shared_gate.row_scales);
    separate_shared[1] = at::_weight_int8pack_mm(
        hidden_mps, bundled_spine.shared_up.quantized,
        bundled_spine.shared_up.row_scales);
  };
  const auto run_bundled_shared = [&] {
    bundled_shared = at::_weight_int8pack_mm(
        hidden_mps, bundled_spine.shared_gate_up.quantized,
        bundled_spine.shared_gate_up.row_scales);
  };
  run_separate_shared();
  run_bundled_shared();
  if (!at::equal(at::cat({separate_shared[0], separate_shared[1]}, 1)
                     .to(at::kCPU),
                 bundled_shared.to(at::kCPU))) {
    throw std::runtime_error(
        "MPS one-dispatch shared gate/up bundle changed fp32 output bits");
  }

  constexpr int rounds = 100;
  const auto measure_shared = [&](const auto& operation,
                                  const bool bundled) {
    const auto started = std::chrono::steady_clock::now();
    for (int round = 0; round < rounds; ++round) {
      operation();
    }
    const at::Tensor& last = bundled ? bundled_shared : separate_shared[1];
    static_cast<void>(last.sum().item<float>());
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - started)
        .count() / rounds;
  };
  static_cast<void>(measure_shared(run_separate_shared, false));
  static_cast<void>(measure_shared(run_bundled_shared, true));
  const double separate_us = measure_shared(run_separate_shared, false);
  const double bundled_us = measure_shared(run_bundled_shared, true);
  std::cout << "provider_moe.shared_gate_up_mps_us=" << separate_us << "->"
            << bundled_us << " dispatches=2->1\n";
  return true;
#endif
}

template <typename Function>
void require_failure(Function&& function, const char* expected) {
  try {
    function();
  } catch (const std::exception& error) {
    if (std::string(error.what()).find(expected) == std::string::npos) {
      throw std::runtime_error(std::string("unexpected failure: ") + error.what());
    }
    return;
  }
  throw std::runtime_error(std::string("expected failure containing: ") + expected);
}

void full_position_union_ceiling_test() {
  static_assert(
      deltafin::provider_internal::kMoePositionTileMaxExperts == 256,
      "16 rows times 16 routed experts must admit a 256-expert union");
  constexpr MoeGeometry geometry{32, 32, 32, 256, 64};
  const std::size_t expert_span =
      static_cast<std::size_t>(geometry.expert_span_bytes());
  std::vector<std::uint16_t> expert_ids(
      deltafin::provider_internal::kMoePositionTileMaxExperts);
  for (std::size_t index = 0; index < expert_ids.size(); ++index) {
    expert_ids[index] = static_cast<std::uint16_t>(index);
  }
  // Allocate exactly the active union, never a fixed structural maximum in
  // production. This focused ceiling case happens to make the two equal.
  std::vector<std::uint8_t> expert_bytes(expert_ids.size() * expert_span, 0);

  std::array<PreparedMoeT1,
             deltafin::provider_internal::kMoePositionTileMaxRows>
      rows{};
  std::array<const PreparedMoeT1*,
             deltafin::provider_internal::kMoePositionTileMaxRows>
      row_pointers{};
  constexpr float route_weight = 1.0F / 16.0F;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    rows[row].layer_index = 1;
    rows[row].spine_generation = 7;
    rows[row].geometry = geometry;
    rows[row].routed_input = at::zeros(
        {1, static_cast<std::int64_t>(geometry.routed_hidden)},
        at::TensorOptions().dtype(at::kFloat));
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      rows[row].route.expert_ids[edge] =
          static_cast<std::uint16_t>(row * kMoeRouteTopK + edge);
      rows[row].route.weight_bits[edge] =
          std::bit_cast<std::uint32_t>(route_weight);
    }
    row_pointers[row] = &rows[row];
  }

  MoeRunOptions options;
  options.expert_backend =
      deltafin::provider_internal::MoeExpertBackend::CpuMxfp4;
  options.cpu_threads = 1;
  const at::Tensor output =
      deltafin::provider_internal::execute_routed_moe_positions_t1(
          row_pointers,
          deltafin::provider_internal::CanonicalExpertPositionTileT1{
              .expert_ids = expert_ids,
              .expert_major_bytes = expert_bytes,
              .layout = MoeExpertLayout::RawV1,
              .expert_span_bytes = expert_span},
          options);
  if (!output.device().is_cpu() || !output.is_contiguous() ||
      output.sizes() != at::IntArrayRef({16, 32}) ||
      !std::all_of(output.const_data_ptr<float>(),
                   output.const_data_ptr<float>() + output.numel(),
                   [](const float value) { return value == 0.0F; })) {
    throw std::runtime_error(
        "full 256-expert position union changed shape/storage/arithmetic");
  }
  std::cout << "provider_moe.full_position_union=PASS experts="
            << expert_ids.size() << " bytes=" << expert_bytes.size() << '\n';
}

void contract_test() {
#if !defined(DELTAFIN_HAVE_CUDA_MOE_V1)
  if (deltafin::provider_internal::cuda_moe_compiled()) {
    throw std::runtime_error(
        "CUDA MXFP4 stub advertised an unlinked native kernel");
  }
#endif
  const MoeGeometry k3 =
      deltafin::provider_internal::k3_moe_geometry();
  if (k3.expert_span_bytes() != K3_RAW_V1_EXPERT_SPAN ||
      K3_SCALE4_V2_EXPERT_SPAN != UINT64_C(17039360) ||
      K3_CAP_RAW_V1 != (UINT64_C(1) << K3_LAYOUT_RAW_V1) ||
      K3_CAP_SCALE4_V2 != (UINT64_C(1) << K3_LAYOUT_SCALE4_V2)) {
    throw std::runtime_error("K3 expert layout ABI contract drifted");
  }

  MoeSpineT1 spine = make_spine();
  at::Tensor hidden = at::zeros({1, 32}, at::TensorOptions().dtype(at::kFloat));
  PreparedMoeT1 prepared =
      deltafin::provider_internal::prepare_moe_t1(hidden, spine);
  const std::size_t span =
      static_cast<std::size_t>(kTestGeometry.expert_span_bytes());
  std::vector<std::uint16_t> ids(kMoeRouteTopK);
  for (std::size_t index = 0; index < ids.size(); ++index) {
    ids[index] = static_cast<std::uint16_t>(index);
  }
  std::vector<std::uint8_t> bytes(kMoeRouteTopK * span);
  MoeRunOptions options;
  options.expert_backend =
      deltafin::provider_internal::MoeExpertBackend::CpuMxfp4;

  std::swap(ids[0], ids[1]);
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::run_moe_t1(
            hidden, spine, raw_batch(ids, bytes), options));
      },
      "strictly ascending");
  std::swap(ids[0], ids[1]);
  bytes.pop_back();
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::run_moe_t1(
            hidden, spine, raw_batch(ids, bytes), options));
      },
      "buffer length");

  const std::vector<std::uint8_t> complete(kMoeRouteTopK * span);
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            prepared,
            CanonicalExpertBatchT1{
                .expert_ids = ids,
                .expert_major_bytes = complete,
                .layout = MoeExpertLayout::RawV1,
                .expert_span_bytes = span + 1},
            options));
      },
      "explicit storage layout");

  PreparedMoeT1 k3_prepared;
  k3_prepared.geometry = k3;
  k3_prepared.routed_input = at::zeros(
      {1, static_cast<std::int64_t>(k3.routed_hidden)},
      at::TensorOptions().dtype(at::kFloat));
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            k3_prepared,
            CanonicalExpertBatchT1{
                .expert_ids = ids,
                .expert_major_bytes = std::span<const std::uint8_t>{},
                .layout = MoeExpertLayout::Scale4V2,
                .expert_span_bytes = K3_SCALE4_V2_EXPERT_SPAN},
            options));
      },
      "qualified only for Metal");

  options.expert_backend =
      deltafin::provider_internal::MoeExpertBackend::MetalMxfp4;
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            k3_prepared,
            CanonicalExpertBatchT1{
                .expert_ids = std::span<const std::uint16_t>(ids).first(
                    kMoeRouteTopK - 1),
                .expert_major_bytes = std::span<const std::uint8_t>{},
                .layout = MoeExpertLayout::Scale4V2,
                .expert_span_bytes = K3_SCALE4_V2_EXPERT_SPAN},
            options));
      },
      "exactly all 16");
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            prepared,
            CanonicalExpertBatchT1{
                .expert_ids = ids,
                .expert_major_bytes = std::span<const std::uint8_t>{},
                .layout = MoeExpertLayout::Scale4V2,
                .expert_span_bytes = K3_SCALE4_V2_EXPERT_SPAN},
            options));
      },
      "exact K3 geometry");

  std::swap(ids[0], ids[1]);
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            k3_prepared,
            CanonicalExpertBatchT1{
                .expert_ids = ids,
                .expert_major_bytes = std::span<const std::uint8_t>{},
                .layout = MoeExpertLayout::Scale4V2,
                .expert_span_bytes = K3_SCALE4_V2_EXPERT_SPAN},
            options));
      },
      "strictly ascending");
  std::swap(ids[0], ids[1]);
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            k3_prepared,
            CanonicalExpertBatchT1{
                .expert_ids = ids,
                .expert_major_bytes = std::span<const std::uint8_t>{},
                .layout = MoeExpertLayout::Scale4V2,
                .expert_span_bytes = K3_SCALE4_V2_EXPERT_SPAN + 1},
            options));
      },
      "explicit storage layout");
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            k3_prepared,
            CanonicalExpertBatchT1{
                .expert_ids = ids,
                .expert_major_bytes = std::span<const std::uint8_t>{},
                .layout = MoeExpertLayout::Scale4V2,
                .expert_span_bytes = K3_SCALE4_V2_EXPERT_SPAN},
            options));
      },
      "buffer length");

  options.expert_backend =
      deltafin::provider_internal::MoeExpertBackend::CpuMxfp4;

  prepared.route.expert_ids[1] = prepared.route.expert_ids[0];
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            prepared, raw_batch(ids, complete), options));
      },
      "repeats an expert");

  prepared = deltafin::provider_internal::prepare_moe_t1(hidden, spine);
  options.expert_backend =
      deltafin::provider_internal::MoeExpertBackend::CudaMxfp4;
  require_failure(
      [&] {
        static_cast<void>(deltafin::provider_internal::execute_routed_moe_t1(
            prepared, raw_batch(ids, complete), options));
      },
      "session-owned expert cache");
}

}  // namespace

#if defined(__APPLE__)
/*
 * Expert early drain must be packaging, not arithmetic.
 *
 * The staged path runs each expert's GLU/W2 as its bytes arrive, in whatever
 * order storage delivered them, spread across however many command buffers
 * that took; the established path runs the whole layer at once. If those two
 * can differ by a single bit, decode output becomes a function of disk timing.
 * This drives the bridge directly with exact K3 raw-v1 spans and demands
 * byte-identical results.
 */
constexpr int kStagedEdges = 16;
constexpr std::size_t kStagedHidden = 3584;

void fill_staged_expert(std::vector<std::uint8_t>& blob, const std::size_t seed) {
  constexpr std::uint32_t packed_bytes = 5505024;
  constexpr std::uint32_t scale_bytes = 344064;
  std::uint64_t state = 0x9E3779B97F4A7C15ULL * (seed + 1);
  for (std::size_t index = 0; index < blob.size(); ++index) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    blob[index] = static_cast<std::uint8_t>(state >> 24);
  }
  // Constrain only the three scale planes. An unbounded MXFP4 exponent decodes
  // to inf, and a comparison against inf/NaN would prove nothing.
  const std::array<std::uint32_t, 3> scale_offsets = {
      packed_bytes, 2 * packed_bytes + scale_bytes,
      3 * packed_bytes + 2 * scale_bytes};
  for (const std::uint32_t offset : scale_offsets) {
    for (std::uint32_t index = 0; index < scale_bytes; ++index) {
      blob[offset + index] =
          static_cast<std::uint8_t>(120 + (blob[offset + index] % 11U));
    }
  }
}

bool metal_staged_equivalence_test() {
  const auto capabilities =
      deltafin::provider_internal::qualify_metal_expert_layouts(
          K3_METAL_MOE_EMBEDDED_SOURCE_V1);
  if (capabilities.descriptor_abi != K3_DESCRIPTOR_ABI_V1 ||
      (capabilities.layout_capabilities & K3_CAP_RAW_V1) == 0 ||
      capabilities.raw_span_bytes != K3_RAW_V1_EXPERT_SPAN) {
    return false;
  }
  std::vector<std::vector<std::uint8_t>> blobs(
      kStagedEdges, std::vector<std::uint8_t>(K3_RAW_V1_EXPERT_SPAN));
  std::vector<K3MetalExpertDescriptorV1> descriptors(kStagedEdges);
  for (std::size_t edge = 0; edge < blobs.size(); ++edge) {
    fill_staged_expert(blobs[edge], edge);
    descriptors[edge] = K3MetalExpertDescriptorV1{
        .abi_version = K3_DESCRIPTOR_ABI_V1,
        .struct_bytes = sizeof(K3MetalExpertDescriptorV1),
        .layout_id = K3_LAYOUT_RAW_V1,
        .reserved = 0,
        .blob_bytes = K3_RAW_V1_EXPERT_SPAN,
        .blob = blobs[edge].data()};
  }
  std::array<float, kStagedHidden> input{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>((index % 17)) * 0.031F - 0.25F;
  }
  std::array<float, kStagedEdges> weights{};
  for (std::size_t edge = 0; edge < weights.size(); ++edge) {
    weights[edge] = 0.02F + static_cast<float>(edge) * 0.011F;
  }

  std::array<float, kStagedHidden> reference{};
  if (k3_metal_moe_layer_desc_v1(descriptors.data(), kStagedEdges,
                                 weights.data(), input.data(),
                                 reference.data()) != 0) {
    throw std::runtime_error("established Metal layer call failed");
  }
  for (const float value : reference) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "staged-equivalence fixture produced a non-finite reference");
    }
  }

  // Arrival order deliberately bears no relation to route order, and the waves
  // are uneven, so a reduction that followed arrival could not survive this.
  const std::array<std::array<int, 4>, 3> waves = {
      std::array<int, 4>{9, 2, 15, 7}, std::array<int, 4>{0, 11, 4, 13},
      std::array<int, 4>{6, 1, 8, 3}};
  for (const std::size_t staged_waves : {std::size_t{1}, std::size_t{3}}) {
    const std::uint64_t token =
        k3_metal_moe_stage_begin_v1(kStagedEdges, input.data());
    if (token == 0) {
      throw std::runtime_error("Metal expert staging refused to begin");
    }
    for (std::size_t wave = 0; wave < staged_waves; ++wave) {
      std::array<K3MetalExpertDescriptorV1, 4> staged_descriptors{};
      for (std::size_t slot = 0; slot < waves[wave].size(); ++slot) {
        staged_descriptors[slot] = descriptors[waves[wave][slot]];
      }
      if (k3_metal_moe_stage_edges_v1(token, staged_descriptors.data(),
                                      waves[wave].data(),
                                      static_cast<int>(waves[wave].size())) !=
          0) {
        throw std::runtime_error("Metal expert staging rejected a wave");
      }
    }
    std::array<float, kStagedHidden> staged{};
    if (k3_metal_moe_stage_finish_v1(token, descriptors.data(), kStagedEdges,
                                     weights.data(), input.data(),
                                     staged.data()) != 0) {
      throw std::runtime_error("staged Metal layer finish failed");
    }
    if (std::memcmp(reference.data(), staged.data(),
                    sizeof(float) * kStagedHidden) != 0) {
      throw std::runtime_error(
          "expert early drain changed the layer output; reduction or "
          "arithmetic is no longer independent of arrival order");
    }
  }

  // An abandoned layer must leave nothing behind that a later layer could
  // mistake for its own staged work.
  const std::uint64_t abandoned =
      k3_metal_moe_stage_begin_v1(kStagedEdges, input.data());
  if (abandoned == 0) {
    throw std::runtime_error("Metal expert staging refused a second layer");
  }
  std::array<K3MetalExpertDescriptorV1, 4> abandoned_descriptors{};
  for (std::size_t slot = 0; slot < waves[0].size(); ++slot) {
    abandoned_descriptors[slot] = descriptors[waves[0][slot]];
  }
  if (k3_metal_moe_stage_edges_v1(abandoned, abandoned_descriptors.data(),
                                  waves[0].data(),
                                  static_cast<int>(waves[0].size())) != 0) {
    throw std::runtime_error("Metal expert staging rejected an abandoned wave");
  }
  k3_metal_moe_stage_abandon_v1(abandoned);
  std::array<float, kStagedHidden> after_abandon{};
  if (k3_metal_moe_stage_finish_v1(abandoned, descriptors.data(), kStagedEdges,
                                   weights.data(), input.data(),
                                   after_abandon.data()) != 0) {
    throw std::runtime_error("finish after abandon failed");
  }
  if (std::memcmp(reference.data(), after_abandon.data(),
                  sizeof(float) * kStagedHidden) != 0) {
    throw std::runtime_error("a finish on an abandoned token was not exact");
  }
  return true;
}
#endif

int main() {
  try {
    parity_test();
    position_prepare_complete_parity_test();
    near_tie_router_batch_test();
    shared_gate_up_bundle_test();
    full_position_union_ceiling_test();
    contract_test();
    const bool mps = mps_parity_test();
#if defined(__APPLE__)
    const bool staged = metal_staged_equivalence_test();
#else
    const bool staged = false;
#endif
    std::cout << "provider_moe.parity=PASS\n"
              << "provider_moe.mps_parity=" << (mps ? "PASS" : "SKIP") << '\n'
              << "provider_moe.staged_expert_equivalence="
              << (staged ? "PASS" : "SKIP") << '\n'
              << "provider_moe.full_k3_span=17547264\n"
              << "provider_moe.python_runtime=ABSENT\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_moe=FAIL: " << error.what() << '\n';
    return 1;
  }
}
