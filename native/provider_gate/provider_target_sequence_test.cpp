#include "provider_target_sequence.h"
#include "provider_kda_batch.h"
#include "provider_abi.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace deltafin::provider_internal;

constexpr std::int64_t kHidden = 32;
constexpr std::int64_t kKdaHeads = 32;
constexpr std::int64_t kKdaHeadWidth = 32;
constexpr std::int64_t kKdaProjection = kKdaHeads * kKdaHeadWidth;
constexpr std::int64_t kKdaConvolution = 4;
constexpr std::int64_t kVocabulary = 64;
constexpr MoeGeometry kMoeGeometry{32, 32, 32, 16, 64};

at::Tensor deterministic_tensor(const at::IntArrayRef shape,
                                const std::int64_t seed,
                                const float scale = 0.0078125F) {
  std::int64_t count = 1;
  for (const std::int64_t extent : shape) {
    count *= extent;
  }
  at::Tensor tensor = at::empty(shape, at::TensorOptions().dtype(at::kFloat));
  float* values = tensor.data_ptr<float>();
  for (std::int64_t index = 0; index < count; ++index) {
    const std::int64_t numerator =
        ((index * 13 + seed * 17 + (index / 11) * 3) % 41) - 20;
    values[index] = static_cast<float>(numerator) * scale;
  }
  return tensor;
}

MoeRowInt8Matrix row_int8(const std::int64_t rows,
                          const std::int64_t columns,
                          const std::int64_t seed,
                          const bool dense_fallback = true) {
  at::Tensor quantized =
      at::empty({rows, columns}, at::TensorOptions().dtype(at::kChar));
  at::Tensor scales = at::empty({rows}, at::TensorOptions().dtype(at::kFloat));
  auto q = quantized.accessor<std::int8_t, 2>();
  auto s = scales.accessor<float, 1>();
  for (std::int64_t row = 0; row < rows; ++row) {
    s[row] = static_cast<float>(1 + ((row + seed) % 3)) / 256.0F;
    for (std::int64_t column = 0; column < columns; ++column) {
      q[row][column] = static_cast<std::int8_t>(
          ((row * 3 + column * 5 + seed * 7) % 15) - 7);
    }
  }
  at::Tensor dense;
  if (dense_fallback) {
    dense = (quantized.to(at::kFloat) * scales.unsqueeze(1)).contiguous();
  }
  return {std::move(quantized), std::move(scales), std::move(dense), {}};
}

MoeRowInt8Matrix original_bf16_linear(const std::int64_t rows,
                                      const std::int64_t columns,
                                      const std::int64_t seed) {
  MoeRowInt8Matrix source = row_int8(rows, columns, seed);
  return {at::Tensor(), at::Tensor(), std::move(source.dense_f32), {}};
}

KdaProjection kda_projection(const std::int64_t rows,
                             const std::int64_t columns,
                             const std::int64_t seed) {
  MoeRowInt8Matrix matrix = original_bf16_linear(rows, columns, seed);
  return {std::move(matrix.dense_f32), at::Tensor()};
}

KdaWeights make_kda_weights() {
  const at::Tensor convolution = deterministic_tensor(
      {kKdaProjection * 3, 1, kKdaConvolution}, 101, 0.001953125F);
  KdaWeights weights{
      .a_log = at::full({kKdaHeadWidth}, -2.0F,
                        at::TensorOptions().dtype(at::kFloat)),
      .dt_bias = deterministic_tensor({kKdaProjection}, 102, 0.0009765625F),
      .query_convolution = convolution.narrow(0, 0, kKdaProjection),
      .key_convolution = convolution.narrow(0, kKdaProjection, kKdaProjection),
      .value_convolution =
          convolution.narrow(0, kKdaProjection * 2, kKdaProjection),
      .output_norm = deterministic_tensor({kKdaHeadWidth}, 103) + 1.0F,
      .query_projection = kda_projection(kKdaProjection, kHidden, 104),
      .key_projection = kda_projection(kKdaProjection, kHidden, 105),
      .value_projection = kda_projection(kKdaProjection, kHidden, 106),
      .recurrent_gate_projection =
          kda_projection(kKdaProjection, kHidden, 107),
      .feature_a_projection = kda_projection(kKdaHeadWidth, kHidden, 108),
      .feature_b_projection =
          kda_projection(kKdaProjection, kKdaHeadWidth, 109),
      .beta_projection = kda_projection(kKdaHeads, kHidden, 110),
      .output_projection = kda_projection(kHidden, kKdaProjection, 111),
  };
  // Mirror the production template arena: all five same-input dense
  // projections are adjacent views, so T-wide preparation must select one
  // zero-copy super-view and one matmul rather than a synthetic row fallback.
  at::Tensor arena = at::cat(
      {weights.query_projection.weight, weights.key_projection.weight,
       weights.value_projection.weight,
       weights.recurrent_gate_projection.weight,
       weights.feature_a_projection.weight},
      0)
                         .contiguous();
  std::int64_t row = 0;
  const auto take = [&](const std::int64_t count) {
    at::Tensor result = arena.narrow(0, row, count);
    row += count;
    return result;
  };
  weights.query_projection.weight = take(kKdaProjection);
  weights.key_projection.weight = take(kKdaProjection);
  weights.value_projection.weight = take(kKdaProjection);
  weights.recurrent_gate_projection.weight = take(kKdaProjection);
  weights.feature_a_projection.weight = take(kKdaHeadWidth);
  return weights;
}

MlaLinearWeight mla_weight(const std::int64_t rows,
                           const std::int64_t columns,
                           const std::int64_t seed) {
  MoeRowInt8Matrix matrix = original_bf16_linear(rows, columns, seed);
  return MlaLinearWeight{
      .encoding = MlaLinearEncoding::DenseF32,
      .data = std::move(matrix.dense_f32),
      .row_scale = at::Tensor(),
      // Explicit: GCC errors on omitted members under -Werror; Clang does not.
      .original_bf16 = {},
  };
}

MlaWeights make_mla_weights(const MlaShape& shape) {
  const std::int64_t query_width = shape.num_heads * shape.query_head_dim();
  const std::int64_t value_width = shape.num_heads * shape.value_head_dim;
  return MlaWeights{
      .query_a = mla_weight(shape.q_lora_rank, shape.hidden_size, 201),
      .query_a_norm = deterministic_tensor({shape.q_lora_rank}, 202) + 1.0F,
      .query_b = mla_weight(query_width, shape.q_lora_rank, 203),
      .key_value_a = mla_weight(
          shape.kv_lora_rank + shape.qk_rope_head_dim, shape.hidden_size, 204),
      .key_value_a_norm =
          deterministic_tensor({shape.kv_lora_rank}, 205) + 1.0F,
      .key_value_b = mla_weight(
          shape.num_heads *
              (shape.qk_nope_head_dim + shape.value_head_dim),
          shape.kv_lora_rank, 206),
      .output_gate = mla_weight(value_width, shape.hidden_size, 207),
      .output = mla_weight(shape.hidden_size, value_width, 208),
  };
}

TargetResidualWeights make_residual_weights() {
  return precompute_target_residual_score_weights(TargetResidualWeights{
      deterministic_tensor({kHidden}, 301) + 1.0F,
      deterministic_tensor({kHidden}, 302) + 1.0F,
      deterministic_tensor({1, kHidden}, 303),
      deterministic_tensor({kHidden}, 304) + 1.0F,
      deterministic_tensor({kHidden}, 305) + 1.0F,
      deterministic_tensor({1, kHidden}, 306),
  });
}

TargetDenseWeights make_dense_weights() {
  return TargetDenseWeights{
      original_bf16_linear(kHidden, kHidden, 401),
      original_bf16_linear(kHidden, kHidden, 402),
      original_bf16_linear(kHidden, kHidden, 403),
      false,
  };
}

MoeSpineT1 make_moe_spine(const std::uint32_t layer_index) {
  return MoeSpineT1{
      .layer_index = layer_index,
      .generation = 5000 + layer_index,
      .geometry = kMoeGeometry,
      .packed_int8_qualified = false,
      .router = original_bf16_linear(kMoeGeometry.experts, kHidden, 501),
      .router_correction_bias =
          at::arange(kMoeGeometry.experts,
                     at::TensorOptions().dtype(at::kFloat)) /
          4096.0F,
      .routed_down =
          original_bf16_linear(kMoeGeometry.routed_hidden, kHidden, 502),
      .routed_norm =
          deterministic_tensor({kMoeGeometry.routed_hidden}, 503) + 1.0F,
      .routed_up =
          original_bf16_linear(kHidden, kMoeGeometry.routed_hidden, 504),
      .shared_gate =
          original_bf16_linear(kMoeGeometry.shared_intermediate, kHidden, 505),
      .shared_up =
          original_bf16_linear(kMoeGeometry.shared_intermediate, kHidden, 506),
      .shared_down =
          original_bf16_linear(kHidden, kMoeGeometry.shared_intermediate, 507),
      // Explicit: GCC errors on omitted members under -Werror; Clang does not.
      .shared_gate_up = {},
  };
}

struct MatrixLayout {
  std::size_t packed_offset = 0;
  std::size_t scale_offset = 0;
  std::size_t rows = 0;
  std::size_t columns = 0;
};

std::array<MatrixLayout, 3> expert_layouts() {
  const auto packed = [](const std::size_t rows, const std::size_t columns) {
    return rows * columns / 2;
  };
  const auto scales = [](const std::size_t rows, const std::size_t columns) {
    return rows * columns / 32;
  };
  const std::size_t p13 =
      packed(kMoeGeometry.intermediate, kMoeGeometry.routed_hidden);
  const std::size_t s13 =
      scales(kMoeGeometry.intermediate, kMoeGeometry.routed_hidden);
  const std::size_t p2 =
      packed(kMoeGeometry.routed_hidden, kMoeGeometry.intermediate);
  const std::size_t s2 =
      scales(kMoeGeometry.routed_hidden, kMoeGeometry.intermediate);
  return {{{0, p13, kMoeGeometry.intermediate, kMoeGeometry.routed_hidden},
           {p13 + s13, p13 + s13 + p2, kMoeGeometry.routed_hidden,
            kMoeGeometry.intermediate},
           {p13 + s13 + p2 + s2, p13 + s13 + p2 + s2 + p13,
            kMoeGeometry.intermediate, kMoeGeometry.routed_hidden}}};
}

void encode_expert(std::span<std::uint8_t> destination,
                   const std::size_t expert) {
  const auto layouts = expert_layouts();
  for (std::size_t matrix = 0; matrix < layouts.size(); ++matrix) {
    const MatrixLayout& layout = layouts[matrix];
    for (std::size_t row = 0; row < layout.rows; ++row) {
      for (std::size_t column = 0; column < layout.columns; column += 2) {
        const auto code = [&](const std::size_t offset) {
          std::uint8_t value = static_cast<std::uint8_t>(
              1 + ((expert * 3 + matrix * 5 + row * 7 +
                    (column + offset) * 11) %
                   4));
          if (((expert + matrix + row + column + offset) & 1U) != 0) {
            value = static_cast<std::uint8_t>(value | 8U);
          }
          return value;
        };
        destination[layout.packed_offset + row * (layout.columns / 2) +
                    column / 2] =
            static_cast<std::uint8_t>(code(0) | (code(1) << 4));
      }
      for (std::size_t group = 0; group < layout.columns / 32; ++group) {
        destination[layout.scale_offset + row * (layout.columns / 32) + group] =
            static_cast<std::uint8_t>(123 +
                                      ((expert + matrix + row + group) % 4));
      }
    }
  }
}

struct Experts {
  std::array<std::uint16_t, kMoeRouteTopK> ids{};
  std::vector<std::uint8_t> bytes;

  Experts() {
    const std::size_t span =
        static_cast<std::size_t>(kMoeGeometry.expert_span_bytes());
    bytes.resize(kMoeRouteTopK * span);
    for (std::size_t expert = 0; expert < kMoeRouteTopK; ++expert) {
      ids[expert] = static_cast<std::uint16_t>(expert);
      encode_expert(std::span<std::uint8_t>(bytes).subspan(expert * span, span),
                    expert);
    }
  }

  CanonicalExpertBatchT1 view() const {
    return CanonicalExpertBatchT1{
        .expert_ids = ids,
        .expert_major_bytes = bytes,
        .layout = MoeExpertLayout::RawV1,
        .expert_span_bytes = kMoeGeometry.expert_span_bytes()};
  }

  std::vector<const std::uint8_t*> span_pointers() const {
    const std::size_t span =
        static_cast<std::size_t>(kMoeGeometry.expert_span_bytes());
    std::vector<const std::uint8_t*> pointers;
    pointers.reserve(ids.size());
    for (std::size_t expert = 0; expert < ids.size(); ++expert) {
      pointers.push_back(bytes.data() + expert * span);
    }
    return pointers;
  }
};

struct Model {
  MlaShape mla_shape = MlaShape::small_canary();
  KdaWeights kda = make_kda_weights();
  MlaWeights mla = make_mla_weights(mla_shape);
  TargetResidualWeights residual = make_residual_weights();
  TargetDenseWeights dense = make_dense_weights();
  MoeRowInt8Matrix embedding = row_int8(kVocabulary, kHidden, 601, false);
  TargetTailWeights tail{
      deterministic_tensor({kHidden}, 602) + 1.0F,
      deterministic_tensor({1, kHidden}, 603),
      deterministic_tensor({kHidden}, 604) + 1.0F,
      original_bf16_linear(kVocabulary, kHidden, 605),
      false,
  };
  std::array<MoeSpineT1, kTargetLayerCount> spines{};
  std::array<TargetLayerBinding, kTargetLayerCount> layers{};

  Model() {
    tail = precompute_target_tail_score_weight(std::move(tail));
    for (std::uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
      TargetLayerBinding binding;
      binding.layer_index = layer;
      binding.residual = &residual;
      if (target_layer_uses_mla(layer)) {
        binding.attention_kind = TargetAttentionKind::Mla;
        binding.mla_weights = &mla;
      } else {
        binding.attention_kind = TargetAttentionKind::Kda;
        binding.kda_weights = &kda;
      }
      if (layer == 0) {
        binding.dense = &dense;
      } else {
        spines[layer] = make_moe_spine(layer);
        binding.moe = &spines[layer];
      }
      layers[layer] = binding;
    }
  }

  at::Tensor hidden(const std::uint32_t token) const {
    return target_embedding_row(token, embedding, false);
  }

  at::Tensor hidden_rows(const std::span<const std::uint32_t> tokens) const {
    std::vector<at::Tensor> rows;
    rows.reserve(tokens.size());
    for (const std::uint32_t token : tokens) {
      rows.push_back(hidden(token));
    }
    return at::cat(rows, 0).contiguous();
  }
};

struct Caches {
  MlaShape shape;
  std::vector<std::unique_ptr<TargetKdaCache>> kda;
  std::vector<std::unique_ptr<MlaCache>> mla;
  std::array<TargetLayerCacheBinding, kTargetLayerCount> bindings{};

  explicit Caches(
      const MlaShape& source_shape,
      const MlaCacheRepresentation representation =
          MlaCacheRepresentation::ExpandedExact)
      : shape(source_shape) {
    kda.reserve(kTargetKdaLayerCount);
    mla.reserve(kTargetMlaLayerCount);
    for (std::uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
      TargetLayerCacheBinding binding;
      binding.layer_index = layer;
      if (target_layer_uses_mla(layer)) {
        binding.attention_kind = TargetAttentionKind::Mla;
        mla.push_back(std::make_unique<MlaCache>(shape, representation));
        binding.mla_cache = mla.back().get();
      } else {
        binding.attention_kind = TargetAttentionKind::Kda;
        auto cache = std::make_unique<TargetKdaCache>();
        cache->layer_index = layer;
        cache->state = zero_small_kda_canary_state(at::Device(at::kCPU));
        kda.push_back(std::move(cache));
        binding.kda_cache = kda.back().get();
      }
      bindings[layer] = binding;
    }
  }

  TargetPositionBindings target_bindings(const Model& model) {
    return TargetPositionBindings{
        .contract = TargetTapeContract::SyntheticK3Schedule,
        .caches = bindings,
        .tail = &model.tail,
    };
  }
};

const MoeRunOptions kCpuOptions{
    .expert_backend = MoeExpertBackend::CpuMxfp4,
    .cpu_threads = 1,
    // This CPU fixture selects no embedded metallib. State it explicitly:
    // GCC errors on omitted members under -Werror; Clang does not.
    .metal_shader_path = {},
};

struct FirstMoeReference {
  at::Tensor router_input;
  at::Tensor routed_input;
  KdaState layer_zero_state;
  KdaState layer_one_state;
  std::vector<MoeRouteT1> routes;
  float minimum_router_order_margin = 0.0F;
};

struct ReferenceKdaLayer {
  std::vector<TargetMlpInput> mlp_inputs;
  TargetMlpRowsInput mlp_rows;
  KdaState final_state;
  std::vector<KdaState> boundaries;
};

ReferenceKdaLayer prepare_reference_kda_layer(
    const at::Tensor& hidden_rows,
    const std::vector<TargetBlockResidual>& residuals,
    const Model& model, const std::uint32_t layer_index,
    const bool live_wide) {
  const std::int64_t positions = hidden_rows.size(0);
  KdaState state = zero_small_kda_canary_state(at::Device(at::kCPU));
  std::vector<TargetMlpInput> mlp_inputs;
  if (live_wide) {
    std::vector<at::Tensor> anchor_rows;
    anchor_rows.reserve(static_cast<std::size_t>(positions));
    for (const TargetBlockResidual& residual : residuals) {
      anchor_rows.push_back(residual.anchors);
    }
    TargetAttentionRowsInput attention = prepare_target_attention_rows(
        hidden_rows, at::cat(anchor_rows, 0).contiguous(), model.residual,
        layer_index, false);
    const at::Tensor& normalized = attention.normalized;
    const KdaBatchInputProjections projected =
        kda_project_inputs_batch(normalized, model.kda, false);
    KdaConvolvedPositions convolved = kda_short_convolve_positions(
        normalized, model.kda, state,
        KdaPreprojectedPositions{projected.query, projected.key,
                                 projected.value},
        false);
    const KdaBatchDependentProjections dependent =
        kda_project_dependent_batch(normalized, model.kda, false);
    KdaPositionsRecurrentResult recurrence =
        kda_recur_convolved_positions(
            normalized, model.kda, state, convolved,
            KdaDependentPositions{dependent.feature_a, dependent.feature_b,
                                  dependent.beta},
            true, false);
    KdaBatchOutputProjection output = kda_finish_output_batch(
        normalized, recurrence.recurrent_output_rows, model.kda, false);
    state = std::move(recurrence.final_state);
    std::vector<KdaState> boundaries = std::move(recurrence.boundaries);
    TargetMlpRowsInput rows = prepare_target_mlp_rows(
        attention, output.output, model.residual, false);
    return ReferenceKdaLayer{
        {}, std::move(rows), std::move(state), std::move(boundaries)};
  } else {
    std::vector<TargetAttentionInput> attention_inputs;
    attention_inputs.reserve(static_cast<std::size_t>(positions));
    std::vector<at::Tensor> normalized_rows;
    normalized_rows.reserve(static_cast<std::size_t>(positions));
    for (std::int64_t row = 0; row < positions; ++row) {
      attention_inputs.push_back(prepare_target_attention(
          hidden_rows.narrow(0, row, 1),
          residuals[static_cast<std::size_t>(row)], model.residual,
          layer_index, false));
      normalized_rows.push_back(attention_inputs.back().normalized);
    }
    const at::Tensor normalized = at::cat(normalized_rows, 0).contiguous();
    mlp_inputs.reserve(static_cast<std::size_t>(positions));
    for (std::int64_t row = 0; row < positions; ++row) {
      KdaDecodeResult decoded = kda_decode_one(
          normalized.narrow(0, row, 1), model.kda, state, false);
      state = decoded.next_state;
      mlp_inputs.push_back(prepare_target_mlp(
          attention_inputs[static_cast<std::size_t>(row)], decoded.output,
          model.residual, false));
    }
  }
  return ReferenceKdaLayer{
      std::move(mlp_inputs), TargetMlpRowsInput{}, std::move(state), {}};
}

FirstMoeReference first_moe_reference(
    const Model& model, const std::span<const std::uint32_t> tokens,
    const bool live_wide) {
  const std::size_t positions = tokens.size();
  at::Tensor hidden_rows = model.hidden_rows(tokens);
  std::vector<TargetBlockResidual> residuals;
  residuals.reserve(positions);
  for (std::size_t row = 0; row < positions; ++row) {
    residuals.push_back(
        empty_target_block_residual(at::Device(at::kCPU), kHidden));
  }

  ReferenceKdaLayer layer_zero = prepare_reference_kda_layer(
      hidden_rows, residuals, model, 0, live_wide);
  std::vector<at::Tensor> dense_inputs;
  if (!live_wide) {
    dense_inputs.reserve(positions);
    for (const TargetMlpInput& input : layer_zero.mlp_inputs) {
      dense_inputs.push_back(input.normalized);
    }
  }
  const at::Tensor dense_input = live_wide
      ? layer_zero.mlp_rows.normalized
      : at::cat(dense_inputs, 0).contiguous();
  at::Tensor dense_output;
  if (live_wide) {
    dense_output = run_target_dense_rows(dense_input, model.dense, false);
  } else {
    std::vector<at::Tensor> output_rows;
    output_rows.reserve(positions);
    for (std::size_t row = 0; row < positions; ++row) {
      output_rows.push_back(run_target_dense(
          dense_input.narrow(0, static_cast<std::int64_t>(row), 1),
          model.dense, false));
    }
    dense_output = at::cat(output_rows, 0).contiguous();
  }
  if (live_wide) {
    hidden_rows = complete_target_layer_rows(
        layer_zero.mlp_rows, dense_output, false);
    for (std::size_t row = 0; row < positions; ++row) {
      residuals[row].anchors = layer_zero.mlp_rows.next_anchors.narrow(
          0, static_cast<std::int64_t>(row), 1);
    }
  } else {
    std::vector<at::Tensor> completed_rows;
    completed_rows.reserve(positions);
    for (std::size_t row = 0; row < positions; ++row) {
      completed_rows.push_back(complete_target_layer(
          layer_zero.mlp_inputs[row],
          dense_output.narrow(0, static_cast<std::int64_t>(row), 1), false));
      residuals[row].anchors = layer_zero.mlp_inputs[row].next_anchors;
    }
    hidden_rows = at::cat(completed_rows, 0).contiguous();
  }

  ReferenceKdaLayer layer_one = prepare_reference_kda_layer(
      hidden_rows, residuals, model, 1, live_wide);
  std::vector<at::Tensor> router_rows;
  if (!live_wide) {
    router_rows.reserve(positions);
    for (const TargetMlpInput& input : layer_one.mlp_inputs) {
      router_rows.push_back(input.normalized);
    }
  }
  const at::Tensor router_input = live_wide
      ? layer_one.mlp_rows.normalized
      : at::cat(router_rows, 0).contiguous();

  // Independent spelling of the live router. This intentionally does not
  // call prepare_moe_positions_t1: it is the oracle for that boundary.
  const MoeSpineT1& spine = model.spines[1];
  const at::Tensor logits = at::matmul(
      router_input, spine.router.dense_f32.transpose(0, 1));
  const at::Tensor scores = at::sigmoid(logits);
  const at::Tensor choice = scores + spine.router_correction_bias;
  const auto [ignored, ids] = at::topk(
      choice, static_cast<std::int64_t>(kMoeRouteTopK), -1, true, false);
  static_cast<void>(ignored);
  const at::Tensor selected = at::gather(scores, 1, ids);
  const at::Tensor weights = selected /
      (at::sum(selected, std::vector<std::int64_t>{-1}, true) + 1.0e-20);
  const at::Tensor ids_cpu = ids.contiguous();
  const at::Tensor weights_cpu = weights.contiguous();
  const auto* id_values = ids_cpu.const_data_ptr<std::int64_t>();
  const auto* weight_values = weights_cpu.const_data_ptr<float>();
  std::vector<MoeRouteT1> routes(positions);
  for (std::size_t row = 0; row < positions; ++row) {
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      const std::size_t index = row * kMoeRouteTopK + edge;
      routes[row].expert_ids[edge] =
          static_cast<std::uint16_t>(id_values[index]);
      routes[row].weight_bits[edge] =
          std::bit_cast<std::uint32_t>(weight_values[index]);
    }
  }

  const at::Tensor sorted_choice =
      std::get<0>(at::sort(choice, -1, true)).to(at::kCPU);
  float minimum_order_margin = std::numeric_limits<float>::infinity();
  const auto sorted = sorted_choice.accessor<float, 2>();
  for (std::int64_t row = 0; row < sorted_choice.size(0); ++row) {
    for (std::int64_t rank = 1; rank < sorted_choice.size(1); ++rank) {
      minimum_order_margin = std::min(
          minimum_order_margin, sorted[row][rank - 1] - sorted[row][rank]);
    }
  }
  const at::Tensor routed_input = at::matmul(
      router_input, spine.routed_down.dense_f32.transpose(0, 1)).contiguous();
  return FirstMoeReference{
      std::move(router_input), std::move(routed_input),
      std::move(layer_zero.final_state), std::move(layer_one.final_state),
      std::move(routes), minimum_order_margin};
}

void require_throws(const std::function<void()>& operation,
                    const char* name) {
  try {
    operation();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(name) + " did not fail closed");
}

void validate_mailbox(const TargetSequenceExpertMailbox& mailbox,
                      const std::uint32_t layer,
                      const std::size_t positions) {
  if (mailbox.layer_index != layer ||
      mailbox.spine_generation != 5000 + layer ||
      mailbox.row_count != positions) {
    throw std::runtime_error("sequence mailbox header is incorrect");
  }
  for (std::size_t row = 0; row < positions; ++row) {
    const TargetSequenceRouteRow& record = mailbox.rows[row];
    if (record.row_index != row || !record.routed_input.defined() ||
        record.routed_input.sizes() != at::IntArrayRef({1, kHidden})) {
      throw std::runtime_error("sequence mailbox lost a routed row");
    }
    std::array<bool, kMoeGeometry.experts> seen{};
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      const std::uint16_t expert = record.route.expert_ids[edge];
      const float weight = std::bit_cast<float>(record.route.weight_bits[edge]);
      if (expert >= seen.size() || seen[expert] || !std::isfinite(weight) ||
          weight < 0.0F) {
        throw std::runtime_error(
            "sequence mailbox changed route order or raw fp32 weights");
      }
      seen[expert] = true;
    }
  }
}

void test_pilot_hint_is_scheduling_only(const Model& model) {
  TargetPilotRoster roster{};
  roster[2].emplace(clone_compact_pilot_router_t1(
      model.spines[2], model.residual.post_attention_norm, false));
  for (std::size_t positions = 1; positions <= 9; ++positions) {
    std::vector<std::uint32_t> tokens(positions);
    for (std::size_t row = 0; row < positions; ++row) {
      tokens[row] = static_cast<std::uint32_t>(43 + row);
    }
    Caches hinted_caches(model.mla_shape);
    Caches control_caches(model.mla_shape);
    TargetPositionBindings hinted_bindings =
        hinted_caches.target_bindings(model);
    hinted_bindings.pilot_routers = &roster;
    TargetSequenceTape hinted(hinted_bindings, model.hidden_rows(tokens),
                              TargetSequenceMode::Prefill);
    TargetSequenceTape control(control_caches.target_bindings(model),
                               model.hidden_rows(tokens),
                               TargetSequenceMode::Prefill);
    if (hinted.take_prefetch_hint().expert_count != 0) {
      throw std::runtime_error(
          "pilot published a hint outside an expert-I/O boundary");
    }
    if (hinted.prepare_layer(model.layers[0]) !=
            TargetSequenceLayerPrepareKind::DenseCompleted ||
        control.prepare_layer(model.layers[0]) !=
            TargetSequenceLayerPrepareKind::DenseCompleted ||
        hinted.prepare_layer(model.layers[1]) !=
            TargetSequenceLayerPrepareKind::ExpertRowsRequired ||
        control.prepare_layer(model.layers[1]) !=
            TargetSequenceLayerPrepareKind::ExpertRowsRequired) {
      throw std::runtime_error("pilot canary did not reach routed layer one");
    }

    const TargetSequenceExpertMailbox hinted_mailbox =
        hinted.expert_mailbox();
    const TargetSequenceExpertMailbox control_mailbox =
        control.expert_mailbox();
    if (hinted_mailbox.layer_index != control_mailbox.layer_index ||
        hinted_mailbox.spine_generation !=
            control_mailbox.spine_generation ||
        hinted_mailbox.row_count != control_mailbox.row_count) {
      throw std::runtime_error("PILOT changed an authoritative mailbox header");
    }
    for (std::size_t row = 0; row < positions; ++row) {
      if (hinted_mailbox.rows[row].route.expert_ids !=
              control_mailbox.rows[row].route.expert_ids ||
          hinted_mailbox.rows[row].route.weight_bits !=
              control_mailbox.rows[row].route.weight_bits ||
          !at::equal(hinted_mailbox.rows[row].routed_input,
                     control_mailbox.rows[row].routed_input)) {
        throw std::runtime_error(
            "multirow PILOT changed an authoritative route or activation");
      }
    }

    const TargetSequencePrefetchHint hint = hinted.take_prefetch_hint();
    if (hint.source_layer != 1 || hint.target_layer != 2 ||
        hint.expert_count != kPilotTopK ||
        !std::is_sorted(hint.expert_ids.begin(),
                        hint.expert_ids.begin() + hint.expert_count)) {
      throw std::runtime_error(
          "multirow PILOT hint is not one canonical next-layer read union");
    }
    const TargetSequenceStats stats = hinted.stats();
    if (stats.pilot_prediction_dispatches != 1 ||
        stats.pilot_prediction_rows != positions ||
        stats.pilot_hint_issues != 1 ||
        stats.pilot_hint_experts != kPilotTopK ||
        stats.pilot_max_union_candidates != kPilotTopK ||
        stats.pilot_score_materializations != 0 ||
        stats.pilot_score_elisions != 1) {
      throw std::runtime_error(
          "multirow PILOT prediction/issue observability is incorrect");
    }
    if (hinted.take_prefetch_hint().expert_count != 0 ||
        control.take_prefetch_hint().expert_count != 0) {
      throw std::runtime_error(
          "pilot hint was duplicated or appeared without a roster");
    }
    hinted.cancel();
    control.cancel();
  }
  std::cout << "pilot_hint.scheduling_only=PASS (T=1..9)\n";
}

std::vector<std::uint32_t> run_sequence(TargetSequenceTape& tape,
                                        const Model& model,
                                        const Experts& experts,
                                        const bool scattered = false,
                                        std::vector<MoeRouteT1>* routes =
                                            nullptr) {
  if (routes != nullptr) {
    routes->assign(tape.position_count() * 92, MoeRouteT1{});
  }
  const std::vector<const std::uint8_t*> span_pointers =
      experts.span_pointers();
  for (std::uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
    const TargetSequenceLayerPrepareKind kind =
        tape.prepare_layer(model.layers[layer]);
    if (layer == 0) {
      if (kind != TargetSequenceLayerPrepareKind::DenseCompleted) {
        throw std::runtime_error("sequence dense layer requested experts");
      }
      continue;
    }
    if (kind != TargetSequenceLayerPrepareKind::ExpertRowsRequired) {
      throw std::runtime_error("sequence MoE layer did not request rows");
    }
    const TargetSequenceExpertMailbox mailbox = tape.expert_mailbox();
    validate_mailbox(mailbox, layer, tape.position_count());
    if (routes != nullptr) {
      for (std::size_t row = 0; row < tape.position_count(); ++row) {
        (*routes)[row * 92 + (layer - 1)] = mailbox.rows[row].route;
      }
    }
    const std::uint64_t generation = mailbox.spine_generation;
    for (std::size_t row = 0; row < tape.position_count();) {
      const std::size_t count = std::min<std::size_t>(
          kMoePositionTileMaxRows, tape.position_count() - row);
      CanonicalExpertPositionTileT1 tile{
          .expert_ids = experts.ids,
          .expert_major_bytes = scattered
              ? std::span<const std::uint8_t>{}
              : std::span<const std::uint8_t>(experts.bytes),
          .layout = MoeExpertLayout::RawV1,
          .expert_span_bytes = kMoeGeometry.expert_span_bytes(),
          .expert_span_pointers = scattered
              ? std::span<const std::uint8_t* const>(span_pointers)
              : std::span<const std::uint8_t* const>{}};
      tape.finish_expert_tile(static_cast<std::uint16_t>(row),
                              static_cast<std::uint16_t>(count), generation,
                              tile, kCpuOptions);
      row += count;
    }
  }
  const std::span<const std::uint32_t> decisions = tape.finish_tail();
  return {decisions.begin(), decisions.end()};
}

struct LiveWideResult {
  std::vector<std::uint32_t> decisions;
  std::vector<MoeRouteT1> routes;
};

std::vector<std::uint32_t> run_sequential(
    Caches& caches, const Model& model, const Experts& experts,
    std::span<const std::uint32_t> tokens,
    std::vector<MoeRouteT1>* routes);

LiveWideResult run_live_wide_reference(
    Caches& caches, const Model& model, const Experts& experts,
    const std::span<const std::uint32_t> tokens,
    const std::size_t publish_positions = 0) {
  const std::size_t positions = tokens.size();
  const std::size_t publish =
      publish_positions == 0 ? positions : publish_positions;
  if (publish == 0 || publish > positions) {
    throw std::invalid_argument(
        "live T-wide reference publication prefix is out of range");
  }
  if (positions == 1) {
    std::vector<MoeRouteT1> routes;
    std::vector<std::uint32_t> decisions =
        run_sequential(caches, model, experts, tokens, &routes);
    return LiveWideResult{std::move(decisions), std::move(routes)};
  }
  at::Tensor hidden_rows = model.hidden_rows(tokens);
  std::vector<TargetBlockResidual> residuals;
  residuals.reserve(positions);
  for (std::size_t row = 0; row < positions; ++row) {
    residuals.push_back(
        empty_target_block_residual(at::Device(at::kCPU), kHidden));
  }
  std::vector<MoeRouteT1> routes(positions * 92);

  for (std::uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
    TargetMlpRowsInput mlp_rows;
    if (target_layer_uses_mla(layer)) {
      std::vector<at::Tensor> anchor_rows;
      anchor_rows.reserve(positions);
      for (const TargetBlockResidual& residual : residuals) {
        anchor_rows.push_back(residual.anchors);
      }
      TargetAttentionRowsInput attention = prepare_target_attention_rows(
          hidden_rows, at::cat(anchor_rows, 0).contiguous(), model.residual,
          layer, false);
      MlaCache& cache = *caches.bindings[layer].mla_cache;
      MlaCacheTransaction transaction(cache, positions);
      MlaPreparedDecode prepared = prepare_mla_positions(
          attention.normalized
              .view({1, static_cast<std::int64_t>(positions), kHidden}),
          model.mla, transaction.working_cache(), nullptr, true, nullptr);
      const at::Tensor output_rows =
          prepared.output.view({static_cast<std::int64_t>(positions),
                                kHidden});
      mlp_rows = prepare_target_mlp_rows(
          attention, output_rows, model.residual, false);
      commit_mla_decode(transaction.working_cache(), prepared);
      transaction.preflight_publish_prefix(publish);
      transaction.publish_prefix_noexcept(publish);
    } else {
      ReferenceKdaLayer prepared = prepare_reference_kda_layer(
          hidden_rows, residuals, model, layer, true);
      TargetKdaCache& cache = *caches.bindings[layer].kda_cache;
      if (prepared.boundaries.size() != positions) {
        throw std::runtime_error(
            "live KDA oracle did not retain every prefix boundary");
      }
      cache.state = prepared.boundaries[publish - 1];
      cache.version += publish;
      mlp_rows = std::move(prepared.mlp_rows);
    }

    const at::Tensor& normalized = mlp_rows.normalized;
    at::Tensor mlp_output;
    if (layer == 0) {
      mlp_output = run_target_dense_rows(normalized, model.dense, false);
    } else {
      PreparedMoePositionsT1 prepared =
          prepare_moe_positions_t1(normalized, model.spines[layer]);
      std::vector<const PreparedMoeT1*> prepared_rows;
      prepared_rows.reserve(positions);
      for (std::size_t row = 0; row < positions; ++row) {
        prepared_rows.push_back(&prepared.rows[row]);
        routes[row * 92 + (layer - 1)] = prepared.rows[row].route;
      }
      const CanonicalExpertPositionTileT1 tile{
          .expert_ids = experts.ids,
          .expert_major_bytes = experts.bytes,
          .layout = MoeExpertLayout::RawV1,
          .expert_span_bytes = kMoeGeometry.expert_span_bytes()};
      const std::span<const PreparedMoeT1* const> prepared_span(
          prepared_rows.data(), prepared_rows.size());
      const at::Tensor routed = execute_routed_moe_positions_t1(
          prepared_span, tile, kCpuOptions);
      mlp_output = complete_moe_positions_t1(
          prepared_span, routed, model.spines[layer]);
    }

    hidden_rows = complete_target_layer_rows(mlp_rows, mlp_output, false);
    for (std::size_t row = 0; row < positions; ++row) {
      residuals[row].anchors = mlp_rows.next_anchors.narrow(
          0, static_cast<std::int64_t>(row), 1);
    }
  }

  std::vector<at::Tensor> anchor_rows;
  anchor_rows.reserve(positions);
  for (const TargetBlockResidual& residual : residuals) {
    anchor_rows.push_back(residual.anchors);
  }
  const at::Tensor logits = finish_target_tail_rows(
      hidden_rows, at::cat(anchor_rows, 0).contiguous(), model.tail, false);
  const at::Tensor tokens_cpu = at::argmax(logits, -1).to(at::kCPU);
  const auto* token_values = tokens_cpu.const_data_ptr<std::int64_t>();
  std::vector<std::uint32_t> decisions(positions);
  for (std::size_t row = 0; row < positions; ++row) {
    decisions[row] = static_cast<std::uint32_t>(token_values[row]);
  }
  return LiveWideResult{std::move(decisions), std::move(routes)};
}

void require_caches_equal(const Caches& actual, const Caches& expected,
                          std::uint64_t expected_positions);

void test_scattered_expert_span_parity(const Model& model,
                                       const Experts& experts) {
  constexpr std::array<std::uint32_t, 2> tokens{13, 17};
  Caches contiguous_caches(model.mla_shape);
  Caches scattered_caches(model.mla_shape);
  TargetSequenceTape contiguous(contiguous_caches.target_bindings(model),
                                model.hidden_rows(tokens),
                                TargetSequenceMode::Verify);
  TargetSequenceTape scattered(scattered_caches.target_bindings(model),
                               model.hidden_rows(tokens),
                               TargetSequenceMode::Verify);
  const std::vector<std::uint32_t> contiguous_decisions =
      run_sequence(contiguous, model, experts);
  const std::vector<std::uint32_t> scattered_decisions =
      run_sequence(scattered, model, experts, true);
  if (scattered_decisions != contiguous_decisions) {
    throw std::runtime_error(
        "scattered expert spans changed authoritative target decisions");
  }
  contiguous.commit_all();
  scattered.commit_all();
  require_caches_equal(scattered_caches, contiguous_caches, tokens.size());
}

std::vector<std::uint32_t> run_sequential(
    Caches& caches, const Model& model, const Experts& experts,
    const std::span<const std::uint32_t> tokens,
    std::vector<MoeRouteT1>* routes = nullptr) {
  std::vector<std::uint32_t> decisions;
  if (routes != nullptr) {
    routes->assign(tokens.size() * 92, MoeRouteT1{});
  }
  for (std::size_t position = 0; position < tokens.size(); ++position) {
    const std::uint32_t token = tokens[position];
    TargetPositionTape tape(caches.target_bindings(model), model.hidden(token));
    if (tape.prepare_layer(model.layers[0]).kind !=
        TargetLayerPrepareKind::DenseCompleted) {
      throw std::runtime_error("sequential dense layer requested experts");
    }
    for (std::uint32_t layer = 1; layer < kTargetLayerCount; ++layer) {
      const TargetLayerPrepareResult prepared =
          tape.prepare_layer(model.layers[layer]);
      if (prepared.kind != TargetLayerPrepareKind::ExpertsRequired) {
        throw std::runtime_error("sequential MoE layer did not request experts");
      }
      if (routes != nullptr) {
        (*routes)[position * 92 + (layer - 1)] = prepared.route.route;
      }
      tape.finish_moe_layer(layer, prepared.route.spine_generation,
                            experts.view(), kCpuOptions);
    }
    decisions.push_back(tape.finish_greedy());
  }
  return decisions;
}

void require_tensor_equal(const at::Tensor& actual,
                          const at::Tensor& expected,
                          const char* name) {
  if (!at::equal(actual, expected)) {
    const double error =
        at::max(at::abs(actual - expected)).item<double>();
    const std::int64_t differing =
        actual.ne(expected).sum().item<std::int64_t>();
    std::ostringstream detail;
    detail << name << " differs, max_abs=" << std::scientific
           << std::setprecision(9) << error
           << " differing=" << differing << '/' << actual.numel();
    throw std::runtime_error(detail.str());
  }
}

// Batched and per-row execution are algebraically equivalent but not
// bit-exact: a batched matmul kernel can choose a different reduction/tiling
// order per row than the unbatched kernel does, producing float32-ULP drift.
void require_tensor_close(const at::Tensor& actual, const at::Tensor& expected,
                          const char* name) {
  if (!at::allclose(actual, expected, /*rtol=*/1.0e-5, /*atol=*/1.0e-6,
                    /*equal_nan=*/true)) {
    const double error =
        at::max(at::abs(actual - expected)).item<double>();
    const std::int64_t differing =
        actual.ne(expected).sum().item<std::int64_t>();
    std::ostringstream detail;
    detail << name << " differs, max_abs=" << std::scientific
           << std::setprecision(9) << error
           << " differing=" << differing << '/' << actual.numel();
    throw std::runtime_error(detail.str());
  }
}

void require_caches_equal(const Caches& actual, const Caches& expected,
                          const std::uint64_t positions) {
  if (actual.kda.size() != expected.kda.size() ||
      actual.mla.size() != expected.mla.size()) {
    throw std::runtime_error("cache fixture counts disagree");
  }
  for (std::size_t index = 0; index < actual.kda.size(); ++index) {
    const TargetKdaCache& left = *actual.kda[index];
    const TargetKdaCache& right = *expected.kda[index];
    if (left.version != positions || right.version != positions) {
      throw std::runtime_error("KDA version did not advance by row count");
    }
    require_tensor_equal(left.state.query_convolution,
                         right.state.query_convolution, "KDA query state");
    require_tensor_equal(left.state.key_convolution,
                         right.state.key_convolution, "KDA key state");
    require_tensor_equal(left.state.value_convolution,
                         right.state.value_convolution, "KDA value state");
    require_tensor_equal(left.state.recurrent, right.state.recurrent,
                         "KDA recurrent state");
  }
  for (std::size_t index = 0; index < actual.mla.size(); ++index) {
    const MlaCache& left = *actual.mla[index];
    const MlaCache& right = *expected.mla[index];
    if (left.version() != positions || right.version() != positions ||
        left.length() != static_cast<std::int64_t>(positions) ||
        right.length() != static_cast<std::int64_t>(positions)) {
      throw std::runtime_error("MLA prefix length/version is incorrect");
    }
    require_tensor_equal(left.committed_keys(), right.committed_keys(),
                         "MLA key cache");
    require_tensor_equal(left.committed_values(), right.committed_values(),
                         "MLA value cache");
  }
}

void require_unpublished(const Caches& caches) {
  for (const auto& cache : caches.kda) {
    if (cache->version != 0) {
      throw std::runtime_error("cancelled sequence published KDA state");
    }
  }
  for (const auto& cache : caches.mla) {
    if (cache->version() != 0 || cache->length() != 0 ||
        cache->has_pending_prepare()) {
      throw std::runtime_error("cancelled sequence published MLA state");
    }
  }
}

void require_unpublished_equal(const Caches& actual,
                               const Caches& expected) {
  require_unpublished(actual);
  require_unpublished(expected);
  if (actual.kda.size() != expected.kda.size() ||
      actual.mla.size() != expected.mla.size()) {
    throw std::runtime_error("unpublished cache fixture counts disagree");
  }
  for (std::size_t index = 0; index < actual.kda.size(); ++index) {
    require_tensor_equal(actual.kda[index]->state.query_convolution,
                         expected.kda[index]->state.query_convolution,
                         "unpublished KDA query state");
    require_tensor_equal(actual.kda[index]->state.key_convolution,
                         expected.kda[index]->state.key_convolution,
                         "unpublished KDA key state");
    require_tensor_equal(actual.kda[index]->state.value_convolution,
                         expected.kda[index]->state.value_convolution,
                         "unpublished KDA value state");
    require_tensor_equal(actual.kda[index]->state.recurrent,
                         expected.kda[index]->state.recurrent,
                         "unpublished KDA recurrent state");
  }
  for (std::size_t index = 0; index < actual.mla.size(); ++index) {
    if (actual.mla[index]->storage_bytes() !=
        expected.mla[index]->storage_bytes()) {
      throw std::runtime_error(
          "unpublished MLA cache changed its base storage");
    }
  }
}

void test_batched_tail_equivalence(const Model& model) {
  constexpr std::int64_t rows = 3;
  const at::Tensor hidden_rows =
      deterministic_tensor({rows, kHidden}, 811).contiguous();
  const at::Tensor anchor_rows =
      deterministic_tensor({rows, 8, kHidden}, 812).contiguous();
  const at::Tensor batched = finish_target_tail_rows(
      hidden_rows, anchor_rows, model.tail, false);
  std::vector<at::Tensor> canonical;
  canonical.reserve(rows);
  for (std::int64_t row = 0; row < rows; ++row) {
    canonical.push_back(finish_target_tail(
        hidden_rows.narrow(0, row, 1),
        TargetBlockResidual{anchor_rows.narrow(0, row, 1)}, model.tail,
        false));
  }
  require_tensor_close(batched, at::cat(canonical, 0).contiguous(),
                       "batched target tail logits");
}

double maximum_absolute_difference(const at::Tensor& left,
                                   const at::Tensor& right) {
  if (left.sizes() != right.sizes()) {
    throw std::runtime_error("numeric comparison shapes disagree");
  }
  return at::max(at::abs(left - right)).item<double>();
}

void test_live_first_route_oracle(const Model& model) {
  constexpr std::array<std::uint32_t, 3> tokens{3, 7, 11};
  const FirstMoeReference live = first_moe_reference(model, tokens, true);
  const FirstMoeReference rowwise =
      first_moe_reference(model, tokens, false);

  Caches caches(model.mla_shape);
  TargetSequenceTape sequence(caches.target_bindings(model),
                              model.hidden_rows(tokens),
                              TargetSequenceMode::Prefill);
  if (sequence.prepare_layer(model.layers[0]) !=
          TargetSequenceLayerPrepareKind::DenseCompleted ||
      sequence.prepare_layer(model.layers[1]) !=
          TargetSequenceLayerPrepareKind::ExpertRowsRequired) {
    throw std::runtime_error(
        "live first-route oracle did not reach the first MoE boundary");
  }
  const TargetSequenceExpertMailbox mailbox = sequence.expert_mailbox();
  if (mailbox.row_count != tokens.size() ||
      live.routes.size() != tokens.size()) {
    throw std::runtime_error("live first-route oracle row count changed");
  }
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    if (mailbox.rows[row].route.expert_ids != live.routes[row].expert_ids ||
        mailbox.rows[row].route.weight_bits != live.routes[row].weight_bits) {
      throw std::runtime_error(
          "target sequence disagrees with the independent live T-wide router oracle at row " +
          std::to_string(row));
    }
    require_tensor_equal(
        mailbox.rows[row].routed_input,
        live.routed_input.narrow(0, static_cast<std::int64_t>(row), 1),
        "live T-wide routed input");
  }

  std::size_t rowwise_id_differences = 0;
  std::size_t rowwise_weight_differences = 0;
  float maximum_route_weight_error = 0.0F;
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      if (live.routes[row].expert_ids[edge] !=
          rowwise.routes[row].expert_ids[edge]) {
        ++rowwise_id_differences;
      }
      const float wide =
          std::bit_cast<float>(live.routes[row].weight_bits[edge]);
      const float one =
          std::bit_cast<float>(rowwise.routes[row].weight_bits[edge]);
      if (live.routes[row].weight_bits[edge] !=
          rowwise.routes[row].weight_bits[edge]) {
        ++rowwise_weight_differences;
      }
      maximum_route_weight_error =
          std::max(maximum_route_weight_error, std::abs(wide - one));
    }
  }
  const double router_input_error = maximum_absolute_difference(
      live.router_input, rowwise.router_input);
  const double layer_zero_conv_error = maximum_absolute_difference(
      live.layer_zero_state.query_convolution,
      rowwise.layer_zero_state.query_convolution);
  const double layer_zero_recurrent_error = maximum_absolute_difference(
      live.layer_zero_state.recurrent, rowwise.layer_zero_state.recurrent);
  const double layer_one_conv_error = maximum_absolute_difference(
      live.layer_one_state.query_convolution,
      rowwise.layer_one_state.query_convolution);
  const double layer_one_recurrent_error = maximum_absolute_difference(
      live.layer_one_state.recurrent, rowwise.layer_one_state.recurrent);
  if (rowwise_id_differences != 0 ||
      !std::isfinite(router_input_error) ||
      !std::isfinite(layer_zero_conv_error) ||
      !std::isfinite(layer_zero_recurrent_error) ||
      !std::isfinite(layer_one_conv_error) ||
      !std::isfinite(layer_one_recurrent_error)) {
    throw std::runtime_error(
        "live T-wide oracle found route-ID drift or non-finite upstream error");
  }
  sequence.cancel();
  std::cout << "live_oracle.first_route=PASS"
            << " rowwise_id_differences=" << rowwise_id_differences
            << " rowwise_weight_bit_differences="
            << rowwise_weight_differences << '/'
            << tokens.size() * kMoeRouteTopK
            << " max_route_weight_abs=" << std::scientific
            << std::setprecision(9) << maximum_route_weight_error
            << " router_input_max_abs=" << router_input_error
            << " layer0_conv_max_abs=" << layer_zero_conv_error
            << " layer0_recurrent_max_abs=" << layer_zero_recurrent_error
            << " layer1_conv_max_abs=" << layer_one_conv_error
            << " layer1_recurrent_max_abs=" << layer_one_recurrent_error
            << " minimum_router_order_margin="
            << live.minimum_router_order_margin << '\n';
}

void test_prefill_equivalence_and_dispatch(const Model& model,
                                           const Experts& experts) {
  constexpr std::array<std::uint32_t, 3> tokens{3, 7, 11};
  Caches sequence_caches(model.mla_shape);
  Caches live_caches(model.mla_shape);
  Caches sequential_caches(model.mla_shape);
  const LiveWideResult live =
      run_live_wide_reference(live_caches, model, experts, tokens);
  TargetSequenceTape sequence(sequence_caches.target_bindings(model),
                              model.hidden_rows(tokens),
                              TargetSequenceMode::Prefill);
  std::vector<MoeRouteT1> sequence_routes;
  std::vector<MoeRouteT1> sequential_routes;
  const auto started = std::chrono::steady_clock::now();
  const std::vector<std::uint32_t> sequence_decisions =
      run_sequence(sequence, model, experts, false, &sequence_routes);
  const auto compute_finished = std::chrono::steady_clock::now();
  const std::vector<std::uint32_t> sequential_decisions =
      run_sequential(sequential_caches, model, experts, tokens,
                     &sequential_routes);
  if (sequence_decisions.size() != 1 || live.decisions.empty() ||
      sequential_decisions.empty() ||
      sequence_decisions.front() != live.decisions.back() ||
      sequence_decisions.front() != sequential_decisions.back()) {
    throw std::runtime_error(
        "layer-major prefill decision differs from live T-wide or T=1 target math");
  }
  if (sequence_routes.size() != live.routes.size() ||
      sequence_routes.size() != sequential_routes.size()) {
    throw std::runtime_error(
        "layer-major prefill route transcript changed its size");
  }
  std::size_t rowwise_id_differences = 0;
  std::size_t rowwise_weight_differences = 0;
  float maximum_rowwise_weight_error = 0.0F;
  for (std::size_t index = 0; index < sequence_routes.size(); ++index) {
    if (sequence_routes[index].expert_ids != live.routes[index].expert_ids ||
        sequence_routes[index].weight_bits != live.routes[index].weight_bits) {
      throw std::runtime_error(
          "layer-major prefill differs from the independent live T-wide route transcript at index " +
          std::to_string(index));
    }
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      if (sequence_routes[index].expert_ids[edge] !=
          sequential_routes[index].expert_ids[edge]) {
        ++rowwise_id_differences;
      }
      const float wide =
          std::bit_cast<float>(sequence_routes[index].weight_bits[edge]);
      const float row =
          std::bit_cast<float>(sequential_routes[index].weight_bits[edge]);
      if (sequence_routes[index].weight_bits[edge] !=
          sequential_routes[index].weight_bits[edge]) {
        ++rowwise_weight_differences;
      }
      maximum_rowwise_weight_error = std::max(
          maximum_rowwise_weight_error, std::abs(wide - row));
    }
  }
  if (rowwise_id_differences != 0) {
    throw std::runtime_error(
        "live T-wide prefill changed an ordered expert ID versus T=1");
  }
  sequence.commit_all();
  require_caches_equal(sequence_caches, live_caches, tokens.size());

  double maximum_kda_convolution_error = 0.0;
  double maximum_kda_recurrent_error = 0.0;
  for (std::size_t index = 0; index < sequence_caches.kda.size(); ++index) {
    maximum_kda_convolution_error = std::max(
        {maximum_kda_convolution_error,
         maximum_absolute_difference(
             sequence_caches.kda[index]->state.query_convolution,
             sequential_caches.kda[index]->state.query_convolution),
         maximum_absolute_difference(
             sequence_caches.kda[index]->state.key_convolution,
             sequential_caches.kda[index]->state.key_convolution),
         maximum_absolute_difference(
             sequence_caches.kda[index]->state.value_convolution,
             sequential_caches.kda[index]->state.value_convolution)});
    maximum_kda_recurrent_error = std::max(
        maximum_kda_recurrent_error,
        maximum_absolute_difference(
            sequence_caches.kda[index]->state.recurrent,
            sequential_caches.kda[index]->state.recurrent));
  }
  double maximum_mla_key_error = 0.0;
  double maximum_mla_value_error = 0.0;
  for (std::size_t index = 0; index < sequence_caches.mla.size(); ++index) {
    maximum_mla_key_error = std::max(
        maximum_mla_key_error,
        maximum_absolute_difference(
            sequence_caches.mla[index]->committed_keys(),
            sequential_caches.mla[index]->committed_keys()));
    maximum_mla_value_error = std::max(
        maximum_mla_value_error,
        maximum_absolute_difference(
            sequence_caches.mla[index]->committed_values(),
            sequential_caches.mla[index]->committed_values()));
  }
  if (!std::isfinite(maximum_kda_convolution_error) ||
      !std::isfinite(maximum_kda_recurrent_error) ||
      !std::isfinite(maximum_mla_key_error) ||
      !std::isfinite(maximum_mla_value_error)) {
    throw std::runtime_error(
        "live T-wide cache drift versus T=1 is non-finite");
  }

  const TargetSequenceStats stats = sequence.stats();
  const std::uint64_t naive_layers = tokens.size() * kTargetLayerCount;
  if (stats.streamed_layer_passes != kTargetLayerCount ||
      stats.streamed_layer_passes >= naive_layers ||
      stats.attention_rows != tokens.size() * kTargetLayerCount ||
      stats.expert_row_requests != tokens.size() * 92 ||
      stats.expert_rows_completed != tokens.size() * 92 ||
      stats.expert_tiles_completed != 92 || stats.tail_rows != 1 ||
      stats.tail_provider_dispatches != 1 ||
      stats.maximum_live_streamed_layers != 1 ||
      stats.maximum_experts_per_request != kMoeRouteTopK ||
      stats.maximum_positions_per_expert_tile != tokens.size() ||
      stats.verify_snapshot_bytes != 0 ||
      stats.dense_mlp_provider_dispatches != 1 ||
      stats.dense_mlp_rows != tokens.size() ||
      stats.kda_input_provider_dispatches !=
          kTargetKdaLayerCount * 3 ||
      stats.kda_input_equivalent_rowwise_dispatches !=
          tokens.size() * kTargetKdaLayerCount * 3 ||
      stats.kda_dependent_provider_dispatches !=
          kTargetKdaLayerCount * 3 ||
      stats.kda_dependent_equivalent_rowwise_dispatches !=
          tokens.size() * kTargetKdaLayerCount * 3 ||
      stats.kda_shortconv_provider_dispatches !=
          kTargetKdaLayerCount * 3 ||
      stats.kda_recurrent_rows !=
          tokens.size() * kTargetKdaLayerCount ||
      stats.kda_output_provider_dispatches != kTargetKdaLayerCount * 2 ||
      stats.kda_output_rows != tokens.size() * kTargetKdaLayerCount ||
      stats.mla_position_provider_dispatches != kTargetMlaLayerCount ||
      stats.mla_position_rows != tokens.size() * kTargetMlaLayerCount ||
      stats.moe_prepare_provider_dispatches != 92 ||
      stats.moe_prepare_rows != tokens.size() * 92 ||
      stats.moe_router_dispatches != 92 ||
      stats.moe_routed_down_dispatches != 92 ||
      stats.moe_shared_dispatches != 92 ||
      stats.moe_route_materializations != 92 ||
      stats.moe_route_host_transfers != 0 ||
      stats.moe_routed_input_host_transfers != 0 ||
      stats.moe_complete_provider_dispatches != 92 ||
      stats.moe_complete_rows != tokens.size() * 92 ||
      stats.moe_routed_up_dispatches != 92) {
    throw std::runtime_error("layer-major dispatch counters are incorrect");
  }
  const double milliseconds = std::chrono::duration<double, std::milli>(
                                  compute_finished - started)
                                  .count();
  std::cout << "prefill rows=" << tokens.size()
            << " streamed_layers=" << stats.streamed_layer_passes
            << " naive_position_major=" << naive_layers
            << " layer_stream_reduction="
            << (1.0 - static_cast<double>(stats.streamed_layer_passes) /
                          static_cast<double>(naive_layers)) *
                   100.0
            << "% bounded_experts=" << stats.maximum_experts_per_request
            << " dense_dispatches="
            << stats.dense_mlp_provider_dispatches << "/"
            << stats.dense_mlp_rows
            << " kda_output_dispatches="
            << stats.kda_output_provider_dispatches << "/"
            << stats.kda_output_rows
            << " mla_dispatches="
            << stats.mla_position_provider_dispatches << "/"
            << stats.mla_position_rows
            << " moe_prepare_dispatches="
            << stats.moe_prepare_provider_dispatches << "/"
            << stats.moe_prepare_rows
            << " moe_complete_dispatches="
            << stats.moe_complete_provider_dispatches << "/"
            << stats.moe_complete_rows
            << " rowwise_weight_bit_differences="
            << rowwise_weight_differences << '/'
            << sequence_routes.size() * kMoeRouteTopK
            << " rowwise_weight_max_abs=" << std::scientific
            << std::setprecision(9) << maximum_rowwise_weight_error
            << " kda_conv_max_abs=" << maximum_kda_convolution_error
            << " kda_recurrent_max_abs=" << maximum_kda_recurrent_error
            << " mla_key_max_abs=" << maximum_mla_key_error
            << " mla_value_max_abs=" << maximum_mla_value_error
            << " cpu_ms=" << milliseconds << '\n';
}

std::uint64_t expected_verify_snapshot_bytes(const std::size_t positions) {
  const std::uint64_t one_kda_state =
      (3ULL * kKdaProjection * kKdaConvolution +
       kKdaHeads * kKdaHeadWidth * kKdaHeadWidth) *
      sizeof(float);
  return one_kda_state * kTargetKdaLayerCount * positions;
}

/* A multi-row verify keeps, per KDA layer, the recurrence inputs that
 * rebuild any accepted prefix: decay/key/value rows, beta, and the three
 * convolution sources. */
std::uint64_t expected_verify_record_bytes(const std::size_t positions) {
  const std::uint64_t rows = positions;
  const std::uint64_t per_layer =
      (3ULL * rows * kKdaProjection + rows * kKdaHeads +
       3ULL * kKdaProjection * (kKdaConvolution - 1 + rows)) *
      sizeof(float);
  return per_layer * kTargetKdaLayerCount;
}

void require_exact_verify_snapshot_accounting(
    const TargetSequenceStats& stats, const std::size_t positions) {
  const std::uint64_t one_generation = expected_verify_snapshot_bytes(1);
  // One row's only boundary is its final state; more rows keep the final
  // state plus a prefix record instead of one state per row.
  const std::uint64_t retained =
      positions == 1 ? one_generation
                     : expected_verify_record_bytes(positions);
  const std::uint64_t staged =
      positions == 1 ? one_generation : retained + one_generation;
  if (stats.positions != positions ||
      stats.verify_snapshot_bytes != retained ||
      stats.staged_kda_storage_bytes != staged) {
    throw std::runtime_error(
        "verify KDA prefix record accounting is not exact");
  }
}

void require_full_commit_only_snapshot_accounting(
    const TargetSequenceStats& stats, const std::size_t positions) {
  const std::uint64_t one_generation = expected_verify_snapshot_bytes(1);
  if (stats.positions != positions || stats.verify_snapshot_bytes != 0 ||
      stats.staged_kda_storage_bytes != one_generation) {
    throw std::runtime_error(
        "full-commit-only Verify did not retain exactly one final KDA generation");
  }
}

void require_routes_equal(const std::vector<MoeRouteT1>& actual,
                          const std::vector<MoeRouteT1>& expected,
                          const char* name) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(name) + " route count changed");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index].expert_ids != expected[index].expert_ids ||
        actual[index].weight_bits != expected[index].weight_bits) {
      throw std::runtime_error(std::string(name) +
                               " route transcript changed at index " +
                               std::to_string(index));
    }
  }
}

void require_full_commit_only_parity(
    const Model& model, const Experts& experts,
    const std::span<const std::uint32_t> candidates) {
  Caches ordinary_caches(model.mla_shape);
  Caches full_only_caches(model.mla_shape);
  TargetSequenceTape ordinary(ordinary_caches.target_bindings(model),
                              model.hidden_rows(candidates),
                              TargetSequenceMode::Verify);
  TargetSequenceTape full_only(full_only_caches.target_bindings(model),
                               model.hidden_rows(candidates),
                               TargetSequenceMode::Verify, false, true);
  std::vector<MoeRouteT1> ordinary_routes;
  std::vector<MoeRouteT1> full_only_routes;
  const std::vector<std::uint32_t> ordinary_decisions =
      run_sequence(ordinary, model, experts, false, &ordinary_routes);
  const std::vector<std::uint32_t> full_only_decisions =
      run_sequence(full_only, model, experts, false, &full_only_routes);
  require_exact_verify_snapshot_accounting(ordinary.stats(),
                                           candidates.size());
  require_full_commit_only_snapshot_accounting(full_only.stats(),
                                               candidates.size());
  if (ordinary_decisions.size() != candidates.size() ||
      full_only_decisions.size() != candidates.size() ||
      full_only_decisions != ordinary_decisions) {
    throw std::runtime_error(
        "full-commit-only Verify did not preserve every target prediction");
  }
  require_routes_equal(full_only_routes, ordinary_routes,
                       "full-commit-only Verify");
  ordinary.commit_all();
  full_only.commit_all();
  require_caches_equal(full_only_caches, ordinary_caches,
                       candidates.size());
}

void test_full_commit_only_verify_contract(const Model& model,
                                           const Experts& experts) {
  constexpr std::array<std::uint32_t, 9> candidates{
      3, 5, 7, 11, 13, 17, 19, 23, 29};
  for (const std::size_t positions : {std::size_t{1}, std::size_t{3},
                                      std::size_t{9}}) {
    require_full_commit_only_parity(
        model, experts,
        std::span<const std::uint32_t>(candidates).first(positions));
  }

  {
    Caches caches(model.mla_shape);
    Caches pristine(model.mla_shape);
    const auto rows = std::span<const std::uint32_t>(candidates).first(3);
    TargetSequenceTape sequence(caches.target_bindings(model),
                                model.hidden_rows(rows),
                                TargetSequenceMode::Verify, false, true);
    static_cast<void>(run_sequence(sequence, model, experts));
    require_throws([&] { sequence.commit_prefix(2); },
                   "partial full-commit-only Verify publication");
    if (sequence.state() != TargetSequenceState::Cancelled) {
      throw std::runtime_error(
          "partial full-commit-only Verify did not cancel its branch");
    }
    require_unpublished_equal(caches, pristine);
  }

  {
    Caches caches(model.mla_shape);
    Caches pristine(model.mla_shape);
    const auto rows = std::span<const std::uint32_t>(candidates).first(3);
    TargetSequenceTape sequence(caches.target_bindings(model),
                                model.hidden_rows(rows),
                                TargetSequenceMode::Verify, false, true);
    static_cast<void>(run_sequence(sequence, model, experts));
    sequence.cancel();
    require_unpublished_equal(caches, pristine);
  }

  {
    Caches caches(model.mla_shape);
    const auto rows = std::span<const std::uint32_t>(candidates).first(1);
    require_throws(
        [&] {
          TargetSequenceTape invalid(caches.target_bindings(model),
                                     model.hidden_rows(rows),
                                     TargetSequenceMode::Prefill, false,
                                     true);
        },
        "full-commit-only Prefill constructor");
  }
}

void test_full_commit_only_begin_abi_gate() {
  static_assert(DELTAFIN_PROVIDER_TARGET_SEQUENCE_CAPTURE_DSPARK_V1 ==
                (1u << 0));
  static_assert(DELTAFIN_PROVIDER_TARGET_SEQUENCE_FULL_COMMIT_ONLY_V1 ==
                (1u << 1));
  std::vector<std::uint16_t> row(7168, 0);
  DeltafinProviderTargetSequenceBeginBf16RequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.data = reinterpret_cast<const std::uint8_t*>(row.data());
  request.byte_length = row.size() * sizeof(std::uint16_t);
  request.positions = 1;

  const auto require_rejected = [&](const std::uint32_t mode,
                                    const std::uint32_t flags,
                                    const char* expected) {
    request.mode = mode;
    request.flags = flags;
    DeltafinProviderTargetSequenceBeginReportV1 report{};
    report.struct_size = sizeof(report);
    std::array<char, 1024> error{};
    if (deltafin_provider_target_sequence_begin_bf16_v1(
            &request, &report, error.data(), error.size()) == 0 ||
        report.sequence != 0 ||
        std::string(error.data()).find(expected) == std::string::npos) {
      throw std::runtime_error(
          "target-sequence full-commit-only ABI gate did not fail closed");
    }
  };
  require_rejected(
      DELTAFIN_PROVIDER_TARGET_SEQUENCE_PREFILL_V1,
      DELTAFIN_PROVIDER_TARGET_SEQUENCE_FULL_COMMIT_ONLY_V1,
      "full-commit-only requires verify mode");
  require_rejected(DELTAFIN_PROVIDER_TARGET_SEQUENCE_VERIFY_V1, 1u << 3,
                   "invalid rows/flags/reserved fields");
  require_rejected(99, 0, "mode is unknown");
}

void test_verify_every_prefix_and_continuation(const Model& model,
                                               const Experts& experts) {
  constexpr std::array<std::uint32_t, 4> candidates{13, 17, 19, 23};
  constexpr std::array<std::size_t, 3> prefixes{1, 2, candidates.size()};
  constexpr std::array<std::uint32_t, 1> continuation{29};

  Caches decision_reference(model.mla_shape);
  const LiveWideResult decision_oracle = run_live_wide_reference(
      decision_reference, model, experts,
      std::span<const std::uint32_t>(candidates));

  for (const std::size_t prefix : prefixes) {
    Caches sequence_caches(model.mla_shape);
    TargetSequenceTape sequence(sequence_caches.target_bindings(model),
                                model.hidden_rows(candidates),
                                TargetSequenceMode::Verify);
    const std::vector<std::uint32_t> sequence_decisions =
        run_sequence(sequence, model, experts);
    require_exact_verify_snapshot_accounting(sequence.stats(),
                                             candidates.size());
    if (sequence_decisions != decision_oracle.decisions) {
      throw std::runtime_error(
          "batched verify tail decisions differ from the live T-wide oracle");
    }

    sequence.commit_prefix(prefix);
    if (sequence.state() != TargetSequenceState::Committed) {
      throw std::runtime_error("verify prefix did not commit atomically");
    }

    Caches prefix_reference(model.mla_shape);
    static_cast<void>(run_live_wide_reference(
        prefix_reference, model, experts,
        std::span<const std::uint32_t>(candidates), prefix));
    require_caches_equal(sequence_caches, prefix_reference, prefix);

    // A cache can have the right visible prefix yet retain a stale version or
    // an over-advanced private branch. The following canonical row must make
    // the same decision and leave the same caches on both sides.
    const std::vector<std::uint32_t> sequence_next = run_sequential(
        sequence_caches, model, experts,
        std::span<const std::uint32_t>(continuation));
    const std::vector<std::uint32_t> reference_next = run_sequential(
        prefix_reference, model, experts,
        std::span<const std::uint32_t>(continuation));
    if (sequence_next != reference_next) {
      throw std::runtime_error(
          "verify prefix changed the following canonical decision");
    }
    require_caches_equal(sequence_caches, prefix_reference, prefix + 1);
  }
}

void test_verify_tail_cancel_is_unpublished(const Model& model,
                                            const Experts& experts) {
  constexpr std::array<std::uint32_t, 3> candidates{31, 37, 41};
  Caches caches(model.mla_shape);
  Caches pristine(model.mla_shape);
  TargetSequenceTape sequence(caches.target_bindings(model),
                              model.hidden_rows(candidates),
                              TargetSequenceMode::Verify);
  static_cast<void>(run_sequence(sequence, model, experts));
  require_exact_verify_snapshot_accounting(sequence.stats(),
                                           candidates.size());
  sequence.cancel();
  if (sequence.state() != TargetSequenceState::Cancelled) {
    throw std::runtime_error("finished verify sequence did not cancel");
  }
  require_unpublished_equal(caches, pristine);
  // Cancellation is deliberately idempotent for RAII cleanup paths.
  sequence.cancel();
  require_unpublished_equal(caches, pristine);
}

void require_dspark_capture_shape(const at::Tensor& captured,
                                  const std::size_t positions) {
  const std::int64_t expected_columns = static_cast<std::int64_t>(
      kDSparkTargetCaptureLayers.size() * kHidden);
  if (!captured.defined() || captured.scalar_type() != at::kBFloat16 ||
      !captured.is_contiguous() || captured.dim() != 2 ||
      captured.size(0) != static_cast<std::int64_t>(positions) ||
      captured.size(1) != expected_columns) {
    throw std::runtime_error(
        "DSpark target capture is not contiguous BF16 [positions,5*hidden]");
  }
}

void test_dspark_target_capture_contract(const Model& model,
                                         const Experts& experts) {
  constexpr std::array<std::uint32_t, 5> expected_layers{2, 23, 47, 71, 89};
  static_assert(kDSparkTargetCaptureLayers == expected_layers);
  constexpr std::array<std::uint32_t, 3> candidates{5, 11, 17};

  Caches wide_caches(model.mla_shape);
  Caches wide_pristine(model.mla_shape);
  TargetSequenceTape wide(wide_caches.target_bindings(model),
                          model.hidden_rows(candidates),
                          TargetSequenceMode::Verify, true);
  static_cast<void>(run_sequence(wide, model, experts));
  const at::Tensor wide_capture = wide.dspark_target_rows();
  require_dspark_capture_shape(wide_capture, candidates.size());
  const at::Tensor repeated_capture = wide.dspark_target_rows();
  if (repeated_capture.const_data_ptr<c10::BFloat16>() !=
      wide_capture.const_data_ptr<c10::BFloat16>()) {
    throw std::runtime_error(
        "DSpark capture extraction copied provider-owned activation storage");
  }

  // Canonical one-row tapes advance the same cache one input at a time. Their
  // five post-layer captures must reconstruct the layer-major wide capture
  // exactly, including BF16 conversion and [row, layer] concatenation order.
  Caches sequential_caches(model.mla_shape);
  std::vector<at::Tensor> sequential_rows;
  sequential_rows.reserve(candidates.size());
  for (const std::uint32_t token : candidates) {
    const std::array<std::uint32_t, 1> one{token};
    TargetSequenceTape row(sequential_caches.target_bindings(model),
                           model.hidden_rows(one),
                           TargetSequenceMode::Prefill, true);
    static_cast<void>(run_sequence(row, model, experts));
    at::Tensor captured = row.dspark_target_rows();
    require_dspark_capture_shape(captured, 1);
    sequential_rows.push_back(std::move(captured));
    row.commit_all();
  }
  require_tensor_equal(wide_capture,
                       at::cat(sequential_rows, 0).contiguous(),
                       "wide DSpark post-layer captures");

  wide.cancel();
  if (wide.state() != TargetSequenceState::Cancelled) {
    throw std::runtime_error("DSpark capture sequence did not cancel");
  }
  require_unpublished_equal(wide_caches, wide_pristine);
  require_throws(
      [&] { static_cast<void>(wide.dspark_target_rows()); },
      "DSpark capture read after cancellation");

  Caches disabled_caches(model.mla_shape);
  Caches disabled_pristine(model.mla_shape);
  TargetSequenceTape disabled(disabled_caches.target_bindings(model),
                              model.hidden_rows(candidates),
                              TargetSequenceMode::Verify);
  static_cast<void>(run_sequence(disabled, model, experts));
  require_throws(
      [&] { static_cast<void>(disabled.dspark_target_rows()); },
      "default-disabled DSpark capture read");
  if (disabled.state() != TargetSequenceState::ReadyToCommit) {
    throw std::runtime_error(
        "disabled DSpark read changed the target transaction state");
  }
  disabled.cancel();
  require_unpublished_equal(disabled_caches, disabled_pristine);
}

void test_invalid_and_double_commit_operations(const Model& model,
                                               const Experts& experts) {
  constexpr std::array<std::uint32_t, 2> candidates{43, 47};
  {
    Caches caches(model.mla_shape);
    Caches pristine(model.mla_shape);
    TargetSequenceTape sequence(caches.target_bindings(model),
                                model.hidden_rows(candidates),
                                TargetSequenceMode::Verify);
    static_cast<void>(run_sequence(sequence, model, experts));
    require_throws(
        [&] { sequence.commit_prefix(candidates.size() + 1); },
        "verify prefix beyond completed rows");
    if (sequence.state() != TargetSequenceState::Cancelled) {
      throw std::runtime_error(
          "invalid verify prefix did not cancel its unpublished branch");
    }
    require_unpublished_equal(caches, pristine);
  }

  {
    Caches caches(model.mla_shape);
    Caches pristine(model.mla_shape);
    TargetSequenceTape sequence(caches.target_bindings(model),
                                model.hidden_rows(candidates),
                                TargetSequenceMode::Prefill);
    static_cast<void>(run_sequence(sequence, model, experts));
    require_throws([&] { sequence.commit_prefix(1); },
                   "partial prefill commit");
    if (sequence.state() != TargetSequenceState::Cancelled) {
      throw std::runtime_error(
          "partial prefill commit did not cancel its unpublished branch");
    }
    require_unpublished_equal(caches, pristine);
  }

  {
    Caches caches(model.mla_shape);
    Caches pristine(model.mla_shape);
    TargetSequenceTape sequence(caches.target_bindings(model),
                                model.hidden_rows(candidates),
                                TargetSequenceMode::Verify);
    static_cast<void>(run_sequence(sequence, model, experts));
    // The provider supports an empty publication for generic transactional
    // cleanup even though the Rust decode engine intentionally never requests
    // it: every live decode transaction has one old pending input to publish.
    sequence.commit_prefix(0);
    if (sequence.state() != TargetSequenceState::Committed) {
      throw std::runtime_error("zero verify prefix did not consume sequence");
    }
    require_unpublished_equal(caches, pristine);
  }

  {
    Caches caches(model.mla_shape);
    TargetSequenceTape sequence(caches.target_bindings(model),
                                model.hidden_rows(candidates),
                                TargetSequenceMode::Verify);
    static_cast<void>(run_sequence(sequence, model, experts));
    sequence.commit_all();

    Caches reference(model.mla_shape);
    static_cast<void>(
        run_live_wide_reference(reference, model, experts, candidates));
    require_caches_equal(caches, reference, candidates.size());
    require_throws([&] { sequence.commit_all(); }, "double verify commit");
    require_throws([&] { sequence.cancel(); }, "cancel after verify commit");
    if (sequence.state() != TargetSequenceState::Committed) {
      throw std::runtime_error(
          "invalid post-commit operation changed committed sequence state");
    }
    require_caches_equal(caches, reference, candidates.size());
  }
}

void test_rollback_and_fail_closed_order(const Model& model,
                                         const Experts& experts) {
  constexpr std::array<std::uint32_t, 9> candidates{
      23, 29, 31, 37, 41, 43, 47, 53, 59};
  for (const std::size_t positions : {std::size_t{1}, std::size_t{3},
                                      std::size_t{9}}) {
    const auto tokens =
        std::span<const std::uint32_t>(candidates).first(positions);
    Caches caches(model.mla_shape);
    TargetSequenceTape sequence(caches.target_bindings(model),
                                model.hidden_rows(tokens),
                                TargetSequenceMode::Verify);
    static_cast<void>(sequence.prepare_layer(model.layers[0]));
    for (std::uint32_t layer = 1; layer < 3; ++layer) {
      static_cast<void>(sequence.prepare_layer(model.layers[layer]));
      const auto ready = sequence.expert_mailbox();
      sequence.finish_expert_tile(
          0, static_cast<std::uint16_t>(positions),
          ready.spine_generation,
          CanonicalExpertPositionTileT1{
              .expert_ids = experts.ids,
              .expert_major_bytes = experts.bytes,
              .layout = MoeExpertLayout::RawV1,
              .expert_span_bytes = kMoeGeometry.expert_span_bytes()},
          kCpuOptions);
    }
    // Layer index 3 is the first MLA layer. It writes candidate K/V rows only
    // into an unpublished cache branch before this deliberate ordering
    // failure. Row one is invalid for T=1 and out of order for T=3/T=9.
    static_cast<void>(sequence.prepare_layer(model.layers[3]));
    const auto mailbox = sequence.expert_mailbox();
    const std::uint64_t generation = mailbox.spine_generation;
    require_throws(
        [&] {
          sequence.finish_expert_row(1, generation, experts.view(),
                                     kCpuOptions);
        },
        "out-of-order expert row");
    if (sequence.state() != TargetSequenceState::Cancelled) {
      throw std::runtime_error(
          "failed T=" + std::to_string(positions) +
          " sequence did not cancel transaction");
    }
    require_unpublished(caches);
  }
}

void test_atomic_stale_preflight(const Model& model,
                                 const Experts& experts) {
  constexpr std::array<std::uint32_t, 1> tokens{31};
  Caches caches(model.mla_shape);
  TargetSequenceTape sequence(caches.target_bindings(model),
                              model.hidden_rows(tokens),
                              TargetSequenceMode::Prefill);
  static_cast<void>(run_sequence(sequence, model, experts));
  TargetKdaCache& late = *caches.kda.back();
  ++late.version;
  require_throws([&] { sequence.commit_all(); }, "stale all-cache preflight");
  if (caches.kda.front()->version != 0) {
    throw std::runtime_error(
        "stale late cache caused a partial early-cache publication");
  }
  for (const auto& cache : caches.mla) {
    if (cache->version() != 0 || cache->length() != 0) {
      throw std::runtime_error(
          "stale all-cache preflight partially published MLA");
    }
  }
}

void test_expanded_cache_budget_is_explicit() {
  MlaCache production(MlaShape::k3());
  if (production.representation() != MlaCacheRepresentation::ExpandedExact ||
      production.bytes_per_position() != 122880 ||
      production.admitted_max_context() != 4369 ||
      production.can_append(4370)) {
    throw std::runtime_error(
        "production expanded MLA cache budget/accounting changed");
  }
  const MlaShape small = MlaShape::small_canary();
  const std::uint64_t one_row =
      static_cast<std::uint64_t>(
          small.num_heads * (small.query_head_dim() + small.value_head_dim)) *
      sizeof(float);
  MlaCache bounded(small, 1.5, one_row);
  require_throws(
      [&] {
        MlaCacheTransaction impossible(bounded, 2);
        static_cast<void>(impossible);
      },
      "expanded MLA budget admission");
}

void test_exact_contract_refuses_compact_cache_state(const Model& model) {
  Caches compact(MlaShape::k3(),
                 MlaCacheRepresentation::CompactLatentF32);
  const TargetPositionBindings bindings{
      .contract = TargetTapeContract::ExactK3,
      .caches = compact.bindings,
      .tail = &model.tail,
  };
  require_throws(
      [&] {
        TargetSequenceTape sequence(bindings, at::Tensor(),
                                    TargetSequenceMode::Prefill);
      },
      "exact target sequence compact MLA cache state");
}

}  // namespace

int main() {
  try {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    Model model;
    Experts experts;
    test_expanded_cache_budget_is_explicit();
    test_exact_contract_refuses_compact_cache_state(model);
    test_pilot_hint_is_scheduling_only(model);
    test_batched_tail_equivalence(model);
    test_live_first_route_oracle(model);
    test_scattered_expert_span_parity(model, experts);
    test_prefill_equivalence_and_dispatch(model, experts);
    test_verify_every_prefix_and_continuation(model, experts);
    test_full_commit_only_verify_contract(model, experts);
    test_full_commit_only_begin_abi_gate();
    test_verify_tail_cancel_is_unpublished(model, experts);
    test_dspark_target_capture_contract(model, experts);
    test_invalid_and_double_commit_operations(model, experts);
    test_rollback_and_fail_closed_order(model, experts);
    test_atomic_stale_preflight(model, experts);
    std::cout << "provider target sequence: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider target sequence: FAIL: " << error.what() << '\n';
    return 1;
  }
}
