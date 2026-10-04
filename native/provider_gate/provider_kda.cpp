#include "provider_kda.h"

#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/linear.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kK3Batch = 1;
constexpr std::int64_t kK3Hidden = 7168;
constexpr std::int64_t kK3Heads = 96;
constexpr std::int64_t kK3HeadWidth = 128;
constexpr std::int64_t kK3Projection = kK3Heads * kK3HeadWidth;
constexpr std::int64_t kK3ConvolutionWidth = 4;
constexpr double kK3RmsEpsilon = 1.0e-5;
constexpr double kK3NormalizeEpsilon = 1.0e-12;
constexpr double kK3GateLowerBound = -5.0;
constexpr std::int64_t kCanaryHidden = 32;
constexpr std::int64_t kCanaryHeads = 32;
constexpr std::int64_t kCanaryHeadWidth = 32;
constexpr std::int64_t kCanaryProjection =
    kCanaryHeads * kCanaryHeadWidth;
constexpr std::int64_t kCanaryConvolutionWidth = 4;

struct KdaShape {
  std::int64_t batch = 0;
  std::int64_t hidden = 0;
  std::int64_t heads = 0;
  std::int64_t head_width = 0;
  std::int64_t projection = 0;
  std::int64_t convolution_width = 0;
};

enum class KdaDispatchPolicy {
  Established,
  FusedWhenEligible,
};

struct KdaShortConvolution3 {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  at::Tensor next_query;
  at::Tensor next_key;
  at::Tensor next_value;
};

void require_tensor(const at::Tensor& tensor, const at::Device& device,
                    const at::ScalarType scalar_type,
                    const at::IntArrayRef shape, const char* name) {
  if (!tensor.defined() || tensor.device() != device ||
      tensor.scalar_type() != scalar_type || tensor.sizes() != shape) {
    throw std::invalid_argument(std::string("KDA ") + name +
                                " does not match its device/dtype/shape contract");
  }
}

void require_projection(const KdaProjection& projection,
                        const at::Device& device, const std::int64_t rows,
                        const std::int64_t columns, const char* name) {
  if (projection.original_bf16.defined()) {
    if (projection.weight.defined() || projection.scale.defined() ||
        !original_bf16_matrix_matches(
            projection.original_bf16, device,
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(columns))) {
      throw std::invalid_argument(std::string("KDA ") + name +
                                  " has an invalid original-BF16 carrier");
    }
    return;
  }
  if (projection.scale.defined()) {
    require_tensor(projection.weight, device, at::kChar, {rows, columns}, name);
    require_tensor(projection.scale, device, at::kFloat, {rows}, name);
    return;
  }
  // Dense-fp32 remains an explicit synthetic-oracle arm. An absent scale is
  // never enough to reinterpret canonical original BF16 as dense storage.
  require_tensor(projection.weight, device, at::kFloat, {rows, columns}, name);
}

KdaShape validate_contract(const at::Tensor& hidden,
                           const KdaWeights& weights,
                           const KdaState& state, const bool exact_k3) {
  if (!hidden.defined() || hidden.dim() != 2 || hidden.size(0) != 1 ||
      hidden.scalar_type() != at::kFloat) {
    throw std::invalid_argument(
        "KDA decode requires one [1, hidden] fp32 position");
  }
  KdaShape shape;
  shape.batch = hidden.size(0);
  shape.hidden = hidden.size(1);
  if (!weights.output_norm.defined() || weights.output_norm.dim() != 1) {
    throw std::invalid_argument("KDA output norm must be rank one");
  }
  shape.head_width = weights.output_norm.numel();
  const std::int64_t beta_rows = weights.beta_projection.original_bf16.defined()
      ? static_cast<std::int64_t>(weights.beta_projection.original_bf16.rows)
      : (weights.beta_projection.weight.dim() == 2
             ? weights.beta_projection.weight.size(0)
             : 0);
  if (beta_rows <= 0) {
    throw std::invalid_argument("KDA beta projection must be rank two");
  }
  shape.heads = beta_rows;
  if (shape.heads <= 0 || shape.head_width <= 0 ||
      shape.heads > std::numeric_limits<std::int64_t>::max() /
                        shape.head_width) {
    throw std::invalid_argument("KDA head dimensions are invalid");
  }
  shape.projection = shape.heads * shape.head_width;
  if (!weights.query_convolution.defined() ||
      weights.query_convolution.dim() != 3) {
    throw std::invalid_argument("KDA query convolution must be rank three");
  }
  shape.convolution_width = weights.query_convolution.size(2);
  if (shape.convolution_width <= 1) {
    throw std::invalid_argument("KDA convolution width must be greater than one");
  }
  if (exact_k3 &&
      (shape.batch != kK3Batch || shape.hidden != kK3Hidden ||
       shape.heads != kK3Heads || shape.head_width != kK3HeadWidth ||
       shape.projection != kK3Projection ||
       shape.convolution_width != kK3ConvolutionWidth)) {
    throw std::invalid_argument(
        "KDA production tape refuses a non-K3 dimension contract");
  }
  if (!exact_k3 &&
      (shape.batch != 1 || shape.hidden != kCanaryHidden ||
       shape.heads != kCanaryHeads ||
       shape.head_width != kCanaryHeadWidth ||
       shape.projection != kCanaryProjection ||
       shape.convolution_width != kCanaryConvolutionWidth)) {
    throw std::invalid_argument(
        "KDA canary tape refuses a non-canary dimension contract");
  }

  const at::Device device = hidden.device();
  require_tensor(weights.a_log, device, at::kFloat, {shape.head_width},
                 "A_log");
  require_tensor(weights.dt_bias, device, at::kFloat, {shape.projection},
                 "dt_bias");
  require_tensor(weights.query_convolution, device, at::kFloat,
                 {shape.projection, 1, shape.convolution_width},
                 "query convolution");
  require_tensor(weights.key_convolution, device, at::kFloat,
                 {shape.projection, 1, shape.convolution_width},
                 "key convolution");
  require_tensor(weights.value_convolution, device, at::kFloat,
                 {shape.projection, 1, shape.convolution_width},
                 "value convolution");
  require_tensor(weights.output_norm, device, at::kFloat,
                 {shape.head_width}, "output norm");
  require_projection(weights.query_projection, device, shape.projection,
                     shape.hidden, "query projection");
  require_projection(weights.key_projection, device, shape.projection,
                     shape.hidden, "key projection");
  require_projection(weights.value_projection, device, shape.projection,
                     shape.hidden, "value projection");
  require_projection(weights.recurrent_gate_projection, device,
                     shape.projection, shape.hidden,
                     "recurrent gate projection");
  require_projection(weights.feature_a_projection, device, shape.head_width,
                     shape.hidden, "feature-a projection");
  require_projection(weights.feature_b_projection, device, shape.projection,
                     shape.head_width, "feature-b projection");
  require_projection(weights.beta_projection, device, shape.heads,
                     shape.hidden, "beta projection");
  require_projection(weights.output_projection, device, shape.hidden,
                     shape.projection, "output projection");
  require_tensor(state.query_convolution, device, at::kFloat,
                 {shape.batch, shape.projection, shape.convolution_width},
                 "query convolution state");
  require_tensor(state.key_convolution, device, at::kFloat,
                 {shape.batch, shape.projection, shape.convolution_width},
                 "key convolution state");
  require_tensor(state.value_convolution, device, at::kFloat,
                 {shape.batch, shape.projection, shape.convolution_width},
                 "value convolution state");
  require_tensor(state.recurrent, device, at::kFloat,
                 {shape.batch, shape.heads, shape.head_width,
                  shape.head_width},
                 "recurrent state");
  return shape;
}

at::Tensor packed_linear(const at::Tensor& hidden,
                         const KdaProjection& projection) {
  if (projection.original_bf16.defined()) {
    return original_bf16_linear(hidden, projection.original_bf16);
  }
  if (!projection.scale.defined()) {
    return at::linear(hidden, projection.weight, std::nullopt);
  }
  return at::_weight_int8pack_mm(hidden, projection.weight,
                                 projection.scale);
}

template <std::size_t Size>
std::optional<KdaProjection> adjacent_projection_bundle(
    const std::array<const KdaProjection*, Size>& parts,
    const std::int64_t columns) {
  static_assert(Size != 0);
  if (columns <= 0) {
    return std::nullopt;
  }
  if (parts.front()->original_bf16.defined()) {
    std::array<const OriginalBf16Matrix*, Size> matrices = {};
    for (std::size_t index = 0; index < Size; ++index) {
      if (parts[index] == nullptr || parts[index]->weight.defined() ||
          parts[index]->scale.defined() ||
          !parts[index]->original_bf16.defined()) {
        return std::nullopt;
      }
      matrices[index] = &parts[index]->original_bf16;
    }
    auto combined = adjacent_original_bf16_matrices(matrices);
    if (!combined.has_value() ||
        combined->columns != static_cast<std::size_t>(columns)) {
      return std::nullopt;
    }
    return KdaProjection{at::Tensor(), at::Tensor(), std::move(*combined)};
  }
  std::int64_t rows = 0;
  std::int64_t weight_offset = 0;
  std::int64_t scale_offset = 0;
  for (std::size_t index = 0; index < Size; ++index) {
    const KdaProjection& part = *parts[index];
    if (!part.weight.defined() || !part.scale.defined() ||
        !part.weight.is_contiguous() || !part.scale.is_contiguous() ||
        part.weight.dim() != 2 || part.scale.dim() != 1 ||
        part.weight.size(0) != part.scale.size(0) ||
        part.weight.size(1) != columns ||
        (index != 0 &&
         (!parts.front()->weight.is_alias_of(part.weight) ||
          !parts.front()->scale.is_alias_of(part.scale)))) {
      return std::nullopt;
    }
    if (index == 0) {
      weight_offset = part.weight.storage_offset();
      scale_offset = part.scale.storage_offset();
    }
    if (part.weight.storage_offset() != weight_offset ||
        part.scale.storage_offset() != scale_offset ||
        part.weight.numel() >
            std::numeric_limits<std::int64_t>::max() - weight_offset ||
        part.scale.numel() >
            std::numeric_limits<std::int64_t>::max() - scale_offset ||
        part.weight.size(0) >
            std::numeric_limits<std::int64_t>::max() - rows) {
      return std::nullopt;
    }
    weight_offset += part.weight.numel();
    scale_offset += part.scale.numel();
    rows += part.weight.size(0);
  }
  return KdaProjection{
      parts.front()->weight.as_strided(
          {rows, columns}, {columns, 1},
          parts.front()->weight.storage_offset()),
      parts.front()->scale.as_strided(
          {rows}, {1}, parts.front()->scale.storage_offset()),
  };
}

template <std::size_t Size>
std::optional<at::Tensor> adjacent_tensor_bundle(
    const std::array<const at::Tensor*, Size>& parts,
    const at::IntArrayRef part_shape,
    const at::IntArrayRef combined_shape,
    const at::IntArrayRef combined_strides) {
  static_assert(Size != 0);
  std::int64_t expected_offset = 0;
  for (std::size_t index = 0; index < Size; ++index) {
    const at::Tensor& part = *parts[index];
    if (!part.defined() || !part.is_contiguous() ||
        part.sizes() != part_shape ||
        (index != 0 && !parts.front()->is_alias_of(part))) {
      return std::nullopt;
    }
    if (index == 0) {
      expected_offset = part.storage_offset();
    }
    if (part.storage_offset() != expected_offset ||
        part.numel() >
            std::numeric_limits<std::int64_t>::max() - expected_offset) {
      return std::nullopt;
    }
    expected_offset += part.numel();
  }
  return parts.front()->as_strided(
      combined_shape, combined_strides, parts.front()->storage_offset());
}

std::optional<KdaPreprojectedInputs> bundled_input_projections(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaShape& shape) {
  // The five adjacent K3 projections share the same hidden row and are laid
  // out consecutively in the authenticated spine pack.  A grouped provider
  // binding publishes them as views of one owned device upload, so this is a
  // zero-copy larger view and one established int8pack operator call.
  const auto bundle_with_feature_a = adjacent_projection_bundle<5>(
      {&weights.query_projection, &weights.key_projection,
       &weights.value_projection, &weights.recurrent_gate_projection,
       &weights.feature_a_projection},
      shape.hidden);
  const auto bundle_without_feature_a = bundle_with_feature_a.has_value()
      ? std::optional<KdaProjection>()
      : adjacent_projection_bundle<4>(
            {&weights.query_projection, &weights.key_projection,
             &weights.value_projection,
             &weights.recurrent_gate_projection},
            shape.hidden);
  const std::optional<KdaProjection>& bundle =
      bundle_with_feature_a.has_value() ? bundle_with_feature_a
                                        : bundle_without_feature_a;
  if (!bundle.has_value()) {
    return std::nullopt;
  }
  const at::Tensor projected = packed_linear(hidden, *bundle);
  std::int64_t row = 0;
  const auto take = [&](const std::int64_t count) {
    const at::Tensor result = projected.narrow(1, row, count);
    row += count;
    return result;
  };
  KdaPreprojectedInputs result;
  result.query = take(shape.projection);
  result.key = take(shape.projection);
  result.value = take(shape.projection);
  result.output_gate = take(shape.projection);
  if (bundle_with_feature_a.has_value()) {
    result.feature_a = take(shape.head_width);
  }
  return result;
}

std::pair<at::Tensor, at::Tensor> short_convolution_one(
    const at::Tensor& projected, const at::Tensor& convolution,
    const at::Tensor& previous) {
  const std::int64_t width = previous.size(2);
  const at::Tensor source = at::cat(
      {previous.slice(2, 1, width), projected.unsqueeze(-1)}, 2);
  const at::Tensor products =
      source * convolution.squeeze(1).unsqueeze(0);
  const at::Tensor output = at::silu(products.sum(-1));
  return {output, source.contiguous()};
}

struct KdaShortConvolutionPositions {
  at::Tensor output;
  at::Tensor source;
};

KdaShortConvolutionPositions short_convolution_positions(
    const at::Tensor& projected_rows, const at::Tensor& convolution,
    const at::Tensor& previous, const KdaShape& shape) {
  const std::int64_t positions = projected_rows.size(0);
  require_tensor(projected_rows, previous.device(), at::kFloat,
                 {positions, shape.projection},
                 "preprojected convolution positions");
  const std::int64_t width = shape.convolution_width;
  const at::Tensor source = at::cat(
      {previous.slice(2, 1, width),
       projected_rows.transpose(0, 1).unsqueeze(0)},
      2)
                                .contiguous();
  constexpr std::array<std::int64_t, 1> kStride{1};
  constexpr std::array<std::int64_t, 1> kPadding{0};
  constexpr std::array<std::int64_t, 1> kDilation{1};
  at::Tensor convolved;
  if (source.device().is_cpu() && positions <= 9) {
    // Live auto mode uses explicit four-tap windows through T=9 on CPU.
    // Preserve unfold -> multiply -> last-dimension sum exactly.
    const at::Tensor windows = source.unfold(2, width, 1);
    const at::Tensor weight = convolution.view(
        {1, shape.projection, 1, width});
    convolved = at::sum(windows * weight, {-1}, false);
  } else {
    const std::optional<at::Tensor> bias;
    convolved = at::conv1d(source, convolution, bias, kStride, kPadding,
                           kDilation, shape.projection);
  }
  const at::Tensor output = at::silu(convolved)
                                .squeeze(0)
                                .transpose(0, 1)
                                .contiguous();
  return KdaShortConvolutionPositions{output, source};
}

std::optional<KdaShortConvolution3> bundled_short_convolution_three(
    const at::Tensor& query_projected,
    const at::Tensor& key_projected,
    const at::Tensor& value_projected,
    const KdaWeights& weights, const KdaState& state,
    const KdaShape& shape) {
  const std::int64_t three_projection = shape.projection * 3;
  const auto projected = adjacent_tensor_bundle<3>(
      {&query_projected, &key_projected, &value_projected},
      {shape.batch, shape.projection},
      {shape.batch, three_projection}, {three_projection, 1});
  const auto convolution = adjacent_tensor_bundle<3>(
      {&weights.query_convolution, &weights.key_convolution,
       &weights.value_convolution},
      {shape.projection, 1, shape.convolution_width},
      {three_projection, 1, shape.convolution_width},
      {shape.convolution_width, shape.convolution_width, 1});
  const auto previous = adjacent_tensor_bundle<3>(
      {&state.query_convolution, &state.key_convolution,
       &state.value_convolution},
      {shape.batch, shape.projection, shape.convolution_width},
      {shape.batch, three_projection, shape.convolution_width},
      {three_projection * shape.convolution_width,
       shape.convolution_width, 1});
  if (!projected.has_value() || !convolution.has_value() ||
      !previous.has_value()) {
    return std::nullopt;
  }
  const std::int64_t width = shape.convolution_width;
  const at::Tensor source = at::cat(
      {previous->slice(2, 1, width), projected->unsqueeze(-1)}, 2);
  const at::Tensor output = at::silu(
      (source * convolution->squeeze(1).unsqueeze(0)).sum(-1));
  return KdaShortConvolution3{
      output.narrow(1, 0, shape.projection),
      output.narrow(1, shape.projection, shape.projection),
      output.narrow(1, shape.projection * 2, shape.projection),
      source.narrow(1, 0, shape.projection),
      source.narrow(1, shape.projection, shape.projection),
      source.narrow(1, shape.projection * 2, shape.projection),
  };
}

at::Tensor normalize_last_dimension(const at::Tensor& value) {
  // This executes twice in every KDA layer. Keep the reduction axis on the
  // stack instead of constructing 138 tiny heap vectors per 69-layer token.
  const std::array<std::int64_t, 1> dimension{-1};
  const at::Tensor denominator =
      at::norm(value, at::Scalar(2.0), dimension, true)
          .clamp_min(kK3NormalizeEpsilon)
          .expand_as(value);
  return value / denominator;
}

KdaDecodeResult execute_tape(const at::Tensor& hidden,
                             const KdaWeights& weights,
                             const KdaState& state,
                             const KdaShape& shape,
                             const KdaDispatchPolicy policy,
                             const KdaPreprojectedInputs* prepared,
                             const bool defer_output_projection = false) {
  at::Tensor query_projected;
  at::Tensor key_projected;
  at::Tensor value_projected;
  at::Tensor output_gate_projected;
  at::Tensor feature_a;
  if (prepared != nullptr) {
    require_tensor(prepared->query, hidden.device(), at::kFloat,
                   {shape.batch, shape.projection},
                   "preprojected query");
    require_tensor(prepared->key, hidden.device(), at::kFloat,
                   {shape.batch, shape.projection},
                   "preprojected key");
    require_tensor(prepared->value, hidden.device(), at::kFloat,
                   {shape.batch, shape.projection},
                   "preprojected value");
    require_tensor(prepared->output_gate, hidden.device(), at::kFloat,
                   {shape.batch, shape.projection},
                   "preprojected output gate");
    require_tensor(prepared->feature_a, hidden.device(), at::kFloat,
                   {shape.batch, shape.head_width},
                   "preprojected feature A");
    if (prepared->feature_b.defined()) {
      require_tensor(prepared->feature_b, hidden.device(), at::kFloat,
                     {shape.batch, shape.projection},
                     "preprojected feature B");
    }
    if (prepared->beta.defined()) {
      require_tensor(prepared->beta, hidden.device(), at::kFloat,
                     {shape.batch, shape.heads}, "preprojected beta");
    }
    query_projected = prepared->query;
    key_projected = prepared->key;
    value_projected = prepared->value;
    output_gate_projected = prepared->output_gate;
    feature_a = prepared->feature_a;
  } else if (policy == KdaDispatchPolicy::FusedWhenEligible) {
    const auto input = bundled_input_projections(hidden, weights, shape);
    if (input.has_value()) {
      query_projected = input->query;
      key_projected = input->key;
      value_projected = input->value;
      output_gate_projected = input->output_gate;
      feature_a = input->feature_a;
    }
  }
  if (!query_projected.defined()) {
    query_projected = packed_linear(hidden, weights.query_projection);
    key_projected = packed_linear(hidden, weights.key_projection);
    value_projected = packed_linear(hidden, weights.value_projection);
  }

  at::Tensor query_conv;
  at::Tensor key_conv;
  at::Tensor value_conv;
  at::Tensor next_query_conv;
  at::Tensor next_key_conv;
  at::Tensor next_value_conv;
  if (policy == KdaDispatchPolicy::FusedWhenEligible) {
    const auto convolution = bundled_short_convolution_three(
        query_projected, key_projected, value_projected, weights, state,
        shape);
    if (convolution.has_value()) {
      query_conv = convolution->query;
      key_conv = convolution->key;
      value_conv = convolution->value;
      next_query_conv = convolution->next_query;
      next_key_conv = convolution->next_key;
      next_value_conv = convolution->next_value;
    }
  }
  if (!query_conv.defined()) {
    std::tie(query_conv, next_query_conv) = short_convolution_one(
        query_projected, weights.query_convolution,
        state.query_convolution);
    std::tie(key_conv, next_key_conv) = short_convolution_one(
        key_projected, weights.key_convolution, state.key_convolution);
    std::tie(value_conv, next_value_conv) = short_convolution_one(
        value_projected, weights.value_convolution,
        state.value_convolution);
  }

  at::Tensor query = query_conv.view(
      {shape.batch, 1, shape.heads, shape.head_width});
  at::Tensor key = key_conv.view(
      {shape.batch, 1, shape.heads, shape.head_width});
  const at::Tensor value = value_conv.view(
      {shape.batch, 1, shape.heads, shape.head_width});
  if (!feature_a.defined()) {
    feature_a = packed_linear(hidden, weights.feature_a_projection);
  }
  at::Tensor raw_gate =
      (prepared != nullptr && prepared->feature_b.defined()
           ? prepared->feature_b
           : packed_linear(feature_a, weights.feature_b_projection))
          .view({shape.batch, 1, shape.heads, shape.head_width});
  at::Tensor beta =
      (prepared != nullptr && prepared->beta.defined()
           ? prepared->beta
           : packed_linear(hidden, weights.beta_projection))
          .view({shape.batch, 1, shape.heads});

  query = normalize_last_dimension(query);
  key = normalize_last_dimension(key);
  beta = at::sigmoid(beta);
  raw_gate = raw_gate +
      weights.dt_bias.view({1, 1, shape.heads, shape.head_width});
  const at::Tensor decay_parameter =
      at::exp(weights.a_log).view({1, 1, 1, shape.head_width});
  const at::Tensor gated_decay =
      kK3GateLowerBound * at::sigmoid(decay_parameter * raw_gate);

  const at::Tensor query_token = query.select(1, 0) /
      std::sqrt(static_cast<double>(shape.head_width));
  const at::Tensor key_token = key.select(1, 0);
  const at::Tensor value_token = value.select(1, 0);
  const at::Tensor beta_token = beta.select(1, 0);
  at::Tensor next_recurrent =
      state.recurrent * at::exp(gated_decay.select(1, 0)).unsqueeze(-1);
  const at::Tensor delta = value_token -
      (key_token.unsqueeze(-1) * next_recurrent).sum(-2);
  next_recurrent = next_recurrent + at::einsum(
      "bhk,bhv->bhkv", {beta_token.unsqueeze(-1) * key_token, delta});
  at::Tensor output = at::einsum(
      "bhk,bhkv->bhv", {query_token, next_recurrent});

  const at::Tensor output_variance = output.pow(2).mean(-1, true);
  output = output * at::rsqrt(output_variance + kK3RmsEpsilon);
  output = output * weights.output_norm;
  if (!output_gate_projected.defined()) {
    output_gate_projected =
        packed_linear(hidden, weights.recurrent_gate_projection);
  }
  const at::Tensor output_gate = output_gate_projected.view(
      {shape.batch, shape.heads, shape.head_width});
  output = output * at::sigmoid(output_gate);
  output = output.reshape({shape.batch, shape.projection});
  if (!defer_output_projection) {
    output = packed_linear(output, weights.output_projection);
  }

  return KdaDecodeResult{
      output,
      KdaState{next_query_conv, next_key_conv, next_value_conv,
               next_recurrent},
  };
}

KdaProjection make_canary_projection(const std::int64_t rows,
                                     const std::int64_t columns,
                                     const std::int64_t seed,
                                     const at::Device& device) {
  auto weight_cpu = at::zeros(
      {rows, columns}, at::TensorOptions().dtype(at::kChar).device(at::kCPU));
  auto scale_cpu = at::empty(
      {rows}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  auto weights = weight_cpu.accessor<std::int8_t, 2>();
  auto scales = scale_cpu.accessor<float, 1>();
  constexpr float kScales[4] = {0.015625F, 0.03125F, 0.0625F, 0.125F};
  for (std::int64_t row = 0; row < rows; ++row) {
    const std::int64_t first = (row + seed) % columns;
    const std::int64_t second = (row * 7 + seed * 3 + 1) % columns;
    weights[row][first] = static_cast<std::int8_t>((row + seed) % 7 + 1);
    if (second != first) {
      weights[row][second] =
          static_cast<std::int8_t>(-((row + seed * 2) % 5 + 1));
    }
    scales[row] = kScales[static_cast<std::size_t>((row + seed) % 4)];
  }
  return KdaProjection{
      weight_cpu.to(at::TensorOptions().dtype(at::kChar).device(device)),
      scale_cpu.to(at::TensorOptions().dtype(at::kFloat).device(device)),
  };
}

at::Tensor dense_weight(const KdaProjection& projection) {
  if (projection.original_bf16.defined()) {
    return materialize_original_bf16_f32(projection.original_bf16);
  }
  if (!projection.scale.defined()) {
    return projection.weight;
  }
  return projection.weight.to(at::kFloat) * projection.scale.unsqueeze(1);
}

at::Tensor dense_linear(const at::Tensor& hidden,
                        const KdaProjection& projection) {
  return at::linear(hidden, dense_weight(projection), std::nullopt);
}

KdaDecodeResult independent_reference(const at::Tensor& hidden,
                                      const KdaWeights& weights,
                                      const KdaState& state,
                                      const KdaShape& shape) {
  // Deliberately do not call any production-tape helper.  This spells out the
  // model equations independently with explicitly dequantized dense weights.
  const auto conv = [&](const at::Tensor& projected,
                        const at::Tensor& kernel,
                        const at::Tensor& previous) {
    const at::Tensor history = previous.slice(2, 1, previous.size(2));
    const at::Tensor window =
        at::cat({history, projected.reshape({1, shape.projection, 1})}, 2);
    return std::pair<at::Tensor, at::Tensor>{
        at::silu((window * kernel.reshape(
                               {1, shape.projection,
                                shape.convolution_width}))
                     .sum(2)),
        window.contiguous()};
  };
  auto [query_flat, query_state] = conv(
      dense_linear(hidden, weights.query_projection),
      weights.query_convolution, state.query_convolution);
  auto [key_flat, key_state] = conv(
      dense_linear(hidden, weights.key_projection), weights.key_convolution,
      state.key_convolution);
  auto [value_flat, value_state] = conv(
      dense_linear(hidden, weights.value_projection),
      weights.value_convolution, state.value_convolution);

  auto query = query_flat.reshape({1, shape.heads, shape.head_width});
  auto key = key_flat.reshape({1, shape.heads, shape.head_width});
  const auto value =
      value_flat.reshape({1, shape.heads, shape.head_width});
  const auto norm = [](const at::Tensor& value) {
    return value /
        value.pow(2).sum(-1, true).sqrt().clamp_min(kK3NormalizeEpsilon);
  };
  query = norm(query) / std::sqrt(static_cast<double>(shape.head_width));
  key = norm(key);
  const auto raw_gate =
      dense_linear(dense_linear(hidden, weights.feature_a_projection),
                   weights.feature_b_projection)
          .reshape({1, shape.heads, shape.head_width});
  const auto gate = kK3GateLowerBound * at::sigmoid(
      at::exp(weights.a_log).reshape({1, 1, shape.head_width}) *
      (raw_gate + weights.dt_bias.reshape(
                      {1, shape.heads, shape.head_width})));
  const auto beta = at::sigmoid(
      dense_linear(hidden, weights.beta_projection).reshape({1, shape.heads}));

  auto recurrent = state.recurrent * at::exp(gate).unsqueeze(-1);
  const auto predicted = (key.unsqueeze(-1) * recurrent).sum(2);
  const auto delta = value - predicted;
  recurrent = recurrent +
      (beta.unsqueeze(-1) * key).unsqueeze(-1) * delta.unsqueeze(-2);
  auto output = (query.unsqueeze(-1) * recurrent).sum(2);
  output = output *
      at::rsqrt(output.square().mean(-1, true) + kK3RmsEpsilon);
  output = output * weights.output_norm.reshape({1, 1, shape.head_width});
  const auto output_gate =
      dense_linear(hidden, weights.recurrent_gate_projection)
          .reshape({1, shape.heads, shape.head_width});
  output = output * at::sigmoid(output_gate);
  output = dense_linear(output.reshape({1, shape.projection}),
                        weights.output_projection);
  return KdaDecodeResult{
      output,
      KdaState{query_state, key_state, value_state, recurrent},
  };
}

double maximum_absolute_error(const at::Tensor& left,
                              const at::Tensor& right) {
  return at::max(at::abs(left.to(at::kCPU) - right.to(at::kCPU))).item<double>();
}

bool within_error(const at::Tensor& actual, const at::Tensor& reference,
                  const double absolute, const double relative,
                  double& maximum_error) {
  const at::Tensor actual_cpu = actual.to(at::kCPU);
  const at::Tensor reference_cpu = reference.to(at::kCPU);
  maximum_error = maximum_absolute_error(actual_cpu, reference_cpu);
  return at::allclose(actual_cpu, reference_cpu, relative, absolute);
}

}  // namespace

KdaDecodeResult kda_decode_one(const at::Tensor& hidden,
                               const KdaWeights& weights,
                               const KdaState& state,
                               const bool exact_k3) {
  const KdaShape shape = validate_contract(hidden, weights, state, exact_k3);
  // A single position retains the established exact operation sequence for
  // original-BF16. The sequence tape separately bundles its row-independent
  // original-BF16 projections across T=2..64 on every selected backend. The
  // single-position fused arm here remains the previously qualified MPS q8
  // optimization.
  const bool row_int8 = weights.query_projection.scale.defined();
  const auto policy = hidden.device().is_mps() && row_int8
      ? KdaDispatchPolicy::FusedWhenEligible
      : KdaDispatchPolicy::Established;
  return execute_tape(hidden, weights, state, shape, policy, nullptr);
}

KdaDecodeResult kda_decode_one_preprojected(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, const KdaPreprojectedInputs& projected,
    const bool exact_k3) {
  const KdaShape shape = validate_contract(hidden, weights, state, exact_k3);
  const bool row_int8 = weights.query_projection.scale.defined();
  const bool original_bf16 =
      weights.query_projection.original_bf16.defined();
  if ((!hidden.device().is_mps() || !row_int8) && !original_bf16) {
    throw std::invalid_argument(
        "KDA preprojected decode requires original-BF16 on the selected "
        "device or packed MPS weights");
  }
  return execute_tape(hidden, weights, state, shape,
                      KdaDispatchPolicy::FusedWhenEligible, &projected);
}

KdaRecurrentResult kda_decode_one_preprojected_deferred_output(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, const KdaPreprojectedInputs& projected,
    const bool exact_k3) {
  const KdaShape shape = validate_contract(hidden, weights, state, exact_k3);
  const bool row_int8 = weights.query_projection.scale.defined();
  const bool original_bf16 =
      weights.query_projection.original_bf16.defined();
  const bool dense = weights.query_projection.weight.defined() &&
      weights.query_projection.weight.scalar_type() == at::kFloat &&
      !weights.query_projection.scale.defined() &&
      !weights.query_projection.original_bf16.defined();
  if ((!hidden.device().is_mps() || !row_int8) && !original_bf16 &&
      !dense) {
    throw std::invalid_argument(
        "KDA deferred preprojected decode requires original-BF16, packed MPS weights, or a validated dense fp32 carrier");
  }
  KdaDecodeResult result = execute_tape(
      hidden, weights, state, shape, KdaDispatchPolicy::FusedWhenEligible,
      &projected, true);
  return KdaRecurrentResult{std::move(result.output),
                            std::move(result.next_state)};
}

KdaConvolvedPositions kda_short_convolve_positions(
    const at::Tensor& hidden_rows, const KdaWeights& weights,
    const KdaState& state, const KdaPreprojectedPositions& projected,
    const bool exact_k3) {
  if (!hidden_rows.defined() || hidden_rows.dim() != 2 ||
      hidden_rows.size(0) < 2 || hidden_rows.size(0) > 64 ||
      hidden_rows.scalar_type() != at::kFloat ||
      !hidden_rows.is_contiguous()) {
    throw std::invalid_argument(
        "KDA positions require contiguous fp32 [2..64,hidden]");
  }
  const std::int64_t positions = hidden_rows.size(0);
  const KdaShape shape = validate_contract(
      hidden_rows.narrow(0, 0, 1), weights, state, exact_k3);
  if (hidden_rows.size(1) != shape.hidden) {
    throw std::invalid_argument(
        "KDA positions hidden width changed after contract validation");
  }
  require_tensor(projected.query, hidden_rows.device(), at::kFloat,
                 {positions, shape.projection},
                 "preprojected position query");
  require_tensor(projected.key, hidden_rows.device(), at::kFloat,
                 {positions, shape.projection},
                 "preprojected position key");
  require_tensor(projected.value, hidden_rows.device(), at::kFloat,
                 {positions, shape.projection},
                 "preprojected position value");

  const KdaShortConvolutionPositions query_conv =
      short_convolution_positions(projected.query,
                                  weights.query_convolution,
                                  state.query_convolution, shape);
  const KdaShortConvolutionPositions key_conv =
      short_convolution_positions(projected.key, weights.key_convolution,
                                  state.key_convolution, shape);
  const KdaShortConvolutionPositions value_conv =
      short_convolution_positions(projected.value,
                                  weights.value_convolution,
                                  state.value_convolution, shape);
  return KdaConvolvedPositions{
      std::move(query_conv.output), std::move(key_conv.output),
      std::move(value_conv.output), std::move(query_conv.source),
      std::move(key_conv.source), std::move(value_conv.source)};
}

namespace {

/*
 * One row of the KDA delta-rule recurrence. The positions loop and prefix
 * replay both call exactly this, which is what makes a replayed boundary
 * bit-identical to the one the loop held.
 */
at::Tensor kda_recurrent_row_update(const at::Tensor& recurrent,
                                    const at::Tensor& decay_row,
                                    const at::Tensor& key_row,
                                    const at::Tensor& value_row,
                                    const at::Tensor& beta_row) {
  const at::Tensor decayed = recurrent * at::exp(decay_row).unsqueeze(-1);
  const at::Tensor delta =
      value_row - (key_row.unsqueeze(-1) * decayed).sum(-2);
  return decayed + at::einsum("bhk,bhv->bhkv",
                              {beta_row.unsqueeze(-1) * key_row, delta});
}

}  // namespace

KdaPositionsRecurrentResult kda_recur_convolved_positions(
    const at::Tensor& hidden_rows, const KdaWeights& weights,
    const KdaState& state, const KdaConvolvedPositions& convolved,
    const KdaDependentPositions& dependent,
    const bool retain_boundaries, const bool exact_k3,
    KdaRecurrenceRecord* const record) {
  if (!hidden_rows.defined() || hidden_rows.dim() != 2 ||
      hidden_rows.size(0) < 2 || hidden_rows.size(0) > 64 ||
      hidden_rows.scalar_type() != at::kFloat ||
      !hidden_rows.is_contiguous()) {
    throw std::invalid_argument(
        "KDA recurrence positions require contiguous fp32 [2..64,hidden]");
  }
  const std::int64_t positions = hidden_rows.size(0);
  const KdaShape shape = validate_contract(
      hidden_rows.narrow(0, 0, 1), weights, state, exact_k3);
  if (hidden_rows.size(1) != shape.hidden) {
    throw std::invalid_argument(
        "KDA recurrence hidden width changed after contract validation");
  }
  require_tensor(convolved.query, hidden_rows.device(), at::kFloat,
                 {positions, shape.projection},
                 "convolved position query");
  require_tensor(convolved.key, hidden_rows.device(), at::kFloat,
                 {positions, shape.projection},
                 "convolved position key");
  require_tensor(convolved.value, hidden_rows.device(), at::kFloat,
                 {positions, shape.projection},
                 "convolved position value");
  const std::int64_t source_width =
      shape.convolution_width - 1 + positions;
  require_tensor(convolved.query_source, hidden_rows.device(), at::kFloat,
                 {1, shape.projection, source_width},
                 "query convolution source");
  require_tensor(convolved.key_source, hidden_rows.device(), at::kFloat,
                 {1, shape.projection, source_width},
                 "key convolution source");
  require_tensor(convolved.value_source, hidden_rows.device(), at::kFloat,
                 {1, shape.projection, source_width},
                 "value convolution source");
  require_tensor(dependent.feature_a, hidden_rows.device(), at::kFloat,
                 {positions, shape.head_width},
                 "dependent position feature A");
  require_tensor(dependent.feature_b, hidden_rows.device(), at::kFloat,
                 {positions, shape.projection},
                 "dependent position feature B");
  require_tensor(dependent.beta, hidden_rows.device(), at::kFloat,
                 {positions, shape.heads},
                 "dependent position beta");

  at::Tensor query = normalize_last_dimension(
      convolved.query.view({positions, shape.heads, shape.head_width}));
  at::Tensor key = normalize_last_dimension(
      convolved.key.view({positions, shape.heads, shape.head_width}));
  const at::Tensor value = convolved.value.view(
      {positions, shape.heads, shape.head_width});
  const at::Tensor beta = at::sigmoid(dependent.beta);
  const at::Tensor raw_gate =
      dependent.feature_b.view(
          {positions, shape.heads, shape.head_width}) +
      weights.dt_bias.view({1, shape.heads, shape.head_width});
  const at::Tensor decay_parameter =
      at::exp(weights.a_log).view({1, 1, shape.head_width});
  const at::Tensor gated_decay =
      kK3GateLowerBound *
      at::sigmoid(decay_parameter * raw_gate);

  std::vector<at::Tensor> recurrent_outputs;
  recurrent_outputs.reserve(static_cast<std::size_t>(positions));
  std::vector<KdaState> boundaries;
  if (retain_boundaries) {
    boundaries.reserve(static_cast<std::size_t>(positions));
  }
  at::Tensor recurrent = state.recurrent;
  const double scale =
      std::sqrt(static_cast<double>(shape.head_width));
  if (record != nullptr) {
    *record = KdaRecurrenceRecord{
        .initial_recurrent = state.recurrent,
        .gated_decay = gated_decay,
        .key = key,
        .value = value,
        .beta = beta,
        .query_source = convolved.query_source,
        .key_source = convolved.key_source,
        .value_source = convolved.value_source,
        .positions = positions,
        .convolution_width = shape.convolution_width,
    };
  }
  for (std::int64_t position = 0; position < positions; ++position) {
    const at::Tensor query_token = query.narrow(0, position, 1) / scale;
    recurrent = kda_recurrent_row_update(
        recurrent, gated_decay.narrow(0, position, 1),
        key.narrow(0, position, 1), value.narrow(0, position, 1),
        beta.narrow(0, position, 1));
    recurrent_outputs.push_back(at::einsum(
        "bhk,bhkv->bhv", {query_token, recurrent}));
    if (retain_boundaries) {
      boundaries.push_back(KdaState{
          convolved.query_source.narrow(
              2, position, shape.convolution_width),
          convolved.key_source.narrow(
              2, position, shape.convolution_width),
          convolved.value_source.narrow(
              2, position, shape.convolution_width),
          recurrent,
      });
    }
  }

  at::Tensor output =
      at::cat(recurrent_outputs, 0)
          .reshape({positions, shape.projection})
          .contiguous();

  const std::int64_t final_start = positions - 1;
  KdaState final_state{
      convolved.query_source
          .narrow(2, final_start, shape.convolution_width)
          .contiguous(),
      convolved.key_source
          .narrow(2, final_start, shape.convolution_width)
          .contiguous(),
      convolved.value_source
          .narrow(2, final_start, shape.convolution_width)
          .contiguous(),
      recurrent,
  };
  return KdaPositionsRecurrentResult{
      std::move(output), std::move(final_state), std::move(boundaries)};
}

KdaState kda_replay_recorded_state(const KdaRecurrenceRecord& record,
                                   const std::int64_t positions) {
  if (!record.initial_recurrent.defined() || record.positions < 1 ||
      positions < 1 || positions > record.positions ||
      record.convolution_width < 1) {
    throw std::invalid_argument(
        "KDA prefix replay needs 1..T rows of a complete record");
  }
  at::Tensor recurrent = record.initial_recurrent;
  for (std::int64_t position = 0; position < positions; ++position) {
    recurrent = kda_recurrent_row_update(
        recurrent, record.gated_decay.narrow(0, position, 1),
        record.key.narrow(0, position, 1),
        record.value.narrow(0, position, 1),
        record.beta.narrow(0, position, 1));
  }
  const std::int64_t start = positions - 1;
  return KdaState{
      record.query_source.narrow(2, start, record.convolution_width),
      record.key_source.narrow(2, start, record.convolution_width),
      record.value_source.narrow(2, start, record.convolution_width),
      std::move(recurrent),
  };
}

std::uint64_t kda_recurrence_record_bytes(const KdaRecurrenceRecord& record) {
  std::uint64_t bytes = 0;
  for (const at::Tensor* tensor :
       {&record.gated_decay, &record.key, &record.value, &record.beta,
        &record.query_source, &record.key_source, &record.value_source}) {
    if (tensor->defined()) {
      bytes += static_cast<std::uint64_t>(tensor->numel()) *
               static_cast<std::uint64_t>(tensor->element_size());
    }
  }
  return bytes;
}

KdaDecodeResult kda_decode_one_unfused_for_test(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, const bool exact_k3) {
  const KdaShape shape = validate_contract(hidden, weights, state, exact_k3);
  return execute_tape(hidden, weights, state, shape,
                      KdaDispatchPolicy::Established, nullptr);
}

KdaDecodeResult kda_decode_one_fused_for_test(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, const bool exact_k3) {
  const KdaShape shape = validate_contract(hidden, weights, state, exact_k3);
  return execute_tape(hidden, weights, state, shape,
                      KdaDispatchPolicy::FusedWhenEligible, nullptr);
}

KdaState zero_k3_kda_state(const at::Device& device) {
  const auto options = at::TensorOptions().dtype(at::kFloat).device(device);
  const at::Tensor convolution = at::zeros(
      {kK3Batch, kK3Projection * 3, kK3ConvolutionWidth}, options);
  return KdaState{
      convolution.narrow(1, 0, kK3Projection),
      convolution.narrow(1, kK3Projection, kK3Projection),
      convolution.narrow(1, kK3Projection * 2, kK3Projection),
      at::zeros({kK3Batch, kK3Heads, kK3HeadWidth, kK3HeadWidth}, options),
  };
}

KdaState zero_small_kda_canary_state(const at::Device& device) {
  const auto options = at::TensorOptions().dtype(at::kFloat).device(device);
  const at::Tensor convolution = at::zeros(
      {1, kCanaryProjection * 3, kCanaryConvolutionWidth}, options);
  return KdaState{
      convolution.narrow(1, 0, kCanaryProjection),
      convolution.narrow(1, kCanaryProjection, kCanaryProjection),
      convolution.narrow(1, kCanaryProjection * 2, kCanaryProjection),
      at::zeros({1, kCanaryHeads, kCanaryHeadWidth, kCanaryHeadWidth},
                options),
  };
}

std::uint64_t kda_state_conv_elements(const KdaState& state) {
  return static_cast<std::uint64_t>(state.query_convolution.numel()) +
      static_cast<std::uint64_t>(state.key_convolution.numel()) +
      static_cast<std::uint64_t>(state.value_convolution.numel());
}

std::uint64_t kda_state_recurrent_elements(const KdaState& state) {
  return static_cast<std::uint64_t>(state.recurrent.numel());
}

bool kda_small_parity_canary(const at::Device& device, std::string& detail) {
  try {
    constexpr std::int64_t hidden_width = kCanaryHidden;
    constexpr std::int64_t heads = kCanaryHeads;
    constexpr std::int64_t head_width = kCanaryHeadWidth;
    constexpr std::int64_t projection = kCanaryProjection;
    constexpr std::int64_t convolution_width = kCanaryConvolutionWidth;
    const auto cpu = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const auto target = cpu.device(device);
    auto hidden_cpu = at::empty({1, hidden_width}, cpu);
    auto hidden_values = hidden_cpu.accessor<float, 2>();
    for (std::int64_t column = 0; column < hidden_width; ++column) {
      hidden_values[0][column] =
          static_cast<float>((column * 5) % 17 - 8) / 16.0F;
    }
    auto raw_vector = [&](const std::int64_t elements,
                          const std::int64_t seed) {
      auto value = at::empty({elements}, cpu);
      auto values = value.accessor<float, 1>();
      for (std::int64_t index = 0; index < elements; ++index) {
        values[index] =
            static_cast<float>((index + seed) % 11 - 5) / 128.0F;
      }
      return value.to(target);
    };
    auto convolution = [&](const std::int64_t seed) {
      auto value = at::empty({projection, 1, convolution_width}, cpu);
      auto values = value.accessor<float, 3>();
      for (std::int64_t row = 0; row < projection; ++row) {
        for (std::int64_t tap = 0; tap < convolution_width; ++tap) {
          values[row][0][tap] =
              static_cast<float>((row + tap + seed) % 7 - 3) / 32.0F;
        }
      }
      return value.to(target);
    };
    KdaWeights weights{
        raw_vector(head_width, 1),
        raw_vector(projection, 2),
        convolution(3),
        convolution(4),
        convolution(5),
        at::ones({head_width}, target),
        make_canary_projection(projection, hidden_width, 1, device),
        make_canary_projection(projection, hidden_width, 2, device),
        make_canary_projection(projection, hidden_width, 3, device),
        make_canary_projection(projection, hidden_width, 4, device),
        make_canary_projection(head_width, hidden_width, 5, device),
        make_canary_projection(projection, head_width, 6, device),
        make_canary_projection(heads, hidden_width, 7, device),
        make_canary_projection(hidden_width, projection, 8, device),
    };
    KdaState state{
        at::zeros({1, projection, convolution_width}, target),
        at::zeros({1, projection, convolution_width}, target),
        at::zeros({1, projection, convolution_width}, target),
        at::zeros({1, heads, head_width, head_width}, target),
    };
    const at::Tensor hidden = hidden_cpu.to(target);
    const KdaShape shape = validate_contract(hidden, weights, state, false);
    const KdaDecodeResult actual = execute_tape(
        hidden, weights, state, shape,
        device.is_mps() ? KdaDispatchPolicy::FusedWhenEligible
                        : KdaDispatchPolicy::Established,
        nullptr);
    const KdaDecodeResult reference =
        independent_reference(hidden, weights, state, shape);

    // The oracle dequantizes to a dense matrix while the production path uses
    // the qualified packed operator.  Those kernels may choose a different
    // fp32 reduction order even on CPU, so parity is numerical rather than a
    // false bit-equality promise.  The bound is deliberately much tighter
    // than the model's stored bf16 source precision.
    const double absolute = device.is_cpu() ? 2.0e-6 : 2.0e-5;
    const double relative = device.is_cpu() ? 2.0e-6 : 2.0e-5;
    double output_error = 0.0;
    double query_cache_error = 0.0;
    double key_cache_error = 0.0;
    double value_cache_error = 0.0;
    double recurrent_error = 0.0;
    const bool output_ok = within_error(
        actual.output, reference.output, absolute, relative, output_error);
    const bool query_ok = within_error(
        actual.next_state.query_convolution,
        reference.next_state.query_convolution, absolute, relative,
        query_cache_error);
    const bool key_ok = within_error(
        actual.next_state.key_convolution,
        reference.next_state.key_convolution, absolute, relative,
        key_cache_error);
    const bool value_ok = within_error(
        actual.next_state.value_convolution,
        reference.next_state.value_convolution, absolute, relative,
        value_cache_error);
    const bool recurrent_ok = within_error(
        actual.next_state.recurrent, reference.next_state.recurrent,
        absolute, relative, recurrent_error);
    const bool passed =
        output_ok && query_ok && key_ok && value_ok && recurrent_ok;
    std::ostringstream summary;
    summary.setf(std::ios::scientific);
    summary.precision(3);
    summary << "output_max_abs=" << output_error
            << " q_cache_max_abs=" << query_cache_error
            << " k_cache_max_abs=" << key_cache_error
            << " v_cache_max_abs=" << value_cache_error
            << " recurrent_max_abs=" << recurrent_error;
    detail = summary.str();
    return passed;
  } catch (const std::exception& error) {
    detail = error.what();
    return false;
  }
}

}  // namespace deltafin::provider_internal
