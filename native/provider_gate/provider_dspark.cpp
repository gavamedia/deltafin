#include "provider_dspark.h"

#include <ATen/ops/cat.h>
#include <ATen/ops/einsum.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/softmax.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kK3Hidden = 7168;
constexpr std::int64_t kK3Intermediate = 14336;
constexpr std::int64_t kK3Heads = 64;
constexpr std::int64_t kK3QueryRank = 1536;
constexpr std::int64_t kK3KeyValueRank = 512;
constexpr std::int64_t kK3Nope = 128;
constexpr std::int64_t kK3Rope = 64;
constexpr std::int64_t kK3Value = 128;
constexpr std::int64_t kK3MaximumPosition = 1048576;
constexpr std::int64_t kK3Vocabulary = 163840;
constexpr std::int64_t kK3TargetHidden = 7168;
constexpr std::int64_t kK3MarkovRank = 256;
constexpr std::int64_t kK3MaskToken = 163837;
constexpr double kK3RmsEpsilon = 1.0e-5;
constexpr double kK3RopeTheta = 50000.0;
constexpr double kK3RopeFactor = 32.0;
constexpr std::int64_t kK3RopeOriginalMaximum = 32768;
constexpr double kK3RopeBetaFast = 32.0;
constexpr double kK3RopeBetaSlow = 1.0;
constexpr double kK3RopeMscale = 1.0;
constexpr double kK3RopeMscaleAllDim = 1.0;
constexpr std::int64_t kMaximumDraftRows = 7;

void require_positive(const std::int64_t value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string("DSpark ") + name +
                                " must be positive");
  }
}

void require_finite_positive(const double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string("DSpark ") + name +
                                " must be positive and finite");
  }
}

void safe_product(const std::int64_t left, const std::int64_t right,
                  const char* name) {
  if (left > std::numeric_limits<std::int64_t>::max() / right) {
    throw std::invalid_argument(std::string("DSpark ") + name +
                                " overflows int64");
  }
}

void require_tensor(const at::Tensor& tensor, const at::Device& device,
                    const at::ScalarType dtype,
                    const at::IntArrayRef shape, const char* name) {
  if (!tensor.defined() || tensor.device() != device ||
      tensor.scalar_type() != dtype || !tensor.is_contiguous() ||
      tensor.sizes() != shape) {
    throw std::invalid_argument(std::string("DSpark ") + name +
                                " violates its contiguous shape/dtype/device contract");
  }
}

void require_rows(const at::Tensor& tensor, const std::int64_t columns,
                  const char* name) {
  if (!tensor.defined() || tensor.scalar_type() != at::kBFloat16 ||
      !tensor.is_contiguous() ||
      tensor.dim() != 2 || tensor.size(0) < 1 ||
      tensor.size(0) > kMaximumDraftRows || tensor.size(1) != columns) {
    throw std::invalid_argument(std::string("DSpark ") + name +
                                " must be contiguous BF16 [1..7,width]");
  }
}

void validate_positions(const at::Tensor& positions, const at::Device& device,
                        const std::int64_t rows,
                        const DSparkShape& shape) {
  require_tensor(positions, device, at::kLong, {rows}, "positions");
  const at::Tensor invalid =
      (positions < 0) | (positions >= shape.max_position);
  if (invalid.any().item<bool>()) {
    throw std::invalid_argument(
        "DSpark positions lie outside the configured context");
  }
}

void validate_linear(const at::Tensor& weight, const at::Device& device,
                     const std::int64_t rows, const std::int64_t columns,
                     const char* name) {
  require_tensor(weight, device, at::kBFloat16, {rows, columns}, name);
}

at::Tensor dense_linear(const at::Tensor& input, const at::Tensor& weight) {
  if (input.dim() != 2 || weight.dim() != 2 ||
      input.size(1) != weight.size(1)) {
    throw std::invalid_argument("DSpark linear dimensions disagree");
  }
  return at::linear(input, weight, std::nullopt);
}

void validate_mla_weights(const DSparkMlaWeights& weights,
                          const DSparkShape& shape,
                          const at::Device& device) {
  const std::int64_t query_width =
      shape.num_heads * shape.query_head_dim();
  const std::int64_t compressed_width =
      shape.kv_lora_rank + shape.qk_rope_head_dim;
  const std::int64_t expanded_width =
      shape.num_heads * (shape.qk_nope_head_dim + shape.value_head_dim);
  validate_linear(weights.query_a, device, shape.q_lora_rank,
                  shape.hidden_size, "query-a weight");
  require_tensor(weights.query_a_norm, device, at::kBFloat16,
                 {shape.q_lora_rank}, "query-a norm");
  validate_linear(weights.query_b, device, query_width, shape.q_lora_rank,
                  "query-b weight");
  validate_linear(weights.key_value_a, device, compressed_width,
                  shape.hidden_size, "key/value-a weight");
  require_tensor(weights.key_value_a_norm, device, at::kBFloat16,
                 {shape.kv_lora_rank}, "key/value-a norm");
  validate_linear(weights.key_value_b, device, expanded_width,
                  shape.kv_lora_rank, "key/value-b weight");
  validate_linear(weights.output, device, shape.hidden_size,
                  shape.num_heads * shape.value_head_dim, "output weight");
}

void validate_context(const DSparkLatentContext& context,
                      const DSparkShape& shape, const at::Device& device) {
  if (!context.latent.defined() || !context.positional.defined() ||
      context.latent.device() != device ||
      context.positional.device() != device ||
      context.latent.scalar_type() != at::kBFloat16 ||
      context.positional.scalar_type() != at::kBFloat16 ||
      !context.latent.is_contiguous() || !context.positional.is_contiguous() ||
      context.latent.dim() != 2 || context.positional.dim() != 2 ||
      context.latent.size(0) != context.positional.size(0) ||
      context.latent.size(1) != shape.kv_lora_rank ||
      context.positional.size(1) != shape.qk_rope_head_dim ||
      context.latent.size(0) > shape.max_position) {
    throw std::invalid_argument(
        "DSpark compact context violates its BF16 [K,L]/[K,R] contract");
  }
}

double yarn_mscale(const double factor, const double multiplier) {
  return factor <= 1.0 ? 1.0 : 0.1 * multiplier * std::log(factor) + 1.0;
}

}  // namespace

DSparkShape DSparkShape::k3() {
  return DSparkShape{
      .hidden_size = kK3Hidden,
      .intermediate_size = kK3Intermediate,
      .num_heads = kK3Heads,
      .q_lora_rank = kK3QueryRank,
      .kv_lora_rank = kK3KeyValueRank,
      .qk_nope_head_dim = kK3Nope,
      .qk_rope_head_dim = kK3Rope,
      .value_head_dim = kK3Value,
      .max_position = kK3MaximumPosition,
      .vocabulary_size = kK3Vocabulary,
      .target_hidden_size = kK3TargetHidden,
      .markov_rank = kK3MarkovRank,
      .mask_token_id = kK3MaskToken,
      .rms_epsilon = kK3RmsEpsilon,
      .rope_theta = kK3RopeTheta,
      .rope_factor = kK3RopeFactor,
      .rope_original_max_position = kK3RopeOriginalMaximum,
      .rope_beta_fast = kK3RopeBetaFast,
      .rope_beta_slow = kK3RopeBetaSlow,
      .rope_mscale = kK3RopeMscale,
      .rope_mscale_all_dim = kK3RopeMscaleAllDim,
  };
}

DSparkShape DSparkShape::small_canary() {
  return DSparkShape{
      .hidden_size = 8,
      .intermediate_size = 12,
      .num_heads = 2,
      .q_lora_rank = 4,
      .kv_lora_rank = 4,
      .qk_nope_head_dim = 2,
      .qk_rope_head_dim = 4,
      .value_head_dim = 2,
      .max_position = 32,
      .vocabulary_size = 32,
      .target_hidden_size = 8,
      .markov_rank = 4,
      .mask_token_id = 31,
      .rms_epsilon = kK3RmsEpsilon,
      .rope_theta = 100.0,
      .rope_factor = 2.0,
      .rope_original_max_position = 8,
      .rope_beta_fast = 4.0,
      .rope_beta_slow = 1.0,
      .rope_mscale = 1.0,
      .rope_mscale_all_dim = 1.0,
  };
}

void DSparkShape::validate() const {
  require_positive(hidden_size, "hidden size");
  require_positive(intermediate_size, "intermediate size");
  require_positive(num_heads, "head count");
  require_positive(q_lora_rank, "query LoRA rank");
  require_positive(kv_lora_rank, "key/value LoRA rank");
  require_positive(qk_nope_head_dim, "non-positional head width");
  require_positive(qk_rope_head_dim, "rotary head width");
  require_positive(value_head_dim, "value head width");
  require_positive(max_position, "maximum position");
  require_positive(vocabulary_size, "vocabulary size");
  require_positive(target_hidden_size, "target hidden size");
  require_positive(markov_rank, "Markov rank");
  require_positive(rope_original_max_position, "original rotary maximum");
  if (mask_token_id < 0 || mask_token_id >= vocabulary_size) {
    throw std::invalid_argument("DSpark mask token lies outside the vocabulary");
  }
  if ((qk_rope_head_dim % 2) != 0) {
    throw std::invalid_argument("DSpark rotary head width must be even");
  }
  require_finite_positive(rms_epsilon, "RMS epsilon");
  require_finite_positive(rope_theta, "rotary theta");
  require_finite_positive(rope_factor, "rotary factor");
  require_finite_positive(rope_beta_fast, "rotary beta-fast");
  require_finite_positive(rope_beta_slow, "rotary beta-slow");
  require_finite_positive(rope_mscale, "rotary mscale");
  require_finite_positive(rope_mscale_all_dim, "rotary all-dimension mscale");
  if (qk_nope_head_dim >
      std::numeric_limits<std::int64_t>::max() - qk_rope_head_dim) {
    throw std::invalid_argument("DSpark query head width overflows int64");
  }
  if (qk_nope_head_dim >
      std::numeric_limits<std::int64_t>::max() - value_head_dim) {
    throw std::invalid_argument("DSpark expanded key/value width overflows int64");
  }
  if (kv_lora_rank >
      std::numeric_limits<std::int64_t>::max() - qk_rope_head_dim) {
    throw std::invalid_argument("DSpark compressed width overflows int64");
  }
  safe_product(num_heads, query_head_dim(), "query width");
  safe_product(num_heads, qk_nope_head_dim + value_head_dim,
               "expanded key/value width");
  safe_product(num_heads, value_head_dim, "value width");
  safe_product(5, target_hidden_size, "target context width");
}

bool DSparkShape::is_exact_k3() const {
  return hidden_size == kK3Hidden && intermediate_size == kK3Intermediate &&
         num_heads == kK3Heads && q_lora_rank == kK3QueryRank &&
         kv_lora_rank == kK3KeyValueRank && qk_nope_head_dim == kK3Nope &&
         qk_rope_head_dim == kK3Rope && value_head_dim == kK3Value &&
         max_position == kK3MaximumPosition &&
         vocabulary_size == kK3Vocabulary &&
         target_hidden_size == kK3TargetHidden &&
         markov_rank == kK3MarkovRank && mask_token_id == kK3MaskToken &&
         rms_epsilon == kK3RmsEpsilon && rope_theta == kK3RopeTheta &&
         rope_factor == kK3RopeFactor &&
         rope_original_max_position == kK3RopeOriginalMaximum &&
         rope_beta_fast == kK3RopeBetaFast &&
         rope_beta_slow == kK3RopeBetaSlow &&
         rope_mscale == kK3RopeMscale &&
         rope_mscale_all_dim == kK3RopeMscaleAllDim;
}

std::int64_t DSparkShape::query_head_dim() const {
  if (qk_nope_head_dim >
      std::numeric_limits<std::int64_t>::max() - qk_rope_head_dim) {
    throw std::invalid_argument("DSpark query head width overflows int64");
  }
  return qk_nope_head_dim + qk_rope_head_dim;
}

std::int64_t DSparkShape::target_context_width() const {
  safe_product(5, target_hidden_size, "target context width");
  return 5 * target_hidden_size;
}

at::Tensor dspark_rms_norm_bf16(const at::Tensor& value,
                                const at::Tensor& weight,
                                const double epsilon) {
  const c10::InferenceMode inference_guard;
  if (!value.defined() || value.scalar_type() != at::kBFloat16 ||
      !value.is_contiguous() || value.dim() < 1 || value.device().is_meta()) {
    throw std::invalid_argument(
        "DSpark RMSNorm input must be contiguous, non-meta BF16");
  }
  require_finite_positive(epsilon, "RMS epsilon");
  require_tensor(weight, value.device(), at::kBFloat16, {value.size(-1)},
                 "RMSNorm weight");
  const at::Tensor promoted = value.to(at::kFloat);
  const at::Tensor variance = at::mean(at::pow(promoted, 2), {-1}, true);
  const at::Tensor normalized =
      promoted * at::rsqrt(variance + epsilon);
  // Match the oracle's cast point: normalize in fp32, cast, then multiply by
  // the checkpoint's BF16 weight.
  return normalized.to(at::kBFloat16) * weight;
}

at::Tensor dspark_yarn_inverse_frequencies(const DSparkShape& shape,
                                           const at::Device& device) {
  const c10::InferenceMode inference_guard;
  shape.validate();
  if (device.is_meta()) {
    throw std::invalid_argument("DSpark YaRN requires a concrete device");
  }
  const std::int64_t half = shape.qk_rope_head_dim / 2;
  const auto options = at::TensorOptions().dtype(at::kFloat).device(device);
  const at::Tensor pair_indices = at::arange(half, options);
  const at::Tensor exponent =
      pair_indices * (2.0 / static_cast<double>(shape.qk_rope_head_dim));
  const at::Tensor extrapolated =
      at::exp(exponent * std::log(shape.rope_theta)).reciprocal();
  const at::Tensor interpolated = extrapolated / shape.rope_factor;
  const auto correction = [&shape](const double rotations) {
    return static_cast<double>(shape.qk_rope_head_dim) *
           std::log(static_cast<double>(shape.rope_original_max_position) /
                    (rotations * 2.0 * std::acos(-1.0))) /
           (2.0 * std::log(shape.rope_theta));
  };
  const double low = std::max(std::floor(correction(shape.rope_beta_fast)),
                              0.0);
  double high = std::min(std::ceil(correction(shape.rope_beta_slow)),
                         static_cast<double>(shape.qk_rope_head_dim - 1));
  if (low == high) {
    high += 0.001;
  }
  const at::Tensor ramp = ((pair_indices - low) / (high - low)).clamp(0.0, 1.0);
  const at::Tensor extrapolation_factor = 1.0 - ramp;
  return interpolated * (1.0 - extrapolation_factor) +
         extrapolated * extrapolation_factor;
}

at::Tensor dspark_apply_yarn_rotary_bf16(const at::Tensor& value,
                                         const at::Tensor& positions,
                                         const DSparkShape& shape) {
  const c10::InferenceMode inference_guard;
  shape.validate();
  if (!value.defined() || value.scalar_type() != at::kBFloat16 ||
      !value.is_contiguous() || value.dim() < 2 || value.device().is_meta() ||
      value.size(0) < 1 || value.size(0) > kMaximumDraftRows ||
      value.size(-1) != shape.qk_rope_head_dim) {
    throw std::invalid_argument(
        "DSpark rotary input must be contiguous BF16 [1..7,...,rope]");
  }
  validate_positions(positions, value.device(), value.size(0), shape);
  const at::Tensor inverse =
      dspark_yarn_inverse_frequencies(shape, value.device());
  const at::Tensor phase = positions.to(at::kFloat).unsqueeze(1) * inverse;
  const double rope_scale =
      yarn_mscale(shape.rope_factor, shape.rope_mscale) /
      yarn_mscale(shape.rope_factor, shape.rope_mscale_all_dim);
  at::Tensor cosine = at::cos(phase) * rope_scale;
  at::Tensor sine = at::sin(phase) * rope_scale;
  for (std::int64_t dimension = 1; dimension < value.dim() - 1; ++dimension) {
    cosine = cosine.unsqueeze(1);
    sine = sine.unsqueeze(1);
  }
  const auto original_shape = value.sizes().vec();
  auto pair_shape = original_shape;
  // Unreachable in practice: value.dim() >= 2 was validated above, so the
  // shape has a trailing dimension. The explicit guard lets GCC's
  // -Warray-bounds analysis prove pair_shape.back() below is in bounds;
  // GCC 15 otherwise reports a false positive through the vector copy.
  if (pair_shape.empty()) {
    throw std::invalid_argument(
        "DSpark rotary shape has no trailing dimension");
  }
  pair_shape.back() = shape.qk_rope_head_dim / 2;
  pair_shape.push_back(2);
  const at::Tensor pairs = value.to(at::kFloat).reshape(pair_shape);
  const at::Tensor even = pairs.select(-1, 0);
  const at::Tensor odd = pairs.select(-1, 1);
  const at::Tensor rotated =
      at::stack({even * cosine - odd * sine,
                 odd * cosine + even * sine},
                -1);
  return rotated.reshape(original_shape).to(at::kBFloat16);
}

DSparkMlaOutput run_dspark_mla(const at::Tensor& hidden,
                              const at::Tensor& positions,
                              const DSparkLatentContext& context,
                              const DSparkMlaWeights& weights,
                              const DSparkShape& shape) {
  const c10::InferenceMode inference_guard;
  shape.validate();
  require_rows(hidden, shape.hidden_size, "MLA input");
  validate_positions(positions, hidden.device(), hidden.size(0), shape);
  validate_context(context, shape, hidden.device());
  if (context.latent.size(0) + hidden.size(0) > shape.max_position) {
    throw std::invalid_argument("DSpark MLA context plus query exceeds capacity");
  }
  const at::Tensor expected_positions = at::arange(
      context.latent.size(0), context.latent.size(0) + hidden.size(0),
      positions.options());
  if (!(positions == expected_positions).all().item<bool>()) {
    throw std::invalid_argument(
        "DSpark MLA query must begin at the exact compact-cache boundary");
  }
  validate_mla_weights(weights, shape, hidden.device());

  const std::int64_t query_rows = hidden.size(0);
  const std::int64_t query_head_dim = shape.query_head_dim();
  const at::Tensor query_low = dspark_rms_norm_bf16(
      dense_linear(hidden, weights.query_a), weights.query_a_norm,
      shape.rms_epsilon);
  const at::Tensor query = dense_linear(query_low, weights.query_b)
                               .view({query_rows, shape.num_heads,
                                      query_head_dim});
  const at::Tensor query_nope =
      query.narrow(-1, 0, shape.qk_nope_head_dim);
  const at::Tensor query_rope = dspark_apply_yarn_rotary_bf16(
      query.narrow(-1, shape.qk_nope_head_dim, shape.qk_rope_head_dim)
          .contiguous(),
      positions, shape);

  const at::Tensor projected = dense_linear(hidden, weights.key_value_a);
  const at::Tensor query_latent = dspark_rms_norm_bf16(
      projected.narrow(-1, 0, shape.kv_lora_rank).contiguous(),
      weights.key_value_a_norm, shape.rms_epsilon);
  const at::Tensor query_key_rope = dspark_apply_yarn_rotary_bf16(
      projected
          .narrow(-1, shape.kv_lora_rank, shape.qk_rope_head_dim)
          .contiguous(),
      positions, shape);
  const at::Tensor all_latent =
      at::cat({context.latent, query_latent}, 0);
  const at::Tensor all_positional =
      at::cat({context.positional, query_key_rope}, 0);

  const at::Tensor up_weight = weights.key_value_b.view(
      {shape.num_heads, shape.qk_nope_head_dim + shape.value_head_dim,
       shape.kv_lora_rank});
  const at::Tensor key_weight =
      up_weight.narrow(1, 0, shape.qk_nope_head_dim);
  const at::Tensor value_weight =
      up_weight.narrow(1, shape.qk_nope_head_dim, shape.value_head_dim);
  const at::Tensor query_latent_space =
      at::einsum("qhd,hdl->qhl", {query_nope, key_weight});
  at::Tensor scores =
      at::einsum("qhl,kl->hqk", {query_latent_space, all_latent});
  scores = scores +
           at::einsum("qhr,kr->hqk", {query_rope, all_positional});
  const double attention_scale =
      std::pow(static_cast<double>(query_head_dim), -0.5) *
      std::pow(yarn_mscale(shape.rope_factor, shape.rope_mscale_all_dim), 2.0);
  const at::Tensor probabilities =
      at::softmax(scores.to(at::kFloat) * attention_scale, -1, at::kFloat)
          .to(at::kBFloat16);
  const at::Tensor latent_output =
      at::einsum("hqk,kl->qhl", {probabilities, all_latent});
  const at::Tensor value_output =
      at::einsum("qhl,hvl->qhv", {latent_output, value_weight});
  const at::Tensor output = dense_linear(
      value_output.reshape({query_rows, shape.num_heads * shape.value_head_dim}),
      weights.output);
  return DSparkMlaOutput{
      .hidden = output,
      .query_context = DSparkLatentContext{
          .latent = query_latent,
          .positional = query_key_rope,
      },
  };
}

at::Tensor run_dspark_mlp(const at::Tensor& hidden,
                          const DSparkMlpWeights& weights,
                          const DSparkShape& shape) {
  const c10::InferenceMode inference_guard;
  shape.validate();
  require_rows(hidden, shape.hidden_size, "MLP input");
  validate_linear(weights.gate, hidden.device(), shape.intermediate_size,
                  shape.hidden_size, "MLP gate weight");
  validate_linear(weights.up, hidden.device(), shape.intermediate_size,
                  shape.hidden_size, "MLP up weight");
  validate_linear(weights.down, hidden.device(), shape.hidden_size,
                  shape.intermediate_size, "MLP down weight");
  return dense_linear(at::silu(dense_linear(hidden, weights.gate)) *
                          dense_linear(hidden, weights.up),
                      weights.down);
}

DSparkDecoderOutput run_dspark_decoder_layer(
    const at::Tensor& hidden, const at::Tensor& residual,
    const at::Tensor& positions, const DSparkLatentContext& context,
    const DSparkDecoderWeights& weights, const DSparkShape& shape) {
  const c10::InferenceMode inference_guard;
  shape.validate();
  require_rows(hidden, shape.hidden_size, "decoder input");
  at::Tensor accumulated;
  if (residual.defined()) {
    require_tensor(residual, hidden.device(), at::kBFloat16, hidden.sizes(),
                   "decoder residual");
    accumulated = residual + hidden;
  } else {
    accumulated = hidden;
  }
  require_tensor(weights.input_norm, hidden.device(), at::kBFloat16,
                 {shape.hidden_size}, "decoder input norm");
  require_tensor(weights.post_attention_norm, hidden.device(), at::kBFloat16,
                 {shape.hidden_size}, "decoder post-attention norm");
  const at::Tensor normalized = dspark_rms_norm_bf16(
      accumulated, weights.input_norm, shape.rms_epsilon);
  DSparkMlaOutput attention = run_dspark_mla(
      normalized, positions, context, weights.attention, shape);
  accumulated = accumulated + attention.hidden;
  const at::Tensor post_attention = dspark_rms_norm_bf16(
      accumulated, weights.post_attention_norm, shape.rms_epsilon);
  return DSparkDecoderOutput{
      .hidden = run_dspark_mlp(post_attention, weights.mlp, shape),
      .residual = accumulated,
      .query_context = std::move(attention.query_context),
  };
}

}  // namespace deltafin::provider_internal
