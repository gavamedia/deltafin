#include "provider_eagle3.h"

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace deltafin::provider_internal {
namespace {

void require_bf16(const at::Tensor& value, const at::Device& device,
                  at::IntArrayRef sizes, const char* label) {
  if (!value.defined() || value.scalar_type() != at::kBFloat16 ||
      value.device() != device || !value.is_contiguous() ||
      value.sizes() != sizes) {
    throw std::invalid_argument(std::string("EAGLE-3 ") + label +
                                " violates its BF16 shape/device contract");
  }
}

at::Tensor dense(const at::Tensor& input, const at::Tensor& weight) {
  if (input.dim() != 2 || weight.dim() != 2 ||
      input.size(1) != weight.size(1)) {
    throw std::invalid_argument("EAGLE-3 linear dimensions disagree");
  }
  return at::linear(input, weight, std::nullopt);
}

double yarn_mscale(const double factor, const double multiplier) {
  return factor <= 1.0 ? 1.0 : 0.1 * multiplier * std::log(factor) + 1.0;
}

/* DeepSeek MLA rotary: adjacent (even, odd) pairs, any row count. */
at::Tensor rotate_pairs(const at::Tensor& value, const at::Tensor& positions,
                        const at::Tensor& inverse_frequencies,
                        const double scale) {
  const at::Tensor phase =
      positions.to(at::kFloat).unsqueeze(1) * inverse_frequencies;
  at::Tensor cosine = at::cos(phase) * scale;
  at::Tensor sine = at::sin(phase) * scale;
  for (std::int64_t dimension = 1; dimension < value.dim() - 1; ++dimension) {
    cosine = cosine.unsqueeze(1);
    sine = sine.unsqueeze(1);
  }
  const auto original_shape = value.sizes().vec();
  auto pair_shape = original_shape;
  pair_shape.back() = value.size(-1) / 2;
  pair_shape.push_back(2);
  const at::Tensor pairs = value.to(at::kFloat).reshape(pair_shape);
  const at::Tensor even = pairs.select(-1, 0);
  const at::Tensor odd = pairs.select(-1, 1);
  return at::stack({even * cosine - odd * sine, odd * cosine + even * sine}, -1)
      .reshape(original_shape)
      .to(at::kBFloat16);
}

/* vLLM's fused residual add + RMSNorm: the fp32 sum is both the new
 * residual (cast) and the normalized input (cast, then weighted). */
std::pair<at::Tensor, at::Tensor> add_rms_norm(const at::Tensor& value,
                                               const at::Tensor& residual,
                                               const at::Tensor& weight,
                                               const double epsilon) {
  const at::Tensor sum = value.to(at::kFloat) + residual.to(at::kFloat);
  const at::Tensor variance = at::mean(at::pow(sum, 2), {-1}, true);
  const at::Tensor normalized =
      (sum * at::rsqrt(variance + epsilon)).to(at::kBFloat16) * weight;
  return {normalized.contiguous(), sum.to(at::kBFloat16).contiguous()};
}

}  // namespace

Eagle3Shape Eagle3Shape::k3(const std::int64_t max_position) {
  return Eagle3Shape{
      .hidden_size = 7168,
      .intermediate_size = 18432,
      .num_heads = 64,
      .q_lora_rank = 1536,
      .kv_lora_rank = 512,
      .qk_nope_head_dim = 128,
      .qk_rope_head_dim = 64,
      .value_head_dim = 128,
      .vocabulary_size = 163840,
      .max_position = max_position,
      .rms_epsilon = 1.0e-5,
      .rope = DSparkShape::k3(),
  };
}

Eagle3Shape Eagle3Shape::small_canary() {
  const DSparkShape rope = DSparkShape::small_canary();
  return Eagle3Shape{
      .hidden_size = 8,
      .intermediate_size = 12,
      .num_heads = 2,
      .q_lora_rank = 4,
      .kv_lora_rank = 4,
      .qk_nope_head_dim = 2,
      .qk_rope_head_dim = rope.qk_rope_head_dim,
      .value_head_dim = 2,
      .vocabulary_size = 32,
      .max_position = 32,
      .rms_epsilon = 1.0e-5,
      .rope = rope,
  };
}

void Eagle3Shape::validate() const {
  if (hidden_size < 1 || intermediate_size < 1 || num_heads < 1 ||
      q_lora_rank < 1 || kv_lora_rank < 1 || qk_nope_head_dim < 1 ||
      qk_rope_head_dim < 2 || qk_rope_head_dim % 2 != 0 ||
      value_head_dim < 1 || vocabulary_size < 2 || max_position < 1 ||
      !(rms_epsilon > 0.0) || !std::isfinite(rms_epsilon)) {
    throw std::invalid_argument("EAGLE-3 shape is not positive and finite");
  }
  rope.validate();
  if (rope.qk_rope_head_dim != qk_rope_head_dim) {
    throw std::invalid_argument("EAGLE-3 rotary schedule has another width");
  }
}

bool Eagle3Shape::is_exact_k3() const {
  const Eagle3Shape exact = k3(max_position);
  return hidden_size == exact.hidden_size &&
         intermediate_size == exact.intermediate_size &&
         num_heads == exact.num_heads && q_lora_rank == exact.q_lora_rank &&
         kv_lora_rank == exact.kv_lora_rank &&
         qk_nope_head_dim == exact.qk_nope_head_dim &&
         qk_rope_head_dim == exact.qk_rope_head_dim &&
         value_head_dim == exact.value_head_dim &&
         vocabulary_size == exact.vocabulary_size &&
         rms_epsilon == exact.rms_epsilon && rope.is_exact_k3();
}

std::int64_t Eagle3Shape::query_head_dim() const {
  return qk_nope_head_dim + qk_rope_head_dim;
}

std::int64_t Eagle3Shape::tap_width() const {
  return static_cast<std::int64_t>(kEagle3TargetCaptureLayers.size()) *
         hidden_size;
}

Eagle3Model::Eagle3Model(Eagle3Shape shape, Eagle3Weights weights,
                         const bool exact_k3)
    : shape_(std::move(shape)),
      weights_(std::move(weights)),
      exact_k3_(exact_k3) {
  const c10::InferenceMode inference_guard;
  shape_.validate();
  if (exact_k3_ && !shape_.is_exact_k3()) {
    throw std::invalid_argument("EAGLE-3 exact mode refuses a non-K3 shape");
  }
  const std::int64_t h = shape_.hidden_size;
  const at::Device device = weights_.fc.device();
  require_bf16(weights_.fc, device, {h, shape_.tap_width()}, "fusion");
  for (const at::Tensor& norm : weights_.fc_norm) {
    require_bf16(norm, device, {h}, "fusion norm");
  }
  require_bf16(weights_.hidden_norm, device, {h}, "hidden norm");
  require_bf16(weights_.input_norm, device, {h}, "input norm");
  require_bf16(weights_.post_attention_norm, device, {h},
               "post-attention norm");
  require_bf16(weights_.final_norm, device, {h}, "final norm");
  require_bf16(weights_.attention.query_a, device, {shape_.q_lora_rank, 2 * h},
               "query down-projection");
  require_bf16(weights_.attention.query_a_norm, device, {shape_.q_lora_rank},
               "query norm");
  require_bf16(weights_.attention.query_b, device,
               {shape_.num_heads * shape_.query_head_dim(), shape_.q_lora_rank},
               "query up-projection");
  require_bf16(weights_.attention.key_value_a, device,
               {shape_.kv_lora_rank + shape_.qk_rope_head_dim, 2 * h},
               "latent down-projection");
  require_bf16(weights_.attention.key_value_a_norm, device,
               {shape_.kv_lora_rank}, "latent norm");
  require_bf16(weights_.attention.key_value_b, device,
               {shape_.num_heads * (shape_.qk_nope_head_dim + shape_.value_head_dim),
                shape_.kv_lora_rank},
               "latent up-projection");
  require_bf16(weights_.attention.output, device,
               {h, shape_.num_heads * shape_.value_head_dim},
               "attention output");
  require_bf16(weights_.mlp.gate, device, {shape_.intermediate_size, h},
               "MLP gate");
  require_bf16(weights_.mlp.up, device, {shape_.intermediate_size, h},
               "MLP up");
  require_bf16(weights_.mlp.down, device, {h, shape_.intermediate_size},
               "MLP down");
  require_bf16(weights_.language_model_head, device,
               {shape_.vocabulary_size, h}, "language-model head");
  inverse_frequencies_ = dspark_yarn_inverse_frequencies(shape_.rope, device);
  const auto options =
      at::TensorOptions().dtype(at::kBFloat16).device(device);
  latent_ = at::zeros({shape_.max_position, shape_.kv_lora_rank}, options);
  positional_ =
      at::zeros({shape_.max_position, shape_.qk_rope_head_dim}, options);
}

const Eagle3Shape& Eagle3Model::shape() const noexcept { return shape_; }

std::int64_t Eagle3Model::length() const noexcept {
  // Speculative chain rows are never committed state.
  return chain_base_ >= 0 ? chain_base_ : length_;
}

std::int64_t Eagle3Model::token_count() const noexcept {
  return length() + (pending_taps_.defined() ? 1 : 0);
}

at::Tensor Eagle3Model::fuse(const at::Tensor& taps) const {
  const std::int64_t h = shape_.hidden_size;
  std::vector<at::Tensor> normalized;
  normalized.reserve(kEagle3TargetCaptureLayers.size());
  for (std::size_t index = 0; index < kEagle3TargetCaptureLayers.size();
       ++index) {
    normalized.push_back(dspark_rms_norm_bf16(
        taps.narrow(1, static_cast<std::int64_t>(index) * h, h).contiguous(),
        weights_.fc_norm[index], shape_.rms_epsilon));
  }
  return dense(at::cat(normalized, 1), weights_.fc).contiguous();
}

void Eagle3Model::require_embeddings(const at::Tensor& embeddings,
                                     const std::int64_t rows,
                                     const char* label) const {
  require_bf16(embeddings, latent_.device(), {rows, shape_.hidden_size}, label);
}

Eagle3Model::StepOutput Eagle3Model::forward(const at::Tensor& embeddings,
                                             const at::Tensor& hidden,
                                             const bool score) {
  const std::int64_t rows = embeddings.size(0);
  const std::int64_t h = shape_.hidden_size;
  if (rows < 1 || length_ + rows > shape_.max_position) {
    throw std::invalid_argument(
        "EAGLE-3 context would exceed the drafter's cache capacity");
  }
  const at::Device device = latent_.device();
  const at::Tensor positions = at::arange(
      length_, length_ + rows,
      at::TensorOptions().dtype(at::kLong).device(device));

  // First (and only) decoder layer: [norm(embedding), norm(hidden)].
  const at::Tensor embeds =
      dspark_rms_norm_bf16(embeddings, weights_.input_norm, shape_.rms_epsilon);
  const at::Tensor residual = hidden;
  const at::Tensor hidden_normed =
      dspark_rms_norm_bf16(hidden, weights_.hidden_norm, shape_.rms_epsilon);
  const at::Tensor input = at::cat({embeds, hidden_normed}, 1).contiguous();

  // Multi-head latent attention over the cache, causal within this call.
  const DSparkMlaWeights& attention = weights_.attention;
  const std::int64_t heads = shape_.num_heads;
  const std::int64_t nope = shape_.qk_nope_head_dim;
  const std::int64_t rope = shape_.qk_rope_head_dim;
  const std::int64_t value = shape_.value_head_dim;
  const std::int64_t latent_width = shape_.kv_lora_rank;
  const double rope_scale = yarn_mscale(shape_.rope.rope_factor,
                                        shape_.rope.rope_mscale) /
                            yarn_mscale(shape_.rope.rope_factor,
                                        shape_.rope.rope_mscale_all_dim);
  const at::Tensor query_low = dspark_rms_norm_bf16(
      dense(input, attention.query_a), attention.query_a_norm,
      shape_.rms_epsilon);
  const at::Tensor query = dense(query_low, attention.query_b)
                               .view({rows, heads, shape_.query_head_dim()});
  const at::Tensor query_nope = query.narrow(-1, 0, nope);
  const at::Tensor query_rope =
      rotate_pairs(query.narrow(-1, nope, rope).contiguous(), positions,
                   inverse_frequencies_, rope_scale);
  const at::Tensor projected = dense(input, attention.key_value_a);
  const at::Tensor new_latent = dspark_rms_norm_bf16(
      projected.narrow(-1, 0, latent_width).contiguous(),
      attention.key_value_a_norm, shape_.rms_epsilon);
  const at::Tensor new_positional =
      rotate_pairs(projected.narrow(-1, latent_width, rope).contiguous(),
                   positions, inverse_frequencies_, rope_scale);
  latent_.narrow(0, length_, rows).copy_(new_latent);
  positional_.narrow(0, length_, rows).copy_(new_positional);
  const std::int64_t keys = length_ + rows;
  const at::Tensor all_latent = latent_.narrow(0, 0, keys);
  const at::Tensor all_positional = positional_.narrow(0, 0, keys);
  const at::Tensor up_weight =
      attention.key_value_b.view({heads, nope + value, latent_width});
  const at::Tensor key_weight = up_weight.narrow(1, 0, nope);
  const at::Tensor value_weight = up_weight.narrow(1, nope, value);
  const at::Tensor query_latent =
      at::einsum("qhd,hdl->qhl", {query_nope, key_weight});
  at::Tensor scores = at::einsum("qhl,kl->hqk", {query_latent, all_latent})
                          .to(at::kFloat) +
                      at::einsum("qhr,kr->hqk", {query_rope, all_positional})
                          .to(at::kFloat);
  const double attention_scale =
      std::pow(static_cast<double>(shape_.query_head_dim()), -0.5) *
      std::pow(yarn_mscale(shape_.rope.rope_factor,
                           shape_.rope.rope_mscale_all_dim),
               2.0);
  scores = scores * attention_scale;
  if (rows > 1) {
    const auto long_options =
        at::TensorOptions().dtype(at::kLong).device(device);
    const at::Tensor visible =
        at::arange(keys, long_options).unsqueeze(0) <=
        (at::arange(rows, long_options) + length_).unsqueeze(1);
    scores = scores.masked_fill(at::logical_not(visible).unsqueeze(0),
                                -std::numeric_limits<float>::infinity());
  }
  const at::Tensor probabilities =
      at::softmax(scores, -1, at::kFloat).to(at::kBFloat16);
  const at::Tensor latent_output =
      at::einsum("hqk,kl->qhl", {probabilities, all_latent});
  const at::Tensor value_output =
      at::einsum("qhl,hvl->qhv", {latent_output, value_weight});
  const at::Tensor attended = dense(
      value_output.reshape({rows, heads * value}).contiguous(),
      attention.output);

  auto [post_attention, residual_after_attention] =
      add_rms_norm(attended, residual, weights_.post_attention_norm,
                   shape_.rms_epsilon);
  const at::Tensor mlp =
      dense(at::silu(dense(post_attention, weights_.mlp.gate)) *
                 dense(post_attention, weights_.mlp.up),
             weights_.mlp.down);
  auto [output, unused_residual] = add_rms_norm(
      mlp, residual_after_attention, weights_.final_norm, shape_.rms_epsilon);
  (void)unused_residual;
  length_ += rows;
  StepOutput step{.hidden = output, .tokens = at::Tensor()};
  if (score) {
    step.tokens =
        dense(output, weights_.language_model_head).argmax(-1).to(at::kLong);
  }
  (void)h;
  return step;
}

void Eagle3Model::advance(const at::Tensor& taps,
                          const at::Tensor& input_embeddings) {
  const c10::InferenceMode inference_guard;
  if (proposing()) {
    throw std::logic_error("EAGLE-3 cannot advance during a proposal");
  }
  const std::int64_t rows = taps.dim() == 2 ? taps.size(0) : 0;
  if (rows < 1) {
    throw std::invalid_argument("EAGLE-3 advance needs at least one row");
  }
  require_bf16(taps, latent_.device(), {rows, shape_.tap_width()},
               "target capture rows");
  require_embeddings(input_embeddings, rows, "input embeddings");
  const bool pending = pending_taps_.defined();
  const std::int64_t appended = rows - 1 + (pending ? 1 : 0);
  if (length_ + appended > shape_.max_position) {
    throw std::invalid_argument(
        "EAGLE-3 context would exceed the drafter's cache capacity");
  }
  // Each committed row pairs with the token that followed it: the pending
  // row with this call's first input, row i with input i + 1.
  if (appended > 0) {
    std::vector<at::Tensor> rows_to_append;
    if (pending) {
      rows_to_append.push_back(pending_taps_);
    }
    if (rows > 1) {
      rows_to_append.push_back(taps.narrow(0, 0, rows - 1));
    }
    const at::Tensor successors =
        pending ? input_embeddings
                : input_embeddings.narrow(0, 1, rows - 1).contiguous();
    (void)forward(successors, fuse(at::cat(rows_to_append, 0).contiguous()),
                  false);
  }
  pending_taps_ = taps.narrow(0, rows - 1, 1).clone();
}

bool Eagle3Model::proposing() const noexcept { return chain_base_ >= 0; }

std::int64_t Eagle3Model::propose_begin(const at::Tensor& pending_embedding) {
  const c10::InferenceMode inference_guard;
  if (!pending_taps_.defined()) {
    throw std::logic_error("EAGLE-3 proposal needs a pending committed row");
  }
  if (proposing()) {
    throw std::logic_error("EAGLE-3 proposal is already active");
  }
  require_embeddings(pending_embedding, 1, "pending embedding");
  if (length_ >= shape_.max_position) {
    throw std::invalid_argument("EAGLE-3 cache has no room to propose");
  }
  chain_base_ = length_;
  try {
    StepOutput step = forward(pending_embedding, fuse(pending_taps_), true);
    chain_hidden_ = step.hidden;
    return step.tokens.item<std::int64_t>();
  } catch (...) {
    propose_end();
    throw;
  }
}

std::int64_t Eagle3Model::propose_step(const at::Tensor& draft_embedding) {
  const c10::InferenceMode inference_guard;
  if (!proposing()) {
    throw std::logic_error("EAGLE-3 proposal step without an active chain");
  }
  require_embeddings(draft_embedding, 1, "draft embedding");
  if (length_ - chain_base_ >= kEagle3MaximumDrafts ||
      length_ >= shape_.max_position) {
    throw std::invalid_argument("EAGLE-3 proposal chain is at its limit");
  }
  try {
    StepOutput step = forward(draft_embedding, chain_hidden_, true);
    chain_hidden_ = step.hidden;
    return step.tokens.item<std::int64_t>();
  } catch (...) {
    propose_end();
    throw;
  }
}

void Eagle3Model::propose_end() noexcept {
  if (chain_base_ >= 0) {
    length_ = chain_base_;
  }
  chain_base_ = -1;
  chain_hidden_ = at::Tensor();
}

Eagle3Snapshot Eagle3Model::snapshot() const {
  if (proposing()) {
    throw std::logic_error("EAGLE-3 cannot snapshot during a proposal");
  }
  return Eagle3Snapshot{
      .owner = this,
      .length = length_,
      .epoch = epoch_,
      .pending_taps = pending_taps_,
      .latent = latent_.narrow(0, 0, length_).clone(),
      .positional = positional_.narrow(0, 0, length_).clone(),
  };
}

void Eagle3Model::restore(const Eagle3Snapshot& snapshot) {
  const c10::InferenceMode inference_guard;
  propose_end();
  if (snapshot.owner != this || snapshot.length < 0 ||
      snapshot.length > shape_.max_position ||
      snapshot.latent.size(0) != snapshot.length ||
      snapshot.positional.size(0) != snapshot.length) {
    throw std::invalid_argument("EAGLE-3 snapshot belongs to another model");
  }
  if (snapshot.length != 0) {
    latent_.narrow(0, 0, snapshot.length).copy_(snapshot.latent);
    positional_.narrow(0, 0, snapshot.length).copy_(snapshot.positional);
  }
  length_ = snapshot.length;
  pending_taps_ = snapshot.pending_taps;
  ++epoch_;
}

void Eagle3Model::clear() {
  propose_end();
  length_ = 0;
  pending_taps_ = at::Tensor();
  ++epoch_;
}

}  // namespace deltafin::provider_internal
