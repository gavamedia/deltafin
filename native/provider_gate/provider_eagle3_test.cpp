#include "provider_eagle3.h"
#include "provider_moe.h"

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::DSparkMlaWeights;
using deltafin::provider_internal::DSparkMlpWeights;
using deltafin::provider_internal::Eagle3Model;
using deltafin::provider_internal::Eagle3Shape;
using deltafin::provider_internal::Eagle3Weights;
using deltafin::provider_internal::MoeRowInt8Matrix;
using deltafin::provider_internal::dspark_yarn_inverse_frequencies;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

at::Tensor deterministic(const at::IntArrayRef shape, const std::int64_t salt,
                         const float scale = 0.125F) {
  std::int64_t elements = 1;
  for (const std::int64_t dimension : shape) {
    elements *= dimension;
  }
  at::Tensor tensor = at::empty(shape, at::TensorOptions().dtype(at::kFloat));
  float* values = tensor.data_ptr<float>();
  for (std::int64_t index = 0; index < elements; ++index) {
    const std::int64_t centered =
        ((index + 1) * (salt * 11 + 13) + salt * 7) % 37 - 18;
    values[index] = static_cast<float>(centered) * scale / 18.0F;
  }
  return tensor;
}

at::Tensor bf16(const at::IntArrayRef shape, const std::int64_t salt,
                const float scale = 0.5F) {
  return deterministic(shape, salt, scale).to(at::kBFloat16).contiguous();
}

at::Tensor norm(const std::int64_t width, const std::int64_t salt) {
  return (bf16({width}, salt, 0.125F) +
          at::ones({width}, at::TensorOptions().dtype(at::kBFloat16)))
      .contiguous();
}

/* Rows 2i and 2i+1 are +/- the i-th unit vector, so the greedy token names
 * the final hidden state's dominant signed coordinate: a wrong attention or
 * residual path changes the proposals instead of hiding behind one row. */
at::Tensor discriminating_head(const Eagle3Shape& s) {
  at::Tensor head = deterministic({s.vocabulary_size, s.hidden_size}, 22, 0.01F);
  for (std::int64_t i = 0; i < s.hidden_size && 2 * i + 1 < s.vocabulary_size; ++i) {
    head[2 * i][i] = 2.0F;
    head[2 * i + 1][i] = -2.0F;
  }
  return head.to(at::kBFloat16).contiguous();
}

Eagle3Weights make_weights(const Eagle3Shape& s) {
  const std::int64_t h = s.hidden_size;
  return Eagle3Weights{
      .fc = bf16({h, s.tap_width()}, 1),
      .fc_norm = {norm(h, 2), norm(h, 3), norm(h, 4)},
      .hidden_norm = norm(h, 5),
      .input_norm = norm(h, 6),
      .attention =
          DSparkMlaWeights{
              .query_a = bf16({s.q_lora_rank, 2 * h}, 7),
              .query_a_norm = norm(s.q_lora_rank, 8),
              .query_b = bf16({s.num_heads * s.query_head_dim(), s.q_lora_rank}, 9),
              .key_value_a = bf16({s.kv_lora_rank + s.qk_rope_head_dim, 2 * h}, 10),
              .key_value_a_norm = norm(s.kv_lora_rank, 11),
              .key_value_b = bf16(
                  {s.num_heads * (s.qk_nope_head_dim + s.value_head_dim), s.kv_lora_rank},
                  12),
              .output = bf16({h, s.num_heads * s.value_head_dim}, 13),
          },
      .post_attention_norm = norm(h, 14),
      .mlp =
          DSparkMlpWeights{
              .gate = bf16({s.intermediate_size, h}, 15),
              .up = bf16({s.intermediate_size, h}, 16),
              .down = bf16({h, s.intermediate_size}, 17),
          },
      .final_norm = norm(h, 18),
      .language_model_head = discriminating_head(s),
  };
}

MoeRowInt8Matrix make_embedding(const Eagle3Shape& s) {
  at::Tensor quantized = at::empty({s.vocabulary_size, s.hidden_size},
                                   at::TensorOptions().dtype(at::kChar));
  auto values = quantized.accessor<std::int8_t, 2>();
  for (std::int64_t row = 0; row < s.vocabulary_size; ++row) {
    for (std::int64_t column = 0; column < s.hidden_size; ++column) {
      values[row][column] =
          static_cast<std::int8_t>(((row + 2) * (column + 5)) % 13 - 6);
    }
  }
  return MoeRowInt8Matrix{
      .quantized = quantized,
      .row_scales = (deterministic({s.vocabulary_size}, 21, 0.05F).abs() + 0.05F)
                        .contiguous(),
      .dense_f32 = {},
      .original_bf16 = {},
  };
}

/* The exact BF16 rows a caller passes in (here: a synthetic int8 table). */
at::Tensor embed(const MoeRowInt8Matrix& table, const std::vector<std::int64_t>& ids) {
  std::vector<at::Tensor> rows;
  for (const std::int64_t id : ids) {
    rows.push_back(table.quantized[id].to(at::kFloat) * table.row_scales[id].item<float>());
  }
  return at::stack(rows).to(at::kBFloat16).contiguous();
}

/* A model driven exactly as the backend drives it: one call per draft. */
std::vector<std::int64_t> propose(Eagle3Model& model, const MoeRowInt8Matrix& table,
                                  const std::int64_t pending, const std::int64_t width) {
  std::vector<std::int64_t> drafts{model.propose_begin(embed(table, {pending}))};
  while (static_cast<std::int64_t>(drafts.size()) < width) {
    drafts.push_back(model.propose_step(embed(table, {drafts.back()})));
  }
  model.propose_end();
  return drafts;
}

/* Independent fp32 reference with expanded per-head keys and values. */
struct Reference {
  Eagle3Shape s;
  Eagle3Weights w;
  MoeRowInt8Matrix embedding;
  at::Tensor inverse;
  std::vector<at::Tensor> latents;    // committed [L] rows
  std::vector<at::Tensor> rotaries;   // committed [R] rows

  static at::Tensor f(const at::Tensor& value) { return value.to(at::kFloat); }

  at::Tensor rms(const at::Tensor& value, const at::Tensor& weight) const {
    const at::Tensor x = f(value);
    return x * at::rsqrt(at::mean(x * x, {-1}, true) + s.rms_epsilon) * f(weight);
  }

  at::Tensor rotate(const at::Tensor& value, const std::int64_t position) const {
    const at::Tensor phase = inverse * static_cast<float>(position);
    const at::Tensor cosine = at::cos(phase);
    const at::Tensor sine = at::sin(phase);
    const at::Tensor pairs = value.reshape({-1, s.qk_rope_head_dim / 2, 2});
    const at::Tensor even = pairs.select(-1, 0);
    const at::Tensor odd = pairs.select(-1, 1);
    return at::stack({even * cosine - odd * sine, odd * cosine + even * sine}, -1)
        .reshape(value.sizes());
  }

  at::Tensor fuse(const at::Tensor& taps) const {
    std::vector<at::Tensor> parts;
    for (std::int64_t index = 0; index < 3; ++index) {
      parts.push_back(rms(taps.narrow(-1, index * s.hidden_size, s.hidden_size),
                          w.fc_norm[static_cast<std::size_t>(index)]));
    }
    return at::matmul(at::cat(parts, -1), f(w.fc).t());
  }

  /* One row at position latents.size(); returns (hidden, logits). */
  std::pair<at::Tensor, at::Tensor> step(const std::int64_t token,
                                         const at::Tensor& hidden,
                                         const bool keep) {
    const std::int64_t position = static_cast<std::int64_t>(latents.size());
    const std::int64_t heads = s.num_heads;
    const std::int64_t nope = s.qk_nope_head_dim;
    const std::int64_t rope = s.qk_rope_head_dim;
    const std::int64_t value_dim = s.value_head_dim;
    const at::Tensor emb = f(embed(embedding, {token})[0]);
    const at::Tensor x = at::cat({rms(emb, w.input_norm), rms(hidden, w.hidden_norm)});
    const at::Tensor q_low =
        rms(at::matmul(f(w.attention.query_a), x), w.attention.query_a_norm);
    const at::Tensor q =
        at::matmul(f(w.attention.query_b), q_low).view({heads, s.query_head_dim()});
    const at::Tensor q_nope = q.narrow(-1, 0, nope);
    at::Tensor q_rope = q.narrow(-1, nope, rope).contiguous();
    for (std::int64_t head = 0; head < heads; ++head) {
      q_rope[head].copy_(rotate(q_rope[head], position));
    }
    const at::Tensor projected = at::matmul(f(w.attention.key_value_a), x);
    const at::Tensor latent =
        rms(projected.narrow(0, 0, s.kv_lora_rank), w.attention.key_value_a_norm);
    const at::Tensor key_rope = rotate(projected.narrow(0, s.kv_lora_rank, rope), position);
    std::vector<at::Tensor> all_latent = latents;
    std::vector<at::Tensor> all_rope = rotaries;
    all_latent.push_back(latent);
    all_rope.push_back(key_rope);
    const at::Tensor up = f(w.attention.key_value_b);  // [H*(nope+v), L]
    const double scale = std::pow(static_cast<double>(s.query_head_dim()), -0.5) *
                         std::pow(0.1 * s.rope.rope_mscale_all_dim *
                                          std::log(s.rope.rope_factor) +
                                      1.0,
                                  2.0);
    at::Tensor out = at::zeros({heads, value_dim});
    for (std::int64_t head = 0; head < heads; ++head) {
      const at::Tensor head_up =
          up.narrow(0, head * (nope + value_dim), nope + value_dim);
      std::vector<float> scores;
      std::vector<at::Tensor> values;
      for (std::size_t key = 0; key < all_latent.size(); ++key) {
        const at::Tensor expanded = at::matmul(head_up, all_latent[key]);
        const at::Tensor key_nope = expanded.narrow(0, 0, nope);
        values.push_back(expanded.narrow(0, nope, value_dim));
        scores.push_back(static_cast<float>(
            (at::dot(q_nope[head], key_nope) + at::dot(q_rope[head], all_rope[key]))
                .item<float>() *
            scale));
      }
      const at::Tensor probabilities =
          at::softmax(at::tensor(scores), 0);
      for (std::size_t key = 0; key < values.size(); ++key) {
        out[head] += probabilities[static_cast<std::int64_t>(key)] * values[key];
      }
    }
    const at::Tensor attended =
        at::matmul(f(w.attention.output), out.reshape({heads * value_dim}));
    const at::Tensor residual = attended + hidden;
    const at::Tensor post = rms(residual, w.post_attention_norm);
    const at::Tensor mlp = at::matmul(
        f(w.mlp.down),
        at::silu(at::matmul(f(w.mlp.gate), post)) * at::matmul(f(w.mlp.up), post));
    const at::Tensor output = rms(mlp + residual, w.final_norm);
    if (keep) {
      latents.push_back(latent);
      rotaries.push_back(key_rope);
    }
    return {output, at::matmul(f(w.language_model_head), output)};
  }
};

/* Greedy chain from the reference, with each logit margin. */
std::vector<std::pair<std::int64_t, float>> reference_chain(
    Reference& reference, const at::Tensor& taps,
    const std::vector<std::int64_t>& inputs, const std::int64_t pending_token,
    const std::int64_t width) {
  const std::int64_t rows = taps.size(0);
  for (std::int64_t row = 0; row + 1 < rows; ++row) {
    (void)reference.step(inputs[static_cast<std::size_t>(row + 1)],
                         reference.fuse(Reference::f(taps[row])), true);
  }
  std::vector<std::pair<std::int64_t, float>> chain;
  const std::size_t committed = reference.latents.size();
  auto [hidden, logits] = reference.step(
      pending_token, reference.fuse(Reference::f(taps[rows - 1])), true);
  for (std::int64_t index = 0; index < width; ++index) {
    const auto top = at::topk(logits, 2);
    const std::int64_t token = std::get<1>(top)[0].item<std::int64_t>();
    const float margin = (std::get<0>(top)[0] - std::get<0>(top)[1]).item<float>();
    chain.emplace_back(token, margin);
    if (index + 1 < width) {
      std::tie(hidden, logits) = reference.step(token, hidden, true);
    }
  }
  reference.latents.resize(committed);
  reference.rotaries.resize(committed);
  return chain;
}

void test_matches_reference_and_protocol() {
  const Eagle3Shape shape = Eagle3Shape::small_canary();
  const Eagle3Weights weights = make_weights(shape);
  const MoeRowInt8Matrix embedding = make_embedding(shape);
  const std::int64_t rows = 6;
  const at::Tensor taps = bf16({rows, shape.tap_width()}, 31, 1.0F);
  const std::vector<std::int64_t> inputs{3, 17, 9, 25, 4, 11};
  const std::int64_t pending_token = 7;
  const std::int64_t width = 6;

  Eagle3Model batched(shape, weights, false);
  batched.advance(taps, embed(embedding, inputs));
  require(batched.length() == rows - 1 && batched.token_count() == rows,
          "advance must hold the newest row pending");
  const std::vector<std::int64_t> proposal = propose(batched, embedding, pending_token, width);
  require(static_cast<std::int64_t>(proposal.size()) == width,
          "proposal width must follow the request");
  require(batched.token_count() == rows, "proposing must not commit rows");

  Reference reference{shape, weights, embedding,
                      dspark_yarn_inverse_frequencies(shape.rope, at::Device(at::kCPU)),
                      {}, {}};
  const auto expected = reference_chain(reference, taps, inputs, pending_token, width);
  std::size_t compared = 0;
  // Agreeing drafts count whatever their margin. A disagreement is only
  // acceptable at a near-tie, which BF16 may round either way; past it the
  // two chains legitimately follow different tokens.
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (proposal[index] != expected[index].first) {
      require(expected[index].second < 0.05F,
              "draft " + std::to_string(index) + " differs from the fp32 reference");
      break;
    }
    ++compared;
  }
  require(compared >= 3, "reference comparison needs clear logit margins");
  std::vector<std::int64_t> distinct(proposal.begin(), proposal.end());
  std::sort(distinct.begin(), distinct.end());
  require(std::unique(distinct.begin(), distinct.end()) - distinct.begin() >= 2,
          "the canary must propose more than one distinct token");
  // Other pending tokens exercise other embeddings through the same cache.
  for (const std::int64_t other : {2, 13, 30}) {
    Reference fresh{shape, weights, embedding,
                    dspark_yarn_inverse_frequencies(shape.rope, at::Device(at::kCPU)),
                    {}, {}};
    const auto other_expected = reference_chain(fresh, taps, inputs, other, width);
    const std::vector<std::int64_t> other_proposal = propose(batched, embedding, other, width);
    for (std::size_t index = 0; index < other_expected.size(); ++index) {
      if (other_proposal[index] != other_expected[index].first) {
        require(other_expected[index].second < 0.05F,
                "pending " + std::to_string(other) + " draft " +
                    std::to_string(index) + " differs from the fp32 reference");
        break;
      }
    }
  }

  // Row-by-row advance must equal the batched causal pass.
  Eagle3Model incremental(shape, weights, false);
  for (std::int64_t row = 0; row < rows; ++row) {
    incremental.advance(taps.narrow(0, row, 1).contiguous(),
                        embed(embedding, {inputs[static_cast<std::size_t>(row)]}));
  }
  const std::vector<std::int64_t> stepwise = propose(incremental, embedding, pending_token, width);
  for (std::size_t index = 0; index < compared; ++index) {
    require(stepwise[index] == proposal[index],
            "row-by-row and batched advances must agree");
  }

  // A snapshot restores both the cache prefix and the pending row.
  const auto saved = batched.snapshot();
  batched.advance(bf16({2, shape.tap_width()}, 41, 1.0F), embed(embedding, {pending_token, 5}));
  require(batched.token_count() == rows + 2, "second advance must commit two rows");
  batched.restore(saved);
  require(batched.token_count() == rows && batched.length() == rows - 1,
          "restore must return to the snapshot");
  require(propose(batched, embedding, pending_token, width) == proposal,
          "a restored model must propose the same chain");
  // An abandoned chain leaves the committed state untouched.
  (void)batched.propose_begin(embed(embedding, {pending_token}));
  require(batched.proposing() && batched.token_count() == rows,
          "an open chain must not commit rows");
  batched.propose_end();
  require(!batched.proposing() && batched.length() == rows - 1,
          "ending a chain must discard its speculative rows");

  batched.clear();
  require(batched.token_count() == 0, "clear must empty the model");
  bool refused = false;
  try {
    (void)batched.propose_begin(embed(embedding, {pending_token}));
  } catch (const std::logic_error&) {
    refused = true;
  }
  require(refused, "proposing without a pending row must be refused");
  refused = false;
  try {
    batched.advance(taps.narrow(0, 0, 1).contiguous(), bf16({1, shape.hidden_size + 1}, 52));
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "malformed embeddings must be refused");
  refused = false;
  try {
    Eagle3Model small(shape, weights, false);
    small.advance(bf16({shape.max_position + 2, shape.tap_width()}, 51, 1.0F),
                  bf16({shape.max_position + 2, shape.hidden_size}, 53));
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "context beyond capacity must be refused");
}

}  // namespace

int main() {
  try {
    const c10::InferenceMode inference_guard;
    test_matches_reference_and_protocol();
    std::cout << "provider_eagle3.synthetic=PASS\n";
    std::cout << "provider_eagle3.authority=PROPOSAL_ONLY\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_eagle3.synthetic=FAIL: " << error.what() << '\n';
    return 1;
  }
}
