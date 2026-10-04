#ifndef DELTAFIN_PROVIDER_EAGLE3_H
#define DELTAFIN_PROVIDER_EAGLE3_H

#include "provider_dspark.h"

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace deltafin::provider_internal {

/*
 * Native arithmetic for an EAGLE-3.1 single-layer MLA proposal model
 * (lightseekorg/kimi-k3-eagle3.1-mla, vLLM's Eagle3DeepseekV2ForCausalLM).
 * Like DSpark, this surface can only transform tensors into untrusted token
 * IDs; full K3 verifies every row before anything is emitted.
 *
 * The drafter reads K3's within-block AttnRes prefix sum after these
 * zero-based layers: vLLM aux ids 2/46/90 captured with
 * VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=0, the stream it was trained on.
 */
constexpr std::array<std::uint32_t, 3> kEagle3TargetCaptureLayers{1, 45, 89};
constexpr std::int64_t kEagle3MaximumDrafts = 7;

struct Eagle3Shape {
  std::int64_t hidden_size = 0;
  std::int64_t intermediate_size = 0;
  std::int64_t num_heads = 0;
  std::int64_t q_lora_rank = 0;
  std::int64_t kv_lora_rank = 0;
  std::int64_t qk_nope_head_dim = 0;
  std::int64_t qk_rope_head_dim = 0;
  std::int64_t value_head_dim = 0;
  std::int64_t vocabulary_size = 0;
  std::int64_t max_position = 0;
  double rms_epsilon = 0.0;
  /* YaRN rotary parameters; the drafter's MLA uses K3's own schedule. */
  DSparkShape rope;

  [[nodiscard]] static Eagle3Shape k3(std::int64_t max_position);
  [[nodiscard]] static Eagle3Shape small_canary();
  void validate() const;
  [[nodiscard]] bool is_exact_k3() const;
  [[nodiscard]] std::int64_t query_head_dim() const;
  [[nodiscard]] std::int64_t tap_width() const;
};

struct Eagle3Weights {
  at::Tensor fc;                      // [H, 3H]
  std::array<at::Tensor, 3> fc_norm;  // [H] each
  at::Tensor hidden_norm;             // [H]
  at::Tensor input_norm;              // [H]
  DSparkMlaWeights attention;         // query_a/key_value_a read 2H inputs
  at::Tensor post_attention_norm;     // [H]
  DSparkMlpWeights mlp;
  at::Tensor final_norm;              // [H]
  at::Tensor language_model_head;     // [V, H]
};

struct Eagle3Snapshot {
  const void* owner = nullptr;
  std::int64_t length = 0;
  std::uint64_t epoch = 0;
  /* The newest committed row, held until its successor token is known. */
  at::Tensor pending_taps;
  /* Copies of the committed cache prefix: later appends may overwrite rows
   * past a restore point, so a snapshot cannot rely on them in place. */
  at::Tensor latent;
  at::Tensor positional;
};

class Eagle3Model {
 public:
  Eagle3Model(Eagle3Shape shape, Eagle3Weights weights, bool exact_k3);

  Eagle3Model(const Eagle3Model&) = delete;
  Eagle3Model& operator=(const Eagle3Model&) = delete;
  Eagle3Model(Eagle3Model&&) = delete;
  Eagle3Model& operator=(Eagle3Model&&) = delete;

  [[nodiscard]] const Eagle3Shape& shape() const noexcept;
  /* Rows in the drafter's KV cache: committed positions whose successor
   * token is known. */
  [[nodiscard]] std::int64_t length() const noexcept;
  /* Committed positions the drafter has seen, including the pending row. */
  [[nodiscard]] std::int64_t token_count() const noexcept;

  /*
   * Advance by committed K3 rows. `taps` is BF16 [n, 3H] (the capture-layer
   * states of the next n positions) and `input_embeddings` BF16 [n, H], the
   * embeddings of the n tokens those rows consumed. Each row pairs with the
   * token that followed it: the previously pending row with input 0, row i
   * with input i + 1. The last row becomes pending until a proposal supplies
   * its successor. The caller owns the embedding table (the exact BF16 K3
   * rows, which this drafter shares), so no copy of it lives here.
   */
  void advance(const at::Tensor& taps, const at::Tensor& input_embeddings);

  /*
   * Greedy proposal chain, one token per call so the caller can embed each
   * new draft: begin() pairs the pending row with the embedding of the token
   * K3 just produced and returns draft 1; step() takes the embedding of the
   * previous draft and returns the next; end() discards the speculative
   * rows. The committed state never changes.
   */
  [[nodiscard]] std::int64_t propose_begin(const at::Tensor& pending_embedding);
  [[nodiscard]] std::int64_t propose_step(const at::Tensor& draft_embedding);
  void propose_end() noexcept;
  [[nodiscard]] bool proposing() const noexcept;

  [[nodiscard]] Eagle3Snapshot snapshot() const;
  void restore(const Eagle3Snapshot& snapshot);
  void clear();

 private:
  struct StepOutput {
    at::Tensor hidden;  // post-norm hidden, fed back on the next chain step
    at::Tensor tokens;  // greedy proposal per row (int64)
  };

  [[nodiscard]] at::Tensor fuse(const at::Tensor& taps) const;
  /* One decoder pass over `rows` new positions starting at length(); the
   * rows attend causally to the cache and to each other, and their KV rows
   * are written at [length(), length() + rows). */
  [[nodiscard]] StepOutput forward(const at::Tensor& embeddings,
                                   const at::Tensor& hidden, bool score);
  void require_embeddings(const at::Tensor& embeddings, std::int64_t rows,
                          const char* label) const;

  Eagle3Shape shape_;
  Eagle3Weights weights_;
  bool exact_k3_;
  at::Tensor inverse_frequencies_;
  at::Tensor latent_;      // [capacity, kv_lora_rank]
  at::Tensor positional_;  // [capacity, qk_rope_head_dim]
  std::int64_t length_ = 0;
  std::uint64_t epoch_ = 0;
  at::Tensor pending_taps_;
  /* Active proposal chain: committed length to return to, and the hidden
   * state the next step continues from. */
  std::int64_t chain_base_ = -1;
  at::Tensor chain_hidden_;
};

}  // namespace deltafin::provider_internal

#endif
