#ifndef DELTAFIN_PROVIDER_KDA_H
#define DELTAFIN_PROVIDER_KDA_H

#include "provider_bf16_cpu.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace deltafin::provider_internal {

/*
 * Internal C++ value types for the K3 KDA attention tape.  They never cross
 * the public C ABI: at::Tensor ownership remains entirely inside the linked
 * provider and Rust sees only opaque integer handles.
 */
struct KdaProjection {
  at::Tensor weight;
  at::Tensor scale;
  OriginalBf16Matrix original_bf16;

  KdaProjection() = default;
  KdaProjection(at::Tensor weight_value, at::Tensor scale_value,
                OriginalBf16Matrix original_value = {})
      : weight(std::move(weight_value)), scale(std::move(scale_value)),
        original_bf16(std::move(original_value)) {}
};

struct KdaWeights {
  at::Tensor a_log;
  at::Tensor dt_bias;
  at::Tensor query_convolution;
  at::Tensor key_convolution;
  at::Tensor value_convolution;
  at::Tensor output_norm;
  KdaProjection query_projection;
  KdaProjection key_projection;
  KdaProjection value_projection;
  KdaProjection recurrent_gate_projection;
  KdaProjection feature_a_projection;
  KdaProjection feature_b_projection;
  KdaProjection beta_projection;
  KdaProjection output_projection;
};

struct KdaState {
  at::Tensor query_convolution;
  at::Tensor key_convolution;
  at::Tensor value_convolution;
  at::Tensor recurrent;
};

struct KdaDecodeResult {
  at::Tensor output;
  KdaState next_state;
};

/*
 * Row-independent input projections prepared by a wider sequence operation.
 * The recurrent/convolution state is intentionally absent. This carrier also
 * remains the exact one-row compatibility entry; the live prompt path uses
 * KdaPreprojectedPositions so output-gate projection stays after recurrence.
 */
struct KdaPreprojectedInputs {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  at::Tensor output_gate;
  at::Tensor feature_a;
  /* Optional T-wide dependent projections supplied by the batch tape. */
  at::Tensor feature_b;
  at::Tensor beta;
};

/*
 * One causally advanced row before KDA's final hidden-width O projection.
 * The recurrent state is already authoritative; output_projection_input is
 * fp32 [1,heads*head_width] and may be projected with adjacent rows once.
 */
struct KdaRecurrentResult {
  at::Tensor output_projection_input;
  KdaState next_state;
};

/* T-wide independent work supplied to the causal recurrence island. */
struct KdaPreprojectedPositions {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
};

struct KdaDependentPositions {
  at::Tensor feature_a;
  at::Tensor feature_b;
  at::Tensor beta;
};

struct KdaConvolvedPositions {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  at::Tensor query_source;
  at::Tensor key_source;
  at::Tensor value_source;
};

struct KdaPositionsRecurrentResult {
  at::Tensor recurrent_output_rows;
  KdaState final_state;
  std::vector<KdaState> boundaries;
};

/*
 * What a T-row recurrence retains so a verify commit can rebuild the state
 * after any accepted prefix: the starting recurrent state, the per-row
 * decay/key/value/beta rows the loop consumed, and the convolution sources.
 * Replaying k rows runs the loop's own per-row update on the same row
 * tensors, so it reproduces boundary k bit for bit while keeping about 1/22
 * of the memory of one recurrent state per row (19 MiB a row on K3 against
 * 453 MiB).
 */
struct KdaRecurrenceRecord {
  at::Tensor initial_recurrent;
  at::Tensor gated_decay;  // [T, heads, width]
  at::Tensor key;          // [T, heads, width], normalized
  at::Tensor value;        // [T, heads, width]
  at::Tensor beta;         // [T, heads], after the sigmoid
  at::Tensor query_source;  // [1, projection, convolution - 1 + T]
  at::Tensor key_source;
  at::Tensor value_source;
  std::int64_t positions = 0;
  std::int64_t convolution_width = 0;
};

/*
 * Execute one decode position.  exact_k3=true rejects every dimension,
 * scalar type, and storage form that is not the released K3 contract.
 */
KdaDecodeResult kda_decode_one(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, bool exact_k3);

KdaDecodeResult kda_decode_one_preprojected(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, const KdaPreprojectedInputs& projected,
    bool exact_k3);

KdaRecurrentResult kda_decode_one_preprojected_deferred_output(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, const KdaPreprojectedInputs& projected,
    bool exact_k3);

/*
 * Prompt/verification path matching the established full-sequence KDA shape:
 * three T-wide depthwise short convolutions and all independent gating/norm
 * work surround only the mathematically causal recurrent-state loop.
 */
KdaConvolvedPositions kda_short_convolve_positions(
    const at::Tensor& hidden_rows, const KdaWeights& weights,
    const KdaState& state, const KdaPreprojectedPositions& projected,
    bool exact_k3);

KdaPositionsRecurrentResult kda_recur_convolved_positions(
    const at::Tensor& hidden_rows, const KdaWeights& weights,
    const KdaState& state, const KdaConvolvedPositions& convolved,
    const KdaDependentPositions& dependent,
    bool retain_boundaries, bool exact_k3,
    KdaRecurrenceRecord* record = nullptr);

/*
 * The state after the first `positions` rows (1..T) of a recorded
 * recurrence, in the same form `retain_boundaries` returns for that row.
 */
KdaState kda_replay_recorded_state(const KdaRecurrenceRecord& record,
                                   std::int64_t positions);

/* Bytes a record keeps alive beyond the final state. */
std::uint64_t kda_recurrence_record_bytes(const KdaRecurrenceRecord& record);

/* Development-gate entry points: same equations, with the dispatch policy
 * forced so an isolated parity/timing executable can compare both arms. */
KdaDecodeResult kda_decode_one_unfused_for_test(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, bool exact_k3);
KdaDecodeResult kda_decode_one_fused_for_test(
    const at::Tensor& hidden, const KdaWeights& weights,
    const KdaState& state, bool exact_k3);

/* Allocate the exact batch-one K3 state on a provider-selected device. */
KdaState zero_k3_kda_state(const at::Device& device);
KdaState zero_small_kda_canary_state(const at::Device& device);

std::uint64_t kda_state_conv_elements(const KdaState& state);
std::uint64_t kda_state_recurrent_elements(const KdaState& state);

/*
 * A deterministic, compact capability/parity gate.  The implementation under
 * test uses the production tape while the reference separately spells out
 * every equation with dense fp32 weights.  No model files are touched.
 */
bool kda_small_parity_canary(const at::Device& device, std::string& detail);

}  // namespace deltafin::provider_internal

#endif
