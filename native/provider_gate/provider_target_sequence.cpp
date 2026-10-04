#include "provider_target_sequence.h"

#include "provider_eagle3.h"

#include "provider_cuda_moe.h"
#include "provider_kda_batch.h"

#include <ATen/ops/argmax.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kExactHidden = 7168;
constexpr std::int64_t kSyntheticHidden = 32;

static_assert(std::is_nothrow_move_assignable_v<KdaState>);
static_assert(std::is_nothrow_move_assignable_v<at::Tensor>);

bool same_geometry(const MoeGeometry& left, const MoeGeometry& right) {
  return left.hidden == right.hidden &&
         left.routed_hidden == right.routed_hidden &&
         left.intermediate == right.intermediate &&
         left.experts == right.experts &&
         left.shared_intermediate == right.shared_intermediate;
}

std::uint64_t checked_add(const std::uint64_t left,
                          const std::uint64_t right, const char* name) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(std::string(name) + " byte count overflowed");
  }
  return left + right;
}

std::uint64_t tensor_bytes(const at::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() < 0) {
    throw std::logic_error("target sequence staged an invalid KDA tensor");
  }
  const auto elements = static_cast<std::uint64_t>(tensor.numel());
  const auto element_size = static_cast<std::uint64_t>(tensor.element_size());
  if (element_size != 0 &&
      elements > std::numeric_limits<std::uint64_t>::max() / element_size) {
    throw std::overflow_error("target sequence KDA snapshot size overflowed");
  }
  return elements * element_size;
}

std::uint64_t kda_state_bytes(const KdaState& state) {
  std::uint64_t bytes = 0;
  for (const at::Tensor* tensor :
       {&state.query_convolution, &state.key_convolution,
        &state.value_convolution, &state.recurrent}) {
    bytes = checked_add(bytes, tensor_bytes(*tensor), "KDA snapshot");
  }
  return bytes;
}

struct SequenceRow {
  at::Tensor hidden;
  TargetBlockResidual residual;
};

struct SequenceCacheStage {
  enum class Kind { Kda, Mla };

  Kind kind = Kind::Kda;
  std::uint32_t layer_index = 0;
  TargetKdaCache* kda_cache = nullptr;
  std::uint64_t expected_kda_version = 0;
  /* The state after every row. An ordinary multi-row verify keeps it in the
   * boundary form (convolution windows as views of the sources). */
  KdaState final_kda_state;
  /* Ordinary multi-row verify only: the recurrence inputs that rebuild the
   * state after any accepted prefix, instead of one state per row. */
  KdaRecurrenceRecord kda_record;
  /* Set by stage_kda_prefix_states for an ordinary verify commit. */
  KdaState prefix_kda_state;
  std::unique_ptr<MlaCacheTransaction> mla;
};

struct PendingExpertRow {
  TargetMlpInput mlp_input;
  PreparedMoeT1 moe;
};

struct PendingSequenceLayer {
  std::uint32_t layer_index = 0;
  MoeSpineT1 spine;
  std::unique_ptr<SequenceCacheStage> cache_stage;
  std::array<std::optional<PendingExpertRow>,
             kTargetSequenceMaxPositions> rows{};
  TargetMlpRowsInput mlp_rows;
  /* T>1 preparation owns one contiguous routed-input device matrix. Metal
   * materializes it to CPU once on the first expert tile, and all row views
   * borrow that stable layer-owned carrier until the transaction completes. */
  at::Tensor routed_inputs_device;
  at::Tensor metal_routed_inputs_cpu;
  std::array<at::Tensor, kTargetSequenceMaxPositions> routed_output_rows{};
  std::size_t next_expert_row = 0;
  bool expert_backend_decided = false;
  bool whole_layer_metal_staging = false;
  /* Expert early drain, single-position layers only. Nonzero while this
   * layer's experts are being computed as their bytes arrive; consumed or
   * abandoned by the layer's one finish call, never carried across layers. */
  std::uint64_t expert_stage_token = 0;
};

/*
 * Releases an early-drain staging token exactly once, on every path out of the
 * finish call. Abandoning an already-consumed token is defined and free, so
 * this needs no bookkeeping about which way the call went.
 */
struct StagedExpertToken {
  std::uint64_t token = 0;

  explicit StagedExpertToken(const std::uint64_t value) : token(value) {}
  StagedExpertToken(const StagedExpertToken&) = delete;
  StagedExpertToken& operator=(const StagedExpertToken&) = delete;
  StagedExpertToken(StagedExpertToken&&) = delete;
  StagedExpertToken& operator=(StagedExpertToken&&) = delete;
  ~StagedExpertToken() { abandon_staged_routed_moe_t1(token); }
};

std::optional<at::Tensor> recover_contiguous_row_carrier(
    const std::array<at::Tensor, kTargetSequenceMaxPositions>& rows,
    const std::size_t row_count, const std::int64_t width) {
  if (row_count == 0 || width <= 0 || !rows.front().defined()) {
    return std::nullopt;
  }
  const at::Tensor& first = rows.front();
  if (first.dim() != 2 || first.sizes() != at::IntArrayRef({1, width}) ||
      !first.is_contiguous()) {
    return std::nullopt;
  }
  for (std::size_t row = 1; row < row_count; ++row) {
    const at::Tensor& candidate = rows[row];
    if (!candidate.defined() || candidate.scalar_type() != first.scalar_type() ||
        candidate.device() != first.device() || !candidate.is_contiguous() ||
        candidate.sizes() != at::IntArrayRef({1, width}) ||
        !candidate.is_alias_of(first) ||
        candidate.storage_offset() !=
            first.storage_offset() + static_cast<std::int64_t>(row) * width) {
      return std::nullopt;
    }
  }
  at::Tensor carrier = first.as_strided(
      {static_cast<std::int64_t>(row_count), width}, {width, 1},
      first.storage_offset());
  if (!carrier.is_contiguous()) {
    throw std::logic_error(
        "target sequence recovered a noncontiguous routed-output carrier");
  }
  return carrier;
}

}  // namespace

class TargetSequenceTape::Impl {
 public:
  Impl(const TargetPositionBindings& bindings, at::Tensor input_hidden_rows,
       const TargetSequenceMode mode, const bool capture_dspark_rows,
       const bool full_commit_only, const bool capture_eagle3_rows)
      : mode_(mode),
        exact_k3_(bindings.contract == TargetTapeContract::ExactK3),
        capture_dspark_rows_(capture_dspark_rows || capture_eagle3_rows),
        capture_layers_(capture_eagle3_rows
                            ? std::span<const std::uint32_t>(
                                  kEagle3TargetCaptureLayers)
                            : std::span<const std::uint32_t>(
                                  kDSparkTargetCaptureLayers)),
        full_commit_only_(full_commit_only),
        tail_(bindings.tail),
        pilot_routers_(bindings.pilot_routers) {
    validate_and_copy_bindings(bindings);
    if (mode_ != TargetSequenceMode::Prefill &&
        mode_ != TargetSequenceMode::Verify) {
      throw std::invalid_argument("target sequence mode is unknown");
    }
    if (full_commit_only_ && mode_ != TargetSequenceMode::Verify) {
      throw std::invalid_argument(
          "target sequence full-commit-only requires verify mode");
    }
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    if (!input_hidden_rows.defined() ||
        input_hidden_rows.scalar_type() != at::kFloat ||
        !input_hidden_rows.is_contiguous() || input_hidden_rows.dim() != 2 ||
        input_hidden_rows.size(0) < 1 ||
        input_hidden_rows.size(0) >
            static_cast<std::int64_t>(kTargetSequenceMaxPositions) ||
        input_hidden_rows.size(1) != hidden ||
        input_hidden_rows.device().is_meta()) {
      throw std::invalid_argument(
          "target sequence input must be contiguous fp32 [1..64,hidden]");
    }

    const c10::InferenceMode inference_guard;
    position_count_ = static_cast<std::size_t>(input_hidden_rows.size(0));
    if (position_count_ == 1) {
      rows_.push_back(SequenceRow{
          input_hidden_rows.narrow(0, 0, 1),
          empty_target_block_residual(input_hidden_rows.device(), hidden),
      });
    } else {
      rows_.resize(position_count_);
      at::Tensor anchor_rows = at::empty(
          {static_cast<std::int64_t>(position_count_), 0, hidden},
          at::TensorOptions().dtype(at::kFloat).device(
              input_hidden_rows.device()));
      install_wide_carriers(std::move(input_hidden_rows),
                            std::move(anchor_rows));
    }
    stages_.reserve(kTargetLayerCount);
    decisions_.reserve(position_count_);
    stats_.positions = position_count_;
  }

  ~Impl() { abort_unlocked(); }

  TargetSequenceLayerPrepareKind
  prepare_layer(const TargetLayerBinding& binding) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::Active,
                  "prepare the next target sequence layer");
    try {
      validate_current_layer(binding);
      pending_pilot_.reset();
      auto stage = std::make_unique<SequenceCacheStage>();
      stage->layer_index = next_layer_;
      const TargetLayerCacheBinding& cache = caches_[next_layer_];
      std::unique_ptr<PendingSequenceLayer> routed;
      if (next_layer_ != 0) {
        routed = std::make_unique<PendingSequenceLayer>();
        routed->layer_index = next_layer_;
        routed->spine = *binding.moe;
      }

      if (binding.attention_kind == TargetAttentionKind::Kda) {
        prepare_kda_rows(binding, cache, *stage, routed.get());
      } else {
        prepare_mla_rows(binding, cache, *stage, routed.get());
      }

      ++stats_.streamed_layer_passes;
      stats_.attention_rows += position_count_;
      stats_.maximum_live_streamed_layers = 1;
      if (next_layer_ == 0) {
        capture_completed_layer(next_layer_);
        stages_.push_back(std::move(stage));
        ++next_layer_;
        state_ = next_layer_ == kTargetLayerCount
                     ? TargetSequenceState::ReadyForTail
                     : TargetSequenceState::Active;
        return TargetSequenceLayerPrepareKind::DenseCompleted;
      }

      routed->cache_stage = std::move(stage);
      pending_ = std::move(routed);
      state_ = TargetSequenceState::WaitingForExperts;
      stats_.expert_row_requests += position_count_;
      return TargetSequenceLayerPrepareKind::ExpertRowsRequired;
    } catch (...) {
      abort_unlocked();
      throw;
    }
  }

  TargetSequenceExpertMailbox expert_mailbox() const {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::WaitingForExperts,
                  "read the target sequence expert mailbox");
    return mailbox_;
  }

  TargetSequencePrefetchHint take_prefetch_hint() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    TargetSequencePrefetchHint hint;
    if (state_ != TargetSequenceState::WaitingForExperts ||
        !pending_pilot_.has_value()) {
      return hint;
    }
    try {
      PilotPredictionRows prediction = std::move(*pending_pilot_);
      pending_pilot_.reset();
      if (pilot_routers_ == nullptr || next_layer_ + 1 >= kTargetLayerCount ||
          !(*pilot_routers_)[next_layer_ + 1].has_value()) {
        return hint;
      }
      const std::uint32_t expected_experts =
          (*pilot_routers_)[next_layer_ + 1]->expert_count;
      if (prediction.layer_index != next_layer_ + 1 ||
          prediction.expert_count != expected_experts ||
          prediction.expert_count < kPilotTopK ||
          prediction.position_count != position_count_ ||
          prediction.expert_ids.sizes() != at::IntArrayRef(
              {static_cast<std::int64_t>(position_count_), kPilotTopK}) ||
          prediction.choice_scores.sizes() !=
              at::IntArrayRef(
                  {static_cast<std::int64_t>(position_count_), kPilotTopK})) {
        return hint;
      }
      const at::Tensor ids =
          prediction.expert_ids.to(at::kCPU).contiguous();
      const std::size_t candidate_slots = position_count_ * kPilotTopK;
      const auto id_span = std::span<const std::int64_t>(
          ids.const_data_ptr<std::int64_t>(), candidate_slots);
      std::vector<bool> unique_seen(prediction.expert_count, false);
      std::size_t unique_count = 0;
      for (const std::int64_t expert : id_span) {
        if (expert < 0 ||
            expert >= static_cast<std::int64_t>(prediction.expert_count)) {
          return hint;
        }
        auto seen = unique_seen[static_cast<std::size_t>(expert)];
        if (!seen) {
          unique_seen[static_cast<std::size_t>(expert)] = true;
          ++unique_count;
        }
      }
      at::Tensor scores;
      std::span<const float> score_span;
      if (unique_count > kPilotMaxPrefetch) {
        scores = prediction.choice_scores.to(at::kCPU, at::kFloat).contiguous();
        score_span = std::span<const float>(scores.const_data_ptr<float>(),
                                            candidate_slots);
        ++stats_.pilot_score_materializations;
      } else {
        ++stats_.pilot_score_elisions;
      }
      const auto canonical = canonicalize_pilot_prefetch_rows(
          id_span, score_span, position_count_, prediction.expert_count,
          kPilotMaxPrefetch);
      hint.source_layer = next_layer_;
      hint.target_layer = prediction.layer_index;
      hint.expert_count = static_cast<std::uint16_t>(canonical.count);
      std::copy_n(canonical.expert_ids.begin(), canonical.count,
                  hint.expert_ids.begin());
      ++stats_.pilot_hint_issues;
      stats_.pilot_hint_experts += canonical.count;
      stats_.pilot_max_union_candidates = std::max<std::uint64_t>(
          stats_.pilot_max_union_candidates, canonical.candidate_count);
      return hint;
    } catch (...) {
      pending_pilot_.reset();
      return TargetSequencePrefetchHint{};
    }
  }

  void finish_expert_row(const std::uint16_t row_index,
                         const std::uint64_t spine_generation,
                         const CanonicalExpertBatchT1& experts,
                         const MoeRunOptions& options) {
    finish_expert_tile(
        row_index, 1, spine_generation,
        CanonicalExpertPositionTileT1{experts.expert_ids,
                                      experts.expert_major_bytes,
                                      experts.layout,
                                      experts.expert_span_bytes},
        options);
  }

  /*
   * Start the arrived experts' independent work now. Everything here is
   * best-effort: a false return leaves the layer exactly as it was, and even
   * an accepted batch is only a head start -- finish_expert_tile owns the
   * authoritative route, the weights, and the single ordered reduction.
   */
  bool stage_expert_spans(const std::uint16_t first_row,
                          const std::uint16_t row_count,
                          const std::uint64_t spine_generation,
                          const CanonicalExpertPositionTileT1& experts,
                          const MoeRunOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != TargetSequenceState::WaitingForExperts || pending_ == nullptr ||
        position_count_ != 1 || first_row != 0 || row_count != 1 ||
        pending_->next_expert_row != 0 ||
        spine_generation != pending_->spine.generation ||
        spine_generation != mailbox_.spine_generation ||
        experts.expert_span_pointers.empty() ||
        experts.expert_ids.size() != experts.expert_span_pointers.size() ||
        !experts.expert_major_bytes.empty() ||
        !pending_->rows[0].has_value() ||
        pending_->rows[0]->moe.spine_generation != spine_generation) {
      ++stats_.expert_stage_declines;
      return false;
    }
    try {
      PreparedMoeT1& prepared = pending_->rows[0]->moe;
      if (pending_->expert_stage_token == 0) {
        // The staged kernels read the routed activation row from host memory,
        // so materialize it once here and let the finish call reuse the exact
        // same bytes instead of crossing the boundary a second time.
        if (!prepared.routed_input_cpu.defined()) {
          prepared.routed_input_cpu =
              prepared.routed_input.to(at::kCPU, at::kFloat).contiguous();
          ++stats_.moe_routed_input_host_transfers;
        }
        const std::uint64_t token = begin_staged_routed_moe_t1(
            prepared, prepared.routed_input_cpu, options);
        if (token == 0) {
          ++stats_.expert_stage_declines;
          return false;
        }
        pending_->expert_stage_token = token;
        ++stats_.expert_stage_layers;
      }
      if (!stage_routed_moe_edges_t1(
              pending_->expert_stage_token, prepared, experts.expert_ids,
              experts.expert_span_pointers, experts.layout,
              experts.expert_span_bytes, options)) {
        ++stats_.expert_stage_declines;
        return false;
      }
      stats_.expert_stage_experts += experts.expert_ids.size();
      return true;
    } catch (...) {
      // Staging must never be able to fail a generation. Drop the token and
      // let the ordinary finish path compute the whole layer.
      if (pending_ != nullptr && pending_->expert_stage_token != 0) {
        abandon_staged_routed_moe_t1(pending_->expert_stage_token);
        pending_->expert_stage_token = 0;
      }
      ++stats_.expert_stage_declines;
      return false;
    }
  }

  void finish_expert_tile(
      const std::uint16_t first_row, const std::uint16_t row_count,
      const std::uint64_t spine_generation,
      const CanonicalExpertPositionTileT1& experts,
      const MoeRunOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::WaitingForExperts,
                  "finish a target sequence expert tile");
    try {
      if (pending_ == nullptr ||
          pending_->next_expert_row >= position_count_ ||
          first_row != pending_->next_expert_row || row_count == 0 ||
          row_count > kMoePositionTileMaxRows ||
          static_cast<std::size_t>(first_row) + row_count > position_count_) {
        throw std::invalid_argument(
            "target sequence expert tiles must be bounded and finish in canonical row order");
      }
      if (spine_generation != pending_->spine.generation ||
          spine_generation != mailbox_.spine_generation) {
        throw std::invalid_argument(
            "target sequence expert row has a stale spine generation");
      }
      std::array<const PreparedMoeT1*, kMoePositionTileMaxRows> prepared{};
      for (std::size_t offset = 0; offset < row_count; ++offset) {
        const std::size_t row_index = first_row + offset;
        if (mailbox_.rows[row_index].row_index != row_index ||
            !pending_->rows[row_index].has_value() ||
            pending_->rows[row_index]->moe.spine_generation !=
                spine_generation) {
          throw std::logic_error(
              "target sequence expert tile lost one provider-owned row");
        }
        prepared[offset] = &pending_->rows[row_index]->moe;
      }

      MoeRunOptions execution_options = options;
      if (position_count_ > 1) {
        const at::Device& routed_device = prepared.front()->routed_input.device();
        const bool selects_metal = routed_device.is_mps() &&
            moe_positions_select_metal(routed_device, options);
        if (!pending_->expert_backend_decided) {
          pending_->expert_backend_decided = true;
          pending_->whole_layer_metal_staging = selects_metal;
          if (selects_metal) {
            const auto routed_hidden = static_cast<std::int64_t>(
                pending_->spine.geometry.routed_hidden);
            const auto positions =
                static_cast<std::int64_t>(position_count_);
            if (!pending_->routed_inputs_device.defined() ||
                pending_->routed_inputs_device.scalar_type() != at::kFloat ||
                pending_->routed_inputs_device.device() != routed_device ||
                !pending_->routed_inputs_device.is_contiguous() ||
                pending_->routed_inputs_device.sizes() !=
                    at::IntArrayRef({positions, routed_hidden})) {
              throw std::logic_error(
                  "target sequence lost its whole-layer routed-input carrier");
            }
            if (options.execution_trace != nullptr) {
              options.execution_trace->record(
                  MoeExecutionStage::RoutedInputHostMaterialization);
            }
            pending_->metal_routed_inputs_cpu =
                pending_->routed_inputs_device.to(at::kCPU, at::kFloat)
                    .contiguous();
            for (std::size_t row_index = 0; row_index < position_count_;
                 ++row_index) {
              if (!pending_->rows[row_index].has_value()) {
                throw std::logic_error(
                    "target sequence lost a row during Metal host staging");
              }
              pending_->rows[row_index]->moe.routed_input_cpu =
                  pending_->metal_routed_inputs_cpu.narrow(
                      0, static_cast<std::int64_t>(row_index), 1);
            }
            ++stats_.moe_routed_input_host_transfers;
          }
        } else if (pending_->whole_layer_metal_staging && !selects_metal) {
          throw std::invalid_argument(
              "target sequence cannot change away from Metal after whole-layer host staging");
        }
        execution_options.metal_retain_position_outputs_cpu =
            pending_->whole_layer_metal_staging && selects_metal;
      }

      // Hand any early-drain staging to this one call, and take it off the
      // layer first: whether the call consumes it, ignores it, or throws, the
      // token is released exactly once below.
      execution_options.metal_staged_token = pending_->expert_stage_token;
      const std::uint64_t staged_token = pending_->expert_stage_token;
      pending_->expert_stage_token = 0;
      const StagedExpertToken staged_guard{staged_token};
      const at::Tensor routed_outputs = execute_routed_moe_positions_t1(
          std::span<const PreparedMoeT1* const>(prepared.data(), row_count),
          experts, execution_options);
      if (!routed_outputs.defined() ||
          routed_outputs.scalar_type() != at::kFloat ||
          !routed_outputs.is_contiguous() ||
          routed_outputs.sizes() != at::IntArrayRef(
              {static_cast<std::int64_t>(row_count),
               static_cast<std::int64_t>(
                   pending_->spine.geometry.routed_hidden)})) {
        throw std::runtime_error(
            "target sequence expert tile returned an invalid output matrix");
      }
      for (std::size_t offset = 0; offset < row_count; ++offset) {
        const std::size_t row_index = first_row + offset;
        std::optional<PendingExpertRow>& pending_row =
            pending_->rows[row_index];
        if (position_count_ == 1) {
          const at::Tensor mlp_output = complete_moe_t1(
              pending_row->moe,
              routed_outputs.narrow(
                  0, static_cast<std::int64_t>(offset), 1),
              pending_->spine);
          SequenceRow& row = rows_[row_index];
          row.hidden = complete_target_layer(
              pending_row->mlp_input, mlp_output, exact_k3_);
          row.residual.anchors =
              std::move(pending_row->mlp_input.next_anchors);
          pending_row.reset();
        } else {
          pending_->routed_output_rows[row_index] =
              routed_outputs.narrow(
                  0, static_cast<std::int64_t>(offset), 1);
        }
        mailbox_.rows[row_index].routed_input = at::Tensor();
      }
      if (position_count_ == 1) {
        ++stats_.moe_shared_dispatches;
        ++stats_.moe_complete_provider_dispatches;
        ++stats_.moe_complete_rows;
        ++stats_.moe_routed_up_dispatches;
      }
      pending_->next_expert_row += row_count;
      stats_.expert_rows_completed += row_count;
      ++stats_.expert_tiles_completed;
      stats_.maximum_experts_per_request =
          std::max<std::uint64_t>(stats_.maximum_experts_per_request,
                                  experts.expert_ids.size());
      stats_.maximum_positions_per_expert_tile =
          std::max<std::uint64_t>(stats_.maximum_positions_per_expert_tile,
                                  row_count);

      if (pending_->next_expert_row == position_count_) {
        if (position_count_ > 1) {
          std::vector<const PreparedMoeT1*> all_prepared;
          std::vector<at::Tensor> routed_rows;
          all_prepared.reserve(position_count_);
          routed_rows.reserve(position_count_);
          for (std::size_t row_index = 0; row_index < position_count_;
               ++row_index) {
            if (!pending_->rows[row_index].has_value() ||
                !pending_->routed_output_rows[row_index].defined()) {
              throw std::logic_error(
                  "target sequence lost a completed batched routed row");
            }
            all_prepared.push_back(&pending_->rows[row_index]->moe);
            routed_rows.push_back(pending_->routed_output_rows[row_index]);
          }
          const std::span<const PreparedMoeT1* const> all_prepared_span(
              all_prepared.data(), all_prepared.size());
          std::optional<at::Tensor> existing_carrier =
              recover_contiguous_row_carrier(
                  pending_->routed_output_rows, position_count_,
                  static_cast<std::int64_t>(
                      pending_->spine.geometry.routed_hidden));
          at::Tensor routed_outputs_full = existing_carrier.has_value()
              ? std::move(*existing_carrier)
              : at::cat(routed_rows, 0).contiguous();
          if (pending_->whole_layer_metal_staging) {
            const at::Device completion_device =
                all_prepared.front()->identity.device();
            if (!routed_outputs_full.device().is_cpu()) {
              throw std::logic_error(
                  "whole-layer Metal outputs crossed to device before completion");
            }
            // This is the layer's only expert-output CPU->MPS boundary. All
            // <=16-row I/O tiles above retained their routed result on CPU.
            routed_outputs_full =
                routed_outputs_full.to(completion_device, at::kFloat)
                    .contiguous();
          }
          const at::Tensor mlp_outputs = complete_moe_positions_t1(
              all_prepared_span, routed_outputs_full, pending_->spine);
          at::Tensor completed = complete_target_layer_rows(
              pending_->mlp_rows, mlp_outputs, exact_k3_);
          install_wide_carriers(
              std::move(completed),
              std::move(pending_->mlp_rows.next_anchors));
          for (std::size_t row_index = 0; row_index < position_count_;
               ++row_index) {
            pending_->rows[row_index].reset();
          }
          ++stats_.moe_shared_dispatches;
          ++stats_.moe_complete_provider_dispatches;
          stats_.moe_complete_rows += position_count_;
          ++stats_.moe_routed_up_dispatches;
        }
        capture_completed_layer(next_layer_);
        stages_.push_back(std::move(pending_->cache_stage));
        pending_.reset();
        mailbox_ = TargetSequenceExpertMailbox{};
        ++next_layer_;
        state_ = next_layer_ == kTargetLayerCount
                     ? TargetSequenceState::ReadyForTail
                     : TargetSequenceState::Active;
      }
    } catch (...) {
      if (options.cuda_cache != nullptr &&
          (options.expert_backend == MoeExpertBackend::CudaMxfp4 ||
           options.cuda_plan != 0)) {
        options.cuda_cache->poison_external(
            "target-sequence MoE transaction failed after CUDA selection");
      }
      abort_unlocked();
      throw;
    }
  }

  std::span<const std::uint32_t> finish_tail() {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::ReadyForTail,
                  "finish the target sequence tail");
    try {
      if (next_layer_ != kTargetLayerCount || pending_ != nullptr ||
          stages_.size() != kTargetLayerCount) {
        throw std::logic_error(
            "target sequence is incomplete at its provider tail");
      }
      decisions_.clear();
      for (const SequenceRow& row : rows_) {
        if (row.residual.anchors.size(1) != 8) {
          throw std::logic_error(
              "target sequence row has an incomplete residual tape");
        }
      }
      if (mode_ == TargetSequenceMode::Prefill) {
        SequenceRow& final_row = rows_.back();
        const at::Tensor logits = finish_target_tail(
            final_row.hidden, final_row.residual, *tail_, exact_k3_);
        if (!logits.defined() || logits.dim() != 2 ||
            logits.size(0) != 1 || logits.size(1) <= 0) {
          throw std::runtime_error(
              "target sequence tail did not produce one vocabulary row");
        }
        const std::int64_t token =
            at::argmax(logits, -1, false).item<std::int64_t>();
        if (token < 0 ||
            token > static_cast<std::int64_t>(
                        std::numeric_limits<std::uint32_t>::max())) {
          throw std::runtime_error(
              "target sequence decision is outside the public token range");
        }
        decisions_.push_back(static_cast<std::uint32_t>(token));
        stats_.tail_rows = 1;
      } else {
        at::Tensor hidden_rows;
        at::Tensor anchor_rows;
        if (position_count_ == 1) {
          hidden_rows = rows_.front().hidden;
          anchor_rows = rows_.front().residual.anchors;
        } else {
          require_wide_carrier_aliases();
          hidden_rows = hidden_rows_carrier_;
          anchor_rows = anchor_rows_carrier_;
        }
        const at::Tensor logits = finish_target_tail_rows(
            hidden_rows, anchor_rows, *tail_, exact_k3_);
        if (!logits.defined() || logits.dim() != 2 ||
            logits.size(0) != static_cast<std::int64_t>(position_count_) ||
            logits.size(1) <= 0) {
          throw std::runtime_error(
              "target sequence wide tail did not produce one row per position");
        }
        const at::Tensor tokens =
            at::argmax(logits, -1, false).to(at::kCPU).contiguous();
        const std::int64_t* token_values =
            tokens.const_data_ptr<std::int64_t>();
        for (std::size_t row = 0; row < position_count_; ++row) {
          const std::int64_t token = token_values[row];
          if (token < 0 ||
              token > static_cast<std::int64_t>(
                          std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error(
                "target sequence wide decision is outside the public token range");
          }
          decisions_.push_back(static_cast<std::uint32_t>(token));
        }
        stats_.tail_rows = position_count_;
      }
      for (SequenceRow& row : rows_) {
        row.hidden = at::Tensor();
        row.residual.anchors = at::Tensor();
      }
      hidden_rows_carrier_ = at::Tensor();
      anchor_rows_carrier_ = at::Tensor();
      stats_.tail_provider_dispatches = 1;
      state_ = TargetSequenceState::ReadyToCommit;
      return decisions_;
    } catch (...) {
      abort_unlocked();
      throw;
    }
  }

  at::Tensor dspark_target_rows() const {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::ReadyToCommit,
                  "read DSpark target auxiliary rows");
    if (!capture_dspark_rows_ || dspark_capture_failed_ ||
        captured_dspark_layers_ != capture_layers_.size()) {
      throw std::logic_error(
          "target sequence did not capture the complete DSpark layer roster");
    }
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    if (!dspark_rows_.defined() ||
        dspark_rows_.scalar_type() != at::kBFloat16 ||
        !dspark_rows_.is_contiguous() || dspark_rows_.dim() != 2 ||
        dspark_rows_.size(0) != static_cast<std::int64_t>(position_count_) ||
        dspark_rows_.size(1) !=
            static_cast<std::int64_t>(capture_layers_.size()) *
                hidden) {
      throw std::logic_error(
          "target sequence retained an invalid DSpark target capture");
    }
    return dspark_rows_;
  }

  void commit_prefix(const std::size_t positions) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::ReadyToCommit,
                  "commit a target sequence cache prefix");
    try {
      if (positions > position_count_ ||
          (mode_ == TargetSequenceMode::Prefill &&
           positions != position_count_) ||
          (full_commit_only_ && positions != position_count_)) {
        throw std::invalid_argument(
            "prefill and full-commit-only verify require the full sequence; ordinary verify accepts a bounded prefix");
      }
      preflight_commit(positions);
      stage_kda_prefix_states(positions);

      // No operation below this point can throw: every pointer/version/shape
      // was checked across all 93 caches first, every published KDA state
      // already exists, tensor handle moves are noexcept, and MLA
      // publication has its own no-throw half.
      for (auto& stage : stages_) {
        if (stage->kind == SequenceCacheStage::Kind::Mla) {
          stage->mla->publish_prefix_noexcept(positions);
          continue;
        }
        if (positions == 0) {
          continue;
        }
        KdaState* selected = nullptr;
        if (mode_ == TargetSequenceMode::Prefill || full_commit_only_) {
          selected = &stage->final_kda_state;
        } else {
          selected = &stage->prefix_kda_state;
        }
        stage->kda_cache->state = std::move(*selected);
        stage->kda_cache->version =
            stage->expected_kda_version + positions;
      }
      stages_.clear();
      state_ = TargetSequenceState::Committed;
    } catch (...) {
      abort_unlocked();
      throw;
    }
  }

  void cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == TargetSequenceState::Committed) {
      throw std::logic_error(
          "a committed target sequence cannot be cancelled");
    }
    if (state_ == TargetSequenceState::Cancelled ||
        state_ == TargetSequenceState::Poisoned) {
      return;
    }
    abort_unlocked();
  }

  TargetSequenceState state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

  std::uint32_t next_layer_index() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_layer_;
  }

  TargetSequenceStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  TargetSequenceMode mode() const noexcept { return mode_; }
 std::size_t position_count() const noexcept { return position_count_; }

 private:
  void require_wide_carrier_aliases() const {
    if (position_count_ <= 1) {
      throw std::logic_error(
          "target sequence wide-carrier invariant used for T=1");
    }
    const std::int64_t positions =
        static_cast<std::int64_t>(position_count_);
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    if (!hidden_rows_carrier_.defined() ||
        hidden_rows_carrier_.scalar_type() != at::kFloat ||
        !hidden_rows_carrier_.is_contiguous() ||
        hidden_rows_carrier_.dim() != 2 ||
        hidden_rows_carrier_.sizes() !=
            at::IntArrayRef({positions, hidden}) ||
        !anchor_rows_carrier_.defined() ||
        anchor_rows_carrier_.scalar_type() != at::kFloat ||
        anchor_rows_carrier_.device() != hidden_rows_carrier_.device() ||
        !anchor_rows_carrier_.is_contiguous() ||
        anchor_rows_carrier_.dim() != 3 ||
        anchor_rows_carrier_.size(0) != positions ||
        anchor_rows_carrier_.size(2) != hidden ||
        rows_.size() != position_count_) {
      throw std::logic_error(
          "target sequence lost its authoritative wide carriers");
    }
    const std::int64_t anchors = anchor_rows_carrier_.size(1);
    for (std::size_t row_index = 0; row_index < position_count_;
         ++row_index) {
      const SequenceRow& row = rows_[row_index];
      const std::int64_t index = static_cast<std::int64_t>(row_index);
      if (!row.hidden.defined() || !row.hidden.is_contiguous() ||
          row.hidden.sizes() != at::IntArrayRef({1, hidden}) ||
          !row.hidden.is_alias_of(hidden_rows_carrier_) ||
          row.hidden.storage_offset() !=
              hidden_rows_carrier_.storage_offset() +
                  index * hidden_rows_carrier_.stride(0) ||
          !row.residual.anchors.defined() ||
          !row.residual.anchors.is_contiguous() ||
          row.residual.anchors.sizes() !=
              at::IntArrayRef({1, anchors, hidden}) ||
          !row.residual.anchors.is_alias_of(anchor_rows_carrier_) ||
          row.residual.anchors.storage_offset() !=
              anchor_rows_carrier_.storage_offset() +
                  index * anchor_rows_carrier_.stride(0)) {
        throw std::logic_error(
            "target sequence row view stopped aliasing its wide carrier");
      }
    }
  }

  void install_wide_carriers(at::Tensor hidden_rows,
                             at::Tensor anchor_rows) {
    if (position_count_ <= 1 || !hidden_rows.defined() ||
        !anchor_rows.defined()) {
      throw std::logic_error(
          "target sequence cannot install an invalid wide carrier");
    }
    hidden_rows_carrier_ = std::move(hidden_rows);
    anchor_rows_carrier_ = std::move(anchor_rows);
    for (std::size_t row_index = 0; row_index < position_count_;
         ++row_index) {
      const std::int64_t index = static_cast<std::int64_t>(row_index);
      rows_[row_index].hidden = hidden_rows_carrier_.narrow(0, index, 1);
      rows_[row_index].residual.anchors =
          anchor_rows_carrier_.narrow(0, index, 1);
    }
    require_wide_carrier_aliases();
  }

  // Diagnostic only (DELTAFIN_EAGLE3_TAP_DUMP=<directory>): append every
  // row's completed-layer hidden state at the EAGLE-3 capture layers, as raw
  // fp32, to <directory>/L<layer>.f32 so an offline reference can measure a
  // drafter's acceptance against real K3 states. Never affects execution.
  void dump_eagle3_taps(const std::uint32_t layer_index) noexcept {
    static const char* const directory = std::getenv("DELTAFIN_EAGLE3_TAP_DUMP");
    if (directory == nullptr || directory[0] == '\0' ||
        (layer_index != 1 && layer_index != 45 && layer_index != 89)) {
      return;
    }
    try {
      const std::string path = std::string(directory) + "/L" +
                               std::to_string(layer_index) + ".f32";
      std::FILE* file = std::fopen(path.c_str(), "ab");
      if (file == nullptr) {
        return;
      }
      for (const SequenceRow& row : rows_) {
        if (!row.hidden.defined()) {
          continue;
        }
        const at::Tensor host =
            row.hidden.to(at::kCPU).to(at::kFloat).contiguous();
        std::fwrite(host.data_ptr<float>(), sizeof(float),
                    static_cast<std::size_t>(host.numel()), file);
      }
      std::fclose(file);
      if (layer_index == 1) {
        // One line per target chunk: how many rows it carried. Committed
        // rows are each chunk's leading rows; the event stream says how many.
        std::FILE* chunks =
            std::fopen((std::string(directory) + "/chunks.txt").c_str(), "a");
        if (chunks != nullptr) {
          std::fprintf(chunks, "%zu\n", rows_.size());
          std::fclose(chunks);
        }
      }
    } catch (...) {
    }
  }

  void capture_completed_layer(const std::uint32_t layer_index) noexcept {
    dump_eagle3_taps(layer_index);
    if (!capture_dspark_rows_ || dspark_capture_failed_) {
      return;
    }
    try {
      if (position_count_ > 1) {
        require_wide_carrier_aliases();
      }
      const auto found = std::find(capture_layers_.begin(),
                                   capture_layers_.end(),
                                   layer_index);
      if (found == capture_layers_.end()) {
        return;
      }
      const std::size_t capture_index = static_cast<std::size_t>(
          std::distance(capture_layers_.begin(), found));
      if (capture_index != captured_dspark_layers_) {
        throw std::logic_error(
            "target sequence DSpark capture layers are out of schedule order");
      }
      const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
      if (!dspark_rows_.defined()) {
        dspark_rows_ = at::empty(
            {static_cast<std::int64_t>(position_count_),
             static_cast<std::int64_t>(capture_layers_.size()) *
                 hidden},
            at::TensorOptions().dtype(at::kBFloat16).device(
                rows_.front().hidden.device()));
      }
      for (std::size_t row_index = 0; row_index < rows_.size(); ++row_index) {
        const SequenceRow& row = rows_[row_index];
        if (!row.hidden.defined() || row.hidden.scalar_type() != at::kFloat ||
            !row.hidden.is_contiguous() || row.hidden.dim() != 2 ||
            row.hidden.sizes() != at::IntArrayRef({1, hidden})) {
          throw std::logic_error(
              "target sequence cannot capture an invalid completed hidden row");
        }
        dspark_rows_
            .select(0, static_cast<std::int64_t>(row_index))
            .narrow(0, static_cast<std::int64_t>(capture_index) * hidden,
                    hidden)
            .copy_(row.hidden.squeeze(0), false);
      }
      ++captured_dspark_layers_;
    } catch (...) {
      // Auxiliary drafting must never make full-K3 target execution fail.
      // Discard only the proposal capture; the target transaction and its
      // exact cache stages remain authoritative and continue normally.
      dspark_rows_ = at::Tensor();
      captured_dspark_layers_ = 0;
      dspark_capture_failed_ = true;
    }
  }

  void validate_and_copy_bindings(const TargetPositionBindings& bindings) {
    if (bindings.contract != TargetTapeContract::ExactK3 &&
        bindings.contract != TargetTapeContract::SyntheticK3Schedule) {
      throw std::invalid_argument("target sequence contract is unknown");
    }
    if (bindings.caches.size() != kTargetLayerCount ||
        bindings.tail == nullptr) {
      throw std::invalid_argument(
          "target sequence requires 93 caches and persistent tail weights");
    }
    std::copy(bindings.caches.begin(), bindings.caches.end(), caches_.begin());
    std::unordered_set<const void*> cache_addresses;
    std::size_t kda_count = 0;
    std::size_t mla_count = 0;
    for (std::uint32_t layer_index = 0;
         layer_index < kTargetLayerCount; ++layer_index) {
      const TargetLayerCacheBinding& layer = caches_[layer_index];
      const TargetAttentionKind expected_kind =
          target_layer_uses_mla(layer_index) ? TargetAttentionKind::Mla
                                             : TargetAttentionKind::Kda;
      if (layer.layer_index != layer_index ||
          layer.attention_kind != expected_kind) {
        throw std::invalid_argument(
            "target sequence cache schedule is invalid");
      }
      if (expected_kind == TargetAttentionKind::Kda) {
        ++kda_count;
        if (layer.kda_cache == nullptr || layer.mla_cache != nullptr ||
            layer.kda_cache->layer_index != layer_index ||
            layer.kda_cache->version ==
                std::numeric_limits<std::uint64_t>::max() ||
            !cache_addresses.insert(layer.kda_cache).second) {
          throw std::invalid_argument(
              "target sequence KDA cache ownership is invalid");
        }
        initial_versions_[layer_index] = layer.kda_cache->version;
      } else {
        ++mla_count;
        if (layer.mla_cache == nullptr || layer.kda_cache != nullptr ||
            layer.mla_cache->has_pending_prepare() ||
            layer.mla_cache->version() ==
                std::numeric_limits<std::uint64_t>::max() ||
            !cache_addresses.insert(layer.mla_cache).second) {
          throw std::invalid_argument(
              "target sequence MLA cache ownership is invalid");
        }
        const MlaShape& shape = layer.mla_cache->shape();
        if ((exact_k3_ &&
             (!shape.is_exact_k3() ||
              layer.mla_cache->representation() !=
                  MlaCacheRepresentation::ExpandedExact)) ||
            (!exact_k3_ &&
             (shape.hidden_size != kSyntheticHidden || shape.is_exact_k3()))) {
          throw std::invalid_argument(
              "target sequence MLA cache does not match its model contract");
        }
        initial_versions_[layer_index] = layer.mla_cache->version();
      }
    }
    if (kda_count != kTargetKdaLayerCount ||
        mla_count != kTargetMlaLayerCount) {
      throw std::logic_error("target sequence KDA/MLA schedule counts changed");
    }
  }

  void validate_current_layer(const TargetLayerBinding& layer) const {
    if (next_layer_ >= kTargetLayerCount ||
        layer.layer_index != next_layer_) {
      throw std::invalid_argument(
          "target sequence streamed layer is out of schedule order");
    }
    const TargetAttentionKind expected_kind =
        target_layer_uses_mla(next_layer_) ? TargetAttentionKind::Mla
                                           : TargetAttentionKind::Kda;
    if (layer.attention_kind != expected_kind || layer.residual == nullptr) {
      throw std::invalid_argument(
          "target sequence streamed attention binding is invalid");
    }
    const TargetLayerCacheBinding& cache = caches_[next_layer_];
    if (expected_kind == TargetAttentionKind::Kda) {
      if (layer.kda_weights == nullptr || layer.mla_weights != nullptr ||
          layer.mla_input_bundle != nullptr ||
          cache.kda_cache->version != initial_versions_[next_layer_]) {
        throw std::invalid_argument(
            "target sequence streamed KDA binding/cache is stale or invalid");
      }
    } else {
      if (layer.mla_weights == nullptr || layer.kda_weights != nullptr ||
          cache.mla_cache->has_pending_prepare() ||
          cache.mla_cache->version() != initial_versions_[next_layer_] ||
          !cache.mla_cache->can_append(position_count_)) {
        throw std::invalid_argument(
            "target sequence streamed MLA binding/cache exceeds its exact budget or is stale");
      }
    }

    if (next_layer_ == 0) {
      if (layer.dense == nullptr || layer.moe != nullptr) {
        throw std::invalid_argument(
            "target sequence layer zero must be dense");
      }
      return;
    }
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    const MoeGeometry expected_moe = k3_moe_geometry();
    if (layer.dense != nullptr || layer.moe == nullptr ||
        layer.moe->layer_index != next_layer_ ||
        layer.moe->generation == 0 ||
        layer.moe->geometry.hidden != static_cast<std::uint32_t>(hidden) ||
        (exact_k3_ && !same_geometry(layer.moe->geometry, expected_moe))) {
      throw std::invalid_argument(
          "target sequence routed-MoE binding is invalid");
    }
  }

  void prepare_kda_rows(const TargetLayerBinding& binding,
                        const TargetLayerCacheBinding& cache,
                        SequenceCacheStage& stage,
                        PendingSequenceLayer* routed) {
    stage.kind = SequenceCacheStage::Kind::Kda;
    stage.kda_cache = cache.kda_cache;
    stage.expected_kda_version = cache.kda_cache->version;
    KdaState working = cache.kda_cache->state;
    if (position_count_ == 1) {
      SequenceRow& row = rows_.front();
      TargetAttentionInput attention = prepare_target_attention(
          row.hidden, row.residual, *binding.residual, next_layer_, exact_k3_);
      KdaDecodeResult decoded = kda_decode_one(
          attention.normalized, *binding.kda_weights, working,
          exact_k3_);
      std::vector<TargetMlpInput> mlp_inputs;
      mlp_inputs.push_back(prepare_target_mlp(
          attention, decoded.output, *binding.residual, exact_k3_));
      working = std::move(decoded.next_state);
      const std::uint64_t bytes = kda_state_bytes(working);
      if (mode_ == TargetSequenceMode::Verify && !full_commit_only_) {
        // One row: its only boundary is the final state.
        stats_.verify_snapshot_bytes = checked_add(
            stats_.verify_snapshot_bytes, bytes, "verify snapshot");
      }
      stats_.staged_kda_storage_bytes = checked_add(
          stats_.staged_kda_storage_bytes, bytes, "staged KDA");
      stage.final_kda_state = working;
      prepare_mlp_rows(binding, std::move(mlp_inputs), routed);
    } else {
      require_wide_carrier_aliases();
      TargetAttentionRowsInput attention = prepare_target_attention_rows(
          hidden_rows_carrier_, anchor_rows_carrier_, *binding.residual,
          next_layer_, exact_k3_);
      const at::Tensor& normalized_hidden = attention.normalized;
      KdaBatchInputProjections batch = kda_project_inputs_batch(
          normalized_hidden, *binding.kda_weights, exact_k3_);
      if (batch.positions != position_count_) {
        throw std::logic_error(
            "KDA batch projection returned a different position count");
      }
      stats_.kda_input_provider_dispatches += batch.provider_dispatches;
      stats_.kda_input_equivalent_rowwise_dispatches +=
          batch.equivalent_rowwise_dispatches;
      KdaConvolvedPositions convolved = kda_short_convolve_positions(
          normalized_hidden, *binding.kda_weights, working,
          KdaPreprojectedPositions{
              .query = batch.query,
              .key = batch.key,
              .value = batch.value,
          },
          exact_k3_);
      stats_.kda_shortconv_provider_dispatches += 3;
      KdaBatchDependentProjections dependent =
          kda_project_dependent_batch(
              normalized_hidden, *binding.kda_weights, exact_k3_);
      stats_.kda_dependent_provider_dispatches +=
          dependent.dependent_provider_dispatches;
      stats_.kda_dependent_equivalent_rowwise_dispatches +=
          dependent.dependent_equivalent_rowwise_dispatches;
      const bool ordinary_verify =
          mode_ == TargetSequenceMode::Verify && !full_commit_only_;
      KdaPositionsRecurrentResult recurrence =
          kda_recur_convolved_positions(
              normalized_hidden, *binding.kda_weights, working, convolved,
              KdaDependentPositions{
                  .feature_a = dependent.feature_a,
                  .feature_b = dependent.feature_b,
                  .beta = dependent.beta,
              },
              false, exact_k3_,
              ordinary_verify ? &stage.kda_record : nullptr);
      stats_.kda_recurrent_rows += position_count_;
      if (ordinary_verify) {
        if (stage.kda_record.positions !=
            static_cast<std::int64_t>(position_count_)) {
          throw std::logic_error(
              "KDA position recurrence lost its verify record");
        }
        // Publishing every row keeps the boundary form an ordinary verify
        // has always published: convolution windows as source views.
        stage.final_kda_state =
            kda_boundary_view(stage.kda_record, recurrence.final_state);
        const std::uint64_t record_bytes =
            kda_recurrence_record_bytes(stage.kda_record);
        stats_.verify_snapshot_bytes = checked_add(
            stats_.verify_snapshot_bytes, record_bytes, "verify snapshot");
        stats_.staged_kda_storage_bytes = checked_add(
            checked_add(stats_.staged_kda_storage_bytes, record_bytes,
                        "staged KDA"),
            kda_state_bytes(stage.final_kda_state), "staged KDA");
      }
      working = std::move(recurrence.final_state);
      KdaBatchOutputProjection outputs = kda_finish_output_batch(
          normalized_hidden, recurrence.recurrent_output_rows,
          *binding.kda_weights, exact_k3_);
      if (outputs.positions != position_count_ ||
          !outputs.output.defined() ||
          outputs.output.sizes() != at::IntArrayRef(
              {static_cast<std::int64_t>(position_count_),
               rows_.front().hidden.size(1)})) {
        throw std::logic_error(
            "KDA output batch returned a different position geometry");
      }
      stats_.kda_output_provider_dispatches += outputs.provider_dispatches;
      stats_.kda_output_rows += outputs.positions;
      TargetMlpRowsInput mlp = prepare_target_mlp_rows(
          attention, outputs.output, *binding.residual, exact_k3_);
      if (mode_ == TargetSequenceMode::Prefill || full_commit_only_) {
        stats_.staged_kda_storage_bytes = checked_add(
            stats_.staged_kda_storage_bytes, kda_state_bytes(working),
            "staged KDA");
        stage.final_kda_state = working;
      }
      prepare_mlp_rows_batch(binding, std::move(mlp), routed);
    }
  }

  void prepare_mla_rows(const TargetLayerBinding& binding,
                        const TargetLayerCacheBinding& cache,
                        SequenceCacheStage& stage,
                        PendingSequenceLayer* routed) {
    stage.kind = SequenceCacheStage::Kind::Mla;
    stage.mla = std::make_unique<MlaCacheTransaction>(
        *cache.mla_cache, position_count_);
    if (position_count_ == 1) {
      SequenceRow& row = rows_.front();
      TargetAttentionInput attention = prepare_target_attention(
          row.hidden, row.residual, *binding.residual, next_layer_, exact_k3_);
      MlaPreparedDecode decoded =
          exact_k3_
              ? prepare_k3_mla_decode(
                    attention.normalized.view({1, 1, row.hidden.size(1)}),
                    *binding.mla_weights, stage.mla->working_cache(),
                    binding.mla_input_bundle)
              : prepare_mla_decode(
                    attention.normalized.view({1, 1, row.hidden.size(1)}),
                    *binding.mla_weights, stage.mla->working_cache(), true,
                    binding.mla_input_bundle);
      try {
        std::vector<TargetMlpInput> mlp_inputs;
        mlp_inputs.push_back(prepare_target_mlp(
            attention,
            decoded.output.view({1, row.hidden.size(1)}).contiguous(),
            *binding.residual, exact_k3_));
        commit_mla_decode(stage.mla->working_cache(), decoded);
        ++stats_.mla_position_provider_dispatches;
        ++stats_.mla_position_rows;
        prepare_mlp_rows(binding, std::move(mlp_inputs), routed);
      } catch (...) {
        if (!decoded.finalized && decoded.owner != nullptr) {
          try {
            cancel_mla_decode(stage.mla->working_cache(), decoded);
          } catch (...) {
            // The branch is unpublished and is destroyed by abort_unlocked;
            // never risk publishing a partially prepared cache during cleanup.
          }
        }
        throw;
      }
    } else {
      require_wide_carrier_aliases();
      TargetAttentionRowsInput attention = prepare_target_attention_rows(
          hidden_rows_carrier_, anchor_rows_carrier_, *binding.residual,
          next_layer_, exact_k3_);
      const at::Tensor hidden =
          attention.normalized
              .view({1, static_cast<std::int64_t>(position_count_),
                     rows_.front().hidden.size(1)});
      MlaPreparedDecode decoded =
          exact_k3_
              ? prepare_k3_mla_positions(
                    hidden, *binding.mla_weights,
                    stage.mla->working_cache(), nullptr,
                    binding.mla_input_bundle)
              : prepare_mla_positions(
                    hidden, *binding.mla_weights,
                    stage.mla->working_cache(), nullptr, true,
                    binding.mla_input_bundle);
      try {
        const at::Tensor output_rows = decoded.output.view(
            {static_cast<std::int64_t>(position_count_),
             rows_.front().hidden.size(1)});
        TargetMlpRowsInput mlp = prepare_target_mlp_rows(
            attention, output_rows.contiguous(), *binding.residual,
            exact_k3_);
        commit_mla_decode(stage.mla->working_cache(), decoded);
        ++stats_.mla_position_provider_dispatches;
        stats_.mla_position_rows += position_count_;
        prepare_mlp_rows_batch(binding, std::move(mlp), routed);
      } catch (...) {
        if (!decoded.finalized && decoded.owner != nullptr) {
          try {
            cancel_mla_decode(stage.mla->working_cache(), decoded);
          } catch (...) {
            // The branch is unpublished and is destroyed by abort_unlocked;
            // never risk publishing a partially prepared cache during cleanup.
          }
        }
        throw;
      }
    }
    const std::uint64_t before = cache.mla_cache->storage_bytes();
    const std::uint64_t after = stage.mla->working_cache().storage_bytes();
    if (after < before) {
      throw std::logic_error("target sequence MLA branch shrank its storage");
    }
    stats_.projected_mla_storage_bytes = checked_add(
        stats_.projected_mla_storage_bytes, after,
        "MLA projected storage");
    stats_.additional_mla_storage_bytes = checked_add(
        stats_.additional_mla_storage_bytes, after - before,
        "MLA additional storage");
  }

  void prepare_mlp_rows_batch(const TargetLayerBinding& binding,
                              TargetMlpRowsInput mlp_input,
                              PendingSequenceLayer* routed) {
    if (position_count_ <= 1 || !mlp_input.normalized.defined() ||
        mlp_input.normalized.dim() != 2 ||
        mlp_input.normalized.size(0) !=
            static_cast<std::int64_t>(position_count_)) {
      throw std::logic_error(
          "target sequence batched MLP input changed its position count");
    }
    if (next_layer_ == 0) {
      const at::Tensor outputs = run_target_dense_rows(
          mlp_input.normalized, *binding.dense, exact_k3_);
      if (!outputs.defined() || outputs.scalar_type() != at::kFloat ||
          !outputs.is_contiguous() ||
          outputs.sizes() != mlp_input.normalized.sizes()) {
        throw std::runtime_error(
            "target sequence dense row batch returned an invalid output matrix");
      }
      at::Tensor completed = complete_target_layer_rows(
          mlp_input, outputs, exact_k3_);
      ++stats_.dense_mlp_provider_dispatches;
      stats_.dense_mlp_rows += position_count_;
      install_wide_carriers(std::move(completed),
                            std::move(mlp_input.next_anchors));
      return;
    }
    if (routed == nullptr) {
      throw std::logic_error("target sequence routed layer lost its mailbox");
    }

    // Mirror Python PILOT's prompt/verify schedule: route every live row with
    // the next-layer lookahead router before the current authoritative router,
    // then publish one bounded union only at Rust's expert-I/O boundary.
    prepare_pilot_rows(mlp_input.lookahead_source);

    PreparedMoePositionsT1 prepared = prepare_moe_positions_t1(
        mlp_input.normalized, routed->spine);
    if (prepared.rows.size() != position_count_) {
      throw std::logic_error(
          "target sequence MoE batch changed its position count");
    }

    ++stats_.moe_prepare_provider_dispatches;
    stats_.moe_prepare_rows += position_count_;
    stats_.moe_router_dispatches += prepared.router_dispatches;
    stats_.moe_routed_down_dispatches += prepared.routed_down_dispatches;
    stats_.moe_shared_dispatches += prepared.shared_dispatches;
    stats_.moe_route_materializations += prepared.route_materializations;
    stats_.moe_route_host_transfers += prepared.route_host_transfers;
    stats_.moe_routed_input_host_transfers +=
        prepared.routed_input_host_transfers;

    routed->mlp_rows = std::move(mlp_input);
    routed->routed_inputs_device = std::move(prepared.routed_inputs);
    mailbox_.layer_index = next_layer_;
    mailbox_.spine_generation = routed->spine.generation;
    mailbox_.row_count = static_cast<std::uint16_t>(position_count_);
    for (std::size_t row_index = 0; row_index < position_count_;
         ++row_index) {
      const auto index = static_cast<std::int64_t>(row_index);
      PreparedMoeT1& moe = prepared.rows[row_index];
      mailbox_.rows[row_index] = TargetSequenceRouteRow{
          .row_index = static_cast<std::uint16_t>(row_index),
          .route = moe.route,
          .routed_input = moe.routed_input,
      };
      TargetMlpInput row_input{
          routed->mlp_rows.normalized.narrow(0, index, 1),
          routed->mlp_rows.lookahead_source.narrow(0, index, 1),
          routed->mlp_rows.prefix_sum.narrow(0, index, 1),
          routed->mlp_rows.next_anchors.narrow(0, index, 1),
          routed->mlp_rows.layer_index};
      routed->rows[row_index].emplace(PendingExpertRow{
          std::move(row_input), std::move(moe)});
    }
  }

  void prepare_mlp_rows(const TargetLayerBinding& binding,
                        std::vector<TargetMlpInput> mlp_inputs,
                        PendingSequenceLayer* routed) {
    if (mlp_inputs.size() != position_count_) {
      throw std::logic_error(
          "target sequence MLP preparation changed its position count");
    }
    if (position_count_ != 1) {
      throw std::logic_error(
          "wide target MLP rows bypassed their live batch carrier");
    }
    prepare_row_mlp(binding, 0, std::move(mlp_inputs.front()), routed);
  }

  void prepare_row_mlp(const TargetLayerBinding& binding,
                       const std::size_t row_index,
                       TargetMlpInput mlp_input,
                       PendingSequenceLayer* routed) {
    SequenceRow& row = rows_[row_index];
    if (next_layer_ == 0) {
      const at::Tensor mlp_output = run_target_dense(
          mlp_input.normalized, *binding.dense, exact_k3_);
      row.hidden =
          complete_target_layer(mlp_input, mlp_output, exact_k3_);
      row.residual.anchors = std::move(mlp_input.next_anchors);
      ++stats_.dense_mlp_provider_dispatches;
      ++stats_.dense_mlp_rows;
      return;
    }
    if (routed == nullptr) {
      throw std::logic_error("target sequence routed layer lost its mailbox");
    }
    // Scheduling prediction is enqueued before the authoritative current-layer
    // router. Its tensors remain device-resident until Rust has
    // already obtained and begun serving the real route. Any failure simply
    // removes the optional hint; target math is unchanged.
    if (row_index == 0) {
      prepare_pilot_rows(mlp_input.lookahead_source);
    }
    PreparedMoeT1 moe = prepare_moe_t1(mlp_input.normalized, routed->spine);
    ++stats_.moe_prepare_provider_dispatches;
    ++stats_.moe_prepare_rows;
    ++stats_.moe_router_dispatches;
    ++stats_.moe_routed_down_dispatches;
    if (moe.shared_output.defined()) {
      ++stats_.moe_shared_dispatches;
    }
    ++stats_.moe_route_materializations;
    if (!mlp_input.normalized.device().is_cpu()) {
      ++stats_.moe_route_host_transfers;
    }
    if (mlp_input.normalized.device().is_mps()) {
      ++stats_.moe_routed_input_host_transfers;
    }
    mailbox_.layer_index = next_layer_;
    mailbox_.spine_generation = routed->spine.generation;
    mailbox_.row_count = static_cast<std::uint16_t>(position_count_);
    mailbox_.rows[row_index] = TargetSequenceRouteRow{
        .row_index = static_cast<std::uint16_t>(row_index),
        .route = moe.route,
        .routed_input = moe.routed_input,
    };
    routed->rows[row_index].emplace(
        PendingExpertRow{std::move(mlp_input), std::move(moe)});
  }

  void prepare_pilot_rows(const at::Tensor& lookahead_source) noexcept {
    if (next_layer_ + 1 >= kTargetLayerCount || pilot_routers_ == nullptr) {
      return;
    }
    const auto& next_router = (*pilot_routers_)[next_layer_ + 1];
    if (!next_router.has_value()) {
      return;
    }
    std::optional<PilotPredictionRows> prediction =
        try_predict_pilot_router_rows(lookahead_source, *next_router,
                                      exact_k3_);
    if (!prediction.has_value() ||
        prediction->position_count != position_count_) {
      return;
    }
    pending_pilot_ = std::move(*prediction);
    ++stats_.pilot_prediction_dispatches;
    stats_.pilot_prediction_rows += position_count_;
  }

  /* The boundary-form state after every row of a recorded recurrence. */
  static KdaState kda_boundary_view(const KdaRecurrenceRecord& record,
                                    const KdaState& final_state) {
    const std::int64_t start = record.positions - 1;
    return KdaState{
        record.query_source.narrow(2, start, record.convolution_width),
        record.key_source.narrow(2, start, record.convolution_width),
        record.value_source.narrow(2, start, record.convolution_width),
        final_state.recurrent,
    };
  }

  /*
   * Ordinary verify: materialize the state each KDA cache publishes for a
   * `positions`-row commit before anything is published. A shorter prefix
   * replays the accepted rows from the recorded inputs; each layer drops
   * its full-sequence state as it goes, so the peak stays one generation.
   * A throw here leaves every cache untouched and the sequence aborts.
   */
  void stage_kda_prefix_states(const std::size_t positions) {
    if (mode_ != TargetSequenceMode::Verify || full_commit_only_ ||
        positions == 0) {
      return;
    }
    for (auto& stage : stages_) {
      if (stage->kind != SequenceCacheStage::Kind::Kda) {
        continue;
      }
      if (positions == position_count_) {
        stage->prefix_kda_state = stage->final_kda_state;
      } else {
        stage->prefix_kda_state = kda_replay_recorded_state(
            stage->kda_record, static_cast<std::int64_t>(positions));
        stage->final_kda_state = KdaState{};
      }
      stage->kda_record = KdaRecurrenceRecord{};
    }
  }

  void preflight_commit(const std::size_t positions) const {
    std::size_t kda_count = 0;
    std::size_t mla_count = 0;
    for (const auto& stage : stages_) {
      if (stage == nullptr) {
        throw std::logic_error("target sequence has an empty cache stage");
      }
      if (stage->kind == SequenceCacheStage::Kind::Mla) {
        ++mla_count;
        if (stage->mla == nullptr ||
            stage->mla->completed_positions() != position_count_) {
          throw std::logic_error(
              "target sequence has an incomplete MLA branch");
        }
        stage->mla->preflight_publish_prefix(positions);
        continue;
      }
      ++kda_count;
      if (stage->kda_cache == nullptr ||
          stage->kda_cache->layer_index != stage->layer_index ||
          stage->kda_cache->version != stage->expected_kda_version ||
          positions > std::numeric_limits<std::uint64_t>::max() -
                          stage->expected_kda_version) {
        throw std::invalid_argument(
            "target sequence KDA cache became stale before commit");
      }
      if (mode_ == TargetSequenceMode::Verify && !full_commit_only_ &&
          position_count_ > 1 &&
          stage->kda_record.positions !=
              static_cast<std::int64_t>(position_count_)) {
        throw std::logic_error(
            "target sequence verify lost its KDA prefix record");
      }
      if (positions != 0 && !stage->final_kda_state.recurrent.defined()) {
        throw std::logic_error(
            "target sequence full publication lost its final KDA state");
      }
    }
    if (kda_count != kTargetKdaLayerCount ||
        mla_count != kTargetMlaLayerCount) {
      throw std::logic_error(
          "target sequence did not stage all 93 attention caches");
    }
  }

  void require_state(const TargetSequenceState expected,
                     const char* operation) const {
    if (state_ != expected) {
      throw std::logic_error(std::string("cannot ") + operation +
                             " in the current target-sequence state");
    }
  }

  void abort_unlocked() noexcept {
    if (state_ == TargetSequenceState::Committed ||
        state_ == TargetSequenceState::Cancelled) {
      return;
    }
    // A layer torn down mid-drain still owns staged GPU work reading caller
    // memory. Abandoning drains it before that memory can be retired.
    if (pending_ != nullptr && pending_->expert_stage_token != 0) {
      abandon_staged_routed_moe_t1(pending_->expert_stage_token);
      pending_->expert_stage_token = 0;
    }
    pending_.reset();
    stages_.clear();
    rows_.clear();
    hidden_rows_carrier_ = at::Tensor();
    anchor_rows_carrier_ = at::Tensor();
    mailbox_ = TargetSequenceExpertMailbox{};
    pending_pilot_.reset();
    decisions_.clear();
    dspark_rows_ = at::Tensor();
    captured_dspark_layers_ = 0;
    state_ = TargetSequenceState::Cancelled;
  }

  TargetSequenceMode mode_ = TargetSequenceMode::Prefill;
  bool exact_k3_ = true;
  bool capture_dspark_rows_ = false;
  /* Layers whose completed rows a proposal model consumes: DSpark's five
   * public boundaries, or EAGLE-3's three. */
  std::span<const std::uint32_t> capture_layers_;
  bool full_commit_only_ = false;
  bool dspark_capture_failed_ = false;
  const TargetTailWeights* tail_ = nullptr;
  const TargetPilotRoster* pilot_routers_ = nullptr;
  std::size_t position_count_ = 0;
  std::array<TargetLayerCacheBinding, kTargetLayerCount> caches_{};
  std::array<std::uint64_t, kTargetLayerCount> initial_versions_{};
  std::vector<SequenceRow> rows_;
  /* T>1 row views are compatibility handles only. These two contiguous
   * matrices remain the authoritative live layer state, so attention and the
   * tail never have to reconstruct them with one cat per carrier. */
  at::Tensor hidden_rows_carrier_;
  at::Tensor anchor_rows_carrier_;
  std::vector<std::unique_ptr<SequenceCacheStage>> stages_;
  std::unique_ptr<PendingSequenceLayer> pending_;
  TargetSequenceExpertMailbox mailbox_{};
  std::optional<PilotPredictionRows> pending_pilot_;
  std::vector<std::uint32_t> decisions_;
  at::Tensor dspark_rows_;
  std::size_t captured_dspark_layers_ = 0;
  std::uint32_t next_layer_ = 0;
  TargetSequenceState state_ = TargetSequenceState::Active;
  TargetSequenceStats stats_{};
  mutable std::mutex mutex_;
};

TargetSequenceTape::TargetSequenceTape(
    const TargetPositionBindings& bindings, at::Tensor input_hidden_rows,
    const TargetSequenceMode mode, const bool capture_dspark_rows,
    const bool full_commit_only, const bool capture_eagle3_rows)
    : impl_(std::make_unique<Impl>(bindings, std::move(input_hidden_rows),
                                   mode, capture_dspark_rows,
                                   full_commit_only, capture_eagle3_rows)) {
  if (capture_dspark_rows && capture_eagle3_rows) {
    throw std::invalid_argument(
        "a target sequence captures rows for one proposal model at most");
  }
}

TargetSequenceTape::~TargetSequenceTape() = default;

TargetSequenceLayerPrepareKind
TargetSequenceTape::prepare_layer(const TargetLayerBinding& layer) {
  return impl_->prepare_layer(layer);
}

TargetSequenceExpertMailbox TargetSequenceTape::expert_mailbox() const {
  return impl_->expert_mailbox();
}

TargetSequencePrefetchHint
TargetSequenceTape::take_prefetch_hint() noexcept {
  return impl_->take_prefetch_hint();
}

void TargetSequenceTape::finish_expert_row(
    const std::uint16_t row_index, const std::uint64_t spine_generation,
    const CanonicalExpertBatchT1& experts, const MoeRunOptions& options) {
  impl_->finish_expert_row(row_index, spine_generation, experts, options);
}

bool TargetSequenceTape::stage_expert_spans(
    const std::uint16_t first_row, const std::uint16_t row_count,
    const std::uint64_t spine_generation,
    const CanonicalExpertPositionTileT1& experts,
    const MoeRunOptions& options) {
  return impl_->stage_expert_spans(first_row, row_count, spine_generation,
                                   experts, options);
}

void TargetSequenceTape::finish_expert_tile(
    const std::uint16_t first_row, const std::uint16_t row_count,
    const std::uint64_t spine_generation,
    const CanonicalExpertPositionTileT1& experts,
    const MoeRunOptions& options) {
  impl_->finish_expert_tile(first_row, row_count, spine_generation, experts,
                            options);
}

std::span<const std::uint32_t> TargetSequenceTape::finish_tail() {
  return impl_->finish_tail();
}

at::Tensor TargetSequenceTape::dspark_target_rows() const {
  return impl_->dspark_target_rows();
}

void TargetSequenceTape::commit_all() {
  impl_->commit_prefix(impl_->position_count());
}

void TargetSequenceTape::commit_prefix(const std::size_t positions) {
  impl_->commit_prefix(positions);
}

void TargetSequenceTape::cancel() { impl_->cancel(); }

TargetSequenceState TargetSequenceTape::state() const {
  return impl_->state();
}

TargetSequenceMode TargetSequenceTape::mode() const noexcept {
  return impl_->mode();
}

std::size_t TargetSequenceTape::position_count() const noexcept {
  return impl_->position_count();
}

std::uint32_t TargetSequenceTape::next_layer_index() const {
  return impl_->next_layer_index();
}

TargetSequenceStats TargetSequenceTape::stats() const {
  return impl_->stats();
}

}  // namespace deltafin::provider_internal
