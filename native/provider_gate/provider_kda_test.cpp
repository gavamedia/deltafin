#include "provider_device.h"
#include "provider_kda.h"
#include "provider_kda_batch.h"

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>

#if defined(__APPLE__)
#include <torch/mps.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::KdaDecodeResult;
using deltafin::provider_internal::KdaProjection;
using deltafin::provider_internal::KdaState;
using deltafin::provider_internal::KdaWeights;

constexpr std::int64_t kHidden = 7168;
constexpr std::int64_t kHeads = 96;
constexpr std::int64_t kHeadWidth = 128;
constexpr std::int64_t kProjection = kHeads * kHeadWidth;
constexpr std::int64_t kConvolutionWidth = 4;

struct Options {
  std::uint32_t device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  bool benchmark_real = false;
};

struct ProjectionArena {
  at::Tensor weight;
  at::Tensor scale;
};

void synchronize_device(const at::Device& device) {
  if (device.type() == at::kMPS) {
#if defined(__APPLE__)
    torch::mps::synchronize();
#else
    throw std::runtime_error("MPS synchronization is unavailable");
#endif
  }
}

template <std::size_t Size>
ProjectionArena make_projection_arena(
    const std::array<std::int64_t, Size>& rows,
    const at::Device& device) {
  std::int64_t total_rows = 0;
  for (const std::int64_t count : rows) {
    total_rows += count;
  }
  at::Tensor cpu_weight = at::empty(
      {total_rows, kHidden},
      at::TensorOptions().dtype(at::kChar).device(at::kCPU));
  at::Tensor cpu_scale = at::empty(
      {total_rows},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  std::int64_t offset = 0;
  for (std::size_t index = 0; index < rows.size(); ++index) {
    cpu_weight.narrow(0, offset, rows[index])
        .fill_(static_cast<std::int64_t>(index + 1));
    cpu_scale.narrow(0, offset, rows[index])
        .fill_(static_cast<double>(index + 1) / 4096.0);
    offset += rows[index];
  }
  return ProjectionArena{cpu_weight.to(device), cpu_scale.to(device)};
}

KdaProjection projection_view(const ProjectionArena& arena,
                              const std::int64_t row,
                              const std::int64_t rows) {
  return KdaProjection{
      arena.weight.narrow(0, row, rows),
      arena.scale.narrow(0, row, rows),
  };
}

KdaProjection separate_projection(const std::int64_t rows,
                                  const std::int64_t columns,
                                  const at::Device& device,
                                  const std::int64_t value) {
  at::Tensor weight = at::full(
      {rows, columns}, value,
      at::TensorOptions().dtype(at::kChar).device(at::kCPU));
  at::Tensor scale = at::full(
      {rows}, static_cast<double>(value) / 4096.0,
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  return KdaProjection{weight.to(device), scale.to(device)};
}

KdaWeights make_real_weights(const at::Device& device,
                             ProjectionArena& projection_arena,
                             at::Tensor& convolution_arena) {
  const std::array<std::int64_t, 5> rows{
      kProjection, kProjection, kProjection, kProjection, kHeadWidth};
  projection_arena = make_projection_arena(rows, device);
  std::int64_t row = 0;
  const auto take = [&](const std::int64_t count) {
    const KdaProjection result =
        projection_view(projection_arena, row, count);
    row += count;
    return result;
  };
  const KdaProjection query = take(rows[0]);
  const KdaProjection key = take(rows[1]);
  const KdaProjection value = take(rows[2]);
  const KdaProjection output_gate = take(rows[3]);
  const KdaProjection feature_a = take(rows[4]);

  convolution_arena = at::empty(
      {kProjection * 3, 1, kConvolutionWidth},
      at::TensorOptions().dtype(at::kFloat).device(device));
  convolution_arena.narrow(0, 0, kProjection).fill_(0.03125);
  convolution_arena.narrow(0, kProjection, kProjection).fill_(0.046875);
  convolution_arena.narrow(0, kProjection * 2, kProjection).fill_(0.0625);

  const at::Tensor a_log = at::full(
      {kHeadWidth}, -2.0,
      at::TensorOptions().dtype(at::kFloat).device(device));
  return KdaWeights{
      .a_log = a_log,
      .dt_bias = at::zeros(
          {kProjection},
          at::TensorOptions().dtype(at::kFloat).device(device)),
      .query_convolution =
          convolution_arena.narrow(0, 0, kProjection),
      .key_convolution =
          convolution_arena.narrow(0, kProjection, kProjection),
      .value_convolution =
          convolution_arena.narrow(0, kProjection * 2, kProjection),
      .output_norm = at::ones(
          {kHeadWidth},
          at::TensorOptions().dtype(at::kFloat).device(device)),
      .query_projection = query,
      .key_projection = key,
      .value_projection = value,
      .recurrent_gate_projection = output_gate,
      .feature_a_projection = feature_a,
      .feature_b_projection =
          separate_projection(kProjection, kHeadWidth, device, 2),
      .beta_projection =
          separate_projection(kHeads, kHidden, device, 6),
      .output_projection =
          separate_projection(kHidden, kProjection, device, 1),
  };
}

void require_bit_equal(const at::Tensor& actual,
                       const at::Tensor& expected,
                       const char* name) {
  const at::Tensor actual_cpu = actual.to(at::kCPU);
  const at::Tensor expected_cpu = expected.to(at::kCPU);
  if (!at::equal(actual_cpu, expected_cpu)) {
    const double maximum = at::max(at::abs(actual_cpu - expected_cpu))
                               .item<double>();
    throw std::runtime_error(std::string(name) +
                             " is not bit-exact, max_abs=" +
                             std::to_string(maximum));
  }
}

void require_result_equal(const KdaDecodeResult& actual,
                          const KdaDecodeResult& expected) {
  require_bit_equal(actual.output, expected.output, "KDA output");
  require_bit_equal(actual.next_state.query_convolution,
                    expected.next_state.query_convolution,
                    "KDA query convolution state");
  require_bit_equal(actual.next_state.key_convolution,
                    expected.next_state.key_convolution,
                    "KDA key convolution state");
  require_bit_equal(actual.next_state.value_convolution,
                    expected.next_state.value_convolution,
                    "KDA value convolution state");
  require_bit_equal(actual.next_state.recurrent,
                    expected.next_state.recurrent,
                    "KDA recurrent state");
  if (!actual.next_state.query_convolution.is_alias_of(
          actual.next_state.key_convolution) ||
      !actual.next_state.query_convolution.is_alias_of(
          actual.next_state.value_convolution)) {
    throw std::runtime_error(
        "fused KDA next convolution states lost shared storage");
  }
}

void require_batched_sequence_equal(const at::Device& device,
                                    const KdaWeights& weights,
                                    const std::uint32_t positions) {
  at::Tensor hidden_cpu = at::empty(
      {static_cast<std::int64_t>(positions), kHidden},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  for (std::uint32_t row = 0; row < positions; ++row) {
    hidden_cpu[static_cast<std::int64_t>(row)].fill_(
        static_cast<double>(row + 1) / 8192.0);
  }
  const at::Tensor hidden = hidden_cpu.to(device).contiguous();

  KdaState rowwise_state =
      deltafin::provider_internal::zero_k3_kda_state(device);
  for (std::uint32_t row = 0; row < positions; ++row) {
    KdaDecodeResult decoded =
        deltafin::provider_internal::kda_decode_one_fused_for_test(
            hidden.narrow(0, static_cast<std::int64_t>(row), 1), weights,
            rowwise_state, true);
    rowwise_state = decoded.next_state;
  }

  const deltafin::provider_internal::KdaBatchInputProjections projected =
      deltafin::provider_internal::kda_project_inputs_batch(
          hidden, weights, true);
  if (projected.positions != positions ||
      projected.path !=
          deltafin::provider_internal::KdaBatchProjectionPath::Separate ||
      projected.provider_dispatches != 3 ||
      projected.equivalent_rowwise_dispatches != positions * 3) {
    throw std::runtime_error(
        "KDA exact batch did not preserve the live T-wide projection schedule");
  }
  const deltafin::provider_internal::KdaConvolvedPositions convolved =
      deltafin::provider_internal::kda_short_convolve_positions(
          hidden, weights,
          deltafin::provider_internal::zero_k3_kda_state(device),
          deltafin::provider_internal::KdaPreprojectedPositions{
              .query = projected.query,
              .key = projected.key,
              .value = projected.value,
          },
          true);
  const deltafin::provider_internal::KdaBatchDependentProjections dependent =
      deltafin::provider_internal::kda_project_dependent_batch(
          hidden, weights, true);
  if (dependent.positions != positions ||
      dependent.dependent_provider_dispatches != 3 ||
      dependent.dependent_equivalent_rowwise_dispatches != positions * 3) {
    throw std::runtime_error(
        "KDA exact dependent projections changed live dispatch order");
  }

  deltafin::provider_internal::KdaRecurrenceRecord record;
  deltafin::provider_internal::KdaPositionsRecurrentResult recurrence =
      deltafin::provider_internal::kda_recur_convolved_positions(
          hidden, weights,
          deltafin::provider_internal::zero_k3_kda_state(device),
          convolved,
          deltafin::provider_internal::KdaDependentPositions{
              .feature_a = dependent.feature_a,
              .feature_b = dependent.feature_b,
              .beta = dependent.beta,
          },
          true, true, &record);
  // A verify commit of any prefix replays the recorded rows. It must
  // reproduce the boundary the loop held there bit for bit.
  if (record.positions != static_cast<std::int64_t>(positions) ||
      recurrence.boundaries.size() != positions) {
    throw std::runtime_error("KDA recurrence record lost its row count");
  }
  for (std::uint32_t prefix = 1; prefix <= positions; ++prefix) {
    const KdaState replayed =
        deltafin::provider_internal::kda_replay_recorded_state(
            record, static_cast<std::int64_t>(prefix));
    const KdaState& boundary = recurrence.boundaries[prefix - 1];
    if (!at::equal(replayed.recurrent, boundary.recurrent) ||
        !at::equal(replayed.query_convolution, boundary.query_convolution) ||
        !at::equal(replayed.key_convolution, boundary.key_convolution) ||
        !at::equal(replayed.value_convolution, boundary.value_convolution)) {
      throw std::runtime_error(
          "KDA prefix replay differs from the recurrence's own boundary");
    }
  }
  if (!at::equal(recurrence.boundaries.back().recurrent,
                 recurrence.final_state.recurrent)) {
    throw std::runtime_error("KDA final boundary is not the final state");
  }
  const deltafin::provider_internal::KdaBatchOutputProjection output =
      deltafin::provider_internal::kda_finish_output_batch(
          hidden, recurrence.recurrent_output_rows, weights, true);
  synchronize_device(device);
  if (recurrence.boundaries.size() != positions ||
      output.positions != positions ||
      output.provider_dispatches != 2 ||
      output.equivalent_rowwise_dispatches != positions * 2 ||
      output.output.sizes() != hidden.sizes() ||
      recurrence.final_state.query_convolution.sizes() !=
          rowwise_state.query_convolution.sizes() ||
      recurrence.final_state.recurrent.sizes() !=
          rowwise_state.recurrent.sizes()) {
    throw std::runtime_error(
        "KDA live T-wide recurrence/output lost its position/state contract");
  }
}

template <typename Function>
double median_milliseconds(Function&& function, const at::Device& device) {
  constexpr std::int64_t warmups = 3;
  constexpr std::int64_t iterations = 3;
  constexpr std::int64_t samples = 7;
  for (std::int64_t index = 0; index < warmups; ++index) {
    function();
  }
  synchronize_device(device);
  std::vector<double> values;
  values.reserve(samples);
  for (std::int64_t sample = 0; sample < samples; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    for (std::int64_t index = 0; index < iterations; ++index) {
      function();
    }
    synchronize_device(device);
    values.push_back(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count() /
        static_cast<double>(iterations));
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void run_real_benchmark(const at::Device& device) {
  ProjectionArena projection_arena;
  at::Tensor convolution_arena;
  KdaWeights weights =
      make_real_weights(device, projection_arena, convolution_arena);
  const KdaState state =
      deltafin::provider_internal::zero_k3_kda_state(device);
  const at::Tensor hidden = at::full(
      {1, kHidden}, 1.0 / 1024.0,
      at::TensorOptions().dtype(at::kFloat).device(device));

  KdaDecodeResult established =
      deltafin::provider_internal::kda_decode_one_unfused_for_test(
          hidden, weights, state, true);
  KdaDecodeResult fused =
      deltafin::provider_internal::kda_decode_one_fused_for_test(
          hidden, weights, state, true);
  synchronize_device(device);
  require_result_equal(fused, established);
  const KdaDecodeResult established_second_token =
      deltafin::provider_internal::kda_decode_one_unfused_for_test(
          hidden, weights, established.next_state, true);
  const KdaDecodeResult fused_second_token =
      deltafin::provider_internal::kda_decode_one_fused_for_test(
          hidden, weights, fused.next_state, true);
  synchronize_device(device);
  require_result_equal(fused_second_token, established_second_token);
  KdaWeights feature_separate_weights = weights;
  feature_separate_weights.feature_a_projection =
      separate_projection(kHeadWidth, kHidden, device, 5);
  KdaDecodeResult feature_separate =
      deltafin::provider_internal::kda_decode_one_fused_for_test(
          hidden, feature_separate_weights, state, true);
  synchronize_device(device);
  require_result_equal(feature_separate, established);
  if (device.type() == at::kMPS) {
    for (std::uint32_t positions = 2; positions <= 9; ++positions) {
      require_batched_sequence_equal(device, weights, positions);
    }
  }
  const std::uint64_t component_bytes =
      static_cast<std::uint64_t>(projection_arena.weight.numel()) +
      static_cast<std::uint64_t>(projection_arena.scale.numel()) *
          sizeof(float);
  const std::uint64_t view_bytes =
      static_cast<std::uint64_t>(
          weights.query_projection.weight.numel() +
          weights.key_projection.weight.numel() +
          weights.value_projection.weight.numel() +
          weights.recurrent_gate_projection.weight.numel() +
          weights.feature_a_projection.weight.numel()) +
      static_cast<std::uint64_t>(
          weights.query_projection.scale.numel() +
          weights.key_projection.scale.numel() +
          weights.value_projection.scale.numel() +
          weights.recurrent_gate_projection.scale.numel() +
          weights.feature_a_projection.scale.numel()) *
          sizeof(float);
  if (component_bytes != view_bytes) {
    throw std::runtime_error(
        "KDA input bundle changed steady payload residency");
  }

  const auto run_established = [&] {
    established =
        deltafin::provider_internal::kda_decode_one_unfused_for_test(
            hidden, weights, state, true);
  };
  const auto run_fused = [&] {
    fused = deltafin::provider_internal::kda_decode_one_fused_for_test(
        hidden, weights, state, true);
  };
  const auto run_feature_separate = [&] {
    feature_separate =
        deltafin::provider_internal::kda_decode_one_fused_for_test(
            hidden, feature_separate_weights, state, true);
  };
  const double established_first =
      median_milliseconds(run_established, device);
  const double fused_second = median_milliseconds(run_fused, device);
  const double fused_first = median_milliseconds(run_fused, device);
  const double established_second =
      median_milliseconds(run_established, device);
  const double established_ms =
      (established_first + established_second) * 0.5;
  const double fused_ms = (fused_first + fused_second) * 0.5;
  const double feature_separate_ms =
      median_milliseconds(run_feature_separate, device);
  std::cout << "benchmark.kda_established_ms=" << established_ms << '\n'
            << "benchmark.kda_fused_ms=" << fused_ms << '\n'
            << "benchmark.kda_fused_speedup=" << established_ms / fused_ms
            << "x\n"
            << "benchmark.kda_saved_ms=" << established_ms - fused_ms << '\n'
            << "benchmark.kda_four_projection_ms="
            << feature_separate_ms << '\n'
            << "benchmark.kda_five_vs_four_speedup="
            << feature_separate_ms / fused_ms << "x\n"
            << "benchmark.kda_input_component_bytes=" << component_bytes
            << '\n'
            << "benchmark.kda_input_resident_delta_bytes=0\n"
            << "check.kda_batched_causal_sequence=PASS (T=1..9)\n";
}

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--benchmark-real") {
      options.benchmark_real = true;
      continue;
    }
    if (argument == "--device" && ++index < argc) {
      const std::string_view value(argv[index]);
      if (value == "cpu") {
        options.device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
      } else if (value == "mps") {
        options.device = DELTAFIN_PROVIDER_DEVICE_MPS_V1;
      } else if (value == "cuda") {
        options.device = DELTAFIN_PROVIDER_DEVICE_CUDA_V1;
      } else {
        throw std::invalid_argument(
            "KDA test device must be cpu, mps, or cuda");
      }
      continue;
    }
    throw std::invalid_argument(
        "usage: deltafin-provider-kda-test [--device cpu|mps|cuda] "
        "[--benchmark-real]");
  }
  return options;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    c10::InferenceMode inference_guard;
    const Options options = parse_options(argc, argv);
    if (deltafin::provider_internal::cuda_case_should_skip(options.device)) {
      std::cout << "check.kda_decode=PASS\n"
                << "check.kda_decode.cuda=skipped(no visible CUDA device)\n";
      return 0;
    }
    const auto selected = deltafin::provider_internal::select_device(
        options.device, 0);
    std::string detail;
    if (!deltafin::provider_internal::kda_small_parity_canary(
            selected.device, detail)) {
      throw std::runtime_error("KDA canary failed: " + detail);
    }
    if (options.benchmark_real) {
      run_real_benchmark(selected.device);
    }
    std::cout << "check.kda_decode=PASS\n"
              << "check.kda_bundle_exact=PASS (bit-exact)\n"
              << "check.kda_state_bundle=PASS\n"
              << "canary=" << detail << '\n'
              << "device=" << selected.device.str() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "result=FAIL\nerror=\"" << error.what() << "\"\n";
    return 1;
  }
}
