#include "provider_abi.h"
#include "provider_cuda_moe.h"
#include "provider_device.h"
#include "provider_spine_bf16_cuda.h"

#include <ATen/ATen.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/add.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/softmax.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>
#if defined(DELTAFIN_HAVE_CUDA_PROVIDER_V1)
#include <ATen/cuda/CUDAContextLight.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#if defined(DELTAFIN_HAVE_CUDA_PROVIDER_V1)
// torch_cuda.dll carries the CUDA ATen kernels and registers them from its own
// static initializers. The provider reaches CUDA only through `c10::cuda`
// (c10_cuda) and the inline `at::cuda::getNumGPUs`, so nothing would otherwise
// keep an import of torch_cuda. This non-const namespace-scope variable has
// external linkage, so the object always carries the relocation and the linker
// emits the import: the DLL loads and its kernels register before any CUDA
// tensor operation.
decltype(&at::cuda::getCurrentDeviceProperties) k3_torch_cuda_anchor_abi =
    &at::cuda::getCurrentDeviceProperties;
#endif

static_assert(sizeof(DeltafinProviderInventoryV1) == 96,
              "provider inventory ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderCanaryRequestV1) == 88,
              "provider canary request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderCanaryReportV1) == 1120,
              "provider canary report ABI v1 layout changed");

struct CheckResult {
  bool passed = false;
  std::string detail;
};

void copy_text(char* destination, const std::size_t capacity,
               const std::string& value) noexcept {
  if (destination == nullptr || capacity == 0) {
    return;
  }
  const std::size_t copied = std::min(capacity - 1, value.size());
  std::memcpy(destination, value.data(), copied);
  destination[copied] = '\0';
}

std::string tensor_summary(const at::Tensor& value) {
  const at::Tensor flat = value.detach().to(at::kCPU).reshape({-1});
  const double minimum = flat.min().item<double>();
  const double maximum = flat.max().item<double>();
  return "shape=" + c10::str(value.sizes()) + " dtype=" +
      c10::toString(value.scalar_type()) + " min=" + c10::str(minimum) +
      " max=" + c10::str(maximum);
}

CheckResult check_rms_program(const at::Device& device) {
  try {
    const auto cpu_float = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const auto target_float = cpu_float.device(device);
    const at::Tensor input = at::full({2, 8}, 2.0, target_float);
    const at::Tensor weight = at::ones({8}, target_float);

    // Preserve K3's ordered fp32 RMS-style ATen program. The exact-valued
    // capability canary uses eps=0: mean(2^2)=4 and rsqrt(4)=1/2.
    const at::Tensor input_float = input.to(at::kFloat);
    const at::Tensor variance = at::mean(
        at::pow(input_float, 2), std::vector<std::int64_t>{-1}, true);
    const at::Tensor normalized = at::mul(
        input_float, at::rsqrt(at::add(variance, 0.0)));
    const at::Tensor output = at::mul(weight, normalized);
    const at::Tensor expected = at::ones({2, 8}, cpu_float);
    const bool exact = at::equal(output.to(at::kCPU), expected);
    return {exact, exact ? tensor_summary(output)
                         : "analytical fp32 RMS canary was not bit-exact"};
  } catch (const std::exception& error) {
    return {false, error.what()};
  }
}

CheckResult check_matmul(const at::Device& device) {
  try {
    const auto target_float =
        at::TensorOptions().dtype(at::kFloat).device(device);
    const auto cpu_float =
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const at::Tensor left = at::ones({2, 8}, target_float);
    const at::Tensor right = at::full({8, 3}, 0.5, target_float);
    const at::Tensor output = at::matmul(left, right);
    const at::Tensor expected = at::full({2, 3}, 4.0, cpu_float);
    const bool exact = at::equal(output.to(at::kCPU), expected);
    return {exact, exact ? tensor_summary(output)
                         : "analytical fp32 matmul canary was not bit-exact"};
  } catch (const std::exception& error) {
    return {false, error.what()};
  }
}

CheckResult check_softmax(const at::Device& device) {
  try {
    const auto target_float =
        at::TensorOptions().dtype(at::kFloat).device(device);
    const auto cpu_float =
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const at::Tensor logits = at::zeros({2, 4}, target_float);
    const at::Tensor output = at::softmax(logits, -1);
    const at::Tensor expected = at::full({2, 4}, 0.25, cpu_float);
    const bool exact = at::equal(output.to(at::kCPU), expected);
    return {exact, exact ? tensor_summary(output)
                         : "analytical fp32 softmax canary was not bit-exact"};
  } catch (const std::exception& error) {
    return {false, error.what()};
  }
}

std::vector<std::int64_t> probe_indices(const std::int64_t extent) {
  const std::array<std::int64_t, 9> candidates = {
      0, 1, 2, 3, 31, 32, extent / 2, extent - 2, extent - 1};
  std::vector<std::int64_t> indices;
  for (const std::int64_t candidate : candidates) {
    if (candidate >= 0 && candidate < extent) {
      indices.push_back(candidate);
    }
  }
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return indices;
}

CheckResult check_packed_int8(const at::Device& device,
                              const std::uint64_t requested_rows,
                              const std::uint64_t requested_columns) {
  try {
    const std::uint64_t rows_u64 = requested_rows == 0 ? 32 : requested_rows;
    const std::uint64_t columns_u64 =
        requested_columns == 0 ? 32 : requested_columns;
    if (rows_u64 > static_cast<std::uint64_t>(
                       std::numeric_limits<std::int64_t>::max()) ||
        columns_u64 > static_cast<std::uint64_t>(
                          std::numeric_limits<std::int64_t>::max()) ||
        rows_u64 > std::numeric_limits<std::uint64_t>::max() / columns_u64) {
      throw std::invalid_argument("packed-int8 canary shape overflows");
    }
    const std::uint64_t elements = rows_u64 * columns_u64;
    if (elements > (UINT64_C(2) << 30)) {
      throw std::invalid_argument(
          "packed-int8 canary exceeds the two-billion-element safety limit");
    }
    const auto rows = static_cast<std::int64_t>(rows_u64);
    const auto columns = static_cast<std::int64_t>(columns_u64);

    auto qweight_cpu = at::zeros(
        {rows, columns}, at::TensorOptions().dtype(at::kChar));
    auto hidden_cpu = at::zeros(
        {1, columns}, at::TensorOptions().dtype(at::kFloat));
    auto scale_cpu = at::ones(
        {rows}, at::TensorOptions().dtype(at::kFloat));
    auto expected = at::zeros(
        {1, rows}, at::TensorOptions().dtype(at::kFloat));

    const std::vector<std::int64_t> tested_rows = probe_indices(rows);
    const std::vector<std::int64_t> tested_columns = probe_indices(columns);
    auto qweight = qweight_cpu.accessor<std::int8_t, 2>();
    auto hidden = hidden_cpu.accessor<float, 2>();
    auto scale = scale_cpu.accessor<float, 1>();
    auto expected_values = expected.accessor<float, 2>();
    constexpr std::array<float, 8> activations = {
        1.0F, -2.0F, 0.5F, 4.0F, -0.25F, 8.0F, 0.125F, -16.0F};
    constexpr std::array<float, 4> scales = {0.5F, 0.25F, 2.0F, 1.0F};

    for (std::size_t column_index = 0;
         column_index < tested_columns.size(); ++column_index) {
      hidden[0][tested_columns[column_index]] =
          activations[column_index % activations.size()];
    }
    for (std::size_t row_index = 0; row_index < tested_rows.size(); ++row_index) {
      const std::int64_t row = tested_rows[row_index];
      float total = 0.0F;
      for (std::size_t column_index = 0;
           column_index < tested_columns.size(); ++column_index) {
        const std::int64_t column = tested_columns[column_index];
        std::int8_t weight = static_cast<std::int8_t>(
            ((row_index + 3) * (column_index + 5)) % 15 - 7);
        if (weight == 0) {
          weight = 3;
        }
        qweight[row][column] = weight;
        total += activations[column_index % activations.size()] *
                 static_cast<float>(weight);
      }
      const float row_scale = scales[row_index % scales.size()];
      scale[row] = row_scale;
      expected_values[0][row] = total * row_scale;
    }

    // Production passes fp32 activations and row scales. Testing half here
    // would validate a different private-operator overload and is forbidden.
    const auto target_float =
        at::TensorOptions().dtype(at::kFloat).device(device);
    const at::Tensor output = at::_weight_int8pack_mm(
        hidden_cpu.to(target_float),
        qweight_cpu.to(at::TensorOptions().dtype(at::kChar).device(device)),
        scale_cpu.to(target_float));
    const at::Tensor actual = output.to(at::kCPU);
    const bool exact = at::equal(actual, expected);
    if (!exact) {
      const double maximum_error =
          at::max(at::abs(actual.to(at::kFloat) - expected)).item<double>();
      return {false, "analytical fp32 packed-int8 canary differed; max_abs=" +
                         c10::str(maximum_error)};
    }
    return {true, tensor_summary(output)};
  } catch (const std::exception& error) {
    return {false, error.what()};
  }
}

void append_detail(std::ostringstream& details, const char* name,
                   const CheckResult& result) {
  if (details.tellp() > 0) {
    details << "; ";
  }
  details << name << '=' << (result.passed ? "PASS" : "FAIL") << " ("
          << result.detail << ')';
}

}  // namespace

extern "C" uint32_t deltafin_provider_abi_version(void) {
  return DELTAFIN_PROVIDER_ABI_VERSION;
}

extern "C" int32_t deltafin_provider_inventory_v1(
    DeltafinProviderInventoryV1* inventory, char* error,
    const size_t error_capacity) {
  try {
    if (inventory == nullptr) {
      throw std::invalid_argument("provider inventory pointer is null");
    }
    if (inventory->struct_size != sizeof(DeltafinProviderInventoryV1)) {
      throw std::invalid_argument("provider inventory struct size does not match ABI v1");
    }
    std::memset(inventory, 0, sizeof(*inventory));
    inventory->struct_size = sizeof(*inventory);
    inventory->abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    inventory->cpu_available = 1;
    inventory->mps_available =
        deltafin::provider_internal::mps_available() ? 1u : 0u;
    inventory->cuda_device_count =
        deltafin::provider_internal::cuda_device_count();
    inventory->provider_features = 0;
    if (deltafin::provider_internal::cuda_moe_compiled()) {
      inventory->provider_features |=
          DELTAFIN_PROVIDER_FEATURE_CUDA_MOE_V1;
    }
    if (deltafin::provider_internal::cuda_spine_bf16_compiled()) {
      inventory->provider_features |=
          DELTAFIN_PROVIDER_FEATURE_CUDA_EXACT_BF16_V1;
    }
    copy_text(inventory->libtorch_version, sizeof(inventory->libtorch_version),
              TORCH_VERSION);
    copy_text(error, error_capacity, "");
    return 0;
  } catch (const std::exception& exception) {
    copy_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    copy_text(error, error_capacity, "unknown C++ provider inventory failure");
    return 1;
  }
}

extern "C" int32_t deltafin_provider_canary_v1(
    const DeltafinProviderCanaryRequestV1* request,
    DeltafinProviderCanaryReportV1* report, char* error,
    const size_t error_capacity) {
  try {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider canary request/report pointer is null");
    }
    if (request->struct_size != sizeof(DeltafinProviderCanaryRequestV1) ||
        request->abi_version != DELTAFIN_PROVIDER_ABI_VERSION) {
      throw std::invalid_argument("provider canary request does not match ABI v1");
    }
    if (report->struct_size != sizeof(DeltafinProviderCanaryReportV1)) {
      throw std::invalid_argument("provider canary report struct size does not match ABI v1");
    }

    std::memset(report, 0, sizeof(*report));
    report->struct_size = sizeof(*report);
    report->abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    const deltafin::provider_internal::SelectedDevice selected =
        deltafin::provider_internal::select_device(
            request->requested_device, request->device_index);
    report->selected_device = selected.kind;
    report->device_index = selected.index;
    report->packed_rows = request->packed_rows == 0 ? 32 : request->packed_rows;
    report->packed_columns =
        request->packed_columns == 0 ? 32 : request->packed_columns;

    const c10::InferenceMode inference_guard;
    const CheckResult rms = check_rms_program(selected.device);
    const CheckResult matmul = check_matmul(selected.device);
    const CheckResult softmax = check_softmax(selected.device);
    const CheckResult packed = check_packed_int8(
        selected.device, report->packed_rows, report->packed_columns);

    report->attempted_checks =
        DELTAFIN_PROVIDER_CHECK_RMS_FP32_V1 |
        DELTAFIN_PROVIDER_CHECK_MATMUL_FP32_V1 |
        DELTAFIN_PROVIDER_CHECK_SOFTMAX_FP32_V1 |
        DELTAFIN_PROVIDER_CHECK_PACKED_INT8_FP32_V1;
    if (rms.passed) {
      report->passed_checks |= DELTAFIN_PROVIDER_CHECK_RMS_FP32_V1;
    }
    if (matmul.passed) {
      report->passed_checks |= DELTAFIN_PROVIDER_CHECK_MATMUL_FP32_V1;
    }
    if (softmax.passed) {
      report->passed_checks |= DELTAFIN_PROVIDER_CHECK_SOFTMAX_FP32_V1;
    }
    if (packed.passed) {
      report->passed_checks |= DELTAFIN_PROVIDER_CHECK_PACKED_INT8_FP32_V1;
    }

    const std::uint32_t core_checks =
        DELTAFIN_PROVIDER_CHECK_RMS_FP32_V1 |
        DELTAFIN_PROVIDER_CHECK_MATMUL_FP32_V1 |
        DELTAFIN_PROVIDER_CHECK_SOFTMAX_FP32_V1;
    const bool require_packed =
        (request->flags & DELTAFIN_PROVIDER_REQUIRE_PACKED_INT8_V1) != 0;
    report->required_passed =
        ((report->passed_checks & core_checks) == core_checks &&
         (!require_packed || packed.passed))
        ? 1u
        : 0u;

    std::ostringstream details;
    append_detail(details, "rms_fp32", rms);
    append_detail(details, "matmul_fp32", matmul);
    append_detail(details, "softmax_fp32", softmax);
    append_detail(details, "packed_int8_fp32", packed);
    copy_text(report->detail, sizeof(report->detail), details.str());
    copy_text(error, error_capacity, "");
    return 0;
  } catch (const std::exception& exception) {
    copy_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    copy_text(error, error_capacity, "unknown C++ provider canary failure");
    return 1;
  }
}
