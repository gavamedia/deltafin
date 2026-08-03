#include "provider_cuda_moe.h"

#include <ATen/ops/cat.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(DELTAFIN_HAVE_CUDA_MOE_V1) && \
    !defined(DELTAFIN_HAVE_CUDA_PROVIDER_V1)
#error "CUDA MXFP4 requires the qualified CUDA LibTorch provider"
#endif

#if defined(DELTAFIN_HAVE_CUDA_PROVIDER_V1)
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#if defined(DELTAFIN_HAVE_CUDA_MOE_V1)
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

extern "C" {
std::uint32_t k3_cuda_moe_abi_version(void);
void k3_cuda_moe_shapes(int* hidden, int* intermediate,
                        std::int64_t* expert_span,
                        std::uint32_t* pointer_layout);
const char* k3_cuda_last_error(void);
int k3_cuda_moe_available(int device);
int k3_cuda_mxfp4_gemv(const std::uint8_t* packed,
                       const std::uint8_t* scales, const float* x,
                       float* output, int rows, int cols, int batches,
                       void* stream_pointer);
int k3_cuda_moe_launch(const std::uint8_t* expert_span, const float* x,
                       int batches, float* gate, float* up,
                       float* expert_output, void* stream_pointer);
}
#endif

namespace deltafin::provider_internal {

namespace {

constexpr std::size_t kCudaLayerStrata = 92;
constexpr std::size_t kCudaMaximumExperts =
    kCudaLayerStrata * kMoeRouteTopK;

void validate_cache_policy(const CudaMoeCachePolicy& policy) {
  if ((policy.automatic_capacity && policy.capacity_experts != 0) ||
      (!policy.automatic_capacity &&
       policy.capacity_experts > kCudaMaximumExperts) ||
      (policy.reserve_kind == CudaMoeCacheReserveKind::Auto &&
       policy.reserve_value != 0) ||
      (policy.reserve_kind == CudaMoeCacheReserveKind::RatioPpm &&
       policy.reserve_value > 1'000'000) ||
      (policy.reserve_kind != CudaMoeCacheReserveKind::Auto &&
       policy.reserve_kind != CudaMoeCacheReserveKind::Bytes &&
       policy.reserve_kind != CudaMoeCacheReserveKind::RatioPpm)) {
    throw std::invalid_argument(
        "CUDA expert cache policy is outside its bounds");
  }
}

constexpr std::size_t clamp_cache_capacity(
    const std::size_t free_bytes, const std::size_t reserve_bytes,
    const bool automatic_capacity, const std::uint64_t requested_experts,
    const std::size_t expert_span_bytes) noexcept {
  if (expert_span_bytes == 0) {
    return 0;
  }
  const std::size_t safe_bytes =
      free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
  const std::size_t safe_capacity = std::min(
      kCudaMaximumExperts, safe_bytes / expert_span_bytes);
  const std::size_t requested =
      requested_experts > std::numeric_limits<std::size_t>::max()
      ? std::numeric_limits<std::size_t>::max()
      : static_cast<std::size_t>(requested_experts);
  return automatic_capacity ? safe_capacity
                            : std::min(safe_capacity, requested);
}

static_assert(clamp_cache_capacity(10, 10, true, 0, 1) == 0);
static_assert(clamp_cache_capacity(10, 0, false, 0, 1) == 0);
static_assert(clamp_cache_capacity(10, 0, false, 4, 1) == 4);
static_assert(clamp_cache_capacity(10, 0, false, 40, 1) == 10);

}  // namespace

#if !defined(DELTAFIN_HAVE_CUDA_MOE_V1)

struct CudaMoeExpertCache::Impl {
  explicit Impl(const at::Device& selected)
      : device(selected),
        reason("CUDA MXFP4 was not compiled into this provider") {}
  at::Device device;
  bool configuration_frozen = false;
  CudaMoeCachePolicy policy;
  std::string reason;
};

CudaMoeExpertCache::CudaMoeExpertCache(const at::Device& device)
    : impl_(std::make_unique<Impl>(device)) {}

CudaMoeExpertCache::~CudaMoeExpertCache() = default;

bool CudaMoeExpertCache::available() {
  impl_->configuration_frozen = true;
  return false;
}

const std::string& CudaMoeExpertCache::detail() const {
  return impl_->reason;
}

void CudaMoeExpertCache::configure(const CudaMoeCachePolicy& policy) {
  if (impl_->configuration_frozen) {
    throw std::logic_error("CUDA expert cache policy is already frozen");
  }
  validate_cache_policy(policy);
  impl_->policy = policy;
  impl_->configuration_frozen = true;
}

CudaMoeResidencyPlanReport CudaMoeExpertCache::plan(
    const std::uint64_t, const std::uint32_t,
    const std::span<const std::uint16_t>) {
  throw std::runtime_error(impl_->reason);
}

void CudaMoeExpertCache::cancel_plan(const std::uint64_t) noexcept {}

CudaMoeHostFallback CudaMoeExpertCache::materialize_plan_for_cpu(
    const std::uint64_t) {
  throw std::runtime_error(impl_->reason);
}

void CudaMoeExpertCache::poison_external(const char* failure) noexcept {
  try {
    impl_->reason = std::string("CUDA MXFP4 disabled after runtime failure: ") +
        (failure == nullptr ? "no detail" : failure);
  } catch (...) {
    // Poisoning is a best-effort diagnostic operation and must never terminate
    // a portable provider process merely because formatting the detail failed.
  }
}

at::Tensor CudaMoeExpertCache::execute_t1(
    const PreparedMoeT1&, const CanonicalExpertBatchT1&) {
  throw std::runtime_error(impl_->reason);
}

at::Tensor CudaMoeExpertCache::execute_positions_t1(
    std::span<const PreparedMoeT1* const>,
    const CanonicalExpertPositionTileT1&) {
  throw std::runtime_error(impl_->reason);
}

at::Tensor CudaMoeExpertCache::execute_positions_plan_t1(
    const std::uint64_t,
    std::span<const PreparedMoeT1* const>,
    const CanonicalExpertPositionTileT1&) {
  throw std::runtime_error(impl_->reason);
}

bool cuda_moe_compiled() noexcept { return false; }

#else

namespace {

constexpr std::uint32_t kCudaMoeAbi = 1;
constexpr std::uint32_t kCudaPointerLayout = 1;
constexpr std::int64_t kCudaHidden = 3584;
constexpr std::int64_t kCudaIntermediate = 3072;
constexpr std::int64_t kCudaPackedBytes = 5'505'024;
constexpr std::int64_t kCudaScaleBytes = 344'064;
constexpr std::int64_t kCudaExpertSpan = 17'547'264;

void require_exact_cuda_geometry(const MoeGeometry& geometry) {
  const MoeGeometry expected = k3_moe_geometry();
  if (geometry.hidden != expected.hidden ||
      geometry.routed_hidden != expected.routed_hidden ||
      geometry.intermediate != expected.intermediate ||
      geometry.experts != expected.experts ||
      geometry.shared_intermediate != expected.shared_intermediate ||
      geometry.expert_span_bytes() !=
          static_cast<std::uint64_t>(kCudaExpertSpan)) {
    throw std::invalid_argument(
        "CUDA MXFP4 accepts only the exact K3 routed-expert geometry");
  }
}

std::string native_error(const char* operation, const int status) {
  const char* detail = k3_cuda_last_error();
  return std::string(operation) + " failed with status " +
      std::to_string(status) + ": " +
      (detail == nullptr ? "no CUDA detail" : detail);
}

void require_launch(const char* operation, const int status) {
  if (status != 0) {
    throw std::runtime_error(native_error(operation, status));
  }
}

void* stream_pointer(const c10::cuda::CUDAStream& stream) {
  return reinterpret_cast<void*>(stream.stream());
}

void record_stream(const at::Tensor& tensor,
                   const c10::cuda::CUDAStream& stream) {
  if (tensor.defined()) {
    c10::cuda::CUDACachingAllocator::recordStream(
        tensor.storage().data_ptr(), stream);
  }
}

template <typename Scalar>
at::Tensor owned_cpu_tensor(const std::vector<Scalar>& values,
                            const at::IntArrayRef shape,
                            const at::ScalarType type) {
  return at::from_blob(const_cast<Scalar*>(values.data()), shape,
                       at::TensorOptions().dtype(type).device(at::kCPU))
      .clone();
}

float e2m1_reference(const std::uint8_t code) {
  constexpr std::array<float, 8> magnitude =
      {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = magnitude[code & 7U];
  return (code & 8U) == 0 ? value : -value;
}

double max_abs(const at::Tensor& left, const at::Tensor& right) {
  return at::max(at::abs(left.to(at::kFloat) - right.to(at::kFloat)))
      .item<double>();
}

at::Tensor ordered_route_reduce(
    const at::Tensor& expert_outputs,
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const at::Device& device) {
  if (prepared_rows.empty() ||
      expert_outputs.sizes() != at::IntArrayRef(
          {static_cast<std::int64_t>(prepared_rows.size() * kMoeRouteTopK),
           kCudaHidden})) {
    throw std::invalid_argument(
        "CUDA ordered reduction received an invalid expert-output matrix");
  }
  std::vector<float> route_weights(prepared_rows.size() * kMoeRouteTopK);
  for (std::size_t row = 0; row < prepared_rows.size(); ++row) {
    if (prepared_rows[row] == nullptr) {
      throw std::invalid_argument("CUDA ordered reduction has a null route row");
    }
    for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
      route_weights[row * kMoeRouteTopK + edge] = std::bit_cast<float>(
          prepared_rows[row]->route.weight_bits[edge]);
    }
  }
  const std::int64_t positions =
      static_cast<std::int64_t>(prepared_rows.size());
  const at::Tensor weights = owned_cpu_tensor(
      route_weights,
      {positions, static_cast<std::int64_t>(kMoeRouteTopK), 1}, at::kFloat)
      .to(device, at::kFloat, false, true);
  const at::Tensor by_slot = expert_outputs.view(
      {positions, static_cast<std::int64_t>(kMoeRouteTopK), kCudaHidden});
  at::Tensor output = at::zeros(
      {positions, kCudaHidden},
      at::TensorOptions().dtype(at::kFloat).device(device));
  for (std::int64_t slot = 0;
       slot < static_cast<std::int64_t>(kMoeRouteTopK); ++slot) {
    output.addcmul_(by_slot.select(1, slot), weights.select(1, slot));
  }
  return output.contiguous();
}

void run_known_answer_tests(const at::Device& device) {
  const c10::cuda::CUDAGuard guard(device);
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  const auto cuda_f32 =
      at::TensorOptions().dtype(at::kFloat).device(device);

  // Every E2M1 code 0x0..0xf appears across these two generated rows.
  std::vector<std::uint8_t> packed(32);
  for (std::size_t index = 0; index < 16; ++index) {
    const int lane = static_cast<int>(index);
    const int forward_phase = (2 * lane) % 16;
    const int reverse_phase = (2 * lane) % 16;
    packed[index] = static_cast<std::uint8_t>(
        forward_phase | (((forward_phase + 1) % 16) << 4));
    packed[16 + index] = static_cast<std::uint8_t>(
        (15 - reverse_phase) | ((14 - reverse_phase) << 4));
  }
  const std::vector<std::uint8_t> scales{127, 126};
  std::vector<float> x(64);
  for (std::size_t column = 0; column < 32; ++column) {
    x[column] = static_cast<float>(column) / 32.0F;
    x[32 + column] =
        (static_cast<float>(column) - 15.0F) / 17.0F;
  }
  std::vector<float> expected_values(4, 0.0F);
  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t row = 0; row < 2; ++row) {
      float total = 0.0F;
      for (std::size_t column = 0; column < 32; ++column) {
        const std::uint8_t byte =
            packed[row * 16 + column / 2];
        const std::uint8_t code = (column & 1U) == 0
            ? static_cast<std::uint8_t>(byte & 15U)
            : static_cast<std::uint8_t>(byte >> 4);
        const float weight = std::ldexp(
            e2m1_reference(code), static_cast<int>(scales[row]) - 127);
        total = std::fma(weight, x[batch * 32 + column], total);
      }
      expected_values[batch * 2 + row] = total;
    }
  }
  const at::Tensor packed_cuda = owned_cpu_tensor(
      packed, {2, 16}, at::kByte).to(device, at::kByte, false, true);
  const at::Tensor scales_cuda = owned_cpu_tensor(
      scales, {2, 1}, at::kByte).to(device, at::kByte, false, true);
  const at::Tensor x_cuda = owned_cpu_tensor(
      x, {2, 32}, at::kFloat).to(device, at::kFloat, false, true);
  at::Tensor first = at::empty({2, 2}, cuda_f32);
  at::Tensor second = at::empty({2, 2}, cuda_f32);
  for (at::Tensor* output : {&first, &second}) {
    for (const at::Tensor& tensor :
         {packed_cuda, scales_cuda, x_cuda, *output}) {
      record_stream(tensor, stream);
    }
    const int status = k3_cuda_mxfp4_gemv(
        packed_cuda.const_data_ptr<std::uint8_t>(),
        scales_cuda.const_data_ptr<std::uint8_t>(),
        x_cuda.const_data_ptr<float>(), output->data_ptr<float>(), 2, 32, 2,
        stream_pointer(stream));
    require_launch("CUDA MXFP4 known-answer launch", status);
  }
  const at::Tensor first_cpu = first.to(at::kCPU);
  const at::Tensor second_cpu = second.to(at::kCPU);
  const at::Tensor expected = owned_cpu_tensor(
      expected_values, {2, 2}, at::kFloat);
  if (!at::equal(first_cpu, second_cpu) ||
      max_abs(first_cpu, expected) > 2.0e-4) {
    throw std::runtime_error(
        "CUDA MXFP4 generated-weight known-answer test failed");
  }

  // Exercise every fixed expert-span offset plus SiTU and W2 at real K3
  // dimensions.  Only one element in each matrix is nonzero.
  std::vector<std::uint8_t> expert(
      static_cast<std::size_t>(kCudaExpertSpan), 0);
  expert[0] = 0x02;
  expert[static_cast<std::size_t>(kCudaPackedBytes)] = 127;
  expert[static_cast<std::size_t>(kCudaPackedBytes + kCudaScaleBytes)] =
      0x02;
  expert[static_cast<std::size_t>(2 * kCudaPackedBytes + kCudaScaleBytes)] =
      127;
  expert[static_cast<std::size_t>(
      2 * (kCudaPackedBytes + kCudaScaleBytes))] = 0x04;
  expert[static_cast<std::size_t>(
      3 * kCudaPackedBytes + 2 * kCudaScaleBytes)] = 127;
  std::vector<float> moe_input(static_cast<std::size_t>(2 * kCudaHidden),
                               0.0F);
  moe_input[0] = 1.0F;
  moe_input[static_cast<std::size_t>(kCudaHidden)] = -0.5F;
  const at::Tensor expert_cuda = owned_cpu_tensor(
      expert, {kCudaExpertSpan}, at::kByte)
      .to(device, at::kByte, false, true);
  const at::Tensor input_cuda = owned_cpu_tensor(
      moe_input, {2, kCudaHidden}, at::kFloat)
      .to(device, at::kFloat, false, true);
  std::array<at::Tensor, 2> moe_outputs{
      at::empty({2, kCudaHidden}, cuda_f32),
      at::empty({2, kCudaHidden}, cuda_f32)};
  for (at::Tensor& output : moe_outputs) {
    at::Tensor gate = at::empty({2, kCudaIntermediate}, cuda_f32);
    at::Tensor up = at::empty({2, kCudaIntermediate}, cuda_f32);
    for (const at::Tensor& tensor :
         {expert_cuda, input_cuda, gate, up, output}) {
      record_stream(tensor, stream);
    }
    const int status = k3_cuda_moe_launch(
        expert_cuda.const_data_ptr<std::uint8_t>(),
        input_cuda.const_data_ptr<float>(), 2, gate.data_ptr<float>(),
        up.data_ptr<float>(), output.data_ptr<float>(),
        stream_pointer(stream));
    require_launch("CUDA full-MoE known-answer launch", status);
  }
  std::vector<float> moe_expected_values(
      static_cast<std::size_t>(2 * kCudaHidden), 0.0F);
  for (std::size_t row = 0; row < 2; ++row) {
    const float input = row == 0 ? 1.0F : -0.5F;
    const float gate = input;
    const float up = 2.0F * input;
    moe_expected_values[row * static_cast<std::size_t>(kCudaHidden)] =
        4.0F * std::tanh(gate / 4.0F) /
        (1.0F + std::exp(-gate)) *
        (25.0F * std::tanh(up / 25.0F));
  }
  const at::Tensor moe_first = moe_outputs[0].to(at::kCPU);
  const at::Tensor moe_second = moe_outputs[1].to(at::kCPU);
  const at::Tensor moe_expected = owned_cpu_tensor(
      moe_expected_values, {2, kCudaHidden}, at::kFloat);
  if (!at::equal(moe_first, moe_second) ||
      max_abs(moe_first, moe_expected) > 2.0e-5) {
    throw std::runtime_error(
        "CUDA full-MoE generated-weight known-answer test failed");
  }
}

std::size_t layer_slot(const std::uint32_t layer_index) {
  if (layer_index < 1 || layer_index > kCudaLayerStrata) {
    throw std::invalid_argument("CUDA expert layer is outside K3 layers 1..92");
  }
  return static_cast<std::size_t>(layer_index - 1);
}

class CudaReadyEvent final {
 public:
  CudaReadyEvent() {
    const cudaError_t status =
        cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaEventCreateWithFlags failed: ") +
          cudaGetErrorString(status));
    }
  }

  ~CudaReadyEvent() {
    if (event_ != nullptr) {
      static_cast<void>(cudaEventDestroy(event_));
    }
  }

  CudaReadyEvent(const CudaReadyEvent&) = delete;
  CudaReadyEvent& operator=(const CudaReadyEvent&) = delete;

  void record(const c10::cuda::CUDAStream& stream) {
    const cudaError_t status = cudaEventRecord(event_, stream.stream());
    if (status != cudaSuccess) {
      // The pinned source cannot be released or overwritten until a failed
      // event record's already-enqueued copy has drained.
      static_cast<void>(cudaStreamSynchronize(stream.stream()));
      throw std::runtime_error(
          std::string("cudaEventRecord failed: ") +
          cudaGetErrorString(status));
    }
  }

  void wait(const c10::cuda::CUDAStream& stream) const {
    const cudaError_t status =
        cudaStreamWaitEvent(stream.stream(), event_, 0);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaStreamWaitEvent failed: ") +
          cudaGetErrorString(status));
    }
  }

  void synchronize() const {
    const cudaError_t status = cudaEventSynchronize(event_);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaEventSynchronize failed: ") +
          cudaGetErrorString(status));
    }
  }

 private:
  cudaEvent_t event_ = nullptr;
};

}  // namespace

struct CudaMoeExpertCache::Impl {
  struct Resident {
    at::Tensor bytes;
    std::shared_ptr<CudaReadyEvent> ready;
  };

  struct Entry {
    std::uint16_t expert = 0;
    Resident resident;
  };

  struct Plan {
    std::uint32_t layer_index = 0;
    std::vector<std::uint16_t> canonical_experts;
    std::vector<Entry> hits;
  };

  explicit Impl(const at::Device& selected) : device(selected) {
    if (!device.is_cuda() || !device.has_index()) {
      reason = "CUDA expert cache requires a canonical indexed CUDA device";
    }
  }

  void poison(const char* failure, const bool terminal_failure) noexcept {
    enabled = false;
    terminal = terminal || terminal_failure;
    try {
      reason = std::string("CUDA MXFP4 disabled after runtime failure: ") +
          (failure == nullptr ? "no detail" : failure);
    } catch (...) {
      // Keep the previous diagnostic if formatting fails. `enabled` and
      // `terminal` above remain the fail-closed authority.
    }
    // Every raw-kernel tensor is recorded on its launch stream before status
    // checking, so dropping these owners cannot recycle storage still in use.
    // Keep the pinned stage/event until destruction: its DMA may still be in
    // flight, and the destructor performs the required host-side wait.
    for (auto& layer : layers) {
      layer.clear();
    }
  }

  void freeze_configuration() noexcept { configuration_frozen = true; }

  void configure(const CudaMoeCachePolicy& requested) {
    if (configuration_frozen || checked || budget_ready || !plans.empty()) {
      throw std::logic_error("CUDA expert cache policy is already frozen");
    }
    validate_cache_policy(requested);
    policy = requested;
    configuration_frozen = true;
  }

  cudaError_t synchronize_execution_stream_noexcept() noexcept {
    try {
      const c10::cuda::CUDAGuard guard(device);
      const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
      return cudaStreamSynchronize(stream.stream());
    } catch (...) {
      return cudaErrorUnknown;
    }
  }

  [[noreturn]] void rethrow_terminal_after_drain(const char* failure) {
    const cudaError_t drained = synchronize_execution_stream_noexcept();
    poison(failure, true);
    if (drained != cudaSuccess) {
      throw std::runtime_error(
          std::string("terminal CUDA expert failure could not drain its stream: ") +
          cudaGetErrorString(drained) + "; original failure: " +
          (failure == nullptr ? "no detail" : failure));
    }
    throw;
  }

  template <typename Function>
  decltype(auto) fail_closed(Function&& function) {
    try {
      return std::forward<Function>(function)();
    } catch (const c10::Error& error) {
      poison(error.what(), true);
      throw;
    } catch (const std::exception& error) {
      poison(error.what(), true);
      throw;
    } catch (...) {
      poison("non-standard CUDA expert exception", true);
      throw;
    }
  }

  template <typename Function>
  decltype(auto) terminal_fail_closed(Function&& function) {
    try {
      return std::forward<Function>(function)();
    } catch (const std::exception& error) {
      rethrow_terminal_after_drain(error.what());
    } catch (...) {
      rethrow_terminal_after_drain("non-standard CUDA expert exception");
    }
  }

  template <typename Function>
  at::Tensor execute_fail_closed(Function&& function) {
    try {
      return std::forward<Function>(function)();
    } catch (const c10::OutOfMemoryError& error) {
      const cudaError_t drained = synchronize_execution_stream_noexcept();
      poison(error.what(), drained != cudaSuccess);
      if (drained == cudaSuccess) {
        throw CudaMoeRecoverableError(
            std::string("recoverable CUDA expert allocation failure: ") +
            error.what());
      }
      throw std::runtime_error(
          std::string("CUDA expert allocation failed and its stream could not drain: ") +
          cudaGetErrorString(drained));
    } catch (const std::exception& error) {
      rethrow_terminal_after_drain(error.what());
    } catch (...) {
      rethrow_terminal_after_drain("non-standard CUDA expert exception");
    }
  }

  ~Impl() {
    if (stage_ready != nullptr) {
      try {
        stage_ready->synchronize();
      } catch (...) {
        // Destructors cannot surface a shutdown-time CUDA failure.  Device
        // tensors remain allocator-owned and CUDA tears the context down.
      }
    }
  }

  std::size_t quota(const std::size_t slot) const {
    const std::size_t base = capacity / kCudaLayerStrata;
    const std::size_t remainder = capacity % kCudaLayerStrata;
    const std::size_t rank = (slot * 53 + 17) % kCudaLayerStrata;
    return base + static_cast<std::size_t>(rank < remainder);
  }

  void establish_budget() {
    if (budget_ready) {
      return;
    }
    const c10::cuda::CUDAGuard guard(device);
    if (const char* override_gb =
        std::getenv("K3_CUDA_EXPERT_CACHE_GB")) {
      // Explicit GiB ceiling for the resident expert cache. The automatic
      // free/5 reserve leaves too little contiguous headroom for transient
      // spine binds on smaller GPUs; this bounds the cache so the engine's
      // own transient arena fits. Decimal GiB, like K3_SPINE_RESIDENT_GB.
      char* end = nullptr;
      const double gigabytes = std::strtod(override_gb, &end);
      if (end == override_gb || *end != '\0' || gigabytes < 0.0 ||
          !std::isfinite(gigabytes)) {
        throw std::invalid_argument(
            "K3_CUDA_EXPERT_CACHE_GB must be a finite non-negative "
            "GiB value");
      }
      const std::size_t bytes =
          static_cast<std::size_t>(gigabytes * 1'000'000'000.0);
      capacity =
          std::min(kCudaMaximumExperts, bytes / kCudaExpertSpan);
      budget_ready = true;
      return;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(status));
    }
    static_cast<void>(total_bytes);
    freeze_configuration();
    constexpr std::size_t two_gib = std::size_t{2} << 30;
    std::size_t reserve = 0;
    switch (policy.reserve_kind) {
      case CudaMoeCacheReserveKind::Auto:
        reserve = std::max(two_gib, free_bytes / 5);
        break;
      case CudaMoeCacheReserveKind::Bytes:
        reserve = policy.reserve_value > std::numeric_limits<std::size_t>::max()
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(policy.reserve_value);
        break;
      case CudaMoeCacheReserveKind::RatioPpm: {
        constexpr std::size_t scale = 1'000'000;
        const std::size_t ratio = static_cast<std::size_t>(policy.reserve_value);
        reserve = (free_bytes / scale) * ratio +
            ((free_bytes % scale) * ratio) / scale;
        break;
      }
    }
    capacity = clamp_cache_capacity(
        free_bytes, reserve, policy.automatic_capacity,
        policy.capacity_experts, static_cast<std::size_t>(kCudaExpertSpan));
    budget_ready = true;
  }

  Resident* hit(const std::uint32_t layer_index,
                const std::uint16_t expert) {
    const std::size_t slot = layer_slot(layer_index);
    auto& entries = layers[slot];
    const auto found = std::find_if(
        entries.begin(), entries.end(), [expert](const Entry& entry) {
          return entry.expert == expert;
        });
    if (found == entries.end()) {
      return nullptr;
    }
    Entry promoted = std::move(*found);
    entries.erase(found);
    entries.push_back(std::move(promoted));
    return &entries.back().resident;
  }

  Plan& require_plan(const std::uint64_t plan_id) {
    const auto found = plans.find(plan_id);
    if (found == plans.end()) {
      throw std::invalid_argument("CUDA expert residency plan is stale or unknown");
    }
    return found->second;
  }

  Resident upload(const std::uint8_t* source,
                  const std::size_t source_bytes) {
    if (source == nullptr ||
        source_bytes != static_cast<std::size_t>(kCudaExpertSpan)) {
      throw std::invalid_argument(
          "CUDA expert source does not have the exact raw-v1 span");
    }
    if (stage_ready != nullptr) {
      // Host writes cannot be ordered by a stream wait.  The CPU must wait
      // until the previous DMA has stopped reading the reusable pinned slab.
      stage_ready->synchronize();
      stage_ready.reset();
    }
    if (!pinned_stage.defined()) {
      pinned_stage = at::empty(
          {kCudaExpertSpan},
          at::TensorOptions()
              .dtype(at::kByte)
              .device(at::kCPU)
              .pinned_memory(true));
      if (!pinned_stage.is_pinned()) {
        throw std::runtime_error(
            "CUDA expert staging allocation is not pinned");
      }
    }
    std::memcpy(pinned_stage.data_ptr<std::uint8_t>(), source, source_bytes);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    auto ready = std::make_shared<CudaReadyEvent>();
    at::Tensor uploaded = at::empty(
        {kCudaExpertSpan},
        at::TensorOptions().dtype(at::kByte).device(device));
    try {
      uploaded.copy_(pinned_stage, true);
      ready->record(stream);
    } catch (...) {
      // `copy_` may have submitted DMA before reporting a later failure. The
      // reusable host slab cannot be overwritten or destroyed until that work
      // has drained, even when no readiness event was successfully recorded.
      static_cast<void>(cudaStreamSynchronize(stream.stream()));
      throw;
    }
    stage_ready = ready;
    return Resident{std::move(uploaded), std::move(ready)};
  }

  Resident resident(const std::uint32_t layer_index,
                    const std::uint16_t expert,
                    const std::uint8_t* source,
                    const std::size_t source_bytes) {
    establish_budget();
    const std::size_t slot = layer_slot(layer_index);
    auto& entries = layers[slot];
    const auto found = std::find_if(
        entries.begin(), entries.end(), [expert](const Entry& entry) {
          return entry.expert == expert;
        });
    if (found != entries.end()) {
      Resident value = found->resident;
      Entry promoted = std::move(*found);
      entries.erase(found);
      entries.push_back(std::move(promoted));
      return value;
    }
    Resident uploaded = upload(source, source_bytes);
    const std::size_t layer_quota = quota(slot);
    if (layer_quota != 0) {
      if (entries.size() >= layer_quota) {
        entries.erase(entries.begin());
      }
      entries.push_back(Entry{expert, uploaded});
    }
    return uploaded;
  }

  Resident planned_resident(
      Plan& plan, const std::uint16_t expert,
      const std::span<const std::uint16_t> missing_ids,
      const std::span<const std::uint8_t> missing_bytes) {
    const auto hit_found = std::find_if(
        plan.hits.begin(), plan.hits.end(), [expert](const Entry& entry) {
          return entry.expert == expert;
        });
    if (hit_found != plan.hits.end()) {
      return hit_found->resident;
    }
    if (!std::binary_search(plan.canonical_experts.begin(),
                            plan.canonical_experts.end(), expert)) {
      throw std::invalid_argument(
          "CUDA execution requested an expert outside its residency plan");
    }
    const std::uint8_t* source = source_for(expert, missing_ids, missing_bytes);
    return resident(plan.layer_index, expert, source,
                    static_cast<std::size_t>(kCudaExpertSpan));
  }

  const std::uint8_t* source_for(
      const std::uint16_t expert,
      const std::span<const std::uint16_t> canonical_ids,
      const std::span<const std::uint8_t> canonical_bytes) const {
    const auto found =
        std::lower_bound(canonical_ids.begin(), canonical_ids.end(), expert);
    if (found == canonical_ids.end() || *found != expert) {
      throw std::invalid_argument(
          "CUDA expert tile is missing a routed expert");
    }
    const std::size_t index =
        static_cast<std::size_t>(found - canonical_ids.begin());
    return canonical_bytes.data() +
        index * static_cast<std::size_t>(kCudaExpertSpan);
  }

  at::Device device;
  bool checked = false;
  bool enabled = false;
  bool terminal = false;
  bool configuration_frozen = false;
  bool budget_ready = false;
  std::size_t capacity = 0;
  CudaMoeCachePolicy policy;
  std::string reason = "CUDA MXFP4 capability has not been checked";
  at::Tensor pinned_stage;
  std::shared_ptr<CudaReadyEvent> stage_ready;
  std::array<std::vector<Entry>, kCudaLayerStrata> layers;
  std::unordered_map<std::uint64_t, Plan> plans;
};

CudaMoeExpertCache::CudaMoeExpertCache(const at::Device& device)
    : impl_(std::make_unique<Impl>(device)) {}

CudaMoeExpertCache::~CudaMoeExpertCache() = default;

bool CudaMoeExpertCache::available() {
  impl_->freeze_configuration();
  if (impl_->checked) {
    if (impl_->terminal) {
      throw std::runtime_error(impl_->reason);
    }
    return impl_->enabled;
  }
  impl_->checked = true;
  try {
    if (!impl_->device.is_cuda() || !impl_->device.has_index()) {
      throw std::invalid_argument(impl_->reason);
    }
    const c10::cuda::CUDAGuard guard(impl_->device);
    if (k3_cuda_moe_abi_version() != kCudaMoeAbi) {
      throw std::runtime_error("CUDA MXFP4 ABI version mismatch");
    }
    int hidden = 0;
    int intermediate = 0;
    std::int64_t span = 0;
    std::uint32_t layout = 0;
    k3_cuda_moe_shapes(&hidden, &intermediate, &span, &layout);
    if (hidden != kCudaHidden || intermediate != kCudaIntermediate ||
        span != kCudaExpertSpan || layout != kCudaPointerLayout) {
      throw std::runtime_error("CUDA MXFP4 shape/layout handshake mismatch");
    }
    const int status = k3_cuda_moe_available(impl_->device.index());
    if (status != 1) {
      throw std::runtime_error(native_error("CUDA device probe", status));
    }
    run_known_answer_tests(impl_->device);
    impl_->enabled = true;
    impl_->reason = "CUDA MXFP4 ABI, device, and known-answer gates passed";
  } catch (const std::exception& error) {
    const cudaError_t drained = impl_->synchronize_execution_stream_noexcept();
    impl_->enabled = false;
    impl_->terminal = drained != cudaSuccess;
    try {
      impl_->reason = drained == cudaSuccess
          ? error.what()
          : std::string("terminal CUDA qualification failure could not drain its stream: ") +
              cudaGetErrorString(drained) + "; original failure: " + error.what();
    } catch (...) {
      // `enabled` and `terminal` remain authoritative if detail formatting
      // itself runs out of memory.
    }
    for (auto& layer : impl_->layers) {
      layer.clear();
    }
  } catch (...) {
    const cudaError_t drained = impl_->synchronize_execution_stream_noexcept();
    impl_->enabled = false;
    impl_->terminal = true;
    try {
      impl_->reason = std::string(
          "terminal non-standard CUDA qualification failure; stream status: ") +
          cudaGetErrorString(drained);
    } catch (...) {
    }
  }
  if (impl_->terminal) {
    throw std::runtime_error(impl_->reason);
  }
  return impl_->enabled;
}

const std::string& CudaMoeExpertCache::detail() const {
  return impl_->reason;
}

void CudaMoeExpertCache::configure(const CudaMoeCachePolicy& policy) {
  impl_->configure(policy);
}

CudaMoeResidencyPlanReport CudaMoeExpertCache::plan(
    const std::uint64_t plan_id, const std::uint32_t layer_index,
    const std::span<const std::uint16_t> canonical_experts) {
  if (plan_id == 0 || canonical_experts.empty() ||
      canonical_experts.size() > kMoePositionTileMaxExperts) {
    throw std::invalid_argument("CUDA expert residency plan has invalid bounds");
  }
  for (std::size_t index = 0; index < canonical_experts.size(); ++index) {
    if (canonical_experts[index] >= k3_moe_geometry().experts ||
        (index != 0 && canonical_experts[index - 1] >=
                           canonical_experts[index])) {
      throw std::invalid_argument(
          "CUDA expert residency plan IDs must be canonical ascending IDs");
    }
  }
  static_cast<void>(layer_slot(layer_index));
  if (!available()) {
    throw std::runtime_error("CUDA MXFP4 unavailable: " + impl_->reason);
  }
  return impl_->fail_closed([&] {
    impl_->establish_budget();
    if (impl_->plans.contains(plan_id)) {
      throw std::invalid_argument("CUDA expert residency plan ID is already live");
    }
    Impl::Plan planned;
    planned.layer_index = layer_index;
    planned.canonical_experts.assign(canonical_experts.begin(),
                                     canonical_experts.end());
    planned.hits.reserve(canonical_experts.size());
    CudaMoeResidencyPlanReport report;
    report.capacity_experts = impl_->capacity;
    report.residency_enabled = impl_->capacity != 0;

    // Snapshot every initial hit before a later miss admission can evict it.
    for (const std::uint16_t expert : canonical_experts) {
      if (Impl::Resident* resident = impl_->hit(layer_index, expert);
          resident != nullptr) {
        planned.hits.push_back(Impl::Entry{expert, *resident});
      } else {
        report.missing_experts.push_back(expert);
      }
    }
    impl_->plans.emplace(plan_id, std::move(planned));
    return report;
  });
}

at::Tensor CudaMoeExpertCache::execute_positions_plan_t1(
    const std::uint64_t plan_id,
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const CanonicalExpertPositionTileT1& missing_experts) {
  const c10::InferenceMode inference_guard;
  Impl::Plan& plan = impl_->require_plan(plan_id);
  if (!available()) {
    throw std::runtime_error("CUDA MXFP4 unavailable: " + impl_->reason);
  }
  if (prepared_rows.empty() ||
      prepared_rows.size() > kMoePositionTileMaxRows ||
      prepared_rows.front() == nullptr) {
    throw std::invalid_argument("CUDA MXFP4 planned position tile has invalid rows");
  }
  require_exact_cuda_geometry(prepared_rows.front()->geometry);
  if (plan.layer_index != prepared_rows.front()->layer_index) {
    throw std::invalid_argument(
        "CUDA expert residency plan belongs to a different layer");
  }

  std::vector<std::uint16_t> expected_missing;
  expected_missing.reserve(plan.canonical_experts.size());
  for (const std::uint16_t expert : plan.canonical_experts) {
    const bool hit = std::any_of(
        plan.hits.begin(), plan.hits.end(), [expert](const Impl::Entry& entry) {
          return entry.expert == expert;
        });
    if (!hit) {
      expected_missing.push_back(expert);
    }
  }
  const std::uint64_t expected_bytes =
      static_cast<std::uint64_t>(expected_missing.size()) *
      static_cast<std::uint64_t>(kCudaExpertSpan);
  if (missing_experts.layout != MoeExpertLayout::RawV1 ||
      missing_experts.expert_span_bytes !=
          static_cast<std::uint64_t>(kCudaExpertSpan) ||
      !std::equal(expected_missing.begin(), expected_missing.end(),
                  missing_experts.expert_ids.begin(),
                  missing_experts.expert_ids.end()) ||
      missing_experts.expert_ids.size() != expected_missing.size() ||
      expected_bytes > std::numeric_limits<std::size_t>::max() ||
      missing_experts.expert_major_bytes.size() !=
          static_cast<std::size_t>(expected_bytes)) {
    throw std::invalid_argument(
        "CUDA planned execution misses do not match its pinned residency snapshot");
  }

  std::vector<at::Tensor> input_rows;
  input_rows.reserve(prepared_rows.size());
  for (const PreparedMoeT1* prepared : prepared_rows) {
    if (prepared == nullptr || prepared->routed_input.device() != impl_->device ||
        prepared->routed_input.scalar_type() != at::kFloat ||
        !prepared->routed_input.is_contiguous() ||
        prepared->routed_input.sizes() != at::IntArrayRef({1, kCudaHidden}) ||
        prepared->layer_index != plan.layer_index) {
      throw std::invalid_argument(
          "CUDA planned position rows violate layer/device/fp32 contracts");
    }
    for (const std::uint16_t expert : prepared->route.expert_ids) {
      if (!std::binary_search(plan.canonical_experts.begin(),
                              plan.canonical_experts.end(), expert)) {
        throw std::invalid_argument(
            "CUDA residency plan is missing one of the ordered route experts");
      }
    }
    input_rows.push_back(prepared->routed_input);
  }
  for (const std::uint16_t expert : plan.canonical_experts) {
    const bool routed = std::any_of(
        prepared_rows.begin(), prepared_rows.end(),
        [expert](const PreparedMoeT1* prepared) {
          return std::find(prepared->route.expert_ids.begin(),
                           prepared->route.expert_ids.end(), expert) !=
              prepared->route.expert_ids.end();
        });
    if (!routed) {
      throw std::invalid_argument(
          "CUDA residency plan contains an expert outside the routed tile");
    }
  }

  at::Tensor output = impl_->execute_fail_closed([&] {
    const c10::cuda::CUDAGuard guard(impl_->device);
    const auto stream =
        c10::cuda::getCurrentCUDAStream(impl_->device.index());
    const at::Tensor inputs = at::cat(input_rows, 0).contiguous();
    const std::int64_t positions =
        static_cast<std::int64_t>(prepared_rows.size());
    const auto float_options =
        at::TensorOptions().dtype(at::kFloat).device(impl_->device);
    at::Tensor expert_outputs = at::empty(
        {positions * static_cast<std::int64_t>(kMoeRouteTopK), kCudaHidden},
        float_options);

    for (const std::uint16_t expert : plan.canonical_experts) {
      std::vector<std::int64_t> token_indices;
      std::vector<std::int64_t> route_indices;
      for (std::size_t row = 0; row < prepared_rows.size(); ++row) {
        for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
          if (prepared_rows[row]->route.expert_ids[edge] == expert) {
            token_indices.push_back(static_cast<std::int64_t>(row));
            route_indices.push_back(static_cast<std::int64_t>(
                row * kMoeRouteTopK + edge));
          }
        }
      }
      if (token_indices.empty()) {
        throw std::logic_error(
            "validated CUDA residency plan lost a routed expert");
      }
      const Impl::Resident resident = impl_->planned_resident(
          plan, expert, missing_experts.expert_ids,
          missing_experts.expert_major_bytes);
      resident.ready->wait(stream);
      const at::Tensor tokens = owned_cpu_tensor(
          token_indices,
          {static_cast<std::int64_t>(token_indices.size())}, at::kLong)
          .to(impl_->device, at::kLong, false, true);
      const at::Tensor routes = owned_cpu_tensor(
          route_indices,
          {static_cast<std::int64_t>(route_indices.size())}, at::kLong)
          .to(impl_->device, at::kLong, false, true);
      const at::Tensor expert_input =
          inputs.index_select(0, tokens).contiguous();
      const std::int64_t batches =
          static_cast<std::int64_t>(token_indices.size());
      at::Tensor gate =
          at::empty({batches, kCudaIntermediate}, float_options);
      at::Tensor up = at::empty_like(gate);
      at::Tensor batch_output =
          at::empty({batches, kCudaHidden}, float_options);
      for (const at::Tensor& tensor :
           {resident.bytes, expert_input, gate, up, batch_output}) {
        record_stream(tensor, stream);
      }
      const int status = k3_cuda_moe_launch(
          resident.bytes.const_data_ptr<std::uint8_t>(),
          expert_input.const_data_ptr<float>(), static_cast<int>(batches),
          gate.data_ptr<float>(), up.data_ptr<float>(),
          batch_output.data_ptr<float>(), stream_pointer(stream));
      require_launch("CUDA planned routed expert position batch", status);
      expert_outputs.index_copy_(0, routes, batch_output);
    }

    at::Tensor reduced =
        ordered_route_reduce(expert_outputs, prepared_rows, impl_->device);
    for (const at::Tensor& tensor : {inputs, expert_outputs, reduced}) {
      record_stream(tensor, stream);
    }
    return reduced;
  });
  impl_->plans.erase(plan_id);
  return output;
}

void CudaMoeExpertCache::cancel_plan(const std::uint64_t plan_id) noexcept {
  impl_->plans.erase(plan_id);
}

CudaMoeHostFallback CudaMoeExpertCache::materialize_plan_for_cpu(
    const std::uint64_t plan_id) {
  Impl::Plan& plan = impl_->require_plan(plan_id);
  return impl_->terminal_fail_closed([&] {
    const c10::cuda::CUDAGuard guard(impl_->device);
    const auto stream =
        c10::cuda::getCurrentCUDAStream(impl_->device.index());
    CudaMoeHostFallback fallback;
    fallback.canonical_experts = plan.canonical_experts;
    fallback.resident_experts.reserve(plan.hits.size());
    for (const Impl::Entry& hit : plan.hits) {
      // A plan can outlive the upload stream that created a hit. Make the D2H
      // copy explicitly consume that readiness event instead of assuming the
      // current fallback stream is the upload stream.
      hit.resident.ready->wait(stream);
      record_stream(hit.resident.bytes, stream);
      at::Tensor host = hit.resident.bytes.to(at::kCPU).contiguous();
      if (host.scalar_type() != at::kByte || host.dim() != 1 ||
          host.numel() != kCudaExpertSpan) {
        throw std::runtime_error(
            "CUDA expert fallback returned an invalid exact host span");
      }
      fallback.resident_experts.push_back(
          CudaMoeHostExpert{hit.expert, std::move(host)});
    }
    return fallback;
  });
}

void CudaMoeExpertCache::poison_external(const char* failure) noexcept {
  impl_->poison(failure, true);
}

at::Tensor CudaMoeExpertCache::execute_t1(
    const PreparedMoeT1& prepared,
    const CanonicalExpertBatchT1& experts) {
  const c10::InferenceMode inference_guard;
  if (!available()) {
    throw std::runtime_error("CUDA MXFP4 unavailable: " + impl_->reason);
  }
  require_exact_cuda_geometry(prepared.geometry);
  if (prepared.routed_input.device() != impl_->device ||
      prepared.routed_input.scalar_type() != at::kFloat ||
      !prepared.routed_input.is_contiguous() ||
      prepared.routed_input.sizes() != at::IntArrayRef({1, kCudaHidden})) {
    throw std::invalid_argument(
        "CUDA MXFP4 input violates its exact fp32 [1,3584] device contract");
  }
  return impl_->terminal_fail_closed([&] {
  const c10::cuda::CUDAGuard guard(impl_->device);
  const auto stream =
      c10::cuda::getCurrentCUDAStream(impl_->device.index());
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(impl_->device);
  at::Tensor gate = at::empty(
      {static_cast<std::int64_t>(kMoeRouteTopK), kCudaIntermediate},
      options);
  at::Tensor up = at::empty_like(gate);
  at::Tensor expert_outputs = at::empty(
      {static_cast<std::int64_t>(kMoeRouteTopK), kCudaHidden}, options);

  for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
    const std::uint16_t expert = prepared.route.expert_ids[edge];
    const std::uint8_t* source = impl_->source_for(
        expert, experts.expert_ids, experts.expert_major_bytes);
    const Impl::Resident resident = impl_->resident(
        prepared.layer_index, expert, source,
        static_cast<std::size_t>(kCudaExpertSpan));
    resident.ready->wait(stream);
    record_stream(resident.bytes, stream);
    record_stream(prepared.routed_input, stream);
    record_stream(gate, stream);
    record_stream(up, stream);
    record_stream(expert_outputs, stream);
    const int status = k3_cuda_moe_launch(
        resident.bytes.const_data_ptr<std::uint8_t>(),
        prepared.routed_input.const_data_ptr<float>(), 1,
        gate.data_ptr<float>() + edge * kCudaIntermediate,
        up.data_ptr<float>() + edge * kCudaIntermediate,
        expert_outputs.data_ptr<float>() + edge * kCudaHidden,
        stream_pointer(stream));
    require_launch("CUDA routed expert", status);
  }
  const std::array<const PreparedMoeT1*, 1> rows{&prepared};
  at::Tensor combined = ordered_route_reduce(expert_outputs, rows, impl_->device);
  record_stream(combined, stream);
  return combined;
  });
}

at::Tensor CudaMoeExpertCache::execute_positions_t1(
    const std::span<const PreparedMoeT1* const> prepared_rows,
    const CanonicalExpertPositionTileT1& experts) {
  const c10::InferenceMode inference_guard;
  if (!available()) {
    throw std::runtime_error("CUDA MXFP4 unavailable: " + impl_->reason);
  }
  if (prepared_rows.empty() ||
      prepared_rows.size() > kMoePositionTileMaxRows) {
    throw std::invalid_argument("CUDA MXFP4 position tile has invalid rows");
  }
  if (prepared_rows.front() == nullptr) {
    throw std::invalid_argument("CUDA MXFP4 position tile has a null first row");
  }
  require_exact_cuda_geometry(prepared_rows.front()->geometry);
  std::vector<at::Tensor> input_rows;
  input_rows.reserve(prepared_rows.size());
  for (const PreparedMoeT1* prepared : prepared_rows) {
    if (prepared == nullptr || prepared->routed_input.device() != impl_->device ||
        prepared->routed_input.scalar_type() != at::kFloat ||
        !prepared->routed_input.is_contiguous() ||
        prepared->routed_input.sizes() !=
            at::IntArrayRef({1, kCudaHidden}) ||
        prepared->layer_index != prepared_rows.front()->layer_index) {
      throw std::invalid_argument(
          "CUDA MXFP4 position rows violate layer/device/fp32 contracts");
    }
    input_rows.push_back(prepared->routed_input);
  }

  return impl_->terminal_fail_closed([&] {
  const c10::cuda::CUDAGuard guard(impl_->device);
  const auto stream =
      c10::cuda::getCurrentCUDAStream(impl_->device.index());
  const at::Tensor inputs = at::cat(input_rows, 0).contiguous();
  const std::int64_t positions =
      static_cast<std::int64_t>(prepared_rows.size());
  const auto float_options =
      at::TensorOptions().dtype(at::kFloat).device(impl_->device);
  at::Tensor expert_outputs = at::empty(
      {positions * static_cast<std::int64_t>(kMoeRouteTopK), kCudaHidden},
      float_options);

  for (std::size_t canonical = 0;
       canonical < experts.expert_ids.size(); ++canonical) {
    const std::uint16_t expert = experts.expert_ids[canonical];
    std::vector<std::int64_t> token_indices;
    std::vector<std::int64_t> route_indices;
    for (std::size_t row = 0; row < prepared_rows.size(); ++row) {
      for (std::size_t edge = 0; edge < kMoeRouteTopK; ++edge) {
        if (prepared_rows[row]->route.expert_ids[edge] == expert) {
          token_indices.push_back(static_cast<std::int64_t>(row));
          route_indices.push_back(static_cast<std::int64_t>(
              row * kMoeRouteTopK + edge));
        }
      }
    }
    if (token_indices.empty()) {
      continue;
    }
    const std::uint8_t* source = experts.expert_major_bytes.data() +
        canonical * static_cast<std::size_t>(kCudaExpertSpan);
    const Impl::Resident resident = impl_->resident(
        prepared_rows.front()->layer_index, expert, source,
        static_cast<std::size_t>(kCudaExpertSpan));
    resident.ready->wait(stream);
    const at::Tensor tokens = owned_cpu_tensor(
        token_indices,
        {static_cast<std::int64_t>(token_indices.size())}, at::kLong)
        .to(impl_->device, at::kLong, false, true);
    const at::Tensor routes = owned_cpu_tensor(
        route_indices,
        {static_cast<std::int64_t>(route_indices.size())}, at::kLong)
        .to(impl_->device, at::kLong, false, true);
    const at::Tensor expert_input = inputs.index_select(0, tokens).contiguous();
    const std::int64_t batches =
        static_cast<std::int64_t>(token_indices.size());
    at::Tensor gate = at::empty({batches, kCudaIntermediate}, float_options);
    at::Tensor up = at::empty_like(gate);
    at::Tensor batch_output =
        at::empty({batches, kCudaHidden}, float_options);
    for (const at::Tensor& tensor :
         {resident.bytes, expert_input, gate, up, batch_output}) {
      record_stream(tensor, stream);
    }
    const int status = k3_cuda_moe_launch(
        resident.bytes.const_data_ptr<std::uint8_t>(),
        expert_input.const_data_ptr<float>(), static_cast<int>(batches),
        gate.data_ptr<float>(), up.data_ptr<float>(),
        batch_output.data_ptr<float>(), stream_pointer(stream));
    require_launch("CUDA routed expert position batch", status);
    expert_outputs.index_copy_(0, routes, batch_output);
  }
  at::Tensor output =
      ordered_route_reduce(expert_outputs, prepared_rows, impl_->device);
  for (const at::Tensor& tensor : {inputs, expert_outputs, output}) {
    record_stream(tensor, stream);
  }
  return output.contiguous();
  });
}

bool cuda_moe_compiled() noexcept { return true; }

#endif

#if defined(DELTAFIN_HAVE_CUDA_PROVIDER_V1)

CudaProviderMemorySnapshot cuda_provider_memory_snapshot(
    const at::Device& device, const bool trim_unused) {
  if (!device.is_cuda() || !device.has_index()) {
    throw std::invalid_argument(
        "CUDA memory snapshot requires an indexed CUDA device");
  }
  const c10::cuda::CUDAGuard guard(device);
  if (trim_unused) {
    // The caller proves that its provider session is between transactions.
    // Synchronizing first promotes stream-deferred frees to allocator-owned
    // inactive blocks; emptyCache then releases only those unused blocks.
    c10::cuda::device_synchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
  const auto device_index = device.index();
  const auto stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
  const auto memory =
      c10::cuda::CUDACachingAllocator::get()->getMemoryInfo(device_index);
  const auto nonnegative = [](const std::int64_t value) -> std::uint64_t {
    return value <= 0 ? 0 : static_cast<std::uint64_t>(value);
  };
  constexpr auto aggregate = static_cast<std::size_t>(
      c10::CachingAllocator::StatType::AGGREGATE);
  CudaProviderMemorySnapshot result;
  result.active_valid = true;
  result.reserved_valid = true;
  result.total_valid = true;
  result.available_valid = true;
  result.cache_trimmed = trim_unused;
  result.active_bytes =
      nonnegative(stats.allocated_bytes[aggregate].current);
  result.reserved_bytes =
      nonnegative(stats.reserved_bytes[aggregate].current);
  result.available_bytes = static_cast<std::uint64_t>(memory.first);
  result.total_bytes = static_cast<std::uint64_t>(memory.second);
  if (result.available_bytes > result.total_bytes) {
    throw std::runtime_error(
        "CUDA allocator reported free memory larger than total memory");
  }
  return result;
}

#else

CudaProviderMemorySnapshot cuda_provider_memory_snapshot(
    const at::Device& device, const bool trim_unused) {
  if (!device.is_cuda()) {
    throw std::invalid_argument(
        "CUDA memory snapshot requires a CUDA device");
  }
  if (trim_unused) {
    throw std::runtime_error(
        "CUDA allocator trimming is unavailable in this provider build");
  }
  return {};
}

#endif

}  // namespace deltafin::provider_internal
