#include "provider_spine_bf16_cuda.h"

#include <c10/core/InferenceMode.h>

#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(DELTAFIN_HAVE_CUDA_SPINE_BF16_V1) && \
    !defined(DELTAFIN_HAVE_CUDA_PROVIDER_V1)
#error "CUDA RAW_BF16 requires the qualified CUDA LibTorch provider"
#endif

#if defined(DELTAFIN_HAVE_CUDA_SPINE_BF16_V1)
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#if defined(_WIN32)
#include <malloc.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(DELTAFIN_HAVE_CUDA_PROVIDER_V1)
#include <ATen/cuda/CUDAContextLight.h>
// Force an import of torch_cuda.dll: its ATen CUDA kernels register from its
// static initializers, but the provider otherwise reaches CUDA only through
// `c10::cuda` (c10_cuda). External linkage keeps the relocation in the object.
decltype(&at::cuda::getCurrentDeviceProperties) k3_torch_cuda_anchor_spine =
    &at::cuda::getCurrentDeviceProperties;
#endif

extern "C" {
std::uint32_t k3_cuda_spine_bf16_abi_version(void);
const char* k3_cuda_spine_bf16_last_error(void);
int k3_cuda_spine_bf16_available(int device, int* compute_major,
                                 int* compute_minor);
int k3_cuda_spine_bf16_launch_v1(const std::uint16_t* weights,
                                const float* activation, float* output,
                                int rows, int columns, int positions,
                                void* stream_pointer);
int k3_cuda_spine_bf16_decode_bits(const std::uint16_t* input,
                                   std::uint32_t* output, int count,
                                   void* stream_pointer);
}
#endif

namespace deltafin::provider_internal {

float cuda_spine_bf16_reference_decode(const std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

namespace {

void validate_host_slab_descriptor(const CudaSpineBf16HostSlab& slab) {
  if (slab.allocation_base == nullptr || slab.logical_slab_bytes == 0 ||
      (slab.logical_slab_bytes & 1U) != 0 ||
      slab.logical_slab_bytes > slab.allocation_bytes ||
      (reinterpret_cast<std::uintptr_t>(slab.allocation_base) & 1U) != 0 ||
      slab.logical_slab_bytes > static_cast<std::size_t>(
                                    std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(
        "CUDA RAW_BF16 host slab has invalid base/logical/allocation bounds");
  }
}

void validate_matrix_view_descriptor(
    const CudaSpineBf16MatrixView& matrix,
    const std::size_t logical_slab_bytes) {
  if (matrix.rows <= 0 || matrix.columns <= 0 || matrix.rows > INT_MAX ||
      matrix.columns > INT_MAX || (matrix.matrix_byte_offset & 1U) != 0) {
    throw std::invalid_argument(
        "CUDA RAW_BF16 matrix view has invalid bounds or alignment");
  }
  const auto rows = static_cast<std::uint64_t>(matrix.rows);
  const auto columns = static_cast<std::uint64_t>(matrix.columns);
  if (columns > std::numeric_limits<std::uint64_t>::max() / rows ||
      rows * columns >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
    throw std::invalid_argument("CUDA RAW_BF16 matrix view byte span overflows");
  }
  const std::size_t expected = static_cast<std::size_t>(
      rows * columns * sizeof(std::uint16_t));
  if (matrix.logical_bytes != expected ||
      matrix.matrix_byte_offset > logical_slab_bytes ||
      matrix.logical_bytes > logical_slab_bytes - matrix.matrix_byte_offset) {
    throw std::invalid_argument(
        "CUDA RAW_BF16 matrix view escapes its prepared slab");
  }
}

}  // namespace

void validate_cuda_spine_bf16_host_view(
    const CudaSpineBf16HostSlab& slab,
    const CudaSpineBf16MatrixView& matrix) {
  validate_host_slab_descriptor(slab);
  validate_matrix_view_descriptor(matrix, slab.logical_slab_bytes);
}

bool cuda_spine_bf16_device_storage_dtype_supported(
    const at::ScalarType scalar_type) noexcept {
  return scalar_type == at::kByte || scalar_type == at::kBFloat16 ||
      scalar_type == at::kUInt16 || scalar_type == at::kShort;
}

void CudaSpineBf16LifetimeState::note_submission() {
  if (phase_ != Phase::Open) {
    throw std::logic_error(
        "CUDA RAW_BF16 submission requires an open prepared layer");
  }
  if (submissions_ == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("CUDA RAW_BF16 submission count overflowed");
  }
  ++submissions_;
}

void CudaSpineBf16LifetimeState::seal() {
  if (phase_ != Phase::Open || submissions_ == 0) {
    throw std::logic_error(
        "CUDA RAW_BF16 seal is consume-once and needs submitted work");
  }
  phase_ = Phase::Sealed;
}

void CudaSpineBf16LifetimeState::require_reclaim_query() const {
  if (phase_ != Phase::Sealed) {
    throw std::logic_error(
        "CUDA RAW_BF16 reclaim requires one sealed prepared layer");
  }
}

void CudaSpineBf16LifetimeState::complete_reclaim() {
  require_reclaim_query();
  phase_ = Phase::Reclaimed;
}

void CudaSpineBf16LifetimeState::complete_abort() {
  if (phase_ == Phase::Reclaimed) {
    throw std::logic_error(
        "CUDA RAW_BF16 prepared layer is already reclaimed");
  }
  phase_ = Phase::Reclaimed;
}

void CudaSpineBf16LifetimeState::poison() noexcept {
  if (phase_ != Phase::Reclaimed) {
    phase_ = Phase::Poisoned;
  }
}

std::size_t CudaSpineBf16LifetimeState::submissions() const noexcept {
  return submissions_;
}
bool CudaSpineBf16LifetimeState::open() const noexcept {
  return phase_ == Phase::Open;
}
bool CudaSpineBf16LifetimeState::sealed() const noexcept {
  return phase_ == Phase::Sealed;
}
bool CudaSpineBf16LifetimeState::reclaimed() const noexcept {
  return phase_ == Phase::Reclaimed;
}

#if !defined(DELTAFIN_HAVE_CUDA_SPINE_BF16_V1)

struct CudaSpineBf16PreparedLayer::Impl {};

CudaSpineBf16PreparedLayer::CudaSpineBf16PreparedLayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
CudaSpineBf16PreparedLayer::~CudaSpineBf16PreparedLayer() = default;
CudaSpineBf16PreparedLayer::CudaSpineBf16PreparedLayer(
    CudaSpineBf16PreparedLayer&&) noexcept = default;
CudaSpineBf16PreparedLayer& CudaSpineBf16PreparedLayer::operator=(
    CudaSpineBf16PreparedLayer&&) noexcept = default;
CudaSpineBf16SourceKind CudaSpineBf16PreparedLayer::source_kind() const
    noexcept {
  return CudaSpineBf16SourceKind::DetachedStaged;
}
std::size_t CudaSpineBf16PreparedLayer::submissions() const noexcept {
  return 0;
}
bool CudaSpineBf16PreparedLayer::sealed() const noexcept { return false; }
bool CudaSpineBf16PreparedLayer::reclaimed() const noexcept { return false; }
at::Tensor CudaSpineBf16PreparedLayer::submit_t1(
    const CudaSpineBf16MatrixView&, const at::Tensor&) {
  throw std::runtime_error("CUDA RAW_BF16 was not compiled into this provider");
}
at::Tensor CudaSpineBf16PreparedLayer::submit(
    const CudaSpineBf16MatrixView&, const at::Tensor&) {
  throw std::runtime_error("CUDA RAW_BF16 was not compiled into this provider");
}
void CudaSpineBf16PreparedLayer::seal() {
  throw std::runtime_error("CUDA RAW_BF16 was not compiled into this provider");
}
bool CudaSpineBf16PreparedLayer::try_reclaim() {
  throw std::runtime_error("CUDA RAW_BF16 was not compiled into this provider");
}
void CudaSpineBf16PreparedLayer::abort_and_reclaim() {
  throw std::runtime_error("CUDA RAW_BF16 was not compiled into this provider");
}

struct CudaSpineBf16Projector::Impl {
  explicit Impl(const at::Device& selected) : device(selected) {
    capability.compiled = false;
    capability.device_index =
        selected.has_index() ? static_cast<std::int32_t>(selected.index()) : -1;
    capability.detail = "CUDA RAW_BF16 was not compiled into this provider";
  }
  at::Device device;
  CudaSpineBf16Capability capability;
};

CudaSpineBf16Projector::CudaSpineBf16Projector(const at::Device& device)
    : impl_(std::make_unique<Impl>(device)) {}
CudaSpineBf16Projector::~CudaSpineBf16Projector() = default;
const CudaSpineBf16Capability& CudaSpineBf16Projector::capability() {
  return impl_->capability;
}
std::unique_ptr<CudaSpineBf16PreparedLayer>
CudaSpineBf16Projector::prepare_host_layer(
    const CudaSpineBf16HostSlab&, const CudaSpineBf16HostPolicy) {
  throw std::runtime_error(impl_->capability.detail);
}
std::unique_ptr<CudaSpineBf16PreparedLayer>
CudaSpineBf16Projector::prepare_direct_host_layer_for_benchmark(
    const CudaSpineBf16HostSlab&) {
  throw std::runtime_error(impl_->capability.detail);
}
std::unique_ptr<CudaSpineBf16PreparedLayer>
CudaSpineBf16Projector::prepare_device_layer(
    const CudaSpineBf16DeviceSlab&) {
  throw std::runtime_error(impl_->capability.detail);
}
bool cuda_spine_bf16_compiled() noexcept { return false; }

#else

namespace {

constexpr std::uint32_t kCudaSpineBf16Abi = 1;
constexpr std::int64_t kDecodeCanaryCount = 65'536;
constexpr std::int64_t kMaximumPositions = 64;
constexpr std::size_t kMaximumOutstandingDeviceStreams = 64;

std::string native_error(const char* operation, const int status) {
  const char* detail = k3_cuda_spine_bf16_last_error();
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
  if (tensor.defined() && tensor.device().is_cuda()) {
    c10::cuda::CUDACachingAllocator::recordStream(
        tensor.storage().data_ptr(), stream);
  }
}

std::size_t system_page_size() {
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  const std::size_t page = static_cast<std::size_t>(info.dwPageSize);
#else
  const long queried = sysconf(_SC_PAGESIZE);
  if (queried <= 0) {
    throw std::runtime_error(
        "CUDA ATS could not query the host system page size");
  }
  const std::size_t page = static_cast<std::size_t>(queried);
#endif
  if (!std::has_single_bit(page)) {
    throw std::runtime_error(
        "CUDA ATS host system page size is not a power of two");
  }
  return page;
}

/// Page-aligned host allocation: `_aligned_malloc` on Windows (whose matching
/// free is `_aligned_free`), `posix_memalign` elsewhere.
void* page_aligned_alloc(const std::size_t alignment, const std::size_t bytes) {
#if defined(_WIN32)
  return _aligned_malloc(bytes, alignment);
#else
  void* allocation = nullptr;
  if (posix_memalign(&allocation, alignment, bytes) != 0) {
    return nullptr;
  }
  return allocation;
#endif
}

void page_aligned_free(void* pointer) {
#if defined(_WIN32)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

void validate_activation(const at::Tensor& activation,
                         const at::Device& device,
                         const std::int64_t columns) {
  if (!activation.defined() || !activation.device().is_cuda() ||
      activation.device() != device || activation.scalar_type() != at::kFloat ||
      !activation.is_contiguous() || activation.dim() != 2 ||
      activation.size(0) < 1 ||
      activation.size(0) > kMaximumPositions ||
      activation.size(1) != columns ||
      activation.numel() != activation.size(0) * columns) {
    throw std::invalid_argument(
        "CUDA RAW_BF16 activation must be contiguous CUDA float32 "
        "[1..64, columns] on the selected device");
  }
}

enum class HostPointerKind { Ordinary, Registered };

HostPointerKind classify_host_pointer(const at::Device& device,
                                      const void* pointer) {
  const c10::cuda::CUDAGuard guard(device);
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status == cudaErrorInvalidValue) {
    static_cast<void>(cudaGetLastError());
    return HostPointerKind::Ordinary;
  }
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaPointerGetAttributes failed: ") +
        cudaGetErrorString(status));
  }
  if (attributes.type == cudaMemoryTypeDevice ||
      attributes.type == cudaMemoryTypeManaged) {
    throw std::invalid_argument(
        "CUDA RAW_BF16 source must be host reader storage, not device or "
        "managed memory");
  }
#if CUDART_VERSION >= 11000
  if (attributes.type == cudaMemoryTypeUnregistered) {
    return HostPointerKind::Ordinary;
  }
#endif
  if (attributes.type == cudaMemoryTypeHost) {
    return HostPointerKind::Registered;
  }
  throw std::invalid_argument(
      "CUDA RAW_BF16 source has an unsupported CUDA pointer classification");
}

bool direct_host_ats_attributes(const int device) {
  const c10::cuda::CUDAGuard guard(
      at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device)));
  int pageable = 0;
  int host_page_tables = 0;
  int unified_addressing = 0;
  for (const auto [attribute, destination] : {
           std::pair{cudaDevAttrPageableMemoryAccess, &pageable},
           std::pair{cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                     &host_page_tables},
           std::pair{cudaDevAttrUnifiedAddressing, &unified_addressing},
       }) {
    const cudaError_t status =
        cudaDeviceGetAttribute(destination, attribute, device);
    if (status == cudaErrorInvalidValue) {
      static_cast<void>(cudaGetLastError());
      return false;
    }
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("CUDA ATS attribute query failed: ") +
          cudaGetErrorString(status));
    }
  }
  return pageable != 0 && host_page_tables != 0 && unified_addressing != 0;
}

void run_decode_canary(const at::Device& device) {
  const c10::cuda::CUDAGuard guard(device);
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  at::Tensor host = at::empty(
      {kDecodeCanaryCount},
      at::TensorOptions().dtype(at::kShort).device(at::kCPU));
  auto* host_bits = reinterpret_cast<std::uint16_t*>(
      host.data_ptr<std::int16_t>());
  for (std::int64_t index = 0; index < kDecodeCanaryCount; ++index) {
    host_bits[index] = static_cast<std::uint16_t>(index);
  }
  at::Tensor input = host.to(device);
  at::Tensor output = at::empty(
      {kDecodeCanaryCount},
      at::TensorOptions().dtype(at::kInt).device(device));
  require_launch(
      "CUDA RAW_BF16 decode canary",
      k3_cuda_spine_bf16_decode_bits(
          reinterpret_cast<const std::uint16_t*>(
              input.const_data_ptr<std::int16_t>()),
          reinterpret_cast<std::uint32_t*>(output.data_ptr<std::int32_t>()),
          static_cast<int>(kDecodeCanaryCount), stream_pointer(stream)));
  const at::Tensor checked = output.to(at::kCPU);
  const auto* found = reinterpret_cast<const std::uint32_t*>(
      checked.const_data_ptr<std::int32_t>());
  for (std::uint32_t bits = 0; bits < kDecodeCanaryCount; ++bits) {
    if (found[bits] != (bits << 16)) {
      throw std::runtime_error(
          "CUDA RAW_BF16 all-encoding decode canary failed");
    }
  }
}

void run_finite_gemv_canary(const at::Device& device) {
  constexpr std::int64_t rows = 11;
  constexpr std::int64_t columns = 259;
  constexpr std::int64_t positions = 3;
  const c10::cuda::CUDAGuard guard(device);
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

  at::Tensor host_weights = at::empty(
      {rows, columns},
      at::TensorOptions().dtype(at::kShort).device(at::kCPU));
  auto* weight_bits = reinterpret_cast<std::uint16_t*>(
      host_weights.data_ptr<std::int16_t>());
  for (std::int64_t index = 0; index < rows * columns; ++index) {
    const std::int32_t centered =
        static_cast<std::int32_t>((index * 29 + 7) % 101) - 50;
    const float value = static_cast<float>(centered) / 32.0F;
    weight_bits[index] = static_cast<std::uint16_t>(
        std::bit_cast<std::uint32_t>(value) >> 16);
  }
  at::Tensor host_activation = at::empty(
      {positions, columns},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  float* activation_values = host_activation.data_ptr<float>();
  for (std::int64_t position = 0; position < positions; ++position) {
    for (std::int64_t column = 0; column < columns; ++column) {
      const std::int32_t centered = static_cast<std::int32_t>(
          (position * 23 + column * 17 + 3) % 67) - 33;
      activation_values[position * columns + column] =
          static_cast<float>(centered) / 64.0F;
    }
  }

  at::Tensor device_weights = host_weights.to(device);
  at::Tensor activation = host_activation.to(device);
  at::Tensor output = at::empty(
      {positions, rows},
      at::TensorOptions().dtype(at::kFloat).device(device));
  cudaEvent_t completion = nullptr;
  cudaError_t status =
      cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA finite GEMV canary event creation failed: ") +
        cudaGetErrorString(status));
  }
  struct EventGuard {
    cudaEvent_t event;
    ~EventGuard() { static_cast<void>(cudaEventDestroy(event)); }
  } event_guard{completion};
  require_launch(
      "CUDA RAW_BF16 finite GEMV canary",
      k3_cuda_spine_bf16_launch_v1(
          reinterpret_cast<const std::uint16_t*>(
              device_weights.const_data_ptr<std::int16_t>()),
          activation.const_data_ptr<float>(), output.data_ptr<float>(),
          static_cast<int>(rows), static_cast<int>(columns),
          static_cast<int>(positions), stream_pointer(stream)));
  status = cudaEventRecord(completion, stream.stream());
  if (status != cudaSuccess) {
    const cudaError_t drained = cudaStreamSynchronize(stream.stream());
    throw std::runtime_error(
        std::string("CUDA finite GEMV canary event record failed: ") +
        cudaGetErrorString(status) + "; stream drain: " +
        cudaGetErrorString(drained));
  }
  status = cudaEventSynchronize(completion);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA finite GEMV canary execution failed: ") +
        cudaGetErrorString(status));
  }
  const at::Tensor checked = output.to(at::kCPU).contiguous();
  const float* found = checked.const_data_ptr<float>();

  // Each result has `columns` FP32 FMAs plus at most eight fixed warp-tree
  // additions. Eight times Higham's gamma bound is deliberately conservative
  // for comparison with a float64 dot-product oracle.
  constexpr double epsilon =
      static_cast<double>(std::numeric_limits<float>::epsilon());
  constexpr double operations = static_cast<double>(columns + 8);
  constexpr double gamma =
      (operations * epsilon) / (1.0 - operations * epsilon);
  for (std::int64_t position = 0; position < positions; ++position) {
    std::int64_t reference_argmax = 0;
    std::int64_t candidate_argmax = 0;
    double best_reference = -std::numeric_limits<double>::infinity();
    float best_candidate = -std::numeric_limits<float>::infinity();
    for (std::int64_t row = 0; row < rows; ++row) {
      double reference = 0.0;
      double absolute_sum = 0.0;
      for (std::int64_t column = 0; column < columns; ++column) {
        const double weight = static_cast<double>(
            cuda_spine_bf16_reference_decode(
                weight_bits[row * columns + column]));
        const double value = static_cast<double>(
            activation_values[position * columns + column]);
        reference += weight * value;
        absolute_sum += std::abs(weight * value);
      }
      const float candidate = found[position * rows + row];
      const double bound = 8.0 * gamma * absolute_sum + 1.0e-6;
      if (!std::isfinite(candidate) ||
          std::abs(static_cast<double>(candidate) - reference) > bound) {
        throw std::runtime_error(
            "CUDA RAW_BF16 finite GEMV canary exceeded its documented "
            "FP32 reduction bound");
      }
      if (reference > best_reference) {
        best_reference = reference;
        reference_argmax = row;
      }
      if (candidate > best_candidate) {
        best_candidate = candidate;
        candidate_argmax = row;
      }
    }
    if (candidate_argmax != reference_argmax) {
      throw std::runtime_error(
          "CUDA RAW_BF16 finite GEMV canary changed argmax");
    }
  }
}

bool run_direct_host_canary(const at::Device& device) {
  const std::size_t page = system_page_size();
  void* allocation = page_aligned_alloc(page, page);
  if (allocation == nullptr) {
    throw std::runtime_error(
        "CUDA ATS canary could not allocate ordinary page-aligned storage");
  }
  struct FreeGuard {
    void operator()(void* pointer) const noexcept { page_aligned_free(pointer); }
  };
  std::unique_ptr<void, FreeGuard> allocation_guard(allocation);
  auto* weights = static_cast<std::uint16_t*>(allocation);
  constexpr std::uint16_t canary_weights[8]{
      0x3f80, 0x4000, 0xbf80, 0x3f00,
      0x4040, 0xc000, 0x3e80, 0x4080};
  std::memcpy(weights, canary_weights, sizeof(canary_weights));
  if (classify_host_pointer(device, weights) != HostPointerKind::Ordinary) {
    return false;
  }

  const c10::cuda::CUDAGuard guard(device);
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  const std::array<float, 4> activation_values{0.25F, -0.5F, 0.75F, 1.25F};
  const at::Tensor activation_host = at::from_blob(
      const_cast<float*>(activation_values.data()), {4},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const at::Tensor activation = activation_host.clone().to(device);
  at::Tensor output = at::empty(
      {2}, at::TensorOptions().dtype(at::kFloat).device(device));
  cudaEvent_t completion = nullptr;
  cudaError_t status =
      cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA ATS canary event creation failed: ") +
        cudaGetErrorString(status));
  }
  struct EventGuard {
    cudaEvent_t event;
    ~EventGuard() { static_cast<void>(cudaEventDestroy(event)); }
  } event_guard{completion};

  require_launch(
      "CUDA ordinary-pageable-host ATS canary",
      k3_cuda_spine_bf16_launch_v1(weights,
                                  activation.const_data_ptr<float>(),
                                  output.data_ptr<float>(), 2, 4, 1,
                                  stream_pointer(stream)));
  status = cudaEventRecord(completion, stream.stream());
  if (status != cudaSuccess) {
    const cudaError_t drained = cudaStreamSynchronize(stream.stream());
    if (drained != cudaSuccess) {
      // The ordinary host page may still be device-visible. Leak this one
      // bounded qualification page instead of risking use-after-free.
      static_cast<void>(allocation_guard.release());
    }
    throw std::runtime_error(
        std::string("CUDA ATS canary event record failed: ") +
        cudaGetErrorString(status) + "; stream drain: " +
        cudaGetErrorString(drained));
  }
  status = cudaEventSynchronize(completion);
  if (status != cudaSuccess) {
    static_cast<void>(allocation_guard.release());
    throw std::runtime_error(
        std::string("CUDA ATS canary execution failed: ") +
        cudaGetErrorString(status));
  }
  const at::Tensor checked = output.to(at::kCPU);
  const float* values = checked.const_data_ptr<float>();
  constexpr float expected0 = -0.875F;
  constexpr float expected1 = 6.9375F;
  if (std::bit_cast<std::uint32_t>(values[0]) !=
          std::bit_cast<std::uint32_t>(expected0) ||
      std::bit_cast<std::uint32_t>(values[1]) !=
          std::bit_cast<std::uint32_t>(expected1)) {
    throw std::runtime_error(
        "CUDA ordinary-pageable-host ATS canary returned the wrong result");
  }
  return true;
}

std::size_t device_tensor_bytes(const at::Tensor& tensor) {
  if (tensor.numel() < 0 || tensor.element_size() == 0 ||
      static_cast<std::uint64_t>(tensor.numel()) >
          std::numeric_limits<std::size_t>::max() / tensor.element_size()) {
    throw std::invalid_argument("CUDA BF16 device slab byte span overflows");
  }
  return static_cast<std::size_t>(tensor.numel()) * tensor.element_size();
}

}  // namespace

struct CudaSpineBf16PreparedLayer::Impl {
  struct StreamCompletion {
    c10::cuda::CUDAStream stream;
    cudaEvent_t event = nullptr;
    bool recorded = false;
  };

  Impl(const at::Device& selected,
       const c10::cuda::CUDAStream selected_stream,
       const CudaSpineBf16SourceKind selected_kind,
       const std::size_t selected_logical_bytes)
      : device(selected),
        stream(selected_stream),
        kind(selected_kind),
        logical_slab_bytes(selected_logical_bytes) {
    const c10::cuda::CUDAGuard guard(device);
    if (kind != CudaSpineBf16SourceKind::DetachedDeviceOwned) {
      const cudaError_t status =
          cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
      if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaEventCreateWithFlags failed: ") +
            cudaGetErrorString(status));
      }
    }
  }

  ~Impl() {
    try {
      const c10::cuda::CUDAGuard guard(device);
      if (completion != nullptr) {
        static_cast<void>(cudaEventDestroy(completion));
      }
      for (const StreamCompletion& item : device_completions) {
        static_cast<void>(cudaEventDestroy(item.event));
      }
    } catch (...) {
      // Teardown cannot safely switch devices. CUDA context destruction
      // eventually reclaims the events; no source/resource is released here.
    }
  }

  void require_captured_stream() const {
    const c10::cuda::CUDAGuard guard(device);
    const auto current = c10::cuda::getCurrentCUDAStream(device.index());
    if (current.stream() != stream.stream()) {
      throw std::logic_error(
          "CUDA RAW_BF16 submission crossed its captured CUDA stream");
    }
  }

  c10::cuda::CUDAStream submission_stream() const {
    const c10::cuda::CUDAGuard guard(device);
    const auto current = c10::cuda::getCurrentCUDAStream(device.index());
    if (kind != CudaSpineBf16SourceKind::DetachedDeviceOwned &&
        current.stream() != stream.stream()) {
      throw std::logic_error(
          "CUDA RAW_BF16 submission crossed its captured CUDA stream");
    }
    return current;
  }

  StreamCompletion* prepare_device_completion(
      const c10::cuda::CUDAStream& selected_stream) {
    if (kind != CudaSpineBf16SourceKind::DetachedDeviceOwned) {
      return nullptr;
    }
    for (StreamCompletion& item : device_completions) {
      if (item.stream.stream() == selected_stream.stream()) {
        return &item;
      }
    }
    for (auto item = device_completions.begin();
         item != device_completions.end();) {
      if (!item->recorded) {
        const cudaError_t destroyed = cudaEventDestroy(item->event);
        if (destroyed != cudaSuccess) {
          completion_untrustworthy = true;
          throw std::runtime_error(
              std::string("CUDA RAW_BF16 unused completion destroy failed: ") +
              cudaGetErrorString(destroyed));
        }
        item = device_completions.erase(item);
        continue;
      }
      const cudaError_t status = cudaEventQuery(item->event);
      if (status == cudaSuccess) {
        const cudaError_t destroyed = cudaEventDestroy(item->event);
        if (destroyed != cudaSuccess) {
          completion_untrustworthy = true;
          throw std::runtime_error(
              std::string("CUDA RAW_BF16 completion destroy failed: ") +
              cudaGetErrorString(destroyed));
        }
        item = device_completions.erase(item);
        continue;
      }
      if (status != cudaErrorNotReady) {
        completion_untrustworthy = true;
        throw std::runtime_error(
            std::string("CUDA RAW_BF16 completion query failed: ") +
            cudaGetErrorString(status));
      }
      ++item;
    }
    if (device_completions.size() >=
        kMaximumOutstandingDeviceStreams) {
      throw std::runtime_error(
          "CUDA RAW_BF16 reached its bounded active-stream limit");
    }
    cudaEvent_t event = nullptr;
    const cudaError_t status =
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("CUDA RAW_BF16 stream event creation failed: ") +
          cudaGetErrorString(status));
    }
    struct PendingEvent final {
      cudaEvent_t value = nullptr;
      ~PendingEvent() {
        if (value != nullptr) {
          static_cast<void>(cudaEventDestroy(value));
        }
      }
      void release() noexcept { value = nullptr; }
    } pending{event};
    device_completions.push_back(
        StreamCompletion{selected_stream, event, false});
    pending.release();
    return &device_completions.back();
  }

  const std::byte* storage_base() const {
    if (kind == CudaSpineBf16SourceKind::BorrowedDirectHostAts) {
      return borrowed_host_base;
    }
    return static_cast<const std::byte*>(device_storage.const_data_ptr());
  }

  void record_latest_completion() {
    const c10::cuda::CUDAGuard guard(device);
    require_captured_stream();
    const cudaError_t status = cudaEventRecord(completion, stream.stream());
    if (status != cudaSuccess) {
      const cudaError_t drained = cudaStreamSynchronize(stream.stream());
      if (drained != cudaSuccess) {
        completion_untrustworthy = true;
        lifetime.poison();
      }
      throw std::runtime_error(
          std::string("CUDA RAW_BF16 completion record failed: ") +
          cudaGetErrorString(status) + "; stream drain: " +
          cudaGetErrorString(drained));
    }
    completion_recorded = true;
  }

  void record_device_completion(
      const c10::cuda::CUDAStream& selected_stream,
      StreamCompletion& selected_completion) {
    const c10::cuda::CUDAGuard guard(device);
    const cudaError_t status = cudaEventRecord(
        selected_completion.event, selected_stream.stream());
    if (status != cudaSuccess) {
      const cudaError_t drained =
          cudaStreamSynchronize(selected_stream.stream());
      if (drained != cudaSuccess) {
        completion_untrustworthy = true;
        lifetime.poison();
      }
      throw std::runtime_error(
          std::string("CUDA RAW_BF16 device completion record failed: ") +
          cudaGetErrorString(status) + "; stream drain: " +
          cudaGetErrorString(drained));
    }
    selected_completion.recorded = true;
  }

  void release_resources() noexcept {
    pinned_stage.reset();
    device_storage.reset();
  }

  at::Device device;
  c10::cuda::CUDAStream stream;
  CudaSpineBf16SourceKind kind;
  std::size_t logical_slab_bytes = 0;
  const std::byte* borrowed_host_base = nullptr;
  CudaSpineBf16LifetimeState lifetime;
  cudaEvent_t completion = nullptr;
  bool completion_recorded = false;
  bool completion_untrustworthy = false;
  at::Tensor pinned_stage;
  at::Tensor device_storage;
  std::vector<StreamCompletion> device_completions;
  std::mutex call_mutex;
};

CudaSpineBf16PreparedLayer::CudaSpineBf16PreparedLayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

CudaSpineBf16PreparedLayer::~CudaSpineBf16PreparedLayer() {
  if (impl_ == nullptr || impl_->lifetime.reclaimed()) {
    return;
  }
  try {
    abort_and_reclaim();
  } catch (...) {
    // Unknown completion means both the provider-owned storage and a possible
    // Rust reader lease must remain live. Leak this bounded prepared layer
    // instead of risking use-after-free after a terminal CUDA failure.
    static_cast<void>(impl_.release());
  }
}

CudaSpineBf16PreparedLayer::CudaSpineBf16PreparedLayer(
    CudaSpineBf16PreparedLayer&&) noexcept = default;
CudaSpineBf16PreparedLayer& CudaSpineBf16PreparedLayer::operator=(
    CudaSpineBf16PreparedLayer&& other) noexcept {
  if (this != &other) {
    CudaSpineBf16PreparedLayer old(std::move(*this));
    impl_ = std::move(other.impl_);
  }
  return *this;
}

CudaSpineBf16SourceKind CudaSpineBf16PreparedLayer::source_kind() const
    noexcept {
  return impl_ == nullptr ? CudaSpineBf16SourceKind::DetachedStaged
                          : impl_->kind;
}
std::size_t CudaSpineBf16PreparedLayer::submissions() const noexcept {
  return impl_ == nullptr ? 0 : impl_->lifetime.submissions();
}
bool CudaSpineBf16PreparedLayer::sealed() const noexcept {
  return impl_ != nullptr && impl_->lifetime.sealed();
}
bool CudaSpineBf16PreparedLayer::reclaimed() const noexcept {
  return impl_ == nullptr || impl_->lifetime.reclaimed();
}

at::Tensor CudaSpineBf16PreparedLayer::submit(
    const CudaSpineBf16MatrixView& matrix,
    const at::Tensor& activation) {
  if (impl_ == nullptr) {
    throw std::logic_error(
        "CUDA RAW_BF16 projection requires a live prepared layer");
  }
  std::lock_guard<std::mutex> call_lock(impl_->call_mutex);
  if (impl_ == nullptr || !impl_->lifetime.open()) {
    throw std::logic_error(
        "CUDA RAW_BF16 projection requires an open prepared layer");
  }
  validate_matrix_view_descriptor(matrix, impl_->logical_slab_bytes);
  validate_activation(activation, impl_->device, matrix.columns);
  const c10::cuda::CUDAGuard guard(impl_->device);
  const c10::cuda::CUDAStream selected_stream =
      impl_->submission_stream();
  Impl::StreamCompletion* device_completion =
      impl_->prepare_device_completion(selected_stream);
  const std::int64_t positions = activation.size(0);
  at::Tensor output = at::empty(
      {positions, matrix.rows},
      at::TensorOptions().dtype(at::kFloat).device(impl_->device));
  const auto* weights = reinterpret_cast<const std::uint16_t*>(
      impl_->storage_base() + matrix.matrix_byte_offset);
  bool launch_submitted = false;
  bool completion_advanced = false;
  try {
    // Register every CUDA allocator owner before the raw launch. If its last
    // Tensor handle is released after this function returns (or throws), the
    // caching allocator records a completion event on this submission stream
    // before it may reuse the allocation. This is both exception-safe and
    // bounded for persistent layers/globals: unlike a retire vector, it does
    // not retain one activation and output handle per generated token.
    record_stream(impl_->device_storage, selected_stream);
    record_stream(activation, selected_stream);
    record_stream(output, selected_stream);
    require_launch(
        "CUDA RAW_BF16 multi-position projection",
        k3_cuda_spine_bf16_launch_v1(
            weights, activation.const_data_ptr<float>(),
            output.data_ptr<float>(), static_cast<int>(matrix.rows),
            static_cast<int>(matrix.columns), static_cast<int>(positions),
            stream_pointer(selected_stream)));
    launch_submitted = true;
    // This private event is updated immediately after every launch. Public
    // source-use seal later changes protocol state only; abort can therefore
    // wait this event rather than synchronizing unrelated stream work.
    if (device_completion == nullptr) {
      impl_->record_latest_completion();
    } else {
      impl_->record_device_completion(selected_stream, *device_completion);
    }
    completion_advanced = true;
    impl_->lifetime.note_submission();
  } catch (...) {
    if (launch_submitted && !completion_advanced &&
        !impl_->completion_untrustworthy) {
      const cudaError_t drained =
          cudaStreamSynchronize(selected_stream.stream());
      if (drained != cudaSuccess) {
        impl_->completion_untrustworthy = true;
      }
    }
    impl_->lifetime.poison();
    throw;
  }
  return output;
}

at::Tensor CudaSpineBf16PreparedLayer::submit_t1(
    const CudaSpineBf16MatrixView& matrix,
    const at::Tensor& activation) {
  if (!activation.defined() || activation.dim() != 2 ||
      activation.size(0) != 1) {
    throw std::invalid_argument(
        "CUDA RAW_BF16 T=1 submission requires activation [1, columns]");
  }
  return submit(matrix, activation);
}

void CudaSpineBf16PreparedLayer::seal() {
  if (impl_ == nullptr) {
    throw std::logic_error("CUDA RAW_BF16 prepared layer is stale");
  }
  std::lock_guard<std::mutex> call_lock(impl_->call_mutex);
  impl_->lifetime.seal();
}

bool CudaSpineBf16PreparedLayer::try_reclaim() {
  if (impl_ == nullptr) {
    throw std::logic_error("CUDA RAW_BF16 prepared layer is stale");
  }
  std::lock_guard<std::mutex> call_lock(impl_->call_mutex);
  impl_->lifetime.require_reclaim_query();
  const c10::cuda::CUDAGuard guard(impl_->device);
  if (impl_->kind == CudaSpineBf16SourceKind::DetachedDeviceOwned) {
    for (const Impl::StreamCompletion& item : impl_->device_completions) {
      if (!item.recorded) {
        continue;
      }
      const cudaError_t status = cudaEventQuery(item.event);
      if (status == cudaErrorNotReady) {
        return false;
      }
      if (status != cudaSuccess) {
        impl_->lifetime.poison();
        throw std::runtime_error(
            std::string("CUDA RAW_BF16 device completion query failed: ") +
            cudaGetErrorString(status));
      }
    }
  } else {
    const cudaError_t status = cudaEventQuery(impl_->completion);
    if (status == cudaErrorNotReady) {
      return false;
    }
    if (status != cudaSuccess) {
      impl_->lifetime.poison();
      throw std::runtime_error(
          std::string("CUDA RAW_BF16 completion query failed: ") +
          cudaGetErrorString(status));
    }
  }
  impl_->release_resources();
  impl_->lifetime.complete_reclaim();
  return true;
}

void CudaSpineBf16PreparedLayer::abort_and_reclaim() {
  if (impl_ == nullptr || impl_->lifetime.reclaimed()) {
    throw std::logic_error(
        "CUDA RAW_BF16 prepared layer is stale or already reclaimed");
  }
  std::lock_guard<std::mutex> call_lock(impl_->call_mutex);
  const c10::cuda::CUDAGuard guard(impl_->device);
  if (impl_->completion_untrustworthy) {
    throw std::runtime_error(
        "CUDA RAW_BF16 completion is untrustworthy; retaining its bounded "
        "prepared storage");
  }
  if (impl_->kind == CudaSpineBf16SourceKind::DetachedDeviceOwned) {
    for (const Impl::StreamCompletion& item : impl_->device_completions) {
      if (!item.recorded) {
        continue;
      }
      const cudaError_t status = cudaEventSynchronize(item.event);
      if (status != cudaSuccess) {
        impl_->lifetime.poison();
        throw std::runtime_error(
            std::string("CUDA RAW_BF16 device completion drain failed: ") +
            cudaGetErrorString(status));
      }
    }
  } else if (impl_->completion_recorded) {
    const cudaError_t status = cudaEventSynchronize(impl_->completion);
    if (status != cudaSuccess) {
      impl_->lifetime.poison();
      throw std::runtime_error(
          std::string("CUDA RAW_BF16 completion drain failed: ") +
          cudaGetErrorString(status));
    }
  }
  impl_->release_resources();
  impl_->lifetime.complete_abort();
}

struct CudaSpineBf16Projector::Impl {
  explicit Impl(const at::Device& selected) : device(selected) {
    capability.compiled = true;
    capability.device_index =
        selected.has_index() ? static_cast<std::int32_t>(selected.index()) : -1;
  }

  void qualify() {
    if (!device.is_cuda() || !device.has_index()) {
      capability.detail =
          "CUDA RAW_BF16 requires a canonical indexed CUDA device";
      return;
    }
    try {
      const c10::cuda::CUDAGuard guard(device);
      if (k3_cuda_spine_bf16_abi_version() != kCudaSpineBf16Abi) {
        throw std::runtime_error("CUDA RAW_BF16 ABI version mismatch");
      }
      int major = 0;
      int minor = 0;
      const int status = k3_cuda_spine_bf16_available(
          static_cast<int>(device.index()), &major, &minor);
      if (status != 1) {
        throw std::runtime_error(
            native_error("CUDA RAW_BF16 device probe", status));
      }
      run_decode_canary(device);
      run_finite_gemv_canary(device);
      const bool ats_attributes =
          direct_host_ats_attributes(static_cast<int>(device.index()));
      const bool ats_canary =
          ats_attributes ? run_direct_host_canary(device) : false;
      capability.compute_major = major;
      capability.compute_minor = minor;
      capability.maximum_positions =
          static_cast<std::uint32_t>(kMaximumPositions);
      capability.direct_host_ats = ats_attributes && ats_canary;
      capability.direct_host_runtime_activation_qualified = false;
      capability.available = true;
      capability.detail = capability.direct_host_ats
          ? "exact CUDA RAW_BF16 decode, finite GEMV, and ordinary-pageable-"
            "host ATS canaries passed; ATS still awaits a device+shape speed "
            "gate"
          : "exact CUDA RAW_BF16 decode and finite GEMV canaries passed; "
            "pinned BF16 slab staging selected";
    } catch (const std::exception& error) {
      capability.available = false;
      capability.direct_host_ats = false;
      capability.direct_host_runtime_activation_qualified = false;
      capability.detail = error.what();
    } catch (...) {
      capability.available = false;
      capability.direct_host_ats = false;
      capability.direct_host_runtime_activation_qualified = false;
      capability.detail = "non-standard CUDA RAW_BF16 qualification failure";
    }
  }

  void ensure_qualified() {
    std::call_once(qualification_once, [this] { qualify(); });
  }

  at::Device device;
  std::once_flag qualification_once;
  CudaSpineBf16Capability capability;
};

CudaSpineBf16Projector::CudaSpineBf16Projector(const at::Device& device)
    : impl_(std::make_unique<Impl>(device)) {}
CudaSpineBf16Projector::~CudaSpineBf16Projector() = default;
const CudaSpineBf16Capability& CudaSpineBf16Projector::capability() {
  impl_->ensure_qualified();
  return impl_->capability;
}

std::unique_ptr<CudaSpineBf16PreparedLayer>
CudaSpineBf16Projector::prepare_host_layer(
    const CudaSpineBf16HostSlab& slab,
    const CudaSpineBf16HostPolicy policy) {
  if (policy != CudaSpineBf16HostPolicy::Auto &&
      policy != CudaSpineBf16HostPolicy::StageOnly) {
    throw std::invalid_argument("CUDA RAW_BF16 host policy is invalid");
  }
  const auto& qualified = capability();
  if (!qualified.available) {
    throw std::runtime_error("CUDA RAW_BF16 unavailable: " + qualified.detail);
  }
  validate_host_slab_descriptor(slab);
  static_cast<void>(classify_host_pointer(impl_->device,
                                          slab.allocation_base));
  const c10::cuda::CUDAGuard guard(impl_->device);
  const auto stream = c10::cuda::getCurrentCUDAStream(impl_->device.index());
  auto prepared = std::make_unique<CudaSpineBf16PreparedLayer::Impl>(
      impl_->device, stream, CudaSpineBf16SourceKind::DetachedStaged,
      slab.logical_slab_bytes);
  try {
    prepared->pinned_stage = at::empty(
        {static_cast<std::int64_t>(slab.logical_slab_bytes)},
        at::TensorOptions()
            .dtype(at::kByte)
            .device(at::kCPU)
            .pinned_memory(true));
    if (!prepared->pinned_stage.is_pinned()) {
      throw std::runtime_error(
          "CUDA RAW_BF16 slab staging allocation is not pinned");
    }
    std::memcpy(prepared->pinned_stage.data_ptr<std::uint8_t>(),
                slab.allocation_base, slab.logical_slab_bytes);
    prepared->device_storage = at::empty(
        {static_cast<std::int64_t>(slab.logical_slab_bytes)},
        at::TensorOptions().dtype(at::kByte).device(impl_->device));
    prepared->device_storage.copy_(prepared->pinned_stage, true);
    record_stream(prepared->device_storage, stream);
    // Protect internal pinned/device storage even if no projection is later
    // submitted. The caller's reader slab is already Detached after memcpy.
    prepared->record_latest_completion();
  } catch (...) {
    const cudaError_t drained = cudaStreamSynchronize(stream.stream());
    if (drained != cudaSuccess) {
      prepared->lifetime.poison();
      static_cast<void>(prepared.release());
      throw std::runtime_error(
          std::string("CUDA RAW_BF16 slab preparation failed and stream "
                      "drain also failed: ") +
          cudaGetErrorString(drained));
    }
    throw;
  }
  return std::unique_ptr<CudaSpineBf16PreparedLayer>(
      new CudaSpineBf16PreparedLayer(std::move(prepared)));
}

std::unique_ptr<CudaSpineBf16PreparedLayer>
CudaSpineBf16Projector::prepare_direct_host_layer_for_benchmark(
    const CudaSpineBf16HostSlab& slab) {
  const auto& qualified = capability();
  if (!qualified.available || !qualified.direct_host_ats) {
    throw std::runtime_error(
        "CUDA direct-host RAW_BF16 is not physically ATS-qualified");
  }
  validate_host_slab_descriptor(slab);
  const std::size_t page = system_page_size();
  if (classify_host_pointer(impl_->device, slab.allocation_base) !=
          HostPointerKind::Ordinary ||
      (reinterpret_cast<std::uintptr_t>(slab.allocation_base) % page) != 0 ||
      (slab.allocation_bytes % page) != 0) {
    throw std::invalid_argument(
        "CUDA direct-host RAW_BF16 needs one ordinary page-aligned rounded "
        "reader allocation");
  }
  const c10::cuda::CUDAGuard guard(impl_->device);
  const auto stream = c10::cuda::getCurrentCUDAStream(impl_->device.index());
  auto prepared = std::make_unique<CudaSpineBf16PreparedLayer::Impl>(
      impl_->device, stream,
      CudaSpineBf16SourceKind::BorrowedDirectHostAts,
      slab.logical_slab_bytes);
  prepared->borrowed_host_base = slab.allocation_base;
  return std::unique_ptr<CudaSpineBf16PreparedLayer>(
      new CudaSpineBf16PreparedLayer(std::move(prepared)));
}

std::unique_ptr<CudaSpineBf16PreparedLayer>
CudaSpineBf16Projector::prepare_device_layer(
    const CudaSpineBf16DeviceSlab& slab) {
  const auto& qualified = capability();
  if (!qualified.available) {
    throw std::runtime_error("CUDA RAW_BF16 unavailable: " + qualified.detail);
  }
  if (!slab.storage.defined() || !slab.storage.device().is_cuda() ||
      slab.storage.device() != impl_->device || !slab.storage.is_contiguous() ||
      slab.storage.storage_offset() != 0 ||
      !cuda_spine_bf16_device_storage_dtype_supported(
          slab.storage.scalar_type()) ||
      slab.logical_slab_bytes == 0 ||
      (slab.logical_slab_bytes & 1U) != 0 ||
      slab.logical_slab_bytes > device_tensor_bytes(slab.storage)) {
    throw std::invalid_argument(
        "CUDA device BF16 slab must be contiguous Byte/BFloat16/UInt16/Short "
        "raw storage with canonical even-byte bounds on the selected device");
  }
  const c10::cuda::CUDAGuard guard(impl_->device);
  const auto stream = c10::cuda::getCurrentCUDAStream(impl_->device.index());
  auto prepared = std::make_unique<CudaSpineBf16PreparedLayer::Impl>(
      impl_->device, stream, CudaSpineBf16SourceKind::DetachedDeviceOwned,
      slab.logical_slab_bytes);
  prepared->device_storage = slab.storage;
  return std::unique_ptr<CudaSpineBf16PreparedLayer>(
      new CudaSpineBf16PreparedLayer(std::move(prepared)));
}

bool cuda_spine_bf16_compiled() noexcept { return true; }

#endif

}  // namespace deltafin::provider_internal
