#include "provider_bf16_cpu.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define DELTAFIN_BF16_CPU_HAVE_NEON 1
#else
#define DELTAFIN_BF16_CPU_HAVE_NEON 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

// GCC and Clang need a target attribute to accept AVX2 intrinsics in a
// translation unit built for a lower baseline. MSVC accepts every intrinsic
// regardless of /arch and emits AVX2 only where one is written, so the same
// isolation needs no attribute there; the baseline stays /arch:AVX and the
// AVX2 kernel is chosen at run time by CPUID.
#if (defined(__x86_64__) || defined(_M_X64)) &&                                \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DELTAFIN_BF16_CPU_HAVE_AVX2_TARGET 1
#define DELTAFIN_BF16_CPU_AVX2_TARGET_ATTRIBUTE                               \
  __attribute__((target("avx2,fma")))
#elif defined(_M_X64) && defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#define DELTAFIN_BF16_CPU_HAVE_AVX2_TARGET 1
#define DELTAFIN_BF16_CPU_AVX2_TARGET_ATTRIBUTE
#else
#define DELTAFIN_BF16_CPU_HAVE_AVX2_TARGET 0
#endif

namespace deltafin::provider_internal {
namespace {

using DotFunction = float (*)(const std::uint16_t *, const float *,
                              std::size_t);

struct AddressRange {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

[[nodiscard]] std::size_t checked_multiply(const std::size_t left,
                                           const std::size_t right,
                                           const char *label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string(label) + " overflows size_t");
  }
  return left * right;
}

template <typename T>
[[nodiscard]] AddressRange checked_address_range(const T *pointer,
                                                 const std::size_t elements,
                                                 const char *label) {
  if (pointer == nullptr) {
    throw std::invalid_argument(std::string(label) + " pointer is null");
  }
  if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0) {
    throw std::invalid_argument(std::string(label) +
                                " pointer is not naturally aligned");
  }
  const std::size_t bytes = checked_multiply(elements, sizeof(T), label);
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    throw std::invalid_argument(std::string(label) +
                                " address range overflows");
  }
  return AddressRange{begin, begin + bytes};
}

[[nodiscard]] bool ranges_overlap(const AddressRange left,
                                  const AddressRange right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] bool fp_environment_qualified() noexcept {
  if (std::fegetround() != FE_TONEAREST) {
    return false;
  }
#if defined(__x86_64__) || defined(_M_X64)
  const unsigned int control = _mm_getcsr();
  constexpr unsigned int kDaz = 1U << 6U;
  constexpr unsigned int kRounding = 3U << 13U;
  constexpr unsigned int kFtz = 1U << 15U;
  if ((control & (kDaz | kRounding | kFtz)) != 0U) {
    return false;
  }
#elif defined(__aarch64__)
  std::uint64_t control = 0;
  __asm__ volatile("mrs %0, fpcr" : "=r"(control));
  constexpr std::uint64_t kRounding = 3ULL << 22U;
  constexpr std::uint64_t kFlushToZero = 1ULL << 24U;
  constexpr std::uint64_t kDefaultNan = 1ULL << 25U;
  if ((control & (kRounding | kFlushToZero | kDefaultNan)) != 0U) {
    return false;
  }
#endif
  return true;
}

void validate_apply(std::span<const std::uint16_t> weights,
                    std::span<const float> input, std::span<float> output,
                    const std::size_t rows, const std::size_t columns,
                    const std::size_t active_workers,
                    const std::size_t worker_capacity) {
  if (rows == 0 || columns == 0) {
    throw std::invalid_argument("BF16 projection dimensions must be positive");
  }
  if (active_workers == 0 || active_workers > worker_capacity) {
    throw std::invalid_argument(
        "BF16 projection active worker count is outside the retained pool");
  }
  const std::size_t weight_elements =
      checked_multiply(rows, columns, "BF16 projection shape");
  if (weights.size() != weight_elements || input.size() != columns ||
      output.size() != rows) {
    throw std::invalid_argument(
        "BF16 projection span size does not match shape");
  }

  const AddressRange weight_range =
      checked_address_range(weights.data(), weights.size(), "BF16 weights");
  const AddressRange input_range =
      checked_address_range(input.data(), input.size(), "FP32 input");
  const AddressRange output_range =
      checked_address_range(output.data(), output.size(), "FP32 output");
  if (ranges_overlap(weight_range, input_range) ||
      ranges_overlap(weight_range, output_range) ||
      ranges_overlap(input_range, output_range)) {
    throw std::invalid_argument(
        "BF16 projection weight, input, and output ranges must not overlap");
  }
}

[[nodiscard]] float dot_scalar(const std::uint16_t *weights, const float *input,
                               const std::size_t columns) noexcept {
  float accumulator = 0.0F;
  for (std::size_t column = 0; column < columns; ++column) {
    accumulator = std::fma(decode_bf16_exact(weights[column]), input[column],
                           accumulator);
  }
  return accumulator;
}

#if DELTAFIN_BF16_CPU_HAVE_NEON
[[nodiscard]] float dot_neon(const std::uint16_t *weights, const float *input,
                             const std::size_t columns) noexcept {
  float32x4_t accumulator0 = vdupq_n_f32(0.0F);
  float32x4_t accumulator1 = vdupq_n_f32(0.0F);
  float32x4_t accumulator2 = vdupq_n_f32(0.0F);
  float32x4_t accumulator3 = vdupq_n_f32(0.0F);
  std::size_t column = 0;
  for (; column + 16 <= columns; column += 16) {
    const uint16x8_t packed0 = vld1q_u16(weights + column);
    const uint16x8_t packed1 = vld1q_u16(weights + column + 8);
    const float32x4_t weight0 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_low_u16(packed0)), 16));
    const float32x4_t weight1 =
        vreinterpretq_f32_u32(vshlq_n_u32(vmovl_high_u16(packed0), 16));
    const float32x4_t weight2 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_low_u16(packed1)), 16));
    const float32x4_t weight3 =
        vreinterpretq_f32_u32(vshlq_n_u32(vmovl_high_u16(packed1), 16));
    accumulator0 = vfmaq_f32(accumulator0, weight0, vld1q_f32(input + column));
    accumulator1 =
        vfmaq_f32(accumulator1, weight1, vld1q_f32(input + column + 4));
    accumulator2 =
        vfmaq_f32(accumulator2, weight2, vld1q_f32(input + column + 8));
    accumulator3 =
        vfmaq_f32(accumulator3, weight3, vld1q_f32(input + column + 12));
  }
  const float32x4_t combined = vaddq_f32(vaddq_f32(accumulator0, accumulator1),
                                         vaddq_f32(accumulator2, accumulator3));
  float accumulator = vaddvq_f32(combined);
  for (; column < columns; ++column) {
    accumulator = std::fma(decode_bf16_exact(weights[column]), input[column],
                           accumulator);
  }
  return accumulator;
}
#endif

#if DELTAFIN_BF16_CPU_HAVE_AVX2_TARGET
DELTAFIN_BF16_CPU_AVX2_TARGET_ATTRIBUTE
[[nodiscard]] float dot_avx2_fma(const std::uint16_t *weights,
                                 const float *input,
                                 const std::size_t columns) noexcept {
  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  __m256 accumulator2 = _mm256_setzero_ps();
  __m256 accumulator3 = _mm256_setzero_ps();
  std::size_t column = 0;
  for (; column + 32 <= columns; column += 32) {
    const __m128i packed0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(weights + column));
    const __m128i packed1 = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(weights + column + 8));
    const __m128i packed2 = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(weights + column + 16));
    const __m128i packed3 = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(weights + column + 24));
    const __m256 weight0 = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed0), 16));
    const __m256 weight1 = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed1), 16));
    const __m256 weight2 = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed2), 16));
    const __m256 weight3 = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed3), 16));
    accumulator0 =
        _mm256_fmadd_ps(weight0, _mm256_loadu_ps(input + column), accumulator0);
    accumulator1 = _mm256_fmadd_ps(weight1, _mm256_loadu_ps(input + column + 8),
                                   accumulator1);
    accumulator2 = _mm256_fmadd_ps(
        weight2, _mm256_loadu_ps(input + column + 16), accumulator2);
    accumulator3 = _mm256_fmadd_ps(
        weight3, _mm256_loadu_ps(input + column + 24), accumulator3);
  }
  const __m256 combined =
      _mm256_add_ps(_mm256_add_ps(accumulator0, accumulator1),
                    _mm256_add_ps(accumulator2, accumulator3));
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, combined);
  float accumulator = 0.0F;
  for (const float lane : lanes) {
    accumulator += lane;
  }
  for (; column < columns; ++column) {
    accumulator = std::fma(decode_bf16_exact(weights[column]), input[column],
                           accumulator);
  }
  _mm256_zeroupper();
  return accumulator;
}

#if defined(_MSC_VER) && !defined(__clang__)
// What `__builtin_cpu_supports` answers on GCC and Clang, spelled with the
// intrinsics MSVC provides: the CPUID bit for each instruction set and the
// operating system's agreement to save the YMM state AVX code uses.
[[nodiscard]] bool avx2_fma_available() noexcept {
  int registers[4];
  __cpuid(registers, 0);
  if (registers[0] < 7) {
    return false;
  }
  __cpuid(registers, 1);
  const bool fma = (registers[2] & (1 << 12)) != 0;
  const bool avx = (registers[2] & (1 << 28)) != 0;
  const bool osxsave = (registers[2] & (1 << 27)) != 0;
  if (!fma || !avx || !osxsave || (_xgetbv(0) & 0x6U) != 0x6U) {
    return false;
  }
  __cpuidex(registers, 7, 0);
  return (registers[1] & (1 << 5)) != 0;
}
#else
[[nodiscard]] bool avx2_fma_available() noexcept {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif
#else
[[nodiscard]] bool avx2_fma_available() noexcept { return false; }
#endif

[[nodiscard]] Bf16CpuT1Dispatch
resolve_dispatch(const Bf16CpuT1Dispatch requested) {
  if (requested == Bf16CpuT1Dispatch::Auto) {
    if (bf16_cpu_t1_dispatch_available(Bf16CpuT1Dispatch::Avx2Fma)) {
      return Bf16CpuT1Dispatch::Avx2Fma;
    }
    if (bf16_cpu_t1_dispatch_available(Bf16CpuT1Dispatch::Neon)) {
      return Bf16CpuT1Dispatch::Neon;
    }
    return Bf16CpuT1Dispatch::Scalar;
  }
  switch (requested) {
  case Bf16CpuT1Dispatch::Scalar:
  case Bf16CpuT1Dispatch::Neon:
  case Bf16CpuT1Dispatch::Avx2Fma:
    if (!bf16_cpu_t1_dispatch_available(requested)) {
      throw std::runtime_error(
          std::string("requested BF16 CPU dispatch is unavailable: ") +
          bf16_cpu_t1_dispatch_name(requested));
    }
    return requested;
  case Bf16CpuT1Dispatch::Auto:
    break;
  }
  throw std::invalid_argument("unknown BF16 CPU dispatch selector");
}

[[nodiscard]] DotFunction dot_function(const Bf16CpuT1Dispatch dispatch) {
  switch (dispatch) {
  case Bf16CpuT1Dispatch::Scalar:
    return &dot_scalar;
  case Bf16CpuT1Dispatch::Neon:
#if DELTAFIN_BF16_CPU_HAVE_NEON
    return &dot_neon;
#else
    break;
#endif
  case Bf16CpuT1Dispatch::Avx2Fma:
#if DELTAFIN_BF16_CPU_HAVE_AVX2_TARGET
    return &dot_avx2_fma;
#else
    break;
#endif
  case Bf16CpuT1Dispatch::Auto:
    break;
  }
  throw std::logic_error("resolved BF16 CPU dispatch has no kernel");
}

} // namespace

float decode_bf16_exact(const std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

const char *
bf16_cpu_t1_dispatch_name(const Bf16CpuT1Dispatch dispatch) noexcept {
  switch (dispatch) {
  case Bf16CpuT1Dispatch::Auto:
    return "auto";
  case Bf16CpuT1Dispatch::Scalar:
    return "scalar-fp32-fma";
  case Bf16CpuT1Dispatch::Neon:
    return "aarch64-neon-fp32-fma";
  case Bf16CpuT1Dispatch::Avx2Fma:
    return "x86_64-avx2-fma3-fp32";
  }
  return "unknown";
}

bool bf16_cpu_t1_dispatch_available(const Bf16CpuT1Dispatch dispatch) noexcept {
  switch (dispatch) {
  case Bf16CpuT1Dispatch::Auto:
  case Bf16CpuT1Dispatch::Scalar:
    return true;
  case Bf16CpuT1Dispatch::Neon:
    return DELTAFIN_BF16_CPU_HAVE_NEON != 0;
  case Bf16CpuT1Dispatch::Avx2Fma:
    return avx2_fma_available();
  }
  return false;
}

bool bf16_cpu_fp_environment_qualified() noexcept {
  return fp_environment_qualified();
}

struct Bf16CpuT1Kernel::Impl final {
  Impl(const std::size_t requested_workers,
       const Bf16CpuT1Dispatch requested_dispatch)
      : worker_capacity(requested_workers),
        dispatch(resolve_dispatch(requested_dispatch)),
        dot(dot_function(dispatch)) {
    if (worker_capacity == 0 || worker_capacity > kBf16CpuT1MaximumWorkers) {
      throw std::invalid_argument(
          "BF16 CPU worker count must be between 1 and the bounded maximum");
    }
    if (!fp_environment_qualified()) {
      throw std::runtime_error(
          "BF16 CPU kernel requires FE_TONEAREST with denormal and NaN "
          "preservation enabled");
    }
    workers.reserve(worker_capacity);
    try {
      for (std::size_t index = 0; index < worker_capacity; ++index) {
        workers.emplace_back([this, index] { worker(index); });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        stop = true;
        ++generation;
      }
      work_ready.notify_all();
      for (std::thread &thread : workers) {
        if (thread.joinable()) {
          thread.join();
        }
      }
      throw;
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      stop = true;
      ++generation;
    }
    work_ready.notify_all();
    for (std::thread &thread : workers) {
      thread.join();
    }
  }

  void run(std::span<const std::uint16_t> new_weights,
           std::span<const float> new_input, std::span<float> new_output,
           const std::size_t rows, const std::size_t columns,
           const std::size_t requested_active_workers) {
    std::lock_guard<std::mutex> call_lock(call_mutex);
    if (!fp_environment_qualified()) {
      throw std::runtime_error(
          "BF16 CPU projection refuses a hostile caller FP environment");
    }
    const std::size_t active = requested_active_workers == 0
                                   ? worker_capacity
                                   : requested_active_workers;
    validate_apply(new_weights, new_input, new_output, rows, columns, active,
                   worker_capacity);
    {
      std::lock_guard<std::mutex> state_lock(state_mutex);
      weights = new_weights.data();
      input = new_input.data();
      output = new_output.data();
      job_rows = rows;
      job_columns = columns;
      active_workers = active;
      completed_workers = 0;
      worker_environment_failure = false;
      ++generation;
    }
    work_ready.notify_all();
    std::unique_lock<std::mutex> state_lock(state_mutex);
    work_done.wait(state_lock,
                   [this] { return completed_workers == active_workers; });
    if (worker_environment_failure) {
      throw std::runtime_error(
          "BF16 CPU projection worker FP environment is not authoritative");
    }
  }

  void worker(const std::size_t worker_index) noexcept {
    std::size_t observed_generation = 0;
    std::unique_lock<std::mutex> lock(state_mutex);
    for (;;) {
      work_ready.wait(lock, [this, observed_generation] {
        return stop || generation != observed_generation;
      });
      if (stop) {
        return;
      }
      observed_generation = generation;
      if (worker_index >= active_workers) {
        continue;
      }

      const std::uint16_t *job_weights = weights;
      const float *job_input = input;
      float *job_output = output;
      const std::size_t rows = job_rows;
      const std::size_t columns = job_columns;
      const std::size_t count = active_workers;
      const std::size_t quotient = rows / count;
      const std::size_t remainder = rows % count;
      const std::size_t begin =
          quotient * worker_index + std::min(worker_index, remainder);
      const std::size_t end =
          begin + quotient + (worker_index < remainder ? 1 : 0);
      lock.unlock();
      const bool environment_ok = fp_environment_qualified();
      if (environment_ok) {
        for (std::size_t row = begin; row < end; ++row) {
          job_output[row] =
              dot(job_weights + row * columns, job_input, columns);
        }
      }
      lock.lock();
      worker_environment_failure |= !environment_ok;
      ++completed_workers;
      if (completed_workers == active_workers) {
        work_done.notify_one();
      }
    }
  }

  const std::size_t worker_capacity;
  const Bf16CpuT1Dispatch dispatch;
  const DotFunction dot;
  std::vector<std::thread> workers;
  std::mutex call_mutex;
  std::mutex state_mutex;
  std::condition_variable work_ready;
  std::condition_variable work_done;
  bool stop = false;
  std::size_t generation = 0;
  std::size_t active_workers = 0;
  std::size_t completed_workers = 0;
  bool worker_environment_failure = false;
  const std::uint16_t *weights = nullptr;
  const float *input = nullptr;
  float *output = nullptr;
  std::size_t job_rows = 0;
  std::size_t job_columns = 0;
};

Bf16CpuT1Kernel::Bf16CpuT1Kernel(const std::size_t worker_count,
                                 const Bf16CpuT1Dispatch dispatch)
    : impl_(std::make_unique<Impl>(worker_count, dispatch)) {}

Bf16CpuT1Kernel::~Bf16CpuT1Kernel() = default;

std::size_t Bf16CpuT1Kernel::worker_count() const noexcept {
  return impl_->worker_capacity;
}

Bf16CpuT1Dispatch Bf16CpuT1Kernel::selected_dispatch() const noexcept {
  return impl_->dispatch;
}

void Bf16CpuT1Kernel::apply(std::span<const std::uint16_t> weights,
                            std::span<const float> input,
                            std::span<float> output, const std::size_t rows,
                            const std::size_t columns,
                            const std::size_t active_workers) {
  impl_->run(weights, input, output, rows, columns, active_workers);
}

std::shared_ptr<ExactBf16Storage>
make_exact_bf16_storage(at::Tensor storage) {
  if (!storage.defined() ||
      (storage.scalar_type() != at::kUInt16 &&
       storage.scalar_type() != at::kShort) ||
      !storage.is_contiguous() || storage.dim() != 1 ||
      storage.device().is_meta()) {
    throw std::invalid_argument(
        "original-BF16 storage must be a contiguous non-meta opaque 16-bit "
        "vector");
  }
  return std::make_shared<ExactBf16Storage>(
      ExactBf16Storage{std::move(storage), nullptr});
}

OriginalBf16Matrix make_owned_original_bf16(
    std::shared_ptr<ExactBf16Storage> storage,
    const std::size_t element_offset, const std::size_t rows,
    const std::size_t columns, Bf16CpuT1Kernel *cpu_kernel) {
  const std::size_t elements = checked_multiply(
      rows, columns, "owned original-BF16 matrix shape");
  if (storage == nullptr || !storage->tensor.defined() ||
      (storage->tensor.scalar_type() != at::kUInt16 &&
       storage->tensor.scalar_type() != at::kShort) ||
      !storage->tensor.is_contiguous() || storage->tensor.dim() != 1 ||
      storage->tensor.numel() < 0 || rows == 0 || columns == 0 ||
      element_offset > static_cast<std::size_t>(storage->tensor.numel()) ||
      elements > static_cast<std::size_t>(storage->tensor.numel()) -
          element_offset) {
    throw std::invalid_argument(
        "owned original-BF16 matrix has invalid storage or dimensions");
  }
  const bool cpu = storage->tensor.device().is_cpu();
  if ((cpu &&
       (storage->tensor.scalar_type() != at::kUInt16 ||
        storage->projection_backend != nullptr || cpu_kernel == nullptr)) ||
      (!cpu &&
       (storage->projection_backend == nullptr || cpu_kernel != nullptr ||
        !storage->projection_backend->matches_device(
            storage->tensor.device())))) {
    throw std::invalid_argument(
        "owned original-BF16 matrix has an invalid selected-device backend");
  }
  return OriginalBf16Matrix{std::move(storage), element_offset, nullptr, rows,
                            columns, cpu_kernel};
}

OriginalBf16Matrix make_owned_original_bf16_cpu(
    std::shared_ptr<ExactBf16Storage> storage,
    const std::size_t element_offset, const std::size_t rows,
    const std::size_t columns, Bf16CpuT1Kernel *kernel) {
  if (storage == nullptr ||
      storage->tensor.device() != at::Device(at::kCPU)) {
    throw std::invalid_argument(
        "owned original-BF16 CPU matrix has invalid storage or dimensions");
  }
  return make_owned_original_bf16(std::move(storage), element_offset, rows,
                                  columns, kernel);
}

OriginalBf16Matrix make_borrowed_original_bf16_cpu(
    const std::uint16_t *data, const std::size_t rows,
    const std::size_t columns, Bf16CpuT1Kernel *kernel) {
  static_cast<void>(checked_multiply(
      rows, columns, "borrowed original-BF16 matrix shape"));
  if (data == nullptr || kernel == nullptr || rows == 0 || columns == 0 ||
      reinterpret_cast<std::uintptr_t>(data) % alignof(std::uint16_t) != 0) {
    throw std::invalid_argument(
        "borrowed original-BF16 CPU matrix has invalid storage or dimensions");
  }
  return OriginalBf16Matrix{{}, 0, data, rows, columns, kernel};
}

bool original_bf16_matrix_matches(
    const OriginalBf16Matrix &matrix, const at::Device &device,
    const std::size_t rows, const std::size_t columns) noexcept {
  if (rows == 0 || columns == 0 || matrix.rows != rows ||
      matrix.columns != columns ||
      matrix.is_owned() == matrix.is_borrowed_cpu() ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    return false;
  }
  const std::size_t elements = rows * columns;
  if (matrix.is_borrowed_cpu()) {
    return device.is_cpu() && matrix.cpu_kernel != nullptr &&
        matrix.owned_storage == nullptr && matrix.owned_element_offset == 0 &&
        reinterpret_cast<std::uintptr_t>(matrix.borrowed_cpu_data) %
                alignof(std::uint16_t) ==
            0;
  }
  if (matrix.borrowed_cpu_data != nullptr ||
      matrix.owned_storage == nullptr) {
    return false;
  }
  const at::Tensor &storage = matrix.owned_storage->tensor;
  const bool scalar_supported = storage.scalar_type() == at::kUInt16 ||
      storage.scalar_type() == at::kShort;
  const bool cpu_contract = device.is_cpu()
      ? storage.scalar_type() == at::kUInt16 && matrix.cpu_kernel != nullptr &&
          matrix.owned_storage->projection_backend == nullptr
      : matrix.cpu_kernel == nullptr &&
          matrix.owned_storage->projection_backend != nullptr &&
          matrix.owned_storage->projection_backend->matches_device(device);
  return storage.defined() && storage.device() == device &&
      scalar_supported && cpu_contract && storage.is_contiguous() &&
      storage.dim() == 1 && storage.numel() >= 0 &&
      matrix.owned_element_offset <=
          static_cast<std::size_t>(storage.numel()) &&
      elements <= static_cast<std::size_t>(storage.numel()) -
          matrix.owned_element_offset;
}

bool original_bf16_cpu_matrix_matches(
    const OriginalBf16Matrix &matrix, const at::Device &device,
    const std::size_t rows, const std::size_t columns) noexcept {
  return device.is_cpu() &&
      original_bf16_matrix_matches(matrix, device, rows, columns);
}

at::Device original_bf16_device(const OriginalBf16Matrix &matrix) {
  const at::Device device = matrix.is_borrowed_cpu()
      ? at::Device(at::kCPU)
      : (matrix.owned_storage == nullptr
             ? at::Device(at::kMeta)
             : matrix.owned_storage->tensor.device());
  if (!original_bf16_matrix_matches(matrix, device, matrix.rows,
                                    matrix.columns)) {
    throw std::invalid_argument(
        "original-BF16 matrix has no valid selected device");
  }
  return device;
}

std::optional<OriginalBf16Matrix> adjacent_original_bf16_cpu_matrices(
    const std::span<const OriginalBf16Matrix *const> matrices) {
  const auto combined = adjacent_original_bf16_matrices(matrices);
  if (!combined.has_value() ||
      !original_bf16_device(*combined).is_cpu()) {
    return std::nullopt;
  }
  return combined;
}

std::optional<OriginalBf16Matrix> adjacent_original_bf16_matrices(
    const std::span<const OriginalBf16Matrix *const> matrices) {
  if (matrices.empty() || matrices.front() == nullptr) {
    return std::nullopt;
  }
  const OriginalBf16Matrix &first = *matrices.front();
  const at::Device device = first.is_borrowed_cpu()
      ? at::Device(at::kCPU)
      : (first.owned_storage == nullptr
             ? at::Device(at::kMeta)
             : first.owned_storage->tensor.device());
  if (!original_bf16_matrix_matches(first, device, first.rows,
                                    first.columns)) {
    return std::nullopt;
  }
  std::size_t rows = 0;
  std::size_t next_offset = first.owned_element_offset;
  const std::uint16_t *next_pointer = first.borrowed_cpu_data;
  for (const OriginalBf16Matrix *candidate : matrices) {
    if (candidate == nullptr ||
        !original_bf16_matrix_matches(*candidate, device, candidate->rows,
                                      first.columns) ||
        candidate->cpu_kernel != first.cpu_kernel ||
        candidate->is_owned() != first.is_owned() ||
        candidate->rows > std::numeric_limits<std::size_t>::max() - rows) {
      return std::nullopt;
    }
    const std::size_t elements = candidate->rows * candidate->columns;
    if (first.is_owned()) {
      if (candidate->owned_storage != first.owned_storage ||
          candidate->owned_element_offset != next_offset ||
          elements > std::numeric_limits<std::size_t>::max() - next_offset) {
        return std::nullopt;
      }
      next_offset += elements;
    } else {
      if (candidate->borrowed_cpu_data != next_pointer) {
        return std::nullopt;
      }
      next_pointer += elements;
    }
    rows += candidate->rows;
  }
  if (first.is_owned()) {
    return make_owned_original_bf16(
        first.owned_storage, first.owned_element_offset, rows, first.columns,
        first.cpu_kernel);
  }
  return make_borrowed_original_bf16_cpu(
      first.borrowed_cpu_data, rows, first.columns, first.cpu_kernel);
}

namespace {

at::Tensor materialize_original_bf16_on_cpu(
    const OriginalBf16Matrix &matrix) {
  const at::Device device = original_bf16_device(matrix);
  if (!original_bf16_matrix_matches(matrix, device, matrix.rows,
                                    matrix.columns)) {
    throw std::invalid_argument(
        "cannot materialize an invalid original-BF16 matrix");
  }
  const std::size_t elements = matrix.rows * matrix.columns;
  const std::uint16_t *unsigned_source = matrix.borrowed_cpu_data;
  const std::int16_t *signed_source = nullptr;
  at::Tensor owned_cpu;
  if (matrix.is_owned()) {
    owned_cpu = matrix.owned_storage->tensor
        .narrow(0, static_cast<std::int64_t>(matrix.owned_element_offset),
                static_cast<std::int64_t>(elements))
        .detach()
        .to(at::kCPU)
        .contiguous();
    if (owned_cpu.scalar_type() == at::kUInt16) {
      unsigned_source = owned_cpu.const_data_ptr<std::uint16_t>();
    } else if (owned_cpu.scalar_type() == at::kShort) {
      signed_source = owned_cpu.const_data_ptr<std::int16_t>();
    } else {
      throw std::logic_error(
          "original-BF16 storage changed scalar type during readback");
    }
  }
  at::Tensor result_cpu = at::empty(
      {static_cast<std::int64_t>(matrix.rows),
       static_cast<std::int64_t>(matrix.columns)},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  float *destination = result_cpu.mutable_data_ptr<float>();
  for (std::size_t index = 0; index < elements; ++index) {
    const std::uint16_t bits = signed_source == nullptr
        ? unsigned_source[index]
        : static_cast<std::uint16_t>(signed_source[index]);
    destination[index] = decode_bf16_exact(bits);
  }
  return result_cpu;
}

} // namespace

at::Tensor materialize_original_bf16_f32(
    const OriginalBf16Matrix &matrix) {
  const at::Device device = original_bf16_device(matrix);
  at::Tensor result_cpu = materialize_original_bf16_on_cpu(matrix);
  return device.is_cpu() ? result_cpu : result_cpu.to(device).contiguous();
}

at::Tensor
materialize_original_bf16_cpu_f32(const OriginalBf16Matrix &matrix) {
  return materialize_original_bf16_on_cpu(matrix);
}

at::Tensor bf16_cpu_linear(const at::Tensor &input,
                           const OriginalBf16Matrix &matrix) {
  const bool owned = matrix.is_owned();
  const bool borrowed = matrix.is_borrowed_cpu();
  if (!matrix.defined() || owned == borrowed || matrix.cpu_kernel == nullptr ||
      matrix.rows == 0 || matrix.columns == 0 ||
      matrix.rows > static_cast<std::size_t>(
                        std::numeric_limits<std::int64_t>::max()) ||
      matrix.columns > static_cast<std::size_t>(
                           std::numeric_limits<std::int64_t>::max()) ||
      matrix.rows > std::numeric_limits<std::size_t>::max() / matrix.columns) {
    throw std::invalid_argument(
        "original-BF16 CPU matrix has invalid storage or dimensions");
  }
  const std::size_t weight_elements = matrix.rows * matrix.columns;
  const std::uint16_t *weight_data = matrix.borrowed_cpu_data;
  if (owned) {
    const at::Tensor &storage = matrix.owned_storage->tensor;
    if (!storage.defined() || storage.device() != at::Device(at::kCPU) ||
        storage.scalar_type() != at::kUInt16 || !storage.is_contiguous() ||
        storage.dim() != 1 || matrix.owned_element_offset >
            static_cast<std::size_t>(storage.numel()) ||
        weight_elements > static_cast<std::size_t>(storage.numel()) -
            matrix.owned_element_offset) {
      throw std::invalid_argument(
          "owned original-BF16 CPU matrix storage is invalid or truncated");
    }
    weight_data = storage.const_data_ptr<std::uint16_t>() +
        matrix.owned_element_offset;
  }
  if (!input.defined() || input.device() != at::Device(at::kCPU) ||
      input.scalar_type() != at::kFloat || !input.is_contiguous() ||
      (input.dim() != 2 && input.dim() != 3) || input.size(-1) <= 0 ||
      static_cast<std::size_t>(input.size(-1)) != matrix.columns ||
      (input.dim() == 3 && input.size(0) != 1)) {
    throw std::invalid_argument(
        "original-BF16 CPU projection requires contiguous fp32 [T,C] or [1,T,C]");
  }
  const std::int64_t positions = input.dim() == 2 ? input.size(0) : input.size(1);
  if (positions < 1 ||
      positions > static_cast<std::int64_t>(kBf16CpuMaximumPositions) ||
      input.numel() != positions * static_cast<std::int64_t>(matrix.columns)) {
    throw std::invalid_argument(
        "original-BF16 CPU projection position count is outside 1..64");
  }

  const auto rows = static_cast<std::int64_t>(matrix.rows);
  at::Tensor output = input.dim() == 2
      ? at::empty({positions, rows},
                  at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
      : at::empty({1, positions, rows},
                  at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const float *input_data = input.const_data_ptr<float>();
  float *output_data = output.mutable_data_ptr<float>();
  const std::span<const std::uint16_t> weights(weight_data, weight_elements);
  for (std::int64_t position = 0; position < positions; ++position) {
    matrix.cpu_kernel->apply(
        weights,
        std::span<const float>(input_data + position * matrix.columns,
                               matrix.columns),
        std::span<float>(output_data + position * matrix.rows, matrix.rows),
        matrix.rows, matrix.columns);
  }
  return output;
}

at::Tensor original_bf16_linear(const at::Tensor &input,
                                const OriginalBf16Matrix &matrix) {
  const at::Device device = original_bf16_device(matrix);
  if (!original_bf16_matrix_matches(matrix, device, matrix.rows,
                                    matrix.columns) ||
      !input.defined() || input.device() != device ||
      input.scalar_type() != at::kFloat || !input.is_contiguous() ||
      (input.dim() != 2 && input.dim() != 3) || input.size(-1) <= 0 ||
      static_cast<std::size_t>(input.size(-1)) != matrix.columns ||
      (input.dim() == 3 && input.size(0) != 1)) {
    throw std::invalid_argument(
        "original-BF16 projection requires contiguous selected-device fp32 "
        "[T,C] or [1,T,C]");
  }
  const std::int64_t positions =
      input.dim() == 2 ? input.size(0) : input.size(1);
  if (positions < 1 ||
      positions > static_cast<std::int64_t>(kBf16CpuMaximumPositions) ||
      input.numel() !=
          positions * static_cast<std::int64_t>(matrix.columns)) {
    throw std::invalid_argument(
        "original-BF16 projection position count is outside 1..64");
  }
  if (device.is_cpu()) {
    return bf16_cpu_linear(input, matrix);
  }
  if (!matrix.is_owned() || matrix.owned_storage == nullptr ||
      matrix.owned_storage->projection_backend == nullptr) {
    throw std::logic_error(
        "accelerator original-BF16 projection lost its exact backend");
  }
  const at::Tensor flat = input.dim() == 2
      ? input
      : input.view({positions, static_cast<std::int64_t>(matrix.columns)});
  at::Tensor output = matrix.owned_storage->projection_backend->linear(
      flat, matrix.owned_element_offset, matrix.rows, matrix.columns);
  if (!output.defined() || output.device() != device ||
      output.scalar_type() != at::kFloat || !output.is_contiguous() ||
      output.sizes() !=
          at::IntArrayRef(
              {positions, static_cast<std::int64_t>(matrix.rows)})) {
    throw std::runtime_error(
        "exact accelerator BF16 backend returned an invalid projection");
  }
  return input.dim() == 2
      ? output
      : output.view(
            {1, positions, static_cast<std::int64_t>(matrix.rows)});
}

} // namespace deltafin::provider_internal
