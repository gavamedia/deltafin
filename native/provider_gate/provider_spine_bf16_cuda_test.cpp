#include "provider_spine_bf16_cuda.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(DELTAFIN_HAVE_CUDA_SPINE_BF16_V1)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#if defined(_WIN32)
#include <malloc.h>
#include <windows.h>
#else
#include <unistd.h>
#endif
#endif

namespace {

using deltafin::provider_internal::CudaSpineBf16DeviceSlab;
using deltafin::provider_internal::CudaSpineBf16HostPolicy;
using deltafin::provider_internal::CudaSpineBf16HostSlab;
using deltafin::provider_internal::CudaSpineBf16LifetimeState;
using deltafin::provider_internal::CudaSpineBf16MatrixView;
using deltafin::provider_internal::CudaSpineBf16Projector;
using deltafin::provider_internal::CudaSpineBf16SourceKind;
using deltafin::provider_internal::cuda_spine_bf16_device_storage_dtype_supported;
using deltafin::provider_internal::cuda_spine_bf16_compiled;
using deltafin::provider_internal::cuda_spine_bf16_reference_decode;
using deltafin::provider_internal::validate_cuda_spine_bf16_host_view;

template <typename Function>
void require_failure(Function&& function, const char* expected) {
  try {
    function();
  } catch (const std::exception& error) {
    if (std::string(error.what()).find(expected) == std::string::npos) {
      throw std::runtime_error(
          std::string("unexpected CUDA RAW_BF16 failure: ") + error.what());
    }
    return;
  }
  throw std::runtime_error(
      std::string("expected CUDA RAW_BF16 failure containing: ") + expected);
}

void portable_decode_and_abi_test() {
  static_assert(static_cast<std::uint32_t>(
                    CudaSpineBf16SourceKind::DetachedStaged) == 0);
  static_assert(static_cast<std::uint32_t>(
                    CudaSpineBf16SourceKind::BorrowedDirectHostAts) == 1);
  static_assert(static_cast<std::uint32_t>(
                    CudaSpineBf16SourceKind::DetachedDeviceOwned) == 2);
  static_assert(static_cast<std::uint32_t>(CudaSpineBf16HostPolicy::Auto) == 0);
  static_assert(
      static_cast<std::uint32_t>(CudaSpineBf16HostPolicy::StageOnly) == 1);

  for (std::uint32_t bits = 0; bits <= 0xffffU; ++bits) {
    const float decoded =
        cuda_spine_bf16_reference_decode(static_cast<std::uint16_t>(bits));
    if (std::bit_cast<std::uint32_t>(decoded) != (bits << 16)) {
      throw std::runtime_error(
          "portable CUDA RAW_BF16 reference lost an encoding");
    }
  }
}

void portable_composite_lifetime_test() {
  CudaSpineBf16LifetimeState state;
  if (!state.open() || state.sealed() || state.reclaimed() ||
      state.submissions() != 0) {
    throw std::runtime_error("CUDA RAW_BF16 lifetime did not begin open");
  }
  require_failure([&] { state.seal(); }, "needs submitted work");
  state.note_submission();
  state.note_submission();
  if (!state.open() || state.submissions() != 2) {
    throw std::runtime_error(
        "one CUDA preparation did not retain multiple submissions");
  }
  state.seal();
  state.require_reclaim_query();
  if (!state.sealed() || state.open()) {
    throw std::runtime_error("CUDA RAW_BF16 lifetime did not seal once");
  }
  require_failure([&] { state.note_submission(); }, "open prepared layer");
  require_failure([&] { state.seal(); }, "consume-once");
  state.complete_reclaim();
  if (!state.reclaimed() || state.submissions() != 2) {
    throw std::runtime_error(
        "CUDA RAW_BF16 composite reclaim lost its submission count");
  }
  require_failure([&] { state.require_reclaim_query(); }, "sealed");

  CudaSpineBf16LifetimeState aborted;
  aborted.note_submission();
  aborted.complete_abort();
  if (!aborted.reclaimed()) {
    throw std::runtime_error("CUDA RAW_BF16 abort did not reclaim");
  }
}

void portable_offset_descriptor_test() {
  alignas(256) std::array<std::byte, 2048> storage{};
  const CudaSpineBf16HostSlab slab{
      .allocation_base = storage.data(),
      .logical_slab_bytes = 1024,
      .allocation_bytes = storage.size()};
  const CudaSpineBf16MatrixView valid{
      .matrix_byte_offset = 256,
      .logical_bytes = 4 * 16 * sizeof(std::uint16_t),
      .rows = 4,
      .columns = 16};
  validate_cuda_spine_bf16_host_view(slab, valid);
  auto odd = valid;
  odd.matrix_byte_offset = 257;
  require_failure(
      [&] { validate_cuda_spine_bf16_host_view(slab, odd); }, "alignment");
  auto escaped = valid;
  escaped.matrix_byte_offset = 960;
  require_failure(
      [&] { validate_cuda_spine_bf16_host_view(slab, escaped); }, "escapes");
  auto wrong_bytes = valid;
  ++wrong_bytes.logical_bytes;
  require_failure(
      [&] { validate_cuda_spine_bf16_host_view(slab, wrong_bytes); },
      "escapes");
  auto null_slab = slab;
  null_slab.allocation_base = nullptr;
  require_failure(
      [&] { validate_cuda_spine_bf16_host_view(null_slab, valid); },
      "host slab");
  auto odd_slab = slab;
  --odd_slab.logical_slab_bytes;
  require_failure(
      [&] { validate_cuda_spine_bf16_host_view(odd_slab, valid); },
      "host slab");
}

void portable_device_storage_dtype_test() {
  for (const at::ScalarType supported :
       {at::kByte, at::kBFloat16, at::kUInt16, at::kShort}) {
    if (!cuda_spine_bf16_device_storage_dtype_supported(supported)) {
      throw std::runtime_error(
          "CUDA RAW_BF16 rejected an exact device carrier dtype");
    }
  }
  for (const at::ScalarType rejected :
       {at::kFloat, at::kHalf, at::kInt, at::kLong}) {
    if (cuda_spine_bf16_device_storage_dtype_supported(rejected)) {
      throw std::runtime_error(
          "CUDA RAW_BF16 accepted a numerically converted device dtype");
    }
  }
}

void portable_capability_test() {
  CudaSpineBf16Projector projector{at::Device(at::kCPU)};
  std::array<const deltafin::provider_internal::CudaSpineBf16Capability*, 8>
      reports{};
  std::vector<std::thread> callers;
  callers.reserve(reports.size());
  for (std::size_t index = 0; index < reports.size(); ++index) {
    callers.emplace_back([&projector, &reports, index] {
      reports[index] = &projector.capability();
    });
  }
  for (std::thread& caller : callers) caller.join();
  const auto& capability = *reports.front();
  for (const auto* report : reports) {
    if (report != reports.front()) {
      throw std::runtime_error(
          "thread-safe CUDA capability qualification returned split state");
    }
  }
#if !defined(DELTAFIN_HAVE_CUDA_SPINE_BF16_V1)
  if (cuda_spine_bf16_compiled() || capability.compiled ||
      capability.available || capability.direct_host_ats ||
      capability.direct_host_runtime_activation_qualified ||
      capability.maximum_positions != 0 ||
      capability.detail.find("not compiled") == std::string::npos) {
    throw std::runtime_error(
        "portable CUDA RAW_BF16 stub advertised physical support");
  }
#else
  if (!cuda_spine_bf16_compiled() || !capability.compiled ||
      capability.available || capability.direct_host_ats ||
      capability.direct_host_runtime_activation_qualified ||
      capability.maximum_positions != 0 ||
      capability.detail.find("indexed CUDA device") == std::string::npos) {
    throw std::runtime_error(
        "CUDA-built RAW_BF16 provider accepted a CPU device");
  }
#endif
  require_failure(
      [&] {
        static_cast<void>(projector.prepare_host_layer(
            CudaSpineBf16HostSlab{}, CudaSpineBf16HostPolicy::Auto));
      },
#if !defined(DELTAFIN_HAVE_CUDA_SPINE_BF16_V1)
      "not compiled"
#else
      "unavailable"
#endif
  );
}

#if defined(DELTAFIN_HAVE_CUDA_SPINE_BF16_V1)

#if defined(_WIN32)
struct FreeDeleter {
  void operator()(void* pointer) const noexcept { _aligned_free(pointer); }
};
#else
struct FreeDeleter {
  void operator()(void* pointer) const noexcept { std::free(pointer); }
};
#endif

std::size_t round_up(const std::size_t value, const std::size_t alignment) {
  if (value > SIZE_MAX - (alignment - 1)) {
    throw std::runtime_error("test allocation span overflow");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

struct MatrixFixture {
  CudaSpineBf16MatrixView view;
  std::vector<std::uint16_t> weights;
  std::vector<float> activation;
  std::vector<float> expected;
};

MatrixFixture make_matrix(const std::size_t offset,
                          const std::int64_t rows,
                          const std::int64_t columns,
                          const std::size_t salt) {
  constexpr std::array<std::uint16_t, 16> finite_patterns{
      0x0000, 0x0001, 0x007f, 0x0080, 0x3d00, 0x3e80, 0x3f00, 0x3f80,
      0x4000, 0x4040, 0x4080, 0x8000, 0xbd00, 0xbe80, 0xbf00, 0xbf80};
  MatrixFixture fixture;
  fixture.view = CudaSpineBf16MatrixView{
      .matrix_byte_offset = offset,
      .logical_bytes = static_cast<std::size_t>(
          rows * columns * static_cast<std::int64_t>(sizeof(std::uint16_t))),
      .rows = rows,
      .columns = columns};
  fixture.weights.resize(static_cast<std::size_t>(rows * columns));
  for (std::size_t index = 0; index < fixture.weights.size(); ++index) {
    fixture.weights[index] =
        finite_patterns[(index * 13 + salt) % finite_patterns.size()];
  }
  fixture.activation.resize(static_cast<std::size_t>(columns));
  for (std::size_t index = 0; index < fixture.activation.size(); ++index) {
    fixture.activation[index] =
        static_cast<float>(static_cast<int>((index + salt) % 29) - 14) /
        31.0F;
  }
  fixture.expected.resize(static_cast<std::size_t>(rows));
  for (std::int64_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (std::int64_t column = 0; column < columns; ++column) {
      const float weight = cuda_spine_bf16_reference_decode(
          fixture.weights[static_cast<std::size_t>(row * columns + column)]);
      sum += static_cast<double>(weight) * static_cast<double>(
          fixture.activation[static_cast<std::size_t>(column)]);
    }
    fixture.expected[static_cast<std::size_t>(row)] =
        static_cast<float>(sum);
  }
  return fixture;
}

at::Tensor cuda_activation(const MatrixFixture& fixture,
                           const std::int64_t positions,
                           const at::Device& device) {
  std::vector<float> values(
      static_cast<std::size_t>(positions * fixture.view.columns));
  for (std::int64_t position = 0; position < positions; ++position) {
    std::memcpy(values.data() + position * fixture.view.columns,
                fixture.activation.data(),
                static_cast<std::size_t>(fixture.view.columns) *
                    sizeof(float));
  }
  const at::Tensor host = at::from_blob(
      values.data(), {positions, fixture.view.columns},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  return host.clone().to(device);
}

void require_close(const at::Tensor& output,
                   const std::vector<float>& expected,
                   const std::int64_t positions) {
  const at::Tensor cpu = output.to(at::kCPU).contiguous();
  if (cpu.scalar_type() != at::kFloat ||
      cpu.dim() != 2 || cpu.size(0) != positions ||
      cpu.size(1) != static_cast<std::int64_t>(expected.size())) {
    throw std::runtime_error("CUDA RAW_BF16 parity output has invalid shape");
  }
  const float* found = cpu.const_data_ptr<float>();
  for (std::int64_t position = 0; position < positions; ++position) {
    for (std::size_t row = 0; row < expected.size(); ++row) {
      const float tolerance =
          2.0e-5F + 2.0e-5F * std::abs(expected[row]);
      const float candidate =
          found[static_cast<std::size_t>(position) * expected.size() + row];
      if (!std::isfinite(candidate) ||
          std::abs(candidate - expected[row]) > tolerance) {
        throw std::runtime_error(
            "CUDA RAW_BF16 projection exceeded its FP32 parity bound");
      }
    }
  }
}

void physical_cuda_device_test(const int device_index) {
  const at::Device device(
      at::kCUDA, static_cast<c10::DeviceIndex>(device_index));
#if defined(_WIN32)
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  const long raw_page = static_cast<long>(system_info.dwPageSize);
#else
  const long raw_page = sysconf(_SC_PAGESIZE);
#endif
  if (raw_page <= 0 ||
      !std::has_single_bit(static_cast<std::size_t>(raw_page))) {
    throw std::runtime_error("test could not prove the CUDA host page size");
  }
  const std::size_t page = static_cast<std::size_t>(raw_page);
  MatrixFixture first = make_matrix(256, 19, 521, 5);
  const std::size_t second_offset =
      round_up(first.view.matrix_byte_offset + first.view.logical_bytes, 256);
  MatrixFixture second = make_matrix(second_offset, 7, 129, 11);
  const std::size_t logical_slab_bytes =
      second.view.matrix_byte_offset + second.view.logical_bytes;
  const std::size_t allocation_bytes = round_up(logical_slab_bytes, page);
  void* allocated = nullptr;
#if defined(_WIN32)
  allocated = _aligned_malloc(allocation_bytes, page);
  if (allocated == nullptr) {
#else
  if (posix_memalign(&allocated, page, allocation_bytes) != 0 ||
      allocated == nullptr) {
#endif
    throw std::runtime_error(
        "could not allocate page-aligned CUDA RAW_BF16 test slab");
  }
  std::unique_ptr<void, FreeDeleter> allocation(allocated);
  auto* slab_bytes = static_cast<std::byte*>(allocation.get());
  std::memset(slab_bytes, 0x5a, allocation_bytes);
  std::memcpy(slab_bytes + first.view.matrix_byte_offset,
              first.weights.data(), first.view.logical_bytes);
  std::memcpy(slab_bytes + second.view.matrix_byte_offset,
              second.weights.data(), second.view.logical_bytes);
  std::vector<std::byte> original_slab(logical_slab_bytes);
  std::memcpy(original_slab.data(), slab_bytes, logical_slab_bytes);

  const CudaSpineBf16HostSlab slab{
      .allocation_base = slab_bytes,
      .logical_slab_bytes = logical_slab_bytes,
      .allocation_bytes = allocation_bytes};
  const at::Tensor first_t1 = cuda_activation(first, 1, device);
  const at::Tensor first_t2 = cuda_activation(first, 2, device);
  const at::Tensor first_t9 = cuda_activation(first, 9, device);
  const at::Tensor first_t64 = cuda_activation(first, 64, device);
  const at::Tensor second_t9 = cuda_activation(second, 9, device);

  CudaSpineBf16Projector projector(device);
  const auto& capability = projector.capability();
  if (!capability.compiled || !capability.available ||
      capability.device_index != device_index || capability.compute_major <= 0 ||
      capability.maximum_positions != 64 ||
      capability.direct_host_runtime_activation_qualified) {
    throw std::runtime_error(
        std::string("physical CUDA RAW_BF16 gate failed: ") +
        capability.detail);
  }

  // Prepare once, then overwrite the reader allocation before either of two
  // projections. Both views must consume the provider-owned exact BF16 copy,
  // and both are covered by one public seal/reclaim.
  auto staged = projector.prepare_host_layer(
      slab, CudaSpineBf16HostPolicy::StageOnly);
  if (staged->source_kind() != CudaSpineBf16SourceKind::DetachedStaged) {
    throw std::runtime_error("staged CUDA BF16 slab borrowed its source");
  }
  const auto captured =
      c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(
          device_index));
  const auto alternate = c10::cuda::getStreamFromPool(
      false, static_cast<c10::DeviceIndex>(device_index));
  if (alternate.stream() != captured.stream()) {
    c10::cuda::CUDAStreamGuard alternate_guard(alternate);
    require_failure(
        [&] { static_cast<void>(staged->submit(first.view, first_t2)); },
        "captured CUDA stream");
  }
  std::memset(slab_bytes, 0, logical_slab_bytes);
  const at::Tensor staged_t1 = staged->submit_t1(first.view, first_t1);
  const at::Tensor staged_t2 = staged->submit(first.view, first_t2);
  const at::Tensor staged_t9 = staged->submit(first.view, first_t9);
  const at::Tensor staged_t64 = staged->submit(first.view, first_t64);
  const at::Tensor staged_second = staged->submit(second.view, second_t9);
  if (staged->submissions() != 5) {
    throw std::runtime_error(
        "one CUDA BF16 slab preparation did not serve all projections");
  }
  require_failure([&] { static_cast<void>(staged->try_reclaim()); }, "sealed");
  staged->seal();
  require_close(staged_t1, first.expected, 1);
  require_close(staged_t2, first.expected, 2);
  require_close(staged_t9, first.expected, 9);
  require_close(staged_t64, first.expected, 64);
  require_close(staged_second, second.expected, 9);
  if (!staged->try_reclaim() || !staged->reclaimed()) {
    throw std::runtime_error(
        "completed CUDA RAW_BF16 staged layer did not reclaim");
  }

  // Device-owned/global path: one resident BF16 slab, two views, no second
  // BF16 allocation and no FP32 weight allocation.
  static_assert(sizeof(c10::BFloat16) == sizeof(std::uint16_t));
  at::Tensor bf16_host = at::empty(
      {static_cast<std::int64_t>(logical_slab_bytes / 2)},
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
  std::memcpy(bf16_host.data_ptr<c10::BFloat16>(), original_slab.data(),
              logical_slab_bytes);
  const at::Tensor bf16_device = bf16_host.to(device);
  auto device_owned = projector.prepare_device_layer(
      CudaSpineBf16DeviceSlab{
          .storage = bf16_device,
          .logical_slab_bytes = logical_slab_bytes});
  at::Tensor device_cross_stream;
  if (alternate.stream() != captured.stream()) {
    const cudaError_t ready = cudaStreamSynchronize(captured.stream());
    if (ready != cudaSuccess) {
      throw std::runtime_error(
          "could not establish CUDA cross-stream test input readiness");
    }
    std::exception_ptr worker_failure;
    std::thread worker([&] {
      try {
        c10::cuda::CUDAStreamGuard alternate_guard(alternate);
        device_cross_stream =
            device_owned->submit(first.view, first_t2);
        const cudaError_t completed =
            cudaStreamSynchronize(alternate.stream());
        if (completed != cudaSuccess) {
          throw std::runtime_error(
              "CUDA device-owned alternate stream did not complete");
        }
      } catch (...) {
        worker_failure = std::current_exception();
      }
    });
    worker.join();
    if (worker_failure != nullptr) {
      std::rethrow_exception(worker_failure);
    }
    require_close(device_cross_stream, first.expected, 2);
  }
  const at::Tensor device_first =
      device_owned->submit(first.view, first_t64);
  const at::Tensor device_second =
      device_owned->submit(second.view, second_t9);
  if (device_owned->source_kind() !=
      CudaSpineBf16SourceKind::DetachedDeviceOwned) {
    throw std::runtime_error(
        "device-owned CUDA BF16 slab reported the wrong source kind");
  }
  device_owned->seal();
  require_close(device_first, first.expected, 64);
  require_close(device_second, second.expected, 9);
  if (!device_owned->try_reclaim()) {
    throw std::runtime_error(
        "synchronized device-owned CUDA BF16 layer did not reclaim");
  }

  // Runtime startup physically tests ATS, but actual use stays benchmark-only
  // until this exact device/shape wins a real-shape staged crossover gate.
  if (capability.direct_host_ats) {
    std::memcpy(slab_bytes, original_slab.data(), logical_slab_bytes);
    auto direct = projector.prepare_direct_host_layer_for_benchmark(slab);
    const at::Tensor direct_first =
        direct->submit(first.view, first_t2);
    const at::Tensor direct_second =
        direct->submit(second.view, second_t9);
    if (direct->source_kind() !=
        CudaSpineBf16SourceKind::BorrowedDirectHostAts) {
      throw std::runtime_error(
          "qualified CUDA ATS slab did not report a borrowed source");
    }
    direct->seal();
    require_close(direct_first, first.expected, 2);
    require_close(direct_second, second.expected, 9);
    if (!direct->try_reclaim()) {
      throw std::runtime_error(
          "synchronized CUDA ATS slab did not become reclaimable");
    }
    std::cout << "provider_spine_bf16_cuda.ats.device=" << device_index
              << "=ok\n";
  }
  std::cout << "provider_spine_bf16_cuda.physical.device=" << device_index
            << "=ok\n";
}

void physical_cuda_test() {
  int devices = 0;
  const cudaError_t probe = cudaGetDeviceCount(&devices);
  if (probe != cudaSuccess || devices == 0) {
    static_cast<void>(cudaGetLastError());
    std::cout <<
        "provider_spine_bf16_cuda.physical=skipped(no visible CUDA device)\n";
    return;
  }
  physical_cuda_device_test(0);
  if (devices > 1) {
    physical_cuda_device_test(devices - 1);
  }
}

#endif

}  // namespace

int main() {
  try {
    portable_decode_and_abi_test();
    portable_composite_lifetime_test();
    portable_offset_descriptor_test();
    portable_device_storage_dtype_test();
    portable_capability_test();
#if defined(DELTAFIN_HAVE_CUDA_SPINE_BF16_V1)
    physical_cuda_test();
#endif
    std::cout << "provider_spine_bf16_cuda=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_spine_bf16_cuda_test failed: " << error.what()
              << '\n';
    return 1;
  }
}
