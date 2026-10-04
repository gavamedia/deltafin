#include "provider_abi.h"
#include "provider_device.h"
#include "provider_kda.h"
#include "provider_spine_debug.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace {

// Storage aligned to `alignment`, released the way its allocator requires:
// `free` after posix_memalign, `_aligned_free` after `_aligned_malloc` (the
// Windows CRT cannot free an aligned block through `free`).
struct AlignedRelease {
  void operator()(std::uint8_t* pointer) const noexcept {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
  }
};
using AlignedBytes = std::unique_ptr<std::uint8_t, AlignedRelease>;

AlignedBytes allocate_aligned_bytes(const std::size_t alignment,
                                    const std::size_t bytes) {
#if defined(_WIN32)
  void* raw = _aligned_malloc(bytes, alignment);
  if (raw == nullptr) {
    return AlignedBytes();
  }
#else
  void* raw = nullptr;
  if (posix_memalign(&raw, alignment, bytes) != 0 || raw == nullptr) {
    return AlignedBytes();
  }
#endif
  return AlignedBytes(static_cast<std::uint8_t*>(raw));
}

struct Options {
  std::uint32_t device = DELTAFIN_PROVIDER_DEVICE_AUTO_V1;
  std::uint32_t device_index = 0;
  bool require_packed_int8 = false;
  bool split_boundary = false;
  bool spine_binding = false;
  bool kda_tape = false;
  std::uint64_t packed_rows = 32;
  std::uint64_t packed_columns = 32;
};

void usage(const char* program) {
  std::cout << "usage: " << program
            << " [--device auto|cpu|mps|cuda|cuda:N]"
               " [--require-packed-int8] [--packed-shape ROWSxCOLS]"
               " [--split-boundary] [--spine-binding] [--kda-tape]\n";
}

std::uint64_t parse_dimension(const std::string_view value) {
  std::size_t parsed = 0;
  const std::uint64_t result = std::stoull(std::string(value), &parsed);
  if (parsed != value.size() || result == 0) {
    throw std::invalid_argument("packed dimensions must be positive integers");
  }
  return result;
}

void parse_device(const std::string_view value, Options& options) {
  if (value == "auto") {
    options.device = DELTAFIN_PROVIDER_DEVICE_AUTO_V1;
  } else if (value == "cpu") {
    options.device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  } else if (value == "mps") {
    options.device = DELTAFIN_PROVIDER_DEVICE_MPS_V1;
  } else if (value == "cuda") {
    options.device = DELTAFIN_PROVIDER_DEVICE_CUDA_V1;
  } else if (value.starts_with("cuda:")) {
    options.device = DELTAFIN_PROVIDER_DEVICE_CUDA_V1;
    const std::string index_text(value.substr(5));
    std::size_t parsed = 0;
    const std::uint64_t index = std::stoull(index_text, &parsed);
    if (parsed != index_text.size()) {
      throw std::invalid_argument("CUDA device index must be a non-negative integer");
    }
    if (index > UINT32_MAX) {
      throw std::invalid_argument("CUDA device index is too large");
    }
    options.device_index = static_cast<std::uint32_t>(index);
  } else {
    throw std::invalid_argument("device must be auto, cpu, mps, cuda, or cuda:N");
  }
}

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (argument == "--require-packed-int8") {
      options.require_packed_int8 = true;
      continue;
    }
    if (argument == "--split-boundary") {
      options.split_boundary = true;
      continue;
    }
    if (argument == "--spine-binding") {
      options.spine_binding = true;
      continue;
    }
    if (argument == "--kda-tape") {
      options.kda_tape = true;
      continue;
    }
    if (argument == "--device" || argument == "--packed-shape") {
      if (++index >= argc) {
        throw std::invalid_argument(std::string(argument) + " requires a value");
      }
      const std::string_view value(argv[index]);
      if (argument == "--device") {
        parse_device(value, options);
      } else {
        const std::size_t separator = value.find('x');
        if (separator == std::string_view::npos) {
          throw std::invalid_argument("--packed-shape must be ROWSxCOLS");
        }
        options.packed_rows = parse_dimension(value.substr(0, separator));
        options.packed_columns = parse_dimension(value.substr(separator + 1));
      }
      continue;
    }
    throw std::invalid_argument("unknown argument: " + std::string(argument));
  }
  return options;
}

const char* device_name(const std::uint32_t device) {
  switch (device) {
    case DELTAFIN_PROVIDER_DEVICE_CPU_V1:
      return "cpu";
    case DELTAFIN_PROVIDER_DEVICE_MPS_V1:
      return "mps";
    case DELTAFIN_PROVIDER_DEVICE_CUDA_V1:
      return "cuda";
    default:
      return "unknown";
  }
}

void print_check(const char* name, const std::uint32_t mask,
                 const DeltafinProviderCanaryReportV1& report) {
  const bool passed = (report.passed_checks & mask) != 0;
  std::cout << "check." << name << '=' << (passed ? "PASS" : "FAIL") << '\n';
}

void require_success(const std::int32_t status, const char* error,
                     const char* operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) + ": " + error);
  }
}

void release_resource(
    const DeltafinProviderSessionHandleV1 session, const std::uint64_t resource,
    std::int32_t (*release)(const DeltafinProviderResourceRequestV1*, char*,
                            std::size_t),
    const char* operation) {
  DeltafinProviderResourceRequestV1 request = {};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.session = session;
  request.resource = resource;
  char error[1024] = {};
  require_success(release(&request, error, sizeof(error)), error, operation);
}

void run_target_pilot_admission() {
  char error[1024] = {};
  DeltafinProviderSessionRequestV1 session_request = {};
  session_request.struct_size = sizeof(session_request);
  session_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  session_request.requested_device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  session_request.max_route_positions = 1;
  DeltafinProviderSessionReportV1 session_report = {};
  session_report.struct_size = sizeof(session_report);
  require_success(deltafin_provider_session_create_v1(
                      &session_request, &session_report, error, sizeof(error)),
                  error, "create target PILOT admission session");

  DeltafinProviderResourceRequestV1 request = {};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.session = session_report.session;
  DeltafinProviderTargetPilotEnableReportV1 report = {};
  report.struct_size = sizeof(report);

  request.reserved[0] = 1;
  if (deltafin_provider_target_pilot_enable_v1(
          &request, &report, error, sizeof(error)) == 0) {
    throw std::runtime_error(
        "target PILOT admission accepted a nonzero reserved field");
  }
  request.reserved[0] = 0;
  std::memset(error, 0, sizeof(error));

  report.struct_size = sizeof(report) - 8;
  if (deltafin_provider_target_pilot_enable_v1(
          &request, &report, error, sizeof(error)) == 0) {
    throw std::runtime_error(
        "target PILOT admission accepted a stale report layout");
  }
  report = {};
  report.struct_size = sizeof(report);
  std::memset(error, 0, sizeof(error));

  require_success(deltafin_provider_target_pilot_enable_v1(
                      &request, &report, error, sizeof(error)),
                  error, "enable target PILOT roster");
  bool reserved_zero = true;
  for (const std::uint64_t value : report.reserved) {
    reserved_zero = reserved_zero && value == 0;
  }
  if (report.abi_version != DELTAFIN_PROVIDER_ABI_VERSION ||
      report.session != session_report.session || report.enabled != 1 ||
      report.layer_capacity !=
          DELTAFIN_PROVIDER_TARGET_PILOT_LAYER_CAPACITY_V1 ||
      report.reserve_bytes != DELTAFIN_PROVIDER_TARGET_PILOT_RESERVE_BYTES_V1 ||
      !reserved_zero) {
    throw std::runtime_error(
        "target PILOT admission returned an invalid resource contract");
  }

  std::memset(error, 0, sizeof(error));
  if (deltafin_provider_target_pilot_enable_v1(
          &request, &report, error, sizeof(error)) == 0) {
    throw std::runtime_error(
        "target PILOT admission was not one-shot immutable state");
  }

  release_resource(session_report.session, 0,
                   deltafin_provider_session_destroy_v1,
                   "destroy target PILOT admission session");
}

void run_mla_expanded_only_admission() {
  char error[1024] = {};
  DeltafinProviderSessionRequestV1 session_request = {};
  session_request.struct_size = sizeof(session_request);
  session_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  session_request.requested_device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  session_request.max_route_positions = 1;
  DeltafinProviderSessionReportV1 session_report = {};
  session_report.struct_size = sizeof(session_report);
  require_success(deltafin_provider_session_create_v1(
                      &session_request, &session_report, error, sizeof(error)),
                  error, "create MLA representation-gate session");

  DeltafinProviderMlaCacheCreateV1 request = {};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.session = session_report.session;
  request.layer_index = 3;
  request.flags = 1;
  DeltafinProviderMlaCacheReportV1 report = {};
  report.struct_size = sizeof(report);
  if (deltafin_provider_mla_cache_create_v1(
          &request, &report, error, sizeof(error)) == 0) {
    throw std::runtime_error(
        "MLA cache ABI accepted an unqualified representation flag");
  }

  request.flags = 0;
  report = {};
  report.struct_size = sizeof(report);
  std::memset(error, 0, sizeof(error));
  require_success(deltafin_provider_mla_cache_create_v1(
                      &request, &report, error, sizeof(error)),
                  error, "create expanded-only MLA cache");
  if (report.abi_version != DELTAFIN_PROVIDER_ABI_VERSION ||
      report.cache == 0 || report.layer_index != request.layer_index ||
      report.flags != 0 || report.version != 0 || report.length != 0 ||
      report.capacity != 0) {
    throw std::runtime_error(
        "expanded-only MLA cache returned an invalid initial boundary");
  }
  release_resource(session_report.session, report.cache,
                   deltafin_provider_mla_cache_release_v1,
                   "release expanded-only MLA cache");
  release_resource(session_report.session, 0,
                   deltafin_provider_session_destroy_v1,
                   "destroy MLA representation-gate session");
}

void run_split_boundary(const Options& options) {
  char error[1024] = {};
  DeltafinProviderSessionRequestV1 session_request = {};
  session_request.struct_size = sizeof(session_request);
  session_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  session_request.requested_device = options.device;
  session_request.device_index = options.device_index;
  session_request.flags = DELTAFIN_PROVIDER_SESSION_SYNTHETIC_SPLIT_V1;
  session_request.max_route_positions = 2;
  session_request.synthetic_hidden_columns = 32;
  session_request.synthetic_experts = 32;
  DeltafinProviderSessionReportV1 session_report = {};
  session_report.struct_size = sizeof(session_report);
  require_success(deltafin_provider_session_create_v1(
                      &session_request, &session_report, error, sizeof(error)),
                  error, "create split session");

  float hidden[64] = {};
  float initial_cache[64] = {};
  float expert[64] = {};
  for (std::size_t index = 0; index < 64; ++index) {
    hidden[index] = (static_cast<float>(index) - 17.0F) / 64.0F;
    initial_cache[index] = 0.25F;
    expert[index] = 1.0F;
  }
  const auto upload = [&](const float* values) {
    DeltafinProviderTensorUploadF32V1 request = {};
    request.struct_size = sizeof(request);
    request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    request.session = session_report.session;
    request.rows = 2;
    request.columns = 32;
    request.data = values;
    request.element_count = 64;
    DeltafinProviderTensorReportV1 report = {};
    report.struct_size = sizeof(report);
    require_success(deltafin_provider_tensor_upload_f32_v1(
                        &request, &report, error, sizeof(error)),
                    error, "upload split tensor");
    return report.tensor;
  };
  const auto hidden_handle = upload(hidden);
  const auto expert_handle = upload(expert);

  DeltafinProviderCacheCreateF32V1 cache_request = {};
  cache_request.struct_size = sizeof(cache_request);
  cache_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  cache_request.session = session_report.session;
  cache_request.rows = 2;
  cache_request.columns = 32;
  cache_request.initial_data = initial_cache;
  cache_request.element_count = 64;
  DeltafinProviderCacheReportV1 cache_report = {};
  cache_report.struct_size = sizeof(cache_report);
  require_success(deltafin_provider_cache_create_f32_v1(
                      &cache_request, &cache_report, error, sizeof(error)),
                  error, "create split cache");

  DeltafinProviderPrepareLayerRequestV1 prepare = {};
  prepare.struct_size = sizeof(prepare);
  prepare.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  prepare.session = session_report.session;
  prepare.hidden = hidden_handle;
  prepare.cache = cache_report.cache;
  DeltafinProviderRouteMailboxV1 mailbox = {};
  mailbox.struct_size = sizeof(mailbox);
  require_success(deltafin_provider_prepare_layer_v1(
                      &prepare, &mailbox, error, sizeof(error)),
                  error, "prepare split layer");
  if (mailbox.positions != 2 ||
      mailbox.top_k != DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 ||
      mailbox.edge_count != 2 * DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 ||
      mailbox.ticket == 0 || mailbox.cache_version != 0) {
    throw std::runtime_error("split route mailbox has an invalid contract");
  }

  // Preparing must not publish speculative cache state.
  float cache_before_finish[64] = {};
  DeltafinProviderCacheReadF32V1 cache_read = {};
  cache_read.struct_size = sizeof(cache_read);
  cache_read.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  cache_read.session = session_report.session;
  cache_read.cache = cache_report.cache;
  cache_read.destination = cache_before_finish;
  cache_read.element_capacity = 64;
  DeltafinProviderCacheReportV1 cache_read_report = {};
  cache_read_report.struct_size = sizeof(cache_read_report);
  require_success(deltafin_provider_cache_read_f32_v1(
                      &cache_read, &cache_read_report, error, sizeof(error)),
                  error, "read staged split cache");
  if (std::memcmp(cache_before_finish, initial_cache,
                  sizeof(initial_cache)) != 0 ||
      cache_read_report.version != 0) {
    throw std::runtime_error("prepare committed split cache before finish");
  }

  DeltafinProviderFinishLayerRequestV1 finish = {};
  finish.struct_size = sizeof(finish);
  finish.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  finish.session = session_report.session;
  finish.ticket = mailbox.ticket;
  finish.expert_output = expert_handle;
  DeltafinProviderFinishLayerReportV1 finish_report = {};
  finish_report.struct_size = sizeof(finish_report);
  require_success(deltafin_provider_finish_layer_v1(
                      &finish, &finish_report, error, sizeof(error)),
                  error, "finish split layer");
  if (finish_report.output == 0 || finish_report.positions != 2 ||
      finish_report.hidden_columns != 32 ||
      finish_report.committed_cache_version != 1) {
    throw std::runtime_error("split finish report has an invalid contract");
  }

  float output[64] = {};
  DeltafinProviderTensorReadF32V1 output_read = {};
  output_read.struct_size = sizeof(output_read);
  output_read.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  output_read.session = session_report.session;
  output_read.tensor = finish_report.output;
  output_read.destination = output;
  output_read.element_capacity = 64;
  require_success(deltafin_provider_tensor_read_f32_v1(
                      &output_read, error, sizeof(error)),
                  error, "read split output");
  for (std::size_t index = 0; index < 64; ++index) {
    const float expected = hidden[index] + initial_cache[index] + expert[index];
    if (output[index] != expected) {
      throw std::runtime_error("split output differs from the exact fp32 tape");
    }
  }

  release_resource(session_report.session, finish_report.output,
                   deltafin_provider_tensor_release_v1,
                   "release split output");
  release_resource(session_report.session, expert_handle,
                   deltafin_provider_tensor_release_v1,
                   "release split expert tensor");
  release_resource(session_report.session, hidden_handle,
                   deltafin_provider_tensor_release_v1,
                   "release split hidden tensor");
  release_resource(session_report.session, cache_report.cache,
                   deltafin_provider_cache_release_v1,
                   "release split cache");
  release_resource(session_report.session, 0,
                   deltafin_provider_session_destroy_v1,
                   "destroy split session");
}

void run_spine_binding(const Options& options) {
  char error[1024] = {};
  DeltafinProviderSessionRequestV1 session_request = {};
  session_request.struct_size = sizeof(session_request);
  session_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  session_request.requested_device = options.device;
  session_request.device_index = options.device_index;
  session_request.max_route_positions = 1;
  DeltafinProviderSessionReportV1 session_report = {};
  session_report.struct_size = sizeof(session_report);
  require_success(deltafin_provider_session_create_v1(
                      &session_request, &session_report, error, sizeof(error)),
                  error, "create spine-binding session");

  const auto put_u16 = [](std::uint8_t* destination,
                          const std::size_t index,
                          const std::uint16_t bits) {
    std::memcpy(destination + index * sizeof(bits), &bits, sizeof(bits));
  };
  const auto put_f32 = [](std::uint8_t* destination,
                          const std::size_t index, const float value) {
    std::memcpy(destination + index * sizeof(value), &value, sizeof(value));
  };
  const auto read = [&](const DeltafinProviderSessionHandleV1 session,
                        const std::uint32_t layer_index,
                        const std::uint64_t generation,
                        const std::uint32_t slot,
                        const std::uint32_t component,
                        const std::uint64_t count,
                        const std::uint32_t expected_scalar) {
    std::vector<float> destination(static_cast<std::size_t>(count));
    DeltafinProviderSpineTensorReadF32V1 request = {};
    request.struct_size = sizeof(request);
    request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    request.session = session;
    request.generation = generation;
    request.layer_index = layer_index;
    request.slot = slot;
    request.component = component;
    request.destination = destination.data();
    request.element_capacity = count;
    DeltafinProviderSpineTensorReadReportV1 report = {};
    report.struct_size = sizeof(report);
    require_success(deltafin_provider_spine_tensor_read_f32_v1(
                        &request, &report, error, sizeof(error)),
                    error, "read grouped spine tensor");
    if (report.element_count != count ||
        report.stored_scalar_type != expected_scalar) {
      throw std::runtime_error("spine readback report has an invalid contract");
    }
    return destination;
  };

  const auto run_grouped_case = [&](const bool packed,
                                    const std::uint32_t layer_index,
                                    const std::uint64_t generation,
                                    const bool retain) {
    alignas(256) std::array<std::uint8_t, 512> quantized = {};
    alignas(256) std::array<std::uint8_t, 512> scales = {};
    alignas(256) std::array<std::uint8_t, 2048> other = {};
    std::uint8_t* quantized_data =
        packed ? other.data() : quantized.data();
    std::uint8_t* scale_data =
        packed ? other.data() + 512 : scales.data();
    std::uint8_t* bf16_data =
        packed ? other.data() + 1024 : other.data();
    std::uint8_t* f32_data =
        packed ? other.data() + 1536 : other.data() + 512;
    for (std::size_t index = 0; index < 512; ++index) {
      const int value = static_cast<int>(index % 127) - 63;
      quantized_data[index] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(value));
    }
    for (std::size_t index = 0; index < 256; ++index) {
      put_u16(scale_data, index, index < 128 ? 0x3800 : 0xc000);
      put_u16(bf16_data, index, index < 128 ? 0x3f80 : 0xc000);
    }
    for (std::size_t index = 0; index < 128; ++index) {
      put_f32(f32_data, index, index < 64 ? 3.5F : -1.25F);
    }

    std::array<DeltafinProviderSpineTensorDescriptorV1, 6> descriptors = {};
    const std::uint32_t data_buffer = packed
        ? DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1
        : DELTAFIN_PROVIDER_SPINE_BUFFER_QUANTIZED_V1;
    const std::uint32_t scale_buffer = packed
        ? DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1
        : DELTAFIN_PROVIDER_SPINE_BUFFER_SCALES_V1;
    const std::uint64_t scale_base = packed ? 512 : 0;
    const std::uint64_t bf16_base = packed ? 1024 : 0;
    const std::uint64_t f32_base = packed ? 1536 : 512;
    for (std::size_t index = 0; index < 2; ++index) {
      auto& raw_bf16 = descriptors[index];
      raw_bf16.slot = static_cast<std::uint32_t>(1 + index);
      raw_bf16.encoding = DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1;
      raw_bf16.rank = 1;
      raw_bf16.shape[0] = 128;
      raw_bf16.data_buffer = DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
      raw_bf16.data_offset = bf16_base + index * 256;
      raw_bf16.data_length = 256;

      auto& raw_f32 = descriptors[2 + index];
      raw_f32.slot = static_cast<std::uint32_t>(7 + index);
      raw_f32.encoding = DELTAFIN_PROVIDER_SPINE_RAW_F32_V1;
      raw_f32.rank = 1;
      raw_f32.shape[0] = 64;
      raw_f32.data_buffer = DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
      raw_f32.data_offset = f32_base + index * 256;
      raw_f32.data_length = 256;

      auto& q8 = descriptors[4 + index];
      q8.slot = static_cast<std::uint32_t>(13 + index);
      q8.encoding = DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1;
      q8.rank = 2;
      q8.shape[0] = 128;
      q8.shape[1] = 2;
      q8.data_buffer = data_buffer;
      q8.data_offset = index * 256;
      q8.data_length = 256;
      q8.auxiliary_buffer = scale_buffer;
      q8.auxiliary_offset = scale_base + index * 256;
      q8.auxiliary_length = 256;
    }

    DeltafinProviderBindSpineLayerRequestV1 bind = {};
    bind.struct_size = sizeof(bind);
    bind.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    bind.session = session_report.session;
    bind.layer_index = layer_index;
    bind.flags = retain ? DELTAFIN_PROVIDER_BIND_SPINE_RETAIN_V1 : 0;
    bind.generation = generation;
    bind.descriptors = descriptors.data();
    bind.descriptor_count = descriptors.size();
    bind.quantized = packed ? nullptr : quantized.data();
    bind.quantized_length = packed ? 0 : quantized.size();
    bind.scales = packed ? nullptr : scales.data();
    bind.scales_length = packed ? 0 : scales.size();
    bind.other = other.data();
    bind.other_length = packed ? other.size() : 1024;
    DeltafinProviderBindSpineLayerReportV1 bind_report = {};
    bind_report.struct_size = sizeof(bind_report);
    require_success(deltafin_provider_bind_spine_layer_v1(
                        &bind, &bind_report, error, sizeof(error)),
                    error, packed ? "bind packed grouped spine layer"
                                  : "bind loose grouped spine layer");
    if (bind_report.layer_index != layer_index ||
        bind_report.generation != generation ||
        bind_report.tensor_count != 6 ||
        bind_report.quantized_tensor_count != 2 ||
        bind_report.raw_tensor_count != 4 ||
        bind_report.quantized_bytes != 512 ||
        bind_report.scales_bytes != 512 ||
        bind_report.other_bytes != 1024 ||
        bind_report.resident_storage_bytes != 3072) {
      throw std::runtime_error(
          "grouped spine binding report has an invalid contract");
    }
    const auto stats =
        deltafin::provider_internal::spine_binding_debug_stats(
            session_report.session, layer_index, generation);
    const char* slab_upload_env =
        std::getenv("K3_MPS_SPINE_SLAB_UPLOAD");
    const bool slab_upload_enabled =
        slab_upload_env == nullptr || slab_upload_env[0] == '\0' ||
        std::strcmp(slab_upload_env, "1") == 0;
    const std::uint64_t expected_upload_runs =
        !packed && slab_upload_enabled &&
                session_report.selected_device ==
                    DELTAFIN_PROVIDER_DEVICE_MPS_V1
            ? 3
            : 4;
    if (stats.source_component_count != 8 ||
        stats.upload_run_count != expected_upload_runs ||
        stats.direct_upload_run_count != expected_upload_runs ||
        stats.gathered_upload_run_count != 0 ||
        stats.source_component_bytes != 2048 ||
        stats.logical_target_bytes != 3072 ||
        stats.resident_storage_bytes != 3072) {
      throw std::runtime_error(
          "grouped spine binding did not use its exact qualified upload count");
    }

    if (generation == 1) {
      auto rejected = bind;
      rejected.generation = 2;
      rejected.flags = UINT32_C(1) << 31;
      DeltafinProviderBindSpineLayerReportV1 rejected_report = {};
      rejected_report.struct_size = sizeof(rejected_report);
      if (deltafin_provider_bind_spine_layer_v1(
              &rejected, &rejected_report, error, sizeof(error)) == 0) {
        throw std::runtime_error(
            "provider accepted an unknown spine bind flag");
      }
      rejected.flags = DELTAFIN_PROVIDER_BIND_SPINE_RETAIN_V1;
      if (deltafin_provider_bind_spine_layer_v1(
              &rejected, &rejected_report, error, sizeof(error)) == 0) {
        throw std::runtime_error(
            "provider allocated a retained spine layer twice");
      }
    }

    // Destroy the complete caller arena before readback. Every returned value
    // must live solely in the four provider-owned backing allocations.
    quantized.fill(0);
    scales.fill(0);
    other.fill(0);
    const auto first_bf16 = read(
        session_report.session, layer_index, generation, 1,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 128,
        DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    const auto second_bf16 = read(
        session_report.session, layer_index, generation, 2,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 128,
        DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    const auto first_f32 = read(
        session_report.session, layer_index, generation, 7,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 64,
        DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    const auto second_f32 = read(
        session_report.session, layer_index, generation, 8,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 64,
        DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    const auto first_q8 = read(
        session_report.session, layer_index, generation, 13,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 256,
        DELTAFIN_PROVIDER_SPINE_SCALAR_I8_V1);
    const auto second_scale = read(
        session_report.session, layer_index, generation, 14,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_AUXILIARY_V1, 128,
        DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    if (first_bf16.front() != 1.0F || first_bf16.back() != 1.0F ||
        second_bf16.front() != -2.0F || second_bf16.back() != -2.0F ||
        first_f32.front() != 3.5F || first_f32.back() != 3.5F ||
        second_f32.front() != -1.25F || second_f32.back() != -1.25F ||
        first_q8.front() != -63.0F || first_q8.back() != -62.0F ||
        second_scale.front() != -2.0F || second_scale.back() != -2.0F) {
      throw std::runtime_error(
          "grouped provider-owned spine values changed after caller destruction");
    }
  };

  run_grouped_case(false, 0, 1, true);
  run_grouped_case(true, 1, 2, true);
  run_grouped_case(false, 2, 3, false);
  run_grouped_case(true, 3, 4, false);
  const auto retained_after_transient_churn = read(
      session_report.session, 0, 1, 1,
      DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 128,
      DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
  if (retained_after_transient_churn.front() != 1.0F ||
      retained_after_transient_churn.back() != 1.0F) {
    throw std::runtime_error(
        "retained spine layer changed after transient slot churn");
  }
  const auto store_stats =
      deltafin::provider_internal::spine_store_debug_stats(
          session_report.session);
  if (store_stats.resident_prefix_layers != 2 ||
      store_stats.resident_storage_bytes != 6144 ||
      !store_stats.transient_bound || store_stats.transient_layer != 3 ||
      store_stats.transient_generation != 4 ||
      store_stats.transient_storage_bytes != 3072 ||
      store_stats.last_generation != 4) {
    throw std::runtime_error(
        "provider spine resident-prefix accounting is invalid");
  }
  const char* slab_upload_env = std::getenv("K3_MPS_SPINE_SLAB_UPLOAD");
  const bool slab_upload_enabled =
      slab_upload_env == nullptr || slab_upload_env[0] == '\0' ||
      std::strcmp(slab_upload_env, "1") == 0;
  std::cout << "spine_binding.loose_upload_runs="
            << (slab_upload_enabled &&
                        session_report.selected_device ==
                            DELTAFIN_PROVIDER_DEVICE_MPS_V1
                    ? 3
                    : 4)
            << "/8\n"
            << "spine_binding.packed_upload_runs=4/8\n"
            << "spine_binding.resident_prefix=2 transient_slots=1\n";

  release_resource(session_report.session, 0,
                   deltafin_provider_session_destroy_v1,
                   "destroy spine-binding session");

  // CPU keeps only large rank-2 original-BF16 matrices in their exact 16-bit
  // checkpoint representation. Rank-1 norms remain fp32 for the arithmetic
  // tapes that consume them. Exercise both arms through the public ABI so a
  // future uploader cannot silently restore the former 2x matrix residency.
  DeltafinProviderSessionRequestV1 exact_request = {};
  exact_request.struct_size = sizeof(exact_request);
  exact_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  exact_request.requested_device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  exact_request.max_route_positions = 1;
  DeltafinProviderSessionReportV1 exact_session = {};
  exact_session.struct_size = sizeof(exact_session);
  require_success(deltafin_provider_session_create_v1(
                      &exact_request, &exact_session, error, sizeof(error)),
                  error, "create exact-BF16 CPU spine session");
  alignas(256) std::array<std::uint8_t, 512> exact_other = {};
  for (std::size_t index = 0; index < 8; ++index) {
    put_u16(exact_other.data(), index, index < 4 ? 0x3f80 : 0xc000);
  }
  for (std::size_t index = 0; index < 4; ++index) {
    put_u16(exact_other.data() + 256, index, index < 2 ? 0x3f00 : 0x4000);
  }
  std::array<DeltafinProviderSpineTensorDescriptorV1, 2> exact_descriptors =
      {};
  exact_descriptors[0].slot = 1;
  exact_descriptors[0].encoding = DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1;
  exact_descriptors[0].rank = 2;
  exact_descriptors[0].shape[0] = 2;
  exact_descriptors[0].shape[1] = 4;
  exact_descriptors[0].data_buffer =
      DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
  exact_descriptors[0].data_length = 16;
  exact_descriptors[1].slot = 2;
  exact_descriptors[1].encoding = DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1;
  exact_descriptors[1].rank = 1;
  exact_descriptors[1].shape[0] = 4;
  exact_descriptors[1].data_buffer =
      DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
  exact_descriptors[1].data_offset = 256;
  exact_descriptors[1].data_length = 8;
  DeltafinProviderBindSpineLayerRequestV1 exact_bind = {};
  exact_bind.struct_size = sizeof(exact_bind);
  exact_bind.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  exact_bind.session = exact_session.session;
  exact_bind.layer_index = 0;
  exact_bind.generation = 1;
  exact_bind.descriptors = exact_descriptors.data();
  exact_bind.descriptor_count = exact_descriptors.size();
  exact_bind.other = exact_other.data();
  exact_bind.other_length = exact_other.size();
  DeltafinProviderBindSpineLayerReportV1 exact_report = {};
  exact_report.struct_size = sizeof(exact_report);
  require_success(deltafin_provider_bind_spine_layer_v1(
                      &exact_bind, &exact_report, error, sizeof(error)),
                  error, "bind exact-BF16 CPU spine layer");
  const auto exact_stats =
      deltafin::provider_internal::spine_binding_debug_stats(
          exact_session.session, 0, 1);
  if (exact_report.tensor_count != 2 || exact_report.raw_tensor_count != 2 ||
      exact_report.other_bytes != 24 ||
      exact_report.resident_storage_bytes != 32 ||
      exact_stats.source_component_bytes != 24 ||
      exact_stats.logical_target_bytes != 32 ||
      exact_stats.resident_storage_bytes != 32) {
    throw std::runtime_error(
        "CPU exact-BF16 matrix/vector residency contract regressed");
  }
  exact_other.fill(0);
  const auto exact_matrix = read(
      exact_session.session, 0, 1, 1,
      DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 8,
      DELTAFIN_PROVIDER_SPINE_SCALAR_BF16_V1);
  const auto promoted_vector = read(
      exact_session.session, 0, 1, 2,
      DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 4,
      DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
  if (exact_matrix.front() != 1.0F || exact_matrix.back() != -2.0F ||
      promoted_vector.front() != 0.5F ||
      promoted_vector.back() != 2.0F) {
    throw std::runtime_error(
        "CPU exact-BF16 readback changed after caller arena destruction");
  }
  std::cout << "spine_binding.cpu_bf16_matrix_bytes=16/16"
               " vector_bytes=16/8\n";
  release_resource(exact_session.session, 0,
                   deltafin_provider_session_destroy_v1,
                   "destroy exact-BF16 CPU spine session");

  // Repeat the same public bind on the physically selected accelerator.  The
  // source is poisoned, released, and deliberately replaced before either a
  // production exact projection or diagnostic readback is requested.  Device
  // allocator churn between those boundaries makes this a lifetime test, not
  // merely a synchronous bind/read smoke test.
  if (session_report.selected_device != DELTAFIN_PROVIDER_DEVICE_CPU_V1) {
    DeltafinProviderSessionRequestV1 device_exact_request = {};
    device_exact_request.struct_size = sizeof(device_exact_request);
    device_exact_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    device_exact_request.requested_device = session_report.selected_device;
    device_exact_request.device_index = session_report.device_index;
    device_exact_request.max_route_positions = 64;
    DeltafinProviderSessionReportV1 device_exact_session = {};
    device_exact_session.struct_size = sizeof(device_exact_session);
    require_success(deltafin_provider_session_create_v1(
                        &device_exact_request, &device_exact_session, error,
                        sizeof(error)),
                    error, "create selected-device exact-BF16 session");
    const at::Device selected_device =
        deltafin::provider_internal::select_device(
            session_report.selected_device, session_report.device_index)
            .device;
    constexpr std::size_t iterations = 8;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      AlignedBytes source = allocate_aligned_bytes(256, 512);
      if (source == nullptr) {
        throw std::runtime_error(
            "allocate aligned selected-device BF16 source failed");
      }
      std::memset(source.get(), 0, 512);
      const bool alternate = (iteration & 1U) != 0;
      const std::uint16_t positive = alternate ? 0x3f00 : 0x3f80;
      const std::uint16_t negative = alternate ? 0xbf80 : 0xc000;
      for (std::size_t index = 0; index < 8; ++index) {
        put_u16(source.get(), index, index < 4 ? positive : negative);
      }
      for (std::size_t index = 0; index < 4; ++index) {
        put_u16(source.get() + 256, index,
                index < 2 ? 0x3f00 : 0x4000);
      }
      DeltafinProviderBindSpineLayerRequestV1 bind = exact_bind;
      bind.session = device_exact_session.session;
      bind.generation = static_cast<std::uint64_t>(iteration + 1);
      bind.other = source.get();
      bind.other_length = 512;
      DeltafinProviderBindSpineLayerReportV1 report = {};
      report.struct_size = sizeof(report);
      require_success(deltafin_provider_bind_spine_layer_v1(
                          &bind, &report, error, sizeof(error)),
                      error, "bind selected-device exact-BF16 spine layer");
      if (report.resident_storage_bytes != 32 ||
          report.other_bytes != 24) {
        throw std::runtime_error(
            "selected-device exact-BF16 residency contract regressed");
      }

      std::memset(source.get(), UINT8_C(0xa5), 512);
      source.reset();
      AlignedBytes reused = allocate_aligned_bytes(256, 512);
      if (reused == nullptr) {
        throw std::runtime_error(
            "allocate aligned selected-device BF16 reuse failed");
      }
      std::memset(reused.get(), UINT8_C(0x5a), 512);
      std::vector<at::Tensor> device_churn;
      device_churn.reserve(32);
      for (std::int64_t allocation = 0; allocation < 32; ++allocation) {
        device_churn.push_back(at::full(
            {128 + allocation * 17}, static_cast<double>(allocation + 1),
            at::TensorOptions().dtype(at::kFloat).device(selected_device)));
      }

      constexpr std::array<std::size_t, 4> positions_roster{1, 2, 9, 64};
      const std::size_t positions = positions_roster[iteration % 4];
      std::vector<float> input(positions * 4);
      for (std::size_t position = 0; position < positions; ++position) {
        for (std::size_t column = 0; column < 4; ++column) {
          input[position * 4 + column] =
              static_cast<float>((position + 1) * (column + 1));
        }
      }
      const std::vector<float> projected =
          deltafin::provider_internal::spine_original_bf16_debug_project(
              device_exact_session.session, 0, bind.generation, 1, input);
      const float expected_positive = alternate ? 5.0F : 10.0F;
      const float expected_negative = alternate ? -10.0F : -20.0F;
      if (projected.size() != positions * 2) {
        throw std::runtime_error(
            "selected-device exact-BF16 projection changed its T-by-rows "
            "shape");
      }
      for (std::size_t position = 0; position < positions; ++position) {
        const float multiplier = static_cast<float>(position + 1);
        if (projected[position * 2] != expected_positive * multiplier ||
            projected[position * 2 + 1] != expected_negative * multiplier) {
          throw std::runtime_error(
              "selected-device exact-BF16 projection retained poisoned "
              "caller storage");
        }
      }
      const auto matrix = read(
          device_exact_session.session, 0, bind.generation, 1,
          DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 8,
          DELTAFIN_PROVIDER_SPINE_SCALAR_BF16_V1);
      if (matrix.front() != (alternate ? 0.5F : 1.0F) ||
          matrix.back() != (alternate ? -1.0F : -2.0F)) {
        throw std::runtime_error(
            "selected-device exact-BF16 readback retained poisoned caller "
            "storage");
      }
      static_cast<void>(reused);
      static_cast<void>(device_churn);
    }
    std::cout << "spine_binding.selected_bf16_source_poison=PASS iterations="
              << iterations << " positions=1,2,9,64 device="
              << device_name(session_report.selected_device) << '\n';
    release_resource(device_exact_session.session, 0,
                     deltafin_provider_session_destroy_v1,
                     "destroy selected-device exact-BF16 session");
  }

  DeltafinProviderSessionRequestV1 cpu_mla_request = {};
  cpu_mla_request.struct_size = sizeof(cpu_mla_request);
  cpu_mla_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  cpu_mla_request.requested_device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  cpu_mla_request.flags = DELTAFIN_PROVIDER_SESSION_SYNTHETIC_MLA_V1;
  cpu_mla_request.max_route_positions = 1;
  DeltafinProviderSessionReportV1 cpu_mla_session = {};
  cpu_mla_session.struct_size = sizeof(cpu_mla_session);
  require_success(deltafin_provider_session_create_v1(
                      &cpu_mla_request, &cpu_mla_session, error,
                      sizeof(error)),
                  error, "create grouped MLA CPU session");
  alignas(256) std::array<std::uint8_t, 8192> cpu_mla_bits = {};
  const std::array<std::uint32_t, 3> cpu_mla_slots{21, 24, 27};
  const std::array<std::uint64_t, 3> cpu_mla_rows{32, 64, 32};
  std::array<DeltafinProviderSpineTensorDescriptorV1, 3>
      cpu_mla_descriptors = {};
  std::uint64_t cpu_mla_offset = 0;
  for (std::size_t index = 0; index < cpu_mla_descriptors.size(); ++index) {
    auto& descriptor = cpu_mla_descriptors[index];
    descriptor.slot = cpu_mla_slots[index];
    descriptor.encoding = DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1;
    descriptor.rank = 2;
    descriptor.shape[0] = cpu_mla_rows[index];
    descriptor.shape[1] = 32;
    descriptor.data_buffer = DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
    descriptor.data_offset = cpu_mla_offset;
    descriptor.data_length = cpu_mla_rows[index] * 32 * 2;
    for (std::uint64_t element = 0; element < descriptor.data_length / 2;
         ++element) {
      put_u16(cpu_mla_bits.data() + descriptor.data_offset,
              static_cast<std::size_t>(element),
              index == 0 ? 0x3f80 : (index == 1 ? 0x3f00 : 0xc000));
    }
    cpu_mla_offset += descriptor.data_length;
  }
  DeltafinProviderBindSpineLayerRequestV1 cpu_mla_bind = {};
  cpu_mla_bind.struct_size = sizeof(cpu_mla_bind);
  cpu_mla_bind.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  cpu_mla_bind.session = cpu_mla_session.session;
  cpu_mla_bind.layer_index = 3;
  cpu_mla_bind.generation = 1;
  cpu_mla_bind.descriptors = cpu_mla_descriptors.data();
  cpu_mla_bind.descriptor_count = cpu_mla_descriptors.size();
  cpu_mla_bind.other = cpu_mla_bits.data();
  cpu_mla_bind.other_length = cpu_mla_bits.size();
  DeltafinProviderBindSpineLayerReportV1 cpu_mla_report = {};
  cpu_mla_report.struct_size = sizeof(cpu_mla_report);
  require_success(deltafin_provider_bind_spine_layer_v1(
                      &cpu_mla_bind, &cpu_mla_report, error, sizeof(error)),
                  error, "bind grouped MLA CPU spine");
  const auto cpu_mla_stats =
      deltafin::provider_internal::spine_binding_debug_stats(
          cpu_mla_session.session, 3, 1);
  if (cpu_mla_report.resident_storage_bytes != cpu_mla_bits.size() ||
      cpu_mla_stats.source_component_count != 3 ||
      cpu_mla_stats.upload_run_count != 1 ||
      cpu_mla_stats.direct_upload_run_count != 1 ||
      cpu_mla_stats.gathered_upload_run_count != 0 ||
      cpu_mla_stats.source_component_bytes != cpu_mla_bits.size() ||
      cpu_mla_stats.logical_target_bytes != cpu_mla_bits.size() ||
      cpu_mla_stats.resident_storage_bytes != cpu_mla_bits.size() ||
      cpu_mla_stats.mla_input_bundle_count != 1) {
    throw std::runtime_error(
        "CPU original-BF16 MLA bundle is not one zero-copy upload run");
  }
  std::cout << "spine_binding.cpu_mla_bundle_upload_runs=1/3"
               " resident_bytes=8192 logical_bytes=8192\n";
  release_resource(cpu_mla_session.session, 0,
                   deltafin_provider_session_destroy_v1,
                   "destroy grouped MLA CPU session");

  // The qualified MLA MPS path gathers three physically discontiguous
  // same-input projections directly into two final resident allocations: one
  // int8 weight matrix and one converted fp32 scale vector. This replaces six
  // per-component uploads plus the old duplicate bind-time concatenation.
  if (session_report.selected_device == DELTAFIN_PROVIDER_DEVICE_MPS_V1) {
    DeltafinProviderSessionRequestV1 mla_request = {};
    mla_request.struct_size = sizeof(mla_request);
    mla_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    mla_request.requested_device = DELTAFIN_PROVIDER_DEVICE_MPS_V1;
    mla_request.flags = DELTAFIN_PROVIDER_SESSION_SYNTHETIC_MLA_V1;
    mla_request.max_route_positions = 1;
    DeltafinProviderSessionReportV1 mla_session = {};
    mla_session.struct_size = sizeof(mla_session);
    require_success(deltafin_provider_session_create_v1(
                        &mla_request, &mla_session, error, sizeof(error)),
                    error, "create grouped MLA MPS session");

    alignas(256) std::array<std::uint8_t, 4864> packed = {};
    const std::array<std::uint32_t, 3> slots{
        21, 24, 27};
    const std::array<std::uint64_t, 3> rows{32, 64, 32};
    const std::array<std::uint64_t, 3> data_offsets{0, 1280, 3584};
    const std::array<std::uint64_t, 3> scale_offsets{1024, 3328, 4608};
    std::array<DeltafinProviderSpineTensorDescriptorV1, 3> descriptors = {};
    for (std::size_t index = 0; index < descriptors.size(); ++index) {
      auto& descriptor = descriptors[index];
      descriptor.slot = slots[index];
      descriptor.encoding = DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1;
      descriptor.rank = 2;
      descriptor.shape[0] = rows[index];
      descriptor.shape[1] = 32;
      descriptor.data_buffer = DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
      descriptor.data_offset = data_offsets[index];
      descriptor.data_length = rows[index] * 32;
      descriptor.auxiliary_buffer = DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
      descriptor.auxiliary_offset = scale_offsets[index];
      descriptor.auxiliary_length = rows[index] * 2;
      for (std::uint64_t element = 0; element < descriptor.data_length;
           ++element) {
        packed[descriptor.data_offset + element] =
            static_cast<std::uint8_t>(static_cast<std::int8_t>(
                static_cast<int>(index) * 11 - 7));
      }
      for (std::uint64_t row = 0; row < rows[index]; ++row) {
        put_u16(packed.data() + descriptor.auxiliary_offset,
                static_cast<std::size_t>(row),
                index == 0 ? 0x3800 : (index == 1 ? 0x3c00 : 0xc000));
      }
    }

    DeltafinProviderBindSpineLayerRequestV1 bind = {};
    bind.struct_size = sizeof(bind);
    bind.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    bind.session = mla_session.session;
    bind.layer_index = 3;
    bind.generation = 1;
    bind.descriptors = descriptors.data();
    bind.descriptor_count = descriptors.size();
    bind.other = packed.data();
    bind.other_length = packed.size();
    DeltafinProviderBindSpineLayerReportV1 report = {};
    report.struct_size = sizeof(report);
    require_success(deltafin_provider_bind_spine_layer_v1(
                        &bind, &report, error, sizeof(error)),
                    error, "bind grouped MLA MPS spine");
    const auto stats =
        deltafin::provider_internal::spine_binding_debug_stats(
            mla_session.session, 3, 1);
    if (stats.source_component_count != 6 || stats.upload_run_count != 2 ||
        stats.direct_upload_run_count != 0 ||
        stats.gathered_upload_run_count != 2 ||
        stats.source_component_bytes != 4352 ||
        stats.logical_target_bytes != 4608 ||
        stats.resident_storage_bytes != 4608) {
      throw std::runtime_error(
          "MLA MPS spine did not reduce six uploads to two nonduplicated bundle allocations");
    }
    std::cout << "spine_binding.mla_bundle_upload_runs=2/6"
                 " resident_bytes=4608 logical_bytes=4608\n";
    packed.fill(0);
    const auto query = read(
        mla_session.session, 3, 1, 21,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 1024,
        DELTAFIN_PROVIDER_SPINE_SCALAR_I8_V1);
    const auto key_value_scale = read(
        mla_session.session, 3, 1, 24,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_AUXILIARY_V1, 64,
        DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    const auto gate = read(
        mla_session.session, 3, 1, 27,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 1024,
        DELTAFIN_PROVIDER_SPINE_SCALAR_I8_V1);
    if (query.front() != -7.0F || query.back() != -7.0F ||
        key_value_scale.front() != 1.0F ||
        key_value_scale.back() != 1.0F || gate.front() != 15.0F ||
        gate.back() != 15.0F) {
      throw std::runtime_error(
          "MLA gathered bundle changed after caller arena destruction: q=" +
          std::to_string(query.front()) + "," +
          std::to_string(query.back()) + " kv_scale=" +
          std::to_string(key_value_scale.front()) + "," +
          std::to_string(key_value_scale.back()) + " gate=" +
          std::to_string(gate.front()) + "," +
          std::to_string(gate.back()));
    }
    release_resource(mla_session.session, 0,
                     deltafin_provider_session_destroy_v1,
                     "destroy grouped MLA MPS session");

    // Exercise the loose three-slab MPS path with the three bundled
    // projections separated by two ordinary projections and with real
    // 256-byte scale-alignment holes. The provider must compact/reorder on
    // device, retain no padding, and still publish the established zero-copy
    // MLA bundle. Setting K3_MPS_SPINE_SLAB_UPLOAD=0 runs the exact grouped
    // fallback in the same test process.
    DeltafinProviderSessionReportV1 loose_mla_session = {};
    loose_mla_session.struct_size = sizeof(loose_mla_session);
    require_success(deltafin_provider_session_create_v1(
                        &mla_request, &loose_mla_session, error,
                        sizeof(error)),
                    error, "create loose-slab MLA MPS session");
    alignas(256) std::array<std::uint8_t, 4608> loose_quantized = {};
    alignas(256) std::array<std::uint8_t, 1088> loose_scales = {};
    alignas(256) std::array<std::uint8_t, 256> loose_other = {};
    const std::array<std::uint32_t, 5> loose_slots{21, 23, 24, 26, 27};
    const std::array<std::uint64_t, 5> loose_rows{32, 8, 64, 8, 32};
    const std::array<std::uint64_t, 5> loose_data_offsets{
        0, 1024, 1280, 3328, 3584};
    const std::array<std::uint64_t, 5> loose_scale_offsets{
        0, 256, 512, 768, 1024};
    std::array<DeltafinProviderSpineTensorDescriptorV1, 6>
        loose_descriptors = {};
    for (std::size_t index = 0; index < loose_slots.size(); ++index) {
      auto& descriptor = loose_descriptors[index];
      descriptor.slot = loose_slots[index];
      descriptor.encoding = DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1;
      descriptor.rank = 2;
      descriptor.shape[0] = loose_rows[index];
      descriptor.shape[1] = 32;
      descriptor.data_buffer = DELTAFIN_PROVIDER_SPINE_BUFFER_QUANTIZED_V1;
      descriptor.data_offset = loose_data_offsets[index];
      descriptor.data_length = loose_rows[index] * 32;
      descriptor.auxiliary_buffer =
          DELTAFIN_PROVIDER_SPINE_BUFFER_SCALES_V1;
      descriptor.auxiliary_offset = loose_scale_offsets[index];
      descriptor.auxiliary_length = loose_rows[index] * 2;
      const std::int8_t q = static_cast<std::int8_t>(
          static_cast<int>(index) * 9 - 13);
      std::memset(loose_quantized.data() + descriptor.data_offset,
                  static_cast<unsigned char>(q), descriptor.data_length);
      for (std::uint64_t row = 0; row < loose_rows[index]; ++row) {
        put_u16(loose_scales.data() + descriptor.auxiliary_offset,
                static_cast<std::size_t>(row),
                index == 0 ? 0x3800
                           : (index == 2 ? 0x3c00
                                         : (index == 4 ? 0xc000 : 0x4000)));
      }
    }
    auto& norm = loose_descriptors.back();
    norm.slot = 22;
    norm.encoding = DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1;
    norm.rank = 1;
    norm.shape[0] = 32;
    norm.data_buffer = DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
    norm.data_length = 64;
    for (std::size_t element = 0; element < 32; ++element) {
      put_u16(loose_other.data(), element, 0x3f00);
    }

    DeltafinProviderBindSpineLayerRequestV1 loose_bind = {};
    loose_bind.struct_size = sizeof(loose_bind);
    loose_bind.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    loose_bind.session = loose_mla_session.session;
    loose_bind.layer_index = 3;
    loose_bind.generation = 1;
    loose_bind.descriptors = loose_descriptors.data();
    loose_bind.descriptor_count = loose_descriptors.size();
    loose_bind.quantized = loose_quantized.data();
    loose_bind.quantized_length = loose_quantized.size();
    loose_bind.scales = loose_scales.data();
    loose_bind.scales_length = loose_scales.size();
    loose_bind.other = loose_other.data();
    loose_bind.other_length = loose_other.size();
    DeltafinProviderBindSpineLayerReportV1 loose_report = {};
    loose_report.struct_size = sizeof(loose_report);
    require_success(deltafin_provider_bind_spine_layer_v1(
                        &loose_bind, &loose_report, error, sizeof(error)),
                    error, "bind loose-slab MLA MPS spine");
    const auto loose_stats =
        deltafin::provider_internal::spine_binding_debug_stats(
            loose_mla_session.session, 3, 1);
    const char* slab_upload_env =
        std::getenv("K3_MPS_SPINE_SLAB_UPLOAD");
    const bool slab_upload_enabled =
        slab_upload_env == nullptr || slab_upload_env[0] == '\0' ||
        std::strcmp(slab_upload_env, "1") == 0;
    const std::uint64_t loose_expected_runs =
        slab_upload_enabled ? 3 : 7;
    const std::uint64_t loose_expected_direct =
        slab_upload_enabled ? 3 : 5;
    const std::uint64_t loose_expected_gathered =
        slab_upload_enabled ? 0 : 2;
    if (loose_report.resident_storage_bytes != 5312 ||
        loose_stats.source_component_count != 11 ||
        loose_stats.upload_run_count != loose_expected_runs ||
        loose_stats.direct_upload_run_count != loose_expected_direct ||
        loose_stats.gathered_upload_run_count != loose_expected_gathered ||
        loose_stats.source_component_bytes != 4960 ||
        loose_stats.logical_target_bytes != 5312 ||
        loose_stats.resident_storage_bytes != 5312 ||
        loose_stats.mla_input_bundle_count != 1) {
      throw std::runtime_error(
          "loose MLA MPS slab upload changed its exact bundle contract");
    }
    loose_quantized.fill(0);
    loose_scales.fill(0);
    loose_other.fill(0);
    const auto loose_query = read(
        loose_mla_session.session, 3, 1, 21,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 1024,
        DELTAFIN_PROVIDER_SPINE_SCALAR_I8_V1);
    const auto loose_key_value_scale = read(
        loose_mla_session.session, 3, 1, 24,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_AUXILIARY_V1, 64,
        DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    const auto loose_gate = read(
        loose_mla_session.session, 3, 1, 27,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 1024,
        DELTAFIN_PROVIDER_SPINE_SCALAR_I8_V1);
    const auto loose_norm = read(
        loose_mla_session.session, 3, 1, 22,
        DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1, 32,
        DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    if (loose_query.front() != -13.0F ||
        loose_query.back() != -13.0F ||
        loose_key_value_scale.front() != 1.0F ||
        loose_key_value_scale.back() != 1.0F ||
        loose_gate.front() != 23.0F || loose_gate.back() != 23.0F ||
        loose_norm.front() != 0.5F || loose_norm.back() != 0.5F) {
      throw std::runtime_error(
          "loose MLA MPS slab upload retained poisoned caller storage");
    }
    std::cout << "spine_binding.mla_loose_slab_upload_runs="
              << loose_expected_runs << "/11 resident_bytes=5312"
                 " logical_bytes=5312\n";
    release_resource(loose_mla_session.session, 0,
                     deltafin_provider_session_destroy_v1,
                     "destroy loose-slab MLA MPS session");
  }
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    char error[1024] = {};
    DeltafinProviderInventoryV1 inventory = {};
    inventory.struct_size = sizeof(inventory);
    if (deltafin_provider_inventory_v1(&inventory, error, sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }

    DeltafinProviderCanaryRequestV1 request = {};
    request.struct_size = sizeof(request);
    request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    request.requested_device = options.device;
    request.device_index = options.device_index;
    request.flags = options.require_packed_int8
        ? DELTAFIN_PROVIDER_REQUIRE_PACKED_INT8_V1
        : 0;
    request.packed_rows = options.packed_rows;
    request.packed_columns = options.packed_columns;
    DeltafinProviderCanaryReportV1 report = {};
    report.struct_size = sizeof(report);
    if (deltafin_provider_canary_v1(
            &request, &report, error, sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }

    std::cout << "deltafin.native_provider_gate=1\n"
              << "provider_abi=" << deltafin_provider_abi_version() << '\n'
              << "libtorch.version=" << inventory.libtorch_version << '\n'
              << "capability.mps="
              << (inventory.mps_available ? "available" : "unavailable") << '\n'
              << "capability.cuda_devices=" << inventory.cuda_device_count << '\n'
              << "selected_device=" << device_name(report.selected_device);
    if (report.selected_device == DELTAFIN_PROVIDER_DEVICE_CUDA_V1) {
      std::cout << ':' << report.device_index;
    }
    std::cout << '\n';
    print_check("rms_fp32", DELTAFIN_PROVIDER_CHECK_RMS_FP32_V1, report);
    print_check("matmul_fp32", DELTAFIN_PROVIDER_CHECK_MATMUL_FP32_V1, report);
    print_check("softmax_fp32", DELTAFIN_PROVIDER_CHECK_SOFTMAX_FP32_V1, report);
    print_check("packed_int8_fp32",
                DELTAFIN_PROVIDER_CHECK_PACKED_INT8_FP32_V1, report);
    std::cout << "packed_shape=" << report.packed_rows << 'x'
              << report.packed_columns << '\n'
              << "detail=\"" << report.detail << "\"\n"
              << "result=" << (report.required_passed ? "PASS" : "FAIL") << '\n';
    run_target_pilot_admission();
    std::cout << "check.target_pilot_admission=PASS\n";
    run_mla_expanded_only_admission();
    std::cout << "check.mla_expanded_only_abi=PASS\n";
    if (options.split_boundary) {
      run_split_boundary(options);
      std::cout << "check.split_boundary=PASS\n";
    }
    if (options.spine_binding) {
      run_spine_binding(options);
      std::cout << "check.spine_binding=PASS "
                   "(synthetic ownership/storage canary; not full-model parity)\n";
    }
    if (options.kda_tape) {
      std::string detail;
      const auto selected = deltafin::provider_internal::select_device(
          options.device, options.device_index);
      if (!deltafin::provider_internal::kda_small_parity_canary(
              selected.device, detail)) {
        throw std::runtime_error("KDA tape parity failed: " + detail);
      }
      std::cout << "check.kda_tape=PASS (small deterministic parity; "
                << detail << ")\n";
    }
    return report.required_passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "result=FAIL\nerror=\"" << error.what() << "\"\n";
    return 2;
  }
}
