#include "provider_abi.h"
#include "provider_bf16_device.h"
#include "provider_cuda_moe.h"
#include "provider_device.h"
#include "provider_dspark_model.h"
#include "provider_kda.h"
#include "provider_mla.h"
#include "provider_precision.h"
#include "provider_qwen.h"
#include "provider_bf16_cpu.h"
#include "provider_spine_debug.h"
#if defined(__APPLE__)
#include "provider_spine_int8_metal.h"
#endif
#include "provider_target.h"
#include "provider_target_sequence.h"
#include "provider_target_tape.h"
#include "../../tools/metal_moe_abi.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/ATen.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/add.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/div.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/zeros.h>
#include <c10/core/InferenceMode.h>
#if defined(__APPLE__)
#include <ATen/detail/MPSHooksInterface.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

static_assert(sizeof(DeltafinProviderSessionRequestV1) == 80,
              "session request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderSessionReportV1) == 80,
              "session report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMemoryRequestV1) == 64,
              "provider memory request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMemoryReportV1) == 104,
              "provider memory report ABI v1 layout changed");
static_assert(offsetof(DeltafinProviderMemoryReportV1, active_bytes) == 32,
              "provider memory report byte fields moved");
static_assert(sizeof(DeltafinProviderTargetPilotEnableReportV1) == 64,
              "target PILOT enable report ABI v1 layout changed");
static_assert(
    offsetof(DeltafinProviderTargetPilotEnableReportV1, reserve_bytes) == 24,
    "target PILOT enable reserve offset changed");
static_assert(
    offsetof(DeltafinProviderTargetPilotEnableReportV1, reserved) == 32,
    "target PILOT enable reserved offset changed");
static_assert(sizeof(DeltafinProviderTensorUploadF32V1) == 80,
              "tensor upload ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTensorUploadBf16V1) == 80,
              "BF16 tensor upload ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTensorReportV1) == 64,
              "tensor report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTensorReadF32V1) == 72,
              "tensor read ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderCacheCreateF32V1) == 80,
              "cache create ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderCacheReportV1) == 64,
              "cache report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderCacheReadF32V1) == 72,
              "cache read ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderResourceRequestV1) == 64,
              "resource request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderPrepareLayerRequestV1) == 80,
              "prepare request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderRouteMailboxV1) == 6224,
              "route mailbox ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderFinishLayerRequestV1) == 80,
              "finish request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderFinishLayerReportV1) == 80,
              "finish report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderSpineTensorDescriptorV1) == 152,
              "spine tensor descriptor ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderBindSpineLayerRequestV1) == 128,
              "bind-spine request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderBindSpineLayerReportV1) == 96,
              "bind-spine report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderBindSpineLayerRequestV2) == 160,
              "bind-spine request ABI v2 layout changed");
static_assert(sizeof(DeltafinProviderBindSpineLayerReportV2) == 112,
              "bind-spine report ABI v2 layout changed");
static_assert(sizeof(DeltafinProviderSpineSourceUseRequestV2) == 64,
              "spine source-use request ABI v2 layout changed");
static_assert(sizeof(DeltafinProviderSpineSourceUseReportV2) == 64,
              "spine source-use report ABI v2 layout changed");
static_assert(sizeof(DeltafinProviderSpineTensorReadF32V1) == 80,
              "spine tensor read ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderSpineTensorReadReportV1) == 96,
              "spine tensor read report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderKdaCacheCreateV1) == 64,
              "KDA cache-create ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderKdaCacheReportV1) == 64,
              "KDA cache report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderKdaDecodeRequestV1) == 80,
              "KDA decode request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderKdaDecodeReportV1) == 80,
              "KDA decode report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderKdaCommitReportV1) == 64,
              "KDA commit report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMlaCacheCreateV1) == 64,
              "MLA cache-create ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMlaCacheReportV1) == 64,
              "MLA cache report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMlaDecodeRequestV1) == 80,
              "MLA decode request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMlaDecodeReportV1) == 96,
              "MLA decode report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMlaCommitReportV1) == 64,
              "MLA commit report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderBindTargetGlobalsRequestV1) == 128,
              "target-global bind request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderBindTargetGlobalsReportV1) == 96,
              "target-global bind report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetBeginRequestV1) == 64,
              "target begin request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetBeginBf16RequestV1) == 64,
              "target BF16 begin request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetBeginReportV1) == 64,
              "target begin report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetPrepareRequestV1) == 64,
              "target prepare request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetPrepareReportV1) == 160,
              "target prepare report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMetalExpertLayoutsRequestV1) == 64,
              "Metal expert-layout request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderMetalExpertLayoutsReportV1) == 64,
              "Metal expert-layout report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetFinishExpertsRequestV1) == 160,
              "target expert-finish request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetFinishExpertsReportV1) == 64,
              "target expert-finish report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetGreedyReportV1) == 64,
              "target greedy report ABI v1 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequenceBeginBf16RequestV1) == 80,
    "target-sequence BF16 begin request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetSequenceBeginReportV1) == 64,
              "target-sequence begin report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetSequencePrepareRequestV1) == 64,
              "target-sequence prepare request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetSequencePrepareReportV1) == 6224,
              "target-sequence prepare report ABI v1 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequencePrefetchHintReportV1) == 128,
    "target-sequence prefetch-hint report ABI v1 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequenceFinishExpertsRequestV1) == 256,
    "target-sequence expert request ABI v1 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequenceFinishExpertSpansRequestV1) == 760,
    "target-sequence scattered expert request ABI v1 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequenceFinishExpertsRequestV2) == 200,
    "target-sequence expert request ABI v2 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequencePlanExpertsRequestV1) == 240,
    "target-sequence CUDA expert-plan request ABI v1 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequencePlanExpertsReportV1) == 208,
    "target-sequence CUDA expert-plan report ABI v1 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequenceFinishPlannedExpertsRequestV1) ==
        240,
    "target-sequence CUDA planned-finish request ABI v1 layout changed");
static_assert(
    sizeof(DeltafinProviderTargetSequenceFinishExpertsReportV1) == 64,
    "target-sequence expert report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetSequenceTailReportV1) == 320,
              "target-sequence tail report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetSequenceCommitRequestV1) == 64,
              "target-sequence commit request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetSequenceCommitReportV1) == 64,
              "target-sequence commit report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetStateReportV1) == 56,
              "target-state report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetStateBranchRequestV1) == 64,
              "target-state branch request ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderTargetSequenceStatsReportV1) == 160,
              "target-sequence stats report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkTensorV1) == 64,
              "DSpark tensor ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkCreateV1) == 88,
              "DSpark create ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkReportV1) == 64,
              "DSpark report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkAppendV1) == 88,
              "DSpark append ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkAppendTensorV1) == 80,
              "DSpark tensor-append ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkSnapshotReportV1) == 64,
              "DSpark snapshot ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkRestoreV1) == 64,
              "DSpark restore ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkProposeV1) == 80,
              "DSpark propose ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderDSparkProposalReportV1) == 120,
              "DSpark proposal report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderQwenTensorV1) == 64,
              "Qwen tensor ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderQwenCreateV1) == 80,
              "Qwen create ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderQwenReportV1) == 64,
              "Qwen report ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderQwenGenerateV1) == 80,
              "Qwen generate ABI v1 layout changed");
static_assert(sizeof(DeltafinProviderQwenGenerationReportV1) == 208,
              "Qwen generation report ABI v1 layout changed");
static_assert(
    alignof(DeltafinProviderTargetSequencePrepareReportV1) == alignof(uint64_t),
    "target-sequence prepare report ABI alignment changed");
static_assert(
    offsetof(DeltafinProviderTargetSequencePrepareReportV1,
             ordered_experts) == 48,
    "target-sequence expert mailbox offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequencePrepareReportV1,
             ordered_weight_bits) == 2096,
    "target-sequence weight mailbox offset changed");
static_assert(
    alignof(DeltafinProviderTargetSequencePrefetchHintReportV1) ==
        alignof(uint64_t),
    "target-sequence prefetch-hint report ABI alignment changed");
static_assert(
    offsetof(DeltafinProviderTargetSequencePrefetchHintReportV1,
             expert_ids) == 32,
    "target-sequence prefetch-hint expert offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequencePrefetchHintReportV1,
             reserved) == 96,
    "target-sequence prefetch-hint reserved offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequenceFinishExpertsRequestV1,
             expert_ids) == 64,
    "target-sequence canonical expert ID offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequenceFinishExpertsRequestV1,
             expert_major_bytes) == 192,
    "target-sequence borrowed expert pointer offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequenceFinishExpertSpansRequestV1,
             expert_span_pointers) == 192,
    "target-sequence scattered expert pointer offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequenceFinishExpertsRequestV2,
             expert_ids) == 64,
    "target-sequence v2 canonical expert ID pointer offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequenceFinishExpertsRequestV2,
             expert_span_pointers) == 96,
    "target-sequence v2 scattered expert pointer offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequenceFinishExpertsRequestV2,
             reserved) == 136,
    "target-sequence v2 reserved offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequencePlanExpertsRequestV1,
             expert_ids) == 60,
    "target-sequence CUDA expert-plan ID offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequencePlanExpertsReportV1,
             missing_experts) == 56,
    "target-sequence CUDA expert-plan miss offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequenceFinishPlannedExpertsRequestV1,
             missing_experts) == 60,
    "target-sequence CUDA planned-finish miss offset changed");
static_assert(
    offsetof(DeltafinProviderTargetSequenceFinishExpertsReportV1,
             layer_index) == 24,
    "target-sequence expert report layer offset changed");
static_assert(std::endian::native == std::endian::little,
              "Deltafin spine storage is little-endian");

constexpr std::uint32_t kSyntheticFlag =
    DELTAFIN_PROVIDER_SESSION_SYNTHETIC_SPLIT_V1;
constexpr std::uint32_t kSyntheticKdaFlag =
    DELTAFIN_PROVIDER_SESSION_SYNTHETIC_KDA_V1;
constexpr std::uint32_t kSyntheticMlaFlag =
    DELTAFIN_PROVIDER_SESSION_SYNTHETIC_MLA_V1;
constexpr std::uint32_t kKnownSessionFlags =
    kSyntheticFlag | kSyntheticKdaFlag | kSyntheticMlaFlag;
constexpr std::uint32_t kRetainSpineFlag =
    DELTAFIN_PROVIDER_BIND_SPINE_RETAIN_V1;
constexpr std::uint32_t kKnownBindSpineFlags = kRetainSpineFlag;
constexpr std::uint32_t kAllowBorrowSpineFlag =
    DELTAFIN_PROVIDER_BIND_SPINE_ALLOW_BORROW_V2;
constexpr std::uint32_t kKnownBindSpineV2Flags =
    kRetainSpineFlag | kAllowBorrowSpineFlag;
constexpr std::uint32_t kK3Layers = 93;
constexpr std::uint32_t kK3Experts = 896;
constexpr std::uint64_t kTargetPilotBytesPerLayer =
    UINT64_C(896) * 7168 + UINT64_C(896) * sizeof(float) +
    UINT64_C(7168) * sizeof(float) + UINT64_C(896) * sizeof(float);
constexpr std::uint64_t kTargetPilotReserveBytes =
    UINT64_C(92) * kTargetPilotBytesPerLayer;
static_assert(DELTAFIN_PROVIDER_TARGET_PILOT_LAYER_CAPACITY_V1 == 92);
static_assert(kTargetPilotReserveBytes ==
              DELTAFIN_PROVIDER_TARGET_PILOT_RESERVE_BYTES_V1);
constexpr std::uint64_t kMaximumTensorElements = UINT64_C(1) << 30;
constexpr std::size_t kMaximumLiveResources = 8192;
constexpr std::uint64_t kMaximumSpineTensorElements = UINT64_C(1) << 34;
constexpr std::uint64_t kMaximumSpineBufferBytes = UINT64_C(16) << 30;
constexpr std::uint64_t kSpineComponentAlignment = 256;
constexpr std::uint64_t kMaximumSpineDescriptors = 64;
constexpr std::uint32_t kFirstLayerWeightSlot = 1;
constexpr std::uint32_t kLastLayerWeightSlot = 39;
constexpr std::uint32_t kLastGlobalWeightSlot = 44;
constexpr std::uint32_t kInputNormSlot = 1;
constexpr std::uint32_t kPostAttentionNormSlot = 2;
constexpr std::uint32_t kAttentionResidualNormSlot = 3;
constexpr std::uint32_t kAttentionResidualProjectionSlot = 4;
constexpr std::uint32_t kMlpResidualNormSlot = 5;
constexpr std::uint32_t kMlpResidualProjectionSlot = 6;
constexpr std::uint32_t kKdaALogSlot = 7;
constexpr std::uint32_t kKdaDtBiasSlot = 8;
constexpr std::uint32_t kKdaQueryConvolutionSlot = 9;
constexpr std::uint32_t kKdaKeyConvolutionSlot = 10;
constexpr std::uint32_t kKdaValueConvolutionSlot = 11;
constexpr std::uint32_t kKdaOutputNormSlot = 12;
constexpr std::uint32_t kKdaQueryProjectionSlot = 13;
constexpr std::uint32_t kKdaKeyProjectionSlot = 14;
constexpr std::uint32_t kKdaValueProjectionSlot = 15;
constexpr std::uint32_t kKdaGateProjectionSlot = 16;
constexpr std::uint32_t kKdaFeatureAProjectionSlot = 17;
constexpr std::uint32_t kKdaFeatureBProjectionSlot = 18;
constexpr std::uint32_t kKdaBetaProjectionSlot = 19;
constexpr std::uint32_t kKdaOutputProjectionSlot = 20;
constexpr std::uint32_t kMlaQueryAProjectionSlot = 21;
constexpr std::uint32_t kMlaQueryANormSlot = 22;
constexpr std::uint32_t kMlaQueryBProjectionSlot = 23;
constexpr std::uint32_t kMlaKeyValueAProjectionSlot = 24;
constexpr std::uint32_t kMlaKeyValueANormSlot = 25;
constexpr std::uint32_t kMlaKeyValueBProjectionSlot = 26;
constexpr std::uint32_t kMlaOutputGateProjectionSlot = 27;
constexpr std::uint32_t kMlaOutputProjectionSlot = 28;
constexpr std::uint32_t kDenseGateProjectionSlot = 29;
constexpr std::uint32_t kDenseUpProjectionSlot = 30;
constexpr std::uint32_t kDenseDownProjectionSlot = 31;
constexpr std::uint32_t kMoeGateWeightSlot = 32;
constexpr std::uint32_t kMoeGateCorrectionBiasSlot = 33;
constexpr std::uint32_t kMoeRoutedDownProjectionSlot = 34;
constexpr std::uint32_t kMoeRoutedNormSlot = 35;
constexpr std::uint32_t kMoeRoutedUpProjectionSlot = 36;
constexpr std::uint32_t kMoeSharedGateProjectionSlot = 37;
constexpr std::uint32_t kMoeSharedUpProjectionSlot = 38;
constexpr std::uint32_t kMoeSharedDownProjectionSlot = 39;
constexpr std::uint32_t kFinalNormSlot = 41;
constexpr std::uint32_t kOutputResidualNormSlot = 42;
constexpr std::uint32_t kOutputResidualProjectionSlot = 43;
constexpr std::uint32_t kLanguageModelHeadSlot = 44;
constexpr std::uint32_t kTargetGlobalTailGroup =
    DELTAFIN_PROVIDER_TARGET_GLOBAL_TAIL_V1;
constexpr std::uint32_t kTargetGlobalHeadGroup =
    DELTAFIN_PROVIDER_TARGET_GLOBAL_HEAD_V1;

void copy_text(char* destination, const std::size_t capacity,
               const std::string& value) noexcept {
  if (destination == nullptr || capacity == 0) {
    return;
  }
  const std::size_t copied = std::min(capacity - 1, value.size());
  std::memcpy(destination, value.data(), copied);
  destination[copied] = '\0';
}

template <typename Function>
std::int32_t ffi_guard(char* error, const std::size_t error_capacity,
                       Function&& function) noexcept {
  try {
    function();
    copy_text(error, error_capacity, "");
    return 0;
  } catch (const std::exception& exception) {
    copy_text(error, error_capacity, exception.what());
    return 1;
  } catch (...) {
    copy_text(error, error_capacity, "unknown C++ provider runtime failure");
    return 1;
  }
}

template <std::size_t Size>
bool all_zero(const std::uint64_t (&values)[Size]) {
  return std::all_of(std::begin(values), std::end(values),
                     [](const std::uint64_t value) { return value == 0; });
}

void require_header(const std::uint32_t size, const std::size_t expected_size,
                    const std::uint32_t version, const char* name) {
  if (size != expected_size || version != DELTAFIN_PROVIDER_ABI_VERSION) {
    throw std::invalid_argument(std::string(name) +
                                " does not match provider ABI v1");
  }
}

std::int64_t checked_dimension(const std::uint64_t value, const char* name) {
  if (value == 0 || value > static_cast<std::uint64_t>(
                                std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(std::string(name) +
                                " must be a positive int64 dimension");
  }
  return static_cast<std::int64_t>(value);
}

std::uint64_t checked_elements(const std::uint64_t rows,
                               const std::uint64_t columns) {
  if (rows == 0 || columns == 0 || rows > kMaximumTensorElements / columns) {
    throw std::invalid_argument(
        "provider tensor shape is empty, overflows, or exceeds its safety limit");
  }
  return rows * columns;
}

struct CacheSlot {
  at::Tensor tensor;
  std::uint64_t version = 0;
};

struct TicketSlot {
  at::Tensor prepared;
  at::Tensor next_cache;
  DeltafinProviderCacheHandleV1 cache = 0;
  std::uint64_t expected_cache_version = 0;
  std::uint32_t layer_index = 0;
};

struct SpineTensorSlot {
  std::uint32_t encoding = 0;
  std::vector<std::int64_t> shape;
  at::Tensor data;
  at::Tensor auxiliary;
  deltafin::provider_internal::OriginalBf16Matrix original_bf16;
};

struct SpineSourceUseSlot {
  std::uint64_t generation = 0;
  std::uint32_t layer_index = 0;
  std::uint32_t state = DELTAFIN_PROVIDER_SPINE_SOURCE_OPEN_V2;
};

struct SpineLayerSlot {
  std::uint32_t layer_index = 0;
  std::uint64_t generation = 0;
  std::uint32_t quantized_tensor_count = 0;
  std::uint32_t raw_tensor_count = 0;
  std::uint64_t quantized_bytes = 0;
  std::uint64_t scales_bytes = 0;
  std::uint64_t other_bytes = 0;
  std::uint32_t tensor_count = 0;
  std::uint32_t borrowed_tensor_count = 0;
  std::uint64_t borrowed_source_bytes = 0;
  deltafin::provider_internal::SpineBindingDebugStats binding_stats;
  // Slots 40..44 stay empty under the current layer-only bind ABI, but the
  // fixed layout already reserves the audited global embedding/tail/head
  // roster so that its next bind path needs no Session representation change.
  std::array<std::optional<SpineTensorSlot>, kLastGlobalWeightSlot + 1>
      tensors;
  std::unique_ptr<deltafin::provider_internal::MlaInputBundle>
      mla_input_bundle;
  // A complete production layer precomputes these two exact score vectors at
  // bind time. Partial diagnostic binds deliberately leave this empty.
  std::unique_ptr<deltafin::provider_internal::TargetResidualWeights>
      target_residual;
};

/*
 * One provider-owned FP32 template for the current streamed layer. Compact
 * q8/BF16 layer storage remains authoritative and may be retained; execution
 * views borrow only this Tensor storage. The abstraction is deliberately
 * source-encoding-neutral so exact original-BF16 can target the same arena in
 * a later qualified path without allocating a second dense template.
 */
struct SpineFp32ExecutionArena {
  at::Tensor storage;
  std::uint64_t owner = 0;
  std::uint64_t spine_generation = 0;
  std::uint32_t layer_index = 0;
  bool occupied = false;
};

struct SpineFp32ExecutionView {
  std::array<std::optional<at::Tensor>, kLastGlobalWeightSlot + 1> tensors;
  std::uint64_t owner = 0;
  std::uint64_t spine_generation = 0;
  std::uint32_t layer_index = 0;
  std::uint64_t required_elements = 0;
};

enum class SpineFp32ExecutionSource {
  None,
  RowInt8,
  OriginalBf16,
};

struct TargetSessionCacheStore {
  std::array<std::unique_ptr<deltafin::provider_internal::TargetKdaCache>,
             kK3Layers>
      kda;
  std::array<std::unique_ptr<deltafin::provider_internal::MlaCache>, kK3Layers>
      mla;
  std::array<deltafin::provider_internal::TargetLayerCacheBinding, kK3Layers>
      bindings;
};

struct TargetStateBranchSlot {
  DeltafinProviderTargetStateBranchHandleV1 handle = 0;
  std::unique_ptr<TargetSessionCacheStore> parent;
  std::uint64_t parent_positions = 0;
  std::uint64_t parent_generation = 0;
};

struct KdaCacheSlot {
  deltafin::provider_internal::KdaState state;
  std::uint64_t version = 0;
  std::uint32_t layer_index = 0;
};

struct KdaTicketSlot {
  deltafin::provider_internal::KdaState next_state;
  DeltafinProviderKdaCacheHandleV1 cache = 0;
  std::uint64_t expected_cache_version = 0;
  std::uint32_t layer_index = 0;
  std::uint64_t spine_generation = 0;
};

struct MlaCacheSlot {
  std::unique_ptr<deltafin::provider_internal::MlaCache> state;
  std::uint32_t layer_index = 0;
};

struct MlaTicketSlot {
  MlaTicketSlot(deltafin::provider_internal::MlaPreparedDecode&& value,
                const DeltafinProviderMlaCacheHandleV1 cache_handle,
                const std::uint32_t ticket_layer,
                const std::uint64_t generation)
      : prepared(std::move(value)),
        cache(cache_handle),
        layer_index(ticket_layer),
        spine_generation(generation) {}

  deltafin::provider_internal::MlaPreparedDecode prepared;
  DeltafinProviderMlaCacheHandleV1 cache = 0;
  std::uint32_t layer_index = 0;
  std::uint64_t spine_generation = 0;
};

struct MoePlanSlot {
  DeltafinProviderTargetSequenceHandleV1 sequence = 0;
  std::uint64_t spine_generation = 0;
  std::uint32_t layer_index = 0;
  std::uint32_t first_row = 0;
  std::uint32_t row_count = 0;
  std::vector<std::uint16_t> canonical_experts;
  std::vector<std::uint16_t> missing_experts;
  deltafin::provider_internal::MoeRunOptions options;
};

struct DSparkSnapshotSlot {
  DeltafinProviderDSparkHandleV1 model = 0;
  deltafin::provider_internal::DSparkCacheSnapshot snapshot;
};

static_assert(std::is_nothrow_move_assignable_v<
                  deltafin::provider_internal::KdaState>,
              "KDA cache commit must remain a no-throw tensor-handle move");

struct Session {
  Session(const deltafin::provider_internal::SelectedDevice& selected_device,
          const std::uint32_t session_flags,
          const std::uint32_t requested_max_route_positions,
          const std::uint32_t requested_hidden_columns,
          const std::uint32_t requested_experts)
      : selected(selected_device),
        flags(session_flags),
        max_route_positions(requested_max_route_positions),
        hidden_columns(requested_hidden_columns),
        experts(requested_experts) {
    if (selected.device.is_cuda()) {
      // PyTorch 2.13's CUDA caching allocator requires explicit initialization
      // before any allocator API (getDeviceStats/emptyCache/allocations) is
      // used; this session is the first CUDA touch in a fresh process and the
      // higher-level PyTorch init paths that would do this are not exercised.
      c10::cuda::CUDACachingAllocator::init(
          c10::cuda::device_count_ensure_non_zero());
      // This is an authoritative target invariant, not a performance hint.
      // Reassert it in require_open() as well so later process-global changes
      // cannot silently switch an already-open K3 session to TF32.
      deltafin::provider_internal::enforce_authoritative_cuda_fp32_precision();
    }
    if ((flags & kSyntheticFlag) != 0) {
      initialize_synthetic_router();
    }
    if (selected.device.is_cuda()) {
      cuda_expert_cache = std::make_unique<
          deltafin::provider_internal::CudaMoeExpertCache>(selected.device);
    }
  }

  std::uint64_t allocate_resource() {
    const std::size_t live = tensors.size() + caches.size() + tickets.size() +
        kda_caches.size() + kda_tickets.size() + mla_caches.size() +
        mla_tickets.size() + moe_plans.size() + dspark_models.size() +
        dspark_snapshots.size() +
        qwen_models.size() + spine_source_uses.size() +
        (target_state_branch == nullptr ? 0 : 1) +
        (target_position == nullptr ? 0 : 1) +
        (target_sequence == nullptr ? 0 : 1);
    if (live >= kMaximumLiveResources) {
      throw std::runtime_error(
          "provider session reached its bounded live-resource limit");
    }
    if (next_resource == 0 ||
        next_resource == std::numeric_limits<std::uint64_t>::max()) {
      throw std::runtime_error("provider resource handle space is exhausted");
    }
    return next_resource++;
  }

  void require_open() const {
    if (closed) {
      throw std::runtime_error("provider session is closed");
    }
    if (selected.device.is_cuda()) {
      deltafin::provider_internal::enforce_authoritative_cuda_fp32_precision();
    }
  }

  void initialize_synthetic_router() {
    auto weight_cpu = at::empty(
        {static_cast<std::int64_t>(hidden_columns),
         static_cast<std::int64_t>(experts)},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    auto bias_cpu = at::empty(
        {static_cast<std::int64_t>(experts)},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    auto weight = weight_cpu.accessor<float, 2>();
    auto bias = bias_cpu.accessor<float, 1>();
    for (std::uint32_t column = 0; column < hidden_columns; ++column) {
      for (std::uint32_t expert = 0; expert < experts; ++expert) {
        const int numerator = static_cast<int>(
            ((column + 1) * (expert + 3) + expert * 5) % 29) - 14;
        weight[column][expert] = static_cast<float>(numerator) / 128.0F;
      }
    }
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      // The small unique correction mirrors K3's noaux_tc choice path and
      // makes tie behavior deterministic for the synthetic contract.
      bias[expert] = static_cast<float>(expert) / 65536.0F;
    }
    const auto options = at::TensorOptions().dtype(at::kFloat).device(
        selected.device);
    router_weight = weight_cpu.to(options);
    router_bias = bias_cpu.to(options);
  }

  deltafin::provider_internal::Bf16CpuT1Kernel& bf16_cpu_t1_kernel() {
    if (!selected.device.is_cpu()) {
      throw std::logic_error(
          "borrowed BF16 CPU kernel requested on a non-CPU session");
    }
    if (bf16_cpu_kernel == nullptr) {
      const std::size_t available =
          std::max<std::size_t>(1, std::thread::hardware_concurrency());
      bf16_cpu_kernel = std::make_unique<
          deltafin::provider_internal::Bf16CpuT1Kernel>(
          std::min<std::size_t>(available, 8));
    }
    return *bf16_cpu_kernel;
  }

  deltafin::provider_internal::ExactBf16DeviceProjector&
  exact_bf16_device_projector() {
    if (!selected.device.is_mps() && !selected.device.is_cuda()) {
      throw std::logic_error(
          "exact BF16 accelerator projector requested on CPU");
    }
    if (exact_bf16_projector == nullptr) {
      exact_bf16_projector = std::make_unique<
          deltafin::provider_internal::ExactBf16DeviceProjector>(
          selected.device);
    }
    return *exact_bf16_projector;
  }

  std::mutex mutex;
  bool closed = false;
  deltafin::provider_internal::SelectedDevice selected;
  std::uint32_t flags;
  std::uint32_t max_route_positions;
  std::uint32_t hidden_columns;
  std::uint32_t experts;
  std::uint64_t next_resource = 1;
  at::Tensor router_weight;
  at::Tensor router_bias;
  std::unordered_map<DeltafinProviderTensorHandleV1, at::Tensor> tensors;
  std::unordered_map<DeltafinProviderCacheHandleV1, CacheSlot> caches;
  std::unordered_map<DeltafinProviderLayerTicketV1, TicketSlot> tickets;
  std::unordered_map<DeltafinProviderKdaCacheHandleV1, KdaCacheSlot>
      kda_caches;
  std::unordered_map<DeltafinProviderKdaTicketHandleV1, KdaTicketSlot>
      kda_tickets;
  std::unordered_map<DeltafinProviderMlaCacheHandleV1, MlaCacheSlot>
      mla_caches;
  std::unordered_map<DeltafinProviderMlaTicketHandleV1,
                     std::unique_ptr<MlaTicketSlot>>
      mla_tickets;
  std::unordered_map<DeltafinProviderMoePlanHandleV1, MoePlanSlot> moe_plans;
  std::unordered_map<DeltafinProviderDSparkHandleV1,
                     std::unique_ptr<deltafin::provider_internal::DSparkModel>>
      dspark_models;
  std::unordered_map<DeltafinProviderDSparkSnapshotHandleV1,
                     DSparkSnapshotSlot>
      dspark_snapshots;
  std::unordered_map<DeltafinProviderQwenHandleV1,
                     std::unique_ptr<deltafin::provider_internal::QwenModel>>
      qwen_models;
  std::unordered_map<DeltafinProviderSpineSourceUseHandleV2,
                     SpineSourceUseSlot>
      spine_source_uses;
  // Retained layers are immutable and append only. Keeping them in a fixed
  // array makes the 93-layer memory bound explicit and avoids a hash lookup
  // on every attention call. All other layers share one replaceable slot.
  std::array<std::unique_ptr<SpineLayerSlot>, kK3Layers>
      resident_spine_layers;
  std::uint32_t resident_spine_prefix_layers = 0;
  std::uint64_t resident_spine_storage_bytes = 0;
  std::unique_ptr<SpineLayerSlot> transient_spine_layer;
  std::uint64_t last_spine_generation = 0;
  std::array<std::unique_ptr<SpineLayerSlot>, 2> target_global_groups;
  std::unique_ptr<deltafin::provider_internal::TargetTailWeights> target_tail;
  // Each slot is published at most once, after its complete authoritative
  // layer has bound successfully. The compact clones own only q8 router,
  // scales, correction bias, and post-attention norm storage; no streamed
  // SpineLayerSlot allocation is retained through this roster.
  deltafin::provider_internal::TargetPilotRoster target_pilot_routers;
  bool target_pilot_enabled = false;
  std::unique_ptr<TargetSessionCacheStore> target_cache_store;
  std::unique_ptr<TargetStateBranchSlot> target_state_branch;
  std::unique_ptr<deltafin::provider_internal::TargetPositionTape>
      target_position;
  DeltafinProviderTargetPositionHandleV1 target_position_handle = 0;
  std::unique_ptr<deltafin::provider_internal::TargetSequenceTape>
      target_sequence;
  DeltafinProviderTargetSequenceHandleV1 target_sequence_handle = 0;
  std::uint64_t committed_target_positions = 0;
  std::uint64_t committed_target_generation = 0;
  std::set<std::pair<std::int64_t, std::int64_t>>
      qualified_target_packed_shapes;
  SpineFp32ExecutionArena spine_fp32_execution_arena;
  bool spine_int8_dense_qualification_attempted = false;
  bool spine_int8_dense_qualified = false;
  bool spine_bf16_dense_qualification_attempted = false;
  bool spine_bf16_dense_qualified = false;
  std::unique_ptr<deltafin::provider_internal::CudaMoeExpertCache>
      cuda_expert_cache;
  std::unique_ptr<deltafin::provider_internal::Bf16CpuT1Kernel>
      bf16_cpu_kernel;
  std::unique_ptr<deltafin::provider_internal::ExactBf16DeviceProjector>
      exact_bf16_projector;
};

std::mutex sessions_mutex;
std::unordered_map<DeltafinProviderSessionHandleV1,
                   std::shared_ptr<Session>> sessions;
std::uint64_t next_session = 1;

std::shared_ptr<Session> find_session(
    const DeltafinProviderSessionHandleV1 handle) {
  if (handle == 0) {
    throw std::invalid_argument("provider session handle is zero");
  }
  std::lock_guard<std::mutex> lock(sessions_mutex);
  const auto found = sessions.find(handle);
  if (found == sessions.end()) {
    throw std::invalid_argument("provider session handle is stale or unknown");
  }
  return found->second;
}

DeltafinProviderSessionHandleV1 insert_session(
    const std::shared_ptr<Session>& session) {
  std::lock_guard<std::mutex> lock(sessions_mutex);
  if (sessions.size() >= 1024) {
    throw std::runtime_error(
        "provider reached its bounded live-session limit");
  }
  if (next_session == 0 ||
      next_session == std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error("provider session handle space is exhausted");
  }
  const DeltafinProviderSessionHandleV1 handle = next_session++;
  const auto [ignored, inserted] = sessions.emplace(handle, session);
  static_cast<void>(ignored);
  if (!inserted) {
    throw std::runtime_error("provider session handle collision");
  }
  return handle;
}

at::Tensor copy_f32_to_device(const float* data, const std::uint64_t rows,
                              const std::uint64_t columns,
                              const at::Device& device, const bool allow_zero) {
  const std::uint64_t elements = checked_elements(rows, columns);
  const auto row_count = checked_dimension(rows, "tensor rows");
  const auto column_count = checked_dimension(columns, "tensor columns");
  const auto target = at::TensorOptions().dtype(at::kFloat).device(device);
  if (data == nullptr) {
    if (!allow_zero) {
      throw std::invalid_argument("provider tensor source pointer is null");
    }
    return at::zeros({row_count, column_count}, target);
  }
  if (device.is_mps()) {
    // MPS may wrap a CPU source page in a no-copy Metal buffer even when the
    // public copy is requested with non_blocking=false.  A caller-owned slice
    // can live on the Rust stack, in a recyclable reader arena, or (for a
    // promoted constant) in the executable mapping itself.  Letting Metal
    // wire that borrowed page past this ABI call caused a reproducible SIGBUS
    // when Rust next executed/read the same __TEXT page.
    //
    // Stage only these small general-purpose fp32 uploads in provider-owned
    // CPU storage before handing them to MPS.  `to(device)` retains that
    // owning Tensor storage for the queued transfer.  Large aligned spine
    // runs keep their separate, explicitly blocking grouped-upload path, so
    // this ownership repair does not restore a full host clone per streamed
    // weight layer. CPU and CUDA also retain their established direct path.
    at::Tensor owned_cpu = at::empty(
        {row_count, column_count},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    std::memcpy(owned_cpu.mutable_data_ptr(), data,
                static_cast<std::size_t>(elements) * sizeof(float));
    return owned_cpu.to(device).contiguous();
  }
  const auto cpu = at::from_blob(
      const_cast<float*>(data), {row_count, column_count},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  static_cast<void>(elements);
  // `copy_` with non_blocking=false is the ownership boundary: the provider
  // allocates the destination first and the call cannot return while it still
  // needs caller storage.  This also avoids clone().to(), which allocated and
  // copied a redundant full host tensor before every provider upload.
  at::Tensor copied = at::empty({row_count, column_count}, target);
  copied.copy_(cpu, false);
  return copied;
}

at::Tensor copy_bf16_to_device(const std::uint8_t* data,
                               const std::uint64_t rows,
                               const std::uint64_t columns,
                               const at::Device& device) {
  const std::uint64_t elements = checked_elements(rows, columns);
  if (data == nullptr || elements > UINT64_MAX / sizeof(std::uint16_t) ||
      elements * sizeof(std::uint16_t) > SIZE_MAX) {
    throw std::invalid_argument("provider BF16 tensor source is invalid");
  }
  const auto row_count = checked_dimension(rows, "BF16 tensor rows");
  const auto column_count = checked_dimension(columns, "BF16 tensor columns");
  at::Tensor cpu = at::empty(
      {row_count, column_count},
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
  std::memcpy(cpu.mutable_data_ptr(), data,
              static_cast<std::size_t>(elements) * sizeof(std::uint16_t));
  return cpu.to(device).contiguous();
}

void copy_f32_to_caller(const at::Tensor& tensor, float* destination,
                        const std::uint64_t capacity) {
  if (destination == nullptr) {
    throw std::invalid_argument("provider tensor destination pointer is null");
  }
  const auto elements = static_cast<std::uint64_t>(tensor.numel());
  if (capacity < elements) {
    throw std::invalid_argument(
        "provider tensor destination does not have enough elements");
  }
  const at::Tensor cpu = tensor.detach().to(at::kCPU).contiguous();
  std::memcpy(destination, cpu.const_data_ptr<float>(),
              static_cast<std::size_t>(elements) * sizeof(float));
}

void require_resource_request(const DeltafinProviderResourceRequestV1* request,
                              const char* name) {
  if (request == nullptr) {
    throw std::invalid_argument(std::string(name) + " pointer is null");
  }
  require_header(request->struct_size, sizeof(*request), request->abi_version,
                 name);
  if (request->flags != 0 || request->reserved0 != 0 ||
      !all_zero(request->reserved)) {
    throw std::invalid_argument(std::string(name) +
                                " contains unknown flags or nonzero reserved fields");
  }
}

void validate_session_dimensions(const DeltafinProviderSessionRequestV1& request) {
  if ((request.flags & ~kKnownSessionFlags) != 0 ||
      !all_zero(request.reserved)) {
    throw std::invalid_argument(
        "provider session request contains unknown flags or reserved fields");
  }
  if (request.max_route_positions == 0 ||
      request.max_route_positions >
          DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1) {
    throw std::invalid_argument(
        "provider max_route_positions is outside the fixed mailbox bound");
  }
  if (std::popcount(request.flags & kKnownSessionFlags) > 1) {
    throw std::invalid_argument(
        "synthetic provider canary session flags are mutually exclusive");
  }
  if ((request.flags & kSyntheticFlag) != 0) {
    if (request.synthetic_hidden_columns == 0 ||
        request.synthetic_hidden_columns > 65536) {
      throw std::invalid_argument(
          "synthetic split hidden width must be in 1..65536");
    }
    if (request.synthetic_experts < DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 ||
        request.synthetic_experts > kK3Experts) {
      throw std::invalid_argument(
          "synthetic split expert count must be in top-k..896");
    }
  } else if (request.synthetic_hidden_columns != 0 ||
             request.synthetic_experts != 0) {
    throw std::invalid_argument(
        "synthetic dimensions require the explicit synthetic-session flag");
  }
}

struct SpineByteRange {
  std::uint32_t buffer = 0;
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  std::uint32_t slot = 0;
};

struct ValidatedSpineDescriptor {
  DeltafinProviderSpineTensorDescriptorV1 raw = {};
  std::vector<std::int64_t> shape;
  std::uint64_t elements = 0;
};

std::uint64_t spine_buffer_length(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const std::uint32_t buffer) {
  switch (buffer) {
    case DELTAFIN_PROVIDER_SPINE_BUFFER_QUANTIZED_V1:
      return request.quantized_length;
    case DELTAFIN_PROVIDER_SPINE_BUFFER_SCALES_V1:
      return request.scales_length;
    case DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1:
      return request.other_length;
    default:
      throw std::invalid_argument("spine descriptor names an invalid buffer");
  }
}

const std::uint8_t* spine_buffer_data(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const std::uint32_t buffer) {
  switch (buffer) {
    case DELTAFIN_PROVIDER_SPINE_BUFFER_QUANTIZED_V1:
      return request.quantized;
    case DELTAFIN_PROVIDER_SPINE_BUFFER_SCALES_V1:
      return request.scales;
    case DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1:
      return request.other;
    default:
      throw std::invalid_argument("spine descriptor names an invalid buffer");
  }
}

void validate_spine_buffer(const std::uint8_t* data,
                           const std::uint64_t length,
                           const char* name) {
  if (length > kMaximumSpineBufferBytes ||
      length > static_cast<std::uint64_t>(SIZE_MAX)) {
    throw std::invalid_argument(std::string("spine ") + name +
                                " buffer exceeds the bounded host size");
  }
  if ((length == 0) != (data == nullptr)) {
    throw std::invalid_argument(std::string("spine ") + name +
                                " pointer/length pair is not canonical");
  }
  if (data != nullptr &&
      reinterpret_cast<std::uintptr_t>(data) % kSpineComponentAlignment != 0) {
    throw std::invalid_argument(std::string("spine ") + name +
                                " buffer is not 256-byte aligned");
  }
}

void validate_spine_allocation(const std::uint8_t* data,
                               const std::uint64_t logical_length,
                               const std::uint64_t allocation_length,
                               const char* name) {
  if (logical_length > allocation_length ||
      allocation_length > static_cast<std::uint64_t>(SIZE_MAX)) {
    throw std::invalid_argument(std::string("spine ") + name +
                                " allocation does not cover its logical bytes");
  }
  if ((allocation_length == 0) != (data == nullptr)) {
    throw std::invalid_argument(std::string("spine ") + name +
                                " allocation pointer/length pair is not canonical");
  }
  if (data != nullptr &&
      reinterpret_cast<std::uintptr_t>(data) % kSpineComponentAlignment != 0) {
    throw std::invalid_argument(std::string("spine ") + name +
                                " allocation is not 256-byte aligned");
  }
  // The detached V1 implementation remains bounded by its logical limit. The
  // rounded allocation may exceed that limit by alignment padding, but never
  // by more than one 64-KiB page on any currently supported host.
  constexpr std::uint64_t kMaximumSpineAllocationPadding = UINT64_C(64) << 10;
  if (allocation_length >
      kMaximumSpineBufferBytes + kMaximumSpineAllocationPadding) {
    throw std::invalid_argument(std::string("spine ") + name +
                                " allocation exceeds the bounded host size");
  }
}

std::uint64_t checked_spine_elements(
    const DeltafinProviderSpineTensorDescriptorV1& descriptor,
    std::vector<std::int64_t>& shape) {
  if (descriptor.rank == 0 || descriptor.rank > 8) {
    throw std::invalid_argument("spine tensor rank must be in 1..8");
  }
  std::uint64_t elements = 1;
  shape.reserve(descriptor.rank);
  for (std::uint32_t index = 0; index < descriptor.rank; ++index) {
    const std::uint64_t dimension = descriptor.shape[index];
    if (dimension == 0 ||
        dimension > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) ||
        elements > kMaximumSpineTensorElements / dimension) {
      throw std::invalid_argument(
          "spine tensor shape is empty, overflows, or exceeds its safety limit");
    }
    elements *= dimension;
    shape.push_back(static_cast<std::int64_t>(dimension));
  }
  for (std::uint32_t index = descriptor.rank; index < 8; ++index) {
    if (descriptor.shape[index] != 0) {
      throw std::invalid_argument(
          "spine tensor has nonzero dimensions beyond its declared rank");
    }
  }
  return elements;
}

void append_spine_range(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const std::uint32_t buffer, const std::uint64_t offset,
    const std::uint64_t length, const std::uint32_t slot,
    std::vector<SpineByteRange>& ranges) {
  if (length == 0 || offset % kSpineComponentAlignment != 0) {
    throw std::invalid_argument(
        "spine tensor byte range is empty or not 256-byte aligned");
  }
  const std::uint64_t buffer_length = spine_buffer_length(request, buffer);
  if (offset > buffer_length || length > buffer_length - offset) {
    throw std::invalid_argument(
        "spine tensor byte range exceeds its LayerBuffers slab");
  }
  const auto* base = spine_buffer_data(request, buffer);
  if (base == nullptr) {
    throw std::invalid_argument("spine tensor uses an absent LayerBuffers slab");
  }
  const auto address = reinterpret_cast<std::uintptr_t>(base);
  const std::uint64_t end = offset + length;
  if (end > std::numeric_limits<std::uintptr_t>::max() - address) {
    throw std::invalid_argument("spine tensor pointer arithmetic overflows");
  }
  ranges.push_back(SpineByteRange{buffer, offset, end, slot});
}

std::vector<ValidatedSpineDescriptor> validate_spine_descriptors(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const std::uint32_t first_slot = kFirstLayerWeightSlot,
    const std::uint32_t last_slot = kLastLayerWeightSlot) {
  if (request.descriptor_count == 0 ||
      request.descriptor_count > kMaximumSpineDescriptors ||
      request.descriptor_count >
          static_cast<std::uint64_t>(SIZE_MAX / sizeof(
              DeltafinProviderSpineTensorDescriptorV1)) ||
      request.descriptors == nullptr) {
    throw std::invalid_argument(
        "spine descriptor array is null, empty, or exceeds its fixed bound");
  }

  // Copy the pointer-bearing request's descriptor array immediately. No
  // caller address is stored in the provider session.
  const auto descriptor_count =
      static_cast<std::size_t>(request.descriptor_count);
  std::vector<DeltafinProviderSpineTensorDescriptorV1> descriptors(
      request.descriptors, request.descriptors + descriptor_count);
  std::vector<ValidatedSpineDescriptor> validated;
  validated.reserve(descriptor_count);
  std::vector<SpineByteRange> ranges;
  ranges.reserve(descriptor_count * 2);
  std::array<bool, kLastGlobalWeightSlot + 1> slots = {};

  for (const auto& descriptor : descriptors) {
    if (first_slot > last_slot || last_slot > kLastGlobalWeightSlot ||
        descriptor.slot < first_slot || descriptor.slot > last_slot) {
      throw std::invalid_argument(
          "spine descriptor has a slot outside its bind contract");
    }
    if (slots[descriptor.slot]) {
      throw std::invalid_argument("spine descriptor repeats a weight slot");
    }
    slots[descriptor.slot] = true;
    if (descriptor.reserved0 != 0 || !all_zero(descriptor.reserved)) {
      throw std::invalid_argument(
          "spine descriptor contains nonzero reserved fields");
    }
    ValidatedSpineDescriptor item;
    item.raw = descriptor;
    item.elements = checked_spine_elements(descriptor, item.shape);

    std::uint64_t expected_data_length = 0;
    switch (descriptor.encoding) {
      case DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1:
        if (item.elements > UINT64_MAX / 2) {
          throw std::invalid_argument("raw-bf16 spine tensor length overflows");
        }
        expected_data_length = item.elements * 2;
        if (descriptor.data_buffer == DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 ||
            descriptor.auxiliary_buffer !=
                DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 ||
            descriptor.auxiliary_offset != 0 ||
            descriptor.auxiliary_length != 0) {
          throw std::invalid_argument(
              "raw-bf16 spine descriptor has invalid buffer fields");
        }
        break;
      case DELTAFIN_PROVIDER_SPINE_RAW_F32_V1:
        if (item.elements > UINT64_MAX / 4) {
          throw std::invalid_argument("raw-f32 spine tensor length overflows");
        }
        expected_data_length = item.elements * 4;
        if (descriptor.data_buffer == DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 ||
            descriptor.auxiliary_buffer !=
                DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 ||
            descriptor.auxiliary_offset != 0 ||
            descriptor.auxiliary_length != 0) {
          throw std::invalid_argument(
              "raw-f32 spine descriptor has invalid buffer fields");
        }
        break;
      case DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1:
        expected_data_length = item.elements;
        if (descriptor.rank != 2 ||
            descriptor.data_buffer ==
                DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 ||
            descriptor.auxiliary_buffer ==
                DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 ||
            descriptor.shape[0] > UINT64_MAX / 2 ||
            descriptor.auxiliary_length != descriptor.shape[0] * 2) {
          throw std::invalid_argument(
              "row-int8 spine descriptor has invalid rank/buffer/scale fields");
        }
        break;
      default:
        throw std::invalid_argument("spine descriptor has an unknown encoding");
    }
    if (descriptor.data_length != expected_data_length) {
      throw std::invalid_argument(
          "spine descriptor byte length does not match its shape/encoding");
    }
    append_spine_range(request, descriptor.data_buffer,
                       descriptor.data_offset, descriptor.data_length,
                       descriptor.slot, ranges);
    if (descriptor.encoding ==
        DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1) {
      append_spine_range(request, descriptor.auxiliary_buffer,
                         descriptor.auxiliary_offset,
                         descriptor.auxiliary_length, descriptor.slot,
                         ranges);
    }
    validated.push_back(std::move(item));
  }

  std::sort(ranges.begin(), ranges.end(), [](const SpineByteRange& left,
                                             const SpineByteRange& right) {
    if (left.buffer != right.buffer) {
      return left.buffer < right.buffer;
    }
    if (left.begin != right.begin) {
      return left.begin < right.begin;
    }
    return left.end < right.end;
  });
  for (std::size_t index = 1; index < ranges.size(); ++index) {
    const auto& previous = ranges[index - 1];
    const auto& current = ranges[index];
    if (previous.buffer == current.buffer && current.begin < previous.end) {
      throw std::invalid_argument(
          "spine descriptor byte ranges overlap within a LayerBuffers slab");
    }
  }
  return validated;
}

const std::uint8_t* spine_component_pointer(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const std::uint32_t buffer, const std::uint64_t offset) {
  // The complete descriptor set and all ranges were validated before this
  // helper can be reached.
  return spine_buffer_data(request, buffer) +
         static_cast<std::size_t>(offset);
}

std::uint64_t spine_scalar_width(const at::ScalarType type) {
  switch (type) {
    case at::kChar:
      return 1;
    case at::kHalf:
    case at::kBFloat16:
    case at::kUInt16:
    case at::kShort:
      return 2;
    case at::kFloat:
      return 4;
    default:
      throw std::logic_error("unsupported spine transfer scalar type");
  }
}

std::uint64_t checked_spine_sum(const std::uint64_t left,
                                const std::uint64_t right,
                                const char* name) {
  if (right > UINT64_MAX - left) {
    throw std::overflow_error(std::string(name) + " overflows uint64");
  }
  return left + right;
}

struct SpineComponentPlan {
  std::size_t descriptor_index = 0;
  bool auxiliary = false;
  bool gathered_mla_input = false;
  std::uint32_t buffer = 0;
  std::uint32_t bundle_order = 0;
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  std::uint64_t elements = 0;
  std::uint64_t target_element_offset = 0;
  at::ScalarType source_type = at::kByte;
  at::ScalarType target_type = at::kByte;
};

struct SpineUploadRun {
  bool gathered = false;
  std::uint32_t buffer = 0;
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  std::uint64_t elements = 0;
  at::ScalarType source_type = at::kByte;
  at::ScalarType target_type = at::kByte;
  std::size_t component_begin = 0;
  std::size_t component_count = 0;
};

bool is_mla_input_bundle_slot(const std::uint32_t slot) {
  return slot == kMlaQueryAProjectionSlot ||
         slot == kMlaKeyValueAProjectionSlot ||
         slot == kMlaOutputGateProjectionSlot;
}

std::uint32_t mla_input_bundle_order(const std::uint32_t slot) {
  switch (slot) {
    case kMlaQueryAProjectionSlot:
      return 0;
    case kMlaKeyValueAProjectionSlot:
      return 1;
    case kMlaOutputGateProjectionSlot:
      return 2;
    default:
      throw std::logic_error("non-MLA projection entered its bundle run");
  }
}

void append_spine_component_plan(
    std::vector<SpineComponentPlan>& components,
    const std::size_t descriptor_index, const bool auxiliary,
    const std::uint32_t buffer, const std::uint64_t begin,
    const std::uint64_t length, const at::ScalarType source_type,
    const at::ScalarType target_type, const bool gathered_mla_input,
    const std::uint32_t bundle_order,
    deltafin::provider_internal::SpineBindingDebugStats& stats) {
  const std::uint64_t source_width = spine_scalar_width(source_type);
  if (length == 0 || length % source_width != 0) {
    throw std::logic_error(
        "validated spine component is not scalar-width aligned");
  }
  const std::uint64_t elements = length / source_width;
  const std::uint64_t target_width = spine_scalar_width(target_type);
  if (elements > UINT64_MAX / target_width) {
    throw std::overflow_error("spine target component byte size overflows");
  }
  stats.source_component_count = checked_spine_sum(
      stats.source_component_count, 1, "spine component count");
  stats.source_component_bytes = checked_spine_sum(
      stats.source_component_bytes, length, "spine source byte count");
  stats.logical_target_bytes = checked_spine_sum(
      stats.logical_target_bytes, elements * target_width,
      "spine target byte count");
  components.push_back(SpineComponentPlan{
      .descriptor_index = descriptor_index,
      .auxiliary = auxiliary,
      .gathered_mla_input = gathered_mla_input,
      .buffer = buffer,
      .bundle_order = bundle_order,
      .begin = begin,
      .end = begin + length,
      .elements = elements,
      .target_element_offset = 0,
      .source_type = source_type,
      .target_type = target_type,
  });
}

std::vector<SpineUploadRun> plan_spine_uploads(
    const std::vector<ValidatedSpineDescriptor>& descriptors,
    const bool gather_mla_inputs,
    deltafin::provider_internal::SpineBindingDebugStats& stats,
    std::vector<SpineComponentPlan>& components,
    std::vector<std::size_t>& ordered_components,
    const std::vector<bool>* borrowed_bf16_cpu = nullptr,
    const std::vector<bool>* exact_bf16 = nullptr,
    const at::ScalarType exact_bf16_type = at::kUInt16) {
  if (exact_bf16 != nullptr && exact_bf16_type != at::kUInt16 &&
      exact_bf16_type != at::kShort) {
    throw std::logic_error(
        "exact BF16 upload requires an opaque 16-bit scalar type");
  }
  components.reserve(descriptors.size() * 2);
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    const auto& descriptor = descriptors[index].raw;
    if (borrowed_bf16_cpu != nullptr &&
        index < borrowed_bf16_cpu->size() && (*borrowed_bf16_cpu)[index]) {
      if (descriptor.encoding != DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1) {
        throw std::logic_error(
            "only original-BF16 components may skip detached upload");
      }
      stats.source_component_count = checked_spine_sum(
          stats.source_component_count, 1, "spine component count");
      stats.source_component_bytes = checked_spine_sum(
          stats.source_component_bytes, descriptor.data_length,
          "spine source byte count");
      continue;
    }
    const bool gathered = gather_mla_inputs &&
                          is_mla_input_bundle_slot(descriptor.slot);
    const std::uint32_t bundle_order =
        gathered ? mla_input_bundle_order(descriptor.slot) : 0;
    switch (descriptor.encoding) {
      case DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1:
        {
        const bool exact = exact_bf16 != nullptr &&
            index < exact_bf16->size() && (*exact_bf16)[index];
        append_spine_component_plan(
            components, index, false, descriptor.data_buffer,
            descriptor.data_offset, descriptor.data_length,
            exact ? exact_bf16_type : at::kBFloat16,
            exact ? exact_bf16_type : at::kFloat, false, 0, stats);
        break;
        }
      case DELTAFIN_PROVIDER_SPINE_RAW_F32_V1:
        append_spine_component_plan(
            components, index, false, descriptor.data_buffer,
            descriptor.data_offset, descriptor.data_length, at::kFloat,
            at::kFloat, false, 0, stats);
        break;
      case DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1:
        append_spine_component_plan(
            components, index, false, descriptor.data_buffer,
            descriptor.data_offset, descriptor.data_length, at::kChar,
            at::kChar, gathered, bundle_order, stats);
        append_spine_component_plan(
            components, index, true, descriptor.auxiliary_buffer,
            descriptor.auxiliary_offset, descriptor.auxiliary_length,
            at::kHalf, at::kFloat, gathered, bundle_order, stats);
        break;
      default:
        throw std::logic_error("validated spine encoding became invalid");
    }
  }

  ordered_components.reserve(components.size());
  for (std::size_t index = 0; index < components.size(); ++index) {
    ordered_components.push_back(index);
  }
  const auto scalar_key = [](const at::ScalarType type) {
    return static_cast<std::underlying_type_t<at::ScalarType>>(type);
  };
  std::sort(ordered_components.begin(), ordered_components.end(),
            [&](const std::size_t left_index,
                const std::size_t right_index) {
              const auto& left = components[left_index];
              const auto& right = components[right_index];
              if (left.gathered_mla_input != right.gathered_mla_input) {
                return !left.gathered_mla_input;
              }
              if (!left.gathered_mla_input) {
                return std::tuple(left.buffer, scalar_key(left.source_type),
                                  scalar_key(left.target_type), left.begin,
                                  left.end) <
                       std::tuple(right.buffer,
                                  scalar_key(right.source_type),
                                  scalar_key(right.target_type), right.begin,
                                  right.end);
              }
              return std::tuple(scalar_key(left.source_type),
                                scalar_key(left.target_type), left.bundle_order,
                                left.auxiliary) <
                     std::tuple(scalar_key(right.source_type),
                                scalar_key(right.target_type),
                                right.bundle_order, right.auxiliary);
            });

  std::vector<SpineUploadRun> runs;
  runs.reserve(ordered_components.size());
  for (std::size_t order_index = 0;
       order_index < ordered_components.size(); ++order_index) {
    const std::size_t component_index = ordered_components[order_index];
    auto& component = components[component_index];
    const bool extend = !runs.empty() &&
        runs.back().gathered == component.gathered_mla_input &&
        runs.back().source_type == component.source_type &&
        runs.back().target_type == component.target_type &&
        (component.gathered_mla_input ||
         (runs.back().buffer == component.buffer &&
          runs.back().end == component.begin));
    if (!extend) {
      SpineUploadRun run;
      run.gathered = component.gathered_mla_input;
      run.buffer = component.buffer;
      run.begin = component.begin;
      run.end = component.end;
      run.elements = component.elements;
      run.source_type = component.source_type;
      run.target_type = component.target_type;
      run.component_begin = order_index;
      run.component_count = 1;
      runs.push_back(std::move(run));
      continue;
    }
    auto& run = runs.back();
    component.target_element_offset = run.elements;
    run.elements = checked_spine_sum(
        run.elements, component.elements,
        component.gathered_mla_input ? "spine gathered-run elements"
                                     : "spine direct-run elements");
    if (!component.gathered_mla_input) {
      run.end = component.end;
    }
    ++run.component_count;
  }

  stats.upload_run_count = runs.size();
  stats.gathered_upload_run_count =
      static_cast<std::uint64_t>(std::count_if(
          runs.begin(), runs.end(),
          [](const SpineUploadRun& run) { return run.gathered; }));
  stats.direct_upload_run_count =
      stats.upload_run_count - stats.gathered_upload_run_count;
  return runs;
}

at::Tensor upload_spine_cpu(const at::Tensor& source_cpu,
                            const std::uint64_t expected_elements,
                            const at::ScalarType target_type,
                            const at::Device& device) {
  if (!source_cpu.defined() || source_cpu.device() != at::Device(at::kCPU) ||
      !source_cpu.is_contiguous() || source_cpu.dim() != 1 ||
      static_cast<std::uint64_t>(source_cpu.numel()) != expected_elements) {
    throw std::logic_error("spine upload received invalid CPU storage");
  }
  // Keep this copy explicitly blocking.  `source_cpu` may be a no-copy view of
  // Rust's recyclable reader arena, so the provider must finish consuming it
  // before the ABI call returns.  Allocating the provider destination first
  // removes the former `from_blob().clone().to()` host clone without retaining
  // a borrowed pointer on CPU, MPS, or CUDA.
  at::Tensor uploaded = at::empty(
      {static_cast<std::int64_t>(expected_elements)},
      at::TensorOptions().dtype(target_type).device(device));
  uploaded.copy_(source_cpu, false);
  const bool matching_device = uploaded.device().type() == device.type() &&
      (!device.has_index() || uploaded.device().index() == device.index());
  if (!uploaded.is_contiguous() || uploaded.dim() != 1 ||
      static_cast<std::uint64_t>(uploaded.numel()) != expected_elements ||
      uploaded.scalar_type() != target_type || !matching_device) {
    throw std::runtime_error(
        "spine grouped upload returned an invalid target tensor");
  }
  return uploaded;
}

at::Tensor upload_direct_spine_run(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const SpineUploadRun& run, const at::Device& device) {
  if (run.gathered || run.end <= run.begin) {
    throw std::logic_error("invalid direct spine upload run");
  }
  const std::uint64_t source_width = spine_scalar_width(run.source_type);
  if ((run.end - run.begin) / source_width != run.elements ||
      run.elements >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::logic_error("direct spine upload run has invalid bounds");
  }
  const auto* source =
      spine_component_pointer(request, run.buffer, run.begin);
  const auto cpu_options =
      at::TensorOptions().dtype(run.source_type).device(at::kCPU);
  if (device.is_mps() && run.source_type == at::kUInt16 &&
      run.target_type == at::kUInt16) {
    // PyTorch/MPS may encode a nominally blocking host-to-device copy by
    // wrapping the CPU source page in a no-copy MTLBuffer.  A from_blob view
    // of the caller's recyclable reader arena has no allocator ownership for
    // MPS to retain, so returning Detached here could otherwise let the
    // caller poison/free those pages before the queued transfer consumes
    // them.  Establish one provider-owned source allocation first.  MPS can
    // then retain that Tensor through its transfer without retaining caller
    // memory; the resident target is still exactly one 16-bit slab.
    at::Tensor owned_cpu = at::empty(
        {static_cast<std::int64_t>(run.elements)}, cpu_options);
    std::memcpy(owned_cpu.mutable_data_ptr(), source,
                static_cast<std::size_t>(run.end - run.begin));
    return upload_spine_cpu(owned_cpu, run.elements, run.target_type, device);
  }
  // This view is borrowed only for the following blocking provider copy.  No
  // tensor that survives this function references the Rust reader arena.
  const at::Tensor borrowed_cpu = at::from_blob(
      const_cast<std::uint8_t*>(source),
      {static_cast<std::int64_t>(run.elements)}, cpu_options);
  return upload_spine_cpu(borrowed_cpu, run.elements, run.target_type, device);
}

at::Tensor upload_gathered_spine_run(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const SpineUploadRun& run,
    const std::vector<SpineComponentPlan>& components,
    const std::vector<std::size_t>& ordered_components,
    const at::Device& device) {
  if (!run.gathered || run.component_count == 0 ||
      run.component_begin > ordered_components.size() ||
      run.component_count > ordered_components.size() - run.component_begin ||
      run.elements >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::logic_error("invalid gathered spine upload run");
  }
  const std::uint64_t width = spine_scalar_width(run.source_type);
  const at::Tensor owned_cpu = at::empty(
      {static_cast<std::int64_t>(run.elements)},
      at::TensorOptions().dtype(run.source_type).device(at::kCPU));
  auto* destination =
      static_cast<std::uint8_t*>(owned_cpu.mutable_data_ptr());
  for (std::size_t offset = 0; offset < run.component_count; ++offset) {
    const std::size_t component_index =
        ordered_components[run.component_begin + offset];
    const auto& component = components[component_index];
    if (component.source_type != run.source_type ||
        component.target_type != run.target_type ||
        component.elements > UINT64_MAX / width ||
        component.target_element_offset > run.elements ||
        component.elements > run.elements - component.target_element_offset) {
      throw std::logic_error(
          "gathered spine upload component escaped its run");
    }
    const auto* source = spine_component_pointer(
        request, component.buffer, component.begin);
    std::memcpy(destination + component.target_element_offset * width, source,
                static_cast<std::size_t>(component.elements * width));
  }
  return upload_spine_cpu(owned_cpu, run.elements, run.target_type, device);
}

bool is_kda_layer(const std::uint32_t layer_index) {
  if (layer_index >= kK3Layers) {
    return false;
  }
  const std::uint32_t one_based = layer_index + 1;
  return one_based != kK3Layers && one_based % 4 != 0;
}

bool is_mla_layer(const std::uint32_t layer_index) {
  if (layer_index >= kK3Layers) {
    return false;
  }
  const std::uint32_t one_based = layer_index + 1;
  return one_based == kK3Layers || one_based % 4 == 0;
}

constexpr std::uint64_t kSpineFp32ArenaAlignmentElements = 64;
// Exact real-inventory maximum: layer 0's eleven row-quantized matrices.
constexpr std::uint64_t kSpineFp32ArenaMaximumElements = 1170243584ULL;

std::uint64_t align_spine_fp32_elements(const std::uint64_t elements) {
  constexpr std::uint64_t mask = kSpineFp32ArenaAlignmentElements - 1;
  static_assert((kSpineFp32ArenaAlignmentElements & mask) == 0);
  if (elements > UINT64_MAX - mask) {
    throw std::overflow_error("FP32 spine arena alignment overflows uint64");
  }
  return (elements + mask) & ~mask;
}

bool qualify_spine_int8_dense_mps(Session& session) {
  if (!session.selected.device.is_mps()) {
    return false;
  }
  if (session.spine_int8_dense_qualification_attempted) {
    return session.spine_int8_dense_qualified;
  }
  session.spine_int8_dense_qualification_attempted = true;
#if defined(__APPLE__)
  try {
    const auto capabilities =
        deltafin::provider_internal::spine_int8_metal_capabilities_v1();
    if (capabilities.abi_version !=
            deltafin::provider_internal::kSpineInt8MetalAbiV1 ||
        capabilities.flags !=
            deltafin::provider_internal::
                kSpineInt8MetalRequiredCapabilitiesV1 ||
        capabilities.threads_per_threadgroup != 256 ||
        capabilities.reserved != 0) {
      return false;
    }
    const auto report =
        deltafin::provider_internal::spine_int8_metal_canary_v1();
    session.spine_int8_dense_qualified =
        report.rows != 0 && report.columns != 0 &&
        report.compared_elements == report.rows * report.columns &&
        report.equal_bits == report.compared_elements &&
        report.nonfinite == 0 && report.source_offset_elements != 0 &&
        report.scale_offset_elements != 0 &&
        report.destination_offset_elements != 0;
  } catch (...) {
    session.spine_int8_dense_qualified = false;
  }
#endif
  return session.spine_int8_dense_qualified;
}

/*
 * Qualify only the established ATen dtype-reinterpretation and cast path used
 * to expand opaque checkpoint BF16 bits into the shared FP32 execution arena.
 * This is intentionally not a custom projection canary: all subsequent
 * matrix multiplication remains with LibTorch's normal MPS provider.
 * CUDA deliberately keeps its separately qualified raw-BF16 path until an
 * equivalent end-to-end stream-lifetime and throughput gate exists there.
 */
bool qualify_spine_bf16_dense_accelerator(Session& session) {
  const at::Device device = session.selected.device;
  if (!device.is_mps()) {
    return false;
  }
  if (session.spine_bf16_dense_qualification_attempted) {
    return session.spine_bf16_dense_qualified;
  }
  session.spine_bf16_dense_qualification_attempted = true;
  try {
    constexpr std::array<std::uint16_t, 4> bits{
        0x3f80U, 0xc020U, 0x0001U, 0x7f7fU};
    const at::ScalarType storage_type = at::kUInt16;
    at::Tensor storage_cpu = at::empty(
        {static_cast<std::int64_t>(bits.size())},
        at::TensorOptions().dtype(storage_type).device(at::kCPU));
    static_assert(sizeof(std::uint16_t) == sizeof(std::int16_t));
    std::memcpy(storage_cpu.mutable_data_ptr(), bits.data(), sizeof(bits));
    const at::Tensor storage = storage_cpu.to(device).contiguous();
    const at::Tensor decoded =
        storage.view(at::kBFloat16).to(at::kFloat).to(at::kCPU).contiguous();
    if (decoded.scalar_type() != at::kFloat ||
        decoded.numel() != static_cast<std::int64_t>(bits.size())) {
      return false;
    }
    const float* values = decoded.const_data_ptr<float>();
    for (std::size_t index = 0; index < bits.size(); ++index) {
      if (std::bit_cast<std::uint32_t>(values[index]) !=
          (static_cast<std::uint32_t>(bits[index]) << 16)) {
        return false;
      }
    }
    session.spine_bf16_dense_qualified = true;
  } catch (...) {
    session.spine_bf16_dense_qualified = false;
  }
  return session.spine_bf16_dense_qualified;
}

/*
 * Materialize the current compact projection layer into one reusable FP32
 * execution template. Row-int8 retains the already-qualified Metal dequant
 * used by the live path. Original BF16 reinterprets the provider-owned raw
 * bits and asks ATen to cast them to FP32; its matrix multiplication therefore
 * stays on the established MPS provider instead of the slower custom
 * direct-GEMV candidate.
 *
 * Returning nullopt before dispatch preserves the representation's exact
 * fallback. Any error after dispatch begins aborts the enclosing target
 * transaction so a partially written arena is never published.
 */
std::optional<SpineFp32ExecutionView> maybe_materialize_spine_fp32(
    Session& session, const SpineLayerSlot& layer,
    const std::uint64_t owner) {
  const at::Device device = session.selected.device;
  if (!device.is_mps() || owner == 0) {
    return std::nullopt;
  }
  SpineFp32ExecutionSource source = SpineFp32ExecutionSource::None;
  std::uint64_t required_elements = 0;
  std::uint32_t matrix_count = 0;
  for (const auto& optional : layer.tensors) {
    if (!optional.has_value()) {
      continue;
    }
    const SpineTensorSlot& tensor = *optional;
    SpineFp32ExecutionSource candidate = SpineFp32ExecutionSource::None;
    std::uint64_t elements = 0;
    if (tensor.encoding ==
        DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1) {
      candidate = SpineFp32ExecutionSource::RowInt8;
      if (!tensor.data.defined() || !tensor.auxiliary.defined() ||
          tensor.data.scalar_type() != at::kChar ||
          tensor.auxiliary.scalar_type() != at::kFloat ||
          tensor.data.device() != device ||
          tensor.auxiliary.device() != device ||
          !tensor.data.is_contiguous() ||
          !tensor.auxiliary.is_contiguous() || tensor.data.dim() != 2 ||
          tensor.auxiliary.dim() != 1 ||
          tensor.auxiliary.size(0) != tensor.data.size(0)) {
        throw std::invalid_argument(
            "FP32 spine materialization received an invalid row-int8 matrix");
      }
      elements = static_cast<std::uint64_t>(tensor.data.numel());
    } else if (tensor.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 &&
               tensor.original_bf16.defined()) {
      candidate = SpineFp32ExecutionSource::OriginalBf16;
      const auto& matrix = tensor.original_bf16;
      if (!matrix.is_owned() || matrix.is_borrowed_cpu() ||
          matrix.owned_storage == nullptr ||
          !matrix.owned_storage->tensor.defined() ||
          matrix.owned_storage->tensor.device() != device ||
          !matrix.owned_storage->tensor.is_contiguous() ||
          matrix.owned_storage->tensor.dim() != 1 ||
          (matrix.owned_storage->tensor.scalar_type() != at::kUInt16 &&
           matrix.owned_storage->tensor.scalar_type() != at::kShort) ||
          tensor.data.defined() || tensor.auxiliary.defined() ||
          tensor.shape.size() != 2 || tensor.shape[0] <= 0 ||
          tensor.shape[1] <= 0 ||
          matrix.rows != static_cast<std::size_t>(tensor.shape[0]) ||
          matrix.columns != static_cast<std::size_t>(tensor.shape[1]) ||
          matrix.rows > std::numeric_limits<std::size_t>::max() /
                            matrix.columns) {
        throw std::invalid_argument(
            "FP32 spine materialization received invalid original-BF16 storage");
      }
      const std::size_t matrix_elements = matrix.rows * matrix.columns;
      const std::size_t storage_elements = static_cast<std::size_t>(
          matrix.owned_storage->tensor.numel());
      if (matrix.owned_element_offset > storage_elements ||
          matrix_elements > storage_elements - matrix.owned_element_offset) {
        throw std::invalid_argument(
            "FP32 spine materialization original-BF16 view is out of bounds");
      }
      elements = static_cast<std::uint64_t>(matrix_elements);
    } else {
      continue;
    }
    if (source != SpineFp32ExecutionSource::None && source != candidate) {
      throw std::invalid_argument(
          "one target layer mixed row-int8 and original-BF16 projections");
    }
    source = candidate;
    required_elements = align_spine_fp32_elements(required_elements);
    required_elements = checked_spine_sum(
        required_elements, elements, "FP32 spine arena elements");
    ++matrix_count;
  }
  required_elements = align_spine_fp32_elements(required_elements);
  if (matrix_count == 0) {
    return std::nullopt;
  }
  if (required_elements == 0 ||
      required_elements > kSpineFp32ArenaMaximumElements ||
      required_elements >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(
        "FP32 spine arena request exceeds the audited K3 maximum");
  }

  // Check representation/provider capability before arena sequencing so a
  // disabled layer-zero attempt cannot make layer one look like an illegal
  // first arena use.
  const bool qualified =
      source == SpineFp32ExecutionSource::RowInt8
          ? (device.is_mps() && qualify_spine_int8_dense_mps(session))
          : qualify_spine_bf16_dense_accelerator(session);
  if (!qualified) {
    return std::nullopt;
  }

  SpineFp32ExecutionArena& arena = session.spine_fp32_execution_arena;
  if (arena.occupied) {
    if (arena.owner == owner) {
      if (layer.layer_index <= arena.layer_index) {
        throw std::runtime_error(
            "FP32 spine arena cannot overwrite an unfinished target layer");
      }
    } else if (layer.layer_index != 0) {
      throw std::runtime_error(
          "FP32 spine arena owner changed outside a layer-zero boundary");
    }
  } else if (layer.layer_index != 0) {
    throw std::runtime_error(
        "FP32 spine arena first use must begin at target layer zero");
  }

  at::Tensor storage = arena.storage;
  if (!storage.defined() ||
      static_cast<std::uint64_t>(storage.numel()) < required_elements) {
    // Allocation happens before any current-layer dispatch, so failure can
    // safely retain the representation's established exact fallback.
    try {
      storage = at::empty(
          {static_cast<std::int64_t>(required_elements)},
          at::TensorOptions().dtype(at::kFloat).device(device));
    } catch (...) {
      if (source == SpineFp32ExecutionSource::RowInt8) {
        session.spine_int8_dense_qualified = false;
      } else {
        session.spine_bf16_dense_qualified = false;
      }
      return std::nullopt;
    }
  }

  SpineFp32ExecutionView view;
  view.owner = owner;
  view.spine_generation = layer.generation;
  view.layer_index = layer.layer_index;
  view.required_elements = required_elements;
  std::uint64_t cursor = 0;
  bool dispatched = false;
  try {
    for (std::size_t slot = 0; slot < layer.tensors.size(); ++slot) {
      const auto& optional = layer.tensors[slot];
      if (!optional.has_value()) {
        continue;
      }
      const SpineTensorSlot& tensor = *optional;
      const bool row_int8 =
          source == SpineFp32ExecutionSource::RowInt8 &&
          tensor.encoding ==
              DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1;
      const bool original_bf16 =
          source == SpineFp32ExecutionSource::OriginalBf16 &&
          tensor.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 &&
          tensor.original_bf16.defined();
      if (!row_int8 && !original_bf16) {
        continue;
      }
      cursor = align_spine_fp32_elements(cursor);
      const std::int64_t elements = row_int8
          ? tensor.data.numel()
          : static_cast<std::int64_t>(
                tensor.original_bf16.rows *
                tensor.original_bf16.columns);
      at::Tensor destination = storage
          .narrow(0, static_cast<std::int64_t>(cursor), elements)
          .view(tensor.shape);
      dispatched = true;
      if (original_bf16) {
        const auto& matrix = tensor.original_bf16;
        const at::Tensor source_bits = matrix.owned_storage->tensor
            .narrow(0,
                    static_cast<std::int64_t>(matrix.owned_element_offset),
                    elements)
            .view(tensor.shape);
        // Equal-width dtype view is zero-copy. copy_ delegates the BF16->FP32
        // conversion to ATen's selected accelerator provider, matching the
        // proven compiled path instead of implementing another GEMV.
        destination.copy_(source_bits.view(at::kBFloat16), false);
      } else {
#if defined(__APPLE__)
        deltafin::provider_internal::spine_int8_metal_dequant_f32(
            destination, tensor.data, tensor.auxiliary);
#else
        throw std::logic_error(
            "MPS FP32 spine materialization reached a non-Apple build");
#endif
      }
      view.tensors[slot].emplace(std::move(destination));
      cursor = checked_spine_sum(
          cursor, static_cast<std::uint64_t>(elements),
          "FP32 spine arena cursor");
    }
  } catch (...) {
    // A partially encoded layer is never published as execution weights.
    // Compact source storage remains provider-owned, and the selected-device
    // stream orders any already-enqueued writes before later cleanup/reuse.
    if (source == SpineFp32ExecutionSource::RowInt8) {
      session.spine_int8_dense_qualified = false;
    } else {
      session.spine_bf16_dense_qualified = false;
    }
    if (dispatched) {
      throw;
    }
    return std::nullopt;
  }
  if (cursor > required_elements) {
    throw std::logic_error("FP32 spine arena cursor escaped its allocation");
  }
  arena.storage = std::move(storage);
  arena.owner = owner;
  arena.spine_generation = layer.generation;
  arena.layer_index = layer.layer_index;
  arena.occupied = true;
  return view;
}

const at::Tensor* spine_fp32_execution_tensor(
    const SpineFp32ExecutionView* execution, const std::uint32_t slot) {
  if (execution == nullptr || slot >= execution->tensors.size() ||
      !execution->tensors[slot].has_value()) {
    return nullptr;
  }
  return &*execution->tensors[slot];
}

bool spine_fp32_execution_matches(const SpineTensorSlot& source,
                                  const at::Tensor& dense,
                                  const at::IntArrayRef shape) {
  std::optional<at::Device> source_device;
  if (source.encoding ==
          DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1 &&
      source.data.defined() && source.auxiliary.defined() &&
      source.data.scalar_type() == at::kChar &&
      source.auxiliary.scalar_type() == at::kFloat &&
      source.data.is_contiguous() && source.auxiliary.is_contiguous() &&
      source.data.sizes() == shape && source.auxiliary.dim() == 1 &&
      shape.size() == 2 && source.auxiliary.size(0) == shape[0] &&
      source.data.device() == source.auxiliary.device()) {
    source_device = source.data.device();
  } else if (source.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 &&
             !source.data.defined() && !source.auxiliary.defined() &&
             source.original_bf16.is_owned() &&
             source.original_bf16.owned_storage != nullptr &&
             source.original_bf16.owned_storage->tensor.defined() &&
             shape.size() == 2 && shape[0] > 0 && shape[1] > 0 &&
             source.original_bf16.rows ==
                 static_cast<std::size_t>(shape[0]) &&
             source.original_bf16.columns ==
                 static_cast<std::size_t>(shape[1])) {
    source_device =
        source.original_bf16.owned_storage->tensor.device();
  }
  return source_device.has_value() && dense.defined() &&
         dense.scalar_type() == at::kFloat && dense.is_contiguous() &&
         dense.sizes() == shape && dense.device() == *source_device;
}

const SpineTensorSlot& require_kda_spine_slot(
    const SpineLayerSlot& layer, const std::uint32_t slot,
    const std::uint32_t encoding, const char* name) {
  if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
    throw std::invalid_argument(std::string("KDA spine is missing ") + name);
  }
  const SpineTensorSlot& tensor = *layer.tensors[slot];
  if (tensor.encoding != encoding) {
    throw std::invalid_argument(std::string("KDA spine ") + name +
                                " has the wrong storage encoding");
  }
  return tensor;
}

at::Tensor require_kda_raw(const SpineLayerSlot& layer,
                           const std::uint32_t slot, const char* name) {
  const SpineTensorSlot& tensor = require_kda_spine_slot(
      layer, slot, DELTAFIN_PROVIDER_SPINE_RAW_F32_V1, name);
  if (!tensor.data.defined() || tensor.auxiliary.defined()) {
    throw std::invalid_argument(std::string("KDA spine ") + name +
                                " has invalid raw components");
  }
  return tensor.data;
}

deltafin::provider_internal::KdaProjection require_kda_projection(
    const SpineLayerSlot& layer, const std::uint32_t slot,
    const char* name,
    const SpineFp32ExecutionView* execution = nullptr) {
  if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
    throw std::invalid_argument(std::string("KDA spine is missing ") + name);
  }
  const SpineTensorSlot& tensor = *layer.tensors[slot];
  if (const at::Tensor* dense =
          spine_fp32_execution_tensor(execution, slot);
      dense != nullptr) {
    if (!spine_fp32_execution_matches(tensor, *dense, tensor.shape)) {
      throw std::invalid_argument(std::string("KDA spine ") + name +
                                  " has an invalid FP32 execution view");
    }
    return deltafin::provider_internal::KdaProjection{
        *dense, at::Tensor(), {}};
  }
  if (tensor.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1) {
    const bool dense = tensor.data.defined();
    const bool original = tensor.original_bf16.defined();
    if (dense == original || tensor.auxiliary.defined() ||
        (dense && tensor.data.scalar_type() != at::kFloat)) {
      throw std::invalid_argument(std::string("KDA spine ") + name +
                                  " has invalid original-BF16 storage");
    }
    return deltafin::provider_internal::KdaProjection{
        tensor.data, at::Tensor(), tensor.original_bf16};
  }
  if (tensor.encoding != DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1 ||
      !tensor.data.defined() || tensor.data.scalar_type() != at::kChar ||
      !tensor.auxiliary.defined() ||
      tensor.auxiliary.scalar_type() != at::kFloat) {
    throw std::invalid_argument(std::string("KDA spine ") + name +
                                " is neither original BF16 nor row-int8");
  }
  return deltafin::provider_internal::KdaProjection{
      tensor.data, tensor.auxiliary};
}

deltafin::provider_internal::KdaWeights kda_weights_from_spine(
    const SpineLayerSlot& layer,
    const SpineFp32ExecutionView* execution = nullptr) {
  if (!is_kda_layer(layer.layer_index)) {
    throw std::invalid_argument(
        "KDA decode refuses a full-attention layer binding");
  }
  return deltafin::provider_internal::KdaWeights{
      require_kda_raw(layer, kKdaALogSlot, "A_log"),
      require_kda_raw(layer, kKdaDtBiasSlot, "dt_bias"),
      require_kda_raw(layer, kKdaQueryConvolutionSlot,
                      "query convolution"),
      require_kda_raw(layer, kKdaKeyConvolutionSlot, "key convolution"),
      require_kda_raw(layer, kKdaValueConvolutionSlot,
                      "value convolution"),
      require_kda_raw(layer, kKdaOutputNormSlot, "output norm"),
      require_kda_projection(layer, kKdaQueryProjectionSlot,
                             "query projection", execution),
      require_kda_projection(layer, kKdaKeyProjectionSlot,
                             "key projection", execution),
      require_kda_projection(layer, kKdaValueProjectionSlot,
                             "value projection", execution),
      require_kda_projection(layer, kKdaGateProjectionSlot,
                             "output-gate projection", execution),
      require_kda_projection(layer, kKdaFeatureAProjectionSlot,
                             "feature-a projection", execution),
      require_kda_projection(layer, kKdaFeatureBProjectionSlot,
                             "feature-b projection", execution),
      require_kda_projection(layer, kKdaBetaProjectionSlot,
                             "beta projection", execution),
      require_kda_projection(layer, kKdaOutputProjectionSlot,
                             "output projection", execution),
  };
}

const SpineTensorSlot& require_mla_spine_slot(
    const SpineLayerSlot& layer, const std::uint32_t slot,
    const std::uint32_t encoding, const char* name) {
  if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
    throw std::invalid_argument(std::string("MLA spine is missing ") + name);
  }
  const SpineTensorSlot& tensor = *layer.tensors[slot];
  if (tensor.encoding != encoding) {
    throw std::invalid_argument(std::string("MLA spine ") + name +
                                " has the wrong storage encoding");
  }
  return tensor;
}

at::Tensor require_mla_norm(const SpineLayerSlot& layer,
                            const std::uint32_t slot, const char* name) {
  const SpineTensorSlot& tensor = require_mla_spine_slot(
      layer, slot, DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, name);
  if (!tensor.data.defined() || tensor.auxiliary.defined()) {
    throw std::invalid_argument(std::string("MLA spine ") + name +
                                " has invalid raw components");
  }
  return tensor.data;
}

deltafin::provider_internal::MlaLinearWeight require_mla_projection(
    const SpineLayerSlot& layer, const std::uint32_t slot,
    const char* name,
    const SpineFp32ExecutionView* execution = nullptr) {
  if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
    throw std::invalid_argument(std::string("MLA spine is missing ") + name);
  }
  const SpineTensorSlot& tensor = *layer.tensors[slot];
  if (const at::Tensor* dense =
          spine_fp32_execution_tensor(execution, slot);
      dense != nullptr) {
    if (!spine_fp32_execution_matches(tensor, *dense, tensor.shape)) {
      throw std::invalid_argument(std::string("MLA spine ") + name +
                                  " has an invalid FP32 execution view");
    }
    return deltafin::provider_internal::MlaLinearWeight{
        deltafin::provider_internal::MlaLinearEncoding::DenseF32,
        *dense, at::Tensor(), {}};
  }
  if (tensor.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1) {
    const bool dense = tensor.data.defined();
    const bool original = tensor.original_bf16.defined();
    if (dense == original || tensor.auxiliary.defined() ||
        (dense && tensor.data.scalar_type() != at::kFloat)) {
      throw std::invalid_argument(std::string("MLA spine ") + name +
                                  " has invalid original-BF16 storage");
    }
    return deltafin::provider_internal::MlaLinearWeight{
        tensor.original_bf16.defined()
            ? deltafin::provider_internal::MlaLinearEncoding::
                  OriginalBf16
            : deltafin::provider_internal::MlaLinearEncoding::DenseF32,
        tensor.data, at::Tensor(), tensor.original_bf16};
  }
  if (tensor.encoding != DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1 ||
      !tensor.data.defined() || tensor.data.scalar_type() != at::kChar ||
      !tensor.auxiliary.defined() ||
      tensor.auxiliary.scalar_type() != at::kFloat) {
    throw std::invalid_argument(std::string("MLA spine ") + name +
                                " is neither original BF16 nor row-int8");
  }
  return deltafin::provider_internal::MlaLinearWeight{
      deltafin::provider_internal::MlaLinearEncoding::RowI8F32Scale,
      tensor.data, tensor.auxiliary, {}};
}

deltafin::provider_internal::MlaWeights mla_weights_from_spine(
    const SpineLayerSlot& layer,
    const SpineFp32ExecutionView* execution = nullptr) {
  if (!is_mla_layer(layer.layer_index)) {
    throw std::invalid_argument(
        "MLA decode refuses a recurrent-attention layer binding");
  }
  return deltafin::provider_internal::MlaWeights{
      require_mla_projection(layer, kMlaQueryAProjectionSlot,
                             "query-a projection", execution),
      require_mla_norm(layer, kMlaQueryANormSlot, "query-a norm"),
      require_mla_projection(layer, kMlaQueryBProjectionSlot,
                             "query-b projection", execution),
      require_mla_projection(layer, kMlaKeyValueAProjectionSlot,
                             "key/value-a projection", execution),
      require_mla_norm(layer, kMlaKeyValueANormSlot, "key/value-a norm"),
      require_mla_projection(layer, kMlaKeyValueBProjectionSlot,
                             "key/value-b projection", execution),
      require_mla_projection(layer, kMlaOutputGateProjectionSlot,
                             "output-gate projection", execution),
      require_mla_projection(layer, kMlaOutputProjectionSlot,
                             "output projection", execution),
  };
}

const SpineTensorSlot& require_target_spine_slot(
    const SpineLayerSlot& layer, const std::uint32_t slot,
    const std::uint32_t encoding, const at::IntArrayRef shape,
    const char* name) {
  if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
    throw std::invalid_argument(std::string("target spine is missing ") +
                                name);
  }
  const SpineTensorSlot& tensor = *layer.tensors[slot];
  if (tensor.encoding != encoding || tensor.shape != shape.vec() ||
      !tensor.data.defined() || !tensor.data.is_contiguous() ||
      tensor.data.sizes() != shape) {
    throw std::invalid_argument(std::string("target spine ") + name +
                                " has the wrong encoding or shape");
  }
  const bool expects_auxiliary =
      encoding == DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1;
  if (tensor.auxiliary.defined() != expects_auxiliary) {
    throw std::invalid_argument(std::string("target spine ") + name +
                                " has invalid component ownership");
  }
  if (expects_auxiliary &&
      (!tensor.auxiliary.is_contiguous() || tensor.auxiliary.dim() != 1 ||
       tensor.auxiliary.size(0) != shape[0])) {
    throw std::invalid_argument(std::string("target spine ") + name +
                                " has invalid fp32 row scales");
  }
  return tensor;
}

at::Tensor require_target_raw(
    const SpineLayerSlot& layer, const std::uint32_t slot,
    const std::uint32_t encoding, const at::IntArrayRef shape,
    const char* name) {
  return require_target_spine_slot(layer, slot, encoding, shape, name).data;
}

deltafin::provider_internal::MoeRowInt8Matrix require_target_linear(
    const SpineLayerSlot& layer, const std::uint32_t slot,
    const std::int64_t rows, const std::int64_t columns, const char* name,
    const SpineFp32ExecutionView* execution = nullptr) {
  if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
    throw std::invalid_argument(std::string("target spine is missing ") +
                                name);
  }
  const SpineTensorSlot& tensor = *layer.tensors[slot];
  if (const at::Tensor* dense =
          spine_fp32_execution_tensor(execution, slot);
      dense != nullptr) {
    if (!spine_fp32_execution_matches(
            tensor, *dense, at::IntArrayRef({rows, columns}))) {
      throw std::invalid_argument(std::string("target spine ") + name +
                                  " has an invalid FP32 execution view");
    }
    return deltafin::provider_internal::MoeRowInt8Matrix{
        at::Tensor(), at::Tensor(), *dense, {}};
  }
  const bool original = tensor.original_bf16.defined();
  if ((!original &&
       (!tensor.data.defined() || !tensor.data.is_contiguous() ||
        tensor.data.sizes() != at::IntArrayRef({rows, columns}))) ||
      (original &&
       (tensor.data.defined() ||
        tensor.original_bf16.rows != static_cast<std::size_t>(rows) ||
        tensor.original_bf16.columns !=
            static_cast<std::size_t>(columns)))) {
    throw std::invalid_argument(std::string("target spine ") + name +
                                " has the wrong matrix shape");
  }
  if (tensor.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1) {
    if ((!original && tensor.data.scalar_type() != at::kFloat) ||
        tensor.auxiliary.defined()) {
      throw std::invalid_argument(std::string("target spine ") + name +
                                  " has invalid original-BF16 storage");
    }
    return deltafin::provider_internal::MoeRowInt8Matrix{
        at::Tensor(), at::Tensor(), tensor.data,
        tensor.original_bf16};
  }
  if (tensor.encoding != DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1 ||
      tensor.data.scalar_type() != at::kChar ||
      !tensor.auxiliary.defined() ||
      tensor.auxiliary.scalar_type() != at::kFloat ||
      tensor.auxiliary.sizes() != at::IntArrayRef({rows}) ||
      tensor.data.device() != tensor.auxiliary.device()) {
    throw std::invalid_argument(std::string("target spine ") + name +
                                " is neither original BF16 nor row-int8");
  }
  return deltafin::provider_internal::MoeRowInt8Matrix{
      tensor.data, tensor.auxiliary, at::Tensor(), {}};
}

bool target_linear_is_packed(
    const deltafin::provider_internal::MoeRowInt8Matrix& matrix) {
  return matrix.quantized.defined();
}

bool has_complete_target_residual_roster(const SpineLayerSlot& layer) {
  for (const std::uint32_t slot :
       {kInputNormSlot, kPostAttentionNormSlot,
        kAttentionResidualNormSlot, kAttentionResidualProjectionSlot,
        kMlpResidualNormSlot, kMlpResidualProjectionSlot}) {
    if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
      return false;
    }
  }
  return true;
}

deltafin::provider_internal::TargetResidualWeights
target_residual_from_spine(const SpineLayerSlot& layer) {
  constexpr std::int64_t hidden = 7168;
  return deltafin::provider_internal::TargetResidualWeights{
      require_target_raw(layer, kInputNormSlot,
                         DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {hidden},
                         "input norm"),
      require_target_raw(layer, kAttentionResidualNormSlot,
                         DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {hidden},
                         "attention-residual norm"),
      require_target_raw(layer, kAttentionResidualProjectionSlot,
                         DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {1, hidden},
                         "attention-residual projection"),
      require_target_raw(layer, kPostAttentionNormSlot,
                         DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {hidden},
                         "post-attention norm"),
      require_target_raw(layer, kMlpResidualNormSlot,
                         DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {hidden},
                         "MLP-residual norm"),
      require_target_raw(layer, kMlpResidualProjectionSlot,
                         DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {1, hidden},
                         "MLP-residual projection")};
}

void maybe_prepare_target_residual(SpineLayerSlot& layer) {
  if (!has_complete_target_residual_roster(layer)) {
    return;
  }
  auto weights = target_residual_from_spine(layer);
  layer.target_residual =
      std::make_unique<deltafin::provider_internal::TargetResidualWeights>(
          deltafin::provider_internal::precompute_target_residual_score_weights(
              std::move(weights)));
}

deltafin::provider_internal::TargetDenseWeights target_dense_from_spine(
    const SpineLayerSlot& layer,
    const SpineFp32ExecutionView* execution = nullptr) {
  constexpr std::int64_t hidden = 7168;
  constexpr std::int64_t intermediate = 33792;
  auto gate = require_target_linear(layer, kDenseGateProjectionSlot,
                                    intermediate, hidden,
                                    "dense gate projection", execution);
  auto up = require_target_linear(layer, kDenseUpProjectionSlot,
                                  intermediate, hidden,
                                  "dense up projection", execution);
  auto down = require_target_linear(layer, kDenseDownProjectionSlot, hidden,
                                    intermediate, "dense down projection",
                                    execution);
  const bool packed = target_linear_is_packed(gate);
  if (target_linear_is_packed(up) != packed ||
      target_linear_is_packed(down) != packed) {
    throw std::invalid_argument(
        "target dense layer mixes original-BF16 and row-int8 projections");
  }
  return deltafin::provider_internal::TargetDenseWeights{
      std::move(gate), std::move(up), std::move(down), packed};
}

void maybe_bundle_target_dense_zero_copy(
    deltafin::provider_internal::TargetDenseWeights& weights) {
  if (!weights.packed_int8_qualified) {
    const at::Tensor& gate_dense = weights.gate.dense_f32;
    const at::Tensor& up_dense = weights.up.dense_f32;
    if (gate_dense.defined() && up_dense.defined() &&
        gate_dense.dim() == 2 && up_dense.sizes() == gate_dense.sizes() &&
        gate_dense.is_contiguous() && up_dense.is_contiguous() &&
        gate_dense.is_alias_of(up_dense) &&
        gate_dense.storage_offset() + gate_dense.numel() ==
            up_dense.storage_offset() &&
        gate_dense.size(0) <=
            std::numeric_limits<std::int64_t>::max() / 2) {
      const std::int64_t rows = gate_dense.size(0);
      const std::int64_t columns = gate_dense.size(1);
      at::Tensor dense_bundle = gate_dense.as_strided(
          {rows * 2, columns}, {columns, 1},
          gate_dense.storage_offset());
      if (dense_bundle.is_contiguous()) {
        weights.gate_up = deltafin::provider_internal::MoeRowInt8Matrix{
            at::Tensor(), at::Tensor(), std::move(dense_bundle), {}};
        weights.bundled_gate_up_qualified = true;
        return;
      }
    }
    const std::array<
        const deltafin::provider_internal::OriginalBf16Matrix*, 2>
        matrices{&weights.gate.original_bf16,
                 &weights.up.original_bf16};
    auto combined =
        deltafin::provider_internal::adjacent_original_bf16_matrices(
            matrices);
    if (combined.has_value()) {
      weights.gate_up = deltafin::provider_internal::MoeRowInt8Matrix{
          at::Tensor(), at::Tensor(), at::Tensor(), std::move(*combined)};
      weights.bundled_gate_up_qualified = true;
    }
    return;
  }
  const at::Tensor& gate_q = weights.gate.quantized;
  const at::Tensor& up_q = weights.up.quantized;
  const at::Tensor& gate_s = weights.gate.row_scales;
  const at::Tensor& up_s = weights.up.row_scales;
  if (!weights.packed_int8_qualified || !gate_q.defined() ||
      !up_q.defined() || !gate_s.defined() || !up_s.defined() ||
      gate_q.dim() != 2 || up_q.sizes() != gate_q.sizes() ||
      gate_s.dim() != 1 || up_s.sizes() != gate_s.sizes() ||
      !gate_q.is_alias_of(up_q) || !gate_s.is_alias_of(up_s) ||
      gate_q.storage_offset() + gate_q.numel() != up_q.storage_offset() ||
      gate_s.storage_offset() + gate_s.numel() != up_s.storage_offset()) {
    return;
  }
  const std::int64_t rows = gate_q.size(0);
  const std::int64_t columns = gate_q.size(1);
  at::Tensor q_bundle = gate_q.as_strided(
      {rows * 2, columns}, {columns, 1}, gate_q.storage_offset());
  at::Tensor s_bundle = gate_s.as_strided(
      {rows * 2}, {1}, gate_s.storage_offset());
  if (!q_bundle.is_contiguous() || !s_bundle.is_contiguous()) {
    return;
  }
  weights.gate_up = deltafin::provider_internal::MoeRowInt8Matrix{
      std::move(q_bundle), std::move(s_bundle), at::Tensor(), {}};
  weights.bundled_gate_up_qualified = true;
}

deltafin::provider_internal::MoeSpineT1 target_moe_from_spine(
    const SpineLayerSlot& layer,
    const SpineFp32ExecutionView* execution = nullptr) {
  constexpr std::int64_t hidden = 7168;
  constexpr std::int64_t routed_hidden = 3584;
  constexpr std::int64_t shared_intermediate = 6144;
  constexpr std::int64_t experts = 896;
  auto router = require_target_linear(
      layer, kMoeGateWeightSlot, experts, hidden, "MoE router", execution);
  auto routed_down = require_target_linear(
      layer, kMoeRoutedDownProjectionSlot, routed_hidden, hidden,
      "MoE routed-down projection", execution);
  auto routed_up = require_target_linear(
      layer, kMoeRoutedUpProjectionSlot, hidden, routed_hidden,
      "MoE routed-up projection", execution);
  auto shared_gate = require_target_linear(
      layer, kMoeSharedGateProjectionSlot, shared_intermediate, hidden,
      "MoE shared gate projection", execution);
  auto shared_up = require_target_linear(
      layer, kMoeSharedUpProjectionSlot, shared_intermediate, hidden,
      "MoE shared up projection", execution);
  auto shared_down = require_target_linear(
      layer, kMoeSharedDownProjectionSlot, hidden, shared_intermediate,
      "MoE shared down projection", execution);
  const bool packed = target_linear_is_packed(router);
  for (const auto* matrix : {&routed_down, &routed_up, &shared_gate,
                             &shared_up, &shared_down}) {
    if (target_linear_is_packed(*matrix) != packed) {
      throw std::invalid_argument(
          "target MoE layer mixes original-BF16 and row-int8 projections");
    }
  }
  return deltafin::provider_internal::MoeSpineT1{
      .layer_index = layer.layer_index,
      .generation = layer.generation,
      .geometry = deltafin::provider_internal::k3_moe_geometry(),
      .packed_int8_qualified = packed,
      .router = std::move(router),
      .router_correction_bias = require_target_raw(
          layer, kMoeGateCorrectionBiasSlot,
          DELTAFIN_PROVIDER_SPINE_RAW_F32_V1, {experts},
          "MoE correction bias"),
      .routed_down = std::move(routed_down),
      .routed_norm = require_target_raw(
          layer, kMoeRoutedNormSlot, DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1,
          {routed_hidden}, "MoE routed norm"),
      .routed_up = std::move(routed_up),
      .shared_gate = std::move(shared_gate),
      .shared_up = std::move(shared_up),
      .shared_down = std::move(shared_down),
      // The optional shared gate/up super-view is qualified separately after
      // construction. State the empty initializer explicitly: GCC rejects the
      // omission under -Werror=missing-field-initializers while Clang accepts
      // it, so leaving it implicit fails only the Linux provider build.
      .shared_gate_up = {}};
}

void require_target_packed_shape(
    Session& session,
    const deltafin::provider_internal::MoeRowInt8Matrix& matrix,
    const char* name) {
  if (!matrix.quantized.defined() || !matrix.row_scales.defined() ||
      matrix.quantized.dim() != 2 || matrix.row_scales.dim() != 1 ||
      matrix.quantized.scalar_type() != at::kChar ||
      matrix.row_scales.scalar_type() != at::kFloat ||
      !matrix.quantized.is_contiguous() ||
      !matrix.row_scales.is_contiguous() ||
      matrix.row_scales.size(0) != matrix.quantized.size(0) ||
      matrix.quantized.device() != session.selected.device ||
      matrix.row_scales.device() != session.selected.device) {
    throw std::invalid_argument(std::string("target packed-int8 ") + name +
                                " violates its device/storage contract");
  }
  const auto shape =
      std::pair(matrix.quantized.size(0), matrix.quantized.size(1));
  if (session.qualified_target_packed_shapes.contains(shape)) {
    return;
  }

  const auto probes = [](const std::int64_t extent) {
    std::array<std::int64_t, 9> values{
        0, 1, 2, 3, 31, 32, extent / 2, extent - 2, extent - 1};
    std::sort(values.begin(), values.end());
    const auto end = std::unique(values.begin(), values.end());
    return std::vector<std::int64_t>(values.begin(), end);
  };
  const std::vector<std::int64_t> probe_columns = probes(shape.second);
  const std::vector<std::int64_t> probe_rows = probes(shape.first);
  constexpr std::array<float, 9> activations{
      1.0F, -2.0F, 4.0F, -8.0F, 16.0F,
      -32.0F, 64.0F, -128.0F, 256.0F};
  at::Tensor hidden_cpu = at::zeros(
      {1, shape.second},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  auto hidden_values = hidden_cpu.accessor<float, 2>();
  for (std::size_t index = 0; index < probe_columns.size(); ++index) {
    hidden_values[0][probe_columns[index]] = activations[index];
  }
  const at::Tensor hidden = hidden_cpu.to(session.selected.device);
  const at::Tensor output = at::_weight_int8pack_mm(
      hidden, matrix.quantized, matrix.row_scales);
  if (!output.defined() || output.scalar_type() != at::kFloat ||
      output.device() != session.selected.device ||
      !output.is_contiguous() ||
      output.sizes() != at::IntArrayRef({1, shape.first})) {
    throw std::runtime_error(std::string("selected provider failed exact ") +
                             name + " packed-int8 production-shape gate");
  }
  for (std::size_t probe = 0; probe < probe_rows.size(); ++probe) {
    const std::int64_t row = probe_rows[probe];
    // Narrow is a metadata-only view on every provider. Copying nine selected
    // rows avoids depending on an accelerator int8 index-select kernel and,
    // critically, never stages the 1.17 GB head back to host.
    const at::Tensor q_row =
        matrix.quantized.narrow(0, row, 1).to(at::kCPU).contiguous();
    const at::Tensor scale =
        matrix.row_scales.narrow(0, row, 1).to(at::kCPU).contiguous();
    const at::Tensor observed =
        output.narrow(1, row, 1).to(at::kCPU).contiguous();
    auto q = q_row.accessor<std::int8_t, 2>();
    std::int64_t dot = 0;
    for (std::size_t column = 0; column < probe_columns.size(); ++column) {
      dot += static_cast<std::int64_t>(q[0][probe_columns[column]]) *
          static_cast<std::int64_t>(activations[column]);
    }
    const float expected =
        static_cast<float>(dot) * scale.const_data_ptr<float>()[0];
    const float got = observed.const_data_ptr<float>()[0];
    if (std::bit_cast<std::uint32_t>(got) !=
        std::bit_cast<std::uint32_t>(expected)) {
      throw std::runtime_error(
          std::string("selected provider failed exact ") + name +
          " packed-int8 arithmetic gate at output row " +
          std::to_string(probe_rows[probe]));
    }
  }
  session.qualified_target_packed_shapes.insert(shape);
}

void maybe_publish_compact_pilot_router(
    Session& session, const SpineLayerSlot& authoritative) noexcept {
  if (!session.target_pilot_enabled || authoritative.layer_index == 0 ||
      authoritative.layer_index >= kK3Layers ||
      session.target_pilot_routers[authoritative.layer_index].has_value() ||
      authoritative.target_residual == nullptr) {
    return;
  }
  try {
    // Reconstructing the authoritative descriptor first validates the entire
    // routed-MoE roster, not merely the three tensors PILOT needs. The clone is
    // built and provider-qualified off to the side, then moved into the fixed
    // session slot in one publication step. Every failure is advisory only.
    const auto moe = target_moe_from_spine(authoritative);
    auto candidate =
        deltafin::provider_internal::clone_compact_pilot_router_t1(
            moe, authoritative.target_residual->post_attention_norm, true);
    require_target_packed_shape(session, candidate.router, "PILOT router");
    session.target_pilot_routers[authoritative.layer_index].emplace(
        std::move(candidate));
  } catch (...) {
    // Optional lookahead can only save I/O. An unsupported packed operator,
    // allocation failure, or incomplete layer leaves this immutable slot
    // empty and the authoritative demand-read path proceeds unchanged.
  }
}

std::unique_ptr<TargetSessionCacheStore> make_target_cache_store(
    const at::Device& device) {
  auto store = std::make_unique<TargetSessionCacheStore>();
  for (std::uint32_t layer = 0; layer < kK3Layers; ++layer) {
    if (deltafin::provider_internal::target_layer_uses_mla(layer)) {
      store->mla[layer] =
          std::make_unique<deltafin::provider_internal::MlaCache>(
              deltafin::provider_internal::MlaShape::k3(),
              deltafin::provider_internal::MlaCacheRepresentation::
                  ExpandedExact);
      store->bindings[layer] =
          deltafin::provider_internal::TargetLayerCacheBinding{
              .layer_index = layer,
              .attention_kind =
                  deltafin::provider_internal::TargetAttentionKind::Mla,
              .kda_cache = nullptr,
              .mla_cache = store->mla[layer].get()};
    } else {
      store->kda[layer] =
          std::make_unique<deltafin::provider_internal::TargetKdaCache>(
              deltafin::provider_internal::TargetKdaCache{
                  .layer_index = layer,
                  .version = 0,
                  .state =
                      deltafin::provider_internal::zero_k3_kda_state(device)});
      store->bindings[layer] =
          deltafin::provider_internal::TargetLayerCacheBinding{
              .layer_index = layer,
              .attention_kind =
                  deltafin::provider_internal::TargetAttentionKind::Kda,
              .kda_cache = store->kda[layer].get(),
              .mla_cache = nullptr};
    }
  }
  return store;
}

void require_complete_target_cache_store(
    const TargetSessionCacheStore& store,
    const std::uint64_t committed_positions) {
  if (committed_positions > static_cast<std::uint64_t>(INT64_MAX)) {
    throw std::logic_error("target cache position count exceeds int64");
  }
  for (std::uint32_t layer = 0; layer < kK3Layers; ++layer) {
    const bool mla =
        deltafin::provider_internal::target_layer_uses_mla(layer);
    const auto& binding = store.bindings[layer];
    if (binding.layer_index != layer ||
        binding.attention_kind !=
            (mla ? deltafin::provider_internal::TargetAttentionKind::Mla
                 : deltafin::provider_internal::TargetAttentionKind::Kda)) {
      throw std::logic_error("target cache store binding roster is corrupt");
    }
    if (mla) {
      if (store.mla[layer] == nullptr || store.kda[layer] != nullptr ||
          binding.mla_cache != store.mla[layer].get() ||
          binding.kda_cache != nullptr ||
          store.mla[layer]->has_pending_prepare() ||
          store.mla[layer]->representation() !=
              deltafin::provider_internal::MlaCacheRepresentation::
                  ExpandedExact ||
          store.mla[layer]->length() !=
              static_cast<std::int64_t>(committed_positions) ||
          store.mla[layer]->version() != committed_positions) {
        throw std::logic_error("target MLA cache boundary is incomplete");
      }
    } else if (store.kda[layer] == nullptr || store.mla[layer] != nullptr ||
               binding.kda_cache != store.kda[layer].get() ||
               binding.mla_cache != nullptr ||
               store.kda[layer]->layer_index != layer ||
               store.kda[layer]->version != committed_positions) {
      throw std::logic_error("target KDA cache boundary is incomplete");
    }
  }
}

std::unique_ptr<TargetSessionCacheStore> fork_target_cache_store(
    const TargetSessionCacheStore& parent,
    const std::uint64_t committed_positions) {
  require_complete_target_cache_store(parent, committed_positions);
  auto child = std::make_unique<TargetSessionCacheStore>();
  for (std::uint32_t layer = 0; layer < kK3Layers; ++layer) {
    if (deltafin::provider_internal::target_layer_uses_mla(layer)) {
      child->mla[layer] = parent.mla[layer]->fork_committed();
      child->bindings[layer] =
          deltafin::provider_internal::TargetLayerCacheBinding{
              .layer_index = layer,
              .attention_kind =
                  deltafin::provider_internal::TargetAttentionKind::Mla,
              .kda_cache = nullptr,
              .mla_cache = child->mla[layer].get()};
    } else {
      child->kda[layer] =
          std::make_unique<deltafin::provider_internal::TargetKdaCache>(
              *parent.kda[layer]);
      child->bindings[layer] =
          deltafin::provider_internal::TargetLayerCacheBinding{
              .layer_index = layer,
              .attention_kind =
                  deltafin::provider_internal::TargetAttentionKind::Kda,
              .kda_cache = child->kda[layer].get(),
              .mla_cache = nullptr};
    }
  }
  require_complete_target_cache_store(*child, committed_positions);
  return child;
}

DeltafinProviderTargetStateReportV1 target_state_report(
    const Session& session) {
  DeltafinProviderTargetStateReportV1 report = {};
  report.struct_size = sizeof(report);
  report.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  report.committed_positions = session.committed_target_positions;
  report.cache_generation = session.committed_target_generation;
  report.active_branch = session.target_state_branch == nullptr
      ? 0
      : session.target_state_branch->handle;
  return report;
}

std::uint32_t target_state_value(
    const deltafin::provider_internal::TargetPositionState state) {
  switch (state) {
    case deltafin::provider_internal::TargetPositionState::Active:
      return DELTAFIN_PROVIDER_TARGET_ACTIVE_V1;
    case deltafin::provider_internal::TargetPositionState::WaitingForExperts:
      return DELTAFIN_PROVIDER_TARGET_WAITING_FOR_EXPERTS_V1;
    case deltafin::provider_internal::TargetPositionState::ReadyForTail:
      return DELTAFIN_PROVIDER_TARGET_READY_FOR_TAIL_V1;
    case deltafin::provider_internal::TargetPositionState::Committed:
      return DELTAFIN_PROVIDER_TARGET_COMMITTED_V1;
    case deltafin::provider_internal::TargetPositionState::Cancelled:
      return DELTAFIN_PROVIDER_TARGET_CANCELLED_V1;
    case deltafin::provider_internal::TargetPositionState::Poisoned:
      return DELTAFIN_PROVIDER_TARGET_POISONED_V1;
  }
  throw std::logic_error("target position returned an unknown state");
}

std::uint32_t target_sequence_state_value(
    const deltafin::provider_internal::TargetSequenceState state) {
  using deltafin::provider_internal::TargetSequenceState;
  switch (state) {
    case TargetSequenceState::Active:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_ACTIVE_V1;
    case TargetSequenceState::WaitingForExperts:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_WAITING_FOR_EXPERTS_V1;
    case TargetSequenceState::ReadyForTail:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_READY_FOR_TAIL_V1;
    case TargetSequenceState::ReadyToCommit:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_READY_TO_COMMIT_V1;
    case TargetSequenceState::Committed:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_COMMITTED_V1;
    case TargetSequenceState::Cancelled:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_CANCELLED_V1;
    case TargetSequenceState::Poisoned:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_POISONED_V1;
  }
  throw std::logic_error("target sequence returned an unknown state");
}

std::uint32_t target_sequence_mode_value(
    const deltafin::provider_internal::TargetSequenceMode mode) {
  using deltafin::provider_internal::TargetSequenceMode;
  switch (mode) {
    case TargetSequenceMode::Prefill:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_PREFILL_V1;
    case TargetSequenceMode::Verify:
      return DELTAFIN_PROVIDER_TARGET_SEQUENCE_VERIFY_V1;
  }
  throw std::logic_error("target sequence returned an unknown mode");
}

bool projection_has_shape(const SpineLayerSlot& layer,
                          const std::uint32_t slot,
                          const std::int64_t rows,
                          const std::int64_t columns) {
  if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
    return false;
  }
  const SpineTensorSlot& tensor = *layer.tensors[slot];
  if (tensor.encoding ==
      DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1) {
    return tensor.data.defined() && tensor.auxiliary.defined() &&
        tensor.data.sizes() == at::IntArrayRef({rows, columns}) &&
        tensor.auxiliary.sizes() == at::IntArrayRef({rows});
  }
  return tensor.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 &&
      tensor.original_bf16.defined() &&
      tensor.original_bf16.rows == static_cast<std::size_t>(rows) &&
      tensor.original_bf16.columns == static_cast<std::size_t>(columns);
}

bool descriptor_projection_has_shape(
    const std::vector<ValidatedSpineDescriptor>& descriptors,
    const std::uint32_t slot, const std::int64_t rows,
    const std::int64_t columns) {
  const auto found = std::find_if(
      descriptors.begin(), descriptors.end(),
      [&](const ValidatedSpineDescriptor& descriptor) {
        return descriptor.raw.slot == slot;
      });
  return found != descriptors.end() &&
         found->raw.encoding ==
             DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1 &&
         found->shape == std::vector<std::int64_t>({rows, columns});
}

bool should_gather_mla_input_weights(
    const std::vector<ValidatedSpineDescriptor>& descriptors,
    const std::uint32_t layer_index,
    const deltafin::provider_internal::MlaShape& shape,
    const std::uint32_t selected_device) {
  if (selected_device != DELTAFIN_PROVIDER_DEVICE_MPS_V1 ||
      !is_mla_layer(layer_index)) {
    return false;
  }
  const std::int64_t key_value_rows =
      shape.kv_lora_rank + shape.qk_rope_head_dim;
  const std::int64_t gate_rows = shape.num_heads * shape.value_head_dim;
  return descriptor_projection_has_shape(
             descriptors, kMlaQueryAProjectionSlot, shape.q_lora_rank,
             shape.hidden_size) &&
         descriptor_projection_has_shape(
             descriptors, kMlaKeyValueAProjectionSlot, key_value_rows,
             shape.hidden_size) &&
         descriptor_projection_has_shape(
             descriptors, kMlaOutputGateProjectionSlot, gate_rows,
             shape.hidden_size);
}

at::Tensor adjacent_spine_projection_view(
    const std::array<at::Tensor, 3>& parts,
    const std::array<std::int64_t, 3>& rows,
    const std::int64_t columns, const char* name) {
  if (columns <= 0) {
    throw std::logic_error(std::string("MLA ") + name +
                           " bundle has invalid columns");
  }
  std::int64_t total_rows = 0;
  std::int64_t expected_offset = 0;
  for (std::size_t index = 0; index < parts.size(); ++index) {
    const at::Tensor& part = parts[index];
    if (!part.defined() || !part.is_contiguous() || part.dim() != 2 ||
        part.size(0) != rows[index] || part.size(1) != columns ||
        (index != 0 && !parts.front().is_alias_of(part))) {
      throw std::logic_error(std::string("MLA ") + name +
                             " bundle is not a compatible storage view");
    }
    if (index == 0) {
      expected_offset = part.storage_offset();
    }
    if (part.storage_offset() != expected_offset ||
        part.numel() > std::numeric_limits<std::int64_t>::max() -
                           expected_offset ||
        rows[index] > std::numeric_limits<std::int64_t>::max() - total_rows) {
      throw std::logic_error(std::string("MLA ") + name +
                             " bundle views are not adjacent");
    }
    expected_offset += part.numel();
    total_rows += rows[index];
  }
  return parts.front().as_strided({total_rows, columns}, {columns, 1},
                                  parts.front().storage_offset());
}

at::Tensor adjacent_spine_scale_view(
    const std::array<at::Tensor, 3>& parts,
    const std::array<std::int64_t, 3>& rows) {
  std::int64_t total_rows = 0;
  std::int64_t expected_offset = 0;
  for (std::size_t index = 0; index < parts.size(); ++index) {
    const at::Tensor& part = parts[index];
    if (!part.defined() || !part.is_contiguous() || part.dim() != 1 ||
        part.size(0) != rows[index] ||
        (index != 0 && !parts.front().is_alias_of(part))) {
      throw std::logic_error(
          "MLA scale bundle is not a compatible storage view");
    }
    if (index == 0) {
      expected_offset = part.storage_offset();
    }
    if (part.storage_offset() != expected_offset ||
        part.numel() > std::numeric_limits<std::int64_t>::max() -
                           expected_offset ||
        rows[index] > std::numeric_limits<std::int64_t>::max() - total_rows) {
      throw std::logic_error("MLA scale bundle views are not adjacent");
    }
    expected_offset += part.numel();
    total_rows += rows[index];
  }
  return parts.front().as_strided({total_rows}, {1},
                                  parts.front().storage_offset());
}

void maybe_bundle_mla_input_weights(
    SpineLayerSlot& layer, const deltafin::provider_internal::MlaShape& shape,
    const std::uint32_t selected_device) {
  // MPS retains its qualified packed-int8 bundle. Every selected device may
  // additionally form a zero-copy original-BF16 super-view over one shared
  // grouped upload run.
  if ((selected_device != DELTAFIN_PROVIDER_DEVICE_MPS_V1 &&
       selected_device != DELTAFIN_PROVIDER_DEVICE_CPU_V1 &&
       selected_device != DELTAFIN_PROVIDER_DEVICE_CUDA_V1) ||
      !is_mla_layer(layer.layer_index)) {
    return;
  }
  const std::int64_t key_value_rows =
      shape.kv_lora_rank + shape.qk_rope_head_dim;
  const std::int64_t gate_rows = shape.num_heads * shape.value_head_dim;
  if (!projection_has_shape(layer, kMlaQueryAProjectionSlot,
                            shape.q_lora_rank, shape.hidden_size) ||
      !projection_has_shape(layer, kMlaKeyValueAProjectionSlot,
                            key_value_rows, shape.hidden_size) ||
      !projection_has_shape(layer, kMlaOutputGateProjectionSlot,
                            gate_rows, shape.hidden_size)) {
    return;
  }
  deltafin::provider_internal::MlaWeights weights;
  weights.query_a = require_mla_projection(
      layer, kMlaQueryAProjectionSlot, "query-a projection");
  weights.key_value_a = require_mla_projection(
      layer, kMlaKeyValueAProjectionSlot, "key/value-a projection");
  weights.output_gate = require_mla_projection(
      layer, kMlaOutputGateProjectionSlot, "output-gate projection");
  const auto encoding = weights.query_a.encoding;
  if (weights.key_value_a.encoding != encoding ||
      weights.output_gate.encoding != encoding ||
      (encoding !=
           deltafin::provider_internal::MlaLinearEncoding::RowI8F32Scale &&
       encoding !=
           deltafin::provider_internal::MlaLinearEncoding::OriginalBf16)) {
    return;
  }
  if ((encoding ==
           deltafin::provider_internal::MlaLinearEncoding::RowI8F32Scale &&
       selected_device != DELTAFIN_PROVIDER_DEVICE_MPS_V1)) {
    return;
  }
  const std::array<std::int64_t, 3> rows{
      shape.q_lora_rank, key_value_rows, gate_rows};
  auto bundle =
      std::make_unique<deltafin::provider_internal::MlaInputBundle>();
  bundle->query_a_rows = rows[0];
  bundle->key_value_a_rows = rows[1];
  bundle->output_gate_rows = rows[2];
  if (encoding ==
      deltafin::provider_internal::MlaLinearEncoding::OriginalBf16) {
    const std::array<
        const deltafin::provider_internal::OriginalBf16Matrix*, 3>
        matrices{&weights.query_a.original_bf16,
                 &weights.key_value_a.original_bf16,
                 &weights.output_gate.original_bf16};
    auto combined =
        deltafin::provider_internal::adjacent_original_bf16_matrices(
            matrices);
    if (!combined.has_value()) {
      return;
    }
    bundle->projection.encoding =
        deltafin::provider_internal::MlaLinearEncoding::OriginalBf16;
    bundle->projection.original_bf16 = std::move(*combined);
    layer.mla_input_bundle = std::move(bundle);
    layer.binding_stats.mla_input_bundle_count = 1;
    return;
  }
  bundle->projection.encoding =
      deltafin::provider_internal::MlaLinearEncoding::RowI8F32Scale;
  bundle->projection.data = adjacent_spine_projection_view(
      {weights.query_a.data, weights.key_value_a.data,
       weights.output_gate.data},
      rows, shape.hidden_size, "data");
  bundle->projection.row_scale = adjacent_spine_scale_view(
      {weights.query_a.row_scale, weights.key_value_a.row_scale,
       weights.output_gate.row_scale},
      rows);
  std::int64_t row = 0;
  const auto bundle_view = [&](const std::int64_t count) {
    deltafin::provider_internal::MlaLinearWeight result{
        .encoding =
            deltafin::provider_internal::MlaLinearEncoding::RowI8F32Scale,
        .data = bundle->projection.data.narrow(0, row, count),
        .row_scale = bundle->projection.row_scale.narrow(0, row, count),
        // This view carries no original bf16 matrix. State it explicitly:
        // GCC rejects the omission under -Werror=missing-field-initializers.
        .original_bf16 = {},
    };
    row += count;
    return result;
  };
  weights.query_a = bundle_view(rows[0]);
  weights.key_value_a = bundle_view(rows[1]);
  weights.output_gate = bundle_view(rows[2]);
  auto install_view = [&](const std::uint32_t slot,
                          const deltafin::provider_internal::MlaLinearWeight&
                              projection) {
    if (slot >= layer.tensors.size() || !layer.tensors[slot].has_value()) {
      throw std::logic_error("MLA bundle target slot disappeared");
    }
    SpineTensorSlot& target = *layer.tensors[slot];
    target.data = projection.data;
    target.auxiliary = projection.row_scale;
  };
  install_view(kMlaQueryAProjectionSlot, weights.query_a);
  install_view(kMlaKeyValueAProjectionSlot, weights.key_value_a);
  install_view(kMlaOutputGateProjectionSlot, weights.output_gate);
  layer.mla_input_bundle = std::move(bundle);
  layer.binding_stats.mla_input_bundle_count = 1;
}

std::uint64_t resident_spine_storage_bytes(const SpineLayerSlot& layer) {
  std::array<const c10::StorageImpl*,
             (kLastGlobalWeightSlot + 1) * 2 + 2>
      unique = {};
  std::size_t unique_count = 0;
  std::uint64_t total = 0;
  const auto account = [&](const at::Tensor& tensor) {
    if (!tensor.defined()) {
      return;
    }
    const auto* implementation = tensor.unsafeGetTensorImpl();
    if (implementation == nullptr || !implementation->has_storage()) {
      throw std::logic_error("bound spine tensor has no resident storage");
    }
    const c10::Storage& storage = implementation->storage();
    const auto* storage_impl = storage.unsafeGetStorageImpl();
    if (std::find(unique.begin(), unique.begin() + unique_count,
                  storage_impl) != unique.begin() + unique_count) {
      return;
    }
    if (unique_count == unique.size()) {
      throw std::logic_error("bound spine has too many storage allocations");
    }
    unique[unique_count++] = storage_impl;
    total = checked_spine_sum(
        total, static_cast<std::uint64_t>(storage.nbytes()),
        "resident spine storage bytes");
  };
  for (const auto& tensor : layer.tensors) {
    if (!tensor.has_value()) {
      continue;
    }
    account(tensor->data);
    account(tensor->auxiliary);
    if (tensor->original_bf16.is_owned()) {
      if (tensor->original_bf16.owned_storage == nullptr) {
        throw std::logic_error(
            "owned original-BF16 spine carrier lost its shared storage");
      }
      account(tensor->original_bf16.owned_storage->tensor);
    }
  }
  if (layer.mla_input_bundle != nullptr) {
    account(layer.mla_input_bundle->projection.data);
    account(layer.mla_input_bundle->projection.row_scale);
    if (layer.mla_input_bundle->projection.original_bf16.is_owned()) {
      account(layer.mla_input_bundle->projection.original_bf16
                  .owned_storage->tensor);
    }
  }
  if (layer.target_residual != nullptr) {
    account(layer.target_residual->input_norm);
    account(layer.target_residual->self_attention_res_norm);
    account(layer.target_residual->self_attention_res_projection);
    account(layer.target_residual->post_attention_norm);
    account(layer.target_residual->mlp_res_norm);
    account(layer.target_residual->mlp_res_projection);
    account(layer.target_residual->self_attention_score_weight);
    account(layer.target_residual->mlp_score_weight);
  }
  return total;
}

/*
 * Mirror the established MPS resident-spine staging contract: transfer the
 * complete q8, fp16-scale, and mixed raw slabs once each, then let compiled
 * device operations form the provider's exact published views.  The generic
 * uploader below groups by source/target dtype.  That is useful for portable
 * detached storage, but on MPS each group is a blocking host-to-device
 * copy.  Real K3 therefore grew from three copies per layer to five on KDA
 * and eleven on MLA (the latter also gathered three projections on host).
 *
 * This fast path is deliberately narrow.  It accepts only the loose int8
 * representation whose q8 payload completely covers the quantized slab and
 * whose large matrices are all row-int8.  Packed/mixed layouts, original-BF16
 * matrices, CPU, CUDA, and every unusual descriptor retain the established
 * uploader byte-for-byte.  The temporary fp16/raw device slabs are consumed
 * by same-stream ATen copies before their handles are dropped; only the exact
 * logical q8/fp32 component storage is published, so residency accounting and
 * the bind ABI remain unchanged.
 */
bool try_upload_mps_three_slab_payload(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const std::vector<ValidatedSpineDescriptor>& descriptors,
    const at::Device& device, const bool gather_mla_inputs,
    SpineLayerSlot& staged) {
  // Parsed once because this switch exists only for exact process-level A/B
  // validation. Unknown values fail closed to the established uploader.
  static const bool enabled = [] {
    const char* value = std::getenv("K3_MPS_SPINE_SLAB_UPLOAD");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "1") == 0) {
      return true;
    }
    return false;
  }();
  if (!enabled || !device.is_mps() || request.quantized == nullptr ||
      request.scales == nullptr || request.other == nullptr ||
      request.quantized_length == 0 || request.scales_length == 0 ||
      request.other_length == 0 || request.scales_length % sizeof(c10::Half) != 0) {
    return false;
  }

  std::uint64_t quantized_logical_bytes = 0;
  std::uint64_t scale_source_bytes = 0;
  std::uint64_t scale_target_elements = 0;
  std::uint64_t raw_source_bytes = 0;
  std::uint64_t raw_target_elements = 0;
  deltafin::provider_internal::SpineBindingDebugStats stats;
  for (const auto& descriptor : descriptors) {
    const auto& raw = descriptor.raw;
    switch (raw.encoding) {
      case DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1:
        if (raw.data_buffer != DELTAFIN_PROVIDER_SPINE_BUFFER_QUANTIZED_V1 ||
            raw.auxiliary_buffer != DELTAFIN_PROVIDER_SPINE_BUFFER_SCALES_V1 ||
            raw.data_length == 0 || raw.auxiliary_length == 0 ||
            raw.auxiliary_length % sizeof(c10::Half) != 0) {
          return false;
        }
        quantized_logical_bytes = checked_spine_sum(
            quantized_logical_bytes, raw.data_length,
            "MPS slab quantized logical bytes");
        scale_source_bytes = checked_spine_sum(
            scale_source_bytes, raw.auxiliary_length,
            "MPS slab scale source bytes");
        scale_target_elements = checked_spine_sum(
            scale_target_elements, raw.auxiliary_length / sizeof(c10::Half),
            "MPS slab scale target elements");
        stats.source_component_count = checked_spine_sum(
            stats.source_component_count, 2, "MPS slab component count");
        stats.source_component_bytes = checked_spine_sum(
            stats.source_component_bytes,
            checked_spine_sum(raw.data_length, raw.auxiliary_length,
                              "MPS slab quantized component bytes"),
            "MPS slab source bytes");
        stats.logical_target_bytes = checked_spine_sum(
            stats.logical_target_bytes,
            checked_spine_sum(
                raw.data_length,
                (raw.auxiliary_length / sizeof(c10::Half)) * sizeof(float),
                "MPS slab quantized target bytes"),
            "MPS slab target bytes");
        break;
      case DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1:
        // Original-BF16 matrices have a separate exact carrier and must never
        // be silently promoted by this int8-only staging optimization.
        if (raw.data_buffer != DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1 ||
            raw.auxiliary_buffer != DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 ||
            raw.data_length == 0 || raw.data_length % sizeof(c10::BFloat16) != 0 ||
            (descriptor.shape.size() == 2 && descriptor.shape[0] > 1)) {
          return false;
        }
        raw_source_bytes = checked_spine_sum(
            raw_source_bytes, raw.data_length, "MPS slab raw source bytes");
        raw_target_elements = checked_spine_sum(
            raw_target_elements, raw.data_length / sizeof(c10::BFloat16),
            "MPS slab raw target elements");
        stats.source_component_count = checked_spine_sum(
            stats.source_component_count, 1, "MPS slab component count");
        stats.source_component_bytes = checked_spine_sum(
            stats.source_component_bytes, raw.data_length,
            "MPS slab source bytes");
        stats.logical_target_bytes = checked_spine_sum(
            stats.logical_target_bytes,
            (raw.data_length / sizeof(c10::BFloat16)) * sizeof(float),
            "MPS slab target bytes");
        break;
      case DELTAFIN_PROVIDER_SPINE_RAW_F32_V1:
        if (raw.data_buffer != DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1 ||
            raw.auxiliary_buffer != DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 ||
            raw.data_length == 0 || raw.data_length % sizeof(float) != 0) {
          return false;
        }
        raw_source_bytes = checked_spine_sum(
            raw_source_bytes, raw.data_length, "MPS slab raw source bytes");
        raw_target_elements = checked_spine_sum(
            raw_target_elements, raw.data_length / sizeof(float),
            "MPS slab raw target elements");
        stats.source_component_count = checked_spine_sum(
            stats.source_component_count, 1, "MPS slab component count");
        stats.source_component_bytes = checked_spine_sum(
            stats.source_component_bytes, raw.data_length,
            "MPS slab source bytes");
        stats.logical_target_bytes = checked_spine_sum(
            stats.logical_target_bytes, raw.data_length,
            "MPS slab target bytes");
        break;
      default:
        return false;
    }
  }

  // Quantized production matrices are naturally 256-byte multiples.  Requiring
  // exact coverage avoids retaining padding while preserving the report's
  // exact logical-residency contract. Scale/raw alignment gaps live only in
  // temporary source slabs and are compacted by device copies below.
  if (quantized_logical_bytes != request.quantized_length ||
      scale_source_bytes == 0 || raw_source_bytes == 0 ||
      scale_target_elements == 0 || raw_target_elements == 0 ||
      request.quantized_length >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      request.scales_length / sizeof(c10::Half) >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      request.other_length >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      scale_target_elements >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      raw_target_elements >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return false;
  }

  const auto upload_complete_slab = [&](const std::uint32_t buffer,
                                        const std::uint64_t byte_length,
                                        const at::ScalarType type) {
    const std::uint64_t width = spine_scalar_width(type);
    if (byte_length == 0 || byte_length % width != 0 ||
        byte_length / width > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::int64_t>::max())) {
      throw std::logic_error("MPS slab upload has invalid scalar bounds");
    }
    const std::uint64_t elements = byte_length / width;
    const auto* source = spine_component_pointer(request, buffer, 0);
    const at::Tensor source_cpu = at::from_blob(
        const_cast<std::uint8_t*>(source),
        {static_cast<std::int64_t>(elements)},
        at::TensorOptions().dtype(type).device(at::kCPU));
    return upload_spine_cpu(source_cpu, elements, type, device);
  };

  // These are the only three host-to-MPS copies in the qualified path.
  at::Tensor quantized_device = upload_complete_slab(
      DELTAFIN_PROVIDER_SPINE_BUFFER_QUANTIZED_V1,
      request.quantized_length, at::kChar);
  at::Tensor scales_half_device = upload_complete_slab(
      DELTAFIN_PROVIDER_SPINE_BUFFER_SCALES_V1,
      request.scales_length, at::kHalf);
  at::Tensor other_bytes_device = upload_complete_slab(
      DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1,
      request.other_length, at::kChar);

  // Convert the complete scale slab in one device operation. Loose-file scale
  // components are 256-byte aligned, so real KDA/MLA layers contain small
  // holes. When holes exist (or MLA needs its q/kv/gate roster adjacent), one
  // device-side cat below compacts every logical scale exactly once; no
  // padding or duplicate scale storage survives publication.
  at::Tensor scales_full_f32 = scales_half_device.to(at::kFloat);
  at::Tensor raw_f32 = at::empty(
      {static_cast<std::int64_t>(raw_target_elements)},
      at::TensorOptions().dtype(at::kFloat).device(device));

  std::vector<std::size_t> quantized_order;
  quantized_order.reserve(descriptors.size());
  if (gather_mla_inputs) {
    for (std::uint32_t bundle_order = 0; bundle_order < 3; ++bundle_order) {
      const auto found = std::find_if(
          descriptors.begin(), descriptors.end(),
          [&](const ValidatedSpineDescriptor& descriptor) {
            return descriptor.raw.encoding ==
                       DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1 &&
                   is_mla_input_bundle_slot(descriptor.raw.slot) &&
                   mla_input_bundle_order(descriptor.raw.slot) == bundle_order;
          });
      if (found == descriptors.end()) {
        return false;
      }
      quantized_order.push_back(static_cast<std::size_t>(
          std::distance(descriptors.begin(), found)));
    }
  }
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    const auto& raw = descriptors[index].raw;
    if (raw.encoding != DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1 ||
        (gather_mla_inputs && is_mla_input_bundle_slot(raw.slot))) {
      continue;
    }
    quantized_order.push_back(index);
  }
  if (quantized_order.empty()) {
    return false;
  }

  const bool compact_quantized = gather_mla_inputs;
  const bool compact_scales =
      gather_mla_inputs || scale_source_bytes != request.scales_length;
  std::vector<at::Tensor> quantized_parts;
  std::vector<at::Tensor> scale_parts;
  if (compact_quantized) {
    quantized_parts.reserve(quantized_order.size());
  }
  if (compact_scales) {
    scale_parts.reserve(quantized_order.size());
  }
  std::array<std::uint64_t, kLastGlobalWeightSlot + 1>
      quantized_offsets = {};
  std::array<std::uint64_t, kLastGlobalWeightSlot + 1> scale_offsets = {};
  std::uint64_t quantized_cursor = 0;
  std::uint64_t scale_cursor = 0;
  for (const std::size_t descriptor_index : quantized_order) {
    const auto& raw = descriptors[descriptor_index].raw;
    if (raw.slot >= quantized_offsets.size()) {
      throw std::logic_error("MPS slab descriptor escaped its slot table");
    }
    const std::int64_t data_offset =
        static_cast<std::int64_t>(raw.data_offset);
    const std::int64_t data_elements =
        static_cast<std::int64_t>(raw.data_length);
    const std::int64_t scale_offset = static_cast<std::int64_t>(
        raw.auxiliary_offset / sizeof(c10::Half));
    const std::int64_t scale_elements = static_cast<std::int64_t>(
        raw.auxiliary_length / sizeof(c10::Half));
    if (compact_quantized) {
      quantized_offsets[raw.slot] = quantized_cursor;
      quantized_parts.push_back(
          quantized_device.narrow(0, data_offset, data_elements));
    } else {
      quantized_offsets[raw.slot] = raw.data_offset;
    }
    if (compact_scales) {
      scale_offsets[raw.slot] = scale_cursor;
      scale_parts.push_back(
          scales_full_f32.narrow(0, scale_offset, scale_elements));
    } else {
      scale_offsets[raw.slot] = raw.auxiliary_offset / sizeof(c10::Half);
    }
    quantized_cursor = checked_spine_sum(
        quantized_cursor, raw.data_length,
        "MPS slab quantized compaction cursor");
    scale_cursor = checked_spine_sum(
        scale_cursor,
        raw.auxiliary_length / sizeof(c10::Half),
        "MPS slab scale compaction cursor");
  }
  if (quantized_cursor != quantized_logical_bytes ||
      scale_cursor != scale_target_elements) {
    throw std::logic_error("MPS slab compaction roster is incomplete");
  }
  at::Tensor quantized_published = compact_quantized
      ? at::cat(quantized_parts, 0)
      : quantized_device;
  at::Tensor scales_published = compact_scales
      ? at::cat(scale_parts, 0)
      : scales_full_f32;
  if (!quantized_published.is_contiguous() ||
      quantized_published.scalar_type() != at::kChar ||
      static_cast<std::uint64_t>(quantized_published.numel()) !=
          quantized_logical_bytes ||
      !scales_published.is_contiguous() ||
      scales_published.scalar_type() != at::kFloat ||
      static_cast<std::uint64_t>(scales_published.numel()) !=
          scale_target_elements) {
    throw std::runtime_error(
        "MPS slab device compaction returned invalid storage");
  }

  std::array<std::optional<at::Tensor>, kLastGlobalWeightSlot + 1>
      data_views;
  std::array<std::optional<at::Tensor>, kLastGlobalWeightSlot + 1>
      auxiliary_views;
  std::uint64_t raw_cursor = 0;
  for (const auto& descriptor : descriptors) {
    const auto& raw = descriptor.raw;
    if (raw.slot >= data_views.size()) {
      throw std::logic_error("MPS slab descriptor escaped its slot table");
    }
    if (raw.encoding == DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1) {
      const std::int64_t data_elements = static_cast<std::int64_t>(
          raw.data_length);
      const std::int64_t scale_elements = static_cast<std::int64_t>(
          raw.auxiliary_length / sizeof(c10::Half));
      at::Tensor data = quantized_published
          .narrow(0, static_cast<std::int64_t>(quantized_offsets[raw.slot]),
                  data_elements)
          .view(descriptor.shape);
      at::Tensor scale = scales_published.narrow(
          0, static_cast<std::int64_t>(scale_offsets[raw.slot]),
          scale_elements);
      data_views[raw.slot].emplace(std::move(data));
      auxiliary_views[raw.slot].emplace(std::move(scale));
      continue;
    }

    const at::ScalarType source_type =
        raw.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1
        ? at::kBFloat16
        : at::kFloat;
    const std::uint64_t width = spine_scalar_width(source_type);
    const std::int64_t source_elements = static_cast<std::int64_t>(
        raw.data_length / width);
    const at::Tensor source = other_bytes_device
        .narrow(0, static_cast<std::int64_t>(raw.data_offset),
                static_cast<std::int64_t>(raw.data_length))
        .view(source_type)
        .view(descriptor.shape);
    at::Tensor destination = raw_f32
        .narrow(0, static_cast<std::int64_t>(raw_cursor), source_elements)
        .view(descriptor.shape);
    destination.copy_(source, false);
    data_views[raw.slot].emplace(std::move(destination));
    raw_cursor = checked_spine_sum(
        raw_cursor, static_cast<std::uint64_t>(source_elements),
        "MPS slab raw cursor");
  }
  if (raw_cursor != raw_target_elements) {
    throw std::logic_error("MPS slab compaction cursor is incomplete");
  }

  for (std::size_t slot = 0; slot < staged.tensors.size(); ++slot) {
    if (!staged.tensors[slot].has_value()) {
      continue;
    }
    if (!data_views[slot].has_value()) {
      throw std::logic_error("MPS slab did not publish every data view");
    }
    staged.tensors[slot]->data = std::move(*data_views[slot]);
    if (auxiliary_views[slot].has_value()) {
      staged.tensors[slot]->auxiliary =
          std::move(*auxiliary_views[slot]);
    }
  }
  stats.upload_run_count = 3;
  stats.direct_upload_run_count = 3;
  stats.gathered_upload_run_count = 0;
  staged.binding_stats = stats;
  return true;
}

std::unique_ptr<SpineLayerSlot> upload_validated_spine_payload(
    const DeltafinProviderBindSpineLayerRequestV1& request,
    const std::vector<ValidatedSpineDescriptor>& descriptors,
    Session& session, const std::uint32_t logical_layer_index,
    const bool gather_mla_inputs, const bool prepare_target_residual,
    const bool allow_borrowed_cpu = false) {
  const at::Device& device = session.selected.device;
  const std::uint32_t selected_device = session.selected.kind;
  auto staged = std::make_unique<SpineLayerSlot>();
  staged->layer_index = logical_layer_index;
  staged->generation = request.generation;
  std::vector<bool> borrowed_bf16_cpu(descriptors.size(), false);
  std::vector<bool> exact_bf16(descriptors.size(), false);
  for (std::size_t descriptor_index = 0;
       descriptor_index < descriptors.size(); ++descriptor_index) {
    const auto& descriptor = descriptors[descriptor_index];
    SpineTensorSlot tensor;
    tensor.encoding = descriptor.raw.encoding;
    tensor.shape = descriptor.shape;
    switch (descriptor.raw.encoding) {
      case DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1:
      case DELTAFIN_PROVIDER_SPINE_RAW_F32_V1:
        ++staged->raw_tensor_count;
        staged->other_bytes += descriptor.raw.data_length;
        break;
      case DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1:
        ++staged->quantized_tensor_count;
        staged->quantized_bytes += descriptor.raw.data_length;
        staged->scales_bytes += descriptor.raw.auxiliary_length;
        break;
      default:
        throw std::logic_error("validated spine encoding became invalid");
    }
    auto& destination = staged->tensors[descriptor.raw.slot];
    if (destination.has_value()) {
      throw std::logic_error("validated spine slot became duplicated");
    }
    const bool exact =
        descriptor.raw.encoding == DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 &&
        descriptor.shape.size() == 2 && descriptor.shape[0] > 1;
    exact_bf16[descriptor_index] = exact;
    const bool borrow = allow_borrowed_cpu && device.is_cpu() && exact;
    if (borrow) {
      const auto* bytes = spine_component_pointer(
          request, descriptor.raw.data_buffer, descriptor.raw.data_offset);
      tensor.original_bf16 =
          deltafin::provider_internal::make_borrowed_original_bf16_cpu(
              reinterpret_cast<const std::uint16_t*>(bytes),
              static_cast<std::size_t>(descriptor.shape[0]),
              static_cast<std::size_t>(descriptor.shape[1]),
              &session.bf16_cpu_t1_kernel());
      borrowed_bf16_cpu[descriptor_index] = true;
      ++staged->borrowed_tensor_count;
      staged->borrowed_source_bytes = checked_spine_sum(
          staged->borrowed_source_bytes, descriptor.raw.data_length,
          "borrowed BF16 source bytes");
    }
    destination.emplace(std::move(tensor));
    ++staged->tensor_count;
  }

  bool mps_three_slab = false;
  try {
    mps_three_slab = try_upload_mps_three_slab_payload(
        request, descriptors, device, gather_mla_inputs, *staged);
  } catch (const c10::Error&) {
    // A provider capability/allocation failure before publication retains the
    // exact established grouped uploader. Clear any views installed during a
    // late eager failure before planning that fallback.
    for (auto& tensor : staged->tensors) {
      if (tensor.has_value()) {
        tensor->data = at::Tensor();
        tensor->auxiliary = at::Tensor();
      }
    }
    staged->binding_stats = {};
  } catch (const std::bad_alloc&) {
    for (auto& tensor : staged->tensors) {
      if (tensor.has_value()) {
        tensor->data = at::Tensor();
        tensor->auxiliary = at::Tensor();
      }
    }
    staged->binding_stats = {};
  }
  std::vector<SpineComponentPlan> components;
  std::vector<std::size_t> ordered_components;
  const at::ScalarType exact_bf16_type =
      device.is_cuda() ? at::kShort : at::kUInt16;
  auto runs = mps_three_slab
      ? std::vector<SpineUploadRun>()
      : plan_spine_uploads(descriptors, gather_mla_inputs,
                           staged->binding_stats, components,
                           ordered_components,
                           allow_borrowed_cpu ? &borrowed_bf16_cpu
                                              : nullptr,
                           &exact_bf16, exact_bf16_type);
  for (const SpineUploadRun& run : runs) {
    at::Tensor uploaded = run.gathered
        ? upload_gathered_spine_run(request, run, components,
                                    ordered_components, device)
        : upload_direct_spine_run(request, run, device);
    std::shared_ptr<deltafin::provider_internal::ExactBf16Storage>
        exact_storage;
    if (run.target_type == exact_bf16_type) {
      if (run.source_type != exact_bf16_type || run.gathered) {
        throw std::logic_error(
            "exact BF16 upload run has an invalid transfer type");
      }
      exact_storage = device.is_cpu()
          ? deltafin::provider_internal::make_exact_bf16_storage(uploaded)
          : session.exact_bf16_device_projector().prepare(uploaded);
    }
    if (run.component_begin > ordered_components.size() ||
        run.component_count >
            ordered_components.size() - run.component_begin) {
      throw std::logic_error("spine upload run escaped its component tape");
    }
    for (std::size_t offset = 0; offset < run.component_count; ++offset) {
      const std::size_t component_index =
          ordered_components[run.component_begin + offset];
      const SpineComponentPlan& component = components[component_index];
      if (component.target_element_offset >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max()) ||
          component.elements > static_cast<std::uint64_t>(
                                   std::numeric_limits<std::int64_t>::max())) {
        throw std::logic_error(
            "spine component view exceeds int64 tensor bounds");
      }
      const auto& descriptor = descriptors[component.descriptor_index];
      const std::vector<std::int64_t> shape = component.auxiliary
          ? std::vector<std::int64_t>{descriptor.shape.front()}
          : descriptor.shape;
      auto& tensor_slot = staged->tensors[descriptor.raw.slot];
      if (!tensor_slot.has_value()) {
        throw std::logic_error("spine descriptor slot disappeared");
      }
      if (component.target_type == exact_bf16_type) {
        if (component.auxiliary || exact_storage == nullptr ||
            descriptor.raw.encoding !=
                DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 ||
            shape.size() != 2 || shape[0] <= 1 || shape[1] <= 0 ||
            tensor_slot->data.defined() ||
            tensor_slot->original_bf16.defined()) {
          throw std::logic_error(
              "exact BF16 component violated its matrix publication contract");
        }
        tensor_slot->original_bf16 =
            deltafin::provider_internal::make_owned_original_bf16(
                exact_storage,
                static_cast<std::size_t>(component.target_element_offset),
                static_cast<std::size_t>(shape[0]),
                static_cast<std::size_t>(shape[1]),
                device.is_cpu() ? &session.bf16_cpu_t1_kernel() : nullptr);
        continue;
      }
      const at::Tensor view = uploaded
          .narrow(0,
                  static_cast<std::int64_t>(
                      component.target_element_offset),
                  static_cast<std::int64_t>(component.elements))
          .view(shape);
      at::Tensor& destination = component.auxiliary
          ? tensor_slot->auxiliary
          : tensor_slot->data;
      if (destination.defined()) {
        throw std::logic_error("spine component was published more than once");
      }
      destination = view;
    }
  }
  for (const auto& tensor : staged->tensors) {
    if (!tensor.has_value()) {
      continue;
    }
    const bool expects_auxiliary = tensor->encoding ==
        DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1;
    const bool original = tensor->original_bf16.defined();
    if ((tensor->data.defined() == original) ||
        tensor->auxiliary.defined() != expects_auxiliary) {
      throw std::logic_error(
          "spine grouped upload did not publish every component exactly once");
    }
  }

  // The helper self-gates: MPS may use its gathered q8 roster, while every
  // selected backend may form one exact original-BF16 super-view when the
  // ordinary grouped upload made all three source projections adjacent.
  const auto mla_shape = (session.flags & kSyntheticMlaFlag) != 0
      ? deltafin::provider_internal::MlaShape::small_canary()
      : deltafin::provider_internal::MlaShape::k3();
  maybe_bundle_mla_input_weights(
      *staged, mla_shape, selected_device);
  if (prepare_target_residual) {
    maybe_prepare_target_residual(*staged);
  }
  staged->binding_stats.resident_storage_bytes =
      resident_spine_storage_bytes(*staged);
  const std::uint64_t residual_score_bytes = staged->target_residual == nullptr
      ? 0
      : static_cast<std::uint64_t>(2 * 7168 * sizeof(float));
  if (staged->binding_stats.resident_storage_bytes != checked_spine_sum(
          staged->binding_stats.logical_target_bytes, residual_score_bytes,
          "spine prepared-score storage bytes")) {
    throw std::logic_error(
        "spine grouped upload retained duplicate or padding storage");
  }
  return staged;
}

void require_target_global_roster(
    const std::uint32_t group,
    const std::vector<ValidatedSpineDescriptor>& descriptors) {
  const auto require_descriptor = [&](const std::uint32_t slot,
                                      const std::uint32_t encoding,
                                      const at::IntArrayRef shape,
                                      const char* name) {
    const auto found = std::find_if(
        descriptors.begin(), descriptors.end(),
        [slot](const ValidatedSpineDescriptor& descriptor) {
          return descriptor.raw.slot == slot;
        });
    if (found == descriptors.end() || found->raw.encoding != encoding ||
        found->shape != shape.vec()) {
      throw std::invalid_argument(std::string("target global group has invalid ") +
                                  name + " descriptor");
    }
  };
  if (group == kTargetGlobalTailGroup) {
    if (descriptors.size() != 3) {
      throw std::invalid_argument(
          "target tail global group requires exactly slots 41..43");
    }
    require_descriptor(kFinalNormSlot,
                       DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {7168},
                       "final norm");
    require_descriptor(kOutputResidualNormSlot,
                       DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {7168},
                       "output-residual norm");
    require_descriptor(kOutputResidualProjectionSlot,
                       DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {1, 7168},
                       "output-residual projection");
    return;
  }
  if (group == kTargetGlobalHeadGroup) {
    if (descriptors.size() != 1) {
      throw std::invalid_argument(
          "target head global group requires exactly slot 44");
    }
    const auto found = std::find_if(
        descriptors.begin(), descriptors.end(),
        [](const ValidatedSpineDescriptor& descriptor) {
          return descriptor.raw.slot == kLanguageModelHeadSlot;
        });
    if (found == descriptors.end() || found->shape !=
            at::IntArrayRef({163840, 7168}).vec() ||
        (found->raw.encoding != DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 &&
         found->raw.encoding !=
             DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1)) {
      throw std::invalid_argument(
          "target global group has invalid language-model head descriptor");
    }
    return;
  }
  throw std::invalid_argument("target global group is unknown");
}

std::unique_ptr<deltafin::provider_internal::TargetTailWeights>
make_target_tail(Session& session, const SpineLayerSlot& tail_group,
                 const SpineLayerSlot& head_group) {
  auto head = require_target_linear(
      head_group, kLanguageModelHeadSlot, 163840, 7168,
      "language-model head");
  const bool packed = target_linear_is_packed(head);
  if (packed) {
    require_target_packed_shape(session, head, "language-model head");
  }
  auto weights = deltafin::provider_internal::TargetTailWeights{
      .output_res_norm = require_target_raw(
          tail_group, kOutputResidualNormSlot,
          DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {7168},
          "output-residual norm"),
      .output_res_projection = require_target_raw(
          tail_group, kOutputResidualProjectionSlot,
          DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {1, 7168},
          "output-residual projection"),
      .final_norm = require_target_raw(
          tail_group, kFinalNormSlot,
          DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1, {7168}, "final norm"),
      .language_model_head = std::move(head),
      .packed_int8_qualified = packed};
  return std::make_unique<deltafin::provider_internal::TargetTailWeights>(
      deltafin::provider_internal::precompute_target_tail_score_weight(
          std::move(weights)));
}

const SpineLayerSlot& require_bound_spine_layer(
    const Session& session, const std::uint32_t layer_index,
    const std::uint64_t generation) {
  if (layer_index >= kK3Layers) {
    throw std::invalid_argument("provider spine layer is outside K3");
  }
  const auto& resident = session.resident_spine_layers[layer_index];
  if (resident != nullptr && resident->generation == generation) {
    return *resident;
  }
  if (session.transient_spine_layer != nullptr &&
      session.transient_spine_layer->layer_index == layer_index &&
      session.transient_spine_layer->generation == generation) {
    return *session.transient_spine_layer;
  }
  throw std::invalid_argument(
      "provider spine layer/generation is stale or not bound");
}

void require_memory_trim_boundary(const Session& session) {
  if (session.target_position != nullptr ||
      session.target_position_handle != 0 ||
      session.target_sequence != nullptr ||
      session.target_sequence_handle != 0 || !session.tickets.empty() ||
      !session.kda_tickets.empty() || !session.mla_tickets.empty() ||
      !session.moe_plans.empty() || !session.spine_source_uses.empty()) {
    throw std::logic_error(
        "provider cache trim requires a quiescent transaction boundary");
  }
}

}  // namespace

deltafin::provider_internal::SpineBindingDebugStats
deltafin::provider_internal::spine_binding_debug_stats(
    const DeltafinProviderSessionHandleV1 handle,
    const std::uint32_t layer_index,
    const std::uint64_t generation) {
  const auto session = find_session(handle);
  std::lock_guard<std::mutex> lock(session->mutex);
  session->require_open();
  return require_bound_spine_layer(*session, layer_index, generation)
      .binding_stats;
}

deltafin::provider_internal::SpineStoreDebugStats
deltafin::provider_internal::spine_store_debug_stats(
    const DeltafinProviderSessionHandleV1 handle) {
  const auto session = find_session(handle);
  std::lock_guard<std::mutex> lock(session->mutex);
  session->require_open();
  deltafin::provider_internal::SpineStoreDebugStats result;
  result.resident_prefix_layers = session->resident_spine_prefix_layers;
  result.resident_storage_bytes = session->resident_spine_storage_bytes;
  result.last_generation = session->last_spine_generation;
  if (session->transient_spine_layer != nullptr) {
    result.transient_bound = true;
    result.transient_layer = session->transient_spine_layer->layer_index;
    result.transient_generation = session->transient_spine_layer->generation;
    result.transient_storage_bytes =
        session->transient_spine_layer->binding_stats.resident_storage_bytes;
  }
  return result;
}

std::vector<float>
deltafin::provider_internal::spine_original_bf16_debug_project(
    const DeltafinProviderSessionHandleV1 handle,
    const std::uint32_t layer_index, const std::uint64_t generation,
    const std::uint32_t slot_index, const std::span<const float> input) {
  const auto session = find_session(handle);
  const c10::InferenceMode inference_guard;
  std::lock_guard<std::mutex> lock(session->mutex);
  session->require_open();
  const SpineLayerSlot& layer =
      require_bound_spine_layer(*session, layer_index, generation);
  if (slot_index >= layer.tensors.size() ||
      !layer.tensors[slot_index].has_value() ||
      !layer.tensors[slot_index]->original_bf16.defined()) {
    throw std::invalid_argument(
        "debug projection requires one bound original-BF16 matrix slot");
  }
  const auto& matrix = layer.tensors[slot_index]->original_bf16;
  if (matrix.columns == 0 || input.empty() ||
      input.size() % matrix.columns != 0 ||
      input.size() / matrix.columns > 64) {
    throw std::invalid_argument(
        "debug projection input must be contiguous fp32 [1..64,columns]");
  }
  const std::size_t positions = input.size() / matrix.columns;
  const at::Tensor device_input = copy_f32_to_device(
      input.data(), static_cast<std::uint64_t>(positions),
      static_cast<std::uint64_t>(matrix.columns),
      session->selected.device, false);
  const at::Tensor projected = original_bf16_linear(device_input, matrix);
  const at::Tensor cpu = projected.detach().to(at::kCPU).contiguous();
  if (cpu.scalar_type() != at::kFloat || cpu.dim() != 2 ||
      static_cast<std::size_t>(cpu.size(0)) != positions ||
      static_cast<std::size_t>(cpu.size(1)) != matrix.rows) {
    throw std::logic_error(
        "debug original-BF16 projection returned an invalid tensor");
  }
  std::vector<float> result(positions * matrix.rows);
  std::memcpy(result.data(), cpu.const_data_ptr<float>(),
              result.size() * sizeof(float));
  return result;
}

deltafin::provider_internal::SpineFp32ExecutionDebugReport
deltafin::provider_internal::spine_fp32_execution_debug(
    const DeltafinProviderSessionHandleV1 handle,
    const std::uint32_t layer_index, const std::uint64_t generation,
    const std::uint64_t owner, const std::uint32_t slot_index) {
  const auto session = find_session(handle);
  const c10::InferenceMode inference_guard;
  std::lock_guard<std::mutex> lock(session->mutex);
  session->require_open();
  const SpineLayerSlot& layer =
      require_bound_spine_layer(*session, layer_index, generation);
  auto execution = maybe_materialize_spine_fp32(*session, layer, owner);
  if (!execution.has_value() || slot_index >= execution->tensors.size() ||
      !execution->tensors[slot_index].has_value()) {
    throw std::runtime_error(
        "debug FP32 execution materializer did not publish the requested slot");
  }
  const at::Tensor& dense = *execution->tensors[slot_index];
  const at::Tensor cpu = dense.to(at::kCPU).contiguous();
  if (cpu.scalar_type() != at::kFloat || cpu.numel() != dense.numel()) {
    throw std::logic_error(
        "debug FP32 execution materializer returned an invalid tensor");
  }
  deltafin::provider_internal::SpineFp32ExecutionDebugReport result;
  result.values.resize(static_cast<std::size_t>(cpu.numel()));
  std::memcpy(result.values.data(), cpu.const_data_ptr<float>(),
              result.values.size() * sizeof(float));
  result.owner = execution->owner;
  result.spine_generation = execution->spine_generation;
  result.required_elements = execution->required_elements;
  result.capacity_elements = static_cast<std::uint64_t>(
      session->spine_fp32_execution_arena.storage.numel());
  result.storage_identity = reinterpret_cast<std::uintptr_t>(
      session->spine_fp32_execution_arena.storage.storage()
          .unsafeGetStorageImpl());
  result.layer_index = execution->layer_index;
  return result;
}

extern "C" int32_t deltafin_provider_session_create_v1(
    const DeltafinProviderSessionRequestV1* request,
    DeltafinProviderSessionReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider session request/report pointer is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "provider session request");
    if (report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider session report does not match provider ABI v1");
    }
    validate_session_dimensions(*request);
    const auto selected = deltafin::provider_internal::select_device(
        request->requested_device, request->device_index);
    const c10::InferenceMode inference_guard;
    auto session = std::make_shared<Session>(
        selected, request->flags, request->max_route_positions,
        request->synthetic_hidden_columns, request->synthetic_experts);
    const auto handle = insert_session(session);

    DeltafinProviderSessionReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.selected_device = selected.kind;
    produced.device_index = selected.index;
    produced.session = handle;
    produced.max_route_positions = request->max_route_positions;
    produced.flags = request->flags;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_memory_snapshot_v1(
    const DeltafinProviderMemoryRequestV1* request,
    DeltafinProviderMemoryReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider memory snapshot request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider memory snapshot request");
    if (report->struct_size != sizeof(*report) || request->reserved0 != 0 ||
        !all_zero(request->reserved) ||
        (request->actions & ~DELTAFIN_PROVIDER_MEMORY_TRIM_UNUSED_V1) != 0) {
      throw std::invalid_argument(
          "provider memory snapshot has invalid actions/reserved fields");
    }
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const bool trim_unused =
        (request->actions & DELTAFIN_PROVIDER_MEMORY_TRIM_UNUSED_V1) != 0;
    if (trim_unused) {
      require_memory_trim_boundary(*session);
      if (session->selected.device.is_cpu()) {
        throw std::invalid_argument(
            "CPU provider has no accelerator cache to trim");
      }
    }

    DeltafinProviderMemoryReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.selected_device = session->selected.kind;
    produced.device_index = session->selected.index;

    if (session->selected.device.is_mps()) {
#if defined(__APPLE__)
      const auto& hooks = at::detail::getMPSHooks();
      if (trim_unused) {
        // The public MPS hooks mirror torch.mps.synchronize/empty_cache.
        // Synchronization first lets completion handlers return deferred
        // buffers; emptyCache then releases only allocator-owned inactive
        // blocks and cannot invalidate a live Tensor.
        hooks.deviceSynchronize();
        hooks.emptyCache();
        produced.performed_actions |=
            DELTAFIN_PROVIDER_MEMORY_TRIM_UNUSED_V1;
      }
      produced.active_bytes = hooks.getCurrentAllocatedMemory();
      produced.reserved_bytes = hooks.getDriverAllocatedMemory();
      produced.recommended_bytes = hooks.getRecommendedMaxMemory();
      produced.available_fields =
          DELTAFIN_PROVIDER_MEMORY_ACTIVE_BYTES_V1 |
          DELTAFIN_PROVIDER_MEMORY_RESERVED_BYTES_V1;
      if (produced.recommended_bytes != 0) {
        produced.available_fields |=
            DELTAFIN_PROVIDER_MEMORY_RECOMMENDED_BYTES_V1;
      }
#else
      throw std::logic_error(
          "MPS provider selected in a non-Apple provider build");
#endif
    } else if (session->selected.device.is_cuda()) {
      const auto snapshot =
          deltafin::provider_internal::cuda_provider_memory_snapshot(
              session->selected.device, trim_unused);
      if (snapshot.active_valid) {
        produced.available_fields |=
            DELTAFIN_PROVIDER_MEMORY_ACTIVE_BYTES_V1;
        produced.active_bytes = snapshot.active_bytes;
      }
      if (snapshot.reserved_valid) {
        produced.available_fields |=
            DELTAFIN_PROVIDER_MEMORY_RESERVED_BYTES_V1;
        produced.reserved_bytes = snapshot.reserved_bytes;
      }
      if (snapshot.total_valid) {
        produced.available_fields |=
            DELTAFIN_PROVIDER_MEMORY_TOTAL_BYTES_V1;
        produced.total_bytes = snapshot.total_bytes;
      }
      if (snapshot.available_valid) {
        produced.available_fields |=
            DELTAFIN_PROVIDER_MEMORY_AVAILABLE_BYTES_V1;
        produced.available_bytes = snapshot.available_bytes;
      }
      if (snapshot.cache_trimmed) {
        produced.performed_actions |=
            DELTAFIN_PROVIDER_MEMORY_TRIM_UNUSED_V1;
      }
    }
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_pilot_enable_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetPilotEnableReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider target PILOT enable request");
    if (request->resource != 0 || report == nullptr ||
        report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider target PILOT enable request/report does not match ABI v1");
    }
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->flags != 0 ||
        (session->selected.kind != DELTAFIN_PROVIDER_DEVICE_CPU_V1 &&
         session->selected.kind != DELTAFIN_PROVIDER_DEVICE_MPS_V1)) {
      throw std::invalid_argument(
          "target PILOT admission requires a real CPU or MPS session");
    }
    if (session->target_pilot_enabled || session->last_spine_generation != 0 ||
        session->resident_spine_prefix_layers != 0 ||
        session->resident_spine_storage_bytes != 0 ||
        session->transient_spine_layer != nullptr ||
        std::any_of(session->resident_spine_layers.begin(),
                    session->resident_spine_layers.end(),
                    [](const auto& layer) { return layer != nullptr; }) ||
        session->target_global_groups[0] != nullptr ||
        session->target_global_groups[1] != nullptr ||
        session->target_tail != nullptr ||
        session->target_position != nullptr ||
        session->target_sequence != nullptr ||
        session->target_cache_store != nullptr ||
        std::any_of(session->target_pilot_routers.begin(),
                    session->target_pilot_routers.end(),
                    [](const auto& router) { return router.has_value(); })) {
      throw std::invalid_argument(
          "target PILOT must be admitted exactly once before any target or spine bind");
    }

    // This publishes only immutable admission state. Tensor allocation occurs
    // one detached layer at a time after each authoritative bind succeeds; an
    // individual optional clone failure never rolls that bind back.
    session->target_pilot_enabled = true;
    DeltafinProviderTargetPilotEnableReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.session = request->session;
    produced.enabled = 1;
    produced.layer_capacity =
        DELTAFIN_PROVIDER_TARGET_PILOT_LAYER_CAPACITY_V1;
    produced.reserve_bytes = kTargetPilotReserveBytes;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_cuda_cache_configure_v1(
    const DeltafinProviderCudaCacheConfigureRequestV1* request,
    DeltafinProviderCudaCacheConfigureReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider CUDA cache configuration request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "provider CUDA cache configuration request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        request->reserved0 != 0 || !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider CUDA cache configuration has invalid flags/reserved fields");
    }
    const bool automatic_capacity = request->capacity_mode ==
        DELTAFIN_PROVIDER_CUDA_CACHE_CAPACITY_AUTO_V1;
    if ((!automatic_capacity && request->capacity_mode !=
             DELTAFIN_PROVIDER_CUDA_CACHE_CAPACITY_EXACT_V1) ||
        (automatic_capacity && request->capacity_experts != 0) ||
        request->capacity_experts > 92u * DELTAFIN_PROVIDER_ROUTE_TOP_K_V1) {
      throw std::invalid_argument(
          "provider CUDA cache capacity policy is invalid");
    }
    using deltafin::provider_internal::CudaMoeCacheReserveKind;
    CudaMoeCacheReserveKind reserve_kind;
    switch (request->reserve_mode) {
      case DELTAFIN_PROVIDER_CUDA_CACHE_RESERVE_AUTO_V1:
        if (request->reserve_value != 0) {
          throw std::invalid_argument(
              "automatic CUDA cache reserve must have a zero value");
        }
        reserve_kind = CudaMoeCacheReserveKind::Auto;
        break;
      case DELTAFIN_PROVIDER_CUDA_CACHE_RESERVE_BYTES_V1:
        reserve_kind = CudaMoeCacheReserveKind::Bytes;
        break;
      case DELTAFIN_PROVIDER_CUDA_CACHE_RESERVE_RATIO_PPM_V1:
        if (request->reserve_value > 1'000'000) {
          throw std::invalid_argument(
              "CUDA cache reserve ratio must be in 0..1,000,000 ppm");
        }
        reserve_kind = CudaMoeCacheReserveKind::RatioPpm;
        break;
      default:
        throw std::invalid_argument("provider CUDA cache reserve mode is unknown");
    }
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->cuda_expert_cache == nullptr) {
      throw std::invalid_argument(
          "CUDA cache configuration requires a selected CUDA provider");
    }
    session->cuda_expert_cache->configure(
        deltafin::provider_internal::CudaMoeCachePolicy{
            .automatic_capacity = automatic_capacity,
            .capacity_experts = request->capacity_experts,
            .reserve_kind = reserve_kind,
            .reserve_value = request->reserve_value});

    DeltafinProviderCudaCacheConfigureReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.session = request->session;
    produced.capacity_mode = request->capacity_mode;
    produced.reserve_mode = request->reserve_mode;
    produced.capacity_experts = request->capacity_experts;
    produced.reserve_value = request->reserve_value;
    produced.configured = 1;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_metal_expert_layouts_v1(
    const DeltafinProviderMetalExpertLayoutsRequestV1* request,
    DeltafinProviderMetalExpertLayoutsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider Metal expert-layout request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider Metal expert-layout request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        request->reserved0 != 0 || !all_zero(request->reserved) ||
        (request->metal_shader_path == nullptr) !=
            (request->metal_shader_path_length == 0) ||
        request->metal_shader_path_length > 4096 ||
        request->metal_shader_path_length > SIZE_MAX) {
      throw std::invalid_argument(
          "provider Metal expert-layout request has invalid path/flags/reserved fields");
    }
    std::string shader_path;
    if (request->metal_shader_path != nullptr) {
      shader_path.assign(request->metal_shader_path,
                         static_cast<std::size_t>(
                             request->metal_shader_path_length));
      if (shader_path.find('\0') != std::string::npos) {
        throw std::invalid_argument(
            "provider Metal expert-layout path contains an embedded NUL");
      }
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (!session->selected.device.is_mps()) {
      throw std::invalid_argument(
          "Metal expert-layout qualification requires the selected MPS provider");
    }
    const auto capabilities =
        deltafin::provider_internal::qualify_metal_expert_layouts(shader_path);
    if ((capabilities.layout_capabilities & K3_CAP_RAW_V1) == 0 ||
        capabilities.raw_span_bytes != K3_RAW_V1_EXPERT_SPAN ||
        capabilities.scale4_span_bytes != K3_SCALE4_V2_EXPERT_SPAN) {
      throw std::runtime_error(
          "Metal expert-layout capability report violated the provider contract");
    }

    DeltafinProviderMetalExpertLayoutsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.session = request->session;
    produced.descriptor_abi = capabilities.descriptor_abi;
    produced.layout_capabilities = capabilities.layout_capabilities;
    produced.raw_span_bytes = capabilities.raw_span_bytes;
    produced.scale4_span_bytes = capabilities.scale4_span_bytes;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_metal_expert_cache_flush_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider Metal expert-cache flush request");
    if (request->resource != 0) {
      throw std::invalid_argument(
          "provider Metal expert-cache flush resource must be zero");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (!session->selected.device.is_mps()) {
      throw std::invalid_argument(
          "Metal expert-cache flush requires the selected MPS provider");
    }
    deltafin::provider_internal::flush_metal_expert_cache();
  });
}

extern "C" int32_t deltafin_provider_metal_expert_cache_stats_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderMetalExpertCacheStatsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider Metal expert-cache stats request");
    if (request->resource != 0 || report == nullptr ||
        report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider Metal expert-cache stats request/report does not match ABI v1");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (!session->selected.device.is_mps()) {
      throw std::invalid_argument(
          "Metal expert-cache stats require the selected MPS provider");
    }
    const auto stats = deltafin::provider_internal::metal_expert_cache_stats();
    DeltafinProviderMetalExpertCacheStatsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.session = request->session;
    produced.calls = stats.calls;
    produced.zero_copy_wraps = stats.zero_copy_wraps;
    produced.copies = stats.copies;
    produced.cache_entries = stats.cache_entries;
    produced.bindless = stats.bindless;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_session_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider session destroy request");
    if (request->resource != 0) {
      throw std::invalid_argument(
          "provider session destroy request resource must be zero");
    }
    std::shared_ptr<Session> session;
    {
      std::lock_guard<std::mutex> lock(sessions_mutex);
      const auto found = sessions.find(request->session);
      if (found == sessions.end()) {
        throw std::invalid_argument("provider session handle is stale or unknown");
      }
      session = found->second;
      sessions.erase(found);
    }
    // Wait for any in-flight call that already retained the session. Calls
    // that race behind this lock observe closed=true and fail without work.
    std::lock_guard<std::mutex> lock(session->mutex);
    session->closed = true;
    // The Metal cache is process-global and its no-copy wrappers alias Rust
    // arena pages. Session destruction is the final defensive boundary; the
    // engine normally flushes earlier, before either reader arena can drop.
    if (session->selected.device.is_mps()) {
      try {
        deltafin::provider_internal::flush_metal_expert_cache();
      } catch (...) {
        // A valid MPS session may use CPU experts, and modular provider builds
        // need not link the Metal MoE bridge. Explicit Metal reader hooks are
        // fail-closed; generic session destruction must remain no-throw here.
      }
    }
    // Target destructors roll back every unpublished MLA/KDA stage and must
    // run while their session-owned caches and immutable tail still live.
    session->target_sequence.reset();
    session->target_sequence_handle = 0;
    session->target_position.reset();
    session->target_position_handle = 0;
    session->target_state_branch.reset();
    session->target_cache_store.reset();
    session->target_tail.reset();
    for (auto& group : session->target_global_groups) {
      group.reset();
    }
    for (auto& router : session->target_pilot_routers) {
      router.reset();
    }
    session->target_pilot_enabled = false;
    session->qualified_target_packed_shapes.clear();
    if (session->cuda_expert_cache != nullptr) {
      for (const auto& [plan, slot] : session->moe_plans) {
        static_cast<void>(slot);
        session->cuda_expert_cache->cancel_plan(plan);
      }
    }
    session->moe_plans.clear();
    session->mla_tickets.clear();
    session->mla_caches.clear();
    session->kda_tickets.clear();
    session->kda_caches.clear();
    session->tickets.clear();
    session->caches.clear();
    session->tensors.clear();
    session->transient_spine_layer.reset();
    session->spine_source_uses.clear();
    for (auto& layer : session->resident_spine_layers) {
      layer.reset();
    }
    session->resident_spine_prefix_layers = 0;
    session->resident_spine_storage_bytes = 0;
    session->last_spine_generation = 0;
  });
}

extern "C" int32_t deltafin_provider_tensor_upload_f32_v1(
    const DeltafinProviderTensorUploadF32V1* request,
    DeltafinProviderTensorReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider tensor upload request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "provider tensor upload request");
    if (report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider tensor report does not match provider ABI v1");
    }
    const std::uint64_t elements =
        checked_elements(request->rows, request->columns);
    if (request->data == nullptr || request->element_count != elements ||
        request->flags != 0 || request->reserved0 != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider tensor upload has invalid data/count/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    at::Tensor tensor = copy_f32_to_device(
        request->data, request->rows, request->columns,
        session->selected.device, false);
    const auto handle = session->allocate_resource();
    session->tensors.emplace(handle, std::move(tensor));

    DeltafinProviderTensorReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.tensor = handle;
    produced.rows = request->rows;
    produced.columns = request->columns;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_tensor_upload_bf16_v1(
    const DeltafinProviderTensorUploadBf16V1* request,
    DeltafinProviderTensorReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider BF16 tensor upload request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider BF16 tensor upload request");
    if (report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider tensor report does not match provider ABI v1");
    }
    const std::uint64_t elements =
        checked_elements(request->rows, request->columns);
    if (request->data == nullptr || elements > UINT64_MAX / 2 ||
        request->byte_length != elements * 2 || request->flags != 0 ||
        request->reserved0 != 0 || !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider BF16 tensor upload has invalid data/length/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    at::Tensor tensor = copy_bf16_to_device(
        request->data, request->rows, request->columns,
        session->selected.device);
    const auto handle = session->allocate_resource();
    const auto [ignored, inserted] =
        session->tensors.emplace(handle, std::move(tensor));
    static_cast<void>(ignored);
    if (!inserted) {
      throw std::runtime_error("provider BF16 tensor handle collision");
    }

    DeltafinProviderTensorReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.tensor = handle;
    produced.rows = request->rows;
    produced.columns = request->columns;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_tensor_read_f32_v1(
    const DeltafinProviderTensorReadF32V1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr) {
      throw std::invalid_argument("provider tensor read request is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "provider tensor read request");
    if (request->flags != 0 || request->reserved0 != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider tensor read contains unknown flags or reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto found = session->tensors.find(request->tensor);
    if (found == session->tensors.end()) {
      throw std::invalid_argument("provider tensor handle is stale or unknown");
    }
    copy_f32_to_caller(found->second, request->destination,
                       request->element_capacity);
  });
}

extern "C" int32_t deltafin_provider_tensor_release_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider tensor release request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->tensors.erase(request->resource) != 1) {
      throw std::invalid_argument("provider tensor handle is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_cache_create_f32_v1(
    const DeltafinProviderCacheCreateF32V1* request,
    DeltafinProviderCacheReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider cache create request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "provider cache create request");
    if (report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider cache report does not match provider ABI v1");
    }
    const std::uint64_t elements =
        checked_elements(request->rows, request->columns);
    const bool zero_initialized = request->initial_data == nullptr;
    if ((zero_initialized && request->element_count != 0) ||
        (!zero_initialized && request->element_count != elements) ||
        request->flags != 0 || request->reserved0 != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider cache create has invalid data/count/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    at::Tensor tensor = copy_f32_to_device(
        request->initial_data, request->rows, request->columns,
        session->selected.device, true);
    const auto handle = session->allocate_resource();
    session->caches.emplace(handle, CacheSlot{std::move(tensor), 0});

    DeltafinProviderCacheReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.cache = handle;
    produced.rows = request->rows;
    produced.columns = request->columns;
    produced.version = 0;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_cache_read_f32_v1(
    const DeltafinProviderCacheReadF32V1* request,
    DeltafinProviderCacheReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider cache read request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "provider cache read request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        request->reserved0 != 0 || !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider cache read has invalid report/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto found = session->caches.find(request->cache);
    if (found == session->caches.end()) {
      throw std::invalid_argument("provider cache handle is stale or unknown");
    }
    copy_f32_to_caller(found->second.tensor, request->destination,
                       request->element_capacity);
    DeltafinProviderCacheReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.cache = request->cache;
    produced.rows = static_cast<std::uint64_t>(found->second.tensor.size(0));
    produced.columns = static_cast<std::uint64_t>(found->second.tensor.size(1));
    produced.version = found->second.version;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_cache_release_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider cache release request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    for (const auto& [ignored, ticket] : session->tickets) {
      static_cast<void>(ignored);
      if (ticket.cache == request->resource) {
        throw std::runtime_error(
            "provider cache still has a live prepare-layer ticket");
      }
    }
    if (session->caches.erase(request->resource) != 1) {
      throw std::invalid_argument("provider cache handle is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_prepare_layer_v1(
    const DeltafinProviderPrepareLayerRequestV1* request,
    DeltafinProviderRouteMailboxV1* mailbox, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || mailbox == nullptr) {
      throw std::invalid_argument("provider prepare request/mailbox is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "provider prepare-layer request");
    if (mailbox->struct_size != sizeof(*mailbox) || request->flags != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider prepare layer has invalid mailbox/flags/reserved fields");
    }
    if (request->layer_index >= kK3Layers) {
      throw std::invalid_argument("provider layer index is outside K3's 93 layers");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if ((session->flags & kSyntheticFlag) == 0) {
      throw std::runtime_error(
          "provider session has no loaded K3 layer tape; placeholder execution is forbidden");
    }
    const auto hidden_found = session->tensors.find(request->hidden);
    const auto cache_found = session->caches.find(request->cache);
    if (hidden_found == session->tensors.end()) {
      throw std::invalid_argument("provider hidden tensor handle is stale or unknown");
    }
    if (cache_found == session->caches.end()) {
      throw std::invalid_argument("provider cache handle is stale or unknown");
    }
    const at::Tensor& hidden = hidden_found->second;
    const CacheSlot& cache = cache_found->second;
    if (hidden.dim() != 2 || cache.tensor.dim() != 2 ||
        hidden.scalar_type() != at::kFloat ||
        cache.tensor.scalar_type() != at::kFloat ||
        hidden.sizes() != cache.tensor.sizes() ||
        hidden.size(1) != static_cast<std::int64_t>(session->hidden_columns)) {
      throw std::invalid_argument(
          "provider split hidden/cache shape or fp32 contract does not match the session");
    }
    const auto positions = static_cast<std::uint64_t>(hidden.size(0));
    if (positions == 0 || positions > session->max_route_positions) {
      throw std::invalid_argument(
          "provider split positions exceed the session's fixed route mailbox");
    }
    const std::uint64_t edges =
        positions * DELTAFIN_PROVIDER_ROUTE_TOP_K_V1;
    if (edges > DELTAFIN_PROVIDER_ROUTE_MAX_EDGES_V1) {
      throw std::runtime_error("provider route edge count exceeds ABI mailbox");
    }

    // This exact synthetic tape proves the split boundary and ownership model:
    // cache state is staged in the ticket, not committed by prepare. The real
    // KDA/MLA layer tape will replace only this program body after weights load.
    const at::Tensor next_cache = at::add(cache.tensor, hidden);
    const at::Tensor logits = at::matmul(hidden, session->router_weight);
    const at::Tensor scores = at::sigmoid(logits);
    const at::Tensor choice = at::add(scores, session->router_bias);
    const auto [ignored_values, topk_ids] = at::topk(
        choice, DELTAFIN_PROVIDER_ROUTE_TOP_K_V1, -1, true, false);
    static_cast<void>(ignored_values);
    const at::Tensor unnormalized = at::gather(scores, 1, topk_ids);
    const at::Tensor denominator = at::add(
        at::sum(unnormalized, std::vector<std::int64_t>{-1}, true), 1e-20);
    const at::Tensor weights = at::div(unnormalized, denominator);
    const at::Tensor ids_cpu = topk_ids.to(at::kCPU).contiguous();
    const at::Tensor weights_cpu = weights.to(at::kCPU).contiguous();
    const auto* ids = ids_cpu.const_data_ptr<std::int64_t>();
    const auto* weight_values = weights_cpu.const_data_ptr<float>();

    DeltafinProviderRouteMailboxV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.positions = static_cast<std::uint32_t>(positions);
    produced.top_k = DELTAFIN_PROVIDER_ROUTE_TOP_K_V1;
    produced.edge_count = static_cast<std::uint32_t>(edges);
    produced.hidden_columns =
        static_cast<std::uint64_t>(hidden.size(1));
    produced.cache_version = cache.version;
    for (std::uint64_t edge = 0; edge < edges; ++edge) {
      const std::int64_t expert = ids[edge];
      const float weight = weight_values[edge];
      if (expert < 0 || expert >= static_cast<std::int64_t>(session->experts) ||
          !std::isfinite(weight) || weight < 0.0F) {
        throw std::runtime_error(
            "provider router produced an invalid expert or fp32 weight");
      }
      produced.ordered_experts[edge] = static_cast<std::uint16_t>(expert);
      produced.ordered_weight_bits[edge] = std::bit_cast<std::uint32_t>(weight);
    }

    const auto ticket = session->allocate_resource();
    const auto [ignored, inserted] = session->tickets.emplace(
        ticket, TicketSlot{next_cache, next_cache, request->cache,
                           cache.version, request->layer_index});
    static_cast<void>(ignored);
    if (!inserted) {
      throw std::runtime_error("provider layer ticket handle collision");
    }
    produced.ticket = ticket;
    *mailbox = produced;
  });
}

extern "C" int32_t deltafin_provider_finish_layer_v1(
    const DeltafinProviderFinishLayerRequestV1* request,
    DeltafinProviderFinishLayerReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider finish request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "provider finish-layer request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        request->reserved0 != 0 || !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider finish layer has invalid report/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto ticket_found = session->tickets.find(request->ticket);
    const auto expert_found = session->tensors.find(request->expert_output);
    if (ticket_found == session->tickets.end()) {
      throw std::invalid_argument("provider layer ticket is stale or unknown");
    }
    if (expert_found == session->tensors.end()) {
      throw std::invalid_argument("provider expert-output tensor is stale or unknown");
    }
    const TicketSlot ticket = ticket_found->second;
    const auto cache_found = session->caches.find(ticket.cache);
    if (cache_found == session->caches.end()) {
      throw std::runtime_error("provider ticket's cache no longer exists");
    }
    if (cache_found->second.version != ticket.expected_cache_version) {
      throw std::runtime_error(
          "provider cache advanced after prepare; stale layer ticket refused");
    }
    if (cache_found->second.version ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::runtime_error("provider cache version is exhausted");
    }
    const at::Tensor& expert = expert_found->second;
    if (expert.scalar_type() != at::kFloat ||
        expert.sizes() != ticket.prepared.sizes()) {
      throw std::invalid_argument(
          "provider expert output does not match prepared fp32 hidden shape");
    }
    const at::Tensor output = at::add(ticket.prepared, expert);
    const auto output_handle = session->allocate_resource();
    const auto [ignored, inserted] =
        session->tensors.emplace(output_handle, output);
    static_cast<void>(ignored);
    if (!inserted) {
      throw std::runtime_error("provider output tensor handle collision");
    }

    // Commit is last. Any exception before this point leaves both the cache and
    // ticket untouched, so Rust can retry or discard the transaction safely.
    cache_found->second.tensor = ticket.next_cache;
    ++cache_found->second.version;
    const std::uint64_t committed_version = cache_found->second.version;
    session->tickets.erase(ticket_found);

    DeltafinProviderFinishLayerReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.output = output_handle;
    produced.positions = static_cast<std::uint64_t>(output.size(0));
    produced.hidden_columns = static_cast<std::uint64_t>(output.size(1));
    produced.committed_cache_version = committed_version;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_ticket_release_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider ticket release request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->tickets.erase(request->resource) != 1) {
      throw std::invalid_argument("provider layer ticket is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_bind_spine_layer_v1(
    const DeltafinProviderBindSpineLayerRequestV1* request,
    DeltafinProviderBindSpineLayerReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider bind-spine request/report pointer is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider bind-spine request");
    if (report->struct_size != sizeof(*report) ||
        (request->flags & ~kKnownBindSpineFlags) != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider bind-spine has invalid report/flags/reserved fields");
    }
    if (request->layer_index >= kK3Layers || request->generation == 0) {
      throw std::invalid_argument(
          "provider bind-spine layer/generation is outside its contract");
    }
    validate_spine_buffer(request->quantized, request->quantized_length,
                          "quantized");
    validate_spine_buffer(request->scales, request->scales_length, "scales");
    validate_spine_buffer(request->other, request->other_length, "other");

    // Descriptor validation includes every rank, encoding, slot, offset,
    // length, overlap, and reserved field. No payload byte is inspected until
    // this complete transaction-wide validation succeeds.
    auto descriptors = validate_spine_descriptors(*request);
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (request->generation <= session->last_spine_generation) {
      throw std::invalid_argument(
          "provider bind-spine generation must advance monotonically");
    }
    const bool retain = (request->flags & kRetainSpineFlag) != 0;
    if (retain) {
      if (request->layer_index != session->resident_spine_prefix_layers) {
        throw std::invalid_argument(
            "provider retained spine layers must append to the ordered prefix");
      }
      if (session->resident_spine_layers[request->layer_index] != nullptr) {
        throw std::logic_error(
            "provider retained spine prefix slot is already allocated");
      }
      if (session->transient_spine_layer != nullptr &&
          session->transient_spine_layer->layer_index ==
              request->layer_index) {
        throw std::invalid_argument(
            "provider cannot retain a layer while the same layer occupies the transient slot");
      }
    } else if (request->layer_index <
               session->resident_spine_prefix_layers) {
      throw std::invalid_argument(
          "provider cannot transiently replace a retained spine layer");
    }

    const auto mla_shape = (session->flags & kSyntheticMlaFlag) != 0
        ? deltafin::provider_internal::MlaShape::small_canary()
        : deltafin::provider_internal::MlaShape::k3();
    const bool gather_mla_inputs = should_gather_mla_input_weights(
        descriptors, request->layer_index, mla_shape,
        session->selected.kind);

    // Build one entirely detached candidate through the same representation-
    // aware uploader used by globals and V2 borrowing. On CPU, large original
    // BF16 matrices remain provider-owned uint16 checkpoint bits; norms and
    // [1,H] residual weights are losslessly promoted to fp32.
    auto staged = upload_validated_spine_payload(
        *request, descriptors, *session, request->layer_index,
        gather_mla_inputs, true, false);

    DeltafinProviderBindSpineLayerReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.layer_index = staged->layer_index;
    produced.tensor_count = staged->tensor_count;
    produced.generation = staged->generation;
    produced.quantized_tensor_count = staged->quantized_tensor_count;
    produced.raw_tensor_count = staged->raw_tensor_count;
    produced.quantized_bytes = staged->quantized_bytes;
    produced.scales_bytes = staged->scales_bytes;
    produced.other_bytes = staged->other_bytes;
    produced.resident_storage_bytes =
        staged->binding_stats.resident_storage_bytes;

    // All arithmetic that can fail happens before the noexcept pointer commit.
    // The previous transient and the complete resident prefix survive every
    // validation, allocation, conversion, and device-upload failure above.
    std::uint64_t next_resident_bytes =
        session->resident_spine_storage_bytes;
    if (retain) {
      next_resident_bytes = checked_spine_sum(
          next_resident_bytes, produced.resident_storage_bytes,
          "resident spine prefix storage bytes");
    }
    SpineLayerSlot* published = nullptr;
    if (retain) {
      session->resident_spine_layers[request->layer_index].swap(staged);
      ++session->resident_spine_prefix_layers;
      session->resident_spine_storage_bytes = next_resident_bytes;
      published =
          session->resident_spine_layers[request->layer_index].get();
    } else {
      session->transient_spine_layer.swap(staged);
      published = session->transient_spine_layer.get();
    }
    session->last_spine_generation = request->generation;
    // The authoritative pointer/generation commit above is complete and is
    // never rolled back by this scheduling-only optimization. Populate a
    // previously empty immutable roster slot only from that published layer.
    if (published != nullptr) {
      maybe_publish_compact_pilot_router(*session, *published);
    }
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_bind_spine_layer_v2(
    const DeltafinProviderBindSpineLayerRequestV2* request,
    DeltafinProviderBindSpineLayerReportV2* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider bind-spine-v2 request/report pointer is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider bind-spine-v2 request");
    if (report->struct_size != sizeof(*report) ||
        (request->flags & ~kKnownBindSpineV2Flags) != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider bind-spine-v2 has invalid report/flags/reserved fields");
    }
    validate_spine_allocation(
        request->quantized, request->quantized_length,
        request->quantized_allocation_length, "quantized");
    validate_spine_allocation(request->scales, request->scales_length,
                              request->scales_allocation_length, "scales");
    validate_spine_allocation(request->other, request->other_length,
                              request->other_allocation_length, "other");

    DeltafinProviderBindSpineLayerRequestV1 detached_request = {};
    detached_request.struct_size = sizeof(detached_request);
    detached_request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    detached_request.session = request->session;
    detached_request.layer_index = request->layer_index;
    detached_request.flags = request->flags & kRetainSpineFlag;
    detached_request.generation = request->generation;
    detached_request.descriptors = request->descriptors;
    detached_request.descriptor_count = request->descriptor_count;
    detached_request.quantized =
        request->quantized_length == 0 ? nullptr : request->quantized;
    detached_request.quantized_length = request->quantized_length;
    detached_request.scales =
        request->scales_length == 0 ? nullptr : request->scales;
    detached_request.scales_length = request->scales_length;
    detached_request.other =
        request->other_length == 0 ? nullptr : request->other;
    detached_request.other_length = request->other_length;

    const auto publish_detached = [&] {
      DeltafinProviderBindSpineLayerReportV1 detached_report = {};
      detached_report.struct_size = sizeof(detached_report);
      std::array<char, 1024> detached_error = {};
      if (deltafin_provider_bind_spine_layer_v1(
              &detached_request, &detached_report, detached_error.data(),
              detached_error.size()) != 0) {
        throw std::runtime_error(std::string("detached spine bind failed: ") +
                                 detached_error.data());
      }
      DeltafinProviderBindSpineLayerReportV2 produced = {};
      produced.struct_size = sizeof(produced);
      produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
      produced.layer_index = detached_report.layer_index;
      produced.tensor_count = detached_report.tensor_count;
      produced.generation = detached_report.generation;
      produced.quantized_tensor_count = detached_report.quantized_tensor_count;
      produced.raw_tensor_count = detached_report.raw_tensor_count;
      produced.quantized_bytes = detached_report.quantized_bytes;
      produced.scales_bytes = detached_report.scales_bytes;
      produced.other_bytes = detached_report.other_bytes;
      produced.resident_storage_bytes = detached_report.resident_storage_bytes;
      produced.source_use_kind =
          DELTAFIN_PROVIDER_SPINE_SOURCE_DETACHED_V2;
      *report = produced;
    };

    const bool allow_borrow =
        (request->flags & kAllowBorrowSpineFlag) != 0;
    if (!allow_borrow) {
      publish_detached();
      return;
    }
    if ((request->flags & kRetainSpineFlag) != 0) {
      throw std::invalid_argument(
          "retained spine layers may never opt into borrowed storage");
    }
    if (request->layer_index >= kK3Layers || request->generation == 0) {
      throw std::invalid_argument(
          "provider bind-spine-v2 layer/generation is outside its contract");
    }

    auto descriptors = validate_spine_descriptors(detached_request);
    const bool has_borrowable_matrix = std::any_of(
        descriptors.begin(), descriptors.end(),
        [](const ValidatedSpineDescriptor& descriptor) {
          return descriptor.raw.encoding ==
                     DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 &&
              descriptor.shape.size() == 2 && descriptor.shape[0] > 1;
        });
    const auto session = find_session(request->session);
    bool cpu_single_row = false;
    {
      std::lock_guard<std::mutex> lock(session->mutex);
      session->require_open();
      cpu_single_row = session->selected.device.is_cpu() &&
          session->target_sequence != nullptr &&
          session->target_sequence->position_count() == 1 &&
          session->target_sequence->next_layer_index() == request->layer_index;
    }
    // The opt-in is permission, not a promise. Non-CPU providers, multi-row
    // verification/prefill, inactive sessions, and representations without a
    // raw matrix retain the established detached V1 behavior byte-for-byte.
    if (!cpu_single_row || !has_borrowable_matrix) {
      publish_detached();
      return;
    }

    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (!session->selected.device.is_cpu() ||
        session->target_sequence == nullptr ||
        session->target_sequence->position_count() != 1 ||
        session->target_sequence->next_layer_index() != request->layer_index) {
      throw std::runtime_error(
          "borrowed CPU spine eligibility changed during bind");
    }
    if (!session->spine_source_uses.empty()) {
      throw std::runtime_error(
          "borrowed CPU spine cannot replace an unreclaimed source use");
    }
    if (request->generation <= session->last_spine_generation ||
        request->layer_index < session->resident_spine_prefix_layers) {
      throw std::invalid_argument(
          "borrowed CPU spine generation/layer conflicts with published state");
    }

    auto staged = upload_validated_spine_payload(
        detached_request, descriptors, *session, request->layer_index, false,
        true, true);
    if (staged->borrowed_tensor_count == 0 ||
        staged->borrowed_source_bytes == 0) {
      throw std::logic_error(
          "borrowed CPU spine selected without a borrowed matrix");
    }

    DeltafinProviderBindSpineLayerReportV2 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.layer_index = staged->layer_index;
    produced.tensor_count = staged->tensor_count;
    produced.generation = staged->generation;
    produced.quantized_tensor_count = staged->quantized_tensor_count;
    produced.raw_tensor_count = staged->raw_tensor_count;
    produced.quantized_bytes = staged->quantized_bytes;
    produced.scales_bytes = staged->scales_bytes;
    produced.other_bytes = staged->other_bytes;
    produced.resident_storage_bytes =
        staged->binding_stats.resident_storage_bytes;
    produced.source_use_kind =
        DELTAFIN_PROVIDER_SPINE_SOURCE_BORROWED_V2;
    produced.borrowed_tensor_count = staged->borrowed_tensor_count;
    produced.borrowed_source_bytes = staged->borrowed_source_bytes;
    produced.source_use = session->allocate_resource();

    const auto [source_position, inserted] =
        session->spine_source_uses.emplace(
            produced.source_use,
            SpineSourceUseSlot{request->generation, request->layer_index,
                               DELTAFIN_PROVIDER_SPINE_SOURCE_OPEN_V2});
    static_cast<void>(source_position);
    if (!inserted) {
      throw std::runtime_error("provider spine source-use handle collision");
    }
    // unordered_map publication above is the final potentially throwing
    // operation. unique_ptr swap, scalar commits, and report assignment below
    // are noexcept; a successful return is therefore the sole ownership handoff.
    session->transient_spine_layer.swap(staged);
    session->last_spine_generation = request->generation;
    maybe_publish_compact_pilot_router(
        *session, *session->transient_spine_layer);
    *report = produced;
  });
}

namespace {

void require_spine_source_use_request(
    const DeltafinProviderSpineSourceUseRequestV2* request,
    DeltafinProviderSpineSourceUseReportV2* report, const char* operation) {
  if (request == nullptr || report == nullptr) {
    throw std::invalid_argument(std::string(operation) +
                                " request/report pointer is null");
  }
  require_header(request->struct_size, sizeof(*request),
                 request->abi_version, operation);
  if (report->struct_size != sizeof(*report) || request->session == 0 ||
      request->source_use == 0 || request->generation == 0 ||
      request->flags != 0 || request->reserved0 != 0 ||
      !all_zero(request->reserved)) {
    throw std::invalid_argument(std::string(operation) +
                                " has invalid handle/flags/reserved fields");
  }
}

DeltafinProviderSpineSourceUseReportV2 spine_source_use_report(
    const DeltafinProviderSpineSourceUseRequestV2& request,
    const std::uint32_t state, const std::uint32_t ready) {
  DeltafinProviderSpineSourceUseReportV2 produced = {};
  produced.struct_size = sizeof(produced);
  produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  produced.source_use = request.source_use;
  produced.generation = request.generation;
  produced.state = state;
  produced.ready = ready;
  return produced;
}

SpineSourceUseSlot& require_spine_source_use(
    Session& session, const DeltafinProviderSpineSourceUseRequestV2& request,
    const char* operation) {
  const auto found = session.spine_source_uses.find(request.source_use);
  if (found == session.spine_source_uses.end() ||
      found->second.generation != request.generation) {
    throw std::invalid_argument(std::string(operation) +
                                " source-use handle is stale or unknown");
  }
  return found->second;
}

void clear_borrowed_cpu_carriers(Session& session,
                                 const SpineSourceUseSlot& source,
                                 const bool discard_layer) {
  if (session.transient_spine_layer == nullptr ||
      session.transient_spine_layer->generation != source.generation ||
      session.transient_spine_layer->layer_index != source.layer_index) {
    throw std::logic_error(
        "spine source-use no longer names the published transient layer");
  }
  if (discard_layer) {
    session.transient_spine_layer.reset();
    return;
  }
  std::uint32_t cleared = 0;
  for (auto& tensor : session.transient_spine_layer->tensors) {
    if (tensor.has_value() && tensor->original_bf16.is_borrowed_cpu()) {
      tensor->original_bf16 = {};
      ++cleared;
    }
  }
  if (session.transient_spine_layer->mla_input_bundle != nullptr &&
      session.transient_spine_layer->mla_input_bundle->projection
          .original_bf16.is_borrowed_cpu()) {
    session.transient_spine_layer->mla_input_bundle.reset();
  }
  if (cleared != session.transient_spine_layer->borrowed_tensor_count) {
    throw std::logic_error(
        "spine source-use borrowed tensor roster is corrupt");
  }
  session.transient_spine_layer->borrowed_tensor_count = 0;
  session.transient_spine_layer->borrowed_source_bytes = 0;
}

}  // namespace

extern "C" int32_t deltafin_provider_spine_source_use_seal_v2(
    const DeltafinProviderSpineSourceUseRequestV2* request,
    DeltafinProviderSpineSourceUseReportV2* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    constexpr const char* operation = "provider spine source-use seal";
    require_spine_source_use_request(request, report, operation);
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    SpineSourceUseSlot& source =
        require_spine_source_use(*session, *request, operation);
    if (source.state != DELTAFIN_PROVIDER_SPINE_SOURCE_OPEN_V2) {
      throw std::invalid_argument(
          "provider spine source-use seal is consume-once");
    }
    source.state = DELTAFIN_PROVIDER_SPINE_SOURCE_SEALED_V2;
    *report = spine_source_use_report(
        *request, DELTAFIN_PROVIDER_SPINE_SOURCE_SEALED_V2, 0);
  });
}

extern "C" int32_t deltafin_provider_spine_source_use_try_reclaim_v2(
    const DeltafinProviderSpineSourceUseRequestV2* request,
    DeltafinProviderSpineSourceUseReportV2* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    constexpr const char* operation = "provider spine source-use reclaim";
    require_spine_source_use_request(request, report, operation);
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const SpineSourceUseSlot source =
        require_spine_source_use(*session, *request, operation);
    if (source.state != DELTAFIN_PROVIDER_SPINE_SOURCE_SEALED_V2) {
      throw std::invalid_argument(
          "provider spine source-use must be sealed before reclaim");
    }
    // CPU projections and expert completion are synchronous. Clearing every
    // carrier is the completion fence; after this point no provider object
    // retains an arena address and Rust may recycle the slab immediately.
    clear_borrowed_cpu_carriers(*session, source, false);
    if (session->spine_source_uses.erase(request->source_use) != 1) {
      throw std::logic_error("provider spine source-use reclaim lost its handle");
    }
    *report = spine_source_use_report(
        *request, DELTAFIN_PROVIDER_SPINE_SOURCE_RECLAIMED_V2, 1);
  });
}

extern "C" int32_t deltafin_provider_spine_source_use_abort_v2(
    const DeltafinProviderSpineSourceUseRequestV2* request,
    DeltafinProviderSpineSourceUseReportV2* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    constexpr const char* operation = "provider spine source-use abort";
    require_spine_source_use_request(request, report, operation);
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const SpineSourceUseSlot source =
        require_spine_source_use(*session, *request, operation);
    if (source.state != DELTAFIN_PROVIDER_SPINE_SOURCE_OPEN_V2 &&
        source.state != DELTAFIN_PROVIDER_SPINE_SOURCE_SEALED_V2) {
      throw std::invalid_argument(
          "provider spine source-use cannot abort from its current state");
    }
    clear_borrowed_cpu_carriers(*session, source, true);
    if (session->spine_source_uses.erase(request->source_use) != 1) {
      throw std::logic_error("provider spine source-use abort lost its handle");
    }
    *report = spine_source_use_report(
        *request, DELTAFIN_PROVIDER_SPINE_SOURCE_ABORTED_V2, 1);
  });
}

extern "C" int32_t deltafin_provider_spine_tensor_read_f32_v1(
    const DeltafinProviderSpineTensorReadF32V1* request,
    DeltafinProviderSpineTensorReadReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider spine tensor read request/report pointer is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider spine tensor read request");
    if (report->struct_size != sizeof(*report) || request->destination == nullptr ||
        request->flags != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider spine tensor read has invalid output/flags/reserved fields");
    }
    if (request->component != DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1 &&
        request->component !=
            DELTAFIN_PROVIDER_SPINE_COMPONENT_AUXILIARY_V1) {
      throw std::invalid_argument(
          "provider spine tensor read names an invalid component");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const SpineLayerSlot& layer = require_bound_spine_layer(
        *session, request->layer_index, request->generation);
    if (request->slot >= layer.tensors.size() ||
        !layer.tensors[request->slot].has_value()) {
      throw std::invalid_argument(
          "provider spine tensor read slot is not loaded");
    }
    const SpineTensorSlot& slot = *layer.tensors[request->slot];
    const bool auxiliary = request->component ==
                           DELTAFIN_PROVIDER_SPINE_COMPONENT_AUXILIARY_V1;
    if (auxiliary && !slot.auxiliary.defined()) {
      throw std::invalid_argument(
          "provider spine tensor has no auxiliary component");
    }
    if (!auxiliary && !slot.data.defined() &&
        !slot.original_bf16.defined()) {
      throw std::invalid_argument(
          "provider spine tensor has no data component");
    }
    const at::Tensor& tensor = auxiliary ? slot.auxiliary : slot.data;
    const bool original_bf16 = !auxiliary && slot.original_bf16.defined();
    const std::uint64_t elements = original_bf16
        ? static_cast<std::uint64_t>(slot.original_bf16.rows) *
            static_cast<std::uint64_t>(slot.original_bf16.columns)
        : static_cast<std::uint64_t>(tensor.numel());
    if (request->element_capacity < elements) {
      throw std::invalid_argument(
          "provider spine tensor destination is too small");
    }
    // Keep diagnostic transfer and scalar conversion separate. In particular,
    // an int8 device view is copied byte-exactly before CPU fp32 conversion;
    // this avoids asking a backend copy kernel to fuse conversion at the final
    // storage boundary and makes this ownership oracle independent of device
    // conversion-vectorization details.
    const at::Tensor cpu = original_bf16
        ? deltafin::provider_internal::materialize_original_bf16_f32(
              slot.original_bf16).detach().to(at::kCPU).contiguous()
        : tensor.detach().to(at::kCPU).contiguous().to(at::kFloat).contiguous();
    std::memcpy(request->destination, cpu.const_data_ptr<float>(),
                static_cast<std::size_t>(elements) * sizeof(float));

    DeltafinProviderSpineTensorReadReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.stored_scalar_type = original_bf16
        ? DELTAFIN_PROVIDER_SPINE_SCALAR_BF16_V1
        : (!auxiliary && slot.encoding ==
                               DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1
               ? DELTAFIN_PROVIDER_SPINE_SCALAR_I8_V1
               : DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1);
    produced.rank = auxiliary ? 1u
                              : static_cast<std::uint32_t>(slot.shape.size());
    produced.element_count = elements;
    if (auxiliary) {
      produced.shape[0] = static_cast<std::uint64_t>(tensor.size(0));
    } else {
      for (std::size_t index = 0; index < slot.shape.size(); ++index) {
        produced.shape[index] =
            static_cast<std::uint64_t>(slot.shape[index]);
      }
    }
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_kda_cache_create_v1(
    const DeltafinProviderKdaCacheCreateV1* request,
    DeltafinProviderKdaCacheReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider KDA cache-create request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider KDA cache-create request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        !all_zero(request->reserved) || !is_kda_layer(request->layer_index)) {
      throw std::invalid_argument(
          "provider KDA cache-create has invalid report/layer/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    auto state = (session->flags & kSyntheticKdaFlag) != 0
        ? deltafin::provider_internal::zero_small_kda_canary_state(
              session->selected.device)
        : deltafin::provider_internal::zero_k3_kda_state(
              session->selected.device);
    const std::uint64_t convolution_elements =
        deltafin::provider_internal::kda_state_conv_elements(state);
    const std::uint64_t recurrent_elements =
        deltafin::provider_internal::kda_state_recurrent_elements(state);
    const auto handle = session->allocate_resource();
    const auto [ignored, inserted] = session->kda_caches.emplace(
        handle, KdaCacheSlot{std::move(state), 0, request->layer_index});
    static_cast<void>(ignored);
    if (!inserted) {
      throw std::runtime_error("provider KDA cache handle collision");
    }

    DeltafinProviderKdaCacheReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.cache = handle;
    produced.layer_index = request->layer_index;
    produced.version = 0;
    produced.convolution_elements = convolution_elements;
    produced.recurrent_elements = recurrent_elements;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_kda_cache_release_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider KDA cache-release request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    for (const auto& [ignored, ticket] : session->kda_tickets) {
      static_cast<void>(ignored);
      if (ticket.cache == request->resource) {
        throw std::runtime_error(
            "provider KDA cache still has a live decode ticket");
      }
    }
    if (session->kda_caches.erase(request->resource) != 1) {
      throw std::invalid_argument(
          "provider KDA cache handle is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_kda_decode_v1(
    const DeltafinProviderKdaDecodeRequestV1* request,
    DeltafinProviderKdaDecodeReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider KDA decode request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider KDA decode request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        !all_zero(request->reserved) || request->spine_generation == 0 ||
        !is_kda_layer(request->layer_index)) {
      throw std::invalid_argument(
          "provider KDA decode has invalid report/layer/generation/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const SpineLayerSlot& spine = require_bound_spine_layer(
        *session, request->layer_index, request->spine_generation);
    const auto hidden_found = session->tensors.find(request->hidden);
    if (hidden_found == session->tensors.end()) {
      throw std::invalid_argument(
          "provider KDA hidden tensor handle is stale or unknown");
    }
    const auto cache_found = session->kda_caches.find(request->cache);
    if (cache_found == session->kda_caches.end()) {
      throw std::invalid_argument(
          "provider KDA cache handle is stale or unknown");
    }
    if (cache_found->second.layer_index != request->layer_index) {
      throw std::invalid_argument(
          "provider KDA cache belongs to a different layer");
    }
    const auto weights = kda_weights_from_spine(spine);
    const bool synthetic_canary =
        (session->flags & kSyntheticKdaFlag) != 0;
    auto decoded = deltafin::provider_internal::kda_decode_one(
        hidden_found->second, weights, cache_found->second.state,
        !synthetic_canary);
    if (decoded.output.dim() != 2 || decoded.output.size(0) != 1 ||
        decoded.output.size(1) != hidden_found->second.size(1) ||
        decoded.output.scalar_type() != at::kFloat) {
      throw std::runtime_error(
          "provider KDA tape produced an invalid output contract");
    }

    const auto output_handle = session->allocate_resource();
    const auto [output_ignored, output_inserted] =
        session->tensors.emplace(output_handle, decoded.output);
    static_cast<void>(output_ignored);
    if (!output_inserted) {
      throw std::runtime_error("provider KDA output handle collision");
    }
    DeltafinProviderKdaTicketHandleV1 ticket_handle = 0;
    try {
      ticket_handle = session->allocate_resource();
      const auto [ticket_ignored, ticket_inserted] =
          session->kda_tickets.emplace(
              ticket_handle,
              KdaTicketSlot{std::move(decoded.next_state), request->cache,
                            cache_found->second.version,
                            request->layer_index,
                            request->spine_generation});
      static_cast<void>(ticket_ignored);
      if (!ticket_inserted) {
        throw std::runtime_error("provider KDA ticket handle collision");
      }
    } catch (...) {
      session->tensors.erase(output_handle);
      throw;
    }

    DeltafinProviderKdaDecodeReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.output = output_handle;
    produced.ticket = ticket_handle;
    produced.cache_version = cache_found->second.version;
    produced.spine_generation = request->spine_generation;
    produced.rows = static_cast<std::uint64_t>(decoded.output.size(0));
    produced.columns = static_cast<std::uint64_t>(decoded.output.size(1));
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_kda_commit_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderKdaCommitReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider KDA commit request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider KDA commit report does not match provider ABI v1");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto ticket_found = session->kda_tickets.find(request->resource);
    if (ticket_found == session->kda_tickets.end()) {
      throw std::invalid_argument(
          "provider KDA ticket handle is stale or unknown");
    }
    const auto cache_found =
        session->kda_caches.find(ticket_found->second.cache);
    if (cache_found == session->kda_caches.end()) {
      throw std::runtime_error("provider KDA ticket's cache no longer exists");
    }
    if (cache_found->second.version !=
        ticket_found->second.expected_cache_version) {
      throw std::runtime_error(
          "provider KDA cache advanced after decode; stale ticket refused");
    }
    if (cache_found->second.version ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::runtime_error("provider KDA cache version is exhausted");
    }

    // All tensor work was staged by decode.  These moves only exchange
    // provider-owned handles and cannot schedule device work.  Version and
    // ticket publication happen after the state handoff.
    cache_found->second.state = std::move(ticket_found->second.next_state);
    ++cache_found->second.version;
    const std::uint64_t committed_version = cache_found->second.version;
    const std::uint32_t layer_index = cache_found->second.layer_index;
    const DeltafinProviderKdaCacheHandleV1 cache_handle =
        ticket_found->second.cache;
    session->kda_tickets.erase(ticket_found);

    DeltafinProviderKdaCommitReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.cache = cache_handle;
    produced.committed_version = committed_version;
    produced.layer_index = layer_index;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_kda_ticket_release_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider KDA ticket-release request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->kda_tickets.erase(request->resource) != 1) {
      throw std::invalid_argument(
          "provider KDA ticket handle is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_mla_cache_create_v1(
    const DeltafinProviderMlaCacheCreateV1* request,
    DeltafinProviderMlaCacheReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider MLA cache-create request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider MLA cache-create request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        !all_zero(request->reserved) || !is_mla_layer(request->layer_index)) {
      throw std::invalid_argument(
          "provider MLA cache-create has invalid report/layer/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if ((session->flags & (kSyntheticFlag | kSyntheticKdaFlag)) != 0) {
      throw std::invalid_argument(
          "provider MLA cache requires a target or synthetic-MLA session");
    }
    const auto shape = (session->flags & kSyntheticMlaFlag) != 0
        ? deltafin::provider_internal::MlaShape::small_canary()
        : deltafin::provider_internal::MlaShape::k3();
    auto cache = std::make_unique<deltafin::provider_internal::MlaCache>(
        shape,
        deltafin::provider_internal::MlaCacheRepresentation::ExpandedExact);
    const auto handle = session->allocate_resource();
    const auto [ignored, inserted] = session->mla_caches.emplace(
        handle, MlaCacheSlot{std::move(cache), request->layer_index});
    static_cast<void>(ignored);
    if (!inserted) {
      throw std::runtime_error("provider MLA cache handle collision");
    }

    DeltafinProviderMlaCacheReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.cache = handle;
    produced.layer_index = request->layer_index;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_mla_cache_release_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider MLA cache-release request");
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    for (const auto& [ignored, ticket] : session->mla_tickets) {
      static_cast<void>(ignored);
      if (ticket->cache == request->resource) {
        throw std::runtime_error(
            "provider MLA cache still has a live decode ticket");
      }
    }
    if (session->mla_caches.erase(request->resource) != 1) {
      throw std::invalid_argument(
          "provider MLA cache handle is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_mla_decode_v1(
    const DeltafinProviderMlaDecodeRequestV1* request,
    DeltafinProviderMlaDecodeReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider MLA decode request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider MLA decode request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        !all_zero(request->reserved) || request->spine_generation == 0 ||
        !is_mla_layer(request->layer_index)) {
      throw std::invalid_argument(
          "provider MLA decode has invalid report/layer/generation/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const SpineLayerSlot& spine = require_bound_spine_layer(
        *session, request->layer_index, request->spine_generation);
    const auto hidden_found = session->tensors.find(request->hidden);
    if (hidden_found == session->tensors.end()) {
      throw std::invalid_argument(
          "provider MLA hidden tensor handle is stale or unknown");
    }
    const auto cache_found = session->mla_caches.find(request->cache);
    if (cache_found == session->mla_caches.end() ||
        cache_found->second.state == nullptr) {
      throw std::invalid_argument(
          "provider MLA cache handle is stale or unknown");
    }
    if (cache_found->second.layer_index != request->layer_index) {
      throw std::invalid_argument(
          "provider MLA cache belongs to a different layer");
    }
    auto weights = mla_weights_from_spine(spine);
    const auto* bundle = spine.mla_input_bundle.get();
    at::Tensor hidden = hidden_found->second.view(
        {1, 1, hidden_found->second.size(1)});
    auto prepared = (session->flags & kSyntheticMlaFlag) != 0
        ? deltafin::provider_internal::prepare_mla_decode(
              hidden, weights, *cache_found->second.state, true, bundle)
        : deltafin::provider_internal::prepare_k3_mla_decode(
              hidden, weights, *cache_found->second.state, bundle);
    if (!prepared.output.defined() || prepared.output.dim() != 3 ||
        prepared.output.size(0) != 1 || prepared.output.size(1) != 1 ||
        prepared.output.size(2) != hidden.size(2)) {
      deltafin::provider_internal::cancel_mla_decode(
          *cache_found->second.state, prepared);
      throw std::runtime_error(
          "provider MLA tape produced an invalid output contract");
    }

    const std::uint64_t cache_version = prepared.expected_version;
    const std::uint64_t proposed_length =
        static_cast<std::uint64_t>(prepared.next_length);
    const std::uint64_t proposed_capacity =
        static_cast<std::uint64_t>(prepared.next_capacity);
    const std::uint64_t bundle_rows = bundle == nullptr
        ? 0
        : static_cast<std::uint64_t>(bundle->projection.data.size(0));
    const std::uint64_t output_columns =
        static_cast<std::uint64_t>(hidden.size(2));

    std::uint64_t output_handle = 0;
    std::uint64_t ticket_handle = 0;
    std::unique_ptr<MlaTicketSlot> staged_ticket;
    bool output_inserted = false;
    try {
      // Every potentially throwing allocation remains inside this rollback
      // region. The moved-from prepared object is explicitly marked finalized,
      // so exactly one of it or staged_ticket owns the pending cache nonce.
      output_handle = session->allocate_resource();
      staged_ticket = std::make_unique<MlaTicketSlot>(
          std::move(prepared), request->cache, request->layer_index,
          request->spine_generation);
      const auto [ignored_output, inserted_output] = session->tensors.try_emplace(
          output_handle,
          staged_ticket->prepared.output.view(
              {1, hidden.size(2)}));
      static_cast<void>(ignored_output);
      if (!inserted_output) {
        throw std::runtime_error("provider MLA output handle collision");
      }
      output_inserted = true;
      // Count the published output before admitting the second resource so a
      // decode cannot step over the session's bounded live-resource limit.
      ticket_handle = session->allocate_resource();
      const auto [ignored_ticket, inserted_ticket] =
          session->mla_tickets.try_emplace(ticket_handle,
                                           std::move(staged_ticket));
      static_cast<void>(ignored_ticket);
      if (!inserted_ticket) {
        throw std::runtime_error("provider MLA ticket handle collision");
      }
    } catch (...) {
      if (staged_ticket != nullptr) {
        deltafin::provider_internal::cancel_mla_decode(
            *cache_found->second.state, staged_ticket->prepared);
      } else if (!prepared.finalized) {
        deltafin::provider_internal::cancel_mla_decode(
            *cache_found->second.state, prepared);
      }
      if (output_inserted) {
        session->tensors.erase(output_handle);
      }
      throw;
    }

    DeltafinProviderMlaDecodeReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.output = output_handle;
    produced.ticket = ticket_handle;
    produced.cache_version = cache_version;
    produced.spine_generation = request->spine_generation;
    produced.rows = 1;
    produced.columns = output_columns;
    produced.proposed_length = proposed_length;
    produced.proposed_capacity = proposed_capacity;
    produced.input_bundle_rows = bundle_rows;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_mla_commit_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderMlaCommitReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider MLA commit request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider MLA commit report does not match provider ABI v1");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto ticket_found = session->mla_tickets.find(request->resource);
    if (ticket_found == session->mla_tickets.end()) {
      throw std::invalid_argument(
          "provider MLA ticket handle is stale or unknown");
    }
    MlaTicketSlot& ticket = *ticket_found->second;
    const auto cache_found = session->mla_caches.find(ticket.cache);
    if (cache_found == session->mla_caches.end() ||
        cache_found->second.state == nullptr) {
      throw std::runtime_error("provider MLA ticket's cache no longer exists");
    }
    deltafin::provider_internal::commit_mla_decode(
        *cache_found->second.state, ticket.prepared);
    const std::uint64_t committed_version =
        cache_found->second.state->version();
    const std::uint64_t committed_length = static_cast<std::uint64_t>(
        cache_found->second.state->length());
    const std::uint64_t capacity = static_cast<std::uint64_t>(
        cache_found->second.state->capacity());
    const DeltafinProviderMlaCacheHandleV1 cache_handle = ticket.cache;
    const std::uint32_t layer_index = ticket.layer_index;
    session->mla_tickets.erase(ticket_found);

    DeltafinProviderMlaCommitReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.cache = cache_handle;
    produced.committed_version = committed_version;
    produced.layer_index = layer_index;
    produced.committed_length = committed_length;
    produced.capacity = capacity;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_mla_ticket_release_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider MLA ticket-release request");
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto ticket_found = session->mla_tickets.find(request->resource);
    if (ticket_found == session->mla_tickets.end()) {
      throw std::invalid_argument(
          "provider MLA ticket handle is stale or unknown");
    }
    MlaTicketSlot& ticket = *ticket_found->second;
    const auto cache_found = session->mla_caches.find(ticket.cache);
    if (cache_found == session->mla_caches.end() ||
        cache_found->second.state == nullptr) {
      throw std::runtime_error("provider MLA ticket's cache no longer exists");
    }
    deltafin::provider_internal::cancel_mla_decode(
        *cache_found->second.state, ticket.prepared);
    session->mla_tickets.erase(ticket_found);
  });
}

namespace {

void require_target_session(const Session& session) {
  if (session.flags != 0) {
    throw std::invalid_argument(
        "target transaction ABI requires a real session with flags=0");
  }
}

void publish_target_begin(
    Session& session, at::Tensor hidden,
    DeltafinProviderTargetBeginReportV1* report) {
  require_target_session(session);
  if (session.selected.device.is_cuda()) {
    // One check per authoritative transaction avoids a global-context lookup
    // on every layer while still rejecting process-wide precision changes
    // made after session creation.
    deltafin::provider_internal::enforce_authoritative_cuda_fp32_precision();
  }
  if (session.target_tail == nullptr ||
      session.target_global_groups[0] == nullptr ||
      session.target_global_groups[1] == nullptr) {
    throw std::runtime_error(
        "target position requires immutable global groups 1 and 2");
  }
  if (session.target_position != nullptr ||
      session.target_position_handle != 0 ||
      session.target_sequence != nullptr ||
      session.target_sequence_handle != 0) {
    throw std::runtime_error(
        "provider session already owns a live target transaction");
  }
  if (!hidden.defined() || hidden.scalar_type() != at::kFloat ||
      hidden.device() != session.selected.device || !hidden.is_contiguous() ||
      hidden.sizes() != at::IntArrayRef({1, 7168})) {
    throw std::invalid_argument(
        "target begin hidden must be exact contiguous fp32 [1,7168]");
  }

  std::unique_ptr<TargetSessionCacheStore> staged_cache_store;
  TargetSessionCacheStore* cache_store = session.target_cache_store.get();
  if (cache_store == nullptr) {
    staged_cache_store = make_target_cache_store(session.selected.device);
    cache_store = staged_cache_store.get();
  }
  const auto bindings = deltafin::provider_internal::TargetPositionBindings{
      .contract = deltafin::provider_internal::TargetTapeContract::ExactK3,
      .caches = cache_store->bindings,
      .tail = session.target_tail.get(),
      .pilot_routers = session.target_pilot_enabled
          ? &session.target_pilot_routers
          : nullptr};
  auto position =
      std::make_unique<deltafin::provider_internal::TargetPositionTape>(
          bindings, std::move(hidden));
  const auto handle = session.allocate_resource();

  // These pointer moves are the sole publication point. TargetPositionTape's
  // copied cache pointers remain stable because the cache store itself is a
  // separately allocated object whose address does not change here.
  if (staged_cache_store != nullptr) {
    session.target_cache_store = std::move(staged_cache_store);
  }
  session.target_position = std::move(position);
  session.target_position_handle = handle;

  DeltafinProviderTargetBeginReportV1 produced = {};
  produced.struct_size = sizeof(produced);
  produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  produced.position = handle;
  produced.next_layer = 0;
  produced.state = DELTAFIN_PROVIDER_TARGET_ACTIVE_V1;
  produced.kda_cache_count =
      deltafin::provider_internal::kTargetKdaLayerCount;
  produced.mla_cache_count =
      deltafin::provider_internal::kTargetMlaLayerCount;
  *report = produced;
}

void require_live_target_position(
    const Session& session,
    const DeltafinProviderTargetPositionHandleV1 handle) {
  require_target_session(session);
  if (handle == 0 || session.target_position == nullptr ||
      session.target_position_handle != handle) {
    throw std::invalid_argument(
        "target position handle is stale or unknown");
  }
}

void publish_target_sequence_begin(
    Session& session, at::Tensor hidden_rows,
    const deltafin::provider_internal::TargetSequenceMode mode,
    const bool capture_dspark_rows, const bool full_commit_only,
    DeltafinProviderTargetSequenceBeginReportV1* report) {
  require_target_session(session);
  if (session.selected.device.is_cuda()) {
    deltafin::provider_internal::enforce_authoritative_cuda_fp32_precision();
  }
  if (session.target_tail == nullptr ||
      session.target_global_groups[0] == nullptr ||
      session.target_global_groups[1] == nullptr) {
    throw std::runtime_error(
        "target sequence requires immutable global groups 1 and 2");
  }
  if (session.target_position != nullptr ||
      session.target_position_handle != 0 ||
      session.target_sequence != nullptr ||
      session.target_sequence_handle != 0) {
    throw std::runtime_error(
        "provider session already owns a live target transaction");
  }
  if (!hidden_rows.defined() || hidden_rows.scalar_type() != at::kFloat ||
      hidden_rows.device() != session.selected.device ||
      !hidden_rows.is_contiguous() || hidden_rows.dim() != 2 ||
      hidden_rows.size(0) < 1 ||
      hidden_rows.size(0) >
          DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1 ||
      hidden_rows.size(1) != 7168) {
    throw std::invalid_argument(
        "target sequence begin hidden must be contiguous fp32 [1..64,7168]");
  }

  std::unique_ptr<TargetSessionCacheStore> staged_cache_store;
  TargetSessionCacheStore* cache_store = session.target_cache_store.get();
  if (cache_store == nullptr) {
    staged_cache_store = make_target_cache_store(session.selected.device);
    cache_store = staged_cache_store.get();
  }
  const auto bindings = deltafin::provider_internal::TargetPositionBindings{
      .contract = deltafin::provider_internal::TargetTapeContract::ExactK3,
      .caches = cache_store->bindings,
      .tail = session.target_tail.get(),
      .pilot_routers = session.target_pilot_enabled
          ? &session.target_pilot_routers
          : nullptr};
  auto sequence =
      std::make_unique<deltafin::provider_internal::TargetSequenceTape>(
          bindings, std::move(hidden_rows), mode, capture_dspark_rows,
          full_commit_only);
  const auto handle = session.allocate_resource();
  const std::uint32_t positions =
      static_cast<std::uint32_t>(sequence->position_count());

  if (staged_cache_store != nullptr) {
    session.target_cache_store = std::move(staged_cache_store);
  }
  session.target_sequence = std::move(sequence);
  session.target_sequence_handle = handle;

  DeltafinProviderTargetSequenceBeginReportV1 produced = {};
  produced.struct_size = sizeof(produced);
  produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  produced.sequence = handle;
  produced.positions = positions;
  produced.mode = target_sequence_mode_value(mode);
  produced.next_layer = 0;
  produced.state = DELTAFIN_PROVIDER_TARGET_SEQUENCE_ACTIVE_V1;
  produced.kda_cache_count =
      deltafin::provider_internal::kTargetKdaLayerCount;
  produced.mla_cache_count =
      deltafin::provider_internal::kTargetMlaLayerCount;
  *report = produced;
}

void require_live_target_sequence(
    const Session& session,
    const DeltafinProviderTargetSequenceHandleV1 handle) {
  require_target_session(session);
  if (handle == 0 || session.target_sequence == nullptr ||
      session.target_sequence_handle != handle) {
    throw std::invalid_argument(
        "target sequence handle is stale or unknown");
  }
}

void qualify_kda_weights(Session& session,
                         const deltafin::provider_internal::KdaWeights& w) {
  const std::array<const deltafin::provider_internal::KdaProjection*, 8>
      projections{&w.query_projection, &w.key_projection,
                  &w.value_projection, &w.recurrent_gate_projection,
                  &w.feature_a_projection, &w.feature_b_projection,
                  &w.beta_projection, &w.output_projection};
  const bool packed = w.query_projection.scale.defined();
  if (std::any_of(projections.begin(), projections.end(),
                  [packed](const auto* projection) {
                    return projection->scale.defined() != packed;
                  })) {
    throw std::invalid_argument(
        "KDA layer mixes original-BF16 and row-int8 projections");
  }
  if (!packed) {
    return;
  }
  const auto qualify = [&](const deltafin::provider_internal::KdaProjection& p,
                           const char* name) {
    require_target_packed_shape(
        session,
        deltafin::provider_internal::MoeRowInt8Matrix{
            p.weight, p.scale, at::Tensor(), {}},
        name);
  };
  qualify(w.query_projection, "KDA query projection");
  qualify(w.key_projection, "KDA key projection");
  qualify(w.value_projection, "KDA value projection");
  qualify(w.recurrent_gate_projection, "KDA recurrent-gate projection");
  qualify(w.feature_a_projection, "KDA feature-a projection");
  qualify(w.feature_b_projection, "KDA feature-b projection");
  qualify(w.beta_projection, "KDA beta projection");
  qualify(w.output_projection, "KDA output projection");
}

void qualify_mla_weights(Session& session,
                         const deltafin::provider_internal::MlaWeights& w) {
  const std::array<const deltafin::provider_internal::MlaLinearWeight*, 6>
      projections{&w.query_a, &w.query_b, &w.key_value_a,
                  &w.key_value_b, &w.output_gate, &w.output};
  const bool packed = w.query_a.encoding ==
      deltafin::provider_internal::MlaLinearEncoding::RowI8F32Scale;
  if (std::any_of(projections.begin(), projections.end(),
                  [packed](const auto* projection) {
                    const bool projection_packed = projection->encoding ==
                        deltafin::provider_internal::MlaLinearEncoding::
                            RowI8F32Scale;
                    return projection_packed != packed;
                  })) {
    throw std::invalid_argument(
        "MLA layer mixes original-BF16 and row-int8 projections");
  }
  if (!packed) {
    return;
  }
  const auto qualify = [&](const deltafin::provider_internal::MlaLinearWeight& p,
                           const char* name) {
    require_target_packed_shape(
        session,
        deltafin::provider_internal::MoeRowInt8Matrix{
            p.data, p.row_scale, at::Tensor(), {}},
        name);
  };
  qualify(w.query_a, "MLA query-a projection");
  qualify(w.query_b, "MLA query-b projection");
  qualify(w.key_value_a, "MLA key/value-a projection");
  qualify(w.key_value_b, "MLA key/value-b projection");
  qualify(w.output_gate, "MLA output-gate projection");
  qualify(w.output, "MLA output projection");
}

deltafin::provider_internal::MoeExpertLayout decode_expert_layout(
    const std::uint32_t value) {
  using deltafin::provider_internal::MoeExpertLayout;
  switch (value) {
    case DELTAFIN_PROVIDER_EXPERT_LAYOUT_RAW_V1:
      return MoeExpertLayout::RawV1;
    case DELTAFIN_PROVIDER_EXPERT_LAYOUT_SCALE4_V2:
      return MoeExpertLayout::Scale4V2;
    default:
      throw std::invalid_argument("target expert storage layout is unknown");
  }
}

std::uint64_t required_k3_expert_span(
    const deltafin::provider_internal::MoeExpertLayout layout) {
  using deltafin::provider_internal::MoeExpertLayout;
  switch (layout) {
    case MoeExpertLayout::RawV1:
      return K3_RAW_V1_EXPERT_SPAN;
    case MoeExpertLayout::Scale4V2:
      return K3_SCALE4_V2_EXPERT_SPAN;
  }
  throw std::invalid_argument("target expert storage layout is unknown");
}

template <typename Request>
deltafin::provider_internal::MoeRunOptions target_moe_options(
    const Request& request, Session& session) {
  using deltafin::provider_internal::MoeExpertBackend;
  const at::Device& device = session.selected.device;
  MoeExpertBackend backend;
  switch (request.expert_backend) {
    case DELTAFIN_PROVIDER_TARGET_EXPERT_AUTO_V1:
      backend = device.is_cuda() && session.cuda_expert_cache != nullptr &&
              session.cuda_expert_cache->available()
          ? MoeExpertBackend::CudaMxfp4
          : MoeExpertBackend::Auto;
      break;
    case DELTAFIN_PROVIDER_TARGET_EXPERT_CPU_V1:
      backend = MoeExpertBackend::CpuMxfp4;
      break;
    case DELTAFIN_PROVIDER_TARGET_EXPERT_METAL_V1:
      if (!device.is_mps()) {
        throw std::invalid_argument(
            "target Metal experts require the selected MPS provider");
      }
      backend = MoeExpertBackend::MetalMxfp4;
      break;
    case DELTAFIN_PROVIDER_TARGET_EXPERT_CUDA_V1:
      if (!device.is_cuda()) {
        throw std::invalid_argument(
            "target CUDA experts require the selected CUDA provider");
      }
      if (session.cuda_expert_cache == nullptr ||
          !session.cuda_expert_cache->available()) {
        const std::string detail = session.cuda_expert_cache == nullptr
            ? "CUDA MXFP4 was not compiled into this provider"
            : session.cuda_expert_cache->detail();
        throw std::runtime_error(
            "target CUDA expert capability gate failed: " + detail);
      }
      backend = MoeExpertBackend::CudaMxfp4;
      break;
    default:
      throw std::invalid_argument("target expert backend is unknown");
  }
  if (request.cpu_threads == 0 || request.cpu_threads > 1024) {
    throw std::invalid_argument("target expert cpu_threads must be in 1..1024");
  }
  const bool retain_metal_wrappers =
      (request.flags &
       DELTAFIN_PROVIDER_TARGET_EXPERT_RETAIN_METAL_WRAPPERS_V1) != 0;
  if (retain_metal_wrappers &&
      (!device.is_mps() ||
       (backend != MoeExpertBackend::MetalMxfp4 &&
        backend != MoeExpertBackend::Auto))) {
    throw std::invalid_argument(
        "retained Metal expert wrappers require the selected Metal/MPS backend");
  }
  const auto layout = decode_expert_layout(request.expert_layout);
  if (layout == deltafin::provider_internal::MoeExpertLayout::Scale4V2 &&
      (!device.is_mps() ||
       (backend != MoeExpertBackend::MetalMxfp4 &&
        backend != MoeExpertBackend::Auto))) {
    throw std::invalid_argument(
        "target scale4-v2 experts require the selected Metal/MPS backend");
  }
  if ((request.metal_shader_path == nullptr) !=
      (request.metal_shader_path_length == 0)) {
    throw std::invalid_argument(
        "target Metal shader path pointer/length is not canonical");
  }
  if (request.metal_shader_path_length > 4096 ||
      request.metal_shader_path_length > SIZE_MAX) {
    throw std::invalid_argument("target Metal shader path is too long");
  }
  std::string shader_path;
  if (request.metal_shader_path != nullptr) {
    shader_path.assign(request.metal_shader_path,
                       static_cast<std::size_t>(
                           request.metal_shader_path_length));
    if (shader_path.find('\0') != std::string::npos) {
      throw std::invalid_argument(
          "target Metal shader path contains an embedded NUL");
    }
  }
  return deltafin::provider_internal::MoeRunOptions{
      .expert_backend = backend,
      .cpu_threads = request.cpu_threads,
      .metal_shader_path = std::move(shader_path),
      .cuda_cache = session.cuda_expert_cache.get(),
      .cuda_plan = 0,
      .cuda_auto_fallback =
          request.expert_backend == DELTAFIN_PROVIDER_TARGET_EXPERT_AUTO_V1,
      .metal_retain_expert_wrappers = retain_metal_wrappers};
}

deltafin::provider_internal::MoeRunOptions target_moe_plan_options(
    const DeltafinProviderTargetSequencePlanExpertsRequestV1& request,
    Session& session) {
  if (!session.selected.device.is_cuda()) {
    throw std::invalid_argument(
        "target expert residency planning requires a selected CUDA provider");
  }
  if (request.expert_backend != DELTAFIN_PROVIDER_TARGET_EXPERT_AUTO_V1 &&
      request.expert_backend != DELTAFIN_PROVIDER_TARGET_EXPERT_CUDA_V1) {
    throw std::invalid_argument(
        "target expert residency planning accepts only Auto or CUDA");
  }

  // The planning ABI is intentionally raw-v1-only. Reuse the ordinary
  // backend/path validator through a complete local request rather than
  // growing a second subtly different backend selection implementation.
  DeltafinProviderTargetSequenceFinishExpertsRequestV1 proxy = {};
  proxy.expert_backend = request.expert_backend;
  proxy.cpu_threads = request.cpu_threads;
  proxy.expert_layout = DELTAFIN_PROVIDER_EXPERT_LAYOUT_RAW_V1;
  proxy.metal_shader_path = request.metal_shader_path;
  proxy.metal_shader_path_length = request.metal_shader_path_length;
  auto options = target_moe_options(proxy, session);
  if (options.expert_backend ==
      deltafin::provider_internal::MoeExpertBackend::Auto) {
    // On a selected CUDA device, Auto remains unresolved only when the CUDA
    // adapter is cleanly unavailable. Freeze CPU before any expert read; a
    // backend can never change after the returned miss list is authenticated.
    options.expert_backend =
        deltafin::provider_internal::MoeExpertBackend::CpuMxfp4;
  }
  if (options.expert_backend !=
          deltafin::provider_internal::MoeExpertBackend::CpuMxfp4 &&
      options.expert_backend !=
          deltafin::provider_internal::MoeExpertBackend::CudaMxfp4) {
    throw std::logic_error(
        "target CUDA expert planner resolved an impossible backend");
  }
  return options;
}

std::uint32_t target_moe_backend_value(
    const deltafin::provider_internal::MoeExpertBackend backend) {
  using deltafin::provider_internal::MoeExpertBackend;
  switch (backend) {
    case MoeExpertBackend::CpuMxfp4:
      return DELTAFIN_PROVIDER_TARGET_EXPERT_CPU_V1;
    case MoeExpertBackend::CudaMxfp4:
      return DELTAFIN_PROVIDER_TARGET_EXPERT_CUDA_V1;
    case MoeExpertBackend::Auto:
    case MoeExpertBackend::MetalMxfp4:
      break;
  }
  throw std::logic_error(
      "target CUDA expert planner did not freeze CPU or CUDA");
}

std::vector<std::uint16_t> target_sequence_route_union(
    const Session& session, const std::uint32_t layer_index,
    const std::uint64_t spine_generation, const std::uint32_t first_row,
    const std::uint32_t row_count, const std::size_t maximum_experts) {
  const auto mailbox = session.target_sequence->expert_mailbox();
  if (mailbox.layer_index != layer_index ||
      mailbox.spine_generation != spine_generation ||
      mailbox.row_count != session.target_sequence->position_count()) {
    throw std::invalid_argument(
        "target expert plan does not match the active route mailbox");
  }

  std::size_t current_row = 0;
  while (current_row < mailbox.row_count &&
         !mailbox.rows[current_row].routed_input.defined()) {
    ++current_row;
  }
  if (current_row == mailbox.row_count || first_row != current_row ||
      row_count > mailbox.row_count - current_row) {
    throw std::invalid_argument(
        "target expert plan must begin at the current canonical row cursor");
  }

  std::set<std::uint16_t> routed;
  for (std::size_t row = first_row;
       row < static_cast<std::size_t>(first_row) + row_count; ++row) {
    if (mailbox.rows[row].row_index != row ||
        !mailbox.rows[row].routed_input.defined()) {
      throw std::logic_error(
          "target expert plan found an incomplete provider route row");
    }
    routed.insert(mailbox.rows[row].route.expert_ids.begin(),
                  mailbox.rows[row].route.expert_ids.end());
  }
  if (routed.empty() || routed.size() > maximum_experts) {
    throw std::invalid_argument(
        "target expert route union exceeds the selected ABI bound");
  }
  return {routed.begin(), routed.end()};
}

}  // namespace

extern "C" int32_t deltafin_provider_bind_target_globals_v1(
    const DeltafinProviderBindTargetGlobalsRequestV1* request,
    DeltafinProviderBindTargetGlobalsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-global bind request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider target-global bind request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider target-global bind has invalid report/flags/reserved fields");
    }
    if (request->group != kTargetGlobalTailGroup &&
        request->group != kTargetGlobalHeadGroup) {
      throw std::invalid_argument("provider target-global group is unknown");
    }
    validate_spine_buffer(request->quantized, request->quantized_length,
                          "target-global quantized");
    validate_spine_buffer(request->scales, request->scales_length,
                          "target-global scales");
    validate_spine_buffer(request->other, request->other_length,
                          "target-global other");

    DeltafinProviderBindSpineLayerRequestV1 payload = {};
    payload.struct_size = sizeof(payload);
    payload.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    payload.session = request->session;
    payload.layer_index = 0;
    payload.generation = request->group;
    payload.descriptors = request->descriptors;
    payload.descriptor_count = request->descriptor_count;
    payload.quantized = request->quantized;
    payload.quantized_length = request->quantized_length;
    payload.scales = request->scales;
    payload.scales_length = request->scales_length;
    payload.other = request->other;
    payload.other_length = request->other_length;
    const std::uint32_t first_slot = request->group == kTargetGlobalTailGroup
        ? kFinalNormSlot
        : kLanguageModelHeadSlot;
    const std::uint32_t last_slot = request->group == kTargetGlobalTailGroup
        ? kOutputResidualProjectionSlot
        : kLanguageModelHeadSlot;
    auto descriptors =
        validate_spine_descriptors(payload, first_slot, last_slot);
    require_target_global_roster(request->group, descriptors);

    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_target_session(*session);
    if (session->target_position != nullptr ||
        session->target_position_handle != 0 ||
        session->target_sequence != nullptr ||
        session->target_sequence_handle != 0) {
      throw std::runtime_error(
          "target globals cannot bind during a live target transaction");
    }
    const std::size_t group_index =
        static_cast<std::size_t>(request->group - 1);
    if (session->target_global_groups[group_index] != nullptr) {
      throw std::invalid_argument(
          "target global group is immutable and already bound");
    }

    auto staged = upload_validated_spine_payload(
        payload, descriptors, *session, 0, false, false, false);
    std::unique_ptr<deltafin::provider_internal::TargetTailWeights>
        staged_tail;
    const SpineLayerSlot* tail_group = request->group == kTargetGlobalTailGroup
        ? staged.get()
        : session->target_global_groups[0].get();
    const SpineLayerSlot* head_group = request->group == kTargetGlobalHeadGroup
        ? staged.get()
        : session->target_global_groups[1].get();
    if (tail_group != nullptr && head_group != nullptr) {
      staged_tail = make_target_tail(*session, *tail_group, *head_group);
    }

    DeltafinProviderBindTargetGlobalsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.group = request->group;
    produced.tensor_count = staged->tensor_count;
    produced.quantized_tensor_count = staged->quantized_tensor_count;
    produced.raw_tensor_count = staged->raw_tensor_count;
    produced.quantized_bytes = staged->quantized_bytes;
    produced.scales_bytes = staged->scales_bytes;
    produced.other_bytes = staged->other_bytes;
    produced.resident_storage_bytes =
        staged->binding_stats.resident_storage_bytes;
    produced.groups_ready = staged_tail == nullptr ? 1u : 2u;

    session->target_global_groups[group_index] = std::move(staged);
    if (staged_tail != nullptr) {
      session->target_tail = std::move(staged_tail);
    }
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_begin_v1(
    const DeltafinProviderTargetBeginRequestV1* request,
    DeltafinProviderTargetBeginReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider target begin request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider target begin request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        request->reserved0 != 0 || !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider target begin has invalid report/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto found = session->tensors.find(request->hidden);
    if (found == session->tensors.end()) {
      throw std::invalid_argument(
          "provider target begin hidden tensor is stale or unknown");
    }
    // Copying an at::Tensor is a provider-owned storage handle copy. The
    // caller may release its opaque tensor immediately after this call.
    publish_target_begin(*session, found->second, report);
  });
}

extern "C" int32_t deltafin_provider_target_begin_bf16_v1(
    const DeltafinProviderTargetBeginBf16RequestV1* request,
    DeltafinProviderTargetBeginReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target BF16 begin request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target BF16 begin request");
    if (report->struct_size != sizeof(*report) || request->data == nullptr ||
        request->byte_length != 7168 * sizeof(std::uint16_t) ||
        request->flags != 0 ||
        request->reserved0 != 0 || !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider target BF16 begin has invalid row/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    at::Tensor owned_cpu = at::empty(
        {1, 7168},
        at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
    std::memcpy(owned_cpu.mutable_data_ptr(), request->data,
                static_cast<std::size_t>(request->byte_length));
    // The explicit memcpy both severs caller ownership and avoids forming a
    // potentially misaligned uint16_t pointer from Rust's byte arena.
    at::Tensor hidden = owned_cpu.to(
        at::TensorOptions().dtype(at::kFloat).device(session->selected.device));
    publish_target_begin(*session, std::move(hidden), report);
  });
}

extern "C" int32_t deltafin_provider_target_prepare_v1(
    const DeltafinProviderTargetPrepareRequestV1* request,
    DeltafinProviderTargetPrepareReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target prepare request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider target prepare request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        request->spine_generation == 0 || !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider target prepare has invalid report/generation/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_position(*session, request->position);
    if (request->layer_index != session->target_position->next_layer_index()) {
      throw std::invalid_argument(
          "target prepare layer is out of transaction order");
    }
    const SpineLayerSlot& spine = require_bound_spine_layer(
        *session, request->layer_index, request->spine_generation);
    if (spine.target_residual == nullptr) {
      throw std::invalid_argument(
          "target streamed layer lacks its complete residual roster");
    }

    const auto fp32_execution = maybe_materialize_spine_fp32(
        *session, spine, request->position);
    const SpineFp32ExecutionView* execution =
        fp32_execution.has_value() ? &*fp32_execution : nullptr;

    deltafin::provider_internal::KdaWeights kda;
    deltafin::provider_internal::MlaWeights mla;
    deltafin::provider_internal::TargetDenseWeights dense;
    deltafin::provider_internal::MoeSpineT1 moe;
    auto binding = deltafin::provider_internal::TargetLayerBinding{
        .layer_index = request->layer_index,
        .attention_kind =
            deltafin::provider_internal::target_layer_uses_mla(
                request->layer_index)
            ? deltafin::provider_internal::TargetAttentionKind::Mla
            : deltafin::provider_internal::TargetAttentionKind::Kda,
        .residual = spine.target_residual.get()};
    if (binding.attention_kind ==
        deltafin::provider_internal::TargetAttentionKind::Kda) {
      kda = kda_weights_from_spine(spine, execution);
      qualify_kda_weights(*session, kda);
      binding.kda_weights = &kda;
    } else {
      mla = mla_weights_from_spine(spine, execution);
      qualify_mla_weights(*session, mla);
      binding.mla_weights = &mla;
      binding.mla_input_bundle = execution == nullptr
          ? spine.mla_input_bundle.get()
          : nullptr;
    }
    if (request->layer_index == 0) {
      dense = target_dense_from_spine(spine, execution);
      if (dense.packed_int8_qualified) {
        require_target_packed_shape(*session, dense.gate,
                                    "dense gate projection");
        require_target_packed_shape(*session, dense.up,
                                    "dense up projection");
        require_target_packed_shape(*session, dense.down,
                                    "dense down projection");
      }
      // Grouped binding commonly made gate/up adjacent views already. Expose
      // that exact storage as one super-view; never cat/copy this ~484 MB pair.
      maybe_bundle_target_dense_zero_copy(dense);
      binding.dense = &dense;
    } else {
      moe = target_moe_from_spine(spine, execution);
      if (moe.packed_int8_qualified) {
        require_target_packed_shape(*session, moe.router, "MoE router");
        require_target_packed_shape(*session, moe.routed_down,
                                    "MoE routed-down projection");
        require_target_packed_shape(*session, moe.routed_up,
                                    "MoE routed-up projection");
        require_target_packed_shape(*session, moe.shared_gate,
                                    "MoE shared gate projection");
        require_target_packed_shape(*session, moe.shared_up,
                                    "MoE shared up projection");
        require_target_packed_shape(*session, moe.shared_down,
                                    "MoE shared down projection");
      }
      static_cast<void>(
          deltafin::provider_internal::qualify_moe_shared_gate_up(moe));
      binding.moe = &moe;
    }

    const auto prepared = session->target_position->prepare_layer(binding);
    DeltafinProviderTargetPrepareReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.position = request->position;
    produced.spine_generation = request->spine_generation;
    produced.layer_index = request->layer_index;
    produced.next_layer = session->target_position->next_layer_index();
    if (prepared.kind ==
        deltafin::provider_internal::TargetLayerPrepareKind::DenseCompleted) {
      produced.kind = DELTAFIN_PROVIDER_TARGET_DENSE_COMPLETE_V1;
    } else {
      produced.kind = DELTAFIN_PROVIDER_TARGET_EXPERTS_REQUIRED_V1;
      produced.top_k = DELTAFIN_PROVIDER_ROUTE_TOP_K_V1;
      std::copy(prepared.route.route.expert_ids.begin(),
                prepared.route.route.expert_ids.end(),
                produced.ordered_experts);
      std::copy(prepared.route.route.weight_bits.begin(),
                prepared.route.route.weight_bits.end(),
                produced.ordered_weight_bits);
    }
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_finish_experts_v1(
    const DeltafinProviderTargetFinishExpertsRequestV1* request,
    DeltafinProviderTargetFinishExpertsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target expert-finish request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target expert-finish request");
    if (report->struct_size != sizeof(*report) ||
        (request->flags &
         ~DELTAFIN_PROVIDER_TARGET_EXPERT_RETAIN_METAL_WRAPPERS_V1) != 0 ||
        !all_zero(request->reserved) ||
        request->spine_generation == 0 ||
        request->expert_count != DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 ||
        request->expert_major_bytes == nullptr) {
      throw std::invalid_argument(
          "provider target expert-finish has invalid payload/flags/reserved fields");
    }
    const auto expert_layout = decode_expert_layout(request->expert_layout);
    const std::uint64_t expert_span = required_k3_expert_span(expert_layout);
    const std::uint64_t expected_bytes =
        expert_span * DELTAFIN_PROVIDER_ROUTE_TOP_K_V1;
    if (request->expert_span_bytes != expert_span ||
        request->expert_major_length != expected_bytes ||
        request->expert_major_length > SIZE_MAX) {
      throw std::invalid_argument(
          "provider target expert-finish byte length is not 16 canonical experts");
    }
    for (std::size_t index = 0;
         index < DELTAFIN_PROVIDER_ROUTE_TOP_K_V1; ++index) {
      if (request->expert_ids[index] >= kK3Experts ||
          (index != 0 &&
           request->expert_ids[index - 1] >= request->expert_ids[index])) {
        throw std::invalid_argument(
            "provider target expert IDs are not unique canonical ascending IDs");
      }
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_position(*session, request->position);
    auto options = target_moe_options(*request, *session);
    const auto experts =
        deltafin::provider_internal::CanonicalExpertBatchT1{
            .expert_ids = std::span<const std::uint16_t>(
                request->expert_ids,
                DELTAFIN_PROVIDER_ROUTE_TOP_K_V1),
            .expert_major_bytes = std::span<const std::uint8_t>(
                request->expert_major_bytes,
                static_cast<std::size_t>(request->expert_major_length)),
            .layout = expert_layout,
            .expert_span_bytes = expert_span};
    session->target_position->finish_moe_layer(
        request->layer_index, request->spine_generation, experts, options);

    DeltafinProviderTargetFinishExpertsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.position = request->position;
    produced.completed_layer = request->layer_index;
    produced.next_layer = session->target_position->next_layer_index();
    produced.state = target_state_value(session->target_position->state());
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_finish_greedy_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetGreedyReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider target greedy request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider target greedy report does not match provider ABI v1");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_position(*session, request->resource);
    if (session->committed_target_positions == UINT64_MAX ||
        session->committed_target_generation == UINT64_MAX) {
      throw std::runtime_error("target committed-position count is exhausted");
    }
    const std::uint32_t token = session->target_position->finish_greedy();
    const auto handle = session->target_position_handle;
    ++session->committed_target_positions;
    ++session->committed_target_generation;
    session->target_position.reset();
    session->target_position_handle = 0;

    DeltafinProviderTargetGreedyReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.position = handle;
    produced.token_id = token;
    produced.state = DELTAFIN_PROVIDER_TARGET_COMMITTED_V1;
    produced.committed_positions = session->committed_target_positions;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_cancel_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider target cancel request");
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_position(*session, request->resource);
    try {
      session->target_position->cancel();
      session->target_position.reset();
      session->target_position_handle = 0;
    } catch (...) {
      // A poisoned MLA cancellation cannot be trusted for reuse. Destroy the
      // entire private cache bank after the tape while preserving globals;
      // the next begin will build a fresh transactionally owned bank.
      session->target_position.reset();
      session->target_position_handle = 0;
      session->target_cache_store.reset();
      throw;
    }
  });
}

extern "C" int32_t deltafin_provider_target_state_reset_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider target-state reset request");
    if (request->resource != 0) {
      throw std::invalid_argument(
          "provider target-state reset resource must be zero");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->flags != 0) {
      throw std::invalid_argument(
          "provider target-state reset requires a real target session");
    }
    if (session->target_position != nullptr ||
        session->target_sequence != nullptr ||
        session->target_position_handle != 0 ||
        session->target_sequence_handle != 0 ||
        session->target_state_branch != nullptr) {
      throw std::runtime_error(
          "provider target-state reset cannot race a live transaction");
    }
    if (session->committed_target_generation == UINT64_MAX) {
      throw std::runtime_error("provider target-state generation is exhausted");
    }
    // Construct the replacement before publishing it. Allocation failure
    // therefore leaves the previous committed cache bank untouched.
    auto replacement = make_target_cache_store(session->selected.device);
    session->target_cache_store = std::move(replacement);
    session->committed_target_positions = 0;
    ++session->committed_target_generation;
  });
}

extern "C" int32_t deltafin_provider_target_state_inspect_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetStateReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider target-state inspect request");
    if (request->resource != 0 || report == nullptr ||
        report->struct_size != sizeof(*report)) {
      throw std::invalid_argument("provider target-state inspect fields are invalid");
    }
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->flags != 0 || session->target_cache_store == nullptr) {
      throw std::invalid_argument(
          "provider target-state inspect requires a real initialized target session");
    }
    if (session->target_position != nullptr ||
        session->target_sequence != nullptr) {
      throw std::runtime_error(
          "provider target-state inspect cannot observe an unpublished target transaction");
    }
    require_complete_target_cache_store(*session->target_cache_store,
                                        session->committed_target_positions);
    *report = target_state_report(*session);
  });
}

extern "C" int32_t deltafin_provider_target_state_branch_begin_v1(
    const DeltafinProviderTargetStateBranchRequestV1* request,
    DeltafinProviderTargetStateReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("provider target-state branch request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "provider target-state branch request");
    if (report->struct_size != sizeof(*report) ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument("provider target-state branch fields are invalid");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->flags != 0 || session->target_cache_store == nullptr) {
      throw std::invalid_argument(
          "provider target-state branching requires a real initialized target session");
    }
    if (session->target_position != nullptr ||
        session->target_sequence != nullptr ||
        session->target_state_branch != nullptr) {
      throw std::runtime_error(
          "provider target-state branch cannot race another transaction");
    }
    if (request->expected_committed_positions !=
            session->committed_target_positions ||
        request->expected_cache_generation !=
            session->committed_target_generation) {
      throw std::invalid_argument(
          "provider target-state branch expected boundary is stale");
    }
    auto child = fork_target_cache_store(*session->target_cache_store,
                                         session->committed_target_positions);
    auto branch = std::make_unique<TargetStateBranchSlot>();
    branch->handle = session->allocate_resource();
    branch->parent_positions = session->committed_target_positions;
    branch->parent_generation = session->committed_target_generation;
    branch->parent = std::move(session->target_cache_store);
    session->target_cache_store = std::move(child);
    session->target_state_branch = std::move(branch);
    *report = target_state_report(*session);
  });
}

extern "C" int32_t deltafin_provider_target_state_branch_publish_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetStateReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider target-state branch publish request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument("provider target-state publish report is invalid");
    }
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->target_position != nullptr || session->target_sequence != nullptr ||
        session->target_state_branch == nullptr ||
        session->target_state_branch->handle != request->resource) {
      throw std::invalid_argument(
          "provider target-state publish branch is stale or not active");
    }
    require_complete_target_cache_store(*session->target_cache_store,
                                        session->committed_target_positions);
    session->target_state_branch.reset();
    *report = target_state_report(*session);
  });
}

extern "C" int32_t deltafin_provider_target_state_branch_discard_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetStateReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider target-state branch discard request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument("provider target-state discard report is invalid");
    }
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (session->target_position != nullptr || session->target_sequence != nullptr ||
        session->target_state_branch == nullptr ||
        session->target_state_branch->handle != request->resource) {
      throw std::invalid_argument(
          "provider target-state discard branch is stale or not active");
    }
    const std::uint64_t current_generation =
        session->committed_target_generation;
    const std::uint64_t parent_generation =
        session->target_state_branch->parent_generation;
    if (std::max(current_generation, parent_generation) == UINT64_MAX) {
      throw std::runtime_error("provider target-state generation is exhausted");
    }
    require_complete_target_cache_store(
        *session->target_state_branch->parent,
        session->target_state_branch->parent_positions);
    auto parent = std::move(session->target_state_branch->parent);
    const std::uint64_t positions =
        session->target_state_branch->parent_positions;
    session->target_state_branch.reset();
    session->target_cache_store = std::move(parent);
    session->committed_target_positions = positions;
    session->committed_target_generation =
        std::max(current_generation, parent_generation) + 1;
    *report = target_state_report(*session);
  });
}

extern "C" int32_t deltafin_provider_target_sequence_begin_bf16_v1(
    const DeltafinProviderTargetSequenceBeginBf16RequestV1* request,
    DeltafinProviderTargetSequenceBeginReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence BF16 begin request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target-sequence BF16 begin request");
    if (report->struct_size != sizeof(*report) || request->data == nullptr ||
        request->positions == 0 ||
        request->positions > DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1 ||
        (request->flags &
         ~(DELTAFIN_PROVIDER_TARGET_SEQUENCE_CAPTURE_DSPARK_V1 |
           DELTAFIN_PROVIDER_TARGET_SEQUENCE_FULL_COMMIT_ONLY_V1)) != 0 ||
        request->reserved0 != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider target-sequence BF16 begin has invalid rows/flags/reserved fields");
    }
    deltafin::provider_internal::TargetSequenceMode mode;
    switch (request->mode) {
      case DELTAFIN_PROVIDER_TARGET_SEQUENCE_PREFILL_V1:
        mode = deltafin::provider_internal::TargetSequenceMode::Prefill;
        break;
      case DELTAFIN_PROVIDER_TARGET_SEQUENCE_VERIFY_V1:
        mode = deltafin::provider_internal::TargetSequenceMode::Verify;
        break;
      default:
        throw std::invalid_argument(
            "provider target-sequence mode is unknown");
    }
    const bool full_commit_only =
        (request->flags &
         DELTAFIN_PROVIDER_TARGET_SEQUENCE_FULL_COMMIT_ONLY_V1) != 0;
    if (full_commit_only && mode !=
                                deltafin::provider_internal::
                                    TargetSequenceMode::Verify) {
      throw std::invalid_argument(
          "provider target-sequence full-commit-only requires verify mode");
    }
    constexpr std::uint64_t row_bytes = 7168 * sizeof(std::uint16_t);
    const std::uint64_t expected_bytes =
        row_bytes * static_cast<std::uint64_t>(request->positions);
    if (request->byte_length != expected_bytes ||
        request->byte_length > SIZE_MAX) {
      throw std::invalid_argument(
          "provider target-sequence BF16 byte length does not match its rows");
    }

    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    at::Tensor owned_cpu = at::empty(
        {static_cast<std::int64_t>(request->positions), 7168},
        at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
    std::memcpy(owned_cpu.mutable_data_ptr(), request->data,
                static_cast<std::size_t>(request->byte_length));
    at::Tensor hidden_rows = owned_cpu.to(
        at::TensorOptions().dtype(at::kFloat).device(session->selected.device));
    const bool capture_dspark_rows =
        (request->flags &
         DELTAFIN_PROVIDER_TARGET_SEQUENCE_CAPTURE_DSPARK_V1) != 0;
    publish_target_sequence_begin(*session, std::move(hidden_rows), mode,
                                  capture_dspark_rows, full_commit_only,
                                  report);
  });
}

extern "C" int32_t deltafin_provider_target_sequence_prepare_v1(
    const DeltafinProviderTargetSequencePrepareRequestV1* request,
    DeltafinProviderTargetSequencePrepareReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence prepare request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target-sequence prepare request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        request->spine_generation == 0 || !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider target-sequence prepare has invalid report/generation/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->sequence);
    if (request->layer_index !=
        session->target_sequence->next_layer_index()) {
      throw std::invalid_argument(
          "target-sequence prepare layer is out of transaction order");
    }
    const SpineLayerSlot& spine = require_bound_spine_layer(
        *session, request->layer_index, request->spine_generation);
    if (spine.target_residual == nullptr) {
      throw std::invalid_argument(
          "target-sequence streamed layer lacks its complete residual roster");
    }

    const auto fp32_execution = maybe_materialize_spine_fp32(
        *session, spine, request->sequence);
    const SpineFp32ExecutionView* execution =
        fp32_execution.has_value() ? &*fp32_execution : nullptr;

    deltafin::provider_internal::KdaWeights kda;
    deltafin::provider_internal::MlaWeights mla;
    deltafin::provider_internal::TargetDenseWeights dense;
    deltafin::provider_internal::MoeSpineT1 moe;
    auto binding = deltafin::provider_internal::TargetLayerBinding{
        .layer_index = request->layer_index,
        .attention_kind =
            deltafin::provider_internal::target_layer_uses_mla(
                request->layer_index)
                ? deltafin::provider_internal::TargetAttentionKind::Mla
                : deltafin::provider_internal::TargetAttentionKind::Kda,
        .residual = spine.target_residual.get()};
    if (binding.attention_kind ==
        deltafin::provider_internal::TargetAttentionKind::Kda) {
      kda = kda_weights_from_spine(spine, execution);
      qualify_kda_weights(*session, kda);
      binding.kda_weights = &kda;
    } else {
      mla = mla_weights_from_spine(spine, execution);
      qualify_mla_weights(*session, mla);
      binding.mla_weights = &mla;
      binding.mla_input_bundle = execution == nullptr
          ? spine.mla_input_bundle.get()
          : nullptr;
    }
    if (request->layer_index == 0) {
      dense = target_dense_from_spine(spine, execution);
      if (dense.packed_int8_qualified) {
        require_target_packed_shape(*session, dense.gate,
                                    "dense gate projection");
        require_target_packed_shape(*session, dense.up,
                                    "dense up projection");
        require_target_packed_shape(*session, dense.down,
                                    "dense down projection");
      }
      maybe_bundle_target_dense_zero_copy(dense);
      binding.dense = &dense;
    } else {
      moe = target_moe_from_spine(spine, execution);
      if (moe.packed_int8_qualified) {
        require_target_packed_shape(*session, moe.router, "MoE router");
        require_target_packed_shape(*session, moe.routed_down,
                                    "MoE routed-down projection");
        require_target_packed_shape(*session, moe.routed_up,
                                    "MoE routed-up projection");
        require_target_packed_shape(*session, moe.shared_gate,
                                    "MoE shared gate projection");
        require_target_packed_shape(*session, moe.shared_up,
                                    "MoE shared up projection");
        require_target_packed_shape(*session, moe.shared_down,
                                    "MoE shared down projection");
      }
      static_cast<void>(
          deltafin::provider_internal::qualify_moe_shared_gate_up(moe));
      binding.moe = &moe;
    }

    const auto kind = session->target_sequence->prepare_layer(binding);
    DeltafinProviderTargetSequencePrepareReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = request->sequence;
    produced.spine_generation = request->spine_generation;
    produced.layer_index = request->layer_index;
    produced.next_layer = session->target_sequence->next_layer_index();
    produced.positions = static_cast<std::uint32_t>(
        session->target_sequence->position_count());
    if (kind == deltafin::provider_internal::
                    TargetSequenceLayerPrepareKind::DenseCompleted) {
      produced.kind = DELTAFIN_PROVIDER_TARGET_DENSE_COMPLETE_V1;
    } else {
      produced.kind = DELTAFIN_PROVIDER_TARGET_EXPERTS_REQUIRED_V1;
      produced.top_k = DELTAFIN_PROVIDER_ROUTE_TOP_K_V1;
      const auto mailbox = session->target_sequence->expert_mailbox();
      if (mailbox.layer_index != request->layer_index ||
          mailbox.spine_generation != request->spine_generation ||
          mailbox.row_count != produced.positions) {
        throw std::logic_error(
            "target-sequence internal route mailbox disagrees with its streamed layer");
      }
      for (std::size_t row = 0; row < produced.positions; ++row) {
        const std::size_t edge =
            row * DELTAFIN_PROVIDER_ROUTE_TOP_K_V1;
        std::copy(mailbox.rows[row].route.expert_ids.begin(),
                  mailbox.rows[row].route.expert_ids.end(),
                  produced.ordered_experts + edge);
        std::copy(mailbox.rows[row].route.weight_bits.begin(),
                  mailbox.rows[row].route.weight_bits.end(),
                  produced.ordered_weight_bits + edge);
      }
    }
    *report = produced;
  });
}

extern "C" int32_t
deltafin_provider_target_sequence_take_prefetch_hint_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetSequencePrefetchHintReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(
        request, "provider target-sequence prefetch-hint request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider target-sequence prefetch-hint report does not match ABI v1");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->resource);

    // take_prefetch_hint is explicitly fail-soft: an absent roster entry,
    // provider error, already-consumed hint, or poor optional
    // prediction all become this successful zero-count report. ABI misuse and
    // stale handles above remain hard failures and are never disguised.
    const auto hint = session->target_sequence->take_prefetch_hint();
    DeltafinProviderTargetSequencePrefetchHintReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = request->resource;
    if (hint.expert_count != 0) {
      const std::uint32_t active_layer =
          session->target_sequence->next_layer_index();
      if (hint.expert_count < DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 ||
          hint.expert_count > DELTAFIN_PROVIDER_PILOT_MAX_PREFETCH_V1 ||
          hint.source_layer != active_layer ||
          hint.target_layer != active_layer + 1 ||
          hint.target_layer >= kK3Layers) {
        throw std::logic_error(
            "internal target-sequence prefetch hint escaped its active next-layer boundary");
      }
      for (std::size_t index = 0; index < hint.expert_count; ++index) {
        if (hint.expert_ids[index] >= kK3Experts ||
            (index != 0 &&
             hint.expert_ids[index - 1] >= hint.expert_ids[index])) {
          throw std::logic_error(
              "internal target-sequence prefetch hint is not canonical");
        }
      }
      produced.source_layer = hint.source_layer;
      produced.target_layer = hint.target_layer;
      produced.expert_count = hint.expert_count;
      std::copy_n(hint.expert_ids.begin(), hint.expert_count,
                  produced.expert_ids);
    }
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_sequence_finish_experts_v1(
    const DeltafinProviderTargetSequenceFinishExpertsRequestV1* request,
    DeltafinProviderTargetSequenceFinishExpertsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence expert request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target-sequence expert request");
    if (report->struct_size != sizeof(*report) ||
        (request->flags &
         ~DELTAFIN_PROVIDER_TARGET_EXPERT_RETAIN_METAL_WRAPPERS_V1) != 0 ||
        !all_zero(request->reserved) ||
        request->spine_generation == 0 || request->row_count == 0 ||
        request->row_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_TILE_ROWS_V1 ||
        request->expert_count == 0 ||
        request->expert_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1 ||
        request->expert_major_bytes == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence expert request has invalid bounds/flags/reserved fields");
    }
    for (std::size_t index = 0; index < request->expert_count; ++index) {
      if (request->expert_ids[index] >= kK3Experts ||
          (index != 0 && request->expert_ids[index - 1] >=
                             request->expert_ids[index])) {
        throw std::invalid_argument(
            "provider target-sequence expert IDs are not unique canonical ascending IDs");
      }
    }
    if (!std::all_of(
            request->expert_ids + request->expert_count,
            std::end(request->expert_ids),
            [](const std::uint16_t value) { return value == 0; })) {
      throw std::invalid_argument(
          "provider target-sequence unused expert ID slots must be zero");
    }
    const auto expert_layout = decode_expert_layout(request->expert_layout);
    const std::uint64_t expert_span = required_k3_expert_span(expert_layout);
    const std::uint64_t expected_bytes =
        expert_span * static_cast<std::uint64_t>(request->expert_count);
    if (request->expert_span_bytes != expert_span ||
        request->expert_major_length != expected_bytes ||
        request->expert_major_length > SIZE_MAX) {
      throw std::invalid_argument(
          "provider target-sequence expert byte length does not match its canonical IDs");
    }

    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->sequence);
    if (!session->moe_plans.empty()) {
      throw std::invalid_argument(
          "target sequence owns a live expert plan; finish or release it before using the legacy expert path");
    }
    if (request->layer_index !=
            session->target_sequence->next_layer_index() ||
        request->first_row >= session->target_sequence->position_count() ||
        request->row_count >
            session->target_sequence->position_count() - request->first_row) {
      throw std::invalid_argument(
          "target-sequence expert tile is outside its active layer/row bounds");
    }
    auto options = target_moe_options(*request, *session);
    const auto experts =
        deltafin::provider_internal::CanonicalExpertPositionTileT1{
            .expert_ids = std::span<const std::uint16_t>(
                request->expert_ids, request->expert_count),
            .expert_major_bytes = std::span<const std::uint8_t>(
                request->expert_major_bytes,
                static_cast<std::size_t>(request->expert_major_length)),
            .layout = expert_layout,
            .expert_span_bytes = expert_span};
    session->target_sequence->finish_expert_tile(
        static_cast<std::uint16_t>(request->first_row),
        static_cast<std::uint16_t>(request->row_count),
        request->spine_generation, experts, options);

    DeltafinProviderTargetSequenceFinishExpertsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = request->sequence;
    produced.spine_generation = request->spine_generation;
    produced.layer_index = request->layer_index;
    produced.first_row = request->first_row;
    produced.row_count = request->row_count;
    produced.next_expert_row = request->first_row + request->row_count;
    produced.state = target_sequence_state_value(
        session->target_sequence->state());
    *report = produced;
  });
}

extern "C" int32_t
deltafin_provider_target_sequence_finish_expert_spans_v1(
    const DeltafinProviderTargetSequenceFinishExpertSpansRequestV1* request,
    DeltafinProviderTargetSequenceFinishExpertsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence scattered expert request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target-sequence scattered expert request");
    if (report->struct_size != sizeof(*report) ||
        (request->flags &
         ~DELTAFIN_PROVIDER_TARGET_EXPERT_RETAIN_METAL_WRAPPERS_V1) != 0 ||
        !all_zero(request->reserved) || request->spine_generation == 0 ||
        request->row_count == 0 ||
        request->row_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_TILE_ROWS_V1 ||
        request->expert_count == 0 ||
        request->expert_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1) {
      throw std::invalid_argument(
          "provider target-sequence scattered expert request has invalid bounds/flags/reserved fields");
    }
    for (std::size_t index = 0; index < request->expert_count; ++index) {
      if (request->expert_ids[index] >= kK3Experts ||
          (index != 0 && request->expert_ids[index - 1] >=
                             request->expert_ids[index]) ||
          request->expert_span_pointers[index] == nullptr) {
        throw std::invalid_argument(
            "provider target-sequence scattered expert IDs/pointers are not canonical and non-null");
      }
    }
    if (!std::all_of(
            request->expert_ids + request->expert_count,
            std::end(request->expert_ids),
            [](const std::uint16_t value) { return value == 0; }) ||
        !std::all_of(
            request->expert_span_pointers + request->expert_count,
            std::end(request->expert_span_pointers),
            [](const std::uint8_t* value) { return value == nullptr; })) {
      throw std::invalid_argument(
          "provider target-sequence unused scattered expert slots must be zero/null");
    }
    const auto expert_layout = decode_expert_layout(request->expert_layout);
    const std::uint64_t expert_span = required_k3_expert_span(expert_layout);
    if (request->expert_span_bytes != expert_span) {
      throw std::invalid_argument(
          "provider target-sequence scattered expert span does not match its layout");
    }

    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->sequence);
    if (!session->moe_plans.empty()) {
      throw std::invalid_argument(
          "target sequence owns a live expert plan; finish or release it before using scattered expert spans");
    }
    if (request->layer_index !=
            session->target_sequence->next_layer_index() ||
        request->first_row >= session->target_sequence->position_count() ||
        request->row_count >
            session->target_sequence->position_count() - request->first_row) {
      throw std::invalid_argument(
          "target-sequence scattered expert tile is outside its active layer/row bounds");
    }
    auto options = target_moe_options(*request, *session);
    if (options.expert_backend ==
        deltafin::provider_internal::MoeExpertBackend::CudaMxfp4) {
      throw std::invalid_argument(
          "target-sequence scattered expert spans are not a CUDA cache-plan input");
    }
    const auto experts =
        deltafin::provider_internal::CanonicalExpertPositionTileT1{
            .expert_ids = std::span<const std::uint16_t>(
                request->expert_ids, request->expert_count),
            .expert_major_bytes = {},
            .layout = expert_layout,
            .expert_span_bytes = expert_span,
            .expert_span_pointers =
                std::span<const std::uint8_t* const>(
                    request->expert_span_pointers, request->expert_count)};
    session->target_sequence->finish_expert_tile(
        static_cast<std::uint16_t>(request->first_row),
        static_cast<std::uint16_t>(request->row_count),
        request->spine_generation, experts, options);

    DeltafinProviderTargetSequenceFinishExpertsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = request->sequence;
    produced.spine_generation = request->spine_generation;
    produced.layer_index = request->layer_index;
    produced.first_row = request->first_row;
    produced.row_count = request->row_count;
    produced.next_expert_row = request->first_row + request->row_count;
    produced.state =
        target_sequence_state_value(session->target_sequence->state());
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_sequence_finish_experts_v2(
    const DeltafinProviderTargetSequenceFinishExpertsRequestV2* request,
    DeltafinProviderTargetSequenceFinishExpertsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence v2 expert request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target-sequence v2 expert request");
    if (report->struct_size != sizeof(*report) ||
        (request->flags &
         ~DELTAFIN_PROVIDER_TARGET_EXPERT_RETAIN_METAL_WRAPPERS_V1) != 0 ||
        !all_zero(request->reserved) || request->spine_generation == 0 ||
        request->row_count == 0 ||
        request->row_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_TILE_ROWS_V1 ||
        request->expert_count <=
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1 ||
        request->expert_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V2 ||
        request->expert_count >
            request->row_count * DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 ||
        request->expert_ids == nullptr ||
        request->expert_ids_length != request->expert_count ||
        (request->expert_backend != DELTAFIN_PROVIDER_TARGET_EXPERT_AUTO_V1 &&
         request->expert_backend != DELTAFIN_PROVIDER_TARGET_EXPERT_CPU_V1 &&
         request->expert_backend !=
             DELTAFIN_PROVIDER_TARGET_EXPERT_METAL_V1)) {
      throw std::invalid_argument(
          "provider target-sequence v2 expert request has invalid bounds/backend/flags/reserved fields");
    }
    for (std::size_t index = 0; index < request->expert_count; ++index) {
      if (request->expert_ids[index] >= kK3Experts ||
          (index != 0 && request->expert_ids[index - 1] >=
                             request->expert_ids[index])) {
        throw std::invalid_argument(
            "provider target-sequence v2 expert IDs are not unique canonical ascending IDs");
      }
    }

    const bool contiguous = request->expert_major_bytes != nullptr;
    const bool scattered = request->expert_span_pointers != nullptr;
    if (contiguous == scattered ||
        (contiguous && request->expert_span_pointer_count != 0) ||
        (scattered &&
         (request->expert_major_length != 0 ||
          request->expert_span_pointer_count != request->expert_count))) {
      throw std::invalid_argument(
          "provider target-sequence v2 experts require exactly one canonical contiguous or scattered storage form");
    }
    if (scattered) {
      for (std::size_t index = 0; index < request->expert_count; ++index) {
        if (request->expert_span_pointers[index] == nullptr ||
            std::find(request->expert_span_pointers,
                      request->expert_span_pointers + index,
                      request->expert_span_pointers[index]) !=
                request->expert_span_pointers + index) {
          throw std::invalid_argument(
              "provider target-sequence v2 scattered expert pointers must be non-null and distinct");
        }
      }
    }

    const auto expert_layout = decode_expert_layout(request->expert_layout);
    const std::uint64_t expert_span = required_k3_expert_span(expert_layout);
    if (request->expert_span_bytes != expert_span) {
      throw std::invalid_argument(
          "provider target-sequence v2 expert span does not match its layout");
    }
    if (contiguous) {
      const std::uint64_t expected_bytes =
          expert_span * static_cast<std::uint64_t>(request->expert_count);
      if (request->expert_major_length != expected_bytes ||
          request->expert_major_length > SIZE_MAX) {
        throw std::invalid_argument(
            "provider target-sequence v2 expert byte length does not match its canonical IDs");
      }
    }

    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->sequence);
    if (session->selected.device.is_cuda()) {
      throw std::invalid_argument(
          "target-sequence v2 expert unions are CPU/Metal-only; CUDA retains its bounded residency-plan path");
    }
    if (!session->moe_plans.empty()) {
      throw std::invalid_argument(
          "target sequence owns a live expert plan; finish or release it before using the v2 expert path");
    }
    if (request->layer_index !=
            session->target_sequence->next_layer_index() ||
        request->first_row >= session->target_sequence->position_count() ||
        request->row_count >
            session->target_sequence->position_count() - request->first_row) {
      throw std::invalid_argument(
          "target-sequence v2 expert tile is outside its active layer/row bounds");
    }
    const std::vector<std::uint16_t> canonical =
        target_sequence_route_union(
            *session, request->layer_index, request->spine_generation,
            request->first_row, request->row_count,
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V2);
    if (canonical.size() != request->expert_count ||
        !std::equal(canonical.begin(), canonical.end(),
                    request->expert_ids)) {
      throw std::invalid_argument(
          "target-sequence v2 expert IDs are not the exact canonical route union");
    }

    auto options = target_moe_options(*request, *session);
    if (options.expert_backend ==
        deltafin::provider_internal::MoeExpertBackend::CudaMxfp4) {
      throw std::logic_error(
          "target-sequence v2 expert backend resolved to forbidden CUDA");
    }
    const auto experts =
        deltafin::provider_internal::CanonicalExpertPositionTileT1{
            .expert_ids = std::span<const std::uint16_t>(
                request->expert_ids, request->expert_count),
            .expert_major_bytes = contiguous
                ? std::span<const std::uint8_t>(
                      request->expert_major_bytes,
                      static_cast<std::size_t>(request->expert_major_length))
                : std::span<const std::uint8_t>{},
            .layout = expert_layout,
            .expert_span_bytes = expert_span,
            .expert_span_pointers = scattered
                ? std::span<const std::uint8_t* const>(
                      request->expert_span_pointers,
                      static_cast<std::size_t>(
                          request->expert_span_pointer_count))
                : std::span<const std::uint8_t* const>{}};
    session->target_sequence->finish_expert_tile(
        static_cast<std::uint16_t>(request->first_row),
        static_cast<std::uint16_t>(request->row_count),
        request->spine_generation, experts, options);

    DeltafinProviderTargetSequenceFinishExpertsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = request->sequence;
    produced.spine_generation = request->spine_generation;
    produced.layer_index = request->layer_index;
    produced.first_row = request->first_row;
    produced.row_count = request->row_count;
    produced.next_expert_row = request->first_row + request->row_count;
    produced.state =
        target_sequence_state_value(session->target_sequence->state());
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_sequence_plan_experts_v1(
    const DeltafinProviderTargetSequencePlanExpertsRequestV1* request,
    DeltafinProviderTargetSequencePlanExpertsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence expert-plan request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target-sequence expert-plan request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        !all_zero(request->reserved) || request->spine_generation == 0 ||
        request->row_count == 0 ||
        request->row_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_TILE_ROWS_V1 ||
        request->expert_count == 0 ||
        request->expert_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1 ||
        (request->expert_backend !=
             DELTAFIN_PROVIDER_TARGET_EXPERT_AUTO_V1 &&
         request->expert_backend !=
             DELTAFIN_PROVIDER_TARGET_EXPERT_CUDA_V1)) {
      throw std::invalid_argument(
          "provider target-sequence expert plan has invalid bounds/backend/flags/reserved fields");
    }
    for (std::size_t index = 0; index < request->expert_count; ++index) {
      if (request->expert_ids[index] >= kK3Experts ||
          (index != 0 && request->expert_ids[index - 1] >=
                             request->expert_ids[index])) {
        throw std::invalid_argument(
            "provider target-sequence expert-plan IDs are not unique canonical ascending IDs");
      }
    }
    if (!std::all_of(
            request->expert_ids + request->expert_count,
            std::end(request->expert_ids),
            [](const std::uint16_t value) { return value == 0; })) {
      throw std::invalid_argument(
          "provider target-sequence unused expert-plan ID slots must be zero");
    }

    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    if (!session->selected.device.is_cuda()) {
      throw std::invalid_argument(
          "target expert residency planning requires a selected CUDA provider");
    }
    require_live_target_sequence(*session, request->sequence);
    if (!session->moe_plans.empty()) {
      throw std::invalid_argument(
          "provider session already owns a live expert residency plan");
    }
    if (request->layer_index !=
            session->target_sequence->next_layer_index() ||
        request->first_row >= session->target_sequence->position_count() ||
        request->row_count >
            session->target_sequence->position_count() - request->first_row) {
      throw std::invalid_argument(
          "target-sequence expert plan is outside its active layer/row bounds");
    }
    const std::vector<std::uint16_t> canonical =
        target_sequence_route_union(
            *session, request->layer_index, request->spine_generation,
            request->first_row, request->row_count,
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1);
    if (canonical.size() != request->expert_count ||
        !std::equal(canonical.begin(), canonical.end(),
                    request->expert_ids)) {
      throw std::invalid_argument(
          "target-sequence expert plan IDs are not the exact canonical route union");
    }

    // Freeze CPU or CUDA before returning a miss list to the Rust reader. No
    // backend decision occurs after caller-owned I/O begins.
    auto options = target_moe_plan_options(*request, *session);
    const std::uint32_t effective_backend =
        target_moe_backend_value(options.expert_backend);
    const auto handle = session->allocate_resource();
    std::vector<std::uint16_t> missing;
    std::size_t capacity_experts = 0;
    bool residency_enabled = false;
    bool cuda_plan_live = false;
    try {
      if (options.expert_backend ==
          deltafin::provider_internal::MoeExpertBackend::CudaMxfp4) {
        if (session->cuda_expert_cache == nullptr) {
          throw std::logic_error(
              "frozen CUDA expert plan lost its session cache");
        }
        const auto planned = session->cuda_expert_cache->plan(
            handle, request->layer_index, canonical);
        cuda_plan_live = true;
        missing = planned.missing_experts;
        capacity_experts = planned.capacity_experts;
        residency_enabled = planned.residency_enabled;
        options.cuda_plan = handle;
      } else {
        missing = canonical;
      }
      if (capacity_experts > UINT32_MAX ||
          missing.size() >
              DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1) {
        throw std::runtime_error(
            "CUDA expert-plan report exceeds the bounded provider ABI");
      }
      MoePlanSlot slot{
          .sequence = request->sequence,
          .spine_generation = request->spine_generation,
          .layer_index = request->layer_index,
          .first_row = request->first_row,
          .row_count = request->row_count,
          .canonical_experts = canonical,
          .missing_experts = missing,
          .options = std::move(options)};
      const auto [ignored, inserted] =
          session->moe_plans.emplace(handle, std::move(slot));
      static_cast<void>(ignored);
      if (!inserted) {
        throw std::runtime_error(
            "provider expert-plan handle collision");
      }
      cuda_plan_live = false;
    } catch (...) {
      if (cuda_plan_live && session->cuda_expert_cache != nullptr) {
        session->cuda_expert_cache->cancel_plan(handle);
      }
      throw;
    }

    DeltafinProviderTargetSequencePlanExpertsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.plan = handle;
    produced.spine_generation = request->spine_generation;
    produced.layer_index = request->layer_index;
    produced.first_row = request->first_row;
    produced.row_count = request->row_count;
    produced.effective_backend = effective_backend;
    produced.missing_count = static_cast<std::uint32_t>(missing.size());
    produced.cache_capacity_experts =
        static_cast<std::uint32_t>(capacity_experts);
    produced.residency_enabled = residency_enabled ? 1u : 0u;
    std::copy(missing.begin(), missing.end(), produced.missing_experts);
    *report = produced;
  });
}

extern "C" int32_t
deltafin_provider_target_sequence_finish_planned_experts_v1(
    const DeltafinProviderTargetSequenceFinishPlannedExpertsRequestV1* request,
    DeltafinProviderTargetSequenceFinishExpertsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence planned-finish request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target-sequence planned-finish request");
    if (report->struct_size != sizeof(*report) || request->plan == 0 ||
        request->flags != 0 || !all_zero(request->reserved) ||
        request->spine_generation == 0 || request->row_count == 0 ||
        request->row_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_TILE_ROWS_V1 ||
        request->missing_count >
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1 ||
        (request->missing_count == 0) !=
            (request->expert_major_bytes == nullptr)) {
      throw std::invalid_argument(
          "provider target-sequence planned finish has invalid bounds/pointer/flags/reserved fields");
    }
    for (std::size_t index = 0; index < request->missing_count; ++index) {
      if (request->missing_experts[index] >= kK3Experts ||
          (index != 0 && request->missing_experts[index - 1] >=
                             request->missing_experts[index])) {
        throw std::invalid_argument(
            "provider target-sequence planned misses are not unique canonical ascending IDs");
      }
    }
    if (!std::all_of(
            request->missing_experts + request->missing_count,
            std::end(request->missing_experts),
            [](const std::uint16_t value) { return value == 0; })) {
      throw std::invalid_argument(
          "provider target-sequence unused planned-miss slots must be zero");
    }
    const std::uint64_t expected_bytes =
        K3_RAW_V1_EXPERT_SPAN *
        static_cast<std::uint64_t>(request->missing_count);
    if (request->expert_major_length != expected_bytes ||
        request->expert_major_length > SIZE_MAX) {
      throw std::invalid_argument(
          "provider target-sequence planned miss bytes do not match raw-v1 IDs");
    }

    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto found = session->moe_plans.find(request->plan);
    if (found == session->moe_plans.end()) {
      throw std::invalid_argument(
          "provider expert-plan handle is stale or unknown");
    }
    const MoePlanSlot& planned = found->second;
    if (planned.sequence != request->sequence ||
        planned.spine_generation != request->spine_generation ||
        planned.layer_index != request->layer_index ||
        planned.first_row != request->first_row ||
        planned.row_count != request->row_count ||
        planned.missing_experts.size() != request->missing_count ||
        !std::equal(planned.missing_experts.begin(),
                    planned.missing_experts.end(),
                    request->missing_experts)) {
      throw std::invalid_argument(
          "provider planned finish does not exactly match its residency snapshot");
    }
    require_live_target_sequence(*session, request->sequence);
    if (request->layer_index !=
            session->target_sequence->next_layer_index() ||
        request->first_row >= session->target_sequence->position_count() ||
        request->row_count >
            session->target_sequence->position_count() - request->first_row) {
      throw std::invalid_argument(
          "target-sequence planned finish is outside its active layer/row bounds");
    }
    const std::vector<std::uint16_t> current_union =
        target_sequence_route_union(
            *session, request->layer_index, request->spine_generation,
            request->first_row, request->row_count,
            DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1);
    if (current_union != planned.canonical_experts) {
      throw std::invalid_argument(
          "target-sequence planned finish route union changed after planning");
    }

    // All request/sequence checks are complete. Moving the slot out now makes
    // this a consume-on-attempt transaction: execution success or failure can
    // never leave an apparently reusable session handle behind.
    MoePlanSlot consumed = std::move(found->second);
    session->moe_plans.erase(found);
    const auto cancel_cuda_plan = [&] {
      if (consumed.options.cuda_plan != 0 &&
          consumed.options.cuda_cache != nullptr) {
        consumed.options.cuda_cache->cancel_plan(consumed.options.cuda_plan);
      }
    };
    try {
      const auto experts =
          deltafin::provider_internal::CanonicalExpertPositionTileT1{
              .expert_ids = std::span<const std::uint16_t>(
                  request->missing_experts, request->missing_count),
              .expert_major_bytes = std::span<const std::uint8_t>(
                  request->expert_major_bytes,
                  static_cast<std::size_t>(request->expert_major_length)),
              .layout =
                  deltafin::provider_internal::MoeExpertLayout::RawV1,
              .expert_span_bytes = K3_RAW_V1_EXPERT_SPAN};
      session->target_sequence->finish_expert_tile(
          static_cast<std::uint16_t>(request->first_row),
          static_cast<std::uint16_t>(request->row_count),
          request->spine_generation, experts, consumed.options);
    } catch (...) {
      cancel_cuda_plan();
      throw;
    }
    // CUDA success and classified Auto fallback consume the adapter plan
    // internally. cancel_plan is deliberately idempotent, so this one cleanup
    // point also covers CPU-frozen plans and future early-success variants.
    cancel_cuda_plan();

    DeltafinProviderTargetSequenceFinishExpertsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = request->sequence;
    produced.spine_generation = request->spine_generation;
    produced.layer_index = request->layer_index;
    produced.first_row = request->first_row;
    produced.row_count = request->row_count;
    produced.next_expert_row = request->first_row + request->row_count;
    produced.state =
        target_sequence_state_value(session->target_sequence->state());
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_moe_plan_release_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "provider expert-plan release request");
    if (request->resource == 0) {
      throw std::invalid_argument(
          "provider expert-plan release handle is zero");
    }
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto found = session->moe_plans.find(request->resource);
    if (found == session->moe_plans.end()) {
      throw std::invalid_argument(
          "provider expert-plan handle is stale or unknown");
    }
    if (found->second.options.cuda_plan != 0 &&
        found->second.options.cuda_cache != nullptr) {
      found->second.options.cuda_cache->cancel_plan(
          found->second.options.cuda_plan);
    }
    // Deliberately independent of target_sequence liveness: callers can
    // cancel a poisoned/abandoned sequence first and release pinned hits next.
    session->moe_plans.erase(found);
  });
}

extern "C" int32_t deltafin_provider_target_sequence_finish_tail_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetSequenceTailReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request,
                             "provider target-sequence tail request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider target-sequence tail report does not match provider ABI v1");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->resource);
    const std::span<const std::uint32_t> decisions =
        session->target_sequence->finish_tail();
    if (decisions.empty() ||
        decisions.size() > DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1) {
      throw std::logic_error(
          "target-sequence tail returned an invalid decision count");
    }
    const auto stats = session->target_sequence->stats();
    DeltafinProviderTargetSequenceTailReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = request->resource;
    produced.token_count = static_cast<std::uint32_t>(decisions.size());
    produced.state = target_sequence_state_value(
        session->target_sequence->state());
    produced.tail_rows = static_cast<std::uint32_t>(stats.tail_rows);
    produced.tail_provider_dispatches =
        static_cast<std::uint32_t>(stats.tail_provider_dispatches);
    std::copy(decisions.begin(), decisions.end(), produced.token_ids);
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_sequence_dspark_rows_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTensorReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request,
                             "provider target-sequence DSpark rows request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider target-sequence DSpark rows report does not match provider ABI v1");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->resource);
    at::Tensor rows = session->target_sequence->dspark_target_rows();
    if (!rows.defined() || rows.scalar_type() != at::kBFloat16 ||
        !rows.is_contiguous() || rows.dim() != 2 || rows.size(0) < 1 ||
        rows.size(0) > DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1 ||
        rows.size(1) != 5 * 7168 ||
        rows.device() != session->selected.device) {
      throw std::logic_error(
          "target-sequence DSpark capture returned an invalid provider tensor");
    }
    const auto handle = session->allocate_resource();
    const auto [ignored, inserted] =
        session->tensors.emplace(handle, std::move(rows));
    static_cast<void>(ignored);
    if (!inserted) {
      throw std::runtime_error(
          "target-sequence DSpark tensor handle collision");
    }

    DeltafinProviderTensorReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.tensor = handle;
    produced.rows = static_cast<std::uint64_t>(
        session->target_sequence->position_count());
    produced.columns = 5 * 7168;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_sequence_commit_v1(
    const DeltafinProviderTargetSequenceCommitRequestV1* request,
    DeltafinProviderTargetSequenceCommitReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "provider target-sequence commit request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version,
                   "provider target-sequence commit request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "provider target-sequence commit has invalid report/flags/reserved fields");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->sequence);
    if (request->positions >
        session->target_sequence->position_count()) {
      throw std::invalid_argument(
          "target-sequence commit prefix exceeds its position count");
    }
    if (request->positions >
            UINT64_MAX - session->committed_target_positions ||
        session->committed_target_generation == UINT64_MAX) {
      throw std::runtime_error(
          "target committed-position count is exhausted");
    }
    session->target_sequence->commit_prefix(request->positions);
    const auto handle = session->target_sequence_handle;
    session->committed_target_positions += request->positions;
    ++session->committed_target_generation;
    session->target_sequence.reset();
    session->target_sequence_handle = 0;

    DeltafinProviderTargetSequenceCommitReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = handle;
    produced.committed_positions = request->positions;
    produced.session_committed_positions =
        session->committed_target_positions;
    produced.state = DELTAFIN_PROVIDER_TARGET_SEQUENCE_COMMITTED_V1;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_sequence_stats_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetSequenceStatsReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request,
                             "provider target-sequence stats request");
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument(
          "provider target-sequence stats report does not match provider ABI v1");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->resource);
    const auto stats = session->target_sequence->stats();
    DeltafinProviderTargetSequenceStatsReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.sequence = request->resource;
    produced.positions = stats.positions;
    produced.streamed_layer_passes = stats.streamed_layer_passes;
    produced.attention_rows = stats.attention_rows;
    produced.expert_row_requests = stats.expert_row_requests;
    produced.expert_rows_completed = stats.expert_rows_completed;
    produced.expert_tiles_completed = stats.expert_tiles_completed;
    produced.tail_rows = stats.tail_rows;
    produced.tail_provider_dispatches = stats.tail_provider_dispatches;
    produced.maximum_live_streamed_layers =
        stats.maximum_live_streamed_layers;
    produced.maximum_experts_per_request =
        stats.maximum_experts_per_request;
    produced.maximum_positions_per_expert_tile =
        stats.maximum_positions_per_expert_tile;
    produced.staged_kda_storage_bytes = stats.staged_kda_storage_bytes;
    produced.verify_snapshot_bytes = stats.verify_snapshot_bytes;
    produced.projected_mla_storage_bytes =
        stats.projected_mla_storage_bytes;
    produced.additional_mla_storage_bytes =
        stats.additional_mla_storage_bytes;
    produced.mode =
        target_sequence_mode_value(session->target_sequence->mode());
    produced.state =
        target_sequence_state_value(session->target_sequence->state());
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_target_sequence_cancel_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request,
                             "provider target-sequence cancel request");
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    require_live_target_sequence(*session, request->resource);
    try {
      session->target_sequence->cancel();
      session->target_sequence.reset();
      session->target_sequence_handle = 0;
    } catch (...) {
      session->target_sequence.reset();
      session->target_sequence_handle = 0;
      session->target_cache_store.reset();
      throw;
    }
  });
}

namespace {

using deltafin::provider_internal::DSparkDecoderWeights;
using deltafin::provider_internal::DSparkModel;
using deltafin::provider_internal::DSparkModelWeights;
using deltafin::provider_internal::DSparkMlaWeights;
using deltafin::provider_internal::DSparkMlpWeights;
using deltafin::provider_internal::DSparkShape;
using deltafin::provider_internal::DSparkTargetIo;

std::vector<std::int64_t> dspark_slot_shape(const std::uint32_t slot,
                                            const DSparkShape& shape) {
  switch (slot) {
    case 1:
      return {shape.hidden_size, shape.target_context_width()};
    case 2:
    case 3:
      return {shape.hidden_size};
    case 4:
    case 5:
      return {shape.vocabulary_size, shape.markov_rank};
    case 6:
      return {1, shape.hidden_size + shape.markov_rank};
    case 7:
      return {1};
    default:
      break;
  }
  if (slot < 8 || slot > DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1) {
    throw std::invalid_argument("DSpark tensor slot is outside 1..67");
  }
  const std::uint32_t component = (slot - 8) % 12;
  switch (component) {
    case 0:
    case 1:
      return {shape.hidden_size};
    case 2:
      return {shape.q_lora_rank, shape.hidden_size};
    case 3:
      return {shape.q_lora_rank};
    case 4:
      return {shape.num_heads * shape.query_head_dim(), shape.q_lora_rank};
    case 5:
      return {shape.kv_lora_rank + shape.qk_rope_head_dim,
              shape.hidden_size};
    case 6:
      return {shape.kv_lora_rank};
    case 7:
      return {shape.num_heads *
                  (shape.qk_nope_head_dim + shape.value_head_dim),
              shape.kv_lora_rank};
    case 8:
      return {shape.hidden_size, shape.num_heads * shape.value_head_dim};
    case 9:
    case 10:
      return {shape.intermediate_size, shape.hidden_size};
    case 11:
      return {shape.hidden_size, shape.intermediate_size};
    default:
      throw std::logic_error("DSpark tensor component is unreachable");
  }
}

at::Tensor copy_dspark_bf16(const std::uint8_t* data,
                            const std::vector<std::int64_t>& shape,
                            const at::Device& device) {
  at::Tensor source = at::empty(
      shape, at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
  std::memcpy(source.mutable_data_ptr(), data,
              static_cast<std::size_t>(source.numel()) * sizeof(std::uint16_t));
  return source.to(device).contiguous();
}

DSparkModelWeights bind_dspark_roster(
    const DeltafinProviderDSparkCreateV1& request, const DSparkShape& shape,
    const at::Device& device) {
  if (request.tensor_count != DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1 ||
      request.tensors == nullptr) {
    throw std::invalid_argument(
        "DSpark bind requires exactly 67 owned checkpoint tensors");
  }
  const std::vector<DeltafinProviderDSparkTensorV1> descriptors(
      request.tensors, request.tensors + request.tensor_count);
  std::array<const DeltafinProviderDSparkTensorV1*,
             DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1 + 1>
      by_slot = {};
  struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
  };
  std::vector<AddressRange> ranges;
  ranges.reserve(descriptors.size());
  for (const auto& descriptor : descriptors) {
    if (descriptor.slot == 0 ||
        descriptor.slot > DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1 ||
        by_slot[descriptor.slot] != nullptr || descriptor.flags != 0 ||
        descriptor.scalar_type != DELTAFIN_PROVIDER_DSPARK_BF16_V1 ||
        !all_zero(descriptor.reserved)) {
      throw std::invalid_argument(
          "DSpark tensor roster has an extra, duplicate, or invalid slot");
    }
    const std::vector<std::int64_t> expected =
        dspark_slot_shape(descriptor.slot, shape);
    if (descriptor.rank != expected.size() || descriptor.rank < 1 ||
        descriptor.rank > 2) {
      throw std::invalid_argument("DSpark tensor rank is invalid");
    }
    std::uint64_t elements = 1;
    for (std::uint32_t dimension = 0; dimension < 2; ++dimension) {
      const std::uint64_t expected_dimension =
          dimension < descriptor.rank
          ? static_cast<std::uint64_t>(expected[dimension])
          : 0;
      if (descriptor.shape[dimension] != expected_dimension) {
        throw std::invalid_argument("DSpark tensor shape is invalid");
      }
      if (dimension < descriptor.rank) {
        elements *= descriptor.shape[dimension];
      }
    }
    if (descriptor.data == nullptr ||
        elements > std::numeric_limits<std::uint64_t>::max() / 2 ||
        descriptor.data_length != elements * 2 ||
        descriptor.data_length > static_cast<std::uint64_t>(SIZE_MAX)) {
      throw std::invalid_argument(
          "DSpark tensor BF16 pointer/length contract is invalid");
    }
    const std::uintptr_t begin =
        reinterpret_cast<std::uintptr_t>(descriptor.data);
    if (descriptor.data_length >
        std::numeric_limits<std::uintptr_t>::max() - begin) {
      throw std::invalid_argument("DSpark tensor address range overflows");
    }
    ranges.push_back(
        AddressRange{begin, begin + descriptor.data_length});
    by_slot[descriptor.slot] = &descriptor;
  }
  std::sort(ranges.begin(), ranges.end(),
            [](const AddressRange& left, const AddressRange& right) {
              return left.begin < right.begin;
            });
  for (std::size_t index = 1; index < ranges.size(); ++index) {
    if (ranges[index].begin < ranges[index - 1].end) {
      throw std::invalid_argument(
          "DSpark tensor byte ranges alias or overlap");
    }
  }
  std::array<at::Tensor,
             DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1 + 1>
      tensors;
  for (std::uint32_t slot = 1;
       slot <= DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1; ++slot) {
    const auto* descriptor = by_slot[slot];
    if (descriptor == nullptr) {
      throw std::invalid_argument("DSpark tensor roster is incomplete");
    }
    tensors[slot] = copy_dspark_bf16(
        descriptor->data, dspark_slot_shape(slot, shape), device);
  }
  DSparkModelWeights weights{
      .context_projection = std::move(tensors[1]),
      .context_norm = std::move(tensors[2]),
      .layers = {},
      .fused_context_projection = {},
      .final_norm = std::move(tensors[3]),
      .markov_embedding = std::move(tensors[4]),
      .markov_output = std::move(tensors[5]),
      .confidence_weight = std::move(tensors[6]),
      .confidence_bias = std::move(tensors[7]),
  };
  for (std::size_t layer = 0;
       layer < deltafin::provider_internal::kDSparkLayers; ++layer) {
    const std::size_t base = 8 + layer * 12;
    weights.layers[layer] = DSparkDecoderWeights{
        .input_norm = std::move(tensors[base]),
        .attention = DSparkMlaWeights{
            .query_a = std::move(tensors[base + 2]),
            .query_a_norm = std::move(tensors[base + 3]),
            .query_b = std::move(tensors[base + 4]),
            .key_value_a = std::move(tensors[base + 5]),
            .key_value_a_norm = std::move(tensors[base + 6]),
            .key_value_b = std::move(tensors[base + 7]),
            .output = std::move(tensors[base + 8]),
        },
        .post_attention_norm = std::move(tensors[base + 1]),
        .mlp = DSparkMlpWeights{
            .gate = std::move(tensors[base + 9]),
            .up = std::move(tensors[base + 10]),
            .down = std::move(tensors[base + 11]),
        },
    };
  }
  return weights;
}

DeltafinProviderDSparkReportV1 dspark_report(
    const DeltafinProviderDSparkHandleV1 handle, const DSparkModel& model,
    const std::uint32_t flags) {
  DeltafinProviderDSparkReportV1 report = {};
  report.struct_size = sizeof(report);
  report.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  report.model = handle;
  report.cache_length = static_cast<std::uint64_t>(model.cache().length());
  report.cache_generation = model.cache().generation();
  report.tensor_count = DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1;
  report.flags = flags;
  return report;
}

std::unique_ptr<DSparkModel>& require_dspark_model(
    Session& session, const DeltafinProviderDSparkHandleV1 handle) {
  const auto found = session.dspark_models.find(handle);
  if (found == session.dspark_models.end()) {
    throw std::invalid_argument("DSpark model handle is stale or unknown");
  }
  return found->second;
}

}  // namespace

extern "C" int32_t deltafin_provider_dspark_create_v1(
    const DeltafinProviderDSparkCreateV1* request,
    DeltafinProviderDSparkReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("DSpark create request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "DSpark create request");
    if (report->struct_size != sizeof(*report) ||
        (request->flags & ~DELTAFIN_PROVIDER_DSPARK_SYNTHETIC_V1) != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument(
          "DSpark create report/flags/reserved fields are invalid");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const bool synthetic =
        (request->flags & DELTAFIN_PROVIDER_DSPARK_SYNTHETIC_V1) != 0;
    const DSparkShape shape =
        synthetic ? DSparkShape::small_canary() : DSparkShape::k3();
    DSparkTargetIo target_io;
    if (synthetic) {
      const std::uint64_t expected =
          static_cast<std::uint64_t>(shape.vocabulary_size) *
          static_cast<std::uint64_t>(shape.hidden_size);
      if (request->synthetic_head_f32 == nullptr ||
          request->synthetic_head_elements != expected) {
        throw std::invalid_argument(
            "synthetic DSpark create requires its exact fp32 head canary");
      }
      target_io.language_model_head.dense_f32 = copy_f32_to_device(
          request->synthetic_head_f32,
          static_cast<std::uint64_t>(shape.vocabulary_size),
          static_cast<std::uint64_t>(shape.hidden_size),
          session->selected.device, false);
      target_io.exact_k3 = false;
    } else {
      if (request->synthetic_head_f32 != nullptr ||
          request->synthetic_head_elements != 0 ||
          session->target_tail == nullptr) {
        throw std::invalid_argument(
            "production DSpark requires the session's already-bound K3 head");
      }
      target_io.language_model_head =
          session->target_tail->language_model_head;
      target_io.head_packed_int8_qualified =
          session->target_tail->packed_int8_qualified;
      target_io.exact_k3 = true;
    }
    DSparkModelWeights weights = bind_dspark_roster(
        *request, shape, session->selected.device);
    auto staged = std::make_unique<DSparkModel>(
        shape, std::move(weights), std::move(target_io));
    const DeltafinProviderDSparkHandleV1 handle =
        session->allocate_resource();
    const auto [ignored, inserted] =
        session->dspark_models.emplace(handle, std::move(staged));
    static_cast<void>(ignored);
    if (!inserted) {
      throw std::runtime_error("DSpark model handle collision");
    }
    *report = dspark_report(handle, *session->dspark_models.at(handle),
                            request->flags);
  });
}

extern "C" int32_t deltafin_provider_dspark_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "DSpark destroy request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    for (const auto& [ignored, snapshot] : session->dspark_snapshots) {
      static_cast<void>(ignored);
      if (snapshot.model == request->resource) {
        throw std::invalid_argument(
            "DSpark model cannot destroy while snapshots remain live");
      }
    }
    if (session->dspark_models.erase(request->resource) != 1) {
      throw std::invalid_argument("DSpark model handle is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_dspark_append_target_v1(
    const DeltafinProviderDSparkAppendV1* request,
    DeltafinProviderDSparkReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("DSpark append request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "DSpark append request");
    if (report->struct_size != sizeof(*report) || request->rows == 0 ||
        request->target_context_bf16 == nullptr || request->positions == nullptr ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument("DSpark append fields are invalid");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    auto& model = require_dspark_model(*session, request->model);
    const std::uint64_t columns = static_cast<std::uint64_t>(
        model->shape().target_context_width());
    if (request->rows > static_cast<std::uint64_t>(INT64_MAX) ||
        columns > UINT64_MAX / request->rows ||
        columns * request->rows > UINT64_MAX / 2 ||
        request->target_context_bytes != columns * request->rows * 2) {
      throw std::invalid_argument("DSpark append byte geometry is invalid");
    }
    const std::vector<std::int64_t> shape = {
        static_cast<std::int64_t>(request->rows),
        static_cast<std::int64_t>(columns)};
    const at::Tensor context = copy_dspark_bf16(
        request->target_context_bf16, shape, session->selected.device);
    const at::Tensor positions_cpu = at::from_blob(
        const_cast<std::int64_t*>(request->positions),
        {static_cast<std::int64_t>(request->rows)},
        at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    const at::Tensor positions = positions_cpu.to(session->selected.device);
    model->append_target_context(context, positions);
    // Both arms must share a type: GCC's -Wextra rejects mixing an unsigned
    // literal with an unscoped enumerator in a conditional expression.
    const std::uint32_t flags =
        model->shape().is_exact_k3()
            ? 0u
            : static_cast<std::uint32_t>(DELTAFIN_PROVIDER_DSPARK_SYNTHETIC_V1);
    *report = dspark_report(request->model, *model, flags);
  });
}

extern "C" int32_t deltafin_provider_dspark_append_target_tensor_v1(
    const DeltafinProviderDSparkAppendTensorV1* request,
    DeltafinProviderDSparkReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument(
          "DSpark tensor append request/report is null");
    }
    require_header(request->struct_size, sizeof(*request),
                   request->abi_version, "DSpark tensor append request");
    if (report->struct_size != sizeof(*report) || request->rows == 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument("DSpark tensor append fields are invalid");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    auto& model = require_dspark_model(*session, request->model);
    if (request->expected_cache_length !=
            static_cast<std::uint64_t>(model->cache().length()) ||
        request->expected_cache_generation != model->cache().generation()) {
      throw std::invalid_argument(
          "DSpark tensor append expected cache boundary is stale");
    }
    const auto found = session->tensors.find(request->target_context);
    if (found == session->tensors.end()) {
      throw std::invalid_argument(
          "DSpark target-context tensor handle is stale or unknown");
    }
    const at::Tensor& context = found->second;
    if (request->rows > static_cast<std::uint64_t>(INT64_MAX) ||
        !context.defined() || context.scalar_type() != at::kBFloat16 ||
        !context.is_contiguous() ||
        context.device() != session->selected.device || context.dim() != 2 ||
        context.size(0) < static_cast<std::int64_t>(request->rows) ||
        context.size(1) != model->shape().target_context_width()) {
      throw std::invalid_argument(
          "DSpark target-context tensor has invalid dtype/device/shape");
    }
    if (model->cache().length() >
        model->shape().max_position - static_cast<std::int64_t>(request->rows)) {
      throw std::invalid_argument(
          "DSpark tensor append exceeds the context limit");
    }
    const at::Tensor positions = at::arange(
        model->cache().length(),
        model->cache().length() + static_cast<std::int64_t>(request->rows),
        at::TensorOptions().dtype(at::kLong).device(session->selected.device));
    model->append_target_context(
        context.narrow(0, 0, static_cast<std::int64_t>(request->rows)),
        positions);
    // Both arms must share a type: GCC's -Wextra rejects mixing an unsigned
    // literal with an unscoped enumerator in a conditional expression.
    const std::uint32_t flags =
        model->shape().is_exact_k3()
            ? 0u
            : static_cast<std::uint32_t>(DELTAFIN_PROVIDER_DSPARK_SYNTHETIC_V1);
    *report = dspark_report(request->model, *model, flags);
  });
}

extern "C" int32_t deltafin_provider_dspark_snapshot_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderDSparkSnapshotReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (report == nullptr || report->struct_size != sizeof(*report)) {
      throw std::invalid_argument("DSpark snapshot report is invalid");
    }
    require_resource_request(request, "DSpark snapshot request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    auto& model = require_dspark_model(*session, request->resource);
    const auto handle = session->allocate_resource();
    DSparkSnapshotSlot slot{
        .model = request->resource,
        .snapshot = model->cache().snapshot(),
    };
    session->dspark_snapshots.emplace(handle, std::move(slot));
    DeltafinProviderDSparkSnapshotReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.snapshot = handle;
    produced.cache_length =
        static_cast<std::uint64_t>(model->cache().length());
    produced.cache_generation = model->cache().generation();
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_dspark_restore_v1(
    const DeltafinProviderDSparkRestoreV1* request,
    DeltafinProviderDSparkReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("DSpark restore request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "DSpark restore request");
    if (report->struct_size != sizeof(*report) ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument("DSpark restore fields are invalid");
    }
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    auto& model = require_dspark_model(*session, request->model);
    const auto found = session->dspark_snapshots.find(request->snapshot);
    if (found == session->dspark_snapshots.end() ||
        found->second.model != request->model) {
      throw std::invalid_argument("DSpark snapshot is stale or belongs elsewhere");
    }
    model->cache().restore(found->second.snapshot);
    // Both arms must share a type: GCC's -Wextra rejects mixing an unsigned
    // literal with an unscoped enumerator in a conditional expression.
    const std::uint32_t flags =
        model->shape().is_exact_k3()
            ? 0u
            : static_cast<std::uint32_t>(DELTAFIN_PROVIDER_DSPARK_SYNTHETIC_V1);
    *report = dspark_report(request->model, *model, flags);
  });
}

extern "C" int32_t deltafin_provider_dspark_snapshot_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "DSpark snapshot destroy request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->dspark_snapshots.erase(request->resource) != 1) {
      throw std::invalid_argument("DSpark snapshot handle is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_dspark_propose_v1(
    const DeltafinProviderDSparkProposeV1* request,
    DeltafinProviderDSparkProposalReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("DSpark proposal request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "DSpark proposal request");
    if (report->struct_size != sizeof(*report) ||
        !all_zero(request->reserved) || request->score_rows < 1 ||
        request->score_rows > DELTAFIN_PROVIDER_DSPARK_QUERY_ROWS_V1 ||
        request->query_embeddings_bf16 == nullptr) {
      throw std::invalid_argument("DSpark proposal fields are invalid");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    auto& model = require_dspark_model(*session, request->model);
    const std::uint64_t expected_bytes =
        DELTAFIN_PROVIDER_DSPARK_QUERY_ROWS_V1 *
        static_cast<std::uint64_t>(model->shape().hidden_size) * 2;
    if (request->query_embedding_bytes != expected_bytes) {
      throw std::invalid_argument(
          "DSpark proposal embedding byte length is invalid");
    }
    const at::Tensor embeddings = copy_dspark_bf16(
        request->query_embeddings_bf16,
        {DELTAFIN_PROVIDER_DSPARK_QUERY_ROWS_V1,
         model->shape().hidden_size},
        session->selected.device);
    const auto proposal = model->propose_from_embeddings(
        request->anchor_token_id, request->score_rows, embeddings);
    const at::Tensor ids = proposal.token_ids.to(at::kCPU).contiguous();
    const at::Tensor confidence =
        proposal.confidence_logits.to(at::kCPU).contiguous();
    DeltafinProviderDSparkProposalReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.score_rows = request->score_rows;
    produced.anchor_position =
        static_cast<std::uint64_t>(proposal.anchor_position);
    produced.cache_generation = proposal.cache_generation;
    const auto* id_values = ids.const_data_ptr<std::int64_t>();
    const auto* confidence_values = confidence.const_data_ptr<float>();
    for (std::uint32_t row = 0; row < request->score_rows; ++row) {
      if (id_values[row] < 0 ||
          id_values[row] > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("DSpark proposal ID exceeds ABI range");
      }
      produced.token_ids[row] = static_cast<std::uint32_t>(id_values[row]);
      produced.confidence_logits[row] = confidence_values[row];
    }
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_qwen_create_v1(
    const DeltafinProviderQwenCreateV1* request,
    DeltafinProviderQwenReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("Qwen create request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "Qwen create request");
    if (report->struct_size != sizeof(*report) ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument("Qwen create report/reserved fields are invalid");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto shape = deltafin::provider_internal::QwenShape::pinned(
        request->variant);
    auto weights = deltafin::provider_internal::bind_qwen_roster(
        *request, shape, session->selected.device);
    auto staged = std::make_unique<deltafin::provider_internal::QwenModel>(
        shape, std::move(weights));
    const DeltafinProviderQwenHandleV1 handle = session->allocate_resource();
    const auto [ignored, inserted] =
        session->qwen_models.emplace(handle, std::move(staged));
    static_cast<void>(ignored);
    if (!inserted) throw std::runtime_error("Qwen model handle collision");
    DeltafinProviderQwenReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.model = handle;
    produced.variant = request->variant;
    produced.tensor_count = DELTAFIN_PROVIDER_QWEN_TENSOR_COUNT_V1;
    *report = produced;
  });
}

extern "C" int32_t deltafin_provider_qwen_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    require_resource_request(request, "Qwen destroy request");
    const auto session = find_session(request->session);
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->qwen_models.erase(request->resource) != 1) {
      throw std::invalid_argument("Qwen model handle is stale or unknown");
    }
  });
}

extern "C" int32_t deltafin_provider_qwen_generate_v1(
    const DeltafinProviderQwenGenerateV1* request,
    DeltafinProviderQwenGenerationReportV1* report, char* error,
    const size_t error_capacity) {
  return ffi_guard(error, error_capacity, [&] {
    if (request == nullptr || report == nullptr) {
      throw std::invalid_argument("Qwen generation request/report is null");
    }
    require_header(request->struct_size, sizeof(*request), request->abi_version,
                   "Qwen generation request");
    if (report->struct_size != sizeof(*report) || request->flags != 0 ||
        !all_zero(request->reserved)) {
      throw std::invalid_argument("Qwen generation fields are invalid");
    }
    const auto session = find_session(request->session);
    const c10::InferenceMode inference_guard;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->require_open();
    const auto found = session->qwen_models.find(request->model);
    if (found == session->qwen_models.end()) {
      throw std::invalid_argument("Qwen model handle is stale or unknown");
    }
    const auto generation = found->second->generate(
        request->input_token_ids,
        static_cast<std::size_t>(request->input_token_count),
        static_cast<std::size_t>(request->max_new_tokens));
    if (generation.token_ids.size() != generation.probabilities.size() ||
        generation.token_ids.size() >
            DELTAFIN_PROVIDER_QWEN_MAX_PROPOSAL_TOKENS_V1) {
      throw std::runtime_error("Qwen generation violated fixed report bounds");
    }
    DeltafinProviderQwenGenerationReportV1 produced = {};
    produced.struct_size = sizeof(produced);
    produced.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
    produced.generated_token_count =
        static_cast<std::uint32_t>(generation.token_ids.size());
    for (std::size_t index = 0; index < generation.token_ids.size(); ++index) {
      produced.token_ids[index] = generation.token_ids[index];
      produced.probabilities[index] = generation.probabilities[index];
    }
    *report = produced;
  });
}
