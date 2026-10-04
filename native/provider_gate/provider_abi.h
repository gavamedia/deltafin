#ifndef DELTAFIN_PROVIDER_ABI_H
#define DELTAFIN_PROVIDER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DELTAFIN_PROVIDER_ABI_VERSION 1u

/*
 * The split-layer mailbox is deliberately fixed-size. Large prefills are
 * submitted in bounded chunks; decode normally uses one position. Keeping
 * every array in caller-owned ABI storage means no std::vector, Tensor, or
 * allocator ownership can cross the C boundary.
 */
#define DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 16u
#define DELTAFIN_PROVIDER_PILOT_MAX_PREFETCH_V1 32u
#define DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1 64u
#define DELTAFIN_PROVIDER_ROUTE_MAX_EDGES_V1 \
  (DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 * \
   DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1)
#define DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1 64u
#define DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V2 256u
#define DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_TILE_ROWS_V1 16u

enum DeltafinProviderDeviceV1 {
  DELTAFIN_PROVIDER_DEVICE_AUTO_V1 = 0,
  DELTAFIN_PROVIDER_DEVICE_CPU_V1 = 1,
  DELTAFIN_PROVIDER_DEVICE_MPS_V1 = 2,
  DELTAFIN_PROVIDER_DEVICE_CUDA_V1 = 3,
};

enum DeltafinProviderCheckV1 {
  DELTAFIN_PROVIDER_CHECK_RMS_FP32_V1 = 1u << 0,
  DELTAFIN_PROVIDER_CHECK_MATMUL_FP32_V1 = 1u << 1,
  DELTAFIN_PROVIDER_CHECK_SOFTMAX_FP32_V1 = 1u << 2,
  DELTAFIN_PROVIDER_CHECK_PACKED_INT8_FP32_V1 = 1u << 3,
};

enum DeltafinProviderCanaryFlagV1 {
  DELTAFIN_PROVIDER_REQUIRE_PACKED_INT8_V1 = 1u << 0,
};

enum DeltafinProviderFeatureV1 {
  /* NVCC kernel plus the session-owned CUDA expert adapter are both linked. */
  DELTAFIN_PROVIDER_FEATURE_CUDA_MOE_V1 = 1u << 0,
  /* Exact RAW_BF16 CUDA decode/GEMV kernel and carrier are both linked. */
  DELTAFIN_PROVIDER_FEATURE_CUDA_EXACT_BF16_V1 = 1u << 1,
};

enum DeltafinProviderSessionFlagV1 {
  /*
   * Deterministic small-shape ATen transaction used only to qualify ownership,
   * routing, cache-commit, and prepare/finish semantics. It can never be
   * mistaken for the separately implemented real K3 target session.
   */
  DELTAFIN_PROVIDER_SESSION_SYNTHETIC_SPLIT_V1 = 1u << 0,
  /* Fixed-size, zero-weight ABI transaction canary; never a model session. */
  DELTAFIN_PROVIDER_SESSION_SYNTHETIC_KDA_V1 = 1u << 1,
  /* Small synthetic MLA cache/decode transaction; never a model session. */
  DELTAFIN_PROVIDER_SESSION_SYNTHETIC_MLA_V1 = 1u << 2,
};

enum DeltafinProviderLayerFlagV1 {
  DELTAFIN_PROVIDER_LAYER_NO_FLAGS_V1 = 0u,
};

enum DeltafinProviderTargetExpertFlagV1 {
  /*
   * Permit the Metal bridge to retain no-copy wrappers after this synchronous
   * call. The caller must own every expert byte in a stable arena and must
   * call deltafin_provider_metal_expert_cache_flush_v1 before any named arena
   * allocation grows, retires, or is destroyed. With flags=0, the provider
   * preserves the original ABI contract and drops every wrapper before return.
   */
  DELTAFIN_PROVIDER_TARGET_EXPERT_RETAIN_METAL_WRAPPERS_V1 = 1u << 0,
};

enum DeltafinProviderSpineEncodingV1 {
  DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1 = 1u,
  DELTAFIN_PROVIDER_SPINE_RAW_F32_V1 = 2u,
  DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1 = 3u,
};

enum DeltafinProviderSpineBufferV1 {
  DELTAFIN_PROVIDER_SPINE_BUFFER_NONE_V1 = 0u,
  DELTAFIN_PROVIDER_SPINE_BUFFER_QUANTIZED_V1 = 1u,
  DELTAFIN_PROVIDER_SPINE_BUFFER_SCALES_V1 = 2u,
  DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1 = 3u,
};

enum DeltafinProviderSpineComponentV1 {
  DELTAFIN_PROVIDER_SPINE_COMPONENT_DATA_V1 = 1u,
  DELTAFIN_PROVIDER_SPINE_COMPONENT_AUXILIARY_V1 = 2u,
};

enum DeltafinProviderSpineScalarTypeV1 {
  DELTAFIN_PROVIDER_SPINE_SCALAR_I8_V1 = 1u,
  DELTAFIN_PROVIDER_SPINE_SCALAR_F32_V1 = 2u,
  DELTAFIN_PROVIDER_SPINE_SCALAR_BF16_V1 = 3u,
};

enum DeltafinProviderBindSpineFlagV1 {
  /*
   * Retain this layer for the lifetime of the session. Retained layers form
   * an immutable ordered prefix beginning at layer zero. A bind without this
   * flag replaces only the session's single transient layer slot.
   */
  DELTAFIN_PROVIDER_BIND_SPINE_RETAIN_V1 = 1u << 0,
  /*
   * V2-only opt-in. The caller must transfer ownership of every source slab
   * into its source-use controller before making this request. V1 and the
   * by-reference Rust compatibility API never set this bit.
   */
  DELTAFIN_PROVIDER_BIND_SPINE_ALLOW_BORROW_V2 = 1u << 1,
};

/*
 * V2 extends the bind boundary without changing or weakening V1. Detached
 * means the provider has copied/converted every source byte before returning.
 * Borrowed means the provider may still read the caller's aligned slabs and
 * the caller must retain them until the consume-once source-use handle is
 * sealed and reclaimed (or synchronously aborted).
 */
enum DeltafinProviderSpineSourceUseKindV2 {
  DELTAFIN_PROVIDER_SPINE_SOURCE_DETACHED_V2 = 1u,
  DELTAFIN_PROVIDER_SPINE_SOURCE_BORROWED_V2 = 2u,
};

enum DeltafinProviderSpineSourceUseStateV2 {
  DELTAFIN_PROVIDER_SPINE_SOURCE_OPEN_V2 = 1u,
  DELTAFIN_PROVIDER_SPINE_SOURCE_SEALED_V2 = 2u,
  DELTAFIN_PROVIDER_SPINE_SOURCE_RECLAIMED_V2 = 3u,
  DELTAFIN_PROVIDER_SPINE_SOURCE_ABORTED_V2 = 4u,
};

typedef struct DeltafinProviderInventoryV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t cpu_available;
  uint32_t mps_available;
  uint32_t cuda_device_count;
  uint32_t provider_features;
  uint32_t reserved[10];
  char libtorch_version[32];
} DeltafinProviderInventoryV1;

typedef struct DeltafinProviderCanaryRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t requested_device;
  uint32_t device_index;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t packed_rows;
  uint64_t packed_columns;
  uint64_t reserved[6];
} DeltafinProviderCanaryRequestV1;

typedef struct DeltafinProviderCanaryReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t selected_device;
  uint32_t device_index;
  uint32_t attempted_checks;
  uint32_t passed_checks;
  uint32_t required_passed;
  uint32_t reserved0;
  uint64_t packed_rows;
  uint64_t packed_columns;
  uint64_t reserved[6];
  char detail[1024];
} DeltafinProviderCanaryReportV1;

/* Opaque integer IDs are namespaced by their owning session. Zero is invalid. */
typedef uint64_t DeltafinProviderSessionHandleV1;
typedef uint64_t DeltafinProviderTensorHandleV1;
typedef uint64_t DeltafinProviderCacheHandleV1;
typedef uint64_t DeltafinProviderLayerTicketV1;
typedef uint64_t DeltafinProviderKdaCacheHandleV1;
typedef uint64_t DeltafinProviderKdaTicketHandleV1;
typedef uint64_t DeltafinProviderMlaCacheHandleV1;
typedef uint64_t DeltafinProviderMlaTicketHandleV1;
typedef uint64_t DeltafinProviderTargetPositionHandleV1;
typedef uint64_t DeltafinProviderTargetSequenceHandleV1;
typedef uint64_t DeltafinProviderMoePlanHandleV1;
typedef uint64_t DeltafinProviderTargetStateBranchHandleV1;
typedef uint64_t DeltafinProviderDSparkHandleV1;
typedef uint64_t DeltafinProviderDSparkSnapshotHandleV1;
typedef uint64_t DeltafinProviderQwenHandleV1;
typedef uint64_t DeltafinProviderEagle3HandleV1;
typedef uint64_t DeltafinProviderEagle3SnapshotHandleV1;
typedef uint64_t DeltafinProviderSpineSourceUseHandleV2;

#define DELTAFIN_PROVIDER_DSPARK_TENSOR_COUNT_V1 67u
#define DELTAFIN_PROVIDER_DSPARK_QUERY_ROWS_V1 7u

enum DeltafinProviderDSparkCreateFlagV1 {
  /* Explicitly model-free compact geometry for native/Rust ABI tests only. */
  DELTAFIN_PROVIDER_DSPARK_SYNTHETIC_V1 = 1u << 0,
};

enum DeltafinProviderDSparkScalarTypeV1 {
  DELTAFIN_PROVIDER_DSPARK_BF16_V1 = 1u,
};

typedef struct DeltafinProviderDSparkTensorV1 {
  uint32_t slot;
  uint32_t scalar_type;
  uint32_t rank;
  uint32_t flags;
  uint64_t shape[2];
  const uint8_t* data;
  uint64_t data_length;
  uint64_t reserved[2];
} DeltafinProviderDSparkTensorV1;

typedef struct DeltafinProviderDSparkCreateV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t flags;
  uint32_t tensor_count;
  const DeltafinProviderDSparkTensorV1* tensors;
  const float* synthetic_head_f32;
  uint64_t synthetic_head_elements;
  uint64_t reserved[5];
} DeltafinProviderDSparkCreateV1;

typedef struct DeltafinProviderDSparkReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderDSparkHandleV1 model;
  uint64_t cache_length;
  uint64_t cache_generation;
  uint32_t tensor_count;
  uint32_t flags;
  uint64_t reserved[3];
} DeltafinProviderDSparkReportV1;

typedef struct DeltafinProviderDSparkAppendV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderDSparkHandleV1 model;
  const uint8_t* target_context_bf16;
  uint64_t target_context_bytes;
  const int64_t* positions;
  uint64_t rows;
  uint64_t reserved[4];
} DeltafinProviderDSparkAppendV1;

/*
 * Zero-hop target-state advance. `target_context` names a provider-owned,
 * contiguous BF16 [rows,5*H] tensor in the same session/device as `model`.
 * The expected cache coordinates make the handle handoff one atomic state
 * transition rather than a shape-only append.
 */
typedef struct DeltafinProviderDSparkAppendTensorV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderDSparkHandleV1 model;
  DeltafinProviderTensorHandleV1 target_context;
  uint64_t expected_cache_length;
  uint64_t expected_cache_generation;
  uint64_t rows;
  uint64_t reserved[3];
} DeltafinProviderDSparkAppendTensorV1;

typedef struct DeltafinProviderDSparkSnapshotReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderDSparkSnapshotHandleV1 snapshot;
  uint64_t cache_length;
  uint64_t cache_generation;
  uint64_t reserved[4];
} DeltafinProviderDSparkSnapshotReportV1;

typedef struct DeltafinProviderDSparkRestoreV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderDSparkHandleV1 model;
  DeltafinProviderDSparkSnapshotHandleV1 snapshot;
  uint64_t reserved[4];
} DeltafinProviderDSparkRestoreV1;

typedef struct DeltafinProviderDSparkProposeV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderDSparkHandleV1 model;
  uint32_t anchor_token_id;
  uint32_t score_rows;
  const uint8_t* query_embeddings_bf16;
  uint64_t query_embedding_bytes;
  uint64_t reserved[4];
} DeltafinProviderDSparkProposeV1;

typedef struct DeltafinProviderDSparkProposalReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t score_rows;
  uint32_t flags;
  uint64_t anchor_position;
  uint64_t cache_generation;
  uint32_t token_ids[DELTAFIN_PROVIDER_DSPARK_QUERY_ROWS_V1];
  float confidence_logits[DELTAFIN_PROVIDER_DSPARK_QUERY_ROWS_V1];
  uint64_t reserved[4];
} DeltafinProviderDSparkProposalReportV1;

/*
 * EAGLE-3.1 single-layer MLA proposal model. Its 19 owned BF16 tensors use
 * DeltafinProviderDSparkTensorV1 descriptors with the slots below; the token
 * embedding is K3's own exact BF16 table, which the caller reads and passes
 * per call, so no copy of it lives in the provider.
 */
#define DELTAFIN_PROVIDER_EAGLE3_TENSOR_COUNT_V1 19u
#define DELTAFIN_PROVIDER_EAGLE3_MAX_DRAFTS_V1 7u

enum DeltafinProviderEagle3CreateFlagV1 {
  /* Explicitly model-free compact geometry for native/Rust ABI tests only. */
  DELTAFIN_PROVIDER_EAGLE3_SYNTHETIC_V1 = 1u << 0,
};

enum DeltafinProviderEagle3SlotV1 {
  DELTAFIN_PROVIDER_EAGLE3_FC_V1 = 1u,
  DELTAFIN_PROVIDER_EAGLE3_FC_NORM0_V1 = 2u,
  DELTAFIN_PROVIDER_EAGLE3_FC_NORM1_V1 = 3u,
  DELTAFIN_PROVIDER_EAGLE3_FC_NORM2_V1 = 4u,
  DELTAFIN_PROVIDER_EAGLE3_HIDDEN_NORM_V1 = 5u,
  DELTAFIN_PROVIDER_EAGLE3_INPUT_NORM_V1 = 6u,
  DELTAFIN_PROVIDER_EAGLE3_QUERY_A_V1 = 7u,
  DELTAFIN_PROVIDER_EAGLE3_QUERY_A_NORM_V1 = 8u,
  DELTAFIN_PROVIDER_EAGLE3_QUERY_B_V1 = 9u,
  DELTAFIN_PROVIDER_EAGLE3_KEY_VALUE_A_V1 = 10u,
  DELTAFIN_PROVIDER_EAGLE3_KEY_VALUE_A_NORM_V1 = 11u,
  DELTAFIN_PROVIDER_EAGLE3_KEY_VALUE_B_V1 = 12u,
  DELTAFIN_PROVIDER_EAGLE3_OUTPUT_V1 = 13u,
  DELTAFIN_PROVIDER_EAGLE3_POST_ATTENTION_NORM_V1 = 14u,
  DELTAFIN_PROVIDER_EAGLE3_GATE_V1 = 15u,
  DELTAFIN_PROVIDER_EAGLE3_UP_V1 = 16u,
  DELTAFIN_PROVIDER_EAGLE3_DOWN_V1 = 17u,
  DELTAFIN_PROVIDER_EAGLE3_FINAL_NORM_V1 = 18u,
  DELTAFIN_PROVIDER_EAGLE3_HEAD_V1 = 19u,
};

typedef struct DeltafinProviderEagle3CreateV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t flags;
  uint32_t tensor_count;
  const DeltafinProviderDSparkTensorV1* tensors;
  uint64_t max_positions;
  uint64_t reserved[5];
} DeltafinProviderEagle3CreateV1;

typedef struct DeltafinProviderEagle3ReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderEagle3HandleV1 model;
  uint64_t token_count;
  uint64_t cache_length;
  uint64_t max_positions;
  uint32_t flags;
  uint32_t proposing;
  uint64_t reserved[3];
} DeltafinProviderEagle3ReportV1;

/*
 * Advance by committed K3 rows. `target_rows` names the provider-owned BF16
 * [>=rows,3H] tensor a target sequence captured with
 * DELTAFIN_PROVIDER_TARGET_SEQUENCE_CAPTURE_EAGLE3_V1; the embeddings are the
 * exact BF16 rows of the `rows` tokens those positions consumed.
 */
typedef struct DeltafinProviderEagle3AdvanceV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderEagle3HandleV1 model;
  DeltafinProviderTensorHandleV1 target_rows;
  uint64_t rows;
  uint64_t expected_token_count;
  const uint8_t* input_embeddings_bf16;
  uint64_t input_embedding_bytes;
  uint64_t reserved[4];
} DeltafinProviderEagle3AdvanceV1;

enum DeltafinProviderEagle3StepPhaseV1 {
  /* The embedding names the token K3 just produced for the pending row. */
  DELTAFIN_PROVIDER_EAGLE3_STEP_BEGIN_V1 = 1u,
  /* The embedding names the previous draft. */
  DELTAFIN_PROVIDER_EAGLE3_STEP_CONTINUE_V1 = 2u,
  /* Discard the chain's speculative rows; no embedding. */
  DELTAFIN_PROVIDER_EAGLE3_STEP_END_V1 = 3u,
};

typedef struct DeltafinProviderEagle3StepV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderEagle3HandleV1 model;
  uint32_t phase;
  uint32_t reserved32;
  const uint8_t* embedding_bf16;
  uint64_t embedding_bytes;
  uint64_t reserved[4];
} DeltafinProviderEagle3StepV1;

typedef struct DeltafinProviderEagle3StepReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  /* Unverified greedy proposal; zero after END. */
  uint32_t token_id;
  uint32_t drafts;
  uint64_t reserved[4];
} DeltafinProviderEagle3StepReportV1;

typedef struct DeltafinProviderEagle3SnapshotReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderEagle3SnapshotHandleV1 snapshot;
  uint64_t token_count;
  uint64_t reserved[4];
} DeltafinProviderEagle3SnapshotReportV1;

typedef struct DeltafinProviderEagle3RestoreV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderEagle3HandleV1 model;
  DeltafinProviderEagle3SnapshotHandleV1 snapshot;
  uint64_t reserved[4];
} DeltafinProviderEagle3RestoreV1;

/* Stateless, proposal-only Qwen3 assistant. It has no target-state operation. */
#define DELTAFIN_PROVIDER_QWEN_TENSOR_COUNT_V1 310u
#define DELTAFIN_PROVIDER_QWEN_MAX_PROPOSAL_TOKENS_V1 20u

enum DeltafinProviderQwenVariantV1 {
  DELTAFIN_PROVIDER_QWEN_06B_V1 = 1u,
  DELTAFIN_PROVIDER_QWEN_17B_V1 = 2u,
};

typedef struct DeltafinProviderQwenTensorV1 {
  uint32_t slot;
  uint32_t scalar_type;
  uint32_t rank;
  uint32_t flags;
  uint64_t shape[2];
  const uint8_t* data;
  uint64_t data_length;
  uint64_t reserved[2];
} DeltafinProviderQwenTensorV1;

typedef struct DeltafinProviderQwenCreateV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t variant;
  uint32_t tensor_count;
  const DeltafinProviderQwenTensorV1* tensors;
  uint64_t reserved[6];
} DeltafinProviderQwenCreateV1;

typedef struct DeltafinProviderQwenReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderQwenHandleV1 model;
  uint32_t variant;
  uint32_t tensor_count;
  uint64_t reserved[5];
} DeltafinProviderQwenReportV1;

typedef struct DeltafinProviderQwenGenerateV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderQwenHandleV1 model;
  const uint32_t* input_token_ids;
  uint64_t input_token_count;
  uint32_t max_new_tokens;
  uint32_t flags;
  uint64_t reserved[4];
} DeltafinProviderQwenGenerateV1;

typedef struct DeltafinProviderQwenGenerationReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t generated_token_count;
  uint32_t flags;
  uint32_t token_ids[DELTAFIN_PROVIDER_QWEN_MAX_PROPOSAL_TOKENS_V1];
  float probabilities[DELTAFIN_PROVIDER_QWEN_MAX_PROPOSAL_TOKENS_V1];
  uint64_t reserved[4];
} DeltafinProviderQwenGenerationReportV1;

typedef struct DeltafinProviderSessionRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t requested_device;
  uint32_t device_index;
  uint32_t flags;
  uint32_t max_route_positions;
  uint32_t synthetic_hidden_columns;
  uint32_t synthetic_experts;
  uint64_t reserved[6];
} DeltafinProviderSessionRequestV1;

typedef struct DeltafinProviderSessionReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t selected_device;
  uint32_t device_index;
  DeltafinProviderSessionHandleV1 session;
  uint32_t max_route_positions;
  uint32_t flags;
  uint64_t reserved[6];
} DeltafinProviderSessionReportV1;

/*
 * A live allocator/device snapshot owned by the selected provider. Querying
 * creates no provider tensor/resource and retains no caller memory (the
 * backend may lazily initialize its own metadata on first access).
 * TRIM_UNUSED first drains the selected accelerator and returns only
 * allocator blocks which no live tensor owns; it is permitted only at a
 * quiescent session boundary. Field-presence bits prevent zero from being
 * confused with an unavailable provider query.
 */
enum DeltafinProviderMemoryActionV1 {
  DELTAFIN_PROVIDER_MEMORY_TRIM_UNUSED_V1 = 1u << 0,
};

enum DeltafinProviderMemoryFieldV1 {
  DELTAFIN_PROVIDER_MEMORY_ACTIVE_BYTES_V1 = 1u << 0,
  DELTAFIN_PROVIDER_MEMORY_RESERVED_BYTES_V1 = 1u << 1,
  DELTAFIN_PROVIDER_MEMORY_RECOMMENDED_BYTES_V1 = 1u << 2,
  DELTAFIN_PROVIDER_MEMORY_TOTAL_BYTES_V1 = 1u << 3,
  DELTAFIN_PROVIDER_MEMORY_AVAILABLE_BYTES_V1 = 1u << 4,
};

typedef struct DeltafinProviderMemoryRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t actions;
  uint32_t reserved0;
  uint64_t reserved[5];
} DeltafinProviderMemoryRequestV1;

typedef struct DeltafinProviderMemoryReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t selected_device;
  uint32_t device_index;
  uint32_t available_fields;
  uint32_t performed_actions;
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t active_bytes;
  uint64_t reserved_bytes;
  uint64_t recommended_bytes;
  uint64_t total_bytes;
  uint64_t available_bytes;
  uint64_t reserved[4];
} DeltafinProviderMemoryReportV1;

/*
 * One-shot admission for the CPU/Metal scheduling-only router roster. The
 * caller must charge this complete upper bound before enabling it and must do
 * so before any layer/global bind. Sessions that never call the function keep
 * zero PILOT roster bytes. CUDA uses its separate contiguous residency plan.
 */
#define DELTAFIN_PROVIDER_TARGET_PILOT_LAYER_CAPACITY_V1 92u
#define DELTAFIN_PROVIDER_TARGET_PILOT_RESERVE_BYTES_V1 594169856ull

typedef struct DeltafinProviderTargetPilotEnableReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t enabled;
  uint32_t layer_capacity;
  uint64_t reserve_bytes;
  uint64_t reserved[4];
} DeltafinProviderTargetPilotEnableReportV1;

enum DeltafinProviderCudaCacheCapacityModeV1 {
  DELTAFIN_PROVIDER_CUDA_CACHE_CAPACITY_AUTO_V1 = 0u,
  DELTAFIN_PROVIDER_CUDA_CACHE_CAPACITY_EXACT_V1 = 1u,
};

enum DeltafinProviderCudaCacheReserveModeV1 {
  DELTAFIN_PROVIDER_CUDA_CACHE_RESERVE_AUTO_V1 = 0u,
  DELTAFIN_PROVIDER_CUDA_CACHE_RESERVE_BYTES_V1 = 1u,
  DELTAFIN_PROVIDER_CUDA_CACHE_RESERVE_RATIO_PPM_V1 = 2u,
};

typedef struct DeltafinProviderCudaCacheConfigureRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t capacity_mode;
  uint32_t reserve_mode;
  uint64_t capacity_experts;
  uint64_t reserve_value;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[2];
} DeltafinProviderCudaCacheConfigureRequestV1;

typedef struct DeltafinProviderCudaCacheConfigureReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t capacity_mode;
  uint32_t reserve_mode;
  uint64_t capacity_experts;
  uint64_t reserve_value;
  uint32_t configured;
  uint32_t flags;
  uint64_t reserved[2];
} DeltafinProviderCudaCacheConfigureReportV1;

/*
 * Session-scoped qualification of the statically linked Metal expert bridge.
 * The source path is borrowed for this synchronous call. The report exposes
 * only layouts backed by the complete versioned descriptor symbol/pipeline
 * suite; raw-v1 remains the sole layout on every other provider.
 */
typedef struct DeltafinProviderMetalExpertLayoutsRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  const char* metal_shader_path;
  uint64_t metal_shader_path_length;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[3];
} DeltafinProviderMetalExpertLayoutsRequestV1;

typedef struct DeltafinProviderMetalExpertLayoutsReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t descriptor_abi;
  uint32_t flags;
  uint64_t layout_capabilities;
  uint64_t raw_span_bytes;
  uint64_t scale4_span_bytes;
  uint64_t reserved[2];
} DeltafinProviderMetalExpertLayoutsReportV1;

typedef struct DeltafinProviderMetalExpertCacheStatsReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint64_t calls;
  uint64_t zero_copy_wraps;
  uint64_t copies;
  uint64_t cache_entries;
  uint64_t bindless;
  uint64_t reserved[2];
} DeltafinProviderMetalExpertCacheStatsReportV1;

typedef struct DeltafinProviderTensorUploadF32V1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint64_t rows;
  uint64_t columns;
  const float* data;
  uint64_t element_count;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[3];
} DeltafinProviderTensorUploadF32V1;

/* Raw BF16 upload exists for ABI canaries and data-source boundaries. Runtime
 * target activations use provider-owned tensor handles and do not call it. */
typedef struct DeltafinProviderTensorUploadBf16V1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint64_t rows;
  uint64_t columns;
  const uint8_t* data;
  uint64_t byte_length;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[3];
} DeltafinProviderTensorUploadBf16V1;

typedef struct DeltafinProviderTensorReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTensorHandleV1 tensor;
  uint64_t rows;
  uint64_t columns;
  uint64_t reserved[4];
} DeltafinProviderTensorReportV1;

typedef struct DeltafinProviderTensorReadF32V1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTensorHandleV1 tensor;
  float* destination;
  uint64_t element_capacity;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[3];
} DeltafinProviderTensorReadF32V1;

typedef struct DeltafinProviderCacheCreateF32V1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint64_t rows;
  uint64_t columns;
  const float* initial_data;
  uint64_t element_count;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[3];
} DeltafinProviderCacheCreateF32V1;

typedef struct DeltafinProviderCacheReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderCacheHandleV1 cache;
  uint64_t rows;
  uint64_t columns;
  uint64_t version;
  uint64_t reserved[3];
} DeltafinProviderCacheReportV1;

typedef struct DeltafinProviderCacheReadF32V1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderCacheHandleV1 cache;
  float* destination;
  uint64_t element_capacity;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[3];
} DeltafinProviderCacheReadF32V1;

typedef struct DeltafinProviderResourceRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint64_t resource;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[4];
} DeltafinProviderResourceRequestV1;

typedef struct DeltafinProviderPrepareLayerRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTensorHandleV1 hidden;
  DeltafinProviderCacheHandleV1 cache;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t reserved[5];
} DeltafinProviderPrepareLayerRequestV1;

typedef struct DeltafinProviderRouteMailboxV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderLayerTicketV1 ticket;
  uint32_t positions;
  uint32_t top_k;
  uint32_t edge_count;
  uint32_t flags;
  uint64_t hidden_columns;
  uint64_t cache_version;
  uint64_t reserved[4];
  uint16_t ordered_experts[DELTAFIN_PROVIDER_ROUTE_MAX_EDGES_V1];
  uint32_t ordered_weight_bits[DELTAFIN_PROVIDER_ROUTE_MAX_EDGES_V1];
} DeltafinProviderRouteMailboxV1;

typedef struct DeltafinProviderFinishLayerRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderLayerTicketV1 ticket;
  DeltafinProviderTensorHandleV1 expert_output;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[5];
} DeltafinProviderFinishLayerRequestV1;

typedef struct DeltafinProviderFinishLayerReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTensorHandleV1 output;
  uint64_t positions;
  uint64_t hidden_columns;
  uint64_t committed_cache_version;
  uint64_t reserved[5];
} DeltafinProviderFinishLayerReportV1;

/*
 * This layout is shared exactly with Rust's SpineTensorDescriptorV1. It never
 * contains a pointer: every byte range is relative to one of the three
 * caller-owned LayerBuffers slabs named by DeltafinProviderSpineBufferV1.
 */
typedef struct DeltafinProviderSpineTensorDescriptorV1 {
  uint32_t slot;
  uint32_t encoding;
  uint32_t rank;
  uint32_t data_buffer;
  uint32_t auxiliary_buffer;
  uint32_t reserved0;
  uint64_t shape[8];
  uint64_t data_offset;
  uint64_t data_length;
  uint64_t auxiliary_offset;
  uint64_t auxiliary_length;
  uint64_t reserved[4];
} DeltafinProviderSpineTensorDescriptorV1;

typedef struct DeltafinProviderBindSpineLayerRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t generation;
  const DeltafinProviderSpineTensorDescriptorV1* descriptors;
  uint64_t descriptor_count;
  const uint8_t* quantized;
  uint64_t quantized_length;
  const uint8_t* scales;
  uint64_t scales_length;
  const uint8_t* other;
  uint64_t other_length;
  uint64_t reserved[4];
} DeltafinProviderBindSpineLayerRequestV1;

typedef struct DeltafinProviderBindSpineLayerReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t layer_index;
  uint32_t tensor_count;
  uint64_t generation;
  uint32_t quantized_tensor_count;
  uint32_t raw_tensor_count;
  uint64_t quantized_bytes;
  uint64_t scales_bytes;
  uint64_t other_bytes;
  uint64_t resident_storage_bytes;
  uint64_t reserved[4];
} DeltafinProviderBindSpineLayerReportV1;

/*
 * Borrow-capable spine binding. Each pointer is described by both its logical
 * manifest length and the allocator's actual rounded backing length. A future
 * Metal/CUDA/Windows provider may borrow only within that allocation envelope.
 * V1 remains available and retains its synchronous detached semantics. A
 * failed V2 bind must cancel or complete every source access before returning;
 * only a successful report may transfer a live borrowed handle to the caller.
 */
typedef struct DeltafinProviderBindSpineLayerRequestV2 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t generation;
  const DeltafinProviderSpineTensorDescriptorV1* descriptors;
  uint64_t descriptor_count;
  const uint8_t* quantized;
  uint64_t quantized_length;
  uint64_t quantized_allocation_length;
  const uint8_t* scales;
  uint64_t scales_length;
  uint64_t scales_allocation_length;
  const uint8_t* other;
  uint64_t other_length;
  uint64_t other_allocation_length;
  uint64_t reserved[5];
} DeltafinProviderBindSpineLayerRequestV2;

typedef struct DeltafinProviderBindSpineLayerReportV2 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t layer_index;
  uint32_t tensor_count;
  uint64_t generation;
  uint32_t quantized_tensor_count;
  uint32_t raw_tensor_count;
  uint64_t quantized_bytes;
  uint64_t scales_bytes;
  uint64_t other_bytes;
  uint64_t resident_storage_bytes;
  uint32_t source_use_kind;
  uint32_t borrowed_tensor_count;
  DeltafinProviderSpineSourceUseHandleV2 source_use;
  uint64_t borrowed_source_bytes;
  uint64_t reserved[3];
} DeltafinProviderBindSpineLayerReportV2;

typedef struct DeltafinProviderSpineSourceUseRequestV2 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderSpineSourceUseHandleV2 source_use;
  uint64_t generation;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[3];
} DeltafinProviderSpineSourceUseRequestV2;

typedef struct DeltafinProviderSpineSourceUseReportV2 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSpineSourceUseHandleV2 source_use;
  uint64_t generation;
  uint32_t state;
  uint32_t ready;
  uint64_t reserved[4];
} DeltafinProviderSpineSourceUseReportV2;

/*
 * Seal forbids submission of any new work against the source. Reclaim is a
 * non-blocking poll and consumes the handle only when ready=1. Abort is
 * synchronous and consumes the handle only on success; after any failed
 * control call the caller must continue retaining the source allocation.
 */

/*
 * A bounded parity/debug read. The provider converts the selected component
 * to fp32 in caller storage during this call and retains no destination
 * pointer. Production layer execution consumes the provider-owned tensors
 * directly and does not use this endpoint.
 */
typedef struct DeltafinProviderSpineTensorReadF32V1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint64_t generation;
  uint32_t slot;
  uint32_t component;
  float* destination;
  uint64_t element_capacity;
  uint32_t flags;
  uint32_t layer_index;
  uint64_t reserved[3];
} DeltafinProviderSpineTensorReadF32V1;

typedef struct DeltafinProviderSpineTensorReadReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t stored_scalar_type;
  uint32_t rank;
  uint64_t element_count;
  uint64_t shape[8];
  uint64_t reserved[1];
} DeltafinProviderSpineTensorReadReportV1;

/*
 * Provider-owned KDA state.  A cache contains all three width-four causal
 * convolution histories plus the [1,96,128,128] recurrent state.  Decode is
 * transactional: it returns an output and a ticket containing new state;
 * commit publishes that state, while release discards it.
 */
typedef struct DeltafinProviderKdaCacheCreateV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t reserved[5];
} DeltafinProviderKdaCacheCreateV1;

typedef struct DeltafinProviderKdaCacheReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderKdaCacheHandleV1 cache;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t version;
  uint64_t convolution_elements;
  uint64_t recurrent_elements;
  uint64_t reserved[2];
} DeltafinProviderKdaCacheReportV1;

typedef struct DeltafinProviderKdaDecodeRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTensorHandleV1 hidden;
  DeltafinProviderKdaCacheHandleV1 cache;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t spine_generation;
  uint64_t reserved[4];
} DeltafinProviderKdaDecodeRequestV1;

typedef struct DeltafinProviderKdaDecodeReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTensorHandleV1 output;
  DeltafinProviderKdaTicketHandleV1 ticket;
  uint64_t cache_version;
  uint64_t spine_generation;
  uint64_t rows;
  uint64_t columns;
  uint64_t reserved[3];
} DeltafinProviderKdaDecodeReportV1;

typedef struct DeltafinProviderKdaCommitReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderKdaCacheHandleV1 cache;
  uint64_t committed_version;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t reserved[4];
} DeltafinProviderKdaCommitReportV1;

/*
 * Provider-owned expanded MLA KV state. Decode stages one new cache position
 * and returns an output plus a ticket. Commit alone publishes the position;
 * releasing the ticket cancels it. input_bundle_rows is zero on the reviewed
 * three-call fallback and nonzero when the bind-time same-input bundle ran.
 */
typedef struct DeltafinProviderMlaCacheCreateV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t reserved[5];
} DeltafinProviderMlaCacheCreateV1;

typedef struct DeltafinProviderMlaCacheReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderMlaCacheHandleV1 cache;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t version;
  uint64_t length;
  uint64_t capacity;
  uint64_t reserved[2];
} DeltafinProviderMlaCacheReportV1;

typedef struct DeltafinProviderMlaDecodeRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTensorHandleV1 hidden;
  DeltafinProviderMlaCacheHandleV1 cache;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t spine_generation;
  uint64_t reserved[4];
} DeltafinProviderMlaDecodeRequestV1;

typedef struct DeltafinProviderMlaDecodeReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTensorHandleV1 output;
  DeltafinProviderMlaTicketHandleV1 ticket;
  uint64_t cache_version;
  uint64_t spine_generation;
  uint64_t rows;
  uint64_t columns;
  uint64_t proposed_length;
  uint64_t proposed_capacity;
  uint64_t input_bundle_rows;
  uint64_t reserved[2];
} DeltafinProviderMlaDecodeReportV1;

typedef struct DeltafinProviderMlaCommitReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderMlaCacheHandleV1 cache;
  uint64_t committed_version;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t committed_length;
  uint64_t capacity;
  uint64_t reserved[2];
} DeltafinProviderMlaCommitReportV1;

enum DeltafinProviderTargetGlobalGroupV1 {
  /* Slots 41..43: final norm and the output residual mixer. */
  DELTAFIN_PROVIDER_TARGET_GLOBAL_TAIL_V1 = 1u,
  /* Slot 44 only: the row-int8/fp32-scale language-model head. */
  DELTAFIN_PROVIDER_TARGET_GLOBAL_HEAD_V1 = 2u,
};

enum DeltafinProviderTargetPrepareKindV1 {
  DELTAFIN_PROVIDER_TARGET_DENSE_COMPLETE_V1 = 1u,
  DELTAFIN_PROVIDER_TARGET_EXPERTS_REQUIRED_V1 = 2u,
};

enum DeltafinProviderTargetPositionStateV1 {
  DELTAFIN_PROVIDER_TARGET_ACTIVE_V1 = 1u,
  DELTAFIN_PROVIDER_TARGET_WAITING_FOR_EXPERTS_V1 = 2u,
  DELTAFIN_PROVIDER_TARGET_READY_FOR_TAIL_V1 = 3u,
  DELTAFIN_PROVIDER_TARGET_COMMITTED_V1 = 4u,
  DELTAFIN_PROVIDER_TARGET_CANCELLED_V1 = 5u,
  DELTAFIN_PROVIDER_TARGET_POISONED_V1 = 6u,
};

enum DeltafinProviderTargetExpertBackendV1 {
  DELTAFIN_PROVIDER_TARGET_EXPERT_AUTO_V1 = 0u,
  DELTAFIN_PROVIDER_TARGET_EXPERT_CPU_V1 = 1u,
  DELTAFIN_PROVIDER_TARGET_EXPERT_METAL_V1 = 2u,
  DELTAFIN_PROVIDER_TARGET_EXPERT_CUDA_V1 = 3u,
};

enum DeltafinProviderExpertStorageLayoutV1 {
  DELTAFIN_PROVIDER_EXPERT_LAYOUT_RAW_V1 = 1u,
  DELTAFIN_PROVIDER_EXPERT_LAYOUT_SCALE4_V2 = 2u,
};

/*
 * Immutable globals bind in two bounded transactions so the 1.17 GB head can
 * stream independently of the tiny tail. Slot 40 (the BF16 embedding table)
 * is intentionally absent: Rust supplies only the exact row needed to begin
 * a position. All caller buffers are borrowed solely for this call.
 */
typedef struct DeltafinProviderBindTargetGlobalsRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint32_t group;
  uint32_t flags;
  const DeltafinProviderSpineTensorDescriptorV1* descriptors;
  uint64_t descriptor_count;
  const uint8_t* quantized;
  uint64_t quantized_length;
  const uint8_t* scales;
  uint64_t scales_length;
  const uint8_t* other;
  uint64_t other_length;
  uint64_t reserved[5];
} DeltafinProviderBindTargetGlobalsRequestV1;

typedef struct DeltafinProviderBindTargetGlobalsReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t group;
  uint32_t tensor_count;
  uint32_t quantized_tensor_count;
  uint32_t raw_tensor_count;
  uint64_t quantized_bytes;
  uint64_t scales_bytes;
  uint64_t other_bytes;
  uint64_t resident_storage_bytes;
  uint32_t groups_ready;
  uint32_t flags;
  uint64_t reserved[4];
} DeltafinProviderBindTargetGlobalsReportV1;

typedef struct DeltafinProviderTargetBeginRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTensorHandleV1 hidden;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[4];
} DeltafinProviderTargetBeginRequestV1;

/* One-call hot path for an exact BF16 embedding row read by Rust. The
 * provider clones and promotes all 7168 elements before returning. */
typedef struct DeltafinProviderTargetBeginBf16RequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  const uint8_t* data;
  uint64_t byte_length;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[3];
} DeltafinProviderTargetBeginBf16RequestV1;

typedef struct DeltafinProviderTargetBeginReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetPositionHandleV1 position;
  uint32_t next_layer;
  uint32_t state;
  uint32_t kda_cache_count;
  uint32_t mla_cache_count;
  uint64_t reserved[4];
} DeltafinProviderTargetBeginReportV1;

typedef struct DeltafinProviderTargetPrepareRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetPositionHandleV1 position;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t reserved[3];
} DeltafinProviderTargetPrepareRequestV1;

typedef struct DeltafinProviderTargetPrepareReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetPositionHandleV1 position;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t kind;
  uint32_t next_layer;
  uint32_t top_k;
  uint16_t ordered_experts[DELTAFIN_PROVIDER_ROUTE_TOP_K_V1];
  uint32_t ordered_weight_bits[DELTAFIN_PROVIDER_ROUTE_TOP_K_V1];
  uint64_t reserved[3];
} DeltafinProviderTargetPrepareReportV1;

/* Expert IDs are canonical ascending IDs, not route order. Expert bytes are
 * borrowed synchronously. With flags=0 no pointer or Metal wrapper survives
 * this call; the explicit retain-wrapper flag uses the arena/flush lifetime
 * contract documented by DeltafinProviderTargetExpertFlagV1. */
typedef struct DeltafinProviderTargetFinishExpertsRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetPositionHandleV1 position;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t expert_backend;
  uint32_t cpu_threads;
  uint32_t expert_count;
  uint16_t expert_ids[DELTAFIN_PROVIDER_ROUTE_TOP_K_V1];
  const uint8_t* expert_major_bytes;
  uint64_t expert_major_length;
  const char* metal_shader_path;
  uint64_t metal_shader_path_length;
  uint32_t flags;
  uint32_t expert_layout;
  uint64_t expert_span_bytes;
  uint64_t reserved[4];
} DeltafinProviderTargetFinishExpertsRequestV1;

typedef struct DeltafinProviderTargetFinishExpertsReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetPositionHandleV1 position;
  uint32_t completed_layer;
  uint32_t next_layer;
  uint32_t state;
  uint32_t flags;
  uint64_t reserved[4];
} DeltafinProviderTargetFinishExpertsReportV1;

typedef struct DeltafinProviderTargetGreedyReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetPositionHandleV1 position;
  uint32_t token_id;
  uint32_t state;
  uint64_t committed_positions;
  uint64_t reserved[4];
} DeltafinProviderTargetGreedyReportV1;

enum DeltafinProviderTargetSequenceModeV1 {
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_PREFILL_V1 = 1u,
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_VERIFY_V1 = 2u,
};

enum DeltafinProviderTargetSequenceBeginFlagV1 {
  /*
   * Retain BF16 post-layer target rows at the five public DSpark boundaries.
   * They remain provider-owned and are exposed only as an opaque tensor for
   * proposal-cache advancement after full K3 finishes successfully.
   */
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_CAPTURE_DSPARK_V1 = 1u << 0,
  /*
   * Verify every requested row, but admit publication only when the caller
   * accepts the complete sequence.  This permits the provider to retain one
   * final KDA state per KDA layer instead of every intermediate prefix state.
   * The flag is invalid for PREFILL; partial sequence commits fail closed.
   */
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_FULL_COMMIT_ONLY_V1 = 1u << 1,
  /*
   * Retain BF16 post-layer target rows at the three EAGLE-3 capture layers
   * (zero-based 1, 45, 89; the within-block AttnRes prefix sum). Exclusive
   * with the DSpark capture; exposed through the same opaque rows tensor.
   */
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_CAPTURE_EAGLE3_V1 = 1u << 2,
};

enum DeltafinProviderTargetSequenceStateV1 {
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_ACTIVE_V1 = 1u,
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_WAITING_FOR_EXPERTS_V1 = 2u,
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_READY_FOR_TAIL_V1 = 3u,
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_READY_TO_COMMIT_V1 = 4u,
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_COMMITTED_V1 = 5u,
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_CANCELLED_V1 = 6u,
  DELTAFIN_PROVIDER_TARGET_SEQUENCE_POISONED_V1 = 7u,
};

/*
 * Coarse layer-major target transaction. The BF16 rows are copied before
 * begin returns; no caller pointer survives the call. Positions are bounded
 * by DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1.
 */
typedef struct DeltafinProviderTargetSequenceBeginBf16RequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  const uint8_t* data;
  uint64_t byte_length;
  uint32_t positions;
  uint32_t mode;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[4];
} DeltafinProviderTargetSequenceBeginBf16RequestV1;

typedef struct DeltafinProviderTargetSequenceBeginReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint32_t positions;
  uint32_t mode;
  uint32_t next_layer;
  uint32_t state;
  uint32_t kda_cache_count;
  uint32_t mla_cache_count;
  uint64_t reserved[3];
} DeltafinProviderTargetSequenceBeginReportV1;

typedef struct DeltafinProviderTargetSequencePrepareRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t flags;
  uint64_t reserved[3];
} DeltafinProviderTargetSequencePrepareRequestV1;

/*
 * Fixed route mailbox. For each row, the 16 entries beginning at
 * row*DELTAFIN_PROVIDER_ROUTE_TOP_K_V1 preserve provider route order and raw
 * fp32 weight bits. Routed activations remain private native tensor handles.
 */
typedef struct DeltafinProviderTargetSequencePrepareReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t kind;
  uint32_t next_layer;
  uint32_t positions;
  uint32_t top_k;
  uint32_t flags;
  uint16_t ordered_experts[DELTAFIN_PROVIDER_ROUTE_MAX_EDGES_V1];
  uint32_t ordered_weight_bits[DELTAFIN_PROVIDER_ROUTE_MAX_EDGES_V1];
  uint64_t reserved[4];
} DeltafinProviderTargetSequencePrepareReportV1;

/* Optional scheduling-only next-layer disk-read hint. A zero expert_count is
 * a normal fail-soft miss. Nonzero IDs are canonical ascending and can never
 * substitute for the authoritative route mailbox above. */
typedef struct DeltafinProviderTargetSequencePrefetchHintReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint32_t source_layer;
  uint32_t target_layer;
  uint32_t expert_count;
  uint32_t flags;
  uint16_t expert_ids[DELTAFIN_PROVIDER_PILOT_MAX_PREFETCH_V1];
  uint64_t reserved[4];
} DeltafinProviderTargetSequencePrefetchHintReportV1;

/*
 * One synchronous expert tile. IDs are unique canonical ascending IDs; bytes
 * contain exactly one explicit expert span per ID in that same order. Unused
 * expert_ids slots are zero. With flags=0 the provider retains neither bytes,
 * Metal wrappers, nor metal_shader_path. The explicit retain-wrapper flag uses
 * the arena/flush lifetime contract above and still never retains the path.
 */
typedef struct DeltafinProviderTargetSequenceFinishExpertsRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t row_count;
  uint32_t expert_backend;
  uint32_t cpu_threads;
  uint32_t expert_count;
  uint32_t flags;
  uint32_t expert_layout;
  uint16_t expert_ids[DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1];
  const uint8_t* expert_major_bytes;
  uint64_t expert_major_length;
  const char* metal_shader_path;
  uint64_t metal_shader_path_length;
  uint64_t expert_span_bytes;
  uint64_t reserved[3];
} DeltafinProviderTargetSequenceFinishExpertsRequestV1;

/*
 * Additive partial-reuse form of the same synchronous expert tile. Each
 * non-null pointer names exactly one complete span for expert_ids[i]. The
 * provider borrows every span only until the call returns and, with flags=0,
 * retains neither pointers nor Metal wrappers. The explicit retain-wrapper
 * flag uses the arena/flush lifetime contract above. Unused ID slots are zero
 * and unused pointer slots are null. CPU and Metal accept this form; CUDA
 * keeps its contiguous cache-plan miss-slab contract.
 */
typedef struct DeltafinProviderTargetSequenceFinishExpertSpansRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t row_count;
  uint32_t expert_backend;
  uint32_t cpu_threads;
  uint32_t expert_count;
  uint32_t flags;
  uint32_t expert_layout;
  uint16_t expert_ids[DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1];
  const uint8_t*
      expert_span_pointers[DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1];
  const char* metal_shader_path;
  uint64_t metal_shader_path_length;
  uint64_t expert_span_bytes;
  uint64_t reserved[4];
} DeltafinProviderTargetSequenceFinishExpertSpansRequestV1;

/*
 * Expert early drain. Same shape as the scattered finish request, but carries
 * only the experts whose bytes have already landed, so their independent
 * matmuls can start while the rest of the layer is still being read.
 *
 * Advisory in both directions. A refusal is reported in the report's
 * `staged` flag rather than as an error, and anything the provider does accept
 * is still recomputed by the finish call if the staged state does not line up
 * exactly. The finish call remains the sole authority for the route, the fp32
 * weights, and the single reduction over every edge in the router's order --
 * disk arrival order never reaches an accumulation.
 *
 * Single-row tiles with the Metal expert backend only; every other shape is a
 * refusal, never an error.
 */
typedef struct DeltafinProviderTargetSequenceStageExpertSpansRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t row_count;
  uint32_t expert_backend;
  uint32_t cpu_threads;
  uint32_t expert_count;
  uint32_t flags;
  uint32_t expert_layout;
  uint16_t expert_ids[DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1];
  const uint8_t*
      expert_span_pointers[DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1];
  const char* metal_shader_path;
  uint64_t metal_shader_path_length;
  uint64_t expert_span_bytes;
  uint64_t reserved[4];
} DeltafinProviderTargetSequenceStageExpertSpansRequestV1;

typedef struct DeltafinProviderTargetSequenceStageExpertSpansReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t expert_count;
  /* 1 when the provider started work for these experts, 0 when it declined. */
  uint32_t staged;
  uint64_t reserved[2];
} DeltafinProviderTargetSequenceStageExpertSpansReportV1;

/*
 * Additive full-tile union form. V1 remains the hot ABI for unions of at most
 * 64 experts. V2 is used only for a 65..256-expert exact route union and keeps
 * all variable-size storage caller-owned, so the provider allocates no
 * structural 256-expert slab. expert_ids_length and
 * expert_span_pointer_count are element counts, not byte lengths.
 *
 * Exactly one borrowed storage form must be supplied:
 *   - contiguous: expert_major_bytes/length, or
 *   - scattered: expert_span_pointers/count.
 *
 * Every pointer is borrowed synchronously and, with flags=0, retained only
 * until the call returns. The explicit retain-wrapper flag uses the
 * arena/flush lifetime contract above. CPU and Metal accept this ABI; CUDA
 * continues to use its bounded V1 residency-plan contract.
 */
typedef struct DeltafinProviderTargetSequenceFinishExpertsRequestV2 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t row_count;
  uint32_t expert_backend;
  uint32_t cpu_threads;
  uint32_t expert_count;
  uint32_t flags;
  uint32_t expert_layout;
  const uint16_t* expert_ids;
  uint64_t expert_ids_length;
  const uint8_t* expert_major_bytes;
  uint64_t expert_major_length;
  const uint8_t* const* expert_span_pointers;
  uint64_t expert_span_pointer_count;
  const char* metal_shader_path;
  uint64_t metal_shader_path_length;
  uint64_t expert_span_bytes;
  uint64_t reserved[8];
} DeltafinProviderTargetSequenceFinishExpertsRequestV2;

typedef struct DeltafinProviderTargetSequenceFinishExpertsReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t row_count;
  uint32_t next_expert_row;
  uint32_t state;
  uint32_t flags;
  uint64_t reserved[2];
} DeltafinProviderTargetSequenceFinishExpertsReportV1;

/*
 * Resolve the effective backend before expert I/O. CUDA plans pin every cache
 * hit and return only exact misses; CPU/Metal plans return the full canonical
 * set. The plan is session-owned and must be finished or released once.
 */
typedef struct DeltafinProviderTargetSequencePlanExpertsRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t row_count;
  uint32_t expert_backend;
  uint32_t cpu_threads;
  uint32_t expert_count;
  uint32_t flags;
  uint16_t expert_ids[DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1];
  const char* metal_shader_path;
  uint64_t metal_shader_path_length;
  uint64_t reserved[4];
} DeltafinProviderTargetSequencePlanExpertsRequestV1;

typedef struct DeltafinProviderTargetSequencePlanExpertsReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderMoePlanHandleV1 plan;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t row_count;
  uint32_t effective_backend;
  uint32_t missing_count;
  uint32_t cache_capacity_experts;
  uint32_t residency_enabled;
  uint32_t flags;
  uint16_t missing_experts[DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1];
  uint64_t reserved[3];
} DeltafinProviderTargetSequencePlanExpertsReportV1;

typedef struct DeltafinProviderTargetSequenceFinishPlannedExpertsRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  DeltafinProviderMoePlanHandleV1 plan;
  uint64_t spine_generation;
  uint32_t layer_index;
  uint32_t first_row;
  uint32_t row_count;
  uint32_t missing_count;
  uint32_t flags;
  uint16_t missing_experts[DELTAFIN_PROVIDER_TARGET_SEQUENCE_MAX_EXPERTS_V1];
  const uint8_t* expert_major_bytes;
  uint64_t expert_major_length;
  uint64_t reserved[4];
} DeltafinProviderTargetSequenceFinishPlannedExpertsRequestV1;

typedef struct DeltafinProviderTargetSequenceTailReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint32_t token_count;
  uint32_t state;
  uint32_t tail_rows;
  uint32_t tail_provider_dispatches;
  uint32_t token_ids[DELTAFIN_PROVIDER_ROUTE_MAX_POSITIONS_V1];
  uint64_t reserved[4];
} DeltafinProviderTargetSequenceTailReportV1;

typedef struct DeltafinProviderTargetSequenceCommitRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint32_t positions;
  uint32_t flags;
  uint64_t reserved[4];
} DeltafinProviderTargetSequenceCommitRequestV1;

typedef struct DeltafinProviderTargetSequenceCommitReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t committed_positions;
  uint64_t session_committed_positions;
  uint32_t state;
  uint32_t flags;
  uint64_t reserved[3];
} DeltafinProviderTargetSequenceCommitReportV1;

/* Complete committed target-cache boundary, never an in-flight sequence. */
typedef struct DeltafinProviderTargetStateReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t committed_positions;
  uint64_t cache_generation;
  DeltafinProviderTargetStateBranchHandleV1 active_branch;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t reserved[2];
} DeltafinProviderTargetStateReportV1;

/*
 * Begin one exclusive copy-on-write child of the exact published boundary.
 * Every following target sequence addresses the child until publish/discard.
 */
typedef struct DeltafinProviderTargetStateBranchRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderSessionHandleV1 session;
  uint64_t expected_committed_positions;
  uint64_t expected_cache_generation;
  uint64_t reserved[4];
} DeltafinProviderTargetStateBranchRequestV1;

typedef struct DeltafinProviderTargetSequenceStatsReportV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeltafinProviderTargetSequenceHandleV1 sequence;
  uint64_t positions;
  uint64_t streamed_layer_passes;
  uint64_t attention_rows;
  uint64_t expert_row_requests;
  uint64_t expert_rows_completed;
  uint64_t expert_tiles_completed;
  uint64_t tail_rows;
  uint64_t tail_provider_dispatches;
  uint64_t maximum_live_streamed_layers;
  uint64_t maximum_experts_per_request;
  uint64_t maximum_positions_per_expert_tile;
  uint64_t staged_kda_storage_bytes;
  uint64_t verify_snapshot_bytes;
  uint64_t projected_mla_storage_bytes;
  uint64_t additional_mla_storage_bytes;
  uint32_t mode;
  uint32_t state;
  uint64_t reserved[2];
} DeltafinProviderTargetSequenceStatsReportV1;

uint32_t deltafin_provider_abi_version(void);

int32_t deltafin_provider_inventory_v1(
    DeltafinProviderInventoryV1* inventory,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_canary_v1(
    const DeltafinProviderCanaryRequestV1* request,
    DeltafinProviderCanaryReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_session_create_v1(
    const DeltafinProviderSessionRequestV1* request,
    DeltafinProviderSessionReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_memory_snapshot_v1(
    const DeltafinProviderMemoryRequestV1* request,
    DeltafinProviderMemoryReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_pilot_enable_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetPilotEnableReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_session_destroy_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_cuda_cache_configure_v1(
    const DeltafinProviderCudaCacheConfigureRequestV1* request,
    DeltafinProviderCudaCacheConfigureReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_metal_expert_layouts_v1(
    const DeltafinProviderMetalExpertLayoutsRequestV1* request,
    DeltafinProviderMetalExpertLayoutsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_metal_expert_cache_flush_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_metal_expert_cache_stats_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderMetalExpertCacheStatsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_tensor_upload_f32_v1(
    const DeltafinProviderTensorUploadF32V1* request,
    DeltafinProviderTensorReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_tensor_upload_bf16_v1(
    const DeltafinProviderTensorUploadBf16V1* request,
    DeltafinProviderTensorReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_tensor_read_f32_v1(
    const DeltafinProviderTensorReadF32V1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_tensor_release_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_cache_create_f32_v1(
    const DeltafinProviderCacheCreateF32V1* request,
    DeltafinProviderCacheReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_cache_read_f32_v1(
    const DeltafinProviderCacheReadF32V1* request,
    DeltafinProviderCacheReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_cache_release_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_prepare_layer_v1(
    const DeltafinProviderPrepareLayerRequestV1* request,
    DeltafinProviderRouteMailboxV1* mailbox,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_finish_layer_v1(
    const DeltafinProviderFinishLayerRequestV1* request,
    DeltafinProviderFinishLayerReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_ticket_release_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_bind_spine_layer_v1(
    const DeltafinProviderBindSpineLayerRequestV1* request,
    DeltafinProviderBindSpineLayerReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_bind_spine_layer_v2(
    const DeltafinProviderBindSpineLayerRequestV2* request,
    DeltafinProviderBindSpineLayerReportV2* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_spine_source_use_seal_v2(
    const DeltafinProviderSpineSourceUseRequestV2* request,
    DeltafinProviderSpineSourceUseReportV2* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_spine_source_use_try_reclaim_v2(
    const DeltafinProviderSpineSourceUseRequestV2* request,
    DeltafinProviderSpineSourceUseReportV2* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_spine_source_use_abort_v2(
    const DeltafinProviderSpineSourceUseRequestV2* request,
    DeltafinProviderSpineSourceUseReportV2* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_spine_tensor_read_f32_v1(
    const DeltafinProviderSpineTensorReadF32V1* request,
    DeltafinProviderSpineTensorReadReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_kda_cache_create_v1(
    const DeltafinProviderKdaCacheCreateV1* request,
    DeltafinProviderKdaCacheReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_kda_cache_release_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_kda_decode_v1(
    const DeltafinProviderKdaDecodeRequestV1* request,
    DeltafinProviderKdaDecodeReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_kda_commit_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderKdaCommitReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_kda_ticket_release_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_mla_cache_create_v1(
    const DeltafinProviderMlaCacheCreateV1* request,
    DeltafinProviderMlaCacheReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_mla_cache_release_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_mla_decode_v1(
    const DeltafinProviderMlaDecodeRequestV1* request,
    DeltafinProviderMlaDecodeReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_mla_commit_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderMlaCommitReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_mla_ticket_release_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_bind_target_globals_v1(
    const DeltafinProviderBindTargetGlobalsRequestV1* request,
    DeltafinProviderBindTargetGlobalsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_begin_v1(
    const DeltafinProviderTargetBeginRequestV1* request,
    DeltafinProviderTargetBeginReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_begin_bf16_v1(
    const DeltafinProviderTargetBeginBf16RequestV1* request,
    DeltafinProviderTargetBeginReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_prepare_v1(
    const DeltafinProviderTargetPrepareRequestV1* request,
    DeltafinProviderTargetPrepareReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_finish_experts_v1(
    const DeltafinProviderTargetFinishExpertsRequestV1* request,
    DeltafinProviderTargetFinishExpertsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_finish_greedy_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetGreedyReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_cancel_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

/*
 * Drop only target attention/cache state while retaining immutable globals,
 * resident spine layers, the transient layer slot, and provider qualification.
 * The request resource must be zero and no target transaction may be live.
 */
int32_t deltafin_provider_target_state_reset_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_state_inspect_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetStateReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_state_branch_begin_v1(
    const DeltafinProviderTargetStateBranchRequestV1* request,
    DeltafinProviderTargetStateReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_state_branch_publish_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetStateReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_state_branch_discard_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetStateReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_begin_bf16_v1(
    const DeltafinProviderTargetSequenceBeginBf16RequestV1* request,
    DeltafinProviderTargetSequenceBeginReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_prepare_v1(
    const DeltafinProviderTargetSequencePrepareRequestV1* request,
    DeltafinProviderTargetSequencePrepareReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_take_prefetch_hint_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetSequencePrefetchHintReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_finish_experts_v1(
    const DeltafinProviderTargetSequenceFinishExpertsRequestV1* request,
    DeltafinProviderTargetSequenceFinishExpertsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_finish_expert_spans_v1(
    const DeltafinProviderTargetSequenceFinishExpertSpansRequestV1* request,
    DeltafinProviderTargetSequenceFinishExpertsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_stage_expert_spans_v1(
    const DeltafinProviderTargetSequenceStageExpertSpansRequestV1* request,
    DeltafinProviderTargetSequenceStageExpertSpansReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_finish_experts_v2(
    const DeltafinProviderTargetSequenceFinishExpertsRequestV2* request,
    DeltafinProviderTargetSequenceFinishExpertsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_plan_experts_v1(
    const DeltafinProviderTargetSequencePlanExpertsRequestV1* request,
    DeltafinProviderTargetSequencePlanExpertsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_finish_planned_experts_v1(
    const DeltafinProviderTargetSequenceFinishPlannedExpertsRequestV1* request,
    DeltafinProviderTargetSequenceFinishExpertsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_moe_plan_release_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_finish_tail_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetSequenceTailReportV1* report,
    char* error,
    size_t error_capacity);

/*
 * Export the rows a completed sequence captured for its proposal model:
 * BF16 [positions, 5*7168] for DSpark or [positions, 3*7168] for EAGLE-3,
 * per the begin flag. The report's columns say which.
 */
int32_t deltafin_provider_target_sequence_dspark_rows_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTensorReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_commit_v1(
    const DeltafinProviderTargetSequenceCommitRequestV1* request,
    DeltafinProviderTargetSequenceCommitReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_stats_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderTargetSequenceStatsReportV1* report,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_target_sequence_cancel_v1(
    const DeltafinProviderResourceRequestV1* request,
    char* error,
    size_t error_capacity);

int32_t deltafin_provider_dspark_create_v1(
    const DeltafinProviderDSparkCreateV1* request,
    DeltafinProviderDSparkReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_dspark_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    size_t error_capacity);
int32_t deltafin_provider_dspark_append_target_v1(
    const DeltafinProviderDSparkAppendV1* request,
    DeltafinProviderDSparkReportV1* report, char* error,
    size_t error_capacity);

int32_t deltafin_provider_dspark_append_target_tensor_v1(
    const DeltafinProviderDSparkAppendTensorV1* request,
    DeltafinProviderDSparkReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_dspark_snapshot_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderDSparkSnapshotReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_dspark_restore_v1(
    const DeltafinProviderDSparkRestoreV1* request,
    DeltafinProviderDSparkReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_dspark_snapshot_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    size_t error_capacity);
int32_t deltafin_provider_dspark_propose_v1(
    const DeltafinProviderDSparkProposeV1* request,
    DeltafinProviderDSparkProposalReportV1* report, char* error,
    size_t error_capacity);

int32_t deltafin_provider_eagle3_create_v1(
    const DeltafinProviderEagle3CreateV1* request,
    DeltafinProviderEagle3ReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_eagle3_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    size_t error_capacity);
int32_t deltafin_provider_eagle3_advance_v1(
    const DeltafinProviderEagle3AdvanceV1* request,
    DeltafinProviderEagle3ReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_eagle3_step_v1(
    const DeltafinProviderEagle3StepV1* request,
    DeltafinProviderEagle3StepReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_eagle3_snapshot_v1(
    const DeltafinProviderResourceRequestV1* request,
    DeltafinProviderEagle3SnapshotReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_eagle3_restore_v1(
    const DeltafinProviderEagle3RestoreV1* request,
    DeltafinProviderEagle3ReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_eagle3_snapshot_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    size_t error_capacity);

int32_t deltafin_provider_qwen_create_v1(
    const DeltafinProviderQwenCreateV1* request,
    DeltafinProviderQwenReportV1* report, char* error,
    size_t error_capacity);
int32_t deltafin_provider_qwen_destroy_v1(
    const DeltafinProviderResourceRequestV1* request, char* error,
    size_t error_capacity);
int32_t deltafin_provider_qwen_generate_v1(
    const DeltafinProviderQwenGenerateV1* request,
    DeltafinProviderQwenGenerationReportV1* report, char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
