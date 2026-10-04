#ifndef DELTAFIN_PROVIDER_MOE_H
#define DELTAFIN_PROVIDER_MOE_H

#include "provider_bf16_cpu.h"

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace deltafin::provider_internal {

class CudaMoeExpertCache;

constexpr std::size_t kMoeRouteTopK = 16;
constexpr std::size_t kMoePositionTileMaxRows = 16;
constexpr std::size_t kMoePositionTileMaxExperts =
    kMoeRouteTopK * kMoePositionTileMaxRows;

/*
 * Geometry is explicit so every byte offset can be checked before a kernel
 * sees a pointer. Production builds accept only k3_moe_geometry(). A tiny
 * geometry is enabled solely in the standalone native parity harness.
 */
struct MoeGeometry {
  std::uint32_t hidden = 0;
  std::uint32_t routed_hidden = 0;
  std::uint32_t intermediate = 0;
  std::uint32_t experts = 0;
  std::uint32_t shared_intermediate = 0;

  [[nodiscard]] std::uint64_t expert_span_bytes() const;
};

[[nodiscard]] constexpr MoeGeometry k3_moe_geometry() {
  return MoeGeometry{7168, 3584, 3072, 896, 6144};
}

/*
 * Resident linear carrier. Historical q8 fields remain ABI-internal names;
 * explicit quantized binding supplies q/scales, synthetic tests may supply a
 * dense fallback, and canonical checkpoints use exact original BF16. Exactly
 * one representation is selected for a production layer.
 */
struct MoeRowInt8Matrix {
  at::Tensor quantized;
  at::Tensor row_scales;

  /*
   * Optional provider-owned fp32 materialization. It is used only when the
   * selected provider has not passed the private packed-int8 capability gate,
   * or the authoritative fp32 expansion of an original BF16 source matrix.
   */
  at::Tensor dense_f32;

  /* Exact original-BF16 storage, provider-owned or V2-borrowed on CPU. */
  OriginalBf16Matrix original_bf16;
};

/* Exactly the eight resident-spine slots needed by a K3 MoE block. */
struct MoeSpineT1 {
  std::uint32_t layer_index = 0;
  std::uint64_t generation = 0;
  MoeGeometry geometry = {};
  bool packed_int8_qualified = false;
  MoeRowInt8Matrix router;
  at::Tensor router_correction_bias;
  MoeRowInt8Matrix routed_down;
  at::Tensor routed_norm;
  MoeRowInt8Matrix routed_up;
  MoeRowInt8Matrix shared_gate;
  MoeRowInt8Matrix shared_up;
  MoeRowInt8Matrix shared_down;
  /* Optional zero-copy super-view over adjacent shared_gate/shared_up rows. */
  MoeRowInt8Matrix shared_gate_up;
  bool shared_gate_up_qualified = false;
  bool shared_gate_up_enabled = false;
};

/* Ordered exactly as produced by torch.topk(sorted=False). */
struct MoeRouteT1 {
  std::array<std::uint16_t, kMoeRouteTopK> expert_ids = {};
  std::array<std::uint32_t, kMoeRouteTopK> weight_bits = {};
};

struct PreparedMoeT1 {
  std::uint32_t layer_index = 0;
  std::uint64_t spine_generation = 0;
  MoeGeometry geometry = {};
  at::Tensor identity;
  at::Tensor routed_input;
  /*
   * Host-backed Metal experts stage the routed vector before execution.
   * shared_output is reserved for an explicitly qualified experimental
   * overlap path; established execution leaves it undefined and evaluates the
   * shared branch after routed norm/up in complete_moe_*.
   */
  at::Tensor routed_input_cpu;
  at::Tensor shared_output;
  MoeRouteT1 route;
};

/*
 * Multi-position preparation retains the exact per-row carrier consumed by
 * the established expert backends, but launches the independent resident
 * projections once for the whole bounded position set.  The counters describe
 * provider/host boundaries actually crossed by this preparation and let the
 * sequence harness prove that they scale with position batches, not rows.
 */
struct PreparedMoePositionsT1 {
  std::vector<PreparedMoeT1> rows;
  /* Whole [T,routed_hidden] device carrier retained for a target-sequence
   * layer. Individual PreparedMoeT1 rows are views into this storage. */
  at::Tensor routed_inputs;
  std::uint64_t router_dispatches = 0;
  std::uint64_t routed_down_dispatches = 0;
  std::uint64_t shared_dispatches = 0;
  std::uint64_t route_materializations = 0;
  std::uint64_t route_host_transfers = 0;
  std::uint64_t routed_input_host_transfers = 0;
};

enum class MoeExpertLayout : std::uint32_t {
  RawV1 = 1,
  Scale4V2 = 2,
};

/*
 * Rust's authenticated ExpertBatchPlan supplies ascending IDs plus one
 * adjacent, explicitly identified storage span per ID. The view is borrowed
 * only for the synchronous expert call; no pointer is retained by this module.
 */
struct CanonicalExpertBatchT1 {
  std::span<const std::uint16_t> expert_ids;
  std::span<const std::uint8_t> expert_major_bytes;
  MoeExpertLayout layout;
  std::uint64_t expert_span_bytes;
};

/*
 * One bounded set of unique, ascending expert IDs shared by 1..16 canonical
 * position rows.  Each row still carries its own ordered route and fp32
 * weights in PreparedMoeT1.  The provider maps repeated position-major edges
 * back into this unique storage, so the reader never has to materialize an
 * unbounded whole-chunk expert union.
 */
struct CanonicalExpertPositionTileT1 {
  std::span<const std::uint16_t> expert_ids;
  std::span<const std::uint8_t> expert_major_bytes;
  MoeExpertLayout layout;
  std::uint64_t expert_span_bytes;
  /*
   * Optional scattered storage for partial speculative-I/O reuse. Exactly
   * one of `expert_major_bytes` and `expert_span_pointers` may be non-empty.
   * Pointer slot i names one complete authenticated span for expert_ids[i]
   * and is borrowed only through the synchronous provider call. CUDA keeps
   * its existing contiguous residency-plan contract and rejects this form.
   */
  std::span<const std::uint8_t* const> expert_span_pointers = {};
};

struct MetalExpertLayoutCapabilities {
  std::uint32_t descriptor_abi = 0;
  std::uint64_t layout_capabilities = 0;
  std::uint64_t raw_span_bytes = 0;
  std::uint64_t scale4_span_bytes = 0;
};

struct MetalExpertCacheStats {
  std::uint64_t calls = 0;
  std::uint64_t zero_copy_wraps = 0;
  std::uint64_t copies = 0;
  std::uint64_t cache_entries = 0;
  std::uint64_t bindless = 0;
};

enum class MoeExpertBackend : std::uint32_t {
  Auto = 0,
  CpuMxfp4 = 1,
  MetalMxfp4 = 2,
  CudaMxfp4 = 3,
};

// Optional focused-test probe. A null pointer is the production default, so
// schedule tracing cannot allocate, synchronize, or alter dispatch topology.
enum class MoeExecutionStage : std::uint8_t {
  Router,
  RoutedDown,
  RouteMaterialization,
  ExpertBytesBorrowed,
  RoutedInputHostMaterialization,
  CpuExpertDispatch,
  MetalExpertRowDispatch,
  MetalExpertPositionDispatch,
  CudaExpertDispatch,
  RoutedUp,
  Shared,
  Merge,
};

struct MoeExecutionTrace {
  std::array<MoeExecutionStage, 256> stages{};
  std::size_t count = 0;

  void record(const MoeExecutionStage stage) {
    if (count >= stages.size()) {
      throw std::logic_error("MoE execution trace overflow");
    }
    stages[count++] = stage;
  }
};

struct MoeRunOptions {
  MoeExpertBackend expert_backend = MoeExpertBackend::Auto;
  std::uint32_t cpu_threads = 1;
  /* Versioned embedded-metallib selector; debug builds may accept a source. */
  std::string metal_shader_path;
  /* Session-owned; null in CPU/Metal builds and all standalone tiny tests. */
  CudaMoeExpertCache* cuda_cache = nullptr;
  /* Nonzero only for one provider-owned pre-I/O CUDA residency snapshot. */
  std::uint64_t cuda_plan = 0;
  /* Only an automatic request may retry a classified context-healthy failure. */
  bool cuda_auto_fallback = false;
  /* Prefer the already-qualified flat position ABI for native T>1 Metal
   * execution. A capability miss steps down to the standard position ABI,
   * then to the established row-serial ABI. T=1 never observes this option. */
  bool metal_position_batch = true;
  /* TargetSequenceTape's whole-layer staging path sets this only after it has
   * materialized every routed input row on CPU. The Metal call then returns
   * its CPU output carrier without a per-tile CPU->MPS crossing; the tape
   * publishes one complete T-wide device transfer before MoE completion. */
  bool metal_retain_position_outputs_cpu = false;
  /* Default false preserves the public synchronous-borrow ABI. True is legal
   * only when the caller has installed flush-before-arena-retirement hooks. */
  bool metal_retain_expert_wrappers = false;
  /* Nonzero consumes a staged single-position Metal layer whose edges were
   * partly computed while their bytes were still arriving. Zero is the
   * established path; a stale token is ignored, never trusted. */
  std::uint64_t metal_staged_token = 0;
  MoeExecutionTrace* execution_trace = nullptr;
};

/*
 * Initialize the same Metal bridge used by expert execution with the exact
 * production shader path, then report only descriptor layouts whose complete
 * symbol and pipeline suite is available. Non-Apple callers fail closed.
 */
[[nodiscard]] MetalExpertLayoutCapabilities qualify_metal_expert_layouts(
    const std::string& metal_shader_path);

/*
 * The Metal bridge's no-copy wrappers alias caller-owned arena pages. Stable
 * arena addresses may retain those wrappers across synchronous refills, but
 * the complete cache must be flushed before an arena allocation is retired.
 * These functions expose only the reviewed bridge's existing cache lifecycle
 * and counters; they do not alter expert arithmetic or dispatch selection.
 */
void flush_metal_expert_cache();
[[nodiscard]] MetalExpertCacheStats metal_expert_cache_stats();

/*
 * The split mirrors the Rust reader boundary: prepare routes and routed input,
 * execute consumes the just-read experts, and complete applies routed norm/up
 * before the shared expert and final add. No Python object or callback
 * participates in any phase.
 */
[[nodiscard]] PreparedMoeT1 prepare_moe_t1(
    const at::Tensor& hidden, const MoeSpineT1& spine,
    MoeExecutionTrace* execution_trace = nullptr);

/*
 * Vectorized resident/router half for 1..64 positions.  The one-position
 * branch delegates directly to prepare_moe_t1 so decode keeps its established
 * operation order and tensor bytes. Wider calls materialize all route rows
 * with one device-to-host boundary. Host-backed expert inputs deliberately
 * remain device-resident until execute, after Rust demand-fetches the bytes.
 */
[[nodiscard]] PreparedMoePositionsT1 prepare_moe_positions_t1(
    const at::Tensor& hidden_rows, const MoeSpineT1& spine,
    MoeExecutionTrace* execution_trace = nullptr);

[[nodiscard]] at::Tensor execute_routed_moe_t1(
    const PreparedMoeT1& prepared, const CanonicalExpertBatchT1& experts,
    const MoeRunOptions& options);

/*
 * Existing C151/N5 position-major backend boundary. Metal prefers the
 * qualified flat GLU -> W2 -> ordered-reduction ABI, steps down to the
 * standard one-command-buffer position ABI on a -6 capability miss, then to
 * canonical row-serial execution when neither position ABI is available.
 * CPU uses the identical routes and unique expert tile with a canonical row
 * loop. The separately experimental fused-reduction ABI is not selected.
 */
[[nodiscard]] at::Tensor execute_routed_moe_positions_t1(
    std::span<const PreparedMoeT1* const> prepared_rows,
    const CanonicalExpertPositionTileT1& experts,
    const MoeRunOptions& options);

/* Exact backend selection probe used only by TargetSequenceTape to decide
 * whether its T>1 transaction should establish whole-layer host staging. */
[[nodiscard]] bool moe_positions_select_metal(
    const at::Device& device, const MoeRunOptions& options);

/*
 * Expert early drain, single-position Metal only.
 *
 * A decode layer's missing experts are read from storage one file at a time,
 * and the established path leaves every expert's matmul waiting until the last
 * one has landed. These calls let the caller run an expert's independent
 * GLU/W2 work the moment that expert's own bytes are readable, so the GPU is
 * busy during the remaining reads.
 *
 * What this deliberately does not touch: the weighted combination of each
 * expert's contribution into the layer's routed output still happens exactly
 * once, inside the ordinary finish call, over every edge in the router's own
 * order, from the per-edge results staging produced. Disk arrival order
 * reaches scheduling only -- never a value, never a summation order -- so the
 * same prompt produces the same bytes regardless of I/O timing.
 *
 * `begin` returns 0 when staging is unavailable for any reason (non-Apple
 * build, unqualified bridge, wrong backend, multi-position tile); the caller
 * then simply runs the established path. `stage` returns false when the batch
 * could not be staged, which is likewise not an error. A begun token must
 * reach exactly one of `execute_routed_moe_positions_t1` (via
 * MoeRunOptions::metal_staged_token) or `abandon_staged_routed_moe_t1`.
 */
[[nodiscard]] std::uint64_t begin_staged_routed_moe_t1(
    const PreparedMoeT1& prepared, const at::Tensor& routed_input_cpu,
    const MoeRunOptions& options);

/* `arrived_expert_ids` and `arrived_span_pointers` describe only the experts
 * that just became readable: unique ascending IDs, one complete authenticated
 * span each, borrowed until the layer's finish call returns. */
[[nodiscard]] bool stage_routed_moe_edges_t1(
    std::uint64_t token, const PreparedMoeT1& prepared,
    std::span<const std::uint16_t> arrived_expert_ids,
    std::span<const std::uint8_t* const> arrived_span_pointers,
    MoeExpertLayout layout, std::uint64_t expert_span_bytes,
    const MoeRunOptions& options);

void abandon_staged_routed_moe_t1(std::uint64_t token);

struct StagedMoeStats {
  std::uint64_t begins = 0;
  std::uint64_t edges = 0;
  std::uint64_t reused_finishes = 0;
  std::uint64_t recomputed_finishes = 0;
};

[[nodiscard]] StagedMoeStats staged_moe_stats();

/* Bind-time-only, allocation-free qualification of an adjacent gate/up slab. */
[[nodiscard]] bool qualify_moe_shared_gate_up(MoeSpineT1& spine);

[[nodiscard]] at::Tensor complete_moe_t1(
    const PreparedMoeT1& prepared, const at::Tensor& routed_output,
    const MoeSpineT1& spine, MoeExecutionTrace* execution_trace = nullptr);

/* One T-wide routed RMS/up projection followed by the live-order T-wide
 * shared branch and merge. Expert execution may arrive in <=16-row I/O tiles;
 * completion accepts the full prepared 1..64-position sequence. */
[[nodiscard]] at::Tensor complete_moe_positions_t1(
    std::span<const PreparedMoeT1* const> prepared_rows,
    const at::Tensor& routed_outputs, const MoeSpineT1& spine,
    MoeExecutionTrace* execution_trace = nullptr);

[[nodiscard]] at::Tensor run_moe_t1(
    const at::Tensor& hidden, const MoeSpineT1& spine,
    const CanonicalExpertBatchT1& experts, const MoeRunOptions& options);

}  // namespace deltafin::provider_internal

#endif
