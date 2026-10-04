#ifndef K3_METAL_MOE_ABI_H
#define K3_METAL_MOE_ABI_H

#include <stddef.h>
#include <stdint.h>

/*
 * Versioned, additive expert-storage ABI shared by Deltafin's native provider
 * and the established Objective-C++ Metal bridge. Legacy raw-pointer entry
 * points remain unchanged. Compact storage is accepted only through these
 * descriptors, so no caller can accidentally reinterpret scale4 bytes as the
 * larger raw-v1 layout.
 */
enum {
  K3_LAYOUT_RAW_V1 = 1,
  K3_LAYOUT_SCALE4_V2 = 2,
  K3_DESCRIPTOR_ABI_V1 = 1,
};

#define K3_RAW_V1_EXPERT_SPAN UINT64_C(17547264)
#define K3_SCALE4_V2_EXPERT_SPAN UINT64_C(17039360)
#define K3_CAP_RAW_V1 (UINT64_C(1) << K3_LAYOUT_RAW_V1)
#define K3_CAP_SCALE4_V2 (UINT64_C(1) << K3_LAYOUT_SCALE4_V2)

/*
 * Source selector ABI, not a filesystem path. Production callers pass this
 * value to k3_metal_init() so the bridge loads the precompiled metallib
 * embedded into the native binary. Debug builds may admit another non-empty
 * string as an explicit development source path; NULL and the empty string
 * also select this embedded version.
 */
#define K3_METAL_MOE_EMBEDDED_SOURCE_V1 \
  "deltafin:embedded-metal-moe-mxfp4:v1"

typedef struct K3MetalExpertDescriptorV1 {
  uint32_t abi_version;
  uint32_t struct_bytes;
  uint32_t layout_id;
  uint32_t reserved;
  uint64_t blob_bytes;
  const uint8_t* blob;
} K3MetalExpertDescriptorV1;

#ifdef __cplusplus
static_assert(sizeof(K3MetalExpertDescriptorV1) == 32,
              "Metal expert descriptor ABI drift");
static_assert(offsetof(K3MetalExpertDescriptorV1, blob_bytes) == 16,
              "Metal expert descriptor byte-count offset drift");
static_assert(offsetof(K3MetalExpertDescriptorV1, blob) == 24,
              "Metal expert descriptor pointer offset drift");
extern "C" {
#endif

uint32_t k3_metal_descriptor_abi_version(void);
uint64_t k3_metal_layout_capabilities(void);

int k3_metal_moe_layer_desc_v1(
    const K3MetalExpertDescriptorV1* experts, int expert_count,
    const float* weights, const float* input, float* output);

int k3_metal_moe_positions_desc_v1(
    const K3MetalExpertDescriptorV1* experts, int edge_count,
    const int* position_offsets, int position_count, const float* weights,
    const float* input, float* output);

/*
 * Staged single-position execution ("expert early drain").
 *
 * The ordinary layer call needs every expert's bytes before it can encode
 * anything. These entry points let a caller whose experts arrive from storage
 * one at a time run each expert's independent GLU/W2 work the moment that
 * expert's own bytes have landed, while the rest of the layer is still being
 * read.
 *
 * Exactness contract: staging changes command-buffer packaging only. A staged
 * edge runs the same per-expert kernels, over the same bytes, into the same
 * per-edge output slot as the unstaged path, and the weighted reduction still
 * runs exactly once, inside the finish call, over every edge in the caller's
 * canonical route order. Arrival order can never reach the fp32 accumulation.
 *
 *   k3_metal_moe_stage_begin_v1(expert_count, input)
 *       Latch one layer's edge count and activation row. Returns a nonzero
 *       staging token, or 0 when staging is unavailable (the caller then just
 *       uses the ordinary layer call). Any other bridge entry point
 *       invalidates an outstanding token rather than corrupting it.
 *
 *   k3_metal_moe_stage_edges_v1(token, experts, edges, count)
 *       Compute GLU+W2 for `count` edges whose bytes are now readable.
 *       experts[i] describes the blob for edge edges[i]. Commits without
 *       waiting: the GPU works while the caller returns to its reads. A
 *       nonzero result means nothing was staged for this call.
 *
 *   k3_metal_moe_stage_finish_v1(token, experts, expert_count, weights,
 *                                input, output)
 *       Complete the layer. Computes whatever was never staged, then reduces.
 *       `experts` covers every edge in route order, exactly as the ordinary
 *       call. A stale, mismatched, or failed staging state is not an error:
 *       the call silently recomputes every edge, so its output never depends
 *       on whether staging happened.
 *
 *   k3_metal_moe_stage_abandon_v1(token)
 *       Drop an unfinished staging token. Harmless for an already-consumed or
 *       already-superseded token.
 */
uint64_t k3_metal_moe_stage_begin_v1(int expert_count, const float* input);

int k3_metal_moe_stage_edges_v1(uint64_t token,
                                const K3MetalExpertDescriptorV1* experts,
                                const int* edges, int count);

int k3_metal_moe_stage_finish_v1(uint64_t token,
                                 const K3MetalExpertDescriptorV1* experts,
                                 int expert_count, const float* weights,
                                 const float* input, float* output);

void k3_metal_moe_stage_abandon_v1(uint64_t token);

/* [begins, edges staged, finishes that reused staging, finishes that fell
 * back to full recomputation]. Diagnostic only. */
void k3_metal_moe_stage_stats_v1(long long* four);

#ifdef __cplusplus
}
#endif

#endif
