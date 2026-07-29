// CUDA MoE kernels for Deltafin — MXFP4 dequant + GEMV, fused FFN,
// grouped dispatch, int8 spine dequant, and multi-position batching.
//
// Compiled:  nvcc -O3 -shared -gencode arch=compute_75,code=sm_75 ... -o libcudamoe.so
//
// Physical layout of one on-disk expert blob (17,547,264 bytes):
//   [w1_packed: 5,505,024] [w1_scales: 344,064]
//   [w2_packed: 5,505,024] [w2_scales: 344,064]
//   [w3_packed: 5,505,024] [w3_scales: 344,064]
//
// MXFP4: e2m1 nibble values are {0,0.5,1,1.5,2,3,4,6,±} x 8 magnitudes.
//        e8m0 scale: as_type<float>((uint32_t)byte << 23)  =  2^(byte-127)
//        Packing: uint8_t[cols/2], low nibble = element[2i], high = element[2i+1]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ───────────────────────── shared constants ────────────────────────────────

// E2M1 lookup table: nibble -> fp32 value
__constant__ float c_e2m1_tab[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// ──────────────────────── helper: e8m0 → fp32 ──────────────────────────────

__device__ __forceinline__ float e8m0_to_f32(uint8_t s) {
    uint32_t bits = (uint32_t)s << 23;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

// ───────────────── phase 1: per-expert MXFP4 dequant + GEMV ─────────────────
//
// One thread block handles one output row.  256 threads each process
// ceil(cols/256) elements, then reduce via shared memory.

__global__ void mxfp4_gemv_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int groups = cols / 32;
    float sum = 0.0f;

    // Each thread handles a strided range of elements along the row
    for (int i = tid; i < cols; i += nthreads) {
        int g = i / 32;
        int byte_idx = i / 2;
        uint8_t byte_val = packed[(size_t)row * (cols / 2) + byte_idx];
        int nibble = (i & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
        float val = c_e2m1_tab[nibble];
        float scale = e8m0_to_f32(scales[(size_t)row * groups + g]);
        sum += val * scale * x[i];
    }

    // Tree reduction in shared memory
    __shared__ float s_partial[256];
    s_partial[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_partial[tid] += s_partial[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = s_partial[0];
}

// ──────────── phase 3: warp-per-row grouped MoE dispatch ───────────────────
//
// Grid:  (I/8, E) for hidden, (H/8, E) for output — ~13K blocks total.
// Block: (32, 8) = 256 threads — 8 warps, each handling 1 row.
// Each warp uses __shfl_xor for dot-product reduction (no shared memory).
// Hidden state in global memory for cross-phase communication.
//
// This gives good occupancy (~93 blocks/SM) and efficient warp-local
// reductions instead of expensive __syncthreads tree reductions.

struct ExpertDesc {
    const uint8_t* w1_packed;
    const uint8_t* w1_scales;
    const uint8_t* w3_packed;
    const uint8_t* w3_scales;
    const uint8_t* w2_packed;
    const uint8_t* w2_scales;
    float routing_weight;
};

// ── Kernel 1: compute hidden[E][I] = SiTU(w1@x, w3@x) ─────────────────────
// Grid(I/8, E). Block(32, 8). Each warp = 1 row.
__global__ void mxfp4_moe_hidden_kernel(
    float* __restrict__ hidden,
    const float* __restrict__ x,
    const struct ExpertDesc* __restrict__ descs,
    int H, int I)
{
    int row_base = blockIdx.x * 8;
    int e = blockIdx.y;
    int warp_id = threadIdx.y;  // 0..7, which row within block
    int row = row_base + warp_id;
    if (row >= I) return;

    const struct ExpertDesc d = descs[e];
    int lane = threadIdx.x;    // 0..31 within warp

    float gate = 0.0f, up = 0.0f;
    for (int i = lane; i < H; i += 32) {
        int g = i / 32;
        int bi = i / 2;
        uint8_t b1 = __ldg(&d.w1_packed[(size_t)row * (H / 2) + bi]);
        int n1 = (i & 1) ? (b1 >> 4) : (b1 & 0x0F);
        float s1 = e8m0_to_f32(__ldg(&d.w1_scales[(size_t)row * (H / 32) + g]));
        gate += c_e2m1_tab[n1] * s1 * x[i];
        uint8_t b3 = __ldg(&d.w3_packed[(size_t)row * (H / 2) + bi]);
        int n3 = (i & 1) ? (b3 >> 4) : (b3 & 0x0F);
        float s3 = e8m0_to_f32(__ldg(&d.w3_scales[(size_t)row * (H / 32) + g]));
        up += c_e2m1_tab[n3] * s3 * x[i];
    }

    // Warp-level shuffle reduction
    for (int s = 16; s > 0; s >>= 1) {
        gate += __shfl_xor_sync(0xFFFFFFFF, gate, s);
        up   += __shfl_xor_sync(0xFFFFFFFF, up,   s);
    }

    // SiTU (lane 0 writes)
    float sig = 1.0f / (1.0f + expf(-gate));
    float hv = (4.0f * tanhf(gate / 4.0f) * sig) * (25.0f * tanhf(up / 25.0f));
    if (lane == 0)
        hidden[(size_t)e * I + row] = hv;
}

// ── Kernel 2: out[H] += w[e] * w2@h[e] ───────────────────────────────────
// Grid(H/8, E). Block(32, 8). Each warp = 1 row.
__global__ void mxfp4_moe_output_kernel(
    float* __restrict__ out,
    const float* __restrict__ hidden,
    const struct ExpertDesc* __restrict__ descs,
    int H, int I)
{
    int row_base = blockIdx.x * 8;
    int e = blockIdx.y;
    int warp_id = threadIdx.y;
    int row = row_base + warp_id;
    if (row >= H) return;

    const struct ExpertDesc d = descs[e];
    int lane = threadIdx.x;

    float dot = 0.0f;
    for (int i = lane; i < I; i += 32) {
        int g = i / 32;
        int bi = i / 2;
        uint8_t b = __ldg(&d.w2_packed[(size_t)row * (I / 2) + bi]);
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        float sc = e8m0_to_f32(__ldg(&d.w2_scales[(size_t)row * (I / 32) + g]));
        dot += c_e2m1_tab[n] * sc * hidden[(size_t)e * I + i];
    }

    for (int s = 16; s > 0; s >>= 1)
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, s);

    if (lane == 0) atomicAdd(&out[row], dot * d.routing_weight);
}


// ──────────── phase 4: int8 spine dequant kernel ───────────────────────────
//
// Fused dequant: out[i] = float(q[i]) * float(sc[i / cols])
// Bit-identical to (q.to(fp32) * sc.to(fp32)).

__global__ void int8_deq_kernel(
    float* __restrict__ out,
    const int8_t* __restrict__ q,
    const half* __restrict__ sc,
    int rows, int cols)
{
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;

    float scale = __half2float(sc[r]);
    out[(size_t)r * cols + c] = (float)q[(size_t)r * cols + c] * scale;
}

// ──────────── phase 5: multi-position warp-per-row MoE ─────────────────────
//
// Grid: (N, I/8, max_experts) for hidden, (N, H/8, max_experts) for output.
// Same warp-per-row, shuffle-reduction approach as phase 3.

struct PosExpertDesc {
    const uint8_t* w1_packed;
    const uint8_t* w1_scales;
    const uint8_t* w3_packed;
    const uint8_t* w3_scales;
    const uint8_t* w2_packed;
    const uint8_t* w2_scales;
    float routing_weight;
    int expert_id;
};

// Kernel 1: hidden
__global__ void mxfp4_moe_pos_hidden_kernel(
    float* __restrict__ hidden,
    const float* __restrict__ x,
    const struct PosExpertDesc* __restrict__ descs,
    const int* __restrict__ expert_counts,
    int H, int I, int max_experts)
{
    int pos = blockIdx.z;
    int row_base = blockIdx.x * 8;
    int e = blockIdx.y;
    int warp_id = threadIdx.y;
    int row = row_base + warp_id;
    int ne = expert_counts[pos];
    if (row >= I || e >= ne) return;

    int flat = pos * max_experts + e;
    const struct PosExpertDesc d = descs[flat];
    int lane = threadIdx.x;

    float gate = 0.0f, up = 0.0f;
    for (int i = lane; i < H; i += 32) {
        int g = i / 32;
        int bi = i / 2;
        uint8_t b1 = __ldg(&d.w1_packed[(size_t)row * (H / 2) + bi]);
        int n1 = (i & 1) ? (b1 >> 4) : (b1 & 0x0F);
        float s1 = e8m0_to_f32(__ldg(&d.w1_scales[(size_t)row * (H / 32) + g]));
        gate += c_e2m1_tab[n1] * s1 * x[(size_t)pos * H + i];
        uint8_t b3 = __ldg(&d.w3_packed[(size_t)row * (H / 2) + bi]);
        int n3 = (i & 1) ? (b3 >> 4) : (b3 & 0x0F);
        float s3 = e8m0_to_f32(__ldg(&d.w3_scales[(size_t)row * (H / 32) + g]));
        up += c_e2m1_tab[n3] * s3 * x[(size_t)pos * H + i];
    }

    for (int s = 16; s > 0; s >>= 1) {
        gate += __shfl_xor_sync(0xFFFFFFFF, gate, s);
        up   += __shfl_xor_sync(0xFFFFFFFF, up,   s);
    }

    float sig = 1.0f / (1.0f + expf(-gate));
    float hv = (4.0f * tanhf(gate / 4.0f) * sig) * (25.0f * tanhf(up / 25.0f));
    if (lane == 0) {
        size_t hidx = ((size_t)pos * (size_t)max_experts + (size_t)e) * (size_t)I + (size_t)row;
        hidden[hidx] = hv;
    }
}

// Kernel 2: output
__global__ void mxfp4_moe_pos_output_kernel(
    float* __restrict__ out,
    const float* __restrict__ hidden,
    const struct PosExpertDesc* __restrict__ descs,
    const int* __restrict__ expert_counts,
    int H, int I, int max_experts)
{
    int pos = blockIdx.z;
    int row_base = blockIdx.x * 8;
    int e = blockIdx.y;
    int warp_id = threadIdx.y;
    int row = row_base + warp_id;
    int ne = expert_counts[pos];
    if (row >= H || e >= ne) return;

    int flat = pos * max_experts + e;
    const struct PosExpertDesc d = descs[flat];
    int lane = threadIdx.x;

    float dot = 0.0f;
    for (int i = lane; i < I; i += 32) {
        int g = i / 32;
        int bi = i / 2;
        uint8_t b = __ldg(&d.w2_packed[(size_t)row * (I / 2) + bi]);
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        float sc = e8m0_to_f32(__ldg(&d.w2_scales[(size_t)row * (I / 32) + g]));
        dot += c_e2m1_tab[n] * sc *
            hidden[((size_t)pos * (size_t)max_experts + (size_t)e) * (size_t)I + (size_t)i];
    }

    for (int s = 16; s > 0; s >>= 1)
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, s);

    if (lane == 0)
        atomicAdd(&out[(size_t)pos * H + row], dot * d.routing_weight);
}

// ═══════════════════════ host-side API ═══════════════════════════════════════

extern "C" {

// ── query: check CUDA device availability ─────────────────────────────────
int cuda_moe_available(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return 0;
    cudaStream_t s;
    err = cudaStreamCreate(&s);
    if (err != cudaSuccess) return 0;
    cudaStreamDestroy(s);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    if (major * 10 + minor < 75) return 0;  // sm_75+ required
    return 1;
}

// ── phase 1: per-matrix MXFP4 GEMV ────────────────────────────────────────
void cuda_mxfp4_gemv(
    float* y, const float* x,
    const uint8_t* packed, const uint8_t* scales,
    int rows, int cols,
    cudaStream_t stream)
{
    mxfp4_gemv_kernel<<<(unsigned)rows, 256u, 0, stream>>>(
        y, x, packed, scales, rows, cols);
}

// Hidden buffer shared across cuda_mxfp4_moe_layer calls.
// Single-threaded usage in Deltafin's MoE dispatch.
static float* g_hidden = NULL;
static size_t g_hidden_cap = 0;

static float* _moe_hidden_buf(size_t needed) {
    if (needed > g_hidden_cap) {
        if (g_hidden) cudaFree(g_hidden);
        cudaMalloc(&g_hidden, needed);
        g_hidden_cap = needed;
    }
    return g_hidden;
}


// ── phase 3: grouped MoE layer dispatch ───────────────────────────────────
void cuda_mxfp4_moe_layer(
    float* out, const float* x,
    const struct ExpertDesc* descs,
    int H, int I, int num_experts,
    cudaStream_t stream)
{
    float* hidden = _moe_hidden_buf((size_t)num_experts * (size_t)I * sizeof(float));

    // Kernel 1: grid(I/8, E) — each block = 8 rows across 8 warps
    dim3 g_hid(((unsigned)I + 7) / 8, (unsigned)num_experts);
    dim3 blk(32, 8);  // 8 warps × 32 threads = 256 threads
    mxfp4_moe_hidden_kernel<<<g_hid, blk, 0, stream>>>(
        hidden, x, descs, H, I);

    // Kernel 2: grid(H/8, E)
    dim3 g_out(((unsigned)H + 7) / 8, (unsigned)num_experts);
    mxfp4_moe_output_kernel<<<g_out, blk, 0, stream>>>(
        out, hidden, descs, H, I);
}

// ── phase 3b: reset output buffer to zero ─────────────────────────────────
void cuda_moe_zero_output(float* out, int n, cudaStream_t stream) {
    cudaMemsetAsync(out, 0, (size_t)n * sizeof(float), stream);
}

// ── phase 4: int8 spine dequant ───────────────────────────────────────────
void cuda_int8_deq(
    float* out, const int8_t* q, const half* sc,
    int rows, int cols,
    cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    int8_deq_kernel<<<grid, block, 0, stream>>>(out, q, sc, rows, cols);
}

// Hidden buffer for multi-position kernels.
static float* g_pos_hidden = NULL;
static size_t g_pos_hidden_cap = 0;

static float* _pos_hidden_buf(size_t needed) {
    if (needed > g_pos_hidden_cap) {
        if (g_pos_hidden) cudaFree(g_pos_hidden);
        cudaMalloc(&g_pos_hidden, needed);
        g_pos_hidden_cap = needed;
    }
    return g_pos_hidden;
}


// ── phase 5: multi-position 2-kernel dispatch ─────────────────────────────
void cuda_mxfp4_moe_positions(
    float* out, const float* x,
    const struct PosExpertDesc* descs,
    const int* expert_counts,
    int H, int I, int num_positions, int max_experts,
    cudaStream_t stream)
{
    float* hidden = _pos_hidden_buf(
        (size_t)num_positions * (size_t)max_experts * (size_t)I * sizeof(float));

    dim3 blk(32, 8);
    // Kernel 1: grid(I/8, max_experts, N)
    dim3 g1(((unsigned)I + 7) / 8, (unsigned)max_experts, (unsigned)num_positions);
    mxfp4_moe_pos_hidden_kernel<<<g1, blk, 0, stream>>>(
        hidden, x, descs, expert_counts, H, I, max_experts);

    // Kernel 2: grid(H/8, max_experts, N)
    dim3 g2(((unsigned)H + 7) / 8, (unsigned)max_experts, (unsigned)num_positions);
    mxfp4_moe_pos_output_kernel<<<g2, blk, 0, stream>>>(
        out, hidden, descs, expert_counts, H, I, max_experts);
}

// ── error string ──────────────────────────────────────────────────────────
const char* cuda_moe_error(void) {
    return cudaGetErrorString(cudaGetLastError());
}

}  // extern "C"
