// Deltafin CUDA MXFP4 kernels.
//
// This translation unit is deliberately stateless.  It never calls cudaMalloc,
// owns no device buffers, and does not retain a pointer after an API call.  The
// Python bridge owns every allocation and passes PyTorch's current CUDA stream
// explicitly.  That makes device, stream, allocator, and lifetime ownership
// visible at the language boundary instead of hiding them in process globals.
//
// Expert layout version 1 is the 17,547,264-byte on-disk span:
//   w1 packed, w1 scale, w2 packed, w2 scale, w3 packed, w3 scale.
//
// Build integration lives in tools/build_native.py.  Keep every exported
// function returning a checked status except the side-effect-free handshake.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kPointerLayout = 1;
constexpr int kHidden = 3584;
constexpr int kIntermediate = 3072;
constexpr int64_t kPackedBytes = 5'505'024;
constexpr int64_t kScaleBytes = 344'064;
constexpr int64_t kExpertSpan = 17'547'264;
constexpr int kThreads = 128;

// An error message is diagnostic state, not an allocation or a retained CUDA
// resource.  thread_local prevents concurrent Python callers from trampling one
// another's message.
thread_local char g_last_error[512] = "no error";

enum Status : int {
    kOk = 0,
    kInvalidArgument = -1,
    kCudaError = -2,
    kUnsupported = -3,
};

void set_error(const char* operation, const char* detail) {
    std::snprintf(
        g_last_error,
        sizeof(g_last_error),
        "%s: %s",
        operation ? operation : "CUDA operation",
        detail ? detail : "unknown error");
}

int cuda_status(const char* operation, cudaError_t status) {
    if (status == cudaSuccess) {
        return kOk;
    }
    set_error(operation, cudaGetErrorString(status));
    return kCudaError;
}

int reject(const char* operation, const char* detail) {
    set_error(operation, detail);
    return kInvalidArgument;
}

__device__ __forceinline__ float e2m1_value(uint8_t code) {
    float magnitude;
    switch (code & 7u) {
        case 0: magnitude = 0.0f; break;
        case 1: magnitude = 0.5f; break;
        case 2: magnitude = 1.0f; break;
        case 3: magnitude = 1.5f; break;
        case 4: magnitude = 2.0f; break;
        case 5: magnitude = 3.0f; break;
        case 6: magnitude = 4.0f; break;
        default: magnitude = 6.0f; break;
    }
    return (code & 8u) ? -magnitude : magnitude;
}

// One block owns one complete output element.  Each thread sums whole MXFP4
// groups, then the fixed binary tree below reduces in a repeatable order.  No
// atomic operation participates in either an expert output or route combine.
__global__ void mxfp4_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols,
    int batches) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x);
    const int64_t total_outputs =
        static_cast<int64_t>(rows) * static_cast<int64_t>(batches);
    if (output_index >= total_outputs) {
        return;
    }

    const int row = static_cast<int>(output_index % rows);
    const int batch = static_cast<int>(output_index / rows);
    const int groups = cols / 32;
    const uint8_t* row_packed =
        packed + static_cast<int64_t>(row) * (cols / 2);
    const uint8_t* row_scales =
        scales + static_cast<int64_t>(row) * groups;
    const float* batch_x = x + static_cast<int64_t>(batch) * cols;

    float local = 0.0f;
    for (int group = threadIdx.x; group < groups; group += blockDim.x) {
        const uint8_t scale_byte = row_scales[group];
        if (scale_byte == 255u) {
#ifndef CUDART_NAN_F
            // CUDA 13.x dropped this macro; provide the IEEE 754 quiet NaN encoding.
#define CUDART_NAN_F  __int_as_float(0x7fffffff)
#endif
            local = CUDART_NAN_F;
            break;
        }
        const int exponent = static_cast<int>(scale_byte) - 127;
        const int column_base = group * 32;
        const int packed_base = group * 16;
#pragma unroll
        for (int lane = 0; lane < 32; ++lane) {
            const uint8_t byte = row_packed[packed_base + (lane >> 1)];
            const uint8_t code =
                static_cast<uint8_t>((byte >> ((lane & 1) * 4)) & 0x0f);
            const float weight = ldexpf(e2m1_value(code), exponent);
            local = fmaf(weight, batch_x[column_base + lane], local);
        }
    }

    __shared__ float partial[kThreads];
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[output_index] = partial[0];
    }
}

__global__ void situ_kernel(
    float* __restrict__ gate,
    const float* __restrict__ up,
    int64_t count) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    constexpr float beta = 4.0f;
    constexpr float linear_beta = 25.0f;
    const float g = gate[index];
    const float u = up[index];
    const float bounded_gate =
        beta * tanhf(g / beta) / (1.0f + expf(-g));
    const float bounded_up = linear_beta * tanhf(u / linear_beta);
    gate[index] = bounded_gate * bounded_up;
}

// One flattened grid covers every element.  In particular, there is no
// rows-sized grid whose y dimension silently truncates a wide matrix.
__global__ void int8_dequant_kernel(
    const int8_t* __restrict__ quantized,
    const __half* __restrict__ scales,
    float* __restrict__ output,
    int64_t count,
    int cols) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int64_t row = index / cols;
    output[index] =
        static_cast<float>(quantized[index]) * __half2float(scales[row]);
}

int validate_gemv(
    const uint8_t* packed,
    const uint8_t* scales,
    const float* x,
    float* output,
    int rows,
    int cols,
    int batches) {
    if (!packed || !scales || !x || !output) {
        return reject("mxfp4 GEMV", "null input or output pointer");
    }
    if (rows <= 0 || cols <= 0 || batches <= 0 || (cols % 32) != 0) {
        return reject(
            "mxfp4 GEMV",
            "rows/batches must be positive and cols must be a positive multiple of 32");
    }
    const int64_t blocks =
        static_cast<int64_t>(rows) * static_cast<int64_t>(batches);
    if (blocks > std::numeric_limits<int>::max()) {
        return reject("mxfp4 GEMV", "launch grid is too large");
    }
    return kOk;
}

int launch_gemv(
    const uint8_t* packed,
    const uint8_t* scales,
    const float* x,
    float* output,
    int rows,
    int cols,
    int batches,
    cudaStream_t stream) {
    const int valid =
        validate_gemv(packed, scales, x, output, rows, cols, batches);
    if (valid != kOk) {
        return valid;
    }
    const int blocks = static_cast<int>(
        static_cast<int64_t>(rows) * static_cast<int64_t>(batches));
    mxfp4_gemv_kernel<<<blocks, kThreads, 0, stream>>>(
        packed, scales, x, output, rows, cols, batches);
    return cuda_status("mxfp4 GEMV launch", cudaPeekAtLastError());
}

}  // namespace

extern "C" uint32_t k3_cuda_moe_abi_version(void) {
    return kAbiVersion;
}

extern "C" void k3_cuda_moe_shapes(
    int* hidden,
    int* intermediate,
    int64_t* expert_span,
    uint32_t* pointer_layout) {
    if (hidden) {
        *hidden = kHidden;
    }
    if (intermediate) {
        *intermediate = kIntermediate;
    }
    if (expert_span) {
        *expert_span = kExpertSpan;
    }
    if (pointer_layout) {
        *pointer_layout = kPointerLayout;
    }
}

extern "C" const char* k3_cuda_last_error(void) {
    return g_last_error;
}

extern "C" int k3_cuda_moe_available(int device) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return cuda_status("cudaGetDeviceCount", status);
    }
    if (device < 0 || device >= count) {
        set_error("CUDA availability", "device index is outside the visible range");
        return kUnsupported;
    }
    int compute_mode = -1;
#if CUDART_VERSION >= 12000
    // CUDA 12+ uses the attribute API; direct struct access was deprecated.
    status = cudaDeviceGetAttribute(
        &compute_mode, cudaDevAttrComputeMode, device);
    if (status != cudaSuccess) {
        return cuda_status("cudaDeviceGetAttribute(computeMode)", status);
    }
#else
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return cuda_status("cudaGetDeviceProperties", status);
    }
    compute_mode = properties.computeMode;
#endif
    if (compute_mode == cudaComputeModeProhibited) {
        set_error("CUDA availability", "device compute mode is prohibited");
        return kUnsupported;
    }

#if CUDART_VERSION >= 12000
    int major = 0, minor = 0;
    status = cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) {
        return cuda_status("cudaDeviceGetAttribute(major)", status);
    }
    status = cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, device);
    if (status != cudaSuccess) {
        return cuda_status("cudaDeviceGetAttribute(minor)", status);
    }
    std::snprintf(
        g_last_error,
        sizeof(g_last_error),
        "available: device %d compute capability %d.%d",
        device,
        major,
        minor);
#else
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return cuda_status("cudaGetDeviceProperties", status);
    }
    std::snprintf(
        g_last_error,
        sizeof(g_last_error),
        "available: %s compute capability %d.%d",
        properties.name,
        properties.major,
        properties.minor);
#endif
    return 1;
}

extern "C" int k3_cuda_mxfp4_gemv(
    const uint8_t* packed,
    const uint8_t* scales,
    const float* x,
    float* output,
    int rows,
    int cols,
    int batches,
    void* stream_pointer) {
    return launch_gemv(
        packed,
        scales,
        x,
        output,
        rows,
        cols,
        batches,
        reinterpret_cast<cudaStream_t>(stream_pointer));
}

extern "C" int k3_cuda_moe_launch(
    const uint8_t* expert_span,
    const float* x,
    int batches,
    float* gate,
    float* up,
    float* expert_output,
    void* stream_pointer) {
    if (!expert_span || !x || !gate || !up || !expert_output) {
        return reject("CUDA MoE", "null input, scratch, or output pointer");
    }
    if (batches <= 0) {
        return reject("CUDA MoE", "batch count must be positive");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);

    const uint8_t* w1_packed = expert_span;
    const uint8_t* w1_scale = expert_span + kPackedBytes;
    const uint8_t* w2_packed = expert_span + kPackedBytes + kScaleBytes;
    const uint8_t* w2_scale =
        expert_span + 2 * kPackedBytes + kScaleBytes;
    const uint8_t* w3_packed =
        expert_span + 2 * (kPackedBytes + kScaleBytes);
    const uint8_t* w3_scale =
        expert_span + 3 * kPackedBytes + 2 * kScaleBytes;

    int status = launch_gemv(
        w1_packed,
        w1_scale,
        x,
        gate,
        kIntermediate,
        kHidden,
        batches,
        stream);
    if (status != kOk) {
        return status;
    }
    status = launch_gemv(
        w3_packed,
        w3_scale,
        x,
        up,
        kIntermediate,
        kHidden,
        batches,
        stream);
    if (status != kOk) {
        return status;
    }

    const int64_t activation_count =
        static_cast<int64_t>(batches) * kIntermediate;
    const int64_t activation_blocks =
        (activation_count + kThreads - 1) / kThreads;
    if (activation_blocks > std::numeric_limits<int>::max()) {
        return reject("CUDA MoE", "activation launch grid is too large");
    }
    situ_kernel<<<
        static_cast<int>(activation_blocks), kThreads, 0, stream>>>(
        gate, up, activation_count);
    status = cuda_status("SiTU launch", cudaPeekAtLastError());
    if (status != kOk) {
        return status;
    }

    return launch_gemv(
        w2_packed,
        w2_scale,
        gate,
        expert_output,
        kHidden,
        kIntermediate,
        batches,
        stream);
}

extern "C" int k3_cuda_int8_dequant(
    const int8_t* quantized,
    const uint16_t* scales_fp16,
    float* output,
    int64_t count,
    int cols,
    void* stream_pointer) {
    if (!quantized || !scales_fp16 || !output) {
        return reject("int8 dequant", "null input, scale, or output pointer");
    }
    if (count <= 0 || cols <= 0 || (count % cols) != 0) {
        return reject(
            "int8 dequant",
            "count and cols must describe a non-empty rectangular matrix");
    }
    const int64_t blocks = 1 + (count - 1) / 256;
    if (blocks > std::numeric_limits<int>::max()) {
        return reject("int8 dequant", "launch grid is too large");
    }
    int8_dequant_kernel<<<
        static_cast<int>(blocks),
        256,
        0,
        reinterpret_cast<cudaStream_t>(stream_pointer)>>>(
        quantized,
        reinterpret_cast<const __half*>(scales_fp16),
        output,
        count,
        cols);
    return cuda_status("int8 dequant launch", cudaPeekAtLastError());
}
