// Fused MXFP4 dequant + GEMV kernel for Kimi-K3 experts.
// Never materializes the fp32 weight matrix.
//
// Layout: packed U8 [rows, cols/2] (two e2m1 nibbles/byte, LOW first),
//         scales U8 [rows, cols/32] (e8m0: 2^(s-127)).
// GEMV: y[r] = dot(W[r,:], x)  (row-major, streaming rows)
//
// macOS arm64:
//   clang -O3 -mcpu=native -shared -DNO_MAIN -o libmxfp4gemv.dylib fused_gemv.c -lpthread
// Linux aarch64:
//   cc -O3 -march=native -fPIC -shared -DNO_MAIN -o libmxfp4gemv.so fused_gemv.c -lpthread -lm
// Linux x86-64 (runtime AVX2 dispatch, exact AVX/FMA3/SSSE3 baseline):
//   cc -O3 -march=x86-64 -mtune=native -msse3 -mssse3 -mavx -mfma \
//      -fPIC -shared -DNO_MAIN -o libmxfp4gemv.so fused_gemv.c -lpthread -lm

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define MXFP4_ARCH_ARM64 1
#elif defined(__x86_64__) || defined(_M_X64)
#include "neon_compat_x86.h"
#define MXFP4_ARCH_X86_64 1
#else
#error "The MXFP4 kernel supports only arm64 and x86-64"
#endif

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) \
        && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The MXFP4 SIMD table expansion currently requires little-endian byte order"
#endif

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#if defined(_WIN32)
// Supplies the pthread, sched and clock_gettime subset used below, plus the
// aligned allocator and the export attribute a DLL needs.
#include "win_compat.h"
#else
#include <pthread.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif
#include <sched.h>
// Mach-O and ELF export every non-static symbol, and both toolchains take the
// aligned attribute in the same declaration slots.
#define K3_EXPORT
#define K3_ALIGNAS(n) __attribute__((aligned(n)))
#define K3_ALIGN_PREFIX(n) __attribute__((aligned(n)))
static inline void *k3_aligned_alloc(size_t alignment, size_t size) {
    return aligned_alloc(alignment, size);
}
static inline void k3_aligned_free(void *ptr) {
    free(ptr);
}
#endif

#ifndef NO_MAIN
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define MXFP4_HAVE_CBLAS 1
#elif defined(MXFP4_USE_CBLAS)
#include <cblas.h>
#define MXFP4_HAVE_CBLAS 1
#else
#define MXFP4_HAVE_CBLAS 0
#endif
#endif

// x86 builds deliberately keep AVX2 inside separately marked functions.  The
// rest of the shared library needs only the exact 128-bit compatibility ISA, so
// one binary can run on pre-AVX2 FMA hosts and select AVX2 at runtime.
//
// GCC and Clang need the target attribute to accept AVX2 intrinsics in a
// translation unit compiled for a lower baseline.  MSVC instead accepts every
// intrinsic regardless of /arch and emits AVX2 only where one is written, so the
// same isolation holds with no attribute; /arch:AVX keeps the surrounding
// baseline code VEX-encoded exactly as -mavx does.
#if defined(MXFP4_ARCH_X86_64) \
        && (defined(__GNUC__) || defined(__clang__))
#define MXFP4_CAN_BUILD_AVX2 1
#define MXFP4_TARGET_AVX2 __attribute__((target("avx2,fma")))
#elif defined(MXFP4_ARCH_X86_64) && defined(_MSC_VER)
#define MXFP4_CAN_BUILD_AVX2 1
#define MXFP4_TARGET_AVX2
#else
#define MXFP4_CAN_BUILD_AVX2 0
#define MXFP4_TARGET_AVX2
#endif

// Keep Darwin's latency-sensitive scheduling policy.  These APIs do not exist on
// Linux, where the helpers deliberately become no-ops.
static inline void mxfp4_qos_self(void) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static inline void mxfp4_qos_attr(pthread_attr_t *attr) {
#if defined(__APPLE__)
    pthread_attr_set_qos_class_np(attr, QOS_CLASS_USER_INTERACTIVE, 0);
#else
    (void)attr;
#endif
}

static inline void mxfp4_cpu_relax(void) {
#if defined(_MSC_VER)
    // MSVC has no GNU inline assembler; these intrinsics emit the same YIELD
    // and PAUSE instructions.
#if defined(MXFP4_ARCH_ARM64)
    __yield();
#elif defined(MXFP4_ARCH_X86_64)
    _mm_pause();
#endif
#elif defined(MXFP4_ARCH_ARM64)
    __asm__ volatile("yield");
#elif defined(MXFP4_ARCH_X86_64)
    __asm__ volatile("pause");
#endif
}

#if defined(MXFP4_TEST_PTHREAD_FAIL_AFTER)
static _Atomic int mxfp4_test_pthread_create_calls = 0;

K3_EXPORT void mxfp4_test_reset_pthread_create(void) {
    atomic_store_explicit(
        &mxfp4_test_pthread_create_calls, 0, memory_order_relaxed);
}
#endif

static inline int mxfp4_pthread_create(
        pthread_t *thread, const pthread_attr_t *attr,
        void *(*start_routine)(void *), void *arg) {
#if defined(MXFP4_TEST_PTHREAD_FAIL_AFTER)
    int call = atomic_fetch_add_explicit(
        &mxfp4_test_pthread_create_calls, 1, memory_order_relaxed);
    if (call >= MXFP4_TEST_PTHREAD_FAIL_AFTER) return 1;
#endif
    return pthread_create(thread, attr, start_routine, arg);
}

static double now_s(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Stable loader handshake.  Increment only for an incompatible exported ABI change.
K3_EXPORT uint32_t mxfp4_abi_version(void) {
    return 1u;
}

// ---------------- shared tables ----------------
// fp32 bit patterns of E2M1 values occupy only the top 16 bits. Applying an
// E8M0 scale is integer addition of (scale-127)<<7 to that halfword, masked so
// +/-0 remain zero. The old hot loop synthesized the same 32 table bytes for
// every weight group. The exact 8 KiB lookup below moves that work to .rodata.
//
// MXFP4_USE_SCALE_LUT=0 retains the original synthesis implementation as a
// compile-time correctness oracle. It is deliberately not a runtime branch.
#ifndef MXFP4_USE_SCALE_LUT
#define MXFP4_USE_SCALE_LUT 1
#endif

#if MXFP4_USE_SCALE_LUT
typedef struct K3_ALIGNAS(32) {
    uint8_t lo[16];
    uint8_t hi[16];
} mxfp4_scale_lut_row_t;

// Every expression is an integer constant expression. Conversion to uint16_t
// is defined modulo 2^16, including scales below 127, and the final cast
// reproduces vaddq_u16/_mm256_add_epi16 wraparound for all 256 input bytes.
#define MXFP4_SCALE_DELTA_CONST(s) \
    ((uint16_t)(((int)(s) - 127) * 128))
#define MXFP4_SCALE_BITS_CONST(s, base, mask) \
    ((uint16_t)((uint16_t)(base) \
        + (uint16_t)(MXFP4_SCALE_DELTA_CONST(s) & (uint16_t)(mask))))
#define MXFP4_SCALE_LO_CONST(s, base, mask) \
    ((uint8_t)MXFP4_SCALE_BITS_CONST(s, base, mask))
#define MXFP4_SCALE_HI_CONST(s, base, mask) \
    ((uint8_t)(MXFP4_SCALE_BITS_CONST(s, base, mask) >> 8))
#define MXFP4_SCALE_LUT_ROW(s) { \
    { \
        MXFP4_SCALE_LO_CONST(s, 0x0000, 0x0000), \
        MXFP4_SCALE_LO_CONST(s, 0x3F00, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0x3F80, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0x3FC0, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0x4000, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0x4040, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0x4080, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0x40C0, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0x8000, 0x0000), \
        MXFP4_SCALE_LO_CONST(s, 0xBF00, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0xBF80, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0xBFC0, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0xC000, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0xC040, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0xC080, 0xFFFF), \
        MXFP4_SCALE_LO_CONST(s, 0xC0C0, 0xFFFF) \
    }, { \
        MXFP4_SCALE_HI_CONST(s, 0x0000, 0x0000), \
        MXFP4_SCALE_HI_CONST(s, 0x3F00, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0x3F80, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0x3FC0, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0x4000, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0x4040, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0x4080, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0x40C0, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0x8000, 0x0000), \
        MXFP4_SCALE_HI_CONST(s, 0xBF00, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0xBF80, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0xBFC0, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0xC000, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0xC040, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0xC080, 0xFFFF), \
        MXFP4_SCALE_HI_CONST(s, 0xC0C0, 0xFFFF) \
    } \
}
#define MXFP4_SCALE_LUT_ROWS_16(s) \
    MXFP4_SCALE_LUT_ROW((s) + 0),  MXFP4_SCALE_LUT_ROW((s) + 1), \
    MXFP4_SCALE_LUT_ROW((s) + 2),  MXFP4_SCALE_LUT_ROW((s) + 3), \
    MXFP4_SCALE_LUT_ROW((s) + 4),  MXFP4_SCALE_LUT_ROW((s) + 5), \
    MXFP4_SCALE_LUT_ROW((s) + 6),  MXFP4_SCALE_LUT_ROW((s) + 7), \
    MXFP4_SCALE_LUT_ROW((s) + 8),  MXFP4_SCALE_LUT_ROW((s) + 9), \
    MXFP4_SCALE_LUT_ROW((s) + 10), MXFP4_SCALE_LUT_ROW((s) + 11), \
    MXFP4_SCALE_LUT_ROW((s) + 12), MXFP4_SCALE_LUT_ROW((s) + 13), \
    MXFP4_SCALE_LUT_ROW((s) + 14), MXFP4_SCALE_LUT_ROW((s) + 15)

K3_ALIGN_PREFIX(64) static const mxfp4_scale_lut_row_t mxfp4_scale_lut[256] = {
    MXFP4_SCALE_LUT_ROWS_16(0),
    MXFP4_SCALE_LUT_ROWS_16(16),
    MXFP4_SCALE_LUT_ROWS_16(32),
    MXFP4_SCALE_LUT_ROWS_16(48),
    MXFP4_SCALE_LUT_ROWS_16(64),
    MXFP4_SCALE_LUT_ROWS_16(80),
    MXFP4_SCALE_LUT_ROWS_16(96),
    MXFP4_SCALE_LUT_ROWS_16(112),
    MXFP4_SCALE_LUT_ROWS_16(128),
    MXFP4_SCALE_LUT_ROWS_16(144),
    MXFP4_SCALE_LUT_ROWS_16(160),
    MXFP4_SCALE_LUT_ROWS_16(176),
    MXFP4_SCALE_LUT_ROWS_16(192),
    MXFP4_SCALE_LUT_ROWS_16(208),
    MXFP4_SCALE_LUT_ROWS_16(224),
    MXFP4_SCALE_LUT_ROWS_16(240),
};

_Static_assert(sizeof(mxfp4_scale_lut_row_t) == 32,
               "E8M0 LUT rows must stay one aligned AVX2 load");

#undef MXFP4_SCALE_LUT_ROWS_16
#undef MXFP4_SCALE_LUT_ROW
#undef MXFP4_SCALE_HI_CONST
#undef MXFP4_SCALE_LO_CONST
#undef MXFP4_SCALE_BITS_CONST
#undef MXFP4_SCALE_DELTA_CONST
#else
static const uint16_t base16[16] = {
    0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
    0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0
};
static const uint16_t mask16[16] = {
    0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF
};
#endif

static const float e2m1_tab[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
static inline float e8m0_to_f32(uint8_t s) {
    union { uint32_t u; float f; } v; v.u = (uint32_t)s << 23; return v.f;
}

#if !MXFP4_USE_SCALE_LUT
// Multiplication is deliberate: left-shifting a negative signed value is
// undefined in C. Conversion to uint16_t is defined modulo 2^16 and produces
// the same exponent-field delta for scales below and above the 127 bias.
static inline uint16_t e8m0_exp_delta_u16(uint8_t s) {
    return (uint16_t)(((int)s - 127) * 128);
}
#endif

static inline void mxfp4_scale_tables(
        uint8_t scale, uint8x16_t *lo, uint8x16_t *hi) {
#if MXFP4_USE_SCALE_LUT
    const mxfp4_scale_lut_row_t *row = &mxfp4_scale_lut[scale];
    *lo = vld1q_u8(row->lo);
    *hi = vld1q_u8(row->hi);
#else
    const uint16x8_t b0 = vld1q_u16(base16);
    const uint16x8_t b1 = vld1q_u16(base16 + 8);
    const uint16x8_t z0 = vld1q_u16(mask16);
    const uint16x8_t z1 = vld1q_u16(mask16 + 8);
    const uint16x8_t delta = vdupq_n_u16(e8m0_exp_delta_u16(scale));
    const uint8x16_t t0 = vreinterpretq_u8_u16(
        vaddq_u16(b0, vandq_u16(delta, z0)));
    const uint8x16_t t1 = vreinterpretq_u8_u16(
        vaddq_u16(b1, vandq_u16(delta, z1)));
    *lo = vuzp1q_u8(t0, t1);
    *hi = vuzp2q_u8(t0, t1);
#endif
}

// identical accumulator reduction for ref and fused (bit-exact comparability)
static inline float reduce8(float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3,
                            float32x4_t a4, float32x4_t a5, float32x4_t a6, float32x4_t a7) {
    float32x4_t b0 = vaddq_f32(a0, a1), b1 = vaddq_f32(a2, a3);
    float32x4_t b2 = vaddq_f32(a4, a5), b3 = vaddq_f32(a6, a7);
    float32x4_t c0 = vaddq_f32(b0, b1), c1 = vaddq_f32(b2, b3);
    return vaddvq_f32(vaddq_f32(c0, c1));
}

// ---------------- reference: scalar dequant row -> buffer, NEON dot (same order) ----------------
static void ref_gemv(const uint8_t *restrict p, const uint8_t *restrict s,
                     const float *restrict x, float *restrict y,
                     int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
    float *wbuf = malloc(cols * sizeof(float));
    for (int r = row0; r < row1; r++) {
        const uint8_t *pp = p + (size_t)r * cp;
        const uint8_t *sp = s + (size_t)r * groups;
        for (int g = 0; g < groups; g++) {
            float scale = e8m0_to_f32(sp[g]);
            const uint8_t *pb = pp + g * 16;
            float *ob = wbuf + g * 32;
            for (int i = 0; i < 16; i++) {
                uint8_t b = pb[i];
                ob[2 * i]     = e2m1_tab[b & 0xF] * scale;
                ob[2 * i + 1] = e2m1_tab[b >> 4]  * scale;
            }
        }
        float32x4_t acc[8];
        for (int j = 0; j < 8; j++) acc[j] = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            const float *wp = wbuf + g * 32, *xp = x + g * 32;
            for (int j = 0; j < 8; j++)
                acc[j] = vfmaq_f32(acc[j], vld1q_f32(wp + 4 * j), vld1q_f32(xp + 4 * j));
        }
        y[r] = reduce8(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
    }
    free(wbuf);
}

// ---------------- fused kernel: TBL -> fp32-bit halves, exp add, widen, FMA ----------------
K3_EXPORT void mxfp4_gemv_rows(
        const uint8_t *restrict p, const uint8_t *restrict s,
        const float *restrict x, float *restrict y,
        int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
    const uint8x16_t M0F = vdupq_n_u8(0x0F);
    for (int r = row0; r < row1; r++) {
        const uint8_t *pp = p + (size_t)r * cp;
        const uint8_t *sp = s + (size_t)r * groups;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0), a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            uint8x16_t TABL, TABH;
            mxfp4_scale_tables(sp[g], &TABL, &TABH);
            uint8x16_t P  = vld1q_u8(pp + g * 16);
            uint8x16_t lo = vandq_u8(P, M0F);
            uint8x16_t hi = vshrq_n_u8(P, 4);
            uint8x16_t n0 = vzip1q_u8(lo, hi);   // logical elements 0..15
            uint8x16_t n1 = vzip2q_u8(lo, hi);   // elements 16..31
            const float *xp = x + g * 32;
            {
                uint8x16_t bl = vqtbl1q_u8(TABL, n0), bh = vqtbl1q_u8(TABH, n0);
                uint16x8_t h0 = vreinterpretq_u16_u8(vzip1q_u8(bl, bh));
                uint16x8_t h1 = vreinterpretq_u16_u8(vzip2q_u8(bl, bh));
                a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)), vld1q_f32(xp + 0));
                a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_high_n_u16(h0, 16)),          vld1q_f32(xp + 4));
                a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)), vld1q_f32(xp + 8));
                a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_high_n_u16(h1, 16)),          vld1q_f32(xp + 12));
            }
            {
                uint8x16_t bl = vqtbl1q_u8(TABL, n1), bh = vqtbl1q_u8(TABH, n1);
                uint16x8_t h0 = vreinterpretq_u16_u8(vzip1q_u8(bl, bh));
                uint16x8_t h1 = vreinterpretq_u16_u8(vzip2q_u8(bl, bh));
                a4 = vfmaq_f32(a4, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)), vld1q_f32(xp + 16));
                a5 = vfmaq_f32(a5, vreinterpretq_f32_u32(vshll_high_n_u16(h0, 16)),          vld1q_f32(xp + 20));
                a6 = vfmaq_f32(a6, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)), vld1q_f32(xp + 24));
                a7 = vfmaq_f32(a7, vreinterpretq_f32_u32(vshll_high_n_u16(h1, 16)),          vld1q_f32(xp + 28));
            }
        }
        y[r] = reduce8(a0, a1, a2, a3, a4, a5, a6, a7);
    }
}

// ---- variant: 2 rows per pass, shared x loads, same per-row accumulation order ----
K3_EXPORT void mxfp4_gemv_rows2(
        const uint8_t *restrict p, const uint8_t *restrict s,
        const float *restrict x, float *restrict y,
        int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
    const uint8x16_t M0F = vdupq_n_u8(0x0F);
    int r = row0;
    for (; r + 1 < row1; r += 2) {
        const uint8_t *ppA = p + (size_t)r * cp,      *spA = s + (size_t)r * groups;
        const uint8_t *ppB = p + (size_t)(r + 1) * cp, *spB = s + (size_t)(r + 1) * groups;
        float32x4_t A0 = vdupq_n_f32(0), A1 = vdupq_n_f32(0), A2 = vdupq_n_f32(0), A3 = vdupq_n_f32(0);
        float32x4_t A4 = vdupq_n_f32(0), A5 = vdupq_n_f32(0), A6 = vdupq_n_f32(0), A7 = vdupq_n_f32(0);
        float32x4_t C0 = vdupq_n_f32(0), C1 = vdupq_n_f32(0), C2 = vdupq_n_f32(0), C3 = vdupq_n_f32(0);
        float32x4_t C4 = vdupq_n_f32(0), C5 = vdupq_n_f32(0), C6 = vdupq_n_f32(0), C7 = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            const float *xp = x + g * 32;
            float32x4_t x0 = vld1q_f32(xp + 0),  x1 = vld1q_f32(xp + 4);
            float32x4_t x2 = vld1q_f32(xp + 8),  x3 = vld1q_f32(xp + 12);
            float32x4_t x4 = vld1q_f32(xp + 16), x5 = vld1q_f32(xp + 20);
            float32x4_t x6 = vld1q_f32(xp + 24), x7 = vld1q_f32(xp + 28);
#define ROWGRP(pp, sp, c0, c1, c2, c3, c4, c5, c6, c7)                                              \
            {                                                                                        \
                uint8x16_t TABL, TABH;                                                               \
                mxfp4_scale_tables((sp)[g], &TABL, &TABH);                                           \
                uint8x16_t P  = vld1q_u8((pp) + g * 16);                                             \
                uint8x16_t lo = vandq_u8(P, M0F), hi = vshrq_n_u8(P, 4);                             \
                uint8x16_t n0 = vzip1q_u8(lo, hi), n1 = vzip2q_u8(lo, hi);                           \
                uint8x16_t bl0 = vqtbl1q_u8(TABL, n0), bh0 = vqtbl1q_u8(TABH, n0);                   \
                uint16x8_t h00 = vreinterpretq_u16_u8(vzip1q_u8(bl0, bh0));                          \
                uint16x8_t h01 = vreinterpretq_u16_u8(vzip2q_u8(bl0, bh0));                          \
                c0 = vfmaq_f32(c0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h00), 16)), x0);   \
                c1 = vfmaq_f32(c1, vreinterpretq_f32_u32(vshll_high_n_u16(h00, 16)), x1);            \
                c2 = vfmaq_f32(c2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h01), 16)), x2);   \
                c3 = vfmaq_f32(c3, vreinterpretq_f32_u32(vshll_high_n_u16(h01, 16)), x3);            \
                uint8x16_t bl1 = vqtbl1q_u8(TABL, n1), bh1 = vqtbl1q_u8(TABH, n1);                   \
                uint16x8_t h10 = vreinterpretq_u16_u8(vzip1q_u8(bl1, bh1));                          \
                uint16x8_t h11 = vreinterpretq_u16_u8(vzip2q_u8(bl1, bh1));                          \
                c4 = vfmaq_f32(c4, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h10), 16)), x4);   \
                c5 = vfmaq_f32(c5, vreinterpretq_f32_u32(vshll_high_n_u16(h10, 16)), x5);            \
                c6 = vfmaq_f32(c6, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h11), 16)), x6);   \
                c7 = vfmaq_f32(c7, vreinterpretq_f32_u32(vshll_high_n_u16(h11, 16)), x7);            \
            }
            ROWGRP(ppA, spA, A0, A1, A2, A3, A4, A5, A6, A7)
            ROWGRP(ppB, spB, C0, C1, C2, C3, C4, C5, C6, C7)
#undef ROWGRP
        }
        y[r]     = reduce8(A0, A1, A2, A3, A4, A5, A6, A7);
        y[r + 1] = reduce8(C0, C1, C2, C3, C4, C5, C6, C7);
    }
    if (r < row1) mxfp4_gemv_rows(p, s, x, y, r, row1, cols);
}

#if MXFP4_CAN_BUILD_AVX2
// ================= native 256-bit AVX2 kernel =================
//
// The exact compatibility path above maps each 128-bit NEON operation to SSSE3/FMA3.
// This version performs the same arithmetic with four 256-bit accumulators.  AVX2
// widening produces lanes in [0..3|16..19] order, so callers prepare x once in that
// order and reuse it for every row.  In particular, fused_gemv_batch.c prepares each
// distinct activation once per whole dispatch; there is no allocation or permutation
// in a worker or 32-row work unit.
#if !MXFP4_USE_SCALE_LUT
static const uint8_t mxfp4_avx2_uzp_bytes[16] = {
    0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
};
#endif

static int mxfp4_permute_x_avx2(
        const float *restrict x, float *restrict xp, int cols) {
    if (!x || !xp || cols <= 0 || cols % 32 != 0) return 0;
    for (int g = 0; g < cols / 32; g++) {
        const float *src = x + (size_t)g * 32;
        float *dst = xp + (size_t)g * 32;
        for (int q = 0; q < 4; q++) {
            memcpy(dst + q * 8,     src + q * 4,      4 * sizeof(float));
            memcpy(dst + q * 8 + 4, src + 16 + q * 4, 4 * sizeof(float));
        }
    }
    return 1;
}

static float *mxfp4_prepare_x_avx2(const float *x, int cols) {
    if (!x || cols <= 0 || cols % 32 != 0) return NULL;
    size_t n = (size_t)cols;
    if (n > SIZE_MAX / sizeof(float)) return NULL;
    size_t bytes = n * sizeof(float);
    if (bytes > SIZE_MAX - 63) return NULL;
    size_t aligned_bytes = (bytes + 63) & ~(size_t)63;
    float *xp = (float *)k3_aligned_alloc(64, aligned_bytes);
    if (!xp) return NULL;
    if (!mxfp4_permute_x_avx2(x, xp, cols)) {
        k3_aligned_free(xp);
        return NULL;
    }
    return xp;
}

// Load one interleaved 32-byte decode row and duplicate each 128-bit half into
// both AVX2 lanes. MXFP4_USE_SCALE_LUT=0 retains the original synthesis oracle.
#if MXFP4_USE_SCALE_LUT
#define MXFP4_AVX2_TABLE(scale_byte, TL, TH)                                               \
    do {                                                                                   \
        const __m256i _row = _mm256_load_si256(                                            \
            (const __m256i *)&mxfp4_scale_lut[(uint8_t)(scale_byte)]);                     \
        (TL) = _mm256_permute4x64_epi64(_row, 0x44);                                       \
        (TH) = _mm256_permute4x64_epi64(_row, 0xEE);                                       \
    } while (0)
#else
#define MXFP4_AVX2_TABLE(scale_byte, TL, TH)                                               \
    do {                                                                                   \
        __m256i _a = _mm256_set1_epi16(                                                    \
            (short)e8m0_exp_delta_u16((uint8_t)(scale_byte)));                             \
        __m256i _t = _mm256_add_epi16(BZ, _mm256_and_si256(_a, ZZ));                       \
        __m256i _u = _mm256_shuffle_epi8(_t, UZP);                                         \
        (TL) = _mm256_permute4x64_epi64(_u, 0x88);                                         \
        (TH) = _mm256_permute4x64_epi64(_u, 0xDD);                                         \
    } while (0)
#endif

// Expand 32 packed values into four fp32 vectors in prepared-x order, preserving
// the compatibility kernel's four independent FMA streams in each 128-bit lane.
#define MXFP4_AVX2_GROUP(pp, TL, TH, xq, c0, c1, c2, c3)                                  \
    do {                                                                                   \
        __m128i _p = _mm_loadu_si128(                                                      \
            (const __m128i *)((pp) + (size_t)g * 16));                                     \
        __m128i _lo = _mm_and_si128(_p, M0F);                                              \
        __m128i _hi = _mm_and_si128(_mm_srli_epi16(_p, 4), M0F);                           \
        __m256i _ix = _mm256_set_m128i(                                                    \
            _mm_unpackhi_epi8(_lo, _hi), _mm_unpacklo_epi8(_lo, _hi));                     \
        __m256i _bl = _mm256_shuffle_epi8((TL), _ix);                                      \
        __m256i _bh = _mm256_shuffle_epi8((TH), _ix);                                      \
        __m256i _w0 = _mm256_unpacklo_epi8(_bl, _bh);                                      \
        __m256i _w1 = _mm256_unpackhi_epi8(_bl, _bh);                                      \
        (c0) = _mm256_fmadd_ps(                                                            \
            _mm256_castsi256_ps(_mm256_unpacklo_epi16(ZR, _w0)), (xq)[0], (c0));            \
        (c1) = _mm256_fmadd_ps(                                                            \
            _mm256_castsi256_ps(_mm256_unpackhi_epi16(ZR, _w0)), (xq)[1], (c1));            \
        (c2) = _mm256_fmadd_ps(                                                            \
            _mm256_castsi256_ps(_mm256_unpacklo_epi16(ZR, _w1)), (xq)[2], (c2));            \
        (c3) = _mm256_fmadd_ps(                                                            \
            _mm256_castsi256_ps(_mm256_unpackhi_epi16(ZR, _w1)), (xq)[3], (c3));            \
    } while (0)

static MXFP4_TARGET_AVX2 inline float mxfp4_reduce4_avx2(
        __m256 a0, __m256 a1, __m256 a2, __m256 a3) {
    __m256 sum = _mm256_add_ps(
        _mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 folded = _mm_add_ps(
        _mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    return vaddvq_f32(folded);
}

// xp must already be in mxfp4_permute_x_avx2 order.  This function is internal
// to the translation unit so callers cannot accidentally pass an ordinary x.
static MXFP4_TARGET_AVX2 void mxfp4_gemv_rows2_avx2_prepared(
        const uint8_t *restrict p, const uint8_t *restrict s,
        const float *restrict xp, float *restrict y,
        int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
#if !MXFP4_USE_SCALE_LUT
    const __m256i BZ = _mm256_loadu_si256((const __m256i *)base16);
    const __m256i ZZ = _mm256_loadu_si256((const __m256i *)mask16);
#endif
    const __m256i ZR = _mm256_setzero_si256();
    const __m128i M0F = _mm_set1_epi8(0x0F);
#if !MXFP4_USE_SCALE_LUT
    const __m256i UZP = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)mxfp4_avx2_uzp_bytes));
#endif

    int r = row0;
    for (; r + 1 < row1; r += 2) {
        const uint8_t *pp_a = p + (size_t)r * cp;
        const uint8_t *sp_a = s + (size_t)r * groups;
        const uint8_t *pp_b = p + (size_t)(r + 1) * cp;
        const uint8_t *sp_b = s + (size_t)(r + 1) * groups;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        for (int g = 0; g < groups; g++) {
            const float *xg = xp + (size_t)g * 32;
            const __m256 xq[4] = {
                _mm256_loadu_ps(xg), _mm256_loadu_ps(xg + 8),
                _mm256_loadu_ps(xg + 16), _mm256_loadu_ps(xg + 24)
            };
            __m256i tla, tha, tlb, thb;
            MXFP4_AVX2_TABLE(sp_a[g], tla, tha);
            MXFP4_AVX2_GROUP(pp_a, tla, tha, xq, a0, a1, a2, a3);
            MXFP4_AVX2_TABLE(sp_b[g], tlb, thb);
            MXFP4_AVX2_GROUP(pp_b, tlb, thb, xq, b0, b1, b2, b3);
        }
        y[r] = mxfp4_reduce4_avx2(a0, a1, a2, a3);
        y[r + 1] = mxfp4_reduce4_avx2(b0, b1, b2, b3);
    }

    // Keep odd-row jobs on AVX2 as well.  This matters for arbitrary synthetic
    // shapes and avoids needing an unpermuted x pointer in persistent workers.
    if (r < row1) {
        const uint8_t *pp = p + (size_t)r * cp;
        const uint8_t *sp = s + (size_t)r * groups;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (int g = 0; g < groups; g++) {
            const float *xg = xp + (size_t)g * 32;
            const __m256 xq[4] = {
                _mm256_loadu_ps(xg), _mm256_loadu_ps(xg + 8),
                _mm256_loadu_ps(xg + 16), _mm256_loadu_ps(xg + 24)
            };
            __m256i tl, th;
            MXFP4_AVX2_TABLE(sp[g], tl, th);
            MXFP4_AVX2_GROUP(pp, tl, th, xq, a0, a1, a2, a3);
        }
        y[r] = mxfp4_reduce4_avx2(a0, a1, a2, a3);
    }
}

#undef MXFP4_AVX2_GROUP
#undef MXFP4_AVX2_TABLE
#endif

// full-matrix NEON dequant (prior neon_a kernel) for the Accelerate baseline
static const uint8_t thi_tab[16] = {
    0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
    0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0
};
static const uint8_t tlo_tab[16] = {
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0
};
static void dequant_neon_a(const uint8_t *restrict p, const uint8_t *restrict s,
                           float *restrict out, int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
    const uint8x16_t THI = vld1q_u8(thi_tab), TLO = vld1q_u8(tlo_tab);
    const uint8x16_t M0F = vdupq_n_u8(0x0F);
    for (int r = row0; r < row1; r++) {
        const uint8_t *pp = p + (size_t)r * cp;
        const uint8_t *sp = s + (size_t)r * groups;
        float *o = out + (size_t)r * cols;
        for (int g = 0; g < groups; g++) {
            const float32x4_t VS = vdupq_n_f32(e8m0_to_f32(sp[g]));
            uint8x16_t P  = vld1q_u8(pp + g * 16);
            uint8x16_t lo = vandq_u8(P, M0F), hi = vshrq_n_u8(P, 4);
            uint8x16_t nib[2] = { vzip1q_u8(lo, hi), vzip2q_u8(lo, hi) };
            float *ob = o + g * 32;
            for (int h = 0; h < 2; h++) {
                uint8x16_t bl = vqtbl1q_u8(TLO, nib[h]), bh = vqtbl1q_u8(THI, nib[h]);
                uint16x8_t h0 = vreinterpretq_u16_u8(vzip1q_u8(bl, bh));
                uint16x8_t h1 = vreinterpretq_u16_u8(vzip2q_u8(bl, bh));
                float *od = ob + h * 16;
                vst1q_f32(od + 0,  vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)), VS));
                vst1q_f32(od + 4,  vmulq_f32(vreinterpretq_f32_u32(vshll_high_n_u16(h0, 16)), VS));
                vst1q_f32(od + 8,  vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)), VS));
                vst1q_f32(od + 12, vmulq_f32(vreinterpretq_f32_u32(vshll_high_n_u16(h1, 16)), VS));
            }
        }
    }
}

// ---------------- threading ----------------
// The restrict qualifiers match both kernels this points at.  C ignores
// top-level parameter qualifiers when comparing function types, but MSVC
// compares them, so spelling them out keeps the assignment warning-free.
typedef void (*gemv_fn)(const uint8_t *restrict, const uint8_t *restrict,
                        const float *restrict, float *restrict, int, int, int);

typedef struct { // generic row-partition job for one gemv
    gemv_fn fn;
    const uint8_t *p, *s; const float *x; float *y;
    int row0, row1, cols, iters;
} gjob_t;

static void *gworker(void *arg) {
    mxfp4_qos_self();
    gjob_t *j = (gjob_t *)arg;
    for (int it = 0; it < j->iters; it++)
        j->fn(j->p, j->s, j->x, j->y, j->row0, j->row1, j->cols);
    return NULL;
}

#define MXFP4_MAX_THREADS 16

static inline int mxfp4_clamp_threads(int nthreads) {
    if (nthreads < 1) return 1;
    if (nthreads > MXFP4_MAX_THREADS) return MXFP4_MAX_THREADS;
    return nthreads;
}

// one gemv, nthreads, iters back-to-back; returns seconds total
static double run_gemv_mt_k(gemv_fn fn, const uint8_t *p, const uint8_t *s, const float *x, float *y,
                            int rows, int cols, int nthreads, int iters) {
    pthread_t th[MXFP4_MAX_THREADS];
    gjob_t jobs[MXFP4_MAX_THREADS];
    nthreads = mxfp4_clamp_threads(nthreads);
    int per = (rows + nthreads - 1) / nthreads;
    per = (per + 1) & ~1;  // even split so rows2 stays on 2-row boundaries
    double t0 = now_s();
    int made = 0;
    for (int t = 0; t < nthreads; t++) {
        jobs[t] = (gjob_t){ fn, p, s, x, y, t * per,
                            (t + 1) * per > rows ? rows : (t + 1) * per, cols, iters };
        if (mxfp4_pthread_create(
                &th[t], NULL, gworker, &jobs[t]) != 0) break;
        made++;
    }
    for (int t = 0; t < made; t++) pthread_join(th[t], NULL);
    // A partial row partition is not a valid result.  If thread creation failed,
    // recompute the complete matrix inline after all successful workers have exited.
    if (made != nthreads) {
        for (int it = 0; it < iters; it++)
            fn(p, s, x, y, 0, rows, cols);
    }
    return now_s() - t0;
}
static double run_gemv_mt(const uint8_t *p, const uint8_t *s, const float *x, float *y,
                          int rows, int cols, int nthreads, int iters) {
    return run_gemv_mt_k(mxfp4_gemv_rows2, p, s, x, y, rows, cols, nthreads, iters);
}

// ---- expert triple: y1=W1@x, y3=W3@x, [barrier], y2=W2@h  (h supplied; activation omitted)
typedef struct {
    const uint8_t *p1, *s1, *p3, *s3, *p2, *s2;
    const float *x, *h; float *y1, *y3, *y2;
    int tid, nthreads, iters;
    _Atomic int *bar, *ctrA, *ctrB, *start;
} tjob_t;

static void barrier_wait(_Atomic int *ctr, int nthreads, int phase) {
    int target = nthreads * (phase + 1);
    atomic_fetch_add_explicit(ctr, 1, memory_order_acq_rel);
    int spins = 0;
    while (atomic_load_explicit(ctr, memory_order_acquire) < target) {
        if (++spins > 2000) { sched_yield(); spins = 0; }  // machine is oversubscribed
        else mxfp4_cpu_relax();
    }
}

static void *tworker(void *arg) {
    mxfp4_qos_self();
    tjob_t *j = (tjob_t *)arg;
    while (!atomic_load_explicit(j->start, memory_order_acquire))
        mxfp4_cpu_relax();
    const int C13 = 3584, C2 = 3072;
    const int CH = 32;                       // rows per work unit (even: rows2-safe)
    const int NA = 2 * (3072 / CH);          // phase A: w1 + w3 chunks
    const int NB = 3584 / CH;                // phase B: w2 chunks
    for (int it = 0; it < j->iters; it++) {
        int u;
        while ((u = atomic_fetch_add_explicit(j->ctrA, 1, memory_order_relaxed)) < NA) {
            int m = u >= NA / 2;             // 0 -> w1, 1 -> w3
            int r0 = (m ? u - NA / 2 : u) * CH;
            if (m) mxfp4_gemv_rows2(j->p3, j->s3, j->x, j->y3, r0, r0 + CH, C13);
            else   mxfp4_gemv_rows2(j->p1, j->s1, j->x, j->y1, r0, r0 + CH, C13);
        }
        barrier_wait(j->bar, j->nthreads, 3 * it);
        while ((u = atomic_fetch_add_explicit(j->ctrB, 1, memory_order_relaxed)) < NB)
            mxfp4_gemv_rows2(j->p2, j->s2, j->h, j->y2, u * CH, u * CH + CH, C2);
        barrier_wait(j->bar, j->nthreads, 3 * it + 1);
        if (j->tid == 0) {
            atomic_store_explicit(j->ctrA, 0, memory_order_relaxed);
            atomic_store_explicit(j->ctrB, 0, memory_order_relaxed);
        }
        barrier_wait(j->bar, j->nthreads, 3 * it + 2);
    }
    return NULL;
}

// exported: full expert triple; returns seconds for iters repetitions
K3_EXPORT double mxfp4_expert_triple(
        const uint8_t *p1, const uint8_t *s1,
        const uint8_t *p3, const uint8_t *s3,
        const uint8_t *p2, const uint8_t *s2,
        const float *x, const float *h,
        float *y1, float *y3, float *y2,
        int nthreads, int iters) {
    _Atomic int bar = 0, ctrA = 0, ctrB = 0, start = 0;
    pthread_t th[MXFP4_MAX_THREADS];
    tjob_t jobs[MXFP4_MAX_THREADS];
    nthreads = mxfp4_clamp_threads(nthreads);
    double t0 = now_s();
    if (nthreads <= 1) {
        for (int it = 0; it < iters; it++) {
            mxfp4_gemv_rows2(p1, s1, x, y1, 0, 3072, 3584);
            mxfp4_gemv_rows2(p3, s3, x, y3, 0, 3072, 3584);
            mxfp4_gemv_rows2(p2, s2, h, y2, 0, 3584, 3072);
        }
        return now_s() - t0;
    }
    int made = 0;
    for (int t = 0; t < nthreads; t++) {
        jobs[t] = (tjob_t){ p1, s1, p3, s3, p2, s2, x, h, y1, y3, y2,
                            t, nthreads, iters, &bar, &ctrA, &ctrB, &start };
        if (mxfp4_pthread_create(
                &th[t], NULL, tworker, &jobs[t]) != 0) break;
        made++;
    }
    // Workers wait on start, so a partial pthread_create can safely reduce the
    // barrier width before any worker enters a phase.
    for (int t = 0; t < made; t++) {
        jobs[t].nthreads = made;
        jobs[t].tid = t;
    }
    atomic_store_explicit(&start, 1, memory_order_release);
    if (made == 0) {
        for (int it = 0; it < iters; it++) {
            mxfp4_gemv_rows2(p1, s1, x, y1, 0, 3072, 3584);
            mxfp4_gemv_rows2(p3, s3, x, y3, 0, 3072, 3584);
            mxfp4_gemv_rows2(p2, s2, h, y2, 0, 3584, 3072);
        }
        return now_s() - t0;
    }
    for (int t = 0; t < made; t++) pthread_join(th[t], NULL);
    return now_s() - t0;
}

// Explicit exact compatibility exports make architecture tests able to compare
// the selected path with the established 128-bit implementation.
K3_EXPORT void mxfp4_gemv_compat(
        const uint8_t *p, const uint8_t *s, const float *x, float *y,
        int rows, int cols) {
    mxfp4_gemv_rows(p, s, x, y, 0, rows, cols);
}
K3_EXPORT void mxfp4_gemv_mt_compat(
        const uint8_t *p, const uint8_t *s, const float *x, float *y,
        int rows, int cols, int nthreads) {
    run_gemv_mt(p, s, x, y, rows, cols, nthreads, 1);
}

// The disable-only override makes dispatch tests deterministic and provides a
// safe diagnostic fallback.  It can never force an unsupported instruction.
#if MXFP4_CAN_BUILD_AVX2
static pthread_once_t mxfp4_dispatch_once = PTHREAD_ONCE_INIT;
static int mxfp4_dispatch_avx2 = 0;

static void mxfp4_detect_dispatch(void) {
    const char *disable = getenv("K3_MXFP4_DISABLE_AVX2");
    if (disable && disable[0] && strcmp(disable, "0") != 0) return;
#if defined(_MSC_VER)
    mxfp4_dispatch_avx2 = k3_cpu_supports_avx2();
#else
    __builtin_cpu_init();
    mxfp4_dispatch_avx2 = !!__builtin_cpu_supports("avx2");
#endif
}

K3_EXPORT int mxfp4_have_avx2(void) {
    pthread_once(&mxfp4_dispatch_once, mxfp4_detect_dispatch);
    return mxfp4_dispatch_avx2;
}

K3_EXPORT MXFP4_TARGET_AVX2 void mxfp4_gemv_avx2(
        const uint8_t *p, const uint8_t *s, const float *x, float *y,
        int rows, int cols) {
    float *xp = mxfp4_prepare_x_avx2(x, cols);
    if (!xp) {
        mxfp4_gemv_compat(p, s, x, y, rows, cols);
        return;
    }
    mxfp4_gemv_rows2_avx2_prepared(p, s, xp, y, 0, rows, cols);
    k3_aligned_free(xp);
}

K3_EXPORT MXFP4_TARGET_AVX2 void mxfp4_gemv_mt_avx2(
        const uint8_t *p, const uint8_t *s, const float *x, float *y,
        int rows, int cols, int nthreads) {
    float *xp = mxfp4_prepare_x_avx2(x, cols);
    if (!xp) {
        mxfp4_gemv_mt_compat(p, s, x, y, rows, cols, nthreads);
        return;
    }
    run_gemv_mt_k(
        mxfp4_gemv_rows2_avx2_prepared,
        p, s, xp, y, rows, cols, nthreads, 1);
    k3_aligned_free(xp);
}
#else
K3_EXPORT int mxfp4_have_avx2(void) { return 0; }
#endif

// Stable ctypes entry points select the best runtime path. ARM and x86 hosts
// without AVX2 retain the exact established implementation.
K3_EXPORT void mxfp4_gemv(
        const uint8_t *p, const uint8_t *s, const float *x, float *y,
        int rows, int cols) {
#if MXFP4_CAN_BUILD_AVX2
    if (mxfp4_have_avx2()) {
        mxfp4_gemv_avx2(p, s, x, y, rows, cols);
        return;
    }
#endif
    mxfp4_gemv_compat(p, s, x, y, rows, cols);
}

K3_EXPORT void mxfp4_gemv_mt(
        const uint8_t *p, const uint8_t *s, const float *x, float *y,
        int rows, int cols, int nthreads) {
#if MXFP4_CAN_BUILD_AVX2
    if (mxfp4_have_avx2()) {
        mxfp4_gemv_mt_avx2(p, s, x, y, rows, cols, nthreads);
        return;
    }
#endif
    mxfp4_gemv_mt_compat(p, s, x, y, rows, cols, nthreads);
}

#ifndef NO_MAIN
// ---------------- harness ----------------
static void *load_file(const char *name, size_t sz) {
    void *buf = k3_aligned_alloc(128, (sz + 127) & ~(size_t)127);
    FILE *f = fopen(name, "rb");
    if (!f || fread(buf, 1, sz, f) != sz) { fprintf(stderr, "load %s failed\n", name); exit(1); }
    fclose(f);
    return buf;
}

static void errstats(const float *y, const double *ref64, int n, const char *tag) {
    double maxrel = 0, maxabs = 0;
    for (int i = 0; i < n; i++) {
        double a = fabs((double)y[i] - ref64[i]);
        double r = a / fmax(fabs(ref64[i]), 1e-6);
        if (a > maxabs) maxabs = a;
        if (r > maxrel) maxrel = r;
    }
    printf("  %-28s max_abs=%.3e max_rel=%.3e\n", tag, maxabs, maxrel);
}

typedef struct { const uint8_t *p, *s; const float *x; int rows, cols; const char *name; } mat_t;

int main(int argc, char **argv) {
    const int R13 = 3072, C13 = 3584, R2 = 3584, C2 = 3072;
    uint8_t *w1p = load_file("w1_p.bin", (size_t)R13 * C13 / 2);
    uint8_t *w1s = load_file("w1_s.bin", (size_t)R13 * C13 / 32);
    uint8_t *w3p = load_file("w3_p.bin", (size_t)R13 * C13 / 2);
    uint8_t *w3s = load_file("w3_s.bin", (size_t)R13 * C13 / 32);
    uint8_t *w2p = load_file("w2_p.bin", (size_t)R2 * C2 / 2);
    uint8_t *w2s = load_file("w2_s.bin", (size_t)R2 * C2 / 32);
    float *x = load_file("x.bin", C13 * sizeof(float));
    float *h = load_file("h.bin", C2 * sizeof(float));
    double *r64_1 = load_file("w1_yref64.bin", R13 * sizeof(double));
    double *r64_3 = load_file("w3_yref64.bin", R13 * sizeof(double));
    double *r64_2 = load_file("w2_yref64.bin", R2 * sizeof(double));
    float *r32_1 = load_file("w1_yref32.bin", R13 * sizeof(float));

    float *y_ref = k3_aligned_alloc(128, R2 * sizeof(float));
    float *y_fus = k3_aligned_alloc(128, R2 * sizeof(float));
    float *y1 = k3_aligned_alloc(128, R13 * sizeof(float));
    float *y3 = k3_aligned_alloc(128, R13 * sizeof(float));
    float *y2 = k3_aligned_alloc(128, R2 * sizeof(float));

    // ---- correctness ----
    printf("== correctness (real L1-E0 data) ==\n");
    mat_t mats[3] = { { w1p, w1s, x, R13, C13, "w1" }, { w3p, w3s, x, R13, C13, "w3" },
                      { w2p, w2s, h, R2, C2, "w2" } };
    const double *r64s[3] = { r64_1, r64_3, r64_2 };
    for (int m = 0; m < 3; m++) {
        ref_gemv(mats[m].p, mats[m].s, mats[m].x, y_ref, 0, mats[m].rows, mats[m].cols);
        mxfp4_gemv_rows(mats[m].p, mats[m].s, mats[m].x, y_fus, 0, mats[m].rows, mats[m].cols);
        int bitexact = !memcmp(y_ref, y_fus, mats[m].rows * sizeof(float));
        double maxrel = 0;
        for (int i = 0; i < mats[m].rows; i++) {
            double r = fabs((double)y_fus[i] - y_ref[i]) / fmax(fabs((double)y_ref[i]), 1e-6);
            if (r > maxrel) maxrel = r;
        }
        printf("%s fused vs C-ref: %s (max_rel=%.1e)\n", mats[m].name,
               bitexact ? "BIT-EXACT" : "differs", maxrel);
        mxfp4_gemv_rows2(mats[m].p, mats[m].s, mats[m].x, y_fus, 0, mats[m].rows, mats[m].cols);
        printf("%s fused2r vs C-ref: %s\n", mats[m].name,
               memcmp(y_ref, y_fus, mats[m].rows * sizeof(float)) ? "differs" : "BIT-EXACT");
        char tag[64];
        snprintf(tag, 64, "%s fused vs numpy-fp64", mats[m].name);
        errstats(y_fus, r64s[m], mats[m].rows, tag);
    }
    // numpy fp32 vs fp64 for context
    errstats(r32_1, r64_1, R13, "w1 numpy-fp32 vs numpy-fp64");
#if MXFP4_HAVE_CBLAS
    // Accelerate/OpenBLAS baseline correctness
    float *Wbuf = k3_aligned_alloc(128, (size_t)R13 * C13 * sizeof(float));
    dequant_neon_a(w1p, w1s, Wbuf, 0, R13, C13);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, R13, C13, 1.0f, Wbuf, C13, x, 1, 0.0f, y_ref, 1);
    errstats(y_ref, r64_1, R13, "w1 dequant+sgemv vs fp64");
#endif

    // ---- benchmarks ----
    double tgt = argc > 1 ? atof(argv[1]) : 0.12;
    int reps = argc > 2 ? atoi(argv[2]) : 8;
    const double PKB13 = (double)R13 * C13 / 2, PKB2 = (double)R2 * C2 / 2;
    const double PK_TRIPLE = 2 * PKB13 + PKB2;              // packed bytes/expert
    const double TOT_TRIPLE = PK_TRIPLE * (1 + 1.0 / 16.0); // + scales

    printf("\n== fused kernel bench (best of %d, ~%.2fs bursts) ==\n", reps, tgt);
    printf("%-24s %2s %10s %10s %12s\n", "op", "T", "us/op", "GB/s-pkd", "GB/s-total");
    for (int nt = 1; nt <= 4; nt++) {
        int iters = (int)(tgt / (PKB13 / (nt * 4.0e9))) + 1;
        double b1 = 1e9, b2 = 1e9;
        for (int rp = 0; rp < reps; rp++) {
            double dt = run_gemv_mt_k(mxfp4_gemv_rows, w1p, w1s, x, y1, R13, C13, nt, iters) / iters;
            if (dt < b1) b1 = dt;
            dt = run_gemv_mt_k(mxfp4_gemv_rows2, w1p, w1s, x, y1, R13, C13, nt, iters) / iters;
            if (dt < b2) b2 = dt;
        }
        printf("%-24s %2d %10.1f %10.2f %12.2f\n", "w1 gemv fused1r", nt,
               b1 * 1e6, PKB13 / b1 / 1e9, PKB13 * (17.0 / 16.0) / b1 / 1e9);
        printf("%-24s %2d %10.1f %10.2f %12.2f\n", "w1 gemv fused2r", nt,
               b2 * 1e6, PKB13 / b2 / 1e9, PKB13 * (17.0 / 16.0) / b2 / 1e9);
    }
    for (int nt = 1; nt <= 4; nt++) {
        int iters = (int)(tgt / (PK_TRIPLE / (nt * 4.0e9))) + 1;
        double best = 1e9;
        for (int rp = 0; rp < reps; rp++) {
            double dt = mxfp4_expert_triple(w1p, w1s, w3p, w3s, w2p, w2s,
                                            x, h, y1, y3, y2, nt, iters) / iters;
            if (dt < best) best = dt;
        }
        printf("%-24s %2d %10.1f %10.2f %12.2f\n", "expert triple (2r)", nt,
               best * 1e6, PK_TRIPLE / best / 1e9, TOT_TRIPLE / best / 1e9);
    }

#if MXFP4_HAVE_CBLAS
    // BLAS baseline: full dequant (1T SIMD) + sgemv, per matrix, x3 for triple
    printf("\n== baseline: full dequant + cblas_sgemv (1T dequant) ==\n");
    {
        double bd = 1e9, bs = 1e9;
        for (int rp = 0; rp < reps; rp++) {
            double t0 = now_s();
            dequant_neon_a(w1p, w1s, Wbuf, 0, R13, C13);
            double t1 = now_s();
            cblas_sgemv(CblasRowMajor, CblasNoTrans, R13, C13, 1.0f, Wbuf, C13, x, 1, 0.0f, y1, 1);
            double t2 = now_s();
            if (t1 - t0 < bd) bd = t1 - t0;
            if (t2 - t1 < bs) bs = t2 - t1;
        }
        printf("w1: dequant %.1f us + sgemv %.1f us = %.1f us  (fp32 buffer 44MB)\n",
               bd * 1e6, bs * 1e6, (bd + bs) * 1e6);
        printf("triple extrapolated: %.1f us/expert\n", 3 * (bd + bs) * 1e6);
    }

    k3_aligned_free(Wbuf);
#endif
    return 0;
}
#endif
