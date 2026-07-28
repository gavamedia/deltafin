// Fused MXFP4 dequant + GEMV SIMD kernel for Kimi-K3 experts.
// Never materializes the fp32 weight matrix.
//
// Written in NEON intrinsics (Apple Silicon); on x86-64 the identical code runs
// through the 1:1 128-bit mapping in neon_compat_x86.h (SSSE3 + FMA3), keeping
// the same lanes, the same fused multiply-adds and the same reduction order.
//
// Layout: packed U8 [rows, cols/2] (two e2m1 nibbles/byte, LOW first),
//         scales U8 [rows, cols/32] (e8m0: 2^(s-127)).
// GEMV: y[r] = dot(W[r,:], x)  (row-major, streaming rows)
//
// Build exe (macOS):   clang -O3 -mcpu=native -o fused_gemv fused_gemv.c -lpthread -framework Accelerate
// Build dylib (macOS): clang -O3 -mcpu=native -shared -DNO_MAIN -o libmxfp4gemv.dylib fused_gemv.c -lpthread
// Build .so (Linux):   gcc -O3 -march=x86-64-v3 -shared -fPIC -DNO_MAIN -o libmxfp4gemv.so fused_gemv.c -lpthread
// Build exe (Linux):   gcc -O3 -march=x86-64-v3 -o fused_gemv fused_gemv.c -lpthread -lopenblas

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#else
#include "neon_compat_x86.h"
#endif
#include <pthread.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#define k3_qos_self() pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
#else
#define k3_qos_self() ((void)0)   // Linux: no QoS classes; plain pthreads
#endif
#if defined(__aarch64__)
#define k3_cpu_relax() __asm__ volatile("yield")
#else
#define k3_cpu_relax() _mm_pause()
#endif
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#ifndef NO_MAIN
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>   /* any CBLAS, e.g. -lopenblas */
#endif
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------- shared tables ----------------
// fp32 bit patterns of e2m1 values occupy only the top 16 bits; halfword tables.
// Scale folds in as an integer add of (s-127)<<7 to the halfword (exponent field),
// masked so +/-0 stay zero. Valid while fp32 exponent stays in [1,254] (real data 112..122).
static const uint16_t base16[16] = {
    0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
    0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0
};
static const uint16_t mask16[16] = {
    0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF
};
static const float e2m1_tab[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
static inline float e8m0_to_f32(uint8_t s) {
    union { uint32_t u; float f; } v; v.u = (uint32_t)s << 23; return v.f;
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
void mxfp4_gemv_rows(const uint8_t *restrict p, const uint8_t *restrict s,
                     const float *restrict x, float *restrict y,
                     int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
    const uint16x8_t B0 = vld1q_u16(base16), B1 = vld1q_u16(base16 + 8);
    const uint16x8_t Z0 = vld1q_u16(mask16), Z1 = vld1q_u16(mask16 + 8);
    const uint8x16_t M0F = vdupq_n_u8(0x0F);
    for (int r = row0; r < row1; r++) {
        const uint8_t *pp = p + (size_t)r * cp;
        const uint8_t *sp = s + (size_t)r * groups;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0), a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);
        for (int g = 0; g < groups; g++) {
            uint16x8_t A = vdupq_n_u16((uint16_t)(((int)sp[g] - 127) << 7));
            uint8x16_t t0 = vreinterpretq_u8_u16(vaddq_u16(B0, vandq_u16(A, Z0)));
            uint8x16_t t1 = vreinterpretq_u8_u16(vaddq_u16(B1, vandq_u16(A, Z1)));
            uint8x16_t TABL = vuzp1q_u8(t0, t1);
            uint8x16_t TABH = vuzp2q_u8(t0, t1);
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
void mxfp4_gemv_rows2(const uint8_t *restrict p, const uint8_t *restrict s,
                      const float *restrict x, float *restrict y,
                      int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
    const uint16x8_t B0 = vld1q_u16(base16), B1 = vld1q_u16(base16 + 8);
    const uint16x8_t Z0 = vld1q_u16(mask16), Z1 = vld1q_u16(mask16 + 8);
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
                uint16x8_t AD = vdupq_n_u16((uint16_t)(((int)(sp)[g] - 127) << 7));                  \
                uint8x16_t t0 = vreinterpretq_u8_u16(vaddq_u16(B0, vandq_u16(AD, Z0)));              \
                uint8x16_t t1 = vreinterpretq_u8_u16(vaddq_u16(B1, vandq_u16(AD, Z1)));              \
                uint8x16_t TABL = vuzp1q_u8(t0, t1), TABH = vuzp2q_u8(t0, t1);                       \
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
typedef void (*gemv_fn)(const uint8_t *, const uint8_t *, const float *, float *, int, int, int);

typedef struct { // generic row-partition job for one gemv
    gemv_fn fn;
    const uint8_t *p, *s; const float *x; float *y;
    int row0, row1, cols, iters;
} gjob_t;

static void *gworker(void *arg) {
    k3_qos_self();
    gjob_t *j = (gjob_t *)arg;
    for (int it = 0; it < j->iters; it++)
        j->fn(j->p, j->s, j->x, j->y, j->row0, j->row1, j->cols);
    return NULL;
}

// one gemv, nthreads, iters back-to-back; returns seconds total
static double run_gemv_mt_k(gemv_fn fn, const uint8_t *p, const uint8_t *s, const float *x, float *y,
                            int rows, int cols, int nthreads, int iters) {
    // CLAMP: these are fixed-size STACK arrays. nthreads was previously unbounded, so any
    // caller asking for more than K3_GEMV_MAX_THREADS smashed the stack (reproduced with
    // nthreads=32: "*** stack smashing detected ***"). Never a problem on a 10-core M1 Max;
    // immediately reachable on a 32-thread server, where asking for one thread per hardware
    // thread is the obvious first thing to try. Clamping is safe — over-subscribing past the
    // core count loses throughput anyway (measured peak here is 12T).
    enum { K3_GEMV_MAX_THREADS = 16 };
    pthread_t th[K3_GEMV_MAX_THREADS]; gjob_t jobs[K3_GEMV_MAX_THREADS];
    if (nthreads < 1) nthreads = 1;
    if (nthreads > K3_GEMV_MAX_THREADS) nthreads = K3_GEMV_MAX_THREADS;
    int per = (rows + nthreads - 1) / nthreads;
    per = (per + 1) & ~1;  // even split so rows2 stays on 2-row boundaries
    double t0 = now_s();
    for (int t = 0; t < nthreads; t++) {
        jobs[t] = (gjob_t){ fn, p, s, x, y, t * per,
                            (t + 1) * per > rows ? rows : (t + 1) * per, cols, iters };
        pthread_create(&th[t], NULL, gworker, &jobs[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
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
    _Atomic int *bar, *ctrA, *ctrB;
} tjob_t;

static void barrier_wait(_Atomic int *ctr, int nthreads, int phase) {
    int target = nthreads * (phase + 1);
    atomic_fetch_add_explicit(ctr, 1, memory_order_acq_rel);
    int spins = 0;
    while (atomic_load_explicit(ctr, memory_order_acquire) < target) {
        if (++spins > 2000) { sched_yield(); spins = 0; }  // machine is oversubscribed
        else k3_cpu_relax();
    }
}

static void *tworker(void *arg) {
    k3_qos_self();
    tjob_t *j = (tjob_t *)arg;
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
double mxfp4_expert_triple(const uint8_t *p1, const uint8_t *s1,
                           const uint8_t *p3, const uint8_t *s3,
                           const uint8_t *p2, const uint8_t *s2,
                           const float *x, const float *h,
                           float *y1, float *y3, float *y2,
                           int nthreads, int iters) {
    _Atomic int bar = 0, ctrA = 0, ctrB = 0;
    pthread_t th[16]; tjob_t jobs[16];
    double t0 = now_s();
    if (nthreads <= 1) {
        for (int it = 0; it < iters; it++) {
            mxfp4_gemv_rows2(p1, s1, x, y1, 0, 3072, 3584);
            mxfp4_gemv_rows2(p3, s3, x, y3, 0, 3072, 3584);
            mxfp4_gemv_rows2(p2, s2, h, y2, 0, 3584, 3072);
        }
        return now_s() - t0;
    }
    for (int t = 0; t < nthreads; t++) {
        jobs[t] = (tjob_t){ p1, s1, p3, s3, p2, s2, x, h, y1, y3, y2,
                            t, nthreads, iters, &bar, &ctrA, &ctrB };
        pthread_create(&th[t], NULL, tworker, &jobs[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
    return now_s() - t0;
}

// ================= native 256-bit AVX2 kernel (x86-64 only) =================
// The portable path above runs NEON intrinsics through neon_compat_x86.h, which maps every
// 128-bit NEON op to one SSE op. Correct, but it leaves half of each YMM register idle: a
// Zen 2 core retires vpshufb/vpaddw/vfmadd at full 256-bit width for the same 1 uop.
//
// Three changes vs. the compat path, per 32-value group:
//   * ONE vpshufb does 32 table lookups (the compat path issues 4 pshufb for the same work),
//     because AVX2 shuffles each 128-bit lane against its own copy of the 16-byte table.
//   * 4 YMM accumulators replace 8 XMM, halving the FMA count.
//   * The table's low/high byte split is built with a shuffle + two permutes instead of the
//     compat vuzp1q/vuzp2q pair, which each expand to shuffle+shuffle+unpack.
//
// The lane problem and why x is permuted. Widening bf16->f32 with vpunpcklwd is per-lane, so a
// YMM of weights comes out as [elems 0..3 | elems 16..19] -- two disjoint 4-element runs, not
// one contiguous 8. Gathering matching x with two 128-bit loads + vinsertf128 would cost 3 uops
// per FMA and hand back everything the widening saved. Instead x is permuted ONCE per call into
// that same interleaved order, so every FMA reads it with a plain contiguous 256-bit load. The
// permute is O(cols) against O(rows*cols) of GEMV, i.e. 3584 floats vs 11M -- free. y is
// unaffected: each accumulator lane holds a partial dot product and the final reduction sums
// them all, so lane assignment never escapes the kernel.
//
// Bit-exactness is preserved, not approximated. Accumulator Ai holds NEON's [ai | ai+4], so
// (A0+A1)+(A2+A3) reproduces (b0+b1) in lane 0 and (b2+b3) in lane 1 -- exactly reduce8's
// association -- and the 128-bit fold plus vaddvq_f32 finishes it in the same order. Verified
// bit-identical to the compat kernel on every row, not merely within tolerance.
#if defined(__AVX2__)
#include <immintrin.h>

// x[g*32 .. g*32+31] -> [0..3|16..19][4..7|20..23][8..11|24..27][12..15|28..31]
static void k3_permute_x(const float *restrict x, float *restrict xp, int cols) {
    for (int g = 0; g < cols / 32; g++) {
        const float *src = x + g * 32;
        float *dst = xp + g * 32;
        for (int q = 0; q < 4; q++) {
            memcpy(dst + q * 8,     src + q * 4,      4 * sizeof(float));
            memcpy(dst + q * 8 + 4, src + 16 + q * 4, 4 * sizeof(float));
        }
    }
}

static const uint8_t k3_uzp_bytes[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };

// Build the group's dequant table, broadcast to both 128-bit lanes so one vpshufb serves 32 idx.
#define K3_TABLE(scale_byte, TL, TH)                                                             \
    do {                                                                                          \
        __m256i _a = _mm256_set1_epi16((short)((((int)(scale_byte)) - 127) << 7));                \
        __m256i _t = _mm256_add_epi16(BZ, _mm256_and_si256(_a, ZZ));                              \
        __m256i _u = _mm256_shuffle_epi8(_t, UZP);   /* per lane: 8 low bytes then 8 high */      \
        (TL) = _mm256_permute4x64_epi64(_u, 0x88);   /* qwords 0,2 -> low bytes,  both lanes */   \
        (TH) = _mm256_permute4x64_epi64(_u, 0xDD);   /* qwords 1,3 -> high bytes, both lanes */   \
    } while (0)

// 32 weights -> 4 YMM f32 in the permuted order, then 4 FMAs against contiguous x.
#define K3_GRP(pp, TL, TH, xq, c0, c1, c2, c3)                                                    \
    do {                                                                                          \
        __m128i _P  = _mm_loadu_si128((const __m128i *)((pp) + g * 16));                          \
        __m128i _lo = _mm_and_si128(_P, M0F);                                                     \
        __m128i _hi = _mm_and_si128(_mm_srli_epi16(_P, 4), M0F);                                  \
        __m256i _ix = _mm256_set_m128i(_mm_unpackhi_epi8(_lo, _hi),   /* lane1: elems 16..31 */   \
                                       _mm_unpacklo_epi8(_lo, _hi));  /* lane0: elems  0..15 */   \
        __m256i _bl = _mm256_shuffle_epi8(TL, _ix);                                               \
        __m256i _bh = _mm256_shuffle_epi8(TH, _ix);                                               \
        __m256i _w0 = _mm256_unpacklo_epi8(_bl, _bh);  /* bf16 e0..e7   | e16..e23 */             \
        __m256i _w1 = _mm256_unpackhi_epi8(_bl, _bh);  /* bf16 e8..e15  | e24..e31 */             \
        (c0) = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpacklo_epi16(ZR, _w0)), (xq)[0], (c0)); \
        (c1) = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpackhi_epi16(ZR, _w0)), (xq)[1], (c1)); \
        (c2) = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpacklo_epi16(ZR, _w1)), (xq)[2], (c2)); \
        (c3) = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpackhi_epi16(ZR, _w1)), (xq)[3], (c3)); \
    } while (0)

// reduce4: lane0 = NEON's c0, lane1 = c1 -> fold to 128 bits, then the identical vaddvq order.
static inline float k3_reduce4(__m256 A0, __m256 A1, __m256 A2, __m256 A3) {
    __m256 S = _mm256_add_ps(_mm256_add_ps(A0, A1), _mm256_add_ps(A2, A3));
    return vaddvq_f32(_mm_add_ps(_mm256_castps256_ps128(S), _mm256_extractf128_ps(S, 1)));
}

void mxfp4_gemv_rows2_avx2(const uint8_t *restrict p, const uint8_t *restrict s,
                           const float *restrict x, float *restrict y,
                           int row0, int row1, int cols) {
    const int cp = cols / 2, groups = cols / 32;
    const __m256i BZ  = _mm256_loadu_si256((const __m256i *)base16);
    const __m256i ZZ  = _mm256_loadu_si256((const __m256i *)mask16);
    const __m256i ZR  = _mm256_setzero_si256();
    const __m128i M0F = _mm_set1_epi8(0x0F);
    const __m256i UZP = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)k3_uzp_bytes));

    float *xp = aligned_alloc(64, ((size_t)cols * sizeof(float) + 63) & ~(size_t)63);
    if (!xp) { mxfp4_gemv_rows2(p, s, x, y, row0, row1, cols); return; }  // fail soft, same math
    k3_permute_x(x, xp, cols);

    int r = row0;
    for (; r + 1 < row1; r += 2) {
        const uint8_t *ppA = p + (size_t)r * cp,       *spA = s + (size_t)r * groups;
        const uint8_t *ppB = p + (size_t)(r + 1) * cp, *spB = s + (size_t)(r + 1) * groups;
        __m256 A0 = _mm256_setzero_ps(), A1 = _mm256_setzero_ps();
        __m256 A2 = _mm256_setzero_ps(), A3 = _mm256_setzero_ps();
        __m256 C0 = _mm256_setzero_ps(), C1 = _mm256_setzero_ps();
        __m256 C2 = _mm256_setzero_ps(), C3 = _mm256_setzero_ps();
        for (int g = 0; g < groups; g++) {
            const float *xg = xp + g * 32;
            const __m256 xq[4] = { _mm256_load_ps(xg),      _mm256_load_ps(xg + 8),
                                   _mm256_load_ps(xg + 16), _mm256_load_ps(xg + 24) };
            __m256i TLA, THA, TLB, THB;
            K3_TABLE(spA[g], TLA, THA);
            K3_GRP(ppA, TLA, THA, xq, A0, A1, A2, A3);
            K3_TABLE(spB[g], TLB, THB);
            K3_GRP(ppB, TLB, THB, xq, C0, C1, C2, C3);
        }
        y[r]     = k3_reduce4(A0, A1, A2, A3);
        y[r + 1] = k3_reduce4(C0, C1, C2, C3);
    }
    free(xp);
    if (r < row1) mxfp4_gemv_rows(p, s, x, y, r, row1, cols);  // odd tail: scalar-count rows
}

void mxfp4_gemv_avx2(const uint8_t *p, const uint8_t *s, const float *x, float *y,
                     int rows, int cols) {
    mxfp4_gemv_rows2_avx2(p, s, x, y, 0, rows, cols);
}
void mxfp4_gemv_mt_avx2(const uint8_t *p, const uint8_t *s, const float *x, float *y,
                        int rows, int cols, int nthreads) {
    run_gemv_mt_k(mxfp4_gemv_rows2_avx2, p, s, x, y, rows, cols, nthreads, 1);
}
int mxfp4_have_avx2(void) { return 1; }
#else
int mxfp4_have_avx2(void) { return 0; }
#endif  // __AVX2__

// simple exported single-gemv for ctypes
void mxfp4_gemv(const uint8_t *p, const uint8_t *s, const float *x, float *y,
                int rows, int cols) {
    mxfp4_gemv_rows(p, s, x, y, 0, rows, cols);
}
void mxfp4_gemv_mt(const uint8_t *p, const uint8_t *s, const float *x, float *y,
                   int rows, int cols, int nthreads) {
    run_gemv_mt(p, s, x, y, rows, cols, nthreads, 1);
}

#ifndef NO_MAIN
// ---------------- harness ----------------
static void *load_file(const char *name, size_t sz) {
    void *buf = aligned_alloc(128, (sz + 127) & ~(size_t)127);
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

    float *y_ref = aligned_alloc(128, R2 * sizeof(float));
    float *y_fus = aligned_alloc(128, R2 * sizeof(float));
    float *y1 = aligned_alloc(128, R13 * sizeof(float));
    float *y3 = aligned_alloc(128, R13 * sizeof(float));
    float *y2 = aligned_alloc(128, R2 * sizeof(float));

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
    // Accelerate baseline correctness
    float *Wbuf = aligned_alloc(128, (size_t)R13 * C13 * sizeof(float));
    dequant_neon_a(w1p, w1s, Wbuf, 0, R13, C13);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, R13, C13, 1.0f, Wbuf, C13, x, 1, 0.0f, y_ref, 1);
    errstats(y_ref, r64_1, R13, "w1 dequant+sgemv vs fp64");

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

    // Accelerate baseline: full dequant (1T neon_a) + sgemv, per matrix, x3 for triple
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

    free(Wbuf);
    return 0;
}
#endif
