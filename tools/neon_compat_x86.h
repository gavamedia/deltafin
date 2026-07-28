// Minimal NEON -> SSE/SSSE3/FMA3 compatibility layer so tools/fused_gemv.c (and
// fused_gemv_batch.c, which #includes it) build unchanged on x86-64.
//
// Scope: ONLY the intrinsics those two files use, mapped 1:1 onto 128-bit vectors
// so every lane, every operation and every rounding matches the NEON build:
//   - float32x4_t maps to __m128; vfmaq_f32 maps to a real FMA3 fused
//     multiply-add (single rounding, same as NEON FMLA).
//   - vaddvq_f32 keeps AArch64's FADDP+FADDP association (v0+v1)+(v2+v3), so the
//     horizontal reduction is bit-identical too.
//   - vqtbl1q_u8 maps to PSHUFB. NEON zeroes lanes whose index is >= 16 while
//     PSHUFB keys off bit 7; every index this codebase feeds it is a nibble
//     (0..15), where the two definitions agree exactly.
//   - vshll_n_u16 / vshll_high_n_u16 are provided for the n == 16 form only
//     (the only form used: u16 halfword -> u32 lane << 16, via punpck with zero).
// A 256-bit (__m256) widening was deliberately NOT done: it would change the
// number of partial accumulators and therefore the last-bit result; this kernel
// is memory-bandwidth-bound streaming packed weights, so 4-wide loses ~nothing.
//
// Requires SSSE3 (PSHUFB) + FMA3, i.e. gcc/clang with -mssse3 -mfma
// (or -mavx2 -mfma / -march=x86-64-v3, both supersets).
#ifndef K3_NEON_COMPAT_X86_H
#define K3_NEON_COMPAT_X86_H

#if !defined(__x86_64__) && !defined(_M_X64)
#error "neon_compat_x86.h is the x86-64 fallback; on ARM include <arm_neon.h>"
#endif
#if !defined(__SSSE3__)
#error "fused_gemv x86 port needs SSSE3 (compile with -mssse3, -mavx2 or -march=x86-64-v3)"
#endif
#if !defined(__FMA__)
#error "fused_gemv x86 port needs FMA3 (compile with -mfma or -march=x86-64-v3)"
#endif

#include <immintrin.h>
#include <stdint.h>

typedef __m128  float32x4_t;
typedef __m128i uint8x16_t;
typedef __m128i uint16x8_t;
typedef __m128i uint32x4_t;
typedef __m128i uint16x4_t;   // vget_low_u16 result; only the low 64 bits are meaningful

// ---- f32 ----
static inline float32x4_t vdupq_n_f32(float v)                      { return _mm_set1_ps(v); }
static inline float32x4_t vld1q_f32(const float *p)                 { return _mm_loadu_ps(p); }
static inline void        vst1q_f32(float *p, float32x4_t v)        { _mm_storeu_ps(p, v); }
static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b)   { return _mm_add_ps(a, b); }
static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b)   { return _mm_mul_ps(a, b); }
// NEON vfmaq_f32(acc, b, c) = acc + b*c with a single rounding; so is VFMADD.
static inline float32x4_t vfmaq_f32(float32x4_t acc, float32x4_t b, float32x4_t c)
                                                                    { return _mm_fmadd_ps(b, c, acc); }
// AArch64 lowers vaddvq_f32 to FADDP+FADDP: (v0+v1)+(v2+v3). Keep that association.
static inline float vaddvq_f32(float32x4_t v) {
    __m128 sw = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));  // v1 v0 v3 v2
    __m128 p  = _mm_add_ps(v, sw);                              // v0+v1 . v2+v3 .
    return _mm_cvtss_f32(_mm_add_ss(p, _mm_movehl_ps(p, p)));   // (v0+v1)+(v2+v3)
}

// ---- u8 ----
static inline uint8x16_t vld1q_u8(const uint8_t *p)  { return _mm_loadu_si128((const __m128i *)p); }
static inline uint8x16_t vdupq_n_u8(uint8_t v)       { return _mm_set1_epi8((char)v); }
static inline uint8x16_t vandq_u8(uint8x16_t a, uint8x16_t b) { return _mm_and_si128(a, b); }
// no 8-bit shift on SSE: 16-bit shift + mask of the bits shifted in from the neighbour
#define vshrq_n_u8(a, n) \
    _mm_and_si128(_mm_srli_epi16((a), (n)), _mm_set1_epi8((char)(0xFFu >> (n))))
static inline uint8x16_t vzip1q_u8(uint8x16_t a, uint8x16_t b) { return _mm_unpacklo_epi8(a, b); }
static inline uint8x16_t vzip2q_u8(uint8x16_t a, uint8x16_t b) { return _mm_unpackhi_epi8(a, b); }
static inline uint8x16_t vuzp1q_u8(uint8x16_t a, uint8x16_t b) {  // even bytes of a, then of b
    const __m128i EV = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
    return _mm_unpacklo_epi64(_mm_shuffle_epi8(a, EV), _mm_shuffle_epi8(b, EV));
}
static inline uint8x16_t vuzp2q_u8(uint8x16_t a, uint8x16_t b) {  // odd bytes of a, then of b
    const __m128i OD = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);
    return _mm_unpacklo_epi64(_mm_shuffle_epi8(a, OD), _mm_shuffle_epi8(b, OD));
}
// valid for indices 0..15 only (see header comment) — always true in this codebase
static inline uint8x16_t vqtbl1q_u8(uint8x16_t tbl, uint8x16_t idx) { return _mm_shuffle_epi8(tbl, idx); }

// ---- u16 ----
static inline uint16x8_t vld1q_u16(const uint16_t *p) { return _mm_loadu_si128((const __m128i *)p); }
static inline uint16x8_t vdupq_n_u16(uint16_t v)      { return _mm_set1_epi16((short)v); }
static inline uint16x8_t vaddq_u16(uint16x8_t a, uint16x8_t b) { return _mm_add_epi16(a, b); }
static inline uint16x8_t vandq_u16(uint16x8_t a, uint16x8_t b) { return _mm_and_si128(a, b); }
static inline uint16x4_t vget_low_u16(uint16x8_t a)   { return a; }
static inline uint32x4_t k3x86_shll16_lo(uint16x4_t a) {           // lanes 0..3  -> u32 << 16
    return _mm_unpacklo_epi16(_mm_setzero_si128(), a);
}
static inline uint32x4_t k3x86_shll16_hi(uint16x8_t a) {           // lanes 4..7  -> u32 << 16
    return _mm_unpackhi_epi16(_mm_setzero_si128(), a);
}
#define vshll_n_u16(a, n)      k3x86_shll16_lo(a)   /* n == 16 form only */
#define vshll_high_n_u16(a, n) k3x86_shll16_hi(a)   /* n == 16 form only */

// ---- reinterprets (integer vectors share __m128i; f32 view is a cast) ----
static inline uint8x16_t  vreinterpretq_u8_u16(uint16x8_t a)  { return a; }
static inline uint16x8_t  vreinterpretq_u16_u8(uint8x16_t a)  { return a; }
static inline float32x4_t vreinterpretq_f32_u32(uint32x4_t a) { return _mm_castsi128_ps(a); }

#endif  // K3_NEON_COMPAT_X86_H
