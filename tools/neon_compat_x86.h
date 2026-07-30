// Minimal AArch64 NEON compatibility layer for the MXFP4 kernels on x86-64.
//
// This is intentionally narrow: it implements only the vector operations used by
// fused_gemv.c.  The normal exported GEMV functions therefore run the same kernel
// source on both architectures instead of maintaining a second, easily-divergent
// implementation.
//
// Build x86-64 with AVX, SSSE3 and FMA3 enabled (for example, -march=native on
// a compatible machine, or explicitly -mavx -mssse3 -mfma). AVX2 is isolated
// in separately target-attributed functions in fused_gemv.c.

#ifndef K3_NEON_COMPAT_X86_H
#define K3_NEON_COMPAT_X86_H

#if !defined(__x86_64__) && !defined(_M_X64)
#error "neon_compat_x86.h is only for x86-64"
#endif

#include <immintrin.h>
#include <stdint.h>

// MSVC has no per-feature predefined macros and accepts every intrinsic
// regardless of /arch, so this baseline cannot be asserted at compile time
// there.  build_native.py compiles Windows with /arch:AVX for the same VEX
// baseline, and the loader refuses a CPU without AVX/FMA3/SSSE3 before the
// library is used.
#if !defined(_MSC_VER) || defined(__clang__)
#if !defined(__SSSE3__)
#error "The x86-64 MXFP4 kernel requires SSSE3 (build with -mssse3 or -march=native)"
#endif
#if !defined(__FMA__)
#error "The x86-64 MXFP4 kernel requires FMA3 (build with -mfma or -march=native)"
#endif
#endif

typedef __m128  float32x4_t;
typedef __m128i uint8x16_t;
typedef __m128i uint16x4_t;
typedef __m128i uint16x8_t;
typedef __m128i uint32x4_t;

static inline float32x4_t vdupq_n_f32(float x) {
    return _mm_set1_ps(x);
}

static inline uint8x16_t vdupq_n_u8(uint8_t x) {
    return _mm_set1_epi8((char)x);
}

static inline uint16x8_t vdupq_n_u16(uint16_t x) {
    return _mm_set1_epi16((short)x);
}

static inline float32x4_t vld1q_f32(const float *p) {
    return _mm_loadu_ps(p);
}

static inline uint8x16_t vld1q_u8(const uint8_t *p) {
    return _mm_loadu_si128((const __m128i *)p);
}

static inline uint16x8_t vld1q_u16(const uint16_t *p) {
    return _mm_loadu_si128((const __m128i *)p);
}

static inline void vst1q_f32(float *p, float32x4_t x) {
    _mm_storeu_ps(p, x);
}

static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) {
    return _mm_add_ps(a, b);
}

static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) {
    return _mm_mul_ps(a, b);
}

static inline float32x4_t vfmaq_f32(float32x4_t acc, float32x4_t a, float32x4_t b) {
    return _mm_fmadd_ps(a, b, acc);
}

// AArch64 FADDP reduces adjacent lane pairs, then the two pair sums.
static inline float vaddvq_f32(float32x4_t a) {
    __m128 pair = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(_mm_hadd_ps(pair, pair));
}

static inline uint16x8_t vaddq_u16(uint16x8_t a, uint16x8_t b) {
    return _mm_add_epi16(a, b);
}

static inline uint16x8_t vandq_u16(uint16x8_t a, uint16x8_t b) {
    return _mm_and_si128(a, b);
}

static inline uint8x16_t vandq_u8(uint8x16_t a, uint8x16_t b) {
    return _mm_and_si128(a, b);
}

static inline uint8x16_t vreinterpretq_u8_u16(uint16x8_t a) {
    return a;
}

static inline uint16x8_t vreinterpretq_u16_u8(uint8x16_t a) {
    return a;
}

static inline float32x4_t vreinterpretq_f32_u32(uint32x4_t a) {
    return _mm_castsi128_ps(a);
}

static inline uint16x4_t vget_low_u16(uint16x8_t a) {
    return a;
}

// Only a shift of 4 is used by this kernel.  Masking each byte is required
// because SSE shifts 16-bit lanes and would otherwise leak the neighbouring
// byte's low nibble into the result.
#define vshrq_n_u8(a, n) \
    _mm_and_si128(_mm_srli_epi16((a), (n)), _mm_set1_epi8((char)(0xffu >> (n))))

// Widen four u16 lanes to four u32 lanes, then place the original halfword in
// the high 16 bits.  The kernel reinterprets the resulting fp32 bit patterns.
#define vshll_n_u16(a, n) \
    _mm_slli_epi32(_mm_unpacklo_epi16((a), _mm_setzero_si128()), (n))
#define vshll_high_n_u16(a, n) \
    _mm_slli_epi32(_mm_unpackhi_epi16((a), _mm_setzero_si128()), (n))

static inline uint8x16_t vzip1q_u8(uint8x16_t a, uint8x16_t b) {
    return _mm_unpacklo_epi8(a, b);
}

static inline uint8x16_t vzip2q_u8(uint8x16_t a, uint8x16_t b) {
    return _mm_unpackhi_epi8(a, b);
}

static inline uint8x16_t vuzp1q_u8(uint8x16_t a, uint8x16_t b) {
    const __m128i even = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    __m128i lo = _mm_shuffle_epi8(a, even);
    __m128i hi = _mm_slli_si128(_mm_shuffle_epi8(b, even), 8);
    return _mm_or_si128(lo, hi);
}

static inline uint8x16_t vuzp2q_u8(uint8x16_t a, uint8x16_t b) {
    const __m128i odd = _mm_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    __m128i lo = _mm_shuffle_epi8(a, odd);
    __m128i hi = _mm_slli_si128(_mm_shuffle_epi8(b, odd), 8);
    return _mm_or_si128(lo, hi);
}

// PSHUFB only zeroes indices with bit 7 set; NEON TBL zeroes every index >= 16.
// Apply an explicit unsigned range mask so this helper retains the NEON contract.
static inline uint8x16_t vqtbl1q_u8(uint8x16_t table, uint8x16_t indices) {
    const __m128i max_index = _mm_set1_epi8(15);
    const __m128i over = _mm_subs_epu8(indices, max_index);
    const __m128i valid = _mm_cmpeq_epi8(over, _mm_setzero_si128());
    return _mm_and_si128(_mm_shuffle_epi8(table, indices), valid);
}

#endif
