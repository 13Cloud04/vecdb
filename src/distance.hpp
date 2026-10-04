#pragma once
// Distance kernels. Smaller is always closer:
//   L2            squared Euclidean distance
//   InnerProduct  1 - <a, b>
//   Cosine        1 - <a, b> on vectors normalised at insert and query time
#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace vecdb {

inline float l2_sq_scalar(const float* a, const float* b, std::size_t d) {
    float r = 0;
    for (std::size_t i = 0; i < d; ++i) {
        const float t = a[i] - b[i];
        r += t * t;
    }
    return r;
}

inline float dot_scalar(const float* a, const float* b, std::size_t d) {
    float r = 0;
    for (std::size_t i = 0; i < d; ++i) r += a[i] * b[i];
    return r;
}

// Four independent accumulators: the additions do not depend on each other,
// so the CPU can run them in parallel instead of waiting on one running sum.
inline float l2_sq(const float* a, const float* b, std::size_t d) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
        s2 = vfmaq_f32(s2, d2, d2);
        s3 = vfmaq_f32(s3, d3, d3);
    }
    for (; i + 4 <= d; i += 4) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        s0 = vfmaq_f32(s0, d0, d0);
    }
    float r = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < d; ++i) {
        const float t = a[i] - b[i];
        r += t * t;
    }
    return r;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float r = _mm_cvtss_f32(lo);
    for (; i < d; ++i) {
        const float t = a[i] - b[i];
        r += t * t;
    }
    return r;
#else
    return l2_sq_scalar(a, b, d);
#endif
}

inline float dot(const float* a, const float* b, std::size_t d) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= d; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    float r = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < d; ++i) r += a[i] * b[i];
    return r;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float r = _mm_cvtss_f32(lo);
    for (; i < d; ++i) r += a[i] * b[i];
    return r;
#else
    return dot_scalar(a, b, d);
#endif
}

// One query against four stored vectors at once. For long vectors (text
// embeddings are 6 KB each) a search is limited by memory, not arithmetic:
// walking four vectors in the same loop keeps four memory streams in flight
// and loads each piece of the query once instead of four times.
inline void dot4(const float* q, const float* a, const float* b, const float* c, const float* d, std::size_t dim,
                 float out[4]) {
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const float32x4_t q0 = vld1q_f32(q + i), q1 = vld1q_f32(q + i + 4);
        a0 = vfmaq_f32(a0, q0, vld1q_f32(a + i));
        a1 = vfmaq_f32(a1, q1, vld1q_f32(a + i + 4));
        b0 = vfmaq_f32(b0, q0, vld1q_f32(b + i));
        b1 = vfmaq_f32(b1, q1, vld1q_f32(b + i + 4));
        c0 = vfmaq_f32(c0, q0, vld1q_f32(c + i));
        c1 = vfmaq_f32(c1, q1, vld1q_f32(c + i + 4));
        d0 = vfmaq_f32(d0, q0, vld1q_f32(d + i));
        d1 = vfmaq_f32(d1, q1, vld1q_f32(d + i + 4));
    }
    out[0] = vaddvq_f32(vaddq_f32(a0, a1));
    out[1] = vaddvq_f32(vaddq_f32(b0, b1));
    out[2] = vaddvq_f32(vaddq_f32(c0, c1));
    out[3] = vaddvq_f32(vaddq_f32(d0, d1));
    for (; i < dim; ++i) {
        out[0] += q[i] * a[i];
        out[1] += q[i] * b[i];
        out[2] += q[i] * c[i];
        out[3] += q[i] * d[i];
    }
#else
    out[0] = dot(q, a, dim);
    out[1] = dot(q, b, dim);
    out[2] = dot(q, c, dim);
    out[3] = dot(q, d, dim);
#endif
}

inline void l2_sq4(const float* q, const float* a, const float* b, const float* c, const float* d, std::size_t dim,
                   float out[4]) {
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const float32x4_t q0 = vld1q_f32(q + i), q1 = vld1q_f32(q + i + 4);
        float32x4_t t;
        t = vsubq_f32(q0, vld1q_f32(a + i));      a0 = vfmaq_f32(a0, t, t);
        t = vsubq_f32(q1, vld1q_f32(a + i + 4));  a1 = vfmaq_f32(a1, t, t);
        t = vsubq_f32(q0, vld1q_f32(b + i));      b0 = vfmaq_f32(b0, t, t);
        t = vsubq_f32(q1, vld1q_f32(b + i + 4));  b1 = vfmaq_f32(b1, t, t);
        t = vsubq_f32(q0, vld1q_f32(c + i));      c0 = vfmaq_f32(c0, t, t);
        t = vsubq_f32(q1, vld1q_f32(c + i + 4));  c1 = vfmaq_f32(c1, t, t);
        t = vsubq_f32(q0, vld1q_f32(d + i));      d0 = vfmaq_f32(d0, t, t);
        t = vsubq_f32(q1, vld1q_f32(d + i + 4));  d1 = vfmaq_f32(d1, t, t);
    }
    out[0] = vaddvq_f32(vaddq_f32(a0, a1));
    out[1] = vaddvq_f32(vaddq_f32(b0, b1));
    out[2] = vaddvq_f32(vaddq_f32(c0, c1));
    out[3] = vaddvq_f32(vaddq_f32(d0, d1));
    for (; i < dim; ++i) {
        float t;
        t = q[i] - a[i]; out[0] += t * t;
        t = q[i] - b[i]; out[1] += t * t;
        t = q[i] - c[i]; out[2] += t * t;
        t = q[i] - d[i]; out[3] += t * t;
    }
#else
    out[0] = l2_sq(q, a, dim);
    out[1] = l2_sq(q, b, dim);
    out[2] = l2_sq(q, c, dim);
    out[3] = l2_sq(q, d, dim);
#endif
}

// Squared L2 distance between two vectors of 8-bit codes (scalar quantisation).
// 16 dimensions per step: |a-b| stays in 8 bits, its square fits in 16, and
// pairs of squares are accumulated into 32-bit lanes.
inline std::uint32_t l2_sq_u8_scalar(const std::uint8_t* a, const std::uint8_t* b, std::size_t d) {
    std::uint32_t r = 0;
    for (std::size_t i = 0; i < d; ++i) {
        const std::int32_t t = static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]);
        r += static_cast<std::uint32_t>(t * t);
    }
    return r;
}

inline std::uint32_t l2_sq_u8(const std::uint8_t* a, const std::uint8_t* b, std::size_t d) {
#if defined(__ARM_NEON)
    uint32x4_t s0 = vdupq_n_u32(0), s1 = s0;
    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const uint8x16_t diff = vabdq_u8(vld1q_u8(a + i), vld1q_u8(b + i));
        const uint8x8_t lo = vget_low_u8(diff), hi = vget_high_u8(diff);
        s0 = vpadalq_u16(s0, vmull_u8(lo, lo));
        s1 = vpadalq_u16(s1, vmull_u8(hi, hi));
    }
    std::uint32_t r = vaddvq_u32(vaddq_u32(s0, s1));
    for (; i < d; ++i) {
        const std::int32_t t = static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]);
        r += static_cast<std::uint32_t>(t * t);
    }
    return r;
#else
    return l2_sq_u8_scalar(a, b, d);  // integer addition is associative, so compilers vectorise this themselves
#endif
}

}  // namespace vecdb
