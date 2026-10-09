// IQ3_S x Q8_K row dot products for AVX2, bit-identical to ggml's ggml_vec_dot_iq3_s_q8_K: the same per-lane int32 sums
// per 256-block (lane j = bytes 4j..4j+3 of each 32-value group, scaled by 2 * ls + 1), the same fma chain over the blocks
// and the same final horizontal sum. Only the grid lookups are built differently.
// Needs iq3s_grid (ggml-common.h with GGML_COMMON_IMPL_*) and block_iq3_s / block_q8_K.
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace hyper {

static inline float iq3s_hsum8(const __m256 x) {   // (ggml's hsum_float_8)
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
static inline float iq3s_h2f(uint16_t h) { return _cvtsh_ss(h); }

// q8 bytes of a 32-value group, negated where the group's 32 sign bits are set
static inline __m256i iq3s_signed_q8(const __m256i q8, uint32_t sign_bits) {
    const __m256i mask1 = _mm256_set_epi64x(0x0303030303030303ll, 0x0202020202020202ll, 0x0101010101010101ll, 0);
    const __m256i mask2 = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    __m256i aux = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) sign_bits), mask1), mask2);
    const __m256i s = _mm256_cmpeq_epi8(aux, mask2);
    return _mm256_sub_epi8(_mm256_xor_si256(s, q8), s);
}

// v1: indices in scalar registers, two grid words per 64-bit move
static inline __m256i iq3s_grid8_v1(const uint8_t * qs, uint32_t qh) {
    const uint64_t a = iq3s_grid[qs[0] | ((qh << 8) & 256)] | (uint64_t) iq3s_grid[qs[1] | ((qh << 7) & 256)] << 32;
    const uint64_t b = iq3s_grid[qs[2] | ((qh << 6) & 256)] | (uint64_t) iq3s_grid[qs[3] | ((qh << 5) & 256)] << 32;
    const uint64_t c = iq3s_grid[qs[4] | ((qh << 4) & 256)] | (uint64_t) iq3s_grid[qs[5] | ((qh << 3) & 256)] << 32;
    const uint64_t d = iq3s_grid[qs[6] | ((qh << 2) & 256)] | (uint64_t) iq3s_grid[qs[7] | ((qh << 1) & 256)] << 32;
    const __m128i lo = _mm_insert_epi64(_mm_cvtsi64_si128((long long) a), (long long) b, 1);
    const __m128i hi = _mm_insert_epi64(_mm_cvtsi64_si128((long long) c), (long long) d, 1);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}
// v2: hardware gather
static inline __m256i iq3s_grid8_v2(const uint8_t * qs, uint32_t qh) {
    const __m256i idx_l = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qs));
    const __m256i idx_h = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32((int) qh), _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8)),
                                           _mm256_set1_epi32(256));
    return _mm256_i32gather_epi32((const int *) iq3s_grid, _mm256_or_si256(idx_l, idx_h), 4);
}

template <__m256i (*grid8)(const uint8_t *, uint32_t)>
static inline float iq3s_dot_t(int n, const void * vx, const void * vy) {
    const block_iq3_s * x = (const block_iq3_s *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    const int nb = n / 256;
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = iq3s_h2f(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs, * qh = x[i].qh;
        const uint16_t * signs = (const uint16_t *) x[i].signs;
        const int8_t * q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q2_1 = grid8(qs, qh[ib32]), q2_2 = grid8(qs + 8, qh[ib32 + 1]);
            qs += 16;
            const __m256i q8s_1 = iq3s_signed_q8(q8_1, signs[0] | ((uint32_t) signs[1] << 16));
            const __m256i q8s_2 = iq3s_signed_q8(q8_2, signs[2] | ((uint32_t) signs[3] << 16));
            signs += 4;
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1), dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32 / 2] & 0xf, ls2 = x[i].scales[ib32 / 2] >> 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16((short) (2 * ls1 + 1))));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16((short) (2 * ls2 + 1))));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return iq3s_hsum8(accumf);
}
static inline void iq3s_dot_v1(int n, float * s, const void * vx, const void * vy) { *s = iq3s_dot_t<iq3s_grid8_v1>(n, vx, vy); }
static inline void iq3s_dot_v2(int n, float * s, const void * vx, const void * vy) { *s = iq3s_dot_t<iq3s_grid8_v2>(n, vx, vy); }

// IQ4_XS x Q8_K over super-blocks [b0, b1) of a row, continuing ggml's ggml_vec_dot_iq4_xs_q8_K accumulator `accum`
// (one fma per super-block in order): parts [0, a) [a, b) ... chained and then iq3s_hsum8 give its result bit for bit
static inline __m256 iq4xs_dot_part(const void * vx, const void * vy, int b0, int b1, __m256 accum) {
    const block_iq4_xs * x = (const block_iq4_xs *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    const __m128i values128 = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    for (int ibl = b0; ibl < b1; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *) qs); qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *) qs); qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q4b_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b))),
                                                          _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)), 1);
            const __m256i q4b_2 = _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b))),
                                                          _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)), 1);
            const __m256i p16_1 = _mm256_maddubs_epi16(_mm256_sign_epi8(q4b_1, q4b_1), _mm256_sign_epi8(q8b_1, q4b_1));
            const __m256i p16_2 = _mm256_maddubs_epi16(_mm256_sign_epi8(q4b_2, q4b_2), _mm256_sign_epi8(q8b_2, q4b_2));
            const int16_t ls1 = ((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            sumi1 = _mm256_add_epi32(_mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1)), sumi1);
            sumi2 = _mm256_add_epi32(_mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2)), sumi2);
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(iq3s_h2f(x[ibl].d) * y[ibl].d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }
    return accum;
}

} // namespace hyper
