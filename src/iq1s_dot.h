// IQ1_S x Q8_K row dot for AVX2, bit-identical to ggml's ggml_vec_dot_iq1_s_q8_K (its AVX2 code without BMI2: the grid
// indices from shifts and masks). With BMI2 ggml builds them with _pdep_u64, which Zen 2 runs in microcode (~250 cycles
// each, 160 per 5120-row): 7.5 us a row on the 3970X instead of ~0.4 (DeepSeek-V4.1 unsloth UD: gate/up of layers 0-9).
// Needs iq1s_grid (ggml-common.h with GGML_COMMON_IMPL_*), block_iq1_s / block_q8_K and IQ1S_DELTA.
#pragma once
#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace hyper {

static inline float iq1s_hsum8(const __m256 x) {   // (ggml's hsum_float_8)
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i iq1s_mul_add_epi8(const __m256i x, const __m256i y) {   // (ggml's mul_add_epi8)
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

static inline void iq1s_dot(int n, float * s, const void * vx, const void * vy) {
    const block_iq1_s * x = (const block_iq1_s *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {
        const int8_t * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;
        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[qs[3] | ((qh[ib + 0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib + 0] << 2) & 0x700)],
                                                    iq1s_grid[qs[1] | ((qh[ib + 0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib + 0] << 8) & 0x700)]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[qs[7] | ((qh[ib + 1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib + 1] << 2) & 0x700)],
                                                    iq1s_grid[qs[5] | ((qh[ib + 1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib + 1] << 8) & 0x700)]);
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i dot1 = iq1s_mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = iq1s_mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2 * ((qh[ib + 0] >> 12) & 7) + 1;
            const int16_t ls2 = 2 * ((qh[ib + 1] >> 12) & 7) + 1;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (y[i].bsums[2 * ib + 0] + y[i].bsums[2 * ib + 1]) * (qh[ib + 0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2 * ib + 2] + y[i].bsums[2 * ib + 3]) * (qh[ib + 1] & 0x8000 ? -1 : 1) * ls2;
        }
        const float d = y[i].d * _cvtsh_ss(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 = fmaf(d, (float) sumi1, accum1);   // (ggml's gnu11 build contracts d * sumi1 + accum1 into an fma)
    }
    *s = fmaf(IQ1S_DELTA, accum1, iq1s_hsum8(accum));   // (likewise)
}

} // namespace hyper
