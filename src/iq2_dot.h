// IQ2_XXS x Q8_K row dot products for AVX2, bit-identical to ggml's ggml_vec_dot_iq2_xxs_q8_K: the same int16 pair sums
// (maddubs), the same per-lane int32 sums per 256-block, the same fma chain over the blocks and the same final sum. The
// grid values and their signs come from one table of signed bytes (256 grid entries x 128 sign patterns, 256 KB), and the
// dot runs on the activations offset to unsigned: maddubs(q8 + 128, w) - maddubs(128, w) = the pair sums of q8 * w
// (|w| <= 43: no int16 saturation in either). ggml looks up grid and signs separately and signs the activations.
// Needs iq2xxs_grid / ksigns_iq2xs (ggml-common.h with GGML_COMMON_IMPL_*) and block_iq2_xxs / block_q8_K.
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace hyper {

// [sign pattern 0..127][grid index 0..255]: 8 signed bytes
inline const uint64_t * iq2xxs_signed_grid() {
    static const uint64_t * tab = [] {
        uint64_t * t = (uint64_t *) aligned_alloc(64, 128 * 256 * 8);
        for (int s = 0; s < 128; ++s) {
            const uint8_t sb = ksigns_iq2xs[s];
            for (int g = 0; g < 256; ++g) {
                const uint64_t gv = iq2xxs_grid[g];
                uint64_t o = 0;
                for (int i = 0; i < 8; ++i) {
                    const int8_t v = (int8_t) ((gv >> (8 * i)) & 0xff);
                    o |= (uint64_t) (uint8_t) ((sb >> i) & 1 ? -v : v) << (8 * i);
                }
                t[s * 256 + g] = o;
            }
        }
        return t;
    }();
    return tab;
}

static inline float iq2_hsum8(const __m256 x) {   // (ggml's hsum_float_8)
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// q8u: the Q8_K row's quants + 128 (uint8), 256 per block, in block order; y: the Q8_K row (its scales)
static inline float iq2xxs_dot_u(int n, const void * vx, const block_q8_K * y, const uint8_t * q8u) {
    const block_iq2_xxs * x = (const block_iq2_xxs *) vx;
    const uint64_t * T = iq2xxs_signed_grid();
    const int nb = n / 256;
    const __m256i c128 = _mm256_set1_epi8((char) 0x80);
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        const uint16_t * q2 = x[i].qs;
        const uint8_t * qu = q8u + (size_t) i * 256;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            uint32_t a[4];
            memcpy(a, q2, 16); q2 += 8;
            const __m256i u1 = _mm256_loadu_si256((const __m256i *) qu); qu += 32;
            const __m256i u2 = _mm256_loadu_si256((const __m256i *) qu); qu += 32;
            const uint64_t * T1a = T + ((a[1] >> 0) & 127) * 256, * T1b = T + ((a[1] >> 7) & 127) * 256;
            const uint64_t * T1c = T + ((a[1] >> 14) & 127) * 256, * T1d = T + ((a[1] >> 21) & 127) * 256;
            const uint64_t * T2a = T + ((a[3] >> 0) & 127) * 256, * T2b = T + ((a[3] >> 7) & 127) * 256;
            const uint64_t * T2c = T + ((a[3] >> 14) & 127) * 256, * T2d = T + ((a[3] >> 21) & 127) * 256;
            const __m256i w1 = _mm256_set_epi64x(T1d[(a[0] >> 24) & 255], T1c[(a[0] >> 16) & 255], T1b[(a[0] >> 8) & 255], T1a[a[0] & 255]);
            const __m256i w2 = _mm256_set_epi64x(T2d[(a[2] >> 24) & 255], T2c[(a[2] >> 16) & 255], T2b[(a[2] >> 8) & 255], T2a[a[2] & 255]);
            const __m256i dot1 = _mm256_sub_epi16(_mm256_maddubs_epi16(u1, w1), _mm256_maddubs_epi16(c128, w1));
            const __m256i dot2 = _mm256_sub_epi16(_mm256_maddubs_epi16(u2, w2), _mm256_maddubs_epi16(c128, w2));
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((short) (2 * (a[1] >> 28) + 1)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((short) (2 * (a[3] >> 28) + 1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125f * iq2_hsum8(accumf);
}

// the unsigned activation copy for iq2xxs_dot_u
static inline void q8k_to_u8(const block_q8_K * y, int nb, uint8_t * q8u) {
    const __m256i c = _mm256_set1_epi8((char) 0x80);
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < 256; j += 32)
            _mm256_storeu_si256((__m256i *) (q8u + (size_t) i * 256 + j),
                                _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) (y[i].qs + j)), c));
}

} // namespace hyper
