// deepseek41 kernels (see kernels7.cuh). Built without fast math: the rotary angles reach ~262k radians, and the softmaxes and
// roundings follow the reference's F32 arithmetic (as maya-ds4's kernels, whose arithmetic these follow; MIT, Copyright (c)
// 2026 Peasant Smith and the Project Maya contributors).
#include "kernels7.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace hyper {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
__device__ __forceinline__ uint16_t f2bf16(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum (blockDim.x a multiple of 32, <= 1024); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float red[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}
// 2^ceil(log2(x)) for a positive normal x (the reference's fast_round_scale)
__device__ __forceinline__ float pow2_ceil(float x) {
    const uint32_t b = __float_as_uint(x);
    const int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t) (e + 127) << 23);
}
// E4M3 round to nearest even, saturated
__device__ __forceinline__ float fp8_rt(float v) {
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    const __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
    return __half2float(__half(h));
}
// E2M1 magnitudes 0 .5 1 1.5 2 3 4 6, nearest, ties to even
__device__ __forceinline__ float e2m1_rt(float v) {
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, v);
}

constexpr int kFinMax = 8;   // values a thread holds (len <= 8 * 256)

__global__ void __launch_bounds__(kThreads) k_fin7(float * x, int len, long long stride, Fin7 f) {
    float * v = x + (long long) blockIdx.x * stride;
    float r[kFinMax];
    const int nj = (len + kThreads - 1) / kThreads;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = block_sum(ss);
        const float inv = rsqrtf(tot / (float) len + f.eps);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    const long long pos = f.pos + (long long) (blockIdx.x / f.vpp) * f.pos_step;
    if (f.rd > 0) {
        const int r0 = len - f.rd;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            const float other = __shfl_xor_sync(0xffffffffu, r[j], 1);
            if (e >= r0 && e < len) {
                const int i = (e - r0) >> 1;
                const float ang = (float) pos * f.freqs[i];
                const float c = cosf(ang);
                const float sn = f.inverse ? -sinf(ang) : sinf(ang);
                if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;   // re = a c - b s
                else r[j] = other * sn + r[j] * c;                          // im = a s + b c
            }
        }
    }
    if (f.quant != 0) {
        const int g = f.quant == 2 ? 16 : 32;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float sc = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = fp8_rt(fminf(fmaxf(r[j] / sc, -448.0f), 448.0f)) * sc;
            } else if (f.quant == 2) {
                const float sc = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / sc, -6.0f), 6.0f)) * sc;
            } else {
                const float sc = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / sc, -6.0f), 6.0f)) * sc;
            }
        }
    }
    uint16_t * d = nullptr;
    if (f.dst != nullptr) {
        const long long row = f.ring > 0 ? pos % f.ring : (long long) blockIdx.x;
        d = f.dst + row * f.dst_stride;
    }
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        if (e >= len) continue;
        if (d != nullptr) d[e] = f2bf16(r[j]);
        else v[e] = r[j];
    }
}

// ---- hyper-connections ----
__global__ void __launch_bounds__(kThreads) k_hc7_mix(const float * __restrict__ h, const float * __restrict__ fn, int hcdim, float eps,
                                                       float * __restrict__ mixes) {
    const int m = blockIdx.x, t = blockIdx.y;
    const float * w = fn + (size_t) m * hcdim;
    const float * x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4 *) (x + k);
        const float4 wv = *(const float4 *) (w + k);
        acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    if (threadIdx.x == 0) mixes[(size_t) t * 32 + m] = acc * rsqrtf(ss / (float) hcdim + eps);
}

// a warp a token: lane j * 4 + k holds comb[j][k] (hc == 4)
__global__ void k_hc7_split(const float * __restrict__ mixes, const float * __restrict__ scale, const float * __restrict__ base, int hc,
                            int iters, float eps, float * pre, float * post, float * comb) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float * mx = mixes + (size_t) t * 32;
    pre += (size_t) t * 8;
    post += (size_t) t * 8;
    comb += (size_t) t * 64;
    if (lane < hc) {
        pre[lane] = 1.0f / (1.0f + expf(-(mx[lane] * scale[0] + base[lane]))) + eps;
        post[lane] = 2.0f / (1.0f + expf(-(mx[hc + lane] * scale[1] + base[hc + lane])));
    }
    const bool on = lane < 16;
    float c = on ? mx[8 + lane] * scale[2] + base[8 + lane] : -FLT_MAX;
    auto row_sum = [&](float v) {
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        return v + __shfl_xor_sync(0xffffffffu, v, 2);
    };
    auto col_sum = [&](float v) {
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        return v + __shfl_xor_sync(0xffffffffu, v, 8);
    };
    float m = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    c = on ? expf(c - m) : 0.0f;
    c = c / row_sum(c) + eps;    // softmax over each row, + eps
    c = on ? c : 0.0f;
    c = c / (col_sum(c) + eps);  // each destination column over its sources
    for (int it = 1; it < iters; ++it) {
        c = c / (row_sum(c) + eps);
        c = c / (col_sum(c) + eps);
    }
    if (on) comb[lane] = c;
}

__global__ void __launch_bounds__(1024) k_hc7_pre_norm(const float * __restrict__ h, const float * __restrict__ mix,
                                                       const float * __restrict__ w, int dim, int hc, float eps, float * __restrict__ x) {
    const int t = blockIdx.x;
    h += (size_t) t * hc * dim;
    mix += (size_t) t * 8;
    x += (size_t) t * dim;
    float r[8];
    float ss = 0.0f;
    int n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) {
        float a = 0.0f;
        for (int s = 0; s < hc; ++s) a += mix[s] * h[(size_t) s * dim + e];
        r[n] = a;
        ss += a * a;
    }
    const float inv = rsqrtf(block_sum(ss) / (float) dim + eps);
    n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) x[e] = w != nullptr ? r[n] * inv * w[e] : r[n];
}

__global__ void k_hc7_post(const float * __restrict__ y, const float * __restrict__ res, const float * __restrict__ post,
                           const float * __restrict__ comb, int dim, int hc, float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= dim * hc) return;
    const int k = i / dim, e = i - k * dim;
    const float * r = res + (size_t) t * hc * dim;
    float a = post[(size_t) t * 8 + k] * y[(size_t) t * dim + e];
    for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * r[(size_t) j * dim + e];
    out[(size_t) t * hc * dim + i] = a;
}

__global__ void k_embed7(const float * __restrict__ x, float * __restrict__ h, int dim, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= dim * hc) return;
    h[(size_t) t * hc * dim + i] = x[(size_t) t * dim + i % dim];
}

// ---- the compressor (ratio > 1), a block, tokens in order ----
__global__ void __launch_bounds__(512) k_compress7(const float * __restrict__ kv, long long kvs, const float * __restrict__ score, long long scs,
                                                   float * kv_state, float * score_state, int ratio, long long p0, int T, int d,
                                                   const float * __restrict__ w, float eps, float * __restrict__ latent) {
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (int) ((p0 + t) % ratio);
        const bool pool = (p0 + t + 1) % ratio == 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x) {
            kv_state[(size_t) slot * d + e] = kv[(size_t) t * kvs + e];
            score_state[(size_t) slot * d + e] = score[(size_t) t * scs + e];
        }
        __syncthreads();
        if (!pool) continue;
        // (kv * softmax(score, over the slots)).sum: each slot's probability first, then the sum, as torch does
        float val[2] = {0.0f, 0.0f};   // d <= 2 * blockDim
        float ss = 0.0f;
        int n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) {
            float m = -FLT_MAX;
            for (int s = 0; s < ratio; ++s) m = fmaxf(m, score_state[(size_t) s * d + e]);
            float den = 0.0f;
            for (int s = 0; s < ratio; ++s) den += expf(score_state[(size_t) s * d + e] - m);
            float acc = 0.0f;
            for (int s = 0; s < ratio; ++s) acc += kv_state[(size_t) s * d + e] * (expf(score_state[(size_t) s * d + e] - m) / den);
            val[n] = acc;
            ss += acc * acc;
        }
        const float inv = rsqrtf(block_sum(ss) / (float) d + eps);
        n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) latent[(size_t) g * d + e] = val[n] * inv * w[e];
        ++g;
        __syncthreads();
    }
}

// ---- the indexer ----
// a thread a position: its key row against every head's query (shared memory), relu, weighted sum
template <int HD>
__global__ void __launch_bounds__(kThreads) k_index_score7(const float * __restrict__ q, const float * __restrict__ w,
                                                           const uint16_t * __restrict__ keys, int heads, long long p0, int R, int width,
                                                           const int * __restrict__ cand, const int * __restrict__ ncand, int cand_stride,
                                                           int blk, float * __restrict__ out, long long out_stride) {
    extern __shared__ float qs[];   // heads * HD, then the heads' weights
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < heads * HD; i += blockDim.x) qs[i] = q[(size_t) t * heads * HD + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * HD + i] = w[(size_t) t * 64 + i];
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    float * o = out + (long long) t * out_stride;
    const long long ln = (p0 + t + 1) / R;
    long long p = i;
    if (cand != nullptr) {
        if (i >= ncand[t] * blk) { o[i] = -INFINITY; return; }
        p = (long long) cand[(size_t) t * cand_stride + i / blk] * blk + i % blk;
    }
    if (p >= ln) { o[i] = -INFINITY; return; }
    const uint16_t * kr = keys + (size_t) p * HD;
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.0f;
#pragma unroll 2
    for (int c = 0; c < HD; c += 8) {
        const uint4 raw = *(const uint4 *) (kr + c);
        const uint16_t * b = (const uint16_t *) &raw;
        float kv[8];
#pragma unroll
        for (int x = 0; x < 8; ++x) kv[x] = bf16f(b[x]);
#pragma unroll
        for (int h = 0; h < 32; ++h) {
            if (h >= heads) break;
            const float * qh = qs + h * HD + c;
#pragma unroll
            for (int x = 0; x < 8; ++x) acc[h] += qh[x] * kv[x];
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < 32; ++h) {
        if (h >= heads) break;
        s += fmaxf(acc[h], 0.0f) * qs[heads * HD + h];
    }
    o[i] = s;
}

__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// a block a row: the k-th largest key by four 8-bit radix passes, then everything above it and enough of its ties (lowest
// positions), in index order
__global__ void __launch_bounds__(1024) k_topk7(const float * __restrict__ vals, long long stride, long long p0, int R, int kcap,
                                                const int * __restrict__ cand, const int * __restrict__ ncand, int cand_stride, int blk,
                                                int nblk_mode, int * __restrict__ out, int out_stride) {
    __shared__ unsigned hist[256];
    __shared__ uint32_t prefix_s, mask_s;
    __shared__ int remaining_s;
    __shared__ int cnt_gt, cnt_eq;
    const int t = blockIdx.x;
    const long long nn = (p0 + t + 1) / R;
    int n, k;
    if (nblk_mode) {   // candidate blocks: n = the query's blocks, k = min(kcap, n)
        n = (int) ((nn + blk - 1) / blk);
        k = min(kcap, n);
    } else if (cand != nullptr) {
        n = ncand[t] * blk;
        k = (int) min((long long) kcap, nn);
    } else {
        n = (int) nn;
        k = (int) min((long long) kcap, nn);
    }
    vals += (long long) t * stride;
    out += (size_t) t * out_stride;
    const int * cd = cand != nullptr ? cand + (size_t) t * cand_stride : nullptr;
    if (n <= 0 || k <= 0) return;
    if (k >= n && cd == nullptr) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = i;
        return;
    }
    if (threadIdx.x == 0) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        cnt_gt = 0;
        cnt_eq = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t u = ord_key(vals[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = (int) hist[d];
                if (cum + h >= rem) {
                    prefix_s = prefix | ((uint32_t) d << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        __syncthreads();
    }
    const uint32_t thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    __shared__ int wgt[32], weq[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const unsigned below = (1u << lane) - 1u;
    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const uint32_t u = i < n ? ord_key(vals[i]) : 0u;
        const bool gt = i < n && u > thr, eq = i < n && u == thr;
        const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            wgt[wid] = __popc(bg);
            weq[wid] = __popc(be);
        }
        __syncthreads();
        if (wid == 0) {
            const int g0 = lane < nw ? wgt[lane] : 0, e0 = lane < nw ? weq[lane] : 0;
            int gs = g0, es = e0;
            for (int o = 1; o < 32; o <<= 1) {
                const int a = __shfl_up_sync(0xffffffffu, gs, o), b = __shfl_up_sync(0xffffffffu, es, o);
                if (lane >= o) {
                    gs += a;
                    es += b;
                }
            }
            if (lane < nw) {
                wgt[lane] = cnt_gt + gs - g0;
                weq[lane] = cnt_eq + es - e0;
            }
            __syncwarp();
            if (lane == 31) {
                cnt_gt += gs;
                cnt_eq += es;
            }
        }
        __syncthreads();
        int slot = -1;
        if (gt) slot = wgt[wid] + __popc(bg & below);
        else if (eq) {
            const int e = weq[wid] + __popc(be & below);
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) out[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        __syncthreads();
    }
}

__global__ void k_block_max7(const float * __restrict__ score, long long stride, long long p0, int R, int blk, float * __restrict__ bm,
                             long long bm_stride) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    const long long n = (p0 + t + 1) / R;
    const long long nb = (n + blk - 1) / blk;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int j = 0; j < blk; ++j) {
        const long long p = (long long) b * blk + j;
        if (p < n) m = fmaxf(m, score[(long long) t * stride + p]);
    }
    if (b == (n - 1) / blk) m = INFINITY;   // the newest position's block is pinned in
    bm[(long long) t * bm_stride + b] = m;
}

__global__ void k_ncand7(long long p0, int R, int blk, int kblocks, int * ncand, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const long long n = (p0 + t + 1) / R;
    ncand[t] = (int) min((long long) kblocks, (n + blk - 1) / blk);
}

// ---- sparse attention, a block a (head, query); d == 512: a thread owns 2 output values, a lane 16 of a row's dot ----
constexpr int kAttTile = 64;

__global__ void __launch_bounds__(kThreads) k_sparse_attn7(const float * __restrict__ q, const uint16_t * __restrict__ ring, int ring_rows,
                                                          int win, const uint16_t * __restrict__ ckv, const int * __restrict__ sel,
                                                          int sel_stride, long long p0, int R, int kcap, const float * __restrict__ sink,
                                                          int heads, int d, float scale, float * __restrict__ out) {
    const int h = blockIdx.x, t = blockIdx.y;
    const long long p = p0 + t;
    const long long first = max(0LL, p - win + 1);
    const int n_win = (int) (p - first + 1);
    const int n_sel = R > 0 ? (int) min((long long) kcap, (p + 1) / R) : 0;
    const int * sl = sel + (size_t) t * sel_stride;
    __shared__ float qs[512];
    __shared__ float sc[kAttTile];
    __shared__ const uint16_t * rows[kAttTile];
    __shared__ float m_run_s, tile_scale_s, den_s;
    for (int i = threadIdx.x; i < d; i += blockDim.x) qs[i] = q[((size_t) t * heads + h) * d + i];
    if (threadIdx.x == 0) {
        m_run_s = -1e30f;
        den_s = 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    float o0 = 0.0f, o1 = 0.0f;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        if (threadIdx.x < nt) {
            const int j = t0 + threadIdx.x;
            rows[threadIdx.x] = j < n_win ? ring + (size_t) ((first + j) % ring_rows) * d : ckv + (size_t) sl[j - n_win] * d;
        }
        __syncthreads();
        for (int jj = wp; jj < nt; jj += kThreads / 32) {
            const uint16_t * rr = rows[jj];
            const uint4 a = *(const uint4 *) (rr + lane * 16);
            const uint4 b = *(const uint4 *) (rr + lane * 16 + 8);
            const uint16_t * pa = (const uint16_t *) &a;
            const uint16_t * pb = (const uint16_t *) &b;
            float s = 0.0f;
#pragma unroll
            for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
            for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
            s = warp_sum(s);
            if (lane == 0) sc[jj] = s * scale;
        }
        __syncthreads();
        if (wp == 0) {
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, sc[j]);
            m = warp_max(m);
            const float m_old = m_run_s;
            const float m_new = fmaxf(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float pr = expf(sc[j] - m_new);
                sc[j] = pr;
                s += pr;
            }
            s = warp_sum(s);
            __syncwarp();
            if (lane == 0) {
                tile_scale_s = expf(m_old - m_new);
                den_s = den_s * tile_scale_s + s;
                m_run_s = m_new;
            }
        }
        __syncthreads();
        const float cs = tile_scale_s;
        o0 *= cs;
        o1 *= cs;
        const int c = threadIdx.x * 2;
        for (int j = 0; j < nt; ++j) {
            const uint32_t pr = *(const uint32_t *) (rows[j] + c);
            o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
            o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
        }
        __syncthreads();
    }
    const float den = den_s + expf(sink[h] - m_run_s);
    const int c = threadIdx.x * 2;
    float * ov = out + ((size_t) t * heads + h) * d;
    ov[c] = o0 / den;
    ov[c + 1] = o1 / den;
}

// ---- the router ----
__global__ void __launch_bounds__(512) k_route7(const float * __restrict__ logits, int ls, const float * __restrict__ bias, int n, int k,
                                                float route_scale, int * ids, float * w) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    const int t = blockIdx.x;
    logits += (size_t) t * ls;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) {
            bv[wp] = best;
            bi[wp] = bix;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int i = 0; i < (int) (blockDim.x >> 5); ++i)
                if (bv[i] > b || (bv[i] == b && bi[i] < ix)) b = bv[i], ix = bi[i];
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        for (int r = 0; r < k; ++r) {
            ids[(size_t) t * k + r] = pick[r];
            w[(size_t) t * k + r] = sc[pick[r]] / (s + 1e-20f) * route_scale;
        }
    }
}

__global__ void k_swiglu7(const float * __restrict__ gu, int gus, float * __restrict__ h, int hs, int ff, float lim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= ff) return;
    float g = gu[(size_t) t * gus + i], u = gu[(size_t) t * gus + ff + i];
    if (lim > 0.0f) {
        g = fminf(g, lim);
        u = fminf(fmaxf(u, -lim), lim);
    }
    h[(size_t) t * hs + i] = g / (1.0f + expf(-g)) * u;
}

__global__ void k_fill7(float * x, float v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ void __launch_bounds__(1024) k_engram_gate7(float * __restrict__ h, const float * __restrict__ kv, const float * __restrict__ w,
                                                       int hc, int dim, float eps) {
    const int k = blockIdx.x, t = blockIdx.y;
    float * hk = h + ((size_t) t * hc + k) * dim;
    const float * kvt = kv + (size_t) t * (hc + 1) * dim;
    const float * key = kvt + (size_t) k * dim;
    const float * val = kvt + (size_t) hc * dim;
    const float * wk = w + (size_t) k * dim;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        const float hv = hk[e], kv_ = key[e];
        sh += hv * hv;
        sk += kv_ * kv_;
        sd += hv * wk[e] * kv_;
    }
    sh = block_sum(sh);
    sk = block_sum(sk);
    sd = block_sum(sd);
    const float rstd = rsqrtf(sh / (float) dim + eps) * rsqrtf(sk / (float) dim + eps);
    const float dot = sd * rstd * rsqrtf((float) dim);
    const float sq = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    const float gate = 1.0f / (1.0f + expf(-sq));
    for (int e = threadIdx.x; e < dim; e += blockDim.x) hk[e] = hk[e] + gate * val[e];
}

} // namespace

void fin7(float * x, int rows, int len, long long stride, const Fin7 & f, cudaStream_t s) {
    if (rows <= 0) return;
    if (len > kFinMax * kThreads || (f.rd & 1) || len % 32) throw std::runtime_error("fin7: bad row length");
    k_fin7<<<rows, kThreads, 0, s>>>(x, len, stride, f);
}
void hc7_mix(const float * h, const float * fn, int hcdim, int nmix, float eps, float * mixes, int nt, cudaStream_t s) {
    k_hc7_mix<<<dim3(nmix, nt), kThreads, 0, s>>>(h, fn, hcdim, eps, mixes);
}
void hc7_split(const float * mixes, const float * scale, const float * base, int hc, int iters, float eps, float * pre, float * post,
               float * comb, int nt, cudaStream_t s) {
    if (hc != 4) throw std::runtime_error("hc7_split: 4 streams only");
    k_hc7_split<<<nt, 32, 0, s>>>(mixes, scale, base, hc, iters, eps, pre, post, comb);
}
void hc7_pre_norm(const float * h, const float * mix, const float * w, int dim, int hc, float eps, float * x, int nt, cudaStream_t s) {
    if (dim > 8 * 1024) throw std::runtime_error("hc7_pre_norm: dim");
    k_hc7_pre_norm<<<nt, 1024, 0, s>>>(h, mix, w, dim, hc, eps, x);
}
void hc7_post(const float * y, const float * res, const float * post, const float * comb, int dim, int hc, float * out, int nt,
              cudaStream_t s) {
    k_hc7_post<<<dim3((dim * hc + 255) / 256, nt), 256, 0, s>>>(y, res, post, comb, dim, hc, out);
}
void embed7(const float * x, float * h, int dim, int hc, int nt, cudaStream_t s) {
    k_embed7<<<dim3((dim * hc + 255) / 256, nt), 256, 0, s>>>(x, h, dim, hc);
}
void compress7(const float * kv, long long kvs, const float * score, long long scs, float * kv_state, float * score_state, int ratio,
               long long p0, int T, int d, const float * w, float eps, float * latent, cudaStream_t s) {
    if (d > 1024) throw std::runtime_error("compress7: d");
    k_compress7<<<1, 512, 0, s>>>(kv, kvs, score, scs, kv_state, score_state, ratio, p0, T, d, w, eps, latent);
}
void index_score7(const float * q, const float * w, const uint16_t * keys, int heads, int hd, int T, long long p0, int R, int width,
                  const int * cand, const int * ncand, int cand_stride, int blk, float * out, long long out_stride, cudaStream_t s) {
    if (hd != 128 || heads > 32) throw std::runtime_error("index_score7: 32 heads of 128");
    if (T <= 0 || width <= 0) return;
    const size_t sm = (size_t) (heads * hd + heads) * sizeof(float);
    k_index_score7<128><<<dim3((width + kThreads - 1) / kThreads, T), kThreads, sm, s>>>(q, w, keys, heads, p0, R, width, cand, ncand,
                                                                                        cand_stride, blk, out, out_stride);
}
void topk7(const float * vals, long long stride, int T, long long p0, int R, int kcap, const int * cand, const int * ncand, int cand_stride,
           int blk, int * out, int out_stride, cudaStream_t s) {
    if (T <= 0) return;
    k_topk7<<<T, 1024, 0, s>>>(vals, stride, p0, R, kcap, cand, ncand, cand_stride, blk, 0, out, out_stride);
}
void cand_blocks7(const float * score, long long stride, int T, long long p0, int R, int blk, int kblocks, float * bm, long long bm_stride,
                  int * cand, int * ncand, int cand_stride, cudaStream_t s) {
    if (T <= 0) return;
    const long long nmax = (p0 + T) / R, nbmax = (nmax + blk - 1) / blk;
    if (nbmax <= 0) return;
    k_block_max7<<<dim3((unsigned) ((nbmax + 255) / 256), T), 256, 0, s>>>(score, stride, p0, R, blk, bm, bm_stride);
    k_topk7<<<T, 1024, 0, s>>>(bm, bm_stride, p0, R, kblocks, nullptr, nullptr, 0, blk, 1, cand, cand_stride);
    k_ncand7<<<(T + 255) / 256, 256, 0, s>>>(p0, R, blk, kblocks, ncand, T);
}
void sparse_attn7(const float * q, const uint16_t * ring, int ring_rows, int win, const uint16_t * ckv, const int * sel, int sel_stride,
                  long long p0, int R, int kcap, const float * sink, int heads, int d, float scale, float * out, int T, cudaStream_t s) {
    if (d != 512) throw std::runtime_error("sparse_attn7: d must be 512");
    k_sparse_attn7<<<dim3(heads, T), kThreads, 0, s>>>(q, ring, ring_rows, win, ckv, sel, sel_stride, p0, R, kcap, sink, heads, d, scale, out);
}
void route7(const float * logits, int ls, const float * bias, int n, int k, float scale, int * ids, float * wts, int nt, cudaStream_t s) {
    if (n > 1024 || k > 16) throw std::runtime_error("route7: sizes");
    k_route7<<<nt, 512, 0, s>>>(logits, ls, bias, n, k, scale, ids, wts);
}
void swiglu7(const float * gu, int gus, float * h, int hs, int ff, float lim, int nt, cudaStream_t s) {
    k_swiglu7<<<dim3((ff + 255) / 256, nt), 256, 0, s>>>(gu, gus, h, hs, ff, lim);
}
void fill7(float * x, float v, int n, cudaStream_t s) { k_fill7<<<(n + 255) / 256, 256, 0, s>>>(x, v, n); }
void engram_gate7(float * h, const float * kv, const float * w, int hc, int dim, float eps, int nt, cudaStream_t s) {
    k_engram_gate7<<<dim3(hc, nt), 1024, 0, s>>>(h, kv, w, hc, dim, eps);
}

} // namespace hyper
