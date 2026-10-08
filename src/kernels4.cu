#include "kernels4.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace hyper {

namespace {

__device__ __forceinline__ float warp_sum4(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ float block_sum4(float v) {
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum4(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    v = lane < nw ? red[lane] : 0.0f;
    return warp_sum4(v);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu4(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float h2f(const uint8_t * p) { __half h; memcpy(&h, p, 2); return __half2float(h); }
// silu(g) * u, or with a limit L > 0: silu(min(g, L)) * clamp(u, -L, L)
__device__ __forceinline__ float swiglu4(float g, float u, float L) {
    if (L > 0.0f) { g = fminf(g, L); u = fminf(fmaxf(u, -L), L); }
    return g / (1.0f + expf(-g)) * u;
}

// ---------------- hyper-connections ----------------
// optional inj: also the hc-row injection dot products restricted to this stream: injp[t][s][j] = inj_w[j][s-slice] . xn
__global__ void k_hc_norm(const float * __restrict__ res, const float * __restrict__ w, float * __restrict__ xn, int n, int hc, float eps,
                          const float * __restrict__ inj_w, float * __restrict__ injp) {
    const int s = blockIdx.x, t = blockIdx.y;
    const float * r = res + ((size_t) t * hc + s) * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += r[i] * r[i];
    const float inv = rsqrtf(block_sum4(ss) / n + eps);
    float * o = xn + ((size_t) t * hc + s) * n;
    float d[4] = {0, 0, 0, 0};
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = r[i] * inv * w[(size_t) s * n + i];
        o[i] = v;
        if (inj_w) for (int j = 0; j < hc; ++j) d[j] += inj_w[(size_t) j * hc * n + (size_t) s * n + i] * v;
    }
    if (inj_w)
        for (int j = 0; j < hc; ++j) {
            const float sum = block_sum4(d[j]);
            if (threadIdx.x == 0) injp[((size_t) t * hc + s) * 4 + j] = sum;
        }
}
// same, thread per float4 (n % 4 == 0, n / 4 <= 1024): no loops, one combined reduction for the injection dots
__global__ void __launch_bounds__(1024) k_hc_norm_v(const float * __restrict__ res, const float * __restrict__ w, float * __restrict__ xn,
                                                    int n, int hc, float eps, const float * __restrict__ inj_w, float * __restrict__ injp) {
    const int s = blockIdx.x, t = blockIdx.y, i = threadIdx.x, lane = i & 31, wid = i >> 5, nw = blockDim.x >> 5;
    const bool on = i < n / 4;
    const size_t row = ((size_t) t * hc + s) * n;
    const float4 r = on ? ((const float4 *) (res + row))[i] : make_float4(0, 0, 0, 0);
    __shared__ float red[32][4];
    float ss = warp_sum4(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
    if (lane == 0) red[wid][0] = ss;
    __syncthreads();
    ss = lane < nw ? red[lane][0] : 0.0f;
    const float inv = rsqrtf(warp_sum4(ss) / n + eps);
    if (!on && !inj_w) return;
    float4 v = make_float4(0, 0, 0, 0);
    if (on) {
        const float4 g = ((const float4 *) (w + (size_t) s * n))[i];
        v = make_float4(r.x * inv * g.x, r.y * inv * g.y, r.z * inv * g.z, r.w * inv * g.w);
        ((float4 *) (xn + row))[i] = v;
    }
    if (!inj_w) return;
    float d[4] = {0, 0, 0, 0};
    if (on)
        for (int j = 0; j < hc; ++j) {
            const float4 q = ((const float4 *) (inj_w + (size_t) j * hc * n + (size_t) s * n))[i];
            d[j] = q.x * v.x + q.y * v.y + q.z * v.z + q.w * v.w;
        }
#pragma unroll
    for (int j = 0; j < 4; ++j) d[j] = warp_sum4(d[j]);
    __syncthreads();   // (red reused)
    if (lane == 0) for (int j = 0; j < 4; ++j) red[wid][j] = d[j];
    __syncthreads();
    if (wid == 0) {
        for (int j = 0; j < hc; ++j) {
            const float x = warp_sum4(lane < nw ? red[lane][j] : 0.0f);
            if (lane == 0) injp[((size_t) t * hc + s) * 4 + j] = x;
        }
    }
}
// inj[t][j] = sum_s injp[t][s][j]  (stream order: deterministic)
__global__ void k_hc_inj_sum(const float * __restrict__ injp, float * __restrict__ inj, int hc) {
    const int t = blockIdx.x, j = threadIdx.x;
    if (j >= hc) return;
    float a = 0.0f;
    for (int s = 0; s < hc; ++s) a += injp[((size_t) t * hc + s) * 4 + j];
    inj[(size_t) t * 4 + j] = a;
}
__global__ void k_silu_scale(float * x, int n, float scale, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float * p = x + (size_t) blockIdx.y * stride + i; *p = silu4(*p * scale); }
}
__global__ void k_hc_mixed(const float * __restrict__ xn, const float * __restrict__ gate, float * __restrict__ mixed, int n, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= n) return;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s) {
        const size_t i = ((size_t) t * hc + s) * n + e;
        acc += xn[i] * sigm(gate[i]);
    }
    mixed[(size_t) t * n + e] = acc / hc;
}
__global__ void k_hc_combine(float * res, const float * __restrict__ bo, const float * __restrict__ inject, int istride, int n, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= hc * n) return;
    const int s = i / n, e = i % n;
    const float w = 2.0f * sigm(inject[(size_t) t * istride + s] / hc);
    res[(size_t) t * hc * n + i] += bo[(size_t) t * n + e] * w;
}
__global__ void k_hc_init(float * res, const float * __restrict__ x, int n, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i < hc * n) res[(size_t) t * hc * n + i] = x[(size_t) t * n + i % n];
}

// block per row (few rows, long k): injections
__global__ void k_gemv_f32_br(const float * __restrict__ W, int k, const float * __restrict__ x, int xs, float * __restrict__ y, int ys) {
    const int r = blockIdx.x, t = blockIdx.y;
    const float * wr = W + (size_t) r * k, * xr = x + (size_t) t * xs;
    float acc = 0.0f;
    for (int i = threadIdx.x * 4; i < k; i += blockDim.x * 4) {
        const float4 a = *(const float4 *) (wr + i), b = *(const float4 *) (xr + i);
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = block_sum4(acc);
    if (threadIdx.x == 0) y[(size_t) t * ys + r] = acc;
}
// warp per row, fp32 weights
__global__ void k_gemv_f32(const float * __restrict__ W, int rows, int k, const float * __restrict__ x, int xs, float * __restrict__ y, int ys) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), t = blockIdx.y, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float * wr = W + (size_t) r * k;
    const float * xr = x + (size_t) t * xs;
    float acc = 0.0f;
    if ((k & 3) == 0) {
        for (int i = lane * 4; i < k; i += 128) {
            const float4 a = *(const float4 *) (wr + i), b = *(const float4 *) (xr + i);
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = lane; i < k; i += 32) acc += wr[i] * xr[i];
    }
    acc = warp_sum4(acc);
    if (lane == 0) y[(size_t) t * ys + r] = acc;
}

__global__ void k_gated_norm_sig(float * o, int o_stride, const float * __restrict__ z, int z_stride, const float * __restrict__ w,
                                 int dh, float eps) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    float * op = o + (size_t) t * o_stride + (size_t) h * dh;
    float x = op[i];
    const float ss = block_sum4(x * x);
    x = x * rsqrtf(ss / dh + eps) * w[i];
    op[i] = x * sigm(z[(size_t) t * z_stride + (size_t) h * dh + i]);
}

// ---------------- routing ----------------
// block per token (256 threads, ne <= 1024): softmax; top-k in two stages: each warp keeps the top-k of its slice
// (k rounds of warp argmax, ties to the lower index), then warp 0 merges the 8 * k candidates the same way
__global__ void k_moe_route(const float * __restrict__ logits, int ls, int ne, int k, int * ids, float * wts, float * sg, bool rank_sel) {
    const int t = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float * l = logits + (size_t) t * ls;
    __shared__ float p[1024];
    __shared__ float cv[8 * MOE_MAX_USED];
    __shared__ int ci[8 * MOE_MAX_USED];
    float mx = -FLT_MAX;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) mx = fmaxf(mx, l[e]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    __shared__ float wm[8];
    if (lane == 0) wm[w] = mx;
    __syncthreads();
    mx = wm[0];
    for (int i = 1; i < nw; ++i) mx = fmaxf(mx, wm[i]);
    float sum = 0.0f;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) { const float v = expf(l[e] - mx); p[e] = v; sum += v; }
    sum = block_sum4(sum);
    for (int e = threadIdx.x; e < ne; e += blockDim.x) p[e] /= sum;
    __syncthreads();
    if (rank_sel) {   // expert e is taken at rank = #experts ahead of it (greater, or equal with a lower index) if rank < k
        for (int e = threadIdx.x; e < ne; e += blockDim.x) {
            const float pe = p[e];
            int ahead = 0;
            for (int j = 0; j < ne && ahead < k; ++j) { const float q = p[j]; ahead += q > pe || (q == pe && j < e); }
            if (ahead < k) { ci[ahead] = e; cv[ahead] = pe; }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.0f;
            for (int j = 0; j < k; ++j) s += cv[j];
            for (int j = 0; j < k; ++j) { ids[t * k + j] = ci[j]; wts[t * k + j] = cv[j] / s; }
            sg[t] = sigm(l[ne]);
        }
        return;
    }
    // stage 1: warp w owns experts [w*span, (w+1)*span)
    const int span = (ne + nw - 1) / nw, e0 = w * span, e1 = min(ne, e0 + span);
    unsigned long long taken[2] = {0ull, 0ull};   // per lane: which of its (up to 4) candidates are used (bit = slot)
    for (int j = 0; j < k; ++j) {
        float v = -1.0f; int vi = 0x7fffffff, slot = -1;
        for (int q = 0, e = e0 + lane; e < e1; e += 32, ++q)
            if (!((taken[q >> 6] >> (q & 63)) & 1ull) && (p[e] > v || (p[e] == v && e < vi))) { v = p[e]; vi = e; slot = q; }
        float bv = v; int bi = vi;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, o); const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (bi == vi && slot >= 0) taken[slot >> 6] |= 1ull << (slot & 63);
        if (lane == 0) { cv[w * k + j] = bv; ci[w * k + j] = bi; }
    }
    __syncthreads();
    // stage 2: warp 0 merges nw * k candidates
    if (w == 0) {
        const int nc = nw * k;
        unsigned used = 0;   // lane-local: bit q = candidate lane + 32 q taken
        for (int j = 0; j < k; ++j) {
            float v = -1.0f; int vi = 0x7fffffff, slot = -1;
            for (int q = 0, c = lane; c < nc; c += 32, ++q)
                if (!((used >> q) & 1u) && ci[c] < 0x7fffffff && (cv[c] > v || (cv[c] == v && ci[c] < vi))) { v = cv[c]; vi = ci[c]; slot = q; }
            float bv = v; int bi = vi;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, bv, o); const int oi = __shfl_xor_sync(0xffffffff, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (bi == vi && slot >= 0) used |= 1u << slot;
            if (lane == 0) { ids[t * k + j] = bi; cv[nc + j] = bv; }   // cv tail: selected probabilities
        }
        __syncwarp();
        if (lane == 0) {
            float s = 0.0f;
            for (int j = 0; j < k; ++j) s += cv[nc + j];
            for (int j = 0; j < k; ++j) wts[t * k + j] = cv[nc + j] / s;
            sg[t] = sigm(l[ne]);
        }
    }
}

// ---------------- expert GEMV on GGUF blocks ----------------
// 8 consecutive weights (chunk c of the row) dequantized
template <GType T> __device__ __forceinline__ void deq8(const uint8_t * __restrict__ row, int c, float * v);

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
template <> __device__ __forceinline__ void deq8<GType::Q4_K>(const uint8_t * __restrict__ row, int c, float * v) {
    const uint8_t * b = row + (size_t) (c >> 5) * 144;
    const int o = (c & 31) * 8, j = o >> 5, l = o & 31;
    int sc, m; scale_min_k4(j, b + 4, sc, m);
    const float d = h2f(b) * sc, mn = h2f(b + 2) * m;
    const uint2 qq = *(const uint2 *) (b + 16 + 32 * (j >> 1) + l);   // 8-byte aligned: blocks are 144 B
    const int sh = 4 * (j & 1);
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = d * ((qq.x >> (8 * i + sh)) & 0xF) - mn; v[4 + i] = d * ((qq.y >> (8 * i + sh)) & 0xF) - mn; }
}
template <> __device__ __forceinline__ void deq8<GType::Q5_K>(const uint8_t * __restrict__ row, int c, float * v) {
    const uint8_t * b = row + (size_t) (c >> 5) * 176;
    const int o = (c & 31) * 8, j = o >> 5, l = o & 31;
    int sc, m; scale_min_k4(j, b + 4, sc, m);
    const float d = h2f(b) * sc, mn = h2f(b + 2) * m;
    const uint8_t * qh = b + 16 + l;
    const uint8_t * ql = b + 48 + 32 * (j >> 1) + l;
    const int sh = 4 * (j & 1);
    const uint8_t u = (uint8_t) (1u << j);   // bit 2*(j>>1) + (j&1) == j
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = d * (((ql[i] >> sh) & 0xF) + ((qh[i] & u) ? 16 : 0)) - mn;
}
template <> __device__ __forceinline__ void deq8<GType::Q5_1>(const uint8_t * __restrict__ row, int c, float * v) {
    const uint8_t * b = row + (size_t) (c >> 2) * 24;
    const float d = h2f(b), m = h2f(b + 2);
    uint32_t qh; memcpy(&qh, b + 4, 4);
    const uint8_t * qs = b + 8;
    const int o = (c & 3) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int jj = o + i;
        int x;
        if (jj < 16) x = (qs[jj] & 0xF) | (((qh >> jj) << 4) & 0x10);
        else { const int j2 = jj - 16; x = (qs[j2] >> 4) | ((qh >> (j2 + 12)) & 0x10); }
        v[i] = x * d + m;
    }
}
template <> __device__ __forceinline__ void deq8<GType::Q5_0>(const uint8_t * __restrict__ row, int c, float * v) {
    const uint8_t * b = row + (size_t) (c >> 2) * 22;   // d, qh[4], qs[16]: x = (nibble | high bit << 4) - 16
    const float d = h2f(b);
    uint32_t qh; memcpy(&qh, b + 2, 4);
    const uint8_t * qs = b + 6;
    const int o = (c & 3) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int jj = o + i;
        int x;
        if (jj < 16) x = (qs[jj] & 0xF) | (((qh >> jj) << 4) & 0x10);
        else { const int j2 = jj - 16; x = (qs[j2] >> 4) | ((qh >> (j2 + 12)) & 0x10); }
        v[i] = (x - 16) * d;
    }
}
// IQ3_S grid and IQ4_NL values: from ggml (ggml-common.h), MIT, Copyright (c) 2023-2026 The ggml authors
__device__ const uint32_t g_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};
__device__ const int8_t g_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
// lookup tables in shared memory (divergent lookups): every kernel that dequantizes calls load_tables<T>() and syncs
__shared__ uint32_t s_iq3s_grid[512];
__shared__ float s_iq4nl[16];
template <GType T> __device__ __forceinline__ void load_tables() {
    if (T == GType::IQ3_S) for (int i = threadIdx.x; i < 512; i += blockDim.x) s_iq3s_grid[i] = g_iq3s_grid[i];
    if (T == GType::IQ4_XS) for (int i = threadIdx.x; i < 16; i += blockDim.x) s_iq4nl[i] = g_iq4nl[i];
}
template <> __device__ __forceinline__ void deq8<GType::IQ4_XS>(const uint8_t * __restrict__ row, int c, float * v) {
    const uint8_t * b = row + (size_t) (c >> 5) * 136;
    const int o = (c & 31) * 8, ib = o >> 5, l = o & 31;
    uint16_t sh; memcpy(&sh, b + 2, 2);
    const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = h2f(b) * (ls - 32);
    const uint8_t * q = b + 8 + ib * 16 + (l & 15);
    const int shift = l >= 16 ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = dl * s_iq4nl[(q[i] >> shift) & 0xf];
}
template <> __device__ __forceinline__ void deq8<GType::Q6_K>(const uint8_t * __restrict__ row, int c, float * v) {
    const uint8_t * b = row + (size_t) (c >> 5) * 210;
    const int e = (c & 31) * 8, hf = e >> 7, qt = (e >> 5) & 3, l = e & 31;
    const uint8_t * ql = b + hf * 64 + (qt & 1) * 32 + l;
    const uint8_t * qh = b + 128 + hf * 32 + l;
    const int8_t sc = (int8_t) b[192 + hf * 8 + (l >> 4) + 2 * qt];
    const float d = h2f(b + 208) * sc;
    const int s1 = qt >= 2 ? 4 : 0, s2 = 2 * qt;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = d * ((((ql[i] >> s1) & 0xF) | (((qh[i] >> s2) & 3) << 4)) - 32);
}
template <> __device__ __forceinline__ void deq8<GType::IQ3_S>(const uint8_t * __restrict__ row, int c, float * v) {
    const uint8_t * b = row + (size_t) (c >> 5) * 110;
    const int cc = c & 31, ib = cc >> 2, l = cc & 3;
    const float db = h2f(b) * (1 + 2 * ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf));
    const uint8_t * qs = b + 2 + ib * 8 + 2 * l;
    const int qh = b[66 + ib];
    const uint8_t sg = b[74 + ib * 4 + l];
    const uint32_t g1 = s_iq3s_grid[qs[0] | ((qh << (8 - 2 * l)) & 256)], g2 = s_iq3s_grid[qs[1] | ((qh << (7 - 2 * l)) & 256)];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = db * (float) ((g1 >> (8 * j)) & 0xff) * ((sg >> j) & 1 ? -1.0f : 1.0f);
        v[4 + j] = db * (float) ((g2 >> (8 * j)) & 0xff) * ((sg >> (4 + j)) & 1 ? -1.0f : 1.0f);
    }
}
template <> __device__ __forceinline__ void deq8<GType::Q8_0>(const uint8_t * __restrict__ row, int c, float * v) {
    const uint8_t * b = row + (size_t) (c >> 2) * 34;
    const float d = h2f(b);
    const int8_t * q = (const int8_t *) (b + 2 + (c & 3) * 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = d * q[i];
}

template <GType T> __device__ __forceinline__ float dot_row(const uint8_t * __restrict__ row, const float * xs, int k, int lane) {
    float acc = 0.0f;
#pragma unroll 2
    for (int c = lane; c < k / 8; c += 32) {
        float v[8];
        deq8<T>(row, c, v);
        const float4 x0 = *(const float4 *) (xs + c * 8), x1 = *(const float4 *) (xs + c * 8 + 4);
        acc += v[0] * x0.x + v[1] * x0.y + v[2] * x0.z + v[3] * x0.w + v[4] * x1.x + v[5] * x1.y + v[6] * x1.z + v[7] * x1.w;
    }
    return warp_sum4(acc);
}

constexpr int MOE_ROWS = 8;   // rows per block: MOE_ROWS / 8 per warp

template <GType T>
__global__ void k_moe_gate_up(MoeDev m, const float * __restrict__ x, int xs, const int * __restrict__ ids, int k, float * __restrict__ h, int kdim,
                              const int * __restrict__ order, const int * __restrict__ order_n) {
    if (order && (int) blockIdx.x >= *order_n) return;
    const int p = order ? order[blockIdx.x] : blockIdx.x, t = p / k;
    const int slot = m.slot[ids[p]];
    if (slot < 0) return;
    extern __shared__ float xsm[];
    load_tables<T>();
    for (int i = threadIdx.x; i < kdim; i += blockDim.x) xsm[i] = x[(size_t) t * xs + i];
    __syncthreads();
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t rb = m.gate_bytes / m.ff;
    // the warp's 4 rows of gate and up advance together: 8 independent load streams per lane
    constexpr int RW = MOE_ROWS / 8;
    const int r0 = blockIdx.y * MOE_ROWS + w * RW;
    const uint8_t * gr[RW], * ur[RW];
#pragma unroll
    for (int rr = 0; rr < RW; ++rr) {
        const int r = min(r0 + rr, m.ff - 1);
        gr[rr] = m.gate + slot * m.gate_bytes + r * rb;
        ur[rr] = m.up + slot * m.gate_bytes + r * rb;
    }
    float ag[RW] = {}, au[RW] = {};
    for (int c = lane; c < kdim / 8; c += 32) {
        const float4 x0 = *(const float4 *) (xsm + c * 8), x1 = *(const float4 *) (xsm + c * 8 + 4);
#pragma unroll
        for (int rr = 0; rr < RW; ++rr) {
            float v[8], q[8];
            deq8<T>(gr[rr], c, v);
            deq8<T>(ur[rr], c, q);
            ag[rr] += v[0] * x0.x + v[1] * x0.y + v[2] * x0.z + v[3] * x0.w + v[4] * x1.x + v[5] * x1.y + v[6] * x1.z + v[7] * x1.w;
            au[rr] += q[0] * x0.x + q[1] * x0.y + q[2] * x0.z + q[3] * x0.w + q[4] * x1.x + q[5] * x1.y + q[6] * x1.z + q[7] * x1.w;
        }
    }
#pragma unroll
    for (int rr = 0; rr < RW; ++rr) {
        const float g = warp_sum4(ag[rr]), u = warp_sum4(au[rr]);
        if (lane == 0 && r0 + rr < m.ff) h[(size_t) p * m.ff + r0 + rr] = swiglu4(g, u, m.clamp);
    }
}

template <GType T>
__global__ void k_moe_down(MoeDev m, const float * __restrict__ h, const int * __restrict__ ids, const float * __restrict__ wts, int k,
                           float * __restrict__ y, const int * __restrict__ order, const int * __restrict__ order_n) {
    if (order && (int) blockIdx.x >= *order_n) return;
    const int p = order ? order[blockIdx.x] : blockIdx.x;
    const int slot = m.slot[ids[p]];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (slot < 0) {
        if (order) return;
        for (int i = threadIdx.x; i < MOE_ROWS; i += blockDim.x) {
            const int r = blockIdx.y * MOE_ROWS + i;
            if (r < m.n) y[(size_t) p * m.n + r] = 0.0f;
        }
        return;
    }
    extern __shared__ float hsm[];
    load_tables<T>();
    for (int i = threadIdx.x; i < m.ff; i += blockDim.x) hsm[i] = h[(size_t) p * m.ff + i];
    __syncthreads();
    const size_t rb = m.down_bytes / m.n;
    const float wt = wts[p];
    constexpr int RW = MOE_ROWS / 8;
    const int r0 = blockIdx.y * MOE_ROWS + w * RW;
    const uint8_t * dr[RW];
#pragma unroll
    for (int rr = 0; rr < RW; ++rr) dr[rr] = m.down + slot * m.down_bytes + min(r0 + rr, m.n - 1) * rb;
    float acc[RW] = {};
    for (int c = lane; c < m.ff / 8; c += 32) {
        const float4 x0 = *(const float4 *) (hsm + c * 8), x1 = *(const float4 *) (hsm + c * 8 + 4);
#pragma unroll
        for (int rr = 0; rr < RW; ++rr) {
            float v[8];
            deq8<T>(dr[rr], c, v);
            acc[rr] += v[0] * x0.x + v[1] * x0.y + v[2] * x0.z + v[3] * x0.w + v[4] * x1.x + v[5] * x1.y + v[6] * x1.z + v[7] * x1.w;
        }
    }
#pragma unroll
    for (int rr = 0; rr < RW; ++rr) {
        const float v = warp_sum4(acc[rr]);
        if (lane == 0 && r0 + rr < m.n) y[(size_t) p * m.n + r0 + rr] = wt * v;
    }
}

// ---- zero-copy share of the CPU experts: hidden slice [f0, f1) read straight from mapped host memory ----
// row bytes staged through shared memory with 8-byte loads (PCIe reads stay coalesced), then the usual block dot
template <GType T>
__global__ void __launch_bounds__(256) k_moe_zc_gate_up(MoeZC z, const float * __restrict__ x, int xs, const int * __restrict__ ids, int k,
                                                       float * __restrict__ h, int kdim) {
    const int p = blockIdx.x, t = p / k;
    const int cs = z.cslot[ids[p]];
    if (cs < 0) return;
    extern __shared__ __align__(16) unsigned char zsm[];
    float * xsm = (float *) zsm;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t rb = z.gate_bytes / z.ff;
    uint2 * rows = (uint2 *) (zsm + (size_t) kdim * 4) + (size_t) w * (2 * rb / 8);
    load_tables<T>();
    for (int i = threadIdx.x; i < kdim; i += blockDim.x) xsm[i] = x[(size_t) t * xs + i];
    __syncthreads();
    for (int rr = 0; rr < 4; ++rr) {
        const int r = z.f0 + blockIdx.y * 32 + w * 4 + rr;
        if (r >= z.f1) break;
        const uint2 * gsrc = (const uint2 *) (z.gate + (size_t) cs * z.gate_bytes + (size_t) r * rb);
        const uint2 * usrc = (const uint2 *) (z.up + (size_t) cs * z.gate_bytes + (size_t) r * rb);
        for (int i = lane; i < (int) (rb / 8); i += 32) { rows[i] = gsrc[i]; rows[rb / 8 + i] = usrc[i]; }
        __syncwarp();
        const float gv = dot_row<T>((const uint8_t *) rows, xsm, kdim, lane);
        const float uv = dot_row<T>((const uint8_t *) (rows + rb / 8), xsm, kdim, lane);
        if (lane == 0) h[(size_t) p * (z.f1 - z.f0) + (r - z.f0)] = swiglu4(gv, uv, z.clamp);
        __syncwarp();
    }
}
template <GType T>
__global__ void __launch_bounds__(256) k_moe_zc_down(MoeZC z, const float * __restrict__ h, const int * __restrict__ ids,
                                                    const float * __restrict__ wts, int k, float * __restrict__ y) {
    const int p = blockIdx.x;
    const int cs = z.cslot[ids[p]];
    if (cs < 0) return;
    const int len = z.f1 - z.f0;
    extern __shared__ __align__(16) unsigned char zsm[];
    float * hsm = (float *) zsm;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t rb = z.down_bytes / z.n;
    const size_t seg = rb / (z.ff / 256) * (len / 256), off = rb / (z.ff / 256) * (z.f0 / 256);   // 256-wide blocks
    uint2 * rows = (uint2 *) (zsm + (size_t) len * 4) + (size_t) w * (seg / 8);
    load_tables<T>();
    for (int i = threadIdx.x; i < len; i += blockDim.x) hsm[i] = h[(size_t) p * len + i];
    __syncthreads();
    const float wt = wts[p];
    for (int rr = 0; rr < 4; ++rr) {
        const int r = blockIdx.y * 32 + w * 4 + rr;
        if (r >= z.n) break;
        const uint2 * src = (const uint2 *) (z.down + (size_t) cs * z.down_bytes + (size_t) r * rb + off);
        for (int i = lane; i < (int) (seg / 8); i += 32) rows[i] = src[i];
        __syncwarp();
        const float v = dot_row<T>((const uint8_t *) rows, hsm, len, lane);
        if (lane == 0) y[(size_t) p * z.n + r] = wt * v;
        __syncwarp();
    }
}

__global__ void k_moe_reduce(const float * __restrict__ shexp, const float * __restrict__ sg, const float * __restrict__ y, int k,
                             float * __restrict__ out, int n, const int * __restrict__ ids, const int * __restrict__ owner, int g, int cpu_owner,
                             const volatile unsigned * cpu_flag, const float * cpu_y, const int * counter, unsigned seq_tag,
                             const float * __restrict__ yzc) {
    const int t = blockIdx.y, r = blockIdx.x * blockDim.x + threadIdx.x;
    bool need = false;
    if (cpu_flag) for (int j = 0; j < k; ++j) need |= owner[ids[t * k + j]] == cpu_owner;
    if (need) {
        if (threadIdx.x == 0) {
            const unsigned want = (unsigned) (*counter) * 64u + seq_tag;
            while (*cpu_flag != want) {}
            __threadfence_system();
        }
        __syncthreads();
    }
    if (r >= n) return;
    float acc = sg[t] * shexp[(size_t) t * n + r];
    for (int j = 0; j < k; ++j) if (owner[ids[t * k + j]] == g) acc += y[(size_t) (t * k + j) * n + r];
    if (yzc) for (int j = 0; j < k; ++j) if (owner[ids[t * k + j]] == cpu_owner) acc += yzc[(size_t) (t * k + j) * n + r];
    if (need) acc += __ldcv(cpu_y + (size_t) t * 4096 + r);
    out[(size_t) t * n + r] = acc;
}

__global__ void k_moe_publish(int * ntp, int * ids_dst, float * wts_dst, float * x_dst, const float * __restrict__ x, int xs, int n,
                              const int * __restrict__ ids, const float * __restrict__ wts, int k, int nt) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t) nt * n) { const int t = (int) (i / n), e = (int) (i % n); x_dst[(size_t) t * 4096 + e] = x[(size_t) t * xs + e]; }
    if (i < (size_t) nt * k) { const int t = (int) (i / k), j = (int) (i % k); ids_dst[t * MOE_MAX_USED + j] = ids[i]; wts_dst[t * MOE_MAX_USED + j] = wts[i]; }
    if (i == 0) *ntp = nt;
    __threadfence_system();
}
// decode (nt <= MAX_NT): one block copies everything and then raises the sequence tag itself (one launch, ordered by the
// block barrier + system fence instead of a second kernel)
__global__ void k_moe_publish1(volatile unsigned * seq, int * ntp, int * ids_dst, float * wts_dst, float * x_dst, const float * __restrict__ x,
                               int xs, int n, const int * __restrict__ ids, const float * __restrict__ wts, int k, int nt, const int * counter,
                               unsigned seq_tag) {
    for (int i = threadIdx.x; i < nt * n / 2; i += blockDim.x) {   // (x_dst sits in a host record, 8-byte aligned only)
        const int t = i / (n / 2), e = (i % (n / 2)) * 2;
        *(float2 *) (x_dst + (size_t) t * 4096 + e) = *(const float2 *) (x + (size_t) t * xs + e);
    }
    for (int i = threadIdx.x; i < nt * k; i += blockDim.x) {
        const int t = i / k, j = i % k;
        ids_dst[t * MOE_MAX_USED + j] = ids[i]; wts_dst[t * MOE_MAX_USED + j] = wts[i];
    }
    if (threadIdx.x == 0) *ntp = nt;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); *seq = (unsigned) (*counter) * 64u + seq_tag; }
}
__global__ void k_moe_seq(volatile unsigned * seq, const int * counter, unsigned seq_tag) {
    __threadfence_system();
    *seq = (unsigned) (*counter) * 64u + seq_tag;
}
// single block: counting sort of the local pairs by expert, plus the active experts as (expert, start, count) in
// egrp[3 * i ..] (i < order_n[1]); pairs within an expert are ordered by pair index (deterministic)
__global__ void k_moe_order(const int * __restrict__ slot, const int * __restrict__ ids, int P, int ne, int * order, int * order_n,
                            int * egrp) {
    __shared__ int cnt[1024], off[1024];
    for (int e = threadIdx.x; e < ne; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += blockDim.x) { const int e = ids[p]; if (slot[e] >= 0) atomicAdd(&cnt[e], 1); }
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0, na = 0;
        for (int e = 0; e < ne; ++e) {
            off[e] = s;
            if (cnt[e] && egrp) { egrp[3 * na] = e; egrp[3 * na + 1] = s; egrp[3 * na + 2] = cnt[e]; ++na; }
            s += cnt[e];
        }
        order_n[0] = s;
        order_n[1] = na;
    }
    __syncthreads();
    // placement (order within an expert does not matter: every pair is computed independently)
    for (int p = threadIdx.x; p < P; p += blockDim.x) { const int e = ids[p]; if (slot[e] >= 0) order[atomicAdd(&off[e], 1)] = p; }
}

// ---- grouped expert GEMM (prefill) ----
__device__ __forceinline__ void mma16816_4(float * c, const unsigned * a, const unsigned * b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldsm_x4(unsigned * r, const void * smem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
constexpr int GE_M = 64, GE_N = 32, GE_K = 128, GE_LD = GE_K + 8;

// Block: one active expert (blockIdx.y < n_active), 64 weight rows (blockIdx.x), the expert's tokens in passes of 32.
// Weights dequantized from their GGUF blocks into fp16 shared tiles; tokens' fp16 rows gathered; m16n8k16 mma.
// GU: rows = 32 gate rows + the same 32 up rows; writes h16[pair][r] = silu(g) * u.  Down: y[pair][r] = w * acc.
template <GType T, bool GU>
__global__ void __launch_bounds__(256) k_moe_gemm(MoeDev m, const half * __restrict__ in, int in_stride, const int * __restrict__ in_row,
                                                  int in_div, const int * __restrict__ order, const int * __restrict__ order_n,
                                                  const int * __restrict__ egrp, const float * __restrict__ wts,
                                                  half * __restrict__ h16, float * __restrict__ y, int kdim, int rows_out) {
    if ((int) blockIdx.y >= order_n[1]) return;
    // the dequantized weight tile serves GE_NP token passes at a time (dequantization is the expensive part)
    constexpr int GE_NP = 2;
    __shared__ __align__(16) half As[GE_M * GE_LD];
    __shared__ __align__(16) half Bs[GE_NP][GE_N * GE_LD];
    __shared__ float Cs[GE_M][GE_N + 1];
    const int e = egrp[3 * blockIdx.y], start = egrp[3 * blockIdx.y + 1], cnt = egrp[3 * blockIdx.y + 2];
    const int slot = m.slot[e];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int wm = w & 3, wn = w >> 2;
    const int r_base = blockIdx.x * (GU ? GE_M / 2 : GE_M);
    const size_t mbytes = GU ? m.gate_bytes : m.down_bytes;
    const int mrows = GU ? m.ff : m.n;
    const size_t rb = mbytes / mrows;
    const uint8_t * W0 = (GU ? m.gate : m.down) + (size_t) slot * mbytes;
    const uint8_t * W1 = GU ? m.up + (size_t) slot * mbytes : nullptr;
    load_tables<T>();   // (synced by the first k-step's barrier)
    for (int t0 = 0; t0 < cnt; t0 += GE_N * GE_NP) {
        const int np = min(GE_NP, (cnt - t0 + GE_N - 1) / GE_N);
        float acc[GE_NP][2][4] = {};
        for (int k0 = 0; k0 < kdim; k0 += GE_K) {
            __syncthreads();
            // A: 64 rows x 128 cols = 1024 chunks of 8
            for (int ci = threadIdx.x; ci < GE_M * (GE_K / 8); ci += blockDim.x) {
                const int rr = ci / (GE_K / 8), cc = ci % (GE_K / 8);
                int row; const uint8_t * base;
                if (GU) { row = r_base + (rr & 31); base = rr < 32 ? W0 : W1; }
                else { row = r_base + rr; base = W0; }
                float v[8];
                if (row < mrows) deq8<T>(base + (size_t) row * rb, (k0 >> 3) + cc, v);
                else for (int i = 0; i < 8; ++i) v[i] = 0.0f;
                __half2 hv[4];
                for (int i = 0; i < 4; ++i) hv[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
                *(uint4 *) &As[rr * GE_LD + cc * 8] = *(const uint4 *) hv;
            }
            // B: the token rows of the np passes x 128 cols
            for (int ci = threadIdx.x; ci < np * GE_N * (GE_K / 8); ci += blockDim.x) {
                const int ps = ci / (GE_N * (GE_K / 8)), rem = ci % (GE_N * (GE_K / 8)), tr = rem / (GE_K / 8), cc = rem % (GE_K / 8);
                uint4 v = make_uint4(0, 0, 0, 0);
                const int tt = t0 + ps * GE_N + tr;
                if (tt < cnt) {
                    const int p = order[start + tt];
                    const int src = in_row ? in_row[p] : p / in_div;
                    v = *(const uint4 *) (in + (size_t) src * in_stride + k0 + cc * 8);
                }
                *(uint4 *) &Bs[ps][tr * GE_LD + cc * 8] = v;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < GE_K / 16; ++kk) {
                unsigned a[4];
                ldsm_x4(a, &As[(wm * 16 + (lane & 15)) * GE_LD + kk * 16 + (lane >> 4) * 8]);
#pragma unroll
                for (int ps = 0; ps < GE_NP; ++ps) {
                    if (ps >= np) break;
                    unsigned b[4];
                    ldsm_x4(b, &Bs[ps][(wn * 16 + (lane & 7) + ((lane >> 4) << 3)) * GE_LD + kk * 16 + ((lane >> 3) & 1) * 8]);
                    const unsigned b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
                    mma16816_4(acc[ps][0], a, b0);
                    mma16816_4(acc[ps][1], a, b1);
                }
            }
        }
        for (int ps = 0; ps < np; ++ps) {
            const int tb = t0 + ps * GE_N, nt = min(GE_N, cnt - tb);
            __syncthreads();
            // results to shared memory [row][token]
#pragma unroll
            for (int ni = 0; ni < 2; ++ni)
#pragma unroll
                for (int q = 0; q < 4; ++q) Cs[wm * 16 + gid + (q >> 1) * 8][wn * 16 + ni * 8 + 2 * tig + (q & 1)] = acc[ps][ni][q];
            __syncthreads();
            if (GU) {
                for (int i = threadIdx.x; i < 32 * nt; i += blockDim.x) {
                    const int rr = i % 32, tr = i / 32, row = r_base + rr;
                    if (row >= mrows) continue;
                    const int p = order[start + tb + tr];
                    h16[(size_t) p * rows_out + row] = __float2half(swiglu4(Cs[rr][tr], Cs[rr + 32][tr], m.clamp));
                }
            } else {
                for (int i = threadIdx.x; i < GE_M * nt; i += blockDim.x) {
                    const int rr = i % GE_M, tr = i / GE_M, row = r_base + rr;
                    if (row >= mrows) continue;
                    const int p = order[start + tb + tr];
                    y[(size_t) p * rows_out + row] = wts[p] * Cs[rr][tr];
                }
            }
        }
    }
}

// ---------------- QSA indexer ----------------
// rotate the first n_rot dims (NEOX pairs i, i + n_rot/2) of a 128-dim vector held in shared memory buf, thread i
__device__ __forceinline__ float rope_dim(const float * buf, int i, int pos, int n_rot, float base) {
    const int half_rot = n_rot / 2;
    if (i >= n_rot) return buf[i];
    const int fi = i < half_rot ? i : i - half_rot;
    const double theta = (double) pos * pow((double) base, -2.0 * fi / n_rot);
    double sn, cs;
    sincos(theta, &sn, &cs);
    return i < half_rot ? buf[i] * (float) cs - buf[i + half_rot] * (float) sn : buf[i - half_rot] * (float) sn + buf[i] * (float) cs;
}
// grid (nt, n_head + 1), 128 threads: query heads normed + rotated into qn; the last "head" stores the raw key (fp16)
__global__ void k_idx_prep(const float * __restrict__ qi, const float * __restrict__ kr, const float * __restrict__ qnorm,
                           float * __restrict__ qn, half * __restrict__ kraw, const int * pos_p, int n_head, int n_rot, float base, float eps) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int pos = *pos_p + t;
    if (h == n_head) { kraw[(size_t) pos * 128 + i] = __float2half(kr[(size_t) t * 128 + i]); return; }
    __shared__ float buf[128];
    float x = qi[((size_t) t * n_head + h) * 128 + i];
    const float ss = block_sum4(x * x);
    buf[i] = x * rsqrtf(ss / 128 + eps) * qnorm[i];
    __syncthreads();
    qn[((size_t) t * n_head + h) * 128 + i] = rope_dim(buf, i, pos, n_rot, base);
}
// grid (candidate blocks), 128 threads: blocks of 4 cells completed by tokens pos..pos+nt-1: mean raw key -> rms norm ->
// rope at the block's first position -> pool[b]
__global__ void k_idx_pool(const half * __restrict__ kraw, half * __restrict__ pool, const float * __restrict__ knorm, const int * pos_p,
                           int nt, int n_rot, float base, float eps) {
    const int pos = *pos_p;
    const int b = pos / 4 + blockIdx.x, last = 4 * b + 3, i = threadIdx.x;
    if (last < pos || last > pos + nt - 1) return;
    __shared__ float buf[128];
    float x = 0.0f;
    for (int j = 0; j < 4; ++j) x += __half2float(kraw[(size_t) (4 * b + j) * 128 + i]);
    x *= 0.25f;
    const float ss = block_sum4(x * x);
    buf[i] = x * rsqrtf(ss / 128 + eps) * knorm[i];
    __syncthreads();
    pool[(size_t) b * 128 + i] = __float2half(rope_dim(buf, i, 4 * b, n_rot, base));
}
// block per token (1024 threads): visible complete pools np = (p+1)/4; np <= top: all cells 0..p. Otherwise the top
// `top` pools by sum_h relu(q_h . pool) / sqrt(128) (radix select on the non-negative score bits, ties to the lower
// pool index), as ascending cell lists, plus the incomplete tail block
__global__ void __launch_bounds__(1024) k_idx_select(const float * __restrict__ qn, const half * __restrict__ pool, const int * pos_p, int t_off,
                                                     int n_head, int top, float * __restrict__ scores, int score_stride,
                                                     int * __restrict__ list, int list_stride, int * __restrict__ list_n) {
    const int tl = blockIdx.x, t = t_off + tl;
    const int p = *pos_p + t, np = (p + 1) / 4;
    int * lt = list + (size_t) t * list_stride;
    if (np <= top) {
        for (int i = threadIdx.x; i <= p; i += blockDim.x) lt[i] = i;
        if (threadIdx.x == 0) list_n[t] = p + 1;
        return;
    }
    __shared__ float qs[4 * 128];
    for (int i = threadIdx.x; i < n_head * 128; i += blockDim.x) qs[i] = qn[(size_t) t * n_head * 128 + i];
    __syncthreads();
    float * sc = scores + (size_t) tl * score_stride;
    for (int b = threadIdx.x; b < np; b += blockDim.x) {
        const half * pk = pool + (size_t) b * 128;
        float acc[4] = {0, 0, 0, 0};
        for (int i = 0; i < 128; i += 8) {
            const uint4 v = *(const uint4 *) (pk + i);
            const __half2 * h2 = (const __half2 *) &v;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(h2[j]);
                for (int h = 0; h < n_head; ++h) acc[h] += f.x * qs[h * 128 + i + 2 * j] + f.y * qs[h * 128 + i + 2 * j + 1];
            }
        }
        float s = 0.0f;
        for (int h = 0; h < n_head; ++h) s += fmaxf(acc[h], 0.0f);
        sc[b] = s * rsqrtf(128.0f);
    }
    __syncthreads();
    // radix select of the top-th largest key (scores >= 0: float bits are monotonic)
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, need;
    if (threadIdx.x == 0) { prefix = 0; need = top; }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned mask_hi = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int b = threadIdx.x; b < np; b += blockDim.x) {
            const unsigned key = __float_as_uint(sc[b]);
            if ((key & mask_hi) == (prefix & mask_hi)) atomicAdd(&hist[(key >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned acc = 0;
            for (int d = 255; d >= 0; --d) {
                if (acc + hist[d] >= need) { prefix |= (unsigned) d << shift; need -= acc; break; }
                acc += hist[d];
            }
        }
        __syncthreads();
    }
    // prefix = threshold key; take every key > threshold and the first `need` keys == threshold (pool order)
    const unsigned thr = prefix;
    __shared__ int base_cnt, eq_left;
    __shared__ int warp_sums[32];
    if (threadIdx.x == 0) { base_cnt = 0; eq_left = (int) need; }
    __syncthreads();
    for (int b0 = 0; b0 < np; b0 += blockDim.x) {
        const int b = b0 + threadIdx.x;
        const unsigned key = b < np ? __float_as_uint(sc[b]) : 0u;
        const int gt = b < np && key > thr, eq = b < np && key == thr;
        // ordered rank of eq within this chunk
        const unsigned m_eq = __ballot_sync(0xffffffff, eq), m_gt = __ballot_sync(0xffffffff, gt);
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        if (lane == 0) warp_sums[wid] = __popc(m_eq) | (__popc(m_gt) << 16);
        __syncthreads();
        int eq_before = 0, gt_before = 0;
        for (int w = 0; w < wid; ++w) { eq_before += warp_sums[w] & 0xffff; gt_before += warp_sums[w] >> 16; }
        eq_before += __popc(m_eq & ((1u << lane) - 1));
        gt_before += __popc(m_gt & ((1u << lane) - 1));
        int tot_eq = 0, tot_gt = 0;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) { tot_eq += warp_sums[w] & 0xffff; tot_gt += warp_sums[w] >> 16; }
        const int eq_take = min(tot_eq, eq_left);
        const bool sel = gt || (eq && eq_before < eq_left);
        // selected pools of this chunk in pool order: rank = (selected before me in the chunk)
        const int sel_before = gt_before + min(eq_before, eq_left);   // gt and eq interleave: count both kinds before me
        (void) sel_before;
        __syncthreads();
        // exact ordered rank: prefix count of sel over the chunk
        const unsigned m_sel = __ballot_sync(0xffffffff, sel);
        if (lane == 0) warp_sums[wid] = __popc(m_sel);
        __syncthreads();
        int r = __popc(m_sel & ((1u << lane) - 1));
        for (int w = 0; w < wid; ++w) r += warp_sums[w];
        int chunk_sel = 0;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) chunk_sel += warp_sums[w];
        if (sel) {
            const int o = (base_cnt + r) * 4;
            lt[o] = 4 * b; lt[o + 1] = 4 * b + 1; lt[o + 2] = 4 * b + 2; lt[o + 3] = 4 * b + 3;
        }
        __syncthreads();
        if (threadIdx.x == 0) { base_cnt += chunk_sel; eq_left -= eq_take; }
        __syncthreads();
        (void) tot_gt;
    }
    // tail: cells after the last complete block
    const int tail0 = 4 * np;
    for (int c = tail0 + threadIdx.x; c <= p; c += blockDim.x) lt[base_cnt * 4 + (c - tail0)] = c;
    if (threadIdx.x == 0) list_n[t] = base_cnt * 4 + (p - tail0 + 1);
}

// ---------------- MTP prelude ----------------
// grid (hc, nt): ecat[t*hc + s] = [rms(x[t]) * enorm | norm(H[t][s]) * hnorm[s]] (2n); H row stride hs (0: one row
// for every token); whole: the hidden norm runs over all hc streams together instead of per stream
__global__ void k_mtp_prep(const float * __restrict__ x, const float * __restrict__ enorm, const float * __restrict__ H, int hs,
                           const float * __restrict__ hnorm, float eps, int n, int hc, int whole, float * __restrict__ ecat) {
    const int s = blockIdx.x, t = blockIdx.y;
    const float * xr = x + (size_t) t * n;
    const float * hr = H + (size_t) t * hs;
    float a = 0.0f, b = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) a += xr[i] * xr[i];
    if (whole) { for (int i = threadIdx.x; i < hc * n; i += blockDim.x) b += hr[i] * hr[i]; }
    else for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = hr[(size_t) s * n + i]; b += v * v; }
    a = block_sum4(a);
    b = block_sum4(b);
    const float ia = rsqrtf(a / n + eps), ib = rsqrtf(b / (whole ? hc * n : n) + eps);
    float * o = ecat + ((size_t) t * hc + s) * 2 * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        o[i] = xr[i] * ia * enorm[i];
        o[n + i] = hr[(size_t) s * n + i] * ib * hnorm[(size_t) s * n + i];
    }
}

// ---------------- PLE ----------------
// grid (hc, nt): per stream sums of key^2, res^2, key*wk*res*wq; stream 0 also sum value^2
__global__ void k_ple_stats(const float * __restrict__ res, const float * __restrict__ key, const float * __restrict__ value,
                            const float * __restrict__ wk, const float * __restrict__ wq, int n, int hc, float * sc) {
    const int s = blockIdx.x, t = blockIdx.y;
    const float * kr = key + ((size_t) t * hc + s) * n;
    const float * rr = res + ((size_t) t * hc + s) * n;
    float a = 0, b = 0, c = 0, d = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        a += kr[i] * kr[i]; b += rr[i] * rr[i];
        c += kr[i] * wk[(size_t) s * n + i] * rr[i] * wq[(size_t) s * n + i];
        if (s == 0) { const float v = value[(size_t) t * n + i]; d += v * v; }
    }
    a = block_sum4(a); b = block_sum4(b); c = block_sum4(c); d = block_sum4(d);
    if (threadIdx.x == 0) {
        float * o = sc + ((size_t) t * hc + s) * 4;
        o[0] = a; o[1] = b; o[2] = c; o[3] = d;
    }
}
// thread per channel c in [0, hc*n): tokens in order
__global__ void k_ple_apply(float * res, const float * __restrict__ value, const float * __restrict__ wconv_norm,
                            const float * __restrict__ conv_w, float * conv_state, float * conv_snap, const float * __restrict__ sc,
                            int n, int hc, int K, int dil, float eps, int nt) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = hc * n;
    if (c >= C) return;
    const int s = c / n, e = c % n;
    const int H = (K - 1) * dil;
    float hist[16];
    for (int j = 0; j < H; ++j) hist[j] = conv_state[(size_t) j * C + c];
    for (int t = 0; t < nt; ++t) {
        const float * st = sc + ((size_t) t * hc + s) * 4;
        const float ik = rsqrtf(st[0] / n + eps), iq = rsqrtf(st[1] / n + eps);
        const float dot = st[2] * ik * iq / sqrtf((float) n);
        const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
        const float gate = sigm(dot > 0 ? mag : (dot < 0 ? -mag : 0.0f));
        const float mv = sc[((size_t) t * hc) * 4 + 3] / n;   // mean value^2 (stream 0's slot)
        const float g = value[(size_t) t * n + e] * gate;
        const float xn = g * rsqrtf(gate * gate * mv + eps) * wconv_norm[c];
        // out = sum_k w[k] * x(t - (K-1-k)*dil); x(t) = xn, older ones from the history (oldest first)
        float acc = conv_w[(size_t) c * K + (K - 1)] * xn;
        for (int kk = 0; kk < K - 1; ++kk) acc += conv_w[(size_t) c * K + kk] * hist[H - (K - 1 - kk) * dil];
        for (int j = 0; j < H - 1; ++j) hist[j] = hist[j + 1];
        hist[H - 1] = xn;
        float * rp = res + (size_t) t * C + c;
        *rp = *rp + g + silu4(acc);
        if (conv_snap && t < nt - 1)
            for (int j = 0; j < H; ++j) conv_snap[(size_t) t * H * C + (size_t) j * C + c] = hist[j];
    }
    for (int j = 0; j < H; ++j) conv_state[(size_t) j * C + c] = hist[j];
}

} // namespace

void hc_norm(const float * res, const float * w, float * xn, int n, int hc, float eps, int nt, cudaStream_t s,
             const float * inj_w, float * injp, float * inj) {
    if (inj_w && hc > 4) throw std::runtime_error("hc_norm: inject fusion supports hc <= 4");
    static const bool old_norm = getenv("HYPER4_OLDNORM") != nullptr;
    if (n % 4 == 0 && n / 4 <= 1024 && !old_norm)
        k_hc_norm_v<<<dim3(hc, nt), (n / 4 + 31) / 32 * 32, 0, s>>>(res, w, xn, n, hc, eps, inj_w, injp);
    else k_hc_norm<<<dim3(hc, nt), 256, 0, s>>>(res, w, xn, n, hc, eps, inj_w, injp);
    if (inj_w) k_hc_inj_sum<<<nt, 32, 0, s>>>(injp, inj, hc);
}
void silu_scale(float * x, int n, float scale, int nt, int stride, cudaStream_t s) {
    k_silu_scale<<<dim3((n + 255) / 256, nt), 256, 0, s>>>(x, n, scale, stride);
}
void hc_mixed(const float * xn, const float * gate, float * mixed, int n, int hc, int nt, cudaStream_t s) {
    k_hc_mixed<<<dim3((n + 255) / 256, nt), 256, 0, s>>>(xn, gate, mixed, n, hc);
}
void hc_combine(float * res, const float * bo, const float * inject, int inject_stride, int n, int hc, int nt, cudaStream_t s) {
    k_hc_combine<<<dim3((hc * n + 255) / 256, nt), 256, 0, s>>>(res, bo, inject, inject_stride, n, hc);
}
void hc_init(float * res, const float * x, int n, int hc, int nt, cudaStream_t s) {
    k_hc_init<<<dim3((hc * n + 255) / 256, nt), 256, 0, s>>>(res, x, n, hc);
}
void gemv_f32(const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nt, cudaStream_t s) {
    if (rows < 64 && (k & 3) == 0) k_gemv_f32_br<<<dim3(rows, nt), 256, 0, s>>>(W, k, x, xs, y, ys);
    else k_gemv_f32<<<dim3((rows + 7) / 8, nt), 256, 0, s>>>(W, rows, k, x, xs, y, ys);
}
void gated_norm_sigmoid(float * o, int o_stride, const float * z, int z_stride, const float * w, int n_heads, int dh, float eps,
                        int nt, cudaStream_t s) {
    k_gated_norm_sig<<<dim3(n_heads, nt), dh, 0, s>>>(o, o_stride, z, z_stride, w, dh, eps);
}
void moe_route(const float * logits, int ls, int n_expert, int k, int * ids, float * wts, float * sg, int nt, cudaStream_t s) {
    if (n_expert > 1024 || k > MOE_MAX_USED) throw std::runtime_error("moe_route: too many experts");
    if (k * 9 > 8 * MOE_MAX_USED) throw std::runtime_error("moe_route: k too large");
    static const bool old_route = getenv("HYPER4_OLDROUTE") != nullptr;
    k_moe_route<<<nt, 256, 0, s>>>(logits, ls, n_expert, k, ids, wts, sg, !old_route);
}

#define MOE_TYPE_SWITCH(T, CALL)                                                  \
    switch (T) {                                                                  \
        case GType::Q4_K: { constexpr GType TT = GType::Q4_K; CALL; } break;       \
        case GType::Q5_K: { constexpr GType TT = GType::Q5_K; CALL; } break;       \
        case GType::Q5_1: { constexpr GType TT = GType::Q5_1; CALL; } break;       \
        case GType::Q5_0: { constexpr GType TT = GType::Q5_0; CALL; } break;       \
        case GType::Q8_0: { constexpr GType TT = GType::Q8_0; CALL; } break;       \
        case GType::Q6_K: { constexpr GType TT = GType::Q6_K; CALL; } break;       \
        case GType::IQ4_XS: { constexpr GType TT = GType::IQ4_XS; CALL; } break;   \
        case GType::IQ3_S: { constexpr GType TT = GType::IQ3_S; CALL; } break;     \
        default: throw std::runtime_error(std::string("moe: unsupported expert type ") + gtype_name(T)); \
    }

void moe_gate_up(const MoeDev & m, const float * x, int xs, const int * ids, int k, float * h, int nt, cudaStream_t s,
                 const int * order, const int * order_n) {
    const int kdim = (int) (m.gate_bytes / m.ff / gtype_block_bytes(m.tg) * gtype_block_elems(m.tg));
    MOE_TYPE_SWITCH(m.tg, (k_moe_gate_up<TT><<<dim3(nt * k, (m.ff + MOE_ROWS - 1) / MOE_ROWS), 256, kdim * sizeof(float), s>>>(m, x, xs, ids, k, h, kdim, order, order_n)));
}
void moe_down(const MoeDev & m, const float * h, const int * ids, const float * wts, int k, float * y, int nt, cudaStream_t s,
              const int * order, const int * order_n) {
    MOE_TYPE_SWITCH(m.td, (k_moe_down<TT><<<dim3(nt * k, (m.n + MOE_ROWS - 1) / MOE_ROWS), 256, m.ff * sizeof(float), s>>>(m, h, ids, wts, k, y, order, order_n)));
}
void moe_zc(const MoeZC & z, const float * x, int xs, const int * ids, const float * wts, int k, float * h, float * y, int nt,
            cudaStream_t s) {
    const int len = z.f1 - z.f0;
    if (len <= 0) return;
    if (z.f0 % 256 || len % 256) throw std::runtime_error("moe_zc: slice must be whole 256-blocks");
    const int kdim = z.n;
    const size_t rbg = z.gate_bytes / z.ff, rbd = z.down_bytes / z.n;
    if (rbg % 8 || rbd % 8 || (rbd / (z.ff / 256)) % 8) throw std::runtime_error("moe_zc: rows not 8-byte aligned");
    const size_t sm1 = (size_t) kdim * 4 + 8 * 2 * rbg, sm2 = (size_t) len * 4 + 8 * (rbd / (z.ff / 256) * (len / 256));
    MOE_TYPE_SWITCH(z.tg, (k_moe_zc_gate_up<TT><<<dim3(nt * k, (len + 31) / 32), 256, sm1, s>>>(z, x, xs, ids, k, h, kdim)));
    MOE_TYPE_SWITCH(z.td, (k_moe_zc_down<TT><<<dim3(nt * k, (z.n + 31) / 32), 256, sm2, s>>>(z, h, ids, wts, k, y)));
}
// test hook: one row of n elements dequantized by the expert kernels' deq8 (thread per 8 elements)
template <GType T> __global__ void k_deq_row(const uint8_t * row, int n, float * out) {
    load_tables<T>();
    __syncthreads();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c * 8 < n) deq8<T>(row, c, out + c * 8);
}
void deq_row_test(GType t, const uint8_t * row, int n, float * out) {
    MOE_TYPE_SWITCH(t, (k_deq_row<TT><<<(n / 8 + 255) / 256, 256>>>(row, n, out)));
}
void moe_zc_init() {
    for (auto f : {(const void *) k_moe_zc_gate_up<GType::IQ3_S>, (const void *) k_moe_zc_gate_up<GType::IQ4_XS>,
                   (const void *) k_moe_zc_gate_up<GType::Q4_K>, (const void *) k_moe_zc_gate_up<GType::Q5_K>,
                   (const void *) k_moe_zc_gate_up<GType::Q6_K>, (const void *) k_moe_zc_gate_up<GType::Q8_0>, (const void *) k_moe_zc_gate_up<GType::Q5_1>,
                   (const void *) k_moe_zc_gate_up<GType::Q5_0>})
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
}
void moe_order(const MoeDev & m, const int * ids, int n_pairs, int n_expert, int * order, int * order_n, cudaStream_t s, int * egrp) {
    if (n_expert > 1024) throw std::runtime_error("moe_order: too many experts");
    k_moe_order<<<1, 1024, 0, s>>>(m.slot, ids, n_pairs, n_expert, order, order_n, egrp);
}
void moe_gemm_gate_up(const MoeDev & m, const half * x16, int xs, int k, const int * order, const int * order_n, const int * egrp,
                      int max_active, half * h16, cudaStream_t s) {
    const int kdim = (int) (m.gate_bytes / m.ff / gtype_block_bytes(m.tg) * gtype_block_elems(m.tg));
    if (kdim % GE_K) throw std::runtime_error("moe_gemm: k must be a multiple of 128");
    MOE_TYPE_SWITCH(m.tg, (k_moe_gemm<TT, true><<<dim3((m.ff + 31) / 32, max_active), 256, 0, s>>>(m, x16, xs, nullptr, k, order, order_n, egrp,
                                                                                                   nullptr, h16, nullptr, kdim, m.ff)));
}
void moe_gemm_down(const MoeDev & m, const half * h16, const int * order, const int * order_n, const int * egrp, int max_active,
                   const float * wts, float * y, cudaStream_t s) {
    if (m.ff % GE_K) throw std::runtime_error("moe_gemm: ff must be a multiple of 128");
    MOE_TYPE_SWITCH(m.td, (k_moe_gemm<TT, false><<<dim3((m.n + GE_M - 1) / GE_M, max_active), 256, 0, s>>>(m, h16, m.ff, nullptr, 1, order, order_n,
                                                                                                          egrp, wts, nullptr, y, m.ff, m.n)));
}
void moe_reduce(const float * shexp, const float * sg, const float * y, int k, float * out, int n, int nt,
                const int * ids, const int * owner, int g, int cpu_owner, const volatile unsigned * cpu_flag, const float * cpu_y,
                const int * counter, unsigned seq_tag, cudaStream_t s, const float * yzc) {
    k_moe_reduce<<<dim3((n + 255) / 256, nt), 256, 0, s>>>(shexp, sg, y, k, out, n, ids, owner, g, cpu_owner, cpu_flag, cpu_y, counter, seq_tag,
                                                            yzc);
}
void moe_publish(volatile unsigned * seq, int * ntp, int * ids_dst, float * wts_dst, float * x_dst, const float * x, int xs, int n,
                 const int * ids, const float * wts, int k, int nt, const int * counter, unsigned seq_tag, cudaStream_t s) {
    if (nt <= MAX_NT && n % 2 == 0 && xs % 2 == 0) {
        k_moe_publish1<<<1, 1024, 0, s>>>(seq, ntp, ids_dst, wts_dst, x_dst, x, xs, n, ids, wts, k, nt, counter, seq_tag);
        return;
    }
    const size_t tot = (size_t) nt * n;
    k_moe_publish<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(ntp, ids_dst, wts_dst, x_dst, x, xs, n, ids, wts, k, nt);
    k_moe_seq<<<1, 1, 0, s>>>(seq, counter, seq_tag);
}
void idx_prep(const float * qi, const float * kr, const float * qnorm, float * qn, half * kraw, const int * pos, int n_head, int n_rot,
              float base, float eps, int nt, cudaStream_t s) {
    k_idx_prep<<<dim3(nt, n_head + 1), 128, 0, s>>>(qi, kr, qnorm, qn, kraw, pos, n_head, n_rot, base, eps);
}
void idx_pool(const half * kraw, half * pool, const float * knorm, const int * pos, int nt, int n_rot, float base, float eps, cudaStream_t s) {
    k_idx_pool<<<nt / 4 + 2, 128, 0, s>>>(kraw, pool, knorm, pos, nt, n_rot, base, eps);
}
void idx_select(const float * qn, const half * pool, const int * pos, int nt, int n_head, int top, float * scores, int score_stride,
                int score_rows, int * list, int list_stride, int * list_n, cudaStream_t s) {
    for (int t0 = 0; t0 < nt; t0 += score_rows)
        k_idx_select<<<std::min(score_rows, nt - t0), 1024, 0, s>>>(qn, pool, pos, t0, n_head, top, scores, score_stride, list, list_stride, list_n);
}

void mtp_prep(const float * x, const float * enorm, const float * H, int hs, const float * hnorm, float eps, int n, int hc, bool whole,
              float * ecat, int nt, cudaStream_t s) {
    k_mtp_prep<<<dim3(hc, nt), 256, 0, s>>>(x, enorm, H, hs, hnorm, eps, n, hc, whole ? 1 : 0, ecat);
}

void ple_apply(float * res, const float * key, const float * value, const float * wk, const float * wq, const float * wconv_norm,
               const float * conv_w, float * conv_state, float * conv_snap, int n, int hc, int K, int dil, float eps, int nt,
               float * scratch, cudaStream_t s) {
    if ((K - 1) * dil > 16) throw std::runtime_error("ple: conv history too long");
    k_ple_stats<<<dim3(hc, nt), 256, 0, s>>>(res, key, value, wk, wq, n, hc, scratch);
    k_ple_apply<<<(hc * n + 255) / 256, 256, 0, s>>>(res, value, wconv_norm, conv_w, conv_state, conv_snap, scratch, n, hc, K, dil, eps, nt);
}

} // namespace hyper
