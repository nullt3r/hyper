#include "kernels5.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace hyper {

namespace {

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ float bsum(float v) {
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    v = lane < nw ? red[lane] : 0.0f;
    return wsum(v);
}
__device__ __forceinline__ float sigm5(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------- mHC ----------------
// block per token, 1024 threads; n <= 4096
__global__ void __launch_bounds__(1024) k_mhc_pre(const float * __restrict__ res, const float * __restrict__ mixraw, int mix_stride,
                                                  const float * __restrict__ scale, const float * __restrict__ base,
                                                  const float * __restrict__ norm_w, float rms_eps, float hc_eps, int iters, int n,
                                                  float * __restrict__ hcw, float * __restrict__ xn) {
    const int t = blockIdx.x;
    const float * r = res + (size_t) t * MHC * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < MHC * n; i += blockDim.x) ss += r[i] * r[i];
    ss = bsum(ss);
    __shared__ float pre[MHC];
    if (threadIdx.x == 0) {
        const float inv = rsqrtf(ss / (MHC * n) + rms_eps);
        float mx[24];
        for (int i = 0; i < 24; ++i) mx[i] = mixraw[(size_t) t * mix_stride + i] * inv;
        for (int s = 0; s < MHC; ++s) pre[s] = sigm5(mx[s] * scale[0] + base[s]) + hc_eps;
        float * hw = hcw + (size_t) t * MHC_W;
        for (int d = 0; d < MHC; ++d) hw[d] = 2.0f * sigm5(mx[4 + d] * scale[1] + base[4 + d]);
        float c[16];
        for (int i = 0; i < 16; ++i) c[i] = mx[8 + i] * scale[2] + base[8 + i];
        // softmax over dst for every src, + eps
        for (int src = 0; src < MHC; ++src) {
            float m = -FLT_MAX;
            for (int d = 0; d < MHC; ++d) m = fmaxf(m, c[d + 4 * src]);
            float z = 0.0f;
            for (int d = 0; d < MHC; ++d) { c[d + 4 * src] = expf(c[d + 4 * src] - m); z += c[d + 4 * src]; }
            for (int d = 0; d < MHC; ++d) c[d + 4 * src] = c[d + 4 * src] / z + hc_eps;
        }
        auto norm_cols = [&]() {   // per dst: divide by the sum over src
            for (int d = 0; d < MHC; ++d) {
                float z = 0.0f;
                for (int src = 0; src < MHC; ++src) z += c[d + 4 * src];
                z += hc_eps;
                for (int src = 0; src < MHC; ++src) c[d + 4 * src] /= z;
            }
        };
        auto norm_rows = [&]() {   // per src: divide by the sum over dst
            for (int src = 0; src < MHC; ++src) {
                float z = 0.0f;
                for (int d = 0; d < MHC; ++d) z += c[d + 4 * src];
                z += hc_eps;
                for (int d = 0; d < MHC; ++d) c[d + 4 * src] /= z;
            }
        };
        norm_cols();
        for (int it = 1; it < iters; ++it) { norm_rows(); norm_cols(); }
        for (int i = 0; i < 16; ++i) hw[4 + i] = c[i];
    }
    __syncthreads();
    __shared__ float xs[4096];
    float s2 = 0.0f;
    for (int e = threadIdx.x; e < n; e += blockDim.x) {
        const float x = pre[0] * r[e] + pre[1] * r[n + e] + pre[2] * r[2 * n + e] + pre[3] * r[3 * n + e];
        xs[e] = x;
        s2 += x * x;
    }
    s2 = bsum(s2);
    const float inv2 = rsqrtf(s2 / n + rms_eps);
    for (int e = threadIdx.x; e < n; e += blockDim.x) xn[(size_t) t * n + e] = xs[e] * inv2 * norm_w[e];
}

// mixes from raw GGUF Q8_0 rows fn [24][4n] (34-byte blocks): grid (4n / 256, nt), 256 threads; partial dot products of a
// 256-column slice and the slice's sum of squares -> part[t][slice][25]
__global__ void __launch_bounds__(256) k_mhc_mix(const float * __restrict__ res, const uint8_t * __restrict__ fn, int n4,
                                                 float * __restrict__ part) {
    const int t = blockIdx.y, sl = blockIdx.x, c = sl * 256 + threadIdx.x;
    const float x = res[(size_t) t * n4 + c];
    const size_t rb = (size_t) n4 / 32 * 34;
    const int blk = c >> 5, j = c & 31;
    float acc[25];
#pragma unroll
    for (int r = 0; r < 24; ++r) {
        const uint8_t * b = fn + (size_t) r * rb + (size_t) blk * 34;
        __half d; memcpy(&d, b, 2);
        acc[r] = __half2float(d) * (float) (int8_t) b[2 + j] * x;
    }
    acc[24] = x * x;
    __shared__ float red[8][25];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < 25; ++r) {
        const float v = wsum(acc[r]);
        if (lane == 0) red[w][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < 25) {
        float v = 0.0f;
        for (int ww = 0; ww < 8; ++ww) v += red[ww][threadIdx.x];
        part[((size_t) t * gridDim.x + sl) * 25 + threadIdx.x] = v;
    }
}
// block per token, 1024 threads: mixes from the partials, Sinkhorn on 16 lanes of warp 0, then as k_mhc_pre
__global__ void __launch_bounds__(1024) k_mhc_pre2(const float * __restrict__ res, const float * __restrict__ part, int nsl,
                                                   const float * __restrict__ scale, const float * __restrict__ base,
                                                   const float * __restrict__ norm_w, float rms_eps, float hc_eps, int iters, int n,
                                                   float * __restrict__ hcw, float * __restrict__ xn) {
    const int t = blockIdx.x, tid = threadIdx.x;
    const float * r = res + (size_t) t * MHC * n;
    __shared__ float mx[25], pre[MHC];
    if (tid < 25) {
        float v = 0.0f;
        for (int i = 0; i < nsl; ++i) v += part[((size_t) t * nsl + i) * 25 + tid];
        mx[tid] = v;
    }
    __syncthreads();
    if (tid < 32) {
        const float inv = rsqrtf(mx[24] / (MHC * n) + rms_eps);
        float * hw = hcw + (size_t) t * MHC_W;
        if (tid < MHC) pre[tid] = sigm5(mx[tid] * inv * scale[0] + base[tid]) + hc_eps;
        else if (tid < 2 * MHC) hw[tid - MHC] = 2.0f * sigm5(mx[tid] * inv * scale[1] + base[tid]);
        // lane l < 16 holds c[d + 4 src] with d = l & 3, src = l >> 2
        const int l = tid & 15, d = l & 3;
        float c = tid < 16 ? mx[8 + l] * inv * scale[2] + base[8 + l] : -FLT_MAX;
        // softmax over dst (lanes of one src: groups of 4 consecutive lanes)
        float m = c;
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1)); m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
        float e = tid < 16 ? expf(c - m) : 0.0f, z = e;
        z += __shfl_xor_sync(0xffffffff, z, 1); z += __shfl_xor_sync(0xffffffff, z, 2);
        c = e / z + hc_eps;
        auto sum_src = [&](float v) {   // over lanes d, d+4, d+8, d+12
            v += __shfl_xor_sync(0xffffffff, v, 4); v += __shfl_xor_sync(0xffffffff, v, 8); return v;
        };
        auto sum_dst = [&](float v) { v += __shfl_xor_sync(0xffffffff, v, 1); v += __shfl_xor_sync(0xffffffff, v, 2); return v; };
        c = c / (sum_src(tid < 16 ? c : 0.0f) + hc_eps);
        for (int it = 1; it < iters; ++it) {
            c = c / (sum_dst(tid < 16 ? c : 0.0f) + hc_eps);
            c = c / (sum_src(tid < 16 ? c : 0.0f) + hc_eps);
        }
        (void) d;
        if (tid < 16) hw[4 + l] = c;
    }
    __syncthreads();
    __shared__ float xs[4096];
    float s2 = 0.0f;
    for (int e = tid; e < n; e += blockDim.x) {
        const float x = pre[0] * r[e] + pre[1] * r[n + e] + pre[2] * r[2 * n + e] + pre[3] * r[3 * n + e];
        xs[e] = x;
        s2 += x * x;
    }
    s2 = bsum(s2);
    const float inv2 = rsqrtf(s2 / n + rms_eps);
    for (int e = tid; e < n; e += blockDim.x) xn[(size_t) t * n + e] = xs[e] * inv2 * norm_w[e];
}

__global__ void k_mhc_post(float * res, const float * __restrict__ out, const float * __restrict__ hcw, int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= n) return;
    float * r = res + (size_t) t * MHC * n;
    const float * hw = hcw + (size_t) t * MHC_W;
    const float o = out[(size_t) t * n + e];
    const float r0 = r[e], r1 = r[n + e], r2 = r[2 * n + e], r3 = r[3 * n + e];
#pragma unroll
    for (int d = 0; d < MHC; ++d)
        r[(size_t) d * n + e] = o * hw[d] + hw[4 + d] * r0 + hw[4 + d + 4] * r1 + hw[4 + d + 8] * r2 + hw[4 + d + 12] * r3;
}
__global__ void k_mhc_init(float * res, const float * __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i < MHC * n) res[(size_t) t * MHC * n + i] = x[(size_t) t * n + i % n];
}
__global__ void k_mhc_head(const float * __restrict__ res, const float * __restrict__ w, float eps, int n, float * __restrict__ xn) {
    const int t = blockIdx.x;
    const float * r = res + (size_t) t * MHC * n;
    __shared__ float xs[4096];
    float ss = 0.0f;
    for (int e = threadIdx.x; e < n; e += blockDim.x) {
        const float x = (r[e] + r[n + e] + r[2 * n + e] + r[3 * n + e]) * 0.25f;
        xs[e] = x;
        ss += x * x;
    }
    ss = bsum(ss);
    const float inv = rsqrtf(ss / n + eps);
    for (int e = threadIdx.x; e < n; e += blockDim.x) xn[(size_t) t * n + e] = xs[e] * inv * w[e];
}

// ---------------- KDA ----------------
// warp per value column j of head h; S[:, j] (128 key dims) in registers, 4 per lane. grid (n_head, 16), 8 warps
__global__ void __launch_bounds__(256) k_kda_step(const float * __restrict__ in, int stride, int q_off, int k_off, int v_off, int fb_off,
                                                  int b_off, float * __restrict__ state, float * __restrict__ snap, float * __restrict__ o,
                                                  int o_stride, const float * __restrict__ dt_bias, const float * __restrict__ A, float lb,
                                                  int n_head, float eps, int nt) {
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int j = blockIdx.y * 8 + w;
    float * S = state + (size_t) h * 128 * 128;
    float s[4];
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) s[ii] = S[(size_t) (lane * 4 + ii) * 128 + j];
    const float4 dtb = *(const float4 *) (dt_bias + (size_t) h * 128 + lane * 4);
    const float dt[4] = {dtb.x, dtb.y, dtb.z, dtb.w};
    const float a = A[h];
    for (int t = 0; t < nt; ++t) {
        const float * row = in + (size_t) t * stride;
        const float4 q4 = *(const float4 *) (row + q_off + (size_t) h * 128 + lane * 4);
        const float4 k4 = *(const float4 *) (row + k_off + (size_t) h * 128 + lane * 4);
        const float4 f4 = *(const float4 *) (row + fb_off + (size_t) h * 128 + lane * 4);
        const float q[4] = {q4.x, q4.y, q4.z, q4.w}, k[4] = {k4.x, k4.y, k4.z, k4.w}, fb[4] = {f4.x, f4.y, f4.z, f4.w};
        float dec[4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) dec[ii] = expf(lb * sigm5(-a * (fb[ii] + dt[ii])));
        float qq = 0.0f, kk = 0.0f, part = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) { qq += q[ii] * q[ii]; kk += k[ii] * k[ii]; s[ii] *= dec[ii]; part += s[ii] * k[ii]; }
        qq = wsum(qq); kk = wsum(kk); part = wsum(part);
        const float qs = rsqrtf(qq + eps), ks = rsqrtf(kk + eps);
        const float beta = sigm5(row[b_off + h]);
        const float delta = (row[v_off + (size_t) h * 128 + j] - part * ks) * beta;
        float out = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            s[ii] += k[ii] * ks * delta;
            out += s[ii] * q[ii];
        }
        out = wsum(out);
        if (lane == 0) o[(size_t) t * o_stride + (size_t) h * 128 + j] = out * qs * 0.08838834764831845f;   // 1/sqrt(128)
        if (snap && t < nt - 1) {
            float * Sn = snap + (size_t) t * n_head * 128 * 128 + (size_t) h * 128 * 128;
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) Sn[(size_t) (lane * 4 + ii) * 128 + j] = s[ii];
        }
    }
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) S[(size_t) (lane * 4 + ii) * 128 + j] = s[ii];
}

// ---------------- MLA ----------------
// warp per output row (h, r) for all nt tokens; fp16 weights
__global__ void k_head_gemv(const half * __restrict__ W, int H, int R, int C, const float * __restrict__ x, int xs, float * __restrict__ y,
                            int ys, int nt) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= H * R) return;
    const int h = row / R, r = row % R;
    const half * wr = W + (size_t) row * C;
    float acc[MAX_NT] = {};
    for (int c = lane * 8; c < C; c += 256) {
        const uint4 wv = *(const uint4 *) (wr + c);
        const __half2 * w2 = (const __half2 *) &wv;
        float wf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(w2[i]); wf[2 * i] = f.x; wf[2 * i + 1] = f.y; }
        for (int t = 0; t < nt; ++t) {
            const float * xr = x + (size_t) t * xs + (size_t) h * C + c;
            const float4 a = *(const float4 *) xr, b = *(const float4 *) (xr + 4);
            acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
        }
    }
    for (int t = 0; t < nt; ++t) {
        const float v = wsum(acc[t]);
        if (lane == 0) y[(size_t) t * ys + (size_t) h * R + r] = v;
    }
}
__global__ void k_mla_kv(const float * __restrict__ kv, int kv_stride, const float * __restrict__ w, float eps, half * __restrict__ lat,
                         const int * pos_p) {
    const int t = blockIdx.x, i = threadIdx.x;
    const float x = kv[(size_t) t * kv_stride + i];
    const float ss = bsum(x * x);
    lat[(size_t) (*pos_p + t) * MLA_LAT + i] = __float2half(x * rsqrtf(ss / MLA_LAT + eps) * w[i]);
}

constexpr int MLA_CC = 32;   // cells per chunk in shared memory
// grid (n_split, nt), 256 threads. Block: the cell slice [n * s / n_split, n * (s+1) / n_split) of token t's cells, all H heads.
// Thread tid owns latent dims 2 tid, 2 tid + 1 of every head's accumulator.
// n_split == 1: o = acc / l directly; else partials [t][s][h] = (m, l, acc[512])
__global__ void __launch_bounds__(256) k_mla_attn(const float * __restrict__ q, int q_stride, const half * __restrict__ lat, const int * pos_p,
                                                  int H, float scale, const int * __restrict__ list, int list_stride,
                                                  const int * __restrict__ list_n, float * __restrict__ part, float * __restrict__ o,
                                                  int o_stride) {
    extern __shared__ __align__(16) unsigned char smem[];
    float * qs = (float *) smem;                                  // [H][512]
    half * ls = (half *) (qs + (size_t) H * MLA_LAT);             // [MLA_CC][512]
    float * ps = (float *) (ls + (size_t) MLA_CC * MLA_LAT);      // [H][MLA_CC]
    float * corr = ps + H * MLA_CC;                               // [H]
    float * ms = corr + H;                                        // [H] running max
    float * lsum = ms + H;                                        // [H] running sum
    const int t = blockIdx.y, sp = blockIdx.x, ns = gridDim.x;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int p = *pos_p + t;
    const int n = list ? list_n[t] : p + 1;
    const int c0 = (int) ((long long) n * sp / ns), c1 = (int) ((long long) n * (sp + 1) / ns);
    const int * lt = list ? list + (size_t) t * list_stride : nullptr;
    for (int i = threadIdx.x; i < H * MLA_LAT; i += blockDim.x) qs[i] = q[(size_t) t * q_stride + i] * scale;
    for (int h = threadIdx.x; h < H; h += blockDim.x) { ms[h] = -FLT_MAX; lsum[h] = 0.0f; }
    float acc[MLA_MAXH][2];
#pragma unroll
    for (int h = 0; h < MLA_MAXH; ++h) { acc[h][0] = 0.0f; acc[h][1] = 0.0f; }
    for (int cb = c0; cb < c1; cb += MLA_CC) {
        const int cnt = min(MLA_CC, c1 - cb);
        __syncthreads();
        for (int i = threadIdx.x; i < MLA_CC * (MLA_LAT / 8); i += blockDim.x) {
            const int j = i / (MLA_LAT / 8), cc = i % (MLA_LAT / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j < cnt) {
                const int cell = lt ? lt[cb + j] : cb + j;
                v = *(const uint4 *) (lat + (size_t) cell * MLA_LAT + cc * 8);
            }
            *(uint4 *) (ls + (size_t) j * MLA_LAT + cc * 8) = v;
        }
        __syncthreads();
        // scores: warp per (h, j)
        for (int pr = w; pr < H * MLA_CC; pr += 8) {
            const int h = pr / MLA_CC, j = pr % MLA_CC;
            float sc = 0.0f;
            if (j < cnt) {
                const float * qh = qs + (size_t) h * MLA_LAT;
                const __half2 * l2 = (const __half2 *) (ls + (size_t) j * MLA_LAT);
#pragma unroll
                for (int k = 0; k < MLA_LAT / 64; ++k) {
                    const float2 lv = __half22float2(l2[lane + 32 * k]);
                    const float2 qv = *(const float2 *) (qh + 2 * (lane + 32 * k));
                    sc += lv.x * qv.x + lv.y * qv.y;
                }
                sc = wsum(sc);
            }
            if (lane == 0) ps[pr] = j < cnt ? sc : -FLT_MAX;
        }
        __syncthreads();
        // online softmax per head: warp per head, lane per cell
        for (int h = w; h < H; h += 8) {
            const float v = ps[h * MLA_CC + lane];
            const float mo = ms[h];
            const float mn = fmaxf(mo, wmax(v));
            const float e = lane < cnt ? expf(v - mn) : 0.0f;
            ps[h * MLA_CC + lane] = e;
            const float z = wsum(e);
            if (lane == 0) {
                const float c = expf(mo - mn);
                corr[h] = c;
                lsum[h] = lsum[h] * c + z;
                ms[h] = mn;
            }
        }
        __syncthreads();
        const int d = 2 * threadIdx.x;
#pragma unroll
        for (int h = 0; h < MLA_MAXH; ++h) {
            if (h >= H) break;
            const float c = corr[h];
            float a0 = acc[h][0] * c, a1 = acc[h][1] * c;
            const float * ph = ps + h * MLA_CC;
            for (int j = 0; j < cnt; ++j) {
                const float2 lv = __half22float2(*(const __half2 *) (ls + (size_t) j * MLA_LAT + d));
                a0 += ph[j] * lv.x;
                a1 += ph[j] * lv.y;
            }
            acc[h][0] = a0; acc[h][1] = a1;
        }
    }
    __syncthreads();
    const int d = 2 * threadIdx.x;
    if (ns == 1) {
#pragma unroll
        for (int h = 0; h < MLA_MAXH; ++h) {
            if (h >= H) break;
            const float il = lsum[h] > 0.0f ? 1.0f / lsum[h] : 0.0f;
            *(float2 *) (o + (size_t) t * o_stride + (size_t) h * MLA_LAT + d) = make_float2(acc[h][0] * il, acc[h][1] * il);
        }
        return;
    }
    float * pp = part + ((size_t) t * ns + sp) * H * (MLA_LAT + 2);
#pragma unroll
    for (int h = 0; h < MLA_MAXH; ++h) {
        if (h >= H) break;
        float * ph = pp + (size_t) h * (MLA_LAT + 2);
        if (threadIdx.x == 0) { ph[0] = ms[h]; ph[1] = lsum[h]; }
        ph[2 + d] = acc[h][0];
        ph[3 + d] = acc[h][1];
    }
}
// ---- prefill: tensor-core attention, block per token, all (<= 32) heads at once ----
__device__ __forceinline__ void mma16816_5(float * c, const unsigned * a, const unsigned * b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldsm5_x4(unsigned * r, const void * smem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
__device__ __forceinline__ void ldsm5_x4_t(unsigned * r, const void * smem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
constexpr int TC_LD = MLA_LAT + 8;   // smem row stride (halves): conflict-free ldmatrix
constexpr int TC_KC = 32;            // cells per tile
constexpr size_t TC_SMEM = (size_t) 32 * TC_LD * 2 * 2 + 2 * 32 * 33 * 4 + 32 * 40 * 2 + 3 * 32 * 4;
// Token t attends its cells (list or 0..pos+t) with every head: S = Q K^T and O += P K on mma.m16n8k16, online softmax
// per head row. Rows: the <= 32 heads (padded); K tiles of 32 cells from the fp16 latent cache.
__global__ void __launch_bounds__(256) k_mla_attn_tc(const float * __restrict__ q, int q_stride, const half * __restrict__ lat,
                                                     const int * pos_p, int H, float scale, const int * __restrict__ list, int list_stride,
                                                     const int * __restrict__ list_n, float * __restrict__ o, int o_stride) {
    extern __shared__ __align__(16) unsigned char smem[];
    half * Qs = (half *) smem;                       // [32][TC_LD]
    half * Ks = Qs + 32 * TC_LD;                     // [32][TC_LD]
    float * Ss = (float *) (Ks + 32 * TC_LD);        // [2][32][33]: partial scores of the two halves of the latent dim
    half * Ps = (half *) (Ss + 2 * 32 * 33);         // [32][40]
    float * mrow = (float *) (Ps + 32 * 40);         // [32] running max
    float * lrow = mrow + 32;                        // [32] running sum
    float * arow = lrow + 32;                        // [32] rescale of this tile
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int p = *pos_p + t;
    const int n = list ? list_n[t] : p + 1;
    const int * lt = list ? list + (size_t) t * list_stride : nullptr;
    // Q (scaled) as fp16 rows; heads >= H zero
    for (int i = tid; i < 32 * (MLA_LAT / 8); i += blockDim.x) {
        const int h = i / (MLA_LAT / 8), c = (i % (MLA_LAT / 8)) * 8;
        __half2 hv[4];
        if (h < H) {
            const float * qr = q + (size_t) t * q_stride + (size_t) h * MLA_LAT + c;
            const float4 a = *(const float4 *) qr, b = *(const float4 *) (qr + 4);
            hv[0] = __floats2half2_rn(a.x * scale, a.y * scale); hv[1] = __floats2half2_rn(a.z * scale, a.w * scale);
            hv[2] = __floats2half2_rn(b.x * scale, b.y * scale); hv[3] = __floats2half2_rn(b.z * scale, b.w * scale);
        } else hv[0] = hv[1] = hv[2] = hv[3] = __floats2half2_rn(0.0f, 0.0f);
        *(uint4 *) (Qs + h * TC_LD + c) = *(const uint4 *) hv;
    }
    if (tid < 32) { mrow[tid] = -FLT_MAX; lrow[tid] = 0.0f; }
    // S: warp = (m-tile of 16 heads, pair of n-tiles = 16 cells, half of the latent dim); O: (m-tile, 128 output columns)
    const int mt = w & 1, np = w >> 1, nps = (w >> 1) & 1, kh = w >> 2;
    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f; }
    for (int cb = 0; cb < n; cb += TC_KC) {
        const int cnt = min(TC_KC, n - cb);
        __syncthreads();
        for (int i = tid; i < TC_KC * (MLA_LAT / 8); i += blockDim.x) {
            const int j = i / (MLA_LAT / 8), c = (i % (MLA_LAT / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j < cnt) { const int cell = lt ? lt[cb + j] : cb + j; v = *(const uint4 *) (lat + (size_t) cell * MLA_LAT + c); }
            *(uint4 *) (Ks + j * TC_LD + c) = v;
        }
        __syncthreads();
        // S tile: this warp's 16 heads x 16 cells
        float sacc[2][4] = {};
#pragma unroll 4
        for (int ks = kh * (MLA_LAT / 32); ks < (kh + 1) * (MLA_LAT / 32); ++ks) {
            unsigned a[4], b[4];
            ldsm5_x4(a, Qs + (mt * 16 + (lane & 15)) * TC_LD + ks * 16 + (lane >> 4) * 8);
            ldsm5_x4(b, Ks + (nps * 16 + (lane & 7) + ((lane >> 4) << 3)) * TC_LD + ks * 16 + ((lane >> 3) & 1) * 8);
            const unsigned b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
            mma16816_5(sacc[0], a, b0);
            mma16816_5(sacc[1], a, b1);
        }
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int qd = 0; qd < 4; ++qd) {
                const int row = mt * 16 + gid + (qd >> 1) * 8, col = nps * 16 + ni * 8 + 2 * tig + (qd & 1);
                Ss[(kh * 32 + row) * 33 + col] = sacc[ni][qd];
            }
        __syncthreads();
        // online softmax: warp w owns rows 4w..4w+3, lane = column
        for (int r = 4 * w; r < 4 * w + 4; ++r) {
            const float v = lane < cnt ? Ss[r * 33 + lane] + Ss[(32 + r) * 33 + lane] : -FLT_MAX;
            const float mo = mrow[r], mn = fmaxf(mo, wmax(v));
            const float e = lane < cnt ? expf(v - mn) : 0.0f;
            const float z = wsum(e);
            Ps[r * 40 + lane] = __float2half(e);
            if (lane == 0) { const float al = expf(mo - mn); arow[r] = al; lrow[r] = lrow[r] * al + z; mrow[r] = mn; }
        }
        __syncthreads();
        // O = O * alpha + P K : this warp's 16 heads x 128 columns
        {
            const float al0 = arow[mt * 16 + gid], al1 = arow[mt * 16 + gid + 8];
#pragma unroll
            for (int i = 0; i < 16; ++i) { acc[i][0] *= al0; acc[i][1] *= al0; acc[i][2] *= al1; acc[i][3] *= al1; }
        }
#pragma unroll
        for (int kk = 0; kk < TC_KC / 16; ++kk) {
            unsigned a[4];
            ldsm5_x4(a, Ps + (mt * 16 + (lane & 15)) * 40 + kk * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int pr = 0; pr < 8; ++pr) {
                unsigned b[4];
                ldsm5_x4_t(b, Ks + (kk * 16 + (lane & 15)) * TC_LD + np * 128 + pr * 16 + (lane >> 4) * 8);
                const unsigned b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
                mma16816_5(acc[2 * pr], a, b0);
                mma16816_5(acc[2 * pr + 1], a, b1);
            }
        }
    }
    __syncthreads();
    const float il0 = lrow[mt * 16 + gid] > 0.0f ? 1.0f / lrow[mt * 16 + gid] : 0.0f;
    const float il1 = lrow[mt * 16 + gid + 8] > 0.0f ? 1.0f / lrow[mt * 16 + gid + 8] : 0.0f;
    const int h0 = mt * 16 + gid, h1 = h0 + 8;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = np * 128 + i * 8 + 2 * tig;
        if (h0 < H) *(float2 *) (o + (size_t) t * o_stride + (size_t) h0 * MLA_LAT + col) = make_float2(acc[i][0] * il0, acc[i][1] * il0);
        if (h1 < H) *(float2 *) (o + (size_t) t * o_stride + (size_t) h1 * MLA_LAT + col) = make_float2(acc[i][2] * il1, acc[i][3] * il1);
    }
}

// grid (H, nt), 256 threads
__global__ void k_mla_combine(const float * __restrict__ part, int ns, int H, float * __restrict__ o, int o_stride) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float * base = part + (size_t) t * ns * H * (MLA_LAT + 2) + (size_t) h * (MLA_LAT + 2);
    float m = -FLT_MAX;
    for (int s = 0; s < ns; ++s) m = fmaxf(m, base[(size_t) s * H * (MLA_LAT + 2)]);
    float l = 0.0f, a0 = 0.0f, a1 = 0.0f;
    const int d = 2 * threadIdx.x;
    for (int s = 0; s < ns; ++s) {
        const float * ps = base + (size_t) s * H * (MLA_LAT + 2);
        if (ps[1] <= 0.0f) continue;
        const float f = expf(ps[0] - m);
        l += ps[1] * f;
        a0 += ps[2 + d] * f;
        a1 += ps[3 + d] * f;
    }
    const float il = l > 0.0f ? 1.0f / l : 0.0f;
    *(float2 *) (o + (size_t) t * o_stride + (size_t) h * MLA_LAT + d) = make_float2(a0 * il, a1 * il);
}

// ---------------- k-pool indexer ----------------
// grid (nt + 3 pools max: nt / 4 + 2), 128 threads. Pass 1 (pool): pools completed in [pos, pos + nt)
__global__ void k_gidx_pool(const float * __restrict__ ikraw, const float * __restrict__ igraw, int stride, const float * __restrict__ lnw,
                            const float * __restrict__ lnb, float eps, const float * __restrict__ ape, const half * __restrict__ ring,
                            half * __restrict__ pooled, const int * pos_p, int nt) {
    const int pos = *pos_p, i = threadIdx.x;
    const int b = pos / 4 + blockIdx.x, last = 4 * b + 3;
    if (last < pos || last > pos + nt - 1) return;
    float kv[4], gv[4];
    for (int j = 0; j < 4; ++j) {
        const int c = 4 * b + j;
        if (c >= pos) {
            const float * kr = ikraw + (size_t) (c - pos) * stride;
            const float x = kr[i];
            const float mean = bsum(x) / GIDX_DIM;
            const float var = bsum((x - mean) * (x - mean)) / GIDX_DIM;
            kv[j] = __half2float(__float2half((x - mean) * rsqrtf(var + eps) * lnw[i] + lnb[i]));
            gv[j] = __half2float(__float2half(igraw[(size_t) (c - pos) * stride + i]));
        } else {
            kv[j] = __half2float(ring[(c & 7) * 2 * GIDX_DIM + i]);
            gv[j] = __half2float(ring[(c & 7) * 2 * GIDX_DIM + GIDX_DIM + i]);
        }
    }
    float m = -FLT_MAX;
    for (int j = 0; j < 4; ++j) { gv[j] += ape[j * GIDX_DIM + i]; m = fmaxf(m, gv[j]); }
    float z = 0.0f, acc = 0.0f;
    for (int j = 0; j < 4; ++j) { const float e = expf(gv[j] - m); z += e; acc += e * kv[j]; }
    pooled[(size_t) b * GIDX_DIM + i] = __float2half(acc / z);
}
// pass 2 (ring): the cells of the pool left open after these tokens
__global__ void k_gidx_ring(const float * __restrict__ ikraw, const float * __restrict__ igraw, int stride, const float * __restrict__ lnw,
                            const float * __restrict__ lnb, float eps, half * __restrict__ ring, const int * pos_p, int nt) {
    const int pos = *pos_p, end = pos + nt, i = threadIdx.x;
    const int c = end - 8 + blockIdx.x;   // the last 8 cells (a verification rollback keeps any prefix of them valid)
    if (c >= end || c < pos) return;
    const float * kr = ikraw + (size_t) (c - pos) * stride;
    const float x = kr[i];
    const float mean = bsum(x) / GIDX_DIM;
    const float var = bsum((x - mean) * (x - mean)) / GIDX_DIM;
    ring[(c & 7) * 2 * GIDX_DIM + i] = __float2half((x - mean) * rsqrtf(var + eps) * lnw[i] + lnb[i]);
    ring[(c & 7) * 2 * GIDX_DIM + GIDX_DIM + i] = __float2half(igraw[(size_t) (c - pos) * stride + i]);
}

// scores: grid (pool blocks of 256, rows), thread per pool
__device__ __forceinline__ unsigned fkey(float f) {   // order-preserving float -> unsigned
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
constexpr int GHIST = 65536;   // score histogram bins: the top 16 bits of the order-preserving key
// hist (optional): per row, counts of key >> 16 (selection without full radix passes)
__global__ void __launch_bounds__(256) k_gidx_score(const float * __restrict__ iq, int iq_stride, const float * __restrict__ wt, int w_stride,
                                                    const half * __restrict__ pooled, const int * pos_p, int t_off, int top,
                                                    float * __restrict__ scores, int score_stride, unsigned * __restrict__ hist) {
    const int tl = blockIdx.y, t = t_off + tl;
    const int p = *pos_p + t, np = (p + 1) / 4;
    if (np <= top) return;
    const int b0 = blockIdx.x * 256;
    if (b0 >= np) return;
    __shared__ float qs[GIDX_HEADS * GIDX_DIM];
    __shared__ float ws[GIDX_HEADS];
    for (int i = threadIdx.x; i < GIDX_HEADS * GIDX_DIM; i += blockDim.x) qs[i] = iq[(size_t) t * iq_stride + i];
    if (threadIdx.x < GIDX_HEADS) ws[threadIdx.x] = wt[(size_t) t * w_stride + threadIdx.x];
    __syncthreads();
    const int b = b0 + threadIdx.x;
    if (b >= np) return;
    float acc[GIDX_HEADS];
#pragma unroll
    for (int h = 0; h < GIDX_HEADS; ++h) acc[h] = 0.0f;
    const half * pk = pooled + (size_t) b * GIDX_DIM;
    for (int d = 0; d < GIDX_DIM; d += 8) {
        const uint4 v = *(const uint4 *) (pk + d);
        const __half2 * h2 = (const __half2 *) &v;
        float f[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 x = __half22float2(h2[j]); f[2 * j] = x.x; f[2 * j + 1] = x.y; }
#pragma unroll
        for (int h = 0; h < GIDX_HEADS; ++h) {
            const float * qh = qs + h * GIDX_DIM + d;
            acc[h] += f[0] * qh[0] + f[1] * qh[1] + f[2] * qh[2] + f[3] * qh[3] + f[4] * qh[4] + f[5] * qh[5] + f[6] * qh[6] + f[7] * qh[7];
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < GIDX_HEADS; ++h) s += ws[h] * fmaxf(acc[h], 0.0f);
    scores[(size_t) tl * score_stride + b] = s;
    if (hist) atomicAdd(&hist[(size_t) tl * GHIST + (fkey(s) >> 16)], 1u);
}

// qwen4exp QSA indexer scores (same arithmetic as k_idx_select in kernels4.cu): sum_h relu(q_h . pool) / sqrt(128), n_head <= 4;
// grid (pool blocks of 256, rows), thread per pool, plus the histogram of key >> 16
__global__ void __launch_bounds__(256) k_qidx_score(const float * __restrict__ qn, const half * __restrict__ pool, const int * pos_p,
                                                    int t_off, int n_head, int top, float * __restrict__ scores, int score_stride,
                                                    unsigned * __restrict__ hist) {
    const int tl = blockIdx.y, t = t_off + tl;
    const int p = *pos_p + t, np = (p + 1) / 4;
    if (np <= top) return;
    const int b0 = blockIdx.x * 256;
    if (b0 >= np) return;
    __shared__ float qs[4 * 128];
    for (int i = threadIdx.x; i < n_head * 128; i += blockDim.x) qs[i] = qn[(size_t) t * n_head * 128 + i];
    __syncthreads();
    const int b = b0 + threadIdx.x;
    if (b >= np) return;
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
    float sum = 0.0f;
    for (int h = 0; h < n_head; ++h) sum += fmaxf(acc[h], 0.0f);
    const float sc = sum * rsqrtf(128.0f);
    scores[(size_t) tl * score_stride + b] = sc;
    atomicAdd(&hist[(size_t) tl * GHIST + (fkey(sc) >> 16)], 1u);
}

// block per token (1024 threads), from the score histogram: the bin B holding the top-th largest key; pools above B are
// selected outright, B's pools (bitmap) are resolved by two 8-bit radix passes; ties at the final key go to the lower pool
// index. Output as the old kernel: ascending cells of the selected pools, then the open pool's cells.
__global__ void __launch_bounds__(1024) k_gidx_select_h(const float * __restrict__ scores, int score_stride, const unsigned * __restrict__ hist,
                                                        const int * pos_p, int t_off, int top, int * __restrict__ list, int list_stride,
                                                        int * __restrict__ list_n) {
    const int tl = blockIdx.x, t = t_off + tl, tid = threadIdx.x;
    const int p = *pos_p + t, np = (p + 1) / 4;
    int * lt = list + (size_t) t * list_stride;
    if (np <= top) {
        for (int i = tid; i <= p; i += blockDim.x) lt[i] = i;
        if (tid == 0) list_n[t] = p + 1;
        return;
    }
    const float * sc = scores + (size_t) tl * score_stride;
    const unsigned * hg = hist + (size_t) tl * GHIST;
    constexpr int MAXW = 65536 / 32 + 64;   // bitmap words (np <= 65536 + 2048)
    __shared__ unsigned sel[MAXW], bnd[MAXW];
    __shared__ unsigned part[1024];
    __shared__ unsigned sB, sNeed, sThr, sNeedEq;
    __shared__ unsigned h8[256];
    const int nw = (np + 31) / 32;
    for (int i = tid; i < nw; i += blockDim.x) { sel[i] = 0; bnd[i] = 0; }
    // 1. bin B: thread i owns bins [65536 - 64 (i+1), 65536 - 64 i) (thread 0 the highest); block scan of the thread sums
    unsigned own = 0;
    {
        const uint4 * hv = (const uint4 *) (hg + GHIST - 64 * (tid + 1));
#pragma unroll
        for (int j = 0; j < 16; ++j) { const uint4 v = hv[j]; own += v.x + v.y + v.z + v.w; }
    }
    part[tid] = own;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {   // inclusive scan: part[i] = bins of threads 0..i
        const unsigned v = tid >= off ? part[tid - off] : 0;
        __syncthreads();
        part[tid] += v;
        __syncthreads();
    }
    {
        const unsigned before = part[tid] - own;
        if (before < (unsigned) top && part[tid] >= (unsigned) top) {   // the crossing is in this thread's bins (exactly one thread)
            unsigned acc = before;
            int bin = GHIST - 64 * tid - 1;
            for (; bin > GHIST - 64 * (tid + 1); --bin) { const unsigned h = hg[bin]; if (acc + h >= (unsigned) top) break; acc += h; }
            sB = (unsigned) bin;
            sNeed = (unsigned) top - acc;
        }
    }
    __syncthreads();
    const unsigned B = sB;
    // 2. mark: above B -> selected, B -> boundary
    for (int b = tid; b < np; b += blockDim.x) {
        const unsigned bin = fkey(sc[b]) >> 16;
        if (bin > B) atomicOr(&sel[b >> 5], 1u << (b & 31));
        else if (bin == B) atomicOr(&bnd[b >> 5], 1u << (b & 31));
    }
    __syncthreads();
    // 3. boundary: radix on the low 16 bits (two 8-bit passes) over the boundary bitmap
    if (tid == 0) { sThr = B << 16; sNeedEq = sNeed; }
    for (int shift = 8; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += blockDim.x) h8[i] = 0;
        __syncthreads();
        const unsigned pre = sThr, mask = shift == 8 ? 0xffff0000u : 0xffffff00u;
        for (int w = tid; w < nw; w += blockDim.x) {
            unsigned m = bnd[w];
            while (m) {
                const int bit = __ffs(m) - 1; m &= m - 1;
                const unsigned key = fkey(sc[w * 32 + bit]);
                if ((key & mask) == pre) atomicAdd(&h8[(key >> shift) & 255], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned acc = 0, need = sNeedEq;
            for (int d = 255; d >= 0; --d) {
                if (acc + h8[d] >= need) { sThr = pre | ((unsigned) d << shift); sNeedEq = need - acc; break; }
                acc += h8[d];
            }
        }
        __syncthreads();
    }
    const unsigned thr = sThr;
    for (int w = tid; w < nw; w += blockDim.x) {   // boundary pools above the threshold key
        unsigned m = bnd[w], add = 0;
        while (m) {
            const int bit = __ffs(m) - 1; m &= m - 1;
            if (fkey(sc[w * 32 + bit]) > thr) add |= 1u << bit;
        }
        sel[w] |= add;
    }
    __syncthreads();
    {   // the first sNeedEq pools at exactly the threshold key (pool order): per-thread word ranges, prefix over the threads
        const int per = (nw + 1023) / 1024, w0 = tid * per, w1 = min(nw, w0 + per);
        unsigned eq = 0;
        for (int w = w0; w < w1; ++w) {
            unsigned m = bnd[w];
            while (m) { const int bit = __ffs(m) - 1; m &= m - 1; eq += fkey(sc[w * 32 + bit]) == thr; }
        }
        part[tid] = eq;
        __syncthreads();
        for (int off = 1; off < 1024; off <<= 1) {
            const unsigned v = tid >= off ? part[tid - off] : 0;
            __syncthreads();
            part[tid] += v;
            __syncthreads();
        }
        const unsigned need = sNeedEq;
        unsigned rank = part[tid] - eq;
        for (int w = w0; w < w1 && rank < need; ++w) {
            unsigned m = bnd[w], add = 0;
            while (m && rank < need) {
                const int bit = __ffs(m) - 1; m &= m - 1;
                if (fkey(sc[w * 32 + bit]) == thr) { add |= 1u << bit; ++rank; }
            }
            if (add) atomicOr(&sel[w], add);
        }
    }
    __syncthreads();
    // 4. ascending cell list: per thread a contiguous range of words, prefix over the threads
    const int per = (nw + 1023) / 1024, w0 = tid * per, w1 = min(nw, w0 + per);
    unsigned cnt = 0;
    for (int w = w0; w < w1; ++w) cnt += __popc(sel[w]);
    part[tid] = cnt;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {   // inclusive scan
        const unsigned v = tid >= off ? part[tid - off] : 0;
        __syncthreads();
        part[tid] += v;
        __syncthreads();
    }
    int o = (int) (part[tid] - cnt) * 4;
    for (int w = w0; w < w1; ++w) {
        unsigned m = sel[w];
        while (m) {
            const int bit = __ffs(m) - 1; m &= m - 1;
            const int b = w * 32 + bit;
            lt[o] = 4 * b; lt[o + 1] = 4 * b + 1; lt[o + 2] = 4 * b + 2; lt[o + 3] = 4 * b + 3;
            o += 4;
        }
    }
    const int nsel = (int) part[1023];
    const int tail0 = 4 * np;
    for (int c = tail0 + tid; c <= p; c += blockDim.x) lt[nsel * 4 + (c - tail0)] = c;
    if (tid == 0) list_n[t] = nsel * 4 + (p - tail0 + 1);
}

// block per token (1024 threads): cell list
__global__ void __launch_bounds__(1024) k_gidx_select(const float * __restrict__ scores, int score_stride, const int * pos_p, int t_off,
                                                      int top, int * __restrict__ list, int list_stride, int * __restrict__ list_n) {
    const int tl = blockIdx.x, t = t_off + tl;
    const int p = *pos_p + t, np = (p + 1) / 4;
    int * lt = list + (size_t) t * list_stride;
    if (np <= top) {
        for (int i = threadIdx.x; i <= p; i += blockDim.x) lt[i] = i;
        if (threadIdx.x == 0) list_n[t] = p + 1;
        return;
    }
    const float * sc = scores + (size_t) tl * score_stride;
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, need;
    if (threadIdx.x == 0) { prefix = 0; need = top; }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned mask_hi = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int b = threadIdx.x; b < np; b += blockDim.x) {
            const unsigned key = fkey(sc[b]);
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
    // every key > threshold and the first `need` keys == threshold (pool order), as ascending cells
    const unsigned thr = prefix;
    __shared__ int base_cnt, eq_left;
    __shared__ int ws_eq[32], ws_sel[32];
    if (threadIdx.x == 0) { base_cnt = 0; eq_left = (int) need; }
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int b0 = 0; b0 < np; b0 += blockDim.x) {
        const int b = b0 + threadIdx.x;
        const unsigned key = b < np ? fkey(sc[b]) : 0u;
        const int gt = b < np && key > thr, eq = b < np && key == thr;
        const unsigned m_eq = __ballot_sync(0xffffffff, eq);
        if (lane == 0) ws_eq[wid] = __popc(m_eq);
        __syncthreads();
        int eq_before = __popc(m_eq & ((1u << lane) - 1)), tot_eq = 0;
        for (int ww = 0; ww < nw; ++ww) { if (ww < wid) eq_before += ws_eq[ww]; tot_eq += ws_eq[ww]; }
        const int el = eq_left;
        const bool sel = gt || (eq && eq_before < el);
        const unsigned m_sel = __ballot_sync(0xffffffff, sel);
        if (lane == 0) ws_sel[wid] = __popc(m_sel);
        __syncthreads();
        int r = __popc(m_sel & ((1u << lane) - 1)), chunk_sel = 0;
        for (int ww = 0; ww < nw; ++ww) { if (ww < wid) r += ws_sel[ww]; chunk_sel += ws_sel[ww]; }
        if (sel) {
            const int o = (base_cnt + r) * 4;
            lt[o] = 4 * b; lt[o + 1] = 4 * b + 1; lt[o + 2] = 4 * b + 2; lt[o + 3] = 4 * b + 3;
        }
        __syncthreads();
        if (threadIdx.x == 0) { base_cnt += chunk_sel; eq_left -= min(tot_eq, el); }
        __syncthreads();
    }
    const int tail0 = 4 * np;
    for (int c = tail0 + threadIdx.x; c <= p; c += blockDim.x) lt[base_cnt * 4 + (c - tail0)] = c;
    if (threadIdx.x == 0) list_n[t] = base_cnt * 4 + (p - tail0 + 1);
}

// ---------------- MoE ----------------
// block per token, warp 0 selects
__global__ void k_moe_route_sig(const float * __restrict__ logits, int ls, const float * __restrict__ bias, int ne, int k, float scale,
                                int * __restrict__ ids, float * __restrict__ wts, float * __restrict__ sg) {
    const int t = blockIdx.x;
    __shared__ float pr[1024], sel[1024];
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        const float pv = sigm5(logits[(size_t) t * ls + e]);
        pr[e] = pv;
        sel[e] = pv + bias[e];
    }
    __syncthreads();
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    float chosen[16];
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) {
        float bv = -FLT_MAX; int bi = 0x7fffffff;
        for (int e = lane; e < ne; e += 32) if (sel[e] > bv || (sel[e] == bv && e < bi)) { bv = sel[e]; bi = e; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, o); const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        __syncwarp();
        if (lane == 0) { ids[t * k + j] = bi; sel[bi] = -FLT_MAX; }
        chosen[j] = pr[bi];
        sum += chosen[j];
        __syncwarp();
    }
    if (lane == 0) {
        const float inv = scale / fmaxf(sum, 6.103515625e-5f);
        for (int j = 0; j < k; ++j) wts[t * k + j] = chosen[j] * inv;
        sg[t] = 1.0f;
    }
}
__global__ void k_swiglu_clamp(const float * __restrict__ gu, int gu_stride, int off, float * __restrict__ h, int h_stride, int n, float L) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float g = gu[(size_t) t * gu_stride + i], u = gu[(size_t) t * gu_stride + off + i];
    if (L > 0.0f) { g = fminf(g, L); u = fminf(fmaxf(u, -L), L); }
    h[(size_t) t * h_stride + i] = g / (1.0f + expf(-g)) * u;
}

} // namespace

void mhc_pre(const float * res, const float * mixraw, int mix_stride, const float * scale, const float * base, const float * norm_w,
             float rms_eps, float hc_eps, int iters, int n, float * hcw, float * xn, int nt, cudaStream_t s) {
    if (n > 4096) throw std::runtime_error("mhc_pre: n_embd > 4096");
    k_mhc_pre<<<nt, 1024, 0, s>>>(res, mixraw, mix_stride, scale, base, norm_w, rms_eps, hc_eps, iters, n, hcw, xn);
}
void mhc_pre_fused(const float * res, const uint8_t * fn_q8, float * part, const float * scale, const float * base, const float * norm_w,
                   float rms_eps, float hc_eps, int iters, int n, float * hcw, float * xn, int nt, cudaStream_t s) {
    if (n > 4096 || (MHC * n) % 256) throw std::runtime_error("mhc_pre_fused: n");
    const int nsl = MHC * n / 256;
    k_mhc_mix<<<dim3(nsl, nt), 256, 0, s>>>(res, fn_q8, MHC * n, part);
    k_mhc_pre2<<<nt, 1024, 0, s>>>(res, part, nsl, scale, base, norm_w, rms_eps, hc_eps, iters, n, hcw, xn);
}
void mhc_post(float * res, const float * out, const float * hcw, int n, int nt, cudaStream_t s) {
    k_mhc_post<<<dim3((n + 255) / 256, nt), 256, 0, s>>>(res, out, hcw, n);
}
void mhc_init(float * res, const float * x, int n, int nt, cudaStream_t s) {
    k_mhc_init<<<dim3((MHC * n + 255) / 256, nt), 256, 0, s>>>(res, x, n);
}
void mhc_head(const float * res, const float * w, float eps, int n, float * xn, int nt, cudaStream_t s) {
    k_mhc_head<<<nt, 1024, 0, s>>>(res, w, eps, n, xn);
}
void kda_step(const float * in, int stride, int q_off, int k_off, int v_off, int fb_off, int b_off, float * state, float * snap,
              float * o, int o_stride, const float * dt_bias, const float * A, float lb, int n_head, float eps, int nt, cudaStream_t s) {
    k_kda_step<<<dim3(n_head, 16), 256, 0, s>>>(in, stride, q_off, k_off, v_off, fb_off, b_off, state, snap, o, o_stride, dt_bias, A, lb,
                                                n_head, eps, nt);
}
void head_gemv(const half * W, int H, int R, int C, const float * x, int xs, float * y, int ys, int nt, cudaStream_t s) {
    if (C % 256 || nt > MAX_NT) throw std::runtime_error("head_gemv: C % 256 / nt");
    k_head_gemv<<<(H * R + 7) / 8, 256, 0, s>>>(W, H, R, C, x, xs, y, ys, nt);
}
void mla_kv(const float * kv, int kv_stride, const float * w, float eps, half * lat, const int * pos, int nt, cudaStream_t s) {
    k_mla_kv<<<nt, MLA_LAT, 0, s>>>(kv, kv_stride, w, eps, lat, pos);
}
void mla_init() {
    if (cudaFuncSetAttribute(k_mla_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024) != cudaSuccess ||
        cudaFuncSetAttribute(k_mla_attn_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) TC_SMEM) != cudaSuccess)
        throw std::runtime_error("mla_init: shared memory limit");
}
size_t mla_part_floats(int H, int nt) { return (size_t) nt * MLA_SPLIT * H * (MLA_LAT + 2); }
void mla_attn(const float * q, int q_stride, const half * lat, const int * pos, int H, float scale, int nt, const int * list,
              int list_stride, const int * list_n, float * part, float * o, int o_stride, cudaStream_t s) {
    if (H > MLA_MAXH) throw std::runtime_error("mla_attn: too many heads");
    const size_t smem = (size_t) H * MLA_LAT * 4 + (size_t) MLA_CC * MLA_LAT * 2 + (size_t) H * MLA_CC * 4 + 3 * H * 4;
    static const bool no_tc = getenv("HYPER5_MLA_NOTC") != nullptr;
    if (nt > MAX_NT && H <= 32 && !no_tc) {   // prefill: tensor cores, block per token
        k_mla_attn_tc<<<nt, 256, TC_SMEM, s>>>(q, q_stride, lat, pos, H, scale, list, list_stride, list_n, o, o_stride);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("mla_attn_tc launch: ") + cudaGetErrorString(e));
        return;
    }
    const int ns = nt <= MAX_NT ? MLA_SPLIT : 1;
    k_mla_attn<<<dim3(ns, nt), 256, smem, s>>>(q, q_stride, lat, pos, H, scale, list, list_stride, list_n, part, o, o_stride);
    if (ns > 1) k_mla_combine<<<dim3(H, nt), 256, 0, s>>>(part, ns, H, o, o_stride);
}
void gidx_pool(const float * ikraw, const float * igraw, int stride, const float * lnw, const float * lnb, float eps, const float * ape,
               half * ring, half * pooled, const int * pos, int nt, cudaStream_t s) {
    k_gidx_pool<<<nt / 4 + 2, GIDX_DIM, 0, s>>>(ikraw, igraw, stride, lnw, lnb, eps, ape, ring, pooled, pos, nt);
    k_gidx_ring<<<8, GIDX_DIM, 0, s>>>(ikraw, igraw, stride, lnw, lnb, eps, ring, pos, nt);
}
void idx_select_hist(const float * qn, const half * pool, const int * pos, int nt, int n_head, int top, float * scores, int score_stride,
                     int score_rows, unsigned * hist, int * list, int list_stride, int * list_n, cudaStream_t s) {
    if (n_head > 4 || score_stride > 65536 + 2048) throw std::runtime_error("idx_select_hist: sizes");
    for (int t0 = 0; t0 < nt; t0 += score_rows) {
        const int r = std::min(score_rows, nt - t0);
        cudaMemsetAsync(hist, 0, (size_t) r * GHIST * sizeof(unsigned), s);
        k_qidx_score<<<dim3((score_stride + 255) / 256, r), 256, 0, s>>>(qn, pool, pos, t0, n_head, top, scores, score_stride, hist);
        k_gidx_select_h<<<r, 1024, 0, s>>>(scores, score_stride, hist, pos, t0, top, list, list_stride, list_n);
    }
}
void gidx_select(const float * iq, int iq_stride, const float * w, int w_stride, const half * pooled, const int * pos, int nt, int top,
                 float * scores, int score_stride, int rows, int * list, int list_stride, int * list_n, cudaStream_t s, unsigned * hist) {
    if (hist && score_stride > 65536 + 2048) hist = nullptr;   // (bitmaps sized for 262k cells)
    for (int t0 = 0; t0 < nt; t0 += rows) {
        const int r = std::min(rows, nt - t0);
        if (hist) cudaMemsetAsync(hist, 0, (size_t) r * GHIST * sizeof(unsigned), s);
        k_gidx_score<<<dim3((score_stride + 255) / 256, r), 256, 0, s>>>(iq, iq_stride, w, w_stride, pooled, pos, t0, top, scores,
                                                                       score_stride, hist);
        if (hist) k_gidx_select_h<<<r, 1024, 0, s>>>(scores, score_stride, hist, pos, t0, top, list, list_stride, list_n);
        else k_gidx_select<<<r, 1024, 0, s>>>(scores, score_stride, pos, t0, top, list, list_stride, list_n);
    }
}
void moe_route_sig(const float * logits, int ls, const float * bias, int n_expert, int k, float scale, int * ids, float * wts,
                   float * sg, int nt, cudaStream_t s) {
    if (n_expert > 1024 || k > 16) throw std::runtime_error("moe_route_sig: sizes");
    k_moe_route_sig<<<nt, 256, 0, s>>>(logits, ls, bias, n_expert, k, scale, ids, wts, sg);
}
void swiglu_clamp(const float * gu, int gu_stride, int off, float * h, int h_stride, int n, float L, int nt, cudaStream_t s) {
    k_swiglu_clamp<<<dim3((n + 255) / 256, nt), 256, 0, s>>>(gu, gu_stride, off, h, h_stride, n, L);
}

} // namespace hyper
