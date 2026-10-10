#include "kernels.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace hyper {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
// block-wide sum; all threads get the result. blockDim.x multiple of 32, <= 1024
__device__ float block_sum(float v) {
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    v = lane < nw ? red[lane] : 0.0f;
    return warp_sum(v);
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// ---------------- GEMV (nt token rows share every weight load) ----------------
constexpr int GEMV_WARPS = 8;

__device__ __forceinline__ float input_inv_rms(const float * __restrict__ ss, int nss, int k, const float * nw, float eps) {
    if (!nw) return 1.0f;
    float t = 0.0f;
    for (int i = 0; i < nss; ++i) t += ss[i];
    return rsqrtf(t / k + eps);
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
// GEMV input activation (NormIn::act) for the two float2 an mma lane loads at columns c0 and c0 + 8
__device__ __forceinline__ void apply_act(const NormIn & nin, const float * xr, int c0, float2 & v0, float2 & v1) {
    if (nin.act == 1) {
        const float s = nin.act_scale;
        v0.x = silu_f(v0.x * s); v0.y = silu_f(v0.y * s); v1.x = silu_f(v1.x * s); v1.y = silu_f(v1.y * s);
    } else {
        float2 u0 = *(const float2 *) (xr + c0 + nin.glu_off), u1 = *(const float2 *) (xr + c0 + 8 + nin.glu_off);
        if (nin.act == 3) {   // SwiGLU with limit act_scale
            const float L = nin.act_scale;
            v0.x = fminf(v0.x, L); v0.y = fminf(v0.y, L); v1.x = fminf(v1.x, L); v1.y = fminf(v1.y, L);
            u0.x = fminf(fmaxf(u0.x, -L), L); u0.y = fminf(fmaxf(u0.y, -L), L); u1.x = fminf(fmaxf(u1.x, -L), L); u1.y = fminf(fmaxf(u1.y, -L), L);
        }
        v0.x = silu_f(v0.x) * u0.x; v0.y = silu_f(v0.y) * u0.y; v1.x = silu_f(v1.x) * u1.x; v1.y = silu_f(v1.y) * u1.y;
    }
}

// ---- tensor-core Q8 GEMM for few tokens ----
// two int8 (bytes of v selected by `sel`) -> half2 exactly: half(1024 + (b ^ 0x80)) - 1152
__device__ __forceinline__ unsigned i8x2_to_h2(unsigned v, unsigned sel) {
    const unsigned xx = __byte_perm(v ^ 0x80808080u, 0x64646464u, sel);
    const unsigned magic = 0x64806480u;   // half2(1152, 1152)
    unsigned r;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(xx), "r"(magic));
    return r;
}
__device__ __forceinline__ void mma16816(float * c, const unsigned * a, const unsigned * b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 hh = __floats2half2_rn(lo, hi);
    return *(const unsigned *) &hh;
}

// block = 8 warps on one 16-row tile (M), tokens are the N dimension (gid < nt valid), warp w takes k blocks
// w, w+8, ...; per k block: one 16-byte fragment load per lane, 2 mma, block-local result scaled by the row
// scales in fp32; partials reduced through shared memory. Optional fused input RMSNorm and residual add.
__global__ void k_mma_q8(const uint4 * __restrict__ wq, const half * __restrict__ ws, int n, int k,
                         const float * __restrict__ x, int xs, float * __restrict__ y, int ys, const float * __restrict__ add,
                         int nt, NormIn nin, float * __restrict__ kpart, unsigned * kcnt) {
    // split-K (gridDim.y > 1): block (tile, part) covers k blocks [part*kb/P, (part+1)*kb/P); partials go to kpart,
    // the last block of a tile to finish sums them in part order (deterministic) and resets the tile's counter
    const int tile = blockIdx.x, part = blockIdx.y, P = gridDim.y;
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int kb = k / 32;
    const uint4 * tq = wq + (size_t) tile * kb * 32 + lane;
    const half * ts = ws + (size_t) tile * kb * 16;
    const bool tok_ok = gid < nt;
    const int tok = tok_ok ? gid : 0;
    const float * xr = x + (size_t) tok * xs;
    const float inv = input_inv_rms(nin.ss + tok * nin.nss, nin.nss, k, nin.w, nin.eps);
    float acc[4] = {0, 0, 0, 0};
    const int b_end = (int) ((int64_t) (part + 1) * kb / P);
    for (int b = (int) ((int64_t) part * kb / P) + w; b < b_end; b += nw) {
        const uint4 q = __ldg(tq + (size_t) b * 32);
        const float s_lo = __half2float(ts[(size_t) b * 16 + gid]), s_hi = __half2float(ts[(size_t) b * 16 + gid + 8]);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        float tmp[4] = {0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned a[4];
            a[0] = i8x2_to_h2(qw[2 * ks], 0x5140);
            a[1] = i8x2_to_h2(qw[2 * ks], 0x7362);
            a[2] = i8x2_to_h2(qw[2 * ks + 1], 0x5140);
            a[3] = i8x2_to_h2(qw[2 * ks + 1], 0x7362);
            const int c0 = b * 32 + ks * 16 + 2 * tig;
            unsigned bb[2] = {0, 0};
            if (tok_ok) {
                float2 v0 = *(const float2 *) (xr + c0), v1 = *(const float2 *) (xr + c0 + 8);
                if (nin.w) {
                    const float2 w0 = *(const float2 *) (nin.w + c0), w1 = *(const float2 *) (nin.w + c0 + 8);
                    v0.x *= inv * w0.x; v0.y *= inv * w0.y; v1.x *= inv * w1.x; v1.y *= inv * w1.y;
                }
                if (nin.act) apply_act(nin, xr, c0, v0, v1);
                bb[0] = pack_h2(v0.x, v0.y);
                bb[1] = pack_h2(v1.x, v1.y);
            }
            mma16816(tmp, a, bb);
        }
        acc[0] += tmp[0] * s_lo; acc[1] += tmp[1] * s_lo;
        acc[2] += tmp[2] * s_hi; acc[3] += tmp[3] * s_hi;
    }
    __shared__ float red[8][32][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) red[w][lane][i] = acc[i];
    __syncthreads();
    if (P > 1) {
        __shared__ unsigned ticket;
        if (w == 0) {
            float * dst = kpart + ((size_t) tile * P + part) * 128 + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) { float t = 0; for (int ww = 0; ww < nw; ++ww) t += red[ww][lane][i]; dst[i] = t; }
            __threadfence();
        }
        __syncthreads();
        if (threadIdx.x == 0) ticket = atomicAdd(&kcnt[tile], 1u);
        __syncthreads();
        if (ticket != (unsigned) P - 1) return;
        __threadfence();
        if (w == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] = 0.0f;
            for (int pp = 0; pp < P; ++pp) {
                const float * src = kpart + ((size_t) tile * P + pp) * 128 + lane * 4;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] += __ldcg(src + i);
            }
            if (lane == 0) kcnt[tile] = 0;
        }
    } else if (w == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { float t = 0; for (int ww = 0; ww < nw; ++ww) t += red[ww][lane][i]; acc[i] = t; }
    }
    if (w == 0) {
        const int r0 = tile * 16 + gid, t0 = 2 * tig;
        auto put = [&](int row, int t, float v) {
            if (row < n && t < nt) { const size_t o = (size_t) t * ys + row; y[o] = add ? add[o] + v : v; }
        };
        put(r0, t0, acc[0]); put(r0, t0 + 1, acc[1]);
        put(r0 + 8, t0, acc[2]); put(r0 + 8, t0 + 1, acc[3]);
    }
}

// ---- K-quant dense GEMV (KQW: Q4_K / Q6_K in fragment order, see kernels.cuh) ----
// pair p of a lane's 8 values in word w: halves (1024 + v[2p], 1024 + v[2p+1]) (nibbles at bits 4p and 16 + 4p)
__device__ __forceinline__ unsigned kq_nib(unsigned w, int p) { return ((w >> (4 * p)) & 0x000f000fu) | 0x64006400u; }
// Q6_K top bits of pair P (bits 2P and 16 + 2P of h) moved to bits 4-5 of each half
__device__ __forceinline__ unsigned kq_hi(unsigned h, int P) { return ((h >> (2 * P)) << 4) & 0x00300030u; }
__device__ __forceinline__ unsigned h2sub(unsigned a, unsigned b) {
    unsigned r;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
// same structure as k_mma_q8 (block per 16-row tile, warps over k blocks, optional split-K over gridDim.y)
template <KQ T>
__global__ void k_mma_kq(KQW W, const float * __restrict__ x, int xs, float * __restrict__ y, int ys, int nt, NormIn nin,
                         float * __restrict__ kpart, unsigned * kcnt) {
    const int tile = blockIdx.x, part = blockIdx.y, P = gridDim.y;
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int kb = W.k / 32, nsb = kb / W.dg;
    const bool tok_ok = gid < nt;
    const int tok = tok_ok ? gid : 0;
    const float * xr = x + (size_t) tok * xs;
    float acc[4] = {0, 0, 0, 0};
    const unsigned ones[4] = {0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u};
    const int b_end = (int) ((int64_t) (part + 1) * kb / P);
    for (int b = (int) ((int64_t) part * kb / P) + w; b < b_end; b += nw) {
        const size_t tb = (size_t) tile * kb + b;
        const uint2 q = __ldg(W.lo + tb * 32 + lane);
        const unsigned hb = T == KQ::Q6K ? __ldg(W.hi + tb * 32 + lane) : 0u;
        const float2 dlo = __half22float2(W.d[((size_t) tile * nsb + b / W.dg) * 16 + gid]);
        const float2 dhi = __half22float2(W.d[((size_t) tile * nsb + b / W.dg) * 16 + gid + 8]);
        float tmp[2][4] = {}, tx[4] = {0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const unsigned wv = ks ? q.y : q.x;
            unsigned a[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                if (T == KQ::Q4K) a[p] = h2sub(kq_nib(wv, p), 0x64006400u);                              // v
                else a[p] = h2sub(kq_nib(wv, p) | kq_hi(hb, ks * 4 + p), 0x64206420u);                  // v - 32
            }
            const int c0 = b * 32 + ks * 16 + 2 * tig;
            unsigned bb[2] = {0, 0};
            if (tok_ok) {
                float2 v0 = *(const float2 *) (xr + c0), v1 = *(const float2 *) (xr + c0 + 8);
                if (nin.act) apply_act(nin, xr, c0, v0, v1);
                bb[0] = pack_h2(v0.x, v0.y);
                bb[1] = pack_h2(v1.x, v1.y);
            }
            mma16816(tmp[ks], a, bb);
            if (T == KQ::Q4K) mma16816(tx, ones, bb);   // sum of x over the block (the min term)
        }
        if (T == KQ::Q4K) {
            const unsigned slo = W.scm[tb * 16 + gid], shi = W.scm[tb * 16 + gid + 8];
            const float s_lo = dlo.x * (float) (slo & 0xff), m_lo = dlo.y * (float) (slo >> 8);
            const float s_hi = dhi.x * (float) (shi & 0xff), m_hi = dhi.y * (float) (shi >> 8);
            acc[0] += (tmp[0][0] + tmp[1][0]) * s_lo - tx[0] * m_lo; acc[1] += (tmp[0][1] + tmp[1][1]) * s_lo - tx[1] * m_lo;
            acc[2] += (tmp[0][2] + tmp[1][2]) * s_hi - tx[2] * m_hi; acc[3] += (tmp[0][3] + tmp[1][3]) * s_hi - tx[3] * m_hi;
        } else {
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const float s_lo = dlo.x * (float) W.sc6[(tb * 2 + ks) * 16 + gid], s_hi = dhi.x * (float) W.sc6[(tb * 2 + ks) * 16 + gid + 8];
                acc[0] += tmp[ks][0] * s_lo; acc[1] += tmp[ks][1] * s_lo;
                acc[2] += tmp[ks][2] * s_hi; acc[3] += tmp[ks][3] * s_hi;
            }
        }
    }
    __shared__ float red[8][32][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) red[w][lane][i] = acc[i];
    __syncthreads();
    if (P > 1) {
        __shared__ unsigned ticket;
        if (w == 0) {
            float * dst = kpart + ((size_t) tile * P + part) * 128 + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) { float t = 0; for (int ww = 0; ww < nw; ++ww) t += red[ww][lane][i]; dst[i] = t; }
            __threadfence();
        }
        __syncthreads();
        if (threadIdx.x == 0) ticket = atomicAdd(&kcnt[tile], 1u);
        __syncthreads();
        if (ticket != (unsigned) P - 1) return;
        __threadfence();
        if (w == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] = 0.0f;
            for (int pp = 0; pp < P; ++pp) {
                const float * src = kpart + ((size_t) tile * P + pp) * 128 + lane * 4;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] += __ldcg(src + i);
            }
            if (lane == 0) kcnt[tile] = 0;
        }
    } else if (w == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { float t = 0; for (int ww = 0; ww < nw; ++ww) t += red[ww][lane][i]; acc[i] = t; }
    }
    if (w == 0) {
        const int r0 = tile * 16 + gid, t0 = 2 * tig;
        auto put = [&](int row, int t, float v) { if (row < W.n && t < nt) y[(size_t) t * ys + row] = v; };
        put(r0, t0, acc[0]); put(r0, t0 + 1, acc[1]);
        put(r0 + 8, t0, acc[2]); put(r0 + 8, t0 + 1, acc[3]);
    }
}
// thread per (tile, block, lane): its 16 values to fp16 rows [r0, r0 + rows)
template <KQ T>
__global__ void k_deq_kq_f16(KQW W, int tile0, int ntile, half * __restrict__ out, int rows) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int kb = W.k / 32, nsb = kb / W.dg;
    if (i >= (size_t) ntile * kb * 32) return;
    const int lane = (int) (i % 32), b = (int) (i / 32 % kb), tile = tile0 + (int) (i / 32 / kb);
    const int gid = lane >> 2, tig = lane & 3;
    const size_t tb = (size_t) tile * kb + b;
    const uint2 q = W.lo[tb * 32 + lane];
    const unsigned hb = T == KQ::Q6K ? W.hi[tb * 32 + lane] : 0u;
    for (int hr = 0; hr < 2; ++hr) {
        const int r = gid + 8 * hr, row = tile * 16 + r - tile0 * 16;
        if (row >= rows) continue;
        const float2 dd = __half22float2(W.d[((size_t) tile * nsb + b / W.dg) * 16 + r]);
        for (int ks = 0; ks < 2; ++ks) {
            float s, m = 0.0f;
            if (T == KQ::Q4K) { const unsigned sm = W.scm[tb * 16 + r]; s = dd.x * (float) (sm & 0xff); m = dd.y * (float) (sm >> 8); }
            else s = dd.x * (float) W.sc6[(tb * 2 + ks) * 16 + r];
            const unsigned wv = ks ? q.y : q.x;
            for (int p = hr; p < 4; p += 2) {   // pairs of this row: p = 0, 2 (row gid) or 1, 3 (row gid + 8)
                for (int e = 0; e < 2; ++e) {
                    int v = (wv >> (4 * p + 16 * e)) & 0xf;
                    if (T == KQ::Q6K) v = (v | (((hb >> (2 * (ks * 4 + p) + 16 * e)) & 3) << 4)) - 32;
                    const int col = b * 32 + ks * 16 + 2 * tig + (p >= 2 ? 8 : 0) + e;
                    out[(size_t) row * W.k + col] = __float2half(s * (float) v - m);
                }
            }
        }
    }
}

// work-balanced variant for matrices with few row tiles (a block per tile would leave a partial last wave): units of
// (tile, k-range) dealt round-robin to every warp of the grid; each unit's partial [nt][16] goes to kpart, the last warp to
// finish a tile sums its UPT partials in unit order (deterministic) and resets the tile's counter
__global__ void __launch_bounds__(256) k_mma_q8_bal(const uint4 * __restrict__ wq, const half * __restrict__ ws, int n, int k,
                                                   const float * __restrict__ x, int xs, float * __restrict__ y, int ys,
                                                   const float * __restrict__ add, int nt, NormIn nin, int upt,
                                                   float * __restrict__ kpart, unsigned * kcnt) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), GW = gridDim.x * (blockDim.x >> 5);
    const int kb = k / 32, tiles = (n + 15) / 16, units = tiles * upt;
    const bool tok_ok = gid < nt;
    const int tok = tok_ok ? gid : 0;
    const float * xr = x + (size_t) tok * xs;
    const float inv = input_inv_rms(nin.ss + tok * nin.nss, nin.nss, k, nin.w, nin.eps);
    for (int u = gw; u < units; u += GW) {
        const int tile = u / upt, kc = u % upt;
        const uint4 * tq = wq + (size_t) tile * kb * 32 + lane;
        const half * ts = ws + (size_t) tile * kb * 16;
        float acc[4] = {0, 0, 0, 0};
        const int b0 = (int) ((int64_t) kc * kb / upt), b1 = (int) ((int64_t) (kc + 1) * kb / upt);
#pragma unroll 2
        for (int b = b0; b < b1; ++b) {
            const uint4 q = __ldg(tq + (size_t) b * 32);
            const float s_lo = __half2float(ts[(size_t) b * 16 + gid]), s_hi = __half2float(ts[(size_t) b * 16 + gid + 8]);
            const unsigned qw[4] = {q.x, q.y, q.z, q.w};
            float tmp[4] = {0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                unsigned a[4];
                a[0] = i8x2_to_h2(qw[2 * ks], 0x5140);
                a[1] = i8x2_to_h2(qw[2 * ks], 0x7362);
                a[2] = i8x2_to_h2(qw[2 * ks + 1], 0x5140);
                a[3] = i8x2_to_h2(qw[2 * ks + 1], 0x7362);
                const int c0 = b * 32 + ks * 16 + 2 * tig;
                unsigned bb[2] = {0, 0};
                if (tok_ok) {
                    float2 v0 = *(const float2 *) (xr + c0), v1 = *(const float2 *) (xr + c0 + 8);
                    if (nin.w) {
                        const float2 w0 = *(const float2 *) (nin.w + c0), w1 = *(const float2 *) (nin.w + c0 + 8);
                        v0.x *= inv * w0.x; v0.y *= inv * w0.y; v1.x *= inv * w1.x; v1.y *= inv * w1.y;
                    }
                    if (nin.act) apply_act(nin, xr, c0, v0, v1);
                    bb[0] = pack_h2(v0.x, v0.y);
                    bb[1] = pack_h2(v1.x, v1.y);
                }
                mma16816(tmp, a, bb);
            }
            acc[0] += tmp[0] * s_lo; acc[1] += tmp[1] * s_lo;
            acc[2] += tmp[2] * s_hi; acc[3] += tmp[3] * s_hi;
        }
        // partial: kpart[u][t][r] for tokens t = 2 tig, 2 tig + 1 (< nt) and rows gid, gid + 8
        float * dst = kpart + (size_t) u * 128;
        const int t0 = 2 * tig;
        if (t0 < nt) { dst[t0 * 16 + gid] = acc[0]; dst[t0 * 16 + gid + 8] = acc[2]; }
        if (t0 + 1 < nt) { dst[(t0 + 1) * 16 + gid] = acc[1]; dst[(t0 + 1) * 16 + gid + 8] = acc[3]; }
        __threadfence();
        __syncwarp();
        unsigned ticket = 0;
        if (lane == 0) ticket = atomicAdd(&kcnt[tile], 1u);
        ticket = __shfl_sync(0xffffffff, ticket, 0);
        if (ticket != (unsigned) upt - 1) continue;
        __threadfence();
        // last unit of the tile: sum the partials in unit order; lane = (token, row pair)
        for (int e = lane; e < nt * 16; e += 32) {
            const int t = e / 16, r = e % 16;
            float v = 0.0f;
            for (int c = 0; c < upt; ++c) v += __ldcg(kpart + ((size_t) tile * upt + c) * 128 + t * 16 + r);
            const int row = tile * 16 + r;
            if (row < n) { const size_t o = (size_t) t * ys + row; y[o] = add ? add[o] + v : v; }
        }
        if (lane == 0) kcnt[tile] = 0;
    }
}

// fp16 weights in fragment order: per k-step (16 cols) one 16-byte load per lane gives a full A fragment
__global__ void k_mma_f16(const uint4 * __restrict__ wq, int n, int k,
                          const float * __restrict__ x, int xs, float * __restrict__ y, int ys, const float * __restrict__ add,
                          int nt, NormIn nin, float * __restrict__ kpart, unsigned * kcnt) {
    // split-K as in k_mma_q8 (gridDim.y parts, last block of a tile sums them in order)
    const int tile = blockIdx.x, part = blockIdx.y, P = gridDim.y;
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int nks = k / 16;
    const uint4 * tq = wq + (size_t) tile * nks * 32 + lane;
    const bool tok_ok = gid < nt;
    const int tok = tok_ok ? gid : 0;
    const float * xr = x + (size_t) tok * xs;
    const float inv = input_inv_rms(nin.ss + tok * nin.nss, nin.nss, k, nin.w, nin.eps);
    float acc[4] = {0, 0, 0, 0};
    const int ks_end = (int) ((int64_t) (part + 1) * nks / P);
    for (int ks = (int) ((int64_t) part * nks / P) + w; ks < ks_end; ks += nw) {
        const uint4 q = __ldg(tq + (size_t) ks * 32);
        const unsigned a[4] = {q.x, q.y, q.z, q.w};
        const int c0 = ks * 16 + 2 * tig;
        unsigned bb[2] = {0, 0};
        if (tok_ok) {
            float2 v0 = *(const float2 *) (xr + c0), v1 = *(const float2 *) (xr + c0 + 8);
            if (nin.w) {
                const float2 w0 = *(const float2 *) (nin.w + c0), w1 = *(const float2 *) (nin.w + c0 + 8);
                v0.x *= inv * w0.x; v0.y *= inv * w0.y; v1.x *= inv * w1.x; v1.y *= inv * w1.y;
            }
            if (nin.act) apply_act(nin, xr, c0, v0, v1);
            bb[0] = pack_h2(v0.x, v0.y);
            bb[1] = pack_h2(v1.x, v1.y);
        }
        mma16816(acc, a, bb);
    }
    __shared__ float red[8][32][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) red[w][lane][i] = acc[i];
    __syncthreads();
    if (P > 1) {
        __shared__ unsigned ticket;
        if (w == 0) {
            float * dst = kpart + ((size_t) tile * P + part) * 128 + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) { float t = 0; for (int ww = 0; ww < nw; ++ww) t += red[ww][lane][i]; dst[i] = t; }
            __threadfence();
        }
        __syncthreads();
        if (threadIdx.x == 0) ticket = atomicAdd(&kcnt[tile], 1u);
        __syncthreads();
        if (ticket != (unsigned) P - 1) return;
        __threadfence();
        if (w == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] = 0.0f;
            for (int pp = 0; pp < P; ++pp) {
                const float * src = kpart + ((size_t) tile * P + pp) * 128 + lane * 4;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] += __ldcg(src + i);
            }
            if (lane == 0) kcnt[tile] = 0;
        }
    } else if (w == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { float t = 0; for (int ww = 0; ww < nw; ++ww) t += red[ww][lane][i]; acc[i] = t; }
    }
    if (w == 0) {
        const int r0 = tile * 16 + gid, t0 = 2 * tig;
        auto put = [&](int row, int t, float v) {
            if (row < n && t < nt) { const size_t o = (size_t) t * ys + row; y[o] = add ? add[o] + v : v; }
        };
        put(r0, t0, acc[0]); put(r0, t0 + 1, acc[1]);
        put(r0 + 8, t0, acc[2]); put(r0 + 8, t0 + 1, acc[3]);
    }
}

// ---- tensor-core GEMM for many tokens (prefill) ----
// Activations are first converted to fp16 [T][k] (optionally RMS-normed). Block = 8 warps, tile 128 weight rows
// x 128 tokens: warp (wm = w & 3, wn = w >> 2) owns 2 row tiles x 64 tokens (16 mma per k16 step). Weight
// fragments are read straight from global memory (fragment order is already coalesced) one k-block ahead;
// Q8 scales are folded into the fp16 fragment (q * d rounded to fp16). Activations go through shared memory
// with cp.async double buffering and ldmatrix.

// x[t] (fp32, stride xs) -> xh[t] (fp16, stride k), times rsqrt(mean(x^2) + eps) * w when w != nullptr
__global__ void k_to_half(const float * __restrict__ x, int xs, const float * __restrict__ w, int k, float eps,
                          half * __restrict__ xh) {
    const float * xr = x + (size_t) blockIdx.x * xs;
    half * yr = xh + (size_t) blockIdx.x * k;
    float inv = 1.0f;
    if (w) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < k; i += blockDim.x) ss += xr[i] * xr[i];
        inv = rsqrtf(block_sum(ss) / k + eps);
    }
    for (int i = 2 * threadIdx.x; i < k; i += 2 * blockDim.x) {
        float2 v = *(const float2 *) (xr + i);
        if (w) { v.x *= inv * w[i]; v.y *= inv * w[i + 1]; }
        *(__half2 *) (yr + i) = __floats2half2_rn(v.x, v.y);
    }
}

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool valid) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;"); }
__device__ __forceinline__ void ldmatrix_x4(unsigned * r, const void * smem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}

// raw fragment data of one 16-row tile for one 32-column k block
struct FragQ8 {
    uint4 q; half s0, s1;
    __device__ __forceinline__ void load(const uint4 * wq, const half * ws, int tile, int kb, int b, int lane) {
        const size_t i = (size_t) tile * kb + b;
        q = __ldg(wq + i * 32 + lane);
        s0 = ws[i * 16 + (lane >> 2)];
        s1 = ws[i * 16 + (lane >> 2) + 8];
    }
    // a[ks][4] for the two k16 steps
    __device__ __forceinline__ void unpack(unsigned (*a)[4]) const {
        const __half2 h0 = __half2half2(s0), h1 = __half2half2(s1);
        const unsigned w2[2][2] = {{q.x, q.y}, {q.z, q.w}};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned r[4];
            r[0] = i8x2_to_h2(w2[ks][0], 0x5140);
            r[1] = i8x2_to_h2(w2[ks][0], 0x7362);
            r[2] = i8x2_to_h2(w2[ks][1], 0x5140);
            r[3] = i8x2_to_h2(w2[ks][1], 0x7362);
            __half2 t;
            t = __hmul2(*(__half2 *) &r[0], h0); a[ks][0] = *(unsigned *) &t;
            t = __hmul2(*(__half2 *) &r[1], h1); a[ks][1] = *(unsigned *) &t;
            t = __hmul2(*(__half2 *) &r[2], h0); a[ks][2] = *(unsigned *) &t;
            t = __hmul2(*(__half2 *) &r[3], h1); a[ks][3] = *(unsigned *) &t;
        }
    }
};
struct FragF16 {
    uint4 q0, q1;
    __device__ __forceinline__ void load(const uint4 * wq, const half *, int tile, int kb, int b, int lane) {
        const size_t i = ((size_t) tile * kb + b) * 2;
        q0 = __ldg(wq + i * 32 + lane);
        q1 = __ldg(wq + (i + 1) * 32 + lane);
    }
    __device__ __forceinline__ void unpack(unsigned (*a)[4]) const {
        a[0][0] = q0.x; a[0][1] = q0.y; a[0][2] = q0.z; a[0][3] = q0.w;
        a[1][0] = q1.x; a[1][1] = q1.y; a[1][2] = q1.z; a[1][3] = q1.w;
    }
};

constexpr int GM_BN = 128, GM_LDS = 40;   // tokens per block, smem row stride in halfs (32 + 8 pad)

template <typename Frag, int NI>   // NI n8 tiles per warp: block covers 2 * NI * 8 tokens
__global__ void __launch_bounds__(256) k_gemm(const uint4 * __restrict__ wq, const half * __restrict__ ws, int n, int k,
                                              const half * __restrict__ xh, int T, float * __restrict__ y, int ys,
                                              const float * __restrict__ add) {
    constexpr int BN = 2 * NI * 8;
    __shared__ __align__(16) half bs[2][BN * GM_LDS];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int wm = w & 3, wn = w >> 2;
    const int t0 = blockIdx.x * BN;
    const int ntile = (n + 15) / 16, kb = k / 32;
    const int tile0 = blockIdx.y * 8 + wm * 2;
    // activation tile loader: BN rows x 64 bytes = BN * 4 16-byte chunks
    auto load_b = [&](int buf, int b) {
        for (int c = threadIdx.x; c < BN * 4; c += 256) {
            const int row = c >> 2, part = c & 3;
            const int tok = t0 + row;
            const bool ok = tok < T;
            cp_async16(&bs[buf][row * GM_LDS + part * 8], xh + (size_t) (ok ? tok : 0) * k + b * 32 + part * 8, ok);
        }
        cp_async_commit();
    };
    float acc[2][NI][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0.0f;
    Frag fr[2];
    const bool tv0 = tile0 < ntile, tv1 = tile0 + 1 < ntile;
    if (tv0) fr[0].load(wq, ws, tile0, kb, 0, lane);
    if (tv1) fr[1].load(wq, ws, tile0 + 1, kb, 0, lane);
    load_b(0, 0);
    for (int b = 0; b < kb; ++b) {
        const int cur = b & 1;
        if (b + 1 < kb) load_b(cur ^ 1, b + 1); else cp_async_commit();
        unsigned a[2][2][4];
        fr[0].unpack(a[0]);
        fr[1].unpack(a[1]);
        if (b + 1 < kb) {
            if (tv0) fr[0].load(wq, ws, tile0, kb, b + 1, lane);
            if (tv1) fr[1].load(wq, ws, tile0 + 1, kb, b + 1, lane);
        }
        cp_async_wait1();
        __syncthreads();
        const half * sb = bs[cur] + (wn * NI * 8 + (lane & 7) + ((lane >> 4) << 3)) * GM_LDS + ((lane >> 3) & 1) * 8;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int np = 0; np < NI / 2; ++np) {
                unsigned r[4];
                ldmatrix_x4(r, sb + np * 16 * GM_LDS + ks * 16);
                const unsigned b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    mma16816(acc[mi][2 * np], a[mi][ks], b0);
                    mma16816(acc[mi][2 * np + 1], a[mi][ks], b1);
                }
            }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = (tile0 + mi) * 16 + gid;
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
            const int tk = t0 + wn * NI * 8 + ni * 8 + 2 * tig;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = r0 + (e >> 1) * 8, t = tk + (e & 1);
                if (row < n && t < T) {
                    const size_t o = (size_t) t * ys + row;
                    y[o] = add ? add[o] + acc[mi][ni][e] : acc[mi][ni][e];
                }
            }
        }
    }
}

// ---------------- norms ----------------
__global__ void k_rmsnorm(const float * __restrict__ x, int xs, const float * __restrict__ w, float * __restrict__ y, int ys,
                          int n, float eps) {
    const float * xr = x + (size_t) blockIdx.x * xs;
    float * yr = y + (size_t) blockIdx.x * ys;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

__global__ void k_sumsq(const float * __restrict__ x, int xs, int n, float * ss, int nss) {
    const float * xr = x + (size_t) blockIdx.x * xs;
    float t = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) t += xr[i] * xr[i];
    t = block_sum(t);
    if (threadIdx.x < nss) ss[blockIdx.x * nss + threadIdx.x] = threadIdx.x == 0 ? t : 0.0f;
}

// ---------------- gated attention ----------------
// grid (n_head + n_kv, nt), blockDim = hd
__global__ void k_attn_prep(float * qkv, int stride, const float * qnorm, const float * knorm, half * kcache, half * vcache,
                            const int * pos_p, int max_pos, int n_head, int n_kv, int hd, int n_rot, float rope_base, float eps) {
    const int b = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = *pos_p + t;
    float * row = qkv + (size_t) t * stride;
    const bool is_q = b < n_head;
    float * vec = is_q ? row + (size_t) b * 2 * hd : row + (size_t) n_head * 2 * hd + (size_t) (b - n_head) * hd;
    const float * nw = is_q ? qnorm : knorm;
    __shared__ float buf[512];
    float xi = vec[i];
    const float ss = block_sum(xi * xi);
    xi = xi * rsqrtf(ss / hd + eps) * nw[i];
    buf[i] = xi;
    __syncthreads();
    const int half_rot = n_rot / 2;
    if (i < n_rot) {
        const int fi = i < half_rot ? i : i - half_rot;
        const double theta = (double) pos * pow((double) rope_base, -2.0 * fi / n_rot);
        double sn, cs;
        sincos(theta, &sn, &cs);
        const float c = (float) cs, s = (float) sn;
        xi = i < half_rot ? buf[i] * c - buf[i + half_rot] * s : buf[i - half_rot] * s + buf[i] * c;
    }
    if (is_q) {
        vec[i] = xi;
    } else {
        const int hk = b - n_head;
        const float * v = row + (size_t) n_head * 2 * hd + (size_t) n_kv * hd + (size_t) hk * hd;
        kcache[((size_t) hk * max_pos + pos) * hd + i] = __float2half(xi);
        vcache[((size_t) hk * max_pos + pos) * hd + i] = __float2half(v[i]);
    }
}

// grid (n_head, nt), 8 warps; each warp walks positions w, w+8, ...; lane owns HD/32 dims
template <int HD>
__global__ void k_attn_decode(const float * __restrict__ qkv, int stride, const half * __restrict__ kcache,
                              const half * __restrict__ vcache, float * __restrict__ out, int out_stride, const int * pos_p,
                              int max_pos, int head_off, int group, int kv_off, float scale) {
    constexpr int PER = HD / 32;
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int hk = (head_off + h) / group - kv_off;
    const int n_pos = *pos_p + t + 1;
    const float * q = qkv + (size_t) t * stride + (size_t) h * 2 * HD;
    float qr[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) qr[j] = q[lane * PER + j] * scale;
    float m = -FLT_MAX, l = 0.0f, acc[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) acc[j] = 0.0f;
    const half * kb = kcache + (size_t) hk * max_pos * HD;
    const half * vb = vcache + (size_t) hk * max_pos * HD;
    for (int p = wid; p < n_pos; p += nw) {
        const half * kt = kb + (size_t) p * HD + lane * PER;
        float sc = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) sc += qr[j] * __half2float(kt[j]);
        sc = warp_sum(sc);
        const float m_new = fmaxf(m, sc);
        const float corr = expf(m - m_new), pw = expf(sc - m_new);
        l = l * corr + pw;
        const half * vt = vb + (size_t) p * HD + lane * PER;
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[j] = acc[j] * corr + pw * __half2float(vt[j]);
        m = m_new;
    }
    __shared__ float sm[32], sl[32];
    __shared__ float sacc[32][HD];
    if (lane == 0) { sm[wid] = m; sl[wid] = l; }
#pragma unroll
    for (int j = 0; j < PER; ++j) sacc[wid][lane * PER + j] = acc[j];
    __syncthreads();
    float gm = -FLT_MAX;
    for (int w = 0; w < nw; ++w) gm = fmaxf(gm, sm[w]);
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float num = 0.0f, den = 0.0f;
        for (int w = 0; w < nw; ++w) {
            if (sl[w] == 0.0f) continue;
            const float c = expf(sm[w] - gm);
            num += sacc[w][i] * c;
            den += sl[w] * c;
        }
        out[(size_t) t * out_stride + (size_t) h * HD + i] = (num / den) * sigmoidf(q[HD + i]);
    }
}

// split-K decode attention (flash-decoding). Block = one local kv head with all of its local q heads (<= G),
// one slice of the positions, one token; 4 warps stride over the slice, lane owns 8 dims. Unnormalized partial
// results {acc[HD], m, l} per (token, q head, split) go to `part`; attn_combine merges them.
template <int HD, int G>
__global__ void __launch_bounds__(128) k_attn_split(const float * __restrict__ qkv, int stride, const half * __restrict__ kcache,
                                                    const half * __restrict__ vcache, float * __restrict__ part, const int * pos_p,
                                                    int max_pos, int n_head_l, int head_off, int group, int kv_off, float scale,
                                                    int n_chunk, const int * __restrict__ list, int list_stride,
                                                    const int * __restrict__ list_n) {
    // list (optional, sparse attention): token t attends to the cells list[t * list_stride + i], i < list_n[t]
    static_assert(HD == 256, "lane owns 8 dims");
    constexpr int NW = 4;
    // blockIdx.x = kv head * n_chunk + chunk of G q heads (GQA groups larger than G take several blocks)
    const int hk = blockIdx.x / n_chunk, hchunk = blockIdx.x % n_chunk, split = blockIdx.y, t = blockIdx.z, nsplit = gridDim.y;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n_pos = list ? list_n[t] : *pos_p + t + 1;
    const int * lt = list ? list + (size_t) t * list_stride : nullptr;
    const int chunk = (n_pos + nsplit - 1) / nsplit;
    const int p0 = split * chunk, p1 = min(n_pos, p0 + chunk);
    const int hb = max(0, (kv_off + hk) * group - head_off), he = min(n_head_l, (kv_off + hk + 1) * group - head_off);
    const int h0 = hb + hchunk * G, h1 = min(he, h0 + G);
    if (h0 >= h1) return;
    const int ng = h1 - h0;
    float q[G][8], acc[G][8], m[G], l[G];
    const float * qrow = qkv + (size_t) t * stride;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -FLT_MAX; l[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[g][j] = 0.0f; q[g][j] = g < ng ? qrow[(size_t) (h0 + g) * 2 * HD + lane * 8 + j] * scale : 0.0f; }
    }
    const half * kb = kcache + (size_t) hk * max_pos * HD + lane * 8;
    const half * vb = vcache + (size_t) hk * max_pos * HD + lane * 8;
    // the next cell's index and K/V rows are loaded while this cell is processed (same arithmetic, latency hidden)
    uint4 kn = make_uint4(0, 0, 0, 0), vn = kn;
    if (p0 + w < p1) {
        const int p = lt ? lt[p0 + w] : p0 + w;
        kn = *(const uint4 *) (kb + (size_t) p * HD);
        vn = *(const uint4 *) (vb + (size_t) p * HD);
    }
    for (int pi = p0 + w; pi < p1; pi += NW) {
        const uint4 kr = kn, vr = vn;
        if (pi + NW < p1) {
            const int p = lt ? lt[pi + NW] : pi + NW;
            kn = *(const uint4 *) (kb + (size_t) p * HD);
            vn = *(const uint4 *) (vb + (size_t) p * HD);
        }
        float kf[8], vf[8];
        const __half2 * kh = (const __half2 *) &kr, * vh = (const __half2 *) &vr;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(kh[j]), b = __half22float2(vh[j]);
            kf[2 * j] = a.x; kf[2 * j + 1] = a.y; vf[2 * j] = b.x; vf[2 * j + 1] = b.y;
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g >= ng) break;
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) s += q[g][j] * kf[j];
            s = warp_sum(s);
            const float mn = fmaxf(m[g], s), corr = __expf(m[g] - mn), pw = __expf(s - mn);
            l[g] = l[g] * corr + pw;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[g][j] = acc[g][j] * corr + pw * vf[j];
            m[g] = mn;
        }
    }
    __shared__ float sm[NW][G], sl[NW][G];
    __shared__ float sacc[NW][G][HD];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        if (lane == 0) { sm[w][g] = m[g]; sl[w][g] = l[g]; }
#pragma unroll
        for (int j = 0; j < 8; ++j) sacc[w][g][lane * 8 + j] = acc[g][j];
    }
    __syncthreads();
    for (int g = 0; g < ng; ++g) {
        float gm = -FLT_MAX;
        for (int ww = 0; ww < NW; ++ww) if (sl[ww][g] > 0.0f) gm = fmaxf(gm, sm[ww][g]);
        float * dst = part + (((size_t) t * n_head_l + h0 + g) * nsplit + split) * (HD + 2);
        for (int i = threadIdx.x; i < HD; i += blockDim.x) {
            float num = 0.0f;
            for (int ww = 0; ww < NW; ++ww) if (sl[ww][g] > 0.0f) num += sacc[ww][g][i] * __expf(sm[ww][g] - gm);
            dst[i] = num;
        }
        if (threadIdx.x == 0) {
            float den = 0.0f;
            for (int ww = 0; ww < NW; ++ww) if (sl[ww][g] > 0.0f) den += sl[ww][g] * __expf(sm[ww][g] - gm);
            dst[HD] = gm; dst[HD + 1] = den;
        }
    }
}

// grid (n_head_l, nt), HD threads: merge the splits, apply the output gate
template <int HD>
__global__ void k_attn_combine(const float * __restrict__ qkv, int stride, const float * __restrict__ part, int nsplit,
                               float * __restrict__ out, int out_stride, int n_head_l) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const float * pp = part + ((size_t) t * n_head_l + h) * nsplit * (HD + 2);
    float gm = -FLT_MAX;
    for (int s = 0; s < nsplit; ++s) if (pp[s * (HD + 2) + HD + 1] > 0.0f) gm = fmaxf(gm, pp[s * (HD + 2) + HD]);
    float num = 0.0f, den = 0.0f;
    for (int s0 = 0; s0 < nsplit; s0 += 8) {   // eight partials' loads in flight, accumulated in order as before
        float pm[8], pl[8], pv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (s0 + u < nsplit) { const float * ps = pp + (s0 + u) * (HD + 2); pm[u] = ps[HD]; pl[u] = ps[HD + 1]; pv[u] = ps[i]; }
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (s0 + u >= nsplit || pl[u] <= 0.0f) continue;
            const float c = __expf(pm[u] - gm);
            num += pv[u] * c; den += pl[u] * c;
        }
    }
    const float gate = qkv[(size_t) t * stride + (size_t) h * 2 * HD + HD + i];
    out[(size_t) t * out_stride + (size_t) h * HD + i] = (num / den) * sigmoidf(gate);
}

// tensor-core causal flash attention for prefill. Block = one local q head x 64 query tokens (4 warps x 16 rows),
// Q (pre-scaled, fp16) stays in shared memory; K/V tiles of 32 positions stream through shared memory. S = Q K^T
// and O += P V run on mma m16n8k16; the S accumulators become the P A-fragments directly.
constexpr int FA_BQ = 64, FA_BK = 32, FA_LD = 256 + 8;
template <int HD>
__global__ void __launch_bounds__(128) k_attn_fa(const float * __restrict__ qkv, int stride, const half * __restrict__ kcache,
                                                 const half * __restrict__ vcache, float * __restrict__ out, int out_stride,
                                                 const int * pos_p, int max_pos, int head_off, int group, int kv_off, float scale, int nt) {
    static_assert(HD == 256, "FA_LD assumes head_dim 256");
    extern __shared__ __align__(16) half fa_smem[];
    half * qs = fa_smem, * ks = qs + FA_BQ * FA_LD, * vs = ks + FA_BK * FA_LD;
    const int h = blockIdx.y, qt0 = blockIdx.x * FA_BQ;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int hk = (head_off + h) / group - kv_off;
    const int pos0 = *pos_p;
    for (int i = threadIdx.x; i < FA_BQ * HD / 2; i += blockDim.x) {
        const int r = i / (HD / 2), d = (i % (HD / 2)) * 2, t = qt0 + r;
        float2 v = make_float2(0.0f, 0.0f);
        if (t < nt) v = *(const float2 *) (qkv + (size_t) t * stride + (size_t) h * 2 * HD + d);
        *(__half2 *) (qs + r * FA_LD + d) = __floats2half2_rn(v.x * scale, v.y * scale);
    }
    const int n_keys = pos0 + min(nt, qt0 + FA_BQ);            // keys [0, n_keys) are visible to some row of the block
    const int wrow0 = qt0 + w * 16;                             // first token row of this warp
    const int wlast = pos0 + min(nt - 1, wrow0 + 15);           // last visible key of the warp
    const half * kb = kcache + (size_t) hk * max_pos * HD, * vb = vcache + (size_t) hk * max_pos * HD;
    float o[HD / 8][4];
#pragma unroll
    for (int i = 0; i < HD / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
    const int qp[2] = {pos0 + wrow0 + gid, pos0 + wrow0 + gid + 8};
    for (int kt0 = 0; kt0 < n_keys; kt0 += FA_BK) {
        __syncthreads();
        for (int i = threadIdx.x; i < FA_BK * HD / 8; i += blockDim.x) {
            const int r = i / (HD / 8), d = (i % (HD / 8)) * 8, p = kt0 + r;
            const bool ok = p < n_keys;
            cp_async16(ks + r * FA_LD + d, kb + (size_t) (ok ? p : 0) * HD + d, ok);
            cp_async16(vs + r * FA_LD + d, vb + (size_t) (ok ? p : 0) * HD + d, ok);
        }
        cp_async_commit();
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (kt0 > wlast || wrow0 >= nt) continue;
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < HD / 16; ++kk) {
            unsigned a[4];
            ldmatrix_x4(a, qs + (w * 16 + (lane & 15)) * FA_LD + kk * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned r[4];
                ldmatrix_x4(r, ks + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * FA_LD + kk * 16 + ((lane >> 3) & 1) * 8);
                const unsigned b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
                mma16816(s[2 * np], a, b0);
                mma16816(s[2 * np + 1], a, b1);
            }
        }
        float tmax[2] = {-1e30f, -1e30f};
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = kt0 + ni * 8 + 2 * tig + (e & 1);
                if (key > qp[e >> 1] || key >= n_keys) s[ni][e] = -INFINITY;
                tmax[e >> 1] = fmaxf(tmax[e >> 1], s[ni][e]);
            }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 2));
            const float mn = fmaxf(m[r], tmax[r]);
            corr[r] = __expf(m[r] - mn);
            m[r] = mn;
            l[r] *= corr[r];
        }
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) { s[ni][e] = __expf(s[ni][e] - m[e >> 1]); l[e >> 1] += s[ni][e]; }
#pragma unroll
        for (int i = 0; i < HD / 8; ++i) { o[i][0] *= corr[0]; o[i][1] *= corr[0]; o[i][2] *= corr[1]; o[i][3] *= corr[1]; }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const unsigned a[4] = {pack_h2(s[2 * kk][0], s[2 * kk][1]), pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                   pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int dn = 0; dn < HD / 16; ++dn) {
                unsigned r[4];
                const half * src = vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * FA_LD + dn * 16 + (lane >> 4) * 8;
                const unsigned sa = (unsigned) __cvta_generic_to_shared(src);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
                const unsigned b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
                mma16816(o[2 * dn], a, b0);
                mma16816(o[2 * dn + 1], a, b1);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
    }
#pragma unroll
    for (int e2 = 0; e2 < 2; ++e2) {
        const int t = wrow0 + gid + e2 * 8;
        if (t >= nt) continue;
        const float inv = 1.0f / l[e2];
        const float * gate = qkv + (size_t) t * stride + (size_t) h * 2 * HD + HD;
        float * dst = out + (size_t) t * out_stride + (size_t) h * HD;
#pragma unroll
        for (int i = 0; i < HD / 8; ++i) {
            const int d = i * 8 + 2 * tig;
            dst[d] = o[i][e2 * 2] * inv * sigmoidf(gate[d]);
            dst[d + 1] = o[i][e2 * 2 + 1] * inv * sigmoidf(gate[d + 1]);
        }
    }
}

// ---------------- gated delta net ----------------
// thread per channel, tokens in order; state holds the K-1 previous inputs, oldest first
__global__ void k_gdn_conv(float * in, int stride, float * st, float * snap, const float * __restrict__ w, int channels, int K, int nt) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float hist[8];
    for (int j = 0; j < K - 1; ++j) hist[j] = st[(size_t) j * channels + c];
    for (int t = 0; t < nt; ++t) {
        float * xp = in + (size_t) t * stride + c;
        const float x = *xp;
        float acc = w[(size_t) c * K + (K - 1)] * x;
        for (int j = 0; j < K - 1; ++j) acc += w[(size_t) c * K + j] * hist[j];
        for (int j = 0; j < K - 2; ++j) hist[j] = hist[j + 1];
        hist[K - 2] = x;
        *xp = silu(acc);
        if (snap && t < nt - 1)
            for (int j = 0; j < K - 1; ++j) snap[(size_t) t * (K - 1) * channels + (size_t) j * channels + c] = hist[j];
    }
    for (int j = 0; j < K - 1; ++j) st[(size_t) j * channels + c] = hist[j];
}

// grid (n_v, dv/32), 8 warps. Block owns head h and 32 value columns; warp w keeps state rows
// [w*IPW, (w+1)*IPW) of those columns in registers across all nt tokens.
template <int IPW>
__global__ void k_gdn_step(const float * __restrict__ in, int stride, int ab_off, float * __restrict__ state,
                           float * __restrict__ snap, float * __restrict__ o, int o_stride,
                           const float * __restrict__ dt_bias, const float * __restrict__ ssm_a,
                           int n_k, int n_v, int dk, int dv, float eps, int nt) {
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int j = blockIdx.y * 32 + lane;
    const int hk = h % n_k;
    const int i0 = w * IPW;
    __shared__ float sq[256], sk[256], red[8][32];
    float * S = state + (size_t) h * dk * dv;
    float sreg[IPW];
#pragma unroll
    for (int ii = 0; ii < IPW; ++ii) sreg[ii] = S[(size_t) (i0 + ii) * dv + j];
    const float dtb = dt_bias[h], sa = ssm_a[h];
    for (int t = 0; t < nt; ++t) {
        const float * row = in + (size_t) t * stride;
        const float * q = row + (size_t) hk * dk;
        const float * k = row + (size_t) n_k * dk + (size_t) hk * dk;
        const float * v = row + (size_t) 2 * n_k * dk + (size_t) h * dv;
        float qq = 0.0f, kk = 0.0f;
        for (int i = threadIdx.x; i < dk; i += blockDim.x) { const float a = q[i], b = k[i]; sq[i] = a; sk[i] = b; qq += a * a; kk += b * b; }
        qq = block_sum(qq);
        kk = block_sum(kk);
        const float qs = rsqrtf(qq + eps), ks = rsqrtf(kk + eps);
        const float g = softplusf(row[ab_off + h] + dtb) * sa;
        const float decay = expf(g);
        const float beta = sigmoidf(row[ab_off + n_v + h]);
        float part = 0.0f;
#pragma unroll
        for (int ii = 0; ii < IPW; ++ii) part += sreg[ii] * sk[i0 + ii];
        red[w][lane] = part * ks;
        __syncthreads();
        float kv = 0.0f;
        for (int ww = 0; ww < nw; ++ww) kv += red[ww][lane];
        kv *= decay;
        const float delta = (v[j] - kv) * beta;
        __syncthreads();
        float out = 0.0f;
#pragma unroll
        for (int ii = 0; ii < IPW; ++ii) {
            const float s = sreg[ii] * decay + sk[i0 + ii] * ks * delta;
            sreg[ii] = s;
            out += s * sq[i0 + ii];
        }
        red[w][lane] = out;
        __syncthreads();
        if (w == 0) {
            float acc = 0.0f;
            for (int ww = 0; ww < nw; ++ww) acc += red[ww][lane];
            o[(size_t) t * o_stride + (size_t) h * dv + j] = acc * qs * rsqrtf((float) dv);
        }
        if (snap && t < nt - 1) {
            float * Sn = snap + (size_t) t * n_v * dk * dv + (size_t) h * dk * dv;
#pragma unroll
            for (int ii = 0; ii < IPW; ++ii) Sn[(size_t) (i0 + ii) * dv + j] = sreg[ii];
        }
        __syncthreads();
    }
#pragma unroll
    for (int ii = 0; ii < IPW; ++ii) S[(size_t) (i0 + ii) * dv + j] = sreg[ii];
}

// warp per value column j of head h, the column S[:, j] (dk = 128) in registers (4 per lane): every reduction is a warp
// shuffle, so tokens advance without block barriers; grid (n_v, dv / 8), 8 warps. Same arithmetic as k_gdn_step.
__global__ void __launch_bounds__(256) k_gdn_step_w(const float * __restrict__ in, int stride, int ab_off, float * __restrict__ state,
                                                    float * __restrict__ snap, float * __restrict__ o, int o_stride,
                                                    const float * __restrict__ dt_bias, const float * __restrict__ ssm_a,
                                                    int n_k, int n_v, int dk, int dv, float eps, int nt) {
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int j = blockIdx.y * 8 + w;
    const int hk = h % n_k;
    float * S = state + (size_t) h * dk * dv;
    float s[4];
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) s[ii] = S[(size_t) (lane * 4 + ii) * dv + j];
    const float dtb = dt_bias[h], sa = ssm_a[h];
    const float rdv = rsqrtf((float) dv);
    for (int t = 0; t < nt; ++t) {
        const float * row = in + (size_t) t * stride;
        const float4 q4 = *(const float4 *) (row + (size_t) hk * dk + lane * 4);
        const float4 k4 = *(const float4 *) (row + (size_t) n_k * dk + (size_t) hk * dk + lane * 4);
        const float q[4] = {q4.x, q4.y, q4.z, q4.w}, k[4] = {k4.x, k4.y, k4.z, k4.w};
        float qq = 0.0f, kk = 0.0f, part = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) { qq += q[ii] * q[ii]; kk += k[ii] * k[ii]; part += s[ii] * k[ii]; }
        qq = warp_sum(qq); kk = warp_sum(kk); part = warp_sum(part);
        const float qs = rsqrtf(qq + eps), ks = rsqrtf(kk + eps);
        const float g = softplusf(row[ab_off + h] + dtb) * sa;
        const float decay = expf(g);
        const float beta = sigmoidf(row[ab_off + n_v + h]);
        const float kv = part * ks * decay;
        const float delta = (row[(size_t) 2 * n_k * dk + (size_t) h * dv + j] - kv) * beta;
        float out = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            s[ii] = s[ii] * decay + k[ii] * ks * delta;
            out += s[ii] * q[ii];
        }
        out = warp_sum(out);
        if (lane == 0) o[(size_t) t * o_stride + (size_t) h * dv + j] = out * qs * rdv;
        if (snap && t < nt - 1) {
            float * Sn = snap + (size_t) t * n_v * dk * dv + (size_t) h * dk * dv;
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) Sn[(size_t) (lane * 4 + ii) * dv + j] = s[ii];
        }
    }
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) S[(size_t) (lane * 4 + ii) * dv + j] = s[ii];
}

// causal conv over many tokens: thread per (channel, token), inputs from a raw copy (in is overwritten); the state
// update (last K-1 raw inputs) is done by the token-0 threads after all reads (separate kernel)
__global__ void k_gdn_conv_par(float * in, int stride, const float * __restrict__ raw, const float * __restrict__ st,
                               const float * __restrict__ w, int channels, int K, int nt) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= channels) return;
    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        const int tt = t - (K - 1) + kk;   // input position for tap kk
        const float x = tt >= 0 ? raw[(size_t) tt * channels + c] : st[(size_t) (K - 1 + tt) * channels + c];
        acc += w[(size_t) c * K + kk] * x;
    }
    in[(size_t) t * stride + c] = silu(acc);
}
__global__ void k_gdn_conv_gather(const float * __restrict__ in, int stride, float * __restrict__ raw, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c < channels) raw[(size_t) t * channels + c] = in[(size_t) t * stride + c];
}
__global__ void k_gdn_conv_state(float * st, const float * __restrict__ raw, int channels, int K, int nt) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    // new state = the last K-1 inputs (from raw, or the old state when nt < K-1)
    float h[8];
    for (int j = 0; j < K - 1; ++j) {
        const int tt = nt - (K - 1) + j;
        h[j] = tt >= 0 ? raw[(size_t) tt * channels + c] : st[(size_t) (K - 1 + tt) * channels + c];
    }
    for (int j = 0; j < K - 1; ++j) st[(size_t) j * channels + c] = h[j];
}

// grid (n_heads, nt), blockDim = dh
__global__ void k_gated_norm(float * o, int o_stride, const float * __restrict__ z, int z_stride, const float * __restrict__ w,
                             int dh, float eps) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    float * op = o + (size_t) t * o_stride + (size_t) h * dh;
    float x = op[i];
    const float ss = block_sum(x * x);
    x = x * rsqrtf(ss / dh + eps) * w[i];
    op[i] = x * silu(z[(size_t) t * z_stride + (size_t) h * dh + i]);
}

__global__ void k_silu_mul(const float * __restrict__ gu, int gu_stride, float * __restrict__ h, int h_stride, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i < n) h[(size_t) t * h_stride + i] = silu(gu[(size_t) t * gu_stride + i]) * gu[(size_t) t * gu_stride + n + i];
}

__global__ void k_argmax_pairs(const float * __restrict__ x, int xs, int n, int offset, float * out) {
    const float * xr = x + (size_t) blockIdx.x * xs;
    float best = -FLT_MAX; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (xr[i] > best) { best = xr[i]; bi = i; }
    __shared__ float sv[1024]; __shared__ int si[1024];
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && (sv[threadIdx.x + s] > sv[threadIdx.x] ||
            (sv[threadIdx.x + s] == sv[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x]))) {
            sv[threadIdx.x] = sv[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[2 * blockIdx.x] = sv[0]; ((int *) out)[2 * blockIdx.x + 1] = si[0] + offset; }
}

// top-K candidates per row: radix select of the K-th largest value (orderable float keys, MSB first), then
// gather. out[row][K] = {value, index + offset (int bits)}, unused slots {-FLT_MAX, -1}
__device__ __forceinline__ unsigned fkey(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ void k_topk(const float * __restrict__ x, int xs, int n, int offset, float * __restrict__ out, int K) {
    const float * xr = x + (size_t) blockIdx.x * xs;
    float * o = out + (size_t) blockIdx.x * K * 2;
    // radix select of the K-th largest key, 8 bits per pass (4 passes)
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, need;
    if (threadIdx.x == 0) { prefix = 0; need = K; }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned mask_hi = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        const unsigned pre = prefix;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned key = fkey(xr[i]);
            if ((key & mask_hi) == (pre & mask_hi)) atomicAdd(&hist[(key >> shift) & 255], 1u);
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
    const unsigned t = prefix;
    __shared__ int ngt, neq;
    if (threadIdx.x == 0) { ngt = 0; neq = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        if (fkey(v) > t) { const int s = atomicAdd(&ngt, 1); if (s < K) { o[2 * s] = v; ((int *) o)[2 * s + 1] = i + offset; } }
    }
    __syncthreads();
    const int base = min(ngt, K);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        if (fkey(v) == t) { const int s = base + atomicAdd(&neq, 1); if (s < K) { o[2 * s] = v; ((int *) o)[2 * s + 1] = i + offset; } }
    }
    __syncthreads();
    for (int s = base + neq + threadIdx.x; s < K; s += blockDim.x) { o[2 * s] = -FLT_MAX; ((int *) o)[2 * s + 1] = -1; }
}

// two-stage top-K for long rows (the vocabulary slice): blocks select the top K of their chunk, one block selects from
// the chunks' candidates. The selected values are those of k_topk (every global top-K element is in its chunk's top K);
// as in k_topk, which of several elements equal to the K-th value fill the last slots is unspecified.
constexpr int TOPK_CHUNKS = 64, TOPK_MAXK = 64;
__device__ float2 g_topk_cand[4][TOPK_CHUNKS * TOPK_MAXK];   // per device: [row][chunk * K + j] = {value, index bits}
// block-wide K-th largest key over m keys produced by key(i) (radix, 8 bits per pass; warp-parallel bin walk)
template <typename F>
__device__ unsigned topk_threshold(int m, int K, F key, unsigned & need_out) {
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, need;
    if (threadIdx.x == 0) { prefix = 0; need = K; }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned mask_hi = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        const unsigned pre = prefix;
        for (int i = threadIdx.x; i < m; i += blockDim.x) {
            const unsigned k = key(i);
            if ((k & mask_hi) == (pre & mask_hi)) atomicAdd(&hist[(k >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x < 32) {   // bins 255 - 8 lane - (0..7); the crossing lane resolves its bins in order
            const unsigned nd = need;
            unsigned c[8], tot = 0;
#pragma unroll
            for (int q = 0; q < 8; ++q) { c[q] = hist[255 - 8 * threadIdx.x - q]; tot += c[q]; }
            unsigned incl = tot;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const unsigned v = __shfl_up_sync(0xffffffff, incl, o); if (threadIdx.x >= o) incl += v; }
            const unsigned excl = incl - tot;
            if (excl < nd && incl >= nd) {
                unsigned acc = excl;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    if (acc + c[q] >= nd) { prefix = pre | ((unsigned) (255 - 8 * threadIdx.x - q) << shift); need = nd - acc; break; }
                    acc += c[q];
                }
            }
        }
        __syncthreads();
    }
    need_out = need;
    return prefix;
}
// stage 1: grid (TOPK_CHUNKS, rows), 256 threads
__global__ void __launch_bounds__(256) k_topk_chunk(const float * __restrict__ x, int xs, int n, int K) {
    const int row = blockIdx.y, b = blockIdx.x;
    const int c0 = (int) ((long long) n * b / TOPK_CHUNKS), c1 = (int) ((long long) n * (b + 1) / TOPK_CHUNKS), m = c1 - c0;
    __shared__ float v[(262144 / 3 + TOPK_CHUNKS) / TOPK_CHUNKS + 64];
    const float * xr = x + (size_t) row * xs + c0;
    for (int i = threadIdx.x; i < m; i += blockDim.x) v[i] = xr[i];
    __syncthreads();
    float2 * o = g_topk_cand[row] + (size_t) b * K;
    if (m <= K) {   // the whole chunk
        for (int i = threadIdx.x; i < K; i += blockDim.x) o[i] = i < m ? make_float2(v[i], __int_as_float(c0 + i)) : make_float2(-FLT_MAX, __int_as_float(-1));
        return;
    }
    unsigned need;
    const unsigned t = topk_threshold(m, K, [&](int i) { return fkey(v[i]); }, need);
    __shared__ int ngt, neq;
    if (threadIdx.x == 0) { ngt = 0; neq = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x)
        if (fkey(v[i]) > t) { const int s = atomicAdd(&ngt, 1); if (s < K) o[s] = make_float2(v[i], __int_as_float(c0 + i)); }
    __syncthreads();
    const int base = min(ngt, K);
    for (int i = threadIdx.x; i < m; i += blockDim.x)
        if (fkey(v[i]) == t) { const int s = base + atomicAdd(&neq, 1); if (s < K) o[s] = make_float2(v[i], __int_as_float(c0 + i)); }
    __syncthreads();
    for (int s = base + neq + threadIdx.x; s < K; s += blockDim.x) o[s] = make_float2(-FLT_MAX, __int_as_float(-1));
}
// stage 2: block per row (1024 threads) over the TOPK_CHUNKS * K candidates; out as k_topk
__global__ void __launch_bounds__(1024) k_topk_merge(int offset, float * __restrict__ out, int K) {
    const int row = blockIdx.x, m = TOPK_CHUNKS * K;
    const float2 * cand = g_topk_cand[row];
    float * o = out + (size_t) row * K * 2;
    // (padding slots hold -FLT_MAX with index -1: they sort below every real value)
    unsigned need;
    const unsigned t = topk_threshold(m, K, [&](int i) { return fkey(cand[i].x); }, need);
    __shared__ int ngt, neq;
    if (threadIdx.x == 0) { ngt = 0; neq = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const float2 c = cand[i];
        if (__float_as_int(c.y) >= 0 && fkey(c.x) > t) {
            const int s = atomicAdd(&ngt, 1);
            if (s < K) { o[2 * s] = c.x; ((int *) o)[2 * s + 1] = __float_as_int(c.y) + offset; }
        }
    }
    __syncthreads();
    const int base = min(ngt, K);
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const float2 c = cand[i];
        if (__float_as_int(c.y) >= 0 && fkey(c.x) == t) {
            const int s = base + atomicAdd(&neq, 1);
            if (s < K) { o[2 * s] = c.x; ((int *) o)[2 * s + 1] = __float_as_int(c.y) + offset; }
        }
    }
    __syncthreads();
    for (int s = base + neq + threadIdx.x; s < K; s += blockDim.x) { o[2 * s] = -FLT_MAX; ((int *) o)[2 * s + 1] = -1; }
}

// thread per element pair: {half2(part[2i], part[2i+1]), seq} in one 8-byte packet; readers spin on packets
__global__ void k_allreduce_add_ll16(float * x, const float * __restrict__ part, uint2 * slots, int g, int ndev, int n2,
                                     const int * counter, int call, float * ss_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) { if (ss_out) { const float t = block_sum(0.0f); if (threadIdx.x == 0) ss_out[blockIdx.x] = t; } return; }
    const unsigned seq = (unsigned) (*counter) * 1024u + (unsigned) call + 1u;
    uint2 * buf = slots + (size_t) (call & 1) * ndev * n2;
    const float2 mine = ((const float2 *) part)[i];
    const __half2 mh = __floats2half2_rn(mine.x, mine.y);
    volatile unsigned long long * dst = (volatile unsigned long long *) &buf[(size_t) g * n2 + i];
    *dst = ((unsigned long long) seq << 32) | (unsigned long long) *(const unsigned *) &mh;
    float2 acc = make_float2(0.0f, 0.0f);
    for (int dd = 0; dd < ndev; ++dd) {
        float2 f;
        if (dd == g) {
            f = __half22float2(mh);   // rounded value, so every GPU sums identical numbers
        } else {
            volatile unsigned long long * src = (volatile unsigned long long *) &buf[(size_t) dd * n2 + i];
            unsigned long long v;
            do { v = *src; } while ((unsigned) (v >> 32) != seq);
            const unsigned bits = (unsigned) (v & 0xffffffffu);
            f = __half22float2(*(const __half2 *) &bits);
        }
        acc.x += f.x; acc.y += f.y;
    }
    float2 * x2 = (float2 *) x;
    float2 xv = x2[i];
    xv.x += acc.x; xv.y += acc.y;
    x2[i] = xv;
    if (ss_out) {
        const float t = block_sum(xv.x * xv.x + xv.y * xv.y);
        if (threadIdx.x == 0) ss_out[blockIdx.x] = t;
    }
}

// bulk variant for many tokens: block = 128 threads x 8 elements; fp16 data written with 16-byte stores, one flag
// per block (after a system fence), peers' data read with 16-byte uncached loads once their flag shows up
constexpr int ARB_ELEMS = 1024;
__global__ void k_allreduce_bulk(float * x, const float * __restrict__ part, half * data, unsigned * flags, int g, int ndev,
                                 int n, const int * counter, int call) {
    const unsigned seq = (unsigned) (*counter) * 1024u + (unsigned) call + 1u;
    const int buf = call & 1, nb = gridDim.x;
    const size_t base = (size_t) blockIdx.x * ARB_ELEMS + threadIdx.x * 8;
    const bool ok = base < (size_t) n;
    float acc[8];
    if (ok) {
        const float4 p0 = *(const float4 *) (part + base), p1 = *(const float4 *) (part + base + 4);
        __half2 hv[4] = {__floats2half2_rn(p0.x, p0.y), __floats2half2_rn(p0.z, p0.w),
                         __floats2half2_rn(p1.x, p1.y), __floats2half2_rn(p1.z, p1.w)};
        *(uint4 *) (data + ((size_t) buf * ndev + g) * n + base) = *(const uint4 *) hv;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(hv[i]); acc[2 * i] = f.x; acc[2 * i + 1] = f.y; }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned *) &flags[((size_t) buf * ndev + g) * nb + blockIdx.x] = seq;
    if (threadIdx.x < ndev && threadIdx.x != g) {
        volatile unsigned * f = &flags[((size_t) buf * ndev + threadIdx.x) * nb + blockIdx.x];
        while (*f != seq) {}
    }
    __syncthreads();
    __threadfence_system();
    if (!ok) return;
    for (int d = 0; d < ndev; ++d) {
        if (d == g) continue;
        const uint4 q = __ldcv((const uint4 *) (data + ((size_t) buf * ndev + d) * n + base));
        const __half2 * hq = (const __half2 *) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(hq[i]); acc[2 * i] += f.x; acc[2 * i + 1] += f.y; }
    }
    float4 * xp = (float4 *) (x + base);
    float4 a = xp[0], b = xp[1];
    a.x += acc[0]; a.y += acc[1]; a.z += acc[2]; a.w += acc[3];
    b.x += acc[4]; b.y += acc[5]; b.z += acc[6]; b.w += acc[7];
    xp[0] = a; xp[1] = b;
}

// x[i] += own[i] + sum_j recv[j * stride + i]   (fp16 parts, 8 elements per thread)
__global__ void k_add_parts(float * x, const half * __restrict__ own, const half * __restrict__ recv, size_t stride, int nparts, int n) {
    const size_t i = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t) n) return;
    float acc[8];
    {
        const uint4 q = *(const uint4 *) (own + i);
        const __half2 * h = (const __half2 *) &q;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float2 f = __half22float2(h[k]); acc[2 * k] = f.x; acc[2 * k + 1] = f.y; }
    }
    for (int j = 0; j < nparts; ++j) {
        const uint4 q = *(const uint4 *) (recv + j * stride + i);
        const __half2 * h = (const __half2 *) &q;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float2 f = __half22float2(h[k]); acc[2 * k] += f.x; acc[2 * k + 1] += f.y; }
    }
    float4 * xp = (float4 *) (x + i);
    float4 a = xp[0], b = xp[1];
    a.x += acc[0]; a.y += acc[1]; a.z += acc[2]; a.w += acc[3];
    b.x += acc[4]; b.y += acc[5]; b.z += acc[6]; b.w += acc[7];
    xp[0] = a; xp[1] = b;
}

// LL allreduce with the hyper-connection scatter as its epilogue (see allreduce_hc_ll16)
__global__ void k_allreduce_hc_ll16(float * res, const float * __restrict__ inj, int width, int hc, const float * __restrict__ part,
                                    uint2 * slots, int g, int ndev, int n2, const int * counter, int call) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    const unsigned seq = (unsigned) (*counter) * 1024u + (unsigned) call + 1u;
    uint2 * buf = slots + (size_t) (call & 1) * ndev * n2;
    const float2 mine = ((const float2 *) part)[i];
    const __half2 mh = __floats2half2_rn(mine.x, mine.y);
    volatile unsigned long long * dst = (volatile unsigned long long *) &buf[(size_t) g * n2 + i];
    *dst = ((unsigned long long) seq << 32) | (unsigned long long) *(const unsigned *) &mh;
    float2 acc = make_float2(0.0f, 0.0f);
    for (int dd = 0; dd < ndev; ++dd) {
        float2 f;
        if (dd == g) f = __half22float2(mh);
        else {
            volatile unsigned long long * src = (volatile unsigned long long *) &buf[(size_t) dd * n2 + i];
            unsigned long long v;
            do { v = *src; } while ((unsigned) (v >> 32) != seq);
            const unsigned bits = (unsigned) (v & 0xffffffffu);
            f = __half22float2(*(const __half2 *) &bits);
        }
        acc.x += f.x; acc.y += f.y;
    }
    const int e = 2 * i, t = e / width, col = e % width;
    for (int s = 0; s < hc; ++s) {
        const float w = 2.0f / (1.0f + __expf(-inj[t * 4 + s] / hc));
        float2 * rp = (float2 *) (res + ((size_t) t * hc + s) * width + col);
        float2 r = *rp;
        r.x += acc.x * w; r.y += acc.y * w;
        *rp = r;
    }
}

// mHC variant (glm5-next): the summed block output goes straight into the 4 residual streams,
// res[t][d][e] = sum[t][e] * hcw[t][d] + sum_s hcw[t][4 + d + 4 s] * res[t][s][e]   (hcw: 20 weights per token)
__global__ void k_allreduce_mhc_ll16(float * res, const float * __restrict__ hcw, int width, const float * __restrict__ part,
                                     uint2 * slots, int g, int ndev, int n2, const int * counter, int call) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    const unsigned seq = (unsigned) (*counter) * 1024u + (unsigned) call + 1u;
    uint2 * buf = slots + (size_t) (call & 1) * ndev * n2;
    const float2 mine = ((const float2 *) part)[i];
    const __half2 mh = __floats2half2_rn(mine.x, mine.y);
    volatile unsigned long long * dst = (volatile unsigned long long *) &buf[(size_t) g * n2 + i];
    *dst = ((unsigned long long) seq << 32) | (unsigned long long) *(const unsigned *) &mh;
    float2 acc = make_float2(0.0f, 0.0f);
    for (int dd = 0; dd < ndev; ++dd) {
        float2 f;
        if (dd == g) f = __half22float2(mh);
        else {
            volatile unsigned long long * src = (volatile unsigned long long *) &buf[(size_t) dd * n2 + i];
            unsigned long long v;
            do { v = *src; } while ((unsigned) (v >> 32) != seq);
            const unsigned bits = (unsigned) (v & 0xffffffffu);
            f = __half22float2(*(const __half2 *) &bits);
        }
        acc.x += f.x; acc.y += f.y;
    }
    const int e = 2 * i, t = e / width, col = e % width;
    const float * hw = hcw + (size_t) t * 20;
    float2 * rp = (float2 *) (res + (size_t) t * 4 * width + col);
    const float2 r0 = rp[0], r1 = rp[(size_t) width / 2], r2 = rp[(size_t) width], r3 = rp[(size_t) 3 * width / 2];
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const float c0 = hw[4 + d], c1 = hw[8 + d], c2 = hw[12 + d], c3 = hw[16 + d], po = hw[d];
        rp[(size_t) d * width / 2] = make_float2(acc.x * po + c0 * r0.x + c1 * r1.x + c2 * r2.x + c3 * r3.x,
                                                acc.y * po + c0 * r0.y + c1 * r1.y + c2 * r2.y + c3 * r3.y);
    }
}

__global__ void k_incr(int * c) { *c += 1; }

} // namespace

// ---------------- launchers ----------------
#define NT_SWITCH(nt, CALL)                                   \
    switch (nt) {                                             \
        case 1: { constexpr int NT = 1; CALL; } break;        \
        case 2: { constexpr int NT = 2; CALL; } break;        \
        case 3: { constexpr int NT = 3; CALL; } break;        \
        case 4: { constexpr int NT = 4; CALL; } break;        \
        default: throw std::runtime_error("unsupported nt");  \
    }

// small K (<= 16 Q8 blocks): one warp per 16-row tile walks all k blocks, no cross-warp reduction. The block stages the
// (activated) inputs as fp16 in shared memory first and every warp loads all its weight blocks up front: otherwise the
// per-k-block global loads of x serialize at L2 latency, which dominated these small matrices
__device__ __forceinline__ float2 act2(const NormIn & nin, const float * xr, int c, float2 v) {
    if (nin.act == 1) { const float s = nin.act_scale; return make_float2(silu_f(v.x * s), silu_f(v.y * s)); }
    float2 u = *(const float2 *) (xr + c + nin.glu_off);
    if (nin.act == 3) {
        const float L = nin.act_scale;
        v.x = fminf(v.x, L); v.y = fminf(v.y, L);
        u.x = fminf(fmaxf(u.x, -L), L); u.y = fminf(fmaxf(u.y, -L), L);
    }
    return make_float2(silu_f(v.x) * u.x, silu_f(v.y) * u.y);
}
constexpr int SMALLK_LD = 256 + 4;   // half2 per token row (k <= 512), padded against bank conflicts
__global__ void k_mma_q8_smallk(const uint4 * __restrict__ wq, const half * __restrict__ ws, int n, int k,
                                const float * __restrict__ x, int xs, float * __restrict__ y, int ys, int nt, NormIn nin) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * (blockDim.x >> 5) + w;
    const int ntile = (n + 15) / 16;
    const int kb = k / 32, k2 = k / 2;
    __shared__ unsigned xh[8 * SMALLK_LD];
    for (int i = threadIdx.x; i < nt * k2; i += blockDim.x) {
        const int t = i / k2, c = 2 * (i % k2);
        const float * xr = x + (size_t) t * xs;
        float2 v = *(const float2 *) (xr + c);
        if (nin.act) v = act2(nin, xr, c, v);
        xh[t * SMALLK_LD + c / 2] = pack_h2(v.x, v.y);
    }
    __syncthreads();
    if (tile >= ntile) return;
    const uint4 * tq = wq + (size_t) tile * kb * 32 + lane;
    const half * ts = ws + (size_t) tile * kb * 16;
    const bool tok_ok = gid < nt;
    const unsigned * xt = xh + (tok_ok ? gid : 0) * SMALLK_LD;
    uint4 qv[16];
    half2 sv[16];
#pragma unroll
    for (int b = 0; b < 16; ++b)
        if (b < kb) { qv[b] = __ldg(tq + (size_t) b * 32); sv[b] = __halves2half2(ts[(size_t) b * 16 + gid], ts[(size_t) b * 16 + gid + 8]); }
    float acc[4] = {0, 0, 0, 0};
#pragma unroll
    for (int b = 0; b < 16; ++b) {
        if (b >= kb) break;
        const unsigned qw[4] = {qv[b].x, qv[b].y, qv[b].z, qv[b].w};
        float tmp[4] = {0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned a[4];
            a[0] = i8x2_to_h2(qw[2 * ks], 0x5140);
            a[1] = i8x2_to_h2(qw[2 * ks], 0x7362);
            a[2] = i8x2_to_h2(qw[2 * ks + 1], 0x5140);
            a[3] = i8x2_to_h2(qw[2 * ks + 1], 0x7362);
            const int c2 = b * 16 + ks * 8 + tig;   // half2 index of column b*32 + ks*16 + 2 tig
            unsigned bb[2] = {0, 0};
            if (tok_ok) { bb[0] = xt[c2]; bb[1] = xt[c2 + 4]; }
            mma16816(tmp, a, bb);
        }
        acc[0] += tmp[0] * __low2float(sv[b]); acc[1] += tmp[1] * __low2float(sv[b]);
        acc[2] += tmp[2] * __high2float(sv[b]); acc[3] += tmp[3] * __high2float(sv[b]);
    }
    const int r0 = tile * 16 + gid, t0 = 2 * tig;
    auto put = [&](int row, int t, float v) { if (row < n && t < nt) y[(size_t) t * ys + row] = v; };
    put(r0, t0, acc[0]); put(r0, t0 + 1, acc[1]);
    put(r0 + 8, t0, acc[2]); put(r0 + 8, t0 + 1, acc[3]);
}

// hyper-connection "up" + mixing (hc = 4): rows of W in the order e * 4 + s (stream s of element e), so a lane's 4 row
// partners (stream 0..3 of one e) sit on lanes 4 and 8 apart; epilogue: mixed[t][e] = sum_s xn[t][s n + e] *
// sigmoid(gate) / 4 (the separate mixing kernel and the gate round trip go away)
__global__ void k_mma_q8_hcmix(const uint4 * __restrict__ wq, const half * __restrict__ ws, int rows, int k, const float * __restrict__ x,
                               int xs, int nt, NormIn nin, const float * __restrict__ xn, int xns, float * __restrict__ mixed, int ms, int n_embd) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * (blockDim.x >> 5) + w;
    const int ntile = (rows + 15) / 16;
    const int kb = k / 32, k2 = k / 2;
    __shared__ unsigned xh[8 * SMALLK_LD];
    for (int i = threadIdx.x; i < nt * k2; i += blockDim.x) {
        const int t = i / k2, c = 2 * (i % k2);
        const float * xr = x + (size_t) t * xs;
        float2 v = *(const float2 *) (xr + c);
        if (nin.act) v = act2(nin, xr, c, v);
        xh[t * SMALLK_LD + c / 2] = pack_h2(v.x, v.y);
    }
    __syncthreads();
    if (tile >= ntile) return;
    const uint4 * tq = wq + (size_t) tile * kb * 32 + lane;
    const half * ts = ws + (size_t) tile * kb * 16;
    const bool tok_ok = gid < nt;
    const unsigned * xt = xh + (tok_ok ? gid : 0) * SMALLK_LD;
    uint4 qv[16];
    half2 sv[16];
#pragma unroll
    for (int b = 0; b < 16; ++b)
        if (b < kb) { qv[b] = __ldg(tq + (size_t) b * 32); sv[b] = __halves2half2(ts[(size_t) b * 16 + gid], ts[(size_t) b * 16 + gid + 8]); }
    float acc[4] = {0, 0, 0, 0};
#pragma unroll
    for (int b = 0; b < 16; ++b) {
        if (b >= kb) break;
        const unsigned qw[4] = {qv[b].x, qv[b].y, qv[b].z, qv[b].w};
        float tmp[4] = {0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned a[4];
            a[0] = i8x2_to_h2(qw[2 * ks], 0x5140);
            a[1] = i8x2_to_h2(qw[2 * ks], 0x7362);
            a[2] = i8x2_to_h2(qw[2 * ks + 1], 0x5140);
            a[3] = i8x2_to_h2(qw[2 * ks + 1], 0x7362);
            const int c2 = b * 16 + ks * 8 + tig;
            unsigned bb[2] = {0, 0};
            if (tok_ok) { bb[0] = xt[c2]; bb[1] = xt[c2 + 4]; }
            mma16816(tmp, a, bb);
        }
        acc[0] += tmp[0] * __low2float(sv[b]); acc[1] += tmp[1] * __low2float(sv[b]);
        acc[2] += tmp[2] * __high2float(sv[b]); acc[3] += tmp[3] * __high2float(sv[b]);
    }
    // acc[i]: row tile * 16 + gid (+ 8 for i >= 2), token 2 tig + (i & 1); row r -> element r / 4, stream r % 4 (= gid % 4)
    const int sidx = gid & 3;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = tile * 16 + gid + (i >= 2 ? 8 : 0), t = 2 * tig + (i & 1), e = r >> 2;
        float v = 0.0f;
        if (r < rows && t < nt) v = xn[(size_t) t * xns + (size_t) sidx * n_embd + e] / (1.0f + expf(-acc[i]));
        v += __shfl_xor_sync(0xffffffff, v, 4);
        v += __shfl_xor_sync(0xffffffff, v, 8);
        if (sidx == 0 && r < rows && t < nt) mixed[(size_t) t * ms + e] = v * 0.25f;
    }
}

// split-K scratch per device (gemv_init): tile partials and per-tile counters
// two scratch slots per device: work issued on a second stream (gemv_scratch_slot(1)) cannot collide with the main one's
namespace { struct KSplit { float * part = nullptr; unsigned * cnt = nullptr; int gw = 0; }; KSplit g_kslots[16][2]; int g_kslot = 0; }
#define g_ksplit_dev(dev) g_kslots[dev][g_kslot]
void gemv_scratch_slot(int slot) { g_kslot = slot & 1; }
constexpr size_t KSPLIT_CNT = 8192;   // tile counters
constexpr int KSPLIT_TILES = 1024, KSPLIT_MAX = 16;
// split-K target: blocks per GEMV launch (HYPER_GEMV_TARGET; 0 = only for matrices with few row tiles)
static const int g_ksplit_blocks = getenv("HYPER_KSPLIT_BLOCKS") ? atoi(getenv("HYPER_KSPLIT_BLOCKS")) : 320;
static const int g_ksplit_minkb = getenv("HYPER_KSPLIT_MINKB") ? atoi(getenv("HYPER_KSPLIT_MINKB")) : 16;
static const int g_gemv_target = getenv("HYPER_GEMV_TARGET") ? atoi(getenv("HYPER_GEMV_TARGET")) : 0;
void gemv_init(int dev) {
    if (dev < 0 || dev >= 16 || g_kslots[dev][0].part) return;
    int nb = 0, sms = 0;   // resident warps of the balanced GEMV
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_mma_q8_bal, 256, 0);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    for (auto & ks : g_kslots[dev]) {
        cudaMalloc(&ks.part, (size_t) KSPLIT_TILES * KSPLIT_MAX * 128 * sizeof(float));
        cudaMalloc(&ks.cnt, KSPLIT_CNT * sizeof(unsigned));
        cudaMemset(ks.cnt, 0, KSPLIT_CNT * sizeof(unsigned));
        ks.gw = nb * sms * 8;
    }
}

void gemv_q8(const Q8W & W, const float * x, int xs, float * y, int ys, const float * add, int nt, cudaStream_t s,
             const NormIn & nin) {
    if (nt < 1 || nt > 8) throw std::runtime_error("gemv_q8: nt must be 1..8");
    const int tiles = (W.n + 15) / 16, kb = W.k / 32;
    if (kb <= 16 && !add && !nin.w) {   // short rows: a warp per tile
        k_mma_q8_smallk<<<(tiles + 7) / 8, 256, 0, s>>>(W.q, W.s, W.n, W.k, x, xs, y, ys, nt, nin);
        return;
    }
    int P = 1;
    int dev = 0;
    cudaGetDevice(&dev);
    static const bool no_bal = getenv("HYPER_GEMV_NOBAL") != nullptr;
    if (!no_bal && dev < 16 && g_ksplit_dev(dev).part && g_ksplit_dev(dev).gw > 0) {
        // balanced units when a block per tile would leave a large partial wave (resident blocks: g_ksplit_dev(dev).gw / 8)
        const int resident = g_ksplit_dev(dev).gw / 8;
        const double waves = (double) tiles / resident;
        const double eff = waves / std::ceil(waves);
        // (measured: gemvbench on the GLM / 27B shapes, k >= 4096; shorter rows make the units too small to pay for the partials)
        if (eff < 0.85 && waves < 8.0 && tiles >= 160 && kb >= 128) {
            static const int minkb = getenv("HYPER_BAL_MINKB") ? atoi(getenv("HYPER_BAL_MINKB")) : 8;
            static const double rounds = getenv("HYPER_BAL_ROUNDS") ? atof(getenv("HYPER_BAL_ROUNDS")) : 1.0;
            int upt = std::max(1, std::min({32, kb / std::max(1, minkb), (int) ((rounds * g_ksplit_dev(dev).gw) / tiles)}));
            while (upt > 1 && (size_t) tiles * upt > (size_t) KSPLIT_TILES * KSPLIT_MAX) --upt;
            if ((size_t) tiles <= KSPLIT_CNT && upt >= 3) {   // (short rows: units too small to pay for the partials)
                const int units = tiles * upt, blocks = std::min(resident, (units + 7) / 8);
                k_mma_q8_bal<<<blocks, 256, 0, s>>>(W.q, W.s, W.n, W.k, x, xs, y, ys, add, nt, nin, upt, g_ksplit_dev(dev).part, g_ksplit_dev(dev).cnt);
                return;
            }
        }
    }
    if (dev < 16 && g_ksplit_dev(dev).part && tiles <= KSPLIT_TILES && tiles < 160)   // few row tiles: spread K over more SMs
        P = std::max(1, std::min({KSPLIT_MAX, g_ksplit_blocks / tiles, kb / g_ksplit_minkb}));
    if (g_gemv_target > 0 && dev < 16 && g_ksplit_dev(dev).part && tiles <= KSPLIT_TILES && tiles < g_gemv_target)   // medium: enough blocks in flight
        P = std::max(P, std::min({KSPLIT_MAX, g_gemv_target / tiles, kb / 16}));
    k_mma_q8<<<dim3(tiles, P), 256, 0, s>>>(W.q, W.s, W.n, W.k, x, xs, y, ys, add, nt, nin,
                                           P > 1 ? g_ksplit_dev(dev).part : nullptr, P > 1 ? g_ksplit_dev(dev).cnt : nullptr);
}

void gemv_q8_hcmix(const Q8W & W, const float * x, int xs, const NormIn & nin, const float * xn, int xns, float * mixed, int ms,
                   int n_embd, int nt, cudaStream_t s) {
    if (nt < 1 || nt > 8 || W.k / 32 > 16 || W.n != 4 * n_embd) throw std::runtime_error("gemv_q8_hcmix: sizes");
    const int tiles = (W.n + 15) / 16;
    k_mma_q8_hcmix<<<(tiles + 7) / 8, 256, 0, s>>>(W.q, W.s, W.n, W.k, x, xs, nt, nin, xn, xns, mixed, ms, n_embd);
}
void gemv_kq(const KQW & W, const float * x, int xs, float * y, int ys, int nt, cudaStream_t s, const NormIn & nin) {
    if (nt < 1 || nt > 8 || W.k % 32 || (W.k / 32) % W.dg) throw std::runtime_error("gemv_kq: sizes");
    if (nin.w) throw std::runtime_error("gemv_kq: fused input norm unsupported");
    const int tiles = (W.n + 15) / 16, kb = W.k / 32;
    int dev = 0;
    cudaGetDevice(&dev);
    int P = 1;   // (split-K as gemv_q8: few row tiles)
    if (dev < 16 && g_ksplit_dev(dev).part && tiles <= KSPLIT_TILES && tiles < 160)
        P = std::max(1, std::min({KSPLIT_MAX, g_ksplit_blocks / tiles, kb / g_ksplit_minkb}));
    float * kp = P > 1 ? g_ksplit_dev(dev).part : nullptr;
    unsigned * kc = P > 1 ? g_ksplit_dev(dev).cnt : nullptr;
    if (W.type == KQ::Q4K) k_mma_kq<KQ::Q4K><<<dim3(tiles, P), 256, 0, s>>>(W, x, xs, y, ys, nt, nin, kp, kc);
    else k_mma_kq<KQ::Q6K><<<dim3(tiles, P), 256, 0, s>>>(W, x, xs, y, ys, nt, nin, kp, kc);
}
void deq_kq_f16(const KQW & W, int r0, int r1, half * out, cudaStream_t s) {
    if (r0 % 16) throw std::runtime_error("deq_kq_f16: r0 must be a multiple of 16");
    const int tile0 = r0 / 16, ntile = (r1 - r0 + 15) / 16;
    const size_t nthr = (size_t) ntile * (W.k / 32) * 32;
    const unsigned blocks = (unsigned) ((nthr + 255) / 256);
    if (W.type == KQ::Q4K) k_deq_kq_f16<KQ::Q4K><<<blocks, 256, 0, s>>>(W, tile0, ntile, out, r1 - r0);
    else k_deq_kq_f16<KQ::Q6K><<<blocks, 256, 0, s>>>(W, tile0, ntile, out, r1 - r0);
}
// GGUF super-block fields: Q4_K (144 B: d, dmin, scales[12], qs[128]), Q6_K (210 B: ql[128], qh[64], scales[16], d)
static void q4k_scale_min(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
int repack_kq(KQ type, const uint8_t * rows, size_t rb, int n, const std::vector<int> & kblocks, KQHost & o) {
    const int kb = (int) kblocks.size(), k = kb * 32, ntile = (n + 15) / 16;
    int dg = kb % 8 == 0 ? 8 : 1;   // whole aligned super-blocks: one (d, dmin) per 8 blocks
    for (int j = 0; j < kb && dg == 8; ++j) if (kblocks[j] != kblocks[j / 8 * 8] + j % 8 || kblocks[j / 8 * 8] % 8) dg = 1;
    const int ng = kb / dg;
    const size_t bb = type == KQ::Q4K ? 144 : 210;
    o.lo.assign((size_t) ntile * kb * 32, make_uint2(0, 0));
    o.hi.assign(type == KQ::Q6K ? (size_t) ntile * kb * 32 : 0, 0u);
    o.scm.assign(type == KQ::Q4K ? (size_t) ntile * kb * 16 : 0, 0);
    o.sc6.assign(type == KQ::Q6K ? (size_t) ntile * kb * 32 : 0, 0);
    o.d.assign((size_t) ntile * ng * 16, __halves2half2(__float2half(0.0f), __float2half(0.0f)));
    // value (0..15 or 0..63, unbiased) of row r, matrix column c
    auto val = [&](int r, int c) -> int {
        const int sc = kblocks[c / 32] * 32 + c % 32;
        const uint8_t * blk = rows + (size_t) r * rb + (size_t) (sc / 256) * bb;
        const int e = sc % 256;
        if (type == KQ::Q4K) { const int j = e / 32, l = e % 32; return (blk[16 + 32 * (j / 2) + l] >> (4 * (j % 2))) & 0xF; }
        const int h = e / 128, rr = e % 128, qd = rr / 32, l = rr % 32;
        const uint8_t lb = blk[h * 64 + l + (qd % 2) * 32], hb = blk[128 + h * 32 + l];
        return (qd < 2 ? lb & 0xF : lb >> 4) | (((hb >> (2 * qd)) & 3) << 4);
    };
#pragma omp parallel for schedule(static)
    for (int t = 0; t < ntile; ++t) {
        for (int r = 0; r < 16; ++r) {   // scales
            const int row = t * 16 + r;
            if (row >= n) continue;
            for (int b = 0; b < kb; ++b) {
                const int src = kblocks[b], j = src % 8;
                const uint8_t * blk = rows + (size_t) row * rb + (size_t) (src / 8) * bb;
                if (b % dg == 0) {
                    half dd, dm;
                    if (type == KQ::Q4K) { memcpy(&dd, blk, 2); memcpy(&dm, blk + 2, 2); }
                    else { memcpy(&dd, blk + 208, 2); dm = __float2half(0.0f); }
                    o.d[((size_t) t * ng + b / dg) * 16 + r] = __halves2half2(dd, dm);
                }
                const size_t tb = (size_t) t * kb + b;
                if (type == KQ::Q4K) { int sc, m; q4k_scale_min(j, blk + 4, sc, m); o.scm[tb * 16 + r] = (uint16_t) (sc | (m << 8)); }
                else for (int ks = 0; ks < 2; ++ks) o.sc6[(tb * 2 + ks) * 16 + r] = (int8_t) blk[192 + j * 2 + ks];
            }
        }
        for (int b = 0; b < kb; ++b)
            for (int lane = 0; lane < 32; ++lane) {
                const int gid = lane >> 2, tig = lane & 3;
                unsigned wv[2] = {0, 0}, hv = 0;
                for (int ks = 0; ks < 2; ++ks)
                    for (int p = 0; p < 4; ++p)
                        for (int e = 0; e < 2; ++e) {
                            const int row = t * 16 + gid + ((p & 1) ? 8 : 0);
                            const int col = b * 32 + ks * 16 + 2 * tig + (p >= 2 ? 8 : 0) + e;
                            const int v = row < n ? val(row, col) : (type == KQ::Q6K ? 32 : 0);
                            wv[ks] |= (unsigned) (v & 0xF) << (4 * p + 16 * e);
                            if (type == KQ::Q6K) hv |= (unsigned) ((v >> 4) & 3) << (2 * (ks * 4 + p) + 16 * e);
                        }
                const size_t li = ((size_t) t * kb + b) * 32 + lane;
                o.lo[li] = make_uint2(wv[0], wv[1]);
                if (type == KQ::Q6K) o.hi[li] = hv;
            }
    }
    return dg;
}
void repack_q8_frag(const int8_t * qs, const half * d, int n, int k, uint8_t * fq, half * fs) {
    const int ntile = (n + 15) / 16, kb = k / 32;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < ntile; ++t)
        for (int b = 0; b < kb; ++b) {
            uint8_t * tile = fq + ((size_t) t * kb + b) * 512;
            for (int lane = 0; lane < 32; ++lane) {
                const int gid = lane >> 2, tig = lane & 3;
                uint8_t * o = tile + lane * 16;
                for (int ks = 0; ks < 2; ++ks) {
                    const int cb = b * 32 + ks * 16 + 2 * tig;
                    const int rows[4] = {gid, gid + 8, gid, gid + 8};
                    const int cols[4] = {cb, cb, cb + 8, cb + 8};
                    for (int p = 0; p < 4; ++p)
                        for (int e = 0; e < 2; ++e) {
                            const int r = t * 16 + rows[p];
                            o[ks * 8 + p * 2 + e] = r < n ? (uint8_t) qs[(size_t) r * k + cols[p] + e] : 0;
                        }
                }
            }
            for (int r = 0; r < 16; ++r) {
                const int rr = t * 16 + r;
                fs[((size_t) t * kb + b) * 16 + r] = rr < n ? d[(size_t) rr * kb + b] : __float2half(0.0f);
            }
        }
}

void gemv_bf16(const BF16W & W, const float * x, int xs, float * y, int ys, const float * add, int nt, cudaStream_t s,
               const NormIn & nin) {
    if (nt < 1 || nt > 8) throw std::runtime_error("gemv_bf16: nt must be 1..8");
    const int tiles = (W.n + 15) / 16, nks = W.k / 16;
    int P = 1, dev = 0;
    cudaGetDevice(&dev);
    if (dev < 16 && g_ksplit_dev(dev).part && tiles <= KSPLIT_TILES && tiles < 160)
        P = std::max(1, std::min({KSPLIT_MAX, 320 / tiles, nks / 32}));
    k_mma_f16<<<dim3(tiles, P), 256, 0, s>>>(W.q, W.n, W.k, x, xs, y, ys, add, nt, nin,
                                            P > 1 ? g_ksplit_dev(dev).part : nullptr, P > 1 ? g_ksplit_dev(dev).cnt : nullptr);
}

void to_half(const float * x, int xs, const float * w, int k, float eps, half * xh, int nt, cudaStream_t s) {
    k_to_half<<<nt, 256, 0, s>>>(x, xs, w, k, eps, xh);
}
// token tile by chunk size: padding a short chunk to 128 tokens would cost a full 128-token tile
template <typename Frag>
static void gemm_launch(const uint4 * q, const half * sc, int n, int k, const half * xh, int T, float * y, int ys, const float * add,
                        cudaStream_t s) {
    const int gy = ((n + 15) / 16 + 7) / 8;
    if (T <= 32)      k_gemm<Frag, 2><<<dim3((T + 31) / 32, gy), 256, 0, s>>>(q, sc, n, k, xh, T, y, ys, add);
    else if (T <= 64) k_gemm<Frag, 4><<<dim3((T + 63) / 64, gy), 256, 0, s>>>(q, sc, n, k, xh, T, y, ys, add);
    else              k_gemm<Frag, 8><<<dim3((T + 127) / 128, gy), 256, 0, s>>>(q, sc, n, k, xh, T, y, ys, add);
}
void gemm_q8(const Q8W & W, const half * xh, int T, float * y, int ys, const float * add, cudaStream_t s) {
    gemm_launch<FragQ8>(W.q, W.s, W.n, W.k, xh, T, y, ys, add, s);
}
void gemm_f16(const BF16W & W, const half * xh, int T, float * y, int ys, const float * add, cudaStream_t s) {
    if (W.k % 32) throw std::runtime_error("gemm_f16: k must be a multiple of 32");
    gemm_launch<FragF16>(W.q, nullptr, W.n, W.k, xh, T, y, ys, add, s);
}

void repack_f32_frag(const float * w, int n, int k, size_t row_stride, uint8_t * out) {
    const int ntile = (n + 15) / 16, nks = k / 16;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < ntile; ++t)
        for (int ks = 0; ks < nks; ++ks) {
            half * tile = (half *) (out + ((size_t) t * nks + ks) * 512);
            for (int lane = 0; lane < 32; ++lane) {
                const int gid = lane >> 2, tig = lane & 3;
                const int cb = ks * 16 + 2 * tig;
                const int rows[4] = {gid, gid + 8, gid, gid + 8};
                const int cols[4] = {cb, cb, cb + 8, cb + 8};
                for (int p = 0; p < 4; ++p)
                    for (int e = 0; e < 2; ++e) {
                        const int r = t * 16 + rows[p];
                        tile[lane * 8 + p * 2 + e] = r < n ? __float2half(w[(size_t) r * row_stride + cols[p] + e]) : __float2half(0.0f);
                    }
            }
        }
}

void repack_bf16_frag(const uint16_t * w, int n, int k, size_t row_stride, uint8_t * out) {
    const int ntile = (n + 15) / 16, nks = k / 16;
    auto bf2h = [](uint16_t b) { uint32_t bits = (uint32_t) b << 16; float f; memcpy(&f, &bits, 4); return __float2half(f); };
#pragma omp parallel for schedule(static)
    for (int t = 0; t < ntile; ++t)
        for (int ks = 0; ks < nks; ++ks) {
            half * tile = (half *) (out + ((size_t) t * nks + ks) * 512);
            for (int lane = 0; lane < 32; ++lane) {
                const int gid = lane >> 2, tig = lane & 3;
                const int cb = ks * 16 + 2 * tig;
                const int rows[4] = {gid, gid + 8, gid, gid + 8};
                const int cols[4] = {cb, cb, cb + 8, cb + 8};
                for (int p = 0; p < 4; ++p)
                    for (int e = 0; e < 2; ++e) {
                        const int r = t * 16 + rows[p];
                        tile[lane * 8 + p * 2 + e] = r < n ? bf2h(w[(size_t) r * row_stride + cols[p] + e]) : __float2half(0.0f);
                    }
            }
        }
}

void rmsnorm(const float * x, int xs, const float * w, float * y, int ys, int n, int nt, float eps, cudaStream_t s) {
    k_rmsnorm<<<nt, 1024, 0, s>>>(x, xs, w, y, ys, n, eps);
}
void sumsq(const float * x, int xs, int n, int nt, float * ss, int nss, cudaStream_t s) {
    k_sumsq<<<nt, 1024, 0, s>>>(x, xs, n, ss, nss);
}
void attn_prep(float * qkv, int stride, const float * qnorm, const float * knorm, half * kcache, half * vcache,
               const int * pos, int max_pos, int n_head, int n_kv, int hd, int n_rot, float rope_base, float eps,
               int nt, cudaStream_t s) {
    k_attn_prep<<<dim3(n_head + n_kv, nt), hd, 0, s>>>(qkv, stride, qnorm, knorm, kcache, vcache, pos, max_pos, n_head, n_kv,
                                                       hd, n_rot, rope_base, eps);
}
void attn_decode(const float * qkv, int stride, const half * kcache, const half * vcache, float * out, int out_stride,
                 const int * pos, int max_pos, int n_head, int n_kv, int head_off, int group, int kv_off, int hd,
                 float scale, int nt, cudaStream_t s) {
    (void) n_kv;
    if (hd != 256) throw std::runtime_error("attn_decode: only head_dim 256 is instantiated");
    k_attn_decode<256><<<dim3(n_head, nt), 256, 0, s>>>(qkv, stride, kcache, vcache, out, out_stride, pos, max_pos,
                                                        head_off, group, kv_off, scale);
}
void attn_prefill(const float * qkv, int stride, const half * kcache, const half * vcache, float * out, int out_stride,
                  const int * pos, int max_pos, int n_head, int head_off, int group, int kv_off, int hd, float scale, int nt,
                  cudaStream_t s) {
    if (hd != 256) throw std::runtime_error("attn_prefill: only head_dim 256 is instantiated");
    const size_t smem = (size_t) (FA_BQ + 2 * FA_BK) * FA_LD * sizeof(half);
    static bool attr_set[16] = {};
    int dev = 0; cudaGetDevice(&dev);
    if (!attr_set[dev]) {
        cudaFuncSetAttribute(k_attn_fa<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr_set[dev] = true;
    }
    k_attn_fa<256><<<dim3((nt + FA_BQ - 1) / FA_BQ, n_head), 128, smem, s>>>(qkv, stride, kcache, vcache, out, out_stride, pos, max_pos,
                                                                           head_off, group, kv_off, scale, nt);
}
int attn_nsplit(int n_kv, int nt) { return std::max(1, std::min(64, 256 / (n_kv * nt))); }
size_t attn_part_floats(int n_head, int n_kv, int nt, int hd) { return (size_t) nt * n_head * attn_nsplit(n_kv, nt) * (hd + 2); }
void attn_split(const float * qkv, int stride, const half * kcache, const half * vcache, float * part, float * out, int out_stride,
                const int * pos, int max_pos, int n_head, int n_kv, int head_off, int group, int kv_off, int hd,
                float scale, int nt, cudaStream_t s, const int * list, int list_stride, const int * list_n) {
    if (hd != 256) throw std::runtime_error("attn_split: only head_dim 256 is instantiated");
    const int ns = attn_nsplit(n_kv, nt);
    const int n_chunk = (std::min(group, n_head) + 5) / 6;   // local q heads of one kv head <= min(group, n_head)
    k_attn_split<256, 6><<<dim3(n_kv * n_chunk, ns, nt), 128, 0, s>>>(qkv, stride, kcache, vcache, part, pos, max_pos, n_head, head_off,
                                                                     group, kv_off, scale, n_chunk, list, list_stride, list_n);
    k_attn_combine<256><<<dim3(n_head, nt), 256, 0, s>>>(qkv, stride, part, ns, out, out_stride, n_head);
}
void gdn_conv(float * in, int stride, float * conv_state, float * conv_snap, const float * conv_w, int channels, int K,
              int nt, cudaStream_t s, float * raw) {
    if (K > 9) throw std::runtime_error("gdn_conv: kernel too large");
    if (raw && !conv_snap && nt > 8) {   // many tokens: parallel over (channel, token) from a raw copy
        const dim3 grid((channels + 255) / 256, nt);
        k_gdn_conv_gather<<<grid, 256, 0, s>>>(in, stride, raw, channels);
        k_gdn_conv_par<<<grid, 256, 0, s>>>(in, stride, raw, conv_state, conv_w, channels, K, nt);
        k_gdn_conv_state<<<(channels + 255) / 256, 256, 0, s>>>(conv_state, raw, channels, K, nt);
        return;
    }
    k_gdn_conv<<<(channels + 255) / 256, 256, 0, s>>>(in, stride, conv_state, conv_snap, conv_w, channels, K, nt);
}
void gdn_step(const float * in, int stride, int ab_off, float * state, float * state_snap, float * o, int o_stride,
              const float * dt_bias, const float * ssm_a, int n_k, int n_v, int dk, int dv, float eps, int nt,
              cudaStream_t s) {
    if (dk != 128 || dv % 32) throw std::runtime_error("gdn_step: expects dk = 128");
    if (dk == 128 && dv % 8 == 0 && nt > 8 && !getenv("HYPER_GDN_OLD"))   // many tokens: warp per column
        k_gdn_step_w<<<dim3(n_v, dv / 8), 256, 0, s>>>(in, stride, ab_off, state, state_snap, o, o_stride, dt_bias, ssm_a,
                                                       n_k, n_v, dk, dv, eps, nt);
    else
        k_gdn_step<16><<<dim3(n_v, dv / 32), 256, 0, s>>>(in, stride, ab_off, state, state_snap, o, o_stride, dt_bias, ssm_a,
                                                          n_k, n_v, dk, dv, eps, nt);
}
void gated_norm(float * o, int o_stride, const float * z, int z_stride, const float * w, int n_heads, int dh, float eps,
                int nt, cudaStream_t s) {
    k_gated_norm<<<dim3(n_heads, nt), dh, 0, s>>>(o, o_stride, z, z_stride, w, dh, eps);
}
void silu_mul(const float * gu, int gu_stride, float * h, int h_stride, int n, int nt, cudaStream_t s) {
    k_silu_mul<<<dim3((n + 255) / 256, nt), 256, 0, s>>>(gu, gu_stride, h, h_stride, n);
}
void argmax_pairs(const float * x, int xs, int n, int offset, float * out, int nt, cudaStream_t s) {
    k_argmax_pairs<<<nt, 1024, 0, s>>>(x, xs, n, offset, out);
}
void allreduce_add_ll16(float * x, const float * part, uint2 * slots, int g, int ndev, int n,
                        const int * counter, int call, cudaStream_t s, float * ss_out) {
    const int n2 = n / 2;
    // AR_SS_SPAN elements per block -> 256 threads x 2 elements
    k_allreduce_add_ll16<<<(n2 + 255) / 256, 256, 0, s>>>(x, part, slots, g, ndev, n2, counter, call, ss_out);
}
void allreduce_add_bulk(float * x, const float * part, half * data, unsigned * flags, int g, int ndev, int n,
                        const int * counter, int call, cudaStream_t s) {
    if (n % 8) throw std::runtime_error("allreduce_add_bulk: n must be a multiple of 8");
    k_allreduce_bulk<<<(n + ARB_ELEMS - 1) / ARB_ELEMS, ARB_ELEMS / 8, 0, s>>>(x, part, data, flags, g, ndev, n, counter, call);
}
void add_parts(float * x, const half * own, const half * recv, size_t stride, int nparts, int n, cudaStream_t s) {
    if (n % 8) throw std::runtime_error("add_parts: n must be a multiple of 8");
    k_add_parts<<<(n / 8 + 255) / 256, 256, 0, s>>>(x, own, recv, stride, nparts, n);
}
void topk_pairs(const float * x, int xs, int n, int offset, float * out, int K, int nt, cudaStream_t s) {
    static const bool one = getenv("HYPER_TOPK_ONEBLOCK") != nullptr;
    if (!one && nt <= 4 && K <= TOPK_MAXK && n >= 4 * TOPK_CHUNKS * K && n <= 262144 / 3 + TOPK_CHUNKS) {
        k_topk_chunk<<<dim3(TOPK_CHUNKS, nt), 256, 0, s>>>(x, xs, n, K);
        k_topk_merge<<<nt, 1024, 0, s>>>(offset, out, K);
        return;
    }
    k_topk<<<nt, 1024, 0, s>>>(x, xs, n, offset, out, K);
}
void allreduce_mhc_ll16(float * res, const float * hcw, int width, const float * part, uint2 * slots, int g, int ndev, int n,
                        const int * counter, int call, cudaStream_t s) {
    if (n & 1 || width & 3) throw std::runtime_error("allreduce_mhc_ll16: n must be even, width a multiple of 4");
    const int n2 = n / 2;
    k_allreduce_mhc_ll16<<<(n2 + 255) / 256, 256, 0, s>>>(res, hcw, width, part, slots, g, ndev, n2, counter, call);
}
void allreduce_hc_ll16(float * res, const float * inj, int width, int hc, const float * part, uint2 * slots, int g, int ndev, int n,
                       const int * counter, int call, cudaStream_t s) {
    const int n2 = n / 2;
    k_allreduce_hc_ll16<<<(n2 + 255) / 256, 256, 0, s>>>(res, inj, width, hc, part, slots, g, ndev, n2, counter, call);
}
void incr_counter(int * c, cudaStream_t s) { k_incr<<<1, 1, 0, s>>>(c); }

} // namespace hyper
