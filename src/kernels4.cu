#include "kernels4.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
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
__global__ void k_moe_route(const float * __restrict__ logits, int ls, int ne, int k, int * ids, float * wts, float * sg) {
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

constexpr int MOE_ROWS = 8;   // rows per block: one per warp

template <GType T>
__global__ void k_moe_gate_up(MoeDev m, const float * __restrict__ x, int xs, const int * __restrict__ ids, int k, float * __restrict__ h, int kdim,
                              const int * __restrict__ order, const int * __restrict__ order_n) {
    if (order && (int) blockIdx.x >= *order_n) return;
    const int p = order ? order[blockIdx.x] : blockIdx.x, t = p / k;
    const int slot = m.slot[ids[p]];
    if (slot < 0) return;
    extern __shared__ float xsm[];
    for (int i = threadIdx.x; i < kdim; i += blockDim.x) xsm[i] = x[(size_t) t * xs + i];
    __syncthreads();
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t rb = m.gate_bytes / m.ff;
    for (int rr = 0; rr < MOE_ROWS / 8; ++rr) {
        const int r = blockIdx.y * MOE_ROWS + w * (MOE_ROWS / 8) + rr;
        if (r >= m.ff) break;
        const float g = dot_row<T>(m.gate + slot * m.gate_bytes + r * rb, xsm, kdim, lane);
        const float u = dot_row<T>(m.up + slot * m.gate_bytes + r * rb, xsm, kdim, lane);
        if (lane == 0) h[(size_t) p * m.ff + r] = silu4(g) * u;
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
    for (int i = threadIdx.x; i < m.ff; i += blockDim.x) hsm[i] = h[(size_t) p * m.ff + i];
    __syncthreads();
    const size_t rb = m.down_bytes / m.n;
    const float wt = wts[p];
    for (int rr = 0; rr < MOE_ROWS / 8; ++rr) {
        const int r = blockIdx.y * MOE_ROWS + w * (MOE_ROWS / 8) + rr;
        if (r >= m.n) break;
        const float v = dot_row<T>(m.down + slot * m.down_bytes + r * rb, hsm, m.ff, lane);
        if (lane == 0) y[(size_t) p * m.n + r] = wt * v;
    }
}

__global__ void k_moe_reduce(const float * __restrict__ shexp, const float * __restrict__ sg, const float * __restrict__ y, int k,
                             float * __restrict__ out, int n, const int * __restrict__ ids, const int * __restrict__ owner, int g, int cpu_owner,
                             const volatile unsigned * cpu_flag, const float * cpu_y, const int * counter, unsigned seq_tag) {
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
    // stable placement: one thread per expert walks the pairs in index order
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        if (!cnt[e]) continue;
        int o = off[e];
        for (int p = 0; p < P; ++p) if (ids[p] == e) order[o++] = p;
    }
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
    __shared__ __align__(16) half As[GE_M * GE_LD];
    __shared__ __align__(16) half Bs[GE_N * GE_LD];
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
    for (int t0 = 0; t0 < cnt; t0 += GE_N) {
        const int nt = min(GE_N, cnt - t0);
        float acc[2][4] = {};
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
            // B: nt token rows x 128 cols
            for (int ci = threadIdx.x; ci < GE_N * (GE_K / 8); ci += blockDim.x) {
                const int tr = ci / (GE_K / 8), cc = ci % (GE_K / 8);
                uint4 v = make_uint4(0, 0, 0, 0);
                if (tr < nt) {
                    const int p = order[start + t0 + tr];
                    const int src = in_row ? in_row[p] : p / in_div;
                    v = *(const uint4 *) (in + (size_t) src * in_stride + k0 + cc * 8);
                }
                *(uint4 *) &Bs[tr * GE_LD + cc * 8] = v;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < GE_K / 16; ++kk) {
                unsigned a[4], b[4];
                ldsm_x4(a, &As[(wm * 16 + (lane & 15)) * GE_LD + kk * 16 + (lane >> 4) * 8]);
                ldsm_x4(b, &Bs[(wn * 16 + (lane & 7) + ((lane >> 4) << 3)) * GE_LD + kk * 16 + ((lane >> 3) & 1) * 8]);
                const unsigned b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
                mma16816_4(acc[0], a, b0);
                mma16816_4(acc[1], a, b1);
            }
        }
        // results to shared memory [row][token]
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int q = 0; q < 4; ++q) Cs[wm * 16 + gid + (q >> 1) * 8][wn * 16 + ni * 8 + 2 * tig + (q & 1)] = acc[ni][q];
        __syncthreads();
        if (GU) {
            for (int i = threadIdx.x; i < 32 * nt; i += blockDim.x) {
                const int rr = i % 32, tr = i / 32, row = r_base + rr;
                if (row >= mrows) continue;
                const int p = order[start + t0 + tr];
                h16[(size_t) p * rows_out + row] = __float2half(silu4(Cs[rr][tr]) * Cs[rr + 32][tr]);
            }
        } else {
            for (int i = threadIdx.x; i < GE_M * nt; i += blockDim.x) {
                const int rr = i % GE_M, tr = i / GE_M, row = r_base + rr;
                if (row >= mrows) continue;
                const int p = order[start + t0 + tr];
                y[(size_t) p * rows_out + row] = wts[p] * Cs[rr][tr];
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
    k_hc_norm<<<dim3(hc, nt), 256, 0, s>>>(res, w, xn, n, hc, eps, inj_w, injp);
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
    k_moe_route<<<nt, 256, 0, s>>>(logits, ls, n_expert, k, ids, wts, sg);
}

#define MOE_TYPE_SWITCH(T, CALL)                                                  \
    switch (T) {                                                                  \
        case GType::Q4_K: { constexpr GType TT = GType::Q4_K; CALL; } break;       \
        case GType::Q5_K: { constexpr GType TT = GType::Q5_K; CALL; } break;       \
        case GType::Q5_1: { constexpr GType TT = GType::Q5_1; CALL; } break;       \
        case GType::Q8_0: { constexpr GType TT = GType::Q8_0; CALL; } break;       \
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
                const int * counter, unsigned seq_tag, cudaStream_t s) {
    k_moe_reduce<<<dim3((n + 255) / 256, nt), 256, 0, s>>>(shexp, sg, y, k, out, n, ids, owner, g, cpu_owner, cpu_flag, cpu_y, counter, seq_tag);
}
void moe_publish(volatile unsigned * seq, int * ntp, int * ids_dst, float * wts_dst, float * x_dst, const float * x, int xs, int n,
                 const int * ids, const float * wts, int k, int nt, const int * counter, unsigned seq_tag, cudaStream_t s) {
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
