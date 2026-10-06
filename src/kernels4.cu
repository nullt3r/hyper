#include "kernels4.cuh"

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
__global__ void k_hc_norm(const float * __restrict__ res, const float * __restrict__ w, float * __restrict__ xn, int n, int hc, float eps) {
    const int s = blockIdx.x, t = blockIdx.y;
    const float * r = res + ((size_t) t * hc + s) * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += r[i] * r[i];
    const float inv = rsqrtf(block_sum4(ss) / n + eps);
    float * o = xn + ((size_t) t * hc + s) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = r[i] * inv * w[(size_t) s * n + i];
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
// block per token, one thread per expert (ne <= 1024): softmax, then each expert's rank among the probabilities
// (ties to the lower index) decides whether it is selected and in which slot
__global__ void k_moe_route(const float * __restrict__ logits, int ls, int ne, int k, int * ids, float * wts, float * sg) {
    const int t = blockIdx.x, e = threadIdx.x;
    const float * l = logits + (size_t) t * ls;
    __shared__ float p[1024];
    __shared__ float bv[32];
    __shared__ float sel_w[MOE_MAX_USED];
    const float v = e < ne ? l[e] : -FLT_MAX;
    float mx = v;
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if ((e & 31) == 0) bv[e >> 5] = mx;
    __syncthreads();
    mx = -FLT_MAX;
    for (int w = 0; w < (int) (blockDim.x >> 5); ++w) mx = fmaxf(mx, bv[w]);
    const float ex = e < ne ? expf(v - mx) : 0.0f;
    const float sum = block_sum4(ex);
    const float pe = ex / sum;
    if (e < ne) p[e] = pe;
    __syncthreads();
    if (e < ne) {
        int rank = 0;
        for (int j = 0; j < ne; ++j) { const float q = p[j]; rank += q > pe || (q == pe && j < e); }
        if (rank < k) { ids[t * k + rank] = e; sel_w[rank] = pe; }
    }
    __syncthreads();
    if (e == 0) {
        float s = 0.0f;
        for (int j = 0; j < k; ++j) s += sel_w[j];
        for (int j = 0; j < k; ++j) wts[t * k + j] = sel_w[j] / s;
        sg[t] = sigm(l[ne]);
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
__global__ void k_moe_gate_up(MoeDev m, const float * __restrict__ x, int xs, const int * __restrict__ ids, int k, float * __restrict__ h, int kdim) {
    const int p = blockIdx.x, t = p / k;
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
                           float * __restrict__ y) {
    const int p = blockIdx.x;
    const int slot = m.slot[ids[p]];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (slot < 0) {
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
                             float * __restrict__ out, int n, const int * __restrict__ ids, const int * __restrict__ owner, int cpu_owner,
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
    for (int j = 0; j < k; ++j) acc += y[(size_t) (t * k + j) * n + r];
    if (need) acc += __ldcv(cpu_y + (size_t) t * 4096 + r);
    out[(size_t) t * n + r] = acc;
}

__global__ void k_moe_publish(CpuMoeRec * rec, const float * __restrict__ x, int xs, int n, const int * __restrict__ ids,
                              const float * __restrict__ wts, int k, int nt, const int * counter, unsigned seq_tag) {
    for (int t = 0; t < nt; ++t)
        for (int i = threadIdx.x; i < n; i += blockDim.x) rec->x[t][i] = x[(size_t) t * xs + i];
    for (int i = threadIdx.x; i < nt * k; i += blockDim.x) { rec->ids[i / k][i % k] = ids[i]; rec->wts[i / k][i % k] = wts[i]; }
    if (threadIdx.x == 0) rec->nt = nt;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) rec->seq = (unsigned) (*counter) * 64u + seq_tag;
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

void hc_norm(const float * res, const float * w, float * xn, int n, int hc, float eps, int nt, cudaStream_t s) {
    k_hc_norm<<<dim3(hc, nt), 256, 0, s>>>(res, w, xn, n, hc, eps);
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
    k_moe_route<<<nt, (n_expert + 31) / 32 * 32, 0, s>>>(logits, ls, n_expert, k, ids, wts, sg);
}

#define MOE_TYPE_SWITCH(T, CALL)                                                  \
    switch (T) {                                                                  \
        case GType::Q4_K: { constexpr GType TT = GType::Q4_K; CALL; } break;       \
        case GType::Q5_K: { constexpr GType TT = GType::Q5_K; CALL; } break;       \
        case GType::Q5_1: { constexpr GType TT = GType::Q5_1; CALL; } break;       \
        case GType::Q8_0: { constexpr GType TT = GType::Q8_0; CALL; } break;       \
        default: throw std::runtime_error(std::string("moe: unsupported expert type ") + gtype_name(T)); \
    }

void moe_gate_up(const MoeDev & m, const float * x, int xs, const int * ids, int k, float * h, int nt, cudaStream_t s) {
    const int kdim = (int) (m.gate_bytes / m.ff / gtype_block_bytes(m.tg) * gtype_block_elems(m.tg));
    MOE_TYPE_SWITCH(m.tg, (k_moe_gate_up<TT><<<dim3(nt * k, (m.ff + MOE_ROWS - 1) / MOE_ROWS), 256, kdim * sizeof(float), s>>>(m, x, xs, ids, k, h, kdim)));
}
void moe_down(const MoeDev & m, const float * h, const int * ids, const float * wts, int k, float * y, int nt, cudaStream_t s) {
    MOE_TYPE_SWITCH(m.td, (k_moe_down<TT><<<dim3(nt * k, (m.n + MOE_ROWS - 1) / MOE_ROWS), 256, m.ff * sizeof(float), s>>>(m, h, ids, wts, k, y)));
}
void moe_reduce(const float * shexp, const float * sg, const float * y, int k, float * out, int n, int nt,
                const int * ids, const int * owner, int cpu_owner, const volatile unsigned * cpu_flag, const float * cpu_y,
                const int * counter, unsigned seq_tag, cudaStream_t s) {
    k_moe_reduce<<<dim3((n + 255) / 256, nt), 256, 0, s>>>(shexp, sg, y, k, out, n, ids, owner, cpu_owner, cpu_flag, cpu_y, counter, seq_tag);
}
void moe_publish(CpuMoeRec * rec, const float * x, int xs, int n, const int * ids, const float * wts, int k, int nt,
                 const int * counter, unsigned seq_tag, cudaStream_t s) {
    k_moe_publish<<<1, 512, 0, s>>>(rec, x, xs, n, ids, wts, k, nt, counter, seq_tag);
}
void ple_apply(float * res, const float * key, const float * value, const float * wk, const float * wq, const float * wconv_norm,
               const float * conv_w, float * conv_state, float * conv_snap, int n, int hc, int K, int dil, float eps, int nt,
               float * scratch, cudaStream_t s) {
    if ((K - 1) * dil > 16) throw std::runtime_error("ple: conv history too long");
    k_ple_stats<<<dim3(hc, nt), 256, 0, s>>>(res, key, value, wk, wq, n, hc, scratch);
    k_ple_apply<<<(hc * n + 255) / 256, 256, 0, s>>>(res, value, wconv_norm, conv_w, conv_state, conv_snap, scratch, n, hc, K, dil, eps, nt);
}

} // namespace hyper
