#include "kernels.cuh"

#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace hyper {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
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

// ---------------- GEMV ----------------
constexpr int GEMV_ROWS = 8;   // warps per block, one row per warp

// each warp computes R consecutive rows; one x chunk (16 floats) serves all R rows
// optional fused RMSNorm of the input: 1/rms from nss partial sums of squares produced upstream
__device__ __forceinline__ float input_inv_rms(const float * __restrict__ ss, int nss, int k, const float * nw, float eps) {
    if (!nw) return 1.0f;
    float t = 0.0f;
    for (int i = 0; i < nss; ++i) t += ss[i];
    return rsqrtf(t / k + eps);
}

template <int R>
__global__ void k_gemv_q8(const int8_t * __restrict__ qs, const half * __restrict__ d, int n, int k,
                          const float * __restrict__ x, float * __restrict__ y, const float * __restrict__ add,
                          const float * __restrict__ nw, const float * __restrict__ ss, int nss, float eps) {
    const float inv = input_inv_rms(ss, nss, k, nw, eps);
    const int row0 = (blockIdx.x * GEMV_ROWS + (threadIdx.x >> 5)) * R;
    const int lane = threadIdx.x & 31;
    if (row0 >= n) return;
    const float4 * nw4 = (const float4 *) nw;
    const float4 * x4 = (const float4 *) x;
    const int nchunk = k / 16, kb = k / 32;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int c = lane; c < nchunk; c += 32) {
        uint4 q[R];
        float sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = min(row0 + r, n - 1);
            q[r] = __ldg((const uint4 *) (qs + (size_t) row * k) + c);
            sc[r] = __half2float(d[(size_t) row * kb + (c >> 1)]);
        }
        float4 xv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xv[j] = __ldg(x4 + c * 4 + j);
            if (nw) {
                const float4 wv = __ldg(nw4 + c * 4 + j);
                xv[j].x *= inv * wv.x; xv[j].y *= inv * wv.y; xv[j].z *= inv * wv.z; xv[j].w *= inv * wv.w;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int8_t * b = (const int8_t *) &q[r];
            float part = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                part += xv[j].x * b[4 * j] + xv[j].y * b[4 * j + 1] + xv[j].z * b[4 * j + 2] + xv[j].w * b[4 * j + 3];
            acc[r] += part * sc[r];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n) y[row] = add ? add[row] + v : v;
    }
}

__global__ void k_gemv_bf16(const __nv_bfloat16 * __restrict__ w, int n, int k,
                            const float * __restrict__ x, float * __restrict__ y, const float * __restrict__ add,
                            const float * __restrict__ nw, const float * __restrict__ ss, int nss, float eps) {
    const float inv = input_inv_rms(ss, nss, k, nw, eps);
    const float4 * nw4 = (const float4 *) nw;
    const int row = blockIdx.x * GEMV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n) return;
    const uint4 * w8 = (const uint4 *) (w + (size_t) row * k);
    const float4 * x4 = (const float4 *) x;
    float acc = 0.0f;
    const int nchunk = k / 8;
    for (int c = lane; c < nchunk; c += 32) {
        const uint4 q = __ldg(w8 + c);
        const __nv_bfloat162 * b = (const __nv_bfloat162 *) &q;
        float4 xa = __ldg(x4 + c * 2), xb = __ldg(x4 + c * 2 + 1);
        if (nw) {
            const float4 wa = __ldg(nw4 + c * 2), wb = __ldg(nw4 + c * 2 + 1);
            xa.x *= inv * wa.x; xa.y *= inv * wa.y; xa.z *= inv * wa.z; xa.w *= inv * wa.w;
            xb.x *= inv * wb.x; xb.y *= inv * wb.y; xb.z *= inv * wb.z; xb.w *= inv * wb.w;
        }
        float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
        float2 f2 = __bfloat1622float2(b[2]), f3 = __bfloat1622float2(b[3]);
        acc += f0.x * xa.x + f0.y * xa.y + f1.x * xa.z + f1.y * xa.w + f2.x * xb.x + f2.y * xb.y + f3.x * xb.z + f3.y * xb.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = add ? add[row] + acc : acc;
}

// ---------------- norms / embedding ----------------
__global__ void k_rmsnorm(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y, int n, float eps) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

__global__ void k_embed_q8_0(const uint8_t * __restrict__ table, int64_t row_bytes, int token, float * __restrict__ y, int n) {
    const uint8_t * row = table + (int64_t) token * row_bytes;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint8_t * blk = row + (i / 32) * 34;
        const float d = __half2float(*(const half *) blk);
        y[i] = d * (float) ((const int8_t *) (blk + 2))[i % 32];
    }
}

// ---------------- gated attention ----------------
// grid: n_head + n_head_kv blocks, blockDim = hd
__global__ void k_attn_prep(float * qg, float * k, const float * v, const float * qnorm, const float * knorm,
                            half * kcache, half * vcache, const int * pos_p, int n_head, int n_head_kv, int hd, int n_rot,
                            float rope_base, float eps, int max_pos) {
    const int b = blockIdx.x, i = threadIdx.x;
    const int pos = *pos_p;
    const bool is_q = b < n_head;
    float * vec = is_q ? qg + (size_t) b * 2 * hd : k + (size_t) (b - n_head) * hd;
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
        kcache[((size_t) hk * max_pos + pos) * hd + i] = __float2half(xi);
        vcache[((size_t) hk * max_pos + pos) * hd + i] = __float2half(v[(size_t) hk * hd + i]);
    }
}

// one block per q head, 8 warps; each warp walks positions w, w+8, ...; lane owns hd/32 dims
template <int HD>
__global__ void k_attn_decode(const float * __restrict__ qg, const half * __restrict__ kcache, const half * __restrict__ vcache,
                              float * __restrict__ out, const int * pos_p, int head_off, int group, int kv_off, float scale, int max_pos) {
    constexpr int PER = HD / 32;
    const int h = blockIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int hk = (head_off + h) / group - kv_off;
    const int n_pos = *pos_p + 1;
    const float * q = qg + (size_t) h * 2 * HD;
    float qr[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) qr[j] = q[lane * PER + j] * scale;
    float m = -FLT_MAX, l = 0.0f, acc[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) acc[j] = 0.0f;
    const half * kb = kcache + (size_t) hk * max_pos * HD;
    const half * vb = vcache + (size_t) hk * max_pos * HD;
    for (int t = wid; t < n_pos; t += nw) {
        const half * kt = kb + (size_t) t * HD + lane * PER;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) s += qr[j] * __half2float(kt[j]);
        s = warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new), p = expf(s - m_new);
        l = l * corr + p;
        const half * vt = vb + (size_t) t * HD + lane * PER;
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[j] = acc[j] * corr + p * __half2float(vt[j]);
        m = m_new;
    }
    // combine warps
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
        const float gate = q[HD + i];
        out[(size_t) h * HD + i] = (num / den) * sigmoidf(gate);
    }
}

// ---------------- gated delta net ----------------
__global__ void k_gdn_conv(float * qkv, float * st, const float * __restrict__ w, int channels, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float x = qkv[c];
    float acc = w[(size_t) c * K + (K - 1)] * x;
    for (int j = 0; j < K - 1; ++j) acc += w[(size_t) c * K + j] * st[(size_t) j * channels + c];
    for (int j = 0; j < K - 2; ++j) st[(size_t) j * channels + c] = st[(size_t) (j + 1) * channels + c];
    st[(size_t) (K - 2) * channels + c] = x;
    qkv[c] = silu(acc);
}

// grid (n_v, dv/32), 8 warps: block owns head h and 32 state columns; warp w sums over k-rows [w*dk/8, (w+1)*dk/8)
// state layout S[i][j] (i: k index, j: v index) so a warp reads 128 contiguous bytes per row
__global__ void k_gdn_step(const float * __restrict__ qkv, const float * __restrict__ ab, const float * __restrict__ dt_bias,
                           const float * __restrict__ ssm_a, float * __restrict__ state, float * __restrict__ o,
                           int n_k, int n_v, int dk, int dv, float eps) {
    const int h = blockIdx.x, j = blockIdx.y * 32 + (threadIdx.x & 31), w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int hk = h % n_k;
    __shared__ float sq[256], sk[256], red[8][32];
    const float * q = qkv + (size_t) hk * dk;
    const float * k = qkv + (size_t) n_k * dk + (size_t) hk * dk;
    const float * v = qkv + (size_t) 2 * n_k * dk + (size_t) h * dv;
    float qq = 0.0f, kk = 0.0f;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) { const float a = q[i], b = k[i]; sq[i] = a; sk[i] = b; qq += a * a; kk += b * b; }
    qq = block_sum(qq);
    kk = block_sum(kk);
    const float qs = rsqrtf(qq + eps), ks = rsqrtf(kk + eps);
    const float g = softplusf(ab[h] + dt_bias[h]) * ssm_a[h];
    const float decay = expf(g);
    const float beta = sigmoidf(ab[n_v + h]);
    float * S = state + (size_t) h * dk * dv;
    const int i0 = w * (dk / nw), i1 = i0 + dk / nw;
    float part = 0.0f;
    for (int i = i0; i < i1; ++i) part += S[(size_t) i * dv + j] * sk[i];
    red[w][threadIdx.x & 31] = part * ks;
    __syncthreads();
    float kv = 0.0f;
    for (int ww = 0; ww < nw; ++ww) kv += red[ww][threadIdx.x & 31];
    kv *= decay;
    const float delta = (v[j] - kv) * beta;
    __syncthreads();
    float out = 0.0f;
    for (int i = i0; i < i1; ++i) {
        const float s = S[(size_t) i * dv + j] * decay + sk[i] * ks * delta;
        S[(size_t) i * dv + j] = s;
        out += s * sq[i];
    }
    red[w][threadIdx.x & 31] = out;
    __syncthreads();
    if (w == 0) {
        float acc = 0.0f;
        for (int ww = 0; ww < nw; ++ww) acc += red[ww][threadIdx.x & 31];
        o[(size_t) h * dv + j] = acc * qs * rsqrtf((float) dv);
    }
}

__global__ void k_gated_norm(float * o, const float * __restrict__ z, const float * __restrict__ w, int dh, float eps) {
    const int h = blockIdx.x, i = threadIdx.x;
    float x = o[(size_t) h * dh + i];
    const float ss = block_sum(x * x);
    x = x * rsqrtf(ss / dh + eps) * w[i];
    o[(size_t) h * dh + i] = x * silu(z[(size_t) h * dh + i]);
}

__global__ void k_silu_mul(const float * __restrict__ gu, float * __restrict__ h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(gu[i]) * gu[n + i];
}

__global__ void k_argmax(const float * __restrict__ x, int n, int * out) {
    float best = -FLT_MAX; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (x[i] > best) { best = x[i]; bi = i; }
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
    if (threadIdx.x == 0) *out = si[0];
}


__global__ void k_argmax_pair(const float * __restrict__ x, int n, int offset, float * out2) {
    float best = -FLT_MAX; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (x[i] > best) { best = x[i]; bi = i; }
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
    if (threadIdx.x == 0) { out2[0] = sv[0]; ((int *) out2)[1] = si[0] + offset; }
}

// grid: nchunk blocks of AR_CHUNK/4 threads; block b owns elements [b*AR_CHUNK, (b+1)*AR_CHUNK)
__global__ void k_allreduce_add(float * x, const float * __restrict__ part, float * slots, unsigned long long * flags,
                                int g, int ndev, int n, const int * counter, int call) {
    const int b = blockIdx.x, nchunk = gridDim.x;
    const int i0 = b * AR_CHUNK;
    const unsigned long long seq = (unsigned long long) (*counter) * 1024ull + (unsigned long long) call + 1ull;
    float * buf = slots + (size_t) (call & 1) * ndev * n;
    for (int i = i0 + threadIdx.x; i < min(i0 + AR_CHUNK, n); i += blockDim.x) buf[(size_t) g * n + i] = part[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *((volatile unsigned long long *) &flags[(size_t) g * nchunk + b]) = seq;
        for (int d = 0; d < ndev; ++d) {
            if (d == g) continue;
            while (*((volatile unsigned long long *) &flags[(size_t) d * nchunk + b]) < seq) { }
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = i0 + threadIdx.x; i < min(i0 + AR_CHUNK, n); i += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < ndev; ++d) acc += d == g ? part[i] : __ldcv(&buf[(size_t) d * n + i]);
        x[i] += acc;
    }
}

// one thread per element; writer publishes {value, seq} with a single 8-byte store, readers spin on the packets
__global__ void k_allreduce_add_ll(float * x, const float * __restrict__ part, uint2 * slots, int g, int ndev, int n,
                                   const int * counter, int call) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned seq = (unsigned) (*counter) * 1024u + (unsigned) call + 1u;
    uint2 * buf = slots + (size_t) (call & 1) * ndev * n;
    const float mine = part[i];
    volatile unsigned long long * dst = (volatile unsigned long long *) &buf[(size_t) g * n + i];
    *dst = ((unsigned long long) seq << 32) | (unsigned long long) __float_as_uint(mine);
    float acc = 0.0f;
    for (int d = 0; d < ndev; ++d) {
        if (d == g) { acc += mine; continue; }
        volatile unsigned long long * src = (volatile unsigned long long *) &buf[(size_t) d * n + i];
        unsigned long long v;
        do { v = *src; } while ((unsigned) (v >> 32) != seq);
        acc += __uint_as_float((unsigned) (v & 0xffffffffu));
    }
    x[i] += acc;
}

// thread per element pair: {half2(part[2i], part[2i+1]), seq} in one 8-byte packet
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
    for (int d = 0; d < ndev; ++d) {
        float2 f;
        if (d == g) {
            f = __half22float2(mh);   // use the rounded value so every GPU sums identical numbers
        } else {
            volatile unsigned long long * src = (volatile unsigned long long *) &buf[(size_t) d * n2 + i];
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

// sum of squares of x into ss[0] (single block)
__global__ void k_sumsq(const float * __restrict__ x, int n, float * ss) {
    float t = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) t += x[i] * x[i];
    t = block_sum(t);
    if (threadIdx.x == 0) ss[0] = t;
}

__global__ void k_incr(int * c) { *c += 1; }

} // namespace

int g_gemv_rows_per_warp = 4;
void gemv_q8(const Q8W & W, const float * x, float * y, const float * add, cudaStream_t s, const NormIn & nin) {
    const int R = g_gemv_rows_per_warp;
    const int per_block = GEMV_ROWS * R;
    const int grid = (W.n + per_block - 1) / per_block;
    switch (R) {
        case 1: k_gemv_q8<1><<<grid, GEMV_ROWS * 32, 0, s>>>(W.qs, W.d, W.n, W.k, x, y, add, nin.w, nin.ss, nin.nss, nin.eps); break;
        case 2: k_gemv_q8<2><<<grid, GEMV_ROWS * 32, 0, s>>>(W.qs, W.d, W.n, W.k, x, y, add, nin.w, nin.ss, nin.nss, nin.eps); break;
        default: k_gemv_q8<4><<<grid, GEMV_ROWS * 32, 0, s>>>(W.qs, W.d, W.n, W.k, x, y, add, nin.w, nin.ss, nin.nss, nin.eps); break;
    }
}
void gemv_bf16(const BF16W & W, const float * x, float * y, const float * add, cudaStream_t s, const NormIn & nin) {
    k_gemv_bf16<<<(W.n + GEMV_ROWS - 1) / GEMV_ROWS, GEMV_ROWS * 32, 0, s>>>(W.w, W.n, W.k, x, y, add, nin.w, nin.ss, nin.nss, nin.eps);
}
void rmsnorm(const float * x, const float * w, float * y, int n, float eps, cudaStream_t s) {
    k_rmsnorm<<<1, 1024, 0, s>>>(x, w, y, n, eps);
}
void embed_q8_0(const uint8_t * table, int64_t row_bytes, int token, float * y, int n, cudaStream_t s) {
    k_embed_q8_0<<<1, 1024, 0, s>>>(table, row_bytes, token, y, n);
}
void attn_prep(float * qg, float * k, const float * v, const float * qnorm, const float * knorm,
               half * kcache, half * vcache, const int * pos, int max_pos, int n_head, int n_kv, int hd, int n_rot,
               float rope_base, float eps, cudaStream_t s) {
    k_attn_prep<<<n_head + n_kv, hd, 0, s>>>(qg, k, v, qnorm, knorm, kcache, vcache, pos, n_head, n_kv,
                                             hd, n_rot, rope_base, eps, max_pos);
}
void attn_decode(const float * qg, const half * kcache, const half * vcache, float * out, const int * pos, int max_pos,
                 int n_head, int head_off, int group, int kv_off, int hd, float scale, cudaStream_t s) {
    if (hd != 256) throw std::runtime_error("attn_decode: only head_dim 256 is instantiated");
    k_attn_decode<256><<<n_head, 256, 0, s>>>(qg, kcache, vcache, out, pos, head_off, group, kv_off, scale, max_pos);
}
void gdn_conv(float * qkv, float * conv_state, const float * conv_w, int channels, int K, cudaStream_t s) {
    k_gdn_conv<<<(channels + 255) / 256, 256, 0, s>>>(qkv, conv_state, conv_w, channels, K);
}
void gdn_step(const float * qkv, const float * ab, const float * dt_bias, const float * ssm_a, float * state, float * o,
              int n_k, int n_v, int dk, int dv, float eps, cudaStream_t s) {
    k_gdn_step<<<dim3(n_v, dv / 32), 256, 0, s>>>(qkv, ab, dt_bias, ssm_a, state, o, n_k, n_v, dk, dv, eps);
}
void gated_norm(float * o, const float * z, const float * w, int n_heads, int dh, float eps, cudaStream_t s) {
    k_gated_norm<<<n_heads, dh, 0, s>>>(o, z, w, dh, eps);
}
void silu_mul(const float * gu, float * h, int n, cudaStream_t s) {
    k_silu_mul<<<(n + 255) / 256, 256, 0, s>>>(gu, h, n);
}
void argmax(const float * x, int n, int * out, cudaStream_t s) {
    k_argmax<<<1, 1024, 0, s>>>(x, n, out);
}

} // namespace hyper

namespace hyper {
void argmax_pair(const float * x, int n, int offset, float * out2, cudaStream_t s) {
    k_argmax_pair<<<1, 1024, 0, s>>>(x, n, offset, out2);
}
void allreduce_add(float * x, const float * part, float * slots, unsigned long long * flags, int g, int ndev, int n,
                   const int * counter, int call, cudaStream_t s) {
    const int nchunk = (n + AR_CHUNK - 1) / AR_CHUNK;
    k_allreduce_add<<<nchunk, AR_CHUNK / 4, 0, s>>>(x, part, slots, flags, g, ndev, n, counter, call);
}
void allreduce_add_ll(float * x, const float * part, uint2 * slots, int g, int ndev, int n,
                      const int * counter, int call, cudaStream_t s) {
    k_allreduce_add_ll<<<(n + 255) / 256, 256, 0, s>>>(x, part, slots, g, ndev, n, counter, call);
}
void allreduce_add_ll16(float * x, const float * part, uint2 * slots, int g, int ndev, int n,
                        const int * counter, int call, cudaStream_t s, float * ss_out) {
    const int n2 = n / 2;
    k_allreduce_add_ll16<<<(n2 + 255) / 256, 256, 0, s>>>(x, part, slots, g, ndev, n2, counter, call, ss_out);
}
int allreduce_ll16_nss(int n) { return (n / 2 + 255) / 256; }
void sumsq(const float * x, int n, float * ss, cudaStream_t s) { k_sumsq<<<1, 1024, 0, s>>>(x, n, ss); }
void incr_counter(int * c, cudaStream_t s) { k_incr<<<1, 1, 0, s>>>(c); }
} // namespace hyper
