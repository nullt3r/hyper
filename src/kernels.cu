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
                         int nt, NormIn nin) {
    const int tile = blockIdx.x;
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
    for (int b = w; b < kb; b += nw) {
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
    if (w == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { float t = 0; for (int ww = 0; ww < nw; ++ww) t += red[ww][lane][i]; acc[i] = t; }
        const int r0 = tile * 16 + gid, t0 = 2 * tig;
        auto put = [&](int row, int t, float v) {
            if (row < n && t < nt) { const size_t o = (size_t) t * ys + row; y[o] = add ? add[o] + v : v; }
        };
        put(r0, t0, acc[0]); put(r0, t0 + 1, acc[1]);
        put(r0 + 8, t0, acc[2]); put(r0 + 8, t0 + 1, acc[3]);
    }
}

template <int NT>
__global__ void k_gemv_bf16(const __nv_bfloat16 * __restrict__ w, int n, int k,
                            const float * __restrict__ x, int xs, float * __restrict__ y, int ys, const float * __restrict__ add,
                            NormIn nin) {
    float inv[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) inv[t] = input_inv_rms(nin.ss + t * nin.nss, nin.nss, k, nin.w, nin.eps);
    const float4 * nw4 = (const float4 *) nin.w;
    const int row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n) return;
    const uint4 * w8 = (const uint4 *) (w + (size_t) row * k);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    const int nchunk = k / 8;
    for (int c = lane; c < nchunk; c += 32) {
        const uint4 q = __ldg(w8 + c);
        const __nv_bfloat162 * b = (const __nv_bfloat162 *) &q;
        const float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
        const float2 f2 = __bfloat1622float2(b[2]), f3 = __bfloat1622float2(b[3]);
        float4 wa, wb;
        if (nin.w) { wa = __ldg(nw4 + c * 2); wb = __ldg(nw4 + c * 2 + 1); }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 * x4 = (const float4 *) (x + (size_t) t * xs);
            float4 xa = __ldg(x4 + c * 2), xb = __ldg(x4 + c * 2 + 1);
            if (nin.w) {
                const float f = inv[t];
                xa.x *= f * wa.x; xa.y *= f * wa.y; xa.z *= f * wa.z; xa.w *= f * wa.w;
                xb.x *= f * wb.x; xb.y *= f * wb.y; xb.z *= f * wb.z; xb.w *= f * wb.w;
            }
            acc[t] += f0.x * xa.x + f0.y * xa.y + f1.x * xa.z + f1.y * xa.w + f2.x * xb.x + f2.y * xb.y + f3.x * xb.z + f3.y * xb.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) {
            const size_t o = (size_t) t * ys + row;
            y[o] = add ? add[o] + v : v;
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

void gemv_q8(const Q8W & W, const float * x, int xs, float * y, int ys, const float * add, int nt, cudaStream_t s,
             const NormIn & nin) {
    if (nt < 1 || nt > 8) throw std::runtime_error("gemv_q8: nt must be 1..8");
    k_mma_q8<<<(W.n + 15) / 16, 256, 0, s>>>(W.q, W.s, W.n, W.k, x, xs, y, ys, add, nt, nin);
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
    const int grid = (W.n + GEMV_WARPS - 1) / GEMV_WARPS;
    NT_SWITCH(nt, (k_gemv_bf16<NT><<<grid, GEMV_WARPS * 32, 0, s>>>(W.w, W.n, W.k, x, xs, y, ys, add, nin)));
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
void gdn_conv(float * in, int stride, float * conv_state, float * conv_snap, const float * conv_w, int channels, int K,
              int nt, cudaStream_t s) {
    if (K > 9) throw std::runtime_error("gdn_conv: kernel too large");
    k_gdn_conv<<<(channels + 255) / 256, 256, 0, s>>>(in, stride, conv_state, conv_snap, conv_w, channels, K, nt);
}
void gdn_step(const float * in, int stride, int ab_off, float * state, float * state_snap, float * o, int o_stride,
              const float * dt_bias, const float * ssm_a, int n_k, int n_v, int dk, int dv, float eps, int nt,
              cudaStream_t s) {
    if (dk != 128 || dv % 32) throw std::runtime_error("gdn_step: expects dk = 128");
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
void incr_counter(int * c, cudaStream_t s) { k_incr<<<1, 1, 0, s>>>(c); }

} // namespace hyper
