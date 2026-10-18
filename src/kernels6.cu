#include "kernels6.cuh"

#include <cfloat>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hyper {

namespace {

__global__ void k_mimo_qkv(float * __restrict__ big, int bs, int nq, int nkv, int dk, int dv, int n_rot, double base, const int * __restrict__ pos,
                           half * __restrict__ kc, half * __restrict__ vc, size_t kpitch, size_t vpitch, int ring) {
    const int hb = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float * row = big + (size_t) t * bs;
    const int p = pos[0] + t;
    const size_t r = ring ? (size_t) (p % ring) : (size_t) p;
    if (hb < nq + nkv) {   // q or k head: rope the first n_rot dims (all reads before the in-place writes)
        float * x = hb < nq ? row + (size_t) hb * dk : row + (size_t) nq * dk + (size_t) (hb - nq) * dk;
        float v = d < dk ? x[d] : 0.0f;
        if (d < n_rot) {
            const int hr = n_rot / 2, i = d < hr ? d : d - hr;
            const double theta = (double) p * pow(base, -2.0 * i / n_rot);
            double sn, cs;
            sincos(theta, &sn, &cs);
            const double x0 = x[i], x1 = x[i + hr];
            v = (float) (d < hr ? x0 * cs - x1 * sn : x0 * sn + x1 * cs);
        }
        __syncthreads();
        if (d < dk) {
            if (hb < nq) x[d] = v;
            else kc[(size_t) (hb - nq) * kpitch + r * dk + d] = __float2half(v);
        }
    } else {
        const int h = hb - nq - nkv;
        if (d < dv) vc[(size_t) h * vpitch + r * dv + d] = __float2half(row[(size_t) nq * dk + (size_t) nkv * dk + (size_t) h * dv + d]);
    }
}

__device__ __forceinline__ float wmax(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

constexpr int AT_KT = 32;            // keys per tile (one per lane)
constexpr int AT_QW = MIMO_QB / 8;   // queries per warp
constexpr int AT_PG = 256;           // keys per page (the unit dealt to the key splits)

template <int DK> constexpr size_t at_smem() { return (size_t) MIMO_QB * DK * 4 + (size_t) AT_KT * (DK + 2) * 2 + (size_t) AT_KT * 128 * 2; }

// block (kv head, key split, row block): the row block's queries (rows x the kv head's q heads, query qi = row * group + head)
// against the split's keys, online softmax in fp32; warp w holds queries w, w + 8, ...; lane = key in a tile for the scores,
// lane = 4 value dims for the output. ns == 1: finished here (sink, value scale); else partials (m, l, o) for the combine.
template <int DK>
__global__ void __launch_bounds__(256) k_mimo_attn(MimoAttn a, int ns, int RB) {
    constexpr int DV = 128;
    const int kvh = blockIdx.x, sp = blockIdx.y, rb = blockIdx.z;
    const int group = a.nq / a.nkv;
    const int r0 = rb * RB, r1 = min(a.nt, r0 + RB), Qn = (r1 - r0) * group;
    const int p0 = a.pos[0];
    const int pf = p0 + r0, pl = p0 + r1 - 1;
    // keys in pages of AT_PG dealt to the splits round robin, tiles at absolute multiples of AT_KT: a query's partials (and so its
    // result) do not depend on the other rows of the call -- a speculative verification row computes what a single decoded row does
    const int kmin = a.window ? max(0, pf - a.window + 1) : 0, kbeg = kmin & ~(AT_KT - 1), kend = pl + 1;
    extern __shared__ __align__(16) float sm[];
    float * qsm = sm;
    half2 * Ks = (half2 *) (qsm + MIMO_QB * DK);       // [AT_KT][DK / 2 + 1]
    half2 * Vs = Ks + AT_KT * (DK / 2 + 1);             // [AT_KT][DV / 2]
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    for (int i = tid; i < Qn * DK; i += 256) {
        const int qi = i / DK, dd = i - qi * DK;
        const int row = r0 + qi / group, h = kvh * group + qi % group;
        qsm[i] = a.q[(size_t) row * a.qs + (size_t) h * DK + dd];
    }
    const int nj = Qn > w ? (Qn - w + 7) / 8 : 0;   // this warp's queries
    int lo[AT_QW], hi[AT_QW];
    float m[AT_QW], l[AT_QW], o[AT_QW][4];
#pragma unroll
    for (int j = 0; j < AT_QW; ++j) {
        const int qi = w + 8 * j, pq = p0 + r0 + (qi < Qn ? qi / group : 0);
        hi[j] = pq; lo[j] = a.window ? pq - a.window + 1 : 0;
        m[j] = -INFINITY; l[j] = 0.0f; o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    }
    const half * kb = a.kc + (size_t) kvh * a.kpitch;
    const half * vb = a.vc + (size_t) kvh * a.vpitch;
    for (int pg = kbeg / AT_PG; pg * AT_PG < kend; ++pg) {
    if (pg % ns != sp) continue;
    const int k1 = min((pg + 1) * AT_PG, kend);
    for (int kt = max(pg * AT_PG, kbeg); kt < k1; kt += AT_KT) {
        __syncthreads();
        for (int i = tid; i < AT_KT * (DK / 2); i += 256) {
            const int j = i / (DK / 2), c = i - j * (DK / 2), key = kt + j;
            half2 v = __floats2half2_rn(0.0f, 0.0f);
            if (key < k1) { const size_t r = a.ring ? (size_t) (key % a.ring) : (size_t) key; v = *(const half2 *) (kb + r * DK + 2 * c); }
            Ks[j * (DK / 2 + 1) + c] = v;
        }
        for (int i = tid; i < AT_KT * (DV / 2); i += 256) {
            const int j = i / (DV / 2), c = i - j * (DV / 2), key = kt + j;
            half2 v = __floats2half2_rn(0.0f, 0.0f);
            if (key < k1) { const size_t r = a.ring ? (size_t) (key % a.ring) : (size_t) key; v = *(const half2 *) (vb + r * DV + 2 * c); }
            Vs[j * (DV / 2) + c] = v;
        }
        __syncthreads();
        if (nj == 0) continue;
        const int key = kt + lane;
        float sc[AT_QW];
#pragma unroll
        for (int j = 0; j < AT_QW; ++j) sc[j] = 0.0f;
        const half2 * kr = Ks + lane * (DK / 2 + 1);
#pragma unroll 4
        for (int c = 0; c < DK / 2; ++c) {
            const float2 kv = __half22float2(kr[c]);
#pragma unroll
            for (int j = 0; j < AT_QW; ++j) {
                if (j < nj) {
                    const float2 qv = *(const float2 *) (qsm + (w + 8 * j) * DK + 2 * c);
                    sc[j] += kv.x * qv.x + kv.y * qv.y;
                }
            }
        }
        float pj[AT_QW];
#pragma unroll
        for (int j = 0; j < AT_QW; ++j) {
            pj[j] = 0.0f;
            if (j >= nj) continue;
            const bool ok = key < k1 && key <= hi[j] && key >= lo[j];
            const float sv = ok ? sc[j] * a.scale : -INFINITY;
            const float mn = fmaxf(m[j], wmax(sv));
            if (mn == -INFINITY) continue;
            const float corr = __expf(m[j] - mn);
            const float p = ok ? __expf(sv - mn) : 0.0f;
            l[j] = l[j] * corr + wsum(p);
            m[j] = mn;
            o[j][0] *= corr; o[j][1] *= corr; o[j][2] *= corr; o[j][3] *= corr;
            pj[j] = p;
        }
        const half2 * vr = Vs + lane * 2;
#pragma unroll 4
        for (int kk = 0; kk < AT_KT; ++kk) {
            const float2 v0 = __half22float2(vr[kk * (DV / 2)]), v1 = __half22float2(vr[kk * (DV / 2) + 1]);
#pragma unroll
            for (int j = 0; j < AT_QW; ++j) {
                if (j < nj) {
                    const float p = __shfl_sync(0xffffffff, pj[j], kk);
                    o[j][0] += p * v0.x; o[j][1] += p * v0.y; o[j][2] += p * v1.x; o[j][3] += p * v1.y;
                }
            }
        }
    }
    }
#pragma unroll
    for (int j = 0; j < AT_QW; ++j) {
        if (j >= nj) continue;
        const int qi = w + 8 * j;
        if (ns == 1) {
            const int row = r0 + qi / group, h = kvh * group + qi % group;
            const float sk = a.sinks ? a.sinks[h] : -INFINITY;
            const float M = fmaxf(m[j], sk);
            const float em = __expf(m[j] - M);
            const float den = l[j] * em + (a.sinks ? __expf(sk - M) : 0.0f);
            const float f = den > 0.0f ? em / den * a.vscale : 0.0f;
            float4 r = make_float4(o[j][0] * f, o[j][1] * f, o[j][2] * f, o[j][3] * f);
            *(float4 *) (a.out + (size_t) row * a.os + (size_t) h * DV + lane * 4) = r;
        } else {
            float * dst = a.part + ((((size_t) kvh * gridDim.z + rb) * ns + sp) * MIMO_QB + qi) * (DV + 4);
            *(float4 *) (dst + 4 + lane * 4) = make_float4(o[j][0], o[j][1], o[j][2], o[j][3]);
            if (lane == 0) { dst[0] = m[j]; dst[1] = l[j]; }
        }
    }
    // (queries of this block that no warp holds do not exist: Qn <= MIMO_QB)
}

// block (local q head, row): merge the key splits' partials with the sink, value scale
__global__ void k_mimo_comb(MimoAttn a, int ns, int RB, int nrb) {
    constexpr int DV = 128;
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int group = a.nq / a.nkv, kvh = h / group, rb = t / RB, qi = (t - rb * RB) * group + h % group;
    const float * base = a.part + (((size_t) kvh * nrb + rb) * ns) * MIMO_QB * (DV + 4);
    float M = a.sinks ? a.sinks[h] : -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, base[((size_t) s * MIMO_QB + qi) * (DV + 4)]);
    float L = a.sinks ? __expf(a.sinks[h] - M) : 0.0f, acc = 0.0f;
    if (M != -INFINITY)
        for (int s = 0; s < ns; ++s) {
            const float * p = base + ((size_t) s * MIMO_QB + qi) * (DV + 4);
            if (p[0] == -INFINITY) continue;
            const float e = __expf(p[0] - M);
            L += p[1] * e;
            acc += p[4 + d] * e;
        }
    a.out[(size_t) t * a.os + (size_t) h * DV + d] = L > 0.0f ? acc / L * a.vscale : 0.0f;
}

__global__ void k_mimo_hstore(const float * __restrict__ R, int nt, const int * __restrict__ pos, float * __restrict__ hring, int HR, int n) {
    const int r = max(0, nt - HR) + blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nt || i >= n) return;
    hring[(size_t) ((pos[0] + r) % HR) * n + i] = R[(size_t) r * n + i];
}
__global__ void k_mimo_hgather(float * __restrict__ dst, const float * __restrict__ hring, int HR, int n, int q0, int nmain,
                               const float * __restrict__ chain) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v;
    if (t < nmain) { const int q = q0 + t - 1; v = q < 0 ? 0.0f : hring[(size_t) (q % HR) * n + i]; }
    else v = chain[(size_t) (t - nmain) * n + i];
    dst[(size_t) t * n + i] = v;
}

} // namespace

void mimo_hstore(const float * R, int nt, const int * pos, float * hring, int HR, int n, cudaStream_t s) {
    k_mimo_hstore<<<dim3((n + 255) / 256, std::min(nt, HR)), 256, 0, s>>>(R, nt, pos, hring, HR, n);
}
void mimo_hgather(float * dst, const float * hring, int HR, int n, int q0, int nrows, int nmain, const float * chain, cudaStream_t s) {
    k_mimo_hgather<<<dim3((n + 255) / 256, nrows), 256, 0, s>>>(dst, hring, HR, n, q0, nmain, chain);
}

void mimo_qkv(float * big, int bs, int nq, int nkv, int dk, int dv, int n_rot, float base, const int * pos, int nt, half * kc,
              half * vc, size_t kpitch, size_t vpitch, int ring, cudaStream_t s) {
    if (dk > 256 || dv > 256) throw std::runtime_error("mimo_qkv: head too large");
    k_mimo_qkv<<<dim3(nq + 2 * nkv, nt), 256, 0, s>>>(big, bs, nq, nkv, dk, dv, n_rot, (double) base, pos, kc, vc, kpitch, vpitch, ring);
}

size_t mimo_attn_part_floats(int nkv, int nt, int group, int ns) {
    const int RB = MIMO_QB / group, nrb = (nt + RB - 1) / RB;
    return (size_t) nkv * nrb * ns * MIMO_QB * 132;
}

void mimo_attn(const MimoAttn & a, int ns, cudaStream_t s) {
    if (a.dk != 192 || a.dv != 128) throw std::runtime_error("mimo_attn: only k 192 / v 128 instantiated");
    const int group = a.nq / a.nkv;
    if (group < 1 || group > MIMO_QB || a.nq % a.nkv) throw std::runtime_error("mimo_attn: bad grouping");
    const int RB = MIMO_QB / group, nrb = (a.nt + RB - 1) / RB;
    if (ns < 1) ns = 1;
    if (ns > 1 && !a.part) throw std::runtime_error("mimo_attn: split without a partial buffer");
    static bool attr[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 16 && !attr[dev]) {
        cudaFuncSetAttribute(k_mimo_attn<192>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) at_smem<192>());
        attr[dev] = true;
    }
    k_mimo_attn<192><<<dim3(a.nkv, ns, nrb), 256, at_smem<192>(), s>>>(a, ns, RB);
    if (ns > 1) k_mimo_comb<<<dim3(a.nq, a.nt), 128, 0, s>>>(a, ns, RB, nrb);
}

} // namespace hyper
