#include "kernels6.cuh"

#include <cfloat>
#include <climits>
#include <cstdlib>
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

// ---- tensor-core prefill attention (global layers) ----
__device__ __forceinline__ void fa_cp16(void * smem, const void * gmem, bool valid) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void fa_ldm4(unsigned * r, const void * smem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
__device__ __forceinline__ void fa_ldm4t(unsigned * r, const void * smem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
__device__ __forceinline__ void fa_mma(float * c, const unsigned * a, const unsigned * b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned fa_h2(float lo, float hi) { const __half2 h = __floats2half2_rn(lo, hi); return *(const unsigned *) &h; }

constexpr int FA6_BQ = 64, FA6_BK = 32;
template <int DK, int DV> __host__ __device__ constexpr int fa6_qld() { return DK + 8; }
template <int DK, int DV> __host__ __device__ constexpr int fa6_vld() { return DV + 8; }
template <int DK, int DV> constexpr size_t fa6_smem() {
    return ((size_t) FA6_BQ * fa6_qld<DK, DV>() + 2 * ((size_t) FA6_BK * fa6_qld<DK, DV>() + (size_t) FA6_BK * fa6_vld<DK, DV>())) * sizeof(half);
}

// as kernels.cu's k_attn_fa (Q pre-scaled in fp16, S = Q K^T and O += P V on m16n8k16 with fp32 accumulation, K/V tiles double-
// buffered by cp.async), for k head DK / v head DV, GQA (q head h reads kv head h / group), an optional window and per-head sinks
// (the sink logit joins the softmax denominator), the value scale. Block = one local q head x 64 rows (4 warps x 16).
template <int DK, int DV>
__global__ void __launch_bounds__(128) k_mimo_fa(MimoAttn a) {
    constexpr int QLD = fa6_qld<DK, DV>(), VLD = fa6_vld<DK, DV>();
    extern __shared__ __align__(16) half fa6_smem_[];
    half * qs = fa6_smem_;
    half * kb0 = qs + FA6_BQ * QLD;
    half * kbuf[2] = {kb0, kb0 + FA6_BK * QLD + FA6_BK * VLD};   // [K | V] per buffer
    const int h = blockIdx.y, qt0 = blockIdx.x * FA6_BQ, nt = a.nt;
    const int group = a.nq / a.nkv, hk = h / group;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int pos0 = a.pos[0];
    for (int i = threadIdx.x; i < FA6_BQ * DK / 2; i += blockDim.x) {
        const int r = i / (DK / 2), d = (i % (DK / 2)) * 2, t = qt0 + r;
        float2 v = make_float2(0.0f, 0.0f);
        if (t < nt) v = *(const float2 *) (a.q + (size_t) t * a.qs + (size_t) h * DK + d);
        *(__half2 *) (qs + r * QLD + d) = __floats2half2_rn(v.x * a.scale, v.y * a.scale);
    }
    const int kend = pos0 + min(nt, qt0 + FA6_BQ);
    const int kbeg = a.window ? (max(0, pos0 + qt0 - a.window + 1) & ~(FA6_BK - 1)) : 0;
    const int wrow0 = qt0 + w * 16;
    const int wlast = pos0 + min(nt - 1, wrow0 + 15), wfirst = a.window ? pos0 + wrow0 - a.window + 1 : 0;
    const half * kb = a.kc + (size_t) hk * a.kpitch, * vb = a.vc + (size_t) hk * a.vpitch;
    float o[DV / 8][4];
#pragma unroll
    for (int i = 0; i < DV / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
    const int qp[2] = {pos0 + wrow0 + gid, pos0 + wrow0 + gid + 8};
    auto load_kv = [&](half * kd, int kt) {
        half * vd = kd + FA6_BK * QLD;
        for (int i = threadIdx.x; i < FA6_BK * DK / 8; i += blockDim.x) {
            const int r = i / (DK / 8), d = (i % (DK / 8)) * 8, p = kt + r;
            const bool ok = p < kend;
            fa_cp16(kd + r * QLD + d, kb + (size_t) (ok ? p : 0) * DK + d, ok);
        }
        for (int i = threadIdx.x; i < FA6_BK * DV / 8; i += blockDim.x) {
            const int r = i / (DV / 8), d = (i % (DV / 8)) * 8, p = kt + r;
            const bool ok = p < kend;
            fa_cp16(vd + r * VLD + d, vb + (size_t) (ok ? p : 0) * DV + d, ok);
        }
        asm volatile("cp.async.commit_group;");
    };
    if (kbeg < kend) load_kv(kbuf[0], kbeg);
    for (int kt0 = kbeg; kt0 < kend; kt0 += FA6_BK) {
        __syncthreads();
        const int cur = ((kt0 - kbeg) / FA6_BK) & 1;
        const half * ks = kbuf[cur], * vs = kbuf[cur] + FA6_BK * QLD;
        if (kt0 + FA6_BK < kend) load_kv(kbuf[cur ^ 1], kt0 + FA6_BK);
        else asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        if (kt0 > wlast || wrow0 >= nt || kt0 + FA6_BK <= wfirst) continue;
        float sc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < DK / 16; ++kk) {
            unsigned qa[4];
            fa_ldm4(qa, qs + (w * 16 + (lane & 15)) * QLD + kk * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned r[4];
                fa_ldm4(r, ks + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * QLD + kk * 16 + ((lane >> 3) & 1) * 8);
                const unsigned b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
                fa_mma(sc[2 * np], qa, b0);
                fa_mma(sc[2 * np + 1], qa, b1);
            }
        }
        float tmax[2] = {-1e30f, -1e30f};
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = kt0 + ni * 8 + 2 * tig + (e & 1), q = qp[e >> 1];
                if (key > q || key >= kend || (a.window && key <= q - a.window)) sc[ni][e] = -INFINITY;
                tmax[e >> 1] = fmaxf(tmax[e >> 1], sc[ni][e]);
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
            for (int e = 0; e < 4; ++e) { sc[ni][e] = __expf(sc[ni][e] - m[e >> 1]); l[e >> 1] += sc[ni][e]; }
#pragma unroll
        for (int i = 0; i < DV / 8; ++i) { o[i][0] *= corr[0]; o[i][1] *= corr[0]; o[i][2] *= corr[1]; o[i][3] *= corr[1]; }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const unsigned pa[4] = {fa_h2(sc[2 * kk][0], sc[2 * kk][1]), fa_h2(sc[2 * kk][2], sc[2 * kk][3]),
                                    fa_h2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]), fa_h2(sc[2 * kk + 1][2], sc[2 * kk + 1][3])};
#pragma unroll
            for (int dn = 0; dn < DV / 16; ++dn) {
                unsigned r[4];
                fa_ldm4t(r, vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * VLD + dn * 16 + (lane >> 4) * 8);
                const unsigned b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
                fa_mma(o[2 * dn], pa, b0);
                fa_mma(o[2 * dn + 1], pa, b1);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
    }
    const float sk = a.sinks ? a.sinks[h] : -INFINITY;
#pragma unroll
    for (int e2 = 0; e2 < 2; ++e2) {
        const int t = wrow0 + gid + e2 * 8;
        if (t >= nt) continue;
        const float M = fmaxf(m[e2], sk), em = __expf(m[e2] - M);
        const float den = l[e2] * em + (a.sinks ? __expf(sk - M) : 0.0f);
        const float f = den > 0.0f ? em / den * a.vscale : 0.0f;
        float * dst = a.out + (size_t) t * a.os + (size_t) h * DV;
#pragma unroll
        for (int i = 0; i < DV / 8; ++i) {
            const int d = i * 8 + 2 * tig;
            *(float2 *) (dst + d) = make_float2(o[i][e2 * 2] * f, o[i][e2 * 2 + 1] * f);
        }
    }
}

// ---- tensor-core decode attention (as kernels.cu's k_attn_dec_tc, for k 192 / v 128, GQA, window, ring caches) ----
// Block = one local kv head, one key slice, R = 16 MT (row, q head) pairs as the MMA rows (pair r = row * group + head).
// q (scaled) and p are split into fp16 hi + lo and k, v are exact in fp16: every product fp32-accurate, sums in fp32. Keys in
// tiles of 32 at absolute positions, K/V double-buffered (cp.async); warp w scores keys 8w..8w+7 of a tile and accumulates
// output dims 32w..32w+31. A row's slice depends only on its own position (keys [kmin, p] in ns equal chunks): one row and four
// rows give the same bits (speculative verification = plain decoding). Partials {m, l, -, -, o[128]} for k_mimo_comb.
constexpr int DT6_K = 32, DT6_LDK = 192 + 8, DT6_LDV = 128 + 8, DT6_PL = DT6_K + 8;
template <int MT> __host__ __device__ constexpr int dt6_nbuf() { return MT == 2 ? 2 : 1; }
template <int MT> __host__ __device__ constexpr size_t dt6_smem() {
    return ((size_t) dt6_nbuf<MT>() * DT6_K * (DT6_LDK + DT6_LDV) + 2 * 16 * MT * DT6_PL) * sizeof(half);
}
template <int MT>
__global__ void __launch_bounds__(128 * MT, MT == 1 ? 2 : 1) k_mimo_dec_tc(MimoAttn a, int ns) {
    constexpr int DK = 192, DV = 128, R = 16 * MT, NB = dt6_nbuf<MT>();
    extern __shared__ __align__(16) half dt6_sm[];
    half * kvb = dt6_sm;                                  // [NB][K: DT6_K x LDK | V: DT6_K x LDV]
    half * ph = kvb + NB * DT6_K * (DT6_LDK + DT6_LDV);   // [R][DT6_PL]
    half * pl = ph + R * DT6_PL;
    __shared__ float red[2][4][R];
    const int hk = blockIdx.x, split = blockIdx.y, rg = blockIdx.z;
    const int w = (threadIdx.x >> 5) & 3, g16 = (threadIdx.x >> 7) * 16, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int group = a.nq / a.nkv, pairs = a.nt * group, r0 = rg * R;
    if (r0 >= pairs) return;
    const int np = min(R, pairs - r0);
    const int pos0 = a.pos[0];
    auto slice = [&](int t, int & p0, int & p1) {
        const int npos = pos0 + t + 1, kmin = a.window ? max(0, npos - a.window) : 0, chunk = (npos - kmin + ns - 1) / ns;
        p0 = kmin + split * chunk;
        p1 = min(npos, p0 + chunk);
    };
    int b0 = INT_MAX, b1 = 0;
    for (int t = r0 / group; t <= (r0 + np - 1) / group; ++t) {
        int x, y;
        slice(t, x, y);
        if (x < y) { b0 = min(b0, x); b1 = max(b1, y); }
    }
    half * qh = kvb, * ql = kvb + R * DT6_LDK;   // (staging in the not yet used tile buffers: 2 R LDK <= NB DT6_K (LDK + LDV))
    for (int i = threadIdx.x; i < R * DK; i += blockDim.x) {
        const int r = i / DK, d = i % DK;
        float v = 0.0f;
        if (r < np) {
            const int t = (r0 + r) / group, h = hk * group + (r0 + r) % group;
            v = a.q[(size_t) t * a.qs + (size_t) h * DK + d] * a.scale;
        }
        const half hi = __float2half_rn(v);
        qh[r * DT6_LDK + d] = hi;
        ql[r * DT6_LDK + d] = __float2half_rn(v - __half2float(hi));
    }
    __syncthreads();
    unsigned qah[DK / 16][4], qal[DK / 16][4];
#pragma unroll
    for (int kk = 0; kk < DK / 16; ++kk) {
        fa_ldm4(qah[kk], qh + (g16 + (lane & 15)) * DT6_LDK + kk * 16 + (lane >> 4) * 8);
        fa_ldm4(qal[kk], ql + (g16 + (lane & 15)) * DT6_LDK + kk * 16 + (lane >> 4) * 8);
    }
    int rp0[2], rp1[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        rp0[e] = rp1[e] = 0;
        if (g16 + gid + 8 * e < np) slice((r0 + g16 + gid + 8 * e) / group, rp0[e], rp1[e]);
    }
    float o[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.0f, 0.0f};
    const half * kb = a.kc + (size_t) hk * a.kpitch, * vb = a.vc + (size_t) hk * a.vpitch;
    const bool active = g16 < np;
    const int tb0 = b0 < b1 ? b0 / DT6_K * DT6_K : 0;
    auto load = [&](int buf, int base) {
        half * kd = kvb + (size_t) buf * DT6_K * (DT6_LDK + DT6_LDV), * vd = kd + DT6_K * DT6_LDK;
        for (int i = threadIdx.x; i < DT6_K * (DK + DV) / 8; i += blockDim.x) {
            const bool isk = i < DT6_K * DK / 8;
            const int j = isk ? i : i - DT6_K * DK / 8, per = (isk ? DK : DV) / 8, r = j / per, d = (j % per) * 8, p = base + r;
            const bool ok = p >= b0 && p < b1;
            const size_t row = ok ? (a.ring ? (size_t) (p % a.ring) : (size_t) p) : 0;
            if (isk) fa_cp16(kd + r * DT6_LDK + d, kb + row * DK + d, ok);
            else fa_cp16(vd + r * DT6_LDV + d, vb + row * DV + d, ok);
        }
        asm volatile("cp.async.commit_group;");
    };
    __syncthreads();   // (the q staging is read)
    if (NB == 2 && tb0 < b1) load(0, tb0);
    int buf = 0;
    for (int base = tb0; base < b1; base += DT6_K, buf ^= (NB - 1)) {
        if (NB == 2) {
            if (base + DT6_K < b1) load(buf ^ 1, base + DT6_K);
            else asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            if (base > tb0) __syncthreads();
            load(0, base);
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const half * ks = kvb + (size_t) buf * DT6_K * (DT6_LDK + DT6_LDV), * vs = ks + DT6_K * DT6_LDK;
        float shi[3][4], slo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int c = 0; c < 3; ++c) shi[c][0] = shi[c][1] = shi[c][2] = shi[c][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < DK / 16; ++kk) {
            if (!active) break;
            unsigned b[2];
            const unsigned ka = (unsigned) __cvta_generic_to_shared(ks + (w * 8 + (lane & 7)) * DT6_LDK + kk * 16 + ((lane >> 3) & 1) * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b[0]), "=r"(b[1]) : "r"(ka));
            fa_mma(shi[kk >> 2], qah[kk], b);
            fa_mma(slo, qal[kk], b);
        }
        float sc[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[e] = ((shi[0][e] + shi[1][e]) + shi[2][e]) + slo[e];
        float tmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int key = base + w * 8 + 2 * tig + (e & 1), r = e >> 1;
            if (key < rp0[r] || key >= rp1[r]) sc[e] = -INFINITY;
            tmax[r] = fmaxf(tmax[r], sc[e]);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 2));
            if (tig == 0) red[0][w][g16 + gid + 8 * r] = tmax[r];
        }
        __syncthreads();
        float corr[2], psum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = g16 + gid + 8 * r;
            const float tm = fmaxf(fmaxf(red[0][0][row], red[0][1][row]), fmaxf(red[0][2][row], red[0][3][row]));
            const float mn = fmaxf(m[r], tm);
            corr[r] = __expf(m[r] - mn);
            m[r] = mn;
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int r = e >> 1;
            const float p = __expf(sc[e] - m[r]);
            psum[r] += p;
            const half hi = __float2half_rn(p);
            const int at = (g16 + gid + 8 * r) * DT6_PL + w * 8 + 2 * tig + (e & 1);
            ph[at] = hi;
            pl[at] = __float2half_rn(p - __half2float(hi));
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            psum[r] += __shfl_xor_sync(0xffffffff, psum[r], 1);
            psum[r] += __shfl_xor_sync(0xffffffff, psum[r], 2);
            if (tig == 0) red[1][w][g16 + gid + 8 * r] = psum[r];
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = g16 + gid + 8 * r;
            l[r] = l[r] * corr[r] + (((red[1][0][row] + red[1][1][row]) + red[1][2][row]) + red[1][3][row]);
        }
        float ot[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) ot[n][0] = ot[n][1] = ot[n][2] = ot[n][3] = 0.0f;
#pragma unroll
        for (int k2 = 0; k2 < DT6_K / 16; ++k2) {
            if (!active) break;
            unsigned ahi[4], alo[4];
            fa_ldm4(ahi, ph + (g16 + (lane & 15)) * DT6_PL + k2 * 16 + (lane >> 4) * 8);
            fa_ldm4(alo, pl + (g16 + (lane & 15)) * DT6_PL + k2 * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int dn = 0; dn < 2; ++dn) {
                unsigned v4[4];
                fa_ldm4t(v4, vs + (k2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * DT6_LDV + w * 32 + dn * 16 + (lane >> 4) * 8);
                const unsigned c0[2] = {v4[0], v4[1]}, c1[2] = {v4[2], v4[3]};
                fa_mma(ot[2 * dn], ahi, c0);
                fa_mma(ot[2 * dn], alo, c0);
                fa_mma(ot[2 * dn + 1], ahi, c1);
                fa_mma(ot[2 * dn + 1], alo, c1);
            }
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            o[n][0] = o[n][0] * corr[0] + ot[n][0]; o[n][1] = o[n][1] * corr[0] + ot[n][1];
            o[n][2] = o[n][2] * corr[1] + ot[n][2]; o[n][3] = o[n][3] * corr[1] + ot[n][3];
        }
        if (NB == 2) __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = g16 + gid + 8 * r;
        if (row >= np) continue;
        const int qi = r0 + row;
        float * dst = a.part + (((size_t) hk * ns + split) * MIMO_QB + qi) * (DV + 4);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int d = w * 32 + n * 8 + 2 * tig;
            dst[4 + d] = o[n][2 * r];
            dst[4 + d + 1] = o[n][2 * r + 1];
        }
        if (w == 0 && tig == 0) { dst[0] = m[r] == -1e30f ? -INFINITY : m[r]; dst[1] = l[r]; }
    }
}

} // namespace

void mimo_attn_fa(const MimoAttn & a, cudaStream_t s) {
    if (a.dk != 192 || a.dv != 128 || a.ring) throw std::runtime_error("mimo_attn_fa: only k 192 / v 128 full caches");
    static bool attr[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 16 && !attr[dev]) {
        cudaFuncSetAttribute(k_mimo_fa<192, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) fa6_smem<192, 128>());
        attr[dev] = true;
    }
    k_mimo_fa<192, 128><<<dim3((a.nt + FA6_BQ - 1) / FA6_BQ, a.nq), 128, fa6_smem<192, 128>(), s>>>(a);
}

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
    static const bool tc_off = getenv("HYPER6_DEC_TC") && atoi(getenv("HYPER6_DEC_TC")) == 0;
    if (!tc_off && nrb == 1 && a.nt * group <= MIMO_QB) {   // decode-sized rows: the tensor-core kernel, partials merged by the combine
        static bool tattr[16] = {};
        if (dev < 16 && !tattr[dev]) {
            cudaFuncSetAttribute(k_mimo_dec_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dt6_smem<1>());
            cudaFuncSetAttribute(k_mimo_dec_tc<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dt6_smem<2>());
            tattr[dev] = true;
        }
        if (!a.part) throw std::runtime_error("mimo_attn: the tensor-core kernel needs a partial buffer");
        const int pairs = a.nt * group;
        if (pairs > 16) k_mimo_dec_tc<2><<<dim3(a.nkv, ns, (pairs + 31) / 32), 256, dt6_smem<2>(), s>>>(a, ns);
        else k_mimo_dec_tc<1><<<dim3(a.nkv, ns, 1), 128, dt6_smem<1>(), s>>>(a, ns);
        k_mimo_comb<<<dim3(a.nq, a.nt), 128, 0, s>>>(a, ns, RB, 1);
        return;
    }
    k_mimo_attn<192><<<dim3(a.nkv, ns, nrb), 256, at_smem<192>(), s>>>(a, ns, RB);
    if (ns > 1) k_mimo_comb<<<dim3(a.nq, a.nt), 128, 0, s>>>(a, ns, RB, nrb);
}

} // namespace hyper
