// Megakernel go/no-go: a chain of dependent Q8 GEMVs with the per-GPU decode shapes of a Flash-Next layer (48 layers,
// distinct weights), (A) one kernel per GEMV in a CUDA graph (hyper's gemv_q8, as the engine runs today) vs (B) one
// persistent kernel: all blocks co-resident, warps take (row tile, k slice) units, a grid barrier between GEMVs, and
// (B2) the next GEMV's weights prefetched into L2 while waiting at the barrier. Outputs compared. usage: mkbench [layers]
// Measured 2026-10-09 (RTX 3090, 48 layers = 384 GEMVs, 1.74 GB of weights): A 3.68 ms (472 GB/s); B 4.54 ms; B + L2
// prefetch 5.17 ms; B + register prefetch of the next GEMV 5.63 ms (1 block / SM: registers) .. 6.08 ms (2 blocks / SM,
// spills). A graph boundary on Ampere is cheaper than a grid barrier, and tuned per-shape kernels beat one generic
// decomposition: the megakernel route is not pursued. (MK_BPSM: blocks per SM, MK_MINKB: minimum k blocks per unit)
#include "../src/kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ unsigned i8x2_to_h2(unsigned v, unsigned sel) {
    const unsigned xx = __byte_perm(v ^ 0x80808080u, 0x64646464u, sel);
    unsigned r;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(xx), "r"(0x64806480u));
    return r;
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) { __half2 h = __floats2half2_rn(lo, hi); return *(const unsigned *) &h; }
__device__ __forceinline__ void mma16816(float * c, const unsigned * a, const unsigned * b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Step { const uint4 * q; const half * s; int n, k, upt; const float * x; float * y; };

__device__ __forceinline__ void grid_sync(unsigned * bar, unsigned & gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned g = gen;
        if (atomicAdd(&bar[0], 1u) == gridDim.x - 1) { bar[0] = 0; __threadfence(); atomicExch(&bar[1], g + 1); }
        else while (*(volatile unsigned *) &bar[1] == g) {}
        __threadfence();
    }
    ++gen;
    __syncthreads();
}

// one unit: tile rows x k blocks [b0, b1) of step st, one token (x read through L2: written by other blocks)
__device__ __forceinline__ void unit(const Step & st, int tile, int b0, int b1, float * acc) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3, kb = st.k / 32;
    const uint4 * tq = st.q + (size_t) tile * kb * 32 + lane;
    const half * ts = st.s + (size_t) tile * kb * 16;
#pragma unroll 2
    for (int b = b0; b < b1; ++b) {
        const uint4 q = __ldg(tq + (size_t) b * 32);
        const float s_lo = __half2float(ts[(size_t) b * 16 + gid]), s_hi = __half2float(ts[(size_t) b * 16 + gid + 8]);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        float tmp[4] = {0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned a[4];
            a[0] = i8x2_to_h2(qw[2 * ks], 0x5140); a[1] = i8x2_to_h2(qw[2 * ks], 0x7362);
            a[2] = i8x2_to_h2(qw[2 * ks + 1], 0x5140); a[3] = i8x2_to_h2(qw[2 * ks + 1], 0x7362);
            const int c0 = b * 32 + ks * 16 + 2 * tig;
            unsigned bb[2] = {0, 0};
            if (gid == 0) {
                const float2 v0 = __ldcg((const float2 *) (st.x + c0)), v1 = __ldcg((const float2 *) (st.x + c0 + 8));
                bb[0] = pack_h2(v0.x, v0.y); bb[1] = pack_h2(v1.x, v1.y);
            }
            mma16816(tmp, a, bb);
        }
        acc[0] += tmp[0] * s_lo; acc[2] += tmp[2] * s_hi;
    }
}

constexpr int KB_PRE = 16;
struct Pre { uint4 q[KB_PRE]; half2 s[KB_PRE]; };
__device__ __forceinline__ void pre_load(const Step & st, int tile, int b0, int b1, Pre & P) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, kb = st.k / 32;
    const uint4 * tq = st.q + (size_t) tile * kb * 32 + lane;
    const half * ts = st.s + (size_t) tile * kb * 16;
#pragma unroll
    for (int i = 0; i < KB_PRE; ++i)
        if (b0 + i < b1) { P.q[i] = __ldg(tq + (size_t) (b0 + i) * 32); P.s[i] = __halves2half2(ts[(size_t) (b0 + i) * 16 + gid], ts[(size_t) (b0 + i) * 16 + gid + 8]); }
}
__device__ __forceinline__ void unit_pre(const Step & st, int b0, int b1, const Pre & P, float * acc) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int i = 0; i < KB_PRE; ++i) {
        if (b0 + i >= b1) break;
        const int b = b0 + i;
        const unsigned qw[4] = {P.q[i].x, P.q[i].y, P.q[i].z, P.q[i].w};
        float tmp[4] = {0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned a[4];
            a[0] = i8x2_to_h2(qw[2 * ks], 0x5140); a[1] = i8x2_to_h2(qw[2 * ks], 0x7362);
            a[2] = i8x2_to_h2(qw[2 * ks + 1], 0x5140); a[3] = i8x2_to_h2(qw[2 * ks + 1], 0x7362);
            const int c0 = b * 32 + ks * 16 + 2 * tig;
            unsigned bb[2] = {0, 0};
            if (gid == 0) {
                const float2 v0 = __ldcg((const float2 *) (st.x + c0)), v1 = __ldcg((const float2 *) (st.x + c0 + 8));
                bb[0] = pack_h2(v0.x, v0.y); bb[1] = pack_h2(v1.x, v1.y);
            }
            mma16816(tmp, a, bb);
        }
        acc[0] += tmp[0] * __low2float(P.s[i]); acc[2] += tmp[2] * __high2float(P.s[i]);
    }
}

__global__ void __launch_bounds__(256) k_chain(const Step * __restrict__ steps, int nsteps, float * part, unsigned * cnt, unsigned * bar,
                                               int prefetch) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int gw = blockIdx.x * 8 + (threadIdx.x >> 5), GW = gridDim.x * 8;
    unsigned gen = 0;
    if (threadIdx.x == 0) gen = *(volatile unsigned *) &bar[1];
    Pre P;
    bool have = false;   // P holds this warp's first unit of the current step (prefetch == 2)
    if (prefetch == 2) {
        const Step st = steps[0];
        const int kb = st.k / 32, units = ((st.n + 15) / 16) * st.upt;
        if (gw < units && kb / st.upt <= KB_PRE) {
            const int tile = gw / st.upt, kc = gw % st.upt;
            pre_load(st, tile, (int) ((int64_t) kc * kb / st.upt), (int) ((int64_t) (kc + 1) * kb / st.upt), P);
            have = true;
        }
    }
    for (int si = 0; si < nsteps; ++si) {
        const Step st = steps[si];
        const int kb = st.k / 32, tiles = (st.n + 15) / 16, units = tiles * st.upt;
        for (int u = gw; u < units; u += GW) {
            const int tile = u / st.upt, kc = u % st.upt;
            const int b0 = (int) ((int64_t) kc * kb / st.upt), b1 = (int) ((int64_t) (kc + 1) * kb / st.upt);
            float acc[4] = {0, 0, 0, 0};
            if (u == gw && have) unit_pre(st, b0, b1, P, acc);
            else unit(st, tile, b0, b1, acc);
            if (st.upt == 1) {
                if (tig == 0) { const int r = tile * 16 + gid; if (r < st.n) st.y[r] = acc[0]; if (r + 8 < st.n) st.y[r + 8] = acc[2]; }
                continue;
            }
            float * dst = part + (size_t) u * 16;
            if (tig == 0) { dst[gid] = acc[0]; dst[gid + 8] = acc[2]; }
            __threadfence();
            __syncwarp();
            unsigned ticket = 0;
            if (lane == 0) ticket = atomicAdd(&cnt[tile], 1u);
            ticket = __shfl_sync(0xffffffff, ticket, 0);
            if (ticket != (unsigned) st.upt - 1) continue;
            __threadfence();
            if (lane < 16) {
                float v = 0.0f;
                for (int c = 0; c < st.upt; ++c) v += __ldcg(part + ((size_t) tile * st.upt + c) * 16 + lane);
                const int r = tile * 16 + lane;
                if (r < st.n) st.y[r] = v;
            }
            if (lane == 0) cnt[tile] = 0;
        }
        have = false;
        if (si + 1 < nsteps && prefetch == 2) {   // the next GEMV's first unit of this warp -> registers, before the barrier
            const Step nx = steps[si + 1];
            const int nkb = nx.k / 32, nunits = ((nx.n + 15) / 16) * nx.upt;
            if (gw < nunits && nkb / nx.upt <= KB_PRE) {
                const int tile = gw / nx.upt, kc = gw % nx.upt;
                pre_load(nx, tile, (int) ((int64_t) kc * nkb / nx.upt), (int) ((int64_t) (kc + 1) * nkb / nx.upt), P);
                have = true;
            }
        }
        if (si + 1 < nsteps && prefetch == 1) {   // the next GEMV's weights for this warp's units -> L2, while others finish
            const Step nx = steps[si + 1];
            const int nkb = nx.k / 32, ntiles = (nx.n + 15) / 16, nunits = ntiles * nx.upt;
            for (int u = gw; u < nunits; u += GW) {
                const int tile = u / nx.upt, kc = u % nx.upt;
                const int b0 = (int) ((int64_t) kc * nkb / nx.upt), b1 = (int) ((int64_t) (kc + 1) * nkb / nx.upt);
                const char * p0 = (const char *) (nx.q + ((size_t) tile * nkb + b0) * 32), * p1 = (const char *) (nx.q + ((size_t) tile * nkb + b1) * 32);
                for (const char * p = p0 + lane * 128; p < p1; p += 32 * 128) asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
            }
        }
        grid_sync(bar, gen);
    }
}

int main(int argc, char ** argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 48;
    struct Shape { int n, k; };
    const Shape layer[] = {{320, 10240}, {10240, 320}, {5461, 2560}, {2560, 2048}, {320, 10240}, {10240, 320}, {426, 2560}, {2560, 224}};
    const int per = sizeof(layer) / sizeof(layer[0]), S = layers * per;
    CK(cudaSetDevice(0));
    hyper::gemv_init(0);
    std::mt19937 rng(1);
    std::vector<hyper::Q8W> W(S);
    size_t total = 0;
    for (int i = 0; i < S; ++i) {
        const Shape sh = layer[i % per];
        const size_t tiles = (sh.n + 15) / 16, kb = sh.k / 32, qb = tiles * kb * 512, sb = tiles * kb * 16 * 2;
        std::vector<uint8_t> hq(qb); for (auto & b : hq) b = (rng() & 1) ? 1 : 255;   // +-1: variance-preserving with scale 1/sqrt(k)
        std::vector<half> hs(tiles * kb * 16); for (auto & v : hs) v = __float2half(1.0f / sqrtf((float) sh.k));
        void * q, * s;
        CK(cudaMalloc(&q, qb)); CK(cudaMalloc(&s, sb));
        CK(cudaMemcpy(q, hq.data(), qb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(s, hs.data(), sb, cudaMemcpyHostToDevice));
        W[i].q = (const uint4 *) q; W[i].s = (const half *) s; W[i].n = sh.n; W[i].k = sh.k;
        total += qb + sb;
    }
    float * buf[2];
    for (auto & b : buf) CK(cudaMalloc(&b, 16384 * 4));
    std::vector<float> hx(16384); for (auto & v : hx) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    auto reset = [&] { CK(cudaMemcpy(buf[0], hx.data(), 16384 * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(buf[1], 0, 16384 * 4)); };
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // (A) graph of gemv_q8 calls, ping-pong buffers
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < S; ++i) hyper::gemv_q8(W[i], buf[i & 1], W[i].k, buf[(i + 1) & 1], W[i].n, nullptr, 1, s);
    CK(cudaStreamEndCapture(s, &gr)); CK(cudaGraphInstantiate(&ge, gr, 0));
    reset(); CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    std::vector<float> ya(16384); CK(cudaMemcpy(ya.data(), buf[S & 1], 16384 * 4, cudaMemcpyDeviceToHost));
    float best = 1e9;
    for (int r = 0; r < 5; ++r) {
        reset(); CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
    }
    printf("A graph of %d gemv_q8:     %.3f ms per %d layers (%.1f us/GEMV, %.0f GB/s)\n", S, best, layers, 1000 * best / S, total / (best * 1e-3) / 1e9);
    // (B) persistent kernel
    int nb = 0, sms = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_chain, 256, 0));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int bpsm = getenv("MK_BPSM") ? atoi(getenv("MK_BPSM")) : nb;
    const int blocks = std::min(nb, bpsm) * sms, GW = blocks * 8;
    std::vector<Step> hs(S);
    for (int i = 0; i < S; ++i) {
        const int tiles = (W[i].n + 15) / 16, kb = W[i].k / 32;
        static const int minkb = getenv("MK_MINKB") ? atoi(getenv("MK_MINKB")) : 4;
        int upt = std::max(1, std::min({kb / minkb, (GW + tiles - 1) / tiles}));   // enough units for every warp, >= minkb k blocks each
        hs[i] = {W[i].q, W[i].s, W[i].n, W[i].k, upt, buf[i & 1], buf[(i + 1) & 1]};
    }
    Step * ds; CK(cudaMalloc(&ds, S * sizeof(Step))); CK(cudaMemcpy(ds, hs.data(), S * sizeof(Step), cudaMemcpyHostToDevice));
    float * part; unsigned * cnt, * bar;
    CK(cudaMalloc(&part, (size_t) 64 * GW * 16 * 4)); CK(cudaMalloc(&cnt, 65536 * 4)); CK(cudaMemset(cnt, 0, 65536 * 4));
    CK(cudaMalloc(&bar, 8)); CK(cudaMemset(bar, 0, 8));
    for (int pf = 0; pf < 3; ++pf) {
        reset(); k_chain<<<blocks, 256, 0, s>>>(ds, S, part, cnt, bar, pf); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
        std::vector<float> yb(16384); CK(cudaMemcpy(yb.data(), buf[S & 1], 16384 * 4, cudaMemcpyDeviceToHost));
        double md = 0, mx = 0; for (int i = 0; i < W[S - 1].n; ++i) { md = std::max(md, (double) std::fabs(yb[i] - ya[i])); mx = std::max(mx, (double) std::fabs(ya[i])); }
        best = 1e9;
        for (int r = 0; r < 5; ++r) {
            reset(); CK(cudaEventRecord(e0, s)); k_chain<<<blocks, 256, 0, s>>>(ds, S, part, cnt, bar, pf); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
        }
        printf("B persistent%s (%d blocks): %.3f ms per %d layers (%.1f us/GEMV, %.0f GB/s)  max |B-A| %.3g of %.3g\n", pf == 2 ? "+reg prefetch" : pf ? "+L2 prefetch" : "            ",
               blocks, best, layers, 1000 * best / S, total / (best * 1e-3) / 1e9, md, mx);
    }
    return 0;
}
