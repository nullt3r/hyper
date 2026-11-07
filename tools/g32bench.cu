// decode gate/up of one layer's experts (real GGUF blocks): moe_gate_up time per call and bandwidth, 6 pairs all local
#include "../src/gguf.h"
#include "../src/kernels4.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace hyper;

#include "../src/iq_grids.h"
__device__ __forceinline__ float h2fb(const uint8_t * p) { return __half2float(*(const __half *) p); }
// variant A: a warp per row, lanes over 32-value groups, grid / signs from global (L1), x from global (L1)

// variant B: a warp takes RW rows of gate and of up at once (2 RW independent row streams), groups unrolled
__device__ __forceinline__ float iq2xxs_g32(const uint8_t * __restrict__ row, int g, const float * __restrict__ x) {
    const uint8_t * b = row + (size_t) (g >> 3) * 66;
    const int ib = g & 7;
    const uint16_t * q16 = (const uint16_t *) (b + 2 + ib * 8);
    const uint32_t a0 = q16[0] | ((uint32_t) q16[1] << 16), a1 = q16[2] | ((uint32_t) q16[3] << 16);
    const float db = h2fb(b) * (0.5f + (float) (a1 >> 28)) * 0.25f;
    const float4 * xv = (const float4 *) (x + g * 32);
    float s = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint64_t gr = __ldg(&g_iq2xxs_grid[(a0 >> (8 * l)) & 0xff]);
        const uint32_t sg = __ldg(&g_ksigns_iq2xs[(a1 >> (7 * l)) & 127]);
        const uint32_t lo = (uint32_t) gr, hi = (uint32_t) (gr >> 32);
        const float4 x0 = __ldg(xv + 2 * l), x1 = __ldg(xv + 2 * l + 1);
        const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = (float) (((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xff) * xs[j];
            s += (sg >> j) & 1 ? -v : v;
        }
    }
    return db * s;
}
template <int RW, int WB = 8>
__global__ void __launch_bounds__(256) k_gu_b(MoeDev m, const float * __restrict__ x, const int * __restrict__ ids, int k, float * __restrict__ h, int kdim) {
    const int p = blockIdx.x;
    const int slot = m.slot[ids[p]];
    if (slot < 0) return;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = (blockIdx.y * WB + w) * RW;
    if (r0 >= m.ff) return;
    const size_t rb = m.gate_bytes / m.ff;
    const uint8_t * gbase = m.gate + slot * m.gate_bytes, * ubase = m.up + slot * m.gate_bytes;
    float ag[RW], au[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) ag[i] = au[i] = 0.0f;
    for (int g = lane; g < kdim / 32; g += 32) {
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            ag[i] += iq2xxs_g32(gbase + (r0 + i) * rb, g, x);
            au[i] += iq2xxs_g32(ubase + (r0 + i) * rb, g, x);
        }
    }
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        float a = ag[i], u = au[i];
        for (int o = 16; o > 0; o >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); u += __shfl_xor_sync(0xffffffffu, u, o); }
        if (lane == 0) { const float gg = fminf(a, 10.0f), uu = fminf(fmaxf(u, -10.0f), 10.0f); h[(size_t) p * m.ff + r0 + i] = gg / (1.0f + expf(-gg)) * uu; }
    }
}
template <int VAR>
__global__ void __launch_bounds__(256) k_gu_a(MoeDev m, const float * __restrict__ x, const int * __restrict__ ids, int k, float * __restrict__ h, int kdim) {
    const int p = blockIdx.x;
    const int slot = m.slot[ids[p]];
    if (slot < 0) return;
    __shared__ uint64_t sg2[256];
    __shared__ uint8_t sks[128];
    if (VAR == 1) { for (int i = threadIdx.x; i < 256; i += 256) sg2[i] = g_iq2xxs_grid[i]; if (threadIdx.x < 128) sks[threadIdx.x] = g_ksigns_iq2xs[threadIdx.x]; __syncthreads(); }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, r = blockIdx.y * 8 + w;
    if (r >= m.ff) return;
    const size_t rb = m.gate_bytes / m.ff;
    float acc[2] = {0, 0};
    for (int q = 0; q < 2; ++q) {
        const uint8_t * row = (q ? m.up : m.gate) + slot * m.gate_bytes + r * rb;
        float a = 0.0f;
        for (int g = lane; g < kdim / 32; g += 32) {
            const uint8_t * b = row + (size_t) (g >> 3) * 66;
            const int ib = g & 7;
            const uint16_t * q16 = (const uint16_t *) (b + 2 + ib * 8);
            const uint32_t a0 = q16[0] | ((uint32_t) q16[1] << 16), a1 = q16[2] | ((uint32_t) q16[3] << 16);
            const float db = h2fb(b) * (0.5f + (float) (a1 >> 28)) * 0.25f;
            const float4 * xv = (const float4 *) (x + g * 32);
            float s = 0.0f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint64_t gr = VAR == 1 ? sg2[(a0 >> (8 * l)) & 0xff] : __ldg(&g_iq2xxs_grid[(a0 >> (8 * l)) & 0xff]);
                const uint32_t sg = VAR == 1 ? sks[(a1 >> (7 * l)) & 127] : __ldg(&g_ksigns_iq2xs[(a1 >> (7 * l)) & 127]);
                const uint32_t lo = (uint32_t) gr, hi = (uint32_t) (gr >> 32);
                const float4 x0 = __ldg(xv + 2 * l), x1 = __ldg(xv + 2 * l + 1);
                const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float v = (float) (((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xff) * xs[j];
                    s += (sg >> j) & 1 ? -v : v;
                }
            }
            a += db * s;
        }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        acc[q] = a;
    }
    if (lane == 0) { const float g = fminf(acc[0], 10.0f), u = fminf(fmaxf(acc[1], -10.0f), 10.0f); h[(size_t) p * m.ff + r] = g / (1.0f + expf(-g)) * u; }
}
int main(int argc, char ** argv) {
    // G32_SYN=mxfp4: synthetic MXFP4 experts (DeepSeek's DSpark drafter: 128 x [5120 -> 2304]) instead of a GGUF layer
    const bool syn = getenv("G32_SYN") && std::string(getenv("G32_SYN")) == "mxfp4";
    GGUF g(argv[1]);
    const int layer = argc > 2 ? atoi(argv[2]) : 10, iters = 200, K = 6;
    const std::string p = syn ? "syn." : "blk." + std::to_string(layer) + ".";
    GTensor tsyn_g, tsyn_u, tsyn_d;
    std::vector<uint8_t> syn_g, syn_u, syn_d;
    if (syn) {
        std::mt19937 r0(7);
        auto fill = [&](std::vector<uint8_t> & v, GTensor & t, int64_t k, int64_t rows, int64_t e) {
            v.resize((size_t) k / 32 * 17 * rows * e);
            const int e0 = getenv("G32_SYN_E0") ? atoi(getenv("G32_SYN_E0")) : 118, er = getenv("G32_SYN_ER") ? atoi(getenv("G32_SYN_ER")) : 20;   // E8M0 scale range
            for (size_t i = 0; i < v.size(); ++i) v[i] = i % 17 == 0 ? (uint8_t) (e0 + r0() % er) : (uint8_t) r0();
            t.type = GType::MXFP4; t.ne = {k, rows, e}; t.nbytes = v.size(); t.data = v.data(); t.name = "syn";
        };
        fill(syn_g, tsyn_g, 5120, 2304, 64); fill(syn_u, tsyn_u, 5120, 2304, 64); fill(syn_d, tsyn_d, 2304, 5120, 64);
    }
    const GTensor & tg = syn ? tsyn_g : g.need(p + "ffn_gate_exps.weight"), & tu = syn ? tsyn_u : g.need(p + "ffn_up_exps.weight");
    const int n = (int) tg.ne[0], ff = (int) tg.ne[1], E = (int) tg.ne[2];
    const size_t gb = tg.nbytes / E;
    const int NE = getenv("G32_NE") ? atoi(getenv("G32_NE")) : 64;   // (resident experts; small next to a running server)
    uint8_t * dg, * du; cudaMalloc(&dg, NE * gb); cudaMalloc(&du, NE * gb);
    cudaMemcpy(dg, tg.data, NE * gb, cudaMemcpyHostToDevice); cudaMemcpy(du, tu.data, NE * gb, cudaMemcpyHostToDevice);
    std::vector<int> slot(E, -1); for (int e = 0; e < NE; ++e) slot[e] = e;
    int * dslot; cudaMalloc(&dslot, E * 4); cudaMemcpy(dslot, slot.data(), E * 4, cudaMemcpyHostToDevice);
    MoeDev m; m.gate = dg; m.up = du; m.tg = tg.type; m.gate_bytes = gb; m.slot = dslot; m.ff = ff; m.n = n; m.clamp = 10.0f;
    std::vector<float> x(n); std::mt19937 rng(1); for (auto & v : x) v = std::normal_distribution<float>()(rng);
    float * dx, * dh; cudaMalloc(&dx, n * 4); cudaMalloc(&dh, K * ff * 4); cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
    int * dids; cudaMalloc(&dids, K * 4 * iters);
    std::vector<int> ids(K * iters); for (auto & v : ids) v = rng() % NE;
    cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int it = 0; it < 10; ++it) moe_gate_up(m, dx, n, dids, K, dh, 1, s);
    cudaStreamSynchronize(s);
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) moe_gate_up(m, dx, n, dids + K * it, K, dh, 1, s);
    cudaStreamSynchronize(s);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::vector<float> h(K * ff); cudaMemcpy(h.data(), dh, h.size() * 4, cudaMemcpyDeviceToHost);
    double cs = 0; for (float v : h) cs += v;
    if (const char * dp = getenv("G32_DUMP")) {   // the production dispatch's outputs of the last id set (kernel A/B across processes)
        moe_gate_up(m, dx, n, dids, K, dh, 1, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(h.data(), dh, h.size() * 4, cudaMemcpyDeviceToHost);
        FILE * f = fopen(dp, "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f);
    }
    for (int var = 0; var < 6; ++var) {
        auto run = [&](int it) {
            if (var == 0) k_gu_a<0><<<dim3(K, ff / 8), 256, 0, s>>>(m, dx, dids + K * it, K, dh, n);
            else if (var == 1) k_gu_b<1><<<dim3(K, ff / 8), 256, 0, s>>>(m, dx, dids + K * it, K, dh, n);
            else if (var == 2) k_gu_b<2><<<dim3(K, ff / 16), 256, 0, s>>>(m, dx, dids + K * it, K, dh, n);
            else if (var == 3) k_gu_b<4><<<dim3(K, ff / 32), 256, 0, s>>>(m, dx, dids + K * it, K, dh, n);
            else if (var == 4) k_gu_b<4, 4><<<dim3(K, ff / 16), 128, 0, s>>>(m, dx, dids + K * it, K, dh, n);
            else k_gu_b<4, 2><<<dim3(K, ff / 8), 64, 0, s>>>(m, dx, dids + K * it, K, dh, n);
        };
        for (int it = 0; it < 10; ++it) run(it);
        cudaStreamSynchronize(s);
        cudaMemset(dh, 0, K * ff * 4);
        auto t1 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) run(it);
        cudaStreamSynchronize(s);
        const double d2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() / iters;
        std::vector<float> h2(K * ff); cudaMemcpy(h2.data(), dh, h2.size() * 4, cudaMemcpyDeviceToHost);
        double c2 = 0; for (float v : h2) c2 += v;
        printf("variant %d: %.1f us/call, %.1f GB/s, checksum %.6g (%s)\n", var, d2 * 1e6, 6.0 * 2 * gb / d2 / 1e9, c2, cudaGetErrorString(cudaGetLastError()));
    }
    printf("G32BENCH %s %s: %.1f us/call, %.1f GB/s (6 experts gate+up %.1f MB), checksum %.9g\n", p.c_str(), gtype_name(tg.type), dt * 1e6,
           6.0 * 2 * gb / dt / 1e9, 6.0 * 2 * gb / 1e6, cs);
    {   // the down projection of the same 6 pairs (production dispatch: moe_down), time and checksum
        const GTensor & td = syn ? tsyn_d : g.need(p + "ffn_down_exps.weight");
        const size_t db = td.nbytes / E;
        uint8_t * dd; cudaMalloc(&dd, NE * db);
        cudaMemcpy(dd, td.data, NE * db, cudaMemcpyHostToDevice);
        MoeDev md = m; md.down = dd; md.td = td.type; md.down_bytes = db;
        std::vector<float> wv(K * iters, 0.5f);
        float * dw, * dy; cudaMalloc(&dw, wv.size() * 4); cudaMalloc(&dy, (size_t) K * n * 4);
        cudaMemcpy(dw, wv.data(), wv.size() * 4, cudaMemcpyHostToDevice);
        moe_gate_up(m, dx, n, dids, K, dh, 1, s);   // (h of the first id set)
        for (int it = 0; it < 10; ++it) moe_down(md, dh, dids, dw, K, dy, 1, s);
        cudaStreamSynchronize(s);
        auto t2 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) moe_down(md, dh, dids, dw, K, dy, 1, s);
        cudaStreamSynchronize(s);
        const double d3 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count() / iters;
        std::vector<float> yv((size_t) K * n); cudaMemcpy(yv.data(), dy, yv.size() * 4, cudaMemcpyDeviceToHost);
        if (const char * dp = getenv("G32_DUMP")) { std::string f2 = std::string(dp) + ".down"; FILE * f = fopen(f2.c_str(), "wb"); fwrite(yv.data(), 4, yv.size(), f); fclose(f); }
        double c3 = 0, a3 = 0; for (float v : yv) { c3 += v; a3 += fabs(v); }
        printf("G32BENCH down %s: %.1f us/call, %.1f GB/s, checksum %.9g (abs %.9g) (%s)\n", gtype_name(td.type), d3 * 1e6, 6.0 * db / d3 / 1e9, c3, a3,
               cudaGetErrorString(cudaGetLastError()));
    }
    return 0;
}
