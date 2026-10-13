// Prefill grouped expert GEMMs (moe_gemm_gate_up + moe_gemm_down) on one GPU, Flash-Next shapes (n 2560, ff 640, top-10 of 512,
// gate/up Q4_K, down Q5_1 or Q8_0) with random valid blocks and random routing: time per layer call and hashes of the outputs
// of the local pairs, to check a kernel change bit for bit. usage: moegemmbench [nt=2048] [local=170] [down: q51|q80] [iters=10]
#include "../src/kernels4.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
using namespace hyper;

static unsigned long long fnv(const void * p, size_t n, unsigned long long h = 1469598103934665603ull) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    const int nt = argc > 1 ? atoi(argv[1]) : 2048, NE = argc > 2 ? atoi(argv[2]) : 170, iters = argc > 4 ? atoi(argv[4]) : 10;
    const std::string dt = argc > 3 ? argv[3] : "q51";
    const GType td = dt == "q80" ? GType::Q8_0 : GType::Q5_1;
    const int n = 2560, ff = 640, K = 10, E = 512;
    const size_t gb = (size_t) ff * (n / 256) * 144, db = (size_t) n * (ff / 32) * gtype_block_bytes(td);
    std::mt19937 rng(5);
    auto fill = [&](std::vector<uint8_t> & v, GType t) {
        for (auto & b : v) b = (uint8_t) rng();
        const size_t bb = gtype_block_bytes(t);
        for (size_t i = 0; i < v.size() / bb; ++i) {
            uint8_t * blk = v.data() + i * bb;
            const __half d = __float2half(0.002f * (1 + rng() % 8)), m = __float2half(-0.001f * (1 + rng() % 8));
            memcpy(blk, &d, 2);
            if (t == GType::Q4_K || t == GType::Q5_1) memcpy(blk + 2, &m, 2);
        }
    };
    std::vector<uint8_t> hg(gb * NE), hu(gb * NE), hd(db * NE);
    fill(hg, GType::Q4_K); fill(hu, GType::Q4_K); fill(hd, td);
    uint8_t * g, * u, * d;
    CK(cudaMalloc(&g, hg.size())); CK(cudaMalloc(&u, hu.size())); CK(cudaMalloc(&d, hd.size()));
    CK(cudaMemcpy(g, hg.data(), hg.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(u, hu.data(), hu.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d, hd.data(), hd.size(), cudaMemcpyHostToDevice));
    // NE local experts at random ids, slots in id order
    std::vector<int> perm(E); for (int e = 0; e < E; ++e) perm[e] = e;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> slot(E, -1);
    std::sort(perm.begin(), perm.begin() + NE);
    for (int i = 0; i < NE; ++i) slot[perm[i]] = i;
    int * dslot; CK(cudaMalloc(&dslot, E * 4)); CK(cudaMemcpy(dslot, slot.data(), E * 4, cudaMemcpyHostToDevice));
    MoeDev m; m.gate = g; m.up = u; m.down = d; m.tg = GType::Q4_K; m.td = td; m.gate_bytes = gb; m.down_bytes = db;
    m.slot = dslot; m.ff = ff; m.n = n; m.clamp = 0.0f;
    // routing: 10 distinct experts per token, skewed (a few hot experts), weights 0.1
    std::vector<int> ids((size_t) nt * K); std::vector<float> wts((size_t) nt * K, 0.1f);
    std::vector<double> pw(E);
    for (int e = 0; e < E; ++e) pw[e] = 1.0 / (1.0 + (e * 37 % E) / 8.0);
    std::discrete_distribution<int> pick(pw.begin(), pw.end());
    for (int t = 0; t < nt; ++t)
        for (int j = 0; j < K;) {
            const int e = pick(rng);
            bool dup = false;
            for (int q = 0; q < j; ++q) dup |= ids[(size_t) t * K + q] == e;
            if (!dup) ids[(size_t) t * K + j++] = e;
        }
    int * dids; float * dwts; CK(cudaMalloc(&dids, ids.size() * 4)); CK(cudaMalloc(&dwts, wts.size() * 4));
    CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dwts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));
    std::vector<half> hx((size_t) nt * n);
    for (auto & v : hx) v = __float2half(std::uniform_real_distribution<float>(-1, 1)(rng));
    half * x16, * h16; float * y; int * order, * order_n, * egrp;
    CK(cudaMalloc(&x16, hx.size() * 2)); CK(cudaMemcpy(x16, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&h16, (size_t) nt * K * ff * 2)); CK(cudaMalloc(&y, (size_t) nt * K * n * 4));
    CK(cudaMalloc(&order, (size_t) nt * K * 4)); CK(cudaMalloc(&order_n, 2 * 4)); CK(cudaMalloc(&egrp, (size_t) 3 * E * 4));
    CK(cudaMemset(h16, 0, (size_t) nt * K * ff * 2)); CK(cudaMemset(y, 0, (size_t) nt * K * n * 4));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    moe_order(m, dids, nt * K, E, order, order_n, s, egrp);
    auto gu = [&] { moe_gemm_gate_up(m, x16, n, K, order, order_n, egrp, E, h16, s); };
    auto dn = [&] { moe_gemm_down(m, h16, order, order_n, egrp, E, dwts, y, s); };
    gu(); dn(); CK(cudaStreamSynchronize(s));
    int on[2]; CK(cudaMemcpy(on, order_n, 8, cudaMemcpyDeviceToHost));
    std::vector<int> ord(on[0]); CK(cudaMemcpy(ord.data(), order, on[0] * 4, cudaMemcpyDeviceToHost));
    std::vector<half> hh((size_t) nt * K * ff); std::vector<float> hy((size_t) nt * K * n);
    CK(cudaMemcpy(hh.data(), h16, hh.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long hh_h = 1469598103934665603ull, hy_h = hh_h;
    std::sort(ord.begin(), ord.end());   // (the order of the pairs within an expert's group is not deterministic; the values are)
    for (int p : ord) { hh_h = fnv(&hh[(size_t) p * ff], (size_t) ff * 2, hh_h); hy_h = fnv(&hy[(size_t) p * n], (size_t) n * 4, hy_h); }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto best = [&](auto && f) {
        float b = 1e30f;
        for (int i = 0; i < iters; ++i) {
            CK(cudaEventRecord(e0, s)); f(); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); b = std::min(b, ms);
        }
        return b;
    };
    const float tg = best(gu), td_ = best(dn);
    const double flop = 2.0 * on[0] * (2.0 * ff * n + (double) n * ff);
    printf("moe_gemm nt %d local %d (%d pairs, %d experts, down %s): gate/up %.3f ms  down %.3f ms  (%.1f TFLOPS)  hash h %016llx y %016llx\n",
           nt, NE, on[0], on[1], dt.c_str(), tg, td_, flop / ((tg + td_) * 1e-3) / 1e12, hh_h, hy_h);
    return 0;
}
