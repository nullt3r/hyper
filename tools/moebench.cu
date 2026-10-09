// MoE decode kernels on one GPU (Flash-Next shapes: n 2560, ff 640, top-10 of 512, gate/up Q4_K, down Q5_1 / Q8_0):
// time of moe_gate_up + moe_down per layer call with `local` of the 10 routed experts on this GPU, and the outputs of
// two kernel variants compared (HYPER_MOE_OLD selects the old ones). usage: moebench [iters] [local] [down: q51|q80|q50]
#include "../src/kernels4.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
using namespace hyper;

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200, local = argc > 2 ? atoi(argv[2]) : 3;
    const std::string dt = argc > 3 ? argv[3] : "q51";
    const GType td = dt == "q80" ? GType::Q8_0 : dt == "q50" ? GType::Q5_0 : GType::Q5_1;
    const int n = 2560, ff = 640, K = 10, E = 512, NE = 24;   // NE resident experts (working set beyond L2)
    const size_t gb = (size_t) ff * (n / 256) * 144, db = (size_t) n * (ff / 32) * gtype_block_bytes(td);
    std::mt19937 rng(3);
    auto fill = [&](std::vector<uint8_t> & v, GType t, size_t row_bytes) {
        for (auto & b : v) b = (uint8_t) rng();
        const size_t bb = gtype_block_bytes(t);
        for (size_t i = 0; i < v.size() / bb; ++i) {
            uint8_t * blk = v.data() + i * bb;
            const __half d = __float2half(0.002f * (1 + rng() % 8)), m = __float2half(-0.001f * (1 + rng() % 8));
            memcpy(blk, &d, 2);
            if (t == GType::Q4_K || t == GType::Q5_1) memcpy(blk + 2, &m, 2);
        }
        (void) row_bytes;
    };
    std::vector<uint8_t> hg(gb * NE), hu(gb * NE), hd(db * NE);
    fill(hg, GType::Q4_K, 0); fill(hu, GType::Q4_K, 0); fill(hd, td, 0);
    uint8_t * g, * u, * d;
    CK(cudaMalloc(&g, hg.size())); CK(cudaMalloc(&u, hu.size())); CK(cudaMalloc(&d, hd.size()));
    CK(cudaMemcpy(g, hg.data(), hg.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(u, hu.data(), hu.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d, hd.data(), hd.size(), cudaMemcpyHostToDevice));
    std::vector<int> slot(E, -1);
    for (int e = 0; e < NE; ++e) slot[e * 7] = e;   // resident experts 0, 7, 14, ...
    int * dslot; CK(cudaMalloc(&dslot, E * 4)); CK(cudaMemcpy(dslot, slot.data(), E * 4, cudaMemcpyHostToDevice));
    MoeDev m; m.gate = g; m.up = u; m.down = d; m.tg = GType::Q4_K; m.td = td; m.gate_bytes = gb; m.down_bytes = db;
    m.slot = dslot; m.ff = ff; m.n = n; m.clamp = 0.0f;
    // routing per call: `local` resident experts (rotating) + 10 - local non-resident
    const int NR = 64;
    std::vector<int> ids((size_t) NR * K);
    std::vector<float> wts((size_t) NR * K);
    for (int r = 0; r < NR; ++r)
        for (int j = 0; j < K; ++j) { ids[r * K + j] = j < local ? ((r * local + j) % NE) * 7 : 1 + j * 7 + r % 5; wts[r * K + j] = 0.1f; }
    int * dids; float * dwts; CK(cudaMalloc(&dids, ids.size() * 4)); CK(cudaMalloc(&dwts, wts.size() * 4));
    CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dwts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));
    std::vector<float> hx(n);
    for (auto & v : hx) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    float * x, * h, * y; CK(cudaMalloc(&x, n * 4)); CK(cudaMalloc(&h, K * ff * 4)); CK(cudaMalloc(&y, K * n * 4));
    CK(cudaMemcpy(x, hx.data(), n * 4, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    // reference output of the selected kernels for routing row 0 (compare runs with / without HYPER_MOE_OLD)
    moe_gate_up(m, x, n, dids, K, h, 1, s);
    moe_down(m, h, dids, dwts, K, y, 1, s);
    std::vector<float> hy(K * n);
    CK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
    double cs = 0; for (int j = 0; j < local; ++j) for (int i = 0; i < n; ++i) cs += hy[(size_t) j * n + i] * (1 + (i % 7));
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    const char * only = getenv("MOE_ONLY");   // gu | down
    for (int i = 0; i < iters; ++i) {
        if (!only || only[0] == 'g') moe_gate_up(m, x, n, dids + (i % NR) * K, K, h, 1, s);
        if (!only || only[0] == 'd') moe_down(m, h, dids + (i % NR) * K, dwts + (i % NR) * K, K, y, 1, s);
    }
    CK(cudaStreamEndCapture(s, &gr));
    CK(cudaGraphInstantiate(&ge, gr, 0));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    const double bytes = (double) local * (2 * gb + db);
    printf("MOE local %d down %s: %.2f us per layer (gate/up + down), %.0f GB/s  checksum %.6e\n", local, gtype_name(td),
           1000.0 * ms / iters, bytes * iters / (ms * 1e-3) / 1e9, cs);
    return 0;
}
