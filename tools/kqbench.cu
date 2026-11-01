// K-quant dense GEMV (gemv_kq) on real GGUF rows: time per call, bandwidth, an output bit hash (HYPER_KQ_UNROLL variants
// must hash the same). usage: kqbench <model.gguf> <tensor> [rows=all] [nt=1] [iters=500]
#include "gguf.h"
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace hyper;

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: kqbench model.gguf tensor [rows] [nt] [iters]\n"); return 1; }
    GGUF g(argv[1]);
    const GTensor & t = g.need(argv[2]);
    if (t.type != GType::Q6_K && t.type != GType::Q4_K) { fprintf(stderr, "%s is not Q6_K / Q4_K\n", argv[2]); return 1; }
    const int k = (int) t.ne[0];
    int n = (int) t.rows();
    if (argc > 3 && atoi(argv[3]) > 0) n = std::min(n, atoi(argv[3]));
    const int nt = argc > 4 ? atoi(argv[4]) : 1, iters = argc > 5 ? atoi(argv[5]) : 500;
    std::vector<int> kbl(k / 32);
    for (int b = 0; b < k / 32; ++b) kbl[b] = b;
    KQHost h;
    const KQ kt = t.type == GType::Q4_K ? KQ::Q4K : KQ::Q6K;
    const int dg = repack_kq(kt, t.data, t.row_bytes(), n, kbl, h);
    gemv_init(0);
    auto up = [&](const auto & v) -> void * {
        if (v.empty()) return nullptr;
        void * p; cudaMalloc(&p, v.size() * sizeof(v[0]));
        cudaMemcpy(p, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice);
        return p;
    };
    KQW w; w.type = kt; w.n = n; w.k = k; w.dg = dg;
    w.lo = (const uint2 *) up(h.lo); w.hi = (const unsigned *) up(h.hi);
    w.scm = (const uint16_t *) up(h.scm); w.sc6 = (const int8_t *) up(h.sc6); w.d = (const half2 *) up(h.d);
    std::vector<float> x((size_t) nt * k);
    std::mt19937 rng(3);
    for (auto & v : x) v = std::normal_distribution<float>(0, 1)(rng);
    float * dx, * dy;
    cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dy, (size_t) nt * n * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int i = 0; i < 20; ++i) gemv_kq(w, dx, k, dy, n, nt, s);
    cudaStreamSynchronize(s);
    // weights larger than the L2 (6 MB on a 3090): a set of copies cycled would be truer; the matrices here are tens of MB
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) gemv_kq(w, dx, k, dy, n, nt, s);
    cudaStreamSynchronize(s);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::vector<float> y((size_t) nt * n);
    cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
    unsigned long long hb = 1469598103934665603ull;
    for (float v : y) { unsigned u; memcpy(&u, &v, 4); hb = (hb ^ u) * 1099511628211ull; }
    const double bytes = (double) n * t.row_bytes();
    printf("KQBENCH %s %s %d x %d nt=%d: %.1f us/call  %.0f GB/s  hash %016llx (%s)\n", argv[2], kt == KQ::Q6K ? "Q6_K" : "Q4_K", n, k, nt,
           us, bytes / us / 1e3, hb, cudaGetErrorString(cudaGetLastError()));
    return 0;
}
