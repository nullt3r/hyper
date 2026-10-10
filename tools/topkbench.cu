// sampling candidates (top-64 of a vocabulary slice): time per call and a hash of the sorted (value, index) set, to compare
// the two-stage selection with the one-block kernel (HYPER_TOPK_ONEBLOCK=1). usage: topkbench [n=82773] [iters=500]
#include "../src/kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
using namespace hyper;

int main(int argc, char ** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 82773, iters = argc > 2 ? atoi(argv[2]) : 500, nt = 4;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 3);
    std::vector<float> x((size_t) nt * n);
    for (int t = 0; t < nt; ++t)
        for (int i = 0; i < n; ++i) {
            float v = nd(rng);
            if (t == 1) v = std::round(v * 2.0f) / 2.0f;   // many ties
            x[(size_t) t * n + i] = v;
        }
    float * dx, * dout; CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dout, nt * TOPK * 2 * 4));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    topk_pairs(dx, n, n, 1000, dout, TOPK, nt, s);
    CK(cudaStreamSynchronize(s));
    std::vector<float> o(nt * TOPK * 2);
    CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long h = 1469598103934665603ull, hv = h;
    for (int t = 0; t < nt; ++t) {
        std::vector<std::pair<float, int>> p;
        for (int j = 0; j < TOPK; ++j) { int id; memcpy(&id, &o[(t * TOPK + j) * 2 + 1], 4); p.push_back({o[(t * TOPK + j) * 2], id}); }
        std::sort(p.begin(), p.end(), [](auto & a, auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        for (auto & q : p) {
            unsigned u; memcpy(&u, &q.first, 4);
            hv = (hv ^ u) * 1099511628211ull;   // values only
            if (t != 1) h = (h ^ u ^ ((unsigned long long) q.second << 32)) * 1099511628211ull;   // values + indices (no ties)
        }
    }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < iters; ++i) topk_pairs(dx, n, n, 1000, dout, TOPK, 1, s);
    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("TOPK n=%d %s: %.1f us per row; value hash %016llx, value+index hash (rows without ties) %016llx\n", n,
           getenv("HYPER_TOPK_ONEBLOCK") ? "one block" : "two-stage", 1000.0 * ms / iters, hv, h);
    return 0;
}
