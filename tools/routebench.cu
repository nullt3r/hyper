// MoE routing kernel (softmax + top-k): warp-per-token vs the block version (HYPER4_BLOCKROUTE), same random logits;
// prints the time per call and a digest of ids / weights to compare runs. usage: routebench [n_expert=512] [k=10] [nt=1]
#include "../src/kernels4.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int ne = argc > 1 ? atoi(argv[1]) : 512, k = argc > 2 ? atoi(argv[2]) : 10, nt = argc > 3 ? atoi(argv[3]) : 1;
    const int R = 256, iters = 400, ls = ne + 1;
    std::mt19937 rng(5);
    std::vector<float> hl((size_t) R * nt * ls);
    for (auto & v : hl) v = std::normal_distribution<float>(0.0f, 2.0f)(rng);
    for (int r = 0; r < 8; ++r) hl[(size_t) r * ls + 3] = hl[(size_t) r * ls + 7];   // a few exact ties
    float * l, * w, * sg; int * ids;
    CK(cudaMalloc(&l, hl.size() * 4)); CK(cudaMalloc(&w, (size_t) R * nt * k * 4)); CK(cudaMalloc(&ids, (size_t) R * nt * k * 4)); CK(cudaMalloc(&sg, R * nt * 4));
    CK(cudaMemcpy(l, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    for (int r = 0; r < R; ++r) hyper::moe_route(l + (size_t) r * nt * ls, ls, ne, k, ids + r * nt * k, w + r * nt * k, sg + r * nt, nt, s);
    std::vector<int> hi((size_t) R * nt * k); std::vector<float> hw(hi.size());
    CK(cudaMemcpy(hi.data(), ids, hi.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hw.data(), w, hw.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long hsh = 1469598103934665603ull; double ws = 0;
    for (size_t i = 0; i < hi.size(); ++i) { hsh = (hsh ^ (unsigned) hi[i]) * 1099511628211ull; ws += hw[i] * (1 + i % 13); }
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; ++i) hyper::moe_route(l + (size_t) (i % R) * nt * ls, ls, ne, k, ids, w, sg, nt, s);
    CK(cudaStreamEndCapture(s, &gr)); CK(cudaGraphInstantiate(&ge, gr, 0));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("ROUTE ne %d k %d nt %d: %.2f us per call  ids hash %016llx  weights digest %.9f\n", ne, k, nt, 1000.0 * ms / iters, hsh, ws);
    return 0;
}
