// Dense Q8 tensor-core GEMM (gemm_q8, prefill) on one GPU, random weights and activations: time per call and a hash of the output
// for a shape, to compare token-tile choices bit for bit (HYPER_GEMM_BN / HYPER_GEMM_WIDE). usage: gemmbench n k T [iters=20]
#include "../src/kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
using namespace hyper;

int main(int argc, char ** argv) {
    if (argc < 4) { printf("usage: gemmbench n k T [iters]\n"); return 1; }
    const int n = atoi(argv[1]), k = atoi(argv[2]), T = atoi(argv[3]), iters = argc > 4 ? atoi(argv[4]) : 20;
    std::mt19937 rng(9);
    std::vector<int8_t> qs((size_t) n * k); std::vector<half> d((size_t) n * (k / 32)), x((size_t) T * k);
    for (auto & v : qs) v = (int8_t) (rng() % 255 - 127);
    for (auto & v : d) v = __float2half(0.001f * (1 + rng() % 16));
    for (auto & v : x) v = __float2half(std::uniform_real_distribution<float>(-1, 1)(rng));
    const size_t ntile = (n + 15) / 16, kb = k / 32;
    std::vector<uint8_t> fq(ntile * kb * 512); std::vector<half> fs(ntile * kb * 16);
    repack_q8_frag(qs.data(), d.data(), n, k, fq.data(), fs.data());
    Q8W W; W.n = n; W.k = k;
    uint4 * pq; half * ps, * px; float * py;
    CK(cudaMalloc(&pq, fq.size())); CK(cudaMalloc(&ps, fs.size() * 2)); CK(cudaMalloc(&px, x.size() * 2)); CK(cudaMalloc(&py, (size_t) T * n * 4));
    CK(cudaMemcpy(pq, fq.data(), fq.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(ps, fs.data(), fs.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(px, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    W.q = pq; W.s = ps;
    cudaStream_t s; CK(cudaStreamCreate(&s));
    gemm_q8(W, px, T, py, n, nullptr, s); CK(cudaStreamSynchronize(s));
    std::vector<float> y((size_t) T * n); CK(cudaMemcpy(y.data(), py, y.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < y.size() * 4; ++i) { h ^= ((const unsigned char *) y.data())[i]; h *= 1099511628211ull; }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(e0, s)); gemm_q8(W, px, T, py, n, nullptr, s); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
    }
    printf("gemm_q8 n %d k %d T %d: %.3f ms (%.1f TFLOPS)  hash %016llx\n", n, k, T, best, 2.0 * n * k * T / (best * 1e-3) / 1e12, h);
    return 0;
}
