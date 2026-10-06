// Microbenchmark: latency of the P2P-free allreduce across all GPUs, 128 calls per "token" in a CUDA graph.
// usage: arbench [n_elems=5120] [tokens=200]
#include "../src/kernels.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 5120;
    const int tokens = argc > 2 ? atoi(argv[2]) : 200;
    const int calls = 128;
    const int mode = argc > 3 ? atoi(argv[3]) : 0; const bool ll = mode >= 1;
    uint2 * llslots; CK(cudaHostAlloc(&llslots, (size_t) 2 * 3 * n * 8, cudaHostAllocPortable | cudaHostAllocMapped)); memset(llslots, 0xff, (size_t) 2 * 3 * n * 8);
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    const int nchunk = 1;
    float * slots; unsigned long long * flags;
    CK(cudaHostAlloc(&slots, (size_t) 2 * nd * n * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped));
    CK(cudaHostAlloc(&flags, (size_t) nd * nchunk * 8, cudaHostAllocPortable | cudaHostAllocMapped));
    memset(flags, 0, (size_t) nd * nchunk * 8);
    std::vector<cudaStream_t> st(nd); std::vector<cudaGraphExec_t> ex(nd);
    std::vector<float *> x(nd), part(nd); std::vector<int *> ctr(nd);
    for (int g = 0; g < nd; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamCreateWithFlags(&st[g], cudaStreamNonBlocking));
        CK(cudaMalloc(&x[g], n * 4)); CK(cudaMalloc(&part[g], n * 4)); CK(cudaMalloc(&ctr[g], 4));
        CK(cudaMemset(x[g], 0, n * 4)); CK(cudaMemset(ctr[g], 0, 4));
        std::vector<float> ones(n, 1.0f); CK(cudaMemcpy(part[g], ones.data(), n * 4, cudaMemcpyHostToDevice));
        cudaGraph_t gr;
        CK(cudaStreamBeginCapture(st[g], cudaStreamCaptureModeThreadLocal));
        hyper::incr_counter(ctr[g], st[g]);
        for (int c = 0; c < calls; ++c) hyper::allreduce_add_ll16(x[g], part[g], llslots, g, nd, n, ctr[g], c, st[g]);
        CK(cudaStreamEndCapture(st[g], &gr));
        CK(cudaGraphInstantiate(&ex[g], gr, 0));
    }
    auto run = [&](int t) {
        for (int i = 0; i < t; ++i) {
            for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaGraphLaunch(ex[g], st[g])); }
            for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(st[g])); }
        }
    };
    run(10);
    auto t0 = std::chrono::steady_clock::now();
    run(tokens);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> h(n); CK(cudaSetDevice(0)); CK(cudaMemcpy(h.data(), x[0], n * 4, cudaMemcpyDeviceToHost));
    printf("ARBENCH %s n=%d devices=%d: %.2f us per allreduce (%.2f ms per 128-call token), check x[0]=%.0f (expect %d)\n",
           mode == 2 ? "LL16" : ll ? "LL" : "flag", n, nd, 1e6 * s / tokens / calls, 1e3 * s / tokens, h[0], nd * calls * (tokens + 10));
    return 0;
}
