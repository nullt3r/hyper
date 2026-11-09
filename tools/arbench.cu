// Microbenchmark: latency of the LL allreduce across all GPUs, 128 calls per "token" in a CUDA graph.
// usage: arbench [n_elems=2560] [tokens=200] [mode: 0 host-mapped packets, 1 peer memory (P2P)]
#include "../src/kernels.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 2560;
    const int tokens = argc > 2 ? atoi(argv[2]) : 200;
    const int mode = argc > 3 ? atoi(argv[3]) : 0;
    const int calls = 128;
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    const size_t ll = (size_t) 2 * nd * n * 2;   // packets: [parity][dev][n/2] (x2: room for the fp32 variant)
    uint2 * host = nullptr;
    std::vector<uint2 *> dbuf(nd, nullptr);
    if (mode == 0) { CK(cudaHostAlloc(&host, ll * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped)); memset(host, 0xff, ll * sizeof(uint2)); }
    else {
        for (int g = 0; g < nd; ++g) for (int p = 0; p < nd; ++p) {
            if (p == g) continue;
            int ok = 0; CK(cudaDeviceCanAccessPeer(&ok, g, p));
            if (!ok) { printf("no peer access %d -> %d\n", g, p); return 1; }
            CK(cudaSetDevice(g));
            cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
        }
        for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaMalloc(&dbuf[g], ll * sizeof(uint2))); CK(cudaMemset(dbuf[g], 0xff, ll * sizeof(uint2))); }
    }
    std::vector<cudaStream_t> st(nd); std::vector<cudaGraphExec_t> ex(nd);
    std::vector<float *> x(nd), part(nd); std::vector<int *> ctr(nd);
    for (int g = 0; g < nd; ++g) {
        CK(cudaSetDevice(g));
        hyper::LLBuf lb;
        if (mode == 0) { lb.rd = host; lb.nw = 1; lb.wr[0] = host; }
        else { lb.rd = dbuf[g]; lb.nw = 0; for (int p = 0; p < nd; ++p) if (p != g) lb.wr[lb.nw++] = dbuf[p]; lb.wide = mode == 2; }
        CK(cudaStreamCreateWithFlags(&st[g], cudaStreamNonBlocking));
        CK(cudaMalloc(&x[g], n * 4)); CK(cudaMalloc(&part[g], n * 4)); CK(cudaMalloc(&ctr[g], 4));
        CK(cudaMemset(x[g], 0, n * 4)); CK(cudaMemset(ctr[g], 0, 4));
        std::vector<float> ones(n, 1.0f); CK(cudaMemcpy(part[g], ones.data(), n * 4, cudaMemcpyHostToDevice));
        cudaGraph_t gr;
        CK(cudaStreamBeginCapture(st[g], cudaStreamCaptureModeThreadLocal));
        hyper::incr_counter(ctr[g], st[g]);
        for (int c = 0; c < calls; ++c) hyper::allreduce_add_ll16(x[g], part[g], lb, g, nd, n, ctr[g], c, st[g]);
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
    printf("ARBENCH %s%s n=%d devices=%d: %.2f us per allreduce (%.2f ms per 128-call token), check x[0]=%.0f (expect %d)\n",
           mode ? "P2P" : "host", mode == 2 ? "-wide" : "", n, nd, 1e6 * s / tokens / calls, 1e3 * s / tokens, h[0], nd * calls * (tokens + 10));
    return 0;
}
