// Microbenchmark: allreduce throughput for prefill-sized messages (rows x 5120 elements).
// usage: arbulk [rows=512] [iters=20] [mode: 0 LL16, 1 bulk flags]
#include "../src/kernels.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 512;
    const int iters = argc > 2 ? atoi(argv[2]) : 20;
    const int mode = argc > 3 ? atoi(argv[3]) : 1;
    const int n = rows * 5120, calls = 16;
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    uint2 * ll; CK(cudaHostAlloc(&ll, (size_t) 2 * nd * n / 2 * 8, cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ll, 0xff, (size_t) 2 * nd * n / 2 * 8);
    half * data; CK(cudaHostAlloc(&data, (size_t) 2 * nd * n * 2, cudaHostAllocPortable | cudaHostAllocMapped));
    unsigned * flags; const size_t nf = (size_t) 2 * nd * (n / 1024 + 1);
    CK(cudaHostAlloc(&flags, nf * 4, cudaHostAllocPortable | cudaHostAllocMapped)); memset(flags, 0, nf * 4);
    std::vector<cudaStream_t> st(nd);
    std::vector<float *> x(nd), part(nd); std::vector<int *> ctr(nd);
    for (int g = 0; g < nd; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamCreateWithFlags(&st[g], cudaStreamNonBlocking));
        CK(cudaMalloc(&x[g], (size_t) n * 4)); CK(cudaMalloc(&part[g], (size_t) n * 4)); CK(cudaMalloc(&ctr[g], 4));
        CK(cudaMemset(x[g], 0, (size_t) n * 4)); CK(cudaMemset(ctr[g], 0, 4));
        std::vector<float> ones(n, 1.0f); CK(cudaMemcpy(part[g], ones.data(), (size_t) n * 4, cudaMemcpyHostToDevice));
    }
    // DMA variants: fp16 staging on each GPU, receive buffers for the peers' parts
    std::vector<half *> p16(nd), recv(nd);
    std::vector<cudaEvent_t> ev(nd * 2);
    half * hst; CK(cudaHostAlloc(&hst, (size_t) 2 * nd * n * 2, cudaHostAllocPortable));
    for (int g = 0; g < nd; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMalloc(&p16[g], (size_t) n * 2)); CK(cudaMalloc(&recv[g], (size_t) nd * n * 2));
        CK(cudaEventCreateWithFlags(&ev[2 * g], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&ev[2 * g + 1], cudaEventDisableTiming));
    }
    auto run_dma = [&](int t) {
        for (int i = 0; i < t; ++i) {
            for (int c = 0; c < calls; ++c) {
                const int b = c & 1;
                for (int g = 0; g < nd; ++g) {
                    CK(cudaSetDevice(g));
                    hyper::to_half(part[g], 5120, nullptr, 5120, 0.0f, p16[g], rows, st[g]);
                    if (mode == 2 || mode == 4) CK(cudaMemcpyAsync(hst + ((size_t) b * nd + g) * n, p16[g], (size_t) n * 2, cudaMemcpyDeviceToHost, st[g]));
                    CK(cudaEventRecord(ev[2 * g + b], st[g]));
                }
                for (int g = 0; g < nd; ++g) {
                    CK(cudaSetDevice(g));
                    for (int d = 0; d < nd; ++d) {
                        if (d == g) continue;
                        if (mode != 4) CK(cudaStreamWaitEvent(st[g], ev[2 * d + b], 0));
                        if (mode == 2 || mode == 4) CK(cudaMemcpyAsync(recv[g] + (size_t) d * n, hst + ((size_t) b * nd + d) * n, (size_t) n * 2, cudaMemcpyHostToDevice, st[g]));
                        else CK(cudaMemcpyPeerAsync(recv[g] + (size_t) d * n, g, p16[d], d, (size_t) n * 2, st[g]));
                    }
                }
                // after this point the next call's to_half on d may overwrite p16[d]; peers must have finished
                // reading: make every GPU wait for the others' copies
                if (mode == 3) for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaEventRecord(ev[2 * g + b], st[g])); }
                if (mode == 3) for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); for (int d = 0; d < nd; ++d) if (d != g) CK(cudaStreamWaitEvent(st[g], ev[2 * d + b], 0)); }
            }
            for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(st[g])); }
        }
    };
    auto run = [&](int t) {
        if (mode >= 2) { run_dma(t); return; }
        for (int i = 0; i < t; ++i) {
            for (int g = 0; g < nd; ++g) {
                CK(cudaSetDevice(g));
                hyper::incr_counter(ctr[g], st[g]);
                for (int c = 0; c < calls; ++c) {
                    if (mode == 0) hyper::allreduce_add_ll16(x[g], part[g], ll, g, nd, n, ctr[g], c, st[g]);
                    else hyper::allreduce_add_bulk(x[g], part[g], data, flags, g, nd, n, ctr[g], c, st[g]);
                }
            }
            for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(st[g])); }
        }
    };
    run(2);
    auto t0 = std::chrono::steady_clock::now();
    run(iters);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> h(16); CK(cudaSetDevice(0)); CK(cudaMemcpy(h.data(), x[0], 64, cudaMemcpyDeviceToHost));
    const double per = s / iters / calls;
    printf("ARBULK mode=%d rows=%d: %.3f ms per allreduce, %.2f GB/s fp16 payload per GPU written, x=%.0f (expect %d)\n", mode, rows,
           1e3 * per, (double) n * 2 / per / 1e9, h[0], nd * calls * (iters + 2));
    return 0;
}
