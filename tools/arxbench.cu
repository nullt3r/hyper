// Microbenchmark: one prefill allreduce exchange alone (rows x n values a device, fp16 in transit), three transports:
// 0 host staging (upload, the peers download), 1 all-to-all peer copies, 2 the copy-engine ring with device flags.
// usage: arxbench [rows=4096] [n=2560] [iters=20]
#include "../src/kernels.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
typedef unsigned long long u64;

int main(int argc, char ** argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 4096, n = argc > 2 ? atoi(argv[2]) : 2560, iters = argc > 3 ? atoi(argv[3]) : 20;
    const size_t N = (size_t) rows * n;
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    const size_t ch = (N + nd - 1) / nd + 8;
    struct Dev { cudaStream_t s, cs; float * part, * x; half * p16, * recv, * rr, * sb[2], * ag; u64 * flags, * tagv; int * ctr; unsigned * err; cudaEvent_t ev_a, ev_up, ev_dn; };
    std::vector<Dev> D(nd);
    half * stage; CK(cudaHostAlloc(&stage, (size_t) nd * N * sizeof(half), cudaHostAllocPortable));
    for (int g = 0; g < nd; ++g) {
        Dev & d = D[g];
        CK(cudaSetDevice(g));
        for (int p = 0; p < nd; ++p) if (p != g) { cudaError_t e = cudaDeviceEnablePeerAccess(p, 0); if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e); }
        CK(cudaStreamCreateWithFlags(&d.s, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&d.cs, cudaStreamNonBlocking));
        CK(cudaMalloc(&d.part, N * 4)); CK(cudaMalloc(&d.x, N * 4)); CK(cudaMalloc(&d.p16, N * 2)); CK(cudaMalloc(&d.recv, (size_t) (nd - 1) * N * 2));
        CK(cudaMalloc(&d.rr, (size_t) (nd - 1) * ch * 2)); CK(cudaMalloc(&d.sb[0], ch * 2)); CK(cudaMalloc(&d.sb[1], ch * 2)); CK(cudaMalloc(&d.ag, N * 2));
        CK(cudaMalloc(&d.flags, 16 * 8)); CK(cudaMemset(d.flags, 0, 16 * 8)); CK(cudaMalloc(&d.tagv, 8)); CK(cudaMalloc(&d.ctr, 4)); CK(cudaMemset(d.ctr, 0, 4));
        CK(cudaMalloc(&d.err, 4)); CK(cudaMemset(d.err, 0, 4));
        std::vector<float> v(N, (float) (g + 1)); CK(cudaMemcpy(d.part, v.data(), N * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(d.x, 0, N * 4));
        CK(cudaEventCreateWithFlags(&d.ev_a, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&d.ev_up, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&d.ev_dn, cudaEventDisableTiming));
    }
    auto lo = [&](int k) { return (size_t) k * N / nd; };
    auto len = [&](int k) { return (int) (lo(k + 1) - lo(k)); };
    int call = 0;
    auto exchange = [&](int mode) {
        ++call;
        for (int g = 0; g < nd; ++g) {   // the partial as fp16, handed to the copy stream
            Dev & d = D[g];
            CK(cudaSetDevice(g));
            hyper::to_half(d.part, n, nullptr, n, 0.0f, d.p16, rows, d.s);
            CK(cudaEventRecord(d.ev_a, d.s)); CK(cudaStreamWaitEvent(d.cs, d.ev_a, 0));
            if (mode == 0) CK(cudaMemcpyAsync(stage + (size_t) g * N, d.p16, N * 2, cudaMemcpyDeviceToHost, d.cs));
            else if (mode == 1) for (int p = 0; p < nd; ++p) { if (p == g) continue; const int j = g < p ? g : g - 1; CK(cudaMemcpyPeerAsync(D[p].recv + (size_t) j * N, p, d.p16, g, N * 2, d.cs)); }
            else {
                const int nx = (g + 1) % nd; Dev & X = D[nx];
                hyper::ring_tag(d.tagv, d.ctr, call, d.cs);
                auto send = [&](const half * src, half * dst, int cnt, int step) {
                    CK(cudaMemcpyPeerAsync(dst, nx, src, g, (size_t) cnt * 2, d.cs));
                    CK(cudaMemcpyPeerAsync(X.flags + step, nx, d.tagv, g, 8, d.cs));
                };
                send(d.p16 + lo(g), X.rr, len(g), 0);
                for (int h = 1; h < nd - 1; ++h) {
                    const int c = (g - h + nd) % nd;
                    hyper::ring_wait(d.flags + h - 1, d.ctr, call, d.err, d.cs);
                    hyper::ring_add(d.rr + (size_t) (h - 1) * ch, d.p16 + lo(c), d.sb[h & 1], len(c), d.cs);
                    send(d.sb[h & 1], X.rr + (size_t) h * ch, len(c), h);
                }
                { const int c = (g + 1) % nd;
                  hyper::ring_wait(d.flags + nd - 2, d.ctr, call, d.err, d.cs);
                  hyper::ring_add(d.rr + (size_t) (nd - 2) * ch, d.p16 + lo(c), d.ag + lo(c), len(c), d.cs);
                  send(d.ag + lo(c), X.ag + lo(c), len(c), nd - 1); }
                for (int f = 1; f < nd - 1; ++f) {
                    const int c = (g - f + 1 + nd) % nd;
                    hyper::ring_wait(d.flags + nd - 2 + f, d.ctr, call, d.err, d.cs);
                    send(d.ag + lo(c), X.ag + lo(c), len(c), nd - 1 + f);
                }
            }
            CK(cudaEventRecord(d.ev_up, d.cs));
        }
        for (int g = 0; g < nd; ++g) {   // the sum
            Dev & d = D[g];
            CK(cudaSetDevice(g));
            if (mode == 0) {
                int j = 0;
                for (int p = 0; p < nd; ++p) { if (p == g) continue; CK(cudaStreamWaitEvent(d.cs, D[p].ev_up, 0)); CK(cudaMemcpyAsync(d.recv + (size_t) j++ * N, stage + (size_t) p * N, N * 2, cudaMemcpyHostToDevice, d.cs)); }
                CK(cudaEventRecord(d.ev_dn, d.cs)); CK(cudaStreamWaitEvent(d.s, d.ev_dn, 0));
                hyper::add_parts(d.x, d.p16, d.recv, N, nd - 1, (int) N, d.s);
            } else if (mode == 1) {
                for (int p = 0; p < nd; ++p) if (p != g) CK(cudaStreamWaitEvent(d.s, D[p].ev_up, 0));
                hyper::add_parts(d.x, d.p16, d.recv, N, nd - 1, (int) N, d.s);
            } else {
                hyper::ring_wait(d.flags + 2 * nd - 3, d.ctr, call, d.err, d.s);
                hyper::ring_out(d.ag, d.x, (int) N, d.s);
            }
        }
        for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(D[g].s)); CK(cudaStreamSynchronize(D[g].cs)); }
    };
    const char * names[3] = {"host staging", "all-to-all peer copies", "ring (copy engines + flags)"};
    for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < 3; ++i) exchange(mode);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) exchange(mode);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
        printf("ARXBENCH rows=%d n=%d (%.1f MB fp16 a device) %-28s %.2f ms per exchange\n", rows, n, N * 2 / 1048576.0, names[mode], ms);
    }
    return 0;
}
