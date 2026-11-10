// Microbenchmark: latency of the LL allreduce across all GPUs, 128 calls per "token" in a CUDA graph.
// usage: arbench [n_elems=2560] [tokens=200] [mode: 0 host-mapped packets, 1 peer memory (P2P)]
#include "../src/kernels.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

typedef unsigned long long u64;
struct RB { u64 * in, * ag, * nin, * nag; int g, nd, n2, ch; };
__device__ __forceinline__ void rput(u64 * p, float2 v, unsigned seq) {
    const __half2 h = __floats2half2_rn(v.x, v.y);
    *(volatile u64 *) p = ((u64) seq << 32) | (u64) *(const unsigned *) &h;
}
__device__ __forceinline__ float2 rtake(const u64 * p, unsigned seq) {
    u64 w;
    do { w = *(const volatile u64 *) p; } while ((unsigned) (w >> 32) != seq);
    const unsigned b = (unsigned) w;
    return __half22float2(*(const __half2 *) &b);
}
// the LL sum as a ring (sfnx): reduce-scatter then all-gather, every device writing only into the next one's memory
__global__ void k_llring(RB r, float * x, const float * part, const int * counter, int call) {
    const unsigned seq = (unsigned) (*counter) * 1024u + (unsigned) call + 1u;
    const int nth = gridDim.x * blockDim.x, tid = blockIdx.x * blockDim.x + threadIdx.x, set = call & 1, nd = r.nd, g = r.g;
    u64 * in = r.in + (size_t) set * (nd - 1) * r.ch, * nin = r.nin + (size_t) set * (nd - 1) * r.ch;
    u64 * ag = r.ag + (size_t) set * r.n2, * nag = r.nag + (size_t) set * r.n2;
    const float2 * p2 = (const float2 *) part;
    auto lo = [&](int k) { return (int) ((long long) k * r.n2 / nd); };
    { const int l0 = lo(g), len = lo(g + 1) - l0; for (int e = tid; e < len; e += nth) rput(nin + e, p2[l0 + e], seq); }
    for (int s = 1; s < nd - 1; ++s) {
        const int c = (g - s + nd) % nd, l0 = lo(c), len = lo(c + 1) - l0;
        for (int e = tid; e < len; e += nth) { float2 a = rtake(in + (size_t) (s - 1) * r.ch + e, seq); const float2 m = p2[l0 + e]; a.x += m.x; a.y += m.y; rput(nin + (size_t) s * r.ch + e, a, seq); }
    }
    { const int c = (g + 1) % nd, l0 = lo(c), len = lo(c + 1) - l0;
      for (int e = tid; e < len; e += nth) { float2 a = rtake(in + (size_t) (nd - 2) * r.ch + e, seq); const float2 m = p2[l0 + e]; a.x += m.x; a.y += m.y; rput(ag + l0 + e, a, seq); rput(nag + l0 + e, a, seq); } }
    for (int f = 1; f < nd - 1; ++f) {
        const int c = (g - f + 1 + nd) % nd, l0 = lo(c), len = lo(c + 1) - l0;
        for (int e = tid; e < len; e += nth) rput(nag + l0 + e, rtake(ag + l0 + e, seq), seq);
    }
    float2 * x2 = (float2 *) x;
    for (int e = tid; e < r.n2; e += nth) { const float2 v = rtake(ag + e, seq); x2[e].x += v.x; x2[e].y += v.y; }
}

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
    const int n2 = n / 2, ch = n2 / nd + 1;
    std::vector<u64 *> rin(nd, nullptr), rag(nd, nullptr);
    if (mode == 4) for (int g = 0; g < nd; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMalloc(&rin[g], (size_t) 2 * (nd - 1) * ch * 8)); CK(cudaMemset(rin[g], 0xff, (size_t) 2 * (nd - 1) * ch * 8));
        CK(cudaMalloc(&rag[g], (size_t) 2 * n2 * 8)); CK(cudaMemset(rag[g], 0xff, (size_t) 2 * n2 * 8));
    }
    std::vector<cudaStream_t> st(nd); std::vector<cudaGraphExec_t> ex(nd);
    std::vector<float *> x(nd), part(nd); std::vector<int *> ctr(nd);
    for (int g = 0; g < nd; ++g) {
        CK(cudaSetDevice(g));
        hyper::LLBuf lb;
        if (mode == 0) { lb.rd = host; lb.nw = 1; lb.wr[0] = host; }
        else if (mode == 3) { lb.rd = dbuf[g]; lb.nw = 1; lb.wr[0] = dbuf[g]; for (int p = 0; p < nd; ++p) lb.rds[p] = dbuf[p]; }   // pull
        else { lb.rd = dbuf[g]; lb.nw = 0; for (int p = 0; p < nd; ++p) if (p != g) lb.wr[lb.nw++] = dbuf[p]; lb.wide = mode == 2; }
        CK(cudaStreamCreateWithFlags(&st[g], cudaStreamNonBlocking));
        CK(cudaMalloc(&x[g], n * 4)); CK(cudaMalloc(&part[g], n * 4)); CK(cudaMalloc(&ctr[g], 4));
        CK(cudaMemset(x[g], 0, n * 4)); CK(cudaMemset(ctr[g], 0, 4));
        std::vector<float> ones(n, 1.0f); CK(cudaMemcpy(part[g], ones.data(), n * 4, cudaMemcpyHostToDevice));
        cudaGraph_t gr;
        CK(cudaStreamBeginCapture(st[g], cudaStreamCaptureModeThreadLocal));
        hyper::incr_counter(ctr[g], st[g]);
        if (mode == 4) {
            const int nx = (g + 1) % nd;
            RB r{rin[g], rag[g], rin[nx], rag[nx], g, nd, n2, ch};
            for (int c = 0; c < calls; ++c) k_llring<<<(n2 + 255) / 256, 256, 0, st[g]>>>(r, x[g], part[g], ctr[g], c);
        } else
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
           mode == 4 ? "P2P-ring" : mode == 3 ? "P2P-pull" : mode ? "P2P-push" : "host", mode == 2 ? "-wide" : "", n, nd, 1e6 * s / tokens / calls, 1e3 * s / tokens, h[0], nd * calls * (tokens + 10));
    return 0;
}
