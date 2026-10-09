// GLM MLA decode attention on one GPU: time per call (1 token, H heads, n selected cells of the fp16 latent cache) and the
// error against a double-precision CPU reference. Variants by environment (HYPER5_MLA_SPLIT, HYPER5_MLA_TCDEC).
// usage: mlabench [H=21] [cells=2048] [iters=500]
#include "../src/kernels5.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
using namespace hyper;

int main(int argc, char ** argv) {
    const int H = argc > 1 ? atoi(argv[1]) : 21, n = argc > 2 ? atoi(argv[2]) : 2048, iters = argc > 3 ? atoi(argv[3]) : 500;
    const int cells = 16384, L = MLA_LAT;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0, 1);
    std::vector<half> lat((size_t) cells * L);
    std::vector<float> latf(lat.size());
    for (size_t i = 0; i < lat.size(); ++i) { lat[i] = __float2half(nd(rng)); latf[i] = __half2float(lat[i]); }
    std::vector<float> q((size_t) H * L);
    for (auto & v : q) v = nd(rng) * 0.6f;
    std::vector<int> list(n);
    for (int i = 0; i < n; ++i) list[i] = (int) (rng() % cells);
    const float scale = 1.0f / sqrtf(576.0f);
    half * dlat; float * dq, * dpart, * dout; int * dlist, * dn, * dpos;
    CK(cudaMalloc(&dlat, lat.size() * 2)); CK(cudaMemcpy(dlat, lat.data(), lat.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dlist, n * 4)); CK(cudaMemcpy(dlist, list.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dn, 4)); CK(cudaMemcpy(dn, &n, 4, cudaMemcpyHostToDevice));
    const int pos = cells - 1;
    CK(cudaMalloc(&dpos, 4)); CK(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dpart, mla_part_floats(H, 4) * 4)); CK(cudaMalloc(&dout, (size_t) H * L * 4));
    mla_init();
    cudaStream_t s; CK(cudaStreamCreate(&s));
    mla_attn(dq, H * L, dlat, dpos, H, scale, 1, dlist, n, dn, dpart, dout, H * L, s);
    CK(cudaStreamSynchronize(s));
    std::vector<float> o((size_t) H * L);
    CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
    // reference
    double maxerr = 0, maxref = 0;
    for (int h = 0; h < H; ++h) {
        std::vector<double> sc(n);
        double m = -1e300;
        for (int j = 0; j < n; ++j) {
            double a = 0;
            for (int d = 0; d < L; ++d) a += (double) q[(size_t) h * L + d] * scale * latf[(size_t) list[j] * L + d];
            sc[j] = a; m = std::max(m, a);
        }
        double z = 0;
        for (int j = 0; j < n; ++j) { sc[j] = exp(sc[j] - m); z += sc[j]; }
        for (int d = 0; d < L; ++d) {
            double a = 0;
            for (int j = 0; j < n; ++j) a += sc[j] * latf[(size_t) list[j] * L + d];
            a /= z;
            maxerr = std::max(maxerr, fabs(a - o[(size_t) h * L + d]));
            maxref = std::max(maxref, fabs(a));
        }
    }
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; ++i) mla_attn(dq, H * L, dlat, dpos, H, scale, 1, dlist, n, dn, dpart, dout, H * L, s);
    CK(cudaStreamEndCapture(s, &gr));
    CK(cudaGraphInstantiate(&ge, gr, 0));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    unsigned long long hsh = 1469598103934665603ull;
    for (float v : o) { unsigned u; memcpy(&u, &v, 4); hsh = (hsh ^ u) * 1099511628211ull; }
    printf("output bits hash %016llx\n", hsh);
    printf("MLA decode H=%d cells=%d split=%s tc=%s: %.2f us per call, max abs error %.3g (max |o| %.3g)\n", H, n,
           getenv("HYPER5_MLA_SPLIT") ? getenv("HYPER5_MLA_SPLIT") : "48", getenv("HYPER5_MLA_TCDEC") ? "yes" : "no",
           1000.0 * ms / iters, maxerr, maxref);
    return 0;
}
