// Gated DeltaNet prefill step (gdn_step, nt > 8: k_gdn_step_w) on one GPU, random inputs: time per call and hashes of the
// output and the final state, to check a kernel change bit for bit (HYPER_GDN_NOPRE=1: the per-warp scales).
// usage: gdnbench [n_k=5] [n_v=16] [nt=2048] [iters=20]   (Flash-Next GPU share: 5-6 k heads, 16 v heads; dk = dv = 128)
#include "../src/kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
using namespace hyper;

static unsigned long long fnv(const void * p, size_t n) {
    unsigned long long h = 1469598103934665603ull;
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    const int n_k = argc > 1 ? atoi(argv[1]) : 5, n_v = argc > 2 ? atoi(argv[2]) : 16, nt = argc > 3 ? atoi(argv[3]) : 2048;
    const int iters = argc > 4 ? atoi(argv[4]) : 20, dk = 128, dv = 128;
    const int ab_off = 2 * n_k * dk + n_v * dv, stride = (ab_off + 2 * n_v + 3) / 4 * 4;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> in((size_t) nt * stride), st((size_t) n_v * dk * dv), dtb(n_v), sa(n_v);
    for (auto & x : in) x = nd(rng);
    for (auto & x : st) x = 0.05f * nd(rng);
    for (int h = 0; h < n_v; ++h) { dtb[h] = nd(rng); sa[h] = -std::exp(0.5f * nd(rng)); }
    float * d_in, * d_st, * d_st0, * d_o, * d_dtb, * d_sa;
    CK(cudaMalloc(&d_in, in.size() * 4)); CK(cudaMalloc(&d_st, st.size() * 4)); CK(cudaMalloc(&d_st0, st.size() * 4));
    CK(cudaMalloc(&d_o, (size_t) nt * n_v * dv * 4)); CK(cudaMalloc(&d_dtb, n_v * 4)); CK(cudaMalloc(&d_sa, n_v * 4));
    CK(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_st0, st.data(), st.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dtb, dtb.data(), n_v * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_sa, sa.data(), n_v * 4, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    auto run = [&] {
        CK(cudaMemcpyAsync(d_st, d_st0, st.size() * 4, cudaMemcpyDeviceToDevice, s));
        gdn_step(d_in, stride, ab_off, d_st, nullptr, d_o, n_v * dv, d_dtb, d_sa, n_k, n_v, dk, dv, 1e-6f, nt, s);
    };
    run(); CK(cudaStreamSynchronize(s));
    std::vector<float> o((size_t) nt * n_v * dv), so(st.size());
    CK(cudaMemcpy(o.data(), d_o, o.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(so.data(), d_st, so.size() * 4, cudaMemcpyDeviceToHost));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        CK(cudaMemcpyAsync(d_st, d_st0, st.size() * 4, cudaMemcpyDeviceToDevice, s));
        CK(cudaEventRecord(e0, s));
        gdn_step(d_in, stride, ab_off, d_st, nullptr, d_o, n_v * dv, d_dtb, d_sa, n_k, n_v, dk, dv, 1e-6f, nt, s);
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
    }
    printf("gdn_step n_k %d n_v %d nt %d: %.3f ms  hash out %016llx state %016llx\n", n_k, n_v, nt, best, fnv(o.data(), o.size() * 4),
           fnv(so.data(), so.size() * 4));
    return 0;
}
