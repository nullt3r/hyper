// Q8 GEMV bandwidth microbenchmark on one GPU, real per-GPU shapes of the 27B model under 3-way TP.
// usage: gemvbench [iters]
#include "../src/kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    struct Shape { const char * name; int n, k; };
    const Shape shapes[] = {
        {"ffn_gate/up (5792x5120)", 5792, 5120},
        {"ffn_down (5120x5792)", 5120, 5792},
        {"gdn qkv (3328x5120)", 3328, 5120},
        {"attn_out (5120x2048)", 5120, 2048},
    };
    CK(cudaSetDevice(0));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int R : {1, 2, 4}) {
    hyper::g_gemv_rows_per_warp = R;
    printf("-- rows per warp %d\n", R);
    for (auto & sh : shapes) {
        // several distinct matrices so the working set exceeds L2 (6 MB on 3090)
        const int nmat = 8;
        std::vector<hyper::Q8W> W(nmat);
        for (int m = 0; m < nmat; ++m) {
            int8_t * qs; half * d;
            CK(cudaMalloc(&qs, (size_t) sh.n * sh.k));
            CK(cudaMalloc(&d, (size_t) sh.n * sh.k / 32 * 2));
            CK(cudaMemset(qs, 1, (size_t) sh.n * sh.k));
            CK(cudaMemset(d, 0, (size_t) sh.n * sh.k / 32 * 2));
            W[m].qs = qs; W[m].d = d; W[m].n = sh.n; W[m].k = sh.k;
        }
        float * x, * y; CK(cudaMalloc(&x, sh.k * 4)); CK(cudaMalloc(&y, sh.n * 4));
        CK(cudaMemset(x, 0, sh.k * 4));
        for (int i = 0; i < 20; ++i) hyper::gemv_q8(W[i % nmat], x, y, nullptr, s);
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < iters; ++i) hyper::gemv_q8(W[i % nmat], x, y, nullptr, s);
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        const double bytes = (double) sh.n * sh.k * (1.0 + 2.0 / 32);
        printf("%-26s %7.1f us  %6.0f GB/s\n", sh.name, 1000.0 * ms / iters, bytes * iters / (ms * 1e-3) / 1e9);
        for (auto & w : W) { cudaFree((void *) w.qs); cudaFree((void *) w.d); }
        cudaFree(x); cudaFree(y);
    }
    }
    return 0;
}
