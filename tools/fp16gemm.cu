// FP16 x FP16 -> FP32-accumulate GEMM throughput per GPU (cuBLAS, n x n x n, the tensor-core rate an unlock changes)
// usage: fp16gemm [n=8192] [iters=50]
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 8192, iters = argc > 2 ? atoi(argv[2]) : 50;
    int nd = 0;
    cudaGetDeviceCount(&nd);
    for (int d = 0; d < nd; ++d) {
        cudaSetDevice(d);
        cudaDeviceProp p; cudaGetDeviceProperties(&p, d);
        half * a, * b; float * c;
        cudaMalloc(&a, (size_t) n * n * 2); cudaMalloc(&b, (size_t) n * n * 2); cudaMalloc(&c, (size_t) n * n * 4);
        cudaMemset(a, 0x3c, (size_t) n * n * 2); cudaMemset(b, 0x3c, (size_t) n * n * 2);   // (values ~1: no denormal slow paths)
        cublasHandle_t h; cublasCreate(&h);
        const float one = 1.0f, zero = 0.0f;
        auto run = [&] { return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, a, CUDA_R_16F, n, b, CUDA_R_16F, n, &zero, c,
                                             CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
        for (int i = 0; i < 5; ++i) run();
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) run();
        cudaDeviceSynchronize();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
        printf("FP16GEMM dev %d %s %s: %d^3 %.2f ms  %.1f TFLOPS (%s)\n", d, p.name, p.pciBusID == 0x4f ? "[4f]" : "", n, s * 1e3,
               2.0 * n * n * n / s / 1e12, cudaGetErrorString(cudaGetLastError()));
        cublasDestroy(h); cudaFree(a); cudaFree(b); cudaFree(c);
    }
    return 0;
}
