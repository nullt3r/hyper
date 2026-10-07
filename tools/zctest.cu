// zero-copy bandwidth: GPU kernels reading pinned mapped host memory, alone and while CPU threads stream RAM
#include <cuda_runtime.h>
#include <omp.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

__global__ void k_read(const uint4 * __restrict__ p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) *out = acc;
}

int main(int argc, char ** argv) {
    const int ndev = 3;
    const size_t bytes = (size_t) 256 << 20;   // per GPU
    const int cpu_threads = argc > 1 ? atoi(argv[1]) : 30;
    std::vector<uint8_t *> host(ndev);
    std::vector<uint4 *> dptr(ndev);
    std::vector<unsigned *> dout(ndev);
    for (int g = 0; g < ndev; ++g) {
        cudaSetDevice(g);
        cudaHostAlloc(&host[g], bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        memset(host[g], 1, bytes);
        cudaHostGetDevicePointer((void **) &dptr[g], host[g], 0);
        cudaMalloc(&dout[g], 4);
    }
    const size_t cbytes = (size_t) 4 << 30;
    uint8_t * cbuf = (uint8_t *) aligned_alloc(4096, cbytes);
    memset(cbuf, 2, cbytes);
    auto gpu_run = [&](int reps) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int g = 0; g < ndev; ++g) { cudaSetDevice(g); k_read<<<328, 512>>>(dptr[g], bytes / 16, dout[g]); }
        for (int g = 0; g < ndev; ++g) { cudaSetDevice(g); cudaDeviceSynchronize(); }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return (double) reps * ndev * bytes / s / 1e9;
    };
    std::atomic<bool> stop{false};
    std::atomic<double> cpu_gbs{0};
    auto cpu_run = [&] {
        double tot = 0; auto t0 = std::chrono::steady_clock::now();
        while (!stop) {
            uint64_t sum = 0;
#pragma omp parallel for num_threads(cpu_threads) reduction(^ : sum) schedule(static)
            for (size_t i = 0; i < cbytes / 8; i += 1) sum ^= ((const uint64_t *) cbuf)[i];
            if (sum == 42) printf("x");
            tot += cbytes;
        }
        cpu_gbs = tot / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9;
    };
    gpu_run(1);
    printf("GPU zero-copy alone (3 GPUs): %.1f GB/s\n", gpu_run(8));
    for (int g = 0; g < ndev; ++g) {
        auto t0 = std::chrono::steady_clock::now();
        cudaSetDevice(g);
        for (int r = 0; r < 8; ++r) k_read<<<328, 512>>>(dptr[g], bytes / 16, dout[g]);
        cudaDeviceSynchronize();
        printf("  GPU %d alone: %.1f GB/s\n", g, 8.0 * bytes / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9);
    }
    { std::thread t(cpu_run); std::this_thread::sleep_for(std::chrono::seconds(3)); stop = true; t.join(); }
    printf("CPU alone (%d threads): %.1f GB/s\n", cpu_threads, cpu_gbs.load());
    stop = false;
    std::thread t(cpu_run);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const double g = gpu_run(24);
    stop = true; t.join();
    printf("together: GPU %.1f GB/s + CPU %.1f GB/s\n", g, cpu_gbs.load());
    return 0;
}
