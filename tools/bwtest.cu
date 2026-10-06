// pinned host <-> device copy bandwidth, each GPU alone and all concurrently
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <string>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
int main() {
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    const size_t sz = 64 << 20; const int reps = 10;
    std::vector<void *> h(nd), d(nd); std::vector<cudaStream_t> s(nd);
    for (int g = 0; g < nd; ++g) {
        CK(cudaSetDevice(g)); CK(cudaMalloc(&d[g], sz)); CK(cudaHostAlloc(&h[g], sz, cudaHostAllocPortable));
        CK(cudaStreamCreate(&s[g]));
    }
    for (int dir = 0; dir < 3; ++dir) {
        const char * nm[3] = {"H2D", "D2H", "both"};
        auto go = [&](int g0, int g1) {
            for (int r = 0; r < reps + 1; ++r) {
                if (r == 1) for (int g = g0; g < g1; ++g) { cudaSetDevice(g); cudaStreamSynchronize(s[g]); }
                for (int g = g0; g < g1; ++g) {
                    cudaSetDevice(g);
                    if (dir != 1) cudaMemcpyAsync(d[g], h[g], dir == 2 ? sz / 2 : sz, cudaMemcpyHostToDevice, s[g]);
                    if (dir != 0) cudaMemcpyAsync(h[g], d[g], dir == 2 ? sz / 2 : sz, cudaMemcpyDeviceToHost, s[g]);
                }
            }
        };
        for (int g = 0; g <= nd; ++g) {
            const int g0 = g < nd ? g : 0, g1 = g < nd ? g + 1 : nd;
            go(g0, g1);
            for (int i = g0; i < g1; ++i) { cudaSetDevice(i); cudaStreamSynchronize(s[i]); }
            auto t0 = std::chrono::steady_clock::now();
            go(g0, g1);
            for (int i = g0; i < g1; ++i) { cudaSetDevice(i); cudaStreamSynchronize(s[i]); }
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("%s %s: %.1f GB/s per GPU\n", nm[dir], g < nd ? (std::string("gpu") + std::to_string(g)).c_str() : "all", (reps + 1) * sz / sec / 1e9);
        }
    }
}
