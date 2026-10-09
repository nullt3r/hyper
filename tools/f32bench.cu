// fp32 router GEMV (warp per row): the current kernel vs an unrolled copy -- time and bitwise equality on random inputs.
// Shapes: Flash-Next router 513 x 2560, GLM router 288 x 4096. usage: f32bench [iters]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ float warp_sum4(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
// (copy of k_gemv_f32, kernels4.cu)
__global__ void k_old(const float * __restrict__ W, int rows, int k, const float * __restrict__ x, int xs, float * __restrict__ y, int ys) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), t = blockIdx.y, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float * wr = W + (size_t) r * k;
    const float * xr = x + (size_t) t * xs;
    float acc = 0.0f;
    for (int i = lane * 4; i < k; i += 128) {
        const float4 a = *(const float4 *) (wr + i), b = *(const float4 *) (xr + i);
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_sum4(acc);
    if (lane == 0) y[(size_t) t * ys + r] = acc;
}
template <int U>
__global__ void k_new(const float * __restrict__ W, int rows, int k, const float * __restrict__ x, int xs, float * __restrict__ y, int ys) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), t = blockIdx.y, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float * wr = W + (size_t) r * k;
    const float * xr = x + (size_t) t * xs;
    float acc = 0.0f;
    int i = lane * 4;
    for (; i + 128 * (U - 1) < k; i += 128 * U) {   // U iterations' loads in flight, accumulated in order
        float4 a[U], b[U];
#pragma unroll
        for (int u = 0; u < U; ++u) { a[u] = *(const float4 *) (wr + i + 128 * u); b[u] = *(const float4 *) (xr + i + 128 * u); }
#pragma unroll
        for (int u = 0; u < U; ++u) acc += a[u].x * b[u].x + a[u].y * b[u].y + a[u].z * b[u].z + a[u].w * b[u].w;
    }
    for (; i < k; i += 128) {
        const float4 a = *(const float4 *) (wr + i), b = *(const float4 *) (xr + i);
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_sum4(acc);
    if (lane == 0) y[(size_t) t * ys + r] = acc;
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 500;
    struct S { const char * name; int rows, k; } shapes[] = {{"flash router 513x2560", 513, 2560}, {"glm router 288x4096", 288, 4096}};
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0, 1);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    for (auto & sh : shapes) {
        const int NW = 48;   // distinct weight copies (> L2) cycled through, like layers
        std::vector<float> w((size_t) NW * sh.rows * sh.k), x((size_t) 4 * sh.k);
        for (auto & v : w) v = nd(rng) * 0.05f;
        float * dw, * dx, * dy1, * dy2;
        CK(cudaMalloc(&dw, w.size() * 4)); CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy1, 4 * sh.rows * 4)); CK(cudaMalloc(&dy2, 4 * sh.rows * 4));
        int bad = 0;
        std::vector<float> y1(4 * sh.rows), y2(4 * sh.rows);
        for (int trial = 0; trial < 200; ++trial) {
            for (auto & v : x) v = nd(rng) * (trial % 3 == 0 ? 10.0f : 1.0f);
            CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            const float * wt = dw + (size_t) (trial % NW) * sh.rows * sh.k;
            k_old<<<dim3((sh.rows + 7) / 8, 4), 256, 0, s>>>(wt, sh.rows, sh.k, dx, sh.k, dy1, sh.rows);
            k_new<5><<<dim3((sh.rows + 7) / 8, 4), 256, 0, s>>>(wt, sh.rows, sh.k, dx, sh.k, dy2, sh.rows);
            CK(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost));
            bad += memcmp(y1.data(), y2.data(), y1.size() * 4) != 0;
        }
        for (int v = 0; v < 3; ++v) {
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < iters; ++i) {
                const float * wt = dw + (size_t) (i % NW) * sh.rows * sh.k;
                if (v == 0) k_old<<<dim3((sh.rows + 7) / 8, 1), 256, 0, s>>>(wt, sh.rows, sh.k, dx, sh.k, dy1, sh.rows);
                else if (v == 1) k_new<5><<<dim3((sh.rows + 7) / 8, 1), 256, 0, s>>>(wt, sh.rows, sh.k, dx, sh.k, dy1, sh.rows);
                else k_new<8><<<dim3((sh.rows + 7) / 8, 1), 256, 0, s>>>(wt, sh.rows, sh.k, dx, sh.k, dy1, sh.rows);
            }
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            printf("%-24s %s: %.2f us  %.0f GB/s\n", sh.name, v == 0 ? "old     " : v == 1 ? "unroll 5" : "unroll 8", 1000.0 * ms / iters,
                   (double) sh.rows * sh.k * 4 / (ms / iters * 1e-3) / 1e9);
        }
        printf("%-24s bitwise: %d of 200 trials differ (unroll 5 vs old)\n", sh.name, bad);
    }
    return 0;
}
