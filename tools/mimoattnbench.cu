// MiMo decode attention (global layer shape: k 192 / v 128, 16 q heads per kv head) on synthetic caches: time per call and the
// effective cache read bandwidth, for a few depths, kv heads per GPU, rows (1 token / 4 = token + 3 drafts) and key splits.
// usage: mimoattnbench [depth=160000] [iters=200]
#include "../src/kernels6.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace hyper;

__global__ void k_fill(half * p, size_t n, unsigned seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        unsigned h = (unsigned) i * 2654435761u ^ seed;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        p[i] = __float2half(((float) (h & 0xffff) / 65535.0f - 0.5f) * 2.0f);
    }
}

int main(int argc, char ** argv) {
    const int depth = argc > 1 ? atoi(argv[1]) : 160000, iters = argc > 2 ? atoi(argv[2]) : 200;
    const int DK = 192, DV = 128, G = 16, maxkv = 2;
    const size_t kpitch = (size_t) depth * DK, vpitch = (size_t) depth * DV;
    half * kc, * vc;
    cudaMalloc(&kc, maxkv * kpitch * sizeof(half));
    cudaMalloc(&vc, maxkv * vpitch * sizeof(half));
    k_fill<<<1024, 256>>>(kc, maxkv * kpitch, 1);
    k_fill<<<1024, 256>>>(vc, maxkv * vpitch, 2);
    const int maxq = maxkv * G, maxt = 4;
    std::vector<float> hq((size_t) maxt * maxq * DK);
    std::mt19937 rng(3);
    for (auto & v : hq) v = std::normal_distribution<float>(0, 1)(rng);
    float * q, * out, * part;
    int * pos;
    cudaMalloc(&q, hq.size() * sizeof(float));
    cudaMemcpy(q, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&out, (size_t) maxt * maxq * DV * sizeof(float));
    const size_t pf = mimo_attn_part_floats(maxkv, maxt, G, 256);
    cudaMalloc(&part, pf * sizeof(float));
    cudaMalloc(&pos, sizeof(int));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int depth_q : {depth / 4, depth / 2, depth}) {
        for (int nkv : {1, 2})
            for (int nt : {1, 4})
                for (int ns : {32, 64, 128, 256}) {
                    const int p0 = depth_q - nt;
                    cudaMemcpy(pos, &p0, sizeof(int), cudaMemcpyHostToDevice);
                    MimoAttn a;
                    a.q = q; a.qs = maxq * DK;
                    a.kc = kc; a.vc = vc; a.kpitch = kpitch; a.vpitch = vpitch; a.ring = 0;
                    a.pos = pos; a.nt = nt; a.nq = nkv * G; a.nkv = nkv; a.dk = DK; a.dv = DV;
                    a.window = 0; a.scale = 1.0f / sqrtf((float) DK); a.vscale = 0.707f; a.sinks = nullptr;
                    a.out = out; a.os = nkv * G * DV; a.part = part;
                    for (int i = 0; i < 5; ++i) mimo_attn(a, ns, 0);
                    cudaEventRecord(e0);
                    for (int i = 0; i < iters; ++i) mimo_attn(a, ns, 0);
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    const double t = ms / iters * 1e-3, bytes = (double) nkv * depth_q * (DK + DV) * 2;
                    const cudaError_t err = cudaGetLastError();
                    if (getenv("DUMP") && depth_q == depth && ns == 32) {   // (outputs of this config, for comparing kernels)
                        std::vector<float> ho((size_t) nt * nkv * G * DV);
                        cudaMemcpy(ho.data(), out, ho.size() * sizeof(float), cudaMemcpyDeviceToHost);
                        FILE * f = fopen((std::string(getenv("DUMP")) + "_" + std::to_string(nkv) + "_" + std::to_string(nt)).c_str(), "wb");
                        fwrite(ho.data(), 4, ho.size(), f); fclose(f);
                    }
                    printf("ATTN depth %6d  kv heads %d  rows %d  splits %3d: %8.1f us  %6.1f GB/s%s\n", depth_q, nkv, nt, ns, t * 1e6, bytes / t / 1e9,
                           err == cudaSuccess ? "" : cudaGetErrorString(err));
                }
    }
    return 0;
}
