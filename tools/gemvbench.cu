// Q8 GEMV bandwidth microbenchmark on one GPU: the per-GPU dense shapes of GLM-5.3-Flash (3-way TP, 21 heads) and the
// 27B, random fragment-ordered weights. usage: gemvbench [iters]
#include "../src/kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    struct Shape { const char * name; int n, k; };
    std::vector<Shape> shapes = {
        {"glm kda in (8341x4096)", 8341, 4096},
        {"glm kda out (4096x2688)", 4096, 2688},
        {"glm mla q_b+idx (9472x1536)", 9472, 1536},
        {"glm mla in (2304x4096)", 2304, 4096},
        {"glm mla out (4096x5376)", 4096, 5376},
        {"glm shexp gu (1344x4096)", 1344, 4096},
        {"glm shexp down (4096x672)", 4096, 672},
        {"glm dense gu (8192x4096)", 8192, 4096},
        {"glm output (51627x4096)", 51627, 4096},
        {"27b ffn gu (11584x5120)", 11584, 5120},
        {"27b ffn down (5120x5792)", 5120, 5792},
        {"27b gdn in (5152x5120)", 5152, 5120},
        {"27b gdn out (5120x1920)", 5120, 1920},
        {"27b attn out (5120x2048)", 5120, 2048},
        {"fn hc up (10240x320)", 10240, 320},
        {"fn hc down (320x10240)", 320, 10240},
    };
    if (argc > 3) shapes = {{"custom", atoi(argv[2]), atoi(argv[3])}};   // gemvbench iters n k
    CK(cudaSetDevice(0));
    hyper::gemv_init(0);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int nt : {1, 4}) {
        printf("-- tokens per call %d\n", nt);
        for (auto & sh : shapes) {
            const size_t tiles = (sh.n + 15) / 16, kb = sh.k / 32;
            const size_t qbytes = tiles * kb * 512, sbytes = tiles * kb * 16 * 2;
            const int nmat = std::max<int>(2, (int) (64e6 / (qbytes + 1)) + 1);   // working set beyond the 6 MB L2
            std::vector<hyper::Q8W> W(nmat);
            for (int m = 0; m < nmat; ++m) {
                void * q, * sc;
                CK(cudaMalloc(&q, qbytes)); CK(cudaMalloc(&sc, sbytes));
                CK(cudaMemset(q, 1, qbytes)); CK(cudaMemset(sc, 0, sbytes));
                W[m].q = (const uint4 *) q; W[m].s = (const half *) sc; W[m].n = sh.n; W[m].k = sh.k;
            }
            float * x, * y; CK(cudaMalloc(&x, 4 * sh.k * 4)); CK(cudaMalloc(&y, 4 * sh.n * 4));
            CK(cudaMemset(x, 0, 4 * sh.k * 4));
            cudaGraph_t gr; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < iters; ++i) hyper::gemv_q8(W[i % nmat], x, sh.k, y, sh.n, nullptr, nt, s);
            CK(cudaStreamEndCapture(s, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e0, s));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            const double bytes = (double) qbytes + sbytes;
            printf("%-30s %7.1f us  %6.0f GB/s\n", sh.name, 1000.0 * ms / iters, bytes * iters / (ms * 1e-3) / 1e9);
            for (auto & w : W) { cudaFree((void *) w.q); cudaFree((void *) w.s); }
            cudaFree(x); cudaFree(y);
            cudaGraphExecDestroy(ge); cudaGraphDestroy(gr);
        }
    }
    return 0;
}
