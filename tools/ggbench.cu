// K-quant dense GEMV (gemv_kq on repacked Q4_K / Q6_K) on one GPU: correctness against ggml's dequantization (fp16-rounded
// inputs, double accumulation) and GB/s of weight bytes on the per-GPU dense shapes of Flash-Next with K-quant dense
// weights (orcarouter's Q4_K_M). usage: ggbench [iters] [q4k|q6k n k]
#include "../src/kernels.cuh"
#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

using hyper::KQ;
int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    struct Shape { const char * name; KQ t; int n, k; int k_src = 0; };   // k_src: gathered 128-column ranges of wider rows
    std::vector<Shape> shapes = {
        {"gdn in q4k (5461x2560)", KQ::Q4K, 5461, 2560},
        {"gdn in q6k (5461x2560)", KQ::Q6K, 5461, 2560},
        {"gdn out q4k (2560x2048)", KQ::Q4K, 2560, 2048},
        {"gdn out q4k gathered", KQ::Q4K, 2560, 2048, 6144},
        {"hc down q4k (320x10240)", KQ::Q4K, 320, 10240},
        {"hc down q6k (320x10240)", KQ::Q6K, 320, 10240},
        {"shexp gu q4k (426x2560)", KQ::Q4K, 426, 2560},
        {"output q6k (20000x2560)", KQ::Q6K, 20000, 2560},
    };
    if (argc > 4) shapes = {{"custom", std::string(argv[2]) == "q6k" ? KQ::Q6K : KQ::Q4K, atoi(argv[3]), atoi(argv[4])}};
    hyper::gemv_init(0);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::mt19937 rng(1);
    int bad = 0;
    for (int nt : {1, 4}) {
        printf("-- tokens per call %d\n", nt);
        for (auto & sh : shapes) {
            const ggml_type gt = sh.t == KQ::Q4K ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
            const int ks = sh.k_src ? sh.k_src : sh.k;
            const size_t bb = sh.t == KQ::Q4K ? 144 : 210, rb = ks / 256 * bb, gbytes = rb * sh.n;
            std::vector<uint8_t> rows(gbytes);
            for (auto & b : rows) b = (uint8_t) rng();
            for (size_t i = 0; i < gbytes / bb; ++i) {   // sane fp16 d (and dmin)
                uint8_t * blk = rows.data() + i * bb;
                const __half h = __float2half(0.001f * (1 + rng() % 16)), hm = __float2half(0.0005f * (1 + rng() % 16));
                if (sh.t == KQ::Q6K) memcpy(blk + 208, &h, 2); else { memcpy(blk, &h, 2); memcpy(blk + 2, &hm, 2); }
            }
            hyper::KQHost h;
            std::vector<int> kbl(sh.k / 32);
            for (int b = 0; b < sh.k / 32; ++b) kbl[b] = sh.k_src ? (b / 4) * 12 + b % 4 : b;   // (gathered: 4 of every 12 blocks)
            const int dg = hyper::repack_kq(sh.t, rows.data(), rb, sh.n, kbl, h);
            auto up = [&](const auto & v) -> void * {
                if (v.empty()) return nullptr;
                void * p; CK(cudaMalloc(&p, v.size() * sizeof(v[0]))); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice));
                return p;
            };
            const size_t kbytes = h.lo.size() * 8 + h.hi.size() * 4 + h.scm.size() * 2 + h.sc6.size() + h.d.size() * 4;
            const int nmat = std::max<int>(2, (int) (64e6 / kbytes) + 1);   // working set beyond the 6 MB L2
            std::vector<hyper::KQW> W(nmat);
            for (auto & w : W) {
                w.type = sh.t; w.n = sh.n; w.k = sh.k; w.dg = dg;
                w.lo = (const uint2 *) up(h.lo); w.hi = (const unsigned *) up(h.hi); w.scm = (const uint16_t *) up(h.scm);
                w.sc6 = (const int8_t *) up(h.sc6); w.d = (const half2 *) up(h.d);
            }
            std::vector<float> xh((size_t) nt * sh.k);
            for (auto & v : xh) v = __half2float(__float2half(std::uniform_real_distribution<float>(-1, 1)(rng)));
            float * x, * y; CK(cudaMalloc(&x, xh.size() * 4)); CK(cudaMalloc(&y, (size_t) nt * sh.n * 4));
            CK(cudaMemcpy(x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
            // correctness: rows dequantized by ggml, double accumulation
            hyper::gemv_kq(W[0], x, sh.k, y, sh.n, nt, s);
            std::vector<float> yg((size_t) nt * sh.n);
            CK(cudaMemcpy(yg.data(), y, yg.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<float> wr(ks), wsel(sh.k);
            double maxrel = 0;
            for (int r = 0; r < sh.n; r += std::max(1, sh.n / 512)) {
                ggml_get_type_traits(gt)->to_float(rows.data() + (size_t) r * rb, wr.data(), ks);
                for (int c = 0; c < sh.k; ++c) wsel[c] = wr[kbl[c / 32] * 32 + c % 32];
                for (int t = 0; t < nt; ++t) {
                    double ref = 0, mag = 0;
                    for (int c = 0; c < sh.k; ++c) { ref += (double) wsel[c] * xh[(size_t) t * sh.k + c]; mag += std::fabs((double) wsel[c] * xh[(size_t) t * sh.k + c]); }
                    maxrel = std::max(maxrel, std::fabs(yg[(size_t) t * sh.n + r] - ref) / (mag + 1e-30));
                }
            }
            const bool ok = maxrel < 1e-5;
            bad += !ok;
            cudaGraph_t gr; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < iters; ++i) hyper::gemv_kq(W[i % nmat], x, sh.k, y, sh.n, nt, s);
            CK(cudaStreamEndCapture(s, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e0, s));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            printf("%-28s %7.1f us  %5.0f GB/s  (%.2f bits/weight)  max rel err %.1e %s\n", sh.name, 1000.0 * ms / iters,
                   (double) kbytes * iters / (ms * 1e-3) / 1e9, kbytes * 8.0 / ((double) sh.n * sh.k), maxrel, ok ? "OK" : "WRONG");
            for (auto & w : W) { cudaFree((void *) w.lo); cudaFree((void *) w.hi); cudaFree((void *) w.scm); cudaFree((void *) w.sc6); cudaFree((void *) w.d); }
            cudaFree(x); cudaFree(y);
            cudaGraphExecDestroy(ge); cudaGraphDestroy(gr);
        }
    }
    return bad;
}
