// grouped GEMV (NormIn xg_tiles / xg_stride) and dense_rows_view on a real DeepSeek attn_output_a (block-diagonal groups):
// one grouped launch vs the per-group views vs a CPU fp32 reference, decode rows (1..4) and a bulk chunk (views -> GEMMs)
// usage: grouptest <model.gguf> <layer,layer,...>
#include "../src/dense.h"
#include "../src/kernels4.cuh"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace hyper;

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: grouptest model.gguf layer,...\n"); return 1; }
    GGUF g(argv[1]);
    const int o_groups = (int) g.get_int("deepseek41.attention.output_group_count"), o_lora = (int) g.get_int("deepseek41.attention.output_lora_rank");
    std::vector<int> layers;
    for (const char * q = argv[2]; *q;) { layers.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    int dev = 0;
    cudaGetDevice(&dev);
    gemv_init(dev);
    cudaStream_t s; cudaStreamCreate(&s);
    cublasHandle_t blas; cublasCreate(&blas); cublasSetStream(blas, s);
    std::vector<void *> allocs;
    auto A = [&](size_t nb) { void * p; cudaMalloc(&p, nb); allocs.push_back(p); return p; };
    int bad = 0;
    for (int il : layers) {
        const GTensor & t = g.need("blk." + std::to_string(il) + ".attn_output_a.weight");
        const int k = (int) t.ne[0], rows = (int) t.rows(), per = k, HQ = o_groups * per, ol = o_groups * o_lora;
        DW all = upload_dense(A, dev, {{&t, 0, rows}});
        std::vector<DW> views;
        for (int gi = 0; gi < o_groups; ++gi) views.push_back(dense_rows_view(all, gi * o_lora, (gi + 1) * o_lora));
        const std::vector<float> W = to_f32(t);   // [rows][k]
        for (int nt : {1, 4, 64}) {
            std::mt19937 rng(il * 7 + nt);
            std::vector<float> x((size_t) nt * HQ);
            for (auto & v : x) v = std::normal_distribution<float>()(rng);
            // reference: group gi's rows on x columns [gi * per, (gi + 1) * per)
            std::vector<double> ref((size_t) nt * ol);
            for (int r = 0; r < nt; ++r)
                for (int gi = 0; gi < o_groups; ++gi)
                    for (int j = 0; j < o_lora; ++j) {
                        double a = 0;
                        const float * w = W.data() + (size_t) (gi * o_lora + j) * k, * xr = x.data() + (size_t) r * HQ + (size_t) gi * per;
                        for (int i = 0; i < k; ++i) a += (double) w[i] * xr[i];
                        ref[(size_t) r * ol + gi * o_lora + j] = a;
                    }
            float * dx = (float *) A(x.size() * 4), * dy = (float *) A((size_t) nt * ol * 4);
            half * xh = (half *) A((size_t) nt * HQ * 2), * ggs = (half *) A((size_t) 32 << 21);
            cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
            const DenseRun dr{s, blas, xh, ggs, (size_t) 32 << 20};
            auto check = [&](const char * what) {
                cudaStreamSynchronize(s);
                std::vector<float> y((size_t) nt * ol);
                cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
                double md = 0, mr = 0;
                for (size_t i = 0; i < y.size(); ++i) { md = std::max(md, std::fabs(y[i] - ref[i])); mr = std::max(mr, std::fabs(ref[i])); }
                const bool ok = md <= 2e-3 * mr;
                bad += !ok;
                printf("blk.%-2d %-7s nt %2d %-8s max|gpu-ref| %.3e / max|ref| %.3e  rel %.2e %s\n", il, gtype_name(t.type), nt, what, md, mr, md / mr,
                       ok ? "OK" : "MISMATCH");
            };
            cudaMemset(dy, 0, (size_t) nt * ol * 4);
            for (int gi = 0; gi < o_groups; ++gi) dense_mm(dr, views[gi], dx + (size_t) gi * per, HQ, dy + (size_t) gi * o_lora, ol, nt);
            check("views");
            if (nt <= MAX_NT) {
                cudaMemset(dy, 0, (size_t) nt * ol * 4);
                NormIn gn; gn.xg_tiles = o_lora / 16; gn.xg_stride = per;
                dense_mm(dr, all, dx, HQ, dy, ol, nt, gn);
                check("grouped");
            }
        }
    }
    printf("GROUPTEST %s\n", bad ? "FAILED" : "all OK");
    return bad ? 1 : 0;
}
