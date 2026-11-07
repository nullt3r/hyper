// CPU expert row dots per block type on real expert rows: ggml's vec_dot against the Q8_K (or Q8_0 / Q8_1) activations,
// one thread with the rows in cache (the compute cost) and all threads streaming (the bandwidth), for every expert tensor
// of the given layers. usage: iqbench <model.gguf> <layer,layer,...> [threads=32]
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "../src/iq1s_dot.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace hyper;

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: iqbench model.gguf layer,layer,... [threads]\n"); return 1; }
    const int nth = argc > 3 ? atoi(argv[3]) : 32;
    GGUF g(argv[1]);
    ggml_cpu_init();
    std::vector<int> layers;
    for (const char * q = argv[2]; *q;) { layers.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    const int NE = getenv("NE") ? atoi(getenv("NE")) : 32;   // experts streamed (working set beyond the L3)
    for (int il : layers)
        for (const char * nm : {"ffn_gate_exps", "ffn_down_exps"}) {
            const GTensor & t = g.need("blk." + std::to_string(il) + "." + nm + ".weight");
            const ggml_type ty = (ggml_type) t.type;
            const auto * tt = ggml_get_type_traits_cpu(ty);
            const ggml_type vt = tt->vec_dot_type;
            const int k = (int) t.ne[0], rows_e = (int) t.ne[1], E = (int) t.ne[2];
            const size_t rb = t.row_bytes(), R = (size_t) rows_e * std::min(NE, E);
            std::vector<uint8_t> w(R * rb);
            memcpy(w.data(), t.data + (size_t) (E / 3) * rows_e * rb, w.size());
            std::vector<float> x(k);
            std::mt19937 rng(il);
            for (auto & v : x) v = std::normal_distribution<float>(0, 1)(rng);
            std::vector<uint8_t> q(ggml_row_size(vt, k));
            ggml_get_type_traits_cpu(vt)->from_float(x.data(), q.data(), k);
            // ours (where we have one): every row's result bitwise against ggml's
            ggml_vec_dot_t ours = nullptr;
            if (ty == GGML_TYPE_IQ1_S) ours = [](int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) { iq1s_dot(n, s, vx, vy); };
            if (ours) {
                long bad = 0;
                for (size_t r = 0; r < R; ++r) {
                    float a, b;
                    tt->vec_dot(k, &a, 0, w.data() + r * rb, 0, q.data(), 0, 1);
                    ours(k, &b, 0, w.data() + r * rb, 0, q.data(), 0, 1);
                    bad += memcmp(&a, &b, 4) != 0;
                }
                printf("blk.%-2d %-14s ours vs ggml: bitwise mismatches %ld of %zu rows\n", il, nm, bad, R);
            }
            std::vector<float> y(R);
            const size_t RC = std::min<size_t>(R, 2048);
            for (int v = 0; v < (ours ? 2 : 1); ++v) {
                const ggml_vec_dot_t fn = v ? ours : tt->vec_dot;
                double best = 1e9;
                for (int rep = 0; rep < 7; ++rep) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (size_t r = 0; r < RC; ++r) fn(k, &y[r], 0, w.data() + r * rb, 0, q.data(), 0, 1);
                    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                double bests = 1e9;
                for (int rep = 0; rep < 5; ++rep) {
                    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
                    for (size_t c = 0; c < R / 32; ++c)
                        for (size_t r = c * 32; r < c * 32 + 32; ++r) fn(k, &y[r], 0, w.data() + r * rb, 0, q.data(), 0, 1);
                    bests = std::min(bests, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                printf("blk.%-2d %-14s %-8s %-4s k %5d %4zu B/row  1 thread cached: %6.1f ns/row %5.2f GB/s  %d threads streaming: %5.1f GB/s\n",
                       il, nm, ggml_type_name(ty), v ? "ours" : "ggml", k, rb, best / RC * 1e9, RC * rb / best / 1e9, nth, R * rb / bests / 1e9);
                fflush(stdout);
            }
        }
    return 0;
}
