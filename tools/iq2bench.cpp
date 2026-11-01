// IQ2_XXS x Q8_K row dot (AVX2) variants vs ggml's vec_dot on real DeepSeek-V4.1 expert rows: speed (one thread, rows in
// cache, and all threads streaming) and bitwise equality of every result.
// usage: iq2bench <model.gguf> [layer=10] [threads=24]
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <immintrin.h>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../src/iq2_dot.h"

using namespace hyper;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: iq2bench model.gguf [layer] [threads]\n"); return 1; }
    const int layer = argc > 2 ? atoi(argv[2]) : 10, nth = argc > 3 ? atoi(argv[3]) : 24;
    GGUF g(argv[1]);
    ggml_cpu_init();
    const GTensor & t = g.need("blk." + std::to_string(layer) + ".ffn_gate_exps.weight");
    if (t.type != GType::IQ2_XXS) { fprintf(stderr, "not IQ2_XXS\n"); return 1; }
    const int k = (int) t.ne[0];
    const size_t rb = t.row_bytes();
    const int E = (int) t.ne[2], rows_e = (int) t.ne[1];
    const int NE = getenv("NE") ? atoi(getenv("NE")) : 32;
    const size_t R = (size_t) rows_e * NE;
    std::vector<uint8_t> w(R * rb);
    memcpy(w.data(), t.data + (size_t) (E / 2) * rows_e * rb, w.size());
    const auto * tt = ggml_get_type_traits_cpu(GGML_TYPE_IQ2_XXS);
    const auto * tq = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    std::mt19937 rng(1);
    const int NX = 8;   // activation rows (several distributions)
    std::vector<std::vector<uint8_t>> q8(NX, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_K, k)));
    std::vector<std::vector<uint8_t>> q8u(NX, std::vector<uint8_t>(k));
    for (int xi = 0; xi < NX; ++xi) {
        std::vector<float> x(k);
        for (auto & v : x) {
            float f = std::normal_distribution<float>(0, 1)(rng);
            if (xi % 4 == 1) f = f * f * f;
            if (xi % 4 == 2) f = (rng() % 50 == 0) ? 40.0f * f : 0.05f * f;
            if (xi % 4 == 3) f = std::round(f * 4) / 4;
            v = f;
        }
        tq->from_float(x.data(), q8[xi].data(), k);
        q8k_to_u8((const block_q8_K *) q8[xi].data(), k / 256, q8u[xi].data());
    }
    long bad = 0;
    for (int xi = 0; xi < NX; ++xi)
        for (size_t r = 0; r < R; ++r) {
            float a, b;
            tt->vec_dot(k, &a, 0, w.data() + r * rb, 0, q8[xi].data(), 0, 1);
            b = iq2xxs_dot_u(k, w.data() + r * rb, (const block_q8_K *) q8[xi].data(), q8u[xi].data());
            bad += memcmp(&a, &b, 4) != 0;
        }
    printf("bitwise mismatches: %ld of %zu\n", bad, R * NX);
    const size_t RC = std::min<size_t>(R, 2048);
    auto run = [&](const char * name, auto fn) {
        std::vector<float> y(R);
        double best = 1e9;
        for (int rep = 0; rep < 7; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            for (size_t r = 0; r < RC; ++r) fn(r, &y[r]);
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        double bests = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
            for (size_t c = 0; c < R / 32; ++c)
                for (size_t r = c * 32; r < c * 32 + 32; ++r) fn(r, &y[r]);
            bests = std::min(bests, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("%-8s 1 thread cached: %6.1f ns/row (%5.2f GB/s)   %d threads streaming: %5.1f GB/s\n",
               name, best / RC * 1e9, RC * rb / best / 1e9, nth, R * rb / bests / 1e9);
    };
    run("ggml", [&](size_t r, float * s) { tt->vec_dot(k, s, 0, w.data() + r * rb, 0, q8[0].data(), 0, 1); });
    run("tab", [&](size_t r, float * s) { *s = iq2xxs_dot_u(k, w.data() + r * rb, (const block_q8_K *) q8[0].data(), q8u[0].data()); });
    return 0;
}
