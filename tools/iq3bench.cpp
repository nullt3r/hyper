// IQ3_S x Q8_K row dot (AVX2) variants vs ggml's vec_dot on real GLM expert rows: speed (one thread, rows in cache, and
// all threads streaming) and bitwise equality of every result.
// usage: iq3bench <model.gguf> [layer=10] [threads=30]
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

#include "../src/iq3s_dot.h"

using namespace hyper;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: iq3bench model.gguf [layer] [threads]\n"); return 1; }
    const int layer = argc > 2 ? atoi(argv[2]) : 10, nth = argc > 3 ? atoi(argv[3]) : 30;
    GGUF g(argv[1]);
    ggml_cpu_init();
    const GTensor & t = g.need("blk." + std::to_string(layer) + ".ffn_gate_exps.weight");
    if (t.type != GType::IQ3_S) { fprintf(stderr, "not IQ3_S\n"); return 1; }
    const int k = (int) t.ne[0];
    const size_t rb = t.row_bytes();
    const int E = (int) t.ne[2], rows_e = (int) t.ne[1];
    const int NE = getenv("NE") ? atoi(getenv("NE")) : 96;
    const size_t R = (size_t) rows_e * NE;
    std::vector<uint8_t> w(R * rb);
    memcpy(w.data(), t.data + (size_t) (E / 2) * rows_e * rb, w.size());
    std::vector<float> x(k);
    std::mt19937 rng(1);
    for (auto & v : x) v = std::normal_distribution<float>(0, 1)(rng);
    const auto * tt = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S);
    const auto * tq = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    std::vector<uint8_t> q8(ggml_row_size(GGML_TYPE_Q8_K, k));
    tq->from_float(x.data(), q8.data(), k);
    std::vector<float> ref(R), y(R);
    for (size_t r = 0; r < R; ++r) tt->vec_dot(k, &ref[r], 0, w.data() + r * rb, 0, q8.data(), 0, 1);
    struct V { const char * name; void (*f)(int, float *, const void *, const void *); };
    const V vs[] = {
        {"ggml", [](int n, float * s, const void * vx, const void * vy) { ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S)->vec_dot(n, s, 0, vx, 0, vy, 0, 1); }},
        {"scalar64", iq3s_dot_v1},
        {"gather", iq3s_dot_v2},
    };
    const size_t RC = std::min<size_t>(R, 2048);   // cached: 2048 rows (~3.5 MB) over and over, one thread
    for (const V & v : vs) {
        int bad = 0;
        for (size_t r = 0; r < R; ++r) { v.f(k, &y[r], w.data() + r * rb, q8.data()); bad += memcmp(&y[r], &ref[r], 4) != 0; }
        double best = 1e9;
        for (int rep = 0; rep < 7; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            for (size_t r = 0; r < RC; ++r) v.f(k, &y[r], w.data() + r * rb, q8.data());
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        double bests = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
            for (size_t c = 0; c < R / 16; ++c)
                for (size_t r = c * 16; r < c * 16 + 16; ++r) v.f(k, &y[r], w.data() + r * rb, q8.data());
            bests = std::min(bests, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("%-9s 1 thread cached: %6.1f ns/row (%5.2f GB/s)   %d threads streaming: %5.1f GB/s   bitwise mismatches %d of %zu\n",
               v.name, best / RC * 1e9, RC * rb / best / 1e9, nth, R * rb / bests / 1e9, bad, R);
    }
    {   // IQ4_XS down rows: 4 resumed parts vs ggml's vec_dot, bitwise
        const GTensor & td = g.need("blk." + std::to_string(layer) + ".ffn_down_exps.weight");
        if (td.type == GType::IQ4_XS) {
            const int kd = (int) td.ne[0], nbk = kd / 256;
            const size_t rbd = td.row_bytes(), Rd = std::min<size_t>((size_t) td.ne[1] * 8, 32768);
            std::vector<float> hx(kd);
            for (auto & v : hx) v = std::normal_distribution<float>(0, 1)(rng);
            std::vector<uint8_t> qd(ggml_row_size(GGML_TYPE_Q8_K, kd));
            tq->from_float(hx.data(), qd.data(), kd);
            const auto * t4 = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS);
            int bad = 0;
            for (size_t r = 0; r < Rd; ++r) {
                float a, b;
                const uint8_t * row = td.data + (size_t) (E / 2) * td.ne[1] * rbd + r * rbd;
                t4->vec_dot(kd, &a, 0, row, 0, qd.data(), 0, 1);
                __m256 acc = _mm256_setzero_ps();
                for (int q = 0; q < 4; ++q) acc = iq4xs_dot_part(row, qd.data(), q * nbk / 4, (q + 1) * nbk / 4, acc);
                b = iq3s_hsum8(acc);
                bad += memcmp(&a, &b, 4) != 0;
            }
            printf("IQ4_XS resumed in 4 parts: bitwise mismatches %d of %zu rows\n", bad, Rd);
            // streaming speed, all threads: ggml vec_dot per row / our kernel per row / our kernel part-major (4 passes)
            const size_t NR = (size_t) td.ne[1] * std::min<int64_t>(td.ne[2], 96);
            const uint8_t * base = td.data;
            std::vector<uint8_t> wd(NR * rbd);
            memcpy(wd.data(), base, wd.size());
            std::vector<float> yo(NR), st(NR * 8);
            auto timeit = [&](const char * nm, auto && body) {
                double best = 1e9;
                for (int rep = 0; rep < 5; ++rep) {
                    auto t0 = std::chrono::steady_clock::now();
                    body();
                    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                printf("IQ4_XS %-22s %5.1f GB/s\n", nm, wd.size() / best / 1e9);
            };
            timeit("ggml rows", [&] {
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
                for (size_t c = 0; c < NR / 32; ++c) for (size_t r = c * 32; r < c * 32 + 32; ++r) t4->vec_dot(kd, &yo[r], 0, wd.data() + r * rbd, 0, qd.data(), 0, 1);
            });
            timeit("ours rows", [&] {
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
                for (size_t c = 0; c < NR / 32; ++c) for (size_t r = c * 32; r < c * 32 + 32; ++r)
                    yo[r] = iq3s_hsum8(iq4xs_dot_part(wd.data() + r * rbd, qd.data(), 0, nbk, _mm256_setzero_ps()));
            });
            timeit("ours 4 part passes", [&] {
                for (int q = 0; q < 4; ++q) {
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
                    for (size_t c = 0; c < NR / 32; ++c) for (size_t r = c * 32; r < c * 32 + 32; ++r) {
                        __m256 a = q ? _mm256_loadu_ps(&st[r * 8]) : _mm256_setzero_ps();
                        a = iq4xs_dot_part(wd.data() + r * rbd, qd.data(), q * nbk / 4, (q + 1) * nbk / 4, a);
                        _mm256_storeu_ps(&st[r * 8], a);
                    }
                }
            });
        }
    }
    return 0;
}
