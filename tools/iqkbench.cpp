// CPU expert dot throughput on real GLM expert weights: mainline ggml vec_dot vs ik_llama's (dlopen, private symbols)
// usage: iqkbench <model.gguf> <ik libggml.so> [layer=10] [threads=30]
#include "gguf.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <dlfcn.h>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace hyper;

// ik_llama's ggml_type_traits_t (ggml/include/ggml.h, 0821d62a)
typedef void (*ik_to_float_t)(const void *, float *, int64_t);
typedef void (*ik_from_float_t)(const float *, void *, int64_t);
typedef void (*ik_vec_dot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
struct IkTraits {
    const char * type_name;
    int64_t blck_size, blck_size_interleave;
    size_t type_size;
    bool is_quantized;
    ik_to_float_t to_float;
    ik_from_float_t from_float, from_float_ref;
    void * from_float_to_mat;
    ik_vec_dot_t vec_dot;
    int vec_dot_type;
    int64_t nrows, ncols;
    void * gemv, * gemm;
    int64_t row_meta_size;
};
typedef IkTraits (*ik_get_traits_t)(int);
typedef bool (*ik_mul_mat_t)(long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int);
struct IkInitParams { size_t mem_size; void * mem_buffer; bool no_alloc; };
typedef void * (*ik_init_t)(IkInitParams);

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: iqkbench model.gguf ik_libggml.so [layer] [threads]\n"); return 1; }
    const int layer = argc > 3 ? atoi(argv[3]) : 10, nth = argc > 4 ? atoi(argv[4]) : 30;
    GGUF g(argv[1]);
    void * h = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto ik_traits = (ik_get_traits_t) dlsym(h, "ggml_internal_get_type_traits");
    auto ik_mm = (ik_mul_mat_t) dlsym(h, "iqk_mul_mat");
    auto ik_init = (ik_init_t) dlsym(h, "ggml_init");
    if (!ik_traits || !ik_mm || !ik_init) { fprintf(stderr, "missing ik symbols\n"); return 1; }
    ik_init({1 << 20, nullptr, false});
    ggml_cpu_init();
    const std::string p = "blk." + std::to_string(layer) + ".";
    const int NE = getenv("NE") ? atoi(getenv("NE")) : 96;   // experts streamed per measurement (> the 128 MB L3)
    for (const char * mat : {"ffn_gate_exps.weight", "ffn_down_exps.weight"}) {
        const GTensor & t = g.need(p + mat);
        const int k = (int) t.ne[0], rows_e = (int) t.ne[1], E = (int) t.ne[2];
        const size_t rb = t.row_bytes(), eb = rb * rows_e;
        const int R = rows_e * NE;
        std::vector<uint8_t> w((size_t) NE * eb);
        memcpy(w.data(), t.data + (size_t) (E / 2) * eb, w.size());   // copy: anonymous memory like hyper's
        std::vector<float> x(k), y(R);
        std::mt19937 rng(1);
        for (auto & v : x) v = std::normal_distribution<float>(0, 1)(rng);
        const auto * mt = ggml_get_type_traits_cpu((ggml_type) t.type);
        const auto * mq = ggml_get_type_traits_cpu(mt->vec_dot_type);
        std::vector<uint8_t> qm(ggml_row_size(mt->vec_dot_type, k) + 64);
        mq->from_float(x.data(), qm.data(), k);
        const IkTraits it = ik_traits((int) t.type), iq = ik_traits(it.vec_dot_type);
        std::vector<uint8_t> qi(16 * k + 4096);
        iq.from_float(x.data(), qi.data(), k);
        std::fill(y.begin(), y.end(), 0.0f);
        auto bench = [&](const char * name, auto && body) {
            body();   // warm
            double best = 1e9;
            for (int rep = 0; rep < 5; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                body();
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            static std::vector<float> ref;
            if (ref.empty() || ref.size() != y.size()) ref = y;
            double md = 0; int bad = 0;
            for (size_t i = 0; i < y.size(); ++i) { const double d = std::fabs(y[i] - ref[i]); md = std::max(md, d); bad += d > 1e-3 * (1 + std::fabs(ref[i])); }
            printf("%-22s %-7s %6.1f GB/s  (%.3f ms for %d rows)  max|dy| %.2g  mismatches %d\n", mat, name, w.size() / best / 1e9, best * 1e3, R, md, bad);
        };
        bench("main", [&] {
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
            for (int c = 0; c < R / 16; ++c)
                for (int r = c * 16; r < c * 16 + 16; ++r) mt->vec_dot(k, &y[r], 0, w.data() + (size_t) r * rb, 0, qm.data(), 0, 1);
        });
        bench("ik", [&] {
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
            for (int c = 0; c < R / 16; ++c)
                for (int r = c * 16; r < c * 16 + 16; ++r) it.vec_dot(k, &y[r], 0, w.data() + (size_t) r * rb, 0, qi.data(), 0, 1);
        });
        bench("iqk_mm", [&] {
#pragma omp parallel num_threads(nth)
            {
                const int ith = omp_get_thread_num();
                if (!ik_mm(R, 1, k, (int) t.type, w.data(), (long) rb, it.vec_dot_type, qi.data(), (long) (iq.type_size * k / iq.blck_size),
                           y.data(), R, ith, nth) && ith == 0)
                    printf("  (iqk_mul_mat declined)\n");
            }
        });
        printf("  types: weights %s, activations main %s / ik %s\n", ggml_type_name((ggml_type) t.type), ggml_type_name(mt->vec_dot_type),
               iq.type_name);
    }
    return 0;
}
