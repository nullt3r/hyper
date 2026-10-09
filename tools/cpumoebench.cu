// CPU expert decode jobs on real GLM / Flash-Next expert weights, without the GPUs: the CpuMoe team as the engines drive
// it (expect -> record -> seq -> output seq), routing drawn from NE resident copies of the layer's experts.
// Prints time per job, bandwidth and an output checksum (compare variants for identical results).
// usage: cpumoebench <model.gguf> [layer=10] [threads=30] [experts/job=3] [tokens/job=1] [jobs=2000]
#include "cpu_moe.h"
#include "gguf.h"
#include "ggml-cpu.h"

#include <sys/mman.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace hyper;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: cpumoebench model.gguf [layer] [threads] [experts/job] [tokens/job] [jobs]\n"); return 1; }
    const int layer = argc > 2 ? atoi(argv[2]) : 10, nth = argc > 3 ? atoi(argv[3]) : 30;
    const int epj = argc > 4 ? atoi(argv[4]) : 3, nt = argc > 5 ? atoi(argv[5]) : 1, jobs = argc > 6 ? atoi(argv[6]) : 2000;
    const int NE = getenv("NE") ? atoi(getenv("NE")) : 64;   // resident experts (working set far beyond the L3)
    if (getenv("QTEST")) {   // Q8_K quantizer of the CPU MoE vs ggml's, bitwise, on random rows of several shapes
        std::mt19937 rq(11);
        std::vector<float> xs(4096);
        std::vector<uint8_t> a(16 * 292), b(16 * 292);
        long bad = 0, rows = 0;
        for (int it = 0; it < atoi(getenv("QTEST")); ++it) {
            const int kind = it % 4;
            for (auto & v : xs) {
                float f = std::normal_distribution<float>(0, 1)(rq);
                if (kind == 1) f = f * f * f * 30.0f;                       // heavy tails
                if (kind == 2) f = std::round(f * 8.0f) / 8.0f;            // ties
                if (kind == 3) f = (rq() % 64 == 0) ? 50.0f * f : 0.01f * f;
                v = f;
            }
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(xs.data(), a.data(), 4096);
            q8k_test_quantize(xs.data(), b.data(), 4096);
            if (memcmp(a.data(), b.data(), a.size()) != 0) {
                if (bad < 4)
                    for (int blk = 0; blk < 16; ++blk) {
                        const uint8_t * pa = a.data() + blk * 292, * pb = b.data() + blk * 292;
                        if (!memcmp(pa, pb, 292)) continue;
                        float da, db; memcpy(&da, pa, 4); memcpy(&db, pb, 4);
                        int j = 4; while (j < 292 && pa[j] == pb[j]) ++j;
                        printf("  kind %d block %d: d %.9g vs %.9g, first diff byte %d (%d vs %d) x=%.9g\n", kind, blk, da, db, j, (int8_t) pa[j], (int8_t) pb[j],
                               j < 260 ? xs[blk * 256 + j - 4] : 0.0f);
                        break;
                    }
                ++bad;
            }
            ++rows;
        }
        printf("QTEST %ld rows of 4096: %ld differ\n", rows, bad);
        return 0;
    }
    GGUF g(argv[1]);
    const std::string p = "blk." + std::to_string(layer) + ".";
    const GTensor & tg = g.need(p + "ffn_gate_exps.weight"), & tu = g.need(p + "ffn_up_exps.weight"), & td = g.need(p + "ffn_down_exps.weight");
    const int n = (int) tg.ne[0], ff = (int) tg.ne[1], E = (int) tg.ne[2];
    const size_t gb = tg.nbytes / E, db = td.nbytes / E;
    const size_t bytes = (size_t) NE * (2 * gb + db), H = 2u << 20, sz = (bytes + H - 1) / H * H;
    uint8_t * buf = (uint8_t *) mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(buf, sz, getenv("NOHUGE") ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
    CpuExpertLayer cl;
    cl.tg = tg.type; cl.td = td.type; cl.gate_bytes = gb; cl.down_bytes = db;
    cl.gate = buf; cl.up = buf + (size_t) NE * gb; cl.down = buf + (size_t) NE * 2 * gb;
    for (int e = 0; e < NE; ++e) {   // experts E/2 .. E/2 + NE - 1 of the layer
        memcpy((uint8_t *) cl.gate + (size_t) e * gb, tg.data + (size_t) (E / 2 + e) * gb, gb);
        memcpy((uint8_t *) cl.up + (size_t) e * gb, tu.data + (size_t) (E / 2 + e) * gb, gb);
        memcpy((uint8_t *) cl.down + (size_t) e * db, td.data + (size_t) (E / 2 + e) * db, db);
    }
    if (getenv("REG")) {   // pinned for the GPUs as in the engines
        if (cudaHostRegister(buf, sz, cudaHostRegisterPortable) != cudaSuccess) { fprintf(stderr, "cudaHostRegister failed\n"); return 1; }
    }
    cl.owned.assign(NE, 1);
    const int K = 8;
    std::vector<CpuMoeRec> recs(1);
    std::vector<CpuMoeOut> outs(1);
    memset(recs.data(), 0, sizeof(CpuMoeRec));
    memset(outs.data(), 0, sizeof(CpuMoeOut));
    CpuMoe cpu(nth, n, ff, K, recs.data(), outs.data(), 1);
    cpu.set_layer(0, cl);
    if (getenv("CLAMP")) cpu.set_clamp((float) atof(getenv("CLAMP")));
    std::mt19937 rng(7);
    std::vector<float> x((size_t) nt * n);
    for (auto & v : x) v = std::normal_distribution<float>(0, 1)(rng);
    double sum_t = 0, cs = 0;
    long long hits = 0;
    for (int it = -50; it < jobs; ++it) {   // (50 warm-up jobs)
        const unsigned counter = (unsigned) (it + 100);
        cpu.expect(counter, {0});
        CpuMoeRec & r = recs[0];
        r.nt = nt;
        for (int t = 0; t < nt; ++t) {
            // epj CPU experts per token (others -1: elsewhere); tokens share a few of them (as in a speculative batch)
            for (int j = 0; j < MOE_MAX_USED; ++j) { r.ids[t][j] = -1; r.wts[t][j] = 0.0f; }
            for (int j = 0; j < epj; ++j) {
                int e;
                if (getenv("REPEAT")) e = (t * epj + j) % NE;   // the same experts every job: the oracle for prefetching
                else if (t > 0 && j < epj / 2) e = r.ids[0][j];
                else { do { e = (int) (rng() % NE); bool dup = false; for (int q = 0; q < j; ++q) dup |= r.ids[t][q] == e; if (!dup) break; } while (true); }
                r.ids[t][j] = e; r.wts[t][j] = 0.1f + 0.01f * j;
            }
            memcpy(r.x[t], x.data() + (size_t) t * n, n * sizeof(float));
        }
        const unsigned want = counter * 64u;
        auto t0 = std::chrono::steady_clock::now();
        __atomic_store_n(&r.seq, want, __ATOMIC_RELEASE);
        while (__atomic_load_n(&outs[0].seq, __ATOMIC_ACQUIRE) != want) __builtin_ia32_pause();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (it < 0) continue;
        sum_t += dt;
        std::vector<int> uniq;
        for (int t = 0; t < nt; ++t) for (int j = 0; j < epj; ++j) {
            bool f = false; for (int u : uniq) f |= u == r.ids[t][j];
            if (!f) uniq.push_back(r.ids[t][j]);
        }
        hits += (long long) uniq.size();
        for (int t = 0; t < nt; ++t) for (int i = 0; i < n; ++i) cs += outs[0].y[t][i] * (1 + (i % 7)) * (1 + it % 5);
    }
    cpu.drain();
    const double per = sum_t / jobs, bpj = (double) hits / jobs * (2 * gb + db);
    printf("CPUMOE %s gate %s down %s  %d threads, %d experts x %d tokens: %.1f us/job  %.1f GB/s  checksum %.10e\n",
           p.c_str(), gtype_name(tg.type), gtype_name(td.type), nth, epj, nt, per * 1e6, bpj / per / 1e9, cs);
    return 0;
}
