// hyper CLI
//   hyper check <model.gguf> <ref.bin>           compare logits against the llama.cpp reference dump
//   hyper bench <model.gguf> [n_prompt] [n_gen]   greedy generation speed
#include "engine.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace hyper;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static int cmd_check(const char * model, const char * ref_path) {
    FILE * f = fopen(ref_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", ref_path); return 1; }
    int n = 0, nv = 0;
    if (fread(&n, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) return 1;
    std::vector<int> toks(n);
    if (fread(toks.data(), 4, n, f) != (size_t) n) return 1;
    std::vector<float> ref((size_t) n * nv);
    if (fread(ref.data(), 4, ref.size(), f) != ref.size()) return 1;
    fclose(f);

    EngineOptions opt;
    opt.max_pos = n + 16;
    Engine eng(model, opt);
    if (eng.config().n_vocab != nv) { fprintf(stderr, "vocab mismatch %d vs %d\n", eng.config().n_vocab, nv); return 1; }

    std::vector<float> lg;
    int top1 = 0;
    double kl_sum = 0, kl_max = 0, maxdiff = 0;
    for (int i = 0; i < n; ++i) {
        eng.decode(toks[i], i);
        eng.get_logits(lg);
        const float * r = &ref[(size_t) i * nv];
        // softmax of both, KL(ref || ours)
        double mr = -1e30, mo = -1e30;
        int ar = 0, ao = 0;
        for (int j = 0; j < nv; ++j) {
            if (r[j] > mr) { mr = r[j]; ar = j; }
            if (lg[j] > mo) { mo = lg[j]; ao = j; }
            maxdiff = std::max(maxdiff, (double) std::fabs(r[j] - lg[j]));
        }
        double zr = 0, zo = 0;
        for (int j = 0; j < nv; ++j) { zr += std::exp(r[j] - mr); zo += std::exp(lg[j] - mo); }
        double kl = 0;
        for (int j = 0; j < nv; ++j) {
            const double pr = std::exp(r[j] - mr) / zr;
            if (pr < 1e-12) continue;
            const double lpr = (r[j] - mr) - std::log(zr), lpo = (lg[j] - mo) - std::log(zo);
            kl += pr * (lpr - lpo);
        }
        top1 += ar == ao;
        kl_sum += kl; kl_max = std::max(kl_max, kl);
        if (i < 4 || i == n - 1 || ar != ao)
            fprintf(stderr, "pos %4d tok %6d  ref top %6d ours %6d  KL %.5f\n", i, toks[i], ar, ao, kl);
    }
    printf("CHECK n=%d top1 %.2f%%  KL mean %.6f max %.5f  max|dlogit| %.3f\n", n, 100.0 * top1 / n, kl_sum / n, kl_max, maxdiff);
    return 0;
}

static int cmd_bench(const char * model, int n_prompt, int n_gen) {
    EngineOptions opt;
    opt.max_pos = n_prompt + n_gen + 16;
    Engine eng(model, opt);
    auto t0 = clk::now();
    for (int i = 0; i < n_prompt; ++i) eng.decode(1000 + (i * 7919) % 50000, i);
    int tok = eng.argmax_last();
    auto t1 = clk::now();
    for (int i = 0; i < n_gen; ++i) {
        eng.decode(tok, n_prompt + i);
        tok = eng.argmax_last();
    }
    auto t2 = clk::now();
    printf("BENCH prompt %d tokens in %.2f s (%.1f t/s, token-by-token)  gen %d tokens: %.2f t/s (%.2f ms/token)\n",
           n_prompt, secs(t0, t1), n_prompt / secs(t0, t1), n_gen, n_gen / secs(t1, t2), 1000.0 * secs(t1, t2) / n_gen);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: hyper check <model> <ref.bin> | hyper bench <model> [n_prompt] [n_gen]\n");
        return 1;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "check" && argc >= 4) return cmd_check(argv[2], argv[3]);
        if (cmd == "bench") return cmd_bench(argv[2], argc > 3 ? atoi(argv[3]) : 32, argc > 4 ? atoi(argv[4]) : 128);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "unknown command\n");
    return 1;
}
