// hyper CLI
//   hyper check  <model> <ref.bin>              logits vs llama.cpp reference, one token per forward
//   hyper check2 <model> <ref.bin>              same, two tokens per forward (speculative verification path)
//   hyper gen    <model> <ref.bin> [n_prompt] [n_gen]   greedy: plain vs MTP speculative (must match), speed
#include "engine.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace hyper;

static bool read_ref(const char * path, std::vector<int> & toks, std::vector<float> & logits, int & nv) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    int n = 0;
    if (fread(&n, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) return false;
    toks.resize(n);
    if (fread(toks.data(), 4, n, f) != (size_t) n) return false;
    logits.resize((size_t) n * nv);
    if (fread(logits.data(), 4, logits.size(), f) != logits.size()) return false;
    fclose(f);
    return true;
}

struct Cmp {
    int n = 0, top1 = 0;
    double kl_sum = 0, kl_max = 0;
    void add(const float * r, const std::vector<float> & lg, int nv) {
        double mr = -1e30, mo = -1e30; int ar = 0, ao = 0;
        for (int j = 0; j < nv; ++j) {
            if (r[j] > mr) { mr = r[j]; ar = j; }
            if (lg[j] > mo) { mo = lg[j]; ao = j; }
        }
        double zr = 0, zo = 0;
        for (int j = 0; j < nv; ++j) { zr += std::exp(r[j] - mr); zo += std::exp(lg[j] - mo); }
        double kl = 0;
        for (int j = 0; j < nv; ++j) {
            const double pr = std::exp(r[j] - mr) / zr;
            if (pr < 1e-12) continue;
            kl += pr * (((r[j] - mr) - std::log(zr)) - ((lg[j] - mo) - std::log(zo)));
        }
        ++n; top1 += ar == ao; kl_sum += kl; kl_max = std::max(kl_max, kl);
    }
    void print(const char * tag) const {
        printf("%s n=%d top1 %.2f%%  KL mean %.6f max %.5f\n", tag, n, 100.0 * top1 / n, kl_sum / n, kl_max);
    }
};

static int cmd_check(const char * model, const char * ref_path, int nt) {
    std::vector<int> toks; std::vector<float> ref; int nv = 0;
    if (!read_ref(ref_path, toks, ref, nv)) return 1;
    const int n = (int) toks.size() / nt * nt;
    EngineOptions opt;
    opt.max_pos = n + 16;
    opt.mtp = nt > 1;
    Engine eng(model, opt);
    if (eng.config().n_vocab != nv) { fprintf(stderr, "vocab mismatch\n"); return 1; }
    Cmp cmp;
    std::vector<float> lg;
    for (int i = 0; i < n; i += nt) {
        eng.forward(&toks[i], nt, i);
        for (int t = 0; t < nt; ++t) {
            eng.get_logits(t, lg);
            cmp.add(&ref[(size_t) (i + t) * nv], lg, nv);
        }
    }
    cmp.print(nt == 1 ? "CHECK" : "CHECK2");
    return 0;
}

static int cmd_gen(const char * model, const char * ref_path, int n_prompt, int n_gen) {
    std::vector<int> toks; std::vector<float> ref; int nv = 0;
    if (!read_ref(ref_path, toks, ref, nv)) return 1;
    if ((int) toks.size() > n_prompt) toks.resize(n_prompt);
    EngineOptions opt;
    opt.max_pos = (int) toks.size() + n_gen + 16;
    Engine eng(model, opt);
    GenStats a, b;
    const std::vector<int> plain = eng.generate(toks, n_gen, false, &a);
    const std::vector<int> spec = eng.generate(toks, n_gen, true, &b);
    int same = 0;
    while (same < n_gen && plain[same] == spec[same]) ++same;
    printf("GEN plain: %d tokens %.2f t/s\n", a.tokens, a.tokens / a.seconds);
    printf("GEN spec : %d tokens %.2f t/s  steps %d  draft acceptance %.1f%%  tokens/step %.2f\n", b.tokens, b.tokens / b.seconds,
           b.steps, 100.0 * b.accepted / b.steps, (double) b.tokens / b.steps);
    printf("GEN identical prefix %d / %d%s\n", same, n_gen, same == n_gen ? " (sequences match)" : "");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: hyper check|check2 <model> <ref.bin> | hyper gen <model> <ref.bin> [n_prompt] [n_gen]\n");
        return 1;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "check") return cmd_check(argv[2], argv[3], 1);
        if (cmd == "check2") return cmd_check(argv[2], argv[3], 2);
        if (cmd == "gen") return cmd_gen(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 64, argc > 5 ? atoi(argv[5]) : 256);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "unknown command\n");
    return 1;
}
