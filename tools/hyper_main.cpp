// hyper CLI
//   hyper check  <model> <ref.bin>              logits vs llama.cpp reference, one token per forward
//   hyper check2 <model> <ref.bin>              same, two tokens per forward (speculative verification path)
//   hyper checkn <model> <ref.bin> [n]          same, n tokens per forward (n > 4: prefill GEMM path)
//   hyper gen    <model> <ref.bin> [n_prompt] [n_gen]   greedy: plain vs MTP speculative (must match), speed
#include "engine.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
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
    opt.mtp = nt > 1 && nt <= 4;
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
    char tag[32];
    snprintf(tag, sizeof tag, nt == 1 ? "CHECK" : "CHECK nt=%d", nt);
    cmp.print(tag);
    return 0;
}

static int cmd_gen(const char * model, const char * ref_path, int n_prompt, int n_gen) {
    std::vector<int> toks; std::vector<float> ref; int nv = 0;
    if (!read_ref(ref_path, toks, ref, nv)) return 1;
    if ((int) toks.size() > n_prompt) toks.resize(n_prompt);
    EngineOptions opt;
    opt.max_pos = (int) toks.size() + n_gen + 16;
    if (getenv("HYPER_DRAFT")) opt.n_draft = atoi(getenv("HYPER_DRAFT"));
    Engine eng(model, opt);
    GenStats a, b;
    const std::vector<int> plain = eng.generate(toks, n_gen, false, &a);
    const std::vector<int> spec = eng.generate(toks, n_gen, true, &b);
    int same = 0;
    while (same < n_gen && plain[same] == spec[same]) ++same;
    printf("GEN plain: %d tokens %.2f t/s\n", a.tokens, a.tokens / a.seconds);
    printf("GEN spec (%d drafts): %d tokens %.2f t/s  steps %d  accepted drafts/step %.2f  tokens/step %.2f\n", opt.n_draft,
           b.tokens, b.tokens / b.seconds, b.steps, (double) b.accepted / b.steps, (double) b.tokens / b.steps);
    printf("GEN spec per step: main %.2f ms  mtp %.2f ms  restore %.3f ms\n", 1e3 * b.t_main / b.steps,
           1e3 * b.t_mtp / b.steps, 1e3 * b.t_restore / b.steps);
    printf("GEN plain per token: %.2f ms\n", 1e3 * a.seconds / a.tokens);
    printf("GEN identical prefix %d / %d%s\n", same, n_gen, same == n_gen ? " (sequences match)" : "");
    return 0;
}

// prefill speed: the reference tokens repeated to n_prompt, then a few decoded tokens
static int cmd_pfbench(const char * model, const char * ref_path, int n_prompt) {
    std::vector<int> toks; std::vector<float> ref; int nv = 0;
    if (!read_ref(ref_path, toks, ref, nv)) return 1;
    std::vector<int> prompt(n_prompt);
    for (int i = 0; i < n_prompt; ++i) prompt[i] = toks[i % toks.size()];
    EngineOptions opt;
    opt.max_pos = n_prompt + 64;
    Engine eng(model, opt);
    for (int rep = 0; rep < 2; ++rep) {
        GenStats a, b;
        eng.generate(prompt, 16, false, &a);
        eng.generate(prompt, 16, true, &b);
        printf("PFBENCH n=%d  prefill %.1f t/s (%.3f s)  prefill+MTP %.1f t/s (%.3f s)  decode %.1f t/s\n", n_prompt,
               n_prompt / a.t_prefill, a.t_prefill, n_prompt / b.t_prefill, b.t_prefill, a.tokens / a.seconds);
    }
    return 0;
}

// prompt cache: output with the cache must match the output computed from scratch (greedy, MTP)
static int cmd_cachetest(const char * model, const char * ref_path) {
    std::vector<int> toks;   // (tokens only: the reference's logits may cover just its tail)
    {
        FILE * f = fopen(ref_path, "rb");
        int n = 0, nv = 0, first = 0;   // v1: n, nv, tokens; v2 ("REF2"): magic, n, nv, first logit row, tokens
        bool ok = f && fread(&n, 4, 1, f) == 1;
        if (ok && n == 0x32464552) ok = fread(&n, 4, 1, f) == 1 && fread(&nv, 4, 1, f) == 1 && fread(&first, 4, 1, f) == 1;
        else if (ok) ok = fread(&nv, 4, 1, f) == 1;
        if (!ok) { fprintf(stderr, "cannot read %s\n", ref_path); return 1; }
        toks.resize(n);
        ok = fread(toks.data(), 4, n, f) == (size_t) n;
        fclose(f);
        if (!ok) { fprintf(stderr, "cannot read %s\n", ref_path); return 1; }
    }
    EngineOptions opt;
    opt.max_pos = 4096;
    opt.n_draft = 3;
    opt.prompt_cache = true;
    Engine eng(model, opt);
    {   // the most frequent token of the prompt stands in for message starts
        std::map<int, int> cnt;
        for (int i = 0; i < 600; ++i) cnt[toks[i]]++;
        int best = toks[0];
        for (auto & [k, v] : cnt) if (v > cnt[best]) best = k;
        eng.set_snapshot_token(best);
        printf("CACHE snapshot token %d occurs %d times\n", best, cnt[best]);
    }
    const int n_gen = 96;
    std::vector<int> A(toks.begin(), toks.begin() + 600);
    auto run = [&](const std::vector<int> & prev, const std::vector<int> & B, const char * name) {
        GenStats st;
        eng.clear_cache();
        if (!prev.empty()) eng.generate(prev, n_gen, true, &st);
        const std::vector<int> cached = eng.generate(B, n_gen, true, &st);
        const int reused = st.prompt_reused;
        const double tp = st.t_prefill;
        eng.clear_cache();
        const std::vector<int> fresh = eng.generate(B, n_gen, true, &st);
        int same = 0;
        while (same < n_gen && cached[same] == fresh[same]) ++same;
        printf("CACHE %-12s prompt %zu reused %d (prefill %.3f s vs %.3f s fresh)  identical %d / %d  snapshots %d\n", name,
               B.size(), reused, tp, st.t_prefill, same, n_gen, eng.n_snapshots());
        return cached;
    };
    run(A, A, "same");
    std::vector<int> B(A.begin(), A.begin() + 450);
    B.insert(B.end(), toks.begin() + 20, toks.begin() + 90);
    run(A, B, "diverged");
    GenStats st;
    eng.clear_cache();
    std::vector<int> outA = eng.generate(A, n_gen, true, &st);
    std::vector<int> C = A;
    C.insert(C.end(), outA.begin(), outA.end());
    C.insert(C.end(), toks.begin() + 600, toks.begin() + 680);
    run(A, C, "extended");
    if (toks.size() >= 3200) {   // parked conversation: long A, unrelated side request S, then A extended; must match the
        // same continuation without the side request bit for bit (both reuse the KV written while generating A)
        std::vector<int> LA(toks.begin(), toks.begin() + 3000), S(toks.begin() + 3000, toks.begin() + 3100);
        auto cont = [&](bool side, int & reused) {
            eng.clear_cache();
            std::vector<int> out = eng.generate(LA, n_gen, true, &st);
            if (side) eng.generate(S, 16, true, &st);
            std::vector<int> LB = LA;
            LB.insert(LB.end(), out.begin(), out.end());
            LB.insert(LB.end(), toks.begin() + 3100, toks.begin() + 3150);
            std::vector<int> r = eng.generate(LB, n_gen, true, &st);
            reused = st.prompt_reused;
            return r;
        };
        int r0 = 0, r1 = 0;
        const std::vector<int> plain = cont(false, r0), parked = cont(true, r1);
        int same = 0;
        while (same < n_gen && plain[same] == parked[same]) ++same;
        printf("CACHE %-12s reused %d (without side request %d)  identical %d / %d\n", "parked", r1, r0, same, n_gen);
    }
    return 0;
}

// forward latency per chunk size
static int cmd_chunkbench(const char * model, const char * ref_path) {
    std::vector<int> toks; std::vector<float> ref; int nv = 0;
    if (!read_ref(ref_path, toks, ref, nv)) return 1;
    EngineOptions opt;
    opt.max_pos = 4096;
    opt.n_draft = 3;
    Engine eng(model, opt);
    for (int n : {1, 4, 5, 8, 16, 32, 64, 128, 256, 512}) {
        eng.forward(toks.data(), n, 0);
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 5; ++r) eng.forward(toks.data(), n, 0);
        const double ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 5;
        printf("CHUNK n=%3d  %.1f ms  (%.0f t/s)\n", n, ms, n / ms * 1e3);
    }
    return 0;
}

// sampling with speculative decoding must match plain sampling in distribution: histogram of tokens 1..3
static int cmd_samptest(const char * model, const char * ref_path, int runs) {
    std::vector<int> toks; std::vector<float> ref; int nv = 0;
    if (!read_ref(ref_path, toks, ref, nv)) return 1;
    std::vector<int> prompt(toks.begin(), toks.begin() + std::min<size_t>(40, toks.size()));
    EngineOptions opt;
    opt.max_pos = 256;
    opt.n_draft = 3;
    Engine eng(model, opt);
    SamplingParams sp;
    sp.temp = 1.0f; sp.top_k = 40;
    std::vector<std::map<int, int>> hp(4), hs(4);
    double acc = 0; int steps = 0;
    for (int r = 0; r < runs; ++r) {
        GenStats st;
        sp.seed = 1000 + r;
        auto a = eng.generate(prompt, 4, false, &st, {}, sp);
        sp.seed = 900000 + r;
        auto b = eng.generate(prompt, 4, true, &st, {}, sp);
        acc += st.accepted; steps += st.steps;
        for (int i = 0; i < 4; ++i) { hp[i][a[i]]++; hs[i][b[i]]++; }
    }
    for (int i = 0; i < 4; ++i) {
        // total variation distance and the same for two plain halves would need more runs; report TV and top tokens
        std::map<int, int> all = hp[i];
        for (auto & [k, v] : hs[i]) all[k] += 0;
        double tv = 0;
        for (auto & [k, v] : all) tv += std::fabs(hp[i][k] - hs[i][k]) / (double) runs;
        int top = -1, topc = 0;
        for (auto & [k, v] : hp[i]) if (v > topc) { top = k; topc = v; }
        printf("SAMP pos %d: TV %.3f  distinct plain %zu spec %zu  top token %d plain %d spec %d\n", i, tv / 2, hp[i].size(),
               hs[i].size(), top, topc, hs[i][top]);
    }
    printf("SAMP accepted drafts/step %.2f\n", acc / steps);
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
        if (cmd == "samptest") return cmd_samptest(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 300);
        if (cmd == "chunkbench") return cmd_chunkbench(argv[2], argv[3]);
        if (cmd == "cachetest") return cmd_cachetest(argv[2], argv[3]);
        if (cmd == "pfbench") return cmd_pfbench(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 2048);
        if (cmd == "checkn") return cmd_check(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 64);
        if (cmd == "gen") return cmd_gen(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 64, argc > 5 ? atoi(argv[5]) : 256);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "unknown command\n");
    return 1;
}
