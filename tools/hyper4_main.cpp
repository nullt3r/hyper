// hyper4 CLI (qwen4exp / Qwen3.8-Flash-Next)
//   hyper4 check <model> <ref.bin> [nt=1] [gpu_frac]   logits vs the llama.cpp reference, nt tokens per forward
//   hyper4 bench <model> <ref.bin> [n_gen=128] [gpu_frac]   greedy decode speed after the reference prompt
#include "engine4.h"
#include "engine5.h"

#include <cuda_profiler_api.h>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#include <algorithm>
#include <type_traits>
#include <vector>

using namespace hyper;

static int g_ref_first = 0;   // v2 references hold logits for positions g_ref_first.. only

static bool read_ref(const char * path, std::vector<int> & toks, std::vector<float> & logits, int & nv) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    int n = 0;
    if (fread(&n, 4, 1, f) != 1) return false;
    const bool v2 = n == 0x32464552;
    if (v2 && fread(&n, 4, 1, f) != 1) return false;
    if (fread(&nv, 4, 1, f) != 1) return false;
    if (v2 && fread(&g_ref_first, 4, 1, f) != 1) return false;
    toks.resize(n);
    if (fread(toks.data(), 4, n, f) != (size_t) n) return false;
    logits.resize((size_t) (n - g_ref_first) * nv);
    if (fread(logits.data(), 4, logits.size(), f) != logits.size()) return false;
    fclose(f);
    return true;
}

struct Cmp {
    int n = 0, top1 = 0, n_nll = 0;
    double kl_sum = 0, kl_max = 0;
    double last = 0;
    double nll_ref = 0, nll_ours = 0;   // next-token negative log-likelihoods (perplexity of the reference text)
    std::vector<double> kls;            // per position (distribution: median / tail)
    void add(const float * r, const std::vector<float> & lg, int nv, int next = -1) {
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
        ++n; top1 += ar == ao; kl_sum += kl; kl_max = std::max(kl_max, kl); last = kl;
        kls.push_back(kl);
        if (next >= 0 && next < nv) {
            nll_ref -= (r[next] - mr) - std::log(zr);
            nll_ours -= (lg[next] - mo) - std::log(zo);
            ++n_nll;
        }
    }
    std::string ppl() const {
        std::string out;
        char b[192];
        if (n_nll) { snprintf(b, sizeof b, "  PPL ours %.4f ref %.4f (%d tokens)", std::exp(nll_ours / n_nll), std::exp(nll_ref / n_nll), n_nll); out += b; }
        if (!kls.empty()) {
            std::vector<double> v = kls;
            std::sort(v.begin(), v.end());
            auto q = [&](double f) { return v[std::min(v.size() - 1, (size_t) (f * v.size()))]; };
            double big = 0, tail = 0;
            for (double x : v) if (x > 0.1) { ++big; tail += x; }
            snprintf(b, sizeof b, "\n  KL median %.5f p90 %.4f p99 %.3f; positions > 0.1: %.1f%% (%.0f%% of the KL sum)", q(0.5), q(0.9), q(0.99),
                     100.0 * big / v.size(), 100.0 * tail / std::max(1e-30, kl_sum));
            out += b;
        }
        return out;
    }
};

static void on_crash(int sig) {
    void * bt[64];
    const int n = backtrace(bt, 64);
    fprintf(stderr, "\n*** signal %d, backtrace:\n", sig);
    backtrace_symbols_fd(bt, n, 2);
    _exit(128 + sig);
}

template <class Engine, class Options>
static int run_cmd(int argc, char ** argv) {
    const std::string cmd = argv[1];
    std::vector<int> toks; std::vector<float> ref; int nv = 0;
    if (!read_ref(argv[3], toks, ref, nv)) return 1;
    Options opt;
    opt.prompt_cache = cmd == "cachetest";
    if (getenv("HYPER4_NOSTREAM")) opt.stream_experts = false;
    if (getenv("HYPER4_STREAM")) opt.stream_experts = true;
    if (getenv("HYPER4_MTP")) opt.mtp_path = getenv("HYPER4_MTP");
    if (getenv("HYPER_DRAFT")) opt.n_draft = atoi(getenv("HYPER_DRAFT"));
    opt.max_pos = getenv("HYPER_MAXPOS") ? atoi(getenv("HYPER_MAXPOS")) : 8192;
    if (argc > 5) opt.gpu_expert_frac = (float) atof(argv[5]);
    if (getenv("HYPER_CPU_THREADS")) opt.cpu_threads = atoi(getenv("HYPER_CPU_THREADS"));
    if (getenv("HYPER_NDEV")) opt.n_devices = atoi(getenv("HYPER_NDEV"));
    try {
        Engine eng(argv[2], opt);
        if (eng.config().n_vocab != nv) { fprintf(stderr, "vocab mismatch %d vs %d\n", eng.config().n_vocab, nv); return 1; }
        if (cmd == "check") {
            const int nt = argc > 4 ? atoi(argv[4]) : 1;
            const int n = (int) toks.size() / nt * nt;
            Cmp cmp;
            std::vector<float> lg;
            eng.reset();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; i += nt) {
                eng.forward(&toks[i], nt, i);
                for (int t = 0; t < nt; ++t) {
                    eng.get_logits(t, lg);
                    if (i < 2) {
                        int nans = 0; double mx = -1e30, mn = 1e30;
                        for (float v : lg) { if (std::isnan(v)) ++nans; else { mx = std::max(mx, (double) v); mn = std::min(mn, (double) v); } }
                        const float * r = &ref[(size_t) (i + t) * nv];
                        double rmx = -1e30, rmn = 1e30;
                        for (int j = 0; j < nv; ++j) { rmx = std::max(rmx, (double) r[j]); rmn = std::min(rmn, (double) r[j]); }
                        fprintf(stderr, "pos %d: nan %d  range [%.2f, %.2f]  ref [%.2f, %.2f]\n", i + t, nans, mn, mx, rmn, rmx);
                    }
                    cmp.add(&ref[(size_t) (i + t) * nv], lg, nv, i + t + 1 < (int) toks.size() ? toks[i + t + 1] : -1);
                    if (i + t < 24 && getenv("HYPER4_PERPOS")) fprintf(stderr, "  pos %3d tok %6d KL %.5f\n", i + t, toks[i + t], cmp.last);
                }
                if (i % 64 == 0) fprintf(stderr, "  %d / %d  KL mean so far %.5f top1 %.1f%%\n", i, n, cmp.kl_sum / cmp.n, 100.0 * cmp.top1 / cmp.n);
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("CHECK4 nt=%d n=%d top1 %.2f%%  KL mean %.6f max %.5f  (%.1f tok/s incl. logits download)%s\n", nt, cmp.n,
                   100.0 * cmp.top1 / cmp.n, cmp.kl_sum / cmp.n, cmp.kl_max, cmp.n / s, cmp.ppl().c_str());
        } else if (cmd == "checkpf") {   // prefill the first n_pf tokens in chunks, then decode the rest token by token
            int n_pf = argc > 4 ? atoi(argv[4]) : 128;
            if (g_ref_first > 0) n_pf = g_ref_first + 1;   // v2 reference: logits from g_ref_first on
            const int n = (int) toks.size();
            Cmp cmp, cmp_pf;
            std::vector<float> lg;
            eng.reset();
            eng.prefill(toks.data(), n_pf, 0);
            eng.get_logits(0, lg);
            cmp_pf.add(&ref[(size_t) (n_pf - 1 - g_ref_first) * nv], lg, nv);
            const int step = getenv("HYPER4_DSTEP") ? atoi(getenv("HYPER4_DSTEP")) : 1;   // tokens per decode forward
            for (int i0 = n_pf; i0 < n; i0 += step) {
                const int nt = std::min(step, n - i0);
                eng.forward(&toks[i0], nt, i0);
                for (int tt = 0; tt < nt; ++tt) {
                const int i = i0 + tt;
                eng.get_logits(tt, lg);
                cmp.add(&ref[(size_t) (i - g_ref_first) * nv], lg, nv);
                if (getenv("HYPER4_PERPOS") && i < n_pf + 24) fprintf(stderr, "  pos %6d (mod4 %d) KL %.5f\n", i, i % 4, cmp.last);
                }
            }
            printf("CHECKPF prefill %d: last-row KL %.5f top1 %d | decoded after it: n=%d top1 %.2f%% KL mean %.6f max %.5f\n", n_pf,
                   cmp_pf.kl_sum, cmp_pf.top1, cmp.n, 100.0 * cmp.top1 / std::max(1, cmp.n), cmp.kl_sum / std::max(1, cmp.n), cmp.kl_max);
        } else if (cmd == "cachetest") {   // prompt cache: cached output == output from scratch (greedy)
            const int n_gen = 64;
            std::map<int, int> cnt;
            for (int i = 0; i < 200; ++i) cnt[toks[i]]++;
            int best = toks[0];
            for (auto & [k, v] : cnt) if (v > cnt[best]) best = k;
            eng.set_snapshot_token(best);
            std::vector<int> A(toks.begin(), toks.begin() + 200);
            auto run = [&](const std::vector<int> & B, const char * name) {
                GenStats st;
                eng.generate(A, n_gen, false, &st);
                const std::vector<int> cached = eng.generate(B, n_gen, false, &st);
                const int reused = st.prompt_reused;
                eng.reset_cache();
                const std::vector<int> fresh = eng.generate(B, n_gen, false, &st);
                int same = 0;
                while (same < n_gen && cached[same] == fresh[same]) ++same;
                printf("CACHE4 %-9s prompt %zu reused %d identical %d / %d\n", name, B.size(), reused, same, n_gen);
            };
            run(A, "same");
            std::vector<int> B(A.begin(), A.begin() + 150);
            B.insert(B.end(), toks.begin() + 210, toks.begin() + 240);
            run(B, "diverged");
            eng.reset_cache();
            GenStats st;
            std::vector<int> outA = eng.generate(A, n_gen, false, &st);
            std::vector<int> C = A;
            C.insert(C.end(), outA.begin(), outA.end());
            C.insert(C.end(), toks.begin() + 200, toks.begin() + 240);
            run(C, "extended");
            if (toks.size() >= 3200) {   // parked conversation: long A, unrelated side request S, then A extended; must match the
                // same continuation without the side request bit for bit (both reuse the KV written while generating A)
                std::vector<int> LA(toks.begin(), toks.begin() + 3000), S(toks.begin() + 3000, toks.begin() + 3100);
                auto cont = [&](bool side, int & reused) {
                    eng.reset_cache();
                    std::vector<int> out = eng.generate(LA, n_gen, false, &st);
                    if (side) eng.generate(S, 16, false, &st);
                    std::vector<int> LB = LA;
                    LB.insert(LB.end(), out.begin(), out.end());
                    LB.insert(LB.end(), toks.begin() + 3100, toks.begin() + 3150);
                    std::vector<int> r = eng.generate(LB, n_gen, false, &st);
                    reused = st.prompt_reused;
                    return r;
                };
                int r0 = 0, r1 = 0;
                const std::vector<int> plain = cont(false, r0), parked = cont(true, r1);
                int same = 0;
                while (same < n_gen && plain[same] == parked[same]) ++same;
                printf("CACHE4 %-9s reused %d (without side request %d)  identical %d / %d\n", "parked", r1, r0, same, n_gen);
            }
        } else if (cmd == "trace") {   // HYPER4_TRACE=dir: tokens 0..n-1 one per forward, then every layer's last-row residual
            if constexpr (std::is_same_v<Engine, Engine4>) {
                const int n = std::min<int>(argc > 4 ? atoi(argv[4]) : (int) toks.size(), (int) toks.size());
                eng.reset();
                for (int i = 0; i < n; ++i) eng.forward(&toks[i], 1, i);
                eng.dump_trace(getenv("HYPER4_TRACE"));
                printf("TRACE %d tokens, residuals in %s\n", n, getenv("HYPER4_TRACE"));
            } else throw std::runtime_error("trace: Flash-Next engine only");
        } else if (cmd == "mtpgen") {   // HYPER4_MTP=file: greedy plain vs MTP speculative (identical output), speed
            const int n_gen = argc > 4 ? atoi(argv[4]) : 256;
            const size_t np = getenv("HYPER4_GENP") ? atoi(getenv("HYPER4_GENP")) : 128;   // prompt length
            std::vector<int> prompt(toks.begin(), toks.begin() + std::min<size_t>(toks.size(), np));
            GenStats a, b;
            const std::vector<int> plain = eng.generate(prompt, n_gen, false, &a);
            {
                const char * sw = getenv("HYPER_SWEEP") ? getenv("HYPER_SWEEP") : getenv("HYPER5_SWEEP");
                if (sw) {   // "k:pmin,k:pmin,...": draft policies after one plain run
                    printf("MTPSWEEP plain %.2f t/s\n", a.tokens / a.seconds);
                    for (const char * q = sw; *q;) {
                        const int k = atoi(q);
                        while (*q && *q != ':') ++q;
                        const double pm = *q ? atof(++q) : 0.0;
                        while (*q && *q != ',') ++q;
                        if (*q) ++q;
                        eng.set_draft(k, pm);
                        GenStats c;
                        const std::vector<int> o = eng.generate(prompt, n_gen, true, &c);
                        int same = 0;
                        while (same < n_gen && plain[same] == o[same]) ++same;
                        uint64_t hs = 1469598103934665603ull;
                        for (int t : o) hs = (hs ^ (uint32_t) t) * 1099511628211ull;
                        printf("MTPSWEEP K=%d pmin %.2f: %.2f t/s (%+.1f %%)  drafted/step %.2f  accepted/step %.2f  tokens/step %.2f  main %.2f ms  "
                               "mtp %.2f ms  identical %d  hash %016llx  prefill %.2f s\n", k, pm, c.tokens / c.seconds,
                               100.0 * (c.tokens / c.seconds) / (a.tokens / a.seconds) - 100.0, (double) c.drafted / std::max(1, c.steps),
                               (double) c.accepted / std::max(1, c.steps), (double) c.tokens / std::max(1, c.steps), 1e3 * c.t_main / std::max(1, c.steps),
                               1e3 * c.t_mtp / std::max(1, c.steps), same, (unsigned long long) hs, c.t_prefill);
                        fflush(stdout);
                    }
                    return 0;
                }
            }
            const std::vector<int> spec = eng.generate(prompt, n_gen, true, &b);
            int same = 0;
            while (same < n_gen && plain[same] == spec[same]) ++same;
            printf("MTPGEN plain %.2f t/s | spec %.2f t/s  steps %d  drafted/step %.2f  accepted/step %.2f  tokens/step %.2f  main %.2f ms  mtp %.2f ms  restore %.2f ms\n",
                   a.tokens / a.seconds, b.tokens / b.seconds, b.steps, (double) b.drafted / std::max(1, b.steps), (double) b.accepted / std::max(1, b.steps),
                   (double) b.tokens / std::max(1, b.steps), 1e3 * b.t_main / std::max(1, b.steps), 1e3 * b.t_mtp / std::max(1, b.steps),
                   1e3 * b.t_restore / std::max(1, b.steps));
            printf("MTPGEN identical prefix %d / %d%s\n", same, n_gen, same == n_gen ? " (sequences match)" : "");
            uint64_t hsh = 1469598103934665603ull;
            for (int t : spec) hsh = (hsh ^ (uint32_t) t) * 1099511628211ull;
            printf("MTPGEN prompt %zu  prefill %.2f s  spec output hash %016llx\n", prompt.size(), b.t_prefill, (unsigned long long) hsh);
        } else if (cmd == "checkbulk") {   // HYPER4_ALLROWS=1: prefill up to the reference's first row, then its rows in one chunk
            const int n = (int) toks.size(), first = g_ref_first;
            Cmp cmp;
            std::vector<float> lg;
            eng.reset();
            if (first > 0) eng.prefill(toks.data(), first, 0);
            for (int c0 = first; c0 < n; c0 += 512) {
                const int nt = std::min(512, n - c0);
                eng.forward(&toks[c0], nt, c0);
                for (int t = 0; t < nt; ++t) { eng.get_logits(t, lg); cmp.add(&ref[(size_t) (c0 + t - first) * nv], lg, nv); }
            }
            printf("CHECKBULK rows %d..%d: n=%d top1 %.2f%% KL mean %.6f max %.5f\n", first, n - 1, cmp.n, 100.0 * cmp.top1 / cmp.n,
                   cmp.kl_sum / cmp.n, cmp.kl_max);
        } else if (cmd == "pfrepeat") {   // determinism: the same prefill R times in one process, a hash of the last row's logits
            const int n = argc > 4 ? atoi(argv[4]) : (int) toks.size();
            const int R = getenv("HYPER4_REPEAT") ? atoi(getenv("HYPER4_REPEAT")) : 6;
            std::vector<float> lg;
            for (int rep = 0; rep < R; ++rep) {
                eng.reset();
                eng.prefill(toks.data(), std::min<int>(n, (int) toks.size()), 0);
                eng.get_logits(0, lg);
                uint64_t h = 1469598103934665603ull;
                for (float v : lg) { uint32_t u; memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
                // then a few decode steps (single rows), hashed as well
                int next = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin()), p = std::min<int>(n, (int) toks.size());
                uint64_t hd = 1469598103934665603ull;
                for (int i = 0; i < 16; ++i) { next = eng.forward(&next, 1, p++)[0]; hd = (hd ^ (uint32_t) next) * 1099511628211ull; }
                printf("PFREPEAT %d: prefill %d, last-row logits hash %016llx, 16 greedy tokens hash %016llx\n", rep, p - 16,
                       (unsigned long long) h, (unsigned long long) hd);
                fflush(stdout);
            }
        } else if (cmd == "pfbench") {   // prompt of n tokens (reference tokens repeated): prefill speed, then 64 decoded tokens
            const int n = argc > 4 ? atoi(argv[4]) : 2048;
            if (getenv("HYPER4_PFREAL") && (int) toks.size() < n) { fprintf(stderr, "reference too short for a real-text prompt\n"); return 1; }
            std::vector<int> prompt(n);
            for (int i = 0; i < n; ++i) prompt[i] = toks[i % toks.size()];   // HYPER4_PFREAL with a long reference: real text
            for (int rep = 0; rep < 2; ++rep) {
                eng.reset();
                auto t0 = std::chrono::steady_clock::now();
                int next = eng.prefill(prompt.data(), n, 0);
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                auto t1 = std::chrono::steady_clock::now();
                int p = n;
                for (int i = 0; i < 64; ++i) next = eng.forward(&next, 1, p++)[0];
                const double s2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                printf("PFBENCH4 n=%d prefill %.1f t/s (%.2f s)  decode after it %.1f t/s\n", n, n / s, s, 64 / s2);
            }
        } else if (cmd == "sums") {   // HYPER4_DEBUG=1 HYPER4_SUMS=1 HYPER_NDEV=1: per-layer tensor sums over the first nt tokens
            const int n = argc > 4 ? atoi(argv[4]) : 3;
            eng.reset();
            for (int i = 0; i < n; ++i) eng.forward(&toks[i], 1, i);
        } else if (cmd == "calib") {   // routing statistics: the reference prompt + n greedy tokens -> argv[6] or expert_stats.bin
            const int n_gen = argc > 4 ? atoi(argv[4]) : 512;
            eng.reset();
            int p = 0, next = 0;
            const int P = (int) toks.size();
            for (; p < P; p += 4) { const int nt = std::min(4, P - p); next = eng.forward(&toks[p], nt, p)[nt - 1]; }
            for (int i = 0; i < n_gen; ++i) next = eng.forward(&next, 1, p++)[0];
            const std::string out = argc > 6 ? argv[6] : "expert_stats.bin";
            eng.save_expert_stats(out);   // (loaded statistics + this run)
            printf("CALIB %d prompt + %d generated tokens -> %s\n", P, n_gen, out.c_str());
        } else if (cmd == "ntbench") {   // decode-forward cost by rows (1..4) after a prefill of HYPER_TF_PF tokens
            const int pf = getenv("HYPER_TF_PF") ? atoi(getenv("HYPER_TF_PF")) : 256;
            eng.reset();
            eng.prefill(toks.data(), pf, 0);
            for (int nt = 1; nt <= 4; ++nt) {
                const int reps = 60;
                double tot = 0;
                for (int r = 0; r < reps; ++r) {
                    const int p0 = pf + (r % 8) * 4;
                    auto t0 = std::chrono::steady_clock::now();
                    eng.forward(&toks[p0], nt, p0);
                    tot += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                }
                printf("NTBENCH nt=%d  %.2f ms per forward  (%.2f ms per row)\n", nt, 1e3 * tot / reps, 1e3 * tot / reps / nt);
            }
        } else if (cmd == "tfbench") {   // decode speed on fixed content: the reference tokens fed one at a time after a 64-token prefill
            const int pf = getenv("HYPER_TF_PF") ? atoi(getenv("HYPER_TF_PF")) : 64;   // prefilled context
            if (getenv("HYPER_TF_REPEAT")) {   // context longer than the reference: its tokens repeated
                const size_t n0 = toks.size(), need = (size_t) pf + (argc > 4 ? atoi(argv[4]) : 256);
                for (size_t i = n0; i < need; ++i) toks.push_back(toks[i % n0]);
            }
            const int n_tok = std::min<int>(argc > 4 ? atoi(argv[4]) : 256, (int) toks.size() - pf);
            eng.reset();
            eng.prefill(toks.data(), pf, 0);
            if (getenv("HYPER_TF_MARK")) cudaProfilerStart();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = pf; i < pf + n_tok; ++i) eng.forward(&toks[i], 1, i);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("TFBENCH decode %d fixed tokens: %.2f t/s (%.2f ms/token)\n", n_tok, n_tok / s, 1e3 * s / n_tok);
        } else if (cmd == "bench") {
            const int n_gen = argc > 4 ? atoi(argv[4]) : 128;
            eng.reset();
            int p = 0;
            const int P = std::min<int>((int) toks.size(), 64);
            int next = 0;
            for (; p < P; p += 4) { const int nt = std::min(4, P - p); next = eng.forward(&toks[p], nt, p)[nt - 1]; }
            p = P;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n_gen; ++i) next = eng.forward(&next, 1, p++)[0];
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("BENCH4 decode %d tokens: %.2f t/s (%.2f ms/token)\n", n_gen, n_gen / s, 1e3 * s / n_gen);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    signal(SIGSEGV, on_crash);
    signal(SIGABRT, on_crash);
    signal(SIGBUS, on_crash);
    if (argc < 4) { fprintf(stderr, "usage: hyper4 check|bench <model> <ref.bin> [nt|n_gen] [gpu_frac]\n"); return 1; }
    std::string arch;
    try { arch = GGUF(argv[2]).arch(); } catch (const std::exception & e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
    if (arch == "glm5-next") return run_cmd<Engine5, Engine5Options>(argc, argv);
    return run_cmd<Engine4, Engine4Options>(argc, argv);
}
