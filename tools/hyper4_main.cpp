// hyper4 CLI (qwen4exp / Qwen3.8-Flash-Next)
//   hyper4 check <model> <ref.bin> [nt=1] [gpu_frac]   logits vs the llama.cpp reference, nt tokens per forward
//   hyper4 bench <model> <ref.bin> [n_gen=128] [gpu_frac]   greedy decode speed after the reference prompt
#include "engine4.h"

#include <csignal>
#include <execinfo.h>
#include <unistd.h>

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
    double last = 0;
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
        ++n; top1 += ar == ao; kl_sum += kl; kl_max = std::max(kl_max, kl); last = kl;
    }
};

static void on_crash(int sig) {
    void * bt[64];
    const int n = backtrace(bt, 64);
    fprintf(stderr, "\n*** signal %d, backtrace:\n", sig);
    backtrace_symbols_fd(bt, n, 2);
    _exit(128 + sig);
}

int main(int argc, char ** argv) {
    signal(SIGSEGV, on_crash);
    signal(SIGABRT, on_crash);
    signal(SIGBUS, on_crash);
    if (argc < 4) { fprintf(stderr, "usage: hyper4 check|bench <model> <ref.bin> [nt|n_gen] [gpu_frac]\n"); return 1; }
    const std::string cmd = argv[1];
    std::vector<int> toks; std::vector<float> ref; int nv = 0;
    if (!read_ref(argv[3], toks, ref, nv)) return 1;
    Engine4Options opt;
    opt.max_pos = 8192;
    if (argc > 5) opt.gpu_expert_frac = (float) atof(argv[5]);
    if (getenv("HYPER_CPU_THREADS")) opt.cpu_threads = atoi(getenv("HYPER_CPU_THREADS"));
    if (getenv("HYPER_NDEV")) opt.n_devices = atoi(getenv("HYPER_NDEV"));
    try {
        Engine4 eng(argv[2], opt);
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
                    cmp.add(&ref[(size_t) (i + t) * nv], lg, nv);
                    if (i + t < 24 && getenv("HYPER4_PERPOS")) fprintf(stderr, "  pos %3d tok %6d KL %.5f\n", i + t, toks[i + t], cmp.last);
                }
                if (i % 64 == 0) fprintf(stderr, "  %d / %d  KL mean so far %.5f top1 %.1f%%\n", i, n, cmp.kl_sum / cmp.n, 100.0 * cmp.top1 / cmp.n);
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("CHECK4 nt=%d n=%d top1 %.2f%%  KL mean %.6f max %.5f  (%.1f tok/s incl. logits download)\n", nt, cmp.n,
                   100.0 * cmp.top1 / cmp.n, cmp.kl_sum / cmp.n, cmp.kl_max, cmp.n / s);
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
            eng.save_expert_stats(out);
            printf("CALIB %d prompt + %d generated tokens -> %s\n", P, n_gen, out.c_str());
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
