// deepseek41 engine driver (tests): prompt token ids from a file (comma- or space-separated)
//   hyper7 <model.gguf> run <ids> [n_gen]         prefill in chunks, top-5 at the last position, greedy continuation
//   hyper7 <model.gguf> dump <ids> <out> [split]  logits of every position from `split` on (default 0) as float32 [rows][vocab]
//                                                 after a header {rows, vocab, first}: positions < split go through the
//                                                 chunked prefill path, the rest in decode steps of up to 4 rows
#include "engine7.h"
#include "cpu_moe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>

using namespace hyper;

static std::vector<int> read_ids(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    for (char & ch : s) if (ch == ',' || ch == '\n') ch = ' ';
    std::istringstream is(s);
    std::vector<int> v;
    int x;
    while (is >> x) v.push_back(x);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: hyper7 <model.gguf> run|dump <ids> ...\n"); return 1; }
    try {
        Engine7Options o;
        o.max_pos = getenv("HYPER7_CTX") ? atoi(getenv("HYPER7_CTX")) : 8192;
        if (getenv("HYPER7_THREADS")) o.cpu_threads = atoi(getenv("HYPER7_THREADS"));
        Engine7 eng(argv[1], o);
        const std::string cmd = argv[2];
        const std::vector<int> ids = read_ids(argv[3]);
        const int V = eng.config().n_vocab, C = eng.chunk();
        using clk = std::chrono::steady_clock;
        if (cmd == "run") {   // generate(): pipelined prefill, greedy decode
            const int n_gen = argc > 4 ? atoi(argv[4]) : 16;
            GenStats st;
            SamplingParams sp;   // HYPER7_TEMP: full-vocabulary sampling at that temperature (seed HYPER7_SEED, default 1234)
            if (getenv("HYPER7_TEMP")) { sp.temp = (float) atof(getenv("HYPER7_TEMP")); sp.seed = getenv("HYPER7_SEED") ? strtoull(getenv("HYPER7_SEED"), nullptr, 10) : 1234; }
            const std::vector<int> out = eng.generate(ids, n_gen, getenv("HYPER7_SPEC") != nullptr, &st, {}, sp);
            printf("prefill %zu tokens: %.2f s (%.1f t/s)\ngen:", ids.size(), st.t_prefill, ids.size() / st.t_prefill);
            for (int t : out) printf(" %d", t);
            printf("\ndecode %d tokens: %.2f s (%.2f t/s), %d steps, %d/%d drafts accepted\n", st.tokens, st.seconds, st.tokens / st.seconds, st.steps, st.accepted, st.drafted);
            if (getenv("HYPER7_SAVE_STATS")) eng.save_expert_stats(getenv("HYPER7_SAVE_STATS"));
        } else if (cmd == "specstat") {   // sampling test: N draws of the first 2 tokens after the prompt at HYPER7_TEMP (seed = draw),
            // the (tok0, tok1) pair counts; the cache keeps the prompt, so a draw costs one or two decode steps. Compared
            // between HYPER7_DSPARK_COUPLED=1 (exact-match drafts: plain sampling) and speculative sampling (same distribution)
            const int N = argc > 4 ? atoi(argv[4]) : 500;
            SamplingParams sp;
            sp.temp = getenv("HYPER7_TEMP") ? (float) atof(getenv("HYPER7_TEMP")) : 1.0f;
            std::map<std::pair<int, int>, int> cnt;
            long acc = 0, drafted = 0;
            for (int i = 1; i <= N; ++i) {
                sp.seed = (uint64_t) i * 7919u + 17;
                GenStats st;
                const std::vector<int> out = eng.generate(ids, 2, true, &st, {}, sp);
                if (out.size() >= 2) cnt[{out[0], out[1]}]++;
                acc += st.accepted; drafted += st.drafted;
            }
            printf("specstat %d draws, drafts accepted %ld / %ld\n", N, acc, drafted);
            for (auto & kv : cnt) printf("pair %d %d %d\n", kv.first.first, kv.first.second, kv.second);
        } else if (cmd == "ab") {   // CPU expert scheduling A/B in one process: HYPER7_AB modes (0 dynamic, 1 L3 groups,
            // 2 groups + prefetch of the previous step's experts, 3 groups + prefetch of the GPU's prediction: HYPER7_PRED=1),
            // interleaved over rounds; greedy outputs must match
            const int n_gen = argc > 4 ? atoi(argv[4]) : 256, rounds = argc > 5 ? atoi(argv[5]) : 3;
            std::vector<int> modes;
            {   std::string m = getenv("HYPER7_AB") ? getenv("HYPER7_AB") : "0,1,2";
                for (char & ch : m) if (ch == ',') ch = ' ';
                std::istringstream is(m); int x; while (is >> x) modes.push_back(x); }
            std::vector<std::vector<double>> tps(modes.size());
            std::vector<int> ref;
            const bool spec = getenv("HYPER7_SPEC") != nullptr;
            for (int r = 0; r < rounds; ++r)
                for (size_t mi = 0; mi < modes.size(); ++mi) {
                    eng.cpu_moe()->set_sched(modes[mi] >= 1, modes[mi] >= 2 ? modes[mi] - 1 : 0);
                    GenStats st;
                    const std::vector<int> out = eng.generate(ids, n_gen, spec, &st);
                    if (ref.empty()) ref = out;
                    tps[mi].push_back(st.tokens / st.seconds);
                    printf("round %d mode %d: %.2f t/s, %d steps%s\n", r, modes[mi], st.tokens / st.seconds, st.steps, out == ref ? "" : "  OUTPUT DIFFERS");
                    fflush(stdout);
                }
            for (size_t mi = 0; mi < modes.size(); ++mi) {
                std::vector<double> v = tps[mi];
                std::sort(v.begin(), v.end());
                printf("mode %d: median %.2f t/s (min %.2f max %.2f)\n", modes[mi], v[v.size() / 2], v.front(), v.back());
            }
        } else if (cmd == "dump") {
            if (argc < 5) throw std::runtime_error("dump: output path missing");
            const int split = argc > 5 ? atoi(argv[5]) : 0, T = (int) ids.size();
            FILE * f = fopen(argv[4], "wb");
            if (!f) throw std::runtime_error("cannot write output");
            const int hdr[3] = {T - split, V, split};
            fwrite(hdr, 4, 3, f);
            for (int c0 = 0; c0 < split; c0 += C) eng.forward(&ids[c0], std::min(C, split - c0), c0, 1);
            std::vector<float> lg;
            for (int p = split; p < T; p += MAX_NT) {
                const int nt = std::min(MAX_NT, T - p);
                eng.forward(&ids[p], nt, p, nt);
                for (int r = 0; r < nt; ++r) { eng.get_logits(r, lg); fwrite(lg.data(), 4, V, f); }
            }
            fclose(f);
            printf("dumped %d rows\n", T - split);
        } else throw std::runtime_error("unknown command " + cmd);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
