// deepseek41 engine driver (tests): prompt token ids from a file (comma- or space-separated)
//   hyper7 <model.gguf> run <ids> [n_gen]         prefill in chunks, top-5 at the last position, greedy continuation
//   hyper7 <model.gguf> dump <ids> <out> [split]  logits of every position from `split` on (default 0) as float32 [rows][vocab]
//                                                 after a header {rows, vocab, first}: positions < split go through the
//                                                 chunked prefill path, the rest in decode steps of up to 4 rows
#include "engine7.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
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
            const std::vector<int> out = eng.generate(ids, n_gen, getenv("HYPER7_SPEC") != nullptr, &st);
            printf("prefill %zu tokens: %.2f s (%.1f t/s)\ngen:", ids.size(), st.t_prefill, ids.size() / st.t_prefill);
            for (int t : out) printf(" %d", t);
            printf("\ndecode %d tokens: %.2f s (%.2f t/s), %d steps, %d/%d drafts accepted\n", st.tokens, st.seconds, st.tokens / st.seconds, st.steps, st.accepted, st.drafted);
            if (getenv("HYPER7_SAVE_STATS")) eng.save_expert_stats(getenv("HYPER7_SAVE_STATS"));
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
