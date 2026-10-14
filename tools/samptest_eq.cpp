// The full-row sampling path against a reference copy of the previous implementation (sampling_old_ns.h: src/sampling.h of a
// given commit with its namespace renamed hyper_old): kept tokens, probability bits and Gumbel draws must be the same.
// build: git show <commit>:src/sampling.h | sed -e 's/^namespace hyper {/namespace hyper_old {\nusing hyper::SamplingParams;/' \
//          -e 's/#pragma once//' -e 's/return gumbel_pick(sampling_set(/return hyper_old::gumbel_pick(hyper_old::sampling_set(/' > sampling_old_ns.h
//        g++ -O2 -std=c++17 -Isrc tools/samptest_eq.cpp -o samptest_eq
#include "sampling.h"
#include "sampling_old_ns.h"
#include <chrono>
#include <cstring>
#include <cstdio>
#include <random>
using clk = std::chrono::steady_clock;
int main() {
    std::mt19937 rng(7);
    int mism = 0, tests = 0; double t_old = 0, t_new = 0;
    for (int V : {248320, 151552}) for (double zs : {0.6, 1.0, 1.5}) for (int topk : {0, 100}) for (float topp : {0.8f, 0.95f, 1.0f})
    for (float minp : {0.0f, 0.05f}) for (float temp : {0.6f, 1.0f}) {
        std::vector<int> perm(V); for (int i = 0; i < V; ++i) perm[i] = i; std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<float> lg(V);
        for (int r = 0; r < V; ++r) lg[perm[r]] = (float) (-zs * std::log(r + 1.0)) + 20.0f + 1e-3f * (rng() % 1000);
        for (int q = 0; q < 200; ++q) lg[rng() % V] = lg[rng() % V];   // ties
        hyper::SamplingParams sp; sp.temp = temp; sp.top_p = topp; sp.top_k = topk; sp.min_p = minp;
        auto stats = [&](double, float & M, double & Z) { M = 1e30f; Z = 1.0; };   // forces the full-row path
        std::vector<std::pair<float, int>> c1 = {{lg[0], 0}}, c2 = c1;
        auto t0 = clk::now();
        hyper_old::KeptSet a = hyper_old::sampling_set(c1, 1, sp, stats, [&](std::vector<float> & o) { o = lg; });
        auto t1 = clk::now();
        hyper::KeptSet b = hyper::sampling_set(c2, 1, sp, stats, [&](std::vector<float> & o) { o = lg; });
        auto t2 = clk::now();
        t_old += std::chrono::duration<double, std::milli>(t1 - t0).count(); t_new += std::chrono::duration<double, std::milli>(t2 - t1).count();
        ++tests;
        bool same = a.tok == b.tok && a.pr.size() == b.pr.size() && a.full == b.full;
        for (size_t i = 0; same && i < a.pr.size(); ++i) same = memcmp(&a.pr[i], &b.pr[i], 8) == 0;
        for (int s = 0; same && s < 20; ++s) same = hyper_old::gumbel_pick(a, 5 + s, 100 + s) == hyper::gumbel_pick(b, 5 + s, 100 + s);
        if (!same) { ++mism; printf("MISMATCH V %d zipf %.1f topk %d topp %.2f minp %.2f T %.1f kept %zu vs %zu\n", V, zs, topk, topp, minp, temp, a.tok.size(), b.tok.size()); }
    }
    printf("%d tests, %d mismatches; full-row path old %.2f ms, new %.2f ms per row\n", tests, mism, t_old / tests, t_new / tests);
}
