// Exact sampling from the GPUs' per-slice top candidates (all engines). Semantics as HF / vLLM: temperature first, then
// top_k (0: off), min_p relative to the best token, top_p over the probabilities normalized over the top_k set.
// The candidates decide alone whenever the kept set cannot reach past them; otherwise (top_k off or above the candidate
// count, with a nucleus / min_p set wider than the candidates) the whole logit row is read and the set found exactly.
#pragma once
#include "llm.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <utility>
#include <vector>

namespace hyper {

// the tokens a draw may return and their probabilities (unnormalized after the cuts: a draw is proportional to them)
struct KeptSet {
    std::vector<int> tok;
    std::vector<double> pr;
    bool full = false;   // (the whole row was read)
};

// cand: (logit, token) pairs from every vocab slice (any order); the first ntop of the sorted list are the exact global top.
// stats(invT, M, Z): max logit M of the whole row and Z = sum exp((x - M) * invT). full(row): the whole logit row.
// sp.temp > 0.
inline KeptSet sampling_set(std::vector<std::pair<float, int>> & cand, int ntop, const SamplingParams & sp,
                            const std::function<void(double, float &, double &)> & stats,
                            const std::function<void(std::vector<float> &)> & full) {
    auto before = [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    std::sort(cand.begin(), cand.end(), before);
    const double invT = 1.0 / sp.temp;
    ntop = std::min<int>(ntop, (int) cand.size());
    KeptSet ks;
    auto pick = [&](const std::vector<double> & pr, int k, const std::function<int(int)> & tok) {
        for (int i = 0; i < k; ++i) { ks.tok.push_back(tok(i)); ks.pr.push_back(pr[i]); }
        return ks;
    };
    // kept prefix of a sorted probability list: min_p, then top_p (crossing token included); -1: may continue past the list
    auto cut = [&](const std::vector<double> & pr, int n, bool complete) {
        int k = n;
        if (sp.min_p > 0) {
            int kk = 1;
            while (kk < k && pr[kk] >= sp.min_p * pr[0]) ++kk;
            if (kk == k && !complete) return -1;
            k = kk;
        }
        if (sp.top_p < 1.0f) {
            double cum = 0;
            int kk = 0;
            while (kk < k) { cum += pr[kk++]; if (cum >= sp.top_p) return kk; }
        }
        return complete || k < n ? k : -1;
    };
    if (sp.top_k > 0 && sp.top_k <= ntop) {   // the top_k set is among the candidates: normalized over it
        const int k = sp.top_k;
        std::vector<double> pr(k);
        double z = 0;
        for (int i = 0; i < k; ++i) { pr[i] = std::exp((cand[i].first - cand[0].first) * invT); z += pr[i]; }
        for (auto & v : pr) v /= z;
        return pick(pr, cut(pr, k, true), [&](int i) { return cand[i].second; });
    }
    if (sp.top_k <= 0) {   // top_k off: probabilities over the whole row; the candidates suffice if the kept set ends inside
        float M = 0;
        double Z = 0;
        stats(invT, M, Z);
        std::vector<double> pr(ntop);
        for (int i = 0; i < ntop; ++i) pr[i] = std::exp((cand[i].first - M) * invT) / Z;
        const int k = cut(pr, ntop, false);
        if (k > 0) return pick(pr, k, [&](int i) { return cand[i].second; });
    }
    // the whole row: the top_k set (if any), weights relative to its best logit, kept set via log-weight bins
    ks.full = true;
    std::vector<float> lg;
    full(lg);
    const int n = (int) lg.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    auto by_logit = [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); };
    if (sp.top_k > 0 && sp.top_k < n) {
        std::nth_element(idx.begin(), idx.begin() + sp.top_k, idx.end(), by_logit);
        idx.resize(sp.top_k);
    }
    float M = -INFINITY;
    for (int i : idx) M = std::max(M, lg[i]);
    std::vector<double> w(idx.size());
    double Z = 0;
    for (size_t j = 0; j < idx.size(); ++j) { w[j] = std::exp((lg[idx[j]] - M) * invT); Z += w[j]; }
    // bins of log weight (1/8 nat), heavier bins first: a bin's tokens all outweigh the next bin's, so only the boundary
    // bin needs sorting (the last bin takes every weight below e^-64 of the best)
    constexpr int NB = 8 * 64;
    std::vector<std::vector<int>> bins(NB + 1);   // positions in idx / w
    std::vector<double> mass(NB + 1, 0.0);
    for (size_t j = 0; j < idx.size(); ++j) {
        if (sp.min_p > 0 && w[j] < sp.min_p) continue;   // (min_p relative to the best token, whose weight is 1)
        const int b = std::min<int>(NB, (int) ((M - lg[idx[j]]) * invT * 8.0));
        bins[b].push_back((int) j);
        mass[b] += w[j] / Z;
    }
    std::vector<int> kept;
    std::vector<double> pr;
    double cum = 0;
    for (int b = 0; b <= NB; ++b) {
        if (bins[b].empty()) continue;
        if (sp.top_p < 1.0f && cum + mass[b] >= sp.top_p) {   // boundary bin: in logit order up to the crossing token
            std::sort(bins[b].begin(), bins[b].end(), [&](int x, int y) { return by_logit(idx[x], idx[y]); });
            for (int j : bins[b]) {
                kept.push_back(idx[j]); pr.push_back(w[j] / Z); cum += w[j] / Z;
                if (cum >= sp.top_p) break;
            }
            break;
        }
        for (int j : bins[b]) { kept.push_back(idx[j]); pr.push_back(w[j] / Z); }
        cum += mass[b];
    }
    return pick(pr, (int) kept.size(), [&](int i) { return kept[i]; });
}

inline int sample_candidates(std::vector<std::pair<float, int>> & cand, int ntop, const SamplingParams & sp, std::mt19937_64 & rng,
                             const std::function<void(double, float &, double &)> & stats,
                             const std::function<void(std::vector<float> &)> & full) {
    if (sp.temp <= 0.0f || cand.size() == 1) {
        int best = 0;
        for (int i = 1; i < (int) cand.size(); ++i)
            if (cand[i].first > cand[best].first || (cand[i].first == cand[best].first && cand[i].second < cand[best].second)) best = i;
        return cand[best].second;
    }
    const KeptSet ks = sampling_set(cand, ntop, sp, stats, full);
    double tot = 0;
    for (double p : ks.pr) tot += p;
    double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
    for (size_t i = 0; i < ks.pr.size(); ++i) { u -= ks.pr[i]; if (u <= 0) return ks.tok[i]; }
    return ks.tok.back();
}

} // namespace hyper
