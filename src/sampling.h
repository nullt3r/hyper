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
    // bins of log weight (1/8 nat), heavier bins first: a bin's tokens all outweigh the next bin's, so only the boundary
    // bin needs sorting (the last bin takes every weight below e^-64 of the best). A stable counting sort by bin keeps the
    // order of idx inside a bin (the same lists, sums and kept set as per-bin vectors; ~3x faster on a 248k row)
    constexpr int NB = 8 * 64;
    thread_local std::vector<double> w;
    thread_local std::vector<uint16_t> bin;
    thread_local std::vector<int> pos;
    const int m = (int) idx.size();
    w.resize(m); bin.resize(m);
    double Z = 0;
    for (int j = 0; j < m; ++j) { w[j] = std::exp((lg[idx[j]] - M) * invT); Z += w[j]; }
    double mass[NB + 1] = {0};
    int cnt[NB + 1] = {0};
    for (int j = 0; j < m; ++j) {
        if (sp.min_p > 0 && w[j] < sp.min_p) { bin[j] = 0xffff; continue; }   // (min_p relative to the best token, weight 1)
        const int b = std::min<int>(NB, (int) ((M - lg[idx[j]]) * invT * 8.0));
        bin[j] = (uint16_t) b;
        mass[b] += w[j] / Z;
        ++cnt[b];
    }
    double cum = 0;
    int bstar = NB + 1;   // the boundary bin (top_p crossed inside it), none: every bin kept
    for (int b = 0; b <= NB; ++b) {
        if (!cnt[b]) continue;
        if (sp.top_p < 1.0f && cum + mass[b] >= sp.top_p) { bstar = b; break; }
        cum += mass[b];
    }
    const int last = std::min(bstar, NB);
    int off[NB + 2];
    off[0] = 0;
    for (int b = 0; b <= last; ++b) off[b + 1] = off[b] + cnt[b];
    pos.resize(off[last + 1]);
    {
        int fill[NB + 1];
        for (int b = 0; b <= last; ++b) fill[b] = off[b];
        for (int j = 0; j < m; ++j) { const int b = bin[j]; if (b <= last) pos[fill[b]++] = j; }
    }
    const int nfull = bstar <= NB ? off[bstar] : off[last + 1];
    for (int i = 0; i < nfull; ++i) { const int j = pos[i]; ks.tok.push_back(idx[j]); ks.pr.push_back(w[j] / Z); }
    if (bstar <= NB) {   // boundary bin: in logit order up to the crossing token
        std::sort(pos.begin() + off[bstar], pos.begin() + off[bstar + 1], [&](int x, int y) { return by_logit(idx[x], idx[y]); });
        for (int i = off[bstar]; i < off[bstar + 1]; ++i) {
            const int j = pos[i];
            ks.tok.push_back(idx[j]); ks.pr.push_back(w[j] / Z); cum += w[j] / Z;
            if (cum >= sp.top_p) break;
        }
    }
    return ks;
}

// Gumbel noise fixed per (seed, position, token): a draw is argmax_i (log p_i + G_i), exact sampling from p for every
// position, and an MTP drafter drawing with the same noise (draft_pick) proposes the main model's sample whenever its
// distribution is close. Positions are those of the sampled token, so a plain and a speculative run with one seed produce
// the same text.
inline uint64_t gmix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}
inline double gumbel_noise(uint64_t seed, int64_t pos, int tok) {
    const uint64_t k = gmix64(seed ^ gmix64(((uint64_t) (uint32_t) pos << 32) | (uint32_t) tok));
    const double u = ((double) (k >> 11) + 0.5) * (1.0 / 9007199254740992.0);   // (0, 1), 53 bits
    return -std::log(-std::log(u));
}

// the draw from a kept set: argmax_i (log p_i + G(seed, pos, token_i))
inline int gumbel_pick(const KeptSet & ks, uint64_t seed, int64_t pos) {
    int best = 0;
    double bv = -INFINITY;
    for (size_t i = 0; i < ks.tok.size(); ++i) {
        const double v = std::log(ks.pr[i]) + gumbel_noise(seed, pos, ks.tok[i]);
        if (v > bv || (v == bv && ks.tok[i] < ks.tok[best])) { bv = v; best = (int) i; }
    }
    return ks.tok[best];
}

inline int sample_candidates(std::vector<std::pair<float, int>> & cand, int ntop, const SamplingParams & sp, uint64_t seed, int64_t pos,
                             const std::function<void(double, float &, double &)> & stats,
                             const std::function<void(std::vector<float> &)> & full) {
    if (sp.temp <= 0.0f || cand.size() == 1) {
        int best = 0;
        for (int i = 1; i < (int) cand.size(); ++i)
            if (cand[i].first > cand[best].first || (cand[i].first == cand[best].first && cand[i].second < cand[best].second)) best = i;
        return cand[best].second;
    }
    return gumbel_pick(sampling_set(cand, ntop, sp, stats, full), seed, pos);
}

// an MTP draft coupled to the main model's sampling: the same truncation (top_k, min_p, top_p) over the MTP head's top
// candidates (the rest of its distribution is not read -- an approximation; drafts are verified) and the same noise, so
// the draft is the main model's sample whenever the two distributions are close
inline int draft_pick(std::vector<std::pair<float, int>> & cand, const SamplingParams & sp, uint64_t seed, int64_t pos) {
    std::sort(cand.begin(), cand.end(), [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    const int n = (int) cand.size();
    int k = sp.top_k > 0 ? std::min(sp.top_k, n) : n;
    std::vector<double> pr(k);
    double z = 0;
    for (int i = 0; i < k; ++i) { pr[i] = std::exp((cand[i].first - cand[0].first) / sp.temp); z += pr[i]; }
    for (auto & v : pr) v /= z;
    if (sp.min_p > 0) { int kk = 1; while (kk < k && pr[kk] >= sp.min_p * pr[0]) ++kk; k = kk; }
    if (sp.top_p < 1.0f) { double cum = 0; int kk = 0; while (kk < k) { cum += pr[kk++]; if (cum >= sp.top_p) break; } k = kk; }
    int best = 0;
    double bv = -INFINITY;
    for (int i = 0; i < k; ++i) {
        const double v = std::log(pr[i]) + gumbel_noise(seed, pos, cand[i].second);
        if (v > bv) { bv = v; best = i; }
    }
    return cand[best].second;
}

// a fresh per-request seed (SamplingParams::seed 0)
inline uint64_t random_seed64() {
    std::random_device rd;
    return ((uint64_t) rd() << 32) ^ rd();
}

} // namespace hyper
