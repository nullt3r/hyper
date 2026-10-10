// exact sampling (src/sampling.h): the kept set built from three vocab slices' top-64 candidates (fast path) or the whole
// row (fallback) against a full sort of the row, on reference logit rows, for many settings.
// usage: samplertest ref.bin [row step]
#include "../src/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace hyper;

constexpr int TOPK_TEST = 64;   // (kernels.cuh TOPK: candidates per vocab slice)

static bool read_ref(const char * path, std::vector<int> & toks, std::vector<float> & logits, int & nv, int & first) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    int n = 0;
    first = 0;
    if (fread(&n, 4, 1, f) != 1) return false;
    const bool v2 = n == 0x32464552;
    if (v2 && fread(&n, 4, 1, f) != 1) return false;
    if (fread(&nv, 4, 1, f) != 1) return false;
    if (v2 && fread(&first, 4, 1, f) != 1) return false;
    toks.resize(n);
    if (fread(toks.data(), 4, n, f) != (size_t) n) return false;
    logits.resize((size_t) (n - first) * nv);
    if (fread(logits.data(), 4, logits.size(), f) != logits.size()) return false;
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s ref.bin [row step]\n", argv[0]); return 1; }
    std::vector<int> toks; std::vector<float> L; int nv = 0, first = 0;
    if (!read_ref(argv[1], toks, L, nv, first)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    const int step = argc > 2 ? atoi(argv[2]) : 16, rows = (int) (L.size() / nv);
    struct Cfg { float T; int k; float p, mp; int n = 0, bad = 0, fb = 0; double maxd = 0; };
    std::vector<Cfg> cfgs;
    for (float T : {0.6f, 1.0f, 1.5f})
        for (int k : {0, 20, 64, 100, 1000})
            for (float p : {0.8f, 0.95f, 1.0f})
                for (float mp : {0.0f, 0.05f}) cfgs.push_back({T, k, p, mp});
    std::vector<int> order(nv);
    for (int r = 0; r < rows; r += step) {
        const float * lg = &L[(size_t) r * nv];
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
        for (auto & c : cfgs) {
            SamplingParams sp; sp.temp = c.T; sp.top_k = c.k; sp.top_p = c.p; sp.min_p = c.mp;
            // three vocab slices (as the GPUs), the top 64 of each
            std::vector<std::pair<float, int>> cand;
            for (int g = 0; g < 3; ++g) {
                const int a = (int) ((long) nv * g / 3), b = (int) ((long) nv * (g + 1) / 3);
                std::vector<int> sl(b - a);
                std::iota(sl.begin(), sl.end(), a);
                const int m = std::min<int>(TOPK_TEST, (int) sl.size());
                std::partial_sort(sl.begin(), sl.begin() + m, sl.end(), [&](int x, int y) { return lg[x] > lg[y] || (lg[x] == lg[y] && x < y); });
                for (int i = 0; i < m; ++i) cand.push_back({lg[sl[i]], sl[i]});
            }
            auto stats = [&](double invT, float & M, double & Z) {
                M = lg[order[0]];
                Z = 0;
                for (int i = 0; i < nv; ++i) Z += std::exp((lg[i] - M) * invT);
            };
            auto full = [&](std::vector<float> & out) { out.assign(lg, lg + nv); };
            const KeptSet ks = sampling_set(cand, TOPK_TEST, sp, stats, full);
            // reference: full sort; temperature; top_k set normalized; min_p vs the best; top_p (crossing token kept)
            const int K = c.k > 0 ? std::min(c.k, nv) : nv;
            const double invT = 1.0 / c.T;
            double Z = 0;
            for (int i = 0; i < K; ++i) Z += std::exp((lg[order[i]] - lg[order[0]]) * invT);
            std::vector<double> pr(K);
            for (int i = 0; i < K; ++i) pr[i] = std::exp((lg[order[i]] - lg[order[0]]) * invT) / Z;
            int k = K;
            if (c.mp > 0) { int kk = 1; while (kk < k && pr[kk] >= c.mp * pr[0]) ++kk; k = kk; }
            if (c.p < 1.0f) { double cum = 0; int kk = 0; while (kk < k) { cum += pr[kk++]; if (cum >= c.p) break; } k = kk; }
            std::vector<std::pair<int, double>> want(k), got(ks.tok.size());
            for (int i = 0; i < k; ++i) want[i] = {order[i], pr[i]};
            for (size_t i = 0; i < ks.tok.size(); ++i) got[i] = {ks.tok[i], ks.pr[i]};
            std::sort(want.begin(), want.end());
            std::sort(got.begin(), got.end());
            bool ok = want.size() == got.size();
            for (size_t i = 0; ok && i < want.size(); ++i) {
                ok = want[i].first == got[i].first;
                c.maxd = std::max(c.maxd, std::fabs(want[i].second - got[i].second) / want[i].second);
            }
            c.n++; c.bad += !ok; c.fb += ks.full;
        }
    }
    int bad = 0;
    for (auto & c : cfgs) {
        printf("T %.1f top_k %4d top_p %.2f min_p %.2f: %d rows, %d set mismatches, whole row read %5.1f %%, max rel. prob. diff %.2e\n",
               c.T, c.k, c.p, c.mp, c.n, c.bad, 100.0 * c.fb / c.n, c.maxd);
        bad += c.bad;
    }
    printf("SAMPLERTEST %s: %d mismatches\n", bad ? "FAILED" : "OK", bad);
    return bad != 0;
}
