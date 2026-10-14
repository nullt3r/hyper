// Flash-Next QSA decode path at depth on one GPU, synthetic caches: indexer scoring + top-512 selection
// (idx_select_hist) and the sparse split-K attention (attn_split) as GPU 1/2 run them (12 q heads on 1 kv head, 256 dims).
// Prints the time per call of each wrapper and hashes of the scores, the selected cells and the attention output, so a
// kernel change can be checked bit for bit. usage: qsabench [ctx=160000] [nt=4] [iters=200]
#include "../src/kernels.cuh"
#include "../src/kernels4.cuh"
#include "../src/vmm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
using namespace hyper;

static unsigned long long fnv(const void * p, size_t n, unsigned long long h = 1469598103934665603ull) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
// the exact gated attention (fp64) of the rows the prefill flash attention computes: q * scale unrounded, K/V as cached;
// grid (n_head, nt), 8 warps over the positions, lane owns 8 dims; the warps' online softmaxes combined at the end
__global__ void k_ref_attn(const float * qkv, int stride, const half * kc, const half * vc, double * out, int out_stride, const int * pos_p,
                           int max_pos, int head_off, int group, int kv_off, float scale) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int hk = (head_off + h) / group - kv_off, n_pos = *pos_p + t + 1;
    const float * q = qkv + (size_t) t * stride + (size_t) h * 512;
    double qr[8], acc[8], m = -1e300, l = 0;
    for (int j = 0; j < 8; ++j) { qr[j] = (double) q[lane * 8 + j] * scale; acc[j] = 0; }
    const half * kb = kc + (size_t) hk * max_pos * 256, * vb = vc + (size_t) hk * max_pos * 256;
    for (int p = w; p < n_pos; p += 8) {
        double sc = 0;
        for (int j = 0; j < 8; ++j) sc += qr[j] * (double) __half2float(kb[(size_t) p * 256 + lane * 8 + j]);
        for (int o = 16; o; o >>= 1) sc += __shfl_xor_sync(0xffffffff, sc, o);
        const double mn = fmax(m, sc), corr = exp(m - mn), pw = exp(sc - mn);
        l = l * corr + pw;
        for (int j = 0; j < 8; ++j) acc[j] = acc[j] * corr + pw * (double) __half2float(vb[(size_t) p * 256 + lane * 8 + j]);
        m = mn;
    }
    __shared__ double sm[8], sl[8], so[8][256];
    if (lane == 0) { sm[w] = m; sl[w] = l; }
    for (int j = 0; j < 8; ++j) so[w][lane * 8 + j] = acc[j];
    __syncthreads();
    const int d = threadIdx.x;
    double M = -1e300, L = 0, O = 0;
    for (int i = 0; i < 8; ++i) M = fmax(M, sm[i]);
    for (int i = 0; i < 8; ++i) { const double f = exp(sm[i] - M); L += sl[i] * f; O += so[i][d] * f; }
    const double g = (double) q[256 + d];
    out[(size_t) t * out_stride + (size_t) h * 256 + d] = O / L / (1.0 + exp(-g));
}

template <class T> static std::vector<T> dl(const T * d, size_t n) {
    std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}

// QSA_LOAD=file (attn_prefill's HYPER_FA_DUMP): the prefill flash attention on real inputs, one pass and the key split against
// the exact fp64 attention
static int load_fa(const char * fn) {
    FILE * f = fopen(fn, "rb");
    if (!f) { printf("cannot open %s\n", fn); return 1; }
    int h[9]; float scale;
    if (fread(h, 4, 9, f) != 9 || fread(&scale, 4, 1, f) != 1) return 1;
    const int nt = h[0], stride = h[1], pos = h[2], n_head = h[3], head_off = h[4], group = h[5], kv_off = h[6], n_kv = h[7], hd = h[8];
    const int n_pos = pos + nt, max_pos = n_pos;
    std::vector<float> q((size_t) nt * stride);
    std::vector<half> kc((size_t) n_kv * n_pos * hd), vc(kc.size());
    if (fread(q.data(), 4, q.size(), f) != q.size() || fread(kc.data(), 2, kc.size(), f) != kc.size() || fread(vc.data(), 2, vc.size(), f) != vc.size()) {
        printf("short file\n"); return 1;
    }
    fclose(f);
    float * d_q, * d_out; half * d_k, * d_v; int * d_pos; double * d_ref;
    CK(cudaMalloc(&d_q, q.size() * 4)); CK(cudaMalloc(&d_k, kc.size() * 2)); CK(cudaMalloc(&d_v, vc.size() * 2));
    CK(cudaMalloc(&d_out, (size_t) nt * n_head * hd * 4)); CK(cudaMalloc(&d_ref, (size_t) nt * n_head * hd * 8)); CK(cudaMalloc(&d_pos, 4));
    CK(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_k, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_pos, &pos, 4, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    k_ref_attn<<<dim3(n_head, nt), 256, 0, s>>>(d_q, stride, d_k, d_v, d_ref, n_head * hd, d_pos, max_pos, head_off, group, kv_off, scale);
    CK(cudaStreamSynchronize(s));
    const auto ref = dl(d_ref, (size_t) nt * n_head * hd);
    printf("%s: %d rows at %d, %d q heads on %d kv heads\n", fn, nt, pos, n_head, n_kv);
    for (int ph : {-1, pos}) {
        attn_prefill(d_q, stride, d_k, d_v, d_out, n_head * hd, d_pos, max_pos, n_head, head_off, group, kv_off, hd, scale, nt, s, ph);
        CK(cudaStreamSynchronize(s));
        const auto o = dl(d_out, ref.size());
        double mx = 0, n2 = 0, d2 = 0;
        for (size_t i = 0; i < o.size(); ++i) { const double e = o[i] - ref[i]; mx = std::max(mx, std::fabs(e)); n2 += e * e; d2 += ref[i] * ref[i]; }
        printf("  %s vs exact fp64: max |err| %.3g  rel L2 %.4g\n", ph < 0 ? "one pass " : "key split", mx, std::sqrt(n2 / d2));
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (getenv("QSA_LOAD")) return load_fa(getenv("QSA_LOAD"));
    const int ctx = argc > 1 ? atoi(argv[1]) : 160000, nt = argc > 2 ? atoi(argv[2]) : 4, iters = argc > 3 ? atoi(argv[3]) : 200;
    const int max_pos = 262144, idx_heads = 4, top = 2048 / 4, list_stride = 2052, score_rows = 64, score_stride = max_pos / 4 + 4;
    // QSA_SHAPE=27b: the 27B's GPU 1 (8 q heads 8..15 on kv heads 1, 2; group 6) instead of Flash-Next's (12 q heads on 1 kv head)
    // QSA_SHAPE=27b0: the 27B's GPU 0 (q heads 0..7: six on kv head 0, two on kv head 1)
    const std::string shape = getenv("QSA_SHAPE") ? getenv("QSA_SHAPE") : "";
    const bool b27 = shape == "27b" || shape == "27b0", g0 = shape == "27b0";
    const int hd = 256, n_head_l = b27 ? 8 : 12, n_kv_l = b27 ? 2 : 1, group = b27 ? 6 : 12, head_off = b27 && !g0 ? 8 : 0,
              kv_off = b27 && !g0 ? 1 : 0;
    const int qstride = n_head_l * 2 * hd + 2 * n_kv_l * hd;   // q|gate per head, then k, v (as in big0)
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // caches: pooled indexer keys and K/V rows for every position up to ctx (+ the nt new ones)
    const size_t npool = (size_t) max_pos / 4 + 1;
    std::vector<half> pool(npool * 128), kc((size_t) n_kv_l * max_pos * hd), vc((size_t) n_kv_l * max_pos * hd);
    for (size_t i = 0; i < (size_t) (ctx + nt) / 4 * 128 + 128; ++i) pool[i] = __float2half(nd(rng));
    for (int h = 0; h < n_kv_l; ++h)
        for (size_t i = 0; i < (size_t) (ctx + nt) * hd; ++i) { kc[(size_t) h * max_pos * hd + i] = __float2half(nd(rng)); vc[(size_t) h * max_pos * hd + i] = __float2half(nd(rng)); }
    std::vector<float> qn((size_t) nt * idx_heads * 128), qkv((size_t) nt * qstride);
    // QSA_ZERO=f: a share f of the pools scores exactly 0 (every head's dot product negative, as relu leaves it), the
    // rest as before -- real scores pile up in few histogram bins
    const float zero = getenv("QSA_ZERO") ? (float) atof(getenv("QSA_ZERO")) : 0.0f;
    for (auto & x : qn) x = nd(rng);
    if (zero > 0) {
        for (auto & x : qn) x = std::fabs(x);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (size_t b = 0; b < (size_t) (ctx + nt) / 4 + 1; ++b) {
            const bool z = u(rng) < zero;
            for (int i = 0; i < 128; ++i) { const float v = std::fabs(__half2float(pool[b * 128 + i])); pool[b * 128 + i] = __float2half(z ? -v : v); }
        }
    }
    for (auto & x : qkv) x = nd(rng);
    // QSA_QMUL=f: the q parts times f (score spread f: peakier attention over the positions)
    if (getenv("QSA_QMUL"))
        for (size_t i = 0; i < qkv.size(); ++i) {
            const size_t c = i % qstride;
            if (c < (size_t) n_head_l * 2 * hd && c % (2 * hd) < (size_t) hd) qkv[i] *= (float) atof(getenv("QSA_QMUL"));
        }
    half * d_pool, * d_k, * d_v; float * d_qn, * d_qkv, * d_scores, * d_part, * d_out; unsigned * d_hist; int * d_list, * d_list_n, * d_pos;
    // QSA_VMM=1: the caches on the CUDA VMM API mapped granule by granule, as the engine's elastic caches
    std::vector<std::unique_ptr<VmmBuf>> vbufs;
    auto alloc_cache = [&](size_t bytes) -> void * {
        if (!getenv("QSA_VMM")) { void * p = nullptr; CK(cudaMalloc(&p, bytes)); return p; }
        auto b = std::make_unique<VmmBuf>();
        int dev = 0; CK(cudaGetDevice(&dev));
        b->reserve(dev, bytes);
        b->map(0, bytes, false);
        void * p = b->ptr(); vbufs.push_back(std::move(b)); return p;
    };
    d_pool = (half *) alloc_cache(pool.size() * 2); d_k = (half *) alloc_cache(kc.size() * 2); d_v = (half *) alloc_cache(vc.size() * 2);
    CK(cudaMalloc(&d_qn, qn.size() * 4)); CK(cudaMalloc(&d_qkv, qkv.size() * 4));
    CK(cudaMalloc(&d_scores, (size_t) score_rows * score_stride * 4)); CK(cudaMalloc(&d_hist, (size_t) score_rows * 65536 * 4));
    CK(cudaMalloc(&d_part, attn_part_floats(n_head_l, n_kv_l, nt, hd) * 4)); CK(cudaMalloc(&d_out, (size_t) nt * n_head_l * hd * 4));
    CK(cudaMalloc(&d_list, (size_t) nt * list_stride * 4)); CK(cudaMalloc(&d_list_n, nt * 4)); CK(cudaMalloc(&d_pos, 4));
    CK(cudaMemcpy(d_pool, pool.data(), pool.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_k, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_v, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_qn, qn.data(), qn.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_qkv, qkv.data(), qkv.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, &ctx, 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_scores, 0, (size_t) score_rows * score_stride * 4));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    const float scale = 1.0f / sqrtf((float) hd);
    auto select = [&] { idx_select_hist(d_qn, d_pool, d_pos, nt, idx_heads, top, d_scores, score_stride, score_rows, d_hist, d_list, list_stride,
                                        d_list_n, s); };
    auto attend = [&] { attn_split(d_qkv, qstride, d_k, d_v, d_part, d_out, n_head_l * hd, d_pos, max_pos, n_head_l, n_kv_l, head_off, group,
                                   kv_off, hd, scale, nt, s, d_list, list_stride, d_list_n); };
    select(); attend(); CK(cudaStreamSynchronize(s));
    // hashes: scores of the scored pools, the selections, the attention output
    unsigned long long hs = 1469598103934665603ull, hl = hs;
    const auto sc = dl(d_scores, (size_t) nt * score_stride);
    const auto ln = dl(d_list_n, nt);
    const auto li = dl(d_list, (size_t) nt * list_stride);
    for (int t = 0; t < nt; ++t) {
        const int np = (ctx + t + 1) / 4;
        hs = fnv(&sc[(size_t) t * score_stride], (size_t) np * 4, hs);
        hl = fnv(&ln[t], 4, hl); hl = fnv(&li[(size_t) t * list_stride], (size_t) ln[t] * 4, hl);
    }
    const auto o = dl(d_out, (size_t) nt * n_head_l * hd);
    const unsigned long long ho = fnv(o.data(), o.size() * 4);
    // reference selection (the one-block kernel of kernels4.cu): must pick the same cells
    std::vector<int> ln2, li2;
    {
        idx_select(d_qn, d_pool, d_pos, nt, idx_heads, top, d_scores, score_stride, score_rows, d_list, list_stride, d_list_n, s);
        CK(cudaStreamSynchronize(s));
        ln2 = dl(d_list_n, nt); li2 = dl(d_list, (size_t) nt * list_stride);
        unsigned long long h2 = 1469598103934665603ull;
        for (int t = 0; t < nt; ++t) { h2 = fnv(&ln2[t], 4, h2); h2 = fnv(&li2[(size_t) t * list_stride], (size_t) ln2[t] * 4, h2); }
        printf("selection %s the reference kernel's\n", h2 == hl ? "matches" : "DIFFERS FROM");
        select(); CK(cudaStreamSynchronize(s));
    }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // cold: a 64 MB buffer is rewritten before every timed call, so nothing of the caches is left in L2 (as in the engine,
    // where a layer's weights pass through L2 between two uses of its caches)
    char * d_flush; const size_t flush_bytes = 64u << 20; CK(cudaMalloc(&d_flush, flush_bytes));
    auto timeit = [&](auto && f, bool cold) {
        for (int i = 0; i < 10; ++i) f();
        double tot = 0;
        for (int i = 0; i < iters; ++i) {
            if (cold) CK(cudaMemsetAsync(d_flush, i & 255, flush_bytes, s));
            CK(cudaEventRecord(e0, s)); f(); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); tot += ms;
        }
        return (float) (1e3 * tot / iters);
    };
    printf("ctx %d nt %d (cells/row %d): warm L2 select %.1f us  attention %.1f us | cold select %.1f us  attention %.1f us\n", ctx, nt,
           ln[0], timeit(select, false), timeit(attend, false), timeit(select, true), timeit(attend, true));
    // QSA_FA=1: the prefill flash attention (attn_prefill) for these nt rows at depth ctx (27B / dense layers), cold L2
    if (getenv("QSA_FA")) {
        int ph = -1;   // (the host position: the deep-context key split may engage)
        auto fa = [&] { attn_prefill(d_qkv, qstride, d_k, d_v, d_out, n_head_l * hd, d_pos, max_pos, n_head_l, head_off, group, kv_off, hd, scale,
                                     nt, s, ph); };
        fa(); CK(cudaStreamSynchronize(s));
        const auto of = dl(d_out, (size_t) nt * n_head_l * hd);
        printf("prefill flash attention, %d rows at %d: cold %.1f us  hash %016llx\n", nt, ctx, timeit(fa, true), fnv(of.data(), of.size() * 4));
        ph = ctx;   // the same rows with the split (when it engages): time, and the difference from the one-pass result
        fa(); CK(cudaStreamSynchronize(s));
        const auto os = dl(d_out, (size_t) nt * n_head_l * hd);
        double md = 0, num = 0, den = 0;
        for (size_t i = 0; i < of.size(); ++i) { const double d = (double) os[i] - of[i]; md = std::max(md, std::fabs(d)); num += d * d; den += (double) of[i] * of[i]; }
        printf("  with the key split: cold %.1f us  hash %016llx  max |diff| %.3g  rel L2 %.3g\n", timeit(fa, true), fnv(os.data(), os.size() * 4), md,
               std::sqrt(num / std::max(den, 1e-30)));
        // QSA_REF=1: both against the exact (fp64) attention of the same inputs
        if (getenv("QSA_REF")) {
            double * d_ref; CK(cudaMalloc(&d_ref, of.size() * 8));
            k_ref_attn<<<dim3(n_head_l, nt), 256, 0, s>>>(d_qkv, qstride, d_k, d_v, d_ref, n_head_l * hd, d_pos, max_pos, head_off, group, kv_off, scale);
            CK(cudaStreamSynchronize(s));
            const auto ref = dl(d_ref, of.size());
            auto err = [&](const std::vector<float> & x, const char * name) {
                double mx = 0, n2 = 0, d2 = 0;
                for (size_t i = 0; i < x.size(); ++i) { const double e = x[i] - ref[i]; mx = std::max(mx, std::fabs(e)); n2 += e * e; d2 += ref[i] * ref[i]; }
                printf("  %s vs exact fp64: max |err| %.3g  rel L2 %.4g\n", name, mx, std::sqrt(n2 / d2));
            };
            err(of, "one pass ");
            err(os, "key split");
        }
        return 0;
    }
    // the MTP layer's dense attention over every cached position (no indexer)
    auto dense = [&] { attn_split(d_qkv, qstride, d_k, d_v, d_part, d_out, n_head_l * hd, d_pos, max_pos, n_head_l, n_kv_l, head_off, group,
                                  kv_off, hd, scale, nt, s, nullptr, 0, nullptr); };
    dense(); CK(cudaStreamSynchronize(s));
    const auto od = dl(d_out, (size_t) nt * n_head_l * hd);
    printf("dense attention over %d cells: cold %.1f us  hash %016llx\n", ctx + 1, timeit(dense, true), fnv(od.data(), od.size() * 4));
    if (getenv("QSA_REF")) {   // the dense rows against the exact (fp64) attention
        double * d_ref; CK(cudaMalloc(&d_ref, od.size() * 8));
        k_ref_attn<<<dim3(n_head_l, nt), 256, 0, s>>>(d_qkv, qstride, d_k, d_v, d_ref, n_head_l * hd, d_pos, max_pos, head_off, group, kv_off, scale);
        CK(cudaStreamSynchronize(s));
        const auto ref = dl(d_ref, od.size());
        double mx = 0, n2 = 0, d2 = 0;
        for (size_t i = 0; i < od.size(); ++i) { const double e = od[i] - ref[i]; mx = std::max(mx, std::fabs(e)); n2 += e * e; d2 += ref[i] * ref[i]; }
        printf("  dense vs exact fp64: max |err| %.3g  rel L2 %.4g\n", mx, std::sqrt(n2 / d2));
    }
    if (getenv("QSA_ROWID") && nt > 1) {   // each row alone (one-row call at its own position): the same bits as in the nt-row call
        int same = 0;
        for (int t = 0; t < nt; ++t) {
            const int pt = ctx + t;
            CK(cudaMemcpy(d_pos, &pt, 4, cudaMemcpyHostToDevice));
            attn_split(d_qkv + (size_t) t * qstride, qstride, d_k, d_v, d_part, d_out, n_head_l * hd, d_pos, max_pos, n_head_l, n_kv_l, head_off,
                       group, kv_off, hd, scale, 1, s, nullptr, 0, nullptr);
            CK(cudaStreamSynchronize(s));
            const auto o1 = dl(d_out, (size_t) n_head_l * hd);
            same += memcmp(o1.data(), od.data() + (size_t) t * n_head_l * hd, o1.size() * 4) == 0;
        }
        CK(cudaMemcpy(d_pos, &ctx, 4, cudaMemcpyHostToDevice));
        printf("  rows alone vs together: %d / %d bit-identical\n", same, nt);
    }
    printf("hash scores %016llx  list %016llx  out %016llx\n", hs, hl, ho);
    return 0;
}
