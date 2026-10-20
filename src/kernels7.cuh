// deepseek41 (DeepSeek-V4.1-Flash) kernels: hyper-connections with the lagged pre-mix, the rows' finish (RMSNorm, rotary on
// the last 64 values, the reference kernels' FP8 / FP4 roundings), the compressor, the indexer (scores, candidate blocks,
// top-k), sparse attention over the window ring and the compressed rows with a sink per head, the sqrt-softplus router and
// the Engram gate. Activations fp32, token rows; the caches hold the rounded values as bf16 (exact).
//
// The arithmetic follows Project Maya's maya-ds4 kernels (src/ds4/ds4_kernels.cu, MIT, Copyright (c) 2026 Peasant Smith and
// the Project Maya contributors), which follow DeepSeek's inference/model.py (MIT).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace hyper {

// a row's finish, in place (or into a bf16 copy): RMSNorm (w), rotary (the last rd values as adjacent pairs, position
// pos + (row / vpp) * pos_step, angle position * freqs[i]; inverse: the conjugate), rounding (quant 1: E4M3 values with a power-
// of-two scale per 32; 2: E2M1 values with an E4M3 scale per 16; 3: E2M1 with a power-of-two scale per 32)
struct Fin7 {
    const float * w = nullptr;
    float eps = 0.0f;
    int rd = 0;
    const float * freqs = nullptr;
    long long pos = 0;
    int pos_step = 1, vpp = 1;
    bool inverse = false;
    int quant = 0;
    uint16_t * dst = nullptr;     // bf16 rows (nullptr: back into x)
    long long dst_stride = 0;
    int ring = 0;                 // > 0: the copy of a row goes to dst row (its position % ring)
};
void fin7(float * x, int rows, int len, long long stride, const Fin7 & f, cudaStream_t s);

// ---- hyper-connections (hc streams of dim) ----
// mixes[t][m] = (h[t] . fn[m]) * rsqrt(mean(h[t]^2) + eps)  (h rows hc * dim, fn [nmix][hc * dim]; mixes rows of 32)
void hc7_mix(const float * h, const float * fn, int hcdim, int nmix, float eps, float * mixes, int nt, cudaStream_t s);
// pre = sigmoid(m * scale0 + base) + eps, post = 2 sigmoid(..), comb = Sinkhorn(softmax rows + eps) (rows of 8 / 8 / 64)
void hc7_split(const float * mixes, const float * scale, const float * base, int hc, int iters, float eps, float * pre, float * post,
               float * comb, int nt, cudaStream_t s);
// x[t] = rmsnorm(sum_s mix[t][s] h[t][s]) * w
void hc7_pre_norm(const float * h, const float * mix, const float * w, int dim, int hc, float eps, float * x, int nt, cudaStream_t s);
// out[t][k] = post[t][k] y[t] + sum_j comb[t][j][k] res[t][j]
void hc7_post(const float * y, const float * res, const float * post, const float * comb, int dim, int hc, float * out, int nt,
              cudaStream_t s);
// h[t][s] = x[t] (bf16 rows widened) for every stream
void embed7(const float * x, float * h, int dim, int hc, int nt, cudaStream_t s);

// ---- the compressor (ratio > 1): tokens in order; a group completed at position p (p + 1 a multiple of ratio) becomes
// latent row g = rmsnorm(sum_slot kv * softmax_slot(score)) * w; state: the open group's rows [ratio][d] ----
void compress7(const float * kv, long long kvs, const float * score, long long scs, float * kv_state, float * score_state, int ratio,
               long long p0, int T, int d, const float * w, float eps, float * latent, cudaStream_t s);

// ---- the indexer ----
// query t (position p0 + t) sees the compressed rows j < n(t) = (p0 + t + 1) / R. Without cand: out[t][j] for j < width
// (-inf for j >= n(t)). With cand (block ids, ncand per query): out[t][i] for i < width, position cand[t][i / blk] * blk + i % blk.
// score = sum_h w[t][h] relu(q[t][h] . key[j]) (q [T][heads][hd], w rows of 64, keys bf16 [rows][hd])
void index_score7(const float * q, const float * w, const uint16_t * keys, int heads, int hd, int T, long long p0, int R, int width,
                  const int * cand, const int * ncand, int cand_stride, int blk, float * out, long long out_stride, cudaStream_t s);
// per row t: the k(t) largest of vals[t][0 .. n(t)) (ties: the lowest indices), written in index order to out[t]; mapped through
// cand (as above) when given. n / k: n(t) = nfix > 0 ? nfix : (p0 + t + 1) / R (times blk with cand: ncand[t] * blk);
// k(t) = min(kcap, (p0 + t + 1) / R) (kcap_blocks > 0: min(kcap_blocks, n(t)))
void topk7(const float * vals, long long stride, int T, long long p0, int R, int kcap, const int * cand, const int * ncand, int cand_stride,
           int blk, int * out, int out_stride, cudaStream_t s);
// candidate blocks of a cand-source layer: block maxima of the scores (the newest block +inf), the top kblocks of them
// (index order) -> cand[t], ncand[t] = min(kblocks, blocks(t)); bm: scratch [T][bm_stride]
void cand_blocks7(const float * score, long long stride, int T, long long p0, int R, int blk, int kblocks, float * bm, long long bm_stride,
                  int * cand, int * ncand, int cand_stride, cudaStream_t s);

// ---- sparse attention ----
// query t (position p0 + t), head h: the window ring's rows of positions max(0, p - win + 1) .. p (ring row = position % ring)
// and nsel(t) = R > 0 ? min(kcap, (p + 1) / R) : 0 compressed rows ckv[sel[t][i]]; softmax(q . row * scale) with the sink;
// out[t][h] = sum p row. q / out [T][heads][d]
void sparse_attn7(const float * q, const uint16_t * ring, int ring_rows, int win, const uint16_t * ckv, const int * sel, int sel_stride,
                  long long p0, int R, int kcap, const float * sink, int heads, int d, float scale, float * out, int T, cudaStream_t s);

// ---- the MoE router: s = sqrt(softplus(logit)); top k of s + bias (ties: lower index); w = s / sum s * scale ----
void route7(const float * logits, int ls, const float * bias, int n, int k, float scale, int * ids, float * wts, int nt, cudaStream_t s);
// h[t][i] = silu(min(g, L)) * clamp(u, -L, L) with g = gu[t][i], u = gu[t][ff + i]
void swiglu7(const float * gu, int gus, float * h, int hs, int ff, float lim, int nt, cudaStream_t s);
// pre[t] = (1, 0, 0, ..) (rows of 8): the first block's pre-mix
void init_pre7(float * pre, int nt, cudaStream_t s);
// full-vocabulary sampling: *out = argmax_i (logits[i] / temp + G(seed, pos, i)) (sampling.h's gumbel_noise)
void gumbel_argmax7(const float * logits, int n, float temp, uint64_t seed, long long pos, int * out, cudaStream_t s);
// ---- DSpark drafter ----
// out[t][e] (row stride ostride) = mean over the hc streams of h[t][s][e]
void hc_mean7(const float * h, int dim, int hc, float * out, int ostride, int nt, cudaStream_t s);
// the draft block's attention: queries at p0 .. p0 + T - 1 see the ring rows of max(0, p0 - win) .. p0 - 1 and all T block rows
void draft_attn7(const float * q, const uint16_t * ring, int ring_rows, int win, const uint16_t * blk, int T, long long p0, const float * sink,
                 int heads, int d, float scale, float * out, cudaStream_t s);
// out[v] = logits[v] + mh[v] . emb[prev] (prev: the index of argmax pair *prev_pair, or y without it)
void markov7(const float * logits, const uint16_t * mh, const uint16_t * emb, int vocab, int rank, const float * prev_pair, int y, float * out,
             cudaStream_t s);
// conf[i] = sigmoid([xh[i] | emb[prev_i]] . w): prev_0 = y, prev_i = pairs[i - 1]'s index
void dconf7(const float * xh, int dim, const uint16_t * emb, int rank, const float * w, const float * pairs, int y, float * conf, int T,
            cudaStream_t s);
// x[i] = v (n values)
void fill7(float * x, float v, int n, cudaStream_t s);

// ---- Engram: h[t][s] += sigmoid(ssqrt(<rms(h[t][s]) * w[s], rms(key[t][s])> / sqrt(dim))) value[t] (kv rows: hc keys, value) ----
void engram_gate7(float * h, const float * kv, const float * w, int hc, int dim, float eps, int nt, cudaStream_t s);

} // namespace hyper
