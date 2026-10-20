// deepseek41 (DeepSeek-V4.1-Flash): 40 layers, hyper-connections (4 streams, Sinkhorn mixes, the pre-mix lagged by a block),
// attention over a 128-token window ring plus up to 512 compressed rows an indexer picks (one shared K = V row of 512 per
// token, rotary on its last 64 values, a sink per head, grouped low-rank output), compressed rows shared by the layers after
// their KV source, Engram n-gram tables at two layers, sqrt-softplus routed MoE (384 experts, top 6) + one shared expert.
#pragma once
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hyper {

struct Ds4Config {
    int n_layer = 0, n_embd = 0, n_vocab = 0, n_head = 0, head_dim = 0, rd = 0, q_lora = 0, o_groups = 0, o_lora = 0, win = 0;
    float eps = 0, hc_eps = 0;
    int hc = 0, sinkhorn = 0;
    std::vector<int> ratio;              // per layer: compress ratio (0: the window only)
    std::vector<char> kv_src, ix_src;    // per layer: owns compressed rows / runs the indexer
    std::vector<int> kv_of, ix_of;       // per layer: the KV source / index source it reads (-1: none)
    int cand_src = -1, cand_topk = 0, cand_blk = 0;
    int ix_heads = 0, ix_hd = 0, ix_topk = 0;
    int n_expert = 0, n_used = 0, ff = 0;
    float w_scale = 1, clamp_exp = 0, clamp_sh = 0;
    float rope_base = 0, comp_base = 0, yarn_factor = 1, beta_fast = 0, beta_slow = 0;
    int orig_ctx = 0;
    // Engram
    std::vector<int> eg_layers;
    int eg_ngram = 0, eg_heads = 0, eg_hd = 0, eg_pad = 0;
    std::vector<int32_t> eg_tmap;           // token -> compressed id
    std::vector<int64_t> eg_mult;           // [n_eg][ngram]
    std::vector<int64_t> eg_primes, eg_off; // [n_eg][cols]
    int eg_cols() const { return (eg_ngram - 1) * eg_heads; }
    int eg_index(int il) const { for (size_t i = 0; i < eg_layers.size(); ++i) if (eg_layers[i] == il) return (int) i; return -1; }
    // rows of Engram table i for the token at position t of toks (toks[0..t] known; earlier positions pad): out[cols]
    void eg_rows(int i, const int * toks, int t, int64_t * out) const;

    static Ds4Config from_gguf(const GGUF & g);
    std::string describe() const;
    // rotary frequencies (rd / 2) of a window-only layer (comp = false) or a compressing layer
    std::vector<float> freqs(bool comp) const;
};

} // namespace hyper
