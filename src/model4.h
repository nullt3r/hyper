// qwen4exp (Qwen3.8-Flash-Next): hyper-connected hybrid MoE.
//   residual: hc parallel streams of n_embd (mixed in and scattered back around every block)
//   token mixer: gated delta net (sigmoid output gate) or, every full_attn_interval-th layer, gated attention
//                with a QSA indexer (top blocks of compress_ratio cells)
//   ffn: softmax-routed MoE (top n_expert_used of n_expert, renormalized) + a sigmoid-gated shared expert
//   PLE: n-gram hash embeddings injected at one linear layer
#pragma once
#include "gguf.h"

#include <string>
#include <vector>

namespace hyper {

struct Q4Config {
    int n_layer = 0;             // trunk layers
    int n_embd = 0;              // 2560
    int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    float rope_base = 0, rms_eps = 0;
    int full_attn_interval = 0;
    // gated delta net
    int ssm_conv = 0, ssm_d_state = 0, ssm_n_group = 0, ssm_dt_rank = 0, ssm_d_inner = 0;
    // MoE
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    // hyper-connections
    int hc = 0, hc_lr = 0;
    // QSA indexer
    int idx_n_head = 0, idx_dim = 0, idx_top_k = 0, kpool = 0;
    std::vector<int> compress;   // per layer, 0 = dense attention
    // PLE
    int ple_layer = -1, ple_ngram = 0, ple_heads_per_ngram = 0, ple_conv = 0, ple_dim = 0, ple_eos = 0;
    std::vector<uint64_t> ple_mult, ple_offsets, ple_vocab;
    int n_vocab = 0;

    int head_v_dim() const { return ssm_d_inner / ssm_dt_rank; }
    int conv_dim() const { return ssm_d_inner + 2 * ssm_n_group * ssm_d_state; }
    bool is_full_attn(int il) const { return (il + 1) % full_attn_interval == 0; }
    int hc_dim() const { return hc * n_embd; }
    int ple_n_heads() const { return (ple_ngram - 1) * ple_heads_per_ngram; }

    static Q4Config from_gguf(const GGUF & g);
    std::string describe() const;
};

} // namespace hyper
