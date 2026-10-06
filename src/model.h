// qwen35 (Qwen3.5/3.8 dense hybrid: gated delta net + gated attention) model description
#pragma once
#include "gguf.h"

#include <string>
#include <vector>

namespace hyper {

struct Qwen35Config {
    int n_layer = 0;          // main layers (excludes NextN/MTP blocks)
    int n_embd = 0;
    int n_ff = 0;
    int n_head = 0;           // full-attention query heads
    int n_head_kv = 0;
    int head_dim = 0;         // 256
    int n_rot = 0;            // rotary dims (partial rotary)
    float rope_base = 0;
    float rms_eps = 0;
    int full_attn_interval = 0;
    // gated delta net
    int ssm_conv = 0;         // conv kernel size (4)
    int ssm_d_state = 0;      // head_k_dim (128)
    int ssm_n_group = 0;      // num k heads (16)
    int ssm_dt_rank = 0;      // num v heads (48)
    int ssm_d_inner = 0;      // head_v_dim * num_v_heads (6144)
    int n_vocab = 0;

    int head_v_dim() const { return ssm_d_inner / ssm_dt_rank; }
    int conv_dim() const { return ssm_d_inner + 2 * ssm_n_group * ssm_d_state; }
    bool is_full_attn(int il) const { return (il + 1) % full_attn_interval == 0; }

    static Qwen35Config from_gguf(const GGUF & g);
    std::string describe() const;
};

} // namespace hyper
