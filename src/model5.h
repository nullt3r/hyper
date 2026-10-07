// glm5-next (GLM-5.3-Flash): mHC hybrid MoE.
//   residual: 4 streams of n_embd mixed by per-token Sinkhorn matrices (mHC) around every block
//   token mixer: KDA (per-channel gated delta rule, 64 heads of 128) or, every 4th layer, nope MLA over a 512-wide latent
//                cache with a k-pool DSA indexer (32 heads, top 512 pools of 4 cells + the open pool's cells)
//   ffn: 3 leading dense layers, then sigmoid-routed MoE (288 experts, top 8, bias for the selection, scaled 2.5) + one
//        shared expert; SwiGLU with limit 10
#pragma once
#include "gguf.h"

#include <string>
#include <vector>

namespace hyper {

struct Glm5Config {
    int n_layer = 0, n_embd = 0, n_vocab = 0, n_head = 0;
    float rms_eps = 0, ln_eps = 0;
    std::vector<int> is_mla;            // per layer: 1 = MLA + indexer, 0 = KDA
    // KDA
    int kda_dim = 0, conv = 0;
    float gate_lb = 0;
    // MLA
    int q_lora = 0, kv_lora = 0, qk_dim = 0, v_dim = 0;
    // indexer
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0, kpool = 0;
    // mHC
    int hc = 0, sinkhorn = 0;
    float hc_eps = 0;
    // FFN
    int n_dense = 0, n_ff = 0, n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_shared = 0;
    float w_scale = 1, clamp_exp = 0, clamp_sh = 0;

    int d_inner() const { return n_head * kda_dim; }
    static Glm5Config from_gguf(const GGUF & g);
    std::string describe() const;
};

} // namespace hyper
