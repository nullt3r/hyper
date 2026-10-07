#include "model5.h"

#include <cstdio>
#include <stdexcept>

namespace hyper {

Glm5Config Glm5Config::from_gguf(const GGUF & g) {
    if (g.arch() != "glm5-next") throw std::runtime_error("expected glm5-next model, got " + g.arch());
    const std::string a = "glm5-next.";
    Glm5Config c;
    c.n_layer = (int) g.get_int(a + "block_count") - (int) g.get_int(a + "nextn_predict_layers", 0);
    c.n_embd = (int) g.get_int(a + "embedding_length");
    c.n_vocab = (int) g.get_int(a + "vocab_size");
    c.n_head = (int) g.get_int(a + "attention.head_count");
    c.rms_eps = (float) g.get_float(a + "attention.layer_norm_rms_epsilon");
    c.ln_eps = (float) g.get_float(a + "attention.layer_norm_epsilon", 1e-6);
    const auto kv = g.get_int_arr(a + "attention.head_count_kv");
    c.is_mla.assign(c.n_layer, 0);
    for (int il = 0; il < c.n_layer; ++il) c.is_mla[il] = (il < (int) kv.size() ? kv[il] : kv[0]) != 0;
    c.kda_dim = (int) g.get_int(a + "kda.head_dim");
    c.conv = (int) g.get_int(a + "ssm.conv_kernel");
    c.gate_lb = (float) g.get_float(a + "kda.gate_lower_bound", -5.0);
    c.q_lora = (int) g.get_int(a + "attention.q_lora_rank");
    c.kv_lora = (int) g.get_int(a + "attention.kv_lora_rank");
    c.qk_dim = (int) g.get_int(a + "attention.key_length_mla");
    c.v_dim = (int) g.get_int(a + "attention.value_length_mla");
    if (g.get_int(a + "rope.dimension_count", 0) != 0) throw std::runtime_error("glm5-next: roped MLA not supported");
    c.idx_heads = (int) g.get_int(a + "attention.indexer.head_count");
    c.idx_dim = (int) g.get_int(a + "attention.indexer.key_length");
    c.idx_top_k = (int) g.get_int(a + "attention.indexer.top_k");
    c.kpool = (int) g.get_int(a + "attention.indexer.kpool");
    c.hc = (int) g.get_int(a + "hyper_connection.count");
    c.sinkhorn = (int) g.get_int(a + "hyper_connection.sinkhorn_iterations");
    c.hc_eps = (float) g.get_float(a + "hyper_connection.epsilon");
    c.n_dense = (int) g.get_int(a + "leading_dense_block_count", 0);
    c.n_ff = (int) g.get_int(a + "feed_forward_length");
    c.n_expert = (int) g.get_int(a + "expert_count");
    c.n_expert_used = (int) g.get_int(a + "expert_used_count");
    c.n_ff_exp = (int) g.get_int(a + "expert_feed_forward_length");
    c.n_shared = (int) g.get_int(a + "expert_shared_count", 1);
    c.w_scale = (float) g.get_float(a + "expert_weights_scale", 1.0);
    if (g.get_int(a + "expert_gating_func", 2) != 2) throw std::runtime_error("glm5-next: only sigmoid gating");
    if (!g.get_int(a + "expert_weights_norm", 1)) throw std::runtime_error("glm5-next: expects normalized expert weights");
    if (g.has(a + "swiglu_clamp_exp")) c.clamp_exp = (float) g.get_float_arr(a + "swiglu_clamp_exp")[0];
    c.clamp_sh = g.has(a + "swiglu_clamp_shexp") ? (float) g.get_float_arr(a + "swiglu_clamp_shexp")[0] : c.clamp_exp;
    if (c.hc != 4 || c.kda_dim != 128 || c.kv_lora != 512 || c.idx_heads != 32 || c.idx_dim != 128 || c.kpool != 4 || c.n_embd > 4096 ||
        c.n_shared != 1)
        throw std::runtime_error("glm5-next: unsupported dimensions");
    return c;
}

std::string Glm5Config::describe() const {
    int nm = 0;
    for (int v : is_mla) nm += v;
    char b[512];
    snprintf(b, sizeof b,
             "glm5-next: %d layers (%d MLA, %d KDA, %d dense FFN), n_embd %d, %d heads, vocab %d, experts %d/%d ff %d (x%.1f, limit %.0f), "
             "q_lora %d kv_lora %d, indexer %dx%d top %d pool %d, mHC %d (sinkhorn %d)",
             n_layer, nm, n_layer - nm, n_dense, n_embd, n_head, n_vocab, n_expert_used, n_expert, n_ff_exp, w_scale, clamp_exp, q_lora,
             kv_lora, idx_heads, idx_dim, idx_top_k, kpool, hc, sinkhorn);
    return b;
}

} // namespace hyper
