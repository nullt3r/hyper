#include "model4.h"

#include <cstdio>

namespace hyper {

Q4Config Q4Config::from_gguf(const GGUF & g) {
    if (g.arch() != "qwen4exp") throw std::runtime_error("expected qwen4exp model, got " + g.arch());
    const std::string a = "qwen4exp.";
    Q4Config c;
    c.n_layer = (int) g.get_int(a + "block_count") - (int) g.get_int(a + "nextn_predict_layers", 0);
    c.n_embd = (int) g.get_int(a + "embedding_length");
    c.n_head = (int) g.get_int(a + "attention.head_count");
    c.n_head_kv = (int) g.get_int(a + "attention.head_count_kv");
    c.head_dim = (int) g.get_int(a + "attention.key_length");
    c.n_rot = (int) g.get_int(a + "rope.dimension_count");
    c.rope_base = (float) g.get_float(a + "rope.freq_base");
    c.rms_eps = (float) g.get_float(a + "attention.layer_norm_rms_epsilon");
    c.full_attn_interval = (int) g.get_int(a + "full_attention_interval");
    c.ssm_conv = (int) g.get_int(a + "ssm.conv_kernel");
    c.ssm_d_state = (int) g.get_int(a + "ssm.state_size");
    c.ssm_n_group = (int) g.get_int(a + "ssm.group_count");
    c.ssm_dt_rank = (int) g.get_int(a + "ssm.time_step_rank");
    c.ssm_d_inner = (int) g.get_int(a + "ssm.inner_size");
    c.n_expert = (int) g.get_int(a + "expert_count");
    c.n_expert_used = (int) g.get_int(a + "expert_used_count");
    c.n_ff_exp = (int) g.get_int(a + "expert_feed_forward_length");
    c.n_ff_shexp = (int) g.get_int(a + "expert_shared_feed_forward_length");
    c.hc = (int) g.get_int(a + "hyper_connection.count");
    c.hc_lr = (int) g.get_int(a + "hyper_connection.low_rank");
    c.idx_n_head = (int) g.get_int(a + "attention.indexer.head_count");
    c.idx_dim = (int) g.get_int(a + "attention.indexer.key_length");
    c.idx_top_k = (int) g.get_int(a + "attention.indexer.top_k");
    for (int64_t r : g.get_int_arr(a + "attention.compress_ratios")) {
        c.compress.push_back((int) r);
        if (r) c.kpool = (int) r;
    }
    c.compress.resize(c.n_layer + 1, 0);
    if (g.has(a + "ple.layers")) {
        const auto layers = g.get_int_arr(a + "ple.layers");
        if (layers.size() != 1) throw std::runtime_error("qwen4exp: exactly one PLE layer supported");
        c.ple_layer = (int) layers[0];
        c.ple_ngram = (int) g.get_int(a + "ple.ngram_size");
        c.ple_heads_per_ngram = (int) g.get_int(a + "ple.heads_per_ngram");
        c.ple_conv = (int) g.get_int(a + "ple.conv_kernel");
        c.ple_eos = (int) g.get_int(a + "ple.eos_token_id");
        c.ple_dim = (int) g.get_int(a + "embedding_length_per_layer_input");
        for (int64_t v : g.get_int_arr(a + "ple.layer_multipliers")) c.ple_mult.push_back((uint64_t) v);
        for (int64_t v : g.get_int_arr(a + "ple.head_offsets")) c.ple_offsets.push_back((uint64_t) v);
        for (int64_t v : g.get_int_arr(a + "ple.head_vocab_sizes")) c.ple_vocab.push_back((uint64_t) v);
    }
    c.n_vocab = (int) g.need("output.weight").ne[1];
    return c;
}

std::string Q4Config::describe() const {
    char buf[640];
    snprintf(buf, sizeof buf,
             "qwen4exp: %d layers (%d attn, kpool %d top %d), embd %d x hc %d (lr %d), heads %d/%d x %d (rot %d), "
             "gdn k%d v%d, moe %d/%d x %d + shared %d, ple layer %d (%d heads x %d), vocab %d",
             n_layer, n_layer / full_attn_interval, kpool, idx_top_k, n_embd, hc, hc_lr, n_head, n_head_kv, head_dim, n_rot,
             ssm_n_group, ssm_dt_rank, n_expert_used, n_expert, n_ff_exp, n_ff_shexp, ple_layer, ple_n_heads(), ple_dim, n_vocab);
    return buf;
}

} // namespace hyper
