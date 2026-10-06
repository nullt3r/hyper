#include "model.h"

#include <cstdio>

namespace hyper {

Qwen35Config Qwen35Config::from_gguf(const GGUF & g) {
    if (g.arch() != "qwen35") throw std::runtime_error("expected qwen35 model, got " + g.arch());
    const std::string a = "qwen35.";
    Qwen35Config c;
    const int n_block = (int) g.get_int(a + "block_count");
    c.n_layer = n_block - (int) g.get_int(a + "nextn_predict_layers", 0);
    c.n_embd = (int) g.get_int(a + "embedding_length");
    c.n_ff = (int) g.get_int(a + "feed_forward_length");
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
    c.n_vocab = (int) g.need("output.weight").ne[1];
    return c;
}

std::string Qwen35Config::describe() const {
    char buf[512];
    snprintf(buf, sizeof buf,
             "qwen35: %d layers (%d full-attn), embd %d, ff %d, heads %d/%d x %d (rot %d, base %.0f), "
             "gdn: k-heads %d v-heads %d dk %d dv %d conv %d, vocab %d",
             n_layer, n_layer / full_attn_interval, n_embd, n_ff, n_head, n_head_kv, head_dim, n_rot, rope_base,
             ssm_n_group, ssm_dt_rank, ssm_d_state, head_v_dim(), ssm_conv, n_vocab);
    return buf;
}

} // namespace hyper
