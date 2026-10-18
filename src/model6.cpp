#include "model6.h"

#include <cstdio>
#include <stdexcept>

namespace hyper {

MimoConfig MimoConfig::from_gguf(const GGUF & g) {
    if (g.arch() != "mimo2") throw std::runtime_error("expected mimo2 model, got " + g.arch());
    const std::string a = "mimo2.";
    MimoConfig c;
    const int blocks = (int) g.get_int(a + "block_count");
    c.n_nextn = (int) g.get_int(a + "nextn_predict_layers", 0);
    c.n_layer = blocks - c.n_nextn;
    c.n_embd = (int) g.get_int(a + "embedding_length");
    c.n_head = (int) g.get_int(a + "attention.head_count");
    c.dk = (int) g.get_int(a + "attention.key_length");
    c.dv = (int) g.get_int(a + "attention.value_length");
    c.n_rot = (int) g.get_int(a + "rope.dimension_count");
    c.window = (int) g.get_int(a + "attention.sliding_window");
    c.rope_base = (float) g.get_float(a + "rope.freq_base");
    c.rope_base_swa = (float) g.get_float(a + "rope.freq_base_swa", c.rope_base);
    c.rms_eps = (float) g.get_float(a + "attention.layer_norm_rms_epsilon");
    c.v_scale = (float) g.get_float(a + "attention.value_scale", 1.0);
    c.n_ff = (int) g.get_int(a + "feed_forward_length");
    c.n_expert = (int) g.get_int(a + "expert_count");
    c.n_expert_used = (int) g.get_int(a + "expert_used_count");
    c.n_ff_exp = (int) g.get_int(a + "expert_feed_forward_length");
    c.w_scale = (float) g.get_float(a + "expert_weights_scale", 1.0);
    if (c.w_scale == 0.0f) c.w_scale = 1.0f;   // (llama.cpp: 0 = no scaling)
    if (g.get_int(a + "expert_gating_func", 2) != 2) throw std::runtime_error("mimo2: only sigmoid gating");
    const auto kv = g.get_int_arr(a + "attention.head_count_kv");
    const auto pat = g.get_int_arr(a + "attention.sliding_window_pattern");
    c.n_kv.assign(blocks, 0); c.swa.assign(blocks, 0); c.moe.assign(blocks, 0);
    for (int il = 0; il < blocks; ++il) {
        c.n_kv[il] = (int) (il < (int) kv.size() ? kv[il] : kv[0]);
        c.swa[il] = il < (int) pat.size() ? (int) pat[il] : 0;
        c.moe[il] = g.tensor("blk." + std::to_string(il) + ".ffn_gate_inp.weight") != nullptr;
    }
    const GTensor * out = g.tensor("output.weight");
    c.n_vocab = out ? (int) out->ne[1] : (int) g.get_int(a + "vocab_size", 0);
    if (c.dk % 32 || c.dv != 128 || c.dk > 256 || c.n_rot > c.dk || c.n_rot % 2 || c.n_embd > 4096 || c.n_embd % 256)
        throw std::runtime_error("mimo2: unsupported dimensions");
    for (int il = 0; il < blocks; ++il)
        if (c.n_kv[il] <= 0 || c.n_head % c.n_kv[il]) throw std::runtime_error("mimo2: heads not a multiple of the kv heads");
    return c;
}

std::string MimoConfig::describe() const {
    int ns = 0, nm = 0;
    for (int il = 0; il < n_layer; ++il) { ns += swa[il]; nm += moe[il]; }
    char b[512];
    snprintf(b, sizeof b,
             "mimo2: %d layers (%d sliding-window %d, %d global; %d MoE), n_embd %d, %d heads (k %d / v %d, rope %d, base %.0f / %.0f), "
             "vocab %d, experts %d/%d ff %d, dense ff %d, value scale %.3f, nextn %d",
             n_layer, ns, window, n_layer - ns, nm, n_embd, n_head, dk, dv, n_rot, rope_base, rope_base_swa, n_vocab, n_expert_used, n_expert,
             n_ff_exp, n_ff, v_scale, n_nextn);
    return b;
}

} // namespace hyper
