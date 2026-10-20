#include "model7.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace hyper {

Ds4Config Ds4Config::from_gguf(const GGUF & g) {
    if (g.arch() != "deepseek41") throw std::runtime_error("expected deepseek41 model, got " + g.arch());
    const std::string a = "deepseek41.";
    Ds4Config c;
    c.n_layer = (int) g.get_int(a + "block_count");
    c.n_embd = (int) g.get_int(a + "embedding_length");
    c.n_vocab = (int) g.need("output.weight").ne[1];
    c.n_head = (int) g.get_int(a + "attention.head_count");
    c.head_dim = (int) g.get_int(a + "attention.key_length");
    c.rd = (int) g.get_int(a + "rope.dimension_count");
    c.q_lora = (int) g.get_int(a + "attention.q_lora_rank");
    c.o_groups = (int) g.get_int(a + "attention.output_group_count");
    c.o_lora = (int) g.get_int(a + "attention.output_lora_rank");
    c.win = (int) g.get_int(a + "attention.sliding_window");
    c.eps = (float) g.get_float(a + "attention.layer_norm_rms_epsilon");
    c.hc = (int) g.get_int(a + "hyper_connection.count");
    c.sinkhorn = (int) g.get_int(a + "hyper_connection.sinkhorn_iterations");
    c.hc_eps = (float) g.get_float(a + "hyper_connection.epsilon");
    const auto ratios = g.get_int_arr(a + "attention.compress_ratios");
    c.ratio.assign(c.n_layer, 0);
    for (int il = 0; il < c.n_layer; ++il) c.ratio[il] = (int) ratios.at(il);
    c.kv_src.assign(c.n_layer, 0);
    c.ix_src.assign(c.n_layer, 0);
    for (auto l : g.get_int_arr(a + "attention.kv_source_layers")) c.kv_src.at(l) = 1;
    for (auto l : g.get_int_arr(a + "attention.indexer.source_layers")) c.ix_src.at(l) = 1;
    c.kv_of.assign(c.n_layer, -1);
    c.ix_of.assign(c.n_layer, -1);
    int kv = -1, ix = -1;
    for (int il = 0; il < c.n_layer; ++il) {
        if (c.kv_src[il]) kv = il;
        if (c.ix_src[il]) ix = il;
        if (c.ratio[il] > 0) {
            if (kv < 0 || ix < 0) throw std::runtime_error("deepseek41: a compressing layer before its sources");
            if (c.ratio[kv] != c.ratio[il]) throw std::runtime_error("deepseek41: a layer's ratio differs from its KV source's");
            c.kv_of[il] = kv;
            c.ix_of[il] = ix;
        }
    }
    c.cand_src = (int) g.get_int(a + "attention.indexer.candidate_source_layer", -1);
    c.cand_topk = (int) g.get_int(a + "attention.indexer.candidate_top_k_blocks");
    c.cand_blk = (int) g.get_int(a + "attention.indexer.candidate_block_size");
    c.ix_heads = (int) g.get_int(a + "attention.indexer.head_count");
    c.ix_hd = (int) g.get_int(a + "attention.indexer.key_length");
    c.ix_topk = (int) g.get_int(a + "attention.indexer.top_k");
    c.n_expert = (int) g.get_int(a + "expert_count");
    c.n_used = (int) g.get_int(a + "expert_used_count");
    c.ff = (int) g.get_int(a + "expert_feed_forward_length");
    c.w_scale = (float) g.get_float(a + "expert_weights_scale", 1.0);
    c.clamp_exp = (float) g.get_float_arr(a + "swiglu_clamp_exp")[0];
    c.clamp_sh = (float) g.get_float_arr(a + "swiglu_clamp_shexp")[0];
    if (g.get_int(a + "expert_shared_count", 1) != 1) throw std::runtime_error("deepseek41: one shared expert expected");
    c.rope_base = (float) g.get_float(a + "rope.freq_base");
    c.comp_base = (float) g.get_float(a + "attention.compress_rope_freq_base");
    c.yarn_factor = (float) g.get_float(a + "rope.scaling.factor");
    c.beta_fast = (float) g.get_float(a + "rope.scaling.yarn_beta_fast");
    c.beta_slow = (float) g.get_float(a + "rope.scaling.yarn_beta_slow");
    c.orig_ctx = (int) g.get_int(a + "rope.scaling.original_context_length");
    for (auto l : g.get_int_arr(a + "engram.layers")) c.eg_layers.push_back((int) l);
    c.eg_ngram = (int) g.get_int(a + "engram.max_ngram_size");
    c.eg_heads = (int) g.get_int(a + "engram.head_count");
    c.eg_hd = (int) g.get_int(a + "engram.head_dim");
    const auto tm = g.get_int_arr(a + "engram.token_map");
    c.eg_tmap.assign(tm.begin(), tm.end());
    c.eg_pad = c.eg_tmap.at((size_t) g.get_int(a + "engram.pad_token_id"));
    c.eg_mult = g.get_int_arr(a + "engram.multipliers");
    c.eg_primes = g.get_int_arr(a + "engram.primes");
    c.eg_off = g.get_int_arr(a + "engram.offsets");
    const size_t ne = c.eg_layers.size();
    if (c.eg_mult.size() != ne * c.eg_ngram || c.eg_primes.size() != ne * c.eg_cols() || c.eg_off.size() != ne * c.eg_cols())
        throw std::runtime_error("deepseek41: Engram hash tables do not match the layout");
    if (c.hc != 4 || c.head_dim != 512 || c.rd != 64 || c.ix_hd != 128 || c.ix_heads > 32 || c.n_embd > 5120 || c.n_used > 16 ||
        c.n_head * c.head_dim % c.o_groups)
        throw std::runtime_error("deepseek41: unsupported dimensions");
    return c;
}

// engram_hash.Hasher.ids: n-grams 2 .. ngram ending at t (compressed ids, pad before position 0), rolling XOR of id * multiplier,
// each (n-gram, head) column modulo its prime, plus the column's offset
void Ds4Config::eg_rows(int i, const int * toks, int t, int64_t * out) const {
    const int64_t * mult = eg_mult.data() + (size_t) i * eg_ngram;
    const int64_t * pr = eg_primes.data() + (size_t) i * eg_cols(), * off = eg_off.data() + (size_t) i * eg_cols();
    int64_t rolling = 0;
    for (int s = 0; s < eg_ngram; ++s) {
        const int64_t id = t - s >= 0 ? (int64_t) eg_tmap.at((size_t) toks[t - s]) : (int64_t) eg_pad;
        const int64_t prod = id * mult[s];
        rolling = s == 0 ? prod : rolling ^ prod;
        if (s == 0) continue;
        for (int h = 0; h < eg_heads; ++h) {
            const int col = (s - 1) * eg_heads + h;
            out[col] = rolling % pr[col] + off[col];
        }
    }
}

// ref.py freqs_cis in F32: base^(-2i / dim), YaRN-blended (original_seq_len > 0) between freq / factor and freq
std::vector<float> Ds4Config::freqs(bool comp) const {
    const int dim = rd;
    const float base = comp ? comp_base : rope_base;
    std::vector<float> f(dim / 2);
    for (int i = 0; i < dim / 2; ++i) f[i] = 1.0f / std::pow(base, (float) (2 * i) / (float) dim);
    if (comp && orig_ctx > 0) {
        auto corr = [&](double rot) { return dim * std::log(orig_ctx / (rot * 2 * M_PI)) / (2 * std::log((double) base)); };   // (python floats)
        const int low = std::max((int) std::floor(corr(beta_fast)), 0), high = std::min((int) std::ceil(corr(beta_slow)), dim - 1);
        for (int i = 0; i < dim / 2; ++i) {
            const float ramp = std::min(1.0f, std::max(0.0f, ((float) i - low) / std::max((float) (high - low), 1e-3f)));
            const float smooth = 1 - ramp;
            f[i] = f[i] / yarn_factor * (1 - smooth) + f[i] * smooth;
        }
    }
    return f;
}

std::string Ds4Config::describe() const {
    char b[512];
    int nk = 0, ni = 0;
    for (int il = 0; il < n_layer; ++il) { nk += kv_src[il]; ni += ix_src[il]; }
    snprintf(b, sizeof b,
             "deepseek41: %d layers, n_embd %d, %d heads x %d (rope %d), q_lora %d, wo %d groups x %d, window %d, %d KV / %d index sources "
             "(candidates from %d: %d blocks of %d), indexer %dx%d top %d, experts %d/%d ff %d (x%.1f, limit %.0f), hc %d (sinkhorn %d), "
             "Engram at %zu layers (%d-grams x %d heads of %d), vocab %d",
             n_layer, n_embd, n_head, head_dim, rd, q_lora, o_groups, o_lora, win, nk, ni, cand_src, cand_topk, cand_blk, ix_heads, ix_hd,
             ix_topk, n_used, n_expert, ff, w_scale, clamp_exp, hc, sinkhorn, eg_layers.size(), eg_ngram, eg_heads, eg_hd, n_vocab);
    return b;
}

} // namespace hyper
