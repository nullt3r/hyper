// hyper engine for qwen4exp (Qwen3.8-Flash-Next) on the tensor-parallel engine: the model's layers (gated attention / gated
// delta net, hyper-connection mixers, PLE n-gram rows, routed + shared experts), its MTP (NextN) block, images (M-RoPE)
#pragma once
#include "tp_engine.h"
#include "model4.h"

#include <unordered_map>

namespace hyper {

using Engine4Options = TpOptions;

class Engine4 : public TpEngine {
public:
    Engine4(const std::string & model_path, const Engine4Options & opt);
    ~Engine4() override;
    const Q4Config & config() const { return cfg_; }
    void reset() override;
    // images: a model with M-RoPE sections (Qwen3-VL style) takes them
    bool set_media(const PromptMedia * m) override { if (m && !has_rpos()) return false; media_ = m; return true; }
    bool has_rpos() const;
    void dump_trace(const std::string & dir);   // HYPER4_TRACE: per-layer residual of the last forward's last row
    // MTP pass over chunk ci of the last multi-chunk forward (tokens/nt/pos: that chunk's MTP inputs) (tests)
    int mtp_draft_chunk(const int * tokens, int nt, int pos, int ci, bool need_draft = true);

private:
    struct DevLayer;
    struct Device;
    Device & dev(int g);
    const Device & dev(int g) const;
    DevLayer & lay(Device & d, int il);   // layer il, the MTP block at n_layer
    // the engine's hooks
    std::unique_ptr<TpDevice> make_device() override;
    void load_device(TpDevice & d) override;
    const GGUF & expert_source(int il) const override { return il < cfg_.n_layer ? *gguf_ : *mtp_g_; }
    void after_load() override;
    void record_main(int gi, int nt, int kind) override;
    void record_restore(int gi, int keep) override;
    bool mtp_direct() const override { return mtp_cache_only_; }
    size_t snap_floats(int gi) const override;
    void snap_copy(std::vector<float *> & h, bool to_host) override;
    std::vector<ConvPark<TpSnap>::Dev> park_devs() const override;
    void embed(const int * tokens, int nt, int pos, int chunk = 0) override;
    const std::vector<int> * prompt_key(const std::vector<int> & prompt) override;
    void prompt_begin(const std::vector<int> & prompt, int s) override;
    void prompt_end() override { mrows_.clear(); }
    int mtp_prompt(const std::vector<int> & prompt, int c0, const int * lens, int nk, int next) override;
    int mtp_update(const int * tokens, int nt, int pos, bool draft) override { return mtp_draft(tokens, nt, pos - 1, draft); }
    int mtp_chain(int token, int pos, bool from_main) override;
    // the model
    void load_layer(Device & dev, DevLayer & L, int il);
    void embed_tok(const int * tokens, int nt, float * dst);
    void image_rows(int pos, int nt, float * dst);   // image positions of [pos, pos + nt): the encoder's rows
    int rpos_cells() const { return opt_.max_pos + 4 * MAX_NT + 8; }
    void set_rpos(int s, int P, const PromptMedia * md);
    // MTP over (tokens[t], main hidden row t) at positions pos + t; returns the draft after the last row (need_draft
    // false: the rows only enter the block's cache, -1)
    int mtp_draft(const int * tokens, int nt, int pos, bool need_draft = true);

    Q4Config cfg_;
    const GGUF * src_ = nullptr;     // file the layer loaders read from
    std::vector<int> seq_;           // tokens by position (PLE n-gram history)
    const PromptMedia * media_ = nullptr;           // images of the next generate()
    const PromptMedia * md_ = nullptr;              // this generate()'s
    std::unordered_map<int, const float *> mrows_;  // this prompt's image rows by position
    float * h_ple_ = nullptr;        // pinned [mc_max][R][ple heads * ple dim]
    float * h_membd_ = nullptr;      // pinned [R][n_embd]: MTP input embeddings
    bool mtp_cache_only_ = false;    // this MTP pass: rows into the block's cache only (no attention output, FFN, head)
    bool mtp_whole_norm_ = false;    // MTP hidden norm over all hc streams (ik layout) instead of per stream
    bool allrows_ = false, grouped_decode_ = false;
    int mtp_src_ = 0;                // MTP pass: residual rows of chunk mtp_src_ of the last multi-chunk forward
    const uint8_t * ple_tab_ = nullptr;   // the PLE n-gram table copied to RAM (nullptr: read through the mmapped file)
};

} // namespace hyper
