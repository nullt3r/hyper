// hyper engine for glm5-next (GLM-5.3-Flash): dense parts tensor-parallel over the GPUs (KDA / MLA heads, FFN hidden),
// mHC mixers, the MLA latent cache and the indexer replicated, experts split between the GPUs and the CPU.
#pragma once
#include "cpu_moe.h"
#include "engine4.h"   // Barrier4
#include "gguf.h"
#include "kernels4.cuh"
#include "kernels5.cuh"
#include "llm.h"
#include "park.h"
#include "model5.h"

#include <atomic>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace hyper {

struct Engine5Options {
    int n_devices = 3;
    int max_pos = 8192;
    float gpu_expert_frac = 1.0f;
    float vram_reserve_gib = 0.8f;
    std::string mtp_path;           // GGUF with the NextN (MTP) block blk.<n_layer> (the original split files): MTP drafts
    int n_draft = 3;                // drafted tokens per step: MTP (with mtp_path), else prompt lookup (0: off)
    bool stream_experts = true;
    int cpu_threads = 30;
    bool prompt_cache = false;
    int max_snapshots = 48;
};

class Engine5 : public LLM {
public:
    Engine5(const std::string & model_path, const Engine5Options & opt);
    ~Engine5();

    std::vector<int> forward(const int * tokens, int nt, int pos);
    // several consecutive prefill chunks (each <= MOE_BULK_ROWS, all streaming) layer by layer: a layer's streamed CPU
    // experts cross PCIe once for all of them; result: the last chunk's next token (as forward)
    int forward_multi(const int * tokens, const int * lens, int nck, int pos);
    int prefill(const int * tokens, int n, int pos);
    void get_logits(int t, std::vector<float> & out);
    void reset();
    const Glm5Config & config() const { return cfg_; }
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                              const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) override;
    void set_snapshot_token(int tok) override { snap_tokens_.push_back(tok); }   // every message-start token given
    void set_prefill_progress(std::function<void(int, int, int)> fn) override { prefill_cb_ = std::move(fn); }
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return std::min(opt_.n_draft, MAX_NT - 1); }
    bool has_mtp() const override { return mtp_g_ != nullptr; }
    void save_expert_stats(const std::string & path);
    // MTP drafts per step (<= MAX_NT - 1; the engine was built with drafts) and the run-probability floor (tests)
    void set_draft(int k, double pmin) { opt_.n_draft = std::max(1, k); mtp_pmin_ = pmin; }
    void reset_cache() override { hist_.clear(); for (auto & s : snaps_) snap_pool_.push_back(s.h); snaps_.clear(); park_.clear(snap_pool_); }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void load_layer(Device & dev, DevLayer & L, int il);
    void load_experts(int il, const std::vector<int> & quota);
    void upload_stage(Device & d, int il, int sb);
    void rebuild_stream(int il);
    void rebalance(int max_swaps);
    // kind 0: main model; 1: MTP block over the main model's hidden rows (decode: rows 0..nt-1 of the last verification;
    // mtp_shift_: prefill chunks, row r takes the hidden of row r - 1); 2: MTP block chained on its own last output
    void record_main(int gi, int nt, int kind = 0);
    void record_restore(int gi, int keep);
    void restore(int keep);   // keep the first `keep` rows of the last verification
    void build_graphs();
    void embed(const int * tokens, int nt, int chunk = 0);
    void run(int nt, int kind = 0);
    int mtp_result();
    int mtp_update(const int * tokens, int nt, int pos);   // kind 1 (decode): draft after the last row
    int mtp_chain(int token, int pos, bool from_prev = false);   // kind 2 (from_prev: on the main hidden before pos)
    void mtp_prefill(int nck);   // kind 1 shifted over the chunks of the last bulk forward (embeddings / positions kept)
    int nl_tot() const { return cfg_.n_layer + (mtp_g_ ? 1 : 0); }   // layers incl. the MTP block (index n_layer)
    const GGUF & src(int il) const { return il >= cfg_.n_layer ? *mtp_g_ : *gguf_; }
    void * host_huge_alloc(size_t bytes);
    int sample_row(int t, const SamplingParams & sp);
    struct Snap { int pos; std::vector<float *> h; };
    size_t snap_floats(int gi) const;
    void snap_copy(Snap & sn, bool to_host);
    void take_snapshot(int pos);
    std::vector<ConvPark<Snap>::Dev> park_devs() const;
    ConvPark<Snap> park_;
    bool is_moe(int il) const { return il >= cfg_.n_dense; }
    std::vector<int> hist_;
    std::vector<Snap> snaps_;
    std::vector<std::vector<float *>> snap_pool_;
    std::vector<int> snap_tokens_;
    std::function<void(int, int, int)> prefill_cb_;
    float * h_topk_ = nullptr;
    std::mt19937_64 rng_;
    std::vector<std::pair<void *, size_t>> host_bufs_;

    Engine5Options opt_;
    std::unique_ptr<GGUF> gguf_, mtp_g_;
    bool mtp_shift_ = false;
    double mtp_p_ = 1.0;   // the MTP block's softmax probability of its last draft
    // drafts while their run's MTP probability stays >= this (HYPER5_MTP_PMIN; 0: the acceptance-driven count). 0.85 measured
    // best: at 16k +16 % (vs +10 % one fixed draft, +0.5 % three), Codex replay +10 % greedy / +11 % at temperature 1
    double mtp_pmin_ = 0.85;
    bool mtp_full_ = false;   // HYPER5_MTP_FULL (test): prompt / kept rows through the whole MTP block, not just into its cache
    Glm5Config cfg_;
    std::vector<std::unique_ptr<Device>> devs_;
    std::unique_ptr<CpuMoe> cpu_;
    std::vector<std::vector<uint64_t>> stats_;
    float * h_embd_ = nullptr;
    int * h_pos_ = nullptr;
    float * h_res_ = nullptr;
    uint2 * ar_ll_ = nullptr;
    CpuMoeRec * cpu_rec_ = nullptr;
    CpuMoeOut * cpu_out_ = nullptr;
    CpuMoeBulk * cpu_bulk_ = nullptr;
    CpuMoeBulkOut * cpu_bulk_out_ = nullptr;
    half * h_stage_ = nullptr;
    std::unique_ptr<Barrier4> barrier_;
    unsigned fwd_counter_ = 0;
    bool graphs_ready_ = false;
    bool debug_ = false, nocpu_ = false;
    int last_nt_ = 0;
    int mc_max_ = 4;               // prefill chunks per layer pass (HYPER5_MC; 1: one at a time)
    int mc_n_ = 1, mc_nt_[8] = {}, mc_pos_[8] = {};   // the chunks of the current bulk forward
    // adaptive expert placement (HYPER5_ADAPT=0: off): host copy of every expert, current placement, routing scores
    struct ExpertHost {
        const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr;
        size_t gb = 0, db = 0;
        std::vector<int> owner;               // [E]: GPU or CPU_OWNER
        std::vector<std::vector<int>> slot;   // [ndev][E]
        std::vector<double> score;            // decayed routing counts
        std::vector<uint64_t> last_count;     // CPU side's counts at the last rebalance
        bool tables_dirty = false, stream_dirty = false;
    };
    std::vector<ExpertHost> ehost_;
    bool adapt_ = true, prompt_routed_ = false;
    // measured on a 20k-prompt / 1500-token generation: the placement is worth far more than the swaps cost
    // (15.5 t/s without adaptation, 20.9 with these settings, 27 once adapted), so swap eagerly
    double adapt_decay_ = 0.92;   // per rebalance (every adapt_every_ decode steps)
    double prompt_weight_ = 0.25; // a prompt token's routing counts less than a generated one's (different content: tools, docs)
    int adapt_every_ = 16, adapt_budget_ = 64;
    double adapt_min_ = 6.0, adapt_ratio_ = 1.25;   // swap when score_in >= ratio * score_out + min
    double t_rebalance_ = 0;      // ms spent in rebalance (decode stalls)
    int steps_ = 0, n_swaps_ = 0;
    double acc_rate_ = 0.5;       // speculation: running fraction of drafted tokens accepted
    int stream_min_ = 280;        // prefill chunks this long stream the CPU experts to the GPUs; shorter ones use the CPU
    int * h_ids_ = nullptr;       // pinned [n_layer][R][K]: prefill routing from GPU 0
    // HYPER5_PREDSTAT (decode, measurement): layer l+1's router applied to layer l's router input on GPU 0; how many of
    // the CPU experts layer l+1 then really routes to were predicted
    std::vector<int> zc_blocks_;   // HYPER5_ZC "cpu,g0,g1,g2": hidden 256-blocks of the CPU experts per side (decode)
    int pred_k_ = 0;
    int * h_pred_ = nullptr;      // pinned [n_layer][MOE_MAX_USED]
    uint64_t pred_hit_ = 0, pred_tot_ = 0, pred_cpu_ = 0, pred_n_ = 0;
};

} // namespace hyper
