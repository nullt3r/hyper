// hyper engine for glm5-next (GLM-5.3-Flash): dense parts tensor-parallel over the GPUs (KDA / MLA heads, FFN hidden),
// mHC mixers, the MLA latent cache and the indexer replicated, experts split between the GPUs and the CPU.
#pragma once
#include "cpu_moe.h"
#include <cstdlib>
#include "engine4.h"   // Barrier4
#include "gguf.h"
#include "kernels4.cuh"
#include "kernels5.cuh"
#include "llm.h"
#include "park.h"
#include "vmm.h"
#include "model5.h"

#include <atomic>
#include <functional>
#include <memory>
#include <random>
#include <thread>
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
    bool answer_reuse = false;   // the end-of-answer snapshot serves a next request that repeats the answer (its K/V and state
                                 // as decoding computed them, not re-prefilled)
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
    CpuMoe * cpu_moe() { return cpu_.get(); }   // (tests)
    const Glm5Config & config() const { return cfg_; }
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                              const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) override;
    void set_snapshot_token(int tok) override { snap_tokens_.push_back(tok); park_.msg_tokens.push_back(tok); }   // every message-start token given
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
    void upload_stage(Device & d, int il, int sb, int p = 0, int np = 1);
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
    void draft_at(int64_t pos) { mtp_pos_ = pos; }   // the next MTP pass drafts the token at pos
    int mtp_update(const int * tokens, int nt, int pos);   // kind 1 (decode): draft after the last row
    int mtp_chain(int token, int pos, bool from_prev = false);   // kind 2 (from_prev: on the main hidden before pos)
    void mtp_prefill(int nck);   // kind 1 shifted over the chunks of the last bulk forward (embeddings / positions kept)
    int nl_tot() const { return cfg_.n_layer + (mtp_g_ ? 1 : 0); }   // layers incl. the MTP block (index n_layer)
    const GGUF & src(int il) const { return il >= cfg_.n_layer ? *mtp_g_ : *gguf_; }
    void * host_huge_alloc(size_t bytes);
    int sample_row(int t, const SamplingParams & sp, int64_t pos);   // pos: the sampled token's position
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
    uint64_t samp_seed_ = 0;   // this request's sampling noise (sampling.h gumbel_noise)
    float samp_invT_ = 0.0f;   // MTP drafts: 1 / temperature when sampling (coupled to the noise), 0: greedy drafts
    SamplingParams samp_;      // this request's sampling (the drafts' truncation)
    int64_t mtp_pos_ = 0;      // the position of the token the next MTP pass drafts
    // CPU load gate of the MTP drafts: EMA of row 0's CPU-held experts per MoE layer and step; drafts are verified only while
    // it stays under mtp_cpu_max_ (a verification row reads its own CPU experts from RAM: worth it when the GPUs pace a step)
    double cpu_ema_ = 8.0, mtp_cpu_max_ = 1.5;
    // decode graph launches for devices 1.. from their own threads, at the same time as device 0's: a cudaGraphLaunch of a
    // decode graph costs ~1 ms of host time (thousands of nodes), and one after another the last device started ~2.5 ms
    // after the first, which waited for it in the first allreduce of every token -- under nsys only (HYPER5_PAR_LAUNCH=1: on)
    struct Launcher { std::thread th; std::atomic<unsigned> go{0}, done{0}; cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr; };
    std::vector<std::unique_ptr<Launcher>> launchers_;
    std::atomic<bool> launch_stop_{false};
    double t_launch_[8] = {}; uint64_t n_launch_ = 0;   // (HYPER5_LAUNCHPROF: host time per launch and device)
    void launch_all(int nt, int kind);
    float * h_mtopk_ = nullptr;   // pinned [ndev][TOPK][2]: the MTP head's candidates per vocab slice (coupled drafts)
    std::vector<std::pair<void *, size_t>> host_bufs_;
    std::vector<void *> host_reg_;   // host_bufs_ registered with CUDA (unregistered before the unmap)

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
    CpuMoeRec * cpu_rec_ = nullptr, * cpu_pred_ = nullptr, * cpu_pred1_ = nullptr;   // pred1: the early (previous layer's) prediction   // pred: HYPER5_PRED=2's routing predictions per layer (ids only)
    const int pred_mode_ = getenv("HYPER5_PRED") ? atoi(getenv("HYPER5_PRED")) : 0;
    // HYPER5_PRED_EARLY=1: also the next layer's prediction during this one, for the CpuMoe helpers (measured slower on GLM)
    const bool pred_early_ = getenv("HYPER5_PRED_EARLY") && atoi(getenv("HYPER5_PRED_EARLY")) != 0;
    const bool pstream_prio_ = getenv("HYPER5_PSTREAM_PRIO") && atoi(getenv("HYPER5_PSTREAM_PRIO")) != 0;
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
        std::vector<char> pend;               // elastic: being copied into an extra slot (a CPU expert until applied)
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
    double t_ld_[4] = {0, 0, 0, 0};   // load: host copies (file -> huge pages), collapse, cudaHostRegister, rest of the layer (s)
    int steps_ = 0, n_swaps_ = 0;
    double acc_rate_ = 0.5;       // speculation: running fraction of drafted tokens accepted
    int up_parts_ = 1;            // prefill expert uploads: parts per half, each after an allreduce's downloads (HYPER5_UP_PARTS; 0: at once)
    // HYPER5_STREAM_CPU (tokens, 0 = off): a streamed single chunk of nt tokens leaves min(stream_cpu_max_, c / nt) of the CPU
    // experts (the coldest) on the CPU
    double stream_cpu_ = 120.0, stream_cpu_max_ = 0.5;
    double stream_cpu_built_ = -1.0;   // (the share the stream lists were built with)
    int stream_min_ = 280;        // prefill chunks this long stream the CPU experts to the GPUs; shorter ones use the CPU
    int * h_ids_ = nullptr;       // pinned [n_layer][R][K]: prefill routing from GPU 0
    FILE * route_trace_ = nullptr;   // HYPER5_ROUTE_TRACE=file: expert ids of every routed row (no text), for placement studies
    template <typename F> void trace_routing(int type, int nrows, F && id);
    // HYPER5_PREDSTAT (decode, measurement): layer l+1's router applied to layer l's router input on GPU 0; how many of
    // the CPU experts layer l+1 then really routes to were predicted
    std::vector<int> zc_blocks_;   // HYPER5_ZC "cpu,g0,g1,g2": hidden 256-blocks of the CPU experts per side (decode)
    int pred_k_ = 0;
    int * h_pred_ = nullptr;      // pinned [n_layer][MOE_MAX_USED]
    uint64_t pred_hit_ = 0, pred_tot_ = 0, pred_cpu_ = 0, pred_n_ = 0;
    // elastic placement (as Engine4; HYPER5_ELASTIC=0: off): the MLA latent caches and pooled indexer keys (every GPU holds
    // them) map only the positions in use; the rest of their full-context memory holds extra expert slots
    bool elastic_ = false, elastic_log_ = false;
    int kv_pos_ = 0, kv_step_ = 8192, elastic_fill_ = 1 << 20;
    double elastic_min_ = 1.0;
    template <typename T> T * kv_alloc(Device & dev, int heads, size_t pitch_elems, size_t row_elems, int div);
    size_t lend_free(const Device & d) const;
    void ensure_kv(int need);
    void shrink_kv(int keep);
    void evict_extra(Device & d, size_t bytes);
    int fill_extra(int budget);
    struct Fill { int il, e, g, k; bool copied; };
    std::vector<Fill> fill_list_;
    std::thread fill_th_;
    std::atomic<bool> fill_busy_{false}, fill_cancel_{false};
    void fill_apply();
    void fill_wait(bool cancel);
    int req_steps_ = 0;
    long dec_steps_ = 0, fill_due_ = 0;
};

} // namespace hyper
