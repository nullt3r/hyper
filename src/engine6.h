// hyper engine for mimo2 (MiMo-V2.6-Flash): attention tensor-parallel over the GPUs by kv heads (global layers' full caches,
// sliding-window layers' ring caches), the dense FFN split by hidden rows, experts split between the GPUs and the CPU (as
// Engine5: adaptive placement, elastic extra slots, prefill streaming).
#pragma once
#include "cpu_moe.h"
#include "engine4.h"   // Barrier4
#include "gguf.h"
#include "kernels4.cuh"
#include "kernels5.cuh"
#include "kernels6.cuh"
#include "llm.h"
#include "vmm.h"
#include "model6.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <thread>
#include <vector>

namespace hyper {

struct Engine6Options {
    int n_devices = 3;
    int max_pos = 8192;
    float gpu_expert_frac = 1.0f;
    float vram_reserve_gib = 0.8f;
    std::string mtp_path;           // GGUF with the NextN blocks blk.<n_layer>.. (the mtp-*.gguf sidecar): MTP drafts
    int n_draft = 3;                // drafts per step: MTP (with mtp_path), else prompt lookup (0: off)
    bool stream_experts = true;
    int cpu_threads = 30;
    bool prompt_cache = false;      // resume a prompt that extends the cached conversation (no snapshots: sliding-window rings)
    int max_snapshots = 48;
    bool answer_reuse = false;
};

class Engine6 : public LLM {
public:
    Engine6(const std::string & model_path, const Engine6Options & opt);
    ~Engine6();

    std::vector<int> forward(const int * tokens, int nt, int pos);
    int forward_multi(const int * tokens, const int * lens, int nck, int pos);
    int prefill(const int * tokens, int n, int pos);
    void get_logits(int t, std::vector<float> & out);
    void reset();
    const MimoConfig & config() const { return cfg_; }
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                              const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) override;
    void set_snapshot_token(int) override {}
    void set_prefill_progress(std::function<void(int, int, int)> fn) override { prefill_cb_ = std::move(fn); }
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return std::min(opt_.n_draft, MAX_NT - 1); }
    bool has_mtp() const override { return mtp_g_ != nullptr; }
    void save_expert_stats(const std::string & path);
    void set_draft(int k, double pmin) { opt_.n_draft = std::max(0, k); mtp_pmin_ = pmin; }
    void reset_cache() override { hist_.clear(); }
    // images: their rows replace the input rows at their positions (plain positions: mimo2 ropes text and image tokens alike);
    // the prompt cache compares keys (an image's hash at its positions)
    bool set_media(const PromptMedia * m) override { media_ = m; return true; }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void load_layer(Device & dev, DevLayer & L, int il);
    void load_experts(int il, const std::vector<int> & quota);
    void upload_stage(Device & d, int il, int sb, int p = 0, int np = 1);
    void rebuild_stream(int il);
    void rebalance(int max_swaps);
    void record_main(int gi, int nt);
    // MTP: NextN head `head` over rows at positions q0.. (tokens embedded in chunk 0; hidden rows: the first nmain the main
    // model's, the rest the chained outputs of heads 0..); all rows enter the head's cache, `full`: the last row goes on
    // through attention, FFN and the head (draft in h_res_, its hidden in mchain[head])
    void record_mtp(int gi, int head, int nrows, int nmain, bool full);
    void mtp_pass(int head, const int * tokens, int nrows, int q0, int nmain, bool full);
    int mtp_result();
    void build_graphs();
    void embed(const int * tokens, int nt, int chunk = 0, int pos = -1);   // (pos >= 0: image rows from the prompt media)
    void run(int nt);
    int nl_tot() const { return cfg_.n_layer; }
    void * host_huge_alloc(size_t bytes);
    int sample_row(int t, const SamplingParams & sp, int64_t pos);
    bool is_moe(int il) const { return cfg_.moe[il] != 0; }
    int ring_ = 4096;              // sliding-window caches: rows per kv head (position % ring_)
    int ns_global_ = 64;           // decode: key splits of a global layer's attention (tensor-core kernel: 64 measured best)
    std::vector<int> hist_;        // tokens (or image keys) whose cache rows are in place (positions 0..)
    const PromptMedia * media_ = nullptr;          // images of the next generate()
    std::map<int, const float *> mrows_;           // this generate's image rows by position
    std::function<void(int, int, int)> prefill_cb_;
    float * h_topk_ = nullptr;
    uint64_t samp_seed_ = 0;
    std::vector<std::pair<void *, size_t>> host_bufs_;

    Engine6Options opt_;
    std::unique_ptr<GGUF> gguf_, mtp_g_;
    int n_mtp_ = 0;                // NextN heads (chained: draft k by head k)
    static constexpr int HR = 4096;     // main hidden rows kept by position (MTP inputs)
    static constexpr int MTPR = 160;    // rows of an MTP pass (the window before a draft position + the chained rows)
    double mtp_p_ = 1.0, mtp_pmin_ = 0.5;   // the last draft's probability; drafts while their run's probability >= pmin
    float * h_mtopk_ = nullptr;
    float samp_invT_ = 0.0f;
    SamplingParams samp_;
    int64_t mtp_pos_ = 0;
    MimoConfig cfg_;
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
    int mc_max_ = 4;
    int mc_n_ = 1, mc_nt_[8] = {}, mc_pos_[8] = {};
    struct ExpertHost {
        const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr;
        size_t gb = 0, db = 0;
        std::vector<int> owner;
        std::vector<std::vector<int>> slot;
        std::vector<double> score;
        std::vector<uint64_t> last_count;
        std::vector<char> pend;
        bool tables_dirty = false, stream_dirty = false;
    };
    std::vector<ExpertHost> ehost_;
    bool adapt_ = true, prompt_routed_ = false;
    double adapt_decay_ = 0.92, prompt_weight_ = 0.25;
    int adapt_every_ = 16, adapt_budget_ = 64;
    double adapt_min_ = 6.0, adapt_ratio_ = 1.25;
    double t_rebalance_ = 0;
    int steps_ = 0, n_swaps_ = 0;
    double acc_rate_ = 0.5;
    int up_parts_ = 1;
    int stream_min_ = 280;
    int * h_ids_ = nullptr;
    // elastic placement (as Engine5): the global layers' caches map only the positions in use, extra expert slots in the rest
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
