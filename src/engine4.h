// hyper engine for qwen4exp (Qwen3.8-Flash-Next): dense parts tensor-parallel over the GPUs, hyper-connection
// mixers and routers replicated, experts split between the GPUs (expert parallel) and the CPU.
#pragma once
#include "cpu_moe.h"
#include "gguf.h"
#include "kernels4.cuh"
#include "llm.h"
#include "park.h"
#include "model4.h"
#include "vmm.h"

#include <atomic>
#include <stdexcept>
#include <functional>
#include <random>
#include <thread>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hyper {

struct Engine4Options {
    int n_devices = 3;
    int max_pos = 8192;
    float gpu_expert_frac = 1.0f;   // cap on the fraction of each layer's experts on the GPUs (the rest: CPU)
    float vram_reserve_gib = 0.8f;  // left free on every GPU after the experts
    std::string mtp_path;           // separate NextN (MTP) GGUF: speculative decoding
    int n_draft = 3;
    bool stream_experts = true;     // prefill: copy the CPU experts' weights to the GPUs instead of computing them on the CPU
    int cpu_threads = 30;
    bool prompt_cache = false;      // reuse the common prefix with the previous sequence (recurrent-state snapshots)
    int max_snapshots = 48;
    bool answer_reuse = false;   // the end-of-answer snapshot serves a next request that repeats the answer (its K/V and state
                                 // as decoding computed them, not re-prefilled)
};

// spin barrier for the per-device recording threads of a prefill chunk
class Barrier4 {
public:
    explicit Barrier4(int n) : n_(n) {}
    void wait() {
        const unsigned gen = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            count_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_acq_rel);
        } else {
            while (gen_.load(std::memory_order_acquire) == gen)
                if (aborted_.load(std::memory_order_acquire)) throw std::runtime_error("barrier: another device's recording failed");
        }
    }
    // a recording thread that failed releases the others (they throw instead of waiting forever); reset once all have joined
    void abort() { aborted_.store(true, std::memory_order_release); }
    void reset() { count_.store(0, std::memory_order_relaxed); aborted_.store(false, std::memory_order_release); }
private:
    int n_;
    std::atomic<int> count_{0};
    std::atomic<unsigned> gen_{0};
    std::atomic<bool> aborted_{false};
};

class Engine4 : public LLM {
public:
    Engine4(const std::string & model_path, const Engine4Options & opt);
    ~Engine4();

    // nt tokens at positions pos..: up to MAX_NT through the decode graphs, up to MOE_BULK_ROWS as a prefill chunk
    // (tiled GEMMs, copy-engine allreduce, flash attention; only the last row gets logits).
    // Returns the greedy argmax per token (prefill chunk: last token only, -1 elsewhere)
    std::vector<int> forward(const int * tokens, int nt, int pos);
    // several consecutive prefill chunks (each <= MOE_BULK_ROWS, all streaming) layer by layer: a layer's streamed CPU
    // experts cross PCIe once for all of them; result: the last chunk's next token. mtp_draft_chunk: the MTP pass over
    // chunk ci of the last forward_multi (its final residual rows)
    int forward_multi(const int * tokens, const int * lens, int nck, int pos);
    int mtp_draft_chunk(const int * tokens, int nt, int pos, int ci, bool need_draft = true);
    bool multi_ok(const int * lens, int nck) const;
    // whole prompt in prefill chunks; returns the argmax after the last token
    int prefill(const int * tokens, int n, int pos);
    void get_logits(int t, std::vector<float> & out);
    void reset();
    const Q4Config & config() const { return cfg_; }
    // LLM: greedy / sampled generation (no speculative decoding yet), prompt cache
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                              const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) override;
    void set_snapshot_token(int tok) override { snap_token_ = tok; park_.msg_tokens = {tok}; }
    void reset_cache() override { hist_.clear(); for (auto & s : snaps_) snap_pool_.push_back(s.h); snaps_.clear(); park_.clear(snap_pool_); }
    void set_prefill_progress(std::function<void(int, int, int)> fn) override { prefill_cb_ = std::move(fn); }
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return mtp_g_ ? opt_.n_draft : 0; }
    bool has_mtp() const override { return mtp_g_ != nullptr; }
    // images: a model with M-RoPE sections (Qwen3-VL style) takes them
    bool set_media(const PromptMedia * m) override { if (m && !has_rpos()) return false; media_ = m; return true; }
    bool has_rpos() const;
    // routing statistics: the loaded ones plus what the CPU side has seen since (decode routing of every layer)
    void save_expert_stats(const std::string & path);
    void dump_trace(const std::string & dir);   // HYPER4_TRACE: per-layer residual of the last forward's last row
    // MTP drafts per step and the run-probability floor (tests)
    void set_draft(int k, double pmin) { opt_.n_draft = std::max(1, std::min(k, MAX_NT - 1)); mtp_pmin_ = pmin; }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void load_layer(Device & dev, DevLayer & L, int il);
    void load_experts(int il, const std::vector<int> & quota);
    void upload_stage(Device & d, int il, int p = 0, int np = 1);
    void rebuild_stream(int il);
    void rebalance(int max_swaps);
    void score_prompt(const int * lens, int nck);   // prefill routing of the last bulk pass into the placement scores
    DevLayer & lay(Device & d, int il);   // layer il, the MTP block at n_layer
    void record_main(int gi, int nt, int kind = 0);
    void record_restore(int gi, int keep);
    void build_graphs();
    void embed(const int * tokens, int nt, int pos, int chunk = 0);
    void embed_tok(const int * tokens, int nt, float * dst);
    void image_rows(int pos, int nt, float * dst);   // image positions of [pos, pos + nt): the encoder's rows
    int rpos_cells() const { return opt_.max_pos + 4 * MAX_NT + 8; }
    void set_rpos(int s, int P, const PromptMedia * md);
    void run(int kind, int nt);   // 0 main, 1 MTP, 2 MTP chain, 3 restore (nt = rows kept)
    int mtp_result();
    void draft_at(int64_t pos) { mtp_pos_ = pos; }   // the next MTP pass drafts the token at pos
    // need_draft false: the rows only enter the MTP block's cache (its K / V and indexer keys depend on its input alone)
    int mtp_draft(const int * tokens, int nt, int pos, bool need_draft = true);
    int mtp_chain(int token, int pos);
    void * host_huge_alloc(size_t bytes);   // anonymous, transparent huge pages; freed with the engine
    int sample_row(int t, const SamplingParams & sp, int64_t pos);   // pos: the sampled token's position
    struct Snap { int pos; std::vector<float *> h; };
    size_t snap_floats(int gi) const;
    void snap_copy(Snap & sn, bool to_host);
    void take_snapshot(int pos);
    std::vector<ConvPark<Snap>::Dev> park_devs() const;
    ConvPark<Snap> park_;
    std::vector<int> hist_;
    std::vector<Snap> snaps_;
    std::vector<std::vector<float *>> snap_pool_;
    int snap_token_ = -1;
    std::function<void(int, int, int)> prefill_cb_;
    float * h_topk_ = nullptr;      // pinned [ndev][MAX_NT][TOPK][2]
    uint64_t samp_seed_ = 0;   // this request's sampling noise (sampling.h gumbel_noise)
    float samp_invT_ = 0.0f;   // MTP drafts: 1 / temperature when sampling (coupled to the noise), 0: greedy drafts
    SamplingParams samp_;      // this request's sampling (the drafts' truncation)
    int64_t mtp_pos_ = 0;      // the position of the token the next MTP pass drafts
    float * h_mtopk_ = nullptr;   // pinned [ndev][TOPK][2]: the MTP head's candidates per vocab slice (coupled drafts)
    std::vector<std::pair<void *, size_t>> host_bufs_;

    Engine4Options opt_;
    std::unique_ptr<GGUF> gguf_, mtp_g_;
    const GGUF * src_ = nullptr;     // file the layer loaders read from
    Q4Config cfg_;
    std::vector<std::unique_ptr<Device>> devs_;
    std::unique_ptr<CpuMoe> cpu_;
    std::vector<std::vector<uint64_t>> stats_;   // expert placement statistics [layer][expert]
    std::vector<int> seq_;          // tokens by position (PLE n-gram history)
    const PromptMedia * media_ = nullptr;           // images of the next generate()
    std::unordered_map<int, const float *> mrows_;  // this prompt's image rows by position
    float * h_embd_ = nullptr;      // pinned [MAX_NT][n_embd]
    float * h_ple_ = nullptr;       // pinned [MAX_NT][ple heads * ple dim]
    int * h_pos_ = nullptr;
    float * h_res_ = nullptr;       // pinned [ndev][MAX_NT][2]
    float * h_membd_ = nullptr;     // pinned [R][n_embd]: MTP input embeddings
    float * h_mres_ = nullptr;      // pinned [ndev][4]: MTP draft argmax, slice max logit, slice sum exp
    bool mtp_cache_only_ = false;   // this MTP pass: rows into the block's cache only (no attention output, FFN, head)
    bool mtp_full_ = false;         // HYPER4_MTP_FULL (test): every MTP row through the whole block
    double mtp_p_ = 1.0;            // the MTP block's probability of its last draft
    // drafts while their run's MTP probability stays >= this (HYPER4_MTP_PMIN; 0: acceptance-driven count). 0.5 measured best
    // (9k prompt): Flash-Next 138 -> 142.5 t/s, Uncensored with the base model's NextN head 124 -> 129.5 t/s
    double mtp_pmin_ = 0.5;
    bool mtp_whole_norm_ = false;   // MTP hidden norm over all hc streams (ik layout) instead of per stream
    uint2 * ar_ll_ = nullptr;
    CpuMoeRec * cpu_rec_ = nullptr; // mapped [n_layer]
    CpuMoeOut * cpu_out_ = nullptr; // mapped [n_layer]
    CpuMoeBulk * cpu_bulk_ = nullptr;        // mapped, prefill
    CpuMoeBulkOut * cpu_bulk_out_ = nullptr;
    half * h_stage_ = nullptr;      // pinned DMA allreduce staging [parity][ndev][MOE_BULK_ROWS * n_embd]
    std::unique_ptr<Barrier4> barrier_;
    unsigned fwd_counter_ = 0;
    bool graphs_ready_ = false;
    bool debug_ = false, nocpu_ = false, allrows_ = false, grouped_decode_ = false;
    int last_nt_ = 0;
    int mc_max_ = 4;               // prefill chunks per layer pass (HYPER4_MC; 1: one at a time)
    // adaptive expert placement (as Engine5; HYPER4_ADAPT=0: off): a host copy of every expert (pinned), the current
    // placement, decayed routing scores; the hottest CPU experts trade places with the coldest GPU ones between forwards
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
    std::vector<ExpertHost> ehost_;   // [n_layer + 1]: the MTP block last
    bool adapt_ = true, prompt_routed_ = false;
    double adapt_decay_ = 0.92, prompt_weight_ = 0.25;   // (Engine5's measured settings)
    int adapt_every_ = 16, adapt_budget_ = 64;
    double adapt_min_ = 6.0, adapt_ratio_ = 1.25;
    double t_rebalance_ = 0;
    int steps_ = 0, n_swaps_ = 0;
    int * h_ids_ = nullptr;   // pinned [n_layer + 1][mc_max_][R][K]: prefill routing from GPU 0
    int mc_n_ = 1, mc_nt_[8] = {}, mc_pos_[8] = {};   // the chunks of the current bulk forward
    int mc_last_ = 1;              // chunks of the last forward_multi
    int mtp_src_ = 0;              // MTP pass: residual rows of chunk mtp_src_ of the last multi-chunk forward
    int * h_cpos_ = nullptr;       // pinned: positions of the chunks
    int up_parts_ = 2;       // prefill expert uploads in parts, each after an allreduce's downloads (HYPER4_UP_PARTS; 1: whole)
    int stream_min_ = 256;   // prefill chunks this long stream the CPU experts to the GPUs; shorter ones use the CPU
    // elastic placement (HYPER4_ELASTIC=0: off): the attention caches (K / V, indexer keys) live on the CUDA virtual memory
    // API and map only the positions in use (kv_pos_); the rest of their full-context memory holds extra expert slots
    // (hottest CPU experts, copied from the host copy), handed back to the caches when the context grows (the expert
    // just turns into a CPU expert again). Caches + extra slots never take more than the full-context caches would.
    bool elastic_ = false, elastic_log_ = false;
    int kv_pos_ = 0;            // positions mapped in every attention cache
    int elastic_fill_ = 1 << 20;   // extra-slot fills per background batch (HYPER4_ELASTIC_FILL)
    int kv_step_ = 8192;        // the caches grow in steps of this many positions
    double elastic_min_ = 1.0;  // fills: any CPU expert the recent routing used (no exchange: only the copy costs)
    template <typename T> T * kv_alloc(Device & dev, int heads, size_t pitch_elems, size_t row_elems, int div);
    size_t lend_free(const Device & d) const;   // bytes of the full-context cache size neither mapped nor lent
    void ensure_kv(int need);   // map the caches up to `need` positions (evicting extra experts as needed)
    void shrink_kv(int keep);   // a shorter conversation: release cache memory beyond `keep` for experts
    void evict_extra(Device & d, size_t bytes);
    int fill_extra(int budget);
    // fills run in the background (a host thread maps the slots and copies on each GPU's fill stream; the CPU keeps
    // computing those experts meanwhile); the copied experts take over at the next forward boundary
    struct Fill { int il, e, g, k; bool copied; };
    std::vector<Fill> fill_list_;
    std::thread fill_th_;
    std::atomic<bool> fill_busy_{false}, fill_cancel_{false};
    void fill_apply();             // (between forwards) the copied experts take over; the rest of a cancelled batch drops
    void fill_wait(bool cancel);   // wait for (or cancel) the running batch, then fill_apply
    int req_steps_ = 0;            // decode forwards of the current request (the caches shrink after 32: not for side requests)
    long dec_steps_ = 0, fill_due_ = 0;   // decode forwards so far; the running batch takes over at fill_due_
};

} // namespace hyper
