// hyper engine for qwen4exp (Qwen3.8-Flash-Next): dense parts tensor-parallel over the GPUs, hyper-connection
// mixers and routers replicated, experts split between the GPUs (expert parallel) and the CPU.
#pragma once
#include "cpu_moe.h"
#include "gguf.h"
#include "kernels4.cuh"
#include "llm.h"
#include "park.h"
#include "model4.h"

#include <atomic>
#include <functional>
#include <random>
#include <memory>
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
            while (gen_.load(std::memory_order_acquire) == gen) {}
        }
    }
private:
    int n_;
    std::atomic<int> count_{0};
    std::atomic<unsigned> gen_{0};
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
    void set_snapshot_token(int tok) override { snap_token_ = tok; }
    void reset_cache() override { hist_.clear(); for (auto & s : snaps_) snap_pool_.push_back(s.h); snaps_.clear(); park_.clear(snap_pool_); }
    void set_prefill_progress(std::function<void(int, int, int)> fn) override { prefill_cb_ = std::move(fn); }
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return mtp_g_ ? opt_.n_draft : 0; }
    bool has_mtp() const override { return mtp_g_ != nullptr; }
    // routing statistics: the loaded ones plus what the CPU side has seen since (decode routing of every layer)
    void save_expert_stats(const std::string & path);
    // MTP drafts per step and the run-probability floor (tests)
    void set_draft(int k, double pmin) { opt_.n_draft = std::max(1, std::min(k, MAX_NT - 1)); mtp_pmin_ = pmin; }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void load_layer(Device & dev, DevLayer & L, int il);
    void load_experts(int il, const std::vector<int> & quota);
    void upload_stage(Device & d, int il);
    void record_main(int gi, int nt, int kind = 0);
    void record_restore(int gi, int keep);
    void build_graphs();
    void embed(const int * tokens, int nt, int pos, int chunk = 0);
    void embed_tok(const int * tokens, int nt, float * dst);
    void run(int kind, int nt);   // 0 main, 1 MTP, 2 MTP chain, 3 restore (nt = rows kept)
    int mtp_result();
    // need_draft false: the rows only enter the MTP block's cache (its K / V and indexer keys depend on its input alone)
    int mtp_draft(const int * tokens, int nt, int pos, bool need_draft = true);
    int mtp_chain(int token, int pos);
    void * host_huge_alloc(size_t bytes);   // anonymous, transparent huge pages; freed with the engine
    int sample_row(int t, const SamplingParams & sp);
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
    std::mt19937_64 rng_;
    std::vector<std::pair<void *, size_t>> host_bufs_;

    Engine4Options opt_;
    std::unique_ptr<GGUF> gguf_, mtp_g_;
    const GGUF * src_ = nullptr;     // file the layer loaders read from
    Q4Config cfg_;
    std::vector<std::unique_ptr<Device>> devs_;
    std::unique_ptr<CpuMoe> cpu_;
    std::vector<std::vector<uint64_t>> stats_;   // expert placement statistics [layer][expert]
    std::vector<int> seq_;          // tokens by position (PLE n-gram history)
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
    int mc_n_ = 1, mc_nt_[8] = {}, mc_pos_[8] = {};   // the chunks of the current bulk forward
    int mc_last_ = 1;              // chunks of the last forward_multi
    int mtp_src_ = 0;              // MTP pass: residual rows of chunk mtp_src_ of the last multi-chunk forward
    int * h_cpos_ = nullptr;       // pinned: positions of the chunks
    int stream_min_ = 256;   // prefill chunks this long stream the CPU experts to the GPUs; shorter ones use the CPU
};

} // namespace hyper
