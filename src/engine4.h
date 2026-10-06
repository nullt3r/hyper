// hyper engine for qwen4exp (Qwen3.8-Flash-Next): dense parts tensor-parallel over the GPUs, hyper-connection
// mixers and routers replicated, experts split between the GPUs (expert parallel) and the CPU.
#pragma once
#include "cpu_moe.h"
#include "gguf.h"
#include "kernels4.cuh"
#include "llm.h"
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
    float gpu_expert_frac = 0.6f;   // fraction of each layer's experts placed on the GPUs (rest on the CPU)
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
    // whole prompt in prefill chunks; returns the argmax after the last token
    int prefill(const int * tokens, int n, int pos);
    void get_logits(int t, std::vector<float> & out);
    void reset();
    const Q4Config & config() const { return cfg_; }
    // LLM: greedy / sampled generation (no speculative decoding yet), prompt cache
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                              const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) override;
    void set_snapshot_token(int tok) override { snap_token_ = tok; }
    void reset_cache() { hist_.clear(); for (auto & s : snaps_) snap_pool_.push_back(s.h); snaps_.clear(); }
    void set_prefill_progress(std::function<void(int, int, int)> fn) override { prefill_cb_ = std::move(fn); }
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return 0; }
    bool has_mtp() const override { return false; }
    void save_expert_stats(const std::string & path) { cpu_->save_stats(path); }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void load_layer(Device & dev, DevLayer & L, int il);
    void record_main(int gi, int nt);
    void build_graphs();
    void embed(const int * tokens, int nt, int pos);
    void * host_huge_alloc(size_t bytes);   // anonymous, transparent huge pages; freed with the engine
    int sample_row(int t, const SamplingParams & sp);
    struct Snap { int pos; std::vector<float *> h; };
    size_t snap_floats(int gi) const;
    void snap_copy(Snap & sn, bool to_host);
    void take_snapshot(int pos);
    std::vector<int> hist_;
    std::vector<Snap> snaps_;
    std::vector<std::vector<float *>> snap_pool_;
    int snap_token_ = -1;
    std::function<void(int, int, int)> prefill_cb_;
    float * h_topk_ = nullptr;      // pinned [ndev][MAX_NT][TOPK][2]
    std::mt19937_64 rng_;
    std::vector<std::pair<void *, size_t>> host_bufs_;

    Engine4Options opt_;
    std::unique_ptr<GGUF> gguf_;
    Q4Config cfg_;
    std::vector<std::unique_ptr<Device>> devs_;
    std::unique_ptr<CpuMoe> cpu_;
    std::vector<std::vector<uint64_t>> stats_;   // expert placement statistics [layer][expert]
    std::vector<int> seq_;          // tokens by position (PLE n-gram history)
    float * h_embd_ = nullptr;      // pinned [MAX_NT][n_embd]
    float * h_ple_ = nullptr;       // pinned [MAX_NT][ple heads * ple dim]
    int * h_pos_ = nullptr;
    float * h_res_ = nullptr;       // pinned [ndev][MAX_NT][2]
    uint2 * ar_ll_ = nullptr;
    CpuMoeRec * cpu_rec_ = nullptr; // mapped [n_layer]
    CpuMoeOut * cpu_out_ = nullptr; // mapped [n_layer]
    CpuMoeBulk * cpu_bulk_ = nullptr;        // mapped, prefill
    CpuMoeBulkOut * cpu_bulk_out_ = nullptr;
    half * h_stage_ = nullptr;      // pinned DMA allreduce staging [parity][ndev][MOE_BULK_ROWS * n_embd]
    std::unique_ptr<Barrier4> barrier_;
    unsigned fwd_counter_ = 0;
    bool graphs_ready_ = false;
    bool debug_ = false, nocpu_ = false, allrows_ = false;
    int last_nt_ = 0;
};

} // namespace hyper
