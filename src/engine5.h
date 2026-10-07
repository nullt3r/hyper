// hyper engine for glm5-next (GLM-5.3-Flash): dense parts tensor-parallel over the GPUs (KDA / MLA heads, FFN hidden),
// mHC mixers, the MLA latent cache and the indexer replicated, experts split between the GPUs and the CPU.
#pragma once
#include "cpu_moe.h"
#include "engine4.h"   // Barrier4
#include "gguf.h"
#include "kernels4.cuh"
#include "kernels5.cuh"
#include "llm.h"
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
    std::string mtp_path;           // (no NextN head in the GLM files yet)
    int n_draft = 0;
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
    int prefill(const int * tokens, int n, int pos);
    void get_logits(int t, std::vector<float> & out);
    void reset();
    const Glm5Config & config() const { return cfg_; }
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                              const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) override;
    void set_snapshot_token(int tok) override { snap_tokens_.push_back(tok); }   // every message-start token given
    void set_prefill_progress(std::function<void(int, int, int)> fn) override { prefill_cb_ = std::move(fn); }
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return 0; }
    bool has_mtp() const override { return false; }
    void save_expert_stats(const std::string & path);
    void reset_cache() { hist_.clear(); for (auto & s : snaps_) snap_pool_.push_back(s.h); snaps_.clear(); }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void load_layer(Device & dev, DevLayer & L, int il);
    void load_experts(int il, const std::vector<int> & quota);
    void upload_stage(Device & d, int il, int sb);
    void record_main(int gi, int nt);
    void build_graphs();
    void embed(const int * tokens, int nt);
    void run(int nt);
    void * host_huge_alloc(size_t bytes);
    int sample_row(int t, const SamplingParams & sp);
    struct Snap { int pos; std::vector<float *> h; };
    size_t snap_floats(int gi) const;
    void snap_copy(Snap & sn, bool to_host);
    void take_snapshot(int pos);
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
    std::unique_ptr<GGUF> gguf_;
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
    std::vector<int> zc_blocks_;   // HYPER5_ZC
};

} // namespace hyper
