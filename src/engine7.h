// hyper engine for deepseek41 (DeepSeek-V4.1-Flash): layers split over the GPUs along the KV-source groups (a group's compressed
// rows, index keys, selections and candidate blocks stay on one GPU), dense weights Q8_0, the hottest routed experts in VRAM
// and the rest on the CPU (CpuMoe), Engram rows read from the GGUF's tables (memory-mapped) on the host.
#pragma once
#include "cpu_moe.h"
#include "gguf.h"
#include "kernels.cuh"
#include "kernels4.cuh"
#include "kernels7.cuh"
#include "llm.h"
#include "model7.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hyper {

struct Engine7Options {
    int n_devices = 3;
    int max_pos = 8192;
    int cpu_threads = 30;
    bool prompt_cache = true;
    float vram_reserve_gib = 1.0f;
    float gpu_expert_frac = 1.0f;   // share of the free VRAM given to experts
};

class Engine7 : public LLM {
public:
    Engine7(const std::string & model_path, const Engine7Options & opt);
    ~Engine7();

    // tokens at positions pos .. pos + nt - 1 (nt <= chunk); logits for the last `nlog` rows (greedy next tokens returned)
    std::vector<int> forward(const int * tokens, int nt, int pos, int nlog);
    void get_logits(int row, std::vector<float> & out);   // a logit row of the last forward (row < nlog)
    void reset();
    const Ds4Config & config() const { return cfg_; }
    int chunk() const { return R_; }

    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                              const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) override;
    void set_snapshot_token(int) override {}
    void set_prefill_progress(std::function<void(int, int, int)> fn) override { prefill_cb_ = std::move(fn); }
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return 0; }
    bool has_mtp() const override { return false; }
    void reset_cache() override { hist_.clear(); }
    void save_expert_stats(const std::string & path);

private:
    struct Layer;
    struct Device;
    void load_layer(Device & d, Layer & L);
    void place_experts();
    void run_layer(Device & d, Layer & L, int nt, int64_t p0);
    int sample_row(int t, const SamplingParams & sp, int64_t pos);

    Engine7Options opt_;
    std::unique_ptr<GGUF> gguf_;
    Ds4Config cfg_;
    int R_ = 1024;          // rows of a prefill chunk
    int ring_ = 16384;      // window ring rows per layer (the reach of a rollback: ring_ - win - chunk)
    std::vector<std::unique_ptr<Device>> devs_;
    std::vector<int> dev_of_;        // layer -> device
    std::unique_ptr<CpuMoe> cpu_;
    CpuMoeRec * cpu_rec_ = nullptr;
    CpuMoeOut * cpu_out_ = nullptr;
    CpuMoeBulk * cpu_bulk_ = nullptr;
    CpuMoeBulkOut * cpu_bulk_out_ = nullptr;
    float * h_embd_ = nullptr;       // pinned [R][n_embd]
    std::vector<float *> h_eg_;      // pinned [R][cols * hd] per Engram layer
    float * h_out_ = nullptr;        // pinned: argmax pairs / top-k candidates of the logit rows
    std::vector<std::vector<uint64_t>> stats_;
    std::vector<std::pair<void *, size_t>> host_bufs_;
    unsigned fwd_counter_ = 0;
    int last_nlog_ = 0;
    std::vector<int> hist_;
    uint64_t samp_seed_ = 0;
    std::function<void(int, int, int)> prefill_cb_;
    bool debug_ = false;
    int stream_min_ = 256;   // (shorter prefill chunks: the CPU computes its experts)
    bool stream_ = true;   // prefill: the CPU's experts streamed to the GPUs (HYPER7_NOSTREAM: computed on the CPU)
};

} // namespace hyper
