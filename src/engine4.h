// hyper engine for qwen4exp (Qwen3.8-Flash-Next): dense parts tensor-parallel over the GPUs, hyper-connection
// mixers and routers replicated, experts split between the GPUs (expert parallel) and the CPU.
#pragma once
#include "cpu_moe.h"
#include "gguf.h"
#include "kernels4.cuh"
#include "model4.h"

#include <memory>
#include <vector>

namespace hyper {

struct Engine4Options {
    int n_devices = 3;
    int max_pos = 8192;
    float gpu_expert_frac = 0.6f;   // fraction of each layer's experts placed on the GPUs (rest on the CPU)
    int cpu_threads = 24;
};

class Engine4 {
public:
    Engine4(const std::string & model_path, const Engine4Options & opt);
    ~Engine4();

    // nt <= MAX_NT tokens at positions pos..; returns greedy argmax per token
    std::vector<int> forward(const int * tokens, int nt, int pos);
    void get_logits(int t, std::vector<float> & out);
    void reset();
    const Q4Config & config() const { return cfg_; }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void load_layer(Device & dev, DevLayer & L, int il);
    void record_main(int gi, int nt);
    void build_graphs();
    void embed(const int * tokens, int nt, int pos);

    Engine4Options opt_;
    std::unique_ptr<GGUF> gguf_;
    Q4Config cfg_;
    std::vector<std::unique_ptr<Device>> devs_;
    std::unique_ptr<CpuMoe> cpu_;
    std::vector<int> seq_;          // tokens by position (PLE n-gram history)
    float * h_embd_ = nullptr;      // pinned [MAX_NT][n_embd]
    float * h_ple_ = nullptr;       // pinned [MAX_NT][ple heads * ple dim]
    int * h_pos_ = nullptr;
    float * h_res_ = nullptr;       // pinned [ndev][MAX_NT][2]
    uint2 * ar_ll_ = nullptr;
    CpuMoeRec * cpu_rec_ = nullptr; // mapped [n_layer]
    CpuMoeOut * cpu_out_ = nullptr; // mapped [n_layer]
    unsigned fwd_counter_ = 0;
    bool graphs_ready_ = false;
    bool debug_ = false;
    int last_nt_ = 0;
};

} // namespace hyper
