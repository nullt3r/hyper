// hyper engine M2: single-token decode for qwen35 with tensor parallelism across all GPUs.
// Every GPU holds a slice of every layer; partial results are summed with a P2P-free allreduce
// through mapped pinned host memory, and each token runs as one CUDA graph per GPU.
#pragma once
#include "gguf.h"
#include "kernels.cuh"
#include "model.h"

#include <memory>
#include <vector>

namespace hyper {

struct EngineOptions {
    int n_devices = 3;
    int max_pos = 32768;
    bool use_graphs = true;
};

class Engine {
public:
    Engine(const std::string & model_path, const EngineOptions & opt);
    ~Engine();

    void decode(int token, int pos);          // runs one token; greedy result available via argmax_last()
    void get_logits(std::vector<float> & out);
    int argmax_last() const { return last_argmax_; }
    void reset();

    const Qwen35Config & config() const { return cfg_; }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void record(int g);                       // enqueue one token's work for device g on its stream
    void build_graphs();

    EngineOptions opt_;
    std::unique_ptr<GGUF> gguf_;
    Qwen35Config cfg_;
    std::vector<std::unique_ptr<Device>> devs_;
    // host side shared buffers (pinned)
    float * h_embd_ = nullptr;
    int * h_pos_ = nullptr;
    float * h_res_ = nullptr;                 // [ndev][2] argmax pairs
    float * ar_slots_ = nullptr;              // mapped [2][ndev][n_embd]
    unsigned long long * ar_flags_ = nullptr; // mapped [ndev][nchunk]
    bool graphs_ready_ = false;
    int last_argmax_ = -1;
};

} // namespace hyper
