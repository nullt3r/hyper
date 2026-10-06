// hyper engine, milestone M1: single-token decode for qwen35, layers split across GPUs.
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
};

class Engine {
public:
    Engine(const std::string & model_path, const EngineOptions & opt);
    ~Engine();

    // run one token at position pos; logits stay on the last device
    void decode(int token, int pos);
    // copy logits of the last decode to host
    void get_logits(std::vector<float> & out);
    int argmax_last();
    void reset();   // clear recurrent state (KV positions are overwritten by position)

    const Qwen35Config & config() const { return cfg_; }

private:
    struct Layer;
    struct Device;
    void load_weights();
    void balance_layers();

    EngineOptions opt_;
    std::unique_ptr<GGUF> gguf_;
    Qwen35Config cfg_;
    std::vector<Layer> layers_;
    std::vector<std::unique_ptr<Device>> devs_;
    std::vector<int> layer_dev_;
    BF16W output_;
    float * output_norm_ = nullptr;
    float * h_embd_ = nullptr;   // pinned host staging for the token embedding
};

} // namespace hyper
