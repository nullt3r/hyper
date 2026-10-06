// hyper engine: qwen35 decode with tensor parallelism across all GPUs and MTP speculative decoding.
// Every GPU holds a slice of every layer; partial results are summed with a P2P-free allreduce through
// mapped pinned host memory (LL protocol). Each forward variant runs as one CUDA graph per GPU.
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
    bool mtp = true;             // load the NextN head for speculative decoding
    int n_draft = 2;             // MTP drafts per step (chained); verification runs n_draft + 1 tokens
};

struct GenStats {
    int tokens = 0, steps = 0, accepted = 0;
    double seconds = 0;
    double t_main = 0, t_mtp = 0, t_restore = 0;   // wall time per phase
};

class Engine {
public:
    Engine(const std::string & model_path, const EngineOptions & opt);
    ~Engine();

    // main model over nt tokens at positions pos..pos+nt-1; returns greedy argmax per token
    std::vector<int> forward(const int * tokens, int nt, int pos);
    // logits of token row t from the last forward
    void get_logits(int t, std::vector<float> & out);
    void reset();

    // greedy generation; prompt processed token by token. spec = use MTP drafts (1 per step)
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats);

    const Qwen35Config & config() const { return cfg_; }

private:
    struct DevLayer;
    struct Device;
    void load_weights();
    void load_layer(Device & dev, DevLayer & L, int il, bool mtp_layer);
    void record_main(int gi, int nt);
    void record_mtp(int gi, int nt, bool chain);
    void record_restore(int gi, int keep);
    void record_attn(Device & d, DevLayer & L, int nt, const int * pos, int & call);
    void record_gdn(Device & d, DevLayer & L, int nt, bool snap, int & call);
    void record_ffn(Device & d, DevLayer & L, int nt, int & call);
    void build_graphs();
    void launch(int kind, int nt);   // kind: 0 main, 1 mtp, 2 restore
    void embed(const int * tokens, int nt, float * dst);
    int mtp_draft(const int * tokens, int nt, int pos);   // MTP over (tokens[t], main hidden row t) at pos+t; argmax of last
    int mtp_chain(int token, int pos);                   // MTP over (token, MTP's own last hidden) at pos

    EngineOptions opt_;
    std::unique_ptr<GGUF> gguf_;
    Qwen35Config cfg_;
    std::vector<std::unique_ptr<Device>> devs_;
    float * h_embd_ = nullptr;   // pinned [MAX_NT][n_embd] main input
    float * h_membd_ = nullptr;  // pinned [MAX_NT][n_embd] MTP input
    int * h_pos_ = nullptr;      // pinned [2]: main pos, mtp pos
    float * h_res_ = nullptr;    // pinned [ndev][MAX_NT][2] main argmax pairs
    float * h_mres_ = nullptr;   // pinned [ndev][MAX_NT][2] mtp argmax pairs
    uint2 * ar_ll_ = nullptr;    // mapped LL slots [2][ndev][MAX_NT * n_embd / 2]
    bool graphs_ready_ = false;
    int last_nt_ = 0;
};

} // namespace hyper
