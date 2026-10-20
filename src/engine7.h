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
    int n_draft = 3;                // drafts per step (DSpark when loaded, else prompt lookup; 0: off)
    std::string dspark;             // DSpark drafter (dspark.bin + dspark.idx: tensor, type, offset, bytes, shape per line)
    float dspark_conf = 0.5f;       // a draft is verified while its confidence is at least this
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
    int n_draft() const override { return std::max(0, std::min(opt_.n_draft, MAX_NT - 1)); }
    bool has_mtp() const override { return false; }
    void reset_cache() override { hist_.clear(); parked_.clear(); }
    void save_expert_stats(const std::string & path);

private:
    struct Layer;
    struct Device;
    struct DBlock;   // a DSpark drafter block (block b on device b)
    void load_layer(Device & d, Layer & L);
    void place_experts();
    void run_layer(Device & d, Layer & L, int nt, int64_t p0);
    int sample_row(int t, const SamplingParams & sp, int64_t pos);

    Engine7Options opt_;
    std::unique_ptr<GGUF> gguf_;
    Ds4Config cfg_;
    int R_ = 2048;          // rows of a prefill chunk (the CPU experts stream through the GPUs once a chunk: 1024 -> 2048 prefill 242 -> 385 t/s)
    int ring_ = 16384;      // window ring rows per layer (the reach of a rollback: ring_ - win - chunk)
    std::vector<std::unique_ptr<Device>> devs_;
    std::vector<int> dev_of_;        // layer -> device
    std::unique_ptr<CpuMoe> cpu_;
    CpuMoeRec * cpu_rec_ = nullptr;
    CpuMoeOut * cpu_out_ = nullptr;
    CpuMoeBulk * cpu_bulk_ = nullptr;
    CpuMoeBulkOut * cpu_bulk_out_ = nullptr;
    // host rows of a chunk, NSET sets: prefill chunks run as a pipeline over the GPUs (the GPU of the first layers takes the
    // next chunk while the later ones still work on this one); a set is reused once its chunk is done (ev_chunk_)
    static constexpr int NSET = 3;
    float * h_embd_[NSET] = {};                  // pinned [R][n_embd]
    std::vector<float *> h_eg_[NSET];            // pinned [R][cols * hd] per Engram layer
    int * h_ids_[NSET] = {};                     // pinned [n_layer][R][k]: streamed prefill routing (statistics)
    int ids_nt_[NSET] = {};                      // (rows of a set's routing still to count)
    cudaEvent_t ev_chunk_[NSET] = {};            // on the last device after a set's chunk
    int cur_set_ = 0, next_set_ = 0;
    float * h_out_ = nullptr;                    // pinned: argmax pairs of the logit rows
    // enqueue a chunk (logits for its last nlog rows when nlog > 0); returns its set
    int enqueue(const int * tokens, int nt, int pos, int nlog);
    // parked conversations: another conversation's request would overwrite the state, so it is kept in host memory (its
    // tokens, the last PARK_ROWS window-ring rows of every layer, the KV sources' compressed rows / index keys and open
    // compressor groups) and restored when its conversation comes back
    struct Parked {
        std::vector<int> hist;
        std::vector<std::vector<uint16_t>> ring, ckv, ik;   // [layer]
        std::vector<std::vector<float>> cstate;             // [layer]: open group kv | score
        int64_t p0 = 0;                                     // the first position with a ring row
        size_t bytes = 0;
        uint64_t used = 0;
    };
    std::vector<Parked> parked_;
    uint64_t park_clock_ = 0;
    void park();
    void unpark(size_t i);
    int reuse_len(const std::vector<int> & h, int64_t p0, const std::vector<int> & prompt) const;
    void wait_set(int set);
    std::vector<std::vector<uint64_t>> stats_;
    std::vector<std::pair<void *, size_t>> host_bufs_;
    unsigned fwd_counter_ = 0;
    int last_nlog_ = 0;
    bool has_r2_ = false;
    // DSpark
    std::vector<std::unique_ptr<DBlock>> dblk_;
    void * dsp_map_ = nullptr;
    size_t dsp_size_ = 0;
    int dsp_T_ = 5, dsp_noise_ = 0, dsp_rank_ = 256, dsp_tap0_ = 0;
    float * h_dsp_ = nullptr;        // pinned: the draft block's rows between devices, main_x rows for the seeds
    float * h_dout_ = nullptr;       // pinned: drafts (argmax pairs) and confidences
    void load_dspark(const std::string & path);
    void dspark_seed(int nt, int64_t pos);   // after a forward's layers: the drafter's ring rows of these positions
    int dspark_draft(int y, int64_t p0, std::vector<int> & drafts, std::vector<float> & conf);
    double dsp_acc_ = 2.0;           // running accepted drafts per round
    float dsp_conf_ = 0.5f;            // a compressor of ratio > 1 (its open group: drafts rejected inside it re-feed its first token)
    double acc_rate_ = 0.5;          // prompt lookup: running fraction of the drafts accepted
    std::vector<int> hist_;
    uint64_t samp_seed_ = 0;
    std::function<void(int, int, int)> prefill_cb_;
    bool debug_ = false;
    int stream_min_ = 256;   // (shorter prefill chunks: the CPU computes its experts)
    bool stream_ = true;   // prefill: the CPU's experts streamed to the GPUs (HYPER7_NOSTREAM: computed on the CPU)
};

} // namespace hyper
