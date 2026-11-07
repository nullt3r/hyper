// hyper engine for deepseek41 (DeepSeek-V4.1-Flash): layers split over the GPUs along the KV-source groups (a group's compressed
// rows, index keys, selections and candidate blocks stay on one GPU), dense weights as dense.h uploads them, the hottest routed experts in VRAM
// and the rest on the CPU (CpuMoe), Engram rows read from the GGUF's tables (memory-mapped) on the host.
#pragma once
#include "cpu_moe.h"
#include "gguf.h"
#include "kernels.cuh"
#include "devices.h"
#include "experts.h"
#include "engine_base.h"
#include "kernels4.cuh"
#include "kernels7.cuh"
#include "model7.h"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
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

class Engine7 : public EngineBase {
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
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return std::max(0, std::min(opt_.n_draft, MAX_NT - 1)); }
    bool has_mtp() const override { return false; }
    void reset_cache() override { hist_.clear(); parked_.clear(); }

private:
    struct Layer;
    struct Device;
    struct DBlock;   // a DSpark drafter block (block b on device b)
    void load_layer(Device & d, Layer & L);
    void place_experts();
    // prefill with experts streamed over every PCIe link (HYPER7_EP_PREFILL, default on): a layer's CPU experts are split
    // between its device and helper devices in proportion to the links' bandwidth; a helper gets the layer's MoE input and
    // routing through pinned memory, streams its share through its own staging on its own stream, and returns the partial sum
    struct Share;
    void rebuild_part_slots(Layer & L);
    // (issued from the owner's thread right after the rows: a separate issuing thread let the owner overrun the ring)
    void issue_helper(Layer & L, const Share & sh, int nt, int r);
    bool ep_prefill_ = false;   // (HYPER7_EP_PREFILL=1: on; off until verified)
    static constexpr int EP_RING = 3;   // pinned slots per helper device (input rows / partials in flight)
    int ep_ring_ = EP_RING;
    int ep_debug_ = 0;                  // HYPER7_EP_DEBUG=1: the owner computes every expert itself, the helpers' partials are fetched and dropped             // slots in use (HYPER7_EP_RING <= EP_RING; 1: every reuse fully serialized)
    // adaptive expert placement: the shared placement's (HYPER7_ADAPT, _ADAPT_EVERY, _ADAPT_BUDGET 8, _DECAY): the hottest CPU
    // experts trade places with a layer's coldest GPU experts (the GPU expert's rows take the CPU expert's compact slot)
    double t_hostrows_ = 0, t_hostrows_max_ = 0; int n_hostrows_ = 0;
    double t_fwd_ = 0, t_head_ = 0, t_seed_ = 0, t_samp_ = 0; int n_fwd_ = 0;   // HYPER7_PROF: decode steps' forward wall, head / seed GPU time, sampling wall (ms)   // HYPER7_PROF: decode steps' host rows (embedding + Engram) ms
    void run_layer(Device & d, Layer & L, int nt, int64_t p0);
    void head_mm(Device & d, const float * x, int rows, cudaStream_t s);   // d.logits = output x (rows of n_embd)
    int sample_row(int t, const SamplingParams & sp, int64_t pos);

    Engine7Options opt_;
    std::unique_ptr<GGUF> gguf_;
    Ds4Config cfg_;
    int R_ = 2048;          // rows of a prefill chunk (the CPU experts stream through the GPUs once a chunk: 1024 -> 2048 prefill 242 -> 385 t/s)
    int ring_ = 16384;      // window ring rows per layer (the reach of a rollback: ring_ - win - chunk)
    std::vector<std::unique_ptr<Device>> devs_;
    std::vector<int> dev_of_;        // layer -> device
    const int pred_mode_ = getenv("HYPER7_PRED") ? atoi(getenv("HYPER7_PRED")) : 0;
    // HYPER7_PRED_EARLY=1 (with HYPER7_PRED=2, as engine5's HYPER5_PRED_EARLY): the next layer's routing also from this layer's
    // MoE input, published as the early prediction (cpu_pred1_) the CPU reads until the next layer's own prediction arrives
    // (HYPER_CPU_PF1GAP=1)
    const bool pred_early_ = getenv("HYPER7_PRED_EARLY") && atoi(getenv("HYPER7_PRED_EARLY")) != 0;
    const int pred_k_ = getenv("HYPER7_PRED_K") ? atoi(getenv("HYPER7_PRED_K")) : 0;   // 1: next layer's router on this layer's input, 2: on its own input   // pred: HYPER7_PRED's next-layer routing per layer (ids only)
    // expert host: a GPU (HYPER7_HOST_DEV, default device 0 with 4+ devices) holding the hottest CPU experts of another
    // device's layers (HYPER7_HOST_FOR, default the last device; HYPER7_HOST_GIB of them). In decode it takes the owner's
    // record like the CPU does and hands back a partial sum through host_out_; in prefill those experts stream from the
    // CPU copy as before
    bool host_on_ = false;                   // HYPER7_HOST (default on with 4+ devices)
    int host_dev_ = -1;                      // HYPER7_HOST_DEV: only this host; -1: every device but the target, from its spare VRAM
    size_t host_pool_bytes_ = 0;             // per host cap (HYPER7_HOST_GIB, default 64)
    std::vector<std::vector<int>> host_layers_;   // per host device: the layers whose pooled experts it holds
    CpuMoeOut * host_out_ = nullptr;
    void host_chain(Layer & L, int nt);
    // host rows of a chunk, NSET sets: prefill chunks run as a pipeline over the GPUs (the GPU of the first layers takes the
    // next chunk while the later ones still work on this one); a set is reused once its chunk is done (ev_chunk_)
    static constexpr int NSET = 3;
    float * h_embd_[NSET] = {};                  // pinned [R][n_embd]
    std::vector<float *> h_eg_[NSET];            // pinned [R][cols * hd] per Engram layer
    int * h_ids_[NSET] = {};                     // pinned [n_layer][R][k]: streamed prefill routing (statistics)
    int ids_nt_[NSET] = {};                      // (rows of a set's routing still to count)
    cudaEvent_t ev_chunk_[NSET] = {};            // on the last device after a set's chunk
    int next_set_ = 0;
    // streamed prefill: each GPU's part of a chunk is issued by that GPU's own host thread (one thread issuing all of them
    // in order is throttled by the fullest launch queue, which serializes the GPUs: 290 t/s instead of the pipeline).
    // Cross-GPU waits are issued after the other thread's records, ordered through these counters (chunk sequence numbers)
    struct IssueJob { int set = 0, nt = 0, nlog = 0; int64_t pos = 0; long seq = 0; const float * he = nullptr; };
    struct Issuer { std::thread th; std::deque<IssueJob> q; bool busy = false; };
    std::vector<std::unique_ptr<Issuer>> issuers_;
    std::mutex imx_;
    std::condition_variable icv_;
    std::vector<long> taken_seq_, done_seq_, issued_seq_;   // per device: the last chunk whose ev_taken / ev_done was recorded / issued
    long chunk_seq_ = -1, ev_chunk_seq_ = -1;               // chunks in order; the last whose ev_chunk_ was recorded
    long set_seq_[NSET] = {-1, -1, -1};                     // the chunk that last took a set
    bool stop_issuers_ = false, issue_threads_ = true;      // HYPER7_ISSUE_THREADS=0: issued inline (the old order)
    std::string issue_error_;
    void issue_device(int gi, const IssueJob & j);
    void issue_thread(int g);
    void start_issuers();
    void issue_drain();
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
    int dspark_draft(int y, int64_t p0, std::vector<int> & drafts, std::vector<float> & conf, float temp = 0.0f, uint64_t seed = 0,
                     std::vector<float> * logq = nullptr);
    // speculative sampling (temp > 0, full-vocabulary sampling): the drafts verified against the target rows; returns the
    // token after the m kept drafts
    int dspark_verify(int off, const std::vector<int> & dr, const std::vector<float> & logq, const SamplingParams & sp, int64_t p, int & m);
    double dsp_acc_ = 2.0;           // running accepted drafts per round
    bool dsp_coupled_ = true;        // greedy drafts kept on an exact match with the sampled token; HYPER7_DSPARK_SPECSAMP=1: speculative sampling
    float dsp_conf_ = 0.5f;            // a compressor of ratio > 1 (its open group: drafts rejected inside it re-feed its first token)
    // prefill progress from completed chunks (the issuing threads make enqueue return at once): reported at most once a
    // second from the waits (wait_set, issue_drain, forward's poll), so a streaming client gets its keepalives
    bool pf_active_ = false;
    int pf_total_ = 0, pf_reused_ = 0, pf_done_ = 0;
    std::chrono::steady_clock::time_point pf_tick_;
    int set_tokens_[NSET] = {};
    bool set_pending_[NSET] = {};
    void prefill_tick(bool force);
    bool debug_ = false;
    int stream_min_ = 256;   // (shorter prefill chunks: the CPU computes its experts)
    bool stream_ = true;   // prefill: the CPU's experts streamed to the GPUs (HYPER7_NOSTREAM: computed on the CPU)
};

} // namespace hyper
