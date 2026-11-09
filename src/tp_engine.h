// The tensor-parallel engine shared by the models whose dense parts split over the GPUs (Qwen3.8 Flash-Next, GLM-5.3-Flash):
// the devices and their common buffers, the CPU side, the experts' placement, decode graphs, prefill chunks and their
// streamed experts, the generate loop with its speculation (MTP drafts, prompt lookup), the prompt cache (recurrent-state
// snapshots, parked conversations), elastic attention caches, sampling. A model supplies its layers: what a device loads
// and how a forward is recorded (TpEngine::record_main), the per-position caches it parks, the recurrent state it
// snapshots, its embeddings.
#pragma once
#include "dense.h"
#include "engine_base.h"
#include "experts.h"
#include "gguf.h"
#include "kernels4.cuh"
#include "park.h"
#include "vmm.h"

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace hyper {

struct TpOptions {
    int n_devices = 0;   // 0: every visible CUDA device (<ENV>_NDEV / HYPER_NDEV override)
    int max_pos = 8192;
    float gpu_expert_frac = 1.0f;   // cap on the fraction of each layer's experts on the GPUs (the rest: CPU)
    float vram_reserve_gib = 0.8f;  // left free on every GPU after the experts
    std::string mtp_path;           // separate NextN (MTP) GGUF: speculative decoding
    int n_draft = 3;                // drafted tokens per step: MTP (with mtp_path), else prompt lookup where the model allows (0: off)
    bool stream_experts = true;     // prefill: copy the CPU experts' weights to the GPUs instead of computing them on the CPU
    int cpu_threads = 30;
    bool prompt_cache = false;      // reuse the common prefix with the previous sequence (recurrent-state snapshots)
    int max_snapshots = 48;
    bool answer_reuse = false;   // the end-of-answer snapshot serves a next request that repeats the answer (its K/V and state
                                 // as decoding computed them, not re-prefilled)
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
            while (gen_.load(std::memory_order_acquire) == gen)
                if (aborted_.load(std::memory_order_acquire)) throw std::runtime_error("barrier: another device's recording failed");
        }
    }
    // a recording thread that failed releases the others (they throw instead of waiting forever); reset once all have joined
    void abort() { aborted_.store(true, std::memory_order_release); }
    void reset() { count_.store(0, std::memory_order_relaxed); aborted_.store(false, std::memory_order_release); }
private:
    int n_;
    std::atomic<int> count_{0};
    std::atomic<unsigned> gen_{0};
    std::atomic<bool> aborted_{false};
};

// what the model tells the engine about itself (its config in the engine's terms) and the defaults it was tuned with
struct TpModel {
    std::string tag = "hyper", env = "HYPER";   // log prefix, env prefix of the knobs (<env>_MC, _STREAM_MIN, _MTP_PMIN, ...)
    int n_layer = 0, n_dense = 0;                // layers; the first n_dense have a dense FFN (no experts)
    int n_embd = 0, n_vocab = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
    float clamp_exp = 0.0f;                      // SwiGLU limit of the experts
    int max_pos_align = 1;                       // the context rounded up to a multiple
    int stream_min = 256;                        // prefill chunks this long stream the CPU experts to the GPUs; shorter ones use the CPU
    int up_parts = 2;                            // streamed expert uploads in parts, each after an allreduce's downloads (0 / 1: whole)
    int halves = 1;                              // streamed share per device in one list or two halves (two staging buffers per layer)
    double stage_factor = 2.0;                   // staging estimate for the quotas: 2 whole buffers, or 1.05 with halves
    int fill_unit = 256;                         // elastic hand-over pacing (experts per 16 decode steps)
    double stream_cpu = 0.0, stream_cpu_max = 0.5;   // a single streamed chunk leaves min(max, stream_cpu / nt) of the CPU experts on the CPU
    double mtp_pmin = 0.5;                       // drafts while their run's MTP probability stays >= this (0: acceptance-driven count)
    double mtp_cpu_max = 1e30;                   // drafts only while the EMA of CPU experts per layer and step stays under this
    bool lookup_spec = false;                    // prompt-lookup speculation without an MTP block
    bool pstream_prio = false;                   // the prediction side stream at the highest priority (graph node priorities)
};

// the per-device state the engine itself uses; a model's Device derives from it with its layers and its own buffers
struct TpDevice : DevBase {
    cudaGraphExec_t g_main[MAX_NT + 1] = {}, g_mtp[MAX_NT + 1] = {}, g_chain = nullptr, g_restore[MAX_NT] = {};
    DW output;                 // this device's vocab slice of the LM head
    int vocab_off = 0;
    // activations (prefill-chunk rows unless noted)
    float * x = nullptr, * res = nullptr, * xn = nullptr, * bo = nullptr, * part = nullptr, * big0 = nullptr, * o = nullptr, * attn_part = nullptr;
    std::vector<float *> res_x;      // residual rows of the further chunks of a multi-chunk prefill pass
    int * pos_x = nullptr;           // their positions
    float * router32 = nullptr;      // prefill: a bf16 router widened
    float * rlog = nullptr, * wts = nullptr, * sg = nullptr, * shgu = nullptr, * shh = nullptr, * shpart = nullptr;
    float * hexp = nullptr, * yexp = nullptr, * logits = nullptr, * conv_raw = nullptr, * iscores = nullptr;
    float * plog = nullptr, * pwts = nullptr, * psg = nullptr;   // routing prediction (GPU 0)
    int * pids = nullptr;
    bool pe_pending = false;         // an early prediction on the side stream; the main stream waits for ev_pe1 before rewriting its input
    unsigned * ticket = nullptr;     // fused route + publish: blocks done
    int * ids = nullptr, * pos = nullptr, * counter = nullptr, * order = nullptr, * order_n = nullptr, * egrp = nullptr;
    int * ilist = nullptr, * ilist_n = nullptr;
    unsigned * ihist = nullptr;      // indexer score histograms
    half * xh = nullptr, * p16 = nullptr, * recv = nullptr, * mix16 = nullptr, * h16 = nullptr;
    // bulk allreduce in flight per chunk slot: the fp16 partial (its DMA source) and the peers' partials (the DMA targets);
    // p16 / recv alias slot 0
    std::vector<half *> p16s, recvs;
    cudaStream_t arstream = nullptr;   // the allreduce copies: off the compute stream, so a chunk's exchange overlaps the next one's kernels
    uint2 * ll_buf = nullptr;          // P2P: this device's LL packet buffer (the peers write into it)
    LLBuf ll;                          // where this device's LL allreduce writes and reads
    half * ggs = nullptr;            // KQ dense weights: fp16 slice for prefill GEMMs
    size_t ggs_elems = 0;
    float * topk = nullptr;          // sampling candidates [MAX_NT][TOPK][2]
    float * lstat = nullptr;         // sampling: a logit row slice's max and sum exp
    int big_stride = 0;
    // streamed expert upload parts not yet enqueued (layer, staging half, part): each goes out after the next allreduce's
    // downloads, so the copy engine (FIFO) never holds an allreduce copy -- on the critical path -- behind a whole upload
    std::deque<std::array<int, 3>> up_pend;
    cudaEvent_t ev_arh = nullptr;
    cudaEvent_t ev_ar[2] = {};
    // per (phase parity, chunk slot): the partial converted (compute -> copy stream), uploaded (peers wait for it), the peers'
    // partials downloaded (compute stream waits; peers may overwrite the staging slot after it)
    cudaEvent_t ev_a[2][8] = {}, ev_up2[2][8] = {}, ev_dn[2][8] = {};
    // elastic placement: the per-position caches as segments of virtual-memory buffers, and their full-context size
    struct KvSeg { VmmBuf * b; size_t pitch, row_bytes; int heads, div; };
    std::vector<KvSeg> kvs;
    size_t lend_limit = 0;
    std::vector<std::unique_ptr<VmmBuf>> vmm;
    ~TpDevice() {
        cudaSetDevice(id);
        for (auto & gr : g_main) if (gr) cudaGraphExecDestroy(gr);
        for (auto & gr : g_mtp) if (gr) cudaGraphExecDestroy(gr);
        for (auto & gr : g_restore) if (gr) cudaGraphExecDestroy(gr);
        if (g_chain) cudaGraphExecDestroy(g_chain);
        for (auto & ev : ev_ar) if (ev) cudaEventDestroy(ev);
        for (int pp = 0; pp < 2; ++pp) for (int sl = 0; sl < 8; ++sl) for (cudaEvent_t e : {ev_a[pp][sl], ev_up2[pp][sl], ev_dn[pp][sl]}) if (e) cudaEventDestroy(e);
        if (ev_arh) cudaEventDestroy(ev_arh);
        if (arstream) cudaStreamDestroy(arstream);
    }
};

class TpEngine : public EngineBase {
public:
    ~TpEngine() override;
    // nt tokens at positions pos..: up to MAX_NT through the decode graphs, up to MOE_BULK_ROWS as a prefill chunk (tiled
    // GEMMs, copy-engine allreduce). Returns the greedy argmax per token (prefill chunk: last token only, -1 elsewhere)
    std::vector<int> forward(const int * tokens, int nt, int pos);
    // several consecutive prefill chunks (each <= MOE_BULK_ROWS, all streaming) layer by layer: a layer's streamed CPU
    // experts cross PCIe once for all of them; result: the last chunk's next token
    int forward_multi(const int * tokens, const int * lens, int nck, int pos);
    bool multi_ok(const int * lens, int nck) const;
    int prefill(const int * tokens, int n, int pos);   // whole prompt in prefill chunks; returns the argmax after the last token
    void get_logits(int t, std::vector<float> & out);
    virtual void reset() = 0;   // the recurrent state of a fresh conversation
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                              const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) override;
    void set_snapshot_token(int tok) override { snap_tokens_.push_back(tok); park_.msg_tokens.push_back(tok); }   // every message-start token given
    void reset_cache() override { hist_.clear(); for (auto & s : snaps_) snap_pool_.push_back(s.h); snaps_.clear(); park_.clear(snap_pool_); }
    int max_pos() const override { return opt_.max_pos; }
    int n_draft() const override { return mtp_g_ || m_.lookup_spec ? std::min(opt_.n_draft, MAX_NT - 1) : 0; }
    bool has_mtp() const override { return mtp_g_ != nullptr; }
    // MTP drafts per step (<= MAX_NT - 1) and the run-probability floor (tests)
    void set_draft(int k, double pmin) { opt_.n_draft = std::max(1, std::min(k, MAX_NT - 1)); mtp_pmin_ = pmin; }

protected:
    struct TpSnap { int pos; std::vector<float *> h; };   // a recurrent-state snapshot (prompt cache)
    // ---- what a model implements ----
    virtual std::unique_ptr<TpDevice> make_device() = 0;
    // every layer of this device (its weights, caches, the MTP block when loaded), the head, the model's own buffers; the
    // engine's common buffers come from alloc_common()
    virtual void load_device(TpDevice & d) = 0;
    virtual const GGUF & expert_source(int il) const = 0;   // the file holding layer il's experts (the MTP file for the block)
    virtual void after_load() {}                            // (e.g. a table copied to RAM)
    // record one forward of `kind` over nt rows on device gi into its stream (captured into a graph for decode): 0 main;
    // 1 MTP block over (token, main hidden) rows; 2 MTP chained on its own last hidden row
    virtual void record_main(int gi, int nt, int kind) = 0;
    virtual void record_restore(int gi, int keep) = 0;      // the recurrent state back to the snapshot after row keep-1 of the last verification
    virtual bool mtp_direct() const { return false; }       // this MTP pass is not a graph's (rows into the block's cache only)
    virtual size_t snap_floats(int gi) const = 0;
    virtual void snap_copy(std::vector<float *> & h, bool to_host) = 0;
    virtual std::vector<ConvPark<TpSnap>::Dev> park_devs() const = 0;   // the per-position GPU buffers a conversation parks
    virtual void embed(const int * tokens, int nt, int pos, int chunk = 0) = 0;   // input rows of positions pos.. into h_embd_ (+ chunk * R rows)
    // the prompt's extras (images, M-RoPE positions, n-gram history): the cache key of the prompt (nullptr: the prompt itself),
    // called first; then once the reused prefix s is known; then after the prefill
    virtual const std::vector<int> * prompt_key(const std::vector<int> & prompt) { (void) prompt; return nullptr; }
    virtual void prompt_begin(const std::vector<int> & prompt, int s) { (void) prompt; (void) s; }
    virtual void prompt_end() {}
    // MTP: after a prefill forward over the chunks lens[0..nk) at c0 (next: the predicted token after them): the block's
    // cache rows for them; returns the first draft when end == P (-1: the engine chains it from the main hidden)
    virtual int mtp_prompt(const std::vector<int> & prompt, int c0, const int * lens, int nk, int next) = 0;
    // MTP rows (tokens[r] at pos + r, the main hidden of pos + r - 1): the kept verification rows (draft: the token after them)
    virtual int mtp_update(const int * tokens, int nt, int pos, bool draft) = 0;
    // MTP over (token at pos, its own last hidden; from_main: the main model's hidden of pos - 1)
    virtual int mtp_chain(int token, int pos, bool from_main) = 0;

    // ---- the engine ----
    TpEngine(TpOptions opt, TpModel m);
    void init();   // (from the model's constructor, after its config is parsed: gguf_, mtp_g_, m_)
    void load_weights();
    struct CommonSizes {   // what the common buffers are sized by (a model's widths)
        int res_w = 0, xn_w = 0;     // residual row width (hc streams x n_embd), the normed input's
        int big_stride = 0, o_w = 0, ff_sh = 0, xh_w = 0, conv_w = 0;
        size_t attn_part = 0;
        int rlog_w = 0;              // router logits per row (the experts, + the shared-expert gate)
        int hexp_rows = 0, logit_rows = 0;
        int iscore_rows = 0, ilist_w = 0;
        bool ihist = false;
        bool pred_all = false;       // routing-prediction buffers on every device (else GPU 0 with a prediction mode)
        int pred_k_w = 0, pred_psg_w = 0;   // their widths (ids / weights per row; the gate scratch)
    };
    int mc_alloc_ = 4;   // chunk buffers allocated (<ENV>_MC_ALLOC: more than used, for an equal VRAM footprint)
    bool ar_sync_ = false;   // <ENV>_AR_SYNC=1: every bulk allreduce completes before the next chunk's work (the old order; tests)
    void alloc_common(TpDevice & d, const CommonSizes & sz);
    template <typename T> T * kv_alloc(TpDevice & dev, int heads, size_t pitch_elems, size_t row_elems, int div);
    void * host_huge_alloc(size_t bytes);   // anonymous, transparent huge pages; freed with the engine
    bool is_moe(int il) const { return il >= m_.n_dense; }
    int nl_tot() const { return m_.n_layer + (mtp_g_ ? 1 : 0); }
    int R() const { return MOE_BULK_ROWS; }
    // record_main helpers
    void mm(TpDevice & d, const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni = NormIn{}) const;
    void f32mm(TpDevice & d, const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nr) const;
    void router_mm(TpDevice & d, const float * router, const uint16_t * router16, int E, const float * x, float * y, int nr) const;
    void dbg(TpDevice & d, const char * what, int il, const float * buf, size_t cnt, int sum_layers, int rows = 1) const;
    // x += sum over the GPUs of d.part (x nullptr: d.bo = the sum); bulk: the copy-engine allreduce through pinned staging
    void allreduce(TpDevice & d, int nt, int & call, int & dcall, bool bulk, float * x = nullptr);
    // the bulk allreduce in two halves, so a multi-chunk pass exchanges chunk c while computing chunk c + 1: ar_begin converts
    // d.part and uploads it (copy stream), ar_end downloads the peers' partials and adds them all into x (d.bo when nullptr).
    // Between every device's ar_begin of a phase and the first ar_end of it the host threads must pass barrier_ (the peers'
    // events exist then); par: 0 / 1 for the two allreduces of a layer, slot: the chunk
    void ar_begin(TpDevice & d, int nt, int slot, int par);
    void ar_fetch(TpDevice & d, int nt, int slot, int par);   // after the barrier: the peers' partials start downloading (copy stream)
    void ar_sum(TpDevice & d, int nt, int slot, int par, float * x = nullptr);   // compute stream: waits for them, adds them all
    void ar_end(TpDevice & d, int nt, int slot, int par, float * x = nullptr) { ar_fetch(d, nt, slot, par); ar_sum(d, nt, slot, par, x); }
    // the streamed experts' uploads at the start of a recording (the first layer's share) and the upload-part queue
    void stream_begin(TpDevice & d, int il0, int il1);
    void stream_next(TpDevice & d, int il, int sb, int il1);   // after layer il's half sb is done: the next layer's half into the staging
    void run(int kind, int nt);   // 0 main, 1 MTP, 2 MTP chain, 3 restore (nt = rows kept)
    void build_graphs();
    void launch_all(int nt, int kind);
    int mtp_result();
    void draft_at(int64_t pos) { mtp_pos_ = pos; }   // the next MTP pass drafts the token at pos
    int sample_row(int t, const SamplingParams & sp, int64_t pos);
    void take_snapshot(int pos);
    void score_prompt(const int * lens, int nck);
    template <typename F> void trace_routing(int type, int nrows, F && id);
    size_t lend_free(const TpDevice & d) const { return place_->lend_free(d.g); }
    void ensure_kv(int need);
    void shrink_kv(int keep);
    TpDevice & tdev(int g) { return *devs_[g]; }

    TpOptions opt_;
    TpModel m_;
    std::unique_ptr<GGUF> gguf_, mtp_g_;
    std::vector<std::unique_ptr<TpDevice>> devs_;
    std::unique_ptr<Barrier4> barrier_;
    // snapshots of the recurrent state (prompt cache) and parked conversations
    ConvPark<TpSnap> park_;
    std::vector<TpSnap> snaps_;
    std::vector<std::vector<float *>> snap_pool_;
    std::vector<int> snap_tokens_;
    // host buffers
    float * h_embd_ = nullptr;      // pinned [mc_max][R][n_embd]
    int * h_pos_ = nullptr;         // pinned [16]: chunk positions ([8]: the MTP update's last row)
    int * h_cpos_ = nullptr;        // pinned [8]: the chunks' positions
    float * h_res_ = nullptr;       // pinned [ndev][MAX_NT][2]: argmax pairs
    float * h_mres_ = nullptr;      // pinned [ndev][4]: MTP draft argmax, slice max logit, slice sum exp
    float * h_topk_ = nullptr;      // pinned [ndev][MAX_NT][TOPK][2]
    float * h_mtopk_ = nullptr;     // pinned [ndev][TOPK][2]: the MTP head's candidates per vocab slice
    half * h_stage_ = nullptr;      // pinned DMA allreduce staging [parity][mc_alloc][ndev][R * n_embd]
    uint2 * ar_ll_ = nullptr;       // host-mode LL packet buffer (mapped)
    bool p2p_ = false;              // peer access between every pair: LL packets and prefill partials go into the peers' VRAM
    int * h_ids_ = nullptr;         // pinned [nl_tot][mc_max][R][K]: prefill routing from GPU 0
    int * h_pred_ = nullptr;        // pinned [n_layer][MOE_MAX_USED] (HYPER*_PREDSTAT)
    std::vector<std::pair<void *, size_t>> host_bufs_;
    // sampling / speculation
    float samp_invT_ = 0.0f;        // MTP drafts: 1 / temperature when sampling (coupled to the noise), 0: greedy drafts
    SamplingParams samp_;           // this request's sampling (the drafts' truncation)
    int64_t mtp_pos_ = 0;           // the position of the token the next MTP pass drafts
    double mtp_p_ = 1.0;            // the MTP block's probability of its last draft
    double mtp_pmin_ = 0.5;
    bool mtp_full_ = false;         // <ENV>_MTP_FULL (test): every MTP row through the whole block
    double cpu_ema_ = 8.0, mtp_cpu_max_ = 1e30;   // the MTP gate: EMA of row 0's CPU experts per MoE layer and step
    // forward state
    unsigned fwd_counter_ = 0;
    bool graphs_ready_ = false;
    bool debug_ = false, nocpu_ = false;
    int last_nt_ = 0;
    int mc_max_ = 4;                // prefill chunks per layer pass (<ENV>_MC; 1: one at a time)
    int mc_n_ = 1, mc_nt_[8] = {}, mc_pos_[8] = {};   // the chunks of the current bulk forward
    int mc_last_ = 1;               // chunks of the last forward_multi
    int up_parts_ = 2;
    int stream_min_ = 256;
    int pred_mode_ = 0;             // <ENV>_PRED: the next layer's routing predicted on GPU 0 for the CPU's prefetch
    bool pred_early_ = false;       // <ENV>_PRED_EARLY=1
    int pred_k_ = 0;                // <ENV>_PREDSTAT (measurement)
    uint64_t pred_hit_ = 0, pred_tot_ = 0, pred_cpu_ = 0, pred_n_ = 0;
    FILE * route_trace_ = nullptr;  // <ENV>_ROUTE_TRACE=file: expert ids of every routed row (no text), for placement studies
    // elastic placement (<ENV>_ELASTIC=0: off): the attention caches live on the CUDA virtual memory API and map only the
    // positions in use (kv_pos_); the rest of their full-context memory holds extra expert slots (the placement's)
    bool elastic_ = false, elastic_log_ = false;
    int kv_pos_ = 0, kv_step_ = 8192;
    int req_steps_ = 0;             // decode forwards of the current request (the caches shrink after 32: not for side requests)
    // decode graph launches from their own threads (<ENV>_PAR_LAUNCH=1; measured: no gain, off)
    struct Launcher { std::thread th; std::atomic<unsigned> go{0}, done{0}; cudaGraphExec_t ge = nullptr; };
    std::vector<std::unique_ptr<Launcher>> launchers_;
    std::atomic<bool> launch_stop_{false};
    double t_launch_[8] = {}; uint64_t n_launch_ = 0;
};

template <typename T> T * TpEngine::kv_alloc(TpDevice & dev, int heads, size_t pitch_elems, size_t row_elems, int div) {
    if (!elastic_ || heads <= 0) return dev.alloc<T>((size_t) std::max(heads, 1) * pitch_elems);
    CUDA_CHECK(cudaSetDevice(dev.id));
    auto b = std::make_unique<VmmBuf>();
    b->reserve(dev.id, (size_t) heads * pitch_elems * sizeof(T));
    dev.kvs.push_back({b.get(), pitch_elems * sizeof(T), row_elems * sizeof(T), heads, div});
    dev.lend_limit += b->reserved();
    place_->set_lend_limit(dev.g, dev.lend_limit);
    dev.used += b->reserved();
    T * p = (T *) b->ptr();
    dev.vmm.push_back(std::move(b));
    return p;
}

// routing trace record: int32 type (1 prompt rows, 2 decode rows), rows, MoE layers, K; then per MoE layer rows x K int16 ids
template <typename F> void TpEngine::trace_routing(int type, int nrows, F && id) {
    if (!route_trace_ || nrows <= 0) return;
    const int K = m_.n_expert_used;
    int nm = 0;
    for (int il = 0; il < m_.n_layer; ++il) nm += is_moe(il);
    const int32_t hdr[4] = {type, nrows, nm, K};
    fwrite(hdr, sizeof hdr, 1, route_trace_);
    std::vector<int16_t> v((size_t) nrows * K);
    for (int il = 0; il < m_.n_layer; ++il) {
        if (!is_moe(il)) continue;
        for (int i = 0; i < nrows * K; ++i) v[i] = (int16_t) id(il, i);
        fwrite(v.data(), sizeof(int16_t), v.size(), route_trace_);
    }
    fflush(route_trace_);
}

} // namespace hyper
