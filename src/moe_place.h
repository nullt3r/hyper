// Expert placement shared by every engine: which expert of which MoE layer lives where (a GPU slot, the CPU, an elastic
// extra slot), the compact pinned host copy the CPU computes from and the GPUs stream from, the routing scores and the
// exchanges that follow them, the tensor-parallel prefill streaming lists and the elastic extra slots.
//
// An engine decides the owners of a layer's experts when it loads them (quotas per device, or one device per layer);
// everything after that -- the device arrays and tables, the host copy, swaps, fills, evictions, stream lists -- is here,
// the same for every model. The host copy is compact: only the experts the CPU may compute (the CPU's, and those in
// elastic extra slots, which can go back to the CPU) have rows in it; a swap with a base-slot expert is an exchange of
// rows (the GPU expert's rows take the CPU expert's compact slot).
#pragma once
#include "cpu_moe.h"
#include "devices.h"
#include "experts.h"
#include "kernels4.cuh"
#include "vmm.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace hyper {

// a device's part of one MoE layer
struct MoeLayerDev {
    MoeDev moe;                      // the expert arrays and the slot table (slot[e] = -1: not on this device)
    int * owner = nullptr;           // [E] on the device: every expert's owner (device index or CPU_OWNER)
    int * owner_bulk = nullptr;      // [E] the owner table of the bulk (prefill) path: a streamed expert as its streaming device
    // elastic: the arrays as virtual ranges with room for every expert; the base slots [0, nbase) stay mapped, the extra
    // slots nbase.. hold xe[i] (-1: a free slot)
    VmmBuf * xg = nullptr, * xu = nullptr, * xd = nullptr;
    int nbase = 0;
    size_t xbase = 0;
    std::vector<int> xe;
    // tensor-parallel prefill streaming: this device's share of the layer's CPU experts, in one list or two halves
    // (st_list[h] goes through staging buffer h), with their slot tables on the device
    int * st_slot[2] = {};
    std::vector<int> st_list[2];
    bool cpu_keep = false;           // a streamed chunk leaves some CPU experts on the CPU (the CPU publishes too)
    bool present() const { return moe.slot != nullptr; }
};

// a MoE layer's placement on the host
struct MoeLayerHost {
    ExpertTensors t;
    HostExperts hx;                        // the compact pinned copy
    std::vector<int> owner;                // [E]
    std::vector<std::vector<int>> slot;    // [ndev][E]: the expert's slot on that device, -1
    std::vector<int> cslot;                // [E]: index in the compact copy, -1 (the CPU's experts and the extra-slot ones)
    std::vector<char> fixed;               // [E]: never swapped (an engine's pooled experts)
    std::vector<double> score;             // decayed routing counts
    std::vector<uint64_t> last_count;      // the CPU side's counts at the last rebalance
    std::vector<char> pend;                // elastic: on its way into an extra slot
    bool loaded = false, tables_dirty = false, stream_dirty = false;
    size_t gb() const { return t.gb; }
    size_t db() const { return t.db; }
};

class MoePlace {
public:
    struct Params {
        std::string tag = "hyper", env = "HYPER";   // log prefix; env prefix of the knobs (<env>_ADAPT, _ADAPT_EVERY, _ADAPT_BUDGET,
                                                    // _DECAY, _PROMPT_W, _ADAPT_MIN, _ADAPT_RATIO, _ADAPT_LOG, _ELASTIC, _ELASTIC_LOG,
                                                    // _ELASTIC_FILL, _ELASTIC_MIN, _ELASTIC_NOHEADROOM, _EP_WEIGHTS, _STREAM_CPU, _STREAM_CPU_MAX)
        int E = 0, K = 0, nl = 0;                    // experts per layer, routed per token, layer slots (an MTP block included)
        int ff = 0, n = 0;                           // expert hidden size, model width
        float clamp = 0.0f;                          // SwiGLU limit
        bool tp = true;                              // tensor-parallel layers: every device takes part (tables, lists); else only the owners
        bool elastic_ok = true;                      // the engine's caches can lend memory for extra slots (false: never elastic)
        int halves = 1;                              // TP streaming lists per device: 1 (one list), 2 (two halves); 0: none (the engine streams its own way)
        int adapt_budget = 64;                       // exchanges per periodic rebalance
        bool adapt_after_prompt = true;              // an unbounded rebalance after a prompt's routing
        int fill_unit = 256;                         // elastic hand-over pacing: 16 decode steps per this many fills
        double stream_cpu = 0.0, stream_cpu_max = 0.5;   // a single streamed chunk leaves min(max, stream_cpu / nt) of the CPU experts on the CPU (0: off)
    };
    MoePlace() = default;
    ~MoePlace();
    MoePlace(const MoePlace &) = delete;
    MoePlace & operator=(const MoePlace &) = delete;

    void init(const Params & p, const std::vector<DevBase *> & devs, CpuMoe * cpu);
    // calibration statistics (HYPER4_STATS, written by save_stats): the placement order and the score prior
    void load_stats(const char * path, int n_layer);
    const std::vector<std::vector<uint64_t>> & stats() const { return stats_; }
    const std::vector<uint64_t> * stats_of(int il) const { return il < (int) stats_.size() ? &stats_[il] : nullptr; }
    // tensor-parallel quotas: experts per layer per device from the free VRAM (free_bytes after the elastic caches'
    // reserve), logged; eb: bytes of one expert (average over the MoE layers), eb_max: of the largest layer
    std::vector<int> plan_tp(const std::vector<size_t> & free_bytes, double reserve_gib, double frac, size_t eb, size_t eb_max,
                             int n_moe_layers, bool stream, double stage_factor);
    // the experts of layer il by the owners the engine decided (device index or CPU_OWNER): device arrays, tables, the
    // compact host copy (the CPU's experts + host_more, e.g. pooled experts the CPU computes in the bulk path), the CPU
    // side's layer. elastic: on the devices whose lend limit is set, the arrays are virtual ranges with room for extra slots
    void load_layer(int il, const ExpertTensors & t, const std::vector<int> & owner, const std::vector<char> * host_more = nullptr,
                    const std::vector<char> * fixed = nullptr, const std::vector<int> * owner_bulk = nullptr,
                    const std::vector<char> * on_devs = nullptr);   // on_devs: the devices that take part (default: all when tp, else the owners)
    MoeLayerDev & dev(int g, int il) { return dev_[g][il]; }
    const MoeLayerDev & dev(int g, int il) const { return dev_[g][il]; }
    MoeLayerHost & host(int il) { return host_[il]; }
    const MoeLayerHost & host(int il) const { return host_[il]; }
    int nl() const { return p_.nl; }
    double t_hostcopy() const { return t_hostcopy_; }

    // ---- tensor-parallel prefill streaming (halves > 0) ----
    // staging buffer sizes from the layer's CPU count (the caller allocates the buffers once every layer is loaded)
    void size_staging(int il);
    // the lists of layer il from the current placement: the CPU experts dealt to the devices by link bandwidth
    void rebuild_stream(int il);
    void rebuild_dirty_streams();
    // a streamed single chunk of nt tokens: the share of the CPU experts left on the CPU (lists rebuilt on a change)
    void set_stream_cpu_for(int nt, int stream_min);
    void set_stream_cpu(double share);
    double stream_cpu_built() const { return stream_cpu_built_; }
    // part p of np of list `lst` (half sb for halves == 2) of layer il into staging buffer sb on the device's copy stream:
    // the first part waits for the buffer, the last one records ev_up[sb]
    void upload_stage(int g, int il, int sb, int p = 0, int np = 1);

    // ---- adaptive placement ----
    bool adapt() const { return adapt_; }
    bool elastic() const { return elastic_; }
    // before a forward: decode steps hand finished fills over and rebalance on the schedule (after a prompt, then every
    // adapt_every steps); bulk forwards rebuild dirty stream lists
    void before_forward(bool bulk);
    // prompt routing of a bulk forward: ids of count pairs of layer il, weighted below decode routing
    void score_prompt(int il, const int * ids, int count);
    void prompt_done() { if (adapt_) prompt_routed_ = true; }
    void rebalance(int max_swaps);
    std::function<void(int il)> on_changed;   // after a layer's placement changed (tables uploaded)

    // ---- elastic extra slots ----
    // the lending pool per device: the full size of the elastic caches (set before the layers load), and what of it the
    // caches have mapped / want for one growth step (the engine's caches)
    void set_lend_limit(int g, size_t bytes) { lend_limit_[g] = bytes; }
    size_t lend_limit(int g) const { return lend_limit_[g]; }
    std::function<size_t(int g)> kv_mapped, kv_headroom;
    size_t extra_mapped(int g) const;   // bytes the extra slots hold beyond the base slots
    size_t lend_free(int g) const;
    void evict_extra(int g, size_t bytes);   // give back at least `bytes` of extra slots on device g (the coldest first)
    int fill_extra(int budget);              // hottest CPU experts into new extra slots (a background batch)
    void fill_apply();
    void fill_wait(bool cancel);
    void fill_join() { if (fill_th_.joinable()) fill_th_.join(); }   // the copies done, the hand-over still at the due step
    void before_kv_change();                 // the caches are about to be mapped / unmapped: fills cancelled, nothing in flight

    // routing statistics: the loaded ones plus what the CPU side has seen since
    void save_stats(const std::string & path);
    int n_swaps() const { return n_swaps_; }
    double t_rebalance() const { return t_rebalance_; }   // ms

private:
    void upload_tables(int il);
    void sync_devices();
    Params p_;
    std::vector<DevBase *> devs_;
    CpuMoe * cpu_ = nullptr;
    std::vector<std::vector<uint64_t>> stats_;
    std::vector<MoeLayerHost> host_;
    std::vector<std::vector<MoeLayerDev>> dev_;   // [g][il]
    std::vector<std::vector<std::unique_ptr<VmmBuf>>> vmm_;   // [g]
    std::vector<cudaStream_t> fstream_;           // elastic fills per device
    std::vector<size_t> lend_limit_;
    std::vector<double> wbw_;
    double t_hostcopy_ = 0;
    // adaptive placement
    bool adapt_ = true, elastic_ = false, elastic_log_ = false, prompt_routed_ = false;
    double adapt_decay_ = 0.92, prompt_weight_ = 0.25, adapt_min_ = 6.0, adapt_ratio_ = 1.25, elastic_min_ = 1.0;
    int adapt_every_ = 16, adapt_budget_ = 64, elastic_fill_ = 1 << 20, steps_ = 0, n_swaps_ = 0;
    double t_rebalance_ = 0;
    bool adapt_log_ = false;
    uint8_t * swap_buf_ = nullptr;
    size_t swap_buf_bytes_ = 0;
    // streaming
    double stream_cpu_ = 0.0, stream_cpu_max_ = 0.5, stream_cpu_built_ = -1.0;
    // elastic fills
    struct Fill { int il, e, g, k; bool copied; };
    std::vector<Fill> fill_list_;
    std::thread fill_th_;
    std::atomic<bool> fill_busy_{false}, fill_cancel_{false};
    long dec_steps_ = 0, fill_due_ = 0;
};

} // namespace hyper
