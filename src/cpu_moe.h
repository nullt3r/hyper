// CPU side of the MoE: experts that do not fit in VRAM stay in RAM (GGUF blocks, mmapped) and are evaluated
// with ggml's CPU dot products. GPU0 publishes each layer's input and routing into mapped host memory
// (CpuMoeRec); a thread team computes the CPU-owned experts and publishes the weighted sum (CpuMoeOut).
#pragma once
#include "gguf.h"
#include "kernels4.cuh"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hyper {

struct CpuExpertLayer {
    GType tg = GType::F32, td = GType::F32;
    const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr;   // [n_expert][...] GGUF data
    size_t gate_bytes = 0, down_bytes = 0;                              // bytes per expert matrix
    std::vector<uint8_t> owned;                                         // [n_expert]
    std::vector<int> cslot;   // optional: expert e lives at index cslot[e] of gate/up/down (compact copy)
    size_t index(int e) const { return cslot.empty() ? (size_t) e : (size_t) cslot[e]; }
};

class CpuMoe {
public:
    // recs/outs: mapped host arrays with one entry per slot
    CpuMoe(int n_threads, int n_embd, int ff, int k, CpuMoeRec * recs, CpuMoeOut * outs, int n_slots,
           CpuMoeBulk * bulk = nullptr, CpuMoeBulkOut * bulk_out = nullptr);
    ~CpuMoe();
    void set_layer(int slot, const CpuExpertLayer & l) { layers_[slot] = l; }
    void set_clamp(float limit) { clamp_ = limit; }
    void set_owned(int slot, int e, bool v) { layers_[slot].owned[e] = v; }   // (between jobs only)
    // decode jobs of this slot compute only the hidden slice [0, a) (the GPUs take the rest, zero-copy)
    void set_split(int slot, int a) { split_[slot] = a; }   // SwiGLU limit: silu(min(g, L)) * clamp(u, -L, L)
    // the next forward (device counter value `counter`) will publish these slots in order
    void expect(unsigned counter, const std::vector<int> & slots, bool bulk = false);
    // wait until every expected slot has been consumed
    void drain();
    std::string state();
    // routing statistics (every token of every published layer): counts[slot][expert]
    void save_stats(const std::string & path);
    std::vector<std::vector<uint64_t>> counts;

private:
    void master_loop();
    void pin(int id);
    void worker_loop(int id);
    // ids/wts rows of MOE_MAX_USED, x/y rows of 4096
    void run_layer(int slot, int nt, const int * ids, const float * wts, const float * x, float * y, int fa);
    // parallel for over [0, n) on the team (master participates)
    template <typename F> void parallel(int n, F && fn);

    int n_threads_, n_embd_, ff_, k_;
    float clamp_ = 0.0f;
    std::vector<int> split_;
    bool prof_ = false, old_path_ = false;
    std::vector<std::atomic<int>> gu_left_, ready_;   // decode path: per expert group
    std::vector<int> task_order_;   // HYPER_CPUPROF: time / bandwidth per layer job
    uint64_t prof_ns_ = 0, prof_bytes_ = 0, prof_jobs_ = 0, prof_experts_ = 0, prof_ph_[6] = {}, prof_wait_ns_ = 0;
    CpuMoeRec * recs_;
    CpuMoeOut * outs_;
    std::vector<CpuExpertLayer> layers_;
    std::thread master_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, cv_done_;
    struct Job { unsigned counter; int slot; bool bulk; };
    std::deque<Job> queue_;
    CpuMoeBulk * bulk_ = nullptr;
    CpuMoeBulkOut * bulk_out_ = nullptr;
    bool stop_ = false;
    int pending_ = 0;
    std::atomic<unsigned> cur_counter_{0};
    std::atomic<int> cur_slot_{-1}, cur_phase_{0};   // phase: 0 idle, 1 waiting for the record, 2 computing
    // team work distribution: work_ = (generation << 32) | next task index, claimed by compare-exchange so a
    // thread still finishing an older generation can never consume an index of the current one
    struct Task { std::atomic<int> done{0}; int n = 0; const void * fn = nullptr; void (*call)(const void *, int) = nullptr; };
    std::atomic<unsigned> gen_{0};
    std::atomic<uint64_t> work_{0};
    Task tasks_[2];
    void run_tasks(unsigned g);
    // scratch
    std::vector<float> h_, y_;
    std::vector<uint8_t> qx_, qh_;
    struct Pair { int t; int e; float w; };
    std::vector<Pair> pairs_;
};

} // namespace hyper
