// CPU side of the MoE: experts that do not fit in VRAM stay in RAM (GGUF blocks, mmapped) and are evaluated
// with ggml's CPU dot products. GPU0 publishes each layer's input and routing into mapped host memory
// (CpuMoeRec); a thread team computes the CPU-owned experts and publishes the weighted sum (CpuMoeOut).
#pragma once
#include "gguf.h"
#include "kernels4.cuh"

#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hyper {

// back a filled host region with 2 MB pages: page faults only get them while free RAM is unfragmented (a large page cache
// leaves the engines' expert copies half on 4 KB pages, ~6 % less CPU expert bandwidth through TLB misses);
// MADV_COLLAPSE (Linux 6.1+) compacts and collapses synchronously. Call before the region is pinned.
inline void collapse_huge(void * p, size_t bytes) {
    const uintptr_t H = 2u << 20, a0 = ((uintptr_t) p + H - 1) & ~(H - 1), a1 = ((uintptr_t) p + bytes) & ~(H - 1);
    if (a1 > a0 && !getenv("HYPER_NO_COLLAPSE")) madvise((void *) a0, a1 - a0, 25 /* MADV_COLLAPSE */);
}

struct CpuExpertLayer {
    GType tg = GType::F32, td = GType::F32;
    const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr;   // [n_expert][...] GGUF data
    size_t gate_bytes = 0, down_bytes = 0;                              // bytes per expert matrix
    std::vector<uint8_t> owned;                                         // [n_expert]
    std::vector<int> cslot;   // optional: expert e lives at index cslot[e] of gate/up/down (compact copy)
    size_t index(int e) const { return cslot.empty() ? (size_t) e : (size_t) cslot[e]; }
};

// the CPU MoE's Q8_K activation quantizer (tests: must match ggml's bit for bit)
void q8k_test_quantize(const float * x, void * y, int64_t k);

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
    // decode path, last expert group in parts: gate/up tasks left / hidden rows quantized per part, down parts done per
    // row task, the down rows' resumable accumulators [pair][row][8]
    std::vector<std::atomic<int>> part_left_, part_ready_, dprog_;
    std::vector<std::atomic<int>> ck_left_;   // decode: expert groups still to finish each output chunk
    std::vector<float> acc_;
    // HYPER_CPUPROF: how many of a job's CPU experts the previous token's job at the same layer also had
    std::vector<std::vector<int>> prev_e_;
    uint64_t prof_hit_ = 0, prof_tot_ = 0;
    // HYPER_CPUPROF: from the end of a job to the next job's record arriving (the GPUs' share of the critical path)
    std::chrono::steady_clock::time_point prof_last_end_{};
    uint64_t prof_gap_ns_ = 0, prof_gaps_ = 0;
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
