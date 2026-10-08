// One parked conversation for an engine's prompt cache (shared by the engines).
//
// The prompt cache keeps one conversation on the GPUs: its tokens (hist), the attention-side caches of those positions and
// recurrent-state snapshots. A request from another conversation (an agent's side request for a title or a summary) would
// throw that away, and the next request of the main conversation would recompute its whole prompt. Instead the current
// conversation is parked: its attention caches go to pinned RAM and its snapshots are kept aside; a later request that
// matches the parked conversation better swaps it back.
//
// An engine describes its per-position GPU buffers as spans (per device) and calls resolve() before matching snapshots.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hyper {

template <class Snap>
class ConvPark {
public:
    // positions [0, n) of a buffer: `heads` blocks `pitch` bytes apart, each holding rows(n) rows of `row_bytes`
    // (div 1: n rows; div k: n / k + 1 pooled rows)
    struct Span {
        void * dev = nullptr;
        size_t row_bytes = 0, pitch = 0;
        int heads = 1, div = 1;
        size_t rows(int n) const { return div == 1 ? (size_t) n : (size_t) (n / div + 1); }
    };
    struct Dev { int id = 0; cudaStream_t stream = nullptr; std::vector<Span> spans; };

    // swap thresholds: a conversation is worth parking from `min_park` tokens; the parked one comes back when it shares
    // `min_gain` more tokens with the prompt than the current one
    int min_park = 2048, min_gain = 512;

    ~ConvPark() { for (auto * set : {&host_, &tmp_}) for (auto & b : *set) if (b.p) cudaFreeHost(b.p); }

    // Before the snapshot match of a new prompt: parks the current conversation or swaps the parked one back.
    // Returns the length of the common prefix of the prompt with the (possibly swapped-in) current conversation.
    int resolve(const std::vector<int> & prompt, std::vector<int> & hist, std::vector<Snap> & snaps,
                std::vector<std::vector<float *>> & pool, const std::vector<Dev> & devs, const char * tag) {
        auto common = [&](const std::vector<int> & h) {
            size_t n = 0;
            while (n < prompt.size() && n < h.size() && prompt[n] == h[n]) ++n;
            return (int) n;
        };
        int L = common(hist);
        const int Lp = hist_.empty() ? -1 : common(hist_);
        const int cur = (int) hist.size();
        if (Lp > L + min_gain) {
            const bool keep = cur >= min_park;   // the current conversation is parked in turn
            if (keep) copy(tmp_, cur, true, devs);
            copy(host_, (int) hist_.size(), false, devs);
            std::swap(host_, tmp_);
            fprintf(stderr, "%s: resumed a parked conversation of %zu tokens\n", tag, hist_.size());
            std::swap(hist, hist_);
            std::swap(snaps, snaps_);
            if (!keep) clear(pool);
            L = Lp;
        } else if (cur >= min_park && L + min_park < cur) {
            clear(pool);
            copy(host_, cur, true, devs);
            hist_ = hist;
            snaps_.swap(snaps);
            fprintf(stderr, "%s: parked a conversation of %zu tokens (%zu snapshots)\n", tag, hist_.size(), snaps_.size());
        }
        return L;
    }

    void clear(std::vector<std::vector<float *>> & pool) {
        for (auto & sn : snaps_) pool.push_back(sn.h);
        snaps_.clear();
        hist_.clear();
    }
    void release(std::vector<std::vector<float *>> & pool) { clear(pool); }   // (destructor of the engine: snapshots back)

private:
    struct HostBuf { void * p = nullptr; size_t bytes = 0; };
    std::vector<int> hist_;
    std::vector<Snap> snaps_;
    std::vector<HostBuf> host_, tmp_;   // per device

    static void check(cudaError_t e, const char * what) {
        if (e != cudaSuccess) throw std::runtime_error(std::string("ConvPark: ") + what + ": " + cudaGetErrorString(e));
    }

    // positions [0, n) of every span, per device, to / from a pinned host set (grown on demand)
    static void copy(std::vector<HostBuf> & set, int n, bool to_host, const std::vector<Dev> & devs) {
        if (n <= 0) return;
        set.resize(devs.size());
        for (size_t gi = 0; gi < devs.size(); ++gi) {
            const Dev & d = devs[gi];
            check(cudaSetDevice(d.id), "set device");
            size_t need = 0;
            for (auto & sp : d.spans) need += sp.rows(n) * sp.row_bytes * sp.heads;
            HostBuf & hb = set[gi];
            if (hb.bytes < need) {
                if (hb.p) check(cudaFreeHost(hb.p), "free");
                hb.p = nullptr;
                hb.bytes = need * 5 / 4;
                check(cudaHostAlloc(&hb.p, hb.bytes, cudaHostAllocPortable), "host alloc");
            }
            char * hp = (char *) hb.p;
            for (auto & sp : d.spans) {
                const size_t w = sp.rows(n) * sp.row_bytes;
                if (to_host) check(cudaMemcpy2DAsync(hp, w, sp.dev, sp.pitch ? sp.pitch : w, w, sp.heads, cudaMemcpyDeviceToHost, d.stream), "copy");
                else check(cudaMemcpy2DAsync(sp.dev, sp.pitch ? sp.pitch : w, hp, w, w, sp.heads, cudaMemcpyHostToDevice, d.stream), "copy");
                hp += w * sp.heads;
            }
        }
        for (auto & d : devs) { check(cudaSetDevice(d.id), "set device"); check(cudaStreamSynchronize(d.stream), "sync"); }
    }
};

}  // namespace hyper
