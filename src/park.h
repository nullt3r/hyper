// Parked conversations for an engine's prompt cache (shared by the engines).
//
// The prompt cache keeps one conversation on the GPUs: its tokens (hist), the attention-side caches of those positions and
// recurrent-state snapshots. A request from another conversation (another chat, an agent's side request for a title or a
// summary) would throw that away, and the next request of the first conversation would recompute its whole prompt.
// Instead the current conversation is parked: its attention caches go to pinned RAM and its latest snapshots are kept
// aside. Several conversations stay parked (least recently used out first); a request that matches one of them better
// than the current conversation swaps it back.
//
// An engine describes its per-position GPU buffers as spans (per device) and calls resolve() before matching snapshots.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
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

    // swap thresholds: a conversation is worth parking from `min_park` tokens; a parked one comes back when it shares
    // `min_gain` more tokens with the prompt than the current one
    int min_park = 2048, min_gain = 512;
    // parked conversations (HYPER_PARK_SLOTS), their pinned caches at most this many bytes (HYPER_PARK_GB), snapshots kept
    // per parked conversation (the latest: a conversation that comes back continues where it stopped)
    int max_slots = getenv("HYPER_PARK_SLOTS") ? std::max(1, atoi(getenv("HYPER_PARK_SLOTS"))) : 4;
    size_t max_bytes = (size_t) ((getenv("HYPER_PARK_GB") ? atof(getenv("HYPER_PARK_GB")) : 32.0) * 1073741824.0);
    int park_snaps = 4;
    // the tokens that start a message (<|im_start|>; GLM: <|user|>, <|assistant|>, ...): a prompt that first differs inside
    // the conversation's last message is the same conversation re-rendered (a template's text of an answer is not always
    // the generated one)
    std::vector<int> msg_tokens;
    // called before a parked conversation of n positions is copied back (elastic caches: map them first)
    std::function<void(int)> before_restore;

    ~ConvPark() {
        for (auto & s : slots_) free_bufs(s.host);
        free_bufs(tmp_);
    }

    // Before the snapshot match of a new prompt: parks the current conversation or swaps a parked one back.
    // Returns the length of the common prefix of the prompt with the (possibly swapped-in) current conversation.
    int resolve(const std::vector<int> & prompt, std::vector<int> & hist, std::vector<Snap> & snaps,
                std::vector<std::vector<float *>> & pool, const std::vector<Dev> & devs, const char * tag) {
        auto common = [&](const std::vector<int> & h) {
            size_t n = 0;
            while (n < prompt.size() && n < h.size() && prompt[n] == h[n]) ++n;
            return (int) n;
        };
        // (what a conversation saves: the latest snapshot inside the common prefix -- the recurrent state resumes there)
        auto usable = [](const std::vector<Snap> & ss, int l) {
            int u = 0;
            for (auto & sn : ss) if (sn.pos <= l - 1 && sn.pos > u) u = sn.pos;
            return u;
        };
        const int L = common(hist), cur = (int) hist.size(), U = usable(snaps, L);
        if (L < cur && L < (int) prompt.size() && L > 0) {   // (where a re-rendered history leaves the generated tokens)
            fprintf(stderr, "%s: prompt leaves the cached tokens at %d of %d: cached", tag, L, cur);
            for (int q = std::max(0, L - 3); q < std::min(cur, L + 4); ++q) fprintf(stderr, " %d", hist[q]);
            fprintf(stderr, " | prompt");
            for (int q = std::max(0, L - 3); q < std::min((int) prompt.size(), L + 4); ++q) fprintf(stderr, " %d", prompt[q]);
            fprintf(stderr, "\n");
        }
        int best = -1, Lp = -1, Up = -1;
        for (int i = 0; i < (int) slots_.size(); ++i) {
            const int l = common(slots_[i].hist), u = usable(slots_[i].snaps, l);
            if (u > Up) { Up = u; Lp = l; best = i; }
        }
        if (best >= 0 && Up > U + min_gain) {   // a parked conversation comes back; the current one takes its slot
            Slot & s = slots_[best];
            const bool keep = cur >= min_park;
            if (keep) copy(tmp_, cur, true, devs);
            if (before_restore) before_restore((int) s.hist.size());
            copy(s.host, (int) s.hist.size(), false, devs);
            fprintf(stderr, "%s: resumed a parked conversation of %zu tokens (%zu parked)\n", tag, s.hist.size(), slots_.size());
            std::swap(hist, s.hist);
            std::swap(snaps, s.snaps);
            std::swap(s.host, tmp_);   // (the slot keeps the parked caches; the restored ones' buffers become the staging set)
            if (keep) {
                thin(s.snaps, s.hist, pool);
                s.used = ++clock_;
            } else {
                for (auto & sn : s.snaps) pool.push_back(sn.h);
                free_bufs(tmp_);
                std::swap(tmp_, s.host);
                slots_.erase(slots_.begin() + best);
            }
            return Lp;
        }
        int last_msg = -1;   // (where the current conversation's last message starts; GLM's answer may end in <|user|>)
        for (int q = cur - 2; q >= 0 && last_msg < 0 && !msg_tokens.empty(); --q) if (is_msg(hist[q])) last_msg = q;
        if (cur >= min_park && L + min_park < cur && !(last_msg >= 0 && L > last_msg)) {   // another conversation: the current one is parked
            // the prompt still shares [0, L): the latest snapshot inside it stays with the current conversation
            // (an edited earlier message keeps its prefix; the parked copy keeps its latest snapshots to continue)
            Snap kept{};
            int ki = -1;
            for (int i = 0; i < (int) snaps.size(); ++i)
                if (snaps[i].pos <= L - 1 && (ki < 0 || snaps[i].pos > snaps[ki].pos)) ki = i;
            if (ki >= 0) { kept = snaps[ki]; snaps.erase(snaps.begin() + ki); }
            int k = -1;
            for (int i = 0; i < (int) slots_.size() && k < 0; ++i)   // an earlier state of this conversation is replaced
                if (!slots_[i].hist.empty() && common_with(slots_[i].hist, hist) == slots_[i].hist.size()) k = i;
            if (k < 0 && (int) slots_.size() < max_slots) { slots_.emplace_back(); k = (int) slots_.size() - 1; }
            if (k < 0)
                k = (int) (std::min_element(slots_.begin(), slots_.end(), [](const Slot & a, const Slot & b) { return a.used < b.used; }) -
                           slots_.begin());
            Slot & s = slots_[k];
            for (auto & sn : s.snaps) pool.push_back(sn.h);
            s.snaps.clear();
            copy(s.host, cur, true, devs);
            s.hist = hist;
            s.snaps.swap(snaps);
            if (ki >= 0) snaps.push_back(kept);
            thin(s.snaps, s.hist, pool);
            s.used = ++clock_;
            // the pinned memory budget: least recently used conversations out (never the one just parked)
            for (;;) {
                size_t tot = 0;
                for (auto & x : slots_) for (auto & b : x.host) tot += b.bytes;
                if (tot <= max_bytes || slots_.size() <= 1) break;
                int lru = -1;
                for (int i = 0; i < (int) slots_.size(); ++i)
                    if (i != k && (lru < 0 || slots_[i].used < slots_[lru].used)) lru = i;
                for (auto & sn : slots_[lru].snaps) pool.push_back(sn.h);
                free_bufs(slots_[lru].host);
                slots_.erase(slots_.begin() + lru);
                if (lru < k) --k;
            }
            fprintf(stderr, "%s: parked a conversation of %zu tokens (%zu snapshots kept, %zu parked)\n", tag, slots_[k].hist.size(),
                    slots_[k].snaps.size(), slots_.size());
        }
        return L;
    }

    void clear(std::vector<std::vector<float *>> & pool) {   // every parked conversation dropped (snapshots back)
        for (auto & s : slots_) {
            for (auto & sn : s.snaps) pool.push_back(sn.h);
            free_bufs(s.host);
        }
        slots_.clear();
    }
    void release(std::vector<std::vector<float *>> & pool) {   // (destructor of the engine: snapshots back)
        for (auto & s : slots_) {
            for (auto & sn : s.snaps) pool.push_back(sn.h);
            s.snaps.clear();
        }
    }

private:
    struct HostBuf { void * p = nullptr; size_t bytes = 0; };
    struct Slot { std::vector<int> hist; std::vector<Snap> snaps; std::vector<HostBuf> host; uint64_t used = 0; };
    std::vector<Slot> slots_;
    std::vector<HostBuf> tmp_;   // per device: staging for a swap
    uint64_t clock_ = 0;

    bool is_msg(int t) const { return std::find(msg_tokens.begin(), msg_tokens.end(), t) != msg_tokens.end(); }
    static size_t common_with(const std::vector<int> & a, const std::vector<int> & b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return n;
    }
    // a parked conversation keeps its latest snapshots only (it comes back to continue), and the one at its last message
    // start (the next prompt re-renders the last answer, which may differ from the generated text early on)
    void thin(std::vector<Snap> & snaps, const std::vector<int> & hist, std::vector<std::vector<float *>> & pool) const {
        if ((int) snaps.size() <= park_snaps) return;
        std::sort(snaps.begin(), snaps.end(), [](const Snap & a, const Snap & b) { return a.pos > b.pos; });
        std::vector<Snap> keep(snaps.begin(), snaps.begin() + park_snaps);
        bool msg = false;
        for (size_t i = 0; i < snaps.size(); ++i) {
            const bool start = snaps[i].pos < (int) hist.size() && is_msg(hist[snaps[i].pos]);
            if (i < (size_t) park_snaps) { msg |= start; continue; }
            if (start && !msg) { keep.push_back(snaps[i]); msg = true; }
            else pool.push_back(snaps[i].h);
        }
        snaps.swap(keep);
    }
    static void free_bufs(std::vector<HostBuf> & set) {
        for (auto & b : set) if (b.p) cudaFreeHost(b.p);
        set.clear();
    }

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
                hb.bytes = 0;   // (until the allocation succeeds: a failed one must not leave a size without memory)
                void * p = nullptr;
                check(cudaHostAlloc(&p, need * 5 / 4, cudaHostAllocPortable), "host alloc");
                hb.p = p;
                hb.bytes = need * 5 / 4;
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

} // namespace hyper
