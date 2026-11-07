// What every hyper engine shares above its forward strategy (tensor-parallel: TpEngine; layer-split pipeline: Engine7):
// the CPU side of the experts (the hand-off records and the CpuMoe), the experts' placement (MoePlace), the conversation
// history and the prefill progress callback, the prompt-lookup drafter and the generation's output bookkeeping.
#pragma once
#include "cpu_moe.h"
#include "devices.h"
#include "llm.h"
#include "moe_place.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hyper {

// prompt-lookup speculation: the drafts are the tokens that followed the latest earlier occurrence of the last NG tokens of
// the history (prompt + output); the match length (backwards, up to 32) decides how far to trust the continuation, and a poor
// recent acceptance rate raises the bar (a rejected verification costs more than a plain step)
class LookupDrafter {
public:
    static constexpr int NG = 3;
    explicit LookupDrafter(const std::vector<int> & prompt) : hist_(prompt) {}
    void push(int tok) { hist_.push_back(tok); }
    const std::vector<int> & hist() const { return hist_; }
    // up to kmax drafts after the current history into out; returns their count
    int draft(int kmax, double acc_rate, int * out) {
        if (kmax <= 0 || hist_.size() <= (size_t) NG) return 0;
        auto it = last_.find(key(hist_.size()));
        index_upto(hist_.size() - 1);   // (the current suffix itself is indexed after the lookup)
        if (it == last_.end()) return 0;
        const int e0 = it->second, e1 = (int) hist_.size();
        int ml = 0;
        while (ml < 32 && e0 - 1 - ml >= 0 && hist_[e0 - 1 - ml] == hist_[e1 - 1 - ml]) ++ml;
        const int need = acc_rate < 0.35 ? 8 : 5;
        const int kk = ml >= need + 6 ? kmax : ml >= need + 2 ? std::min(kmax, 2) : ml >= need ? 1 : 0;
        int nd = 0;
        for (int j = e0; j < e1 && nd < kk; ++j) out[nd++] = hist_[j];
        return nd;
    }
private:
    uint64_t key(size_t end) const {
        uint64_t h = 1469598103934665603ull;
        for (size_t i = end - NG; i < end; ++i) h = (h ^ (uint32_t) hist_[i]) * 1099511628211ull;
        return h;
    }
    void index_upto(size_t n) { for (; indexed_ <= n; ++indexed_) last_[key(indexed_)] = (int) indexed_; }
    std::vector<int> hist_;
    std::unordered_map<uint64_t, int> last_;   // n-gram -> position after its latest occurrence
    size_t indexed_ = NG;
};

// the generated tokens and the stop condition (the caller's callback, the count)
struct Emitter {
    std::vector<int> out;
    bool stop = false;
    int n_gen = 0;
    const std::function<bool(int)> * on_token = nullptr;
    bool emit(int tok) {
        if (stop) return false;
        out.push_back(tok);
        if (on_token && *on_token && !(*on_token)(tok)) stop = true;
        if ((int) out.size() >= n_gen) stop = true;
        return !stop;
    }
};

class EngineBase : public LLM {
public:
    void set_prefill_progress(std::function<void(int, int, int)> fn) override { prefill_cb_ = std::move(fn); }
    void save_expert_stats(const std::string & path) { place_->save_stats(path); }
    CpuMoe * cpu_moe() { return cpu_.get(); }   // (tests)

protected:
    EngineBase() = default;
    ~EngineBase() override;
    // the CPU side: the hand-off records (one decode record and output per layer slot, one bulk record), the routing
    // prediction records, the CpuMoe over them; pred_early: the early prediction records too (the helpers off unless
    // HYPER_CPU_PREFETCH is set: the tests switch them on per mode)
    void init_cpu_side(int threads, int n_embd, int ff, int K, int slots, float clamp, bool pred_early);
    // the experts' placement over these devices, with the calibration statistics (HYPER4_STATS)
    void init_place(const MoePlace::Params & pp, const std::vector<DevBase *> & devs, int n_layer);
    void release_cpu_side();   // before the devices go: the CpuMoe, the placement (its host copies), the records

    std::unique_ptr<CpuMoe> cpu_;
    std::unique_ptr<MoePlace> place_;
    CpuMoeRec * cpu_rec_ = nullptr, * cpu_pred_ = nullptr, * cpu_pred1_ = nullptr;
    CpuMoeOut * cpu_out_ = nullptr;
    CpuMoeBulk * cpu_bulk_ = nullptr;
    CpuMoeBulkOut * cpu_bulk_out_ = nullptr;
    std::vector<int> hist_;                            // the tokens the state holds
    std::function<void(int, int, int)> prefill_cb_;
    uint64_t samp_seed_ = 0;                           // this request's sampling noise (sampling.h gumbel_noise)
    double acc_rate_ = 0.5;                            // prompt lookup: running fraction of drafted tokens accepted
};

} // namespace hyper
