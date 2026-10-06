#include "cpu_moe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hyper {

namespace {
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline const ggml_type_traits_cpu * traits(GType t) { return ggml_get_type_traits_cpu((ggml_type) t); }
// spin with backoff: hot while work is flowing, sleeps after ~200 us of nothing
template <typename P> void spin_until(P && ready) {
    auto t0 = std::chrono::steady_clock::now();
    int i = 0;
    while (!ready()) {
        if (++i < 4096) { __builtin_ia32_pause(); continue; }
        i = 0;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(200)) std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}
} // namespace

CpuMoe::CpuMoe(int n_threads, int n_embd, int ff, int k, CpuMoeRec * recs, CpuMoeOut * outs, int n_slots,
               CpuMoeBulk * bulk, CpuMoeBulkOut * bulk_out)
    : n_threads_(n_threads), n_embd_(n_embd), ff_(ff), k_(k), recs_(recs), outs_(outs), layers_(n_slots), bulk_(bulk), bulk_out_(bulk_out) {
    if (n_embd > 4096 || k > MOE_MAX_USED) throw std::runtime_error("CpuMoe: dimensions too large");
    counts.assign(n_slots, std::vector<uint64_t>(1024, 0));
    ggml_cpu_init();
    const int rows = bulk ? MOE_BULK_ROWS : MAX_NT, P = rows * k;
    h_.resize((size_t) P * ff);
    y_.resize((size_t) P * n_embd);
    qx_.resize((size_t) rows * n_embd * 2 + 4096);
    qh_.resize((size_t) P * ff * 2 + 4096);
    master_ = std::thread([this] { master_loop(); });
    for (int i = 1; i < n_threads_; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
}

CpuMoe::~CpuMoe() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    gen_.fetch_add(1);   // wake workers; with stop_ set they exit
    master_.join();
    for (auto & w : workers_) w.join();
}

void CpuMoe::expect(unsigned counter, const std::vector<int> & slots, bool bulk) {
    if (bulk && !bulk_) throw std::runtime_error("CpuMoe: no bulk record");
    std::lock_guard<std::mutex> lk(mu_);
    for (int s : slots) queue_.push_back({counter, s, bulk});
    pending_ += (int) slots.size();
    cv_.notify_all();
}

void CpuMoe::save_stats(const std::string & path) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("save_stats: cannot open " + path);
    const int n = (int) counts.size(), w = 1024;
    fwrite(&n, 4, 1, f); fwrite(&w, 4, 1, f);
    for (auto & c : counts) fwrite(c.data(), 8, w, f);
    fclose(f);
}

std::string CpuMoe::state() {
    std::lock_guard<std::mutex> lk(mu_);
    char buf[160];
    snprintf(buf, sizeof buf, "job fwd %u slot %d phase %d, queued %zu, pending %d", cur_counter_.load(), cur_slot_.load(),
             cur_phase_.load(), queue_.size(), pending_);
    return buf;
}

void CpuMoe::drain() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return pending_ == 0; });
}

void CpuMoe::master_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = queue_.front();
            queue_.pop_front();
        }
        const unsigned want = job.counter * 64u + (unsigned) job.slot;
        cur_counter_ = job.counter; cur_slot_ = job.slot; cur_phase_ = 1;
        volatile unsigned & rseq = job.bulk ? bulk_->seq : recs_[job.slot].seq;
        bool run = false;
        spin_until([&] {
            const unsigned s = rseq;
            if (s == want) { run = true; return true; }
            if ((int) (s - want) > 0) return true;   // overwritten by a later forward: the GPU did not need us
            std::lock_guard<std::mutex> lk(mu_);
            return stop_;
        });
        std::atomic_thread_fence(std::memory_order_acquire);
        cur_phase_ = 2;
        const int rows = job.bulk ? MOE_BULK_ROWS : MAX_NT;
        const int nt = std::min(job.bulk ? bulk_->nt : recs_[job.slot].nt, rows);
        const int * ids = job.bulk ? &bulk_->ids[0][0] : &recs_[job.slot].ids[0][0];
        const float * wts = job.bulk ? &bulk_->wts[0][0] : &recs_[job.slot].wts[0][0];
        const float * x = job.bulk ? &bulk_->x[0][0] : &recs_[job.slot].x[0][0];
        float * y = job.bulk ? &bulk_out_->y[0][0] : &outs_[job.slot].y[0][0];
        if (run) {
            auto & cnt = counts[job.slot];
            for (int t = 0; t < nt; ++t)
                for (int j = 0; j < k_; ++j) { const int e = ids[t * MOE_MAX_USED + j]; if (e >= 0 && e < (int) cnt.size()) cnt[e]++; }
            run_layer(job.slot, nt, ids, wts, x, y);
        }
        cur_phase_ = 0;
        std::atomic_thread_fence(std::memory_order_release);
        (job.bulk ? bulk_out_->seq : outs_[job.slot].seq) = want;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (--pending_ == 0) cv_done_.notify_all();
        }
    }
}

// claim and run tasks of generation g until none are left (or the generation moved on)
void CpuMoe::run_tasks(unsigned g) {
    Task & T = tasks_[g & 1];
    const int n = T.n;
    const void * fn = T.fn;
    auto call = T.call;
    uint64_t w = work_.load(std::memory_order_acquire);
    for (;;) {
        if ((unsigned) (w >> 32) != g) return;
        const int i = (int) (w & 0xffffffffu);
        if (i >= n) return;
        if (!work_.compare_exchange_weak(w, w + 1, std::memory_order_acq_rel)) continue;   // w reloaded
        call(fn, i);
        T.done.fetch_add(1, std::memory_order_release);
        w = work_.load(std::memory_order_acquire);
    }
}

template <typename F> void CpuMoe::parallel(int n, F && fn) {
    if (n <= 0) return;
    const unsigned g = gen_.load(std::memory_order_relaxed) + 1;
    Task & T = tasks_[g & 1];
    T.n = n;
    T.fn = &fn;
    T.call = [](const void * f, int i) { (*(const F *) f)(i); };
    T.done.store(0, std::memory_order_relaxed);
    work_.store((uint64_t) g << 32, std::memory_order_release);
    gen_.store(g, std::memory_order_release);
    run_tasks(g);
    while (T.done.load(std::memory_order_acquire) < n) __builtin_ia32_pause();
}

void CpuMoe::worker_loop(int) {
    unsigned seen = gen_.load();
    for (;;) {
        spin_until([&] { return gen_.load(std::memory_order_acquire) != seen; });
        seen = gen_.load(std::memory_order_acquire);
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stop_) return;
        }
        run_tasks(seen);
    }
}

void CpuMoe::run_layer(int slot, int nt, const int * ids, const float * wts, const float * x, float * yout) {
    const CpuExpertLayer & L = layers_[slot];
    const int k = k_, n = n_embd_, ff = ff_;
    // the CPU-owned (token, expert) pairs, grouped by expert: each weight row is then read once per layer
    auto & pairs = pairs_;
    pairs.clear();
    std::vector<uint8_t> tok(nt, 0);
    for (int t = 0; t < nt; ++t)
        for (int j = 0; j < k; ++j) {
            const int e = ids[t * MOE_MAX_USED + j];
            if (e < 0 || e >= (int) L.owned.size()) continue;   // garbage routing (NaN logits): the GPU side reports it
            if (L.owned[e]) { pairs.push_back({t, e, wts[t * MOE_MAX_USED + j]}); tok[t] = 1; }
        }
    if (pairs.empty()) return;
    std::stable_sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) { return a.e < b.e; });
    std::vector<int> grp;   // group start indices (+ end)
    for (int p = 0; p < (int) pairs.size(); ++p) if (p == 0 || pairs[p].e != pairs[p - 1].e) grp.push_back(p);
    grp.push_back((int) pairs.size());
    const int G = (int) grp.size() - 1, P = (int) pairs.size();
    const auto * tg = traits(L.tg), * td = traits(L.td);
    const ggml_type vg = tg->vec_dot_type, vd = td->vec_dot_type;
    const auto from_g = traits((GType) vg)->from_float, from_d = traits((GType) vd)->from_float;
    const size_t qx_row = ggml_row_size(vg, n), qh_row = ggml_row_size(vd, ff);
    if (qx_.size() < (size_t) nt * qx_row) qx_.resize((size_t) nt * qx_row);
    if (qh_.size() < (size_t) P * qh_row) qh_.resize((size_t) P * qh_row);
    if (h_.size() < (size_t) P * ff) h_.resize((size_t) P * ff);
    if (y_.size() < (size_t) P * n) y_.resize((size_t) P * n);
    // activations to the dot product's partner type (Q8_K for k-quants, Q8_1 for Q5_1, ...)
    std::vector<int> qt;
    for (int t = 0; t < nt; ++t) if (tok[t]) qt.push_back(t);
    parallel((int) qt.size(), [&](int i) { const int t = qt[i]; from_g(x + (size_t) t * 4096, qx_.data() + t * qx_row, n); });
    const size_t g_row = L.gate_bytes / ff, d_row = L.down_bytes / n;
    constexpr int RC = 16;
    const int gu_chunks = (ff + RC - 1) / RC;
    parallel(G * gu_chunks, [&](int task) {
        const int gi = task / gu_chunks, p0 = grp[gi], p1 = grp[gi + 1];
        const int r0 = (task % gu_chunks) * RC, r1 = std::min(ff, r0 + RC);
        const uint8_t * gb = L.gate + L.index(pairs[p0].e) * L.gate_bytes, * ub = L.up + L.index(pairs[p0].e) * L.gate_bytes;
        for (int r = r0; r < r1; ++r)
            for (int p = p0; p < p1; ++p) {
                float g, u;
                const void * qx = qx_.data() + pairs[p].t * qx_row;
                tg->vec_dot(n, &g, 0, gb + r * g_row, 0, qx, 0, 1);
                tg->vec_dot(n, &u, 0, ub + r * g_row, 0, qx, 0, 1);
                h_[(size_t) p * ff + r] = silu(g) * u;
            }
    });
    parallel(P, [&](int p) { from_d(h_.data() + (size_t) p * ff, qh_.data() + p * qh_row, ff); });
    const int d_chunks = (n + 31) / 32;
    parallel(G * d_chunks, [&](int task) {
        const int gi = task / d_chunks, p0 = grp[gi], p1 = grp[gi + 1];
        const int r0 = (task % d_chunks) * 32, r1 = std::min(n, r0 + 32);
        const uint8_t * db = L.down + L.index(pairs[p0].e) * L.down_bytes;
        for (int r = r0; r < r1; ++r)
            for (int p = p0; p < p1; ++p) td->vec_dot(ff, &y_[(size_t) p * n + r], 0, db + r * d_row, 0, qh_.data() + p * qh_row, 0, 1);
    });
    // per token: the weighted sum of its CPU pairs (pairs of one token in expert order: deterministic)
    std::vector<std::vector<int>> by_tok(nt);
    for (int p = 0; p < P; ++p) by_tok[pairs[p].t].push_back(p);
    parallel((int) qt.size(), [&](int i) {
        const int t = qt[i];
        float * o = yout + (size_t) t * 4096;
        bool first = true;
        for (int p : by_tok[t]) {
            const float w = pairs[p].w, * y = y_.data() + (size_t) p * n;
            if (first) { for (int r = 0; r < n; ++r) o[r] = w * y[r]; first = false; }
            else for (int r = 0; r < n; ++r) o[r] += w * y[r];
        }
    });
}

} // namespace hyper
