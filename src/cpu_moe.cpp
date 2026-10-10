#include "cpu_moe.h"

#include "ggml-cpu.h"
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "iq3s_dot.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace hyper {

namespace {
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline const ggml_type_traits_cpu * traits(GType t) { return ggml_get_type_traits_cpu((ggml_type) t); }
// activations -> Q8_K with AVX2, bit-identical to ggml's quantize_row_q8_K (on x86 the scalar reference: ~10 us for a
// 4096-row on the critical path of every decode job): same first-occurrence signed max, iscale = -127 / max, the same
// x * iscale products rounded by the same 1.5 * 2^23 trick, min(127, v), int16 sums of 16
struct BlockQ8K { float d; int8_t qs[256]; int16_t bsums[16]; };
void quantize_q8_K_avx2(const float * __restrict x, void * __restrict vy, int64_t k) {
    BlockQ8K * y = (BlockQ8K *) vy;
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int64_t i = 0; i < k / 256; ++i, x += 256) {
        __m256 am = _mm256_setzero_ps();
        for (int j = 0; j < 256; j += 8) am = _mm256_max_ps(am, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + j)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);
        if (!(amax > 0.0f)) {   // (all zero; NaN rows: the reference's scalar path)
            if (amax == 0.0f) { y[i].d = 0; memset(y[i].qs, 0, 256); continue; }
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x, &y[i], 256);
            continue;
        }
        float mx = 0.0f;
        for (int j = 0; j < 256; ++j) if (std::fabs(x[j]) == amax) { mx = x[j]; break; }
        const float iscale = -127.f / mx;
        const __m256 vs = _mm256_set1_ps(iscale), magic = _mm256_set1_ps(12582912.f);
        const __m256i mant = _mm256_set1_epi32(0x007fffff), off = _mm256_set1_epi32(0x00400000), lim = _mm256_set1_epi32(127);
        for (int j = 0; j < 256; j += 32) {
            __m256i v[4];
            for (int q = 0; q < 4; ++q) {
                __m256 pr = _mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 8 * q));
                asm("" : "+x"(pr));   // (no fma contraction: the reference rounds the product first)
                const __m256 f = _mm256_add_ps(pr, magic);
                v[q] = _mm256_min_epi32(_mm256_sub_epi32(_mm256_and_si256(_mm256_castps_si256(f), mant), off), lim);
            }
            // pack 32 int32 -> 32 int8 in order (values within [-128, 127]: saturation never applies)
            const __m256i a = _mm256_packs_epi32(v[0], v[1]), b = _mm256_packs_epi32(v[2], v[3]);
            __m256i c = _mm256_packs_epi16(a, b);
            c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256((__m256i *) (y[i].qs + j), c);
        }
        for (int j = 0; j < 16; ++j) {
            const __m128i q = _mm_loadu_si128((const __m128i *) (y[i].qs + 16 * j));
            const __m256i w = _mm256_cvtepi8_epi16(q);
            __m128i h = _mm_add_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256(w, 1));
            h = _mm_hadd_epi16(h, h); h = _mm_hadd_epi16(h, h); h = _mm_hadd_epi16(h, h);
            y[i].bsums[j] = (int16_t) _mm_extract_epi16(h, 0);
        }
        y[i].d = 1 / iscale;
    }
}
// row dot products: IQ3_S x Q8_K with the grid lookups in scalar registers (+17 % on one core, bit-identical; the
// gate/up phase of a decode job is compute-bound)
void iq3s_vec_dot(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) { iq3s_dot_v1(n, s, vx, vy); }
ggml_vec_dot_t vec_dot_fast(GType t, GType tv) {
    static const bool off = getenv("HYPER_CPU_REFDOT") != nullptr;
    if (t == GType::IQ3_S && tv == GType::Q8_K && !off) return iq3s_vec_dot;
    return traits(t)->vec_dot;
}
ggml_from_float_t from_float_fast(GType t) {
    static const bool off = getenv("HYPER_CPU_REFQ") != nullptr;
    if (t == GType::Q8_K && !off) return quantize_q8_K_avx2;
    return traits(t)->from_float;
}
// spin with backoff: hot while work is flowing (a decode step's layers are ~0.3-0.5 ms apart; waking from a sleep costs
// 50+ us), then real sleeps once nothing has happened for HYPER_SPIN_US (default 3000): an idle server costs no CPU
const long g_spin_us = getenv("HYPER_SPIN_US") ? atol(getenv("HYPER_SPIN_US")) : 3000;
template <typename P> void spin_until(P && ready) {
    auto t0 = std::chrono::steady_clock::now();
    int i = 0;
    while (!ready()) {
        if (++i < 256) { __builtin_ia32_pause(); continue; }   // (~256 pauses: a few microseconds between clock reads)
        i = 0;
        const auto idle = std::chrono::steady_clock::now() - t0;
        if (idle > std::chrono::microseconds(g_spin_us))
            std::this_thread::sleep_for(idle > std::chrono::milliseconds(100) ? std::chrono::microseconds(1000) : std::chrono::microseconds(200));
    }
}
} // namespace

void q8k_test_quantize(const float * x, void * y, int64_t k) { quantize_q8_K_avx2(x, y, k); }

CpuMoe::CpuMoe(int n_threads, int n_embd, int ff, int k, CpuMoeRec * recs, CpuMoeOut * outs, int n_slots,
               CpuMoeBulk * bulk, CpuMoeBulkOut * bulk_out)
    : n_threads_(n_threads), n_embd_(n_embd), ff_(ff), k_(k), recs_(recs), outs_(outs), layers_(n_slots), bulk_(bulk), bulk_out_(bulk_out) {
    if (n_embd > 4096 || k > MOE_MAX_USED) throw std::runtime_error("CpuMoe: dimensions too large");
    counts.assign(n_slots, std::vector<uint64_t>(1024, 0));
    split_.assign(n_slots, ff);
    ggml_cpu_init();
    prof_ = getenv("HYPER_CPUPROF") != nullptr;
    old_path_ = getenv("HYPER_CPU_OLD") != nullptr;
    const int rows = bulk ? MOE_BULK_ROWS : MAX_NT, P = rows * k;
    h_.resize((size_t) P * ff);
    y_.resize((size_t) P * n_embd);
    qx_.resize((size_t) rows * n_embd * 2 + 4096);
    qh_.resize((size_t) P * ff * 2 + 4096);
    master_ = std::thread([this] { pin(0); master_loop(); });
    for (int i = 1; i < n_threads_; ++i) workers_.emplace_back([this, i] { pin(i); worker_loop(i); });
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
        const auto t_seen = std::chrono::steady_clock::now();
        if (prof_ && run && !job.bulk && prof_last_end_.time_since_epoch().count()) {
            const int64_t gap = std::chrono::duration_cast<std::chrono::nanoseconds>(t_seen - prof_last_end_).count();
            if (gap < 2000000) { prof_gap_ns_ += gap; ++prof_gaps_; }   // (gaps between tokens / requests excluded)
        }
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
            const auto t0 = std::chrono::steady_clock::now();
            if (prof_) prof_wait_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - t_seen).count();
            run_layer(job.slot, nt, ids, wts, x, y, job.bulk ? ff_ : split_[job.slot]);
            if (prof_) {
                prof_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                if (++prof_jobs_ % 2000 == 0) {
                    fprintf(stderr, "cpu_moe: %llu jobs, %.3f ms/job, %.1f GB/s, %.2f experts/job, seen->start %.1f us"
                                    " | setup %.1f, to last gu %.1f, last down %.1f, sum %.1f us\n",
                            (unsigned long long) prof_jobs_, prof_ns_ / 1e6 / 2000, prof_bytes_ / (double) prof_ns_, prof_experts_ / 2000.0,
                            prof_wait_ns_ / 2e6, prof_ph_[0] / 2e6, prof_ph_[1] / 2e6, prof_ph_[2] / 2e6, prof_ph_[3] / 2e6);
                    fprintf(stderr, "cpu_moe: job end -> next record %.1f us (%llu gaps)\n", prof_gap_ns_ / 1e3 / std::max<uint64_t>(1, prof_gaps_),
                            (unsigned long long) prof_gaps_);
                    prof_gap_ns_ = prof_gaps_ = 0;
                    fprintf(stderr, "cpu_moe: previous token's experts at the same layer: %.1f %% of %llu\n",
                            100.0 * prof_hit_ / std::max<uint64_t>(1, prof_tot_), (unsigned long long) prof_tot_);
                    prof_hit_ = prof_tot_ = 0;
                    for (auto & v : prof_ph_) v = 0;
                    prof_wait_ns_ = 0;
                    prof_ns_ = 0; prof_bytes_ = 0; prof_experts_ = 0;
                }
            }
        }
        cur_phase_ = 0;
        std::atomic_thread_fence(std::memory_order_release);
        (job.bulk ? bulk_out_->seq : outs_[job.slot].seq) = want;
        if (prof_ && run) prof_last_end_ = std::chrono::steady_clock::now();
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

// pin team thread i to one physical core (HYPER_CPU_PIN=1 enables): the OS otherwise lands two memory-bound threads on
// SMT siblings of one core now and then, which halves their bandwidth; cores are taken by topology (siblings last)
void CpuMoe::pin(int i) {
    static const bool on = getenv("HYPER_CPU_PIN") && atoi(getenv("HYPER_CPU_PIN")) != 0;   // (measured: no gain on the 3970X)
    if (!on) return;
    static std::vector<int> order = [] {   // cpu ids: first sibling of every core, then the second siblings
        std::vector<int> first, second;
        const int n = (int) sysconf(_SC_NPROCESSORS_ONLN);
        std::vector<char> seen(n, 0);
        for (int c = 0; c < n; ++c) {
            if (seen[c]) continue;
            char path[128];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            FILE * f = fopen(path, "r");
            std::vector<int> sib;
            if (f) {   // "0,32" or "0-1"
                char buf[64] = {0};
                if (fgets(buf, sizeof buf, f)) {
                    for (char * q = buf; *q;) {
                        int a = (int) strtol(q, &q, 10), b = a;
                        if (*q == '-') b = (int) strtol(q + 1, &q, 10);
                        for (int x = a; x <= b && x < n; ++x) sib.push_back(x);
                        while (*q && (*q == ',' || *q == ' ')) ++q;
                        if (*q == '\n') break;
                    }
                }
                fclose(f);
            }
            if (sib.empty()) sib.push_back(c);
            for (size_t k = 0; k < sib.size(); ++k) { seen[sib[k]] = 1; (k == 0 ? first : second).push_back(sib[k]); }
        }
        first.insert(first.end(), second.begin(), second.end());
        return first;
    }();
    if (i >= (int) order.size()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(order[i], &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
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

void CpuMoe::run_layer(int slot, int nt, const int * ids, const float * wts, const float * x, float * yout, int fa) {
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
    if (prof_) {
        prof_bytes_ += (uint64_t) G * (2 * L.gate_bytes + L.down_bytes) / ff * fa; prof_experts_ += G;
        if ((int) prev_e_.size() <= slot) prev_e_.resize(slot + 1);
        std::vector<int> cur;
        for (int q = 0; q < G; ++q) cur.push_back(pairs[grp[q]].e);
        for (int e : cur) prof_hit_ += std::find(prev_e_[slot].begin(), prev_e_[slot].end(), e) != prev_e_[slot].end();
        prof_tot_ += cur.size();
        prev_e_[slot] = cur;
    }
    const auto * tg = traits(L.tg), * td = traits(L.td);
    const ggml_type vg = tg->vec_dot_type, vd = td->vec_dot_type;
    const ggml_vec_dot_t gdot = vec_dot_fast(L.tg, (GType) vg);
    const auto from_g = from_float_fast((GType) vg), from_d = from_float_fast((GType) vd);
    const size_t qx_row = ggml_row_size(vg, n), qh_row = ggml_row_size(vd, ff);
    if (qx_.size() < (size_t) nt * qx_row) qx_.resize((size_t) nt * qx_row);
    if (qh_.size() < (size_t) P * qh_row) qh_.resize((size_t) P * qh_row);
    if (h_.size() < (size_t) P * ff) h_.resize((size_t) P * ff);
    if (y_.size() < (size_t) P * n) y_.resize((size_t) P * n);
    // activations to the dot product's partner type (Q8_K for k-quants, Q8_1 for Q5_1, ...)
    std::vector<int> qt;
    for (int t = 0; t < nt; ++t) if (tok[t]) qt.push_back(t);
    const size_t g_row = L.gate_bytes / ff, d_row = L.down_bytes / n;
    if (nt <= MAX_NT && !old_path_) {
        // decode: one parallel phase. Tasks in claim order gu(0) gu(1) down(0) gu(2) down(1) ... down(G-1); the last gate/up
        // task of an expert quantizes its hidden rows and releases the expert's down tasks, so the compute-bound gate/up of
        // one expert overlaps the bandwidth-bound down projection of the previous one
        std::vector<int> qt;
        for (int t = 0; t < nt; ++t) if (tok[t]) qt.push_back(t);
        const auto tj0 = std::chrono::steady_clock::now();   // (HYPER_CPUPROF: phase times of the decode job)
        for (int t : qt) from_g(x + (size_t) t * 4096, qx_.data() + t * qx_row, n);
        static const int RG = getenv("HYPER_CPU_RG") ? atoi(getenv("HYPER_CPU_RG")) : 16, RD = getenv("HYPER_CPU_RD") ? atoi(getenv("HYPER_CPU_RD")) : 32;
        // guided chunking: full-size row chunks, except that the last ~2 chunks per thread of a phase are split in 4 (the
        // phase's tail, where threads idle for up to one chunk, shrinks accordingly)
        static const int tail_mult = getenv("HYPER_CPU_TAIL") ? atoi(getenv("HYPER_CPU_TAIL")) : 0;   // (measured: smaller tail chunks lose bandwidth)
        if ((int) gu_left_.size() < G) { gu_left_ = std::vector<std::atomic<int>>(G + 16); ready_ = std::vector<std::atomic<int>>(G + 16); }
        auto & order = task_order_;
        order.clear();
        // task codes: bit 30 down, bit 29 down part of the last group (part in bits 26..28), group in bits 20..25,
        // r0 / 4 in bits 10..19, r1 / 4 - 1 in bits 0..9 (4096 rows fit)
        auto code = [](int g, int r0, int r1, bool dn, int part) {
            return (dn ? 0x40000000 : 0) | (part >= 0 ? 0x20000000 | (part << 26) : 0) | (g << 20) | ((r0 >> 2) << 10) | (((r1 + 3) >> 2) - 1);
        };
        // balanced: a multiple of the team size of near-equal tasks (dynamic claiming then ends a phase with every thread
        // finishing at about the same time; 128 tasks for 30 threads left 4.3 rounds)
        static const bool bal = getenv("HYPER_CPU_BAL") != nullptr;   // (measured: equal or slightly slower; off)
        auto add = [&](std::vector<int> & v, int g, int a, int b, int R, bool dn, int part) {
            if (bal && tail_mult == 0 && part < 0 && (b - a) % 4 == 0) {
                const int units = (b - a) / 4, nth = n_threads_;
                int nt_ = std::max(1, (int) std::lround((double) units * 4 / R / nth)) * nth;
                nt_ = std::min(nt_, units);
                for (int i = 0, u0 = 0; i < nt_; ++i) {
                    const int u1 = (int) ((long) units * (i + 1) / nt_);
                    v.push_back(code(g, a + 4 * u0, a + 4 * u1, dn, part));
                    u0 = u1;
                }
                return nt_;
            }
            const int tail = std::min(b - a, tail_mult * n_threads_ * R);
            int r = a, cnt = 0;
            for (; r + R <= b - tail; r += R) { v.push_back(code(g, r, r + R, dn, part)); ++cnt; }
            for (; r < b; r += std::max(4, R / 4)) { v.push_back(code(g, r, std::min(b, r + std::max(4, R / 4)), dn, part)); ++cnt; }
            return cnt;
        };
        // the last group's down projection overlaps its own gate/up: gate/up in S parts of whole 256-row blocks; a finished
        // part's hidden rows are quantized (Q8_K is per block: the same bytes) and the down rows continue their IQ4_XS
        // accumulators over that part (ggml's fma chain, paused and resumed: bit-identical)
        // (measured: the down rows read in S partial passes halve the DRAM bandwidth -- off by default; a part-major
        // layout would need the GPUs' streaming copies to change)
        static const int S_env = getenv("HYPER_CPU_SPLIT") ? atoi(getenv("HYPER_CPU_SPLIT")) : 0;
        const int S = S_env;
        const bool split = S > 1 && L.td == GType::IQ4_XS && vd == GGML_TYPE_Q8_K && fa == ff && fa % (256 * S) == 0 && 256 % RG == 0 &&
                           n % RD == 0 && n / RD <= 4096 && S <= 8 && G < 64 && tail_mult == 0;
        std::vector<int> gcount(G);
        for (int g = 0; g < G; ++g) ready_[g].store(0, std::memory_order_relaxed);
        if (!split) {
            gcount[0] = add(order, 0, 0, fa, RG, false, -1);
            for (int g = 1; g < G; ++g) { gcount[g] = add(order, g, 0, fa, RG, false, -1); add(order, g - 1, 0, n, RD, true, -1); }
            add(order, G - 1, 0, n, RD, true, -1);
        } else {
            if ((int) part_left_.size() < S) { part_left_ = std::vector<std::atomic<int>>(8); part_ready_ = std::vector<std::atomic<int>>(8); }
            if ((int) dprog_.size() < n / RD) dprog_ = std::vector<std::atomic<int>>(n / RD);
            const int np_last = grp[G] - grp[G - 1];
            if (acc_.size() < (size_t) np_last * n * 8) acc_.resize((size_t) np_last * n * 8);
            for (int r = 0; r < n / RD; ++r) dprog_[r].store(0, std::memory_order_relaxed);
            for (int g = 0; g < G - 1; ++g) {
                gcount[g] = add(order, g, 0, fa, RG, false, -1);
                if (g > 0) add(order, g - 1, 0, n, RD, true, -1);
            }
            // gate/up part q interleaved with down work that is ready by then: the previous expert's down next to part 1,
            // the down rows' part q - 2 next to part q (a lag of one part: no thread waits on a part still in progress)
            const int pl = fa / S;
            static const int lag = getenv("HYPER_CPU_LAG") ? atoi(getenv("HYPER_CPU_LAG")) : 2;
            std::vector<std::vector<int>> dq(S + 1);
            if (G > 1) add(dq[0], G - 2, 0, n, RD, true, -1);
            for (int q = 0; q < S; ++q) add(dq[q + 1], G - 1, 0, n, RD, true, q);
            // dq[0]: previous expert, dq[q + 1]: down part q; interleave dq[k] with gate/up part k + lag - 1
            int next = 0;
            std::vector<int> a;
            for (int q = 0; q < S; ++q) {
                a.clear();
                part_left_[q].store(add(a, G - 1, q * pl, (q + 1) * pl, RG, false, q), std::memory_order_relaxed);
                part_ready_[q].store(0, std::memory_order_relaxed);
                const int k = q - (lag - 1);
                const std::vector<int> & b = k >= 0 && k <= S ? dq[k] : dq[S];   // (dq[S]: placeholder, never empty-checked)
                const bool use_b = k >= 0 && k <= S;
                for (size_t i = 0, j = 0; i < a.size() || (use_b && j < b.size());) {
                    if (i < a.size()) order.push_back(a[i++]);
                    if (use_b && j < b.size()) order.push_back(b[j++]);
                }
                if (use_b) next = k + 1;
            }
            for (int k = next; k <= S; ++k) order.insert(order.end(), dq[k].begin(), dq[k].end());
            gcount[G - 1] = 0;
        }
        for (int g = 0; g < G; ++g) gu_left_[g].store(gcount[g], std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        const auto tj1 = std::chrono::steady_clock::now();
        std::atomic<int64_t> t_last_gu{0};
        const int nbp = fa / 256 / std::max(S, 1);   // super-blocks of a part
        parallel((int) order.size(), [&](int i) {
            const int c = order[i], g = (c >> 20) & 0x3f, part = c & 0x20000000 ? (c >> 26) & 7 : -1;
            const int p0 = grp[g], p1 = grp[g + 1];
            if (!(c & 0x40000000)) {
                const int r0 = ((c >> 10) & 0x3ff) << 2, r1 = std::min(fa, ((c & 0x3ff) + 1) << 2);
                const uint8_t * gb = L.gate + L.index(pairs[p0].e) * L.gate_bytes, * ub = L.up + L.index(pairs[p0].e) * L.gate_bytes;
                for (int r = r0; r < r1; ++r)
                    for (int p = p0; p < p1; ++p) {
                        float gv, uv;
                        const void * qx = qx_.data() + pairs[p].t * qx_row;
                        gdot(n, &gv, 0, gb + r * g_row, 0, qx, 0, 1);
                        gdot(n, &uv, 0, ub + r * g_row, 0, qx, 0, 1);
                        if (clamp_ > 0.0f) { gv = std::min(gv, clamp_); uv = std::min(std::max(uv, -clamp_), clamp_); }
                        h_[(size_t) p * ff + r] = silu(gv) * uv;
                    }
                if (part >= 0) {
                    if (part_left_[part].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        const int f0 = part * nbp * 256;
                        for (int p = p0; p < p1; ++p)
                            from_d(h_.data() + (size_t) p * ff + f0, qh_.data() + p * qh_row + (size_t) part * nbp * sizeof(BlockQ8K), nbp * 256);
                        part_ready_[part].store(1, std::memory_order_release);
                        if (prof_ && part == S - 1) t_last_gu.store(std::chrono::steady_clock::now().time_since_epoch().count());
                    }
                } else if (gu_left_[g].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    for (int p = p0; p < p1; ++p) from_d(h_.data() + (size_t) p * ff, qh_.data() + p * qh_row, fa);
                    ready_[g].store(1, std::memory_order_release);
                    if (prof_ && g == G - 1) t_last_gu.store(std::chrono::steady_clock::now().time_since_epoch().count());
                }
            } else if (part >= 0) {
                const int r0 = ((c >> 10) & 0x3ff) << 2, r1 = std::min(n, ((c & 0x3ff) + 1) << 2), rt = r0 / RD;
                while (!part_ready_[part].load(std::memory_order_acquire)) __builtin_ia32_pause();
                while (dprog_[rt].load(std::memory_order_acquire) != part) __builtin_ia32_pause();
                const uint8_t * db = L.down + L.index(pairs[p0].e) * L.down_bytes;
                for (int r = r0; r < r1; ++r)
                    for (int p = p0; p < p1; ++p) {
                        float * st = acc_.data() + ((size_t) (p - p0) * n + r) * 8;
                        __m256 acc = part == 0 ? _mm256_setzero_ps() : _mm256_loadu_ps(st);
                        acc = iq4xs_dot_part(db + r * d_row, qh_.data() + p * qh_row, part * nbp, (part + 1) * nbp, acc);
                        if (part == S - 1) y_[(size_t) p * n + r] = iq3s_hsum8(acc);
                        else _mm256_storeu_ps(st, acc);
                    }
                dprog_[rt].store(part + 1, std::memory_order_release);
            } else {
                while (!ready_[g].load(std::memory_order_acquire)) __builtin_ia32_pause();
                const int r0 = ((c >> 10) & 0x3ff) << 2, r1 = std::min(n, ((c & 0x3ff) + 1) << 2);
                const uint8_t * db = L.down + L.index(pairs[p0].e) * L.down_bytes;
                for (int r = r0; r < r1; ++r)
                    for (int p = p0; p < p1; ++p) td->vec_dot(fa, &y_[(size_t) p * n + r], 0, db + r * d_row, 0, qh_.data() + p * qh_row, 0, 1);
            }
        });
        const auto tj2 = std::chrono::steady_clock::now();
        for (int t : qt) {   // weighted sum per token (pairs in expert order: deterministic)
            float * o = yout + (size_t) t * 4096;
            bool first = true;
            for (int p = 0; p < P; ++p) {
                if (pairs[p].t != t) continue;
                const float w = pairs[p].w, * y = y_.data() + (size_t) p * n;
                if (first) { for (int r = 0; r < n; ++r) o[r] = w * y[r]; first = false; }
                else for (int r = 0; r < n; ++r) o[r] += w * y[r];
            }
        }
        if (prof_) {   // ph 0: setup, 1: tasks until the last expert's hidden rows are ready, 2: its down tail, 3: weighted sum
            const auto tj3 = std::chrono::steady_clock::now();
            const int64_t tg = t_last_gu.load();
            prof_ph_[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(tj1 - tj0).count();
            prof_ph_[1] += tg - tj1.time_since_epoch().count();
            prof_ph_[2] += tj2.time_since_epoch().count() - tg;
            prof_ph_[3] += std::chrono::duration_cast<std::chrono::nanoseconds>(tj3 - tj2).count();
        }
        return;
    }
    auto tp = std::chrono::steady_clock::now();
    auto lap = [&](int ph) { if (!prof_) return; auto t = std::chrono::steady_clock::now(); prof_ph_[ph] += std::chrono::duration_cast<std::chrono::nanoseconds>(t - tp).count(); tp = t; };
    lap(0);
    parallel((int) qt.size(), [&](int i) { const int t = qt[i]; from_g(x + (size_t) t * 4096, qx_.data() + t * qx_row, n); });
    lap(1);
    constexpr int RC = 16;
    const int gu_chunks = (fa + RC - 1) / RC;
    parallel(G * gu_chunks, [&](int task) {
        const int gi = task / gu_chunks, p0 = grp[gi], p1 = grp[gi + 1];
        const int r0 = (task % gu_chunks) * RC, r1 = std::min(fa, r0 + RC);
        const uint8_t * gb = L.gate + L.index(pairs[p0].e) * L.gate_bytes, * ub = L.up + L.index(pairs[p0].e) * L.gate_bytes;
        for (int r = r0; r < r1; ++r)
            for (int p = p0; p < p1; ++p) {
                float g, u;
                const void * qx = qx_.data() + pairs[p].t * qx_row;
                gdot(n, &g, 0, gb + r * g_row, 0, qx, 0, 1);
                gdot(n, &u, 0, ub + r * g_row, 0, qx, 0, 1);
                if (clamp_ > 0.0f) { g = std::min(g, clamp_); u = std::min(std::max(u, -clamp_), clamp_); }
                h_[(size_t) p * ff + r] = silu(g) * u;
            }
    });
    lap(2);
    parallel(P, [&](int p) { from_d(h_.data() + (size_t) p * ff, qh_.data() + p * qh_row, fa); });
    lap(3);
    const int d_chunks = (n + 31) / 32;
    parallel(G * d_chunks, [&](int task) {
        const int gi = task / d_chunks, p0 = grp[gi], p1 = grp[gi + 1];
        const int r0 = (task % d_chunks) * 32, r1 = std::min(n, r0 + 32);
        const uint8_t * db = L.down + L.index(pairs[p0].e) * L.down_bytes;
        for (int r = r0; r < r1; ++r)
            for (int p = p0; p < p1; ++p) td->vec_dot(fa, &y_[(size_t) p * n + r], 0, db + r * d_row, 0, qh_.data() + p * qh_row, 0, 1);
    });
    lap(4);
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
    lap(5);
}

} // namespace hyper
