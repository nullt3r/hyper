#include "tp_engine.h"
#include "dense.h"
#include "sampling.h"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace hyper {

namespace {
const char * envp(const std::string & prefix, const char * name, std::string & buf) {
    buf = prefix + "_" + name;
    return getenv(buf.c_str());
}
size_t kv_rows(int div, int n) { return div == 1 ? (size_t) n : (size_t) (n / div + 1); }
} // namespace

TpEngine::TpEngine(TpOptions opt, TpModel m) : opt_(std::move(opt)), m_(std::move(m)) {}

void TpEngine::init() {
    std::string b;
    const std::string & env = m_.env;
    debug_ = getenv("HYPER4_DEBUG") != nullptr;
    nocpu_ = getenv("HYPER4_NOCPU") != nullptr;
    stream_min_ = m_.stream_min;
    up_parts_ = m_.up_parts;
    mtp_pmin_ = m_.mtp_pmin;
    mtp_cpu_max_ = m_.mtp_cpu_max;
    if (const char * v = envp(env, "STREAM_MIN", b)) stream_min_ = atoi(v);
    if (const char * v = envp(env, "UP_PARTS", b)) up_parts_ = std::max(0, atoi(v));
    if (const char * v = envp(env, "MC", b)) mc_max_ = std::max(1, std::min(8, atoi(v)));
    mc_alloc_ = std::max(mc_max_, envp(env, "MC_ALLOC", b) ? atoi(envp(env, "MC_ALLOC", b)) : 0);
    ar_sync_ = envp(env, "AR_SYNC", b) && atoi(envp(env, "AR_SYNC", b)) != 0;
    if (const char * v = envp(env, "MTP_PMIN", b)) mtp_pmin_ = atof(v);
    if (const char * v = envp(env, "MTP_CPU_MAX", b)) mtp_cpu_max_ = atof(v);
    mtp_full_ = envp(env, "MTP_FULL", b) != nullptr;
    if (const char * v = envp(env, "PRED", b)) pred_mode_ = atoi(v);
    pred_early_ = envp(env, "PRED_EARLY", b) && atoi(envp(env, "PRED_EARLY", b)) != 0;
    if (const char * v = envp(env, "PREDSTAT", b)) pred_k_ = std::max(1, std::min(MOE_MAX_USED, atoi(v)));
    if (const char * rt = envp(env, "ROUTE_TRACE", b)) {
        route_trace_ = fopen(rt, "ab");
        if (!route_trace_) throw std::runtime_error(m_.tag + ": cannot open " + rt);
        fprintf(stderr, "%s: routing trace -> %s\n", m_.tag.c_str(), rt);
    }
    elastic_log_ = envp(env, "ELASTIC_LOG", b) != nullptr;
    if (const char * v = envp(env, "KV_STEP", b)) kv_step_ = std::max(64, atoi(v));   // (tests: frequent growth)
    if (m_.max_pos_align > 1 && opt_.max_pos % m_.max_pos_align) opt_.max_pos += m_.max_pos_align - opt_.max_pos % m_.max_pos_align;
    if (m_.n_embd > CPU_ROW || m_.n_expert_used > MOE_MAX_USED) throw std::runtime_error(m_.tag + ": model dimensions unsupported");
    opt_.n_devices = device_count(opt_.n_devices, (env + "_NDEV").c_str(), m_.tag.c_str());
    for (int g = 0; g < opt_.n_devices; ++g) {
        std::unique_ptr<TpDevice> dev = make_device();
        dev->init_base(g, g, false, m_.pstream_prio);   // (plain fp32 cuBLAS, no TF32: routing must not move)
        for (auto & ev : dev->ev_ar) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev->ev_arh, cudaEventDisableTiming));
        for (auto * row : {dev->ev_a, dev->ev_up2, dev->ev_dn}) for (int pp = 0; pp < 2; ++pp) for (int sl = 0; sl < 8; ++sl)
            CUDA_CHECK(cudaEventCreateWithFlags(&row[pp][sl], cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->arstream, cudaStreamNonBlocking));
        devs_.push_back(std::move(dev));
    }
    const int nd = opt_.n_devices, n = m_.n_embd, slots = nl_tot();
    CUDA_CHECK(cudaHostAlloc(&h_embd_, (size_t) mc_max_ * R() * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_cpos_, 8 * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_stage_, (size_t) 2 * mc_alloc_ * nd * R() * n * sizeof(half), cudaHostAllocPortable));
    barrier_ = std::make_unique<Barrier4>(nd);
    CUDA_CHECK(cudaHostAlloc(&h_pos_, 16 * sizeof(int), cudaHostAllocPortable));
    memset(h_pos_, 0, 16 * sizeof(int));
    CUDA_CHECK(cudaHostAlloc(&h_res_, (size_t) nd * MAX_NT * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_topk_, (size_t) nd * MAX_NT * TOPK * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_mres_, (size_t) nd * 4 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_mtopk_, (size_t) nd * TOPK * 2 * sizeof(float), cudaHostAllocPortable));
    const size_t ll = ll_slots(nd, MAX_NT, n);   // (fp32 parts: a packet per element)
    CUDA_CHECK(cudaHostAlloc(&ar_ll_, ll * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_ll_, 0xff, ll * sizeof(uint2));
    // peer access between every pair (a P2P-capable driver): the LL packets go straight into the peers' VRAM and the
    // prefill partials into their receive buffers; <ENV>_P2P=0 keeps the host paths
    p2p_ = nd > 1 && !(envp(env, "P2P", b) && atoi(envp(env, "P2P", b)) == 0);
    for (int g = 0; g < nd && p2p_; ++g)
        for (int p = 0; p < nd && p2p_; ++p) {
            if (p == g) continue;
            int ok = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&ok, devs_[g]->id, devs_[p]->id));
            if (!ok) { p2p_ = false; break; }
            CUDA_CHECK(cudaSetDevice(devs_[g]->id));
            const cudaError_t e = cudaDeviceEnablePeerAccess(devs_[p]->id, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) { (void) cudaGetLastError(); p2p_ = false; }
        }
    if (nd > 1) fprintf(stderr, "%s: allreduce %s\n", m_.tag.c_str(), p2p_ ? "through peer memory (P2P)" : "through host memory (no P2P)");
    for (int g = 0; g < nd; ++g) {
        TpDevice & d = *devs_[g];
        if (p2p_) {
            d.ll_buf = d.alloc<uint2>(ll);
            CUDA_CHECK(cudaMemset(d.ll_buf, 0xff, ll * sizeof(uint2)));
        }
    }
    for (int g = 0; g < nd; ++g) {
        TpDevice & d = *devs_[g];
        if (p2p_) { d.ll.rd = d.ll_buf; d.ll.nw = 0; for (int p = 0; p < nd; ++p) if (p != g) d.ll.wr[d.ll.nw++] = devs_[p]->ll_buf; }
        else { d.ll.rd = ar_ll_; d.ll.nw = 1; d.ll.wr[0] = ar_ll_; }
    }
    CUDA_CHECK(cudaHostAlloc(&h_ids_, (size_t) slots * mc_max_ * R() * m_.n_expert_used * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_pred_, (size_t) std::max(1, m_.n_layer) * MOE_MAX_USED * sizeof(int), cudaHostAllocPortable));
    init_cpu_side(opt_.cpu_threads, n, m_.n_ff_exp, m_.n_expert_used, slots, m_.clamp_exp, pred_mode_ == 2 && pred_early_);
    {   // the experts' placement: routing statistics from the calibration (HYPER4_STATS), adaptive knobs
        MoePlace::Params pp;
        pp.tag = m_.tag; pp.env = m_.env;
        pp.E = m_.n_expert; pp.K = m_.n_expert_used; pp.nl = slots;
        pp.ff = m_.n_ff_exp; pp.n = n; pp.clamp = m_.clamp_exp;
        pp.tp = true; pp.halves = opt_.stream_experts ? m_.halves : 0;
        pp.adapt_budget = 64; pp.adapt_after_prompt = true; pp.fill_unit = m_.fill_unit;
        pp.stream_cpu = m_.stream_cpu; pp.stream_cpu_max = m_.stream_cpu_max;
        std::vector<DevBase *> db;
        for (auto & dp : devs_) db.push_back(dp.get());
        init_place(pp, db, m_.n_layer);
        place_->kv_mapped = [this](int g) { size_t mm = 0; for (auto & k : devs_[g]->kvs) mm += k.b->mapped(); return mm; };
        place_->kv_headroom = [this](int g) {
            size_t h = 0;
            for (auto & k : devs_[g]->kvs) h += (kv_rows(k.div, kv_step_) * k.row_bytes + k.b->granule()) * k.heads;
            return h;
        };
    }
    elastic_ = place_->elastic();
    park_.before_restore = [this](int nn) { ensure_kv(nn + 2 * MAX_NT); };
    load_weights();
}

TpEngine::~TpEngine() {
    launch_stop_.store(true);
    for (auto & l : launchers_) { l->go.fetch_add(1); if (l->th.joinable()) l->th.join(); }
    for (auto & sn : snaps_) snap_pool_.push_back(sn.h);
    park_.release(snap_pool_);
    for (auto & v : snap_pool_) for (float * p : v) cudaFreeHost(p);
    if (route_trace_) fclose(route_trace_);
    release_cpu_side();   // (the CPU side and the host copies first, the device ranges before the devices)
    for (auto & [p, sz] : host_bufs_) munmap(p, sz);
    devs_.clear();
    for (void * p : {(void *) h_embd_, (void *) h_cpos_, (void *) h_pos_, (void *) h_res_, (void *) h_mres_, (void *) h_topk_, (void *) h_mtopk_,
                     (void *) ar_ll_, (void *) h_stage_, (void *) h_ids_, (void *) h_pred_})
        if (p) cudaFreeHost(p);
}

void * TpEngine::host_huge_alloc(size_t bytes) {
    size_t sz = 0;
    void * p = hyper::host_huge_alloc(bytes, sz);
    host_bufs_.push_back({p, sz});
    return p;
}

// ---- loading ----------------------------------------------------------------------------------------------------------

void TpEngine::alloc_common(TpDevice & dev, const CommonSizes & sz) {
    const int nd = opt_.n_devices, n = m_.n_embd, K = m_.n_expert_used, Rr = R();
    dev.big_stride = sz.big_stride;
    dev.x = dev.alloc<float>((size_t) Rr * n);
    dev.res = dev.alloc<float>((size_t) Rr * sz.res_w);
    for (int ci = 1; ci < mc_alloc_; ++ci) dev.res_x.push_back(dev.alloc<float>((size_t) Rr * sz.res_w));
    dev.pos_x = dev.alloc<int>(8);
    dev.xn = dev.alloc<float>((size_t) Rr * sz.xn_w);
    dev.bo = dev.alloc<float>((size_t) Rr * n);
    dev.part = dev.alloc<float>((size_t) Rr * n);
    dev.big0 = dev.alloc<float>((size_t) Rr * sz.big_stride);
    dev.o = dev.alloc<float>((size_t) Rr * sz.o_w);
    dev.attn_part = dev.alloc<float>(sz.attn_part);
    dev.rlog = dev.alloc<float>((size_t) Rr * sz.rlog_w);
    dev.router32 = dev.alloc<float>((size_t) sz.rlog_w * n);
    dev.ticket = dev.alloc<unsigned>(1);
    CUDA_CHECK(cudaMemset(dev.ticket, 0, sizeof(unsigned)));
    dev.ids = dev.alloc<int>((size_t) Rr * K);
    dev.wts = dev.alloc<float>((size_t) Rr * K);
    dev.sg = dev.alloc<float>(Rr);
    if (sz.pred_all ? (pred_k_ || pred_mode_ == 2) : (pred_mode_ && dev.g == 0)) {
        dev.plog = dev.alloc<float>((size_t) MAX_NT * sz.rlog_w);
        dev.pids = dev.alloc<int>((size_t) MAX_NT * sz.pred_k_w);
        dev.pwts = dev.alloc<float>((size_t) MAX_NT * sz.pred_k_w);
        dev.psg = dev.alloc<float>((size_t) MAX_NT * sz.pred_psg_w);
    }
    dev.shgu = dev.alloc<float>((size_t) Rr * 2 * sz.ff_sh);
    dev.shh = dev.alloc<float>((size_t) Rr * sz.ff_sh);
    dev.shpart = dev.alloc<float>((size_t) Rr * n);
    dev.hexp = dev.alloc<float>((size_t) sz.hexp_rows * K * m_.n_ff_exp);
    dev.yexp = dev.alloc<float>((size_t) Rr * K * n);
    dev.logits = dev.alloc<float>((size_t) sz.logit_rows * dev.output.n());
    dev.xh = dev.alloc<half>((size_t) Rr * sz.xh_w);
    if (dense_ggs_elems()) {   // prefill scratch for KQ dense weights (bigger segments go in slices)
        dev.ggs_elems = dense_ggs_elems();
        dev.ggs = dev.alloc<half>(dev.ggs_elems);
    }
    for (int sl = 0; sl < mc_alloc_; ++sl) {   // (a slot per chunk of a multi-chunk pass: their exchanges are in flight together)
        dev.p16s.push_back(dev.alloc<half>((size_t) Rr * n));
        dev.recvs.push_back(dev.alloc<half>((size_t) std::max(1, nd - 1) * Rr * n));
    }
    dev.p16 = dev.p16s[0];
    dev.recv = dev.recvs[0];
    dev.order = dev.alloc<int>((size_t) Rr * K);
    dev.order_n = dev.alloc<int>(2);
    dev.topk = dev.alloc<float>((size_t) MAX_NT * TOPK * 2);
    dev.egrp = dev.alloc<int>((size_t) 3 * m_.n_expert);
    dev.mix16 = dev.alloc<half>((size_t) Rr * n);
    dev.conv_raw = dev.alloc<float>((size_t) Rr * sz.conv_w);
    dev.iscores = dev.alloc<float>((size_t) sz.iscore_rows * (opt_.max_pos / 4 + 4));
    if (sz.ihist) dev.ihist = dev.alloc<unsigned>((size_t) sz.iscore_rows * 65536);
    dev.ilist = dev.alloc<int>((size_t) Rr * sz.ilist_w);
    dev.ilist_n = dev.alloc<int>(Rr);
    dev.h16 = dev.alloc<half>((size_t) Rr * K * m_.n_ff_exp);
    dev.pos = dev.alloc<int>(1);
    dev.counter = dev.alloc<int>(1);
    dev.lstat = dev.alloc<float>(2);
}

void TpEngine::load_weights() {
    const auto t0 = std::chrono::steady_clock::now();
    const int nd = opt_.n_devices, E = m_.n_expert;
    for (auto & dp : devs_) load_device(*dp);
    // experts fill what is left on each GPU (minus a runtime reserve), capped at gpu_expert_frac of every layer
    size_t eb = 0, eb_max = 0;   // bytes of one expert: average over the MoE layers, largest layer
    int nm = 0;
    for (int il = 0; il < nl_tot(); ++il) {
        if (!is_moe(il)) continue;
        const std::string p = "blk." + std::to_string(il) + ".";
        const GGUF & G = expert_source(il);
        const size_t bb = (G.need(p + "ffn_gate_exps.weight").nbytes * 2 + G.need(p + "ffn_down_exps.weight").nbytes) / E;
        eb += bb;
        eb_max = std::max(eb_max, bb);
        ++nm;
    }
    eb /= std::max(nm, 1);
    std::vector<size_t> freeb(nd);
    for (int g = 0; g < nd; ++g) {
        CUDA_CHECK(cudaSetDevice(devs_[g]->id));
        size_t tot = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeb[g], &tot));
        // (elastic caches are not mapped yet: their full-context size is set aside, as the plain allocation would take it)
        const size_t pend = place_->lend_limit(g);
        freeb[g] = freeb[g] > pend ? freeb[g] - pend : 0;
    }
    const std::vector<int> quota = place_->plan_tp(freeb, opt_.vram_reserve_gib, opt_.gpu_expert_frac, eb, eb_max, std::max(nm, 1),
                                                   opt_.stream_experts, m_.stage_factor);
    for (int il = 0; il < nl_tot(); ++il) {
        if (!is_moe(il)) continue;
        // owners by weighted round robin over the devices' quotas (hottest first); the rest is the placement's (device
        // arrays, the compact host copy, the CPU side, stream lists)
        const ExpertTensors xt = expert_tensors(expert_source(il), il, E);
        const std::vector<int> order = expert_order(place_->stats_of(il), E);
        const std::vector<int> owner = assign_owners_rr(order, quota, E);
        place_->load_layer(il, xt, owner);
        if (opt_.stream_experts) place_->size_staging(il);
    }
    if (opt_.stream_experts)
        for (auto & dp : devs_) {
            for (auto & sb : dp->stage) sb = dp->alloc<uint8_t>(std::max<size_t>(dp->stage_bytes, 1));
            fprintf(stderr, "%s: device %d streams prefill experts through 2 x %.0f MiB\n", m_.tag.c_str(), dp->id, dp->stage_bytes / 1048576.0);
        }
    after_load();
    for (auto & dp : devs_) fprintf(stderr, "%s: device %d holds %.2f GiB\n", m_.tag.c_str(), dp->id, dp->used / 1073741824.0);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "%s: weights loaded in %.1f s (%d GPUs); host expert copies %.1f s\n", m_.tag.c_str(), s, nd, place_->t_hostcopy());
}

// ---- elastic attention caches ----------------------------------------------------------------------------------------

// map every cache up to `need` positions (in kv_step_ steps); extra expert slots give the memory back first
void TpEngine::ensure_kv(int need) {
    if (!elastic_ || need <= kv_pos_) return;
    const int target = std::min(opt_.max_pos, (need + kv_step_ - 1) / kv_step_ * kv_step_);
    place_->before_kv_change();
    for (auto & dp : devs_) {
        TpDevice & d = *dp;
        if (d.kvs.empty()) continue;
        size_t add = 0;
        for (auto & k : d.kvs)
            for (int h = 0; h < k.heads; ++h) add += k.b->cost(h * k.pitch, std::min(kv_rows(k.div, target) * k.row_bytes, k.pitch));
        const size_t fr = lend_free(d);
        if (add > fr) place_->evict_extra(d.g, add - fr);
        CUDA_CHECK(cudaSetDevice(d.id));
        for (auto & k : d.kvs)
            for (int h = 0; h < k.heads; ++h) k.b->map(h * k.pitch, std::min(kv_rows(k.div, target) * k.row_bytes, k.pitch), true, d.stream);
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
    }
    if (elastic_log_) {
        fprintf(stderr, "%s: elastic: caches mapped to %d positions; lendable free", m_.tag.c_str(), target);
        for (auto & dp : devs_) if (!dp->kvs.empty()) fprintf(stderr, " %.0f MiB", lend_free(*dp) / 1048576.0);
        fprintf(stderr, "\n");
    }
    kv_pos_ = target;
}

// a conversation shorter than the mapped caches by two steps or more: the caches beyond `keep` go back to the lending pool
void TpEngine::shrink_kv(int keep) {
    if (!elastic_) return;
    const int target = std::min(opt_.max_pos, (keep + kv_step_ - 1) / kv_step_ * kv_step_);
    if (target + 2 * kv_step_ > kv_pos_) return;
    place_->fill_wait(true);
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaDeviceSynchronize()); }
    size_t freed = 0;
    for (auto & dp : devs_)
        for (auto & k : dp->kvs)
            for (int h = 0; h < k.heads; ++h) freed += k.b->unmap(h * k.pitch + kv_rows(k.div, target) * k.row_bytes, (h + 1) * k.pitch);
    if (elastic_log_) fprintf(stderr, "%s: elastic: caches %d -> %d positions (%.0f MiB to lend)\n", m_.tag.c_str(), kv_pos_, target, freed / 1048576.0);
    kv_pos_ = target;
}

// ---- recording helpers -----------------------------------------------------------------------------------------------

// matmul: decode GEMV (split-K for few rows) or, for a prefill chunk, fp16 conversion + tiled tensor-core GEMM
void TpEngine::mm(TpDevice & d, const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni) const {
    const DenseRun dr{d.stream, d.blas, d.xh, d.ggs, d.ggs_elems};
    dense_mm(dr, W, x, xs, y, ys, rows, ni);
}

// fp32 weights (routers, alpha/beta, injections): warp GEMV per token, or cuBLAS sgemm for a prefill chunk
void TpEngine::f32mm(TpDevice & d, const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nr) const {
    if (nr <= MAX_NT) { gemv_f32(W, rows, k, x, xs, y, ys, nr, d.stream); return; }
    const float one = 1.0f, zero = 0.0f;
    if (cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, nr, k, &one, W, k, x, xs, &zero, y, ys) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSgemm failed");
}

// router logits: fp32, or bf16 (decode: the same products and sums; prefill: widened to fp32 for the same sgemm)
void TpEngine::router_mm(TpDevice & d, const float * router, const uint16_t * router16, int E, const float * x, float * y, int nr) const {
    const int k = m_.n_embd;
    if (!router16) f32mm(d, router, E, k, x, k, y, E, nr);
    else if (nr <= MAX_NT) gemv_bf16w(router16, E, k, x, k, y, E, nr, d.stream);
    else {
        bf16_to_f32(router16, (size_t) E * k, d.router32, d.stream);
        f32mm(d, d.router32, E, k, x, k, y, E, nr);
    }
}

// debug (HYPER4_DEBUG, direct recording): sync after a stage, report errors and non-finite values; HYPER4_SUMS: every stage's sums
void TpEngine::dbg(TpDevice & d, const char * what, int il, const float * buf, size_t cnt, int sum_layers, int rows) const {
    if (!debug_) return;
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    const cudaError_t err = cudaGetLastError();
    std::vector<float> h(cnt);
    CUDA_CHECK(cudaMemcpy(h.data(), buf, cnt * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0; double mx = 0, sum = 0, sq = 0;
    for (float v : h) { if (!std::isfinite(v)) ++bad; else { mx = std::max(mx, (double) std::fabs(v)); sum += v; sq += (double) v * v; } }
    if (bad || err != cudaSuccess || (il < sum_layers && !getenv("HYPER4_SUMS")))
        fprintf(stderr, "dbg L%d %-12s err=%s nonfinite=%d max|x|=%.3g\n", il, what, cudaGetErrorString(err), bad, mx);
    if (getenv("HYPER4_SUMS")) {
        fprintf(stderr, "SUMS %d %s %.17g %.17g\n", il, what, sum, sq);
        if (rows > 1 && cnt % rows == 0) {   // the first row alone (a verification row vs a plain decode step)
            double s0 = 0, q0 = 0;
            for (size_t i = 0; i < cnt / rows; ++i) { s0 += h[i]; q0 += (double) h[i] * h[i]; }
            fprintf(stderr, "SUMS0 %d %s %.17g %.17g\n", il, what, s0, q0);
        }
    }
    if (bad || err != cudaSuccess) throw std::runtime_error("debug stop");
}

// x += sum over the GPUs of d.part (default: d.bo = the sum). Decode: the LL16 kernel; a prefill chunk: copy engines, own
// fp16 part -> pinned staging -> peers (ordered by events; every device's recording thread meets the others at the barrier
// between its upload and its downloads)
void TpEngine::allreduce(TpDevice & d, int nt, int & call, int & dcall, bool bulk, float * x) {
    cudaStream_t s = d.stream;
    const int nd = opt_.n_devices, n = m_.n_embd;
    if (!x) { x = d.bo; CUDA_CHECK(cudaMemsetAsync(d.bo, 0, (size_t) nt * n * sizeof(float), s)); }
    if (!bulk) { allreduce_add_ll16(x, d.part, d.ll, d.g, nd, nt * n, d.counter, call++, s, nullptr); return; }
    const int par = dcall++ & 1;
    ar_begin(d, nt, 0, par);
    barrier_->wait();
    ar_end(d, nt, 0, par, x);
}

void TpEngine::ar_begin(TpDevice & d, int nt, int slot, int par) {
    cudaStream_t s = d.stream, cs = d.arstream;
    const int nd = opt_.n_devices, n = m_.n_embd;
    const size_t N = (size_t) nt * n;
    auto stage = [&](int g) { return h_stage_ + (((size_t) par * mc_alloc_ + slot) * nd + g) * R() * n; };
    half * p16 = d.p16s[slot];
    // (the slot's previous partial -- the other phase's -- must have left for the host before it is overwritten)
    CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_up2[par ^ 1][slot], 0));
    to_half(d.part, n, nullptr, n, 0.0f, p16, nt, s);
    CUDA_CHECK(cudaEventRecord(d.ev_a[par][slot], s));
    CUDA_CHECK(cudaStreamWaitEvent(cs, d.ev_a[par][slot], 0));
    // the staging slot is free once every peer took the previous use of it (their downloads were enqueued before the last barrier)
    for (int p = 0; p < nd; ++p) if (p != d.g) CUDA_CHECK(cudaStreamWaitEvent(cs, devs_[p]->ev_dn[par][slot], 0));
    if (p2p_) {   // straight into every peer's receive slot (its index of this device among its peers)
        for (int p = 0; p < nd; ++p) {
            if (p == d.g) continue;
            const int j = d.g < p ? d.g : d.g - 1;
            CUDA_CHECK(cudaMemcpyPeerAsync(devs_[p]->recvs[slot] + (size_t) j * R() * n, devs_[p]->id, p16, d.id, N * sizeof(half), cs));
        }
    } else CUDA_CHECK(cudaMemcpyAsync(stage(d.g), p16, N * sizeof(half), cudaMemcpyDeviceToHost, cs));
    CUDA_CHECK(cudaEventRecord(d.ev_up2[par][slot], cs));
}

void TpEngine::ar_fetch(TpDevice & d, int nt, int slot, int par) {
    cudaStream_t cs = d.arstream;
    const int nd = opt_.n_devices, n = m_.n_embd;
    const size_t N = (size_t) nt * n;
    auto stage = [&](int g) { return h_stage_ + (((size_t) par * mc_alloc_ + slot) * nd + g) * R() * n; };
    half * recv = d.recvs[slot];
    if (!p2p_) {
        int j = 0;
        for (int p = 0; p < nd; ++p) {
            if (p == d.g) continue;
            CUDA_CHECK(cudaStreamWaitEvent(cs, devs_[p]->ev_up2[par][slot], 0));
            CUDA_CHECK(cudaMemcpyAsync(recv + (size_t) j * R() * n, stage(p), N * sizeof(half), cudaMemcpyHostToDevice, cs));
            ++j;
        }
        CUDA_CHECK(cudaEventRecord(d.ev_dn[par][slot], cs));   // (the host staging slot may be reused)
    }
    if (!d.up_pend.empty()) {   // the next expert upload part, behind these downloads (P2P: behind this device's peer writes)
        CUDA_CHECK(cudaStreamWaitEvent(d.cstream, p2p_ ? d.ev_up2[par][slot] : d.ev_dn[par][slot], 0));
        const auto q = d.up_pend.front();
        d.up_pend.pop_front();
        place_->upload_stage(d.g, q[0], q[1], q[2], up_parts_);
    }
}

void TpEngine::ar_sum(TpDevice & d, int nt, int slot, int par, float * x) {
    cudaStream_t s = d.stream;
    const int nd = opt_.n_devices, n = m_.n_embd;
    const size_t N = (size_t) nt * n;
    if (!x) { x = d.bo; CUDA_CHECK(cudaMemsetAsync(d.bo, 0, N * sizeof(float), s)); }
    if (p2p_) {
        for (int p = 0; p < nd; ++p) if (p != d.g) CUDA_CHECK(cudaStreamWaitEvent(s, devs_[p]->ev_up2[par][slot], 0));   // their writes landed
        add_parts(x, d.p16s[slot], d.recvs[slot], (size_t) R() * n, nd - 1, (int) N, s);
        CUDA_CHECK(cudaEventRecord(d.ev_dn[par][slot], s));   // (the receive slot may be overwritten by the peers again)
        return;
    }
    CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_dn[par][slot], 0));
    add_parts(x, d.p16s[slot], d.recvs[slot], (size_t) R() * n, nd - 1, (int) N, s);
}

// the streamed experts at the start of a recording over layers [il0, il1): one list per device: layer il's share goes
// through staging buffer il & 1 (two layers in flight); two halves: both halves of the first streamed layer
void TpEngine::stream_begin(TpDevice & d, int il0, int il1) {
    d.up_pend.clear();
    if (m_.halves == 2) {
        int first = il0;
        while (first < il1 && !place_->dev(d.g, first).owner_bulk) ++first;
        if (first < il1) { place_->upload_stage(d.g, first, 0); place_->upload_stage(d.g, first, 1); }
    } else {
        int k = 0;
        for (int il = il0; il < il1 && k < 2; ++il)
            if (place_->dev(d.g, il).owner_bulk) { place_->upload_stage(d.g, il, il & 1); ++k; }
    }
}

// layer il's share (half sb) is done: the buffer is free again, the next layer's share goes into it (in up_parts_ parts, each
// behind an allreduce's downloads; 0: at once)
void TpEngine::stream_next(TpDevice & d, int il, int sb, int il1) {
    CUDA_CHECK(cudaEventRecord(d.ev_free[sb], d.stream));
    int nx = il + 1;
    if (m_.halves != 2) ++nx;   // (one list per device: the layer after the one in the other buffer)
    while (nx < il1 && !place_->dev(d.g, nx).owner_bulk) ++nx;
    if (nx >= il1) return;
    if (up_parts_ > 0) for (int pp = 0; pp < up_parts_; ++pp) d.up_pend.push_back({nx, sb, pp});
    else place_->upload_stage(d.g, nx, sb);
}

// ---- forwards --------------------------------------------------------------------------------------------------------

void TpEngine::build_graphs() {
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        TpDevice & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        auto capture = [&](const std::function<void()> & rec, unsigned flags) {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
            rec();
            CUDA_CHECK(cudaStreamEndCapture(d.stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, flags));
            CUDA_CHECK(cudaGraphDestroy(graph));
            return exec;
        };
        for (int nt = 1; nt <= MAX_NT; ++nt) d.g_main[nt] = capture([&] { record_main(gi, nt, 0); }, m_.pstream_prio ? cudaGraphInstantiateFlagUseNodePriority : 0);
        if (mtp_g_) {   // MTP: update over the kept verification rows (1..MAX_NT), chained (1 row)
            for (int nt = 1; nt <= MAX_NT; ++nt) d.g_mtp[nt] = capture([&] { record_main(gi, nt, 1); }, 0);
            d.g_chain = capture([&] { record_main(gi, 1, 2); }, 0);
        }
        if (mtp_g_ || (m_.lookup_spec && opt_.n_draft > 0))
            for (int keep = 1; keep < MAX_NT; ++keep) d.g_restore[keep] = capture([&] { record_restore(gi, keep); }, 0);
    }
    graphs_ready_ = true;
}

void TpEngine::launch_all(int nt, int kind) {
    std::string b;
    static const bool prof = envp(m_.env, "LAUNCHPROF", b) != nullptr;
    // (measured 2026-10-25: a decode graph launch costs 15-21 us of host time -- the ~1.3 ms seen under nsys was its
    // --cuda-graph-trace=node; parallel launches gained nothing (31.6 vs 31.5 t/s) and their threads compete with the CPU
    // experts' for cores: off unless <ENV>_PAR_LAUNCH=1)
    static const bool par = envp(m_.env, "PAR_LAUNCH", b) && atoi(envp(m_.env, "PAR_LAUNCH", b)) != 0;
    const int nd = (int) devs_.size();
    auto exec_of = [&](TpDevice & dp) { return kind == 3 ? dp.g_restore[nt] : kind == 2 ? dp.g_chain : kind == 1 ? dp.g_mtp[nt] : dp.g_main[nt]; };
    const bool direct = (debug_ && kind < 3) || (kind == 1 && mtp_direct());   // (no graph: debugging; MTP rows into the cache only)
    if (direct || !par || nd < 2) {
        for (int gi = 0; gi < nd; ++gi) {
            auto & dp = devs_[gi];
            CUDA_CHECK(cudaSetDevice(dp->id));
            if (direct) { record_main(gi, nt, kind); continue; }
            cudaGraphExec_t ex = exec_of(*dp);
            if (!ex) throw std::runtime_error("run: graph missing (kind " + std::to_string(kind) + ", nt " + std::to_string(nt) + ")");
            const auto t0 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaGraphLaunch(ex, dp->stream));
            if (prof) t_launch_[std::min(gi, 7)] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        }
    } else {
        if ((int) launchers_.size() != nd - 1) {   // threads for devices 1..nd-1 (spin while decoding, sleep when idle)
            for (int gi = 1; gi < nd; ++gi) {
                auto l = std::make_unique<Launcher>();
                Launcher * lp = l.get();
                const int dev_id = devs_[gi]->id;
                cudaStream_t st = devs_[gi]->stream;
                l->th = std::thread([this, lp, dev_id, st, gi] {
                    cudaSetDevice(dev_id);
                    unsigned seen = 0;
                    while (true) {
                        int spins = 0;
                        while (lp->go.load(std::memory_order_acquire) == seen) {
                            if (launch_stop_.load()) return;
                            if (++spins < 20000) __builtin_ia32_pause(); else std::this_thread::sleep_for(std::chrono::microseconds(100));
                        }
                        seen = lp->go.load(std::memory_order_acquire);
                        if (launch_stop_.load()) return;
                        const auto t0 = std::chrono::steady_clock::now();
                        const cudaError_t e = cudaGraphLaunch(lp->ge, st);
                        if (prof) t_launch_[std::min(gi, 7)] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                        lp->done.store(e == cudaSuccess ? seen : ~0u, std::memory_order_release);
                    }
                });
                launchers_.push_back(std::move(l));
            }
        }
        unsigned want[8] = {};
        for (int gi = 1; gi < nd; ++gi) {
            Launcher & l = *launchers_[gi - 1];
            l.ge = exec_of(*devs_[gi]);
            want[gi] = l.go.fetch_add(1, std::memory_order_acq_rel) + 1;
        }
        auto & d0 = devs_[0];
        CUDA_CHECK(cudaSetDevice(d0->id));
        const auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGraphLaunch(exec_of(*d0), d0->stream));
        if (prof) t_launch_[0] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        for (int gi = 1; gi < nd; ++gi) {   // (the launches queued: the wait for completion follows in run())
            Launcher & l = *launchers_[gi - 1];
            unsigned dd;
            while ((dd = l.done.load(std::memory_order_acquire)) != want[gi]) {
                if (dd == ~0u) throw std::runtime_error(m_.tag + ": graph launch failed on device " + std::to_string(devs_[gi]->id));
                __builtin_ia32_pause();
            }
        }
    }
    if (prof && ++n_launch_ % 500 == 0) {
        fprintf(stderr, "%s: graph launch host time per forward:", m_.tag.c_str());
        for (int gi = 0; gi < nd; ++gi) { fprintf(stderr, " dev%d %.0f us", gi, t_launch_[std::min(gi, 7)] / 500); t_launch_[std::min(gi, 7)] = 0; }
        fprintf(stderr, " (%s)\n", par ? "parallel" : "serial");
    }
}

// launch one forward of `kind` over nt rows on every device and wait (hang diagnostics after 10 s)
void TpEngine::run(int kind, int nt) {
    const bool bulk = nt > MAX_NT;
    if (!graphs_ready_ && !debug_) build_graphs();
    ++fwd_counter_;
    if (kind == 3) --fwd_counter_;   // restore graphs do not count
    // the CPU's experts: expected unless the chunk streams them to the GPUs (MTP: the FFN runs for the last row only, in
    // decode mode, and not at all for cache-only rows; <ENV>_MTP_FULL: as main)
    const bool mtp_full = kind == 0 || mtp_full_;
    const bool streamed = bulk && opt_.stream_experts && nt >= stream_min_ && mtp_full;
    if (kind <= 2 && (!streamed || (place_->stream_cpu_built() > 0.0 && mc_n_ == 1)) && !(kind == 1 && mtp_direct() && !mtp_full_)) {
        std::vector<int> slots;
        if (kind) { if (!streamed || place_->dev(0, m_.n_layer).cpu_keep) slots.push_back(m_.n_layer); }
        else for (int i = 0; i < m_.n_layer; ++i) if (is_moe(i) && (!streamed || place_->dev(0, i).cpu_keep)) slots.push_back(i);
        if (!slots.empty()) cpu_->expect(fwd_counter_, slots, bulk && mtp_full);
    }
    if (bulk) {   // record straight into the streams, one host thread per device (barriers inside the allreduces)
        std::vector<std::thread> th;
        std::vector<std::string> err(devs_.size());
        for (int gi = 0; gi < (int) devs_.size(); ++gi)
            th.emplace_back([&, gi] {
                try { CUDA_CHECK(cudaSetDevice(devs_[gi]->id)); record_main(gi, nt, kind); }
                catch (const std::exception & ex) { err[gi] = ex.what(); barrier_->abort(); }
            });
        for (auto & t : th) t.join();
        std::string first;   // (the failing device's message, not the others' "barrier" ones)
        for (auto & mm : err) if (!mm.empty() && (first.empty() || first.rfind("barrier:", 0) == 0)) first = mm;
        if (!first.empty()) { barrier_->reset(); throw std::runtime_error(first); }
    } else {
        launch_all(nt, kind);
    }
    auto t_wait = std::chrono::steady_clock::now();
    bool reported = false;
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        for (;;) {
            const cudaError_t q = cudaStreamQuery(dp->stream);
            if (q == cudaSuccess) break;
            if (q != cudaErrorNotReady) CUDA_CHECK(q);
            if (!reported && std::chrono::steady_clock::now() - t_wait > std::chrono::seconds(10)) {
                reported = true;
                fprintf(stderr, "%s: forward %u (kind %d) stuck on device %d; cpu: %s\n", m_.tag.c_str(), fwd_counter_, kind, dp->id, cpu_->state().c_str());
            }
            std::this_thread::yield();
        }
    }
}

std::vector<int> TpEngine::forward(const int * tokens, int nt, int pos) {
    if (nt < 1 || nt > R()) throw std::runtime_error("forward: bad token count");
    if (pos + nt > opt_.max_pos) throw std::runtime_error("forward: position exceeds max_pos");
    ensure_kv(pos + nt + 2 * MAX_NT);   // (MTP drafts run up to MAX_NT positions further)
    const bool bulk = nt > MAX_NT;
    std::string b;
    static const bool hprof = envp(m_.env, "HOSTPROF", b) != nullptr;   // host time per decode forward (GPUs idle meanwhile)
    static double h_embed = 0, h_run = 0, h_gap = 0, h_tok = 0, h_fill = 0, h_reb = 0; static long h_n = 0;
    static auto h_last = std::chrono::steady_clock::now();
    const auto th0 = std::chrono::steady_clock::now();
    embed(tokens, nt, pos);
    h_pos_[0] = pos;
    const auto thf = std::chrono::steady_clock::now();
    // a conversation much shorter than the mapped caches gives the rest to experts once it is under way (side requests of
    // an agent -- a title, a summary -- end before that and leave the main conversation's room alone)
    if (elastic_ && !bulk && ++req_steps_ == 32) shrink_kv(pos + nt + 2 * MAX_NT);
    const auto thr = std::chrono::steady_clock::now();
    if (bulk) place_->set_stream_cpu_for(nt, stream_min_);   // (a single streamed chunk may leave a share of the CPU experts on the CPU)
    place_->before_forward(bulk);   // (decode: finished fills take over, the placement follows the routing; bulk: stream lists)
    const auto th1 = std::chrono::steady_clock::now();
    run(0, nt);
    if (bulk) score_prompt(&nt, 1);
    if (!bulk && mtp_cpu_max_ < 1e29) {   // the MTP gate's CPU load: row 0's experts held by the CPU, per MoE layer
        const int K = m_.n_expert_used;
        int ncpu = 0, nl = 0;
        for (int il = 0; il < m_.n_layer; ++il) {
            if (!is_moe(il)) continue;
            ++nl;
            for (int j = 0; j < K; ++j) {
                const int e = cpu_rec_[il].ids[0][j];
                if (e >= 0 && e < m_.n_expert && place_->host(il).owner[e] == CPU_OWNER) ++ncpu;
            }
        }
        if (nl) cpu_ema_ = 0.9 * cpu_ema_ + 0.1 * (double) ncpu / nl;
    }
    if (route_trace_ && !bulk) trace_routing(2, nt, [&](int il, int i) { return cpu_rec_[il].ids[i / m_.n_expert_used][i % m_.n_expert_used]; });
    if (pred_k_ && !bulk) {   // (measurement) how many of the CPU experts layer l+1 routes to were predicted
        const int K = m_.n_expert_used;
        for (int il = 1; il < m_.n_layer; ++il) {
            if (!is_moe(il) || !is_moe(il - 1)) continue;
            const int * pr = h_pred_ + (size_t) il * MOE_MAX_USED;
            for (int j = 0; j < pred_k_; ++j) pred_cpu_ += place_->host(il).owner[pr[j]] == CPU_OWNER;
            for (int j = 0; j < K; ++j) {
                const int e = cpu_rec_[il].ids[0][j];
                if (e < 0 || place_->host(il).owner[e] != CPU_OWNER) continue;
                ++pred_tot_;
                for (int q = 0; q < pred_k_; ++q) if (pr[q] == e) { ++pred_hit_; break; }
            }
        }
        if (++pred_n_ % 200 == 0) {
            fprintf(stderr, "%s: next-layer prediction (top %d): %.1f %% of %llu CPU experts predicted, %.2f predicted CPU experts per layer\n",
                    m_.tag.c_str(), pred_k_, 100.0 * pred_hit_ / std::max<uint64_t>(1, pred_tot_), (unsigned long long) pred_tot_,
                    (double) pred_cpu_ / (200.0 * std::max(1, m_.n_layer - m_.n_dense)));
            pred_hit_ = pred_tot_ = pred_cpu_ = 0;
        }
    }
    if (hprof && !bulk) {
        const auto th2 = std::chrono::steady_clock::now();
        h_gap += std::chrono::duration<double>(th0 - h_last).count();
        h_embed += std::chrono::duration<double>(th1 - th0).count();
        h_tok += std::chrono::duration<double>(thf - th0).count();
        h_fill += std::chrono::duration<double>(thr - thf).count();
        h_reb += std::chrono::duration<double>(th1 - thr).count();
        h_run += std::chrono::duration<double>(th2 - th1).count();
        h_last = th2;
        if (++h_n % 200 == 0)
            fprintf(stderr, "%s: per decode forward: before launch %.1f us (embed %.1f, elastic %.1f, rebalance %.1f), run %.1f us, "
                    "caller between forwards %.1f us\n", m_.tag.c_str(), h_embed / h_n * 1e6, h_tok / h_n * 1e6, h_fill / h_n * 1e6, h_reb / h_n * 1e6,
                    h_run / h_n * 1e6, h_gap / h_n * 1e6);
    }
    last_nt_ = nt;
    std::vector<int> out(nt, -1);
    for (int t = bulk ? nt - 1 : 0; t < nt; ++t) {
        const int row = bulk ? 0 : t;
        float best = -INFINITY; int bi = -1;
        for (size_t g = 0; g < devs_.size(); ++g) {
            const float v = h_res_[(g * MAX_NT + row) * 2];
            const int idx = ((const int *) h_res_)[(g * MAX_NT + row) * 2 + 1];
            if (v > best) { best = v; bi = idx; }
        }
        out[t] = bi;
    }
    return out;
}

// the last bulk pass's prompt routing (GPU 0's ids per chunk) into the placement scores, weighted below decode routing
void TpEngine::score_prompt(const int * lens, int nck) {
    if (!place_->adapt()) return;
    const int K = m_.n_expert_used;
    for (int il = 0; il < m_.n_layer; ++il) {
        if (!is_moe(il)) continue;
        for (int ci = 0; ci < nck; ++ci) place_->score_prompt(il, h_ids_ + ((size_t) il * mc_max_ + ci) * R() * K, lens[ci] * K);
    }
    for (int ci = 0; ci < nck; ++ci) trace_routing(1, lens[ci], [&](int il, int i) { return h_ids_[((size_t) il * mc_max_ + ci) * R() * K + i]; });
    place_->prompt_done();
}

bool TpEngine::multi_ok(const int * lens, int nck) const {
    bool ok = nck >= 1 && nck <= mc_max_ && opt_.stream_experts;
    for (int ci = 0; ok && ci < nck; ++ci) ok = lens[ci] > MAX_NT && lens[ci] <= R() && lens[ci] >= stream_min_;
    return ok;
}

int TpEngine::forward_multi(const int * tokens, const int * lens, int nck, int pos) {
    int total = 0;
    for (int ci = 0; ci < nck; ++ci) total += lens[ci];
    if (!multi_ok(lens, nck) || nck == 1) {   // one chunk at a time
        int r = -1;
        for (int ci = 0, off = 0; ci < nck; off += lens[ci++]) r = forward(tokens + off, lens[ci], pos + off).back();
        mc_n_ = 1;
        return r;
    }
    if (pos + total > opt_.max_pos) throw std::runtime_error("forward_multi: position exceeds max_pos");
    ensure_kv(pos + total + 2 * MAX_NT);
    for (int ci = 0, off = 0; ci < nck; off += lens[ci++]) {
        embed(tokens + off, lens[ci], pos + off, ci);
        mc_nt_[ci] = lens[ci]; mc_pos_[ci] = pos + off; h_cpos_[ci] = pos + off; h_pos_[ci] = pos + off;
    }
    h_pos_[0] = pos;
    place_->set_stream_cpu(0.0);   // (several chunks in one pass: everything streams, the CPU bulk record holds one chunk)
    place_->before_forward(true);
    mc_n_ = nck;
    try { run(0, lens[0]); } catch (...) { mc_n_ = 1; throw; }
    score_prompt(lens, nck);
    mc_n_ = 1;   // (later bulk forwards are single-chunk unless set again; mc_nt_/mc_pos_ stay for the MTP pass over the chunks)
    mc_last_ = nck;
    last_nt_ = lens[nck - 1];
    float best = -INFINITY; int bi = -1;   // the last chunk's last row
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float v = h_res_[g * MAX_NT * 2];
        const int idx = ((const int *) h_res_)[g * MAX_NT * 2 + 1];
        if (v > best) { best = v; bi = idx; }
    }
    return bi;
}

int TpEngine::prefill(const int * tokens, int n, int pos) {
    int next = -1;
    static const int chunk = getenv("HYPER4_CHUNK") ? std::max(1, std::min(R(), atoi(getenv("HYPER4_CHUNK")))) : R();
    for (int c0 = 0; c0 < n;) {
        int lens[8], nk = 0, q = c0;   // up to mc_max_ full chunks per layer-major pass
        while (nk < mc_max_ && q < n) { lens[nk] = std::min(chunk, n - q); q += lens[nk++]; }
        if (nk > 1 && !multi_ok(lens, nk)) nk = 1;
        if (nk > 1) next = forward_multi(tokens + c0, lens, nk, pos + c0);
        else next = forward(tokens + c0, lens[0], pos + c0)[lens[0] - 1];
        for (int ci = 0; ci < nk; ++ci) c0 += lens[ci];
    }
    return next;
}

void TpEngine::get_logits(int t, std::vector<float> & out) {
    out.resize(m_.n_vocab);
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaMemcpy(out.data() + dp->vocab_off, dp->logits + (size_t) t * dp->output.n(), (size_t) dp->output.n() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

// ---- MTP, sampling, snapshots -----------------------------------------------------------------------------------------

int TpEngine::mtp_result() {
    float best = -INFINITY; int bi = -1;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float v = h_mres_[g * 4];
        const int idx = ((const int *) h_mres_)[g * 4 + 1];
        if (v > best) { best = v; bi = idx; }
    }
    float M = -INFINITY;   // softmax normalizer relative to the best logit: [2] = slice max, [3] = sum exp(x - slice max)
    for (size_t g = 0; g < devs_.size(); ++g) M = std::max(M, h_mres_[g * 4 + 2]);
    double z = 0;
    for (size_t g = 0; g < devs_.size(); ++g) z += h_mres_[g * 4 + 3] * std::exp((double) h_mres_[g * 4 + 2] - M);
    mtp_p_ = z > 0 ? 1.0 / z : 0.0;
    if (samp_invT_ > 0) {   // sampling: a draft drawn like the main model's sample (truncation, noise) from the head's candidates
        std::vector<std::pair<float, int>> cand;
        for (size_t g = 0; g < devs_.size(); ++g)
            for (int i = 0; i < TOPK; ++i) {
                const float * q = h_mtopk_ + ((size_t) g * TOPK + i) * 2;
                const int idx = ((const int *) q)[1];
                if (idx >= 0) cand.push_back({q[0], idx});
            }
        if (!cand.empty()) return draft_pick(cand, samp_, samp_seed_, mtp_pos_);
    }
    return bi;
}

// candidates of row t from every device's top-K, then temperature / top-k / min-p / top-p
int TpEngine::sample_row(int t, const SamplingParams & sp, int64_t pos) {
    std::vector<std::pair<float, int>> cand;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float * p = h_topk_ + ((size_t) g * MAX_NT + t) * TOPK * 2;
        for (int i = 0; i < TOPK; ++i) { const int idx = ((const int *) p)[2 * i + 1]; if (idx >= 0) cand.push_back({p[2 * i], idx}); }
    }
    // probabilities over the whole row when the candidates are not enough (top_k off): normalizer per vocab slice on the GPUs
    auto stats = [&](double invT, float & M, double & Z) {
        std::vector<float> ms(2 * devs_.size());
        for (size_t g = 0; g < devs_.size(); ++g) {
            TpDevice & d = *devs_[g];
            CUDA_CHECK(cudaSetDevice(d.id));
            max_sumexp(d.logits + (size_t) t * d.output.n(), d.output.n(), d.lstat, d.stream, (float) invT);
            CUDA_CHECK(cudaMemcpyAsync(&ms[2 * g], d.lstat, 2 * sizeof(float), cudaMemcpyDeviceToHost, d.stream));
        }
        M = -INFINITY;
        for (size_t g = 0; g < devs_.size(); ++g) {
            CUDA_CHECK(cudaSetDevice(devs_[g]->id));
            CUDA_CHECK(cudaStreamSynchronize(devs_[g]->stream));
            M = std::max(M, ms[2 * g]);
        }
        Z = 0;
        for (size_t g = 0; g < devs_.size(); ++g) Z += ms[2 * g + 1] * std::exp(((double) ms[2 * g] - M) * invT);
    };
    return sample_candidates(cand, TOPK, sp, samp_seed_, pos, stats, [&](std::vector<float> & lg) { get_logits(t, lg); });
}

void TpEngine::take_snapshot(int pos) {
    for (auto & s : snaps_) if (s.pos == pos) return;
    if ((int) snaps_.size() >= opt_.max_snapshots) {
        auto victim = snaps_.end();
        for (auto it = snaps_.begin(); it != snaps_.end(); ++it)
            if (it->pos % 4096 && (victim == snaps_.end() || it->pos < victim->pos)) victim = it;
        if (victim == snaps_.end())
            victim = std::min_element(snaps_.begin(), snaps_.end(), [](const TpSnap & a, const TpSnap & b) { return a.pos < b.pos; });
        snap_pool_.push_back(victim->h);
        snaps_.erase(victim);
    }
    TpSnap sn;
    sn.pos = pos;
    if (!snap_pool_.empty()) { sn.h = snap_pool_.back(); snap_pool_.pop_back(); }
    else {
        for (size_t gi = 0; gi < devs_.size(); ++gi) {
            float * p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, snap_floats((int) gi) * sizeof(float), cudaHostAllocPortable));
            sn.h.push_back(p);
        }
    }
    snap_copy(sn.h, true);
    snaps_.push_back(sn);
}

// ---- generation -------------------------------------------------------------------------------------------------------

std::vector<int> TpEngine::generate(const std::vector<int> & prompt, int n_gen, bool spec_req, GenStats * stats,
                                    const std::function<bool(int)> & on_token, const SamplingParams & sp) {
    const std::vector<int> * kp = prompt_key(prompt);   // (taken first: a request that fails leaves no media for the next)
    if (prompt.empty()) throw std::runtime_error("generate: empty prompt");
    if ((int) prompt.size() + 8 > opt_.max_pos) throw std::runtime_error("generate: prompt longer than the context");
    using clk = std::chrono::steady_clock;
    const bool sampling = sp.temp > 0.0f;
    samp_seed_ = sp.seed ? sp.seed : random_seed64();
    // sampled drafts coupled to the sampling noise (HYPER_GUMBEL_DRAFT=0/1; default: Flash-Next 128 -> 132 t/s at T 0.6; GLM,
    // fixed-placement A/B at T 1: acceptance never lower, one ungated draft 0.79 -> 0.88 accepted)
    static const bool gumbel_draft = getenv("HYPER_GUMBEL_DRAFT") ? atoi(getenv("HYPER_GUMBEL_DRAFT")) != 0 : true;
    samp_invT_ = sampling && gumbel_draft ? 1.0f / sp.temp : 0.0f;   // (MTP drafts coupled to the sampling noise)
    samp_ = sp;
    const int P = (int) prompt.size();
    // prompt cache: KV / indexer entries stay valid for the common prefix L; the recurrent state comes from the latest
    // snapshot at s <= L - 1. The cache compares keys (an image's hash at its positions) when the model gives them
    const std::vector<int> & key = kp ? *kp : prompt;
    int s = 0;
    if (opt_.prompt_cache) {
        // a fill batch started in the previous request's decode may still be mapping VRAM and copying on its own thread:
        // it finishes before parking / resuming allocates, frees and copies pinned memory (no other change: the batch
        // still hands over at its due step)
        place_->fill_join();
        const int L = park_.resolve(key, hist_, snaps_, snap_pool_, park_devs(), m_.tag.c_str());   // (see park.h)
        for (size_t i = 0; i < snaps_.size();)
            if (snaps_[i].pos > L) { snap_pool_.push_back(snaps_[i].h); snaps_.erase(snaps_.begin() + i); } else ++i;
        TpSnap * best = nullptr;
        for (auto & sn : snaps_) if (sn.pos <= std::min(L, P - 1) && (!best || sn.pos > best->pos)) best = &sn;
        if (best) { s = best->pos; snap_copy(best->h, false); }
    }
    if (s == 0) reset();
    req_steps_ = 0;
    ensure_kv(P + 2 * MAX_NT);
    hist_.assign(key.begin(), key.begin() + s);
    prompt_begin(prompt, s);
    std::vector<int> snap_at;
    if (opt_.prompt_cache) {
        std::vector<int> msg;
        for (int q = s + 1; q < P; ++q)
            if (std::find(snap_tokens_.begin(), snap_tokens_.end(), prompt[q]) != snap_tokens_.end()) msg.push_back(q);
        // snapshots at every full-chunk boundary (prefill runs in whole R-token chunks: every MoE chunk pays a fixed cost,
        // one pass over the CPU-resident experts) and at the last message start (edits / regenerations resume there);
        // generation adds its own snapshots, so the end of a previous answer is covered too
        // (multi-chunk prefill: every mc_max_ chunks, so a whole group runs as one layer-major pass)
        for (int q = s + R() * mc_max_; q < P; q += R() * mc_max_) snap_at.push_back(q);
        if (!msg.empty() && msg.back() - s >= 64) snap_at.push_back(msg.back());
        std::sort(snap_at.begin(), snap_at.end());
        snap_at.erase(std::unique(snap_at.begin(), snap_at.end()), snap_at.end());
    }
    GenStats st;
    const int K = std::min(opt_.n_draft, MAX_NT - 1);
    const bool use_mtp = mtp_g_ && K > 0 && spec_req;   // (the MTP block's cache follows the prompt chunk by chunk)
    const bool use_lookup = !use_mtp && m_.lookup_spec && K > 0 && spec_req;
    std::vector<int> drafts(std::max(1, K), -1);
    auto tp = clk::now();
    int next = -1, d0 = -1;
    size_t si = 0;
    for (int c0 = s; c0 < P;) {
        // the stretch up to the next snapshot point in equal chunks of at most R (no short tail chunk inside it), the
        // first mc_max_ of them as one layer-major pass when every one of them streams its experts
        while (si < snap_at.size() && snap_at[si] <= c0) ++si;
        const int seg = (si < snap_at.size() ? snap_at[si] : P) - c0, nch = (seg + R() - 1) / R();
        int lens[8], nk = 0, tot = 0;
        for (int rest = seg, left = nch; rest > 0 && nk < mc_max_; --left) { lens[nk] = (rest + left - 1) / left; tot += lens[nk]; rest -= lens[nk++]; }
        if (!(nk > 1 && multi_ok(lens, nk))) { nk = 1; tot = lens[0]; }
        const int end = c0 + tot, len = lens[nk - 1];
        next = nk > 1 ? forward_multi(&prompt[c0], lens, nk, c0) : forward(&prompt[c0], len, c0)[len - 1];
        if (sampling) next = sample_row(nk == 1 && len <= MAX_NT ? len - 1 : 0, sp, end);
        if (use_mtp) { const int dd = mtp_prompt(prompt, c0, lens, nk, next); if (end == P) d0 = dd; }
        if (si < snap_at.size() && snap_at[si] == end) take_snapshot(end);
        if (prefill_cb_) prefill_cb_(end, P, s);
        c0 = end;
    }
    hist_ = key;
    prompt_end();
    st.t_prefill = std::chrono::duration<double>(clk::now() - tp).count();
    st.prompt_reused = s;
    Emitter em;
    em.n_gen = n_gen; em.on_token = &on_token;
    std::vector<int> & out = em.out;
    bool & stop = em.stop;
    auto emit = [&](int tok) { return em.emit(tok); };
    int p = P;
    auto t0 = clk::now();
    auto since = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    // generated tokens get recurrent-state snapshots too (every 1024 positions and at the end): the next request repeats
    // this answer in its prompt and resumes close to where the re-rendered history first differs
    int snap_mark = p / 1024;
    auto gen_snapshot = [&] {
        if (!opt_.prompt_cache || p / 1024 == snap_mark) return;
        snap_mark = p / 1024;
        take_snapshot(p);
    };
    if (use_mtp) {
        // MTP drafts: the first from the MTP block over (token at p, main hidden of p - 1), the rest chained on its own
        // output; a draft is kept iff the token sampled (or argmax) at its row equals it: exact. Drafts stop once the MTP
        // block's probability of the drafted run drops below mtp_pmin_ (0: the count follows the recent acceptance,
        // kc = floor(avg accepted + 1.5) in [1, K]; <ENV>_FIXED_DRAFT: always K); the CPU-load gate (mtp_cpu_max_) holds
        // them back on CPU-paced steps, where a verification row would cost more than it saves
        std::string b;
        static const bool fixed_k = envp(m_.env, "FIXED_DRAFT", b) != nullptr;
        const double pmin = mtp_pmin_;
        int kc = K;
        double avg_acc = K;
        std::vector<int> in(K + 1), mt(K + 1);
        auto extend = [&](int pos) {   // drafts[0] just made (the token at pos + 1): the chained ones, the count to verify
            double pc = mtp_p_;
            if (pc < pmin) return 0;
            if (cpu_ema_ > mtp_cpu_max_) { st.gated++; return 0; }
            int k = 1;
            for (; k < kc && pos + k < opt_.max_pos; ++k) {   // (the MTP cache holds positions < max_pos)
                drafts[k] = mtp_chain(drafts[k - 1], pos + k, false);
                pc *= mtp_p_;
                if (pc < pmin) break;
            }
            return k;
        };
        // prompt-lookup drafts beside the MTP chain (<ENV>_MTP_LOOKUP=0: off): when the recent tokens repeat an earlier passage
        // (edits, quotes, code), the continuation found there replaces the chain when it is longer -- it costs no MTP pass
        // and a verification window of K rows costs little more than one of three
        static const bool lookup_too = !envp(m_.env, "MTP_LOOKUP", b) || atoi(envp(m_.env, "MTP_LOOKUP", b)) != 0;
        LookupDrafter lk(prompt);
        std::vector<int> lkd(K);
        auto lookup = [&](int nd) {   // (more lookup drafts than the chain gave: take them instead)
            if (!lookup_too || cpu_ema_ > mtp_cpu_max_) return nd;
            const int nl = lk.draft(K, acc_rate_, lkd.data());
            if (nl <= nd) return nd;
            for (int j = 0; j < nl; ++j) drafts[j] = lkd[j];
            return nl;
        };
        int cur = next;   // token at position p, not yet in the main model
        auto ta = clk::now();
        drafts[0] = d0 >= 0 ? d0 : mtp_chain(cur, p, true);
        int nd = extend(p);
        st.t_mtp += since(ta);
        lk.push(cur);
        nd = lookup(nd);
        while (!stop && p + K + 1 < opt_.max_pos) {
            // the token at p is final before its forward: a stop on it leaves the state at p (nothing past the output)
            if (!emit(cur)) break;
            in[0] = cur;
            for (int j = 0; j < nd; ++j) in[j + 1] = drafts[j];
            ta = clk::now();
            std::vector<int> a = forward(in.data(), nd + 1, p);
            st.t_main += since(ta);
            st.steps++;
            st.drafted += nd;
            int mm = 0;
            if (sampling) { while (mm < nd && (a[mm] = sample_row(mm, sp, p + mm + 1)) == drafts[mm]) ++mm; if (mm == nd) a[nd] = sample_row(nd, sp, p + nd + 1); }
            else while (mm < nd && a[mm] == drafts[mm]) ++mm;
            st.accepted += mm;
            if (nd > 0) acc_rate_ = 0.9 * acc_rate_ + 0.1 * ((double) mm / nd);
            int r = 0;   // drafts emitted
            while (r < mm) { lk.push(drafts[r]); if (!emit(drafts[r++])) break; }
            if (stop) {
                // the output ends with drafts[r - 1] at p + r: keep rows 0..r-1 (the state up to its last token, as when a stop
                // comes before a forward) and the MTP cache rows p+1..p+r, so the end-of-answer snapshot can serve the next request
                if (opt_.answer_reuse) {
                    ta = clk::now(); run(3, r); st.t_restore += since(ta);
                    ta = clk::now(); mtp_update(drafts.data(), r, p + 1, false); st.t_mtp += since(ta);
                    p += r;
                }
                break;
            }
            if (mm < nd) { ta = clk::now(); run(3, mm + 1); st.t_restore += since(ta); }   // keep rows 0..mm of the verified block
            cur = a[mm];
            for (int i = 0; i < mm; ++i) mt[i] = drafts[i];
            mt[mm] = cur;
            if (!fixed_k && pmin <= 0) {
                avg_acc = 0.9 * avg_acc + 0.1 * mm;
                kc = std::max(1, std::min(K, (int) (avg_acc + 1.5)));
            }
            p += mm + 1;
            lk.push(cur);
            if (p + K + 1 < opt_.max_pos) {   // (rows p - mm .. p: the kept rows' tokens after them)
                ta = clk::now();
                drafts[0] = mtp_update(mt.data(), mm + 1, p - mm, true);
                nd = extend(p);
                st.t_mtp += since(ta);
                nd = lookup(nd);
            }
            gen_snapshot();
        }
    } else if (!use_lookup) {
        while (emit(next) && p + 1 < opt_.max_pos) {
            next = forward(&next, 1, p++)[0];
            if (sampling) next = sample_row(0, sp, p);
            st.steps++;
            gen_snapshot();
        }
    } else {
        // prompt-lookup speculation (LookupDrafter); a draft is kept iff the token sampled (or argmax) at its row equals it: exact
        LookupDrafter lk(prompt);
        int cur = next;
        std::vector<int> in(K + 1);
        while (!stop && p + K + 1 < opt_.max_pos) {
            if (!emit(cur)) break;   // (before its forward, as above)
            lk.push(cur);
            const int nd = lk.draft(K, acc_rate_, in.data() + 1);
            in[0] = cur;
            auto ta = clk::now();
            std::vector<int> a = forward(in.data(), nd + 1, p);
            st.t_main += since(ta);
            st.steps++;
            st.drafted += nd;
            int mm = 0;
            if (sampling) { while (mm < nd && (a[mm] = sample_row(mm, sp, p + mm + 1)) == in[1 + mm]) ++mm; if (mm == nd) a[nd] = sample_row(nd, sp, p + nd + 1); }
            else while (mm < nd && a[mm] == in[1 + mm]) ++mm;
            st.accepted += mm;
            if (nd > 0) acc_rate_ = 0.9 * acc_rate_ + 0.1 * ((double) mm / nd);
            int r = 0;   // drafts emitted
            while (r < mm) { lk.push(in[1 + r]); if (!emit(in[1 + r++])) break; }
            if (stop) {   // (rows 0..r-1: the state up to the output's last token, as above)
                if (opt_.answer_reuse) { ta = clk::now(); run(3, r); st.t_restore += since(ta); p += r; }
                break;
            }
            if (mm < nd) { ta = clk::now(); run(3, mm + 1); st.t_restore += since(ta); }
            cur = a[mm];
            p += mm + 1;
            gen_snapshot();
        }
    }
    // the state up to the output's last token (a stop token not run; a cut-off answer's last token as the template's
    // end-of-message token would follow it): the next request repeats the answer and resumes there; the history holds that
    // last token too (the MTP cache row at p was made with it)
    if (opt_.prompt_cache && p > P && opt_.answer_reuse) take_snapshot(p);
    hist_ = key;
    hist_.insert(hist_.end(), out.begin(), out.end());
    if (!opt_.answer_reuse) hist_.resize(std::min<size_t>(hist_.size(), (size_t) p));   // (positions < p: the earlier snapshots' reach)
    st.tokens = (int) out.size();
    st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

} // namespace hyper
