#include "engine7.h"
#include "dense.h"
#include "sampling.h"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>

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

constexpr int SUB = 64;   // indexer: queries scored at a time


} // namespace

struct Engine7::Share {
    int dev = 0, c0 = 0, c1 = 0;
    std::vector<int> part_i0;      // compact indices (absolute)
    std::vector<int *> part_slot;  // on device dev
    int * owner_tab = nullptr;     // on device dev
    int np() const { return part_i0.empty() ? 0 : (int) part_i0.size() - 1; }
};

struct Engine7::Layer {
    int il = 0, ratio = 0, eg = -1;
    bool kv_src = false, ix_src = false, cand_src = false, uses_cand = false;
    int w_out = 0, off_kv = 0, off_ckv = 0, off_cg = 0;   // fused input projection: rows [q_a | kv | c_kv | c_gate]
    DW win, wq_b, iq_b, wo_b, sh_gu, sh_down, eg_kv;
    // wo_a: the o_groups blocks of the block-diagonal output projection stacked ([groups * o_lora][HQ / groups]): decode runs
    // them as one grouped GEMV (wo_a_all; 8 launches of 4 MB each ran ~3x their bandwidth time), prefill the views per group
    DW wo_a_all;
    std::vector<DW> wo_a;
    float * q_norm = nullptr, * kv_norm = nullptr, * c_norm = nullptr, * ik_norm = nullptr;
    float * attn_norm = nullptr, * ffn_norm = nullptr, * sink = nullptr;
    float * hca_fn = nullptr, * hca_scale = nullptr, * hca_base = nullptr, * hcf_fn = nullptr, * hcf_scale = nullptr, * hcf_base = nullptr;
    float * router = nullptr, * bias = nullptr, * eg_w = nullptr, * ik_w = nullptr, * iproj = nullptr;
    // HYPER7_PRED: the next layer's router on this device, its columns scaled by ffn_norm(il + 1) / ffn_norm(il): applied to
    // this layer's MoE input it predicts the next layer's experts for the CPU's prefetch
    float * pr_router = nullptr, * pr_bias = nullptr;
    uint16_t * ring = nullptr;                        // [ring][head_dim] bf16
    float * c_kv = nullptr, * c_score = nullptr;      // the compressor's open group [ratio][head_dim]
    uint16_t * ckv = nullptr, * ik = nullptr;         // KV sources: compressed rows [max_pos / ratio][head_dim], index keys [..][ix_hd]
    // experts: this GPU's (place_->dev(g, il)), the compact host copy (place_->host(il)); n_gpu of them in VRAM here
    int n_gpu = 0;
    size_t gb = 0, db = 0;   // bytes of one expert's gate (= up) / down matrix
    // prefill: the CPU's experts streamed to the GPU in parts (from the pinned compact host copy, index i): part j holds compact
    // indices [part_i0[j], part_i0[j + 1]), part_slot[j] maps an expert to its index in the part (-1: not in it)
    std::vector<int> part_i0;
    std::vector<int *> part_slot;
    int np() const { return part_i0.empty() ? 0 : (int) part_i0.size() - 1; }
    // EP prefill: compact indices [c0, c1) of this layer's CPU experts streamed by device dev (its parts / slots there), the
    // owner's own share is [0, n_own); owner_tab (owner device) / Share::owner_tab (helper): the experts the device sums
    int n_own = 0;
    int * owner_tab = nullptr;
    std::vector<Engine7::Share> shares;
    // expert host: this layer's pooled experts on device host_dev (their copies, slots and owner table: place_->dev(host_dev, il);
    // the owner device's owner_bulk table has them as CPU experts, for the non-streamed bulk path)
    int host_dev = -1, n_host = 0;
};

struct Engine7::DBlock {
    int b = 0, dev = 0;
    DW win, wq_b, wo_b, sh_gu, sh_down, main_proj;
    DW wo_a_all;   // (as the trunk's: one grouped GEMV)
    float * q_norm = nullptr, * kv_norm = nullptr, * attn_norm = nullptr, * ffn_norm = nullptr, * sink = nullptr;
    float * hca_fn = nullptr, * hca_scale = nullptr, * hca_base = nullptr, * hcf_fn = nullptr, * hcf_scale = nullptr, * hcf_base = nullptr;
    float * router = nullptr, * bias = nullptr;
    MoeDev moex;
    int * owner = nullptr;
    int n_exp = 0, k_exp = 0;
    uint16_t * ring = nullptr, * blk = nullptr;   // [ring][hd] seeds; the draft block's own rows [8][hd]
    // the last block: the final norm, the markov head (bf16), the confidence head; main_proj / main_norm (the taps' device)
    float * norm = nullptr, * conf_w = nullptr, * main_norm = nullptr;
    uint16_t * mk_emb = nullptr, * mk_head = nullptr;
    float * mrow = nullptr, * pairs = nullptr, * conf = nullptr;
    // sampled drafts (speculative sampling): the drafter's rows [8][vocab] with their {max, sum exp} [8][2], the drafts'
    // log q [8]; the target rows' {M, Z} [8][2] and log p [8]; the residual draw
    float * dq = nullptr, * qmz = nullptr, * qlog = nullptr, * tmz = nullptr, * plog = nullptr;
    int * resid = nullptr;
};

struct Engine7::Device : DevBase {
    int l0 = 0, l1 = 0;   // layers [l0, l1)
    cudaEvent_t ev_done[NSET] = {}, ev_taken[NSET] = {};   // per chunk set: its rows done / the previous device's rows copied in
    int cur_set = 0;                                    // the chunk set being issued on this device
    // HYPER7_PRED=2: a layer's routing predicted from its input (before the attention) on a side stream, for the CPU's prefetch
    cudaEvent_t ev_pf0 = nullptr, ev_pf1 = nullptr;   // (pe: HYPER7_PRED_EARLY, in the base)
    float * pmix = nullptr, * ppre = nullptr, * ppost = nullptr, * pcomb = nullptr, * px = nullptr, * prlog = nullptr;
    int * owner_all = nullptr;                          // [E] = this device (prefill: every expert computed here)
    // EP prefill helper (taker) resources, one set per owner device it helps (hres[owner's g]): the owner's issuing thread
    // runs its share on these streams, beside this device's own layers and the other owner's share
    bool taker = false;
    size_t hstage_bytes = 0;
    struct HelperRes {
        cudaStream_t hstream = nullptr, hcstream = nullptr;   // helper compute / helper expert uploads
        uint8_t * hstage[2] = {};
        cudaEvent_t hev_up[2] = {}, hev_free[2] = {};
        float * hx = nullptr, * hwts = nullptr, * hyexp = nullptr, * hpart = nullptr;
        int * hids = nullptr, * horder = nullptr, * horder_n = nullptr, * hegrp = nullptr;
        half * hx16 = nullptr, * hh16 = nullptr;
        // the pinned ring between the owner and this helper: input rows + routing in, partials out
        float * ring_x[EP_RING] = {}, * ring_p[EP_RING] = {}, * ring_wts[EP_RING] = {};
        int * ring_ids[EP_RING] = {};
        // (an event is recorded on its own device's streams: the helper's ev_xfree / ev_p, the owner's ev_x / ev_pfree)
        cudaEvent_t ev_xfree[EP_RING] = {}, ev_p[EP_RING] = {};
        int ring_next = 0;
    };
    std::vector<HelperRes> hres;
    float * zeros = nullptr;
    cudaEvent_t ev_x[EP_RING] = {}, ev_pfree[EP_RING] = {};   // as an owner: rows written / partial taken, per ring slot (one helper)
    float * hp_in = nullptr;                            // owner side: a helper's partial copied in [R][n]
    // expert host: decode scratch for the pooled experts (x, routing, hidden, per-pair outputs, the partial, zeros)
    float * qx = nullptr, * qwts = nullptr, * qhexp = nullptr, * qyexp = nullptr, * qout = nullptr, * qzx = nullptr, * qzs = nullptr;
    int * qids = nullptr;
    cudaEvent_t ev_plast = nullptr;                     // owner side: the last partial copied in (the next layer's expert
                                                        // uploads queue behind it: the copy engine served them first, ~80 ms a layer)
    float * h_in = nullptr;                             // pinned [R][hc * n + 8]: this device's input rows (no P2P: through the host)
    // DSpark: the target taps (the device of layers 37..39), main_x rows, the seeds' own stream and fp16 scratch
    float * taps = nullptr, * mainx = nullptr;
    half * xh2 = nullptr;
    cudaStream_t dstream = nullptr;
    cudaEvent_t ev_seed = nullptr;
    cudaEvent_t ev_t[4] = {};   // HYPER7_PROF: timing events around the head (0, 1) and the DSpark seed (2, 3)
    std::vector<Layer> layers;
    // activations (R rows)
    float * h = nullptr, * h2 = nullptr, * x = nullptr, * mix = nullptr, * pre = nullptr, * a_pre = nullptr, * a_post = nullptr,
          * a_comb = nullptr, * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    float * wout = nullptr, * q = nullptr, * o = nullptr, * oa = nullptr, * ao = nullptr, * latent = nullptr, * ikr = nullptr, * iq = nullptr,
          * iw = nullptr, * scores = nullptr, * bm = nullptr, * rlog = nullptr, * wts = nullptr, * ones = nullptr, * shgu = nullptr,
          * shh = nullptr, * shout = nullptr, * hexp = nullptr, * yexp = nullptr, * ffo = nullptr, * eg_x = nullptr, * eg_kvo = nullptr,
          * logits = nullptr, * outp = nullptr;
    int * ids = nullptr, * sel = nullptr, * cand = nullptr, * ncand = nullptr, * order = nullptr, * order_n = nullptr, * egrp = nullptr,
        * counter = nullptr, * pids = nullptr;
    float * pwts = nullptr;
    half * xh = nullptr, * x16 = nullptr, * h16 = nullptr;
    half * ggs = nullptr;   // KQ dense weights: prefill dequantization scratch
    size_t ggs_elems = 0;
    long long score_w = 0, bm_w = 0;
    float * freqs_win = nullptr, * freqs_comp = nullptr;
    // the head (last device)
    float * out_norm = nullptr;
    DW output;

    ~Device() {
        cudaSetDevice(id);
        for (auto ev : ev_done) if (ev) cudaEventDestroy(ev);
        for (auto ev : ev_taken) if (ev) cudaEventDestroy(ev);
        if (h_in) cudaFreeHost(h_in);
        if (dstream) cudaStreamDestroy(dstream);
        if (ev_seed) cudaEventDestroy(ev_seed);
        for (auto ev : {ev_pf0, ev_pf1}) if (ev) cudaEventDestroy(ev);
        for (int r = 0; r < EP_RING; ++r) for (auto ev : {ev_x[r], ev_pfree[r]}) if (ev) cudaEventDestroy(ev);
        for (HelperRes & H : hres) {
            for (int r = 0; r < EP_RING; ++r) {
                for (auto ev : {H.ev_xfree[r], H.ev_p[r]}) if (ev) cudaEventDestroy(ev);
                for (void * p : {(void *) H.ring_x[r], (void *) H.ring_p[r], (void *) H.ring_wts[r], (void *) H.ring_ids[r]}) if (p) cudaFreeHost(p);
            }
            for (auto ev : H.hev_up) if (ev) cudaEventDestroy(ev);
            for (auto ev : H.hev_free) if (ev) cudaEventDestroy(ev);
            if (H.hstream) cudaStreamDestroy(H.hstream);
            if (H.hcstream) cudaStreamDestroy(H.hcstream);
        }
        for (auto ev : ev_t) if (ev) cudaEventDestroy(ev);
    }
};

Engine7::Engine7(const std::string & model_path, const Engine7Options & opt) : opt_(opt) {
    debug_ = getenv("HYPER7_DEBUG") != nullptr;
    stream_ = getenv("HYPER7_NOSTREAM") == nullptr;
    if (getenv("HYPER7_STREAM_MIN")) stream_min_ = atoi(getenv("HYPER7_STREAM_MIN"));
    gguf_ = std::make_unique<GGUF>(model_path);
    cfg_ = Ds4Config::from_gguf(*gguf_);
    fprintf(stderr, "hyper7: %s\n", cfg_.describe().c_str());
    const Ds4Config & c = cfg_;
    if (getenv("HYPER7_CHUNK")) R_ = std::max(16, std::min(MOE_BULK_ROWS, atoi(getenv("HYPER7_CHUNK"))));
    R_ = std::min(R_, MOE_BULK_ROWS);
    if (getenv("HYPER7_RING")) ring_ = atoi(getenv("HYPER7_RING"));
    if (ring_ < R_ + c.win + 64) ring_ = R_ + c.win + 64;
    if (opt_.max_pos % 64) opt_.max_pos += 64 - opt_.max_pos % 64;
    const int nd = opt_.n_devices = device_count(opt_.n_devices, "HYPER7_NDEV", "hyper7");   // (a 4th GPU: with HYPER7_SPLIT)
    // expert hosts: the other devices' spare VRAM (after their own experts) holds the hottest CPU experts of the target
    // device's layers (2026-11-05: pooling 12 GiB out of a host's own budget gained ~0; the spare VRAM is free gain)
    host_on_ = getenv("HYPER7_HOST") ? atoi(getenv("HYPER7_HOST")) != 0 : nd >= 4;
    host_dev_ = getenv("HYPER7_HOST_DEV") ? atoi(getenv("HYPER7_HOST_DEV")) : -1;
    if (host_dev_ >= nd) host_dev_ = -1;
    host_pool_bytes_ = (size_t) ((getenv("HYPER7_HOST_GIB") ? atof(getenv("HYPER7_HOST_GIB")) : 64.0) * 1073741824.0);
    host_layers_.assign(nd, {});
    // layers over the devices along the KV-source groups (HYPER7_SPLIT="8,20": the first layers of devices 1, 2, ..)
    std::vector<int> starts = {0};
    if (const char * sp = getenv("HYPER7_SPLIT")) {
        for (const char * q = sp; *q;) { starts.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    } else if (nd == 4) starts = {0, 8, 14, 20};   // (the production split on this box: the 32 GB card takes layers 0-7)
    else if (nd == 3) starts = {0, 8, 20};
    else if (nd == 2) starts = {0, 14};
    starts.resize(std::min<size_t>(starts.size(), nd));
    dev_of_.assign(c.n_layer, 0);
    for (int g = 0; g < (int) starts.size(); ++g)
        for (int il = starts[g]; il < (g + 1 < (int) starts.size() ? starts[g + 1] : c.n_layer); ++il) dev_of_[il] = g;
    for (int il = 0; il < c.n_layer; ++il) {   // a layer reads its sources' state on its own device
        if (c.kv_of[il] >= 0 && dev_of_[c.kv_of[il]] != dev_of_[il]) throw std::runtime_error("hyper7: split separates a KV group");
        if (c.ix_of[il] >= 0 && dev_of_[c.ix_of[il]] != dev_of_[il]) throw std::runtime_error("hyper7: split separates an index group");
        if (c.cand_src >= 0 && c.ix_src[il] && il > c.cand_src && dev_of_[c.cand_src] != dev_of_[il])
            throw std::runtime_error("hyper7: split separates the candidate source");
    }
    const int n = c.n_embd;
    for (int g = 0; g < (int) starts.size(); ++g) {
        auto d = std::make_unique<Device>();
        d->init_base(g, g, true);
        d->l0 = starts[g];
        d->l1 = g + 1 < (int) starts.size() ? starts[g + 1] : c.n_layer;
        for (auto & ev : d->ev_done) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        for (auto & ev : d->ev_taken) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        if (g > 0) CUDA_CHECK(cudaHostAlloc(&d->h_in, (size_t) R_ * (c.hc * c.n_embd + 8) * sizeof(float), cudaHostAllocPortable));
        CUDA_CHECK(cudaEventCreateWithFlags(&d->ev_pf0, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&d->ev_pf1, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(&d->dstream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&d->ev_seed, cudaEventDisableTiming));
        devs_.push_back(std::move(d));
    }
    for (int k = 0; k < NSET; ++k) {
        CUDA_CHECK(cudaHostAlloc(&h_embd_[k], (size_t) R_ * n * sizeof(float), cudaHostAllocPortable));
        for (size_t i = 0; i < c.eg_layers.size(); ++i) {
            float * p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, (size_t) R_ * c.eg_cols() * c.eg_hd * sizeof(float), cudaHostAllocPortable));
            h_eg_[k].push_back(p);
        }
        CUDA_CHECK(cudaHostAlloc(&h_ids_[k], (size_t) c.n_layer * R_ * c.n_used * sizeof(int), cudaHostAllocPortable));
    }
    CUDA_CHECK(cudaHostAlloc(&h_out_, (size_t) MAX_NT * TOPK * 2 * sizeof(float) + 64, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&host_out_, (size_t) c.n_layer * sizeof(CpuMoeOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) host_out_, 0, (size_t) c.n_layer * sizeof(CpuMoeOut));
    init_cpu_side(opt_.cpu_threads, n, c.ff, c.n_used, c.n_layer, c.clamp_exp, pred_mode_ == 2 && pred_early_);
    const auto t0 = std::chrono::steady_clock::now();
    for (int il = 0; il < c.n_layer; ++il) has_r2_ |= c.ratio[il] > 1;
    // per-device buffers
    const std::vector<float> fw = c.freqs(false), fc = c.freqs(true);
    const int HQ = c.n_head * c.head_dim, K = c.n_used;
    int w_max = 0;
    for (int il = 0; il < c.n_layer; ++il) w_max = std::max(w_max, c.q_lora + c.head_dim + (c.kv_src[il] ? c.head_dim * (c.ratio[il] > 1 ? 2 : 1) : 0));
    for (auto & dp : devs_) {
        Device & d = *dp;
        const size_t R = R_;
        d.h = d.alloc<float>(R * c.hc * n);
        d.h2 = d.alloc<float>(R * c.hc * n);
        d.x = d.alloc<float>(R * n);
        d.mix = d.alloc<float>(R * 32);
        for (float ** p : {&d.pre, &d.a_pre, &d.a_post, &d.f_pre, &d.f_post}) *p = d.alloc<float>(R * 8);
        d.a_comb = d.alloc<float>(R * 64);
        d.f_comb = d.alloc<float>(R * 64);
        d.wout = d.alloc<float>(R * w_max);
        d.q = d.alloc<float>(R * HQ);
        d.o = d.alloc<float>(R * HQ);
        d.oa = d.alloc<float>(R * c.o_groups * c.o_lora);
        d.ao = d.alloc<float>(R * n);
        d.latent = d.alloc<float>(R * c.head_dim);
        d.ikr = d.alloc<float>(R * c.ix_hd);
        d.iq = d.alloc<float>(R * c.ix_heads * c.ix_hd);
        d.iw = d.alloc<float>(R * 64);
        d.score_w = (long long) opt_.max_pos + 64;
        d.scores = d.alloc<float>((size_t) SUB * d.score_w);
        d.bm_w = (long long) opt_.max_pos / std::max(1, c.cand_blk) + 64;
        d.bm = d.alloc<float>((size_t) SUB * d.bm_w);
        d.sel = d.alloc<int>(R * c.ix_topk);
        d.cand = d.alloc<int>(R * std::max(1, c.cand_topk));
        d.ncand = d.alloc<int>(R);
        d.rlog = d.alloc<float>(R * c.n_expert);
        d.ids = d.alloc<int>(R * K);
        d.wts = d.alloc<float>(R * K);
        d.pids = d.alloc<int>(MAX_NT * MOE_MAX_USED);
        d.pwts = d.alloc<float>(MAX_NT * MOE_MAX_USED);
        if (pred_mode_ == 2) {
            d.pmix = d.alloc<float>(MAX_NT * 32);
            d.ppre = d.alloc<float>(MAX_NT * 8);
            d.ppost = d.alloc<float>(MAX_NT * 8);
            d.pcomb = d.alloc<float>(MAX_NT * 64);
            d.px = d.alloc<float>(MAX_NT * n);
            d.prlog = d.alloc<float>(MAX_NT * c.n_expert);
        }
        d.ones = d.alloc<float>(R);
        fill7(d.ones, 1.0f, (int) R, d.stream);
        d.shgu = d.alloc<float>(R * 2 * c.ff);
        d.shh = d.alloc<float>(R * c.ff);
        d.shout = d.alloc<float>(R * n);
        d.hexp = d.alloc<float>((size_t) MAX_NT * K * c.ff);
        d.yexp = d.alloc<float>(R * K * n);
        d.ffo = d.alloc<float>(R * n);
        d.order = d.alloc<int>(R * K);
        d.order_n = d.alloc<int>(4);
        d.egrp = d.alloc<int>(3 * c.n_expert + 16);
        d.counter = d.alloc<int>(1);
        d.xh = d.alloc<half>(R * std::max(HQ, std::max(c.eg_cols() * c.eg_hd, c.o_groups * c.o_lora)));
        d.x16 = d.alloc<half>(R * n);
        d.h16 = d.alloc<half>(R * K * c.ff);
        bool has_eg = false;
        for (int il = d.l0; il < d.l1; ++il) has_eg |= c.eg_index(il) >= 0;
        if (has_eg) {
            d.eg_x = d.alloc<float>(R * c.eg_cols() * c.eg_hd);
            d.eg_kvo = d.alloc<float>(R * (c.hc + 1) * n);
        }
        d.freqs_win = d.upload(fw.data(), fw.size());
        d.freqs_comp = d.upload(fc.data(), fc.size());
        d.layers.resize(d.l1 - d.l0);
        for (int il = d.l0; il < d.l1; ++il) {
            Layer & L = d.layers[il - d.l0];
            L.il = il;
            load_layer(d, L);
        }
        if (d.g == (int) devs_.size() - 1) {
            d.out_norm = d.upload(to_f32(gguf_->need("output_norm.weight")).data(), n);
            const GTensor & to = gguf_->need("output.weight");
            d.output = upload_dense([&](size_t nb) { return (void *) d.alloc<uint8_t>(nb); }, d.id, {{&to, 0, to.rows()}});
            d.logits = d.alloc<float>((size_t) 8 * c.n_vocab);   // (DSpark: a draft block's rows too)
            d.outp = d.alloc<float>((size_t) MAX_NT * TOPK * 2 + 64);
        }
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        fprintf(stderr, "hyper7: device %d: layers %d..%d, dense weights + caches + buffers %.2f GiB\n", d.id, d.l0, d.l1 - 1,
                d.used / 1073741824.0);
    }
    if (dense_ggs_elems())   // (KQ dense weights: unsloth's Q6_K attention, shared experts, head)
        for (auto & dp : devs_) {
            CUDA_CHECK(cudaSetDevice(dp->id));
            dp->ggs_elems = dense_ggs_elems();
            dp->ggs = dp->alloc<half>(dp->ggs_elems);
        }
    {
        std::string dp = opt_.dspark;
        if (getenv("HYPER7_DSPARK")) dp = getenv("HYPER7_DSPARK");
        dsp_conf_ = getenv("HYPER7_DSPARK_CONF") ? (float) atof(getenv("HYPER7_DSPARK_CONF")) : opt_.dspark_conf;
        // speculative sampling (sampled drafts, min(1, p/q)) measured 2026-11-05 at T=1 over 256 tokens: drafts accepted 59 %
        // vs 60 % with the exact match of greedy drafts, and ~20 ms more a step (extra stream syncs): off unless asked
        dsp_coupled_ = !(getenv("HYPER7_DSPARK_SPECSAMP") && atoi(getenv("HYPER7_DSPARK_SPECSAMP")) != 0);
        if (!dp.empty() && dp != "none") load_dspark(dp);
    }
    {   // the experts' placement (shared with the other engines): routing statistics, adaptive exchanges
        MoePlace::Params pp;
        pp.tag = "hyper7"; pp.env = "HYPER7";
        pp.E = c.n_expert; pp.K = c.n_used; pp.nl = c.n_layer;
        pp.ff = c.ff; pp.n = n; pp.clamp = c.clamp_exp;
        pp.tp = false; pp.halves = 0; pp.elastic_ok = false;
        pp.adapt_budget = 8; pp.adapt_after_prompt = false;   // (26k A/B: budget 32 cost 2 t/s of 46, 8 within noise)
        std::vector<DevBase *> db;
        for (auto & dp : devs_) db.push_back(dp.get());
        init_place(pp, db, c.n_layer);
        place_->on_changed = [this](int il) { Device & d = *devs_[dev_of_[il]]; rebuild_part_slots(d.layers[il - d.l0]); };
    }
    place_experts();
    CUDA_CHECK(cudaSetDevice(devs_.back()->id));
    for (auto & ev : ev_chunk_) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    fprintf(stderr, "hyper7: loaded in %.1f s (chunk %d, ring %d, max_pos %d)\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), R_, ring_, opt_.max_pos);
}

void * engine7_huge_alloc(size_t bytes, std::vector<std::pair<void *, size_t>> & bufs) {
    const size_t H = 2u << 20, sz = (bytes + H - 1) / H * H;
    void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("hyper7: mmap failed");
    madvise(p, sz, MADV_HUGEPAGE);
    bufs.push_back({p, sz});
    return p;
}

void Engine7::load_layer(Device & d, Layer & L) {
    const Ds4Config & c = cfg_;
    const GGUF & G = *gguf_;
    const int il = L.il;
    auto A = [&](size_t nb) { return (void *) d.alloc<uint8_t>(nb); };
    const std::string p = "blk." + std::to_string(il) + ".";
    auto T = [&](const std::string & name) { return &G.need(p + name); };
    auto f32 = [&](const std::string & name, float scale = 1.0f) {
        std::vector<float> v = to_f32(G.need(p + name));
        if (scale != 1.0f) for (float & x : v) x *= scale;
        return d.upload(v.data(), v.size());
    };
    L.ratio = c.ratio[il];
    L.kv_src = c.kv_src[il];
    L.ix_src = c.ix_src[il];
    L.cand_src = il == c.cand_src;
    L.uses_cand = L.ix_src && c.cand_src >= 0 && il > c.cand_src;
    L.eg = c.eg_index(il);
    // fused input projection [q_a | kv | c_kv | c_gate]
    std::vector<RowRange> parts;
    const GTensor * tqa = T("attn_q_a.weight"), * tkv = T("attn_kv.weight");
    parts.push_back({tqa, 0, tqa->rows()});
    parts.push_back({tkv, 0, tkv->rows()});
    L.off_kv = c.q_lora;
    L.w_out = c.q_lora + c.head_dim;
    if (L.kv_src) {
        const GTensor * tck = T("attn_compressor_kv.weight");
        parts.push_back({tck, 0, tck->rows()});
        L.off_ckv = L.w_out;
        L.w_out += c.head_dim;
        if (L.ratio > 1) {
            const GTensor * tcg = T("attn_compressor_gate.weight");
            parts.push_back({tcg, 0, tcg->rows()});
            L.off_cg = L.w_out;
            L.w_out += c.head_dim;
        }
        L.c_norm = f32("attn_compressor_norm.weight");
    }
    L.win = upload_dense(A, d.id, parts);
    { const GTensor * t = T("attn_q_b.weight"); L.wq_b = upload_dense(A, d.id, {{t, 0, t->rows()}}); }
    {   // wo_a: group g = rows g * o_lora .. on input values g * (HQ / groups) ..
        const GTensor * t = T("attn_output_a.weight");
        if (t->rows() != (int64_t) c.o_groups * c.o_lora || t->ne[0] != (int64_t) c.n_head * c.head_dim / c.o_groups)   // ([in][groups * o_lora] or ISTA's [in][o_lora][groups])
            throw std::runtime_error("hyper7: attn_output_a shape");
        L.wo_a_all = upload_dense(A, d.id, {{t, 0, t->rows()}});
        for (int gi = 0; gi < c.o_groups; ++gi) L.wo_a.push_back(dense_rows_view(L.wo_a_all, gi * c.o_lora, (gi + 1) * c.o_lora));
        if (L.wo_a[0].n() != c.o_lora || c.o_lora % 16) {   // (no view of this form: the groups uploaded apart, decode loops)
            L.wo_a.clear();
            L.wo_a_all = DW{};
            for (int gi = 0; gi < c.o_groups; ++gi) L.wo_a.push_back(upload_dense(A, d.id, {{t, (int64_t) gi * c.o_lora, (int64_t) (gi + 1) * c.o_lora}}));
        }
    }
    { const GTensor * t = T("attn_output_b.weight"); L.wo_b = upload_dense(A, d.id, {{t, 0, t->rows()}}); }
    L.q_norm = f32("attn_q_a_norm.weight");
    L.kv_norm = f32("attn_kv_a_norm.weight");
    L.sink = f32("attn_sinks.weight");
    L.attn_norm = f32("attn_norm.weight");
    L.ffn_norm = f32("ffn_norm.weight");
    L.hca_fn = f32("hc_attn_fn.weight");
    L.hca_scale = f32("hc_attn_scale.weight");
    L.hca_base = f32("hc_attn_base.weight");
    L.hcf_fn = f32("hc_ffn_fn.weight");
    L.hcf_scale = f32("hc_ffn_scale.weight");
    L.hcf_base = f32("hc_ffn_base.weight");
    if (L.ix_src) {
        const GTensor * t = T("indexer.attn_q_b.weight");
        L.iq_b = upload_dense(A, d.id, {{t, 0, t->rows()}});
        // the weights' scale 1 / sqrt(hd) / sqrt(heads) (= 1/64: a power of two, exact) folded into the projection
        L.iproj = f32("indexer.proj.weight", 1.0f / std::sqrt((float) c.ix_hd) / std::sqrt((float) c.ix_heads));
        if (L.kv_src) {
            L.ik_w = f32("indexer.attn_k.weight");
            L.ik_norm = f32("indexer.k_norm.weight");
        }
    }
    // window ring, compressor state, compressed caches
    L.ring = d.alloc<uint16_t>((size_t) ring_ * c.head_dim);
    if (L.kv_src) {
        const size_t rows = (size_t) opt_.max_pos / L.ratio + 8;
        L.ckv = d.alloc<uint16_t>(rows * c.head_dim);
        if (L.ix_src) L.ik = d.alloc<uint16_t>(rows * c.ix_hd);
        if (L.ratio > 1) {
            L.c_kv = d.alloc<float>((size_t) L.ratio * c.head_dim);
            L.c_score = d.alloc<float>((size_t) L.ratio * c.head_dim);
        }
    }
    // shared expert: [gate | up] and down
    {
        const GTensor * tg = T("ffn_gate_shexp.weight"), * tu = T("ffn_up_shexp.weight"), * td = T("ffn_down_shexp.weight");
        L.sh_gu = upload_dense(A, d.id, {{tg, 0, tg->rows()}, {tu, 0, tu->rows()}});
        L.sh_down = upload_dense(A, d.id, {{td, 0, td->rows()}});
    }
    L.router = f32("ffn_gate_inp.weight");
    L.bias = f32("exp_probs_b.bias");
    if ((pred_mode_ == 1 || (pred_mode_ == 2 && pred_early_)) && il + 1 < c.n_layer) {
        const std::string pn = "blk." + std::to_string(il + 1) + ".";
        std::vector<float> W = to_f32(G.need(pn + "ffn_gate_inp.weight")), b = to_f32(G.need(pn + "exp_probs_b.bias"));
        const std::vector<float> n0 = to_f32(G.need(p + "ffn_norm.weight")), n1 = to_f32(G.need(pn + "ffn_norm.weight"));
        const size_t k = n0.size();
        if (W.size() != k * (size_t) c.n_expert) throw std::runtime_error("hyper7: router shape");
        for (size_t e = 0; e < (size_t) c.n_expert; ++e)
            for (size_t j = 0; j < k; ++j) W[e * k + j] *= std::fabs(n0[j]) > 1e-6f ? n1[j] / n0[j] : 0.0f;
        L.pr_router = d.upload(W.data(), W.size());
        L.pr_bias = d.upload(b.data(), b.size());
    }
    if (L.eg >= 0) {
        const GTensor * t = G.tensor(p + "engram_kv.weight") ? T("engram_kv.weight") : T("engram_wkv.weight");
        L.eg_kv = upload_dense(A, d.id, {{t, 0, t->rows()}});
        if (G.tensor(p + "engram_gate.weight")) L.eg_w = f32("engram_gate.weight");
        else {   // (ISTA-DASLab: q_weight and k_weight apart, BF16 -- their product is exact in F32)
            std::vector<float> q = to_f32(G.need(p + "engram_q.weight")), k = to_f32(G.need(p + "engram_k.weight"));
            if (q.size() != k.size() || q.size() != (size_t) c.hc * c.n_embd) throw std::runtime_error("hyper7: engram_q / engram_k shape");
            for (size_t i = 0; i < q.size(); ++i) q[i] *= k[i];
            L.eg_w = d.upload(q.data(), q.size());
        }
    }
}


// ---- DSpark drafter -------------------------------------------------------------------------------------------------
// dspark.bin (Project Maya's drafter_pack.py from DeepSeek's checkpoint: the three mtp.* blocks; dense matrices Q8_0, routed
// experts MXFP4 stacked per role, the rest BF16 / F32) with dspark.idx beside it (name, type, offset, bytes, shape [rows, cols]).
// The drafter follows DeepSeek's inference/model.py (DSparkBlock, forward_spec) as Maya's tools/ds4/dspark_ref.py does: the
// streams' mean entering target layers 37..39 -> main_proj -> main_norm -> every block's window ring (wkv, kv_norm, rope,
// FP8); a round runs [y, noise x 4] at the positions after y's through the three blocks (window attention over the ring and
// the whole block, 128-expert MoE with 3 picks), collapses the streams, and the trunk's head plus the markov bias of the token
// before each position gives the drafts (greedy chain), a sigmoid confidence head their confidences.
void Engine7::load_dspark(const std::string & path) {
    const Ds4Config & c = cfg_;
    std::string idx = path;
    if (idx.size() > 4 && idx.compare(idx.size() - 4, 4, ".bin") == 0) idx = idx.substr(0, idx.size() - 4);
    idx += ".idx";
    FILE * fi = fopen(idx.c_str(), "r");
    if (!fi) { fprintf(stderr, "hyper7: no DSpark index %s, drafter off\n", idx.c_str()); return; }
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fclose(fi); fprintf(stderr, "hyper7: cannot open %s, drafter off\n", path.c_str()); return; }
    dsp_size_ = (size_t) lseek(fd, 0, SEEK_END);
    dsp_map_ = mmap(nullptr, dsp_size_, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (dsp_map_ == MAP_FAILED) throw std::runtime_error("hyper7: mmap of the drafter failed");
    std::map<std::string, GTensor> T;
    char line[512];
    while (fgets(line, sizeof line, fi)) {
        char name[256], dt[32];
        long long off, nb, r = 0, cc = 0;
        const int n = sscanf(line, "%255s %31s %lld %lld %lld %lld", name, dt, &off, &nb, &r, &cc);
        if (n < 5) continue;
        GTensor t;
        t.name = name;
        const std::string ds = dt;
        t.type = ds == "Q8_0" ? GType::Q8_0 : ds == "BF16" ? GType::BF16 : ds == "F32" ? GType::F32 : ds == "MXFP4" ? GType::MXFP4 : GType::I8;
        if (t.type == GType::I8) throw std::runtime_error("hyper7: drafter tensor type " + ds);
        t.ne = n >= 6 ? std::vector<int64_t>{cc, r} : std::vector<int64_t>{r};
        t.data = (const uint8_t *) dsp_map_ + off;
        t.nbytes = (size_t) nb;
        T[name] = t;
    }
    fclose(fi);
    auto need = [&](const std::string & k) -> const GTensor & {
        auto it = T.find(k);
        if (it == T.end()) throw std::runtime_error("hyper7: drafter tensor " + k + " missing");
        return it->second;
    };
    dsp_noise_ = 128799;   // (DeepSeek's dspark_noise_token_id; the GGUF does not carry the drafter's config)
    if (gguf_->has("deepseek41.dspark.noise_token_id")) dsp_noise_ = (int) gguf_->get_int("deepseek41.dspark.noise_token_id");
    if (gguf_->has("deepseek41.dspark.block_size")) dsp_T_ = (int) gguf_->get_int("deepseek41.dspark.block_size");
    std::vector<int64_t> tl = gguf_->has("deepseek41.dspark.target_layers") ? gguf_->get_int_arr("deepseek41.dspark.target_layers")
                                                                             : std::vector<int64_t>{37, 38, 39};
    dsp_tap0_ = (int) tl[0];
    if (tl.size() != 3 || tl[1] != tl[0] + 1 || tl[2] != tl[0] + 2 || dev_of_[tl[0]] != dev_of_[tl[2]])
        throw std::runtime_error("hyper7: DSpark taps must be three consecutive layers on one device");
    if (dsp_T_ > 8) throw std::runtime_error("hyper7: DSpark block too long");
    const int nb = 3, HD = c.head_dim, n = c.n_embd;
    size_t tot = 0;
    for (int b = 0; b < nb; ++b) {
        auto B = std::make_unique<DBlock>();
        B->b = b;
        B->dev = std::min(b, (int) devs_.size() - 1) + std::max(0, (int) devs_.size() - 3);   // (the last block with the head: the last device)
        Device & d = *devs_[B->dev];
        const size_t used0 = d.used;
        auto A = [&](size_t bytes) { return (void *) d.alloc<uint8_t>(bytes); };
        const std::string P = "mtp." + std::to_string(b) + ".";
        auto f32 = [&](const std::string & k) { const std::vector<float> v = to_f32(need(k)); return d.upload(v.data(), v.size()); };
        {
            const GTensor & qa = need(P + "attn.wq_a.weight"), & kv = need(P + "attn.wkv.weight");
            B->win = upload_dense(A, d.id, {{&qa, 0, qa.rows()}, {&kv, 0, kv.rows()}}, {}, true);
        }
        { const GTensor & t = need(P + "attn.wq_b.weight"); B->wq_b = upload_dense(A, d.id, {{&t, 0, t.rows()}}, {}, true); }
        { const GTensor & t = need(P + "attn.wo_a.weight"); B->wo_a_all = upload_dense(A, d.id, {{&t, 0, t.rows()}}, {}, true); }
        { const GTensor & t = need(P + "attn.wo_b.weight"); B->wo_b = upload_dense(A, d.id, {{&t, 0, t.rows()}}, {}, true); }
        { const GTensor & g1 = need(P + "ffn.shared_experts.w1.weight"), & g3 = need(P + "ffn.shared_experts.w3.weight"), & g2 = need(P + "ffn.shared_experts.w2.weight");
          B->sh_gu = upload_dense(A, d.id, {{&g1, 0, g1.rows()}, {&g3, 0, g3.rows()}}, {}, true);
          B->sh_down = upload_dense(A, d.id, {{&g2, 0, g2.rows()}}, {}, true); }
        B->q_norm = f32(P + "attn.q_norm.weight");
        B->kv_norm = f32(P + "attn.kv_norm.weight");
        B->sink = f32(P + "attn.attn_sink");
        B->attn_norm = f32(P + "attn_norm.weight");
        B->ffn_norm = f32(P + "ffn_norm.weight");
        B->hca_fn = f32(P + "hc_attn_fn"); B->hca_scale = f32(P + "hc_attn_scale"); B->hca_base = f32(P + "hc_attn_base");
        B->hcf_fn = f32(P + "hc_ffn_fn"); B->hcf_scale = f32(P + "hc_ffn_scale"); B->hcf_base = f32(P + "hc_ffn_base");
        B->router = f32(P + "ffn.gate.weight");
        B->bias = f32(P + "ffn.gate.bias");
        B->n_exp = (int) need(P + "ffn.gate.weight").ne[1];
        B->k_exp = 3;
        {   // routed experts: every one in VRAM, gate / up / down stacked as the pack wrote them
            const GTensor & e1 = need(P + "ffn.experts.0.w1.weight"), & e3 = need(P + "ffn.experts.0.w3.weight"), & e2 = need(P + "ffn.experts.0.w2.weight");
            const size_t gb = e1.nbytes, db = e2.nbytes, E = (size_t) B->n_exp;
            for (size_t e = 1; e < E; ++e)
                if (need(P + "ffn.experts." + std::to_string(e) + ".w1.weight").data != e1.data + e * gb ||
                    need(P + "ffn.experts." + std::to_string(e) + ".w3.weight").data != e3.data + e * gb ||
                    need(P + "ffn.experts." + std::to_string(e) + ".w2.weight").data != e2.data + e * db)
                    throw std::runtime_error("hyper7: drafter experts not stacked");
            uint8_t * gg = d.alloc<uint8_t>(E * gb), * gu = d.alloc<uint8_t>(E * gb), * gd = d.alloc<uint8_t>(E * db);
            CUDA_CHECK(cudaMemcpy(gg, e1.data, E * gb, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gu, e3.data, E * gb, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gd, e2.data, E * db, cudaMemcpyHostToDevice));
            std::vector<int> slot(E), owner(E, d.g);
            for (size_t e = 0; e < E; ++e) slot[e] = (int) e;
            MoeDev & m = B->moex;
            m.gate = gg; m.up = gu; m.down = gd;
            m.tg = e1.type; m.td = e2.type;
            m.gate_bytes = gb; m.down_bytes = db;
            m.slot = d.upload(slot.data(), E);
            m.ff = (int) e1.ne[1]; m.n = n;
            m.clamp = c.clamp_exp;
            B->owner = d.upload(owner.data(), E);
        }
        B->ring = d.alloc<uint16_t>((size_t) ring_ * HD);
        B->blk = d.alloc<uint16_t>((size_t) 8 * HD);
        if (b == nb - 1) {
            B->norm = f32(P + "norm.weight");
            B->conf_w = f32(P + "confidence_head.proj.weight");
            const GTensor & me = need(P + "markov_head.embed.weight"), & mh = need(P + "markov_head.head.weight");
            dsp_rank_ = (int) me.ne[0];
            B->mk_emb = d.upload((const uint16_t *) me.data, me.nbytes / 2);
            B->mk_head = d.upload((const uint16_t *) mh.data, mh.nbytes / 2);
            B->mrow = d.alloc<float>(c.n_vocab);
            B->pairs = d.alloc<float>(2 * 8);
            B->conf = d.alloc<float>(8);
            B->dq = d.alloc<float>((size_t) 8 * c.n_vocab);
            B->qmz = d.alloc<float>(16);
            B->qlog = d.alloc<float>(8);
            B->tmz = d.alloc<float>(16);
            B->plog = d.alloc<float>(8);
            B->resid = d.alloc<int>(1);
        }
        tot += d.used - used0;
        dblk_.push_back(std::move(B));
    }
    {   // main_proj / main_norm, the taps and main_x on the taps' device; main_x on every drafter device; the seeds' scratch
        Device & td = *devs_[dev_of_[dsp_tap0_]];
        DBlock & B0 = *dblk_[0];
        const GTensor & mp = need("mtp.0.main_proj.weight");
        B0.main_proj = upload_dense([&](size_t bytes) { return (void *) td.alloc<uint8_t>(bytes); }, td.id, {{&mp, 0, mp.rows()}}, {}, true);
        B0.main_norm = td.upload(to_f32(need("mtp.0.main_norm.weight")).data(), n);
        td.taps = td.alloc<float>((size_t) R_ * 3 * n);
        for (auto & dp : devs_) {
            dp->mainx = dp->alloc<float>((size_t) R_ * n);
            dp->xh2 = dp->alloc<half>((size_t) R_ * 3 * n);
        }
    }
    CUDA_CHECK(cudaHostAlloc(&h_dsp_, (size_t) R_ * (c.hc * n + 8) * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_dout_, 64 * sizeof(float), cudaHostAllocPortable));
    munmap(dsp_map_, dsp_size_);   // (everything is in VRAM now)
    dsp_map_ = nullptr;
    fprintf(stderr, "hyper7: DSpark drafter from %s: 3 blocks (%d experts, top %d), block of %d, %.2f GiB in VRAM\n", path.c_str(),
            dblk_[0]->n_exp, dblk_[0]->k_exp, dsp_T_, tot / 1073741824.0);
}

// the drafter's ring rows of positions pos .. pos + nt - 1 from this forward's taps (enqueued after the forward; the other
// devices' blocks on their seed streams)
void Engine7::dspark_seed(int nt, int64_t pos) {
    const Ds4Config & c = cfg_;
    const int n = c.n_embd, HD = c.head_dim;
    Device & td = *devs_[dev_of_[dsp_tap0_]];
    DBlock & B0 = *dblk_[0];
    CUDA_CHECK(cudaSetDevice(td.id));
    cudaStream_t s = td.stream;
    // main_x = rmsnorm(main_proj taps) * main_norm
    if (nt <= MAX_NT) gemv_q8(B0.main_proj.q8, td.taps, 3 * n, td.mainx, n, nullptr, nt, s);
    else { to_half(td.taps, 3 * n, nullptr, 3 * n, 0.0f, td.xh2, nt, s); gemm_q8(B0.main_proj.q8, td.xh2, nt, td.mainx, n, nullptr, s); }
    rmsnorm(td.mainx, n, B0.main_norm, td.mainx, n, n, nt, c.eps, s);
    // the other devices take main_x through the host (their previous seeds' copies are done first)
    bool remote = false;
    for (auto & B : dblk_) remote |= devs_[B->dev].get() != &td;
    if (remote) {
        for (auto & B : dblk_) if (devs_[B->dev].get() != &td) CUDA_CHECK(cudaStreamWaitEvent(s, devs_[B->dev]->ev_seed, 0));
        CUDA_CHECK(cudaMemcpyAsync(h_dsp_, td.mainx, (size_t) nt * n * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaEventRecord(td.ev_seed, s));
    }
    for (auto & Bp : dblk_) {
        DBlock & B = *Bp;
        Device & d = *devs_[B.dev];
        CUDA_CHECK(cudaSetDevice(d.id));
        cudaStream_t ss = &d == &td ? s : d.dstream;
        if (&d != &td) {
            CUDA_CHECK(cudaStreamWaitEvent(ss, td.ev_seed, 0));
            CUDA_CHECK(cudaMemcpyAsync(d.mainx, h_dsp_, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, ss));
            CUDA_CHECK(cudaEventRecord(d.ev_seed, ss));
        }
        // kv = rmsnorm(main_x wkv) * kv_norm, rope, FP8 -> ring rows of the positions (wkv: rows q_lora .. of win)
        float * kv = (float *) d.xh2;   // [nt][q_lora + HD] floats (xh2 holds R * 3n halves)
        if (&d != &td) gemv_scratch_slot(1);
        if (nt <= MAX_NT) gemv_q8(B.win.q8, d.mainx, n, kv, c.q_lora + HD, nullptr, nt, ss);
        else {
            half * xh = (half *) ((float *) d.xh2 + (size_t) R_ * (c.q_lora + HD));
            to_half(d.mainx, n, nullptr, n, 0.0f, xh, nt, ss);
            gemm_q8(B.win.q8, xh, nt, kv, c.q_lora + HD, nullptr, ss);
        }
        if (&d != &td) gemv_scratch_slot(0);
        Fin7 f; f.w = B.kv_norm; f.eps = c.eps; f.rd = c.rd; f.freqs = d.freqs_win; f.pos = pos; f.quant = 1;
        f.dst = B.ring; f.dst_stride = HD; f.ring = ring_;
        fin7(kv + c.q_lora, nt, HD, c.q_lora + HD, f, ss);
    }
    CUDA_CHECK(cudaSetDevice(td.id));
}


// a draft round: y (the token at p0, not fed yet) and noise tokens at p0 .. p0 + T - 1 through the three blocks; drafts[i] is
// the token at p0 + 1 + i, conf[i] its confidence
int Engine7::dspark_draft(int y, int64_t p0, std::vector<int> & drafts, std::vector<float> & conf, float temp, uint64_t seed,
                          std::vector<float> * logq) {
    const Ds4Config & c = cfg_;
    const int n = c.n_embd, hcn = c.hc * n, HD = c.head_dim, HQ = c.n_head * HD, T = dsp_T_;
    // the block's input rows (the token embedding as BF16, as the trunk's)
    const GTensor & te = gguf_->need("token_embd.weight");
    const auto * te_tr = ggml_get_type_traits((ggml_type) te.type);
    for (int t = 0; t < T; ++t) {
        float * out = h_dsp_ + (size_t) t * n;
        te_tr->to_float(te.data + (size_t) (t == 0 ? y : dsp_noise_) * te.row_bytes(), out, n);
        for (int e = 0; e < n; ++e) out[e] = bf_round(out[e]);
    }
    for (size_t bi = 0; bi < dblk_.size(); ++bi) {
        DBlock & B = *dblk_[bi];
        Device & d = *devs_[B.dev];
        CUDA_CHECK(cudaSetDevice(d.id));
        cudaStream_t s = d.stream;
        if (bi == 0) {
            CUDA_CHECK(cudaMemcpyAsync(d.x, h_dsp_, (size_t) T * n * sizeof(float), cudaMemcpyHostToDevice, s));
            embed7(d.x, d.h, n, c.hc, T, s);
            init_pre7(d.pre, T, s);
        } else if (dblk_[bi - 1]->dev != B.dev) {
            CUDA_CHECK(cudaMemcpyAsync(d.h, h_dsp_, (size_t) T * hcn * sizeof(float), cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(d.pre, h_dsp_ + (size_t) T * hcn, (size_t) T * 8 * sizeof(float), cudaMemcpyHostToDevice, s));
        }
        auto mm = [&](const DW & W, const float * x, int xs, float * y_, int ys, const NormIn & ni = NormIn{}) {
            gemv_q8(W.q8, x, xs, y_, ys, nullptr, T, s, ni);   // (the drafter's weights: one Q8 matrix each, requant)
        };
        const int wo = c.q_lora + HD;
        // attention
        hc7_mix(d.h, B.hca_fn, hcn, (2 + c.hc) * c.hc, c.eps, d.mix, T, s);
        hc7_split(d.mix, B.hca_scale, B.hca_base, c.hc, c.sinkhorn, c.hc_eps, d.a_pre, d.a_post, d.a_comb, T, s);
        hc7_pre_norm(d.h, d.pre, B.attn_norm, n, c.hc, c.eps, d.x, T, s);
        mm(B.win, d.x, n, d.wout, wo);
        {   Fin7 f; f.w = B.q_norm; f.eps = c.eps; fin7(d.wout, T, c.q_lora, wo, f, s); }
        mm(B.wq_b, d.wout, wo, d.q, HQ);
        {   Fin7 f; f.rd = c.rd; f.freqs = d.freqs_win; f.pos = p0; f.vpp = c.n_head; fin7(d.q, T * c.n_head, HD, HD, f, s); }
        {   Fin7 f; f.w = B.kv_norm; f.eps = c.eps; f.rd = c.rd; f.freqs = d.freqs_win; f.pos = p0; f.quant = 1; f.dst = B.blk; f.dst_stride = HD;
            fin7(d.wout + c.q_lora, T, HD, wo, f, s); }
        draft_attn7(d.q, B.ring, ring_, c.win, B.blk, T, p0, B.sink, c.n_head, HD, 1.0f / std::sqrt((float) HD), d.o, s);
        {   Fin7 f; f.rd = c.rd; f.freqs = d.freqs_win; f.pos = p0; f.vpp = c.n_head; f.inverse = true; f.quant = 1;
            fin7(d.o, T * c.n_head, HD, HD, f, s); }
        const int per = HQ / c.o_groups, ol = c.o_groups * c.o_lora;
        {   NormIn g; g.xg_tiles = c.o_lora / 16; g.xg_stride = per; gemv_q8(B.wo_a_all.q8, d.o, HQ, d.oa, ol, nullptr, T, s, g); }
        mm(B.wo_b, d.oa, ol, d.ao, n);
        hc7_post(d.ao, d.h, d.a_post, d.a_comb, n, c.hc, d.h2, T, s);
        std::swap(d.h, d.h2);
        // MoE (every expert in VRAM)
        hc7_mix(d.h, B.hcf_fn, hcn, (2 + c.hc) * c.hc, c.eps, d.mix, T, s);
        hc7_split(d.mix, B.hcf_scale, B.hcf_base, c.hc, c.sinkhorn, c.hc_eps, d.f_pre, d.f_post, d.f_comb, T, s);
        hc7_pre_norm(d.h, d.a_pre, B.ffn_norm, n, c.hc, c.eps, d.x, T, s);
        gemv_f32(B.router, B.n_exp, n, d.x, n, d.rlog, B.n_exp, T, s);
        route7(d.rlog, B.n_exp, B.bias, B.n_exp, B.k_exp, c.w_scale, d.ids, d.wts, T, s);
        mm(B.sh_gu, d.x, n, d.shgu, 2 * c.ff);
        {   NormIn glu; glu.act = 3; glu.glu_off = c.ff; glu.act_scale = c.clamp_sh; mm(B.sh_down, d.shgu, 2 * c.ff, d.shout, n, glu); }
        moe_gate_up(B.moex, d.x, n, d.ids, B.k_exp, d.hexp, T, s);
        moe_down(B.moex, d.hexp, d.ids, d.wts, B.k_exp, d.yexp, T, s);
        moe_reduce(d.shout, d.ones, d.yexp, B.k_exp, d.ffo, n, T, d.ids, B.owner, d.g, CPU_OWNER, nullptr, nullptr, d.counter, 0, s);
        hc7_post(d.ffo, d.h, d.f_post, d.f_comb, n, c.hc, d.h2, T, s);
        std::swap(d.h, d.h2);
        std::swap(d.pre, d.f_pre);
        if (bi + 1 < dblk_.size() && dblk_[bi + 1]->dev != B.dev) {
            CUDA_CHECK(cudaMemcpyAsync(h_dsp_, d.h, (size_t) T * hcn * sizeof(float), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaMemcpyAsync(h_dsp_ + (size_t) T * hcn, d.pre, (size_t) T * 8 * sizeof(float), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
    }
    // the drafts: collapse, the final norm, the trunk's head + the markov bias (greedy chain), the confidences
    DBlock & B = *dblk_.back();
    Device & d = *devs_[B.dev];
    if (&d != devs_.back().get()) throw std::runtime_error("hyper7: the drafter's last block must sit with the head");
    CUDA_CHECK(cudaSetDevice(d.id));
    cudaStream_t s = d.stream;
    hc7_pre_norm(d.h, d.pre, nullptr, n, c.hc, c.eps, d.ffo, T, s);   // xh (no norm)
    hc7_pre_norm(d.h, d.pre, B.norm, n, c.hc, c.eps, d.x, T, s);      // rms(xh) * norm
    head_mm(d, d.x, T, s);
    for (int i = 0; i < T; ++i) {
        if (temp > 0.0f) {   // sampled drafts: d_i ~ q_i = softmax(row_i / T), the chain on the drawn tokens; log q_i(d_i) for the verification
            float * row = B.dq + (size_t) i * c.n_vocab;
            markov7(d.logits + (size_t) i * c.n_vocab, B.mk_head, B.mk_emb, c.n_vocab, dsp_rank_, i == 0 ? nullptr : B.pairs + 2 * (i - 1), y, row, s);
            max_sumexp(row, c.n_vocab, B.qmz + 2 * i, s, 1.0f / temp);
            gumbel_argmax7(row, c.n_vocab, temp, seed, p0 + 1 + i, (int *) (B.pairs + 2 * i + 1), s);
            qlog7(row, B.qmz + 2 * i, temp, B.pairs + 2 * i, 0, B.qlog + i, s);
            continue;
        }
        markov7(d.logits + (size_t) i * c.n_vocab, B.mk_head, B.mk_emb, c.n_vocab, dsp_rank_, i == 0 ? nullptr : B.pairs + 2 * (i - 1), y, B.mrow, s);
        argmax_pairs(B.mrow, c.n_vocab, c.n_vocab, 0, B.pairs + 2 * i, 1, s);
    }
    dconf7(d.ffo, n, B.mk_emb, dsp_rank_, B.conf_w, B.pairs, y, B.conf, T, s);
    CUDA_CHECK(cudaMemcpyAsync(h_dout_, B.pairs, (size_t) 2 * T * sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(h_dout_ + 16, B.conf, (size_t) T * sizeof(float), cudaMemcpyDeviceToHost, s));
    if (temp > 0.0f) CUDA_CHECK(cudaMemcpyAsync(h_dout_ + 32, B.qlog, (size_t) T * sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    drafts.resize(T);
    conf.resize(T);
    for (int i = 0; i < T; ++i) { int id; memcpy(&id, &h_dout_[2 * i + 1], 4); drafts[i] = id; conf[i] = h_dout_[16 + i]; }
    if (logq) { logq->resize(T); for (int i = 0; i < T; ++i) (*logq)[i] = temp > 0.0f ? h_dout_[32 + i] : 0.0f; }
    return T;
}

// speculative sampling over sampled drafts (full-vocabulary sampling at temp): draft j is kept with probability
// min(1, p_j(d_j) / q_j(d_j)) (p: the target's row off + j); the first one rejected is replaced by a draw from max(0, p - q);
// all kept: the target's row after them is sampled. Returns the token after the kept drafts; m: how many were kept
int Engine7::dspark_verify(int off, const std::vector<int> & dr, const std::vector<float> & logq, const SamplingParams & sp, int64_t p, int & m) {
    const Ds4Config & c = cfg_;
    DBlock & B = *dblk_.back();
    Device & d = *devs_[B.dev];
    CUDA_CHECK(cudaSetDevice(d.id));
    cudaStream_t s = d.stream;
    const int nd = (int) dr.size();
    for (int j = 0; j < nd; ++j) {
        const float * row = d.logits + (size_t) (off + j) * c.n_vocab;
        max_sumexp(row, c.n_vocab, B.tmz + 2 * j, s, 1.0f / sp.temp);
        qlog7(row, B.tmz + 2 * j, sp.temp, nullptr, dr[j], B.plog + j, s);
    }
    float plog[8] = {};
    if (nd) CUDA_CHECK(cudaMemcpyAsync(plog, B.plog, (size_t) nd * sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    const uint64_t useed = samp_seed_ ^ 0xa5a5a5a55a5a5a5aull, rseed = samp_seed_ ^ 0x3c3c3c3cc3c3c3c3ull;
    for (m = 0; m < nd; ++m) {
        const int64_t pos = p + m + 1;
        const uint64_t k = gmix64(useed ^ gmix64((uint64_t) pos));
        const double u = ((double) (k >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        if (u < std::exp((double) plog[m] - (double) logq[m])) continue;
        // rejected: a draw from the residual
        int tok = 0;
        resid7(d.logits + (size_t) (off + m) * c.n_vocab, B.tmz + 2 * m, B.dq + (size_t) m * c.n_vocab, B.qmz + 2 * m, c.n_vocab, sp.temp, rseed,
               pos, B.resid, s);
        CUDA_CHECK(cudaMemcpyAsync(&tok, B.resid, sizeof(int), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        return tok;
    }
    return sample_row(off + nd, sp, p + nd + 1);
}

void Engine7::place_experts() {
    const Ds4Config & c = cfg_;
    const int E = c.n_expert, ff = c.ff, n = c.n_embd, K = c.n_used;
    const size_t R = (size_t) R_;
    // EP prefill: the links' bandwidth (x16 : x8 : x16 on this box, as engine5's streaming shares) and which devices help:
    // those whose own layers' expert bytes per unit of link bandwidth are well below the average (GPU 0 with its 8 layers)
    std::vector<double> wbw = link_weights((int) devs_.size(), "HYPER7_EP_WEIGHTS");
    if (ep_prefill_ && devs_.size() > 1) {
        std::vector<double> load(devs_.size(), 0.0);
        double tot = 0;
        for (auto & dp : devs_)
            for (Layer & L : dp->layers) {
                const std::string p = "blk." + std::to_string(L.il) + ".";
                const double b = (double) (gguf_->need(p + "ffn_gate_exps.weight").nbytes * 2 + gguf_->need(p + "ffn_down_exps.weight").nbytes);
                load[dp->g] += b / wbw[dp->g];
                tot += b / wbw[dp->g];
            }
        const size_t hst = (size_t) (getenv("HYPER7_HSTAGE_MIB") ? atoi(getenv("HYPER7_HSTAGE_MIB")) : 256) << 20;
        // takers: devices whose own streaming (per link weight) is light: HYPER7_EP_TAKER_FRAC of the average (default 0.6),
        // or listed explicitly (HYPER7_EP_TAKERS="0,1,2"); every taker helps every non-taker
        const double tf = getenv("HYPER7_EP_TAKER_FRAC") ? atof(getenv("HYPER7_EP_TAKER_FRAC")) : 0.6;
        for (auto & dp : devs_) dp->taker = load[dp->g] < tf * tot / (double) devs_.size();
        if (const char * tk = getenv("HYPER7_EP_TAKERS")) {
            for (auto & dp : devs_) dp->taker = false;
            for (const char * q = tk; *q;) { const int g = atoi(q); if (g >= 0 && g < (int) devs_.size()) devs_[g]->taker = true; while (*q && *q != ',') ++q; if (*q) ++q; }
        }
        for (auto & dp : devs_) fprintf(stderr, "hyper7: EP prefill: device %d own streaming load %.1f GiB/weight%s\n", dp->id, load[dp->g] / 1073741824.0, dp->taker ? " (taker)" : "");
        for (auto & dp : devs_) {
            Device & d = *dp;
            if (!d.taker) continue;
            CUDA_CHECK(cudaSetDevice(d.id));
            d.hstage_bytes = hst;
            d.zeros = d.alloc<float>(R);
            d.hres.resize(devs_.size());
            for (auto & op : devs_) {   // one set of streams, staging and scratch per owner it helps
                if (op->taker) continue;
                Device::HelperRes & H = d.hres[op->g];
                CUDA_CHECK(cudaStreamCreateWithFlags(&H.hstream, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&H.hcstream, cudaStreamNonBlocking));
                for (auto & ev : H.hev_up) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                for (auto & ev : H.hev_free) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                for (auto & b : H.hstage) b = d.alloc<uint8_t>(hst);
                H.hx = d.alloc<float>(R * n);
                H.hx16 = d.alloc<half>(R * n);
                H.hids = d.alloc<int>(R * K);
                H.hwts = d.alloc<float>(R * K);
                H.horder = d.alloc<int>(R * K);
                H.horder_n = d.alloc<int>(4);
                H.hegrp = d.alloc<int>(3 * E + 16);
                H.hh16 = d.alloc<half>(R * K * ff);
                H.hyexp = d.alloc<float>(R * K * n);
                H.hpart = d.alloc<float>(R * n);
                for (int r = 0; r < EP_RING; ++r) {
                    CUDA_CHECK(cudaHostAlloc(&H.ring_x[r], R * n * sizeof(float), cudaHostAllocPortable));
                    CUDA_CHECK(cudaHostAlloc(&H.ring_p[r], R * n * sizeof(float), cudaHostAllocPortable));
                    CUDA_CHECK(cudaHostAlloc(&H.ring_wts[r], R * K * sizeof(float), cudaHostAllocPortable));
                    CUDA_CHECK(cudaHostAlloc(&H.ring_ids[r], R * K * sizeof(int), cudaHostAllocPortable));
                    for (auto * ev : {&H.ev_xfree[r], &H.ev_p[r]}) CUDA_CHECK(cudaEventCreateWithFlags(ev, cudaEventDisableTiming));
                }
            }
            fprintf(stderr, "hyper7: device %d helps the others' prefill (staging 2 x %zu MiB per owner)\n", d.id, hst >> 20);
        }
        bool any = false;
        for (auto & dp : devs_) any |= dp->taker;
        if (any)
            for (auto & dp : devs_) {
                if (dp->taker) continue;
                CUDA_CHECK(cudaSetDevice(dp->id));
                dp->hp_in = dp->alloc<float>(R * n);
                for (int r = 0; r < EP_RING; ++r)
                    for (auto * ev : {&dp->ev_x[r], &dp->ev_pfree[r]}) CUDA_CHECK(cudaEventCreateWithFlags(ev, cudaEventDisableTiming));
            }
    }
    for (auto & dp : devs_) {
        Device & d = *dp;
        CUDA_CHECK(cudaSetDevice(d.id));
        // prefill staging: two buffers the CPU's experts stream through (HYPER7_STAGE_MIB each)
        d.stage_bytes = (size_t) (getenv("HYPER7_STAGE_MIB") ? atoi(getenv("HYPER7_STAGE_MIB")) : 1024) << 20;
        for (auto & b : d.stage) b = d.alloc<uint8_t>(d.stage_bytes);
        {   std::vector<int> all(E, d.g); d.owner_all = d.upload(all.data(), E); }
        if (host_on_) {   // expert host: decode scratch (every device may host)
            d.qx = d.alloc<float>((size_t) MAX_NT * n); d.qwts = d.alloc<float>((size_t) MAX_NT * K); d.qids = d.alloc<int>((size_t) MAX_NT * K);
            d.qhexp = d.alloc<float>((size_t) MAX_NT * K * ff); d.qyexp = d.alloc<float>((size_t) MAX_NT * K * n);
            d.qout = d.alloc<float>((size_t) MAX_NT * n); d.qzx = d.alloc<float>((size_t) MAX_NT * n); d.qzs = d.alloc<float>(MAX_NT);
            CUDA_CHECK(cudaMemset(d.qzx, 0, (size_t) MAX_NT * n * sizeof(float)));
            CUDA_CHECK(cudaMemset(d.qzs, 0, (size_t) MAX_NT * sizeof(float)));
        }
        size_t fr = 0, tot = 0;
        CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = (size_t) (opt_.vram_reserve_gib * 1073741824.0);
        size_t budget = fr > reserve ? (size_t) ((fr - reserve) * opt_.gpu_expert_frac) : 0;
        if (getenv("HYPER4_NOGPUEXP")) budget = 0;
        // equal expert counts per layer of this device (by bytes: the first and last layers' types are larger)
        size_t per_all = 0;
        for (Layer & L : d.layers) {
            const std::string p = "blk." + std::to_string(L.il) + ".";
            per_all += (gguf_->need(p + "ffn_gate_exps.weight").nbytes * 2 + gguf_->need(p + "ffn_down_exps.weight").nbytes) / E;
        }
        int n_gpu = per_all ? (int) std::min<size_t>(E, budget / per_all) : 0;
        if (const char * fx = getenv("HYPER7_GPU_EXPERTS")) {   // (tests: fixed counts per device, "n0,n1,n2", for identical placements)
            int i = 0, v = n_gpu;
            for (const char * q = fx; *q; ++i) { v = atoi(q); if (i == d.g) n_gpu = std::min(E, v); while (*q && *q != ',') ++q; if (*q) ++q; }
        }
        size_t dev_bytes = 0, host_bytes = 0;
        // expert host: the hottest of this device's CPU experts (count per byte, after its own VRAM experts) go to the pool
        std::vector<int> npool(c.n_layer, 0), phost(c.n_layer, -1);
        const int host_for = getenv("HYPER7_HOST_FOR") ? atoi(getenv("HYPER7_HOST_FOR")) : (int) devs_.size() - 1;
        if (host_on_ && d.g == host_for) {
            // the hosts' budgets: their spare VRAM now (their own experts are placed: devices are processed in order and the
            // target is the last), capped
            std::vector<size_t> remain(devs_.size(), 0);
            for (auto & hp : devs_) {
                if (hp->g == host_for || (host_dev_ >= 0 && hp->g != host_dev_)) continue;
                CUDA_CHECK(cudaSetDevice(hp->id));
                size_t hfr = 0, htot = 0;
                CUDA_CHECK(cudaMemGetInfo(&hfr, &htot));
                const size_t hres = (size_t) (opt_.vram_reserve_gib * 1073741824.0);
                remain[hp->g] = std::min(host_pool_bytes_, hfr > hres ? hfr - hres : 0);
                fprintf(stderr, "hyper7: device %d: %.2f GiB free after its own experts, %.2f GiB reserve -> %.2f GiB for pooled experts\n", hp->id,
                        hfr / 1073741824.0, hres / 1073741824.0, remain[hp->g] / 1073741824.0);
            }
            CUDA_CHECK(cudaSetDevice(d.id));
            struct Cand { double key; int il, r; size_t bytes; };
            std::vector<Cand> cands;
            for (Layer & L : d.layers) {
                const int il = L.il;
                const std::vector<uint64_t> * cnt = place_->stats_of(il);
                if (!cnt) continue;
                const std::string p = "blk." + std::to_string(il) + ".";
                const size_t eb = (gguf_->need(p + "ffn_gate_exps.weight").nbytes * 2 + gguf_->need(p + "ffn_down_exps.weight").nbytes) / E;
                const std::vector<int> order = expert_order(cnt, E);
                for (int r = n_gpu; r < E; ++r) cands.push_back({(double) (*cnt)[order[r]] / (double) eb, il, r, eb});
            }
            std::stable_sort(cands.begin(), cands.end(), [](const Cand & a, const Cand & b) { return a.key > b.key; });
            // a layer's pooled experts sit on one host (the owner's reduce takes one host partial): the host with the most
            // room when the layer's first expert is taken
            std::vector<size_t> used(devs_.size(), 0);
            std::vector<int> nph(devs_.size(), 0);
            for (const Cand & q : cands) {
                if (q.r != n_gpu + npool[q.il]) continue;
                int hh = phost[q.il];
                if (hh < 0) { for (int g = 0; g < (int) devs_.size(); ++g) if (remain[g] >= q.bytes && (hh < 0 || remain[g] > remain[hh])) hh = g; if (hh < 0) continue; }
                if (remain[hh] < q.bytes) continue;
                phost[q.il] = hh; remain[hh] -= q.bytes; used[hh] += q.bytes; ++npool[q.il]; ++nph[hh];
            }
            for (int g = 0; g < (int) devs_.size(); ++g)
                if (nph[g]) fprintf(stderr, "hyper7: device %d hosts %d experts of device %d's layers (%.1f GiB)\n", devs_[g]->id, nph[g], d.id, used[g] / 1073741824.0);
        }
        for (Layer & L : d.layers) {
            const int il = L.il;
            const ExpertTensors xt = expert_tensors(*gguf_, il, E);
            const std::vector<int> order = expert_order(place_->stats_of(il), E);
            // owners: the hottest n_gpu here, the next npool on the host device, the rest on the CPU; the compact host copy
            // holds the CPU's and the pooled ones (the bulk path computes the pooled experts on the CPU); pooled experts
            // never swap
            std::vector<int> owner(E, CPU_OWNER), owner_bulk(E, CPU_OWNER);
            std::vector<char> pooled(E, 0), on_devs(devs_.size(), 0);
            for (int r = 0; r < n_gpu; ++r) owner[order[r]] = owner_bulk[order[r]] = d.g;
            for (int r = 0; r < npool[il]; ++r) { owner[order[n_gpu + r]] = phost[il]; pooled[order[n_gpu + r]] = 1; }
            on_devs[d.g] = 1;
            if (npool[il] > 0) on_devs[phost[il]] = 1;
            place_->load_layer(il, xt, owner, npool[il] > 0 ? &pooled : nullptr, npool[il] > 0 ? &pooled : nullptr,
                               npool[il] > 0 ? &owner_bulk : nullptr, &on_devs);
            L.n_gpu = n_gpu;
            L.gb = xt.gb; L.db = xt.db;
            L.n_own = E - n_gpu;   // (the compact copy: CPU + pooled experts; the shares, if any, are cut below)
            if (npool[il] > 0) { L.host_dev = phost[il]; L.n_host = npool[il]; host_layers_[phost[il]].push_back(il); }
            dev_bytes += (size_t) n_gpu * (2 * xt.gb + xt.db);
            host_bytes += place_->host(il).hx.bytes;
            CUDA_CHECK(cudaSetDevice(d.id));
        }
        fprintf(stderr, "hyper7: device %d: %d of %d experts per layer in VRAM (%.1f GiB), %.1f GiB on the CPU\n", d.id, n_gpu, E,
                dev_bytes / 1073741824.0, host_bytes / 1073741824.0);
    }
    // EP prefill shares: every device's streamed bytes per chunk in proportion to its link (givers hand their excess to the
    // takers in proportion to the takers' deficits); then the part tables of every layer
    {
        const int nd = (int) devs_.size();
        std::vector<double> B(nd, 0.0);
        double tot = 0, wsum = 0;
        for (auto & dp : devs_) { for (Layer & L : dp->layers) B[dp->g] += (double) (L.n_own) * (2 * L.gb + L.db); tot += B[dp->g]; wsum += wbw[dp->g]; }
        std::vector<double> T(nd), deficit(nd, 0.0), excess(nd, 0.0);
        double D = 0, X = 0;
        for (int g = 0; g < nd; ++g) {
            T[g] = tot * wbw[g] / wsum;
            if (devs_[g]->taker && B[g] < T[g]) { deficit[g] = T[g] - B[g]; D += deficit[g]; }
            if (!devs_[g]->taker && B[g] > T[g]) { excess[g] = B[g] - T[g]; X += excess[g]; }
        }
        const double scale = X > 0 ? std::min(1.0, D / X) : 0.0;   // (the takers may not absorb every excess)
        for (auto & dp : devs_) {
            Device & d = *dp;
            const double give = excess[d.g] * scale;
            for (Layer & L : d.layers) {
                if (L.n_own == 0 || B[d.g] <= 0) { rebuild_part_slots(L); continue; }
                const int ncpu = L.n_own;
                const int ngive = std::min(ncpu - 1, (int) (ncpu * give / B[d.g]));
                int c0 = ncpu - ngive;
                L.n_own = c0;
                for (int h = 0; h < nd && ngive > 0; ++h) {
                    if (deficit[h] <= 0) continue;
                    const int cnt = (int) std::lround(ngive * deficit[h] / D);
                    const int c1 = std::min(ncpu, c0 + cnt);
                    if (c1 > c0) { Share sh; sh.dev = h; sh.c0 = c0; sh.c1 = c1; L.shares.push_back(sh); c0 = c1; }
                }
                if (c0 < ncpu) {   // (rounding: the rest to the last share, or back to the owner)
                    if (!L.shares.empty()) L.shares.back().c1 = ncpu; else L.n_own = ncpu;
                }
                if (ep_debug_ == 1) L.n_own = ncpu;   // (the owner streams every expert; the helpers' shares overlap)
                rebuild_part_slots(L);
            }
        }
        if (ep_prefill_)
            for (auto & dp : devs_) {
                double own = 0, helped = 0;
                for (Layer & L : dp->layers) own += (double) L.n_own * (2 * L.gb + L.db);
                for (auto & op : devs_) for (Layer & L : op->layers) for (Share & sh : L.shares) if (sh.dev == dp->g) helped += (double) (sh.c1 - sh.c0) * (2 * L.gb + L.db);
                fprintf(stderr, "hyper7: EP prefill: device %d streams %.1f GiB of its own CPU experts + %.1f GiB for the others per chunk\n",
                        dp->id, own / 1073741824.0, helped / 1073741824.0);
            }
    }
}

// the part tables of a layer's CPU experts (the owner's own share and the helpers' shares) from the compact indices:
// part j holds compact indices [part_i0[j], part_i0[j + 1]), its slot table maps an expert to its index in the part; the
// owner tables mark the experts a device sums in its moe_reduce (its share + the experts in its VRAM for the owner)
void Engine7::rebuild_part_slots(Layer & L) {
    const Ds4Config & c = cfg_;
    const int E = c.n_expert, il = L.il;
    Device & d = *devs_[dev_of_[il]];
    const std::vector<int> & cslot = place_->host(il).cslot;
    auto build = [&](Device & dev, int c0, int c1, size_t stage_bytes, std::vector<int> & part_i0, std::vector<int *> & part_slot) {
        std::vector<int> i0s;
        if (c1 > c0 && 2 * L.gb + L.db > 0) {
            const int per = std::max<int>(1, (int) (stage_bytes / (2 * L.gb + L.db)));
            for (int i0 = c0; i0 < c1; i0 += per) i0s.push_back(i0);
            i0s.push_back(c1);
        }
        CUDA_CHECK(cudaSetDevice(dev.id));
        const int np = i0s.empty() ? 0 : (int) i0s.size() - 1;
        while ((int) part_slot.size() < np) part_slot.push_back(dev.alloc<int>(E));
        part_i0 = i0s;
        for (int j = 0; j < np; ++j) {
            std::vector<int> sl(E, -1);
            for (int e = 0; e < E; ++e)
                if (cslot[e] >= part_i0[j] && cslot[e] < part_i0[j + 1]) sl[e] = cslot[e] - part_i0[j];
            CUDA_CHECK(cudaMemcpy(part_slot[j], sl.data(), E * sizeof(int), cudaMemcpyHostToDevice));
        }
    };
    build(d, 0, L.n_own, d.stage_bytes, L.part_i0, L.part_slot);
    if (!L.shares.empty()) {
        std::vector<int> own(E, -1);
        for (int e = 0; e < E; ++e) if (cslot[e] < 0 || cslot[e] < L.n_own) own[e] = d.g;
        CUDA_CHECK(cudaSetDevice(d.id));
        if (!L.owner_tab) L.owner_tab = d.alloc<int>(E);
        CUDA_CHECK(cudaMemcpy(L.owner_tab, own.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    }
    for (Share & sh : L.shares) {
        Device & h = *devs_[sh.dev];
        build(h, sh.c0, sh.c1, h.hstage_bytes, sh.part_i0, sh.part_slot);
        std::vector<int> tab(E, -1);
        for (int e = 0; e < E; ++e) if (cslot[e] >= sh.c0 && cslot[e] < sh.c1) tab[e] = h.g;
        CUDA_CHECK(cudaSetDevice(h.id));
        if (!sh.owner_tab) sh.owner_tab = h.alloc<int>(E);
        CUDA_CHECK(cudaMemcpy(sh.owner_tab, tab.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    }
}

// a helper's work for one streamed layer: the owner has put the layer's MoE input rows and routing into ring slot r
// (ev_x[r]); the helper streams its share of the experts through its own staging and returns the partial sum (ev_p[r])
void Engine7::issue_helper(Layer & L, const Share & sh, int nt, int r) {
    const Ds4Config & c = cfg_;
    const int E = c.n_expert, n = c.n_embd, K = c.n_used;
    Device & h = *devs_[sh.dev], & d = *devs_[dev_of_[L.il]];
    const MoeLayerHost & HL = place_->host(L.il);
    const MoeLayerDev & X = place_->dev(d.g, L.il);
    CUDA_CHECK(cudaSetDevice(h.id));
    Device::HelperRes & H = h.hres[d.g];
    cudaStream_t hs = H.hstream;
    CUDA_CHECK(cudaStreamWaitEvent(hs, d.ev_x[r], 0));
    CUDA_CHECK(cudaMemcpyAsync(H.hx, H.ring_x[r], (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, hs));
    CUDA_CHECK(cudaMemcpyAsync(H.hids, H.ring_ids[r], (size_t) nt * K * sizeof(int), cudaMemcpyHostToDevice, hs));
    CUDA_CHECK(cudaMemcpyAsync(H.hwts, H.ring_wts[r], (size_t) nt * K * sizeof(float), cudaMemcpyHostToDevice, hs));
    CUDA_CHECK(cudaEventRecord(H.ev_xfree[r], hs));
    to_half(H.hx, n, nullptr, n, 0.0f, H.hx16, nt, hs);
    if (ep_debug_ == 8) CUDA_CHECK(cudaMemsetAsync(H.hyexp, 0, (size_t) R_ * K * n * sizeof(float), hs));   // (test: no stale pair outputs)
    auto stage_copy = [&](int j) {
        const int b = j & 1, i0 = sh.part_i0[j], cnt = sh.part_i0[j + 1] - i0;
        CUDA_CHECK(cudaStreamWaitEvent(H.hcstream, H.hev_free[b], 0));
        uint8_t * st = H.hstage[b];
        CUDA_CHECK(cudaMemcpyAsync(st, HL.hx.gate + (size_t) i0 * L.gb, (size_t) cnt * L.gb, cudaMemcpyHostToDevice, H.hcstream));
        CUDA_CHECK(cudaMemcpyAsync(st + (size_t) cnt * L.gb, HL.hx.up + (size_t) i0 * L.gb, (size_t) cnt * L.gb, cudaMemcpyHostToDevice, H.hcstream));
        CUDA_CHECK(cudaMemcpyAsync(st + (size_t) 2 * cnt * L.gb, HL.hx.down + (size_t) i0 * L.db, (size_t) cnt * L.db, cudaMemcpyHostToDevice, H.hcstream));
        CUDA_CHECK(cudaEventRecord(H.hev_up[b], H.hcstream));
    };
    for (int j = 0; j < std::min(2, sh.np()); ++j) stage_copy(j);
    for (int j = 0; j < sh.np(); ++j) {
        const int b = j & 1, cnt = sh.part_i0[j + 1] - sh.part_i0[j];
        MoeDev ms = X.moe;
        ms.gate = H.hstage[b];
        ms.up = H.hstage[b] + (size_t) cnt * L.gb;
        ms.down = H.hstage[b] + (size_t) 2 * cnt * L.gb;
        ms.slot = sh.part_slot[j];
        CUDA_CHECK(cudaStreamWaitEvent(hs, H.hev_up[b], 0));
        moe_order(ms, H.hids, nt * K, E, H.horder, H.horder_n, hs, H.hegrp);
        if (ep_debug_ >= 7) {   // (tests) the pairs the part selected vs the routing's pairs of the share (from the ring's ids)
            CUDA_CHECK(cudaStreamSynchronize(hs));
            int on[2] = {0, 0};
            CUDA_CHECK(cudaMemcpy(on, H.horder_n, 2 * sizeof(int), cudaMemcpyDeviceToHost));
            const std::vector<int> & cs = HL.cslot;
            int want = 0, exp_in = 0;
            for (int p = 0; p < nt * K; ++p) { const int e = H.ring_ids[r][p]; if (cs[e] >= sh.part_i0[j] && cs[e] < sh.part_i0[j + 1]) ++want; }
            for (int e = 0; e < E; ++e) if (cs[e] >= sh.part_i0[j] && cs[e] < sh.part_i0[j + 1]) ++exp_in;
            fprintf(stderr, "hyper7: EP layer %d part %d [%d,%d) experts %d: pairs selected %d, routing's pairs in the part %d, groups %d\n",
                    L.il, j, sh.part_i0[j], sh.part_i0[j + 1], exp_in, on[0], want, on[1]);
        }
        moe_gemm_gate_up(ms, H.hx16, n, K, H.horder, H.horder_n, H.hegrp, E, H.hh16, hs);
        moe_gemm_down(ms, H.hh16, H.horder, H.horder_n, H.hegrp, E, H.hwts, H.hyexp, hs);
        CUDA_CHECK(cudaEventRecord(H.hev_free[b], hs));
        if (j + 2 < sh.np()) stage_copy(j + 2);
    }
    // the partial: the share's pairs only (owner_tab), no shared expert (sg = 0)
    moe_reduce(H.hx, h.zeros, H.hyexp, K, H.hpart, n, nt, H.hids, sh.owner_tab, h.g, CPU_OWNER, nullptr, nullptr, h.counter, (unsigned) L.il, hs);
    CUDA_CHECK(cudaStreamWaitEvent(hs, d.ev_pfree[r], 0));   // (the slot's previous partial taken by the owner)
    CUDA_CHECK(cudaMemcpyAsync(H.ring_p[r], H.hpart, (size_t) nt * n * sizeof(float), cudaMemcpyDeviceToHost, hs));
    CUDA_CHECK(cudaEventRecord(H.ev_p[r], hs));
}

Engine7::~Engine7() {
    if (!issuers_.empty()) {
        { std::lock_guard<std::mutex> lk(imx_); stop_issuers_ = true; }
        icv_.notify_all();
        for (auto & I : issuers_) if (I->th.joinable()) I->th.join();
    }
    release_cpu_side();   // (the CPU side and the host copies first)
    for (auto & [p, sz] : host_bufs_) { cudaHostUnregister(p); munmap(p, sz); }
    devs_.clear();
    for (int k = 0; k < NSET; ++k) {
        for (float * p : h_eg_[k]) cudaFreeHost(p);
        cudaFreeHost(h_embd_[k]);
        cudaFreeHost(h_ids_[k]);
        if (ev_chunk_[k]) cudaEventDestroy(ev_chunk_[k]);
    }
    if (dsp_map_) munmap(dsp_map_, dsp_size_);
    for (void * p : {(void *) h_out_, (void *) h_dsp_, (void *) h_dout_, (void *) host_out_})
        if (p) cudaFreeHost(p);
}

void Engine7::reset() { hist_.clear(); }

void Engine7::run_layer(Device & d, Layer & L, int nt, int64_t p0) {
    const Ds4Config & c = cfg_;
    cudaStream_t s = d.stream;
    const int n = c.n_embd, hcn = c.hc * n, HQ = c.n_head * c.head_dim, HD = c.head_dim, K = c.n_used, il = L.il;
    const float eps = c.eps;
    const bool bulk = nt > MAX_NT;
    const DenseRun dr{s, d.blas, d.xh, d.ggs, d.ggs_elems};
    auto mm = [&](const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni = NormIn{}) {
        dense_mm(dr, W, x, xs, y, ys, rows, ni);
    };
    auto mf32 = [&](const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nr) {   // y = x W^T (fp32)
        if (nr <= MAX_NT) { gemv_f32(W, rows, k, x, xs, y, ys, nr, s); return; }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, nr, k, &one, W, k, x, xs, &zero, y, ys) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("hyper7: cublasSgemm failed");
    };
    const float * freqs = L.ratio ? d.freqs_comp : d.freqs_win;
    if (!dblk_.empty() && il >= dsp_tap0_ && il < dsp_tap0_ + 3)   // DSpark taps: the streams' mean entering layers 37..39
        hc_mean7(d.h, c.n_embd, c.hc, d.taps + (size_t) (il - dsp_tap0_) * c.n_embd, 3 * c.n_embd, nt, d.stream);
    // prefill: the CPU's experts go to the GPU in parts through the two staging buffers (the first two copies overlap the
    // attention; part j + 2's copy is issued once part j's experts are done)
    const bool stream_exp = bulk && stream_ && nt >= stream_min_ && (L.np() > 0 || !L.shares.empty());
    const MoeLayerHost & HL = place_->host(il);   // the layer's compact host copy
    const MoeLayerDev & X = place_->dev(d.g, il);  // its experts on this device
    auto stage_copy = [&](int j) {
        const int b = j & 1, i0 = L.part_i0[j], cnt = L.part_i0[j + 1] - i0;
        CUDA_CHECK(cudaStreamWaitEvent(d.cstream, d.ev_free[b], 0));
        uint8_t * st = d.stage[b];
        CUDA_CHECK(cudaMemcpyAsync(st, HL.hx.gate + (size_t) i0 * L.gb, (size_t) cnt * L.gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(st + (size_t) cnt * L.gb, HL.hx.up + (size_t) i0 * L.gb, (size_t) cnt * L.gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(st + (size_t) 2 * cnt * L.gb, HL.hx.down + (size_t) i0 * L.db, (size_t) cnt * L.db, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaEventRecord(d.ev_up[b], d.cstream));
    };
    if (stream_exp && d.ev_plast) CUDA_CHECK(cudaStreamWaitEvent(d.cstream, d.ev_plast, 0));
    if (stream_exp) for (int j = 0; j < std::min(2, L.np()); ++j) stage_copy(j);
    // ---- Engram ----
    if (L.eg >= 0) {
        const int ecols = c.eg_cols() * c.eg_hd;
        CUDA_CHECK(cudaMemcpyAsync(d.eg_x, h_eg_[d.cur_set][L.eg], (size_t) nt * ecols * sizeof(float), cudaMemcpyHostToDevice, s));
        mm(L.eg_kv, d.eg_x, ecols, d.eg_kvo, (c.hc + 1) * n, nt);
        engram_gate7(d.h, d.eg_kvo, L.eg_w, c.hc, n, eps, nt, s);
    }
    // HYPER7_PRED=2: this layer's MoE routing predicted from its input (the attention's contribution left out) on the side
    // stream, published for the CPU's prefetch while the attention runs; the prediction reads d.h, which the MoE's output
    // overwrites at the end of the layer (it waits for ev_pf1 there)
    const bool pred2 = pred_mode_ == 2 && !bulk && !stream_exp && nt <= MAX_NT && L.n_gpu < c.n_expert;
    if (pred2) {
        cudaStream_t ps = d.pstream;
        CUDA_CHECK(cudaEventRecord(d.ev_pf0, s));
        CUDA_CHECK(cudaStreamWaitEvent(ps, d.ev_pf0, 0));
        hc7_mix(d.h, L.hcf_fn, hcn, (2 + c.hc) * c.hc, eps, d.pmix, nt, ps);
        hc7_split(d.pmix, L.hcf_scale, L.hcf_base, c.hc, c.sinkhorn, c.hc_eps, d.ppre, d.ppost, d.pcomb, nt, ps);
        hc7_pre_norm(d.h, d.ppre, L.ffn_norm, n, c.hc, eps, d.px, nt, ps);
        gemv_f32(L.router, c.n_expert, n, d.px, n, d.prlog, c.n_expert, nt, ps);
        // (HYPER7_PRED_K: more candidates than the routing takes, best first: the prefetch reads the ones it has time for)
        const int pk = pred_k_ > 0 ? std::min(pred_k_, MOE_MAX_USED) : K;
        route7(d.prlog, c.n_expert, L.bias, c.n_expert, pk, c.w_scale, d.pids, d.pwts, nt, ps);
        moe_publish(&cpu_pred_[il].seq, &cpu_pred_[il].nt, &cpu_pred_[il].ids[0][0], &cpu_pred_[il].wts[0][0], &cpu_pred_[il].x[0][0],
                    d.px, n, 0, d.pids, d.pwts, pk, nt, d.counter, (unsigned) il, ps);
        CUDA_CHECK(cudaEventRecord(d.ev_pf1, ps));
    }
    // ---- attention ----
    hc7_mix(d.h, L.hca_fn, hcn, (2 + c.hc) * c.hc, eps, d.mix, nt, s);
    hc7_split(d.mix, L.hca_scale, L.hca_base, c.hc, c.sinkhorn, c.hc_eps, d.a_pre, d.a_post, d.a_comb, nt, s);
    hc7_pre_norm(d.h, d.pre, L.attn_norm, n, c.hc, eps, d.x, nt, s);
    mm(L.win, d.x, n, d.wout, L.w_out, nt);
    {   Fin7 f; f.w = L.q_norm; f.eps = eps; fin7(d.wout, nt, c.q_lora, L.w_out, f, s); }
    mm(L.wq_b, d.wout, L.w_out, d.q, HQ, nt);
    {   Fin7 f; f.rd = c.rd; f.freqs = freqs; f.pos = p0; f.vpp = c.n_head; fin7(d.q, nt * c.n_head, HD, HD, f, s); }
    {   Fin7 f; f.w = L.kv_norm; f.eps = eps; f.rd = c.rd; f.freqs = freqs; f.pos = p0; f.quant = 1; f.dst = L.ring; f.dst_stride = HD;
        f.ring = ring_;
        fin7(d.wout + L.off_kv, nt, HD, L.w_out, f, s); }
    const uint16_t * ckv = nullptr;
    const int R = L.ratio;
    if (R > 0) {
        Layer & S = devs_[dev_of_[c.kv_of[il]]]->layers[c.kv_of[il] - d.l0];
        const int64_t g_first = p0 / R, g_end = (p0 + nt) / R;
        const int ng = (int) std::max<int64_t>(0, g_end - g_first);
        const float * lat = d.latent;
        long long lat_s = HD;
        if (L.kv_src) {
            if (R == 1) {
                Fin7 f; f.w = L.c_norm; f.eps = eps;
                fin7(d.wout + L.off_ckv, nt, HD, L.w_out, f, s);
                lat = d.wout + L.off_ckv;
                lat_s = L.w_out;
            } else {
                compress7(d.wout + L.off_ckv, L.w_out, d.wout + L.off_cg, L.w_out, L.c_kv, L.c_score, R, p0, nt, HD, L.c_norm, eps, d.latent, s);
            }
        }
        if (L.ix_src) {
            Layer & IK = S;   // the index keys of this layer's KV source
            if (L.kv_src && ng > 0) {
                mf32(L.ik_w, c.ix_hd, HD, lat, (int) lat_s, d.ikr, c.ix_hd, ng);
                Fin7 f; f.w = L.ik_norm; f.eps = eps; f.rd = c.rd; f.freqs = freqs; f.pos = g_first * R; f.pos_step = R; f.quant = 3;
                f.dst = L.ik + (size_t) g_first * c.ix_hd; f.dst_stride = c.ix_hd;
                fin7(d.ikr, ng, c.ix_hd, c.ix_hd, f, s);
            }
            const int64_t clen_max = (p0 + nt) / R;
            if (clen_max > 0) {
                mm(L.iq_b, d.wout, L.w_out, d.iq, c.ix_heads * c.ix_hd, nt);
                {   Fin7 f; f.rd = c.rd; f.freqs = freqs; f.pos = p0; f.vpp = c.ix_heads; f.quant = 3;
                    fin7(d.iq, nt * c.ix_heads, c.ix_hd, c.ix_hd, f, s); }
                mf32(L.iproj, c.ix_heads, n, d.x, n, d.iw, 64, nt);
                for (int a = 0; a < nt; a += SUB) {
                    const int m = std::min(SUB, nt - a);
                    const int64_t pa = p0 + a;
                    const int64_t nmax = (pa + m) / R;   // the sub-batch's widest row
                    if (nmax <= 0) continue;
                    const float * qa = d.iq + (size_t) a * c.ix_heads * c.ix_hd, * wa = d.iw + (size_t) a * 64;
                    int * sa = d.sel + (size_t) a * c.ix_topk;
                    int * ca = d.cand + (size_t) a * c.cand_topk, * na = d.ncand + a;
                    const bool restrict_ = L.uses_cand && (nmax + c.cand_blk - 1) / c.cand_blk > c.cand_topk;
                    if (restrict_) {
                        const int width = c.cand_topk * c.cand_blk;
                        index_score7(qa, wa, IK.ik, c.ix_heads, c.ix_hd, m, pa, R, width, ca, na, c.cand_topk, c.cand_blk, d.scores, d.score_w, s);
                        topk7(d.scores, d.score_w, m, pa, R, c.ix_topk, ca, na, c.cand_topk, c.cand_blk, sa, c.ix_topk, s);
                    } else {
                        index_score7(qa, wa, IK.ik, c.ix_heads, c.ix_hd, m, pa, R, (int) nmax, nullptr, nullptr, 0, 1, d.scores, d.score_w, s);
                        if (L.cand_src)
                            cand_blocks7(d.scores, d.score_w, m, pa, R, c.cand_blk, c.cand_topk, d.bm, d.bm_w, ca, na, c.cand_topk, s);
                        topk7(d.scores, d.score_w, m, pa, R, c.ix_topk, nullptr, nullptr, 0, 1, sa, c.ix_topk, s);
                    }
                }
            }
        }
        if (L.kv_src && ng > 0) {
            Fin7 f; f.rd = c.rd; f.freqs = freqs; f.pos = g_first * R; f.pos_step = R; f.quant = 2;
            f.dst = L.ckv + (size_t) g_first * HD; f.dst_stride = HD;
            fin7(const_cast<float *>(lat), ng, HD, lat_s, f, s);
        }
        ckv = S.ckv;
    }
    sparse_attn7(d.q, L.ring, ring_, c.win, ckv, d.sel, c.ix_topk, p0, R, c.ix_topk, L.sink, c.n_head, HD, 1.0f / std::sqrt((float) HD),
                 d.o, nt, s);
    {   Fin7 f; f.rd = c.rd; f.freqs = freqs; f.pos = p0; f.vpp = c.n_head; f.inverse = true; f.quant = 1;
        fin7(d.o, nt * c.n_head, HD, HD, f, s); }
    const int per = HQ / c.o_groups, ol = c.o_groups * c.o_lora;
    static const bool wo_grp = !getenv("HYPER7_WO_GROUPED") || atoi(getenv("HYPER7_WO_GROUPED")) != 0;
    if (wo_grp && nt <= MAX_NT && L.wo_a_all.n() > 0) {
        NormIn g; g.xg_tiles = c.o_lora / 16; g.xg_stride = per;
        mm(L.wo_a_all, d.o, HQ, d.oa, ol, nt, g);
    } else
        for (int gi = 0; gi < c.o_groups; ++gi) mm(L.wo_a[gi], d.o + (size_t) gi * per, HQ, d.oa + (size_t) gi * c.o_lora, ol, nt);
    mm(L.wo_b, d.oa, ol, d.ao, n, nt);
    hc7_post(d.ao, d.h, d.a_post, d.a_comb, n, c.hc, d.h2, nt, s);
    std::swap(d.h, d.h2);
    // ---- MoE ----
    hc7_mix(d.h, L.hcf_fn, hcn, (2 + c.hc) * c.hc, eps, d.mix, nt, s);
    hc7_split(d.mix, L.hcf_scale, L.hcf_base, c.hc, c.sinkhorn, c.hc_eps, d.f_pre, d.f_post, d.f_comb, nt, s);
    hc7_pre_norm(d.h, d.a_pre, L.ffn_norm, n, c.hc, eps, d.x, nt, s);
    mf32(L.router, c.n_expert, n, d.x, n, d.rlog, c.n_expert, nt);
    route7(d.rlog, c.n_expert, L.bias, c.n_expert, K, c.w_scale, d.ids, d.wts, nt, s);
    // HYPER7_PRED_EARLY: the next layer's experts from this layer's MoE input on the side stream, published as the early
    // prediction (the CPU reads them after this layer's job until the next layer's own prediction arrives); joined below
    const bool pred1 = pred_mode_ == 2 && pred_early_ && L.pr_router && !bulk && !stream_exp && nt <= MAX_NT && il + 1 < c.n_layer;
    if (pred1) {
        cudaStream_t ps = d.pstream;
        CUDA_CHECK(cudaEventRecord(d.ev_pe0, s));
        CUDA_CHECK(cudaStreamWaitEvent(ps, d.ev_pe0, 0));
        gemv_f32(L.pr_router, c.n_expert, n, d.x, n, d.prlog, c.n_expert, nt, ps);
        route7(d.prlog, c.n_expert, L.pr_bias, c.n_expert, K, c.w_scale, d.pids, d.pwts, nt, ps);
        moe_publish(&cpu_pred1_[il + 1].seq, &cpu_pred1_[il + 1].nt, &cpu_pred1_[il + 1].ids[0][0], &cpu_pred1_[il + 1].wts[0][0],
                    &cpu_pred1_[il + 1].x[0][0], d.x, n, 0, d.pids, d.pwts, K, nt, d.counter, (unsigned) (il + 1), ps);
        CUDA_CHECK(cudaEventRecord(d.ev_pe1, ps));
    }
    std::vector<int> share_slot;   // EP prefill: the helpers' ring slots of this layer
    if (stream_exp && !L.shares.empty()) {
        // the MoE input rows and routing to the helpers through their pinned rings (slot r: after the helper has taken the
        // previous rows of that slot), then their work on their own streams
        for (const Share & sh : L.shares) {
            Device & h = *devs_[sh.dev];
            Device::HelperRes & H = h.hres[d.g];
            const int r = H.ring_next % ep_ring_;
            H.ring_next = (H.ring_next + 1) % ep_ring_;
            share_slot.push_back(r);
            CUDA_CHECK(cudaStreamWaitEvent(s, H.ev_xfree[r], 0));
            CUDA_CHECK(cudaMemcpyAsync(H.ring_x[r], d.x, (size_t) nt * n * sizeof(float), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaMemcpyAsync(H.ring_ids[r], d.ids, (size_t) nt * K * sizeof(int), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaMemcpyAsync(H.ring_wts[r], d.wts, (size_t) nt * K * sizeof(float), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaEventRecord(d.ev_x[r], s));
            // (issued from this thread, in order: the ring's reuse waits then always follow the previous use's records)
            issue_helper(L, sh, nt, r);
            CUDA_CHECK(cudaSetDevice(d.id));
        }
    }
    if (stream_exp) {}
    else if (bulk) moe_publish(&cpu_bulk_->seq, &cpu_bulk_->nt, &cpu_bulk_->ids[0][0], &cpu_bulk_->wts[0][0], &cpu_bulk_->x[0][0], d.x, n, n,
                          d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
    else moe_publish(&cpu_rec_[il].seq, &cpu_rec_[il].nt, &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0], d.x, n, n,
                     d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
    mm(L.sh_gu, d.x, n, d.shgu, 2 * c.ff, nt);
    if (!bulk) {
        NormIn glu; glu.act = 3; glu.glu_off = c.ff; glu.act_scale = c.clamp_sh;
        mm(L.sh_down, d.shgu, 2 * c.ff, d.shout, n, nt, glu);
    } else {
        swiglu7(d.shgu, 2 * c.ff, d.shh, c.ff, c.ff, c.clamp_sh, nt, s);
        mm(L.sh_down, d.shh, c.ff, d.shout, n, nt);
    }
    if (L.n_gpu > 0) {
        if (bulk) {
            moe_order(X.moe, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
            to_half(d.x, n, nullptr, n, 0.0f, d.x16, nt, s);
            moe_gemm_gate_up(X.moe, d.x16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
            moe_gemm_down(X.moe, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
        } else {
            moe_gate_up(X.moe, d.x, n, d.ids, K, d.hexp, nt, s);
            moe_down(X.moe, d.hexp, d.ids, d.wts, K, d.yexp, nt, s);
        }
    }
    if (stream_exp) {   // (routing statistics: the CPU counts only the records it gets)
        CUDA_CHECK(cudaMemcpyAsync(h_ids_[d.cur_set] + (size_t) il * R_ * K, d.ids, (size_t) nt * K * sizeof(int), cudaMemcpyDeviceToHost, s));
        if (L.n_gpu == 0) to_half(d.x, n, nullptr, n, 0.0f, d.x16, nt, s);
        for (int j = 0; j < L.np(); ++j) {
            const int b = j & 1, cnt = L.part_i0[j + 1] - L.part_i0[j];
            MoeDev ms = X.moe;
            ms.gate = d.stage[b];
            ms.up = d.stage[b] + (size_t) cnt * L.gb;
            ms.down = d.stage[b] + (size_t) 2 * cnt * L.gb;
            ms.slot = L.part_slot[j];
            CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_up[b], 0));
            moe_order(ms, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
            moe_gemm_gate_up(ms, d.x16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
            moe_gemm_down(ms, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
            CUDA_CHECK(cudaEventRecord(d.ev_free[b], s));
            if (j + 2 < L.np()) stage_copy(j + 2);
        }
    }
    if (pred_mode_ == 1 && L.pr_router && !bulk && !stream_exp && nt <= MAX_NT && il + 1 < c.n_layer) {
        // the next layer's experts from this layer's MoE input, for the CPU's prefetch while the GPUs run the next attention
        mf32(L.pr_router, c.n_expert, n, d.x, n, d.rlog, c.n_expert, nt);
        route7(d.rlog, c.n_expert, L.pr_bias, c.n_expert, K, c.w_scale, d.pids, d.pwts, nt, s);
        moe_publish(&cpu_pred_[il + 1].seq, &cpu_pred_[il + 1].nt, &cpu_pred_[il + 1].ids[0][0], &cpu_pred_[il + 1].wts[0][0],
                    &cpu_pred_[il + 1].x[0][0], d.x, n, 0, d.pids, d.pwts, K, nt, d.counter, (unsigned) (il + 1), s);
    }
    if (stream_exp) {
        moe_reduce(d.shout, d.ones, d.yexp, K, d.ffo, n, nt, d.ids, L.shares.empty() || ep_debug_ == 1 ? d.owner_all : L.owner_tab, d.g, CPU_OWNER,
                   nullptr, nullptr, d.counter, (unsigned) il, s);
        for (size_t i = 0; i < share_slot.size(); ++i) {   // EP prefill: + the helpers' partials (through their rings)
            Device & h = *devs_[L.shares[i].dev];
            Device::HelperRes & H = h.hres[d.g];
            const int r = share_slot[i];
            CUDA_CHECK(cudaStreamWaitEvent(s, H.ev_p[r], 0));
            CUDA_CHECK(cudaMemcpyAsync(d.hp_in, H.ring_p[r], (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaEventRecord(d.ev_pfree[r], s));
            d.ev_plast = d.ev_pfree[r];
            if (ep_debug_ == 7 || ep_debug_ == 8) {   // (tests) the partial as received, appended to <HYPER7_DUMP>.p<il>
                CUDA_CHECK(cudaStreamSynchronize(s));
                std::vector<float> pv((size_t) nt * n);
                CUDA_CHECK(cudaMemcpy(pv.data(), d.hp_in, pv.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (const char * dp = getenv("HYPER7_DUMP"))
                    if (FILE * f = fopen((std::string(dp) + ".p" + std::to_string(il)).c_str(), "ab")) { fwrite(pv.data(), 4, pv.size(), f); fclose(f); }
            }
            if (ep_debug_ != 1 && ep_debug_ != 5) vec_add(d.ffo, d.hp_in, (size_t) nt * n, s);   // (5: the partial dropped, the owner's share only)
        }
    } else {
        const bool hp = !bulk && L.host_dev >= 0;   // (the expert host's partial, decode only)
        moe_reduce(d.shout, d.ones, d.yexp, K, d.ffo, n, nt, d.ids, bulk ? X.owner_bulk : X.owner, d.g, CPU_OWNER,
                   bulk ? &cpu_bulk_out_->seq : &cpu_out_[il].seq, bulk ? &cpu_bulk_out_->y[0][0] : &cpu_out_[il].y[0][0], d.counter, (unsigned) il, s,
                   nullptr, hp ? L.host_dev : -1, hp ? &host_out_[il].seq : nullptr, hp ? &host_out_[il].y[0][0] : nullptr);
    }
    if (pred2) CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_pf1, 0));
    if (pred1) CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_pe1, 0));
    hc7_post(d.ffo, d.h, d.f_post, d.f_comb, n, c.hc, d.h2, nt, s);
    std::swap(d.h, d.h2);
    std::swap(d.pre, d.f_pre);   // the next block's attention mixes the streams with this block's FFN pre-weights
    if (const char * dp = getenv("HYPER7_DUMP")) {   // (tests) every layer's residual streams [nt][hc][n], appended to <dp>.l<il>
        CUDA_CHECK(cudaStreamSynchronize(s));
        if (ep_debug_) {   // + the FFN output before the post mix and the routing ids (EP prefill tests), <dp>.f<il> / .i<il>
            std::vector<float> fv((size_t) nt * n);
            CUDA_CHECK(cudaMemcpy(fv.data(), d.ffo, fv.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (FILE * f = fopen((std::string(dp) + ".f" + std::to_string(il)).c_str(), "ab")) { fwrite(fv.data(), 4, fv.size(), f); fclose(f); }
            std::vector<int> iv((size_t) nt * K);
            CUDA_CHECK(cudaMemcpy(iv.data(), d.ids, iv.size() * sizeof(int), cudaMemcpyDeviceToHost));
            if (FILE * f = fopen((std::string(dp) + ".i" + std::to_string(il)).c_str(), "ab")) { fwrite(iv.data(), 4, iv.size(), f); fclose(f); }
        }
        std::vector<float> hv((size_t) nt * hcn);
        CUDA_CHECK(cudaMemcpy(hv.data(), d.h, hv.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (FILE * f = fopen((std::string(dp) + ".l" + std::to_string(il)).c_str(), "ab")) { fwrite(hv.data(), 4, hv.size(), f); fclose(f); }
    }
    if (debug_) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<float> hv((size_t) hcn);
        CUDA_CHECK(cudaMemcpy(hv.data(), d.h + (size_t) (nt - 1) * hcn, hcn * sizeof(float), cudaMemcpyDeviceToHost));
        double ss = 0;
        for (float v : hv) ss += (double) v * v;
        fprintf(stderr, "hyper7: layer %d out rms %.5g (last row)\n", il, std::sqrt(ss / hcn));
    }
}

void Engine7::prefill_tick(bool force) {
    if (!prefill_cb_ || !pf_active_) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - pf_tick_ < std::chrono::seconds(1)) return;
    pf_tick_ = now;
    // chunks done since the last tick (their event recorded by the issuing thread and passed), so the progress follows
    // the pipeline instead of the set reuse (which showed the first chunk, then nothing until the last)
    CUDA_CHECK(cudaSetDevice(devs_.back()->id));
    for (int k = 0; k < NSET; ++k) {
        if (!set_pending_[k]) continue;
        bool recorded = false;
        { std::lock_guard<std::mutex> lk(imx_); recorded = ev_chunk_seq_ >= set_seq_[k]; }
        if (!recorded) continue;
        const cudaError_t q = cudaEventQuery(ev_chunk_[k]);
        if (q == cudaErrorNotReady) continue;
        CUDA_CHECK(q);
        set_pending_[k] = false;
        pf_done_ += set_tokens_[k];
    }
    prefill_cb_(pf_done_, pf_total_, pf_reused_);
}

void Engine7::wait_set(int set) {
    const Ds4Config & c = cfg_;
    for (;;) {   // (the set's chunk's event recorded by the last device's issuing thread first)
        std::unique_lock<std::mutex> lk(imx_);
        if (!issue_error_.empty()) throw std::runtime_error("hyper7: issuing failed: " + issue_error_);
        if (ev_chunk_seq_ >= set_seq_[set]) break;
        icv_.wait_for(lk, std::chrono::milliseconds(50));
        lk.unlock();
        prefill_tick(false);
    }
    CUDA_CHECK(cudaSetDevice(devs_.back()->id));
    if (pf_active_) {
        cudaError_t q;
        while ((q = cudaEventQuery(ev_chunk_[set])) == cudaErrorNotReady) { prefill_tick(false); std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
        CUDA_CHECK(q);
    } else CUDA_CHECK(cudaEventSynchronize(ev_chunk_[set]));
    if (set_pending_[set]) {
        set_pending_[set] = false;
        if (pf_active_) { pf_done_ += set_tokens_[set]; prefill_tick(false); }
    }
    if (ids_nt_[set] > 0) {   // the streamed chunk's routing into the statistics
        for (auto & dp : devs_)
            for (Layer & L : dp->layers) {
                if (L.np() == 0) continue;
                auto & cnt = cpu_->counts[L.il];
                const int * ids = h_ids_[set] + (size_t) L.il * R_ * c.n_used;
                for (int i = 0; i < ids_nt_[set] * c.n_used; ++i) if (ids[i] >= 0 && ids[i] < (int) cnt.size()) cnt[ids[i]]++;
            }
        ids_nt_[set] = 0;
    }
}

// a device's part of a chunk: its input rows (the previous device's, through the host), its layers, the rows for the next
// device; the last device adds the head, the DSpark seeds and the chunk's event. The cross-device stream waits are issued
// only after the other device's thread has recorded the event they wait for (a wait on a never-recorded event is a no-op)
void Engine7::issue_device(int gi, const IssueJob & j) {
    const Ds4Config & c = cfg_;
    const int n = c.n_embd, nt = j.nt, set = j.set, nlog = j.nlog;
    const bool bulk = nt > MAX_NT;
    static const bool prof = getenv("HYPER7_PROF") != nullptr;
    Device & d = *devs_[gi];
    const int prev_set = (set + NSET - 1) % NSET;
    auto wait_for = [&](auto pred) {
        std::unique_lock<std::mutex> lk(imx_);
        icv_.wait(lk, [&] { return !issue_error_.empty() || pred(); });
        if (!issue_error_.empty()) throw std::runtime_error("hyper7: issuing failed: " + issue_error_);
    };
    auto mark = [&](long & v) { { std::lock_guard<std::mutex> lk(imx_); v = j.seq; } icv_.notify_all(); };
    CUDA_CHECK(cudaSetDevice(d.id));
    d.cur_set = set;
    cudaStream_t s = d.stream;
    // the next device has taken this device's rows of the previous chunk (they are about to be overwritten)
    if (gi + 1 < (int) devs_.size()) {
        wait_for([&] { return taken_seq_[gi + 1] >= j.seq - 1; });
        CUDA_CHECK(cudaStreamWaitEvent(s, devs_[gi + 1]->ev_taken[prev_set], 0));
    }
    incr_counter(d.counter, s);
    if (gi == 0) {
        CUDA_CHECK(cudaMemcpyAsync(d.x, j.he, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
        embed7(d.x, d.h, n, c.hc, nt, s);
        init_pre7(d.pre, nt, s);
    } else {
        // the previous device's rows through pinned host memory (cudaMemcpyPeerAsync without P2P blocks the host thread
        // until the source is done: no pipeline)
        wait_for([&] { return done_seq_[gi - 1] >= j.seq; });
        CUDA_CHECK(cudaStreamWaitEvent(s, devs_[gi - 1]->ev_done[set], 0));   // (recorded after its copies into h_in)
        CUDA_CHECK(cudaMemcpyAsync(d.h, d.h_in, (size_t) nt * c.hc * n * sizeof(float), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d.pre, d.h_in + (size_t) R_ * c.hc * n, (size_t) nt * 8 * sizeof(float), cudaMemcpyHostToDevice, s));
    }
    CUDA_CHECK(cudaEventRecord(d.ev_taken[set], s));
    mark(taken_seq_[gi]);
    for (Layer & L : d.layers) run_layer(d, L, nt, j.pos);
    if (gi + 1 < (int) devs_.size()) {
        Device & nd = *devs_[gi + 1];
        CUDA_CHECK(cudaMemcpyAsync(nd.h_in, d.h, (size_t) nt * c.hc * n * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(nd.h_in + (size_t) R_ * c.hc * n, d.pre, (size_t) nt * 8 * sizeof(float), cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaEventRecord(d.ev_done[set], s));
    mark(done_seq_[gi]);
    // expert host: the pooled experts of the other device's layers, as their records arrive (after this device's rows went
    // out: the chain's kernels spin on the stream until the owner publishes each layer)
    if (!bulk) for (int il : host_layers_[gi]) host_chain(devs_[dev_of_[il]]->layers[il - devs_[dev_of_[il]]->l0], nt);
    if (gi + 1 < (int) devs_.size()) return;
    if (prof && !bulk && !d.ev_t[0]) for (auto & ev : d.ev_t) CUDA_CHECK(cudaEventCreate(&ev));
    if (nlog > 0) {   // the head: the last nlog rows
        if (prof && !bulk) CUDA_CHECK(cudaEventRecord(d.ev_t[0], s));
        const int r0 = nt - nlog;
        hc7_pre_norm(d.h + (size_t) r0 * c.hc * n, d.pre + (size_t) r0 * 8, d.out_norm, n, c.hc, c.eps, d.x, nlog, s);
        head_mm(d, d.x, nlog, s);
        argmax_pairs(d.logits, c.n_vocab, c.n_vocab, 0, d.outp, nlog, s);
        CUDA_CHECK(cudaMemcpyAsync(h_out_, d.outp, (size_t) nlog * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
        if (prof && !bulk) CUDA_CHECK(cudaEventRecord(d.ev_t[1], s));
    }
    if (prof && !bulk) CUDA_CHECK(cudaEventRecord(d.ev_t[2], d.stream));
    if (!dblk_.empty()) dspark_seed(nt, j.pos);
    if (prof && !bulk) CUDA_CHECK(cudaEventRecord(d.ev_t[3], d.stream));
    CUDA_CHECK(cudaEventRecord(ev_chunk_[set], d.stream));
    mark(ev_chunk_seq_);
}

// expert host, one decode layer: the owner's record (published for the CPU) -> the pooled experts -> the partial sum into
// host_out_[il] with the step's tag, which the owner's reduce waits for (on this device's stream: in order, after its own
// layers of the step; the kernels spin on the record like the owner's reduce spins on the CPU's output)
void Engine7::host_chain(Layer & L, int nt) {
    const Ds4Config & c = cfg_;
    const int n = c.n_embd, K = c.n_used, il = L.il;
    Device & h = *devs_[L.host_dev];
    const MoeLayerDev & HX = place_->dev(L.host_dev, il);   // the pooled experts on the host device
    CUDA_CHECK(cudaSetDevice(h.id));
    cudaStream_t s = h.stream;
    host_take(&cpu_rec_[il].seq, h.counter, (unsigned) il, &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0],
              h.qids, h.qwts, h.qx, n, K, nt, s);
    moe_gate_up(HX.moe, h.qx, n, h.qids, K, h.qhexp, nt, s);
    moe_down(HX.moe, h.qhexp, h.qids, h.qwts, K, h.qyexp, nt, s);
    moe_reduce(h.qzx, h.qzs, h.qyexp, K, h.qout, n, nt, h.qids, HX.owner, L.host_dev, CPU_OWNER, nullptr, nullptr, h.counter, (unsigned) il, s);
    host_put(&host_out_[il].seq, &host_out_[il].y[0][0], h.qout, n, nt, h.counter, (unsigned) il, s);
}

void Engine7::issue_thread(int g) {
    Issuer & I = *issuers_[g];
    for (;;) {
        IssueJob j;
        {
            std::unique_lock<std::mutex> lk(imx_);
            icv_.wait(lk, [&] { return stop_issuers_ || !I.q.empty(); });
            if (stop_issuers_ && I.q.empty()) return;
            j = I.q.front();
            I.q.pop_front();
            I.busy = true;
        }
        try { issue_device(g, j); }
        catch (const std::exception & e) { std::lock_guard<std::mutex> lk(imx_); if (issue_error_.empty()) issue_error_ = e.what(); }
        { std::lock_guard<std::mutex> lk(imx_); I.busy = false; issued_seq_[g] = j.seq; }
        icv_.notify_all();
    }
}

void Engine7::start_issuers() {
    for (size_t g = 0; g < devs_.size(); ++g) {
        issuers_.emplace_back(new Issuer);
        issuers_.back()->th = std::thread(&Engine7::issue_thread, this, (int) g);
    }
    fprintf(stderr, "hyper7: streamed prefill issued by %zu device threads\n", devs_.size());
}

// every queued chunk issued (not necessarily executed): before anything else touches the streams from this thread
void Engine7::issue_drain() {
    if (issuers_.empty()) return;
    for (;;) {
        std::unique_lock<std::mutex> lk(imx_);
        if (!issue_error_.empty()) throw std::runtime_error("hyper7: issuing failed: " + issue_error_);
        bool idle = true;
        for (auto & I : issuers_) if (I->busy || !I->q.empty()) idle = false;
        if (idle) break;
        icv_.wait_for(lk, std::chrono::milliseconds(50));
        lk.unlock();
        prefill_tick(false);
    }
}

int Engine7::enqueue(const int * tokens, int nt, int pos, int nlog) {
    const Ds4Config & c = cfg_;
    if (nt < 1 || nt > R_) throw std::runtime_error("hyper7: bad token count");
    if (pos + nt > opt_.max_pos) throw std::runtime_error("hyper7: position exceeds max_pos");
    const int n = c.n_embd;
    const bool bulk = nt > MAX_NT, streamed = bulk && stream_ && nt >= stream_min_;
    const int set = next_set_;
    next_set_ = (next_set_ + 1) % NSET;
    static const bool prof = getenv("HYPER7_PROF") != nullptr;
    const auto tq0 = std::chrono::steady_clock::now();
    if (taken_seq_.empty()) { taken_seq_.assign(devs_.size(), -1); done_seq_ = taken_seq_; issued_seq_ = taken_seq_; }
    wait_set(set);   // (its previous chunk has read the host rows)
    const auto tq1 = std::chrono::steady_clock::now();
    // token rows (Q8_0 dequantized, as BF16 like the reference) and the Engram rows (the n-grams need the tokens before pos)
    const GTensor & te = gguf_->need("token_embd.weight");
    const auto * te_tr = ggml_get_type_traits((ggml_type) te.type);
    std::vector<int> toks;
    const int back = std::min<int>(pos, c.eg_ngram - 1);
    if ((int) hist_.size() < pos) throw std::runtime_error("hyper7: history shorter than the position");
    toks.insert(toks.end(), hist_.begin() + (pos - back), hist_.begin() + pos);
    toks.insert(toks.end(), tokens, tokens + nt);
    float * he = h_embd_[set];
    // (a few rows: no OpenMP team -- its threads compete with the CPU experts' spinning workers, ~40 ms a decode step)
#pragma omp parallel for schedule(static) if (nt >= 64)
    for (int t = 0; t < nt; ++t) {
        float * out = he + (size_t) t * n;
        te_tr->to_float(te.data + (size_t) tokens[t] * te.row_bytes(), out, n);
        for (int e = 0; e < n; ++e) out[e] = bf_round(out[e]);
    }
    const int cols = c.eg_cols(), hd = c.eg_hd;
    for (size_t i = 0; i < c.eg_layers.size(); ++i) {
        // (unsloth: one table per_layer_token_embd for every Engram layer, its offsets global)
        const std::string tn = "blk." + std::to_string(c.eg_layers[i]) + ".engram_embd.weight";
        const GTensor & tab = gguf_->tensor(tn) ? gguf_->need(tn) : gguf_->need("per_layer_token_embd.weight");
        const auto * tr = ggml_get_type_traits((ggml_type) tab.type);
        const size_t rb = tab.row_bytes();
        float * eg = h_eg_[set][i];
        bool bad = false;
        // every row of the chunk first, then readahead for all of them (the tables live on the disk: one fault at a time
        // per thread was ~12k reads/s; the kernel queues these at once), then the reads and dequantization
        std::vector<int64_t> rows((size_t) nt * cols);
        for (int t = 0; t < nt; ++t) c.eg_rows((int) i, toks.data(), back + t, rows.data() + (size_t) t * cols);   // (before the start: pad)
        if (nt > 1)
            for (int64_t r : rows) {
                if (r < 0 || r >= tab.ne[1]) continue;
                const uintptr_t a0 = (uintptr_t) (tab.data + (size_t) r * rb) & ~(uintptr_t) 4095;
                madvise((void *) a0, ((uintptr_t) (tab.data + (size_t) r * rb + rb) - a0 + 4095) & ~(uintptr_t) 4095, MADV_WILLNEED);
            }
#pragma omp parallel for schedule(dynamic, 4) if (nt >= 64)
        for (int t = 0; t < nt; ++t) {
            float * out = eg + (size_t) t * cols * hd;
            for (int j = 0; j < cols; ++j) {
                const int64_t r = rows[(size_t) t * cols + j];
                if (r < 0 || r >= tab.ne[1]) { bad = true; continue; }
                tr->to_float(tab.data + (size_t) r * rb, out + (size_t) j * hd, hd);
                for (int e = 0; e < hd; ++e) out[(size_t) j * hd + e] = bf_round(out[(size_t) j * hd + e]);
            }
        }
        if (bad) throw std::runtime_error("hyper7: Engram row out of range");
    }
    const auto tq2 = std::chrono::steady_clock::now();
    ++fwd_counter_;
    {
        std::vector<int> slots;
        // (a layer whose CPU experts all went to EP helpers has no parts of its own but is streamed all the same)
        for (auto & dp : devs_) for (Layer & L : dp->layers) if (!(streamed && (L.np() > 0 || !L.shares.empty())) && L.n_gpu < c.n_expert) slots.push_back(L.il);
        if (!slots.empty()) cpu_->expect(fwd_counter_, slots, bulk);
    }
    const long seq = ++chunk_seq_;
    set_seq_[set] = seq;
    set_tokens_[set] = nt;
    set_pending_[set] = true;
    const IssueJob job{set, nt, nlog, (int64_t) pos, seq, he};
    if (streamed && issue_threads_) {
        if (issuers_.empty()) start_issuers();
        { std::lock_guard<std::mutex> lk(imx_); for (auto & I : issuers_) I->q.push_back(job); }
        icv_.notify_all();
    } else {
        issue_drain();
        for (size_t gi = 0; gi < devs_.size(); ++gi) issue_device((int) gi, job);
    }
    if (streamed) ids_nt_[set] = nt;
    if (nlog > 0) last_nlog_ = nlog;
    if (prof) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        Device & d = *devs_.back();
        if (bulk)
            fprintf(stderr, "hyper7: chunk %d at %d: wait %.1f ms, host rows %.1f ms, enqueue %.1f ms\n", nt, pos, ms(tq0, tq1), ms(tq1, tq2),
                    ms(tq2, std::chrono::steady_clock::now()));
        else { const double h = ms(tq1, tq2); t_hostrows_ += h; t_hostrows_max_ = std::max(t_hostrows_max_, h); ++n_hostrows_; }
    }
    // the state now holds these tokens
    hist_.resize(pos);
    hist_.insert(hist_.end(), tokens, tokens + nt);
    return set;
}

std::vector<int> Engine7::forward(const int * tokens, int nt, int pos, int nlog) {
    nlog = std::max(1, std::min({nlog, nt, MAX_NT}));
    static const bool prof = getenv("HYPER7_PROF") != nullptr;
    const auto tf0 = std::chrono::steady_clock::now();
    enqueue(tokens, nt, pos, nlog);
    issue_drain();
    Device & d = *devs_.back();
    CUDA_CHECK(cudaSetDevice(d.id));
    auto t_wait = std::chrono::steady_clock::now();
    bool reported = false;
    for (;;) {
        const cudaError_t q = cudaStreamQuery(d.stream);
        if (q == cudaSuccess) break;
        if (q != cudaErrorNotReady) CUDA_CHECK(q);
        if (!reported && std::chrono::steady_clock::now() - t_wait > std::chrono::seconds(20)) {
            reported = true;
            fprintf(stderr, "hyper7: forward %u slow/stuck; cpu: %s\n", fwd_counter_, cpu_->state().c_str());
        }
        if (pf_active_) { prefill_tick(false); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        else std::this_thread::yield();
    }
    for (int k = 0; k < NSET; ++k) wait_set(k);
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaStreamSynchronize(dp->stream));
        CUDA_CHECK(cudaStreamSynchronize(dp->dstream));
        for (auto & H : dp->hres) if (H.hstream) { CUDA_CHECK(cudaStreamSynchronize(H.hstream)); CUDA_CHECK(cudaStreamSynchronize(H.hcstream)); }
    }
    if (prof && nt <= MAX_NT && d.ev_t[0]) {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, d.ev_t[0], d.ev_t[1])); t_head_ += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, d.ev_t[2], d.ev_t[3])); t_seed_ += ms;
        t_fwd_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf0).count();
        ++n_fwd_;
    }
    std::vector<int> out(nlog);
    for (int t = 0; t < nlog; ++t) { int idx; memcpy(&idx, &h_out_[2 * t + 1], 4); out[t] = idx; }
    return out;
}

// the LM head: Q8_0 as one GEMV over every row (as before); KQ / segments (unsloth's Q6_K head) in GEMVs of MAX_NT rows
void Engine7::head_mm(Device & d, const float * x, int rows, cudaStream_t s) {
    const Ds4Config & c = cfg_;
    if (d.output.seg.empty() && !d.output.kq && !d.output.f16) {
        gemv_q8(d.output.q8, x, c.n_embd, d.logits, c.n_vocab, nullptr, rows, s);
        return;
    }
    const DenseRun dr{s, d.blas, d.xh, d.ggs, d.ggs_elems};
    for (int r0 = 0; r0 < rows; r0 += MAX_NT)
        dense_mm(dr, d.output, x + (size_t) r0 * c.n_embd, c.n_embd, d.logits + (size_t) r0 * c.n_vocab, c.n_vocab, std::min(MAX_NT, rows - r0));
}

void Engine7::get_logits(int row, std::vector<float> & out) {
    const Ds4Config & c = cfg_;
    if (row < 0 || row >= last_nlog_) throw std::runtime_error("hyper7: no such logit row");
    Device & d = *devs_.back();
    CUDA_CHECK(cudaSetDevice(d.id));
    out.resize(c.n_vocab);
    CUDA_CHECK(cudaMemcpy(out.data(), d.logits + (size_t) row * c.n_vocab, (size_t) c.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
}

int Engine7::sample_row(int t, const SamplingParams & sp, int64_t pos) {
    const Ds4Config & c = cfg_;
    Device & d = *devs_.back();
    CUDA_CHECK(cudaSetDevice(d.id));
    float * row = d.logits + (size_t) t * c.n_vocab;
    if (sp.temp > 0.0f && sp.top_k <= 0 && sp.top_p >= 1.0f && sp.min_p <= 0.0f) {   // the whole vocabulary: on the GPU
        int tok = 0;
        gumbel_argmax7(row, c.n_vocab, sp.temp, samp_seed_, pos, (int *) d.outp, d.stream);
        CUDA_CHECK(cudaMemcpyAsync(&tok, d.outp, sizeof(int), cudaMemcpyDeviceToHost, d.stream));
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        return tok;
    }
    topk_pairs(row, c.n_vocab, c.n_vocab, 0, d.outp, TOPK, 1, d.stream);
    std::vector<float> tk(2 * TOPK);
    CUDA_CHECK(cudaMemcpyAsync(tk.data(), d.outp, tk.size() * sizeof(float), cudaMemcpyDeviceToHost, d.stream));
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    std::vector<std::pair<float, int>> cand;
    for (int i = 0; i < TOPK; ++i) { int idx; memcpy(&idx, &tk[2 * i + 1], 4); if (idx >= 0) cand.push_back({tk[2 * i], idx}); }
    auto stats = [&](double invT, float & M, double & Z) {
        float ms[2];
        max_sumexp(row, c.n_vocab, d.outp + 2 * TOPK, d.stream, (float) invT);
        CUDA_CHECK(cudaMemcpyAsync(ms, d.outp + 2 * TOPK, 2 * sizeof(float), cudaMemcpyDeviceToHost, d.stream));
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        M = ms[0];
        Z = ms[1];
    };
    return sample_candidates(cand, TOPK, sp, samp_seed_, pos, stats, [&](std::vector<float> & lg) { get_logits(t, lg); });
}


// tokens of prompt the state h (ring rows from position p0 on) serves: the common prefix, as far back as the ring rows reach
// (the window before the resume position), at a whole compressor group (an odd position re-feeds the token before it)
int Engine7::reuse_len(const std::vector<int> & h, int64_t p0, const std::vector<int> & prompt) const {
    const int cur = (int) h.size(), P = (int) prompt.size();
    int L = 0;
    while (L < cur && L < P - 1 && h[L] == prompt[L]) ++L;
    if (std::max<int64_t>(0, L - cfg_.win) < p0) return 0;   // (the window before the resume position: gone)
    return L & ~1;
}

void Engine7::park() {
    const Ds4Config & c = cfg_;
    const int64_t pos = (int64_t) hist_.size();
    if (pos == 0) return;
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->stream)); }
    static const int keep = getenv("HYPER7_PARK_ROWS") ? atoi(getenv("HYPER7_PARK_ROWS")) : 4096;
    Parked pk;
    pk.hist = hist_;
    pk.p0 = std::max<int64_t>(0, pos - std::min(keep, ring_));
    pk.ring.resize(c.n_layer + dblk_.size()); pk.ckv.resize(c.n_layer); pk.ik.resize(c.n_layer); pk.cstate.resize(c.n_layer);
    const int HD = c.head_dim;
    for (auto & dp : devs_) {
        Device & d = *dp;
        CUDA_CHECK(cudaSetDevice(d.id));
        for (Layer & L : d.layers) {
            auto & rg = pk.ring[L.il];
            rg.resize((size_t) (pos - pk.p0) * HD);
            for (int64_t q = pk.p0; q < pos;) {   // (at most two pieces: the ring wraps)
                const int64_t r = q % ring_, n = std::min<int64_t>(pos - q, ring_ - r);
                CUDA_CHECK(cudaMemcpy(rg.data() + (size_t) (q - pk.p0) * HD, L.ring + (size_t) r * HD, (size_t) n * HD * 2, cudaMemcpyDeviceToHost));
                q += n;
            }
            pk.bytes += rg.size() * 2;
            if (!L.kv_src) continue;
            const size_t rows = (size_t) (pos / L.ratio);
            pk.ckv[L.il].resize(rows * HD);
            if (rows) CUDA_CHECK(cudaMemcpy(pk.ckv[L.il].data(), L.ckv, rows * HD * 2, cudaMemcpyDeviceToHost));
            pk.bytes += rows * HD * 2;
            if (L.ik) {
                pk.ik[L.il].resize(rows * c.ix_hd);
                if (rows) CUDA_CHECK(cudaMemcpy(pk.ik[L.il].data(), L.ik, rows * c.ix_hd * 2, cudaMemcpyDeviceToHost));
                pk.bytes += rows * c.ix_hd * 2;
            }
            if (L.c_kv) {
                auto & cs = pk.cstate[L.il];
                cs.resize((size_t) 2 * L.ratio * HD);
                CUDA_CHECK(cudaMemcpy(cs.data(), L.c_kv, (size_t) L.ratio * HD * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(cs.data() + (size_t) L.ratio * HD, L.c_score, (size_t) L.ratio * HD * 4, cudaMemcpyDeviceToHost));
            }
        }
    }
    for (size_t b = 0; b < dblk_.size(); ++b) {   // the drafter's rings
        Device & d = *devs_[dblk_[b]->dev];
        CUDA_CHECK(cudaSetDevice(d.id));
        CUDA_CHECK(cudaStreamSynchronize(d.dstream));
        auto & rg = pk.ring[c.n_layer + b];
        rg.resize((size_t) (pos - pk.p0) * HD);
        for (int64_t q = pk.p0; q < pos;) {
            const int64_t r = q % ring_, n = std::min<int64_t>(pos - q, ring_ - r);
            CUDA_CHECK(cudaMemcpy(rg.data() + (size_t) (q - pk.p0) * HD, dblk_[b]->ring + (size_t) r * HD, (size_t) n * HD * 2, cudaMemcpyDeviceToHost));
            q += n;
        }
        pk.bytes += rg.size() * 2;
    }
    pk.used = ++park_clock_;
    // the same conversation parked before: replaced
    for (size_t i = 0; i < parked_.size(); ++i)
        if (parked_[i].hist.size() <= pk.hist.size() && std::equal(parked_[i].hist.begin(), parked_[i].hist.end(), pk.hist.begin())) {
            parked_.erase(parked_.begin() + i);
            break;
        }
    fprintf(stderr, "hyper7: parked a conversation of %lld tokens (%.0f MiB)\n", (long long) pos, pk.bytes / 1048576.0);
    parked_.push_back(std::move(pk));
    static const double cap_gib = getenv("HYPER7_PARK_GIB") ? atof(getenv("HYPER7_PARK_GIB")) : 12.0;
    for (;;) {   // least recently used out, past the memory cap
        size_t tot = 0;
        for (auto & q : parked_) tot += q.bytes;
        if (tot <= (size_t) (cap_gib * 1073741824.0) || parked_.size() <= 1) break;
        size_t lru = 0;
        for (size_t i = 1; i < parked_.size(); ++i) if (parked_[i].used < parked_[lru].used) lru = i;
        parked_.erase(parked_.begin() + lru);
    }
}

void Engine7::unpark(size_t i) {
    const Ds4Config & c = cfg_;
    Parked pk = std::move(parked_[i]);
    parked_.erase(parked_.begin() + i);
    const int64_t pos = (int64_t) pk.hist.size();
    const int HD = c.head_dim;
    for (auto & dp : devs_) {
        Device & d = *dp;
        CUDA_CHECK(cudaSetDevice(d.id));
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        for (Layer & L : d.layers) {
            const auto & rg = pk.ring[L.il];
            for (int64_t q = pk.p0; q < pos;) {
                const int64_t r = q % ring_, n = std::min<int64_t>(pos - q, ring_ - r);
                CUDA_CHECK(cudaMemcpy(L.ring + (size_t) r * HD, rg.data() + (size_t) (q - pk.p0) * HD, (size_t) n * HD * 2, cudaMemcpyHostToDevice));
                q += n;
            }
            if (!L.kv_src) continue;
            if (!pk.ckv[L.il].empty()) CUDA_CHECK(cudaMemcpy(L.ckv, pk.ckv[L.il].data(), pk.ckv[L.il].size() * 2, cudaMemcpyHostToDevice));
            if (L.ik && !pk.ik[L.il].empty()) CUDA_CHECK(cudaMemcpy(L.ik, pk.ik[L.il].data(), pk.ik[L.il].size() * 2, cudaMemcpyHostToDevice));
            if (L.c_kv) {
                const auto & cs = pk.cstate[L.il];
                CUDA_CHECK(cudaMemcpy(L.c_kv, cs.data(), (size_t) L.ratio * HD * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(L.c_score, cs.data() + (size_t) L.ratio * HD, (size_t) L.ratio * HD * 4, cudaMemcpyHostToDevice));
            }
        }
    }
    for (size_t b = 0; b < dblk_.size(); ++b) {   // the drafter's rings
        Device & d = *devs_[dblk_[b]->dev];
        CUDA_CHECK(cudaSetDevice(d.id));
        const auto & rg = pk.ring[c.n_layer + b];
        for (int64_t q = pk.p0; q < pos && !rg.empty();) {
            const int64_t r = q % ring_, n = std::min<int64_t>(pos - q, ring_ - r);
            CUDA_CHECK(cudaMemcpy(dblk_[b]->ring + (size_t) r * HD, rg.data() + (size_t) (q - pk.p0) * HD, (size_t) n * HD * 2, cudaMemcpyHostToDevice));
            q += n;
        }
    }
    hist_ = std::move(pk.hist);
    fprintf(stderr, "hyper7: restored a parked conversation of %lld tokens\n", (long long) pos);
}

std::vector<int> Engine7::generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                                   const std::function<bool(int)> & on_token, const SamplingParams & sp) {
    using clk = std::chrono::steady_clock;
    if (prompt.empty()) throw std::runtime_error("generate: empty prompt");
    if ((int) prompt.size() + 8 > opt_.max_pos) throw std::runtime_error("generate: prompt longer than the context");
    const bool sampling = sp.temp > 0.0f;
    samp_seed_ = sp.seed ? sp.seed : random_seed64();
    const int P = (int) prompt.size();
    // prompt cache: the common prefix with what the state holds, as far as the window ring reaches back (and at a whole
    // compressor group: an odd position re-feeds the token before it)
    int s = 0;
    if (opt_.prompt_cache) {
        const int cur = (int) hist_.size();
        s = reuse_len(hist_, std::max<int64_t>(0, cur - (ring_ - R_)), prompt);
        // a parked conversation that serves this prompt better comes back (the current one is parked); a different
        // conversation parks the current one before overwriting it
        int best = -1, bs = s;
        for (size_t i = 0; i < parked_.size(); ++i) {
            const int r = reuse_len(parked_[i].hist, parked_[i].p0, prompt);
            if (r > bs) { bs = r; best = (int) i; }
        }
        static const int park_min = getenv("HYPER7_PARK_MIN") ? atoi(getenv("HYPER7_PARK_MIN")) : 1024;
        if (best >= 0 && bs >= s + 256) {
            if (cur >= park_min && s < cur / 2) park();
            unpark((size_t) best);
            s = bs;
        } else if (cur >= park_min && s < cur / 2) park();
    }
    if (s == 0) hist_.clear();
    GenStats st;
    auto tp = clk::now();
    int next = -1;
    pf_active_ = true; pf_total_ = P; pf_reused_ = s; pf_done_ = s;
    pf_tick_ = std::chrono::steady_clock::now();
    for (int c0 = s; c0 < P;) {
        // equal chunks of at most R_; all but the last enqueued without waiting (a pipeline over the GPUs) when streamed
        const int left = P - c0, nch = (left + R_ - 1) / R_, len = std::min(left, (left + nch - 1) / nch);
        if (c0 + len < P && len >= stream_min_ && stream_) enqueue(&prompt[c0], len, c0, 0);
        else next = forward(&prompt[c0], len, c0, 1)[0];
        c0 += len;
    }
    pf_active_ = false;
    if (prefill_cb_) prefill_cb_(P, P, s);
    if (sampling) next = sample_row(0, sp, P);
    if (getenv("HYPER7_DSPARK_DEBUG")) fprintf(stderr, "hyper7: prefill done (%d tokens), next %d\n", P, next);
    st.t_prefill = std::chrono::duration<double>(clk::now() - tp).count();
    st.prompt_reused = s;
    Emitter em;
    em.n_gen = n_gen; em.on_token = &on_token;
    std::vector<int> & out = em.out;
    bool & stop = em.stop;
    auto emit = [&](int tok) { return em.emit(tok); };
    int p = P;
    auto t0 = clk::now();
    const int K = spec ? n_draft() : 0;
    if (K <= 0) {
        while (emit(next) && p + 1 < opt_.max_pos) {
            next = forward(&next, 1, p++, 1)[0];
            if (sampling) next = sample_row(0, sp, p);
            st.steps++;
            place_->before_forward(false);   // (the placement follows the routing every few steps)
        }
    } else {
        // prompt-lookup speculation: drafts = the tokens that followed the latest earlier occurrence of the last NG tokens
        // (history = prompt + output); a draft is kept iff the token sampled (or argmax) at its row equals it: exact.
        // Rejected rows leave nothing behind (their window-ring rows, compressed rows and selections are rewritten when the
        // positions come again) except a ratio-2 compressor's open group: when the next position is its second slot and a
        // rejected row overwrote the first, the step re-feeds the group's first token
        LookupDrafter lk(prompt);
        const std::vector<int> & hist = lk.hist();
        int cur = next;
        bool refeed = false;
        std::vector<int> in(MAX_NT);
        while (!stop && p + K + 1 < opt_.max_pos) {
            if (!emit(cur)) break;
            lk.push(cur);
            const int kmax = K - (refeed ? 1 : 0);
            std::vector<int> dr;
            std::vector<float> lq;
            // speculative sampling: sampled drafts verified by min(1, p/q) (HYPER7_DSPARK_COUPLED=1: the old exact match)
            const bool spec_samp = !dblk_.empty() && sampling && sp.top_k <= 0 && sp.top_p >= 1.0f && sp.min_p <= 0.0f && !dsp_coupled_;
            if (!dblk_.empty() && kmax > 0) {   // DSpark: the drafts while their confidence holds
                std::vector<int> dd;
                std::vector<float> cf;
                auto ta = clk::now();
                static const bool dbg = getenv("HYPER7_DSPARK_DEBUG") != nullptr;
                if (dbg) fprintf(stderr, "hyper7: draft at %d (y %d)\n", p, cur);
                dspark_draft(cur, p, dd, cf, spec_samp ? sp.temp : 0.0f, samp_seed_ ^ 0x5d5d5d5dd5d5d5d5ull, &lq);
                if (dbg) { fprintf(stderr, "hyper7: drafts"); for (size_t j = 0; j < dd.size(); ++j) fprintf(stderr, " %d (%.2f)", dd[j], cf[j]); fprintf(stderr, "\n"); }
                st.t_mtp += std::chrono::duration<double>(clk::now() - ta).count();
                for (int j = 0; j < (int) dd.size() && (int) dr.size() < kmax && cf[j] >= dsp_conf_; ++j) dr.push_back(dd[j]);
                lq.resize(dr.size());
            } else if (kmax > 0) {
                dr.resize(kmax);
                dr.resize(lk.draft(kmax, acc_rate_, dr.data()));
            }
            const int nd = (int) dr.size(), off = refeed ? 1 : 0, start = p - off;
            int n_in = 0;
            if (refeed) in[n_in++] = hist[hist.size() - 2];
            in[n_in++] = cur;
            for (int j = 0; j < nd; ++j) in[n_in++] = dr[j];
            std::vector<int> a = forward(in.data(), n_in, start, n_in);
            st.steps++;
            st.drafted += nd;
            int m = 0, nxt;
            const auto ts0 = clk::now();
            if (spec_samp && nd > 0) nxt = dspark_verify(off, dr, lq, sp, p, m);
            else {
                auto pick = [&](int j) { return sampling ? sample_row(off + j, sp, p + j + 1) : a[off + j]; };
                nxt = pick(0);
                while (m < nd && nxt == dr[m]) { ++m; nxt = pick(m); }
            }
            t_samp_ += std::chrono::duration<double, std::milli>(clk::now() - ts0).count();
            st.accepted += m;
            if (nd > 0) acc_rate_ = 0.9 * acc_rate_ + 0.1 * ((double) m / nd);
            int r = 0;
            while (r < m) { lk.push(dr[r]); if (!emit(dr[r++])) break; }
            if (stop) { p += r + 1; break; }
            p += m + 1;
            refeed = has_r2_ && (p & 1) && m + 2 <= nd;
            cur = nxt;
            place_->before_forward(false);   // (the placement follows the routing every few steps)
        }
    }
    st.tokens = (int) out.size();
    st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    if (n_hostrows_)
        fprintf(stderr, "hyper7: decode host rows (embedding + Engram): %.2f ms/step mean, %.1f max over %d steps; rebalance %.1f ms (%d exchanges)\n",
                t_hostrows_ / n_hostrows_, t_hostrows_max_, n_hostrows_, place_->t_rebalance(), place_->n_swaps());
    if (n_fwd_)
        fprintf(stderr, "hyper7: decode per step (%d): forward %.2f ms wall (head %.2f, DSpark seed %.2f GPU ms), draft %.2f ms, sampling %.2f ms, step %.2f ms\n",
                n_fwd_, t_fwd_ / n_fwd_, t_head_ / n_fwd_, t_seed_ / n_fwd_, st.t_mtp * 1e3 / std::max(1, st.steps), t_samp_ / std::max(1, st.steps),
                st.seconds * 1e3 / std::max(1, st.steps));
    if (stats) *stats = st;
    return out;
}

} // namespace hyper
