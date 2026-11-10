#include "engine4.h"
#include "dense.h"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <pthread.h>

#include <algorithm>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace hyper {




constexpr int R4 = MOE_BULK_ROWS;   // activation rows (prefill chunk)
constexpr int QSA_SCORE_ROWS = 64; // tokens scored at a time (scratch rows of max_pos/4 scores)
constexpr int QSA_LIST = 2052;     // max attended cells per token: 512 pools * 4 + 3 tail cells (+1)

struct Engine4::DevLayer {
    bool full = false, ple = false;
    // hyper-connection mixers (replicated): norm [hc*n], down [lr x hc*n], up [hc*n x lr], inject [hc][hc*n]
    float * hca_norm = nullptr, * hcf_norm = nullptr, * hca_inj = nullptr, * hcf_inj = nullptr;
    DW hca_down, hca_up, hcf_down, hcf_up;
    // gated attention: local q heads [head_off, +n_head_l), local kv heads [kv_off, +n_kv_l)
    DW wqkv, wo;
    float * q_norm = nullptr, * k_norm = nullptr;
    half * kcache = nullptr, * vcache = nullptr;
    int n_head_l = 0, head_off = 0, n_kv_l = 0, kv_off = 0;
    // QSA indexer (replicated): query / raw key projections, norms, raw key cache [max_pos][128], pooled keys [max_pos/4][128]
    BF16W idx_q, idx_k;
    float * idx_qn = nullptr, * idx_kn = nullptr;
    half * kraw = nullptr, * kpool = nullptr;
    // gated delta net (head-aligned partition)
    DW win, wout;
    float * ab_w = nullptr;   // fp32 rows [alpha local | beta local]
    float * conv_w = nullptr, * dt_bias = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    float * conv_state = nullptr, * state = nullptr;
    float * conv_snap = nullptr, * state_snap = nullptr;   // MTP verification: state after each of the first nt-1 rows
    int n_v_l = 0, n_k_l = 0, conv_ch = 0;
    // MoE
    float * router = nullptr;  // fp32 [n_expert + 1][n]: experts, then the shared-expert gate
    uint16_t * router16 = nullptr;   // ... or as bf16, when every value is one exactly (Flash-Next: all; half the bytes)
    DW sh_gu, sh_down;
    int n_sh_l = 0;
    // PLE
    DW ple_key, ple_value;
    float * ple_wk = nullptr, * ple_wq = nullptr, * ple_wc = nullptr, * ple_conv = nullptr, * ple_state = nullptr, * ple_snap = nullptr;
};

constexpr int TRACE2_SLOTS = 10;   // HYPER4_TRACE stages: ple_embd, hc_init, 4 per layer for layers 0-1

struct Engine4::Device : TpDevice {
    std::vector<DevLayer> layers;
    // MTP (NextN) block: layer `mtp`, its embedding / hidden norms, eh_proj (per hc stream), its own exit mixer
    DevLayer mtp;
    float * m_enorm = nullptr, * m_hnorm = nullptr, * m_head_norm = nullptr;
    DW m_eh, m_head_down, m_head_up;
    float * mres = nullptr, * mh = nullptr, * ecat = nullptr;   // MTP residual rows, chained hidden (1 row), eh_proj input
    int * mpos = nullptr;
    int * rpos = nullptr;   // [cells][3]: M-RoPE positions (t, h, w) per cache cell (cfg_.rope_sec), see set_rpos
    float * head_norm = nullptr;
    DW head_down, head_up;
    // activations [MAX_NT] rows
    float * res2 = nullptr, * gate = nullptr, * lo = nullptr, * inj = nullptr, * mixed = nullptr, * injp = nullptr;
    unsigned * inj_cnt = nullptr;   // hc norm: per token, the stream blocks' tickets
    float * ple_emb = nullptr, * ple_key = nullptr, * ple_val = nullptr, * ple_sc = nullptr;
    std::vector<float *> ple_x;     // the further chunks of a multi-chunk prefill pass: PLE rows
    float * trace = nullptr;   // HYPER4_TRACE: the last row's residual after every layer [n_layer][hc * n_embd]
    float * trace2 = nullptr;  // HYPER4_TRACE: stages of layers 0-1 (TRACE2_SLOTS slots of trace2_w floats)
    float * iq = nullptr, * ik = nullptr, * iqn = nullptr;   // QSA: projections, normed queries
    cudaStream_t s2 = nullptr;                                // decode: the shared expert, concurrently with the routed ones
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;

    ~Device() {
        cudaSetDevice(id);
        if (s2) cudaStreamDestroy(s2);
        if (ev_fork) cudaEventDestroy(ev_fork);
        if (ev_join) cudaEventDestroy(ev_join);
    }
};

Engine4::Engine4(const std::string & model_path, const Engine4Options & opt) : TpEngine(opt, TpModel{}) {
    allrows_ = getenv("HYPER4_ALLROWS") != nullptr;
    grouped_decode_ = getenv("HYPER4_GROUPED") != nullptr;   // timing experiment only: CPU experts ignored (wrong output)
    mtp_whole_norm_ = getenv("HYPER4_MTP_WHOLE_NORM") != nullptr;
    gguf_ = std::make_unique<GGUF>(model_path);
    cfg_ = Q4Config::from_gguf(*gguf_);
    src_ = gguf_.get();
    if (!opt_.mtp_path.empty()) {
        mtp_g_ = std::make_unique<GGUF>(opt_.mtp_path);
        if (!mtp_g_->tensor("blk." + std::to_string(cfg_.n_layer) + ".nextn.eh_proj.weight"))
            throw std::runtime_error("MTP file has no NextN block for layer " + std::to_string(cfg_.n_layer));
        fprintf(stderr, "hyper4: MTP head from %s\n", opt_.mtp_path.c_str());
    }
    fprintf(stderr, "hyper4: %s\n", cfg_.describe().c_str());
    m_.tag = "hyper4"; m_.env = "HYPER4";
    m_.n_layer = cfg_.n_layer; m_.n_dense = 0;
    m_.n_embd = cfg_.n_embd; m_.n_vocab = cfg_.n_vocab;
    m_.n_expert = cfg_.n_expert; m_.n_expert_used = cfg_.n_expert_used; m_.n_ff_exp = cfg_.n_ff_exp;
    m_.clamp_exp = 0.0f;
    m_.stream_min = 256; m_.up_parts = 2; m_.halves = 1; m_.stage_factor = 2.0; m_.fill_unit = 256;
    // 0.5 measured best (9k prompt): Flash-Next 138 -> 142.5 t/s, Uncensored with the base model's NextN head 124 -> 129.5 t/s
    m_.mtp_pmin = 0.5;
    init();
    const int n = cfg_.n_embd;
    CUDA_CHECK(cudaHostAlloc(&h_ple_, (size_t) mc_max_ * R4 * std::max(1, cfg_.ple_n_heads() * cfg_.ple_dim) * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_membd_, (size_t) R4 * n * sizeof(float), cudaHostAllocPortable));
}

Engine4::~Engine4() {
    if (h_ple_) cudaFreeHost(h_ple_);
    if (h_membd_) cudaFreeHost(h_membd_);
}

std::unique_ptr<TpDevice> Engine4::make_device() { return std::make_unique<Device>(); }
Engine4::Device & Engine4::dev(int g) { return static_cast<Device &>(*devs_[g]); }
const Engine4::Device & Engine4::dev(int g) const { return static_cast<const Device &>(*devs_[g]); }

void Engine4::load_layer(Device & dev, DevLayer & L, int il) {
    const Q4Config & c = cfg_;
    const int nd = opt_.n_devices, g = dev.g, n = c.n_embd;
    const int dk = c.ssm_d_state, dv = c.head_v_dim(), nk = c.ssm_n_group, nv = c.ssm_dt_rank;
    auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
    auto T = [&](const std::string & name) { return &src_->need(name); };
    auto f32 = [&](const std::string & name) {
        const std::vector<float> v = to_f32(src_->need(name));
        return dev.upload(v.data(), v.size());
    };
    auto q8full = [&](const std::string & name) { const GTensor * t = T(name); return upload_dense(A, dev.id, {{t, 0, t->rows()}}); };
    const std::string p = "blk." + std::to_string(il) + ".";
    L.full = c.is_full_attn(il) || il == c.n_layer;   // the MTP block is an attention layer
    if (c.plain) {   // qwen35moe: plain pre-norm residual (the block norms in the mixers' place)
        L.hca_norm = f32(p + "attn_norm.weight");
        L.hcf_norm = f32(p + "post_attention_norm.weight");
    } else {   // hyper-connection mixers
        L.hca_norm = f32(p + "hc_attn_norm.weight");
        L.hcf_norm = f32(p + "hc_ffn_norm.weight");
        L.hca_inj = f32(p + "hc_attn_inject.weight");
        L.hcf_inj = f32(p + "hc_ffn_inject.weight");
        L.hca_down = q8full(p + "hc_attn_down.weight");
        L.hca_up = upload_hc_up(A, dev.id, *T(p + "hc_attn_up.weight"), c.hc);
        L.hcf_down = q8full(p + "hc_ffn_down.weight");
        L.hcf_up = upload_hc_up(A, dev.id, *T(p + "hc_ffn_up.weight"), c.hc);
    }
    if (L.full) {
        // whole kv groups per GPU (each KV head and its cache live on one GPU only); with fewer kv heads than GPUs,
        // some GPUs do no attention and keep that memory for experts
        const int hd = c.head_dim, group = c.n_head / c.n_head_kv;
        auto [kv0, kv1] = split(c.n_head_kv, nd, g);
        // HYPER4_KV_DEVS="0,1": with fewer kv heads than GPUs, which GPUs take them (head j to the j-th listed) -- the
        // attention layers' whole work lands on those GPUs, so the lighter ones should take it
        static const std::vector<int> kv_devs = [] {
            std::vector<int> v;
            if (const char * e = getenv("HYPER4_KV_DEVS")) for (const char * q = e; *q;) { v.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
            return v;
        }();
        if ((int) kv_devs.size() == c.n_head_kv && c.n_head_kv <= nd) {
            kv0 = kv1 = 0;
            for (int j = 0; j < c.n_head_kv; ++j) if (kv_devs[j] == g) { kv0 = j; kv1 = j + 1; }
        }
        L.kv_off = (int) kv0;
        L.n_kv_l = (int) (kv1 - kv0);
        L.head_off = L.kv_off * group;
        L.n_head_l = L.n_kv_l * group;
    }
    if (L.full && L.n_head_l > 0) {
        const int hd = c.head_dim;
        L.wqkv = upload_dense(A, dev.id, {{T(p + "attn_q.weight"), (int64_t) L.head_off * 2 * hd, (int64_t) (L.head_off + L.n_head_l) * 2 * hd},
                                       {T(p + "attn_k.weight"), (int64_t) L.kv_off * hd, (int64_t) (L.kv_off + L.n_kv_l) * hd},
                                       {T(p + "attn_v.weight"), (int64_t) L.kv_off * hd, (int64_t) (L.kv_off + L.n_kv_l) * hd}});
        L.wo = upload_dense(A, dev.id, {{T(p + "attn_output.weight"), 0, n}},
                         {{(int64_t) L.head_off * hd / 32, (int64_t) (L.head_off + L.n_head_l) * hd / 32}});
        L.q_norm = f32(p + "attn_q_norm.weight");
        L.k_norm = f32(p + "attn_k_norm.weight");
        L.kcache = kv_alloc<half>(dev, L.n_kv_l, (size_t) opt_.max_pos * hd, hd, 1);
        L.vcache = kv_alloc<half>(dev, L.n_kv_l, (size_t) opt_.max_pos * hd, hd, 1);
        if (c.compress[il] > 0) {
            if (c.compress[il] != 4 || c.idx_dim != 128 || c.idx_n_head > 4) throw std::runtime_error("QSA: only kpool 4, 128-dim, <= 4 heads");
            L.idx_q = upload_bf16(A, dev.id, src_->need(p + "indexer.q_proj.weight"));
            L.idx_k = upload_bf16(A, dev.id, src_->need(p + "indexer.k_proj.weight"));
            L.idx_qn = f32(p + "indexer.q_norm.weight");
            L.idx_kn = f32(p + "indexer.k_norm.weight");
            L.kraw = kv_alloc<half>(dev, 1, (size_t) opt_.max_pos * 128, 128, 1);
            L.kpool = kv_alloc<half>(dev, 1, (size_t) (opt_.max_pos / 4 + 1) * 128, 128, 4);
        }
    } else if (!L.full) {
        auto [k0, k1] = split(nk, nd, g);
        std::vector<int> vh;
        for (int grp = 0; grp < nv / nk; ++grp) for (int64_t kk = k0; kk < k1; ++kk) vh.push_back(grp * nk + (int) kk);
        L.n_k_l = (int) (k1 - k0);
        L.n_v_l = (int) vh.size();
        const int64_t koff = (int64_t) nk * dk, voff = (int64_t) 2 * nk * dk;
        L.conv_ch = 2 * L.n_k_l * dk + L.n_v_l * dv;
        const GTensor * qkv = T(p + "attn_qkv.weight");
        const GTensor * zt = T(p + "attn_gate.weight");
        std::vector<RowRange> rows = {{qkv, k0 * dk, k1 * dk}, {qkv, koff + k0 * dk, koff + k1 * dk}};
        std::vector<RowRange> z_rows;
        ColRanges out_cols;
        std::vector<int64_t> chans;
        for (int64_t ch = k0 * dk; ch < k1 * dk; ++ch) chans.push_back(ch);
        for (int64_t ch = koff + k0 * dk; ch < koff + k1 * dk; ++ch) chans.push_back(ch);
        for (int h : vh) {
            rows.push_back({qkv, voff + (int64_t) h * dv, voff + (int64_t) (h + 1) * dv});
            z_rows.push_back({zt, (int64_t) h * dv, (int64_t) (h + 1) * dv});
            out_cols.push_back({(int64_t) h * dv / 32, (int64_t) (h + 1) * dv / 32});
            for (int64_t ch = voff + (int64_t) h * dv; ch < voff + (int64_t) (h + 1) * dv; ++ch) chans.push_back(ch);
        }
        rows.insert(rows.end(), z_rows.begin(), z_rows.end());
        L.win = upload_dense(A, dev.id, rows);
        L.wout = upload_dense(A, dev.id, {{T(p + "ssm_out.weight"), 0, n}}, out_cols);
        {   // alpha / beta rows (fp32)
            const std::vector<float> fa = to_f32(src_->need(p + "ssm_alpha.weight")), fb = to_f32(src_->need(p + "ssm_beta.weight"));
            std::vector<float> buf;
            for (int h : vh) buf.insert(buf.end(), fa.begin() + (size_t) h * n, fa.begin() + (size_t) (h + 1) * n);
            for (int h : vh) buf.insert(buf.end(), fb.begin() + (size_t) h * n, fb.begin() + (size_t) (h + 1) * n);
            L.ab_w = dev.upload(buf.data(), buf.size());
        }
        {
            const std::vector<float> cw = to_f32(src_->need(p + "ssm_conv1d.weight"));
            const int K = c.ssm_conv;
            std::vector<float> buf((size_t) L.conv_ch * K);
            for (size_t j = 0; j < chans.size(); ++j)
                memcpy(buf.data() + j * K, cw.data() + chans[j] * K, K * sizeof(float));
            L.conv_w = dev.upload(buf.data(), buf.size());
        }
        auto gather = [&](const std::string & name) {
            const std::vector<float> t = to_f32(src_->need(name));
            std::vector<float> buf;
            for (int h : vh) buf.push_back(t[h]);
            return dev.upload(buf.data(), buf.size());
        };
        L.dt_bias = gather(p + "ssm_dt.bias");
        L.ssm_a = gather(p + "ssm_a");
        L.ssm_norm = f32(p + "ssm_norm.weight");
        L.conv_state = dev.alloc<float>((size_t) (c.ssm_conv - 1) * L.conv_ch);
        L.state = dev.alloc<float>((size_t) L.n_v_l * dk * dv);
        if (!opt_.mtp_path.empty()) {
            L.conv_snap = dev.alloc<float>((size_t) (c.ssm_conv - 1) * L.conv_ch * (MAX_NT - 1));
            L.state_snap = dev.alloc<float>((size_t) L.n_v_l * dk * dv * (MAX_NT - 1));
        }
    }
    // PLE (replicated)
    L.ple = il == c.ple_layer;
    if (L.ple) {
        L.ple_key = q8full(p + "ple_key.weight");
        L.ple_value = q8full(p + "ple_value.weight");
        L.ple_wk = f32(p + "ple_norm_key.weight");
        L.ple_wq = f32(p + "ple_norm_query.weight");
        L.ple_wc = f32(p + "ple_norm_conv.weight");
        L.ple_conv = f32(p + "ple_conv1d.weight");
        L.ple_state = dev.alloc<float>((size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim());
        if (!opt_.mtp_path.empty()) L.ple_snap = dev.alloc<float>((size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim() * (MAX_NT - 1));
    }
    // router + shared-expert gate (replicated, fp32)
    {
        std::vector<float> buf = to_f32(src_->need(p + "ffn_gate_inp.weight"));
        const std::vector<float> sgt = to_f32(src_->need(p + "ffn_gate_inp_shexp.weight"));
        buf.insert(buf.end(), sgt.begin(), sgt.end());
        static const bool r32 = getenv("HYPER4_ROUTER32") != nullptr;   // (A/B: the fp32 router)
        std::vector<uint16_t> b16(buf.size());
        bool exact = !r32;
        for (size_t i = 0; i < buf.size() && exact; ++i) {
            uint32_t u;
            memcpy(&u, &buf[i], 4);
            exact = (u & 0xffffu) == 0;
            b16[i] = (uint16_t) (u >> 16);
        }
        if (exact) L.router16 = dev.upload(b16.data(), b16.size());
        else {
            L.router = dev.upload(buf.data(), buf.size());
            if (!r32 && dev.g == 0) fprintf(stderr, "hyper4: layer %d router stays fp32 (not exactly bf16)\n", il);
        }
    }
    // shared expert: hidden split
    {
        auto [f0, f1] = split(c.n_ff_shexp, nd, g, 32);
        L.n_sh_l = (int) (f1 - f0);   // (K-sliced down projection: blocks of 32 columns)
        L.sh_gu = upload_dense(A, dev.id, {{T(p + "ffn_gate_shexp.weight"), f0, f1}, {T(p + "ffn_up_shexp.weight"), f0, f1}});
        L.sh_down = upload_dense(A, dev.id, {{T(p + "ffn_down_shexp.weight"), 0, n}}, {{f0 / 32, f1 / 32}});
    }
}

// experts of layer il: the most used ones (routing statistics) on the GPUs, dealt in proportion to each GPU's quota,
// the rest on the CPU (compact huge-page copy)
// the experts of layer il: owners by weighted round robin over the devices' quotas (hottest first); the rest is the
// shared placement's (device arrays, the compact host copy, the CPU side, stream lists)
// a device's layers (the MTP block when loaded), its exit mixer and vocab slice, the activations
void Engine4::load_device(TpDevice & td) {
    Device & dev = static_cast<Device &>(td);
    const Q4Config & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd, hcn = c.hc_dim(), g = dev.g;
    CUDA_CHECK(cudaSetDevice(dev.id));
    CUDA_CHECK(cudaStreamCreateWithFlags(&dev.s2, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&dev.ev_fork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&dev.ev_join, cudaEventDisableTiming));
    auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
    dev.layers.resize(c.n_layer);
    src_ = gguf_.get();
    for (int il = 0; il < c.n_layer; ++il) load_layer(dev, dev.layers[il], il);
    if (mtp_g_) {   // NextN block from the separate MTP file
        src_ = mtp_g_.get();
        const std::string p = "blk." + std::to_string(c.n_layer) + ".";
        load_layer(dev, dev.mtp, c.n_layer);
        auto up32 = [&](const std::string & name) { const std::vector<float> v = to_f32(mtp_g_->need(name)); return dev.upload(v.data(), v.size()); };
        dev.m_enorm = up32(p + "nextn.enorm.weight");
        dev.m_hnorm = up32(p + "nextn.hnorm.weight");
        { const GTensor * t = &mtp_g_->need(p + "nextn.eh_proj.weight"); dev.m_eh = upload_dense(A, dev.id, {{t, 0, t->rows()}}); }
        if (c.plain) {   // qwen35moe: the shared head norm (or the model's output norm)
            const bool own = mtp_g_->tensor(p + "nextn.shared_head_norm.weight") != nullptr;
            dev.m_head_norm = up32(own ? p + "nextn.shared_head_norm.weight" : "output_norm.weight");
            src_ = gguf_.get();
        } else {
        // exit mixer: the file's own nextn.hc_head_* or, in predictor-only files, its output_hc_*
        const bool own = mtp_g_->tensor(p + "nextn.hc_head_norm.weight") != nullptr;
        const std::string hn = own ? p + "nextn.hc_head_norm.weight" : "output_hc_norm.weight";
        const std::string hd = own ? p + "nextn.hc_head_down.weight" : "output_hc_down.weight";
        const std::string hu = own ? p + "nextn.hc_head_up.weight" : "output_hc_up.weight";
        dev.m_head_norm = up32(hn);
        { const GTensor * t = &mtp_g_->need(hd); dev.m_head_down = upload_dense(A, dev.id, {{t, 0, t->rows()}}); }
        dev.m_head_up = upload_hc_up(A, dev.id, mtp_g_->need(hu), c.hc);
        src_ = gguf_.get();
        }
    }
    if (c.plain) { const std::vector<float> v = to_f32(gguf_->need("output_norm.weight")); dev.head_norm = dev.upload(v.data(), v.size()); }
    else {
    { const std::vector<float> v = to_f32(gguf_->need("output_hc_norm.weight")); dev.head_norm = dev.upload(v.data(), v.size()); }
    { const GTensor * t = &gguf_->need("output_hc_down.weight"); dev.head_down = upload_dense(A, dev.id, {{t, 0, t->rows()}}); }
    dev.head_up = upload_hc_up(A, dev.id, gguf_->need("output_hc_up.weight"), c.hc);
    }
    auto [o0, o1] = split(c.n_vocab, nd, g);
    dev.vocab_off = (int) o0;
    dev.output = upload_dense(A, dev.id, {{&gguf_->need("output.weight"), o0, o1}});
    // activations: the engine's common buffers, then the model's own
    const int R = R4;
    CommonSizes sz;
    sz.res_w = hcn; sz.xn_w = hcn;
    sz.big_stride = c.conv_dim() + c.ssm_d_inner + 2 * c.ssm_dt_rank + 2 * c.n_head * c.head_dim + 2 * c.n_head_kv * c.head_dim;
    sz.o_w = std::max(c.ssm_d_inner, c.n_head * c.head_dim);
    for (int r = 1; r <= R; ++r) sz.attn_part = std::max(sz.attn_part, attn_part_floats(c.n_head, 1, r, c.head_dim));
    sz.rlog_w = c.n_expert + 1;   // (the experts, then the shared-expert gate)
    sz.pred_k_w = c.n_expert_used; sz.pred_psg_w = 1;
    sz.ff_sh = c.n_ff_shexp;
    sz.hexp_rows = R;
    sz.logit_rows = allrows_ ? R : MAX_NT;
    // (fp16 rows of every bulk matmul input: the output projections read ssm_d_inner / n_head * head_dim wide rows, wider
    // than the residual when hc = 1)
    sz.xh_w = std::max({hcn, c.n_ff_shexp, c.ssm_d_inner, c.n_head * c.head_dim});
    sz.conv_w = c.conv_dim();
    sz.iscore_rows = QSA_SCORE_ROWS; sz.ilist_w = QSA_LIST;
    sz.ihist = opt_.max_pos / 4 + 4 <= 65536 + 2048;
    alloc_common(dev, sz);
    for (int ci = 1; ci < mc_alloc_; ++ci) if (c.ple_layer >= 0) dev.ple_x.push_back(dev.alloc<float>((size_t) R * std::max(1, c.ple_n_heads() * c.ple_dim)));
    dev.res2 = dev.alloc<float>((size_t) R * hcn);
    dev.gate = dev.alloc<float>((size_t) R * hcn);
    dev.lo = dev.alloc<float>((size_t) R * c.hc_lr);
    dev.inj = dev.alloc<float>((size_t) mc_alloc_ * R * 4);   // (a slot per chunk of a multi-chunk pass)
    dev.injp = dev.alloc<float>((size_t) MAX_NT * c.hc * 4);
    dev.inj_cnt = dev.alloc<unsigned>(MAX_NT);
    CUDA_CHECK(cudaMemset(dev.inj_cnt, 0, MAX_NT * sizeof(unsigned)));
    dev.mixed = dev.alloc<float>((size_t) R * n);
    dev.ple_emb = dev.alloc<float>((size_t) R * std::max(1, c.ple_n_heads() * c.ple_dim));
    dev.ple_key = dev.alloc<float>((size_t) R * hcn);
    dev.ple_val = dev.alloc<float>((size_t) R * n);
    dev.ple_sc = dev.alloc<float>((size_t) R * c.hc * 4);
    dev.iq = dev.alloc<float>((size_t) R * c.idx_n_head * 128);
    dev.ik = dev.alloc<float>((size_t) R * 128);
    dev.iqn = dev.alloc<float>((size_t) R * c.idx_n_head * 128);
    dev.mpos = dev.alloc<int>(1);
    if (c.rope_sec && !getenv("HYPER4_NO_MROPE")) {   // (the cells MTP drafts reach past max_pos included; the switch: plain rope, text only)
        dev.rpos = dev.alloc<int>((size_t) rpos_cells() * 3);
        rpos_fill(dev.rpos, 0, rpos_cells(), 0, dev.stream);
    }
    if (mtp_g_) {
        dev.mres = dev.alloc<float>((size_t) R * hcn);
        dev.mh = dev.alloc<float>((size_t) hcn);
        dev.ecat = dev.alloc<float>((size_t) R * c.hc * 2 * n);
    }
    if (getenv("HYPER4_TRACE")) {
        dev.trace = dev.alloc<float>((size_t) c.n_layer * hcn);
        dev.trace2 = dev.alloc<float>((size_t) TRACE2_SLOTS * std::max<size_t>(hcn, (size_t) c.ple_n_heads() * c.ple_dim));
    }
}

void Engine4::after_load() {
    const Q4Config & c = cfg_;
    // the PLE n-gram table in RAM (2 MB pages): its rows are read at random, 16 per token, and read through the mmapped file a
    // cold page cache made every row a disk read (prefill ~550 t/s until the file was cached, and again whenever other files
    // pushed it out); HYPER4_PLE_MMAP=1 keeps reading the file
    if (c.ple_layer >= 0 && !getenv("HYPER4_PLE_MMAP")) {
        const GTensor & pt = gguf_->need("per_layer_token_embd.weight");
        const size_t bytes = pt.nbytes, step = 64u << 20;
        auto tp = std::chrono::steady_clock::now();
        uint8_t * buf = (uint8_t *) host_huge_alloc(bytes);
#pragma omp parallel for schedule(dynamic)
        for (size_t o = 0; o < bytes; o += step) memcpy(buf + o, pt.data + o, std::min(step, bytes - o));
        collapse_huge(buf, bytes);
        ple_tab_ = buf;
        fprintf(stderr, "hyper4: PLE table (%.2f GiB) in RAM in %.1f s\n", bytes / 1073741824.0,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
    }
}

// kind 0: main model; 1: MTP block over (token t+1, main hidden row t); 2: MTP chained on its own last hidden row
void Engine4::record_main(int gi, int nt, int kind) {
    const Q4Config & c = cfg_;
    Device & d = dev(gi);
    cudaStream_t s = d.stream;
    const int n = c.n_embd, hc = c.hc, hcn = c.hc_dim(), lr = c.hc_lr, K = c.n_expert_used, bs = d.big_stride;
    const int nd = opt_.n_devices;
    const float eps = c.rms_eps;
    bool bulk = nt > MAX_NT;   // (an MTP pass continues its last row in decode mode)
    d.pe_pending = false;
    // MTP rows that only enter the block's cache (its K / V and indexer keys depend on its input alone): prompt rows, and
    // all but the last of the kept verification rows; the last one alone goes on through the FFN to the head
    const bool cache_only = kind == 1 && mtp_cache_only_ && !mtp_full_;
    const bool last_only = kind == 1 && !mtp_cache_only_ && !mtp_full_ && nt > 1;
    // ... and, already from the attention output on, alone: the other rows only write their K/V (HYPER4_MTP_ATT_ALL: attention,
    // output projection and allreduce for every row, the last one kept -- the same values for it)
    static const bool att_all = getenv("HYPER4_MTP_ATT_ALL") != nullptr;
    const bool last_att = last_only && !att_all;   // (also a prefill tail: its MTP pass attended over the whole cache for every row)
    // matmul: decode GEMV (split-K for few rows) or, for a prefill chunk, fp16 conversion + tiled tensor-core GEMM
    const DenseRun dr{s, d.blas, d.xh, d.ggs, d.ggs_elems};
    auto mm = [&](const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni = NormIn{}) {
        dense_mm(dr, W, x, xs, y, ys, rows, ni);
    };
    auto mmo = [&](const Q8W & W, const float * x, int xs, float * y, int ys, int rows) {   // LM head (Q8_0)
        if (rows <= MAX_NT) { gemv_q8(W, x, xs, y, ys, nullptr, rows, s, NormIn{}); return; }
        to_half(x, xs, nullptr, W.k, 0.0f, d.xh, rows, s);
        gemm_q8(W, d.xh, rows, y, ys, nullptr, s);
    };
    // fp32 weights (routers, alpha/beta, injections): warp GEMV per token, or cuBLAS sgemm for a prefill chunk
    auto f32mm = [&](const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nr) {
        if (nr <= MAX_NT) { gemv_f32(W, rows, k, x, xs, y, ys, nr, s); return; }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, nr, k, &one, W, k, x, xs, &zero, y, ys) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemm failed");
    };
    // chunks of this pass (a multi-chunk prefill runs each layer for all of them before the next layer): residual rows,
    // positions, embeddings and PLE rows per chunk; everything else is per-layer scratch
    struct Ck { int nt; float * R; int * P; int pos; float * ple; };
    const int nck = !kind && bulk ? mc_n_ : 1;
    std::vector<Ck> ck(nck);
    for (int ci = 0; ci < nck; ++ci) {
        ck[ci].nt = ci == 0 ? nt : mc_nt_[ci];
        ck[ci].R = ci == 0 ? (kind ? d.mres : d.res) : d.res_x[ci - 1];
        ck[ci].P = ci == 0 ? (kind ? d.mpos : d.pos) : d.pos_x + ci;
        ck[ci].pos = ci == 0 ? h_pos_[kind ? 1 : 0] : mc_pos_[ci];
        ck[ci].ple = ci == 0 || d.ple_x.empty() ? d.ple_emb : d.ple_x[ci - 1];
    }
    float * R = ck[0].R;   // hc-wide residual rows
    int * P = ck[0].P;
    int pos_host = ck[0].pos;
    float * ple_emb = ck[0].ple;
    float * injc = d.inj;   // the chunk's injection logits
    const size_t pe_row = (size_t) std::max(1, c.ple_n_heads() * c.ple_dim);
    for (int ci = 0; ci < nck; ++ci) {
        if (nck > 1) CUDA_CHECK(cudaMemcpyAsync(ck[ci].P, h_cpos_ + ci, sizeof(int), cudaMemcpyHostToDevice, s));
        else CUDA_CHECK(cudaMemcpyAsync(P, h_pos_ + (kind ? 1 : 0), sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d.x, kind ? h_membd_ : h_embd_ + (size_t) ci * R4 * n, (size_t) ck[ci].nt * n * sizeof(float),
                                   cudaMemcpyHostToDevice, s));
        if (!kind && c.ple_layer >= 0)
            CUDA_CHECK(cudaMemcpyAsync(ck[ci].ple, h_ple_ + (size_t) ci * R4 * pe_row, (size_t) ck[ci].nt * pe_row * sizeof(float),
                                       cudaMemcpyHostToDevice, s));
        if (!kind) hc_init(ck[ci].R, d.x, n, hc, ck[ci].nt, s);
    }
    incr_counter(d.counter, s);
    // HYPER4_TRACE (single-chunk passes): a stage's last row into trace2 slot `slot`
    const size_t tr_w = std::max<size_t>(hcn, pe_row);
    auto tr = [&](int slot, const float * buf, size_t width) {
        if (!d.trace2 || kind || nck != 1) return;
        CUDA_CHECK(cudaMemcpyAsync(d.trace2 + (size_t) slot * tr_w, buf + (size_t) (nt - 1) * width, width * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
    };
    if (c.ple_layer >= 0) tr(0, ck[0].ple, pe_row);
    tr(1, ck[0].R, hcn);
    // short chunks: the CPU computes its experts (the MTP block's FFN never runs over a whole chunk)
    const bool streaming = bulk && opt_.stream_experts && nt >= stream_min_ && (!kind || mtp_full_);
    if (streaming) stream_begin(d, kind ? c.n_layer : 0, kind ? c.n_layer + 1 : c.n_layer);
    int call = 0;
    auto dbg = [&](const char * what, int il, const float * buf, size_t cnt) { TpEngine::dbg(d, what, il, buf, cnt, 2, nt); };
    dbg("embed", -1, d.res, (size_t) nt * hcn);
    // hyper-connection mixer: res -> mixed (and the inject logits)
    auto hc_mix = [&](const float * norm, const DW & down, const DW & up, const float * inj, const float * res = nullptr, int rows = -1) {
        if (!res) res = R;
        if (rows < 0) rows = nt;
        if (c.plain) { rmsnorm(res, n, norm, d.mixed, n, n, rows, eps, s); return; }   // (qwen35moe: the block's RMSNorm)
        const bool finj = inj && rows <= MAX_NT;   // decode: the injection dots ride along the norm kernel
        hc_norm(res, norm, d.xn, n, hc, eps, rows, s, finj ? inj : nullptr, d.injp, injc, d.inj_cnt);
        mm(down, d.xn, hcn, d.lo, lr, rows);
        if (inj && !finj) f32mm(inj, hc, hcn, d.xn, hcn, injc, 4, rows);
        if (rows <= MAX_NT) {
            NormIn act; act.act = 1; act.act_scale = 1.0f / hc;   // silu(lo / hc) on load
            if (up.hcperm) { gemv_q8_hcmix(up.q8, d.lo, lr, act, d.xn, hcn, d.mixed, n, n, rows, s); return; }   // mixing fused
            mm(up, d.lo, lr, d.gate, hcn, rows, act);
        } else {
            silu_scale(d.lo, lr, 1.0f / hc, rows, lr, s);
            mm(up, d.lo, lr, d.gate, hcn, rows);
        }
        hc_mixed(d.xn, d.gate, d.mixed, n, hc, rows, s, up.hcperm);
    };
    if (kind) {   // [rms(e) * enorm | rms(h_s) * hnorm_s] -> eh_proj, one row per hc stream
        const float * mH = kind == 1 ? (mtp_src_ > 0 ? d.res_x[mtp_src_ - 1] : d.res) : d.mh;
        if (c.plain && kind == 1) {   // qwen35moe: the MTP block takes the output-normed hidden (the chained one: shared_head_norm's)
            rmsnorm(mH, n, d.head_norm, d.mres, n, n, nt, eps, s);
            mH = d.mres;
        }
        mtp_prep(d.x, d.m_enorm, mH, kind == 1 ? hcn : 0, d.m_hnorm, eps, n, hc, mtp_whole_norm_, d.ecat, nt, s);
        const int blk = std::max(1, R4 * std::max(hcn, c.n_ff_shexp) / (2 * n));   // rows the fp16 scratch holds
        for (int r0 = 0; r0 < nt * hc; r0 += blk) {
            const int rr = std::min(blk, nt * hc - r0);
            mm(d.m_eh, d.ecat + (size_t) r0 * 2 * n, 2 * n, d.mres + (size_t) r0 * n, n, rr);
        }
    }
    const int il0 = kind ? c.n_layer : 0, il1 = kind ? c.n_layer + 1 : c.n_layer;
    const bool snap = !kind && !bulk && nt > 1 && mtp_g_;   // verification: keep the state after every row for rollback
    auto set_ck = [&](int ci) {   // (a single chunk is set once: an MTP pass narrows nt mid-layer)
        if (nck <= 1) return;
        nt = ck[ci].nt; R = ck[ci].R; P = ck[ci].P; pos_host = ck[ci].pos; ple_emb = ck[ci].ple; injc = d.inj + (size_t) ci * R4 * 4;
    };
    // decode: independent work on a second stream (indexer || q/k/v + attention prep, alpha/beta || the GDN input
    // projection, shared expert || routed experts): the small kernels' fixed latencies overlap. The second stream's
    // split-K GEMVs use their own scratch slot. HYPER4_SERIAL_SHEXP: one stream
    static const bool serial_shexp = getenv("HYPER4_SERIAL_SHEXP") != nullptr;
    auto fork = [&] { CUDA_CHECK(cudaEventRecord(d.ev_fork, s)); CUDA_CHECK(cudaStreamWaitEvent(d.s2, d.ev_fork, 0)); };
    auto on_s2 = [&](auto && f) {
        const cudaStream_t sm = s;
        s = d.s2;
        gemv_scratch_slot(1);
        f();
        gemv_scratch_slot(0);
        CUDA_CHECK(cudaEventRecord(d.ev_join, s));
        s = sm;
    };
    auto join = [&] { CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_join, 0)); };
    // a prefill pass exchanges the partial sums of chunk c (copy stream) while computing chunk c + 1: every layer runs in
    // three phases over the chunks -- the token mixer (ar_begin, a host barrier, ar_fetch), the mixer's sum + the MoE (ar_sum,
    // then the same hand-over), the MoE's sum (ar_sum). Decode rows keep the one-kernel LL allreduce; an MTP pass has one
    // chunk, so its order is as before.
    for (int il = il0; il < il1; ++il) {
    DevLayer & L = kind ? d.mtp : d.layers[il];
    for (int ci = 0; ci < nck; ++ci) {   // ---- phase A: the token mixer ----
        set_ck(ci);
        if (L.ple) {
            const int pe = c.ple_n_heads() * c.ple_dim;
            mm(L.ple_key, ple_emb, pe, d.ple_key, hcn, nt);
            mm(L.ple_value, ple_emb, pe, d.ple_val, n, nt);
            ple_apply(R, d.ple_key, d.ple_val, L.ple_wk, L.ple_wq, L.ple_wc, L.ple_conv, L.ple_state, snap ? L.ple_snap : nullptr, n, hc, c.ple_conv,
                      c.ple_ngram, eps, nt, d.ple_sc, s);
        }
        if (L.ple) dbg("ple", il, R, (size_t) nt * hcn);
        if (il < 2) tr(2 + 4 * il, R, hcn);
        // ---- token mixer ----
        if (d.pe_pending) { CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_pe1, 0)); d.pe_pending = false; }   // (it reads the mixed rows)
        hc_mix(L.hca_norm, L.hca_down, L.hca_up, L.hca_inj);
        if (il < 2) tr(3 + 4 * il, d.mixed, n);
        dbg("hc_mix_attn", il, d.mixed, (size_t) nt * n);
        const bool par = !bulk && !serial_shexp;
        if (L.full && L.n_head_l == 0) {   // no kv head on this GPU: contributes nothing to the attention output
            CUDA_CHECK(cudaMemsetAsync(d.part, 0, (size_t) (last_att ? 1 : nt) * n * sizeof(float), s));
        } else if (L.full) {
            const bool qsa = L.kraw != nullptr;
            const bool par_idx = par && qsa;
            if (par_idx) fork();
            static const bool idx_late = getenv("HYPER4_IDX_LATE") != nullptr;
            if (!par_idx || idx_late) mm(L.wqkv, d.mixed, n, d.big0, bs, nt);
            const int top = getenv("HYPER4_NOQSA") ? (1 << 28) : getenv("HYPER4_TOP") ? atoi(getenv("HYPER4_TOP")) : c.idx_top_k / 4;   // experiments
            auto indexer = [&] {   // indexer: cells each token attends to (all of them while the context has <= top complete blocks)
                if (bulk) {
                    to_half(d.mixed, n, nullptr, n, 0.0f, d.xh, nt, s);
                    gemm_f16(L.idx_q, d.xh, nt, d.iq, c.idx_n_head * 128, nullptr, s);
                    gemm_f16(L.idx_k, d.xh, nt, d.ik, 128, nullptr, s);
                } else {
                    gemv_bf16(L.idx_q, d.mixed, n, d.iq, c.idx_n_head * 128, nullptr, nt, s);
                    gemv_bf16(L.idx_k, d.mixed, n, d.ik, 128, nullptr, nt, s);
                }
                static const bool sep_pool = getenv("HYPER4_SEP_POOL") != nullptr;
                if (nt == 1 && !sep_pool)
                    idx_prep_pool(d.iq, d.ik, L.idx_qn, d.iqn, L.kraw, L.kpool, L.idx_kn, P, c.idx_n_head, c.n_rot, c.rope_base, eps, s,
                                  d.rpos, c.rope_sec);
                else {
                    idx_prep(d.iq, d.ik, L.idx_qn, d.iqn, L.kraw, P, c.idx_n_head, c.n_rot, c.rope_base, eps, nt, s, d.rpos, c.rope_sec);
                    idx_pool(L.kraw, L.kpool, L.idx_kn, P, nt, c.n_rot, c.rope_base, eps, s, d.rpos, c.rope_sec);
                }
                if (cache_only) return;
                static const bool old_sel = getenv("HYPER4_OLDSEL") != nullptr;
                if (d.ihist && !old_sel)
                    idx_select_hist(d.iqn, L.kpool, P, nt, c.idx_n_head, top, d.iscores, opt_.max_pos / 4 + 4, QSA_SCORE_ROWS, d.ihist,
                                    d.ilist, QSA_LIST, d.ilist_n, s);
                else
                    idx_select(d.iqn, L.kpool, P, nt, c.idx_n_head, top, d.iscores, opt_.max_pos / 4 + 4, QSA_SCORE_ROWS, d.ilist,
                               QSA_LIST, d.ilist_n, s);
            };
            // the indexer chain (projections, pooling, scoring, selection) is the longer branch: enqueued first so its kernels
            // are scheduled ahead of the q/k/v projection's blocks
            if (par_idx) on_s2(indexer);
            else if (qsa) indexer();
            if (par_idx && !idx_late) mm(L.wqkv, d.mixed, n, d.big0, bs, nt);
            if (par_idx) {   // attention prep runs meanwhile (q/k norms, rope, the cache rows): it needs only q/k/v
                attn_prep(d.big0, bs, L.q_norm, L.k_norm, L.kcache, L.vcache, P, opt_.max_pos, L.n_head_l, L.n_kv_l, c.head_dim,
                          c.n_rot, c.rope_base, eps, nt, s, d.rpos, c.rope_sec);
                join();
            }
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(s, &cap);
            if (qsa && cap == cudaStreamCaptureStatusNone && getenv("HYPER4_DUMPSEL") && atoi(getenv("HYPER4_DUMPSEL")) == il && d.g == 0) {   // debugging: last token's cells
                CUDA_CHECK(cudaStreamSynchronize(s));
                int ln = 0;
                CUDA_CHECK(cudaMemcpy(&ln, d.ilist_n + nt - 1, sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<int> lst(ln);
                CUDA_CHECK(cudaMemcpy(lst.data(), d.ilist + (size_t) (nt - 1) * QSA_LIST, ln * sizeof(int), cudaMemcpyDeviceToHost));
                FILE * f = fopen("/tmp/hyper_sel.bin", "wb");
                fwrite(lst.data(), 4, ln, f);
                fclose(f);
                fprintf(stderr, "DUMPSEL layer %d: %d cells\n", il, ln);
            }
                        // a prefill chunk entirely below the sparse regime keeps the (identical, faster) dense flash attention
            const bool dense_chunk = !qsa || (pos_host + nt) / 4 <= top;
            if (!par_idx)
                attn_prep(d.big0, bs, L.q_norm, L.k_norm, L.kcache, L.vcache, P, opt_.max_pos, L.n_head_l, L.n_kv_l, c.head_dim,
                          c.n_rot, c.rope_base, eps, nt, s, d.rpos, c.rope_sec);
            if (cache_only) continue;
            const int ostride = L.n_head_l * c.head_dim;
            if (last_att)   // the last row alone (position P + nt - 1), into row 0
                attn_split(d.big0 + (size_t) (nt - 1) * bs, bs, L.kcache, L.vcache, d.attn_part, d.o, ostride, P, opt_.max_pos, L.n_head_l,
                           L.n_kv_l, L.head_off, c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), 1, s,
                           nullptr, 0, nullptr, nt - 1);
            else if (bulk && dense_chunk)
                attn_prefill(d.big0, bs, L.kcache, L.vcache, d.o, ostride, P, opt_.max_pos, L.n_head_l, L.head_off,
                             c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s);
            else if (getenv("HYPER4_OLDATTN"))
                attn_decode(d.big0, bs, L.kcache, L.vcache, d.o, ostride, P, opt_.max_pos, L.n_head_l, L.n_kv_l, L.head_off,
                            c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s);
            else
                attn_split(d.big0, bs, L.kcache, L.vcache, d.attn_part, d.o, ostride, P, opt_.max_pos, L.n_head_l, L.n_kv_l,
                           L.head_off, c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s,
                           qsa ? d.ilist : nullptr, QSA_LIST, qsa ? d.ilist_n : nullptr);
            mm(L.wo, d.o, ostride, d.part, n, last_att ? 1 : nt);
        } else {
            const int dv = c.head_v_dim();
            const int z_off = L.conv_ch, ab_off = L.conv_ch + L.n_v_l * dv;
            if (par) {   // alpha / beta (other columns of big0) next to the input projection
                fork();
                on_s2([&] { f32mm(L.ab_w, 2 * L.n_v_l, n, d.mixed, n, d.big0 + ab_off, bs, nt); });
                mm(L.win, d.mixed, n, d.big0, bs, nt);
                join();
            } else {
                mm(L.win, d.mixed, n, d.big0, bs, nt);
                f32mm(L.ab_w, 2 * L.n_v_l, n, d.mixed, n, d.big0 + ab_off, bs, nt);
            }
            gdn_conv(d.big0, bs, L.conv_state, snap ? L.conv_snap : nullptr, L.conv_w, L.conv_ch, c.ssm_conv, nt, s, bulk ? d.conv_raw : nullptr);
            const int ostride = L.n_v_l * dv;
            gdn_step(d.big0, bs, ab_off, L.state, snap ? L.state_snap : nullptr, d.o, ostride, L.dt_bias, L.ssm_a, L.n_k_l, L.n_v_l, c.ssm_d_state, dv, eps, nt, s);
            if (c.plain) gated_norm(d.o, ostride, d.big0 + z_off, bs, L.ssm_norm, L.n_v_l, dv, eps, nt, s);   // (qwen35moe: silu gate)
            else gated_norm_sigmoid(d.o, ostride, d.big0 + z_off, bs, L.ssm_norm, L.n_v_l, dv, eps, nt, s);
            mm(L.wout, d.o, ostride, d.part, n, nt);
        }
        if (cache_only) continue;
        if (last_att && L.full) {   // the last row's residual streams and injection weights to row 0: it goes on alone, in decode mode
            CUDA_CHECK(cudaMemcpyAsync(R, R + (size_t) (nt - 1) * hcn, (size_t) hcn * sizeof(float), cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(injc, injc + (size_t) (nt - 1) * 4, 4 * sizeof(float), cudaMemcpyDeviceToDevice, s));
            nt = 1;
            bulk = false;
        }
        dbg(L.full ? "attn_part" : "gdn_part", il, d.part, (size_t) nt * n);
        if (bulk && ar_sync_) {   // (tests: the old order, one exchange at a time)
            ar_begin(d, nt, ci, 0); barrier_->wait(); ar_end(d, nt, ci, 0);
            if (c.plain) vec_add(R, d.bo, (size_t) nt * n, s); else hc_combine(R, d.bo, injc, 4, n, hc, nt, s);
        } else if (bulk) { ar_begin(d, nt, ci, 0); barrier_->wait(); ar_fetch(d, nt, ci, 0); }   // (the sum lands in phase B)
        else if (c.plain) allreduce_add_ll16(R, d.part, d.ll, d.g, nd, nt * n, d.counter, call++, s, nullptr);   // R += sum of the parts
        else allreduce_hc_ll16(R, injc, n, hc, d.part, d.ll, d.g, nd, nt * n, d.counter, call++, s);
    }
    for (int ci = 0; ci < nck; ++ci) {   // ---- phase B: the mixer's sum, the MoE ----
        set_ck(ci);
        const bool last_ck = ci == nck - 1;   // (the streamed experts' staging is released after the last chunk)
        if (cache_only) continue;
        const bool par = !bulk && !serial_shexp;
        if (bulk && !ar_sync_) {
            ar_sum(d, nt, ci, 0);
            if (c.plain) vec_add(R, d.bo, (size_t) nt * n, s); else hc_combine(R, d.bo, injc, 4, n, hc, nt, s);
        }
        if (il < 2) tr(4 + 4 * il, R, hcn);
        if (last_only && nt > 1) {   // the last row moves to row 0 and continues alone, in decode mode
            CUDA_CHECK(cudaMemcpyAsync(R, R + (size_t) (nt - 1) * hcn, (size_t) hcn * sizeof(float), cudaMemcpyDeviceToDevice, s));
            nt = 1;
            bulk = false;
        }
        // ---- MoE ----
        hc_mix(L.hcf_norm, L.hcf_down, L.hcf_up, L.hcf_inj);
        if (il < 2) tr(5 + 4 * il, d.mixed, n);
        auto shexp = [&] {
            mm(L.sh_gu, d.mixed, n, d.shgu, 2 * L.n_sh_l, nt);
            if (nt <= MAX_NT) {   // silu(gate) * up on load
                NormIn glu; glu.act = 2; glu.glu_off = L.n_sh_l;
                mm(L.sh_down, d.shgu, 2 * L.n_sh_l, d.shpart, n, nt, glu);
            } else {
                silu_mul(d.shgu, 2 * L.n_sh_l, d.shh, c.n_ff_shexp, L.n_sh_l, nt, s);
                mm(L.sh_down, d.shh, c.n_ff_shexp, d.shpart, n, nt);
            }
        };
        if (par) { fork(); on_s2(shexp); }   // (the shared expert || routing and the routed experts, until moe_reduce)
        if (!L.router16) f32mm(L.router, c.n_expert + 1, n, d.mixed, n, d.rlog, c.n_expert + 1, nt);
        else if (nt <= MAX_NT) gemv_bf16w(L.router16, c.n_expert + 1, n, d.mixed, n, d.rlog, c.n_expert + 1, nt, s);
        else {   // prefill: widened to fp32 for the same sgemm
            bf16_to_f32(L.router16, (size_t) (c.n_expert + 1) * n, d.router32, s);
            f32mm(d.router32, c.n_expert + 1, n, d.mixed, n, d.rlog, c.n_expert + 1, nt);
        }
        dbg("router", il, d.rlog, (size_t) nt * (c.n_expert + 1));
        MoeLayerDev & X = place_->dev(d.g, il);   // the layer's experts on this device
        const bool stream = streaming && X.owner_bulk;
        static const bool sep_pub = getenv("HYPER4_SEP_PUBLISH") != nullptr;
        // decode on GPU 0: routing and the CPU record in one launch
        const bool fused_pub = !bulk && d.g == 0 && !sep_pub &&
            moe_route_publish(d.rlog, c.n_expert + 1, c.n_expert, K, d.ids, d.wts, d.sg, nt, &cpu_rec_[il].seq, &cpu_rec_[il].nt,
                              &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0], d.mixed, n, n, d.counter,
                              (unsigned) il, d.ticket, s);
        if (!fused_pub) moe_route(d.rlog, c.n_expert + 1, c.n_expert, K, d.ids, d.wts, d.sg, nt, s);
        dbg("route_w", il, d.wts, (size_t) nt * K);
        if (pred_mode_ && !bulk && !kind && d.g == 0 && nt <= MAX_NT && il + 1 < c.n_layer) {
            // HYPER4_PRED: the next layer's routing from this layer's MoE input (its router on the mixed rows), on the side stream
            // while this layer's experts run; published as the next job's prediction, which the CPU reads ahead of its record
            const DevLayer & N = d.layers[il + 1];
            cudaStream_t ps = d.pstream;
            CUDA_CHECK(cudaEventRecord(d.ev_pe0, s));
            CUDA_CHECK(cudaStreamWaitEvent(ps, d.ev_pe0, 0));
            if (N.router16) gemv_bf16w(N.router16, c.n_expert + 1, n, d.mixed, n, d.plog, c.n_expert + 1, nt, ps);
            else gemv_f32(N.router, c.n_expert + 1, n, d.mixed, n, d.plog, c.n_expert + 1, nt, ps);
            moe_route(d.plog, c.n_expert + 1, c.n_expert, K, d.pids, d.pwts, d.psg, nt, ps);
            moe_publish(&cpu_pred_[il + 1].seq, &cpu_pred_[il + 1].nt, &cpu_pred_[il + 1].ids[0][0], &cpu_pred_[il + 1].wts[0][0],
                        &cpu_pred_[il + 1].x[0][0], d.mixed, n, 0, d.pids, d.pwts, K, nt, d.counter, (unsigned) (il + 1), ps);
            CUDA_CHECK(cudaEventRecord(d.ev_pe1, ps));
            d.pe_pending = true;
        }
        if (bulk && !kind && d.g == 0 && place_->adapt())   // prompt routing for the adaptive placement
            CUDA_CHECK(cudaMemcpyAsync(h_ids_ + ((size_t) il * mc_max_ + ci) * R4 * K, d.ids, (size_t) nt * K * sizeof(int),
                                       cudaMemcpyDeviceToHost, s));
        if (d.g == 0 && !stream && !fused_pub) {
            if (bulk) moe_publish(&cpu_bulk_->seq, &cpu_bulk_->nt, &cpu_bulk_->ids[0][0], &cpu_bulk_->wts[0][0], &cpu_bulk_->x[0][0],
                                  d.mixed, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
            else moe_publish(&cpu_rec_[il].seq, &cpu_rec_[il].nt, &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0],
                             d.mixed, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
        }
        if (!par) shexp();
        if (bulk) {   // local pairs grouped by expert: consecutive blocks share the expert's weights in L2
            moe_order(X.moe, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
            to_half(d.mixed, n, nullptr, n, 0.0f, d.mix16, nt, s);
            moe_gemm_gate_up(X.moe, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
            moe_gemm_down(X.moe, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
            if (stream) {   // this GPU's share of the CPU experts, uploaded into staging buffer il % 2
                const int sb = il & 1;
                MoeDev ms = X.moe;
                const int cnt = (int) X.st_list[0].size();
                ms.gate = d.stage[sb];
                ms.up = d.stage[sb] + (size_t) cnt * X.moe.gate_bytes;
                ms.down = d.stage[sb] + (size_t) 2 * cnt * X.moe.gate_bytes;
                ms.slot = X.st_slot[0];
                while (!d.up_pend.empty() && d.up_pend.front()[0] <= il) {   // (parts no allreduce took yet)
                    const auto q = d.up_pend.front();
                    d.up_pend.pop_front();
                    place_->upload_stage(d.g, q[0], q[1], q[2], up_parts_);
                }
                CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_up[sb], 0));
                moe_order(ms, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
                moe_gemm_gate_up(ms, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
                moe_gemm_down(ms, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
                if (last_ck) stream_next(d, il, sb, kind ? c.n_layer + 1 : c.n_layer);
            }
        } else if (nt > 1 && grouped_decode_) {   // verification rows: each local expert read once (grouped GEMM)
            moe_order(X.moe, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
            to_half(d.mixed, n, nullptr, n, 0.0f, d.mix16, nt, s);
            moe_gemm_gate_up(X.moe, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
            moe_gemm_down(X.moe, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
        } else {
            moe_gate_up(X.moe, d.mixed, n, d.ids, K, d.hexp, nt, s);
            moe_down(X.moe, d.hexp, d.ids, d.wts, K, d.yexp, nt, s);
        }
        if (par) join();
        dbg("shexp", il, d.shpart, (size_t) nt * n);
        dbg("experts", il, d.yexp, (size_t) nt * K * n);
        const volatile unsigned * cflag = d.g == 0 && !nocpu_ && !stream ? (bulk ? &cpu_bulk_out_->seq : &cpu_out_[il].seq) : nullptr;
        moe_reduce(d.shpart, d.sg, d.yexp, K, d.part, n, nt, d.ids, stream ? X.owner_bulk : X.owner, d.g, CPU_OWNER, cflag,
                   bulk ? &cpu_bulk_out_->y[0][0] : &cpu_out_[il].y[0][0], d.counter, (unsigned) il, s);
        if (moe_wait_ && !bulk && d.g != 0 && !nocpu_ && !stream)   // (the peers: one thread on the CPU's flag, then the allreduce)
            moe_wait_cpu(d.ids, X.owner, K, nt, CPU_OWNER, &cpu_out_[il].seq, d.counter, (unsigned) il, s);
        dbg("moe_part", il, d.part, (size_t) nt * n);
        if (bulk && ar_sync_) {
            ar_begin(d, nt, ci, 1); barrier_->wait(); ar_end(d, nt, ci, 1);
            if (c.plain) vec_add(R, d.bo, (size_t) nt * n, s); else hc_combine(R, d.bo, injc, 4, n, hc, nt, s);
        } else if (bulk) { ar_begin(d, nt, ci, 1); barrier_->wait(); ar_fetch(d, nt, ci, 1); }
        else if (c.plain) allreduce_add_ll16(R, d.part, d.ll, d.g, nd, nt * n, d.counter, call++, s, nullptr);
        else allreduce_hc_ll16(R, injc, n, hc, d.part, d.ll, d.g, nd, nt * n, d.counter, call++, s);
    }
    for (int ci = 0; ci < nck; ++ci) {   // ---- phase C: the MoE's sum ----
        set_ck(ci);
        const bool last_ck = ci == nck - 1;
        if (cache_only) continue;
        if (bulk && !ar_sync_) {
            ar_sum(d, nt, ci, 1);
            if (c.plain) vec_add(R, d.bo, (size_t) nt * n, s); else hc_combine(R, d.bo, injc, 4, n, hc, nt, s);
        }
        dbg("l_last", il, R, (size_t) nt * hcn);
        if (d.trace && !kind && last_ck)
            CUDA_CHECK(cudaMemcpyAsync(d.trace + (size_t) il * hcn, R + (size_t) (nt - 1) * hcn, (size_t) hcn * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
    }
    }
    if (kind) {   // drafts: the last row only; its residual feeds the next chained step
        if (cache_only) return;
        if (!c.plain) CUDA_CHECK(cudaMemcpyAsync(d.mh, d.mres + (size_t) (nt - 1) * hcn, (size_t) hcn * sizeof(float), cudaMemcpyDeviceToDevice, s));
        hc_mix(d.m_head_norm, d.m_head_down, d.m_head_up, nullptr, d.mres + (size_t) (nt - 1) * hcn, 1);
        if (c.plain) CUDA_CHECK(cudaMemcpyAsync(d.mh, d.mixed, (size_t) n * sizeof(float), cudaMemcpyDeviceToDevice, s));   // (the chained input)
        mm(d.output, d.mixed, n, d.logits, d.output.n(), 1);
        argmax_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.wts, 1, s);
        max_sumexp(d.logits, d.output.n(), d.wts + 2, s);   // (the best token's probability: the drafts' gate)
        topk_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.topk, TOPK, 1, s);   // (sampled drafts' candidates)
        CUDA_CHECK(cudaMemcpyAsync(h_mtopk_ + (size_t) gi * TOPK * 2, d.topk, (size_t) TOPK * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(h_mres_ + (size_t) gi * 4, d.wts, 4 * sizeof(float), cudaMemcpyDeviceToHost, s));
        return;
    }
    // final mixer = output norm (prefill chunk: last row only)
    const int hr = bulk && !allrows_ ? 1 : nt;   // HYPER4_ALLROWS (debugging): logits for every row of a prefill chunk
    hc_mix(d.head_norm, d.head_down, d.head_up, nullptr, R + (size_t) (nt - hr) * hcn, hr);
    mm(d.output, d.mixed, n, d.logits, d.output.n(), hr);
    argmax_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.wts, hr, s);   // wts reused as the result pairs
    CUDA_CHECK(cudaMemcpyAsync(h_res_ + (size_t) gi * MAX_NT * 2, d.wts, (size_t) std::min(hr, MAX_NT) * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
    {   // sampling candidates of the result rows (top-64 of this vocab slice)
        const int tr = std::min(hr, MAX_NT);
        topk_pairs(d.logits + (size_t) (hr - tr) * d.output.n(), d.output.n(), d.output.n(), d.vocab_off, d.topk, TOPK, tr, s);
        CUDA_CHECK(cudaMemcpyAsync(h_topk_ + (size_t) gi * MAX_NT * TOPK * 2, d.topk, (size_t) tr * TOPK * 2 * sizeof(float),
                                   cudaMemcpyDeviceToHost, s));
    }
}

Engine4::DevLayer & Engine4::lay(Device & d, int il) { return il < cfg_.n_layer ? d.layers[il] : d.mtp; }

// the last bulk pass's prompt routing (GPU 0's ids per chunk) into the placement scores, weighted below decode routing
// roll the recurrent state (GDN conv + state, PLE conv) back to the snapshot after row keep-1 of the last verification
void Engine4::record_restore(int gi, int keep) {
    const Q4Config & c = cfg_;
    Device & d = dev(gi);
    for (auto & L : d.layers) {
        if (!L.full) {
            const size_t cs = (size_t) (c.ssm_conv - 1) * L.conv_ch, ssz = (size_t) L.n_v_l * c.ssm_d_state * c.head_v_dim();
            CUDA_CHECK(cudaMemcpyAsync(L.conv_state, L.conv_snap + (keep - 1) * cs, cs * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
            CUDA_CHECK(cudaMemcpyAsync(L.state, L.state_snap + (keep - 1) * ssz, ssz * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
        }
        if (L.ple) {
            const size_t ps = (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim();
            CUDA_CHECK(cudaMemcpyAsync(L.ple_state, L.ple_snap + (keep - 1) * ps, ps * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
        }
    }
}

void Engine4::embed_tok(const int * tokens, int nt, float * dst) {
    const GTensor & te = gguf_->need("token_embd.weight");
    const auto * te_tr = ggml_get_type_traits((ggml_type) te.type);
    for (int t = 0; t < nt; ++t) {
        const uint8_t * row = te.data + (size_t) tokens[t] * te.row_bytes();
        float * out = dst + (size_t) t * cfg_.n_embd;
        if (te.type == GType::F32) memcpy(out, row, cfg_.n_embd * sizeof(float));
        else te_tr->to_float(row, out, cfg_.n_embd);
    }
}

// an image's positions (this request's prompt): the vision encoder's rows instead of the image token's embedding
// an image's positions (this request's prompt): the vision encoder's rows instead of the image token's embedding
void Engine4::image_rows(int pos, int nt, float * dst) {
    if (mrows_.empty()) return;
    for (int t = 0; t < nt; ++t) {
        const auto it = mrows_.find(pos + t);
        if (it != mrows_.end()) memcpy(dst + (size_t) t * cfg_.n_embd, it->second, cfg_.n_embd * sizeof(float));
    }
}

// token embeddings and the PLE n-gram hash rows, for positions pos..pos+nt-1 (an image's positions hash as its token)
// token embeddings and the PLE n-gram hash rows, for positions pos..pos+nt-1 (an image's positions hash as its token)
void Engine4::embed(const int * tokens, int nt, int pos, int chunk) {
    const Q4Config & c = cfg_;
    if ((int) seq_.size() < pos + nt) seq_.resize(pos + nt, -1);
    for (int t = 0; t < nt; ++t) seq_[pos + t] = tokens[t];
    embed_tok(tokens, nt, h_embd_ + (size_t) chunk * R4 * c.n_embd);
    image_rows(pos, nt, h_embd_ + (size_t) chunk * R4 * c.n_embd);
    if (c.ple_layer < 0) return;
    const GTensor & pt = gguf_->need("per_layer_token_embd.weight");
    const auto * pt_tr = ggml_get_type_traits((ggml_type) pt.type);
    const int ng = c.ple_ngram, nh = c.ple_n_heads(), dim = c.ple_dim;
    for (int t = 0; t < nt; ++t) {
        const int p = pos + t;
        int64_t ctx[8];
        ctx[0] = seq_[p];
        bool cut = false;
        for (int s = 1; s < ng; ++s) {
            const int q = p - s;
            const int64_t tk = cut || q < 0 ? -1 : seq_[q];
            cut = cut || tk < 0 || tk == c.ple_eos;
            ctx[s] = cut ? c.ple_eos : tk;
        }
        float * out = h_ple_ + ((size_t) chunk * R4 + t) * std::max(1, nh * dim);
        for (int ngr = 2; ngr <= ng; ++ngr) {
            uint64_t mixed = (uint64_t) ctx[0] * c.ple_mult[0];
            for (int j = 1; j < ngr; ++j) mixed ^= (uint64_t) ctx[j] * c.ple_mult[j];
            for (int gq = 0; gq < c.ple_heads_per_ngram; ++gq) {
                const int h = (ngr - 2) * c.ple_heads_per_ngram + gq;
                const uint64_t row = mixed % c.ple_vocab[h] + c.ple_offsets[h];
                pt_tr->to_float((ple_tab_ ? ple_tab_ : pt.data) + row * pt.row_bytes(), out + h * dim, dim);
            }
        }
    }
}

// M-RoPE positions of cells [s, P) (the prompt's: images' grid positions; text: the index) and of every cell after it (text
// generation goes on from md->next), on every device before the prompt's forwards
// M-RoPE positions of cells [s, P) (the prompt's: images' grid positions; text: the index) and of every cell after it (text
// generation goes on from md->next), on every device before the prompt's forwards
bool Engine4::has_rpos() const { return !devs_.empty() && dev(0).rpos; }
void Engine4::set_rpos(int s, int P, const PromptMedia * md) {
    for (size_t g = 0; g < devs_.size(); ++g) {
        Device & dp = dev((int) g);
        CUDA_CHECK(cudaSetDevice(dp.id));
        if (md && P > s)
            CUDA_CHECK(cudaMemcpyAsync(dp.rpos + 3 * (size_t) s, md->pos3.data() + 3 * (size_t) s, 3 * sizeof(int) * (size_t) (P - s),
                                       cudaMemcpyHostToDevice, dp.stream));
        rpos_fill(dp.rpos, md ? P : s, rpos_cells(), md ? md->next - P : 0, dp.stream);
        CUDA_CHECK(cudaStreamSynchronize(dp.stream));   // (md's vectors are pageable and the caller's)
    }
}

// launch one forward of `kind` over nt rows on every device and wait (hang diagnostics after 10 s)
// MTP pass over chunk ci of the last multi-chunk forward (tokens/nt/pos: that chunk's MTP inputs)
int Engine4::mtp_draft_chunk(const int * tokens, int nt, int pos, int ci, bool need_draft) {
    if (ci < 0 || ci >= std::max(1, mc_last_)) throw std::runtime_error("mtp_draft_chunk: bad chunk");
    mtp_src_ = ci;
    try { const int r = mtp_draft(tokens, nt, pos, need_draft); mtp_src_ = 0; return r; } catch (...) { mtp_src_ = 0; throw; }
}

// MTP over (tokens[t], main hidden row t) at positions pos + t; returns the draft after the last row (need_draft false:
// the rows only enter the block's cache, -1)
int Engine4::mtp_draft(const int * tokens, int nt, int pos, bool need_draft) {
    embed_tok(tokens, nt, h_membd_);
    image_rows(pos + 1, nt, h_membd_);   // (row t holds the token at pos + t + 1)
    h_pos_[1] = pos;
    draft_at(pos + nt + 1);   // (the row at q holds token q + 1 and drafts token q + 2)
    mtp_cache_only_ = !need_draft;
    try { run(1, nt); } catch (...) { mtp_cache_only_ = false; throw; }
    mtp_cache_only_ = false;
    return need_draft ? mtp_result() : -1;
}

// MTP over (token, its own last hidden row) at pos
// MTP over (token at pos, its own last hidden row; the engine counts token positions, this block the hidden's: pos - 1)
int Engine4::mtp_chain(int token, int pos, bool) {
    --pos;
    embed_tok(&token, 1, h_membd_);
    h_pos_[1] = pos;
    draft_at(pos + 2);
    run(2, 1);
    return mtp_result();
}

// ---------------- prompt cache: recurrent-state snapshots (GDN conv + state, PLE conv history) ----------------
// ---------------- prompt cache: recurrent-state snapshots (GDN conv + state, PLE conv history) ----------------
size_t Engine4::snap_floats(int gi) const {
    const Q4Config & c = cfg_;
    size_t n = 0;
    for (auto & L : dev(gi).layers) {
        if (!L.full) n += (size_t) (c.ssm_conv - 1) * L.conv_ch + (size_t) L.n_v_l * c.ssm_d_state * c.head_v_dim();
        if (L.ple) n += (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim();
    }
    return n;
}

void Engine4::snap_copy(std::vector<float *> & hv, bool to_host) {
    const Q4Config & c = cfg_;
    for (size_t gi = 0; gi < devs_.size(); ++gi) {
        Device & d = dev(gi);
        CUDA_CHECK(cudaSetDevice(d.id));
        float * hp = hv[gi];
        auto cp = [&](float * dev, size_t n) {
            if (to_host) CUDA_CHECK(cudaMemcpyAsync(hp, dev, n * sizeof(float), cudaMemcpyDeviceToHost, d.stream));
            else CUDA_CHECK(cudaMemcpyAsync(dev, hp, n * sizeof(float), cudaMemcpyHostToDevice, d.stream));
            hp += n;
        };
        for (auto & L : d.layers) {
            if (!L.full) {
                cp(L.conv_state, (size_t) (c.ssm_conv - 1) * L.conv_ch);
                cp(L.state, (size_t) L.n_v_l * c.ssm_d_state * c.head_v_dim());
            }
            if (L.ple) cp(L.ple_state, (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim());
        }
    }
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->stream)); }
}

// per-position GPU buffers of a conversation (parked to RAM): the attention layers' K / V (head-major, max_pos rows per
// head), raw and pooled indexer keys, and the MTP layer's K / V
// per-position GPU buffers of a conversation (parked to RAM): the attention layers' K / V (head-major, max_pos rows per
// head), raw and pooled indexer keys, and the MTP layer's K / V
std::vector<ConvPark<TpEngine::TpSnap>::Dev> Engine4::park_devs() const {
    std::vector<ConvPark<TpSnap>::Dev> v;
    const size_t hd = (size_t) cfg_.head_dim;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const Device & dp = dev((int) g);
        ConvPark<TpSnap>::Dev pd;
        pd.id = dp.id;
        pd.stream = dp.stream;
        auto attn = [&](const DevLayer & L) {
            if (L.kcache && L.n_kv_l > 0) {
                pd.spans.push_back({L.kcache, hd * sizeof(half), opt_.max_pos * hd * sizeof(half), L.n_kv_l, 1});
                pd.spans.push_back({L.vcache, hd * sizeof(half), opt_.max_pos * hd * sizeof(half), L.n_kv_l, 1});
            }
            if (L.kraw) pd.spans.push_back({L.kraw, 128 * sizeof(half), 0, 1, 1});
            if (L.kpool) pd.spans.push_back({L.kpool, 128 * sizeof(half), 0, 1, 4});
        };
        for (auto & L : dp.layers) attn(L);
        attn(dp.mtp);
        if (dp.rpos) pd.spans.push_back({dp.rpos, 3 * sizeof(int), 0, 1, 1});
        v.push_back(std::move(pd));
    }
    return v;
}

void Engine4::reset() {
    const Q4Config & c = cfg_;
    for (size_t g = 0; g < devs_.size(); ++g) {
        Device & dp = dev((int) g);
        CUDA_CHECK(cudaSetDevice(dp.id));
        for (auto & L : dp.layers) {
            if (!L.full) {
                CUDA_CHECK(cudaMemset(L.conv_state, 0, (size_t) (c.ssm_conv - 1) * L.conv_ch * sizeof(float)));
                CUDA_CHECK(cudaMemset(L.state, 0, (size_t) L.n_v_l * c.ssm_d_state * c.head_v_dim() * sizeof(float)));
            }
            if (L.ple) CUDA_CHECK(cudaMemset(L.ple_state, 0, (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim() * sizeof(float)));
        }
    }
    seq_.clear();
}

// kind 0: main model; 1: MTP block over (token t+1, main hidden row t); 2: MTP chained on its own last hidden row
// the prompt's images (set_media): their rows by position; the cache compares keys (an image's hash at its positions)
const std::vector<int> * Engine4::prompt_key(const std::vector<int> & prompt) {
    md_ = media_;   // (taken first: a request that fails leaves none for the next)
    media_ = nullptr;
    mrows_.clear();
    if (!md_) return nullptr;
    const int P = (int) prompt.size();
    if ((int) md_->keys.size() != P || (int) md_->pos3.size() != 3 * P) throw std::runtime_error("generate: media do not match the prompt");
    for (const auto & r : md_->rows) mrows_[r.first] = r.second;
    return &md_->keys;
}

void Engine4::prompt_begin(const std::vector<int> & prompt, int s) {
    seq_.assign(prompt.begin(), prompt.begin() + s);   // PLE n-gram history
    if (dev(0).rpos) set_rpos(s, (int) prompt.size(), md_);
}

// MTP after a prefill forward: the block consumes (t_{q+1}, h_q) at q for every chunk row; the last pair uses the predicted
// next token, and only the prompt's last row drafts
int Engine4::mtp_prompt(const std::vector<int> & prompt, int c0, const int * lens, int nk, int next) {
    const int P = (int) prompt.size();
    int end = c0;
    for (int ci = 0; ci < nk; ++ci) end += lens[ci];
    int d0 = -1;
    for (int ci = 0, q = c0; ci < nk; q += lens[ci++]) {
        std::vector<int> mt(lens[ci]);
        for (int j = 0; j < lens[ci]; ++j) mt[j] = q + 1 + j < P ? prompt[q + 1 + j] : next;
        const bool last = ci == nk - 1 && end == P;
        const int dd = nk > 1 ? mtp_draft_chunk(mt.data(), lens[ci], q, ci, last) : mtp_draft(mt.data(), lens[ci], q, last);
        if (last) d0 = dd;
    }
    return d0;
}

// HYPER4_TRACE: GPU 0's copy of the last forward's last-row residual after every layer, <dir>/hyper_l_last-<il>.bin
void Engine4::dump_trace(const std::string & dir) {
    Device & d = dev(0);
    if (!d.trace) throw std::runtime_error("dump_trace: run with HYPER4_TRACE");
    const size_t hcn = cfg_.hc_dim();
    std::vector<float> h((size_t) cfg_.n_layer * hcn);
    CUDA_CHECK(cudaSetDevice(d.id));
    CUDA_CHECK(cudaMemcpy(h.data(), d.trace, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int il = 0; il < cfg_.n_layer; ++il) {
        FILE * f = fopen((dir + "/hyper_l_last-" + std::to_string(il) + ".bin").c_str(), "wb");
        if (!f) throw std::runtime_error("dump_trace: cannot write to " + dir);
        fwrite(h.data() + il * hcn, sizeof(float), hcn, f);
        fclose(f);
    }
    // stages of layers 0-1, named like llama.cpp's tensors (.1: the second tensor of that name in a layer)
    const size_t pe = (size_t) cfg_.ple_n_heads() * cfg_.ple_dim, tw = std::max(hcn, pe), n = cfg_.n_embd;
    std::vector<float> t2((size_t) TRACE2_SLOTS * tw);
    CUDA_CHECK(cudaMemcpy(t2.data(), d.trace2, t2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto put = [&](int slot, const std::string & name, size_t w) {
        FILE * f = fopen((dir + "/hyper_" + name + ".bin").c_str(), "wb");
        if (f) { fwrite(t2.data() + slot * tw, sizeof(float), w, f); fclose(f); }
    };
    if (pe) put(0, "ple_embd", pe);
    put(1, "hc_init", hcn);
    for (int il = 0; il < 2; ++il) {
        const std::string l = "-" + std::to_string(il);
        put(2 + 4 * il, "ple_out" + l, hcn);
        put(3 + 4 * il, "hc_mixed" + l, n);
        put(4 + 4 * il, "hc_combine" + l, hcn);
        put(5 + 4 * il, "hc_mixed" + l + ".1", n);
    }
}

} // namespace hyper
