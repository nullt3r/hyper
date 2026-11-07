#include "engine5.h"
#include "dense.h"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <pthread.h>

#include <algorithm>
#include <array>
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
#include <unordered_map>

namespace hyper {


namespace {

constexpr int R5 = MOE_BULK_ROWS;   // activation rows (prefill chunk)
constexpr int GSCORE_ROWS = 64;     // indexer: tokens scored at a time
constexpr int GLIST = 2052;         // max attended cells per token: 512 pools * 4 + 3 tail cells (+1)

} // namespace

struct Engine5::DevLayer {
    bool mla = false, moe = false;
    // mHC (replicated): fn [24][4n], scale [3], base [24]; block norms
    DW hca_fn, hcf_fn;
    uint8_t * hca_raw = nullptr, * hcf_raw = nullptr;   // the same rows as raw GGUF Q8_0 (fused mHC mix; nullptr: not Q8_0)
    float * hcf32 = nullptr;   // HYPER5_PRED=2 without hcf_raw: hc_ffn_fn in fp32 for the prediction's mix (side stream)
    int hcf_rows = 0;
    float * hca_scale = nullptr, * hca_base = nullptr, * hcf_scale = nullptr, * hcf_base = nullptr;
    float * attn_norm = nullptr, * ffn_norm = nullptr;
    int nh = 0, h0 = 0;          // local heads (KDA or MLA)
    DW wo;                        // output projection, local columns
    // KDA: kin rows [q | k | v | f_a | g_a | beta] on x, kgate = block-diagonal [f_b ; g_b] on [f_a | g_a]
    DW kin, kgate;
    float * conv_w = nullptr, * dt_bias = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    float * conv_state = nullptr, * state = nullptr;
    float * conv_snap = nullptr, * state_snap = nullptr;   // speculative verification: state after each of the first nt-1 rows
    // MLA: min rows [q_a | kv_a | idx_k | idx_gate] on x (replicated), mq rows [q_b local | idx_q_b] on the normed q_a
    DW min, mq;
    half * wkb = nullptr, * wvb = nullptr;   // fp16 [nh][512][256], [nh][256][512] (sources not Q8_0, or HYPER5_HEAD_F16)
    int8_t * wkb_q = nullptr, * wvb_q = nullptr;   // Q8_0 sources as they are: int8 values + fp16 scales per 32
    half * wkb_s = nullptr, * wvb_s = nullptr;
    float * q_a_norm = nullptr, * kv_a_norm = nullptr, * idx_proj = nullptr, * idx_lnw = nullptr, * idx_lnb = nullptr, * idx_ape = nullptr;
    half * lat = nullptr, * pooled = nullptr, * ring = nullptr;
    // FFN: dense layer or the shared expert (local hidden slice)
    DW gu, down;
    int ff_l = 0;
    float * router = nullptr, * exp_bias = nullptr;
    uint16_t * router16 = nullptr;   // the router as bf16, when every value is one exactly (GLM-5.3-Flash: all; half the bytes)
};

struct Engine5::Device : TpDevice {
    std::vector<DevLayer> layers;   // (+ the MTP block at index n_layer)
    float * out_norm = nullptr;
    // MTP block input: eh_proj on [enorm(token embedding) | hnorm(hidden)]; its head norm; mprev: the main model's hidden row
    // before the next MTP row; mchain: the MTP block's last output after its head norm (the chained hidden)
    DW m_eh;
    float * m_enorm = nullptr, * m_hnorm = nullptr, * m_shnorm = nullptr, * mprev = nullptr, * mchain = nullptr;
    // activations [R5] rows
    float * mix = nullptr, * mixpart = nullptr, * hcw = nullptr;
    int * pos_last = nullptr;        // MTP update: the last row's position
    float * qabs = nullptr, * olat = nullptr;
    // HYPER5_PRED=2 (GPU 0): a MoE layer's routing predicted from its input before the token mixer, on a side stream, for the
    // CPU's prefetch (CpuMoe prefetch mode 2)
    cudaEvent_t ev_pf0 = nullptr, ev_pf1 = nullptr;
    float * pmixpart = nullptr, * phcw = nullptr, * pxn = nullptr;
    float * hf32 = nullptr;          // prefill: one layer's Q8_0 per-head matrix as exact fp32
    size_t hf32_n = 0;

    ~Device() {
        cudaSetDevice(id);
        for (auto ev : {ev_pf0, ev_pf1}) if (ev) cudaEventDestroy(ev);
    }
};

// KDA row layout (per token): q, k, v [nh*128 each] | f_a [128] | g_a [128] | beta [nh] (padded to 32) | f_b [nh*128] | g_b [nh*128]
static int kda_fa(int nh) { return 3 * nh * 128; }
static int kda_beta(int nh) { return 3 * nh * 128 + 256; }
static int kda_fb(int nh) { return 3 * nh * 128 + 288; }
static int kda_stride(int nh) { return kda_fb(nh) + 2 * nh * 128; }
// MLA row layout: q_a raw [1536] | kv raw [512] | idx k raw [128] | idx gate raw [128] | idx weights [32] | q_a normed [1536] |
// q [nh*256] | idx q [32*128]
static int mla_w(const Glm5Config & c) { return c.q_lora + c.kv_lora + 2 * GIDX_DIM; }
static int mla_qr(const Glm5Config & c) { return mla_w(c) + GIDX_HEADS; }
static int mla_q(const Glm5Config & c) { return mla_qr(c) + c.q_lora; }
static int mla_stride(const Glm5Config & c, int nh) { return mla_q(c) + nh * c.qk_dim + GIDX_HEADS * GIDX_DIM; }

Engine5::Engine5(const std::string & model_path, const Engine5Options & opt) : TpEngine(opt, TpModel{}) {
    gguf_ = std::make_unique<GGUF>(model_path);
    cfg_ = Glm5Config::from_gguf(*gguf_);
    fprintf(stderr, "hyper5: %s\n", cfg_.describe().c_str());
    if (!opt_.mtp_path.empty()) {
        mtp_g_ = std::make_unique<GGUF>(opt_.mtp_path);
        if (!mtp_g_->tensor("blk." + std::to_string(cfg_.n_layer) + ".nextn.eh_proj.weight"))
            throw std::runtime_error("hyper5: " + opt_.mtp_path + " has no NextN block blk." + std::to_string(cfg_.n_layer));
        fprintf(stderr, "hyper5: MTP block from %s\n", opt_.mtp_path.c_str());
    }
    if (getenv("HYPER5_ZC")) fprintf(stderr, "hyper5: HYPER5_ZC (zero-copy expert slices) is gone with the compact host copies: ignored\n");
    m_.tag = "hyper5"; m_.env = "HYPER5";
    m_.n_layer = cfg_.n_layer; m_.n_dense = cfg_.n_dense;
    m_.n_embd = cfg_.n_embd; m_.n_vocab = cfg_.n_vocab;
    m_.n_expert = cfg_.n_expert; m_.n_expert_used = cfg_.n_expert_used; m_.n_ff_exp = cfg_.n_ff_exp;
    m_.clamp_exp = cfg_.clamp_exp;
    m_.max_pos_align = 4;
    m_.stream_min = 280; m_.up_parts = 1; m_.halves = 2; m_.stage_factor = 1.05; m_.fill_unit = 64;   // (GLM experts ~4x larger than Flash-Next's)
    m_.stream_cpu = 120.0; m_.stream_cpu_max = 0.5;
    // 0.85 measured best: at 16k +16 % (vs +10 % one fixed draft, +0.5 % three), Codex replay +10 % greedy / +11 % at temperature 1;
    // the CPU-load gate: a verification row reads its own CPU experts from RAM, worth it when the GPUs pace a step
    m_.mtp_pmin = 0.85; m_.mtp_cpu_max = 1.5;
    m_.lookup_spec = true;
    // HYPER5_PSTREAM_PRIO: the prediction side stream at the highest priority (with the graphs' per-node priorities), so its
    // few small kernels are not queued behind the token mixer's: the CPU's prefetch gets the prediction sooner
    m_.pstream_prio = getenv("HYPER5_PSTREAM_PRIO") && atoi(getenv("HYPER5_PSTREAM_PRIO")) != 0;
    init();
}

std::unique_ptr<TpDevice> Engine5::make_device() { return std::make_unique<Device>(); }
Engine5::Device & Engine5::dev(int g) { return static_cast<Device &>(*devs_[g]); }
const Engine5::Device & Engine5::dev(int g) const { return static_cast<const Device &>(*devs_[g]); }

void Engine5::load_layer(Device & dev, DevLayer & L, int il) {
    const Glm5Config & c = cfg_;
    const int nd = opt_.n_devices, g = dev.g, n = c.n_embd;
    const GGUF & G = src(il);
    const bool mtp = il == c.n_layer;   // the NextN block: an MLA + MoE layer without mHC
    auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
    auto T = [&](const std::string & name) { return &G.need(name); };
    auto f32 = [&](const std::string & name) {
        const std::vector<float> v = to_f32(G.need(name));
        return dev.upload(v.data(), v.size());
    };
    auto full = [&](const std::string & name) { const GTensor * t = T(name); return upload_dense(A, dev.id, {{t, 0, t->rows()}}); };
    auto fp16_rows = [&](const std::string & name, int64_t r0, int64_t r1) {
        const std::vector<float> v = to_f32(G.need(name), r0, r1);
        std::vector<half> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
        return dev.upload(h.data(), h.size());
    };
    const std::string p = "blk." + std::to_string(il) + ".";
    L.mla = mtp || c.is_mla[il];
    L.moe = is_moe(il);
    if (!mtp) {
        L.hca_fn = full(p + "hc_attn_fn.weight");
        L.hcf_fn = full(p + "hc_ffn_fn.weight");
        for (auto [nm, dst] : {std::pair<const char *, uint8_t **>{"hc_attn_fn.weight", &L.hca_raw}, {"hc_ffn_fn.weight", &L.hcf_raw}}) {
            const GTensor & t = G.need(p + nm);
            if (t.type == GType::Q8_0) *dst = dev.upload(t.data, t.nbytes);   // (other types, e.g. Maya's F32: the unfused mix)
        }
        if (pred_mode_ == 2 && !L.hcf_raw && L.moe && g == 0) {
            const GTensor & t = G.need(p + "hc_ffn_fn.weight");
            const std::vector<float> v = to_f32(t);
            L.hcf32 = dev.upload(v.data(), v.size());
            L.hcf_rows = (int) t.rows();
        }
        L.hca_scale = f32(p + "hc_attn_scale.weight");
        L.hca_base = f32(p + "hc_attn_base.weight");
        L.hcf_scale = f32(p + "hc_ffn_scale.weight");
        L.hcf_base = f32(p + "hc_ffn_base.weight");
    }
    L.attn_norm = f32(p + "attn_norm.weight");
    L.ffn_norm = f32(p + "ffn_norm.weight");
    auto [h0, h1] = split(c.n_head, nd, g);
    L.h0 = (int) h0;
    L.nh = (int) (h1 - h0);
    const int nh = L.nh;
    if (!L.mla) {
        const int64_t a0 = h0 * 128, a1 = h1 * 128;
        L.kin = upload_dense(A, dev.id, {{T(p + "attn_q.weight"), a0, a1}, {T(p + "attn_k.weight"), a0, a1}, {T(p + "attn_v.weight"), a0, a1},
                                         {T(p + "ssm_f_a.weight"), 0, 128}, {T(p + "ssm_g_a.weight"), 0, 128},
                                         {T(p + "ssm_beta.weight"), h0, h1}});
        {   // block-diagonal [f_b ; g_b] over [f_a | g_a] (k = 256): zero blocks are exact in Q8_0
            const GTensor * fb = T(p + "ssm_f_b.weight"), * gb = T(p + "ssm_g_b.weight");
            if (fb->type != GType::Q8_0 || gb->type != GType::Q8_0 || fb->ne[0] != 128) throw std::runtime_error("KDA gates: expected Q8_0, k 128");
            const int rows = (int) (2 * (a1 - a0)), k = 256, kb = 8;
            std::vector<int8_t> qs((size_t) rows * k, 0);
            std::vector<half> d((size_t) rows * kb, __float2half(0.0f));
            for (int r = 0; r < rows; ++r) {
                const bool second = r >= (int) (a1 - a0);
                const GTensor * t = second ? gb : fb;
                const int64_t sr = a0 + (second ? r - (a1 - a0) : r);
                const uint8_t * src = t->data + (size_t) sr * t->row_bytes();
                for (int j = 0; j < 4; ++j) {
                    const int oj = (second ? 4 : 0) + j;
                    memcpy(&d[(size_t) r * kb + oj], src + j * 34, 2);
                    memcpy(&qs[(size_t) r * k + oj * 32], src + j * 34 + 2, 32);
                }
            }
            L.kgate.q8 = to_device_q8(A, dev.id, qs.data(), d.data(), rows, k);
        }
        L.wo = upload_dense(A, dev.id, {{T(p + "attn_output.weight"), 0, n}}, {{a0 / 32, a1 / 32}});
        {   // conv taps of the local q, k, v channels
            std::vector<float> buf;
            for (const char * nm : {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"}) {
                const std::vector<float> cw = to_f32(G.need(p + nm));
                buf.insert(buf.end(), cw.begin() + a0 * c.conv, cw.begin() + a1 * c.conv);
            }
            L.conv_w = dev.upload(buf.data(), buf.size());
        }
        {
            const std::vector<float> dt = to_f32(G.need(p + "ssm_dt.bias")), sa = to_f32(G.need(p + "ssm_a"));
            L.dt_bias = dev.upload(dt.data() + a0, (size_t) (a1 - a0));
            L.ssm_a = dev.upload(sa.data() + h0, (size_t) nh);
        }
        L.ssm_norm = f32(p + "ssm_norm.weight");
        L.conv_state = dev.alloc<float>((size_t) (c.conv - 1) * 3 * nh * 128);
        L.state = dev.alloc<float>((size_t) nh * 128 * 128);
        if (opt_.n_draft > 0) {
            L.conv_snap = dev.alloc<float>((size_t) (c.conv - 1) * 3 * nh * 128 * (MAX_NT - 1));
            L.state_snap = dev.alloc<float>((size_t) nh * 128 * 128 * (MAX_NT - 1));
        }
    } else {
        L.min = upload_dense(A, dev.id, {{T(p + "attn_q_a.weight"), 0, c.q_lora}, {T(p + "attn_kv_a_mqa.weight"), 0, c.kv_lora},
                                         {T(p + "indexer.attn_k.weight"), 0, GIDX_DIM}, {T(p + "indexer_compressor_gate.weight"), 0, GIDX_DIM}});
        L.mq = upload_dense(A, dev.id, {{T(p + "attn_q_b.weight"), h0 * c.qk_dim, h1 * c.qk_dim},
                                        {T(p + "indexer.attn_q_b.weight"), 0, GIDX_HEADS * GIDX_DIM}});
        static const bool head_f16 = getenv("HYPER5_HEAD_F16") != nullptr;
        auto q8_rows = [&](const std::string & name, int64_t r0, int64_t r1, int8_t ** q, half ** sc) {
            const GTensor & t = G.need(name);
            const int64_t C = t.ne[0], kb = C / 32, nr = r1 - r0;
            std::vector<int8_t> qs((size_t) nr * C);
            std::vector<half> d((size_t) nr * kb);
            for (int64_t r = 0; r < nr; ++r)
                for (int64_t b = 0; b < kb; ++b) {
                    const uint8_t * blk = t.data + (size_t) (r0 + r) * t.row_bytes() + (size_t) b * 34;
                    memcpy(&d[(size_t) r * kb + b], blk, 2);
                    memcpy(&qs[(size_t) r * C + b * 32], blk + 2, 32);
                }
            *q = dev.upload(qs.data(), qs.size());
            *sc = dev.upload(d.data(), d.size());
        };
        const bool hq = !head_f16 && G.need(p + "attn_k_b.weight").type == GType::Q8_0 && G.need(p + "attn_v_b.weight").type == GType::Q8_0;
        if (hq) {
            q8_rows(p + "attn_k_b.weight", h0 * c.kv_lora, h1 * c.kv_lora, &L.wkb_q, &L.wkb_s);
            q8_rows(p + "attn_v_b.weight", h0 * c.v_dim, h1 * c.v_dim, &L.wvb_q, &L.wvb_s);
        } else {
            L.wkb = fp16_rows(p + "attn_k_b.weight", h0 * c.kv_lora, h1 * c.kv_lora);
            L.wvb = fp16_rows(p + "attn_v_b.weight", h0 * c.v_dim, h1 * c.v_dim);
        }
        L.wo = upload_dense(A, dev.id, {{T(p + "attn_output.weight"), 0, n}}, {{h0 * c.v_dim / 32, h1 * c.v_dim / 32}});
        L.q_a_norm = f32(p + "attn_q_a_norm.weight");
        L.kv_a_norm = f32(p + "attn_kv_a_norm.weight");
        {
            std::vector<float> pw = to_f32(G.need(p + "indexer.proj.weight"));
            const float sc = 1.0f / sqrtf((float) (GIDX_DIM * GIDX_HEADS));
            for (auto & v : pw) v *= sc;
            L.idx_proj = dev.upload(pw.data(), pw.size());
        }
        L.idx_lnw = f32(p + "indexer.k_norm.weight");
        L.idx_lnb = f32(p + "indexer.k_norm.bias");
        L.idx_ape = f32(p + "indexer_compressor_ape.weight");
        L.lat = kv_alloc<half>(dev, 1, (size_t) opt_.max_pos * MLA_LAT, MLA_LAT, 1);
        L.pooled = kv_alloc<half>(dev, 1, (size_t) (opt_.max_pos / 4 + 1) * GIDX_DIM, GIDX_DIM, 4);
        L.ring = dev.alloc<half>(8 * 2 * GIDX_DIM);
    }
    // FFN: hidden slice of the dense FFN or the shared expert
    {
        const int ff = L.moe ? c.n_ff_exp : c.n_ff;
        const std::string gn = L.moe ? "ffn_gate_shexp.weight" : "ffn_gate.weight", un = L.moe ? "ffn_up_shexp.weight" : "ffn_up.weight",
                          dn = L.moe ? "ffn_down_shexp.weight" : "ffn_down.weight";
        auto [f0, f1] = split(ff, nd, g, 32);
        L.ff_l = (int) (f1 - f0);
        L.gu = upload_dense(A, dev.id, {{T(p + gn), f0, f1}, {T(p + un), f0, f1}});
        L.down = upload_dense(A, dev.id, {{T(p + dn), 0, n}}, {{f0 / 32, f1 / 32}});
    }
    if (L.moe) {
        const std::vector<float> rw = to_f32(G.need(p + "ffn_gate_inp.weight"));
        static const bool r32 = getenv("HYPER5_ROUTER32") != nullptr;   // (A/B: the fp32 router)
        std::vector<uint16_t> b16(rw.size());
        bool exact = !r32;
        for (size_t i = 0; i < rw.size() && exact; ++i) {
            uint32_t u;
            memcpy(&u, &rw[i], 4);
            exact = (u & 0xffffu) == 0;
            b16[i] = (uint16_t) (u >> 16);
        }
        if (exact) L.router16 = dev.upload(b16.data(), b16.size());
        else {
            L.router = dev.upload(rw.data(), rw.size());
            if (!r32 && dev.g == 0) fprintf(stderr, "hyper5: layer %d router stays fp32 (not exactly bf16)\n", il);
        }
        L.exp_bias = f32(p + "exp_probs_b.bias");
    }
}

// the experts of layer il: owners by weighted round robin over the devices' quotas (hottest first); the rest is the
// shared placement's (device arrays, the compact host copy, the CPU side, stream lists)
// a device's layers (the MTP block when loaded), its output norm, MTP input projection and vocab slice, the activations
void Engine5::load_device(TpDevice & td) {
    Device & dev = static_cast<Device &>(td);
    const Glm5Config & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd, hcn = MHC * n, g = dev.g;
    CUDA_CHECK(cudaSetDevice(dev.id));
    mla_init();
    CUDA_CHECK(cudaEventCreateWithFlags(&dev.ev_pf0, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&dev.ev_pf1, cudaEventDisableTiming));
    auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
    dev.layers.resize(nl_tot());
    for (int il = 0; il < nl_tot(); ++il) load_layer(dev, dev.layers[il], il);
    { const std::vector<float> v = to_f32(gguf_->need("output_norm.weight")); dev.out_norm = dev.upload(v.data(), v.size()); }
    if (mtp_g_) {   // (replicated: eh_proj is one GEMV per MTP row)
        const std::string p = "blk." + std::to_string(c.n_layer) + ".nextn.";
        auto up32 = [&](const std::string & name) { const std::vector<float> v = to_f32(mtp_g_->need(name)); return dev.upload(v.data(), v.size()); };
        { const GTensor * t = &mtp_g_->need(p + "eh_proj.weight"); dev.m_eh = upload_dense(A, dev.id, {{t, 0, t->rows()}}); }
        dev.m_enorm = up32(p + "enorm.weight");
        dev.m_hnorm = up32(p + "hnorm.weight");
        dev.m_shnorm = up32(p + "shared_head_norm.weight");
        dev.mprev = dev.alloc<float>(n);
        dev.mchain = dev.alloc<float>(n);
    }
    auto [o0, o1] = split(c.n_vocab, nd, g);
    dev.vocab_off = (int) o0;
    {   // LM head: Q4_K / Q6_K natively (exact); HYPER5_OUT_Q8: re-quantized to Q8_0, HYPER5_OUT_F16: fp16
        const GTensor & ot = gguf_->need("output.weight");
        const bool out_q8 = getenv("HYPER5_OUT_Q8") != nullptr, out_f16 = getenv("HYPER5_OUT_F16") != nullptr;
        if ((ot.type == GType::Q6_K || ot.type == GType::Q4_K) && !out_q8 && !out_f16) {
            std::vector<int> kbl(ot.ne[0] / 32);
            for (size_t b = 0; b < kbl.size(); ++b) kbl[b] = (int) b;
            KQHost h;
            const KQ kt = ot.type == GType::Q4_K ? KQ::Q4K : KQ::Q6K;
            const int dg = repack_kq(kt, ot.data + (size_t) o0 * ot.row_bytes(), ot.row_bytes(), (int) (o1 - o0), kbl, h);
            CUDA_CHECK(cudaSetDevice(dev.id));
            auto up = [&](const auto & v) -> void * {
                if (v.empty()) return nullptr;
                void * pd = A(v.size() * sizeof(v[0]));
                CUDA_CHECK(cudaMemcpy(pd, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice));
                return pd;
            };
            DW & w = dev.output;
            w.kq = true;
            w.kw.type = kt; w.kw.n = (int) (o1 - o0); w.kw.k = (int) ot.ne[0]; w.kw.dg = dg;
            w.kw.lo = (const uint2 *) up(h.lo); w.kw.hi = (const unsigned *) up(h.hi);
            w.kw.scm = (const uint16_t *) up(h.scm); w.kw.sc6 = (const int8_t *) up(h.sc6); w.kw.d = (const half2 *) up(h.d);
        } else dev.output = upload_dense(A, dev.id, {{&ot, o0, o1}}, {}, !out_f16);
    }
    // activations: the engine's common buffers, then the model's own
    int nh_max = 0;
    for (auto & L : dev.layers) nh_max = std::max(nh_max, L.nh);
    const int R = R5;
    int ffl = 0;   // largest local FFN hidden slice
    for (auto & L : dev.layers) ffl = std::max(ffl, L.ff_l);
    CommonSizes sz;
    sz.res_w = hcn; sz.xn_w = n;
    sz.big_stride = std::max(kda_stride(nh_max), mla_stride(c, nh_max));
    sz.o_w = std::max(nh_max * 128, nh_max * c.v_dim);
    sz.attn_part = mla_part_floats(nh_max, MAX_NT);
    sz.rlog_w = c.n_expert;
    sz.pred_all = true; sz.pred_k_w = MOE_MAX_USED; sz.pred_psg_w = 16;
    sz.ff_sh = ffl;
    sz.hexp_rows = MAX_NT; sz.logit_rows = MAX_NT;
    sz.xh_w = std::max({hcn, ffl, nh_max * MLA_LAT});
    sz.conv_w = 3 * nh_max * 128;
    sz.iscore_rows = GSCORE_ROWS; sz.ilist_w = GLIST; sz.ihist = true;
    alloc_common(dev, sz);
    dev.pos_last = dev.alloc<int>(1);
    dev.mix = dev.alloc<float>((size_t) R * 32);
    dev.mixpart = dev.alloc<float>((size_t) R * (hcn / 256) * 25);
    dev.hcw = dev.alloc<float>((size_t) R * MHC_W);
    dev.qabs = dev.alloc<float>((size_t) R * nh_max * MLA_LAT);
    dev.olat = dev.alloc<float>((size_t) R * nh_max * MLA_LAT);
    if (pred_mode_ == 2) {
        dev.pmixpart = dev.alloc<float>((size_t) MAX_NT * (hcn / 256) * 25);
        dev.phcw = dev.alloc<float>((size_t) MAX_NT * MHC_W);
        dev.pxn = dev.alloc<float>((size_t) MAX_NT * n);
    }
    dev.hf32_n = (size_t) nh_max * MLA_LAT * std::max(c.qk_dim, c.v_dim);   // (one layer's wkb or wvb as fp32, prefill)
    dev.hf32 = dev.alloc<float>(dev.hf32_n);
}

void Engine5::record_main(int gi, int nt, int kind) {
    const Glm5Config & c = cfg_;
    Device & d = dev(gi);
    cudaStream_t s = d.stream;
    const int n = c.n_embd, hcn = MHC * n, K = c.n_expert_used, bs = d.big_stride;
    const int nd = opt_.n_devices;
    const float eps = c.rms_eps;
    const bool bulk = nt > MAX_NT, mtp = kind != 0;
    const DenseRun dr{s, d.blas, d.xh, d.ggs, d.ggs_elems};
    auto mm = [&](const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni = NormIn{}) {
        dense_mm(dr, W, x, xs, y, ys, rows, ni);
    };
    auto f32mm = [&](const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nr) {
        if (nr <= MAX_NT) { gemv_f32(W, rows, k, x, xs, y, ys, nr, s); return; }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, nr, k, &one, W, k, x, xs, &zero, y, ys) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemm failed");
    };
    // router logits: fp32, or bf16 (decode: the same products and sums; prefill: widened to fp32 for the same sgemm)
    auto router_mm = [&](const DevLayer & RL, const float * x, float * y, int nr) {
        const int E = cfg_.n_expert, k = cfg_.n_embd;
        if (!RL.router16) f32mm(RL.router, E, k, x, k, y, E, nr);
        else if (nr <= MAX_NT) gemv_bf16w(RL.router16, E, k, x, k, y, E, nr, s);
        else {
            bf16_to_f32(RL.router16, (size_t) E * k, d.router32, s);
            f32mm(d.router32, E, k, x, k, y, E, nr);
        }
    };
    // per-head projections (fp16 W [H][R][C]): warp GEMV, or a strided batched tensor-core GEMM for a prefill chunk
    auto hmm = [&](const half * W, int H, int R, int C, const float * x, int xs, float * y, int ys, int nr) {
        if (nr <= MAX_NT) { head_gemv(W, H, R, C, x, xs, y, ys, nr, s); return; }
        to_half(x, xs, nullptr, H * C, 0.0f, d.xh, nr, s);
        const float one = 1.0f, zero = 0.0f;
        if (cublasGemmStridedBatchedEx(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, R, nr, C, &one, W, CUDA_R_16F, C, (long long) R * C, d.xh, CUDA_R_16F,
                                       H * C, C, &zero, y, CUDA_R_32F, ys, R, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cublasGemmStridedBatchedEx failed (device: ") + cudaGetErrorString(cudaDeviceSynchronize()) + ")");
    };
    // Q8_0 per-head matrices: GEMV on int8 + scales; prefill: exact fp32 copy + fp32 batched GEMM (no rounding to fp16)
    auto hmmq = [&](const int8_t * Q, const half * S, int H, int Rr, int C, const float * x, int xs, float * y, int ys, int nr) {
        if (nr <= MAX_NT) { head_gemv_q8(Q, S, H, Rr, C, x, xs, y, ys, nr, s); return; }
        const size_t ne = (size_t) H * Rr * C;
        if (ne > d.hf32_n) throw std::runtime_error("hmmq: scratch too small");
        q8_rows_f32(Q, S, ne, d.hf32, s);
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemmStridedBatched(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, Rr, nr, C, &one, d.hf32, C, (long long) Rr * C, x, xs, C, &zero,
                                      y, ys, Rr, H) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemmStridedBatched failed");
    };
    // chunks of this pass (a multi-chunk prefill runs each layer for all of them before the next layer): residual rows,
    // positions and embeddings per chunk; everything else is per-layer scratch
    struct Ck { int nt; float * R; int * P; int pos; };
    const int nck = bulk ? mc_n_ : 1;
    std::vector<Ck> ck(nck);
    for (int ci = 0; ci < nck; ++ci) {
        ck[ci].nt = ci == 0 ? nt : mc_nt_[ci];
        ck[ci].R = ci == 0 ? d.res : d.res_x[ci - 1];
        ck[ci].P = ci == 0 ? d.pos : d.pos_x + ci;
        ck[ci].pos = ci == 0 ? h_pos_[0] : mc_pos_[ci];
    }
    float * R = d.res;
    int * P = d.pos;
    int pos_host = h_pos_[0];
    for (int ci = 0; ci < nck; ++ci) {
        const int m = ck[ci].nt;
        CUDA_CHECK(cudaMemcpyAsync(ck[ci].P, h_pos_ + ci, sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d.x, h_embd_ + (size_t) ci * R5 * n, (size_t) m * n * sizeof(float), cudaMemcpyHostToDevice, s));
        if (!mtp) { mhc_init(ck[ci].R, d.x, n, m, s); continue; }
        // MTP block input: eh_proj([enorm(emb) | hnorm(hidden)]); its residual stream (rows of n) replaces the chunk's
        // main residual rows once their hidden rows (the main LM head's input) are taken
        float * ecat = d.big0;   // [m][2n]
        rmsnorm(d.x, n, d.m_enorm, ecat, 2 * n, n, m, eps, s);
        if (kind == 2) rmsnorm(d.mchain, n, d.m_hnorm, ecat + n, 2 * n, n, 1, eps, s);
        else {
            mhc_head(ck[ci].R, d.out_norm, eps, n, d.xn, m, s);
            if (mtp_shift_) {   // row r: the hidden of row r - 1 (row 0: the previous chunk's last)
                rmsnorm(d.mprev, n, d.m_hnorm, ecat + n, 2 * n, n, 1, eps, s);
                if (m > 1) rmsnorm(d.xn, n, d.m_hnorm, ecat + 3 * n, 2 * n, n, m - 1, eps, s);
            } else rmsnorm(d.xn, n, d.m_hnorm, ecat + n, 2 * n, n, m, eps, s);
            CUDA_CHECK(cudaMemcpyAsync(d.mprev, d.xn + (size_t) (m - 1) * n, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
        }
        mm(d.m_eh, ecat, 2 * n, ck[ci].R, n, m);
    }
    // MTP update over several rows: all of them enter the block's cache, only the last one goes on (attention, FFN, head)
    const bool last_only = kind == 1 && !mtp_shift_ && !bulk && nt > 1 && !mtp_full_;
    if (last_only) CUDA_CHECK(cudaMemcpyAsync(d.pos_last, h_pos_ + 8, sizeof(int), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    const int il0 = mtp ? c.n_layer : 0, il1 = mtp ? c.n_layer + 1 : c.n_layer;   // (this pass's layers)
    // prompt rows of the MTP block: only its cache (latents, indexer keys) is used later, and that depends on the block's
    // input alone -- no attention output, FFN or head for them
    const bool cache_only = mtp && mtp_shift_ && !mtp_full_;
    const bool streaming = bulk && opt_.stream_experts && nt >= stream_min_ && !cache_only;
    if (streaming) stream_begin(d, il0, il1);
    int call = 0, dcall = 0;
    auto dbg = [&](const char * what, int il, const float * buf, size_t cnt) { TpEngine::dbg(d, what, il, buf, cnt, 4); };
    auto allreduce = [&](float * x = nullptr) { TpEngine::allreduce(d, nt, call, dcall, bulk, x); };   // x += sum over the GPUs of d.part (default: d.bo = the sum)
    // block output: allreduce of d.part over the GPUs, scattered into the residual streams (decode: in the allreduce epilogue)
    static const bool old_post = getenv("HYPER5_OLDPOST") != nullptr;
    auto block_out = [&] {
        if (!bulk && !old_post) { allreduce_mhc_ll16(R, d.hcw, n, d.part, ar_ll_, d.g, nd, nt * n, d.counter, call++, s); return; }
        allreduce();
        mhc_post(R, d.bo, d.hcw, n, nt, s);
    };
    // mHC pre-mix: res -> d.xn (block input after its RMS norm) and the post / comb weights
    static const bool old_mhc = getenv("HYPER5_OLDMHC") != nullptr;
    auto hc_pre = [&](const DW & fn, const uint8_t * raw, const float * scale, const float * base, const float * norm_w) {
        if (raw && !old_mhc && !bulk) { mhc_pre_fused(R, raw, d.mixpart, scale, base, norm_w, eps, c.hc_eps, c.sinkhorn, n, d.hcw, d.xn, nt, s); return; }
        mm(fn, R, hcn, d.mix, 32, nt);
        mhc_pre(R, d.mix, 32, scale, base, norm_w, eps, c.hc_eps, c.sinkhorn, n, d.hcw, d.xn, nt, s);
    };
    dbg("embed", -1, R, (size_t) nt * hcn);
    for (int il = il0; il < il1; ++il)
    for (int ci = 0; ci < nck; ++ci) {
        nt = ck[ci].nt; R = ck[ci].R; P = ck[ci].P; pos_host = ck[ci].pos;
        const bool last_ck = ci == nck - 1;   // (the streamed experts' staging is released after the last chunk)
        DevLayer & L = d.layers[il];
        const int nh = L.nh;
        // HYPER5_PRED=2: this MoE layer's routing predicted from its input (the token mixer's contribution left out) on the side
        // stream while the token mixer runs; it reads R, which the token mixer's block output rewrites (that waits for ev_pf1)
        if (d.pe_pending) { CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_pe1, 0)); d.pe_pending = false; }
        const bool pred2 = pred_mode_ == 2 && !bulk && !mtp && d.g == 0 && L.moe && nt <= MAX_NT && (L.hcf_raw || L.hcf32);
        if (pred2) {
            cudaStream_t ps = d.pstream;
            CUDA_CHECK(cudaEventRecord(d.ev_pf0, s));
            CUDA_CHECK(cudaStreamWaitEvent(ps, d.ev_pf0, 0));
            if (L.hcf_raw)
                mhc_pre_fused(R, L.hcf_raw, d.pmixpart, L.hcf_scale, L.hcf_base, L.ffn_norm, eps, c.hc_eps, c.sinkhorn, n, d.phcw, d.pxn, nt, ps);
            else {
                gemv_f32(L.hcf32, L.hcf_rows, hcn, R, hcn, d.pmixpart, 32, nt, ps);
                mhc_pre(R, d.pmixpart, 32, L.hcf_scale, L.hcf_base, L.ffn_norm, eps, c.hc_eps, c.sinkhorn, n, d.phcw, d.pxn, nt, ps);
            }
            const int E = cfg_.n_expert;
            if (L.router16) gemv_bf16w(L.router16, E, n, d.pxn, n, d.plog, E, nt, ps);
            else gemv_f32(L.router, E, n, d.pxn, n, d.plog, E, nt, ps);
            moe_route_sig(d.plog, E, L.exp_bias, E, K, c.w_scale, d.pids, d.pwts, d.psg, nt, ps);
            moe_publish(&cpu_pred_[il].seq, &cpu_pred_[il].nt, &cpu_pred_[il].ids[0][0], &cpu_pred_[il].wts[0][0], &cpu_pred_[il].x[0][0],
                        d.pxn, n, 0, d.pids, d.pwts, K, nt, d.counter, (unsigned) il, ps);
            CUDA_CHECK(cudaEventRecord(d.ev_pf1, ps));
        }
        // ---- token mixer ----
        if (mtp) rmsnorm(R, n, L.attn_norm, d.xn, n, n, nt, eps, s);   // (the MTP block: plain pre-norm residual layer)
        else hc_pre(L.hca_fn, L.hca_raw, L.hca_scale, L.hca_base, L.attn_norm);
        dbg("attn_in", il, d.xn, (size_t) nt * n);
        if (!L.mla) {
            const int fa = kda_fa(nh), fb = kda_fb(nh);
            mm(L.kin, d.xn, n, d.big0, bs, nt);
            mm(L.kgate, d.big0 + fa, bs, d.big0 + fb, bs, nt);
            const bool snap = !bulk && nt > 1 && L.conv_snap;   // verification rows: keep the state after each row
            gdn_conv(d.big0, bs, L.conv_state, snap ? L.conv_snap : nullptr, L.conv_w, 3 * nh * 128, c.conv, nt, s, bulk ? d.conv_raw : nullptr);
            const int ostride = nh * 128;
            kda_step(d.big0, bs, 0, nh * 128, 2 * nh * 128, fb, kda_beta(nh), L.state, snap ? L.state_snap : nullptr, d.o, ostride, L.dt_bias,
                     L.ssm_a, c.gate_lb, nh,
                     1e-6f, nt, s);
            gated_norm_sigmoid(d.o, ostride, d.big0 + fb + nh * 128, bs, L.ssm_norm, nh, 128, eps, nt, s);
            mm(L.wo, d.o, ostride, d.part, n, nt);
        } else {
            const int wq = mla_w(c), qr = mla_qr(c), qo = mla_q(c);
            mm(L.min, d.xn, n, d.big0, bs, nt);
            if (cache_only || last_only) {
                mla_kv(d.big0 + c.q_lora, bs, L.kv_a_norm, eps, L.lat, P, nt, s);
                gidx_pool(d.big0 + c.q_lora + c.kv_lora, d.big0 + c.q_lora + c.kv_lora + GIDX_DIM, bs, L.idx_lnw, L.idx_lnb, c.ln_eps, L.idx_ape,
                          L.ring, L.pooled, P, nt, s);
                if (cache_only) continue;
                // the last row moves to row 0 (its block input, normed input and projections), then a 1-row layer
                const size_t lr = (size_t) (nt - 1);
                CUDA_CHECK(cudaMemcpyAsync(R, R + lr * n, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(d.xn, d.xn + lr * n, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(d.big0, d.big0 + lr * bs, (size_t) mla_w(c) * sizeof(float), cudaMemcpyDeviceToDevice, s));
                nt = 1;
                P = d.pos_last;
            }
            f32mm(L.idx_proj, GIDX_HEADS, n, d.xn, n, d.big0 + wq, bs, nt);
            rmsnorm(d.big0, bs, L.q_a_norm, d.big0 + qr, bs, c.q_lora, nt, eps, s);
            mm(L.mq, d.big0 + qr, bs, d.big0 + qo, bs, nt);
            mla_kv(d.big0 + c.q_lora, bs, L.kv_a_norm, eps, L.lat, P, nt, s);
            gidx_pool(d.big0 + c.q_lora + c.kv_lora, d.big0 + c.q_lora + c.kv_lora + GIDX_DIM, bs, L.idx_lnw, L.idx_lnb, c.ln_eps, L.idx_ape,
                      L.ring, L.pooled, P, nt, s);
            const int top = c.idx_top_k / c.kpool;
            const bool dense_chunk = (pos_host + nt) / 4 <= top;   // every token attends to all of its cells
            if (!dense_chunk || !bulk)
                gidx_select(d.big0 + qo + nh * c.qk_dim, bs, d.big0 + wq, bs, L.pooled, P, nt, top, d.iscores, opt_.max_pos / 4 + 4,
                            GSCORE_ROWS, d.ilist, GLIST, d.ilist_n, s, getenv("HYPER5_OLDSEL") ? nullptr : d.ihist);
            if (L.wkb_q) hmmq(L.wkb_q, L.wkb_s, nh, MLA_LAT, c.qk_dim, d.big0 + qo, bs, d.qabs, nh * MLA_LAT, nt);
            else hmm(L.wkb, nh, MLA_LAT, c.qk_dim, d.big0 + qo, bs, d.qabs, nh * MLA_LAT, nt);
            const bool use_list = !(bulk && dense_chunk);
            mla_attn(d.qabs, nh * MLA_LAT, L.lat, P, nh, 1.0f / sqrtf((float) c.qk_dim), nt, use_list ? d.ilist : nullptr, GLIST,
                     use_list ? d.ilist_n : nullptr, d.attn_part, d.olat, nh * MLA_LAT, s);
            if (L.wvb_q) hmmq(L.wvb_q, L.wvb_s, nh, c.v_dim, MLA_LAT, d.olat, nh * MLA_LAT, d.o, nh * c.v_dim, nt);
            else hmm(L.wvb, nh, c.v_dim, MLA_LAT, d.olat, nh * MLA_LAT, d.o, nh * c.v_dim, nt);
            mm(L.wo, d.o, nh * c.v_dim, d.part, n, nt);
        }
        dbg(L.mla ? "mla_part" : "kda_part", il, d.part, (size_t) nt * n);
        if (pred2) CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_pf1, 0));
        if (mtp) allreduce(R); else block_out();
        dbg("attn_res", il, R, (size_t) nt * (mtp ? n : hcn));
        // ---- FFN ----
        if (mtp) rmsnorm(R, n, L.ffn_norm, d.xn, n, n, nt, eps, s);
        else hc_pre(L.hcf_fn, L.hcf_raw, L.hcf_scale, L.hcf_base, L.ffn_norm);
        float * ffo = L.moe ? d.shpart : d.part;
        auto shared_ffn = [&] {   // (MoE layers: after the routing is published, so the CPU experts start without waiting for it)
            mm(L.gu, d.xn, n, d.shgu, 2 * L.ff_l, nt);
            if (nt <= MAX_NT) {
                NormIn glu; glu.act = 3; glu.glu_off = L.ff_l; glu.act_scale = L.moe ? c.clamp_sh : c.clamp_sh;
                mm(L.down, d.shgu, 2 * L.ff_l, ffo, n, nt, glu);
            } else {
                swiglu_clamp(d.shgu, 2 * L.ff_l, L.ff_l, d.shh, L.ff_l, L.ff_l, c.clamp_sh, nt, s);
                mm(L.down, d.shh, L.ff_l, ffo, n, nt);
            }
        };
        static const bool shared_first = getenv("HYPER5_SHARED_FIRST") != nullptr;
        if (!L.moe || shared_first) shared_ffn();
        if (L.moe) {
            router_mm(L, d.xn, d.rlog, nt);
            static const bool sep_pub = getenv("HYPER5_SEP_PUBLISH") != nullptr;
            const bool fused_pub = !bulk && d.g == 0 && !sep_pub;   // (decode never streams)
            if (fused_pub)
                moe_route_sig_publish(d.rlog, c.n_expert, L.exp_bias, c.n_expert, K, c.w_scale, d.ids, d.wts, d.sg, nt, &cpu_rec_[il].seq,
                                      &cpu_rec_[il].nt, &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0], d.xn, n, n,
                                      d.counter, (unsigned) il, d.ticket, s);
            else moe_route_sig(d.rlog, c.n_expert, L.exp_bias, c.n_expert, K, c.w_scale, d.ids, d.wts, d.sg, nt, s);
            dbg("route_w", il, d.wts, (size_t) nt * K);
            if (pred_k_ && !bulk && d.g == 0 && il + 1 < c.n_layer && is_moe(il + 1)) {   // (HYPER5_PREDSTAT)
                const DevLayer & N = d.layers[il + 1];
                router_mm(N, d.xn, d.plog, 1);
                moe_route_sig(d.plog, c.n_expert, N.exp_bias, c.n_expert, pred_k_, c.w_scale, d.pids, d.pwts, d.psg, 1, s);
                CUDA_CHECK(cudaMemcpyAsync(h_pred_ + (size_t) (il + 1) * MOE_MAX_USED, d.pids, pred_k_ * sizeof(int), cudaMemcpyDeviceToHost, s));
            }
            if (pred_early_ && pred_mode_ == 2 && !bulk && !mtp && d.g == 0 && nt <= MAX_NT && il + 1 < c.n_layer && is_moe(il + 1)) {
                // early prediction of the next layer's routing for the CPU's helpers (they read during this layer's CPU job)
                const DevLayer & N = d.layers[il + 1];
                cudaStream_t ps = d.pstream;
                const int E = cfg_.n_expert;
                CUDA_CHECK(cudaEventRecord(d.ev_pe0, s));
                CUDA_CHECK(cudaStreamWaitEvent(ps, d.ev_pe0, 0));
                if (N.router16) gemv_bf16w(N.router16, E, n, d.xn, n, d.plog, E, nt, ps);
                else gemv_f32(N.router, E, n, d.xn, n, d.plog, E, nt, ps);
                moe_route_sig(d.plog, E, N.exp_bias, E, K, c.w_scale, d.pids, d.pwts, d.psg, nt, ps);
                moe_publish(&cpu_pred1_[il + 1].seq, &cpu_pred1_[il + 1].nt, &cpu_pred1_[il + 1].ids[0][0], &cpu_pred1_[il + 1].wts[0][0],
                            &cpu_pred1_[il + 1].x[0][0], d.xn, n, 0, d.pids, d.pwts, K, nt, d.counter, (unsigned) (il + 1), ps);
                CUDA_CHECK(cudaEventRecord(d.ev_pe1, ps));
                d.pe_pending = true;
            }
            if (bulk && d.g == 0 && place_->adapt())   // prompt routing for the adaptive placement
                CUDA_CHECK(cudaMemcpyAsync(h_ids_ + ((size_t) il * mc_max_ + ci) * R5 * K, d.ids, (size_t) nt * K * sizeof(int), cudaMemcpyDeviceToHost, s));
            MoeLayerDev & X = place_->dev(d.g, il);   // the layer's experts on this device
            const bool stream = streaming && X.owner_bulk;
            const bool cpu_part = !stream || (X.cpu_keep && mc_n_ == 1);   // (a streamed chunk with CPU-kept experts publishes too)
            if (d.g == 0 && cpu_part) {
                if (bulk) moe_publish(&cpu_bulk_->seq, &cpu_bulk_->nt, &cpu_bulk_->ids[0][0], &cpu_bulk_->wts[0][0], &cpu_bulk_->x[0][0],
                                      d.xn, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
                else if (!fused_pub)
                    moe_publish(&cpu_rec_[il].seq, &cpu_rec_[il].nt, &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0],
                                d.xn, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
            }
            if (!shared_first) shared_ffn();
            if (bulk) {
                moe_order(X.moe, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
                to_half(d.xn, n, nullptr, n, 0.0f, d.mix16, nt, s);
                moe_gemm_gate_up(X.moe, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
                moe_gemm_down(X.moe, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
                if (stream)   // this GPU's share of the CPU experts, in two halves through the two staging buffers
                    for (int sb = 0; sb < 2; ++sb) {
                        MoeDev ms = X.moe;
                        const int cnt = (int) X.st_list[sb].size();
                        ms.gate = d.stage[sb];
                        ms.up = d.stage[sb] + (size_t) cnt * X.moe.gate_bytes;
                        ms.down = d.stage[sb] + (size_t) 2 * cnt * X.moe.gate_bytes;
                        ms.slot = X.st_slot[sb];
                        while (!d.up_pend.empty() && (d.up_pend.front()[0] < il || (d.up_pend.front()[0] == il && d.up_pend.front()[1] <= sb))) {
                            const auto q = d.up_pend.front();   // (parts no allreduce took yet)
                            d.up_pend.pop_front();
                            place_->upload_stage(d.g, q[0], q[1], q[2], up_parts_);
                        }
                        CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_up[sb], 0));
                        moe_order(ms, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
                        moe_gemm_gate_up(ms, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
                        moe_gemm_down(ms, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
                        if (last_ck) stream_next(d, il, sb, il1);
                    }
            } else {
                moe_gate_up(X.moe, d.xn, n, d.ids, K, d.hexp, nt, s);
                moe_down(X.moe, d.hexp, d.ids, d.wts, K, d.yexp, nt, s);
            }
            const volatile unsigned * cflag = d.g == 0 && !nocpu_ && cpu_part ? (bulk ? &cpu_bulk_out_->seq : &cpu_out_[il].seq) : nullptr;
            moe_reduce(d.shpart, d.sg, d.yexp, K, d.part, n, nt, d.ids, stream ? X.owner_bulk : X.owner, d.g, CPU_OWNER, cflag,
                       bulk ? &cpu_bulk_out_->y[0][0] : &cpu_out_[il].y[0][0], d.counter, (unsigned) il, s);
        }
        dbg("ffn_part", il, d.part, (size_t) nt * n);
        if (mtp) allreduce(R); else block_out();
        dbg("l_out", il, R, (size_t) nt * (mtp ? n : hcn));
    }
    if (mtp) {   // draft: argmax after the last row (the head norm's output is the next chained hidden)
        if (cache_only) return;
        rmsnorm(R + (size_t) (nt - 1) * n, n, d.m_shnorm, d.mchain, n, n, 1, eps, s);
        mm(d.output, d.mchain, n, d.logits, d.output.n(), 1);
        argmax_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.wts, 1, s);
        max_sumexp(d.logits, d.output.n(), d.wts + 2, s);   // (the best token's probability: the drafts' gate)
        topk_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.topk, TOPK, 1, s);   // (sampled drafts' candidates)
        CUDA_CHECK(cudaMemcpyAsync(h_mtopk_ + (size_t) gi * TOPK * 2, d.topk, (size_t) TOPK * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(h_mres_ + (size_t) gi * 4, d.wts, 4 * sizeof(float), cudaMemcpyDeviceToHost, s));
        return;
    }
    const int hr = bulk ? 1 : nt;
    mhc_head(R + (size_t) (nt - hr) * hcn, d.out_norm, eps, n, d.xn, hr, s);
    mm(d.output, d.xn, n, d.logits, d.output.n(), hr);
    argmax_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.wts, hr, s);
    CUDA_CHECK(cudaMemcpyAsync(h_res_ + (size_t) gi * MAX_NT * 2, d.wts, (size_t) hr * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
    topk_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.topk, TOPK, hr, s);
    CUDA_CHECK(cudaMemcpyAsync(h_topk_ + (size_t) gi * MAX_NT * TOPK * 2, d.topk, (size_t) hr * TOPK * 2 * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
}

// part p of np of half sb (split by expert index); the first part waits for the staging buffer, the last one signals ev_up
// roll the KDA state back to the snapshot after row keep-1 of the last verification (the MLA side needs nothing: rejected rows'
// latents / pools are rewritten when those positions are processed again)
// part p of np of half sb (split by expert index); the first part waits for the staging buffer, the last one signals ev_up
// roll the KDA state back to the snapshot after row keep-1 of the last verification (the MLA side needs nothing: rejected rows'
// latents / pools are rewritten when those positions are processed again)
void Engine5::record_restore(int gi, int keep) {
    const Glm5Config & c = cfg_;
    Device & d = dev(gi);
    for (auto & L : d.layers) {
        if (L.mla || !L.conv_snap) continue;
        const size_t cs = (size_t) (c.conv - 1) * 3 * L.nh * 128, ss = (size_t) L.nh * 128 * 128;
        CUDA_CHECK(cudaMemcpyAsync(L.conv_state, L.conv_snap + (keep - 1) * cs, cs * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
        CUDA_CHECK(cudaMemcpyAsync(L.state, L.state_snap + (keep - 1) * ss, ss * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
    }
}

void Engine5::embed(const int * tokens, int nt, int, int chunk) {
    const GTensor & te = gguf_->need("token_embd.weight");
    const auto * te_tr = ggml_get_type_traits((ggml_type) te.type);
    for (int t = 0; t < nt; ++t) {
        const uint8_t * row = te.data + (size_t) tokens[t] * te.row_bytes();
        float * out = h_embd_ + ((size_t) chunk * R5 + t) * cfg_.n_embd;
        if (te.type == GType::F32) memcpy(out, row, cfg_.n_embd * sizeof(float));
        else te_tr->to_float(row, out, cfg_.n_embd);
    }
}

// ---------------- prompt cache: KDA conv + state, the open indexer pool's cells ----------------
size_t Engine5::snap_floats(int gi) const {
    const Glm5Config & c = cfg_;
    size_t n = 0;
    for (auto & L : dev(gi).layers)
        n += L.mla ? 8 * GIDX_DIM : (size_t) (c.conv - 1) * 3 * L.nh * 128 + (size_t) L.nh * 128 * 128;
    if (dev(gi).mprev) n += c.n_embd;
    return n;
}

void Engine5::snap_copy(std::vector<float *> & hv, bool to_host) {
    const Glm5Config & c = cfg_;
    for (size_t gi = 0; gi < devs_.size(); ++gi) {
        Device & d = dev(gi);
        CUDA_CHECK(cudaSetDevice(d.id));
        float * hp = hv[gi];
        auto cp = [&](void * dev, size_t n) {
            if (to_host) CUDA_CHECK(cudaMemcpyAsync(hp, dev, n * sizeof(float), cudaMemcpyDeviceToHost, d.stream));
            else CUDA_CHECK(cudaMemcpyAsync(dev, hp, n * sizeof(float), cudaMemcpyHostToDevice, d.stream));
            hp += n;
        };
        for (auto & L : d.layers) {
            if (L.mla) cp(L.ring, 8 * GIDX_DIM);   // 8 cells * 256 halves
            else {
                cp(L.conv_state, (size_t) (c.conv - 1) * 3 * L.nh * 128);
                cp(L.state, (size_t) L.nh * 128 * 128);
            }
        }
        if (d.mprev) cp(d.mprev, c.n_embd);
    }
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->stream)); }
}

// per-position GPU buffers of a conversation (parked to RAM): every MLA layer's latent rows and pooled indexer keys
// per-position GPU buffers of a conversation (parked to RAM): every MLA layer's latent rows and pooled indexer keys
std::vector<ConvPark<TpEngine::TpSnap>::Dev> Engine5::park_devs() const {
    std::vector<ConvPark<TpSnap>::Dev> v;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const Device & dp = dev((int) g);
        ConvPark<TpSnap>::Dev pd;
        pd.id = dp.id;
        pd.stream = dp.stream;
        for (auto & L : dp.layers)
            if (L.mla) {
                pd.spans.push_back({L.lat, MLA_LAT * sizeof(half), 0, 1, 1});
                pd.spans.push_back({L.pooled, GIDX_DIM * sizeof(half), 0, 1, 4});
            }
        v.push_back(std::move(pd));
    }
    return v;
}

void Engine5::reset() {
    const Glm5Config & c = cfg_;
    for (size_t g = 0; g < devs_.size(); ++g) {
        Device & dp = dev((int) g);
        CUDA_CHECK(cudaSetDevice(dp.id));
        for (auto & L : dp.layers)
            if (!L.mla) {
                CUDA_CHECK(cudaMemset(L.conv_state, 0, (size_t) (c.conv - 1) * 3 * L.nh * 128 * sizeof(float)));
                CUDA_CHECK(cudaMemset(L.state, 0, (size_t) L.nh * 128 * 128 * sizeof(float)));
            }
        if (dp.mprev) CUDA_CHECK(cudaMemset(dp.mprev, 0, c.n_embd * sizeof(float)));   // (MTP row 0: no hidden before it)
    }
}

// tokens[r] at pos + r with the main hidden row r of the last verification (which started at pos - 1): the kept rows
int Engine5::mtp_update(const int * tokens, int nt, int pos, bool) {
    if (nt < 1 || nt > MAX_NT || pos + nt > opt_.max_pos) throw std::runtime_error("mtp_update: bad rows");
    embed(tokens, nt, pos);
    h_pos_[0] = pos;
    h_pos_[8] = pos + nt - 1;
    draft_at(pos + nt);   // (the last row drafts the token after it)
    run(1, nt);
    return mtp_result();
}

// token at pos with the MTP block's own last output (from_prev: with the main model's hidden of pos - 1)
// token at pos with the MTP block's own last output (from_prev: with the main model's hidden of pos - 1)
int Engine5::mtp_chain(int token, int pos, bool from_prev) {   // (from_prev: the main model's hidden of pos - 1)
    if (pos >= opt_.max_pos) throw std::runtime_error("mtp_chain: position exceeds max_pos");
    if (from_prev)
        for (size_t g = 0; g < devs_.size(); ++g) {
            Device & dp = dev((int) g);
            CUDA_CHECK(cudaSetDevice(dp.id));
            CUDA_CHECK(cudaMemcpyAsync(dp.mchain, dp.mprev, cfg_.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, dp.stream));
        }
    embed(&token, 1, pos);
    h_pos_[0] = pos;
    draft_at(pos + 1);
    run(2, 1);
    return mtp_result();
}

// the MTP block over the chunks of the last prompt forward (its embeddings and positions are still in place; row r takes
// the main hidden of row r - 1, row 0 the previous chunk's last): the block's cache for the prompt, mprev = the last hidden
// the MTP block over the chunks of the last prompt forward (its embeddings and positions are still in place; row r takes
// the main hidden of row r - 1, row 0 the previous chunk's last): the block's cache for the prompt, mprev = the last hidden
void Engine5::mtp_prefill(int nck) {
    const int nt0 = nck > 1 ? mc_nt_[0] : last_nt_;
    if (h_pos_[0] == 0) memset(h_embd_, 0, cfg_.n_embd * sizeof(float));   // (position 0: no token embedding either, as ik)
    if (mtp_full_) place_->rebuild_dirty_streams();   // (the test path streams the block's experts)
    mtp_shift_ = true;
    mc_n_ = nck;
    try { run(1, nt0); } catch (...) { mtp_shift_ = false; mc_n_ = 1; throw; }
    mtp_shift_ = false;
    mc_n_ = 1;
}

// MTP after a prefill forward: the block's cache rows for the chunks (shifted: row r takes the main hidden of row r - 1);
// the first draft is chained from the main hidden afterwards (-1 here)
int Engine5::mtp_prompt(const std::vector<int> &, int, const int *, int nk, int) {
    mtp_prefill(nk);
    return -1;
}

} // namespace hyper
