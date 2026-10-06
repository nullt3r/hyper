#include "engine4.h"

#include <cuda_runtime.h>

#include <algorithm>
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

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

constexpr int CPU_OWNER = 3;

struct Engine4::DevLayer {
    bool full = false, ple = false;
    // hyper-connection mixers (replicated): norm [hc*n], down [lr x hc*n], up [hc*n x lr], inject [hc][hc*n]
    float * hca_norm = nullptr, * hcf_norm = nullptr, * hca_inj = nullptr, * hcf_inj = nullptr;
    Q8W hca_down, hca_up, hcf_down, hcf_up;
    // gated attention: local q heads [head_off, +n_head_l), local kv heads [kv_off, +n_kv_l)
    Q8W wqkv, wo;
    float * q_norm = nullptr, * k_norm = nullptr;
    half * kcache = nullptr, * vcache = nullptr;
    int n_head_l = 0, head_off = 0, n_kv_l = 0, kv_off = 0;
    // gated delta net (head-aligned partition)
    Q8W win, wout;
    float * ab_w = nullptr;   // fp32 rows [alpha local | beta local]
    float * conv_w = nullptr, * dt_bias = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    float * conv_state = nullptr, * state = nullptr;
    int n_v_l = 0, n_k_l = 0, conv_ch = 0;
    // MoE
    float * router = nullptr;  // fp32 [n_expert + 1][n]: experts, then the shared-expert gate
    Q8W sh_gu, sh_down;
    int n_sh_l = 0;
    MoeDev moe;
    int * owner = nullptr;     // [n_expert]: device or CPU_OWNER
    // PLE
    Q8W ple_key, ple_value;
    float * ple_wk = nullptr, * ple_wq = nullptr, * ple_wc = nullptr, * ple_conv = nullptr, * ple_state = nullptr;
};

struct Engine4::Device {
    int id = 0, g = 0;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t g_main[MAX_NT + 1] = {};
    std::vector<DevLayer> layers;
    float * head_norm = nullptr;
    Q8W head_down, head_up, output;
    int vocab_off = 0;
    // activations [MAX_NT] rows
    float * x = nullptr, * res = nullptr, * xn = nullptr, * gate = nullptr, * lo = nullptr, * inj = nullptr, * mixed = nullptr;
    float * bo = nullptr, * part = nullptr, * big0 = nullptr, * o = nullptr, * attn_part = nullptr;
    float * rlog = nullptr, * wts = nullptr, * sg = nullptr, * shgu = nullptr, * shh = nullptr, * shpart = nullptr;
    float * hexp = nullptr, * yexp = nullptr, * ple_emb = nullptr, * ple_key = nullptr, * ple_val = nullptr, * ple_sc = nullptr;
    float * logits = nullptr, * res2 = nullptr;
    int * ids = nullptr, * pos = nullptr, * counter = nullptr;
    int big_stride = 0;
    size_t used = 0;
    std::vector<void *> allocs;

    template <typename T> T * alloc(size_t n) {
        void * p = nullptr;
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
        CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
        allocs.push_back(p);
        used += n * sizeof(T);
        return (T *) p;
    }
    template <typename T> T * upload(const T * src, size_t n) {
        T * p = alloc<T>(n);
        CUDA_CHECK(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    ~Device() {
        cudaSetDevice(id);
        for (auto & gr : g_main) if (gr) cudaGraphExecDestroy(gr);
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

namespace {

struct RowRange { const GTensor * t; int64_t r0, r1; };
using ColRanges = std::vector<std::pair<int64_t, int64_t>>;

Q8W to_device_q8(const std::function<void *(size_t)> & alloc, int dev, const int8_t * qs, const half * d, int n, int k) {
    const size_t ntile = (n + 15) / 16, kb = k / 32;
    std::vector<uint8_t> fq(ntile * kb * 512);
    std::vector<half> fs(ntile * kb * 16);
    repack_q8_frag(qs, d, n, k, fq.data(), fs.data());
    CUDA_CHECK(cudaSetDevice(dev));
    Q8W w; w.n = n; w.k = k;
    void * pq = alloc(fq.size());
    void * ps = alloc(fs.size() * sizeof(half));
    CUDA_CHECK(cudaMemcpy(pq, fq.data(), fq.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ps, fs.data(), fs.size() * sizeof(half), cudaMemcpyHostToDevice));
    w.q = (const uint4 *) pq; w.s = (const half *) ps;
    return w;
}

// Q8_0 rows from several tensors (same k), restricted to column blocks
Q8W upload_q8(const std::function<void *(size_t)> & alloc, int dev, const std::vector<RowRange> & parts, ColRanges cols = {}) {
    const int64_t k_full = parts[0].t->ne[0];
    if (cols.empty()) cols.push_back({0, k_full / 32});
    std::vector<int64_t> blocks;
    for (auto & [b0, b1] : cols) for (int64_t b = b0; b < b1; ++b) blocks.push_back(b);
    const int64_t kb = (int64_t) blocks.size(), k = kb * 32;
    int64_t n = 0;
    for (auto & p : parts) {
        if (p.t->type != GType::Q8_0 || p.t->ne[0] != k_full) throw std::runtime_error("upload_q8: bad tensor " + p.t->name);
        n += p.r1 - p.r0;
    }
    std::vector<int8_t> qs((size_t) n * k);
    std::vector<half> d((size_t) n * kb);
    int64_t row0 = 0;
    for (auto & p : parts) {
        const size_t rb = p.t->row_bytes();
#pragma omp parallel for schedule(static)
        for (int64_t r = p.r0; r < p.r1; ++r) {
            const int64_t orow = row0 + (r - p.r0);
            const uint8_t * src = p.t->data + (size_t) r * rb;
            for (int64_t j = 0; j < kb; ++j) {
                const uint8_t * blk = src + (size_t) blocks[j] * 34;
                memcpy(&d[(size_t) orow * kb + j], blk, 2);
                memcpy(&qs[(size_t) orow * k + (size_t) j * 32], blk + 2, 32);
            }
        }
        row0 += p.r1 - p.r0;
    }
    return to_device_q8(alloc, dev, qs.data(), d.data(), (int) n, (int) k);
}

std::pair<int64_t, int64_t> split(int64_t n, int ndev, int g, int64_t align = 1) {
    const int64_t units = n / align;
    return {units * g / ndev * align, units * (g + 1) / ndev * align};
}

const float kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

float f16(const uint8_t * p) { half h; memcpy(&h, p, 2); return __half2float(h); }

void mmq(const Q8W & W, const float * x, int xs, float * y, int ys, int nt, cudaStream_t s) {
    gemv_q8(W, x, xs, y, ys, nullptr, nt, s, NormIn{});
}

} // namespace

Engine4::Engine4(const std::string & model_path, const Engine4Options & opt) : opt_(opt) {
    debug_ = getenv("HYPER4_DEBUG") != nullptr;
    gguf_ = std::make_unique<GGUF>(model_path);
    cfg_ = Q4Config::from_gguf(*gguf_);
    fprintf(stderr, "hyper4: %s\n", cfg_.describe().c_str());
    if (cfg_.n_embd > 4096 || cfg_.n_expert_used > MOE_MAX_USED) throw std::runtime_error("hyper4: model dimensions unsupported");
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    opt_.n_devices = std::min(opt_.n_devices, ndev);
    for (int g = 0; g < opt_.n_devices; ++g) {
        auto dev = std::make_unique<Device>();
        dev->id = g; dev->g = g;
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->stream, cudaStreamNonBlocking));
        devs_.push_back(std::move(dev));
    }
    const int nd = opt_.n_devices, n = cfg_.n_embd;
    CUDA_CHECK(cudaHostAlloc(&h_embd_, (size_t) MAX_NT * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_ple_, (size_t) MAX_NT * std::max(1, cfg_.ple_n_heads() * cfg_.ple_dim) * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_pos_, 4 * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_res_, (size_t) nd * MAX_NT * 2 * sizeof(float), cudaHostAllocPortable));
    const size_t ll = (size_t) 2 * nd * MAX_NT * n / 2;
    CUDA_CHECK(cudaHostAlloc(&ar_ll_, ll * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_ll_, 0xff, ll * sizeof(uint2));
    CUDA_CHECK(cudaHostAlloc(&cpu_rec_, (size_t) cfg_.n_layer * sizeof(CpuMoeRec), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_out_, (size_t) cfg_.n_layer * sizeof(CpuMoeOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_rec_, 0, (size_t) cfg_.n_layer * sizeof(CpuMoeRec));
    memset((void *) cpu_out_, 0, (size_t) cfg_.n_layer * sizeof(CpuMoeOut));
    cpu_ = std::make_unique<CpuMoe>(opt_.cpu_threads, n, cfg_.n_ff_exp, cfg_.n_expert_used, cpu_rec_, cpu_out_, cfg_.n_layer);
    load_weights();
}

Engine4::~Engine4() {
    cpu_.reset();
    devs_.clear();
    for (void * p : {(void *) h_embd_, (void *) h_ple_, (void *) h_pos_, (void *) h_res_, (void *) ar_ll_, (void *) cpu_rec_, (void *) cpu_out_})
        if (p) cudaFreeHost(p);
}

void Engine4::load_layer(Device & dev, DevLayer & L, int il) {
    const Q4Config & c = cfg_;
    const int nd = opt_.n_devices, g = dev.g, n = c.n_embd;
    const int dk = c.ssm_d_state, dv = c.head_v_dim(), nk = c.ssm_n_group, nv = c.ssm_dt_rank;
    auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
    auto T = [&](const std::string & name) { return &gguf_->need(name); };
    auto f32 = [&](const std::string & name) {
        const GTensor & t = gguf_->need(name);
        if (t.type != GType::F32) throw std::runtime_error("expected F32: " + name);
        return dev.upload((const float *) t.data, (size_t) t.nelements());
    };
    auto q8full = [&](const std::string & name) { const GTensor * t = T(name); return upload_q8(A, dev.id, {{t, 0, t->rows()}}); };
    const std::string p = "blk." + std::to_string(il) + ".";
    L.full = c.is_full_attn(il);
    // hyper-connection mixers
    L.hca_norm = f32(p + "hc_attn_norm.weight");
    L.hcf_norm = f32(p + "hc_ffn_norm.weight");
    L.hca_inj = f32(p + "hc_attn_inject.weight");
    L.hcf_inj = f32(p + "hc_ffn_inject.weight");
    L.hca_down = q8full(p + "hc_attn_down.weight");
    L.hca_up = q8full(p + "hc_attn_up.weight");
    L.hcf_down = q8full(p + "hc_ffn_down.weight");
    L.hcf_up = q8full(p + "hc_ffn_up.weight");
    if (L.full) {
        const int hd = c.head_dim, group = c.n_head / c.n_head_kv;
        L.n_head_l = c.n_head / nd;
        L.head_off = g * L.n_head_l;
        L.kv_off = L.head_off / group;
        L.n_kv_l = (L.head_off + L.n_head_l - 1) / group - L.kv_off + 1;
        L.wqkv = upload_q8(A, dev.id, {{T(p + "attn_q.weight"), (int64_t) L.head_off * 2 * hd, (int64_t) (L.head_off + L.n_head_l) * 2 * hd},
                                       {T(p + "attn_k.weight"), (int64_t) L.kv_off * hd, (int64_t) (L.kv_off + L.n_kv_l) * hd},
                                       {T(p + "attn_v.weight"), (int64_t) L.kv_off * hd, (int64_t) (L.kv_off + L.n_kv_l) * hd}});
        L.wo = upload_q8(A, dev.id, {{T(p + "attn_output.weight"), 0, n}},
                         {{(int64_t) L.head_off * hd / 32, (int64_t) (L.head_off + L.n_head_l) * hd / 32}});
        L.q_norm = f32(p + "attn_q_norm.weight");
        L.k_norm = f32(p + "attn_k_norm.weight");
        const size_t kv = (size_t) L.n_kv_l * opt_.max_pos * hd;
        L.kcache = dev.alloc<half>(kv);
        L.vcache = dev.alloc<half>(kv);
    } else {
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
        L.win = upload_q8(A, dev.id, rows);
        L.wout = upload_q8(A, dev.id, {{T(p + "ssm_out.weight"), 0, n}}, out_cols);
        {   // alpha / beta rows (fp32)
            const GTensor & ta = gguf_->need(p + "ssm_alpha.weight"), & tb = gguf_->need(p + "ssm_beta.weight");
            if (ta.type != GType::F32 || tb.type != GType::F32) throw std::runtime_error("ssm_alpha/beta: expected F32");
            std::vector<float> buf;
            for (int h : vh) buf.insert(buf.end(), (const float *) ta.data + (size_t) h * n, (const float *) ta.data + (size_t) (h + 1) * n);
            for (int h : vh) buf.insert(buf.end(), (const float *) tb.data + (size_t) h * n, (const float *) tb.data + (size_t) (h + 1) * n);
            L.ab_w = dev.upload(buf.data(), buf.size());
        }
        {
            const GTensor & cw = gguf_->need(p + "ssm_conv1d.weight");
            const int K = c.ssm_conv;
            std::vector<float> buf((size_t) L.conv_ch * K);
            for (size_t j = 0; j < chans.size(); ++j)
                memcpy(buf.data() + j * K, (const float *) cw.data + chans[j] * K, K * sizeof(float));
            L.conv_w = dev.upload(buf.data(), buf.size());
        }
        auto gather = [&](const std::string & name) {
            const GTensor & t = gguf_->need(name);
            std::vector<float> buf;
            for (int h : vh) buf.push_back(((const float *) t.data)[h]);
            return dev.upload(buf.data(), buf.size());
        };
        L.dt_bias = gather(p + "ssm_dt.bias");
        L.ssm_a = gather(p + "ssm_a");
        L.ssm_norm = f32(p + "ssm_norm.weight");
        L.conv_state = dev.alloc<float>((size_t) (c.ssm_conv - 1) * L.conv_ch);
        L.state = dev.alloc<float>((size_t) L.n_v_l * dk * dv);
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
    }
    // router + shared-expert gate (replicated, fp32)
    {
        const GTensor & r = gguf_->need(p + "ffn_gate_inp.weight"), & sgt = gguf_->need(p + "ffn_gate_inp_shexp.weight");
        if (r.type != GType::F32 || sgt.type != GType::F32) throw std::runtime_error("router: expected F32");
        std::vector<float> buf((const float *) r.data, (const float *) r.data + r.nelements());
        buf.insert(buf.end(), (const float *) sgt.data, (const float *) sgt.data + sgt.nelements());
        L.router = dev.upload(buf.data(), buf.size());
    }
    // shared expert: hidden split
    {
        auto [f0, f1] = split(c.n_ff_shexp, nd, g, 32);
        L.n_sh_l = (int) (f1 - f0);
        L.sh_gu = upload_q8(A, dev.id, {{T(p + "ffn_gate_shexp.weight"), f0, f1}, {T(p + "ffn_up_shexp.weight"), f0, f1}});
        L.sh_down = upload_q8(A, dev.id, {{T(p + "ffn_down_shexp.weight"), 0, n}}, {{f0 / 32, f1 / 32}});
    }
    // experts: per GPU a contiguous index range, the rest on the CPU
    {
        const GTensor & tg = gguf_->need(p + "ffn_gate_exps.weight"), & tu = gguf_->need(p + "ffn_up_exps.weight"),
                      & tdn = gguf_->need(p + "ffn_down_exps.weight");
        const int E = c.n_expert, ff = c.n_ff_exp;
        const int per = (int) (E * opt_.gpu_expert_frac / nd);
        std::vector<int> owner(E, CPU_OWNER), slot(E, -1);
        for (int gg = 0; gg < nd; ++gg) for (int e = gg * per; e < (gg + 1) * per; ++e) owner[e] = gg;
        const size_t gb = tg.nbytes / E, db = tdn.nbytes / E;
        int nl = 0;
        for (int e = 0; e < E; ++e) if (owner[e] == g) slot[e] = nl++;
        uint8_t * dg = dev.alloc<uint8_t>(gb * std::max(nl, 1)), * du = dev.alloc<uint8_t>(gb * std::max(nl, 1));
        uint8_t * dd = dev.alloc<uint8_t>(db * std::max(nl, 1));
        for (int e = 0; e < E; ++e) {
            if (slot[e] < 0) continue;
            CUDA_CHECK(cudaMemcpy(dg + slot[e] * gb, tg.data + (size_t) e * gb, gb, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(du + slot[e] * gb, tu.data + (size_t) e * gb, gb, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dd + slot[e] * db, tdn.data + (size_t) e * db, db, cudaMemcpyHostToDevice));
        }
        L.moe.gate = dg; L.moe.up = du; L.moe.down = dd;
        L.moe.tg = tg.type; L.moe.td = tdn.type;
        if (tu.type != tg.type) throw std::runtime_error("expert gate/up types differ");
        L.moe.gate_bytes = gb; L.moe.down_bytes = db;
        L.moe.ff = ff; L.moe.n = n;
        L.moe.slot = dev.upload(slot.data(), slot.size());
        L.owner = dev.upload(owner.data(), owner.size());
        if (g == 0) {
            CpuExpertLayer cl;
            cl.tg = tg.type; cl.td = tdn.type;
            cl.gate = tg.data; cl.up = tu.data; cl.down = tdn.data;
            cl.gate_bytes = gb; cl.down_bytes = db;
            cl.owned.resize(E);
            for (int e = 0; e < E; ++e) cl.owned[e] = owner[e] == CPU_OWNER;
            cpu_->set_layer(il, cl);
        }
    }
}

void Engine4::load_weights() {
    auto t0 = std::chrono::steady_clock::now();
    const Q4Config & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd, hcn = c.hc_dim();
    if (c.n_head % nd) throw std::runtime_error("attention head count must divide the device count");
    for (auto & dp : devs_) {
        Device & dev = *dp;
        const int g = dev.g;
        auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
        dev.layers.resize(c.n_layer);
        for (int il = 0; il < c.n_layer; ++il) load_layer(dev, dev.layers[il], il);
        dev.head_norm = dev.upload((const float *) gguf_->need("output_hc_norm.weight").data, (size_t) hcn);
        { const GTensor * t = &gguf_->need("output_hc_down.weight"); dev.head_down = upload_q8(A, dev.id, {{t, 0, t->rows()}}); }
        { const GTensor * t = &gguf_->need("output_hc_up.weight"); dev.head_up = upload_q8(A, dev.id, {{t, 0, t->rows()}}); }
        auto [o0, o1] = split(c.n_vocab, nd, g);
        dev.vocab_off = (int) o0;
        dev.output = upload_q8(A, dev.id, {{&gguf_->need("output.weight"), o0, o1}});
        // activations
        const int conv_dim = c.conv_dim();
        dev.big_stride = conv_dim + c.ssm_d_inner + 2 * c.ssm_dt_rank + 2 * c.n_head * c.head_dim + 2 * c.n_head_kv * c.head_dim;
        const int R = MAX_NT, K = c.n_expert_used;
        dev.x = dev.alloc<float>((size_t) R * n);
        dev.res = dev.alloc<float>((size_t) R * hcn);
        dev.res2 = dev.alloc<float>((size_t) R * hcn);
        dev.xn = dev.alloc<float>((size_t) R * hcn);
        dev.gate = dev.alloc<float>((size_t) R * hcn);
        dev.lo = dev.alloc<float>((size_t) R * c.hc_lr);
        dev.inj = dev.alloc<float>((size_t) R * 4);
        dev.mixed = dev.alloc<float>((size_t) R * n);
        dev.bo = dev.alloc<float>((size_t) R * n);
        dev.part = dev.alloc<float>((size_t) R * n);
        dev.big0 = dev.alloc<float>((size_t) R * dev.big_stride);
        dev.o = dev.alloc<float>((size_t) R * std::max(c.ssm_d_inner, c.n_head * c.head_dim));
        dev.attn_part = dev.alloc<float>(attn_part_floats(c.n_head, c.n_head_kv, 1, c.head_dim) * 8 + (size_t) R * 64 * 1024);
        dev.rlog = dev.alloc<float>((size_t) R * (c.n_expert + 1));
        dev.ids = dev.alloc<int>((size_t) R * K);
        dev.wts = dev.alloc<float>((size_t) R * K);
        dev.sg = dev.alloc<float>(R);
        dev.shgu = dev.alloc<float>((size_t) R * 2 * c.n_ff_shexp);
        dev.shh = dev.alloc<float>((size_t) R * c.n_ff_shexp);
        dev.shpart = dev.alloc<float>((size_t) R * n);
        dev.hexp = dev.alloc<float>((size_t) R * K * c.n_ff_exp);
        dev.yexp = dev.alloc<float>((size_t) R * K * n);
        dev.ple_emb = dev.alloc<float>((size_t) R * std::max(1, c.ple_n_heads() * c.ple_dim));
        dev.ple_key = dev.alloc<float>((size_t) R * hcn);
        dev.ple_val = dev.alloc<float>((size_t) R * n);
        dev.ple_sc = dev.alloc<float>((size_t) R * c.hc * 4);
        dev.logits = dev.alloc<float>((size_t) R * dev.output.n);
        dev.pos = dev.alloc<int>(1);
        dev.counter = dev.alloc<int>(1);
        fprintf(stderr, "hyper4: device %d holds %.2f GiB\n", dev.id, dev.used / 1073741824.0);
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "hyper4: weights loaded in %.1f s (%d GPUs, %.0f%% of experts on GPU)\n", s, nd, 100.0 * opt_.gpu_expert_frac);
}

void Engine4::reset() {
    const Q4Config & c = cfg_;
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        for (auto & L : dp->layers) {
            if (!L.full) {
                CUDA_CHECK(cudaMemset(L.conv_state, 0, (size_t) (c.ssm_conv - 1) * L.conv_ch * sizeof(float)));
                CUDA_CHECK(cudaMemset(L.state, 0, (size_t) L.n_v_l * c.ssm_d_state * c.head_v_dim() * sizeof(float)));
            }
            if (L.ple) CUDA_CHECK(cudaMemset(L.ple_state, 0, (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim() * sizeof(float)));
        }
    }
    seq_.clear();
}

void Engine4::record_main(int gi, int nt) {
    const Q4Config & c = cfg_;
    Device & d = *devs_[gi];
    cudaStream_t s = d.stream;
    const int n = c.n_embd, hc = c.hc, hcn = c.hc_dim(), lr = c.hc_lr, K = c.n_expert_used, bs = d.big_stride;
    const int nd = opt_.n_devices;
    const float eps = c.rms_eps;
    CUDA_CHECK(cudaMemcpyAsync(d.pos, h_pos_, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.x, h_embd_, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
    if (c.ple_layer >= 0)
        CUDA_CHECK(cudaMemcpyAsync(d.ple_emb, h_ple_, (size_t) nt * c.ple_n_heads() * c.ple_dim * sizeof(float), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    hc_init(d.res, d.x, n, hc, nt, s);
    int call = 0;
    // debug (HYPER4_DEBUG, direct recording, single GPU): sync after a stage, report errors and non-finite values
    auto dbg = [&](const char * what, int il, const float * buf, size_t cnt) {
        if (!debug_) return;
        CUDA_CHECK(cudaStreamSynchronize(s));
        const cudaError_t err = cudaGetLastError();
        std::vector<float> h(cnt);
        CUDA_CHECK(cudaMemcpy(h.data(), buf, cnt * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0; double mx = 0;
        for (float v : h) { if (!std::isfinite(v)) ++bad; else mx = std::max(mx, (double) std::fabs(v)); }
        double sum = 0, sq = 0;
        for (float v : h) { sum += v; sq += (double) v * v; }
        if (bad || err != cudaSuccess || (il < 2 && !getenv("HYPER4_SUMS")))
            fprintf(stderr, "dbg L%d %-12s err=%s nonfinite=%d max|x|=%.3g\n", il, what, cudaGetErrorString(err), bad, mx);
        if (getenv("HYPER4_SUMS")) fprintf(stderr, "SUMS %d %s %.6g %.6g\n", il, what, sum, sq);
        if (bad || err != cudaSuccess) throw std::runtime_error("debug stop");
    };
    dbg("embed", -1, d.res, (size_t) nt * hcn);
    auto allreduce = [&] {
        CUDA_CHECK(cudaMemsetAsync(d.bo, 0, (size_t) nt * n * sizeof(float), s));
        allreduce_add_ll16(d.bo, d.part, ar_ll_, d.g, nd, nt * n, d.counter, call++, s, nullptr);
    };
    // hyper-connection mixer: res -> mixed (and the inject logits)
    auto hc_mix = [&](const float * norm, const Q8W & down, const Q8W & up, const float * inj) {
        hc_norm(d.res, norm, d.xn, n, hc, eps, nt, s);
        mmq(down, d.xn, hcn, d.lo, lr, nt, s);
        silu_scale(d.lo, lr, 1.0f / hc, nt, lr, s);
        mmq(up, d.lo, lr, d.gate, hcn, nt, s);
        if (inj) gemv_f32(inj, hc, hcn, d.xn, hcn, d.inj, 4, nt, s);
        hc_mixed(d.xn, d.gate, d.mixed, n, hc, nt, s);
    };
    for (int il = 0; il < c.n_layer; ++il) {
        DevLayer & L = d.layers[il];
        if (L.ple) {
            const int pe = c.ple_n_heads() * c.ple_dim;
            mmq(L.ple_key, d.ple_emb, pe, d.ple_key, hcn, nt, s);
            mmq(L.ple_value, d.ple_emb, pe, d.ple_val, n, nt, s);
            ple_apply(d.res, d.ple_key, d.ple_val, L.ple_wk, L.ple_wq, L.ple_wc, L.ple_conv, L.ple_state, nullptr, n, hc, c.ple_conv,
                      c.ple_ngram, eps, nt, d.ple_sc, s);
        }
        if (L.ple) dbg("ple", il, d.res, (size_t) nt * hcn);
        // ---- token mixer ----
        hc_mix(L.hca_norm, L.hca_down, L.hca_up, L.hca_inj);
        dbg("hc_mix_attn", il, d.mixed, (size_t) nt * n);
        if (L.full) {
            mmq(L.wqkv, d.mixed, n, d.big0, bs, nt, s);
            attn_prep(d.big0, bs, L.q_norm, L.k_norm, L.kcache, L.vcache, d.pos, opt_.max_pos, L.n_head_l, L.n_kv_l, c.head_dim,
                      c.n_rot, c.rope_base, eps, nt, s);
            const int ostride = L.n_head_l * c.head_dim;
            if (getenv("HYPER4_OLDATTN"))
                attn_decode(d.big0, bs, L.kcache, L.vcache, d.o, ostride, d.pos, opt_.max_pos, L.n_head_l, L.n_kv_l, L.head_off,
                            c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s);
            else
                attn_split(d.big0, bs, L.kcache, L.vcache, d.attn_part, d.o, ostride, d.pos, opt_.max_pos, L.n_head_l, L.n_kv_l,
                           L.head_off, c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s);
            mmq(L.wo, d.o, ostride, d.part, n, nt, s);
        } else {
            const int dv = c.head_v_dim();
            mmq(L.win, d.mixed, n, d.big0, bs, nt, s);
            const int z_off = L.conv_ch, ab_off = L.conv_ch + L.n_v_l * dv;
            gemv_f32(L.ab_w, 2 * L.n_v_l, n, d.mixed, n, d.big0 + ab_off, bs, nt, s);
            gdn_conv(d.big0, bs, L.conv_state, nullptr, L.conv_w, L.conv_ch, c.ssm_conv, nt, s);
            const int ostride = L.n_v_l * dv;
            gdn_step(d.big0, bs, ab_off, L.state, nullptr, d.o, ostride, L.dt_bias, L.ssm_a, L.n_k_l, L.n_v_l, c.ssm_d_state, dv, eps, nt, s);
            gated_norm_sigmoid(d.o, ostride, d.big0 + z_off, bs, L.ssm_norm, L.n_v_l, dv, eps, nt, s);
            mmq(L.wout, d.o, ostride, d.part, n, nt, s);
        }
        dbg(L.full ? "attn_part" : "gdn_part", il, d.part, (size_t) nt * n);
        allreduce();
        hc_combine(d.res, d.bo, d.inj, 4, n, hc, nt, s);
        // ---- MoE ----
        hc_mix(L.hcf_norm, L.hcf_down, L.hcf_up, L.hcf_inj);
        gemv_f32(L.router, c.n_expert + 1, n, d.mixed, n, d.rlog, c.n_expert + 1, nt, s);
        dbg("router", il, d.rlog, (size_t) nt * (c.n_expert + 1));
        moe_route(d.rlog, c.n_expert + 1, c.n_expert, K, d.ids, d.wts, d.sg, nt, s);
        dbg("route_w", il, d.wts, (size_t) nt * K);
        if (d.g == 0) moe_publish(&cpu_rec_[il], d.mixed, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
        mmq(L.sh_gu, d.mixed, n, d.shgu, 2 * L.n_sh_l, nt, s);
        silu_mul(d.shgu, 2 * L.n_sh_l, d.shh, c.n_ff_shexp, L.n_sh_l, nt, s);
        mmq(L.sh_down, d.shh, c.n_ff_shexp, d.shpart, n, nt, s);
        moe_gate_up(L.moe, d.mixed, n, d.ids, K, d.hexp, nt, s);
        moe_down(L.moe, d.hexp, d.ids, d.wts, K, d.yexp, nt, s);
        dbg("shexp", il, d.shpart, (size_t) nt * n);
        dbg("experts", il, d.yexp, (size_t) nt * K * n);
        moe_reduce(d.shpart, d.sg, d.yexp, K, d.part, n, nt, d.ids, L.owner, CPU_OWNER, d.g == 0 ? &cpu_out_[il].seq : nullptr,
                   cpu_out_[il].y[0], d.counter, (unsigned) il, s);
        dbg("moe_part", il, d.part, (size_t) nt * n);
        allreduce();
        hc_combine(d.res, d.bo, d.inj, 4, n, hc, nt, s);
        dbg("l_last", il, d.res, (size_t) nt * hcn);
    }
    // final mixer = output norm
    hc_mix(d.head_norm, d.head_down, d.head_up, nullptr);
    mmq(d.output, d.mixed, n, d.logits, d.output.n, nt, s);
    argmax_pairs(d.logits, d.output.n, d.output.n, d.vocab_off, d.wts, nt, s);   // wts reused as the result pairs
    CUDA_CHECK(cudaMemcpyAsync(h_res_ + (size_t) gi * MAX_NT * 2, d.wts, (size_t) nt * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
}

void Engine4::build_graphs() {
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        for (int nt = 1; nt <= MAX_NT; ++nt) {
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
            record_main(gi, nt);
            CUDA_CHECK(cudaStreamEndCapture(d.stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&d.g_main[nt], graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    graphs_ready_ = true;
}

// token embeddings (Q8_0 rows) and the PLE n-gram hash rows (IQ4_NL), for positions pos..pos+nt-1
void Engine4::embed(const int * tokens, int nt, int pos) {
    const Q4Config & c = cfg_;
    if ((int) seq_.size() < pos + nt) seq_.resize(pos + nt, -1);
    for (int t = 0; t < nt; ++t) seq_[pos + t] = tokens[t];
    const GTensor & te = gguf_->need("token_embd.weight");
    if (te.type != GType::Q8_0) throw std::runtime_error("token_embd: expected Q8_0");
    for (int t = 0; t < nt; ++t) {
        const uint8_t * row = te.data + (size_t) tokens[t] * te.row_bytes();
        float * out = h_embd_ + (size_t) t * c.n_embd;
        for (int b = 0; b < c.n_embd / 32; ++b) {
            const uint8_t * blk = row + b * 34;
            const float dd = f16(blk);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = dd * (float) ((const int8_t *) (blk + 2))[i];
        }
    }
    if (c.ple_layer < 0) return;
    const GTensor & pt = gguf_->need("per_layer_token_embd.weight");
    if (pt.type != GType::IQ4_NL) throw std::runtime_error("per_layer_token_embd: expected IQ4_NL");
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
        float * out = h_ple_ + (size_t) t * nh * dim;
        for (int ngr = 2; ngr <= ng; ++ngr) {
            uint64_t mixed = (uint64_t) ctx[0] * c.ple_mult[0];
            for (int j = 1; j < ngr; ++j) mixed ^= (uint64_t) ctx[j] * c.ple_mult[j];
            for (int gq = 0; gq < c.ple_heads_per_ngram; ++gq) {
                const int h = (ngr - 2) * c.ple_heads_per_ngram + gq;
                const uint64_t row = mixed % c.ple_vocab[h] + c.ple_offsets[h];
                const uint8_t * r = pt.data + row * pt.row_bytes();
                for (int b = 0; b < dim / 32; ++b) {
                    const uint8_t * blk = r + b * 18;
                    const float dd = f16(blk);
                    for (int j = 0; j < 16; ++j) {
                        out[h * dim + b * 32 + j] = dd * kIQ4NL[blk[2 + j] & 0xF];
                        out[h * dim + b * 32 + j + 16] = dd * kIQ4NL[blk[2 + j] >> 4];
                    }
                }
            }
        }
    }
}

std::vector<int> Engine4::forward(const int * tokens, int nt, int pos) {
    if (nt < 1 || nt > MAX_NT) throw std::runtime_error("forward: bad token count");
    if (pos + nt > opt_.max_pos) throw std::runtime_error("forward: position exceeds max_pos");
    if (!graphs_ready_ && !debug_) build_graphs();
    embed(tokens, nt, pos);
    h_pos_[0] = pos;
    ++fwd_counter_;
    std::vector<int> slots(cfg_.n_layer);
    for (int i = 0; i < cfg_.n_layer; ++i) slots[i] = i;
    cpu_->expect(fwd_counter_, slots);
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        auto & dp = devs_[gi];
        CUDA_CHECK(cudaSetDevice(dp->id));
        if (debug_) record_main(gi, nt);
        else CUDA_CHECK(cudaGraphLaunch(dp->g_main[nt], dp->stream));
    }
    // wait; after 10 s report the hand-off state (hang diagnostics)
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
                fprintf(stderr, "hyper4: forward %u stuck on device %d; cpu: %s\n", fwd_counter_, dp->id, cpu_->state().c_str());
                for (int l = 0; l < cfg_.n_layer; ++l)
                    fprintf(stderr, "  L%-2d rec %u (fwd %u slot %u)  out %u (fwd %u)\n", l, cpu_rec_[l].seq, cpu_rec_[l].seq / 64,
                            cpu_rec_[l].seq % 64, cpu_out_[l].seq, cpu_out_[l].seq / 64);
            }
            std::this_thread::yield();
        }
    }
    last_nt_ = nt;
    std::vector<int> out(nt);
    for (int t = 0; t < nt; ++t) {
        float best = -INFINITY; int bi = -1;
        for (size_t g = 0; g < devs_.size(); ++g) {
            const float v = h_res_[(g * MAX_NT + t) * 2];
            const int idx = ((const int *) h_res_)[(g * MAX_NT + t) * 2 + 1];
            if (v > best) { best = v; bi = idx; }
        }
        out[t] = bi;
    }
    return out;
}

void Engine4::get_logits(int t, std::vector<float> & out) {
    out.resize(cfg_.n_vocab);
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaMemcpy(out.data() + dp->vocab_off, dp->logits + (size_t) t * dp->output.n, (size_t) dp->output.n * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

} // namespace hyper
