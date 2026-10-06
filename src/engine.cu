#include "engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

namespace hyper {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

// one GPU's slice of one layer
struct Engine::DevLayer {
    bool full = false;
    float * attn_norm = nullptr, * post_norm = nullptr;
    // gated attention: local q heads [head_off, head_off + n_head_l), local kv heads [kv_off, kv_off + n_kv_l)
    BF16W wqkv;               // rows: [q|gate per local head | k local | v local]
    Q8W wo;
    float * q_norm = nullptr, * k_norm = nullptr;
    half * kcache = nullptr, * vcache = nullptr;
    int n_head_l = 0, head_off = 0, n_kv_l = 0, kv_off = 0;
    // gated delta net: all q/k heads (replicated) + local v heads
    Q8W win, wout;            // win rows: [q k v (local) | z | alpha | beta]
    float * conv_w = nullptr, * dt_bias = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    float * conv_state = nullptr, * state = nullptr;
    int n_v_l = 0, n_k_l = 0, conv_ch = 0;
    // ffn: local slice of the hidden dimension
    Q8W ffn_gu, ffn_down;     // gate and up rows stacked: output [gate | up]
    int n_ff_l = 0;
};

struct Engine::Device {
    int id = 0, g = 0;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph = nullptr;
    std::vector<DevLayer> layers;
    BF16W output;                 // vocab slice
    int vocab_off = 0;
    float * output_norm = nullptr;
    float * x = nullptr, * xn = nullptr, * part = nullptr;
    float * big0 = nullptr, * big1 = nullptr, * kbuf = nullptr, * vbuf = nullptr, * ab = nullptr, * o = nullptr, * h = nullptr;
    float * logits = nullptr, * res = nullptr, * ss = nullptr;
    int * pos = nullptr, * counter = nullptr;
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
    ~Device() {
        cudaSetDevice(id);
        if (graph) cudaGraphExecDestroy(graph);
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

namespace {

struct RowRange { const GTensor * t; int64_t r0, r1; };

// Q8_0 rows from several tensors (same k), restricted to column blocks [cb0, cb1), repacked
using ColRanges = std::vector<std::pair<int64_t, int64_t>>;   // column-block ranges [b0, b1)

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
    CUDA_CHECK(cudaSetDevice(dev));
    Q8W w; w.n = (int) n; w.k = (int) k;
    int8_t * dq = (int8_t *) alloc(qs.size());
    half * dd = (half *) alloc(d.size() * sizeof(half));
    CUDA_CHECK(cudaMemcpy(dq, qs.data(), qs.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd, d.data(), d.size() * sizeof(half), cudaMemcpyHostToDevice));
    w.qs = dq; w.d = dd;
    return w;
}

// split [0, n) into ndev contiguous parts aligned to `align`
std::pair<int64_t, int64_t> split(int64_t n, int ndev, int g, int64_t align = 1) {
    const int64_t units = n / align;
    return {units * g / ndev * align, units * (g + 1) / ndev * align};
}

} // namespace

Engine::Engine(const std::string & model_path, const EngineOptions & opt) : opt_(opt) {
    gguf_ = std::make_unique<GGUF>(model_path);
    cfg_ = Qwen35Config::from_gguf(*gguf_);
    fprintf(stderr, "hyper: %s\n", cfg_.describe().c_str());
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
    const int nd = opt_.n_devices;
    CUDA_CHECK(cudaHostAlloc(&h_embd_, cfg_.n_embd * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_pos_, sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_res_, nd * 2 * sizeof(float), cudaHostAllocPortable));
    const int nchunk = (cfg_.n_embd + AR_CHUNK - 1) / AR_CHUNK;
    CUDA_CHECK(cudaHostAlloc(&ar_slots_, (size_t) 2 * nd * cfg_.n_embd * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&ar_flags_, (size_t) nd * nchunk * sizeof(unsigned long long), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_flags_, 0, (size_t) nd * nchunk * sizeof(unsigned long long));
    CUDA_CHECK(cudaHostAlloc(&ar_ll_, (size_t) 2 * nd * cfg_.n_embd * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_ll_, 0xff, (size_t) 2 * nd * cfg_.n_embd * sizeof(uint2));
    load_weights();
}

Engine::~Engine() {
    devs_.clear();
    if (h_embd_) cudaFreeHost(h_embd_);
    if (h_pos_) cudaFreeHost(h_pos_);
    if (h_res_) cudaFreeHost(h_res_);
    if (ar_slots_) cudaFreeHost(ar_slots_);
    if (ar_flags_) cudaFreeHost(ar_flags_);
    if (ar_ll_) cudaFreeHost(ar_ll_);
}

void Engine::load_weights() {
    auto t0 = std::chrono::steady_clock::now();
    const Qwen35Config & c = cfg_;
    const int nd = opt_.n_devices;
    const int dk = c.ssm_d_state, dv = c.head_v_dim(), nk = c.ssm_n_group, nv = c.ssm_dt_rank;
    if (c.n_head % nd) throw std::runtime_error("attention head count must divide the device count");

    for (auto & dp : devs_) {
        Device & dev = *dp;
        const int g = dev.g;
        auto A = [&](size_t n) { return (void *) dev.alloc<uint8_t>(n); };
        auto f32 = [&](const std::string & name, int64_t e0 = 0, int64_t e1 = -1) {
            const GTensor & t = gguf_->need(name);
            if (t.type != GType::F32) throw std::runtime_error("expected F32: " + name);
            if (e1 < 0) e1 = t.nelements();
            float * p = dev.alloc<float>(e1 - e0);
            CUDA_CHECK(cudaMemcpy(p, (const float *) t.data + e0, (e1 - e0) * sizeof(float), cudaMemcpyHostToDevice));
            return p;
        };
        auto bf16 = [&](const std::string & name, int64_t r0, int64_t r1) {
            const GTensor & t = gguf_->need(name);
            if (t.type != GType::BF16) throw std::runtime_error("expected BF16: " + name);
            BF16W w; w.k = (int) t.ne[0]; w.n = (int) (r1 - r0);
            __nv_bfloat16 * p = dev.alloc<__nv_bfloat16>((size_t) w.n * w.k);
            CUDA_CHECK(cudaMemcpy(p, t.data + (size_t) r0 * t.row_bytes(), (size_t) w.n * t.row_bytes(), cudaMemcpyHostToDevice));
            w.w = p;
            return w;
        };
        auto T = [&](const std::string & name) { return &gguf_->need(name); };
        auto bf16_rows = [&](const std::vector<RowRange> & parts) {
            BF16W w; w.k = (int) parts[0].t->ne[0]; w.n = 0;
            for (auto & pr : parts) w.n += (int) (pr.r1 - pr.r0);
            __nv_bfloat16 * pd = dev.alloc<__nv_bfloat16>((size_t) w.n * w.k);
            size_t off = 0;
            for (auto & pr : parts) {
                if (pr.t->type != GType::BF16 || pr.t->ne[0] != w.k) throw std::runtime_error("bf16_rows: bad tensor " + pr.t->name);
                const size_t bytes = (size_t) (pr.r1 - pr.r0) * pr.t->row_bytes();
                CUDA_CHECK(cudaMemcpy((uint8_t *) pd + off, pr.t->data + (size_t) pr.r0 * pr.t->row_bytes(), bytes, cudaMemcpyHostToDevice));
                off += bytes;
            }
            w.w = pd;
            return w;
        };

        dev.layers.resize(c.n_layer);
        for (int il = 0; il < c.n_layer; ++il) {
            DevLayer & L = dev.layers[il];
            const std::string p = "blk." + std::to_string(il) + ".";
            L.full = c.is_full_attn(il);
            L.attn_norm = f32(p + "attn_norm.weight");
            L.post_norm = f32(p + "post_attention_norm.weight");
            if (L.full) {
                const int hd = c.head_dim, group = c.n_head / c.n_head_kv;
                L.n_head_l = c.n_head / nd;
                L.head_off = g * L.n_head_l;
                L.kv_off = L.head_off / group;
                L.n_kv_l = (L.head_off + L.n_head_l - 1) / group - L.kv_off + 1;
                L.wqkv = bf16_rows({{T(p + "attn_q.weight"), (int64_t) L.head_off * 2 * hd, (int64_t) (L.head_off + L.n_head_l) * 2 * hd},
                                    {T(p + "attn_k.weight"), (int64_t) L.kv_off * hd, (int64_t) (L.kv_off + L.n_kv_l) * hd},
                                    {T(p + "attn_v.weight"), (int64_t) L.kv_off * hd, (int64_t) (L.kv_off + L.n_kv_l) * hd}});
                // attn_output: k = n_head*hd; local column blocks of the local heads
                L.wo = upload_q8(A, dev.id, {{T(p + "attn_output.weight"), 0, c.n_embd}},
                                 {{(int64_t) L.head_off * hd / 32, (int64_t) (L.head_off + L.n_head_l) * hd / 32}});
                L.q_norm = f32(p + "attn_q_norm.weight");
                L.k_norm = f32(p + "attn_k_norm.weight");
                const size_t kv = (size_t) L.n_kv_l * opt_.max_pos * hd;
                L.kcache = dev.alloc<half>(kv);
                L.vcache = dev.alloc<half>(kv);
            } else {
                // head-aligned partition: v-head h reads k-head h % nk, so a GPU owning k-heads [k0, k1)
                // takes v-heads {grp * nk + k} for k in [k0, k1); local v order is k-major per group,
                // which keeps the kernel's local mapping (v-head hl -> k-head hl % n_k_l) valid.
                auto [k0, k1] = split(nk, nd, g);
                const int n_k_l = (int) (k1 - k0);
                std::vector<int> vh;
                for (int grp = 0; grp < nv / nk; ++grp) for (int64_t kk = k0; kk < k1; ++kk) vh.push_back(grp * nk + (int) kk);
                L.n_k_l = n_k_l;
                L.n_v_l = (int) vh.size();
                const int64_t qoff = 0, koff = (int64_t) nk * dk, voff = (int64_t) 2 * nk * dk;
                L.conv_ch = 2 * n_k_l * dk + L.n_v_l * dv;
                const GTensor * qkv = T(p + "attn_qkv.weight");
                const GTensor * zt = T(p + "attn_gate.weight");
                std::vector<RowRange> qkv_rows = {{qkv, qoff + k0 * dk, qoff + k1 * dk}, {qkv, koff + k0 * dk, koff + k1 * dk}};
                std::vector<RowRange> z_rows, a_rows, b_rows;
                ColRanges out_cols;
                std::vector<int64_t> chans;   // conv channels in local order
                for (int64_t ch = qoff + k0 * dk; ch < qoff + k1 * dk; ++ch) chans.push_back(ch);
                for (int64_t ch = koff + k0 * dk; ch < koff + k1 * dk; ++ch) chans.push_back(ch);
                for (int h : vh) {
                    qkv_rows.push_back({qkv, voff + (int64_t) h * dv, voff + (int64_t) (h + 1) * dv});
                    z_rows.push_back({zt, (int64_t) h * dv, (int64_t) (h + 1) * dv});
                    a_rows.push_back({T(p + "ssm_alpha.weight"), h, h + 1});
                    b_rows.push_back({T(p + "ssm_beta.weight"), h, h + 1});
                    out_cols.push_back({(int64_t) h * dv / 32, (int64_t) (h + 1) * dv / 32});
                    for (int64_t ch = voff + (int64_t) h * dv; ch < voff + (int64_t) (h + 1) * dv; ++ch) chans.push_back(ch);
                }
                std::vector<RowRange> ab_rows = a_rows; ab_rows.insert(ab_rows.end(), b_rows.begin(), b_rows.end());
                std::vector<RowRange> in_rows = qkv_rows;
                in_rows.insert(in_rows.end(), z_rows.begin(), z_rows.end());
                in_rows.insert(in_rows.end(), ab_rows.begin(), ab_rows.end());
                L.win = upload_q8(A, dev.id, in_rows);
                L.wout = upload_q8(A, dev.id, {{T(p + "ssm_out.weight"), 0, c.n_embd}}, out_cols);
                {
                    const GTensor & cw = gguf_->need(p + "ssm_conv1d.weight");
                    const int K = c.ssm_conv;
                    std::vector<float> buf((size_t) L.conv_ch * K);
                    for (size_t j = 0; j < chans.size(); ++j)
                        memcpy(buf.data() + j * K, (const float *) cw.data + chans[j] * K, K * sizeof(float));
                    L.conv_w = dev.alloc<float>(buf.size());
                    CUDA_CHECK(cudaMemcpy(L.conv_w, buf.data(), buf.size() * sizeof(float), cudaMemcpyHostToDevice));
                }
                auto gather = [&](const std::string & name) {
                    const GTensor & t = gguf_->need(name);
                    std::vector<float> buf;
                    for (int h : vh) buf.push_back(((const float *) t.data)[h]);
                    float * pd = dev.alloc<float>(buf.size());
                    CUDA_CHECK(cudaMemcpy(pd, buf.data(), buf.size() * sizeof(float), cudaMemcpyHostToDevice));
                    return pd;
                };
                L.dt_bias = gather(p + "ssm_dt.bias");
                L.ssm_a = gather(p + "ssm_a");
                L.ssm_norm = f32(p + "ssm_norm.weight");
                L.conv_state = dev.alloc<float>((size_t) (c.ssm_conv - 1) * L.conv_ch);
                L.state = dev.alloc<float>((size_t) L.n_v_l * dk * dv);
            }
            // ffn: hidden dim split in 32-blocks; gate/up rows and down column blocks match
            auto [f0, f1] = split(c.n_ff, nd, g, 32);
            L.n_ff_l = (int) (f1 - f0);
            L.ffn_gu = upload_q8(A, dev.id, {{T(p + "ffn_gate.weight"), f0, f1}, {T(p + "ffn_up.weight"), f0, f1}});
            L.ffn_down = upload_q8(A, dev.id, {{T(p + "ffn_down.weight"), 0, c.n_embd}}, {{f0 / 32, f1 / 32}});
        }
        auto [o0, o1] = split(c.n_vocab, nd, g);
        dev.vocab_off = (int) o0;
        dev.output = bf16("output.weight", o0, o1);
        dev.output_norm = f32("output_norm.weight");

        dev.x = dev.alloc<float>(c.n_embd);
        dev.xn = dev.alloc<float>(c.n_embd);
        dev.part = dev.alloc<float>(c.n_embd);
        const size_t big = std::max<size_t>({(size_t) c.conv_dim(), (size_t) 2 * c.n_head * c.head_dim, (size_t) 2 * c.n_ff});
        dev.big0 = dev.alloc<float>(big);
        dev.big1 = dev.alloc<float>(big);
        dev.kbuf = dev.alloc<float>((size_t) c.n_head_kv * c.head_dim);
        dev.vbuf = dev.alloc<float>((size_t) c.n_head_kv * c.head_dim);
        dev.ab = dev.alloc<float>(2 * nv);
        dev.o = dev.alloc<float>(std::max<size_t>((size_t) c.ssm_d_inner, (size_t) c.n_head * c.head_dim));
        dev.h = dev.alloc<float>(c.n_ff);
        dev.logits = dev.alloc<float>(dev.output.n);
        dev.res = dev.alloc<float>(2);
        dev.ss = dev.alloc<float>(64);
        dev.pos = dev.alloc<int>(1);
        dev.counter = dev.alloc<int>(1);
        fprintf(stderr, "hyper: device %d holds %.2f GiB\n", dev.id, dev.used / 1073741824.0);
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "hyper: weights loaded in %.1f s (tensor parallel over %d GPUs)\n", s, nd);
}

void Engine::reset() {
    const Qwen35Config & c = cfg_;
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        for (auto & L : dp->layers) {
            if (L.full) continue;
            CUDA_CHECK(cudaMemset(L.conv_state, 0, (size_t) (c.ssm_conv - 1) * L.conv_ch * sizeof(float)));
            CUDA_CHECK(cudaMemset(L.state, 0, (size_t) L.n_v_l * c.ssm_d_state * c.head_v_dim() * sizeof(float)));
        }
    }
}

void Engine::record(int gi) {
    const Qwen35Config & c = cfg_;
    Device & d = *devs_[gi];
    const int nd = opt_.n_devices, g = d.g;
    cudaStream_t s = d.stream;
    const float eps = c.rms_eps;
    CUDA_CHECK(cudaMemcpyAsync(d.pos, h_pos_, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.x, h_embd_, c.n_embd * sizeof(float), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    int call = 0;
    auto reduce = [&]() {
        if (opt_.ar_mode == 2) allreduce_add_ll16(d.x, d.part, ar_ll_, g, nd, c.n_embd, d.counter, call++, s, d.ss);
        else if (opt_.ar_mode == 1) allreduce_add_ll(d.x, d.part, ar_ll_, g, nd, c.n_embd, d.counter, call++, s);
        else allreduce_add(d.x, d.part, ar_slots_, ar_flags_, g, nd, c.n_embd, d.counter, call++, s);
    };
    // fused-norm input statistics: from the allreduce (ll16) or an explicit sum-of-squares kernel
    const bool fused_stats = opt_.ar_mode == 2;
    const int nss = fused_stats ? allreduce_ll16_nss(c.n_embd) : 1;
    auto stats = [&]() { if (!fused_stats) sumsq(d.x, c.n_embd, d.ss, s); };
    auto nin = [&](const float * w) { NormIn r; r.w = w; r.ss = d.ss; r.nss = nss; r.eps = eps; return r; };
    sumsq(d.x, c.n_embd, d.ss, s);
    if (fused_stats) {  // first layer reads ss[0..nss): zero the rest after the explicit sum
        CUDA_CHECK(cudaMemsetAsync(d.ss + 1, 0, (nss - 1) * sizeof(float), s));
    }
    for (int il = 0; il < c.n_layer; ++il) {
        DevLayer & L = d.layers[il];
        if (L.full) {
            gemv_bf16(L.wqkv, d.x, d.big0, nullptr, s, nin(L.attn_norm));
            float * kb = d.big0 + (size_t) L.n_head_l * 2 * c.head_dim;
            float * vb = kb + (size_t) L.n_kv_l * c.head_dim;
            attn_prep(d.big0, kb, vb, L.q_norm, L.k_norm, L.kcache, L.vcache, d.pos, opt_.max_pos,
                      L.n_head_l, L.n_kv_l, c.head_dim, c.n_rot, c.rope_base, eps, s);
            attn_decode(d.big0, L.kcache, L.vcache, d.o, d.pos, opt_.max_pos, L.n_head_l, L.head_off,
                        c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), s);
            gemv_q8(L.wo, d.o, d.part, nullptr, s);
        } else {
            gemv_q8(L.win, d.x, d.big0, nullptr, s, nin(L.attn_norm));
            float * z = d.big0 + L.conv_ch;
            float * ab = z + (size_t) L.n_v_l * c.head_v_dim();
            gdn_conv(d.big0, L.conv_state, L.conv_w, L.conv_ch, c.ssm_conv, s);
            gdn_step(d.big0, ab, L.dt_bias, L.ssm_a, L.state, d.o, L.n_k_l, L.n_v_l, c.ssm_d_state, c.head_v_dim(), eps, s);
            gated_norm(d.o, z, L.ssm_norm, L.n_v_l, c.head_v_dim(), eps, s);
            gemv_q8(L.wout, d.o, d.part, nullptr, s);
        }
        reduce();
        stats();
        gemv_q8(L.ffn_gu, d.x, d.big1, nullptr, s, nin(L.post_norm));
        silu_mul(d.big1, d.h, L.n_ff_l, s);
        gemv_q8(L.ffn_down, d.h, d.part, nullptr, s);
        reduce();
        stats();
    }
    gemv_bf16(d.output, d.x, d.logits, nullptr, s, nin(d.output_norm));
    argmax_pair(d.logits, d.output.n, d.vocab_off, d.res, s);
    CUDA_CHECK(cudaMemcpyAsync(h_res_ + 2 * gi, d.res, 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
}

void Engine::build_graphs() {
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
        record(gi);
        CUDA_CHECK(cudaStreamEndCapture(d.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&d.graph, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    graphs_ready_ = true;
}

void Engine::decode(int token, int pos) {
    const Qwen35Config & c = cfg_;
    if (pos >= opt_.max_pos) throw std::runtime_error("decode: position exceeds max_pos");
    {
        const GTensor & te = gguf_->need("token_embd.weight");
        const uint8_t * row = te.data + (size_t) token * te.row_bytes();
        for (int b = 0; b < c.n_embd / 32; ++b) {
            const uint8_t * blk = row + b * 34;
            half hd; memcpy(&hd, blk, 2);
            const float dd = __half2float(hd);
            for (int i = 0; i < 32; ++i) h_embd_[b * 32 + i] = dd * (float) ((const int8_t *) (blk + 2))[i];
        }
    }
    *h_pos_ = pos;
    if (opt_.use_graphs && !graphs_ready_) build_graphs();
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        if (opt_.use_graphs) CUDA_CHECK(cudaGraphLaunch(d.graph, d.stream));
        else record(gi);
    }
    float best = -INFINITY; int bi = -1;
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        CUDA_CHECK(cudaSetDevice(devs_[gi]->id));
        CUDA_CHECK(cudaStreamSynchronize(devs_[gi]->stream));
        const float v = h_res_[2 * gi];
        const int idx = ((const int *) h_res_)[2 * gi + 1];
        if (v > best) { best = v; bi = idx; }
    }
    last_argmax_ = bi;
}

void Engine::get_logits(std::vector<float> & out) {
    out.resize(cfg_.n_vocab);
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaMemcpy(out.data() + dp->vocab_off, dp->logits, (size_t) dp->output.n * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

} // namespace hyper
