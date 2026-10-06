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
    // gated delta net: head-aligned partition
    Q8W win, wout;            // win rows: [q k v (local) | z | alpha | beta]
    float * conv_w = nullptr, * dt_bias = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    float * conv_state = nullptr, * state = nullptr, * conv_snap = nullptr, * state_snap = nullptr;
    int n_v_l = 0, n_k_l = 0, conv_ch = 0;
    // ffn: local slice of the hidden dimension, gate and up rows stacked
    Q8W ffn_gu, ffn_down;
    int n_ff_l = 0;
    // NextN (MTP) block extras
    BF16W eh_proj;            // column slice [eh_c0, eh_c0 + k) of the 2*n_embd input
    int eh_c0 = 0;
    float * enorm = nullptr, * hnorm = nullptr, * head_norm = nullptr;
};

struct Engine::Device {
    int id = 0, g = 0;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t g_main[MAX_NT + 1] = {}, g_mtp[MAX_NT + 1] = {}, g_restore[MAX_NT] = {}, g_chain = nullptr;
    std::vector<DevLayer> layers;
    DevLayer mtp;
    BF16W output;             // vocab slice
    Q8W output_q8;            // same slice quantized to Q8_0: cheaper LM head for MTP drafts
    int vocab_off = 0;
    float * output_norm = nullptr;
    // activations, MAX_NT rows each
    float * x = nullptr, * part = nullptr, * big0 = nullptr, * big1 = nullptr, * o = nullptr, * h = nullptr;
    float * hn = nullptr, * mhn = nullptr, * me = nullptr, * cat = nullptr;   // mhn: MTP's last normed output
    int big_stride = 0;
    float * logits = nullptr, * res = nullptr, * mres = nullptr, * ss = nullptr;
    int * pos = nullptr, * mpos = nullptr, * counter = nullptr;
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
        for (auto & gr : g_main) if (gr) cudaGraphExecDestroy(gr);
        for (auto & gr : g_mtp) if (gr) cudaGraphExecDestroy(gr);
        for (auto & gr : g_restore) if (gr) cudaGraphExecDestroy(gr);
        if (g_chain) cudaGraphExecDestroy(g_chain);
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

namespace {

struct RowRange { const GTensor * t; int64_t r0, r1; };

// row-major int8 + scales -> fragment-ordered device weight
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
using ColRanges = std::vector<std::pair<int64_t, int64_t>>;   // column-block ranges [b0, b1)

// Q8_0 rows from several tensors (same k), restricted to column blocks, repacked to qs/d arrays
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
    if (opt_.mtp && !gguf_->tensor("blk." + std::to_string(cfg_.n_layer) + ".nextn.eh_proj.weight")) {
        fprintf(stderr, "hyper: no NextN head in the model, MTP disabled\n");
        opt_.mtp = false;
    }
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
    CUDA_CHECK(cudaHostAlloc(&h_membd_, (size_t) MAX_NT * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_pos_, 2 * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_res_, (size_t) nd * MAX_NT * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_mres_, (size_t) nd * MAX_NT * 2 * sizeof(float), cudaHostAllocPortable));
    const size_t ll = (size_t) 2 * nd * MAX_NT * n / 2;
    CUDA_CHECK(cudaHostAlloc(&ar_ll_, ll * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_ll_, 0xff, ll * sizeof(uint2));
    load_weights();
}

Engine::~Engine() {
    devs_.clear();
    for (void * p : {(void *) h_embd_, (void *) h_membd_, (void *) h_pos_, (void *) h_res_, (void *) h_mres_, (void *) ar_ll_})
        if (p) cudaFreeHost(p);
}

void Engine::load_layer(Device & dev, DevLayer & L, int il, bool mtp_layer) {
    const Qwen35Config & c = cfg_;
    const int nd = opt_.n_devices, g = dev.g;
    const int dk = c.ssm_d_state, dv = c.head_v_dim(), nk = c.ssm_n_group, nv = c.ssm_dt_rank;
    auto A = [&](size_t n) { return (void *) dev.alloc<uint8_t>(n); };
    auto T = [&](const std::string & name) { return &gguf_->need(name); };
    auto f32 = [&](const std::string & name) {
        const GTensor & t = gguf_->need(name);
        if (t.type != GType::F32) throw std::runtime_error("expected F32: " + name);
        float * p = dev.alloc<float>(t.nelements());
        CUDA_CHECK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
        return p;
    };
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
    const std::string p = "blk." + std::to_string(il) + ".";
    L.full = mtp_layer || c.is_full_attn(il);
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
        L.wo = upload_q8(A, dev.id, {{T(p + "attn_output.weight"), 0, c.n_embd}},
                         {{(int64_t) L.head_off * hd / 32, (int64_t) (L.head_off + L.n_head_l) * hd / 32}});
        L.q_norm = f32(p + "attn_q_norm.weight");
        L.k_norm = f32(p + "attn_k_norm.weight");
        const size_t kv = (size_t) L.n_kv_l * opt_.max_pos * hd;
        L.kcache = dev.alloc<half>(kv);
        L.vcache = dev.alloc<half>(kv);
    } else {
        // head-aligned partition: v-head h reads k-head h % nk, so a GPU owning k-heads [k0, k1) takes v-heads
        // {grp * nk + k}; local v order is k-major per group, keeping the kernel mapping (hl -> hl % n_k_l) valid
        auto [k0, k1] = split(nk, nd, g);
        std::vector<int> vh;
        for (int grp = 0; grp < nv / nk; ++grp) for (int64_t kk = k0; kk < k1; ++kk) vh.push_back(grp * nk + (int) kk);
        L.n_k_l = (int) (k1 - k0);
        L.n_v_l = (int) vh.size();
        const int64_t qoff = 0, koff = (int64_t) nk * dk, voff = (int64_t) 2 * nk * dk;
        L.conv_ch = 2 * L.n_k_l * dk + L.n_v_l * dv;
        const GTensor * qkv = T(p + "attn_qkv.weight");
        const GTensor * zt = T(p + "attn_gate.weight");
        std::vector<RowRange> rows = {{qkv, qoff + k0 * dk, qoff + k1 * dk}, {qkv, koff + k0 * dk, koff + k1 * dk}};
        std::vector<RowRange> z_rows, a_rows, b_rows;
        ColRanges out_cols;
        std::vector<int64_t> chans;
        for (int64_t ch = qoff + k0 * dk; ch < qoff + k1 * dk; ++ch) chans.push_back(ch);
        for (int64_t ch = koff + k0 * dk; ch < koff + k1 * dk; ++ch) chans.push_back(ch);
        for (int h : vh) {
            rows.push_back({qkv, voff + (int64_t) h * dv, voff + (int64_t) (h + 1) * dv});
            z_rows.push_back({zt, (int64_t) h * dv, (int64_t) (h + 1) * dv});
            a_rows.push_back({T(p + "ssm_alpha.weight"), h, h + 1});
            b_rows.push_back({T(p + "ssm_beta.weight"), h, h + 1});
            out_cols.push_back({(int64_t) h * dv / 32, (int64_t) (h + 1) * dv / 32});
            for (int64_t ch = voff + (int64_t) h * dv; ch < voff + (int64_t) (h + 1) * dv; ++ch) chans.push_back(ch);
        }
        rows.insert(rows.end(), z_rows.begin(), z_rows.end());
        rows.insert(rows.end(), a_rows.begin(), a_rows.end());
        rows.insert(rows.end(), b_rows.begin(), b_rows.end());
        L.win = upload_q8(A, dev.id, rows);
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
        const size_t cs = (size_t) (c.ssm_conv - 1) * L.conv_ch, ssz = (size_t) L.n_v_l * dk * dv;
        L.conv_state = dev.alloc<float>(cs);
        L.state = dev.alloc<float>(ssz);
        if (opt_.mtp) {
            L.conv_snap = dev.alloc<float>(cs * (MAX_NT - 1));
            L.state_snap = dev.alloc<float>(ssz * (MAX_NT - 1));
        }
    }
    auto [f0, f1] = split(c.n_ff, nd, g, 32);
    L.n_ff_l = (int) (f1 - f0);
    L.ffn_gu = upload_q8(A, dev.id, {{T(p + "ffn_gate.weight"), f0, f1}, {T(p + "ffn_up.weight"), f0, f1}});
    L.ffn_down = upload_q8(A, dev.id, {{T(p + "ffn_down.weight"), 0, c.n_embd}}, {{f0 / 32, f1 / 32}});
    if (mtp_layer) {
        // eh_proj: [2*n_embd -> n_embd]; each GPU takes a column slice, partial sums are allreduced
        const GTensor & eh = gguf_->need(p + "nextn.eh_proj.weight");
        if (eh.type != GType::BF16) throw std::runtime_error("nextn.eh_proj: expected BF16");
        auto [c0, c1] = split(eh.ne[0], nd, g, 32);
        L.eh_c0 = (int) c0;
        L.eh_proj.n = (int) eh.ne[1];
        L.eh_proj.k = (int) (c1 - c0);
        __nv_bfloat16 * pd = dev.alloc<__nv_bfloat16>((size_t) L.eh_proj.n * L.eh_proj.k);
        CUDA_CHECK(cudaMemcpy2D(pd, (size_t) L.eh_proj.k * 2, eh.data + c0 * 2, (size_t) eh.ne[0] * 2, (size_t) L.eh_proj.k * 2,
                                L.eh_proj.n, cudaMemcpyHostToDevice));
        L.eh_proj.w = pd;
        L.enorm = f32(p + "nextn.enorm.weight");
        L.hnorm = f32(p + "nextn.hnorm.weight");
        L.head_norm = gguf_->tensor(p + "nextn.shared_head_norm.weight") ? f32(p + "nextn.shared_head_norm.weight") : nullptr;
    }
}

void Engine::load_weights() {
    auto t0 = std::chrono::steady_clock::now();
    const Qwen35Config & c = cfg_;
    const int nd = opt_.n_devices;
    if (c.n_head % nd) throw std::runtime_error("attention head count must divide the device count");
    for (auto & dp : devs_) {
        Device & dev = *dp;
        const int g = dev.g;
        dev.layers.resize(c.n_layer);
        for (int il = 0; il < c.n_layer; ++il) load_layer(dev, dev.layers[il], il, false);
        if (opt_.mtp) load_layer(dev, dev.mtp, c.n_layer, true);
        auto [o0, o1] = split(c.n_vocab, nd, g);
        dev.vocab_off = (int) o0;
        {
            const GTensor & t = gguf_->need("output.weight");
            dev.output.k = (int) t.ne[0]; dev.output.n = (int) (o1 - o0);
            __nv_bfloat16 * pd = dev.alloc<__nv_bfloat16>((size_t) dev.output.n * dev.output.k);
            CUDA_CHECK(cudaMemcpy(pd, t.data + (size_t) o0 * t.row_bytes(), (size_t) dev.output.n * t.row_bytes(), cudaMemcpyHostToDevice));
            dev.output.w = pd;
            if (opt_.mtp) {
                // draft head: quantize the bf16 slice to Q8_0 (absmax per 32 weights); drafts are verified
                // against the exact bf16 head, so this only affects acceptance, never the output
                const int64_t rows = dev.output.n, k = dev.output.k, kb = k / 32;
                std::vector<int8_t> qs((size_t) rows * k);
                std::vector<half> dsc((size_t) rows * kb);
                const uint16_t * src = (const uint16_t *) (t.data + (size_t) o0 * t.row_bytes());
#pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < rows; ++r) {
                    for (int64_t b = 0; b < kb; ++b) {
                        float v[32], amax = 0.0f;
                        for (int e = 0; e < 32; ++e) {
                            const uint32_t bits = (uint32_t) src[(size_t) r * k + b * 32 + e] << 16;
                            float f; memcpy(&f, &bits, 4);
                            v[e] = f; amax = std::max(amax, std::fabs(f));
                        }
                        const float dd = amax / 127.0f, id = dd > 0 ? 1.0f / dd : 0.0f;
                        dsc[(size_t) r * kb + b] = __float2half(dd);
                        for (int e = 0; e < 32; ++e) qs[(size_t) r * k + b * 32 + e] = (int8_t) lrintf(v[e] * id);
                    }
                }
                dev.output_q8 = to_device_q8([&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); }, dev.id,
                                             qs.data(), dsc.data(), (int) rows, (int) k);
            }
        }
        {
            const GTensor & t = gguf_->need("output_norm.weight");
            dev.output_norm = dev.alloc<float>(t.nelements());
            CUDA_CHECK(cudaMemcpy(dev.output_norm, t.data, t.nbytes, cudaMemcpyHostToDevice));
        }
        const int n = c.n_embd;
        dev.big_stride = std::max<int>({c.conv_dim() + c.ssm_d_inner + 2 * c.ssm_dt_rank,
                                        2 * c.n_head * c.head_dim + 2 * c.n_head_kv * c.head_dim, 2 * c.n_ff});
        dev.x = dev.alloc<float>((size_t) MAX_NT * n);
        dev.part = dev.alloc<float>((size_t) MAX_NT * n);
        dev.big0 = dev.alloc<float>((size_t) MAX_NT * dev.big_stride);
        dev.big1 = dev.alloc<float>((size_t) MAX_NT * dev.big_stride);
        dev.o = dev.alloc<float>((size_t) MAX_NT * std::max(c.ssm_d_inner, c.n_head * c.head_dim));
        dev.h = dev.alloc<float>((size_t) MAX_NT * c.n_ff);
        dev.hn = dev.alloc<float>((size_t) MAX_NT * n);
        dev.mhn = dev.alloc<float>(n);
        dev.me = dev.alloc<float>((size_t) MAX_NT * n);
        dev.cat = dev.alloc<float>((size_t) MAX_NT * 2 * n);
        dev.logits = dev.alloc<float>((size_t) MAX_NT * dev.output.n);
        dev.res = dev.alloc<float>(MAX_NT * 2);
        dev.mres = dev.alloc<float>(MAX_NT * 2);
        dev.ss = dev.alloc<float>(MAX_NT * 64);
        dev.pos = dev.alloc<int>(1);
        dev.mpos = dev.alloc<int>(1);
        dev.counter = dev.alloc<int>(1);
        fprintf(stderr, "hyper: device %d holds %.2f GiB\n", dev.id, dev.used / 1073741824.0);
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "hyper: weights loaded in %.1f s (tensor parallel over %d GPUs%s)\n", s, nd, opt_.mtp ? ", MTP head" : "");
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

// ---------------- graph recording ----------------
// Each allreduce also writes the sums of squares of the new residual, feeding the next GEMV's fused RMSNorm.

void Engine::record_attn(Device & d, DevLayer & L, int nt, const int * pos, int & call) {
    const Qwen35Config & c = cfg_;
    cudaStream_t s = d.stream;
    const float eps = c.rms_eps;
    const int n = c.n_embd, nss = n / AR_SS_SPAN, bs = d.big_stride;
    NormIn ni; ni.w = L.attn_norm; ni.ss = d.ss; ni.nss = nss; ni.eps = eps;
    gemv_bf16(L.wqkv, d.x, n, d.big0, bs, nullptr, nt, s, ni);
    attn_prep(d.big0, bs, L.q_norm, L.k_norm, L.kcache, L.vcache, pos, opt_.max_pos, L.n_head_l, L.n_kv_l, c.head_dim,
              c.n_rot, c.rope_base, eps, nt, s);
    const int ostride = L.n_head_l * c.head_dim;
    attn_decode(d.big0, bs, L.kcache, L.vcache, d.o, ostride, pos, opt_.max_pos, L.n_head_l, L.n_kv_l, L.head_off,
                c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s);
    gemv_q8(L.wo, d.o, ostride, d.part, n, nullptr, nt, s);
    allreduce_add_ll16(d.x, d.part, ar_ll_, d.g, opt_.n_devices, nt * n, d.counter, call++, s, d.ss);
}

void Engine::record_gdn(Device & d, DevLayer & L, int nt, bool snap, int & call) {
    const Qwen35Config & c = cfg_;
    cudaStream_t s = d.stream;
    const float eps = c.rms_eps;
    const int n = c.n_embd, nss = n / AR_SS_SPAN, bs = d.big_stride, dv = c.head_v_dim();
    NormIn ni; ni.w = L.attn_norm; ni.ss = d.ss; ni.nss = nss; ni.eps = eps;
    gemv_q8(L.win, d.x, n, d.big0, bs, nullptr, nt, s, ni);
    const int z_off = L.conv_ch, ab_off = L.conv_ch + L.n_v_l * dv;
    gdn_conv(d.big0, bs, L.conv_state, snap ? L.conv_snap : nullptr, L.conv_w, L.conv_ch, c.ssm_conv, nt, s);
    const int ostride = L.n_v_l * dv;
    gdn_step(d.big0, bs, ab_off, L.state, snap ? L.state_snap : nullptr, d.o, ostride, L.dt_bias, L.ssm_a, L.n_k_l, L.n_v_l,
             c.ssm_d_state, dv, eps, nt, s);
    gated_norm(d.o, ostride, d.big0 + z_off, bs, L.ssm_norm, L.n_v_l, dv, eps, nt, s);
    gemv_q8(L.wout, d.o, ostride, d.part, n, nullptr, nt, s);
    allreduce_add_ll16(d.x, d.part, ar_ll_, d.g, opt_.n_devices, nt * n, d.counter, call++, s, d.ss);
}

void Engine::record_ffn(Device & d, DevLayer & L, int nt, int & call) {
    const Qwen35Config & c = cfg_;
    cudaStream_t s = d.stream;
    const int n = c.n_embd, nss = n / AR_SS_SPAN, bs = d.big_stride;
    NormIn ni; ni.w = L.post_norm; ni.ss = d.ss; ni.nss = nss; ni.eps = c.rms_eps;
    gemv_q8(L.ffn_gu, d.x, n, d.big1, bs, nullptr, nt, s, ni);
    silu_mul(d.big1, bs, d.h, c.n_ff, L.n_ff_l, nt, s);
    gemv_q8(L.ffn_down, d.h, c.n_ff, d.part, n, nullptr, nt, s);
    allreduce_add_ll16(d.x, d.part, ar_ll_, d.g, opt_.n_devices, nt * n, d.counter, call++, s, d.ss);
}

void Engine::record_main(int gi, int nt) {
    const Qwen35Config & c = cfg_;
    Device & d = *devs_[gi];
    cudaStream_t s = d.stream;
    const int n = c.n_embd, nss = n / AR_SS_SPAN;
    CUDA_CHECK(cudaMemcpyAsync(d.pos, h_pos_, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.x, h_embd_, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    sumsq(d.x, n, n, nt, d.ss, nss, s);
    int call = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        DevLayer & L = d.layers[il];
        if (L.full) record_attn(d, L, nt, d.pos, call);
        else record_gdn(d, L, nt, opt_.mtp && nt > 1, call);
        record_ffn(d, L, nt, call);
    }
    if (opt_.mtp) rmsnorm(d.x, n, d.output_norm, d.hn, n, n, nt, c.rms_eps, s);   // hidden fed to the MTP head
    NormIn ni; ni.w = d.output_norm; ni.ss = d.ss; ni.nss = nss; ni.eps = c.rms_eps;
    gemv_bf16(d.output, d.x, n, d.logits, d.output.n, nullptr, nt, s, ni);
    argmax_pairs(d.logits, d.output.n, d.output.n, d.vocab_off, d.res, nt, s);
    CUDA_CHECK(cudaMemcpyAsync(h_res_ + (size_t) gi * MAX_NT * 2, d.res, (size_t) nt * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// chain = false: hidden rows come from the main model (hn); chain = true (nt = 1): from MTP's own last output
void Engine::record_mtp(int gi, int nt, bool chain) {
    const Qwen35Config & c = cfg_;
    Device & d = *devs_[gi];
    DevLayer & L = d.mtp;
    cudaStream_t s = d.stream;
    const int n = c.n_embd, nss = n / AR_SS_SPAN;
    CUDA_CHECK(cudaMemcpyAsync(d.mpos, h_pos_ + 1, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.me, h_membd_, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    // concat [rms(e) * enorm | rms(h) * hnorm] -> eh_proj (column slice) -> allreduce into a zeroed residual
    rmsnorm(d.me, n, L.enorm, d.cat, 2 * n, n, nt, c.rms_eps, s);
    rmsnorm(chain ? d.mhn : d.hn, n, L.hnorm, d.cat + n, 2 * n, n, nt, c.rms_eps, s);
    gemv_bf16(L.eh_proj, d.cat + L.eh_c0, 2 * n, d.part, n, nullptr, nt, s);
    CUDA_CHECK(cudaMemsetAsync(d.x, 0, (size_t) nt * n * sizeof(float), s));
    int call = 0;
    allreduce_add_ll16(d.x, d.part, ar_ll_, d.g, opt_.n_devices, nt * n, d.counter, call++, s, d.ss);
    record_attn(d, L, nt, d.mpos, call);
    record_ffn(d, L, nt, call);
    const float * hw = L.head_norm ? L.head_norm : d.output_norm;
    rmsnorm(d.x + (size_t) (nt - 1) * n, n, hw, d.mhn, n, n, 1, c.rms_eps, s);   // feeds chained drafts
    NormIn ni; ni.w = hw; ni.ss = d.ss; ni.nss = nss; ni.eps = c.rms_eps;
    // only the last row's draft is used: run the (Q8) head on that row alone
    NormIn nl = ni; nl.ss = d.ss + (size_t) (nt - 1) * nss;
    gemv_q8(d.output_q8, d.x + (size_t) (nt - 1) * n, n, d.logits, d.output.n, nullptr, 1, s, nl);
    argmax_pairs(d.logits, d.output.n, d.output.n, d.vocab_off, d.mres + (size_t) (nt - 1) * 2, 1, s);
    CUDA_CHECK(cudaMemcpyAsync(h_mres_ + (size_t) gi * MAX_NT * 2, d.mres, (size_t) nt * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// roll the recurrent state back to the snapshot taken after token keep-1 of the last multi-token forward
void Engine::record_restore(int gi, int keep) {
    const Qwen35Config & c = cfg_;
    Device & d = *devs_[gi];
    for (auto & L : d.layers) {
        if (L.full) continue;
        const size_t cs = (size_t) (c.ssm_conv - 1) * L.conv_ch, ssz = (size_t) L.n_v_l * c.ssm_d_state * c.head_v_dim();
        CUDA_CHECK(cudaMemcpyAsync(L.conv_state, L.conv_snap + (keep - 1) * cs, cs * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
        CUDA_CHECK(cudaMemcpyAsync(L.state, L.state_snap + (keep - 1) * ssz, ssz * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
    }
}

void Engine::build_graphs() {
    auto capture = [&](Device & d, const std::function<void()> & rec) {
        CUDA_CHECK(cudaSetDevice(d.id));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
        rec();
        CUDA_CHECK(cudaStreamEndCapture(d.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        return exec;
    };
    const int max_nt = opt_.mtp ? opt_.n_draft + 1 : 1;
    if (max_nt > MAX_NT) throw std::runtime_error("n_draft too large");
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        for (int nt = 1; nt <= max_nt; ++nt) d.g_main[nt] = capture(d, [&] { record_main(gi, nt); });
        if (opt_.mtp) {
            for (int nt = 1; nt <= max_nt; ++nt) d.g_mtp[nt] = capture(d, [&] { record_mtp(gi, nt, false); });
            d.g_chain = capture(d, [&] { record_mtp(gi, 1, true); });
            for (int keep = 1; keep < max_nt; ++keep) d.g_restore[keep] = capture(d, [&] { record_restore(gi, keep); });
        }
    }
    graphs_ready_ = true;
}

void Engine::launch(int kind, int nt) {
    if (!graphs_ready_) build_graphs();
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        cudaGraphExec_t ex = kind == 0 ? dp->g_main[nt] : kind == 1 ? dp->g_mtp[nt] : kind == 2 ? dp->g_restore[nt] : dp->g_chain;
        if (!ex) throw std::runtime_error("launch: graph not built for nt=" + std::to_string(nt));
        CUDA_CHECK(cudaGraphLaunch(ex, dp->stream));
    }
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaStreamSynchronize(dp->stream));
    }
}

void Engine::embed(const int * tokens, int nt, float * dst) {
    const Qwen35Config & c = cfg_;
    const GTensor & te = gguf_->need("token_embd.weight");
    for (int t = 0; t < nt; ++t) {
        const uint8_t * row = te.data + (size_t) tokens[t] * te.row_bytes();
        float * out = dst + (size_t) t * c.n_embd;
        for (int b = 0; b < c.n_embd / 32; ++b) {
            const uint8_t * blk = row + b * 34;
            half hd; memcpy(&hd, blk, 2);
            const float dd = __half2float(hd);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = dd * (float) ((const int8_t *) (blk + 2))[i];
        }
    }
}

static int best_of(const float * res, int ndev, int t) {
    float best = -INFINITY; int bi = -1;
    for (int g = 0; g < ndev; ++g) {
        const float v = res[(size_t) g * MAX_NT * 2 + 2 * t];
        const int idx = ((const int *) res)[(size_t) g * MAX_NT * 2 + 2 * t + 1];
        if (v > best) { best = v; bi = idx; }
    }
    return bi;
}

std::vector<int> Engine::forward(const int * tokens, int nt, int pos) {
    if (pos + nt > opt_.max_pos) throw std::runtime_error("forward: position exceeds max_pos");
    embed(tokens, nt, h_embd_);
    h_pos_[0] = pos;
    launch(0, nt);
    last_nt_ = nt;
    std::vector<int> out(nt);
    for (int t = 0; t < nt; ++t) out[t] = best_of(h_res_, (int) devs_.size(), t);
    return out;
}

int Engine::mtp_draft(const int * tokens, int nt, int pos) {
    embed(tokens, nt, h_membd_);
    h_pos_[1] = pos;
    launch(1, nt);
    return best_of(h_mres_, (int) devs_.size(), nt - 1);
}

int Engine::mtp_chain(int token, int pos) {
    embed(&token, 1, h_membd_);
    h_pos_[1] = pos;
    launch(3, 1);
    return best_of(h_mres_, (int) devs_.size(), 0);
}

void Engine::get_logits(int t, std::vector<float> & out) {
    out.resize(cfg_.n_vocab);
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaMemcpy(out.data() + dp->vocab_off, dp->logits + (size_t) t * dp->output.n,
                              (size_t) dp->output.n * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

std::vector<int> Engine::generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats) {
    if (spec && !opt_.mtp) throw std::runtime_error("generate: MTP head not loaded");
    if (prompt.empty()) throw std::runtime_error("generate: empty prompt");
    reset();
    // prompt token by token; with MTP the head also consumes (t_{q+1}, h_q) at position q
    int next = -1;
    for (int q = 0; q < (int) prompt.size(); ++q) {
        if (spec && q > 0) mtp_draft(&prompt[q], 1, q - 1);
        next = forward(&prompt[q], 1, q)[0];
    }
    std::vector<int> out;
    int p = (int) prompt.size();
    auto t0 = std::chrono::steady_clock::now();
    GenStats st;
    if (!spec) {
        while ((int) out.size() < n_gen) {
            out.push_back(next);
            next = forward(&next, 1, p++)[0];
            st.steps++;
        }
    } else {
        using clk = std::chrono::steady_clock;
        auto since = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
        const int K = opt_.n_draft;
        std::vector<int> drafts(K);
        auto make_drafts = [&](const int * mt, int nt, int pos) {   // first from the main hidden, the rest chained
            auto ta = clk::now();
            drafts[0] = mtp_draft(mt, nt, pos);
            for (int j = 1; j < K; ++j) drafts[j] = mtp_chain(drafts[j - 1], pos + nt - 1 + j);
            st.t_mtp += since(ta);
        };
        make_drafts(&next, 1, p - 1);             // (t_P, h_{P-1}) at P-1 predicts t_{P+1}
        int cur = next;                           // token at position p, not yet in the main model
        std::vector<int> in(K + 1), mt(K + 1);
        while ((int) out.size() < n_gen) {
            in[0] = cur;
            for (int j = 0; j < K; ++j) in[j + 1] = drafts[j];
            auto ta = clk::now();
            const std::vector<int> a = forward(in.data(), K + 1, p);
            st.t_main += since(ta);
            st.steps++;
            int m = 0;
            while (m < K && a[m] == drafts[m]) ++m;
            st.accepted += m;
            out.push_back(cur);
            for (int j = 0; j < m; ++j) out.push_back(drafts[j]);
            if (m < K) {                           // keep tokens 0..m of the verified block
                ta = clk::now();
                launch(2, m + 1);
                st.t_restore += since(ta);
            }
            for (int i = 0; i <= m; ++i) mt[i] = i < m ? drafts[i] : a[m];
            make_drafts(mt.data(), m + 1, p);      // positions p..p+m with main hidden rows 0..m
            cur = a[m];
            p += m + 1;
        }
        out.resize(n_gen);
    }
    st.tokens = (int) out.size();
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

} // namespace hyper
