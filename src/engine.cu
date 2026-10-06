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
#include <thread>
#include <type_traits>

namespace hyper {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

// rows per forward: up to MAX_NT through the decode GEMV + CUDA graphs, up to MAX_ROWS (prefill chunk)
// through the tensor-core GEMM, recorded directly
constexpr int MAX_ROWS = 512;
constexpr int MIN_MICRO = 64;   // prefill chunks of at least 2 * MIN_MICRO rows run as two overlapping micro-batches

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
    half * xh = nullptr;      // fp16 GEMM input scratch [MAX_ROWS][max k]
    // bulk (prefill) allreduce and micro-batching
    half * p16 = nullptr, * recv = nullptr;   // own part in fp16 [MAX_ROWS][n]; peers' parts [ndev-1][MAX_ROWS][n]
    cudaStream_t stream2 = nullptr;
    cudaEvent_t ev_ar[2][2] = {}, ev_fork = nullptr, ev_join = nullptr;
    std::vector<cudaEvent_t> ev_layer;
    int * pos2 = nullptr;
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
        for (auto & r : ev_ar) for (auto & ev : r) if (ev) cudaEventDestroy(ev);
        for (auto ev : ev_layer) cudaEventDestroy(ev);
        if (ev_fork) cudaEventDestroy(ev_fork);
        if (ev_join) cudaEventDestroy(ev_join);
        if (stream2) cudaStreamDestroy(stream2);
        if (stream) cudaStreamDestroy(stream);
    }
};

namespace {

struct RowRange { const GTensor * t; int64_t r0, r1; };

// row-major bf16 rows (possibly several tensors / ranges, optional column window) -> fp16 fragment-ordered weight
BF16W to_device_bf16(const std::function<void *(size_t)> & alloc, int dev, const std::vector<std::pair<const uint16_t *, int64_t>> & rows_src,
                     int k, size_t row_stride) {
    // gather rows into one contiguous row-major buffer, then repack
    const int n = (int) rows_src.size();
    std::vector<uint16_t> buf((size_t) n * k);
    for (int r = 0; r < n; ++r) memcpy(&buf[(size_t) r * k], rows_src[r].first, (size_t) k * 2);
    (void) row_stride;
    const size_t ntile = (n + 15) / 16;
    std::vector<uint8_t> fq(ntile * (k / 16) * 512);
    repack_bf16_frag(buf.data(), n, k, (size_t) k, fq.data());
    CUDA_CHECK(cudaSetDevice(dev));
    BF16W w; w.n = n; w.k = k;
    void * p = alloc(fq.size());
    CUDA_CHECK(cudaMemcpy(p, fq.data(), fq.size(), cudaMemcpyHostToDevice));
    w.q = (const uint4 *) p;
    return w;
}

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
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->stream2, cudaStreamNonBlocking));
        for (auto & r : dev->ev_ar) for (auto & ev : r) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev->ev_fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev->ev_join, cudaEventDisableTiming));
        devs_.push_back(std::move(dev));
    }
    const int nd = opt_.n_devices, n = cfg_.n_embd;
    CUDA_CHECK(cudaHostAlloc(&h_embd_, (size_t) MAX_ROWS * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_membd_, (size_t) MAX_ROWS * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_pos_, 4 * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_stage_, (size_t) 2 * 2 * nd * MAX_ROWS * n * sizeof(half), cudaHostAllocPortable));
    barrier_ = std::make_unique<Barrier>(nd);
    CUDA_CHECK(cudaHostAlloc(&h_res_, (size_t) nd * MAX_ROWS * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_mres_, (size_t) nd * MAX_ROWS * 2 * sizeof(float), cudaHostAllocPortable));
    const size_t ll = (size_t) 2 * nd * MAX_NT * n / 2;
    CUDA_CHECK(cudaHostAlloc(&ar_ll_, ll * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_ll_, 0xff, ll * sizeof(uint2));

    load_weights();
}

Engine::~Engine() {
    devs_.clear();
    for (void * p : {(void *) h_embd_, (void *) h_membd_, (void *) h_pos_, (void *) h_res_, (void *) h_mres_, (void *) ar_ll_, (void *) h_stage_})
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
        const int k = (int) parts[0].t->ne[0];
        std::vector<std::pair<const uint16_t *, int64_t>> rows;
        for (auto & pr : parts) {
            if (pr.t->type != GType::BF16 || pr.t->ne[0] != k) throw std::runtime_error("bf16_rows: bad tensor " + pr.t->name);
            for (int64_t r = pr.r0; r < pr.r1; ++r) rows.push_back({(const uint16_t *) (pr.t->data + (size_t) r * pr.t->row_bytes()), r});
        }
        return to_device_bf16(A, dev.id, rows, k, (size_t) k);
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
        std::vector<std::pair<const uint16_t *, int64_t>> rows;
        for (int64_t r = 0; r < eh.ne[1]; ++r) rows.push_back({(const uint16_t *) (eh.data + (size_t) r * eh.row_bytes()) + c0, r});
        L.eh_proj = to_device_bf16(A, dev.id, rows, (int) (c1 - c0), (size_t) eh.ne[0]);
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
            std::vector<std::pair<const uint16_t *, int64_t>> rows;
            for (int64_t r = o0; r < o1; ++r) rows.push_back({(const uint16_t *) (t.data + (size_t) r * t.row_bytes()), r});
            dev.output = to_device_bf16([&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); }, dev.id, rows, (int) t.ne[0], (size_t) t.ne[0]);
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
        dev.x = dev.alloc<float>((size_t) MAX_ROWS * n);
        dev.part = dev.alloc<float>((size_t) MAX_ROWS * n);
        dev.big0 = dev.alloc<float>((size_t) MAX_ROWS * dev.big_stride);
        dev.big1 = dev.alloc<float>((size_t) MAX_ROWS * dev.big_stride);
        dev.o = dev.alloc<float>((size_t) MAX_ROWS * std::max(c.ssm_d_inner, c.n_head * c.head_dim));
        dev.h = dev.alloc<float>((size_t) MAX_ROWS * c.n_ff);
        dev.hn = dev.alloc<float>((size_t) MAX_ROWS * n);
        dev.mhn = dev.alloc<float>(n);
        dev.me = dev.alloc<float>((size_t) MAX_ROWS * n);
        dev.cat = dev.alloc<float>((size_t) MAX_ROWS * 2 * n);
        dev.logits = dev.alloc<float>((size_t) MAX_ROWS * dev.output.n);
        dev.res = dev.alloc<float>(MAX_ROWS * 2);
        dev.mres = dev.alloc<float>(MAX_ROWS * 2);
        dev.ss = dev.alloc<float>(MAX_ROWS * 64);
        dev.xh = dev.alloc<half>((size_t) MAX_ROWS * std::max(c.n_ff, 2 * n));
        dev.p16 = dev.alloc<half>((size_t) MAX_ROWS * n);
        dev.recv = dev.alloc<half>((size_t) (nd - 1) * MAX_ROWS * n + 1);
        dev.pos2 = dev.alloc<int>(1);
        dev.ev_layer.resize(c.n_layer);
        for (auto & ev : dev.ev_layer) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
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

// ---------------- matmul dispatch ----------------
// decode sizes: split-K GEMV with the norm fused from allreduce statistics; bulk (prefill): fp16 conversion
// (+ norm computed in place) and the tiled GEMM
template <typename W>
static void mm(const Engine::Act & a, const W & w, const float * x, int xs, float * y, int ys, int nt, const NormIn & ni = {}) {
    if (!a.bulk) {
        if constexpr (std::is_same_v<W, Q8W>) gemv_q8(w, x, xs, y, ys, nullptr, nt, a.s, ni);
        else gemv_bf16(w, x, xs, y, ys, nullptr, nt, a.s, ni);
        return;
    }
    to_half(x, xs, ni.w, w.k, ni.eps, a.xh, nt, a.s);
    if constexpr (std::is_same_v<W, Q8W>) gemm_q8(w, a.xh, nt, y, ys, nullptr, a.s);
    else gemm_f16(w, a.xh, nt, y, ys, nullptr, a.s);
}

// ---------------- graph recording ----------------

// activation view: rows [row0, row0 + nt) of the device buffers on one stream
Engine::Act Engine::act(Device & d, int row0, int sid, cudaStream_t s, const int * pos, bool bulk) {
    const Qwen35Config & c = cfg_;
    const size_t n = c.n_embd;
    Act a;
    a.x = d.x + row0 * n; a.part = d.part + row0 * n;
    a.big0 = d.big0 + (size_t) row0 * d.big_stride; a.big1 = d.big1 + (size_t) row0 * d.big_stride;
    a.o = d.o + (size_t) row0 * std::max(c.ssm_d_inner, c.n_head * c.head_dim);
    a.h = d.h + (size_t) row0 * c.n_ff;
    a.ss = d.ss + (size_t) row0 * (n / AR_SS_SPAN);
    a.xh = d.xh + (size_t) row0 * std::max(c.n_ff, 2 * c.n_embd);
    a.p16 = d.p16 + row0 * n;
    a.recv = d.recv + row0 * n;
    a.pos = pos; a.s = s; a.sid = sid; a.bulk = bulk;
    return a;
}

// a.x += sum over devices of a.part (nt rows). Decode: LL kernel, also refreshing a.ss for the next fused norm.
// Bulk: fp16 parts through pinned host memory with the copy engines; peers' copies are ordered by events, which
// is why every device's host thread meets at a barrier between recording its upload and the peers' downloads.
void Engine::allreduce(Device & d, Act & a, int nt, int & call) {
    const int n = nt * cfg_.n_embd, nd = opt_.n_devices;
    if (!a.bulk) { allreduce_add_ll16(a.x, a.part, ar_ll_, d.g, nd, n, d.counter, call++, a.s, a.ss); return; }
    const int par = call++ & 1;
    auto stage = [&](int sid, int p, int g) { return h_stage_ + (((size_t) sid * 2 + p) * nd + g) * MAX_ROWS * cfg_.n_embd; };
    to_half(a.part, cfg_.n_embd, nullptr, cfg_.n_embd, 0.0f, a.p16, nt, a.s);
    CUDA_CHECK(cudaMemcpyAsync(stage(a.sid, par, d.g), a.p16, (size_t) n * sizeof(half), cudaMemcpyDeviceToHost, a.s));
    CUDA_CHECK(cudaEventRecord(d.ev_ar[a.sid][par], a.s));
    barrier_->wait();
    int j = 0;
    for (int p = 0; p < nd; ++p) {
        if (p == d.g) continue;
        CUDA_CHECK(cudaStreamWaitEvent(a.s, devs_[p]->ev_ar[a.sid][par], 0));
        CUDA_CHECK(cudaMemcpyAsync(a.recv + (size_t) j * MAX_ROWS * cfg_.n_embd, stage(a.sid, par, p), (size_t) n * sizeof(half),
                                   cudaMemcpyHostToDevice, a.s));
        ++j;
    }
    add_parts(a.x, a.p16, a.recv, (size_t) MAX_ROWS * cfg_.n_embd, nd - 1, n, a.s);
}

// prefill micro-batch ordering: dep_wait (stream B) waits for what stream A recorded at the same layer
void Engine::record_attn(Device & d, DevLayer & L, Act & a, int nt, int & call, const Dep & dep) {
    const Qwen35Config & c = cfg_;
    const float eps = c.rms_eps;
    const int n = c.n_embd, nss = n / AR_SS_SPAN, bs = d.big_stride;
    NormIn ni; ni.w = L.attn_norm; ni.ss = a.ss; ni.nss = nss; ni.eps = eps;
    mm(a, L.wqkv, a.x, n, a.big0, bs, nt, ni);
    attn_prep(a.big0, bs, L.q_norm, L.k_norm, L.kcache, L.vcache, a.pos, opt_.max_pos, L.n_head_l, L.n_kv_l, c.head_dim,
              c.n_rot, c.rope_base, eps, nt, a.s);
    if (dep.signal) CUDA_CHECK(cudaEventRecord(dep.signal, a.s));   // this micro-batch's K/V are in the cache
    if (dep.wait) CUDA_CHECK(cudaStreamWaitEvent(a.s, dep.wait, 0));
    const int ostride = L.n_head_l * c.head_dim;
    attn_decode(a.big0, bs, L.kcache, L.vcache, a.o, ostride, a.pos, opt_.max_pos, L.n_head_l, L.n_kv_l, L.head_off,
                c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, a.s);
    mm(a, L.wo, a.o, ostride, a.part, n, nt);
    allreduce(d, a, nt, call);
}

void Engine::record_gdn(Device & d, DevLayer & L, Act & a, int nt, bool snap, int & call, const Dep & dep) {
    const Qwen35Config & c = cfg_;
    const float eps = c.rms_eps;
    const int n = c.n_embd, nss = n / AR_SS_SPAN, bs = d.big_stride, dv = c.head_v_dim();
    NormIn ni; ni.w = L.attn_norm; ni.ss = a.ss; ni.nss = nss; ni.eps = eps;
    mm(a, L.win, a.x, n, a.big0, bs, nt, ni);
    const int z_off = L.conv_ch, ab_off = L.conv_ch + L.n_v_l * dv;
    if (dep.wait) CUDA_CHECK(cudaStreamWaitEvent(a.s, dep.wait, 0));   // recurrent state after the previous micro-batch
    gdn_conv(a.big0, bs, L.conv_state, snap ? L.conv_snap : nullptr, L.conv_w, L.conv_ch, c.ssm_conv, nt, a.s);
    const int ostride = L.n_v_l * dv;
    gdn_step(a.big0, bs, ab_off, L.state, snap ? L.state_snap : nullptr, a.o, ostride, L.dt_bias, L.ssm_a, L.n_k_l, L.n_v_l,
             c.ssm_d_state, dv, eps, nt, a.s);
    if (dep.signal) CUDA_CHECK(cudaEventRecord(dep.signal, a.s));
    gated_norm(a.o, ostride, a.big0 + z_off, bs, L.ssm_norm, L.n_v_l, dv, eps, nt, a.s);
    mm(a, L.wout, a.o, ostride, a.part, n, nt);
    allreduce(d, a, nt, call);
}

void Engine::record_ffn(Device & d, DevLayer & L, Act & a, int nt, int & call) {
    const Qwen35Config & c = cfg_;
    const int n = c.n_embd, nss = n / AR_SS_SPAN, bs = d.big_stride;
    NormIn ni; ni.w = L.post_norm; ni.ss = a.ss; ni.nss = nss; ni.eps = c.rms_eps;
    mm(a, L.ffn_gu, a.x, n, a.big1, bs, nt, ni);
    silu_mul(a.big1, bs, a.h, c.n_ff, L.n_ff_l, nt, a.s);
    mm(a, L.ffn_down, a.h, c.n_ff, a.part, n, nt);
    allreduce(d, a, nt, call);
}

// transformer stack over act a; with two micro-batches (b != nullptr) both are recorded layer by layer so that
// one stream's allreduce copies overlap the other's compute
void Engine::record_layers(Device & d, Act & a, int nta, Act * b, int ntb, bool snap) {
    const Qwen35Config & c = cfg_;
    int call_a = 0, call_b = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        DevLayer & L = d.layers[il];
        Dep da, db;
        if (b) { da.signal = d.ev_layer[il]; db.wait = d.ev_layer[il]; }
        if (L.full) record_attn(d, L, a, nta, call_a, da); else record_gdn(d, L, a, nta, snap, call_a, da);
        if (b) { if (L.full) record_attn(d, L, *b, ntb, call_b, db); else record_gdn(d, L, *b, ntb, snap, call_b, db); }
        record_ffn(d, L, a, nta, call_a);
        if (b) record_ffn(d, L, *b, ntb, call_b);
    }
}

void Engine::record_main(int gi, int nt) {
    const Qwen35Config & c = cfg_;
    Device & d = *devs_[gi];
    cudaStream_t s = d.stream;
    const int n = c.n_embd, nss = n / AR_SS_SPAN;
    const bool bulk = nt > MAX_NT;
    CUDA_CHECK(cudaMemcpyAsync(d.pos, h_pos_, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.x, h_embd_, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    sumsq(d.x, n, n, nt, d.ss, nss, s);
    Act a = act(d, 0, 0, s, d.pos, bulk);
    if (bulk && nt >= 2 * MIN_MICRO) {
        const int nta = (nt / 2 + 63) / 64 * 64, ntb = nt - nta;
        CUDA_CHECK(cudaMemcpyAsync(d.pos2, h_pos_ + 2, sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaEventRecord(d.ev_fork, s));
        CUDA_CHECK(cudaStreamWaitEvent(d.stream2, d.ev_fork, 0));
        Act b = act(d, nta, 1, d.stream2, d.pos2, true);
        record_layers(d, a, nta, &b, ntb, false);
        CUDA_CHECK(cudaEventRecord(d.ev_join, d.stream2));
        CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_join, 0));
    } else {
        record_layers(d, a, nt, nullptr, 0, opt_.mtp && nt > 1 && !bulk);
    }
    if (opt_.mtp) rmsnorm(d.x, n, d.output_norm, d.hn, n, n, nt, c.rms_eps, s);   // hidden fed to the MTP head
    NormIn ni; ni.w = d.output_norm; ni.ss = d.ss; ni.nss = nss; ni.eps = c.rms_eps;
    mm(a, d.output, d.x, n, d.logits, d.output.n, nt, ni);
    argmax_pairs(d.logits, d.output.n, d.output.n, d.vocab_off, d.res, nt, s);
    CUDA_CHECK(cudaMemcpyAsync(h_res_ + (size_t) gi * MAX_ROWS * 2, d.res, (size_t) nt * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// chain = false: hidden rows come from the main model (hn); chain = true (nt = 1): from MTP's own last output
void Engine::record_mtp(int gi, int nt, bool chain) {
    const Qwen35Config & c = cfg_;
    Device & d = *devs_[gi];
    DevLayer & L = d.mtp;
    cudaStream_t s = d.stream;
    const int n = c.n_embd, nss = n / AR_SS_SPAN;
    Act a = act(d, 0, 0, s, d.mpos, nt > MAX_NT);
    CUDA_CHECK(cudaMemcpyAsync(d.mpos, h_pos_ + 1, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.me, h_membd_, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    // concat [rms(e) * enorm | rms(h) * hnorm] -> eh_proj (column slice) -> allreduce into a zeroed residual
    rmsnorm(d.me, n, L.enorm, d.cat, 2 * n, n, nt, c.rms_eps, s);
    rmsnorm(chain ? d.mhn : d.hn, n, L.hnorm, d.cat + n, 2 * n, n, nt, c.rms_eps, s);
    mm(a, L.eh_proj, d.cat + L.eh_c0, 2 * n, d.part, n, nt);
    CUDA_CHECK(cudaMemsetAsync(d.x, 0, (size_t) nt * n * sizeof(float), s));
    int call = 0;
    allreduce(d, a, nt, call);
    record_attn(d, L, a, nt, call, {});
    record_ffn(d, L, a, nt, call);
    const float * hw = L.head_norm ? L.head_norm : d.output_norm;
    rmsnorm(d.x + (size_t) (nt - 1) * n, n, hw, d.mhn, n, n, 1, c.rms_eps, s);   // feeds chained drafts
    // only the last row's draft is used: run the (Q8) head on that row alone, normed by its own statistics
    sumsq(d.x + (size_t) (nt - 1) * n, n, n, 1, d.ss, nss, s);
    NormIn nl; nl.w = hw; nl.ss = d.ss; nl.nss = nss; nl.eps = c.rms_eps;
    Act a1 = act(d, 0, 0, s, d.mpos, false);
    mm(a1, d.output_q8, d.x + (size_t) (nt - 1) * n, n, d.logits, d.output.n, 1, nl);
    argmax_pairs(d.logits, d.output.n, d.output.n, d.vocab_off, d.mres, 1, s);
    CUDA_CHECK(cudaMemcpyAsync(h_mres_ + (size_t) gi * 2, d.mres, 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
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
    if (nt > MAX_NT) {   // prefill chunk: record straight into the streams (allreduce kernels spin until all GPUs arrive)
        if (kind != 0 && kind != 1) throw std::runtime_error("launch: bad kind for a prefill chunk");
        std::vector<std::thread> th;
        std::vector<std::string> err(devs_.size());
        for (int gi = 0; gi < (int) devs_.size(); ++gi)
            th.emplace_back([&, gi] {
                try {
                    CUDA_CHECK(cudaSetDevice(devs_[gi]->id));
                    if (kind == 0) record_main(gi, nt); else record_mtp(gi, nt, false);
                } catch (const std::exception & ex) { err[gi] = ex.what(); }
            });
        for (auto & t : th) t.join();
        for (auto & m : err) if (!m.empty()) throw std::runtime_error(m);
        for (auto & dp : devs_) {
            CUDA_CHECK(cudaSetDevice(dp->id));
            CUDA_CHECK(cudaStreamSynchronize(dp->stream));
        }
        return;
    }
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

static int best_of(const float * res, int ndev, int t, int stride) {
    float best = -INFINITY; int bi = -1;
    for (int g = 0; g < ndev; ++g) {
        const float v = res[(size_t) g * stride * 2 + 2 * t];
        const int idx = ((const int *) res)[(size_t) g * stride * 2 + 2 * t + 1];
        if (v > best) { best = v; bi = idx; }
    }
    return bi;
}

std::vector<int> Engine::forward(const int * tokens, int nt, int pos) {
    if (pos + nt > opt_.max_pos) throw std::runtime_error("forward: position exceeds max_pos");
    if (nt < 1 || nt > MAX_ROWS) throw std::runtime_error("forward: bad token count");
    embed(tokens, nt, h_embd_);
    h_pos_[0] = pos;
    h_pos_[2] = pos + (nt / 2 + 63) / 64 * 64;   // second prefill micro-batch
    launch(0, nt);
    last_nt_ = nt;
    std::vector<int> out(nt);
    for (int t = 0; t < nt; ++t) out[t] = best_of(h_res_, (int) devs_.size(), t, MAX_ROWS);
    return out;
}

int Engine::mtp_draft(const int * tokens, int nt, int pos) {
    if (nt < 1 || nt > MAX_ROWS) throw std::runtime_error("mtp_draft: bad token count");
    embed(tokens, nt, h_membd_);
    h_pos_[1] = pos;
    launch(1, nt);
    return best_of(h_mres_, (int) devs_.size(), 0, 1);
}

int Engine::mtp_chain(int token, int pos) {
    embed(&token, 1, h_membd_);
    h_pos_[1] = pos;
    launch(3, 1);
    return best_of(h_mres_, (int) devs_.size(), 0, 1);
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
    // prompt in chunks of up to MAX_ROWS; with MTP the head then consumes (t_{q+1}, h_q) at position q for the
    // chunk's rows (the last chunk's final pair uses the predicted next token and yields the first draft)
    using clk = std::chrono::steady_clock;
    auto since = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    GenStats st;
    const int P = (int) prompt.size();
    const int K = opt_.n_draft;
    std::vector<int> drafts(K);
    auto tp = clk::now();
    int next = -1;
    for (int c0 = 0; c0 < P; c0 += MAX_ROWS) {
        const int len = std::min(MAX_ROWS, P - c0);
        next = forward(&prompt[c0], len, c0)[len - 1];
        if (spec) {
            std::vector<int> mt(len);
            for (int j = 0; j < len; ++j) mt[j] = c0 + 1 + j < P ? prompt[c0 + 1 + j] : next;
            const int d0 = mtp_draft(mt.data(), len, c0);
            if (c0 + len == P) drafts[0] = d0;
        }
    }
    st.t_prefill = since(tp);
    std::vector<int> out;
    int p = P;
    auto t0 = clk::now();
    if (!spec) {
        while ((int) out.size() < n_gen) {
            out.push_back(next);
            next = forward(&next, 1, p++)[0];
            st.steps++;
        }
    } else {
        auto make_drafts = [&](const int * mt, int nt, int pos) {   // first from the main hidden, the rest chained
            auto ta = clk::now();
            drafts[0] = mtp_draft(mt, nt, pos);
            for (int j = 1; j < K; ++j) drafts[j] = mtp_chain(drafts[j - 1], pos + nt - 1 + j);
            st.t_mtp += since(ta);
        };
        for (int j = 1; j < K; ++j) drafts[j] = mtp_chain(drafts[j - 1], p - 1 + j);   // first draft came with the prompt
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
