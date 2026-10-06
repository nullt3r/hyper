#include "engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <functional>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hyper {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

struct Engine::Layer {
    bool full = false;
    float * attn_norm = nullptr;
    float * post_norm = nullptr;
    // full attention
    BF16W wq, wk, wv;
    Q8W wo;
    float * q_norm = nullptr, * k_norm = nullptr;
    half * kcache = nullptr, * vcache = nullptr;
    // gated delta net
    Q8W wqkv, wgate, wab, wout;
    float * conv_w = nullptr, * dt_bias = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    float * conv_state = nullptr, * state = nullptr;
    // ffn
    Q8W ffn_gate, ffn_up, ffn_down;
    size_t bytes = 0;
};

struct Engine::Device {
    int id = 0;
    cudaStream_t stream = nullptr;
    float * x = nullptr, * xn = nullptr;      // residual stream, normed input
    float * big0 = nullptr, * big1 = nullptr; // qkv/qg, z/gu
    float * kbuf = nullptr, * vbuf = nullptr, * ab = nullptr, * o = nullptr, * h = nullptr;
    float * logits = nullptr;
    int * d_arg = nullptr;
    size_t used = 0;
    std::vector<void *> allocs;

    template <typename T> T * alloc(size_t n) {
        void * p = nullptr;
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
        allocs.push_back(p);
        used += n * sizeof(T);
        return (T *) p;
    }
    ~Device() {
        cudaSetDevice(id);
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

namespace {

size_t q8_bytes(const GTensor & t) { return (size_t) t.nelements() + (size_t) t.nelements() / 32 * 2; }

Q8W upload_q8(Engine * /*unused*/, std::function<void *(size_t)> alloc, const std::vector<const GTensor *> & parts, int dev) {
    // concatenates the rows of all parts (same k) into one repacked matrix
    const int k = (int) parts[0]->ne[0];
    int n = 0;
    for (auto p : parts) {
        if (p->type != GType::Q8_0 || p->ne[0] != k) throw std::runtime_error("upload_q8: bad tensor " + p->name);
        n += (int) p->rows();
    }
    std::vector<int8_t> qs((size_t) n * k);
    std::vector<half> d((size_t) n * (k / 32));
    int row0 = 0;
    for (auto p : parts) {
        const int rows = (int) p->rows();
        const size_t rb = p->row_bytes();
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            const uint8_t * src = p->data + (size_t) r * rb;
            for (int b = 0; b < k / 32; ++b) {
                const uint8_t * blk = src + (size_t) b * 34;
                memcpy(&d[(size_t) (row0 + r) * (k / 32) + b], blk, 2);
                memcpy(&qs[(size_t) (row0 + r) * k + (size_t) b * 32], blk + 2, 32);
            }
        }
        row0 += rows;
    }
    CUDA_CHECK(cudaSetDevice(dev));
    Q8W w;
    w.n = n; w.k = k;
    int8_t * dq = (int8_t *) alloc(qs.size());
    half * dd = (half *) alloc(d.size() * sizeof(half));
    CUDA_CHECK(cudaMemcpy(dq, qs.data(), qs.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd, d.data(), d.size() * sizeof(half), cudaMemcpyHostToDevice));
    w.qs = dq; w.d = dd;
    return w;
}

} // namespace

Engine::Engine(const std::string & model_path, const EngineOptions & opt) : opt_(opt) {
    gguf_ = std::make_unique<GGUF>(model_path);
    cfg_ = Qwen35Config::from_gguf(*gguf_);
    fprintf(stderr, "hyper: %s\n", cfg_.describe().c_str());
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (opt_.n_devices > ndev) opt_.n_devices = ndev;
    for (int d = 0; d < opt_.n_devices; ++d) {
        auto dev = std::make_unique<Device>();
        dev->id = d;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->stream, cudaStreamNonBlocking));
        devs_.push_back(std::move(dev));
    }
    layers_.resize(cfg_.n_layer);
    balance_layers();
    load_weights();
    CUDA_CHECK(cudaMallocHost(&h_embd_, cfg_.n_embd * sizeof(float)));
}

Engine::~Engine() {
    if (h_embd_) cudaFreeHost(h_embd_);
}

void Engine::balance_layers() {
    // estimate per-layer bytes, then fill devices greedily; the LM head lives on the last device
    std::vector<size_t> bytes(cfg_.n_layer);
    size_t total = 0;
    for (int il = 0; il < cfg_.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        size_t b = 0;
        for (auto & [name, t] : gguf_->tensors()) if (name.rfind(p, 0) == 0) b += t.nbytes;
        bytes[il] = b; total += b;
    }
    const size_t head = gguf_->need("output.weight").nbytes;
    const size_t per_dev = (total + head) / devs_.size();
    layer_dev_.assign(cfg_.n_layer, 0);
    size_t acc = 0; int d = 0;
    for (int il = 0; il < cfg_.n_layer; ++il) {
        if (acc + bytes[il] / 2 > per_dev * (d + 1) && d + 1 < (int) devs_.size()) ++d;
        layer_dev_[il] = d;
        acc += bytes[il];
    }
    for (int dd = 0; dd < (int) devs_.size(); ++dd) {
        int first = -1, last = -1;
        for (int il = 0; il < cfg_.n_layer; ++il) if (layer_dev_[il] == dd) { if (first < 0) first = il; last = il; }
        fprintf(stderr, "hyper: device %d -> layers %d..%d\n", dd, first, last);
    }
}

void Engine::load_weights() {
    auto t0 = std::chrono::steady_clock::now();
    const Qwen35Config & c = cfg_;
    auto f32 = [&](Device & dev, const std::string & name) {
        const GTensor & t = gguf_->need(name);
        if (t.type != GType::F32) throw std::runtime_error("expected F32: " + name);
        float * p = dev.alloc<float>(t.nelements());
        CUDA_CHECK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
        return p;
    };
    auto bf16 = [&](Device & dev, const std::string & name) {
        const GTensor & t = gguf_->need(name);
        if (t.type != GType::BF16) throw std::runtime_error("expected BF16: " + name);
        BF16W w; w.k = (int) t.ne[0]; w.n = (int) t.rows();
        __nv_bfloat16 * p = dev.alloc<__nv_bfloat16>(t.nelements());
        CUDA_CHECK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
        w.w = p;
        return w;
    };
    auto q8 = [&](Device & dev, std::vector<std::string> names) {
        std::vector<const GTensor *> parts;
        for (auto & n : names) parts.push_back(&gguf_->need(n));
        return upload_q8(this, [&](size_t n) { return (void *) dev.alloc<uint8_t>(n); }, parts, dev.id);
    };

    const int dk = c.ssm_d_state, nv = c.ssm_dt_rank;
    for (int il = 0; il < c.n_layer; ++il) {
        Device & dev = *devs_[layer_dev_[il]];
        Layer & L = layers_[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        L.full = c.is_full_attn(il);
        L.attn_norm = f32(dev, p + "attn_norm.weight");
        L.post_norm = f32(dev, p + "post_attention_norm.weight");
        if (L.full) {
            L.wq = bf16(dev, p + "attn_q.weight");
            L.wk = bf16(dev, p + "attn_k.weight");
            L.wv = bf16(dev, p + "attn_v.weight");
            L.wo = q8(dev, {p + "attn_output.weight"});
            L.q_norm = f32(dev, p + "attn_q_norm.weight");
            L.k_norm = f32(dev, p + "attn_k_norm.weight");
            const size_t kv = (size_t) c.n_head_kv * opt_.max_pos * c.head_dim;
            L.kcache = dev.alloc<half>(kv);
            L.vcache = dev.alloc<half>(kv);
        } else {
            L.wqkv = q8(dev, {p + "attn_qkv.weight"});
            L.wgate = q8(dev, {p + "attn_gate.weight"});
            L.wab = q8(dev, {p + "ssm_alpha.weight", p + "ssm_beta.weight"});
            L.wout = q8(dev, {p + "ssm_out.weight"});
            L.conv_w = f32(dev, p + "ssm_conv1d.weight");
            L.dt_bias = f32(dev, p + "ssm_dt.bias");
            L.ssm_a = f32(dev, p + "ssm_a");
            L.ssm_norm = f32(dev, p + "ssm_norm.weight");
            L.conv_state = dev.alloc<float>((size_t) (c.ssm_conv - 1) * c.conv_dim());
            L.state = dev.alloc<float>((size_t) nv * dk * c.head_v_dim());
        }
        L.ffn_gate = q8(dev, {p + "ffn_gate.weight"});
        L.ffn_up = q8(dev, {p + "ffn_up.weight"});
        L.ffn_down = q8(dev, {p + "ffn_down.weight"});
    }
    Device & last = *devs_.back();
    output_ = bf16(last, "output.weight");
    output_norm_ = f32(last, "output_norm.weight");
    for (auto & dp : devs_) {
        Device & dev = *dp;
        dev.x = dev.alloc<float>(c.n_embd);
        dev.xn = dev.alloc<float>(c.n_embd);
        const size_t big = std::max<size_t>({(size_t) c.conv_dim(), (size_t) 2 * c.n_head * c.head_dim, (size_t) 2 * c.n_ff});
        dev.big0 = dev.alloc<float>(big);
        dev.big1 = dev.alloc<float>(big);
        dev.kbuf = dev.alloc<float>((size_t) c.n_head_kv * c.head_dim);
        dev.vbuf = dev.alloc<float>((size_t) c.n_head_kv * c.head_dim);
        dev.ab = dev.alloc<float>(2 * nv);
        dev.o = dev.alloc<float>(std::max<size_t>((size_t) c.ssm_d_inner, (size_t) c.n_head * c.head_dim));
        dev.h = dev.alloc<float>(c.n_ff);
        dev.d_arg = dev.alloc<int>(1);
    }
    last.logits = last.alloc<float>(c.n_vocab);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (auto & dp : devs_) fprintf(stderr, "hyper: device %d uses %.2f GiB\n", dp->id, dp->used / 1073741824.0);
    fprintf(stderr, "hyper: weights loaded in %.1f s\n", s);
}

void Engine::reset() {
    const Qwen35Config & c = cfg_;
    for (int il = 0; il < c.n_layer; ++il) {
        Layer & L = layers_[il];
        if (L.full) continue;
        CUDA_CHECK(cudaSetDevice(layer_dev_[il]));
        CUDA_CHECK(cudaMemset(L.conv_state, 0, (size_t) (c.ssm_conv - 1) * c.conv_dim() * sizeof(float)));
        CUDA_CHECK(cudaMemset(L.state, 0, (size_t) c.ssm_dt_rank * c.ssm_d_state * c.head_v_dim() * sizeof(float)));
    }
}

void Engine::decode(int token, int pos) {
    const Qwen35Config & c = cfg_;
    if (pos >= opt_.max_pos) throw std::runtime_error("decode: position exceeds max_pos");
    // embedding row from the mmapped Q8_0 table, dequantized on the host
    {
        const GTensor & te = gguf_->need("token_embd.weight");
        const uint8_t * row = te.data + (size_t) token * te.row_bytes();
        for (int b = 0; b < c.n_embd / 32; ++b) {
            const uint8_t * blk = row + b * 34;
            half hd; memcpy(&hd, blk, 2);
            const float d = __half2float(hd);
            for (int i = 0; i < 32; ++i) h_embd_[b * 32 + i] = d * (float) ((const int8_t *) (blk + 2))[i];
        }
    }
    Device * cur = devs_[layer_dev_[0]].get();
    CUDA_CHECK(cudaSetDevice(cur->id));
    CUDA_CHECK(cudaMemcpyAsync(cur->x, h_embd_, c.n_embd * sizeof(float), cudaMemcpyHostToDevice, cur->stream));

    const float eps = c.rms_eps;
    for (int il = 0; il < c.n_layer; ++il) {
        Device * dev = devs_[layer_dev_[il]].get();
        if (dev != cur) {
            CUDA_CHECK(cudaMemcpyPeerAsync(dev->x, dev->id, cur->x, cur->id, c.n_embd * sizeof(float), cur->stream));
            CUDA_CHECK(cudaStreamSynchronize(cur->stream));
            cur = dev;
            CUDA_CHECK(cudaSetDevice(cur->id));
        }
        cudaStream_t s = cur->stream;
        Layer & L = layers_[il];
        rmsnorm(cur->x, L.attn_norm, cur->xn, c.n_embd, eps, s);
        if (L.full) {
            gemv_bf16(L.wq, cur->xn, cur->big0, nullptr, s);
            gemv_bf16(L.wk, cur->xn, cur->kbuf, nullptr, s);
            gemv_bf16(L.wv, cur->xn, cur->vbuf, nullptr, s);
            attn_prep(cur->big0, cur->kbuf, cur->vbuf, L.q_norm, L.k_norm, L.kcache, L.vcache, pos, opt_.max_pos,
                      c.n_head, c.n_head_kv, c.head_dim, c.n_rot, c.rope_base, eps, s);
            attn_decode(cur->big0, L.kcache, L.vcache, cur->o, pos + 1, opt_.max_pos, c.n_head, c.n_head_kv, c.head_dim,
                        1.0f / sqrtf((float) c.head_dim), s);
            gemv_q8(L.wo, cur->o, cur->x, cur->x, s);
        } else {
            gemv_q8(L.wqkv, cur->xn, cur->big0, nullptr, s);
            gemv_q8(L.wgate, cur->xn, cur->big1, nullptr, s);
            gemv_q8(L.wab, cur->xn, cur->ab, nullptr, s);
            gdn_conv(cur->big0, L.conv_state, L.conv_w, c.conv_dim(), c.ssm_conv, s);
            gdn_step(cur->big0, cur->ab, L.dt_bias, L.ssm_a, L.state, cur->o, c.ssm_n_group, c.ssm_dt_rank,
                     c.ssm_d_state, c.head_v_dim(), eps, s);
            gated_norm(cur->o, cur->big1, L.ssm_norm, c.ssm_dt_rank, c.head_v_dim(), eps, s);
            gemv_q8(L.wout, cur->o, cur->x, cur->x, s);
        }
        rmsnorm(cur->x, L.post_norm, cur->xn, c.n_embd, eps, s);
        gemv_q8(L.ffn_gate, cur->xn, cur->big1, nullptr, s);
        gemv_q8(L.ffn_up, cur->xn, cur->big1 + c.n_ff, nullptr, s);
        silu_mul(cur->big1, cur->h, c.n_ff, s);
        gemv_q8(L.ffn_down, cur->h, cur->x, cur->x, s);
    }
    Device * last = devs_.back().get();
    if (cur != last) {
        CUDA_CHECK(cudaMemcpyPeerAsync(last->x, last->id, cur->x, cur->id, c.n_embd * sizeof(float), cur->stream));
        CUDA_CHECK(cudaStreamSynchronize(cur->stream));
        CUDA_CHECK(cudaSetDevice(last->id));
    }
    rmsnorm(last->x, output_norm_, last->xn, c.n_embd, eps, last->stream);
    gemv_bf16(output_, last->xn, last->logits, nullptr, last->stream);
    CUDA_CHECK(cudaGetLastError());
}

void Engine::get_logits(std::vector<float> & out) {
    Device * last = devs_.back().get();
    out.resize(cfg_.n_vocab);
    CUDA_CHECK(cudaSetDevice(last->id));
    CUDA_CHECK(cudaMemcpyAsync(out.data(), last->logits, out.size() * sizeof(float), cudaMemcpyDeviceToHost, last->stream));
    CUDA_CHECK(cudaStreamSynchronize(last->stream));
}

int Engine::argmax_last() {
    Device * last = devs_.back().get();
    CUDA_CHECK(cudaSetDevice(last->id));
    argmax(last->logits, cfg_.n_vocab, last->d_arg, last->stream);
    int r = 0;
    CUDA_CHECK(cudaMemcpyAsync(&r, last->d_arg, sizeof(int), cudaMemcpyDeviceToHost, last->stream));
    CUDA_CHECK(cudaStreamSynchronize(last->stream));
    return r;
}

} // namespace hyper
