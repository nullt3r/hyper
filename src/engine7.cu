#include "engine7.h"
#include "sampling.h"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace hyper {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace {

constexpr int CPU_OWNER = 3;
constexpr int SUB = 64;   // indexer: queries scored at a time

// dense weight: Q8_0 rows (fragment order) or, for other sources, re-quantized to Q8_0 at load
struct DW {
    Q8W q8;
    int n() const { return q8.n; }
    int k() const { return q8.k; }
};
struct RowRange { const GTensor * t; int64_t r0, r1; };
using Alloc = std::function<void *(size_t)>;

Q8W to_device_q8(const Alloc & alloc, int dev, const int8_t * qs, const half * d, int n, int k) {
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

std::vector<float> to_f32(const GTensor & t, int64_t r0 = 0, int64_t r1 = -1) {
    if (r1 < 0) r1 = t.rows();
    const int64_t k = t.ne[0];
    std::vector<float> out((size_t) (r1 - r0) * k);
    if (t.type == GType::F32) { memcpy(out.data(), t.data + (size_t) r0 * t.row_bytes(), out.size() * sizeof(float)); return out; }
    const auto * tr = ggml_get_type_traits((ggml_type) t.type);
    if (!tr || !tr->to_float) throw std::runtime_error("to_f32: no dequantizer for " + t.name);
#pragma omp parallel for schedule(static)
    for (int64_t r = r0; r < r1; ++r) tr->to_float(t.data + (size_t) r * t.row_bytes(), out.data() + (size_t) (r - r0) * k, k);
    return out;
}

// rows of several Q8_0 tensors (same k) stacked; other types are re-quantized to Q8_0
DW upload_dense(const Alloc & alloc, int dev, const std::vector<RowRange> & parts) {
    const int64_t k = parts[0].t->ne[0], kb = k / 32;
    int64_t n = 0;
    for (auto & p : parts) {
        if (p.t->ne[0] != k) throw std::runtime_error("upload_dense: k mismatch " + p.t->name);
        n += p.r1 - p.r0;
    }
    std::vector<int8_t> qs((size_t) n * k);
    std::vector<half> d((size_t) n * kb);
    int64_t row0 = 0;
    for (auto & p : parts) {
        if (p.t->type == GType::Q8_0) {
            const size_t rb = p.t->row_bytes();
#pragma omp parallel for schedule(static)
            for (int64_t r = p.r0; r < p.r1; ++r) {
                const int64_t orow = row0 + (r - p.r0);
                const uint8_t * src = p.t->data + (size_t) r * rb;
                for (int64_t j = 0; j < kb; ++j) {
                    memcpy(&d[(size_t) orow * kb + j], src + j * 34, 2);
                    memcpy(&qs[(size_t) orow * k + j * 32], src + j * 34 + 2, 32);
                }
            }
        } else {
            const std::vector<float> rows = to_f32(*p.t, p.r0, p.r1);
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < (p.r1 - p.r0) * kb; ++i) {
                const float * x = rows.data() + (size_t) i * 32;
                float amax = 0.0f;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
                const float dd = amax / 127.0f, id = dd > 0.0f ? 1.0f / dd : 0.0f;
                d[(size_t) row0 * kb + i] = __float2half(dd);
                for (int j = 0; j < 32; ++j) qs[(size_t) row0 * k + (size_t) i * 32 + j] = (int8_t) roundf(x[j] * id);
            }
        }
        row0 += p.r1 - p.r0;
    }
    DW w;
    w.q8 = to_device_q8(alloc, dev, qs.data(), d.data(), (int) n, (int) k);
    return w;
}

inline uint16_t f2bf(float f) {   // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
inline float bf_round(float f) {
    uint32_t u = (uint32_t) f2bf(f) << 16;
    float r;
    memcpy(&r, &u, 4);
    return r;
}

} // namespace

struct Engine7::Layer {
    int il = 0, ratio = 0, eg = -1;
    bool kv_src = false, ix_src = false, cand_src = false, uses_cand = false;
    int w_out = 0, off_kv = 0, off_ckv = 0, off_cg = 0;   // fused input projection: rows [q_a | kv | c_kv | c_gate]
    DW win, wq_b, iq_b, wo_b, sh_gu, sh_down, eg_kv;
    std::vector<DW> wo_a;
    float * q_norm = nullptr, * kv_norm = nullptr, * c_norm = nullptr, * ik_norm = nullptr;
    float * attn_norm = nullptr, * ffn_norm = nullptr, * sink = nullptr;
    float * hca_fn = nullptr, * hca_scale = nullptr, * hca_base = nullptr, * hcf_fn = nullptr, * hcf_scale = nullptr, * hcf_base = nullptr;
    float * router = nullptr, * bias = nullptr, * eg_w = nullptr, * ik_w = nullptr, * iproj = nullptr;
    uint16_t * ring = nullptr;                        // [ring][head_dim] bf16
    float * c_kv = nullptr, * c_score = nullptr;      // the compressor's open group [ratio][head_dim]
    uint16_t * ckv = nullptr, * ik = nullptr;         // KV sources: compressed rows [max_pos / ratio][head_dim], index keys [..][ix_hd]
    // experts on this GPU
    MoeDev moex;
    int * owner = nullptr;
    int n_gpu = 0;
    // prefill: the CPU's experts streamed to the GPU in parts (from the pinned compact host copy, index i): part j holds compact
    // indices [part_i0[j], part_i0[j + 1]), part_slot[j] maps an expert to its index in the part (-1: not in it)
    const uint8_t * hg = nullptr, * hu = nullptr, * hd = nullptr;
    size_t gb = 0, db = 0;
    std::vector<int> part_i0;
    std::vector<int *> part_slot;
    int np() const { return part_i0.empty() ? 0 : (int) part_i0.size() - 1; }
};

struct Engine7::Device {
    int id = 0, g = 0;
    int l0 = 0, l1 = 0;   // layers [l0, l1)
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cudaEvent_t ev_done = nullptr;
    cudaStream_t cstream = nullptr;                     // prefill: expert uploads
    cudaEvent_t ev_up[2] = {}, ev_free[2] = {};
    uint8_t * stage[2] = {};
    size_t stage_bytes = 0;
    int * owner_all = nullptr;                          // [E] = this device (prefill: every expert computed here)
    std::vector<Layer> layers;
    std::vector<void *> allocs;
    size_t used = 0;
    // activations (R rows)
    float * h = nullptr, * h2 = nullptr, * x = nullptr, * mix = nullptr, * pre = nullptr, * a_pre = nullptr, * a_post = nullptr,
          * a_comb = nullptr, * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    float * wout = nullptr, * q = nullptr, * o = nullptr, * oa = nullptr, * ao = nullptr, * latent = nullptr, * ikr = nullptr, * iq = nullptr,
          * iw = nullptr, * scores = nullptr, * bm = nullptr, * rlog = nullptr, * wts = nullptr, * ones = nullptr, * shgu = nullptr,
          * shh = nullptr, * shout = nullptr, * hexp = nullptr, * yexp = nullptr, * ffo = nullptr, * eg_x = nullptr, * eg_kvo = nullptr,
          * logits = nullptr, * outp = nullptr;
    int * ids = nullptr, * sel = nullptr, * cand = nullptr, * ncand = nullptr, * order = nullptr, * order_n = nullptr, * egrp = nullptr,
        * counter = nullptr;
    half * xh = nullptr, * x16 = nullptr, * h16 = nullptr;
    long long score_w = 0, bm_w = 0;
    float * freqs_win = nullptr, * freqs_comp = nullptr;
    // the head (last device)
    float * out_norm = nullptr;
    DW output;

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
        if (blas) cublasDestroy(blas);
        if (ev_done) cudaEventDestroy(ev_done);
        for (auto & ev : ev_up) if (ev) cudaEventDestroy(ev);
        for (auto & ev : ev_free) if (ev) cudaEventDestroy(ev);
        if (cstream) cudaStreamDestroy(cstream);
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
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
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    const int nd = std::min(opt_.n_devices, ndev);
    // layers over the devices along the KV-source groups (HYPER7_SPLIT="8,20": the first layers of devices 1, 2, ..)
    std::vector<int> starts = {0};
    if (const char * sp = getenv("HYPER7_SPLIT")) {
        for (const char * q = sp; *q;) { starts.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    } else if (nd == 3) starts = {0, 8, 20};
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
        d->id = g; d->g = g;
        d->l0 = starts[g];
        d->l1 = g + 1 < (int) starts.size() ? starts[g + 1] : c.n_layer;
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreateWithFlags(&d->stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&d->ev_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(&d->cstream, cudaStreamNonBlocking));
        for (auto & ev : d->ev_up) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        for (auto & ev : d->ev_free) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        if (cublasCreate(&d->blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        cublasSetStream(d->blas, d->stream);
        cublasSetMathMode(d->blas, CUBLAS_PEDANTIC_MATH);
        gemv_init(g);
        devs_.push_back(std::move(d));
    }
    CUDA_CHECK(cudaHostAlloc(&h_embd_, (size_t) R_ * n * sizeof(float), cudaHostAllocPortable));
    for (size_t i = 0; i < c.eg_layers.size(); ++i) {
        float * p = nullptr;
        CUDA_CHECK(cudaHostAlloc(&p, (size_t) R_ * c.eg_cols() * c.eg_hd * sizeof(float), cudaHostAllocPortable));
        h_eg_.push_back(p);
    }
    CUDA_CHECK(cudaHostAlloc(&h_out_, (size_t) MAX_NT * TOPK * 2 * sizeof(float) + 64, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&cpu_rec_, (size_t) c.n_layer * sizeof(CpuMoeRec), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_out_, (size_t) c.n_layer * sizeof(CpuMoeOut), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_, sizeof(CpuMoeBulk), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_out_, sizeof(CpuMoeBulkOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_rec_, 0, (size_t) c.n_layer * sizeof(CpuMoeRec));
    memset((void *) cpu_out_, 0, (size_t) c.n_layer * sizeof(CpuMoeOut));
    memset((void *) cpu_bulk_, 0, sizeof(CpuMoeBulk));
    memset((void *) cpu_bulk_out_, 0, sizeof(CpuMoeBulkOut));
    if (const char * sp = getenv("HYPER4_STATS")) {
        if (FILE * f = fopen(sp, "rb")) {
            int nl = 0, w = 0;
            if (fread(&nl, 4, 1, f) == 1 && fread(&w, 4, 1, f) == 1 && nl == c.n_layer) {
                stats_.assign(nl, std::vector<uint64_t>(w));
                for (auto & s : stats_) if (fread(s.data(), 8, w, f) != (size_t) w) { stats_.clear(); break; }
            }
            fclose(f);
            fprintf(stderr, "hyper7: expert placement from %s (%s)\n", sp, stats_.empty() ? "unreadable, ignored" : "ok");
        }
    }
    cpu_ = std::make_unique<CpuMoe>(opt_.cpu_threads, n, c.ff, c.n_used, cpu_rec_, cpu_out_, c.n_layer, cpu_bulk_, cpu_bulk_out_);
    cpu_->set_clamp(c.clamp_exp);
    const auto t0 = std::chrono::steady_clock::now();
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
            d.logits = d.alloc<float>((size_t) MAX_NT * c.n_vocab);
            d.outp = d.alloc<float>((size_t) MAX_NT * TOPK * 2 + 64);
        }
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        fprintf(stderr, "hyper7: device %d: layers %d..%d, dense weights + caches + buffers %.2f GiB\n", d.id, d.l0, d.l1 - 1,
                d.used / 1073741824.0);
    }
    place_experts();
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
        if (t->ne[1] != (int64_t) c.o_groups * c.o_lora || t->ne[0] != (int64_t) c.n_head * c.head_dim / c.o_groups)
            throw std::runtime_error("hyper7: attn_output_a shape");
        for (int gi = 0; gi < c.o_groups; ++gi) L.wo_a.push_back(upload_dense(A, d.id, {{t, (int64_t) gi * c.o_lora, (int64_t) (gi + 1) * c.o_lora}}));
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
    if (L.eg >= 0) {
        const GTensor * t = T("engram_kv.weight");
        L.eg_kv = upload_dense(A, d.id, {{t, 0, t->rows()}});
        L.eg_w = f32("engram_gate.weight");
    }
    L.owner = d.alloc<int>(c.n_expert);
}

void Engine7::place_experts() {
    const Ds4Config & c = cfg_;
    const int E = c.n_expert, ff = c.ff, n = c.n_embd;
    for (auto & dp : devs_) {
        Device & d = *dp;
        CUDA_CHECK(cudaSetDevice(d.id));
        // prefill staging: two buffers the CPU's experts stream through (HYPER7_STAGE_MIB each)
        d.stage_bytes = (size_t) (getenv("HYPER7_STAGE_MIB") ? atoi(getenv("HYPER7_STAGE_MIB")) : 1024) << 20;
        for (auto & b : d.stage) b = d.alloc<uint8_t>(d.stage_bytes);
        {   std::vector<int> all(E, d.g); d.owner_all = d.upload(all.data(), E); }
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
        const int n_gpu = per_all ? (int) std::min<size_t>(E, budget / per_all) : 0;
        size_t dev_bytes = 0, host_bytes = 0;
        for (Layer & L : d.layers) {
            const int il = L.il;
            const std::string p = "blk." + std::to_string(il) + ".";
            const GTensor & tg = gguf_->need(p + "ffn_gate_exps.weight"), & tu = gguf_->need(p + "ffn_up_exps.weight"),
                          & tdn = gguf_->need(p + "ffn_down_exps.weight");
            if (tu.type != tg.type) throw std::runtime_error("hyper7: expert gate/up types differ");
            const size_t gb = tg.nbytes / E, db = tdn.nbytes / E;
            std::vector<int> order(E);
            for (int e = 0; e < E; ++e) order[e] = e;
            if (il < (int) stats_.size()) {
                const auto & cnt = stats_[il];
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
            }
            std::vector<int> owner(E, CPU_OWNER), slot(E, -1);
            L.n_gpu = n_gpu;
            for (int r = 0; r < n_gpu; ++r) { owner[order[r]] = d.g; slot[order[r]] = r; }
            // GPU copies
            uint8_t * gg = d.alloc<uint8_t>((size_t) std::max(1, n_gpu) * gb), * gu = d.alloc<uint8_t>((size_t) std::max(1, n_gpu) * gb),
                    * gd = d.alloc<uint8_t>((size_t) std::max(1, n_gpu) * db);
            for (int e = 0; e < E; ++e) {
                if (slot[e] < 0) continue;
                CUDA_CHECK(cudaMemcpy(gg + (size_t) slot[e] * gb, tg.data + (size_t) e * gb, gb, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(gu + (size_t) slot[e] * gb, tu.data + (size_t) e * gb, gb, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(gd + (size_t) slot[e] * db, tdn.data + (size_t) e * db, db, cudaMemcpyHostToDevice));
            }
            dev_bytes += (size_t) n_gpu * (2 * gb + db);
            MoeDev & m = L.moex;
            m.gate = gg; m.up = gu; m.down = gd;
            m.tg = tg.type; m.td = tdn.type;
            m.gate_bytes = gb; m.down_bytes = db;
            m.slot = d.upload(slot.data(), E);
            m.ff = ff; m.n = n;
            m.clamp = c.clamp_exp;
            CUDA_CHECK(cudaMemcpy(L.owner, owner.data(), E * sizeof(int), cudaMemcpyHostToDevice));
            // the CPU's experts: a compact copy (2 MB pages)
            const int ncpu = E - n_gpu;
            CpuExpertLayer cl;
            cl.tg = tg.type; cl.td = tdn.type;
            cl.gate_bytes = gb; cl.down_bytes = db;
            cl.cslot.assign(E, -1);
            if (ncpu > 0) {
                const size_t bytes = (size_t) ncpu * (2 * gb + db);
                uint8_t * buf = (uint8_t *) engine7_huge_alloc(bytes, host_bufs_);
                cl.gate = buf; cl.up = buf + (size_t) ncpu * gb; cl.down = buf + (size_t) ncpu * 2 * gb;
                std::vector<int> list;
                for (int e = 0; e < E; ++e) if (owner[e] == CPU_OWNER) { cl.cslot[e] = (int) list.size(); list.push_back(e); }
#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < ncpu; ++i) {
                    const int e = list[i];
                    memcpy((uint8_t *) cl.gate + (size_t) i * gb, tg.data + (size_t) e * gb, gb);
                    memcpy((uint8_t *) cl.up + (size_t) i * gb, tu.data + (size_t) e * gb, gb);
                    memcpy((uint8_t *) cl.down + (size_t) i * db, tdn.data + (size_t) e * db, db);
                }
                collapse_huge(buf, bytes);
                CUDA_CHECK(cudaHostRegister(buf, bytes, cudaHostRegisterPortable));
                host_bytes += bytes;
                // prefill parts: as many of the compact copy's experts as fit a staging buffer
                L.hg = cl.gate; L.hu = cl.up; L.hd = cl.down; L.gb = gb; L.db = db;
                const int per = std::max<int>(1, (int) (d.stage_bytes / (2 * gb + db)));
                for (int i0 = 0; i0 < ncpu; i0 += per) L.part_i0.push_back(i0);
                L.part_i0.push_back(ncpu);
                for (int j = 0; j + 1 < (int) L.part_i0.size(); ++j) {
                    std::vector<int> sl(E, -1);
                    for (int i = L.part_i0[j]; i < L.part_i0[j + 1]; ++i) sl[list[i]] = i - L.part_i0[j];
                    L.part_slot.push_back(d.upload(sl.data(), E));
                }
            }
            for (const GTensor * t : {&tg, &tu, &tdn}) {   // (the file's pages are not needed any more)
                const uintptr_t a0 = ((uintptr_t) t->data + 4095) & ~(uintptr_t) 4095, a1 = ((uintptr_t) t->data + t->nbytes) & ~(uintptr_t) 4095;
                if (a1 > a0) madvise((void *) a0, a1 - a0, MADV_DONTNEED);
            }
            cl.owned.resize(E);
            for (int e = 0; e < E; ++e) cl.owned[e] = owner[e] == CPU_OWNER;
            cpu_->set_layer(il, cl);
        }
        fprintf(stderr, "hyper7: device %d: %d of %d experts per layer in VRAM (%.1f GiB), %.1f GiB on the CPU\n", d.id, n_gpu, E,
                dev_bytes / 1073741824.0, host_bytes / 1073741824.0);
    }
}

Engine7::~Engine7() {
    cpu_.reset();
    for (auto & [p, sz] : host_bufs_) { cudaHostUnregister(p); munmap(p, sz); }
    devs_.clear();
    for (float * p : h_eg_) cudaFreeHost(p);
    for (void * p : {(void *) h_embd_, (void *) h_out_, (void *) cpu_rec_, (void *) cpu_out_, (void *) cpu_bulk_, (void *) cpu_bulk_out_})
        if (p) cudaFreeHost(p);
}

void Engine7::reset() { hist_.clear(); }

void Engine7::run_layer(Device & d, Layer & L, int nt, int64_t p0) {
    const Ds4Config & c = cfg_;
    cudaStream_t s = d.stream;
    const int n = c.n_embd, hcn = c.hc * n, HQ = c.n_head * c.head_dim, HD = c.head_dim, K = c.n_used;
    const float eps = c.eps;
    const bool bulk = nt > MAX_NT;
    auto mm = [&](const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni = NormIn{}) {
        if (rows <= MAX_NT) { gemv_q8(W.q8, x, xs, y, ys, nullptr, rows, s, ni); return; }
        to_half(x, xs, nullptr, W.k(), 0.0f, d.xh, rows, s);
        gemm_q8(W.q8, d.xh, rows, y, ys, nullptr, s);
    };
    auto mf32 = [&](const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nr) {   // y = x W^T (fp32)
        if (nr <= MAX_NT) { gemv_f32(W, rows, k, x, xs, y, ys, nr, s); return; }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, nr, k, &one, W, k, x, xs, &zero, y, ys) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("hyper7: cublasSgemm failed");
    };
    const float * freqs = L.ratio ? d.freqs_comp : d.freqs_win;
    const int il = L.il;
    // prefill: the CPU's experts go to the GPU in parts through the two staging buffers (the first two copies overlap the
    // attention; part j + 2's copy is issued once part j's experts are done)
    const bool stream_exp = bulk && stream_ && nt >= stream_min_ && L.np() > 0;
    auto stage_copy = [&](int j) {
        const int b = j & 1, i0 = L.part_i0[j], cnt = L.part_i0[j + 1] - i0;
        CUDA_CHECK(cudaStreamWaitEvent(d.cstream, d.ev_free[b], 0));
        uint8_t * st = d.stage[b];
        CUDA_CHECK(cudaMemcpyAsync(st, L.hg + (size_t) i0 * L.gb, (size_t) cnt * L.gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(st + (size_t) cnt * L.gb, L.hu + (size_t) i0 * L.gb, (size_t) cnt * L.gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(st + (size_t) 2 * cnt * L.gb, L.hd + (size_t) i0 * L.db, (size_t) cnt * L.db, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaEventRecord(d.ev_up[b], d.cstream));
    };
    if (stream_exp) for (int j = 0; j < std::min(2, L.np()); ++j) stage_copy(j);
    // ---- Engram ----
    if (L.eg >= 0) {
        const int ecols = c.eg_cols() * c.eg_hd;
        CUDA_CHECK(cudaMemcpyAsync(d.eg_x, h_eg_[L.eg], (size_t) nt * ecols * sizeof(float), cudaMemcpyHostToDevice, s));
        mm(L.eg_kv, d.eg_x, ecols, d.eg_kvo, (c.hc + 1) * n, nt);
        engram_gate7(d.h, d.eg_kvo, L.eg_w, c.hc, n, eps, nt, s);
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
            moe_order(L.moex, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
            to_half(d.x, n, nullptr, n, 0.0f, d.x16, nt, s);
            moe_gemm_gate_up(L.moex, d.x16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
            moe_gemm_down(L.moex, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
        } else {
            moe_gate_up(L.moex, d.x, n, d.ids, K, d.hexp, nt, s);
            moe_down(L.moex, d.hexp, d.ids, d.wts, K, d.yexp, nt, s);
        }
    }
    if (stream_exp) {
        if (L.n_gpu == 0) to_half(d.x, n, nullptr, n, 0.0f, d.x16, nt, s);
        for (int j = 0; j < L.np(); ++j) {
            const int b = j & 1, cnt = L.part_i0[j + 1] - L.part_i0[j];
            MoeDev ms = L.moex;
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
    if (stream_exp) moe_reduce(d.shout, d.ones, d.yexp, K, d.ffo, n, nt, d.ids, d.owner_all, d.g, CPU_OWNER, nullptr, nullptr, d.counter,
                               (unsigned) il, s);
    else moe_reduce(d.shout, d.ones, d.yexp, K, d.ffo, n, nt, d.ids, L.owner, d.g, CPU_OWNER, bulk ? &cpu_bulk_out_->seq : &cpu_out_[il].seq,
               bulk ? &cpu_bulk_out_->y[0][0] : &cpu_out_[il].y[0][0], d.counter, (unsigned) il, s);
    hc7_post(d.ffo, d.h, d.f_post, d.f_comb, n, c.hc, d.h2, nt, s);
    std::swap(d.h, d.h2);
    std::swap(d.pre, d.f_pre);   // the next block's attention mixes the streams with this block's FFN pre-weights
    if (const char * dp = getenv("HYPER7_DUMP")) {   // (tests) every layer's residual streams [nt][hc][n], appended to <dp>.l<il>
        CUDA_CHECK(cudaStreamSynchronize(s));
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

std::vector<int> Engine7::forward(const int * tokens, int nt, int pos, int nlog) {
    const Ds4Config & c = cfg_;
    if (nt < 1 || nt > R_) throw std::runtime_error("hyper7: bad token count");
    if (pos + nt > opt_.max_pos) throw std::runtime_error("hyper7: position exceeds max_pos");
    nlog = std::max(1, std::min({nlog, nt, MAX_NT}));
    const int n = c.n_embd;
    const bool bulk = nt > MAX_NT;
    // token rows (Q8_0 dequantized, as BF16 like the reference) and the Engram rows (the n-grams need the tokens before pos)
    const GTensor & te = gguf_->need("token_embd.weight");
    const auto * te_tr = ggml_get_type_traits((ggml_type) te.type);
    std::vector<int> toks;
    const int back = std::min<int>(pos, c.eg_ngram - 1);
    if ((int) hist_.size() < pos) throw std::runtime_error("hyper7: history shorter than the position");
    toks.insert(toks.end(), hist_.begin() + (pos - back), hist_.begin() + pos);
    toks.insert(toks.end(), tokens, tokens + nt);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; ++t) {
        float * out = h_embd_ + (size_t) t * n;
        te_tr->to_float(te.data + (size_t) tokens[t] * te.row_bytes(), out, n);
        for (int e = 0; e < n; ++e) out[e] = bf_round(out[e]);
    }
    const int cols = c.eg_cols(), hd = c.eg_hd;
    for (size_t i = 0; i < c.eg_layers.size(); ++i) {
        const GTensor & tab = gguf_->need("blk." + std::to_string(c.eg_layers[i]) + ".engram_embd.weight");
        const auto * tr = ggml_get_type_traits((ggml_type) tab.type);
        const size_t rb = tab.row_bytes();
#pragma omp parallel for schedule(dynamic, 4)
        for (int t = 0; t < nt; ++t) {
            int64_t rows[64];
            c.eg_rows((int) i, toks.data(), back + t, rows);
            // the absolute tokens: positions before 0 are pad (back < ngram - 1 only at the start)
            if (pos + t < c.eg_ngram - 1 && back < c.eg_ngram - 1) { /* eg_rows treats indices < 0 as pad */ }
            float * out = h_eg_[i] + (size_t) t * cols * hd;
            for (int j = 0; j < cols; ++j) {
                if (rows[j] < 0 || rows[j] >= tab.ne[1]) throw std::runtime_error("hyper7: Engram row out of range");
                tr->to_float(tab.data + (size_t) rows[j] * rb, out + (size_t) j * hd, hd);
                for (int e = 0; e < hd; ++e) out[(size_t) j * hd + e] = bf_round(out[(size_t) j * hd + e]);
            }
        }
    }
    ++fwd_counter_;
    {
        std::vector<int> slots;
        for (auto & dp : devs_) for (Layer & L : dp->layers) if (!(bulk && stream_ && nt >= stream_min_ && L.np() > 0) && L.n_gpu < c.n_expert) slots.push_back(L.il);
        if (!slots.empty()) cpu_->expect(fwd_counter_, slots, bulk);
    }
    for (size_t gi = 0; gi < devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        cudaStream_t s = d.stream;
        incr_counter(d.counter, s);
        if (gi == 0) {
            CUDA_CHECK(cudaMemcpyAsync(d.x, h_embd_, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
            embed7(d.x, d.h, n, c.hc, nt, s);
            std::vector<float> pre0((size_t) nt * 8, 0.0f);
            for (int t = 0; t < nt; ++t) pre0[(size_t) t * 8] = 1.0f;
            CUDA_CHECK(cudaMemcpyAsync(d.pre, pre0.data(), pre0.size() * sizeof(float), cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaStreamSynchronize(s));   // (pre0 is a host temporary)
        } else {
            Device & pd = *devs_[gi - 1];
            CUDA_CHECK(cudaStreamWaitEvent(s, pd.ev_done, 0));
            CUDA_CHECK(cudaMemcpyPeerAsync(d.h, d.id, pd.h, pd.id, (size_t) nt * c.hc * n * sizeof(float), s));
            CUDA_CHECK(cudaMemcpyPeerAsync(d.pre, d.id, pd.pre, pd.id, (size_t) nt * 8 * sizeof(float), s));
        }
        for (Layer & L : d.layers) run_layer(d, L, nt, pos);
        CUDA_CHECK(cudaEventRecord(d.ev_done, s));
    }
    // the head: the last nlog rows
    Device & d = *devs_.back();
    CUDA_CHECK(cudaSetDevice(d.id));
    cudaStream_t s = d.stream;
    const int r0 = nt - nlog;
    hc7_pre_norm(d.h + (size_t) r0 * c.hc * n, d.pre + (size_t) r0 * 8, d.out_norm, n, c.hc, c.eps, d.x, nlog, s);
    gemv_q8(d.output.q8, d.x, n, d.logits, c.n_vocab, nullptr, nlog, s);
    argmax_pairs(d.logits, c.n_vocab, c.n_vocab, 0, d.outp, nlog, s);
    CUDA_CHECK(cudaMemcpyAsync(h_out_, d.outp, (size_t) nlog * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
    auto t_wait = std::chrono::steady_clock::now();
    bool reported = false;
    for (;;) {
        const cudaError_t q = cudaStreamQuery(s);
        if (q == cudaSuccess) break;
        if (q != cudaErrorNotReady) CUDA_CHECK(q);
        if (!reported && std::chrono::steady_clock::now() - t_wait > std::chrono::seconds(20)) {
            reported = true;
            fprintf(stderr, "hyper7: forward %u slow/stuck; cpu: %s\n", fwd_counter_, cpu_->state().c_str());
        }
        std::this_thread::yield();
    }
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->stream)); }
    last_nlog_ = nlog;
    // the state now holds these tokens
    hist_.resize(pos);
    hist_.insert(hist_.end(), tokens, tokens + nt);
    std::vector<int> out(nlog);
    for (int t = 0; t < nlog; ++t) { int idx; memcpy(&idx, &h_out_[2 * t + 1], 4); out[t] = idx; }
    return out;
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

std::vector<int> Engine7::generate(const std::vector<int> & prompt, int n_gen, bool, GenStats * stats,
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
        int L = 0;
        while (L < cur && L < P - 1 && hist_[L] == prompt[L]) ++L;
        if (cur - L <= ring_ - cfg_.win - R_ - 8) s = L;
        if (s & 1) --s;
    }
    if (s == 0) hist_.clear();
    GenStats st;
    auto tp = clk::now();
    int next = -1;
    for (int c0 = s; c0 < P;) {
        const int len = std::min(R_, P - c0);
        next = forward(&prompt[c0], len, c0, 1)[0];
        c0 += len;
        if (prefill_cb_) prefill_cb_(c0, P, s);
    }
    if (sampling) next = sample_row(0, sp, P);
    st.t_prefill = std::chrono::duration<double>(clk::now() - tp).count();
    st.prompt_reused = s;
    std::vector<int> out;
    int p = P;
    auto t0 = clk::now();
    bool stop = false;
    auto emit = [&](int tok) {
        if (stop) return false;
        out.push_back(tok);
        if (on_token && !on_token(tok)) stop = true;
        if ((int) out.size() >= n_gen) stop = true;
        return !stop;
    };
    while (emit(next) && p + 1 < opt_.max_pos) {
        next = forward(&next, 1, p++, 1)[0];
        if (sampling) next = sample_row(0, sp, p);
        st.steps++;
    }
    st.tokens = (int) out.size();
    st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

void Engine7::save_expert_stats(const std::string & path) {
    const int nl = cfg_.n_layer, w = 1024;
    std::vector<std::vector<uint64_t>> tot(nl, std::vector<uint64_t>(w, 0));
    for (int l = 0; l < nl; ++l)
        for (int e = 0; e < w; ++e) {
            if (l < (int) stats_.size() && e < (int) stats_[l].size()) tot[l][e] += stats_[l][e];
            if (l < (int) cpu_->counts.size() && e < (int) cpu_->counts[l].size()) tot[l][e] += cpu_->counts[l][e];
        }
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    fwrite(&nl, 4, 1, f); fwrite(&w, 4, 1, f);
    for (auto & cc : tot) fwrite(cc.data(), 8, w, f);
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}

} // namespace hyper
