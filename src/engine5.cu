#include "engine5.h"

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
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace hyper {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace {

constexpr int CPU_OWNER = 3;
constexpr int R5 = MOE_BULK_ROWS;   // activation rows (prefill chunk)
constexpr int GSCORE_ROWS = 64;     // indexer: tokens scored at a time
constexpr int GLIST = 2052;         // max attended cells per token: 512 pools * 4 + 3 tail cells (+1)

// dense weight: Q8_0 (fragment-ordered int8) or anything else dequantized to fp16 at load
struct DW {
    Q8W q8;
    BF16W f;
    bool f16 = false;
    int n() const { return f16 ? f.n : q8.n; }
    int k() const { return f16 ? f.k : q8.k; }
};
struct RowRange { const GTensor * t; int64_t r0, r1; };
using ColRanges = std::vector<std::pair<int64_t, int64_t>>;
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

// Q8_0 rows from several tensors (same k), restricted to column blocks; a part with t == nullptr is r1 - r0 zero rows
Q8W upload_q8(const Alloc & alloc, int dev, const std::vector<RowRange> & parts, ColRanges cols = {}, int64_t k_full = 0) {
    for (auto & p : parts) if (p.t) k_full = p.t->ne[0];
    if (cols.empty()) cols.push_back({0, k_full / 32});
    std::vector<int64_t> blocks;
    for (auto & [b0, b1] : cols) for (int64_t b = b0; b < b1; ++b) blocks.push_back(b);
    const int64_t kb = (int64_t) blocks.size(), k = kb * 32;
    int64_t n = 0;
    for (auto & p : parts) {
        if (p.t && (p.t->type != GType::Q8_0 || p.t->ne[0] != k_full)) throw std::runtime_error("upload_q8: bad tensor " + p.t->name);
        n += p.r1 - p.r0;
    }
    std::vector<int8_t> qs((size_t) n * k, 0);
    std::vector<half> d((size_t) n * kb, __float2half(0.0f));
    int64_t row0 = 0;
    for (auto & p : parts) {
        if (p.t) {
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
        }
        row0 += p.r1 - p.r0;
    }
    return to_device_q8(alloc, dev, qs.data(), d.data(), (int) n, (int) k);
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

// requant: non-Q8_0 weights are re-quantized to Q8_0 (half the memory / bandwidth of fp16, error far below the source's)
DW upload_dense(const Alloc & alloc, int dev, const std::vector<RowRange> & parts, ColRanges cols = {}, bool requant = false) {
    DW w;
    bool all_q8 = true;
    for (auto & p : parts) all_q8 &= p.t->type == GType::Q8_0;
    if (all_q8) { w.q8 = upload_q8(alloc, dev, parts, cols); return w; }
    const int64_t k_full = parts[0].t->ne[0];
    if (cols.empty()) cols.push_back({0, k_full / 32});
    std::vector<int64_t> cidx;
    for (auto & [b0, b1] : cols) for (int64_t c = b0 * 32; c < b1 * 32; ++c) cidx.push_back(c);
    const int k = (int) cidx.size();
    int n = 0;
    for (auto & p : parts) n += (int) (p.r1 - p.r0);
    std::vector<float> rows((size_t) n * k);
    int row0 = 0;
    for (auto & p : parts) {
        if (p.t->ne[0] != k_full) throw std::runtime_error("upload_dense: k mismatch " + p.t->name);
        const std::vector<float> full = to_f32(*p.t, p.r0, p.r1);
#pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < p.r1 - p.r0; ++r)
            for (int j = 0; j < k; ++j) rows[(size_t) (row0 + r) * k + j] = full[(size_t) r * k_full + cidx[j]];
        row0 += (int) (p.r1 - p.r0);
    }
    if (requant) {
        const int kb = k / 32;
        std::vector<int8_t> qs((size_t) n * k);
        std::vector<half> d((size_t) n * kb);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t) n * kb; ++i) {
            const float * x = rows.data() + (size_t) i * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
            const float dd = amax / 127.0f, id = dd > 0.0f ? 1.0f / dd : 0.0f;
            d[i] = __float2half(dd);
            for (int j = 0; j < 32; ++j) qs[(size_t) i * 32 + j] = (int8_t) roundf(x[j] * id);
        }
        w.q8 = to_device_q8(alloc, dev, qs.data(), d.data(), n, k);
        return w;
    }
    const size_t ntile = (n + 15) / 16;
    std::vector<uint8_t> fq(ntile * (k / 16) * 512);
    repack_f32_frag(rows.data(), n, k, (size_t) k, fq.data());
    CUDA_CHECK(cudaSetDevice(dev));
    w.f16 = true;
    w.f.n = n; w.f.k = k;
    void * pd = alloc(fq.size());
    CUDA_CHECK(cudaMemcpy(pd, fq.data(), fq.size(), cudaMemcpyHostToDevice));
    w.f.q = (const uint4 *) pd;
    return w;
}

std::pair<int64_t, int64_t> split(int64_t n, int ndev, int g, int64_t align = 1) {
    const int64_t units = n / align;
    return {units * g / ndev * align, units * (g + 1) / ndev * align};
}

} // namespace

struct Engine5::DevLayer {
    bool mla = false, moe = false;
    // mHC (replicated): fn [24][4n], scale [3], base [24]; block norms
    DW hca_fn, hcf_fn;
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
    half * wkb = nullptr, * wvb = nullptr;   // fp16 [nh][512][256], [nh][256][512]
    float * q_a_norm = nullptr, * kv_a_norm = nullptr, * idx_proj = nullptr, * idx_lnw = nullptr, * idx_lnb = nullptr, * idx_ape = nullptr;
    half * lat = nullptr, * pooled = nullptr, * ring = nullptr;
    // FFN: dense layer or the shared expert (local hidden slice)
    DW gu, down;
    int ff_l = 0;
    float * router = nullptr, * exp_bias = nullptr;
    MoeDev moex;
    int * owner = nullptr;
    // prefill streaming: this GPU's share of the CPU-owned experts in two halves (st_list[h], uploaded into staging buffer h);
    // st_slot[h][e] = index in the half or -1; owner_bulk[e] = the GPU that computes e in a prefill chunk
    int * st_slot[2] = {}, * owner_bulk = nullptr;
    std::vector<int> st_list[2];
};

struct Engine5::Device {
    int id = 0, g = 0;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t g_main[MAX_NT + 1] = {}, g_restore[MAX_NT] = {};
    std::vector<DevLayer> layers;
    float * out_norm = nullptr;
    DW output;
    int vocab_off = 0;
    // activations [R5] rows
    float * x = nullptr, * res = nullptr, * xn = nullptr, * mix = nullptr, * hcw = nullptr, * bo = nullptr, * part = nullptr;
    float * big0 = nullptr, * o = nullptr, * qabs = nullptr, * olat = nullptr, * attn_part = nullptr;
    float * rlog = nullptr, * wts = nullptr, * sg = nullptr, * shgu = nullptr, * shh = nullptr, * shpart = nullptr;
    float * hexp = nullptr, * yexp = nullptr, * logits = nullptr, * conv_raw = nullptr, * iscores = nullptr;
    int * ids = nullptr, * pos = nullptr, * counter = nullptr, * order = nullptr, * order_n = nullptr, * egrp = nullptr;
    int * ilist = nullptr, * ilist_n = nullptr;
    unsigned * ihist = nullptr;   // indexer score histograms [GSCORE_ROWS][65536]
    half * xh = nullptr, * p16 = nullptr, * recv = nullptr, * mix16 = nullptr, * h16 = nullptr;
    float * topk = nullptr;
    int big_stride = 0;
    cublasHandle_t blas = nullptr;
    cudaStream_t cstream = nullptr;
    cudaEvent_t ev_up[2] = {}, ev_free[2] = {};
    uint8_t * stage[2] = {};
    size_t stage_bytes = 0;
    cudaEvent_t ev_ar[2] = {};
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
        for (auto & gr : g_restore) if (gr) cudaGraphExecDestroy(gr);
        for (auto & ev : ev_ar) if (ev) cudaEventDestroy(ev);
        if (blas) cublasDestroy(blas);
        for (auto & ev : ev_up) if (ev) cudaEventDestroy(ev);
        for (auto & ev : ev_free) if (ev) cudaEventDestroy(ev);
        if (cstream) cudaStreamDestroy(cstream);
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
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

Engine5::Engine5(const std::string & model_path, const Engine5Options & opt) : opt_(opt) {
    debug_ = getenv("HYPER4_DEBUG") != nullptr;
    nocpu_ = getenv("HYPER4_NOCPU") != nullptr;
    gguf_ = std::make_unique<GGUF>(model_path);
    cfg_ = Glm5Config::from_gguf(*gguf_);
    fprintf(stderr, "hyper5: %s\n", cfg_.describe().c_str());
    if (cfg_.n_expert_used > MOE_MAX_USED) throw std::runtime_error("hyper5: too many experts per token");
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    opt_.n_devices = std::min(opt_.n_devices, ndev);
    if (opt_.max_pos % 4) opt_.max_pos += 4 - opt_.max_pos % 4;
    for (int g = 0; g < opt_.n_devices; ++g) {
        auto dev = std::make_unique<Device>();
        dev->id = g; dev->g = g;
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->stream, cudaStreamNonBlocking));
        gemv_init(g);
        mla_init();
        for (auto & ev : dev->ev_ar) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->cstream, cudaStreamNonBlocking));
        for (auto & ev : dev->ev_up) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        for (auto & ev : dev->ev_free) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        if (cublasCreate(&dev->blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        cublasSetStream(dev->blas, dev->stream);
        cublasSetMathMode(dev->blas, CUBLAS_DEFAULT_MATH);
        devs_.push_back(std::move(dev));
    }
    const int nd = opt_.n_devices, n = cfg_.n_embd;
    CUDA_CHECK(cudaHostAlloc(&h_embd_, (size_t) R5 * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_stage_, (size_t) 2 * nd * R5 * n * sizeof(half), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_, sizeof(CpuMoeBulk), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_out_, sizeof(CpuMoeBulkOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_bulk_, 0, sizeof(CpuMoeBulk));
    memset((void *) cpu_bulk_out_, 0, sizeof(CpuMoeBulkOut));
    barrier_ = std::make_unique<Barrier4>(nd);
    CUDA_CHECK(cudaHostAlloc(&h_pos_, 4 * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_res_, (size_t) nd * MAX_NT * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_topk_, (size_t) nd * MAX_NT * TOPK * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_ids_, (size_t) cfg_.n_layer * R5 * cfg_.n_expert_used * sizeof(int), cudaHostAllocPortable));
    const size_t ll = (size_t) 2 * nd * MAX_NT * n / 2;
    CUDA_CHECK(cudaHostAlloc(&ar_ll_, ll * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_ll_, 0xff, ll * sizeof(uint2));
    const int slots = cfg_.n_layer;
    CUDA_CHECK(cudaHostAlloc(&cpu_rec_, (size_t) slots * sizeof(CpuMoeRec), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_out_, (size_t) slots * sizeof(CpuMoeOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_rec_, 0, (size_t) slots * sizeof(CpuMoeRec));
    memset((void *) cpu_out_, 0, (size_t) slots * sizeof(CpuMoeOut));
    if (const char * sp = getenv("HYPER4_STATS")) {
        FILE * f = fopen(sp, "rb");
        if (f) {
            int nl = 0, w = 0;
            if (fread(&nl, 4, 1, f) == 1 && fread(&w, 4, 1, f) == 1 && nl == cfg_.n_layer) {
                stats_.assign(nl, std::vector<uint64_t>(w));
                for (auto & s : stats_) if (fread(s.data(), 8, w, f) != (size_t) w) { stats_.clear(); break; }
            }
            fclose(f);
            fprintf(stderr, "hyper5: expert placement from %s (%s)\n", sp, stats_.empty() ? "unreadable, ignored" : "ok");
        }
    }
    cpu_ = std::make_unique<CpuMoe>(opt_.cpu_threads, n, cfg_.n_ff_exp, cfg_.n_expert_used, cpu_rec_, cpu_out_, slots, cpu_bulk_,
                                    cpu_bulk_out_);
    cpu_->set_clamp(cfg_.clamp_exp);
    adapt_ = !getenv("HYPER5_ADAPT") || atoi(getenv("HYPER5_ADAPT")) != 0;
    if (getenv("HYPER5_STREAM_MIN")) stream_min_ = atoi(getenv("HYPER5_STREAM_MIN"));
    if (getenv("HYPER5_DECAY")) adapt_decay_ = atof(getenv("HYPER5_DECAY"));
    if (getenv("HYPER5_PROMPT_W")) prompt_weight_ = atof(getenv("HYPER5_PROMPT_W"));
    if (getenv("HYPER5_ADAPT_EVERY")) adapt_every_ = std::max(1, atoi(getenv("HYPER5_ADAPT_EVERY")));
    if (getenv("HYPER5_ADAPT_BUDGET")) adapt_budget_ = atoi(getenv("HYPER5_ADAPT_BUDGET"));
    ehost_.resize(cfg_.n_layer);
    for (auto & H : ehost_) { H.score.assign(cfg_.n_expert, 0.0); H.last_count.assign(1024, 0); }
    load_weights();
}

void * Engine5::host_huge_alloc(size_t bytes) {
    const size_t H = 2u << 20, sz = (bytes + H - 1) / H * H;
    void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("host_huge_alloc: mmap failed");
    madvise(p, sz, MADV_HUGEPAGE);
    host_bufs_.push_back({p, sz});
    return p;
}

Engine5::~Engine5() {
    for (auto & sn : snaps_) snap_pool_.push_back(sn.h);
    for (auto & v : snap_pool_) for (float * p : v) cudaFreeHost(p);
    if (h_topk_) cudaFreeHost(h_topk_);
    if (h_ids_) cudaFreeHost(h_ids_);
    cpu_.reset();
    for (auto & [p, sz] : host_bufs_) munmap(p, sz);
    devs_.clear();
    for (void * p : {(void *) h_embd_, (void *) h_pos_, (void *) h_res_, (void *) ar_ll_, (void *) cpu_rec_, (void *) cpu_out_,
                     (void *) h_stage_, (void *) cpu_bulk_, (void *) cpu_bulk_out_})
        if (p) cudaFreeHost(p);
}

void Engine5::load_layer(Device & dev, DevLayer & L, int il) {
    const Glm5Config & c = cfg_;
    const int nd = opt_.n_devices, g = dev.g, n = c.n_embd;
    auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
    auto T = [&](const std::string & name) { return &gguf_->need(name); };
    auto f32 = [&](const std::string & name) {
        const std::vector<float> v = to_f32(gguf_->need(name));
        return dev.upload(v.data(), v.size());
    };
    auto full = [&](const std::string & name) { const GTensor * t = T(name); return upload_dense(A, dev.id, {{t, 0, t->rows()}}); };
    auto fp16_rows = [&](const std::string & name, int64_t r0, int64_t r1) {
        const std::vector<float> v = to_f32(gguf_->need(name), r0, r1);
        std::vector<half> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
        return dev.upload(h.data(), h.size());
    };
    const std::string p = "blk." + std::to_string(il) + ".";
    L.mla = c.is_mla[il];
    L.moe = is_moe(il);
    L.hca_fn = full(p + "hc_attn_fn.weight");
    L.hcf_fn = full(p + "hc_ffn_fn.weight");
    L.hca_scale = f32(p + "hc_attn_scale.weight");
    L.hca_base = f32(p + "hc_attn_base.weight");
    L.hcf_scale = f32(p + "hc_ffn_scale.weight");
    L.hcf_base = f32(p + "hc_ffn_base.weight");
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
                const std::vector<float> cw = to_f32(gguf_->need(p + nm));
                buf.insert(buf.end(), cw.begin() + a0 * c.conv, cw.begin() + a1 * c.conv);
            }
            L.conv_w = dev.upload(buf.data(), buf.size());
        }
        {
            const std::vector<float> dt = to_f32(gguf_->need(p + "ssm_dt.bias")), sa = to_f32(gguf_->need(p + "ssm_a"));
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
        L.wkb = fp16_rows(p + "attn_k_b.weight", h0 * c.kv_lora, h1 * c.kv_lora);
        L.wvb = fp16_rows(p + "attn_v_b.weight", h0 * c.v_dim, h1 * c.v_dim);
        L.wo = upload_dense(A, dev.id, {{T(p + "attn_output.weight"), 0, n}}, {{h0 * c.v_dim / 32, h1 * c.v_dim / 32}});
        L.q_a_norm = f32(p + "attn_q_a_norm.weight");
        L.kv_a_norm = f32(p + "attn_kv_a_norm.weight");
        {
            std::vector<float> pw = to_f32(gguf_->need(p + "indexer.proj.weight"));
            const float sc = 1.0f / sqrtf((float) (GIDX_DIM * GIDX_HEADS));
            for (auto & v : pw) v *= sc;
            L.idx_proj = dev.upload(pw.data(), pw.size());
        }
        L.idx_lnw = f32(p + "indexer.k_norm.weight");
        L.idx_lnb = f32(p + "indexer.k_norm.bias");
        L.idx_ape = f32(p + "indexer_compressor_ape.weight");
        L.lat = dev.alloc<half>((size_t) opt_.max_pos * MLA_LAT);
        L.pooled = dev.alloc<half>((size_t) (opt_.max_pos / 4 + 1) * GIDX_DIM);
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
        L.router = f32(p + "ffn_gate_inp.weight");
        L.exp_bias = f32(p + "exp_probs_b.bias");
    }
}

void Engine5::load_experts(int il, const std::vector<int> & quota) {
    const Glm5Config & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd;
    const std::string p = "blk." + std::to_string(il) + ".";
    const GTensor & tg = gguf_->need(p + "ffn_gate_exps.weight"), & tu = gguf_->need(p + "ffn_up_exps.weight"),
                  & tdn = gguf_->need(p + "ffn_down_exps.weight");
    if (tu.type != tg.type) throw std::runtime_error("expert gate/up types differ");
    const int E = c.n_expert, ff = c.n_ff_exp;
    std::vector<int> order(E);
    for (int e = 0; e < E; ++e) order[e] = e;
    if (il < (int) stats_.size()) {
        const auto & cnt = stats_[il];
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
    }
    std::vector<int> owner(E, CPU_OWNER), given(nd, 0);
    int total = 0;
    for (int q : quota) total += q;
    for (int r = 0; r < std::min(total, E); ++r) {
        int best = -1; double bd = -1;
        for (int g = 0; g < nd; ++g) {
            if (given[g] >= quota[g]) continue;
            const double d = 1.0 - (double) given[g] / quota[g];
            if (d > bd) { bd = d; best = g; }
        }
        if (best < 0) break;
        owner[order[r]] = best;
        given[best]++;
    }
    const size_t gb = tg.nbytes / E, db = tdn.nbytes / E;
    // host copy of every expert (2 MB pages, pinned): the CPU computes whichever experts it owns at the moment, the GPUs
    // stream them in a prefill chunk and take them over when the routing shifts
    const size_t bytes = (size_t) E * (2 * gb + db);
    uint8_t * buf = (uint8_t *) host_huge_alloc(bytes);
    CpuExpertLayer cl;
    cl.tg = tg.type; cl.td = tdn.type;
    cl.gate_bytes = gb; cl.down_bytes = db;
    cl.gate = buf; cl.up = buf + (size_t) E * gb; cl.down = buf + (size_t) E * 2 * gb;
#pragma omp parallel for schedule(dynamic)
    for (int e = 0; e < E; ++e) {
        memcpy((uint8_t *) cl.gate + (size_t) e * gb, tg.data + (size_t) e * gb, gb);
        memcpy((uint8_t *) cl.up + (size_t) e * gb, tu.data + (size_t) e * gb, gb);
        memcpy((uint8_t *) cl.down + (size_t) e * db, tdn.data + (size_t) e * db, db);
    }
    CUDA_CHECK(cudaHostRegister(buf, bytes, cudaHostRegisterPortable));
    for (const GTensor * t : {&tg, &tu, &tdn}) {   // (the file's pages are not needed any more)
        const uintptr_t a0 = ((uintptr_t) t->data + 4095) & ~(uintptr_t) 4095, a1 = ((uintptr_t) t->data + t->nbytes) & ~(uintptr_t) 4095;
        if (a1 > a0) madvise((void *) a0, a1 - a0, MADV_DONTNEED);
    }
    cl.owned.resize(E);
    for (int e = 0; e < E; ++e) cl.owned[e] = owner[e] == CPU_OWNER;
    cpu_->set_layer(il, cl);
    ExpertHost & H = ehost_[il];
    if (il < (int) stats_.size()) {   // prior: the calibration's routing shares, weighted like ~256 tokens of routing
        double tot = 0;
        for (int e = 0; e < E; ++e) tot += (double) stats_[il][e];
        if (tot > 0) for (int e = 0; e < E; ++e) H.score[e] = (double) stats_[il][e] / tot * 256.0 * c.n_expert_used;
    }
    H.gate = cl.gate; H.up = cl.up; H.down = cl.down; H.gb = gb; H.db = db;
    H.owner = owner;
    H.slot.assign(nd, std::vector<int>(E, -1));
    for (auto & dp : devs_) {
        Device & dev = *dp;
        DevLayer & L = dev.layers[il];
        const int g = dev.g;
        std::vector<int> & slot = H.slot[g];
        int nl = 0;
        for (int e = 0; e < E; ++e) if (owner[e] == g) slot[e] = nl++;
        uint8_t * dg = dev.alloc<uint8_t>(gb * std::max(nl, 1)), * du = dev.alloc<uint8_t>(gb * std::max(nl, 1));
        uint8_t * dd = dev.alloc<uint8_t>(db * std::max(nl, 1));
        for (int e = 0; e < E; ++e) {
            if (slot[e] < 0) continue;
            CUDA_CHECK(cudaMemcpy(dg + slot[e] * gb, cl.gate + (size_t) e * gb, gb, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(du + slot[e] * gb, cl.up + (size_t) e * gb, gb, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dd + slot[e] * db, cl.down + (size_t) e * db, db, cudaMemcpyHostToDevice));
        }
        L.moex.gate = dg; L.moex.up = du; L.moex.down = dd;
        L.moex.tg = tg.type; L.moex.td = tdn.type;
        L.moex.gate_bytes = gb; L.moex.down_bytes = db;
        L.moex.ff = ff; L.moex.n = n;
        L.moex.clamp = c.clamp_exp;
        L.moex.slot = dev.upload(slot.data(), slot.size());
        L.owner = dev.upload(owner.data(), owner.size());
        if (opt_.stream_experts) {
            for (auto & sl : L.st_slot) sl = dev.alloc<int>(E);
            L.owner_bulk = dev.alloc<int>(E);
        }
    }
    if (opt_.stream_experts) {   // staging: the largest half of a GPU's share (the CPU-owned count per layer never changes)
        const int nc = E - total;
        std::vector<double> wbw(nd, 2.0);
        if (nd > 1) wbw[1] = 1.0;   // GPU 1 sits on a x8 link on this box
        double tw = 0;
        for (double w : wbw) tw += w;
        for (auto & dp : devs_) {
            const int share = (int) std::ceil(nc * wbw[dp->g] / tw) + 1;
            dp->stage_bytes = std::max(dp->stage_bytes, (size_t) ((share + 1) / 2) * (2 * gb + db));
        }
        H.stream_dirty = true;
    }
}

// prefill streaming maps of layer il from the current placement: the CPU-owned experts dealt to the GPUs by PCIe bandwidth
// (x16 : x8 : x16), each GPU's share in two halves (staging buffers 0 / 1)
void Engine5::rebuild_stream(int il) {
    ExpertHost & H = ehost_[il];
    const int nd = opt_.n_devices, E = cfg_.n_expert;
    std::vector<int> cpu;
    for (int e = 0; e < E; ++e) if (H.owner[e] == CPU_OWNER) cpu.push_back(e);
    const int nc = (int) cpu.size();
    std::vector<double> wbw(nd, 2.0);
    if (nd > 1) wbw[1] = 1.0;
    double tw = 0, acc = 0;
    for (double w : wbw) tw += w;
    std::vector<int> bulk_owner = H.owner;
    int a = 0;
    for (auto & dp : devs_) {
        DevLayer & L = dp->layers[il];
        acc += wbw[dp->g];
        const int b = dp->g == nd - 1 ? nc : (int) (nc * acc / tw + 0.5);
        const int m = (a + b) / 2;
        const int r0[2] = {a, m}, r1[2] = {m, b};
        for (int h = 0; h < 2; ++h) {
            std::vector<int> sl(E, -1);
            L.st_list[h].assign(cpu.begin() + r0[h], cpu.begin() + r1[h]);
            if ((size_t) L.st_list[h].size() * (2 * H.gb + H.db) > dp->stage_bytes) throw std::runtime_error("rebuild_stream: staging too small");
            for (int i = 0; i < (int) L.st_list[h].size(); ++i) { sl[L.st_list[h][i]] = i; bulk_owner[L.st_list[h][i]] = dp->g; }
            CUDA_CHECK(cudaSetDevice(dp->id));
            CUDA_CHECK(cudaMemcpy(L.st_slot[h], sl.data(), E * sizeof(int), cudaMemcpyHostToDevice));
        }
        a = b;
    }
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaMemcpy(dp->layers[il].owner_bulk, bulk_owner.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    }
    H.stream_dirty = false;
}

// adaptive placement: routing scores (decayed counts: prompt routing of the prefill chunks, the CPU side's decode counts)
// pull the hottest CPU experts onto the GPUs in exchange for their coldest experts (same GPU, same slot), at most max_swaps
// exchanges in total, the largest gains first. Runs between forwards (nothing in flight).
void Engine5::rebalance(int max_swaps) {
    const Glm5Config & c = cfg_;
    const int E = c.n_expert;
    // decode routing seen by the CPU side since the last call
    for (int il = 0; il < c.n_layer; ++il) {
        if (!is_moe(il)) continue;
        auto & sc = ehost_[il].score;
        const auto & cnt = cpu_->counts[il];
        auto & last = ehost_[il].last_count;
        for (int e = 0; e < E; ++e) { sc[e] = sc[e] * adapt_decay_ + (double) (cnt[e] - last[e]); last[e] = cnt[e]; }
    }
    struct Sw { double gain; int il, ein, eout, g; };
    std::vector<Sw> cand;
    for (int il = 0; il < c.n_layer; ++il) {
        if (!is_moe(il)) continue;
        ExpertHost & H = ehost_[il];
        std::vector<int> cpu, gpu;
        for (int e = 0; e < E; ++e) (H.owner[e] == CPU_OWNER ? cpu : gpu).push_back(e);
        std::sort(cpu.begin(), cpu.end(), [&](int a, int b) { return H.score[a] > H.score[b]; });
        std::sort(gpu.begin(), gpu.end(), [&](int a, int b) { return H.score[a] < H.score[b]; });
        for (size_t i = 0; i < std::min(cpu.size(), gpu.size()); ++i) {
            const double si = H.score[cpu[i]], so = H.score[gpu[i]];
            if (si < so * 1.5 + 6.0) break;   // (hysteresis: noise must not shuffle experts back and forth)
            cand.push_back({si - so, il, cpu[i], gpu[i], H.owner[gpu[i]]});
        }
    }
    std::sort(cand.begin(), cand.end(), [](const Sw & a, const Sw & b) { return a.gain > b.gain; });
    if ((int) cand.size() > max_swaps) cand.resize(max_swaps);
    if (cand.empty()) return;
    // weights first (copy engines of all GPUs in parallel), then the tables
    for (const Sw & w : cand) {
        ExpertHost & H = ehost_[w.il];
        Device & d = *devs_[w.g];
        DevLayer & L = d.layers[w.il];
        const int k = H.slot[w.g][w.eout];
        CUDA_CHECK(cudaSetDevice(d.id));
        CUDA_CHECK(cudaMemcpyAsync((uint8_t *) L.moex.gate + (size_t) k * H.gb, H.gate + (size_t) w.ein * H.gb, H.gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync((uint8_t *) L.moex.up + (size_t) k * H.gb, H.up + (size_t) w.ein * H.gb, H.gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync((uint8_t *) L.moex.down + (size_t) k * H.db, H.down + (size_t) w.ein * H.db, H.db, cudaMemcpyHostToDevice, d.cstream));
        H.slot[w.g][w.ein] = k;
        H.slot[w.g][w.eout] = -1;
        H.owner[w.ein] = w.g;
        H.owner[w.eout] = CPU_OWNER;
        cpu_->set_owned(w.il, w.ein, false);
        cpu_->set_owned(w.il, w.eout, true);
        H.tables_dirty = true;
        H.stream_dirty = true;
    }
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->cstream)); }
    for (int il = 0; il < c.n_layer; ++il) {
        ExpertHost & H = ehost_[il];
        if (!is_moe(il) || !H.tables_dirty) continue;
        for (auto & dp : devs_) {
            DevLayer & L = dp->layers[il];
            CUDA_CHECK(cudaSetDevice(dp->id));
            CUDA_CHECK(cudaMemcpy((void *) L.moex.slot, H.slot[dp->g].data(), E * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(L.owner, H.owner.data(), E * sizeof(int), cudaMemcpyHostToDevice));
        }
        H.tables_dirty = false;
    }
    n_swaps_ += (int) cand.size();
    if (getenv("HYPER5_ADAPT_LOG")) fprintf(stderr, "hyper5: rebalance: %zu exchanges (total %d)\n", cand.size(), n_swaps_);
}

void Engine5::load_weights() {
    auto t0 = std::chrono::steady_clock::now();
    const Glm5Config & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd, hcn = MHC * n;
    for (auto & dp : devs_) {
        Device & dev = *dp;
        const int g = dev.g;
        auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
        dev.layers.resize(c.n_layer);
        for (int il = 0; il < c.n_layer; ++il) load_layer(dev, dev.layers[il], il);
        { const std::vector<float> v = to_f32(gguf_->need("output_norm.weight")); dev.out_norm = dev.upload(v.data(), v.size()); }
        auto [o0, o1] = split(c.n_vocab, nd, g);
        dev.vocab_off = (int) o0;
        dev.output = upload_dense(A, dev.id, {{&gguf_->need("output.weight"), o0, o1}}, {}, !getenv("HYPER5_OUT_F16"));
        int nh_max = 0;
        for (auto & L : dev.layers) nh_max = std::max(nh_max, L.nh);
        dev.big_stride = std::max(kda_stride(nh_max), mla_stride(c, nh_max));
        const int R = R5, K = c.n_expert_used;
        int ffl = 0;   // largest local FFN hidden slice
        for (auto & L : dev.layers) ffl = std::max(ffl, L.ff_l);
        dev.x = dev.alloc<float>((size_t) R * n);
        dev.res = dev.alloc<float>((size_t) R * hcn);
        dev.xn = dev.alloc<float>((size_t) R * n);
        dev.mix = dev.alloc<float>((size_t) R * 32);
        dev.hcw = dev.alloc<float>((size_t) R * MHC_W);
        dev.bo = dev.alloc<float>((size_t) R * n);
        dev.part = dev.alloc<float>((size_t) R * n);
        dev.big0 = dev.alloc<float>((size_t) R * dev.big_stride);
        dev.o = dev.alloc<float>((size_t) R * std::max(nh_max * 128, nh_max * c.v_dim));
        dev.qabs = dev.alloc<float>((size_t) R * nh_max * MLA_LAT);
        dev.olat = dev.alloc<float>((size_t) R * nh_max * MLA_LAT);
        dev.attn_part = dev.alloc<float>(mla_part_floats(nh_max, MAX_NT));
        dev.rlog = dev.alloc<float>((size_t) R * c.n_expert);
        dev.ids = dev.alloc<int>((size_t) R * K);
        dev.wts = dev.alloc<float>((size_t) R * K);
        dev.sg = dev.alloc<float>(R);
        dev.shgu = dev.alloc<float>((size_t) R * 2 * ffl);
        dev.shh = dev.alloc<float>((size_t) R * ffl);
        dev.shpart = dev.alloc<float>((size_t) R * n);
        dev.hexp = dev.alloc<float>((size_t) MAX_NT * K * c.n_ff_exp);
        dev.yexp = dev.alloc<float>((size_t) R * K * n);
        dev.logits = dev.alloc<float>((size_t) MAX_NT * dev.output.n());
        dev.xh = dev.alloc<half>((size_t) R * std::max({hcn, ffl, nh_max * MLA_LAT}));
        dev.p16 = dev.alloc<half>((size_t) R * n);
        dev.recv = dev.alloc<half>((size_t) std::max(1, nd - 1) * R * n);
        dev.order = dev.alloc<int>((size_t) R * K);
        dev.order_n = dev.alloc<int>(2);
        dev.topk = dev.alloc<float>((size_t) MAX_NT * TOPK * 2);
        dev.egrp = dev.alloc<int>((size_t) 3 * c.n_expert);
        dev.mix16 = dev.alloc<half>((size_t) R * n);
        dev.h16 = dev.alloc<half>((size_t) R * K * c.n_ff_exp);
        dev.conv_raw = dev.alloc<float>((size_t) R * 3 * nh_max * 128);
        dev.iscores = dev.alloc<float>((size_t) GSCORE_ROWS * (opt_.max_pos / 4 + 4));
        dev.ilist = dev.alloc<int>((size_t) R * GLIST);
        dev.ilist_n = dev.alloc<int>(R);
        dev.ihist = dev.alloc<unsigned>((size_t) GSCORE_ROWS * 65536);
        dev.pos = dev.alloc<int>(1);
        dev.counter = dev.alloc<int>(1);
    }
    // experts fill what is left on each GPU (minus a runtime reserve)
    {
        size_t eb = 0, eb_max = 0;
        int nm = 0;
        for (int il = 0; il < c.n_layer; ++il) {
            if (!is_moe(il)) continue;
            const std::string p = "blk." + std::to_string(il) + ".";
            const size_t b = (gguf_->need(p + "ffn_gate_exps.weight").nbytes * 2 + gguf_->need(p + "ffn_down_exps.weight").nbytes) / c.n_expert;
            eb += b;
            eb_max = std::max(eb_max, b);
            ++nm;
        }
        eb /= std::max(nm, 1);
        std::vector<int> quota(nd);
        std::vector<size_t> freeb(nd);
        for (int g = 0; g < nd; ++g) {
            CUDA_CHECK(cudaSetDevice(devs_[g]->id));
            size_t tot = 0;
            CUDA_CHECK(cudaMemGetInfo(&freeb[g], &tot));
        }
        std::vector<double> stage_est(nd, 0.0);
        for (int it = 0; it < 3; ++it) {
            int tq = 0;
            for (int g = 0; g < nd; ++g) {
                const double cap = std::max(0.0, (double) freeb[g] - opt_.vram_reserve_gib * 1073741824.0 - stage_est[g]);
                quota[g] = std::min((int) (cap / eb / std::max(nm, 1)), (int) (c.n_expert * opt_.gpu_expert_frac / nd + 0.999));
                tq += quota[g];
            }
            if (!opt_.stream_experts) break;
            const double cold = (std::max(0, c.n_expert - tq) + 2) * (double) eb_max;
            for (int g = 0; g < nd; ++g) stage_est[g] = 1.05 * cold * (nd > 1 && g == 1 ? 1.0 : 2.0) / (2.0 * nd - (nd > 1 ? 1.0 : 0.0));
        }
        int tq = 0;
        for (int q : quota) tq += q;
        fprintf(stderr, "hyper5: experts per layer on GPUs:");
        for (int q : quota) fprintf(stderr, " %d", q);
        fprintf(stderr, " (%.0f%% of %d), the rest on the CPU\n", 100.0 * tq / c.n_expert, c.n_expert);
        for (int il = 0; il < c.n_layer; ++il) if (is_moe(il)) load_experts(il, quota);
        if (opt_.stream_experts)
            for (auto & dp : devs_) {
                for (auto & sb : dp->stage) sb = dp->alloc<uint8_t>(std::max<size_t>(dp->stage_bytes, 1));
                fprintf(stderr, "hyper5: device %d streams prefill experts through 2 x %.0f MiB\n", dp->id, dp->stage_bytes / 1048576.0);
            }
    }
    for (auto & dp : devs_) fprintf(stderr, "hyper5: device %d holds %.2f GiB\n", dp->id, dp->used / 1073741824.0);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "hyper5: weights loaded in %.1f s (%d GPUs)\n", s, nd);
}

void Engine5::reset() {
    const Glm5Config & c = cfg_;
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        for (auto & L : dp->layers)
            if (!L.mla) {
                CUDA_CHECK(cudaMemset(L.conv_state, 0, (size_t) (c.conv - 1) * 3 * L.nh * 128 * sizeof(float)));
                CUDA_CHECK(cudaMemset(L.state, 0, (size_t) L.nh * 128 * 128 * sizeof(float)));
            }
    }
}

void Engine5::record_main(int gi, int nt) {
    const Glm5Config & c = cfg_;
    Device & d = *devs_[gi];
    cudaStream_t s = d.stream;
    const int n = c.n_embd, hcn = MHC * n, K = c.n_expert_used, bs = d.big_stride;
    const int nd = opt_.n_devices;
    const float eps = c.rms_eps;
    const bool bulk = nt > MAX_NT;
    auto mm = [&](const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni = NormIn{}) {
        if (rows <= MAX_NT) {
            if (W.f16) gemv_bf16(W.f, x, xs, y, ys, nullptr, rows, s, ni);
            else gemv_q8(W.q8, x, xs, y, ys, nullptr, rows, s, ni);
            return;
        }
        to_half(x, xs, nullptr, W.k(), 0.0f, d.xh, rows, s);
        if (W.f16) gemm_f16(W.f, d.xh, rows, y, ys, nullptr, s);
        else gemm_q8(W.q8, d.xh, rows, y, ys, nullptr, s);
    };
    auto f32mm = [&](const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nr) {
        if (nr <= MAX_NT) { gemv_f32(W, rows, k, x, xs, y, ys, nr, s); return; }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, nr, k, &one, W, k, x, xs, &zero, y, ys) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemm failed");
    };
    // per-head projections (fp16 W [H][R][C]): warp GEMV, or a strided batched tensor-core GEMM for a prefill chunk
    auto hmm = [&](const half * W, int H, int R, int C, const float * x, int xs, float * y, int ys, int nr) {
        if (nr <= MAX_NT) { head_gemv(W, H, R, C, x, xs, y, ys, nr, s); return; }
        to_half(x, xs, nullptr, H * C, 0.0f, d.xh, nr, s);
        const float one = 1.0f, zero = 0.0f;
        if (cublasGemmStridedBatchedEx(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, R, nr, C, &one, W, CUDA_R_16F, C, (long long) R * C, d.xh, CUDA_R_16F,
                                       H * C, C, &zero, y, CUDA_R_32F, ys, R, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasGemmStridedBatchedEx failed");
    };
    float * R = d.res;
    int * P = d.pos;
    const int pos_host = h_pos_[0];
    CUDA_CHECK(cudaMemcpyAsync(P, h_pos_, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.x, h_embd_, (size_t) nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    const bool streaming = bulk && opt_.stream_experts && nt >= stream_min_;
    if (streaming) {
        int first = 0;
        while (first < c.n_layer && !d.layers[first].owner_bulk) ++first;
        if (first < c.n_layer) { upload_stage(d, first, 0); upload_stage(d, first, 1); }
    }
    mhc_init(R, d.x, n, nt, s);
    int call = 0;
    auto dbg = [&](const char * what, int il, const float * buf, size_t cnt) {
        if (!debug_) return;
        CUDA_CHECK(cudaStreamSynchronize(s));
        const cudaError_t err = cudaGetLastError();
        std::vector<float> h(cnt);
        CUDA_CHECK(cudaMemcpy(h.data(), buf, cnt * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0; double mx = 0, sum = 0, sq = 0;
        for (float v : h) { if (!std::isfinite(v)) ++bad; else { mx = std::max(mx, (double) std::fabs(v)); sum += v; sq += (double) v * v; } }
        if (bad || err != cudaSuccess || (il < 4 && !getenv("HYPER4_SUMS")))
            fprintf(stderr, "dbg L%d %-12s err=%s nonfinite=%d max|x|=%.3g\n", il, what, cudaGetErrorString(err), bad, mx);
        if (getenv("HYPER4_SUMS")) fprintf(stderr, "SUMS %d %s %.6g %.6g\n", il, what, sum, sq);
        if (bad || err != cudaSuccess) throw std::runtime_error("debug stop");
    };
    int dcall = 0;
    auto allreduce = [&] {   // d.bo = sum over the GPUs of d.part
        CUDA_CHECK(cudaMemsetAsync(d.bo, 0, (size_t) nt * n * sizeof(float), s));
        if (!bulk) { allreduce_add_ll16(d.bo, d.part, ar_ll_, d.g, nd, nt * n, d.counter, call++, s, nullptr); return; }
        const int par = dcall++ & 1;
        const size_t N = (size_t) nt * n;
        auto stage = [&](int g) { return h_stage_ + ((size_t) par * nd + g) * R5 * n; };
        to_half(d.part, n, nullptr, n, 0.0f, d.p16, nt, s);
        CUDA_CHECK(cudaMemcpyAsync(stage(d.g), d.p16, N * sizeof(half), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaEventRecord(d.ev_ar[par], s));
        barrier_->wait();
        int j = 0;
        for (int p = 0; p < nd; ++p) {
            if (p == d.g) continue;
            CUDA_CHECK(cudaStreamWaitEvent(s, devs_[p]->ev_ar[par], 0));
            CUDA_CHECK(cudaMemcpyAsync(d.recv + (size_t) j * R5 * n, stage(p), N * sizeof(half), cudaMemcpyHostToDevice, s));
            ++j;
        }
        add_parts(d.bo, d.p16, d.recv, (size_t) R5 * n, nd - 1, (int) N, s);
    };
    // mHC pre-mix: res -> d.xn (block input after its RMS norm) and the post / comb weights
    auto hc_pre = [&](const DW & fn, const float * scale, const float * base, const float * norm_w) {
        mm(fn, R, hcn, d.mix, 32, nt);
        mhc_pre(R, d.mix, 32, scale, base, norm_w, eps, c.hc_eps, c.sinkhorn, n, d.hcw, d.xn, nt, s);
    };
    dbg("embed", -1, R, (size_t) nt * hcn);
    for (int il = 0; il < c.n_layer; ++il) {
        DevLayer & L = d.layers[il];
        const int nh = L.nh;
        // ---- token mixer ----
        hc_pre(L.hca_fn, L.hca_scale, L.hca_base, L.attn_norm);
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
            hmm(L.wkb, nh, MLA_LAT, c.qk_dim, d.big0 + qo, bs, d.qabs, nh * MLA_LAT, nt);
            const bool use_list = !(bulk && dense_chunk);
            mla_attn(d.qabs, nh * MLA_LAT, L.lat, P, nh, 1.0f / sqrtf((float) c.qk_dim), nt, use_list ? d.ilist : nullptr, GLIST,
                     use_list ? d.ilist_n : nullptr, d.attn_part, d.olat, nh * MLA_LAT, s);
            hmm(L.wvb, nh, c.v_dim, MLA_LAT, d.olat, nh * MLA_LAT, d.o, nh * c.v_dim, nt);
            mm(L.wo, d.o, nh * c.v_dim, d.part, n, nt);
        }
        dbg(L.mla ? "mla_part" : "kda_part", il, d.part, (size_t) nt * n);
        allreduce();
        mhc_post(R, d.bo, d.hcw, n, nt, s);
        dbg("attn_res", il, R, (size_t) nt * hcn);
        // ---- FFN ----
        hc_pre(L.hcf_fn, L.hcf_scale, L.hcf_base, L.ffn_norm);
        float * ffo = L.moe ? d.shpart : d.part;
        mm(L.gu, d.xn, n, d.shgu, 2 * L.ff_l, nt);
        if (nt <= MAX_NT) {
            NormIn glu; glu.act = 3; glu.glu_off = L.ff_l; glu.act_scale = L.moe ? c.clamp_sh : c.clamp_sh;
            mm(L.down, d.shgu, 2 * L.ff_l, ffo, n, nt, glu);
        } else {
            swiglu_clamp(d.shgu, 2 * L.ff_l, L.ff_l, d.shh, L.ff_l, L.ff_l, c.clamp_sh, nt, s);
            mm(L.down, d.shh, L.ff_l, ffo, n, nt);
        }
        if (L.moe) {
            f32mm(L.router, c.n_expert, n, d.xn, n, d.rlog, c.n_expert, nt);
            moe_route_sig(d.rlog, c.n_expert, L.exp_bias, c.n_expert, K, c.w_scale, d.ids, d.wts, d.sg, nt, s);
            dbg("route_w", il, d.wts, (size_t) nt * K);
            if (bulk && d.g == 0 && adapt_)   // prompt routing for the adaptive placement
                CUDA_CHECK(cudaMemcpyAsync(h_ids_ + (size_t) il * R5 * K, d.ids, (size_t) nt * K * sizeof(int), cudaMemcpyDeviceToHost, s));
            const bool stream = streaming && L.owner_bulk;
            if (d.g == 0 && !stream) {
                if (bulk) moe_publish(&cpu_bulk_->seq, &cpu_bulk_->nt, &cpu_bulk_->ids[0][0], &cpu_bulk_->wts[0][0], &cpu_bulk_->x[0][0],
                                      d.xn, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
                else moe_publish(&cpu_rec_[il].seq, &cpu_rec_[il].nt, &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0],
                                 d.xn, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
            }
            if (bulk) {
                moe_order(L.moex, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
                to_half(d.xn, n, nullptr, n, 0.0f, d.mix16, nt, s);
                moe_gemm_gate_up(L.moex, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
                moe_gemm_down(L.moex, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
                if (stream)   // this GPU's share of the CPU experts, in two halves through the two staging buffers
                    for (int sb = 0; sb < 2; ++sb) {
                        MoeDev ms = L.moex;
                        const int cnt = (int) L.st_list[sb].size();
                        ms.gate = d.stage[sb];
                        ms.up = d.stage[sb] + (size_t) cnt * L.moex.gate_bytes;
                        ms.down = d.stage[sb] + (size_t) 2 * cnt * L.moex.gate_bytes;
                        ms.slot = L.st_slot[sb];
                        CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_up[sb], 0));
                        moe_order(ms, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
                        moe_gemm_gate_up(ms, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
                        moe_gemm_down(ms, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
                        CUDA_CHECK(cudaEventRecord(d.ev_free[sb], s));
                        int nx = il + 1;
                        while (nx < c.n_layer && !d.layers[nx].owner_bulk) ++nx;
                        if (nx < c.n_layer) upload_stage(d, nx, sb);
                    }
            } else {
                moe_gate_up(L.moex, d.xn, n, d.ids, K, d.hexp, nt, s);
                moe_down(L.moex, d.hexp, d.ids, d.wts, K, d.yexp, nt, s);
            }
            const volatile unsigned * cflag = d.g == 0 && !nocpu_ && !stream ? (bulk ? &cpu_bulk_out_->seq : &cpu_out_[il].seq) : nullptr;
            moe_reduce(d.shpart, d.sg, d.yexp, K, d.part, n, nt, d.ids, stream ? L.owner_bulk : L.owner, d.g, CPU_OWNER, cflag,
                       bulk ? &cpu_bulk_out_->y[0][0] : &cpu_out_[il].y[0][0], d.counter, (unsigned) il, s);
        }
        dbg("ffn_part", il, d.part, (size_t) nt * n);
        allreduce();
        mhc_post(R, d.bo, d.hcw, n, nt, s);
        dbg("l_out", il, R, (size_t) nt * hcn);
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

void Engine5::upload_stage(Device & d, int il, int sb) {
    DevLayer & L = d.layers[il];
    const ExpertHost & H = ehost_[il];
    const std::vector<int> & lst = L.st_list[sb];
    const int cnt = (int) lst.size();
    CUDA_CHECK(cudaStreamWaitEvent(d.cstream, d.ev_free[sb], 0));
    const size_t gb = H.gb, db = H.db;
    for (int i = 0; i < cnt;) {   // runs of consecutive experts in one copy each
        int j = i + 1;
        while (j < cnt && lst[j] == lst[j - 1] + 1) ++j;
        const size_t e = (size_t) lst[i], r = (size_t) (j - i);
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) i * gb, H.gate + e * gb, r * gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) (cnt + i) * gb, H.up + e * gb, r * gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) 2 * cnt * gb + (size_t) i * db, H.down + e * db, r * db, cudaMemcpyHostToDevice,
                                   d.cstream));
        i = j;
    }
    CUDA_CHECK(cudaEventRecord(d.ev_up[sb], d.cstream));
}

// roll the KDA state back to the snapshot after row keep-1 of the last verification (the MLA side needs nothing: rejected rows'
// latents / pools are rewritten when those positions are processed again)
void Engine5::record_restore(int gi, int keep) {
    const Glm5Config & c = cfg_;
    Device & d = *devs_[gi];
    for (auto & L : d.layers) {
        if (L.mla || !L.conv_snap) continue;
        const size_t cs = (size_t) (c.conv - 1) * 3 * L.nh * 128, ss = (size_t) L.nh * 128 * 128;
        CUDA_CHECK(cudaMemcpyAsync(L.conv_state, L.conv_snap + (keep - 1) * cs, cs * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
        CUDA_CHECK(cudaMemcpyAsync(L.state, L.state_snap + (keep - 1) * ss, ss * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
    }
}

void Engine5::restore(int keep) {
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaGraphLaunch(dp->g_restore[keep], dp->stream));
    }
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->stream)); }
}

void Engine5::build_graphs() {
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
        if (opt_.n_draft > 0)
            for (int keep = 1; keep < MAX_NT; ++keep) {
                cudaGraph_t graph;
                CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
                record_restore(gi, keep);
                CUDA_CHECK(cudaStreamEndCapture(d.stream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&d.g_restore[keep], graph, 0));
                CUDA_CHECK(cudaGraphDestroy(graph));
            }
    }
    graphs_ready_ = true;
}

void Engine5::embed(const int * tokens, int nt) {
    const GTensor & te = gguf_->need("token_embd.weight");
    const auto * te_tr = ggml_get_type_traits((ggml_type) te.type);
    for (int t = 0; t < nt; ++t) {
        const uint8_t * row = te.data + (size_t) tokens[t] * te.row_bytes();
        float * out = h_embd_ + (size_t) t * cfg_.n_embd;
        if (te.type == GType::F32) memcpy(out, row, cfg_.n_embd * sizeof(float));
        else te_tr->to_float(row, out, cfg_.n_embd);
    }
}

void Engine5::run(int nt) {
    const bool bulk = nt > MAX_NT;
    if (!graphs_ready_ && !debug_) {
        build_graphs();
    }
    ++fwd_counter_;
    if (!(bulk && opt_.stream_experts && nt >= stream_min_)) {
        std::vector<int> slots;
        for (int i = 0; i < cfg_.n_layer; ++i) if (is_moe(i)) slots.push_back(i);
        cpu_->expect(fwd_counter_, slots, bulk);
    }
    if (bulk) {
        std::vector<std::thread> th;
        std::vector<std::string> err(devs_.size());
        for (int gi = 0; gi < (int) devs_.size(); ++gi)
            th.emplace_back([&, gi] {
                try { CUDA_CHECK(cudaSetDevice(devs_[gi]->id)); record_main(gi, nt); }
                catch (const std::exception & ex) { err[gi] = ex.what(); }
            });
        for (auto & t : th) t.join();
        for (auto & m : err) if (!m.empty()) throw std::runtime_error(m);
    } else {
        for (int gi = 0; gi < (int) devs_.size(); ++gi) {
            auto & dp = devs_[gi];
            CUDA_CHECK(cudaSetDevice(dp->id));
            if (debug_) { record_main(gi, nt); continue; }
            CUDA_CHECK(cudaGraphLaunch(dp->g_main[nt], dp->stream));
        }
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
                fprintf(stderr, "hyper5: forward %u stuck on device %d; cpu: %s\n", fwd_counter_, dp->id, cpu_->state().c_str());
            }
            std::this_thread::yield();
        }
    }
}

std::vector<int> Engine5::forward(const int * tokens, int nt, int pos) {
    if (nt < 1 || nt > R5) throw std::runtime_error("forward: bad token count");
    if (pos + nt > opt_.max_pos) throw std::runtime_error("forward: position exceeds max_pos");
    const bool bulk = nt > MAX_NT;
    embed(tokens, nt);
    h_pos_[0] = pos;
    if (adapt_ && !bulk) {   // placement follows the routing: after a prompt, then every 32 decode steps
        if (prompt_routed_) { rebalance(1 << 20); prompt_routed_ = false; steps_ = 0; }
        else if (++steps_ % adapt_every_ == 0) rebalance(adapt_budget_);
    }
    if (bulk) for (int il = 0; il < cfg_.n_layer; ++il) if (is_moe(il) && ehost_[il].stream_dirty) rebuild_stream(il);
    run(nt);
    if (adapt_ && bulk) {
        const int K = cfg_.n_expert_used;
        for (int il = 0; il < cfg_.n_layer; ++il) {
            if (!is_moe(il)) continue;
            auto & sc = ehost_[il].score;
            for (int i = 0; i < nt * K; ++i) { const int e = h_ids_[(size_t) il * R5 * K + i]; if (e >= 0 && e < cfg_.n_expert) sc[e] += prompt_weight_; }
        }
        prompt_routed_ = true;
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

// ---------------- prompt cache: KDA conv + state, the open indexer pool's cells ----------------
size_t Engine5::snap_floats(int gi) const {
    const Glm5Config & c = cfg_;
    size_t n = 0;
    for (auto & L : devs_[gi]->layers)
        n += L.mla ? 8 * GIDX_DIM : (size_t) (c.conv - 1) * 3 * L.nh * 128 + (size_t) L.nh * 128 * 128;
    return n;
}

void Engine5::snap_copy(Snap & sn, bool to_host) {
    const Glm5Config & c = cfg_;
    for (size_t gi = 0; gi < devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        float * hp = sn.h[gi];
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
    }
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->stream)); }
}

void Engine5::take_snapshot(int pos) {
    for (auto & s : snaps_) if (s.pos == pos) return;
    if ((int) snaps_.size() >= opt_.max_snapshots) {
        auto victim = snaps_.end();
        for (auto it = snaps_.begin(); it != snaps_.end(); ++it)
            if (it->pos % 4096 && (victim == snaps_.end() || it->pos < victim->pos)) victim = it;
        if (victim == snaps_.end())
            victim = std::min_element(snaps_.begin(), snaps_.end(), [](const Snap & a, const Snap & b) { return a.pos < b.pos; });
        snap_pool_.push_back(victim->h);
        snaps_.erase(victim);
    }
    Snap sn;
    sn.pos = pos;
    if (!snap_pool_.empty()) { sn.h = snap_pool_.back(); snap_pool_.pop_back(); }
    else {
        for (size_t gi = 0; gi < devs_.size(); ++gi) {
            float * p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, snap_floats((int) gi) * sizeof(float), cudaHostAllocPortable));
            sn.h.push_back(p);
        }
    }
    snap_copy(sn, true);
    snaps_.push_back(sn);
}

int Engine5::sample_row(int t, const SamplingParams & sp) {
    std::vector<std::pair<float, int>> cand;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float * p = h_topk_ + ((size_t) g * MAX_NT + t) * TOPK * 2;
        for (int i = 0; i < TOPK; ++i) { const int idx = ((const int *) p)[2 * i + 1]; if (idx >= 0) cand.push_back({p[2 * i], idx}); }
    }
    std::sort(cand.begin(), cand.end(), [](auto & a, auto & b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    int k = std::min<int>((int) cand.size(), sp.top_k > 0 ? std::min(sp.top_k, TOPK) : TOPK);
    if (sp.temp <= 0.0f || k == 1) return cand[0].second;
    std::vector<double> pr(k);
    double z = 0;
    for (int i = 0; i < k; ++i) { pr[i] = std::exp((cand[i].first - cand[0].first) / sp.temp); z += pr[i]; }
    for (auto & v : pr) v /= z;
    if (sp.min_p > 0) { int kk = 1; while (kk < k && pr[kk] >= sp.min_p * pr[0]) ++kk; k = kk; }
    if (sp.top_p < 1.0f) { double cum = 0; int kk = 0; while (kk < k) { cum += pr[kk++]; if (cum >= sp.top_p) break; } k = kk; }
    double tot = 0;
    for (int i = 0; i < k; ++i) tot += pr[i];
    double u = std::uniform_real_distribution<double>(0.0, tot)(rng_);
    for (int i = 0; i < k; ++i) { u -= pr[i]; if (u <= 0) return cand[i].second; }
    return cand[k - 1].second;
}

std::vector<int> Engine5::generate(const std::vector<int> & prompt, int n_gen, bool spec_req, GenStats * stats,
                                   const std::function<bool(int)> & on_token, const SamplingParams & sp) {
    if (prompt.empty()) throw std::runtime_error("generate: empty prompt");
    if ((int) prompt.size() + 8 > opt_.max_pos) throw std::runtime_error("generate: prompt longer than the context");
    using clk = std::chrono::steady_clock;
    const bool sampling = sp.temp > 0.0f;
    rng_.seed(sp.seed ? sp.seed : std::random_device{}());
    const int P = (int) prompt.size();
    int s = 0;
    if (opt_.prompt_cache) {
        int L = 0;
        while (L < P && L < (int) hist_.size() && prompt[L] == hist_[L]) ++L;
        for (size_t i = 0; i < snaps_.size();)
            if (snaps_[i].pos > L) { snap_pool_.push_back(snaps_[i].h); snaps_.erase(snaps_.begin() + i); } else ++i;
        Snap * best = nullptr;
        for (auto & sn : snaps_) if (sn.pos <= std::min(L - 1, P - 1) && (!best || sn.pos > best->pos)) best = &sn;
        if (best) { s = best->pos; snap_copy(*best, false); }
    }
    if (s == 0) reset();
    hist_.assign(prompt.begin(), prompt.begin() + s);
    std::vector<int> snap_at;
    if (opt_.prompt_cache) {
        std::vector<int> msg;
        for (int q = s + 1; q < P; ++q)
            if (std::find(snap_tokens_.begin(), snap_tokens_.end(), prompt[q]) != snap_tokens_.end()) msg.push_back(q);
        for (int q = (s / 4096 + 1) * 4096; q < P; q += 4096) snap_at.push_back(q);
        int last = s;
        for (size_t i = 0; i < msg.size(); ++i)
            if (i + 2 >= msg.size() || msg[i] - last >= 256) { snap_at.push_back(msg[i]); last = msg[i]; }
        std::sort(snap_at.begin(), snap_at.end());
        snap_at.erase(std::unique(snap_at.begin(), snap_at.end()), snap_at.end());
    }
    GenStats st;
    auto tp = clk::now();
    int next = -1;
    size_t si = 0;
    for (int c0 = s; c0 < P;) {
        int end = std::min(c0 + R5, P);
        while (si < snap_at.size() && snap_at[si] <= c0) ++si;
        if (si < snap_at.size() && snap_at[si] < end) end = snap_at[si];
        const int len = end - c0;
        next = forward(&prompt[c0], len, c0)[len - 1];
        if (sampling) next = sample_row(len <= MAX_NT ? len - 1 : 0, sp);
        if (si < snap_at.size() && snap_at[si] == end) take_snapshot(end);
        if (prefill_cb_) prefill_cb_(end, P, s);
        c0 = end;
    }
    hist_ = prompt;
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
    auto since = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    // generated tokens get recurrent-state snapshots too (every 1024 positions and at the end): the next request repeats this
    // answer in its prompt and resumes close to where the re-rendered history first differs
    int snap_mark = p / 1024;
    bool state_ok = true;
    auto gen_snapshot = [&] {
        if (!opt_.prompt_cache || p / 1024 == snap_mark) return;
        snap_mark = p / 1024;
        take_snapshot(p);
    };
    const int K = std::min(opt_.n_draft, MAX_NT - 1);
    if (K <= 0 || !spec_req) {
        while (emit(next) && p + 1 < opt_.max_pos) {
            next = forward(&next, 1, p++)[0];
            if (sampling) next = sample_row(0, sp);
            st.steps++;
            gen_snapshot();
        }
    } else {
        // prompt-lookup speculation: drafts = the tokens that followed the latest earlier occurrence of the last NG tokens
        // (history = prompt + output); a draft is kept iff the token sampled (or argmax) at its row equals it: exact
        constexpr int NG = 3;
        std::vector<int> hist(prompt.begin(), prompt.end());
        std::unordered_map<uint64_t, int> last;   // n-gram -> position after its latest occurrence
        auto key = [&](size_t end) { uint64_t h = 1469598103934665603ull; for (size_t i = end - NG; i < end; ++i) h = (h ^ (uint32_t) hist[i]) * 1099511628211ull; return h; };
        size_t indexed = NG;
        auto index_upto = [&](size_t n) { for (; indexed <= n; ++indexed) last[key(indexed)] = (int) indexed; };
        int cur = next;
        std::vector<int> in(K + 1);
        while (!stop && p + K + 1 < opt_.max_pos) {
            hist.push_back(cur);
            int nd = 0;
            if (hist.size() > NG) {
                auto it = last.find(key(hist.size()));
                index_upto(hist.size() - 1);   // (the current suffix itself is indexed after the lookup)
                if (it != last.end()) {
                    // match length (backwards, up to 32) decides how far to trust the continuation; a poor recent
                    // acceptance rate raises the bar (a rejected verification costs more than a plain step)
                    const int e0 = it->second, e1 = (int) hist.size();
                    int ml = 0;
                    while (ml < 32 && e0 - 1 - ml >= 0 && hist[e0 - 1 - ml] == hist[e1 - 1 - ml]) ++ml;
                    const int need = acc_rate_ < 0.35 ? 8 : 5;
                    const int kk = ml >= need + 6 ? K : ml >= need + 2 ? std::min(K, 2) : ml >= need ? 1 : 0;
                    for (int j = e0; j < e1 && nd < kk; ++j) in[1 + nd++] = hist[j];
                }
            }
            in[0] = cur;
            auto ta = clk::now();
            std::vector<int> a = forward(in.data(), nd + 1, p);
            st.t_main += since(ta);
            st.steps++;
            int m = 0;
            if (sampling) { while (m < nd && (a[m] = sample_row(m, sp)) == in[1 + m]) ++m; if (m == nd) a[nd] = sample_row(nd, sp); }
            else while (m < nd && a[m] == in[1 + m]) ++m;
            st.accepted += m;
            if (nd > 0) acc_rate_ = 0.9 * acc_rate_ + 0.1 * ((double) m / nd);
            if (emit(cur)) for (int j = 0; j < m; ++j) { hist.push_back(in[1 + j]); if (!emit(in[1 + j])) break; }
            if (stop) { state_ok = false; break; }   // (the state holds rows past the end of the output)
            if (m < nd) { ta = clk::now(); restore(m + 1); st.t_restore += since(ta); }
            cur = a[m];
            p += m + 1;
            gen_snapshot();
        }
    }
    if (opt_.prompt_cache && p > P && state_ok) take_snapshot(p);   // state after the last processed token
    {
        std::vector<int> sq = prompt;
        sq.insert(sq.end(), out.begin(), out.end());
        sq.resize(std::min<size_t>(sq.size(), (size_t) p));
        hist_ = sq;
    }
    st.tokens = (int) out.size();
    st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

void Engine5::save_expert_stats(const std::string & path) {
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

int Engine5::prefill(const int * tokens, int n, int pos) {
    int next = -1;
    static const int chunk = getenv("HYPER4_CHUNK") ? std::max(1, std::min(R5, atoi(getenv("HYPER4_CHUNK")))) : R5;
    for (int c0 = 0; c0 < n;) {
        const int len = std::min(chunk, n - c0);
        next = forward(tokens + c0, len, pos + c0)[len - 1];
        c0 += len;
    }
    return next;
}

void Engine5::get_logits(int t, std::vector<float> & out) {
    out.resize(cfg_.n_vocab);
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaMemcpy(out.data() + dp->vocab_off, dp->logits + (size_t) t * dp->output.n(), (size_t) dp->output.n() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

} // namespace hyper
