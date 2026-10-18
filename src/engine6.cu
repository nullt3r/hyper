#include "engine6.h"
#include "sampling.h"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/mman.h>

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

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace {

constexpr int CPU_OWNER = 3;
constexpr int R6 = MOE_BULK_ROWS;   // activation rows (prefill chunk)

// dense weight: Q8_0 (fragment-ordered int8), Q4_K / Q6_K as KQ (same bits, the LM head), or anything else dequantized to
// fp16 at load
struct DW {
    Q8W q8;
    BF16W f;
    bool f16 = false, kq = false;
    KQW kw;
    int n() const { return kq ? kw.n : f16 ? f.n : q8.n; }
    int k() const { return kq ? kw.k : f16 ? f.k : q8.k; }
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

struct Engine6::DevLayer {
    bool moe = false, swa = false;
    float * attn_norm = nullptr, * ffn_norm = nullptr;
    int nq = 0, nkv = 0, q0 = 0;   // local q heads, local kv heads, first local q head
    DW wqkv, wo;                    // rows [q local | k local | v local]; output projection: the local heads' columns
    float * sinks = nullptr;        // local q heads (sliding-window layers)
    half * kc = nullptr, * vc = nullptr;
    size_t kpitch = 0, vpitch = 0;  // elements per kv head
    int ring = 0;                   // 0: full cache (row = position), else ring rows
    // dense FFN (layer 0): local hidden slice
    DW gu, down;
    int ff_l = 0;
    float * router = nullptr, * exp_bias = nullptr;
    uint16_t * router16 = nullptr;   // the router as bf16, when every value is one exactly
    MoeDev moex;
    VmmBuf * xg = nullptr, * xu = nullptr, * xd = nullptr;   // elastic: expert arrays with room for extra slots
    int nbase = 0;
    size_t xbase = 0;
    std::vector<int> xe;
    int * owner = nullptr;
    int * st_slot[2] = {}, * owner_bulk = nullptr;
    std::vector<int> st_list[2];
};

struct Engine6::Device {
    int id = 0, g = 0;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t g_main[MAX_NT + 1] = {};
    std::vector<DevLayer> layers;
    // MTP: the NextN blocks (SWA attention + dense FFN), eh_proj on [enorm(embedding) | hnorm(hidden)], head norms; main hidden
    // ring, pass rows (hidden in, residual), chained outputs, pass positions [first, last]
    std::vector<DevLayer> mtp;
    std::vector<DW> m_eh;
    std::vector<float *> m_enorm, m_hnorm, m_onorm;
    float * hring = nullptr, * mh = nullptr, * mres = nullptr, * mchain = nullptr;
    int * mpos = nullptr;
    float * out_norm = nullptr;
    DW output;
    int vocab_off = 0;
    // activations [R6] rows
    float * res = nullptr, * xn = nullptr, * bo = nullptr, * part = nullptr;
    std::vector<float *> res_x;      // residual rows of the further chunks of a multi-chunk prefill pass
    int * pos_x = nullptr;
    float * big0 = nullptr, * o = nullptr, * attn_part = nullptr;
    size_t attn_part_n = 0;
    float * router32 = nullptr;
    float * rlog = nullptr, * wts = nullptr, * sg = nullptr, * shgu = nullptr, * shh = nullptr, * shpart = nullptr;
    unsigned * ticket = nullptr;
    float * hexp = nullptr, * yexp = nullptr, * logits = nullptr;
    int * ids = nullptr, * pos = nullptr, * counter = nullptr, * order = nullptr, * order_n = nullptr, * egrp = nullptr;
    half * xh = nullptr, * p16 = nullptr, * recv = nullptr, * mix16 = nullptr, * h16 = nullptr;
    float * topk = nullptr;
    float * lstat = nullptr;
    int big_stride = 0, o_stride = 0;
    cublasHandle_t blas = nullptr;
    cudaStream_t cstream = nullptr;
    cudaEvent_t ev_up[2] = {}, ev_free[2] = {};
    std::deque<std::array<int, 3>> up_pend;
    cudaEvent_t ev_arh = nullptr;
    uint8_t * stage[2] = {};
    size_t stage_bytes = 0;
    cudaEvent_t ev_ar[2] = {};
    size_t used = 0;
    std::vector<void *> allocs;
    struct KvSeg { VmmBuf * b; size_t pitch, row_bytes; int heads, div; };   // elastic caches
    std::vector<KvSeg> kvs;
    size_t lend_limit = 0;
    std::vector<std::unique_ptr<VmmBuf>> vmm;
    cudaStream_t fstream = nullptr;

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
        for (auto & ev : ev_ar) if (ev) cudaEventDestroy(ev);
        if (blas) cublasDestroy(blas);
        for (auto & ev : ev_up) if (ev) cudaEventDestroy(ev);
        for (auto & ev : ev_free) if (ev) cudaEventDestroy(ev);
        if (ev_arh) cudaEventDestroy(ev_arh);
        if (cstream) cudaStreamDestroy(cstream);
        if (fstream) cudaStreamDestroy(fstream);
        vmm.clear();
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

namespace {
size_t kv_rows5(int div, int n) { return div == 1 ? (size_t) n : (size_t) (n / div + 1); }
// local kv heads of layer il on GPU g: global layers rotate which GPUs get the larger shares (their caches are the big ones)
std::pair<int, int> kv_range(const MimoConfig & c, int il, int nd, int g) {
    int rot = 0;
    if (!c.swa[il]) for (int j = 0; j < il; ++j) rot += !c.swa[j];
    const int gv = (g + rot) % nd, n = c.n_kv[il];
    return {n * gv / nd, n * (gv + 1) / nd};
}
}

Engine6::Engine6(const std::string & model_path, const Engine6Options & opt) : opt_(opt) {
    debug_ = getenv("HYPER4_DEBUG") != nullptr;
    nocpu_ = getenv("HYPER4_NOCPU") != nullptr;
    gguf_ = std::make_unique<GGUF>(model_path);
    cfg_ = MimoConfig::from_gguf(*gguf_);
    fprintf(stderr, "hyper6: %s\n", cfg_.describe().c_str());
    if (cfg_.n_expert_used > MOE_MAX_USED) throw std::runtime_error("hyper6: too many experts per token");
    if (!opt_.mtp_path.empty() && opt_.n_draft > 0) {
        mtp_g_ = std::make_unique<GGUF>(opt_.mtp_path);
        const MimoConfig mc = MimoConfig::from_gguf(*mtp_g_);
        const int L = cfg_.n_layer;
        for (int h = 0; h < mc.n_nextn && h < MAX_NT - 1; ++h)
            if (mtp_g_->tensor("blk." + std::to_string(L + h) + ".nextn.eh_proj.weight") && L + h < (int) mc.n_kv.size()) n_mtp_ = h + 1;
        if (!n_mtp_) throw std::runtime_error("hyper6: " + opt_.mtp_path + " has no NextN block blk." + std::to_string(L));
        cfg_.n_kv.resize(L + n_mtp_); cfg_.swa.resize(L + n_mtp_); cfg_.moe.resize(L + n_mtp_);
        for (int h = 0; h < n_mtp_; ++h) {
            cfg_.n_kv[L + h] = mc.n_kv[L + h]; cfg_.swa[L + h] = mc.swa[L + h]; cfg_.moe[L + h] = 0;
            if (!mc.swa[L + h]) throw std::runtime_error("hyper6: global-attention NextN blocks not supported");
        }
        if (getenv("HYPER6_MTP_PMIN")) mtp_pmin_ = atof(getenv("HYPER6_MTP_PMIN"));
        fprintf(stderr, "hyper6: MTP: %d NextN heads from %s\n", n_mtp_, opt_.mtp_path.c_str());
    }
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    opt_.n_devices = std::min(opt_.n_devices, ndev);
    if (opt_.max_pos % 4) opt_.max_pos += 4 - opt_.max_pos % 4;
    while (ring_ < R6 + cfg_.window + 2 * MAX_NT) ring_ *= 2;
    if (getenv("HYPER6_NS")) ns_global_ = std::max(1, atoi(getenv("HYPER6_NS")));
    for (int g = 0; g < opt_.n_devices; ++g) {
        auto dev = std::make_unique<Device>();
        dev->id = g; dev->g = g;
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->stream, cudaStreamNonBlocking));
        gemv_init(g);
        for (auto & ev : dev->ev_ar) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->cstream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->fstream, cudaStreamNonBlocking));
        for (auto & ev : dev->ev_up) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        for (auto & ev : dev->ev_free) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev->ev_arh, cudaEventDisableTiming));
        if (cublasCreate(&dev->blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        cublasSetStream(dev->blas, dev->stream);
        cublasSetMathMode(dev->blas, CUBLAS_DEFAULT_MATH);
        devs_.push_back(std::move(dev));
    }
    const int nd = opt_.n_devices, n = cfg_.n_embd;
    if (getenv("HYPER6_MC")) mc_max_ = std::max(1, std::min(8, atoi(getenv("HYPER6_MC"))));
    CUDA_CHECK(cudaHostAlloc(&h_embd_, (size_t) mc_max_ * R6 * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_stage_, (size_t) 2 * nd * R6 * n * sizeof(half), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_, sizeof(CpuMoeBulk), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_out_, sizeof(CpuMoeBulkOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_bulk_, 0, sizeof(CpuMoeBulk));
    memset((void *) cpu_bulk_out_, 0, sizeof(CpuMoeBulkOut));
    barrier_ = std::make_unique<Barrier4>(nd);
    CUDA_CHECK(cudaHostAlloc(&h_pos_, 16 * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_res_, (size_t) nd * MAX_NT * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_topk_, (size_t) nd * MAX_NT * TOPK * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_mtopk_, (size_t) nd * TOPK * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_ids_, (size_t) nl_tot() * mc_max_ * R6 * cfg_.n_expert_used * sizeof(int), cudaHostAllocPortable));
    const size_t ll = ll_slots(nd, MAX_NT, n);
    CUDA_CHECK(cudaHostAlloc(&ar_ll_, ll * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_ll_, 0xff, ll * sizeof(uint2));
    const int slots = nl_tot();
    CUDA_CHECK(cudaHostAlloc(&cpu_rec_, (size_t) slots * sizeof(CpuMoeRec), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_out_, (size_t) slots * sizeof(CpuMoeOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_rec_, 0, (size_t) slots * sizeof(CpuMoeRec));
    memset((void *) cpu_out_, 0, (size_t) slots * sizeof(CpuMoeOut));
    if (const char * sp = getenv("HYPER4_STATS")) {
        FILE * f = fopen(sp, "rb");
        if (f) {
            int nl = 0, w = 0;
            if (fread(&nl, 4, 1, f) == 1 && fread(&w, 4, 1, f) == 1 && nl >= cfg_.n_layer && nl <= cfg_.n_layer + 3) {
                stats_.assign(nl, std::vector<uint64_t>(w));
                for (auto & s : stats_) if (fread(s.data(), 8, w, f) != (size_t) w) { stats_.clear(); break; }
            }
            fclose(f);
            fprintf(stderr, "hyper6: expert placement from %s (%s)\n", sp, stats_.empty() ? "unreadable, ignored" : "ok");
        }
    }
    cpu_ = std::make_unique<CpuMoe>(opt_.cpu_threads, n, cfg_.n_ff_exp, cfg_.n_expert_used, cpu_rec_, cpu_out_, slots, cpu_bulk_,
                                    cpu_bulk_out_);
    cpu_->set_clamp(0.0f);
    adapt_ = !getenv("HYPER6_ADAPT") || atoi(getenv("HYPER6_ADAPT")) != 0;
    if (getenv("HYPER6_STREAM_MIN")) stream_min_ = atoi(getenv("HYPER6_STREAM_MIN"));
    if (getenv("HYPER6_UP_PARTS")) up_parts_ = std::max(0, atoi(getenv("HYPER6_UP_PARTS")));
    if (getenv("HYPER6_DECAY")) adapt_decay_ = atof(getenv("HYPER6_DECAY"));
    if (getenv("HYPER6_PROMPT_W")) prompt_weight_ = atof(getenv("HYPER6_PROMPT_W"));
    if (getenv("HYPER6_ADAPT_EVERY")) adapt_every_ = std::max(1, atoi(getenv("HYPER6_ADAPT_EVERY")));
    if (getenv("HYPER6_ADAPT_BUDGET")) adapt_budget_ = atoi(getenv("HYPER6_ADAPT_BUDGET"));
    if (getenv("HYPER6_ADAPT_MIN")) adapt_min_ = atof(getenv("HYPER6_ADAPT_MIN"));
    if (getenv("HYPER6_ADAPT_RATIO")) adapt_ratio_ = atof(getenv("HYPER6_ADAPT_RATIO"));
    elastic_ = adapt_ && (!getenv("HYPER6_ELASTIC") || atoi(getenv("HYPER6_ELASTIC")) != 0);
    elastic_log_ = getenv("HYPER6_ELASTIC_LOG") != nullptr;
    if (getenv("HYPER6_ELASTIC_MIN")) elastic_min_ = atof(getenv("HYPER6_ELASTIC_MIN"));
    if (getenv("HYPER6_KV_STEP")) kv_step_ = std::max(64, atoi(getenv("HYPER6_KV_STEP")));
    park_.before_restore = [this](int n) { ensure_kv(n + 2 * MAX_NT); };
    ehost_.resize(nl_tot());
    for (auto & H : ehost_) { H.score.assign(cfg_.n_expert, 0.0); H.last_count.assign(1024, 0); H.pend.assign(cfg_.n_expert, 0); }
    load_weights();
}

Engine6::~Engine6() {
    if (fill_th_.joinable()) { fill_cancel_.store(true); fill_th_.join(); }
    for (auto & sn : snaps_) snap_pool_.push_back(sn.h);
    park_.release(snap_pool_);
    for (auto & v : snap_pool_) for (float * p : v) cudaFreeHost(p);
    if (h_topk_) cudaFreeHost(h_topk_);
    if (h_mtopk_) cudaFreeHost(h_mtopk_);
    if (h_ids_) cudaFreeHost(h_ids_);
    cpu_.reset();
    for (auto & [p, sz] : host_bufs_) munmap(p, sz);
    devs_.clear();
    for (void * p : {(void *) h_embd_, (void *) h_pos_, (void *) h_res_, (void *) ar_ll_, (void *) cpu_rec_, (void *) cpu_out_,
                     (void *) h_stage_, (void *) cpu_bulk_, (void *) cpu_bulk_out_})
        if (p) cudaFreeHost(p);
}

void Engine6::load_layer(Device & dev, DevLayer & L, int il) {
    const MimoConfig & c = cfg_;
    const int nd = opt_.n_devices, g = dev.g, n = c.n_embd;
    const GGUF & G = il >= c.n_layer ? *mtp_g_ : *gguf_;
    auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
    auto T = [&](const std::string & name) { return &G.need(name); };
    auto f32 = [&](const std::string & name) {
        const std::vector<float> v = to_f32(G.need(name));
        return dev.upload(v.data(), v.size());
    };
    const std::string p = "blk." + std::to_string(il) + ".";
    L.moe = is_moe(il);
    L.swa = c.swa[il] != 0;
    L.attn_norm = f32(p + "attn_norm.weight");
    L.ffn_norm = f32(p + "ffn_norm.weight");
    // attention: local kv heads and their q heads
    const int nkv_t = c.n_kv[il], group = c.n_head / nkv_t;
    auto [kv0, kv1] = kv_range(c, il, nd, g);
    L.nkv = kv1 - kv0;
    L.q0 = kv0 * group;
    L.nq = L.nkv * group;
    const int64_t qrows = (int64_t) c.n_head * c.dk, krows = (int64_t) nkv_t * c.dk;
    const GTensor * qkv = T(p + "attn_qkv.weight");
    if (qkv->ne[1] != qrows + krows + (int64_t) nkv_t * c.dv) throw std::runtime_error("mimo2: unexpected attn_qkv rows");
    L.wqkv = upload_dense(A, dev.id, {{qkv, (int64_t) L.q0 * c.dk, (int64_t) (L.q0 + L.nq) * c.dk},
                                      {qkv, qrows + (int64_t) kv0 * c.dk, qrows + (int64_t) kv1 * c.dk},
                                      {qkv, qrows + krows + (int64_t) kv0 * c.dv, qrows + krows + (int64_t) kv1 * c.dv}});
    L.wo = upload_dense(A, dev.id, {{T(p + "attn_output.weight"), 0, n}}, {{(int64_t) L.q0 * c.dv / 32, (int64_t) (L.q0 + L.nq) * c.dv / 32}});
    if (const GTensor * sk = G.tensor(p + "attn_sinks.weight")) {
        const std::vector<float> v = to_f32(*sk);
        L.sinks = dev.upload(v.data() + L.q0, (size_t) L.nq);
    }
    if (L.swa) {
        L.ring = ring_;
        L.kpitch = (size_t) ring_ * c.dk;
        L.vpitch = (size_t) ring_ * c.dv;
        L.kc = dev.alloc<half>(std::max(1, L.nkv) * L.kpitch);
        L.vc = dev.alloc<half>(std::max(1, L.nkv) * L.vpitch);
    } else {
        L.ring = 0;
        L.kpitch = (size_t) opt_.max_pos * c.dk;
        L.vpitch = (size_t) opt_.max_pos * c.dv;
        L.kc = kv_alloc<half>(dev, L.nkv, L.kpitch, c.dk, 1);
        L.vc = kv_alloc<half>(dev, L.nkv, L.vpitch, c.dv, 1);
    }
    if (!L.moe) {   // dense FFN: hidden slice
        auto [f0, f1] = split(c.n_ff, nd, g, 32);
        L.ff_l = (int) (f1 - f0);
        L.gu = upload_dense(A, dev.id, {{T(p + "ffn_gate.weight"), f0, f1}, {T(p + "ffn_up.weight"), f0, f1}});
        L.down = upload_dense(A, dev.id, {{T(p + "ffn_down.weight"), 0, n}}, {{f0 / 32, f1 / 32}});
    } else {
        const std::vector<float> rw = to_f32(G.need(p + "ffn_gate_inp.weight"));
        static const bool r32 = getenv("HYPER6_ROUTER32") != nullptr;
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
            if (!r32 && dev.g == 0 && il == 1) fprintf(stderr, "hyper6: routers stay fp32 (not exactly bf16)\n");
        }
        L.exp_bias = f32(p + "exp_probs_b.bias");
    }
}

void * Engine6::host_huge_alloc(size_t bytes) {
    const size_t H = 2u << 20, sz = (bytes + H - 1) / H * H;
    void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("host_huge_alloc: mmap failed");
    madvise(p, sz, MADV_HUGEPAGE);
    host_bufs_.push_back({p, sz});
    return p;
}

template <typename T> T * Engine6::kv_alloc(Device & dev, int heads, size_t pitch_elems, size_t row_elems, int div) {
    if (!elastic_ || heads <= 0) return dev.alloc<T>((size_t) std::max(heads, 1) * pitch_elems);
    CUDA_CHECK(cudaSetDevice(dev.id));
    auto b = std::make_unique<VmmBuf>();
    b->reserve(dev.id, (size_t) heads * pitch_elems * sizeof(T));
    dev.kvs.push_back({b.get(), pitch_elems * sizeof(T), row_elems * sizeof(T), heads, div});
    dev.lend_limit += b->reserved();
    dev.used += b->reserved();
    T * p = (T *) b->ptr();
    dev.vmm.push_back(std::move(b));
    return p;
}

size_t Engine6::lend_free(const Device & d) const {
    size_t used = 0;
    for (auto & k : d.kvs) used += k.b->mapped();
    for (const DevLayer & L : d.layers)
        if (L.xg) used += L.xg->mapped() + L.xu->mapped() + L.xd->mapped() - L.xbase;
    return d.lend_limit > used ? d.lend_limit - used : 0;
}

void Engine6::ensure_kv(int need) {
    if (!elastic_ || need <= kv_pos_) return;
    const int target = std::min(opt_.max_pos, (need + kv_step_ - 1) / kv_step_ * kv_step_);
    fill_wait(true);
    cpu_->drain();
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaDeviceSynchronize()); }
    for (auto & dp : devs_) {
        Device & d = *dp;
        if (d.kvs.empty()) continue;
        size_t add = 0;
        for (auto & k : d.kvs)
            for (int h = 0; h < k.heads; ++h) add += k.b->cost(h * k.pitch, std::min(kv_rows5(k.div, target) * k.row_bytes, k.pitch));
        const size_t fr = lend_free(d);
        if (add > fr) evict_extra(d, add - fr);
        CUDA_CHECK(cudaSetDevice(d.id));
        for (auto & k : d.kvs)
            for (int h = 0; h < k.heads; ++h) k.b->map(h * k.pitch, std::min(kv_rows5(k.div, target) * k.row_bytes, k.pitch), true, d.stream);
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
    }
    if (elastic_log_) {
        fprintf(stderr, "hyper6: elastic: caches mapped to %d positions; lendable free", target);
        for (auto & dp : devs_) if (!dp->kvs.empty()) fprintf(stderr, " %.0f MiB", lend_free(*dp) / 1048576.0);
        fprintf(stderr, "\n");
    }
    kv_pos_ = target;
}

void Engine6::shrink_kv(int keep) {
    if (!elastic_) return;
    const int target = std::min(opt_.max_pos, (keep + kv_step_ - 1) / kv_step_ * kv_step_);
    if (target + 2 * kv_step_ > kv_pos_) return;
    fill_wait(true);
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaDeviceSynchronize()); }
    size_t freed = 0;
    for (auto & dp : devs_)
        for (auto & k : dp->kvs)
            for (int h = 0; h < k.heads; ++h) freed += k.b->unmap(h * k.pitch + kv_rows5(k.div, target) * k.row_bytes, (h + 1) * k.pitch);
    if (elastic_log_) fprintf(stderr, "hyper6: elastic: caches %d -> %d positions (%.0f MiB to lend)\n", kv_pos_, target, freed / 1048576.0);
    kv_pos_ = target;
}

void Engine6::evict_extra(Device & d, size_t bytes) {
    const int g = d.g, E = cfg_.n_expert;
    size_t freed = 0;
    int n = 0;
    while (freed < bytes) {
        int best = -1;
        double bs = 0;
        for (int il = 0; il < nl_tot(); ++il) {
            DevLayer & L = d.layers[il];
            if (!L.xg || L.xe.empty()) continue;
            const int e = L.xe.back();
            const double sc = e < 0 ? -1.0 : ehost_[il].score[e];
            if (best < 0 || sc < bs) { best = il; bs = sc; }
        }
        if (best < 0) break;
        DevLayer & L = d.layers[best];
        ExpertHost & H = ehost_[best];
        const int e = L.xe.back(), k = L.nbase + (int) L.xe.size() - 1;
        if (e >= 0) {
            H.owner[e] = CPU_OWNER;
            H.slot[g][e] = -1;
            cpu_->set_owned(best, e, true);
            H.tables_dirty = H.stream_dirty = true;
            ++n;
        }
        L.xe.pop_back();
        freed += L.xg->unmap((size_t) k * H.gb, L.xg->reserved()) + L.xu->unmap((size_t) k * H.gb, L.xu->reserved()) +
                 L.xd->unmap((size_t) k * H.db, L.xd->reserved());
    }
    for (int il = 0; il < nl_tot(); ++il) {
        ExpertHost & H = ehost_[il];
        if (!is_moe(il) || !H.tables_dirty) continue;
        for (auto & dp : devs_) {
            DevLayer & L2 = dp->layers[il];
            CUDA_CHECK(cudaSetDevice(dp->id));
            CUDA_CHECK(cudaMemcpy((void *) L2.moex.slot, H.slot[dp->g].data(), E * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(L2.owner, H.owner.data(), E * sizeof(int), cudaMemcpyHostToDevice));
        }
        H.tables_dirty = false;
    }
    if (elastic_log_) fprintf(stderr, "hyper6: elastic: device %d gave back %d experts (%.0f MiB)\n", d.id, n, freed / 1048576.0);
}

int Engine6::fill_extra(int budget) {
    if (!elastic_ || budget <= 0 || fill_busy_.load() || !fill_list_.empty()) return 0;
    const int E = cfg_.n_expert, nd = (int) devs_.size();
    std::vector<size_t> fr(nd, 0), headroom(nd, 0);
    bool any = false;
    for (auto & dp : devs_) {
        if (dp->kvs.empty()) continue;
        fr[dp->g] = lend_free(*dp);
        for (auto & k : dp->kvs) headroom[dp->g] += (kv_rows5(k.div, kv_step_) * k.row_bytes + k.b->granule()) * k.heads;
        static const bool no_head = getenv("HYPER6_ELASTIC_NOHEADROOM") != nullptr;
        if (no_head) headroom[dp->g] = 0;
        any |= fr[dp->g] > headroom[dp->g];
    }
    if (!any) return 0;
    struct C { double s; int il, e; };
    std::vector<C> cand;
    for (int il = 0; il < nl_tot(); ++il) {
        if (!is_moe(il)) continue;
        const ExpertHost & H = ehost_[il];
        for (int e = 0; e < E; ++e) if (H.owner[e] == CPU_OWNER && H.score[e] >= elastic_min_) cand.push_back({H.score[e], il, e});
    }
    std::sort(cand.begin(), cand.end(), [](const C & a, const C & b) { return a.s > b.s; });
    for (const C & c : cand) {
        if ((int) fill_list_.size() >= budget) break;
        ExpertHost & H = ehost_[c.il];
        Device * best = nullptr;
        size_t bleft = 0, bcost = 0;
        for (auto & dp : devs_) {
            DevLayer & L = dp->layers[c.il];
            if (!L.xg) continue;
            const int k = L.nbase + (int) L.xe.size();
            if (k >= E) continue;
            const size_t cost = L.xg->cost((size_t) k * H.gb, H.gb) + L.xu->cost((size_t) k * H.gb, H.gb) + L.xd->cost((size_t) k * H.db, H.db);
            const int g = dp->g;
            if (fr[g] < cost + headroom[g]) continue;
            const size_t left = fr[g] - cost;
            if (!best || left > bleft) { best = dp.get(); bleft = left; bcost = cost; }
        }
        if (!best) continue;
        DevLayer & L = best->layers[c.il];
        const int k = L.nbase + (int) L.xe.size();
        L.xe.push_back(c.e);
        H.pend[c.e] = 1;
        fill_list_.push_back({c.il, c.e, best->g, k, false});
        fr[best->g] -= bcost;
    }
    if (fill_list_.empty()) return 0;
    fill_due_ = dec_steps_ + 16 * (int) ((fill_list_.size() + 63) / 64);   // (GLM experts ~4x larger than Flash-Next's)
    fill_busy_.store(true);
    fill_cancel_.store(false);
    if (fill_th_.joinable()) fill_th_.join();
    fill_th_ = std::thread([this] {
        try {
            for (Fill & f : fill_list_) {
                if (fill_cancel_.load()) break;
                Device & d = *devs_[f.g];
                DevLayer & L = d.layers[f.il];
                const ExpertHost & H = ehost_[f.il];
                CUDA_CHECK(cudaSetDevice(d.id));
                L.xg->map((size_t) f.k * H.gb, H.gb, false);
                L.xu->map((size_t) f.k * H.gb, H.gb, false);
                L.xd->map((size_t) f.k * H.db, H.db, false);
                CUDA_CHECK(cudaMemcpyAsync((uint8_t *) L.moex.gate + (size_t) f.k * H.gb, H.gate + (size_t) f.e * H.gb, H.gb, cudaMemcpyHostToDevice, d.fstream));
                CUDA_CHECK(cudaMemcpyAsync((uint8_t *) L.moex.up + (size_t) f.k * H.gb, H.up + (size_t) f.e * H.gb, H.gb, cudaMemcpyHostToDevice, d.fstream));
                CUDA_CHECK(cudaMemcpyAsync((uint8_t *) L.moex.down + (size_t) f.k * H.db, H.down + (size_t) f.e * H.db, H.db, cudaMemcpyHostToDevice, d.fstream));
                f.copied = true;
            }
            for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->fstream)); }
        } catch (const std::exception & ex) {
            fprintf(stderr, "hyper6: elastic fill failed: %s\n", ex.what());
            for (Fill & f : fill_list_) f.copied = false;
        }
        fill_busy_.store(false);
    });
    return 0;
}

void Engine6::fill_apply() {
    if (fill_list_.empty() || fill_busy_.load()) return;
    if (fill_th_.joinable()) fill_th_.join();
    if (fill_cancel_.load()) for (Fill & f : fill_list_) f.copied = false;
    cpu_->drain();
    const int E = cfg_.n_expert;
    int n = 0;
    std::vector<char> dirty(ehost_.size(), 0);
    for (const Fill & f : fill_list_) {
        ExpertHost & H = ehost_[f.il];
        DevLayer & L = devs_[f.g]->layers[f.il];
        H.pend[f.e] = 0;
        if (f.copied) {
            H.slot[f.g][f.e] = f.k;
            H.owner[f.e] = f.g;
            cpu_->set_owned(f.il, f.e, false);
            H.stream_dirty = true;
            dirty[f.il] = 1;
            ++n;
        } else {
            L.xe[f.k - L.nbase] = -1;
        }
    }
    // a cancelled batch: its reservations are the tops of their layers' extra regions -- they give their memory back
    for (const Fill & f : fill_list_) {
        if (f.copied) continue;
        Device & d = *devs_[f.g];
        DevLayer & L = d.layers[f.il];
        const ExpertHost & H = ehost_[f.il];
        CUDA_CHECK(cudaSetDevice(d.id));
        while (!L.xe.empty() && L.xe.back() < 0) {
            const int k = L.nbase + (int) L.xe.size() - 1;
            L.xe.pop_back();
            L.xg->unmap((size_t) k * H.gb, L.xg->reserved());
            L.xu->unmap((size_t) k * H.gb, L.xu->reserved());
            L.xd->unmap((size_t) k * H.db, L.xd->reserved());
        }
    }
    for (int il = 0; il < (int) ehost_.size(); ++il) {
        if (!dirty[il]) continue;
        const ExpertHost & H = ehost_[il];
        for (auto & dp : devs_) {
            DevLayer & L = dp->layers[il];
            CUDA_CHECK(cudaSetDevice(dp->id));
            CUDA_CHECK(cudaMemcpy((void *) L.moex.slot, H.slot[dp->g].data(), E * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(L.owner, H.owner.data(), E * sizeof(int), cudaMemcpyHostToDevice));
        }
    }
    if (elastic_log_) {
        int ex = 0, nm = 0;
        for (int il = 0; il < nl_tot(); ++il) if (is_moe(il)) ++nm;
        for (auto & dp : devs_) for (const DevLayer & L : dp->layers) for (int e : L.xe) ex += e >= 0;
        fprintf(stderr, "hyper6: elastic: %d experts moved to extra slots (%zu reserved; %d in extra slots, %.1f per MoE layer)\n", n,
                fill_list_.size(), ex, (double) ex / std::max(1, nm));
    }
    fill_list_.clear();
}

void Engine6::fill_wait(bool cancel) {
    if (fill_list_.empty()) return;
    if (cancel) fill_cancel_.store(true);
    if (fill_th_.joinable()) fill_th_.join();
    fill_apply();
}

void Engine6::load_experts(int il, const std::vector<int> & quota) {
    const MimoConfig & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd;
    const std::string p = "blk." + std::to_string(il) + ".";
    const GGUF & G = *gguf_;
    const GTensor & tg = G.need(p + "ffn_gate_exps.weight"), & tu = G.need(p + "ffn_up_exps.weight"),
                  & tdn = G.need(p + "ffn_down_exps.weight");
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
    collapse_huge(buf, bytes);
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
        uint8_t * dg, * du, * dd;
        if (elastic_ && dev.lend_limit > 0) {   // virtual ranges for every expert, the base slots mapped
            CUDA_CHECK(cudaSetDevice(dev.id));
            auto mk = [&](size_t per) {
                auto b = std::make_unique<VmmBuf>();
                b->reserve(dev.id, per * E);
                b->map(0, per * nl, false);
                VmmBuf * r = b.get();
                dev.vmm.push_back(std::move(b));
                dev.used += r->mapped();
                return r;
            };
            L.xg = mk(gb); L.xu = mk(gb); L.xd = mk(db);
            L.nbase = nl;
            L.xbase = L.xg->mapped() + L.xu->mapped() + L.xd->mapped();
            L.xe.clear();
            dg = (uint8_t *) L.xg->ptr(); du = (uint8_t *) L.xu->ptr(); dd = (uint8_t *) L.xd->ptr();
        } else {
            dg = dev.alloc<uint8_t>(gb * std::max(nl, 1)); du = dev.alloc<uint8_t>(gb * std::max(nl, 1));
            dd = dev.alloc<uint8_t>(db * std::max(nl, 1));
        }
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
        L.moex.clamp = 0.0f;
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
void Engine6::rebuild_stream(int il) {
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
void Engine6::rebalance(int max_swaps) {
    const MimoConfig & c = cfg_;
    const int E = c.n_expert;
    const auto t_reb = std::chrono::steady_clock::now();
    // every record of the last forward consumed first: layers whose experts were all on the GPUs are not waited for by
    // the GPU, so their routing counts (and a still-running job) could lag -- placement then depended on timing
    cpu_->drain();
    // decode routing seen by the CPU side since the last call
    for (int il = 0; il < nl_tot(); ++il) {
        if (!is_moe(il)) continue;
        auto & sc = ehost_[il].score;
        const auto & cnt = cpu_->counts[il];
        auto & last = ehost_[il].last_count;
        for (int e = 0; e < E; ++e) { sc[e] = sc[e] * adapt_decay_ + (double) (cnt[e] - last[e]); last[e] = cnt[e]; }
    }
    fill_extra(elastic_fill_);   // free extra slots first (a background batch)
    struct Sw { double gain; int il, ein, eout, g; };
    std::vector<Sw> cand;
    for (int il = 0; il < nl_tot(); ++il) {
        if (!is_moe(il)) continue;
        ExpertHost & H = ehost_[il];
        std::vector<int> cpu, gpu;
        for (int e = 0; e < E; ++e) {
            if (H.pend[e]) continue;   // (on its way into an extra slot)
            (H.owner[e] == CPU_OWNER ? cpu : gpu).push_back(e);
        }
        std::sort(cpu.begin(), cpu.end(), [&](int a, int b) { return H.score[a] > H.score[b]; });
        std::sort(gpu.begin(), gpu.end(), [&](int a, int b) { return H.score[a] < H.score[b]; });
        for (size_t i = 0; i < std::min(cpu.size(), gpu.size()); ++i) {
            const double si = H.score[cpu[i]], so = H.score[gpu[i]];
            // a swap costs ~0.5 ms of PCIe + a decode stall and pays ~0.2 ms per future hit: only clear, persistent differences
            if (si < so * adapt_ratio_ + adapt_min_) break;
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
        if (L.xg && k >= L.nbase) L.xe[k - L.nbase] = w.ein;   // (elastic: an extra slot changes hands)
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
    for (int il = 0; il < nl_tot(); ++il) {
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
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_reb).count();
    t_rebalance_ += ms;
    if (getenv("HYPER6_ADAPT_LOG")) fprintf(stderr, "hyper6: rebalance: %zu exchanges in %.1f ms (total %d, %.1f s)\n", cand.size(), ms, n_swaps_, t_rebalance_ / 1e3);
}

void Engine6::load_weights() {
    auto t0 = std::chrono::steady_clock::now();
    const MimoConfig & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd;
    for (auto & dp : devs_) {
        Device & dev = *dp;
        const int g = dev.g;
        auto A = [&](size_t nb) { return (void *) dev.alloc<uint8_t>(nb); };
        dev.layers.resize(nl_tot());
        for (int il = 0; il < nl_tot(); ++il) load_layer(dev, dev.layers[il], il);
        { const std::vector<float> v = to_f32(gguf_->need("output_norm.weight")); dev.out_norm = dev.upload(v.data(), v.size()); }
        if (n_mtp_) {
            dev.mtp.resize(n_mtp_);
            auto up32 = [&](const std::string & name) { const std::vector<float> v = to_f32(mtp_g_->need(name)); return dev.upload(v.data(), v.size()); };
            for (int h = 0; h < n_mtp_; ++h) {
                const int il = c.n_layer + h;
                load_layer(dev, dev.mtp[h], il);
                const std::string p = "blk." + std::to_string(il) + ".";
                { const GTensor * t = &mtp_g_->need(p + "nextn.eh_proj.weight"); dev.m_eh.push_back(upload_dense(A, dev.id, {{t, 0, t->rows()}})); }
                dev.m_enorm.push_back(up32(p + "nextn.enorm.weight"));
                dev.m_hnorm.push_back(up32(p + "nextn.hnorm.weight"));
                // head norm: shared_head_norm, else the block's layer_output_norm, else the model's output norm (as llama.cpp)
                const char * hn = mtp_g_->tensor(p + "nextn.shared_head_norm.weight") ? "nextn.shared_head_norm.weight"
                                : mtp_g_->tensor(p + "layer_output_norm.weight") ? "layer_output_norm.weight" : nullptr;
                dev.m_onorm.push_back(hn ? up32(p + hn) : dev.out_norm);
            }
            dev.hring = dev.alloc<float>((size_t) HR * n);
            dev.mh = dev.alloc<float>((size_t) MTPR * n);
            dev.mres = dev.alloc<float>((size_t) MTPR * n);
            dev.mchain = dev.alloc<float>((size_t) MAX_NT * n);
            dev.mpos = dev.alloc<int>(2);
        }
        auto [o0, o1] = split(c.n_vocab, nd, g);
        dev.vocab_off = (int) o0;
        { const GTensor & ot = gguf_->need("output.weight"); dev.output = upload_dense(A, dev.id, {{&ot, o0, o1}}, {}, true); }
        int bs = 0, nq_max = 0, ffl = 0, nkv_max = 0;
        std::vector<DevLayer *> all;
        for (auto & L : dev.layers) all.push_back(&L);
        for (auto & L : dev.mtp) all.push_back(&L);
        for (DevLayer * pl : all) {
            const DevLayer & L = *pl;
            bs = std::max(bs, L.nq * c.dk + L.nkv * (c.dk + c.dv));
            nq_max = std::max(nq_max, L.nq);
            nkv_max = std::max(nkv_max, L.nkv);
            ffl = std::max(ffl, L.ff_l);
        }
        dev.big_stride = (bs + 31) / 32 * 32;
        dev.o_stride = nq_max * c.dv;
        const int R = R6, K = c.n_expert_used;
        dev.res = dev.alloc<float>((size_t) R * n);
        for (int ci = 1; ci < mc_max_; ++ci) dev.res_x.push_back(dev.alloc<float>((size_t) R * n));
        dev.pos_x = dev.alloc<int>(8);
        dev.xn = dev.alloc<float>((size_t) R * n);
        dev.bo = dev.alloc<float>((size_t) R * n);
        dev.part = dev.alloc<float>((size_t) R * n);
        dev.big0 = dev.alloc<float>((size_t) R * dev.big_stride);
        dev.o = dev.alloc<float>((size_t) R * dev.o_stride);
        // attention partials: decode (MAX_NT rows, ns_global_ splits) or a prefill call's split blocks (<= 2048 blocks)
        dev.attn_part_n = std::max(mimo_attn_part_floats(nkv_max, MAX_NT, 1, ns_global_), (size_t) 2048 * MIMO_QB * 132);
        dev.attn_part = dev.alloc<float>(dev.attn_part_n);
        dev.rlog = dev.alloc<float>((size_t) R * c.n_expert);
        dev.router32 = dev.alloc<float>((size_t) c.n_expert * c.n_embd);
        dev.ticket = dev.alloc<unsigned>(1);
        dev.ids = dev.alloc<int>((size_t) R * K);
        dev.wts = dev.alloc<float>((size_t) R * K);
        dev.sg = dev.alloc<float>(R);
        dev.shgu = dev.alloc<float>((size_t) R * 2 * std::max(ffl, 1));
        dev.shh = dev.alloc<float>((size_t) R * std::max(ffl, 1));
        dev.shpart = dev.alloc<float>((size_t) R * n);   // (no shared expert: stays zero)
        dev.hexp = dev.alloc<float>((size_t) MAX_NT * K * c.n_ff_exp);
        dev.yexp = dev.alloc<float>((size_t) R * K * n);
        dev.logits = dev.alloc<float>((size_t) MAX_NT * dev.output.n());
        dev.xh = dev.alloc<half>((size_t) R * std::max({n, ffl, dev.o_stride}));
        dev.p16 = dev.alloc<half>((size_t) R * n);
        dev.recv = dev.alloc<half>((size_t) std::max(1, nd - 1) * R * n);
        dev.order = dev.alloc<int>((size_t) R * K);
        dev.order_n = dev.alloc<int>(2);
        dev.topk = dev.alloc<float>((size_t) MAX_NT * TOPK * 2);
        dev.egrp = dev.alloc<int>((size_t) 3 * c.n_expert);
        dev.mix16 = dev.alloc<half>((size_t) R * n);
        dev.h16 = dev.alloc<half>((size_t) R * K * c.n_ff_exp);
        dev.pos = dev.alloc<int>(1);
        dev.counter = dev.alloc<int>(1);
        dev.lstat = dev.alloc<float>(2);
    }
    // experts fill what is left on each GPU (minus a runtime reserve)
    {
        size_t eb = 0, eb_max = 0;
        int nm = 0;
        for (int il = 0; il < nl_tot(); ++il) {
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
            const size_t pend = devs_[g]->lend_limit;
            freeb[g] = freeb[g] > pend ? freeb[g] - pend : 0;
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
        fprintf(stderr, "hyper6: experts per layer on GPUs:");
        for (int q : quota) fprintf(stderr, " %d", q);
        fprintf(stderr, " (%.0f%% of %d), the rest on the CPU\n", 100.0 * tq / c.n_expert, c.n_expert);
        for (int il = 0; il < nl_tot(); ++il) if (is_moe(il)) load_experts(il, quota);
        if (opt_.stream_experts)
            for (auto & dp : devs_) {
                for (auto & sb : dp->stage) sb = dp->alloc<uint8_t>(std::max<size_t>(dp->stage_bytes, 1));
                fprintf(stderr, "hyper6: device %d streams prefill experts through 2 x %.0f MiB\n", dp->id, dp->stage_bytes / 1048576.0);
            }
    }
    for (auto & dp : devs_) fprintf(stderr, "hyper6: device %d holds %.2f GiB\n", dp->id, dp->used / 1073741824.0);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "hyper6: weights loaded in %.1f s (%d GPUs)\n", s, nd);
}

void Engine6::reset() {}   // (no recurrent state: caches are rewritten where positions are processed again)

void Engine6::record_main(int gi, int nt) {
    const MimoConfig & c = cfg_;
    Device & d = *devs_[gi];
    cudaStream_t s = d.stream;
    const int n = c.n_embd, K = c.n_expert_used, bs = d.big_stride;
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
    auto router_mm = [&](const DevLayer & RL, const float * x, float * y, int nr) {
        const int E = cfg_.n_expert, k = cfg_.n_embd;
        if (!RL.router16) f32mm(RL.router, E, k, x, k, y, E, nr);
        else if (nr <= MAX_NT) gemv_bf16w(RL.router16, E, k, x, k, y, E, nr, s);
        else {
            bf16_to_f32(RL.router16, (size_t) E * k, d.router32, s);
            f32mm(d.router32, E, k, x, k, y, E, nr);
        }
    };
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
        CUDA_CHECK(cudaMemcpyAsync(ck[ci].P, h_pos_ + ci, sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(ck[ci].R, h_embd_ + (size_t) ci * R6 * n, (size_t) ck[ci].nt * n * sizeof(float), cudaMemcpyHostToDevice, s));
    }
    incr_counter(d.counter, s);
    const bool streaming = bulk && opt_.stream_experts && nt >= stream_min_;
    if (streaming) {
        d.up_pend.clear();
        int first = 0;
        while (first < c.n_layer && !d.layers[first].owner_bulk) ++first;
        if (first < c.n_layer) { upload_stage(d, first, 0); upload_stage(d, first, 1); }
    }
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
        if (getenv("HYPER4_SUMS")) fprintf(stderr, "SUMS %d %s %.17g %.17g\n", il, what, sum, sq);
        if (bad || err != cudaSuccess) throw std::runtime_error("debug stop");
    };
    int dcall = 0;
    auto allreduce = [&](float * x) {   // x += sum over the GPUs of d.part
        if (!bulk) { allreduce_add_ll16(x, d.part, ar_ll_, d.g, nd, nt * n, d.counter, call++, s, nullptr); return; }
        const int par = dcall++ & 1;
        const size_t N = (size_t) nt * n;
        auto stage = [&](int g) { return h_stage_ + ((size_t) par * nd + g) * R6 * n; };
        to_half(d.part, n, nullptr, n, 0.0f, d.p16, nt, s);
        CUDA_CHECK(cudaMemcpyAsync(stage(d.g), d.p16, N * sizeof(half), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaEventRecord(d.ev_ar[par], s));
        barrier_->wait();
        int j = 0;
        for (int p = 0; p < nd; ++p) {
            if (p == d.g) continue;
            CUDA_CHECK(cudaStreamWaitEvent(s, devs_[p]->ev_ar[par], 0));
            CUDA_CHECK(cudaMemcpyAsync(d.recv + (size_t) j * R6 * n, stage(p), N * sizeof(half), cudaMemcpyHostToDevice, s));
            ++j;
        }
        if (!d.up_pend.empty()) {   // the next expert upload part, behind these downloads
            CUDA_CHECK(cudaEventRecord(d.ev_arh, s));
            CUDA_CHECK(cudaStreamWaitEvent(d.cstream, d.ev_arh, 0));
            const auto q = d.up_pend.front();
            d.up_pend.pop_front();
            upload_stage(d, q[0], q[1], q[2], up_parts_);
        }
        add_parts(x, d.p16, d.recv, (size_t) R6 * n, nd - 1, (int) N, s);
    };
    dbg("embed", -1, R, (size_t) nt * n);
    for (int il = 0; il < c.n_layer; ++il)
    for (int ci = 0; ci < nck; ++ci) {
        nt = ck[ci].nt; R = ck[ci].R; P = ck[ci].P; pos_host = ck[ci].pos;
        const bool last_ck = ci == nck - 1;
        DevLayer & L = d.layers[il];
        // ---- attention ----
        rmsnorm(R, n, L.attn_norm, d.xn, n, n, nt, eps, s);
        mm(L.wqkv, d.xn, n, d.big0, bs, nt);
        mimo_qkv(d.big0, bs, L.nq, L.nkv, c.dk, c.dv, c.n_rot, L.swa ? c.rope_base_swa : c.rope_base, P, nt, L.kc, L.vc, L.kpitch, L.vpitch,
                 L.ring, s);
        {
            MimoAttn a;
            a.q = d.big0; a.qs = bs;
            a.kc = L.kc; a.vc = L.vc; a.kpitch = L.kpitch; a.vpitch = L.vpitch; a.ring = L.ring;
            a.pos = P; a.nt = nt; a.nq = L.nq; a.nkv = L.nkv; a.dk = c.dk; a.dv = c.dv;
            a.window = L.swa ? c.window : 0;
            a.scale = 1.0f / sqrtf((float) c.dk);
            a.vscale = c.v_scale;
            a.sinks = L.sinks;
            a.out = d.o; a.os = L.nq * c.dv;
            a.part = d.attn_part;
            int ns;
            if (!bulk) ns = L.swa ? 4 : ns_global_;
            else {   // prefill: splits only when the row blocks alone leave the GPU idle (deep context, short chunk)
                const int group = L.nq / std::max(1, L.nkv), RB = MIMO_QB / group, nrb = (nt + RB - 1) / RB, blocks = L.nkv * nrb;
                const int keys = L.swa ? c.window + RB : pos_host + nt;
                ns = 1;
                while (ns < 32 && blocks * ns * 2 <= 256 && keys / (ns * 2) >= 1024) ns *= 2;
                if (mimo_attn_part_floats(L.nkv, nt, group, ns) > d.attn_part_n) ns = 1;
            }
            static const bool fa_off = getenv("HYPER6_FA") && atoi(getenv("HYPER6_FA")) == 0;
            if (bulk && !L.swa && !fa_off) mimo_attn_fa(a, s);   // (global layers: the keys of the whole context)
            else mimo_attn(a, ns, s);
        }
        mm(L.wo, d.o, L.nq * c.dv, d.part, n, nt);
        dbg("attn_part", il, d.part, (size_t) nt * n);
        allreduce(R);
        // ---- FFN ----
        rmsnorm(R, n, L.ffn_norm, d.xn, n, n, nt, eps, s);
        if (!L.moe) {
            mm(L.gu, d.xn, n, d.shgu, 2 * L.ff_l, nt);
            if (nt <= MAX_NT) {
                NormIn glu; glu.act = 2; glu.glu_off = L.ff_l;
                mm(L.down, d.shgu, 2 * L.ff_l, d.part, n, nt, glu);
            } else {
                silu_mul(d.shgu, 2 * L.ff_l, d.shh, L.ff_l, L.ff_l, nt, s);
                mm(L.down, d.shh, L.ff_l, d.part, n, nt);
            }
        } else {
            router_mm(L, d.xn, d.rlog, nt);
            const bool fused_pub = !bulk && d.g == 0;
            if (fused_pub)
                moe_route_sig_publish(d.rlog, c.n_expert, L.exp_bias, c.n_expert, K, c.w_scale, d.ids, d.wts, d.sg, nt, &cpu_rec_[il].seq,
                                      &cpu_rec_[il].nt, &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0], d.xn, n, n,
                                      d.counter, (unsigned) il, d.ticket, s);
            else moe_route_sig(d.rlog, c.n_expert, L.exp_bias, c.n_expert, K, c.w_scale, d.ids, d.wts, d.sg, nt, s);
            dbg("route_w", il, d.wts, (size_t) nt * K);
            if (bulk && d.g == 0 && adapt_)
                CUDA_CHECK(cudaMemcpyAsync(h_ids_ + ((size_t) il * mc_max_ + ci) * R6 * K, d.ids, (size_t) nt * K * sizeof(int), cudaMemcpyDeviceToHost, s));
            const bool stream = streaming && L.owner_bulk;
            if (d.g == 0 && !stream && bulk)
                moe_publish(&cpu_bulk_->seq, &cpu_bulk_->nt, &cpu_bulk_->ids[0][0], &cpu_bulk_->wts[0][0], &cpu_bulk_->x[0][0],
                            d.xn, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
            if (bulk) {
                moe_order(L.moex, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
                to_half(d.xn, n, nullptr, n, 0.0f, d.mix16, nt, s);
                moe_gemm_gate_up(L.moex, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
                moe_gemm_down(L.moex, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
                if (stream)
                    for (int sb = 0; sb < 2; ++sb) {
                        MoeDev ms = L.moex;
                        const int cnt = (int) L.st_list[sb].size();
                        ms.gate = d.stage[sb];
                        ms.up = d.stage[sb] + (size_t) cnt * L.moex.gate_bytes;
                        ms.down = d.stage[sb] + (size_t) 2 * cnt * L.moex.gate_bytes;
                        ms.slot = L.st_slot[sb];
                        while (!d.up_pend.empty() && (d.up_pend.front()[0] < il || (d.up_pend.front()[0] == il && d.up_pend.front()[1] <= sb))) {
                            const auto q = d.up_pend.front();
                            d.up_pend.pop_front();
                            upload_stage(d, q[0], q[1], q[2], up_parts_);
                        }
                        CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_up[sb], 0));
                        moe_order(ms, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
                        moe_gemm_gate_up(ms, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
                        moe_gemm_down(ms, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
                        if (!last_ck) continue;
                        CUDA_CHECK(cudaEventRecord(d.ev_free[sb], s));
                        int nx = il + 1;
                        while (nx < c.n_layer && !d.layers[nx].owner_bulk) ++nx;
                        if (nx < c.n_layer) {
                            if (up_parts_ > 0) for (int pp = 0; pp < up_parts_; ++pp) d.up_pend.push_back({nx, sb, pp});
                            else upload_stage(d, nx, sb);
                        }
                    }
            } else {
                moe_gate_up(L.moex, d.xn, n, d.ids, K, d.hexp, nt, s);
                moe_down(L.moex, d.hexp, d.ids, d.wts, K, d.yexp, nt, s);
            }
            const volatile unsigned * cflag = d.g == 0 && !nocpu_ && !stream ? (bulk ? &cpu_bulk_out_->seq : &cpu_out_[il].seq) : nullptr;
            moe_reduce(d.shpart, d.sg, d.yexp, K, d.part, n, nt, d.ids, stream ? L.owner_bulk : L.owner, d.g, CPU_OWNER, cflag,
                       bulk ? &cpu_bulk_out_->y[0][0] : &cpu_out_[il].y[0][0], d.counter, (unsigned) il, s, nullptr);
        }
        dbg("ffn_part", il, d.part, (size_t) nt * n);
        allreduce(R);
        dbg("l_out", il, R, (size_t) nt * n);
    }
    if (d.hring)   // (the main hidden rows, MTP inputs; every chunk of a multi-chunk pass)
        for (int ci = 0; ci < nck; ++ci) mimo_hstore(ck[ci].R, ck[ci].nt, ck[ci].P, d.hring, HR, n, s);
    const int hr = bulk ? 1 : nt;
    rmsnorm(R + (size_t) (nt - hr) * n, n, d.out_norm, d.xn, n, n, hr, eps, s);
    mm(d.output, d.xn, n, d.logits, d.output.n(), hr);
    argmax_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.wts, hr, s);
    CUDA_CHECK(cudaMemcpyAsync(h_res_ + (size_t) gi * MAX_NT * 2, d.wts, (size_t) hr * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
    topk_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.topk, TOPK, hr, s);
    CUDA_CHECK(cudaMemcpyAsync(h_topk_ + (size_t) gi * MAX_NT * TOPK * 2, d.topk, (size_t) hr * TOPK * 2 * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
}

void Engine6::record_mtp(int gi, int head, int nrows, int nmain, bool full) {
    const MimoConfig & c = cfg_;
    Device & d = *devs_[gi];
    cudaStream_t s = d.stream;
    const int n = c.n_embd, bs = d.big_stride, nd = opt_.n_devices;
    const float eps = c.rms_eps;
    DevLayer & L = d.mtp[head];
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
    CUDA_CHECK(cudaMemcpyAsync(d.mpos, h_pos_ + 12, 2 * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.bo, h_embd_, (size_t) nrows * n * sizeof(float), cudaMemcpyHostToDevice, s));
    incr_counter(d.counter, s);
    mimo_hgather(d.mh, d.hring, HR, n, h_pos_[12], nrows, nmain, d.mchain, s);
    float * ecat = d.yexp;   // [nrows][2n]
    rmsnorm(d.bo, n, d.m_enorm[head], ecat, 2 * n, n, nrows, eps, s);
    rmsnorm(d.mh, n, d.m_hnorm[head], ecat + n, 2 * n, n, nrows, eps, s);
    mm(d.m_eh[head], ecat, 2 * n, d.mres, n, nrows);
    float * R = d.mres;
    rmsnorm(R, n, L.attn_norm, d.xn, n, n, nrows, eps, s);
    mm(L.wqkv, d.xn, n, d.big0, bs, nrows);
    mimo_qkv(d.big0, bs, L.nq, L.nkv, c.dk, c.dv, c.n_rot, c.rope_base_swa, d.mpos, nrows, L.kc, L.vc, L.kpitch, L.vpitch, L.ring, s);
    if (!full) return;
    if (nrows > 1) {   // the last row goes on alone
        CUDA_CHECK(cudaMemcpyAsync(R, R + (size_t) (nrows - 1) * n, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d.big0, d.big0 + (size_t) (nrows - 1) * bs, (size_t) L.nq * c.dk * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    MimoAttn a;
    a.q = d.big0; a.qs = bs;
    a.kc = L.kc; a.vc = L.vc; a.kpitch = L.kpitch; a.vpitch = L.vpitch; a.ring = L.ring;
    a.pos = d.mpos + 1; a.nt = 1; a.nq = L.nq; a.nkv = L.nkv; a.dk = c.dk; a.dv = c.dv;
    a.window = c.window;
    a.scale = 1.0f / sqrtf((float) c.dk);
    a.vscale = c.v_scale;
    a.sinks = L.sinks;
    a.out = d.o; a.os = L.nq * c.dv;
    a.part = d.attn_part;
    mimo_attn(a, 4, s);
    int call = 0;
    mm(L.wo, d.o, L.nq * c.dv, d.part, n, 1);
    allreduce_add_ll16(R, d.part, ar_ll_, d.g, nd, n, d.counter, call++, s, nullptr);
    rmsnorm(R, n, L.ffn_norm, d.xn, n, n, 1, eps, s);
    mm(L.gu, d.xn, n, d.shgu, 2 * L.ff_l, 1);
    NormIn glu; glu.act = 2; glu.glu_off = L.ff_l;
    mm(L.down, d.shgu, 2 * L.ff_l, d.part, n, 1, glu);
    allreduce_add_ll16(R, d.part, ar_ll_, d.g, nd, n, d.counter, call++, s, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(d.mchain + (size_t) head * n, R, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    rmsnorm(R, n, d.m_onorm[head], d.xn, n, n, 1, eps, s);
    mm(d.output, d.xn, n, d.logits, d.output.n(), 1);
    argmax_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.wts, 1, s);
    max_sumexp(d.logits, d.output.n(), d.wts + 2, s);   // (the draft's probability: the drafts' gate)
    topk_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.topk, TOPK, 1, s);
    CUDA_CHECK(cudaMemcpyAsync(h_mtopk_ + (size_t) gi * TOPK * 2, d.topk, (size_t) TOPK * 2 * sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(h_res_ + (size_t) gi * MAX_NT * 2, d.wts, 4 * sizeof(float), cudaMemcpyDeviceToHost, s));
}

void Engine6::mtp_pass(int head, const int * tokens, int nrows, int q0, int nmain, bool full) {
    if (nrows < 1 || nrows > MTPR || head < 0 || head >= n_mtp_) throw std::runtime_error("mtp_pass: bad rows");
    if (q0 + nrows > opt_.max_pos) throw std::runtime_error("mtp_pass: position exceeds max_pos");
    embed(tokens, nrows, 0, q0);
    h_pos_[12] = q0;
    h_pos_[13] = q0 + nrows - 1;
    mtp_pos_ = q0 + nrows;   // (the drafted token's position)
    ++fwd_counter_;          // (the devices' counters advance with every pass)
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        CUDA_CHECK(cudaSetDevice(devs_[gi]->id));
        record_mtp(gi, head, nrows, nmain, full);
    }
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->stream)); }
}

int Engine6::mtp_result() {
    float best = -INFINITY; int bi = -1;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float v = h_res_[(g * MAX_NT) * 2];
        const int idx = ((const int *) h_res_)[(g * MAX_NT) * 2 + 1];
        if (v > best) { best = v; bi = idx; }
    }
    float M = -INFINITY;
    for (size_t g = 0; g < devs_.size(); ++g) M = std::max(M, h_res_[(g * MAX_NT) * 2 + 2]);
    double z = 0;
    for (size_t g = 0; g < devs_.size(); ++g) z += h_res_[(g * MAX_NT) * 2 + 3] * std::exp((double) h_res_[(g * MAX_NT) * 2 + 2] - M);
    mtp_p_ = z > 0 ? 1.0 / z : 0.0;
    if (samp_invT_ > 0) {
        std::vector<std::pair<float, int>> cand;
        for (size_t g = 0; g < devs_.size(); ++g)
            for (int i = 0; i < TOPK; ++i) {
                const float * q = h_mtopk_ + ((size_t) g * TOPK + i) * 2;
                const int idx = ((const int *) q)[1];
                if (idx >= 0) cand.push_back({q[0], idx});
            }
        if (!cand.empty()) return draft_pick(cand, samp_, samp_seed_, mtp_pos_);
    }
    return bi;
}

void Engine6::build_graphs() {
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

void Engine6::run(int nt) {
    const bool bulk = nt > MAX_NT;
    if (!graphs_ready_ && !debug_) build_graphs();
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
                catch (const std::exception & ex) { err[gi] = ex.what(); barrier_->abort(); }
            });
        for (auto & t : th) t.join();
        std::string first;
        for (auto & m : err) if (!m.empty() && (first.empty() || first.rfind("barrier:", 0) == 0)) first = m;
        if (!first.empty()) { barrier_->reset(); throw std::runtime_error(first); }
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
                fprintf(stderr, "hyper6: forward %u stuck on device %d; cpu: %s\n", fwd_counter_, dp->id, cpu_->state().c_str());
            }
            std::this_thread::yield();
        }
    }
}

std::vector<int> Engine6::forward(const int * tokens, int nt, int pos) {
    if (nt < 1 || nt > R6) throw std::runtime_error("forward: bad token count");
    if (pos + nt > opt_.max_pos) throw std::runtime_error("forward: position exceeds max_pos");
    ensure_kv(pos + nt + 2 * MAX_NT);
    const bool bulk = nt > MAX_NT;
    embed(tokens, nt, 0, pos);
    h_pos_[0] = pos;
    if (elastic_ && !bulk) {
        ++dec_steps_;
        if (!fill_list_.empty() && dec_steps_ >= fill_due_) fill_wait(false);
        if (++req_steps_ == 32) shrink_kv(pos + nt + 2 * MAX_NT);
    }
    if (adapt_ && !bulk) {
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
            for (int i = 0; i < nt * K; ++i) { const int e = h_ids_[(size_t) il * mc_max_ * R6 * K + i]; if (e >= 0 && e < cfg_.n_expert) sc[e] += prompt_weight_; }
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

// ---------------- prompt-cache snapshots (the window before a position) ----------------
size_t Engine6::snap_bytes(int gi) const {
    const MimoConfig & c = cfg_;
    const Device & d = *devs_[gi];
    const size_t W = (size_t) c.window;
    size_t b = 0;
    for (const auto * v : {&d.layers, &d.mtp})
        for (const DevLayer & L : *v) if (L.swa) b += (size_t) L.nkv * W * (c.dk + c.dv) * sizeof(half);
    if (d.hring) b += W * c.n_embd * sizeof(float);
    return b;
}

void Engine6::snap_xfer(Snap & sn, bool to_host) {
    const MimoConfig & c = cfg_;
    const int W = c.window, a = std::max(0, sn.pos - W), b = sn.pos;
    for (size_t gi = 0; gi < devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        uint8_t * hp = (uint8_t *) sn.h[gi];
        // rows q in [a, b) of a ring of R rows (row bytes rb) <-> host rows q - (pos - W)
        auto ring = [&](uint8_t * dev, size_t rb, int R) {
            for (int q = a; q < b;) {
                const int r = q % R, n = std::min(b - q, R - r);
                uint8_t * hrow = hp + (size_t) (q - (sn.pos - W)) * rb, * drow = dev + (size_t) r * rb;
                if (to_host) CUDA_CHECK(cudaMemcpyAsync(hrow, drow, (size_t) n * rb, cudaMemcpyDeviceToHost, d.stream));
                else CUDA_CHECK(cudaMemcpyAsync(drow, hrow, (size_t) n * rb, cudaMemcpyHostToDevice, d.stream));
                q += n;
            }
            hp += (size_t) W * rb;
        };
        for (auto * v : {&d.layers, &d.mtp})
            for (DevLayer & L : *v) {
                if (!L.swa) continue;
                for (int h = 0; h < L.nkv; ++h) {
                    ring((uint8_t *) (L.kc + (size_t) h * L.kpitch), (size_t) c.dk * sizeof(half), L.ring);
                    ring((uint8_t *) (L.vc + (size_t) h * L.vpitch), (size_t) c.dv * sizeof(half), L.ring);
                }
            }
        if (d.hring) ring((uint8_t *) d.hring, (size_t) c.n_embd * sizeof(float), HR);
    }
    for (auto & dp : devs_) { CUDA_CHECK(cudaSetDevice(dp->id)); CUDA_CHECK(cudaStreamSynchronize(dp->stream)); }
}

// per-position GPU buffers of a conversation (parked to RAM): the global layers' K and V rows
std::vector<ConvPark<Engine6::Snap>::Dev> Engine6::park_devs() const {
    std::vector<ConvPark<Snap>::Dev> v;
    for (auto & dp : devs_) {
        ConvPark<Snap>::Dev pd;
        pd.id = dp->id;
        pd.stream = dp->stream;
        for (auto & L : dp->layers)
            if (!L.swa && L.nkv > 0) {
                pd.spans.push_back({L.kc, (size_t) cfg_.dk * sizeof(half), L.kpitch * sizeof(half), L.nkv, 1});
                pd.spans.push_back({L.vc, (size_t) cfg_.dv * sizeof(half), L.vpitch * sizeof(half), L.nkv, 1});
            }
        v.push_back(std::move(pd));
    }
    return v;
}

void Engine6::take_snapshot(int pos) {
    for (auto & s : snaps_) if (s.pos == pos) return;
    if ((int) snaps_.size() >= opt_.max_snapshots) {   // (the oldest position, sparing every 4096th)
        auto victim = snaps_.end();
        for (auto it = snaps_.begin(); it != snaps_.end(); ++it)
            if (it->pos % 4096 && (victim == snaps_.end() || it->pos < victim->pos)) victim = it;
        if (victim == snaps_.end())
            victim = std::min_element(snaps_.begin(), snaps_.end(), [](const Snap & x, const Snap & y) { return x.pos < y.pos; });
        snap_pool_.push_back(victim->h);
        snaps_.erase(victim);
    }
    Snap sn;
    sn.pos = pos;
    if (!snap_pool_.empty()) { sn.h = snap_pool_.back(); snap_pool_.pop_back(); }
    else
        for (size_t gi = 0; gi < devs_.size(); ++gi) {
            void * p = nullptr;
            CUDA_CHECK(cudaHostAlloc(&p, std::max<size_t>(snap_bytes((int) gi), 1), cudaHostAllocPortable));
            sn.h.push_back((float *) p);
        }
    snap_xfer(sn, true);
    snaps_.push_back(sn);
}

std::vector<int> Engine6::generate(const std::vector<int> & prompt, int n_gen, bool spec_req, GenStats * stats,
                                   const std::function<bool(int)> & on_token, const SamplingParams & sp) {
    const PromptMedia * md = media_;   // (taken first: a request that fails leaves none for the next)
    media_ = nullptr;
    if (prompt.empty()) throw std::runtime_error("generate: empty prompt");
    if ((int) prompt.size() + 8 > opt_.max_pos) throw std::runtime_error("generate: prompt longer than the context");
    using clk = std::chrono::steady_clock;
    const bool sampling = sp.temp > 0.0f;
    samp_seed_ = sp.seed ? sp.seed : random_seed64();
    mrows_.clear();
    if (md) {
        if ((int) md->keys.size() != (int) prompt.size()) throw std::runtime_error("generate: media do not match the prompt");
        for (const auto & r : md->rows) mrows_[r.first] = r.second;
    }
    const std::vector<int> & key = md ? md->keys : prompt;
    static const bool gumbel_draft = getenv("HYPER_GUMBEL_DRAFT") ? atoi(getenv("HYPER_GUMBEL_DRAFT")) != 0 : true;
    samp_invT_ = sampling && gumbel_draft ? 1.0f / sp.temp : 0.0f;   // (MTP drafts coupled to the sampling noise)
    samp_ = sp;
    const int P = (int) prompt.size();
    int s = 0;
    if (opt_.prompt_cache) {
        // resume after the common prefix L: directly while the sliding-window rings (and the hidden ring) still hold the window
        // before it, else from the latest snapshot at or before it (an answer that was cut off or re-rendered: the rings moved on)
        // another conversation: the current one parked (or a parked one swapped back: its rings are not the live ones)
        const std::vector<int> before = hist_;
        int L = std::min(park_.resolve(key, hist_, snaps_, snap_pool_, park_devs(), "hyper6"), P - 1);
        const bool swapped = hist_ != before;
        for (size_t i = 0; i < snaps_.size();)
            if (snaps_[i].pos > L) { snap_pool_.push_back(snaps_[i].h); snaps_.erase(snaps_.begin() + i); } else ++i;
        const int newest = (int) hist_.size() - 1;
        if (L > 0 && !swapped && newest - L < std::min(ring_, (int) HR) - cfg_.window - 2 * MAX_NT) s = L;
        else {
            Snap * best = nullptr;
            for (auto & sn : snaps_) if (sn.pos <= L && (!best || sn.pos > best->pos)) best = &sn;
            if (best) { snap_xfer(*best, false); s = best->pos; }
        }
    }
    std::vector<int> snap_at;   // prefill: snapshots at the last message start and every pass of mc_max_ chunks
    if (opt_.prompt_cache) {
        std::vector<int> msg;
        for (int q = s + 1; q < P; ++q)
            if (std::find(snap_tokens_.begin(), snap_tokens_.end(), prompt[q]) != snap_tokens_.end()) msg.push_back(q);
        for (int q = s + R6 * mc_max_; q < P; q += R6 * mc_max_) snap_at.push_back(q);
        if (!msg.empty() && msg.back() - s >= 64) snap_at.push_back(msg.back());
        std::sort(snap_at.begin(), snap_at.end());
        snap_at.erase(std::unique(snap_at.begin(), snap_at.end()), snap_at.end());
    }
    req_steps_ = 0;
    ensure_kv(P + 2 * MAX_NT);
    GenStats st;
    const int K = std::min(opt_.n_draft, MAX_NT - 1);
    auto tp = clk::now();
    int next = -1;
    size_t si = 0;
    for (int c0 = s; c0 < P;) {
        while (si < snap_at.size() && snap_at[si] <= c0) ++si;
        const int seg = (si < snap_at.size() ? snap_at[si] : P) - c0, nch = (seg + R6 - 1) / R6;
        const int L = (seg + nch - 1) / nch;
        int lens[8], nk = 0, tot = 0;
        while (nk < mc_max_ && tot < seg) { lens[nk] = std::min(L, seg - tot); tot += lens[nk++]; }
        bool multi = nk > 1 && opt_.stream_experts;
        for (int i = 0; i < nk && multi; ++i) multi = lens[i] >= stream_min_ && lens[i] > MAX_NT;
        if (!multi) { nk = 1; tot = lens[0]; }
        const int end = c0 + tot, len = lens[nk - 1];
        next = multi ? forward_multi(&prompt[c0], lens, nk, c0) : forward(&prompt[c0], len, c0)[len - 1];
        if (sampling) next = sample_row(len <= MAX_NT ? len - 1 : 0, sp, end);
        if (si < snap_at.size() && snap_at[si] == end) take_snapshot(end);
        if (prefill_cb_) prefill_cb_(end, P, s);
        c0 = end;
    }
    const bool use_mtp = mtp_g_ && K > 0 && spec_req;
    if (use_mtp) {   // the NextN heads' caches over the window before the first draft position (main hidden rows of the prompt)
        const int q0 = std::max(0, P - (cfg_.window - 1));
        for (int h = 0; h < n_mtp_; ++h) mtp_pass(h, &prompt[q0], P - q0, q0, P - q0, false);
    }
    hist_ = key;
    if (opt_.prompt_cache) take_snapshot(P);   // (an answer cut off or re-rendered resumes from the prompt's end)
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
    int snap_mark = p / 1024;   // generated positions get snapshots every 1024 too (a long answer cut off resumes near the cut)
    auto gen_snapshot = [&] {
        if (!opt_.prompt_cache || p / 1024 == snap_mark) return;
        snap_mark = p / 1024;
        take_snapshot(p);
    };
    if (use_mtp) {
        // MTP drafts by the chained NextN heads (llama.cpp's chain_heads): head h runs over the rows at p .. p + h (the token at p
        // with the main hidden of p - 1, then the drafts with the outputs of heads 0 .. h - 1); its last row drafts the token at
        // p + h + 1. Drafts while their run's probability stays >= mtp_pmin_; a draft is kept iff the token sampled (or argmax)
        // at its row equals it: exact. After a step, the kept rows enter every head's cache with the main model's hidden rows.
        std::vector<int> drafts(K), in(K + 1), toks(K + 1);
        int cur = next;
        while (!stop && p + K + 1 < opt_.max_pos) {
            if (!emit(cur)) break;
            auto ta = clk::now();
            int nd = 0;
            double pc = 1.0;
            toks[0] = cur;
            for (int h = 0; h < std::min(K, n_mtp_); ++h) {
                mtp_pass(h, toks.data(), h + 1, p, 1, true);
                const int dt = mtp_result();
                pc *= mtp_p_;
                if (pc < mtp_pmin_) break;
                drafts[nd++] = dt;
                toks[h + 1] = dt;
            }
            st.t_mtp += since(ta);
            in[0] = cur;
            for (int j = 0; j < nd; ++j) in[j + 1] = drafts[j];
            ta = clk::now();
            std::vector<int> a = forward(in.data(), nd + 1, p);
            st.t_main += since(ta);
            st.steps++;
            st.drafted += nd;
            int m = 0;
            if (sampling) { while (m < nd && (a[m] = sample_row(m, sp, p + m + 1)) == drafts[m]) ++m; if (m == nd) a[nd] = sample_row(nd, sp, p + nd + 1); }
            else while (m < nd && a[m] == drafts[m]) ++m;
            st.accepted += m;
            int r = 0;
            while (r < m && emit(drafts[r++])) {}
            if (stop) { if (opt_.answer_reuse) p += r; break; }
            ta = clk::now();
            for (int h = 0; h < n_mtp_; ++h) mtp_pass(h, in.data(), m + 1, p, m + 1, false);
            st.t_mtp += since(ta);
            cur = a[m];
            p += m + 1;
            gen_snapshot();
        }
    } else if (K <= 0 || !spec_req) {
        while (emit(next) && p + 1 < opt_.max_pos) {
            next = forward(&next, 1, p++)[0];
            if (sampling) next = sample_row(0, sp, p);
            st.steps++;
            gen_snapshot();
        }
    } else {
        // prompt-lookup speculation (as Engine5); a rejected draft's cache rows are rewritten when its position comes again
        constexpr int NG = 3;
        std::vector<int> hist(prompt.begin(), prompt.end());
        std::unordered_map<uint64_t, int> last;
        auto key = [&](size_t end) { uint64_t h = 1469598103934665603ull; for (size_t i = end - NG; i < end; ++i) h = (h ^ (uint32_t) hist[i]) * 1099511628211ull; return h; };
        size_t indexed = NG;
        auto index_upto = [&](size_t n) { for (; indexed <= n; ++indexed) last[key(indexed)] = (int) indexed; };
        int cur = next;
        std::vector<int> in(K + 1);
        while (!stop && p + K + 1 < opt_.max_pos) {
            if (!emit(cur)) break;
            hist.push_back(cur);
            int nd = 0;
            if (hist.size() > NG) {
                auto it = last.find(key(hist.size()));
                index_upto(hist.size() - 1);
                if (it != last.end()) {
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
            st.drafted += nd;
            int m = 0;
            if (sampling) { while (m < nd && (a[m] = sample_row(m, sp, p + m + 1)) == in[1 + m]) ++m; if (m == nd) a[nd] = sample_row(nd, sp, p + nd + 1); }
            else while (m < nd && a[m] == in[1 + m]) ++m;
            st.accepted += m;
            if (nd > 0) acc_rate_ = 0.9 * acc_rate_ + 0.1 * ((double) m / nd);
            int r = 0;
            while (r < m) { hist.push_back(in[1 + r]); if (!emit(in[1 + r++])) break; }
            if (stop) { if (opt_.answer_reuse) p += r; break; }
            cur = a[m];
            p += m + 1;
            gen_snapshot();
        }
    }
    hist_ = key;
    hist_.insert(hist_.end(), out.begin(), out.end());
    hist_.resize(std::min<size_t>(hist_.size(), (size_t) p));   // (positions < p have their cache rows)
    st.tokens = (int) out.size();
    st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

// part p of np of half sb (split by expert index); the first part waits for the staging buffer, the last one signals ev_up
void Engine6::upload_stage(Device & d, int il, int sb, int p, int np) {
    DevLayer & L = d.layers[il];
    const ExpertHost & H = ehost_[il];
    const std::vector<int> & lst = L.st_list[sb];
    const int cnt = (int) lst.size();
    if (np < 1) np = 1;
    const int i0 = (int) ((int64_t) cnt * p / np), i1 = (int) ((int64_t) cnt * (p + 1) / np);
    if (p == 0) CUDA_CHECK(cudaStreamWaitEvent(d.cstream, d.ev_free[sb], 0));
    const size_t gb = H.gb, db = H.db;
    for (int i = i0; i < i1;) {   // runs of consecutive experts in one copy each
        int j = i + 1;
        while (j < i1 && lst[j] == lst[j - 1] + 1) ++j;
        const size_t e = (size_t) lst[i], r = (size_t) (j - i);
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) i * gb, H.gate + e * gb, r * gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) (cnt + i) * gb, H.up + e * gb, r * gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) 2 * cnt * gb + (size_t) i * db, H.down + e * db, r * db, cudaMemcpyHostToDevice,
                                   d.cstream));
        i = j;
    }
    if (p == np - 1) CUDA_CHECK(cudaEventRecord(d.ev_up[sb], d.cstream));
}

void Engine6::embed(const int * tokens, int nt, int chunk, int pos) {
    const GTensor & te = gguf_->need("token_embd.weight");
    const auto * te_tr = ggml_get_type_traits((ggml_type) te.type);
    for (int t = 0; t < nt; ++t) {
        float * out = h_embd_ + ((size_t) chunk * R6 + t) * cfg_.n_embd;
        if (pos >= 0 && !mrows_.empty()) {
            const auto it = mrows_.find(pos + t);
            if (it != mrows_.end()) { memcpy(out, it->second, cfg_.n_embd * sizeof(float)); continue; }
        }
        const uint8_t * row = te.data + (size_t) tokens[t] * te.row_bytes();
        if (te.type == GType::F32) memcpy(out, row, cfg_.n_embd * sizeof(float));
        else te_tr->to_float(row, out, cfg_.n_embd);
    }
}

int Engine6::sample_row(int t, const SamplingParams & sp, int64_t pos) {
    std::vector<std::pair<float, int>> cand;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float * p = h_topk_ + ((size_t) g * MAX_NT + t) * TOPK * 2;
        for (int i = 0; i < TOPK; ++i) { const int idx = ((const int *) p)[2 * i + 1]; if (idx >= 0) cand.push_back({p[2 * i], idx}); }
    }
    // probabilities over the whole row when the candidates are not enough (top_k off): normalizer per vocab slice on the GPUs
    auto stats = [&](double invT, float & M, double & Z) {
        std::vector<float> ms(2 * devs_.size());
        for (size_t g = 0; g < devs_.size(); ++g) {
            Device & d = *devs_[g];
            CUDA_CHECK(cudaSetDevice(d.id));
            max_sumexp(d.logits + (size_t) t * d.output.n(), d.output.n(), d.lstat, d.stream, (float) invT);
            CUDA_CHECK(cudaMemcpyAsync(&ms[2 * g], d.lstat, 2 * sizeof(float), cudaMemcpyDeviceToHost, d.stream));
        }
        M = -INFINITY;
        for (size_t g = 0; g < devs_.size(); ++g) {
            CUDA_CHECK(cudaSetDevice(devs_[g]->id));
            CUDA_CHECK(cudaStreamSynchronize(devs_[g]->stream));
            M = std::max(M, ms[2 * g]);
        }
        Z = 0;
        for (size_t g = 0; g < devs_.size(); ++g) Z += ms[2 * g + 1] * std::exp(((double) ms[2 * g] - M) * invT);
    };
    return sample_candidates(cand, TOPK, sp, samp_seed_, pos, stats, [&](std::vector<float> & lg) { get_logits(t, lg); });
}

void Engine6::save_expert_stats(const std::string & path) {
    const int nl = nl_tot(), w = 1024;
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

int Engine6::forward_multi(const int * tokens, const int * lens, int nck, int pos) {
    bool ok = nck > 1 && nck <= mc_max_ && opt_.stream_experts;
    for (int ci = 0; ci < nck && ok; ++ci) ok = lens[ci] >= stream_min_ && lens[ci] <= R6 && lens[ci] > MAX_NT;
    if (!ok) {   // one chunk at a time
        int next = -1, off = 0;
        for (int ci = 0; ci < nck; ++ci) { next = forward(tokens + off, lens[ci], pos + off)[lens[ci] - 1]; off += lens[ci]; }
        return next;
    }
    int off = 0;
    for (int ci = 0; ci < nck; ++ci) {
        embed(tokens + off, lens[ci], ci, pos + off);
        mc_nt_[ci] = lens[ci];
        mc_pos_[ci] = pos + off;
        h_pos_[ci] = pos + off;
        off += lens[ci];
    }
    if (pos + off > opt_.max_pos) throw std::runtime_error("forward_multi: position exceeds max_pos");
    ensure_kv(pos + off + 2 * MAX_NT);
    for (int il = 0; il < cfg_.n_layer; ++il) if (is_moe(il) && ehost_[il].stream_dirty) rebuild_stream(il);
    mc_n_ = nck;
    try { run(lens[0]); } catch (...) { mc_n_ = 1; throw; }
    mc_n_ = 1;
    if (adapt_) {
        const int K = cfg_.n_expert_used;
        for (int il = 0; il < cfg_.n_layer; ++il) {
            if (!is_moe(il)) continue;
            auto & sc = ehost_[il].score;
            for (int ci = 0; ci < nck; ++ci)
                for (int i = 0; i < lens[ci] * K; ++i) {
                    const int e = h_ids_[((size_t) il * mc_max_ + ci) * R6 * K + i];
                    if (e >= 0 && e < cfg_.n_expert) sc[e] += prompt_weight_;
                }
        }
        prompt_routed_ = true;
    }
    last_nt_ = lens[nck - 1];
    float best = -INFINITY; int bi = -1;   // the last chunk's last row
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float v = h_res_[(g * MAX_NT) * 2];
        const int idx = ((const int *) h_res_)[(g * MAX_NT) * 2 + 1];
        if (v > best) { best = v; bi = idx; }
    }
    return bi;
}

int Engine6::prefill(const int * tokens, int n, int pos) {
    int next = -1;
    static const int chunk = getenv("HYPER4_CHUNK") ? std::max(1, std::min(R6, atoi(getenv("HYPER4_CHUNK")))) : R6;
    for (int c0 = 0; c0 < n;) {   // up to mc_max_ chunks per layer pass
        int lens[8], nk = 0, tot = 0;
        while (nk < mc_max_ && c0 + tot < n) { lens[nk] = std::min(chunk, n - c0 - tot); tot += lens[nk++]; }
        next = forward_multi(tokens + c0, lens, nk, pos + c0);
        c0 += tot;
    }
    return next;
}

void Engine6::get_logits(int t, std::vector<float> & out) {
    out.resize(cfg_.n_vocab);
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaMemcpy(out.data() + dp->vocab_off, dp->logits + (size_t) t * dp->output.n(), (size_t) dp->output.n() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

} // namespace hyper
