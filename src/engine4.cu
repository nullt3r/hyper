#include "engine4.h"

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

namespace hyper {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

constexpr int CPU_OWNER = 3;

// dense weight: Q8_0 (fragment-ordered int8, the fast path); mixed / other block types as segments of the source parts
// (output rows from `off` on): Q8 fragments (Q8_0, and Q5_0 / Q4_0 converted exactly) or KQ (Q4_K / Q6_K, same bits in
// fragment order); anything else dequantized to fp16 at load
struct DWSeg { bool kq = false; Q8W q8; KQW w; int off = 0; };
struct DW {
    Q8W q8;
    BF16W f;
    bool f16 = false;
    std::vector<DWSeg> seg;
    int sn = 0, sk = 0;
    bool hcperm = false;   // hyper-connection "up": rows in the order e * hc + s (mixing fused into the GEMV)
    int n() const { return !seg.empty() ? sn : f16 ? f.n : q8.n; }
    int k() const { return !seg.empty() ? sk : f16 ? f.k : q8.k; }
};
size_t g_kq_max_elems = 0;   // largest KQ segment (rows x k): prefill dequantization scratch
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
    DW sh_gu, sh_down;
    int n_sh_l = 0;
    MoeDev moe;
    int * owner = nullptr;     // [n_expert]: device or CPU_OWNER
    // prefill streaming: this GPU's share of the layer's CPU experts (cslot range [st_a, st_b)) is copied into a staging
    // buffer; st_slot[e] = index in the share or -1; owner_bulk[e] = the GPU that computes e in a prefill chunk
    int * st_slot = nullptr, * owner_bulk = nullptr;
    int st_a = 0, st_b = 0;
    const uint8_t * st_host_g = nullptr, * st_host_u = nullptr, * st_host_d = nullptr;   // pinned host (whole CPU layer)
    // PLE
    DW ple_key, ple_value;
    float * ple_wk = nullptr, * ple_wq = nullptr, * ple_wc = nullptr, * ple_conv = nullptr, * ple_state = nullptr, * ple_snap = nullptr;
};

struct Engine4::Device {
    int id = 0, g = 0;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t g_main[MAX_NT + 1] = {};
    std::vector<DevLayer> layers;
    // MTP (NextN) block: layer `mtp`, its embedding / hidden norms, eh_proj (per hc stream), its own exit mixer
    DevLayer mtp;
    float * m_enorm = nullptr, * m_hnorm = nullptr, * m_head_norm = nullptr;
    DW m_eh, m_head_down, m_head_up;
    float * mres = nullptr, * mh = nullptr, * ecat = nullptr;   // MTP residual rows, chained hidden (1 row), eh_proj input
    int * mpos = nullptr;
    cudaGraphExec_t g_mtp[MAX_NT + 1] = {}, g_chain = nullptr, g_restore[MAX_NT] = {};
    float * head_norm = nullptr;
    DW head_down, head_up, output;
    int vocab_off = 0;
    // activations [MAX_NT] rows
    float * x = nullptr, * res = nullptr, * xn = nullptr, * gate = nullptr, * lo = nullptr, * inj = nullptr, * mixed = nullptr;
    float * bo = nullptr, * part = nullptr, * big0 = nullptr, * o = nullptr, * attn_part = nullptr, * injp = nullptr;
    unsigned * inj_cnt = nullptr;   // hc norm: per token, the stream blocks' tickets
    float * rlog = nullptr, * wts = nullptr, * sg = nullptr, * shgu = nullptr, * shh = nullptr, * shpart = nullptr;
    float * hexp = nullptr, * yexp = nullptr, * ple_emb = nullptr, * ple_key = nullptr, * ple_val = nullptr, * ple_sc = nullptr;
    std::vector<float *> res_x, ple_x;   // the further chunks of a multi-chunk prefill pass: residual rows, PLE rows
    unsigned * ticket = nullptr;          // fused route + publish: blocks done
    int * pos_x = nullptr;
    float * logits = nullptr, * res2 = nullptr;
    int * ids = nullptr, * pos = nullptr, * counter = nullptr, * order = nullptr, * order_n = nullptr;
    half * xh = nullptr, * p16 = nullptr, * recv = nullptr;   // GEMM input scratch; DMA allreduce own / peers' parts
    half * mix16 = nullptr, * h16 = nullptr;                  // prefill MoE: fp16 token rows, fp16 expert hidden rows
    float * conv_raw = nullptr;                               // prefill: raw conv inputs
    int * egrp = nullptr;                                     // prefill MoE: active experts (expert, start, count)
    float * topk = nullptr;                                   // sampling candidates [MAX_NT][TOPK][2]
    float * iq = nullptr, * ik = nullptr, * iqn = nullptr, * iscores = nullptr;   // QSA: projections, normed queries, scores
    int * ilist = nullptr, * ilist_n = nullptr;                                 // QSA: attended cells per token
    unsigned * ihist = nullptr;                                                 // QSA: score key histograms [rows][65536]
    cublasHandle_t blas = nullptr;
    half * ggs = nullptr;            // native dense weights: fp16 slice for prefill GEMMs
    size_t ggs_elems = 0;
    cudaStream_t cstream = nullptr;                           // prefill expert uploads
    cudaEvent_t ev_up[2] = {}, ev_free[2] = {};
    uint8_t * stage[2] = {};                                  // staging buffers for streamed experts
    size_t stage_bytes = 0;
    cudaEvent_t ev_ar[2] = {};
    cudaStream_t s2 = nullptr;                                // decode: the shared expert, concurrently with the routed ones
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
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
        for (auto & gr : g_mtp) if (gr) cudaGraphExecDestroy(gr);
        for (auto & gr : g_restore) if (gr) cudaGraphExecDestroy(gr);
        if (g_chain) cudaGraphExecDestroy(g_chain);
        for (auto & ev : ev_ar) if (ev) cudaEventDestroy(ev);
        if (blas) cublasDestroy(blas);
        for (auto & ev : ev_up) if (ev) cudaEventDestroy(ev);
        for (auto & ev : ev_free) if (ev) cudaEventDestroy(ev);
        if (cstream) cudaStreamDestroy(cstream);
        if (s2) cudaStreamDestroy(s2);
        if (ev_fork) cudaEventDestroy(ev_fork);
        if (ev_join) cudaEventDestroy(ev_join);
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

namespace {

struct RowRange { const GTensor * t; int64_t r0, r1; };

// bf16 rows -> fp16 fragment-ordered weight
BF16W upload_bf16(const std::function<void *(size_t)> & alloc, int dev, const GTensor & t) {
    if (t.type != GType::BF16) throw std::runtime_error("expected BF16: " + t.name);
    const int n = (int) t.rows(), k = (int) t.ne[0];
    const size_t ntile = (n + 15) / 16;
    std::vector<uint8_t> fq(ntile * (k / 16) * 512);
    repack_bf16_frag((const uint16_t *) t.data, n, k, (size_t) k, fq.data());
    CUDA_CHECK(cudaSetDevice(dev));
    BF16W w; w.n = n; w.k = k;
    void * p = alloc(fq.size());
    CUDA_CHECK(cudaMemcpy(p, fq.data(), fq.size(), cudaMemcpyHostToDevice));
    w.q = (const uint4 *) p;
    return w;
}
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

// any GGUF type -> fp32 (ggml's reference dequantization)
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

// dense rows from several tensors (same k), restricted to column blocks of 32: Q8_0 stays int8, the rest -> fp16
DW upload_dense(const std::function<void *(size_t)> & alloc, int dev, const std::vector<RowRange> & parts, ColRanges cols = {}) {
    DW w;
    bool all_q8 = true;
    for (auto & p : parts) all_q8 &= p.t->type == GType::Q8_0;
    if (all_q8) { w.q8 = upload_q8(alloc, dev, parts, cols); return w; }
    const int64_t k_full = parts[0].t->ne[0];
    if (cols.empty()) cols.push_back({0, k_full / 32});
    static const bool dense_f16 = getenv("HYPER4_DENSE_F16") != nullptr;
    auto seg_type = [](GType t) {   // 0: fp16 fallback, 1: Q8 fragments, 2: KQ
        switch (t) {
            case GType::Q8_0: case GType::Q5_0: case GType::Q4_0: return 1;
            case GType::Q4_K: case GType::Q6_K: return 2;
            default: return 0;
        }
    };
    bool native = !dense_f16;
    for (auto & p : parts) {
        if (!seg_type(p.t->type) || p.t->ne[0] != k_full) { native = false; break; }
        // (Q8 types: blocks of 32 = the column granularity; KQ takes any 32-column blocks)
    }
    if (native) {   // runs of parts with one type -> one segment: the file's blocks restricted to the column ranges
        int kk = 0;
        for (auto & [b0, b1] : cols) kk += (int) (b1 - b0) * 32;
        for (size_t p0 = 0; p0 < parts.size();) {
            const GType ty = parts[p0].t->type;
            size_t p1 = p0 + 1;
            while (p1 < parts.size() && parts[p1].t->type == ty) ++p1;
            const bool kq = ty == GType::Q4_K || ty == GType::Q6_K;
            // KQ: full source rows (the repack gathers the column blocks); Q8 types: the selected blocks
            const size_t bb = gtype_block_bytes(ty), be = gtype_block_elems(ty), rb = kq ? (size_t) k_full / be * bb : kk / be * bb;
            int nr = 0;
            for (size_t q = p0; q < p1; ++q) nr += (int) (parts[q].r1 - parts[q].r0);
            std::vector<uint8_t> buf(rb * nr);
            int rbase = 0;
            for (size_t q = p0; q < p1; ++q) {
                const RowRange & p = parts[q];
                const int pr = (int) (p.r1 - p.r0);
#pragma omp parallel for schedule(static)
                for (int r = 0; r < pr; ++r) {
                    uint8_t * dst = buf.data() + (size_t) (rbase + r) * rb;
                    const uint8_t * src = p.t->data + (size_t) (p.r0 + r) * p.t->row_bytes();
                    if (kq) { memcpy(dst, src, rb); continue; }
                    for (auto & [b0, b1] : cols) {
                        const size_t len = (size_t) (b1 - b0) * 32 / be * bb;
                        memcpy(dst, src + (size_t) b0 * 32 / be * bb, len);
                        dst += len;
                    }
                }
                rbase += pr;
            }
            p0 = p1;
            DWSeg sg;
            sg.off = w.sn;
            if (ty == GType::Q8_0 || ty == GType::Q5_0 || ty == GType::Q4_0) {   // -> int8 + fp16 scale, exactly (q - 16, q - 8 fit in int8)
                const int kb = kk / 32;
                std::vector<int8_t> qs((size_t) nr * kk);
                std::vector<half> d((size_t) nr * kb);
#pragma omp parallel for schedule(static)
                for (int r = 0; r < nr; ++r)
                    for (int b = 0; b < kb; ++b) {
                        const uint8_t * blk = buf.data() + (size_t) r * rb + (size_t) b * bb;
                        int8_t * q = qs.data() + (size_t) r * kk + b * 32;
                        memcpy(&d[(size_t) r * kb + b], blk, 2);
                        if (ty == GType::Q8_0) memcpy(q, blk + 2, 32);
                        else if (ty == GType::Q4_0) for (int l = 0; l < 16; ++l) { q[l] = (int8_t) ((blk[2 + l] & 0xF) - 8); q[l + 16] = (int8_t) ((blk[2 + l] >> 4) - 8); }
                        else {   // Q5_0: d, qh[4], qs[16]
                            uint32_t qh; memcpy(&qh, blk + 2, 4);
                            for (int l = 0; l < 16; ++l) {
                                q[l] = (int8_t) (((blk[6 + l] & 0xF) | (((qh >> l) << 4) & 0x10)) - 16);
                                q[l + 16] = (int8_t) (((blk[6 + l] >> 4) | ((qh >> (l + 12)) & 0x10)) - 16);
                            }
                        }
                    }
                sg.q8 = to_device_q8(alloc, dev, qs.data(), d.data(), nr, kk);
            } else {
                KQHost h;
                const KQ kt = ty == GType::Q4_K ? KQ::Q4K : KQ::Q6K;
                std::vector<int> kbl;
                for (auto & [b0, b1] : cols) for (int64_t b = b0; b < b1; ++b) kbl.push_back((int) b);
                const int dg = repack_kq(kt, buf.data(), rb, nr, kbl, h);
                CUDA_CHECK(cudaSetDevice(dev));
                auto up = [&](const auto & v) -> void * {
                    if (v.empty()) return nullptr;
                    void * pd = alloc(v.size() * sizeof(v[0]));
                    CUDA_CHECK(cudaMemcpy(pd, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice));
                    return pd;
                };
                sg.kq = true;
                sg.w.type = kt; sg.w.n = nr; sg.w.k = kk; sg.w.dg = dg;
                sg.w.lo = (const uint2 *) up(h.lo); sg.w.hi = (const unsigned *) up(h.hi);
                sg.w.scm = (const uint16_t *) up(h.scm); sg.w.sc6 = (const int8_t *) up(h.sc6); sg.w.d = (const half2 *) up(h.d);
                g_kq_max_elems = std::max(g_kq_max_elems, (size_t) nr * kk);
            }
            w.seg.push_back(sg);
            w.sn += nr;
        }
        w.sk = kk;
        static const bool seg_log = getenv("HYPER4_SEGLOG") != nullptr;
        if (seg_log) fprintf(stderr, "hyper4: dense %s: %zu parts -> %zu segments (%d x %d)\n", parts[0].t->name.c_str(), parts.size(), w.seg.size(), w.sn, kk);
        return w;
    }
    std::vector<int64_t> cidx;
    for (auto & [b0, b1] : cols) for (int64_t c = b0 * 32; c < b1 * 32; ++c) cidx.push_back(c);
    const int k = (int) cidx.size();
    int n = 0;
    for (auto & p : parts) n += (int) (p.r1 - p.r0);
    if (getenv("HYPER4_SEGLOG"))
        fprintf(stderr, "hyper4: dense %s (%s, %zu parts, %zu column ranges, first [%lld, %lld)): fp16\n", parts[0].t->name.c_str(),
                gtype_name(parts[0].t->type), parts.size(), cols.size(), (long long) cols[0].first * 32, (long long) cols[0].second * 32);
    std::vector<float> rows((size_t) n * k);
    int row0 = 0;
    for (auto & p : parts) {
        if (p.t->ne[0] != k_full) throw std::runtime_error("upload_dense: k mismatch " + p.t->name);
        const std::vector<float> full = to_f32(*p.t, p.r0, p.r1);
        for (int64_t r = 0; r < p.r1 - p.r0; ++r)
            for (int j = 0; j < k; ++j) rows[(size_t) (row0 + r) * k + j] = full[(size_t) r * k_full + cidx[j]];
        row0 += (int) (p.r1 - p.r0);
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

// hyper-connection "up" matrix [hc * n][lr]: Q8_0 / Q5_0 / Q4_0 rows as int8 + fp16 scales (exact), reordered to e * hc + s
// (the 4 streams of an element in one row tile: gemv_q8_hcmix mixes them in its epilogue); other types load as usual
DW upload_hc_up(const std::function<void *(size_t)> & alloc, int dev, const GTensor & t, int hc) {
    static const bool off = getenv("HYPER4_NO_HCMIX") != nullptr;
    const GType ty = t.type;
    if (off || hc != 4 || !(ty == GType::Q8_0 || ty == GType::Q5_0 || ty == GType::Q4_0) || t.rows() % hc) return upload_dense(alloc, dev, {{&t, 0, t.rows()}});
    const int rows = (int) t.rows(), k = (int) t.ne[0], kb = k / 32, n = rows / hc;
    const size_t bb = gtype_block_bytes(ty);
    std::vector<int8_t> qs((size_t) rows * k);
    std::vector<half> d((size_t) rows * kb);
#pragma omp parallel for schedule(static)
    for (int r2 = 0; r2 < rows; ++r2) {
        const int src = (r2 % hc) * n + r2 / hc;
        for (int b = 0; b < kb; ++b) {
            const uint8_t * blk = t.data + (size_t) src * t.row_bytes() + (size_t) b * bb;
            int8_t * q = qs.data() + (size_t) r2 * k + b * 32;
            memcpy(&d[(size_t) r2 * kb + b], blk, 2);
            if (ty == GType::Q8_0) memcpy(q, blk + 2, 32);
            else if (ty == GType::Q4_0) for (int l = 0; l < 16; ++l) { q[l] = (int8_t) ((blk[2 + l] & 0xF) - 8); q[l + 16] = (int8_t) ((blk[2 + l] >> 4) - 8); }
            else {
                uint32_t qh; memcpy(&qh, blk + 2, 4);
                for (int l = 0; l < 16; ++l) {
                    q[l] = (int8_t) (((blk[6 + l] & 0xF) | (((qh >> l) << 4) & 0x10)) - 16);
                    q[l + 16] = (int8_t) (((blk[6 + l] >> 4) | ((qh >> (l + 12)) & 0x10)) - 16);
                }
            }
        }
    }
    DW w;
    w.q8 = to_device_q8(alloc, dev, qs.data(), d.data(), rows, k);
    w.hcperm = true;
    return w;
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
    nocpu_ = getenv("HYPER4_NOCPU") != nullptr;
    allrows_ = getenv("HYPER4_ALLROWS") != nullptr;
    grouped_decode_ = getenv("HYPER4_GROUPED") != nullptr;   // timing experiment only: CPU experts ignored (wrong output)
    if (getenv("HYPER4_STREAM_MIN")) stream_min_ = atoi(getenv("HYPER4_STREAM_MIN"));
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
    if (cfg_.n_embd > 4096 || cfg_.n_expert_used > MOE_MAX_USED) throw std::runtime_error("hyper4: model dimensions unsupported");
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    opt_.n_devices = std::min(opt_.n_devices, ndev);
    for (int g = 0; g < opt_.n_devices; ++g) {
        auto dev = std::make_unique<Device>();
        dev->id = g; dev->g = g;
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->stream, cudaStreamNonBlocking));
        gemv_init(g);
        for (auto & ev : dev->ev_ar) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->cstream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev->s2, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev->ev_fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev->ev_join, cudaEventDisableTiming));
        for (auto & ev : dev->ev_up) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        for (auto & ev : dev->ev_free) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        if (cublasCreate(&dev->blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        cublasSetStream(dev->blas, dev->stream);
        cublasSetMathMode(dev->blas, CUBLAS_DEFAULT_MATH);   // plain fp32 (no TF32): routing must not move
        devs_.push_back(std::move(dev));
    }
    const int nd = opt_.n_devices, n = cfg_.n_embd;
    if (getenv("HYPER4_MC")) mc_max_ = std::max(1, std::min(8, atoi(getenv("HYPER4_MC"))));
    CUDA_CHECK(cudaHostAlloc(&h_embd_, (size_t) mc_max_ * R4 * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_ple_, (size_t) mc_max_ * R4 * std::max(1, cfg_.ple_n_heads() * cfg_.ple_dim) * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_cpos_, 8 * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_stage_, (size_t) 2 * nd * R4 * n * sizeof(half), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_, sizeof(CpuMoeBulk), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_out_, sizeof(CpuMoeBulkOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_bulk_, 0, sizeof(CpuMoeBulk));
    memset((void *) cpu_bulk_out_, 0, sizeof(CpuMoeBulkOut));
    barrier_ = std::make_unique<Barrier4>(nd);
    CUDA_CHECK(cudaHostAlloc(&h_pos_, 4 * sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_res_, (size_t) nd * MAX_NT * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_topk_, (size_t) nd * MAX_NT * TOPK * 2 * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_membd_, (size_t) R4 * n * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_mres_, (size_t) nd * 4 * sizeof(float), cudaHostAllocPortable));
    mtp_whole_norm_ = getenv("HYPER4_MTP_WHOLE_NORM") != nullptr;
    mtp_full_ = getenv("HYPER4_MTP_FULL") != nullptr;
    if (getenv("HYPER4_MTP_PMIN")) mtp_pmin_ = atof(getenv("HYPER4_MTP_PMIN"));
    const size_t ll = (size_t) 2 * nd * MAX_NT * n / 2;
    CUDA_CHECK(cudaHostAlloc(&ar_ll_, ll * sizeof(uint2), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar_ll_, 0xff, ll * sizeof(uint2));
    const int slots = cfg_.n_layer + 1;   // + the MTP block
    CUDA_CHECK(cudaHostAlloc(&cpu_rec_, (size_t) slots * sizeof(CpuMoeRec), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_out_, (size_t) slots * sizeof(CpuMoeOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_rec_, 0, (size_t) slots * sizeof(CpuMoeRec));
    memset((void *) cpu_out_, 0, (size_t) slots * sizeof(CpuMoeOut));
    if (const char * sp = getenv("HYPER4_STATS")) {   // routing statistics from `hyper4 calib`
        FILE * f = fopen(sp, "rb");
        if (f) {
            int nl = 0, w = 0;
            if (fread(&nl, 4, 1, f) == 1 && fread(&w, 4, 1, f) == 1 && nl == cfg_.n_layer) {
                stats_.assign(nl, std::vector<uint64_t>(w));
                for (auto & s : stats_) if (fread(s.data(), 8, w, f) != (size_t) w) { stats_.clear(); break; }
            }
            fclose(f);
            fprintf(stderr, "hyper4: expert placement from %s (%s)\n", sp, stats_.empty() ? "unreadable, ignored" : "ok");
        }
    }
    cpu_ = std::make_unique<CpuMoe>(opt_.cpu_threads, n, cfg_.n_ff_exp, cfg_.n_expert_used, cpu_rec_, cpu_out_, slots,
                                    cpu_bulk_, cpu_bulk_out_);
    load_weights();
}

void * Engine4::host_huge_alloc(size_t bytes) {
    const size_t H = 2u << 20, sz = (bytes + H - 1) / H * H;
    void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("host_huge_alloc: mmap failed");
    madvise(p, sz, MADV_HUGEPAGE);
    host_bufs_.push_back({p, sz});
    return p;
}

Engine4::~Engine4() {
    for (auto & sn : snaps_) snap_pool_.push_back(sn.h);
    park_.release(snap_pool_);
    for (auto & v : snap_pool_) for (float * p : v) cudaFreeHost(p);
    if (h_topk_) cudaFreeHost(h_topk_);
    if (h_membd_) cudaFreeHost(h_membd_);
    if (h_mres_) cudaFreeHost(h_mres_);
    cpu_.reset();
    for (auto & [p, sz] : host_bufs_) munmap(p, sz);
    devs_.clear();
    for (void * p : {(void *) h_embd_, (void *) h_ple_, (void *) h_pos_, (void *) h_res_, (void *) ar_ll_, (void *) cpu_rec_, (void *) cpu_out_,
                     (void *) h_stage_, (void *) cpu_bulk_, (void *) cpu_bulk_out_})
        if (p) cudaFreeHost(p);
}

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
    // hyper-connection mixers
    L.hca_norm = f32(p + "hc_attn_norm.weight");
    L.hcf_norm = f32(p + "hc_ffn_norm.weight");
    L.hca_inj = f32(p + "hc_attn_inject.weight");
    L.hcf_inj = f32(p + "hc_ffn_inject.weight");
    L.hca_down = q8full(p + "hc_attn_down.weight");
    L.hca_up = upload_hc_up(A, dev.id, *T(p + "hc_attn_up.weight"), c.hc);
    L.hcf_down = q8full(p + "hc_ffn_down.weight");
    L.hcf_up = upload_hc_up(A, dev.id, *T(p + "hc_ffn_up.weight"), c.hc);
    if (L.full) {
        // whole kv groups per GPU (each KV head and its cache live on one GPU only); with fewer kv heads than GPUs,
        // some GPUs do no attention and keep that memory for experts
        const int hd = c.head_dim, group = c.n_head / c.n_head_kv;
        auto [kv0, kv1] = split(c.n_head_kv, nd, g);
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
        const size_t kv = (size_t) L.n_kv_l * opt_.max_pos * hd;
        L.kcache = dev.alloc<half>(kv);
        L.vcache = dev.alloc<half>(kv);
        if (c.compress[il] > 0) {
            if (c.compress[il] != 4 || c.idx_dim != 128 || c.idx_n_head > 4) throw std::runtime_error("QSA: only kpool 4, 128-dim, <= 4 heads");
            L.idx_q = upload_bf16(A, dev.id, src_->need(p + "indexer.q_proj.weight"));
            L.idx_k = upload_bf16(A, dev.id, src_->need(p + "indexer.k_proj.weight"));
            L.idx_qn = f32(p + "indexer.q_norm.weight");
            L.idx_kn = f32(p + "indexer.k_norm.weight");
            L.kraw = dev.alloc<half>((size_t) opt_.max_pos * 128);
            L.kpool = dev.alloc<half>((size_t) (opt_.max_pos / 4 + 1) * 128);
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
        L.router = dev.upload(buf.data(), buf.size());
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
void Engine4::load_experts(int il, const std::vector<int> & quota) {
    const Q4Config & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd;
    const std::string p = "blk." + std::to_string(il) + ".";
    const GTensor & tg = src_->need(p + "ffn_gate_exps.weight"), & tu = src_->need(p + "ffn_up_exps.weight"),
                  & tdn = src_->need(p + "ffn_down_exps.weight");
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
    for (int r = 0; r < std::min(total, E); ++r) {   // weighted round robin: the GPU furthest below its share next
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
    for (auto & dp : devs_) {
        Device & dev = *dp;
        DevLayer & L = il < c.n_layer ? dev.layers[il] : dev.mtp;
        const int g = dev.g;
        std::vector<int> slot(E, -1);
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
        L.moe.gate_bytes = gb; L.moe.down_bytes = db;
        L.moe.ff = ff; L.moe.n = n;
        L.moe.slot = dev.upload(slot.data(), slot.size());
        L.owner = dev.upload(owner.data(), owner.size());
    }
    // CPU experts: compact copy into 2 MB pages (the mmapped file would be read through 4 KB pages)
    CpuExpertLayer cl;
    cl.tg = tg.type; cl.td = tdn.type;
    cl.gate_bytes = gb; cl.down_bytes = db;
    cl.owned.resize(E);
    cl.cslot.assign(E, -1);
    int nc = 0;
    for (int e = 0; e < E; ++e) { cl.owned[e] = owner[e] == CPU_OWNER; if (cl.owned[e]) cl.cslot[e] = nc++; }
    const size_t bytes = (size_t) std::max(nc, 1) * (2 * gb + db);
    uint8_t * buf = (uint8_t *) host_huge_alloc(bytes);
    cl.gate = buf; cl.up = buf + (size_t) nc * gb; cl.down = buf + (size_t) nc * 2 * gb;
#pragma omp parallel for schedule(dynamic)
    for (int e = 0; e < E; ++e) {
        if (cl.cslot[e] < 0) continue;
        const size_t i = (size_t) cl.cslot[e];
        memcpy((uint8_t *) cl.gate + i * gb, tg.data + (size_t) e * gb, gb);
        memcpy((uint8_t *) cl.up + i * gb, tu.data + (size_t) e * gb, gb);
        memcpy((uint8_t *) cl.down + i * db, tdn.data + (size_t) e * db, db);
    }
    collapse_huge(buf, bytes);
    cpu_->set_layer(il, cl);
    // prefill streaming: pin the CPU copy, split it among the GPUs by PCIe bandwidth (x16 : x8 : x16)
    if (opt_.stream_experts && nc > 0) {
        CUDA_CHECK(cudaHostRegister(buf, bytes, cudaHostRegisterPortable));
        std::vector<double> wbw(nd, 2.0);
        if (nd > 1) wbw[1] = 1.0;   // GPU 1 sits on a x8 link on this box
        double tw = 0;
        for (double w : wbw) tw += w;
        std::vector<int> bulk_owner = owner;
        int a = 0;
        double acc = 0;
        for (auto & dp : devs_) {
            DevLayer & L = il < c.n_layer ? dp->layers[il] : dp->mtp;
            acc += wbw[dp->g];
            const int b = dp->g == nd - 1 ? nc : (int) (nc * acc / tw + 0.5);
            L.st_a = a; L.st_b = b;
            std::vector<int> sl(E, -1);
            for (int e = 0; e < E; ++e) if (cl.cslot[e] >= a && cl.cslot[e] < b) { sl[e] = cl.cslot[e] - a; bulk_owner[e] = dp->g; }
            L.st_slot = dp->upload(sl.data(), sl.size());
            L.st_host_g = cl.gate; L.st_host_u = cl.up; L.st_host_d = cl.down;
            dp->stage_bytes = std::max(dp->stage_bytes, (size_t) (b - a) * (2 * gb + db));
            a = b;
        }
        for (auto & dp : devs_) (il < c.n_layer ? dp->layers[il] : dp->mtp).owner_bulk = dp->upload(bulk_owner.data(), bulk_owner.size());
    }
}

void Engine4::load_weights() {
    auto t0 = std::chrono::steady_clock::now();
    const Q4Config & c = cfg_;
    const int nd = opt_.n_devices, n = c.n_embd, hcn = c.hc_dim();
    for (auto & dp : devs_) {
        Device & dev = *dp;
        const int g = dev.g;
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
        { const std::vector<float> v = to_f32(gguf_->need("output_hc_norm.weight")); dev.head_norm = dev.upload(v.data(), v.size()); }
        { const GTensor * t = &gguf_->need("output_hc_down.weight"); dev.head_down = upload_dense(A, dev.id, {{t, 0, t->rows()}}); }
        dev.head_up = upload_hc_up(A, dev.id, gguf_->need("output_hc_up.weight"), c.hc);
        auto [o0, o1] = split(c.n_vocab, nd, g);
        dev.vocab_off = (int) o0;
        dev.output = upload_dense(A, dev.id, {{&gguf_->need("output.weight"), o0, o1}});
        // activations
        const int conv_dim = c.conv_dim();
        dev.big_stride = conv_dim + c.ssm_d_inner + 2 * c.ssm_dt_rank + 2 * c.n_head * c.head_dim + 2 * c.n_head_kv * c.head_dim;
        const int R = R4, K = c.n_expert_used;
        dev.x = dev.alloc<float>((size_t) R * n);
        dev.res = dev.alloc<float>((size_t) R * hcn);
        {   // (HYPER4_MC_ALLOC: buffers for more chunks than used: equal VRAM, so the expert placement matches another setting)
            const int mc_alloc = std::max(mc_max_, getenv("HYPER4_MC_ALLOC") ? atoi(getenv("HYPER4_MC_ALLOC")) : 0);
            for (int ci = 1; ci < mc_alloc; ++ci) {
                dev.res_x.push_back(dev.alloc<float>((size_t) R * hcn));
                if (c.ple_layer >= 0) dev.ple_x.push_back(dev.alloc<float>((size_t) R * std::max(1, c.ple_n_heads() * c.ple_dim)));
            }
            dev.pos_x = dev.alloc<int>(8);
        }
        dev.res2 = dev.alloc<float>((size_t) R * hcn);
        dev.xn = dev.alloc<float>((size_t) R * hcn);
        dev.gate = dev.alloc<float>((size_t) R * hcn);
        dev.lo = dev.alloc<float>((size_t) R * c.hc_lr);
        dev.inj = dev.alloc<float>((size_t) R * 4);
        dev.injp = dev.alloc<float>((size_t) MAX_NT * c.hc * 4);
        dev.inj_cnt = dev.alloc<unsigned>(MAX_NT);
        CUDA_CHECK(cudaMemset(dev.inj_cnt, 0, MAX_NT * sizeof(unsigned)));
        dev.mixed = dev.alloc<float>((size_t) R * n);
        dev.bo = dev.alloc<float>((size_t) R * n);
        dev.part = dev.alloc<float>((size_t) R * n);
        dev.big0 = dev.alloc<float>((size_t) R * dev.big_stride);
        dev.o = dev.alloc<float>((size_t) R * std::max(c.ssm_d_inner, c.n_head * c.head_dim));
        {
            size_t mx = 0;
            for (int r = 1; r <= R; ++r) mx = std::max(mx, attn_part_floats(c.n_head, 1, r, c.head_dim));
            dev.attn_part = dev.alloc<float>(mx);
        }
        dev.rlog = dev.alloc<float>((size_t) R * (c.n_expert + 1));
        dev.ticket = dev.alloc<unsigned>(1);
        CUDA_CHECK(cudaMemset(dev.ticket, 0, sizeof(unsigned)));
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
        dev.logits = dev.alloc<float>((size_t) (allrows_ ? R : MAX_NT) * dev.output.n());
        dev.xh = dev.alloc<half>((size_t) R * std::max(hcn, c.n_ff_shexp));
        if (g_kq_max_elems) {   // prefill scratch for KQ dense weights (bigger segments go in slices)
            dev.ggs_elems = std::max<size_t>(std::min<size_t>(g_kq_max_elems, (size_t) 32 << 20), (size_t) 16 * 16384);
            dev.ggs = dev.alloc<half>(dev.ggs_elems);
        }
        dev.p16 = dev.alloc<half>((size_t) R * n);
        dev.recv = dev.alloc<half>((size_t) std::max(1, nd - 1) * R * n);
        dev.order = dev.alloc<int>((size_t) R * K);
        dev.order_n = dev.alloc<int>(2);
        dev.topk = dev.alloc<float>((size_t) MAX_NT * TOPK * 2);
        dev.egrp = dev.alloc<int>((size_t) 3 * c.n_expert);
        dev.mix16 = dev.alloc<half>((size_t) R * n);
        dev.conv_raw = dev.alloc<float>((size_t) R * c.conv_dim());
        dev.iq = dev.alloc<float>((size_t) R * c.idx_n_head * 128);
        dev.ik = dev.alloc<float>((size_t) R * 128);
        dev.iqn = dev.alloc<float>((size_t) R * c.idx_n_head * 128);
        dev.iscores = dev.alloc<float>((size_t) QSA_SCORE_ROWS * (opt_.max_pos / 4 + 4));
        if (opt_.max_pos / 4 + 4 <= 65536 + 2048) dev.ihist = dev.alloc<unsigned>((size_t) QSA_SCORE_ROWS * 65536);
        dev.ilist = dev.alloc<int>((size_t) R * QSA_LIST);
        dev.ilist_n = dev.alloc<int>(R);
        dev.h16 = dev.alloc<half>((size_t) R * K * c.n_ff_exp);
        dev.pos = dev.alloc<int>(1);
        dev.mpos = dev.alloc<int>(1);
        if (mtp_g_) {
            dev.mres = dev.alloc<float>((size_t) R * hcn);
            dev.mh = dev.alloc<float>((size_t) hcn);
            dev.ecat = dev.alloc<float>((size_t) R * c.hc * 2 * n);
        }
        dev.counter = dev.alloc<int>(1);
    }
    // experts fill what is left on each GPU (minus a runtime reserve), capped at gpu_expert_frac of every layer
    {
        size_t eb = 0, eb_max = 0;   // bytes of one expert: average over the layers, largest layer
        for (int il = 0; il < c.n_layer; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const size_t b = (gguf_->need(p + "ffn_gate_exps.weight").nbytes * 2 + gguf_->need(p + "ffn_down_exps.weight").nbytes) / c.n_expert;
            eb += b;
            eb_max = std::max(eb_max, b);
        }
        eb /= c.n_layer;
        std::vector<int> quota(nd);
        std::vector<size_t> freeb(nd);
        for (int g = 0; g < nd; ++g) {
            CUDA_CHECK(cudaSetDevice(devs_[g]->id));
            size_t tot = 0;
            CUDA_CHECK(cudaMemGetInfo(&freeb[g], &tot));
        }
        // staging for prefill streaming: two buffers of this GPU's share of a layer's CPU experts (fixed point: the
        // share depends on how many experts stay on the CPU, which depends on the quotas)
        std::vector<double> stage_est(nd, 0.0);
        for (int it = 0; it < 3; ++it) {
            int tq = 0;
            for (int g = 0; g < nd; ++g) {
                const double cap = std::max(0.0, (double) freeb[g] - opt_.vram_reserve_gib * 1073741824.0 - stage_est[g]);
                quota[g] = std::min((int) (cap / eb / (c.n_layer + (mtp_g_ ? 1 : 0))), (int) (c.n_expert * opt_.gpu_expert_frac / nd + 0.999));
                tq += quota[g];
            }
            if (!opt_.stream_experts) break;
            const double cold = (std::max(0, c.n_expert - tq) + 2) * (double) eb_max;   // the layer with the biggest experts
            for (int g = 0; g < nd; ++g) stage_est[g] = 2.0 * cold * (nd > 1 && g == 1 ? 1.0 : 2.0) / (2.0 * nd - (nd > 1 ? 1.0 : 0.0));
        }
        int tq = 0;
        for (int q : quota) tq += q;
        fprintf(stderr, "hyper4: experts per layer on GPUs:");
        for (int q : quota) fprintf(stderr, " %d", q);
        fprintf(stderr, " (%.0f%% of %d), the rest on the CPU\n", 100.0 * tq / c.n_expert, c.n_expert);
        for (int il = 0; il < c.n_layer; ++il) load_experts(il, quota);
        if (mtp_g_) { src_ = mtp_g_.get(); load_experts(c.n_layer, quota); src_ = gguf_.get(); }
        if (opt_.stream_experts)
            for (auto & dp : devs_) {
                for (auto & sb : dp->stage) sb = dp->alloc<uint8_t>(std::max<size_t>(dp->stage_bytes, 1));
                fprintf(stderr, "hyper4: device %d streams prefill experts through 2 x %.0f MiB\n", dp->id, dp->stage_bytes / 1048576.0);
            }
    }
    for (auto & dp : devs_) fprintf(stderr, "hyper4: device %d holds %.2f GiB\n", dp->id, dp->used / 1073741824.0);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "hyper4: weights loaded in %.1f s (%d GPUs)\n", s, nd);
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

// kind 0: main model; 1: MTP block over (token t+1, main hidden row t); 2: MTP chained on its own last hidden row
void Engine4::record_main(int gi, int nt, int kind) {
    const Q4Config & c = cfg_;
    Device & d = *devs_[gi];
    cudaStream_t s = d.stream;
    const int n = c.n_embd, hc = c.hc, hcn = c.hc_dim(), lr = c.hc_lr, K = c.n_expert_used, bs = d.big_stride;
    const int nd = opt_.n_devices;
    const float eps = c.rms_eps;
    bool bulk = nt > MAX_NT;   // (an MTP pass continues its last row in decode mode)
    // MTP rows that only enter the block's cache (its K / V and indexer keys depend on its input alone): prompt rows, and
    // all but the last of the kept verification rows; the last one alone goes on through the FFN to the head
    const bool cache_only = kind == 1 && mtp_cache_only_ && !mtp_full_;
    const bool last_only = kind == 1 && !mtp_cache_only_ && !mtp_full_ && nt > 1;
    // matmul: decode GEMV (split-K for few rows) or, for a prefill chunk, fp16 conversion + tiled tensor-core GEMM
    auto mm = [&](const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni = NormIn{}) {
        if (!W.seg.empty()) {   // segments: Q8 fragments or KQ (prefill: dequantized to fp16 in slices + cuBLAS)
            if (rows <= MAX_NT) {
                for (auto & g : W.seg) {
                    if (g.kq) gemv_kq(g.w, x, xs, y + g.off, ys, rows, s, ni);
                    else gemv_q8(g.q8, x, xs, y + g.off, ys, nullptr, rows, s, ni);
                }
                return;
            }
            to_half(x, xs, nullptr, W.k(), 0.0f, d.xh, rows, s);
            const float one = 1.0f, zero = 0.0f;
            for (auto & g : W.seg) {
                if (!g.kq) { gemm_q8(g.q8, d.xh, rows, y + g.off, ys, nullptr, s); continue; }
                const int step = std::max(16, (int) (d.ggs_elems / g.w.k) / 16 * 16);
                for (int r0 = 0; r0 < g.w.n; r0 += step) {
                    const int r1 = std::min(g.w.n, r0 + step);
                    deq_kq_f16(g.w, r0, r1, d.ggs, s);
                    if (cublasGemmEx(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, r1 - r0, rows, g.w.k, &one, d.ggs, CUDA_R_16F, g.w.k, d.xh, CUDA_R_16F,
                                     g.w.k, &zero, y + g.off + r0, CUDA_R_32F, ys, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                        throw std::runtime_error("mm: cublasGemmEx failed");
                }
            }
            return;
        }
        if (rows <= MAX_NT) {   // (ni: fused input activation; bulk callers apply it separately)
            if (W.f16) gemv_bf16(W.f, x, xs, y, ys, nullptr, rows, s, ni);
            else gemv_q8(W.q8, x, xs, y, ys, nullptr, rows, s, ni);
            return;
        }
        to_half(x, xs, nullptr, W.k(), 0.0f, d.xh, rows, s);
        if (W.f16) gemm_f16(W.f, d.xh, rows, y, ys, nullptr, s);
        else gemm_q8(W.q8, d.xh, rows, y, ys, nullptr, s);
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
    // short chunks: the CPU computes its experts (the MTP block's FFN never runs over a whole chunk)
    const bool streaming = bulk && opt_.stream_experts && nt >= stream_min_ && (!kind || mtp_full_);
    if (streaming) {
        if (!kind && d.layers[0].owner_bulk) { upload_stage(d, 0); if (c.n_layer > 1) upload_stage(d, 1); }
        if (kind && d.mtp.owner_bulk) upload_stage(d, c.n_layer);
    }
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
    int dcall = 0;
    auto allreduce = [&] {
        CUDA_CHECK(cudaMemsetAsync(d.bo, 0, (size_t) nt * n * sizeof(float), s));
        if (!bulk) { allreduce_add_ll16(d.bo, d.part, ar_ll_, d.g, nd, nt * n, d.counter, call++, s, nullptr); return; }
        // copy engines: own fp16 part -> pinned staging -> peers (ordered by events; every device's recording thread
        // meets the others at the barrier between its upload and its downloads)
        const int par = dcall++ & 1;
        const size_t N = (size_t) nt * n;
        auto stage = [&](int g) { return h_stage_ + ((size_t) par * nd + g) * R4 * n; };
        to_half(d.part, n, nullptr, n, 0.0f, d.p16, nt, s);
        CUDA_CHECK(cudaMemcpyAsync(stage(d.g), d.p16, N * sizeof(half), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaEventRecord(d.ev_ar[par], s));
        barrier_->wait();
        int j = 0;
        for (int p = 0; p < nd; ++p) {
            if (p == d.g) continue;
            CUDA_CHECK(cudaStreamWaitEvent(s, devs_[p]->ev_ar[par], 0));
            CUDA_CHECK(cudaMemcpyAsync(d.recv + (size_t) j * R4 * n, stage(p), N * sizeof(half), cudaMemcpyHostToDevice, s));
            ++j;
        }
        add_parts(d.bo, d.p16, d.recv, (size_t) R4 * n, nd - 1, (int) N, s);
    };
    // hyper-connection mixer: res -> mixed (and the inject logits)
    auto hc_mix = [&](const float * norm, const DW & down, const DW & up, const float * inj, const float * res = nullptr, int rows = -1) {
        if (!res) res = R;
        if (rows < 0) rows = nt;
        const bool finj = inj && rows <= MAX_NT;   // decode: the injection dots ride along the norm kernel
        hc_norm(res, norm, d.xn, n, hc, eps, rows, s, finj ? inj : nullptr, d.injp, d.inj, d.inj_cnt);
        mm(down, d.xn, hcn, d.lo, lr, rows);
        if (inj && !finj) f32mm(inj, hc, hcn, d.xn, hcn, d.inj, 4, rows);
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
        mtp_prep(d.x, d.m_enorm, kind == 1 ? (mtp_src_ > 0 ? d.res_x[mtp_src_ - 1] : d.res) : d.mh, kind == 1 ? hcn : 0, d.m_hnorm, eps, n, hc, mtp_whole_norm_, d.ecat, nt, s);
        const int blk = std::max(1, R4 * std::max(hcn, c.n_ff_shexp) / (2 * n));   // rows the fp16 scratch holds
        for (int r0 = 0; r0 < nt * hc; r0 += blk) {
            const int rr = std::min(blk, nt * hc - r0);
            mm(d.m_eh, d.ecat + (size_t) r0 * 2 * n, 2 * n, d.mres + (size_t) r0 * n, n, rr);
        }
    }
    const int il0 = kind ? c.n_layer : 0, il1 = kind ? c.n_layer + 1 : c.n_layer;
    const bool snap = !kind && !bulk && nt > 1 && mtp_g_;   // verification: keep the state after every row for rollback
    for (int il = il0; il < il1; ++il)
    for (int ci = 0; ci < nck; ++ci) {
        nt = ck[ci].nt; R = ck[ci].R; P = ck[ci].P; pos_host = ck[ci].pos; ple_emb = ck[ci].ple;
        const bool last_ck = ci == nck - 1;   // (the streamed experts' staging is released after the last chunk)
        DevLayer & L = kind ? d.mtp : d.layers[il];
        if (L.ple) {
            const int pe = c.ple_n_heads() * c.ple_dim;
            mm(L.ple_key, ple_emb, pe, d.ple_key, hcn, nt);
            mm(L.ple_value, ple_emb, pe, d.ple_val, n, nt);
            ple_apply(R, d.ple_key, d.ple_val, L.ple_wk, L.ple_wq, L.ple_wc, L.ple_conv, L.ple_state, snap ? L.ple_snap : nullptr, n, hc, c.ple_conv,
                      c.ple_ngram, eps, nt, d.ple_sc, s);
        }
        if (L.ple) dbg("ple", il, R, (size_t) nt * hcn);
        // ---- token mixer ----
        hc_mix(L.hca_norm, L.hca_down, L.hca_up, L.hca_inj);
        dbg("hc_mix_attn", il, d.mixed, (size_t) nt * n);
        // decode: independent work on a second stream (indexer || q/k/v + attention prep, alpha/beta || the GDN input
        // projection, shared expert || routed experts): the small kernels' fixed latencies overlap. The second stream's
        // split-K GEMVs use their own scratch slot. HYPER4_SERIAL_SHEXP: one stream
        static const bool serial_shexp = getenv("HYPER4_SERIAL_SHEXP") != nullptr;
        const bool par = !bulk && !serial_shexp;
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
        if (L.full && L.n_head_l == 0) {   // no kv head on this GPU: contributes nothing to the attention output
            CUDA_CHECK(cudaMemsetAsync(d.part, 0, (size_t) nt * n * sizeof(float), s));
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
                    idx_prep_pool(d.iq, d.ik, L.idx_qn, d.iqn, L.kraw, L.kpool, L.idx_kn, P, c.idx_n_head, c.n_rot, c.rope_base, eps, s);
                else {
                    idx_prep(d.iq, d.ik, L.idx_qn, d.iqn, L.kraw, P, c.idx_n_head, c.n_rot, c.rope_base, eps, nt, s);
                    idx_pool(L.kraw, L.kpool, L.idx_kn, P, nt, c.n_rot, c.rope_base, eps, s);
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
                          c.n_rot, c.rope_base, eps, nt, s);
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
                          c.n_rot, c.rope_base, eps, nt, s);
            if (cache_only) continue;
            const int ostride = L.n_head_l * c.head_dim;
            if (bulk && dense_chunk)
                attn_prefill(d.big0, bs, L.kcache, L.vcache, d.o, ostride, P, opt_.max_pos, L.n_head_l, L.head_off,
                             c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s);
            else if (getenv("HYPER4_OLDATTN"))
                attn_decode(d.big0, bs, L.kcache, L.vcache, d.o, ostride, P, opt_.max_pos, L.n_head_l, L.n_kv_l, L.head_off,
                            c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s);
            else
                attn_split(d.big0, bs, L.kcache, L.vcache, d.attn_part, d.o, ostride, P, opt_.max_pos, L.n_head_l, L.n_kv_l,
                           L.head_off, c.n_head / c.n_head_kv, L.kv_off, c.head_dim, 1.0f / sqrtf((float) c.head_dim), nt, s,
                           qsa ? d.ilist : nullptr, QSA_LIST, qsa ? d.ilist_n : nullptr);
            mm(L.wo, d.o, ostride, d.part, n, nt);
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
            gated_norm_sigmoid(d.o, ostride, d.big0 + z_off, bs, L.ssm_norm, L.n_v_l, dv, eps, nt, s);
            mm(L.wout, d.o, ostride, d.part, n, nt);
        }
        if (cache_only) continue;
        dbg(L.full ? "attn_part" : "gdn_part", il, d.part, (size_t) nt * n);
        if (bulk) { allreduce(); hc_combine(R, d.bo, d.inj, 4, n, hc, nt, s); }
        else allreduce_hc_ll16(R, d.inj, n, hc, d.part, ar_ll_, d.g, nd, nt * n, d.counter, call++, s);
        if (last_only) {   // the last row moves to row 0 and continues alone, in decode mode
            CUDA_CHECK(cudaMemcpyAsync(R, R + (size_t) (nt - 1) * hcn, (size_t) hcn * sizeof(float), cudaMemcpyDeviceToDevice, s));
            nt = 1;
            bulk = false;
        }
        // ---- MoE ----
        hc_mix(L.hcf_norm, L.hcf_down, L.hcf_up, L.hcf_inj);
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
        f32mm(L.router, c.n_expert + 1, n, d.mixed, n, d.rlog, c.n_expert + 1, nt);
        dbg("router", il, d.rlog, (size_t) nt * (c.n_expert + 1));
        const bool stream = streaming && L.owner_bulk;
        static const bool sep_pub = getenv("HYPER4_SEP_PUBLISH") != nullptr;
        // decode on GPU 0: routing and the CPU record in one launch
        const bool fused_pub = !bulk && d.g == 0 && !sep_pub &&
            moe_route_publish(d.rlog, c.n_expert + 1, c.n_expert, K, d.ids, d.wts, d.sg, nt, &cpu_rec_[il].seq, &cpu_rec_[il].nt,
                              &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0], d.mixed, n, n, d.counter,
                              (unsigned) il, d.ticket, s);
        if (!fused_pub) moe_route(d.rlog, c.n_expert + 1, c.n_expert, K, d.ids, d.wts, d.sg, nt, s);
        dbg("route_w", il, d.wts, (size_t) nt * K);
        if (d.g == 0 && !stream && !fused_pub) {
            if (bulk) moe_publish(&cpu_bulk_->seq, &cpu_bulk_->nt, &cpu_bulk_->ids[0][0], &cpu_bulk_->wts[0][0], &cpu_bulk_->x[0][0],
                                  d.mixed, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
            else moe_publish(&cpu_rec_[il].seq, &cpu_rec_[il].nt, &cpu_rec_[il].ids[0][0], &cpu_rec_[il].wts[0][0], &cpu_rec_[il].x[0][0],
                             d.mixed, n, n, d.ids, d.wts, K, nt, d.counter, (unsigned) il, s);
        }
        if (!par) shexp();
        if (bulk) {   // local pairs grouped by expert: consecutive blocks share the expert's weights in L2
            moe_order(L.moe, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
            to_half(d.mixed, n, nullptr, n, 0.0f, d.mix16, nt, s);
            moe_gemm_gate_up(L.moe, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
            moe_gemm_down(L.moe, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
            if (stream) {   // this GPU's share of the CPU experts, uploaded into staging buffer il % 2
                const int sb = il & 1;
                MoeDev ms = L.moe;
                const int cnt = L.st_b - L.st_a;
                ms.gate = d.stage[sb];
                ms.up = d.stage[sb] + (size_t) cnt * L.moe.gate_bytes;
                ms.down = d.stage[sb] + (size_t) 2 * cnt * L.moe.gate_bytes;
                ms.slot = L.st_slot;
                CUDA_CHECK(cudaStreamWaitEvent(s, d.ev_up[sb], 0));
                moe_order(ms, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
                moe_gemm_gate_up(ms, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
                moe_gemm_down(ms, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
                if (last_ck) {
                    CUDA_CHECK(cudaEventRecord(d.ev_free[sb], s));
                    if (il + 2 < c.n_layer) upload_stage(d, il + 2);
                }
            }
        } else if (nt > 1 && grouped_decode_) {   // verification rows: each local expert read once (grouped GEMM)
            moe_order(L.moe, d.ids, nt * K, c.n_expert, d.order, d.order_n, s, d.egrp);
            to_half(d.mixed, n, nullptr, n, 0.0f, d.mix16, nt, s);
            moe_gemm_gate_up(L.moe, d.mix16, n, K, d.order, d.order_n, d.egrp, c.n_expert, d.h16, s);
            moe_gemm_down(L.moe, d.h16, d.order, d.order_n, d.egrp, c.n_expert, d.wts, d.yexp, s);
        } else {
            moe_gate_up(L.moe, d.mixed, n, d.ids, K, d.hexp, nt, s);
            moe_down(L.moe, d.hexp, d.ids, d.wts, K, d.yexp, nt, s);
        }
        if (par) join();
        dbg("shexp", il, d.shpart, (size_t) nt * n);
        dbg("experts", il, d.yexp, (size_t) nt * K * n);
        const volatile unsigned * cflag = d.g == 0 && !nocpu_ && !stream ? (bulk ? &cpu_bulk_out_->seq : &cpu_out_[il].seq) : nullptr;
        moe_reduce(d.shpart, d.sg, d.yexp, K, d.part, n, nt, d.ids, stream ? L.owner_bulk : L.owner, d.g, CPU_OWNER, cflag,
                   bulk ? &cpu_bulk_out_->y[0][0] : &cpu_out_[il].y[0][0], d.counter, (unsigned) il, s);
        dbg("moe_part", il, d.part, (size_t) nt * n);
        if (bulk) { allreduce(); hc_combine(R, d.bo, d.inj, 4, n, hc, nt, s); }
        else allreduce_hc_ll16(R, d.inj, n, hc, d.part, ar_ll_, d.g, nd, nt * n, d.counter, call++, s);
        dbg("l_last", il, R, (size_t) nt * hcn);
    }
    if (kind) {   // drafts: the last row only; its residual feeds the next chained step
        if (cache_only) return;
        CUDA_CHECK(cudaMemcpyAsync(d.mh, d.mres + (size_t) (nt - 1) * hcn, (size_t) hcn * sizeof(float), cudaMemcpyDeviceToDevice, s));
        hc_mix(d.m_head_norm, d.m_head_down, d.m_head_up, nullptr, d.mres + (size_t) (nt - 1) * hcn, 1);
        mm(d.output, d.mixed, n, d.logits, d.output.n(), 1);
        argmax_pairs(d.logits, d.output.n(), d.output.n(), d.vocab_off, d.wts, 1, s);
        max_sumexp(d.logits, d.output.n(), d.wts + 2, s);   // (the draft's probability)
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

// copy this GPU's share of layer il's CPU experts into staging buffer il % 2 (on the copy stream, after the buffer's
// previous user has finished)
void Engine4::upload_stage(Device & d, int il) {
    DevLayer & L = il < cfg_.n_layer ? d.layers[il] : d.mtp;
    const int sb = il & 1, cnt = L.st_b - L.st_a;
    CUDA_CHECK(cudaStreamWaitEvent(d.cstream, d.ev_free[sb], 0));
    if (cnt > 0) {
        const size_t gb = L.moe.gate_bytes, db = L.moe.down_bytes;
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb], L.st_host_g + (size_t) L.st_a * gb, (size_t) cnt * gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) cnt * gb, L.st_host_u + (size_t) L.st_a * gb, (size_t) cnt * gb,
                                   cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) 2 * cnt * gb, L.st_host_d + (size_t) L.st_a * db, (size_t) cnt * db,
                                   cudaMemcpyHostToDevice, d.cstream));
    }
    CUDA_CHECK(cudaEventRecord(d.ev_up[sb], d.cstream));
}

// roll the recurrent state (GDN conv + state, PLE conv) back to the snapshot after row keep-1 of the last verification
void Engine4::record_restore(int gi, int keep) {
    const Q4Config & c = cfg_;
    Device & d = *devs_[gi];
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

void Engine4::build_graphs() {
    for (int gi = 0; gi < (int) devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        auto capture = [&](const std::function<void()> & rec) {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
            rec();
            CUDA_CHECK(cudaStreamEndCapture(d.stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            return exec;
        };
        for (int nt = 1; nt <= MAX_NT; ++nt) d.g_main[nt] = capture([&] { record_main(gi, nt, 0); });
        if (mtp_g_) {
            for (int nt = 1; nt <= MAX_NT; ++nt) d.g_mtp[nt] = capture([&] { record_main(gi, nt, 1); });
            d.g_chain = capture([&] { record_main(gi, 1, 2); });
            for (int keep = 1; keep < MAX_NT; ++keep) d.g_restore[keep] = capture([&] { record_restore(gi, keep); });
        }
    }
    graphs_ready_ = true;
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

// token embeddings and the PLE n-gram hash rows, for positions pos..pos+nt-1
void Engine4::embed(const int * tokens, int nt, int pos, int chunk) {
    const Q4Config & c = cfg_;
    if ((int) seq_.size() < pos + nt) seq_.resize(pos + nt, -1);
    for (int t = 0; t < nt; ++t) seq_[pos + t] = tokens[t];
    embed_tok(tokens, nt, h_embd_ + (size_t) chunk * R4 * c.n_embd);
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
                pt_tr->to_float(pt.data + row * pt.row_bytes(), out + h * dim, dim);
            }
        }
    }
}

// launch one forward of `kind` over nt rows on every device and wait (hang diagnostics after 10 s)
void Engine4::run(int kind, int nt) {
    const bool bulk = nt > MAX_NT;
    if (!graphs_ready_ && !debug_) build_graphs();
    ++fwd_counter_;
    if (kind == 3) --fwd_counter_;   // restore graphs do not count
    // (MTP: the FFN runs for the last row only, in decode mode, and not at all for cache-only rows; HYPER4_MTP_FULL: as main)
    const bool mtp_full = kind == 0 || mtp_full_;
    if (kind <= 2 && !(bulk && opt_.stream_experts && nt >= stream_min_ && mtp_full) && !(kind == 1 && mtp_cache_only_ && !mtp_full_)) {
        std::vector<int> slots;
        if (kind == 0) for (int i = 0; i < cfg_.n_layer; ++i) slots.push_back(i);
        else slots.push_back(cfg_.n_layer);
        cpu_->expect(fwd_counter_, slots, bulk && mtp_full);
    }
    if (bulk) {   // record straight into the streams, one host thread per device (barriers inside the allreduces)
        std::vector<std::thread> th;
        std::vector<std::string> err(devs_.size());
        for (int gi = 0; gi < (int) devs_.size(); ++gi)
            th.emplace_back([&, gi] {
                try { CUDA_CHECK(cudaSetDevice(devs_[gi]->id)); record_main(gi, nt, kind); }
                catch (const std::exception & ex) { err[gi] = ex.what(); }
            });
        for (auto & t : th) t.join();
        for (auto & m : err) if (!m.empty()) throw std::runtime_error(m);
    } else {
        for (int gi = 0; gi < (int) devs_.size(); ++gi) {
            auto & dp = devs_[gi];
            CUDA_CHECK(cudaSetDevice(dp->id));
            if ((debug_ && kind < 3) || (kind == 1 && mtp_cache_only_)) { record_main(gi, nt, kind); continue; }   // (no graph)
            cudaGraphExec_t ex = kind == 0 ? dp->g_main[nt] : kind == 1 ? dp->g_mtp[nt] : kind == 2 ? dp->g_chain : dp->g_restore[nt];
            if (!ex) throw std::runtime_error("run: graph missing (kind " + std::to_string(kind) + ", nt " + std::to_string(nt) + ")");
            CUDA_CHECK(cudaGraphLaunch(ex, dp->stream));
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
                fprintf(stderr, "hyper4: forward %u (kind %d) stuck on device %d; cpu: %s\\n", fwd_counter_, kind, dp->id, cpu_->state().c_str());
            }
            std::this_thread::yield();
        }
    }
}

std::vector<int> Engine4::forward(const int * tokens, int nt, int pos) {
    if (nt < 1 || nt > R4) throw std::runtime_error("forward: bad token count");
    if (pos + nt > opt_.max_pos) throw std::runtime_error("forward: position exceeds max_pos");
    const bool bulk = nt > MAX_NT;
    static const bool hprof = getenv("HYPER4_HOSTPROF") != nullptr;   // host time per decode forward (GPUs idle meanwhile)
    static double h_embed = 0, h_run = 0, h_gap = 0; static long h_n = 0;
    static auto h_last = std::chrono::steady_clock::now();
    const auto th0 = std::chrono::steady_clock::now();
    embed(tokens, nt, pos);
    h_pos_[0] = pos;
    const auto th1 = std::chrono::steady_clock::now();
    run(0, nt);
    if (hprof && !bulk) {
        const auto th2 = std::chrono::steady_clock::now();
        h_gap += std::chrono::duration<double>(th0 - h_last).count();
        h_embed += std::chrono::duration<double>(th1 - th0).count();
        h_run += std::chrono::duration<double>(th2 - th1).count();
        h_last = th2;
        if (++h_n % 200 == 0) {
            fprintf(stderr, "hyper4: per decode forward: embed %.1f us, run %.1f us, caller between forwards %.1f us\n",
                    h_embed / h_n * 1e6, h_run / h_n * 1e6, h_gap / h_n * 1e6);
        }
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

bool Engine4::multi_ok(const int * lens, int nck) const {
    bool ok = nck >= 1 && nck <= mc_max_ && opt_.stream_experts;
    for (int ci = 0; ok && ci < nck; ++ci) ok = lens[ci] > MAX_NT && lens[ci] <= R4 && lens[ci] >= stream_min_;
    return ok;
}

int Engine4::forward_multi(const int * tokens, const int * lens, int nck, int pos) {
    int total = 0;
    for (int ci = 0; ci < nck; ++ci) total += lens[ci];
    if (!multi_ok(lens, nck) || nck == 1) {   // one chunk at a time
        int r = -1;
        for (int ci = 0, off = 0; ci < nck; off += lens[ci++]) r = forward(tokens + off, lens[ci], pos + off).back();
        mc_n_ = 1;
        return r;
    }
    if (pos + total > opt_.max_pos) throw std::runtime_error("forward_multi: position exceeds max_pos");
    for (int ci = 0, off = 0; ci < nck; off += lens[ci++]) {
        embed(tokens + off, lens[ci], pos + off, ci);
        mc_nt_[ci] = lens[ci]; mc_pos_[ci] = pos + off; h_cpos_[ci] = pos + off;
    }
    h_pos_[0] = pos;
    mc_n_ = nck;
    try { run(0, lens[0]); } catch (...) { mc_n_ = 1; throw; }
    mc_n_ = 1;   // (later bulk forwards are single-chunk unless set again; mc_nt_/mc_pos_ stay for mtp_draft_chunk)
    mc_last_ = nck;
    last_nt_ = lens[nck - 1];
    float best = -INFINITY; int bi = -1;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float v = h_res_[g * MAX_NT * 2];
        const int idx = ((const int *) h_res_)[g * MAX_NT * 2 + 1];
        if (v > best) { best = v; bi = idx; }
    }
    return bi;
}

// MTP pass over chunk ci of the last multi-chunk forward (tokens/nt/pos: that chunk's MTP inputs)
int Engine4::mtp_draft_chunk(const int * tokens, int nt, int pos, int ci, bool need_draft) {
    if (ci < 0 || ci >= std::max(1, mc_last_)) throw std::runtime_error("mtp_draft_chunk: bad chunk");
    mtp_src_ = ci;
    try { const int r = mtp_draft(tokens, nt, pos, need_draft); mtp_src_ = 0; return r; } catch (...) { mtp_src_ = 0; throw; }
}

int Engine4::mtp_result() {
    float best = -INFINITY; int bi = -1;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const float v = h_mres_[g * 4];
        const int idx = ((const int *) h_mres_)[g * 4 + 1];
        if (v > best) { best = v; bi = idx; }
    }
    double z = 0;   // softmax normalizer relative to the best logit: [2] = slice max, [3] = sum exp(x - slice max)
    for (size_t g = 0; g < devs_.size(); ++g) z += h_mres_[g * 4 + 3] * std::exp((double) h_mres_[g * 4 + 2] - best);
    mtp_p_ = z > 0 ? 1.0 / z : 0.0;
    return bi;
}

// MTP over (tokens[t], main hidden row t) at positions pos + t; returns the draft after the last row (need_draft false:
// the rows only enter the block's cache, -1)
int Engine4::mtp_draft(const int * tokens, int nt, int pos, bool need_draft) {
    embed_tok(tokens, nt, h_membd_);
    h_pos_[1] = pos;
    mtp_cache_only_ = !need_draft;
    try { run(1, nt); } catch (...) { mtp_cache_only_ = false; throw; }
    mtp_cache_only_ = false;
    return need_draft ? mtp_result() : -1;
}

// MTP over (token, its own last hidden row) at pos
int Engine4::mtp_chain(int token, int pos) {
    embed_tok(&token, 1, h_membd_);
    h_pos_[1] = pos;
    run(2, 1);
    return mtp_result();
}

// ---------------- prompt cache: recurrent-state snapshots (GDN conv + state, PLE conv history) ----------------
size_t Engine4::snap_floats(int gi) const {
    const Q4Config & c = cfg_;
    size_t n = 0;
    for (auto & L : devs_[gi]->layers) {
        if (!L.full) n += (size_t) (c.ssm_conv - 1) * L.conv_ch + (size_t) L.n_v_l * c.ssm_d_state * c.head_v_dim();
        if (L.ple) n += (size_t) (c.ple_conv - 1) * c.ple_ngram * c.hc_dim();
    }
    return n;
}

void Engine4::snap_copy(Snap & sn, bool to_host) {
    const Q4Config & c = cfg_;
    for (size_t gi = 0; gi < devs_.size(); ++gi) {
        Device & d = *devs_[gi];
        CUDA_CHECK(cudaSetDevice(d.id));
        float * hp = sn.h[gi];
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
std::vector<ConvPark<Engine4::Snap>::Dev> Engine4::park_devs() const {
    std::vector<ConvPark<Snap>::Dev> v;
    const size_t hd = (size_t) cfg_.head_dim;
    for (auto & dp : devs_) {
        ConvPark<Snap>::Dev pd;
        pd.id = dp->id;
        pd.stream = dp->stream;
        auto attn = [&](const DevLayer & L) {
            if (L.kcache && L.n_kv_l > 0) {
                pd.spans.push_back({L.kcache, hd * sizeof(half), opt_.max_pos * hd * sizeof(half), L.n_kv_l, 1});
                pd.spans.push_back({L.vcache, hd * sizeof(half), opt_.max_pos * hd * sizeof(half), L.n_kv_l, 1});
            }
            if (L.kraw) pd.spans.push_back({L.kraw, 128 * sizeof(half), 0, 1, 1});
            if (L.kpool) pd.spans.push_back({L.kpool, 128 * sizeof(half), 0, 1, 4});
        };
        for (auto & L : dp->layers) attn(L);
        attn(dp->mtp);
        v.push_back(std::move(pd));
    }
    return v;
}

void Engine4::take_snapshot(int pos) {
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

// candidates of row t from every device's top-K, then temperature / top-k / min-p / top-p
int Engine4::sample_row(int t, const SamplingParams & sp) {
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

std::vector<int> Engine4::generate(const std::vector<int> & prompt, int n_gen, bool spec_req, GenStats * stats,
                                   const std::function<bool(int)> & on_token, const SamplingParams & sp) {
    const bool spec = spec_req && mtp_g_;
    if (prompt.empty()) throw std::runtime_error("generate: empty prompt");
    if ((int) prompt.size() + 8 > opt_.max_pos) throw std::runtime_error("generate: prompt longer than the context");
    using clk = std::chrono::steady_clock;
    const bool sampling = sp.temp > 0.0f;
    rng_.seed(sp.seed ? sp.seed : std::random_device{}());
    const int P = (int) prompt.size();
    // prompt cache: KV / indexer entries stay valid for the common prefix L; the recurrent state comes from the
    // latest snapshot at s <= L - 1
    int s = 0;
    if (opt_.prompt_cache) {
        const int L = park_.resolve(prompt, hist_, snaps_, snap_pool_, park_devs(), "hyper4");   // (see park.h)
        for (size_t i = 0; i < snaps_.size();)
            if (snaps_[i].pos > L) { snap_pool_.push_back(snaps_[i].h); snaps_.erase(snaps_.begin() + i); } else ++i;
        Snap * best = nullptr;
        for (auto & sn : snaps_) if (sn.pos <= std::min(L - 1, P - 1) && (!best || sn.pos > best->pos)) best = &sn;
        if (best) { s = best->pos; snap_copy(*best, false); }
    }
    if (s == 0) reset();
    seq_.assign(prompt.begin(), prompt.begin() + s);   // PLE n-gram history
    hist_.assign(prompt.begin(), prompt.begin() + s);
    std::vector<int> snap_at;
    if (opt_.prompt_cache) {   // last two message starts always, older ones >= 256 tokens apart, every 4096 tokens
        std::vector<int> msg;
        for (int q = s + 1; q < P; ++q) if (prompt[q] == snap_token_) msg.push_back(q);
        // snapshots at every full-chunk boundary (prefill runs in whole R4-token chunks: every MoE chunk pays a fixed cost,
        // one pass over the CPU-resident experts) and at the last message start (edits / regenerations resume there);
        // generation adds its own snapshots, so the end of a previous answer is covered too
        // (multi-chunk prefill: every mc_max_ chunks, so a whole group runs as one layer-major pass)
        for (int q = s + R4 * mc_max_; q < P; q += R4 * mc_max_) snap_at.push_back(q);
        if (!msg.empty() && msg.back() - s >= 64) snap_at.push_back(msg.back());
        std::sort(snap_at.begin(), snap_at.end());
        snap_at.erase(std::unique(snap_at.begin(), snap_at.end()), snap_at.end());
    }
    GenStats st;
    const int K = opt_.n_draft;
    std::vector<int> drafts(std::max(1, K));
    auto tp = clk::now();
    int next = -1;
    size_t si = 0;
    for (int c0 = s; c0 < P;) {
        // the stretch up to the next snapshot point in equal chunks of at most R4 (no short tail chunk inside it)
        while (si < snap_at.size() && snap_at[si] <= c0) ++si;
        const int seg = (si < snap_at.size() ? snap_at[si] : P) - c0, nch = (seg + R4 - 1) / R4;
        {   // the whole stretch as one multi-chunk pass (all chunks streaming), MTP drafts per chunk afterwards
            int lens[8], nk = 0, tot = 0;   // (the stretch's equal chunks, the first mc_max_ of them)
            for (int rest = seg, left = nch; rest > 0 && nk < mc_max_; --left) { lens[nk] = (rest + left - 1) / left; tot += lens[nk]; rest -= lens[nk++]; }
            if (nk > 1 && multi_ok(lens, nk)) {
                const int end = c0 + tot;
                next = forward_multi(&prompt[c0], lens, nk, c0);
                if (sampling) next = sample_row(0, sp);
                if (spec)
                    for (int ci = 0, q = c0; ci < nk; q += lens[ci++]) {
                        std::vector<int> mt(lens[ci]);
                        for (int j = 0; j < lens[ci]; ++j) mt[j] = q + 1 + j < P ? prompt[q + 1 + j] : next;
                        const bool last = ci == nk - 1 && end == P;   // (only the prompt's last row drafts)
                        const int d0 = mtp_draft_chunk(mt.data(), lens[ci], q, ci, last);
                        if (last) drafts[0] = d0;
                    }
                if (si < snap_at.size() && snap_at[si] == end) take_snapshot(end);
                if (prefill_cb_) prefill_cb_(end, P, s);
                c0 = end;
                continue;
            }
        }
        const int end = c0 + (seg + nch - 1) / nch;
        const int len = end - c0;
        next = forward(&prompt[c0], len, c0)[len - 1];
        if (sampling) next = sample_row(len <= MAX_NT ? len - 1 : 0, sp);
        if (spec) {   // MTP consumes (t_{q+1}, h_q) at q for the chunk; the last pair uses the predicted next token
            std::vector<int> mt(len);
            for (int j = 0; j < len; ++j) mt[j] = c0 + 1 + j < P ? prompt[c0 + 1 + j] : next;
            const int d0 = mtp_draft(mt.data(), len, c0, end == P);
            if (end == P) drafts[0] = d0;
        }
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
    // generated tokens get recurrent-state snapshots too (every 1024 positions and at the end): the next request repeats
    // this answer in its prompt and resumes close to where the re-rendered history first differs
    int snap_mark = p / 1024;
    bool state_ok = true;
    auto gen_snapshot = [&] {
        if (!opt_.prompt_cache || p / 1024 == snap_mark) return;
        snap_mark = p / 1024;
        take_snapshot(p);
    };
    if (!spec) {
        while (emit(next) && p + 1 < opt_.max_pos) {
            next = forward(&next, 1, p++)[0];
            if (sampling) next = sample_row(0, sp);
            st.steps++;
            gen_snapshot();
        }
    } else {
        // drafts per step: with a probability floor (mtp_pmin_ > 0), while the MTP block's probability of the drafted run
        // stays above it; otherwise adapted to the recent acceptance, kc = floor(avg + 1.5) in [1, K], avg = moving average
        // of accepted drafts per step (HYPER4_FIXED_DRAFT: always K)
        static const bool fixed_k = getenv("HYPER4_FIXED_DRAFT") != nullptr;
        const double pmin = mtp_pmin_;
        int kc = K;
        double avg_acc = K;
        auto extend = [&](int base) {   // drafts[0] just made by the MTP row at base: the chained ones, the count to verify
            double pc = mtp_p_;
            if (pc < pmin) return 0;
            int k = 1;
            for (; k < kc; ++k) {
                drafts[k] = mtp_chain(drafts[k - 1], base + k);
                pc *= mtp_p_;
                if (pc < pmin) break;
            }
            return k;
        };
        auto ta = clk::now();
        int nd = extend(p - 1);   // (the first draft came with the prompt)
        st.t_mtp += since(ta);
        int cur = next;           // token at position p, not yet in the main model
        std::vector<int> in(K + 1), mt(K + 1);
        while (!stop && p + K + 1 < opt_.max_pos) {
            in[0] = cur;
            for (int j = 0; j < nd; ++j) in[j + 1] = drafts[j];
            ta = clk::now();
            std::vector<int> a = forward(in.data(), nd + 1, p);
            st.t_main += since(ta);
            st.steps++;
            st.drafted += nd;
            // a draft is kept iff the token sampled (or argmax) at its row equals it: exact plain sampling
            int m = 0;
            if (sampling) { while (m < nd && (a[m] = sample_row(m, sp)) == drafts[m]) ++m; if (m == nd) a[nd] = sample_row(nd, sp); }
            else while (m < nd && a[m] == drafts[m]) ++m;
            st.accepted += m;
            if (emit(cur)) for (int j = 0; j < m; ++j) if (!emit(drafts[j])) break;
            if (stop) { state_ok = false; break; }   // (the state holds rows past the end of the output)
            if (m < nd) {                           // keep rows 0..m of the verified block
                ta = clk::now();
                run(3, m + 1);
                st.t_restore += since(ta);
            }
            for (int i = 0; i <= m; ++i) mt[i] = i < m ? drafts[i] : a[m];
            if (!fixed_k && pmin <= 0) {
                avg_acc = 0.9 * avg_acc + 0.1 * m;
                kc = std::max(1, std::min(K, (int) (avg_acc + 1.5)));
            }
            ta = clk::now();
            drafts[0] = mtp_draft(mt.data(), m + 1, p);   // positions p..p+m with main hidden rows 0..m
            nd = extend(p + m);
            st.t_mtp += since(ta);
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

void Engine4::save_expert_stats(const std::string & path) {
    const int nl = cfg_.n_layer, w = 1024;
    std::vector<std::vector<uint64_t>> tot(nl, std::vector<uint64_t>(w, 0));
    for (int l = 0; l < nl; ++l) {
        for (int e = 0; e < w; ++e) {
            if (l < (int) stats_.size() && e < (int) stats_[l].size()) tot[l][e] += stats_[l][e];
            if (l < (int) cpu_->counts.size() && e < (int) cpu_->counts[l].size()) tot[l][e] += cpu_->counts[l][e];
        }
    }
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    fwrite(&nl, 4, 1, f); fwrite(&w, 4, 1, f);
    for (auto & c : tot) fwrite(c.data(), 8, w, f);
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}

int Engine4::prefill(const int * tokens, int n, int pos) {
    int next = -1;
    static const int chunk = getenv("HYPER4_CHUNK") ? std::max(1, std::min(R4, atoi(getenv("HYPER4_CHUNK")))) : R4;
    for (int c0 = 0; c0 < n;) {
        int lens[8], nk = 0, q = c0;   // up to mc_max_ full chunks per layer-major pass
        while (nk < mc_max_ && q < n) { lens[nk] = std::min(chunk, n - q); q += lens[nk++]; }
        if (nk > 1 && !multi_ok(lens, nk)) nk = 1;
        if (nk > 1) next = forward_multi(tokens + c0, lens, nk, pos + c0);
        else next = forward(tokens + c0, lens[0], pos + c0)[lens[0] - 1];
        for (int ci = 0; ci < nk; ++ci) c0 += lens[ci];
    }
    return next;
}

void Engine4::get_logits(int t, std::vector<float> & out) {
    out.resize(cfg_.n_vocab);
    for (auto & dp : devs_) {
        CUDA_CHECK(cudaSetDevice(dp->id));
        CUDA_CHECK(cudaMemcpy(out.data() + dp->vocab_off, dp->logits + (size_t) t * dp->output.n(), (size_t) dp->output.n() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

} // namespace hyper
