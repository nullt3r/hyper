#include "dense.h"
#include "cuda_check.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace hyper {


namespace {

size_t g_kq_max_elems = 0;

// the first of these environment variables that is set (the engines' old per-engine names keep working)
const char * env_any(std::initializer_list<const char *> names) {
    for (const char * n : names) if (const char * v = getenv(n)) return v;
    return nullptr;
}

// one 32-block of Q8_0 / Q5_0 / Q4_0 -> int8 values (the scale stays the block's fp16 d): exact
void q8_block(GType ty, const uint8_t * blk, int8_t * q) {
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

BF16W to_device_f16(const Alloc & alloc, int dev, const float * rows, int n, int k) {
    const size_t ntile = (n + 15) / 16;
    std::vector<uint8_t> fq(ntile * (k / 16) * 512);
    repack_f32_frag(rows, n, k, (size_t) k, fq.data());
    CUDA_CHECK(cudaSetDevice(dev));
    BF16W w; w.n = n; w.k = k;
    void * pd = alloc(fq.size());
    CUDA_CHECK(cudaMemcpy(pd, fq.data(), fq.size(), cudaMemcpyHostToDevice));
    w.q = (const uint4 *) pd;
    return w;
}

// rows of parts (same k_full) restricted to the column ranges, as fp32
std::vector<float> gather_f32(const std::vector<RowRange> & parts, size_t p0, size_t p1, const ColRanges & cols, int64_t k_full, int & n, int & k) {
    std::vector<int64_t> cidx;
    for (auto & [b0, b1] : cols) for (int64_t c = b0 * 32; c < b1 * 32; ++c) cidx.push_back(c);
    k = (int) cidx.size();
    n = 0;
    for (size_t q = p0; q < p1; ++q) n += (int) (parts[q].r1 - parts[q].r0);
    std::vector<float> rows((size_t) n * k);
    int row0 = 0;
    for (size_t q = p0; q < p1; ++q) {
        const RowRange & p = parts[q];
        if (p.t->ne[0] != k_full) throw std::runtime_error("upload_dense: k mismatch " + p.t->name);
        const std::vector<float> full = to_f32(*p.t, p.r0, p.r1);
#pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < p.r1 - p.r0; ++r)
            for (int j = 0; j < k; ++j) rows[(size_t) (row0 + r) * k + j] = full[(size_t) r * k_full + cidx[j]];
        row0 += (int) (p.r1 - p.r0);
    }
    return rows;
}

} // namespace

size_t dense_kq_max_elems() { return g_kq_max_elems; }
size_t dense_ggs_elems() {
    return g_kq_max_elems ? std::max<size_t>(std::min<size_t>(g_kq_max_elems, (size_t) 32 << 20), (size_t) 16 * 16384) : 0;
}

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

Q8W upload_q8(const Alloc & alloc, int dev, const std::vector<RowRange> & parts, ColRanges cols, int64_t k_full) {
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

std::vector<float> to_f32(const GTensor & t, int64_t r0, int64_t r1) {
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

BF16W upload_bf16(const Alloc & alloc, int dev, const GTensor & t) {
    if (t.type != GType::BF16 && t.type != GType::F16) throw std::runtime_error("expected BF16/F16: " + t.name);
    const int n = (int) t.rows(), k = (int) t.ne[0];
    if (t.type == GType::F16) {   // f16 -> f32 -> f16: exact
        const std::vector<float> f = to_f32(t);
        return to_device_f16(alloc, dev, f.data(), n, k);
    }
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

DW upload_dense(const Alloc & alloc, int dev, const std::vector<RowRange> & parts, ColRanges cols, bool requant) {
    DW w;
    bool all_q8 = true;
    for (auto & p : parts) all_q8 &= p.t->type == GType::Q8_0;
    if (all_q8) { w.q8 = upload_q8(alloc, dev, parts, cols); return w; }
    const int64_t k_full = parts[0].t->ne[0];
    if (cols.empty()) cols.push_back({0, k_full / 32});
    // native segments (Q8_0 / Q5_0 / Q4_0 as int8 fragments, Q4_K / Q6_K as KQ, BF16 / F16 as fp16): the file's values
    // exactly (Maya-L's Q6_K attention / shared experts were fp16 copies before: 2.4x the bytes per decode GEMV)
    static const bool dense_f16 = env_any({"HYPER_DENSE_F16", "HYPER4_DENSE_F16", "HYPER5_DENSE_F16"}) != nullptr;   // (A/B)
    auto seg_type = [](GType t) {   // -1: no native form, else DWSeg::kind
        switch (t) {
            case GType::Q8_0: case GType::Q5_0: case GType::Q4_0: return 0;
            case GType::Q4_K: case GType::Q6_K: return 1;
            case GType::BF16: case GType::F16: return 2;
            default: return -1;
        }
    };
    bool native = !dense_f16 && !requant;
    for (auto & p : parts) if (seg_type(p.t->type) < 0 || p.t->ne[0] != k_full) { native = false; break; }
    if (const char * only = native ? env_any({"HYPER_KQ_ONLY", "HYPER5_KQ_ONLY"}) : nullptr) {   // (debugging: native only for these names)
        bool any = false;
        for (const char * q = only; *q;) {
            const char * e = q; while (*e && *e != ',') ++e;
            if (parts[0].t->name.find(std::string(q, e)) != std::string::npos) any = true;
            q = *e ? e + 1 : e;
        }
        native = any;
    }
    if (env_any({"HYPER_SEGLOG", "HYPER4_SEGLOG", "HYPER5_SEGLOG"}))
        fprintf(stderr, "hyper: dense %s (%s, %zu parts, cols %zu [%lld, %lld)) -> %s\n", parts[0].t->name.c_str(), gtype_name(parts[0].t->type),
                parts.size(), cols.size(), (long long) cols[0].first * 32, (long long) cols[0].second * 32,
                native ? "native" : requant ? "q8" : "fp16");
    if (native) {   // runs of parts with one type -> one segment: the file's blocks restricted to the column ranges
        int kk = 0;
        for (auto & [b0, b1] : cols) kk += (int) (b1 - b0) * 32;
        for (size_t p0 = 0; p0 < parts.size();) {
            const GType ty = parts[p0].t->type;
            size_t p1 = p0 + 1;
            while (p1 < parts.size() && parts[p1].t->type == ty) ++p1;
            DWSeg sg;
            sg.off = w.sn;
            sg.kind = seg_type(ty);
            int nr = 0;
            for (size_t q = p0; q < p1; ++q) nr += (int) (parts[q].r1 - parts[q].r0);
            if (sg.kind == 2) {   // fp16 fragments (BF16 -> fp16 rounds; F16 exact)
                int n2 = 0, k2 = 0;
                const std::vector<float> rows = gather_f32(parts, p0, p1, cols, k_full, n2, k2);
                sg.f = to_device_f16(alloc, dev, rows.data(), n2, k2);
            } else {
                const bool kq = sg.kind == 1;
                // KQ: full source rows (the repack gathers the column blocks); Q8 types: the selected blocks
                const size_t bb = gtype_block_bytes(ty), be = gtype_block_elems(ty), rb = kq ? (size_t) k_full / be * bb : kk / be * bb;
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
                if (!kq) {   // -> int8 + fp16 scale, exactly (q - 16, q - 8 fit in int8)
                    const int kb = kk / 32;
                    std::vector<int8_t> qs((size_t) nr * kk);
                    std::vector<half> d((size_t) nr * kb);
#pragma omp parallel for schedule(static)
                    for (int r = 0; r < nr; ++r)
                        for (int b = 0; b < kb; ++b) {
                            const uint8_t * blk = buf.data() + (size_t) r * rb + (size_t) b * bb;
                            memcpy(&d[(size_t) r * kb + b], blk, 2);
                            q8_block(ty, blk, qs.data() + (size_t) r * kk + b * 32);
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
                    sg.w.type = kt; sg.w.n = nr; sg.w.k = kk; sg.w.dg = dg;
                    sg.w.lo = (const uint2 *) up(h.lo); sg.w.hi = (const unsigned *) up(h.hi);
                    sg.w.scm = (const uint16_t *) up(h.scm); sg.w.sc6 = (const int8_t *) up(h.sc6); sg.w.d = (const half2 *) up(h.d);
                    g_kq_max_elems = std::max(g_kq_max_elems, (size_t) nr * kk);
                }
            }
            w.seg.push_back(sg);
            w.sn += nr;
            p0 = p1;
        }
        w.sk = kk;
        return w;
    }
    if (requant) {   // one Q8 matrix: Q8_0 parts' blocks exactly, the others re-quantized per block of 32
        int64_t n = 0, kb = 0;
        for (auto & [b0, b1] : cols) kb += b1 - b0;
        for (auto & p : parts) n += p.r1 - p.r0;
        const int64_t k = kb * 32;
        std::vector<int8_t> qs((size_t) n * k);
        std::vector<half> d((size_t) n * kb);
        int64_t row0 = 0;
        for (size_t q = 0; q < parts.size(); ++q) {
            const RowRange & p = parts[q];
            if (p.t->ne[0] != k_full) throw std::runtime_error("upload_dense: k mismatch " + p.t->name);
            if (p.t->type == GType::Q8_0) {
                const size_t rb = p.t->row_bytes();
#pragma omp parallel for schedule(static)
                for (int64_t r = p.r0; r < p.r1; ++r) {
                    const int64_t orow = row0 + (r - p.r0);
                    int64_t j = 0;
                    for (auto & [b0, b1] : cols)
                        for (int64_t b = b0; b < b1; ++b, ++j) {
                            const uint8_t * blk = p.t->data + (size_t) r * rb + (size_t) b * 34;
                            memcpy(&d[(size_t) orow * kb + j], blk, 2);
                            memcpy(&qs[(size_t) orow * k + j * 32], blk + 2, 32);
                        }
                }
            } else {
                int pn = 0, pk = 0;
                const std::vector<float> rows = gather_f32(parts, q, q + 1, cols, k_full, pn, pk);
#pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < (int64_t) pn * kb; ++i) {
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
        w.q8 = to_device_q8(alloc, dev, qs.data(), d.data(), (int) n, (int) k);
        return w;
    }
    int n = 0, k = 0;
    const std::vector<float> rows = gather_f32(parts, 0, parts.size(), cols, k_full, n, k);
    w.f16 = true;
    w.f = to_device_f16(alloc, dev, rows.data(), n, k);
    return w;
}

DW upload_hc_up(const Alloc & alloc, int dev, const GTensor & t, int hc) {
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
            memcpy(&d[(size_t) r2 * kb + b], blk, 2);
            q8_block(ty, blk, qs.data() + (size_t) r2 * k + b * 32);
        }
    }
    DW w;
    w.q8 = to_device_q8(alloc, dev, qs.data(), d.data(), rows, k);
    w.hcperm = true;
    return w;
}

DW dense_rows_view(const DW & w, int r0, int r1) {
    DW v;
    if (r0 % 16 || w.hcperm || w.seg.size() > 1 || r1 > w.n() || r0 >= r1) return v;
    const int t0 = r0 / 16, nr = r1 - r0, k = w.k(), kb = k / 32;
    auto q8v = [&](const Q8W & q) { Q8W o = q; o.n = nr; o.q = q.q + (size_t) t0 * kb * 32; o.s = q.s + (size_t) t0 * kb * 16; return o; };
    auto f16v = [&](const BF16W & f) { BF16W o = f; o.n = nr; o.q = f.q + (size_t) t0 * (k / 16) * 32; return o; };
    auto kqv = [&](const KQW & m) {
        KQW o = m; o.n = nr;
        o.lo = m.lo + (size_t) t0 * kb * 32; o.hi = m.hi ? m.hi + (size_t) t0 * kb * 32 : nullptr;
        o.scm = m.scm ? m.scm + (size_t) t0 * kb * 16 : nullptr; o.sc6 = m.sc6 ? m.sc6 + (size_t) t0 * kb * 32 : nullptr;   // (two per k block)
        o.d = m.d + (size_t) t0 * (kb / m.dg) * 16;
        return o;
    };
    if (!w.seg.empty()) {
        const DWSeg & g = w.seg[0];
        DWSeg s; s.kind = g.kind; s.off = 0;
        if (g.kind == 0) s.q8 = q8v(g.q8); else if (g.kind == 1) s.w = kqv(g.w); else s.f = f16v(g.f);
        v.seg.push_back(s); v.sn = nr; v.sk = k;
    } else if (w.kq) { v.kq = true; v.kw = kqv(w.kw); }
    else if (w.f16) { v.f16 = true; v.f = f16v(w.f); }
    else v.q8 = q8v(w.q8);
    return v;
}

static std::vector<double> g_tp_shares;
void set_tp_shares(const std::vector<double> & w) { g_tp_shares = w; }
const std::vector<double> & tp_shares() { return g_tp_shares; }

std::pair<int64_t, int64_t> split(int64_t n, int ndev, int g, int64_t align) {
    const int64_t units = n / align;
    if ((int) g_tp_shares.size() != ndev) return {units * g / ndev * align, units * (g + 1) / ndev * align};
    // weighted: quotas, their floors, the remaining units to the largest fractional parts (ties: the lower device)
    double tot = 0;
    for (double w : g_tp_shares) tot += w;
    std::vector<int64_t> cnt(ndev);
    std::vector<std::pair<double, int>> rem;
    int64_t given = 0;
    for (int d = 0; d < ndev; ++d) {
        const double q = tot > 0 ? units * g_tp_shares[d] / tot : (double) units / ndev;
        cnt[d] = (int64_t) q;
        given += cnt[d];
        rem.push_back({q - (double) cnt[d], d});
    }
    std::stable_sort(rem.begin(), rem.end(), [](const std::pair<double, int> & a, const std::pair<double, int> & b) { return a.first > b.first; });
    for (int64_t i = 0; given < units; ++i, ++given) cnt[rem[i % ndev].second]++;
    int64_t lo = 0;
    for (int d = 0; d < g; ++d) lo += cnt[d];
    return {lo * align, (lo + cnt[g]) * align};
}

void dense_mm(const DenseRun & r, const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni) {
    const cudaStream_t s = r.s;
    if (W.kq) {   // (a KQ LM head: a few rows)
        for (int r0 = 0; r0 < rows; r0 += MAX_NT)
            gemv_kq(W.kw, x + (size_t) r0 * xs, xs, y + (size_t) r0 * ys, ys, std::min(MAX_NT, rows - r0), s, ni);
        return;
    }
    if (!W.seg.empty()) {   // segments: Q8 fragments, KQ (prefill: dequantized to fp16 in slices + cuBLAS) or fp16
        if (rows <= MAX_NT) {
            for (auto & g : W.seg) {
                if (g.kind == 1) gemv_kq(g.w, x, xs, y + g.off, ys, rows, s, ni);
                else if (g.kind == 2) gemv_bf16(g.f, x, xs, y + g.off, ys, nullptr, rows, s, ni);
                else gemv_q8(g.q8, x, xs, y + g.off, ys, nullptr, rows, s, ni);
            }
            return;
        }
        // short chunks (an agent's tool output, a chat turn): the KQ segments as GEMVs of 8 rows (the MMA's token width) --
        // dequantizing the whole matrix to fp16 for cuBLAS reads + writes ~4.4 bytes a weight, a GEMV pass 0.41 (break-even
        // ~90 rows); the bulk callers have applied the input norm already (ni not used, as for the GEMMs)
        static const int kq_gemv_rows = [] { const char * v = env_any({"HYPER_KQ_GEMV_ROWS", "HYPER5_KQ_GEMV_ROWS"}); return v ? atoi(v) : 96; }();
        const bool kq_loop = rows <= kq_gemv_rows;
        if (kq_loop)
            for (auto & g : W.seg)
                if (g.kind == 1)
                    for (int r0 = 0; r0 < rows; r0 += 8)
                        gemv_kq(g.w, x + (size_t) r0 * xs, xs, y + (size_t) r0 * ys + g.off, ys, std::min(8, rows - r0), s);
        to_half(x, xs, nullptr, W.k(), 0.0f, r.xh, rows, s);
        const float one = 1.0f, zero = 0.0f;
        for (auto & g : W.seg) {
            if (g.kind == 0) { gemm_q8(g.q8, r.xh, rows, y + g.off, ys, nullptr, s); continue; }
            if (g.kind == 2) { gemm_f16(g.f, r.xh, rows, y + g.off, ys, nullptr, s); continue; }
            if (kq_loop) continue;
            if (!r.ggs) throw std::runtime_error("dense_mm: KQ weights without dequantization scratch");
            const int step = std::max(16, (int) (r.ggs_elems / g.w.k) / 16 * 16);
            for (int r0 = 0; r0 < g.w.n; r0 += step) {
                const int r1 = std::min(g.w.n, r0 + step);
                deq_kq_f16(g.w, r0, r1, r.ggs, s);
                if (cublasGemmEx(r.blas, CUBLAS_OP_T, CUBLAS_OP_N, r1 - r0, rows, g.w.k, &one, r.ggs, CUDA_R_16F, g.w.k, r.xh, CUDA_R_16F,
                                 g.w.k, &zero, y + g.off + r0, CUDA_R_32F, ys, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("dense_mm: cublasGemmEx failed");
            }
        }
        return;
    }
    if (rows <= MAX_NT) {   // (ni: fused input activation; bulk callers apply it separately)
        if (W.f16) gemv_bf16(W.f, x, xs, y, ys, nullptr, rows, s, ni);
        else gemv_q8(W.q8, x, xs, y, ys, nullptr, rows, s, ni);
        return;
    }
    to_half(x, xs, nullptr, W.k(), 0.0f, r.xh, rows, s);
    if (W.f16) gemm_f16(W.f, r.xh, rows, y, ys, nullptr, s);
    else gemm_q8(W.q8, r.xh, rows, y, ys, nullptr, s);
}

} // namespace hyper
