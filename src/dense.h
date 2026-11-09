// Dense weights of every engine: GGUF tensors uploaded to one GPU in the fastest exact form, and their matrix products.
//   Q8_0 (and Q5_0 / Q4_0, converted exactly)   int8 fragments + fp16 scales (tensor-core GEMV / GEMM)
//   Q4_K / Q6_K                                  KQ: the file's bits in fragment order (GEMV; prefill dequantizes slices)
//   BF16 / F16                                   fp16 fragments
//   anything else                                fp16 fragments of the dequantized values (or Q8_0 with requant)
// A matrix stacked from parts of several types (a fused input projection) keeps each run of one type as a segment.
#pragma once
#include "gguf.h"
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

namespace hyper {

struct RowRange { const GTensor * t; int64_t r0, r1; };   // rows [r0, r1) of t (t == nullptr: r1 - r0 zero rows, upload_q8 only)
using ColRanges = std::vector<std::pair<int64_t, int64_t>>;   // column blocks of 32 [b0, b1)
using Alloc = std::function<void *(size_t)>;               // device bytes on the target GPU

struct DWSeg {
    int kind = 0;   // 0: Q8 fragments, 1: KQ, 2: fp16 fragments
    Q8W q8;
    KQW w;
    BF16W f;
    int off = 0;    // first output row
};
struct DW {
    Q8W q8;
    BF16W f;
    bool f16 = false, kq = false;
    KQW kw;
    std::vector<DWSeg> seg;
    int sn = 0, sk = 0;
    bool hcperm = false;   // hyper-connection "up": rows in the order e * hc + s (mixing fused into the GEMV)
    int n() const { return !seg.empty() ? sn : kq ? kw.n : f16 ? f.n : q8.n; }
    int k() const { return !seg.empty() ? sk : kq ? kw.k : f16 ? f.k : q8.k; }
    bool has_kq() const { if (kq) return true; for (auto & g : seg) if (g.kind == 1) return true; return false; }
};

// largest KQ segment (rows x k) uploaded so far: the size of the prefill dequantization scratch (DenseRun::ggs)
size_t dense_kq_max_elems();

Q8W to_device_q8(const Alloc & alloc, int dev, const int8_t * qs, const half * d, int n, int k);
// Q8_0 rows from several tensors (same k), restricted to column blocks
Q8W upload_q8(const Alloc & alloc, int dev, const std::vector<RowRange> & parts, ColRanges cols = {}, int64_t k_full = 0);
// any GGUF type -> fp32 (ggml's reference dequantization), rows [r0, r1)
std::vector<float> to_f32(const GTensor & t, int64_t r0 = 0, int64_t r1 = -1);
// BF16 / F16 rows -> fp16 fragments (exact for F16)
BF16W upload_bf16(const Alloc & alloc, int dev, const GTensor & t);
// requant: always one Q8 matrix (w.q8): Q8_0 parts exactly, the others re-quantized to Q8_0 per block of 32 (callers that
// read w.q8 directly, e.g. DeepSeek's DSpark drafter)
DW upload_dense(const Alloc & alloc, int dev, const std::vector<RowRange> & parts, ColRanges cols = {}, bool requant = false);
// rows [r0, r1) (multiples of 16) of a matrix as a weight of their own, sharing its memory: a plain Q8 / fp16 matrix or one
// with a single segment (the arrays are row-tile-major). Else (mixed segments, hcperm) a DW with n() == 0
DW dense_rows_view(const DW & w, int r0, int r1);
// hyper-connection "up" matrix [hc * n][lr] with rows reordered to e * hc + s (gemv_q8_hcmix); other types: upload_dense
DW upload_hc_up(const Alloc & alloc, int dev, const GTensor & t, int hc);
// part g of ndev of n units of `align`
std::pair<int64_t, int64_t> split(int64_t n, int ndev, int g, int64_t align = 1);
// uneven tensor-parallel shards: device g gets a share proportional to w[g] (largest-remainder rounding of the units);
// empty: equal shards as before (<ENV>_TP_SHARES="1.5,1,1,1" sets them at engine init)
void set_tp_shares(const std::vector<double> & w);
const std::vector<double> & tp_shares();

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

// what dense_mm needs of the GPU it runs on: the stream, cuBLAS, fp16 activation rows (xh: MOE_BULK_ROWS x the widest k)
// and the KQ dequantization scratch (ggs: ggs_elems halves; nullptr without KQ weights)
struct DenseRun {
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    half * xh = nullptr, * ggs = nullptr;
    size_t ggs_elems = 0;
};
// the size of DenseRun::ggs for this process's KQ weights (0: none)
size_t dense_ggs_elems();
// y[t] = W x[t] for t < rows (x stride xs, y stride ys). Decode rows (<= MAX_NT): GEMVs with the fused input norm /
// activation ni; more rows: the input as fp16 and GEMMs (the bulk callers apply ni themselves)
void dense_mm(const DenseRun & r, const DW & W, const float * x, int xs, float * y, int ys, int rows, const NormIn & ni = NormIn{});

} // namespace hyper
