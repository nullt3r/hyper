// Decode (single token) kernels. All activations are fp32; weights are repacked at load time.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace hyper {

// Q8_0 weight repacked for aligned vector loads: qs[n][k] int8 (row-major), d[n][k/32] fp16
struct Q8W {
    const int8_t * qs = nullptr;
    const half * d = nullptr;
    int n = 0, k = 0;   // out features, in features
};
// BF16 weight [n][k] row-major
struct BF16W {
    const __nv_bfloat16 * w = nullptr;
    int n = 0, k = 0;
};

// y[n] = W x  (+ optional residual add: y[n] = add[n] + W x)
void gemv_q8(const Q8W & W, const float * x, float * y, const float * add, cudaStream_t s);
void gemv_bf16(const BF16W & W, const float * x, float * y, const float * add, cudaStream_t s);

// y = rmsnorm(x) * w   (n elements, single row)
void rmsnorm(const float * x, const float * w, float * y, int n, float eps, cudaStream_t s);
// token embedding lookup from a raw Q8_0 row (34-byte blocks) -> fp32
void embed_q8_0(const uint8_t * table, int64_t row_bytes, int token, float * y, int n, cudaStream_t s);

// ---- gated attention (full-attention layers) ----
// qg: [n_head][2*hd] (q | gate per head), k,v: [n_head_kv][hd]; normalizes q,k per head, applies
// partial NEOX rope on the first n_rot dims, writes q (fp32) and appends k,v (fp16) to the cache at pos
void attn_prep(float * qg, float * k, const float * v, const float * qnorm, const float * knorm,
               half * kcache, half * vcache, int pos, int max_pos, int n_head, int n_head_kv, int hd, int n_rot,
               float rope_base, float eps, cudaStream_t s);
// out[h][hd] = softmax(q k^T * scale) V over positions [0, pos], multiplied by sigmoid(gate)
void attn_decode(const float * qg, const half * kcache, const half * vcache, float * out, int n_pos, int max_pos,
                 int n_head, int n_head_kv, int hd, float scale, cudaStream_t s);

// ---- gated delta net (linear-attention layers) ----
// conv1d step over [q|k|v] channels with rolling state (kernel K, state holds K-1 previous inputs), then SiLU
void gdn_conv(float * qkv, float * conv_state, const float * conv_w, int channels, int K, cudaStream_t s);
// l2-normalize q,k heads, gates, recurrent update of S (per v-head [dv][dk] fp32), output o [n_v][dv]
void gdn_step(const float * qkv, const float * ab /*alpha|beta raw, 2*n_v*/, const float * dt_bias,
              const float * ssm_a, float * state, float * o, int n_k, int n_v, int dk, int dv, float eps,
              cudaStream_t s);
// o = rmsnorm_per_head(o) * w * silu(z)
void gated_norm(float * o, const float * z, const float * w, int n_heads, int dh, float eps, cudaStream_t s);

// ---- ffn ----
// h[i] = silu(gu[i]) * gu[n + i]   (gate and up computed into one buffer of 2n)
void silu_mul(const float * gu, float * h, int n, cudaStream_t s);

void argmax(const float * x, int n, int * out, cudaStream_t s);

} // namespace hyper
