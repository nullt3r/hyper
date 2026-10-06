// hyper kernels. Activations are fp32, laid out as nt token rows with an explicit row stride.
// Weights are repacked at load time. nt (tokens per call) is small: 1 for plain decode,
// 2..MAX_NT for speculative verification.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace hyper {

constexpr int MAX_NT = 4;

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
// fused input RMSNorm: x' = x * rsqrt(sum(ss[t*nss .. t*nss+nss)) / k + eps) * w   (w == nullptr: no norm)
struct NormIn { const float * w = nullptr; const float * ss = nullptr; int nss = 0; float eps = 1e-6f; };

// y[t][r] = (add ? add[t][r] : 0) + W x[t]   for t < nt; x rows have stride xs, y/add rows stride ys
void gemv_q8(const Q8W & W, const float * x, int xs, float * y, int ys, const float * add, int nt, cudaStream_t s,
             const NormIn & nin = {});
void gemv_bf16(const BF16W & W, const float * x, int xs, float * y, int ys, const float * add, int nt, cudaStream_t s,
               const NormIn & nin = {});

// y[t] = rmsnorm(x[t]) * w
void rmsnorm(const float * x, int xs, const float * w, float * y, int ys, int n, int nt, float eps, cudaStream_t s);

// ---- gated attention (full-attention layers), local heads ----
// per token row of qkv: [n_head x (q hd | gate hd)] [n_kv x k hd] [n_kv x v hd]
// normalizes q,k per head, partial NEOX rope (first n_rot dims) at position *pos + t, keeps q in place and
// appends k,v (fp16) to the caches [n_kv][max_pos][hd]
void attn_prep(float * qkv, int stride, const float * qnorm, const float * knorm, half * kcache, half * vcache,
               const int * pos, int max_pos, int n_head, int n_kv, int hd, int n_rot, float rope_base, float eps,
               int nt, cudaStream_t s);
// out[t][h][hd] = softmax(q k^T * scale) V over positions [0, *pos + t], times sigmoid(gate);
// local q head h uses local kv head (head_off + h) / group - kv_off
void attn_decode(const float * qkv, int stride, const half * kcache, const half * vcache, float * out, int out_stride,
                 const int * pos, int max_pos, int n_head, int n_kv, int head_off, int group, int kv_off, int hd,
                 float scale, int nt, cudaStream_t s);

// ---- gated delta net, local heads ----
// in rows: [q n_k*dk | k n_k*dk | v n_v*dv | z n_v*dv | alpha n_v | beta n_v]  (conv runs over the first `channels`)
// causal conv1d (kernel K) + SiLU over tokens in order; conv_state holds the K-1 previous inputs.
// conv_snap (optional): state after token t is stored at conv_snap[t] for t < nt-1
void gdn_conv(float * in, int stride, float * conv_state, float * conv_snap, const float * conv_w, int channels, int K,
              int nt, cudaStream_t s);
// recurrent gated delta rule over the nt tokens; state [n_v][dk][dv] fp32 (S[i][j], i: key dim, j: value dim);
// state_snap (optional): state after token t at state_snap[t] for t < nt-1; o rows have stride o_stride
void gdn_step(const float * in, int stride, int ab_off, float * state, float * state_snap, float * o, int o_stride,
              const float * dt_bias, const float * ssm_a, int n_k, int n_v, int dk, int dv, float eps, int nt,
              cudaStream_t s);
// o[t] = rmsnorm_per_head(o[t]) * w * silu(z[t])
void gated_norm(float * o, int o_stride, const float * z, int z_stride, const float * w, int n_heads, int dh, float eps,
                int nt, cudaStream_t s);

// ---- ffn ----
// h[t][i] = silu(gu[t][i]) * gu[t][n + i]
void silu_mul(const float * gu, int gu_stride, float * h, int h_stride, int n, int nt, cudaStream_t s);

// out[t] = {max value, index + offset} (index stored as int bits)
void argmax_pairs(const float * x, int xs, int n, int offset, float * out, int nt, cudaStream_t s);

// ---- multi-GPU allreduce without P2P (LL protocol, fp16 payload) ----
// x[i] += sum_d part_d[i] for i < n (n = nt * n_embd, even); exchanged through host-mapped uint2 slots
// [2][ndev][n/2]. ss_out (optional): partial sums of squares of the new x, AR_SS_SPAN elements per entry,
// so token t's statistics are ss_out[t * (n_embd / AR_SS_SPAN) ...]
constexpr int AR_SS_SPAN = 512;
void allreduce_add_ll16(float * x, const float * part, uint2 * slots, int g, int ndev, int n,
                        const int * counter, int call, cudaStream_t s, float * ss_out = nullptr);
// sum of squares of each token row into ss[t * nss] (remaining nss-1 entries zeroed)
void sumsq(const float * x, int xs, int n, int nt, float * ss, int nss, cudaStream_t s);
void incr_counter(int * c, cudaStream_t s);

} // namespace hyper
