// hyper kernels. Activations are fp32, laid out as nt token rows with an explicit row stride.
// Weights are repacked at load time. nt (tokens per call) is small: 1 for plain decode,
// 2..MAX_NT for speculative verification.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace hyper {

constexpr int MAX_NT = 4;

// Q8_0 weight repacked into mma fragment order. Tile = 16 rows x 32 columns (one Q8 block per row) stored as
// 512 lane-ordered bytes, tiles ordered [row tile][k block]; scales s[row tile][k block][16] fp16. Rows padded to 16.
struct Q8W {
    const uint4 * q = nullptr;
    const half * s = nullptr;
    int n = 0, k = 0;   // out features (unpadded), in features
};
// host: repack row-major int8 qs[n][k] + scales d[n][k/32] into fragment order (fq: ntile*kb*512 bytes, fs: ntile*kb*16)
void repack_q8_frag(const int8_t * qs, const half * d, int n, int k, uint8_t * fq, half * fs);
// 16-bit float weight (stored as fp16, converted from bf16 at load), fragment order: tile = 16 rows x 16 cols
// (one mma k-step) = 512 lane-ordered bytes, tiles ordered [row tile][k step]. Rows padded to 16.
struct BF16W {
    const uint4 * q = nullptr;
    int n = 0, k = 0;
};
// host: repack row-major bf16 w[n][k] into fp16 fragment tiles (out: ntile * (k/16) * 512 bytes)
void repack_bf16_frag(const uint16_t * w, int n, int k, size_t row_stride, uint8_t * out);
// same from fp32 rows (dequantized K-quants etc.)
void repack_f32_frag(const float * w, int n, int k, size_t row_stride, uint8_t * out);
// fused input RMSNorm: x' = x * rsqrt(sum(ss[t*nss .. t*nss+nss)) / k + eps) * w   (w == nullptr: no norm)
// act (applied after the norm): 1 = silu(x * act_scale); 2 = silu(x[c]) * x[c + glu_off] (gated pair in one input row)
struct NormIn { const float * w = nullptr; const float * ss = nullptr; int nss = 0; float eps = 1e-6f;
                int act = 0; float act_scale = 1.0f; int glu_off = 0; };

// y[t][r] = (add ? add[t][r] : 0) + W x[t]   for t < nt; x rows have stride xs, y/add rows stride ys.
// Q8: tensor cores (mma m16n8k16, fp16 activations, fp32 accumulation), split-K over 8 warps per row tile.
// allocate the split-K scratch on device dev (enables split-K for matrices with few row tiles)
void gemv_init(int dev);
void gemv_q8(const Q8W & W, const float * x, int xs, float * y, int ys, const float * add, int nt, cudaStream_t s,
             const NormIn & nin = {});
void gemv_bf16(const BF16W & W, const float * x, int xs, float * y, int ys, const float * add, int nt, cudaStream_t s,
               const NormIn & nin = {});
// many tokens (prefill): x converted once to fp16 xh[T][k] (to_half, optional RMSNorm with weight w), then
// y[t][r] = (add ? add : 0) + W xh[t] on tensor cores, 128 rows x 128 tokens per block
void to_half(const float * x, int xs, const float * w, int k, float eps, half * xh, int nt, cudaStream_t s);
void gemm_q8(const Q8W & W, const half * xh, int T, float * y, int ys, const float * add, cudaStream_t s);
void gemm_f16(const BF16W & W, const half * xh, int T, float * y, int ys, const float * add, cudaStream_t s);

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
// prefill: tensor-core causal flash attention (64 query tokens x one q head per block)
void attn_prefill(const float * qkv, int stride, const half * kcache, const half * vcache, float * out, int out_stride,
                  const int * pos, int max_pos, int n_head, int head_off, int group, int kv_off, int hd, float scale, int nt,
                  cudaStream_t s);
// split-K variant: a block per (local kv head, position slice, token) serves all q heads of that kv head;
// part: scratch of attn_part_floats(n_head, n_kv, nt, hd) floats
int attn_nsplit(int n_kv, int nt);
size_t attn_part_floats(int n_head, int n_kv, int nt, int hd);
// list (optional): sparse attention, token t attends to cells list[t * list_stride + i] for i < list_n[t]
void attn_split(const float * qkv, int stride, const half * kcache, const half * vcache, float * part, float * out, int out_stride,
                const int * pos, int max_pos, int n_head, int n_kv, int head_off, int group, int kv_off, int hd,
                float scale, int nt, cudaStream_t s, const int * list = nullptr, int list_stride = 0, const int * list_n = nullptr);

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

// out[t][K] = the K largest {value, index + offset} of row t (index as int bits; unordered)
void topk_pairs(const float * x, int xs, int n, int offset, float * out, int K, int nt, cudaStream_t s);
constexpr int TOPK = 64;
// out[t] = {max value, index + offset} (index stored as int bits)
void argmax_pairs(const float * x, int xs, int n, int offset, float * out, int nt, cudaStream_t s);

// ---- multi-GPU allreduce without P2P (LL protocol, fp16 payload) ----
// x[i] += sum_d part_d[i] for i < n (n = nt * n_embd, even); exchanged through host-mapped uint2 slots
// [2][ndev][n/2]. ss_out (optional): partial sums of squares of the new x, AR_SS_SPAN elements per entry,
// so token t's statistics are ss_out[t * (n_embd / AR_SS_SPAN) ...]
constexpr int AR_SS_SPAN = 512;
void allreduce_add_ll16(float * x, const float * part, uint2 * slots, int g, int ndev, int n,
                        const int * counter, int call, cudaStream_t s, float * ss_out = nullptr);
// hyper-connection variant: no x; the sum of row t is scattered into the hc streams of res,
// res[t][s][e] += sum[t][e] * 2 * sigmoid(inj[t * 4 + s] / hc)   (n = nt * width, width = row length)
void allreduce_hc_ll16(float * res, const float * inj, int width, int hc, const float * part, uint2 * slots, int g, int ndev, int n,
                       const int * counter, int call, cudaStream_t s);
// bulk variant (many tokens, no sum-of-squares output): data [2][ndev][n] fp16 and flags [2][ndev][n/1024]
// in mapped host memory
void allreduce_add_bulk(float * x, const float * part, half * data, unsigned * flags, int g, int ndev, int n,
                        const int * counter, int call, cudaStream_t s);
// x[i] += own[i] + sum_{j < nparts} recv[j * stride + i]  (fp16 parts; copy-engine allreduce)
void add_parts(float * x, const half * own, const half * recv, size_t stride, int nparts, int n, cudaStream_t s);
// sum of squares of each token row into ss[t * nss] (remaining nss-1 entries zeroed)
void sumsq(const float * x, int xs, int n, int nt, float * ss, int nss, cudaStream_t s);
void incr_counter(int * c, cudaStream_t s);

} // namespace hyper
