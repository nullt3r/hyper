// glm5-next (GLM-5.3-Flash) kernels: mHC residual streams (Sinkhorn mixing), KDA (per-channel gated delta rule),
// nope MLA over a compressed latent cache with a k-pool DSA indexer, sigmoid routing.
// Activations fp32, nt token rows.
#pragma once
#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace hyper {

constexpr int MHC = 4;            // residual streams
constexpr int MHC_W = 20;         // per-token mixing weights: post[4] | comb[4][4] (comb[dst + 4 * src])
constexpr int MLA_LAT = 512;      // latent width (kv_lora_rank)
constexpr int MLA_MAXH = 24;      // local MLA heads per GPU
constexpr int MLA_SPLIT = 32;     // decode: cell slices per token
constexpr int GIDX_HEADS = 32, GIDX_DIM = 128;

// ---- mHC ----
// mixraw[t][0..24) = hc_fn . res[t] (unnormalized). Per token: mixes = mixraw * rsqrt(mean(res^2) + rms_eps);
// pre = sigmoid(mixes[0..4) * scale[0] + base[0..4)) + hc_eps; post = 2 sigmoid(mixes[4..8) * scale[1] + base[4..8));
// comb = sinkhorn(mixes[8..24) * scale[2] + base[8..24)); hcw[t] = post | comb;
// xn[t] = rmsnorm(sum_s pre_s res[t][s]) * norm_w
void mhc_pre(const float * res, const float * mixraw, int mix_stride, const float * scale, const float * base, const float * norm_w,
             float rms_eps, float hc_eps, int iters, int n, float * hcw, float * xn, int nt, cudaStream_t s);
// res[t][d][e] = out[t][e] * post[d] + sum_s comb[d][s] res[t][s][e]
void mhc_post(float * res, const float * out, const float * hcw, int n, int nt, cudaStream_t s);
// res[t][s][e] = x[t][e]
void mhc_init(float * res, const float * x, int n, int nt, cudaStream_t s);
// xn[t] = rmsnorm(mean_s res[t][s]) * w
void mhc_head(const float * res, const float * w, float eps, int n, float * xn, int nt, cudaStream_t s);

// ---- KDA ----
// rows of `in` (stride): q, k, v (after conv + silu) at q_off / k_off / v_off, decay pre-activation fb at fb_off, beta logits
// at b_off (one per head), all for the n_head local heads of 128. q, k l2-normalized (eps); per key channel i:
// g_i = lb * sigmoid(-A[h] * (fb_i + dt_bias_i)); S[i][:] *= exp(g_i); delta = (v - S^T k) * sigmoid(beta);
// S += k delta^T; o = S^T q / sqrt(128). state [n_head][128][128] (S[i][j], i key dim); snap optional (after token t < nt-1)
void kda_step(const float * in, int stride, int q_off, int k_off, int v_off, int fb_off, int b_off, float * state, float * snap,
              float * o, int o_stride, const float * dt_bias, const float * A, float lb, int n_head, float eps, int nt, cudaStream_t s);

// ---- MLA ----
// y[t][h * R + r] = sum_c W[h][r][c] x[t][h * C + c]  (fp16 W [H][R][C]); decode (nt <= MAX_NT)
void head_gemv(const half * W, int H, int R, int C, const float * x, int xs, float * y, int ys, int nt, cudaStream_t s);
// lat[pos + t] = half(rmsnorm(kv[t]) * w)  (512 wide)
void mla_kv(const float * kv, int kv_stride, const float * w, float eps, half * lat, const int * pos, int nt, cudaStream_t s);
// o[t][h] = sum_j softmax_j(scale * q[t][h] . lat[c_j]) lat[c_j] over the token's cells c_j: list[t][0..list_n[t]) or, without a
// list, 0..pos+t. q [nt][H][512] (q_stride per token), o likewise. part: decode scratch of mla_part_floats(H, nt) floats
size_t mla_part_floats(int H, int nt);
void mla_init();   // once per device (current device): the attention kernel's shared memory limit
void mla_attn(const float * q, int q_stride, const half * lat, const int * pos, int H, float scale, int nt, const int * list,
              int list_stride, const int * list_n, float * part, float * o, int o_stride, cudaStream_t s);

// ---- k-pool indexer ----
// ik = layernorm(ikraw) * lnw + lnb (eps), ig = igraw; both as fp16 into the ring of the open pool's cells (ring[cell % 4] =
// ik | ig, 256 halves); every pool b completed by these tokens: pooled[b][d] = sum_j softmax_j(ig_j[d] + ape[j][d]) ik_j[d]
void gidx_pool(const float * ikraw, const float * igraw, int stride, const float * lnw, const float * lnb, float eps, const float * ape,
               half * ring, half * pooled, const int * pos, int nt, cudaStream_t s);
// per token (rows of `rows` at a time through the score scratch [rows][score_stride]): np = (p + 1) / 4 complete pools; np <= top:
// every cell 0..p; else the `top` pools with the largest sum_h w[t][h] relu(iq[t][h] . pooled[b]) (ascending cells) + the tail
void gidx_select(const float * iq, int iq_stride, const float * w, int w_stride, const half * pooled, const int * pos, int nt, int top,
                 float * scores, int score_stride, int rows, int * list, int list_stride, int * list_n, cudaStream_t s);

// ---- MoE ----
// p = sigmoid(logits); the k experts with the largest p + bias (ties to the lower index); wts = p / max(sum, 6.1e-5) * scale;
// sg[t] = 1 (the shared expert is ungated)
void moe_route_sig(const float * logits, int ls, const float * bias, int n_expert, int k, float scale, int * ids, float * wts,
                   float * sg, int nt, cudaStream_t s);
// h[t][i] = silu(min(g, L)) * clamp(u, -L, L) with g = gu[t][i], u = gu[t][off + i]
void swiglu_clamp(const float * gu, int gu_stride, int off, float * h, int h_stride, int n, float L, int nt, cudaStream_t s);

} // namespace hyper
