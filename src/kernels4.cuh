// qwen4exp kernels: hyper-connections, MoE (experts in their GGUF block formats), PLE, routing, the CPU hand-off.
// Activations fp32; nt token rows (decode: nt <= MAX_NT).
#pragma once
#include "gguf.h"
#include "kernels.cuh"

#include <cuda_runtime.h>

namespace hyper {

constexpr int MOE_MAX_USED = 16;
constexpr int MOE_BULK_ROWS = 2048;  // prefill chunk

// ---- hyper-connections (hc streams of n) ----
// xn[t][s*n + e] = res[t][s*n + e] * rsqrt(mean_e res[t][s*n + e]^2 + eps) * w[s*n + e]
void hc_norm(const float * res, const float * w, float * xn, int n, int hc, float eps, int nt, cudaStream_t s);
// lo = silu(lo * scale) in place
void silu_scale(float * x, int n, float scale, int nt, int stride, cudaStream_t s);
// mixed[t][e] = (1/hc) sum_s xn[t][s*n+e] * sigmoid(gate[t][s*n+e])
void hc_mixed(const float * xn, const float * gate, float * mixed, int n, int hc, int nt, cudaStream_t s);
// res[t][s*n+e] += bo[t][e] * 2 * sigmoid(inject[t][s] / hc)
void hc_combine(float * res, const float * bo, const float * inject, int inject_stride, int n, int hc, int nt, cudaStream_t s);
// res[t][s*n+e] = x[t][e] (hc copies)
void hc_init(float * res, const float * x, int n, int hc, int nt, cudaStream_t s);

// y[t][r] = W[r] . x[t] for an fp32 row-major W[rows][k] (small: routers, injections, alpha/beta)
void gemv_f32(const float * W, int rows, int k, const float * x, int xs, float * y, int ys, int nt, cudaStream_t s);

// gated delta net output with a sigmoid gate: o = rmsnorm_per_head(o) * w * sigmoid(z)
void gated_norm_sigmoid(float * o, int o_stride, const float * z, int z_stride, const float * w, int n_heads, int dh, float eps,
                        int nt, cudaStream_t s);

// ---- MoE routing ----
// logits[t][0..n_expert) router logits, logits[t][n_expert] the shared-expert gate logit.
// softmax over the experts, top-k (ties to the lower index), renormalized weights; sg[t] = sigmoid(shared gate)
void moe_route(const float * logits, int ls, int n_expert, int k, int * ids, float * wts, float * sg, int nt, cudaStream_t s);

// expert weights of one layer on one device: n_local experts, gate/up [n_local][ff][k] and down [n_local][n][ff]
// in their GGUF types; slot[e] = local index or -1
struct MoeDev {
    const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr;
    GType tg = GType::F32, td = GType::F32;
    size_t gate_bytes = 0, down_bytes = 0;   // bytes per expert matrix
    const int * slot = nullptr;              // [n_expert]
    int ff = 0, n = 0;
};
// order (optional): the local pairs sorted by expert (count in order_n), so that consecutive blocks share an expert's
// weights in L2; without it every pair p = t*k + j is visited and non-local pairs are skipped
// h[p][f] = silu(gate_e . x[t]) * (up_e . x[t]) for every local pair
void moe_gate_up(const MoeDev & m, const float * x, int xs, const int * ids, int k, float * h, int nt, cudaStream_t s,
                 const int * order = nullptr, const int * order_n = nullptr);
// y[p][r] = w[p] * (down_e . h[p]) for local pairs (without order: 0 for the others)
void moe_down(const MoeDev & m, const float * h, const int * ids, const float * wts, int k, float * y, int nt, cudaStream_t s,
              const int * order = nullptr, const int * order_n = nullptr);
// order = the pairs whose expert has slot >= 0, grouped by expert; order_n[0] = their count, order_n[1] = active experts,
// egrp (optional) = (expert, start, count) per active expert (single block)
void moe_order(const MoeDev & m, const int * ids, int n_pairs, int n_expert, int * order, int * order_n, cudaStream_t s,
               int * egrp = nullptr);
// prefill: grouped tensor-core expert GEMMs over the ordered pairs (x16: fp16 token rows; pair p reads row p / k)
void moe_gemm_gate_up(const MoeDev & m, const half * x16, int xs, int k, const int * order, const int * order_n, const int * egrp,
                      int max_active, half * h16, cudaStream_t s);
void moe_gemm_down(const MoeDev & m, const half * h16, const int * order, const int * order_n, const int * egrp, int max_active,
                   const float * wts, float * y, cudaStream_t s);
// out[t][r] = sg[t] * shexp[t][r] + sum_j [owner(e_j) == g] y[t*k + j][r] (+ cpu[t][r] once the CPU result for seq
// is there, when any of token t's experts lives on the CPU: owner[e] == cpu_owner)
void moe_reduce(const float * shexp, const float * sg, const float * y, int k, float * out, int n, int nt,
                const int * ids, const int * owner, int g, int cpu_owner, const volatile unsigned * cpu_flag, const float * cpu_y,
                const int * counter, unsigned seq_tag, cudaStream_t s);

// CPU hand-off record in mapped host memory (one per layer)
struct CpuMoeRec {
    volatile unsigned seq;
    int nt;
    int ids[MAX_NT][MOE_MAX_USED];
    float wts[MAX_NT][MOE_MAX_USED];
    float x[MAX_NT][4096];
};
struct CpuMoeOut {
    volatile unsigned seq;
    float y[MAX_NT][4096];
};
// prefill: one record reused by every layer (the CPU consumes them in order)
struct CpuMoeBulk {
    volatile unsigned seq;
    int nt;
    int ids[MOE_BULK_ROWS][MOE_MAX_USED];
    float wts[MOE_BULK_ROWS][MOE_MAX_USED];
    float x[MOE_BULK_ROWS][4096];
};
struct CpuMoeBulkOut {
    volatile unsigned seq;
    float y[MOE_BULK_ROWS][4096];
};
// write x, ids, wts (row strides 4096 / MOE_MAX_USED) and then the sequence tag (counter * 64 + tag)
void moe_publish(volatile unsigned * seq, int * ntp, int * ids_dst, float * wts_dst, float * x_dst, const float * x, int xs, int n,
                 const int * ids, const float * wts, int k, int nt, const int * counter, unsigned seq_tag, cudaStream_t s);

// ---- QSA indexer (sparse attention over the top blocks of 4 cells) ----
// qi [nt][n_head*128] -> qn (rms norm * qnorm, rope at pos + t); kr [nt][128] -> kraw[pos + t] (fp16)
void idx_prep(const float * qi, const float * kr, const float * qnorm, float * qn, half * kraw, const int * pos, int n_head, int n_rot,
              float base, float eps, int nt, cudaStream_t s);
// blocks completed by these tokens: pool[b] = rope(rmsnorm(mean of kraw[4b..4b+3]) * knorm, 4b)
void idx_pool(const half * kraw, half * pool, const float * knorm, const int * pos, int nt, int n_rot, float base, float eps, cudaStream_t s);
// per token: the attended cells (ascending): everything while (p+1)/4 <= top, else the top pools' cells + the tail.
// scores: scratch [score_rows][score_stride >= max_pos/4]; tokens are processed score_rows at a time
void idx_select(const float * qn, const half * pool, const int * pos, int nt, int n_head, int top, float * scores, int score_stride,
                int score_rows, int * list, int list_stride, int * list_n, cudaStream_t s);

// ---- MTP ----
// ecat[t*hc + s] = [rms(x[t]) * enorm | rms(H[t][s]) * hnorm[s]] for the per-stream eh_proj (hs: H row stride, 0 = shared row;
// whole: one norm over all streams)
void mtp_prep(const float * x, const float * enorm, const float * H, int hs, const float * hnorm, float eps, int n, int hc, bool whole,
              float * ecat, int nt, cudaStream_t s);

// ---- PLE ----
// res += gated + silu(conv(rmsnorm_stream(gated) * w_conv)); gated_s = value * sigmoid(ssqrt(<rms(key_s)*wk, rms(res_s)*wq> / sqrt(n)))
// conv: depthwise causal, kernel K, dilation dil, history state [(K-1)*dil][hc*n] (oldest first); snap (optional):
// history after token t at snap[t] for t < nt-1
void ple_apply(float * res, const float * key, const float * value, const float * wk, const float * wq, const float * wconv_norm,
               const float * conv_w, float * conv_state, float * conv_snap, int n, int hc, int K, int dil, float eps, int nt,
               float * scratch, cudaStream_t s);

} // namespace hyper
