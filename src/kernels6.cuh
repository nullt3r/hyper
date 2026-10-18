// mimo2 kernels: q/k/v rope + cache rows, GQA attention (k head <= 256, v head 128) with attention sinks and a sliding
// window, over full-length caches (global layers) or ring caches (sliding-window layers).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace hyper {

constexpr int MIMO_QB = 64;   // queries per attention block (rows x q heads of one kv head)

// q/k/v rows per token in big (stride bs): [q: nq heads of dk | k: nkv heads of dk | v: nkv heads of dv]. Token t sits at
// position pos[0] + t. q and k heads: the first n_rot dims NeoX-roped (pairs i, i + n_rot/2; theta = p * base^(-2i/n_rot), in
// double); q stays in place (fp32), k and v go to the caches as fp16: kv head h's row at kc + h * kpitch + r * dk (and
// vc + h * vpitch + r * dv), r = p (ring == 0) or p % ring
void mimo_qkv(float * big, int bs, int nq, int nkv, int dk, int dv, int n_rot, float base, const int * pos, int nt, half * kc,
              half * vc, size_t kpitch, size_t vpitch, int ring, cudaStream_t s);

struct MimoAttn {
    const float * q = nullptr; int qs = 0;   // row t, local q head h: q + t * qs + h * dk
    const half * kc = nullptr, * vc = nullptr;
    size_t kpitch = 0, vpitch = 0;
    int ring = 0;                 // 0: cache row = position
    const int * pos = nullptr;    // device: position of row 0
    int nt = 0, nq = 0, nkv = 0, dk = 0, dv = 128;
    int window = 0;               // 0: keys 0..p; else keys p - window + 1 .. p
    float scale = 1.0f, vscale = 1.0f;
    const float * sinks = nullptr;   // per local q head (nullptr: none)
    float * out = nullptr; int os = 0;   // row t, local q head h: out + t * os + h * dv
    float * part = nullptr;          // split partials (ns > 1)
};
// ns: key splits per row block (fixed for a captured graph; empty splits are skipped by the combine)
void mimo_attn(const MimoAttn & a, int ns, cudaStream_t s);
// prefill on the tensor cores (Q and P rounded to fp16, fp32 accumulation; full caches)
void mimo_attn_fa(const MimoAttn & a, cudaStream_t s);

// MTP inputs: the main model's last residual rows (before the output norm) kept by position, hring[(pos[0] + r) % HR] = R[r]
// (the last HR rows of a call)
void mimo_hstore(const float * R, int nt, const int * pos, float * hring, int HR, int n, cudaStream_t s);
// hidden input rows of an MTP pass at positions q0 + t: t < nmain the main model's row of position q0 + t - 1 (zero before
// position 0), then the chained rows chain[t - nmain]
void mimo_hgather(float * dst, const float * hring, int HR, int n, int q0, int nrows, int nmain, const float * chain, cudaStream_t s);
size_t mimo_attn_part_floats(int nkv, int nt, int group, int ns);

} // namespace hyper
