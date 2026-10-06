// Prototype: tensor-core Q8_0 GEMM for small token counts (mma.sync m16n8k16, fp16 inputs, fp32 accumulate).
// Weights are repacked into fragment order: tile = 16 rows x 32 cols (one Q8 block) = 512 bytes, lane-ordered,
// so each lane fetches exactly its A fragments for two k-steps with one 16-byte load.
// Compares against the production FMA GEMV for nt = 1..4 and checks the results agree.
#include "../src/kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// two int8 (in the low 16 bits of v, selected by `sel`) -> half2 via 1024 + (b ^ 0x80) - 1152
__device__ __forceinline__ unsigned i8x2_to_h2(unsigned v, unsigned sel) {
    const unsigned x = __byte_perm(v ^ 0x80808080u, 0x64646464u, sel);
    const unsigned magic = 0x64806480u;   // half2(1152, 1152)
    unsigned r;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(magic));
    return r;
}

__device__ __forceinline__ void mma16816(float * c, const unsigned * a, const unsigned * b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const unsigned *) &h;
}

// one warp per 16-row tile; loops over all k blocks. x: [nt][k] fp32 (nt <= 8), y: [nt][n]
__global__ void k_mma_q8(const uint4 * __restrict__ wq, const half * __restrict__ ws, int n, int k,
                         const float * __restrict__ x, int xs, float * __restrict__ y, int ys, int nt) {
    const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int ntile = (n + 15) / 16, kb = k / 32;
    if (warp >= ntile) return;
    const uint4 * tq = wq + (size_t) warp * kb * 32 + lane;
    const half * ts = ws + (size_t) warp * kb * 16;
    const bool tok_ok = gid < nt;
    const float * xr = x + (size_t) (tok_ok ? gid : 0) * xs;
    float acc[4] = {0, 0, 0, 0};
    for (int b = 0; b < kb; ++b) {
        const uint4 q = __ldg(tq + (size_t) b * 32);
        const float s_lo = __half2float(ts[(size_t) b * 16 + gid]), s_hi = __half2float(ts[(size_t) b * 16 + gid + 8]);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        float tmp[4] = {0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            // A: bytes [8ks .. 8ks+8): (r,c),(r,c+1) | (r+8,c),(r+8,c+1) | (r,c+8),(r,c+9) | (r+8,c+8),(r+8,c+9)
            const unsigned w0 = qw[2 * ks], w1 = qw[2 * ks + 1];
            unsigned a[4];
            a[0] = i8x2_to_h2(w0, 0x5140);   // bytes 0,1
            a[1] = i8x2_to_h2(w0, 0x7362);   // bytes 2,3
            a[2] = i8x2_to_h2(w1, 0x5140);
            a[3] = i8x2_to_h2(w1, 0x7362);
            const int c0 = b * 32 + ks * 16 + 2 * tig;
            unsigned bb[2];
            if (tok_ok) {
                const float2 v0 = *(const float2 *) (xr + c0), v1 = *(const float2 *) (xr + c0 + 8);
                bb[0] = pack_h2(v0.x, v0.y);
                bb[1] = pack_h2(v1.x, v1.y);
            } else {
                bb[0] = bb[1] = 0;
            }
            mma16816(tmp, a, bb);
        }
        acc[0] += tmp[0] * s_lo; acc[1] += tmp[1] * s_lo;
        acc[2] += tmp[2] * s_hi; acc[3] += tmp[3] * s_hi;
    }
    const int r0 = warp * 16 + gid, t0 = 2 * tig;
    if (r0 < n) {
        if (t0 < nt) y[(size_t) t0 * ys + r0] = acc[0];
        if (t0 + 1 < nt) y[(size_t) (t0 + 1) * ys + r0] = acc[1];
    }
    if (r0 + 8 < n) {
        if (t0 < nt) y[(size_t) t0 * ys + r0 + 8] = acc[2];
        if (t0 + 1 < nt) y[(size_t) (t0 + 1) * ys + r0 + 8] = acc[3];
    }
}


// split-K: block = 8 warps on one 16-row tile; warp w takes k blocks w, w+8, ... ; partials reduced in shared memory
__global__ void k_mma_q8_sk(const uint4 * __restrict__ wq, const half * __restrict__ ws, int n, int k,
                            const float * __restrict__ x, int xs, float * __restrict__ y, int ys, int nt) {
    const int tile = blockIdx.x;
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int kb = k / 32;
    const uint4 * tq = wq + (size_t) tile * kb * 32 + lane;
    const half * ts = ws + (size_t) tile * kb * 16;
    const bool tok_ok = gid < nt;
    const float * xr = x + (size_t) (tok_ok ? gid : 0) * xs;
    float acc[4] = {0, 0, 0, 0};
    for (int b = w; b < kb; b += nw) {
        const uint4 q = __ldg(tq + (size_t) b * 32);
        const float s_lo = __half2float(ts[(size_t) b * 16 + gid]), s_hi = __half2float(ts[(size_t) b * 16 + gid + 8]);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        float tmp[4] = {0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const unsigned w0 = qw[2 * ks], w1 = qw[2 * ks + 1];
            unsigned a[4];
            a[0] = i8x2_to_h2(w0, 0x5140);
            a[1] = i8x2_to_h2(w0, 0x7362);
            a[2] = i8x2_to_h2(w1, 0x5140);
            a[3] = i8x2_to_h2(w1, 0x7362);
            const int c0 = b * 32 + ks * 16 + 2 * tig;
            unsigned bb[2];
            if (tok_ok) {
                const float2 v0 = *(const float2 *) (xr + c0), v1 = *(const float2 *) (xr + c0 + 8);
                bb[0] = pack_h2(v0.x, v0.y);
                bb[1] = pack_h2(v1.x, v1.y);
            } else {
                bb[0] = bb[1] = 0;
            }
            mma16816(tmp, a, bb);
        }
        acc[0] += tmp[0] * s_lo; acc[1] += tmp[1] * s_lo;
        acc[2] += tmp[2] * s_hi; acc[3] += tmp[3] * s_hi;
    }
    __shared__ float red[8][32][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) red[w][lane][i] = acc[i];
    __syncthreads();
    if (w == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { float t = 0; for (int ww = 0; ww < nw; ++ww) t += red[ww][lane][i]; acc[i] = t; }
        const int r0 = tile * 16 + gid, t0 = 2 * tig;
        if (r0 < n) {
            if (t0 < nt) y[(size_t) t0 * ys + r0] = acc[0];
            if (t0 + 1 < nt) y[(size_t) (t0 + 1) * ys + r0] = acc[1];
        }
        if (r0 + 8 < n) {
            if (t0 < nt) y[(size_t) t0 * ys + r0 + 8] = acc[2];
            if (t0 + 1 < nt) y[(size_t) (t0 + 1) * ys + r0 + 8] = acc[3];
        }
    }
}

// repack row-major qs[n][k] + d[n][k/32] into fragment tiles
static void repack(const std::vector<int8_t> & qs, const std::vector<half> & d, int n, int k,
                   std::vector<uint8_t> & fq, std::vector<half> & fs) {
    const int ntile = (n + 15) / 16, kb = k / 32;
    fq.assign((size_t) ntile * kb * 512, 0);
    fs.assign((size_t) ntile * kb * 16, __float2half(0.0f));
    for (int t = 0; t < ntile; ++t)
        for (int b = 0; b < kb; ++b) {
            uint8_t * tile = &fq[((size_t) t * kb + b) * 512];
            for (int lane = 0; lane < 32; ++lane) {
                const int gid = lane >> 2, tig = lane & 3;
                uint8_t * o = tile + lane * 16;
                for (int ks = 0; ks < 2; ++ks) {
                    const int cb = b * 32 + ks * 16 + 2 * tig;
                    const int rows[4] = {gid, gid + 8, gid, gid + 8};
                    const int cols[4] = {cb, cb, cb + 8, cb + 8};
                    for (int p = 0; p < 4; ++p)
                        for (int e = 0; e < 2; ++e) {
                            const int r = t * 16 + rows[p], c = cols[p] + e;
                            o[ks * 8 + p * 2 + e] = r < n ? (uint8_t) qs[(size_t) r * k + c] : 0;
                        }
                }
            }
            for (int r = 0; r < 16; ++r) {
                const int rr = t * 16 + r;
                fs[((size_t) t * kb + b) * 16 + r] = rr < n ? d[(size_t) rr * kb + b] : __float2half(0.0f);
            }
        }
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 300;
    struct Shape { const char * name; int n, k; };
    const Shape shapes[] = {{"ffn gate+up", 11584, 5120}, {"ffn_down", 5120, 5792}, {"gdn in", 5152, 5120}, {"gdn out", 5120, 1920}};
    CK(cudaSetDevice(0));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::mt19937 rng(1);
    for (auto & sh : shapes) {
        const int n = sh.n, k = sh.k, kb = k / 32;
        std::vector<int8_t> qs((size_t) n * k);
        std::vector<half> d((size_t) n * kb);
        std::uniform_int_distribution<int> qi(-127, 127);
        std::uniform_real_distribution<float> sd(0.001f, 0.01f);
        for (auto & v : qs) v = (int8_t) qi(rng);
        for (auto & v : d) v = __float2half(sd(rng));
        std::vector<uint8_t> fq; std::vector<half> fs;
        repack(qs, d, n, k, fq, fs);
        const int nmat = 6;   // working set > L2
        std::vector<hyper::Q8W> W(nmat);
        std::vector<uint4 *> FQ(nmat); std::vector<half *> FS(nmat);
        for (int m = 0; m < nmat; ++m) {
            int8_t * dq; half * dd;
            CK(cudaMalloc(&dq, qs.size())); CK(cudaMemcpy(dq, qs.data(), qs.size(), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dd, d.size() * 2)); CK(cudaMemcpy(dd, d.data(), d.size() * 2, cudaMemcpyHostToDevice));
            W[m].qs = dq; W[m].d = dd; W[m].n = n; W[m].k = k;
            CK(cudaMalloc(&FQ[m], fq.size())); CK(cudaMemcpy(FQ[m], fq.data(), fq.size(), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&FS[m], fs.size() * 2)); CK(cudaMemcpy(FS[m], fs.data(), fs.size() * 2, cudaMemcpyHostToDevice));
        }
        std::vector<float> hx((size_t) 8 * k);
        std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
        for (auto & v : hx) v = xd(rng);
        float * x, * y1, * y2;
        CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&y1, (size_t) 8 * n * 4)); CK(cudaMalloc(&y2, (size_t) 8 * n * 4));
        printf("%-12s (%5dx%5d):", sh.name, n, k);
        for (int nt = 1; nt <= 4; ++nt) {
            const double bytes = (double) n * k * (1.0 + 2.0 / 32);
            float ms_old, ms_new;
            for (int variant = 0; variant < 2; ++variant) {
                cudaGraph_t gr; cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int i = 0; i < iters; ++i) {
                    const int m = i % nmat;
                    if (variant == 0) hyper::gemv_q8(W[m], x, k, y1, n, nullptr, nt, s);
                    else k_mma_q8_sk<<<(n + 15) / 16, 256, 0, s>>>(FQ[m], FS[m], n, k, x, k, y2, n, nt);
                }
                CK(cudaStreamEndCapture(s, &gr));
                CK(cudaGraphInstantiate(&ge, gr, 0));
                CK(cudaGraphLaunch(ge, s));
                CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                (variant == 0 ? ms_old : ms_new) = ms;
                CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
            }
            // compare results (last matrix used by both)
            std::vector<float> a((size_t) nt * n), b((size_t) nt * n);
            CK(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost));
            double maxrel = 0, ref = 0;
            for (size_t i = 0; i < a.size(); ++i) { ref = std::max(ref, (double) std::fabs(a[i])); }
            for (size_t i = 0; i < a.size(); ++i) maxrel = std::max(maxrel, std::fabs((double) a[i] - b[i]) / ref);
            printf("  nt%d %5.0f->%5.0f GB/s (err %.1e)", nt, bytes * iters / (ms_old * 1e-3) / 1e9, bytes * iters / (ms_new * 1e-3) / 1e9, maxrel);
        }
        printf("\n");
        for (int m = 0; m < nmat; ++m) { cudaFree((void *) W[m].qs); cudaFree((void *) W[m].d); cudaFree(FQ[m]); cudaFree(FS[m]); }
        cudaFree(x); cudaFree(y1); cudaFree(y2);
    }
    return 0;
}
