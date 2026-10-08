// expert dequantization on the GPU (deq8) vs ggml's reference dequantization, random blocks of each type
#include "../src/kernels4.cuh"
#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    using hyper::GType;
    const GType types[] = {GType::Q5_0, GType::Q5_1, GType::Q8_0, GType::Q4_K, GType::Q5_K, GType::Q6_K, GType::IQ4_XS};
    const int n = 2560;
    std::mt19937 rng(1);
    int bad = 0;
    for (GType t : types) {
        const size_t bb = hyper::gtype_block_bytes(t), be = hyper::gtype_block_elems(t), nb = n / be;
        std::vector<uint8_t> row(nb * bb);
        for (auto & b : row) b = (uint8_t) rng();
        // sane fp16 scales: overwrite every half that ggml reads as a scale (first 2 or 4 bytes of the block for the
        // simple types, d/dmin at the start of K blocks; Q6_K's d is the last 2 bytes)
        for (size_t i = 0; i < nb; ++i) {
            uint8_t * blk = row.data() + i * bb;
            auto put = [&](size_t off) { const __half h = __float2half(0.01f + 0.01f * (rng() % 100) / 100.0f); memcpy(blk + off, &h, 2); };
            if (t == GType::Q6_K) put(bb - 2);
            else { put(0); if (t == GType::Q5_1 || t == GType::Q4_K || t == GType::Q5_K) put(2); }
        }
        std::vector<float> ref(n), got(n);
        ggml_get_type_traits((ggml_type) t)->to_float(row.data(), ref.data(), n);
        uint8_t * drow; float * dout;
        cudaMalloc(&drow, row.size()); cudaMalloc(&dout, n * sizeof(float));
        cudaMemcpy(drow, row.data(), row.size(), cudaMemcpyHostToDevice);
        hyper::deq_row_test(t, drow, n, dout);
        cudaMemcpy(got.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(drow); cudaFree(dout);
        double mx = 0, ma = 0;
        for (int i = 0; i < n; ++i) { mx = std::max(mx, (double) std::fabs(ref[i] - got[i])); ma = std::max(ma, (double) std::fabs(ref[i])); }
        const bool ok = mx <= 1e-6 * std::max(1.0, ma);
        bad += !ok;
        printf("DEQ %-7s max |gpu - ggml| %.3g (max |x| %.3g) %s\n", hyper::gtype_name(t), mx, ma, ok ? "OK" : "MISMATCH");
    }
    return bad;
}
