// The CUDA device of an engine: the handles every engine's Device has (streams, events, cuBLAS), its allocations and
// their teardown. An engine derives its Device from DevBase and keeps its own fields; init_base() sets the base up in
// the constructor, ~DevBase() tears it down after the derived destructor. Also: how many devices an engine uses and the
// PCIe link weights of this box.
#pragma once
#include "cuda_check.h"
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace hyper {

struct DevBase {
    int id = 0, g = 0;                // CUDA device, the engine's index of it
    cudaStream_t stream = nullptr;    // the layers
    cudaStream_t cstream = nullptr;   // prefill: expert uploads
    cudaStream_t pstream = nullptr;   // routing prediction side stream
    cudaEvent_t ev_up[2] = {}, ev_free[2] = {};         // staging buffer b uploaded / free again
    cudaEvent_t ev_pe0 = nullptr, ev_pe1 = nullptr;     // early routing prediction: input ready / prediction done
    cublasHandle_t blas = nullptr;
    uint8_t * stage[2] = {};          // prefill: staging buffers for streamed experts
    size_t stage_bytes = 0;
    size_t used = 0;                  // device bytes allocated through alloc()
    std::vector<void *> allocs;

    template <typename T> T * alloc(size_t n) {
        void * p = nullptr;
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
        CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
        allocs.push_back(p);
        used += n * sizeof(T);
        return (T *) p;
    }
    template <typename T> T * upload(const T * src, size_t n) {
        T * p = alloc<T>(n);
        CUDA_CHECK(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    // device dev_id as the engine's device `group`: the streams, events and cuBLAS (pedantic: no reduced-precision paths;
    // else plain fp32 without TF32), the GEMV kernels' per-device setup; pstream_hi: the prediction stream at the highest priority
    void init_base(int dev_id, int group, bool pedantic, bool pstream_hi = false) {
        id = dev_id; g = group;
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
        if (pstream_hi) {
            int lo = 0, hi = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            CUDA_CHECK(cudaStreamCreateWithPriority(&pstream, cudaStreamNonBlocking, hi));
        } else CUDA_CHECK(cudaStreamCreateWithFlags(&pstream, cudaStreamNonBlocking));
        for (auto & ev : ev_up) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        for (auto & ev : ev_free) CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_pe0, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_pe1, cudaEventDisableTiming));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        cublasSetStream(blas, stream);
        cublasSetMathMode(blas, pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH);   // (routing must not move)
        gemv_init(id);
    }
    ~DevBase() {
        cudaSetDevice(id);
        if (blas) cublasDestroy(blas);
        for (auto & ev : ev_up) if (ev) cudaEventDestroy(ev);
        for (auto & ev : ev_free) if (ev) cudaEventDestroy(ev);
        for (auto ev : {ev_pe0, ev_pe1}) if (ev) cudaEventDestroy(ev);
        if (cstream) cudaStreamDestroy(cstream);
        if (pstream) cudaStreamDestroy(pstream);
        for (void * p : allocs) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

// how many devices the engine uses: its option, overridden by HYPER_NDEV or its own env (4th GPU), at most the count present
inline int device_count(int wanted, const char * env_name, const char * tag) {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (const char * e = getenv("HYPER_NDEV")) wanted = std::max(1, atoi(e));
    if (const char * e = env_name ? getenv(env_name) : nullptr) wanted = std::max(1, atoi(e));
    if (wanted <= 0) wanted = ndev;   // (default: every visible device -- CUDA_VISIBLE_DEVICES picks the set)
    wanted = std::min(wanted, ndev);
    fprintf(stderr, "%s: %d of %d CUDA devices\n", tag, wanted, ndev);
    return wanted;
}

// PCIe link weights per device for the prefill streaming shares (x16 = 2, x8 = 1): this box has device 1 on x8 with
// three GPUs, devices 1 and 2 on x8 with four; the env ("2,1,1,2") or HYPER_EP_WEIGHTS overrides
inline std::vector<double> link_weights(int nd, const char * env_name = nullptr) {
    std::vector<double> wbw(nd, 2.0);
    if (nd > 1) wbw[1] = 1.0;
    if (nd == 4) wbw[2] = 1.0;
    const char * w = env_name ? getenv(env_name) : nullptr;
    if (!w) w = getenv("HYPER_EP_WEIGHTS");
    if (w) {
        size_t i = 0;
        for (const char * q = w; *q && i < wbw.size(); ++i) { wbw[i] = atof(q); while (*q && *q != ',') ++q; if (*q) ++q; }
    }
    return wbw;
}

} // namespace hyper
