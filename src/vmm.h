// Growable device buffers on the CUDA virtual memory API: a fixed address range (pointers stay valid, also inside
// captured CUDA graphs) whose physical memory is mapped and released in allocation granules (2 MB on the 3090s).
// Used for elastic placement: the attention caches map only the positions in use, the expert arrays map extra slots in
// the memory the caches do not use yet.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace hyper {

#define CU_CHECK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char * s_ = nullptr; cuGetErrorString(r_, &s_); \
    throw std::runtime_error(std::string("CUDA driver: ") + (s_ ? s_ : "?") + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)

inline size_t vmm_granularity(int dev) {
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    size_t g = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return g;
}

class VmmBuf {
public:
    VmmBuf() = default;
    VmmBuf(const VmmBuf &) = delete;
    VmmBuf & operator=(const VmmBuf &) = delete;
    ~VmmBuf() { release(); }

    void reserve(int dev, size_t bytes) {
        dev_ = dev;
        gran_ = vmm_granularity(dev);
        size_ = (bytes + gran_ - 1) / gran_ * gran_;
        if (size_ == 0) size_ = gran_;
        CU_CHECK(cuMemAddressReserve(&base_, size_, gran_, 0, 0));
        h_.assign(size_ / gran_, 0);
    }
    void * ptr() const { return (void *) base_; }
    size_t granule() const { return gran_; }
    size_t reserved() const { return size_; }
    size_t mapped() const { return mapped_ * gran_; }
    bool is_mapped(size_t i) const { return i < h_.size() && h_[i] != 0; }
    // granules covering [off, off + len)
    size_t first(size_t off) const { return off / gran_; }
    size_t last(size_t off, size_t len) const { return len == 0 ? first(off) : (off + len - 1) / gran_ + 1; }
    // bytes that mapping [off, off + len) would add
    size_t cost(size_t off, size_t len) const {
        size_t n = 0;
        for (size_t i = first(off); i < last(off, len) && i < h_.size(); ++i) n += h_[i] == 0;
        return n * gran_;
    }
    // map the granules covering [off, off + len); new granules are zeroed when `zero` (stream s)
    void map(size_t off, size_t len, bool zero, cudaStream_t s = nullptr) {
        if (off + len > size_) throw std::runtime_error("VmmBuf::map beyond the reserved range");
        for (size_t i = first(off); i < last(off, len); ++i) {
            if (h_[i]) continue;
            CUmemAllocationProp p = {};
            p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            p.location.id = dev_;
            CUmemGenericAllocationHandle h = 0;
            CU_CHECK(cuMemCreate(&h, gran_, &p, 0));
            CU_CHECK(cuMemMap(base_ + i * gran_, gran_, 0, h, 0));
            CUmemAccessDesc a = {};
            a.location = p.location;
            a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(base_ + i * gran_, gran_, &a, 1));
            h_[i] = h;
            ++mapped_;
            if (zero) {
                if (cudaMemsetAsync((void *) (base_ + i * gran_), 0, gran_, s) != cudaSuccess) throw std::runtime_error("VmmBuf: memset failed");
            }
        }
    }
    // release the granules lying entirely inside [off, end) (nothing may use them any more)
    size_t unmap(size_t off, size_t end) {
        size_t freed = 0;
        for (size_t i = (off + gran_ - 1) / gran_; (i + 1) * gran_ <= end && i < h_.size(); ++i) {
            if (!h_[i]) continue;
            CU_CHECK(cuMemUnmap(base_ + i * gran_, gran_));
            CU_CHECK(cuMemRelease(h_[i]));
            h_[i] = 0;
            --mapped_;
            freed += gran_;
        }
        return freed;
    }
    void release() {
        if (!base_) return;
        cudaSetDevice(dev_);
        cudaDeviceSynchronize();
        for (size_t i = 0; i < h_.size(); ++i)
            if (h_[i]) { cuMemUnmap(base_ + i * gran_, gran_); cuMemRelease(h_[i]); h_[i] = 0; }
        cuMemAddressFree(base_, size_);
        base_ = 0;
        mapped_ = 0;
    }

private:
    int dev_ = 0;
    CUdeviceptr base_ = 0;
    size_t size_ = 0, gran_ = 0, mapped_ = 0;
    std::vector<CUmemGenericAllocationHandle> h_;
};

} // namespace hyper
