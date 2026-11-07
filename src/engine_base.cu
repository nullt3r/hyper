#include "engine_base.h"
#include "cuda_check.h"

#include <cstdlib>
#include <cstring>

namespace hyper {

EngineBase::~EngineBase() { release_cpu_side(); }

void EngineBase::init_cpu_side(int threads, int n_embd, int ff, int K, int slots, float clamp, bool pred_early) {
    CUDA_CHECK(cudaHostAlloc(&cpu_rec_, (size_t) slots * sizeof(CpuMoeRec), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_out_, (size_t) slots * sizeof(CpuMoeOut), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_pred_, (size_t) slots * sizeof(CpuMoeRec), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_pred1_, (size_t) slots * sizeof(CpuMoeRec), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_, sizeof(CpuMoeBulk), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&cpu_bulk_out_, sizeof(CpuMoeBulkOut), cudaHostAllocPortable | cudaHostAllocMapped));
    memset((void *) cpu_rec_, 0, (size_t) slots * sizeof(CpuMoeRec));
    memset((void *) cpu_out_, 0, (size_t) slots * sizeof(CpuMoeOut));
    memset((void *) cpu_pred_, 0, (size_t) slots * sizeof(CpuMoeRec));
    memset((void *) cpu_pred1_, 0, (size_t) slots * sizeof(CpuMoeRec));
    memset((void *) cpu_bulk_, 0, sizeof(CpuMoeBulk));
    memset((void *) cpu_bulk_out_, 0, sizeof(CpuMoeBulkOut));
    cpu_ = std::make_unique<CpuMoe>(threads, n_embd, ff, K, cpu_rec_, cpu_out_, slots, cpu_bulk_, cpu_bulk_out_);
    cpu_->set_clamp(clamp);
    cpu_->set_pred(cpu_pred_);
    if (pred_early) {
        cpu_->set_pred1(cpu_pred1_);
        if (!getenv("HYPER_CPU_PREFETCH")) cpu_->set_helpers(false);
    }
}

void EngineBase::init_place(const MoePlace::Params & pp, const std::vector<DevBase *> & devs, int n_layer) {
    place_ = std::make_unique<MoePlace>();
    place_->init(pp, devs, cpu_.get());
    if (const char * sp = getenv("HYPER4_STATS")) place_->load_stats(sp, n_layer);
}

void EngineBase::release_cpu_side() {
    cpu_.reset();
    place_.reset();   // (the host copies after the CPU side)
    for (void * p : {(void *) cpu_rec_, (void *) cpu_pred_, (void *) cpu_pred1_, (void *) cpu_out_, (void *) cpu_bulk_, (void *) cpu_bulk_out_})
        if (p) cudaFreeHost(p);
    cpu_rec_ = cpu_pred_ = cpu_pred1_ = nullptr;
    cpu_out_ = nullptr;
    cpu_bulk_ = nullptr;
    cpu_bulk_out_ = nullptr;
}

} // namespace hyper
