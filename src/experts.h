// The experts of a MoE layer, shared by the engines: the GGUF tensors, the routing-count order, the pinned host copy
// (2 MB pages, registered with CUDA) and the device copies with their slot tables. An engine decides who owns which
// expert; this does the copying the same way everywhere.
#pragma once
#include "cpu_moe.h"
#include "devices.h"
#include "gguf.h"
#include "kernels4.cuh"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace hyper {

constexpr int CPU_OWNER = 255;   // the owner value of an expert computed on the CPU (never a device index)

// blk.<il>.ffn_{gate,up,down}_exps.weight: the three tensors, bytes per expert matrix, their types (gate and up alike)
struct ExpertTensors {
    const GTensor * g = nullptr, * u = nullptr, * d = nullptr;
    size_t gb = 0, db = 0;   // bytes of one expert's gate (= up) / down matrix
    int E = 0;
    GType tg = GType::F32, td = GType::F32;
};
ExpertTensors expert_tensors(const GGUF & gguf, int il, int E);

// experts by routing count (descending, stable; the identity order without statistics)
template <class C> std::vector<int> expert_order(const C * counts, int E) {
    std::vector<int> order(E);
    for (int e = 0; e < E; ++e) order[e] = e;
    if (counts) std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return (*counts)[a] > (*counts)[b]; });
    return order;
}

// a pinned host copy of the experts in `list` (every expert when empty), compact: gate | up | down arrays of n experts in
// 2 MB pages (mmap + MADV_HUGEPAGE, collapsed, then cudaHostRegister; mapped: also device-mapped for zero-copy reads).
// cslot[e]: an expert's index in the arrays (-1: not copied). The caller keeps base / bytes for the unregister + munmap
struct HostExperts {
    uint8_t * base = nullptr, * gate = nullptr, * up = nullptr, * down = nullptr;
    size_t bytes = 0;
    int n = 0;
    std::vector<int> cslot;
};
HostExperts host_expert_copy(const ExpertTensors & t, const std::vector<int> & list, bool mapped = false);
// a 2 MB-page host buffer (mmap + MADV_HUGEPAGE), size rounded up; the caller unmaps
void * host_huge_alloc(size_t bytes, size_t & mapped_bytes);

// the experts with slot[e] >= 0 copied to device d at those slots: the MoeDev with the arrays, types and the slot table
MoeDev upload_experts(DevBase & d, const ExpertTensors & t, const std::vector<int> & slot, int ff, int n, float clamp);

// tensor-parallel placement (every device holds a share of every layer): experts per layer per device from each device's
// free bytes (after a reserve), its share of the prefill staging (a fixed point: the staging holds the CPU experts' share
// of the largest layer, which depends on the quotas), at most frac of a layer in all, trimmed from the largest quotas (a
// GPU with more VRAM takes more than an equal share). eb: bytes of one expert averaged over the layers, eb_max: of the
// largest layer; n_layers: MoE layers (+ an MTP block)
std::vector<int> plan_quotas_tp(const std::vector<size_t> & free_bytes, double reserve_gib, double frac, size_t eb, size_t eb_max,
                                int n_layers, int E, bool stream_experts, double stage_factor = 2.0);   // (stage_factor: 2 buffers; engine5 streams halves: 1.05)
// owners by weighted round robin over the quotas: the experts in `order`, each to the device furthest below its share
std::vector<int> assign_owners_rr(const std::vector<int> & order, const std::vector<int> & quota, int E);

} // namespace hyper
