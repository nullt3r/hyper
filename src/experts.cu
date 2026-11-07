#include "experts.h"
#include "cuda_check.h"

#include <sys/mman.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace hyper {

ExpertTensors expert_tensors(const GGUF & gguf, int il, int E) {
    const std::string p = "blk." + std::to_string(il) + ".";
    ExpertTensors t;
    t.g = &gguf.need(p + "ffn_gate_exps.weight");
    t.u = &gguf.need(p + "ffn_up_exps.weight");
    t.d = &gguf.need(p + "ffn_down_exps.weight");
    if (t.u->type != t.g->type) throw std::runtime_error("expert gate/up types differ in layer " + std::to_string(il));
    t.E = E;
    t.gb = t.g->nbytes / E; t.db = t.d->nbytes / E;
    t.tg = t.g->type; t.td = t.d->type;
    return t;
}

void * host_huge_alloc(size_t bytes, size_t & mapped_bytes) {
    const size_t H = 2u << 20;
    mapped_bytes = (bytes + H - 1) / H * H;
    void * p = mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("host_huge_alloc: mmap failed");
    madvise(p, mapped_bytes, MADV_HUGEPAGE);
    return p;
}

HostExperts host_expert_copy(const ExpertTensors & t, const std::vector<int> & list_in, bool mapped) {
    std::vector<int> list = list_in;
    if (list.empty()) { list.resize(t.E); for (int e = 0; e < t.E; ++e) list[e] = e; }
    HostExperts h;
    h.n = (int) list.size();
    h.cslot.assign(t.E, -1);
    for (int i = 0; i < h.n; ++i) h.cslot[list[i]] = i;
    const size_t gb = t.gb, db = t.db, need = (size_t) h.n * (2 * gb + db);
    h.base = (uint8_t *) host_huge_alloc(std::max<size_t>(need, 1), h.bytes);
    h.gate = h.base; h.up = h.base + (size_t) h.n * gb; h.down = h.base + (size_t) h.n * 2 * gb;
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < h.n; ++i) {
        const int e = list[i];
        memcpy(h.gate + (size_t) i * gb, t.g->data + (size_t) e * gb, gb);
        memcpy(h.up + (size_t) i * gb, t.u->data + (size_t) e * gb, gb);
        memcpy(h.down + (size_t) i * db, t.d->data + (size_t) e * db, db);
    }
    collapse_huge(h.base, h.bytes);
    CUDA_CHECK(cudaHostRegister(h.base, h.bytes, cudaHostRegisterPortable | (mapped ? cudaHostRegisterMapped : 0)));
    return h;
}

MoeDev upload_experts(DevBase & d, const ExpertTensors & t, const std::vector<int> & slot, int ff, int n, float clamp) {
    int nl = 0;
    for (int e = 0; e < t.E; ++e) nl = std::max(nl, slot[e] + 1);
    uint8_t * dg = d.alloc<uint8_t>(t.gb * std::max(nl, 1)), * du = d.alloc<uint8_t>(t.gb * std::max(nl, 1)), * dd = d.alloc<uint8_t>(t.db * std::max(nl, 1));
    for (int e = 0; e < t.E; ++e) {
        if (slot[e] < 0) continue;
        CUDA_CHECK(cudaMemcpy(dg + (size_t) slot[e] * t.gb, t.g->data + (size_t) e * t.gb, t.gb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(du + (size_t) slot[e] * t.gb, t.u->data + (size_t) e * t.gb, t.gb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dd + (size_t) slot[e] * t.db, t.d->data + (size_t) e * t.db, t.db, cudaMemcpyHostToDevice));
    }
    MoeDev m;
    m.gate = dg; m.up = du; m.down = dd;
    m.tg = t.tg; m.td = t.td;
    m.gate_bytes = t.gb; m.down_bytes = t.db;
    m.slot = d.upload(slot.data(), slot.size());
    m.ff = ff; m.n = n;
    m.clamp = clamp;
    return m;
}

std::vector<int> plan_quotas_tp(const std::vector<size_t> & free_bytes, double reserve_gib, double frac, size_t eb, size_t eb_max,
                                int n_layers, int E, bool stream_experts, double stage_factor) {
    const int nd = (int) free_bytes.size();
    std::vector<int> quota(nd, 0);
    std::vector<double> stage_est(nd, 0.0);
    for (int it = 0; it < 3; ++it) {
        int tq = 0;
        for (int g = 0; g < nd; ++g) {
            const double cap = std::max(0.0, (double) free_bytes[g] - reserve_gib * 1073741824.0 - stage_est[g]);
            quota[g] = std::min((int) (cap / (double) eb / std::max(n_layers, 1)), (int) (E * frac + 0.999));
            tq += quota[g];
        }
        for (const int lim = (int) (E * frac + 0.999); tq > lim;) {
            int gm = 0;
            for (int g = 1; g < nd; ++g) if (quota[g] > quota[gm]) gm = g;
            --quota[gm]; --tq;
        }
        if (!stream_experts) break;
        // (the staging estimate keeps the engines' historical shares: device 1 half of the others')
        const double cold = (std::max(0, E - tq) + 2) * (double) eb_max;
        for (int g = 0; g < nd; ++g) stage_est[g] = stage_factor * cold * (nd > 1 && g == 1 ? 1.0 : 2.0) / (2.0 * nd - (nd > 1 ? 1.0 : 0.0));
    }
    return quota;
}

std::vector<int> assign_owners_rr(const std::vector<int> & order, const std::vector<int> & quota, int E) {
    const int nd = (int) quota.size();
    std::vector<int> owner(E, CPU_OWNER), given(nd, 0);
    int total = 0;
    for (int q : quota) total += q;
    for (int r = 0; r < std::min(total, E); ++r) {
        int best = -1;
        double bd = -1;
        for (int g = 0; g < nd; ++g) {
            if (given[g] >= quota[g]) continue;
            const double d = 1.0 - (double) given[g] / quota[g];
            if (d > bd) { bd = d; best = g; }
        }
        if (best < 0) break;
        owner[order[r]] = best;
        given[best]++;
    }
    return owner;
}

} // namespace hyper
