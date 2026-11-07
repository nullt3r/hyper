#include "moe_place.h"
#include "cuda_check.h"

#include <sys/mman.h>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace hyper {

namespace {
const char * envp(const std::string & prefix, const char * name, std::string & buf) {
    buf = prefix + "_" + name;
    return getenv(buf.c_str());
}
} // namespace

void MoePlace::init(const Params & p, const std::vector<DevBase *> & devs, CpuMoe * cpu) {
    p_ = p;
    devs_ = devs;
    cpu_ = cpu;
    const int nd = (int) devs.size();
    std::string b;
    const std::string & env = p_.env;
    adapt_ = !envp(env, "ADAPT", b) || atoi(envp(env, "ADAPT", b)) != 0;
    if (const char * v = envp(env, "ADAPT_EVERY", b)) adapt_every_ = std::max(1, atoi(v));
    adapt_budget_ = p_.adapt_budget;
    if (const char * v = envp(env, "ADAPT_BUDGET", b)) adapt_budget_ = std::max(1, atoi(v));
    if (const char * v = envp(env, "DECAY", b)) adapt_decay_ = atof(v);
    if (const char * v = envp(env, "PROMPT_W", b)) prompt_weight_ = atof(v);
    if (const char * v = envp(env, "ADAPT_MIN", b)) adapt_min_ = atof(v);
    if (const char * v = envp(env, "ADAPT_RATIO", b)) adapt_ratio_ = atof(v);
    adapt_log_ = envp(env, "ADAPT_LOG", b) != nullptr;
    elastic_ = p_.elastic_ok && adapt_ && (!envp(env, "ELASTIC", b) || atoi(envp(env, "ELASTIC", b)) != 0);
    elastic_log_ = envp(env, "ELASTIC_LOG", b) != nullptr;
    if (const char * v = envp(env, "ELASTIC_FILL", b)) elastic_fill_ = atoi(v);
    if (const char * v = envp(env, "ELASTIC_MIN", b)) elastic_min_ = atof(v);
    stream_cpu_ = p_.stream_cpu; stream_cpu_max_ = p_.stream_cpu_max;
    if (const char * v = envp(env, "STREAM_CPU", b)) stream_cpu_ = std::max(0.0, atof(v));
    if (const char * v = envp(env, "STREAM_CPU_MAX", b)) stream_cpu_max_ = std::max(0.0, std::min(0.95, atof(v)));
    wbw_ = link_weights(nd, (env + "_EP_WEIGHTS").c_str());
    host_.assign(p_.nl, MoeLayerHost{});
    for (auto & H : host_) { H.score.assign(p_.E, 0.0); H.last_count.assign(std::max(1024, p_.E), 0); H.pend.assign(p_.E, 0); H.fixed.assign(p_.E, 0); }
    dev_.assign(nd, std::vector<MoeLayerDev>(p_.nl));
    vmm_.resize(nd);
    lend_limit_.assign(nd, 0);
    fstream_.assign(nd, nullptr);
    for (int g = 0; g < nd; ++g) {
        CUDA_CHECK(cudaSetDevice(devs_[g]->id));
        CUDA_CHECK(cudaStreamCreateWithFlags(&fstream_[g], cudaStreamNonBlocking));
    }
}

MoePlace::~MoePlace() {
    if (fill_th_.joinable()) { fill_cancel_.store(true); fill_th_.join(); }
    for (MoeLayerHost & H : host_)
        if (H.hx.base) { cudaHostUnregister(H.hx.base); munmap(H.hx.base, H.hx.bytes); H.hx.base = nullptr; }
    for (auto & v : vmm_) v.clear();
    for (size_t g = 0; g < fstream_.size(); ++g) if (fstream_[g]) { cudaSetDevice(devs_[g]->id); cudaStreamDestroy(fstream_[g]); }
    if (swap_buf_) cudaFreeHost(swap_buf_);
}

void MoePlace::load_stats(const char * path, int n_layer) {
    FILE * f = fopen(path, "rb");
    if (!f) return;
    int nl = 0, w = 0;
    // (n_layer + 1 rows: written with an MTP block)
    if (fread(&nl, 4, 1, f) == 1 && fread(&w, 4, 1, f) == 1 && (nl == n_layer || nl == n_layer + 1)) {
        stats_.assign(nl, std::vector<uint64_t>(w));
        for (auto & s : stats_) if (fread(s.data(), 8, w, f) != (size_t) w) { stats_.clear(); break; }
    }
    fclose(f);
    fprintf(stderr, "%s: expert placement from %s (%s)\n", p_.tag.c_str(), path, stats_.empty() ? "unreadable, ignored" : "ok");
}

std::vector<int> MoePlace::plan_tp(const std::vector<size_t> & free_bytes, double reserve_gib, double frac, size_t eb, size_t eb_max,
                                   int n_moe_layers, bool stream, double stage_factor) {
    std::vector<int> quota = plan_quotas_tp(free_bytes, reserve_gib, frac, eb, eb_max, std::max(n_moe_layers, 1), p_.E, stream, stage_factor);
    if (const char * q = getenv("HYPER_QUOTA")) {   // (tests: fixed experts per layer per device, "130,128,126", for identical placements across builds)
        size_t g = 0;
        for (const char * c = q; *c && g < quota.size(); ++g) { quota[g] = std::min(p_.E, std::max(0, atoi(c))); while (*c && *c != ',') ++c; if (*c) ++c; }
    }
    int tq = 0;
    for (int q : quota) tq += q;
    fprintf(stderr, "%s: experts per layer on GPUs:", p_.tag.c_str());
    for (int q : quota) fprintf(stderr, " %d", q);
    fprintf(stderr, " (%.0f%% of %d), the rest on the CPU\n", 100.0 * tq / p_.E, p_.E);
    return quota;
}

void MoePlace::load_layer(int il, const ExpertTensors & t, const std::vector<int> & owner, const std::vector<char> * host_more,
                          const std::vector<char> * fixed, const std::vector<int> * owner_bulk, const std::vector<char> * on_devs) {
    const int E = p_.E, nd = (int) devs_.size();
    MoeLayerHost & H = host_[il];
    H.t = t;
    H.owner = owner;
    H.slot.assign(nd, std::vector<int>(E, -1));
    std::vector<int> count(nd, 0);
    for (int e = 0; e < E; ++e) if (owner[e] != CPU_OWNER) H.slot[owner[e]][e] = count[owner[e]]++;
    if (fixed) H.fixed = *fixed;
    // the compact host copy: the CPU's experts and the ones the engine wants there too
    std::vector<int> list;
    static const bool host_full = getenv("HYPER_HOST_FULL") != nullptr;   // (tests: every expert on the host, in expert order)
    for (int e = 0; e < E; ++e) if (host_full || owner[e] == CPU_OWNER || (host_more && (*host_more)[e])) list.push_back(e);
    H.cslot.assign(E, -1);
    const auto t0 = std::chrono::steady_clock::now();
    if (!list.empty()) {
        H.hx = host_expert_copy(t, list);
        H.cslot = H.hx.cslot;
    }
    t_hostcopy_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // the devices: every device takes part in a tensor-parallel layer (owner tables, stream lists); otherwise those with experts
    for (int g = 0; g < nd; ++g) {
        if (on_devs ? !(*on_devs)[g] : (!p_.tp && count[g] == 0)) continue;
        DevBase & d = *devs_[g];
        MoeLayerDev & D = dev_[g][il];
        CUDA_CHECK(cudaSetDevice(d.id));
        const int nl = count[g];
        uint8_t * dg, * du, * dd;
        if (elastic_ && lend_limit_[g] > 0) {   // virtual ranges for every expert, the base slots mapped
            auto mk = [&](size_t per) {
                auto b = std::make_unique<VmmBuf>();
                b->reserve(d.id, per * E);
                b->map(0, per * nl, false);
                VmmBuf * r = b.get();
                vmm_[g].push_back(std::move(b));
                d.used += r->mapped();
                return r;
            };
            D.xg = mk(t.gb); D.xu = mk(t.gb); D.xd = mk(t.db);
            D.nbase = nl;
            D.xbase = D.xg->mapped() + D.xu->mapped() + D.xd->mapped();
            D.xe.clear();
            dg = (uint8_t *) D.xg->ptr(); du = (uint8_t *) D.xu->ptr(); dd = (uint8_t *) D.xd->ptr();
        } else {
            dg = d.alloc<uint8_t>(t.gb * std::max(nl, 1)); du = d.alloc<uint8_t>(t.gb * std::max(nl, 1));
            dd = d.alloc<uint8_t>(t.db * std::max(nl, 1));
        }
        const std::vector<int> & slot = H.slot[g];
        for (int e = 0; e < E; ++e) {
            if (slot[e] < 0) continue;
            CUDA_CHECK(cudaMemcpy(dg + (size_t) slot[e] * t.gb, t.g->data + (size_t) e * t.gb, t.gb, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(du + (size_t) slot[e] * t.gb, t.u->data + (size_t) e * t.gb, t.gb, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dd + (size_t) slot[e] * t.db, t.d->data + (size_t) e * t.db, t.db, cudaMemcpyHostToDevice));
        }
        D.moe.gate = dg; D.moe.up = du; D.moe.down = dd;
        D.moe.tg = t.tg; D.moe.td = t.td;
        D.moe.gate_bytes = t.gb; D.moe.down_bytes = t.db;
        D.moe.ff = p_.ff; D.moe.n = p_.n;
        D.moe.clamp = p_.clamp;
        D.moe.slot = d.upload(slot.data(), E);
        D.owner = d.upload(owner.data(), E);
        D.owner_bulk = d.upload(owner_bulk ? owner_bulk->data() : owner.data(), E);
        for (int h = 0; h < p_.halves; ++h) D.st_slot[h] = d.alloc<int>(E);
    }
    for (const GTensor * tt : {t.g, t.u, t.d}) {   // (the file's pages are not needed any more)
        const uintptr_t a0 = ((uintptr_t) tt->data + 4095) & ~(uintptr_t) 4095, a1 = ((uintptr_t) tt->data + tt->nbytes) & ~(uintptr_t) 4095;
        if (a1 > a0) madvise((void *) a0, a1 - a0, MADV_DONTNEED);
    }
    // the CPU side: the compact copy, its index, what it owns
    CpuExpertLayer cl;
    cl.tg = t.tg; cl.td = t.td;
    cl.gate_bytes = t.gb; cl.down_bytes = t.db;
    cl.gate = H.hx.gate; cl.up = H.hx.up; cl.down = H.hx.down;
    cl.cslot = H.cslot;
    cl.owned.resize(E);
    for (int e = 0; e < E; ++e) cl.owned[e] = owner[e] == CPU_OWNER;
    if (host_more) {   // (the bulk path: the CPU computes those experts too)
        cl.owned_bulk.resize(E);
        for (int e = 0; e < E; ++e) cl.owned_bulk[e] = cl.owned[e] || (*host_more)[e];
    }
    cpu_->set_layer(il, cl);
    if (const std::vector<uint64_t> * st = stats_of(il)) {   // prior: the calibration's routing shares, weighted like ~256 tokens of routing
        double tot = 0;
        for (int e = 0; e < E; ++e) tot += (double) (*st)[e];
        if (tot > 0) for (int e = 0; e < E; ++e) H.score[e] = (double) (*st)[e] / tot * 256.0 * p_.K;
    }
    H.loaded = true;
    H.tables_dirty = false;
    H.stream_dirty = p_.halves > 0;
}

// ---- tensor-parallel prefill streaming ---------------------------------------------------------------------------------

// The count per layer only drops below its load-time value (elastic extra slots), but a rounded share is not monotone in it
// (209 -> 84 41 84, 208 -> 83 42 83): staging for ceil(nc w / tw) + 1 bounds the share of any count up to nc
void MoePlace::size_staging(int il) {
    const MoeLayerHost & H = host_[il];
    int nc = 0;
    for (int e = 0; e < p_.E; ++e) nc += H.owner[e] == CPU_OWNER;
    if (nc == 0) return;
    double tw = 0;
    for (double w : wbw_) tw += w;
    for (size_t g = 0; g < devs_.size(); ++g) {
        const int share = std::min(nc, (int) std::ceil(nc * wbw_[g] / tw) + 1);
        const int per = p_.halves == 2 ? (share + 1) / 2 : share;
        devs_[g]->stage_bytes = std::max(devs_[g]->stage_bytes, (size_t) per * (2 * H.gb() + H.db()));
    }
}

void MoePlace::rebuild_stream(int il) {
    MoeLayerHost & H = host_[il];
    const int nd = (int) devs_.size(), E = p_.E;
    std::vector<int> cpu;
    for (int e = 0; e < E; ++e) if (H.owner[e] == CPU_OWNER) cpu.push_back(e);
    // a single streamed chunk leaves the coldest stream_cpu share to the CPU: experts the chunk does not route cost nothing
    // there, those of a few tokens little, and none of them crosses PCIe
    std::vector<uint8_t> keep;
    if (stream_cpu_built_ > 0.0 && !cpu.empty()) {
        std::stable_sort(cpu.begin(), cpu.end(), [&](int a, int b) { return H.score[a] < H.score[b]; });
        const int nk = std::min((int) cpu.size(), (int) (cpu.size() * stream_cpu_built_ + 0.5));
        keep.assign(E, 0);
        for (int i = 0; i < nk; ++i) keep[cpu[i]] = 1;
        cpu.erase(cpu.begin(), cpu.begin() + nk);
        std::sort(cpu.begin(), cpu.end());
    }
    cpu_->set_owned_bulk(il, keep);
    for (int g = 0; g < nd; ++g) dev_[g][il].cpu_keep = !keep.empty();
    const int nc = (int) cpu.size();
    double tw = 0, acc = 0;
    for (double w : wbw_) tw += w;
    std::vector<int> bulk_owner = H.owner;
    int a = 0;
    for (int g = 0; g < nd; ++g) {
        MoeLayerDev & D = dev_[g][il];
        DevBase & d = *devs_[g];
        acc += wbw_[g];
        const int b = g == nd - 1 ? nc : (int) (nc * acc / tw + 0.5);
        int r0[2] = {a, b}, r1[2] = {b, b};
        if (p_.halves == 2) { const int m = (a + b) / 2; r0[0] = a; r1[0] = m; r0[1] = m; r1[1] = b; }
        CUDA_CHECK(cudaSetDevice(d.id));
        for (int h = 0; h < p_.halves; ++h) {
            D.st_list[h].assign(cpu.begin() + r0[h], cpu.begin() + r1[h]);
            if ((size_t) D.st_list[h].size() * (2 * H.gb() + H.db()) > d.stage_bytes && d.stage[0]) throw std::runtime_error("rebuild_stream: staging too small");
            std::vector<int> sl(E, -1);
            for (int i = 0; i < (int) D.st_list[h].size(); ++i) { sl[D.st_list[h][i]] = i; bulk_owner[D.st_list[h][i]] = g; }
            CUDA_CHECK(cudaMemcpy(D.st_slot[h], sl.data(), E * sizeof(int), cudaMemcpyHostToDevice));
        }
        a = b;
    }
    for (int g = 0; g < nd; ++g) {
        CUDA_CHECK(cudaSetDevice(devs_[g]->id));
        CUDA_CHECK(cudaMemcpy(dev_[g][il].owner_bulk, bulk_owner.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    }
    H.stream_dirty = false;
}

void MoePlace::rebuild_dirty_streams() {
    if (p_.halves == 0) return;
    for (int il = 0; il < p_.nl; ++il) if (host_[il].loaded && host_[il].stream_dirty) rebuild_stream(il);
}

// the CPU's share falls with the chunk's tokens (its cost grows with them, the uploads' does not): min(max, c / nt) with
// c = stream_cpu tokens (pfsweep, Maya-L at 4k: 128 / 200 / 400 / 700 tokens best at 0.5 / 0.5 / 0.35 / 0.2, 1000 at ~0;
// 2.00 -> 1.32 s, 2.03 -> 1.52, 2.07 -> 1.76, 2.13 -> 1.97), in steps of 0.05 (the lists are rebuilt on a change)
void MoePlace::set_stream_cpu_for(int nt, int stream_min) {
    double want = nt >= stream_min && stream_cpu_ > 0 ? std::min(stream_cpu_max_, stream_cpu_ / nt) : 0.0;
    want = std::floor(want * 20.0 + 0.5) / 20.0;
    set_stream_cpu(want);
}

void MoePlace::set_stream_cpu(double share) {
    if (share == stream_cpu_built_) return;
    cpu_->drain();
    stream_cpu_built_ = share;
    for (MoeLayerHost & H : host_) if (H.loaded) H.stream_dirty = true;
}

void MoePlace::upload_stage(int g, int il, int sb, int p, int np) {
    DevBase & d = *devs_[g];
    MoeLayerDev & D = dev_[g][il];
    const MoeLayerHost & H = host_[il];
    const std::vector<int> & lst = D.st_list[p_.halves == 2 ? sb : 0];
    const int cnt = (int) lst.size();
    if (np < 1) np = 1;
    const int i0 = (int) ((int64_t) cnt * p / np), i1 = (int) ((int64_t) cnt * (p + 1) / np);
    if (p == 0) CUDA_CHECK(cudaStreamWaitEvent(d.cstream, d.ev_free[sb], 0));
    const size_t gb = H.gb(), db = H.db();
    for (int i = i0; i < i1;) {   // runs of consecutive compact rows in one copy each
        int j = i + 1;
        while (j < i1 && H.cslot[lst[j]] == H.cslot[lst[j - 1]] + 1) ++j;
        const size_t c = (size_t) H.cslot[lst[i]], r = (size_t) (j - i);
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) i * gb, H.hx.gate + c * gb, r * gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) (cnt + i) * gb, H.hx.up + c * gb, r * gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync(d.stage[sb] + (size_t) 2 * cnt * gb + (size_t) i * db, H.hx.down + c * db, r * db, cudaMemcpyHostToDevice, d.cstream));
        i = j;
    }
    if (p == np - 1) CUDA_CHECK(cudaEventRecord(d.ev_up[sb], d.cstream));
}

// ---- adaptive placement ---------------------------------------------------------------------------------------------

void MoePlace::before_forward(bool bulk) {
    if (bulk) { rebuild_dirty_streams(); return; }
    if (elastic_) {
        ++dec_steps_;
        if (!fill_list_.empty() && dec_steps_ >= fill_due_) fill_wait(false);   // (the batch's due step: waits if needed)
    }
    if (adapt_) {   // placement follows the routing: after a prompt, then every adapt_every_ decode steps
        if (prompt_routed_ && p_.adapt_after_prompt) { rebalance(1 << 20); prompt_routed_ = false; steps_ = 0; }
        else if (++steps_ % adapt_every_ == 0) rebalance(adapt_budget_);
    }
}

void MoePlace::score_prompt(int il, const int * ids, int count) {
    if (!adapt_) return;
    auto & sc = host_[il].score;
    for (int i = 0; i < count; ++i) { const int e = ids[i]; if (e >= 0 && e < p_.E) sc[e] += prompt_weight_; }
}

void MoePlace::sync_devices() {
    for (DevBase * d : devs_) { CUDA_CHECK(cudaSetDevice(d->id)); CUDA_CHECK(cudaDeviceSynchronize()); }
}

void MoePlace::upload_tables(int il) {
    MoeLayerHost & H = host_[il];
    for (size_t g = 0; g < devs_.size(); ++g) {
        MoeLayerDev & D = dev_[g][il];
        if (!D.present()) continue;
        CUDA_CHECK(cudaSetDevice(devs_[g]->id));
        CUDA_CHECK(cudaMemcpy((void *) D.moe.slot, H.slot[g].data(), p_.E * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(D.owner, H.owner.data(), p_.E * sizeof(int), cudaMemcpyHostToDevice));
    }
    H.tables_dirty = false;
    if (on_changed) on_changed(il);
}

// Routing scores (decayed counts: prompt routing of the prefill chunks, the CPU side's decode counts) pull the hottest CPU
// experts onto the GPUs in exchange for their coldest experts (same GPU, same slot), at most max_swaps exchanges in all,
// the largest gains first. Runs between forwards. A swap costs ~0.5 ms of PCIe + a decode stall and pays ~0.2 ms per
// future hit: only clear, persistent differences (score_in >= ratio * score_out + min) are taken.
void MoePlace::rebalance(int max_swaps) {
    const int E = p_.E;
    const auto t_reb = std::chrono::steady_clock::now();
    // every record of the last forward consumed first: layers whose experts were all on the GPUs are not waited for by the
    // GPU, so their routing counts (and a still-running job) could lag -- placement then depended on timing
    cpu_->drain();
    for (int il = 0; il < p_.nl; ++il) {
        MoeLayerHost & H = host_[il];
        if (!H.loaded) continue;
        const auto & cnt = cpu_->counts[il];
        for (int e = 0; e < E; ++e) { H.score[e] = H.score[e] * adapt_decay_ + (double) (cnt[e] - H.last_count[e]); H.last_count[e] = cnt[e]; }
    }
    if (elastic_) fill_extra(elastic_fill_);   // free extra slots first (a background batch)
    struct Sw { double gain; int il, ein, eout, g; };
    std::vector<Sw> cand;
    for (int il = 0; il < p_.nl; ++il) {
        MoeLayerHost & H = host_[il];
        if (!H.loaded) continue;
        std::vector<int> cpu, gpu;
        for (int e = 0; e < E; ++e) {
            if (H.pend[e] || H.fixed[e]) continue;   // (on its way into an extra slot; pooled for good)
            (H.owner[e] == CPU_OWNER ? cpu : gpu).push_back(e);
        }
        std::sort(cpu.begin(), cpu.end(), [&](int a, int b) { return H.score[a] > H.score[b]; });
        std::sort(gpu.begin(), gpu.end(), [&](int a, int b) { return H.score[a] < H.score[b]; });
        for (size_t i = 0; i < std::min(cpu.size(), gpu.size()); ++i) {
            const double si = H.score[cpu[i]], so = H.score[gpu[i]];
            if (si < so * adapt_ratio_ + adapt_min_) break;
            cand.push_back({si - so, il, cpu[i], gpu[i], H.owner[gpu[i]]});
        }
    }
    std::sort(cand.begin(), cand.end(), [](const Sw & a, const Sw & b) { return a.gain > b.gain; });
    if ((int) cand.size() > max_swaps) cand.resize(max_swaps);
    if (cand.empty()) return;
    // nothing may read an expert being exchanged (the decode graphs of the last forward are done; the issuing threads too)
    sync_devices();
    // a base-slot expert's rows are not on the host: they come down into the CPU expert's compact slot (exchange); an
    // extra-slot expert keeps its compact rows, and so does the CPU expert that takes its extra slot
    size_t need = 0;
    for (const Sw & w : cand) if (host_[w.il].cslot[w.eout] < 0) need += 2 * host_[w.il].gb() + host_[w.il].db();
    if (need > swap_buf_bytes_) {
        if (swap_buf_) CUDA_CHECK(cudaFreeHost(swap_buf_));
        swap_buf_bytes_ = need;
        CUDA_CHECK(cudaHostAlloc(&swap_buf_, swap_buf_bytes_, cudaHostAllocPortable));
    }
    std::vector<size_t> boff(cand.size(), (size_t) -1);
    size_t off = 0;
    for (size_t j = 0; j < cand.size(); ++j) {   // weights first (copy engines of all GPUs in parallel), then the tables
        const Sw & w = cand[j];
        MoeLayerHost & H = host_[w.il];
        DevBase & d = *devs_[w.g];
        MoeLayerDev & D = dev_[w.g][w.il];
        const size_t gb = H.gb(), db = H.db();
        const int k = H.slot[w.g][w.eout], i = H.cslot[w.ein];
        if (i < 0) throw std::runtime_error("rebalance: a CPU expert without compact rows");
        if (D.xg && k >= D.nbase) D.xe[k - D.nbase] = w.ein;   // (elastic: an extra slot changes hands)
        CUDA_CHECK(cudaSetDevice(d.id));
        if (H.cslot[w.eout] < 0) {
            uint8_t * b = swap_buf_ + off;
            boff[j] = off;
            off += 2 * gb + db;
            CUDA_CHECK(cudaMemcpyAsync(b, (const uint8_t *) D.moe.gate + (size_t) k * gb, gb, cudaMemcpyDeviceToHost, d.cstream));
            CUDA_CHECK(cudaMemcpyAsync(b + gb, (const uint8_t *) D.moe.up + (size_t) k * gb, gb, cudaMemcpyDeviceToHost, d.cstream));
            CUDA_CHECK(cudaMemcpyAsync(b + 2 * gb, (const uint8_t *) D.moe.down + (size_t) k * db, db, cudaMemcpyDeviceToHost, d.cstream));
        }
        CUDA_CHECK(cudaMemcpyAsync((uint8_t *) D.moe.gate + (size_t) k * gb, H.hx.gate + (size_t) i * gb, gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync((uint8_t *) D.moe.up + (size_t) k * gb, H.hx.up + (size_t) i * gb, gb, cudaMemcpyHostToDevice, d.cstream));
        CUDA_CHECK(cudaMemcpyAsync((uint8_t *) D.moe.down + (size_t) k * db, H.hx.down + (size_t) i * db, db, cudaMemcpyHostToDevice, d.cstream));
    }
    for (DevBase * d : devs_) { CUDA_CHECK(cudaSetDevice(d->id)); CUDA_CHECK(cudaStreamSynchronize(d->cstream)); }
#pragma omp parallel for schedule(static, 1)
    for (size_t j = 0; j < cand.size(); ++j) {   // the GPU expert's rows into the CPU expert's compact slot
        if (boff[j] == (size_t) -1) continue;
        const Sw & w = cand[j];
        MoeLayerHost & H = host_[w.il];
        const size_t gb = H.gb(), db = H.db();
        const int i = H.cslot[w.ein];
        const uint8_t * b = swap_buf_ + boff[j];
        memcpy(H.hx.gate + (size_t) i * gb, b, gb);
        memcpy(H.hx.up + (size_t) i * gb, b + gb, gb);
        memcpy(H.hx.down + (size_t) i * db, b + 2 * gb, db);
    }
    for (size_t j = 0; j < cand.size(); ++j) {
        const Sw & w = cand[j];
        MoeLayerHost & H = host_[w.il];
        const int k = H.slot[w.g][w.eout];
        H.slot[w.g][w.ein] = k;
        H.slot[w.g][w.eout] = -1;
        H.owner[w.ein] = w.g;
        H.owner[w.eout] = CPU_OWNER;
        if (boff[j] != (size_t) -1) {   // exchanged rows: the compact slot changes hands
            const int i = H.cslot[w.ein];
            H.cslot[w.eout] = i; H.cslot[w.ein] = -1;
            cpu_->set_cslot(w.il, w.eout, i); cpu_->set_cslot(w.il, w.ein, -1);
        }
        cpu_->set_owned(w.il, w.ein, false);
        cpu_->set_owned(w.il, w.eout, true);
        H.tables_dirty = true;
        H.stream_dirty = true;
    }
    for (int il = 0; il < p_.nl; ++il) if (host_[il].loaded && host_[il].tables_dirty) upload_tables(il);
    n_swaps_ += (int) cand.size();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_reb).count();
    t_rebalance_ += ms;
    if (adapt_log_) fprintf(stderr, "%s: rebalance: %zu exchanges in %.1f ms (total %d, %.1f s)\n", p_.tag.c_str(), cand.size(), ms, n_swaps_, t_rebalance_ / 1e3);
}

// ---- elastic extra slots ---------------------------------------------------------------------------------------------

size_t MoePlace::extra_mapped(int g) const {
    size_t used = 0;
    for (int il = 0; il < p_.nl; ++il) {
        const MoeLayerDev & D = dev_[g][il];
        if (D.xg) used += D.xg->mapped() + D.xu->mapped() + D.xd->mapped() - D.xbase;
    }
    return used;
}

size_t MoePlace::lend_free(int g) const {
    const size_t used = (kv_mapped ? kv_mapped(g) : 0) + extra_mapped(g);
    return lend_limit_[g] > used ? lend_limit_[g] - used : 0;
}

void MoePlace::before_kv_change() {
    fill_wait(true);
    cpu_->drain();
    sync_devices();
}

// give back at least `bytes` of extra expert slots on device g: the top extra slot of the layer whose expert there is the
// coldest, one at a time (the expert becomes a CPU expert again; its compact rows are still there)
void MoePlace::evict_extra(int g, size_t bytes) {
    const int E = p_.E;
    size_t freed = 0;
    int n = 0;
    CUDA_CHECK(cudaSetDevice(devs_[g]->id));
    while (freed < bytes) {
        int best = -1;
        double bs = 0;
        for (int il = 0; il < p_.nl; ++il) {
            const MoeLayerDev & D = dev_[g][il];
            if (!D.xg || D.xe.empty()) continue;
            const int e = D.xe.back();
            const double sc = e < 0 ? -1.0 : host_[il].score[e];
            if (best < 0 || sc < bs) { best = il; bs = sc; }
        }
        if (best < 0) break;
        MoeLayerDev & D = dev_[g][best];
        MoeLayerHost & H = host_[best];
        const int e = D.xe.back(), k = D.nbase + (int) D.xe.size() - 1;
        if (e >= 0) {
            if (H.cslot[e] < 0) throw std::runtime_error("evict_extra: an extra-slot expert without compact rows");
            H.owner[e] = CPU_OWNER;
            H.slot[g][e] = -1;
            cpu_->set_owned(best, e, true);
            H.tables_dirty = H.stream_dirty = true;
            ++n;
        }
        D.xe.pop_back();
        freed += D.xg->unmap((size_t) k * H.gb(), D.xg->reserved()) + D.xu->unmap((size_t) k * H.gb(), D.xu->reserved()) +
                 D.xd->unmap((size_t) k * H.db(), D.xd->reserved());
    }
    for (int il = 0; il < p_.nl; ++il) if (host_[il].loaded && host_[il].tables_dirty) upload_tables(il);
    if (elastic_log_) fprintf(stderr, "%s: elastic: device %d gave back %d experts (%.0f MiB)\n", p_.tag.c_str(), devs_[g]->id, n, freed / 1048576.0);
    (void) E;
}

// hottest CPU experts into new extra slots (no exchange: memory the caches do not use), the device with the most room
// first; keeps one cache growth step free per device. The slots are reserved here; a host thread maps and copies them in
// the background (fill_apply switches the copied ones over between forwards)
int MoePlace::fill_extra(int budget) {
    if (!elastic_ || budget <= 0 || fill_busy_.load() || !fill_list_.empty()) return 0;
    const int E = p_.E, nd = (int) devs_.size();
    std::vector<size_t> fr(nd, 0), headroom(nd, 0);
    bool any = false;
    std::string b;
    static const bool no_head = envp(p_.env, "ELASTIC_NOHEADROOM", b) != nullptr;   // (tests: every growth step evicts)
    for (int g = 0; g < nd; ++g) {
        if (lend_limit_[g] == 0) continue;
        fr[g] = lend_free(g);
        headroom[g] = no_head || !kv_headroom ? 0 : kv_headroom(g);
        any |= fr[g] > headroom[g];
    }
    if (!any) return 0;
    struct C { double s; int il, e; };
    std::vector<C> cand;
    for (int il = 0; il < p_.nl; ++il) {
        const MoeLayerHost & H = host_[il];
        if (!H.loaded) continue;
        for (int e = 0; e < E; ++e) if (H.owner[e] == CPU_OWNER && H.score[e] >= elastic_min_) cand.push_back({H.score[e], il, e});
    }
    std::sort(cand.begin(), cand.end(), [](const C & a, const C & b) { return a.s > b.s; });
    for (const C & c : cand) {
        if ((int) fill_list_.size() >= budget) break;
        MoeLayerHost & H = host_[c.il];
        int best = -1;
        size_t bleft = 0, bcost = 0;
        for (int g = 0; g < nd; ++g) {
            const MoeLayerDev & D = dev_[g][c.il];
            if (!D.xg) continue;
            const int k = D.nbase + (int) D.xe.size();
            if (k >= E) continue;
            // (reserved slots are not mapped yet: the cost counts their granules once, slot by slot in order)
            const size_t cost = D.xg->cost((size_t) k * H.gb(), H.gb()) + D.xu->cost((size_t) k * H.gb(), H.gb()) + D.xd->cost((size_t) k * H.db(), H.db());
            if (fr[g] < cost + headroom[g]) continue;
            const size_t left = fr[g] - cost;
            if (best < 0 || left > bleft) { best = g; bleft = left; bcost = cost; }
        }
        if (best < 0) continue;
        MoeLayerDev & D = dev_[best][c.il];
        const int k = D.nbase + (int) D.xe.size();
        D.xe.push_back(c.e);
        H.pend[c.e] = 1;
        fill_list_.push_back({c.il, c.e, best, k, false});
        fr[best] -= bcost;
    }
    if (fill_list_.empty()) return 0;
    // (cost above is approximate for consecutive reserved slots sharing a granule: it can only overcount)
    // deterministic hand-over: at a fixed decode step (16 per fill_unit experts, done long before on any link), not when
    // the copies happen to finish; a cancelled batch is dropped as a whole
    fill_due_ = dec_steps_ + 16 * (int) ((fill_list_.size() + p_.fill_unit - 1) / p_.fill_unit);
    fill_busy_.store(true);
    fill_cancel_.store(false);
    if (fill_th_.joinable()) fill_th_.join();
    fill_th_ = std::thread([this] {
        pthread_setname_np(pthread_self(), "moe-fill");
        try {
            for (Fill & f : fill_list_) {
                if (fill_cancel_.load()) break;
                DevBase & d = *devs_[f.g];
                MoeLayerDev & D = dev_[f.g][f.il];
                const MoeLayerHost & H = host_[f.il];
                const size_t gb = H.gb(), db = H.db(), i = (size_t) H.cslot[f.e];
                CUDA_CHECK(cudaSetDevice(d.id));
                D.xg->map((size_t) f.k * gb, gb, false);
                D.xu->map((size_t) f.k * gb, gb, false);
                D.xd->map((size_t) f.k * db, db, false);
                CUDA_CHECK(cudaMemcpyAsync((uint8_t *) D.moe.gate + (size_t) f.k * gb, H.hx.gate + i * gb, gb, cudaMemcpyHostToDevice, fstream_[f.g]));
                CUDA_CHECK(cudaMemcpyAsync((uint8_t *) D.moe.up + (size_t) f.k * gb, H.hx.up + i * gb, gb, cudaMemcpyHostToDevice, fstream_[f.g]));
                CUDA_CHECK(cudaMemcpyAsync((uint8_t *) D.moe.down + (size_t) f.k * db, H.hx.down + i * db, db, cudaMemcpyHostToDevice, fstream_[f.g]));
                f.copied = true;
            }
            for (size_t g = 0; g < devs_.size(); ++g) { CUDA_CHECK(cudaSetDevice(devs_[g]->id)); CUDA_CHECK(cudaStreamSynchronize(fstream_[g])); }
        } catch (const std::exception & ex) {
            fprintf(stderr, "%s: elastic fill failed: %s\n", p_.tag.c_str(), ex.what());
            for (Fill & f : fill_list_) f.copied = false;
            for (size_t g = 0; g < devs_.size(); ++g) { cudaSetDevice(devs_[g]->id); cudaStreamSynchronize(fstream_[g]); }
        }
        fill_busy_.store(false);
    });
    return 0;   // (nothing changes hands until fill_apply)
}

// between forwards: the copied experts of a finished batch take over their extra slots; reservations that were not copied
// (cancelled batch) become free slots
void MoePlace::fill_apply() {
    if (fill_list_.empty() || fill_busy_.load()) return;
    if (fill_th_.joinable()) fill_th_.join();
    if (fill_cancel_.load()) for (Fill & f : fill_list_) f.copied = false;   // (how far it got depends on timing)
    cpu_->drain();
    int n = 0;
    std::vector<char> dirty(p_.nl, 0);
    for (const Fill & f : fill_list_) {
        MoeLayerHost & H = host_[f.il];
        MoeLayerDev & D = dev_[f.g][f.il];
        H.pend[f.e] = 0;
        if (f.copied) {
            H.slot[f.g][f.e] = f.k;
            H.owner[f.e] = f.g;
            cpu_->set_owned(f.il, f.e, false);
            H.stream_dirty = true;
            dirty[f.il] = 1;
            ++n;
        } else {
            D.xe[f.k - D.nbase] = -1;   // (a free slot: evictions pop it, fills only append)
        }
    }
    // a cancelled batch: its reservations are the tops of their layers' extra regions -- they give their memory back
    for (const Fill & f : fill_list_) {
        if (f.copied) continue;
        MoeLayerDev & D = dev_[f.g][f.il];
        const MoeLayerHost & H = host_[f.il];
        CUDA_CHECK(cudaSetDevice(devs_[f.g]->id));
        while (!D.xe.empty() && D.xe.back() < 0) {
            const int k = D.nbase + (int) D.xe.size() - 1;
            D.xe.pop_back();
            D.xg->unmap((size_t) k * H.gb(), D.xg->reserved());
            D.xu->unmap((size_t) k * H.gb(), D.xu->reserved());
            D.xd->unmap((size_t) k * H.db(), D.xd->reserved());
        }
    }
    for (int il = 0; il < p_.nl; ++il) if (dirty[il]) { host_[il].tables_dirty = true; upload_tables(il); }
    if (elastic_log_) {
        int ex = 0;
        for (size_t g = 0; g < devs_.size(); ++g) for (int il = 0; il < p_.nl; ++il) for (int e : dev_[g][il].xe) ex += e >= 0;
        fprintf(stderr, "%s: elastic: %d experts moved to extra slots (%zu reserved; %d in extra slots, %.1f per layer)\n", p_.tag.c_str(), n,
                fill_list_.size(), ex, (double) ex / std::max(1, p_.nl));
    }
    fill_list_.clear();
}

void MoePlace::fill_wait(bool cancel) {
    if (fill_list_.empty()) return;
    if (cancel) fill_cancel_.store(true);
    if (fill_th_.joinable()) fill_th_.join();
    fill_apply();
}

void MoePlace::save_stats(const std::string & path) {
    const int nl = p_.nl, w = 1024;
    std::vector<std::vector<uint64_t>> tot(nl, std::vector<uint64_t>(w, 0));
    for (int l = 0; l < nl; ++l)
        for (int e = 0; e < w; ++e) {
            if (l < (int) stats_.size() && e < (int) stats_[l].size()) tot[l][e] += stats_[l][e];
            if (l < (int) cpu_->counts.size() && e < (int) cpu_->counts[l].size()) tot[l][e] += cpu_->counts[l][e];
        }
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    fwrite(&nl, 4, 1, f); fwrite(&w, 4, 1, f);
    for (auto & cc : tot) fwrite(cc.data(), 8, w, f);
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}

} // namespace hyper
