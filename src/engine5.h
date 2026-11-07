// hyper engine for glm5-next (GLM-5.3-Flash) on the tensor-parallel engine: the model's layers (KDA / MLA heads with the
// indexer, mHC mixers, dense or routed + shared FFN), its MTP (NextN) block
#pragma once
#include "tp_engine.h"
#include "kernels5.cuh"
#include "model5.h"

namespace hyper {

using Engine5Options = TpOptions;

class Engine5 : public TpEngine {
public:
    Engine5(const std::string & model_path, const Engine5Options & opt);
    ~Engine5() override = default;
    const Glm5Config & config() const { return cfg_; }
    void reset() override;

private:
    struct DevLayer;
    struct Device;
    Device & dev(int g);
    const Device & dev(int g) const;
    // the engine's hooks
    std::unique_ptr<TpDevice> make_device() override;
    void load_device(TpDevice & d) override;
    const GGUF & expert_source(int il) const override { return src(il); }
    void record_main(int gi, int nt, int kind) override;
    void record_restore(int gi, int keep) override;
    bool mtp_direct() const override { return mtp_shift_; }
    size_t snap_floats(int gi) const override;
    void snap_copy(std::vector<float *> & h, bool to_host) override;
    std::vector<ConvPark<TpSnap>::Dev> park_devs() const override;
    void embed(const int * tokens, int nt, int pos, int chunk = 0) override;
    int mtp_prompt(const std::vector<int> & prompt, int c0, const int * lens, int nk, int next) override;
    int mtp_update(const int * tokens, int nt, int pos, bool draft) override;
    int mtp_chain(int token, int pos, bool from_main) override;
    // the model
    void load_layer(Device & dev, DevLayer & L, int il);
    const GGUF & src(int il) const { return il >= cfg_.n_layer ? *mtp_g_ : *gguf_; }
    void mtp_prefill(int nck);   // the MTP block shifted over the chunks of the last bulk forward (embeddings / positions kept)

    Glm5Config cfg_;
    bool mtp_shift_ = false;   // this MTP pass: prompt rows into the block's cache only (row r takes the hidden of row r - 1)
};

} // namespace hyper
