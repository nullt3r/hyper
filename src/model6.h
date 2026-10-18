// mimo2 (MiMo-V2.6-Flash): plain pre-norm transformer with sigmoid-routed MoE.
//   attention: GQA, q/k head 192 (the first 64 dims NeoX-roped), v head 128; every 6th layer global (4 kv heads, rope base
//              1e7), the rest sliding-window 128 (8 kv heads, rope base 1e4, per-head attention sinks); output x value scale
//   ffn: layer 0 dense SwiGLU (16384), then MoE (256 experts, top 8, sigmoid + selection bias, normalized weights, no shared
//        expert); experts MXFP4
//   MTP: 3 NextN blocks in a sidecar GGUF (blk.48..50: SWA attention + dense FFN)
#pragma once
#include "gguf.h"

#include <string>
#include <vector>

namespace hyper {

struct MimoConfig {
    int n_layer = 0, n_embd = 0, n_vocab = 0, n_head = 0;
    std::vector<int> n_kv, swa, moe;   // per layer (incl. the NextN blocks of a sidecar when loaded from it)
    int dk = 0, dv = 0, n_rot = 0, window = 0;
    float rope_base = 0, rope_base_swa = 0, rms_eps = 0, v_scale = 1;
    int n_ff = 0, n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_nextn = 0;
    float w_scale = 1;

    static MimoConfig from_gguf(const GGUF & g);
    std::string describe() const;
};

} // namespace hyper
