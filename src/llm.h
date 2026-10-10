// Common interface of the hyper engines (used by the server): greedy / sampled generation with a prompt cache.
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

namespace hyper {

struct SamplingParams {
    float temp = 0.0f;           // 0: greedy
    int top_k = 0;               // 0: off (exact: the whole row is read when the kept set reaches past the GPUs' candidates)
    float top_p = 1.0f, min_p = 0.0f;
    uint64_t seed = 0;           // 0: random
};

struct GenStats {
    int tokens = 0, steps = 0, accepted = 0, drafted = 0;
    double seconds = 0;
    double t_main = 0, t_mtp = 0, t_restore = 0;   // wall time per phase
    double t_prefill = 0;
    int prompt_reused = 0;       // prompt tokens served from the prompt cache
};

class LLM {
public:
    virtual ~LLM() = default;
    // on_token: called for every generated token in order; returning false stops generation
    virtual std::vector<int> generate(const std::vector<int> & prompt, int n_gen, bool spec, GenStats * stats,
                                      const std::function<bool(int)> & on_token = {}, const SamplingParams & sp = {}) = 0;
    virtual void set_snapshot_token(int tok) = 0;
    // called after every prefill chunk with (tokens done incl. reused, prompt length, reused tokens)
    virtual void set_prefill_progress(std::function<void(int, int, int)> fn) = 0;
    virtual int max_pos() const = 0;
    virtual int n_draft() const = 0;
    virtual bool has_mtp() const = 0;
    virtual void reset_cache() = 0;   // forget the cached conversation(s); snapshot buffers stay allocated for reuse
};

} // namespace hyper
