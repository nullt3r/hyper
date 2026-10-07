// Golden reference: run mainline llama.cpp on a prompt and dump token ids + full logits for every
// position. hyper's correctness check replays the same tokens and compares distributions.
//
// usage: ref <model.gguf> <prompt.txt> <out.bin> [n_gpu_layers=999] [max_tokens=0]
// out.bin: int32 n_tokens, int32 n_vocab, int32 tokens[n_tokens], float logits[n_tokens][n_vocab]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// REF_DUMP=regex-ish prefixes (comma separated): print the sum of every matching tensor (debugging layer by layer)
static std::vector<std::string> g_dump;
static bool dump_cb(struct ggml_tensor * t, bool ask, void *) {
    const std::string name = t->name;
    bool match = false;
    for (auto & p : g_dump) if (name.rfind(p, 0) == 0) match = true;
    const char * raw = getenv("REF_DUMPRAW");   // exact tensor name: raw data to /tmp/<name>.bin (+ shape on stderr)
    if (raw && name == raw) {
        if (ask) return true;
        std::vector<uint8_t> buf(ggml_nbytes(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
        FILE * f = fopen((std::string("/tmp/") + name + ".bin").c_str(), "wb");
        fwrite(buf.data(), 1, buf.size(), f);
        fclose(f);
        fprintf(stderr, "DUMPRAW %s type %d ne %lld %lld %lld\n", name.c_str(), (int) t->type, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
        return true;
    }
    if (ask) return match;
    if (!match || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return true;
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    double s = 0, s2 = 0;
    for (float v : buf) { s += v; s2 += (double) v * v; }
    fprintf(stderr, "DUMP %-24s sum %.6g  sumsq %.6g  n %lld\n", name.c_str(), s, s2, (long long) ggml_nelements(t));
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf prompt.txt out.bin [n_gpu_layers] [max_tokens]\n", argv[0]);
        return 1;
    }
    const int ngl = argc > 4 ? atoi(argv[4]) : 999;
    const int max_tokens = argc > 5 ? atoi(argv[5]) : 0;

    std::ifstream in(argv[2]);
    std::stringstream ss; ss << in.rdbuf();
    const std::string prompt = ss.str();

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    // REF_CPU_MOE=1: routed experts stay in host memory (models larger than the VRAM)
    static llama_model_tensor_buft_override ovr[2] = {};
    if (getenv("REF_CPU_MOE")) {
        ovr[0].pattern = "_exps";
        ovr[0].buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        mp.tensor_buft_overrides = ovr;
    }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    if (max_tokens > 0 && (int) toks.size() > max_tokens) toks.resize(max_tokens);
    n = (int) toks.size();

    auto cp = llama_context_default_params();
    cp.n_ctx = n + 16;
    cp.n_batch = n;
    cp.n_ubatch = n;
    if (getenv("REF_UBATCH")) cp.n_ubatch = atoi(getenv("REF_UBATCH"));   // other kernels / numerics (noise floor)
    if (getenv("REF_DUMPRAW") && !getenv("REF_DUMP")) { cp.cb_eval = dump_cb; cp.cb_eval_user_data = nullptr; }
    if (const char * d = getenv("REF_DUMP")) {
        std::string s = d;
        size_t p = 0;
        while (p != std::string::npos) { const size_t q = s.find(',', p); g_dump.push_back(s.substr(p, q == std::string::npos ? q : q - p)); p = q == std::string::npos ? q : q + 1; }
        cp.cb_eval = dump_cb;
        cp.cb_eval_user_data = nullptr;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
        if (getenv("REF_LAST") && i < n - atoi(getenv("REF_LAST"))) batch.logits[i] = 0;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    FILE * f = fopen(argv[3], "wb");
    int first = 0;
    if (getenv("REF_LAST")) {   // v2: 'REF2', n, n_vocab, first, tokens, logits of positions first..n-1
        first = std::max(0, n - atoi(getenv("REF_LAST")));
        const int magic = 0x32464552;
        fwrite(&magic, 4, 1, f);
    }
    fwrite(&n, 4, 1, f);
    fwrite(&n_vocab, 4, 1, f);
    if (getenv("REF_LAST")) fwrite(&first, 4, 1, f);
    fwrite(toks.data(), 4, n, f);
    for (int i = first; i < n; ++i) fwrite(llama_get_logits_ith(ctx, i), 4, n_vocab, f);
    fclose(f);
    fprintf(stderr, "wrote %d tokens x %d vocab to %s\n", n, n_vocab, argv[3]);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
