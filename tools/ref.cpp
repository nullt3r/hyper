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
#include <map>
#include <sstream>
#include <string>
#include <vector>

// REF_DUMP=regex-ish prefixes (comma separated): print the sum of every matching tensor (debugging layer by layer)
static std::vector<std::string> g_dump;
static bool dump_cb(struct ggml_tensor * t, bool ask, void *) {
    const std::string name = t->name;
    bool match = false;
    for (auto & p : g_dump) if (name.rfind(p, 0) == 0) match = true;
    // REF_DUMPLAST=prefix: the last token's vector (last ne[ndims-1] slice) of every matching tensor to
    // $REF_DUMPDIR/<name>.bin (rewritten per ubatch: the file holds the prompt's last token)
    // (comma-separated prefixes; a name computed several times per graph gets .<occurrence> before .bin)
    static std::map<std::string, int> occ;
    if (ask && name == "hc_init") occ.clear();   // (a new graph)
    bool want = false;
    if (const char * last = getenv("REF_DUMPLAST")) {
        std::string l = last;
        for (size_t p = 0; p <= l.size();) {
            const size_t q = std::min(l.find(',', p), l.size());
            if (q > p && name.rfind(l.substr(p, q - p), 0) == 0) want = true;
            p = q + 1;
        }
    }
    if (want) {
        if (ask) return true;
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return true;
        // REF_DUMPLAST_N: elements per token (a 1-token ubatch drops the token dimension, so it cannot be read off the shape)
        const int nd = ggml_n_dims(t);
        const int64_t per = getenv("REF_DUMPLAST_N") ? atoll(getenv("REF_DUMPLAST_N")) : ggml_nelements(t) / std::max<int64_t>(1, t->ne[nd - 1]);
        if (per <= 0 || ggml_nelements(t) % per) return true;
        std::vector<float> buf(per);
        ggml_backend_tensor_get(t, buf.data(), (size_t) (ggml_nelements(t) - per) * sizeof(float), per * sizeof(float));
        const char * dir = getenv("REF_DUMPDIR") ? getenv("REF_DUMPDIR") : "/tmp";
        const int k = occ[name]++;
        FILE * f = fopen((std::string(dir) + "/" + name + (k ? "." + std::to_string(k) : std::string()) + ".bin").c_str(), "wb");
        if (f) { fwrite(buf.data(), sizeof(float), per, f); fclose(f); }
        return true;
    }
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
    int n = 0;
    if (const char * tf = getenv("REF_TOKENS")) {   // the tokens of an existing reference file (the prompt file is not read)
        FILE * fr = fopen(tf, "rb");
        if (!fr) { fprintf(stderr, "cannot read %s\n", tf); return 1; }
        int hd = 0, nv = 0, fst = 0;
        if (fread(&hd, 4, 1, fr) != 1) return 1;
        if (hd == 0x32464552 && fread(&hd, 4, 1, fr) != 1) return 1;
        if (fread(&nv, 4, 1, fr) != 1) return 1;
        (void) fst;
        long pos = ftell(fr);
        // (v2 files carry 'first' after n_vocab)
        fseek(fr, 0, SEEK_SET);
        int m = 0; if (fread(&m, 4, 1, fr) != 1) return 1;
        fseek(fr, m == 0x32464552 ? 16 : pos, SEEK_SET);
        toks.resize(hd);
        if (fread(toks.data(), 4, hd, fr) != (size_t) hd) { fprintf(stderr, "short token file\n"); return 1; }
        fclose(fr);
        n = hd;
    } else {
        n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, false);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        toks.resize(n);
    }
    if (max_tokens > 0 && (int) toks.size() > max_tokens) toks.resize(max_tokens);
    n = (int) toks.size();

    auto cp = llama_context_default_params();
    cp.n_ctx = n + 16;
    cp.n_batch = n;
    cp.n_ubatch = n;
    if (getenv("REF_UBATCH")) cp.n_ubatch = atoi(getenv("REF_UBATCH"));   // other kernels / numerics (noise floor)
    if ((getenv("REF_DUMPRAW") || getenv("REF_DUMPLAST")) && !getenv("REF_DUMP")) { cp.cb_eval = dump_cb; cp.cb_eval_user_data = nullptr; }
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
