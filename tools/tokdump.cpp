// Tokenizer only (the vocabulary of a GGUF, no weights), for offline experiments on request dumps.
// usage: tokdump model.gguf < segments   (segments separated by NUL bytes; special tokens parsed)
// output: one line of space-separated token ids per segment
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf < segments\n", argv[0]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string all((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::vector<llama_token> ids;
    for (size_t a = 0; a <= all.size();) {
        size_t b = all.find('\0', a);
        if (b == std::string::npos) b = all.size();
        const std::string seg = all.substr(a, b - a);
        const int n = -llama_tokenize(vocab, seg.data(), (int) seg.size(), nullptr, 0, false, true);
        ids.resize(std::max(n, 0));
        if (n > 0) llama_tokenize(vocab, seg.data(), (int) seg.size(), ids.data(), n, false, true);
        for (int j = 0; j < n; ++j) printf(j ? " %d" : "%d", ids[j]);
        printf("\n");
        a = b + 1;
    }
    llama_model_free(model);
    return 0;
}
