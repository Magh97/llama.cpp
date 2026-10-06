// tools/expert-profile/expert-profile.cpp
//
// Measures how often each MoE expert is selected, per layer, and writes a profile.
//
// Every MoE layer's selected-expert tensor is named "ffn_moe_topk-<il>" by
// llama_context::graph_get_cb, so this tool only hooks the scheduler's eval
// callback, reads those tensors back and counts. The profile is the input for
// expert placement: which experts are worth keeping in VRAM and which can live
// in system RAM.
//
// Usage: llama-expert-profile -m model.gguf [-f corpus.txt] [-n tokens] [-o profile.bin] [-v]
//
// Without -f a small built-in corpus of code and prose is used. The profile is
// a flat little-endian file: "EXPR", version, n_layer, n_expert, n_tokens,
// then n_layer * n_expert uint64 activation counts.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

struct expert_profile {
    std::vector<std::vector<uint64_t>> counts;  // [layer][expert]
    uint64_t tokens = 0;
    uint64_t reads  = 0;
};

static const char * k_prefix = "ffn_moe_topk-";

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    expert_profile * p = (expert_profile *) user_data;

    if (ask) {
        // the selected-expert ids of every MoE layer, as named by the graph
        return t->type == GGML_TYPE_I32 && t->ne[0] > 0 &&
               strncmp(t->name, k_prefix, strlen(k_prefix)) == 0;
    }

    const int il = atoi(t->name + strlen(k_prefix));
    if (il < 0 || il >= (int) p->counts.size()) {
        return true;
    }

    // the ids tensor is a view of the argsort result: rows of n_expert_used int32 with a wider stride,
    // so read it row by row (ggml_nbytes would span the padding between rows)
    const int64_t k     = t->ne[0];
    const int64_t nrows = ggml_nrows(t);
    std::vector<int32_t> row(k);
    std::vector<uint64_t> & counts = p->counts[il];
    for (int64_t r = 0; r < nrows; r++) {
        ggml_backend_tensor_get(t, row.data(), (size_t) r * t->nb[1], (size_t) k * sizeof(int32_t));
        for (const int32_t id : row) {
            if (id < 0) {
                continue;
            }
            if (id >= (int32_t) counts.size()) {
                counts.resize(id + 1, 0);
            }
            counts[id]++;
        }
    }
    p->reads++;
    return true;
}

static const char * k_default_corpus =
    "Write a Python function that counts the words in a file.\n\n"
    "```python\n"
    "def count_words(path):\n"
    "    with open(path, 'r', encoding='utf-8') as f:\n"
    "        return len(f.read().split())\n"
    "```\n\n"
    "The function opens the file, reads it, splits on whitespace and returns the number of tokens. "
    "It does not handle binary files, missing files or very large files; those need separate checks.\n\n"
    "A hash map maps keys to values and gives average constant time lookup. In C++ that is "
    "std::unordered_map, in Rust HashMap, in Python dict. The worst case is linear when every key "
    "collides, which is why the load factor matters.\n\n"
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. The five boxing wizards jump quickly.\n\n"
    "function merge(a, b) {\n"
    "  const out = [];\n"
    "  let i = 0, j = 0;\n"
    "  while (i < a.length && j < b.length) out.push(a[i] <= b[j] ? a[i++] : b[j++]);\n"
    "  return out.concat(a.slice(i), b.slice(j));\n"
    "}\n\n"
    "A compiler translates source code into another representation, usually machine code or bytecode. "
    "It parses the input into a syntax tree, checks types, optimises, and emits the result. "
    "Interpreters skip the last step and execute the tree directly, which is slower but simpler.\n\n"
    "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2026-01-01' "
    "GROUP BY customer_id HAVING SUM(amount) > 100 ORDER BY total DESC LIMIT 20;\n\n"
    "The lighthouse keeper wrote in his log that the wind had turned north during the night, "
    "and that the sea beyond the reef was running higher than he had seen it all winter.\n";

int main(int argc, char ** argv) {
    std::string model_path;
    std::string corpus_path;
    std::string out_path = "expert-profile.bin";
    int n_tokens_max = 4096;
    int n_batch      = 512;
    int n_ctx        = 4096;
    bool verbose     = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "-m" || a == "--model") {
            const char * v = next(); if (!v) { return 1; } model_path = v;
        } else if (a == "-f" || a == "--file") {
            const char * v = next(); if (!v) { return 1; } corpus_path = v;
        } else if (a == "-n" || a == "--tokens") {
            const char * v = next(); if (!v) { return 1; } n_tokens_max = atoi(v);
        } else if (a == "-o" || a == "--out") {
            const char * v = next(); if (!v) { return 1; } out_path = v;
        } else if (a == "-b" || a == "--batch") {
            const char * v = next(); if (!v) { return 1; } n_batch = atoi(v);
        } else if (a == "-c" || a == "--ctx") {
            const char * v = next(); if (!v) { return 1; } n_ctx = atoi(v);
        } else if (a == "-v" || a == "--verbose") {
            verbose = true;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr,
                "usage: %s -m model.gguf [-f corpus.txt] [-n tokens] [-o profile.bin] [-b batch] [-c ctx] [-v]\n",
                argv[0]);
        return 1;
    }

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }

    expert_profile prof;
    prof.counts.resize(llama_model_n_layer(model) + llama_model_n_layer_nextn(model));

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx              = n_ctx;
    cparams.n_batch            = n_batch;
    cparams.cb_eval            = eval_cb;
    cparams.cb_eval_user_data  = &prof;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        llama_model_free(model);
        return 1;
    }

    std::string text;
    if (corpus_path.empty()) {
        text = k_default_corpus;
    } else {
        std::ifstream f(corpus_path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "cannot open %s\n", corpus_path.c_str());
            return 1;
        }
        text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 8);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), true, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), true, false);
    }
    if (n < 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    tokens.resize(n);
    if (n_tokens_max > 0 && n > n_tokens_max) {
        tokens.resize(n_tokens_max);
        n = n_tokens_max;
    }

    const int window = std::max(n_batch, n_ctx - n_batch);
    for (int i = 0; i < n; ) {
        if (i > 0 && i % window == 0) {
            llama_memory_clear(llama_get_memory(ctx), true);  // window the corpus, routing is per token
        }
        const int chunk = std::min(n_batch, n - i);
        struct llama_batch batch = llama_batch_get_one(tokens.data() + i, chunk);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at token %d\n", i);
            return 1;
        }
        prof.tokens += chunk;
        i += chunk;
    }

    // ---- summary ----------------------------------------------------------
    int n_layer = 0;
    for (size_t l = 0; l < prof.counts.size(); l++) {
        if (!prof.counts[l].empty()) {
            n_layer = (int) l + 1;
        }
    }
    size_t n_expert = 0;
    for (const auto & row : prof.counts) {
        n_expert = std::max(n_expert, row.size());
    }

    if (n_layer == 0 || n_expert == 0) {
        printf("no MoE routing found in %s (no ffn_moe_topk-* tensors were read)\n", model_path.c_str());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    printf("model:  %s\n", model_path.c_str());
    printf("layers: %d with routing, experts: %zu, tokens: %llu, tensor reads: %llu\n",
           n_layer, n_expert, (unsigned long long) prof.tokens, (unsigned long long) prof.reads);
    printf("\nVRAM budget -> share of expert activations served (mean over layers):\n");

    double total_acts = 0.0;
    for (int l = 0; l < n_layer; l++) {
        for (const uint64_t c : prof.counts[l]) {
            total_acts += (double) c;
        }
    }

    for (const double frac : { 0.10, 0.25, 0.50, 0.75 }) {
        const size_t k = std::max<size_t>(1, (size_t) (frac * (double) n_expert + 0.5));
        double cov = 0.0, acts = 0.0;
        for (int l = 0; l < n_layer; l++) {
            std::vector<uint64_t> row = prof.counts[l];
            if (row.empty()) {
                continue;
            }
            std::sort(row.begin(), row.end(), std::greater<uint64_t>());
            const uint64_t tot = [&] { uint64_t s = 0; for (const uint64_t c : row) s += c; return s; }();
            uint64_t top = 0;
            for (size_t e = 0; e < k && e < row.size(); e++) {
                top += row[e];
            }
            if (tot > 0) {
                cov  += (double) top / (double) tot * (double) tot;
                acts += (double) tot;
            }
        }
        printf("  %3.0f%% (%4zu experts) -> %5.1f%%\n", frac * 100.0, k, acts > 0 ? 100.0 * cov / acts : 0.0);
    }

    if (verbose) {
        printf("\nper layer (top 8 experts, share of the layer's activations):\n");
        for (int l = 0; l < n_layer; l++) {
            const std::vector<uint64_t> & row = prof.counts[l];
            if (row.empty()) {
                continue;
            }
            std::vector<std::pair<uint64_t, int>> top;
            uint64_t tot = 0;
            for (size_t e = 0; e < row.size(); e++) {
                tot += row[e];
                top.push_back({ row[e], (int) e });
            }
            std::sort(top.begin(), top.end(), std::greater<std::pair<uint64_t,int>>());
            printf("  layer %2d (%8llu activations):", l, (unsigned long long) tot);
            for (int i = 0; i < 8 && i < (int) top.size(); i++) {
                printf(" %d:%.1f%%", top[i].second, tot > 0 ? 100.0 * (double) top[i].first / (double) tot : 0.0);
            }
            printf("\n");
        }
    }

    // ---- profile file -----------------------------------------------------
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }
    const uint32_t version = 1;
    const uint32_t nl = (uint32_t) n_layer;
    const uint32_t ne = (uint32_t) n_expert;
    out.write("EXPR", 4);
    out.write((const char *) &version, 4);
    out.write((const char *) &nl, 4);
    out.write((const char *) &ne, 4);
    out.write((const char *) &prof.tokens, 8);
    for (int l = 0; l < n_layer; l++) {
        for (size_t e = 0; e < n_expert; e++) {
            const uint64_t c = (l < (int) prof.counts.size() && e < prof.counts[l].size()) ? prof.counts[l][e] : 0;
            out.write((const char *) &c, 8);
        }
    }
    printf("\nwrote %s (%d layers x %zu experts)\n", out_path.c_str(), n_layer, n_expert);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
