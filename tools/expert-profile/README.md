# Expert profile

Measures how often each MoE expert is selected, per layer, and writes a profile. It is the input for
expert placement: which experts are worth keeping in VRAM and which can stay in system RAM.

```sh
source /opt/intel/oneapi/setvars.sh
./build-sycl/bin/llama-expert-profile -m Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf -f corpus.txt -n 8192 -v
```

Without `-f` a small built-in corpus is used; `-n` caps the number of tokens (default 4096). The profile is
a flat little-endian file (`EXPR`, version, n_layer, n_expert, n_tokens, then the counts).

Every MoE layer's selected-expert tensor is named `ffn_moe_topk-<il>` by the graph, so the tool hooks the
scheduler's eval callback and reads those tensors back - no changes to the engine are needed. Reading them
forces a synchronisation per layer, so a profiling run is slower than a normal one.

## What it measures

Routing is skewed, and that is what makes per-expert placement worth doing. `Qwopus3.6-35B-A3B` (40 layers,
256 experts, 8 active per token), 8,192 tokens of code:

| VRAM holds | share of expert activations served | CPU work left | layer-level offload at the same budget |
| --- | ---: | ---: | ---: |
| 10% (26 experts) | 61.8% | 38% | 90% |
| 25% (64 experts) | 81.0% | 19% | 75% |
| 50% (128 experts) | 93.6% | 6.4% | 50% |
| 75% (192 experts) | 98.7% | 1.3% | 25% |

For the same VRAM budget, keeping the hot experts of *every* layer on the GPU leaves the CPU 2.4x to 19x
less work than keeping whole layers' experts there. A generic (non-code) corpus is less concentrated but the
same shape: 43.1% / 69.7% / 91.3% / 98.8% on a 396-token run.
