# Decode attention on Intel Arc (SYCL)

`run.sh` measures one model twice with llama-bench: once on the generic vector attention path
(`GGML_SYCL_FA_DEC_OFF=1`) and once on the default fast q4_0 decode-attention path. It prints `tg128`
tokens/s for each context depth, so the before/after on your card is visible directly.

```sh
source /opt/intel/oneapi/setvars.sh
LLAMA_BENCH=./build-sycl/bin/llama-bench benches/arc-decode-attention/run.sh Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf
```

The fast path needs a **q4_0 KV cache** (`-ctk q4_0 -ctv q4_0`, the script's default) and flash attention
on. Models whose attention shape is not served by the kernel - head size other than 256 (512 for Gemma 4's
global layers), or a GQA ratio other than 4:1, 6:1, 8:1 at head 256 - fall back to the generic path and show
no difference; that is expected, not a bug.

Reference, Arc Pro B60 24 GB, Linux, oneAPI 2025.3, Level Zero 26.35, `tg128`, 2 repetitions:

| model | 0 | 8K | 32K | 128K |
| --- | ---: | ---: | ---: | ---: |
| Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M (MoE, 8:1) before | 91.3 | 77.2 | 53.7 | 24.6 |
| ... after | 92.4 | 89.8 | 81.2 | 59.8 |
| Ornith-1.5-9B Q4_K_M (dense, 4:1) before | 63.7 | 54.0 | 37.1 | 16.5 |
| ... after | 64.1 | 62.2 | 55.9 | 42.2 |

Results from other Arc cards (B580, B70, Lunar Lake, ...) are the most useful thing: post card, driver
version, model and quant, the depths and what you got in the repository's Discussions.

## Numerical check

`numeric-ab.py` answers the other half of the question: not "is it faster" but "does it still compute the
same thing". One binary, one fixed text corpus, one fixed prompt, and only the kernel switched off between
runs - so a difference can only come from the kernel. It prints perplexity and a greedy output hash for four
configurations, so each pair isolates one thing with the KV type held fixed:

| pair | isolates |
| --- | --- |
| `q8_0+fork` vs `q8_0+up` | the q8_0 decode fast path, same q8_0 KV |
| `f16+fork` vs `f16+up` | XMX (DPAS) decode attention and the work-group splits, same f16 KV |

Perplexity runs with `-b 1` on purpose: with a large batch llama-perplexity only exercises prefill
attention, which this work did not touch, while a batch of one goes through the decode kernel that did
change. Greedy decoding is an argmax, so both numbers are deterministic and comparable across days.

```sh
LLAMA_BIN=./build-sycl/bin MODELS_DIR=~/ai-sys/lm-studio-models \
    python3 benches/arc-decode-attention/numeric-ab.py            # optional regex filters the models
```

Over 35 models on the Arc Pro B60: `ΔPPL` is **0.0000%** for the XMX decode attention and the splits on
every model, **−0.68% … +1.70%** (mean 0.19%) for the q8_0 fast path, and the greedy output is **identical
in all four configurations on every model**. The raw 36-row output is in `numeric-ab-results.json`.
