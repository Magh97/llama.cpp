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
