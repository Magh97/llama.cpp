# Bonsai 2 27B (PTQ1_0 ternary) on Intel Arc B580

https://github.com/user-attachments/assets/b6df0d56-1492-4d38-8f28-8fd323ffb92b

*Real time in the llama.cpp web UI on a B580: a new ~1000-token answer at ~65 t/s, then a whole-file edit at ~205 t/s (temperature 0, 128K context).*

This branch makes PrismML's ternary Bonsai 2 27B run fast on a 12 GB Intel Arc B580 (Xe2, "Battlemage") with the SYCL
backend, at the full 128K context. The Vulkan backend got most of the same kernel work, but SYCL is clearly faster on
this card.

## Results

B580 12 GB, Linux, oneAPI 2025.3, Level Zero driver 26.35. Everything below is at 131072 context with a q4_0 KV cache,
speculative decoding on (MTP head, 3 drafts, plus n-gram drafts), thinking off.

|                                   | SYCL (this branch) | Vulkan (this branch) |
|-----------------------------------|--------------------|----------------------|
| Fresh code answer                 | 83 t/s             | 53 t/s               |
| Rename a symbol in pasted code    | 361 t/s            | 194 t/s              |
| Small edit to pasted code         | 250 t/s            | 135 t/s              |
| Plain generation, no speculation  | 41 t/s             | 32 t/s               |
| 29K-token document: prompt / gen  | 786 / 40 t/s       | 281 / 29 t/s         |

These are greedy (temperature 0) numbers on short benchmark prompts. Speculation speed depends on the text and the
sampling: in the chat UI at temperature 0 a ~1000-token new answer ran at about 65 t/s and returning a whole edited file
at about 205 t/s. With sampling (temperature 0.6) fewer drafts are accepted: roughly 60-65 t/s for new code and 90-100
t/s for edits. New prose drafts worse than code.

Against the first working SYCL port of this model (same card, same settings, 32K context): fresh code 55.8 -> 85.5 t/s,
rename 217.5 -> 368.8, edit 143.0 -> 255.1, plain generation 31.4 -> 40.7. Quality: KL divergence against the reference logits is 0.00022
(99.2% same top token; the plain PTQ1_0 kernels score 0.0003), and greedy outputs on our test prompts are byte-identical to the plain PTQ1_0 kernels.

## Model

The speculative numbers need a PTQ1_0 GGUF that includes the MTP head. PrismML's own
[Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) PTQ1_0 file has no MTP head
(use `--spec-type ngram-mod` with it). A public build with the head grafted on is
[sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF),
file `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` (6.3 GB).

The table above was measured on my own derivative of Bonsai 2 27B with the same MTP head. With the public mtp-lean file
on the same card, build and settings (128K): fresh code 77 t/s, rename 306 t/s, edit 205 t/s, plain generation 38 t/s.
Different weights produce different text, so speculation lands a little less often.

## What changed

- **Ternary weights on the XMX matrix units.** At load, every PTQ1_0 weight is repacked in place to a 2-bit layout and
  multiplied with the int8 x int2 DPAS kernels from [libxsmm/TernSYCL](https://github.com/libxsmm/TernSYCL) (BSD 3-Clause,
  vendored in `ggml/src/ggml-sycl/ternsycl`). The base-3 PTQ1_0 packing has to be decoded on the ALUs first, which made
  the multi-token verify step of speculative decoding compute-bound. The 2-bit layout costs about 31% more weight memory.
  Activations are quantized to int8 with round-to-nearest (TernSYCL truncates; rounding to nearest cut KLD 5x).
- **Decode attention for a q4_0 KV cache** serving 1-4 query tokens per launch from one read of the cache (GQA-aware).
- **Gated delta-net** blocked kernel, fused state writes, and several fusions for single-token decode (same-input mat-vecs
  in one launch, narrow concat, fused gate/up).
- **Memory for 128K on 12 GB:** the MTP draft context now takes its own KV cache type (`-ctkd/-ctvd q4_0`) and a smaller
  physical batch (`LLAMA_ARG_SPEC_DRAFT_UBATCH=512`), and the main prompt batch is 1024.

## Build (SYCL)

Needs the Intel oneAPI Base Toolkit 2025.3 or newer and a recent Level Zero GPU driver.

```sh
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_TARGET=INTEL
cmake --build build-sycl -j --target llama-server llama-bench llama-cli
```

## Run

```sh
source /opt/intel/oneapi/setvars.sh
export GGML_SYCL_PTQ1_T2=all              # PTQ1_0 weights on XMX (ffn = feed-forward only, unset = off)
export LLAMA_ARG_SPEC_DRAFT_UBATCH=512    # smaller compute buffer for the MTP draft context
./build-sycl/bin/llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf -ngl 99 \
  -c 131072 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -np 1 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-ngram-mod-n-max 256 \
  -ub 1024 -b 2048 --chat-template-kwargs '{"enable_thinking":false}' --host 0.0.0.0 --port 8080
```

With a shorter context you can use `-ub 2048` for faster prompt reading. `GGML_SYCL_PTQ1_T2=ffn` puts only the
feed-forward weights on XMX: about 380 MiB less weight memory, enough for `-ub 2048` at 128K, and most of the speed
(fresh 80, rename 314, edit 213).

The first long prompt after the very first start compiles the XMX kernels (about 30 s); the GPU driver caches them after
that. If the server ever hangs during start-up in GPU initialisation after being killed mid-compile, move
`~/.cache/neo_compiler_cache` aside.

## Build and run (Vulkan)

SYCL is faster on the B580 (see the table), but the Vulkan backend has most of the same kernel work and runs on any
Vulkan driver. Tested on Mesa ANV 26.2. Needs the Vulkan SDK (glslc and headers).

```sh
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j --target llama-server llama-bench llama-cli
./build-vk/bin/llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf -ngl 99 \
  -c 131072 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -np 1 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-ngram-mod-n-max 256 \
  -ub 2048 -b 2048 --chat-template-kwargs '{"enable_thinking":false}' --host 0.0.0.0 --port 8080
```

At 128K this left about 1.2 GB of VRAM free on the B580. The `GGML_SYCL_*` switches below do not apply to Vulkan; its
new paths can be turned off with `GGML_VK_PTQ1_MC_OFF=1` (multi-column PTQ1_0 mat-vec).

## Switches (SYCL)

All optimisations are on by default except the XMX path. Set any of these to turn a piece off for comparison:
`GGML_SYCL_PTQ1_T2_GEMM_OFF`, `GGML_SYCL_PTQ1_MULTI=0`, `GGML_SYCL_PTQ1_MULTI_NCOLS=0`, `GGML_SYCL_PTQ1_GLU1=0`,
`GGML_SYCL_PTQ1_PAIRS=0`, `GGML_SYCL_PTQ1_NCOLS_DEC_OFF`, `GGML_SYCL_FA_DEC_OFF`, `GGML_SYCL_GDN_BLOCKED_OFF`,
`GGML_SYCL_GLU_FUSE_OFF`.

## Feedback and your numbers

If you run this on a B580 or another Arc card, please post your results (card, driver, context, the numbers you get) in
this repository's Discussions, and report problems as issues. Results from other setups are the most useful thing
right now.

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), PrismML's PTQ1_0 support, and the `bonsai-combo`
branch of [professorpalmer/llama.cpp-ada-ternary](https://github.com/professorpalmer/llama.cpp-ada-ternary) (PrismML
PR #221) that this branch started from, with the ternary DPAS
kernels from [libxsmm/TernSYCL](https://github.com/libxsmm/TernSYCL) (BSD 3-Clause; licence and notice included).
