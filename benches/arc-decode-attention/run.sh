#!/usr/bin/env bash
# Decode attention before/after on Intel Arc (SYCL).
#
# "before" runs the generic vector attention path (GGML_SYCL_FA_DEC_OFF=1),
# "after" runs the default fast q4_0 decode-attention path. Same model, quant,
# KV cache type and depths, so the two lines differ only by the kernel.
#
# Usage:
#   benches/arc-decode-attention/run.sh <model.gguf> [more.gguf ...]
#
# Environment:
#   LLAMA_BENCH  path to llama-bench        (default ./build-sycl/bin/llama-bench)
#   DEPTHS       context depths             (default 0,8192,32768,131072)
#   KV           KV cache type              (default q4_0; the fast path needs q4_0)
#   NGL          layers on the GPU          (default 99)
#   REPS         repetitions per test       (default 2)
#   EXTRA        extra llama-bench options  (default empty)
set -euo pipefail

BIN="${LLAMA_BENCH:-./build-sycl/bin/llama-bench}"
DEPTHS="${DEPTHS:-0,8192,32768,131072}"
KV="${KV:-q4_0}"
NGL="${NGL:-99}"
REPS="${REPS:-2}"
EXTRA="${EXTRA:-}"

if [ $# -lt 1 ]; then
    sed -n '2,18p' "$0"
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "llama-bench not found at '$BIN' (set LLAMA_BENCH)" >&2
    exit 1
fi

# one llama-bench run -> "t/s at depth 1  t/s at depth 2  ..."
run() {
    local env="$1" model="$2"
    env $env "$BIN" -m "$model" -p 512 -n 128 -d "$DEPTHS" -r "$REPS" \
        -ctk "$KV" -ctv "$KV" -fa on -ngl "$NGL" $EXTRA -o md 2>/dev/null \
        | grep -oE 'tg[0-9]+.*\| *[0-9.]+' | grep -oE '[0-9.]+$' | paste -sd' ' -
}

echo "depths (tokens): $DEPTHS"
echo "kv: $KV, ngl: $NGL, reps: $REPS"
for m in "$@"; do
    if [ ! -f "$m" ]; then
        echo "no such model: $m" >&2
        exit 1
    fi
    echo
    echo "== $(basename "$m")"
    echo "   before: $(run 'GGML_SYCL_FA_DEC_OFF=1' "$m")"
    echo "   after : $(run '' "$m")"
done
