#!/usr/bin/env python3
"""Numerical check of the Arc decode-attention kernels: same binary, same text, same prompt.

run.sh answers "is it faster". This answers "does it still compute the same thing", which is the question
the speed numbers raise and cannot settle.  Everything here is deterministic - perplexity has no sampling
and greedy decoding is an argmax - and the binary never changes, so a difference can only come from the
kernel that was switched off.

Four configurations, so each comparison isolates one thing.  The KV *type* changes perplexity too, so it
is held fixed inside every pair:

    q8_0+fork   fork kernels, q8_0 KV   A
    q8_0+up     upstream,    q8_0 KV   B    A vs B -> the fork's q8_0 decode fast path
    f16+fork    fork kernels, f16 KV    C
    f16+up      upstream,    f16 KV    D    C vs D -> XMX (DPAS) decode attention and the work-group splits
                                            B vs D -> the quantization cost alone

Perplexity is measured with `-b 1` on purpose.  With a large batch llama-perplexity only exercises prefill
attention, which the fork did not touch; a batch of one goes through the decode kernel that did change.
The text is a fixed corpus (sha256 is printed) and the greedy pass uses one fixed prompt, so runs are
comparable across days, cards and builds.

Usage:
    python3 benches/arc-decode-attention/numeric-ab.py [regex]        # regex filters model names

Environment:
    LLAMA_BIN    directory with llama-perplexity/llama-cli  (default ./build-sycl/bin)
    MODELS_DIR   directory with *.gguf                      (default ~/ai-sys/lm-studio-models)
    CORPUS       text file for perplexity                   (default: built from this repo's docs)
    DEV          device to pin                             (default SYCL1; empty = let llama.cpp choose)
    CTX          perplexity context / chunk size            (default 256)
    CHUNKS       chunks to score, CTX*CHUNKS tokens         (default 2)
    GREEDY_N     tokens for the greedy comparison           (default 96)
    OUT          output directory for the json              (default: this directory)
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

BIN = os.environ.get("LLAMA_BIN", os.path.join(REPO, "build-sycl", "bin"))
MODELS = os.path.expanduser(os.environ.get("MODELS_DIR", "~/ai-sys/lm-studio-models"))
OUT = os.path.expanduser(os.environ.get("OUT", HERE))
DEV = os.environ.get("DEV", "SYCL1")
CTX = int(os.environ.get("CTX", "256"))
CHUNKS = int(os.environ.get("CHUNKS", "2"))
GREEDY_N = int(os.environ.get("GREEDY_N", "96"))
PROMPT = "Explain in detail how a KV cache works in a transformer inference engine."

# switching the fork's decode path off is the only difference between the pairs
OFF = {"GGML_SYCL_FA_DEC_OFF": "1"}
CONFIGS = {
    "q8_0+fork": (["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"], {}),
    "q8_0+up":   (["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"], OFF),
    "f16+fork":  (["-fa", "on", "-ctk", "f16", "-ctv", "f16"], {}),
    "f16+up":    (["-fa", "on", "-ctk", "f16", "-ctv", "f16"], OFF),
}


def corpus_path():
    """A fixed text file.  Default: the repo's own docs, so there is nothing to download."""
    p = os.environ.get("CORPUS")
    if p:
        return os.path.expanduser(p)
    p = os.path.join(OUT, "corpus.txt")
    if not os.path.exists(p):
        os.makedirs(OUT, exist_ok=True)
        docs = os.path.join(REPO, "docs")
        text = b""
        for f in sorted(os.listdir(docs)):
            if f.endswith(".md"):
                text += open(os.path.join(docs, f), "rb").read()
        open(p, "wb").write(text[:400000])
    return p


def run(cmd, env_extra, timeout=1800):
    env = dict(os.environ, **env_extra)
    env.pop("ONEAPI_DEVICE_SELECTOR", None)   # -dev does the pinning here
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return -9, "(timeout)"


def dev_args():
    return ["-dev", DEV] if DEV else []


def perplexity(model, args, env_extra, corpus):
    cmd = [f"{BIN}/llama-perplexity", "-m", model, "-f", corpus,
           "-c", str(CTX), "-b", "1", "--chunks", str(CHUNKS),
           "-ngl", "999", "--no-warmup"] + dev_args() + args
    rc, out = run(cmd, env_extra)
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out)
    if not m:
        return None, f"rc={rc} " + " ".join(out.strip().split("\n")[-2:])[:200]
    return float(m.group(1)), ""


def greedy(model, args, env_extra):
    cmd = [f"{BIN}/llama-cli", "-m", model, "-p", PROMPT, "-n", str(GREEDY_N),
           "-no-cnv", "--temp", "0", "--no-display-prompt", "--log-disable",
           "-ngl", "999", "--no-warmup"] + dev_args() + args
    rc, out = run(cmd, env_extra)
    txt = "\n".join(l for l in out.split("\n")
                    if l.strip() and not l.startswith("llama_perf") and "load time" not in l)
    if rc != 0 and not txt:
        return None, f"rc={rc}"
    return hashlib.sha256(txt.encode()).hexdigest()[:12], ""


def main():
    if not os.path.isdir(BIN):
        sys.exit(f"no such LLAMA_BIN: {BIN}")
    corpus = corpus_path()
    chash = hashlib.sha256(open(corpus, "rb").read()).hexdigest()[:16]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    models = sorted(f for f in os.listdir(MODELS)
                    if f.endswith(".gguf") and "mmproj" not in f)
    if only:
        models = [m for m in models if re.search(only, m, re.I)]
    if not models:
        sys.exit(f"no .gguf in {MODELS}")

    print(f"bin {BIN}")
    print(f"corpus {corpus} sha256:{chash}  ctx {CTX} x {CHUNKS} chunks (decode path), greedy {GREEDY_N} tokens")
    print(f"{'model':34} {'q8_0 fork':>10} {'q8_0 up':>9} {'f16 fork':>9} {'f16 up':>8} "
          f"{'d% q8_0':>8} {'d% f16':>7} {'greedy':>7} {'s':>4}")

    outfile = os.path.join(OUT, "numeric-ab-results.json")
    results = json.load(open(outfile)) if os.path.exists(outfile) else {}
    for mf in models:
        name, path = mf[:-5], os.path.join(MODELS, mf)
        row, t0 = {"file": mf, "corpus_sha256": chash, "ctx": CTX, "chunks": CHUNKS}, time.time()
        for cfg, (args, env_extra) in CONFIGS.items():
            v, err = perplexity(path, args, env_extra, corpus)
            row[f"ppl_{cfg}"] = v
            if v is None:
                row["error"] = err
                break
            row[f"greedy_{cfg}"] = greedy(path, args, env_extra)[0]
        row["secs"] = round(time.time() - t0, 1)
        results[name] = row

        a, b = row.get("ppl_q8_0+fork"), row.get("ppl_q8_0+up")
        c, d = row.get("ppl_f16+fork"), row.get("ppl_f16+up")
        dq = f"{(a - b) / b * 100:+.2f}" if a and b else "-"
        df = f"{(c - d) / d * 100:+.2f}" if c and d else "-"
        gh = len({row.get(f"greedy_{k}") for k in CONFIGS if row.get(f"greedy_{k}")})
        f2 = lambda v: f"{v:.4f}" if v else "-"
        print(f"{name[:34]:34} {f2(a):>10} {f2(b):>9} {f2(c):>9} {f2(d):>8} {dq:>8} {df:>7} "
              f"{gh if gh else '-':>7} {row['secs']:>4}", flush=True)
        json.dump(results, open(outfile, "w"), indent=1)

    print(f"\nwrote {outfile}")
    print("d% q8_0 = fork q8_0 fast path vs upstream, same KV type (A vs B)")
    print("d% f16  = XMX decode attention + splits vs upstream, same KV type (C vs D)")
    print("greedy  = distinct outputs across the four configurations; 1 means all four were identical")


if __name__ == "__main__":
    main()
