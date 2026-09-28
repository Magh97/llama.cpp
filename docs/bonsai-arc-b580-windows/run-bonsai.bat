@echo off
rem Bonsai 2 27B on an Intel Arc B580 (12 GB): 128K context, MTP + n-gram drafts. Settings match docs/bonsai-arc-b580.md.
rem Put Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf in this folder (or change MODEL), run this, open http://localhost:8080
set MODEL=Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf
rem ternary weights on the XMX units (needs an Xe2 GPU: Arc B-series, Lunar Lake; switches itself off elsewhere)
set GGML_SYCL_PTQ1_T2=all
rem smaller compute buffer for the MTP draft context
set LLAMA_ARG_SPEC_DRAFT_UBATCH=512
rem long prompts use the chunked attention path above this many tokens (the fast one runs out of VRAM near 119K)
set GGML_SYCL_FA_ONEDNN_MAX_KV=98304
rem decode / verify attention on XMX straight from the q4_0 cache (Xe2). Set to 0 if it errors on your driver
set GGML_SYCL_FA_DEC_DPAS=1
rem full 1024-token prompt batches via oneDNN's int8 GEMM (~10% faster long prompts)
set GGML_SYCL_T2_W8A8_MIN=1024
rem bonsai-chat-template.jinja = the model's template, but accepts the mid-conversation system notes Claude Code sends
rem less VRAM or a smaller card: lower -c (e.g. 32768)
llama-server.exe -m "%MODEL%" -ngl 99 -c 131072 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -np 1 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-ngram-mod-n-max 256 -ub 1024 -b 2048 --chat-template-kwargs "{\"enable_thinking\":false}" --host 127.0.0.1 --port 8080 --jinja --chat-template-file "%~dp0bonsai-chat-template.jinja"
pause
