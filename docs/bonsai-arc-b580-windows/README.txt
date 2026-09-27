llama.cpp arc-b580 branch - Windows x64 SYCL build (EXPERIMENTAL: built by CI, not yet tested on Windows)

Needs: an Intel Arc GPU with a recent driver (B580 12 GB for the 27B at 128K context). Nothing else to install:
the Intel runtime files are included.

1. Download the model into this folder:
   https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF  (Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf)
2. Double-click run-bonsai.bat (first start compiles GPU kernels, can take a minute).
3. Open http://localhost:8080

Check the GPU is seen:  sycl-ls.exe   (should list your Arc under Level-Zero)
Smaller context if it runs out of memory: edit run-bonsai.bat, -c 131072 -> -c 65536

Guide and numbers: https://github.com/Torchit1/llama.cpp/blob/arc-b580/docs/bonsai-arc-b580.md
Problems / results: https://github.com/Torchit1/llama.cpp/discussions/1  (please include the last 30 lines of the console)
