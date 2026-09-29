llama.cpp arc-b580 branch - Windows x64 SYCL build (EXPERIMENTAL: built by CI, not yet tested on Windows)

Needs: an Intel Arc GPU (B580 12 GB for the 27B at 128K context) and a CURRENT Intel graphics driver
(https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html).
Nothing else to install: the Intel runtime files are included. The Level Zero loader (ze_loader.dll) comes
with the driver, so an old driver makes sycl-ls / llama-server crash at start (0xC0000005) or fall back to CPU.
Older iGPUs on Intel's legacy driver (UHD 6xx and earlier) are not supported.

1. Download the model into this folder:
   https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF  (Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf)
2. Double-click run-bonsai.bat (first start compiles GPU kernels, can take a minute).
3. Open http://localhost:8080

Use it from Claude Code (coding agent): see "As a coding agent" in the guide below.

Check the GPU is seen:  sycl-ls.exe   (should list your Arc under Level-Zero)
Smaller context if it runs out of memory: edit run-bonsai.bat, -c 131072 -> -c 65536

Guide and numbers: https://github.com/Torchit1/llama.cpp/blob/arc-b580/docs/bonsai-arc-b580.md
Problems / results: https://github.com/Torchit1/llama.cpp/discussions/1  (please include the last 30 lines of the console)
