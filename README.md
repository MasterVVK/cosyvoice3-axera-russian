# CosyVoice3 Russian TTS: RK3588 CPU + AX650N NPU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20%2B%20AX650N-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[Русский](README_RU.md)** | **[中文](README_ZH.md)**

Run CosyVoice3 text-to-speech with **Russian voice cloning** on RK3588 + AX650N NPU edge hardware. High quality audio with **RTF 1.3x** (near real-time).

## Architecture

```
Text → [Tokenizer] → [LLM llama.cpp] → tokens → [Flow DiT] → mel → [HiFT ONNX] → WAV
                        RK3588 CPU                AX650N NPU         RK3588 CPU
                        A76 cores                 PCIe M.2            A76 cores
                        Q4_K_M GGUF               FP16 axmodel        FP32 dynamic
                        ~36 tok/s                  5 ODE steps         ~2.3s vocoder
```

**Why this hybrid?**
- **LLM on CPU** (llama.cpp Q4_K_M): 36 tok/s decode — better quality than NPU W8A8 quantization
- **Flow DiT on AX650N NPU**: ODE solver with 5 steps — perfect for NPU (large tensor matmul)
- **HiFT on CPU** (ONNX Runtime FP32): vocoder has SineGen sin²(x) that breaks under FP16/INT8 quantization

## Key Features

- **Mel crossfade** — 10-frame linear blend at chunk boundaries eliminates artifacts
- **KV-cache persistence** — saves LLM state to disk, 33x faster prefill on repeated speaker
- **Fullmel accumulation** — mel from all flow chunks merged before single HiFT pass (no crossfade seams in audio)
- **Natural EOS** — LLM produces token 6562 naturally, no forced cutoff
- **Tanh soft limiter** — smooth peak limiting without harsh clipping
- **Russian voice cloning** — zero-shot with proper system prefix for natural prosody

## Performance

Tested on CM3588 (RK3588, 32 GB RAM) + M5Stack AI-8850 (AX650N) via PCIe M.2:

| Metric | Cold start | KV-cache hit |
|--------|-----------|--------------|
| Prefill | 2.0s | **0.06s** |
| LLM decode | 3.2s (36 tok/s) | 3.2s |
| Flow (AX650N) | ~3.0s (overlapped) | ~3.0s |
| HiFT (ONNX CPU) | 2.3s | 2.3s |
| **Total** | **~8.3s** | **~6.3s** |
| Audio duration | ~5.5s | ~5.5s |
| **RTF** | **1.5x** | **1.3x** |

### Timing Breakdown (typical 150 tokens)

```
A76 CPU:  ████ LLM prefill ████████ LLM decode ████████████████ HiFT ONNX ████
AX650N:   ░░░░░░░░░░░░░░░░ Flow chunk 1 ░░ Flow 2 ░░ Flow 3 ░░ Final ░░░░░░░
A55 CPU:  coordination ────────────────────────────────────────────────────────
```

## Hardware

- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS Kit — RK3588 SoC (4x A76 + 4x A55), 32 GB RAM
- **[M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)** — M.2 M-Key NPU (AX650N, 24 TOPS INT8, 8 GB)
- **[JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk)** — passive copper heatsink for M.2 2280

> Any board with M.2 M-Key + PCIe and an AX650N-based accelerator can work.

## Quick Start

### Prerequisites

- **AXCL runtime** 3.6+ on CM3588
- **llama.cpp** built for aarch64 with shared libs
- **Python** 3.8+ with `numpy`, `onnxruntime`, `transformers`
- **Models**: AX650N axmodels from [AXERA-TECH/CosyVoice3](https://huggingface.co/AXERA-TECH/CosyVoice3), GGUF LLM, ONNX HiFT

### Install

```bash
# On device
cd /root/cosyvoice3-build

# Build C++ binary
cd cosyvoice3-axera/cpp
mkdir -p build && cd build
cmake .. && make -j4
cp cosyvoice3_tts install/bin/

# Build llama wrapper
cd /root/cosyvoice3-build/dual_npu
gcc -shared -fPIC -O2 -o llama_wrapper.so llama_wrapper.c \
    -I/path/to/llama.cpp/include -L/path/to/llama.cpp/lib \
    -lllama -lggml -lggml-base -lggml-cpu
```

### Run

```bash
cd /root/cosyvoice3-build/dual_npu

# Best quality: ONNX HiFT (fullmel mode)
ONNX_HIFT=1 bash launch_dual_npu.sh "Привет, как дела? Сегодня хорошая погода."

# Fast mode: NPU HiFT (lower quality, RTF ~1.2x)
bash launch_dual_npu.sh "Привет, как дела?"

# Custom temperature
TEMPERATURE=0.8 ONNX_HIFT=1 bash launch_dual_npu.sh "Ваш текст здесь"
```

Output: `output.wav` (24 kHz, float32)

## Project Structure

```
cosyvoice3-axera-russian/
├── cpp/                          # C++ pipeline (RK3588 + AX650N)
│   ├── CMakeLists.txt
│   └── src/
│       ├── main.cpp              # Main: token reader + Token2Wav thread
│       └── runner/
│           ├── Token2wav.hpp     # Flow (NPU) + HiFT + mel accumulation
│           ├── LLM.hpp           # LLM runner (unused in dual_npu mode)
│           ├── CV2_Tokenizer.hpp # Tokenizer wrapper
│           └── utils/
│               └── sampling.hpp  # Token sampling utilities
├── dual_npu/                     # Python servers + launch scripts
│   ├── launch_dual_npu.sh        # Main launch script
│   ├── llamacpp_token_server.py  # LLM token server (llama.cpp backend)
│   ├── onnx_hift_server.py       # ONNX HiFT vocoder server
│   ├── llama_wrapper.c           # C wrapper for llama.cpp
│   ├── llama_cpp_bindings.py     # Python ctypes bindings
│   ├── tts_client.py             # Test client
│   └── rkllm_token_server.py     # Legacy RKLLM backend
├── scripts/                      # Original all-NPU pipeline scripts
├── docs/                         # Documentation
├── examples/                     # Example prompts
├── README.md
├── README_RU.md
├── README_ZH.md
└── LICENSE
```

## Technical Details

### Mel Crossfade (key quality fix)

Flow encoder runs produce mel for overlapping token windows. Without crossfade, hard concatenation creates audible clicks and rustling at chunk boundaries.

Solution: 10-frame linear blend at each boundary in `Token2wav.hpp`:
```
Chunk N:   [...existing mel...████████████]
Chunk N+1:                    [██blend██|fresh mel frames...]
                               ↑ 10-frame crossfade zone
```

This reduced vocoder artifacts by 3x (despike count: 2779 → 998).

### Fullmel Exact Start Fix

The flow estimator NPU models have fixed output sizes:
- `flow_estimator_200`: 200 - 150 prompt = **50** generated mel frames
- `flow_estimator_250`: 250 - 150 = **100** frames
- `flow_estimator_300`: 300 - 150 = **150** frames

The original formula `exact_start = token_offset * 2` assumed 78 tokens = 156 mel frames, but flow_estimator_300 only produces 150. After chunk 3, mel accumulator stopped growing entirely.

Fix: use sliding window formula matching the non-fullmel mode:
```cpp
exact_start = min(min(offset/hop, max_chunk-1) * hop * ratio, mel_frames);
// Capped at 100 instead of growing linearly to 150+
```

### KV-Cache Persistence

After first prefill, LLM KV-state (72 MB) is saved to disk. Subsequent requests with the same speaker prompt skip prefill entirely:
- Cold: 2.0s prefill
- Cached: 0.06s load (33x speedup)

### CPU Core Affinity (Critical)

RK3588 has heterogeneous cores. Without pinning, 5x slowdown due to contention:
- **LLM + ONNX HiFT**: `taskset -c 4-7` (A76 big cores, 2.35 GHz)
- **Token2Wav (C++)**: `taskset -c 0-3` (A55 little cores, 1.8 GHz)
- **OPENBLAS_NUM_THREADS=1** to prevent numpy stealing A76 cores

## Failed Acceleration Attempts

We extensively tested alternatives to CPU ONNX for HiFT:

| Approach | Result | Why |
|----------|--------|-----|
| RKNN NPU (RK3588) | All-zeros or corr -0.26 | SineGen sin²(x) breaks under FP16 |
| AX650N NPU | SNR 13.7 dB (noisy) | Same quantization sensitivity |
| INT8 ONNX | 5.5x slower, SNR -1 dB | SnakeAlpha activation kills INT8 |
| MNN Mali GPU | Shape broadcast error | SineGen 96000 vs 95520 mismatch |

**Conclusion**: HiFT vocoder requires FP32 precision. CPU ONNX is currently the only viable option.

## Models Required

| Model | Format | Size | Source |
|-------|--------|------|--------|
| LLM (Qwen2-0.5B) | GGUF Q4_K_M | 469 MB | Convert from HuggingFace |
| Flow encoder (28/53/78/50_final) | axmodel | ~50 MB each | [AXERA-TECH/CosyVoice3](https://huggingface.co/AXERA-TECH/CosyVoice3) |
| Flow estimator (200/250/300) | axmodel | ~100 MB each | Same |
| HiFT vocoder | ONNX FP32 | 88.5 MB | Export from CosyVoice3 |
| Flow embeddings | binary | ~50 MB | Same |
| Speaker prompt | npy/bin | ~2 MB | Generate from reference audio |

## Credits

- [FunAudioLLM / Alibaba](https://github.com/FunAudioLLM/CosyVoice) — CosyVoice3 model
- [AXERA-TECH](https://github.com/AXERA-TECH) — NPU quantization, AXCL runtime, axmodel conversion
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — efficient LLM inference on CPU
- [M5Stack](https://shop.m5stack.com/) — Module LLM (AI-8850) M.2 NPU accelerator
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS Kit

## License

MIT License. See [LICENSE](LICENSE).

CosyVoice3 model: Apache-2.0 (Alibaba/FunAudioLLM).
AXERA-TECH runtime and models: see respective repositories.
