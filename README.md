# CosyVoice3 Russian TTS on AX650N NPU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-AX650N%20NPU-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[Русский](README_RU.md)** | **[中文](README_ZH.md)**

Run CosyVoice3 text-to-speech with **Russian voice cloning** on edge NPU hardware (AXERA AX650N / FriendlyElec CM3588).

This is the first known open-source project running CosyVoice3 with Russian voice cloning on NPU.

## Features

- **Russian voice cloning** — zero-shot voice cloning with proper system prefix for natural prosody
- **NPU inference** — runs entirely on AX650N NPU (no GPU required)
- **Post-processing** — Butterworth LP filter removes NPU HiFT quantization artifacts
- **Buffer phrase** — prevents end-of-speech truncation common on NPU
- **Standalone tokenizer** — no CosyVoice repo dependency, just `transformers`
- **No torchaudio** — WAV I/O via struct (works on minimal aarch64 setups)

## Architecture

```
Text → [Tokenizer HTTP] → [LLM NPU] → [Flow NPU] → [HiFT NPU] → [PostProcess] → WAV
         EOS=1773          24 layers    ODE steps     vocoder      LP 5kHz filter
```

## Hardware Setup

Our setup uses:
- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS Kit (RK3588 SoC)
- **[M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)** — M.2 M-Key NPU accelerator card inserted into CM3588's M.2 slot
- **[JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk)** — passive copper heatsink for M.2 2280 (50 copper fins, 401 W/mK) mounted on the Module LLM for cooling

The JEYI Q150 keeps the NPU at 63°C idle / 71°C peak under full load — well within safe operating range without any active fan.

> Any board with an M.2 M-Key slot and PCIe support can work: CM3588, Raspberry Pi 5, RK3588-based SBCs, x86 PCs with AXCL support.

## Quick Start

### Prerequisites

- **Hardware**: [CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588) + [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) in M.2 slot (or AX650N demo board / M4N-Dock)
- **Cooling**: [JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk) or similar M.2 passive heatsink (recommended)
- **Runtime**: AXCL runtime 3.6+ installed
- **Python**: 3.8+ with `transformers`, `scipy`, `numpy`
- **Models**: Downloaded from [AXERA-TECH/CosyVoice3](https://huggingface.co/AXERA-TECH/CosyVoice3)

### 1. Download models

```bash
# On your host machine
pip install huggingface_hub
huggingface-cli download AXERA-TECH/CosyVoice3 --local-dir ./AXERA-CosyVoice3
```

### 2. Copy to device

```bash
rsync -avz --progress ./AXERA-CosyVoice3/ root@<DEVICE_IP>:/root/AXERA-CosyVoice3/
```

### 3. Install our scripts

```bash
# Copy scripts to device
scp scripts/*.py scripts/*.sh root@<DEVICE_IP>:/root/AXERA-CosyVoice3/

# On the device
ssh root@<DEVICE_IP>
cd /root/AXERA-CosyVoice3
chmod +x run_tts.sh
pip3 install scipy --break-system-packages  # for post-processing
```

### 4. Run

```bash
./run_tts.sh "Привет, как дела? Сегодня хорошая погода." output.wav 30
```

## Usage

```bash
# Basic usage (Russian voice, best quality)
./run_tts.sh "Ваш текст здесь" output.wav 30 prompt_files_russian_v2

# Fast mode (lower quality, ~2x faster)
./run_tts.sh "Ваш текст здесь" output.wav 10

# English with default Chinese voice
./run_tts.sh "Hello, how are you today?" output.wav 30 prompt_files

# Custom voice prompt
./run_tts.sh "Текст" output.wav 30 my_custom_prompt
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| Text | required | Text to synthesize |
| Output | `output.wav` | Output WAV file path |
| Steps | `30` | ODE steps: 10 (fast), 20 (balanced), 30 (best) |
| Prompt | `prompt_files_russian_v2` | Voice prompt directory |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COSYVOICE_DIR` | `/root/AXERA-CosyVoice3` | Base directory on device |
| `TOKENIZER_HOST` | `127.0.0.1` | Tokenizer server host |
| `TOKENIZER_PORT` | `12345` | Tokenizer server port |
| `LP_CUTOFF` | `5000` | Low-pass filter cutoff (Hz) |
| `PAD_MS` | `300` | Silence padding (ms) |

## Russian Voice Cloning

To create a custom Russian voice prompt:

1. **Prepare reference audio** (3-10 seconds, clear speech, single speaker)
2. **Run prompt generation** (requires [frontend-onnx models](https://huggingface.co/AXERA-TECH/CosyVoice3)):

```bash
./generate_prompt.sh reference_audio.wav "Текст произнесённый в аудио" prompt_files_my_voice
```

**Critical**: The prompt text MUST include the system prefix:
```
You are a helpful assistant.<|endofprompt|>Ваш текст промпта здесь.
```
Without this prefix, voice quality degrades significantly.

See [docs/RUSSIAN_VOICE.md](docs/RUSSIAN_VOICE.md) for detailed instructions.

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| High-frequency squeak (4-6 kHz) | NPU HiFT quantization artifacts | Butterworth LP filter 5 kHz (applied automatically) |
| Last word truncated | NPU LLM hits EOS early | Buffer phrase "Вот так." appended (automatic) |
| Notch filters don't help | Artifact is broadband, not tonal | Only full low-pass works |
| `wave.Error: unknown format: 3` | Python wave module doesn't support IEEE float | Custom struct-based WAV I/O |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more details.

## Benchmarks

Tested on CM3588 + AX650N (PCIe Gen3 x1), AXCL 3.6.5:

| Metric | 30 steps | 10 steps |
|---|---|---|
| TTFT (time to first token) | 368 ms | ~300 ms |
| Decode speed | 5.72 tok/s | ~10 tok/s |
| TTS total time (typical phrase) | 27 sec | ~14 sec |
| Output audio duration | 5.6 sec | 5.6 sec |
| Real-time factor | 4.8x | ~2.5x |

### NPU Utilization

| Phase | NPU % | Temperature | CMM Memory |
|---|---|---|---|
| Model loading | 0% | 63°C | 18 → 1790 MiB |
| LLM decode | 80-97% | 67-71°C | 1790 MiB |
| Token2Wav (Flow+HiFT) | 96-100% | 70-71°C | 1790 MiB |
| Idle | 0% | 63°C | 18 MiB |

NPU memory: 1.8 GB / 7 GB (25% utilization). Far from thermal throttling.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for detailed measurements.

## Project Structure

```
cosyvoice3-axera-russian/
├── scripts/
│   ├── run_tts.sh              # Full TTS pipeline
│   ├── postprocess.py          # LP filter + padding (no torchaudio)
│   ├── tokenizer_server.py     # Standalone HTTP tokenizer
│   └── generate_prompt.sh      # Russian voice prompt generation
├── docs/
│   ├── SETUP.md                # Step-by-step installation
│   ├── RUSSIAN_VOICE.md        # Russian voice cloning guide
│   ├── TROUBLESHOOTING.md      # Common issues & fixes
│   └── BENCHMARKS.md           # Performance measurements
├── examples/
│   └── prompt_text_russian.txt # Example prompt with system prefix
├── README.md
├── README_RU.md
└── LICENSE
```

Models and binaries are **not included** — download from:
- **Models + binaries**: [AXERA-TECH/CosyVoice3](https://huggingface.co/AXERA-TECH/CosyVoice3) (~3 GB)
- **Original model**: [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- **NPU runtime source**: [AXERA-TECH/CosyVoice3.Axera](https://github.com/AXERA-TECH/CosyVoice3.Axera)

## Credits

- [FunAudioLLM / Alibaba](https://github.com/FunAudioLLM/CosyVoice) — CosyVoice3 model
- [AXERA-TECH](https://github.com/AXERA-TECH) — NPU quantization, runtime, and pre-built binaries
- [M5Stack](https://shop.m5stack.com/) — Module LLM (AI-8850) M.2 NPU accelerator
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS Kit

## License

MIT License. See [LICENSE](LICENSE).

The CosyVoice3 model is licensed under Apache-2.0 by Alibaba/FunAudioLLM.
The AXERA-TECH NPU runtime and models have their own licenses — see the respective repositories.
