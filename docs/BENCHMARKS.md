# Benchmarks

Performance measurements for CosyVoice3 on CM3588 + AX650N NPU.

## Test Setup

- **Board**: FriendlyElec CM3588 NAS Kit (RK3588 SoC)
- **NPU**: M5Stack Module LLM (AX650N) via M.2 PCIe Gen3 x1
- **Cooling**: JEYI Finscold Q150 passive copper heatsink (no fan)
- **AXCL Runtime**: v3.6.5
- **Models**: CosyVoice-BlankEN-Ax650-C64-P256-CTX512 (w8a16, 24 layers)
- **Voice prompt**: prompt_files_russian_v2 (Russian, 99 speech tokens)
- **Test text**: "Искусственный интеллект открывает новые возможности для человечества."

## Speed

| Metric | 30 steps | 10 steps |
|---|---|---|
| TTFT (time to first token) | 368 ms | ~300 ms |
| Decode speed | 5.72 tok/s | ~10 tok/s |
| Tokens generated | 137 | ~137 |
| LLM decode time | ~24 sec | ~14 sec |
| TTS total (incl. Token2Wav) | 27.2 sec | ~14 sec |
| Post-processing | ~1 sec | ~1 sec |
| Output audio duration | 5.62 sec | ~5.6 sec |
| Real-time factor | 4.84x | ~2.5x |

**Note**: Real-time factor = TTS_time / audio_duration. Lower is better. 1.0x = real-time.

## NPU Utilization (axcl-smi)

Monitoring via `/usr/bin/axcl/axcl-smi` at 0.5s intervals during 30-step generation:

| Phase | Duration | NPU % | CPU % | Temperature | CMM Memory |
|---|---|---|---|---|---|
| Model loading (cold start) | ~10-30 sec | 0% | 2-12% | 63-64°C | 18 → 1790 MiB |
| Prefill | ~0.4 sec | 0% | 3-6% | 64°C | 1790 MiB |
| LLM decode (main) | ~24 sec | 80-97% | 1-5% | 67-71°C | 1790 MiB |
| Token2Wav (Flow + HiFT) | ~3 sec | 96-100% | 1-5% | 70-71°C | 1380-1790 MiB |
| Idle | - | 0% | 0% | 63°C | 18 MiB |

### Resource Usage

| Resource | Value | Capacity | Utilization |
|---|---|---|---|
| CMM memory (NPU) | 1,790 MiB | 7,040 MiB | 25% |
| System memory (NPU) | ~190 MiB | 945 MiB | 20% |
| Temperature (peak) | 71°C | ~85°C throttle | Safe |
| PCIe bandwidth | Gen3 x1 | 8 GT/s | Bottleneck |

## ODE Steps Comparison

| Steps | Speed (tok/s) | Quality | Use Case |
|---|---|---|---|
| 10 | ~10 | Acceptable, some artifacts | Real-time / interactive |
| 20 | ~7-8 | Good balance | General purpose |
| 30 | ~5.7 | Best quality | Final output / production |

## Comparison: NPU vs GPU

Tested with the same text and voice prompt:

| Metric | AX650N NPU | RTX 3090 GPU |
|---|---|---|
| HF artifacts (4-6 kHz) | Present (filtered by LP) | None |
| End truncation | Present (fixed by buffer) | None |
| Voice quality | Good (with post-processing) | Excellent |
| Decode speed | 5.7 tok/s | ~50+ tok/s |
| Power consumption | ~5W (NPU only) | ~350W (whole GPU) |
| Cost | ~$100 (CM3588 + M.2) | ~$1500 (GPU alone) |

## Bottlenecks

1. **PCIe Gen3 x1 bandwidth** — data transfer between CPU and NPU
2. **Autoregressive decoding** — inherently sequential, cannot be parallelized
3. **Cold start** — model loading takes 10-30 seconds per run (no persistent daemon yet)

## Monitoring Command

```bash
# Real-time NPU monitoring
watch -n 0.5 /usr/bin/axcl/axcl-smi
```
