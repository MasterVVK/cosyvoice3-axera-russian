# CosyVoice3 俄语TTS：RK3588 CPU + AX650N NPU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20%2B%20AX650N-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[English](README.md)** | **[Русский](README_RU.md)**

在RK3588 + AX650N NPU边缘硬件上运行CosyVoice3语音合成，支持**俄语语音克隆**。高质量音频，**RTF 1.3x**（接近实时）。

## 架构

```
文本 → [分词器] → [LLM llama.cpp] → tokens → [Flow DiT] → mel → [HiFT ONNX] → WAV
                   RK3588 CPU                  AX650N NPU         RK3588 CPU
                   A76核心                      PCIe M.2           A76核心
                   Q4_K_M GGUF                 FP16 axmodel        FP32动态
                   ~36 tok/s                    5步ODE              ~2.3s声码器
```

**为什么混合架构？**
- **LLM在CPU上**（llama.cpp Q4_K_M）：36 tok/s — 比NPU W8A8量化质量更好
- **Flow DiT在AX650N NPU上**：5步ODE求解器 — NPU擅长大规模张量运算
- **HiFT在CPU上**（ONNX Runtime FP32）：声码器的SineGen sin²(x)在FP16/INT8下会失真

## 主要特性

- **Mel交叉淡入** — 块边界处10帧线性混合消除伪影
- **KV缓存持久化** — 将LLM状态保存到磁盘，重复说话人预填充加速33倍
- **全Mel累积** — 所有flow块的mel合并后单次HiFT处理
- **自然EOS** — LLM自然生成6562结束标记
- **Tanh软限幅** — 平滑峰值限制
- **俄语语音克隆** — 零样本，带系统前缀实现自然韵律

## 性能

CM3588（RK3588，32 GB RAM）+ M5Stack AI-8850（AX650N），PCIe M.2：

| 指标 | 冷启动 | KV缓存 |
|------|--------|--------|
| 预填充 | 2.0s | **0.06s** |
| LLM解码 | 3.2s（36 tok/s） | 3.2s |
| Flow（AX650N） | ~3.0s（并行） | ~3.0s |
| HiFT（ONNX CPU） | 2.3s | 2.3s |
| **总计** | **~8.3s** | **~6.3s** |
| 音频时长 | ~5.5s | ~5.5s |
| **RTF** | **1.5x** | **1.3x** |

## 硬件

- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS套件 — RK3588（4x A76 + 4x A55），32 GB RAM
- **[M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)** — M.2 NPU（AX650N，24 TOPS INT8，8 GB）
- **[JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk)** — M.2被动散热器

## 快速开始

```bash
cd /root/cosyvoice3-build/dual_npu

# 最佳质量：ONNX HiFT
ONNX_HIFT=1 bash launch_dual_npu.sh "你好，今天天气很好。"

# 快速模式：NPU HiFT
bash launch_dual_npu.sh "你好，今天天气很好。"
```

详细说明请参阅 [English README](README.md)。

## 致谢

- [FunAudioLLM / 阿里巴巴](https://github.com/FunAudioLLM/CosyVoice) — CosyVoice3模型
- [AXERA-TECH](https://github.com/AXERA-TECH) — NPU量化、AXCL运行时
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — CPU高效LLM推理
- [M5Stack](https://shop.m5stack.com/) — Module LLM (AI-8850)
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS套件

## 许可证

MIT License。参见 [LICENSE](LICENSE)。
