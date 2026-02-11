# CosyVoice3 俄语TTS — AX650N NPU端侧部署

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-AX650N%20NPU-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[English](README.md)** | **[Русский](README_RU.md)**

在边缘NPU硬件上运行CosyVoice3语音合成，支持**俄语语音克隆**（AXERA AX650N / FriendlyElec CM3588）。

据我们所知，这是首个在NPU上运行CosyVoice3俄语语音克隆的开源项目。

## 功能特性

- **俄语语音克隆** — 零样本语音克隆，使用正确的系统前缀实现自然韵律
- **NPU推理** — 完全在AX650N NPU上运行（无需GPU）
- **后处理** — Butterworth低通滤波器消除NPU HiFT量化噪声
- **缓冲短语** — 防止NPU上常见的语音末尾截断
- **独立分词器** — 无需CosyVoice仓库依赖，仅需`transformers`
- **无需torchaudio** — 通过struct实现WAV读写（适用于最小化aarch64环境）

## 架构

```
文本 → [分词器 HTTP] → [LLM NPU] → [Flow NPU] → [HiFT NPU] → [后处理] → WAV
         EOS=1773       24层         ODE步数       声码器        LP 5kHz滤波
```

## 硬件配置

我们的配置：
- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS套件（RK3588 SoC）
- **[M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)** — M.2 M-Key NPU加速卡，插入CM3588的M.2插槽
- **[JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk)** — M.2 2280被动式铜散热器（50片铜鳍片，401 W/mK），安装在Module LLM上

JEYI Q150在无风扇情况下将NPU温度控制在空闲63°C / 满载71°C — 安全范围内。

> 任何带M.2 M-Key插槽和PCIe支持的板卡均可使用：CM3588、Raspberry Pi 5、RK3588开发板、支持AXCL的x86 PC。

## 快速开始

### 前提条件

- **硬件**：[CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588) + [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)（或AX650N开发板 / M4N-Dock）
- **散热**：[JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk) 或类似M.2被动散热器（推荐）
- **运行时**：AXCL runtime 3.6+
- **Python**：3.8+，需安装 `transformers`、`scipy`、`numpy`
- **模型**：从 [AXERA-TECH/CosyVoice3](https://huggingface.co/AXERA-TECH/CosyVoice3) 下载

### 1. 下载模型

```bash
pip install huggingface_hub
huggingface-cli download AXERA-TECH/CosyVoice3 --local-dir ./AXERA-CosyVoice3
```

### 2. 复制到设备

```bash
rsync -avz --progress ./AXERA-CosyVoice3/ root@<设备IP>:/root/AXERA-CosyVoice3/
```

### 3. 安装脚本

```bash
scp scripts/*.py scripts/*.sh root@<设备IP>:/root/AXERA-CosyVoice3/
ssh root@<设备IP>
cd /root/AXERA-CosyVoice3
chmod +x run_tts.sh
pip3 install scipy --break-system-packages
```

### 4. 运行

```bash
# 俄语语音（最佳质量）
./run_tts.sh "Привет, как дела? Сегодня хорошая погода." output.wav 30

# 英语语音
./run_tts.sh "Hello, how are you today?" output.wav 30 prompt_files

# 中文语音（默认提示）
./run_tts.sh "你好，今天天气很好。" output.wav 30 prompt_files
```

## 使用方法

```bash
./run_tts.sh "要合成的文本" output.wav 30 prompt_files_russian_v2
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| 文本 | 必填 | 要合成的文本 |
| 输出 | `output.wav` | 输出WAV文件路径 |
| 步数 | `30` | ODE步数：10（快速）、20（平衡）、30（最佳） |
| 提示 | `prompt_files_russian_v2` | 语音提示目录 |

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `COSYVOICE_DIR` | `/root/AXERA-CosyVoice3` | 设备上的基础目录 |
| `TOKENIZER_HOST` | `127.0.0.1` | 分词器服务器主机 |
| `TOKENIZER_PORT` | `12345` | 分词器服务器端口 |
| `LP_CUTOFF` | `5000` | 低通滤波器截止频率（Hz） |
| `PAD_MS` | `300` | 静音填充（ms） |

## 俄语语音克隆

创建自定义俄语语音提示：

1. **准备参考音频**（3-10秒，清晰语音，单一说话人）
2. **运行提示生成**（需要 [frontend-onnx模型](https://huggingface.co/AXERA-TECH/CosyVoice3)）：

```bash
./generate_prompt.sh reference_audio.wav "音频中说的文字" prompt_files_my_voice
```

**关键**：prompt_text必须包含系统前缀：
```
You are a helpful assistant.<|endofprompt|>你的提示文本。
```
没有此前缀，语音质量会显著下降。

详见 [docs/RUSSIAN_VOICE.md](docs/RUSSIAN_VOICE.md)。

## 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|---|---|---|
| 高频尖啸（4-6 kHz） | NPU HiFT量化噪声 | Butterworth低通5kHz滤波（自动应用） |
| 末尾词被截断 | NPU LLM提前生成EOS | 缓冲短语"Вот так."（自动添加） |
| 陷波滤波器无效 | 噪声是宽带的，非单频 | 只有全频低通有效 |
| `wave.Error: unknown format: 3` | Python wave不支持IEEE浮点 | 基于struct的自定义WAV读写 |

详见 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。

## 性能基准

CM3588 + AX650N（PCIe Gen3 x1），AXCL 3.6.5：

| 指标 | 30步 | 10步 |
|---|---|---|
| TTFT（首个token延迟） | 368 ms | ~300 ms |
| 解码速度 | 5.72 token/s | ~10 token/s |
| TTS总时间 | 27秒 | ~14秒 |
| 输出音频时长 | 5.6秒 | ~5.6秒 |
| 实时因子 | 4.8x | ~2.5x |

### NPU利用率

| 阶段 | NPU % | 温度 | CMM内存 |
|---|---|---|---|
| 模型加载 | 0% | 63°C | 18 → 1790 MiB |
| LLM解码 | 80-97% | 67-71°C | 1790 MiB |
| Token2Wav | 96-100% | 70-71°C | 1790 MiB |
| 空闲 | 0% | 63°C | 18 MiB |

详见 [docs/BENCHMARKS.md](docs/BENCHMARKS.md)。

## 致谢

- [FunAudioLLM / 阿里巴巴](https://github.com/FunAudioLLM/CosyVoice) — CosyVoice3模型
- [AXERA（爱芯元智）](https://github.com/AXERA-TECH) — NPU量化、运行时和预编译二进制文件
- [M5Stack](https://shop.m5stack.com/) — Module LLM (AI-8850) M.2 NPU加速卡
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS套件

## 许可证

MIT许可证。CosyVoice3模型采用Apache-2.0许可证（阿里巴巴/FunAudioLLM）。
