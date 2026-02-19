# CosyVoice3 — Русский TTS: RK3588 CPU + AX650N NPU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20%2B%20AX650N-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[English](README.md)** | **[中文](README_ZH.md)**

Синтез речи CosyVoice3 с **клонированием русского голоса** на гибридном железе RK3588 + AX650N NPU. Качественное аудио с **RTF 1.3x** (почти real-time).

## Архитектура

```
Текст → [Tokenizer] → [LLM llama.cpp] → токены → [Flow DiT] → mel → [HiFT ONNX] → WAV
                        RK3588 CPU                  AX650N NPU         RK3588 CPU
                        ядра A76                    PCIe M.2            ядра A76
                        Q4_K_M GGUF                 FP16 axmodel        FP32 dynamic
                        ~36 ток/с                   5 шагов ODE         ~2.3с вокодер
```

**Почему гибрид?**
- **LLM на CPU** (llama.cpp Q4_K_M): 36 ток/с — лучше качество чем NPU W8A8
- **Flow DiT на AX650N NPU**: ODE solver за 5 шагов — идеально для NPU (тензорные операции)
- **HiFT на CPU** (ONNX Runtime FP32): вокодер с SineGen sin²(x) ломается при FP16/INT8

## Ключевые особенности

- **Mel crossfade** — 10-фреймовый линейный blend на стыках чанков убирает артефакты
- **KV-cache** — сохранение состояния LLM на диск, 33x ускорение prefill при повторном спикере
- **Fullmel accumulation** — mel со всех flow-чанков объединяется перед одним проходом HiFT
- **Естественный EOS** — LLM генерирует токен 6562 сам, без принудительного обрезания
- **Tanh soft limiter** — мягкое ограничение пиков без жёсткого клиппинга
- **Клонирование русского голоса** — zero-shot с system prefix для естественной просодии

## Производительность

CM3588 (RK3588, 32 ГБ RAM) + M5Stack AI-8850 (AX650N) через PCIe M.2:

| Метрика | Холодный старт | KV-cache |
|---------|---------------|----------|
| Prefill | 2.0с | **0.06с** |
| LLM decode | 3.2с (36 ток/с) | 3.2с |
| Flow (AX650N) | ~3.0с (параллельно) | ~3.0с |
| HiFT (ONNX CPU) | 2.3с | 2.3с |
| **Итого** | **~8.3с** | **~6.3с** |
| Длительность аудио | ~5.5с | ~5.5с |
| **RTF** | **1.5x** | **1.3x** |

## Оборудование

- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS Kit — RK3588 (4x A76 + 4x A55), 32 ГБ RAM
- **[M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)** — M.2 NPU (AX650N, 24 TOPS INT8, 8 ГБ)
- **[JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk)** — пассивный медный радиатор для M.2

## Быстрый старт

### Требования

- **AXCL runtime** 3.6+ на CM3588
- **llama.cpp** собранный для aarch64 с shared libs
- **Python** 3.8+ с `numpy`, `onnxruntime`, `transformers`
- **Модели**: axmodel от [AXERA-TECH/CosyVoice3](https://huggingface.co/AXERA-TECH/CosyVoice3), GGUF LLM, ONNX HiFT

### Запуск

```bash
cd /root/cosyvoice3-build/dual_npu

# Лучшее качество: ONNX HiFT (fullmel режим)
ONNX_HIFT=1 bash launch_dual_npu.sh "Привет, как дела? Сегодня хорошая погода."

# Быстрый режим: NPU HiFT (ниже качество, RTF ~1.2x)
bash launch_dual_npu.sh "Привет, как дела?"

# С настройкой температуры
TEMPERATURE=0.8 ONNX_HIFT=1 bash launch_dual_npu.sh "Ваш текст здесь"
```

Результат: `output.wav` (24 кГц, float32)

## Технические детали

### Mel crossfade

Flow encoder генерирует mel для перекрывающихся окон токенов. Без crossfade жёсткая склейка создаёт щелчки и шуршание. Решение: 10-фреймовый линейный blend в `Token2wav.hpp`. Артефакты снижены в 3 раза.

### Fullmel exact_start fix

Flow estimator на NPU имеет фиксированные размеры выхода (200/250/300 mel frames). При 78 токенах генерируется 150 mel frames (не 156 = 78×2). Оригинальная формула `exact_start = token_offset * 2` приводила к остановке mel-аккумулятора после 3-го чанка (~3 секунды). Исправлено формулой со скользящим окном.

### CPU Core Affinity (критично!)

Без привязки к ядрам — замедление в 5 раз:
- **LLM + HiFT**: `taskset -c 4-7` (A76, 2.35 ГГц)
- **Token2Wav (C++)**: `taskset -c 0-3` (A55, 1.8 ГГц)
- **OPENBLAS_NUM_THREADS=1** — чтобы numpy не занимал A76

## Неудачные попытки ускорения HiFT

| Подход | Результат | Причина |
|--------|-----------|---------|
| RKNN NPU (RK3588) | All-zeros / corr -0.26 | SineGen sin²(x) ломается при FP16 |
| AX650N NPU | SNR 13.7 dB (шум) | Та же чувствительность к квантизации |
| INT8 ONNX | 5.5x медленнее, SNR -1 dB | SnakeAlpha убивает INT8 |
| MNN Mali GPU | Ошибка broadcast | Несовпадение размеров SineGen |

**Вывод**: HiFT вокодер требует FP32. CPU ONNX — единственный рабочий вариант.

## Структура проекта

```
cosyvoice3-axera-russian/
├── cpp/                          # C++ пайплайн (RK3588 + AX650N)
│   ├── CMakeLists.txt
│   └── src/
│       ├── main.cpp              # Token reader + Token2Wav поток
│       └── runner/
│           ├── Token2wav.hpp     # Flow (NPU) + HiFT + mel accumulation
│           ├── LLM.hpp           # LLM runner
│           └── CV2_Tokenizer.hpp # Обёртка токенизатора
├── dual_npu/                     # Python серверы + скрипты запуска
│   ├── launch_dual_npu.sh        # Главный скрипт запуска
│   ├── llamacpp_token_server.py  # LLM token server (llama.cpp)
│   ├── onnx_hift_server.py       # ONNX HiFT vocoder server
│   ├── llama_wrapper.c           # C wrapper для llama.cpp
│   └── tts_client.py             # Тестовый клиент
├── scripts/                      # Оригинальные скрипты (all-NPU)
├── docs/                         # Документация
└── examples/                     # Примеры промптов
```

## Благодарности

- [FunAudioLLM / Alibaba](https://github.com/FunAudioLLM/CosyVoice) — модель CosyVoice3
- [AXERA-TECH](https://github.com/AXERA-TECH) — квантизация, AXCL runtime, axmodel
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — эффективный инференс LLM на CPU
- [M5Stack](https://shop.m5stack.com/) — Module LLM (AI-8850)
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS Kit

## Лицензия

MIT License. См. [LICENSE](LICENSE).

CosyVoice3: Apache-2.0 (Alibaba/FunAudioLLM).
AXERA-TECH runtime: см. соответствующие репозитории.
