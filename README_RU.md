# CosyVoice3 — Русский TTS на NPU AX650N

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-AX650N%20NPU-blue)]()
[![Language](https://img.shields.io/badge/Language-Russian%20%7C%20English%20%7C%20Chinese-green)]()

**[English](README.md)** | **[中文](README_ZH.md)**

Синтез речи CosyVoice3 с **клонированием русского голоса** на edge-NPU (AXERA AX650N / FriendlyElec CM3588).

Первый известный open-source проект, запускающий CosyVoice3 с русским voice cloning на NPU.

## Возможности

- **Клонирование русского голоса** — zero-shot с правильным system prefix для естественной просодии
- **Инференс на NPU** — полностью на AX650N (GPU не нужен)
- **Пост-обработка** — Butterworth LP фильтр убирает артефакты квантизации HiFT
- **Буферная фраза** — предотвращает обрезку конца речи на NPU
- **Автономный tokenizer** — без зависимости от репозитория CosyVoice
- **Без torchaudio** — работает на минимальных aarch64 системах

## Архитектура

```
Текст → [Tokenizer HTTP] → [LLM NPU] → [Flow NPU] → [HiFT NPU] → [PostProcess] → WAV
           EOS=1773          24 слоя     ODE steps     вокодер      LP 5kHz фильтр
```

## Железо

Наша конфигурация:
- **[FriendlyElec CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588)** NAS Kit (RK3588 SoC)
- **[M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)** — M.2 M-Key NPU ускоритель, вставляется в M.2 слот CM3588
- **[JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk)** — пассивный медный радиатор для M.2 2280 (50 медных рёбер, 401 Вт/мК), установлен на Module LLM

JEYI Q150 держит NPU на 63°C в простое / 71°C при полной нагрузке — без активного охлаждения.

> Подойдёт любая плата с M.2 M-Key слотом и PCIe: CM3588, Raspberry Pi 5, SBC на RK3588, x86 ПК с поддержкой AXCL.

## Быстрый старт

### Требования

- **Железо**: [CM3588](https://wiki.friendlyelec.com/wiki/index.php/CM3588) + [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) в M.2 слоте (или AX650N demo board / M4N-Dock)
- **Охлаждение**: [JEYI Finscold Q150](https://www.jeyi.com/products/jeyi-m-2-2280-ssd-high-performance-heatsink-copper-fins-with-aluminum-frame-passive-heat-sinks-50pcs-fins-cold-401-w-mk) или аналогичный M.2 радиатор (рекомендуется)
- **Runtime**: AXCL runtime 3.6+
- **Python**: 3.8+ с `transformers`, `scipy`, `numpy`
- **Модели**: скачать с [AXERA-TECH/CosyVoice3](https://huggingface.co/AXERA-TECH/CosyVoice3)

### 1. Скачать модели

```bash
pip install huggingface_hub
huggingface-cli download AXERA-TECH/CosyVoice3 --local-dir ./AXERA-CosyVoice3
```

### 2. Скопировать на устройство

```bash
rsync -avz --progress ./AXERA-CosyVoice3/ root@<IP_УСТРОЙСТВА>:/root/AXERA-CosyVoice3/
```

### 3. Установить наши скрипты

```bash
scp scripts/*.py scripts/*.sh root@<IP_УСТРОЙСТВА>:/root/AXERA-CosyVoice3/
ssh root@<IP_УСТРОЙСТВА>
cd /root/AXERA-CosyVoice3
chmod +x run_tts.sh
pip3 install scipy --break-system-packages
```

### 4. Запуск

```bash
./run_tts.sh "Привет, как дела? Сегодня хорошая погода." output.wav 30
```

## Использование

```bash
# Русский голос, лучшее качество
./run_tts.sh "Ваш текст здесь" output.wav 30 prompt_files_russian_v2

# Быстрый режим (~2x быстрее, чуть хуже качество)
./run_tts.sh "Ваш текст здесь" output.wav 10

# Свой голосовой промпт
./run_tts.sh "Текст" output.wav 30 my_custom_prompt
```

### Параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| Текст | обязательный | Текст для синтеза |
| Выход | `output.wav` | Путь к выходному файлу |
| Шаги | `30` | ODE steps: 10 (быстро), 20 (баланс), 30 (лучшее) |
| Промпт | `prompt_files_russian_v2` | Директория голосового промпта |

## Клонирование русского голоса

Для создания собственного голоса:

1. **Подготовьте аудио** (3-10 секунд, чистая речь, один голос)
2. **Запустите генерацию** (нужны [frontend-onnx модели](https://huggingface.co/AXERA-TECH/CosyVoice3)):

```bash
./generate_prompt.sh reference.wav "Текст из аудио" prompt_files_my_voice
```

**Важно**: prompt_text ОБЯЗАТЕЛЬНО должен начинаться с:
```
You are a helpful assistant.<|endofprompt|>Ваш текст здесь.
```
Без этого префикса качество голоса сильно ухудшается.

Подробнее: [docs/RUSSIAN_VOICE.md](docs/RUSSIAN_VOICE.md)

## Известные проблемы и решения

| Проблема | Причина | Решение |
|---|---|---|
| ВЧ писк (4-6 кГц) | Артефакты квантизации HiFT | LP фильтр 5 кГц (автоматически) |
| Обрезка последнего слова | NPU LLM рано генерирует EOS | Буферная фраза (автоматически) |
| Notch-фильтры не помогают | Артефакт широкополосный | Только полный low-pass |

Подробнее: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Производительность

CM3588 + AX650N (PCIe Gen3 x1), AXCL 3.6.5:

| Метрика | 30 шагов | 10 шагов |
|---|---|---|
| TTFT | 368 мс | ~300 мс |
| Скорость декодирования | 5.72 ток/с | ~10 ток/с |
| Общее время TTS | 27 сек | ~14 сек |
| Длительность аудио | 5.6 сек | 5.6 сек |
| Real-time factor | 4.8x | ~2.5x |

### Загрузка NPU

| Фаза | NPU % | Температура | CMM память |
|---|---|---|---|
| Загрузка моделей | 0% | 63°C | 18 → 1790 МиБ |
| LLM decode | 80-97% | 67-71°C | 1790 МиБ |
| Token2Wav | 96-100% | 70-71°C | 1790 МиБ |
| Простой | 0% | 63°C | 18 МиБ |

Подробнее: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

## Благодарности

- [FunAudioLLM / Alibaba](https://github.com/FunAudioLLM/CosyVoice) — модель CosyVoice3
- [AXERA-TECH](https://github.com/AXERA-TECH) — квантизация, runtime, бинарники для NPU
- [M5Stack](https://shop.m5stack.com/) — Module LLM (AI-8850) M.2 NPU ускоритель
- [FriendlyElec](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — CM3588 NAS Kit

## Лицензия

MIT. Модель CosyVoice3 под Apache-2.0 (Alibaba/FunAudioLLM).
