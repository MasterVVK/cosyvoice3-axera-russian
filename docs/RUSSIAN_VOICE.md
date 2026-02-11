# Russian Voice Cloning Guide

How to create custom Russian voice prompts for CosyVoice3 on AX650N NPU.

## Overview

CosyVoice3 uses zero-shot voice cloning: given a short reference audio + its transcript, it can reproduce that voice for any new text. The quality depends heavily on:

1. **Reference audio quality** — clear, single speaker, minimal noise
2. **System prefix** — MUST be included in prompt text
3. **Minimum speech tokens** — at least 75 tokens (roughly 3+ seconds of speech)

## Prerequisites

You need the `frontend-onnx/` models from AXERA-TECH:
- `campplus.onnx` (27 MB) — speaker verification model
- `speech_tokenizer_v3.onnx` (925 MB) — speech tokenizer

These are included in the full [AXERA-TECH/CosyVoice3](https://huggingface.co/AXERA-TECH/CosyVoice3) download.

Additional Python packages (on the machine running prompt generation):
```bash
pip install torch torchaudio onnxruntime transformers wetext inflect
```

Also download wetext model:
```bash
pip install modelscope
modelscope download --model pengzhendong/wetext --local_dir pengzhendong/wetext
```

## Step 1: Prepare Reference Audio

**Requirements:**
- Duration: 3-10 seconds
- Sample rate: 16 kHz or higher
- Single speaker, clear pronunciation
- Minimal background noise
- WAV format preferred

**Good sources:**
- Record yourself
- Use a short clip from a podcast/audiobook
- [Common Voice](https://commonvoice.mozilla.org/) dataset (open-source multilingual speech)

## Step 2: Prepare Prompt Text

Write the exact text spoken in the reference audio.

**CRITICAL**: Add the system prefix before your text:

```
You are a helpful assistant.<|endofprompt|>Ваш текст, произнесённый в аудио.
```

Without this prefix, the generated speech has:
- Degraded prosody (unnatural intonation)
- Stronger foreign accent
- Less speaker similarity

Example file `prompt_text.txt`:
```
You are a helpful assistant.<|endofprompt|>Добрый день! Сегодня отличная погода для прогулки.
```

## Step 3: Generate Prompt Embeddings

Using our wrapper script:
```bash
./generate_prompt.sh reference_audio.wav "You are a helpful assistant.<|endofprompt|>Текст из аудио" prompt_files_my_voice
```

Or manually with AXERA-TECH's process_prompt.py:
```bash
cd /root/AXERA-CosyVoice3

python3 scripts/process_prompt.py \
    --prompt_text "You are a helpful assistant.<|endofprompt|>Текст из аудио" \
    --prompt_speech reference_audio.wav \
    --output prompt_files_my_voice \
    --model_dir scripts/CosyVoice-BlankEN/ \
    --wetext_dir pengzhendong/wetext
```

This generates 6 files in `prompt_files_my_voice/`:
```
flow_embedding.txt
flow_prompt_speech_token.txt
llm_embedding.txt
llm_prompt_speech_token.txt
prompt_speech_feat.txt
prompt_text.txt
```

**Check**: The script asserts `speech_token >= 75`. If it fails, use a longer reference audio.

## Step 4: Test

```bash
./run_tts.sh "Привет, как дела? Сегодня хорошая погода, давайте пойдём на прогулку." \
    test_voice.wav 30 prompt_files_my_voice
```

## Tips for Best Quality

1. **Use 30 ODE steps** — significantly better quality than 10 steps
2. **Reference audio in Russian** — while cross-lingual works, same-language gives better accent
3. **Clear pronunciation** — avoid mumbling or fast speech in reference
4. **5-7 seconds reference** — too short and it lacks speaker characteristics, too long and it may confuse the model
5. **Post-processing is automatic** — LP filter and padding are applied by `run_tts.sh`

## Pre-built Prompts

We provide two tested Russian voice prompts:

| Prompt Directory | Source | Quality |
|---|---|---|
| `prompt_files_russian_v2` | TTS-cloned voice + system prefix | Best overall |
| `prompt_files_human_ru_v2` | Real human voice (Common Voice) + prefix | Good, natural |

To generate these yourself, see the reference audio preparation above.
