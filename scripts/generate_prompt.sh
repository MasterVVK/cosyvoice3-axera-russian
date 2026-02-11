#!/bin/bash
# Generate Russian voice prompt embeddings for CosyVoice3
#
# This script wraps AXERA-TECH's process_prompt.py with the correct
# system prefix required for Russian voice cloning.
#
# Prerequisites:
#   - AXERA-TECH CosyVoice3 repo with frontend-onnx/ models
#   - Python packages: torch, torchaudio, onnxruntime, wetext, inflect, transformers
#   - wetext model: modelscope download --model pengzhendong/wetext --local_dir pengzhendong/wetext
#
# Usage:
#   ./generate_prompt.sh /path/to/reference_audio.wav "Текст произнесённый в аудио" output_dir
#
# The reference audio should be:
#   - 3-10 seconds of clear speech
#   - 16kHz+ sample rate
#   - Single speaker, minimal background noise

set -euo pipefail

AXERA_DIR="${AXERA_DIR:-/root/AXERA-CosyVoice3}"
AUDIO="${1:?Usage: ./generate_prompt.sh audio.wav \"prompt text\" output_dir}"
PROMPT_TEXT="${2:?Provide the text spoken in the reference audio}"
OUTPUT_DIR="${3:?Provide output directory name}"

# IMPORTANT: System prefix is required for proper voice quality
# Without it, the generated speech has degraded prosody and accent
SYSTEM_PREFIX="You are a helpful assistant.<|endofprompt|>"
FULL_PROMPT="${SYSTEM_PREFIX}${PROMPT_TEXT}"

echo "Reference audio: ${AUDIO}"
echo "Prompt text: ${FULL_PROMPT}"
echo "Output: ${OUTPUT_DIR}"
echo "---"

cd "${AXERA_DIR}"

python3 scripts/process_prompt.py \
    --prompt_text "${FULL_PROMPT}" \
    --prompt_speech "${AUDIO}" \
    --output "${OUTPUT_DIR}" \
    --model_dir scripts/CosyVoice-BlankEN/ \
    --wetext_dir pengzhendong/wetext

echo "---"
echo "Generated prompt files in ${OUTPUT_DIR}/"
echo "Use with: ./run_tts.sh \"text\" output.wav 30 ${OUTPUT_DIR}"
