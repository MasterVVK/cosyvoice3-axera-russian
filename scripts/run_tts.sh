#!/bin/bash
# CosyVoice3 TTS Pipeline — AX650N NPU with post-processing
# Usage: ./run_tts.sh "Text to synthesize" [output.wav] [steps] [prompt_dir]
#
# Steps: 10 (fast, ~10 tok/s), 20 (balanced), 30 (best quality, ~5.7 tok/s)
# Prompt: prompt_files_russian_v2 (default), prompt_files_human_ru_v2, prompt_files (Chinese)

set -euo pipefail

# ============================================================
# Configuration — adjust these paths for your setup
# ============================================================
COSYVOICE_DIR="${COSYVOICE_DIR:-/root/AXERA-CosyVoice3}"
TOKENIZER_HOST="${TOKENIZER_HOST:-127.0.0.1}"
TOKENIZER_PORT="${TOKENIZER_PORT:-12345}"
TOKENIZER_MODEL_DIR="${TOKENIZER_MODEL_DIR:-scripts/CosyVoice-BlankEN}"

# Model paths (relative to COSYVOICE_DIR)
MODEL_DIR="CosyVoice-BlankEN-Ax650-C64-P256-CTX512"
TOKEN2WAV_DIR="token2wav-axmodels"

# Post-processing defaults
LP_CUTOFF="${LP_CUTOFF:-5000}"       # Butterworth low-pass filter cutoff (Hz)
PAD_MS="${PAD_MS:-300}"              # Silence padding at end (ms)
BUFFER_PHRASE="${BUFFER_PHRASE:-Вот так.}"  # Buffer phrase to prevent end truncation

# ============================================================
# Arguments
# ============================================================
TEXT="${1:-Hello, this is a test.}"
OUTPUT="${2:-output.wav}"
STEPS="${3:-30}"
PROMPT="${4:-prompt_files_russian_v2}"

cd "${COSYVOICE_DIR}"

# Ensure tokenizer is running
if ! curl -s "http://${TOKENIZER_HOST}:${TOKENIZER_PORT}/health" > /dev/null 2>&1; then
    echo "Starting tokenizer server..."
    nohup python3 tokenizer_server.py \
        --model_dir "${TOKENIZER_MODEL_DIR}" \
        --host "${TOKENIZER_HOST}" \
        --port "${TOKENIZER_PORT}" > /tmp/tokenizer.log 2>&1 &
    sleep 3
    if ! curl -s "http://${TOKENIZER_HOST}:${TOKENIZER_PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: Tokenizer failed to start. Check /tmp/tokenizer.log"
        exit 1
    fi
fi

# Add buffer phrase to prevent end truncation on NPU
FULL_TEXT="${TEXT} ${BUFFER_PHRASE}"

echo "Text: ${TEXT}"
echo "Steps: ${STEPS}, Prompt: ${PROMPT}"
echo "---"

# Run NPU inference
./main_axcl_aarch64 \
    --template_filename_axmodel "${MODEL_DIR}/qwen2_p64_l%d_together.axmodel" \
    --token2wav_axmodel_dir "${TOKEN2WAV_DIR}/" \
    --n_timesteps "${STEPS}" \
    --axmodel_num 24 \
    --bos 0 --eos 0 \
    --filename_tokenizer_model "http://${TOKENIZER_HOST}:${TOKENIZER_PORT}" \
    --filename_post_axmodel "${MODEL_DIR}/qwen2_post.axmodel" \
    --filename_decoder_axmodel "${MODEL_DIR}/llm_decoder.axmodel" \
    --filename_tokens_embed "${MODEL_DIR}/model.embed_tokens.weight.bfloat16.bin" \
    --filename_llm_embed "${MODEL_DIR}/llm.speech_embedding.float16.bin" \
    --filename_speech_embed "${MODEL_DIR}/llm.speech_embedding.float16.bin" \
    --continue 0 \
    --prompt_files "${PROMPT}" \
    --devices "0," \
    --text "${FULL_TEXT}"

# Post-process: LP filter + fade-out + padding
if [ -f output.wav ]; then
    python3 postprocess.py output.wav "${OUTPUT}" "${LP_CUTOFF}" "${PAD_MS}"
    if [ "${OUTPUT}" != "output.wav" ]; then
        rm -f output.wav
    fi
else
    echo "ERROR: NPU inference did not produce output.wav"
    exit 1
fi

echo "---"
echo "Done: ${OUTPUT}"
