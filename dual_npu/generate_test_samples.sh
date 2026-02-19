#!/bin/bash
# Generate test audio samples via Dual-NPU pipeline
# RKLLM (RK3588) for LLM + AX650N for Token2Wav

set -e

AXERA_BINARY="/root/cosyvoice3-build/cosyvoice3-axera/cpp/build/install/bin/cosyvoice3_tts"
AXERA_MODELS="/root/cosyvoice3-build/cosyvoice3-axera/models"
TOKEN_SOCKET="/tmp/cv3_tokens.sock"
OUTPUT_DIR="/root/dual_npu_samples"
N_TIMESTEPS=5

mkdir -p "$OUTPUT_DIR"

# Test phrases
declare -a PHRASES=(
    "Привет, как дела?"
    "Сегодня хорошая погода для прогулки в парке."
    "Добро пожаловать в систему умного дома."
    "Температура в комнате двадцать два градуса."
    "Напоминание: через пятнадцать минут начнётся совещание."
)

declare -a NAMES=(
    "01_privet"
    "02_pogoda"
    "03_umny_dom"
    "04_temperatura"
    "05_napominanie"
)

if [ ! -S "$TOKEN_SOCKET" ]; then
    echo "ERROR: Token server not running ($TOKEN_SOCKET)"
    exit 1
fi

echo "=== Generating ${#PHRASES[@]} test samples ==="
echo "Output: $OUTPUT_DIR"
echo ""

for i in "${!PHRASES[@]}"; do
    TEXT="${PHRASES[$i]}"
    NAME="${NAMES[$i]}"
    echo "--- [$((i+1))/${#PHRASES[@]}] $NAME ---"
    echo "  Text: $TEXT"

    cd "$OUTPUT_DIR"
    rm -f output.wav output_*.wav

    $AXERA_BINARY \
        --text "$TEXT" \
        --token2wav_axmodel_dir "$AXERA_MODELS" \
        --prompt_files "${AXERA_MODELS}/prompt_files" \
        --n_timesteps "$N_TIMESTEPS" \
        --devices "0," \
        --filename_decoder_weight "${AXERA_MODELS}/decoder_weight.bin" \
        --external_tokens "$TOKEN_SOCKET" \
        --continue 0 2>&1 | grep -E "tts total|LLM tokens"

    # Rename output
    if [ -f output.wav ]; then
        mv output.wav "${NAME}.wav"
        echo "  Saved: ${NAME}.wav ($(stat -c%s "${NAME}.wav") bytes)"
    else
        echo "  WARNING: No output.wav generated!"
    fi
    # Keep chunk files too
    for chunk in output_*.wav; do
        [ -f "$chunk" ] && mv "$chunk" "${NAME}_${chunk}"
    done
    echo ""
done

echo "=== All samples generated ==="
ls -la "$OUTPUT_DIR"/*.wav 2>/dev/null | awk '{print $5, $9}'
echo ""
echo "Copy to NAS: rsync -av $OUTPUT_DIR/*.wav user@nv-02:/path/"
