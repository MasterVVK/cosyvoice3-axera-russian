#!/bin/bash
# Stability test: 20 consecutive TTS requests to dual-NPU daemon
#
# Prerequisites: both services running
#   systemctl start cosyvoice3-rkllm
#   systemctl start cosyvoice3-t2w
#
# Usage:
#   bash test_stability.sh              # Test via daemon socket
#   bash test_stability.sh --launch     # Launch daemons, test, then stop

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTS_SOCKET="/tmp/cv3_tts.sock"
TOKEN_SOCKET="/tmp/cv3_tokens.sock"
RESULTS_DIR="$SCRIPT_DIR/stability_test_results"
LAUNCH_MODE=false

if [ "$1" = "--launch" ]; then
    LAUNCH_MODE=true
fi

# --- Test phrases (20 diverse Russian phrases) ---
PHRASES=(
    "Привет, как дела?"
    "Сегодня хорошая погода для прогулки в парке."
    "Искусственный интеллект меняет мир."
    "Доброе утро! Завтрак готов."
    "Московское метро — одно из красивейших в мире."
    "Пожалуйста, передайте мне соль."
    "Через тернии к звёздам."
    "Книга лежит на столе рядом с лампой."
    "Завтра обещают дождь и сильный ветер."
    "Спасибо за вашу помощь, это очень важно."
    "Красная площадь — символ России."
    "Программирование требует терпения и внимания."
    "Давайте встретимся в кафе в три часа."
    "Музыка Чайковского известна во всём мире."
    "Этот рецепт передаётся из поколения в поколение."
    "Зимой дни короткие, а ночи длинные."
    "Робот успешно выполнил поставленную задачу."
    "Новый год — самый любимый праздник."
    "Технологии будущего уже здесь."
    "Спокойной ночи, приятных снов."
)

# --- Setup ---
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/test_log.txt"
echo "=== CosyVoice3 Stability Test ===" | tee "$LOG"
echo "Date: $(date)" | tee -a "$LOG"
echo "Phrases: ${#PHRASES[@]}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Check prerequisites ---
check_daemon() {
    if [ ! -S "$TTS_SOCKET" ]; then
        echo "ERROR: TTS daemon socket not found: $TTS_SOCKET"
        echo "Start daemons first or use --launch flag"
        exit 1
    fi
}

# --- Launch mode ---
if [ "$LAUNCH_MODE" = true ]; then
    echo "Launching dual-NPU daemons..."
    # Use launch_dual_npu.sh style but with daemon mode
    echo "NOTE: Launch mode requires daemons to already be running via systemd"
    echo "  systemctl start cosyvoice3-rkllm"
    echo "  sleep 30"
    echo "  systemctl start cosyvoice3-t2w"
    exit 1
fi

check_daemon

# --- Run tests ---
PASS=0
FAIL=0
TOTAL_TIME=0

for i in "${!PHRASES[@]}"; do
    n=$((i + 1))
    phrase="${PHRASES[$i]}"
    echo -n "[$n/20] \"${phrase:0:40}...\" " | tee -a "$LOG"

    t_start=$(date +%s%N)

    # Send request via tts_client.py
    result=$(python3 "$SCRIPT_DIR/tts_client.py" --timeout 120 "$phrase" 2>&1) || true

    t_end=$(date +%s%N)
    elapsed_ms=$(( (t_end - t_start) / 1000000 ))
    elapsed_s=$(echo "scale=2; $elapsed_ms / 1000" | bc)

    if echo "$result" | grep -q "^OK:"; then
        PASS=$((PASS + 1))
        TOTAL_TIME=$((TOTAL_TIME + elapsed_ms))
        echo "OK (${elapsed_s}s)" | tee -a "$LOG"

        # Copy output WAV
        if [ -f output.wav ]; then
            cp output.wav "$RESULTS_DIR/phrase_${n}.wav"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: $result" | tee -a "$LOG"
    fi
done

# --- Summary ---
echo "" | tee -a "$LOG"
echo "=== Results ===" | tee -a "$LOG"
echo "Pass: $PASS / ${#PHRASES[@]}" | tee -a "$LOG"
echo "Fail: $FAIL / ${#PHRASES[@]}" | tee -a "$LOG"
if [ $PASS -gt 0 ]; then
    AVG_MS=$((TOTAL_TIME / PASS))
    AVG_S=$(echo "scale=2; $AVG_MS / 1000" | bc)
    echo "Avg time: ${AVG_S}s" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED" | tee -a "$LOG"
    exit 0
else
    echo "SOME TESTS FAILED" | tee -a "$LOG"
    exit 1
fi
