#!/bin/bash
# Launch CosyVoice3 in Dual-NPU mode:
#   LLM: llama.cpp CPU A76 (~40 tok/s) or RK3588 NPU RKLLM (~10 tok/s)
#   Token2Wav: AX650N NPU (Flow + HiFT, dedicated)
#
# Usage:
#   bash launch_dual_npu.sh "Привет, как дела?"   # Single-shot mode
#   bash launch_dual_npu.sh                        # Default text
#   bash launch_dual_npu.sh --daemon               # Daemon mode (persistent)
#
# Environment:
#   LLM_BACKEND=llamacpp|rkllm  (default: llamacpp)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- LLM Backend selection ---
LLM_BACKEND="${LLM_BACKEND:-llamacpp}"

# --- Configuration ---
# llama.cpp backend
GGUF_MODEL="${GGUF_MODEL:-/root/cosyvoice3-rknn/cosyvoice3_qwen2_q4_k_m.gguf}"
GGUF_THREADS="${GGUF_THREADS:-4}"

# RKLLM backend (legacy)
RKLLM_MODEL="${RKLLM_MODEL:-/root/cosyvoice3-rknn/cosyvoice3_llm_rk3588.rkllm}"
RKLLM_PIPELINE="${RKLLM_PIPELINE:-/root/cosyvoice3-rknn/cosyvoice3_rknn_pipeline.py}"

# Shared
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-/root/cosyvoice3-rknn/cosyvoice3_embeddings}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/root/cosyvoice3-rknn/cosyvoice3_qwen2_for_rkllm}"

AXERA_BINARY="${AXERA_BINARY:-/root/cosyvoice3-build/cosyvoice3-axera/cpp/build/install/bin/cosyvoice3_tts}"
AXERA_MODELS="${AXERA_MODELS:-/root/cosyvoice3-build/cosyvoice3-axera/models}"

TOKEN_SOCKET="/tmp/cv3_tokens.sock"
TTS_SOCKET="/tmp/cv3_tts.sock"
HIFT_SOCKET="/tmp/cv3_hift.sock"

TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-25}"
TOP_P="${TOP_P:-0.8}"
N_TIMESTEPS="${N_TIMESTEPS:-5}"

# ONNX HiFT: set ONNX_HIFT=1 to use CPU ONNX instead of NPU for HiFT (cleaner audio)
# Dynamic mode (default): uses f0_predictor + decode_dynamic models, any mel_len
# Legacy mode: ONNX_HIFT_LEGACY=1 uses fixed P1/P2 models with crossfade
ONNX_HIFT="${ONNX_HIFT:-0}"
ONNX_HIFT_LEGACY="${ONNX_HIFT_LEGACY:-0}"
ONNX_HIFT_DIR="${ONNX_HIFT_DIR:-/root/cosyvoice3-rknn}"
ONNX_HIFT_COMPONENTS="${ONNX_HIFT_COMPONENTS:-/root/cosyvoice3-rknn/hift-components}"
ONNX_HIFT_THREADS="${ONNX_HIFT_THREADS:-4}"

# --- Parse arguments ---
DAEMON_MODE=false
TEXT=""
for arg in "$@"; do
    if [ "$arg" = "--daemon" ]; then
        DAEMON_MODE=true
    else
        TEXT="$arg"
    fi
done

if [ "$DAEMON_MODE" = false ] && [ -z "$TEXT" ]; then
    TEXT="Привет, как дела? Сегодня хорошая погода для прогулки."
fi

# --- Functions ---
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "$TOKEN_SERVER_PID" ] && kill -0 "$TOKEN_SERVER_PID" 2>/dev/null; then
        kill "$TOKEN_SERVER_PID" 2>/dev/null
        wait "$TOKEN_SERVER_PID" 2>/dev/null || true
        echo "Token server stopped (PID $TOKEN_SERVER_PID)"
    fi
    if [ -n "$HIFT_SERVER_PID" ] && kill -0 "$HIFT_SERVER_PID" 2>/dev/null; then
        kill "$HIFT_SERVER_PID" 2>/dev/null
        wait "$HIFT_SERVER_PID" 2>/dev/null || true
        echo "ONNX HiFT server stopped (PID $HIFT_SERVER_PID)"
    fi
    rm -f "$TOKEN_SOCKET" "$TTS_SOCKET" "$HIFT_SOCKET"
}
trap cleanup EXIT

# --- Pre-flight checks ---
echo "=== CosyVoice3 Dual-NPU TTS ==="
if [ "$LLM_BACKEND" = "llamacpp" ]; then
    echo "LLM:       llama.cpp CPU A76 (Q4_K_M, ~40 tok/s)"
else
    echo "LLM:       RK3588 RKLLM (W8A8, ~10 tok/s)"
fi
if [ "$ONNX_HIFT" = "1" ]; then
    echo "Token2Wav: AX650N (Flow) + CPU ONNX (HiFT) — clean audio"
else
    echo "Token2Wav: AX650N (Flow + HiFT)"
fi
echo "ODE steps: $N_TIMESTEPS"
if [ "$DAEMON_MODE" = true ]; then
    echo "Mode:      DAEMON (persistent, socket: $TTS_SOCKET)"
else
    echo "Mode:      Single-shot"
fi
echo ""

# Check AX650N binary
if [ ! -f "$AXERA_BINARY" ]; then
    echo "ERROR: cosyvoice3_tts binary not found: $AXERA_BINARY"
    echo "  Build: cd /root/cosyvoice3-build/cosyvoice3-axera/cpp && bash build.sh"
    exit 1
fi

# --- Step 1: Start Token Server ---
rm -f "$TOKEN_SOCKET"

if [ "$LLM_BACKEND" = "llamacpp" ]; then
    # llama.cpp backend: pin to A76 cores (4-7) for maximum performance
    TOKEN_SERVER_SCRIPT="${SCRIPT_DIR}/llamacpp_token_server.py"

    if [ ! -f "$GGUF_MODEL" ]; then
        echo "ERROR: GGUF model not found: $GGUF_MODEL"
        exit 1
    fi
    if [ ! -f "$TOKEN_SERVER_SCRIPT" ]; then
        echo "ERROR: Token server not found: $TOKEN_SERVER_SCRIPT"
        exit 1
    fi

    echo "Starting llama.cpp Token Server (A76 cores 4-7)..."
    OPENBLAS_NUM_THREADS=1 taskset -c 4-7 python3 -u "$TOKEN_SERVER_SCRIPT" \
        --model "$GGUF_MODEL" \
        --embeddings "$EMBEDDINGS_DIR" \
        --tokenizer "$TOKENIZER_DIR" \
        --socket "$TOKEN_SOCKET" \
        --temperature "$TEMPERATURE" \
        --top_k "$TOP_K" \
        --top_p "$TOP_P" \
        --threads "$GGUF_THREADS" \
        --kv_cache_dir "/tmp" &
    TOKEN_SERVER_PID=$!
else
    # RKLLM backend: pin to A55 cores (0-3) to avoid NPU contention
    TOKEN_SERVER_SCRIPT="${SCRIPT_DIR}/rkllm_token_server.py"

    if [ ! -f "$RKLLM_MODEL" ]; then
        echo "ERROR: RKLLM model not found: $RKLLM_MODEL"
        exit 1
    fi
    if [ ! -f "$TOKEN_SERVER_SCRIPT" ]; then
        echo "ERROR: Token server not found: $TOKEN_SERVER_SCRIPT"
        exit 1
    fi

    echo "Starting RKLLM Token Server (A55 cores 0-3)..."
    taskset -c 0-3 python3 -u "$TOKEN_SERVER_SCRIPT" \
        --model "$RKLLM_MODEL" \
        --embeddings "$EMBEDDINGS_DIR" \
        --tokenizer "$TOKENIZER_DIR" \
        --socket "$TOKEN_SOCKET" \
        --temperature "$TEMPERATURE" \
        --top_k "$TOP_K" \
        --top_p "$TOP_P" &
    TOKEN_SERVER_PID=$!
fi

# Wait for socket to appear (max 30s for model loading)
echo "Waiting for token server to be ready..."
for i in $(seq 1 60); do
    if [ -S "$TOKEN_SOCKET" ]; then
        echo "Token server ready (PID $TOKEN_SERVER_PID, socket $TOKEN_SOCKET)"
        break
    fi
    if ! kill -0 "$TOKEN_SERVER_PID" 2>/dev/null; then
        echo "ERROR: Token server process died"
        exit 1
    fi
    sleep 0.5
done

if [ ! -S "$TOKEN_SOCKET" ]; then
    echo "ERROR: Token server socket not created after 30s"
    kill "$TOKEN_SERVER_PID" 2>/dev/null
    exit 1
fi

# --- Step 1.5: Start ONNX HiFT server (if enabled) ---
ONNX_HIFT_ARG=""
if [ "$ONNX_HIFT" = "1" ]; then
    HIFT_SERVER_SCRIPT="${SCRIPT_DIR}/onnx_hift_server.py"
    if [ ! -f "$HIFT_SERVER_SCRIPT" ]; then
        echo "ERROR: ONNX HiFT server not found: $HIFT_SERVER_SCRIPT"
        exit 1
    fi
    rm -f "$HIFT_SOCKET"

    HIFT_SERVER_ARGS="--components_dir $ONNX_HIFT_COMPONENTS --onnx_dir $ONNX_HIFT_DIR --socket $HIFT_SOCKET --threads $ONNX_HIFT_THREADS"
    if [ "$ONNX_HIFT_LEGACY" = "1" ]; then
        HIFT_SERVER_ARGS="$HIFT_SERVER_ARGS --legacy"
        echo "Starting ONNX HiFT server (LEGACY P1/P2, A76 cores 4-7)..."
    else
        echo "Starting ONNX HiFT server (DYNAMIC, A76 cores 4-7)..."
    fi

    taskset -c 4-7 python3 -u "$HIFT_SERVER_SCRIPT" $HIFT_SERVER_ARGS &
    HIFT_SERVER_PID=$!

    # Wait for ONNX HiFT socket
    for i in $(seq 1 30); do
        if [ -S "$HIFT_SOCKET" ]; then
            echo "ONNX HiFT server ready (PID $HIFT_SERVER_PID)"
            break
        fi
        if ! kill -0 "$HIFT_SERVER_PID" 2>/dev/null; then
            echo "ERROR: ONNX HiFT server died"
            exit 1
        fi
        sleep 0.5
    done
    if [ ! -S "$HIFT_SOCKET" ]; then
        echo "ERROR: ONNX HiFT socket not created after 15s"
        exit 1
    fi
    ONNX_HIFT_ARG="--onnx_hift $HIFT_SOCKET"
    if [ "${ONNX_CHUNKED:-0}" = "1" ]; then
        echo "  Mode: chunked (per-chunk HiFT + crossfade)"
    else
        ONNX_HIFT_ARG="$ONNX_HIFT_ARG --onnx_fullmel 1"
        echo "  Mode: fullmel (accumulate mel, single HiFT pass) — default"
    fi
fi

# --- Step 2: Run cosyvoice3_tts ---
echo ""

# Pin Token2Wav to A55 cores (0-3) if using llamacpp, otherwise let it use any core
if [ "$LLM_BACKEND" = "llamacpp" ]; then
    TTS_TASKSET="taskset -c 0-3"
else
    TTS_TASKSET=""
fi

if [ "$DAEMON_MODE" = true ]; then
    echo "Starting Token2Wav daemon..."
    echo "Send requests via: python3 tts_client.py \"text\""
    echo ""

    $TTS_TASKSET $AXERA_BINARY \
        --text "" \
        --token2wav_axmodel_dir "$AXERA_MODELS" \
        --prompt_files "${AXERA_MODELS}/prompt_files" \
        --n_timesteps "$N_TIMESTEPS" \
        --devices "0," \
        --filename_decoder_weight "${AXERA_MODELS}/decoder_weight.bin" \
        --external_tokens "$TOKEN_SOCKET" \
        --daemon "$TTS_SOCKET" \
        $ONNX_HIFT_ARG \
        --continue 0
else
    echo "Running TTS: \"$TEXT\""
    echo ""

    rm -f output*.wav

    $TTS_TASKSET $AXERA_BINARY \
        --text "$TEXT" \
        --token2wav_axmodel_dir "$AXERA_MODELS" \
        --prompt_files "${AXERA_MODELS}/prompt_files" \
        --n_timesteps "$N_TIMESTEPS" \
        --devices "0," \
        --filename_decoder_weight "${AXERA_MODELS}/decoder_weight.bin" \
        --external_tokens "$TOKEN_SOCKET" \
        $ONNX_HIFT_ARG \
        --continue 0

    echo ""
    echo "Output: output.wav"
    echo "Chunks: output_*.wav"
fi
