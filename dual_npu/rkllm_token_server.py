#!/usr/bin/env python3
"""
RKLLM Token Server for dual-NPU CosyVoice3 TTS.

Runs CosyVoice3 LLM on RK3588 NPU (RKLLM W8A8) and streams speech tokens
to a Token2Wav consumer (AX650N) via Unix domain socket.

Protocol (per request):
  Client → Server: [4 bytes msg_len] [msg_len bytes JSON: {"text":"...", "prompt_speech_tokens":[...], "prompt_text_tokens":[...]}]
  Server → Client: [4 bytes int32: token_id] repeated per token
                   [4 bytes int32: -1] when done (EOS/max)
                   [4 bytes int32: -2] on error

Usage on CM3588:
  # Start server:
  python3 rkllm_token_server.py \
      --model /root/cosyvoice3-rknn/cosyvoice3_llm_rk3588.rkllm \
      --embeddings /root/cosyvoice3-rknn/cosyvoice3_embeddings \
      --tokenizer /root/cosyvoice3-rknn/cosyvoice3_qwen2_for_rkllm \
      --socket /tmp/cv3_tokens.sock

  # Client example (Python):
  import socket, struct, json
  s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  s.connect("/tmp/cv3_tokens.sock")
  msg = json.dumps({"text": "Привет"}).encode()
  s.sendall(struct.pack("<I", len(msg)) + msg)
  while True:
      data = s.recv(4)
      token = struct.unpack("<i", data)[0]
      if token < 0: break
      print(f"Token: {token}")
  s.close()
"""

import os

# Pin Python/OpenBLAS to A55 cores (0-3) BEFORE importing numpy.
# RKLLM uses A76 cores (4-7) — avoids CPU contention that causes 2x NPU slowdown.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
try:
    os.sched_setaffinity(0, {0, 1, 2, 3})
except OSError:
    pass  # Non-Linux or restricted

import sys
import socket
import struct
import json
import argparse
import time
import signal
import numpy as np

SPEECH_TOKEN_SIZE = 6561
HIDDEN_SIZE = 896
SENTINEL_DONE = -1
SENTINEL_ERROR = -2


class RKLLMTokenServer:
    def __init__(self, model_path, embeddings_dir, tokenizer_dir,
                 socket_path="/tmp/cv3_tokens.sock",
                 temperature=1.0, top_k=25, top_p=0.8, max_tokens=500, min_tokens=10):
        self.socket_path = socket_path
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

        # Load tokenizer
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        # Load RKLLM
        print("Loading RKLLM model...")
        # Import CosyVoiceLLM from pipeline
        sys.path.insert(0, os.path.dirname(model_path))
        sys.path.insert(0, '/root/cosyvoice3-rknn')
        from cosyvoice3_rknn_pipeline import CosyVoiceLLM
        self.llm = CosyVoiceLLM(model_path, embeddings_dir)
        print(f"RKLLM loaded. Temperature={temperature}, top_k={top_k}")

        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        self._running = False

    def _tokenize_text(self, text):
        """Tokenize text for CosyVoice3 LLM.

        CosyVoice3 uses raw text token IDs (no chat template).
        The prefix format is: [sos, text_emb, task_id, prompt_speech_emb]
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _nucleus_sampling(self, logits, top_p, top_k):
        """Nucleus (top-p + top-k) sampling. Returns token ID."""
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        cum_prob = 0.0
        candidates = []
        candidate_probs = []
        for idx in sorted_idx:
            if cum_prob >= top_p and len(candidates) > 0:
                break
            if len(candidates) >= top_k:
                break
            candidates.append(idx)
            candidate_probs.append(probs[idx])
            cum_prob += probs[idx]
        candidate_probs = np.array(candidate_probs)
        candidate_probs /= candidate_probs.sum()
        return int(candidates[np.random.choice(len(candidates), p=candidate_probs)])

    def _sample_with_ras(self, logits, decoded_tokens, win_size=10, tau_r=0.1):
        """Sample with RAS (Repetition Aware Sampling) + no-repeat n-gram blocking.

        From CosyVoice3 original pipeline:
        1. No-repeat n-gram: block tokens that would create repeated 10-grams
        2. Nucleus sampling (top-p + top-k)
        3. RAS: if sampled token repeated in recent window, fall back to random
        """
        # No-repeat n-gram blocking
        if decoded_tokens is not None and len(decoded_tokens) >= 10:
            n1 = 9  # 10-gram: check last 9 tokens
            prefix = decoded_tokens[-n1:]
            for j in range(len(decoded_tokens) - n1):
                if decoded_tokens[j:j + n1] == prefix:
                    blocked = decoded_tokens[j + n1]
                    if blocked < len(logits):
                        logits[blocked] = -1e9

        # Nucleus sampling (top-p + top-k)
        token = self._nucleus_sampling(logits, self.top_p, self.top_k)

        # RAS: if token appeared too often in recent window, resample randomly
        if decoded_tokens is not None and len(decoded_tokens) >= 1:
            window = decoded_tokens[-win_size:]
            rep_count = sum(1 for t in window if t == token)
            if rep_count >= max(1, int(win_size * tau_r)):
                # Random sampling from full distribution
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                token = int(np.random.choice(len(probs), p=probs))

        return token

    def _generate_streaming(self, conn, text, prompt_speech_tokens=None, prompt_text_tokens=None):
        """Generate speech tokens and stream them over the socket."""
        # Tokenize
        text_token_ids = self._tokenize_text(text)
        text_only_len = len(text_token_ids)
        if prompt_text_tokens:
            text_token_ids = prompt_text_tokens + text_token_ids

        # EOS scheduling for RKLLM (adapted from LLM.hpp:454,845):
        # RKLLM W8A8 does NOT naturally produce EOS tokens like AX650N.
        # Tuned: 3.0 tokens per text token (was 3.5) for shorter, tighter output.
        total_text_len = len(text_token_ids)  # prompt_text + new text combined
        expected_len = int(total_text_len * 3.0)
        hard_cap = int(expected_len * 3 / 2)
        hard_cap = max(hard_cap, self.min_tokens + 10)

        # Build prefix and prefill
        prefix = self.llm.build_prefix(text_token_ids, prompt_speech_tokens)
        print(f"  Prefix: {prefix.shape[0]} tokens, text_total={total_text_len}, text_new={text_only_len}, expected_len={expected_len}, hard_cap={hard_cap}")

        t0 = time.time()
        hidden = self.llm.get_hidden(prefix, keep_history=0)
        if hidden is None:
            conn.sendall(struct.pack("<i", SENTINEL_ERROR))
            return 0

        t_prefill = time.time() - t0
        print(f"  Prefill: {t_prefill:.3f}s")

        # Pre-transpose decoder weight for faster matmul (contiguous column-major)
        decoder_weight_T = self.llm.llm_decoder_weight.T.copy()  # [896, 6761] contiguous

        # Profiling accumulators
        prof_rkllm = 0.0
        prof_matmul = 0.0
        prof_sample = 0.0
        prof_send = 0.0
        prof_embed = 0.0

        inv_temp = np.float32(1.0 / max(self.temperature, 1e-6))

        # Autoregressive generation with streaming + EOS scheduling
        out_tokens = []
        max_gen = min(self.max_tokens, hard_cap)
        for i in range(max_gen):
            # --- Matmul ---
            t1 = time.time()
            logits = hidden[-1] @ decoder_weight_T
            t2 = time.time()
            prof_matmul += t2 - t1

            logits *= inv_temp

            # EOS scheduling
            ignore_eos = False
            eos_adj = 0.0
            if expected_len > 0:
                threshold_70 = int(expected_len * 0.7)
                threshold_100 = expected_len
                threshold_115 = int(expected_len * 1.15)
                if i < threshold_70:
                    ignore_eos = True
                elif i < threshold_100:
                    p = (i - threshold_70) / max(threshold_100 - threshold_70, 1)
                    eos_adj = -5.0 * (1.0 - p)
                elif i < threshold_115:
                    p = (i - threshold_100) / max(threshold_115 - threshold_100, 1)
                    eos_adj = p * 20.0
                else:
                    eos_adj = 50.0

            if ignore_eos:
                logits[SPEECH_TOKEN_SIZE:] = -1e30
            elif eos_adj != 0.0:
                logits[SPEECH_TOKEN_SIZE:] += eos_adj

            # RAS + n-gram blocking + nucleus sampling
            token = self._sample_with_ras(logits, out_tokens)
            t3 = time.time()
            prof_sample += t3 - t2

            # EOS check
            if token >= SPEECH_TOKEN_SIZE:
                if i >= self.min_tokens:
                    print(f"  EOS at token {i} (token={token})")
                    break
                # Before min_tokens: re-sample with EOS blocked
                logits[SPEECH_TOKEN_SIZE:] = -1e30
                for _ in range(50):
                    token = self._sample_with_ras(logits, out_tokens)
                    if token < SPEECH_TOKEN_SIZE:
                        break
                if token >= SPEECH_TOKEN_SIZE:
                    print(f"  Forced EOS at token {i} (before min_tokens)")
                    break

            out_tokens.append(token)

            # Stream token to client immediately
            t4 = time.time()
            try:
                conn.sendall(struct.pack("<i", token))
            except (BrokenPipeError, ConnectionResetError):
                print("  Client disconnected")
                return len(out_tokens)
            t5 = time.time()
            prof_send += t5 - t4

            # Feed back for next token
            t6 = time.time()
            next_emb = self.llm.speech_embedding[token].reshape(1, HIDDEN_SIZE)
            t7 = time.time()
            prof_embed += t7 - t6

            hidden = self.llm.get_hidden(next_emb, keep_history=1)
            t8 = time.time()
            prof_rkllm += t8 - t7

            if hidden is None:
                break

        if i >= max_gen - 1:
            print(f"  Hard cap reached at {max_gen} tokens")

        # Send done sentinel
        try:
            conn.sendall(struct.pack("<i", SENTINEL_DONE))
        except (BrokenPipeError, ConnectionResetError):
            pass

        elapsed = time.time() - t0
        n_tok = len(out_tokens)
        tok_s = n_tok / elapsed if elapsed > 0 else 0

        # Print profiling breakdown
        print(f"  Generated {n_tok} tokens in {elapsed:.2f}s ({tok_s:.1f} tok/s)")
        print(f"  === Per-token timing breakdown ===")
        if n_tok > 0:
            print(f"    RKLLM NPU:   {prof_rkllm:.3f}s total, {prof_rkllm/n_tok*1000:.1f}ms/tok ({prof_rkllm/elapsed*100:.1f}%)")
            print(f"    Matmul:      {prof_matmul:.3f}s total, {prof_matmul/n_tok*1000:.1f}ms/tok ({prof_matmul/elapsed*100:.1f}%)")
            print(f"    Sampling:    {prof_sample:.3f}s total, {prof_sample/n_tok*1000:.1f}ms/tok ({prof_sample/elapsed*100:.1f}%)")
            print(f"    Embed:       {prof_embed:.3f}s total, {prof_embed/n_tok*1000:.1f}ms/tok ({prof_embed/elapsed*100:.1f}%)")
            print(f"    Socket:      {prof_send:.3f}s total, {prof_send/n_tok*1000:.1f}ms/tok ({prof_send/elapsed*100:.1f}%)")
            print(f"    Prefill:     {t_prefill:.3f}s")
            other = elapsed - prof_rkllm - prof_matmul - prof_sample - prof_send - prof_embed - t_prefill
            print(f"    Other/overhead: {other:.3f}s ({other/elapsed*100:.1f}%)")
        return len(out_tokens)

    def serve(self):
        """Main server loop — listen for connections and process requests."""
        # Remove stale socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)
        sock.settimeout(1.0)  # For clean shutdown
        os.chmod(self.socket_path, 0o666)

        print(f"\nRKLLM Token Server listening on {self.socket_path}")
        print("Waiting for connections...\n")

        request_count = 0
        while self._running:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue

            request_count += 1
            print(f"--- Request #{request_count} ---")

            try:
                # Read message length
                raw_len = conn.recv(4)
                if len(raw_len) < 4:
                    print("  Invalid message (no length)")
                    conn.close()
                    continue

                msg_len = struct.unpack("<I", raw_len)[0]
                if msg_len > 65536:
                    print(f"  Message too large: {msg_len}")
                    conn.sendall(struct.pack("<i", SENTINEL_ERROR))
                    conn.close()
                    continue

                # Read message
                raw_msg = b""
                while len(raw_msg) < msg_len:
                    chunk = conn.recv(msg_len - len(raw_msg))
                    if not chunk:
                        break
                    raw_msg += chunk

                msg = json.loads(raw_msg.decode())
                text = msg.get("text", "")
                prompt_speech_tokens = msg.get("prompt_speech_tokens")
                prompt_text_tokens = msg.get("prompt_text_tokens")

                print(f"  Text: '{text[:60]}{'...' if len(text)>60 else ''}'")

                self._generate_streaming(conn, text, prompt_speech_tokens, prompt_text_tokens)

            except Exception as e:
                print(f"  Error: {e}")
                try:
                    conn.sendall(struct.pack("<i", SENTINEL_ERROR))
                except:
                    pass
            finally:
                conn.close()

        sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        print("Server stopped.")
        self.llm.destroy()


def main():
    parser = argparse.ArgumentParser(description="RKLLM Token Server for dual-NPU CosyVoice3")
    parser.add_argument("--model", required=True, help="Path to .rkllm model")
    parser.add_argument("--embeddings", required=True, help="Dir with embedding .npy files")
    parser.add_argument("--tokenizer", required=True, help="Dir with Qwen2 tokenizer")
    parser.add_argument("--socket", default="/tmp/cv3_tokens.sock", help="Unix socket path")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default 1.0, same as CosyVoice3 original)")
    parser.add_argument("--top_k", type=int, default=25,
                        help="Top-k for nucleus sampling (default 25)")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="Top-p for nucleus sampling (default 0.8)")
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--min_tokens", type=int, default=10)
    args = parser.parse_args()

    server = RKLLMTokenServer(
        model_path=args.model,
        embeddings_dir=args.embeddings,
        tokenizer_dir=args.tokenizer,
        socket_path=args.socket,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    server.serve()


if __name__ == "__main__":
    main()
