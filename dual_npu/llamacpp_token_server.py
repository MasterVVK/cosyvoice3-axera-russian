#!/usr/bin/env python3
"""
llama.cpp Token Server for dual-NPU CosyVoice3 TTS.

Runs CosyVoice3 LLM on CPU (A76 cores, llama.cpp Q4_K_M) and streams speech tokens
to a Token2Wav consumer (AX650N) via Unix domain socket.

This replaces rkllm_token_server.py with 4x faster inference:
  RKLLM NPU: ~10 tok/s  →  llama.cpp CPU: ~40 tok/s

Protocol (same as rkllm_token_server.py):
  Client → Server: [4 bytes msg_len] [msg_len bytes JSON: {"text":"...", "prompt_speech_tokens":[...]}]
  Server → Client: [4 bytes int32: token_id] repeated per token
                   [4 bytes int32: -1] when done (EOS/max)
                   [4 bytes int32: -2] on error

Usage on CM3588:
  python3 llamacpp_token_server.py \
      --model /root/cosyvoice3-rknn/cosyvoice3_qwen2_q4_k_m.gguf \
      --embeddings /root/cosyvoice3-rknn/cosyvoice3_embeddings \
      --tokenizer /root/cosyvoice3-rknn/cosyvoice3_qwen2_for_rkllm \
      --socket /tmp/cv3_tokens.sock
"""

import os
import sys
import socket
import struct
import json
import argparse
import time
import signal
import hashlib
import numpy as np

# Add current dir to path for llama_cpp_bindings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_cpp_bindings import LlamaCppModel

SPEECH_TOKEN_SIZE = 6561
HIDDEN_SIZE = 896
SENTINEL_DONE = -1
SENTINEL_ERROR = -2

# Special token indices in speech_embedding (must match cosyvoice3_rknn_pipeline.py)
SOS_TOKEN = 6561          # SPEECH_TOKEN_SIZE + 0
EOS_TOKEN = 6562          # SPEECH_TOKEN_SIZE + 1
TASK_ID_TOKEN = 6563      # SPEECH_TOKEN_SIZE + 2
FILL_TOKEN = 6564         # SPEECH_TOKEN_SIZE + 3


class LlamaCppTokenServer:
    def __init__(self, model_path, embeddings_dir, tokenizer_dir,
                 socket_path="/tmp/cv3_tokens.sock",
                 temperature=1.0, top_k=25, top_p=0.8,
                 max_tokens=500, min_tokens=10, n_threads=4,
                 kv_cache_dir="/tmp"):
        self.socket_path = socket_path
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.kv_cache_dir = kv_cache_dir
        self._cached_prefix_hash = None
        self._cached_prefix_pos = 0

        # Load tokenizer
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        # Load embeddings (external numpy files)
        print("Loading embeddings...")
        self.embed_tokens = np.load(os.path.join(embeddings_dir, "embed_tokens.npy"))
        self.speech_embedding = np.load(os.path.join(embeddings_dir, "speech_embedding.npy"))
        self.llm_decoder_weight = np.load(os.path.join(embeddings_dir, "llm_decoder_weight.npy"))
        print(f"  embed_tokens: {self.embed_tokens.shape}")
        print(f"  speech_embedding: {self.speech_embedding.shape}")
        print(f"  llm_decoder_weight: {self.llm_decoder_weight.shape}")

        # Pre-transpose decoder weight for faster matmul
        self.decoder_weight_T = self.llm_decoder_weight.T.copy().astype(np.float32)

        # Verify speech_embedding shape matches expected constants
        assert self.speech_embedding.shape[0] >= TASK_ID_TOKEN + 1, \
            f"speech_embedding too small: {self.speech_embedding.shape[0]}, need >= {TASK_ID_TOKEN + 1}"

        # Load llama.cpp model
        print(f"Loading llama.cpp model: {model_path}")
        print(f"  Threads: {n_threads}")
        self.llm = LlamaCppModel(model_path, n_ctx=512, n_threads=n_threads)
        print(f"  Model loaded. n_embd={self.llm.n_embd}")

        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        self._running = False

    def _tokenize_text(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def build_prefix(self, text_token_ids, prompt_speech_tokens=None):
        """Build prefix embeddings: [SOS, text_emb, TASK_ID, prompt_speech_emb].

        Same logic as CosyVoiceLLM.build_prefix() in cosyvoice3_rknn_pipeline.py.
        """
        parts = []

        # SOS token
        parts.append(self.speech_embedding[SOS_TOKEN].reshape(1, HIDDEN_SIZE))

        # Text embeddings from Qwen2 embed_tokens
        text_emb = self.embed_tokens[text_token_ids]  # [n_text, 896]
        parts.append(text_emb)

        # TASK_ID token
        parts.append(self.speech_embedding[TASK_ID_TOKEN].reshape(1, HIDDEN_SIZE))

        # Prompt speech embeddings (for voice cloning)
        if prompt_speech_tokens is not None and len(prompt_speech_tokens) > 0:
            prompt_emb = self.speech_embedding[prompt_speech_tokens]  # [n_prompt, 896]
            parts.append(prompt_emb)

        prefix = np.concatenate(parts, axis=0).astype(np.float32)
        return prefix

    def _nucleus_sampling(self, logits, top_p, top_k):
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
        # No-repeat n-gram blocking (10-gram)
        if decoded_tokens is not None and len(decoded_tokens) >= 10:
            n1 = 9
            prefix = decoded_tokens[-n1:]
            for j in range(len(decoded_tokens) - n1):
                if decoded_tokens[j:j + n1] == prefix:
                    blocked = decoded_tokens[j + n1]
                    if blocked < len(logits):
                        logits[blocked] = -1e9

        token = self._nucleus_sampling(logits, self.top_p, self.top_k)

        # RAS: resample if token repeated in window
        if decoded_tokens is not None and len(decoded_tokens) >= 1:
            window = decoded_tokens[-win_size:]
            rep_count = sum(1 for t in window if t == token)
            if rep_count >= max(1, int(win_size * tau_r)):
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                token = int(np.random.choice(len(probs), p=probs))

        return token

    def _prefix_hash(self, prefix):
        """Compute hash of prefix embeddings for KV cache lookup."""
        return hashlib.md5(prefix.tobytes()).hexdigest()[:16]

    def _kv_cache_path(self, prefix_hash):
        return os.path.join(self.kv_cache_dir, f"cv3_kv_{prefix_hash}.bin")

    def _hidden_cache_path(self, prefix_hash):
        return os.path.join(self.kv_cache_dir, f"cv3_hidden_{prefix_hash}.npy")

    def _generate_streaming(self, conn, text, prompt_speech_tokens=None, prompt_text_tokens=None):
        text_token_ids = self._tokenize_text(text)
        text_only_len = len(text_token_ids)
        if prompt_text_tokens:
            text_token_ids = prompt_text_tokens + text_token_ids

        # EOS scheduling
        total_text_len = len(text_token_ids)
        expected_len = int(total_text_len * 3.0)
        hard_cap = int(expected_len * 3 / 2)
        hard_cap = max(hard_cap, self.min_tokens + 10)

        # Build prefix and prefill (with KV cache)
        prefix = self.build_prefix(text_token_ids, prompt_speech_tokens)
        prefix_hash = self._prefix_hash(prefix)
        kv_path = self._kv_cache_path(prefix_hash)
        hidden_path = self._hidden_cache_path(prefix_hash)
        print(f"  Prefix: {prefix.shape[0]} tokens, text_total={total_text_len}, expected_len={expected_len}, hash={prefix_hash}")

        t0 = time.time()
        cache_hit = False

        # Try KV cache hit: need both KV state file and hidden state file
        if os.path.exists(kv_path) and os.path.exists(hidden_path):
            try:
                ret = self.llm.state_load(kv_path)
                if ret == 0:
                    hidden = np.load(hidden_path)
                    self.llm.pos = prefix.shape[0]
                    cache_hit = True
                    t_prefill = time.time() - t0
                    print(f"  KV CACHE HIT: {t_prefill:.3f}s (vs ~{prefix.shape[0] / 67:.1f}s full prefill)")
                else:
                    print(f"  KV cache load failed, doing full prefill")
            except Exception as e:
                print(f"  KV cache error: {e}, doing full prefill")

        if not cache_hit:
            hidden = self.llm.get_hidden(prefix, keep_history=0)
            t_prefill = time.time() - t0
            print(f"  Prefill: {t_prefill:.3f}s ({prefix.shape[0] / t_prefill:.0f} tok/s)")

            # Save KV state + hidden for future reuse
            try:
                self.llm.state_save(kv_path)
                np.save(hidden_path, hidden)
            except Exception as e:
                print(f"  KV cache save error: {e}")

        # Profiling
        prof_llm = 0.0
        prof_matmul = 0.0
        prof_sample = 0.0
        prof_send = 0.0
        prof_embed = 0.0

        inv_temp = np.float32(1.0 / max(self.temperature, 1e-6))

        out_tokens = []
        max_gen = min(self.max_tokens, hard_cap)
        for i in range(max_gen):
            # Matmul: hidden @ decoder_weight.T -> logits
            t1 = time.time()
            logits = hidden @ self.decoder_weight_T
            t2 = time.time()
            prof_matmul += t2 - t1

            logits *= inv_temp

            # EOS scheduling (same as RKLLM version)
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

            token = self._sample_with_ras(logits, out_tokens)
            t3 = time.time()
            prof_sample += t3 - t2

            # EOS check
            if token >= SPEECH_TOKEN_SIZE:
                if i >= self.min_tokens:
                    print(f"  EOS at token {i} (token={token})")
                    break
                logits[SPEECH_TOKEN_SIZE:] = -1e30
                for _ in range(50):
                    token = self._sample_with_ras(logits, out_tokens)
                    if token < SPEECH_TOKEN_SIZE:
                        break
                if token >= SPEECH_TOKEN_SIZE:
                    print(f"  Forced EOS at token {i}")
                    break

            out_tokens.append(token)

            # Stream token
            t4 = time.time()
            try:
                conn.sendall(struct.pack("<i", token))
            except (BrokenPipeError, ConnectionResetError):
                print("  Client disconnected")
                return len(out_tokens)
            t5 = time.time()
            prof_send += t5 - t4

            # Next embedding
            t6 = time.time()
            next_emb = self.speech_embedding[token].reshape(1, HIDDEN_SIZE).astype(np.float32)
            t7 = time.time()
            prof_embed += t7 - t6

            # Forward pass (incremental, keeps KV cache)
            hidden = self.llm.get_hidden(next_emb, keep_history=1)
            t8 = time.time()
            prof_llm += t8 - t7

        if i >= max_gen - 1:
            print(f"  Hard cap reached at {max_gen} tokens")

        # Send done
        try:
            conn.sendall(struct.pack("<i", SENTINEL_DONE))
        except (BrokenPipeError, ConnectionResetError):
            pass

        elapsed = time.time() - t0
        n_tok = len(out_tokens)
        tok_s = n_tok / elapsed if elapsed > 0 else 0

        print(f"  Generated {n_tok} tokens in {elapsed:.2f}s ({tok_s:.1f} tok/s)")
        if n_tok > 0:
            print(f"  === Per-token timing ===")
            print(f"    llama.cpp:   {prof_llm:.3f}s total, {prof_llm/n_tok*1000:.1f}ms/tok ({prof_llm/elapsed*100:.1f}%)")
            print(f"    Matmul:      {prof_matmul:.3f}s total, {prof_matmul/n_tok*1000:.1f}ms/tok ({prof_matmul/elapsed*100:.1f}%)")
            print(f"    Sampling:    {prof_sample:.3f}s total, {prof_sample/n_tok*1000:.1f}ms/tok ({prof_sample/elapsed*100:.1f}%)")
            print(f"    Embed:       {prof_embed:.3f}s total, {prof_embed/n_tok*1000:.1f}ms/tok ({prof_embed/elapsed*100:.1f}%)")
            print(f"    Socket:      {prof_send:.3f}s total, {prof_send/n_tok*1000:.1f}ms/tok ({prof_send/elapsed*100:.1f}%)")
            print(f"    Prefill:     {t_prefill:.3f}s")
        return n_tok

    def serve(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)
        sock.settimeout(1.0)
        os.chmod(self.socket_path, 0o666)

        print(f"\nllama.cpp Token Server listening on {self.socket_path}")
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
                raw_len = conn.recv(4)
                if len(raw_len) < 4:
                    conn.close()
                    continue

                msg_len = struct.unpack("<I", raw_len)[0]
                if msg_len > 65536:
                    conn.sendall(struct.pack("<i", SENTINEL_ERROR))
                    conn.close()
                    continue

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
                import traceback
                traceback.print_exc()
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
    parser = argparse.ArgumentParser(description="llama.cpp Token Server for CosyVoice3")
    parser.add_argument("--model", required=True, help="Path to .gguf model")
    parser.add_argument("--embeddings", required=True, help="Dir with embedding .npy files")
    parser.add_argument("--tokenizer", required=True, help="Dir with Qwen2 tokenizer")
    parser.add_argument("--socket", default="/tmp/cv3_tokens.sock")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--min_tokens", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of CPU threads for llama.cpp (default 4 = A76 cores)")
    parser.add_argument("--kv_cache_dir", default="/tmp",
                        help="Directory for KV cache files (use fast SSD)")
    args = parser.parse_args()

    server = LlamaCppTokenServer(
        model_path=args.model,
        embeddings_dir=args.embeddings,
        tokenizer_dir=args.tokenizer,
        socket_path=args.socket,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        n_threads=args.threads,
        kv_cache_dir=args.kv_cache_dir,
    )
    server.serve()


if __name__ == "__main__":
    main()
