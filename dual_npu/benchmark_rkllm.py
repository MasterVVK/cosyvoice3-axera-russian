#!/usr/bin/env python3
"""
Direct RKLLM benchmark — measures pure NPU inference speed without socket server.
Run on CM3588:
  python3 benchmark_rkllm.py --model /root/cosyvoice3-rknn/cosyvoice3_llm_rk3588.rkllm \
      --embeddings /root/cosyvoice3-rknn/cosyvoice3_embeddings \
      --tokenizer /root/cosyvoice3-rknn/cosyvoice3_qwen2_for_rkllm
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, '/root/cosyvoice3-rknn')
from cosyvoice3_rknn_pipeline import CosyVoiceLLM, HIDDEN_SIZE, SPEECH_TOKEN_SIZE

def benchmark(llm, text_token_ids, n_tokens=50, label=""):
    """Generate n_tokens and measure timing breakdown."""
    prefix = llm.build_prefix(text_token_ids)
    print(f"\n=== Benchmark: {label} ===")
    print(f"  Prefix: {prefix.shape[0]} tokens, generating {n_tokens} tokens")

    # Prefill
    t0 = time.time()
    hidden = llm.get_hidden(prefix, keep_history=0)
    t_prefill = time.time() - t0
    print(f"  Prefill: {t_prefill:.3f}s ({prefix.shape[0]} tokens)")

    if hidden is None:
        print("  ERROR: No hidden states from prefill")
        return

    # Pre-transpose decoder weight
    decoder_weight_T = llm.llm_decoder_weight.T.copy()

    # Accumulators
    prof_rkllm = 0.0
    prof_matmul_f32 = 0.0
    prof_matmul_f64 = 0.0
    prof_sample = 0.0
    prof_embed = 0.0
    prof_ctypes = 0.0

    out_tokens = []
    t_gen_start = time.time()

    for i in range(n_tokens):
        h = hidden[-1]

        # Matmul (float32)
        t1 = time.time()
        logits_f32 = h @ decoder_weight_T
        t2 = time.time()
        prof_matmul_f32 += t2 - t1

        # Matmul (float64 — current code path)
        t1b = time.time()
        logits_f64 = (h @ decoder_weight_T).astype(np.float64)
        t2b = time.time()
        prof_matmul_f64 += t2b - t1b

        # Sampling (simple top-k for benchmark)
        t3 = time.time()
        logits_f32 /= 0.6  # temperature
        logits_f32[:SPEECH_TOKEN_SIZE]  # just speech tokens
        top_k_indices = np.argpartition(logits_f32, -15)[-15:]
        top_k_logits = logits_f32[top_k_indices].astype(np.float64)
        top_k_logits -= np.max(top_k_logits)
        probs = np.exp(top_k_logits)
        probs /= probs.sum()
        idx = np.random.choice(len(top_k_indices), p=probs)
        token = int(top_k_indices[idx])
        if token >= SPEECH_TOKEN_SIZE:
            token = np.random.randint(0, SPEECH_TOKEN_SIZE)
        t4 = time.time()
        prof_sample += t4 - t3

        out_tokens.append(token)

        # Embed lookup
        t5 = time.time()
        next_emb = llm.speech_embedding[token].reshape(1, HIDDEN_SIZE)
        t6 = time.time()
        prof_embed += t6 - t5

        # RKLLM NPU inference
        t7 = time.time()
        hidden = llm.get_hidden(next_emb, keep_history=1)
        t8 = time.time()
        prof_rkllm += t8 - t7

        if hidden is None:
            print(f"  ERROR: get_hidden returned None at token {i}")
            break

    t_gen_end = time.time()
    elapsed = t_gen_end - t_gen_start
    n = len(out_tokens)
    tok_s = n / elapsed if elapsed > 0 else 0

    print(f"\n  Results: {n} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")
    print(f"  === Per-token breakdown ===")
    if n > 0:
        print(f"    RKLLM NPU:    {prof_rkllm:.3f}s total, {prof_rkllm/n*1000:.1f}ms/tok ({prof_rkllm/elapsed*100:.1f}%)")
        print(f"    Matmul(f32):  {prof_matmul_f32:.3f}s total, {prof_matmul_f32/n*1000:.1f}ms/tok ({prof_matmul_f32/elapsed*100:.1f}%)")
        print(f"    Matmul(f64):  {prof_matmul_f64:.3f}s total, {prof_matmul_f64/n*1000:.1f}ms/tok ({prof_matmul_f64/elapsed*100:.1f}%)")
        print(f"    Sampling:     {prof_sample:.3f}s total, {prof_sample/n*1000:.1f}ms/tok ({prof_sample/elapsed*100:.1f}%)")
        print(f"    Embed:        {prof_embed:.3f}s total, {prof_embed/n*1000:.1f}ms/tok ({prof_embed/elapsed*100:.1f}%)")
        other = elapsed - prof_rkllm - prof_matmul_f64 - prof_sample - prof_embed
        print(f"    Other:        {other:.3f}s ({other/elapsed*100:.1f}%)")
        print(f"    Prefill:      {t_prefill:.3f}s")

    return tok_s


def main():
    parser = argparse.ArgumentParser(description="Direct RKLLM benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--n_tokens", type=int, default=50)
    args = parser.parse_args()

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print("Loading RKLLM...")
    llm = CosyVoiceLLM(args.model, args.embeddings)

    # Test phrases
    texts = [
        "Привет",
        "Привет, как дела? Сегодня хорошая погода для прогулки.",
    ]

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        benchmark(llm, token_ids, n_tokens=args.n_tokens, label=text[:40])

    # Warmup-free second run
    print("\n\n=== Second run (warm cache) ===")
    token_ids = tokenizer.encode(texts[0], add_special_tokens=False)
    benchmark(llm, token_ids, n_tokens=args.n_tokens, label="Warm: " + texts[0][:30])

    llm.destroy()


if __name__ == "__main__":
    main()
