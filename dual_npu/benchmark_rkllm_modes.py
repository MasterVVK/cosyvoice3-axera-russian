#!/usr/bin/env python3
"""
RKLLM benchmark: test different modes and settings.
Tests: use_gpu on/off, embed_flash on/off, npu core count, logits mode.
"""

import os
import sys
import time
import ctypes
import argparse
import numpy as np

sys.path.insert(0, '/root/cosyvoice3-rknn')
from cosyvoice3_rknn_pipeline import (
    CosyVoiceLLM, HIDDEN_SIZE, SPEECH_TOKEN_SIZE,
    RKLLMParam, RKLLMInput, RKLLMInferParam, RKLLMEmbedInput,
    RKLLM_Handle_t, RKLLM_INPUT_EMBED, RKLLM_RUN_NORMAL,
    callback_type, RKLLMResult, RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
)


def benchmark_hidden_mode(llm, prefix, n_tokens=20, label=""):
    """Benchmark get_hidden mode (current approach)."""
    hidden = llm.get_hidden(prefix, keep_history=0)
    if hidden is None:
        print(f"  {label}: FAILED (no hidden from prefill)")
        return

    times = []
    out_tokens = []
    for i in range(n_tokens):
        t0 = time.time()
        # Token selection (fast path, no matmul for pure NPU benchmark)
        token = np.random.randint(0, SPEECH_TOKEN_SIZE)
        out_tokens.append(token)
        next_emb = llm.speech_embedding[token].reshape(1, HIDDEN_SIZE)
        hidden = llm.get_hidden(next_emb, keep_history=1)
        elapsed = time.time() - t0
        times.append(elapsed)
        if hidden is None:
            break

    times = np.array(times)
    print(f"  {label}: {n_tokens} tokens, mean={times.mean()*1000:.1f}ms/tok, "
          f"min={times.min()*1000:.1f}ms, max={times.max()*1000:.1f}ms, "
          f"std={times.std()*1000:.1f}ms, "
          f"tok/s={1.0/times.mean():.1f}")


def test_logits_mode(llm, prefix):
    """Test if RKLLM supports logits mode (mode=0 with result.logits)."""
    print("\n=== Testing logits mode ===")

    # First, prefill with hidden mode
    hidden = llm.get_hidden(prefix, keep_history=0)
    if hidden is None:
        print("  FAILED: no hidden from prefill")
        return

    # Now try mode=0 (normal) to see if we get logits
    logits_result = [None]
    token_result = [None]

    def logits_callback(result_ptr, userdata, state):
        if state == RKLLM_RUN_NORMAL:
            r = result_ptr.contents
            token_result[0] = r.token_id
            if r.logits.logits and r.logits.vocab_size > 0:
                n = r.logits.num_tokens
                v = r.logits.vocab_size
                arr = np.ctypeslib.as_array(r.logits.logits, shape=(n * v,)).copy()
                logits_result[0] = arr.reshape(n, v)
                print(f"    Got logits: shape=({n}, {v})")
            if r.last_hidden_layer.hidden_states and r.last_hidden_layer.embd_size > 0:
                n = r.last_hidden_layer.num_tokens
                d = r.last_hidden_layer.embd_size
                print(f"    Got hidden: shape=({n}, {d})")

    c_callback = callback_type(logits_callback)

    # Try mode=0 (normal inference)
    token = np.random.randint(0, SPEECH_TOKEN_SIZE)
    next_emb = llm.speech_embedding[token].reshape(1, HIDDEN_SIZE).astype(np.float32)
    flat = next_emb.flatten()
    c_embed = (ctypes.c_float * len(flat))(*flat)

    rk_input = RKLLMInput()
    rk_input.role = None
    rk_input.enable_thinking = False
    rk_input.input_type = RKLLM_INPUT_EMBED
    rk_input.input_data.embed_input.embed = c_embed
    rk_input.input_data.embed_input.n_tokens = 1

    # Mode 0 = normal
    infer_params = RKLLMInferParam()
    ctypes.memset(ctypes.byref(infer_params), 0, ctypes.sizeof(RKLLMInferParam))
    infer_params.mode = 0  # Normal mode
    infer_params.keep_history = 1

    t0 = time.time()
    ret = llm.lib.rkllm_run(llm.handle, ctypes.byref(rk_input), ctypes.byref(infer_params), None)
    elapsed = time.time() - t0

    print(f"  mode=0: ret={ret}, token_id={token_result[0]}, elapsed={elapsed*1000:.1f}ms")
    if logits_result[0] is not None:
        print(f"  Logits available! Shape: {logits_result[0].shape}")
    else:
        print(f"  No logits returned in mode=0")


def test_gpu_impact(model_path, embeddings_dir, prefix_token_ids, tokenizer):
    """Test use_gpu=True vs use_gpu=False."""
    from cosyvoice3_rknn_pipeline import CosyVoiceLLM

    prefix_ids = tokenizer.encode("Привет", add_special_tokens=False)

    print("\n=== Test use_gpu=True (default) ===")
    llm1 = CosyVoiceLLM(model_path, embeddings_dir)
    prefix1 = llm1.build_prefix(prefix_ids)
    benchmark_hidden_mode(llm1, prefix1, n_tokens=20, label="use_gpu=True")
    llm1.destroy()

    # Modify CosyVoiceLLM to not use GPU
    print("\n=== Test use_gpu=False ===")
    # We need to create a modified version
    import importlib
    import cosyvoice3_rknn_pipeline as pipeline
    original_init = pipeline.CosyVoiceLLM.__init__

    def patched_init(self, model_path, embeddings_dir, lib_path=None, max_context=512):
        # Same as original but with use_gpu=False
        self.embed_tokens = np.load(os.path.join(embeddings_dir, "embed_tokens.npy")).astype(np.float32)
        self.speech_embedding = np.load(os.path.join(embeddings_dir, "speech_embedding.npy")).astype(np.float32)
        self.llm_decoder_weight = np.load(os.path.join(embeddings_dir, "llm_decoder_weight.npy")).astype(np.float32)

        lib_path = lib_path or self._find_lib()
        self.lib = ctypes.CDLL(lib_path)
        self._setup()
        self._last_hidden = None
        self._c_callback = callback_type(self._callback)

        param = self.lib.rkllm_createDefaultParam()
        param.model_path = model_path.encode()
        param.max_context_len = max_context
        param.max_new_tokens = 1
        param.top_k = ctypes.c_float(25.0)
        param.top_p = ctypes.c_float(0.9)
        param.temperature = ctypes.c_float(1.0)
        param.repeat_penalty = ctypes.c_float(1.0)
        param.skip_special_token = False
        param.is_async = False
        param.use_gpu = False  # <-- CHANGED
        param.extend_param.base_domain_id = 1
        param.extend_param.embed_flash = 1

        self.handle = RKLLM_Handle_t()
        ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), self._c_callback)
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed: {ret}")

    pipeline.CosyVoiceLLM.__init__ = patched_init
    llm2 = pipeline.CosyVoiceLLM(model_path, embeddings_dir)
    prefix2 = llm2.build_prefix(prefix_ids)
    benchmark_hidden_mode(llm2, prefix2, n_tokens=20, label="use_gpu=False")
    llm2.destroy()

    # Restore
    pipeline.CosyVoiceLLM.__init__ = original_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    prefix_ids = tokenizer.encode("Привет", add_special_tokens=False)

    # Test 1: Current hidden mode
    print("=== Current hidden mode benchmark ===")
    llm = CosyVoiceLLM(args.model, args.embeddings)
    prefix = llm.build_prefix(prefix_ids)

    # Warmup
    benchmark_hidden_mode(llm, prefix, n_tokens=5, label="Warmup")
    # Real
    prefix = llm.build_prefix(prefix_ids)
    benchmark_hidden_mode(llm, prefix, n_tokens=30, label="Hidden mode (30 tok)")

    # Test 2: Try logits mode
    prefix = llm.build_prefix(prefix_ids)
    test_logits_mode(llm, prefix)

    llm.destroy()

    # Test 3: GPU impact
    test_gpu_impact(args.model, args.embeddings, prefix_ids, tokenizer)


if __name__ == "__main__":
    main()
