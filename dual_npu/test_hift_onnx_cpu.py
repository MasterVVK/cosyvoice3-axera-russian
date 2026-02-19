#!/usr/bin/env python3
"""ONNX HiFT on CPU — quality reference and alternative to NPU HiFT.

Reads mel dumps from C++ pipeline (MEL_DUMP_DIR), runs HiFT P1+P2 via ONNX Runtime,
applies crossfade between chunks (same logic as C++ Token2wav), saves clean audio.

Usage:
    # Step 1: Run TTS with mel dump
    MEL_DUMP_DIR=/tmp/mel_dump bash launch_dual_npu.sh "text"

    # Step 2: Post-process with ONNX HiFT
    python3 test_hift_onnx_cpu.py --mel_dir /tmp/mel_dump --output /tmp/clean.wav
"""

import argparse
import struct
import time
import numpy as np
import os
import wave


SOURCE_CACHE_LEN = 8 * 480  # 3840 samples
WINDOW_SIZE = 2 * 8 * 480   # 7680


def load_mel_chunk(path):
    """Load mel chunk from binary dump."""
    with open(path, 'rb') as f:
        mel_len = struct.unpack('<i', f.read(4))[0]
        mel_data = np.frombuffer(f.read(80 * mel_len * 4), dtype=np.float32).copy()
        is_first = struct.unpack('<?', f.read(1))[0]
        # Extended format: finalize + neg_offset
        extra = f.read(5)
        if len(extra) == 5:
            finalize = struct.unpack('<?', extra[0:1])[0]
            neg_offset = struct.unpack('<i', extra[1:5])[0]
        else:
            finalize = False
            neg_offset = 0
    mel = mel_data.reshape(80, mel_len)
    return mel, mel_len, is_first, finalize, neg_offset


def make_hamming_window():
    """np.hamming(2 * 8 * 480) — same as C++ speech_window."""
    return np.hamming(WINDOW_SIZE).astype(np.float32)


def crossfade(speech_in, speech_cache, window):
    """Apply crossfade: fade_in_out from C++ Token2wav."""
    overlap = SOURCE_CACHE_LEN  # 3840
    # speech_in[:overlap] = speech_in[:overlap] * window[:overlap] + speech_cache[-overlap:] * window[overlap:]
    speech_in[:overlap] = speech_in[:overlap] * window[:overlap] + speech_cache[-overlap:] * window[overlap:]
    return speech_in


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mel_dir', default='/tmp/mel_dump')
    parser.add_argument('--onnx_dir', default='/root/cosyvoice3-rknn')
    parser.add_argument('--output', default='/tmp/hift_onnx_output.wav')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1

    print("Loading ONNX HiFT models...")
    models = {}
    for name in ['hift_p1_50_first', 'hift_p2_50_first', 'hift_p1_58', 'hift_p2_58']:
        path = os.path.join(args.onnx_dir, f'{name}.onnx')
        models[name] = ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
    print("  Loaded.")

    window = make_hamming_window()

    chunks = sorted([f for f in os.listdir(args.mel_dir) if f.startswith('mel_chunk_') and f.endswith('.bin')])
    print(f"Found {len(chunks)} mel chunks")

    all_output = []
    speech_cache = None
    total_hift_time = 0.0

    for i, chunk_file in enumerate(chunks):
        mel, mel_len, is_first, finalize, neg_offset = load_mel_chunk(os.path.join(args.mel_dir, chunk_file))
        is_last = (i == len(chunks) - 1)
        if is_last:
            finalize = True

        print(f"\n{chunk_file}: mel_len={mel_len}, is_first={is_first}, finalize={finalize}, neg_offset={neg_offset}")
        print(f"  mel: min={mel.min():.3f}, max={mel.max():.3f}, mean={mel.mean():.3f}")

        # Select models
        if mel_len == 50 and is_first:
            p1, p2 = models['hift_p1_50_first'], models['hift_p2_50_first']
        elif mel_len == 58 and not is_first:
            p1, p2 = models['hift_p1_58'], models['hift_p2_58']
        else:
            print(f"  SKIP: unsupported mel_len={mel_len}, is_first={is_first}")
            continue

        mel_input = mel.reshape(1, 80, mel_len).astype(np.float32)

        # P1: mel -> source
        t0 = time.time()
        s = p1.run(None, {'mel': mel_input})[0]
        t1 = time.time()

        # P2: mel + s -> audio
        audio = p2.run(None, {'mel': mel_input, 's': s})[0]
        t2 = time.time()
        total_hift_time += t2 - t0

        speech = audio.flatten().copy()
        print(f"  P1: {(t1-t0)*1000:.1f}ms, P2: {(t2-t1)*1000:.1f}ms")
        print(f"  audio: {len(speech)} samples, min={speech.min():.4f}, max={speech.max():.4f}")

        # Apply crossfade logic (same as C++ Token2wav::infer)
        if not finalize:
            if not is_first and speech_cache is not None:
                speech = crossfade(speech, speech_cache, window)

            # Cache tail for next crossfade
            cache_len = min(len(speech), SOURCE_CACHE_LEN)
            speech_cache = speech[-cache_len:].copy()
            # Output = speech without cached tail
            output = speech[:-cache_len]
            all_output.append(output)
            print(f"  output: {len(output)} samples, cached: {cache_len}")
        else:
            if not is_first and speech_cache is not None:
                if len(speech) >= SOURCE_CACHE_LEN:
                    if neg_offset != 0 and -neg_offset * 480 >= SOURCE_CACHE_LEN:
                        # Trim from end based on neg_offset
                        speech = speech[neg_offset * 480:]
                        speech = crossfade(speech, speech_cache, window)
                    else:
                        speech = speech[-SOURCE_CACHE_LEN:]
                        speech = crossfade(speech, speech_cache, window)
                        if neg_offset != 0:
                            trim = len(speech) + neg_offset * 480 - (len(audio.flatten()) - SOURCE_CACHE_LEN)
                            if trim > 0 and trim < len(speech):
                                speech = speech[trim:]
                else:
                    pass  # Short final chunk, no crossfade needed

            all_output.append(speech)
            print(f"  output (final): {len(speech)} samples")

    if all_output:
        full_audio = np.concatenate(all_output)
        duration = len(full_audio) / 24000
        print(f"\nTotal ONNX HiFT: {total_hift_time*1000:.0f}ms")
        print(f"Audio: {len(full_audio)} samples ({duration:.2f}s at 24000 Hz)")

        audio_int16 = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(args.output, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_int16.tobytes())
        print(f"Saved: {args.output}")
    else:
        print("No audio generated!")


if __name__ == '__main__':
    main()
