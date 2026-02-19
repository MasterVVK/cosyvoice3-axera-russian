#!/usr/bin/env python3
"""ONNX HiFT Server — replaces NPU HiFT with CPU ONNX Runtime.

Listens on Unix socket, receives mel data from C++ cosyvoice3_tts binary,
runs HiFT via ONNX Runtime on CPU, returns audio samples.

Two modes:
  --dynamic  (default): Uses full HiFT ONNX model (mel→audio in single pass).
                        Accepts ANY mel_len. Works in both chunked and fullmel modes.
  --legacy:            Uses fixed-shape P1/P2 models (mel_len=50 or 58 only).
                        For chunked mode with crossfade.

Protocol:
  Client → Server: [4 bytes mel_len (int32)] [4 bytes is_first (int32: 0/1)]
                    [mel_len*80*4 bytes mel_data (float32)]
  Server → Client: [4 bytes audio_len (int32)] [audio_len*4 bytes audio_data (float32)]
                   [4 bytes -1 (int32)] on error

Usage:
    taskset -c 4-7 python3 onnx_hift_server.py \
        --components_dir /root/cosyvoice3-rknn/hift-components \
        --socket /tmp/cv3_hift.sock \
        --threads 4
"""

import os
import sys
import socket
import struct
import argparse
import time
import signal
import json
import numpy as np


class HiFTDynamic:
    """Full HiFT vocoder — single ONNX model, dynamic mel_len.
    mel [1, 80, N] → audio [1, T]. No numpy SineGen2/ISTFT needed."""

    def __init__(self, components_dir, n_threads=4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1

        model_path = os.path.join(components_dir, "hift_full_dynamic.onnx")
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=['CPUExecutionProvider'])

        print(f"  HiFT full dynamic model loaded: {model_path}")
        print(f"    Inputs: {[i.name for i in self.session.get_inputs()]}")
        print(f"    Outputs: {[o.name for o in self.session.get_outputs()]}")

    def inference(self, mel_data, mel_len):
        """mel_data: flat float32 array [mel_len*80] → audio float32 array"""
        mel_input = mel_data.reshape(1, 80, mel_len).astype(np.float32)

        t0 = time.time()
        audio = self.session.run(None, {"mel": mel_input})[0]  # [1, audio_len]
        t1 = time.time()

        audio = audio.flatten()
        raw_peak = np.max(np.abs(audio))

        # Gain scaling (model already clips to ±0.99)
        audio *= 0.85
        peak = np.max(np.abs(audio))

        print(f"  HiFT full: {1000*(t1-t0):.0f}ms mel_len={mel_len} "
              f"audio={len(audio)} ({len(audio)/24000:.2f}s) "
              f"raw_peak={raw_peak:.3f} peak={peak:.3f}")
        return audio


class OnnxHiftServer:
    def __init__(self, args):
        self.socket_path = args.socket
        self._running = True
        self.dynamic = not args.legacy

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        opts.inter_op_num_threads = 1

        if self.dynamic:
            # Full HiFT model — single ONNX, any mel_len
            model_path = os.path.join(args.components_dir, "hift_full_dynamic.onnx")
            print(f"Loading HiFT full dynamic model: {model_path}")
            self.full_session = ort.InferenceSession(
                model_path, sess_options=opts, providers=['CPUExecutionProvider'])
            print("  Full dynamic model loaded.")
        else:
            print("Loading HiFT legacy P1/P2 models...")
            self.models = {}
            for name in ['hift_p1_50_first', 'hift_p2_50_first', 'hift_p1_58', 'hift_p2_58']:
                path = os.path.join(args.onnx_dir, f'{name}.onnx')
                self.models[name] = ort.InferenceSession(
                    path, sess_options=opts, providers=['CPUExecutionProvider'])
            print("  Legacy models loaded.")

    def _signal_handler(self, signum, frame):
        print(f"\nSignal {signum}, shutting down...")
        self._running = False

    def _run_hift_legacy(self, mel_data, mel_len, is_first):
        """Legacy: P1+P2 fixed-shape models."""
        mel = mel_data.reshape(1, 80, mel_len)
        if mel_len == 50 and is_first:
            p1, p2 = self.models['hift_p1_50_first'], self.models['hift_p2_50_first']
        elif mel_len == 58 and not is_first:
            p1, p2 = self.models['hift_p1_58'], self.models['hift_p2_58']
        else:
            raise ValueError(f"Unsupported mel_len={mel_len}, is_first={is_first}")

        t0 = time.time()
        s = p1.run(None, {'mel': mel})[0]
        audio = p2.run(None, {'mel': mel, 's': s})[0]
        t1 = time.time()

        audio_flat = audio.flatten()
        raw_peak = np.max(np.abs(audio_flat))
        audio_flat *= 0.85

        peak = np.max(np.abs(audio_flat))
        print(f"  HiFT legacy: {1000*(t1-t0):.0f}ms audio={len(audio_flat)} raw_peak={raw_peak:.3f} peak={peak:.3f}")
        return audio_flat

    @staticmethod
    def _despike(audio, threshold=0.4):
        """Remove isolated spikes: samples where delta is large on BOTH sides.
        Only true spikes (not normal speech transitions) are affected."""
        n_fixed = 0
        for i in range(2, len(audio) - 2):
            d_before = abs(audio[i] - audio[i-1])
            d_after = abs(audio[i] - audio[i+1])
            if d_before > threshold and d_after > threshold:
                audio[i] = 0.5 * (audio[i-1] + audio[i+1])
                n_fixed += 1
        return n_fixed

    def _run_hift_full(self, mel_data, mel_len, is_first):
        """Full dynamic model: single ONNX, any mel_len."""
        mel = mel_data.reshape(1, 80, mel_len).astype(np.float32)

        # Debug: dump mel and raw audio for offline analysis
        dump_dir = os.environ.get("MEL_DUMP_DIR")
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            mel_path = os.path.join(dump_dir, f"mel_{mel_len}.npy")
            np.save(mel_path, mel)
            print(f"  [DUMP] mel saved: {mel_path} shape={mel.shape} "
                  f"min={mel.min():.4f} max={mel.max():.4f} mean={mel.mean():.4f}")

        t0 = time.time()
        audio = self.full_session.run(None, {"mel": mel})[0]
        t1 = time.time()

        if dump_dir:
            audio_path = os.path.join(dump_dir, f"audio_raw_{mel_len}.npy")
            np.save(audio_path, audio)
            print(f"  [DUMP] raw audio saved: {audio_path}")

        audio_flat = audio.flatten()
        raw_peak = np.max(np.abs(audio_flat))

        # Normalize to 0.85 peak
        if raw_peak > 0.01:
            gain = min(0.85 / raw_peak, 2.0)
        else:
            gain = 1.0
        audio_flat *= gain

        # Tanh soft limiter (smooth peaks >0.7)
        knee = 0.7
        mask = np.abs(audio_flat) > knee
        if np.any(mask):
            s = np.sign(audio_flat[mask])
            x = np.abs(audio_flat[mask])
            audio_flat[mask] = s * (knee + (1.0 - knee) * np.tanh((x - knee) / (1.0 - knee)))

        peak = np.max(np.abs(audio_flat))
        elapsed = 1000 * (t1 - t0)
        print(f"  HiFT full: {elapsed:.0f}ms mel_len={mel_len} "
              f"audio={len(audio_flat)} ({len(audio_flat)/24000:.2f}s) "
              f"raw_peak={raw_peak:.3f} gain={gain:.2f} peak={peak:.3f}")
        return audio_flat

    def _recv_exact(self, conn, n):
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data

    def serve(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)
        sock.settimeout(1.0)
        os.chmod(self.socket_path, 0o666)

        mode = "dynamic" if self.dynamic else "legacy"
        print(f"\nONNX HiFT Server ({mode}) listening on {self.socket_path}")
        print("Waiting for connections...\n")

        request_count = 0
        while self._running:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue

            request_count += 1

            try:
                header = self._recv_exact(conn, 8)
                mel_len = struct.unpack('<i', header[0:4])[0]
                is_first = struct.unpack('<i', header[4:8])[0] != 0

                mel_bytes = self._recv_exact(conn, mel_len * 80 * 4)
                mel_data = np.frombuffer(mel_bytes, dtype=np.float32).copy()

                print(f"--- HiFT #{request_count}: mel_len={mel_len}, is_first={is_first} ---")

                if self.dynamic:
                    audio = self._run_hift_full(mel_data, mel_len, is_first)
                else:
                    audio = self._run_hift_legacy(mel_data, mel_len, is_first)

                audio_len = len(audio)
                conn.sendall(struct.pack('<i', audio_len))
                conn.sendall(audio.astype(np.float32).tobytes())

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    conn.sendall(struct.pack('<i', -1))
                except:
                    pass
            finally:
                conn.close()

        sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        print("Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="ONNX HiFT Server")
    parser.add_argument('--components_dir', default='/root/cosyvoice3-rknn/hift-components',
                        help='Directory with dynamic HiFT component models')
    parser.add_argument('--onnx_dir', default='/root/cosyvoice3-rknn',
                        help='Directory with legacy P1/P2 ONNX models')
    parser.add_argument('--socket', default='/tmp/cv3_hift.sock', help='Unix socket path')
    parser.add_argument('--threads', type=int, default=4, help='ONNX Runtime threads')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy P1/P2 fixed-shape models instead of dynamic')
    args = parser.parse_args()

    server = OnnxHiftServer(args)
    server.serve()


if __name__ == '__main__':
    main()
