#!/usr/bin/env python3
"""
Post-process CosyVoice3 NPU output.

Applies:
1. Butterworth low-pass filter (8th order) to remove HF artifacts from NPU HiFT quantization
2. Fade-out on the last 50ms to smooth the buffer phrase
3. Silence padding at the end

No torchaudio dependency — reads/writes IEEE float WAV (format tag 3) directly via struct.

Usage:
    python3 postprocess.py input.wav output.wav [cutoff_hz] [pad_ms]

Default: cutoff=5000 Hz, padding=300 ms
"""
import sys
import struct
import numpy as np
from scipy.signal import butter, sosfilt


def read_wav_float(path):
    """Read IEEE float32 WAV file (format tag 3) using struct."""
    with open(path, 'rb') as f:
        riff = f.read(4)
        assert riff == b'RIFF', f'Not a WAV file: {riff}'
        f.read(4)  # file size
        wave = f.read(4)
        assert wave == b'WAVE'

        sr = 24000
        data = None

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                fmt_tag = struct.unpack('<H', fmt_data[0:2])[0]
                n_channels = struct.unpack('<H', fmt_data[2:4])[0]
                sr = struct.unpack('<I', fmt_data[4:8])[0]
            elif chunk_id == b'data':
                raw = f.read(chunk_size)
                data = np.frombuffer(raw, dtype=np.float32)
            else:
                f.read(chunk_size)

    return data, sr


def write_wav_float(path, data, sr):
    """Write IEEE float32 WAV file (format tag 3)."""
    n_samples = len(data)
    data_bytes = data.astype(np.float32).tobytes()
    data_size = len(data_bytes)

    with open(path, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        # fmt chunk (IEEE float = format tag 3)
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<HHIIHH', 3, 1, sr, sr * 4, 4, 32))
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(data_bytes)


def postprocess(input_path, output_path, cutoff_hz=5000, pad_ms=300):
    """Apply LP filter, fade-out, and padding to NPU output."""
    data, sr = read_wav_float(input_path)

    # Butterworth low-pass filter (8th order)
    # Removes broadband HF artifacts (4-6 kHz) from NPU HiFT quantization
    sos = butter(8, cutoff_hz, btype='low', fs=sr, output='sos')
    filtered = sosfilt(sos, data).astype(np.float32)

    # Fade-out last 50ms (smooths the buffer phrase ending)
    fade_len = int(sr * 0.05)
    if len(filtered) > fade_len:
        filtered[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)

    # Add padding silence
    pad_samples = int(sr * pad_ms / 1000)
    result = np.concatenate([filtered, np.zeros(pad_samples, dtype=np.float32)])

    write_wav_float(output_path, result, sr)
    dur = len(result) / sr
    print(f'{input_path} -> {output_path} ({dur:.2f}s, lp={cutoff_hz}Hz, pad={pad_ms}ms)')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 postprocess.py input.wav output.wav [cutoff_hz] [pad_ms]')
        sys.exit(1)
    cutoff = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    pad = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    postprocess(sys.argv[1], sys.argv[2], cutoff, pad)
