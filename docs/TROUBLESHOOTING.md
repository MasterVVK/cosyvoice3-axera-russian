# Troubleshooting

Common issues and their solutions when running CosyVoice3 on AX650N NPU.

## Audio Quality Issues

### High-frequency squeak / whistle (4-6 kHz)

**Symptom**: Intermittent high-pitched noise in the output audio.

**Cause**: NPU HiFT vocoder quantization introduces broadband artifacts in the 4-6 kHz range. This does NOT happen on GPU (tested on RTX 3090) — it's specific to the w8a16 NPU quantization.

**Solution**: Butterworth low-pass filter at 5 kHz (8th order). This is applied automatically by `postprocess.py`.

**What doesn't work**:
- FFT-based frequency masking — doesn't effectively remove transient artifacts
- Notch/band-stop filters (4-5.5 kHz or 3.5-6 kHz) — the artifact is broadband, not tonal
- Only full low-pass filtering removes it

If you want to adjust the cutoff:
```bash
# More aggressive filtering (may slightly muffle speech)
LP_CUTOFF=4000 ./run_tts.sh "text" output.wav 30

# Less aggressive (some squeak may remain)
LP_CUTOFF=6000 ./run_tts.sh "text" output.wav 30
```

### Last word gets cut off

**Symptom**: The final word in the output is truncated (e.g., "прогулку" → "прогул").

**Cause**: NPU LLM generates EOS token slightly earlier than the GPU version.

**Solution**: A buffer phrase "Вот так." is automatically appended to the input text. The model generates it, but the main text plays fully. The fade-out and padding mask the buffer phrase.

If you want a different buffer phrase:
```bash
BUFFER_PHRASE="Это всё." ./run_tts.sh "your text" output.wav 30
```

### Unnatural prosody / strong accent

**Symptom**: Russian speech sounds robotic or has a strong Chinese accent.

**Cause**: Missing system prefix in prompt_text.

**Solution**: Ensure prompt_text starts with:
```
You are a helpful assistant.<|endofprompt|>
```

This prefix is critical — discovered by comparing NPU output with GPU reference (which uses it internally).

## Runtime Errors

### `wave.Error: unknown format: 3`

**Cause**: Python's built-in `wave` module doesn't support IEEE float WAV files (format tag 3). The AXERA-TECH binary outputs float32 WAV.

**Solution**: Our `postprocess.py` uses `struct` for manual WAV parsing instead of the `wave` module.

### Tokenizer fails to start

**Symptom**: `curl http://127.0.0.1:12345/health` returns nothing.

**Fixes**:
```bash
# Check if transformers is installed
python3 -c "from transformers import AutoTokenizer; print('OK')"

# Check tokenizer model files
ls scripts/CosyVoice-BlankEN/
# Needs: vocab.json, merges.txt, tokenizer_config.json

# Start manually and check errors
python3 tokenizer_server.py --model_dir scripts/CosyVoice-BlankEN --port 12345
```

### NPU not detected

```bash
# Check PCIe connection
lspci | grep -i axera

# Check AXCL runtime
/usr/bin/axcl/axcl-smi

# Check kernel modules
lsmod | grep -i ax
```

### Model loading fails with "remain_cmm" errors

**Cause**: Not enough NPU memory (CMM). Another process may be using the NPU.

```bash
# Check what's using NPU
/usr/bin/axcl/axcl-smi

# Kill other NPU processes if needed
```

### scipy installation fails on aarch64

```bash
# On CM3588 / Debian-based aarch64
pip3 install scipy --break-system-packages

# If numpy version conflict
pip3 install --upgrade numpy scipy --break-system-packages
```

## Performance Issues

### Very slow decode (< 3 tok/s)

- Check PCIe link speed: should be Gen3 x1 (8 GT/s)
- Check thermal throttling: `axcl-smi` should show < 85°C
- Try fewer ODE steps: `./run_tts.sh "text" out.wav 10` (10 steps ≈ 10 tok/s)

### Long startup time (~30 seconds)

Model loading takes ~10-30 seconds on each run. This is normal for cold start. For production use, consider implementing a persistent daemon that keeps models loaded.

## Getting Help

- [AXERA-TECH CosyVoice3 Repo](https://github.com/AXERA-TECH/CosyVoice3.Axera)
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [AXCL Documentation](https://axcl-docs.readthedocs.io/)
