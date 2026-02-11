# Setup Guide

Step-by-step installation of CosyVoice3 Russian TTS on CM3588 + AX650N.

## Hardware Requirements

- **FriendlyElec CM3588** (or any board with AX650N NPU via PCIe)
  - CM3588 NAS Kit, M4N-Dock, or AX650N demo board
  - AX650N connected via M.2 (PCIe Gen3 x1)
- **Storage**: ~3 GB for models + binaries
- **RAM**: 4 GB+ system RAM (NPU has its own 7 GB CMM)

## Software Requirements

- **AXCL runtime** 3.6+ ([installation guide](https://axcl-docs.readthedocs.io/))
- **Python** 3.8+
- **pip packages**: `transformers`, `scipy`, `numpy`

## Step 1: Install AXCL Runtime

Follow the [official AXCL documentation](https://axcl-docs.readthedocs.io/) for your platform.

Verify installation:
```bash
/usr/bin/axcl/axcl-smi
```

Expected output:
```
+------------------------------------------------------------------------------------------------+
| AXCL-SMI  V3.6.x                                                                              |
+-----------------------------------------+--------------+---------------------------------------+
| Card  Name                     Firmware | Bus-Id       |                          Memory-Usage |
|    0  AX650N                     V3.6.x | xxxx:xx:xx.x |                  xx MiB /      945 MiB |
+-----------------------------------------+--------------+---------------------------------------+
```

## Step 2: Download Models

On a machine with internet access:

```bash
pip install huggingface_hub

# Download AXERA-TECH CosyVoice3 (~3 GB)
huggingface-cli download AXERA-TECH/CosyVoice3 --local-dir ./AXERA-CosyVoice3
```

If behind a proxy:
```bash
export HTTPS_PROXY=socks5://your-proxy:port
huggingface-cli download AXERA-TECH/CosyVoice3 --local-dir ./AXERA-CosyVoice3
```

## Step 3: Copy to Device

```bash
# Copy everything to the device
rsync -avz --progress ./AXERA-CosyVoice3/ root@<DEVICE_IP>:/root/AXERA-CosyVoice3/
```

**Note**: If space is limited, you can skip `frontend-onnx/` (~1 GB) initially — it's only needed for voice cloning prompt generation:
```bash
rsync -avz --progress --exclude='frontend-onnx/' \
    ./AXERA-CosyVoice3/ root@<DEVICE_IP>:/root/AXERA-CosyVoice3/
```

## Step 4: Install Our Scripts

```bash
# From this repository
scp scripts/run_tts.sh scripts/postprocess.py scripts/tokenizer_server.py \
    root@<DEVICE_IP>:/root/AXERA-CosyVoice3/

# On the device
ssh root@<DEVICE_IP>
cd /root/AXERA-CosyVoice3

# Make executable
chmod +x run_tts.sh main_axcl_aarch64

# Install Python dependencies
pip3 install scipy --break-system-packages
pip3 install transformers --break-system-packages
```

## Step 5: Test

```bash
# Basic test with default Chinese voice
./run_tts.sh "Hello, this is a test of CosyVoice three." test.wav 10 prompt_files

# If you have Russian prompts set up:
./run_tts.sh "Привет, как дела?" test_ru.wav 30 prompt_files_russian_v2
```

Expected output:
```
Text: Hello, this is a test of CosyVoice three.
Steps: 10, Prompt: prompt_files
---
[I]... ttft: 300.00 ms
[I]... total decode tokens: 95
[N]... hit eos, decode avg 10.15 token/s
[I]... tts total use time: 12.345 s
output.wav -> test.wav (4.50s, lp=5000Hz, pad=300ms)
---
Done: test.wav
```

## Step 6: Set Up Russian Voice (Optional)

See [RUSSIAN_VOICE.md](RUSSIAN_VOICE.md) for generating Russian voice prompts.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

### Quick fixes

**Tokenizer won't start:**
```bash
# Check if transformers is installed
python3 -c "from transformers import AutoTokenizer; print('OK')"

# Check tokenizer model files exist
ls scripts/CosyVoice-BlankEN/
# Should contain: vocab.json, merges.txt, tokenizer_config.json
```

**NPU not found:**
```bash
# Verify AX650N is visible
lspci | grep -i axera
/usr/bin/axcl/axcl-smi
```

**Permission denied on main_axcl_aarch64:**
```bash
chmod +x main_axcl_aarch64
```
