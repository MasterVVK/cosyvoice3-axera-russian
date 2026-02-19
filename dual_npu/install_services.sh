#!/bin/bash
# Install CosyVoice3 dual-NPU daemons as systemd services
# Run on CM3588 as root

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing CosyVoice3 Dual-NPU Services ==="

# Copy service files
cp "$SCRIPT_DIR/cosyvoice3-rkllm.service" /etc/systemd/system/
cp "$SCRIPT_DIR/cosyvoice3-t2w.service" /etc/systemd/system/

# Copy token server script to deployment location
mkdir -p /root/cosyvoice3-build/dual_npu
cp "$SCRIPT_DIR/rkllm_token_server.py" /root/cosyvoice3-build/dual_npu/

# Reload systemd
systemctl daemon-reload

echo ""
echo "Services installed:"
echo "  cosyvoice3-rkllm  - RKLLM Token Server (RK3588 NPU)"
echo "  cosyvoice3-t2w    - Token2Wav Daemon (AX650N NPU)"
echo ""
echo "Usage:"
echo "  systemctl start cosyvoice3-rkllm    # Start token server (loads model ~30s)"
echo "  systemctl start cosyvoice3-t2w      # Start Token2Wav daemon"
echo "  systemctl enable cosyvoice3-rkllm cosyvoice3-t2w  # Auto-start on boot"
echo ""
echo "Quick start both:"
echo "  systemctl start cosyvoice3-rkllm && sleep 30 && systemctl start cosyvoice3-t2w"
echo ""
echo "Check status:"
echo "  systemctl status cosyvoice3-rkllm cosyvoice3-t2w"
echo "  journalctl -u cosyvoice3-rkllm -f   # Follow RKLLM logs"
echo "  journalctl -u cosyvoice3-t2w -f      # Follow Token2Wav logs"
