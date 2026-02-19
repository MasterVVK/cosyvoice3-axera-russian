#!/usr/bin/env python3
"""
Client for CosyVoice3 TTS daemon.

Sends text to the daemon running on /tmp/cv3_tts.sock and receives WAV output.

Usage:
    python3 tts_client.py "Привет, как дела?"
    python3 tts_client.py --socket /tmp/cv3_tts.sock "Текст для синтеза"
"""

import socket
import struct
import json
import sys
import argparse
import time


def tts_request(text, socket_path="/tmp/cv3_tts.sock", timeout=60):
    """Send TTS request and return response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(socket_path)
    except (ConnectionRefusedError, FileNotFoundError) as e:
        print(f"ERROR: Cannot connect to daemon at {socket_path}: {e}")
        print("Is the daemon running? Start with: systemctl start cosyvoice3-t2w")
        return None

    # Send: [4 bytes msg_len] [msg_len bytes text]
    msg = text.encode('utf-8')
    sock.sendall(struct.pack("<I", len(msg)) + msg)

    # Receive: [4 bytes resp_len] [resp_len bytes JSON]
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            print("ERROR: Connection closed before response")
            return None
        raw_len += chunk

    resp_len = struct.unpack("<I", raw_len)[0]
    raw_resp = b""
    while len(raw_resp) < resp_len:
        chunk = sock.recv(resp_len - len(raw_resp))
        if not chunk:
            break
        raw_resp += chunk

    sock.close()
    return json.loads(raw_resp.decode())


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 TTS Client")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--socket", default="/tmp/cv3_tts.sock", help="Daemon socket path")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    args = parser.parse_args()

    t0 = time.time()
    result = tts_request(args.text, args.socket, args.timeout)
    elapsed = time.time() - t0

    if result is None:
        sys.exit(1)

    if result.get("ok"):
        print(f"OK: {result.get('wav', 'output.wav')} ({elapsed:.2f}s)")
    else:
        print(f"ERROR: {result.get('error', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
