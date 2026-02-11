#!/usr/bin/env python3
"""
Standalone CosyVoice3 Tokenizer HTTP Server.

Uses transformers.AutoTokenizer directly — no CosyVoice repo dependency needed.
Compatible with AXERA-TECH main_axcl_aarch64 binary.

EOS token: 1773 (hardcoded, matching AXERA-TECH official binary)

API:
    GET  /eos_id  -> {"eos_id": 1773}
    GET  /bos_id  -> {"bos_id": -1}
    GET  /health  -> {"status": "ok"}
    POST /encode  <- {"text": "..."} -> {"token_ids": [...]}
    POST /decode  <- {"token_ids": [...]} -> {"text": "..."}

Usage:
    python3 tokenizer_server.py --model_dir /path/to/CosyVoice-BlankEN --port 12345

Requirements:
    pip install transformers
"""
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoTokenizer


class CosyVoice3Tokenizer:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print(f"Loaded tokenizer from {model_dir}")
        print(f"  Vocab size: {self.tokenizer.vocab_size}")
        print(f"  EOS (hardcoded): 1773")

        test = self.encode("hello world")
        print(f"  Test encode 'hello world': {test}")

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eos_id(self):
        return 1773  # CosyVoice3 specific, matching AXERA-TECH official

    @property
    def bos_id(self):
        return -1


tokenizer_instance = None


class RequestHandler(BaseHTTPRequestHandler):
    timeout = 5

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if self.path == '/eos_id':
            msg = json.dumps({'eos_id': tokenizer_instance.eos_id})
        elif self.path == '/bos_id':
            msg = json.dumps({'bos_id': tokenizer_instance.bos_id})
        elif self.path == '/health':
            msg = json.dumps({'status': 'ok'})
        else:
            msg = json.dumps({'error': f'unknown path: {self.path}'})

        self.wfile.write(msg.encode())

    def do_POST(self):
        data = self.rfile.read(int(self.headers['content-length']))
        data = data.decode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if self.path == '/encode':
            req = json.loads(data)
            text = req['text']
            token_ids = tokenizer_instance.encode(text)
            msg = json.dumps({'token_ids': token_ids})
        elif self.path == '/decode':
            req = json.loads(data)
            token_ids = req['token_ids']
            text = tokenizer_instance.decode(token_ids)
            msg = json.dumps({'text': text})
        else:
            msg = json.dumps({'error': f'unknown path: {self.path}'})

        self.wfile.write(msg.encode())


def main():
    global tokenizer_instance

    parser = argparse.ArgumentParser(description='CosyVoice3 Tokenizer HTTP Server')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to CosyVoice-BlankEN tokenizer directory')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=12345)
    args = parser.parse_args()

    tokenizer_instance = CosyVoice3Tokenizer(args.model_dir)

    host = (args.host, args.port)
    print(f"\nTokenizer server at http://{args.host}:{args.port}")
    print(f"  EOS ID: {tokenizer_instance.eos_id}")

    server = HTTPServer(host, RequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
