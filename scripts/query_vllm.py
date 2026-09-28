#!/usr/bin/env python3
"""Send a single prompt to a running vLLM server and print the reply.

Usage:
    python3 scripts/query_vllm.py "Write a haiku about GPUs"
"""
import json
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:8001/v1"
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"


def main():
    prompt = " ".join(sys.argv[1:]) or "Say hello in one sentence."

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)

    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
