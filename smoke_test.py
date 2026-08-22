# -*- coding: utf-8 -*-
"""
Qwen3.5-4B smoke test: verify text / image / thinking mode.
Run (in project directory):
    python smoke_test.py [--image image.jpg] [--no-thinking]
"""
from __future__ import annotations

import argparse
import sys
import time

from qwen_engine import QwenEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="optional: test one image")
    parser.add_argument("--no-thinking", action="store_true", help="disable thinking mode (faster)")
    parser.add_argument("--max-tokens", type=int, default=128, help="max new tokens")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    engine = QwenEngine()
    print(f"[1/3] Loading model ({engine.model_path}) ...")
    t0 = time.time()
    engine.load()
    print(f"      done in {time.time() - t0:.1f}s\n")

    messages = []
    if args.image:
        print("[2/3] Image understanding test ...")
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "file": args.image},
                    {"type": "text", "text": "请用一句话描述这张图片的内容。"},
                ],
            }
        )
    else:
        print("[2/3] Text chat test ...")
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "用一句话介绍你自己。"}],
            }
        )

    def on_chunk(phase, text):
        tag = "think" if phase == "thinking" else "answer"
        print(f"[{tag}] {text}", end="", flush=True)

    print("[3/3] Generating (CPU, please wait) ...\n")
    res = engine.chat(
        messages,
        thinking=not args.no_thinking,
        params={"temperature": 0.6, "max_new_tokens": args.max_tokens},
        on_chunk=on_chunk,
    )
    print("\n\n===== RESULT =====")
    print("reasoning:", repr(res["reasoning"]))
    print("answer:", repr(res["answer"]))
    print(f"time: {res['time_s']}s | aborted: {res['aborted']}")
    if res["answer"]:
        print("\nOK: smoke test passed")
    else:
        print("\nFAIL: no answer generated")


if __name__ == "__main__":
    main()
