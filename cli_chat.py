# -*- coding: utf-8 -*-
"""
Qwen3.5-4B 命令行多模态对话
============================
用法示例：
    python cli_chat.py
    python cli_chat.py --image photo.jpg --system "你是中文助手"
    python cli_chat.py --video demo.mp4 --no-thinking

对话内命令：
    /help            查看帮助
    /clear           清空历史
    /think           开启思考模式
    /nothink         关闭思考模式（直答）
    /image <路径>    附加一张图片
    /video <路径>    附加一个视频
    /exit            退出
"""
from __future__ import annotations

import argparse
import os
import sys

from qwen_engine import DEFAULT_MODEL_PATH, QwenEngine

GRAY = "\033[90m"
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_help():
    print(
        f"""
{BOLD}可用命令{RESET}
  /help            查看帮助
  /clear           清空当前对话历史
  /think           开启思考模式
  /nothink         关闭思考模式（直答）
  /image <路径>    附加一张图片（jpg/png/webp…）
  /video <路径>    附加一个视频（mp4/webm…）
  /exit            退出
"""
    )


def stream_print(phase: str, text: str):
    if phase == "thinking":
        sys.stdout.write(GRAY + text + RESET)
    else:
        sys.stdout.write(text)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B 命令行多模态对话")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="模型目录路径")
    parser.add_argument("--system", default="", help="系统提示词")
    parser.add_argument("--no-thinking", action="store_true", help="关闭思考模式")
    parser.add_argument("--image", action="append", default=[], help="开场附加图片（可多次）")
    parser.add_argument("--video", action="append", default=[], help="开场附加视频（可多次）")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tools", action="store_true", help="启用工具调用演示")
    args = parser.parse_args()

    os.system("")  # 启用 Windows 终端 ANSI 颜色
    engine = QwenEngine(args.model)
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"{BLUE}正在加载模型 {engine.model_name} …（首次加载可能需要数分钟）{RESET}")
    engine.load()
    print(f"{GREEN}模型就绪（设备: {engine.device}）{RESET}")
    print_help()

    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    thinking = not args.no_thinking
    media_pending = []

    def run_turn(text: str, media: list | None = None):
        nonlocal media_pending
        content = []
        for kind, path in media or []:
            content.append({"type": kind, "file": path})
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            return
        history.append({"role": "user", "content": content})
        messages = list(history)

        print(f"\n{BOLD}你{RESET}（思考{'开' if thinking else '关'}）: {text or '[媒体]'}")
        if thinking:
            print(f"{BLUE}模型{RESET}（思考中，灰色为思考内容）: ", end="", flush=True)
        else:
            print(f"{BLUE}模型{RESET}: ", end="", flush=True)

        params = {"max_new_tokens": args.max_new_tokens}
        try:
            if args.tools:
                result = engine.chat_with_tools(
                    messages,
                    thinking=thinking,
                    params=params,
                )
                print()
                for t in result.get("trace", []):
                    print(f"{GREEN}  [工具] {t['tool']}({t['args']}) → {t['result']}{RESET}")
                print(f"{GREEN}  [最终回答]{RESET}")
                print(result.get("answer", ""))
            else:
                result = engine.chat(
                    messages,
                    thinking=thinking,
                    params=params,
                    on_chunk=stream_print,
                )
                print()
        except KeyboardInterrupt:
            print(f"\n{RESET}[已中断]")
            return
        except Exception as exc:
            print(f"\n{RED}[发生错误] {type(exc).__name__}: {exc}{RESET}")
            return

        history.append(
            {
                "role": "assistant",
                "content": result.get("answer", ""),
                "reasoning_content": result.get("reasoning", ""),
            }
        )
        if thinking and not result.get("answer") and result.get("reasoning"):
            print(f"{RED}提示：思考内容占满了 token 上限，没有生成正式回答。"
                  f"可调大 --max-new-tokens 或使用 --no-thinking{RESET}")
        print(f"{GRAY}（生成耗时 {result.get('time_s', 0)} 秒）{RESET}")

    # 开场媒体
    opening = [("image", p) for p in args.image] + [("video", p) for p in args.video]
    if opening:
        run_turn("请描述我提供的内容。" if not args.no_thinking else "请描述我提供的内容。", opening)

    while True:
        try:
            line = input(f"\n{BOLD}你 > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not line:
            continue

        if line in ("/exit", "/quit", "/q"):
            print("再见！")
            break
        if line == "/help":
            print_help()
            continue
        if line == "/clear":
            history.clear()
            if args.system:
                history.append({"role": "system", "content": args.system})
            print("历史已清空")
            continue
        if line == "/think":
            thinking = True
            print("已开启思考模式")
            continue
        if line == "/nothink":
            thinking = False
            print("已关闭思考模式（直答）")
            continue
        if line.startswith("/image "):
            media_pending.append(("image", line.split(maxsplit=1)[1].strip()))
            print(f"已附加图片: {media_pending[-1][1]}")
            continue
        if line.startswith("/video "):
            media_pending.append(("video", line.split(maxsplit=1)[1].strip()))
            print(f"已附加视频: {media_pending[-1][1]}")
            continue

        run_turn(line, media_pending or None)
        media_pending = []


if __name__ == "__main__":
    main()
