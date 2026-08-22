# -*- coding: utf-8 -*-
"""
Qwen3.5-4B 本地多模态推理引擎
==============================
能力清单：
  * 纯文本对话
  * 图片理解（多图、图文混排）
  * 视频理解（本地 mp4/webm 等，自动抽帧）
  * 多轮对话历史（配合前端/CLI 持久化，跨重启记忆）
  * 思考模式 / 直答模式切换（enable_thinking）
  * 工具调用演示（calculator / current_time）

说明：该模型是视觉语言模型（无音频编码器），因此输入模态为 文本 + 图片 + 视频。
无 GPU 也可运行（CPU 推理，速度较慢），建议 16GB 以上内存。
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", r"E:\LLMModels\Qwen3.5-4B")
DEFAULT_MAX_IMAGE_PIXELS = int(os.environ.get("QWEN_MAX_IMAGE_PIXELS", "1500000"))
DEFAULT_MAX_VIDEO_EDGE = int(os.environ.get("QWEN_MAX_VIDEO_EDGE", "768"))

THINK_START = "<think>"
THINK_END = "</think>"

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TOOL_FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
TOOL_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.S)

DEFAULT_GEN_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 2048,
}

DEMO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，例如 12.5*(3+4)/2",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "获取当前日期与时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class GenerationCancelled(Exception):
    """生成被用户中断。"""


def _safe_eval(expr: str):
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"[^0-9+\-*/().% \t]", "", expr)
    if not expr.strip():
        raise ValueError("空表达式")
    return eval(expr, {"__builtins__": {}}, {})


def _run_demo_tool(name: str, args: dict):
    if name == "calculator":
        return {"result": _safe_eval(args.get("expression", ""))}
    if name == "current_time":
        return {"datetime": time.strftime("%Y-%m-%d %H:%M:%S")}
    return {"error": f"未知工具: {name}"}


class _CancellableStreamer:
    """包装 TextIteratorStreamer，支持在 generate 过程中取消。"""

    def __init__(self, tokenizer, cancel_event=None, **kwargs):
        from transformers import TextIteratorStreamer

        self._inner = TextIteratorStreamer(tokenizer, **kwargs)
        self.cancel_event = cancel_event

    def put(self, value):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError("generation-cancelled")
        self._inner.put(value)

    def end(self):
        self._inner.end()

    def __iter__(self):
        return self._inner.__iter__()

    def __next__(self):
        return next(self._inner)


class QwenEngine:
    """Qwen3.5-4B 本地模型封装：加载、多模态输入、流式生成、历史裁剪、工具调用。"""

    def __init__(
        self,
        model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
        device: str | None = None,
        dtype=None,
        max_ctx_tokens: int = 16384,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        max_video_edge: int = DEFAULT_MAX_VIDEO_EDGE,
    ):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.bfloat16
        self.max_ctx_tokens = max_ctx_tokens
        self.max_image_pixels = max_image_pixels
        self.max_video_edge = max_video_edge

        self._model = None
        self._processor = None
        self._state = "idle"
        self._error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._supports_min_p = False

    # ---------- 状态 ----------

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._state == "ready"

    @property
    def is_loading(self) -> bool:
        return self._state == "loading"

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def model_name(self) -> str:
        return self.model_path.name

    # ---------- 加载 ----------

    def load(self):
        """加载模型与处理器（线程安全，重复调用安全）。"""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._state = "loading"
            self._error = None
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor

                try:
                    self._processor = AutoProcessor.from_pretrained(self.model_path)
                except Exception:
                    from transformers import Qwen3VLProcessor

                    self._processor = Qwen3VLProcessor.from_pretrained(self.model_path)

                try:
                    self._model = AutoModelForImageTextToText.from_pretrained(
                        self.model_path,
                        dtype=self.dtype,
                        low_cpu_mem_usage=True,
                        device_map=self.device,
                    )
                except Exception:
                    from transformers import Qwen3_5ForConditionalGeneration

                    self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
                        self.model_path,
                        dtype=self.dtype,
                        low_cpu_mem_usage=True,
                        device_map=self.device,
                    )

                self._model.eval()
                if hasattr(self._processor, "tokenizer"):
                    self._processor.tokenizer.model_max_length = self.max_ctx_tokens

                import inspect

                self._supports_min_p = "min_p" in inspect.signature(self._model.generate).parameters
                self._state = "ready"
            except Exception as exc:
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(
                    "模型加载失败。若报错与 Qwen3_5ForConditionalGeneration 相关，请升级 transformers：\n"
                    'pip install -U "transformers @ https://github.com/huggingface/transformers/archive/refs/heads/main.zip"'
                ) from exc

    # ---------- 媒体处理 ----------

    @staticmethod
    def _open_image(path: str | os.PathLike) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_video(self, path: str | os.PathLike, max_frames: int = 48):
        """使用 PyAV 读取视频并均匀抽帧（最多 max_frames 帧）。

        返回 (frames, info)：
          frames: (T, H, W, 3) uint8 数组，按原视频时间轴均匀覆盖
          info:   {"total_num_frames", "fps", "duration", "width", "height"}
                  元数据中的 fps 为抽帧后的有效帧率，供处理器精确取样
        """
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("缺少视频解码依赖，请执行: pip install av") from exc

        path = Path(path)
        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            avg_rate = float(stream.average_rate) if stream.average_rate else 25.0
            # 估算总帧数（mp4 等容器一般带 duration）
            total_est = None
            if stream.duration:
                time_base = float(stream.time_base) if stream.time_base else 1.0
                total_est = int(float(stream.duration) * time_base * avg_rate)
            step = 1
            if total_est and total_est > max_frames:
                step = max(1, math.ceil(total_est / max_frames))

            frames = []
            last_frame = None
            idx = 0
            for frame in container.decode(stream):
                last_frame = frame.to_ndarray(format="rgb24")
                if step == 1 and len(frames) >= max_frames:
                    break
                if idx % step == 0:
                    frames.append(last_frame)
                idx += 1
            # 补上结尾帧，保证覆盖视频尾部
            if step > 1 and last_frame is not None and (idx - 1) % step != 0:
                frames.append(last_frame)
        finally:
            container.close()
        if not frames:
            raise ValueError(f"视频中未解码到有效帧: {path.name}")

        # 内存保护：把帧缩放到最大边长不超过上限，显著降低视觉编码的内存与耗时
        h, w = frames[0].shape[:2]
        if max(h, w) > self.max_video_edge:
            scale = self.max_video_edge / max(h, w)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            from PIL import Image as _PIL

            frames = [
                np.asarray(_PIL.fromarray(f).resize((nw, nh), _PIL.LANCZOS), dtype=np.uint8)
                for f in frames
            ]
            h, w = nh, nw

        kept = len(frames)
        effective_fps = avg_rate / step
        info = {
            "total_num_frames": kept,
            "fps": effective_fps,
            "duration": kept / effective_fps,
            "width": w,
            "height": h,
        }
        return np.stack(frames, axis=0).astype(np.uint8), info

    # ---------- 消息解析 ----------

    def _resolve_messages(self, messages, media_dir=None):
        """把历史消息中的文件引用解析成 PIL 图片 / 视频帧数组，并返回处理器所需的独立列表。"""
        images, videos = [], []
        resolved = []

        def _locate(item):
            src = item.get("file") or item.get("path") or item.get("url")
            if not src:
                return None
            p = Path(src)
            if not p.is_absolute() and media_dir:
                # 兼容旧格式：file 字段可能带 "media/" 前缀，而 media_dir 本身已含 media 目录
                if p.parts and p.parts[0] == "media":
                    p = Path(*p.parts[1:])
                p = Path(media_dir) / p
            return p

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                resolved.append(msg)
                continue
            items = []
            for it in content or []:
                if not isinstance(it, dict):
                    continue
                kind = it.get("type")
                if kind == "text":
                    items.append({"type": "text", "text": it.get("text", "")})
                elif kind in ("image", "image_url"):
                    p = _locate(it)
                    if p is None:
                        continue
                    images.append(self._open_image(p))
                    items.append({"type": "image"})
                elif kind == "video":
                    p = _locate(it)
                    if p is None:
                        continue
                    frames, info = self._load_video(p)
                    videos.append((frames, info))
                    items.append({"type": "video"})
            resolved.append({**msg, "content": items})
        return resolved, images, videos

    # ---------- 生成 ----------

    def chat(
        self,
        messages,
        media_dir=None,
        *,
        thinking: bool = True,
        tools=None,
        params: dict | None = None,
        on_chunk: Optional[Callable[[str, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        """
        多模态对话。on_chunk(phase, text) 用于流式回调：
          phase = "thinking" | "answer"
        返回 {"reasoning", "answer", "time_s", "aborted"}。
        """
        if not self.is_ready:
            self.load()

        gen_params = {**DEFAULT_GEN_PARAMS, **(params or {})}
        max_new_tokens = int(gen_params.get("max_new_tokens", 2048))

        prompt_messages, images, videos = self._resolve_messages(messages, media_dir)
        try:
            # transformers 5.x：模板参数直接以关键字传入
            text = self._processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools or None,
                enable_thinking=thinking,
            )
        except TypeError:
            # transformers 4.x 兼容写法
            text = self._processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools or None,
                chat_template_kwargs={"enable_thinking": thinking},
            )

        kw = {"text": [text], "return_tensors": "pt"}
        if images:
            kw["images"] = images
            kw["images_kwargs"] = {"max_pixels": self.max_image_pixels}
        if videos:
            kw["videos"] = [frames for frames, _ in videos]
            kw["videos_kwargs"] = {"video_metadata": [info for _, info in videos]}
        inputs = self._processor(**kw)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        streamer = _CancellableStreamer(
            self._processor.tokenizer,
            cancel_event=cancel_event,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=120,
        )

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=float(gen_params.get("temperature", 1.0)) > 0.01,
            temperature=float(gen_params.get("temperature", 1.0)),
            top_p=float(gen_params.get("top_p", 0.95)),
            top_k=int(gen_params.get("top_k", 20)),
            repetition_penalty=float(gen_params.get("repetition_penalty", 1.0)),
            eos_token_id=self._processor.tokenizer.eos_token_id,
            pad_token_id=self._processor.tokenizer.pad_token_id,
        )
        if self._supports_min_p:
            gen_kwargs["min_p"] = float(gen_params.get("min_p", 0.0))

        gen_errors: list[BaseException] = []

        def _run():
            try:
                with torch.inference_mode():
                    self._model.generate(**inputs, streamer=streamer, **gen_kwargs)
            except Exception as exc:
                gen_errors.append(exc)
            finally:
                # 无论正常结束还是崩溃，都唤醒阻塞中的流式读取
                try:
                    streamer.end()
                except Exception:
                    pass

        t0 = time.time()
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        reasoning, answer, aborted = self._collect_stream(
            streamer, thinking=thinking, on_chunk=on_chunk, cancel_event=cancel_event
        )
        thread.join(timeout=5)
        if gen_errors and not aborted and "generation-cancelled" not in str(gen_errors[0]):
            exc = gen_errors[0]
            msg = str(exc) or type(exc).__name__
            low = msg.lower()
            if "memory" in low or "cuda" in low or "out of" in low:
                raise RuntimeError(
                    f"{type(exc).__name__}: {msg}\n"
                    "（内存不足：请先关闭其他占用内存的程序，换更小的图片/视频，或减小 max_new_tokens）"
                ) from exc
            raise exc

        return {
            "reasoning": reasoning,
            "answer": answer,
            "time_s": round(time.time() - t0, 2),
            "aborted": aborted,
        }

    def _collect_stream(self, streamer, thinking: bool, on_chunk, cancel_event):
        buffer = ""
        reasoning_parts, answer_parts = [], []
        # 注意：chat 模板在生成提示词末尾已经写入了 "<think>\n"，
        # 因此开启思考模式时，流式输出直接以思考内容开头。
        state = "think" if thinking else "answer"
        hold = max(len(THINK_END) - 1, 0)

        def _emit(phase, text):
            if on_chunk and text:
                on_chunk(phase, text)

        while True:
            if cancel_event is not None and cancel_event.is_set():
                return "".join(reasoning_parts), "".join(answer_parts), True
            try:
                chunk = next(streamer)
            except StopIteration:
                break
            except Exception:
                break
            if not chunk:
                continue
            buffer += chunk

            while True:
                if state == "think":
                    if THINK_END in buffer:
                        content, _, rest = buffer.partition(THINK_END)
                        if content.strip():
                            _emit("thinking", content)
                            reasoning_parts.append(content)
                        buffer = rest
                        state = "answer"
                    else:
                        if len(buffer) > hold:
                            emit, buffer = buffer[:-hold], buffer[-hold:]
                            if emit.strip():
                                _emit("thinking", emit)
                                reasoning_parts.append(emit)
                        break
                elif state == "answer":
                    if not buffer:
                        break
                    _emit("answer", buffer)
                    answer_parts.append(buffer)
                    buffer = ""
                    break
                else:
                    break

        # 生成结束时的残留内容
        if state == "think":
            if buffer.strip():
                _emit("thinking", buffer)
                reasoning_parts.append(buffer)
        elif buffer:
            _emit("answer", buffer)
            answer_parts.append(buffer)

        return "".join(reasoning_parts), "".join(answer_parts), False

    # ---------- 工具调用 ----------

    @staticmethod
    def parse_tool_calls(text: str):
        calls = []
        for block in TOOL_CALL_RE.findall(text or ""):
            m = TOOL_FUNC_RE.search(block)
            if not m:
                continue
            name = m.group(1).strip()
            args = {}
            for k, v in TOOL_PARAM_RE.findall(m.group(2)):
                args[k.strip()] = v.strip()
            calls.append((name, args))
        return calls

    def chat_with_tools(
        self,
        messages,
        media_dir=None,
        *,
        thinking: bool = True,
        tools=None,
        params: dict | None = None,
        cancel_event: Optional[threading.Event] = None,
        max_rounds: int = 4,
    ) -> dict:
        """带工具调用循环的对话；返回结果中附带 trace 记录工具执行过程。"""
        tools = tools or DEMO_TOOLS
        msgs = list(messages)
        trace, total_time, rounds = [], 0.0, 0

        while rounds < max_rounds:
            res = self.chat(
                msgs,
                media_dir=media_dir,
                thinking=thinking,
                tools=tools,
                params=params,
                cancel_event=cancel_event,
            )
            total_time += res["time_s"]
            if res.get("aborted"):
                res["time_s"] = total_time
                res["tool_rounds"] = rounds
                res["trace"] = trace
                return res

            calls = self.parse_tool_calls(res["answer"])
            if not calls:
                res["time_s"] = total_time
                res["tool_rounds"] = rounds
                res["trace"] = trace
                return res

            msgs.append({"role": "assistant", "content": res["answer"]})
            for name, args in calls:
                try:
                    result = _run_demo_tool(name, args)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                trace.append({"tool": name, "args": args, "result": result})
                msgs.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
            rounds += 1

        res["time_s"] = total_time
        res["tool_rounds"] = rounds
        res["trace"] = trace
        return res

    # ---------- 历史裁剪 ----------

    def _estimate_message_tokens(self, msg, image_cost: int = 512, video_cost: int = 2048) -> int:
        tokenizer = self._processor.tokenizer
        content = msg.get("content", "")
        if isinstance(content, str):
            return max(1, len(tokenizer.encode(content)))
        total = 0
        for it in content or []:
            if not isinstance(it, dict):
                continue
            kind = it.get("type")
            if kind == "text":
                total += max(1, len(tokenizer.encode(it.get("text", ""))))
            elif kind in ("image", "image_url"):
                total += image_cost
            elif kind == "video":
                total += video_cost
        return max(1, total)

    def trim_history(self, messages, max_tokens: int | None = None) -> list:
        """按 token 估算从尾部保留消息，防止上下文无限增长。"""
        if self._processor is None:
            return messages[-20:]
        budget = max_tokens or self.max_ctx_tokens
        keep, total = [], 0
        for msg in reversed(list(messages)):
            cost = self._estimate_message_tokens(msg)
            if keep and total + cost > budget:
                break
            total += cost
            keep.append(msg)
        return list(reversed(keep))

    # ---------- 元信息 ----------

    def system_info(self) -> dict:
        config = {}
        cfg_path = self.model_path / "config.json"
        if cfg_path.exists():
            try:
                config = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
        return {
            "model": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "state": self._state,
            "error": self._error,
            "architectures": config.get("architectures", []),
            "context_length": config.get("text_config", {}).get("max_position_embeddings", None),
            "modalities": ["text", "image", "video"],
            "features": ["thinking", "history", "tools", "streaming"],
        }
