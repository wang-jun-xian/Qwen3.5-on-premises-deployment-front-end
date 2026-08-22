# -*- coding: utf-8 -*-
"""
Qwen3.5-4B 本地多模态 Web 服务（Flask + SSE 流式输出）
========================================================
运行：
    python app.py [--preload] [--port 7860]

打开浏览器访问 http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import base64
import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from qwen_engine import QwenEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

GEN_LOCK = threading.Lock()
CANCEL_EVENTS: dict[str, threading.Event] = {}
CANCEL_LOCK = threading.Lock()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
IMAGE_MAX_BYTES = 10 * 1024 * 1024
VIDEO_MAX_BYTES = 200 * 1024 * 1024

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 330 * 1024 * 1024

engine = QwenEngine()


# ---------------- 会话存储 ----------------

class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    @staticmethod
    def _check_sid(sid: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_-]{8,64}", sid or ""))

    def _path(self, sid: str) -> Path:
        if not self._check_sid(sid):
            raise ValueError("非法的会话 ID")
        return self.root / f"{sid}.json"

    def new(self) -> dict:
        sid = uuid.uuid4().hex
        data = {
            "session_id": sid,
            "title": "新对话",
            "system_prompt": "",
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
        }
        self.save(data)
        return data

    def get(self, sid: str) -> dict | None:
        p = self._path(sid)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self, data: dict):
        p = self._path(data["session_id"])
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    def list(self) -> list[dict]:
        items = []
        for p in self.root.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                items.append(
                    {
                        "session_id": data.get("session_id"),
                        "title": data.get("title", "新对话"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except Exception:
                continue
        items.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
        return items

    def delete(self, sid: str):
        if not self._check_sid(sid):
            raise ValueError("非法的会话 ID")
        p = self.root / f"{sid}.json"
        media_dir = self.root / sid / "media"
        for target in (p, media_dir):
            if target.exists():
                if target.is_dir():
                    import shutil

                    shutil.rmtree(target)
                else:
                    target.unlink()

    def media_dir(self, sid: str) -> Path:
        if not self._check_sid(sid):
            raise ValueError("非法的会话 ID")
        d = self.root / sid / "media"
        d.mkdir(parents=True, exist_ok=True)
        return d


session_store = SessionStore(SESSIONS_DIR)
# ---------------- 工具函数 ----------------

def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def save_attachment(sid: str, att: dict) -> dict | None:
    kind = att.get("type")
    name = str(att.get("name") or "attachment")
    raw = att.get("data") or ""
    if not raw:
        return None
    try:
        blob = base64.b64decode(raw)
    except Exception:
        raise ValueError("附件 base64 解码失败")

    ext = Path(name).suffix.lower()
    if kind == "image":
        if ext not in IMAGE_EXTS:
            ext = ".png"
        if len(blob) > IMAGE_MAX_BYTES:
            raise ValueError(f"图片过大（>{IMAGE_MAX_BYTES // (1024*1024)}MB）")
    elif kind == "video":
        if ext not in VIDEO_EXTS:
            ext = ".mp4"
        if len(blob) > VIDEO_MAX_BYTES:
            raise ValueError(f"视频过大（>{VIDEO_MAX_BYTES // (1024*1024)}MB）")
    else:
        raise ValueError(f"不支持的附件类型: {kind}")

    fname = f"m_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}{ext}"
    target = session_store.media_dir(sid) / fname
    target.write_bytes(blob)
    return {
        "type": kind,
        "file": fname,
        "name": name,
        "size": len(blob),
        "mime": att.get("mime") or "application/octet-stream",
    }


def build_engine_messages(session: dict, params: dict) -> list:
    msgs = []
    system = (params.get("system_prompt") or session.get("system_prompt") or "").strip()
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(engine.trim_history(session.get("messages", [])))
    return msgs


def media_url(sid: str, rel: str) -> str:
    return f"/media/{sid}/{Path(rel).name}"


def session_view(session: dict) -> dict:
    sid = session["session_id"]
    view = dict(session)
    view["messages"] = []
    for msg in session.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            view["messages"].append(msg)
            continue
        items = []
        for it in content or []:
            if not isinstance(it, dict):
                continue
            item = dict(it)
            if item.get("file"):
                item["url"] = media_url(sid, item["file"])
            items.append(item)
        view["messages"].append({**msg, "content": items})
    return view


# ---------------- 页面 ----------------


@app.route("/")
def index():
    resp = send_from_directory(STATIC_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/static/<path:filename>")
def static_files(filename: str):
    resp = send_from_directory(STATIC_DIR, filename)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/media/<sid>/<filename>")
def media(sid: str, filename: str):
    if not session_store._check_sid(sid):
        return jsonify({"error": "非法的会话 ID"}), 400
    media_dir = SESSIONS_DIR / sid / "media"
    return send_from_directory(media_dir, filename, conditional=True)


# ---------------- API ----------------


@app.get("/api/status")
def api_status():
    return jsonify(engine.system_info())


@app.get("/api/sessions")
def api_sessions():
    return jsonify({"sessions": session_store.list()})


@app.get("/api/sessions/<sid>")
def api_session(sid: str):
    session = session_store.get(sid)
    if session is None:
        return jsonify({"error": "会话不存在"}), 404
    return jsonify(session_view(session))


@app.delete("/api/sessions/<sid>")
def api_session_delete(sid: str):
    session_store.delete(sid)
    return jsonify({"ok": True})


@app.post("/api/sessions/<sid>/clear")
def api_session_clear(sid: str):
    session = session_store.get(sid)
    if session is None:
        return jsonify({"error": "会话不存在"}), 404
    session["messages"] = []
    session["title"] = "新对话"
    session["updated_at"] = time.time()
    session_store.save(session)
    media_dir = SESSIONS_DIR / sid / "media"
    if media_dir.exists():
        import shutil

        shutil.rmtree(media_dir)
    return jsonify({"ok": True})


@app.post("/api/chat/stop")
def api_chat_stop():
    payload = request.get_json(force=True, silent=True) or {}
    sid = payload.get("session_id")
    with CANCEL_LOCK:
        ev = CANCEL_EVENTS.get(sid) if sid else None
    if ev is not None:
        ev.set()
    return jsonify({"ok": True})
@app.post("/api/chat")
def api_chat():
    payload = request.get_json(force=True, silent=True) or {}
    sid = payload.get("session_id")
    if sid and session_store.get(sid) is None:
        sid = None
    session = session_store.get(sid) if sid else session_store.new()
    sid = session["session_id"]

    if payload.get("reset"):
        session["messages"] = []
        session["title"] = "新对话"

    user_text = str(payload.get("message") or "").strip()
    attachments = payload.get("attachments") or []
    if not user_text and not attachments:
        return jsonify({"error": "消息内容为空"}), 400

    params = payload.get("params") or {}
    thinking = bool(params.get("thinking", True))
    tools_enabled = bool(params.get("tools", False))
    gen_params = {
        "temperature": float(params.get("temperature", 1.0)),
        "top_p": float(params.get("top_p", 0.95)),
        "top_k": int(params.get("top_k", 20)),
        "min_p": float(params.get("min_p", 0.0)),
        "repetition_penalty": float(params.get("repetition_penalty", 1.0)),
        "max_new_tokens": int(params.get("max_new_tokens", 2048)),
    }

    media_items = []
    for att in attachments:
        item = save_attachment(sid, att)
        if item:
            media_items.append(item)

    user_content = list(media_items)
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    session.setdefault("messages", []).append({"role": "user", "content": user_content})
    if not session.get("title") or session.get("title") == "新对话":
        session["title"] = (user_text or "图片/视频对话")[:30]
    if params.get("system_prompt") is not None:
        session["system_prompt"] = params["system_prompt"]
    session["updated_at"] = time.time()
    session_store.save(session)

    engine_messages = build_engine_messages(session, params)
    media_dir = SESSIONS_DIR / sid / "media"

    cancel_event = threading.Event()
    with CANCEL_LOCK:
        CANCEL_EVENTS[sid] = cancel_event

    q: queue.Queue = queue.Queue()

    def worker():
        try:
            q.put(("status", "generating"))
            if tools_enabled:
                result = engine.chat_with_tools(
                    engine_messages,
                    media_dir=media_dir,
                    thinking=thinking,
                    params=gen_params,
                    cancel_event=cancel_event,
                )
                for t in result.get("trace", []):
                    q.put(("tool", t))
                q.put(("done", result))
            else:
                result = engine.chat(
                    engine_messages,
                    media_dir=media_dir,
                    thinking=thinking,
                    params=gen_params,
                    on_chunk=lambda phase, text: q.put(("chunk", phase, text)),
                    cancel_event=cancel_event,
                )
                q.put(("done", result))
        except Exception as exc:
            q.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            with CANCEL_LOCK:
                CANCEL_EVENTS.pop(sid, None)

    @stream_with_context
    def generate():
        try:
            yield sse("status", {"state": "queued", "session_id": sid})
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            while True:
                try:
                    kind, *rest = q.get(timeout=0.5)
                except queue.Empty:
                    if not thread.is_alive() and q.empty():
                        break
                    continue
                if kind == "chunk":
                    yield sse(rest[0], {"text": rest[1]})
                elif kind == "status":
                    yield sse("status", {"state": rest[0]})
                elif kind == "tool":
                    yield sse("tool", rest[0])
                elif kind == "done":
                    result = rest[0]
                    session["messages"].append(
                        {
                            "role": "assistant",
                            "content": result.get("answer", ""),
                            "reasoning_content": result.get("reasoning", ""),
                        }
                    )
                    session["updated_at"] = time.time()
                    session_store.save(session)
                    yield sse(
                        "done",
                        {
                            "session_id": sid,
                            "time_s": result.get("time_s", 0),
                            "aborted": bool(result.get("aborted")),
                            "tool_rounds": result.get("tool_rounds", 0),
                        },
                    )
                    break
                elif kind == "error":
                    yield sse("error", {"message": rest[0]})
                    break
        except GeneratorExit:
            cancel_event.set()
            raise

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------- 启动 ----------------


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B 多模态 Web 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--preload", action="store_true", help="启动时在后台预加载模型")
    parser.add_argument("--model", default=None, help="模型目录路径")
    args = parser.parse_args()

    if args.model:
        engine.model_path = Path(args.model)

    if args.preload:
        def _load():
            try:
                engine.load()
            except Exception:
                pass

        threading.Thread(target=_load, daemon=True).start()

    print("=" * 60)
    print(" Qwen3.5-4B 本地多模态助手")
    print(f" 模型目录 : {engine.model_path}")
    print(f" 设备     : {engine.device}（CPU 推理较慢，请耐心等待）")
    print(f" 访问地址 : http://{args.host}:{args.port}")
    print("=" * 60)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()