"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  sessionId: null,
  attachments: [],
  streaming: false,
  controller: null,
  currentAssistant: null,
};

const DEFAULT_PARAMS = {
  system_prompt: "",
  thinking: true,
  tools: false,
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  max_new_tokens: 2048,
};

let params = loadParams();

function loadParams() {
  try {
    return { ...DEFAULT_PARAMS, ...JSON.parse(localStorage.getItem("qwen_params_v2") || "{}") };
  } catch (e) {
    return { ...DEFAULT_PARAMS };
  }
}
function saveParams() {
  localStorage.setItem("qwen_params_v2", JSON.stringify(params));
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showToast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), 3500);
}

function scrollBottom() {
  const m = $("#messages");
  m.scrollTop = m.scrollHeight;
}

function setSubtitle(text) {
  $("#chat-subtitle").textContent = text;
}

function autoResize() {
  const ta = $("#input");
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
}

// ---------------- 会话管理 ----------------

async function api(path, opts = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!resp.ok) {
    let msg = "请求失败";
    try { msg = (await resp.json()).error || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return resp.json();
}

async function refreshSessions(activeSid) {
  try {
    const data = await api("/api/sessions");
    const listEl = $("#session-list");
    listEl.innerHTML = "";
    if (!data.sessions.length) {
      listEl.innerHTML = '<div class="session-meta" style="padding:8px 12px">暂无历史会话</div>';
      return;
    }
    for (const s of data.sessions) {
      const item = document.createElement("div");
      item.className = "session-item" + (s.session_id === activeSid ? " active" : "");
      item.innerHTML =
        '<div class="session-title">' + escapeHtml(s.title) + "</div>" +
        '<div class="session-meta">' + s.message_count + " 条消息</div>";
      item.addEventListener("click", () => openSession(s.session_id));
      listEl.appendChild(item);
    }
  } catch (e) {
    console.warn("刷新会话列表失败", e);
  }
}

function showWelcome() {
  const w = $("#welcome");
  if (w) w.hidden = false;
}

function hideWelcome() {
  const w = $("#welcome");
  if (w) w.hidden = true;
}

function clearMessages() {
  const m = $("#messages");
  for (const child of Array.from(m.children)) {
    if (child.id === "welcome") continue;
    child.remove();
  }
  hideWelcome();
}

async function openSession(sid) {
  if (state.streaming) { showToast("请先停止当前生成"); return; }
  try {
    const data = await api("/api/sessions/" + sid);
    state.sessionId = sid;
    clearMessages();
    for (const msg of data.messages) {
      if (msg.role === "user") renderUserMessage(msg.content || []);
      else renderAssistantMessage(msg, false);
    }
    $("#chat-title").textContent = data.title || "新对话";
    setSubtitle("文本 · 图片 · 视频 · 多轮记忆");
    scrollBottom();
    refreshSessions(sid);
  } catch (e) {
    showToast("打开会话失败：" + e.message);
  }
}

function newChat() {
  if (state.streaming) { showToast("请先停止当前生成"); return; }
  state.sessionId = null;
  state.attachments = [];
  updateAttachments();
  clearMessages();
  showWelcome();
  $("#chat-title").textContent = "新对话";
  setSubtitle("文本 · 图片 · 视频 · 多轮记忆");
  $("#input").value = "";
  autoResize();
  refreshSessions(null);
}

// ---------------- 消息渲染 ----------------

function renderUserMessage(content) {
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  for (const it of content || []) {
    if (it.type === "image" && it.url) {
      const img = document.createElement("img");
      img.src = it.url;
      img.className = "attach-img";
      img.alt = it.name || "图片";
      img.addEventListener("click", () => window.open(it.url, "_blank"));
      bubble.appendChild(img);
    } else if (it.type === "video" && it.url) {
      const vid = document.createElement("video");
      vid.src = it.url;
      vid.controls = true;
      vid.preload = "metadata";
      vid.className = "attach-video";
      bubble.appendChild(vid);
    } else if (it.type === "text" && it.text) {
      const p = document.createElement("div");
      p.className = "user-text";
      p.textContent = it.text;
      bubble.appendChild(p);
    }
  }
  if (!bubble.children.length) return;
  wrap.appendChild(bubble);
  $("#messages").appendChild(wrap);
}

function createAssistantBubble() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const answerEl = document.createElement("div");
  answerEl.className = "markdown";
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  answerEl.appendChild(cursor);
  bubble.appendChild(answerEl);
  wrap.appendChild(bubble);
  $("#messages").appendChild(wrap);
  const a = {
    el: wrap,
    bubble,
    thinkingBox: null,
    thinkingContent: document.createElement("div"),
    answerEl,
    thinkingRaw: "",
    answerRaw: "",
    done: false,
  };
  a.thinkingContent.className = "thinking-content";
  scrollBottom();
  return a;
}

function showThinkingBox(a) {
  if (a.thinkingBox) return;
  const details = document.createElement("details");
  details.className = "thinking-box";
  details.open = true;
  const summary = document.createElement("summary");
  summary.textContent = "🧠 思考过程";
  details.appendChild(summary);
  details.appendChild(a.thinkingContent);
  a.bubble.insertBefore(details, a.answerEl);
  a.thinkingBox = details;
}

function appendThinking(text) {
  const a = state.currentAssistant;
  if (!a) return;
  a.thinkingRaw += text;
  a.thinkingContent.textContent = a.thinkingRaw;
  showThinkingBox(a);
  scrollBottom();
}

function appendAnswer(text) {
  const a = state.currentAssistant;
  if (!a) return;
  a.answerRaw += text;
  const cursor = a.answerEl.querySelector(".cursor");
  a.answerEl.innerHTML = renderMarkdown(a.answerRaw);
  if (cursor && cursor.parentNode === a.answerEl) a.answerEl.appendChild(cursor);
  scrollBottom();
}

function finishAssistant(aborted) {
  const a = state.currentAssistant;
  if (!a) return;
  a.done = true;
  const cursor = a.answerEl.querySelector(".cursor");
  if (cursor) cursor.remove();
  if (!a.answerRaw && !a.thinkingRaw) {
    a.answerEl.innerHTML = '<span style="color:var(--text-dim)">（无输出' + (aborted ? " · 已停止" : "") + "）</span>";
  } else if (!a.answerRaw && a.thinkingRaw) {
    a.answerEl.innerHTML += '<div class="tool-note">（已生成思考内容，但没有正式回答：多半是“最大新 tokens”被思考占满。可调大该值，或关闭思考模式）</div>';
  } else if (aborted) {
    a.answerEl.innerHTML += '<div class="tool-note" style="color:var(--danger)">■ 已停止生成（部分内容）</div>';
  }
  if (a.thinkingBox) a.thinkingBox.open = false;
}

function renderAssistantMessage(msg, streaming) {
  const a = createAssistantBubble();
  const reasoning = msg.reasoning_content || "";
  const answer = msg.content || "";
  a.done = true;
  if (reasoning) {
    a.thinkingRaw = reasoning;
    a.thinkingContent.textContent = reasoning;
    showThinkingBox(a);
    a.thinkingBox.open = false;
  }
  a.answerRaw = answer;
  a.answerEl.innerHTML = renderMarkdown(answer);
  if (streaming) {
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    a.answerEl.appendChild(cursor);
  }
}

// ---------------- Markdown 迷你渲染 ----------------

function inlineMarkdown(escaped) {
  let s = escaped;
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/^###### (.*)$/gm, "<h6>$1</h6>")
       .replace(/^##### (.*)$/gm, "<h5>$1</h5>")
       .replace(/^#### (.*)$/gm, "<h4>$1</h4>")
       .replace(/^### (.*)$/gm, "<h3>$1</h3>")
       .replace(/^## (.*)$/gm, "<h2>$1</h2>")
       .replace(/^# (.*)$/gm, "<h1>$1</h1>");
  s = s.replace(/^&gt; (.*)$/gm, "<blockquote>$1</blockquote>");
  s = listify(s);
  s = s.replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>");
  return "<p>" + s + "</p>";
}

function listify(s) {
  const lines = s.split("\n");
  const out = [];
  let list = null;
  for (const line of lines) {
    const um = line.match(/^[-*+] (.+)$/);
    const om = line.match(/^\d+[.)] (.+)$/);
    if (um || om) {
      const tag = um ? "ul" : "ol";
      if (list !== tag) {
        if (list) out.push("</" + list + ">");
        list = tag;
        out.push("<" + tag + ">");
      }
      out.push("<li>" + (um || om)[1] + "</li>");
    } else {
      if (list) { out.push("</" + list + ">"); list = null; }
      out.push(line);
    }
  }
  if (list) out.push("</" + list + ">");
  return out.join("\n");
}

function renderMarkdown(text) {
  const escaped = escapeHtml(text);
  const parts = [];
  const re = /```(\w*)\n([\s\S]*?)```/g;
  let last = 0, m;
  while ((m = re.exec(escaped))) {
    parts.push(inlineMarkdown(escaped.slice(last, m.index)));
    const lang = m[1] || "text";
    parts.push(
      '<pre><button class="copy-btn" onclick="copyCode(this)">复制</button><code class="lang-' + escapeHtml(lang) + '">' +
      m[2] + "</code></pre>"
    );
    last = re.lastIndex;
  }
  parts.push(inlineMarkdown(escaped.slice(last)));
  return parts.join("");
}

window.copyCode = function (btn) {
  const code = btn.parentElement.querySelector("code");
  navigator.clipboard.writeText(code.textContent).then(() => {
    btn.textContent = "已复制";
    setTimeout(() => (btn.textContent = "复制"), 1200);
  });
};

// ---------------- 附件 ----------------

function updateAttachments() {
  const box = $("#attachment-preview");
  box.innerHTML = "";
  state.attachments.forEach((a, i) => {
    const chip = document.createElement("div");
    chip.className = "attach-chip";
    if (a.type === "image") {
      const img = document.createElement("img");
      img.src = a.url;
      chip.appendChild(img);
    } else {
      const icon = document.createElement("span");
      icon.textContent = "🎬";
      chip.appendChild(icon);
    }
    const info = document.createElement("span");
    info.className = "chip-info";
    info.textContent = a.name + " · " + (a.sizeMB > 0 ? a.sizeMB + "MB" : "<1MB");
    chip.appendChild(info);
    const rm = document.createElement("span");
    rm.className = "chip-remove";
    rm.textContent = "✕";
    rm.addEventListener("click", () => {
      state.attachments.splice(i, 1);
      updateAttachments();
    });
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

// ---------------- SSE 流式处理 ----------------

async function consumeSSE(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      handleSSE(block);
    }
  }
}

function handleSSE(block) {
  let event = "message", dataStr = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
  }
  let data;
  try { data = JSON.parse(dataStr); } catch (e) { return; }

  switch (event) {
    case "status":
      if (data.session_id) state.sessionId = data.session_id;
      setSubtitle(data.state === "queued" ? "排队等待生成…" : "生成中…");
      break;
    case "thinking":
      appendThinking(data.text || "");
      break;
    case "answer":
      appendAnswer(data.text || "");
      break;
    case "tool":
      showToolNote(data);
      break;
    case "done":
      setSubtitle(data.aborted ? "已停止" : "完成 · " + (data.time_s || 0) + "s");
      finishAssistant(!!data.aborted);
      break;
    case "error":
      showToast("错误：" + (data.message || "未知错误"));
      setSubtitle("出错了");
      const errA = state.currentAssistant;
      if (errA) {
        errA.done = true;
        const c = errA.answerEl.querySelector(".cursor");
        if (c) c.remove();
        errA.answerEl.innerHTML += '<div class="tool-note" style="color:var(--danger)">出错了：' + escapeHtml(data.message || "未知错误") + "</div>";
      }
      break;
  }
}

function showToolNote(data) {
  const a = state.currentAssistant;
  if (!a) return;
  const note = document.createElement("div");
  note.className = "tool-note";
  const result = data.result && data.result.error ? "错误: " + data.result.error : JSON.stringify(data.result);
  note.textContent = "🔧 调用 " + data.tool + "(" + JSON.stringify(data.args || {}).slice(0, 120) + ") → " + String(result).slice(0, 200);
  a.bubble.appendChild(note);
  scrollBottom();
}

// ---------------- 发送 ----------------

async function sendMessage() {
  if (state.streaming) return;
  const text = $("#input").value.trim();
  if (!text && state.attachments.length === 0) return;

  const attachments = state.attachments;
  renderUserMessage([
    ...attachments.map((a) => ({ type: a.type, url: a.url, name: a.name })),
    ...(text ? [{ type: "text", text }] : []),
  ]);
  hideWelcome();
  state.attachments = [];
  updateAttachments();
  $("#input").value = "";
  autoResize();

  state.currentAssistant = createAssistantBubble();
  state.streaming = true;
  $("#btn-send").hidden = true;
  $("#btn-stop").hidden = false;
  state.controller = new AbortController();
  const genStart = Date.now();
  state.elapsedTimer = setInterval(() => {
    setSubtitle("生成中… 已用 " + Math.round((Date.now() - genStart) / 1000) + " 秒（CPU 较慢请耐心等待）");
  }, 1000);

  const payload = {
    session_id: state.sessionId,
    message: text,
    attachments: attachments.map((a) => ({
      type: a.type,
      name: a.name,
      mime: a.mime,
      data: a.data,
    })),
    params: { ...params },
  };

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: state.controller.signal,
    });
    if (!resp.ok) {
      let msg = "请求失败";
      try { msg = (await resp.json()).error || msg; } catch (e) { /* ignore */ }
      throw new Error(msg);
    }
    await consumeSSE(resp.body);
  } catch (err) {
    if (err.name !== "AbortError") {
      showToast("发送失败：" + err.message);
      setSubtitle("出错了");
      finishAssistant(true);
    }
  } finally {
    state.streaming = false;
    if (state.elapsedTimer) { clearInterval(state.elapsedTimer); state.elapsedTimer = null; }
    $("#btn-send").hidden = false;
    $("#btn-stop").hidden = true;
    if (state.currentAssistant && !state.currentAssistant.done) finishAssistant(false);
    state.currentAssistant = null;
    refreshSessions(state.sessionId);
    checkStatus();
  }
}

// ---------------- 事件绑定 ----------------

$("#new-chat").addEventListener("click", newChat);
$("#btn-send").addEventListener("click", sendMessage);
$("#btn-attach").addEventListener("click", () => $("#file-input").click());

$("#file-input").addEventListener("change", (e) => {
  for (const f of e.target.files) {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    let type = null;
    if (f.type.startsWith("image/") || ["png", "jpg", "jpeg", "webp", "bmp", "gif"].includes(ext)) type = "image";
    else if (f.type.startsWith("video/") || ["mp4", "webm", "mov", "mkv", "avi", "m4v"].includes(ext)) type = "video";
    if (!type) { showToast("仅支持图片或视频文件：" + f.name); continue; }
    const max = type === "image" ? 10 * 1024 * 1024 : 200 * 1024 * 1024;
    if (f.size > max) { showToast(f.name + " 超出大小限制"); continue; }
    const url = URL.createObjectURL(f);
    const reader = new FileReader();
    reader.onload = () => {
      const data = String(reader.result).split(",")[1] || "";
      state.attachments.push({
        type, name: f.name, mime: f.type, url, data,
        sizeMB: Math.round((f.size / 1048576) * 10) / 10,
      });
      updateAttachments();
    };
    reader.readAsDataURL(f);
  }
  e.target.value = "";
});

$("#input").addEventListener("input", autoResize);
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

$("#btn-stop").addEventListener("click", async () => {
  try {
    if (state.sessionId) {
      await fetch("/api/chat/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId }),
      });
    }
  } catch (e) { /* ignore */ }
  if (state.controller) state.controller.abort();
});

$("#btn-clear").addEventListener("click", async () => {
  if (!state.sessionId) { showToast("当前没有会话"); return; }
  if (!confirm("确定清空当前会话的所有消息？")) return;
  try {
    await api("/api/sessions/" + state.sessionId + "/clear", { method: "POST" });
    clearMessages();
    showWelcome();
    $("#chat-title").textContent = "新对话";
    refreshSessions(state.sessionId);
  } catch (e) {
    showToast("清空失败：" + e.message);
  }
});

$("#btn-delete").addEventListener("click", async () => {
  if (!state.sessionId) { showToast("当前没有会话"); return; }
  if (!confirm("确定删除当前会话（含历史与附件）？此操作不可恢复。")) return;
  try {
    await api("/api/sessions/" + state.sessionId, { method: "DELETE" });
    newChat();
  } catch (e) {
    showToast("删除失败：" + e.message);
  }
});

// ---------------- 参数弹窗 ----------------

function updateThinkState() {
  const el = $("#think-state");
  if (el) el.textContent = params.thinking ? "🧠 思考模式：开" : "思考模式：关（直答）";
}

function syncParamsUI() {
  $("#p-system").value = params.system_prompt || "";
  $("#p-thinking").checked = !!params.thinking;
  $("#p-tools").checked = !!params.tools;
  $("#p-temp").value = params.temperature;
  $("#p-temp-v").textContent = Number(params.temperature).toFixed(2);
  $("#p-topp").value = params.top_p;
  $("#p-topp-v").textContent = Number(params.top_p).toFixed(2);
  $("#p-topk").value = params.top_k;
  $("#p-topk-v").textContent = params.top_k;
  $("#p-max").value = params.max_new_tokens;
  updateThinkState();
}

function collectParams() {
  params.system_prompt = $("#p-system").value;
  params.thinking = $("#p-thinking").checked;
  params.tools = $("#p-tools").checked;
  params.temperature = parseFloat($("#p-temp").value);
  params.top_p = parseFloat($("#p-topp").value);
  params.top_k = parseInt($("#p-topk").value, 10);
  params.max_new_tokens = parseInt($("#p-max").value, 10) || 2048;
  saveParams();
  updateThinkState();
}

$("#btn-params").addEventListener("click", () => {
  syncParamsUI();
  $("#params-modal").classList.remove("hidden");
});
$("#params-close").addEventListener("click", () => {
  collectParams();
  $("#params-modal").classList.add("hidden");
});
$("#params-modal").addEventListener("click", (e) => {
  if (e.target === $("#params-modal")) {
    collectParams();
    $("#params-modal").classList.add("hidden");
  }
});
$("#p-temp").addEventListener("input", () => ($("#p-temp-v").textContent = Number($("#p-temp").value).toFixed(2)));
$("#p-topp").addEventListener("input", () => ($("#p-topp-v").textContent = Number($("#p-topp").value).toFixed(2)));
$("#p-topk").addEventListener("input", () => ($("#p-topk-v").textContent = $("#p-topk").value));

// ---------------- 状态轮询 ----------------

async function checkStatus() {
  try {
    const info = await api("/api/status");
    const el = $("#model-status");
    const text = $("#status-text");
    el.className = "model-status " + info.state;
    if (info.state === "ready") {
      text.textContent = "就绪 · " + info.device.toUpperCase() + " · 4B";
      el.title = "模态：文本/图片/视频 · 特性：" + info.features.join("、");
    } else if (info.state === "loading") {
      text.textContent = "模型加载中…";
    } else if (info.state === "error") {
      text.textContent = "加载失败";
      el.title = info.error || "";
    } else {
      text.textContent = "未加载 · 发送消息后自动加载";
    }
  } catch (e) {
    $("#status-text").textContent = "服务未连接";
  }
}

// ---------------- 初始化 ----------------

syncParamsUI();
checkStatus();
refreshSessions(null);
setInterval(checkStatus, 3000);
$("#input").focus();
