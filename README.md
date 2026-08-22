# Qwen3.5-4B 本地多模态助手

在本地调用 `E:\LLMModels\Qwen3.5-4B`（视觉语言模型，4B 参数），提供：

- 纯文本对话（中文/多语言）
- 图片理解：单图 / 多图 / 图文混排
- 视频理解：上传本地视频，自动抽帧
- 多轮对话历史：会话内记忆 + 本地持久化（跨重启可恢复）
- 思考模式（`<think>`）与直答模式切换
- 工具调用演示（计算器 / 当前时间）
- 流式输出（SSE）、采样参数调节、自定义系统提示词

> 注意：这是 Qwen3.5 的 **Vision-Language** 版本，支持 文本/图片/视频，**不支持音频输入**（没有音频编码器）。

## 目录结构

```
E:\LLMpractice
├── app.py            # Flask Web 服务（SSE 流式 + 会话持久化 + API）
├── qwen_engine.py    # 模型引擎：加载、多模态输入、流式生成、历史裁剪、工具调用
├── cli_chat.py       # 命令行对话工具
├── smoke_test.py     # 冒烟测试（文本/图片）
├── static/           # 网页前端（index.html / style.css / app.js）
├── requirements.txt  # Python 依赖
├── run.bat           # 一键启动脚本（自动优先使用 test_env 的 Python）
└── sessions/         # 会话历史（自动生成，JSON + 图片/视频附件）
```

## 环境与依赖

推荐使用已配置好的 conda 环境 `test_env`（Python 3.13，依赖已安装）：

```bat
E:\conda\condaenvs\test_env\python.exe -m pip install -r requirements.txt
```

核心依赖：`torch`（CPU 版）、`transformers>=5.15`、`accelerate`、`av`（视频解码）、`pillow`、`flask`。

> 提示：环境里**不要安装 scikit-learn**。本机 base 环境曾因 numpy/scipy 旧版 ABI 导致 transformers 导入失败，test_env 是干净环境，没有这个问题。

## 启动网页界面

### 方式一：双击 `run.bat`

脚本会自动优先使用 `E:\conda\condaenvs\test_env\python.exe` 启动，并预加载模型。

### 方式二：命令行

```bat
cd /d "E:\LLMpractice"
E:\conda\condaenvs\test_env\python.exe app.py --preload --port 7860
```

然后浏览器打开 **http://127.0.0.1:7860**。

可选参数：

```bat
--port 7860        # 端口
--model 路径       # 覆盖模型目录（默认 E:\LLMModels\Qwen3.5-4B）
--preload          # 启动时后台预加载模型（推荐）
```

模型目录也可用环境变量 `QWEN_MODEL_PATH` 指定。

## 网页使用要点

1. **发送消息**：底部输入框输入文字，回车发送；`Shift+Enter` 换行。
2. **图片**：点 📎 选择图片（png/jpg/jpeg/webp/bmp/gif，≤10MB），可一次多张，与文字一起发送。
3. **视频**：点 📎 选择视频（mp4/webm/mov 等，≤200MB），模型会自动抽帧理解。建议用 30 秒以内的短视频，CPU 上长视频会很慢。
4. **多轮记忆**：同一个会话内模型记得之前聊过的内容；会话自动保存到 `sessions/`，刷新浏览器、重启服务后左侧列表仍可恢复。点「清空」只清消息，点「删除」连历史带附件一起删。
5. **参数面板**（右上 ⚙）：
   - 系统提示词：设定角色/语气，逐会话保存
   - 思考模式：默认开启（模型先想再答，效果更好但更慢）；关闭后直答
   - 工具调用：开启后可用「计算器 / 当前时间」两个演示工具
   - 温度 / Top-P / Top-K / 最大新 tokens
6. **停止**：生成过程中点 ■ 停止（会保留已生成的部分）。
7. **速度**：本机无 GPU，纯 CPU 推理大约每秒 0.5–1 个 token，长回答请耐心等待；首次加载模型约 5–10 秒。

## 命令行使用

```bat
cd /d "E:\LLMpractice"
E:\conda\condaenvs\test_env\python.exe cli_chat.py
E:\conda\condaenvs\test_env\python.exe cli_chat.py --image photo.jpg --video demo.mp4
E:\conda\condaenvs\test_env\python.exe cli_chat.py --no-thinking --system "你是中文助手"
```

对话内命令：

```
/help            查看帮助
/clear           清空历史
/think /nothink  开启/关闭思考模式
/image 路径      附加图片
/video 路径      附加视频
/exit            退出
```

## 冒烟测试

```bat
E:\conda\condaenvs\test_env\python.exe smoke_test.py                 # 纯文本
E:\conda\condaenvs\test_env\python.exe smoke_test.py --image 1.jpg   # 图片
```

## API 简表（供二次开发）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 网页前端 |
| GET | `/api/status` | 模型/设备/能力状态 |
| POST | `/api/chat` | 聊天（SSE 流式；`session_id`、`message`、`attachments`、`params`） |
| POST | `/api/chat/stop` | 停止当前会话生成 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/<id>` | 会话详情（含历史与附件 URL） |
| POST | `/api/sessions/<id>/clear` | 清空会话消息 |
| DELETE | `/api/sessions/<id>` | 删除会话 |
| GET | `/media/<id>/<file>` | 附件文件 |

## 已知限制

- CPU 推理较慢；无 GPU 时建议控制回答长度（最大新 tokens 默认 2048）。
- 内存：模型加载后约占用 10–14GB。本机 16.8GB 内存时**运行前请关闭浏览器/IDE 等大内存程序**，
  否则视觉编码可能内存不足（现在会红字提示，不再静默无输出）。
- 内存保护（可用环境变量调整）：
  - `QWEN_MAX_IMAGE_PIXELS`：图片最大像素数，默认 1500000（约 150 万像素，大图自动缩小）
  - `QWEN_MAX_VIDEO_EDGE`：视频帧最长边，默认 768（高清视频自动缩小后再理解）
- 视频按 2fps 均匀抽帧、最多 48 帧，长视频只能看到抽样内容。
- 工具调用为演示实现（计算器、当前时间），不支持任意 Python 代码执行。
- 前端 Markdown 为本地轻量渲染，不包含数学公式渲染（KaTeX）。

## 常见问题

**图片/视频发出去后没有输出？**

1. **旧缓存**：先用 Ctrl+F5 强制刷新页面，或重启服务后再打开。若浏览器仍加载旧版前端，发送会被中断且无任何提示（现已对前端文件禁用缓存，不会再出现）。
2. **多个服务实例**：确保只运行一个 `python app.py`，否则浏览器可能连到旧实例。清理后重启：

   ```powershell
   Get-NetTCPConnection -LocalPort 7860 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
   ```

3. **思考模式较慢**：默认开启思考，模型会先输出灰色思考内容，CPU 上等待几分钟属正常。若“最大新 tokens”被思考占满，界面会提示“只有思考没有正式回答”，把该值调大或关闭思考模式即可。
4. **出现红色“出错了”**：多为视频编码不受支持或文件损坏，建议换 H.264 编码的 mp4 短视频、小图先单独测试。
5. **红字提示“内存不足”**：模型约占 10–14GB 内存，运行前关闭浏览器、IDE 等其他大内存程序；图片/视频已自动降像素保护，仍不足可减小“最大新 tokens”或换更小的文件。