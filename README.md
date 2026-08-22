# Video Highlight

短剧高光检测桌面应用。三个目录保持独立部署边界：

```text
video-highlight/
├── vh-frontend/   # Electron 本地原片、任务归档、审阅与编辑对话
├── vh-backend/    # FastAPI 临时任务、会话编排与自动清理
└── vh-agent/      # 高光检测与离线评测（本次未修改）
```

## 数据流

```text
Electron 将原片和任务草稿永久保存到本机
  → 以 multipart 将视频上传到 FastAPI 临时目录
  → FastAPI 调用 vh-agent（不导出高光 MP4）
  → 前端轮询并保存高光时间段 JSON
  → 用户在本地原片上预览、采用或排除高光
  → 后端对话 Agent 将自然语言编译为类型化计划并执行受控查询或编辑
  → 会话过期后后端清理视频、缓存、结果与会话记录
```

前端只接触任务 ID、公开状态、高光时间段和会话回复，不读取 Agent trace、服务端路径或模型内部字段。后端不会永久保存用户视频；本地原片、结果、复核状态和完整对话历史由 Electron 管理。限时会话内，后端仅保留最近 20 轮上下文，用于解析“这段”“再短一点”等连续指令，并随临时任务一起清理。

后端对话 Agent 支持查询候选及理由，调整时间边界，删除、拆分、合并高光，修改标题或说明，以及撤销。明确的基础编辑指令可以离线执行；要启用口语化规划，在 `vh-backend/.env` 中配置 OpenAI 兼容接口：

```dotenv
VH_CHAT_API_KEY=your-api-key
VH_CHAT_BASE_URL=https://api.siliconflow.cn/v1
VH_CHAT_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

模型只生成 `EditPlan`、`QueryPlan`、`UndoPlan` 或 `ClarifyPlan`，实际结果修改和边界校验由后端确定性工具完成。

两个内置 Demo 也支持对话。用户从任务归档打开 Demo 时，Electron 会把公开高光结果注册为后端临时会话；演示视频不会上传，仍从本机播放。Demo 会话与普通任务一样在每次消息后续期。

## 部署、启动与停止

### Ubuntu 服务器部署后端

以下命令以 `/home/tnx/video-highlight` 为项目目录，后端监听 `8777`。部署前确认 `vh-agent/.env` 中的 `VH_ASR_MODEL` 指向实际的 Whisper `snapshots/<revision>` 目录，例如：

```dotenv
VH_ASR_MODEL=/data1/video-highlight-models/faster-whisper/models--Systran--faster-whisper-large-v3/snapshots/<revision>
```

首次部署或依赖发生变化时执行：

```bash
cd /home/tnx/video-highlight/vh-agent
unset VIRTUAL_ENV
uv sync --extra enhanced

cd ../vh-backend
unset VIRTUAL_ENV
uv sync --group dev
mkdir -p runtime
```

启动后端并保存 PID 和日志：

```bash
cd /home/tnx/video-highlight/vh-backend
nohup uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8777 \
  > runtime/backend.log 2>&1 &
echo $! > runtime/backend.pid
curl -fsS http://127.0.0.1:8777/health
```

正常时健康检查返回：

```json
{"status":"ok"}
```

查看运行日志：

```bash
tail -f /home/tnx/video-highlight/vh-backend/runtime/backend.log
```

停止后端：

```bash
cd /home/tnx/video-highlight/vh-backend
if [ -f runtime/backend.pid ]; then
  pid=$(cat runtime/backend.pid)
  kill "$pid" 2>/dev/null || true
  rm -f runtime/backend.pid
fi
```

如果服务是前台启动的，直接在对应终端按 `Ctrl+C`；如果没有 PID 文件，可先通过 `ss -ltnp | grep ':8777'` 找到进程后停止该进程。

### 重新部署/重启后端

代码和配置更新后，按以下顺序执行。重启只会停止并重新启动 API 服务，不会删除 `vh-backend/runtime` 中的任务数据。

```bash
cd /home/tnx/video-highlight/vh-backend
if [ -f runtime/backend.pid ]; then
  kill "$(cat runtime/backend.pid)" 2>/dev/null || true
  rm -f runtime/backend.pid
fi

cd /home/tnx/video-highlight/vh-backend
unset VIRTUAL_ENV
mkdir -p runtime
nohup .venv/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 --port 8777 \
  > runtime/backend.log 2>&1 &
echo $! > runtime/backend.pid
curl -fsS http://127.0.0.1:8777/health
```

若重启后任务仍失败，查看对应任务目录中的 `agent.log`，或先查看最新后端日志：

```bash
tail -n 200 /home/tnx/video-highlight/vh-backend/runtime/backend.log
```

### 启动桌面前端

前端在用户本机运行，不需要部署到 Ubuntu。先在 `vh-frontend/.env.local` 设置服务器地址，例如：

```dotenv
VITE_API_BASE_URL=http://服务器IP:8777
```

开发模式启动：

```powershell
cd vh-frontend
nvm use
npm install
npm run dev
```

开发模式下前端和 Electron 会由同一个命令启动，停止时在该终端按 `Ctrl+C`。

打包模式启动：

```powershell
cd vh-frontend
npm install
npm run build
npm start
```

前端默认访问 `http://122.193.22.119:8777`。服务器部署后，在 `vh-frontend/.env.local` 中将 `VITE_API_BASE_URL` 改为服务器 IP 或域名。详细配置分别见 [前端说明](vh-frontend/README.md) 和 [后端说明](vh-backend/README.md)。
