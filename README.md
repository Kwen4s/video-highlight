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

## 启动

服务器侧先准备 Agent：

```powershell
cd vh-agent
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync --extra enhanced
```

随后启动后端：

```powershell
cd ..\vh-backend
Copy-Item .env.example .env
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync --group dev
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

本机只启动桌面前端：

```powershell
cd vh-frontend
nvm use
npm install
Copy-Item .env.example .env.local
npm run dev
```

前端默认访问 `http://localhost:8000`。服务器部署后，在 `vh-frontend/.env.local` 中将 `VITE_API_BASE_URL` 改为服务器 IP 或域名。详细配置分别见 [前端说明](vh-frontend/README.md)和[后端说明](vh-backend/README.md)。
