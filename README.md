# Video Highlight

短剧高光检测桌面应用。仓库中的三个目录保持独立部署边界：

```text
video-highlight/
├── vh-frontend/   # Electron 上传、状态、播放器与人工复核
├── vh-backend/    # FastAPI 任务编排、视频保存与结果持久化
└── vh-agent/      # 高光检测与离线评测（本次未修改）
```

## 当前 MVP

视频导入链路已经接通：

```text
Electron 创建任务目录
  → 浏览器进程以 multipart 流上传视频
  → FastAPI 分块保存原片并写入 SQLite
  → 后台进程调用 vh-agent 的 `vh run` CLI
  → 前端轮询任务状态
  → 播放高光并保存采用/排除结果
```

本地开发时，任务数据统一保存在 `vh-frontend/video-data/`。前端只接触任务 ID、公开状态和媒体 API，不读取 Agent trace、服务端路径或模型内部字段。

## 启动

先按 [vh-agent/README.md](vh-agent/README.md) 准备模型、FFmpeg 和环境变量，再为桌面集成创建 Agent 的 uv 环境：

```powershell
cd vh-agent
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync --extra enhanced
```

随后启动完整桌面开发环境：

```powershell
cd ..\vh-frontend
nvm use
npm install
npm run dev
```

`npm run dev` 会并行启动 Vite、FastAPI 和 Electron。更多后端配置与 API 见 [vh-backend/README.md](vh-backend/README.md)。
