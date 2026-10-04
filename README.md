# Video Highlight

短剧高光检测桌面应用。各目录保持独立部署边界：

```text
video-highlight/
├── vh-frontend/   # Electron 任务归档、审阅、编辑与导出
├── vh-backend/    # FastAPI 持久化任务、媒体存储与会话编排
├── vh-agent/      # 原生视频 ReAct Agent、本地候选工具与离线评测
└── vh-data/       # 视频标注、模型复核、训练数据快照与小模型反馈
```

检测小模型的数据生产见 [vh-data/README.md](vh-data/README.md)。标注网关单独配置，按短剧划分训练、验证、测试；模型复核后导出数据，漏检反馈进入下一轮复核。

## 数据流

```text
Electron 以 multipart 将视频上传到 FastAPI
  → FastAPI 将任务记录和原片写入持久化目录
  → FastAPI 调用唯一的 vh run，Agent 检查视频并复核真实高光 MP4
  → 前端轮询覆盖进度、complete/partial 状态和高光 JSON
  → 用户预览已复核成片、采用、排除或直接拖动原片时间轴调整高光
  → Electron 可将高光直接导出为新 MP4，或追加片尾广告后导出
  → 后端对话 Agent 将自然语言编译为类型化计划并执行受控查询或编辑
  → 任务、原片、结果与会话持续保留，仅在用户明确确认后删除
```

前端只接触任务 ID、公开状态、高光时间段和会话回复，不读取 Agent trace、服务端物理路径或模型内部字段。后端持久保存原片、任务状态、结果和对话；没有会话有效期或定时任务清理。对话上下文最多保留最近 20 轮，用于解析“这段”“再短一点”等连续指令。

后端对话 Agent 支持查询候选及理由，调整时间边界，删除、拆分、合并高光，修改标题或说明，以及撤销。明确的基础编辑指令可以离线执行；要启用口语化规划，在 `vh-backend/.env` 中配置 OpenAI 兼容接口：

```dotenv
VH_CHAT_API_KEY=your-api-key
VH_CHAT_BASE_URL=https://api.siliconflow.cn/v1
VH_CHAT_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

模型只生成 `EditPlan`、`QueryPlan`、`UndoPlan` 或 `ClarifyPlan`，实际结果修改和边界校验由后端确定性工具完成。

工作区中已采用的高光可进入“高光广告编排”。桌面端内置“悬念追更”“霓虹下载”“丝绒扫码”三个短剧片尾预设，支持修改文案、时长和二维码，也可导入最长 60 秒的自有广告视频。合成在 Electron 本机通过随应用安装的 FFmpeg 完成，输出为 H.264/AAC MP4；原片和高光时间范围不会被改写。

两个内置 Demo 也支持对话。演示视频从本机播放；Demo 会话与普通任务一样没有过期时间。

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

日常启动或重启直接运行脚本，不会执行 `uv sync`，也不会重新下载环境：

```bash
cd /home/tnx/video-highlight/vh-backend
./scripts/restart_backend.sh
```

脚本会同时检查 PID 文件和 8777 实际监听进程：先优雅停止旧进程，15 秒未退出时强制结束；确认端口释放后，直接使用现有 `.venv/bin/python` 启动。只有新 PID 确实监听 8777 且健康检查成功，才会写入 `runtime/backend.pid`。

正常时健康检查返回：

```json
{"status":"ok"}
```

查看运行日志：

```bash
tail -f /home/tnx/video-highlight/vh-backend/runtime/backend.log
```

完全停止后端：

```bash
cd /home/tnx/video-highlight/vh-backend
./scripts/stop_backend.sh
```

停止脚本会同时查找当前项目的 Uvicorn 进程、PID 文件记录和 8777 实际监听者，确认端口释放后才返回成功。重复执行是安全的，不会删除任务目录、数据库、日志或 `.venv`。

### 重新部署/重启后端

代码或配置更新后仍运行同一个脚本。它只会重启 API 服务，不会删除 `vh-backend/runtime` 中的任务数据；只有依赖文件变化时才需要重新执行 `uv sync`。

```bash
cd /home/tnx/video-highlight/vh-backend
./scripts/restart_backend.sh
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

新版检测架构与运行方式见 [Agent README](vh-agent/README.md)。运行中断后通过任务的“继续分析”恢复检查点。
