# VH Backend

面向 Electron 的本地 FastAPI 服务，负责视频上传、任务状态、SQLite 持久化、Agent 调用和受控媒体读取。GPU 推理逻辑仍只存在于 `vh-agent`。

## 安装与运行

所有 Python 依赖使用 uv 和清华 PyPI 镜像：

```powershell
cd vh-backend
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync --group dev
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

也可以直接在 `vh-frontend/` 执行 `npm run dev` 启动整套开发环境。

## Agent 准备

后端通过稳定的 `vh run` CLI 调用相邻的 `vh-agent`。运行时使用 `--no-sync`，避免任务执行过程中修改 Agent 环境或临时下载依赖，因此需要预先完成一次安装：

```powershell
cd ..\vh-agent
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync --extra enhanced
```

模型路径、SiliconFlow 密钥和设备配置继续由 `vh-agent/.env` 管理；后端不读取或复制这些内部配置。

## 数据目录

默认数据根目录是 `../vh-frontend/video-data`：

```text
video-data/
├── tasks.sqlite3
├── cache/
└── jobs/
    └── <job_id>/
        ├── source/original.mp4
        ├── agent.log
        ├── result.json
        └── clips/*.mp4
```

`agent.log`、缓存和物理路径都不会通过 API 返回给前端。

## API

- `GET /health`：健康检查。
- `POST /api/jobs`：multipart 上传；字段为 `file`、`job_id`、`language`。
- `GET /api/jobs`：列出本机任务。
- `GET /api/jobs/{job_id}`：查询状态和公开结果。
- `DELETE /api/jobs/{job_id}`：删除已完成或失败的任务及其本地文件。
- `GET /api/jobs/{job_id}/source`：流式读取原片。
- `GET /api/jobs/{job_id}/highlights/{highlight_id}/video`：读取高光视频。
- `PATCH /api/jobs/{job_id}/highlights/{highlight_id}`：保存 `pending`、`accepted`、`rejected` 或 `revised` 复核状态。

任务状态为 `queued → processing → completed`，失败时为 `failed`。当前 MVP 使用单 Agent 工作线程，避免多个 GPU 任务并发争用显存。

## 配置

复制 `.env.example` 为本地 `.env` 后可调整：

- `VH_STORAGE_DIR`：数据根目录；打包时应与 Electron 的 `<安装目录>/video-data` 一致。
- `VH_AGENT_ROOT`：`vh-agent` 根目录。
- `VH_AGENT_UV_EXECUTABLE`：uv 可执行文件。
- `VH_AGENT_TIMEOUT_SEC`：单任务超时秒数。
- `VH_MAX_UPLOAD_BYTES`：上传上限，默认 20 GB。
- `VH_ALLOWED_ORIGINS`：逗号分隔的 Electron/Vite 来源。

## 验证

```powershell
uv run ruff check app tests
uv run pytest -q
```

## 前端演示数据

可用 FFmpeg 生成两个 18 秒画布原片，并为每个原片生成三个 4 秒高光片段。脚本只创建固定的 demo 任务，已存在时会跳过，不会覆盖其他本地任务：

```powershell
uv run python scripts/seed_demo_data.py --ffmpeg "D:\FFmpeg\ffmpeg-2023-12-23-git-f5f414d9c4-full_build\bin\ffmpeg.exe"
```

生成后启动前端，在“任务归档”中可打开 `城市节拍_原片.mp4` 和 `新品发布_原片.mp4`，并验证原片播放、三个高光片段切换和人工复核。
