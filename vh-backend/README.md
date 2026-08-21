# VH Backend

面向 Electron 的 FastAPI 服务，负责视频上传、临时任务与会话持久化、`vh-agent` 调用和高光对话编排。它与 `vh-agent` 部署在服务器侧；GPU 推理逻辑仍只存在于 `vh-agent`。

## 安装与运行

所有 Python 依赖使用 uv 和清华 PyPI 镜像：

```powershell
cd vh-backend
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync --group dev
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

服务器需要接受其他机器的 Electron 连接时，监听所有网卡：

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

此时前端将 `VITE_API_BASE_URL` 配置为服务器的实际 IP 或域名，不要配置为 `0.0.0.0`。`vh-frontend` 的 `npm run dev` 只启动桌面前端，不会代为启动本服务。

## 检测 Agent 准备

后端通过稳定的 `vh run` CLI 调用相邻的 `vh-agent`。运行时使用 `--no-sync`，因此需要预先完成一次安装：

```powershell
cd ..\vh-agent
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync --extra enhanced
```

模型路径、检测模型密钥和设备配置继续由 `vh-agent` 自己管理；后端不读取或复制其内部配置。

## 对话 Agent

右侧对话由后端的受约束单 Agent 处理：

```text
自然语言
  → EditPlan / QueryPlan / UndoPlan / ClarifyPlan
  → 确定性查询或编辑工具
  → 边界与稳定 ID 校验
  → 带版本号的结构化结果
```

明确的基础编辑指令不依赖模型。配置 OpenAI 兼容接口后，Agent 还能理解口语化指代，并规划查询、拆分和合并等操作：

```dotenv
VH_CHAT_API_KEY=your-api-key
VH_CHAT_BASE_URL=https://api.siliconflow.cn/v1
VH_CHAT_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

模型只负责生成计划，不能直接写结果、访问文件或执行 shell。Agent 只接收公开高光摘要、当前选中 `highlight_id` 和最近 6 轮对话；服务端最多临时保存 20 轮上下文。

## 数据目录

默认数据根目录是 `vh-backend/runtime`：

```text
runtime/
├── tasks.sqlite3
└── jobs/
    └── <job_id>/
        ├── source/original.mp4
        ├── agent.log
        ├── result.json
        └── cache/
```

`agent.log`、缓存、对话计划和物理路径都不会通过 API 返回给前端。任务会话过期后，后端清理上传视频、缓存、结果和临时对话；Electron 本机保存原片、公开结果、复核状态和完整对话历史。

## API

- `GET /health`：健康检查。
- `POST /api/jobs`：multipart 上传；字段为 `file`、`job_id`、`language`。
- `GET /api/jobs`：列出仍处于服务端临时会话中的任务。
- `GET /api/jobs/{job_id}`：查询任务状态和公开结果。
- `POST /api/demo-jobs/{job_id}/session`：为白名单内置 Demo 创建或续开临时对话会话，不上传视频。
- `POST /api/jobs/{job_id}/messages`：发送查询或编辑请求；请求携带 `message`、`revision` 和可选的 `selected_highlight_id`。
- `DELETE /api/jobs/{job_id}`：提前清理已完成或失败的临时任务。

任务状态为 `queued → processing → completed`，失败时为 `failed`。当前使用单检测工作线程，避免多个 GPU 任务并发争用显存。结果修改使用乐观版本号，过期版本返回 `409`；会话过期返回 `410`。

## 配置

复制 `.env.example` 为本地 `.env` 后可调整：

- `VH_STORAGE_DIR`：服务端临时数据根目录，默认 `./runtime`。
- `VH_AGENT_ROOT`：`vh-agent` 根目录。
- `VH_AGENT_UV_EXECUTABLE`：uv 可执行文件。
- `VH_AGENT_TIMEOUT_SEC`：检测任务超时秒数。
- `VH_MAX_UPLOAD_BYTES`：上传上限，默认 20 GB。
- `VH_EDIT_SESSION_TTL_SEC`：完成任务的可编辑会话时长。
- `VH_CHAT_*`：对话模型的密钥、OpenAI 兼容地址、模型、超时和重试次数。
- `VH_CLEANUP_INTERVAL_SEC`：过期任务清理周期。
- `VH_ORPHAN_JOB_TTL_SEC`：异常中断任务的保留时长。
- `VH_ALLOWED_ORIGINS`：逗号分隔的 Electron/Vite 页面来源。

## CORS 配置

默认配置允许本地 Vite 与打包后的 Electron 页面调用 API：

```dotenv
VH_ALLOWED_ORIGINS=http://127.0.0.1:5173,http://localhost:5173,null
```

这里填写的是前端页面来源，不是后端 API 地址。`null` 用于打包后从 `file://` 加载的 Electron 页面，不应替换成 `*`。若服务暴露到不可信网络，还需要在网关层增加 HTTPS、身份认证和访问控制。

## 验证

```powershell
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv run ruff check app tests
uv run pytest -q
```
