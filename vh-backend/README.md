# VH Backend

面向 Electron 的 FastAPI 服务，负责视频上传、任务与会话持久化、公开媒体流、`vh-agent` 调用和高光对话编排。它与 `vh-agent` 部署在服务器侧；GPU 推理逻辑仍只存在于 `vh-agent`。

## 安装与运行

所有 Python 依赖使用 uv 和清华 PyPI 镜像：

```powershell
cd vh-backend
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync --group dev
uv run uvicorn app.main:app --host 127.0.0.1 --port 8777
```

服务器首次安装依赖后，日常启动和重启直接复用现有 `.venv`：

```bash
cd /home/tnx/video-highlight/vh-backend
./scripts/restart_backend.sh
```

完全停止服务：

```bash
cd /home/tnx/video-highlight/vh-backend
./scripts/stop_backend.sh
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

`agent.log`、缓存、对话计划和物理路径都不会通过 API 返回给前端。前端只使用后端返回的公开 API 路径，不读取服务器文件路径。任务不会按超时自动删除，包括长期停留在排队或处理状态的任务；已完成或失败的任务也会一直保留，直到用户在确认对话框中主动删除。

## API

- `GET /health`：健康检查。
- `POST /api/jobs`：multipart 上传；字段为 `file`、`job_id`、`language`。
- `GET /api/jobs`：列出服务端保留的全部任务，不按时间或数量隐藏旧任务。
- `GET /api/jobs/{job_id}`：查询任务状态和公开结果。
- `GET|HEAD /api/jobs/{job_id}/source`：读取原视频；支持浏览器媒体播放所需的字节范围请求。
- `POST /api/jobs/{job_id}/highlights/{highlight_id}/range`：使用 `start_sec`、`end_sec` 和 `revision` 直接调整高光时间范围。
- `POST /api/jobs/{job_id}/messages`：发送查询或编辑请求；请求携带 `message`、`revision` 和可选的 `selected_highlight_id`。
- `POST /api/jobs/{job_id}/messages/stream`：以 NDJSON 流式返回对话回复与最终任务结果。
- `DELETE /api/jobs/{job_id}`：删除已完成或失败的任务；仅接受用户确认后发送的 `{"confirmed": true, "job_id": "<job_id>"}`，任务号不匹配或缺少确认时不会删除。

任务状态为 `queued → processing → completed`，失败时为 `failed`。检测不自动重跑整任务。schema 2.0 的 `completion` 区分 complete 与 partial；`POST /api/jobs/{job_id}/resume` 从检查点继续。`attempt` 记录显式执行次数。当前使用单检测工作线程，避免多个 GPU 任务并发争用显存。结果和编辑对话没有有效期限制；结果修改仍使用乐观版本号，版本冲突返回 `409`。

## 配置

复制 `.env.example` 为本地 `.env` 后可调整：

- `VH_STORAGE_DIR`：服务端持久化数据根目录，默认 `./runtime`。
- `VH_AGENT_ROOT`：`vh-agent` 根目录。
- `VH_AGENT_UV_EXECUTABLE`：uv 可执行文件。
- `VH_AGENT_TIMEOUT_SEC`：检测任务超时秒数。
- `VH_MAX_UPLOAD_BYTES`：上传上限，默认 20 GB。
- `VH_CHAT_*`：对话模型的密钥、OpenAI 兼容地址、模型、超时和重试次数。
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

上传表单支持 `options` JSON（提取目标、时长、数量、重叠约束）及可选 `subtitles` 文件。`GET/HEAD /api/jobs/{job_id}/clips/{highlight_id}` 返回已复核媒体；手工修改边界后不再提供旧媒体。

任务列表、详情及其他任务响应新增两个只读字段，均从既有输入与进度契约投影，不改变 Agent 调用、排队、恢复或推理流程：

- `task`：原始提取目标、最短与最长片段、输出数量、总时长、重叠约束和 `subtitle_provided`。字幕仅返回是否提供，不返回物理路径；没有输入记录的历史任务返回 `null`。
- `workflow`：`schema_version=1.0` 的处理快照，包含更新时间、原片时长、检查区间数、下一原片范围、事件量、候选量、就绪成片数、文字读取行数、模型调用次数和分析轮数。
- `workflow.capabilities`：`transcript`（字幕 / ASR 检索）、`read_text`（OCR）、`read_frames`、`semantic_search`、`local_proposals`。`true` 只确认能力可用，不代表执行中或已经执行；`false` 表示本任务未启用，`null` 表示未回传。当前能力按问题调用，不重建旧版固定并行预处理阶段。

`progress` 保留公开覆盖率、待处理数量、计数和能力字段，通过白名单排除内部事件 ID、下一步控制指令及任意未声明字段。未回传值保持缺失或 `null`，不补成 0。进度文件缺失或暂时不完整时，任务状态与结果仍可正常查询。前端不读取 trace、日志、模型配置或缓存文件。更新后端后，已有任务也可从已保存的公开进度获得快照；无需重新分析。
