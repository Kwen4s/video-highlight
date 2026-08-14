# VH Frontend

Electron + React + TypeScript 桌面端，负责视频导入、真实上传进度、任务状态、高光播放、任务归档和人工复核。

## 本地开发

要求 Node.js 24 LTS（见 `.nvmrc`）、npm、uv，并已完成 `vh-agent` 环境与模型配置。

```powershell
nvm use
npm install
npm run dev
```

开发命令会同时启动：

- Vite：`http://127.0.0.1:5173`
- FastAPI：`http://127.0.0.1:8000`
- Electron（等待前两个服务健康后启动）

FastAPI 在开发模式启用热重载；首次加入新的启动参数或 API 中间件配置后，仍需重启一次 `npm run dev`。

只构建渲染进程：

```powershell
npm run build
npm start
```

## 文件目录

Electron 主进程在每次导入前创建：

```text
<软件安装目录>/video-data/jobs/<job_id>/source/
```

开发环境中的“软件安装目录”是 `vh-frontend/`。FastAPI 的默认 `VH_STORAGE_DIR` 指向同一个 `video-data/`，因此保存原片与 Agent 输出时不会出现两份存储。

打包部署时，后端的 `VH_STORAGE_DIR` 必须配置为 Electron 可执行文件所在目录下的 `video-data`。安装位置需要对当前用户可写；若安装器使用受保护的系统目录，应改用可写的按用户安装目录。

可通过 `VITE_API_BASE_URL` 修改前端 API 地址。渲染进程不拥有 Node.js 文件访问权限，只能通过受限 IPC 创建任务目录，并通过后端公开 API 获取媒体和结果。
