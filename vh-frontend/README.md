# VH Frontend

Electron 桌面端界面，负责视频上传、任务状态、高光播放器、资源库和人工复核交互。

## 技术栈

- Electron
- React + TypeScript
- Vite
- Node.js 24 LTS（见 `.nvmrc`）

## 本地开发

```powershell
nvm use
npm install
npm run dev
```

构建渲染进程：

```powershell
npm run build
npm start
```

当前界面使用本地模拟数据演示完整交互。正式接入时以 `vh-backend` 的公共 API 为唯一数据入口，不读取 Agent trace、服务器文件路径或模型内部字段。
