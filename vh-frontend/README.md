# VH Frontend

Electron 桌面前端，只通过 `VITE_API_BASE_URL` 指向的 FastAPI 公开接口上传视频、查询任务、播放原片和进行人工复核。前端不会读取后端文件路径或 `vh-agent` 内部字段。

后端在任务响应中返回相对 `source_url`，前端再以 `VITE_API_BASE_URL` 为基准解析；因此开发、测试和生产环境可以独立部署，不需要把服务器地址写入任务数据。

```powershell
nvm use 24.14.1
npm install
npm run dev
```

复制 `.env.example` 为相应环境配置，并设置不带末尾斜杠的后端地址：

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8777
```
