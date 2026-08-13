# Video Highlight

短剧高光检测产品仓库。三个顶层目录对应三个长期独立的部署单元：

```text
video-highlight/
├── vh-frontend/   # 上传、任务状态、播放器与人工复核界面
├── vh-backend/    # 任务 API、队列、视频存储、鉴权与结果查询
└── vh-agent/      # 多模态高光检测与离线评测
```

当前已实现并验证的是 `vh-agent`。`vh-backend` 和 `vh-frontend` 只定义职责边界，等进入对应开发阶段再创建技术栈和代码，不预先搭空壳。

Agent 的安装、运行、评测与契约说明见 [vh-agent/README.md](vh-agent/README.md)。
