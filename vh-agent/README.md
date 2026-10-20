# Video Highlight Agent

从短剧中发现有看点的片段，输出可播放的粗剪、看点说明和推荐理由，交给编辑采用与调整。

## 处理流程

`gpt-6.1-sol`（high）负责理解剧情、选择工具和取舍；Gemini 3.8 Flash（medium）观察原片与实际粗剪；本地 Qwen3.5-9B 提供精彩核心位置，语义检索定位相关材料。

```text
分析：检测位置、观看全片、按问题查证、保存候选
  → 选片：从完整候选池提交采用顺序
  → 成片检查：制作并独立观看选中的粗剪
      → 返回实际观察：主 Agent 核对说明，引用 review_id 确认采用
      → 需要修订：修改候选或改选，再核对当前复核
      → 确认完成：交付 MP4 和说明
```

输入原视频、可选字幕、语言和提取要求。数量、总时长和重叠预算只约束最终选择。候选范围包含必要前情、核心和直接反应，时长上限由任务设置，默认 24 秒。

## 七个工具

所有工具使用原片秒数。主模型只处理原片范围、问题和候选 ID；检测分窗、输入构建、观察持久化和成片检查由程序完成。

| 工具 | 输入 | 输出 |
| --- | --- | --- |
| `detect_highlights` | `start_sec`、`end_sec`，可选分页 | 精彩核心区间、总数、下一页和无效区间数量 |
| `search_video` | 台词或问题、检索方式和范围 | 带来源的位置与材料，支持分页 |
| `watch_video` | 原片范围、观看问题 | 观察 ID、音画观察、实际范围和原帧 |
| `read_frames` | 原片时刻、可选裁剪和文字识别 | 实际帧及可选文字位置 |
| `save_candidates` | 新增或修订的完整候选，可选剧情记忆 | 已保存候选与稳定 ID |
| `read_memory` | 候选、观察、原帧及 OCR、已读字幕或检测记录，按 ID 或分页 | 已保存的完整材料 |
| `submit_highlights` | 按交付顺序排列的候选 ID；收到复核后提交确认 | `awaiting_confirmation` 与实际观察，或硬性问题 `needs_revision`；确认后交付 |

`watch_video` 的观察成功送达主模型后，程序自动记录观看覆盖；检测完成只表示本地模型处理过。候选引用已看观察作为依据，新候选 `id=null`，修订沿用原 ID。候选池完整保留，分页只控制展示。 原帧、文字识别和首次记忆回查计入有效进展，重复读取相同材料不会重置停滞计数。

本地检测内部按 **20 秒负责范围、前后各 15 秒上下文、2 FPS** 处理长请求，输出转换回原片时间。SFT、GSPO 和服务共用 [detector_inputs.py](src/vh_agent/detector_inputs.py)，权重与效果见 [VL_TRAINING.md](VL_TRAINING.md)。

## 安装与运行

需要 FFmpeg、FFprobe。在 `vh-agent/` 目录安装并配置：

```bash
uv sync --extra enhanced --extra dev
cp .env.example .env
```

填写模型密钥与服务地址。本地检测设置 `VH_LOCAL_DETECTOR_URL=http://127.0.0.1:8004`，在项目根目录另启动常驻服务，复用已有训练环境与最佳 LoRA：

```bash
bash vh-agent/run_detector.sh --device cuda:0
```

服务监听本机 8004 端口；`GET /profile` 查看权重指纹和输入配置。缺少检索或 OCR 能力时，工具声明只开放实际可用操作。语义检索按需配置 embedding、reranker 地址，见 [.env.example](.env.example)。未配置时使用字幕原文搜索。

在 `vh-agent/` 运行分析：

```bash
uv run vh run /path/to/video.mp4 --job-id example --language zh --max-clip-sec 15
uv run vh run /path/to/video.mp4 --job-id example --language zh --max-clip-sec 15 --resume
```

`--subtitles` 提供字幕，`--instruction` 设置提取目标，`--task-file` 设置任务约束，`--verbose` 显示进度。CLI 参数覆盖任务文件。

观看请求按视频、原帧、说明和结构定义的完整体积检查；超限返回可修正的工具结果，缩短范围继续，已有有效观察保留。选片上下文携带候选对应的关键观察、疑点和相关 OCR 材料。

主提示词和示例位于 [prompts.py](src/vh_agent/prompts.py)，工具参数与说明位于 [contracts.py](src/vh_agent/runtime/contracts.py)。

## 输出与恢复

任务保存在 `outputs/jobs/{job_id}/`：

| 文件 | 内容 |
| --- | --- |
| `result.json`、`clips.json`、`clips/` | 完成状态、推荐片段、说明与 MP4 |
| `state.json`、`progress.json` | 剧情记忆、候选、覆盖进度和待执行调用 |
| `trace.jsonl` | 工具参数、结果、证据来源、用量、耗时及错误 |
| `selection.json` | 本次交付选择 |

原始观察和检测窗口按视频及输入配置缓存。网关返回无法解析的 JSON 时，沿用现有请求重试上限；无效响应不进入观察记录。中断后用相同参数加 `--resume`，已提交的工具结果与已成功的观察复用；模型或任务配置改变时使用新任务。候选边界或依据变化后重新制作并检查对应粗剪；只改说明时复用成片观察，重新核对并确认当前说明。确认绑定候选和实际复核，旧 review_id 不能确认修改后的候选。

公开结果保持 schema 2.0，`review_status=pending` 表示等待编辑采用。执行异常退出码为 1，主动暂停等中途停止为 2，未完成结果为 `partial`；完整任务允许零条片段。

## Agent 评测

```bash
uv run vh evaluate run --run-id quality_run --task-file configs/evaluation/agent_quality.json
uv run vh evaluate score --run-id quality_run
uv run vh evaluate run --run-id quality_run --resume
```

报告事件发现、关键证据覆盖、边界、完成率与耗时。`datasets/test/` 有 20 条参考标签，默认使用 10 条 AI 复核标签，`--include-silver` 加入另外 10 条银标。检测训练数据由 [vh-data](../vh-data/README.md) 管理。
