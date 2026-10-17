# Video Highlight Agent

从短剧中发现值得单独观看的看点，输出可播放的粗剪、看点说明和推荐理由，交给编辑采用与调整。

## 当前流程

`gpt-6.1-sol`（high）负责理解剧情、调用工具和最终取舍；Gemini 3.8 Flash 观察原视频的画面与声音；Qwen3-VL-Embedding 和 Reranker 按具体问题检索材料。

输入原视频、可选字幕、语言和提取要求。Agent 连续查看全片，用简短剧情记忆保留人物关系、事情变化和疑点；发现候选后按问题补看证据，再从完整候选池选择片段。系统制作并独立观看拟采用的粗剪，确认后交付。候选可以重叠，数量和总时长预算只在最终选择时生效。默认 24 秒是片段上限，可通过参数调整。

| 工具 | 用途 |
| --- | --- |
| `inspect_video` | 带着问题补看音画，记录答案、台词和疑点 |
| `search_video` | 按语义或原文定位相关材料 |
| `read_frames` | 核对动作、道具、表情与局部画面 |
| `read_text` | 识别原帧文字并保留时间和位置 |
| `record_observations` | 保存观察、剧情记忆和候选 |
| `update_event` | 修订候选判断、证据与粗边界 |
| `read_state` | 分页回查事件和观察 |
| `select_highlights` | 初选、检查粗剪并确认交付 |

工具按实际可用能力开放。运行时保留 `propose_highlights` 接口供新检测模型接入，目前没有启用本地检测器。旧 embedding 检测网络和训练入口已删除；新 VL 模型方案见 [VL_TRAINING.md](VL_TRAINING.md)。检索分数用于找材料，高光价值由 Agent 判断。

## 安装与运行

在 `vh-agent/` 目录运行，需要 FFmpeg 与 FFprobe：

```bash
uv sync --extra dev
cp .env.example .env
# 填写两家模型的密钥、端点和模型名
uv run vh run /path/to/video.mp4 --job-id example --language zh --max-clip-sec 15
uv run vh run /path/to/video.mp4 --job-id example --language zh --max-clip-sec 15 --resume
```

`--subtitles` 提供字幕，`--instruction` 设置提取目标，`--task-file` 设置任务约束，CLI 参数覆盖对应字段。`--verbose` 显示工具进度。

Sol 使用 Responses API，Gemini 使用 Chat Completions 接收视频、声音与原帧。检索服务默认端口为 8002（embedding）和 8003（reranker），配置见 [.env.example](.env.example)。OCR 与 ASR 依赖通过 `uv sync --extra enhanced --extra dev` 安装。

任务提示词集中在 [prompts.py](src/vh_agent/prompts.py)，工具参数与说明在 [runtime/contracts.py](src/vh_agent/runtime/contracts.py)。

## 输出与恢复

任务保存在 `outputs/jobs/{job_id}/`：

| 文件 | 内容 |
| --- | --- |
| `result.json`、`clips.json`、`clips/` | 完成状态、推荐片段、说明及实际 MP4 |
| `state.json`、`progress.json` | 剧情记忆、候选、覆盖进度与会话检查点 |
| `trace.jsonl` | 工具调用、证据来源、用量、耗时和错误 |
| `selection.json` | 候选采用或排除的理由 |

观察和候选统一使用原片秒数；独立检查粗剪时使用粗剪播放秒数。公开输出保持 schema 2.0，推荐项的 `review_status` 为 `pending`。

中断后使用相同任务参数加 `--resume`。恢复核对视频、模型、提示词和任务约束，并复用已保存观察。候选边界或必要证据发生变化时，需要重新检查对应粗剪。执行异常返回退出码 1，主动暂停等中途停止返回 2；未完成结果为 `partial`，待定候选保存在状态中。完整任务可以输出零条。

## 评测与检查

```bash
uv run vh evaluate run --run-id quality_run --task-file configs/evaluation/agent_quality.json
uv run vh evaluate score --run-id quality_run
uv run vh evaluate run --run-id quality_run --resume
uv run pytest -q
uv run ruff check src tests
```

评测报告事件发现、关键证据覆盖、边界、完成率、无效工具调用和耗时。`datasets/test/` 有 20 条视频参考标签，默认使用 10 条 AI 复核标签，`--include-silver` 加入另外 10 条银标。这是 Agent 评测集；检测模型训练数据由 [vh-data](../vh-data/README.md) 管理。
