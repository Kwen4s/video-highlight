# Video Highlight Agent

从短剧中发现值得独立观看的片段，输出可播放的粗剪、看点说明和推荐理由，交给人类编辑采用与调整。

检测入口已使用 `gpt-6.1-sol`，思考强度 `high`。Sol 负责剧情判断、工具调用和最终取舍；Gemini 观察原视频的画面与声音；Qwen3-VL-Embedding 和 Reranker 按具体问题找相关材料。完整流程见 [叙事状态与工具设计](docs/narrative-video-agent-design.md)。

任务提示词集中在 `src/vh_agent/prompts.py`，工具参数与调用说明位于 `runtime/contracts.py`。模块分工见设计文档的“代码职责”。

## 输入与处理

输入包括原视频、可选字幕、语言、提取目标与输出约束。提供的字幕随对应视频页面送入，ASR、语义检索与本地预测按问题调用。系统连续检查全片并保存播放进度；模型提出的候选数量不受最终输出预算限制。每条候选围绕一个可独立采用的看点，可以共享铺垫并重叠。默认 24 秒是上限，可以调整，不是目标时长。

剧情记忆只用几句话保留人物关系、当前变化和疑点。发现线索后，Agent 选择工具解决影响真实性、可懂性或采用决定的疑问，再从完整候选池初选。系统只制作并独立观看拟采用的粗剪，Sol 核对实际看点和说明后确认交付。舍弃的候选仍可重新采用。边界或必要证据修改使旧复核失效，说明文字修改复用已有成片；最终数量和总时长预算只在选择时生效。

| 工具 | 用途 |
| --- | --- |
| `inspect_video` | 带着问题补看原视频，保存问题、答案、台词及不确定处 |
| `search_video` | 语义定位画面或台词；有字幕或 ASR 时也能按原文和时间检索 |
| `read_frames` | 直接看实际帧与可选局部区域，核对动作、道具和表情 |
| `read_text` | 识别原帧文字，保留位置、置信度和真实帧时间 |
| `propose_highlights` | 可选本地模型提供位置线索，逐条观看后确认或排除 |
| `record_observations` | 原子保存已看材料、简短剧情记忆与候选 |
| `update_event` | 修订已有候选的判断、证据和粗边界 |
| `read_state` | 分页回查事件与原始观察 |
| `select_highlights` | 从完整候选池初选，核对拟采用片段后确认交付 |

工具随实际能力开放。检索相关性和本地分数用于找材料，不作为高光价值；没有配置这些工具时，连续音画检查仍覆盖全片。

## 安装与运行

```bash
uv sync --extra dev
cp .env.example .env
# 填写 OPENAI_API_KEY、GEMINI_API_KEY、明确的端点和模型
uv run vh run /path/to/video.mp4 --job-id example --language zh
uv run vh run /path/to/video.mp4 --job-id example --language zh --resume
uv run vh run /path/to/video.mp4 --subtitles /path/to/dialogue.srt --max-clip-sec 15 --max-highlights 8
```

需要 FFmpeg 与 FFprobe。`.env.example` 中的主模型默认是 Sol `high`；Gemini 3.8 Flash 通过 Foxrouter 的 Chat Completions 接口接收真实视频（含声音）和按指定密度提取的原帧，思考强度由 `GEMINI_THINKING_LEVEL=high` 设置；Sol 使用 Responses API。Qwen 服务沿用本机部署的 `8002` embedding 和 `8003` reranker，通过环境变量配置；声明检索能力前检查实际模型服务。

`--instruction` 设置提取目标；`--task-file` 设置时长、数量和重叠约束，CLI 参数覆盖对应字段。默认终端只显示完成状态与结果路径，`--verbose` 显示工具进度。

OCR、ASR 和本地候选依赖使用 `uv sync --extra enhanced --extra dev` 安装。`VH_LOCAL_CHECKPOINT` 指向真实 checkpoint，同目录 `config.json` 定义训练网络和特征；`VH_LOCAL_DEVICE` 选择设备。启用本地模型须提供视频语言。它独立训练与提供粗定位，不决定最终取舍。

## 输出与恢复

每个任务写入 `outputs/jobs/{job_id}/`：

```text
result.json      完成状态、推荐片段、看点与理由
clips.json       稳定片段 ID 与已复核媒体的对应关系
clips/           最终采用的实际 MP4
state.json       剧情记忆、事件、检索进度和原生会话检查点
progress.json    覆盖率、能力和待处理数量
trace.jsonl      工具输入输出、来源、模型用量、耗时及错误
selection.json   全部候选的采用或排除理由
```

观察视频、原始帧、检索向量和音画答案复用 `VH_MEDIA_CACHE_DIR`，不在每个任务重复保存。推荐 MP4 保存到任务目录，与核验媒体逐字节核对，供后端播放和导出。新推荐按采用价值排列，`review_status` 为 `pending`，由人工更新采用状态。前后端继续消费 schema 2.0。

观察、提问、候选与片段边界统一使用原片秒数。独立观看成片时，缺陷位置使用该成片的播放秒数。

检查点保存已提交动作与待执行调用。`--resume` 核对视频、模型、提示词与任务约束，请求和日志代码的修复不会使已保存观察失效。连接故障只重试同一请求，逐次记录耗时，以及 DNS、TLS、代理或连接错误的异常类型与系统错误码。Gemini 直接访问配置的网关，不继承终端中的代理环境变量。

旧版使用整份源码指纹的检查点需要新开任务；新版检查点按明确的任务配置恢复。

未完成时返回 `partial`，待定候选保留在状态里，公开高光为空。执行异常退出码为 1，后端按现有接口显示分析失败，并允许从检查点继续；主动暂停等中途停止退出码为 2。完成可以输出零条，不为填满数量制造片段。

会话按处理阶段和上下文预算切换；完整剧情与检索记录保持持久化，可分页回查。每批尚未登记的观察完成保存后，才继续取证。修改候选使该片段的旧复核失效；其他片段继续复用。

## 评测与训练

```bash
uv run vh evaluate run --run-id quality_run --task-file configs/evaluation/agent_quality.json
uv run vh evaluate score --run-id quality_run
uv run vh evaluate run --run-id quality_run --resume
uv run vh train run --annotations ../vh-data/outputs/snapshots/round-003/annotations.jsonl
uv run vh train run --help
uv run pytest -q
uv run ruff check src tests
```

评测冻结视频、标注、配置和源码，报告事件发现、关键证据覆盖、边界、完成率、无效工具调用和耗时。现有 20 条视频包含 10 条 AI 复核标签与 10 条银标，等待人工确认；默认评测 10 条 AI 复核视频，`--include-silver` 加入银标。详见 [评测与人工复核](docs/evaluation.md)。全部测试中的本地训练用例需要 enhanced 依赖。

本地检测模型的训练标注统一由 [vh-data](../vh-data/README.md) 生产：两次独立观看、统一选片复核、按短剧固定分组，再导出不可覆盖的训练快照。训练只学习明确标出的正负片段；验证组选权重，测试组只在选完后评估。

检测任务是提供高光候选的粗边界与分数。标注优先选择 15 秒以内的看点，几秒的动作或反应也可保留；模型学习实际标注长度，解码仅裁到视频范围并去重。候选池保持完整，Agent 决定最终采用。Qwen 特征描述人物、动作、对白、情绪与场景，覆盖不同观看价值；视觉与音频编码器冻结，只训练检测网络。

`--stage all` 提取特征并训练，`--stage features` 单独准备特征，`--stage train` 核对特征配置后训练。修改特征提示或模型配置后先重新提取。当前权重格式为 schema 5，新数据从头训练检测网络。
