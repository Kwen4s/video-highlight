# VH Agent

短剧多模态高光检测 Agent。输入一个本地视频，输出结构化高光结果和可播放片段。

## Structure

```text
vh-agent/
├── src/vh_agent/   # Agent 运行代码
├── tests/          # 核心单元与回归测试
├── datasets/test/  # 5 条人工金标 + 10 条长视频银标
└── outputs/        # 任务结果与评测结果
```

根目录不保存生成式契约、一次性调参脚本或独立架构文档。Agent 的公共边界直接由 `DetectionTask` 和 `DetectionResult` 两个 Pydantic 模型定义。

## Single-Agent Development Contract

本项目长期采用一个面向用户的 `HighlightAgent`，不把 Chapter Mapper、Judge、Listwise Ranker、预处理器或视频编辑器包装成多个 Agent。它们是单 Agent 可调用的模型角色或确定性工具。只有当单 Agent 因工具严重重叠或提示词条件长期不可维护而无法达到评测目标时，才重新评估多 Agent 架构。

自然语言是控制层，稳定的视频处理流程是执行层：

```text
natural-language request
  -> typed command plan
  -> deterministic tools
  -> evidence verification
  -> at most one targeted refinement
  -> structured result
```

采用受约束的 Plan-Execute-Verify 循环，不使用自由运行的 ReAct 循环。简单命令只执行一个工具；只有证据不足时才允许一次局部补查。Agent 必须有明确的完成条件、失败状态和最大调用预算。

### Command Plans

自然语言请求最终只能编译为四类带类型的计划：

- `DetectPlan`: 检测指定类型、数量和时长的高光。
- `EditPlan`: 调整、删除、拆分、合并或替换已有高光。
- `QueryPlan`: 查询视频内容、剧情证据或检测结果。
- `ExportPlan`: 导出单段、合集或指定媒体格式。

含义不完整且会改变视频结果的命令必须先澄清。例如“第二段缩短两秒”需要确认修改开头、结尾还是两端；“第二段尾部缩短两秒”可以直接执行。用户界面的序号在执行前必须解析为稳定的 `highlight_id`，不能把可变化的数组下标当作资源标识。

### Agent Tools

单 Agent 最多暴露五组职责互斥的工具：

- `inspect_video`: 读取视频元信息和现有任务状态。
- `detect_highlights`: 执行高光检测工作流。
- `edit_highlights`: 对已检测片段执行确定性非破坏编辑。
- `query_video`: 针对时间轴和剧情证据进行局部检索。
- `export_highlights`: 使用 ffmpeg 生成或重新生成媒体文件。

模型不得直接拼接 shell 或 ffmpeg 命令。所有编辑先校验视频边界、最小时长和稳定 ID，只重新导出受影响的片段。

### State Ownership

`vh-backend` 负责会话、当前 `job_id`、结果版本、操作历史和撤销记录；`vh-agent` 接收显式状态并返回更新后的结构化结果，不维护隐藏的跨请求内存。一次 Agent 调用的最小状态包括：

```text
session_id, active_job_id, revision, user_message,
ordered highlight summaries, last action
```

检测运行内部仍只有五步：

```text
local preprocessing
  -> high-recall event discovery
  -> narrative memory and hypothesis construction
  -> semantic verification and boundary refinement
  -> global ranking and export
```

### Context And Prompt Rules

- Planner 只接收用户目标、视频元信息、可用工具和输出约束，不接收帧或整集字幕。
- Chapter Map 只抽取可观察事件，不做最终高光裁决，也不读取本地候选分数。
- Judge 只接收与当前假设相关的 `before/core/after` 证据和精简剧情事实，不接收 Chapter Mapper 的高光概率，避免分数锚定。
- Listwise Ranker 只接收已经 Judge 验证并精修边界的候选摘要，负责相对强弱、去重、剧情覆盖和数量选择，不修改事实、类型或时间边界。
- 每张关键帧必须带稳定编号和时间戳；每条剧情事实必须保留时间范围与证据来源。
- 事实、假设和最终决策使用不同字段，模型不得把推测写入已知事实。
- 提示词要求结论、分项分数和可核验的证据 ID，不要求或保存模型隐藏思维链。
- 上下文按任务检索，不把整集字幕、全部事件或历史提示词不断追加到下一次调用。
- 模型输出必须通过 Pydantic 严格校验；校验失败即终止当前步骤，不静默补默认值或切换兜底模型。
- Prompt 只表达可推广的任务定义和判据，不写入具体失败样本的人名、台词或剧情；规则变更必须配套通用回归测试。

### Engineering Rules

以下规则继承自全局 `AGENTS.md`，对本项目所有后续开发生效：

1. 不保留向后兼容。过时代码直接删除，不增加兼容层、migration 或 fallback。
2. 选择满足当前需求的最简单实现，不做预防性抽象或多余配置层。
3. 从可运行的最小端到端版本逐步生长，不为未完成的复杂度拆掉工作链路。
4. 组件模块化并保持关注点分离。
5. 优先采用成熟、活跃维护的库，没有明确理由不自行重写。
6. 增加依赖或自研前，先检查项目现有依赖已经提供的能力。
7. 架构决策面向长期，不接受“先这样以后再换”的临时设计。
8. 先研究成熟产品和论文的已验证模式，再设计同类能力。

当前迭代优先级是检测效果而不是扩展 Agent 功能。先冻结命令和工具边界，再集中优化 Judge 召回、剧情证据、时间边界和全局排序；检测指标稳定后再实现完整自然语言编辑体验。
### Accuracy Roadmap

1. 第一阶段：把 `EventCard` 编译为不可信的 `HighlightHypothesis`，由 Judge 使用原始帧、带时间码字幕和声音证据完成支持/反证验证；随后用本地多模态每秒显著性曲线精修边界。
2. 第二阶段：把逐候选 Scout 改为章节级 Map，一次抽取章节事件并合并重复候选；剧情记忆只保留经证据验证的角色、事实、关系变化和未决问题。
3. 第三阶段：对已验证候选做全局 listwise 排序，统一处理相对强弱、剧情覆盖、重复事件和数量预算。
4. 人工标注扩充后，使用 `EventCard`、`HighlightHypothesis`、验证证据、显著性曲线和最终边界训练多模态高光模型，并作为同一检测接口的本地实现接入。

三个阶段均已实现。第三阶段只处理已验证候选的全局相对排序、去重与数量预算，不反向改变章节 Map、Judge 或边界。


## Pipeline

```text
video
  -> ffmpeg / PySceneDetect / Whisper / PP-OCRv6 / SenseVoice / Qwen3-VL Embedding
  -> temporally balanced candidates
  -> Qwen3-VL-8B chapter Map
  -> evidence-only StoryMemory / EventCard / HighlightHypothesis
  -> Qwen3-VL-32B evidence verification
  -> saliency-curve boundary refinement
  -> Qwen3-VL-32B global listwise ranking
  -> selected highlights and MP4 clips
```

Chapter Map 负责高召回事件抽取，不能直接出片；Judge 负责逐候选正式裁决；Listwise Ranker 只比较通过裁决的候选。每条最终高光最长 24 秒。

## Source Modules

- `pipeline.py`: 端到端编排和后端调用入口。
- `models.py`: Agent 输入、输出、事件和 trace 数据模型。
- `candidates.py`: 本地多模态候选生成和时间覆盖。
- `reasoning.py`: SiliconFlow 调用、结构化输出解析和剧情记忆。
- `preprocessing/`: ffmpeg、ASR、OCR、声音事件、镜头和 embedding 适配器。
- `evaluation.py`: 测试集运行与指标计算。
- `cli.py`: `vh` 命令行入口。

## Setup

```bash
conda activate env_vh
cd /home/wkw/video-highlight/vh-agent
python -m pip install -e '.[enhanced,dev]'
```

`.env` 只保存在本机。模型位于 `/data1/video-highlight-models` 和 `/data1/modelscope_models`，媒体缓存位于 `/data1/video-highlight-cache`。

## Run

```bash
vh inspect /path/to/video.mp4 --language zh
vh run /path/to/video.mp4 --language zh --video-id vid_xxx --trace
```

后端直接调用：

```python
from vh_agent import DetectionTask, HighlightDetectionService

result = HighlightDetectionService().detect(
    DetectionTask(video_path="/path/to/video.mp4", video_id="vid_xxx", language="zh")
)
```

每个任务只写入：

```text
outputs/jobs/{job_id}/
├── result.json
├── clips/
└── trace.json       # 仅启用 trace 时生成
```

## Evaluate

`datasets/test` 是唯一测试集，共 15 条真实视频。默认只评分 5 条人工金标；`--include-silver` 同时评分 10 条长视频银标。

```bash
vh evaluate run --run-id phase3_v0_13 --resume
vh evaluate score --run-id phase3_v0_13
vh evaluate score --run-id phase3_v0_13 --include-silver
```

0.13.0 的 5 条人工金标结果保存在 `outputs/evaluations/phase3_v0_13`：候选召回、语义验证召回和最终召回均为 100%，精确率 61.5%，F1 76.2%，共输出 13 段。该目录是隔离 listwise 评测，复用了未变更的 0.12.0 预处理、Chapter Map、Judge 和边界 trace，仅新增 5 次全局排序调用，不重复导出评测片段。

0.12.0 的 5 条人工金标结果保存在 `outputs/evaluations/phase2_v0_12`：候选召回、语义验证召回和最终召回均为 100%，精确率 60.0%，F1 75.0%；平均云调用从 0.11.0 的 12 次降到 7 次（下降 41.7%）。第二阶段已完成。

0.11.0 的第一阶段结果保存在 `outputs/evaluations/phase1_v0_11`：候选召回 100%，语义验证召回 87.5%，最终召回 100%，精确率 61.5%，F1 76.2%。

历史 10 条长视频的 0.8.0 结果保存在 `outputs/evaluations/baseline_0.8`：候选召回 96.7%，最终召回 40.0%，精确率 33.3%，F1 36.4%。

## Verify

```bash
ruff check src tests
pytest -q
```
