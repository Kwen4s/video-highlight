# VH Agent

短剧多模态高光检测 Agent。输入一个本地视频，输出结构化高光结果和可播放片段。

## Structure

```text
vh-agent/
├── src/vh_agent/   # Agent 运行代码
├── tests/          # 核心单元与回归测试
├── datasets/test/  # 5 条人工金标 + 10 条长视频银标
├── datasets/silver/ # 全量模型银标，按运行 ID 隔离
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
  -> high-recall semantic scene construction
  -> one-scene narrative mapping
  -> dual-perspective verification and boundary refinement
  -> causal merge, global ranking and export
```

### Context And Prompt Rules

- Planner 只接收用户目标、视频元信息、可用工具和输出约束，不接收帧或整集字幕。
- Scene Map 逐场抽取可观察叙事，不做最终高光裁决；本地滑窗只聚合成 `local_score`，不作为 Map 或 Judge 的语义单位。
- Judge 只验证当前 `SceneCard` 的 Map 三元组、当前场原始材料、紧邻前场铺垫和未验证 `EvidenceLedger`；证据核验视角与成片视角各调用一次，只有结论、类型、分数或边界存在实质分歧时才增加一次仲裁，不改写剧情或另起高光事件。
- Listwise Ranker 只在已验证候选超过视频输出预算时接收其摘要，负责全局压缩、相对强弱、去重、剧情覆盖和数量选择，不修改事实、类型或时间边界；预算内候选由确定性去重后直接保留。
- 每张关键帧必须带稳定编号和时间戳；每条剧情事实必须保留时间范围与证据来源。
- 事实、假设和最终决策使用不同字段，模型不得把推测写入已知事实。
- 提示词要求结论、分项分数和可核验的证据 ID，不要求或保存模型隐藏思维链。
- 上下文按任务检索，不把整集字幕、全部事件或历史提示词不断追加到下一次调用。
- 模型输出必须通过 Pydantic 严格校验；空内容或非法 JSON 使用同一模型和同一请求最多重试两次，第三次失败即终止当前步骤，不静默补默认值或切换兜底模型。
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

1. 第一阶段：本地将镜头合成为 8-40 秒的 `SceneCard`，滑窗分数只作为场景召回与切分软约束；Scene Map 为每场生成带原始证据的叙事三元组。
2. 第二阶段：两个互补 Judge 视角使用原始帧、带时间码字幕、声音、紧邻前场和未验证 `EvidenceLedger` 验证 Scene Map；Map 不成立时直接否决，分歧时才仲裁，不在 Judge 内改写剧情。
3. 第三阶段：相邻且因果连续的通过场景合并为 6-24 秒片段；确定性去重后，只有候选超过数量预算时才进行全局 listwise 压缩。
4. 人工标注扩充后，使用 `SceneCard`、Map 证据、Judge 决策、显著性曲线和最终边界训练多模态高光模型，并作为同一检测接口的本地实现接入。

三个阶段均已实现。第三阶段只处理已验证候选的全局相对排序、去重与数量预算，不反向改变章节 Map、Judge 或边界。

### Local Highlight Model

本地模型采用无 Query 的事件中心叙事转变定位。标注文件保持不变，Gemini 的剧情描述和判断理由不进入学生模型；公开输出仍只有带分数的时间段。

```text
3-frame video moment + separated ASR/OCR -> frozen Qwen3-VL-Embedding-2B
30-second audio windows -> frozen SenseVoice encoder -> second-aligned audio states
  -> availability-aware gated moment fusion
  -> causal hierarchical local attention at 1s / 4s / 16s
  -> temporally pooled scene memory
  -> causal before / local event / bounded after + global salience
  -> narrative transition vector
  -> eventness peak + start/end offsets + anchor-aware segment quality
  -> temporal NMS
```

每一秒使用 `[t-1,t,t+1]` 三帧的 Qwen video 接口，并在文本中分开写入 Speech 与 On-screen text。音频以带重叠的 30 秒窗口提取 SenseVoice encoder hidden，再对齐到秒级；两路冻结特征按视频缓存，训练阶段只优化融合、层次时序、场景记忆、转变解码器和输出头。

场景记忆只由媒体缓存中的镜头边界和时序特征池化得到。层级骨干的局部注意力、下采样和上采样均为因果计算；before 分支只能访问过去 32 秒和在当前时刻前已经结束的场景；event 分支访问 `[t-2,t+2]`；after 分支最多访问未来 8 秒。全片场景记忆仅进入显著性上下文，不进入三种状态，因此不会把未来剧情泄漏给 before。

模型以 decisive anchor 为事件中心生成 eventness 热图，并在其中心采样区域内密集回归完整银标段的起止偏移，避免只在真实峰值附近训练、却从预测峰读取未受监督偏移。setup、decisive 与 reaction 时间监督三个受时间掩码约束的注意力位置；它们的位置、存在性与预测边界共同进入片段质量头。训练损失包括 event focal loss、Smooth L1 + temporal IoU 边界损失、anchor attention loss、预测片段 IoU 质量损失和片段内 hard-negative 排序损失。推理分数由 eventness 与 anchor-aware segment quality 联合给出，再执行 temporal NMS。验证集报告段级 `F1@IoU 0.3/0.5/0.7`，仍按较严格的 `0.5/0.7` 均值选择阈值、Top-K 与最优 checkpoint。

训练数据使用固定 seed 按视频随机划分为 80%/10%/10%，同一部剧的不同视频允许进入不同子集；运行目录保存完整 `video_split.json`，确保实验可复现。


## Pipeline

```text
video
  -> ffmpeg / PySceneDetect / Whisper / PP-OCRv6 / SenseVoice / Qwen3-VL Embedding
  -> local high-recall windows (score and scene-boundary hints only)
  -> semantic SceneCard construction
  -> concurrent OpenAI-compatible one-SceneCard Scene Map calls
  -> unverified EvidenceLedger
  -> evidence-focused Judge + editing-focused Judge
  -> conditional adjudication on material disagreement
  -> scene-aware state-change gate and boundary refinement
  -> post-Judge causal scene merge and deterministic deduplication
  -> global listwise compression only above the output budget
  -> selected highlights and MP4 clips
```

`SceneSegment` 只是镜头边界；`SceneCard` 才是一场戏或完整事件跨度，目标 8-40 秒，可跨多个镜头，最终高光仍限制在 24 秒内。局部滑窗只向 SceneCard 聚合 `local_score`，绝不作为 Judge 主输入。Scene Map 对每场生成唯一、带证据的叙事命题；Judge 只输出证据充分性、叙事影响、独立可懂性和成片完整性四项分数，总分和通过结论由代码计算。`EvidenceLedger` 由更早 Scene Map 的带时间证据观察组成，明确是未验证账本，不能混称为 Judge 已验证记忆。只有两张相邻 SceneCard 都通过 Judge、后场明确标记为前场的因果连续，且合并后不超过 24 秒时才合并出片。确定性去重后，已验证候选在输出预算内会直接保留；仅超出预算时调用全局 listwise 压缩。短视频最多输出两段。
0.21.0 的性能优化不删减 SceneCard：PP-OCRv6 以 8 帧批量推理；镜头检测、ASR、OCR 与 SenseVoice 首次运行并行；ASR/OCR/声音事件/镜头/embedding 结果按视频指纹和预处理签名写入媒体缓存；Scene Map 均匀检查最多 6 张低细节关键帧，Judge 均匀检查最多 8 张高细节关键帧。Scene Map 按模型、完整 Prompt、SceneCard 和实际提交帧逐场缓存，模型、Prompt、帧或预处理参数变化会直接使对应缓存失效，不读取旧格式。Map 与 Judge 默认并发均为 6；Whisper 使用确定性贪心解码，硬字幕继续由 PP-OCRv6 校正，避免 beam search 在全量标注中接近实时地重复搜索。最终结果仍等待全片所有场景完成，因此这些优化不改变召回范围。


## Source Modules

- `pipeline.py`: 端到端编排和后端调用入口。
- `models.py`: Agent 输入、输出、SceneCard 和 trace 数据模型。
- `candidates.py`: 本地高召回滑窗评分和时间覆盖。
- `reasoning.py`: OpenAI 兼容的单场 Scene Map、Judge、结构化校验和 EvidenceLedger。
- `preprocessing/`: ffmpeg、ASR、OCR、声音事件、镜头和 embedding 适配器。
- `highlight_model/`: 本地高光模型的编码器、数据集、网络、损失、解码和拟合流程。
- `evaluation.py`: 测试集运行与指标计算。
- `cli.py`: `vh` 命令行入口。

## Setup

```bash
conda activate env_vh
cd /home/wkw/video-highlight/vh-agent
python -m pip install -e '.[enhanced,dev]'
```

`.env` 只保存在本机。模型位于 `/data1/video-highlight-models` 和 `/data1/modelscope_models`，媒体缓存位于 `/data1/video-highlight-cache`。

推理模型使用 OpenAI 兼容接口。当前默认路由为 Gemini：Scene Map 使用
`gemini-3.1-flash-lite`，Judge 与全局排序使用 `gemini-3.7-flash`，端点为
`https://yetoken.vip/v1`。Qwen/SiliconFlow 配置仍保留；将
`VH_REASONING_PROVIDER` 设置为 `siliconflow` 即可显式切换。

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

`datasets/test` 是唯一测试集，共 15 条真实视频。默认只评分 5 条人工金标；`--include-silver` 同时评分 10 条长视频银标。评分同时报告 IoU 0.30、0.50、0.70 的候选召回、验证召回、边界召回、精确率和 F1。

```bash
vh evaluate run --run-id phase3_v0_13 --resume
vh evaluate score --run-id phase3_v0_13
vh evaluate score --run-id phase3_v0_13 --include-silver
```

0.13.0 的 5 条人工金标结果保存在 `outputs/evaluations/phase3_v0_13`：候选召回、语义验证召回和最终召回均为 100%，精确率 61.5%，F1 76.2%，共输出 13 段。该目录是 Qwen/SiliconFlow 的历史隔离 listwise 基线；每次推理或边界逻辑变更都必须使用独立运行 ID 重跑 Gemini，不能复用旧 trace。

0.12.0 的 5 条人工金标结果保存在 `outputs/evaluations/phase2_v0_12`：候选召回、语义验证召回和最终召回均为 100%，精确率 60.0%，F1 75.0%；平均云调用从 0.11.0 的 12 次降到 7 次（下降 41.7%）。第二阶段已完成。

0.11.0 的第一阶段结果保存在 `outputs/evaluations/phase1_v0_11`：候选召回 100%，语义验证召回 87.5%，最终召回 100%，精确率 61.5%，F1 76.2%。

历史 10 条长视频的 0.8.0 结果保存在 `outputs/evaluations/baseline_0.8`：候选召回 96.7%，最终召回 40.0%，精确率 33.3%，F1 36.4%。

## Silver Labels

`vh label run` 读取 `/data1/my_short_drama/metadata/{en,zh}/{train,test}.jsonl` 的每条视频，按时长从短到长调度。Map 与 Judge 均固定为 `gemini-3.7-flash`，并裁决每一个 `SceneCard`，不使用在线检测的 18 场召回预算或 12 段成片预算；银标保留全部已验证场景，不调用 listwise 压缩。每场由证据核验 Judge 和成片 Judge 独立判断；存在实质分歧时才调用第三次仲裁。本地保留帧、ASR、OCR 与音频证据，云端只接收带时间戳的 SceneCard 证据包；yetoken 的 OpenAI 兼容接口没有可用的视频文件上传端点，因此不直传整段视频。

```bash
vh label run --run-id gemini37_scene_v1
```

结果只写入 `datasets/silver/{run_id}/`：

```text
run.json             # 模型、总量与完成状态
annotations.jsonl    # 每条源元数据加 silver highlights，适合作为训练数据
errors.jsonl         # 仅失败项；下次 --resume 会重试
```

命令默认断点续跑，保留源 metadata 不变，不导出 MP4、不写 trace，也不创建每条视频的 job 目录。同一 run 可以继续既有进度；新规则生成的记录带 `annotation_revision: scene_v2`，无该字段的旧记录视为 `scene_v1`。每个高光包含时间边界、类型、分数、描述、证据、两次原始评分、共识置信度和 `pending` 人工复核状态。顶层 `highlights` 保存全部去重后的验证事件，残余重叠会在相邻边界中点切开；每条记录还包含全部 `scene_labels`：已裁决场景保存正负标签、共识决策和原始投票，未进入 Judge 的场景标为 `null`。空高光视频同样写入，因此负样本和难负样本不会丢失。

## Verify

```bash
ruff check src tests
pytest -q
```
