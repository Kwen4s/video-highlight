# Video Highlight Agent

项目只保留一条检测链路：`vh run` → 原生 Gemini 视频 ReAct Agent → 实际成片复核 → 完整候选池取舍 → 冻结成片。旧 Scene Map、双 Judge、仲裁、滑窗启发式和中点切分已删除。

## 输入与工具

输入是原视频、可选字幕、语言、提取目标和输出约束。24 秒是可配置上限，不是目标长度；目标是有看点、能独立看懂，保留必要铺垫和反应的片段。输出数量只在最终选择时生效。

- 原片分页由运行时按时间顺序连续送入；能放进同一请求的相邻页面会批量发送，媒体过大时拆页，不截断。
- `inspect_interval`：按原片时间补查，可设置 `sampling_fps` 提高动作观察密度。
- `search_transcript`：按原文或时间分页查询 SRT/VTT/ASS 字幕或按需 ASR；未配置可用文字来源时不声明该工具。
- `propose_highlights`：可选本地定位模型，对完整视频推理并分页返回全部预测；粗边界、分数只是线索。每个返回池中的候选须以视频证据处置。
- `record_observations`：一次登记本轮送入的全部观察。每条发现直接登记为事件并与实际观察关联；确实没有发现候选时说明情况。
- `update_event`：修订已有事件的判断、证据、边界和合并关系。
- `read_state`：需要回查时分页读取事件、观察、已读字幕、查询历史和有效页面。
- `select_highlights`：依据实际成片所见比较完整候选池的独立吸引力和新增价值，逐项保留或舍弃；一次原子提交同时冻结候选、记录理由和价值顺序。内容需要核对时先补看或修订。

Agent 在各工作段内保留原生工具交互，按 token 估算、请求字节预算和处理阶段换段。每批视频送达后，下一步只开放绑定这些观察的批量登记工具；登记完成后才继续取证。全片取证完成后，运行时自动生成实际片段并并发独立复核，不再为确定性步骤消耗 Agent 轮次。已读字幕、分页进度、当前事件和完整候选池持久保存。每条候选只围绕一个可独立采用的核心看点；相邻看点分开登记，可以共享铺垫并重叠。最终选择不修改冻结边界；禁止重叠时由 Agent 修订冲突的选择，程序不切分片段。独立复核只判断实际成片是否缺少必要上下文、存在截断、混合多个独立看点或有音画故障；粗边界直接交给人类编辑调整。公开片段的描述和类型使用同一次独立复核的实际所见，取舍理由来自全局选择；原始事件判断保留在内部状态中。

事件核验与编辑取舍分开：清楚发生的事件先进入成片复核，价值较弱、开放悬念或剧情继续都留到最终选择。`rejected` 只允许事件未发生、被证据否定、明确超出用户任务，或已经证明无法形成合格片段，并保存结构化类别。

v23 首次进入最终选择时建立新的会话工作段，依据完整候选池的 `visible_event` 比较可用看点。独立复核只描述实际内容、分类并检查具体剪辑缺陷；采用价值 `score` 和理由由主 Agent 看完整候选后统一生成。初看描述和早期价值理由按需回查。选择期间补看、登记和修订保持同一工作段，并持久保留核对结论；公开描述和类型来自实际成片复核，分数和理由来自同一次最终取舍。

每轮模型只允许执行一个工具动作；`record_observations` 和 `select_highlights` 分别在一次动作中提交一批观察或完整候选决定。多工具调用整体拒绝且不改变业务状态。已启用的输入来源必须全部消费：全文字幕分页读完，本地位置线索完整载入并逐项用原视频核验，之后才开放最终选择。每轮请求都提供最新持久状态，包含运行时自动生成的成片与独立复核结果。选择阶段直接提供完整候选池，同时保留补看、修订和可用的字幕检索；候选较多时还可读取完整事件。修订会使旧复核失效，重新成片复核后才能选择。

## 安装与运行

```bash
uv sync --extra dev
cp .env.example .env  # 填写 GEMINI_API_KEY 与明确的端点、模型
uv run vh run /path/to/video.mp4 --job-id job_example1 --language zh
uv run vh run /path/to/video.mp4 --job-id job_example1 --language zh --resume
uv run vh run /path/to/video.mp4 --subtitles /path/to/dialogue.srt --max-clip-sec 15 --max-highlights 8
```

默认终端仅显示完成状态和结果路径，`--verbose` 显示逐步工具进度。

`--instruction` 设置目标；`--task-file task.json` 可设置 `min_clip_sec`、`max_clip_sec`、`max_highlights`（null 表示不限数量）、`total_duration_sec` 和 `allow_overlap`。CLI 参数覆盖任务文件对应项。

本地候选及 ASR 使用 `uv sync --extra enhanced --extra dev` 安装依赖；`VH_LOCAL_CHECKPOINT` 指向实际 checkpoint，其同目录 `config.json` 定义网络与特征配置，`VH_LOCAL_DEVICE` 指定设备。启用本地候选时必须提供视频语言，OCR 按该语言选择模型，不根据数据集名称猜测。没有 checkpoint 时 Agent 通过原生音视频独立发现事件。特征冷启动复用现有 ASR、OCR、镜头和冻结编码器；缓存签名不匹配时重新提取。

Gemini 走原生 `generateContent`，发送实际音视频与函数响应；只使用配置端点和模型，不自动切换协议或模型。上传帧率设置不等于已证明网关逐帧检查，真实召回和边界质量需人工评测。

## 结果、检查点和恢复

每个任务写入 `outputs/jobs/{job_id}/`：

```text
result.json     # schema 2.0、complete/partial、高光与公开播放 URL
clips.json      # 稳定片段 ID → 已复核媒体及对应事件
state.json      # 检查点：业务记录、原生会话、待执行调用与已提交结果、检索和停滞记忆
progress.json   # 实际覆盖及待处理数量
trace.jsonl     # 工具动作、复核、API 用量和耗时，不保存隐藏思维
selection.json # 每个候选的最终选择/排除理由
media/          # 按实际源时间戳生成的观察与成片
```

全部页面已观察登记、全部已发现事件及工具候选已处置后才能 `complete`。空高光可以是完整结果；partial 保留待选片段于 state，公开 highlights 为空，避免把尚未取舍的候选当作最终输出；API 错误、无效模型输出或无进展循环停止时返回 `partial`、退出码 2。`--resume` 继续同一检查点，要求输入、模型、代码和约束一致；暂时性请求失败按 `VH_MAX_REQUEST_ATTEMPTS` 有界重试同一请求，所有尝试写入 trace；不会重跑整个任务。前后端都消费 schema 2.0，不兼容旧结果格式。

状态更新和调用结果一同提交；恢复继续未执行调用，复用已提交结果。未完成的模型请求重发同一会话输入。`VH_CONTEXT_TOKEN_BUDGET`（含 8192 输出预留）、`VH_CONTEXT_BYTE_BUDGET` 和 `VH_MAX_STAGNANT_STEPS` 控制资源；token 使用保守估算，字节另由请求端强校验。改代码后的旧检查点不迁移，需要新任务。

后端 `/api/jobs/{id}/clips/{highlight_id}` 返回复核的同一份媒体。手工编辑边界使旧成片关联失效，按编辑后的原片区间导出。

## 评测与本地训练

```bash
uv run vh evaluate run --run-id react_v1 --task-file configs/evaluation/agent_quality.json
uv run vh evaluate score --run-id react_v1
uv run vh evaluate compare --comparison-id selection_change --kind revision --run-id before --run-id after
uv run vh label run --run-id react_silver_v1
uv run vh train run --help
```

评测冻结输入、标注、配置与代码，支持恢复和离线重算；分别报告事件发现、原始边界 IoU、阶段召回、完成率、置信区间和运行时指标。partial 保留在评测分母中，其片段不当作最终预测。数据集现有 20 条：10 条 AI 复核标签和 10 条银标，均未经过人类确认。默认只选择 `annotation_status=labeled` 的 10 条 AI 复核样本；显式 `--include-silver` 才运行全部 20 条。详见 [评测协议与复用方法](docs/evaluation.md)、[五条新增标注审计](docs/annotation-expansion-20261001.md)和[稳定性复查](docs/stability-review-20261001.md)。银标使用同一 Agent，不能作为独立准确率证据。用户负责后续人工定稿。

评测请求默认固定 `VH_GENERATION_SEED=7`，并可用 `vh evaluate compare` 比较同协议重复运行的完成率、输出数量和边界一致性。固定 seed 只提高可复现性，不能替代重复运行。

`highlight_model/` 保留既有冻结多模态编码器与叙事转变定位器、训练和指标；推理工具不按最终输出数量截断。本地模型仍可独立训练。

```bash
uv run pytest -q             # 全部测试需 enhanced 中的 torch
uv run ruff check src tests
```

模块职责与处理约束见 [架构文档](docs/react-video-agent-design.md)，任务和工具字段见 [输入契约审计](docs/input-contract-audit.md)，五视频状态机结果见 [v8 验证](docs/state-v8-evaluation-20260930.md)，当前事件粒度定向实测见 [v9 验证](docs/state-v9-granularity-evaluation-20260930.md)，下一轮对照见 [Agent 消融协议](docs/agent-ablation.md)。
