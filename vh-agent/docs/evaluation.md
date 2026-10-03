# Agent 评测与人工复核

评测持久保存输入、标注、任务约束、模型配置和源码，支持恢复及离线重算。它检查短剧高光发现、关键内容保留、完成率与耗时；人工审核判断是否值得采用以及需要改多少。比赛验证使用本项目的视频与实际提取目标。

## 使用

从 `vh-agent/` 运行：

```bash
uv run vh evaluate run --run-id quality_run --task-file configs/evaluation/agent_quality.json
uv run vh evaluate run --run-id quality_run --resume
uv run vh evaluate score --run-id quality_run
# 只运行指定视频，可重复 --video-id
uv run vh evaluate run --run-id focused_run --video-id VIDEO_ID --task-file configs/evaluation/agent_quality.json
```

`--dataset-dir` 指向含 `manifest.jsonl` 与 `annotations.json` 的目录；`--output-dir` 指定保存位置；`--limit` 选择前 N 条符合标注条件的视频。新评测使用新 ID。恢复沿用冻结的选样和任务，要求检测源码、配置与输入内容一致。

## 数据与协议

当前标注统一位于 `datasets/test/`，共有 20 条：10 条 `labeled` AI 复核标签与 10 条 `silver` 银标，全部等待人工确认。默认评测 10 条 AI 复核视频，显式 `--include-silver` 才运行全部 20 条。AI 复核分为看过旧预测与盲标两组，指标按来源分别报告。银标来自 Agent，适合检查流程与长视频压力，不能作为独立准确率证据。

复核原帧保留在 `outputs/annotation_audit_20260929/` 和 `outputs/annotation_audit_20261001/`。

manifest 每条包含 `video_id`、`title`、`path`、`duration_sec`，可提供 `language`、`drama_id` 与 `split`。annotations 包含 `video_id`、`annotation_status` 和 `highlights`；每条高光至少有起止时间。还可标记必要证据、可接受替代边界及可选看点。标注仅交给评分器，不发送给检测 Agent。

`configs/evaluation/agent_quality.json` 使用 3–30 秒范围，不限制输出数量与总时长，允许片段重叠，用于先检查看点发现和成片能力。实际产品默认 24 秒上限，须用另一份任务文件评测产品约束。片段长度是可配置限制，不是固定目标。

## 保存内容

```text
outputs/evaluations/{run_id}/
  protocol.json         冻结选样、标注、任务、文件哈希、无密钥配置与代码指纹
  source/               当轮检测源码
  jobs/{video_id}/
    execution.json      每次尝试的状态、耗时与停止原因
    result.json         complete 或 partial 的公开结果
    state.json          完整剧情、候选与恢复点
    trace.jsonl         原始工具、模型、复核记录
    clips/              最终采用的实际视频
  metrics.json          匹配关系、来源分组、完成率与耗时
  report.md             指标和逐视频标注／预测对照
```

每完成一条视频更新报告。后续修改原始标注不影响旧运行；离线评分使用冻结答案，不调用模型。未完成与失败的视频继续留在分母中，待定候选不作为最终预测。

## 指标如何阅读

| 指标 | 回答的问题 |
| --- | --- |
| 完成率与首轮完成率 | 能否稳定完成整条视频，需要多少次恢复 |
| 事件发现 P／R／F1 | 必选看点是否找到，有多少未标注或重复输出 |
| 决定性证据覆盖 | 命中事件后是否保留关键台词或动作 |
| 原始边界 IoU | 粗剪范围与推荐范围相差多少 |
| 事件池与复核后召回 | 漏检发生在发现、成片还是最终选择 |
| 输出数量与总时长 | 是否堆积背景或重复片段 |
| 无效工具调用与请求错误 | 时间浪费来自参数、流程还是接口 |
| 单视频耗时与模型耗时 | 当前体验有多慢，主要时间花在哪里 |

事件发现对“可接受边界最大 IoU”和“决定性证据保留率”取较大值，在 0.3／0.5／0.7 阈值下一对一匹配。同一必选事件只命中一次。先匹配必选事件，再匹配可选看点；可选命中单列并从精度分母排除。额外预测若再次命中已匹配目标，记录为时间重复并保留在精度分母。

边界 IoU 单独报告。这样能区分“看点已找到，初稿偏宽”和“核心事件漏掉”。时间匹配仍不能确认人物身份、语义重复或观看价值；Agent 自评分也不是准确率。Precision 和 Recall 同时提供计数与 Wilson 95% 区间。

候选池召回包含已登记后排除的事件，帮助定位发现后误删的问题。`completed_only` 只解释执行问题与内容问题的差别，不能代替包含失败视频的主指标。零预测或零标注时比率为 0，需同时看计数。

耗时累加每条视频的全部尝试；任一次缺少计时，累计耗时记为未知。报告明示计时分母，中位数和 P90 只使用计时完整的视频。并发请求耗时不能直接相加成整条视频的墙钟时间。Sol 没有使用 Gemini 的生成 seed；模型版本和推理强度保存在协议中，托管模型仍可能输出不同结果。

## 人工审核

人工逐段核对看点、关键证据与身份表述，记录采用、删除、拆分、首尾修改和编辑耗时。当前 20 条按同一格式继续人工定稿，保留 `human_verified` 与标注来源。采用率、语义误判和实际修改量由这些记录计算，避免把时间命中当作全部质量。
