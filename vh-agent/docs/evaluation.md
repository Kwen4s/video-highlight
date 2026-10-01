# 可复用 Agent 评测

从 `vh-agent/` 运行：

```bash
# 新建评测：默认仅 annotations.json 中 annotation_status=labeled 的视频
uv run vh evaluate run --run-id quality_v1 --task-file configs/evaluation/agent_quality.json

# 中断或 partial 后显式继续；使用已冻结的选样及任务配置
uv run vh evaluate run --run-id quality_v1 --resume

# 离线重算，不调用模型、不读取后续修改的原始标注
uv run vh evaluate score --run-id quality_v1

# 同代码、配置、视频、任务和标注重复运行后，量化输出漂移
uv run vh evaluate compare --comparison-id quality_v1_repeat \
  --run-id quality_v1_a --run-id quality_v1_b

# 同视频、配置、任务和标注，比较实现改动
uv run vh evaluate compare --comparison-id selection_revision --kind revision \
  --run-id before --run-id after
```

`--dataset-dir` 指向含 `manifest.jsonl` 与 `annotations.json` 的目录，`--output-dir` 指定产物目录。`--limit` 只选前 N 条符合标注条件的视频；重复传入 `--video-id` 可冻结指定子集，评分分母都只包含实际选中的视频。只有显式 `--include-silver` 才加入银标。新评测必须使用新 ID；恢复不接受重新选样或覆盖任务配置。

## 固定协议

`configs/evaluation/agent_quality.json` 使用统一的 3–30 秒片段范围、不限制最终数量及总时长、允许完整片段重叠。目的是检查事件发现与成片能力，减少输出预算对召回的干扰。默认任务目标是挑选有看点、能独立看懂的片段，保留必要铺垫和反应，剪辑简洁流畅；不会按每条标注自动改变约束。若评估实际产品配置，使用另一份任务文件和新的 run ID。

输入格式沿用项目数据集：manifest 每条至少包含 video_id、title、path、duration_sec，可选 language、drama_id、split；annotations 是数组，每条包含 video_id、annotation_status 和 highlights，片段至少有 start_sec/end_sec。标注只给评分器，检测任务不包含答案。

当前数据集有 20 条视频：10 条 `labeled` AI 复核标签和 10 条 `silver` Agent 银标。前 5 条复标过程见[旧标注审查](annotation-audit-20260929.md)，新增 5 条见[标注扩充审计](annotation-expansion-20261001.md)。全部 `human_verified=false`，不能称为人类金标；其中银标来自被评链路，只适合压力测试和流程诊断。默认评测使用 10 条 AI 复核短视频，显式 `--include-silver` 才把 10 条长视频银标加入，形成 20 条运行集。

## 产物

```text
outputs/evaluations/{run_id}/
  protocol.json        冻结的视频列表、完整标注记录、完整任务、文件哈希、配置及代码指纹
  source/              当轮 Python 源码快照，不包含密钥
  jobs/{video_id}/
    execution.json     每次尝试的开始时间、耗时、完成/失败状态和停止原因
    result.json        正常检测结果，可能 complete 或 partial
    state.json         检查点及事件、成片状态
    trace.jsonl        工具动作、模型调用和复核记录
    ...                冻结剪辑与其他正常检测产物
  metrics.json         全体及仅已完成样本指标、匹配关系、未命中索引和逐视频数据
  report.md            可直接阅读的指标和标注/预测对照
```

每完成一个视频更新报告。原始标注之后发生变化，不会改变旧评测的评分答案。恢复检查代码和配置一致性，并校验待运行视频哈希；改变实现后新建评测。源码快照用于审计。当前请求固定 `VH_GENERATION_SEED`，但托管模型只承诺尽力复现，仍需用 `evaluate compare` 检查重复运行，而不能假设逐字一致。

## 指标口径

- **完成率**：complete 视频数 / 本轮全部视频数。partial、failed、running、not_started 均不算完成。
- **事件发现 P/R/F1**：对每个标注事件取“可接受边界最大 IoU”和“结构化决定性证据保留率”的较大值，在 0.3/0.5/0.7 阈值下做一对一匹配。它回答核心事件是否被发现；宽片段仍会在原始边界 IoU、输出时长和人工修改量上受到诊断。未完成视频不提交最终预测，但其标注保留在召回分母中。
- **95% 区间**：Precision 与 Recall 同时保存 Wilson 区间。当前必选事件很少，点估计应与原始计数和区间一起阅读。
- **标注来源分组**：`cohort_metrics` 分开报告人类确认、盲标 AI 复核、看过旧预测的 AI 复核、silver 和来源未声明的数据，避免混合总分掩盖数据依赖。
- **运行稳定性**：`runtime_metrics` 汇总首轮完成、多次尝试、模型调用、请求错误、缺失工具调用与无效工具请求。视频耗时为全部尝试之和；任一次未计时，该视频及全组总耗时为未知，中位数/P90 只使用计时完整的视频。`timed_videos` 明示计时分母；请求耗时包含内部重试，不能直接相加为并发任务的墙钟时间。
- **completed_only**：仅完成视频的指标，配合主指标区分执行问题与内容问题，不能单独作为成绩。
- **事件池召回 / 复核后召回**：已完成视频的非合并/非否决事件证据区间，以及 ready 成片对标注的匹配。主口径分母包含全体标注。事件池区间由必要证据外包得到，与最终成片边界不同；它是定位问题的代理指标，不等同于语义发现召回。
- **逐视频对照**：G/P 索引从 0 开始；报告列出标注、预测、IoU≥0.5 匹配、未匹配标注和预测。metrics.json 另保留 partial 片段，但不冒充最终输出。

IoU 只比较时间，不能判断剧情身份、独立可懂性和观看价值。人工采用率及关键证据缺失率需另行人工审核；Agent 自评分不当作准确率。零预测/零标注时当前实现相关比率为 0，应同时查看原始计数。

重复运行比较须保持共有视频的标注与任务，以及 Agent 代码和完整配置一致。`evaluate compare` 会检查这些条件，只比较运行间共有的视频，并报告完成一致率、输出数量差、不同 IoU 阈值的输出 F1 及匹配边界 IoU。评分器或命令行的改动不会伪装成 Agent 变化。输出一致不代表正确；它只用于判断同协议重复运行的漂移。旧 current_v1 的历史分数使用旧评分实现，不与新成绩直接作增益比较。

默认 `--kind repeat` 要求 Agent 代码一致，用于重复运行稳定性。`--kind revision` 允许 Agent 实现变化，仍严格核对完整配置以及共有视频的哈希、任务和标注；每轮源码指纹写入 `comparison.json`。两种模式都对共有样本报告事件精度和召回，不能只以输出一致性判断改进。

首轮实际结果见 [2026-09-29 开发集基线](evaluation-baseline-20260929.md)。

当前状态机的五视频完成率、分阶段召回和边界诊断见 [2026-09-30 状态机与编辑协议验证](state-v8-evaluation-20260930.md)。

当前单看点事件契约的定向实跑见 [2026-09-30 v9 粒度验证](state-v9-granularity-evaluation-20260930.md)。

组件对照、事件粒度和人工编辑成本的固定口径见 [Agent 消融协议](agent-ablation.md)。现有 10 条 AI 复核视频可以用于短视频开发诊断，不能产生人工采用率、拆分率或编辑时间结论。


## 事件口径（v6）

metrics schema_version=3 同时保留原始 iou_metrics、event_metrics、置信区间和标注来源分组。
事件发现匹配把推荐片段及其 alternative_clips 看作一个目标，并在可接受边界最大 IoU 与结构化 decisive 证据保留率之间取较大值；同一目标只能命中一次。原始 `iou_metrics` 单独衡量边界贴合程度，因此不会把“核心事件已找到但粗剪较宽”混写成事件漏检。
先匹配推荐事件，再对剩余预测匹配 optional_highlights。可选命中单列，不计入推荐召回，也从精度分母排除。
未匹配预测若仍与已命中的目标达到阈值，记为 duplicate_predictions，并保留在精度分母内。它是时间重复诊断，不是语义重复的证明。
候选召回包含已登记但最终 rejected 的事件，用于区分没发现与发现后排除；旧时间指标保留原口径。
逐视频 event_thresholds 保存匹配索引，output_count / output_duration_sec 保存实际输出规模。失败与未完成的最终预测仍为空。
离线重算只使用 protocol.json 冻结标注；AI 复标仍属于开发诊断，不能据此报告人工采用率。

对命中的推荐事件，decisive_evidence_coverage 记录结构化 `role=decisive` 时间区间的保留时长占比；时间 IoU 命中仍可能只保留部分核心证据。只有文字说明、没有决定性证据时间区间的银标不产生该条诊断。
