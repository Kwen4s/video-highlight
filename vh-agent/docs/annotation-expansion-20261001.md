# 五条短剧标注扩充

日期：2026-10-01。目标是把 `datasets/test` 从 15 条扩到 20 条，并增加短视频题材覆盖。新增视频来自五部不同中文短剧，时长 59–67 秒，原始 metadata split 均为 `test`。

## 方法和身份

每条视频先由 `gemini-3.7-flash` 在不读取当前 Agent 结果的情况下完整观看原生音视频并生成结构化草稿，再在新的模型上下文中重新观看原片、逐项复核并返回完整修订。Codex 随后检查覆盖全片的 2 秒画面和候选附近的 1 秒/0.5 秒连续画面，修正事件粒度、人物说法、推荐等级、边界和类型词表。

最终记录统一为：

- `annotation_version=ai_review_v3_20261001`
- `annotation_method=codex_gemini_native_video_double_pass_frame_audit`
- `annotation_tier=ai_reviewed`
- `human_verified=false`
- `prediction_exposure=false`，表示标注过程没有读取当前 Agent 的候选或结果
- `review_status=pending_human_review`

这些仍是 AI 复核标签，只用于开发诊断，不能当作人类金标或独立产品准确率证据。完整中间结果、连续画面和最终记录位于 `outputs/annotation_audit_20261001/{video_id}/`。

## 新增结果

| 视频 | 推荐事件 | 可选事件 | 主要人工修正 |
|---|---|---|---|
| 京婚有染 | 14.5–35.5 家庭团聚；49.5–64.5 接受共同生活 | 37.5–47.2 儿子体贴牵手 | 补出模型遗漏的独立内心转变；删除“下集预告”误判 |
| 东北爱情往事2 | 35.0–59.0 当面要求经理安排亲戚工作 | 无 | 压到 24 秒并保留经理答应和女子追问 |
| 帝后也疯狂 | 38.0–50.0 七年信任破裂质问 | 13.5–31.5 皇帝单方面指控 | 删除推荐片段前 6 秒静止停顿；明确指控不等于事实 |
| 督军夫人竟是赊刀人2 | 46.5–61.5 玄门弟子奉命出山 | 0.0–22.5 同去幽灵船；23.0–41.5 神秘鳞片线索 | 把前段两个不同看点拆开，修正错误的背景归属 |
| 绝代枭雄 | 0.0–23.8 威胁家人后被踢飞 | 43.0–58.0 保镖围攻悬念 | 向前补齐完整威胁因果；围攻未交手，只列可选悬念 |

新增共 6 个推荐事件、5 个可选事件。加上原有 5 条 AI 复核样本，默认 `labeled` 开发集现在有 10 条；加上 10 条银标，`--include-silver` 的运行集共有 20 条。

## 使用方式

```bash
# 10 条 AI 复核短视频，用于默认开发诊断
uv run vh evaluate run --run-id react_v10_reviewed10 --task-file configs/evaluation/agent_quality.json

# 全部 20 条，包含同链路生成的银标，只用于执行压力测试
uv run vh evaluate run --run-id react_v10_all20 --include-silver --task-file configs/evaluation/agent_quality.json
```
