# 输入契约审计

判断一个输入是否应该保留，只看它是否会改变证据、状态、约束或最终结果。只被记录、回显，或要求模型填写却不参与任何决策的字段应删除。

## 任务输入

| 输入 | 作用 | 结论 |
|---|---|---|
| `video_path` | 提供原始音画，也是所有事件确认和成片复核的依据 | 必需 |
| `instruction` | 同时进入主 Agent、独立成片复核和最终选择 | 必需 |
| `language` | 决定 ASR、OCR 和本地模型的语言处理 | 条件必需；只在这些来源启用时生效 |
| `subtitle_path` | 增加可检索的带时间文本，辅助召回和定位 | 有用的可选证据 |
| `max_highlights` | 最终选择的数量预算，`null` 表示不限 | 有用；显式值必须大于 0 |
| `min_clip_sec` | 生成粗边界时补齐过短片段 | 有用的剪辑约束 |
| `max_clip_sec` | 约束单段长度；证据放不下时明确标为不可行 | 有用的剪辑约束 |
| `total_duration_sec` | 约束所有选中片段的总时长 | 有用的可选预算；显式值必须大于 0 |
| `allow_overlap` | 控制最终选择是否允许不同事件共享上下文 | 有用的取舍约束 |
| `video_id` | 作为外部视频标识，并稳定关联本地模型特征 | 有用的身份输入 |
| `job_id` | 隔离输出、检查点和媒体文件 | 必需的运行标识 |
| `resume` | 从同一输入、模型、提示词和约束的检查点继续 | 有用的运行控制 |

`job_id`、`resume` 和大多数情况下的 `video_id` 属于执行层，不应写进模型提示词。`language` 和 `subtitle_path` 属于证据配置，也不应被当作内容偏好。

## Agent 工具输入

| 工具 | 有效输入 | 输入如何被消费 |
|---|---|---|
| `inspect_interval` | `start_sec`、`end_sec`、`question`、`sampling_fps` | 区间决定媒体；问题保留在原生函数调用中作为本轮观看目标；帧率改变 Gemini 的视频采样 |
| `search_transcript` | `query`、时间范围、分页参数 | 决定检索结果，并持久化查询和已读行 |
| `propose_highlights` | `action` 及各动作自己的分页或核验字段 | `list` 只分页；`resolve` 必须提交位置 ID、完整视频观察、判断理由，可选关联事件 |
| `record_observations` | 一批 `observation_id`，每条各含 `findings` 或 `no_event_reason` | 一次登记本轮全部已送达视频，并新增或补充事件；`required_spans` 表示成片必留内容，允许沿用已观看的旧证据；成功后才增加页面覆盖率 |
| `update_event` | 完整 `event`；其中 `id` 指向已有事件，`expected_version` 原样使用状态返回的 `event.version` | 用版本检查修订事件，并使旧成片失效；服务器版本不作为写入字段 |
| `read_state` | `collection`、分页参数，或单个 `event_id` | 按需回查持久事件、观察、字幕、查询和页面；最终候选池直接进入选择状态 |
| `select_highlights` | 每个事件的 `event_id`、`selected`、`score`、`reason`、可选 `duplicate_of` | 依据实际成片所见，对完整候选池逐项取舍；模型只使用统一的事件 ID，运行时映射成片并原子冻结、持久化全部决定、生成最终结果 |

动作型输入现在按动作严格校验。`propose_highlights.list` 不接受核验字段，`resolve` 不接受分页字段；按 `event_id` 读取状态时不接受分页字段。

最终选择视图只使用当前事件 ID、版本、实际边界和复核的 `visible_event`；采用价值 `score` 和理由由完整池选择生成。初看描述和原始事件的价值理由通过按需回查读取，不在选择视图中预先传入。进入选择时更新会话工作段，移除扫描历史和旧工具回显；持久记录和完整候选池保留。选择中的补看和修订沿用同一会话，核对结论持久化进入 `inspection_memory`，新缺陷进入 `repair_events`。

## 已删除的无效输入和输出

- `FinalizeInput` 与 `finalize_highlights`：生成、渲染和独立复核是确定性运行阶段，不再消耗 Agent 的 draft 调用。
- `AcceptReviewsInput` 与 `accept_reviews`：身份确认和全局取舍读取同一批事件与复核结果，合并进原子的 `select_highlights`，避免连续两次模型决策。
- `ScanInput` 与 `scan_video`：按时间顺序覆盖原片由运行时调度，模型不再为“读取下一页”消耗一次决策。
- 候选阅读前置条件：完整候选池已直接进入选择上下文，删除选择前的重复 `read_state candidates` 往返。
- `EventContent.highlight_type`：初看分类会把早期推断带到公开结果；删除重复分类，类型只由实际成片复核生成。
- 选择候选中的 `event_reason`：早期价值理由会干扰基于成片事实的比较；移除该副本，需要核对时读取原始事件。
- `ReviewResult.score`、`reason` 和选择视图的 `review_reason`：独立观看无法比较完整候选池，删除重复的价值判断；最终选择只依据成片事实，评分和理由由同一决定生成。成片类型保留作公开分类，不参与去重判断。
- `ReviewResult.boundary`：独立复核生成首尾建议，但产品结果和人工编辑界面都不读取它。当前只输出粗边界，人工编辑直接调整。
- 每条 finding 必须引用当次观察的限制：补看用于修订首尾或判断时，不必改变原有必留证据。事件仍须引用已经观看、时间范围有效的证据，版本变化仍使旧成片和复核失效。
- `FrameSample.change_score` 和 `semantic_change_score`：没有任何候选召回、特征提取或排序模块读取；同时删除了逐帧灰度差计算。
- `VideoInfo.title` 和 `language`：标题由任务输出层从文件名生成，语言由任务证据配置持有，媒体探测结果中的副本无人读取。
- `AudioEvent.confidence`：SenseVoice 没有提供该值，本地特征也不消费它。
- `audio_energy_per_second`：没有调用方；本地模型已经从原始音频提取冻结特征，并在特征编码阶段计算实际使用的能量先验。
- `inspect_interval` 结果中的 `question` 回显：问题已经存在于原生函数调用中，重复回显不增加信息。

## 仍需用消融确认的条件输入

- 字幕/ASR 的价值看“可用看点召回率”和“每个采用片段的观看次数”，不能只看是否调用过。
- 本地候选模型的价值看它是否在不降低可用看点召回率的前提下减少全片反复查看。没有配置 checkpoint 时，它不是当前运行的输入。
- `sampling_fps` 应通过短动作、手机文字和快速反应样本验证；如果从未改变判断，可以改成内部策略并从工具契约移除。
- `total_duration_sec` 当前由 API/CLI 支持，编辑端没有暴露。若产品只交付候选池而不承诺成片总时长，应在界面设计确定后删除该预算，而不是长期保留隐藏能力。
