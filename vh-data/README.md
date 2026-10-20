# 短剧检测数据

负责视频导入、模型标注与独立复核、固定训练数据生成、检测结果反馈。标签圈定精彩发生的核心，Agent 后续补充前后文和剪辑范围。

## 当前数据入口

唯一训练目录：`/data0/wkw/video-highlight/datasets/vl-001/`。

```text
vl-001/
  sft/train.jsonl
  gspo/train.jsonl
  val.jsonl
  test.jsonl
  schema.json
  manifest.json
  source/
    annotations.jsonl
    dialogues.jsonl
    provenance.json
    manifest.json
```

| 阶段 | 空答案 | 单高光 | 多高光 | 总数 |
| --- | ---: | ---: | ---: | ---: |
| SFT | 1,466 | 2,318 | 994 | 4,778 |
| GSPO | 279 | 440 | 189 | 908 |

验证集 550 个窗口，测试集 446 个窗口。整视频来源共 952 条、53 部剧；按整部剧固定划分，训练／验证／测试分别为 44／5／4 部剧、797／83／72 个视频。

SFT 使用全部训练窗口。GSPO 包含 454 个历史易误选位置关联窗口和 454 个分层补充窗口，空／单／多高光比例接近 SFT，各类别内按剧轮流抽取，种子为 42。历史判断只决定采样优先级，答案沿用当前有效标签。GSPO 的 908 条均来自 SFT，是继续优化；验证和测试剧独立。

`manifest.json` 记录参数、统计、样本来源和文件指纹；`source/` 保存整视频标签、完整对白及来源。

## 生成数据

在项目根目录安装并配置 `vh-data/.env`：

```bash
vh-agent/.venv/bin/uv pip install --python vh-agent/.venv/bin/python -e ./vh-data
vh-agent/.venv/bin/vh-data status
# 下一批使用新的版本目录；已有目录不会覆盖
vh-agent/.venv/bin/vh-data prepare-training /data0/wkw/video-highlight/datasets/vl-002
```

`prepare-training` 是唯一生成命令，直接使用数据库当前有效标签，一次发布两个阶段、共享验证／测试及来源，不调用标注 API。默认检测区域 20 秒、前后各看 15 秒、此前对白最多 90 秒；可通过 `--window-sec`、`--context-sec`、`--dialogue-history-sec` 调整。20 秒是负责范围，与高光长度无关。

默认 GSPO 数量为关联历史窗口的两倍，最多为全部训练窗口；`--gspo-size` 可指定数量，`--seed` 控制抽样。没有历史线索时明确指定数量。导出中断不发布半成品。

## 固定窗口格式

每行一个窗口，结构以 `schema.json` 为准：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 视频 ID 与负责范围组成的唯一编号 |
| `video_id`、`drama_id`、`label_id` | 视频、短剧及不可变标注版本 |
| `video` | `path`、`sha256`、`start_sec`、`end_sec`；观看范围使用原片秒数 |
| `detect_range` | 本次负责范围，使用观看窗口相对秒数 |
| `dialogue` | 带时间的文字与 `source=asr/ocr`；相对秒数，负数表示前情 |
| `segments` | 完整高光核心的相对起止；空列表表示没有高光 |

输入由视频、负责范围和对白构成。SFT 将 `segments` 作为目标回答；GSPO 将其交给奖励函数。

高光中点决定所属窗口，保留跨边界的完整核心。只有负责范围被明确正负标注完整覆盖、核心完整可见的窗口才进入训练；未知区域保持未知。原视频按路径和指纹引用，不复制重叠窗口。恢复原片时间时加上 `video.start_sec`。

`source/annotations.jsonl` 和 `source/dialogues.jsonl` 按视频 ID 关联，使用原片时间，供后续整片评估。当前 SFT、GSPO 和固定窗口测试均已完成，模型与效果见 [VL_TRAINING.md](../vh-agent/VL_TRAINING.md)。

## 继续生产与反馈

```bash
vh-agent/.venv/bin/vh-data ingest /path/to/videos.jsonl
vh-agent/.venv/bin/vh-data run --limit 10 --retry-failed
vh-agent/.venv/bin/vh-data inspect vid_example
vh-agent/.venv/bin/vh-data requeue vid_example
vh-agent/.venv/bin/vh-data feedback /path/to/predictions.jsonl --model-id detector-v1
```

导入清单每行包含 `video_id`、`drama_id`、`path`、`duration_sec`、`language`，可提供 `sha256`。相对路径以清单目录为准；按内容去重，同剧固定进入同一分组。新增视频不改变已有分组。

反馈每行只含视频 ID 和原片区间：

```json
{"video_id":"vid_example","segments":[{"start_sec":12.5,"end_sec":18.2}]}
```

反馈用于安排训练视频重新标注，随后由标注与复核流程生成新标签。反馈文件采用上面的原片区间格式；当前模型输出的窗口相对时间需先转换。

标注先连续观看，再独立复核全片；候选只提供位置，复核依据原视频决定取舍和边界。只剩重叠问题时局部重看。精彩核心优先 15 秒以内，允许短至一两个动作；普通内容标负例，证据不足保持未知，没有固定高光配额。

提示词集中在 [label.md](src/vh_data/prompts/label.md)、[review.md](src/vh_data/prompts/review.md)、[highlight_types.md](src/vh_data/prompts/highlight_types.md)。修改模型或提示词后，指定视频 `requeue` 再运行，旧版本保留追溯。

## 持久化与运行

配置见 [.env.example](.env.example)，密钥仅保存在本机 `.env`。`outputs/data/data.sqlite3` 保存任务和标签；同目录 `readings/`、`dialogues/` 保存成功调用和带时间文字。成功页面可复用，长视频分页覆盖，过大请求拆页。网络错误最多等待 5 秒再重试，同页连续失败三次让出 worker 后重新排队；无效标注有次数限制。请求采用网络空闲超时，持续收到数据时继续等待。

当前固定数据来自 952 条有效视频。使用 `status` 查看任务状态，`run` 继续生产；数据库、成功调用、对白和媒体缓存用于复用与追溯。
