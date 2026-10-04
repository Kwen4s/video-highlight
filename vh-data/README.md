# 短剧检测数据飞轮

给本地高光检测模型持续整理数据，独立于主 Agent 和产品前后端。

视频清单 → 两次独立观看 → 统一选片复核 → 冻结训练数据 → 小模型漏检反馈 → 下一轮模型复核。

整个流程由模型完成。证据不足的内容保持未知，不强行作出判断。

## 开始

在项目根目录安装：

```bash
cd /home/wkw/video-highlight
vh-agent/.venv/bin/python -m pip install -e ./vh-data -e ./vh-agent
```

配置 `vh-data/.env`（参考 `.env.example`）。标注使用独立的 `VH_DATA_API_KEY`、`VH_DATA_BASE_URL` 和 `VH_DATA_MODEL`；通过流式 Responses 接口请求高思考强度，记录网关返回的模型、思考参数和 token 用量。当前使用 `gpt-6-astra`，输入为连续帧、带时间的对白转写和已识别字幕。密钥和生产数据不进入 Git。

```bash
# 查询网关模型，确认 Responses、图片和结构化输出支持后配置 VH_DATA_MODEL
vh-agent/.venv/bin/vh-data models

# 导入原视频清单，不沿用旧标注或旧 split
vh-agent/.venv/bin/vh-data ingest /data1/my_short_drama/metadata/zh/train.jsonl /data1/my_short_drama/metadata/zh/test.jsonl

# 首轮分别生产训练、验证和测试组；后续再扩大每组数量
vh-agent/.venv/bin/vh-data run --limit 5 --split train
vh-agent/.venv/bin/vh-data run --limit 2 --split val
vh-agent/.venv/bin/vh-data run --limit 2 --split test
vh-agent/.venv/bin/vh-data status
vh-agent/.venv/bin/vh-data inspect vid_example

# 指定视频重跑，便于复现与检查粒度；可重复传入 --video-id
vh-agent/.venv/bin/vh-data requeue vid_example
vh-agent/.venv/bin/vh-data run --limit 1 --video-id vid_example

# 页面自动重试；重复运行可恢复中断，并重新尝试先前失败的视频
vh-agent/.venv/bin/vh-data run --limit 100 --retry-failed

# 冻结已完成模型复核的记录，目录名使用这一轮数据版本
vh-agent/.venv/bin/vh-data export vh-data/outputs/snapshots/round-003

# 训练使用现有 vh-agent；需在装有小模型依赖的 Python 环境运行
vh train run --annotations vh-data/outputs/snapshots/round-003/annotations.jsonl --output-dir vh-data/outputs/models/round-003

# 导入小模型预测，优先重新检查漏检或意见不同的视频
vh-agent/.venv/bin/vh-data feedback vh-data/outputs/models/round-003/train_predictions.jsonl --model-id detector-round-003
vh-agent/.venv/bin/vh-data run --limit 20
vh-agent/.venv/bin/vh-data export vh-data/outputs/snapshots/round-004
```

新视频继续导入原清单即可。正式训练需要训练组和验证组都有完成复核的记录。没有预处理记录的新视频会生成转写，默认 CPU/int8 和 beam size 5；可用 `VH_DATA_ASR_MODEL`、`VH_DATA_ASR_DEVICE`、`VH_DATA_ASR_COMPUTE_TYPE` 指定本机模型和设备。已有预处理目录通过 `VH_DATA_PREPROCESS_CACHE` 指定。按视频内容去重；同一短剧用固定哈希分入训练、验证或测试，目标比例 8:1:1，实际数量取决于短剧分布。新增集数不会改变已有分组。验证和测试不参与小模型反馈采样。

## 字段

输入清单为 JSONL，每行一个视频：

| 字段 | 含义 |
| --- | --- |
| `video_id` | 视频的唯一编号 |
| `drama_id` | 所属短剧编号；同一部剧必须相同，用来防止跨组泄漏 |
| `path` | 本机原视频路径；相对路径以清单所在目录为准 |
| `duration_sec` | 清单记录的秒数；模型标注时以实际视频时间轴更新 |
| `language` | 视频语言，如 `zh`、`en` |
| `sha256` | 可选的源文件指纹；提供时会核对实际文件内容 |

模型每次观看只返回简短的 `story_so_far` 和 `segments`。前者记住题材、人物关系和当前事情，只用于理解下一段；每段只有四个字段：

| 字段 | 含义 |
| --- | --- |
| `start_sec` | 开始位置，原视频秒数 |
| `end_sec` | 结束位置，原视频秒数，晚于开始位置 |
| `kind` | `positive` 高光；`negative` 明确普通的片段；`uncertain` 证据不足 |
| `reason` | 一句具体说明，如“拿出录音拆穿指控，对方当场改口” |

训练快照仅保留高光时间段 `highlights` 和明确普通片段 `negative_intervals`；`uncertain` 和空白区域不参与监督。每行还记录输入字段、固定 `split` 和 `label_source=model_reviewed`。小模型对外预测仍只有开始、结束和分数：

```json
{"video_id":"vid_example","segments":[{"start_sec":12.5,"end_sec":21.2,"score":0.81}]}
```

## 模型怎么复核

1. 两次全片观看彼此看不到标注，也看不到小模型预测，各自保留简短剧情上下文。帧和对白都保留原视频时间。已有预处理记录按源文件指纹复用；新视频使用本地 Whisper 转写。转写可能有误，需要与画面字幕和剧情上下文核对。
2. 两次标注按时间重合程度核对，普通内容取共同确认部分。所有高光候选和未确认内容进入选片复核，包括两次一致选中的高光。
3. 选片复核集中看相关页面的原片和前后文，一起确定观看价值、重复和边界。它可以删掉普通剧情、补事件或改边界；一段视频可以没有高光。小模型候选只提供待检查位置。
4. 程序分配跨页片段的归属，保留完整高光边界；落到邻页的候选不会使整份响应重跑。只有仍存在实际重叠时，再用同一个选片步骤检查那一处原片。
5. 确认的高光和普通片段进入训练快照；证据不足的部分保持未知。整条视频没有可训练片段时记录为 `unresolved`，不导出。

训练结束后，用验证组选出的权重生成 `train_predictions.jsonl`，直接交给 `feedback` 命令即可开启下一轮。小模型反馈只决定检查哪些内容，不直接成为标签。复核仍然看原视频。两次观看一致也只是模型标注；验证和测试衡量对模型复核参考的符合程度，不等同于人工认可的高光准确率。

## 提示词在哪里改

- [prompts/label.md](src/vh_data/prompts/label.md)：怎样发现高光、怎样定粒度和边界。
- [prompts/review.md](src/vh_data/prompts/review.md)：怎样复核分歧和小模型反馈。
- [prompts/highlight_types.md](src/vh_data/prompts/highlight_types.md)：从观众角度判断观看价值，附常见看点示例。

标注和复核选择值得单独观看的精彩瞬间，类型作为举例。一般剧情推进和新信息可以标为普通内容；选择依据是具体的观看吸引力。优先选择 15 秒以内的短片段，围绕关键动作或台词保留必要铺垫与反应，背景放进剧情记忆。改提示词、模型、思考强度或输出额度会改变调用缓存标识。对已完成的视频，使用 `vh-data requeue vid_example` 入队后重新运行，旧标注保留供追溯。

时长偏好统一由 `VH_DATA_PREFERRED_HIGHLIGHT_SEC` 设置，默认 15 秒，用于标注与复核提示。所有候选交给模型检查取舍与边界，训练与推理学习最终标注的长度。

## 持久化和检查

`outputs/data/data.sqlite3` 保存源文件指纹、任务状态、不可变标注版本和模型反馈；`readings/` 保存观看范围、提示词、模型、结构化结果、耗时和用量，不保存思考正文；`dialogues/` 记录文字证据的来源和时间轴。成功页面会复用，中断后重新运行即可继续。

默认每页负责 30 秒，前后各带 5 秒上下文；复核再扩一层页面重叠区，集中看当前问题。用 `VH_DATA_PAGE_SEC` 调整页面长度。

短于一页的视频完整观看，长视频逐页覆盖到末尾。短尾页已经完全出现在上一页观看范围中时，直接合并负责范围，保留同一份视频证据。图片请求过大或网关返回 HTTP 413 时拆页，`pages/` 保存完整分页计划。中断后恢复该计划并复用成功调用。跨页重叠逐处复核，避免把长视频里连续重叠的候选拼成一次全片请求。

每页最多 3 次尝试，由 `VH_DATA_REQUEST_ATTEMPTS` 设置。连接或流中断、超时、HTTP 408/409/429/5xx、未完成、空响应和无效标注会自动重试，失败后逐次延长等待。每次尝试有完整期限，默认 300 秒，包括连接和读取整个流，持续保活也会按时结束。SDK 内部重试关闭，避免重试次数叠加。流式响应先检查结束状态，再校验标注；输出额度给思考与标注共用 32768 tokens。

失败尝试写入 `failures/`，记录尝试次数、错误代码或结束原因、耗时，以及已返回的用量和输出。默认通过 SDK 异步调用并行处理两条视频，由 `VH_DATA_WORKERS` 控制；每条视频有独立的剧情状态，单条视频内部逐次请求，所以同时在途的模型请求最多等于 worker 数。领取任务和保存版本均通过 SQLite 事务完成，一个生产进程持有排他锁。单条视频仍失败时保存错误并处理下一条；鉴权或模型配置错误（401/403/404）或连续 3 条视频网关失败时停止领取新视频，已经在途的任务保存完再退出。失败结果不进入训练快照。恢复网关后加 `--retry-failed` 继续；指定 `--split` 或 `--video-id` 时只重试对应的失败视频。

后台标注使用 tmux，运行日志写到 `outputs/labeling.log`。查看当前任务：

```bash
tmux attach -t vh-data-label
tail -f vh-data/outputs/labeling.log
vh-agent/.venv/bin/vh-data status
```

从 tmux 分离按 `Ctrl-b` 再按 `d`，任务继续运行。按 `Ctrl-C` 取消在途模型调用与重试等待；再次执行 `run --retry-failed` 恢复未完成视频，成功视频和成功页面会复用。

状态含义：`queued` 待处理、`labeling` 处理中、`review` 小模型反馈待复核、`ready` 可导出、`unresolved` 全部未确认、`failed` 调用或校验失败。更新有版本检查，旧任务不能覆盖新标注。

`status` 的 `ready_splits` 显示各组可导出的数量，`labels` 统计这些可导出记录的标注来源，`quality` 显示高光总时长、视频总时长和高光覆盖比例；重标中的旧版本可通过 `inspect` 回查。

快照目录包含 `annotations.jsonl`、`provenance.json` 和 `manifest.json`，记录标注来源、版本、各组数量和内容指纹，已有快照不会覆盖。训练读取时再次校验源文件和短剧分组；仅明确标出的区域计算监督。验证组选权重，选完后才评一次测试组，不提前截断检测候选。完全落在未知区域的预测不计假阳性，并单独报告数量。

`manifest.json` 同时统计短剧数、正例/负例/未确认段数、可监督和未知时长，高光时长、覆盖比例和最长高光，方便检查当前数据量、取舍与粒度。覆盖比例是诊断指标，标注没有固定配额。

```bash
vh-agent/.venv/bin/python -m pytest vh-data/tests
```
