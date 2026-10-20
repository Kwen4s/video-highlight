# 短剧高光检测模型

更新：2026-10-20。Qwen3.5-9B 的 SFT 和 GSPO 训练已完成，当前采用验证集选出的第 150 步 GSPO 权重。模型提供精彩核心的位置，Agent 负责补充前后文、判断和剪辑；候选工具已接入 Agent，内部保持训练时的窗口与上下文。

## 当前模型与效果

大文件根目录为 `/data0/wkw/video-highlight/`：

| 内容 | 路径 |
| --- | --- |
| 基座 | `models/Qwen3.5-9B/` |
| 最佳 LoRA | `runs/qwen35-9b-gspo-001/adapter/` |
| 固定数据 | `datasets/vl-001/` |
| 评分与视频报告 | `runs/qwen35-9b-gspo-001/effect/report.json`、`index.html` |

在 4 部独立测试短剧的 72 个视频、446 个完整监督窗口上，参考高光共 419 段。按 **时间 IoU ≥ 0.3** 一对一匹配：

| 精确率 | 召回率 | F1 | 命中高光 |
| ---: | ---: | ---: | ---: |
| 66.59% | 68.02% | 67.30% | 285 / 419 |

模型共输出 428 段，中位长度 6.5 秒，平均生成 2.74 秒/窗口。138 个空窗口中有 48 个误报（34.78%）；当前重点是控制误选。窗口结果用于衡量检测能力，Agent 的整片效果在接入后评估。

训练指标见 [SwanLab](https://swanlab.cn/@kevinw/video-highlight/runs/qwen35-9b-gspo-001)。权重、数据、预测和指标已保留；两阶段的恢复检查点已清理，后续训练从保留权重初始化。

## 输入与输出

每次负责 20 秒，前后各看 15 秒，结合此前最多 90 秒及当前的 ASR 对白、观看范围内的 OCR 文字。视频以 2 FPS 采样，视觉预算为 8,192 token，总序列预算为 16,384 token。

模型输出精彩实际发生的台词、动作或反应，通常几秒，优先 15 秒以内。一个回答可包含多段，也可为空：

```json
{"segments":[{"start_sec":18.2,"end_sec":23.6}]}
```

时间使用观看窗口的相对秒数；核心中点决定负责窗口，跨边界核心保留完整起止。映射回原片时加 `video.start_sec`，合并结果时去重。无高光返回 `{"segments":[]}`。

输入、采样与检测提示词统一在 [`detector_inputs.py`](src/vh_agent/detector_inputs.py)，训练通过 `training/data.py` 复用。现有标签主要由 1 FPS 画面和对白生成，适合核心区间定位。

## 数据与训练方法

固定数据的划分、字段和生成命令见 [vh-data README](../vh-data/README.md)。SFT 使用 4,778 个窗口，GSPO 使用其中 908 个；验证 550 个，测试 446 个，按整部剧隔离。

短剧的铺垫与精彩常共享人物、场景和情绪，差别集中在揭晓、反击和结果发生的瞬间。GSPO 优先采样标注中发生过取舍变化的易混淆窗口，并补充相同空／单／多高光分布的窗口，在共同语境中学习选什么、选几个、圈在哪里。

两阶段均冻结视觉编码器和连接模块，训练语言部分的 LoRA；rank 32、alpha 64，共 86,556,672 个可训练参数。

| 配置 | SFT | GSPO |
| --- | --- | --- |
| 框架 | Transformers Trainer + Liger | verl V1 + FSDP2 + vLLM |
| 训练目标 | 仅监督答案 token | 序列级策略优化 |
| 学习率 | 5e-5 | 1e-6 |
| 有效 batch | 16 个窗口 | 16 个窗口，每窗 4 个回答 |
| LoRA dropout | 0.05 | 0 |
| 轮数／更新数 | 1 轮／299 步 | 3 轮／168 步 |

GSPO 采样温度 0.8、预热 5%，裁剪范围为 3e-4 / 4e-4，沿用 SFT 的视频与文字输入。奖励为预测区间集合 P 与参考集合 G 的软 F1，M 是按时间 IoU 最大权重的一对一匹配：

```text
R(P,G) = 2 × Σ[(p,g) ∈ M] tIoU(p,g) / (|P| + |G|)
```

双方都空奖励为 1，仅一方为空或 JSON 无效为 0。无效区间计入预测数量但不能匹配；重复、多选、漏选和过宽边界都会降分。

每 50 步及末步保存并验证。SFT 报告验证 loss；GSPO 在全部验证窗口生成区间，以 F1@0.5 选权重，同分依次比较召回@0.3 和空窗口误报。固定测试集用于最终报告。SwanLab 记录训练指标、奖励差异、验证效果和耗时。

## 运行

以下命令在项目根目录执行。现有环境直接激活；安装新环境使用 `bash vh-agent/training/setup_env.sh`，依赖版本由脚本固定。

```bash
source vh-agent/training/env.sh
```

重新评分并生成视频报告，复用已有预测：

```bash
python vh-agent/training/score.py /data0/wkw/video-highlight/runs/qwen35-9b-gspo-001
```

生成固定测试窗口的预测，已有结果会续用：

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 \
  torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  vh-agent/training/predict.py /data0/wkw/video-highlight/runs/qwen35-9b-gspo-001
```

新一轮 GSPO 使用新的输出目录，默认继承已保留的 SFT 权重：

```bash
bash vh-agent/training/run_gspo.sh /data0/wkw/video-highlight/runs/qwen35-9b-gspo-002
```

入口自动同步 SwanLab，结束后生成固定测试报告。密钥配置在 `training/.env` 的 `SWANLAB_API_KEY`。手动补录指标可运行 `python vh-agent/training/track.py 训练目录`。

实现分工：`detector_inputs.py` 负责统一输入，`data.py` 负责训练批次，`sft.py` 负责监督训练，`gspo*.py` 负责强化学习与视频接口，`reward.py` 负责奖励，`predict.py`、`score.py`、`examples.py` 负责效果报告，`track.py` 负责指标同步。

本地服务启动与 Agent 工具输入输出见 [README](README.md)。Agent 调用 `detect_highlights` 只传原片开始、结束秒数，工具负责分窗与时间转换。
