"""Measure frozen-test answer loss for the final SFT adapter."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

import torch
from accelerate import PartialState
from data import InputConfig, SFTCollator, VideoInputs, WindowDataset, input_record, read_rows
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class EvaluationProgress(TrainerCallback):
    def __init__(self, batches: int, enabled: bool):
        self.batches = batches
        self.enabled = enabled
        self.current = 0

    def on_prediction_step(self, args, state, control, **kwargs):
        self.current += 1
        if self.enabled and (self.current % 10 == 0 or self.current == self.batches):
            print(f"test_loss_batches={self.current}/{self.batches}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="已完成 SFT 的训练目录")
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "completed.json").is_file():
        parser.error("训练尚未完整结束")
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_num_threads(4)
    distributed = PartialState()
    metadata = json.loads((output / "run.json").read_text())
    dataset = Path(metadata["dataset"])
    rows = read_rows(dataset, "test.jsonl")
    manifest = json.loads((dataset / "manifest.json").read_text())["files"]
    if manifest["test.jsonl"] != metadata["dataset_manifest"]["test.jsonl"]:
        raise ValueError("测试数据与本轮训练冻结的划分不一致")
    test_dramas = {row["drama_id"] for row in rows}
    for name in ("sft/train.jsonl", "val.jsonl"):
        if test_dramas & {row["drama_id"] for row in read_rows(dataset, name)}:
            raise ValueError("测试集与训练或验证存在整剧交叉")
    config = InputConfig(
        **{key: metadata["input"][key] for key in ("fps", "visual_tokens", "max_length")}
    )
    if input_record(config) != metadata["input"]:
        raise ValueError("当前输入构建规则与训练记录不一致")

    # Keep the lock open until all ranks finish; no training state is modified.
    lock = None
    if distributed.is_main_process:
        lock = (output / ".test.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"test_windows={len(rows)} test_dramas={len(test_dramas)}", flush=True)
    distributed.wait_for_everyone()
    adapter = output / "adapter"
    processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        adapter,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        device_map={"": distributed.device},
    )
    if not model.active_adapters():
        raise ValueError("最终 LoRA 未加载")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    apply_liger_kernel_to_qwen3_5(model=model)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            per_device_eval_batch_size=1,
            prediction_loss_only=True,
            remove_unused_columns=False,
            bf16=True,
            tf32=True,
            ddp_find_unused_parameters=False,
            ddp_broadcast_buffers=False,
            dataloader_num_workers=2,
            dataloader_prefetch_factor=2,
            report_to="none",
            disable_tqdm=True,
            seed=metadata["training"]["seed"],
        ),
        processing_class=processor,
        eval_dataset=WindowDataset(rows),
        data_collator=SFTCollator(VideoInputs(processor, config)),
        callbacks=[
            EvaluationProgress(
                math.ceil(len(rows) / distributed.num_processes), distributed.is_main_process
            )
        ],
    )
    metrics = {
        key: value
        for key, value in trainer.evaluate(metric_key_prefix="test").items()
        if key.startswith("test_")
    }
    if distributed.is_main_process:
        step = json.loads((output / "trainer_state.json").read_text())["global_step"]
        with (adapter / "adapter_model.safetensors").open("rb") as stream:
            adapter_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        result = {
            **metrics,
            "test_windows": len(rows),
            "test_videos": len({row["video_id"] for row in rows}),
            "test_dramas": len(test_dramas),
            "step": step,
            "dataset_sha256": manifest["test.jsonl"],
            "adapter_sha256": adapter_sha256,
            "input": metadata["input"],
        }
        (output / "test_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, **metrics}) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    distributed.wait_for_everyone()
    trainer.accelerator.end_training()
    if lock is not None:
        lock.close()


if __name__ == "__main__":
    main()
