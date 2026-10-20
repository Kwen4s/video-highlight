"""Qwen3.5 highlight LoRA SFT using the frozen stage dataset."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import psutil
import torch
from accelerate import PartialState
from data import (
    InputConfig,
    SFTCollator,
    VideoInputs,
    WindowDataset,
    input_record,
    messages,
    read_rows,
)
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import disable_progress_bar

ROOT = Path("/data0/wkw/video-highlight")


class ActivationCheckpointing(TrainerCallback):
    """Keep memory-heavy linear blocks checkpointed; retain some attention activations."""

    def __init__(self, full_attention_stride: int):
        self.stride = full_attention_stride

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        attention_index = 0
        for layer in model.modules():
            if isinstance(layer, modeling_qwen3_5.Qwen3_5DecoderLayer):
                layer.gradient_checkpointing = True
                if layer.layer_type == "full_attention":
                    layer.gradient_checkpointing = attention_index % self.stride == 0
                    attention_index += 1
        if state.is_world_process_zero:
            print(f"full_attention_checkpoint_stride={self.stride}", flush=True)


def build_model(path: Path, device: torch.device):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        device_map={"": device},
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    apply_liger_kernel_to_qwen3_5(model=model)
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.language_model.") and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise ValueError("未找到语言模型的 LoRA 目标模块")
    return get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=targets,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    ), targets


class TrainingSchedule(TrainerCallback):
    """Apply the requested intervals after Trainer restores checkpoint state."""

    def on_train_begin(self, args, state, control, **kwargs):
        state.compute_steps(args, state.max_steps)
        if state.is_world_process_zero:
            print(
                f"eval_steps={state.eval_steps} save_steps={state.save_steps} "
                f"save_total_limit={args.save_total_limit}",
                flush=True,
            )


class ProgressLog(TrainerCallback):
    def __init__(self, output: Path):
        self.path = output / "metrics.jsonl"
        self.started = time.monotonic()
        self.previous_elapsed = 0.0
        if self.path.exists():
            lines = self.path.read_text().splitlines()
            if lines:
                self.previous_elapsed = json.loads(lines[-1])["elapsed_sec"]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        item = {
            "step": state.global_step,
            "epoch": state.epoch,
            **(logs or {}),
            "elapsed_sec": round(self.previous_elapsed + time.monotonic() - self.started, 2),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "rss_gib": round(psutil.Process().memory_info().rss / 1024**3, 3),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/vl-001")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--visual-tokens", type=int, default=8192)
    parser.add_argument("--full-attention-checkpoint-stride", type=int, default=1)
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn", force=True)
    distributed = PartialState()
    if args.effective_batch < 1 or args.effective_batch % distributed.num_processes:
        parser.error("--effective-batch 必须为进程数的正整数倍")
    accumulation = args.effective_batch // distributed.num_processes
    if args.full_attention_checkpoint_stride < 1:
        parser.error("--full-attention-checkpoint-stride 必须大于零")
    if distributed.is_main_process:
        if args.output.exists() and not args.resume:
            raise FileExistsError("输出目录已存在，请使用 --resume 或指定新目录")
        args.output.mkdir(parents=True, exist_ok=True)
    distributed.wait_for_everyone()
    checkpoint = get_last_checkpoint(str(args.output)) if args.resume else None
    if args.resume and checkpoint is None:
        raise FileNotFoundError("没有可恢复的训练检查点")
    set_seed(42)
    disable_progress_bar()
    torch.set_num_threads(4)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    config = InputConfig(visual_tokens=args.visual_tokens)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    train = read_rows(args.dataset, "sft/train.jsonl")
    val = read_rows(args.dataset, "val.jsonl")
    if args.sample_index is not None:
        train = [train[args.sample_index]]
    # Check all text before GPU loading. Video tokens and timestamps are bounded separately.
    lengths = []
    for row in train + val:
        text = processor.apply_chat_template(
            messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        lengths.append(len(processor.tokenizer.encode(text, add_special_tokens=False)))
    if distributed.is_main_process:
        print(f"text_tokens_max={max(lengths)}, train={len(train)}, val={len(val)}", flush=True)
    if max(lengths) + config.visual_tokens + 1024 > config.max_length:
        raise ValueError("完整输入可能超出序列预算，需调整输入预算后再训练")
    model, targets = build_model(args.model, distributed.device)
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if any("language_model" not in name or "lora_" not in name for name, _ in trainable):
        raise ValueError("视觉模块或非 LoRA 参数被错误解冻")
    record = {
        "base_model": str(args.model),
        "dataset": str(args.dataset),
        "dataset_manifest": json.loads((args.dataset / "manifest.json").read_text())["files"],
        "input": input_record(config),
        "train_samples": len(train),
        "val_samples": len(val),
        "max_text_tokens": max(lengths),
        "lora_targets": targets,
        "trainable_parameters": sum(count for _, count in trainable),
        "gpu": torch.cuda.get_device_name(),
        "stage": "sft",
        "fused_causal_conv": modeling_qwen3_5.causal_conv1d_fn is not None,
        "answer_only_projection": True,
        "resume_checkpoint": checkpoint,
        "training": {
            "epochs": 1,
            "learning_rate": 5e-5,
            "effective_batch": args.effective_batch,
            "world_size": distributed.num_processes,
            "gradient_accumulation_steps": accumulation,
            "full_attention_checkpoint_stride": args.full_attention_checkpoint_stride,
            "max_steps": args.max_steps,
            "eval_steps": 50,
            "save_steps": 50,
            "save_total_limit": 1,
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "seed": 42,
        },
    }
    if distributed.is_main_process:
        (args.output / "run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
        model.print_trainable_parameters()
    training = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=math.ceil(math.ceil(len(train) / args.effective_batch) * 0.05)
        if args.max_steps < 0
        else 0,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        logging_steps=1,
        save_steps=50,
        save_total_limit=1,
        eval_steps=50,
        eval_strategy="steps",
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        report_to="none",
        seed=42,
        data_seed=42,
        optim="adamw_torch",
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=training,
        processing_class=processor,
        train_dataset=WindowDataset(train),
        eval_dataset=WindowDataset(val),
        data_collator=SFTCollator(VideoInputs(processor, config)),
        callbacks=[
            TrainingSchedule(),
            ActivationCheckpointing(args.full_attention_checkpoint_stride),
            ProgressLog(args.output),
        ],
    )
    if checkpoint and distributed.is_main_process:
        print(f"resuming_from={checkpoint}", flush=True)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(args.output / "adapter"))
    if distributed.is_main_process:
        processor.save_pretrained(args.output / "adapter")
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    distributed.wait_for_everyone()
    if distributed.is_main_process:
        (args.output / "completed.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
