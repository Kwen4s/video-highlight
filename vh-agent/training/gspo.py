"""GSPO for video interval detection, using verl's synchronous training engine."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from data import InputConfig, input_record, read_rows
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoProcessor

ROOT = Path("/data0/wkw/video-highlight")
SOURCE = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(output, sft, check_steps):
    source = json.loads((sft / "run.json").read_text())
    if not (sft / "completed.json").is_file():
        raise ValueError("SFT 尚未完整结束")
    dataset = Path(source["dataset"])
    train = read_rows(dataset, "gspo/train.jsonl")
    val = read_rows(dataset, "val.jsonl")
    if {r["drama_id"] for r in train} & {r["drama_id"] for r in val}:
        raise ValueError("训练与验证剧交叉")
    output.mkdir(parents=True, exist_ok=True)
    initial = output / "initial-adapter"
    initial.mkdir(exist_ok=True)
    # RL log probabilities require deterministic dropout, including PEFT modules.
    for path in (sft / "adapter").iterdir():
        if path.is_file() and path.name not in (
            "adapter_config.json",
            "README.md",
            "training_args.bin",
        ):
            target = initial / path.name
            if not target.exists():
                target.symlink_to(path)
    adapter_config = json.loads((sft / "adapter/adapter_config.json").read_text())
    adapter_config["lora_dropout"] = 0.0
    write_json(initial / "adapter_config.json", adapter_config)
    processor = AutoProcessor.from_pretrained(initial, local_files_only=True)
    vp = processor.video_processor
    config = InputConfig(**{k: source["input"][k] for k in ("fps", "visual_tokens", "max_length")})
    if input_record(config) != source["input"]:
        raise ValueError("GSPO 与 SFT 输入约定不同")
    budget = config.visual_tokens * vp.patch_size**2 * vp.merge_size**2 * vp.temporal_patch_size
    loop_path = output / "agent-loop.yaml"
    OmegaConf.save(
        OmegaConf.create([{"name": "video_intervals", "_target_": "gspo_loop.VideoIntervalLoop"}]),
        loop_path,
    )
    import verl

    with initialize_config_dir(
        config_dir=str(Path(verl.__file__).parent / "trainer/config"), version_base=None
    ):
        cfg = compose(config_name="ppo_trainer")
    settings = {
        "data": {
            "train_files": str(dataset / "gspo/train.jsonl"),
            "val_files": str(dataset / "val.jsonl"),
            "custom_cls": {"path": str(SOURCE / "gspo_data.py"), "name": "HighlightDataset"},
            "train_batch_size": 16,
            "val_batch_size": 16,
            "max_prompt_length": config.max_length - 512,
            "max_response_length": 512,
            "dataloader_num_workers": 2,
            "seed": 42,
            "truncation": "error",
            "video_fps": config.fps,
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {
                "do_sample_frames": False,
                "size": {"shortest_edge": vp.size["shortest_edge"], "longest_edge": budget},
            },
        },
        "actor_rollout_ref": {
            "nccl_timeout": 1800,
            "model": {
                "path": source["base_model"],
                "lora_adapter_path": str(initial),
                "lora_rank": adapter_config["r"],
                "lora_alpha": adapter_config["lora_alpha"],
                "exclude_modules": ".*visual.*",
                "enable_gradient_checkpointing": True,
                "use_remove_padding": True,
                "use_fused_kernels": True,
                "fused_kernel_options": {"impl_backend": "triton"},
            },
            "actor": {
                "strategy": "fsdp2",
                "ppo_mini_batch_size": 16,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_epochs": 1,
                "freeze_vision_tower": True,
                "use_torch_compile": False,
                "policy_loss": {"loss_mode": "gspo"},
                "loss_agg_mode": "seq-mean-token-mean",
                "clip_ratio_low": 0.0003,
                "clip_ratio_high": 0.0004,
                "use_kl_loss": False,
                "entropy_coeff": 0,
                "optim": {"lr": 1e-6, "lr_warmup_steps_ratio": 0.05, "weight_decay": 0.0},
                "fsdp_config": {
                    "strategy": "fsdp2",
                    "model_dtype": "bf16",
                    "param_offload": True,
                    "optimizer_offload": True,
                    "use_torch_compile": False,
                },
                "checkpoint": {"save_lora_only": True},
            },
            "rollout": {
                "name": "vllm",
                "load_format": "auto",
                "n": 4,
                "temperature": 0.8,
                "top_p": 0.95,
                "tensor_model_parallel_size": 1,
                "gpu_memory_utilization": 0.55,
                "max_model_len": config.max_length,
                "max_num_seqs": 8,
                "max_num_batched_tokens": config.max_length,
                "enforce_eager": True,
                "log_prob_micro_batch_size_per_gpu": 1,
                "agent": {
                    "num_workers": 2,
                    "default_agent_loop": "video_intervals",
                    "agent_loop_config_path": str(loop_path),
                },
                "checkpoint_engine": {
                    "backend": "naive",
                    "update_weights_bucket_megabytes": 256,
                },
                "val_kwargs": {"n": 1, "do_sample": False},
            },
        },
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "reward": {
            "num_workers": 2,
            "custom_reward_function": {"path": str(SOURCE / "reward.py"), "name": "compute_score"},
        },
        "trainer": {
            "project_name": "video-highlight",
            "experiment_name": output.name,
            "logger": ["console", "file"],
            "n_gpus_per_node": 2,
            "nnodes": 1,
            "total_epochs": 3,
            "save_freq": 50,
            "test_freq": 50,
            "val_before_train": True,
            "max_actor_ckpt_to_keep": 1,
            "default_local_dir": str(output / "checkpoints"),
            "rollout_data_dir": str(output / "rollouts"),
            "validation_data_dir": str(output / "validation"),
            "log_val_generations": 0,
        },
        "transfer_queue": {
            "enable": True,
            "backend": {"SimpleStorage": {"num_data_storage_units": 2}},
        },
        "ray_kwargs": {
            "ray_init": {
                "num_cpus": 20,
                "object_store_memory": 8589934592,
                "include_dashboard": False,
                "runtime_env": {
                    "py_executable": None,
                    "env_vars": {
                        "PYTHONPATH": str(SOURCE),
                        "VERL_FILE_LOGGER_PATH": os.environ["VERL_FILE_LOGGER_PATH"],
                    },
                },
            }
        },
    }
    if check_steps:
        settings["trainer"].update(total_training_steps=check_steps)
        settings["data"].update(train_max_samples=32, val_max_samples=16)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, settings)
    OmegaConf.resolve(cfg)
    train_args = {
        "epochs": 3,
        "learning_rate": 1e-6,
        "effective_batch": 64,
        "world_size": 2,
        "rank": adapter_config["r"],
        "alpha": adapter_config["lora_alpha"],
        "dropout": 0,
        "responses_per_window": 4,
        "temperature": 0.8,
        "clip_low": 0.0003,
        "clip_high": 0.0004,
        "seed": 42,
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 1,
    }
    metadata = {
        "stage": "gspo",
        "base_model": source["base_model"],
        "dataset": str(dataset),
        "dataset_manifest": source["dataset_manifest"],
        "input": source["input"],
        "initial_adapter": str(initial),
        "sft_adapter_sha256": hashlib.sha256(
            (sft / "adapter/adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "train_samples": min(32, len(train)) if check_steps else len(train),
        "val_samples": min(16, len(val)) if check_steps else len(val),
        "trainable_parameters": source["trainable_parameters"],
        "training": train_args,
        "check_steps": check_steps,
        "reward": "one_to_one_iou_soft_f1",
        "selection": "validation F1 at temporal IoU >= 0.5",
    }
    path = output / "run.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError("已有训练目录的数据、权重或参数不同")
    write_json(path, metadata)
    OmegaConf.save(cfg, output / "config.yaml")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sft", type=Path, default=ROOT / "runs/qwen35-9b-sft-001")
    parser.add_argument(
        "--check-steps", type=int, default=0, help="小批真实训练检查；使用独立输出目录"
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    cfg = prepare(output, args.sft, args.check_steps)
    if args.prepare_only:
        return
    if (output / "completed.json").exists():
        return
    from gspo_trainer import run

    run(cfg)
    best = json.loads((output / "best.json").read_text())
    steps = int((output / "checkpoints/latest_checkpointed_iteration.txt").read_text())
    if not (output / f"validation/{steps}.json").is_file():
        raise ValueError("末步尚未完成验证")
    write_json(output / "completed.json", {"steps": steps, "best": best, "stage": "gspo"})


if __name__ == "__main__":
    main()
