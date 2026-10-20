"""Task-specific validation and adapter export around the native verl GSPO trainer."""

import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import ray
import torch
from gspo import write_json
from reward import compute_score
from safetensors.torch import save_file
from score import rates
from torch.distributed.tensor import DTensor, Replicate, Shard
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, PPOTrainerSync


def detection_metrics(records):
    results = [compute_score("short_drama", r["output"], r["gts"]) for r in records]
    count = len(results)
    negative = sum(not r["positive"] for r in results)
    positive = count - negative
    values = {
        "validation/soft_f1": mean(r["score"] for r in results),
        "validation/valid_output_rate": mean(r["format_valid"] for r in results),
        "validation/empty_positive_rate": sum(r["empty_positive"] for r in results) / positive
        if positive
        else 0,
        "validation/empty_window_false_positive_rate": sum(
            r["empty_negative_false_alarm"] for r in results
        )
        / negative
        if negative
        else 0,
        "validation/invalid_segments": sum(r["invalid_segments"] for r in results),
    }
    for threshold in (30, 50):
        stats = rates(
            sum(r[f"hits{threshold}"] for r in results),
            sum(r["predicted_count"] for r in results),
            sum(r["target_count"] for r in results),
        )
        values.update({f"validation/{k}_iou{threshold}": v for k, v in stats.items()})
    return values


def sampling_metrics(records):
    groups = defaultdict(list)
    positive, negative = [], []
    for record in records:
        truth = json.loads(record["gts"])
        groups[truth["sample_id"]].append(record["score"])
        result = compute_score("short_drama", record["output"], truth)
        (positive if truth["segments"] else negative).append(result)
    stds = [pstdev(scores) for scores in groups.values()]
    metrics = {
        "sampling/reward_std": mean(stds),
        "sampling/groups_with_signal": mean(s > 1e-8 for s in stds),
        "sampling/all_zero_groups": mean(all(s == 0 for s in scores) for scores in groups.values()),
    }
    if positive:
        metrics.update(
            {
                "sampling/positive_reward": mean(r["score"] for r in positive),
                "sampling/empty_positive_rate": mean(r["empty_positive"] for r in positive),
            }
        )
    if negative:
        metrics.update(
            {
                "sampling/negative_reward": mean(r["score"] for r in negative),
                "sampling/empty_window_false_positive_rate": mean(
                    r["empty_negative_false_alarm"] for r in negative
                ),
            }
        )
    return metrics


def export_adapter(checkpoint, target):
    # verl's general merger unconditionally rounds weights to BF16. Preserve the
    # FP32 LoRA master weights: RL updates can be smaller than a BF16 rounding step.
    world = json.loads((checkpoint / "fsdp_config.json").read_text())["world_size"]
    shards = [
        torch.load(
            checkpoint / f"model_world_size_{world}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in range(world)
    ]
    if any(set(part) != set(shards[0]) for part in shards):
        raise ValueError("LoRA 分片缺失权重")
    weights = {}
    for name, tensor in shards[0].items():
        if "lora_" not in name:
            raise ValueError("需要仅包含 LoRA 的完整 FSDP2 检查点")
        parts = [part[name] for part in shards]
        if isinstance(tensor, DTensor):
            if len(tensor.placements) != 1 or any(p.placements != tensor.placements for p in parts):
                raise ValueError("导出只支持当前单轴 FSDP2 分片")
            placement = tensor.placements[0]
            parts = [part.to_local() for part in parts]
        else:
            placement = Replicate()
        if isinstance(placement, Shard):
            value = torch.cat(parts, dim=placement.dim)
        elif isinstance(placement, Replicate):
            if not all(torch.equal(parts[0], part) for part in parts[1:]):
                raise ValueError(f"复制权重不一致：{name}")
            value = parts[0]
        else:
            raise TypeError(f"未知分片方式：{placement}")
        if value.shape != tensor.shape or value.dtype != tensor.dtype:
            raise ValueError(f"权重形状或精度改变：{name}")
        weights[name.replace(".default.weight", ".weight")] = value.contiguous()
    if not weights:
        raise ValueError("检查点没有 LoRA 权重")
    target.mkdir(parents=True)
    save_file(weights, target / "adapter_model.safetensors")


class HighlightTrainer(PPOTrainerSync):
    def _save_checkpoint(self):
        super()._save_checkpoint()
        root = Path(self.config.trainer.default_local_dir)
        current = root / f"global_step_{self.global_steps}"
        # verl removes old actor shards; also remove their obsolete dataloader state.
        for path in root.glob("global_step_*"):
            if path != current and path.is_dir():
                shutil.rmtree(path)

    def _validate(self):
        metrics = super()._validate()
        for future in self._dump_futures:
            future.result()
        path = Path(self.config.trainer.validation_data_dir) / f"{self.global_steps}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        expected = {row["sample_id"] for row in self.val_dataset.rows}
        ids = [json.loads(r["gts"])["sample_id"] for r in records]
        if set(ids) != expected or len(ids) != len(expected):
            raise ValueError("验证预测不完整或重复，不能选择权重")
        effect = detection_metrics(records)
        metrics.update(effect)
        output = Path(self.config.trainer.default_local_dir).parent
        write_json(output / f"validation/{self.global_steps}.json", effect)
        selection = (
            effect["validation/f1_iou50"],
            effect["validation/recall_iou30"],
            -effect["validation/empty_window_false_positive_rate"],
        )
        best_path = output / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        if best is None or selection > tuple(best["selection"]):
            if self.global_steps == 0:
                candidate = output / "initial-adapter"
            else:
                temporary = output / ".export"
                shutil.rmtree(temporary, ignore_errors=True)
                candidate = temporary / "adapter"
                export_adapter(
                    output / f"checkpoints/global_step_{self.global_steps}/actor", candidate
                )
                # Exact target names and the base location are part of the trained adapter.
                shutil.copy2(
                    output / "initial-adapter/adapter_config.json",
                    candidate / "adapter_config.json",
                )
                for path in (output / "initial-adapter").iterdir():
                    if (
                        path.name not in ("adapter_config.json", "adapter_model.safetensors")
                        and path.is_file()
                    ):
                        shutil.copy2(path, candidate / path.name)
                from safetensors import safe_open

                with (
                    safe_open(
                        str(candidate / "adapter_model.safetensors"), framework="pt"
                    ) as current,
                    safe_open(
                        str(output / "initial-adapter/adapter_model.safetensors"), framework="pt"
                    ) as initial,
                ):
                    if set(current.keys()) != set(initial.keys()):
                        raise ValueError("GSPO 导出的 LoRA 层与 SFT 不同")
                    for key in set(current.keys()):
                        if current.get_slice(key).get_shape() != initial.get_slice(key).get_shape():
                            raise ValueError(f"GSPO 权重形状改变：{key}")
            target = output / ".adapter"
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(candidate, target)
            shutil.rmtree(output / "adapter", ignore_errors=True)
            target.replace(output / "adapter")
            write_json(
                best_path, {"step": self.global_steps, "selection": selection, "metrics": effect}
            )
            shutil.rmtree(output / ".export", ignore_errors=True)
        return metrics

    def _log_rollout_data(self, batch, timing_raw, rollout_data_dir):
        super()._log_rollout_data(batch, timing_raw, rollout_data_dir)
        for future in self._dump_futures:
            future.result()
        records = [
            json.loads(line)
            for line in (Path(rollout_data_dir) / f"{self.global_steps}.jsonl")
            .read_text()
            .splitlines()
        ]
        self.logger.log(sampling_metrics(records), step=self.global_steps)


@ray.remote
class TaskRunner:
    def run(self, config):
        import transfer_queue as tq

        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = HighlightTrainer(config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                if trainer is not None and hasattr(trainer, "logger"):
                    trainer.logger.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


def run(config):
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.config import validate_config

    validate_config(config, use_reference_policy=False, use_critic=False)
    try:
        run_ppo(config, task_runner_class=TaskRunner)
    finally:
        ray.shutdown()
