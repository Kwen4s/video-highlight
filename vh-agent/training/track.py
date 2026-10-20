"""Sync a training directory to SwanLab without interrupting GPU training."""

import argparse
import fcntl
import json
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv

METRICS = {
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/learning_rate",
    "epoch": "train/epoch",
    "elapsed_sec": "time/training_seconds",
    "peak_gpu_gib": "memory/training_gpu_peak_gib",
    "rss_gib": "memory/training_process_gib",
    "eval_loss": "validation/loss",
    "eval_runtime": "validation/seconds",
    "eval_samples_per_second": "validation/windows_per_second",
    "effect_precision": "effect/precision_iou50",
    "effect_precision_iou30": "effect/precision_iou30",
    "effect_recall_iou30": "effect/recall_iou30",
    "effect_recall": "effect/recall_iou50",
    "effect_core_recall": "effect/core_recall_coverage80",
    "effect_f1": "effect/f1_iou50",
    "effect_valid_output_rate": "effect/valid_output_rate",
    "effect_empty_window_false_positive_rate": "effect/empty_window_false_positive_rate",
    "effect_predicted_duration_fraction": "effect/predicted_duration_fraction",
    "test_loss": "test/loss",
    "test_runtime": "test/seconds",
    "test_samples_per_second": "test/windows_per_second",
    "train_loss": "summary/train_loss",
    "train_runtime": "summary/train_seconds",
}

GSPO_METRICS = {
    "actor/entropy",
    "actor/loss",
    "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb",
    "actor/perf/cpu_memory_used_gb",
    "timing_s/old_log_prob",
    "timing_s/save_checkpoint",
    "timing_s/update_weights",
    "perf/total_num_tokens",
    "perf/time_per_step",
    "perf/throughput",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/pg_clipfrac_lower",
    "actor/ppo_kl",
    "actor/grad_norm",
    "actor/lr",
    "critic/score/mean",
    "critic/score/max",
    "critic/score/min",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "critic/advantages/max",
    "critic/advantages/min",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "prompt_length/mean",
    "prompt_length/max",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/testing",
    "timing_s/update_actor",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "training/global_step",
    "training/epoch",
}


def read_gspo_metrics(output):
    result = {}
    for path in sorted((output / "metrics").glob("*.jsonl")):
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            values = {
                key: value
                for key, value in row["data"].items()
                if (key in GSPO_METRICS or key.startswith(("validation/", "sampling/", "effect/")))
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            }
            result.setdefault(row["step"], {}).update(values)
    return dict(sorted(result.items()))


def read_metrics(path: Path, train_samples: int) -> dict[int, dict]:
    """Merge resumed updates and same-step evaluation; wait for complete lines."""
    result = {}
    previous = None
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            step = row["step"]
            values = {name: row[key] for key, name in METRICS.items() if key in row}
            if "loss" in row:
                # A repeated update belongs to a new attempt, not a new batch.
                if previous is not None and step == previous["step"] + 1:
                    duration = row["elapsed_sec"] - previous["elapsed_sec"]
                    windows = (row["epoch"] - previous["epoch"]) * train_samples
                    if duration > 0 and windows > 0:
                        values["speed/seconds_per_update"] = duration
                        values["speed/windows_per_second"] = windows / duration
                previous = row
                # Replace measurements from a superseded training attempt.
                result[step] = values
            else:
                result.setdefault(step, {}).update(values)
    return dict(sorted(result.items()))


def sync_once(run, path: Path, train_samples: int, sent: dict) -> int:
    history = read_gspo_metrics(path) if path.is_dir() else read_metrics(path, train_samples)
    for step, values in history.items():
        if values != sent.get(step):
            run.log(values, step=step)
            sent[step] = values
    return max(history, default=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="包含 run.json 和 metrics.jsonl 的训练目录")
    parser.add_argument("--project", default="video-highlight")
    args = parser.parse_args()
    output = args.output.resolve()
    metadata = json.loads((output / "run.json").read_text())
    # Explicit numeric allowlist: no local paths, fingerprints, prompts or media.
    config = {
        key: metadata[key]
        for key in ("train_samples", "val_samples", "trainable_parameters", "max_text_tokens")
        if key in metadata
    }
    config.update(
        {
            key: metadata["training"][key]
            for key in (
                "epochs",
                "learning_rate",
                "effective_batch",
                "world_size",
                "gradient_accumulation_steps",
                "full_attention_checkpoint_stride",
                "rank",
                "alpha",
                "dropout",
                "seed",
                "responses_per_window",
                "temperature",
                "clip_low",
                "clip_high",
            )
            if key in metadata["training"]
        }
    )
    config.update({key: metadata["input"][key] for key in ("fps", "visual_tokens", "max_length")})
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.environ.get("SWANLAB_API_KEY"):
        parser.error("请在 training/.env 中设置 SWANLAB_API_KEY")

    import swanlab

    with (output / ".swanlab.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        swanlab.login(api_key=os.environ["SWANLAB_API_KEY"], save=False)
        run = swanlab.init(
            project=args.project,
            name=output.name,
            id=output.name,
            resume="allow",
            mode="online",
            public=False,
            log_dir=str(output / "swanlab"),
            config=config,
            settings=swanlab.Settings(
                interactive=False,
                terminal={"proxy_type": "none"},
                probe={
                    "hardware": False,
                    "runtime": False,
                    "requirements": False,
                    "git": False,
                    "monitor": False,
                },
            ),
        )
        state_path = output / "swanlab.json"
        state = {"id": run.id, "url": run.url, "project": args.project}
        sent = {}
        metrics_path = output if metadata["stage"] == "gspo" else output / "metrics.jsonl"
        try:
            while True:
                step = sync_once(run, metrics_path, config["train_samples"], sent)
                if step != state.get("last_logged_step"):
                    state["last_logged_step"] = step
                    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
                    print(f"SwanLab logged step {step}: {run.url}", flush=True)
                if (output / "completed.json").exists():
                    # The producer writes completion only after its final metrics.
                    sync_once(run, metrics_path, config["train_samples"], sent)
                    run.finish()
                    break
                if (output / "failed.json").exists():
                    run.finish(state="crashed")
                    break
                time.sleep(5)
        except BaseException:
            run.finish(state="crashed")
            raise


if __name__ == "__main__":
    main()
