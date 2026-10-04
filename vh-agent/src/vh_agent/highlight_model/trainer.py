"""Train with drama-separated validation and evaluate the selected checkpoint once on test."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..storage import write_json
from .config import CHECKPOINT_SCHEMA, HighlightModelConfig
from .dataset import (
    EpisodeBatch,
    EpisodeDataset,
    VideoExample,
    load_videos,
    split_dataset,
)
from .decoding import (
    decode_segments,
    predictions_as_json,
    segment_metrics,
)
from .encoders import FrozenMomentFeatureExtractor
from .losses import highlight_localization_loss
from .network import NarrativeTransitionLocalizer, TransitionOutput


def fit_highlight_model(config: HighlightModelConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    videos = load_videos(config.annotations)
    splits = split_dataset(videos)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "config.json", asdict(config))
    write_json(
        config.output_dir / "video_split.json",
        {name: [video.video_id for video in subset] for name, subset in splits.items()},
    )

    feature_report: dict[str, int] | None = None
    if config.stage in {"features", "all"}:
        extractor = FrozenMomentFeatureExtractor(config)
        try:
            feature_report = extractor.prepare(videos)
        finally:
            extractor.close()
        write_json(config.output_dir / "feature_report.json", feature_report)
    if config.stage == "features":
        return {
            "stage": "features",
            "videos": len(videos),
            "features": feature_report,
        }

    datasets = {
        name: EpisodeDataset(subset, config.feature_cache_dir, config)
        for name, subset in splits.items()
    }
    sample = datasets["train"][0]
    model = NarrativeTransitionLocalizer(
        vision_dim=sample.vision.shape[1],
        audio_dim=sample.audio.shape[1],
        audio_prior_dim=sample.audio_prior.shape[1],
        model_dim=config.model_dim,
        heads=config.attention_heads,
        temporal_layers_per_level=config.temporal_layers_per_level,
        dropout=config.dropout,
        max_before_sec=config.max_before_sec,
        max_after_sec=config.max_after_sec,
    ).to(config.device)
    init_epoch = 0
    if config.init_checkpoint is not None:
        init_epoch = _load_checkpoint(model, config.init_checkpoint, config.device)
        print(f"loaded_checkpoint={config.init_checkpoint} epoch={init_epoch}")
    if config.finetune_heads:
        _freeze_backbone(model)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.05,
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    if config.init_checkpoint is not None:
        val_loss, val_predictions, val_metrics = _evaluate_epoch(
            model, datasets["val"], splits["val"], config
        )
        print(f"init val_f1_iou_0.3={val_metrics['f1_iou_0.3']:.4f}")
        _save_checkpoint(model, config.output_dir / "best.pt", init_epoch)
        best = {
            "epoch": init_epoch,
            "val_loss": val_loss,
            "val": val_metrics,
            "predictions": val_predictions,
        }
    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(model, datasets["train"], optimizer, config, epoch)
        val_loss, val_predictions, val_metrics = _evaluate_epoch(
            model, datasets["val"], splits["val"], config
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_json(config.output_dir / "history.json", history)
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_f1_iou_0.3={val_metrics['f1_iou_0.3']:.4f}"
        )
        if best is None or val_metrics["f1_iou_0.3"] > best["val"]["f1_iou_0.3"]:
            _save_checkpoint(model, config.output_dir / "best.pt", init_epoch + epoch)
            best = {
                "epoch": init_epoch + epoch,
                "val_loss": val_loss,
                "val": val_metrics,
                "predictions": val_predictions,
            }

    _save_checkpoint(model, config.output_dir / "last.pt", init_epoch + config.epochs)
    best_path = config.output_dir / "best.pt"
    checkpoint_path = config.output_dir / "checkpoint.pt"
    if best_path.is_file():
        checkpoint_path.write_bytes(best_path.read_bytes())
    report = {
        "schema": CHECKPOINT_SCHEMA,
        "method": "NarrativeTransitionHardNegChangeLocalizer",
        "split": "persistent_drama_train_val_test",
        "reference_labels": "model_reviewed",
        "videos": {name: len(subset) for name, subset in splits.items()},
        "seed": config.seed,
        "init_checkpoint": str(config.init_checkpoint) if config.init_checkpoint else None,
        "finetune_heads": config.finetune_heads,
        "checkpoint_epoch": best["epoch"] if best else config.epochs,
        "selection": "best_val_f1_iou_0.3",
        "threshold": config.score_threshold,
        "evaluation_candidate_cap": None,
        "val_loss": best["val_loss"] if best else None,
        "val": best["val"] if best else {},
    }
    write_json(
        config.output_dir / "val_predictions.json",
        predictions_as_json(best["predictions"] if best else {}),
    )
    _load_checkpoint(model, best_path, config.device)
    train_outputs, _ = _evaluate(model, datasets["train"], config)
    feedback = predictions_as_json(_decode_outputs(train_outputs, splits["train"], config))
    feedback_path = config.output_dir / "train_predictions.jsonl"
    feedback_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in feedback), encoding="utf-8"
    )
    report["training_feedback"] = str(feedback_path)
    if splits["test"]:
        test_loss, test_predictions, test_metrics = _evaluate_epoch(
            model, datasets["test"], splits["test"], config
        )
        report["test_loss"], report["test"] = test_loss, test_metrics
        write_json(
            config.output_dir / "test_predictions.json", predictions_as_json(test_predictions)
        )
    write_json(config.output_dir / "metrics.json", report)
    return report


def _train_epoch(
    model: NarrativeTransitionLocalizer,
    dataset: EpisodeDataset,
    optimizer: AdamW,
    config: HighlightModelConfig,
    epoch: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(
        config.seed + epoch + (10_000 if config.init_checkpoint is not None else 0)
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda rows: rows[0],
    )
    model.train()
    if config.finetune_heads:
        _freeze_backbone(model)
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "event": 0.0,
        "boundary": 0.0,
        "quality": 0.0,
        "ranking": 0.0,
    }
    for step, batch in enumerate(loader, start=1):
        with _autocast(config.device):
            output = _forward(model, batch, config.device)
            losses = highlight_localization_loss(output, batch.targets)
        (losses.total / config.gradient_accumulation).backward()
        if step % config.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        totals["loss"] += float(losses.total.detach())
        totals["event"] += float(losses.event.detach())
        totals["boundary"] += float(losses.boundary.detach())
        totals["quality"] += float(losses.quality.detach())
        totals["ranking"] += float(losses.ranking.detach())
        if step % 25 == 0 or step == len(loader):
            print(f"epoch={epoch} episodes={step}/{len(loader)}")
    return {key: value / max(1, len(loader)) for key, value in totals.items()}


def _evaluate_epoch(
    model: NarrativeTransitionLocalizer,
    dataset: EpisodeDataset,
    videos: list[VideoExample],
    config: HighlightModelConfig,
) -> tuple[float, dict[str, list], dict[str, float]]:
    outputs, loss = _evaluate(model, dataset, config)
    predictions = _decode_outputs(outputs, videos, config)
    return loss, predictions, segment_metrics(predictions, videos)


def _decode_outputs(outputs, videos, config):
    return {
        video.video_id: decode_segments(
            outputs[video.video_id],
            video.duration_sec,
            config.score_threshold,
            config.nms_iou,
        )
        for video in videos
    }


@torch.inference_mode()
def _evaluate(
    model: NarrativeTransitionLocalizer,
    dataset: EpisodeDataset,
    config: HighlightModelConfig,
) -> tuple[dict[str, TransitionOutput], float]:
    model.eval()
    outputs: dict[str, TransitionOutput] = {}
    total_loss = 0.0
    for batch in DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: rows[0],
    ):
        with _autocast(config.device):
            output = _forward(model, batch, config.device)
            losses = highlight_localization_loss(output, batch.targets)
        total_loss += float(losses.total)
        outputs[batch.video.video_id] = TransitionOutput(
            event_logits=output.event_logits.float().cpu(),
            segment_quality_logits=output.segment_quality_logits.float().cpu(),
            offsets=output.offsets.float().cpu(),
            anchor_positions=output.anchor_positions.float().cpu(),
            anchor_presence=output.anchor_presence.float().cpu(),
            transition=torch.empty(0),
        )
    return outputs, total_loss / max(1, len(dataset))


def _forward(
    model: NarrativeTransitionLocalizer,
    batch: EpisodeBatch,
    device: str,
) -> TransitionOutput:
    return model(
        batch.vision.to(device),
        batch.audio.to(device),
        batch.audio_prior.to(device),
        batch.availability.to(device),
        batch.scene_bounds.to(device),
    )


def _freeze_backbone(model: NarrativeTransitionLocalizer) -> None:
    trainable = {"event_head", "segment_quality_head", "offset_head"}
    for name, module in model.named_children():
        if name in trainable:
            continue
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False


def _load_checkpoint(
    model: NarrativeTransitionLocalizer,
    path: Path,
    device: str,
) -> int:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint requires schema {CHECKPOINT_SCHEMA}")
    model.load_state_dict(payload["model"], strict=True)
    return int(payload["epoch"])


def _save_checkpoint(
    model: NarrativeTransitionLocalizer,
    path: Path,
    epoch: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": epoch,
            "model": model.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def _autocast(device: str) -> Any:
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
