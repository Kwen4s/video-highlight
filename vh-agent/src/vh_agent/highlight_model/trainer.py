"""Feature preparation, temporal optimization, and fixed-split evaluation."""

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

from .config import HighlightModelConfig
from .dataset import EpisodeBatch, EpisodeDataset, load_silver_videos, split_by_drama
from .decoding import (
    choose_decoder,
    decode_segments,
    predictions_as_json,
    segment_metrics,
)
from .encoders import FrozenMomentFeatureExtractor
from .losses import highlight_localization_loss
from .network import NarrativeTransitionLocalizer, TransitionOutput

CHECKPOINT_SCHEMA = 2


def fit_highlight_model(config: HighlightModelConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    videos = load_silver_videos(config.annotations)
    splits = split_by_drama(videos, config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "config.json", asdict(config))
    _write_json(
        config.output_dir / "drama_split.json",
        {
            name: sorted({video.drama_id for video in subset})
            for name, subset in splits.items()
        },
    )

    feature_report: dict[str, int] | None = None
    if config.stage in {"features", "all"}:
        extractor = FrozenMomentFeatureExtractor(config)
        try:
            feature_report = extractor.prepare(videos)
        finally:
            extractor.close()
        _write_json(config.output_dir / "feature_report.json", feature_report)
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
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.05,
    )
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(model, datasets["train"], optimizer, config, epoch)
        validation_outputs, validation_loss = _evaluate(model, datasets["val"], config)
        threshold, top_k, validation_metrics = choose_decoder(
            validation_outputs,
            splits["val"],
            config.nms_iou,
            config.max_highlights,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "validation_loss": validation_loss,
            "validation_threshold": threshold,
            "validation_top_k": top_k,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        _write_json(config.output_dir / "history.json", history)
        _save_checkpoint(
            model,
            optimizer,
            config.output_dir / "last.pt",
            epoch,
            threshold,
            top_k,
            row,
        )
        if validation_metrics["mean_f1"] > best_metric:
            best_metric = validation_metrics["mean_f1"]
            _save_checkpoint(
                model,
                optimizer,
                config.output_dir / "best.pt",
                epoch,
                threshold,
                top_k,
                row,
            )
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_loss:.4f} val_mean_f1={validation_metrics['mean_f1']:.4f}"
        )

    best = torch.load(
        config.output_dir / "best.pt",
        map_location=config.device,
        weights_only=False,
    )
    if best.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("best checkpoint has an incompatible schema")
    model.load_state_dict(best["model"])
    best_threshold = float(best["threshold"])
    best_top_k = int(best["max_highlights"])
    test_outputs, test_loss = _evaluate(model, datasets["test"], config)
    test_predictions = {
        video.video_id: decode_segments(
            test_outputs[video.video_id],
            video.duration_sec,
            best_threshold,
            config.nms_iou,
            best_top_k,
        )
        for video in splits["test"]
    }
    test_metrics = segment_metrics(test_predictions, splits["test"])
    report = {
        "schema": CHECKPOINT_SCHEMA,
        "method": "NarrativeTransitionLocalizer",
        "videos": {name: len(subset) for name, subset in splits.items()},
        "best_epoch": int(best["epoch"]),
        "threshold": best_threshold,
        "max_highlights": best_top_k,
        "validation_mean_f1": best_metric,
        "test_loss": test_loss,
        "test": test_metrics,
    }
    _write_json(config.output_dir / "metrics.json", report)
    _write_json(
        config.output_dir / "test_predictions.json",
        predictions_as_json(test_predictions),
    )
    return report


def _train_epoch(
    model: NarrativeTransitionLocalizer,
    dataset: EpisodeDataset,
    optimizer: AdamW,
    config: HighlightModelConfig,
    epoch: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(config.seed + epoch)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda rows: rows[0],
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "event": 0.0, "boundary": 0.0, "anchor": 0.0}
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
        totals["anchor"] += float(losses.anchor.detach())
        if step % 25 == 0 or step == len(loader):
            print(f"epoch={epoch} episodes={step}/{len(loader)}")
    return {key: value / max(1, len(loader)) for key, value in totals.items()}


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


def _save_checkpoint(
    model: NarrativeTransitionLocalizer,
    optimizer: AdamW,
    path: Path,
    epoch: int,
    threshold: float,
    max_highlights: int,
    metrics: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": epoch,
            "threshold": threshold,
            "max_highlights": max_highlights,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def _autocast(device: str) -> Any:
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, default=str, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
