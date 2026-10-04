"""Segment decoding and temporal IoU metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .dataset import VideoExample
from .network import TransitionOutput


@dataclass(frozen=True)
class PredictedSegment:
    start_sec: float
    end_sec: float
    score: float


def decode_segments(
    output: TransitionOutput,
    duration_sec: float,
    threshold: float,
    nms_iou: float,
) -> list[PredictedSegment]:
    scores = (
        torch.sigmoid(output.event_logits.detach())
        * torch.sigmoid(output.segment_quality_logits.detach())
    ).cpu()
    if len(scores) == 1:
        peaks = torch.ones_like(scores, dtype=torch.bool)
    else:
        padded = torch.nn.functional.pad(scores[None, None, :], (1, 1), value=-1.0)
        local_max = torch.nn.functional.max_pool1d(padded, kernel_size=3, stride=1)
        peaks = scores >= local_max.flatten()
    candidates: list[PredictedSegment] = []
    for index in (peaks & (scores >= threshold)).nonzero(as_tuple=False).flatten().tolist():
        offset = output.offsets[index].detach().cpu()
        start = max(0.0, min(duration_sec, index + float(offset[0])))
        end = max(0.0, min(duration_sec, index + float(offset[1])))
        if end > start:
            candidates.append(PredictedSegment(start, end, float(scores[index])))
    selected: list[PredictedSegment] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(_temporal_iou(candidate, kept) < nms_iou for kept in selected):
            selected.append(candidate)
    return selected


def segment_metrics(
    predictions: dict[str, list[PredictedSegment]],
    videos: list[VideoExample],
) -> dict[str, float]:
    evaluated = {}
    ignored = 0
    for video in videos:
        known = [
            *((h.start_sec, h.end_sec) for h in video.highlights),
            *video.hard_negative_intervals,
        ]
        raw = predictions.get(video.video_id, [])
        evaluated[video.video_id] = [
            p
            for p in raw
            if any(min(p.end_sec, end) > max(p.start_sec, start) for start, end in known)
        ]
        ignored += len(raw) - len(evaluated[video.video_id])
    candidates = sum(len(predictions.get(video.video_id, [])) for video in videos)
    minutes = sum(video.duration_sec for video in videos) / 60
    metrics: dict[str, float] = {
        "ignored_unknown_predictions": float(ignored),
        "candidates_per_minute": candidates / minutes if minutes else 0.0,
    }
    for threshold in (0.3, 0.5, 0.7):
        tp = fp = fn = 0
        for video in videos:
            predicted = evaluated[video.video_id]
            target = [
                PredictedSegment(item.start_sec, item.end_sec, 1.0) for item in video.highlights
            ]
            local_tp, local_fp, local_fn = _match(predicted, target, threshold)
            tp += local_tp
            fp += local_fp
            fn += local_fn
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        metrics[f"precision_iou_{threshold:.1f}"] = precision
        metrics[f"recall_iou_{threshold:.1f}"] = recall
        metrics[f"f1_iou_{threshold:.1f}"] = 2 * precision * recall / max(1e-8, precision + recall)
    metrics["mean_f1"] = 0.5 * (metrics["f1_iou_0.5"] + metrics["f1_iou_0.7"])
    return metrics


def predictions_as_json(
    predictions: dict[str, list[PredictedSegment]],
) -> list[dict[str, object]]:
    return [
        {"video_id": video_id, "segments": [asdict(item) for item in segments]}
        for video_id, segments in sorted(predictions.items())
    ]


def _match(
    predicted: list[PredictedSegment],
    target: list[PredictedSegment],
    threshold: float,
) -> tuple[int, int, int]:
    remaining = set(range(len(target)))
    true_positive = 0
    for candidate in sorted(predicted, key=lambda item: item.score, reverse=True):
        if not remaining:
            break
        best = max(remaining, key=lambda index: _temporal_iou(candidate, target[index]))
        if _temporal_iou(candidate, target[best]) >= threshold:
            remaining.remove(best)
            true_positive += 1
    return true_positive, len(predicted) - true_positive, len(target) - true_positive


def _temporal_iou(left: PredictedSegment, right: PredictedSegment) -> float:
    intersection = max(0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec))
    union = max(left.end_sec, right.end_sec) - min(left.start_sec, right.start_sec)
    return intersection / max(1e-8, union)
