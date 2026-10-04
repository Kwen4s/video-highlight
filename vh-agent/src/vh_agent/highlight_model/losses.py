"""Losses aligned with event-centered segment localization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .dataset import EpisodeTargets
from .network import TransitionOutput


@dataclass
class LossOutput:
    total: Tensor
    event: Tensor
    boundary: Tensor
    quality: Tensor
    ranking: Tensor


def highlight_localization_loss(output: TransitionOutput, targets: EpisodeTargets) -> LossOutput:
    device = output.event_logits.device
    event_target = targets.eventness.to(device)
    sample_weight = targets.sample_weight.to(device)
    event_loss = _balanced_focal_target(
        output.event_logits,
        event_target,
        sample_weight,
        positive_mask=event_target >= 0.3,
        negative_mask=targets.hard_negative_mask.to(device)
        | ((event_target <= 0.05) & ~targets.ignore_mask.to(device)),
    )

    boundary_mask = targets.boundary_mask.to(device)
    centers = torch.arange(len(output.offsets), device=device, dtype=output.offsets.dtype)
    predicted_segments = torch.stack(
        [centers + output.offsets[:, 0], centers + output.offsets[:, 1]], dim=1
    )
    if boundary_mask.any():
        target_offsets = targets.offsets.to(device)[boundary_mask]
        predicted_offsets = output.offsets[boundary_mask]
        regression = F.smooth_l1_loss(
            predicted_offsets,
            target_offsets,
            reduction="mean",
            beta=1.0,
        )
        positive_centers = centers[boundary_mask]
        positive_segments = predicted_segments[boundary_mask]
        target_segments = torch.stack(
            [
                positive_centers + target_offsets[:, 0],
                positive_centers + target_offsets[:, 1],
            ],
            dim=1,
        )
        boundary_loss = regression + _temporal_iou_loss(
            positive_segments,
            target_segments,
        )
    else:
        boundary_loss = torch.zeros((), device=device)

    labeled_segments = targets.segments.to(device)
    if len(labeled_segments):
        quality_target = _pairwise_temporal_iou(predicted_segments.detach(), labeled_segments).amax(
            dim=1
        )
    else:
        quality_target = torch.zeros(len(predicted_segments), device=device)
    quality_loss = _balanced_focal_target(
        output.segment_quality_logits,
        quality_target,
        sample_weight,
        positive_mask=(quality_target >= 0.3) & ~targets.ignore_mask.to(device),
        negative_mask=(quality_target < 0.1) & ~targets.ignore_mask.to(device),
    )
    combined_score = F.logsigmoid(output.event_logits) + F.logsigmoid(output.segment_quality_logits)
    decode_score = torch.sigmoid(output.event_logits) * torch.sigmoid(output.segment_quality_logits)
    peaks = _local_max_mask(decode_score.detach())
    ignore = targets.ignore_mask.to(device)
    hard_negative = targets.hard_negative_mask.to(device)
    positive = event_target >= 0.3
    negative = (hard_negative | ((event_target <= 0.05) & ~ignore)) & (quality_target < 0.1)
    ranking_loss = (
        _hard_negative_ranking_loss(combined_score, positive, negative)
        + _local_peak_score_loss(decode_score, peaks & positive, peaks & negative)
        + _far_false_peak_loss(decode_score, peaks, hard_negative, quality_target)
    )

    total = (
        event_loss
        + 0.5 * boundary_loss
        + 0.5 * quality_loss
        + 0.5 * ranking_loss
    )
    return LossOutput(
        total,
        event_loss,
        boundary_loss,
        quality_loss,
        ranking_loss,
    )


def _balanced_focal_target(
    logits: Tensor,
    target: Tensor,
    weight: Tensor,
    positive_mask: Tensor,
    negative_mask: Tensor,
) -> Tensor:
    probability = torch.sigmoid(logits)
    focal = (probability - target).abs().square()
    losses = focal * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    parts: list[Tensor] = []
    for mask in (positive_mask, negative_mask):
        if mask.any():
            parts.append((losses[mask] * weight[mask]).sum() / weight[mask].sum().clamp_min(1))
    return torch.stack(parts).mean() if parts else torch.zeros((), device=logits.device)


def _far_false_peak_loss(
    scores: Tensor,
    peaks: Tensor,
    hard_negative: Tensor,
    quality_iou: Tensor,
    ceiling: float = 0.03,
) -> Tensor:
    far_false = peaks & hard_negative & (quality_iou < 0.1)
    if not far_false.any():
        return torch.zeros((), device=scores.device)
    return F.relu(scores.float()[far_false] - ceiling).mean()


def _local_peak_score_loss(
    scores: Tensor,
    positive: Tensor,
    negative: Tensor,
) -> Tensor:
    parts: list[Tensor] = []
    # Use log-space BCE so mixed-precision autocast cannot rewrite the op.
    score = scores.float()
    if positive.any():
        parts.append((-score[positive].clamp_min(1e-4).log()).mean())
    if negative.any():
        parts.append((-(1.0 - score[negative]).clamp_min(1e-4).log()).mean())
    return torch.stack(parts).mean() if parts else torch.zeros((), device=scores.device)


def _local_max_mask(scores: Tensor) -> Tensor:
    if len(scores) == 1:
        return torch.ones_like(scores, dtype=torch.bool)
    padded = F.pad(scores[None, None, :], (1, 1), value=-1.0)
    local_max = F.max_pool1d(padded, kernel_size=3, stride=1)
    return scores >= local_max.flatten()


def _temporal_iou_loss(predicted: Tensor, target: Tensor) -> Tensor:
    intersection = (
        torch.minimum(predicted[:, 1], target[:, 1]) - torch.maximum(predicted[:, 0], target[:, 0])
    ).clamp_min(0)
    union = (
        torch.maximum(predicted[:, 1], target[:, 1]) - torch.minimum(predicted[:, 0], target[:, 0])
    ).clamp_min(1e-6)
    return (1.0 - intersection / union).mean()


def _pairwise_temporal_iou(predicted: Tensor, target: Tensor) -> Tensor:
    intersection = (
        torch.minimum(predicted[:, None, 1], target[None, :, 1])
        - torch.maximum(predicted[:, None, 0], target[None, :, 0])
    ).clamp_min(0)
    union = (
        torch.maximum(predicted[:, None, 1], target[None, :, 1])
        - torch.minimum(predicted[:, None, 0], target[None, :, 0])
    ).clamp_min(1e-6)
    return intersection / union


def _hard_negative_ranking_loss(
    scores: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 0.2,
    hard_negative_count: int = 64,
) -> Tensor:
    if not positive.any() or not negative.any():
        return torch.zeros((), device=scores.device)
    positive_scores = scores[positive]
    negative_scores = scores[negative].topk(min(hard_negative_count, int(negative.sum()))).values
    return F.softplus(margin + negative_scores[:, None] - positive_scores[None, :]).mean()
