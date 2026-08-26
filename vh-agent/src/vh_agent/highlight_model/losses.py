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
    anchor: Tensor
    quality: Tensor
    ranking: Tensor


def highlight_localization_loss(output: TransitionOutput, targets: EpisodeTargets) -> LossOutput:
    device = output.event_logits.device
    event_target = targets.eventness.to(device)
    sample_weight = targets.sample_weight.to(device)
    probability = torch.sigmoid(output.event_logits)
    focal = (probability - event_target).abs().square()
    event_loss = (
        sample_weight
        * focal
        * F.binary_cross_entropy_with_logits(
            output.event_logits,
            event_target,
            reduction="none",
        )
    ).sum() / sample_weight.sum().clamp_min(1)

    boundary_mask = targets.boundary_mask.to(device)
    centers = torch.arange(
        len(output.offsets), device=device, dtype=output.offsets.dtype
    )
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

    anchor_mask = targets.anchor_mask.to(device) & boundary_mask[:, None]
    if anchor_mask.any():
        target_positions = targets.anchor_positions.to(device)
        position_loss = F.smooth_l1_loss(
            output.anchor_positions[anchor_mask],
            target_positions[anchor_mask],
            reduction="mean",
            beta=1.0,
        )
    else:
        position_loss = torch.zeros((), device=device)
    if boundary_mask.any():
        presence_probability = output.anchor_presence[boundary_mask].float().clamp(
            1e-6, 1.0 - 1e-6
        )
        presence_loss = F.binary_cross_entropy_with_logits(
            torch.logit(presence_probability),
            targets.anchor_mask.to(device)[boundary_mask].float(),
        )
    else:
        presence_loss = torch.zeros((), device=device)
    anchor_loss = position_loss + 0.25 * presence_loss

    target_segments = targets.segments.to(device)
    if len(target_segments):
        quality_target = _pairwise_temporal_iou(
            predicted_segments.detach(), target_segments
        ).amax(dim=1)
    else:
        quality_target = torch.zeros(len(predicted_segments), device=device)
    quality_loss = _balanced_quality_loss(
        output.segment_quality_logits,
        quality_target,
        sample_weight,
    )
    combined_score = F.logsigmoid(output.event_logits) + F.logsigmoid(
        output.segment_quality_logits
    )
    positive = event_target >= 0.3
    negative = (event_target <= 0.05) & (quality_target < 0.1)
    ranking_loss = _hard_negative_ranking_loss(combined_score, positive, negative)

    total = (
        event_loss
        + 0.5 * boundary_loss
        + 0.25 * anchor_loss
        + 0.5 * quality_loss
        + 0.25 * ranking_loss
    )
    return LossOutput(
        total,
        event_loss,
        boundary_loss,
        anchor_loss,
        quality_loss,
        ranking_loss,
    )


def _temporal_iou_loss(predicted: Tensor, target: Tensor) -> Tensor:
    intersection = (
        torch.minimum(predicted[:, 1], target[:, 1])
        - torch.maximum(predicted[:, 0], target[:, 0])
    ).clamp_min(0)
    union = (
        torch.maximum(predicted[:, 1], target[:, 1])
        - torch.minimum(predicted[:, 0], target[:, 0])
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


def _balanced_quality_loss(logits: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    focal = (probability - target).abs().square()
    losses = focal * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = target >= 0.3
    negative = target < 0.1
    parts: list[Tensor] = []
    for mask in (positive, negative):
        if mask.any():
            parts.append(
                (losses[mask] * weight[mask]).sum()
                / weight[mask].sum().clamp_min(1)
            )
    return torch.stack(parts).mean() if parts else torch.zeros((), device=logits.device)


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
    negative_scores = scores[negative].topk(
        min(hard_negative_count, int(negative.sum()))
    ).values
    return F.softplus(
        margin + negative_scores[:, None] - positive_scores[None, :]
    ).mean()
