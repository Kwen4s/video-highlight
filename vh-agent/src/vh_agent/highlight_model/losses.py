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
    if boundary_mask.any():
        target_offsets = targets.offsets.to(device)[boundary_mask]
        predicted_offsets = output.offsets[boundary_mask]
        regression = F.smooth_l1_loss(
            predicted_offsets,
            target_offsets,
            reduction="mean",
            beta=1.0,
        )
        centers = torch.arange(
            len(output.offsets),
            device=device,
            dtype=output.offsets.dtype,
        )[boundary_mask]
        predicted_segments = torch.stack(
            [
                centers + predicted_offsets[:, 0],
                centers + predicted_offsets[:, 1],
            ],
            dim=1,
        )
        target_segments = torch.stack(
            [
                centers + target_offsets[:, 0],
                centers + target_offsets[:, 1],
            ],
            dim=1,
        )
        boundary_loss = regression + _temporal_iou_loss(
            predicted_segments,
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
        presence_loss = F.binary_cross_entropy(
            output.anchor_presence[boundary_mask].clamp(1e-6, 1.0 - 1e-6),
            targets.anchor_mask.to(device)[boundary_mask].float(),
        )
    else:
        presence_loss = torch.zeros((), device=device)
    anchor_loss = position_loss + 0.25 * presence_loss
    total = event_loss + 0.5 * boundary_loss + 0.25 * anchor_loss
    return LossOutput(total, event_loss, boundary_loss, anchor_loss)


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
