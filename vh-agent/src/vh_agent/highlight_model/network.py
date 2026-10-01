"""Hierarchical scene-memory narrative transition localizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TransitionOutput:
    event_logits: Tensor
    segment_quality_logits: Tensor
    offsets: Tensor
    anchor_positions: Tensor
    anchor_presence: Tensor
    transition: Tensor


def _window_memory(sequence: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    length = len(sequence)
    positions = torch.arange(length, device=sequence.device)[:, None] + offsets[None, :]
    valid = (positions >= 0) & (positions < length)
    memory = sequence[positions.clamp(0, length - 1)]
    return memory, valid, positions.float()


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, radius: int, dropout: float) -> None:
        super().__init__()
        self.radius = radius
        self.heads = heads
        self.head_dim = dim // heads
        if dim % heads:
            raise ValueError("model dimension must be divisible by attention heads")
        self.norm1 = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        normalized = self.norm1(sequence)
        offsets = torch.arange(
            -self.radius,
            1,
            device=sequence.device,
        )
        memory, valid, _ = _window_memory(normalized, offsets)
        query = self.query(normalized).view(len(sequence), self.heads, self.head_dim)
        key = self.key(memory).view(len(sequence), len(offsets), self.heads, self.head_dim)
        value = self.value(memory).view(len(sequence), len(offsets), self.heads, self.head_dim)
        scores = torch.einsum("thd,tlhd->thl", query, key) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("thl,tlhd->thd", weights, value).reshape(len(sequence), -1)
        sequence = sequence + self.dropout(self.output(attended))
        return sequence + self.dropout(self.feed_forward(self.norm2(sequence)))


class TemporalLevel(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        radius: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            WindowAttentionBlock(dim, heads, radius, dropout) for _ in range(layers)
        )

    def forward(self, sequence: Tensor) -> Tensor:
        for block in self.blocks:
            sequence = block(sequence)
        return sequence


class HierarchicalTemporalEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.level0 = TemporalLevel(dim, heads, radius=8, layers=layers, dropout=dropout)
        self.level1 = TemporalLevel(dim, heads, radius=4, layers=layers, dropout=dropout)
        self.level2 = TemporalLevel(dim, heads, radius=4, layers=layers, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim),
            nn.GELU(),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        level0 = self.level0(sequence)
        level1 = self.level1(_downsample(level0, 4))
        level2 = self.level2(_downsample(level1, 4))
        up1 = _upsample(level1, len(level0), factor=4)
        up2 = _upsample(level2, len(level0), factor=16)
        return self.fusion(torch.cat([level0, up1, up2], dim=-1))


class SceneMemoryPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )
        self.position = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, sequence: Tensor, bounds: Tensor) -> tuple[Tensor, Tensor]:
        length = len(sequence)
        time = torch.arange(length, device=sequence.device)
        membership = (time[None, :] >= bounds[:, :1]) & (time[None, :] < bounds[:, 1:2])
        scores = self.score(sequence).squeeze(-1)[None, :].expand(len(bounds), -1)
        weights = torch.softmax(scores.masked_fill(~membership, -torch.inf), dim=1)
        scenes = weights @ sequence
        normalized = torch.stack(
            [
                bounds[:, 0] / max(1, length),
                bounds[:, 1] / max(1, length),
                (bounds[:, 1] - bounds[:, 0]) / max(1, length),
            ],
            dim=1,
        ).float()
        return scenes + self.position(normalized), bounds.float().mean(dim=1)


class MaskedMemoryAttention(nn.Module):
    """Attend query moments to variable valid memories plus a learned empty state."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        if dim % heads:
            raise ValueError("model dimension must be divisible by attention heads")
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.empty = nn.Parameter(torch.empty(dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.empty, std=0.02)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        valid: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        length = len(queries)
        empty = self.empty.view(1, 1, -1).expand(length, 1, -1)
        memory = torch.cat([memory, empty], dim=1)
        valid = torch.cat(
            [valid, torch.ones((length, 1), dtype=torch.bool, device=valid.device)],
            dim=1,
        )
        positions = torch.cat(
            [positions, torch.arange(length, device=queries.device)[:, None].float()],
            dim=1,
        )
        query = self.query(queries).view(length, self.heads, self.head_dim)
        key = self.key(memory).view(length, len(memory[0]), self.heads, self.head_dim)
        value = self.value(memory).view(length, len(memory[0]), self.heads, self.head_dim)
        scores = torch.einsum("thd,tlhd->thl", query, key) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("thl,tlhd->thd", weights, value).reshape(length, -1)
        context = self.norm(queries + self.dropout(self.output(attended)))
        mean_weights = weights.mean(dim=1)
        nonempty = mean_weights[:, :-1].sum(dim=1)
        expected = (mean_weights[:, :-1] * positions[:, :-1]).sum(dim=1) / nonempty.clamp_min(1e-6)
        return context, expected, nonempty


class NarrativeTransitionDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        max_before_sec: int,
        max_after_sec: int,
    ) -> None:
        super().__init__()
        self.max_before_sec = max_before_sec
        self.max_after_sec = max_after_sec
        self.before = MaskedMemoryAttention(dim, heads, dropout)
        self.event = MaskedMemoryAttention(dim, heads, dropout)
        self.after = MaskedMemoryAttention(dim, heads, dropout)
        self.salience = MaskedMemoryAttention(dim, heads, dropout)
        self.transition = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
        )

    def forward(
        self,
        sequence: Tensor,
        scenes: Tensor,
        scene_bounds: Tensor,
        scene_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        length = len(sequence)
        before_offsets = torch.arange(
            -self.max_before_sec,
            0,
            device=sequence.device,
        )
        local_before, before_valid, before_positions = _window_memory(sequence, before_offsets)
        scene_memory = scenes[None, :, :].expand(length, -1, -1)
        scene_valid = (
            scene_bounds[:, 1][None, :] <= torch.arange(length, device=sequence.device)[:, None]
        )
        expanded_scene_positions = scene_positions[None, :].expand(length, -1)
        before_memory = torch.cat([local_before, scene_memory], dim=1)
        before_mask = torch.cat([before_valid, scene_valid], dim=1)
        before_times = torch.cat(
            [before_positions, expanded_scene_positions],
            dim=1,
        )
        before_state, before_time, before_presence = self.before(
            sequence,
            before_memory,
            before_mask,
            before_times,
        )

        event_offsets = torch.arange(-2, 3, device=sequence.device)
        event_memory, event_valid, event_positions = _window_memory(sequence, event_offsets)
        event_state, event_time, event_presence = self.event(
            sequence,
            event_memory,
            event_valid,
            event_positions,
        )

        after_offsets = torch.arange(
            1,
            self.max_after_sec + 1,
            device=sequence.device,
        )
        after_memory, after_valid, after_positions = _window_memory(sequence, after_offsets)
        after_state, after_time, after_presence = self.after(
            sequence,
            after_memory,
            after_valid,
            after_positions,
        )

        all_scene_valid = torch.ones(
            (length, len(scenes)),
            dtype=torch.bool,
            device=sequence.device,
        )
        salience, _, _ = self.salience(
            sequence,
            scene_memory,
            all_scene_valid,
            expanded_scene_positions,
        )
        transition = self.transition(
            torch.cat(
                [
                    event_state - before_state,
                    after_state - event_state,
                    salience,
                ],
                dim=-1,
            )
        )
        anchor_positions = torch.stack([before_time, event_time, after_time], dim=1)
        anchor_presence = torch.stack(
            [before_presence, event_presence, after_presence],
            dim=1,
        )
        return transition, anchor_positions, anchor_presence


class NarrativeTransitionLocalizer(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        audio_dim: int,
        audio_prior_dim: int,
        model_dim: int,
        heads: int,
        temporal_layers_per_level: int,
        dropout: float,
        max_before_sec: int,
        max_after_sec: int,
        max_center_offset_sec: float = 8.0,
        min_segment_duration_sec: float = 6.0,
        max_segment_duration_sec: float = 24.0,
    ) -> None:
        super().__init__()
        self.max_center_offset_sec = max_center_offset_sec
        self.min_segment_duration_sec = min_segment_duration_sec
        self.max_segment_duration_sec = max_segment_duration_sec
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, model_dim),
            nn.GELU(),
        )
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(audio_dim + audio_prior_dim),
            nn.Linear(audio_dim + audio_prior_dim, model_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * model_dim + 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.change_proj = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.temporal = HierarchicalTemporalEncoder(
            model_dim,
            heads,
            temporal_layers_per_level,
            dropout,
        )
        self.scene_pool = SceneMemoryPool(model_dim)
        self.decoder = NarrativeTransitionDecoder(
            model_dim,
            heads,
            dropout,
            max_before_sec,
            max_after_sec,
        )
        self.event_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.offset_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 2))
        self.segment_quality_head = nn.Sequential(
            nn.LayerNorm(model_dim + 8),
            nn.Linear(model_dim + 8, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        nn.init.constant_(self.event_head[-1].bias, -2.0)
        nn.init.constant_(self.segment_quality_head[-1].bias, -0.4)

    def forward(
        self,
        vision: Tensor,
        audio: Tensor,
        audio_prior: Tensor,
        availability: Tensor,
        scene_bounds: Tensor,
    ) -> TransitionOutput:
        visual = self.vision_projection(vision)
        acoustic = self.audio_projection(torch.cat([audio, audio_prior], dim=1))
        visual_weight = torch.sigmoid(self.gate(torch.cat([visual, acoustic, availability], dim=1)))
        visual_weight = torch.where(
            availability[:, 1:2] == 0,
            torch.ones_like(visual_weight),
            visual_weight,
        )
        visual_weight = torch.where(
            availability[:, 0:1] == 0,
            torch.zeros_like(visual_weight),
            visual_weight,
        )
        moments = visual_weight * visual + (1.0 - visual_weight) * acoustic
        energy = audio_prior[:, :1] if audio_prior.shape[1] else visual.new_zeros(len(visual), 1)
        moments = moments + self.change_proj(
            torch.cat(
                [
                    _adjacent_change(vision),
                    _adjacent_change(audio),
                    _adjacent_change(energy),
                ],
                dim=1,
            )
        )
        encoded = self.temporal(moments)
        scenes, scene_positions = self.scene_pool(encoded, scene_bounds)
        transition, anchor_positions, anchor_presence = self.decoder(
            encoded,
            scenes,
            scene_bounds,
            scene_positions,
        )
        distances = F.softplus(self.offset_head(transition))
        offsets = torch.stack([-distances[:, 0], distances[:, 1]], dim=1)
        centers = torch.arange(len(transition), device=transition.device, dtype=transition.dtype)[
            :, None
        ]
        anchor_scale = transition.new_tensor(
            [max(1, self.decoder.max_before_sec), 2, max(1, self.decoder.max_after_sec)]
        )
        anchor_delta = ((anchor_positions - centers) / anchor_scale).clamp(-1.0, 1.0)
        distance_scale = transition.new_tensor(
            [max(1, self.decoder.max_before_sec), max(1, self.decoder.max_after_sec)]
        )
        quality_features = torch.cat(
            [
                transition,
                anchor_delta,
                anchor_presence,
                (distances.detach() / distance_scale).clamp_max(2.0),
            ],
            dim=1,
        )
        return TransitionOutput(
            event_logits=self.event_head(transition).squeeze(-1),
            segment_quality_logits=self.segment_quality_head(quality_features).squeeze(-1),
            offsets=offsets,
            anchor_positions=anchor_positions,
            anchor_presence=anchor_presence,
            transition=transition,
        )


def _adjacent_change(sequence: Tensor) -> Tensor:
    if len(sequence) == 1:
        return sequence.new_zeros((1, 1))
    change = sequence.new_zeros((len(sequence), 1))
    if sequence.shape[1] == 1:
        change[1:] = (sequence[1:] - sequence[:-1]).abs()
    else:
        similarity = F.cosine_similarity(sequence[1:], sequence[:-1], dim=-1, eps=1e-6)
        change[1:, 0] = (1.0 - similarity).clamp_min(0)
    change[0] = change[1]
    return change


def _downsample(sequence: Tensor, factor: int) -> Tensor:
    # Each coarse token at index j summarizes only moments up to j * factor.
    padded = F.pad(sequence.T.unsqueeze(0), (factor - 1, 0))
    return (
        F.avg_pool1d(
            padded,
            kernel_size=factor,
            stride=factor,
        )
        .squeeze(0)
        .T
    )


def _upsample(sequence: Tensor, length: int, factor: int) -> Tensor:
    return sequence.repeat_interleave(factor, dim=0)[:length]
