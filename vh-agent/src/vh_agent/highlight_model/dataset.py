"""Unchanged silver annotations and event-centered supervision targets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import HighlightModelConfig


@dataclass(frozen=True)
class HighlightAnnotation:
    start_sec: float
    end_sec: float
    confidence: float
    setup_times_sec: tuple[float, ...]
    decisive_times_sec: tuple[float, ...]
    reaction_times_sec: tuple[float, ...]


@dataclass(frozen=True)
class SilverVideo:
    video_id: str
    drama_id: str
    path: Path
    duration_sec: float
    language: str
    highlights: tuple[HighlightAnnotation, ...]
    hard_negative_intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class EpisodeTargets:
    eventness: Tensor
    sample_weight: Tensor
    offsets: Tensor
    boundary_mask: Tensor
    anchor_positions: Tensor
    anchor_mask: Tensor
    segments: Tensor
    event_peaks: Tensor
    event_peak_gt: Tensor
    ignore_mask: Tensor
    hard_negative_mask: Tensor


@dataclass(frozen=True)
class EpisodeBatch:
    video: SilverVideo
    vision: Tensor
    audio: Tensor
    audio_prior: Tensor
    availability: Tensor
    scene_bounds: Tensor
    targets: EpisodeTargets


def load_silver_videos(path: Path) -> list[SilverVideo]:
    """Load every completed annotation without filtering or rewriting the source file."""
    videos: list[SilverVideo] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row["video_id"])
        if video_id in seen:
            raise ValueError(f"duplicate silver annotation: {video_id}")
        seen.add(video_id)
        source = Path(str(row["path"]))
        if not source.is_file():
            raise FileNotFoundError(f"missing source video for {video_id}: {source}")
        highlights = tuple(
            HighlightAnnotation(
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
                confidence=float(item.get("confidence", 1.0)),
                setup_times_sec=_times(item.get("setup_times_sec", [])),
                decisive_times_sec=_times(item.get("decisive_times_sec", [])),
                reaction_times_sec=_times(item.get("reaction_times_sec", [])),
            )
            for item in row.get("highlights", [])
        )
        hard_negatives = tuple(
            (
                float(item["scene"]["start_sec"]),
                float(item["scene"]["end_sec"]),
            )
            for item in row.get("scene_labels", [])
            if item.get("label") == 0 and isinstance(item.get("scene"), dict)
        )
        videos.append(
            SilverVideo(
                video_id=video_id,
                drama_id=str(row["drama_id"]),
                path=source,
                duration_sec=float(row["duration_sec"]),
                language=str(row.get("language") or "zh"),
                highlights=highlights,
                hard_negative_intervals=hard_negatives,
            )
        )
    if not videos:
        raise ValueError(f"no annotations in {path}")
    return videos


def split_by_video(videos: list[SilverVideo], seed: int) -> dict[str, list[SilverVideo]]:
    if len(videos) < 2:
        raise ValueError("at least two videos are required")
    shuffled = sorted(
        videos,
        key=lambda video: hashlib.sha256(f"{seed}:{video.video_id}".encode()).hexdigest(),
    )
    train_end = min(len(shuffled) - 1, max(1, round(len(shuffled) * 0.9)))
    return {
        "train": shuffled[:train_end],
        "test": shuffled[train_end:],
    }


class EpisodeDataset(Dataset[EpisodeBatch]):
    def __init__(
        self,
        videos: list[SilverVideo],
        feature_cache_dir: Path,
        config: HighlightModelConfig,
    ) -> None:
        self.videos = videos
        self.feature_cache_dir = feature_cache_dir
        self.config = config

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> EpisodeBatch:
        video = self.videos[index]
        path = feature_path(self.feature_cache_dir, video.video_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing moment feature cache: {path}")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        targets = build_targets(video, self.config)
        expected_length = len(targets.eventness)
        tensors = {
            name: torch.as_tensor(cached[name]).float()
            for name in ("vision", "audio", "audio_prior", "availability")
        }
        if any(len(value) != expected_length for value in tensors.values()):
            raise RuntimeError(f"feature length mismatch for {video.video_id}")
        return EpisodeBatch(
            video=video,
            vision=tensors["vision"],
            audio=tensors["audio"],
            audio_prior=tensors["audio_prior"],
            availability=tensors["availability"],
            scene_bounds=torch.as_tensor(cached["scene_bounds"], dtype=torch.long),
            targets=targets,
        )


def build_targets(video: SilverVideo, config: HighlightModelConfig) -> EpisodeTargets:
    """Center supervision on one decisive peak per highlight; ignore the rest of the clip."""
    length = max(1, math.ceil(video.duration_sec))
    grid = torch.arange(length, dtype=torch.float32)
    eventness = torch.zeros(length)
    sample_weight = torch.ones(length)
    offsets = torch.zeros((length, 2))
    boundary_mask = torch.zeros(length, dtype=torch.bool)
    anchor_positions = torch.zeros((length, 3))
    anchor_mask = torch.zeros((length, 3), dtype=torch.bool)
    assignment_quality = torch.full((length,), -1.0)
    segments = torch.tensor(
        [[item.start_sec, item.end_sec] for item in video.highlights],
        dtype=torch.float32,
    ).reshape(-1, 2)
    event_peaks = torch.zeros(length, dtype=torch.bool)
    event_peak_gt = torch.full((length,), -1, dtype=torch.long)
    inside_highlight = torch.zeros(length, dtype=torch.bool)

    for highlight in video.highlights:
        start = min(length, max(0, math.floor(highlight.start_sec)))
        end = min(length, max(start + 1, math.ceil(highlight.end_sec)))
        inside_highlight[start:end] = True

    hard_negative_mask = torch.zeros(length, dtype=torch.bool)
    for start, end in video.hard_negative_intervals:
        negative = (grid >= start) & (grid <= end) & ~inside_highlight
        hard_negative_mask |= negative
        sample_weight[negative] = torch.maximum(
            sample_weight[negative],
            torch.full_like(sample_weight[negative], config.hard_negative_weight),
        )

    for highlight_index, highlight in enumerate(video.highlights):
        decisive = highlight.decisive_times_sec or ((highlight.start_sec + highlight.end_sec) / 2,)
        for event_time in decisive:
            gaussian = torch.exp(-0.5 * ((grid - event_time) / config.event_sigma_sec).square())
            quality = max(0.05, min(1.0, highlight.confidence))
            eventness = torch.maximum(eventness, gaussian * quality)
            center = min(length - 1, max(0, round(event_time)))
            event_peaks[center] = True
            event_peak_gt[center] = highlight_index
            sample_weight[center] = max(float(sample_weight[center]), config.event_peak_weight)
            for index in range(
                max(0, center - config.center_sampling_radius_sec),
                min(length, center + config.center_sampling_radius_sec + 1),
            ):
                distance = abs(index - event_time)
                assignment = quality * (1.0 - distance / (config.center_sampling_radius_sec + 1.0))
                if assignment < assignment_quality[index]:
                    continue
                assignment_quality[index] = assignment
                boundary_mask[index] = True
                offsets[index, 0] = highlight.start_sec - index
                offsets[index, 1] = highlight.end_sec - index
                _assign_anchors(
                    anchor_positions,
                    anchor_mask,
                    index,
                    event_time,
                    highlight,
                )
    ignore_mask = inside_highlight & (eventness < 0.3)
    return EpisodeTargets(
        eventness=eventness,
        sample_weight=sample_weight,
        offsets=offsets,
        boundary_mask=boundary_mask,
        anchor_positions=anchor_positions,
        anchor_mask=anchor_mask,
        segments=segments,
        event_peaks=event_peaks,
        event_peak_gt=event_peak_gt,
        ignore_mask=ignore_mask,
        hard_negative_mask=hard_negative_mask,
    )


def feature_path(root: Path, video_id: str) -> Path:
    return root / video_id / "moments.pt"


def _assign_anchors(
    positions: Tensor,
    mask: Tensor,
    index: int,
    event_time: float,
    highlight: HighlightAnnotation,
) -> None:
    setup = [value for value in highlight.setup_times_sec if value <= event_time]
    reaction = [value for value in highlight.reaction_times_sec if value >= event_time]
    if setup:
        positions[index, 0] = max(setup)
        mask[index, 0] = True
    positions[index, 1] = event_time
    mask[index, 1] = True
    if reaction:
        positions[index, 2] = min(reaction)
        mask[index, 2] = True


def _times(values: list[Any]) -> tuple[float, ...]:
    return tuple(sorted(float(value) for value in values))
