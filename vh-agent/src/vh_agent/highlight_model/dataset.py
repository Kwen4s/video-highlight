"""Versioned interval annotations with explicit positives, negatives and unknown regions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import HighlightModelConfig


@dataclass(frozen=True)
class HighlightAnnotation:
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class VideoExample:
    video_id: str
    drama_id: str
    path: Path
    duration_sec: float
    language: str
    highlights: tuple[HighlightAnnotation, ...]
    hard_negative_intervals: tuple[tuple[float, float], ...]
    split: str | None = None


@dataclass(frozen=True)
class EpisodeTargets:
    eventness: Tensor
    sample_weight: Tensor
    offsets: Tensor
    boundary_mask: Tensor
    segments: Tensor
    ignore_mask: Tensor
    hard_negative_mask: Tensor


@dataclass(frozen=True)
class EpisodeBatch:
    video: VideoExample
    vision: Tensor
    audio: Tensor
    audio_prior: Tensor
    availability: Tensor
    scene_bounds: Tensor
    targets: EpisodeTargets


def load_videos(path: Path) -> list[VideoExample]:
    """Read vh-data snapshots; never infer negative labels or resplit episodes."""
    videos, ids, hashes = [], set(), set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row["video_id"])
        if video_id in ids or row["sha256"] in hashes:
            raise ValueError(f"duplicate source video: {video_id}")
        ids.add(video_id)
        hashes.add(row["sha256"])
        source = Path(row["path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError(f"source content changed: {video_id}")
        duration = float(row["duration_sec"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"invalid duration: {video_id}")
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split: {video_id}")
        if row["label_source"] != "model_reviewed":
            raise ValueError(f"unreviewed annotation: {video_id}")
        positive = _intervals(row["highlights"], duration, video_id)
        negative = _intervals(row["negative_intervals"], duration, video_id)
        if any(min(b, d) > max(a, c) for a, b in positive for c, d in negative):
            raise ValueError(f"positive/negative conflict: {video_id}")
        if not positive and not negative:
            raise ValueError(f"no supervised region: {video_id}")
        videos.append(
            VideoExample(
                video_id=video_id,
                drama_id=str(row["drama_id"]),
                path=source,
                duration_sec=duration,
                language=str(row["language"]),
                highlights=tuple(HighlightAnnotation(a, b) for a, b in positive),
                hard_negative_intervals=negative,
                split=split,
            )
        )
    if not videos:
        raise ValueError(f"no annotations in {path}")
    return videos


def _intervals(items: list[dict], duration: float, video_id: str):
    values = tuple((float(item["start_sec"]), float(item["end_sec"])) for item in items)
    if any(
        not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration) for a, b in values
    ):
        raise ValueError(f"invalid interval: {video_id}")
    return values


def split_dataset(videos: list[VideoExample]) -> dict[str, list[VideoExample]]:
    """Honor the persistent drama split and reject any group leakage."""
    splits = {name: [] for name in ("train", "val", "test")}
    groups = {}
    for video in sorted(videos, key=lambda item: item.video_id):
        if video.split not in splits:
            raise ValueError(f"missing split: {video.video_id}")
        if video.drama_id in groups and groups[video.drama_id] != video.split:
            raise ValueError(f"drama appears in multiple splits: {video.drama_id}")
        groups[video.drama_id] = video.split
        splits[video.split].append(video)
    if not splits["train"] or not splits["val"]:
        raise ValueError("training needs a train group and a reviewed validation group")
    return splits


class EpisodeDataset(Dataset[EpisodeBatch]):
    def __init__(
        self,
        videos: list[VideoExample],
        feature_cache_dir: Path,
        config: HighlightModelConfig,
    ) -> None:
        self.videos = videos
        self.feature_cache_dir = feature_cache_dir
        self.config = config

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> EpisodeBatch:
        from .encoders import feature_signature

        video = self.videos[index]
        path = feature_path(self.feature_cache_dir, video.video_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing moment feature cache: {path}")
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["signature"] != feature_signature(video, self.config):
            raise ValueError(f"stale feature cache for {video.video_id}; run --stage features")
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


def build_targets(video: VideoExample, config: HighlightModelConfig) -> EpisodeTargets:
    """Use geometric proposal centers; only explicitly labeled negatives receive negative loss."""
    length = max(1, math.ceil(video.duration_sec))
    grid = torch.arange(length, dtype=torch.float32)
    eventness = torch.zeros(length)
    sample_weight = torch.ones(length)
    offsets = torch.zeros((length, 2))
    boundary_mask = torch.zeros(length, dtype=torch.bool)
    assignment_quality = torch.full((length,), -1.0)
    segments = torch.tensor(
        [[item.start_sec, item.end_sec] for item in video.highlights],
        dtype=torch.float32,
    ).reshape(-1, 2)
    inside_highlight = torch.zeros(length, dtype=torch.bool)

    for highlight in video.highlights:
        start = min(length, max(0, math.floor(highlight.start_sec)))
        end = min(length, max(start + 1, math.ceil(highlight.end_sec)))
        inside_highlight[start:end] = True

    hard_negative_mask = torch.zeros(length, dtype=torch.bool)
    for start, end in video.hard_negative_intervals:
        negative = (grid >= start) & (grid < end) & ~inside_highlight
        hard_negative_mask |= negative
        sample_weight[negative] = torch.maximum(
            sample_weight[negative],
            torch.full_like(sample_weight[negative], config.hard_negative_weight),
        )

    for highlight in video.highlights:
        event_time = (highlight.start_sec + highlight.end_sec) / 2
        gaussian = torch.exp(-0.5 * ((grid - event_time) / config.event_sigma_sec).square())
        eventness = torch.maximum(
            eventness, gaussian * ((grid >= highlight.start_sec) & (grid < highlight.end_sec))
        )
        center = min(length - 1, max(0, round(event_time)))
        sample_weight[center] = max(float(sample_weight[center]), config.event_peak_weight)
        for index in range(
            max(0, center - config.center_sampling_radius_sec),
            min(length, center + config.center_sampling_radius_sec + 1),
        ):
            if not highlight.start_sec <= index < highlight.end_sec:
                continue
            distance = abs(index - event_time)
            assignment = 1.0 - distance / (config.center_sampling_radius_sec + 1.0)
            if assignment < assignment_quality[index]:
                continue
            assignment_quality[index] = assignment
            boundary_mask[index] = True
            offsets[index, 0] = highlight.start_sec - index
            offsets[index, 1] = highlight.end_sec - index
    ignore_mask = ~hard_negative_mask & (eventness < 0.3)
    return EpisodeTargets(
        eventness=eventness,
        sample_weight=sample_weight,
        offsets=offsets,
        boundary_mask=boundary_mask,
        segments=segments,
        ignore_mask=ignore_mask,
        hard_negative_mask=hard_negative_mask,
    )


def feature_path(root: Path, video_id: str) -> Path:
    return root / video_id / "moments.pt"
