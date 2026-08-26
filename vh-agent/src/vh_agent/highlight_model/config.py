"""Configuration for frozen feature extraction and temporal model fitting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class HighlightModelConfig:
    annotations: Path
    output_dir: Path
    vision_model_path: Path
    audio_model_path: Path
    media_cache_dir: Path
    feature_cache_dir: Path
    stage: Literal["features", "train", "all"] = "all"
    device: str = "cuda:0"
    feature_device: str = "cuda:0"
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    gradient_accumulation: int = 4
    model_dim: int = 512
    attention_heads: int = 8
    temporal_layers_per_level: int = 2
    dropout: float = 0.1
    vision_batch_size: int = 8
    audio_window_sec: int = 30
    audio_overlap_sec: int = 5
    event_sigma_sec: float = 1.5
    center_sampling_radius_sec: int = 4
    hard_negative_weight: float = 2.0
    max_before_sec: int = 32
    max_after_sec: int = 8
    nms_iou: float = 0.4
    max_highlights: int = 12
    seed: int = 13
