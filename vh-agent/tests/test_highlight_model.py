from dataclasses import replace
from pathlib import Path

import torch

from vh_agent.highlight_model.config import HighlightModelConfig
from vh_agent.highlight_model.dataset import (
    HighlightAnnotation,
    SilverVideo,
    build_targets,
    split_by_video,
)
from vh_agent.highlight_model.decoding import decode_segments, segment_metrics
from vh_agent.highlight_model.losses import highlight_localization_loss
from vh_agent.highlight_model.network import (
    HierarchicalTemporalEncoder,
    NarrativeTransitionLocalizer,
    TransitionOutput,
)


def _config(tmp_path: Path) -> HighlightModelConfig:
    return HighlightModelConfig(
        annotations=tmp_path / "annotations.jsonl",
        output_dir=tmp_path / "output",
        vision_model_path=tmp_path / "vision",
        audio_model_path=tmp_path / "audio",
        media_cache_dir=tmp_path / "media",
        feature_cache_dir=tmp_path / "features",
        model_dim=32,
        attention_heads=4,
        temporal_layers_per_level=1,
        max_before_sec=8,
        max_after_sec=4,
    )


def _video(tmp_path: Path) -> SilverVideo:
    return SilverVideo(
        video_id="video_1",
        drama_id="drama_1",
        path=tmp_path / "video.mp4",
        duration_sec=40.0,
        language="zh",
        highlights=(
            HighlightAnnotation(
                start_sec=14.0,
                end_sec=28.0,
                confidence=0.9,
                setup_times_sec=(16.0,),
                decisive_times_sec=(20.0,),
                reaction_times_sec=(24.0,),
            ),
        ),
        hard_negative_intervals=((30.0, 36.0),),
    )


def test_targets_are_centered_on_decisive_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    targets = build_targets(_video(tmp_path), config)

    assert targets.eventness[20] == torch.tensor(0.9)
    assert targets.eventness[14] < 0.001
    assert targets.boundary_mask.nonzero().flatten().tolist() == list(range(16, 25))
    assert targets.offsets[20].tolist() == [-6.0, 8.0]
    assert targets.anchor_positions[20].tolist() == [16.0, 20.0, 24.0]
    assert targets.anchor_mask[20].all()
    assert targets.sample_weight[32] == config.hard_negative_weight

    assert targets.segments.tolist() == [[14.0, 28.0]]


def test_video_split_is_seeded_and_disjoint(tmp_path: Path) -> None:
    videos = [
        replace(
            _video(tmp_path),
            video_id=f"video_{index}",
            drama_id=f"drama_{index % 2}",
        )
        for index in range(10)
    ]
    splits = split_by_video(videos, seed=13)

    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 9,
        "test": 1,
    }
    assert splits == split_by_video(list(reversed(videos)), seed=13)
    assert len({video.video_id for rows in splits.values() for video in rows}) == 10


def test_transition_model_runs_end_to_end_and_constrains_offsets(tmp_path: Path) -> None:
    torch.manual_seed(3)
    config = _config(tmp_path)
    video = _video(tmp_path)
    targets = build_targets(video, config)
    model = NarrativeTransitionLocalizer(
        vision_dim=16,
        audio_dim=12,
        audio_prior_dim=4,
        model_dim=config.model_dim,
        heads=config.attention_heads,
        temporal_layers_per_level=config.temporal_layers_per_level,
        dropout=0.0,
        max_before_sec=config.max_before_sec,
        max_after_sec=config.max_after_sec,
    )
    output = model(
        torch.randn(40, 16),
        torch.randn(40, 12),
        torch.randn(40, 4),
        torch.ones(40, 2),
        torch.tensor([[0, 12], [12, 26], [26, 40]]),
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses = highlight_localization_loss(output, targets)
    losses.total.backward()

    assert output.event_logits.shape == (40,)
    assert output.segment_quality_logits.shape == (40,)
    assert output.offsets.shape == (40, 2)
    assert output.anchor_positions.shape == (40, 3)
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.quality)
    assert torch.isfinite(losses.ranking)
    assert (output.offsets[:, 0] <= 0).all()
    assert (output.offsets[:, 1] >= 0).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_peak_decoder_and_segment_iou_metrics(tmp_path: Path) -> None:
    logits = torch.full((40,), -10.0)
    logits[20] = 10.0
    offsets = torch.zeros((40, 2))
    offsets[20] = torch.tensor([-6.0, 8.0])
    output = TransitionOutput(
        segment_quality_logits=logits,
        event_logits=logits,
        offsets=offsets,
        anchor_positions=torch.zeros(40, 3),
        anchor_presence=torch.zeros(40, 3),
        transition=torch.empty(0),
    )
    predictions = decode_segments(output, 40.0, 0.5, 0.4, 12)
    metrics = segment_metrics({"video_1": predictions}, [_video(tmp_path)])

    assert len(predictions) == 1
    assert predictions[0].start_sec == 14.0
    assert predictions[0].end_sec == 28.0
    assert metrics["f1_iou_0.5"] == 1.0
    assert metrics["f1_iou_0.3"] == 1.0
    assert metrics["f1_iou_0.7"] == 1.0


def test_hierarchical_encoder_does_not_leak_future_context() -> None:
    torch.manual_seed(11)
    encoder = HierarchicalTemporalEncoder(
        dim=32,
        heads=4,
        layers=1,
        dropout=0.0,
    ).eval()
    original = torch.randn(24, 32)
    changed_future = original.clone()
    changed_future[10:] = torch.randn_like(changed_future[10:])

    with torch.inference_mode():
        first = encoder(original)
        second = encoder(changed_future)
    torch.testing.assert_close(first[:10], second[:10], rtol=0.0, atol=1e-6)
