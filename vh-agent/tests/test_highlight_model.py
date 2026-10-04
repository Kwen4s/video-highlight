import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vh_agent.highlight_model.config import HighlightModelConfig
from vh_agent.highlight_model.dataset import (
    EpisodeDataset,
    HighlightAnnotation,
    VideoExample,
    build_targets,
    feature_path,
    load_videos,
    split_dataset,
)
from vh_agent.highlight_model.decoding import PredictedSegment, decode_segments, segment_metrics
from vh_agent.highlight_model.encoders import feature_signature
from vh_agent.highlight_model.losses import highlight_localization_loss
from vh_agent.highlight_model.network import (
    HierarchicalTemporalEncoder,
    NarrativeTransitionLocalizer,
    TransitionOutput,
)
from vh_agent.preprocessing.media import video_fingerprint


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


def _video(tmp_path: Path) -> VideoExample:
    return VideoExample(
        video_id="video_1",
        drama_id="drama_1",
        path=tmp_path / "video.mp4",
        duration_sec=40.0,
        language="zh",
        highlights=(
            HighlightAnnotation(
                start_sec=14.0,
                end_sec=28.0,
            ),
        ),
        hard_negative_intervals=((30.0, 36.0),),
    )


def test_interval_targets_ignore_unknown_regions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    targets = build_targets(_video(tmp_path), config)

    assert targets.eventness[21] == torch.tensor(1.0)
    assert targets.eventness[14] < 0.001
    assert targets.eventness[20] > 0.5
    assert targets.boundary_mask.nonzero().flatten().tolist() == list(range(17, 26))
    assert targets.offsets[20].tolist() == [-6.0, 8.0]
    assert targets.sample_weight[21] == config.event_peak_weight
    assert targets.sample_weight[32] == config.hard_negative_weight
    assert targets.segments.tolist() == [[14.0, 28.0]]
    assert bool(targets.ignore_mask[5])
    assert bool(targets.ignore_mask[14])
    assert not bool(targets.ignore_mask[20])
    assert not bool(targets.ignore_mask[32])
    assert bool(targets.hard_negative_mask[32])
    assert not bool(targets.hard_negative_mask[20])


def test_drama_split_preserves_assignments_and_rejects_leakage(tmp_path: Path) -> None:
    import pytest

    videos = [
        replace(
            _video(tmp_path),
            video_id=f"video_{i}",
            drama_id=f"drama_{i // 2}",
            split=("train", "val", "test")[i // 2],
        )
        for i in range(6)
    ]
    splits = split_dataset(videos)
    assert splits == split_dataset(list(reversed(videos)))
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 2, "val": 2, "test": 2}
    with pytest.raises(ValueError, match="multiple splits"):
        split_dataset([*videos, replace(videos[0], video_id="other_episode", split="test")])
    with pytest.raises(ValueError, match="validation"):
        split_dataset([v for v in videos if v.split == "train"])


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
    assert (output.offsets[:, 0] < output.offsets[:, 1]).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_candidate_ranking_penalizes_off_peak_scores(tmp_path: Path) -> None:
    targets = build_targets(_video(tmp_path), _config(tmp_path))
    offsets = torch.zeros((40, 2))
    offsets[:, 0] = -6.0
    offsets[:, 1] = 8.0

    def output_for(peak_values: dict[int, float]) -> TransitionOutput:
        logits = torch.full((40,), -4.0)
        for index, value in peak_values.items():
            logits[index] = value
        return TransitionOutput(
            event_logits=logits,
            segment_quality_logits=logits.clone(),
            offsets=offsets,
            anchor_positions=torch.zeros(40, 3),
            anchor_presence=torch.zeros(40, 3),
            transition=torch.empty(0),
        )

    centered = highlight_localization_loss(output_for({20: 4.0}), targets)
    distractor = highlight_localization_loss(output_for({32: 6.0, 20: 4.0}), targets)
    assert centered.ranking < distractor.ranking
    assert torch.isfinite(centered.quality)
    assert torch.isfinite(distractor.quality)


def test_unknown_regions_receive_no_classification_gradient(tmp_path: Path) -> None:
    event = torch.zeros(40, requires_grad=True)
    quality = torch.zeros(40, requires_grad=True)
    offsets = torch.tensor([[-6.0, 8.0]]).repeat(40, 1)
    output = TransitionOutput(
        event, quality, offsets, torch.zeros(40, 3), torch.zeros(40, 3), torch.empty(0)
    )
    loss = highlight_localization_loss(output, build_targets(_video(tmp_path), _config(tmp_path)))
    loss.total.backward()
    assert event.grad[5] == quality.grad[5] == 0
    assert event.grad[32] != 0 and quality.grad[32] != 0


def test_training_uses_validation_then_tests_selected_checkpoint_once(tmp_path, monkeypatch):
    from vh_agent.highlight_model import trainer

    config = replace(_config(tmp_path), stage="train", device="cpu", epochs=2)
    rows = []
    for split in ("train", "val", "test"):
        source = tmp_path / f"{split}.mp4"
        source.write_bytes(split.encode())
        rows.append(
            {
                "video_id": split,
                "drama_id": split,
                "split": split,
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "duration_sec": 40,
                "language": "zh",
                "label_source": "model_reviewed",
                "highlights": [{"start_sec": 14, "end_sec": 28}],
                "negative_intervals": [
                    {"start_sec": 0, "end_sec": 10},
                    {"start_sec": 30, "end_sec": 40},
                ],
            }
        )
        cache = feature_path(config.feature_cache_dir, split)
        cache.parent.mkdir(parents=True)
        media = config.media_cache_dir / video_fingerprint(source)
        media.mkdir(parents=True)
        (media / "preprocess.json").write_text(json.dumps({"signature": "test-input"}))
        video = replace(_video(tmp_path), video_id=split, path=source)
        torch.save(
            {
                "signature": feature_signature(video, config),
                "vision": torch.randn(40, 16),
                "audio": torch.randn(40, 12),
                "audio_prior": torch.randn(40, 4),
                "availability": torch.ones(40, 2),
                "scene_bounds": torch.tensor([[0, 40]]),
            },
            cache,
        )
    config.annotations.write_text("\n".join(json.dumps(row) for row in rows))
    assert len(load_videos(config.annotations)) == 3
    evaluations = []
    evaluate = trainer._evaluate_epoch

    def observe(model, dataset, videos, settings):
        evaluations.append(videos[0].split)
        return evaluate(model, dataset, videos, settings)

    monkeypatch.setattr(trainer, "_evaluate_epoch", observe)
    report = trainer.fit_highlight_model(config)
    assert evaluations == ["val", "val", "test"]
    assert report["selection"] == "best_val_f1_iou_0.3"
    assert "test" in report and "val" in report
    feedback = [
        json.loads(line) for line in Path(report["training_feedback"]).read_text().splitlines()
    ]
    assert [row["video_id"] for row in feedback] == ["train"]
    assert all(set(s) == {"start_sec", "end_sec", "score"} for s in feedback[0]["segments"])
    assert (config.output_dir / "checkpoint.pt").read_bytes() == (
        config.output_dir / "best.pt"
    ).read_bytes()


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
    predictions = decode_segments(output, 40.0, 0.5, 0.4)
    metrics = segment_metrics({"video_1": predictions}, [_video(tmp_path)])

    assert len(predictions) == 1
    assert predictions[0].start_sec == 14.0
    assert predictions[0].end_sec == 28.0
    assert metrics["f1_iou_0.5"] == 1.0
    assert metrics["f1_iou_0.3"] == 1.0
    assert metrics["f1_iou_0.7"] == 1.0
    assert metrics["candidates_per_minute"] == 1.5
    checked = segment_metrics(
        {"video_1": [*predictions, PredictedSegment(1, 4, 0.9), PredictedSegment(30, 35, 0.9)]},
        [_video(tmp_path)],
    )
    assert checked["ignored_unknown_predictions"] == 1
    assert checked["precision_iou_0.5"] == 0.5
    assert checked["candidates_per_minute"] == 4.5


def test_decoder_preserves_learned_lengths_and_all_nonduplicate_candidates():
    logits = torch.full((140,), -10.0)
    offsets = torch.zeros((140, 2))
    intervals = [(1.5, 4.5), (10.0, 70.0), *[(float(i), float(i + 2)) for i in range(80, 135, 4)]]
    for start, end in intervals:
        center = round((start + end) / 2)
        logits[center] = 10
        offsets[center] = torch.tensor([start - center, end - center])
    output = TransitionOutput(
        logits, logits, offsets, torch.zeros(140, 3), torch.zeros(140, 3), torch.empty(0)
    )
    predictions = decode_segments(output, 140, 0.5, 0.4)
    assert len(predictions) == len(intervals) > 12
    assert sorted((p.start_sec, p.end_sec) for p in predictions) == intervals


def test_training_rejects_features_from_an_old_embedding_instruction(tmp_path, monkeypatch):
    from vh_agent.highlight_model import encoders

    config = _config(tmp_path)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source")
    video = _video(tmp_path)
    media = config.media_cache_dir / video_fingerprint(source)
    media.mkdir(parents=True)
    (media / "preprocess.json").write_text(json.dumps({"signature": "test-input"}))
    cache = feature_path(config.feature_cache_dir, video.video_id)
    cache.parent.mkdir(parents=True)
    torch.save({"signature": feature_signature(video, config)}, cache)
    monkeypatch.setattr(encoders, "VISION_INSTRUCTION", "different task")
    with pytest.raises(ValueError, match="stale feature cache"):
        EpisodeDataset([video], config.feature_cache_dir, config)[0]


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
