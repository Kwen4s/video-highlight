import json
from pathlib import Path

from vh_agent.models import (
    DecisionTrace,
    DetectionResult,
    DetectionStats,
    DetectionTrace,
    GlobalRanking,
    Highlight,
    JudgeDecision,
    PreprocessTrace,
    ReasoningTrace,
    SceneCard,
    VideoInfo,
    VideoSummary,
)
from vh_agent.silver_labeling import (
    SILVER_ANNOTATION_REVISION,
    SILVER_METHOD,
    SILVER_MODEL,
    run_silver_labeling,
)


def _metadata(
    video_id: str,
    *,
    language: str,
    status: str = "unlabeled",
    duration_sec: float = 30.0,
) -> dict:
    return {
        "video_id": video_id,
        "title": f"Episode {video_id}",
        "path": f"/data/{video_id}.mp4",
        "language": language,
        "duration_sec": duration_sec,
        "annotation_status": status,
        "annotation_method": "",
        "highlights": [],
    }


def test_silver_labeling_writes_resumable_training_records(tmp_path, monkeypatch) -> None:
    metadata_dir = tmp_path / "metadata"
    for language, row in {
        "en": _metadata("video_en", language="en"),
        "zh": _metadata(
            "video_zh", language="zh", status="annotated", duration_sec=20.0
        ),
    }.items():
        path = metadata_dir / language / "train.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    calls = []

    class FakeService:
        def __init__(self, settings) -> None:
            assert settings.reasoning_map_model == SILVER_MODEL
            assert settings.reasoning_judge_model == SILVER_MODEL
            assert settings.reasoning_provider == "gemini"
            assert settings.max_judge_candidates == 10_000
            assert settings.max_highlights == 10_000
            assert not settings.export_clips
            assert not settings.write_result_file
            assert not settings.write_trace

        def detect_with_trace(self, task, *, retain_all_verified=False):
            assert retain_all_verified is True
            calls.append(task.video_id)
            result = DetectionResult(
                job_id=task.job_id,
                video=VideoSummary(video_id=task.video_id, title="Episode", duration_sec=30.0),
                highlights=[
                    Highlight(
                        highlight_id="internal",
                        start_sec=4.0,
                        end_sec=16.0,
                        score=0.82,
                        highlight_type="reveal",
                        description="identity revealed",
                        reason="old state -> proof -> new state",
                        clip_url="",
                    )
                ],
            )
            scene = SceneCard(
                scene_id="scene_0001",
                start_sec=4.0,
                end_sec=16.0,
                event_type=["reveal"],
                state_before="identity unknown",
                new_evidence="identity proof",
                state_after="identity known",
                evidence=["F02 8s identity proof"],
            )
            first = JudgeDecision(
                map_supported=True,
                is_highlight=True,
                score=0.82,
                highlight_type="reveal",
                description="identity revealed",
                reason="old state -> proof -> new state",
                confidence=0,
                evidence_grounding=0.9,
                narrative_impact=0.8,
                standalone_clarity=0.8,
                clipability=0.8,
                start_sec=4,
                end_sec=16,
                evidence=["F02 8s identity proof"],
                setup_evidence_times_sec=[4],
                decisive_evidence_times_sec=[8],
                reaction_evidence_times_sec=[12],
            )
            second = first.model_copy(update={"score": 0.8, "narrative_impact": 0.75})
            consensus = first.model_copy(update={"confidence": 0.84})
            trace = DetectionTrace(
                preprocess=PreprocessTrace(
                    video=VideoInfo(
                        path=Path(task.video_path),
                        duration_sec=30,
                        width=480,
                        height=852,
                        fps=25,
                        has_audio=True,
                    ),
                    scenes=[],
                    transcript=[],
                    audio_events=[],
                    frame_samples=[],
                    saliency_per_second=[],
                ),
                local_candidates=[],
                reasoning=ReasoningTrace(
                    scenes=[scene],
                    decisions=[
                        DecisionTrace(
                            scene=scene,
                            decision=consensus,
                            votes=[first, second],
                        )
                    ],
                    ranking=GlobalRanking(
                        ranked_highlight_ids=["internal"],
                        selected_highlight_ids=["internal"],
                    ),
                ),
                stats=DetectionStats(
                    sampled_frames=0,
                    detected_scenes=1,
                    transcript_segments=0,
                    ocr_segments=0,
                    audio_events=0,
                    local_candidate_windows=0,
                    scene_map_calls=1,
                    judge_calls=2,
                    listwise_calls=0,
                ),
            )
            return result, trace

    monkeypatch.setattr("vh_agent.silver_labeling.HighlightDetectionService", FakeService)
    output_dir = tmp_path / "silver"
    annotations = run_silver_labeling(
        "run_1", metadata_dir=metadata_dir, output_dir=output_dir
    )

    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    assert calls == ["video_zh", "video_en"]
    assert {row["annotation_status"] for row in rows} == {"silver"}
    assert {row["annotation_method"] for row in rows} == {SILVER_METHOD}
    assert {row["annotation_revision"] for row in rows} == {
        SILVER_ANNOTATION_REVISION
    }
    assert all(row["annotator_ids"] == [SILVER_MODEL] for row in rows)
    assert all(row["highlights"][0]["duration_sec"] == 12.0 for row in rows)
    assert all(row["highlights"][0]["review_status"] == "pending" for row in rows)
    assert all(row["global_quality_score"] == 0.84 for row in rows)
    assert all(row["highlights"][0]["annotator_scores"] == [0.82, 0.8] for row in rows)
    assert all(row["highlights"][0]["setup_times_sec"] == [4.0] for row in rows)
    assert all(row["highlights"][0]["decisive_times_sec"] == [8.0] for row in rows)
    assert all(row["highlights"][0]["reaction_times_sec"] == [12.0] for row in rows)
    assert all(row["scene_labels"][0]["label"] == 1 for row in rows)
    assert all(len(row["scene_labels"][0]["votes"]) == 2 for row in rows)

    run_silver_labeling("run_1", metadata_dir=metadata_dir, output_dir=output_dir)
    assert calls == ["video_zh", "video_en"]
    manifest = json.loads((output_dir / "run_1" / "run.json").read_text())
    assert manifest["selection_policy"] == "all_verified_scenes_without_listwise_budget"
    assert manifest["current_annotation_revision"] == SILVER_ANNOTATION_REVISION
    assert manifest["revision_policy"] == "records_without_annotation_revision_are_scene_v1"
    assert manifest["completed_records"] == 2
    assert not (output_dir / "run_1" / "work").exists()
