import json
from pathlib import Path

from vh_agent import pipeline as pipeline_module
from vh_agent.config import Settings
from vh_agent.models import (
    AudioEvent,
    CandidateWindow,
    DetectionResult,
    DetectionTask,
    FrameSample,
    JudgeConsensus,
    JudgeDecision,
    SceneNarrative,
    SceneSegment,
    TranscriptSegment,
    VideoInfo,
)
from vh_agent.pipeline import HighlightOrchestrator


def test_job_writes_only_result_clips_and_optional_trace(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"video")

    def fake_frames(_video: Path, output_dir: Path, *_args) -> list[FrameSample]:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_path = output_dir / "frame_000001.jpg"
        if not frame_path.exists():
            frame_path.write_bytes(b"frame")
        return [FrameSample(timestamp_sec=1, path=frame_path)]

    def fake_audio(_video: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return output_path

    def fake_export(_video: Path, output_path: Path, *_args) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clip")
        return output_path

    cache_only = False

    class FakeReasoner:
        def __init__(self, _settings) -> None:
            pass

        def map_scene(self, _video, scene):
            if cache_only:
                raise AssertionError("Scene Map should be loaded from cache")
            return SceneNarrative(
                scene_id=scene.scene_id,
                event_type=["reveal"],
                state_before="人物相信旧事实",
                new_evidence="身份台词出现",
                state_after="人物确认真实身份",
                salience=0.9,
                uncertainty=0.2,
                evidence=["[2.00s ASR] 原来是你"],
            )

        def judge(
            self,
            _video,
            scene,
            _previous_scene,
            _ledger,
        ):
            decision = JudgeDecision(
                map_supported=True,
                is_highlight=True,
                score=0.9,
                highlight_type="reveal",
                description="身份揭露",
                reason="新证据改变此前认知",
                confidence=0.9,
                evidence_grounding=0.9,
                narrative_impact=0.9,
                standalone_clarity=0.9,
                clipability=0.9,
                start_sec=scene.start_sec,
                end_sec=scene.end_sec,
                evidence=["[2.00s ASR] 原来是你"],
                setup_evidence_times_sec=[2.0],
                decisive_evidence_times_sec=[8.0],
            )
            return JudgeConsensus(decision=decision, votes=[decision, decision], calls=2)

    monkeypatch.setattr(
        pipeline_module,
        "probe_video",
        lambda path, language: VideoInfo(
            path=path,
            duration_sec=20,
            width=480,
            height=852,
            fps=25,
            has_audio=True,
            title="测试短剧",
            language=language,
        ),
    )
    monkeypatch.setattr(pipeline_module, "extract_frames", fake_frames)
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_audio)
    monkeypatch.setattr(pipeline_module, "audio_energy_per_second", lambda _path: [0.5] * 20)
    monkeypatch.setattr(
        pipeline_module,
        "detect_scenes",
        lambda _path: [SceneSegment(start_sec=0, end_sec=20)],
    )
    monkeypatch.setattr(
        HighlightOrchestrator,
        "_transcribe",
        lambda *_args: [TranscriptSegment(start_sec=0, end_sec=20, text="原来是你")],
    )
    monkeypatch.setattr(pipeline_module, "extract_subtitle_segments", lambda *_args: [])
    monkeypatch.setattr(
        pipeline_module,
        "extract_audio_events",
        lambda *_args: [AudioEvent(start_sec=0, end_sec=20, event="speech")],
    )
    monkeypatch.setattr(pipeline_module, "score_semantic_transitions", lambda *_args: None)
    monkeypatch.setattr(
        pipeline_module,
        "build_candidates",
        lambda **_kwargs: [CandidateWindow(start_sec=2, end_sec=14, local_score=0.8)],
    )
    monkeypatch.setattr(pipeline_module, "OpenAIReasoner", FakeReasoner)
    monkeypatch.setattr(pipeline_module, "export_clip", fake_export)

    settings = Settings(
        VH_MODEL_CACHE_DIR=tmp_path / "models",
        VH_MEDIA_CACHE_DIR=tmp_path / "cache",
        VH_JOB_OUTPUT_DIR=tmp_path / "outputs/jobs",
        VH_WRITE_TRACE=True,
    )
    result = HighlightOrchestrator(settings).run(
        DetectionTask(
            video_path=video_path,
            video_id="vid_demo",
            job_id="job_demo",
            language="zh",
        )
    )

    job_files = {
        path.relative_to(settings.job_output_dir / "job_demo").as_posix()
        for path in (settings.job_output_dir / "job_demo").rglob("*")
        if path.is_file()
    }
    cache_files = {
        path.relative_to(settings.media_cache_dir).as_posix()
        for path in settings.media_cache_dir.rglob("*")
        if path.is_file()
    }
    assert job_files == {
        "result.json",
        "trace.json",
        f"clips/{result.highlights[0].highlight_id}.mp4",
    }
    assert {Path(path).name for path in cache_files} == {
        "audio.wav",
        "frame_000001.jpg",
        "preprocess.json",
        "scene_0001.json",
    }
    assert result.highlights[0].clip_url.startswith("clips/")
    trace = json.loads((settings.job_output_dir / "job_demo" / "trace.json").read_text())
    assert "local_candidates" in trace
    assert trace["reasoning"]["decisions"][0]["scene"]["scene_id"] == "scene_0001"
    assert len(trace["reasoning"]["decisions"][0]["votes"]) == 2
    assert trace["pipeline_version"] == "0.21.0"
    cache_only = True

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("expensive preprocessing should be loaded from cache")

    monkeypatch.setattr(pipeline_module, "detect_scenes", fail_if_recomputed)
    monkeypatch.setattr(HighlightOrchestrator, "_transcribe", fail_if_recomputed)
    monkeypatch.setattr(pipeline_module, "extract_subtitle_segments", fail_if_recomputed)
    monkeypatch.setattr(pipeline_module, "extract_audio_events", fail_if_recomputed)
    monkeypatch.setattr(pipeline_module, "score_semantic_transitions", fail_if_recomputed)
    cached_result = HighlightOrchestrator(settings).run(
        DetectionTask(
            video_path=video_path,
            video_id="vid_demo",
            job_id="job_cached",
            language="zh",
        )
    )
    cached_trace = json.loads((settings.job_output_dir / "job_cached" / "trace.json").read_text())
    assert cached_result.highlights[0].highlight_id == result.highlights[0].highlight_id
    assert cached_trace["stats"]["scene_map_calls"] == 0


def test_listwise_compression_keeps_highest_scoring_distinct_clips() -> None:
    from vh_agent.models import RankedHighlight
    from vh_agent.pipeline import _budgeted_ranking, _output_limit

    def highlight(identifier: str, start: float, judge_score: float) -> RankedHighlight:
        return RankedHighlight(
            highlight_id=identifier,
            start_sec=start,
            end_sec=start + 8,
            score=judge_score,
            local_score=0.7,
            judge_score=judge_score,
            highlight_type="conflict",
            description=identifier,
            reason="独立状态变化",
            confidence=0.9,
        )

    weak = highlight("hl_weak", 2, 0.68)
    mid = highlight("hl_mid", 20, 0.70)
    strong = highlight("hl_strong", 40, 0.78)
    assert _output_limit(54.0, 90.0, 12) == 2
    kept = _budgeted_ranking([weak, mid], 2)
    assert kept.selected_highlight_ids == ["hl_weak", "hl_mid"]
    compressed = _budgeted_ranking([weak, mid, strong], 2)
    assert compressed.selected_highlight_ids == ["hl_strong", "hl_mid"]
    assert compressed.rationale == "top distinct highlights by judge score"


def test_residual_output_overlap_is_split_at_the_midpoint() -> None:
    from vh_agent.models import RankedHighlight
    from vh_agent.pipeline import _resolve_output_overlaps

    def highlight(identifier: str, start: float, end: float) -> RankedHighlight:
        return RankedHighlight(
            highlight_id=identifier,
            start_sec=start,
            end_sec=end,
            score=0.8,
            local_score=0.7,
            judge_score=0.8,
            highlight_type="conflict",
            description=identifier,
            reason="独立事件",
            confidence=0.9,
        )

    resolved = _resolve_output_overlaps(
        [highlight("first", 10, 30), highlight("second", 24, 40)]
    )
    assert len(resolved) == 2
    assert resolved[0].end_sec == 27.0
    assert resolved[1].start_sec == 27.0


def test_public_result_schema_matches_frontend_contract() -> None:
    schema = DetectionResult.model_json_schema()
    highlight = schema["$defs"]["Highlight"]["properties"]

    assert set(schema["properties"]) == {"schema_version", "job_id", "video", "highlights"}
    assert set(highlight) == {
        "highlight_id",
        "start_sec",
        "end_sec",
        "score",
        "highlight_type",
        "description",
        "reason",
        "clip_url",
        "review_status",
    }


def test_scene_map_cache_survives_a_later_scene_failure(tmp_path) -> None:
    import pytest

    from vh_agent.models import SceneCard

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")
    frame = FrameSample(timestamp_sec=1.0, path=frame_path)
    scenes = [
        SceneCard(
            scene_id="scene_0001",
            start_sec=0,
            end_sec=10,
            frame_samples=[frame],
        ),
        SceneCard(
            scene_id="scene_0002",
            start_sec=10,
            end_sec=20,
            frame_samples=[frame],
        ),
    ]
    video = VideoInfo(
        path=tmp_path / "video.mp4",
        duration_sec=20,
        width=480,
        height=852,
        fps=25,
        has_audio=True,
        title="cache test",
        language="zh",
    )
    settings = Settings(
        VH_MODEL_CACHE_DIR=tmp_path / "models",
        VH_MEDIA_CACHE_DIR=tmp_path / "media",
        VH_JOB_OUTPUT_DIR=tmp_path / "jobs",
        VH_MAP_WORKERS=1,
    )
    orchestrator = HighlightOrchestrator(settings)
    cache_dir = tmp_path / "scene_map"

    class FlakyReasoner:
        def map_scene(self, _video, scene):
            if scene.scene_id == "scene_0002":
                raise RuntimeError("late failure")
            return SceneNarrative(scene_id=scene.scene_id, action="mapped first")

    with pytest.raises(RuntimeError, match="late failure"):
        orchestrator._map_scenes(FlakyReasoner(), video, scenes, cache_dir)

    assert (cache_dir / "scene_0001.json").is_file()
    assert not (cache_dir / "scene_0002.json").exists()

    class StableReasoner:
        def __init__(self):
            self.calls = []

        def map_scene(self, _video, scene):
            self.calls.append(scene.scene_id)
            return SceneNarrative(scene_id=scene.scene_id, action="mapped second")

    stable = StableReasoner()
    mapped, calls = orchestrator._map_scenes(stable, video, scenes, cache_dir)
    assert calls == 1
    assert stable.calls == ["scene_0002"]
    assert [scene.action for scene in mapped] == ["mapped first", "mapped second"]
