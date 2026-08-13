from pathlib import Path

from vh_agent import pipeline as pipeline_module
from vh_agent.config import Settings
from vh_agent.models import (
    AudioEvent,
    CandidateWindow,
    DetectionResult,
    DetectionTask,
    EventCard,
    FrameSample,
    JudgeDecision,
    SceneSegment,
    TranscriptSegment,
    VideoInfo,
)
from vh_agent.pipeline import HighlightOrchestrator


def test_job_writes_only_result_clips_and_optional_trace(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"video")

    def fake_frames(_video: Path, output_dir: Path, *_args) -> list[FrameSample]:
        output_dir.mkdir(parents=True)
        frame_path = output_dir / "frame_000001.jpg"
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

    class FakeReasoner:
        def __init__(self, _settings) -> None:
            pass

        def map_chapter(self, _video, _chapter) -> list[EventCard]:
            return [
                EventCard(
                    start_sec=2,
                    end_sec=14,
                    event_type=["reveal"],
                    salience=0.9,
                    uncertainty=0.2,
                    evidence=["[2.00-14.00s ASR] 原来是你"],
                )
            ]

        def judge(
            self,
            _video,
            candidate,
            _event,
            _memory,
            _context_before,
            _context_core,
            _context_after,
        ):
            return JudgeDecision(
                hypothesis_supported=True,
                is_highlight=True,
                score=0.9,
                highlight_type="reveal",
                description="身份揭露",
                reason="新证据改变此前认知",
                confidence=0.9,
                start_sec=candidate.start_sec,
                end_sec=candidate.end_sec,
            )

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
    monkeypatch.setattr(pipeline_module, "attach_storyboards", lambda *_args: None)
    monkeypatch.setattr(pipeline_module, "SiliconFlowReasoner", FakeReasoner)
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
    assert {Path(path).name for path in cache_files} == {"audio.wav", "frame_000001.jpg"}
    assert result.highlights[0].clip_url.startswith("clips/")


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
