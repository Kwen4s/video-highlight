import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIError, APIStatusError, AsyncOpenAI
from pydantic import ValidationError
from vh_agent.models import TranscriptSegment
from vh_agent.preprocessing.media import video_fingerprint
from vh_agent.runtime.evidence import EvidenceStore, VideoPage

from vh_data.config import Settings
from vh_data.dialogue import DialogueLoader
from vh_data.exporter import export
from vh_data.labeler import Labeler, PageNeedsSplit, _run, run
from vh_data.models import Annotation, Label, PageLabels, VideoPredictions, digest
from vh_data.quality import overlapping_highlights, reconcile
from vh_data.store import ConflictError, Store


def group_for(split):
    for index in range(100):
        name = f"drama-{index}"
        bucket = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 10
        if ("train" if bucket < 8 else "val" if bucket == 8 else "test") == split:
            return name
    raise AssertionError("fixture needs a matching drama")


def ingest_video(store, tmp_path, video_id="v1", split="train", drama_id=None):
    video = tmp_path / f"{video_id}.mp4"
    video.write_bytes(video_id.encode())
    manifest = tmp_path / f"{video_id}.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "drama_id": drama_id or group_for(split),
                "path": video.name,
                "duration_sec": 40,
                "language": "zh",
            }
        )
    )
    store.ingest([manifest])
    return store.get(video_id), manifest


def annotation():
    return Annotation(
        duration_sec=40,
        segments=[
            Label(start_sec=10, end_sec=20, kind="positive", reason="秘密揭开"),
            Label(start_sec=0, end_sec=5, kind="negative", reason="重复铺垫"),
        ],
    )


def approve(store, video_id):
    store.save(
        video_id,
        annotation(),
        "model_reviewed",
        {"model": "teacher"},
        expected_label_id=store.get(video_id)["label_id"],
    )


def test_consensus_keeps_unknown_and_never_makes_a_disputed_event_negative():
    first = annotation()
    second = Annotation(
        duration_sec=40,
        segments=[
            Label(start_sec=11, end_sec=19, kind="positive", reason="秘密被发现"),
            Label(start_sec=25, end_sec=30, kind="positive", reason="另一事件"),
            Label(start_sec=1, end_sec=5, kind="negative", reason="铺垫"),
        ],
    )
    merged, issues = reconcile(first.segments, second.segments, first.duration_sec, 0.5)
    assert [(s.start_sec, s.end_sec) for s in merged.segments if s.kind == "positive"] == [(10, 20)]
    assert [(s.start_sec, s.end_sec) for s in merged.segments if s.kind == "negative"] == [(1, 5)]
    assert [(s.start_sec, s.end_sec) for s in merged.segments if s.kind == "uncertain"] == [
        (25, 30)
    ]
    assert issues
    assert not any(s.start_sec <= 35 < s.end_sec for s in merged.segments)


@pytest.mark.parametrize(
    "segments",
    [
        [{"start_sec": 0, "end_sec": 50, "kind": "positive", "reason": "越界"}],
        [{"start_sec": float("nan"), "end_sec": 20, "kind": "positive", "reason": "无效"}],
        [
            {"start_sec": 10, "end_sec": 20, "kind": "positive", "reason": "事件"},
            {"start_sec": 15, "end_sec": 25, "kind": "negative", "reason": "矛盾"},
        ],
    ],
)
def test_bad_labels_are_rejected(segments):
    with pytest.raises(ValidationError):
        Annotation(duration_sec=40, segments=segments)


def test_ingest_is_idempotent_and_preserves_drama_groups(tmp_path):
    store = Store(tmp_path / "data")
    first, manifest = ingest_video(store, tmp_path, split="val")
    assert store.ingest([manifest]) == {"added": 0, "duplicates": 1}
    second, _ = ingest_video(store, tmp_path, "v2", drama_id=first["drama_id"])
    assert first["split"] == second["split"] == "val"
    Path(first["path"]).write_bytes(b"changed")
    with pytest.raises(ConflictError, match="改变"):
        store.ingest([manifest])


def test_versions_and_stale_worker_cannot_overwrite_a_new_review(tmp_path):
    store = Store(tmp_path / "data")
    video, _ = ingest_video(store, tmp_path)
    assert store.claim()["video_id"] == video["video_id"]
    approve(store, "v1")
    first_version = store.get("v1")["label_id"]
    approve(store, "v1")
    assert store.get("v1")["status"] == "ready"
    with pytest.raises(ConflictError, match="更新"):
        store.save("v1", annotation(), "model_reviewed", {}, expected_label_id=first_version)
    store.fail("v1", "stale worker")
    assert store.get("v1")["status"] == "ready"
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 2


def test_failed_and_interrupted_tasks_have_explicit_resume(tmp_path):
    store = Store(tmp_path / "data")
    ingest_video(store, tmp_path)
    store.claim()
    store.recover(False)
    assert store.get("v1")["status"] == "queued"
    store.claim()
    store.fail("v1", "network")
    store.recover(False)
    assert store.claim() is None
    store.recover(True)
    assert store.claim()["video_id"] == "v1"


def test_claim_can_reproduce_a_selected_video_without_consuming_other_work(tmp_path):
    store = Store(tmp_path / "data")
    ingest_video(store, tmp_path, "first")
    ingest_video(store, tmp_path, "selected")
    assert store.claim(video_ids=["selected"])["video_id"] == "selected"
    assert store.get("first")["status"] == "queued"
    assert store.claim(video_ids=["selected"]) is None
    store.fail("selected", "network")
    store.claim(video_ids=["first"])
    store.fail("first", "network")
    store.recover(True, video_ids=["selected"])
    assert store.get("first")["status"] == "failed"
    assert store.get("selected")["status"] == "queued"


def test_unresolved_data_stays_out_of_training_and_can_be_explicitly_requeued(tmp_path):
    store = Store(tmp_path / "data")
    ingest_video(store, tmp_path)
    store.save(
        "v1",
        Annotation(
            duration_sec=40,
            segments=[Label(start_sec=0, end_sec=40, kind="uncertain", reason="证据不足")],
        ),
        "model_reviewed",
        {},
        expected_label_id=None,
    )
    version = store.get("v1")["label_id"]
    assert store.get("v1")["status"] == "unresolved"
    with pytest.raises(ValueError, match="复核通过"):
        export(store, tmp_path / "unresolved")
    assert store.requeue(["v1"])["queued"] == 1
    assert store.get("v1")["label_id"] == version
    store.claim()
    with pytest.raises(ConflictError, match="正在标注"):
        store.requeue(["v1"])


def test_worker_persists_model_review_and_retries_only_when_requested(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    ingest_video(store, tmp_path)
    attempts = []

    class Teacher:
        def __init__(self, settings):
            pass

        def close(self):
            pass

        async def label(self, video):
            attempts.append(video["video_id"])
            if len(attempts) == 1:
                raise ValueError("标注越界")
            return annotation(), {"unresolved_segments": 0}

    monkeypatch.setattr("vh_data.labeler.Labeler", Teacher)
    settings = Settings(_env_file=None, root=store.root)
    assert run(store, settings, 1, None, False) == {"completed": 0, "failed": 1}
    assert store.get("v1")["status"] == "failed"
    assert run(store, settings, 1, None, False) == {"completed": 0, "failed": 0}
    assert run(store, settings, 1, None, True) == {"completed": 1, "failed": 0}
    assert store.get("v1")["source"] == "model_reviewed"
    assert export(store, tmp_path / "worker-snapshot")["videos"] == 1


@pytest.mark.parametrize("status,attempted", [(401, 1), (503, 3)])
def test_bad_gateway_stops_batch_without_consuming_whole_queue(
    tmp_path, monkeypatch, status, attempted
):
    store = Store(tmp_path / "data")
    for index in range(5):
        ingest_video(store, tmp_path, f"v{index}")

    class Teacher:
        def __init__(self, settings):
            pass

        def close(self):
            pass

        async def label(self, video):
            response = httpx.Response(status, request=httpx.Request("POST", "https://teacher.test"))
            raise APIStatusError("gateway failure", response=response, body=None)

    monkeypatch.setattr("vh_data.labeler.Labeler", Teacher)
    assert run(store, Settings(_env_file=None, workers=1), 5, None, False) == {
        "completed": 0,
        "failed": attempted,
    }
    assert store.stats()["status"] == {"failed": attempted, "queued": 5 - attempted}


@pytest.mark.parametrize("duration", [0.5, 65.5, 1583.5])
def test_short_and_long_videos_keep_full_coverage_after_split_and_resume(
    tmp_path, monkeypatch, duration
):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s=32x32:r=2:d={duration}",
            "-c:v",
            "libx264",
            str(video),
        ],
        check=True,
    )
    settings = Settings(
        _env_file=None,
        api_key="test",
        model="teacher",
        root=tmp_path / "data",
        media_cache=tmp_path / "media",
        preprocess_cache=tmp_path / "preprocess",
    )
    labeler = Labeler(settings)
    seen = {1: [], 2: []}
    split = False

    async def call(media, page, source_hash, pass_id, story, dialogue, notes=None):
        nonlocal split
        if duration > 60 and not split and pass_id == 1 and page.core_start_sec == 30:
            split = True
            raise PageNeedsSplit("HTTP 413")
        seen[pass_id].append((page.core_start_sec, page.core_end_sec))
        result = PageLabels(
            story_so_far="普通片段",
            segments=[
                Label(
                    start_sec=page.core_start_sec,
                    end_sec=page.core_end_sec,
                    kind="negative",
                    reason="没有新看点",
                )
            ],
        )
        labeler._validate_page(
            result, page.read_start_sec, page.read_end_sec, media.video_info.duration_sec
        )
        return result, "offline-test"

    monkeypatch.setattr(labeler, "_call", call)
    source = {"path": str(video), "sha256": digest(video), "language": "zh"}
    annotation, _ = asyncio.run(labeler.label(source))
    for ranges in seen.values():
        assert ranges[0][0] == 0
        assert ranges[-1][1] == pytest.approx(duration)
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:], strict=False))
        assert sum(end - start for start, end in ranges) == pytest.approx(duration)
    assert annotation.duration_sec == pytest.approx(duration)
    original = seen[1].copy()
    seen = {1: [], 2: []}
    asyncio.run(labeler.label(source))
    assert seen[1] == seen[2] == original
    if duration > 60:
        assert labeler._plan_path(source["sha256"]).is_file()
        assert (30, 60) not in seen[1]
    media = EvidenceStore(video, settings.media_cache)
    assert media.frames([max(0, duration - 0.25)], width=32)
    labeler.close()


def test_selection_review_keeps_long_video_requests_local(tmp_path, monkeypatch):
    labeler = Labeler(Settings(_env_file=None, api_key="test", model="teacher"))
    annotation = Annotation(
        duration_sec=1625,
        segments=[
            Label(
                start_sec=index * 30,
                end_sec=index * 30 + 35,
                kind="positive",
                reason="相邻看点有重叠前后文",
            )
            for index in range(54)
        ],
    )
    ranges = []

    async def call(media, page, source_hash, pass_id, story, dialogue, notes=None):
        ranges.append((page.core_start_sec, page.core_end_sec))
        return PageLabels(
            story_so_far="已复核",
            segments=[
                Label(
                    start_sec=page.core_start_sec + offset,
                    end_sec=page.core_start_sec + offset + 15,
                    kind="positive",
                    reason="保留这一处看点",
                )
                for offset in (0, 30)
            ],
        ), "offline-test"

    monkeypatch.setattr(labeler, "_call", call)
    result, _ = asyncio.run(
        labeler._review(
            SimpleNamespace(overlap_sec=5, pages=[]),
            "test",
            annotation,
            list(annotation.segments),
            [],
            [],
            {0: ""},
            [],
            [],
        )
    )
    assert len(result.segments) == 54
    assert not overlapping_highlights(result.segments)
    assert ranges and max(end - start for start, end in ranges) < 100
    labeler.close()


def test_only_model_reviewed_labels_are_exported_and_snapshots_are_immutable(tmp_path):
    store = Store(tmp_path / "data")
    for split in ("train", "val", "test"):
        ingest_video(store, tmp_path, split, split)
    with pytest.raises(ValueError, match="复核通过"):
        export(store, tmp_path / "empty")
    store.save("train", annotation(), "draft", {}, expected_label_id=None)
    assert store.stats()["ready_splits"] == {"train": 0, "val": 0, "test": 0}
    with pytest.raises(ValueError, match="复核通过"):
        export(store, tmp_path / "unreviewed")
    for split in ("train", "val", "test"):
        approve(store, split)
    report = export(store, tmp_path / "frozen")
    assert store.stats()["ready_splits"] == {"train": 1, "val": 1, "test": 1}
    assert store.stats()["labels"] == {"model_reviewed": 3}
    assert report["splits"] == {"train": 1, "val": 1, "test": 1}
    assert report["dramas"] == 3
    assert report["labels"] == {"positive": 3, "negative": 3, "uncertain": 0}
    assert report["duration_sec"] == {
        "total": 120.0,
        "supervised": 45.0,
        "highlight": 30.0,
        "unknown": 75.0,
        "longest_highlight": 10.0,
    }
    assert report["highlight_ratio"] == 0.25
    assert store.stats()["quality"]["highlight_ratio"] == 0.25
    loaded = [
        json.loads(line)
        for line in (tmp_path / "frozen/annotations.jsonl").read_text().splitlines()
    ]
    assert len(loaded) == 3
    assert all(v["negative_intervals"] == [{"start_sec": 0, "end_sec": 5}] for v in loaded)
    assert all(v["label_source"] == "model_reviewed" for v in loaded)
    assert digest(tmp_path / "frozen/annotations.jsonl") == report["annotations_sha256"]
    with pytest.raises(FileExistsError):
        export(store, tmp_path / "frozen")
    Path(store.get("train")["path"]).write_bytes(b"changed")
    with pytest.raises(ConflictError):
        export(store, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_overlapping_highlights_cannot_enter_training_snapshot(tmp_path):
    store = Store(tmp_path / "data")
    ingest_video(store, tmp_path)
    duplicate = Annotation(
        duration_sec=40,
        segments=[
            Label(start_sec=10, end_sec=20, kind="positive", reason="同一次反击"),
            Label(start_sec=11, end_sec=21, kind="positive", reason="相邻页重复标注"),
        ],
    )
    store.save("v1", duplicate, "model_reviewed", {}, expected_label_id=None)
    with pytest.raises(ConflictError, match="重叠"):
        export(store, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def test_feedback_requeues_training_videos_for_model_review_without_self_labeling(tmp_path):
    store = Store(tmp_path / "data")
    ingest_video(store, tmp_path, "train", "train")
    ingest_video(store, tmp_path, "test", "test")
    approve(store, "train")
    version = store.get("train")["label_id"]
    prediction = VideoPredictions(
        video_id="train",
        segments=[
            {"start_sec": 30, "end_sec": 35, "score": 0.8},
        ],
    )
    assert store.feedback([prediction], "student-v1", 0.5)["queued_for_review"] == 1
    current = store.get("train")
    assert current["status"] == "review" and current["priority"] == 2
    assert current["label_id"] == version
    assert current["predictions"][0]["model_id"] == "student-v1"
    assert store.feedback([prediction], "student-v1", 0.5)["unchanged"] == 1
    assert store.claim()["video_id"] == "train"
    with pytest.raises(ConflictError, match="正在标注"):
        store.feedback([prediction], "student-v2", 0.5)
    approve(store, "train")
    assert store.get("train")["priority"] == 0
    assert store.feedback([prediction], "student-v2", 0.5)["queued_for_review"] == 1
    changed = prediction.model_copy(update={"segments": []})
    store.feedback([changed], "student-v1", 0.5)
    assert store.get("train")["predictions"][0]["model_id"] == "student-v1"
    with pytest.raises(ValueError, match="验证与测试"):
        store.feedback([VideoPredictions(video_id="test", segments=[])], "student-v1", 0.5)
    assert store.get("test")["status"] == "queued"


def test_new_dialogue_uses_aligned_audio_and_tracks_recognition_profile(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-with-offset")
    settings = Settings(_env_file=None, root=tmp_path / "data", preprocess_cache=tmp_path / "old")
    video = {"path": str(source), "sha256": digest(source), "language": "zh"}
    # An old preprocessing record cannot be reused when the source has a PTS offset.
    previous = settings.preprocess_cache / video_fingerprint(source) / "preprocess.json"
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps({"transcript": []}))
    normalized = tmp_path / "normalized.mp4"
    normalized.write_bytes(b"normalized")
    calls = []

    class Media:
        source_start_time_sec = 2
        video_info = SimpleNamespace(duration_sec=10, has_audio=True)
        output_dir = tmp_path

        def inspect(self, start, end):
            assert (start, end) == (0, 10)
            return SimpleNamespace(path=normalized)

    def extract(path, target):
        assert path == normalized
        target.write_bytes(b"aligned-audio")
        return target

    class Transcriber:
        def __init__(self, model, device, compute_type):
            calls.append((model, device, compute_type))

        def transcribe(self, audio, language, *, beam_size):
            assert audio.read_bytes() == b"aligned-audio"
            assert language == "zh" and beam_size == 5
            return [TranscriptSegment(start_sec=1, end_sec=12, text="台词", source="asr")]

    monkeypatch.setattr("vh_data.dialogue.extract_audio", extract)
    monkeypatch.setattr("vh_data.dialogue.FasterWhisperTranscriber", Transcriber)
    loader = DialogueLoader(settings)
    segments, reference = loader.load(video, Media())
    assert segments == [{"start_sec": 1.0, "end_sec": 10, "text": "台词", "source": "asr"}]
    assert json.loads((settings.root / reference).read_text())["origin"] == "source_aligned_whisper"
    assert loader.load(video, Media()) == (segments, reference)
    assert len(calls) == 1
    changed = DialogueLoader(settings.model_copy(update={"asr_compute_type": "float32"}))
    _, new_reference = changed.load(video, Media())
    assert new_reference != reference and len(calls) == 2


@pytest.mark.parametrize(
    "case",
    [
        "agree",
        "agreed_ordinary",
        "neighbor_owned",
        "disagree",
        "contradiction",
        "resize",
        "feedback",
        "incomplete",
        "empty",
        "stream_error",
        "timeout",
        "long_preference",
        "page_overlap",
        "transient",
        "rate_limit",
        "stream_retry",
        "oversize",
    ],
)
def test_video_annotation_review_and_resume_use_real_sdk_requests(tmp_path, monkeypatch, case):
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=128x128:r=10:d=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
    )
    settings = Settings(
        _env_file=None,
        api_key="test",
        model="video-teacher",
        root=tmp_path / "data",
        media_cache=tmp_path / "media",
        page_sec=4,
        preprocess_cache=tmp_path / "preprocess",
        request_timeout_sec=0.1 if case == "timeout" else 300,
        request_attempts=3 if case in {"transient", "rate_limit", "stream_retry", "empty"} else 1,
        preferred_highlight_sec=1.5 if case == "long_preference" else 15,
    )
    cached = settings.preprocess_cache / video_fingerprint(video) / "preprocess.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(
        json.dumps(
            {
                "transcript": [
                    {"start_sec": 0.25, "end_sec": 7.5, "text": "对白内容", "source": "asr"}
                ]
            }
        )
    )
    labeler = Labeler(settings)
    call = labeler._call

    async def resized_call(media, page, source_hash, pass_id, story, dialogue, notes=None):
        if (
            case == "resize"
            and pass_id == 2
            and page.core_start_sec == 0
            and page.core_end_sec == 4
        ):
            raise PageNeedsSplit
        return await call(media, page, source_hash, pass_id, story, dialogue, notes)

    monkeypatch.setattr(labeler, "_call", resized_call)
    contexts = []

    async def no_delay(_):
        pass

    if case != "timeout":
        monkeypatch.setattr("vh_data.labeler.asyncio.sleep", no_delay)

    blind_calls = []

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b": heartbeat\n\n"
                await asyncio.sleep(0.02)

    def respond(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert body["stream"] and not body["store"]
        assert body["reasoning"]["effort"] in {"high", "low"}
        assert body["max_output_tokens"] == 32768
        assert "{preferred_highlight_sec}" not in body["instructions"]
        assert f"{settings.preferred_highlight_sec:g} 秒以内" in body["instructions"]
        parts = body["input"][0]["content"]
        context = json.loads(parts[0]["text"])
        contexts.append(context)
        assert parts[2]["image_url"].startswith("data:image/jpeg;base64,")
        assert all(t["source"] == "asr" for t in context["对白与字幕"])
        if case == "resize" and context["负责范围"] == [0.0, 2.0] and "待核对意见" not in context:
            assert context["对白与字幕"] == []
        assert all(
            context["前后文范围"][0] <= t["start_sec"] < t["end_sec"] <= context["前后文范围"][1]
            for t in context["对白与字幕"]
        )
        assert not any("data:video" in json.dumps(p) for p in parts)
        assert context["原片有音轨"]
        assert body["text"]["format"]["strict"]
        if case == "timeout":
            return httpx.Response(
                200, stream=SlowBody(), headers={"Content-Type": "text/event-stream"}
            )
        if case == "rate_limit" and len(contexts) == 1:
            return httpx.Response(
                429, json={"error": {"message": "slow down", "type": "rate_limit_error"}}
            )
        first_page = context["负责范围"][0] == 0
        reviewing = "待核对意见" in context
        if not reviewing:
            blind_calls.append(context)
        second_pass = (len(blind_calls) - 1) % 4 >= 2
        if case == "oversize" and len(blind_calls) == 3 and not reviewing:
            return httpx.Response(
                413, json={"error": {"message": "too large", "type": "invalid_request_error"}}
            )
        reject_event = (case == "disagree" and (second_pass or reviewing)) or (
            case == "agreed_ordinary" and reviewing
        )
        if case == "neighbor_owned":
            segments = [Label(start_sec=4.5, end_sec=6.5, kind="positive", reason="反击奏效")]
            if first_page:
                segments.append(Label(start_sec=1, end_sec=3, kind="negative", reason="普通交谈"))
        elif case == "page_overlap":
            start, end = (2.8, 5.0) if first_page else (3.0, 5.2)
            if reviewing and context["待核对意见"].get("已有复核"):
                start, end = 3.0, 5.0
            segments = [Label(start_sec=start, end_sec=end, kind="positive", reason="同一次反击")]
        elif first_page:
            kind = "negative" if reject_event else "positive"
            start, end = (0.5, 1.5) if context["负责范围"][1] == 2 else (1, 3)
            segments = [Label(start_sec=start, end_sec=end, kind=kind, reason="核对画面中的事情")]
        elif context["负责范围"][0] == 2:
            segments = [Label(start_sec=2.5, end_sec=3.5, kind="negative", reason="普通片段")]
        else:
            segments = [Label(start_sec=5, end_sec=7, kind="negative", reason="普通片段")]
        if case == "contradiction" and first_page and second_pass and not reviewing:
            segments.append(Label(start_sec=2, end_sec=3, kind="negative", reason="矛盾意见"))
        result = PageLabels(story_so_far="已经发生的剧情", segments=segments)
        response = {
            "id": "annotation",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "video-teacher",
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": result.model_dump_json(), "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }
        if case == "incomplete":
            response["status"] = "incomplete"
            response["incomplete_details"] = {"reason": "max_output_tokens"}
            response["output"] = []
        elif case == "empty" or (case == "transient" and len(contexts) == 1):
            response["output"][0]["content"][0]["text"] = " "
        events = [
            {
                "type": "response.created",
                "response": {**response, "output": [], "status": "in_progress"},
                "sequence_number": 0,
            },
            {
                "type": "response.incomplete" if case == "incomplete" else "response.completed",
                "response": response,
                "sequence_number": 1,
            },
        ]
        if case == "stream_error" or (case == "stream_retry" and len(contexts) == 1):
            events = [
                {
                    "type": "error",
                    "error": {
                        "code": "upstream_failed",
                        "type": "server_error",
                        "message": "failure",
                    },
                }
            ]
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, content=data, headers={"Content-Type": "text/event-stream"})

    monkeypatch.setattr(
        "vh_data.labeler.AsyncOpenAI",
        lambda **kwargs: AsyncOpenAI(
            **{**kwargs, "base_url": "https://teacher.test/v1", "max_retries": 0},
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )
    source = {"path": str(video), "sha256": digest(video), "split": "train", "language": "zh"}
    if case in {"incomplete", "empty", "stream_error", "timeout"}:
        exception = (
            TimeoutError
            if case == "timeout"
            else (APIError if case == "stream_error" else ValueError)
        )
        with pytest.raises(exception):
            asyncio.run(labeler.label(source))
        failures = list((settings.root / "failures").glob("*.json"))
        assert len(failures) == settings.request_attempts
        assert len(contexts) == settings.request_attempts
        record = json.loads(failures[0].read_text())
        if case in {"stream_error", "timeout"}:
            assert record["status"] == "request_failed"
            if case == "stream_error":
                assert record["error_code"] == "upstream_failed"
            else:
                assert record["error_class"] == "TimeoutError"
                assert record["elapsed_sec"] < 1
        else:
            assert record["usage"]["total_tokens"] == 120
            assert record["status"] == ("incomplete" if case == "incomplete" else "completed")
        assert not list((settings.root / "readings").glob("*.json"))
        labeler.close()
        return
    if case == "feedback":
        source["predictions"] = [
            {"model_id": "student-v1", "segments": [{"start_sec": 5, "end_sec": 7, "score": 0.9}]}
        ]
    result, provenance = asyncio.run(labeler.label(source))
    expected_calls = {
        "agree": 5,
        "agreed_ordinary": 5,
        "neighbor_owned": 5,
        "disagree": 5,
        "contradiction": 5,
        "resize": 7,
        "feedback": 6,
        "long_preference": 5,
        "page_overlap": 7,
        "transient": 6,
        "rate_limit": 6,
        "stream_retry": 6,
        "oversize": 8,
    }[case]
    blind_count = 5 if case in {"resize", "oversize"} else 4
    request_count = expected_calls
    if case in {"transient", "rate_limit", "stream_retry", "oversize"}:
        expected_calls -= 1
    assert len(contexts) == request_count
    assert len(provenance["calls"]) == expected_calls
    assert len(provenance["review_calls"]) == expected_calls - blind_count
    initial_contexts = []
    for reference in provenance["calls"]:
        record = json.loads((settings.root / reference).read_text())
        if record["input"]["pass"] in {1, 2} and record["input"]["page"]["core_start_sec"] == 0:
            initial_contexts.append(record["input"]["context"])
    assert len(initial_contexts) == 2
    assert all(c["之前的剧情"] == "" for c in initial_contexts)
    assert not any("待核对意见" in c for c in blind_calls)
    expected_segments = 1 if case == "page_overlap" else 3 if case in {"resize", "oversize"} else 2
    assert len(result.segments) == expected_segments
    if case in {"disagree", "agreed_ordinary"}:
        assert all(s.kind == "negative" for s in result.segments)
    else:
        assert any(s.kind == "positive" for s in result.segments)
    for reference in provenance["calls"]:
        assert json.loads((settings.root / reference).read_text())["usage"]["total_tokens"] == 120
    asyncio.run(labeler.label(source))
    assert len(contexts) == request_count + (3 if case in {"resize", "oversize"} else 0)
    cached_count = len(contexts)
    asyncio.run(labeler.label(source))
    assert len(contexts) == cached_count
    labeler.settings = settings.model_copy(update={"reasoning_effort": "low"})
    asyncio.run(labeler.label(source))
    assert len(contexts) == cached_count + (7 if case in {"resize", "oversize"} else expected_calls)
    if case in {"transient", "rate_limit", "stream_retry", "oversize"}:
        failures = list((settings.root / "failures").glob("*.json"))
        assert len(failures) == 1
        assert json.loads(failures[0].read_text())["attempt"] == 1
    labeler.close()


def test_workers_are_bounded_and_claim_each_video_once(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    for index in range(6):
        ingest_video(store, tmp_path, f"v{index}")
    active = maximum = 0
    claimed = []
    barrier = asyncio.Barrier(2)

    class Teacher:
        def __init__(self, settings):
            self.video = None

        def close(self):
            pass

        async def label(self, video):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            claimed.append(video["video_id"])
            assert self.video is None
            self.video = video["video_id"]
            assert store.get(video["video_id"])["status"] == "labeling"
            await asyncio.wait_for(barrier.wait(), timeout=5)
            await asyncio.sleep(0.01)
            active -= 1
            return annotation(), {"unresolved_segments": 0}

    monkeypatch.setattr("vh_data.labeler.Labeler", Teacher)
    result = run(store, Settings(_env_file=None, workers=2), 4, None, False)
    assert result == {"completed": 4, "failed": 0}
    assert maximum == 2 and len(set(claimed)) == 4
    assert store.stats()["status"] == {"ready": 4, "queued": 2}


def test_page_edges_keep_valid_candidates_and_request_more_context():
    response = PageLabels(
        story_so_far="",
        segments=[
            Label(start_sec=5, end_sec=8, kind="positive", reason="精彩反击"),
            Label(start_sec=29, end_sec=34, kind="positive", reason="邻页所属的完整看点"),
            Label(start_sec=32, end_sec=35, kind="positive", reason="还没看完"),
        ],
    )
    Labeler._validate_page(response, 0, 35, 100)
    assert [s.kind for s in response.segments] == ["positive", "positive", "uncertain"]
    assert response.segments[1].start_sec == 29 and response.segments[1].end_sec == 34
    outside = PageLabels(
        story_so_far="",
        segments=[
            Label(start_sec=34, end_sec=36, kind="positive", reason="真实越界"),
        ],
    )
    with pytest.raises(ValueError, match="实际观看"):
        Labeler._validate_page(outside, 0, 35, 100)


def test_redundant_tail_keeps_full_video_and_the_same_evidence(tmp_path):
    pages = [
        VideoPage(
            page_id=str(index),
            core_start_sec=start,
            core_end_sec=end,
            read_start_sec=max(0, start - 5),
            read_end_sec=min(60.5, end + 5),
        )
        for index, (start, end) in enumerate([(0, 30), (30, 60), (60, 60.5)])
    ]
    media = SimpleNamespace(pages=pages, video_info=SimpleNamespace(duration_sec=60.5))
    media.restore_pages = lambda updated: setattr(media, "pages", updated)
    Labeler._compact_tail(media)
    assert [(p.core_start_sec, p.core_end_sec) for p in media.pages] == [(0, 30), (30, 60.5)]
    assert media.pages[-1].read_start_sec == pages[1].read_start_sec
    assert media.pages[-1].read_end_sec == pages[1].read_end_sec


def test_consensus_does_not_invent_a_longer_boundary():
    first = [Label(start_sec=10, end_sec=18, kind="positive", reason="精彩动作")]
    second = [Label(start_sec=12, end_sec=20, kind="positive", reason="同一精彩动作")]
    result, _ = reconcile(first, second, 40, 0.5)
    assert [(s.start_sec, s.end_sec) for s in result.segments] == [(10, 18)]


def test_gateway_stop_preserves_an_inflight_success(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    for index in range(4):
        ingest_video(store, tmp_path, f"v{index}")
    failed_saved = asyncio.Event()
    fail = store.fail

    def record_failure(video_id, message):
        fail(video_id, message)
        failed_saved.set()

    class Teacher:
        def __init__(self, settings):
            pass

        def close(self):
            pass

        async def label(self, video):
            if video["video_id"] == "v0":
                response = httpx.Response(
                    401, request=httpx.Request("POST", "https://teacher.test")
                )
                raise APIStatusError("gateway failure", response=response, body=None)
            await asyncio.wait_for(failed_saved.wait(), timeout=5)
            return annotation(), {"unresolved_segments": 0}

    monkeypatch.setattr("vh_data.labeler.Labeler", Teacher)
    monkeypatch.setattr(store, "fail", record_failure)
    assert run(store, Settings(_env_file=None, workers=2), 4, None, False) == {
        "completed": 1,
        "failed": 1,
    }
    assert store.stats()["status"] == {"failed": 1, "ready": 1, "queued": 2}


def test_video_without_highlights_produces_valid_negative_training_data(tmp_path):
    store = Store(tmp_path / "data")
    ingest_video(store, tmp_path)
    store.save(
        "v1",
        Annotation(
            duration_sec=40,
            segments=[
                Label(start_sec=0, end_sec=40, kind="negative", reason="普通剧情没有突出看点"),
            ],
        ),
        "model_reviewed",
        {},
        expected_label_id=None,
    )
    report = export(store, tmp_path / "ordinary")
    assert report["labels"] == {"positive": 0, "negative": 1, "uncertain": 0}
    assert report["highlight_ratio"] == store.stats()["quality"]["highlight_ratio"] == 0
    row = json.loads((tmp_path / "ordinary/annotations.jsonl").read_text())
    assert row["highlights"] == []
    assert row["negative_intervals"] == [{"start_sec": 0, "end_sec": 40}]


def test_requeued_old_labels_do_not_count_as_available_drama_data(tmp_path):
    store = Store(tmp_path / "data")
    ingest_video(store, tmp_path, "a-old", drama_id="drama-a")
    ingest_video(store, tmp_path, "a-new", drama_id="drama-a")
    ingest_video(store, tmp_path, "b-reviewed", drama_id="drama-b")
    ingest_video(store, tmp_path, "b-new", drama_id="drama-b")
    approve(store, "a-old")
    approve(store, "b-reviewed")
    store.requeue(["a-old"])
    with store.connect() as db:
        db.execute("UPDATE videos SET duration_sec=1 WHERE video_id='b-new'")
    assert store.claim()["drama_id"] == "drama-a"


def test_cancel_stops_active_models_and_leaves_work_resumable(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    for index in range(4):
        ingest_video(store, tmp_path, f"v{index}")
    started = asyncio.Event()
    active, closed = [], []

    class Teacher:
        def __init__(self, settings):
            self.video_id = None

        def close(self):
            if self.video_id is not None:
                closed.append(self.video_id)

        async def label(self, video):
            self.video_id = video["video_id"]
            active.append(self.video_id)
            if len(active) == 2:
                started.set()
            await asyncio.Event().wait()

    async def cancel_batch():
        task = asyncio.create_task(
            _run(
                store,
                Settings(_env_file=None, workers=2),
                4,
                None,
                False,
                None,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

    monkeypatch.setattr("vh_data.labeler.Labeler", Teacher)
    asyncio.run(cancel_batch())
    assert set(active) == set(closed) and len(active) == 2
    assert store.stats()["status"] == {"labeling": 2, "queued": 2}
    store.recover(False)
    assert store.stats()["status"] == {"queued": 4}
