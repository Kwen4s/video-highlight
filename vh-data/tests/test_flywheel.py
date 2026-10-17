import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI
from pydantic import ValidationError
from vh_agent.models import TranscriptSegment
from vh_agent.preprocessing.media import video_fingerprint
from vh_agent.runtime.evidence import EvidenceStore, VideoPage

from vh_data.config import Settings
from vh_data.dialogue import DialogueLoader
from vh_data.exporter import export
from vh_data.labeler import DeferredPage, Labeler, PageNeedsSplit, _run, run
from vh_data.models import Annotation, Label, PageLabels, VideoPredictions, digest
from vh_data.quality import overlapping_highlights
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


@pytest.mark.parametrize("status,attempted", [(401, 1), (503, 5), (None, 5)])
def test_only_configuration_errors_stop_the_batch(tmp_path, monkeypatch, status, attempted):
    store = Store(tmp_path / "data")
    for index in range(5):
        ingest_video(store, tmp_path, f"v{index}")

    class Teacher:
        def __init__(self, settings):
            pass

        def close(self):
            pass

        async def label(self, video):
            if status is None:
                raise ValueError("模型响应缺少有效的输出文本")
            response = httpx.Response(status, request=httpx.Request("POST", "https://teacher.test"))
            raise APIStatusError("gateway failure", response=response, body=None)

    monkeypatch.setattr("vh_data.labeler.Labeler", Teacher)
    assert run(store, Settings(_env_file=None, workers=1), 5, None, False) == {
        "completed": 0,
        "failed": attempted,
    }
    expected = {"failed": attempted}
    if attempted < 5:
        expected["queued"] = 5 - attempted
    assert store.stats()["status"] == expected


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
    seen = {"scan": [], "review": []}
    split = False

    async def call(media, page, source_hash, pass_id, story, dialogue, notes=None):
        nonlocal split
        if duration > 60 and not split and pass_id == "scan" and page.core_start_sec == 30:
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
    original = seen["scan"].copy()
    seen = {"scan": [], "review": []}
    asyncio.run(labeler.label(source))
    assert seen["scan"] == seen["review"] == original
    if duration > 60:
        assert labeler._plan_path(source["sha256"]).is_file()
        assert (30, 60) not in seen["scan"]
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
            {"start_sec": 30, "end_sec": 35},
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
        "ordinary_candidate",
        "missed",
        "all_negative",
        "neighbor_owned",
        "contradiction",
        "resize",
        "feedback",
        "incomplete",
        "empty",
        "failed_output",
        "missing_output",
        "malformed_content",
        "failed_retry",
        "stream_error",
        "read_timeout",
        "slow_stream",
        "long_preference",
        "page_overlap",
        "transient",
        "rate_limit",
        "retry_after",
        "retry_date",
        "gateway_wait",
        "persistent_gateway",
        "stream_retry",
        "oversize",
    ],
)
def test_video_annotation_review_and_resume_use_real_sdk_requests(tmp_path, monkeypatch, case):
    server_wait_cases = {"retry_after", "retry_date", "gateway_wait"}
    retry_cases = {
        "transient",
        "rate_limit",
        "stream_retry",
        "empty",
        "failed_output",
        "missing_output",
        "malformed_content",
        "failed_retry",
        "persistent_gateway",
    } | server_wait_cases
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
        idle_timeout_sec=0.1 if case in {"read_timeout", "slow_stream"} else 300,
        response_attempts=3 if case in retry_cases else 1,
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
    current_page = None

    async def resized_call(media, page, source_hash, pass_id, story, dialogue, notes=None):
        nonlocal current_page
        current_page = page
        if (
            case == "resize"
            and pass_id == "review"
            and page.core_start_sec == 0
            and page.core_end_sec == 4
        ):
            raise PageNeedsSplit
        return await call(media, page, source_hash, pass_id, story, dialogue, notes)

    monkeypatch.setattr(labeler, "_call", resized_call)
    contexts = []
    delays = []
    actual_sleep = asyncio.sleep

    async def no_delay(delay):
        delays.append(delay)

    monkeypatch.setattr("vh_data.labeler.asyncio.sleep", no_delay)
    if case == "retry_date":
        monkeypatch.setattr("vh_data.labeler.time.time", lambda: 1_700_000_000)

    blind_calls = []

    class SlowBody(httpx.AsyncByteStream):
        def __init__(self, payload=None):
            self.payload = payload

        async def __aiter__(self):
            if self.payload is None:
                await actual_sleep(settings.idle_timeout_sec)
                raise httpx.ReadTimeout("server stopped sending data")
            for _ in range(8):
                yield b": heartbeat\n\n"
                await actual_sleep(0.02)
            yield self.payload
            raise AssertionError("完成事件后不应继续等待流关闭")

    def respond(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert body["stream"] and not body["store"]
        assert body["reasoning"]["effort"] in {"high", "low"}
        assert body["max_output_tokens"] == 32768
        assert request.extensions["timeout"]["read"] == settings.idle_timeout_sec
        assert "{preferred_highlight_sec}" not in body["instructions"]
        assert f"{settings.preferred_highlight_sec:g} 秒以内" in body["instructions"]
        parts = body["input"][0]["content"]
        context = json.loads(parts[0]["text"])
        contexts.append(context)
        assert parts[2]["image_url"].startswith("data:image/jpeg;base64,")
        assert all(t["source"] == "asr" for t in context["对白与字幕"])
        assert "负责范围" not in context and "前后文范围" not in context
        assert all(
            context["观看范围"][0] <= t["start_sec"] < t["end_sec"] <= context["观看范围"][1]
            for t in context["对白与字幕"]
        )
        assert not any("data:video" in json.dumps(p) for p in parts)
        assert context["原片有音轨"]
        assert body["text"]["format"]["strict"]
        if case == "read_timeout" and len(contexts) == 1:
            return httpx.Response(
                200, stream=SlowBody(), headers={"Content-Type": "text/event-stream"}
            )
        if case == "rate_limit" and len(contexts) == 1:
            return httpx.Response(
                429, json={"error": {"message": "slow down", "type": "rate_limit_error"}}
            )
        if case == "persistent_gateway" and len(contexts) <= 5:
            return httpx.Response(502, json={"error": {"message": "gateway unavailable"}})
        if case in server_wait_cases and len(contexts) == 1:
            if case == "gateway_wait":
                return httpx.Response(
                    502,
                    json={"error_name": "origin_bad_gateway", "retry_after": 60},
                )
            retry_after = "Tue, 14 Nov 2023 22:14:20 GMT" if case == "retry_date" else "60"
            return httpx.Response(
                502,
                headers={"Retry-After": retry_after},
                json={"error": {"message": "try later", "type": "server_error"}},
            )
        first_page = current_page.core_start_sec == 0
        reviewing = "候选位置" in context
        if reviewing:
            assert all(set(s) == {"start_sec", "end_sec"} for s in context["候选位置"])
            assert len(context["候选位置"]) == len(
                {(s["start_sec"], s["end_sec"]) for s in context["候选位置"]}
            )
            if not current_page.page_id.startswith("review-conflict"):
                assert context["之前的剧情"] == ("" if first_page else "复核确认的剧情")
        if not reviewing:
            blind_calls.append(context)
        if case == "oversize" and len(contexts) == 3 and reviewing:
            return httpx.Response(
                413, json={"error": {"message": "too large", "type": "invalid_request_error"}}
            )
        reject_event = (
            case == "all_negative"
            or (case == "ordinary_candidate" and reviewing)
            or (case == "missed" and not reviewing)
        )
        if case == "ordinary_candidate":
            segments = [
                Label(
                    start_sec=current_page.core_start_sec if reviewing else 2.8,
                    end_sec=current_page.core_end_sec if reviewing else 5.2,
                    kind="negative" if reviewing else "positive",
                    reason="复核认为是普通剧情" if reviewing else "跨页候选",
                )
            ]
        elif case == "neighbor_owned":
            segments = [Label(start_sec=4.5, end_sec=6.5, kind="positive", reason="反击奏效")]
            if first_page:
                segments.append(Label(start_sec=1, end_sec=3, kind="negative", reason="普通交谈"))
        elif case == "page_overlap":
            start, end = (2.8, 5.0) if first_page else (3.0, 5.2)
            if reviewing and current_page.page_id.startswith("review-conflict"):
                start, end = 3.0, 5.0
            segments = [Label(start_sec=start, end_sec=end, kind="positive", reason="同一次反击")]
        elif first_page:
            kind = "negative" if reject_event else "positive"
            start, end = (0.5, 1.5) if current_page.core_end_sec == 2 else (1, 3)
            segments = [Label(start_sec=start, end_sec=end, kind=kind, reason="核对画面中的事情")]
        elif current_page.core_start_sec == 2:
            segments = [Label(start_sec=2.5, end_sec=3.5, kind="negative", reason="普通片段")]
        else:
            segments = [Label(start_sec=5, end_sec=7, kind="negative", reason="普通片段")]
        if case == "contradiction" and first_page and not reviewing:
            segments.append(Label(start_sec=2, end_sec=3, kind="negative", reason="矛盾意见"))
        result = PageLabels(
            story_so_far="复核确认的剧情" if reviewing else "候选观看的剧情", segments=segments
        )
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
        elif case == "failed_output" or (case == "failed_retry" and len(contexts) == 1):
            response["status"] = "failed"
            response["output"] = None
            response["error"] = {"code": "server_error", "message": "upstream failed"}
        elif case == "missing_output":
            del response["output"]
        elif case == "malformed_content":
            response["output"][0]["content"] = None
        elif (
            case == "empty"
            or (case == "transient" and len(contexts) == 1)
            or (case == "persistent_gateway" and len(contexts) == 6)
        ):
            response["output"][0]["content"][0]["text"] = " "
        events = [
            {
                "type": "response.created",
                "response": {**response, "output": [], "status": "in_progress"},
                "sequence_number": 0,
            },
            {
                "type": f"response.{response['status']}",
                "response": response,
                "sequence_number": 1,
            },
        ]
        if (case == "stream_error" and len(contexts) <= 4) or (
            case == "stream_retry" and len(contexts) == 1
        ):
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
        if case == "slow_stream":
            return httpx.Response(
                200, stream=SlowBody(data.encode()), headers={"Content-Type": "text/event-stream"}
            )
        return httpx.Response(200, content=data, headers={"Content-Type": "text/event-stream"})

    monkeypatch.setattr(
        "vh_data.labeler.AsyncOpenAI",
        lambda **kwargs: AsyncOpenAI(
            **{**kwargs, "base_url": "https://teacher.test/v1", "max_retries": 0},
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )
    source = {"path": str(video), "sha256": digest(video), "split": "train", "language": "zh"}
    if case in {"incomplete", "empty", "failed_output", "missing_output", "malformed_content"}:
        with pytest.raises(ValueError):
            asyncio.run(labeler.label(source))
        failures = list((settings.root / "failures").glob("*.json"))
        assert len(failures) == settings.response_attempts
        assert len(contexts) == settings.response_attempts
        record = json.loads(failures[0].read_text())
        assert record["usage"]["total_tokens"] == 120
        assert record["status"] == {
            "incomplete": "incomplete",
            "failed_output": "failed",
        }.get(case, "completed")
        if case in {"failed_output", "missing_output", "malformed_content"}:
            assert record["output_text"] is None
        if case == "failed_output":
            assert record["error_code"] == "server_error"
        assert not list((settings.root / "readings").glob("*.json"))
        labeler.close()
        return
    if case == "feedback":
        source["predictions"] = [
            {
                "model_id": "student-v1",
                "segments": [
                    {"start_sec": 1, "end_sec": 3},
                    {"start_sec": 5, "end_sec": 7},
                ],
            }
        ]
    annotation = labeler.label(source)
    if case == "slow_stream":
        annotation = asyncio.wait_for(annotation, timeout=5)
    result, provenance = asyncio.run(annotation)
    expected_calls = {
        "agree": 4,
        "ordinary_candidate": 4,
        "missed": 4,
        "all_negative": 4,
        "neighbor_owned": 4,
        "contradiction": 4,
        "resize": 5,
        "feedback": 4,
        "long_preference": 4,
        "page_overlap": 5,
        "transient": 5,
        "rate_limit": 5,
        "retry_after": 5,
        "retry_date": 5,
        "gateway_wait": 5,
        "persistent_gateway": 10,
        "stream_error": 8,
        "read_timeout": 5,
        "slow_stream": 4,
        "failed_retry": 5,
        "stream_retry": 5,
        "oversize": 6,
    }[case]
    blind_count = 2
    request_count = expected_calls
    failure_count = {
        "transient": 1,
        "rate_limit": 1,
        "retry_after": 1,
        "retry_date": 1,
        "gateway_wait": 1,
        "stream_retry": 1,
        "oversize": 1,
        "persistent_gateway": 6,
        "stream_error": 4,
        "read_timeout": 1,
        "failed_retry": 1,
    }.get(case, 0)
    expected_calls -= failure_count
    assert len(contexts) == request_count
    assert len(provenance["calls"]) == expected_calls
    assert len(provenance["review_calls"]) == expected_calls - blind_count
    if case in server_wait_cases:
        assert delays == [5]
    elif case == "rate_limit":
        assert delays == [2]
    elif case == "persistent_gateway":
        assert delays == [2, 4, 5, 5, 5, 5]
    initial_contexts = []
    for reference in provenance["calls"]:
        record = json.loads((settings.root / reference).read_text())
        if case == "slow_stream":
            assert record["elapsed_sec"] > settings.idle_timeout_sec
        if record["input"]["page"]["core_start_sec"] == 0:
            initial_contexts.append(record["input"]["context"])
    assert len(initial_contexts) == 2
    assert all(c["之前的剧情"] == "" for c in initial_contexts)
    assert not any("候选位置" in c for c in blind_calls)
    expected_segments = 1 if case == "page_overlap" else 3 if case in {"resize", "oversize"} else 2
    assert len(result.segments) == expected_segments
    if case in {"ordinary_candidate", "all_negative"}:
        assert all(s.kind == "negative" for s in result.segments)
        if case == "ordinary_candidate":
            assert sum(s.end_sec - s.start_sec for s in result.segments) == 8
    else:
        assert any(s.kind == "positive" for s in result.segments)
    for reference in provenance["calls"]:
        assert json.loads((settings.root / reference).read_text())["usage"]["total_tokens"] == 120
    asyncio.run(labeler.label(source))
    # Restored child pages change two scans and the resulting review cues.
    assert len(contexts) == request_count + (4 if case in {"resize", "oversize"} else 0)
    cached_count = len(contexts)
    asyncio.run(labeler.label(source))
    assert len(contexts) == cached_count
    labeler.settings = settings.model_copy(update={"reasoning_effort": "low"})
    asyncio.run(labeler.label(source))
    assert len(contexts) == cached_count + (6 if case in {"resize", "oversize"} else expected_calls)
    if failure_count:
        failures = list((settings.root / "failures").glob("*.json"))
        records = [json.loads(path.read_text()) for path in failures]
        assert len(records) == failure_count
        assert sorted(record["attempt"] for record in records) == list(range(1, failure_count + 1))
        if case == "stream_error":
            assert all(record["error_code"] == "upstream_failed" for record in records)
        elif case == "read_timeout":
            assert records[0]["error_class"] == "ReadTimeout"
            assert records[0]["elapsed_sec"] < 1
        elif case == "failed_retry":
            assert records[0]["status"] == "failed"
            assert records[0]["output_text"] is None
            assert records[0]["error_code"] == "server_error"
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


def test_transiently_blocked_video_yields_its_worker_and_is_resumed(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    for index in range(3):
        ingest_video(store, tmp_path, f"v{index}")
    calls = []

    class Teacher:
        def __init__(self, settings):
            pass

        def close(self):
            pass

        async def label(self, video):
            calls.append(video["video_id"])
            if video["video_id"] == "v0" and calls.count("v0") == 1:
                raise DeferredPage("ReadTimeout 连续 3 次；暂缓当前视频以继续其他任务")
            return annotation(), {"unresolved_segments": 0}

    monkeypatch.setattr("vh_data.labeler.Labeler", Teacher)
    assert run(store, Settings(_env_file=None, workers=1), 3, None, False) == {
        "completed": 3,
        "failed": 0,
    }
    assert calls == ["v0", "v1", "v2", "v0"]
    assert store.stats()["status"] == {"ready": 3}


def test_deferred_videos_keep_moving_behind_the_current_queue(tmp_path):
    store = Store(tmp_path / "data")
    for index in range(3):
        ingest_video(store, tmp_path, f"v{index}")
    assert store.claim()["video_id"] == "v0"
    store.defer("v0", "gateway unavailable")
    assert store.claim()["video_id"] == "v1"
    store.defer("v1", "gateway unavailable")
    assert store.get("v0")["priority"] == -1
    assert store.get("v1")["priority"] == -2
    assert store.claim()["video_id"] == "v2"
    store.recover(False)
    assert store.get("v0")["priority"] == store.get("v1")["priority"] == 0


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


@pytest.mark.parametrize("phase", ["request", "retry_wait"])
def test_cancel_stops_active_models_and_leaves_work_resumable(tmp_path, monkeypatch, phase):
    store = Store(tmp_path / "data")
    for index in range(4):
        ingest_video(store, tmp_path, f"v{index}")
    started = asyncio.Event()
    active, closed = [], []

    class Teacher(Labeler):
        def __init__(self, settings):
            self.video_id = None
            self.settings = settings

        def close(self):
            if self.video_id is not None:
                closed.append(self.video_id)

        async def label(self, video):
            self.video_id = video["video_id"]
            active.append(self.video_id)
            if len(active) == 2:
                started.set()
            if phase == "request":
                await asyncio.Event().wait()
            else:
                await self._call(None, None, video["sha256"], "scan", "", [])

        async def _call_once(self, *args):
            response = httpx.Response(502, request=httpx.Request("POST", "https://teacher.test"))
            raise APIStatusError("gateway failure", response=response, body={"retry_after": 60})

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
