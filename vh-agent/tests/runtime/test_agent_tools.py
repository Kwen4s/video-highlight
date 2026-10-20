"""Candidate/evidence and submission invariants for the seven-tool interface."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vh_agent.models import VideoInfo
from vh_agent.providers.local_detector import detection_windows
from vh_agent.runtime.evidence import VideoObservation, VideoPage
from vh_agent.runtime.finalization import ReviewResult
from vh_agent.runtime.tools import VideoTools


class FakeEvidenceStore:
    def __init__(self, root: Path):
        self.media_id = "media_test"
        self.video_info = VideoInfo(
            path=root / "source.mp4",
            duration_sec=60,
            width=320,
            height=240,
            fps=25,
            has_audio=True,
        )
        self.pages = [
            VideoPage(
                page_id=f"page_{index}",
                core_start_sec=index * 30,
                core_end_sec=(index + 1) * 30,
                read_start_sec=index * 30,
                read_end_sec=(index + 1) * 30,
            )
            for index in range(2)
        ]
        self.output_dir = root / "media"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fail_next_render = False
        self.render_calls = 0
        self.bytes_per_second = 0

    def inspect(self, start_sec: float, end_sec: float) -> VideoObservation:
        if not 0 <= start_sec < end_sec <= self.video_info.duration_sec:
            raise ValueError("Requested observation is outside the video")
        key = hashlib.sha256(f"{start_sec:.3f}:{end_sec:.3f}".encode()).hexdigest()[:12]
        path = self.output_dir / f"{key}.mp4"
        cache_hit = path.exists()
        path.write_bytes(b"fake-media-no-decoding-needed")
        if self.bytes_per_second:
            with path.open("r+b") as stream:
                stream.truncate(int((end_sec - start_sec) * self.bytes_per_second))
        return VideoObservation(
            observation_id=f"obs_{key}",
            media_id=self.media_id,
            requested_start_sec=start_sec,
            requested_end_sec=end_sec,
            src_start_sec=start_sec,
            src_end_sec=end_sec,
            path=path,
            has_audio=True,
            duration_sec=end_sec - start_sec,
            source_start_time_sec=7.25,
            timestamp_precision_sec=0.04,
            cache_hit=cache_hit,
        )

    def scan(self, page_id: str) -> VideoObservation:
        page = next((page for page in self.pages if page.page_id == page_id), None)
        if page is None:
            raise ValueError("Unknown page_id")
        return self.inspect(page.read_start_sec, page.read_end_sec).model_copy(
            update={"page_id": page_id}
        )

    def render_clip(self, start_sec: float, end_sec: float) -> VideoObservation:
        self.render_calls += 1
        if self.fail_next_render:
            self.fail_next_render = False
            raise ValueError("Media render interrupted")
        return self.inspect(start_sec, end_sec)

    def split_page(self, page_id: str) -> list[VideoPage]:
        index = next(index for index, page in enumerate(self.pages) if page.page_id == page_id)
        page = self.pages[index]
        midpoint = (page.core_start_sec + page.core_end_sec) / 2
        children = [
            VideoPage(
                page_id=f"{page_id}_{index}",
                core_start_sec=start,
                core_end_sec=end,
                read_start_sec=start,
                read_end_sec=end,
            )
            for index, (start, end) in enumerate(
                [(page.core_start_sec, midpoint), (midpoint, page.core_end_sec)]
            )
        ]
        self.pages[index : index + 1] = children
        return children

    def restore_pages(self, pages: list[VideoPage]) -> None:
        self.pages = [page.model_copy(deep=True) for page in pages]


@pytest.fixture
def tools(tmp_path):
    return VideoTools(FakeEvidenceStore(tmp_path), tmp_path)


def watch(tools, start=0, end=60):
    _, media = tools.execute(
        "watch_video", {"start_sec": start, "end_sec": end, "question": "发生了什么？"}
    )
    tools.mark_delivered(media)
    return media[0]


def candidate(observation, **updates):
    return {
        "id": None,
        "start_sec": 5,
        "end_sec": 10,
        "description": "证件改变了双方态度",
        "reason": "身份揭晓带来转折",
        "highlight_type": "身份揭晓",
        "score": 0.8,
        "evidence": [
            {
                "start_sec": 6,
                "end_sec": 9,
                "role": "decisive",
                "evidence_id": observation.observation_id,
            }
        ],
        **updates,
    }


def save(tools, observation, **updates):
    result, _ = tools.execute(
        "save_candidates", {"candidates": [candidate(observation, **updates)]}
    )
    return result["candidates"][0]["id"]


def test_observation_delivery_earns_coverage_without_record_tool(tools):
    assert tools.phase == "analysis"
    watch(tools, 0, 30)
    assert tools.progress()["scan_coverage"] == 0.5
    watch(tools, 30, 60)
    assert tools.phase == "selection" and tools.selection_ready()
    assert "submit_highlights" in {d["name"] for d in tools.declarations()}


def test_detection_is_not_video_evidence_or_output_budget(tools):
    tools.detector = SimpleNamespace(
        detect=lambda *_: {
            "segments": [{"start_sec": i, "end_sec": i + 1} for i in range(25)],
            "windows": [],
        }
    )
    result, _ = tools.execute("detect_highlights", {"start_sec": 0, "end_sec": 30, "limit": 10})
    assert result["total"] == 25 and result["next_offset"] == 10
    assert tools.progress()["scan_coverage"] == 0
    with pytest.raises(ValueError, match="observation_id"):
        save(tools, SimpleNamespace(observation_id="unknown"))


def test_save_is_atomic_and_revision_reuses_id(tools):
    observation = watch(tools)
    key = save(tools, observation)
    assert save(tools, observation, id=key, description="修订同一个看点") == key
    assert len(tools.events) == 1
    snapshot = tools.checkpoint()
    with pytest.raises(ValueError):
        tools.execute(
            "save_candidates",
            {
                "story_so_far": "不应保存",
                "candidates": [candidate(observation), candidate(observation, id="missing")],
            },
        )
    assert tools.checkpoint() == snapshot
    with pytest.raises(ValueError, match="完整包含"):
        save(tools, observation, start_sec=8)


def test_full_pool_is_retained_and_submission_is_ordered(tools):
    observation = watch(tools)
    ids = [save(tools, observation, description=str(i)) for i in range(15)]
    assert len(tools.candidate_ledger()) == 15
    tools.execute("submit_highlights", {"candidate_ids": [ids[2], ids[0]]})
    tools.prepare_reviews()
    for plan in tools.pending_reviews():
        tools.record_review(plan.event_id, ReviewResult(visible_event="身份揭晓", issues=[]))
    report = tools.finish_submission()
    assert report["status"] == "awaiting_confirmation"
    tools.mark_reviews_presented(report)
    tools.execute(
        "submit_highlights",
        {
            "candidate_ids": [ids[2], ids[0]],
            "confirmations": [
                {
                    "candidate_id": review["candidate_id"],
                    "review_id": review["review_id"],
                    "reason": "已核对原片与实际成片",
                }
                for review in report["reviews"]
            ],
        },
    )
    assert tools.finish_submission()["status"] == "complete"
    assert tools.progress()["pending_event_count"] == 0
    assert [p.event_id for p in tools.selection.selected] == [ids[2], ids[0]]
    assert len(tools.candidate_ledger()) == 15


def test_review_problem_requires_revision_and_resume_keeps_result(tools, tmp_path):
    observation = watch(tools)
    key = save(tools, observation)
    tools.execute("submit_highlights", {"candidate_ids": [key]})
    tools.prepare_reviews()
    tools.record_review(
        key,
        ReviewResult(
            visible_event="缺少揭晓",
            issues=[{"category": "cutoff", "description": "关键台词被截断", "at_sec": 1}],
        ),
    )
    assert tools.finish_submission()["status"] == "awaiting_confirmation"
    assert not tools.finished
    restored = VideoTools(tools.evidence, tmp_path)
    restored.restore(json.loads(json.dumps(tools.checkpoint())))
    assert restored.finish_submission()["status"] == "awaiting_confirmation"
    assert tools.evidence.render_calls == 1
    save(restored, observation, id=key, end_sec=12)
    restored.execute("submit_highlights", {"candidate_ids": [key]})
    restored.prepare_reviews()
    assert tools.evidence.render_calls == 2


def test_empty_submit_still_requires_video_coverage(tools):
    with pytest.raises(ValueError, match="看完"):
        tools.execute("submit_highlights", {"candidate_ids": []})
    watch(tools)
    tools.execute("submit_highlights", {"candidate_ids": []})
    assert tools.finish_submission()["status"] == "complete"
    assert tools.progress()["pending_event_count"] == 0


def test_tool_capabilities_and_windows_are_explicit(tools):
    declarations = {d["name"]: d for d in tools.declarations()}
    assert set(declarations) == {"watch_video", "read_frames", "save_candidates", "read_memory"}
    assert declarations["read_frames"]["parameters"]["properties"]["with_text"]["enum"] == [False]
    tools.transcript = SimpleNamespace()
    search = next(d for d in tools.declarations() if d["name"] == "search_video")
    assert search["parameters"]["properties"]["mode"]["enum"] == ["exact"]
    for duration in [0.3, 20, 49, 125, 1800]:
        windows = detection_windows(0, duration, duration)
        assert windows[0]["start_sec"] == 0 and windows[-1]["end_sec"] == duration
        assert all(0 < w["end_sec"] - w["start_sec"] <= 20 for w in windows)
        assert all(
            w["start_sec"] - w["read_start_sec"] <= 15 and w["read_end_sec"] - w["end_sec"] <= 15
            for w in windows
        )
    assert [(w["start_sec"], w["end_sec"]) for w in detection_windows(65, 105, 125)] == [
        (60, 80),
        (80, 100),
        (100, 120),
    ]


def test_confirmation_requires_presented_current_review(tools):
    observation = watch(tools)
    key = save(tools, observation, description="嫌疑人认罪")
    tools.execute("submit_highlights", {"candidate_ids": [key]})
    tools.prepare_reviews()
    tools.record_review(key, ReviewResult(visible_event="只见普通寒暄，没有认罪", issues=[]))
    report = tools.finish_submission()
    assert report["status"] == "awaiting_confirmation" and not tools.finished
    confirmation = {
        "candidate_id": key,
        "review_id": report["reviews"][0]["review_id"],
        "reason": "核对说明",
    }
    with pytest.raises(ValueError, match="实际复核"):
        tools.execute(
            "submit_highlights", {"candidate_ids": [key], "confirmations": [confirmation]}
        )
    tools.mark_reviews_presented(report)
    save(tools, observation, id=key, description="双方普通寒暄")
    tools.execute("submit_highlights", {"candidate_ids": [key]})
    assert tools.evidence.render_calls == 1
    with pytest.raises(ValueError, match="实际复核"):
        tools.execute(
            "submit_highlights", {"candidate_ids": [key], "confirmations": [confirmation]}
        )
    current = tools.finish_submission()
    assert current["reviews"][0]["review_id"] != confirmation["review_id"]


def test_main_agent_can_accept_a_mistaken_semantic_review(tools):
    observation = watch(tools)
    key = save(tools, observation)
    tools.execute("submit_highlights", {"candidate_ids": [key]})
    tools.prepare_reviews()
    tools.record_review(
        key,
        ReviewResult(
            visible_event="身份揭晓完整",
            issues=[{"category": "cutoff", "description": "结尾还有悬念", "at_sec": 4}],
        ),
    )
    report = tools.finish_submission()
    tools.mark_reviews_presented(report)
    tools.execute(
        "submit_highlights",
        {
            "candidate_ids": [key],
            "confirmations": [
                {
                    "candidate_id": key,
                    "review_id": report["reviews"][0]["review_id"],
                    "reason": "身份揭晓与反应均完整，开放结尾可以保留",
                }
            ],
        },
    )
    assert tools.finish_submission()["status"] == "complete"
    assert tools.progress()["pending_event_count"] == 0
