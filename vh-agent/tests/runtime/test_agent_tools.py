"""State-machine checks using real tool contracts and fake media rendering."""

import hashlib
import json
from pathlib import Path

import pytest

from vh_agent.models import VideoInfo
from vh_agent.runtime.contracts import EventInput
from vh_agent.runtime.evidence import VideoObservation, VideoPage
from vh_agent.runtime.finalization import EventRecord, RequiredSpan, ReviewResult, complete_review
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
        self.output_dir.mkdir(parents=True)
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


def execute(tools, name, **arguments):
    return tools.execute(name, arguments)


def observe(tools, observation, conclusion="No unresolved event in this observation", events=None):
    findings = []
    for event in events or []:
        payload = event.model_dump(mode="json", exclude={"version"})
        if event.id in tools.events:
            payload["expected_version"] = tools.events[event.id].version
        findings.append(payload)
    return execute(
        tools,
        "record_observations",
        observations=[
            {
                "observation_id": observation.observation_id,
                "no_event_reason": conclusion if not events else None,
                "findings": findings,
            }
        ],
    )


def cover_all_pages(tools):
    observations = []
    while len(tools.completed_pages) < len(tools.evidence.pages):
        media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
        tools.mark_delivered(media)
        observe(tools, media[0])
        observations.extend(media)
    return observations


def supported_event(observation, identifier="event_1", start=5, end=10, **updates):
    return EventRecord.model_validate(
        {
            "id": identifier,
            "description": f"Identity reveal in {identifier}",
            "reason": "The revealed identity changes the encounter",
            "status": "supported",
            "required_spans": [
                RequiredSpan(
                    start_sec=start,
                    end_sec=end,
                    evidence_id=observation.observation_id,
                    role="decisive",
                ).model_dump()
            ],
            **updates,
        }
    )


def upsert(tools, event, expected_version=None):
    payload = event.model_dump(mode="json", exclude={"version"})
    if event.id not in tools.events:
        return tools._save_event(EventInput.model_validate(payload))
    payload["expected_version"] = expected_version
    return execute(
        tools,
        "update_event",
        event=payload,
    )[0]


def review(**updates):
    return ReviewResult.model_validate(
        {
            "visible_event": "The character reveals her identity",
            "highlight_type": "identity_reveal",
            "blocking_issues": [],
            **updates,
        }
    )


def prepare_ready(tools, event):
    upsert(tools, event)
    tools.prepare_reviews()
    tools.record_review(event.id, review())
    plan = tools.plans[event.id]
    tools.plans[event.id] = complete_review(plan, tools.events[event.id], plan.review)
    return tools.plans[event.id]


def test_prepared_delivered_and_acknowledged_are_separate_coverage_states(tools):
    media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    assert tools.progress()["scan_coverage"] == 0
    with pytest.raises(ValueError, match="本轮收到"):
        observe(tools, media[0])

    tools.mark_delivered(media)
    assert tools.progress()["scan_coverage"] == 0
    assert tools.progress()["unacknowledged_observations"][0]["observation_id"] == (
        media[0].observation_id
    )
    with pytest.raises(ValueError, match="no_event_reason"):
        observe(tools, media[0], "  ")
    observe(tools, media[0])
    assert tools.progress()["scan_coverage"] == 0.5
    assert tools.completed_pages == {"page_0"}


def test_observation_requires_findings_or_explicit_no_event_reason(tools):
    media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    tools.mark_delivered(media)
    with pytest.raises(ValueError, match="findings|no_event_reason"):
        execute(
            tools,
            "record_observations",
            observations=[{"observation_id": media[0].observation_id}],
        )
    assert tools.progress()["scan_coverage"] == 0
    observe(tools, media[0], events=[])
    assert tools.progress()["scan_coverage"] == 0.5


def test_observation_records_discovered_events_before_earning_page_coverage(tools):
    media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    tools.mark_delivered(media)
    item = supported_event(media[0])
    observe(tools, media[0], conclusion="A supported identity reveal occurs", events=[item])
    assert tools.events[item.id].status == "supported"
    assert tools.unresolved_events() == [item.id]
    assert tools.progress()["scan_coverage"] == 0.5
    assert len(tools.events) == 1


def test_observation_event_batch_failure_rolls_back_every_event_and_coverage(tools):
    media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    tools.mark_delivered(media)
    valid = supported_event(media[0], identifier="valid")
    invalid = supported_event(media[0], identifier="invalid", start=25, end=35)
    before = tools.checkpoint()
    with pytest.raises(ValueError, match="超出"):
        observe(tools, media[0], conclusion="Two events found", events=[valid, invalid])
    assert tools.checkpoint() == before
    assert tools.events == {}
    assert tools.progress()["scan_coverage"] == 0
    assert media[0].observation_id not in tools.acknowledged


def test_observation_batch_rollback_restores_prior_event_version_and_frozen_clip(tools):
    original_observation = cover_all_pages(tools)[0]
    original = supported_event(original_observation)
    original_plan = prepare_ready(tools, original)
    _, media = execute(
        tools, "inspect_interval", start_sec=20, end_sec=25, question="Check a possible update"
    )
    tools.mark_delivered(media)
    changed = supported_event(media[0], start=20, end=24, description="A corrected identity reveal")
    invalid = supported_event(media[0], identifier="invalid", start=20, end=30)
    before = tools.checkpoint()
    with pytest.raises(ValueError, match="超出"):
        observe(tools, media[0], conclusion="Reconsider two events", events=[changed, invalid])
    assert tools.checkpoint() == before
    assert tools.events[original.id].version == 1
    assert tools.plans[original.id] == original_plan


def test_boundary_recheck_can_update_event_without_forcing_new_required_footage(tools):
    observation = cover_all_pages(tools)[0]
    original = supported_event(observation, proposed_start_sec=5, proposed_end_sec=15)
    prepare_ready(tools, original)
    _, media = execute(
        tools, "inspect_interval", start_sec=10, end_sec=18, question="核对结尾应在哪里停下"
    )
    tools.mark_delivered(media)
    changed = original.model_copy(update={"proposed_end_sec": 12})
    observe(tools, media[0], events=[changed])
    assert tools.events[original.id].version == 2
    assert tools.events[original.id].required_spans == original.required_spans
    assert tools.acknowledged[media[0].observation_id]["event_ids"] == [original.id]
    assert original.id not in tools.plans
    tools.prepare_reviews()
    plan = tools.plans[original.id]
    assert (plan.start_sec, plan.end_sec) == (5, 12)
    assert plan.review is None


def test_cannot_complete_before_all_pages_and_pending_events_are_resolved(tools):
    with pytest.raises(ValueError, match="尚未完成"):
        execute(tools, "select_highlights", decisions=[])
    observations = cover_all_pages(tools)
    pending = supported_event(observations[0], status="pending")
    upsert(tools, pending)
    with pytest.raises(ValueError, match="尚未完成"):
        execute(tools, "select_highlights", decisions=[])
    assert not tools.finished
    assert tools.progress()["pending_event_count"] == 1


def test_full_video_with_no_events_completes_normally(tools):
    cover_all_pages(tools)
    result, media = execute(tools, "select_highlights", decisions=[])
    assert media == []
    assert result == {"selected": [], "decisions": []}
    assert tools.finished
    assert tools.progress()["scan_coverage"] == 1


def test_infeasible_duration_requires_explicit_event_disposition(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation, start=1, end=29)
    upsert(tools, item)
    view = tools._event_view(tools.events[item.id])
    assert view["draft_preview"]["status"] == "infeasible_duration"
    assert view["draft_preview"]["duration_sec"] == 28
    tools.prepare_reviews()
    assert tools.plans[item.id].status == "infeasible_duration"
    assert tools.evidence.render_calls == 0
    assert tools.unresolved_events() == [item.id]
    with pytest.raises(ValueError, match="尚未完成"):
        execute(tools, "select_highlights", decisions=[])
    upsert(
        tools,
        item.model_copy(
            update={
                "status": "rejected",
                "rejection_category": "clip_infeasible",
                "reason": "Cannot fit necessary evidence",
            }
        ),
        expected_version=1,
    )
    result, _ = execute(tools, "select_highlights", decisions=[])
    assert tools.finished
    assert result["selected"] == []


@pytest.mark.parametrize("delivered", [False, True])
def test_new_unacknowledged_inspection_blocks_completion_even_after_full_scan(tools, delivered):
    cover_all_pages(tools)
    _, media = execute(
        tools,
        "inspect_interval",
        start_sec=5,
        end_sec=12,
        question="Does the reaction finish the event?",
    )
    if delivered:
        tools.mark_delivered(media)
    expected = "record_observations" if delivered else "尚未完成"
    with pytest.raises(ValueError, match=expected):
        execute(tools, "select_highlights", decisions=[])
    assert not tools.finished
    tools.mark_delivered(media)
    observe(tools, media[0])
    execute(tools, "select_highlights", decisions=[])
    assert tools.finished


def test_required_spans_must_reference_delivered_evidence_within_its_actual_range(tools):
    _, media = execute(
        tools,
        "inspect_interval",
        start_sec=5,
        end_sec=12,
        question="Inspect the reveal",
    )
    item = supported_event(media[0])
    with pytest.raises(ValueError, match="已经观看|尚未成功送达"):
        upsert(tools, item)
    tools.mark_delivered(media)
    outside = supported_event(media[0], start=4, end=10)
    with pytest.raises(ValueError, match="超出"):
        upsert(tools, outside)
    unknown = supported_event(
        media[0],
        required_spans=[
            RequiredSpan(start_sec=5, end_sec=10, evidence_id="invented", role="decisive")
        ],
    )
    with pytest.raises(ValueError, match="已经观看|尚未成功送达"):
        upsert(tools, unknown)
    assert upsert(tools, item)["event"]["version"] == 1


def test_selection_accepts_review_and_freezes_clip_atomically(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation)
    upsert(tools, item)
    tools.prepare_reviews()
    assert [plan.event_id for plan in tools.pending_reviews()] == [item.id]
    tools.record_review(item.id, review())
    assert tools.plans[item.id].status == "draft"
    assert tools.unresolved_events() == []
    assert {declaration["name"] for declaration in tools.declarations()} == {
        "select_highlights",
        "update_event",
        "inspect_interval",
        "read_state",
    }
    current, _ = execute(tools, "read_state", event_id=item.id)
    assert current["event"]["version"] == item.version
    execute(
        tools,
        "select_highlights",
        decisions=[{"event_id": "event_1", "score": 0.8, "selected": True, "reason": "Useful"}],
    )
    assert tools.plans[item.id].status == "ready"
    assert tools.finished


def test_event_update_uses_server_version_and_invalidates_frozen_clip(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation)
    plan = prepare_ready(tools, item)
    execute(
        tools,
        "select_highlights",
        decisions=[{"event_id": plan.event_id, "score": 0.8, "selected": True, "reason": "Useful"}],
    )
    changed = supported_event(observation, version=999, description="Corrected identity reveal")
    with pytest.raises(ValueError, match="版本冲突"):
        upsert(tools, changed, expected_version=999)
    assert tools.plans[item.id].status == "ready"
    row = upsert(tools, changed, expected_version=1)
    assert row["event"]["version"] == 2
    assert item.id not in tools.plans
    assert not tools.finished
    assert tools.unresolved_events() == [item.id]
    with pytest.raises(ValueError, match="尚未完成"):
        execute(tools, "select_highlights", decisions=[])


def test_identical_event_upsert_preserves_existing_review_and_version(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation)
    plan = prepare_ready(tools, item)
    row = upsert(tools, item, expected_version=1)
    assert row["event"]["version"] == 1
    assert tools.plans[item.id] == plan


def test_irreparable_blocking_review_can_be_explicitly_rejected(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation)
    upsert(tools, item)
    tools.prepare_reviews()
    tools.record_review(
        item.id,
        review(
            blocking_issues=[
                {"category": "missing_context", "description": "The relationship is unexplained"}
            ],
        ),
    )
    assert tools.plans[item.id].status == "draft"
    assert tools.unresolved_events() == [item.id]
    rejected = supported_event(
        observation,
        status="rejected",
        rejection_category="clip_infeasible",
        reason="The source lacks the context needed to understand the observed interaction",
    )
    upsert(tools, rejected, expected_version=1)
    result, _ = execute(tools, "select_highlights", decisions=[])
    assert tools.finished
    assert result["selected"] == []


def test_clip_infeasible_rejection_requires_an_actual_blocking_plan(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation)
    upsert(tools, item)
    rejected = item.model_copy(
        update={
            "status": "rejected",
            "rejection_category": "clip_infeasible",
            "reason": "The event is not worth selecting",
        }
    )
    with pytest.raises(ValueError, match="时长不可行或阻断复核"):
        upsert(tools, rejected, expected_version=1)
    assert tools.events[item.id].status == "supported"


def test_mixed_focus_review_requires_splitting_instead_of_rejecting_the_event(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation)
    upsert(tools, item)
    tools.prepare_reviews()
    tools.record_review(
        item.id,
        review(
            blocking_issues=[
                {
                    "category": "mixed_focus",
                    "description": "The clip combines two independently usable reveals",
                }
            ]
        ),
    )
    rejected = supported_event(
        observation,
        status="rejected",
        rejection_category="clip_infeasible",
        reason="The clip contains two hooks",
    )
    with pytest.raises(ValueError, match="多个独立看点应拆成事件"):
        upsert(tools, rejected, expected_version=1)


def test_all_twenty_ready_events_participate_before_output_cap_twelve(tools):
    observation = cover_all_pages(tools)[0]
    for index in range(20):
        # Every interval overlaps. Distinct event identity must not be inferred from IoU.
        item = supported_event(observation, identifier=f"event_{index:02d}")
        prepare_ready(tools, item)
    first, _ = execute(tools, "read_state", limit=7, collection="events")
    assert len(first["events"]) == 7
    assert first["total"] == 20
    assert first["next_offset"] == 7
    pool, _ = execute(tools, "read_state", collection="candidates", limit=100)
    assert len(pool["candidates"]) == 20
    result, _ = execute(
        tools,
        "select_highlights",
        decisions=[
            {
                "event_id": f"event_{i:02d}",
                "score": 0.8,
                "selected": i >= 8,
                "reason": "Editorial choice",
            }
            for i in reversed(range(20))
        ],
    )
    assert len(result["decisions"]) == 20
    assert len(result["selected"]) == 12
    assert {plan["event_id"] for plan in result["selected"]} == {
        f"event_{index:02d}" for index in range(8, 20)
    }
    assert len(tools.events) == 20


def test_selection_receives_the_complete_candidate_pool_without_a_read_roundtrip(tools):
    observation = cover_all_pages(tools)[0]
    plans = [
        prepare_ready(
            tools,
            supported_event(
                observation,
                identifier=f"event_{index:02d}",
                description="An early interpretation to verify",
                reason="An unsupported claim about the character's motives",
            ),
        )
        for index in range(20)
    ]
    candidates = tools.candidate_ledger()
    assert len(candidates) == 20
    assert candidates[0]["visible_event"] == review().visible_event
    assert set(candidates[0]) == {
        "event_id",
        "event_version",
        "start_sec",
        "end_sec",
        "visible_event",
    }
    assert "early interpretation" not in json.dumps(candidates)
    assert "unsupported claim" not in json.dumps(candidates)
    decisions = [
        {
            "event_id": plan.event_id,
            "score": 0.8,
            "selected": False,
            "reason": "Lower incremental value",
        }
        for plan in plans
    ]
    result, _ = execute(tools, "select_highlights", decisions=decisions)
    assert result["selected"] == []


def test_completed_page_must_be_revisited_with_interval_tool(tools):
    observation = cover_all_pages(tools)[0]
    assert tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000) == []
    _, media = execute(
        tools,
        "inspect_interval",
        start_sec=observation.src_start_sec,
        end_sec=observation.src_end_sec,
        question="重新核对人物反应",
    )
    assert media


def test_checkpoint_restores_pending_media_and_review_work_without_false_coverage(tools):
    media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    restored = VideoTools(tools.evidence, tools.output_dir)
    restored.restore(json.loads(json.dumps(tools.checkpoint())))
    assert restored.progress()["scan_coverage"] == 0
    assert media[0].observation_id in restored.observations
    with pytest.raises(ValueError, match="本轮收到"):
        observe(restored, media[0])
    restored.mark_delivered(media)
    observe(restored, media[0])
    item = supported_event(media[0])
    upsert(restored, item)
    cover_all_pages(restored)
    restored.prepare_reviews()

    resumed = VideoTools(tools.evidence, tools.output_dir)
    resumed.restore(json.loads(json.dumps(restored.checkpoint())))
    assert resumed.progress()["scan_coverage"] == 1
    assert [plan.event_id for plan in resumed.pending_reviews()] == [item.id]
    assert resumed.plans[item.id].media_path.is_file()
    resumed.record_review(item.id, review())
    final_resume = VideoTools(tools.evidence, tools.output_dir)
    final_resume.restore(json.loads(json.dumps(resumed.checkpoint())))
    assert final_resume.pending_reviews() == []
    assert final_resume.unresolved_events() == []
    assert [row["event_id"] for row in final_resume.candidate_ledger()] == [item.id]
    assert final_resume.plans[item.id].review == review()


def test_oversized_page_is_split_before_the_first_child_is_prepared(tools, monkeypatch):
    monkeypatch.setattr("vh_agent.runtime.tools.MAX_OBSERVATION_BYTES", 48)
    tools.evidence.bytes_per_second = 2
    media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    assert len(media) == 1
    assert media[0].page_id == "page_0_0"
    assert tools.delivered == set()
    assert tools.progress()["scan_coverage"] == 0
    assert tools.progress()["page_count"] == 3
    assert sum(page.core_end_sec - page.core_start_sec for page in tools.evidence.pages) == 60
    assert [(page.core_start_sec, page.core_end_sec) for page in tools.evidence.pages[:2]] == [
        (0, 15),
        (15, 30),
    ]
    tools.mark_delivered(media)
    observe(tools, media[0])
    observations = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    tools.mark_delivered(observations)
    observe(tools, observations[0])
    assert tools.progress()["scan_coverage"] == 0.5


def test_oversized_inspection_checks_base64_size_and_does_not_register_observation(
    tools, monkeypatch
):
    monkeypatch.setattr("vh_agent.runtime.tools.MAX_OBSERVATION_BYTES", 48)
    tools.evidence.bytes_per_second = 2
    # Forty raw bytes fit the limit, but the 56-byte base64 representation does not.
    with pytest.raises(ValueError, match="区间过大"):
        execute(tools, "inspect_interval", start_sec=0, end_sec=20, question="Read the scene")
    assert tools.observations == {}
    assert tools.progress()["pending_observation_count"] == 0
    assert tools.progress()["scan_coverage"] == 0
    assert tools.progress()["page_count"] == 2


def test_restore_preserves_split_page_manifest_and_partial_coverage(tools, tmp_path, monkeypatch):
    monkeypatch.setattr("vh_agent.runtime.tools.MAX_OBSERVATION_BYTES", 48)
    tools.evidence.bytes_per_second = 2
    media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    tools.mark_delivered(media)
    observe(tools, media[0])
    snapshot = json.loads(json.dumps(tools.checkpoint()))

    resumed_evidence = FakeEvidenceStore(tmp_path / "resumed")
    resumed = VideoTools(resumed_evidence, tmp_path / "resumed")
    assert len(resumed_evidence.pages) == 2
    resumed.restore(snapshot)
    assert resumed_evidence.pages == tools.evidence.pages
    assert len(resumed_evidence.pages) == 3
    assert resumed.progress()["scan_coverage"] == 0.25
    assert resumed.progress()["next_page"]["page_id"] == "page_0_1"
    assert resumed.completed_pages == tools.completed_pages


def test_checkpoint_is_a_snapshot_not_a_reference_to_live_acknowledgements(tools):
    media = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    tools.mark_delivered(media)
    before = tools.checkpoint()
    observe(tools, media[0])
    assert before["acknowledged"] == {}
    assert before["completed_pages"] == []
    resumed = VideoTools(tools.evidence, tools.output_dir)
    resumed.restore(before)
    observe(resumed, media[0])
    assert before["acknowledged"] == {}


def test_failed_render_remains_retryable_and_never_creates_a_ready_plan(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation)
    upsert(tools, item)
    tools.evidence.fail_next_render = True
    with pytest.raises(ValueError, match="render interrupted"):
        tools.prepare_reviews()
    assert item.id not in tools.plans
    assert tools.unresolved_events() == [item.id]
    tools.prepare_reviews()
    assert tools.plans[item.id].status == "draft"
    assert tools.evidence.render_calls == 2


def test_merge_cannot_erase_original_decisive_evidence_from_the_source(tools):
    observation = cover_all_pages(tools)[0]
    target = supported_event(observation, identifier="target", start=15, end=20)
    source = supported_event(observation, identifier="source", start=5, end=10)
    upsert(tools, target)
    upsert(tools, source)
    erased = supported_event(
        observation,
        identifier="source",
        status="merged",
        merged_into="target",
        required_spans=[],
    )
    with pytest.raises(ValueError, match="decisive|evidence|span"):
        upsert(tools, erased, expected_version=1)
    assert tools.events["source"].status == "supported"


def test_merge_target_must_preserve_decisive_role_not_only_overlapping_setup(tools):
    observation = cover_all_pages(tools)[0]
    source = supported_event(observation, identifier="source", start=5, end=10)
    target = supported_event(
        observation,
        identifier="target",
        start=15,
        end=20,
        required_spans=[
            RequiredSpan(
                start_sec=5, end_sec=10, evidence_id=observation.observation_id, role="setup"
            ),
            RequiredSpan(
                start_sec=15, end_sec=20, evidence_id=observation.observation_id, role="decisive"
            ),
        ],
    )
    upsert(tools, target)
    upsert(tools, source)
    merged = source.model_copy(update={"status": "merged", "merged_into": "target"})
    with pytest.raises(ValueError, match="decisive|evidence|span"):
        upsert(tools, merged, expected_version=1)


@pytest.mark.parametrize("target_status", ["supported", "rejected"])
def test_surviving_merge_target_cannot_later_drop_merged_source_evidence(tools, target_status):
    observation = cover_all_pages(tools)[0]
    source = supported_event(observation, identifier="source", start=5, end=10)
    target = supported_event(observation, identifier="target", start=5, end=20)
    upsert(tools, target)
    upsert(tools, source)
    merged = source.model_copy(update={"status": "merged", "merged_into": "target"})
    upsert(tools, merged, expected_version=1)
    changed_target = supported_event(
        observation,
        identifier="target",
        start=15,
        end=20,
        status=target_status,
        rejection_category="contradicted" if target_status == "rejected" else None,
    )
    with pytest.raises(ValueError, match="merged|decisive|evidence|span"):
        upsert(tools, changed_target, expected_version=1)
    assert tools.events["target"].version == 1


def test_local_proposals_do_not_use_output_cap_and_require_native_inspection(tools):
    tools.local_proposals = lambda: [{"start_sec": 1, "end_sec": 4, "score": 0.2}] * 25
    result, _ = execute(tools, "propose_highlights", limit=10)
    assert result["total"] == 25 and result["next_offset"] == 10
    assert tools.progress()["pending_proposal_count"] == 25
    with pytest.raises(ValueError, match="送达|已经观看"):
        execute(
            tools,
            "propose_highlights",
            action="resolve",
            proposal_id="proposal_0",
            observation_ids=["unknown_observation"],
            reason="No qualifying event visible",
        )
    observation = cover_all_pages(tools)[0]
    execute(
        tools,
        "propose_highlights",
        action="resolve",
        proposal_id="proposal_0",
        observation_ids=[observation.observation_id],
        reason="No qualifying event visible",
    )
    assert tools.progress()["pending_proposal_count"] == 24
    with pytest.raises(ValueError, match="尚未完成"):
        execute(tools, "select_highlights", decisions=[])
    restored = VideoTools(tools.evidence, tools.output_dir)
    restored.restore(tools.checkpoint())
    assert len(restored.proposals) == 25


def test_enabled_local_proposal_source_must_be_loaded_before_selection(tools):
    tools.local_proposals = list
    cover_all_pages(tools)
    with pytest.raises(ValueError, match="尚未完成"):
        execute(tools, "select_highlights", decisions=[])
    assert tools.progress()["local_proposals_loaded"] is False
    execute(tools, "propose_highlights", action="list")
    assert tools.progress()["local_proposals_loaded"] is True
    execute(tools, "select_highlights", decisions=[])
    assert tools.finished


def test_explicit_dense_inspection_has_distinct_observation_identity(tools):
    normal, _ = execute(tools, "inspect_interval", start_sec=1, end_sec=4, question="look")
    dense, _ = execute(
        tools, "inspect_interval", start_sec=1, end_sec=4, question="fast action", sampling_fps=20
    )
    assert normal["observation_id"] != dense["observation_id"]
    assert dense["sampling_fps"] == 20


def test_proposal_resolution_accepts_contiguous_evidence_across_pages(tools):
    tools.local_proposals = lambda: [{"start_sec": 1, "end_sec": 19, "score": 0.8}]
    execute(tools, "propose_highlights")
    observations = cover_all_pages(tools)
    execute(
        tools,
        "propose_highlights",
        action="resolve",
        proposal_id="proposal_0",
        observation_ids=[o.observation_id for o in observations],
        reason="No qualifying event",
    )
    assert tools.progress()["pending_proposal_count"] == 0


def test_two_findings_are_saved_and_linked_in_one_observation(tools):
    observations = tools.scan_next_observations(max_items=1, max_encoded_bytes=14_000_000)
    tools.mark_delivered(observations)
    first = supported_event(observations[0], identifier="first", start=1, end=5)
    second = supported_event(observations[0], identifier="second", start=15, end=20)
    observe(tools, observations[0], events=[first, second])
    assert tools.acknowledged[observations[0].observation_id]["event_ids"] == ["first", "second"]
    assert tools.unresolved_events() == ["first", "second"]
    assert tools.progress()["scan_coverage"] == 0.5


def test_reason_rewrite_does_not_count_as_progress(tools):
    observation = cover_all_pages(tools)[0]
    event = supported_event(observation)
    upsert(tools, event)
    before = tools.information_facts()
    upsert(tools, event.model_copy(update={"reason": "Another wording"}), expected_version=1)
    assert tools.information_facts() == before


def test_enabled_transcript_must_be_read_to_completion_before_selection(tools):
    from types import SimpleNamespace

    rows = [
        {"row_id": f"line_{index}", "start_sec": index, "end_sec": index + 1, "text": text}
        for index, text in enumerate(["第一句", "第二句"])
    ]

    def search(**arguments):
        offset, limit = arguments["offset"], arguments["limit"]
        return {
            "matches": rows[offset : offset + limit],
            "total": len(rows),
            "next_offset": offset + limit if offset + limit < len(rows) else None,
        }

    tools.transcript = SimpleNamespace(search=search)
    cover_all_pages(tools)
    assert tools.progress()["transcript_complete"] is False
    execute(tools, "search_transcript", query="", offset=0, limit=1)
    with pytest.raises(ValueError, match="尚未完成"):
        execute(tools, "select_highlights", decisions=[])
    execute(tools, "search_transcript", query="", offset=1, limit=1)
    assert tools.progress()["transcript_complete"] is True
    execute(tools, "select_highlights", decisions=[])
    assert tools.finished


def test_transcript_memory_survives_checkpoint_and_equivalent_queries(tools):
    from types import SimpleNamespace

    transcript = SimpleNamespace(
        search=lambda **kwargs: {
            "matches": [{"row_id": "line_1", "start_sec": 1, "end_sec": 2, "text": "你好"}],
            "total": 1,
            "next_offset": None,
        }
    )
    tools.transcript = transcript
    first, _ = execute(tools, "search_transcript")
    assert first["new_row_ids"] == ["line_1"]
    restored = VideoTools(tools.evidence, tools.output_dir, transcript=transcript)
    restored.restore(json.loads(json.dumps(tools.checkpoint())))
    before = restored.information_facts()
    repeated, _ = execute(restored, "search_transcript", query="", offset=0, limit=100)
    assert repeated["new_row_ids"] == []
    assert repeated["query_complete"]
    assert restored.information_facts() == before
    rows, _ = execute(restored, "read_state", collection="transcript")
    assert rows["transcript"][0]["text"] == "你好"


def test_explicit_selection_restores_and_failed_selection_does_not_commit(tools):
    observation = cover_all_pages(tools)[0]
    plans = [
        prepare_ready(tools, supported_event(observation, identifier=f"event_{i}"))
        for i in range(2)
    ]
    tools.max_highlights = 1
    execute(tools, "read_state", collection="candidates")
    before = tools.checkpoint()
    with pytest.raises(ValueError, match="count budget"):
        execute(
            tools,
            "select_highlights",
            decisions=[
                {"event_id": p.event_id, "score": 0.8, "selected": True, "reason": "Useful"}
                for p in plans
            ],
        )
    assert tools.checkpoint() == before
    decisions = [
        {"event_id": p.event_id, "score": 0.8, "selected": i == 1, "reason": "Editorial choice"}
        for i, p in enumerate(plans)
    ]
    execute(tools, "select_highlights", decisions=decisions)
    restored = VideoTools(tools.evidence, tools.output_dir)
    restored.restore(tools.checkpoint())
    assert restored.finished
    assert restored.selection == tools.selection
    assert restored.selection.selected == [plans[1]]


def test_ready_review_does_not_force_selection(tools):
    observation = cover_all_pages(tools)[0]
    item = supported_event(observation)
    plan = prepare_ready(tools, item)
    assert plan.status == "ready"
    execute(tools, "read_state", collection="candidates")
    result, _ = execute(
        tools,
        "select_highlights",
        decisions=[
            {
                "event_id": plan.event_id,
                "score": 0.8,
                "selected": False,
                "reason": "No additional viewing value",
            }
        ],
    )
    assert not result["selected"]
    assert tools.events[item.id].status == "supported"
