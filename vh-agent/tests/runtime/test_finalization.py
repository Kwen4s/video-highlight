from pathlib import Path

import pytest
from pydantic import ValidationError

from vh_agent.runtime.finalization import (
    ClipPlan,
    EventRecord,
    RequiredSpan,
    ReviewIssue,
    ReviewResult,
    SelectionChoice,
    attach_media,
    complete_review,
    draft_plan,
    select_clips,
)


def span(start: float, end: float, role: str = "decisive", identifier: str = "e1"):
    return RequiredSpan(start_sec=start, end_sec=end, evidence_id=identifier, role=role)


def event(identifier="event_1", spans=None, **updates):
    return EventRecord(
        **{
            "id": identifier,
            "description": "The character reveals her identity",
            "reason": "The revealed identity changes the encounter",
            "status": "supported",
            "required_spans": [span(10, 15)] if spans is None else spans,
            **updates,
        }
    )


def review(**updates):
    return ReviewResult(
        **{
            "visible_event": "The character reveals her identity",
            "highlight_type": "identity_reveal",
            "blocking_issues": [],
            **updates,
        }
    )


def ready(item, video_duration=1000):
    return complete_review(draft_plan(item, video_duration), item, review())


def test_pending_and_rejected_events_can_have_no_evidence():
    assert event(spans=[], status="pending").required_spans == []
    assert (
        event(spans=[], status="rejected", rejection_category="not_observed").required_spans == []
    )
    with pytest.raises(ValidationError, match="decisive"):
        event(spans=[])
    with pytest.raises(ValidationError, match="decisive"):
        event(spans=[span(1, 2, "setup")])
    with pytest.raises(ValidationError):
        event(spans=[], status="rejected", rejection_category="not_observed", reason="")
    with pytest.raises(ValidationError, match="rejection category"):
        event(spans=[], status="rejected")
    with pytest.raises(ValidationError, match="Only rejected"):
        event(rejection_category="outside_request")


def test_merge_requires_a_different_event_and_does_not_infer_identity():
    assert event(status="merged", merged_into="event_2").merged_into == "event_2"
    with pytest.raises(ValidationError):
        event(status="merged", merged_into="event_1")
    with pytest.raises(ValidationError):
        event(merged_into="event_2")


@pytest.mark.parametrize("start,end", [(3, 3), (3, 2), (-1, 2), (1, float("inf"))])
def test_required_evidence_has_a_finite_positive_interval(start, end):
    with pytest.raises(ValidationError):
        span(start, end)


def test_draft_preserves_entire_setup_decisive_and_reaction_intervals():
    item = event(
        spans=[span(5, 8, "setup"), span(10, 13), span(17, 20, "reaction")],
        proposed_start_sec=10,
        proposed_end_sec=13,
    )
    plan = draft_plan(item, 30)
    assert (plan.start_sec, plan.end_sec) == (5, 20)
    assert plan.status == "draft"
    assert plan.required_spans == item.required_spans
    assert plan.required_spans is not item.required_spans


@pytest.mark.parametrize("start,end,expected", [(0, 1, (0, 3)), (9, 10, (7, 10))])
def test_minimum_duration_padding_stays_inside_video(start, end, expected):
    plan = draft_plan(event(spans=[span(start, end)]), 10)
    assert (plan.start_sec, plan.end_sec) == expected
    assert plan.status == "draft"


def test_explicit_zero_start_preserves_requested_context():
    plan = draft_plan(event(proposed_start_sec=0), 30)
    assert plan.start_sec == 0


def test_overlong_evidence_is_infeasible_without_cutting_any_span():
    item = event(spans=[span(1, 4, "setup"), span(25, 30)])
    plan = draft_plan(item, 40)
    assert plan.status == "infeasible_duration"
    assert (plan.start_sec, plan.end_sec) == (1, 30)
    assert plan.issues
    with pytest.raises(ValueError, match="infeasible"):
        complete_review(plan, item, review())


def test_overlong_proposed_context_requires_an_explicit_revision():
    plan = draft_plan(event(proposed_start_sec=0, proposed_end_sec=30), 40)
    assert plan.status == "infeasible_duration"
    assert (plan.start_sec, plan.end_sec) == (0, 30)


def test_short_video_is_explicitly_infeasible_under_minimum_duration():
    plan = draft_plan(event(spans=[span(0, 1)]), 2)
    assert plan.status == "infeasible_duration"
    assert (plan.start_sec, plan.end_sec) == (0, 2)


def test_draft_rejects_out_of_video_evidence_and_context():
    with pytest.raises(ValueError, match="evidence is outside"):
        draft_plan(event(spans=[span(5, 15)]), 10)
    with pytest.raises(ValueError, match="boundary is outside"):
        draft_plan(event(proposed_end_sec=40), 30)
    with pytest.raises(ValueError, match="supported"):
        draft_plan(event(status="pending"), 30)


def test_successful_review_freezes_the_same_boundaries():
    item = event()
    plan = draft_plan(item, 30)
    accepted = complete_review(plan, item, review())
    assert accepted.status == "ready"
    assert accepted.review.visible_event == review().visible_event
    assert (accepted.start_sec, accepted.end_sec) == (plan.start_sec, plan.end_sec)
    assert plan.status == "draft"
    with pytest.raises(ValidationError, match="frozen"):
        accepted.end_sec = 13
    with pytest.raises(ValueError, match="frozen"):
        complete_review(accepted, item, review())


@pytest.mark.parametrize(
    "issue",
    [
        ReviewIssue(category="missing_context", description="The relationship is unexplained"),
        ReviewIssue(category="cutoff", description="The last sentence is cut off", at_sec=4),
        ReviewIssue(category="technical", description="The audio is inaudible"),
    ],
)
def test_failed_review_preserves_observation_and_returns_a_draft(issue):
    item = event()
    observed = review(blocking_issues=[issue])
    plan = complete_review(draft_plan(item, 30), item, observed)
    assert plan.status == "draft"
    assert plan.review == observed
    assert plan.issues
    with pytest.raises(ValueError, match="every candidate"):
        select_clips([plan], [item], 12, decisions=[choice(plan)])


def test_multiple_independent_hooks_are_a_blocking_clip_issue():
    item = event()
    observed = review(
        blocking_issues=[
            ReviewIssue(
                category="mixed_focus",
                description="The clip combines an identity reveal with a separate retaliation",
            )
        ]
    )
    plan = complete_review(draft_plan(item, 30), item, observed)
    assert plan.status == "draft"
    assert plan.issues == ["The clip combines an identity reveal with a separate retaliation"]


def choice(plan, selected=True, score=0.8, **kw):
    return SelectionChoice(
        event_id=plan.event_id, selected=selected, score=score, reason="Editorial judgment", **kw
    )


def test_stale_and_changed_evidence_cannot_be_selected():
    item = event()
    plan = ready(item)
    with pytest.raises(ValueError, match="stale_event_version"):
        select_clips([plan], [event(version=2)], None, decisions=[choice(plan)])
    item.required_spans.append(span(20, 22, "reaction"))
    with pytest.raises(ValueError, match="stale_required_spans"):
        select_clips([plan], [item], None, decisions=[choice(plan)])


def test_full_pool_explicit_subset_and_no_automatic_fill():
    items = [event(f"event_{i}", [span(i * 30, i * 30 + 5)]) for i in range(20)]
    plans = [ready(item) for item in items]
    decisions = [choice(p, i == 19) for i, p in enumerate(plans)]
    result = select_clips(plans, items, None, decisions=decisions)
    assert result.selected == [plans[-1]]
    assert len(result.decisions) == 20
    assert result.decisions[0].reason == "Editorial judgment"
    for invalid in (decisions[:-1], decisions[:-1] + [decisions[0]]):
        with pytest.raises(ValueError, match="every candidate"):
            select_clips(plans, items, None, decisions=invalid)
    assert not select_clips(
        plans, items, None, decisions=[choice(p, False) for p in plans]
    ).selected


def test_duplicate_content_disposition_does_not_merge_events():
    items = [event("first"), event("second")]
    plans = [ready(i) for i in items]
    result = select_clips(
        plans,
        items,
        None,
        decisions=[
            choice(plans[1]),
            choice(plans[0], False, duplicate_of=plans[1].event_id),
        ],
    )
    assert result.selected == [plans[1]]
    assert all(i.status == "supported" for i in items)
    with pytest.raises(ValueError, match="reference a selected"):
        select_clips(
            plans,
            items,
            None,
            decisions=[
                choice(plans[0], False, duplicate_of=plans[1].event_id),
                choice(plans[1], False),
            ],
        )


@pytest.mark.parametrize(
    "limits,match",
    [
        ({"max_highlights": 1}, "count budget"),
        ({"total_duration": 5}, "duration budget"),
        ({"allow_overlap": False}, "overlap constraint"),
    ],
)
def test_budget_violations_require_explicit_revision(limits, match):
    items = [event("first", [span(10, 20)]), event("second", [span(15, 25)])]
    plans = [ready(i) for i in items]
    with pytest.raises(ValueError, match=match):
        select_clips(
            plans, items, decisions=[choice(p) for p in plans], **{"max_highlights": None, **limits}
        )
    result = select_clips(plans, items, None, decisions=[choice(p) for p in reversed(plans)])
    assert result.selected == list(reversed(plans))
    assert [(p.start_sec, p.end_sec) for p in plans] == [(10, 20), (15, 25)]


def test_low_value_is_advisory_and_empty_selection_is_valid():
    item = event()
    plan = ready(item)
    assert plan.status == "ready"
    assert select_clips([plan], [item], None, decisions=[choice(plan, score=0)]).selected == [plan]
    assert not select_clips([plan], [item], 1, decisions=[choice(plan, False)]).selected
    with pytest.raises(ValueError, match="positive integer"):
        select_clips([plan], [item], 0, decisions=[choice(plan, False)])
    assert select_clips([], [], None, decisions=[]).model_dump() == {
        "selected": [],
        "decisions": [],
    }


def test_unknown_and_duplicate_event_pool_entries_are_errors():
    item = event()
    plan = ready(item)
    with pytest.raises(ValueError, match="unknown_event"):
        select_clips([plan], [], None, decisions=[choice(plan)])
    with pytest.raises(ValueError, match="one current version"):
        select_clips([], [item, event(version=2)], None, decisions=[])
    with pytest.raises(ValueError, match="Duplicate plan"):
        select_clips([plan, plan], [item], None, decisions=[choice(plan)])


def test_cannot_construct_a_ready_plan_without_a_passing_review():
    plan = draft_plan(event(), 30)
    with pytest.raises(ValidationError, match="accepted review"):
        ClipPlan.model_validate({**plan.model_dump(), "status": "ready"})


def test_actual_media_boundaries_invalidate_review_and_preserve_required_spans():
    item = event()
    plan = ready(item)
    rendered = attach_media(
        plan,
        start_sec=9.96,
        end_sec=15.04,
        media_path=Path("clip.mp4"),
        video_duration=30,
    )
    assert rendered.status == "draft"
    assert rendered.review is None
    assert rendered.id == plan.id
    assert rendered.media_path == Path("clip.mp4")
    assert (rendered.start_sec, rendered.end_sec) == (9.96, 15.04)
    assert rendered.required_spans == plan.required_spans
    assert complete_review(rendered, item, review()).status == "ready"


def test_frame_alignment_over_duration_limit_is_infeasible_without_trimming():
    plan = draft_plan(event(spans=[span(1, 25)]), 30)
    rendered = attach_media(
        plan,
        start_sec=0.96,
        end_sec=25.04,
        media_path=Path("clip.mp4"),
        video_duration=30,
    )
    assert rendered.status == "infeasible_duration"
    assert (rendered.start_sec, rendered.end_sec) == (0.96, 25.04)


def test_media_alignment_cannot_drop_context_or_reference_time_outside_video():
    plan = draft_plan(event(proposed_start_sec=5), 30)
    with pytest.raises(ValueError, match="discard"):
        attach_media(plan, start_sec=6, end_sec=15, media_path=Path("clip.mp4"), video_duration=30)
    with pytest.raises(ValueError, match="inside the video"):
        attach_media(plan, start_sec=0, end_sec=31, media_path=Path("clip.mp4"), video_duration=30)


@pytest.mark.parametrize(
    "updates",
    [
        {"reason": "   "},
        {"selected": True, "duplicate_of": "other"},
        {"selected": False, "duplicate_of": "event_1"},
    ],
)
def test_selection_rejects_empty_reason_and_invalid_duplicate_relationship(updates):
    with pytest.raises(ValidationError):
        SelectionChoice.model_validate(
            {"event_id": "event_1", "score": 0.8, "selected": True, "reason": "Useful", **updates}
        )
