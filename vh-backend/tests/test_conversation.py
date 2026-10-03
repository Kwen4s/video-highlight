from app.config import Settings
from app.conversation import (
    EditPlan,
    HighlightConversationAgent,
    QueryPlan,
    execute_plan,
)
from app.models import DetectionResult


def sample_result() -> DetectionResult:
    return DetectionResult.model_validate(
        {
            "job_id": "job_conversation1",
            "completion": "complete",
            "message": "done",
            "analysis": dict(
                scan_coverage=1,
                pending_event_count=0,
                pending_observation_count=0,
                pending_proposal_count=0,
                pending_review_count=0,
                stop_reason="complete",
                model_calls=1,
            ),
            "video": {"video_id": "video_1", "title": "demo", "duration_sec": 60},
            "highlights": [
                {
                    "highlight_id": "hl_first",
                    "start_sec": 2,
                    "end_sec": 8,
                    "score": 0.91,
                    "highlight_type": "reversal",
                    "description": "真相揭晓",
                    "reason": "决定性证据改变了人物认知",
                },
                {
                    "highlight_id": "hl_second",
                    "start_sec": 8,
                    "end_sec": 13,
                    "score": 0.82,
                    "highlight_type": "emotion",
                    "description": "人物回应",
                    "reason": "人物给出了明确的情绪反应",
                },
            ],
        }
    )


class RecordingPlanner:
    def __init__(self, plan) -> None:
        self.next_plan = plan
        self.conversation = []

    def plan(self, **kwargs):
        self.conversation = kwargs["conversation"]
        return self.next_plan


def test_agent_uses_typed_query_plan_and_receives_recent_conversation(tmp_path) -> None:
    planner = RecordingPlanner(
        QueryPlan(kind="query", operation="details", highlight_ids=["hl_first"])
    )
    agent = HighlightConversationAgent(
        Settings(VH_STORAGE_DIR=tmp_path / "runtime"),
        planner=planner,
    )
    history = [
        {
            "user_message": "列出高光",
            "assistant_reply": "当前有两段",
            "action": {"kind": "query", "operation": "list"},
        }
    ]

    outcome = agent.respond(
        result=sample_result(),
        message="为什么第一段是高光？",
        selected_highlight_id="hl_first",
        conversation=history,
    )

    assert outcome.changed is False
    assert "决定性证据" in outcome.reply
    assert planner.conversation == history


def test_split_and_merge_are_validated_deterministic_edits() -> None:
    split = execute_plan(
        EditPlan(
            kind="edit",
            operation="split",
            highlight_ids=["hl_first"],
            split_sec=5,
        ),
        sample_result(),
        [],
    )

    assert split.changed is True
    assert len(split.result.highlights) == 3
    assert split.result.highlights[0].end_sec == 5
    assert split.result.highlights[1].start_sec == 5
    right_id = split.result.highlights[1].highlight_id

    merged = execute_plan(
        EditPlan(
            kind="edit",
            operation="merge",
            highlight_ids=["hl_first", right_id],
            text="完整反转",
        ),
        split.result,
        [],
    )

    assert merged.changed is True
    assert len(merged.result.highlights) == 2
    assert merged.result.highlights[0].start_sec == 2
    assert merged.result.highlights[0].end_sec == 8
    assert merged.result.highlights[0].description == "完整反转"


def test_invalid_planner_target_is_returned_as_clarification(tmp_path) -> None:
    planner = RecordingPlanner(
        EditPlan(
            kind="edit",
            operation="delete",
            highlight_ids=["hl_missing"],
        )
    )
    agent = HighlightConversationAgent(
        Settings(VH_STORAGE_DIR=tmp_path / "runtime"),
        planner=planner,
    )

    outcome = agent.respond(
        result=sample_result(),
        message="删掉那一段",
        selected_highlight_id=None,
        conversation=[],
    )

    assert outcome.changed is False
    assert "找不到" in outcome.reply
