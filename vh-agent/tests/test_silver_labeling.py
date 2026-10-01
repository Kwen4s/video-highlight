from vh_agent.models import DetectionResult
from vh_agent.silver_labeling import _silver_record


def test_silver_uses_exact_event_provenance_not_overlap():
    result = DetectionResult.model_validate(
        {
            "job_id": "test",
            "video": {"video_id": "v", "title": "test", "duration_sec": 20},
            "completion": "complete",
            "message": "done",
            "analysis": {
                "scan_coverage": 1,
                "pending_event_count": 0,
                "pending_observation_count": 0,
                "pending_proposal_count": 0,
                "pending_review_count": 0,
                "stop_reason": "complete",
                "model_calls": 3,
            },
            "highlights": [
                {
                    "highlight_id": "h",
                    "start_sec": 1,
                    "end_sec": 8,
                    "score": 0.8,
                    "highlight_type": "reveal",
                    "description": "event",
                    "reason": "seen",
                    "clip_url": "/clip",
                }
            ],
        }
    )
    spans = [{"start_sec": 3, "end_sec": 5, "role": "decisive", "evidence_id": "o1"}]
    row = _silver_record(
        {"video_id": "v"},
        result,
        {
            "profile": {"model": "gemini"},
            "tools": {
                "events": [
                    {"id": "e1", "required_spans": spans},
                    {"id": "e2", "required_spans": []},
                ]
            },
        },
        {"h": {"event_id": "e1"}},
    )
    assert row["highlights"][0]["evidence"] == spans
    assert row["highlights"][0]["decisive_times_sec"] == [4]
    assert row["annotation_status"] == "silver"
