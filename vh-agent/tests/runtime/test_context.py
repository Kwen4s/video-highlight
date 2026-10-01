import json

import pytest

from vh_agent.providers.gemini_client import text_part
from vh_agent.runtime.context import Conversation, estimate
from vh_agent.runtime.state import ProgressMemory, ReadMemory


def test_rotation_keeps_committed_facts_without_orphan_function_response():
    conversation = Conversation(
        [
            {"role": "user", "parts": [text_part("x" * 8000)]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "search_transcript"}, "thoughtSignature": "original"}
                ],
            },
        ]
    )
    response = {"functionResponse": {"name": "search_transcript", "response": {"matches": ["a"]}}}
    facts = {"transcript_memory": {"rows": ["a"]}, "last_tool_results": [response]}
    contents = conversation.prepare(facts, [response], token_budget=2000, byte_budget=2000)
    assert len(contents) == 1
    assert json.loads(contents[0]["parts"][0]["text"]) == facts
    assert all("functionResponse" not in p for c in contents for p in c["parts"])
    assert conversation.segment == 1
    assert estimate(contents)[0] < 2000


def test_single_context_over_budget_fails_without_silently_truncating():
    conversation = Conversation()
    with pytest.raises(ValueError, match="预算"):
        conversation.prepare({"fact": "x" * 5000}, [], token_budget=1000, byte_budget=1000)
    assert conversation.contents == []


def test_query_coverage_is_by_rows_not_last_page_and_survives_restore():
    memory = ReadMemory()
    args = {"query": "", "start_sec": 0, "end_sec": None, "offset": 1, "limit": 1}
    last = memory.record(
        args, {"matches": [{"row_id": "b", "text": "B"}], "total": 2, "next_offset": None}
    )
    assert not last["query_complete"]
    memory = ReadMemory(**json.loads(json.dumps(memory.checkpoint())))
    first = memory.record(
        {**args, "offset": 0},
        {"matches": [{"row_id": "a", "text": "A"}], "total": 2, "next_offset": 1},
    )
    assert first["query_complete"]
    repeated = memory.record(
        {**args, "offset": 0, "limit": 100},
        {
            "matches": [{"row_id": "a", "text": "A"}, {"row_id": "b", "text": "B"}],
            "total": 2,
            "next_offset": None,
        },
    )
    assert repeated["new_row_ids"] == []
    assert len(memory.queries) == 1
    assert set(memory.rows) == {"a", "b"}


def test_progress_oscillation_and_restore_do_not_create_new_information():
    progress = ProgressMemory()
    assert progress.record({"event:pending"}) == 1
    assert progress.record({"event:supported"}) == 1
    assert progress.record({"event:pending"}) == 0
    resumed = ProgressMemory.restore(progress.checkpoint())
    assert resumed.record({"event:supported"}) == 0
    assert resumed.stagnant_steps == 2


def test_rotation_prioritizes_requested_media_and_reports_nonresident_observations(tmp_path):
    from types import SimpleNamespace

    from vh_agent.runtime.context import media_part

    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    old = SimpleNamespace(path=path, observation_id="old", sampling_fps=4, duration_sec=33)
    new = SimpleNamespace(path=path, observation_id="new", sampling_fps=4, duration_sec=17)
    conversation = Conversation([{"role": "user", "parts": [media_part(old, 4)]}])
    result = conversation.prepare(
        {"unrecorded": ["old", "new"]},
        [media_part(new, 4)],
        token_budget=90_000,
        byte_budget=18_000_000,
        replay_parts=[media_part(old, 4)],
    )
    refs = [
        p["video_ref"]["observation_id"]
        for p in conversation.contents[0]["parts"]
        if "video_ref" in p
    ]
    assert refs == ["new"]
    note = json.loads(result[0]["parts"][-1]["text"])
    assert note["read_again_observation_ids"] == ["old"]
    assert note["visible_observation_ids"] == ["new"]
