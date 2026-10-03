import json

import pytest

from vh_agent.runtime.context import Conversation, estimate, hydrate, image_part, text_part
from vh_agent.runtime.state import ProgressMemory, ReadMemory


def test_rotation_keeps_facts_without_orphan_native_outputs():
    history = [
        {"role": "user", "content": [text_part("x" * 8000)]},
        {"type": "function_call", "call_id": "call_1", "name": "search_video", "arguments": "{}"},
    ]
    conversation = Conversation(history)
    output = {"type": "function_call_output", "call_id": "call_1", "output": "a"}
    facts = {"last_tool_results": [output], "story_so_far": "已有剧情"}
    result = conversation.prepare(facts, [output], token_budget=2000, byte_budget=2000)
    assert len(result) == 1
    assert json.loads(result[0]["content"][0]["text"]) == facts
    assert conversation.segment == 1 and estimate(result)[0] < 2000


def test_native_encrypted_reasoning_and_call_ids_survive_checkpoint():
    conversation = Conversation(
        [
            {"type": "reasoning", "encrypted_content": "opaque"},
            {"type": "function_call", "call_id": "call_1", "name": "read_state", "arguments": "{}"},
        ]
    )
    restored = Conversation(**json.loads(json.dumps(conversation.checkpoint())))
    result = restored.prepare(
        {},
        [{"type": "function_call_output", "call_id": "call_1", "output": "{}"}],
        token_budget=2000,
        byte_budget=2000,
    )
    assert result[0]["encrypted_content"] == "opaque" and result[2]["call_id"] == "call_1"


def test_native_tool_result_is_sent_once_until_history_rotates():
    conversation = Conversation(
        [{"type": "function_call", "call_id": "call_1", "name": "read_state", "arguments": "{}"}]
    )
    result = {"events": [{"id": "event_1"}]}
    output = {"type": "function_call_output", "call_id": "call_1", "output": json.dumps(result)}
    context = {
        "story_so_far": "当前剧情",
        "last_tool_results": [{"tool": "read_state", "result": result}],
    }
    request = conversation.prepare(context, [output], token_budget=2000, byte_budget=2000)
    assert request[1] == output
    assert "last_tool_results" not in json.loads(request[-1]["content"][0]["text"])


def test_over_budget_does_not_truncate_committed_facts():
    conversation = Conversation()
    with pytest.raises(ValueError, match="预算"):
        conversation.prepare({"facts": "x" * 5000}, [], token_budget=1000, byte_budget=1000)
    assert conversation.contents == []


def test_image_reference_is_durable_and_detects_changed_media(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"image")
    items = [{"role": "user", "content": [image_part(path)]}]
    assert hydrate(items)[0]["content"][0]["image_url"].startswith("data:image/jpeg;base64,")
    assert "image_ref" in items[0]["content"][0]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="已改变"):
        hydrate(items)


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
