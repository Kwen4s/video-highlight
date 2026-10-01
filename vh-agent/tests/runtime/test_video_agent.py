import base64
import json
from types import SimpleNamespace

import pytest

from vh_agent.runtime.agent import VideoAgent
from vh_agent.runtime.evidence import VideoObservation, VideoPage


def call(name, **args):
    return {"name": name, "args": args}


class FakeGemini:
    model = "test-model"
    endpoint = "https://example.invalid/v1beta/models/test-model:generateContent"

    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []

    def generate(self, system, parts, tools=None, **kwargs):
        self.requests.append({"system": system, "parts": parts, "tools": tools, **kwargs})
        if tools and tools[0]["name"] == "submit_review":
            function = call(
                "submit_review",
                visible_event="男子展示证件后获准进入。",
                highlight_type="identity_reveal",
                blocking_issues=[],
            )
        else:
            function = next(self.actions)
            if isinstance(function, Exception):
                raise function
        calls = function if isinstance(function, list) else ([function] if function else [])
        return {
            "function_calls": calls,
            "text": "" if function else "All done",
            "content": {
                "role": "model",
                "parts": [
                    {"functionCall": item, "thoughtSignature": "opaque-signature"} for item in calls
                ]
                if function
                else [{"text": "All done"}],
            },
            "usage": {},
            "model_version": "test-model",
        }


@pytest.fixture
def evidence(tmp_path):
    media = tmp_path / "media.mp4"
    media.write_bytes(b"test media payload")
    pages = [
        VideoPage(
            page_id=f"page_{i}",
            core_start_sec=i * 10,
            core_end_sec=(i + 1) * 10,
            read_start_sec=i * 10,
            read_end_sec=(i + 1) * 10,
        )
        for i in range(2)
    ]

    def observation(start, end, page_id=None):
        return VideoObservation(
            observation_id=f"obs_{page_id}" if page_id else f"obs_{start}_{end}",
            media_id="test_media",
            requested_start_sec=start,
            requested_end_sec=end,
            src_start_sec=start,
            src_end_sec=end,
            path=media,
            has_audio=True,
            duration_sec=end - start,
            page_id=page_id,
            source_start_time_sec=5,
            timestamp_precision_sec=0.04,
            cache_hit=False,
        )

    store = SimpleNamespace(
        media_id="test_media",
        video_info=SimpleNamespace(duration_sec=20),
        pages=pages,
        scan=lambda page_id: observation(
            0 if page_id == "page_0" else 10, 10 if page_id == "page_0" else 20, page_id
        ),
        inspect=observation,
        render_clip=observation,
    )
    store.restore_pages = lambda restored: setattr(store, "pages", restored)
    return store


def highlight_actions():
    event = {
        "id": "entry",
        "description": "男子展示证件后获准进入。",
        "reason": "证件改变阻拦决定。",
        "status": "supported",
        "required_spans": [
            {"start_sec": 3, "end_sec": 6, "evidence_id": "obs_page_0", "role": "decisive"},
        ],
    }
    return [
        call(
            "record_observations",
            observations=[
                {
                    "findings": [event],
                    "observation_id": "obs_page_0",
                },
                {
                    "findings": [],
                    "observation_id": "obs_page_1",
                    "no_event_reason": "无新增事件",
                },
            ],
        ),
        call(
            "select_highlights",
            decisions=[
                {
                    "event_id": "entry",
                    "score": 0.8,
                    "selected": True,
                    "reason": "开场冲突和身份反转都清楚，适合优先采用。",
                }
            ],
        ),
    ]


def test_native_tool_loop_reviews_and_publishes_same_media(evidence, tmp_path):
    actions = highlight_actions()
    actions[0]["args"]["observations"][0]["findings"][0]["description"] = (
        "男子展示伪造的证件后获准进入。"
    )
    actions[0]["args"]["observations"][0]["findings"][0]["reason"] = "伪造身份骗过保安。"
    actions[1]["args"]["decisions"][0]["score"] = 0.42
    client = FakeGemini(actions)
    result = VideoAgent(client, evidence, tmp_path).run()
    assert result["completion"] == "complete"
    assert result["analysis"]["scan_coverage"] == 1
    assert result["analysis"]["pending_event_count"] == 0
    assert len(result["highlights"]) == 1
    assert result["highlights"][0]["clip_path"] == "media.mp4"
    assert result["highlights"][0]["start_sec"] == 3
    assert result["highlights"][0]["reason"] == "开场冲突和身份反转都清楚，适合优先采用。"
    assert result["highlights"][0]["review_status"] == "accepted"
    # Public facts come from viewing the actual clip, not an earlier interpretation.
    assert result["highlights"][0]["description"] == "男子展示证件后获准进入。"
    assert result["highlights"][0]["highlight_type"] == "identity_reveal"
    assert result["highlights"][0]["score"] == 0.42
    # Review gets actual video in a fresh context; no event proposal or history leaks into it.
    review = next(r for r in client.requests if r["tools"][0]["name"] == "submit_review")
    assert "history" not in review
    assert "男子展示证件" not in json.dumps(review["parts"], ensure_ascii=False)
    assert any("inlineData" in part for part in review["parts"])
    second_agent_turn = client.requests[2]
    assert second_agent_turn["history"] is None
    assert not any("functionResponse" in part for part in second_agent_turn["parts"])
    current_state = json.loads(second_agent_turn["parts"][0]["text"])
    candidates = current_state["selection_ledger"]["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["event_id"] == "entry"
    assert candidates[0]["visible_event"] == "男子展示证件后获准进入。"
    assert "伪造" not in json.dumps(current_state, ensure_ascii=False)
    assert "伪造身份骗过保安" not in json.dumps(current_state, ensure_ascii=False)
    assert "working_events" not in current_state
    assert {tool["name"] for tool in second_agent_turn["tools"]} == {
        "select_highlights",
        "update_event",
        "inspect_interval",
        "read_state",
    }
    assert "opaque-signature" not in (tmp_path / "trace.jsonl").read_text()


def test_explicit_stop_and_resume_keeps_all_events(evidence, tmp_path):
    client = FakeGemini(highlight_actions())
    partial = VideoAgent(client, evidence, tmp_path).run(max_turns=1)
    assert partial["completion"] == "partial"
    assert partial["analysis"]["pending_event_count"] == 0
    assert partial["analysis"]["candidate_count"] == 1
    assert partial["analysis"]["stop_reason"] == "turn_limit"
    result = VideoAgent(client, evidence, tmp_path).run(resume=True)
    assert result["completion"] == "complete"
    assert len(result["highlights"]) == 1


def test_partial_retains_reviewed_plans_without_publishing_unselected_clips(evidence, tmp_path):
    result = VideoAgent(FakeGemini(highlight_actions()), evidence, tmp_path).run(max_turns=1)
    assert result["completion"] == "partial"
    assert result["analysis"]["scan_coverage"] == 1
    assert result["highlights"] == []
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["tools"]["plans"][0]["status"] == "draft"
    assert state["tools"]["plans"][0]["review"] is not None


def test_failed_delivery_does_not_mark_page_observed_and_can_resume(evidence, tmp_path):
    client = FakeGemini([TimeoutError("network timeout")])
    partial = VideoAgent(client, evidence, tmp_path).run()
    assert partial["completion"] == "partial"
    assert partial["analysis"]["scan_coverage"] == 0
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["tools"]["delivered"] == []
    assert saved["pending_observation_ids"] == ["obs_page_0", "obs_page_1"]
    no_events = call(
        "record_observations",
        observations=[
            {"observation_id": "obs_page_0", "no_event_reason": "无事件"},
            {"observation_id": "obs_page_1", "no_event_reason": "无事件"},
        ],
    )
    resumed_client = FakeGemini([no_events, call("select_highlights", decisions=[])])
    result = VideoAgent(resumed_client, evidence, tmp_path).run(resume=True)
    assert result["completion"] == "complete"
    assert result["highlights"] == []
    assert any("inlineData" in p for p in resumed_client.requests[0]["parts"])


def test_no_progress_loop_stops_as_partial(evidence, tmp_path):
    bad = call("read_state", event_id="unknown", collection="events")
    result = VideoAgent(FakeGemini([bad] * 3), evidence, tmp_path, max_stagnant_steps=3).run()
    assert result["completion"] == "partial"
    assert result["analysis"]["stop_reason"] == "repeated_action_without_progress"
    assert result["analysis"]["turns"] == 3


def test_text_completion_cannot_forge_task_complete(evidence, tmp_path):
    result = VideoAgent(FakeGemini([None]), evidence, tmp_path).run()
    assert result["completion"] == "partial"
    assert result["analysis"]["stop_reason"] == "missing_tool_call"
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert records[-1] == {"kind": "missing_tool_call", "turn": 1, "text": "All done"}


def test_resume_rejects_changed_model_or_constraints(evidence, tmp_path):
    client = FakeGemini(highlight_actions()[:1])
    VideoAgent(client, evidence, tmp_path).run(max_turns=1)
    with pytest.raises(ValueError, match="same video"):
        VideoAgent(client, evidence, tmp_path, video_fps=8).run(resume=True)


def test_existing_output_is_not_overwritten(evidence, tmp_path):
    client = FakeGemini([None])
    VideoAgent(client, evidence, tmp_path).run()
    with pytest.raises(ValueError, match="--resume"):
        VideoAgent(client, evidence, tmp_path).run()


def test_reading_observations_does_not_erase_working_events(evidence, tmp_path):
    actions = highlight_actions()[:1]
    actions[0]["args"]["observations"][0]["findings"][0]["status"] = "pending"
    client = FakeGemini(actions)
    agent = VideoAgent(client, evidence, tmp_path)
    agent.run(max_turns=1)
    context = agent._context()
    assert context["working_events"][0]["event"]["id"] == "entry"
    assert context["source_observations"][0]["observation_id"] == "obs_page_0"
    assert context["source_observations"][0]["record"]["event_ids"] == ["entry"]
    resumed = VideoAgent(FakeGemini([]), evidence, tmp_path)
    resumed._restore()
    assert resumed._context()["working_events"] == context["working_events"]


def test_context_refreshes_and_acknowledged_video_is_not_added_again(evidence, tmp_path):
    client = FakeGemini(highlight_actions())
    agent = VideoAgent(client, evidence, tmp_path)
    agent.run(max_turns=2)
    # Independent reviews must be visible without an extra read_state call.
    agent_requests = [
        request for request in client.requests if request["tools"][0]["name"] != "submit_review"
    ]
    second = agent_requests[1]
    assert second["history"] is None
    current = json.loads(second["parts"][0]["text"])
    assert current["progress"]["scan_coverage"] == 1
    assert current["selection_ledger"]["candidates"][0]["visible_event"]
    assert current["phase"] == "selection"
    assert not any("inlineData" in p for p in second["parts"])


def test_main_agent_can_repair_semantic_mismatch_after_clean_review(evidence, tmp_path):
    actions = highlight_actions()
    corrected = dict(actions[0]["args"]["observations"][0]["findings"][0])
    corrected.update(
        description="男子展示证件后获准进入，纠正了最初的人物身份判断。",
        expected_version=1,
    )
    client = FakeGemini([actions[0], call("update_event", event=corrected), actions[1]])
    result = VideoAgent(client, evidence, tmp_path).run()
    assert result["completion"] == "complete"
    assert result["highlights"][0]["description"] == "男子展示证件后获准进入。"
    reviews = [r for r in client.requests if r["tools"][0]["name"] == "submit_review"]
    assert len(reviews) == 2
    final_state = json.loads(client.requests[-1]["parts"][0]["text"])
    candidate = final_state["selection_ledger"]["candidates"][0]
    assert candidate["event_version"] == 2
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["tools"]["events"][0]["description"] == corrected["description"]
    assert candidate["visible_event"] == result["highlights"][0]["description"]


def test_selection_recheck_keeps_media_and_facts_after_registration_and_resume(evidence, tmp_path):
    actions = highlight_actions()
    note = "核对证件展示，没有看到伪造行为；没有新增候选。"
    client = FakeGemini(
        [
            actions[0],
            call("inspect_interval", start_sec=3, end_sec=8, question="核对证件是否伪造"),
            call(
                "record_observations",
                observations=[
                    {
                        "observation_id": "obs_3.0_8.0",
                        "no_event_reason": note,
                    }
                ],
            ),
            actions[1],
        ]
    )
    partial = VideoAgent(client, evidence, tmp_path).run(max_turns=2)
    assert partial["completion"] == "partial"
    complete = VideoAgent(client, evidence, tmp_path).run(resume=True)
    assert complete["completion"] == "complete"
    final = client.requests[-1]
    context = json.loads(final["parts"][0]["text"])
    assert context["phase"] == "selection"
    assert any("inlineData" in p for c in final["history"] for p in c["parts"])
    assert context["inspection_memory"]["records"][0]["record"]["no_event_reason"] == note
    assert context["inspection_memory"]["total"] == 1


def test_selection_repair_keeps_session_and_exposes_latest_review_issues(evidence, tmp_path):
    actions = highlight_actions()
    original = actions[0]["args"]["observations"][0]["findings"][0]
    corrected = {**original, "expected_version": 1, "proposed_start_sec": 2}
    repaired = {**corrected, "expected_version": 2, "proposed_start_sec": 1}

    class RepairClient(FakeGemini):
        review_count = 0

        def generate(self, system, parts, tools=None, **kwargs):
            reply = super().generate(system, parts, tools=tools, **kwargs)
            if tools[0]["name"] == "submit_review":
                self.review_count += 1
                if self.review_count == 2:
                    reply["function_calls"][0]["args"]["blocking_issues"] = [
                        {
                            "category": "cutoff",
                            "description": "开头台词被截断",
                            "at_sec": 0,
                        }
                    ]
            return reply

    client = RepairClient(
        [
            actions[0],
            call("update_event", event=corrected),
            call("update_event", event=repaired),
            actions[1],
        ]
    )
    result = VideoAgent(client, evidence, tmp_path).run()
    assert result["completion"] == "complete"
    request = [r for r in client.requests if r["tools"][0]["name"] != "submit_review"][2]
    context = json.loads(request["parts"][0]["text"])
    assert context["phase"] == "selection"
    assert request["history"]
    issue = context["repair_events"][0]["clip"]["review"]["blocking_issues"][0]
    assert issue["category"] == "cutoff"
    assert context["repair_events"][0]["event"]["version"] == 2


def test_failed_parallel_review_preserves_success_and_resume_only_reviews_missing(
    evidence, tmp_path, monkeypatch
):
    from vh_agent.providers.gemini_client import GeminiClientError

    actions = highlight_actions()
    second_event = {
        **actions[0]["args"]["observations"][0]["findings"][0],
        "id": "second_entry",
        "required_spans": [
            {"start_sec": 13, "end_sec": 16, "evidence_id": "obs_page_1", "role": "decisive"}
        ],
    }
    actions[0]["args"]["observations"][1] = {
        "observation_id": "obs_page_1",
        "findings": [second_event],
    }
    actions[1]["args"]["decisions"].append(
        {"event_id": "second_entry", "score": 0.8, "selected": True, "reason": "另一个独立看点。"}
    )

    def render(start, end):
        path = tmp_path / f"clip_{start}.mp4"
        path.write_bytes(str(start).encode())
        return evidence.inspect(start, end).model_copy(update={"path": path})

    monkeypatch.setattr(evidence, "render_clip", render)

    class FailingReviewClient(FakeGemini):
        def __init__(self):
            super().__init__(actions)
            self.reviewed = []

        def generate(self, system, parts, tools=None, **kwargs):
            if tools[0]["name"] == "submit_review":
                start = float(base64.b64decode(parts[1]["inlineData"]["data"]))
                self.reviewed.append(start)
                if start == 13 and self.reviewed.count(13) == 1:
                    raise GeminiClientError("Review request failed")
            return super().generate(system, parts, tools=tools, **kwargs)

    client = FailingReviewClient()
    partial = VideoAgent(client, evidence, tmp_path).run()
    assert partial["completion"] == "partial"
    saved = json.loads((tmp_path / "state.json").read_text())
    plans = {plan["event_id"]: plan for plan in saved["tools"]["plans"]}
    assert plans["entry"]["review"] is not None
    assert plans["second_entry"]["review"] is None

    complete = VideoAgent(client, evidence, tmp_path).run(resume=True)
    assert complete["completion"] == "complete"
    assert len(complete["highlights"]) == 2
    assert sorted(client.reviewed) == [3, 13, 13]


def test_video_delivery_exposes_only_the_matching_record_tool(evidence, tmp_path):
    client = FakeGemini(highlight_actions()[:1])
    VideoAgent(client, evidence, tmp_path).run(max_turns=1)
    declarations = client.requests[0]["tools"]
    assert [declaration["name"] for declaration in declarations] == ["record_observations"]
    schema = declarations[0]["parametersJsonSchema"]
    observation_schema = schema["$defs"]["ObservationRecord"]
    assert observation_schema["properties"]["observation_id"]["enum"] == [
        "obs_page_0",
        "obs_page_1",
    ]
    event_fields = schema["$defs"]["EventInput"]["properties"]
    assert "expected_version" in event_fields
    assert "version" not in event_fields
    assert client.requests[0]["tool_config"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["record_observations"],
    }


def test_multiple_tool_calls_are_rejected_as_one_model_action(evidence, tmp_path):
    batch = highlight_actions()[0]
    client = FakeGemini([[batch, batch]])
    agent = VideoAgent(client, evidence, tmp_path)
    result = agent.run(max_turns=1)
    assert result["completion"] == "partial"
    assert len(agent.tools.observations) == 2
    assert agent.tools.completed_pages == set()
    assert agent.progress_memory.stagnant_steps == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["pending_calls"] == []
    assert state["reject_pending_batch"] is False
    traces = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in traces].count("tool_batch_rejected") == 1
    assert not any(row["kind"] == "tool" for row in traces)


def test_validation_error_returns_field_paths_without_dumping_inputs(evidence, tmp_path):
    no_events = call(
        "record_observations",
        observations=[
            {"observation_id": "obs_page_0", "no_event_reason": "无事件"},
            {"observation_id": "obs_page_1", "no_event_reason": "无事件"},
        ],
    )
    client = FakeGemini([no_events, call("inspect_interval", start_sec=0, question="查看结尾")])
    agent = VideoAgent(client, evidence, tmp_path)
    agent.run(max_turns=2)
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    result = next(row["result"] for row in trace if row["kind"] == "tool" and row["error"])
    assert result["status"] == "invalid_tool_request"
    assert result["fields"] == [{"path": "end_sec", "code": "missing", "message": "Field required"}]
    assert "input_value" not in json.dumps(result)
    assert "errors.pydantic.dev" not in json.dumps(result)


def test_analysis_history_survives_resume_and_selection_starts_fresh(
    evidence, tmp_path, monkeypatch
):
    monkeypatch.setattr("vh_agent.runtime.agent.MAX_PAGE_BATCH_ITEMS", 1)
    actions = highlight_actions()
    records = actions[0]["args"]["observations"]
    client = FakeGemini(
        [
            call("record_observations", observations=[records[0]]),
            call("record_observations", observations=[records[1]]),
            actions[1],
        ]
    )
    VideoAgent(client, evidence, tmp_path).run(max_turns=2)
    agent_requests = [
        request for request in client.requests if request["tools"][0]["name"] != "submit_review"
    ]
    analysis_request = agent_requests[-1]
    models = [c for c in analysis_request["history"] if c["role"] == "model"]
    assert models
    assert all(c["parts"][0]["thoughtSignature"] == "opaque-signature" for c in models)
    state = json.loads((tmp_path / "state.json").read_text())
    assert "inlineData" not in json.dumps(state)
    assert "video_ref" in json.dumps(state["conversation"])
    assert state["pending_calls"] == []
    assert state["conversation"]["phase"] == "analysis"
    result = VideoAgent(client, evidence, tmp_path).run(resume=True)
    assert result["completion"] == "complete"
    assert client.requests[-1]["history"] is None
    assert json.loads(client.requests[-1]["parts"][0]["text"])["phase"] == "selection"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["conversation"]["phase"] == "selection"
    assert len(state["tools"]["observations"]) == 2


def test_selection_request_resume_reuses_exact_fresh_view(evidence, tmp_path):
    actions = highlight_actions()
    client = FakeGemini([actions[0], TimeoutError("selection interrupted")])
    result = VideoAgent(client, evidence, tmp_path).run()
    assert result["completion"] == "partial"
    failed = client.requests[-1]
    assert failed["history"] is None
    assert json.loads(failed["parts"][0]["text"])["phase"] == "selection"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["request_ready"]
    assert state["conversation"]["phase"] == "selection"
    resumed_client = FakeGemini([actions[1]])
    result = VideoAgent(resumed_client, evidence, tmp_path).run(resume=True)
    assert result["completion"] == "complete"
    assert resumed_client.requests[0] == failed


def test_resume_keeps_stagnation_instead_of_resetting_it(evidence, tmp_path):
    bad = call("read_state", collection="events", event_id="missing")
    VideoAgent(FakeGemini([bad, bad]), evidence, tmp_path, max_stagnant_steps=3).run(max_turns=2)
    client = FakeGemini([bad])
    result = VideoAgent(client, evidence, tmp_path, max_stagnant_steps=3).run(resume=True)
    assert result["analysis"]["stop_reason"] == "repeated_action_without_progress"
    assert result["analysis"]["turns"] == 3
    assert len(client.requests) == 1


def test_committed_selection_is_not_executed_again_after_interruption(
    evidence, tmp_path, monkeypatch
):
    agent = VideoAgent(FakeGemini(highlight_actions()), evidence, tmp_path)
    original_trace = agent._trace

    def interrupt_after_commit(kind, **data):
        original_trace(kind, **data)
        if kind == "tool" and data["name"] == "select_highlights":
            raise KeyboardInterrupt()

    monkeypatch.setattr(agent, "_trace", interrupt_after_commit)
    partial = agent.run()
    assert partial["analysis"]["stop_reason"] == "user_stopped"
    client = FakeGemini([])
    resumed = VideoAgent(client, evidence, tmp_path)
    assert resumed.run(resume=True)["completion"] == "complete"
    assert resumed.tools.plans["entry"].status == "ready"
    traces = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len([row for row in traces if row.get("name") == "select_highlights"]) == 1


def test_received_call_survives_stop_before_execution(evidence, tmp_path, monkeypatch):
    agent = VideoAgent(FakeGemini(highlight_actions()[:1]), evidence, tmp_path)
    monkeypatch.setattr(
        agent.tools, "execute", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    agent.run()
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["pending_calls"][0]["name"] == "record_observations"
    client = FakeGemini(highlight_actions()[1:])
    resumed = VideoAgent(client, evidence, tmp_path)
    resumed.run(resume=True, max_turns=1)
    assert resumed.tools.progress()["scan_coverage"] == 1
    assert resumed.turn == 2


def test_long_scan_rotates_context_without_losing_coverage(evidence, tmp_path):
    evidence.pages = [
        VideoPage(
            page_id=f"page_{i}",
            core_start_sec=i * 10,
            core_end_sec=(i + 1) * 10,
            read_start_sec=i * 10,
            read_end_sec=(i + 1) * 10,
        )
        for i in range(30)
    ]
    evidence.video_info.duration_sec = 300

    def scan(page_id):
        page = next(p for p in evidence.pages if p.page_id == page_id)
        return evidence.inspect(page.read_start_sec, page.read_end_sec, page_id)

    evidence.scan = scan
    actions = []
    for offset in range(0, len(evidence.pages), 2):
        actions.append(
            call(
                "record_observations",
                observations=[
                    {
                        "observation_id": f"obs_{page.page_id}",
                        "no_event_reason": "无事件",
                    }
                    for page in evidence.pages[offset : offset + 2]
                ],
            )
        )
    actions.append(call("select_highlights", decisions=[]))
    client = FakeGemini(actions)
    agent = VideoAgent(client, evidence, tmp_path)
    result = agent.run()
    assert result["completion"] == "complete"
    assert result["analysis"]["scan_coverage"] == 1
    assert len(agent.tools.acknowledged) == 30
    assert agent.conversation.segment > 1
    assert agent.pending_calls == []


def test_transient_retry_uses_identical_request_and_does_not_repeat_tools(
    evidence, tmp_path, monkeypatch
):
    from vh_agent.providers.gemini_client import GeminiClientError

    monkeypatch.setattr("vh_agent.runtime.agent.time.sleep", lambda _: None)
    actions = highlight_actions()
    client = FakeGemini([GeminiClientError("temporary empty response", retryable=True), *actions])
    result = VideoAgent(client, evidence, tmp_path).run()
    assert result["completion"] == "complete"
    assert client.requests[0] == client.requests[1]
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len([r for r in trace if r.get("name") == "record_observations"]) == 1
    assert len([r for r in trace if r["kind"] == "request_error"]) == 1


def test_request_retry_budget_stops_and_retains_exact_pending_request(
    evidence, tmp_path, monkeypatch
):
    from vh_agent.providers.gemini_client import GeminiClientError

    monkeypatch.setattr("vh_agent.runtime.agent.time.sleep", lambda _: None)
    client = FakeGemini([GeminiClientError("temporary", retryable=True)] * 2)
    result = VideoAgent(client, evidence, tmp_path, max_request_attempts=2).run()
    assert result["completion"] == "partial"
    assert result["analysis"]["model_calls"] == 2
    assert client.requests[0] == client.requests[1]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["request_ready"]


def test_completed_resume_reuses_committed_choice_without_model_calls(evidence, tmp_path):
    first = VideoAgent(FakeGemini(highlight_actions()), evidence, tmp_path).run()
    assert first["completion"] == "complete"
    client = FakeGemini([])
    resumed = VideoAgent(client, evidence, tmp_path).run(resume=True)
    assert resumed["highlights"] == first["highlights"]
    assert not client.requests


def test_crash_after_selection_commit_before_export_resumes_without_reselection(
    evidence, tmp_path, monkeypatch
):
    from vh_agent.runtime import agent as runtime

    original = runtime.write_json

    def interrupt_export(path, value):
        if path.name == "selection.json":
            raise KeyboardInterrupt
        original(path, value)

    monkeypatch.setattr(runtime, "write_json", interrupt_export)
    with pytest.raises(KeyboardInterrupt):
        VideoAgent(FakeGemini(highlight_actions()), evidence, tmp_path).run()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["tools"]["finished"]
    assert state["tools"]["selection"]["selected"][0]["id"] == "clip_entry"
    monkeypatch.setattr(runtime, "write_json", original)
    client = FakeGemini([])
    result = VideoAgent(client, evidence, tmp_path).run(resume=True)
    assert result["completion"] == "complete"
    assert [h["id"] for h in result["highlights"]] == ["clip_entry"]
    assert not client.requests
