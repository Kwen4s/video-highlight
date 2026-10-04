"""End-to-end state machine checks with separate controller and video observers."""

import copy
import json
from types import SimpleNamespace

import pytest

from vh_agent.providers.gemini_client import GeminiClientError
from vh_agent.providers.openai_client import OpenAIClientError
from vh_agent.providers.video_perception import VideoPerception
from vh_agent.runtime.agent import VideoAgent
from vh_agent.runtime.evidence import VideoObservation, VideoPage


def call(name, **args):
    return {"name": name, "args": args}


class Controller:
    model, effort, endpoint = "test-sol", "high", "https://example.invalid/v1/responses"

    def __init__(self, actions):
        self.actions, self.requests = iter(actions), []

    def generate(self, system, inputs, *, tools, **kwargs):
        self.requests.append(
            copy.deepcopy({"system": system, "inputs": inputs, "tools": tools, **kwargs})
        )
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        calls = action if isinstance(action, list) else ([action] if action else [])
        calls = [{**c, "id": f"call_{len(self.requests)}_{i}"} for i, c in enumerate(calls)]
        return {
            "function_calls": calls,
            "output": [
                {
                    "type": "reasoning",
                    "id": "r",
                    "summary": [],
                    "encrypted_content": "opaque-reasoning",
                },
                *[
                    {
                        "type": "function_call",
                        "call_id": c["id"],
                        "name": c["name"],
                        "arguments": json.dumps(c["args"]),
                    }
                    for c in calls
                ],
            ],
            "usage": {},
            "model_version": self.model,
            "response_id": "response_test",
            "first_event_sec": 0.1,
        }


class Observer:
    seed = 7
    thinking_level = "high"
    model, endpoint = "test-video", "https://example.invalid/generateContent"

    def __init__(self, uncertainties=None, review_issues=None):
        self.requests, self.fail_review, self.fail_observation = [], None, None
        self.uncertainties = uncertainties or []
        self.review_issues = review_issues or []

    def generate(self, system, parts, *, schema, **kwargs):
        self.requests.append({"system": system, "parts": parts, "schema": schema, **kwargs})
        if schema["title"] == "ReviewResult":
            if self.fail_review:
                failure, self.fail_review = self.fail_review, None
                raise failure
            args = {
                "visible_event": "男子展示证件后获准进入。",
                "issues": self.review_issues,
            }
        else:
            if self.fail_observation:
                failure, self.fail_observation = self.fail_observation, None
                raise failure
            origin = json.loads(parts[0]["text"])["src_start_sec"]
            args = {
                "answer": "原片中可见人物动作。",
                "items": [
                    {
                        "start_sec": origin,
                        "end_sec": origin + 1,
                        "kind": "action",
                        "content": "人物行动",
                        "speaker": None,
                    }
                ],
                "uncertainties": self.uncertainties,
            }
        return {"json": args, "usage": {}, "model_version": self.model, "thinking_returned": True}


@pytest.fixture
def evidence(tmp_path):
    media = tmp_path / "media.mp4"
    media.write_bytes(b"video")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
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
        scan=lambda key: observation(
            int(key.split("_")[1]) * 10, (int(key.split("_")[1]) + 1) * 10, key
        ),
        inspect=observation,
        render_clip=observation,
        frames=lambda times, **kw: [
            {"path": str(frame), "timestamp_sec": t, "region": None} for t in times
        ],
    )
    store.restore_pages = lambda restored: setattr(store, "pages", restored)
    return store


def actions():
    entries = [
        call(
            "record_observations",
            observations=[
                {
                    "observation_id": "obs_page_0",
                    "findings": [
                        {
                            "id": "entry",
                            "description": "男子展示证件后获准进入。",
                            "reason": "证件改变阻拦决定。",
                            "status": "supported",
                            "required_spans": [
                                {
                                    "start_sec": 3,
                                    "end_sec": 6,
                                    "evidence_id": "obs_page_0",
                                    "role": "decisive",
                                }
                            ],
                        }
                    ],
                },
                {"observation_id": "obs_page_1", "no_event_reason": "无新增事件"},
            ],
            story_so_far="男子被阻拦，展示证件后获准进入。",
        ),
        call(
            "select_highlights",
            decisions=[
                {
                    "event_id": "entry",
                    "score": 0.8,
                    "selected": True,
                    "description": "男子展示证件后获准进入。",
                    "highlight_type": "身份揭示",
                    "reason": "独立的处境变化。",
                }
            ],
        ),
    ]

    return [*entries, copy.deepcopy(entries[-1])]


def agent(client, evidence, output, observer=None, **kwargs):
    return VideoAgent(
        client,
        evidence,
        output,
        perception=VideoPerception(observer or Observer(), 4, evidence),
        **kwargs,
    )


def context(request):
    return json.loads(request["inputs"][-1]["content"][0]["text"])


def test_complete_uses_raw_frames_separate_review_and_full_candidate_pool(evidence, tmp_path):
    script = actions()
    script[0]["args"]["observations"][1] = {
        "observation_id": "obs_page_1",
        "findings": [
            {
                "id": "late_reveal",
                "description": "人物展示另一项证据。",
                "reason": "后半段的重要揭示。",
                "status": "supported",
                "required_spans": [
                    {
                        "start_sec": 12,
                        "end_sec": 15,
                        "evidence_id": "obs_page_1",
                        "role": "decisive",
                    }
                ],
            }
        ],
    }
    for selection in script[1:]:
        selection["args"]["decisions"].append(
            {
                "event_id": "late_reveal",
                "score": 0.95,
                "selected": True,
                "description": "人物展示另一项证据。",
                "highlight_type": "证据揭示",
                "reason": "关键证据直接改变判断。",
            }
        )
    controller, observer = (
        Controller(script),
        Observer(
            ["画面右侧人物的身份未确认。"],
            [
                {
                    "category": "technical",
                    "description": "首帧保留上一镜头字幕，核心看点完整。",
                    "at_sec": 0,
                }
            ],
        ),
    )
    result = agent(controller, evidence, tmp_path, observer).run()
    assert result["completion"] == "complete"
    assert result["analysis"]["scan_coverage"] == 1
    assert result["analysis"]["pending_event_count"] == 0
    assert [item["start_sec"] for item in result["highlights"]] == [12, 3]
    assert result["highlights"][1]["description"] == "男子展示证件后获准进入。"
    assert all(item["review_status"] == "pending" for item in result["highlights"])
    assert (tmp_path / result["highlights"][0]["clip_path"]).read_bytes() == (
        tmp_path / "media.mp4"
    ).read_bytes()
    assert any(p["type"] == "input_image" for p in controller.requests[0]["inputs"][0]["content"])
    assert len(controller.requests) == 3
    review = next(r for r in observer.requests if r["schema"]["title"] == "ReviewResult")
    assert any(
        p.get("image_url", {}).get("url", "").startswith("data:video/mp4;") for p in review["parts"]
    )
    assert "history" not in review
    assert "story_so_far" not in json.dumps(review)
    selection = controller.requests[1]
    assert len(selection["inputs"]) == 1  # New selection segment has no orphan tool result.
    pool = context(selection)["selection_ledger"]["candidates"]
    assert context(selection)["story_so_far"] == "男子被阻拦，展示证件后获准进入。"
    assert {item["event_id"] for item in pool} == {"entry", "late_reveal"}
    assert context(selection)["source_notes"] == [
        {
            "observation_id": "obs_page_0",
            "src_start_sec": 0,
            "src_end_sec": 10,
            "uncertainties": ["画面右侧人物的身份未确认。"],
        },
        {
            "observation_id": "obs_page_1",
            "src_start_sec": 10,
            "src_end_sec": 20,
            "uncertainties": ["画面右侧人物的身份未确认。"],
        },
    ]
    assert "opaque-reasoning" not in (tmp_path / "trace.jsonl").read_text()
    assert all(
        "timestamp" in json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    )


def test_stop_and_resume_reuses_observations_and_review(evidence, tmp_path):
    controller, observer = Controller(actions()), Observer()
    partial = agent(controller, evidence, tmp_path, observer).run(max_turns=2)
    assert partial["completion"] == "partial" and partial["highlights"] == []
    assert json.loads((tmp_path / "state.json").read_text())["tools"]["story_so_far"] == (
        "男子被阻拦，展示证件后获准进入。"
    )
    calls = len(observer.requests)
    result = agent(controller, evidence, tmp_path, observer).run(resume=True)
    assert result["completion"] == "complete"
    assert len(observer.requests) == calls


def test_failed_controller_delivery_retains_identical_request_and_perception(evidence, tmp_path):
    controller = Controller([OpenAIClientError("disconnected"), *actions()])
    observer = Observer()
    partial = agent(controller, evidence, tmp_path, observer).run()
    assert partial["analysis"]["scan_coverage"] == 0
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["request_ready"] and saved["tools"]["delivered"] == []
    assert len(saved["tools"]["readings"]) == 2
    result = agent(controller, evidence, tmp_path, observer).run(resume=True)
    assert result["completion"] == "complete"
    assert controller.requests[0] == controller.requests[1]
    assert len([r for r in observer.requests if r["schema"]["title"] == "VideoReading"]) == 2


@pytest.mark.parametrize("failure_phase", ["analysis", "video_observation", "clip_review"])
def test_retry_is_same_request_and_mutation_is_not_replayed(
    evidence, tmp_path, monkeypatch, failure_phase
):
    monkeypatch.setattr("vh_agent.runtime.agent.time.sleep", lambda _: None)
    observer = Observer()
    controller = Controller(
        [OpenAIClientError("temporary", retryable=True), *actions()]
        if failure_phase == "analysis"
        else actions()
    )
    if failure_phase == "video_observation":
        observer.fail_observation = GeminiClientError("temporary", retryable=True)
    elif failure_phase == "clip_review":
        observer.fail_review = GeminiClientError("temporary", retryable=True)
    result = agent(controller, evidence, tmp_path, observer).run()
    assert result["completion"] == "complete"
    assert result["analysis"]["model_calls"] == len(controller.requests) + len(observer.requests)
    if failure_phase == "analysis":
        assert controller.requests[0] == controller.requests[1]
    traces = [json.loads(s) for s in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert sum(r.get("name") == "record_observations" for r in traces) == 1
    errors = [r for r in traces if r["kind"] == "request_error"]
    assert len(errors) == 1 and errors[0]["phase"] == failure_phase


def test_review_failure_preserves_observations_for_resume(evidence, tmp_path):
    observer = Observer()
    observer.fail_review = GeminiClientError("review failed")
    controller = Controller(actions())
    partial = agent(controller, evidence, tmp_path, observer).run()
    assert partial["completion"] == "partial"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["tools"]["plans"][0]["review"] is None
    assert (
        agent(controller, evidence, tmp_path, observer).run(resume=True)["completion"] == "complete"
    )


def test_text_only_response_does_not_forge_completion(evidence, tmp_path):
    result = agent(Controller([None]), evidence, tmp_path).run()
    assert result["completion"] == "partial"
    assert result["analysis"]["stop_reason"] == "missing_tool_call"


def test_duplicate_batch_cannot_commit(evidence, tmp_path):
    controller = Controller([[actions()[0], actions()[0]]])
    runtime = agent(controller, evidence, tmp_path)
    result = runtime.run(max_turns=1)
    assert result["completion"] == "partial"
    assert runtime.tools.completed_pages == set() and runtime.tools.events == {}


def test_bad_arguments_expose_fields_without_dumping_inputs(evidence, tmp_path):
    controller = Controller([actions()[0], call("inspect_video", start_sec=0, question="检查台词")])
    agent(controller, evidence, tmp_path).run(max_turns=2)
    traces = [json.loads(s) for s in (tmp_path / "trace.jsonl").read_text().splitlines()]
    error = next(r["result"] for r in traces if r["kind"] == "tool" and r["error"])
    assert any(f["path"] == "end_sec" for f in error["fields"])
    assert "input" not in str(error)


def test_received_call_and_completed_choice_survive_export_crash(evidence, tmp_path, monkeypatch):
    from vh_agent.runtime import agent as module

    original = module.write_json

    def interrupt(path, value):
        if path.name == "selection.json":
            raise KeyboardInterrupt
        original(path, value)

    monkeypatch.setattr(module, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        agent(Controller(actions()), evidence, tmp_path).run()
    monkeypatch.setattr(module, "write_json", original)
    controller = Controller([])
    assert agent(controller, evidence, tmp_path).run(resume=True)["completion"] == "complete"
    assert controller.requests == []


def test_changed_profile_and_existing_run_are_rejected(evidence, tmp_path):
    agent(Controller(actions()), evidence, tmp_path).run(max_turns=1)
    with pytest.raises(ValueError, match="same video"):
        agent(Controller([]), evidence, tmp_path, max_clip_sec=20).run(resume=True)
    with pytest.raises(ValueError, match="already"):
        agent(Controller([]), evidence, tmp_path).run()


def test_resume_checks_prompt_content_and_has_no_source_hash(evidence, tmp_path, monkeypatch):
    controller, observer = Controller(actions()), Observer()
    agent(controller, evidence, tmp_path, observer).run(max_turns=1)
    path = tmp_path / "state.json"
    state = json.loads(path.read_text())
    assert "implementation_hash" not in state["profile"]
    with monkeypatch.context() as patch:
        patch.setattr("vh_agent.runtime.agent.ANALYSIS_PROMPT", "修改后的高光判断标准")
        with pytest.raises(ValueError, match="same video, model, prompt"):
            agent(controller, evidence, tmp_path, observer).run(resume=True)
    assert (
        agent(controller, evidence, tmp_path, observer).run(resume=True)["completion"] == "complete"
    )


def test_failed_video_batch_exhausts_retries_and_resumes_saved_peer(
    evidence, tmp_path, monkeypatch
):
    monkeypatch.setattr("vh_agent.runtime.agent.time.sleep", lambda _: None)

    class DisconnectedObserver(Observer):
        def generate(self, system, parts, *, schema, **kwargs):
            if schema["title"] == "VideoReading" and json.loads(parts[0]["text"])["src_start_sec"]:
                self.requests.append({"system": system, "parts": parts, "schema": schema})
                raise GeminiClientError("Gemini DNS resolution failed", retryable=True)
            return super().generate(system, parts, schema=schema, **kwargs)

    controller = Controller(actions())
    runtime = agent(controller, evidence, tmp_path, DisconnectedObserver())
    result = runtime.run()
    assert result["completion"] == "partial" and result["highlights"] == []
    assert result["analysis"]["stop_reason"] == "execution_error"
    assert result["analysis"]["scan_coverage"] == 0
    assert controller.requests == []
    saved = json.loads((tmp_path / "state.json").read_text())
    assert set(saved["tools"]["readings"]) == {"obs_page_0"}
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    errors = [row for row in trace if row["kind"] == "request_error"]
    assert len(errors) == 3 and all(row["elapsed_sec"] >= 0 for row in errors)
    observer = Observer()
    resumed = agent(controller, evidence, tmp_path, observer).run(resume=True)
    assert resumed["completion"] == "complete"
    readings = [r for r in observer.requests if r["schema"]["title"] == "VideoReading"]
    assert len(readings) == 1 and json.loads(readings[0]["parts"][0]["text"])["src_start_sec"] == 10


def test_long_scan_rotates_native_history_and_preserves_every_page(evidence, tmp_path):
    evidence.video_info.duration_sec = 300
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
    entries = [
        call(
            "record_observations",
            observations=[
                {"observation_id": f"obs_page_{j}", "no_event_reason": "无候选"}
                for j in range(i, i + 2)
            ],
        )
        for i in range(0, 30, 2)
    ]
    controller = Controller([*entries, call("select_highlights", decisions=[])])
    runtime = agent(controller, evidence, tmp_path, context_token_budget=50000)
    result = runtime.run()
    assert result["completion"] == "complete" and result["analysis"]["scan_coverage"] == 1
    assert runtime.conversation.segment > 1
    assert len(runtime.tools.acknowledged) == 30


def test_record_binding_includes_earlier_delivered_but_unregistered_observations(
    evidence, tmp_path
):
    controller = Controller(
        [
            call(
                "record_observations",
                observations=[
                    {"observation_id": "obs_page_0", "findings": [], "no_event_reason": None}
                ],
            ),
            *actions(),
        ]
    )
    runtime = agent(controller, evidence, tmp_path, context_token_budget=50000)
    runtime.run()
    schema = controller.requests[1]["tools"][0]["parameters"]
    assert set(schema["$defs"]["ObservationRecord"]["properties"]["observation_id"]["enum"]) == {
        "obs_page_0",
        "obs_page_1",
    }
    assert set(schema["$defs"]["RequiredSpan"]["properties"]["evidence_id"]["enum"]) == {
        "obs_page_0",
        "obs_page_1",
    }
    assert context(controller.requests[1])["unrecorded_readings"] == []


def test_complete_input_budget_splits_pages_without_losing_coverage(evidence, tmp_path):
    class VerboseObserver(Observer):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            if kwargs["schema"]["title"] == "VideoReading":
                result["json"]["answer"] = "观察内容" * 3000
            return result

    controller = Controller(
        [
            call(
                "record_observations",
                observations=[{"observation_id": f"obs_page_{i}", "no_event_reason": "无独立看点"}],
            )
            for i in range(2)
        ]
        + [call("select_highlights", decisions=[])]
    )
    runtime = agent(controller, evidence, tmp_path, VerboseObserver(), context_token_budget=50000)
    result = runtime.run()
    assert result["completion"] == "complete"
    assert result["analysis"]["scan_coverage"] == 1
    assert len(controller.requests) == 3
    assert set(runtime.tools.acknowledged) == {"obs_page_0", "obs_page_1"}
    for request in controller.requests[:2]:
        images = [
            p
            for item in request["inputs"]
            for p in item.get("content", [])
            if p.get("type") == "input_image"
        ]
        assert len(images) == 5


def test_observations_use_source_time_and_reviews_use_playback_time(evidence):
    observer = Observer()
    perception = VideoPerception(observer, 4, evidence)
    observation = evidence.inspect(10, 15)
    reading = perception.observe(observation, "核对动作")
    request = observer.requests[0]
    frames = [
        part
        for part in request["parts"]
        if part.get("image_url", {}).get("url", "").startswith("data:image/jpeg;")
    ]
    assert len(frames) == 20
    metadata = json.loads(request["parts"][0]["text"])
    assert (metadata["src_start_sec"], metadata["src_end_sec"]) == (10, 15)
    labels = [p["text"] for p in request["parts"] if p.get("text", "").startswith("原片帧")]
    assert labels[0] == "原片帧 10.000 秒" and labels[-1] == "原片帧 14.750 秒"
    assert (reading["items"][0]["start_sec"], reading["items"][0]["end_sec"]) == (10, 11)
    cached = perception.observe(observation, "核对动作")
    assert cached["cache_hit"] and cached["items"] == reading["items"]
    repeated_material = perception.observe(evidence.inspect(0, 5), "核对动作")
    assert not repeated_material["cache_hit"]
    assert repeated_material["items"][0]["start_sec"] == 0
    perception.review(observation.path, "挑选高光", start_sec=10, end_sec=15, fps=4)
    labels = [
        p["text"]
        for p in observer.requests[-1]["parts"]
        if p.get("text", "").startswith("本段原帧")
    ]
    assert labels[0] == "本段原帧 0.000 秒" and labels[-1] == "本段原帧 4.750 秒"


def test_supplied_subtitles_are_delivered_with_pages_without_an_extra_agent_turn(
    evidence, tmp_path
):
    import pysubs2

    from vh_agent.providers.retrieval import TranscriptSearch

    subtitles = pysubs2.SSAFile()
    subtitles.events = [
        pysubs2.SSAEvent(start=1000 + index * 50, end=1040 + index * 50, text=f"台词{index}")
        for index in range(121)
    ]
    path = tmp_path / "dialogue.srt"
    subtitles.save(str(path))
    evidence.output_dir = tmp_path
    settings = SimpleNamespace(
        asr_model=tmp_path / "asr", asr_compute_type="float16", asr_device="cpu"
    )
    transcript = TranscriptSearch(evidence, settings, path, "zh")
    controller = Controller(actions())
    detector = agent(controller, evidence, tmp_path, transcript=transcript)
    result = detector.run()
    assert result["completion"] == "complete" and len(controller.requests) == 3
    packets = [
        json.loads(part["text"])
        for part in controller.requests[0]["inputs"][0]["content"][1:]
        if part.get("type") == "input_text" and part["text"].startswith("{")
    ]
    assert [row["text"] for row in packets[0]["transcript"]] == [f"台词{i}" for i in range(121)]
    assert packets[1]["transcript"] == []
    assert len(detector.tools.read_memory.rows) == 121
