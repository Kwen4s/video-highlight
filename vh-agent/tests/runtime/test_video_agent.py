"""Controller loop, automatic observation persistence and clip delivery."""

import copy
import json
from types import SimpleNamespace

from test_agent_tools import FakeEvidenceStore, candidate

from vh_agent.providers.gemini_client import GeminiClientError
from vh_agent.providers.video_perception import VideoPerception
from vh_agent.runtime.agent import VideoAgent


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


class Evidence(FakeEvidenceStore):
    def frames(self, times, region=None, **kwargs):
        image = self.output_dir / "frame.jpg"
        image.write_bytes(b"image")
        return [{"timestamp_sec": t, "path": str(image)} for t in times]


def make_agent(tmp_path, actions, issues=None):
    evidence = Evidence(tmp_path)
    controller = Controller(actions)
    observer = Observer(review_issues=issues)
    agent = VideoAgent(
        controller,
        evidence,
        tmp_path / "job",
        perception=VideoPerception(observer, 1, evidence),
        max_clip_sec=20,
    )
    return agent, controller, observer


def test_retryable_observation_error_retries_before_delivery(tmp_path):
    agent, _, observer = make_agent(
        tmp_path,
        [
            call("watch_video", start_sec=0, end_sec=60, question="发生了什么？"),
            call("submit_highlights", candidate_ids=[]),
        ],
    )
    observer.fail_observation = GeminiClientError("invalid structured JSON", retryable=True)
    result = agent.run()
    assert result["completion"] == "complete"
    assert result["analysis"]["scan_coverage"] == 1
    assert len(observer.requests) == 2
    records = [json.loads(line) for line in (tmp_path / "job/trace.jsonl").read_text().splitlines()]
    assert sum(r["kind"] == "request_error" for r in records) == 1
    assert sum(r["kind"] == "video_observation" for r in records) == 1


def test_real_loop_delivers_and_resumes_without_manual_recording(tmp_path):
    def actions():
        yield call("watch_video", start_sec=0, end_sec=60, question="人物发生了什么变化？")
        key = next(iter(agent.tools.observations))
        yield call(
            "save_candidates",
            candidates=[candidate(SimpleNamespace(observation_id=key))],
            story_so_far="男子展示身份，双方态度改变。",
        )
        candidate_id = next(iter(agent.tools.events))
        yield call("submit_highlights", candidate_ids=[candidate_id])
        report = agent.submission_result
        yield call(
            "submit_highlights",
            candidate_ids=[candidate_id],
            confirmations=[
                {
                    "candidate_id": candidate_id,
                    "review_id": report["reviews"][0]["review_id"],
                    "reason": "已核对成片与原片说明，身份揭晓完整",
                }
            ],
        )

    agent, controller, observer = make_agent(tmp_path, actions())
    result = agent.run()
    assert result["completion"] == "complete" and len(result["highlights"]) == 1
    assert result["analysis"]["scan_coverage"] == 1
    assert len(controller.requests) == 4 and len(observer.requests) == 2
    assert all("record_observations" not in str(r["tools"]) for r in controller.requests)
    resumed, _, _ = make_agent(tmp_path, [])
    assert resumed.run(resume=True)["highlights"] == result["highlights"]


def test_resume_keeps_delivered_observation_and_tool_result(tmp_path):
    agent, _, _ = make_agent(
        tmp_path, [call("watch_video", start_sec=0, end_sec=30, question="发生了什么？")]
    )
    result = agent.run(max_turns=1)
    assert result["completion"] == "partial" and result["analysis"]["scan_coverage"] == 0
    restored, controller2, _ = make_agent(
        tmp_path,
        [
            call("watch_video", start_sec=30, end_sec=60, question="后来呢？"),
            call("submit_highlights", candidate_ids=[]),
        ],
    )
    assert restored.run(resume=True)["completion"] == "complete"
    first = controller2.requests[0]
    assert "observation_id" in str(first["inputs"])
    assert len(controller2.requests) == 2


def test_tool_result_is_followed_before_next_video(tmp_path):
    agent, controller, observer = make_agent(
        tmp_path,
        [
            call("read_memory", collection="candidates"),
            call("watch_video", start_sec=0, end_sec=30, question="发生了什么？"),
        ],
    )
    agent.run(max_turns=2)
    assert not observer.requests
    assert "function_call_output" in str(controller.requests[1]["inputs"])
    assert agent.tools.progress()["scan_coverage"] == 0


def test_review_problem_returns_to_agent_and_can_be_omitted(tmp_path):
    def actions():
        yield call("watch_video", start_sec=0, end_sec=60, question="发生了什么？")
        key = next(iter(agent.tools.observations))
        yield call(
            "save_candidates",
            candidates=[candidate(SimpleNamespace(observation_id=key))],
        )
        yield call("submit_highlights", candidate_ids=list(agent.tools.events))
        yield call("submit_highlights", candidate_ids=[])

    agent, controller, _ = make_agent(
        tmp_path, actions(), [{"category": "cutoff", "description": "关键台词截断", "at_sec": 1}]
    )
    result = agent.run()
    assert result["completion"] == "complete" and not result["highlights"]
    assert "awaiting_confirmation" in str(controller.requests[-1]["inputs"])


def test_new_frames_and_first_memory_access_count_as_progress(tmp_path):
    actions = [call("read_frames", times=[i]) for i in range(9)] + [
        call("watch_video", start_sec=0, end_sec=60, question="发生了什么？"),
        call("submit_highlights", candidate_ids=[]),
    ]
    agent, _, _ = make_agent(tmp_path, actions)
    assert agent.run()["completion"] == "complete"
    assert len(agent.tools.frame_readings) == 9
    facts = agent.tools.information_facts()
    agent.tools.execute("read_memory", {"collection": "frames", "offset": 0, "limit": 1})
    assert agent.tools.information_facts() - facts
    facts = agent.tools.information_facts()
    agent.tools.execute("read_memory", {"collection": "frames", "offset": 0, "limit": 1})
    assert agent.tools.information_facts() == facts


def test_confirmation_survives_pause_and_source_notes_survive_rotation(tmp_path):
    def actions():
        yield call("watch_video", start_sec=0, end_sec=60, question="发生了什么？")
        key = next(iter(agent.tools.observations))
        yield call(
            "save_candidates",
            candidates=[
                candidate(
                    SimpleNamespace(observation_id=key),
                    start_sec=0,
                    end_sec=5,
                    evidence=[
                        {"start_sec": 0, "end_sec": 1, "role": "decisive", "evidence_id": key}
                    ],
                )
            ],
        )
        yield call("submit_highlights", candidate_ids=list(agent.tools.events))

    agent, _, _ = make_agent(tmp_path, actions())
    agent.perception.client.uncertainties = ["说话者尚未确认"]
    result = agent.run(max_turns=3)
    assert result["completion"] == "partial"
    report = agent.submission_result
    assert report["status"] == "awaiting_confirmation"
    restored, controller, _ = make_agent(
        tmp_path,
        [
            call(
                "submit_highlights",
                candidate_ids=list(agent.tools.events),
                confirmations=[
                    {
                        "candidate_id": report["reviews"][0]["candidate_id"],
                        "review_id": report["reviews"][0]["review_id"],
                        "reason": "复核与原片一致",
                    }
                ],
            )
        ],
    )
    assert restored.run(resume=True)["completion"] == "complete"
    request = str(controller.requests[0]["inputs"])
    assert "说话者尚未确认" in request and "source_notes" in request


def test_over_budget_watch_returns_to_controller_without_poisoning_state(tmp_path, monkeypatch):
    agent, controller, observer = make_agent(
        tmp_path,
        [
            call("watch_video", start_sec=0, end_sec=60, question="发生了什么？"),
            call("watch_video", start_sec=0, end_sec=30, question="先看前半段"),
            call("watch_video", start_sec=30, end_sec=60, question="再看后半段"),
            call("submit_highlights", candidate_ids=[]),
        ],
    )
    size = agent.perception.encoded_size
    monkeypatch.setattr(
        agent.perception,
        "encoded_size",
        lambda observation, question: (
            21_000_000 if observation.duration_sec > 30 else size(observation, question)
        ),
    )
    assert agent.run()["completion"] == "complete"
    assert "20 MB" in str(controller.requests[1]["inputs"])
    assert len(agent.tools.observations) == 2 and len(observer.requests) == 2
