import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vh_agent import reasoning as reasoning_module
from vh_agent.config import PROJECT_ROOT, Settings
from vh_agent.models import (
    EvidenceLedger,
    FrameSample,
    JudgeDecision,
    RankedHighlight,
    SceneCard,
    VideoInfo,
)
from vh_agent.reasoning import (
    OpenAIReasoner,
    _decode_json_object,
    _needs_adjudication,
    _score_decision,
    build_evidence_ledgers,
    validate_highlight_decision,
)


def _scene(scene_id: str, start: float, end: float, **updates: object) -> SceneCard:
    values: dict[str, object] = {
        "scene_id": scene_id,
        "start_sec": start,
        "end_sec": end,
        "state_before": "人物相信旧身份",
        "new_evidence": "亲子鉴定公开",
        "state_after": "人物确认真实身份",
        "evidence": ["[12.00s ASR] 亲子鉴定公开"],
    }
    values.update(updates)
    return SceneCard(**values)


def _decision(**updates: object) -> JudgeDecision:
    values: dict[str, object] = {
        "map_supported": True,
        "is_highlight": True,
        "highlight_type": "reveal",
        "description": "身份揭露",
        "reason": "旧认知被证据推翻",
        "evidence_grounding": 0.9,
        "narrative_impact": 0.8,
        "standalone_clarity": 0.8,
        "clipability": 0.8,
        "start_sec": 10,
        "end_sec": 18,
        "evidence": ["F03 12s 证件"],
        "setup_evidence_times_sec": [10],
        "decisive_evidence_times_sec": [12],
    }
    values.update(updates)
    return JudgeDecision(**values)


def test_judge_score_is_computed_from_required_dimensions() -> None:
    decision = _score_decision(_decision())
    assert decision.score == 0.83
    with pytest.raises(ValueError):
        JudgeDecision.model_validate({"map_supported": True})


def test_json_decoder_accepts_only_plain_or_single_fenced_object() -> None:
    assert _decode_json_object('{"ok": true}') == {"ok": True}
    assert _decode_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(json.JSONDecodeError):
        _decode_json_object('result:\n```json\n{"ok": true}\n```')


def test_disagreement_triggers_adjudication_only_for_material_difference() -> None:
    first = _score_decision(_decision())
    close = _score_decision(_decision(narrative_impact=0.78))
    rejected = _score_decision(_decision(map_supported=False))
    assert _needs_adjudication(first, close, 0.65) is False
    assert _needs_adjudication(first, rejected, 0.65) is True


def test_settings_routes_gemini_models_over_openai_compatible_endpoint() -> None:
    settings = Settings(
        VH_REASONING_PROVIDER="gemini",
        GEMINI_API_KEY="test",
        GEMINI_BASE_URL="https://yetoken.vip/v1",
        GEMINI_MAP_MODEL="gemini-3.1-flash-lite",
        GEMINI_JUDGE_MODEL="gemini-3.7-flash",
    )
    assert settings.reasoning_api_key == "test"
    assert settings.reasoning_base_url == "https://yetoken.vip/v1"


def test_settings_uses_agent_project_env_file() -> None:
    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


def test_scene_gate_accepts_adjacent_scene_setup_but_not_ungrounded_map() -> None:
    previous = _scene("scene_0001", 0, 14)
    current = _scene("scene_0002", 14, 30)
    complete = _decision(
        score=0.9,
        confidence=0.9,
        start_sec=6,
        end_sec=22,
        setup_evidence_times_sec=[8],
        decisive_evidence_times_sec=[18],
    )
    normalized = validate_highlight_decision(complete, current, previous)
    assert normalized.is_highlight is True
    assert normalized.start_sec == 6

    unsupported = complete.model_copy(update={"map_supported": False})
    assert validate_highlight_decision(unsupported, current, previous).is_highlight is False


def test_scene_gate_keeps_judge_window_when_setup_is_far_from_the_event() -> None:
    scene = _scene("scene_0001", 0, 40)
    decision = _decision(
        score=0.8,
        confidence=0.9,
        start_sec=20,
        end_sec=36,
        setup_evidence_times_sec=[3, 9, 21],
        decisive_evidence_times_sec=[28, 33],
    )
    normalized = validate_highlight_decision(decision, scene, None)
    assert normalized.is_highlight is True
    assert normalized.start_sec == 20
    assert normalized.end_sec == 36
    assert normalized.setup_evidence_times_sec == [21.0]
    assert normalized.end_sec - normalized.start_sec <= 24.0


def test_scene_gate_trims_padding_but_rejects_an_overlong_causal_core() -> None:
    scene = _scene("scene_0001", 0, 40)
    padded = _decision(
        score=0.8,
        confidence=0.9,
        highlight_type="reversal",
        start_sec=4,
        end_sec=40,
        setup_evidence_times_sec=[16],
        decisive_evidence_times_sec=[18, 28],
    )
    normalized = validate_highlight_decision(padded, scene, None)
    assert normalized.is_highlight is True
    assert normalized.end_sec - normalized.start_sec <= 24.0
    assert normalized.start_sec <= 16 <= normalized.end_sec
    assert normalized.start_sec <= 28 <= normalized.end_sec

    too_long = _decision(
        score=0.8,
        confidence=0.9,
        highlight_type="reversal",
        start_sec=8,
        end_sec=40,
        setup_evidence_times_sec=[8],
        decisive_evidence_times_sec=[10, 36],
    )
    assert validate_highlight_decision(too_long, scene, None).is_highlight is False


def test_scene_gate_allows_unknown_state_before_when_setup_is_anchored() -> None:
    scene = _scene("scene_0001", 14, 30, state_before="unknown")
    decision = _decision(
        score=0.8,
        confidence=0.9,
        start_sec=16,
        end_sec=28,
        setup_evidence_times_sec=[16],
        decisive_evidence_times_sec=[20],
    )
    assert validate_highlight_decision(decision, scene, None).is_highlight is True


def test_scene_gate_does_not_reinterpret_a_grounded_judge_decision_by_type() -> None:
    scene = _scene("scene_0001", 14, 30, new_evidence="", state_after="规则被上级叫停")
    decision = _decision(
        score=0.8,
        confidence=0.9,
        highlight_type="conflict",
        start_sec=16,
        end_sec=28,
        setup_evidence_times_sec=[16],
        decisive_evidence_times_sec=[20],
    )
    assert validate_highlight_decision(decision, scene, None).is_highlight is True


def test_scene_gate_does_not_require_a_setup_anchor() -> None:
    scene = _scene("scene_0001", 14, 30)
    decision = _decision(
        start_sec=16,
        end_sec=28,
        setup_evidence_times_sec=[],
        decisive_evidence_times_sec=[20],
    )
    assert validate_highlight_decision(decision, scene, None).is_highlight is True


def test_system_prompts_state_core_contracts() -> None:
    from vh_agent.reasoning import (
        JUDGE_SYSTEM_PROMPT,
        LISTWISE_SYSTEM_PROMPT,
        MAP_SYSTEM_PROMPT,
    )

    assert "claims" in MAP_SYSTEM_PROMPT
    assert "new_evidence" in MAP_SYSTEM_PROMPT
    assert "unknown" in MAP_SYSTEM_PROMPT
    assert "约束" in JUDGE_SYSTEM_PROMPT
    assert "对抗" in JUDGE_SYSTEM_PROMPT
    assert "叙事变化" in JUDGE_SYSTEM_PROMPT
    assert "情绪或关系兑现" in JUDGE_SYSTEM_PROMPT
    assert "动作峰值" in JUDGE_SYSTEM_PROMPT
    assert "悬念钩子" in JUDGE_SYSTEM_PROMPT
    assert "旧状态 → 决定性证据 → 新状态" in JUDGE_SYSTEM_PROMPT
    assert "continue_previous_scene" in JUDGE_SYSTEM_PROMPT
    assert "evidence_grounding" in JUDGE_SYSTEM_PROMPT
    assert "搜身" not in MAP_SYSTEM_PROMPT + JUDGE_SYSTEM_PROMPT
    assert "拦门" not in MAP_SYSTEM_PROMPT + JUDGE_SYSTEM_PROMPT
    assert "不同叙事功能" in LISTWISE_SYSTEM_PROMPT
    assert "P/Q/E 只是占位符" in reasoning_module.MAP_FEW_SHOT_MESSAGES[0]["content"]
    assert len(reasoning_module.JUDGE_FEW_SHOT_MESSAGES) == 4


def test_map_keeps_claims_separate_from_observed_evidence() -> None:
    from vh_agent.models import SceneNarrative

    narrative = SceneNarrative.model_validate(
        {
            "scene_id": "scene_0001",
            "claims": ["你就是杀人凶手"],
            "event_type": ["power_dynamics"],
            "state_before": "unknown",
            "new_evidence": "",
            "state_after": "",
            "salience": 0.2,
            "uncertainty": 0.5,
            "evidence": ["F02 8s 对峙"],
        }
    )
    assert narrative.claims == ["你就是杀人凶手"]
    assert narrative.state_before == "unknown"
    assert narrative.new_evidence == ""
    assert narrative.event_type == ["power_dynamics"]


def test_evidence_ledger_is_unverified_and_time_ordered() -> None:
    first = _scene(
        "scene_0001",
        1,
        8,
        actors=["女主"],
        action="发现戒指",
        claims=["这是你妈的戒指"],
    )
    second = _scene("scene_0002", 10, 18, actors=["女主", "母亲"], action="母女相认")
    before, final = build_evidence_ledgers([first, second])
    assert before["scene_0002"].observations[0] == (
        "[1.00-8.00s] 亲子鉴定公开（证据：[12.00s ASR] 亲子鉴定公开）"
    )
    assert any("声称：这是你妈的戒指" in item for item in before["scene_0002"].observations)
    assert "母女相认" in final.recent_summaries[-1]
    assert EvidenceLedger() != final


def test_openai_reasoner_maps_one_scene_and_verifies_without_rewriting(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    requests: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, *, model, **kwargs):
            calls.append(model)
            requests.append(kwargs)
            if "8B" in model:
                payload = {
                    "scene_id": "scene_0001",
                    "actors": ["女主", "母亲"],
                    "action": "母亲拿出亲子鉴定",
                    "event_type": ["reveal"],
                    "state_before": "女主认为自己是养女",
                    "new_evidence": "亲子鉴定上的名字一致",
                    "state_after": "女主得知自己是亲生女儿",
                    "relationship_change": "母女身份确认",
                    "emotion": ["震惊"],
                    "salience": 0.9,
                    "uncertainty": 0.2,
                    "evidence": ["F01 12.5s 亲子鉴定特写"],
                }
            elif "ranked_highlight_ids" in kwargs["messages"][1]["content"]:
                payload = {
                    "ranked_highlight_ids": ["hl_reveal", "hl_emotion"],
                    "selected_highlight_ids": ["hl_reveal"],
                    "rationale": "揭露造成明确状态变化",
                }
            else:
                payload = {
                    "map_supported": True,
                    "highlight_type": "reveal",
                    "description": "身世真相揭露",
                    "reason": "养女认知被亲子鉴定推翻",
                    "evidence_grounding": 0.95,
                    "narrative_impact": 0.95,
                    "standalone_clarity": 0.9,
                    "clipability": 0.9,
                    "start_sec": 9,
                    "end_sec": 20,
                    "evidence": ["F01 12.5s 亲子鉴定特写"],
                    "setup_evidence_times_sec": [9],
                    "decisive_evidence_times_sec": [12],
                    "continue_previous_scene": False,
                }
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(reasoning_module, "OpenAI", FakeClient)
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), "red").save(image)
    settings = Settings(
        VH_REASONING_PROVIDER="siliconflow",
        SILICONFLOW_API_KEY="test",
        SILICONFLOW_MAP_MODEL="Qwen/Qwen3-VL-8B-Instruct",
        SILICONFLOW_JUDGE_MODEL="Qwen/Qwen3-VL-32B-Instruct",
    )
    video = VideoInfo(path=tmp_path / "video.mp4", duration_sec=60, width=480, height=852, fps=25, has_audio=True)
    scene = _scene(
        "scene_0001", 8, 24, transcript="[10.00-12.00s OCR] 原来我才是亲生女儿",
        frame_samples=[FrameSample(timestamp_sec=12.5, path=image)],
    )
    reasoner = OpenAIReasoner(settings)
    mapped = reasoner.map_scene(video, scene)
    decision = reasoner.judge(video, scene, None, EvidenceLedger())
    ranking = reasoner.rank_highlights(
        video,
        [
            RankedHighlight(highlight_id="hl_reveal", start_sec=9, end_sec=20, score=0.9, local_score=0.7, judge_score=0.95, highlight_type="reveal", description="揭露", reason="反转", confidence=0.9),
            RankedHighlight(highlight_id="hl_emotion", start_sec=24, end_sec=30, score=0.75, local_score=0.6, judge_score=0.78, highlight_type="emotion", description="震惊", reason="反应", confidence=0.8),
        ],
        max_selected=1,
    )
    assert mapped.state_after == "女主得知自己是亲生女儿"
    assert decision.decision.map_supported is True
    assert len(decision.votes) == 2
    assert all(vote.is_highlight for vote in decision.votes)
    assert decision.calls == 2
    assert ranking.selected_highlight_ids == ["hl_reveal"]
    assert calls == [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-32B-Instruct",
        "Qwen/Qwen3-VL-32B-Instruct",
        "Qwen/Qwen3-VL-32B-Instruct",
    ]
    assert "SceneCard" in requests[1]["messages"][0]["content"]
    assert "scene_0001" in requests[0]["messages"][-1]["content"][-1]["text"]
