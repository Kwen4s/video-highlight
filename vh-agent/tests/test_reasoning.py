import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vh_agent import reasoning as reasoning_module
from vh_agent.config import Settings
from vh_agent.models import (
    CandidateWindow,
    ChapterContext,
    EventCard,
    FrameSample,
    RankedHighlight,
    StoryMemory,
    VideoInfo,
)
from vh_agent.reasoning import (
    SiliconFlowReasoner,
    build_highlight_hypothesis,
    build_story_memories,
    parse_judge_decision,
)


def test_parse_judge_decision_from_fenced_json() -> None:
    raw = """```json
    {"hypothesis_supported": true, "is_highlight": true, "score": 0.9, "highlight_type": ["reversal", "reveal"],
     "description": "身份揭露", "reason": "旧认知被证据推翻", "confidence": 0.8,
     "start_sec": 10, "end_sec": 18, "evidence": ["证件曝光"]}
    ```"""
    decision = parse_judge_decision(raw)
    assert decision.is_highlight is True
    assert decision.highlight_type == "reversal"
    assert decision.score == 0.9


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            '{"hypothesis_supported": true, "is_highlight": true, "score": ["bad"], "highlight_type": "other"}',
            None,
        ),
        (
            '{"hypothesis_supported": true, "is_highlight": true, "confidence": 0.8, "highlight_type": "other"}',
            "missing required fields",
        ),
    ],
)
def test_invalid_judge_response_is_rejected(raw: str, message: str | None) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        parse_judge_decision(raw)


def test_chapter_map_and_judge_use_separate_models(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    requests: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, *, model, **kwargs):
            calls.append(model)
            requests.append(kwargs)
            if "8B" in model:
                payload = {
                    "events": [
                        {
                            "start_sec": 10,
                            "end_sec": 22,
                            "actors": ["女主", "母亲"],
                            "action": "母亲拿出亲子鉴定",
                            "event_type": ["reveal"],
                            "state_before": "女主认为自己是养女",
                            "new_evidence": ["亲子鉴定", "名字一致"],
                            "state_after": "女主得知自己是亲生女儿",
                            "relationship_change": "母女身份确认",
                            "emotion": ["震惊", "不敢置信"],
                            "salience": 0.9,
                            "uncertainty": 0.2,
                            "evidence": ["F01 12.5s 亲子鉴定特写"],
                        }
                    ]
                }
            elif "listwise" in kwargs["messages"][0]["content"]:
                payload = {
                    "ranked_highlight_ids": ["hl_reveal", "hl_emotion"],
                    "selected_highlight_ids": ["hl_reveal"],
                    "rationale": "揭露造成明确状态变化，另一条仅为重复反应",
                }
            else:
                payload = {
                    "hypothesis_supported": True,
                    "is_highlight": True,
                    "score": 0.95,
                    "highlight_type": ["reversal", "reveal"],
                    "description": ["身世真相", "身份揭露"],
                    "reason": "养女认知被亲子鉴定推翻",
                    "confidence": 0.9,
                    "start_sec": 9,
                    "end_sec": 24,
                    "evidence": ["亲子鉴定", "女主震惊反应"],
                }
            message = SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(reasoning_module, "OpenAI", FakeClient)
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), "red").save(frame)
    settings = Settings(
        SILICONFLOW_API_KEY="test",
        SILICONFLOW_MAP_MODEL="Qwen/Qwen3-VL-8B-Instruct",
        SILICONFLOW_JUDGE_MODEL="Qwen/Qwen3-VL-32B-Instruct",
    )
    reasoner = SiliconFlowReasoner(settings)
    video = VideoInfo(
        path=tmp_path / "video.mp4",
        duration_sec=60,
        width=480,
        height=852,
        fps=25,
        has_audio=True,
        title="测试短剧",
    )
    chapter = ChapterContext(
        chapter_id="chapter_001",
        start_sec=0,
        end_sec=60,
        transcript="[10.00-12.00s OCR] 原来我才是你的亲生女儿",
        frame_samples=[FrameSample(timestamp_sec=12.5, path=frame)],
    )
    card = reasoner.map_chapter(video, chapter)[0]
    candidate = CandidateWindow(
        start_sec=8,
        end_sec=24,
        local_score=0.7,
        transcript="原来我才是你的亲生女儿",
        frame_samples=[FrameSample(timestamp_sec=12.5, path=frame)],
    )
    hypothesis = build_highlight_hypothesis(card)
    memory = StoryMemory(known_facts=["女主一直被当作养女"])
    decision = reasoner.judge(
        video, candidate, hypothesis, memory, "此前她被当作养女", "亲子鉴定出现", "母女相认"
    )
    ranking = reasoner.rank_highlights(
        video,
        [
            RankedHighlight(
                highlight_id="hl_reveal",
                start_sec=9,
                end_sec=24,
                score=0.9,
                local_score=0.7,
                judge_score=0.95,
                highlight_type="reveal",
                description="亲子鉴定揭露身世",
                reason="新证据改变人物身份认知",
                evidence=["亲子鉴定"],
                confidence=0.9,
            ),
            RankedHighlight(
                highlight_id="hl_emotion",
                start_sec=24,
                end_sec=30,
                score=0.75,
                local_score=0.6,
                judge_score=0.78,
                highlight_type="emotion",
                description="女主震惊",
                reason="揭露后的情绪反应",
                evidence=["女主震惊反应"],
                confidence=0.8,
            ),
        ],
        max_selected=1,
    )

    assert card.state_after == "女主得知自己是亲生女儿"
    assert card.new_evidence == "亲子鉴定；名字一致"
    assert card.emotion == "震惊；不敢置信"
    assert decision.highlight_type == "reversal"
    assert ranking.ranked_highlight_ids == ["hl_reveal", "hl_emotion"]
    assert ranking.selected_highlight_ids == ["hl_reveal"]
    assert calls == [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-32B-Instruct",
        "Qwen/Qwen3-VL-32B-Instruct",
    ]
    map_content = requests[0]["messages"][1]["content"]
    assert "所有有证据的独立原子事件" in map_content[-1]["text"]
    judge_content = requests[1]["messages"][1]["content"]
    assert "事件核心字幕：亲子鉴定出现" in judge_content[-1]["text"]
    assert "decisive_evidence_times_sec" in judge_content[-1]["text"]
    assert judge_content[0]["text"] == "F01  12.5s"
    assert judge_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert requests[2]["messages"][1]["content"].startswith("视频：测试短剧")
    assert "因果相连本身既不代表重复，也不代表独立" in requests[2]["messages"][0]["content"]


def test_story_memory_is_available_before_next_event() -> None:
    first = EventCard(
        start_sec=1,
        end_sec=5,
        actors=["女主"],
        evidence=["[2.00s OCR] 戒指刻有名字"],
        action="发现戒指",
        new_evidence="戒指刻有名字",
        state_after="戒指属于女主母亲",
    )
    second = EventCard(
        start_sec=10,
        end_sec=15,
        actors=["女主", "母亲"],
        action="母女相认",
        relationship_change="确认母女关系",
        evidence=["[12.00s ASR] 我们是母女"],
    )
    before, final = build_story_memories([first, second])
    assert any("戒指刻有名字" in fact for fact in before[id(second)].known_facts)
    assert any("确认母女关系" in fact for fact in final.relationship_states)


def test_unverified_event_does_not_pollute_story_memory() -> None:
    event = EventCard(
        start_sec=1,
        end_sec=5,
        actors=["未知说话者", "女主"],
        action="猜测隐藏身份",
        state_after="女主身份改变",
        uncertainty=0.9,
    )
    before, final = build_story_memories([event])
    assert before[id(event)].known_facts == []
    assert final == StoryMemory()


def test_rejected_hypothesis_can_yield_a_revised_highlight() -> None:
    decision = parse_judge_decision(
        '{"hypothesis_supported": false, "is_highlight": true, '
        '"score": 0.95, "confidence": 0.9, "highlight_type": "reveal"}'
    )
    assert decision.hypothesis_supported is False
    assert decision.is_highlight is True
