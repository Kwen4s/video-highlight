import math

from vh_agent.candidates import build_candidates, refine_boundaries, timestamped_transcript
from vh_agent.models import CandidateWindow, JudgeDecision, RankedHighlight, SceneCard, SceneSegment, TranscriptSegment, FrameSample


def _judge(**updates) -> JudgeDecision:
    values = {
        "map_supported": True,
        "evidence_grounding": 0.9,
        "narrative_impact": 0.8,
        "standalone_clarity": 0.8,
        "clipability": 0.8,
    }
    values.update(updates)
    return JudgeDecision(**values)


def test_narrative_cue_raises_local_candidate_score(tmp_path) -> None:
    frames = [
        FrameSample(timestamp_sec=float(second), path=tmp_path / f"{second}.jpg", change_score=0.1)
        for second in range(0, 30, 2)
    ]
    candidates = build_candidates(
        duration_sec=30,
        frames=frames,
        audio_energy=[1.0] * 30,
        transcript=[TranscriptSegment(start_sec=13, end_sec=16, text="原来你才是我的亲生女儿，真相终于揭开")],
        scenes=[],
        window_sec=10,
        stride_sec=5,
        max_candidates=3,
    )
    assert any(candidate.start_sec <= 13 <= candidate.end_sec for candidate in candidates)
    assert max(candidate.cue_score for candidate in candidates) == 1.0


def test_promo_candidate_is_penalized(tmp_path) -> None:
    from vh_agent.candidates import apply_content_filter

    candidate = CandidateWindow(
        start_sec=20, end_sec=30, local_score=0.8, transcript="免费观看全集，评论区链接自取"
    )
    filtered = apply_content_filter(candidate, duration_sec=30)
    assert filtered.filter_penalty >= 0.44
    assert filtered.local_score < 0.5


def test_semantic_scene_merges_shots_until_a_real_dialogue_and_semantic_break(tmp_path) -> None:
    from vh_agent.pipeline import _build_semantic_scenes

    frames = [
        FrameSample(
            timestamp_sec=float(second), path=tmp_path / f"{second}.jpg",
            semantic_change_score=0.9 if second == 16 else 0.1,
        )
        for second in range(0, 33)
    ]
    cards = _build_semantic_scenes(
        32,
        [CandidateWindow(start_sec=4, end_sec=24, local_score=0.9)],
        frames,
        [
            TranscriptSegment(start_sec=1, end_sec=14, text="同一场对话"),
            TranscriptSegment(start_sec=17, end_sec=28, text="下一场对话"),
        ],
        [],
        [
            SceneSegment(start_sec=0, end_sec=4),
            SceneSegment(start_sec=4, end_sec=8),
            SceneSegment(start_sec=8, end_sec=16),
            SceneSegment(start_sec=16, end_sec=24),
            SceneSegment(start_sec=24, end_sec=32),
        ],
    )
    assert [(card.start_sec, card.end_sec) for card in cards] == [(0.0, 16.0), (16.0, 32.0)]
    assert cards[0].local_score == 0.9
    assert cards[0].shot_ids == ["shot_0001", "shot_0002", "shot_0003"]


def test_semantic_scene_splits_on_embedding_break_without_dialogue_pause(tmp_path) -> None:
    from vh_agent.pipeline import _build_semantic_scenes

    frames = [
        FrameSample(
            timestamp_sec=float(second), path=tmp_path / f"{second}.jpg",
            semantic_change_score=0.9 if second == 16 else 0.1,
        )
        for second in range(0, 33)
    ]
    cards = _build_semantic_scenes(
        32,
        [CandidateWindow(start_sec=4, end_sec=24, local_score=0.9)],
        frames,
        [TranscriptSegment(start_sec=1, end_sec=28, text="连续对白穿过叙事转折")],
        [],
        [
            SceneSegment(start_sec=0, end_sec=16),
            SceneSegment(start_sec=16, end_sec=32),
        ],
    )
    assert [(card.start_sec, card.end_sec) for card in cards] == [(0.0, 16.0), (16.0, 32.0)]


def test_semantic_scene_caps_unbroken_material_at_playable_scene_length(tmp_path) -> None:
    from vh_agent.config import MAX_SCENE_SEC
    from vh_agent.pipeline import _build_semantic_scenes

    frames = [
        FrameSample(timestamp_sec=float(second), path=tmp_path / f"{second}.jpg")
        for second in range(61)
    ]
    cards = _build_semantic_scenes(
        60, [], frames, [], [], [SceneSegment(start_sec=0, end_sec=60)]
    )
    assert max(card.end_sec - card.start_sec for card in cards) <= MAX_SCENE_SEC
    assert len(cards) >= math.ceil(60 / MAX_SCENE_SEC)


def test_semantic_scene_absorbs_a_short_trailing_stub(tmp_path) -> None:
    from vh_agent.config import MIN_SCENE_SEC
    from vh_agent.pipeline import _build_semantic_scenes

    duration = 48.03
    frames = [
        FrameSample(
            timestamp_sec=float(second),
            path=tmp_path / f"{second}.jpg",
            semantic_change_score=0.9 if second == 48 else 0.1,
        )
        for second in range(49)
    ]
    cards = _build_semantic_scenes(
        duration,
        [],
        frames,
        [],
        [],
        [
            SceneSegment(start_sec=0, end_sec=20),
            SceneSegment(start_sec=20, end_sec=40),
            SceneSegment(start_sec=40, end_sec=48),
            SceneSegment(start_sec=48, end_sec=duration),
        ],
    )
    assert cards[-1].end_sec == duration
    assert all(card.end_sec - card.start_sec >= MIN_SCENE_SEC - 1e-6 for card in cards)
    assert not any(abs(card.start_sec - 48.0) < 1e-6 for card in cards)


def test_scene_selection_keeps_long_video_coverage(tmp_path) -> None:
    from vh_agent.pipeline import _select_judge_scenes, _uniformly_sample_frames

    scenes = [
        SceneCard(
            scene_id=f"scene_{index:04d}", start_sec=float(index * 80), end_sec=float(index * 80 + 20),
            local_score=0.3 + (index % 5) * 0.1, salience=0.6 + (index % 3) * 0.1,
        )
        for index in range(20)
    ]
    selected = _select_judge_scenes(scenes, max_candidates=12, coverage_segment_sec=180)
    assert len(selected) == 12
    assert {int(((scene.start_sec + scene.end_sec) / 2) // 180) for scene in selected} == {0, 1, 2, 3, 4, 5, 6, 7, 8}

    frames = [
        FrameSample(timestamp_sec=float(index), path=tmp_path / f"{index}.jpg")
        for index in range(1583)
    ]
    sampled = _uniformly_sample_frames(frames, 600)
    assert len(sampled) == 600
    assert sampled[0] is frames[0]
    assert sampled[-1] is frames[-1]


def test_post_judge_merge_requires_adjacent_continuation_and_clip_cap() -> None:
    from vh_agent.pipeline import _merge_verified_scenes

    def highlight(identifier: str, start: float, end: float) -> RankedHighlight:
        return RankedHighlight(
            highlight_id=identifier,
            start_sec=start,
            end_sec=end,
            score=0.8,
            local_score=0.6,
            judge_score=0.9,
            highlight_type="reveal",
            description=identifier,
            reason="状态变化",
            confidence=0.9,
        )

    first = SceneCard(scene_id="scene_0001", start_sec=0, end_sec=10)
    second = SceneCard(scene_id="scene_0002", start_sec=10, end_sec=20)
    merged = _merge_verified_scenes(
        [
            (first, _judge(is_highlight=True, score=0.9, confidence=0.9), highlight("hl_1", 1, 10)),
            (second, _judge(is_highlight=True, score=0.9, confidence=0.9, continue_previous_scene=True), highlight("hl_2", 10, 20)),
        ]
    )
    assert len(merged) == 1
    assert (merged[0].start_sec, merged[0].end_sec) == (1, 20)

    distant = SceneCard(scene_id="scene_0004", start_sec=20, end_sec=30)
    unmerged = _merge_verified_scenes(
        [
            (first, _judge(is_highlight=True, score=0.9, confidence=0.9), highlight("hl_1", 1, 10)),
            (distant, _judge(is_highlight=True, score=0.9, confidence=0.9, continue_previous_scene=True), highlight("hl_4", 20, 30)),
        ]
    )
    assert len(unmerged) == 2


def test_trailing_result_attaches_when_next_scene_continues_but_is_not_a_highlight() -> None:
    from vh_agent.pipeline import _attach_trailing_result

    first = SceneCard(scene_id="scene_0002", start_sec=11.6, end_sec=33.2)
    second = SceneCard(scene_id="scene_0003", start_sec=33.2, end_sec=41.4)
    highlight = RankedHighlight(
        highlight_id="hl_insult",
        start_sec=24.0,
        end_sec=33.2,
        score=0.8,
        local_score=0.6,
        judge_score=0.7,
        highlight_type="conflict",
        description="当众贬低",
        reason="约束被公开对抗",
        confidence=0.9,
    )
    attached = _attach_trailing_result(
        [
            (
                first,
                _judge(is_highlight=True, score=0.7, confidence=0.9),
                highlight,
            )
        ],
        [first, second],
        {
            "scene_0002": _judge(is_highlight=True, score=0.7, confidence=0.9),
            "scene_0003": _judge(
                is_highlight=False,
                score=0.48,
                confidence=0.8,
                continue_previous_scene=True,
                end_sec=41.4,
                evidence=["F08 36s 当场接话"],
            ),
        },
    )
    assert len(attached) == 1
    assert (attached[0][2].start_sec, attached[0][2].end_sec) == (24.0, 41.4)


def test_refined_boundary_keeps_decisive_evidence() -> None:
    curve = [0.1] * 50
    curve[42] = 1.0
    start, end = refine_boundaries(5, 47, 50, [], curve, [], [42.0])
    assert start <= 42 <= end
    assert end - start <= 24


def test_timestamped_transcript_preserves_evidence_times() -> None:
    segments = [
        TranscriptSegment(start_sec=1.25, end_sec=2.5, text="普通铺垫"),
        TranscriptSegment(start_sec=3.0, end_sec=4.75, text="真相出现", source="ocr"),
    ]
    assert timestamped_transcript(segments, 2.8, 5.0) == "[3.00-4.75s OCR] 真相出现"
