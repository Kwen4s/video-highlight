from vh_agent.candidates import build_candidates
from vh_agent.models import CandidateWindow, EventCard, FrameSample, TranscriptSegment


def test_narrative_cue_raises_candidate_score(tmp_path) -> None:
    frames = [
        FrameSample(timestamp_sec=float(second), path=tmp_path / f"{second}.jpg", change_score=0.1)
        for second in range(0, 30, 2)
    ]
    transcript = [
        TranscriptSegment(start_sec=13, end_sec=16, text="原来你才是我的亲生女儿，真相终于揭开")
    ]
    candidates = build_candidates(
        duration_sec=30,
        frames=frames,
        audio_energy=[1.0] * 30,
        transcript=transcript,
        scenes=[],
        audio_events=[],
        window_sec=10,
        stride_sec=5,
        max_candidates=3,
    )
    assert any(candidate.start_sec <= 13 <= candidate.end_sec for candidate in candidates)
    assert max(candidate.cue_score for candidate in candidates) == 1.0


def test_promo_candidate_is_penalized(tmp_path) -> None:
    from vh_agent.candidates import apply_content_filter
    from vh_agent.models import CandidateWindow

    candidate = CandidateWindow(
        start_sec=20,
        end_sec=30,
        local_score=0.8,
        transcript="免费观看全集，评论区链接自取",
    )
    filtered = apply_content_filter(candidate, duration_sec=30)
    assert filtered.filter_penalty >= 0.44
    assert filtered.local_score < 0.5


def test_narrative_window_includes_truncated_setup(tmp_path) -> None:
    from vh_agent.config import Settings
    from vh_agent.models import CandidateWindow, EventCard, JudgeDecision
    from vh_agent.pipeline import HighlightOrchestrator

    orchestrator = HighlightOrchestrator(Settings(VH_MODEL_CACHE_DIR=tmp_path / "models"))
    candidate = CandidateWindow(start_sec=32, end_sec=44, local_score=0.7)
    event = EventCard(start_sec=32, end_sec=44, event_type=["reveal"])
    decision = JudgeDecision(
        hypothesis_supported=True,
        is_highlight=True,
        score=0.9,
        highlight_type="other",
        start_sec=32,
        end_sec=44,
    )
    segment = orchestrator._to_highlight(
        duration_sec=60,
        video_path=tmp_path / "video.mp4",
        candidate=candidate,
        event_card=event,
        decision=decision,
        transcript=[],
        saliency_curve=[],
    )
    assert segment is not None
    assert segment.start_sec == 31.5
    assert (
        orchestrator._to_highlight(
            duration_sec=60,
            video_path=tmp_path / "video.mp4",
            candidate=candidate,
            event_card=event,
            decision=None,
            transcript=[],
            saliency_curve=[],
        )
        is None
    )
    rejected = decision.model_copy(update={"is_highlight": False})
    assert (
        orchestrator._to_highlight(
            duration_sec=60,
            video_path=tmp_path / "video.mp4",
            candidate=candidate,
            event_card=event,
            decision=rejected,
            transcript=[],
            saliency_curve=[],
        )
        is None
    )

    weak_local = candidate.model_copy(update={"local_score": 0.0})
    marginal_decision = decision.model_copy(update={"score": 0.7})
    assert (
        orchestrator._to_highlight(
            duration_sec=60,
            video_path=tmp_path / "video.mp4",
            candidate=weak_local,
            event_card=event,
            decision=marginal_decision,
            transcript=[],
            saliency_curve=[],
        )
        is not None
    )


def test_final_segments_cap_duration_and_remove_nested_tail() -> None:
    from vh_agent.candidates import refine_boundaries
    from vh_agent.models import RankedHighlight
    from vh_agent.pipeline import (
        _shares_decisive_evidence,
        _shorter_overlap_ratio,
    )

    start, end = refine_boundaries(10, 50, 60, [], [], [])
    assert (start, end) == (18.0, 42.0)

    common = {
        "score": 0.8,
        "local_score": 0.5,
        "judge_score": 0.9,
        "highlight_type": "reversal",
        "description": "反转",
        "reason": "真相揭露",
        "confidence": 0.9,
    }
    long = RankedHighlight(highlight_id="hl_long", start_sec=18, end_sec=48, **common)
    tail = RankedHighlight(highlight_id="hl_tail", start_sec=40, end_sec=54, **common)
    assert _shorter_overlap_ratio(long, tail) >= 0.55

    long.evidence = ["字幕：珠宝是我亲手放进去的"]
    tail.evidence = ["字幕: 珠宝是我亲手放进去的"]
    assert _shares_decisive_evidence(long, tail)


def test_long_video_candidate_budget_is_dynamic_and_bounded() -> None:
    from vh_agent.candidates import candidate_budget

    budget = lambda duration: candidate_budget(duration, 90, 4, 6, 72)
    assert budget(54) == 6
    assert budget(126) == 8
    assert budget(376) == 20
    assert budget(1583) == 72
    assert budget(3600) == 72


def test_long_video_candidates_cover_each_time_segment_and_boundary(tmp_path) -> None:
    from vh_agent.models import TranscriptSegment

    duration = 360
    frames = [
        FrameSample(
            timestamp_sec=float(second),
            path=tmp_path / f"{second}.jpg",
            change_score=(second % 20) / 20,
            semantic_change_score=(second % 28) / 28,
        )
        for second in range(0, duration, 4)
    ]
    transcript = [
        TranscriptSegment(start_sec=40, end_sec=44, text="原来真相是这样"),
        TranscriptSegment(start_sec=88, end_sec=96, text="身份竟然被揭穿"),
        TranscriptSegment(start_sec=220, end_sec=224, text="其实你一直骗我"),
        TranscriptSegment(start_sec=310, end_sec=314, text="秘密终于公开"),
    ]
    candidates = build_candidates(
        duration_sec=duration,
        frames=frames,
        audio_energy=[float(second % 17) for second in range(duration)],
        transcript=transcript,
        scenes=[],
        audio_events=[],
        window_sec=20,
        stride_sec=4,
        min_candidates=6,
        max_candidates=36,
        segment_sec=90,
        candidates_per_segment=2,
    )

    assert len(candidates) == 8
    covered_segments = {
        int(((candidate.start_sec + candidate.end_sec) / 2) // 90) for candidate in candidates
    }
    assert covered_segments == {0, 1, 2, 3}
    assert any(candidate.start_sec <= 88 and candidate.end_sec >= 96 for candidate in candidates)


def test_long_video_judge_budget_keeps_timeline_coverage(tmp_path) -> None:
    from vh_agent.models import CandidateWindow, EventCard
    from vh_agent.pipeline import (
        _select_judge_candidates,
        _uniformly_sample_frames,
    )

    candidates = [
        CandidateWindow(
            start_sec=float(index * 80),
            end_sec=float(index * 80 + 20),
            local_score=0.3 + (index % 5) * 0.1,
        )
        for index in range(20)
    ]
    events = {
        id(candidate): EventCard(
            start_sec=candidate.start_sec,
            end_sec=candidate.end_sec,
            salience=0.6 + (index % 3) * 0.1,
            uncertainty=0.2,
        )
        for index, candidate in enumerate(candidates)
    }
    selected = _select_judge_candidates(
        candidates,
        events,
        max_candidates=12,
        coverage_segment_sec=180,
    )

    assert len(selected) == 12
    all_segments = {
        int(((candidate.start_sec + candidate.end_sec) / 2) // 180) for candidate in candidates
    }
    selected_segments = {
        int(((candidate.start_sec + candidate.end_sec) / 2) // 180) for candidate in selected
    }
    assert selected_segments == all_segments

    frames = [
        FrameSample(timestamp_sec=float(index), path=tmp_path / f"{index}.jpg")
        for index in range(1583)
    ]
    sampled = _uniformly_sample_frames(frames, 600)
    assert len(sampled) == 600
    assert sampled[0] is frames[0]
    assert sampled[-1] is frames[-1]


def test_output_limit_grows_for_long_video_but_stays_bounded(tmp_path) -> None:
    from vh_agent.pipeline import _output_limit

    assert _output_limit(126, 90, 12) == 3
    assert _output_limit(1583, 90, 12) == 9
    assert _output_limit(3600, 90, 12) == 12


def test_higher_judge_score_keeps_precise_duplicate(tmp_path) -> None:
    from vh_agent.config import Settings
    from vh_agent.models import RankedHighlight
    from vh_agent.pipeline import HighlightOrchestrator

    orchestrator = HighlightOrchestrator(Settings(VH_MODEL_CACHE_DIR=tmp_path / "models"))
    common = {
        "local_score": 0.6,
        "highlight_type": "conflict",
        "description": "同一冲突",
        "reason": "冲突升级",
        "evidence": ["共同证据"],
        "confidence": 0.8,
    }
    precise = RankedHighlight(
        highlight_id="hl_precise", start_sec=9, end_sec=15, score=0.75, judge_score=0.75, **common
    )
    broad = RankedHighlight(
        highlight_id="hl_broad", start_sec=5, end_sec=20, score=0.7, judge_score=0.7, **common
    )

    selected = orchestrator._build_ranking_pool([broad, precise])
    assert [(item.start_sec, item.end_sec) for item in selected] == [(9.0, 15.0)]


def test_candidate_selection_reserves_clean_opening_and_ending(tmp_path) -> None:
    frames = [
        FrameSample(timestamp_sec=float(second), path=tmp_path / f"{second}.jpg")
        for second in range(180)
    ]
    candidates = build_candidates(
        duration_sec=180,
        frames=frames,
        audio_energy=[1.0] * 180,
        transcript=[],
        scenes=[],
        audio_events=[],
        window_sec=20,
        stride_sec=4,
        min_candidates=6,
        max_candidates=6,
        segment_sec=90,
        candidates_per_segment=3,
    )

    assert min(item.start_sec for item in candidates) == 0
    assert max(item.end_sec for item in candidates) == 180


def test_nearby_paraphrased_evidence_is_deduplicated() -> None:
    from vh_agent.models import RankedHighlight
    from vh_agent.pipeline import _shares_decisive_evidence

    common = {
        "score": 0.9,
        "local_score": 0.7,
        "judge_score": 0.9,
        "highlight_type": "reveal",
        "description": "真相揭露",
        "reason": "新证据改变认知",
        "confidence": 0.9,
    }
    first = RankedHighlight(
        highlight_id="hl_first",
        start_sec=80,
        end_sec=100,
        evidence=["字幕：父亲拿烟灰缸砸你的头，你流了好多血"],
        **common,
    )
    second = RankedHighlight(
        highlight_id="hl_second",
        start_sec=96,
        end_sec=116,
        evidence=["字幕：他拿烟灰缸砸你的头，你流了很多血"],
        **common,
    )
    assert _shares_decisive_evidence(first, second)


def test_timestamped_transcript_preserves_evidence_times() -> None:
    from vh_agent.candidates import timestamped_transcript

    segments = [
        TranscriptSegment(start_sec=1.25, end_sec=2.5, text="普通铺垫"),
        TranscriptSegment(start_sec=3.0, end_sec=4.75, text="真相出现", source="ocr"),
    ]
    evidence = timestamped_transcript(segments, 2.8, 5.0)
    assert evidence == "[3.00-4.75s OCR] 真相出现"


def test_hypothesis_and_saliency_boundary_are_traceable(tmp_path) -> None:
    from vh_agent.candidates import build_saliency_curve, refine_boundaries
    from vh_agent.models import EventCard
    from vh_agent.reasoning import build_highlight_hypothesis

    event = EventCard(
        start_sec=10,
        end_sec=20,
        action="新证据改变人物目标",
        event_type=["reveal"],
        state_before="人物相信旧事实",
        new_evidence="关键证据出现",
        evidence=["带时间码字幕"],
    )
    hypothesis = build_highlight_hypothesis(event)
    assert hypothesis.statement == "新证据改变人物目标"
    assert hypothesis.trigger == "关键证据出现"
    assert hypothesis.verification_gaps == ["state_after"]

    frames = [
        FrameSample(
            timestamp_sec=float(second),
            path=tmp_path / f"{second}.jpg",
            semantic_change_score=1.0 if second == 15 else 0.0,
        )
        for second in range(30)
    ]
    audio = [0.0] * 30
    audio[15] = 10.0
    transcript = [TranscriptSegment(start_sec=14, end_sec=17, text="关键事实现在揭晓")]
    curve = build_saliency_curve(30, frames, audio, transcript, [], [])
    start, end = refine_boundaries(0, 30, 30, transcript, curve, [15.0])

    assert len(curve) == 30
    assert 13 <= curve.index(max(curve)) <= 17
    assert start <= 15 <= end
    assert 6 <= end - start < 24


def test_scene_centered_short_candidates_cover_brief_events(tmp_path) -> None:
    from vh_agent.models import SceneSegment

    frames = [
        FrameSample(
            timestamp_sec=float(second),
            path=tmp_path / f"{second}.jpg",
            semantic_change_score=1.0 if 40 <= second <= 44 else 0.1,
        )
        for second in range(50)
    ]
    candidates = build_candidates(
        duration_sec=50,
        frames=frames,
        audio_energy=[0.2] * 50,
        transcript=[TranscriptSegment(start_sec=41, end_sec=44, text="关键事实被公开")],
        scenes=[
            SceneSegment(start_sec=0, end_sec=39.5),
            SceneSegment(start_sec=39.5, end_sec=44.5),
            SceneSegment(start_sec=44.5, end_sec=50),
        ],
        audio_events=[],
        window_sec=20,
        stride_sec=4,
        max_candidates=6,
    )
    assert any(
        candidate.end_sec - candidate.start_sec == 8
        and candidate.start_sec <= 41
        and candidate.end_sec >= 44
        for candidate in candidates
    )


def test_decisive_evidence_anchors_boundary_to_its_event() -> None:
    from vh_agent.candidates import refine_boundaries

    curve = [0.1] * 50
    curve[10] = 1.0
    curve[40:45] = [0.5, 0.8, 1.0, 0.8, 0.5]
    start, end = refine_boundaries(5, 47, 50, [], curve, [42.0])
    assert 36 <= start <= 40
    assert 44 <= end <= 47
    assert start <= 42 <= end


def test_saliency_does_not_collapse_a_verified_event_span() -> None:
    from vh_agent.candidates import refine_boundaries

    curve = [0.5] * 50
    curve[9] = 0.0
    curve[15] = 0.0
    start, end = refine_boundaries(5, 20, 50, [], curve, [10.0, 20.0])
    assert start <= 7
    assert end >= 19
    assert end - start >= 12


def test_chapters_use_existing_segment_size_and_overlap(tmp_path) -> None:
    from vh_agent.pipeline import _build_chapters

    frames = [
        FrameSample(timestamp_sec=float(second), path=tmp_path / f"{second}.jpg")
        for second in range(0, 201, 10)
    ]
    candidates = [
        CandidateWindow(
            start_sec=float(start),
            end_sec=float(start + 20),
            local_score=0.5,
            frame_samples=[frames[min(index, len(frames) - 1)]],
        )
        for index, start in enumerate((0, 80, 160))
    ]
    chapters = _build_chapters(
        200,
        candidates,
        frames,
        [TranscriptSegment(start_sec=88, end_sec=92, text="边界事件")],
        [],
        90,
        8,
    )
    assert [(item.start_sec, item.end_sec) for item in chapters] == [
        (0.0, 98.0),
        (82.0, 188.0),
        (172.0, 200.0),
    ]
    assert "边界事件" in chapters[0].transcript
    assert "边界事件" in chapters[1].transcript


def test_chapter_event_dedup_requires_time_and_semantic_match() -> None:
    from vh_agent.pipeline import _deduplicate_events

    first = EventCard(
        start_sec=80,
        end_sec=94,
        action="人物公开身份",
        salience=0.8,
        uncertainty=0.3,
        evidence=["[90s] 公开真实身份"],
    )
    duplicate = EventCard(
        start_sec=82,
        end_sec=95,
        action="角色揭示真实身份",
        salience=0.9,
        uncertainty=0.2,
        evidence=["[90s] 公开真实身份", "F03 91s 震惊反应"],
    )
    distinct = EventCard(
        start_sec=84,
        end_sec=96,
        action="另一人物离开房间",
        salience=0.5,
        uncertainty=0.2,
        evidence=["F04 94s 离开房间"],
    )
    result = _deduplicate_events([first, duplicate, distinct])
    assert result == [duplicate, distinct]


def test_local_candidate_aggregates_atomic_events_into_a_chain(tmp_path) -> None:
    from vh_agent.pipeline import _build_event_chains

    frame = FrameSample(timestamp_sec=12, path=tmp_path / "12.jpg")
    first = EventCard(
        start_sec=10,
        end_sec=12,
        action="人物公开物证",
        event_type=["reveal"],
        new_evidence="物证出现",
        salience=0.8,
        uncertainty=0.2,
        evidence=["[11s OCR] 物证出现"],
    )
    second = EventCard(
        start_sec=12,
        end_sec=15,
        action="众人改变判断",
        event_type=["reversal"],
        state_after="旧判断被推翻",
        salience=0.9,
        uncertainty=0.1,
        evidence=["F01 12s 震惊反应"],
    )
    local = CandidateWindow(
        start_sec=4, end_sec=24, local_score=0.7, transcript="完整上下文", frame_samples=[frame]
    )
    mapped = _build_event_chains([first, second], [local])
    candidate, chain = mapped[0]
    assert candidate is local
    assert (chain.start_sec, chain.end_sec) == (10.0, 15.0)
    assert chain.event_type == ["reveal", "reversal"]
    assert chain.action == "人物公开物证；众人改变判断"
    assert chain.evidence == ["[11s OCR] 物证出现", "F01 12s 震惊反应"]
