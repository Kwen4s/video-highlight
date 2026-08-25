from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import MAX_HIGHLIGHT_SEC
from .models import (
    AudioEvent,
    CandidateWindow,
    FrameSample,
    SceneSegment,
    TranscriptSegment,
)

NARRATIVE_CUES = re.compile(
    r"(原来|其实|没想到|竟然|真相|秘密|身份|不是.*而是|你骗|为什么|不可能|"
    r"离婚|分手|结婚|怀孕|亲生|报仇|复仇|住手|滚|去死|救命|对不起|"
    r"actually|truth|secret|identity|impossible|lied|divorce|marry|pregnant|"
    r"revenge|stop|help|sorry|love you)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateWeights:
    audio: float = 0.20
    visual: float = 0.10
    semantic: float = 0.25
    scene: float = 0.15
    cue: float = 0.30

    def normalized(self) -> CandidateWeights:
        total = self.audio + self.visual + self.semantic + self.scene + self.cue
        if total <= 0:
            raise ValueError("Candidate weights must have a positive sum")
        return CandidateWeights(
            audio=self.audio / total,
            visual=self.visual / total,
            semantic=self.semantic / total,
            scene=self.scene / total,
            cue=self.cue / total,
        )


DEFAULT_WEIGHTS = CandidateWeights()


def candidate_budget(
    duration_sec: float,
    segment_sec: float,
    candidates_per_segment: int,
    min_candidates: int,
    max_candidates: int,
) -> int:
    if segment_sec <= 0 or candidates_per_segment < 1:
        raise ValueError("Candidate segment and quota must be positive")
    segments = max(1, math.ceil(duration_sec / segment_sec))
    return min(max_candidates, max(min_candidates, segments * candidates_per_segment))


def _robust_scale(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(array, [20, 90])
    if high <= low + 1e-8:
        return [0.0 for _ in values]
    return np.clip((array - low) / (high - low), 0, 1).tolist()


def _segment_scale(
    values: list[float],
    starts: list[float],
    window_sec: float,
    segment_sec: float,
) -> list[float]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, start in enumerate(starts):
        groups[int((start + window_sec / 2.0) // segment_sec)].append(index)

    scaled = [0.0] * len(values)
    for indices in groups.values():
        local = _robust_scale([values[index] for index in indices])
        for index, score in zip(indices, local, strict=True):
            scaled[index] = score
    return scaled


def _top_mean(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    return float(np.mean(sorted(values, reverse=True)[:count]))


def transcript_text(
    segments: list[TranscriptSegment],
    start_sec: float,
    end_sec: float,
) -> str:
    selected: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if segment.end_sec < start_sec or segment.start_sec > end_sec:
            continue
        normalized = "".join(segment.text.split())
        if normalized and normalized not in seen:
            selected.append(segment.text)
            seen.add(normalized)
    return " ".join(selected)


def timestamped_transcript(
    segments: list[TranscriptSegment],
    start_sec: float,
    end_sec: float,
) -> str:
    lines: list[str] = []
    seen: set[tuple[float, float, str]] = set()
    for segment in segments:
        if segment.end_sec < start_sec or segment.start_sec > end_sec:
            continue
        key = (segment.start_sec, segment.end_sec, "".join(segment.text.split()))
        if key in seen or not key[2]:
            continue
        seen.add(key)
        lines.append(
            f"[{segment.start_sec:.2f}-{segment.end_sec:.2f}s {segment.source.upper()}] {segment.text}"
        )
    return "\n".join(lines)


def audio_event_text(
    events: list[AudioEvent],
    start_sec: float,
    end_sec: float,
) -> str:
    selected: list[str] = []
    for event in events:
        if event.end_sec < start_sec or event.start_sec > end_sec:
            continue
        if event.emotion == "neutral" and event.event == "speech":
            continue
        label = f"{event.emotion}/{event.event}"
        if label not in selected:
            selected.append(label)
    return " | ".join(selected)


def build_candidates(
    duration_sec: float,
    frames: list[FrameSample],
    audio_energy: list[float],
    transcript: list[TranscriptSegment],
    scenes: list[SceneSegment],
    window_sec: float,
    stride_sec: float,
    max_candidates: int,
    *,
    min_candidates: int = 6,
    segment_sec: float = 90.0,
    candidates_per_segment: int = 2,
    weights: CandidateWeights = DEFAULT_WEIGHTS,
    nms_iou: float = 0.60,
) -> list[CandidateWindow]:
    weights = weights.normalized()
    cut_points = {
        round(point, 3)
        for scene in scenes
        for point in (scene.start_sec, scene.end_sec)
        if 0 < point < duration_sec
    }
    starts = {float(value) for value in np.arange(0, max(duration_sec - 1, 0.1), stride_sec)}
    for cut in cut_points:
        starts.add(max(0.0, min(cut, duration_sec - 1)))
        starts.add(max(0.0, min(cut - window_sec, duration_sec - 1)))
    starts_list = sorted(starts) or [0.0]

    raw_audio: list[float] = []
    raw_visual: list[float] = []
    raw_semantic: list[float] = []
    raw_scenes: list[float] = []
    raw_cues: list[float] = []
    texts: list[str] = []

    for start in starts_list:
        end = min(start + window_sec, duration_sec)
        audio_slice = audio_energy[int(start) : max(math.ceil(end), int(start) + 1)]
        frame_slice = [frame for frame in frames if start <= frame.timestamp_sec <= end]
        text = transcript_text(transcript, start, end)
        raw_audio.append(_top_mean(audio_slice, 0.30))
        raw_visual.append(_top_mean([frame.change_score for frame in frame_slice], 0.20))
        raw_semantic.append(_top_mean([frame.semantic_change_score for frame in frame_slice], 0.30))
        raw_scenes.append(float(sum(start < cut < end for cut in cut_points)))
        raw_cues.append(float(len(NARRATIVE_CUES.findall(text))))
        texts.append(text)

    signal_args = (starts_list, window_sec, segment_sec)
    audio_scores = _segment_scale(raw_audio, *signal_args)
    visual_scores = _segment_scale(raw_visual, *signal_args)
    semantic_scores = _segment_scale(raw_semantic, *signal_args)
    scene_scores = _segment_scale(raw_scenes, *signal_args)
    cue_scores = _segment_scale(raw_cues, *signal_args)
    windows: list[CandidateWindow] = []

    for index, start in enumerate(starts_list):
        end = min(start + window_sec, duration_sec)
        local = (
            weights.audio * audio_scores[index]
            + weights.visual * visual_scores[index]
            + weights.semantic * semantic_scores[index]
            + weights.scene * scene_scores[index]
            + weights.cue * cue_scores[index]
        )
        candidate = CandidateWindow(
            start_sec=float(start),
            end_sec=float(end),
            local_score=float(np.clip(local, 0, 1)),
            audio_score=audio_scores[index],
            visual_score=visual_scores[index],
            semantic_score=semantic_scores[index],
            scene_score=scene_scores[index],
            cue_score=cue_scores[index],
            transcript=texts[index],
        )
        windows.append(apply_content_filter(candidate, duration_sec))

    windows.extend(
        _build_short_scene_candidates(
            duration_sec, frames, audio_energy, transcript, cut_points, window_sec
        )
    )
    target_count = candidate_budget(
        duration_sec,
        segment_sec,
        candidates_per_segment,
        min_candidates,
        max_candidates,
    )
    selected = _select_temporally_balanced(
        windows,
        segment_sec=segment_sec,
        candidates_per_segment=candidates_per_segment,
        target_count=target_count,
        nms_iou=nms_iou,
    )
    return sorted(selected, key=lambda item: item.start_sec)


def _build_short_scene_candidates(
    duration_sec: float,
    frames: list[FrameSample],
    audio_energy: list[float],
    transcript: list[TranscriptSegment],
    cut_points: set[float],
    window_sec: float,
) -> list[CandidateWindow]:
    short_sec = min(8.0, window_sec)
    if short_sec >= window_sec or not cut_points:
        return []
    scaled_audio = _robust_scale(audio_energy)
    windows: list[CandidateWindow] = []
    seen: set[tuple[float, float]] = set()
    for cut in sorted(cut_points):
        start = max(0.0, min(cut - short_sec / 2.0, duration_sec - short_sec))
        end = min(duration_sec, start + short_sec)
        key = (round(start, 3), round(end, 3))
        if key in seen:
            continue
        seen.add(key)
        frame_slice = [frame for frame in frames if start <= frame.timestamp_sec <= end]
        text = transcript_text(transcript, start, end)
        audio_slice = scaled_audio[int(start) : max(math.ceil(end), int(start) + 1)]
        audio_score = _top_mean(audio_slice, 0.30)
        visual_score = _top_mean([frame.change_score for frame in frame_slice], 0.20)
        semantic_score = _top_mean([frame.semantic_change_score for frame in frame_slice], 0.30)
        cue_score = min(1.0, float(len(NARRATIVE_CUES.findall(text))))
        local_score = (
            0.15 * audio_score
            + 0.10 * visual_score
            + 0.30 * semantic_score
            + 0.20
            + 0.25 * cue_score
        )
        windows.append(
            apply_content_filter(
                CandidateWindow(
                    start_sec=start,
                    end_sec=end,
                    local_score=float(np.clip(local_score, 0, 1)),
                    audio_score=audio_score,
                    visual_score=visual_score,
                    semantic_score=semantic_score,
                    scene_score=1.0,
                    cue_score=cue_score,
                    transcript=text,
                ),
                duration_sec,
            )
        )
    return windows


def _select_temporally_balanced(
    windows: list[CandidateWindow],
    *,
    segment_sec: float,
    candidates_per_segment: int,
    target_count: int,
    nms_iou: float,
) -> list[CandidateWindow]:
    groups: dict[int, list[CandidateWindow]] = defaultdict(list)
    duration_sec = max((window.end_sec for window in windows), default=segment_sec)
    group_count = max(1, math.ceil(duration_sec / segment_sec))
    effective_segment_sec = duration_sec / group_count
    for window in windows:
        midpoint = (window.start_sec + window.end_sec) / 2.0
        group = min(group_count - 1, int(midpoint / effective_segment_sec))
        groups[group].append(window)

    selected: list[CandidateWindow] = []
    selected_ids: set[int] = set()

    def try_add(window: CandidateWindow) -> bool:
        if id(window) in selected_ids:
            return False
        if any(_intersection_over_union(window, other) >= nms_iou for other in selected):
            return False
        selected.append(window)
        selected_ids.add(id(window))
        return True

    usable = [window for window in windows if window.filter_penalty < 0.5]
    if usable:
        try_add(min(usable, key=lambda item: item.start_sec))
        if len(selected) < target_count:
            try_add(max(usable, key=lambda item: item.end_sec))

    for group in sorted(groups):
        items = groups[group]
        midpoints = [(item.start_sec + item.end_sec) / 2.0 for item in items]
        low, high = min(midpoints), max(midpoints)
        width = max((high - low) / candidates_per_segment, 1e-6)
        bins: list[list[CandidateWindow]] = [[] for _ in range(candidates_per_segment)]
        for item, midpoint in zip(items, midpoints, strict=True):
            slot = min(candidates_per_segment - 1, int((midpoint - low) / width))
            bins[slot].append(item)
        for pool in bins:
            for window in sorted(pool, key=lambda item: item.local_score, reverse=True):
                if len(selected) >= target_count:
                    break
                if try_add(window):
                    break

    for window in sorted(windows, key=lambda item: item.local_score, reverse=True):
        if len(selected) >= target_count:
            break
        try_add(window)
    return selected


def build_saliency_curve(
    duration_sec: float,
    frames: list[FrameSample],
    audio_energy: list[float],
    transcript: list[TranscriptSegment],
    scenes: list[SceneSegment],
    audio_events: list[AudioEvent],
) -> list[float]:
    size = max(1, math.ceil(duration_sec))

    def scaled(values: np.ndarray) -> np.ndarray:
        return np.asarray(_robust_scale(values.tolist()), dtype=np.float32)

    audio = np.zeros(size, dtype=np.float32)
    usable_audio = min(size, len(audio_energy))
    if usable_audio:
        audio[:usable_audio] = scaled(np.asarray(audio_energy[:usable_audio], dtype=np.float32))

    visual = np.zeros(size, dtype=np.float32)
    semantic = np.zeros(size, dtype=np.float32)
    for frame in frames:
        second = min(size - 1, max(0, int(frame.timestamp_sec)))
        visual[second] = max(visual[second], frame.change_score)
        semantic[second] = max(semantic[second], frame.semantic_change_score)
    visual = scaled(visual)
    semantic = scaled(semantic)

    text = np.zeros(size, dtype=np.float32)
    for segment in transcript:
        begin = min(size - 1, max(0, int(segment.start_sec)))
        finish = min(size, max(begin + 1, math.ceil(segment.end_sec)))
        cue_count = len(NARRATIVE_CUES.findall(segment.text))
        text[begin:finish] = np.maximum(text[begin:finish], min(1.0, 0.25 + 0.25 * cue_count))

    scene = np.zeros(size, dtype=np.float32)
    for item in scenes:
        for boundary in (item.start_sec, item.end_sec):
            if 0 < boundary < duration_sec:
                scene[min(size - 1, int(boundary))] = 1.0

    sound = np.zeros(size, dtype=np.float32)
    for event in audio_events:
        if event.emotion == "neutral" and event.event == "speech":
            continue
        begin = min(size - 1, max(0, int(event.start_sec)))
        finish = min(size, max(begin + 1, math.ceil(event.end_sec)))
        sound[begin:finish] = 1.0

    curve = (
        0.20 * audio + 0.10 * visual + 0.25 * semantic + 0.25 * text + 0.10 * scene + 0.10 * sound
    )
    if size >= 5:
        curve = np.convolve(curve, np.asarray([0.1, 0.2, 0.4, 0.2, 0.1]), mode="same")
    peak = float(curve.max())
    if peak > 1e-8:
        curve = curve / peak
    return np.round(np.clip(curve, 0, 1), 4).tolist()


def _intersection_over_union(left: CandidateWindow, right: CandidateWindow) -> float:
    overlap = max(0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec))
    union = max(left.end_sec, right.end_sec) - min(left.start_sec, right.start_sec)
    return overlap / union if union else 0.0


PROMO_PATTERN = re.compile(
    r"(上集回顾|前情提要|下集预告|未完待续|红果短剧|免费观看全集|评论区|"
    r"点击下载|关注.{0,4}(账号|主播)|点赞|转发|previously on|next episode|"
    r"download|follow us|subscribe)",
    re.IGNORECASE,
)


def apply_content_filter(candidate: CandidateWindow, duration_sec: float) -> CandidateWindow:
    text = candidate.transcript
    matches = list(dict.fromkeys(match.group(0) for match in PROMO_PATTERN.finditer(text)))
    penalty = min(0.8, 0.22 * len(matches))
    reasons = [f"promo_or_recap:{match}" for match in matches]

    if candidate.start_sec <= 3 and not text.strip():
        penalty = max(penalty, 0.15)
        reasons.append("silent_opening")
    if candidate.end_sec >= duration_sec - 1 and matches:
        penalty = max(penalty, 0.55)

    candidate.filter_penalty = penalty
    candidate.filter_reasons = reasons
    candidate.local_score = max(0.0, candidate.local_score * (1.0 - penalty))
    return candidate


def refine_boundaries(
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    transcript: list[TranscriptSegment],
    saliency_curve: list[float],
    setup_evidence_times_sec: list[float],
    decisive_evidence_times_sec: list[float],
) -> tuple[float, float]:
    """Snap semantic boundaries to nearby valleys without discarding evidence."""
    start = max(0.0, min(start_sec - 0.5, duration_sec))
    end = max(start + 0.5, min(end_sec + 0.5, duration_sec))
    anchors = sorted(
        value
        for value in [*setup_evidence_times_sec, *decisive_evidence_times_sec]
        if start <= value <= end and 0 <= value <= duration_sec
    )

    def valley(search_start: float, search_end: float) -> float:
        low = max(0, math.floor(search_start))
        high = min(len(saliency_curve) - 1, math.ceil(search_end))
        if high < low:
            return search_start
        second = min(range(low, high + 1), key=lambda index: saliency_curve[index])
        return float(second)

    if end - start > MAX_HIGHLIGHT_SEC:
        if anchors:
            start = max(start, anchors[0] - 6.0)
            end = min(end, anchors[-1] + 6.0)
        else:
            center = (start + end) / 2.0
            start = max(0.0, center - MAX_HIGHLIGHT_SEC / 2.0)
            end = min(duration_sec, start + MAX_HIGHLIGHT_SEC)
            start = max(0.0, end - MAX_HIGHLIGHT_SEC)
    if anchors and end - start > max(10.0, anchors[-1] - anchors[0] + 8.0):
        start = max(start, anchors[0] - 4.0)
        end = min(end, anchors[-1] + 4.0)

    if saliency_curve:
        original_start, original_end = start, end
        start = valley(max(0.0, start - 2.0), min(end, start + 2.0)) - 0.5
        end = valley(max(start, end - 2.0), min(duration_sec, end + 2.0)) + 1.0
        start = max(original_start, min(start, original_start + 2.0))
        end = min(original_end, max(end, original_end - 2.0))
        if anchors:
            start = max(original_start, min(start, anchors[0] - 1.0))
            end = min(original_end, max(end, anchors[-1] + 1.0))

    if end - start < 6.0:
        center = (start + end) / 2.0
        start = max(0.0, center - 3.0)
        end = min(duration_sec, start + 6.0)
        start = max(0.0, end - 6.0)

    nearby = [
        segment
        for segment in transcript
        if segment.end_sec >= start - 1.25 and segment.start_sec <= end + 1.25
    ]
    start_edges = [
        segment.start_sec for segment in nearby if abs(segment.start_sec - start) <= 1.25
    ]
    end_edges = [segment.end_sec for segment in nearby if abs(segment.end_sec - end) <= 1.25]
    if start_edges:
        start = min(start_edges, key=lambda value: abs(value - start))
    if end_edges:
        end = min(end_edges, key=lambda value: abs(value - end))

    if end - start > MAX_HIGHLIGHT_SEC:
        if anchors:
            start = max(0.0, anchors[0] - 1.0)
            end = min(duration_sec, max(start + 6.0, anchors[-1] + 1.0))
        else:
            center = (start + end) / 2.0
            start = max(0.0, center - MAX_HIGHLIGHT_SEC / 2.0)
            end = min(duration_sec, start + MAX_HIGHLIGHT_SEC)
            start = max(0.0, end - MAX_HIGHLIGHT_SEC)
    end = min(duration_sec, end)
    start = max(0.0, min(start, end - 0.5))
    return round(start, 3), round(end, 3)
