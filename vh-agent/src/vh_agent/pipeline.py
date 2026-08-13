import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .candidates import (
    attach_storyboards,
    audio_event_text,
    build_candidates,
    build_saliency_curve,
    refine_boundaries,
    timestamped_transcript,
)
from .config import Settings
from .models import (
    AudioEvent,
    CandidateWindow,
    ChapterContext,
    DecisionTrace,
    DetectionResult,
    DetectionStats,
    DetectionTask,
    DetectionTrace,
    EventCard,
    FrameSample,
    GlobalRanking,
    Highlight,
    HighlightHypothesis,
    JudgeDecision,
    PreprocessTrace,
    RankedHighlight,
    ReasoningTrace,
    StoryMemory,
    TranscriptSegment,
    VideoInfo,
    VideoSummary,
)
from .preprocessing.media import (
    audio_energy_per_second,
    export_clip,
    extract_audio,
    extract_frames,
    probe_video,
    video_fingerprint,
)
from .preprocessing.scene_detection import detect_scenes
from .preprocessing.semantic_embedding import score_semantic_transitions
from .preprocessing.sensevoice import extract_audio_events
from .preprocessing.subtitle_ocr import extract_subtitle_segments
from .preprocessing.transcription import FasterWhisperTranscriber
from .reasoning import SiliconFlowReasoner, build_highlight_hypothesis, build_story_memories


class HighlightOrchestrator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._configure_model_caches()
        self._transcriber: FasterWhisperTranscriber | None = None

    def run(self, task: DetectionTask) -> DetectionResult:
        video = probe_video(task.video_path, language=task.language)
        if not video.has_audio:
            raise ValueError("Short-drama pipeline requires an audio track")

        fingerprint = video_fingerprint(video.path)
        cache_dir = self.settings.media_cache_dir / fingerprint
        job_dir = self.settings.job_output_dir / task.job_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        frames = extract_frames(
            video.path,
            cache_dir / "frames",
            self.settings.sample_fps,
            self.settings.frame_width,
        )
        audio_path = extract_audio(video.path, cache_dir / "audio.wav")
        energies = audio_energy_per_second(audio_path)
        if not energies:
            raise RuntimeError("Audio extraction produced no energy samples")

        scenes = detect_scenes(video.path)
        asr_segments = self._transcribe(audio_path, task.language)
        ocr_frames = _uniformly_sample_frames(frames, self.settings.ocr_max_frames)
        ocr_segments = extract_subtitle_segments(
            ocr_frames,
            task.language,
            self.settings.ocr_device,
            self.settings.ocr_version,
        )
        transcript = _merge_transcript(asr_segments, ocr_segments)
        if not transcript:
            raise RuntimeError("ASR and OCR produced no transcript")

        audio_events = extract_audio_events(
            audio_path,
            self.settings.sensevoice_model,
            self.settings.sensevoice_vad_model,
            self.settings.sensevoice_device,
            video.duration_sec,
        )
        if not audio_events:
            raise RuntimeError("SenseVoice produced no audio events")
        score_semantic_transitions(
            frames,
            transcript,
            self.settings.embedding_model,
            self.settings.embedding_device,
        )
        saliency_curve = build_saliency_curve(
            video.duration_sec, frames, energies, transcript, scenes, audio_events
        )
        candidates = build_candidates(
            duration_sec=video.duration_sec,
            frames=frames,
            audio_energy=energies,
            transcript=transcript,
            scenes=scenes,
            audio_events=audio_events,
            window_sec=self.settings.window_sec,
            stride_sec=self.settings.stride_sec,
            max_candidates=self.settings.max_local_candidates,
            min_candidates=self.settings.min_local_candidates,
            segment_sec=self.settings.candidate_segment_sec,
            candidates_per_segment=self.settings.candidates_per_segment,
            nms_iou=self.settings.candidate_nms_iou,
        )
        if not candidates:
            raise RuntimeError("Candidate generation returned no windows")
        attach_storyboards(candidates, frames)

        reasoner = SiliconFlowReasoner(self.settings)
        chapters = _build_chapters(
            video.duration_sec,
            candidates,
            frames,
            transcript,
            audio_events,
            self.settings.candidate_segment_sec,
            self.settings.context_after_sec,
        )
        events, chapter_map_calls = self._map_chapters(reasoner, video, chapters)
        event_candidates = _build_event_chains(events, candidates)
        event_cards = {id(candidate): event for candidate, event in event_candidates}
        memories_before, _ = build_story_memories(events)
        hypotheses = {
            id(candidate): build_highlight_hypothesis(event)
            for candidate, event in event_candidates
        }
        judge_candidates = _select_judge_candidates(
            [candidate for candidate, _ in event_candidates],
            event_cards,
            max_candidates=self.settings.max_judge_candidates,
            coverage_segment_sec=self.settings.candidate_segment_sec * 2.0,
        )

        decisions, judge_calls = self._judge(
            reasoner,
            video,
            judge_candidates,
            transcript,
            event_cards,
            hypotheses,
            memories_before,
        )
        decision_traces = [
            DecisionTrace(
                candidate=candidate,
                event=event_cards[id(candidate)],
                hypothesis=hypotheses[id(candidate)],
                decision=decisions[id(candidate)],
            )
            for candidate, _ in event_candidates
            if id(candidate) in decisions
        ]

        ranked_highlights: list[RankedHighlight] = []
        for candidate, _ in event_candidates:
            highlight = self._to_highlight(
                duration_sec=video.duration_sec,
                video_path=video.path,
                candidate=candidate,
                event_card=event_cards.get(id(candidate)),
                decision=decisions.get(id(candidate)),
                transcript=transcript,
                saliency_curve=saliency_curve,
            )
            if highlight is not None:
                ranked_highlights.append(highlight)

        ranking_pool = self._build_ranking_pool(ranked_highlights)
        output_limit = _output_limit(
            video.duration_sec,
            self.settings.candidate_segment_sec,
            self.settings.max_highlights,
        )
        if len(ranking_pool) > 1:
            ranking = reasoner.rank_highlights(video, ranking_pool, output_limit)
            listwise_calls = 1
        elif ranking_pool:
            only_id = ranking_pool[0].highlight_id
            ranking = GlobalRanking(
                ranked_highlight_ids=[only_id],
                selected_highlight_ids=[only_id],
                rationale="single verified highlight",
            )
            listwise_calls = 0
        else:
            ranking = GlobalRanking(
                ranked_highlight_ids=[],
                selected_highlight_ids=[],
                rationale="no verified highlights",
            )
            listwise_calls = 0
        selected_ids = set(ranking.selected_highlight_ids)
        ranked_highlights = sorted(
            (item for item in ranking_pool if item.highlight_id in selected_ids),
            key=lambda item: item.start_sec,
        )
        highlights = self._export_clips(video.path, job_dir, ranked_highlights)
        stats = DetectionStats(
            sampled_frames=len(frames),
            detected_scenes=len(scenes),
            transcript_segments=len(transcript),
            ocr_segments=len(ocr_segments),
            audio_events=len(audio_events),
            candidate_windows=len(candidates),
            chapter_map_calls=chapter_map_calls,
            judge_calls=judge_calls,
            listwise_calls=listwise_calls,
        )
        result = DetectionResult(
            job_id=task.job_id,
            video=VideoSummary(
                video_id=task.video_id or f"vid_{fingerprint}",
                title=video.title,
                duration_sec=video.duration_sec,
            ),
            highlights=highlights,
        )
        if self.settings.write_result_file:
            self._write_json(job_dir / "result.json", result)
        if self.settings.write_trace:
            trace = DetectionTrace(
                preprocess=PreprocessTrace(
                    video=video,
                    scenes=scenes,
                    transcript=transcript,
                    audio_events=audio_events,
                    frame_samples=frames,
                    saliency_per_second=saliency_curve,
                ),
                candidates=candidates,
                reasoning=ReasoningTrace(events=events, decisions=decision_traces, ranking=ranking),
                stats=stats,
            )
            self._write_json(job_dir / "trace.json", trace)
        return result

    def _configure_model_caches(self) -> None:
        root = self.settings.model_cache_dir
        cache_paths = {
            "HF_HOME": root / "huggingface",
            "MODELSCOPE_CACHE": root / "modelscope",
            "PADDLE_PDX_CACHE_HOME": root / "paddlex",
        }
        for name, path in cache_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    def _transcribe(
        self,
        audio_path: Path,
        language: str | None,
    ) -> list[TranscriptSegment]:
        if self._transcriber is None:
            self._transcriber = FasterWhisperTranscriber(
                self.settings.asr_model,
                self.settings.asr_device,
                self.settings.asr_compute_type,
            )
        return self._transcriber.transcribe(audio_path, language)

    def _map_chapters(
        self,
        reasoner: SiliconFlowReasoner,
        video: VideoInfo,
        chapters: list[ChapterContext],
    ) -> tuple[list[EventCard], int]:
        mapped_events: list[EventCard] = []
        with ThreadPoolExecutor(max_workers=self.settings.map_workers) as executor:
            futures = [
                executor.submit(reasoner.map_chapter, video, chapter) for chapter in chapters
            ]
            for future in as_completed(futures):
                mapped_events.extend(future.result())
        return _deduplicate_events(mapped_events), len(chapters)

    def _judge(
        self,
        reasoner: SiliconFlowReasoner,
        video: VideoInfo,
        candidates: list[CandidateWindow],
        transcript: list[TranscriptSegment],
        event_cards: dict[int, EventCard],
        hypotheses: dict[int, HighlightHypothesis],
        memories_before: dict[int, StoryMemory],
    ) -> tuple[dict[int, JudgeDecision], int]:
        targets = candidates

        def judge_one(candidate: CandidateWindow) -> JudgeDecision:
            card = event_cards[id(candidate)]
            context_before = timestamped_transcript(
                transcript,
                max(0, candidate.start_sec - self.settings.context_before_sec),
                candidate.start_sec,
            )
            context_core = timestamped_transcript(
                transcript, candidate.start_sec, candidate.end_sec
            )
            context_after = timestamped_transcript(
                transcript,
                candidate.end_sec,
                min(video.duration_sec, candidate.end_sec + self.settings.context_after_sec),
            )
            return reasoner.judge(
                video,
                candidate,
                hypotheses[id(candidate)],
                memories_before.get(id(card), StoryMemory()),
                context_before,
                context_core,
                context_after,
            )

        decisions: dict[int, JudgeDecision] = {}
        with ThreadPoolExecutor(max_workers=self.settings.judge_workers) as executor:
            futures = {executor.submit(judge_one, item): item for item in targets}
            for future in as_completed(futures):
                candidate = futures[future]
                decisions[id(candidate)] = future.result()
        return decisions, len(targets)

    def _to_highlight(
        self,
        *,
        duration_sec: float,
        video_path: Path,
        candidate: CandidateWindow,
        event_card: EventCard | None,
        decision: JudgeDecision | None,
        transcript: list[TranscriptSegment],
        saliency_curve: list[float],
    ) -> RankedHighlight | None:
        if decision is None or event_card is None or not decision.is_highlight:
            return None

        score = 0.15 * candidate.local_score + 0.85 * decision.score
        if decision.score < self.settings.final_threshold:
            return None
        start = decision.start_sec if decision.start_sec is not None else candidate.start_sec
        end = decision.end_sec if decision.end_sec is not None else candidate.end_sec
        start, end = refine_boundaries(
            float(start),
            float(end),
            duration_sec,
            transcript,
            saliency_curve,
            decision.decisive_evidence_times_sec,
        )
        digest = hashlib.sha1(f"{video_path}:{start:.3f}:{end:.3f}".encode()).hexdigest()[:12]
        return RankedHighlight(
            highlight_id=f"hl_{digest}",
            start_sec=start,
            end_sec=end,
            score=round(float(score), 4),
            local_score=round(candidate.local_score, 4),
            judge_score=round(decision.score, 4),
            highlight_type=decision.highlight_type,
            description=decision.description,
            reason=decision.reason,
            transcript=candidate.transcript,
            evidence=decision.evidence,
            confidence=round(decision.confidence, 4),
        )

    def _build_ranking_pool(self, highlights: list[RankedHighlight]) -> list[RankedHighlight]:
        selected: list[RankedHighlight] = []

        def try_add(item: RankedHighlight, *, allow_new: bool = True) -> bool:
            if any(
                _segment_iou(item, other) >= self.settings.final_nms_iou
                or _shorter_overlap_ratio(item, other) >= 0.55
                or _shares_decisive_evidence(item, other)
                for other in selected
            ):
                return False
            if not allow_new:
                return False
            selected.append(item)
            return True

        ranked = sorted(
            highlights,
            key=lambda value: (
                value.judge_score,
                value.end_sec - value.start_sec,
                value.score,
            ),
            reverse=True,
        )
        for item in ranked:
            if len(selected) >= self.settings.max_highlights:
                break
            try_add(item)
        return selected

    @staticmethod
    def _export_clips(
        video_path: Path,
        job_dir: Path,
        highlights: list[RankedHighlight],
    ) -> list[Highlight]:
        clip_dir = job_dir / "clips"
        for stale_clip in clip_dir.glob("*.mp4"):
            stale_clip.unlink()

        exported: list[Highlight] = []
        for item in highlights:
            filename = f"{item.highlight_id}.mp4"
            export_clip(video_path, clip_dir / filename, item.start_sec, item.end_sec)
            exported.append(
                Highlight(
                    highlight_id=item.highlight_id,
                    start_sec=item.start_sec,
                    end_sec=item.end_sec,
                    score=item.score,
                    highlight_type=item.highlight_type,
                    description=item.description,
                    reason=item.reason,
                    clip_url=f"clips/{filename}",
                )
            )
        return exported

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _output_limit(duration_sec: float, segment_sec: float, max_highlights: int) -> int:
    return min(max_highlights, max(3, math.ceil(duration_sec / (segment_sec * 2.0))))


def _build_chapters(
    duration_sec: float,
    candidates: list[CandidateWindow],
    frames: list[FrameSample],
    transcript: list[TranscriptSegment],
    audio_events: list[AudioEvent],
    chapter_sec: float,
    overlap_sec: float,
) -> list[ChapterContext]:
    chapter_count = max(1, math.ceil(duration_sec / chapter_sec))
    chapters: list[ChapterContext] = []
    for index in range(chapter_count):
        core_start = index * chapter_sec
        core_end = min(duration_sec, (index + 1) * chapter_sec)
        start = max(0.0, core_start - (overlap_sec if index else 0.0))
        end = min(
            duration_sec,
            core_end + (overlap_sec if index + 1 < chapter_count else 0.0),
        )
        relevant = [
            candidate
            for candidate in candidates
            if candidate.end_sec >= start and candidate.start_sec <= end
        ]
        keyed_frames = {
            frame.timestamp_sec: frame
            for candidate in relevant
            for frame in candidate.frame_samples
            if start <= frame.timestamp_sec <= end
        }
        if not keyed_frames:
            keyed_frames = {
                frame.timestamp_sec: frame
                for frame in frames
                if start <= frame.timestamp_sec <= end
            }
        chapter_frames = _uniformly_sample_frames(
            [keyed_frames[key] for key in sorted(keyed_frames)], 16
        )
        chapters.append(
            ChapterContext(
                chapter_id=f"chapter_{index + 1:03d}",
                start_sec=start,
                end_sec=end,
                transcript=timestamped_transcript(transcript, start, end),
                audio_context=audio_event_text(audio_events, start, end),
                frame_samples=chapter_frames,
            )
        )
    return chapters


def _deduplicate_events(events: list[EventCard]) -> list[EventCard]:
    selected: list[EventCard] = []

    def signature(event: EventCard) -> str:
        return "".join(
            character
            for character in (event.action + " " + " ".join(event.evidence)).casefold()
            if character.isalnum()
        )

    def duplicate(left: EventCard, right: EventCard) -> bool:
        overlap = max(
            0.0,
            min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec),
        )
        shorter = min(
            left.end_sec - left.start_sec,
            right.end_sec - right.start_sec,
        )
        if shorter <= 0 or overlap / shorter < 0.6:
            return False
        return SequenceMatcher(None, signature(left), signature(right)).ratio() >= 0.5

    for event in sorted(events, key=lambda item: (item.start_sec, item.end_sec)):
        match = next(
            (index for index, item in enumerate(selected) if duplicate(event, item)),
            None,
        )
        if match is None:
            selected.append(event)
            continue
        current = selected[match]
        if (len(event.evidence), -event.uncertainty, event.salience) > (
            len(current.evidence),
            -current.uncertainty,
            current.salience,
        ):
            selected[match] = event
    return sorted(selected, key=lambda item: item.start_sec)


def _build_event_chains(
    events: list[EventCard],
    local_candidates: list[CandidateWindow],
) -> list[tuple[CandidateWindow, EventCard]]:
    mapped: list[tuple[CandidateWindow, EventCard]] = []

    def unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    for candidate in local_candidates:
        matched = [
            event
            for event in events
            if event.end_sec >= candidate.start_sec and event.start_sec <= candidate.end_sec
        ]
        if not matched:
            continue
        matched.sort(key=lambda item: item.start_sec)
        chain = EventCard(
            start_sec=max(candidate.start_sec, min(item.start_sec for item in matched)),
            end_sec=min(candidate.end_sec, max(item.end_sec for item in matched)),
            actors=unique([actor for item in matched for actor in item.actors]),
            action="；".join(unique([item.action for item in matched])),
            event_type=unique([kind for item in matched for kind in item.event_type]),
            state_before=next((item.state_before for item in matched if item.state_before), ""),
            new_evidence="；".join(unique([item.new_evidence for item in matched])),
            state_after=next(
                (item.state_after for item in reversed(matched) if item.state_after), ""
            ),
            relationship_change="；".join(unique([item.relationship_change for item in matched])),
            emotion="；".join(unique([item.emotion for item in matched])),
            salience=max(item.salience for item in matched),
            uncertainty=max(item.uncertainty for item in matched),
            evidence=unique([evidence for item in matched for evidence in item.evidence]),
        )
        mapped.append((candidate, chain))
    return mapped


def _uniformly_sample_frames(frames: list[FrameSample], max_frames: int) -> list[FrameSample]:
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    return [frames[index] for index in indices]


def _select_judge_candidates(
    candidates: list[CandidateWindow],
    event_cards: dict[int, EventCard],
    *,
    max_candidates: int,
    coverage_segment_sec: float,
) -> list[CandidateWindow]:
    ranked = sorted(
        (candidate for candidate in candidates if id(candidate) in event_cards),
        key=lambda candidate: _event_priority(candidate, event_cards[id(candidate)]),
        reverse=True,
    )
    if len(ranked) <= max_candidates:
        return sorted(ranked, key=lambda candidate: candidate.start_sec)

    group_winners: dict[int, CandidateWindow] = {}
    for candidate in ranked:
        midpoint = (candidate.start_sec + candidate.end_sec) / 2.0
        group = int(midpoint / coverage_segment_sec)
        group_winners.setdefault(group, candidate)

    winner_groups = sorted(group_winners)
    if len(winner_groups) > max_candidates:
        indices = np.linspace(0, len(winner_groups) - 1, max_candidates).astype(int)
        winner_groups = [winner_groups[index] for index in indices]
    selected = [group_winners[group] for group in winner_groups]
    selected_ids = {id(candidate) for candidate in selected}
    for candidate in ranked:
        if len(selected) >= max_candidates:
            break
        if id(candidate) not in selected_ids:
            selected.append(candidate)
            selected_ids.add(id(candidate))
    return sorted(selected, key=lambda candidate: candidate.start_sec)


def _event_priority(candidate: CandidateWindow, event_card: EventCard) -> float:
    narrative = float(
        bool({"reversal", "reveal", "payoff", "cliffhanger"}.intersection(event_card.event_type))
    )
    causal = float(bool({"conflict", "action"}.intersection(event_card.event_type)))
    return (
        0.30 * candidate.local_score
        + 0.45 * event_card.salience
        + 0.15 * event_card.uncertainty
        + 0.08 * narrative
        + 0.04 * causal
        - 0.30 * candidate.filter_penalty
    )


def _merge_transcript(
    asr_segments: list[TranscriptSegment],
    ocr_segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    merged = list(asr_segments)
    for ocr in ocr_segments:
        normalized = "".join(ocr.text.split())
        duplicate = any(
            asr.end_sec >= ocr.start_sec - 1.5
            and asr.start_sec <= ocr.end_sec + 1.5
            and normalized in "".join(asr.text.split())
            for asr in asr_segments
        )
        if not duplicate:
            merged.append(ocr)
    return sorted(merged, key=lambda item: (item.start_sec, item.source))


def _segment_iou(left: RankedHighlight, right: RankedHighlight) -> float:
    overlap = max(0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec))
    union = max(left.end_sec, right.end_sec) - min(left.start_sec, right.start_sec)
    return overlap / union if union else 0.0


def _shorter_overlap_ratio(left: RankedHighlight, right: RankedHighlight) -> float:
    overlap = max(0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec))
    shorter = min(left.end_sec - left.start_sec, right.end_sec - right.start_sec)
    return overlap / shorter if shorter > 0 else 0.0


def _shares_decisive_evidence(left: RankedHighlight, right: RankedHighlight) -> bool:
    def keys(item: RankedHighlight) -> set[str]:
        return {
            "".join(character for character in evidence.casefold() if character.isalnum())
            for evidence in item.evidence
            if len("".join(character for character in evidence if character.isalnum())) >= 8
        }

    left_keys = keys(left)
    right_keys = keys(right)
    if left_keys.intersection(right_keys):
        return True
    gap = max(0.0, left.start_sec - right.end_sec, right.start_sec - left.end_sec)
    if gap > 4.0:
        return False
    return any(
        SequenceMatcher(None, left_key, right_key).ratio() >= 0.72
        for left_key in left_keys
        for right_key in right_keys
    )


class HighlightDetectionService:
    """Backend-to-agent entry point."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._orchestrator = HighlightOrchestrator(settings)

    def detect(self, task: DetectionTask) -> DetectionResult:
        return self._orchestrator.run(task)
