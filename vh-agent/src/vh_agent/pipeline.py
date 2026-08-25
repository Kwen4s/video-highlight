import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import numpy as np

from .candidates import (
    audio_event_text,
    build_candidates,
    build_saliency_curve,
    refine_boundaries,
    timestamped_transcript,
)
from .config import MAX_HIGHLIGHT_SEC, MAX_SCENE_SEC, MIN_SCENE_SEC, Settings
from .models import (
    AudioEvent,
    CandidateWindow,
    DecisionTrace,
    DetectionResult,
    DetectionStats,
    DetectionTask,
    DetectionTrace,
    EvidenceLedger,
    FrameSample,
    GlobalRanking,
    Highlight,
    JudgeConsensus,
    JudgeDecision,
    PreprocessArtifacts,
    PreprocessTrace,
    RankedHighlight,
    ReasoningTrace,
    SceneCard,
    SceneMapArtifact,
    SceneNarrative,
    SceneSegment,
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
from .preprocessing.transcription import ASR_DECODING_POLICY, FasterWhisperTranscriber
from .reasoning import (
    OpenAIReasoner,
    build_evidence_ledgers,
    scene_map_request_fingerprint,
    validate_highlight_decision,
)


class HighlightOrchestrator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._configure_model_caches()
        self._transcriber: FasterWhisperTranscriber | None = None

    def run(self, task: DetectionTask) -> DetectionResult:
        result, _ = self.run_with_trace(task)
        return result

    def run_with_trace(
        self,
        task: DetectionTask,
        *,
        retain_all_verified: bool = False,
    ) -> tuple[DetectionResult, DetectionTrace]:
        video = probe_video(task.video_path, language=task.language)
        if not video.has_audio:
            raise ValueError("Short-drama pipeline requires an audio track")

        fingerprint = video_fingerprint(video.path)
        cache_dir = self.settings.media_cache_dir / fingerprint
        job_dir = self.settings.job_output_dir / task.job_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        if (
            self.settings.export_clips
            or self.settings.write_result_file
            or self.settings.write_trace
        ):
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

        preprocess_signature = _preprocess_signature(self.settings, task.language)
        preprocess_path = cache_dir / "preprocess.json"
        artifacts = _load_preprocess_artifacts(preprocess_path, preprocess_signature)
        if artifacts is not None and len(artifacts.frame_samples) != len(frames):
            artifacts = None
        if artifacts is None:
            ocr_frames = _uniformly_sample_frames(frames, self.settings.ocr_max_frames)
            with ThreadPoolExecutor(max_workers=4) as executor:
                scenes_future = executor.submit(detect_scenes, video.path)
                asr_future = executor.submit(self._transcribe, audio_path, task.language)
                ocr_future = executor.submit(
                    extract_subtitle_segments,
                    ocr_frames,
                    task.language,
                    self.settings.ocr_device,
                    self.settings.ocr_version,
                )
                audio_events_future = executor.submit(
                    extract_audio_events,
                    audio_path,
                    self.settings.sensevoice_model,
                    self.settings.sensevoice_vad_model,
                    self.settings.sensevoice_device,
                    video.duration_sec,
                )
                shot_segments = scenes_future.result()
                asr_segments = asr_future.result()
                ocr_segments = ocr_future.result()
                audio_events = audio_events_future.result()

            transcript = _merge_transcript(asr_segments, ocr_segments)
            if not transcript:
                raise RuntimeError("ASR and OCR produced no transcript")
            if not audio_events:
                raise RuntimeError("SenseVoice produced no audio events")
            score_semantic_transitions(
                frames,
                transcript,
                self.settings.embedding_model,
                self.settings.embedding_device,
            )
            artifacts = PreprocessArtifacts(
                signature=preprocess_signature,
                scenes=shot_segments,
                transcript=transcript,
                audio_events=audio_events,
                frame_samples=frames,
            )
            _write_cache_json(preprocess_path, artifacts)
        else:
            frames = artifacts.frame_samples
            shot_segments = artifacts.scenes
            transcript = artifacts.transcript
            audio_events = artifacts.audio_events
            ocr_segments = [segment for segment in transcript if segment.source == "ocr"]
        saliency_curve = build_saliency_curve(
            video.duration_sec, frames, energies, transcript, shot_segments, audio_events
        )
        candidates = build_candidates(
            duration_sec=video.duration_sec,
            frames=frames,
            audio_energy=energies,
            transcript=transcript,
            scenes=shot_segments,
            window_sec=self.settings.window_sec,
            stride_sec=self.settings.stride_sec,
            max_candidates=self.settings.max_local_candidates,
            min_candidates=self.settings.min_local_candidates,
            segment_sec=self.settings.local_coverage_sec,
            candidates_per_segment=self.settings.candidates_per_segment,
            nms_iou=self.settings.candidate_nms_iou,
        )
        if not candidates:
            raise RuntimeError("Candidate generation returned no windows")
        reasoner = OpenAIReasoner(self.settings)
        semantic_scenes = _build_semantic_scenes(
            video.duration_sec,
            candidates,
            frames,
            transcript,
            audio_events,
            shot_segments,
        )
        mapped_scenes, scene_map_calls = self._map_scenes(
            reasoner,
            video,
            semantic_scenes,
            cache_dir / "scene_map",
        )
        ledgers_before, _ = build_evidence_ledgers(mapped_scenes)
        judge_scenes = _select_judge_scenes(
            mapped_scenes,
            max_candidates=self.settings.max_judge_candidates,
            coverage_segment_sec=self.settings.local_coverage_sec * 2.0,
        )

        decisions, judge_votes, judge_calls = self._judge(
            reasoner,
            video,
            judge_scenes,
            mapped_scenes,
            ledgers_before,
        )
        decision_traces = [
            DecisionTrace(
                scene=scene,
                decision=decisions[scene.scene_id],
                votes=judge_votes[scene.scene_id],
            )
            for scene in mapped_scenes
            if scene.scene_id in decisions
        ]

        verified: list[tuple[SceneCard, JudgeDecision, RankedHighlight]] = []
        for scene in mapped_scenes:
            highlight = self._to_highlight(
                duration_sec=video.duration_sec,
                video_path=video.path,
                scene=scene,
                decision=decisions.get(scene.scene_id),
                transcript=transcript,
                saliency_curve=saliency_curve,
            )
            if highlight is not None:
                verified.append((scene, decisions[scene.scene_id], highlight))

        verified = _attach_trailing_result(verified, mapped_scenes, decisions)
        ranked_highlights = _merge_verified_scenes(verified)
        ranking_pool = self._build_ranking_pool(ranked_highlights)
        if retain_all_verified:
            ranking = _budgeted_ranking(ranking_pool, len(ranking_pool))
            listwise_calls = 0
        else:
            output_limit = _output_limit(
                video.duration_sec,
                self.settings.local_coverage_sec,
                self.settings.max_highlights,
            )
            if len(ranking_pool) > output_limit:
                ranking = reasoner.rank_highlights(video, ranking_pool, output_limit)
                listwise_calls = 1
            else:
                ranking = _budgeted_ranking(ranking_pool, output_limit)
                listwise_calls = 0
        selected_ids = set(ranking.selected_highlight_ids)
        ranked_highlights = _resolve_output_overlaps(
            [item for item in ranking_pool if item.highlight_id in selected_ids]
        )
        ranking = ranking.model_copy(
            update={
                "selected_highlight_ids": [
                    item.highlight_id for item in ranked_highlights
                ]
            }
        )
        if self.settings.export_clips:
            highlights = self._export_clips(video.path, job_dir, ranked_highlights)
        else:
            highlights = self._without_clips(ranked_highlights)
        stats = DetectionStats(
            sampled_frames=len(frames),
            detected_scenes=len(shot_segments),
            transcript_segments=len(transcript),
            ocr_segments=len(ocr_segments),
            audio_events=len(audio_events),
            local_candidate_windows=len(candidates),
            scene_map_calls=scene_map_calls,
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
        trace = DetectionTrace(
            preprocess=PreprocessTrace(
                video=video,
                scenes=shot_segments,
                transcript=transcript,
                audio_events=audio_events,
                frame_samples=frames,
                saliency_per_second=saliency_curve,
            ),
            local_candidates=candidates,
            reasoning=ReasoningTrace(
                scenes=mapped_scenes, decisions=decision_traces, ranking=ranking
            ),
            stats=stats,
        )
        if self.settings.write_result_file:
            self._write_json(job_dir / "result.json", result)
        if self.settings.write_trace:
            self._write_json(job_dir / "trace.json", trace)
        return result, trace

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

    def _map_scenes(
        self,
        reasoner: OpenAIReasoner,
        video: VideoInfo,
        scenes: list[SceneCard],
        cache_dir: Path,
    ) -> tuple[list[SceneCard], int]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        narratives: dict[str, SceneNarrative] = {}
        misses: list[tuple[SceneCard, str, Path]] = []
        for scene in scenes:
            signature = scene_map_request_fingerprint(self.settings, video, scene)
            path = cache_dir / f"{scene.scene_id}.json"
            artifact = _load_scene_map_artifact(path, signature)
            if artifact is None:
                misses.append((scene, signature, path))
            else:
                narratives[scene.scene_id] = artifact.narrative

        def map_and_cache(scene: SceneCard, signature: str, path: Path) -> SceneNarrative:
            narrative = reasoner.map_scene(video, scene)
            _write_cache_json(
                path,
                SceneMapArtifact(signature=signature, narrative=narrative),
            )
            return narrative

        with ThreadPoolExecutor(max_workers=self.settings.map_workers) as executor:
            futures = [
                executor.submit(map_and_cache, scene, signature, path)
                for scene, signature, path in misses
            ]
            for future in as_completed(futures):
                narrative = future.result()
                narratives[narrative.scene_id] = narrative
        mapped = [
            scene.model_copy(update=narratives[scene.scene_id].model_dump()) for scene in scenes
        ]
        return mapped, len(misses)

    def _judge(
        self,
        reasoner: OpenAIReasoner,
        video: VideoInfo,
        targets: list[SceneCard],
        all_scenes: list[SceneCard],
        ledgers_before: dict[str, EvidenceLedger],
    ) -> tuple[dict[str, JudgeDecision], dict[str, list[JudgeDecision]], int]:
        previous_by_id = {
            scene.scene_id: all_scenes[index - 1] if index else None
            for index, scene in enumerate(all_scenes)
        }

        def judge_one(scene: SceneCard) -> JudgeConsensus:
            return reasoner.judge(
                video,
                scene,
                previous_by_id[scene.scene_id],
                ledgers_before[scene.scene_id],
            )

        decisions: dict[str, JudgeDecision] = {}
        votes: dict[str, list[JudgeDecision]] = {}
        calls = 0
        with ThreadPoolExecutor(max_workers=self.settings.judge_workers) as executor:
            futures = {executor.submit(judge_one, item): item for item in targets}
            for future in as_completed(futures):
                scene = futures[future]
                consensus = future.result()
                decisions[scene.scene_id] = validate_highlight_decision(
                    consensus.decision, scene, previous_by_id[scene.scene_id]
                )
                votes[scene.scene_id] = consensus.votes
                calls += consensus.calls
        return decisions, votes, calls

    def _to_highlight(
        self,
        *,
        duration_sec: float,
        video_path: Path,
        scene: SceneCard,
        decision: JudgeDecision | None,
        transcript: list[TranscriptSegment],
        saliency_curve: list[float],
    ) -> RankedHighlight | None:
        if decision is None or not decision.is_highlight:
            return None

        score = 0.15 * scene.local_score + 0.85 * decision.score
        if decision.score < self.settings.final_threshold:
            return None
        start = decision.start_sec if decision.start_sec is not None else scene.start_sec
        end = decision.end_sec if decision.end_sec is not None else scene.end_sec
        start, end = refine_boundaries(
            float(start),
            float(end),
            duration_sec,
            transcript,
            saliency_curve,
            decision.setup_evidence_times_sec,
            decision.decisive_evidence_times_sec,
        )
        digest = hashlib.sha1(f"{video_path}:{start:.3f}:{end:.3f}".encode()).hexdigest()[:12]
        return RankedHighlight(
            highlight_id=f"hl_{digest}",
            start_sec=start,
            end_sec=end,
            score=round(float(score), 4),
            local_score=round(scene.local_score, 4),
            judge_score=round(decision.score, 4),
            highlight_type=decision.highlight_type,
            description=decision.description,
            reason=decision.reason,
            transcript=scene.transcript,
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
    def _without_clips(highlights: list[RankedHighlight]) -> list[Highlight]:
        return [
            Highlight(
                highlight_id=item.highlight_id,
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                score=item.score,
                highlight_type=item.highlight_type,
                description=item.description,
                reason=item.reason,
                clip_url="",
            )
            for item in highlights
        ]

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _preprocess_signature(settings: Settings, language: str | None) -> str:
    payload = {
        "cache_version": "1",
        "language": language,
        "sample_fps": settings.sample_fps,
        "frame_width": settings.frame_width,
        "ocr_max_frames": settings.ocr_max_frames,
        "ocr_version": settings.ocr_version,
        "asr_model": str(settings.asr_model),
        "asr_compute_type": settings.asr_compute_type,
        "asr_decoding_policy": ASR_DECODING_POLICY,
        "sensevoice_model": str(settings.sensevoice_model),
        "sensevoice_vad_model": str(settings.sensevoice_vad_model),
        "embedding_model": settings.embedding_model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_preprocess_artifacts(path: Path, signature: str) -> PreprocessArtifacts | None:
    if not path.is_file():
        return None
    artifact = PreprocessArtifacts.model_validate_json(path.read_text(encoding="utf-8"))
    return artifact if artifact.signature == signature else None


def _load_scene_map_artifact(path: Path, signature: str) -> SceneMapArtifact | None:
    if not path.is_file():
        return None
    artifact = SceneMapArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    return artifact if artifact.signature == signature else None


def _write_cache_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _output_limit(duration_sec: float, segment_sec: float, max_highlights: int) -> int:
    if duration_sec <= segment_sec:
        return min(max_highlights, 2)
    return min(max_highlights, max(2, math.ceil(duration_sec / (segment_sec * 2.0))))


def _budgeted_ranking(highlights: list[RankedHighlight], output_limit: int) -> GlobalRanking:
    """Keep NMS-distinct clips; if they exceed the budget, keep the highest Judge scores."""
    if not highlights:
        return GlobalRanking(
            ranked_highlight_ids=[],
            selected_highlight_ids=[],
            rationale="no verified highlights",
        )
    ordered = sorted(
        highlights,
        key=lambda item: (
            item.judge_score,
            item.end_sec - item.start_sec,
            item.score,
        ),
        reverse=True,
    )
    if len(highlights) <= output_limit:
        selected_ids = [item.highlight_id for item in highlights]
        rationale = "verified highlights within output budget"
    else:
        selected_ids = [item.highlight_id for item in ordered[:output_limit]]
        rationale = "top distinct highlights by judge score"
    return GlobalRanking(
        ranked_highlight_ids=[item.highlight_id for item in ordered],
        selected_highlight_ids=selected_ids,
        rationale=rationale,
    )


def _resolve_output_overlaps(
    highlights: list[RankedHighlight],
) -> list[RankedHighlight]:
    """Split residual overlaps, keeping the stronger event if both cannot fit."""
    ordered = sorted(highlights, key=lambda item: (item.start_sec, item.end_sec))
    resolved: list[RankedHighlight] = []
    for current in ordered:
        if not resolved or current.start_sec >= resolved[-1].end_sec:
            resolved.append(current)
            continue
        previous = resolved[-1]
        split = round((current.start_sec + previous.end_sec) / 2.0, 3)
        if split - previous.start_sec < 0.5 or current.end_sec - split < 0.5:
            if current.judge_score > previous.judge_score:
                resolved[-1] = current
            continue
        resolved[-1] = previous.model_copy(update={"end_sec": split})
        resolved.append(current.model_copy(update={"start_sec": split}))
    return resolved


def _build_semantic_scenes(
    duration_sec: float,
    candidates: list[CandidateWindow],
    frames: list[FrameSample],
    transcript: list[TranscriptSegment],
    audio_events: list[AudioEvent],
    shot_segments: list[SceneSegment],
) -> list[SceneCard]:
    min_scene_sec = MIN_SCENE_SEC
    max_scene_sec = MAX_SCENE_SEC
    shot_boundaries = {
        max(0.0, min(duration_sec, point))
        for shot in shot_segments
        for point in (shot.start_sec, shot.end_sec)
        if 0.0 < point < duration_sec
    }
    shot_boundaries.update(
        float(point) for point in np.arange(max_scene_sec, duration_sec, max_scene_sec)
    )
    ordered_boundaries = sorted(shot_boundaries | {duration_sec})

    def semantic_break(boundary: float) -> bool:
        nearby = [
            frame.semantic_change_score
            for frame in frames
            if abs(frame.timestamp_sec - boundary) <= 1.0
        ]
        return bool(nearby) and max(nearby) >= 0.55

    def dialogue_pause(boundary: float) -> bool:
        if any(segment.start_sec < boundary < segment.end_sec for segment in transcript):
            return False
        earlier = [segment.end_sec for segment in transcript if segment.end_sec <= boundary]
        later = [segment.start_sec for segment in transcript if segment.start_sec >= boundary]
        return bool(earlier and later and min(later) - max(earlier) >= 1.0)

    boundaries = [0.0]
    for boundary in ordered_boundaries:
        span = boundary - boundaries[-1]
        if boundary != duration_sec and span < min_scene_sec:
            continue
        if (
            boundary != duration_sec
            and span < max_scene_sec
            and not (semantic_break(boundary) or dialogue_pause(boundary))
        ):
            continue
        boundaries.append(boundary)

    if boundaries[-1] != duration_sec:
        boundaries.append(duration_sec)
    while len(boundaries) >= 3 and boundaries[-1] - boundaries[-2] < min_scene_sec:
        boundaries.pop(-2)

    cards: list[SceneCard] = []
    for index, (start, end) in enumerate(pairwise(boundaries), start=1):
        local = [
            candidate
            for candidate in candidates
            if candidate.end_sec >= start and candidate.start_sec <= end
        ]
        card_frames = _uniformly_sample_frames(
            [frame for frame in frames if start <= frame.timestamp_sec <= end], 8
        )
        cards.append(
            SceneCard(
                scene_id=f"scene_{index:04d}",
                start_sec=start,
                end_sec=end,
                shot_ids=[
                    f"shot_{shot_index:04d}"
                    for shot_index, shot in enumerate(shot_segments, start=1)
                    if shot.end_sec > start and shot.start_sec < end
                ],
                local_score=max((candidate.local_score for candidate in local), default=0.0),
                transcript=timestamped_transcript(transcript, start, end),
                audio_context=audio_event_text(audio_events, start, end),
                frame_samples=card_frames,
            )
        )
    if not cards:
        raise RuntimeError("Semantic scene construction produced no SceneCards")
    return cards


def _uniformly_sample_frames(frames: list[FrameSample], max_frames: int) -> list[FrameSample]:
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    return [frames[index] for index in indices]


def _select_judge_scenes(
    scenes: list[SceneCard],
    *,
    max_candidates: int,
    coverage_segment_sec: float,
) -> list[SceneCard]:
    ranked = sorted(
        scenes,
        key=_scene_priority,
        reverse=True,
    )
    if len(ranked) <= max_candidates:
        return sorted(ranked, key=lambda candidate: candidate.start_sec)

    group_winners: dict[int, SceneCard] = {}
    for scene in ranked:
        midpoint = (scene.start_sec + scene.end_sec) / 2.0
        group = int(midpoint / coverage_segment_sec)
        group_winners.setdefault(group, scene)

    winner_groups = sorted(group_winners)
    if len(winner_groups) > max_candidates:
        indices = np.linspace(0, len(winner_groups) - 1, max_candidates).astype(int)
        winner_groups = [winner_groups[index] for index in indices]
    selected = [group_winners[group] for group in winner_groups]
    selected_ids = {scene.scene_id for scene in selected}
    for scene in ranked:
        if len(selected) >= max_candidates:
            break
        if scene.scene_id not in selected_ids:
            selected.append(scene)
            selected_ids.add(scene.scene_id)
    return sorted(selected, key=lambda scene: scene.start_sec)


def _scene_priority(scene: SceneCard) -> float:
    narrative = float(
        bool({"reversal", "reveal", "payoff", "cliffhanger"}.intersection(scene.event_type))
    )
    causal = float(bool({"conflict", "action"}.intersection(scene.event_type)))
    return (
        0.30 * scene.local_score
        + 0.45 * scene.salience
        + 0.15 * scene.uncertainty
        + 0.08 * narrative
        + 0.04 * causal
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


def _next_scene_id(scene_id: str) -> str | None:
    prefix, separator, index = scene_id.rpartition("_")
    if not separator or not index.isdigit():
        return None
    return f"{prefix}_{int(index) + 1:0{len(index)}d}"


def _attach_trailing_result(
    verified: list[tuple[SceneCard, JudgeDecision, RankedHighlight]],
    mapped_scenes: list[SceneCard],
    decisions: dict[str, JudgeDecision],
) -> list[tuple[SceneCard, JudgeDecision, RankedHighlight]]:
    """Fold the next scene's reaction into a highlight when Judge says it continues."""
    if not verified:
        return verified
    verified_ids = {scene.scene_id for scene, _, _ in verified}
    scenes = {scene.scene_id: scene for scene in mapped_scenes}
    attached: list[tuple[SceneCard, JudgeDecision, RankedHighlight]] = []
    for scene, decision, highlight in verified:
        next_id = _next_scene_id(scene.scene_id)
        next_scene = scenes.get(next_id) if next_id else None
        next_decision = decisions.get(next_id) if next_id else None
        if (
            next_scene is None
            or next_decision is None
            or next_id in verified_ids
            or not next_decision.continue_previous_scene
            or next_scene.start_sec - scene.end_sec > 2.0
        ):
            attached.append((scene, decision, highlight))
            continue
        trail_end = (
            next_decision.end_sec if next_decision.end_sec is not None else next_scene.end_sec
        )
        combined_end = max(highlight.end_sec, trail_end)
        if combined_end - highlight.start_sec > MAX_HIGHLIGHT_SEC:
            attached.append((scene, decision, highlight))
            continue
        attached.append(
            (
                scene,
                decision,
                highlight.model_copy(
                    update={
                        "end_sec": combined_end,
                        "evidence": list(
                            dict.fromkeys(highlight.evidence + next_decision.evidence)
                        ),
                        "reason": "因果连续结果并入：" + highlight.reason,
                    }
                ),
            )
        )
    return attached


def _merge_verified_scenes(
    verified: list[tuple[SceneCard, JudgeDecision, RankedHighlight]],
) -> list[RankedHighlight]:
    """Join only Judge-approved adjacent causal scene pairs into one playable clip."""

    merged: list[tuple[SceneCard, RankedHighlight]] = []
    for scene, decision, highlight in sorted(verified, key=lambda item: item[0].start_sec):
        if not merged or not decision.continue_previous_scene:
            merged.append((scene, highlight))
            continue
        previous_scene, previous = merged[-1]
        previous_index = int(previous_scene.scene_id.rsplit("_", 1)[-1])
        current_index = int(scene.scene_id.rsplit("_", 1)[-1])
        combined_start = previous.start_sec
        combined_end = highlight.end_sec
        if (
            current_index != previous_index + 1
            or scene.start_sec - previous_scene.end_sec > 2.0
            or combined_end - combined_start > MAX_HIGHLIGHT_SEC
        ):
            merged.append((scene, highlight))
            continue
        digest = hashlib.sha1(
            f"{previous.highlight_id}:{highlight.highlight_id}".encode()
        ).hexdigest()[:12]
        merged[-1] = (
            scene,
            RankedHighlight(
                highlight_id=f"hl_{digest}",
                start_sec=combined_start,
                end_sec=combined_end,
                score=round(max(previous.score, highlight.score), 4),
                local_score=round(max(previous.local_score, highlight.local_score), 4),
                judge_score=round(max(previous.judge_score, highlight.judge_score), 4),
                highlight_type=highlight.highlight_type,
                description="；".join(
                    value for value in (previous.description, highlight.description) if value
                ),
                reason="因果连续场景合并："
                + "；".join(value for value in (previous.reason, highlight.reason) if value),
                transcript="\n".join(
                    value for value in (previous.transcript, highlight.transcript) if value
                ),
                evidence=list(dict.fromkeys(previous.evidence + highlight.evidence)),
                confidence=round(max(previous.confidence, highlight.confidence), 4),
            ),
        )
    return [highlight for _, highlight in merged]


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

    def detect_with_trace(
        self,
        task: DetectionTask,
        *,
        retain_all_verified: bool = False,
    ) -> tuple[DetectionResult, DetectionTrace]:
        return self._orchestrator.run_with_trace(
            task,
            retain_all_verified=retain_all_verified,
        )
