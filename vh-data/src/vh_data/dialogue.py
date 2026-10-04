"""Timestamped dialogue: reuse source-specific evidence or transcribe new media."""

import hashlib
import json
from pathlib import Path

from vh_agent.models import TranscriptSegment
from vh_agent.preprocessing.media import extract_audio, video_fingerprint
from vh_agent.preprocessing.transcription import FasterWhisperTranscriber

from .config import Settings
from .models import digest


class DialogueLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.transcriber = None

    def load(self, video: dict, media) -> tuple[list[dict], str]:
        settings = self.settings
        previous = (
            settings.preprocess_cache / video_fingerprint(Path(video["path"])) / "preprocess.json"
        )
        reuse = previous.is_file() and media.source_start_time_sec == 0
        profile = {
            "source_sha256": video["sha256"],
            "duration_sec": media.video_info.duration_sec,
            "source_start_time_sec": media.source_start_time_sec,
            "upstream_sha256": digest(previous) if reuse else None,
            "recognition": {"method": "existing_preprocess"}
            if reuse
            else {
                "model": str(settings.asr_model),
                "beam_size": 5,
                "compute_type": settings.asr_compute_type,
            },
            "language": video["language"],
        }
        identifier = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
        path = settings.root / "dialogues" / f"{identifier}.json"
        if path.exists():
            record = json.loads(path.read_text())
            return record["segments"], str(path.relative_to(settings.root))
        if reuse:
            payload = json.loads(previous.read_text())
            segments = [TranscriptSegment.model_validate(t) for t in payload["transcript"]]
            origin = str(previous)
        elif media.video_info.has_audio:
            if self.transcriber is None:
                self.transcriber = FasterWhisperTranscriber(
                    settings.asr_model, settings.asr_device, settings.asr_compute_type
                )
            observation = media.inspect(0, media.video_info.duration_sec)
            audio = extract_audio(observation.path, media.output_dir / "dialogue.wav")
            segments = self.transcriber.transcribe(audio, video["language"], beam_size=5)
            origin = "source_aligned_whisper"
        else:
            segments, origin = [], "silent_video"
        clean = []
        for segment in segments:
            # OCR spans and audio can extend beyond the last displayed frame. Keep only
            # the part that belongs to this video's authoritative timeline.
            start = max(0, segment.start_sec)
            end = min(media.video_info.duration_sec, segment.end_sec)
            if end > start:
                clean.append(
                    {
                        "start_sec": start,
                        "end_sec": end,
                        "text": segment.text,
                        "source": segment.source,
                    }
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"input": profile, "origin": origin, "segments": clean},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
        return clean, str(path.relative_to(settings.root))
