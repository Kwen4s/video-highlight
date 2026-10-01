"""Lazy, source-attributed text retrieval; video remains the verification evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

from ..preprocessing.media import extract_audio
from ..storage import write_json


class TranscriptSearch:
    def __init__(self, evidence, settings, subtitle_path: Path | None, language: str | None):
        self.evidence, self.settings = evidence, settings
        self.subtitle_path, self.language = subtitle_path, language
        self.profile = {
            "subtitles": hashlib.sha256(subtitle_path.read_bytes()).hexdigest()
            if subtitle_path
            else None,
            "language": language,
            "asr_model": str(settings.asr_model),
            "asr_compute_type": settings.asr_compute_type,
            "asr_device": settings.asr_device,
            "media_id": evidence.media_id,
        }
        self.row_namespace = hashlib.sha256(
            json.dumps(self.profile, sort_keys=True).encode()
        ).hexdigest()[:16]
        self.cache = evidence.output_dir / "transcript.json"
        self.rows = None
        self.available = subtitle_path is not None or (
            evidence.video_info.has_audio
            and settings.asr_model.is_dir()
            and importlib.util.find_spec("faster_whisper") is not None
        )

    def load(self):
        if self.rows is not None:
            return self.rows
        if self.cache.is_file():
            cache = json.loads(self.cache.read_text())
            if cache["profile"] == self.profile:
                self.rows = cache["rows"]
                return self.rows
        if self.subtitle_path:
            import pysubs2

            self.rows = [
                {
                    "start_sec": item.start / 1000,
                    "end_sec": item.end / 1000,
                    "text": item.plaintext,
                    "source": "subtitle",
                }
                for item in pysubs2.load(str(self.subtitle_path))
                if not item.is_comment
            ]
        else:
            if not self.evidence.video_info.has_audio:
                raise ValueError(
                    "No audio or supplied subtitles; inspect native video for on-screen text"
                )
            from ..preprocessing.transcription import FasterWhisperTranscriber

            audio = extract_audio(
                self.evidence.video_info.path, self.cache.parent / "search_audio.wav"
            )
            model = FasterWhisperTranscriber(
                self.settings.asr_model, self.settings.asr_device, self.settings.asr_compute_type
            )
            self.rows = [s.model_dump(mode="json") for s in model.transcribe(audio, self.language)]
            shift = (
                float(self.evidence._audio_streams[0].get("start_time", 0))
                - self.evidence.source_start_time_sec
            )
            for row in self.rows:
                row["start_sec"] = max(0, row["start_sec"] + shift)
                row["end_sec"] = min(self.evidence.video_info.duration_sec, row["end_sec"] + shift)
        self.rows = [r for r in self.rows if r["end_sec"] > r["start_sec"]]
        write_json(self.cache, {"profile": self.profile, "rows": self.rows})
        return self.rows

    def search(self, query, start_sec, end_sec, offset, limit):
        rows = [
            {**r, "row_id": f"{self.row_namespace}:{index}"}
            for index, r in enumerate(self.load())
            if r["end_sec"] > start_sec
            and (end_sec is None or r["start_sec"] < end_sec)
            and (not query or query.casefold() in r["text"].casefold())
        ]
        return {
            "matches": rows[offset : offset + limit],
            "total": len(rows),
            "next_offset": offset + limit if offset + limit < len(rows) else None,
            "evidence_policy": "字幕和 ASR 用于定位内容；观看对应视频，确认事件及发生时间。",
        }
