import math
from pathlib import Path

from ..models import TranscriptSegment


class ASRUnavailable(RuntimeError):
    pass


ASR_DECODING_POLICY = "greedy_v1"


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_path: Path,
        device: str,
        compute_type: str,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRUnavailable(
                "faster-whisper is not installed; run pip install -e '.[enhanced]'"
            ) from exc
        model_path = model_path.expanduser().resolve()
        if not model_path.is_dir():
            raise ASRUnavailable(f"Local Whisper model is missing: {model_path}")
        device_name, _, index = device.partition(":")
        device_index = int(index) if index else 0
        self.model = WhisperModel(
            str(model_path),
            device=device_name,
            device_index=device_index,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[TranscriptSegment]:
        segments, _ = self.model.transcribe(
            str(audio_path),
            language=language if language in {"zh", "en"} else None,
            vad_filter=True,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        return [
            TranscriptSegment(
                start_sec=item.start,
                end_sec=item.end,
                text=item.text.strip(),
                source="asr",
                confidence=float(math.exp(item.avg_logprob))
                if item.avg_logprob is not None
                else None,
            )
            for item in segments
            if item.text.strip()
        ]
