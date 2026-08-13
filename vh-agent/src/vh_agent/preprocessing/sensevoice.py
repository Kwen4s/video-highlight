import re
from functools import lru_cache
from pathlib import Path

from ..models import AudioEvent


class SenseVoiceUnavailable(RuntimeError):
    pass


EMOTIONS = ("HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED")
EVENTS = ("Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough")


def extract_audio_events(
    audio_path: Path,
    model_path: Path,
    vad_model_path: Path,
    device: str,
    duration_sec: float,
) -> list[AudioEvent]:
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError as exc:
        raise SenseVoiceUnavailable("FunASR is not installed") from exc

    model_path = model_path.expanduser().resolve()
    vad_model_path = vad_model_path.expanduser().resolve()
    if not model_path.is_dir() or not vad_model_path.is_dir():
        raise SenseVoiceUnavailable(
            f"Local SenseVoice/VAD model is missing: {model_path}, {vad_model_path}"
        )
    model = _load_model(str(model_path), str(vad_model_path), device)
    results = model.generate(
        input=str(audio_path),
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        sentence_timestamp=True,
    )
    events: list[AudioEvent] = []
    for result in results:
        raw = str(result.get("text", ""))
        clean = rich_transcription_postprocess(raw)
        emotion = _tag(raw, EMOTIONS, "NEUTRAL").lower()
        event = _tag(raw, EVENTS, "Speech").lower()
        sentence_info = result.get("sentence_info") or []
        if sentence_info:
            for sentence in sentence_info:
                start = float(sentence.get("start", 0)) / 1000.0
                end = float(sentence.get("end", start * 1000)) / 1000.0
                events.append(
                    AudioEvent(
                        start_sec=start,
                        end_sec=end,
                        emotion=emotion,
                        event=event,
                    )
                )
        elif clean or event != "speech" or emotion != "neutral":
            events.append(
                AudioEvent(
                    start_sec=0.0,
                    end_sec=duration_sec,
                    emotion=emotion,
                    event=event,
                )
            )
    return events


@lru_cache(maxsize=4)
def _load_model(model_path: str, vad_model_path: str, device: str) -> object:
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise SenseVoiceUnavailable("FunASR is not installed") from exc
    return AutoModel(
        model=model_path,
        vad_model=vad_model_path,
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        disable_update=True,
    )


def _tag(raw: str, candidates: tuple[str, ...], default: str) -> str:
    for candidate in candidates:
        if re.search(rf"<\|{re.escape(candidate)}\|>", raw, re.IGNORECASE):
            return candidate
    return default
