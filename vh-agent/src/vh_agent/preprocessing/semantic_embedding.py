import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..models import FrameSample, TranscriptSegment


class EmbeddingUnavailable(RuntimeError):
    pass


INSTRUCTION = (
    "Represent this short-drama moment for detecting narrative state changes, "
    "identity reveals, reversals, relationship changes, and repeated content."
)


def score_semantic_transitions(
    frames: list[FrameSample],
    transcript: list[TranscriptSegment],
    model_name: str,
    device: str,
    batch_size: int = 8,
    transition_lag_sec: float = 3.0,
) -> list[FrameSample]:
    """Attach multimodal semantic-change scores using local Qwen3-VL embeddings."""
    if not frames:
        raise EmbeddingUnavailable("No frames available for embedding inference")
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        raise EmbeddingUnavailable(f"Local embedding model does not exist: {model_path}")

    embedder = _load_embedder(str(model_path.resolve()), device)
    vectors: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        inputs = [
            {
                "image": str(frame.path),
                "text": _nearby_text(transcript, frame.timestamp_sec),
                "instruction": INSTRUCTION,
            }
            for frame in batch
        ]
        try:
            encoded = embedder.process(inputs, normalize=True)
        except Exception as exc:
            raise EmbeddingUnavailable(f"Qwen3-VL embedding inference failed: {exc}") from exc
        vectors.append(encoded.float().cpu().numpy())

    embeddings = np.concatenate(vectors, axis=0)
    raw_changes: list[float] = []
    previous_index = 0
    for index, frame in enumerate(frames):
        target_time = frame.timestamp_sec - transition_lag_sec
        while (
            previous_index + 1 < index and frames[previous_index + 1].timestamp_sec <= target_time
        ):
            previous_index += 1
        if target_time < frames[0].timestamp_sec:
            raw_changes.append(0.0)
            continue
        similarity = float(np.dot(embeddings[previous_index], embeddings[index]))
        raw_changes.append(max(0.0, 1.0 - similarity))
    scaled = _robust_transition_scale(raw_changes)
    for frame, score in zip(frames, scaled, strict=True):
        frame.semantic_change_score = score
    return frames


@lru_cache(maxsize=4)
def _load_embedder(model_path: str, device: str) -> object:
    try:
        import torch
    except ImportError as exc:
        raise EmbeddingUnavailable("PyTorch is not installed") from exc

    script = Path(model_path) / "scripts" / "qwen3_vl_embedding.py"
    if not script.exists():
        raise EmbeddingUnavailable(f"Qwen3-VL embedding script is missing: {script}")
    spec = importlib.util.spec_from_file_location("vh_qwen3_vl_embedding", script)
    if spec is None or spec.loader is None:
        raise EmbeddingUnavailable(f"Cannot import embedding script: {script}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, RuntimeError) as exc:
        raise EmbeddingUnavailable(f"Cannot load Qwen3-VL embedding implementation: {exc}") from exc

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise EmbeddingUnavailable(f"CUDA requested but unavailable: {device}")
        if ":" in device:
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return module.Qwen3VLEmbedder(
        model_name_or_path=model_path,
        max_length=2048,
        max_pixels=512 * 32 * 32,
        torch_dtype=dtype,
        default_instruction=INSTRUCTION,
    )


def _nearby_text(segments: list[TranscriptSegment], timestamp_sec: float) -> str:
    selected = [
        segment.text
        for segment in segments
        if segment.end_sec >= timestamp_sec - 1.5 and segment.start_sec <= timestamp_sec + 1.5
    ]
    return " ".join(selected) or "[no speech or subtitle at this moment]"


def _robust_transition_scale(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    nonzero = array[array > 0]
    if nonzero.size < 2:
        return [0.0 for _ in values]
    low, high = np.percentile(nonzero, [35, 90])
    if high <= low + 1e-8:
        return np.clip(array / max(high, 1e-8), 0, 1).tolist()
    return np.clip((array - low) / (high - low), 0, 1).tolist()
