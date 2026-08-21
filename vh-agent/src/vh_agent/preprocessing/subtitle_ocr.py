import json
from functools import lru_cache

import numpy as np
from PIL import Image

from ..models import FrameSample, TranscriptSegment


class OCRUnavailable(RuntimeError):
    pass


OCR_BATCH_SIZE = 8


def extract_subtitle_segments(
    frames: list[FrameSample],
    language: str | None,
    device: str,
    ocr_version: str = "PP-OCRv6",
    min_confidence: float = 0.55,
) -> list[TranscriptSegment]:
    ocr = _load_ocr(
        # The English collection contains English audio with Chinese hard subtitles.
        lang="ch" if language in {"zh", "en", None} else "en",
        ocr_version=ocr_version,
        device=device,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    raw_segments: list[TranscriptSegment] = []
    for start in range(0, len(frames), OCR_BATCH_SIZE):
        batch_frames = frames[start : start + OCR_BATCH_SIZE]
        images = [_subtitle_region(frame) for frame in batch_frames]
        predictions = _predict_many(ocr, images)
        for frame, (texts, scores) in zip(batch_frames, predictions, strict=True):
            kept_pairs = [
                (text.strip(), score)
                for text, score in zip(texts, scores, strict=True)
                if text.strip() and score >= min_confidence
            ]
            if kept_pairs:
                raw_segments.append(
                    TranscriptSegment(
                        start_sec=frame.timestamp_sec,
                        end_sec=frame.timestamp_sec + 1.5,
                        text=" ".join(text for text, _score in kept_pairs),
                        source="ocr",
                        confidence=float(np.mean([score for _text, score in kept_pairs])),
                    )
                )
    return _merge_repeated(raw_segments)


@lru_cache(maxsize=8)
def _load_ocr(
    *,
    lang: str,
    ocr_version: str,
    device: str,
    enable_mkldnn: bool,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
) -> object:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRUnavailable("PaddleOCR is not installed") from exc
    return PaddleOCR(
        lang=lang,
        ocr_version=ocr_version,
        device=device,
        enable_mkldnn=enable_mkldnn,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
    )


def _subtitle_region(frame: FrameSample) -> np.ndarray:
    with Image.open(frame.path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        return np.asarray(rgb.crop((0, int(height * 0.42), width, height)))


def _predict_many(ocr: object, images: list[np.ndarray]) -> list[tuple[list[str], list[float]]]:
    if not images:
        return []
    if not hasattr(ocr, "predict"):
        raise OCRUnavailable("PaddleOCR 3.7+ predict API is required")
    output = list(ocr.predict(input=images))
    if len(output) != len(images):
        raise OCRUnavailable(f"PaddleOCR returned {len(output)} results for {len(images)} frames")
    return [_parse_prediction(item) for item in output]


def _parse_prediction(item: object) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    scores: list[float] = []
    payload = getattr(item, "json", item)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise OCRUnavailable(f"Unexpected PaddleOCR result type: {type(payload).__name__}")
    result = payload.get("res", payload)
    texts.extend(str(value) for value in result.get("rec_texts", []))
    scores.extend(float(value) for value in result.get("rec_scores", []))
    if len(texts) != len(scores):
        raise OCRUnavailable("PaddleOCR returned mismatched texts and scores")
    return texts, scores


def _merge_repeated(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for segment in segments:
        normalized = "".join(segment.text.split())
        if merged and normalized == "".join(merged[-1].text.split()):
            merged[-1].end_sec = segment.end_sec
            continue
        merged.append(segment)
    return merged
