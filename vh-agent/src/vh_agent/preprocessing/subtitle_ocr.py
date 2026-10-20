import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from ..models import FrameSample, TranscriptSegment
from ..storage import write_json


class OCRUnavailable(RuntimeError):
    pass


OCR_BATCH_SIZE = 8
OCR_MERGE_POLICY = "contiguous_v1"


def extract_subtitle_segments(
    frames: list[FrameSample],
    language: str,
    device: str,
    ocr_version: str = "PP-OCRv6",
    min_confidence: float = 0.55,
    cache_dir: Path | None = None,
) -> list[TranscriptSegment]:
    raw_segments: list[TranscriptSegment] = []
    for start in range(0, len(frames), OCR_BATCH_SIZE):
        batch_frames = frames[start : start + OCR_BATCH_SIZE]
        paths = [
            cache_dir
            / (
                hashlib.sha256(
                    frame.path.read_bytes()
                    + str([language, ocr_version, "subtitle_region_0.42"]).encode()
                ).hexdigest()
                + ".json"
            )
            if cache_dir is not None
            else None
            for frame in batch_frames
        ]
        predictions = [
            json.loads(path.read_text()) if path is not None and path.is_file() else None
            for path in paths
        ]
        missing = [i for i, result in enumerate(predictions) if result is None]
        if missing:
            ocr = _load_ocr(
                lang="ch" if language.startswith("zh") else language.split("-", 1)[0],
                ocr_version=ocr_version,
                device=device,
            )
            fresh = _predict_many(ocr, [_subtitle_region(batch_frames[i]) for i in missing])
            for i, result in zip(missing, fresh, strict=True):
                predictions[i] = result
                if paths[i] is not None:
                    write_json(paths[i], result)
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
) -> object:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRUnavailable("PaddleOCR is not installed") from exc
    return PaddleOCR(
        lang=lang,
        ocr_version=ocr_version,
        device=device,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _subtitle_region(frame: FrameSample) -> np.ndarray:
    with Image.open(frame.path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        return np.asarray(rgb.crop((0, int(height * 0.42), width, height)))


def _predict_many(ocr: object, images: list[np.ndarray]) -> list[tuple[list[str], list[float]]]:
    output = list(ocr.predict(input=images))
    if len(output) != len(images):
        raise OCRUnavailable(f"PaddleOCR returned {len(output)} results for {len(images)} frames")
    return [
        ([str(text) for text in item["rec_texts"]], [float(score) for score in item["rec_scores"]])
        for item in output
    ]


def _merge_repeated(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for segment in segments:
        normalized = "".join(segment.text.split())
        if (
            merged
            and segment.start_sec <= merged[-1].end_sec + 1e-6
            and normalized == "".join(merged[-1].text.split())
        ):
            merged[-1].end_sec = segment.end_sec
            continue
        merged.append(segment)
    return merged
