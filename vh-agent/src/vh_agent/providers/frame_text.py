"""Full-frame/ROI OCR with real frame times and text polygons."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..preprocessing.subtitle_ocr import _load_ocr
from ..storage import write_json


class FrameText:
    def __init__(self, settings, language):
        self.settings, self.language = settings, language or "zh"
        self.available = importlib.util.find_spec("paddleocr") is not None
        self.profile = {
            "version": settings.ocr_version,
            "language": self.language,
            "device": settings.ocr_device,
        }

    def read(self, frames):
        result = []
        s = self.settings
        ocr = _load_ocr(
            lang="ch" if self.language.startswith("zh") else self.language.split("-")[0],
            ocr_version=s.ocr_version,
            device=s.ocr_device,
        )
        for frame in frames:
            path = Path(frame["path"])
            signature = hashlib.sha256(
                path.read_bytes() + str([s.ocr_version, self.language]).encode()
            ).hexdigest()[:16]
            cache = path.with_name(f"text_{signature}.json")
            if cache.is_file():
                lines = json.loads(cache.read_text())
            else:
                with Image.open(path) as image:
                    predictions = list(ocr.predict(input=np.asarray(image.convert("RGB"))))
                if len(predictions) != 1:
                    raise ValueError("OCR 必须返回一张帧的完整结果。")
                data = predictions[0]
                lines = [
                    {"text": text, "confidence": float(score), "polygon": np.asarray(poly).tolist()}
                    for text, score, poly in zip(
                        data["rec_texts"], data["rec_scores"], data["rec_polys"], strict=True
                    )
                ]
                write_json(cache, lines)
            result.append(
                {**frame, "text_lines": lines, "polygon_coordinates": "pixels_in_returned_frame"}
            )
        return {
            "frames": result,
            "source": "ocr",
            "note": "文字来自指定原帧，时间仅代表该帧；结合原视频判断含义。",
        }
