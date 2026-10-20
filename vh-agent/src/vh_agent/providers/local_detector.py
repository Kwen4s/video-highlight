"""Local detector client: global 20-second responsibility windows and durable results."""

import json
import math
import time

import httpx

from ..models import FrameSample
from ..preprocessing.subtitle_ocr import OCR_MERGE_POLICY, extract_subtitle_segments
from ..runtime.state import identity
from ..storage import write_json

WINDOW_SEC = 20
CONTEXT_SEC = 15
HISTORY_SEC = 90


class DetectorError(RuntimeError):
    pass


def detection_windows(start, end, duration):
    if not 0 <= start < end <= duration:
        raise ValueError("检测范围必须位于原视频内")
    return [
        {
            "start_sec": i * WINDOW_SEC,
            "end_sec": min((i + 1) * WINDOW_SEC, duration),
            "read_start_sec": max(0, i * WINDOW_SEC - CONTEXT_SEC),
            "read_end_sec": min(duration, (i + 1) * WINDOW_SEC + CONTEXT_SEC),
        }
        for i in range(math.floor(start / WINDOW_SEC), math.ceil(end / WINDOW_SEC))
    ]


class LocalDetector:
    def __init__(self, evidence, settings, transcript=None, language=None):
        self.evidence, self.settings, self.transcript = evidence, settings, transcript
        self.language = language or "zh"
        self.http = httpx.Client(
            base_url=settings.local_detector_url,
            timeout=settings.request_timeout_sec,
            trust_env=False,
        )
        try:
            response = self.http.get("/profile")
            response.raise_for_status()
            model = response.json()
        except (httpx.HTTPError, ValueError):
            self.http.close()
            raise DetectorError("本地检测服务不可用，检查服务地址和启动日志") from None
        self.profile = {
            "model": model,
            "media_id": evidence.media_id,
            "transcript": transcript.profile if transcript else None,
            "ocr": {
                "version": settings.ocr_version,
                "language": self.language,
                "fps": 1,
                "merge_policy": OCR_MERGE_POLICY,
            },
            "window_sec": WINDOW_SEC,
            "context_sec": CONTEXT_SEC,
            "history_sec": HISTORY_SEC,
        }
        if model["window_sec"] != WINDOW_SEC or model["context_sec"] != CONTEXT_SEC:
            self.http.close()
            raise DetectorError("服务窗口配置与训练输入不一致")
        self.folder = evidence.output_dir / ("detector_" + identity(self.profile)[:24])
        self.folder.mkdir(exist_ok=True)

    def close(self):
        self.http.close()

    def _dialogue(self, window):
        start, end = window["read_start_sec"], window["read_end_sec"]
        rows = []
        if self.transcript is not None:
            rows.extend(
                {
                    "start_sec": r["start_sec"],
                    "end_sec": r["end_sec"],
                    "text": r["text"],
                    "source": "asr",
                }
                for r in self.transcript.load()
                if max(0, start - HISTORY_SEC) <= r["start_sec"] < r["end_sec"] <= end
            )
        times = [float(i) for i in range(math.ceil(start), math.ceil(end))]
        frames = [
            FrameSample(timestamp_sec=f["timestamp_sec"], path=f["path"])
            for f in self.evidence.frames(times, width=768)
        ]
        rows.extend(
            r.model_dump(exclude={"confidence"})
            for r in extract_subtitle_segments(
                frames,
                self.language,
                self.settings.ocr_device,
                self.settings.ocr_version,
                cache_dir=self.folder / "ocr",
            )
        )
        return sorted(
            [
                {
                    **r,
                    "start_sec": round(r["start_sec"] - start, 6),
                    "end_sec": round(r["end_sec"] - start, 6),
                }
                for r in rows
                if max(0, start - HISTORY_SEC) <= r["start_sec"] < r["end_sec"] <= end
            ],
            key=lambda r: (r["start_sec"], r["end_sec"]),
        )

    def detect(self, start_sec, end_sec):
        segments, records = [], []
        duration = self.evidence.video_info.duration_sec
        for window in detection_windows(start_sec, end_sec, duration):
            key = identity(window)
            path = self.folder / (key + ".json")
            started = time.perf_counter()
            hit = path.is_file()
            if hit:
                result = json.loads(path.read_text())
            else:
                request = {
                    "video_path": str(self.evidence.video_info.path.resolve()),
                    "source_start_time_sec": self.evidence.source_start_time_sec,
                    "window": window,
                    "dialogue": self._dialogue(window),
                }
                try:
                    response = self.http.post("/detect", json=request)
                    response.raise_for_status()
                    result = response.json()
                except httpx.HTTPError as exc:
                    code = (
                        exc.response.status_code
                        if isinstance(exc, httpx.HTTPStatusError)
                        else type(exc).__name__
                    )
                    raise DetectorError(
                        f"本地检测窗口 {window['start_sec']:g}–{window['end_sec']:g} 秒失败：{code}；已完成窗口保留"
                    ) from None
                if result["model_revision"] != self.profile["model"]["revision"]:
                    raise DetectorError("运行期间检测权重改变，请重新建立任务")
                write_json(path, result)
            records.append(
                {
                    **window,
                    "cache_hit": hit,
                    "elapsed_sec": time.perf_counter() - started,
                    "generation_sec": result["generation_sec"],
                    "invalid_segments": result["invalid_segments"],
                    "model_revision": result["model_revision"],
                }
            )
            segments.extend(
                s
                for s in result["segments"]
                if start_sec <= (s["start_sec"] + s["end_sec"]) / 2 < end_sec
            )
        unique = {(s["start_sec"], s["end_sec"]): s for s in segments}
        return {
            "segments": sorted(unique.values(), key=lambda s: (s["start_sec"], s["end_sec"])),
            "windows": records,
        }
