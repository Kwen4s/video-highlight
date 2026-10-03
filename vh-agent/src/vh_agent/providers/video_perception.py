"""Question-bound audiovisual observations, separately cached from rendered media."""

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..prompts import OBSERVATION_PROMPT, REVIEW_PROMPT
from ..runtime.finalization import ReviewResult
from ..storage import write_json
from .gemini_client import inline_frame_part, inline_video_part, text_part


class PerceivedItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    kind: Literal["action", "dialogue", "reaction", "on_screen_text", "interpretation"]
    content: str = Field(min_length=1)
    speaker: str | None = None


class VideoReading(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    answer: str = Field(min_length=1)
    items: list[PerceivedItem] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class VideoPerception:
    def __init__(self, client, fps, evidence):
        if not math.isfinite(fps) or not 0 < fps <= 24:
            raise ValueError("Video sampling fps must be finite and in (0, 24]")
        self.client, self.fps = client, fps
        self.evidence = evidence
        self.profile = {
            "version": 4,
            "model": client.model,
            "endpoint": client.endpoint,
            "seed": client.seed,
            "thinking_level": client.thinking_level,
            "fps": fps,
            "prompt": OBSERVATION_PROMPT,
            "schema": VideoReading.model_json_schema(),
        }

    def _media_parts(self, path, start, end, fps, *, source_time=False):
        # Explicit frames preserve requested visual density across gateway conversions.
        count = max(1, math.ceil((end - start) * fps))
        frames = self.evidence.frames(
            [start + i * (end - start) / count for i in range(count)], width=768
        )
        parts = [inline_video_part(path)]
        for frame in frames:
            timestamp = (
                frame["timestamp_sec"] if source_time else max(0, frame["timestamp_sec"] - start)
            )
            label = "原片帧" if source_time else "本段原帧"
            parts.extend(
                [
                    text_part(f"{label} {timestamp:.3f} 秒"),
                    inline_frame_part(frame["path"]),
                ]
            )
        return parts

    def review(self, media_path, task, *, start_sec, end_sec, fps, invoke=None):
        reply = (invoke or self.client.generate)(
            REVIEW_PROMPT,
            [text_part(task), *self._media_parts(media_path, start_sec, end_sec, fps)],
            schema=ReviewResult.model_json_schema(),
        )
        return {
            "review": ReviewResult.model_validate(reply["json"]),
            "usage": reply["usage"],
            "model_version": reply["model_version"],
            "thinking_returned": reply["thinking_returned"],
        }

    def encoded_size(self, observation):
        parts = self._media_parts(
            observation.path,
            observation.src_start_sec,
            observation.src_end_sec,
            observation.sampling_fps or self.fps,
            source_time=True,
        )
        return len(json.dumps(parts, ensure_ascii=False).encode())

    def observe(self, observation, question, invoke=None):
        profile = {
            **self.profile,
            "media_sha256": hashlib.sha256(observation.path.read_bytes()).hexdigest(),
            "question": question,
            "fps": observation.sampling_fps or self.fps,
            "source_range_sec": [observation.src_start_sec, observation.src_end_sec],
        }
        key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:24]
        cache = observation.path.parent / f"reading_{key}.json"
        if cache.is_file():
            cached = json.loads(cache.read_text())
            reading, usage, model = cached["reading"], {}, cached["model_version"]
            hit = True
        else:
            reply = (invoke or self.client.generate)(
                OBSERVATION_PROMPT,
                [
                    text_part(
                        json.dumps(
                            {
                                "question": question,
                                "duration_sec": observation.duration_sec,
                                "src_start_sec": observation.src_start_sec,
                                "src_end_sec": observation.src_end_sec,
                                "time_coordinate": "原片秒数；本段视频 0 秒对应 src_start_sec",
                            },
                            ensure_ascii=False,
                        )
                    ),
                    *self._media_parts(
                        observation.path,
                        observation.src_start_sec,
                        observation.src_end_sec,
                        observation.sampling_fps or self.fps,
                        source_time=True,
                    ),
                ],
                schema=VideoReading.model_json_schema(),
                max_output_tokens=8192,
            )
            reading = VideoReading.model_validate(reply["json"]).model_dump(mode="json")
            for item in reading["items"]:
                if (
                    not observation.src_start_sec
                    <= item["start_sec"]
                    <= item["end_sec"]
                    <= observation.src_end_sec + observation.timestamp_precision_sec
                ):
                    raise ValueError("音画观察时间超出实际视频范围。")
                item["end_sec"] = min(item["end_sec"], observation.src_end_sec)
            usage, model = reply["usage"], reply["model_version"]
            write_json(cache, {"profile": profile, "reading": reading, "model_version": model})
            hit = False
        reading = VideoReading.model_validate(reading).model_dump(mode="json")
        return {
            "question": question,
            "observation_id": observation.observation_id,
            **reading,
            "cache_hit": hit,
            "usage": usage,
            "model_version": model,
        }
