"""Native conversation segments with durable media references and bounded requests."""

import copy
import hashlib
import json
from pathlib import Path

from ..providers.gemini_client import inline_video_part, text_part


def media_part(observation, fps):
    return {
        "video_ref": {
            "path": str(observation.path),
            "sha256": hashlib.sha256(observation.path.read_bytes()).hexdigest(),
            "observation_id": observation.observation_id,
            "fps": observation.sampling_fps or fps,
            "duration_sec": observation.duration_sec,
        }
    }


def hydrate(contents):
    result = copy.deepcopy(contents)
    for content in result:
        for part in content["parts"]:
            if "video_ref" in part:
                ref = part["video_ref"]
                if hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() != ref["sha256"]:
                    raise ValueError("会话引用的视频文件已改变，请重新建立任务。")
        content["parts"] = [
            inline_video_part(Path(p["video_ref"]["path"]), fps=p["video_ref"]["fps"])
            if "video_ref" in p
            else p
            for p in content["parts"]
        ]
    return result


def estimate(contents):
    """Conservative token estimate; byte budget is measured after serialization."""
    text_size = len(json.dumps(contents, ensure_ascii=False))
    media_tokens = sum(
        p["video_ref"]["duration_sec"] * (p["video_ref"]["fps"] * 512 + 32)
        for c in contents
        for p in c["parts"]
        if "video_ref" in p
    )
    media_bytes = sum(
        4 * ((Path(p["video_ref"]["path"]).stat().st_size + 2) // 3)
        for c in contents
        for p in c["parts"]
        if "video_ref" in p
    )
    return text_size + media_tokens, len(
        json.dumps(contents, ensure_ascii=False).encode()
    ) + media_bytes


class Conversation:
    def __init__(self, contents=None, segment=0, phase="analysis"):
        self.contents = contents or []
        self.segment = segment
        self.phase = phase

    def checkpoint(self):
        return {"contents": self.contents, "segment": self.segment, "phase": self.phase}

    def prepare(
        self, context, parts, *, token_budget, byte_budget, replay_parts=(), phase="analysis"
    ):
        # Runtime rendering and independent reviews are not model tool responses.
        # Refresh committed facts on every turn, including within the same segment.
        current = text_part(json.dumps(context, ensure_ascii=False))
        candidate = [*self.contents, {"role": "user", "parts": [current, *parts]}]
        tokens, size = estimate(candidate)
        if phase != self.phase or not self.contents or tokens > token_budget or size > byte_budget:
            # A new session receives committed results as facts, never orphan function responses.
            material = [p for p in parts if "functionResponse" not in p]
            candidate = [
                {
                    "role": "user",
                    "parts": [current, *material],
                }
            ]
            # Previously delivered media are a cache, not a second mandatory upload queue.
            # Preserve their identities and explicitly expose which ones need rereading.
            visible = [p["video_ref"]["observation_id"] for p in material if "video_ref" in p]
            replay = [p for p in replay_parts if "video_ref" in p]
            deferred = [p["video_ref"]["observation_id"] for p in replay]

            def access_note(loaded, omitted):
                return text_part(
                    json.dumps(
                        {
                            "visible_observation_ids": loaded,
                            "read_again_observation_ids": omitted,
                            "media_note": "未装载的观察仍在状态中；需要再次核对画面时，按原片范围调用 inspect_interval。",
                        },
                        ensure_ascii=False,
                    )
                )

            if replay:
                candidate[0]["parts"].append(access_note(visible, deferred))
            for part in replay:
                key = part["video_ref"]["observation_id"]
                trial = copy.deepcopy(candidate)
                loaded = [*visible, key]
                omitted = [item for item in deferred if item != key]
                trial[0]["parts"][-1:] = [part, access_note(loaded, omitted)]
                trial_tokens, trial_bytes = estimate(trial)
                if trial_tokens <= token_budget and trial_bytes <= byte_budget:
                    candidate, visible, deferred = trial, loaded, omitted
            self.segment += 1
        tokens, size = estimate(candidate)
        if tokens > token_budget or size > byte_budget:
            raise ValueError("当前工作视图或单段媒体超过上下文预算；请缩小观察区间或提高运行预算。")
        self.contents = candidate
        self.phase = phase
        return hydrate(candidate)
