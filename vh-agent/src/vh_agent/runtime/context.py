"""Responses history with durable image references and committed state rotation."""

import base64
import copy
import hashlib
import json
from pathlib import Path


def text_part(text):
    return {"type": "input_text", "text": text}


def image_part(path):
    path = Path(path)
    return {
        "image_ref": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    }


def hydrate(contents):
    result = copy.deepcopy(contents)
    for item in result:
        for index, part in enumerate(item.get("content", [])):
            if isinstance(part, dict) and "image_ref" in part:
                ref = part["image_ref"]
                data = Path(ref["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != ref["sha256"]:
                    raise ValueError("会话引用的帧文件已改变，请重新建立任务。")
                item["content"][index] = {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
                }
    return result


def estimate(contents):
    size = len(json.dumps(contents, ensure_ascii=False).encode())
    refs = [
        part["image_ref"]
        for item in contents
        for part in item.get("content", [])
        if isinstance(part, dict) and "image_ref" in part
    ]
    image_bytes = sum(4 * ((Path(ref["path"]).stat().st_size + 2) // 3) for ref in refs)
    return len(json.dumps(contents, ensure_ascii=False)) + len(refs) * 2048, size + image_bytes


class Conversation:
    def __init__(self, contents=None, segment=0, phase="analysis"):
        self.contents = contents or []
        self.segment, self.phase = segment, phase

    def checkpoint(self):
        return {"contents": self.contents, "segment": self.segment, "phase": self.phase}

    def _candidate(self, context, parts, *, token_budget, byte_budget, phase):
        tool_outputs = [p for p in parts if p.get("type") == "function_call_output"]
        material = [p for p in parts if p.get("type") != "function_call_output"]
        current = {
            "role": "user",
            "content": [
                text_part(
                    json.dumps(
                        {k: v for k, v in context.items() if k != "last_tool_results"},
                        ensure_ascii=False,
                    )
                ),
                *material,
            ],
        }
        candidate = [*self.contents, *tool_outputs, current]
        tokens, size = estimate(candidate)
        rotated = (
            phase != self.phase or not self.contents or tokens > token_budget or size > byte_budget
        )
        if rotated:
            # Committed tool results are in context; a new segment has no orphan calls.
            current["content"][0] = text_part(json.dumps(context, ensure_ascii=False))
            candidate = [current]
        return candidate, rotated

    def fits(self, context, parts, *, token_budget, byte_budget, phase="analysis"):
        candidate, _ = self._candidate(
            context, parts, token_budget=token_budget, byte_budget=byte_budget, phase=phase
        )
        tokens, size = estimate(candidate)
        return tokens <= token_budget and size <= byte_budget

    def prepare(self, context, parts, *, token_budget, byte_budget, phase="analysis"):
        candidate, rotated = self._candidate(
            context, parts, token_budget=token_budget, byte_budget=byte_budget, phase=phase
        )
        tokens, size = estimate(candidate)
        if tokens > token_budget or size > byte_budget:
            raise ValueError("当前工作状态或媒体超过上下文预算，请缩小观察范围或提高预算。")
        self.contents, self.phase = candidate, phase
        self.segment += int(rotated)
        return hydrate(candidate)
