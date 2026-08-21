import re
from dataclasses import dataclass

from .models import DetectionResult, Highlight

MIN_HIGHLIGHT_SEC = 0.5
MAX_HIGHLIGHT_SEC = 24.0
NUMBER_PATTERN = r"(\d+(?:\.\d+)?)"
CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


@dataclass
class EditOutcome:
    result: DetectionResult
    reply: str
    changed: bool = False
    undo: bool = False


def edit_highlights(
    result: DetectionResult,
    message: str,
    selected_highlight_id: str | None,
) -> EditOutcome:
    text = message.strip()
    working = result.model_copy(deep=True)

    if re.search(r"撤销|恢复上一步|undo", text, re.IGNORECASE):
        return EditOutcome(working, "正在撤销上一次高光修改。", undo=True)

    if re.search(r"列出|有哪些|几段|多少.*高光", text):
        if not working.highlights:
            return EditOutcome(working, "当前没有高光候选。")
        summary = "；".join(
            f"第{index}段 {item.start_sec:.1f}–{item.end_sec:.1f}秒「{item.description}」"
            for index, item in enumerate(working.highlights, start=1)
        )
        return EditOutcome(working, f"当前共有 {len(working.highlights)} 段：{summary}。")

    target = _resolve_target(working, text, selected_highlight_id)
    if target is None:
        return EditOutcome(
            working,
            "请先在左侧选择一段高光，或在指令中写明“第 2 段”。",
        )
    highlight, index = target
    label = f"第 {index + 1} 段「{highlight.description}」"

    if re.search(r"删除|移除|去掉|不要这段", text):
        working.highlights = [
            item for item in working.highlights if item.highlight_id != highlight.highlight_id
        ]
        return EditOutcome(working, f"已删除{label}。你可以发送“撤销”恢复。", changed=True)

    direct_range = _parse_direct_range(text)
    if direct_range:
        start_sec, end_sec = direct_range
        error = validate_highlight_range(start_sec, end_sec, working.video.duration_sec)
        if error:
            return EditOutcome(working, error)
        highlight.start_sec = round(start_sec, 3)
        highlight.end_sec = round(end_sec, 3)
        highlight.review_status = "revised"
        return EditOutcome(
            working,
            f"已将{label}调整为 {start_sec:.1f}–{end_sec:.1f} 秒。",
            changed=True,
        )

    title = _extract_replacement(text, ("标题", "名称", "描述"))
    if title is not None:
        if not title:
            return EditOutcome(working, "请说明新的标题，例如“标题改为：真相揭晓”。")
        highlight.description = title[:120]
        highlight.review_status = "revised"
        return EditOutcome(
            working,
            f"已把{label}的标题改为「{highlight.description}」。",
            changed=True,
        )

    reason = _extract_replacement(text, ("说明", "理由", "原因"))
    if reason is not None:
        if not reason:
            return EditOutcome(working, "请说明新的高光说明文本。")
        highlight.reason = reason[:300]
        highlight.review_status = "revised"
        return EditOutcome(working, f"已更新{label}的说明。", changed=True)

    seconds = _extract_seconds(text)
    if re.search(r"前后|两端", text) and re.search(r"缩短|收紧", text):
        if seconds is None:
            return EditOutcome(working, "请说明两端各缩短多少秒。")
        return _apply_boundary_change(working, highlight, index, seconds, -seconds)

    if re.search(r"开头|入点", text):
        if seconds is None:
            return EditOutcome(working, "请说明入点需要移动多少秒。")
        if re.search(r"后移|往后|向后|推后|缩短|收紧", text):
            return _apply_boundary_change(working, highlight, index, seconds, 0)
        if re.search(r"前移|往前|向前|提前|延长", text):
            return _apply_boundary_change(working, highlight, index, -seconds, 0)
        return EditOutcome(working, "请说明入点是前移还是后移，例如“入点后移 1 秒”。")

    if re.search(r"结尾|尾部|出点", text):
        if seconds is None:
            return EditOutcome(working, "请说明出点需要移动多少秒。")
        if re.search(r"前移|往前|向前|提前|缩短|收紧", text):
            return _apply_boundary_change(working, highlight, index, 0, -seconds)
        if re.search(r"后移|往后|向后|推后|延长", text):
            return _apply_boundary_change(working, highlight, index, 0, seconds)
        return EditOutcome(working, "请说明出点是前移还是后移，例如“出点前移 1 秒”。")

    if re.search(r"缩短|延长|收紧", text):
        return EditOutcome(
            working,
            "这条指令会改变时间边界，请说明调整哪一端，例如“入点后移 1 秒”或“出点前移 1 秒”。",
        )

    return EditOutcome(
        working,
        "我目前支持调整入点/出点、指定时间范围、删除高光、修改标题或说明，以及撤销。请给出明确的片段和操作。",
    )


def _resolve_target(
    result: DetectionResult,
    text: str,
    selected_highlight_id: str | None,
) -> tuple[Highlight, int] | None:
    index_match = re.search(r"第\s*(\d+)\s*(?:段|个)", text)
    index: int | None = int(index_match.group(1)) - 1 if index_match else None
    if index is None:
        chinese_match = re.search(r"第\s*(十二|十一|十|[一二三四五六七八九])\s*(?:段|个)", text)
        if chinese_match:
            index = CHINESE_NUMBERS[chinese_match.group(1)] - 1
    if index is not None:
        return (result.highlights[index], index) if 0 <= index < len(result.highlights) else None
    if selected_highlight_id:
        for current_index, item in enumerate(result.highlights):
            if item.highlight_id == selected_highlight_id:
                return item, current_index
    return None


def _parse_direct_range(text: str) -> tuple[float, float] | None:
    clock = re.search(
        r"(\d{1,3}):(\d{1,2}(?:\.\d+)?)\s*(?:到|至|[-—~])\s*"
        r"(\d{1,3}):(\d{1,2}(?:\.\d+)?)",
        text,
    )
    if clock:
        return (
            int(clock.group(1)) * 60 + float(clock.group(2)),
            int(clock.group(3)) * 60 + float(clock.group(4)),
        )
    seconds = re.search(
        rf"{NUMBER_PATTERN}\s*秒?\s*(?:到|至|[-—~])\s*{NUMBER_PATTERN}\s*秒",
        text,
    )
    if seconds:
        return float(seconds.group(1)), float(seconds.group(2))
    return None


def _extract_seconds(text: str) -> float | None:
    match = re.search(rf"{NUMBER_PATTERN}\s*秒", text)
    return float(match.group(1)) if match else None


def _extract_replacement(text: str, fields: tuple[str, ...]) -> str | None:
    field_pattern = "|".join(fields)
    match = re.search(
        rf"(?:{field_pattern})\s*(?:改为|改成|修改为|修改成|是)?\s*[：:]?\s*"
        rf"[「『\"“]?(.+?)[」』\"”]?\s*$",
        text,
    )
    if match:
        return match.group(1).strip(" 。；;")
    if re.search(rf"(?:改写|修改).*(?:{field_pattern})|(?:{field_pattern}).*(?:改写|修改)", text):
        return ""
    return None


def _apply_boundary_change(
    result: DetectionResult,
    highlight: Highlight,
    index: int,
    start_delta: float,
    end_delta: float,
) -> EditOutcome:
    start_sec = max(0.0, highlight.start_sec + start_delta)
    end_sec = min(result.video.duration_sec, highlight.end_sec + end_delta)
    error = validate_highlight_range(start_sec, end_sec, result.video.duration_sec)
    if error:
        return EditOutcome(result, error)
    highlight.start_sec = round(start_sec, 3)
    highlight.end_sec = round(end_sec, 3)
    highlight.review_status = "revised"
    return EditOutcome(
        result,
        f"已将第 {index + 1} 段调整为 {start_sec:.1f}–{end_sec:.1f} 秒。",
        changed=True,
    )


def validate_highlight_range(
    start_sec: float,
    end_sec: float,
    duration_sec: float,
) -> str | None:
    if start_sec < 0 or end_sec > duration_sec:
        return f"时间范围必须位于原片 0–{duration_sec:.1f} 秒之内。"
    duration = end_sec - start_sec
    if duration < MIN_HIGHLIGHT_SEC:
        return f"高光片段至少需要 {MIN_HIGHLIGHT_SEC:.1f} 秒。"
    if duration > MAX_HIGHLIGHT_SEC:
        return f"高光片段最长不能超过 {MAX_HIGHLIGHT_SEC:.0f} 秒。"
    return None
