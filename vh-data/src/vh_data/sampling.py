"""Stratified GSPO sampling over the full SFT training pool."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .training_data import VLWindow

CATEGORIES = ("empty", "single", "multiple")


def _category(row: VLWindow) -> str:
    count = len(row.segments)
    return "empty" if count == 0 else "single" if count == 1 else "multiple"


def _quotas(rows: list[VLWindow], size: int) -> dict[str, int]:
    counts = Counter(_category(row) for row in rows)
    result = {kind: size * counts[kind] // len(rows) for kind in CATEGORIES}
    remainder = sorted(CATEGORIES, key=lambda k: (-(size * counts[k] % len(rows)), k))
    for kind in remainder[: size - sum(result.values())]:
        result[kind] += 1
    return result


def _sample_by_drama(rows: list[VLWindow], size: int, rng: random.Random) -> list[VLWindow]:
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r.sample_id):
        groups[row.drama_id].append(row)
    for values in groups.values():
        rng.shuffle(values)
    order = sorted(groups)
    rng.shuffle(order)
    chosen = []
    while len(chosen) < size:
        for drama in order:
            if groups[drama]:
                chosen.append(groups[drama].pop())
                if len(chosen) == size:
                    break
        order = [drama for drama in order if groups[drama]]
    return chosen


def select_gspo(
    rows: list[VLWindow],
    focus_ids: set[str],
    size: int,
    seed: int,
) -> list[VLWindow]:
    if not rows or not 0 < size <= len(rows):
        raise ValueError("GSPO 数量必须在 1 和训练窗口总数之间")
    if not focus_ids <= {r.sample_id for r in rows}:
        raise ValueError("历史采样线索包含训练集之外的窗口")
    quotas = _quotas(rows, size)
    rng = random.Random(seed)
    selected = []
    for kind in CATEGORIES:
        focused = [r for r in rows if _category(r) == kind and r.sample_id in focus_ids]
        regular = [r for r in rows if _category(r) == kind and r.sample_id not in focus_ids]
        if len(focused) > quotas[kind]:
            raise ValueError(f"{kind} 的分层配额不足以容纳历史窗口，请增大 --gspo-size")
        selected.extend(focused)
        selected.extend(_sample_by_drama(regular, quotas[kind] - len(focused), rng))
    rng.shuffle(selected)
    return selected


def stage_statistics(rows: list[VLWindow], focus_ids: set[str]) -> dict:
    categories = Counter(_category(r) for r in rows)
    return {
        "samples": len(rows),
        "videos": len({r.video_id for r in rows}),
        "dramas": len({r.drama_id for r in rows}),
        "categories": {kind: categories[kind] for kind in CATEGORIES},
        "history_focus": sum(r.sample_id in focus_ids for r in rows),
    }
