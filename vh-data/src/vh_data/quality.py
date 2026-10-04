"""Keep agreed labels; unmarked or disputed material stays unknown."""

from .models import Annotation, Label, Span, iou, overlap


def union(spans: list[Span]) -> list[Span]:
    result = []
    for span in sorted(spans, key=lambda s: s.start_sec):
        if result and span.start_sec <= result[-1].end_sec:
            result[-1] = Span(
                start_sec=result[-1].start_sec, end_sec=max(result[-1].end_sec, span.end_sec)
            )
        else:
            result.append(Span(start_sec=span.start_sec, end_sec=span.end_sec))
    return result


def overlapping_highlights(segments: list[Label]) -> list[Span]:
    positive = [s for s in segments if s.kind == "positive"]
    return [
        Span(start_sec=min(a.start_sec, b.start_sec), end_sec=max(a.end_sec, b.end_sec))
        for i, a in enumerate(positive)
        for b in positive[i + 1 :]
        if overlap(a, b)
    ]


def subtract(span: Span, excluded: list[Span]) -> list[Span]:
    pieces = [Span(start_sec=span.start_sec, end_sec=span.end_sec)]
    for cut in union(excluded):
        remaining = []
        for piece in pieces:
            if overlap(piece, cut) == 0:
                remaining.append(piece)
                continue
            if piece.start_sec < cut.start_sec:
                remaining.append(Span(start_sec=piece.start_sec, end_sec=cut.start_sec))
            if cut.end_sec < piece.end_sec:
                remaining.append(Span(start_sec=cut.end_sec, end_sec=piece.end_sec))
        pieces = remaining
    return pieces


def contradictions(segments: list[Label]) -> list[Label]:
    """Draft opinions may conflict; the reviewer must resolve the affected labels."""
    return [
        s for s in segments if any(s.kind != other.kind and overlap(s, other) for other in segments)
    ]


def reconcile(
    first: list[Label], second: list[Label], duration: float, threshold: float
) -> tuple[Annotation, list[str]]:
    conflicting = [*contradictions(first), *contradictions(second)]
    left = [s for s in first if s.kind == "positive" and s not in conflicting]
    right = [s for s in second if s.kind == "positive" and s not in conflicting]
    pairs = sorted(
        [(iou(a, b), i, j) for i, a in enumerate(left) for j, b in enumerate(right)], reverse=True
    )
    used_left, used_right = set(), set()
    segments, issues = [], []
    for score, i, j in pairs:
        if score < threshold or i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        segments.append(
            Label(
                start_sec=left[i].start_sec,
                end_sec=left[i].end_sec,
                kind="positive",
                reason=left[i].reason,
            )
        )
    disputed = [s for i, s in enumerate(left) if i not in used_left] + [
        s for j, s in enumerate(right) if j not in used_right
    ]
    uncertain = [s for s in [*first, *second] if s.kind == "uncertain"]
    uncertain.extend(conflicting)
    if conflicting:
        issues.append("同一时间有不同判断，需要核对分类或边界")
    if disputed:
        issues.append("两次独立观看对高光位置或边界有分歧")
    if uncertain:
        issues.append("存在未确认内容")
    protected = [*left, *right, *uncertain]
    negative_intersections = [
        Span(start_sec=max(a.start_sec, b.start_sec), end_sec=min(a.end_sec, b.end_sec))
        for a in first
        for b in second
        if a.kind == b.kind == "negative"
        and a not in conflicting
        and b not in conflicting
        and overlap(a, b) > 0
    ]
    for span in union(negative_intersections):
        segments.extend(
            Label(**piece.model_dump(), kind="negative", reason="两次观看均确认为普通背景或铺垫")
            for piece in subtract(span, protected)
        )
    for span in union([*disputed, *uncertain]):
        segments.extend(
            Label(**piece.model_dump(), kind="uncertain", reason="两次判断不同，需要再看原片")
            for piece in subtract(span, [s for s in segments if s.kind == "positive"])
        )
    positives = [s for s in segments if s.kind == "positive"]
    if any(overlap(a, b) > 0 for i, a in enumerate(positives) for b in positives[i + 1 :]):
        issues.append("重叠候选需要确认是否为同一看点")
    if not segments:
        issues.append("没有明确的可训练标注")
    return Annotation(duration_sec=duration, segments=segments), issues
