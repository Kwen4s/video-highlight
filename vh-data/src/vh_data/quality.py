"""Time range operations for reviewed labels and detector feedback."""

from .models import Label, Span, overlap


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
