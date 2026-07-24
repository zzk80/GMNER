"""Non-overlapping record-level entity set decoding."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class IntervalCandidate:
    index: int
    start: int
    end: int
    score: float


def weighted_interval_decode(
    spans: Sequence[tuple[int, int]],
    scores: Sequence[float],
    *,
    threshold: float = 0.0,
) -> list[int]:
    """Return the maximum-score set of non-overlapping half-open spans."""

    candidates = sorted(
        (
            IntervalCandidate(i, int(span[0]), int(span[1]), float(score))
            for i, (span, score) in enumerate(zip(spans, scores))
            if int(span[0]) < int(span[1]) and float(score) > float(threshold)
        ),
        key=lambda item: (item.end, item.start, -item.score, item.index),
    )
    if not candidates:
        return []
    ends = [item.end for item in candidates]
    previous = [
        bisect_right(ends, item.start, hi=index) - 1
        for index, item in enumerate(candidates)
    ]
    best = [0.0] * (len(candidates) + 1)
    take = [False] * len(candidates)
    for index, item in enumerate(candidates, start=1):
        include = item.score + best[previous[index - 1] + 1]
        exclude = best[index - 1]
        take[index - 1] = include > exclude
        best[index] = include if include > exclude else exclude

    selected: list[int] = []
    index = len(candidates)
    while index > 0:
        item = candidates[index - 1]
        include = item.score + best[previous[index - 1] + 1]
        if take[index - 1] and include >= best[index - 1]:
            selected.append(item.index)
            index = previous[index - 1] + 1
        else:
            index -= 1
    return sorted(selected, key=lambda i: (int(spans[i][0]), int(spans[i][1])))


def greedy_interval_decode(
    spans: Sequence[tuple[int, int]],
    scores: Sequence[float],
    *,
    threshold: float = 0.0,
) -> list[int]:
    """Diagnostic greedy decoder ordered by utility."""

    ranked = sorted(
        range(min(len(spans), len(scores))),
        key=lambda index: (-float(scores[index]), spans[index][0], spans[index][1]),
    )
    selected: list[int] = []
    for index in ranked:
        start, end = map(int, spans[index])
        if float(scores[index]) <= float(threshold) or start >= end:
            continue
        if any(
            not (end <= spans[other][0] or spans[other][1] <= start)
            for other in selected
        ):
            continue
        selected.append(index)
    return sorted(selected, key=lambda i: (int(spans[i][0]), int(spans[i][1])))

