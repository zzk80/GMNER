"""Train-only CRF radius estimation and constrained Viterbi reachability."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

import torch


@dataclass(frozen=True)
class KBestSequence:
    labels: tuple[int, ...]
    score: float


def _crf_parameters(crf) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        return crf.start_transitions, crf.transitions, crf.end_transitions
    except AttributeError as exc:
        raise TypeError("TP reachability requires a torchcrf-compatible CRF.") from exc


def sequence_score(emissions: torch.Tensor, labels: torch.Tensor, crf) -> torch.Tensor:
    """Unnormalized CRF sequence score for one valid sequence."""

    if emissions.ndim != 2 or labels.ndim != 1 or emissions.size(0) != labels.numel():
        raise ValueError("Expected emissions [L,C] and labels [L].")
    if labels.numel() == 0:
        return emissions.sum() * 0.0
    start, transitions, end = _crf_parameters(crf)
    positions = torch.arange(labels.numel(), device=labels.device)
    score = start[labels[0]] + emissions[positions, labels].sum()
    if labels.numel() > 1:
        score = score + transitions[labels[:-1], labels[1:]].sum()
    return score + end[labels[-1]]


def k_best_viterbi(emissions: torch.Tensor, crf, k: int = 2) -> list[KBestSequence]:
    """Exact small-k Viterbi for one sequence."""

    if emissions.ndim != 2 or emissions.size(0) == 0:
        raise ValueError("k-best Viterbi requires non-empty [L,C] emissions.")
    k = max(1, int(k))
    start, transitions, end = _crf_parameters(crf)
    num_labels = emissions.size(1)
    beams: list[list[tuple[float, tuple[int, ...]]]] = [
        [(float((start[tag] + emissions[0, tag]).item()), (tag,))]
        for tag in range(num_labels)
    ]
    for position in range(1, emissions.size(0)):
        next_beams: list[list[tuple[float, tuple[int, ...]]]] = []
        for tag in range(num_labels):
            candidates: list[tuple[float, tuple[int, ...]]] = []
            for previous in range(num_labels):
                edge = float((transitions[previous, tag] + emissions[position, tag]).item())
                candidates.extend((score + edge, path + (tag,)) for score, path in beams[previous])
            candidates.sort(key=lambda item: (-item[0], item[1]))
            next_beams.append(candidates[:k])
        beams = next_beams
    completed = [
        (score + float(end[tag].item()), path)
        for tag, tag_beams in enumerate(beams)
        for score, path in tag_beams
    ]
    completed.sort(key=lambda item: (-item[0], item[1]))
    return [KBestSequence(labels=path, score=score) for score, path in completed[:k]]


def estimate_sequence_radius(emissions: torch.Tensor, crf) -> float:
    best = k_best_viterbi(emissions, crf, k=2)
    if len(best) < 2:
        return 0.0
    distance = sum(left != right for left, right in zip(best[0].labels, best[1].labels))
    return max(0.0, (best[0].score - best[1].score) / (2.0 * max(distance, 1)))


def estimate_train_rho(sequences: Iterable[tuple[torch.Tensor, object]]) -> float:
    values = [estimate_sequence_radius(emissions, crf) for emissions, crf in sequences]
    if not values:
        raise ValueError("Cannot estimate rho from an empty Train sequence set.")
    return float(median(values))


def constrained_gold_reachability(
    emissions: torch.Tensor,
    gold_labels: torch.Tensor,
    crf,
    rho: float,
    tolerance: float = 1e-7,
) -> dict[str, float | bool | tuple[int, ...]]:
    """Evaluate max_y S(y)-S(g)-2*rho*d_H(y,g) <= 0 exactly."""

    if rho < 0:
        raise ValueError("rho must be non-negative.")
    penalty = torch.full_like(emissions, -2.0 * float(rho))
    positions = torch.arange(gold_labels.numel(), device=gold_labels.device)
    penalty[positions, gold_labels] = 0.0
    best = k_best_viterbi(emissions + penalty, crf, k=1)[0]
    gold_score = float(sequence_score(emissions, gold_labels, crf).item())
    objective = best.score - gold_score
    return {
        "reachable": bool(objective <= float(tolerance)),
        "max_objective": float(objective),
        "best_labels": best.labels,
        "gold_score": gold_score,
        "rho": float(rho),
    }
