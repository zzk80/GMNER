"""CRF k-best and record-level span candidate utilities.

The candidate generator is deliberately independent from the training model.
It is used to measure whether a structured span/type/region verifier has enough
oracle headroom before the record-level training path is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class SequenceCandidate:
    """One decoded tag sequence and its unnormalized CRF score."""

    tag_ids: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class SpanCandidate:
    """A deduplicated word-level span proposal."""

    start: int
    end: int
    score: float
    source: str
    sequence_ranks: tuple[int, ...] = ()
    boundary_distance: int = 0
    preserve_stage1: bool = False

    @property
    def boundary(self) -> tuple[int, int]:
        return self.start, self.end


def bio_constraint_masks(
    id2label: Mapping[int, str],
    num_tags: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return legal BIO start and prev-to-current transition masks."""

    labels = [str(id2label.get(index, "O")) for index in range(num_tags)]
    allowed_start = torch.ones(num_tags, dtype=torch.bool, device=device)
    allowed_transitions = torch.ones(
        (num_tags, num_tags),
        dtype=torch.bool,
        device=device,
    )

    for current, label in enumerate(labels):
        if not label.startswith("I-"):
            continue
        allowed_start[current] = False
        entity_type = label[2:]
        for previous, previous_label in enumerate(labels):
            allowed_transitions[previous, current] = previous_label in {
                f"B-{entity_type}",
                f"I-{entity_type}",
            }

    return allowed_start, allowed_transitions


def extract_crf_parameters(
    crf: object | None,
    num_tags: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Read transition tensors from either ``torchcrf`` or ``TorchCRF``.

    When CRF initialization fell back to token cross-entropy, zero transition
    tensors preserve a valid emission-only k-best diagnostic.
    """

    def read_tensor(names: Sequence[str], shape: tuple[int, ...]) -> torch.Tensor | None:
        if crf is None:
            return None
        for name in names:
            value = getattr(crf, name, None)
            if isinstance(value, torch.Tensor) and tuple(value.shape) == shape:
                return value.detach().to(device=device, dtype=dtype)
        return None

    transitions = read_tensor(("transitions", "trans_matrix"), (num_tags, num_tags))
    start = read_tensor(("start_transitions", "start_trans"), (num_tags,))
    end = read_tensor(("end_transitions", "end_trans"), (num_tags,))
    source = "crf"
    if transitions is None:
        transitions = torch.zeros((num_tags, num_tags), device=device, dtype=dtype)
        source = "emissions_only"
    if start is None:
        start = torch.zeros(num_tags, device=device, dtype=dtype)
        source = "emissions_only" if crf is None else f"{source}_without_start"
    if end is None:
        end = torch.zeros(num_tags, device=device, dtype=dtype)
        source = "emissions_only" if crf is None else f"{source}_without_end"
    return transitions, start, end, source


@torch.no_grad()
def k_best_viterbi_decode(
    emissions: torch.Tensor,
    *,
    k: int,
    transitions: torch.Tensor | None = None,
    start_transitions: torch.Tensor | None = None,
    end_transitions: torch.Tensor | None = None,
    allowed_start: torch.Tensor | None = None,
    allowed_transitions: torch.Tensor | None = None,
) -> list[SequenceCandidate]:
    """Decode the exact top-k paths of a first-order linear-chain CRF.

    ``transitions[previous, current]`` follows the convention used by
    ``pytorch-crf``. The implementation retains k paths per current tag and is
    exact for the supplied transition system.
    """

    if emissions.ndim != 2:
        raise ValueError("emissions must have shape [sequence_length, num_tags]")
    sequence_length, num_tags = emissions.shape
    if sequence_length == 0 or num_tags == 0:
        return []
    k = max(1, int(k))
    device = emissions.device
    dtype = emissions.dtype

    if transitions is None:
        transitions = torch.zeros((num_tags, num_tags), device=device, dtype=dtype)
    else:
        transitions = transitions.to(device=device, dtype=dtype)
    if tuple(transitions.shape) != (num_tags, num_tags):
        raise ValueError("transitions must have shape [num_tags, num_tags]")

    if start_transitions is None:
        start_transitions = torch.zeros(num_tags, device=device, dtype=dtype)
    else:
        start_transitions = start_transitions.to(device=device, dtype=dtype)
    if end_transitions is None:
        end_transitions = torch.zeros(num_tags, device=device, dtype=dtype)
    else:
        end_transitions = end_transitions.to(device=device, dtype=dtype)

    negative_infinity = torch.tensor(float("-inf"), device=device, dtype=dtype)
    if allowed_start is None:
        allowed_start = torch.ones(num_tags, dtype=torch.bool, device=device)
    else:
        allowed_start = allowed_start.to(device=device, dtype=torch.bool)
    if allowed_transitions is None:
        allowed_transitions = torch.ones(
            (num_tags, num_tags), dtype=torch.bool, device=device
        )
    else:
        allowed_transitions = allowed_transitions.to(device=device, dtype=torch.bool)

    transition_scores = transitions.masked_fill(~allowed_transitions, negative_infinity)
    beam_scores = emissions[0] + start_transitions
    beam_scores = beam_scores.masked_fill(~allowed_start, negative_infinity)
    expanded_scores = torch.full(
        (num_tags, k),
        fill_value=float("-inf"),
        device=device,
        dtype=dtype,
    )
    expanded_scores[:, 0] = beam_scores
    beam_scores = expanded_scores

    back_tags: list[torch.Tensor] = []
    back_ranks: list[torch.Tensor] = []
    for position in range(1, sequence_length):
        # [previous_tag, previous_rank, current_tag]
        candidates = beam_scores[:, :, None] + transition_scores[:, None, :]
        # [current_tag, previous_tag * previous_rank]
        candidates = candidates.permute(2, 0, 1).reshape(num_tags, num_tags * k)
        candidates = candidates + emissions[position].unsqueeze(-1)
        next_scores, source_indices = torch.topk(candidates, k=k, dim=-1)
        back_tags.append(source_indices // k)
        back_ranks.append(source_indices % k)
        beam_scores = next_scores

    final_scores = beam_scores + end_transitions.unsqueeze(-1)
    flattened = final_scores.reshape(-1)
    final_k = min(k, int(torch.isfinite(flattened).sum().item()))
    if final_k <= 0:
        return []
    scores, indices = torch.topk(flattened, k=final_k)

    decoded: list[SequenceCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for score, index in zip(scores.tolist(), indices.tolist()):
        current_tag = int(index // k)
        current_rank = int(index % k)
        reversed_path = [current_tag]
        for step in range(sequence_length - 2, -1, -1):
            previous_tag = int(back_tags[step][current_tag, current_rank].item())
            previous_rank = int(back_ranks[step][current_tag, current_rank].item())
            reversed_path.append(previous_tag)
            current_tag = previous_tag
            current_rank = previous_rank
        path = tuple(reversed(reversed_path))
        if path in seen:
            continue
        seen.add(path)
        decoded.append(SequenceCandidate(tag_ids=path, score=float(score)))
    return decoded


def extract_word_spans(
    tag_ids: Sequence[int],
    word_indices: Sequence[int],
    id2label: Mapping[int, str],
) -> list[tuple[int, int, str]]:
    """Extract typed word spans from a compact BIO sequence."""

    if len(tag_ids) != len(word_indices):
        raise ValueError("tag_ids and word_indices must have equal length")
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    end: int | None = None
    current_type: str | None = None

    def close() -> None:
        nonlocal start, end, current_type
        if start is not None and end is not None and current_type is not None:
            spans.append((start, end, current_type))
        start = None
        end = None
        current_type = None

    for tag_id, word_index in zip(tag_ids, word_indices):
        label = str(id2label.get(int(tag_id), "O"))
        if label == "O" or "-" not in label:
            close()
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or current_type != entity_type or start is None:
            close()
            start = int(word_index)
            end = int(word_index) + 1
            current_type = entity_type
        elif prefix == "I":
            end = max(int(end or 0), int(word_index) + 1)
        else:
            close()
    close()
    return spans


def sequence_hamming_diversity(candidates: Sequence[SequenceCandidate]) -> float:
    """Mean normalized Hamming distance among unique decoded sequences."""

    if len(candidates) < 2:
        return 0.0
    distances: list[float] = []
    for left, right in combinations(candidates, 2):
        length = min(len(left.tag_ids), len(right.tag_ids))
        if length == 0:
            continue
        changed = sum(
            int(left.tag_ids[index] != right.tag_ids[index]) for index in range(length)
        )
        changed += abs(len(left.tag_ids) - len(right.tag_ids))
        distances.append(changed / max(len(left.tag_ids), len(right.tag_ids), 1))
    return sum(distances) / max(len(distances), 1)


def build_span_candidates(
    sequences: Sequence[SequenceCandidate],
    *,
    word_indices: Sequence[int],
    id2label: Mapping[int, str],
    num_words: int,
    max_candidates: int = 12,
    boundary_shift: int = 1,
    boundary_penalty: float = 0.25,
    max_span_length: int = 10,
    required_spans: Sequence[tuple[int, int]] = (),
) -> list[SpanCandidate]:
    """Deduplicate k-best spans and add bounded word-level perturbations."""

    if max_candidates <= 0:
        return []
    sequence_scores = torch.tensor([item.score for item in sequences], dtype=torch.float64)
    sequence_log_probs = (
        sequence_scores - torch.logsumexp(sequence_scores, dim=0)
        if sequence_scores.numel() > 0
        else sequence_scores
    )

    raw: dict[tuple[int, int], dict[str, object]] = {}
    for rank, (sequence, log_probability) in enumerate(
        zip(sequences, sequence_log_probs.tolist())
    ):
        for start, end, _ in extract_word_spans(
            sequence.tag_ids,
            word_indices,
            id2label,
        ):
            if end <= start or end - start > max_span_length:
                continue
            entry = raw.setdefault(
                (start, end),
                {"scores": [], "ranks": set()},
            )
            entry["scores"].append(float(log_probability))
            entry["ranks"].add(rank)

    originals: list[SpanCandidate] = []
    for (start, end), entry in raw.items():
        scores = torch.tensor(entry["scores"], dtype=torch.float64)
        ranks = tuple(sorted(int(value) for value in entry["ranks"]))
        originals.append(
            SpanCandidate(
                start=start,
                end=end,
                score=float(torch.logsumexp(scores, dim=0).item()),
                source="viterbi" if 0 in ranks else "kbest",
                sequence_ranks=ranks,
            )
        )
    originals.sort(key=lambda item: (-item.score, item.start, item.end))

    proposals: dict[tuple[int, int], SpanCandidate] = {
        candidate.boundary: candidate for candidate in originals
    }
    required = {
        (int(start), int(end))
        for start, end in required_spans
        if 0 <= int(start) < int(end) <= num_words
        and int(end) - int(start) <= max_span_length
    }
    for boundary in required:
        existing = proposals.get(boundary)
        if existing is not None:
            proposals[boundary] = SpanCandidate(
                start=existing.start,
                end=existing.end,
                score=existing.score,
                source=existing.source,
                sequence_ranks=existing.sequence_ranks,
                boundary_distance=existing.boundary_distance,
                preserve_stage1=True,
            )
        else:
            # Sequence log-probability proposal scores are non-positive. A zero
            # score plus hard priority keeps the original Stage-1 span in the
            # bounded set without pretending it came from constrained k-best.
            proposals[boundary] = SpanCandidate(
                start=boundary[0],
                end=boundary[1],
                score=0.0,
                source="stage1",
                preserve_stage1=True,
            )
    boundary_shift = max(0, int(boundary_shift))
    for source in originals:
        for amount in range(1, boundary_shift + 1):
            variants = {
                (source.start - amount, source.end),
                (source.start + amount, source.end),
                (source.start, source.end - amount),
                (source.start, source.end + amount),
            }
            for start, end in variants:
                if start < 0 or end > num_words or end <= start:
                    continue
                if end - start > max_span_length or (start, end) in proposals:
                    continue
                distance = abs(start - source.start) + abs(end - source.end)
                proposals[(start, end)] = SpanCandidate(
                    start=start,
                    end=end,
                    score=source.score - float(boundary_penalty) * distance,
                    source="perturbation",
                    sequence_ranks=source.sequence_ranks,
                    boundary_distance=distance,
                )

    source_priority = {"viterbi": 0, "kbest": 1, "perturbation": 2}
    ranked = sorted(
        proposals.values(),
        key=lambda item: (
            not item.preserve_stage1,
            -item.score,
            source_priority.get(item.source, 3),
            item.start,
            item.end,
        ),
    )
    return ranked[: max(1, int(max_candidates))]
