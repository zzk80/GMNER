"""Constrained word-level O/B/I CRF for S3.1."""

from __future__ import annotations

import torch
import torch.nn as nn

from gmner.constants import DEFAULT_LABEL2ID, IGNORE_INDEX


BOUNDARY_O = 0
BOUNDARY_B = 1
BOUNDARY_I = 2
NUM_BOUNDARY_TAGS = 3

_TYPED_TO_BOUNDARY = torch.tensor(
    [
        BOUNDARY_O,
        BOUNDARY_B,
        BOUNDARY_I,
        BOUNDARY_B,
        BOUNDARY_I,
        BOUNDARY_B,
        BOUNDARY_I,
        BOUNDARY_B,
        BOUNDARY_I,
    ],
    dtype=torch.long,
)


def typed_bio_to_boundary(
    typed_labels: torch.Tensor,
) -> torch.Tensor:
    """Collapse the repository's 9-way typed BIO labels to O/B/I."""

    output = torch.full_like(typed_labels, IGNORE_INDEX)
    valid = typed_labels.ne(IGNORE_INDEX)
    if valid.any():
        values = typed_labels[valid]
        if values.lt(0).any() or values.ge(len(DEFAULT_LABEL2ID)).any():
            raise ValueError("typed BIO labels contain an unknown ID.")
        mapping = _TYPED_TO_BOUNDARY.to(typed_labels.device)
        output[valid] = mapping[values]
    return output


class WordBoundaryCRF(nn.Module):
    """Linear-chain CRF with a hard O-to-I transition constraint."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(hidden_size, NUM_BOUNDARY_TAGS)
        self.start_transitions = nn.Parameter(
            torch.zeros(NUM_BOUNDARY_TAGS)
        )
        self.end_transitions = nn.Parameter(
            torch.zeros(NUM_BOUNDARY_TAGS)
        )
        self.transitions = nn.Parameter(
            torch.zeros(NUM_BOUNDARY_TAGS, NUM_BOUNDARY_TAGS)
        )
        allowed_start = torch.tensor([True, True, False])
        allowed_transitions = torch.ones(
            NUM_BOUNDARY_TAGS,
            NUM_BOUNDARY_TAGS,
            dtype=torch.bool,
        )
        allowed_transitions[BOUNDARY_O, BOUNDARY_I] = False
        self.register_buffer("allowed_start", allowed_start)
        self.register_buffer(
            "allowed_transitions",
            allowed_transitions,
        )

    def forward(self, word_states: torch.Tensor) -> torch.Tensor:
        return self.emission(self.dropout(word_states))

    def initialize_from_legacy(self, legacy_head: nn.Module) -> None:
        """Deterministically collapse the frozen 9-way BIO head."""

        classifier = getattr(legacy_head, "classifier", None)
        if not isinstance(classifier, nn.Linear):
            raise ValueError("Legacy NER classifier is unavailable.")
        groups = (
            (DEFAULT_LABEL2ID["O"],),
            (
                DEFAULT_LABEL2ID["B-PER"],
                DEFAULT_LABEL2ID["B-LOC"],
                DEFAULT_LABEL2ID["B-ORG"],
                DEFAULT_LABEL2ID["B-OTHER"],
            ),
            (
                DEFAULT_LABEL2ID["I-PER"],
                DEFAULT_LABEL2ID["I-LOC"],
                DEFAULT_LABEL2ID["I-ORG"],
                DEFAULT_LABEL2ID["I-OTHER"],
            ),
        )
        with torch.no_grad():
            for boundary_id, typed_ids in enumerate(groups):
                indices = torch.tensor(
                    typed_ids,
                    dtype=torch.long,
                    device=classifier.weight.device,
                )
                self.emission.weight[boundary_id].copy_(
                    classifier.weight.index_select(0, indices).mean(dim=0)
                )
                self.emission.bias[boundary_id].copy_(
                    classifier.bias.index_select(0, indices).mean()
                )

            legacy_crf = getattr(legacy_head, "crf", None)
            if legacy_crf is None:
                return
            legacy_transitions = _read_crf_tensor(
                legacy_crf,
                ("transitions", "trans_matrix"),
                (len(DEFAULT_LABEL2ID), len(DEFAULT_LABEL2ID)),
            )
            legacy_start = _read_crf_tensor(
                legacy_crf,
                ("start_transitions", "start_trans"),
                (len(DEFAULT_LABEL2ID),),
            )
            legacy_end = _read_crf_tensor(
                legacy_crf,
                ("end_transitions", "end_trans"),
                (len(DEFAULT_LABEL2ID),),
            )
            if legacy_transitions is not None:
                for previous, previous_ids in enumerate(groups):
                    for current, current_ids in enumerate(groups):
                        rows = torch.tensor(
                            previous_ids,
                            device=legacy_transitions.device,
                        )
                        columns = torch.tensor(
                            current_ids,
                            device=legacy_transitions.device,
                        )
                        values = legacy_transitions.index_select(
                            0, rows
                        ).index_select(1, columns)
                        self.transitions[previous, current].copy_(
                            values.mean()
                        )
            if legacy_start is not None:
                for boundary_id, typed_ids in enumerate(groups):
                    indices = torch.tensor(
                        typed_ids,
                        device=legacy_start.device,
                    )
                    self.start_transitions[boundary_id].copy_(
                        legacy_start.index_select(0, indices).mean()
                    )
            if legacy_end is not None:
                for boundary_id, typed_ids in enumerate(groups):
                    indices = torch.tensor(
                        typed_ids,
                        device=legacy_end.device,
                    )
                    self.end_transitions[boundary_id].copy_(
                        legacy_end.index_select(0, indices).mean()
                    )

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return summed CRF NLL divided by the number of valid words."""

        _validate_shapes(emissions, labels, mask)
        working_emissions = emissions.float()
        total = working_emissions.sum() * 0.0
        valid_words = mask.sum()
        for row in range(emissions.size(0)):
            for start, end in _valid_segments(mask[row]):
                row_emissions = working_emissions[row, start:end]
                row_labels = labels[row, start:end]
                if row_labels.eq(IGNORE_INDEX).any():
                    raise ValueError(
                        "A valid Boundary word has an ignored label."
                    )
                self._validate_gold_path(row_labels)
                total = total + (
                    self._log_partition(row_emissions)
                    - self._gold_score(row_emissions, row_labels)
                )
        denominator = valid_words.clamp_min(1).to(emissions.dtype)
        return total / denominator, valid_words

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if emissions.ndim != 3:
            raise ValueError("Boundary emissions must have shape [B,W,3].")
        if mask.shape != emissions.shape[:2]:
            raise ValueError("Boundary mask must have shape [B,W].")
        output = torch.full(
            emissions.shape[:2],
            IGNORE_INDEX,
            dtype=torch.long,
            device=emissions.device,
        )
        working_emissions = emissions.float()
        transitions = self._constrained_transitions(working_emissions)
        start = self._constrained_start(working_emissions)
        for row in range(emissions.size(0)):
            for segment_start, segment_end in _valid_segments(mask[row]):
                values = working_emissions[row, segment_start:segment_end]
                scores = start + values[0]
                history: list[torch.Tensor] = []
                for step in range(1, values.size(0)):
                    candidates = scores[:, None] + transitions
                    scores, previous = candidates.max(dim=0)
                    scores = scores + values[step]
                    history.append(previous)
                current = int(
                    (scores + self.end_transitions).argmax().item()
                )
                path = [current]
                for previous in reversed(history):
                    current = int(previous[current].item())
                    path.append(current)
                path.reverse()
                output[row, segment_start:segment_end] = torch.tensor(
                    path,
                    dtype=torch.long,
                    device=output.device,
                )
        return output

    def _constrained_start(self, emissions: torch.Tensor) -> torch.Tensor:
        return self.start_transitions.masked_fill(
            ~self.allowed_start,
            torch.finfo(emissions.dtype).min,
        )

    def _constrained_transitions(
        self,
        emissions: torch.Tensor,
    ) -> torch.Tensor:
        return self.transitions.masked_fill(
            ~self.allowed_transitions,
            torch.finfo(emissions.dtype).min,
        )

    def _log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        scores = self._constrained_start(emissions) + emissions[0]
        transitions = self._constrained_transitions(emissions)
        for step in range(1, emissions.size(0)):
            scores = torch.logsumexp(
                scores[:, None] + transitions,
                dim=0,
            ) + emissions[step]
        return torch.logsumexp(scores + self.end_transitions, dim=0)

    def _gold_score(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        score = (
            self.start_transitions[labels[0]]
            + emissions[0, labels[0]]
        )
        for step in range(1, emissions.size(0)):
            score = (
                score
                + self.transitions[labels[step - 1], labels[step]]
                + emissions[step, labels[step]]
            )
        return score + self.end_transitions[labels[-1]]

    def _validate_gold_path(self, labels: torch.Tensor) -> None:
        if labels.lt(0).any() or labels.ge(NUM_BOUNDARY_TAGS).any():
            raise ValueError("Boundary labels contain an unknown tag.")
        if not bool(self.allowed_start[labels[0]].item()):
            raise ValueError("Gold Boundary sequence starts with I.")
        if labels.numel() > 1:
            allowed = self.allowed_transitions[
                labels[:-1], labels[1:]
            ]
            if not bool(allowed.all().item()):
                raise ValueError("Gold Boundary sequence contains O->I.")


def _read_crf_tensor(
    crf: nn.Module,
    names: tuple[str, ...],
    shape: tuple[int, ...],
) -> torch.Tensor | None:
    for name in names:
        value = getattr(crf, name, None)
        if isinstance(value, torch.Tensor) and tuple(value.shape) == shape:
            return value.detach()
    return None


def _validate_shapes(
    emissions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if (
        emissions.ndim != 3
        or emissions.size(-1) != NUM_BOUNDARY_TAGS
    ):
        raise ValueError("Boundary emissions must have shape [B,W,3].")
    if labels.shape != emissions.shape[:2]:
        raise ValueError("Boundary labels must have shape [B,W].")
    if mask.shape != emissions.shape[:2]:
        raise ValueError("Boundary mask must have shape [B,W].")


def _valid_segments(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return contiguous valid runs so truncation gaps reset the CRF."""

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(mask.bool().tolist()):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, mask.numel()))
    return segments
