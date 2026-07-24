from __future__ import annotations

import itertools

import torch

from gmner.constants import DEFAULT_LABEL2ID
from gmner.utils.candidate_decoding import (
    SequenceCandidate,
    bio_constraint_masks,
    build_span_candidates,
    extract_word_spans,
    k_best_viterbi_decode,
    sequence_hamming_diversity,
)


def _brute_force_paths(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    k: int,
) -> list[tuple[tuple[int, ...], float]]:
    paths = []
    for tags in itertools.product(range(emissions.size(1)), repeat=emissions.size(0)):
        score = start[tags[0]] + emissions[0, tags[0]]
        for position in range(1, emissions.size(0)):
            score = (
                score
                + transitions[tags[position - 1], tags[position]]
                + emissions[position, tags[position]]
            )
        score = score + end[tags[-1]]
        paths.append((tuple(tags), float(score.item())))
    return sorted(paths, key=lambda item: item[1], reverse=True)[:k]


def test_k_best_viterbi_matches_brute_force() -> None:
    emissions = torch.tensor(
        [
            [0.2, 0.8, -0.1],
            [0.5, -0.2, 0.4],
            [0.1, 0.3, 0.7],
        ]
    )
    transitions = torch.tensor(
        [
            [0.1, -0.2, 0.0],
            [0.3, 0.2, -0.1],
            [-0.2, 0.4, 0.1],
        ]
    )
    start = torch.tensor([0.0, 0.2, -0.1])
    end = torch.tensor([0.1, -0.1, 0.2])

    expected = _brute_force_paths(emissions, transitions, start, end, k=6)
    actual = k_best_viterbi_decode(
        emissions,
        k=6,
        transitions=transitions,
        start_transitions=start,
        end_transitions=end,
    )

    assert [candidate.tag_ids for candidate in actual] == [item[0] for item in expected]
    assert torch.allclose(
        torch.tensor([candidate.score for candidate in actual]),
        torch.tensor([item[1] for item in expected]),
    )


def test_bio_constraints_disallow_inside_at_start_and_after_other_type() -> None:
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    allowed_start, allowed_transitions = bio_constraint_masks(
        id2label,
        len(id2label),
        device=torch.device("cpu"),
    )

    inside_per = DEFAULT_LABEL2ID["I-PER"]
    begin_per = DEFAULT_LABEL2ID["B-PER"]
    inside_org = DEFAULT_LABEL2ID["I-ORG"]
    assert not allowed_start[inside_per]
    assert allowed_transitions[begin_per, inside_per]
    assert not allowed_transitions[inside_org, inside_per]

    emissions = torch.full((2, len(id2label)), -5.0)
    emissions[0, inside_per] = 20.0
    emissions[0, begin_per] = 10.0
    emissions[1, inside_per] = 10.0
    decoded = k_best_viterbi_decode(
        emissions,
        k=1,
        allowed_start=allowed_start,
        allowed_transitions=allowed_transitions,
    )
    assert decoded[0].tag_ids == (begin_per, inside_per)


def test_span_candidates_deduplicate_sequences_and_add_boundary_variants() -> None:
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    begin_per = DEFAULT_LABEL2ID["B-PER"]
    inside_per = DEFAULT_LABEL2ID["I-PER"]
    outside = DEFAULT_LABEL2ID["O"]
    sequences = [
        SequenceCandidate((outside, begin_per, inside_per, outside), 4.0),
        SequenceCandidate((outside, begin_per, inside_per, begin_per), 3.0),
    ]

    candidates = build_span_candidates(
        sequences,
        word_indices=[0, 1, 2, 3],
        id2label=id2label,
        num_words=4,
        max_candidates=8,
        boundary_shift=1,
        boundary_penalty=0.25,
    )
    by_span = {candidate.boundary: candidate for candidate in candidates}

    assert (1, 3) in by_span
    assert by_span[(1, 3)].source == "viterbi"
    assert by_span[(1, 3)].sequence_ranks == (0, 1)
    assert (0, 3) in by_span or (1, 4) in by_span
    assert any(candidate.source == "perturbation" for candidate in candidates)
    assert len(candidates) <= 8


def test_required_stage1_spans_survive_candidate_truncation() -> None:
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    outside = DEFAULT_LABEL2ID["O"]
    begin_per = DEFAULT_LABEL2ID["B-PER"]
    sequences = [
        SequenceCandidate((begin_per, outside, begin_per, outside), 5.0),
        SequenceCandidate((outside, begin_per, outside, begin_per), 4.0),
    ]
    candidates = build_span_candidates(
        sequences,
        word_indices=[0, 1, 2, 3],
        id2label=id2label,
        num_words=4,
        max_candidates=2,
        boundary_shift=1,
        required_spans=[(1, 3)],
    )

    by_span = {candidate.boundary: candidate for candidate in candidates}
    assert (1, 3) in by_span
    assert by_span[(1, 3)].preserve_stage1


def test_extract_word_spans_repairs_invalid_inside_transition() -> None:
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    spans = extract_word_spans(
        [DEFAULT_LABEL2ID["I-ORG"], DEFAULT_LABEL2ID["I-ORG"], DEFAULT_LABEL2ID["O"]],
        [2, 3, 4],
        id2label,
    )
    assert spans == [(2, 4, "ORG")]


def test_sequence_hamming_diversity() -> None:
    candidates = [
        SequenceCandidate((0, 1, 2), 2.0),
        SequenceCandidate((0, 1, 0), 1.0),
        SequenceCandidate((0, 0, 0), 0.0),
    ]
    assert sequence_hamming_diversity(candidates) > 0.0
