from __future__ import annotations

import torch

from scripts.analyze_record_set_assignment_oracle import (
    maximum_bipartite_matching,
    parse_top_k,
    ranked_candidate_indices,
    record_oracle_at_k,
)


def test_parse_top_k_sorts_and_deduplicates():
    assert parse_top_k("8,2,4,2") == (2, 4, 8)


def test_ranked_candidate_indices_respects_mask():
    logits = torch.tensor([0.5, 3.0, 2.0, 9.0])
    mask = torch.tensor([True, False, True, False])
    assert ranked_candidate_indices(logits, mask) == [2, 0]


def test_maximum_bipartite_matching_uses_augmenting_path():
    # Entity 0 can move to region 2 so entity 1 can claim region 1.
    assert maximum_bipartite_matching([{1, 2}, {1}, {3}]) == 3


def test_record_oracle_keeps_correct_and_fixes_null_and_real():
    items = [
        {
            "gold_visible": True,
            "current_correct": False,
            "current_region": 0,
            "positive_regions": {1},
            "ranked_candidates": [1, 2],
        },
        {
            "gold_visible": True,
            "current_correct": True,
            "current_region": 2,
            "positive_regions": {2},
            "ranked_candidates": [2, 1],
        },
        {
            "gold_visible": False,
            "current_correct": False,
            "current_region": 3,
            "positive_regions": set(),
            "ranked_candidates": [3, 4],
        },
    ]
    result = record_oracle_at_k(items, 1)
    assert result["current_correct"] == 1
    assert result["to_real_fixes"] == 1
    assert result["to_null_fixes"] == 1
    assert result["independent_oracle_correct"] == 3
    assert result["strict_capacity_oracle_correct"] == 3


def test_strict_capacity_reports_legitimate_region_sharing_gap():
    items = [
        {
            "gold_visible": True,
            "current_correct": True,
            "current_region": 1,
            "positive_regions": {1},
            "ranked_candidates": [1],
        },
        {
            "gold_visible": True,
            "current_correct": True,
            "current_region": 1,
            "positive_regions": {1},
            "ranked_candidates": [1],
        },
    ]
    result = record_oracle_at_k(items, 1)
    assert result["independent_oracle_correct"] == 2
    assert result["strict_capacity_oracle_correct"] == 1
    assert result["sharing_gap"] == 1
    assert result["current_collision_regions"] == 1
    assert result["current_collision_entities"] == 2
    assert result["shared_positive_pairs"] == 1
