from __future__ import annotations

from collections import Counter

import pytest
import torch

from scripts.analyze_hierarchical_action_oracle import (
    ACTION_FAMILIES,
    action_label,
    summarize_action_cases,
    summarize_action_labels,
    topk_real_indices,
)


def _hits(top_ks: list[int], **minimum_rank: int) -> dict[str, dict[int, bool]]:
    return {
        family: {
            top_k: family in minimum_rank and top_k >= minimum_rank[family]
            for top_k in top_ks
        }
        for family in ACTION_FAMILIES
    }


def test_topk_real_indices_excludes_masked_regions() -> None:
    result = topk_real_indices(
        torch.tensor([0.1, 0.9, 0.8, 100.0]),
        torch.tensor([True, True, True, False]),
        [1, 2, 8],
    )
    assert result == {1: {1}, 2: {1, 2}, 8: {0, 1, 2}}


def test_action_label_is_direct_triple_delta() -> None:
    positives = {2, 3}
    assert action_label(
        keep_region=0, action_region=2, positive_regions=positives
    ) == "fix"
    assert action_label(
        keep_region=2, action_region=0, positive_regions=positives
    ) == "damage"
    assert action_label(
        keep_region=2, action_region=3, positive_regions=positives
    ) == "neutral"
    assert action_label(
        keep_region=0, action_region=1, positive_regions=positives
    ) == "neutral"


def test_action_label_summary_does_not_hide_neutral_actions() -> None:
    summary = summarize_action_labels(
        Counter({"fix": 1, "damage": 1, "neutral": 3})
    )
    assert summary["actions"] == 5.0
    assert summary["net_if_all_executed"] == 0.0
    assert summary["fix_rate_over_all_actions"] == pytest.approx(0.2)
    assert summary["fix_precision_excluding_neutral"] == pytest.approx(0.5)


def test_candidate_oracle_unions_action_families_without_double_counting() -> None:
    top_ks = [1, 2, 4]
    cases = [
        {"to_null": True, "family_hits": _hits(top_ks)},
        {
            "to_null": False,
            "family_hits": _hits(top_ks, residual=1, fused=1, base=2),
        },
        {
            "to_null": False,
            "family_hits": _hits(top_ks, base=4),
        },
        {"to_null": False, "family_hits": _hits(top_ks)},
    ]
    summary = summarize_action_cases(
        cases,
        top_ks=top_ks,
        keep_correct=10,
        predicted=20,
        gold=20,
    )

    assert summary["to_null_oracle"]["fix_count"] == 1.0
    assert summary["family_oracles"]["residual"]["top1"][
        "candidate_oracle_fix_count"
    ] == 2.0
    assert summary["family_oracles"]["base"]["top4"][
        "candidate_oracle_fix_count"
    ] == 3.0
    assert summary["union_oracles"]["top4"][
        "candidate_oracle_fix_count"
    ] == 3.0
    assert summary["candidate_set_oracle"]["candidate_oracle_gmner"] == pytest.approx(
        0.65
    )
    assert summary["candidate_set_oracle"]["unique_fixable_span_count"] == 3.0
    assert summary["candidate_set_oracle"]["oracle_selected_action_count"] == 3.0
    assert summary["candidate_set_oracle"]["oracle_final_gmner"] == pytest.approx(
        0.65
    )
    assert summary["candidate_set_oracle"]["go_no_go"] == "stop"
    assert summary["recoverable_family_overlap_at_max_k"] == {
        "residual+fused+base": 1,
        "base": 1,
    }
