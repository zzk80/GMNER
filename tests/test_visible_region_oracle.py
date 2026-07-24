from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from scripts.analyze_visible_region_oracle import (
    _validate_cache_pair,
    classify_visible_case,
    parse_int_list,
    proposal_covered,
)


def _classify(**overrides: bool) -> str:
    values = {
        "span_present": True,
        "type_correct": True,
        "final_present": True,
        "final_correct": False,
        "formal_covered": True,
        "final_is_null": False,
        "base_correct": False,
    }
    values.update(overrides)
    return classify_visible_case(**values)


def test_proposal_coverage_uses_raw_budget_and_excludes_null() -> None:
    positives = [7, 21, 36]
    assert proposal_covered(positives, budget=16, null_index=36)
    assert proposal_covered(positives, budget=36, null_index=36)
    assert not proposal_covered([21, 36], budget=16, null_index=36)
    assert proposal_covered([21, 36], budget=36, null_index=36)
    assert not proposal_covered([36], budget=36, null_index=36)


def test_visible_attribution_is_mutually_exclusive_and_prioritized() -> None:
    assert _classify(span_present=False) == "unactionable_span_missing"
    assert _classify(type_correct=False) == "unactionable_type_wrong"
    assert _classify(final_correct=True) == "final_correct"
    assert (
        _classify(formal_covered=False, final_is_null=True)
        == "A_region_missing_formal_budget"
    )
    assert _classify(final_present=False) == "unactionable_final_span_removed"
    assert _classify(final_is_null=True) == "D_visibility_false_null"
    assert (
        _classify(base_correct=True) == "C_verifier_real_region_damage"
    )
    assert _classify() == "B_base_misrank_remaining"


def test_parse_int_list_rejects_non_positive_budgets() -> None:
    assert parse_int_list("36,16,16") == [16, 36]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_int_list("0,16")


def test_cache_pair_may_differ_only_in_region_budget() -> None:
    formal = SimpleNamespace(
        metadata={
            "stage1_checkpoint_sha256": "stage1",
            "candidate_config": {"max_regions": 16, "k_best": 6},
        }
    )
    expanded = SimpleNamespace(
        metadata={
            "stage1_checkpoint_sha256": "stage1",
            "candidate_config": {"max_regions": 36, "k_best": 6},
        }
    )
    assert _validate_cache_pair(
        formal, expanded, checkpoint={"stage1_checkpoint_sha256": "stage1"}
    ) == (16, 36)

    expanded.metadata["candidate_config"]["k_best"] = 8
    with pytest.raises(ValueError, match="differ only in max_regions"):
        _validate_cache_pair(
            formal, expanded, checkpoint={"stage1_checkpoint_sha256": "stage1"}
        )
