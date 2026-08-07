from __future__ import annotations

import copy
import json
from pathlib import Path

from gmner.data.j0_candidate_lattice import (
    baseline_result,
    build_lattice_record,
    contains_gold_or_supervision,
    evaluate_oracle_stage,
    oracle_action_breakdown,
)
from scripts.audit_j0_candidate_lattice_oracle import aggregate_result


ROOT = Path(__file__).resolve().parents[1]


TYPE_LOGITS = [0.0, 1.0, 2.0, 3.0]


def span(start: int, end: int) -> dict:
    return {"start": start, "end": end, "space": "word_half_open"}


def candidate(
    candidate_id: str,
    start: int,
    end: int,
    type_id: int,
    source: str,
    score: float,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "span": span(start, end),
        "type_id": type_id,
        "source": source,
        "region_candidate_id": "region:" + "a" * 64,
        "scores": {
            "span_base_score": score,
            "type_logits": TYPE_LOGITS,
            "region_score": 0.0,
        },
    }


def source_row() -> dict:
    candidates = [
        candidate("candidate:" + "1" * 64, 0, 1, 1, "stage1", 2.0),
        candidate("candidate:" + "2" * 64, 0, 2, 0, "kbest", 1.5),
        candidate("candidate:" + "3" * 64, 3, 4, 2, "kbest", 1.0),
    ]
    prediction = {
        "prediction_id": "prediction:" + "4" * 64,
        "span": span(0, 1),
        "type_id": 1,
        "type_logits": TYPE_LOGITS,
        "region_candidate_id": "region:" + "a" * 64,
        "observable_features": {
            "span_base_score": 2.0,
            "type_order": ["LOC", "PER", "ORG", "OTHER"],
        },
    }
    return {
        "kind": "final_chain_oof_record",
        "format_version": 1,
        "record_id": "7",
        "fold_id": 0,
        "heldout": True,
        "test_accessed": False,
        "formal_predictions": [prediction],
        "r16_candidates": {"span_candidates": copy.deepcopy(candidates)},
        "r36_candidates": {"span_candidates": copy.deepcopy(candidates)},
        "replacement_actions": [],
    }


def test_build_is_deterministic_gold_free_and_has_keep_none() -> None:
    row = source_row()
    first = build_lattice_record(row)
    second = build_lattice_record(copy.deepcopy(row))
    assert first == second
    assert not contains_gold_or_supervision(first)
    assert [group["group_kind"] for group in first["groups"]] == [
        "replacement",
        "addition",
    ]
    assert first["groups"][0]["control"]["operation"] == "KEEP"
    assert first["groups"][1]["control"]["operation"] == "NONE"


def test_lattice_contains_type_and_joint_boundary_hypotheses() -> None:
    lattice = build_lattice_record(source_row())
    replacement = lattice["groups"][0]
    typed = {
        (
            item["span"]["start"],
            item["span"]["end"],
            item["type_id"],
        )
        for item in replacement["alternatives"]
    }
    assert (0, 1, 0) in typed
    assert (0, 2, 0) in typed
    assert (0, 2, 3) in typed
    assert (0, 1, 1) not in typed
    for item in replacement["alternatives"]:
        assert item["typed_score"] == round(item["typed_score"], 12)
        assert item["type_log_probability"] == round(
            item["type_log_probability"], 12
        )


def test_constrained_oracle_can_replace_and_add_without_overlap() -> None:
    lattice = build_lattice_record(source_row())
    gold = [(0, 2, 0), (3, 4, 2)]
    baseline = baseline_result(lattice, gold)
    assert baseline == {
        "correct": 0,
        "span_correct": 0,
        "predicted": 1,
        "gold": 2,
    }
    result = evaluate_oracle_stage(
        lattice,
        gold,
        top_k=None,
        enforce_nonoverlap=True,
        max_record_alternatives=None,
        max_additions=1,
    )
    assert result["correct"] == 2
    assert result["predicted"] == 2
    assert result["additions"] == 1
    breakdown = oracle_action_breakdown(
        lattice, gold, result["selected_hypothesis_ids"]
    )
    assert breakdown["replacement_boundary_and_type"] == 1
    assert breakdown["replacement_corrected"] == 1
    assert breakdown["add_correct"] == 1
    assert breakdown["net_correct_contribution"] == 2


def test_record_constraint_blocks_overlapping_addition() -> None:
    row = source_row()
    row["r16_candidates"]["span_candidates"][2]["span"] = span(0, 3)
    row["r36_candidates"]["span_candidates"][2]["span"] = span(0, 3)
    lattice = build_lattice_record(row)
    # The third proposal overlaps the formal span and is therefore a replacement,
    # not an ADD proposal.
    assert all(group["group_kind"] == "replacement" for group in lattice["groups"])


def test_source_semantic_mismatch_is_audited_without_union() -> None:
    baseline = build_lattice_record(source_row())
    row = source_row()
    row["r16_candidates"]["span_candidates"][0]["type_id"] = 3
    lattice = build_lattice_record(row)
    assert not lattice["candidate_source_audit"]["r16_r36_semantic_match"]
    assert lattice["candidate_source_audit"]["r16_only_candidates"] == 1
    assert lattice["candidate_source_audit"]["r36_only_candidates"] == 1
    # The R16-only type mutation must not enter the formal R36 lattice.
    assert lattice["groups"] == baseline["groups"]


def test_one_addition_group_cannot_emit_two_entities() -> None:
    lattice = {
        "groups": [
            {
                "group_kind": "addition",
                "control": {
                    "hypothesis_id": "hypothesis:none",
                    "operation": "NONE",
                    "span": None,
                    "type_id": None,
                },
                "alternatives": [
                    {
                        "hypothesis_id": "hypothesis:left",
                        "operation": "ADD",
                        "span": span(0, 1),
                        "type_id": 0,
                        "typed_score": 2.0,
                        "source_priority": 1,
                    },
                    {
                        "hypothesis_id": "hypothesis:right",
                        "operation": "ADD",
                        "span": span(2, 3),
                        "type_id": 1,
                        "typed_score": 1.0,
                        "source_priority": 1,
                    },
                ],
            }
        ]
    }
    result = evaluate_oracle_stage(
        lattice,
        [(0, 1, 0), (2, 3, 1)],
        top_k=None,
        enforce_nonoverlap=True,
        max_additions=None,
    )
    assert result["correct"] == 1
    assert result["additions"] == 1


def test_baseline_aggregation_defaults_additions_to_zero() -> None:
    aggregate = aggregate_result(
        [{"correct": 2, "span_correct": 3, "predicted": 4, "gold": 5}]
    )
    assert aggregate["additions"] == 0
    assert aggregate["span_correct"] == 3


def test_preregistration_authorizes_only_j0_a() -> None:
    payload = json.loads(
        (
            ROOT
            / "docs"
            / "experiments"
            / "j0_a_candidate_lattice_oracle_preregistration.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "AUTHORIZED_J0_A_ONLY"
    assert payload["continuation_gate"]["minimum_oof_net_correct_gain"] == 308
    assert payload["budgets"]["final_per_group_top_k"] == 4
    assert payload["authorization"]["j0_a_gold_free_build"]
    assert payload["authorization"]["j0_a_postseal_oracle"]
    for key in (
        "j0_b_latent_rematerialization",
        "j1_training",
        "j2_visual",
        "j3_structured_decoder",
        "dev_access",
        "test_access",
    ):
        assert not payload["authorization"][key]
