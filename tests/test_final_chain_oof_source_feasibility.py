import json
from pathlib import Path

import pytest

from gmner.constants import ENTITY_TYPE2ID
from scripts.audit_final_chain_oof_source_feasibility import (
    build_inventory,
    derive_status,
)


ROOT = Path(__file__).resolve().parents[1]


def test_status_derivation_order() -> None:
    required = ["a", "b"]
    complete = {"a": True, "b": True}
    assert derive_status({"artifact_state": "NOT_FOUND"}, required)[0] == "MISSING"
    assert derive_status(
        {"artifact_state": "PRESENT", "provenance_valid": False}, required
    )[0] == "PROVENANCE_INVALID"
    assert derive_status(
        {
            "artifact_state": "PRESENT",
            "provenance_valid": True,
            "semantic_valid": False,
        },
        required,
    )[0] == "SEMANTICALLY_INVALID"
    assert derive_status(
        {
            "artifact_state": "PRESENT",
            "provenance_valid": True,
            "semantic_valid": True,
            "capabilities": {"a": True},
        },
        required,
    ) == ("INCOMPLETE", ["b"])
    assert derive_status(
        {
            "artifact_state": "PRESENT",
            "provenance_valid": True,
            "semantic_valid": True,
            "capabilities": complete,
        },
        required,
    ) == ("VALID", [])


def test_frozen_inventory_has_no_valid_source() -> None:
    registry = json.loads(
        (ROOT / "docs/experiments/final_chain_oof_source_registry.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (ROOT / "docs/experiments/final_chain_oof_minimum_row_schema.json").read_text(
            encoding="utf-8"
        )
    )
    report = build_inventory(registry, schema)
    assert report["status"] == "BLOCKED_NO_VALID_SOURCE"
    assert report["valid_sources"] == []
    assert report["next_authorized_step"] is None
    assert (
        report["next_required_decision"]
        == "explicitly_authorize_new_single_fold_dry_run"
    )
    assert report["status_counts"] == {
        "INCOMPLETE": 2,
        "MISSING": 1,
        "PROVENANCE_INVALID": 0,
        "SEMANTICALLY_INVALID": 1,
        "VALID": 0,
    }
    assert not any(report["access"].values())


def test_declared_status_mismatch_is_rejected() -> None:
    registry = {
        "access": {},
        "sources": [
            {
                "source_id": "bad",
                "description": "bad",
                "artifact_state": "PRESENT",
                "provenance_valid": True,
                "semantic_valid": True,
                "declared_status": "VALID",
                "capabilities": {},
            }
        ],
    }
    schema = {"x_final_chain_oof_required_capabilities": ["required"]}
    with pytest.raises(ValueError, match="derives as INCOMPLETE"):
        build_inventory(registry, schema)


def test_fold0_dry_run_is_authorized_while_later_stages_remain_locked() -> None:
    payload = json.loads(
        (
            ROOT
            / "docs/experiments/final_chain_oof_fold0_dry_run_preregistration.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "AUTHORIZED_NOT_STARTED"
    assert payload["authorization"] == {
        "source_inventory_audit": True,
        "fold0_execution": True,
        "folds_1_9_execution": False,
        "b1_a1_population_training": False,
        "dev_execution": False,
        "test_execution": False,
    }
    assert not any(payload["access"].values())
    assert payload["excluded_modules"] == [
        "siglip2",
        "reliability",
        "null_release",
        "clip",
        "fmnerg_subtype",
    ]


def test_fold0_type_region_identity_and_numeric_contracts_are_frozen() -> None:
    schema = json.loads(
        (ROOT / "docs/experiments/final_chain_oof_minimum_row_schema.json").read_text(
            encoding="utf-8"
        )
    )
    preregistration = json.loads(
        (
            ROOT
            / "docs/experiments/final_chain_oof_fold0_dry_run_preregistration.json"
        ).read_text(encoding="utf-8")
    )
    expected_types = {"LOC": 0, "PER": 1, "ORG": 2, "OTHER": 3}
    assert {name: ENTITY_TYPE2ID[name] for name in expected_types} == expected_types
    assert schema["x_type_contract"]["type_id_map"] == expected_types
    assert schema["x_type_contract"]["type_logits_order"] == [
        "LOC",
        "PER",
        "ORG",
        "OTHER",
    ]
    assert (
        schema["x_region_contract"]["formal_region_index_namespace"]
        == "expanded_r36_local_index"
    )
    assert schema["x_region_contract"]["local_index_excluded_from_stable_identity"]
    assert schema["x_identity_contract"]["float_inputs_forbidden"]
    assert schema["x_identity_contract"]["runtime_row_indices_forbidden"]
    assert schema["x_numeric_replay_contract"]["continuous_atol"] == 3e-5
    assert schema["x_numeric_replay_contract"]["continuous_rtol"] == 1e-6
    assert schema["x_numeric_replay_contract"]["nan_or_inf"] == "hard_stop"
    semantics = preregistration["semantic_contract"]
    assert semantics["coarse_type_ids"] == expected_types
    assert semantics["formal_region_index_namespace"] == "expanded_r36_local_index"
    assert semantics["digest_includes_raw_float_bytes"] is False
