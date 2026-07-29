from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from gmner.data.p4_actionability_contract import P4_PROVENANCE_REPORT_KIND
from gmner.data.p4_formal_r16_recovery import (
    P4_FORMAL_RECOVERY_BLOCKED,
    build_blocked_recovery_report,
    build_formal_span_sidecar,
    discover_formal_cache_candidates,
    formal_cache_expectations,
    hash_recovery_candidates,
    load_and_validate_exact_formal_cache,
    match_exact_formal_caches,
    validate_formal_span_sidecar,
    validate_recovery_report,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _phase_a_report(hashes: list[str] | None = None) -> dict:
    hashes = hashes or [f"{fold_id:064x}" for fold_id in range(1, 9)]
    return {
        "kind": P4_PROVENANCE_REPORT_KIND,
        "access_contract": {
            "folds_read": list(range(8)),
            "calibration_folds_opened": False,
            "dev_accessed": False,
            "test_accessed": False,
            "oracle_labels_computed": False,
        },
        "provenance": [
            {
                "fold_id": fold_id,
                "status": "PASSED",
                "records": 1,
                "heldout_record_ids_sha256": stable_id_digest([f"record-{fold_id}"]),
                "fold_proof_sha256": SHA_A,
                "heldout_feature_sha256": SHA_B,
                "artifact_sha256": {"formal_cache": hashes[fold_id]},
                "test_accessed": False,
            }
            for fold_id in range(8)
        ],
    }


def _formal_payload(fold_id: int = 0) -> dict:
    record_id = f"record-{fold_id}"
    return {
        "metadata": {
            "format_version": 2,
            "oof_heldout": True,
            "oof_fold_id": fold_id,
        },
        "records": [
            {
                "span_candidates": torch.tensor([[0, 1], [2, 3]]),
                "span_mask": torch.tensor([True, True]),
                "span_source_ids": torch.tensor([0, 2]),
                "fixed_type_ids": torch.tensor([1, 2]),
                "base_region_indices": torch.tensor([0, 2]),
                "metadata": {
                    "record_id": record_id,
                    "tokens": ["alpha", "beta", "gamma"],
                    "stage1_predictions": [
                        {"span": [0, 1], "type_id": 1, "region_index": 0}
                    ],
                },
            }
        ],
    }


def _full_chain_payload(fold_id: int = 0) -> dict:
    record_id = f"record-{fold_id}"
    candidate_mask = torch.tensor([[[True, True, False], [True, False, False]]])
    fine = {
        "candidate_mask": candidate_mask,
        "final_region_logits": torch.tensor([[[1.0, 2.0, -1.0], [2.0, -1.0, -1.0]]]),
        "fine_top4_indices": torch.tensor([[[0, 1, 0, 0], [0, 0, 0, 0]]]),
        "fine_top4_valid_mask": torch.tensor(
            [[[True, True, False, False], [True, False, False, False]]]
        ),
        "span_grounding_state": torch.zeros(1, 2, 2),
        "region_grounding_state": torch.zeros(1, 3, 2),
        "type_grounding_state": torch.zeros(1, 2, 2),
        "candidate_source_ids": torch.zeros(1, 2, 3, dtype=torch.long),
        "base_log_prior": torch.zeros(1, 2, 3),
        "coarse_log_prior": torch.zeros(1, 2, 3),
        "base_rank": torch.zeros(1, 2, 3, dtype=torch.long),
        "coarse_rank": torch.zeros(1, 2, 3, dtype=torch.long),
        "detector_rank": torch.zeros(1, 2, 3, dtype=torch.long),
        "fixed_type_region_compatibility": torch.zeros(1, 2, 3),
        "promoted_candidate_mask": torch.zeros(1, 2, 3, dtype=torch.bool),
        "fixed_type_ids": torch.tensor([[1, 2]]),
    }
    return {
        "metadata": {
            "format_version": 1,
            "kind": "null_release_full_chain_oof",
            "full_chain_oof": True,
            "fold_id": fold_id,
            "num_folds": 10,
            "records": 1,
            "record_ids_sha256": stable_id_digest([record_id]),
            "includes_reliability": True,
        },
        "batches": [
            {
                "fold_id": fold_id,
                "record_ids": [record_id],
                "fine_outputs": fine,
                "hierarchy_outputs": {"fixed_type_ids": torch.tensor([[1, 2]])},
                "evidence_outputs": {
                    "evidence_scalar_features": torch.zeros(1, 2, 3)
                },
                "expanded": {
                    "span_mask": torch.tensor([[True, True]]),
                    "span_source_ids": torch.tensor([[0, 2]]),
                    "gold_span_mask": torch.tensor([[False, False]]),
                    "visibility_targets": torch.tensor([[0.0, 0.0]]),
                    "type_candidates": torch.tensor([[[1], [2]]]),
                    "gold_type_mask": torch.tensor([[[False], [False]]]),
                    "gold_region_positive_mask": torch.zeros(
                        1, 2, 3, dtype=torch.bool
                    ),
                    "region_mask": torch.tensor([[True, True, True]]),
                    "region_is_null": torch.tensor([[False, False, True]]),
                    "region_detector_scores": torch.zeros(1, 3),
                },
                "reliability_outputs": {
                    "reliability_probability": torch.zeros(1, 2)
                },
                "current_visible": torch.tensor([[True, False]]),
                "base_is_null": torch.tensor([[False, True]]),
                "deployment_span_mask": torch.tensor([[True, False]]),
            }
        ],
    }


def test_recovery_expectations_are_exactly_folds_zero_to_seven() -> None:
    expectations = formal_cache_expectations(_phase_a_report())

    assert [item["fold_id"] for item in expectations] == list(range(8))
    assert expectations[0]["expected_sha256"] == f"{1:064x}"

    invalid = _phase_a_report()
    invalid["access_contract"]["dev_accessed"] = True
    with pytest.raises(PermissionError, match="Dev"):
        formal_cache_expectations(invalid)


def test_discovery_skips_calibration_dev_and_test(tmp_path: Path) -> None:
    allowed = tmp_path / "backup" / "fold0"
    locked_fold = tmp_path / "backup" / "fold8"
    locked_dev = tmp_path / "backup" / "dev"
    for directory in (allowed, locked_fold, locked_dev):
        directory.mkdir(parents=True)
        (directory / "heldout_r16.pt").write_bytes(b"cache")

    result = discover_formal_cache_candidates(search_roots=[tmp_path / "backup"])

    assert result["candidate_paths"] == [
        (allowed / "heldout_r16.pt").resolve().as_posix()
    ]
    assert len(result["locked_paths_skipped"]) == 2
    assert result["calibration_folds_opened"] is False
    assert result["dev_accessed"] is False
    assert result["test_accessed"] is False


def test_incomplete_hash_set_never_loads_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "heldout_r16.pt"
    candidate.write_bytes(b"not a torch payload")
    descriptors = hash_recovery_candidates([candidate])
    hashes = [descriptors[0]["sha256"]] + [
        f"{fold_id:064x}" for fold_id in range(2, 9)
    ]
    expectations = formal_cache_expectations(_phase_a_report(hashes))
    matches = match_exact_formal_caches(expectations, descriptors)

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("torch.load must not run"),
    )
    report = build_blocked_recovery_report(
        expectations=expectations,
        discovery={
            "patterns": ["heldout_r16.pt"],
            "search_roots": [tmp_path.as_posix()],
            "missing_search_roots": [],
            "candidate_paths": [candidate.as_posix()],
            "locked_paths_skipped": [],
            "calibration_folds_opened": False,
            "dev_accessed": False,
            "test_accessed": False,
        },
        candidate_descriptors=descriptors,
        matches=matches,
        implementation={"git_head": "head"},
        phase_a_report_path="phase_a.json",
        phase_a_report_sha256=SHA_A,
    )

    assert report["status"] == P4_FORMAL_RECOVERY_BLOCKED
    assert report["missing_exact_artifact_folds"] == list(range(1, 8))
    assert report["payload_deserialization"]["attempted"] is False
    assert report["formal_span_sidecars"]["generated"] is False
    assert report["source_manifest"]["sealed"] is False
    validate_recovery_report(report)


def test_exact_cache_can_build_and_reload_preserved_formal_sidecar(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "heldout_r16.pt"
    torch.save(_formal_payload(), cache_path)
    digest = sha256_file(cache_path)
    expectation = {
        "fold_id": 0,
        "expected_sha256": digest,
        "expected_records": 1,
        "expected_record_ids_sha256": stable_id_digest(["record-0"]),
    }
    formal = load_and_validate_exact_formal_cache(
        cache_path,
        expectation=expectation,
        expected_record_ids=["record-0"],
    )
    sidecar = build_formal_span_sidecar(
        formal_cache=formal,
        full_chain_payload=_full_chain_payload(),
        expectation=expectation,
        full_chain_feature_sha256=SHA_A,
        generator_git_head="head",
        generator_path="scripts/audit.py",
        generator_sha256=SHA_B,
    )
    reloaded = copy.deepcopy(json.loads(json.dumps(sidecar)))
    validation = validate_formal_span_sidecar(reloaded)

    assert reloaded["source_formal_cache_sha256"] == digest
    assert reloaded["formal_prediction_count"] == 1
    assert reloaded["records_payload"][0]["formal_predictions"] == [
        {
            "span_start": 0,
            "span_end": 1,
            "type_id": 1,
            "region_index": 1,
            "region_is_null": False,
        }
    ]
    assert validation["formal_predictions_preserved"] is True
