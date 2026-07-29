from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from gmner.data.p4_r0_replay_contract import (
    P4_R0_A_BLOCKED,
    P4_R0_A_REPORT_KIND,
    P4_R0_EXTERNAL_INVENTORY_KIND,
    P4_R0_FORMAT_VERSION,
    P4_R0_PREREGISTRATION_KIND,
    SUPERVISED_STAGES,
    build_r0_a_report,
    discover_named_artifacts,
    external_available_artifacts,
    git_source_tree_sha256,
    match_expectations_by_sha256,
    validate_external_inventory,
    validate_r0_a_report,
    validate_r0_preregistration,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _preregistration() -> dict:
    return {
        "kind": P4_R0_PREREGISTRATION_KIND,
        "format_version": P4_R0_FORMAT_VERSION,
        "development_folds": list(range(8)),
        "required_supervised_stages": list(SUPERVISED_STAGES),
        "authorization": {
            "r0_a_read_only_audit": True,
            "r0_a_checkpoint_replay_execution": False,
            "r0_b_full_oof_retraining": False,
            "checkpoint_or_cache_deserialization": False,
            "model_execution": False,
            "candidate_generation": False,
            "oracle": False,
            "p4_1": False,
            "downstream_rebuild": False,
            "calibration_folds_8_9": False,
            "dev_access": False,
            "test_access": False,
        },
    }


def _external_inventory() -> dict:
    return {
        "kind": P4_R0_EXTERNAL_INVENTORY_KIND,
        "format_version": P4_R0_FORMAT_VERSION,
        "folds_checked": list(range(8)),
        "access_contract": {
            "payloads_deserialized": 0,
            "training_records_parsed": 0,
            "calibration_folds_opened": False,
            "dev_accessed": False,
            "test_accessed": False,
            "oracle_labels_computed": False,
            "model_executed": False,
        },
        "locations": [
            {
                "name": "archive",
                "available_artifacts": [
                    {"path": "/archive/file", "bytes": 10, "sha256": SHA_A}
                ],
            }
        ],
    }


def _fold_report(fold_id: int) -> dict:
    return {
        "fold_id": fold_id,
        "status": "PROVENANCE_VALID",
        "config_artifacts": [{} for _ in SUPERVISED_STAGES],
        "config_seeds": {stage: 42 for stage in SUPERVISED_STAGES},
        "recorded_input_fingerprints": {
            "vinvl_feature_tree_sha256": None,
            "text_tokenizer_tree_sha256": None,
            "grounding_prior_bundle_sha256": None,
        },
    }


def test_preregistration_keeps_replay_retraining_and_locked_scopes_disabled() -> None:
    payload = _preregistration()
    validate_r0_preregistration(payload)
    payload["authorization"]["r0_a_checkpoint_replay_execution"] = True
    with pytest.raises(PermissionError, match="locked authorizations"):
        validate_r0_preregistration(payload)


def test_external_inventory_rejects_dev_or_payload_access() -> None:
    payload = _external_inventory()
    validate_external_inventory(payload)
    assert external_available_artifacts(payload)[0]["sha256"] == SHA_A
    payload["access_contract"]["dev_accessed"] = True
    with pytest.raises(PermissionError, match="dev_accessed"):
        validate_external_inventory(payload)


def test_artifact_matching_requires_exact_sha256() -> None:
    matches = match_expectations_by_sha256(
        [
            {"expected_sha256": SHA_A, "stage": "stage1"},
            {"expected_sha256": SHA_B, "stage": "fine"},
        ],
        [{"path": "/archive/file", "bytes": 10, "sha256": SHA_A}],
    )
    assert matches[0]["status"] == "EXACT_SHA256_AVAILABLE"
    assert matches[1]["status"] == "MISSING"


def test_discovery_prunes_locked_fold_and_split_directories(tmp_path: Path) -> None:
    allowed = tmp_path / "fold0" / "stage1"
    allowed.mkdir(parents=True)
    (allowed / "best_model.pt").write_bytes(b"allowed")
    for locked in ("fold8", "fold9", "dev", "test"):
        directory = tmp_path / locked
        directory.mkdir()
        (directory / "best_model.pt").write_bytes(b"locked")

    result = discover_named_artifacts(
        [tmp_path],
        basenames=["best_model.pt"],
    )
    assert result["paths"] == [(allowed / "best_model.pt").resolve()]
    assert len(result["locked_directories_skipped"]) == 4


def test_blocked_report_never_authorizes_replay_or_r0_b() -> None:
    preregistration = _preregistration()
    fold_reports = [_fold_report(fold_id) for fold_id in range(8)]
    checkpoint_matches = [
        {"status": "MISSING", "fold_id": fold_id}
        for fold_id in range(8)
    ]
    source_matches = [
        {"status": "EXACT_SHA256_AVAILABLE", "fold_id": fold_id}
        for fold_id in range(8)
    ]
    report = build_r0_a_report(
        preregistration=preregistration,
        fold_reports=fold_reports,
        checkpoint_matches=checkpoint_matches,
        source_matches=source_matches,
        source_tree={
            "expected_sha256": SHA_A,
            "exact_source_tree_available": False,
            "matching_commits": [],
        },
        external_inventory=_external_inventory(),
        implementation={"git_head": "commit"},
    )
    validate_r0_a_report(report)
    assert report["kind"] == P4_R0_A_REPORT_KIND
    assert report["status"] == P4_R0_A_BLOCKED
    assert report["checkpoint_replay_execution_authorized"] is False
    assert report["r0_b_authorized"] is False
    assert report["next_state"] == "STOP_WITHOUT_AUTHORIZING_R0_B"
    assert "all_supervised_checkpoints_exact" in report["gate_blockers"]
    assert "source_tree_exactly_recoverable" in report["gate_blockers"]
    assert "required_input_fingerprints_recorded" in report["gate_blockers"]


def test_git_source_tree_hash_matches_worktree_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()
    assert len(git_source_tree_sha256(project_root, head)) == 64


def test_report_digest_detects_tampering() -> None:
    report = build_r0_a_report(
        preregistration=_preregistration(),
        fold_reports=[_fold_report(fold_id) for fold_id in range(8)],
        checkpoint_matches=[{"status": "MISSING"}],
        source_matches=[{"status": "MISSING"}],
        source_tree={"exact_source_tree_available": False},
        external_inventory=_external_inventory(),
        implementation={},
    )
    altered = copy.deepcopy(report)
    altered["r0_b_authorized"] = True
    with pytest.raises(ValueError, match="digest"):
        validate_r0_a_report(altered)
