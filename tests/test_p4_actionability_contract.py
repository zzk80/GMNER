from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gmner.data.full_chain_oof_contract import (
    REQUIRED_PIPELINE_STAGES,
    SUPERVISED_PIPELINE_STAGES,
)
from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from gmner.data.p4_actionability_contract import (
    attach_manifest_sha256,
    audit_cross_cache_candidate_identity,
    build_gold_free_candidate_payload,
    build_source_manifest,
    enforce_p4_development_access,
    gold_free_selector_record,
    parse_p4_development_folds,
    source_seal_blockers,
    validate_archived_full_chain_fold,
    validate_manifest_sha256,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _fine_outputs(batch_size: int = 1, spans: int = 2, regions: int = 3) -> dict:
    candidate_mask = torch.ones(batch_size, spans, regions, dtype=torch.bool)
    top4 = torch.tensor([0, 1, 2, 0]).view(1, 1, 4).expand(batch_size, spans, 4)
    top4_valid = (
        torch.tensor([True, True, True, False])
        .view(1, 1, 4)
        .expand(batch_size, spans, 4)
    )
    return {
        "candidate_mask": candidate_mask,
        "final_region_logits": torch.zeros(batch_size, spans, regions),
        "fine_top4_indices": top4,
        "fine_top4_valid_mask": top4_valid,
        "span_grounding_state": torch.zeros(batch_size, spans, 2),
        "region_grounding_state": torch.zeros(batch_size, regions, 2),
        "type_grounding_state": torch.zeros(batch_size, spans, 2),
        "candidate_source_ids": torch.zeros(
            batch_size, spans, regions, dtype=torch.long
        ),
        "base_log_prior": torch.zeros(batch_size, spans, regions),
        "coarse_log_prior": torch.zeros(batch_size, spans, regions),
        "base_rank": torch.zeros(batch_size, spans, regions, dtype=torch.long),
        "coarse_rank": torch.zeros(batch_size, spans, regions, dtype=torch.long),
        "detector_rank": torch.zeros(batch_size, spans, regions, dtype=torch.long),
        "fixed_type_region_compatibility": torch.zeros(batch_size, spans, regions),
        "promoted_candidate_mask": torch.zeros(
            batch_size, spans, regions, dtype=torch.bool
        ),
        "fixed_type_ids": torch.tensor([[1, 2]], dtype=torch.long),
    }


def _full_chain_payload(
    *,
    fold_id: int = 0,
    proof_sha: str = SHA_A,
    artifact_sha: dict[str, str] | None = None,
) -> dict:
    artifact_sha = artifact_sha or {"formal_cache": SHA_B}
    train_ids = ["train"]
    heldout_ids = ["held"]
    expanded = {
        "span_mask": torch.tensor([[True, True]]),
        "span_source_ids": torch.tensor([[0, 2]]),
        "gold_span_mask": torch.tensor([[True, False]]),
        "visibility_targets": torch.tensor([[1.0, 0.0]]),
        "type_candidates": torch.tensor([[[1, 0], [2, 0]]]),
        "gold_type_mask": torch.tensor([[[True, False], [False, False]]]),
        "gold_region_positive_mask": torch.tensor(
            [[[True, False, False], [False, False, True]]]
        ),
        "region_mask": torch.ones(1, 3, dtype=torch.bool),
        "region_is_null": torch.tensor([[False, False, True]]),
        "region_detector_scores": torch.zeros(1, 3),
    }
    batch = {
        "fold_id": fold_id,
        "record_ids": heldout_ids,
        "fine_outputs": _fine_outputs(),
        "hierarchy_outputs": {"fixed_type_ids": torch.tensor([[1, 2]])},
        "evidence_outputs": {"evidence_scalar_features": torch.zeros(1, 2, 3)},
        "expanded": expanded,
        "reliability_outputs": {"reliability_probability": torch.zeros(1, 2)},
        "current_visible": torch.tensor([[True, False]]),
        "base_is_null": torch.tensor([[False, True]]),
        "deployment_span_mask": torch.tensor([[True, True]]),
    }
    return {
        "metadata": {
            "format_version": 1,
            "kind": "null_release_full_chain_oof",
            "full_chain_oof": True,
            "fold_id": fold_id,
            "num_folds": 10,
            "records": 1,
            "record_ids_sha256": stable_id_digest(heldout_ids),
            "training_record_ids": train_ids,
            "heldout_record_ids": heldout_ids,
            "excluded_heldout": True,
            "includes_reliability": True,
            "fold_proof_sha256": proof_sha,
            "artifact_sha256": artifact_sha,
        },
        "batches": [batch],
    }


def _selector_record(record_id: str = "held") -> dict:
    return {
        "span_candidates": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        "span_mask": torch.tensor([True, True]),
        "span_features": torch.zeros(2, 4, dtype=torch.float16),
        "span_base_scores": torch.tensor([2.0, 1.0]),
        "span_source_ids": torch.tensor([0, 2]),
        "span_lengths": torch.tensor([1, 1]),
        "type_candidates": torch.tensor([[1, 0], [2, 0]]),
        "type_base_scores": torch.tensor([[3.0, 0.0], [2.0, 0.0]]),
        "fixed_type_ids": torch.tensor([1, 2]),
        "base_region_indices": torch.tensor([0, 2]),
        "gold_span_mask": torch.tensor([True, False]),
        "gold_type_mask": torch.tensor([[True, False], [False, False]]),
        "formal_candidate_mask": torch.tensor([True, False]),
        "metadata": {
            "record_id": record_id,
            "text": "alpha beta gamma",
            "tokens": ["alpha", "beta", "gamma"],
            "candidate_sources": ["stage1", "kbest"],
            "stage1_predictions": [{"span": [0, 1], "type_id": 1, "region_index": 0}],
            "gold_entities": [{"span": [0, 1], "type_id": 1}],
            "null_region_index": 2,
        },
    }


def _selector_payload(
    *,
    checkpoint_sha: str = SHA_B,
    proof_sha: str = SHA_A,
) -> dict:
    record = _selector_record()
    return {
        "metadata": {
            "format_version": 1,
            "kind": "stage1_candidate_selector_oof",
            "scope": "oof_fold",
            "oof": True,
            "test_accessed": False,
            "fold_id": 0,
            "num_folds": 10,
            "records": 1,
            "record_ids": ["held"],
            "record_ids_sha256": stable_id_digest(["held"]),
            "formal_source_id": 0,
            "source2id": {
                "stage1": 0,
                "viterbi": 1,
                "kbest": 2,
                "perturbation": 3,
            },
            "candidate_config": {
                "inject_gold_types": False,
                "max_span_candidates": 12,
            },
            "candidate_config_sha256": SHA_A,
            "stage1_checkpoint_sha256": checkpoint_sha,
            "data_source_sha256": SHA_A,
            "source_candidate_cache_sha256": SHA_A,
            "stage1_config_sha256": SHA_A,
            "fold_manifest_sha256": SHA_A,
            "reference_fold_proof_sha256": proof_sha,
            "source_tree_sha256": SHA_A,
        },
        "records": [record],
    }


def _write_archived_fold(directory: Path) -> None:
    directory.mkdir(parents=True)
    train_ids = ["train"]
    heldout_ids = ["held"]
    train_digest = stable_id_digest(train_ids)
    heldout_digest = stable_id_digest(heldout_ids)
    descriptor = {"path": "/archived/artifact", "sha256": SHA_A}
    stages = {}
    for name in REQUIRED_PIPELINE_STAGES:
        stage = {
            "status": "complete",
            "test_accessed": False,
            "inputs": [descriptor],
            "outputs": [descriptor],
        }
        if name in SUPERVISED_PIPELINE_STAGES:
            stage.update(
                {
                    "heldout_excluded": True,
                    "train_record_ids_sha256": train_digest,
                    "config": descriptor,
                    "checkpoint": descriptor,
                }
            )
        stages[name] = stage
    pipeline = {
        "format_version": 1,
        "kind": "null_release_full_chain_fold_pipeline",
        "fold_id": 0,
        "train_record_ids_sha256": train_digest,
        "heldout_record_ids_sha256": heldout_digest,
        "source_tree_sha256": SHA_A,
        "test_accessed": False,
        "sealed": True,
        "stages": stages,
        "source_revision_history": [],
    }
    pipeline_path = directory / "pipeline_manifest.json"
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")
    pipeline_sha = sha256_file(pipeline_path)

    artifact_sha = {"formal_cache": SHA_B}
    proof = {
        "format_version": 1,
        "kind": "null_release_full_chain_fold_proof",
        "fold_id": 0,
        "num_folds": 10,
        "excluded_heldout": True,
        "training_record_ids": train_ids,
        "heldout_record_ids": heldout_ids,
        "pipeline_manifest_sha256": pipeline_sha,
        "artifact_sha256": artifact_sha,
    }
    proof_path = directory / "fold_proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    proof_sha = sha256_file(proof_path)

    feature_path = directory / "heldout_features.pt"
    torch.save(
        _full_chain_payload(proof_sha=proof_sha, artifact_sha=artifact_sha),
        feature_path,
    )
    feature_sha = sha256_file(feature_path)
    feature_path.with_suffix(".pt.sha256").write_text(
        f"{feature_sha}  heldout_features.pt\n",
        encoding="utf-8",
    )
    archive = {
        "kind": "null_release_oof_fold_archive",
        "format_version": 1,
        "status": "cleaned",
        "fold_id": 0,
        "records": 1,
        "pipeline_sealed": True,
        "test_accessed": False,
        "heldout_features": {
            "path": "/archive/heldout_features.pt",
            "bytes": feature_path.stat().st_size,
            "sha256": feature_sha,
        },
        "fold_proof": {
            "path": "/archive/fold_proof.json",
            "bytes": proof_path.stat().st_size,
            "sha256": proof_sha,
        },
        "pipeline_manifest": {
            "path": "/archive/pipeline_manifest.json",
            "bytes": pipeline_path.stat().st_size,
            "sha256": pipeline_sha,
        },
        "proof_artifact_sha256": artifact_sha,
        "proof_artifact_matches": {"formal_cache": ["/archive/formal.pt"]},
        "pre_cleanup_validation": {
            "all_required_stages_complete": True,
            "pipeline_sealed": True,
            "test_accessed": False,
            "fixed_top4_valid": True,
            "self_contained_payload": True,
            "artifact_hashes_verified": True,
        },
        "post_cleanup_validation": {
            "records": 1,
            "self_contained_reload": True,
            "test_accessed": False,
        },
    }
    (directory / "fold_archive_manifest.json").write_text(
        json.dumps(archive),
        encoding="utf-8",
    )


def test_p4_access_gate_rejects_calibration_dev_and_test() -> None:
    assert parse_p4_development_folds("0-7") == tuple(range(8))
    with pytest.raises(PermissionError, match="folds 0-7"):
        parse_p4_development_folds("0-8")
    with pytest.raises(PermissionError, match="Dev/Test"):
        enforce_p4_development_access([0], scope_labels=["outputs/dev/cache.pt"])
    with pytest.raises(PermissionError, match="Dev/Test"):
        enforce_p4_development_access([0], scope_labels=["data/test.jsonl"])


def test_gold_free_candidate_view_drops_all_label_fields() -> None:
    output = gold_free_selector_record(_selector_record())

    assert output["metadata"]["record_id"] == "held"
    assert "gold_span_mask" not in output
    assert "gold_type_mask" not in output
    assert "gold_entities" not in output["metadata"]
    assert output["fixed_type_scores"].tolist() == [3.0, 2.0]


def test_archived_full_chain_fold_validates_without_live_checkpoints(
    tmp_path: Path,
) -> None:
    fold_dir = tmp_path / "fold0"
    _write_archived_fold(fold_dir)

    report = validate_archived_full_chain_fold(
        fold_dir,
        expected_fold_id=0,
    )

    assert report["status"] == "PASSED"
    assert report["records"] == 1
    assert report["stage_provenance"]["fine"]["heldout_excluded"] is True
    assert report["formal_span_identity_available"] is False
    assert report["test_accessed"] is False


def test_cross_cache_join_is_blocked_without_formal_span_identity() -> None:
    provenance = {
        "_heldout_record_ids": ["held"],
        "fold_proof_sha256": SHA_A,
        "heldout_feature_sha256": SHA_A,
        "formal_span_identity_available": False,
        "stage_provenance": {"stage1": {"checkpoint_sha256": SHA_A}},
    }
    gold_free = build_gold_free_candidate_payload(
        _selector_payload(checkpoint_sha=SHA_B),
        fold_id=0,
        source_cache_sha256=SHA_A,
        full_chain_provenance=provenance,
    )
    report = audit_cross_cache_candidate_identity(
        _full_chain_payload(),
        gold_free,
    )

    assert report["observable_row_identity_matches"] == 1
    assert report["stage1_checkpoint_identity"] is False
    assert report["formal_span_identity_available"] is False
    assert report["index_attachment_permitted"] is False
    assert (
        "candidate_and_full_chain_stage1_checkpoints_differ"
        in (report["index_attachment_blockers"])
    )


def test_manifest_digest_is_deterministic_and_records_blockers() -> None:
    provenance = [
        {
            "fold_id": fold_id,
            "formal_span_identity_available": False,
        }
        for fold_id in range(8)
    ]
    alignment = [
        {"fold_id": fold_id, "index_attachment_permitted": False}
        for fold_id in range(8)
    ]
    blockers = source_seal_blockers(provenance, alignment)
    first = build_source_manifest(
        provenance_reports=provenance,
        candidate_descriptors=[],
        alignment_reports=alignment,
        blockers=blockers,
    )
    second = build_source_manifest(
        provenance_reports=provenance,
        candidate_descriptors=[],
        alignment_reports=alignment,
        blockers=blockers,
    )

    assert first == second
    assert first["sealed"] is False
    assert first["status"] == "BLOCKED_UNSEALED"
    assert "frozen_model_g_formal_span_coordinates_unavailable" in blockers
    validate_manifest_sha256(first)
    assert attach_manifest_sha256(first) == first
