from __future__ import annotations

import copy

import pytest
import torch

from gmner.data.null_release_oof_cache import (
    NULL_RELEASE_OOF_FORMAT_VERSION,
    NULL_RELEASE_OOF_KIND,
    stable_id_digest,
    validate_fold_oof_payload,
    validate_full_chain_oof_payload,
)
from scripts.merge_null_release_oof_features import main as merge_main


def _batch(fold_id: int, record_id: str) -> dict:
    span_mask = torch.ones(1, 2, dtype=torch.bool)
    candidate_mask = torch.ones(1, 2, 5, dtype=torch.bool)
    top4 = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]])
    return {
        "fold_id": fold_id,
        "record_ids": [record_id],
        "fine_outputs": {
            "candidate_mask": candidate_mask,
            "final_region_logits": torch.zeros(1, 2, 5),
            "fine_top4_indices": top4,
            "fine_top4_valid_mask": torch.ones_like(top4, dtype=torch.bool),
            "span_grounding_state": torch.zeros(1, 2, 4),
            "region_grounding_state": torch.zeros(1, 5, 4),
            "type_grounding_state": torch.zeros(1, 2, 4),
            "candidate_source_ids": torch.zeros(1, 2, 5, dtype=torch.long),
            "base_log_prior": torch.zeros(1, 2, 5),
            "coarse_log_prior": torch.zeros(1, 2, 5),
            "base_rank": torch.zeros(1, 2, 5),
            "coarse_rank": torch.zeros(1, 2, 5),
            "detector_rank": torch.zeros(1, 2, 5),
            "fixed_type_region_compatibility": torch.zeros(1, 2, 5),
            "promoted_candidate_mask": torch.zeros(1, 2, 5, dtype=torch.bool),
            "fixed_type_ids": torch.zeros(1, 2, dtype=torch.long),
        },
        "hierarchy_outputs": {},
        "evidence_outputs": {},
        "expanded": {"span_mask": span_mask},
        "reliability_outputs": {},
        "current_visible": torch.zeros_like(span_mask),
        "base_is_null": torch.ones_like(span_mask),
        "deployment_span_mask": span_mask.clone(),
    }


def _payload() -> dict:
    record_ids = ["0", "1"]
    return {
        "metadata": {
            "format_version": NULL_RELEASE_OOF_FORMAT_VERSION,
            "kind": NULL_RELEASE_OOF_KIND,
            "full_chain_oof": True,
            "num_folds": 2,
            "fold_ids": [0, 1],
            "records": 2,
            "record_ids_sha256": stable_id_digest(record_ids),
            "includes_reliability": True,
        },
        "batches": [_batch(0, "0"), _batch(1, "1")],
    }


def test_full_chain_oof_cache_requires_complete_disjoint_records() -> None:
    result = validate_full_chain_oof_payload(
        _payload(),
        expected_num_folds=2,
        expected_records=2,
        require_reliability=True,
    )
    assert result["records"] == 2


def test_full_chain_oof_cache_rejects_duplicate_record_ids() -> None:
    payload = copy.deepcopy(_payload())
    payload["batches"][1]["record_ids"] = ["0"]
    with pytest.raises(ValueError, match="duplicate record ids"):
        validate_full_chain_oof_payload(
            payload,
            expected_num_folds=2,
            expected_records=2,
            require_reliability=True,
        )


def test_full_chain_oof_cache_rejects_missing_reliability_provenance() -> None:
    payload = _payload()
    payload["metadata"]["includes_reliability"] = False
    with pytest.raises(ValueError, match="requires OOF Reliability"):
        validate_full_chain_oof_payload(
            payload,
            expected_num_folds=2,
            expected_records=2,
            require_reliability=True,
        )


def test_full_chain_oof_cache_rejects_duplicate_fixed_top4_actions() -> None:
    payload = _payload()
    payload["batches"][0]["fine_outputs"]["fine_top4_indices"][0, 0] = torch.tensor(
        [0, 1, 1, 2]
    )
    with pytest.raises(ValueError, match="duplicate actions"):
        validate_full_chain_oof_payload(
            payload,
            expected_num_folds=2,
            expected_records=2,
            require_reliability=True,
        )


def test_single_fold_cache_preserves_manifest_record_order() -> None:
    payload = _payload()
    payload["metadata"].pop("fold_ids")
    payload["metadata"]["fold_id"] = 0
    payload["metadata"]["num_folds"] = 10
    payload["metadata"]["records"] = 1
    payload["metadata"]["record_ids_sha256"] = stable_id_digest(["0"])
    payload["batches"] = [payload["batches"][0]]

    result = validate_fold_oof_payload(
        payload,
        expected_fold_id=0,
        expected_record_ids=["0"],
    )
    assert result["records"] == 1


def test_merge_requires_and_preserves_ten_fold_complements(
    tmp_path, monkeypatch
) -> None:
    record_ids = [str(index) for index in range(10)]
    inputs = []
    for fold_id, heldout_id in enumerate(record_ids):
        payload = {
            "metadata": {
                "format_version": NULL_RELEASE_OOF_FORMAT_VERSION,
                "kind": NULL_RELEASE_OOF_KIND,
                "full_chain_oof": True,
                "fold_id": fold_id,
                "num_folds": 10,
                "records": 1,
                "record_ids_sha256": stable_id_digest([heldout_id]),
                "training_record_ids": [
                    value for value in record_ids if value != heldout_id
                ],
                "heldout_record_ids": [heldout_id],
                "excluded_heldout": True,
                "includes_reliability": True,
                "fold_proof_sha256": f"proof-{fold_id}",
                "artifact_sha256": {"stage1": f"stage1-{fold_id}"},
            },
            "batches": [_batch(fold_id, heldout_id)],
        }
        path = tmp_path / f"fold{fold_id}.pt"
        torch.save(payload, path)
        inputs.append(path)
    output = tmp_path / "merged.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge_null_release_oof_features.py",
            "--inputs",
            *map(str, inputs),
            "--output",
            str(output),
            "--expected-records",
            "10",
        ],
    )
    merge_main()
    merged = torch.load(output, map_location="cpu")
    result = validate_full_chain_oof_payload(
        merged,
        expected_num_folds=10,
        expected_records=10,
        require_reliability=True,
    )
    assert result["records"] == 10
