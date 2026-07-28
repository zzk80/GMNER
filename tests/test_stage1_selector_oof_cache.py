from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gmner.data.full_chain_oof_contract import (
    FULL_CHAIN_FOLD_MANIFEST_KIND,
    FULL_CHAIN_FOLD_MANIFEST_VERSION,
    source_tree_sha256,
)
from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from gmner.data.stage1_selector_oof_cache import (
    build_dev_selector_payload,
    build_fold_selector_payload,
    compact_candidate_record,
    validate_selector_dev_payload,
    validate_selector_oof_payload,
    validate_selector_record,
)
from gmner.utils.io import write_jsonl
from scripts.audit_stage1_selector_phase1 import audit_phase1
from scripts.merge_stage1_selector_oof import merge_caches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SOURCE_TREE_SHA256 = source_tree_sha256(PROJECT_ROOT)


def _full_record(record_id: str) -> dict:
    return {
        "span_candidates": torch.tensor([[0, 1], [2, 3], [4, 5]]),
        "span_mask": torch.ones(3, dtype=torch.bool),
        "span_features": torch.randn(3, 4),
        "span_base_scores": torch.tensor([3.0, 1.0, -1.0]),
        "span_source_ids": torch.tensor([0, 2, 3]),
        "span_lengths": torch.ones(3),
        "type_candidates": torch.tensor([[1, 0], [2, 1], [3, 0]]),
        "type_base_scores": torch.tensor(
            [[2.0, 0.0], [1.5, 0.5], [0.5, 0.0]]
        ),
        "fixed_type_ids": torch.tensor([1, 2, 3]),
        "base_region_indices": torch.tensor([1, 2, 0]),
        "gold_span_mask": torch.tensor([True, True, False]),
        "gold_type_mask": torch.tensor(
            [[True, False], [True, False], [False, False]]
        ),
        "region_features": torch.randn(4, 8),
        "metadata": {
            "record_id": record_id,
            "text": "alpha beta gamma",
            "tokens": ["alpha", "beta", "gamma"],
            "candidate_sources": ["stage1", "kbest", "perturbation"],
            "stage1_predictions": [
                {"span": [0, 1], "type_id": 1, "region_index": 1}
            ],
            "gold_entities": [
                {
                    "span": [0, 1],
                    "type_id": 1,
                    "visible": True,
                    "region_positive_indices": [1],
                },
                {
                    "span": [2, 3],
                    "type_id": 2,
                    "visible": True,
                    "region_positive_indices": [2],
                },
            ],
            "null_region_index": 0,
        },
    }


def _candidate_payload(record_id: str, fold_id: int) -> dict:
    return {
        "metadata": {
            "format_version": 2,
            "oof_heldout": True,
            "oof_fold_id": fold_id,
            "source2id": {
                "stage1": 0,
                "viterbi": 1,
                "kbest": 2,
                "perturbation": 3,
            },
            "hidden_size": 4,
            "candidate_config": {"k_best": 6, "max_span_candidates": 12},
            "candidate_config_sha256": "candidate-sha",
            "stage1_checkpoint_sha256": f"checkpoint-{fold_id}",
            "data_source": f"heldout-{fold_id}.jsonl",
            "data_source_sha256": f"heldout-sha-{fold_id}",
        },
        "records": [_full_record(record_id)],
    }


def _selector_payload(record_id: str, fold_id: int) -> dict:
    return build_fold_selector_payload(
        _candidate_payload(record_id, fold_id),
        fold_id=fold_id,
        num_folds=10,
        source_candidate_cache=f"full-{fold_id}.pt",
        source_candidate_cache_sha256=f"full-sha-{fold_id}",
        stage1_config=f"stage1-{fold_id}.yaml",
        stage1_config_sha256=f"config-sha-{fold_id}",
        fold_manifest="fold_summary.json",
        fold_manifest_sha256="manifest-sha",
        reference_fold_proof=f"fold{fold_id}/fold_proof.json",
        reference_fold_proof_sha256=f"proof-sha-{fold_id}",
        git_commit="commit",
        source_tree_sha256=TEST_SOURCE_TREE_SHA256,
    )


def _write_fold_manifest(path: Path) -> dict:
    record_ids = [str(index) for index in range(10)]
    folds = []
    for fold_id, heldout_id in enumerate(record_ids):
        heldout_ids = [heldout_id]
        train_ids = [value for value in record_ids if value != heldout_id]
        train_path = path.parent / f"train_fold{fold_id}.jsonl"
        heldout_path = path.parent / f"heldout_fold{fold_id}.jsonl"
        write_jsonl(train_path, [{"id": value} for value in train_ids])
        write_jsonl(heldout_path, [{"id": heldout_id}])
        folds.append(
            {
                "fold": fold_id,
                "train_file": str(train_path.resolve()),
                "train_file_sha256": sha256_file(train_path),
                "heldout_file": str(heldout_path.resolve()),
                "heldout_file_sha256": sha256_file(heldout_path),
                "train_records": len(train_ids),
                "heldout_records": 1,
                "train_record_ids": train_ids,
                "heldout_record_ids": heldout_ids,
                "train_record_ids_sha256": stable_id_digest(train_ids),
                "heldout_record_ids_sha256": stable_id_digest(heldout_ids),
            }
        )
    manifest = {
        "format_version": FULL_CHAIN_FOLD_MANIFEST_VERSION,
        "kind": FULL_CHAIN_FOLD_MANIFEST_KIND,
        "source_split": "train",
        "test_accessed": False,
        "source": "train.jsonl",
        "source_sha256": "source",
        "config": "config.yaml",
        "config_sha256": "config",
        "git_commit": "commit",
        "source_tree_sha256": TEST_SOURCE_TREE_SHA256,
        "records": len(record_ids),
        "record_ids": record_ids,
        "record_ids_sha256": stable_id_digest(record_ids),
        "num_folds": 10,
        "seed": 42,
        "folds": folds,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_compact_record_preserves_formal_predictions_and_drops_regions() -> None:
    compact = compact_candidate_record(_full_record("7"), formal_source_id=0)

    assert compact["span_features"].dtype == torch.float16
    assert compact["span_lengths"].dtype == torch.int64
    assert compact["formal_candidate_mask"].tolist() == [True, False, False]
    assert compact["base_region_indices"].tolist() == [1, 2, 0]
    assert "region_features" not in compact
    validate_selector_record(compact, formal_source_id=0)


def test_compact_record_rejects_formal_prediction_drift() -> None:
    compact = compact_candidate_record(_full_record("7"), formal_source_id=0)
    compact["base_region_indices"][0] = 2

    with pytest.raises(ValueError, match="does not reproduce Stage1"):
        validate_selector_record(compact, formal_source_id=0)


def test_fold_payload_keeps_nonformal_gold_and_negative_candidates() -> None:
    payload = _selector_payload("7", 0)
    validated = validate_selector_oof_payload(
        payload,
        expected_fold_id=0,
        expected_record_ids=["7"],
    )
    record = validated["records"][0]

    assert (record["gold_span_mask"] & ~record["formal_candidate_mask"]).sum() == 1
    assert (
        record["span_mask"]
        & ~record["gold_span_mask"]
        & ~record["formal_candidate_mask"]
    ).sum() == 1


def test_dev_payload_is_non_oof_and_test_free() -> None:
    source = _candidate_payload("7", 0)
    source["metadata"].pop("oof_heldout")
    source["metadata"].pop("oof_fold_id")
    source["metadata"]["split"] = "dev"
    payload = build_dev_selector_payload(
        source,
        source_candidate_cache="dev_full.pt",
        source_candidate_cache_sha256="dev-full-sha",
        stage1_config="stage1.yaml",
        stage1_config_sha256="config-sha",
        git_commit="commit",
        source_tree_sha256=TEST_SOURCE_TREE_SHA256,
    )

    validated = validate_selector_dev_payload(
        payload,
        expected_record_ids=["7"],
    )
    assert validated["metadata"]["scope"] == "dev"
    assert validated["metadata"]["oof"] is False
    assert validated["metadata"]["test_accessed"] is False


def test_merge_requires_exact_ten_fold_coverage_and_audits_candidates(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "fold_summary.json"
    _write_fold_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    inputs = []
    for fold_id in range(10):
        payload = _selector_payload(str(fold_id), fold_id)
        payload["metadata"]["fold_manifest"] = str(manifest_path)
        payload["metadata"]["fold_manifest_sha256"] = manifest_sha
        path = tmp_path / f"fold{fold_id}.pt"
        torch.save(payload, path)
        inputs.append(path)
    output = tmp_path / "train_candidates.pt"

    summary = merge_caches(
        inputs,
        fold_summary=manifest_path,
        output=output,
    )
    merged = torch.load(output, map_location="cpu")

    assert summary["records"] == 10
    assert summary["audit"]["candidates"] == 30
    assert summary["audit"]["gold_span_formal"] == 10
    assert summary["audit"]["gold_span_nonformal"] == 10
    assert summary["audit"]["non_gold_nonformal"] == 10
    assert merged["metadata"]["scope"] == "oof_train"
    assert merged["metadata"]["test_accessed"] is False
    assert [
        record["metadata"]["record_id"] for record in merged["records"]
    ] == [str(index) for index in range(10)]


def test_merge_rejects_duplicate_fold(tmp_path: Path) -> None:
    manifest_path = tmp_path / "fold_summary.json"
    _write_fold_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    inputs = []
    for index in range(10):
        fold_id = 0 if index == 9 else index
        payload = _selector_payload(str(index), fold_id)
        payload["metadata"]["fold_manifest"] = str(manifest_path)
        payload["metadata"]["fold_manifest_sha256"] = manifest_sha
        path = tmp_path / f"input{index}.pt"
        torch.save(payload, path)
        inputs.append(path)

    with pytest.raises(ValueError, match="Duplicate selector OOF fold"):
        merge_caches(
            inputs,
            fold_summary=manifest_path,
            output=tmp_path / "merged.pt",
        )


def test_phase1_audit_requires_and_reports_train_dev_supervision(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "fold_summary.json"
    _write_fold_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    inputs = []
    for fold_id in range(10):
        payload = _selector_payload(str(fold_id), fold_id)
        payload["metadata"]["fold_manifest"] = str(manifest_path)
        payload["metadata"]["fold_manifest_sha256"] = manifest_sha
        path = tmp_path / f"fold{fold_id}.pt"
        torch.save(payload, path)
        inputs.append(path)
    train_path = tmp_path / "train.pt"
    merge_caches(inputs, fold_summary=manifest_path, output=train_path)

    dev_source = _candidate_payload("dev-0", 0)
    dev_source["metadata"].pop("oof_heldout")
    dev_source["metadata"].pop("oof_fold_id")
    dev_source["metadata"]["split"] = "dev"
    dev_payload = build_dev_selector_payload(
        dev_source,
        source_candidate_cache="dev_full.pt",
        source_candidate_cache_sha256="dev-full-sha",
        stage1_config="stage1.yaml",
        stage1_config_sha256="config-sha",
        git_commit="commit",
        source_tree_sha256=TEST_SOURCE_TREE_SHA256,
    )
    dev_path = tmp_path / "dev.pt"
    torch.save(dev_payload, dev_path)

    report = audit_phase1(
        train_cache=train_path,
        dev_cache=dev_path,
        output=tmp_path / "audit.json",
        expected_train_records=10,
        expected_dev_records=1,
    )

    assert report["contract_passed"] is True
    assert report["selector_training_supervision_present"] is True
    assert report["train"]["audit"]["gold_span_nonformal"] == 10
    assert report["dev"]["audit"]["non_gold_nonformal"] == 1
    assert report["test_accessed"] is False
