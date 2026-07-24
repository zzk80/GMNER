from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gmner.data.full_chain_oof_contract import (
    FULL_CHAIN_FOLD_MANIFEST_KIND,
    FULL_CHAIN_FOLD_MANIFEST_VERSION,
    FULL_CHAIN_PIPELINE_KIND,
    FULL_CHAIN_PIPELINE_VERSION,
    REQUIRED_PIPELINE_STAGES,
    SUPERVISED_PIPELINE_STAGES,
)
from gmner.data.null_release_oof_cache import (
    NULL_RELEASE_OOF_FORMAT_VERSION,
    NULL_RELEASE_OOF_KIND,
    sha256_file,
    stable_id_digest,
)
from tools.archive_null_release_oof_fold import archive_fold


def _write_records(path: Path, record_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record_id in record_ids:
            stream.write(json.dumps({"id": record_id}) + "\n")


def _descriptor(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _batch(record_id: str) -> dict:
    span_mask = torch.ones(1, 1, dtype=torch.bool)
    candidate_mask = torch.ones(1, 1, 5, dtype=torch.bool)
    top4 = torch.tensor([[[0, 1, 2, 3]]])
    return {
        "fold_id": 0,
        "record_ids": [record_id],
        "fine_outputs": {
            "candidate_mask": candidate_mask,
            "final_region_logits": torch.zeros(1, 1, 5),
            "fine_top4_indices": top4,
            "fine_top4_valid_mask": torch.ones_like(top4, dtype=torch.bool),
            "span_grounding_state": torch.zeros(1, 1, 4),
            "region_grounding_state": torch.zeros(1, 5, 4),
            "type_grounding_state": torch.zeros(1, 1, 4),
            "candidate_source_ids": torch.zeros(1, 1, 5, dtype=torch.long),
            "base_log_prior": torch.zeros(1, 1, 5),
            "coarse_log_prior": torch.zeros(1, 1, 5),
            "base_rank": torch.zeros(1, 1, 5),
            "coarse_rank": torch.zeros(1, 1, 5),
            "detector_rank": torch.zeros(1, 1, 5),
            "fixed_type_region_compatibility": torch.zeros(1, 1, 5),
            "promoted_candidate_mask": torch.zeros(
                1, 1, 5, dtype=torch.bool
            ),
            "fixed_type_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        "hierarchy_outputs": {"fixed_type_ids": torch.zeros(1, 1)},
        "evidence_outputs": {"evidence_scalar_features": torch.zeros(1, 1, 2)},
        "expanded": {"span_mask": span_mask},
        "reliability_outputs": {"reliability_probability": torch.zeros(1, 1)},
        "current_visible": torch.zeros_like(span_mask),
        "base_is_null": torch.ones_like(span_mask),
        "deployment_span_mask": span_mask.clone(),
    }


def _build_fold(tmp_path: Path, *, sealed: bool = True) -> dict[str, Path]:
    project_root = tmp_path
    fold_work = (
        project_root / "knowledge" / "null_release_oof" / "roberta128" / "fold0"
    )
    output_root = (
        project_root / "outputs" / "null_release_oof" / "roberta128" / "fold0"
    )
    folds_root = fold_work.parent / "folds"
    fold_work.mkdir(parents=True)
    output_root.mkdir(parents=True)

    record_ids = [str(index) for index in range(10)]
    folds = []
    for fold_id, heldout_id in enumerate(record_ids):
        heldout_ids = [heldout_id]
        train_ids = [value for value in record_ids if value != heldout_id]
        train_file = folds_root / f"train_fold{fold_id}.jsonl"
        heldout_file = folds_root / f"heldout_fold{fold_id}.jsonl"
        _write_records(train_file, train_ids)
        _write_records(heldout_file, heldout_ids)
        folds.append(
            {
                "fold": fold_id,
                "train_file": str(train_file.resolve()),
                "heldout_file": str(heldout_file.resolve()),
                "train_file_sha256": sha256_file(train_file),
                "heldout_file_sha256": sha256_file(heldout_file),
                "train_record_ids": train_ids,
                "heldout_record_ids": heldout_ids,
                "train_record_ids_sha256": stable_id_digest(train_ids),
                "heldout_record_ids_sha256": stable_id_digest(heldout_ids),
            }
        )
    fold_manifest = {
        "format_version": FULL_CHAIN_FOLD_MANIFEST_VERSION,
        "kind": FULL_CHAIN_FOLD_MANIFEST_KIND,
        "num_folds": 10,
        "source_split": "train",
        "test_accessed": False,
        "source_tree_sha256": "source-tree",
        "records": len(record_ids),
        "record_ids": record_ids,
        "record_ids_sha256": stable_id_digest(record_ids),
        "folds": folds,
    }
    fold_manifest_path = folds_root / "fold_summary.json"
    fold_manifest_path.write_text(json.dumps(fold_manifest), encoding="utf-8")

    candidate = fold_work / "candidates" / "train.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"candidate")
    siglip_manifest = fold_work / "siglip2" / "train" / "manifest.json"
    siglip_manifest.parent.mkdir(parents=True)
    siglip_manifest.write_text("{}", encoding="utf-8")

    stages = {}
    for name in REQUIRED_PIPELINE_STAGES:
        config = fold_work / "configs" / f"{name}.yaml"
        config.parent.mkdir(exist_ok=True)
        config.write_text(f"stage: {name}\n", encoding="utf-8")
        if name in SUPERVISED_PIPELINE_STAGES:
            checkpoint = output_root / name / "best_model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"checkpoint-{name}".encode())
            output = checkpoint
        elif name == "candidate_caches":
            output = candidate
        else:
            output = siglip_manifest
        stage = {
            "status": "complete",
            "test_accessed": False,
            "inputs": [_descriptor(Path(folds[0]["train_file"]))],
            "outputs": [_descriptor(output)],
        }
        if name in SUPERVISED_PIPELINE_STAGES:
            stage.update(
                {
                    "heldout_excluded": True,
                    "train_record_ids_sha256": folds[0][
                        "train_record_ids_sha256"
                    ],
                    "config": _descriptor(config),
                    "checkpoint": _descriptor(checkpoint),
                }
            )
        stages[name] = stage
    release_config = fold_work / "configs" / "release_materialize.yaml"
    release_config.write_text("mode: release\n", encoding="utf-8")
    pipeline = {
        "format_version": FULL_CHAIN_PIPELINE_VERSION,
        "kind": FULL_CHAIN_PIPELINE_KIND,
        "fold_id": 0,
        "num_folds": 10,
        "fold_manifest": str(fold_manifest_path.resolve()),
        "fold_manifest_sha256": sha256_file(fold_manifest_path),
        "source_tree_sha256": fold_manifest["source_tree_sha256"],
        "train_record_ids_sha256": folds[0]["train_record_ids_sha256"],
        "heldout_record_ids_sha256": folds[0]["heldout_record_ids_sha256"],
        "test_accessed": False,
        "sealed": sealed,
        "stages": stages,
    }
    pipeline_path = fold_work / "pipeline_manifest.json"
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")

    artifact_hashes = {
        "stage1": stages["stage1"]["checkpoint"]["sha256"],
        "formal_cache": sha256_file(candidate),
        "siglip2_manifest": sha256_file(siglip_manifest),
        "action_config": sha256_file(release_config),
    }
    proof = {
        "format_version": 1,
        "kind": "null_release_full_chain_fold_proof",
        "fold_id": 0,
        "num_folds": 10,
        "excluded_heldout": True,
        "fold_summary": str(fold_manifest_path.resolve()),
        "fold_summary_sha256": sha256_file(fold_manifest_path),
        "pipeline_manifest": str(pipeline_path.resolve()),
        "pipeline_manifest_sha256": sha256_file(pipeline_path),
        "train_file": folds[0]["train_file"],
        "train_file_sha256": folds[0]["train_file_sha256"],
        "heldout_file": folds[0]["heldout_file"],
        "heldout_file_sha256": folds[0]["heldout_file_sha256"],
        "training_record_ids": folds[0]["train_record_ids"],
        "heldout_record_ids": folds[0]["heldout_record_ids"],
        "artifact_sha256": artifact_hashes,
    }
    proof_path = fold_work / "fold_proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    payload = {
        "metadata": {
            "format_version": NULL_RELEASE_OOF_FORMAT_VERSION,
            "kind": NULL_RELEASE_OOF_KIND,
            "full_chain_oof": True,
            "fold_id": 0,
            "num_folds": 10,
            "records": 1,
            "record_ids_sha256": stable_id_digest(["0"]),
            "training_record_ids": folds[0]["train_record_ids"],
            "heldout_record_ids": ["0"],
            "excluded_heldout": True,
            "includes_reliability": True,
            "fold_proof": str(proof_path.resolve()),
            "fold_proof_sha256": sha256_file(proof_path),
            "artifact_sha256": artifact_hashes,
        },
        "batches": [_batch("0")],
    }
    torch.save(payload, fold_work / "heldout_features.pt")

    (output_root / "fine" / "metrics.json").write_text(
        '{"gmner_score": 0.6}', encoding="utf-8"
    )
    (output_root / "fine" / "train.log").write_text(
        "training complete\n", encoding="utf-8"
    )
    (project_root / "null_release_oof_fold0.log").write_text(
        "pipeline complete\n", encoding="utf-8"
    )
    return {
        "project_root": project_root,
        "fold_work": fold_work,
        "output_root": output_root,
    }


def test_archive_fold_is_dry_run_by_default_and_cleanup_is_idempotent(
    tmp_path: Path,
) -> None:
    paths = _build_fold(tmp_path)

    dry_run = archive_fold(
        project_root=paths["project_root"],
        fold_work=paths["fold_work"],
        fold_id=0,
        output_work_root=paths["output_root"],
        execute=False,
    )

    assert dry_run["status"] == "dry_run"
    assert dry_run["records"] == 1
    assert dry_run["test_accessed"] is False
    assert paths["output_root"].exists()
    assert (paths["fold_work"] / "candidates").exists()
    assert not (paths["fold_work"] / "fold_archive_manifest.json").exists()

    cleaned = archive_fold(
        project_root=paths["project_root"],
        fold_work=paths["fold_work"],
        fold_id=0,
        output_work_root=paths["output_root"],
        execute=True,
        checkpoint_backup_note="test backup",
    )

    assert cleaned["status"] == "cleaned"
    assert cleaned["post_cleanup_validation"]["records"] == 1
    assert cleaned["post_cleanup_validation"]["self_contained_reload"] is True
    assert not paths["output_root"].exists()
    assert not (paths["fold_work"] / "candidates").exists()
    assert not (paths["fold_work"] / "siglip2").exists()
    assert (paths["fold_work"] / "configs").is_dir()
    assert (paths["fold_work"] / "heldout_features.pt").is_file()
    assert (paths["fold_work"] / "archive" / "outputs" / "fine" / "metrics.json").is_file()
    assert (
        paths["fold_work"]
        / "archive"
        / "pipeline_logs"
        / "null_release_oof_fold0.log"
    ).is_file()

    repeated = archive_fold(
        project_root=paths["project_root"],
        fold_work=paths["fold_work"],
        fold_id=0,
        output_work_root=None,
        execute=True,
    )
    assert repeated["status"] == "cleaned"
    assert repeated["post_cleanup_validation"]["records"] == 1


def test_archive_fold_rejects_an_unsealed_pipeline(tmp_path: Path) -> None:
    paths = _build_fold(tmp_path, sealed=False)

    with pytest.raises(ValueError, match="must be sealed"):
        archive_fold(
            project_root=paths["project_root"],
            fold_work=paths["fold_work"],
            fold_id=0,
            output_work_root=paths["output_root"],
            execute=False,
        )

    assert paths["output_root"].exists()
    assert (paths["fold_work"] / "candidates").exists()
