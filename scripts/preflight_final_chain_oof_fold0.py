#!/usr/bin/env python3
"""Seal and validate the authorized Train-only Fold-0 dry-run inputs."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.full_chain_oof_contract import (
    atomic_write_json,
    fold_from_manifest,
    record_id,
    source_tree_sha256,
    validate_fold_manifest,
)
from gmner.data.p4_r0b_regeneration_contract import (
    P4_R0B_ARTIFACT_IDENTITY,
    P4_R0B_EXECUTION_FOLDS,
    file_bundle_sha256,
    validate_r0b_preregistration,
)
from gmner.utils.io import read_jsonl, write_jsonl


FORMAL_CONFIGS = (
    "configs/fmnerg_twitter10000_stage1.yaml",
    "configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml",
    "configs/fmnerg_twitter10000_coarse_selector.yaml",
    "configs/fmnerg_twitter10000_fine_grounding_adapter.yaml",
    "configs/fmnerg_twitter10000_evidence_visibility.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        default="docs/experiments/final_chain_oof_fold0_dry_run_preregistration.json",
    )
    parser.add_argument(
        "--stage1-config", default="configs/fmnerg_twitter10000_stage1.yaml"
    )
    parser.add_argument(
        "--work-root", default="knowledge/final_chain_oof/fold0_dry_run"
    )
    parser.add_argument("--gpu-free-gib", type=float, default=None)
    parser.add_argument("--disk-free-gib", type=float, default=None)
    parser.add_argument("--force-fold-manifest", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_r0b_preregistration(authorization)
    work_root = resolve(root, args.work_root)
    expected_work_root = resolve(
        root, authorization["storage_contract"]["work_root"]
    )
    if work_root != expected_work_root:
        raise ValueError("Preflight work root differs from authorization.")

    resources = authorization["resource_budget"]
    disk_free_gib = (
        float(args.disk_free_gib)
        if args.disk_free_gib is not None
        else shutil.disk_usage(root).free / (1024**3)
    )
    if disk_free_gib < float(resources["preferred_free_disk_gib"]):
        raise RuntimeError(
            f"Disk gate failed: {disk_free_gib:.3f} GiB free, "
            f"requires {resources['preferred_free_disk_gib']} GiB."
        )
    if args.gpu_free_gib is None:
        raise ValueError("--gpu-free-gib must be supplied by the launcher.")
    if float(args.gpu_free_gib) < float(resources["gpu_free_gate_gib"]):
        raise RuntimeError(
            f"GPU gate failed: {args.gpu_free_gib:.3f} GiB free, "
            f"requires {resources['gpu_free_gate_gib']} GiB."
        )

    fold_dir = work_root / "fold0"
    folds_dir = work_root / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = folds_dir / "fold_summary.json"
    stage1_config = resolve(root, args.stage1_config)
    if args.force_fold_manifest or not manifest_path.exists():
        command = [
            sys.executable,
            str(root / "scripts" / "build_evidence_folds.py"),
            "--config",
            str(stage1_config),
            "--output-dir",
            str(folds_dir),
            "--num-folds",
            "10",
            "--seed",
            str(authorization["source_contract"]["outer_seed"]),
        ]
        if args.force_fold_manifest:
            command.append("--force")
        subprocess.run(command, cwd=root, check=True)

    manifest = validate_fold_manifest(manifest_path, expected_num_folds=10)
    fold = fold_from_manifest(manifest, 0)
    outer_train = read_jsonl(Path(fold["train_file"]))
    heldout = read_jsonl(Path(fold["heldout_file"]))
    if len(outer_train) != 6300 or len(heldout) != 700:
        raise ValueError("Fold-0 outer split is not 6300/700.")
    outer_ids = [record_id(row) for row in outer_train]
    heldout_ids = [record_id(row) for row in heldout]
    if set(outer_ids) & set(heldout_ids):
        raise ValueError("Fold-0 outer train and heldout IDs overlap.")

    shuffled = list(range(len(outer_train)))
    random.Random(
        int(authorization["source_contract"]["inner_selection_seed"])
    ).shuffle(shuffled)
    selection_set = set(shuffled[:700])
    fit_records = [row for index, row in enumerate(outer_train) if index not in selection_set]
    selection_records = [outer_train[index] for index in shuffled[:700]]
    fit_path = fold_dir / "fit_train.jsonl"
    selection_path = fold_dir / "checkpoint_selection.jsonl"
    write_jsonl(fit_path, fit_records)
    write_jsonl(selection_path, selection_records)
    fit_ids = [record_id(row) for row in fit_records]
    selection_ids = [record_id(row) for row in selection_records]
    if len(fit_ids) != 5600 or len(selection_ids) != 700:
        raise ValueError("Nested Fold-0 split is not 5600/700.")
    if set(fit_ids) & set(selection_ids) or set(heldout_ids) & (
        set(fit_ids) | set(selection_ids)
    ):
        raise ValueError("Nested Fold-0 partitions overlap.")
    if set(fit_ids) | set(selection_ids) != set(outer_ids):
        raise ValueError("Nested Fold-0 partitions do not cover outer train.")

    implementation_paths = [
        root / "scripts" / "preflight_final_chain_oof_fold0.py",
        root / "scripts" / "run_null_release_full_chain_oof_fold.py",
        root / "scripts" / "build_record_candidate_cache.py",
        root / "scripts" / "build_p4_r0b_m33a_formal_oof.py",
        root / "scripts" / "materialize_final_chain_oof_fold0_rows.py",
        root / "gmner" / "data" / "mmner_dataset.py",
        root / "gmner" / "data" / "collator.py",
        root / "gmner" / "data" / "p4_r0b_regeneration_contract.py",
    ]
    implementation = file_bundle_sha256(implementation_paths)
    manifest["source_tree_sha256"] = source_tree_sha256(root)
    manifest["git_commit"] = git_head(root)
    manifest["regeneration"] = {
        "artifact_identity": P4_R0B_ARTIFACT_IDENTITY,
        "regeneration_authorization_sha256": sha256_file(authorization_path),
        "regeneration_experiment_id": authorization["experiment_id"],
        "execution_folds": list(P4_R0B_EXECUTION_FOLDS),
        "chain_contract": authorization["chain_contract"],
        "implementation_fingerprints": implementation,
        "official_dev_access": False,
        "test_accessed": False,
    }
    atomic_write_json(manifest_path, manifest)
    validate_fold_manifest(manifest_path, expected_num_folds=10)

    config_descriptors = []
    for value in FORMAL_CONFIGS:
        path = resolve(root, value)
        config_descriptors.append(
            {"path": str(path), "sha256": sha256_file(path)}
        )
    schema_path = root / "docs" / "experiments" / "final_chain_oof_minimum_row_schema.json"
    report = {
        "kind": "final_chain_oof_fold0_d0_preflight",
        "format_version": 1,
        "status": "PASSED",
        "fold0_execution_authorized": True,
        "git_head": git_head(root),
        "source_tree_sha256": source_tree_sha256(root),
        "authorization": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "fold_manifest": str(manifest_path),
        "fold_manifest_sha256": sha256_file(manifest_path),
        "outer_train_records": len(outer_ids),
        "fit_records": len(fit_ids),
        "selection_records": len(selection_ids),
        "heldout_records": len(heldout_ids),
        "outer_train_ids_sha256": stable_id_digest(outer_ids),
        "fit_ids_sha256": stable_id_digest(fit_ids),
        "selection_ids_sha256": stable_id_digest(selection_ids),
        "heldout_ids_sha256": stable_id_digest(heldout_ids),
        "fit_file": str(fit_path),
        "fit_file_sha256": sha256_file(fit_path),
        "selection_file": str(selection_path),
        "selection_file_sha256": sha256_file(selection_path),
        "heldout_file": str(Path(fold["heldout_file"])),
        "heldout_file_sha256": fold["heldout_file_sha256"],
        "formal_configs": config_descriptors,
        "schema": str(schema_path.resolve()),
        "schema_sha256": sha256_file(schema_path),
        "gpu_free_gib": float(args.gpu_free_gib),
        "disk_free_gib": disk_free_gib,
        "folds_1_9_accessed": False,
        "official_dev_accessed": False,
        "test_accessed": False,
    }
    report_path = fold_dir / "d0_preflight.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
