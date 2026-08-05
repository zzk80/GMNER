#!/usr/bin/env python3
"""Prepare one authorized folds 1-9 final-chain OOF population fold."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.final_chain_oof_population_contract import (
    validate_final_chain_authorization,
)
from gmner.data.full_chain_oof_contract import (
    atomic_write_json,
    fold_from_manifest,
    record_id,
    source_tree_sha256,
    validate_fold_manifest,
)
from gmner.data.p4_r0b_regeneration_contract import file_bundle_sha256
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
        default="docs/experiments/final_chain_oof_folds1_9_authorization.json",
    )
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument(
        "--work-root", default="knowledge/final_chain_oof/population_folds1_9"
    )
    parser.add_argument("--gpu-free-gib", type=float, required=True)
    parser.add_argument("--disk-free-gib", type=float, default=None)
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
    contract = validate_final_chain_authorization(
        authorization, fold_id=args.fold_id
    )
    storage = dict(authorization["storage_contract"])
    work_root = resolve(root, args.work_root)
    if work_root != resolve(root, storage["work_root"]):
        raise ValueError("Population work root differs from authorization.")
    resources = dict(authorization["resource_budget"])
    disk_free_gib = (
        float(args.disk_free_gib)
        if args.disk_free_gib is not None
        else shutil.disk_usage(root).free / (1024**3)
    )
    if float(args.gpu_free_gib) < float(resources["gpu_free_gate_gib"]):
        raise RuntimeError("GPU availability Gate failed.")
    if disk_free_gib < float(resources["preferred_free_disk_gib"]):
        raise RuntimeError("Preferred disk availability Gate failed.")

    source_manifest = resolve(root, storage["source_fold_manifest"])
    if sha256_file(source_manifest) != str(storage["source_fold_manifest_sha256"]):
        raise RuntimeError("Frozen Fold-0 source manifest SHA256 changed.")
    target_manifest = work_root / "folds" / "fold_summary.json"
    if not target_manifest.exists():
        source_payload = validate_fold_manifest(
            source_manifest, expected_num_folds=10
        )
        payload = copy.deepcopy(source_payload)
        implementation_paths = [
            root / "scripts" / "preflight_final_chain_oof_population_fold.py",
            root / "scripts" / "run_null_release_full_chain_oof_fold.py",
            root / "scripts" / "build_record_candidate_cache.py",
            root / "scripts" / "build_p4_r0b_m33a_formal_oof.py",
            root / "scripts" / "materialize_final_chain_oof_fold0_rows.py",
            root / "scripts" / "audit_final_chain_oof_fold_completion.py",
            root / "scripts" / "audit_final_chain_oof_fold0_supervision.py",
            root / "scripts" / "seal_cleanup_final_chain_oof_population_fold.py",
            root / "scripts" / "summarize_final_chain_oof_population.py",
            root / "tools" / "monitor_final_chain_oof_fold.py",
            root / "tools" / "run_final_chain_oof_folds1_9.sh",
            root / "gmner" / "data" / "final_chain_oof_population_contract.py",
            root / "gmner" / "data" / "mmner_dataset.py",
            root / "gmner" / "data" / "collator.py",
            root / "docs" / "experiments" / "final_chain_oof_folds1_9_authorization.json",
            root / "docs" / "experiments" / "final_chain_oof_minimum_row_schema.json",
        ]
        payload["source_tree_sha256"] = source_tree_sha256(root)
        payload["git_commit"] = git_head(root)
        payload["regeneration"] = {
            "artifact_identity": contract.artifact_identity,
            "regeneration_authorization_sha256": sha256_file(authorization_path),
            "regeneration_experiment_id": contract.experiment_id,
            "execution_folds": list(contract.execution_folds),
            "chain_contract": copy.deepcopy(authorization["chain_contract"]),
            "implementation_fingerprints": file_bundle_sha256(implementation_paths),
            "source_fold_manifest_sha256": sha256_file(source_manifest),
            "official_dev_access": False,
            "test_accessed": False,
        }
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target_manifest, payload)
    manifest = validate_fold_manifest(target_manifest, expected_num_folds=10)
    regeneration = dict(manifest.get("regeneration") or {})
    if (
        regeneration.get("artifact_identity") != contract.artifact_identity
        or regeneration.get("regeneration_authorization_sha256")
        != sha256_file(authorization_path)
        or tuple(regeneration.get("execution_folds") or ())
        != contract.execution_folds
    ):
        raise RuntimeError("Population fold manifest identity changed.")

    fold = fold_from_manifest(manifest, args.fold_id)
    outer_train = read_jsonl(Path(fold["train_file"]))
    heldout = read_jsonl(Path(fold["heldout_file"]))
    outer_ids = [record_id(row) for row in outer_train]
    heldout_ids = [record_id(row) for row in heldout]
    if len(outer_ids) != 6300 or len(heldout_ids) != 700:
        raise ValueError("Outer split is not 6300/700.")
    if set(outer_ids) & set(heldout_ids):
        raise ValueError("Outer train and heldout IDs overlap.")

    fold_dir = work_root / f"fold{args.fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed = int(authorization["source_contract"]["inner_selection_seed_base"]) + int(args.fold_id)
    shuffled = list(range(len(outer_train)))
    random.Random(seed).shuffle(shuffled)
    selection_set = set(shuffled[:700])
    fit_records = [row for index, row in enumerate(outer_train) if index not in selection_set]
    selection_records = [outer_train[index] for index in shuffled[:700]]
    fit_path = fold_dir / "fit_train.jsonl"
    selection_path = fold_dir / "checkpoint_selection.jsonl"
    write_jsonl(fit_path, fit_records)
    write_jsonl(selection_path, selection_records)
    fit_ids = [record_id(row) for row in fit_records]
    selection_ids = [record_id(row) for row in selection_records]
    if set(fit_ids) & set(selection_ids) or set(heldout_ids) & (set(fit_ids) | set(selection_ids)):
        raise ValueError("Nested partitions overlap.")
    if set(fit_ids) | set(selection_ids) != set(outer_ids):
        raise ValueError("Nested partitions do not cover outer train.")

    schema_path = root / "docs" / "experiments" / "final_chain_oof_minimum_row_schema.json"
    report = {
        "kind": "final_chain_oof_population_fold_preflight",
        "format_version": 1,
        "status": "PASSED",
        "fold_id": int(args.fold_id),
        "execution_authorized": True,
        "git_head": git_head(root),
        "source_tree_sha256": source_tree_sha256(root),
        "authorization": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "fold_manifest": str(target_manifest),
        "fold_manifest_sha256": sha256_file(target_manifest),
        "outer_train_records": len(outer_ids),
        "fit_records": len(fit_ids),
        "selection_records": len(selection_ids),
        "heldout_records": len(heldout_ids),
        "inner_selection_seed": seed,
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
        "formal_configs": [
            {"path": str(resolve(root, value)), "sha256": sha256_file(resolve(root, value))}
            for value in FORMAL_CONFIGS
        ],
        "schema": str(schema_path.resolve()),
        "schema_sha256": sha256_file(schema_path),
        "gpu_free_gib": float(args.gpu_free_gib),
        "disk_free_gib": disk_free_gib,
        "other_folds_accessed": False,
        "official_dev_accessed": False,
        "test_accessed": False,
    }
    atomic_write_json(fold_dir / "d0_preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
