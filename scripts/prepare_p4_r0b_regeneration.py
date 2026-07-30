"""Prepare the authorized P4-R0-B full-chain OOF regeneration manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    atomic_write_json,
    validate_fold_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file
from gmner.data.p4_r0b_regeneration_contract import (
    P4_R0B_EXECUTION_FOLDS,
    build_regeneration_fold_manifest,
    file_bundle_sha256,
    tree_sha256,
    validate_r0b_preregistration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/"
            "p4_r0_b_full_chain_oof_regeneration_preregistration.json"
        ),
    )
    parser.add_argument(
        "--stage1-config",
        default="configs/fmnerg_twitter10000_stage1.yaml",
    )
    parser.add_argument(
        "--output-fold-summary",
        default=(
            "knowledge/p4_r0b_full_chain_oof/roberta128/"
            "folds/fold_summary.json"
        ),
    )
    parser.add_argument(
        "--output-report",
        default=(
            "knowledge/p4_r0b_full_chain_oof/roberta128/"
            "regeneration_preflight.json"
        ),
    )
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(args.authorization, root)
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    validate_r0b_preregistration(authorization)
    superseded = dict(authorization["supersedes_prior_lock"])
    superseded_path = resolve(superseded["path"], root)
    if sha256_file(superseded_path) != superseded["sha256"]:
        raise ValueError("The prior R0-A authorization lock has changed.")

    source = dict(authorization["source_contract"])
    storage = dict(authorization["storage_contract"])
    archived_path = resolve(source["archived_fold_summary"], root)
    archived_sha256 = sha256_file(archived_path)
    if archived_sha256 != source["archived_fold_summary_sha256"]:
        raise ValueError("Archived fold-summary SHA256 differs from authorization.")
    archived = validate_fold_manifest(
        archived_path,
        expected_num_folds=int(source["num_folds_in_partition"]),
        verify_fold_ids=P4_R0B_EXECUTION_FOLDS,
    )

    stage1_config_path = resolve(args.stage1_config, root)
    stage1_config = (
        yaml.safe_load(stage1_config_path.read_text(encoding="utf-8")) or {}
    )
    work_root = resolve(storage["work_root"], root)
    output_root = resolve(storage["output_root"], root)
    legacy_root = resolve(storage["legacy_evidence_root"], root)
    if len({work_root, output_root, legacy_root}) != 3:
        raise ValueError("R0-B work, output, and legacy roots must be distinct.")

    output_fold_summary = resolve(args.output_fold_summary, root)
    output_report = resolve(args.output_report, root)
    try:
        output_fold_summary.relative_to(work_root)
        output_report.relative_to(work_root)
    except ValueError as error:
        raise ValueError(
            "R0-B preflight outputs must remain under the independent work root."
        ) from error

    data = dict(stage1_config["data"])
    model = dict(stage1_config["model"])
    roberta_path = resolve(model["text_model_name"], root)
    vinvl_path = resolve(data["image_feature_dir"], root)
    dev_path = resolve(data["dev_file"], root)
    prior_paths = [
        resolve(data["groundability_type_priors"], root),
        resolve(data["groundability_mention_priors"], root),
    ]
    input_fingerprints = {
        "roberta": tree_sha256(roberta_path),
        "vinvl": tree_sha256(vinvl_path),
        "grounding_priors": file_bundle_sha256(prior_paths),
        "official_dev": {
            "path": str(dev_path),
            "bytes": dev_path.stat().st_size,
            "sha256": sha256_file(dev_path),
        },
        "archived_fold_summary": {
            "path": str(archived_path),
            "sha256": archived_sha256,
        },
        "superseded_prior_lock": {
            "path": str(superseded_path),
            "sha256": sha256_file(superseded_path),
        },
    }
    implementation_paths = [
        root / "gmner" / "data" / "p4_r0b_regeneration_contract.py",
        root / "scripts" / "prepare_p4_r0b_regeneration.py",
        root / "scripts" / "run_null_release_full_chain_oof_fold.py",
        root / "scripts" / "build_p4_r0b_m33a_formal_oof.py",
        root / "scripts" / "seal_p4_r0b_regenerated_fold.py",
        root / "scripts" / "aggregate_p4_r0b_regeneration.py",
        root / "tools" / "run_p4_r0b_full_chain_oof.sh",
    ]
    implementation_fingerprints = file_bundle_sha256(implementation_paths)

    regenerated = build_regeneration_fold_manifest(
        archived,
        root=root,
        stage1_config=stage1_config_path,
        authorization_path=authorization_path,
        authorization=authorization,
    )
    regenerated["regeneration"]["input_fingerprints"] = input_fingerprints
    regenerated["regeneration"]["implementation_fingerprints"] = (
        implementation_fingerprints
    )
    regenerated["regeneration"]["work_root"] = str(work_root)
    regenerated["regeneration"]["output_root"] = str(output_root)
    regenerated["regeneration"]["legacy_evidence_root"] = str(legacy_root)
    regenerated["regeneration"]["preflight_passed"] = True
    atomic_write_json(output_fold_summary, regenerated)

    validated = validate_fold_manifest(
        output_fold_summary,
        expected_num_folds=int(source["num_folds_in_partition"]),
        verify_fold_ids=P4_R0B_EXECUTION_FOLDS,
    )
    report = {
        "kind": "p4_r0_b_regeneration_preflight",
        "format_version": 1,
        "status": "PASSED",
        "artifact_identity": authorization["artifact_identity"],
        "experiment_id": authorization["experiment_id"],
        "chain_identity": authorization["chain_contract"]["identity"],
        "siglip2_included": False,
        "reliability_included": False,
        "null_release_included": False,
        "authorization": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "archived_fold_summary_sha256_exact": True,
        "execution_folds": list(P4_R0B_EXECUTION_FOLDS),
        "verified_fold_files": list(P4_R0B_EXECUTION_FOLDS),
        "folds_8_9_files_accessed": False,
        "upstream_official_dev_checkpoint_validation": True,
        "p4_dev_accessed": False,
        "test_accessed": False,
        "source_records": int(validated["records"]),
        "fold_summary": str(output_fold_summary),
        "fold_summary_sha256": sha256_file(output_fold_summary),
        "input_fingerprints": input_fingerprints,
        "implementation_fingerprints": implementation_fingerprints,
        "independent_roots": True,
    }
    atomic_write_json(output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
