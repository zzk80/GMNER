"""Aggregate the eight isolated P4-R0-B semantic consistency reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    atomic_write_json,
    validate_fold_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file
from gmner.data.p4_r0b_regeneration_contract import (
    P4_R0B_AGGREGATE_REPORT_KIND,
    P4_R0B_ARTIFACT_IDENTITY,
    P4_R0B_EXECUTION_FOLDS,
    P4_R0B_FOLD_REPORT_KIND,
    canonical_json_sha256,
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
        "--fold-summary",
        default=(
            "knowledge/p4_r0b_full_chain_oof/roberta128/"
            "folds/fold_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "knowledge/p4_r0b_full_chain_oof/roberta128/"
            "regeneration_aggregate_report.json"
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
    authorization_sha256 = sha256_file(authorization_path)
    experiment_id = str(authorization["experiment_id"])
    work_root = resolve(
        authorization["storage_contract"]["work_root"], root
    )
    manifest = validate_fold_manifest(
        resolve(args.fold_summary, root),
        expected_num_folds=10,
        verify_fold_ids=P4_R0B_EXECUTION_FOLDS,
    )

    reports = []
    archives = []
    covered_ids: set[str] = set()
    canonical_fold_digests = []
    for fold_id in P4_R0B_EXECUTION_FOLDS:
        fold_work = work_root / f"fold{fold_id}"
        report_path = fold_work / "regeneration_semantic_report.json"
        archive_path = fold_work / "fold_archive_manifest.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
        if report.get("kind") != P4_R0B_FOLD_REPORT_KIND:
            raise ValueError(f"Fold {fold_id} has an invalid semantic report.")
        expected_identity = {
            "artifact_identity": P4_R0B_ARTIFACT_IDENTITY,
            "regeneration_authorization_sha256": authorization_sha256,
            "regeneration_fold_id": fold_id,
            "regeneration_experiment_id": experiment_id,
        }
        for key, expected in expected_identity.items():
            if report.get(key) != expected or archive.get(key) != expected:
                raise ValueError(f"Fold {fold_id} identity mismatch for {key}.")
        if archive.get("status") != "CLEANED":
            raise ValueError(f"Fold {fold_id} is not cleaned and sealed.")
        if archive.get("post_cleanup_reload_passed") is not True:
            raise ValueError(f"Fold {fold_id} failed post-cleanup reload.")
        for excluded in (
            "siglip2_included",
            "reliability_included",
            "null_release_included",
        ):
            if report.get(excluded) is not False:
                raise ValueError(
                    f"Fold {fold_id} unexpectedly includes {excluded}."
                )
        for artifact in archive["retained_artifacts"].values():
            path = Path(str(artifact["path"]))
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"Fold {fold_id} retained artifact changed.")

        fold = next(
            value
            for value in manifest["folds"]
            if int(value["fold"]) == fold_id
        )
        heldout_ids = {str(value) for value in fold["heldout_record_ids"]}
        if covered_ids & heldout_ids:
            raise ValueError("Regenerated fold coverage overlaps.")
        covered_ids.update(heldout_ids)
        if int(report["records"]) != len(heldout_ids):
            raise ValueError(f"Fold {fold_id} record count differs.")
        reports.append(
            {
                "fold_id": fold_id,
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "semantic_gate_passed": bool(
                    report["fold_semantic_gate_passed"]
                ),
            }
        )
        archives.append(
            {
                "fold_id": fold_id,
                "path": str(archive_path),
                "sha256": sha256_file(archive_path),
            }
        )
        canonical_fold_digests.append(
            {
                "fold_id": fold_id,
                **dict(report["canonical_formal_predictions"]),
            }
        )

    required_coverage = int(
        authorization["semantic_consistency_gate"]["required_record_coverage"]
    )
    all_folds_passed = all(
        item["semantic_gate_passed"] for item in reports
    )
    coverage_passed = len(covered_ids) == required_coverage
    gate_passed = all_folds_passed and coverage_passed
    output = {
        "kind": P4_R0B_AGGREGATE_REPORT_KIND,
        "format_version": 1,
        "status": (
            "SEMANTIC_GATE_PASSED_AWAITING_SEPARATE_SIDECAR_AUTHORIZATION"
            if gate_passed
            else "SEMANTIC_GATE_FAILED_P4_REMAINS_BLOCKED"
        ),
        "artifact_identity": P4_R0B_ARTIFACT_IDENTITY,
        "regeneration_authorization_sha256": authorization_sha256,
        "regeneration_experiment_id": experiment_id,
        "folds": list(P4_R0B_EXECUTION_FOLDS),
        "fold_reports": reports,
        "fold_archives": archives,
        "record_coverage": len(covered_ids),
        "required_record_coverage": required_coverage,
        "record_coverage_passed": coverage_passed,
        "all_fold_semantic_gates_passed": all_folds_passed,
        "semantic_gate_passed": gate_passed,
        "canonical_formal_fold_digest_sha256": canonical_json_sha256(
            canonical_fold_digests
        ),
        "formal_sidecar_generated": False,
        "attached_to_p4": False,
        "siglip2_included": False,
        "reliability_included": False,
        "null_release_included": False,
        "folds_8_9_accessed": False,
        "p4_dev_accessed": False,
        "oracle_run": False,
        "p4_1_run": False,
        "test_accessed": False,
    }
    output_path = resolve(args.output, root)
    atomic_write_json(output_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
