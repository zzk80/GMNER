#!/usr/bin/env python3
"""Prepare and audit P4.0 source-development artifacts for OOF folds 0-7.

This command does not read calibration folds, Dev, or Test.  It does not
compute oracle labels and cannot authorize P4.1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from gmner.data.null_release_oof_cache import sha256_file
from gmner.data.p4_actionability_contract import (
    P4_DEVELOPMENT_FOLDS,
    P4_FORMAT_VERSION,
    P4_PROVENANCE_REPORT_KIND,
    audit_cross_cache_candidate_identity,
    build_gold_free_candidate_payload,
    build_source_manifest,
    candidate_source_statistics,
    enforce_p4_development_access,
    json_safe_provenance,
    parse_p4_development_folds,
    source_seal_blockers,
    validate_archived_full_chain_fold,
    validate_gold_free_candidate_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "gmner" / "data" / "p4_actionability_contract.py"
DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "p4_protected_joint_promotion_preregistration.json"
)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _validate_preregistration(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    authorization = dict(payload.get("authorization") or {})
    partition = dict(payload.get("oof_partition") or {})
    expected_development = list(P4_DEVELOPMENT_FOLDS)
    if authorization.get("p4_0_read_only_audit") is not True:
        raise PermissionError("P4.0 read-only audit is not authorized.")
    if authorization.get("p4_1_selector_training") is not False:
        raise PermissionError(
            "P4.1 must remain unauthorized during source preparation."
        )
    if authorization.get("downstream_rebuild") is not False:
        raise PermissionError("Downstream rebuild must remain locked.")
    if authorization.get("test_access") is not False:
        raise PermissionError("Test access must remain locked.")
    if partition.get("source_and_feature_development_folds") != expected_development:
        raise ValueError("P4 preregistration development folds differ from 0-7.")
    if partition.get("threshold_calibration_folds") != [8, 9]:
        raise ValueError("P4 preregistration calibration folds differ from 8-9.")
    return payload


def prepare_sources(
    *,
    folds: tuple[int, ...],
    full_chain_root: Path,
    candidate_root: Path,
    candidate_name_template: str,
    materialized_root: Path,
    preregistration: Path,
    report_path: Path,
    manifest_path: Path,
) -> dict:
    """Run provenance and source-schema audits without consulting labels."""

    folds = enforce_p4_development_access(
        folds,
        scope_labels=(
            full_chain_root,
            candidate_root,
            materialized_root,
            preregistration,
            report_path,
            manifest_path,
        ),
    )
    preregistration_payload = _validate_preregistration(preregistration)
    preregistration_sha = sha256_file(preregistration)
    implementation = {
        "git_head": _git_commit(),
        "contract_path": _logical_path(CONTRACT_PATH),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "driver_path": _logical_path(Path(__file__)),
        "driver_sha256": sha256_file(Path(__file__)),
    }

    provenance_reports: list[dict] = []
    alignment_reports: list[dict] = []
    candidate_descriptors: list[dict] = []
    source_statistics: list[dict] = []
    for fold_id in folds:
        fold_dir = full_chain_root / f"fold{fold_id}"
        provenance = validate_archived_full_chain_fold(
            fold_dir,
            expected_fold_id=fold_id,
        )
        candidate_path = candidate_root / candidate_name_template.format(fold=fold_id)
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"Missing P4 candidate source for fold {fold_id}: {candidate_path}"
            )
        candidate_sha = sha256_file(candidate_path)
        selector_payload = torch.load(candidate_path, map_location="cpu")
        gold_free = build_gold_free_candidate_payload(
            selector_payload,
            fold_id=fold_id,
            source_cache_sha256=candidate_sha,
            full_chain_provenance=provenance,
        )
        output_path = materialized_root / f"fold{fold_id}" / "candidates.pt"
        _atomic_torch_save(output_path, gold_free)
        reloaded = torch.load(output_path, map_location="cpu")
        validate_gold_free_candidate_payload(
            reloaded,
            expected_fold_id=fold_id,
        )
        materialized_sha = sha256_file(output_path)

        alignment = audit_cross_cache_candidate_identity(
            provenance["_payload"],
            reloaded,
        )
        alignment["fold_id"] = fold_id
        statistics = candidate_source_statistics(reloaded)
        statistics["fold_id"] = fold_id

        provenance_reports.append(provenance)
        alignment_reports.append(alignment)
        source_statistics.append(statistics)
        candidate_descriptors.append(
            {
                "fold_id": fold_id,
                "input_path": _logical_path(candidate_path),
                "input_sha256": candidate_sha,
                "materialized_path": _logical_path(output_path),
                "materialized_sha256": materialized_sha,
                "records": statistics["records"],
                "candidate_rows": statistics["candidate_rows"],
                "gold_free": True,
                "test_accessed": False,
            }
        )

    blockers = source_seal_blockers(provenance_reports, alignment_reports)
    manifest = build_source_manifest(
        provenance_reports=provenance_reports,
        candidate_descriptors=candidate_descriptors,
        alignment_reports=alignment_reports,
        blockers=blockers,
        implementation=implementation,
    )
    _atomic_write_json(manifest_path, manifest)

    total_statistics = {
        "records": sum(item["records"] for item in source_statistics),
        "candidate_rows": sum(item["candidate_rows"] for item in source_statistics),
        "source_formal_rows": sum(
            item["source_formal_rows"] for item in source_statistics
        ),
        "nonformal_rows": sum(item["nonformal_rows"] for item in source_statistics),
        "records_with_nonformal_rows": sum(
            item["records_with_nonformal_rows"] for item in source_statistics
        ),
        "null_region_rows": sum(item["null_region_rows"] for item in source_statistics),
        "real_region_rows": sum(item["real_region_rows"] for item in source_statistics),
    }
    source_counts: dict[str, int] = {}
    for item in source_statistics:
        for source, count in item["candidate_rows_by_source"].items():
            source_counts[source] = source_counts.get(source, 0) + int(count)
    total_statistics["candidate_rows_by_source"] = dict(sorted(source_counts.items()))
    total_statistics["finite_span_scores"] = all(
        item["finite_span_scores"] for item in source_statistics
    )
    total_statistics["finite_fixed_type_scores"] = all(
        item["finite_fixed_type_scores"] for item in source_statistics
    )

    report = {
        "kind": P4_PROVENANCE_REPORT_KIND,
        "format_version": P4_FORMAT_VERSION,
        "phase": "P4.0_source_preparation",
        "status": (
            "SOURCE_MANIFEST_SEALED"
            if manifest["sealed"]
            else "SOURCE_PREPARATION_BLOCKED"
        ),
        "implementation": implementation,
        "preregistration": {
            "path": _logical_path(preregistration),
            "sha256": preregistration_sha,
            "status": preregistration_payload.get("status"),
        },
        "access_contract": {
            "folds_read": list(folds),
            "calibration_folds_opened": False,
            "dev_accessed": False,
            "test_accessed": False,
            "gold_values_used_for_candidate_generation": False,
            "oracle_labels_computed": False,
            "p4_1_code_executed": False,
        },
        "provenance": [
            json_safe_provenance(item)
            for item in sorted(provenance_reports, key=lambda value: value["fold_id"])
        ],
        "candidate_source_statistics": source_statistics,
        "candidate_source_totals": total_statistics,
        "cross_cache_alignment": alignment_reports,
        "source_manifest": {
            "path": _logical_path(manifest_path),
            "sha256": sha256_file(manifest_path),
            "canonical_manifest_sha256": manifest["manifest_sha256"],
            "sealed": manifest["sealed"],
            "status": manifest["status"],
        },
        "seal_blockers": blockers,
        "next_authorized_step": (
            "P4.0_folds_0_7_actionability_audit"
            if not blockers
            else "restore_and_verify_frozen_model_g_formal_span_identity"
        ),
        "p4_1_authorized": False,
    }
    _atomic_write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        default="0-7",
        help="Source-development folds only; calibration folds 8-9 are blocked.",
    )
    parser.add_argument(
        "--full-chain-root",
        type=Path,
        default=PROJECT_ROOT / "knowledge" / "null_release_oof" / "roberta128",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=(
            PROJECT_ROOT / "knowledge" / "p4_actionability_audit" / "source_cache"
        ),
    )
    parser.add_argument(
        "--candidate-name-template",
        default="fold{fold}_stage1_candidates.pt",
    )
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=(PROJECT_ROOT / "knowledge" / "p4_actionability_audit" / "gold_free"),
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "docs"
            / "experiments"
            / "p4_0_source_preparation_report.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT / "docs" / "experiments" / "p4_0_source_manifest.draft.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = parse_p4_development_folds(args.folds)
    report = prepare_sources(
        folds=folds,
        full_chain_root=args.full_chain_root,
        candidate_root=args.candidate_root,
        candidate_name_template=args.candidate_name_template,
        materialized_root=args.materialized_root,
        preregistration=args.preregistration,
        report_path=args.report,
        manifest_path=args.manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
