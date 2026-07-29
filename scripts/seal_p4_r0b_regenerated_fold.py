"""Validate, archive, and clean one P4-R0-B regenerated OOF fold."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    atomic_write_json,
    validate_fold_manifest,
    validate_pipeline_manifest,
)
from gmner.data.null_release_oof_cache import (
    sha256_file,
)
from gmner.data.p4_r0b_regeneration_contract import (
    P4_R0B_ARTIFACT_IDENTITY,
    P4_R0B_EXECUTION_FOLDS,
    P4_R0B_FOLD_REPORT_KIND,
    P4_R0B_M33A_REQUIRED_STAGES,
    P4_R0B_M33A_SUPERVISED_STAGES,
    canonical_formal_triple_digest,
    compare_compact_semantics,
    validate_fold_cleanup_path,
    validate_m33a_formal_oof_payload,
    validate_r0b_preregistration,
    validate_regeneration_metadata,
)


ARCHIVE_SUFFIXES = {".json", ".yaml", ".yml", ".log", ".txt"}


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
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def archive_metadata_files(source: Path, destination: Path) -> list[dict]:
    archived = []
    if not source.exists():
        return archived
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if path.suffix.lower() not in ARCHIVE_SUFFIXES:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        archived.append(
            {
                "source": str(path),
                "archive": str(target),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    return archived


def checkpoint_inventory(pipeline: dict) -> list[dict]:
    inventory = []
    for stage_name, stage in dict(pipeline.get("stages") or {}).items():
        checkpoint = dict(stage.get("checkpoint") or {})
        if checkpoint:
            inventory.append({"stage": stage_name, **checkpoint})
    return inventory


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(args.authorization, root)
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    validate_r0b_preregistration(authorization)
    if args.fold_id not in P4_R0B_EXECUTION_FOLDS:
        raise PermissionError("P4-R0-B sealing is limited to folds 0-7.")

    authorization_sha256 = sha256_file(authorization_path)
    experiment_id = str(authorization["experiment_id"])
    storage = dict(authorization["storage_contract"])
    work_root = resolve(storage["work_root"], root)
    output_root = resolve(storage["output_root"], root)
    legacy_root = resolve(storage["legacy_evidence_root"], root)
    fold_work = work_root / f"fold{args.fold_id}"
    fold_output = output_root / f"fold{args.fold_id}"
    candidate_dir = fold_work / "candidates"
    pipeline_path = fold_work / "pipeline_manifest.json"
    compact_path = fold_work / "m33a_formal_state.pt"
    regenerated_r16_source = candidate_dir / "heldout_r16.pt"
    retained_r16 = fold_work / "regenerated_heldout_r16.pt"
    reference_compact = (
        legacy_root / f"fold{args.fold_id}" / "heldout_features.pt"
    )
    report_path = fold_work / "regeneration_semantic_report.json"
    archive_manifest_path = fold_work / "fold_archive_manifest.json"

    manifest = validate_fold_manifest(
        resolve(args.fold_summary, root),
        expected_num_folds=10,
        verify_fold_ids=P4_R0B_EXECUTION_FOLDS,
    )
    pipeline = validate_pipeline_manifest(
        pipeline_path,
        fold_manifest=manifest,
        fold_id=args.fold_id,
        required_stages=P4_R0B_M33A_REQUIRED_STAGES,
        supervised_stages=P4_R0B_M33A_SUPERVISED_STAGES,
    )
    expected_identity = {
        "artifact_identity": P4_R0B_ARTIFACT_IDENTITY,
        "regeneration_authorization_sha256": authorization_sha256,
        "regeneration_fold_id": args.fold_id,
        "regeneration_experiment_id": experiment_id,
    }
    if dict(pipeline.get("regeneration") or {}) != expected_identity:
        raise ValueError("Pipeline regeneration identity differs from authorization.")

    regenerated_compact = torch.load(compact_path, map_location="cpu")
    reference_payload = torch.load(reference_compact, map_location="cpu")
    r16_payload = torch.load(regenerated_r16_source, map_location="cpu")
    heldout_ids = list(
        next(
            fold["heldout_record_ids"]
            for fold in manifest["folds"]
            if int(fold["fold"]) == args.fold_id
        )
    )
    heldout_ids_sha256 = next(
        fold["heldout_record_ids_sha256"]
        for fold in manifest["folds"]
        if int(fold["fold"]) == args.fold_id
    )
    validate_m33a_formal_oof_payload(
        regenerated_compact,
        expected_fold_id=args.fold_id,
        expected_record_ids=heldout_ids,
    )
    validate_regeneration_metadata(
        dict(regenerated_compact["metadata"]),
        authorization_sha256=authorization_sha256,
        fold_id=args.fold_id,
        experiment_id=experiment_id,
    )
    validate_regeneration_metadata(
        dict(r16_payload["metadata"]),
        authorization_sha256=authorization_sha256,
        fold_id=args.fold_id,
        experiment_id=experiment_id,
    )

    semantic = compare_compact_semantics(
        reference_payload,
        regenerated_compact,
        fold_id=args.fold_id,
        authorization_sha256=authorization_sha256,
        experiment_id=experiment_id,
    )
    formal_digest = canonical_formal_triple_digest(
        r16_payload,
        regenerated_compact,
        fold_id=args.fold_id,
        authorization_sha256=authorization_sha256,
        experiment_id=experiment_id,
    )
    temporary_r16 = retained_r16.with_suffix(".pt.tmp")
    shutil.copy2(regenerated_r16_source, temporary_r16)
    temporary_r16.replace(retained_r16)
    if sha256_file(retained_r16) != sha256_file(regenerated_r16_source):
        raise RuntimeError("Retained regenerated R16 copy changed.")

    archive_root = fold_work / "archive_metadata"
    metadata_archive = []
    metadata_archive.extend(
        archive_metadata_files(candidate_dir, archive_root / "candidates")
    )
    metadata_archive.extend(
        archive_metadata_files(fold_output, archive_root / "training")
    )

    report = {
        "kind": P4_R0B_FOLD_REPORT_KIND,
        "format_version": 1,
        "status": "SEMANTIC_GATE_EVALUATED",
        **expected_identity,
        "fold_id": args.fold_id,
        "records": len(heldout_ids),
        "heldout_record_ids_sha256": heldout_ids_sha256,
        "semantic_consistency": semantic,
        "formal_coordinates_valid": True,
        "canonical_formal_predictions": formal_digest,
        "reference_compact": {
            "path": str(reference_compact),
            "sha256": sha256_file(reference_compact),
        },
        "regenerated_m33a_formal_state": {
            "path": str(compact_path),
            "sha256": sha256_file(compact_path),
        },
        "regenerated_r16": {
            "path": str(retained_r16),
            "sha256": sha256_file(retained_r16),
        },
        "fold_semantic_gate_passed": bool(semantic["gate_passed"]),
        "formal_sidecar_generated": False,
        "attached_to_p4": False,
        "folds_8_9_accessed": False,
        "p4_dev_accessed": False,
        "siglip2_included": False,
        "reliability_included": False,
        "null_release_included": False,
        "oracle_run": False,
        "p4_1_run": False,
        "test_accessed": False,
    }
    atomic_write_json(report_path, report)

    bytes_to_delete = sum(
        directory_bytes(path)
        for path in (candidate_dir, fold_output)
    )
    archive_manifest = {
        "kind": "p4_r0_b_regenerated_fold_archive",
        "format_version": 1,
        "status": "SEALED",
        **expected_identity,
        "fold_id": args.fold_id,
        "records": len(heldout_ids),
        "pipeline_manifest": {
            "path": str(pipeline_path),
            "sha256": sha256_file(pipeline_path),
        },
        "checkpoint_inventory": checkpoint_inventory(pipeline),
        "metadata_archive": metadata_archive,
        "semantic_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
        "retained_artifacts": {
            "m33a_formal_state": {
                "path": str(compact_path),
                "sha256": sha256_file(compact_path),
            },
            "regenerated_r16": {
                "path": str(retained_r16),
                "sha256": sha256_file(retained_r16),
            },
        },
        "bytes_to_delete": bytes_to_delete,
        "cleanup_requested": bool(args.cleanup),
        "test_accessed": False,
    }
    atomic_write_json(archive_manifest_path, archive_manifest)

    if args.cleanup:
        cleanup_targets = (
            validate_fold_cleanup_path(
                candidate_dir,
                allowed_root=work_root,
                fold_id=args.fold_id,
            ),
            validate_fold_cleanup_path(
                fold_output,
                allowed_root=output_root,
                fold_id=args.fold_id,
            ),
        )
        for target in cleanup_targets:
            if target.exists():
                shutil.rmtree(target)

    reloaded_compact = torch.load(compact_path, map_location="cpu")
    reloaded_r16 = torch.load(retained_r16, map_location="cpu")
    validate_m33a_formal_oof_payload(
        reloaded_compact,
        expected_fold_id=args.fold_id,
        expected_record_ids=heldout_ids,
    )
    validate_regeneration_metadata(
        dict(reloaded_compact["metadata"]),
        authorization_sha256=authorization_sha256,
        fold_id=args.fold_id,
        experiment_id=experiment_id,
    )
    validate_regeneration_metadata(
        dict(reloaded_r16["metadata"]),
        authorization_sha256=authorization_sha256,
        fold_id=args.fold_id,
        experiment_id=experiment_id,
    )
    archive_manifest["status"] = "CLEANED" if args.cleanup else "SEALED"
    archive_manifest["post_cleanup_reload_passed"] = True
    archive_manifest["retained_bytes"] = directory_bytes(fold_work)
    archive_manifest["test_accessed"] = False
    atomic_write_json(archive_manifest_path, archive_manifest)
    print(json.dumps(archive_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
