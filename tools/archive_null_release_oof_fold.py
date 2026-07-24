"""Validate, archive, and remove rebuildable artifacts from one sealed OOF fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmner.data.full_chain_oof_contract import (  # noqa: E402
    SUPERVISED_PIPELINE_STAGES,
    fold_from_manifest,
    validate_fold_manifest,
    validate_pipeline_manifest,
)
from gmner.data.null_release_oof_cache import (  # noqa: E402
    sha256_file,
    validate_fold_oof_payload,
)


ARCHIVE_KIND = "null_release_oof_fold_archive"
ARCHIVE_VERSION = 1
REPORT_NAMES = {
    "metrics.json",
    "resolved_config.yaml",
    "train.log",
    "train_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-work", required=True)
    parser.add_argument("--fold-id", required=True, type=int)
    parser.add_argument("--output-work-root")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument(
        "--extra-retain-file",
        action="append",
        default=[],
        help="Additional log or report copied into the fold archive.",
    )
    parser.add_argument(
        "--max-retained-mb",
        type=float,
        default=500.0,
        help="Abort before deletion if estimated retained fold data exceeds this size.",
    )
    parser.add_argument(
        "--checkpoint-backup-note",
        default="",
        help="Audit note identifying an external checkpoint backup, when present.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform archival and deletion. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = root / value
    return value.resolve()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _require_descendant(path: Path, parent: Path, label: str) -> None:
    if path == parent or parent not in path.parents:
        raise ValueError(f"{label} must be a strict descendant of {parent}: {path}")


def _validate_roots(
    project_root: Path,
    fold_work: Path,
    output_work_root: Path,
    fold_id: int,
) -> None:
    if fold_id not in range(10):
        raise ValueError("Formal NULL Release OOF fold id must be in 0..9.")
    _require_descendant(fold_work, project_root / "knowledge", "fold work")
    _require_descendant(
        output_work_root, project_root / "outputs", "fold output work root"
    )
    expected_name = f"fold{fold_id}"
    if fold_work.name != expected_name or output_work_root.name != expected_name:
        raise ValueError(
            f"Fold roots must end in {expected_name!r}: "
            f"{fold_work}, {output_work_root}"
        )


def _descriptor_paths(pipeline: dict[str, Any]) -> set[Path]:
    paths: set[Path] = set()
    for stage in dict(pipeline.get("stages") or {}).values():
        for role in ("config", "checkpoint"):
            descriptor = dict(stage.get(role) or {})
            if descriptor.get("path"):
                paths.add(Path(str(descriptor["path"])).resolve())
        for role in ("inputs", "outputs"):
            for descriptor in list(stage.get(role) or []):
                if descriptor.get("path"):
                    paths.add(Path(str(descriptor["path"])).resolve())
    return paths


def _assert_self_contained(value: Any, location: str = "payload") -> None:
    if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise ValueError(
                    f"OOF cache contains a non-primitive key at {location}: {type(key)}"
                )
            _assert_self_contained(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_self_contained(item, f"{location}[{index}]")
        return
    raise ValueError(
        f"OOF cache is not self-contained at {location}: {type(value).__name__}"
    )


def _load_and_validate_features(
    feature_path: Path,
    *,
    fold_id: int,
    expected_record_ids: list[str],
    proof_path: Path,
    proof: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(feature_path, map_location="cpu")
    validated = validate_fold_oof_payload(
        payload,
        expected_fold_id=fold_id,
        expected_record_ids=expected_record_ids,
        require_reliability=True,
    )
    _assert_self_contained(payload)
    metadata = dict(validated["metadata"])
    if metadata.get("excluded_heldout") is not True:
        raise ValueError("Held-out feature cache lacks excluded_heldout=true.")
    if metadata.get("fold_proof_sha256") != sha256_file(proof_path):
        raise ValueError("Held-out feature cache references another fold proof.")
    if dict(metadata.get("artifact_sha256") or {}) != dict(
        proof.get("artifact_sha256") or {}
    ):
        raise ValueError("Feature and fold-proof artifact hashes differ.")
    training_ids = [str(value) for value in metadata.get("training_record_ids") or []]
    heldout_ids = [str(value) for value in metadata.get("heldout_record_ids") or []]
    if heldout_ids != expected_record_ids:
        raise ValueError("Feature metadata held-out ids differ from the fold manifest.")
    if set(training_ids) & set(heldout_ids):
        raise ValueError("Feature metadata train and held-out ids overlap.")
    return payload, validated


def _validate_proof(
    proof_path: Path,
    *,
    fold_manifest_path: Path,
    fold_manifest: dict[str, Any],
    pipeline_path: Path,
    pipeline: dict[str, Any],
    fold_id: int,
    fold_work: Path,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("kind") != "null_release_full_chain_fold_proof":
        raise ValueError("Not a NULL Release full-chain fold proof.")
    if int(proof.get("format_version", -1)) != 1:
        raise ValueError("Unsupported NULL Release fold-proof version.")
    if int(proof.get("fold_id", -1)) != fold_id:
        raise ValueError("Fold proof has the wrong fold id.")
    if int(proof.get("num_folds", -1)) != 10:
        raise ValueError("Fold proof must declare exactly 10 folds.")
    if proof.get("excluded_heldout") is not True:
        raise ValueError("Fold proof lacks excluded_heldout=true.")
    if Path(str(proof.get("fold_summary", ""))).resolve() != fold_manifest_path:
        raise ValueError("Fold proof references another fold manifest.")
    if proof.get("fold_summary_sha256") != sha256_file(fold_manifest_path):
        raise ValueError("Fold proof manifest hash changed.")
    if Path(str(proof.get("pipeline_manifest", ""))).resolve() != pipeline_path:
        raise ValueError("Fold proof references another pipeline manifest.")
    if proof.get("pipeline_manifest_sha256") != sha256_file(pipeline_path):
        raise ValueError("Fold proof pipeline hash changed.")

    fold = fold_from_manifest(fold_manifest, fold_id)
    for role in ("train", "heldout"):
        actual_path = Path(str(proof.get(f"{role}_file", ""))).resolve()
        expected_path = Path(str(fold[f"{role}_file"])).resolve()
        if actual_path != expected_path:
            raise ValueError(f"Fold proof references another {role} file.")
        if proof.get(f"{role}_file_sha256") != sha256_file(expected_path):
            raise ValueError(f"Fold proof {role} file hash changed.")
        id_key = "training_record_ids" if role == "train" else "heldout_record_ids"
        proof_ids = [str(value) for value in proof.get(id_key, [])]
        expected_ids = [str(value) for value in fold[f"{role}_record_ids"]]
        if proof_ids != expected_ids:
            raise ValueError(f"Fold proof {role} record ids changed.")

    candidates = _descriptor_paths(pipeline)
    candidates.update(
        path.resolve()
        for path in (fold_work / "configs").rglob("*")
        if path.is_file()
    )
    hash_to_paths: dict[str, list[str]] = {}
    for path in sorted(candidates):
        if path.is_file():
            hash_to_paths.setdefault(sha256_file(path), []).append(str(path))
    artifact_matches: dict[str, list[str]] = {}
    artifact_hashes = dict(proof.get("artifact_sha256") or {})
    if not artifact_hashes:
        raise ValueError("Fold proof has no artifact hash inventory.")
    for name, digest in artifact_hashes.items():
        matches = hash_to_paths.get(str(digest), [])
        if not matches:
            raise ValueError(
                f"Fold proof artifact {name!r} has no matching source file."
            )
        artifact_matches[str(name)] = matches
    return proof, artifact_matches


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _tree_inventory(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {
            "path": str(root),
            "exists": False,
            "files": 0,
            "bytes": 0,
            "tree_sha256": None,
        }
    if root.is_symlink():
        raise ValueError(f"Refusing to archive or delete a symlinked root: {root}")
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_hash))
        count += 1
        total += size
    return {
        "path": str(root.resolve()),
        "exists": True,
        "files": count,
        "bytes": total,
        "tree_sha256": digest.hexdigest(),
    }


def _discover_reports(
    project_root: Path,
    output_work_root: Path,
    fold_id: int,
    extra_files: Iterable[Path],
) -> list[tuple[Path, Path]]:
    reports: list[tuple[Path, Path]] = []
    for path in sorted(output_work_root.rglob("*")):
        if path.is_file() and path.name in REPORT_NAMES:
            reports.append(
                (path, Path("outputs") / path.relative_to(output_work_root))
            )
    for path in sorted(project_root.glob(f"null_release_oof_fold{fold_id}*.log")):
        if path.is_file():
            reports.append((path, Path("pipeline_logs") / path.name))
    for path in extra_files:
        if not path.is_file():
            raise FileNotFoundError(f"Extra retained file does not exist: {path}")
        reports.append((path, Path("extra") / path.name))

    unique: dict[tuple[str, str], tuple[Path, Path]] = {}
    for source, relative in reports:
        unique[(str(source.resolve()), relative.as_posix())] = (source, relative)
    return list(unique.values())


def _copy_reports(
    reports: list[tuple[Path, Path]], archive_dir: Path
) -> list[dict[str, Any]]:
    copied = []
    for source, relative in reports:
        destination = archive_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        if sha256_file(destination) != source_hash:
            raise RuntimeError(f"Archived report hash mismatch: {source}")
        copied.append(
            {
                "source": str(source.resolve()),
                "archive": str(destination.resolve()),
                "bytes": source.stat().st_size,
                "sha256": source_hash,
            }
        )
    return copied


def _checkpoint_inventory(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for name in SUPERVISED_PIPELINE_STAGES:
        descriptor = dict(
            dict(pipeline.get("stages") or {}).get(name, {}).get("checkpoint") or {}
        )
        if descriptor:
            path = Path(str(descriptor["path"])).resolve()
            result.append(
                {
                    "stage": name,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": descriptor["sha256"],
                }
            )
    return result


def _fold_bytes(fold_work: Path) -> int:
    return sum(path.stat().st_size for path in fold_work.rglob("*") if path.is_file())


def _post_cleanup_validate(
    manifest: dict[str, Any],
    *,
    fold_work: Path,
    fold_id: int,
) -> dict[str, Any]:
    feature_path = Path(manifest["heldout_features"]["path"])
    proof_path = Path(manifest["fold_proof"]["path"])
    pipeline_path = Path(manifest["pipeline_manifest"]["path"])
    for name, path, expected_hash in (
        ("heldout features", feature_path, manifest["heldout_features"]["sha256"]),
        ("fold proof", proof_path, manifest["fold_proof"]["sha256"]),
        (
            "pipeline manifest",
            pipeline_path,
            manifest["pipeline_manifest"]["sha256"],
        ),
    ):
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Retained {name} is missing or changed: {path}")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    expected_ids = [str(value) for value in proof["heldout_record_ids"]]
    _, validated = _load_and_validate_features(
        feature_path,
        fold_id=fold_id,
        expected_record_ids=expected_ids,
        proof_path=proof_path,
        proof=proof,
    )
    for item in manifest["deletion_roots"]:
        if Path(item["path"]).exists():
            raise ValueError(f"Cleanup root still exists: {item['path']}")
    retained_bytes = _fold_bytes(fold_work)
    max_bytes = int(float(manifest["max_retained_mb"]) * 1024 * 1024)
    if retained_bytes > max_bytes:
        raise ValueError(
            f"Retained fold size {retained_bytes} exceeds limit {max_bytes}."
        )
    return {
        "records": validated["records"],
        "batches": len(validated["batches"]),
        "retained_bytes": retained_bytes,
        "self_contained_reload": True,
        "test_accessed": False,
    }


def archive_fold(
    *,
    project_root: Path,
    fold_work: Path,
    fold_id: int,
    output_work_root: Path | None,
    execute: bool,
    max_retained_mb: float = 500.0,
    extra_retain_files: Iterable[Path] = (),
    checkpoint_backup_note: str = "",
) -> dict[str, Any]:
    project_root = project_root.resolve()
    fold_work = fold_work.resolve()
    pipeline_path = fold_work / "pipeline_manifest.json"
    proof_path = fold_work / "fold_proof.json"
    feature_path = fold_work / "heldout_features.pt"
    archive_manifest_path = fold_work / "fold_archive_manifest.json"
    archive_dir = fold_work / "archive"

    if archive_manifest_path.is_file():
        existing = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        if existing.get("kind") != ARCHIVE_KIND or int(
            existing.get("fold_id", -1)
        ) != fold_id:
            raise ValueError("Existing archive manifest belongs to another contract.")
        if existing.get("status") == "cleaned":
            post = _post_cleanup_validate(
                existing, fold_work=fold_work, fold_id=fold_id
            )
            existing["post_cleanup_validation"] = post
            return existing
        if existing.get("status") not in {"prepared", "cleaning"}:
            raise ValueError(
                f"Cannot resume archive status {existing.get('status')!r}."
            )
        if not execute:
            return existing
        manifest = existing
    else:
        if not pipeline_path.is_file() or not proof_path.is_file():
            raise FileNotFoundError("Fold pipeline manifest or proof is missing.")
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        if output_work_root is None:
            roots = set()
            for name in SUPERVISED_PIPELINE_STAGES:
                descriptor = dict(
                    dict(pipeline.get("stages") or {})
                    .get(name, {})
                    .get("checkpoint")
                    or {}
                )
                checkpoint_path = str(descriptor.get("path", ""))
                if not checkpoint_path:
                    raise ValueError(
                        f"Cannot infer output root; {name} checkpoint is missing."
                    )
                roots.add(Path(checkpoint_path).resolve().parent.parent)
            if len(roots) != 1:
                raise ValueError(
                    "Pass --output-work-root; checkpoint paths do not share one root."
                )
            output_work_root = roots.pop()
        output_work_root = output_work_root.resolve()
        _validate_roots(project_root, fold_work, output_work_root, fold_id)

        fold_manifest_path = Path(str(pipeline.get("fold_manifest", ""))).resolve()
        fold_manifest = validate_fold_manifest(
            fold_manifest_path, expected_num_folds=10
        )
        validate_pipeline_manifest(
            pipeline_path, fold_manifest=fold_manifest, fold_id=fold_id
        )
        if pipeline.get("sealed") is not True:
            raise ValueError("Fold pipeline must be sealed before archival.")
        if pipeline.get("test_accessed") is not False:
            raise ValueError("Fold pipeline accessed test data.")
        proof, artifact_matches = _validate_proof(
            proof_path,
            fold_manifest_path=fold_manifest_path,
            fold_manifest=fold_manifest,
            pipeline_path=pipeline_path,
            pipeline=pipeline,
            fold_id=fold_id,
            fold_work=fold_work,
        )
        fold = fold_from_manifest(fold_manifest, fold_id)
        _, validated = _load_and_validate_features(
            feature_path,
            fold_id=fold_id,
            expected_record_ids=[
                str(value) for value in fold["heldout_record_ids"]
            ],
            proof_path=proof_path,
            proof=proof,
        )

        deletion_paths = [
            fold_work / "candidates",
            fold_work / "siglip2",
            output_work_root,
        ]
        for path in deletion_paths[:2]:
            _require_descendant(path.resolve(), fold_work, "fold cleanup root")
        _require_descendant(
            output_work_root, project_root / "outputs", "output cleanup root"
        )
        reports = _discover_reports(
            project_root,
            output_work_root,
            fold_id,
            [path.resolve() for path in extra_retain_files],
        )
        report_bytes = sum(source.stat().st_size for source, _ in reports)
        retained_now = sum(
            path.stat().st_size
            for path in fold_work.rglob("*")
            if path.is_file()
            and not any(root in path.parents for root in deletion_paths[:2])
        )
        estimated_retained = retained_now + report_bytes + 1024 * 1024
        max_retained_bytes = int(max_retained_mb * 1024 * 1024)
        if estimated_retained > max_retained_bytes:
            raise ValueError(
                "Estimated retained fold size exceeds the configured limit: "
                f"{estimated_retained} > {max_retained_bytes} bytes."
            )
        deletion_roots = [_tree_inventory(path) for path in deletion_paths]
        dry_run_result = {
            "kind": ARCHIVE_KIND,
            "format_version": ARCHIVE_VERSION,
            "status": "dry_run",
            "fold_id": fold_id,
            "records": validated["records"],
            "batches": len(validated["batches"]),
            "pipeline_sealed": True,
            "test_accessed": False,
            "heldout_features": _file_descriptor(feature_path),
            "fold_proof": _file_descriptor(proof_path),
            "pipeline_manifest": _file_descriptor(pipeline_path),
            "checkpoint_artifacts": _checkpoint_inventory(pipeline),
            "proof_artifact_sha256": dict(proof["artifact_sha256"]),
            "proof_artifact_matches": artifact_matches,
            "reports_to_archive": [
                {
                    "source": str(source.resolve()),
                    "archive_relative": relative.as_posix(),
                    "bytes": source.stat().st_size,
                }
                for source, relative in reports
            ],
            "deletion_roots": deletion_roots,
            "bytes_to_delete": sum(item["bytes"] for item in deletion_roots),
            "estimated_retained_bytes": estimated_retained,
            "max_retained_mb": max_retained_mb,
            "checkpoint_backup_note": checkpoint_backup_note,
        }
        if not execute:
            return dry_run_result

        copied_reports = _copy_reports(reports, archive_dir)
        checksum_path = fold_work / "heldout_features.pt.sha256"
        checksum_path.write_text(
            f"{dry_run_result['heldout_features']['sha256']}  "
            f"{feature_path.name}\n",
            encoding="ascii",
        )
        retained_files = [
            _file_descriptor(path)
            for path in sorted(
                item
                for item in fold_work.rglob("*")
                if item.is_file()
                and not any(root in item.parents for root in deletion_paths[:2])
                and item != archive_manifest_path
            )
        ]
        manifest = {
            **dry_run_result,
            "status": "prepared",
            "prepared_at_utc": _utc_now(),
            "pre_cleanup_validation": {
                "all_required_stages_complete": True,
                "pipeline_sealed": True,
                "test_accessed": False,
                "fold_records": validated["records"],
                "fixed_top4_valid": True,
                "self_contained_payload": True,
                "artifact_hashes_verified": True,
            },
            "archived_reports": copied_reports,
            "retained_files_before_cleanup": retained_files,
            "deleted_roots_completed": [],
        }
        _atomic_write_json(archive_manifest_path, manifest)

    manifest["status"] = "cleaning"
    _atomic_write_json(archive_manifest_path, manifest)
    completed = set(manifest.get("deleted_roots_completed") or [])
    for item in manifest["deletion_roots"]:
        path = Path(item["path"])
        path_value = str(path)
        if path_value not in completed:
            if path.exists():
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(f"Refusing to recursively delete unsafe path: {path}")
                shutil.rmtree(path)
            completed.add(path_value)
            manifest["deleted_roots_completed"] = sorted(completed)
            _atomic_write_json(archive_manifest_path, manifest)

    post = _post_cleanup_validate(manifest, fold_work=fold_work, fold_id=fold_id)
    manifest["status"] = "cleaned"
    manifest["completed_at_utc"] = _utc_now()
    manifest["post_cleanup_validation"] = post
    manifest["actual_retained_bytes"] = post["retained_bytes"]
    manifest["actual_deleted_bytes"] = sum(
        int(item["bytes"]) for item in manifest["deletion_roots"]
    )
    _atomic_write_json(archive_manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    project_root = _resolve(args.project_root, ROOT)
    fold_work = _resolve(args.fold_work, project_root)
    output_work_root = (
        _resolve(args.output_work_root, project_root)
        if args.output_work_root
        else None
    )
    result = archive_fold(
        project_root=project_root,
        fold_work=fold_work,
        fold_id=args.fold_id,
        output_work_root=output_work_root,
        execute=args.execute,
        max_retained_mb=args.max_retained_mb,
        extra_retain_files=[
            _resolve(path, project_root) for path in args.extra_retain_file
        ],
        checkpoint_backup_note=args.checkpoint_backup_note,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "fold_id": result["fold_id"],
                "records": result.get("records"),
                "bytes_to_delete": result.get("bytes_to_delete"),
                "retained_bytes": result.get(
                    "actual_retained_bytes",
                    result.get("estimated_retained_bytes"),
                ),
                "test_accessed": result.get("test_accessed"),
                "archive_manifest": str(
                    (fold_work / "fold_archive_manifest.json").resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
