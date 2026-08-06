#!/usr/bin/env python3
"""Validate, archive, and clean one completed final-chain OOF population fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from gmner.data.artifact_utils import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retained-target-mib", type=int, default=500)
    return parser.parse_args()


def bytes_under(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def descriptor(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def ensure_under(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Cleanup path is outside authorized root: {path}") from error


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if run_dir.name != f"fold{args.fold_id}" or output_dir.name != f"fold{args.fold_id}":
        raise ValueError("Fold cleanup directory identity mismatch.")
    materialization = json.loads((run_dir / f"fold{args.fold_id}_materialization_report.json").read_text())
    completion = json.loads((run_dir / f"fold{args.fold_id}_completion_audit.json").read_text())
    supervision = json.loads((run_dir / f"fold{args.fold_id}_supervision_audit.json").read_text())
    resources = json.loads((run_dir / f"fold{args.fold_id}_resource_summary.json").read_text())
    pipeline = json.loads((run_dir / "pipeline_manifest.json").read_text())
    rows = run_dir / "final_chain_oof_rows.jsonl"
    sidecar = run_dir / f"fold{args.fold_id}_supervision.jsonl"
    if (
        materialization.get("status") != "PASSED"
        or completion.get("status") != "PASSED"
        or supervision.get("sealed_rows_unchanged") is not True
        or resources.get("status") != "COMPLETE"
        or pipeline.get("sealed") is not True
        or materialization.get("rows_sha256") != sha256_file(rows)
        or supervision.get("sidecar_sha256") != sha256_file(sidecar)
    ):
        raise RuntimeError("Fold cannot be cleaned before every Gate passes.")

    stage_artifacts = run_dir / "stage_artifacts"
    if output_dir.exists():
        for path in output_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".log", ".txt"}:
                target = stage_artifacts / path.relative_to(output_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    before = bytes_under(run_dir) + bytes_under(output_dir)
    delete_paths = [
        run_dir / "candidates",
        run_dir / "fit_train.jsonl",
        run_dir / "checkpoint_selection.jsonl",
        run_dir / "m33a_formal_state.pt",
        output_dir,
    ]
    for path in delete_paths:
        ensure_under(path, run_dir.parent if path != output_dir else output_dir.parent)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    if json.loads(rows.read_text(encoding="utf-8").splitlines()[0])["fold_id"] != args.fold_id:
        raise RuntimeError("Rows failed post-cleanup reload.")
    if json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])["fold_id"] != args.fold_id:
        raise RuntimeError("Supervision sidecar failed post-cleanup reload.")
    retained_files = [path for path in run_dir.rglob("*") if path.is_file()]
    retained = bytes_under(run_dir)
    if retained > int(args.retained_target_mib) * 1024**2:
        raise RuntimeError("Retained fold exceeds the preregistered target.")
    report = {
        "kind": "final_chain_oof_population_fold_archive",
        "format_version": 1,
        "status": "CLEANED",
        "fold_id": args.fold_id,
        "bytes_before_cleanup": before,
        "retained_bytes": retained,
        "deleted_bytes": before - retained,
        "retained_files": [descriptor(path) for path in sorted(retained_files)],
        "rows_sha256": sha256_file(rows),
        "sidecar_sha256": sha256_file(sidecar),
        "test_accessed": False,
    }
    (run_dir / "fold_archive_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
