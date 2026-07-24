"""Create out-of-fold train/heldout files for predicted evidence training."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.data.full_chain_oof_contract import (
    FULL_CHAIN_FOLD_MANIFEST_KIND,
    FULL_CHAIN_FOLD_MANIFEST_VERSION,
    atomic_write_json,
    record_id,
    source_tree_sha256,
    validate_fold_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from gmner.utils.io import maybe_convert_conll, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build K-fold files from the configured train split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def main() -> None:
    args = parse_args()
    if args.num_folds < 2:
        raise ValueError("--num-folds must be at least 2.")
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    output_dir = resolve_path(args.output_dir, project_root)
    summary_path = output_dir / "fold_summary.json"
    if summary_path.exists() and not args.force:
        existing = validate_fold_manifest(
            summary_path,
            expected_num_folds=args.num_folds,
        )
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = maybe_convert_conll(
        resolve_path(config.data.train_file, project_root),
        output_dir / "_source_cache",
    )
    records = read_jsonl(source_path)
    record_ids = [record_id(record) for record in records]
    if any(not value for value in record_ids):
        raise ValueError("Training source contains records without ids.")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Training source contains duplicate record ids.")
    indices = list(range(len(records)))
    random.Random(args.seed).shuffle(indices)
    folds = [indices[idx:: args.num_folds] for idx in range(args.num_folds)]

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = None
    config_path = resolve_path(args.config, project_root)
    summary = {
        "format_version": FULL_CHAIN_FOLD_MANIFEST_VERSION,
        "kind": FULL_CHAIN_FOLD_MANIFEST_KIND,
        "source_split": "train",
        "test_accessed": False,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git_commit": git_commit,
        "source_tree_sha256": source_tree_sha256(project_root),
        "records": len(records),
        "record_ids": record_ids,
        "record_ids_sha256": stable_id_digest(record_ids),
        "num_folds": args.num_folds,
        "seed": args.seed,
        "folds": [],
    }
    for fold_id, heldout_indices in enumerate(folds):
        heldout_set = set(heldout_indices)
        train_records = [records[idx] for idx in indices if idx not in heldout_set]
        heldout_records = [records[idx] for idx in heldout_indices]
        train_path = output_dir / f"train_fold{fold_id}.jsonl"
        heldout_path = output_dir / f"heldout_fold{fold_id}.jsonl"
        write_jsonl(train_path, train_records)
        write_jsonl(heldout_path, heldout_records)
        train_ids = [record_id(record) for record in train_records]
        heldout_ids = [record_id(record) for record in heldout_records]
        summary["folds"].append(
            {
                "fold": fold_id,
                "train_file": str(train_path.resolve()),
                "train_file_sha256": sha256_file(train_path),
                "heldout_file": str(heldout_path.resolve()),
                "heldout_file_sha256": sha256_file(heldout_path),
                "train_records": len(train_records),
                "heldout_records": len(heldout_records),
                "train_record_ids": train_ids,
                "heldout_record_ids": heldout_ids,
                "train_record_ids_sha256": stable_id_digest(train_ids),
                "heldout_record_ids_sha256": stable_id_digest(heldout_ids),
            }
        )

    atomic_write_json(summary_path, summary)
    validate_fold_manifest(summary_path, expected_num_folds=args.num_folds)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
