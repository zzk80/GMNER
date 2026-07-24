"""Create the auditable sidecar for one full-chain NULL Release OOF fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    fold_from_manifest,
    validate_fold_manifest,
    validate_pipeline_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file
from scripts.build_null_release_oof_features import _artifact_paths
from scripts.train_fine_grounding_adapter import resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold-summary", required=True)
    parser.add_argument("--pipeline-manifest", required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.fold_id not in range(10):
        raise ValueError("Formal NULL Release OOF fold id must be in 0..9.")
    summary_path = resolve(args.fold_summary, root)
    summary = validate_fold_manifest(summary_path, expected_num_folds=10)
    fold = fold_from_manifest(summary, args.fold_id)
    train_path = resolve(fold["train_file"], root)
    heldout_path = resolve(fold["heldout_file"], root)
    train_ids = list(fold["train_record_ids"])
    heldout_ids = list(fold["heldout_record_ids"])
    pipeline_path = resolve(args.pipeline_manifest, root)
    pipeline = validate_pipeline_manifest(
        pipeline_path,
        fold_manifest=summary,
        fold_id=args.fold_id,
    )
    if pipeline.get("fold_manifest_sha256") != sha256_file(summary_path):
        raise ValueError("Pipeline manifest references another fold manifest.")

    config_path = resolve(args.config, root)
    artifacts = _artifact_paths(config_path, root)
    artifact_hashes = {}
    for name, path in artifacts.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing full-chain artifact {name}: {path}")
        artifact_hashes[name] = sha256_file(path)
    proof = {
        "format_version": 1,
        "kind": "null_release_full_chain_fold_proof",
        "fold_id": args.fold_id,
        "num_folds": 10,
        "excluded_heldout": True,
        "fold_summary": str(summary_path.resolve()),
        "fold_summary_sha256": sha256_file(summary_path),
        "pipeline_manifest": str(pipeline_path.resolve()),
        "pipeline_manifest_sha256": sha256_file(pipeline_path),
        "train_file": str(train_path.resolve()),
        "train_file_sha256": sha256_file(train_path),
        "heldout_file": str(heldout_path.resolve()),
        "heldout_file_sha256": sha256_file(heldout_path),
        "training_record_ids": train_ids,
        "heldout_record_ids": heldout_ids,
        "artifact_sha256": artifact_hashes,
    }
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fold_id": args.fold_id,
                "training_records": len(train_ids),
                "heldout_records": len(heldout_ids),
                "artifacts": len(artifact_hashes),
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
