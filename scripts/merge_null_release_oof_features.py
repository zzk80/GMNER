"""Merge ten held-out full-chain caches for formal NULL Release training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.null_release_oof_cache import (
    NULL_RELEASE_OOF_FORMAT_VERSION,
    NULL_RELEASE_OOF_KIND,
    sha256_file,
    stable_id_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-records", type=int, default=7000)
    return parser.parse_args()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    inputs = [resolve(value, root) for value in args.inputs]
    if len(inputs) != 10:
        raise ValueError(f"Formal NULL Release OOF requires 10 inputs, got {len(inputs)}.")

    folds = []
    for path in inputs:
        payload = torch.load(path, map_location="cpu")
        metadata = dict(payload.get("metadata") or {})
        if metadata.get("kind") != NULL_RELEASE_OOF_KIND:
            raise ValueError(f"Invalid fold feature cache: {path}")
        if int(metadata.get("format_version", -1)) != NULL_RELEASE_OOF_FORMAT_VERSION:
            raise ValueError(f"Unsupported fold feature cache: {path}")
        if not bool(metadata.get("full_chain_oof")) or not bool(
            metadata.get("excluded_heldout")
        ):
            raise ValueError(f"Fold cache lacks strict OOF provenance: {path}")
        folds.append((path, payload, metadata))

    fold_ids = sorted(int(metadata["fold_id"]) for _, _, metadata in folds)
    if fold_ids != list(range(10)):
        raise ValueError(f"Expected fold ids 0..9, found {fold_ids}.")
    heldout_sets = {
        int(metadata["fold_id"]): set(map(str, metadata["heldout_record_ids"]))
        for _, _, metadata in folds
    }
    global_ids = set().union(*heldout_sets.values())
    heldout_total = sum(len(values) for values in heldout_sets.values())
    if heldout_total != len(global_ids):
        raise ValueError("Held-out record ids overlap across folds.")
    if len(global_ids) != int(args.expected_records):
        raise ValueError(
            f"Expected {args.expected_records} unique records, found {len(global_ids)}."
        )
    for _, _, metadata in folds:
        fold_id = int(metadata["fold_id"])
        train_ids = set(map(str, metadata["training_record_ids"]))
        expected_train = global_ids - heldout_sets[fold_id]
        if train_ids != expected_train:
            raise ValueError(
                f"Fold {fold_id} training ids are not exactly all non-held-out records."
            )

    batches = []
    batch_record_ids: list[str] = []
    fold_summaries = []
    for path, payload, metadata in sorted(
        folds, key=lambda item: int(item[2]["fold_id"])
    ):
        local_ids = []
        for batch in payload["batches"]:
            if int(batch["fold_id"]) != int(metadata["fold_id"]):
                raise ValueError(f"Batch/fold mismatch in {path}.")
            local_ids.extend(map(str, batch["record_ids"]))
            batches.append(batch)
        if set(local_ids) != heldout_sets[int(metadata["fold_id"])]:
            raise ValueError(f"Cached batch ids do not match proof for {path}.")
        batch_record_ids.extend(local_ids)
        fold_summaries.append(
            {
                "fold_id": int(metadata["fold_id"]),
                "records": len(local_ids),
                "cache": str(path.resolve()),
                "cache_sha256": sha256_file(path),
                "fold_proof_sha256": metadata["fold_proof_sha256"],
                "artifact_sha256": metadata["artifact_sha256"],
            }
        )
    if len(batch_record_ids) != len(set(batch_record_ids)):
        raise ValueError("Merged feature batches contain duplicate record ids.")

    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = {
        "metadata": {
            "format_version": NULL_RELEASE_OOF_FORMAT_VERSION,
            "kind": NULL_RELEASE_OOF_KIND,
            "full_chain_oof": True,
            "num_folds": 10,
            "fold_ids": list(range(10)),
            "records": len(batch_record_ids),
            "record_ids_sha256": stable_id_digest(batch_record_ids),
            "includes_reliability": all(
                bool(metadata.get("includes_reliability"))
                for _, _, metadata in folds
            ),
            "folds": fold_summaries,
        },
        "batches": batches,
    }
    torch.save(merged, output)
    print(
        json.dumps(
            {
                "folds": 10,
                "records": len(batch_record_ids),
                "batches": len(batches),
                "output": str(output.resolve()),
                "sha256": sha256_file(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
