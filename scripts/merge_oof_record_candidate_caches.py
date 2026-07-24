"""Merge held-out Stage1 candidate caches into one OOF train cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.hierarchical_record_candidate_collator import (
    missing_hierarchical_cache_fields,
)
from gmner.data.record_candidate_dataset import CACHE_FORMAT_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-records", type=int, default=None)
    return parser.parse_args()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def stable_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _f1(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    score = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": score}


def _match_counts(predictions: list[dict], gold: list[dict]) -> dict[str, int]:
    matched = {name: set() for name in ("span", "mner", "eeg", "gmner")}
    for prediction in predictions:
        span = tuple(prediction["span"])
        for name in matched:
            for index, target in enumerate(gold):
                if index in matched[name] or tuple(target["span"]) != span:
                    continue
                type_ok = int(prediction["type_id"]) == int(target["type_id"])
                region_ok = int(prediction["region_index"]) in set(
                    target.get("region_positive_indices") or []
                )
                if (
                    name == "span"
                    or (name == "mner" and type_ok)
                    or (name == "eeg" and region_ok)
                    or (name == "gmner" and type_ok and region_ok)
                ):
                    matched[name].add(index)
                    break
    return {name: len(indices) for name, indices in matched.items()}


def _record_sort_key(record: dict) -> tuple[int, int | str]:
    record_id = str((record.get("metadata") or {}).get("record_id", ""))
    try:
        return 0, int(record_id)
    except ValueError:
        return 1, record_id


def merge_caches(
    input_paths: list[Path],
    output_path: Path,
    *,
    expected_records: int | None = None,
) -> dict:
    if len(input_paths) < 2:
        raise ValueError("At least two held-out fold caches are required.")

    payloads: list[dict] = []
    for path in input_paths:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "records" not in payload:
            raise ValueError(f"Invalid candidate cache: {path}")
        payloads.append(payload)

    metadata = [dict(payload.get("metadata") or {}) for payload in payloads]
    fold_ids = [item.get("oof_fold_id") for item in metadata]
    if any(value is None for value in fold_ids):
        raise ValueError("Every input cache must contain oof_fold_id metadata.")
    normalized_fold_ids = [int(value) for value in fold_ids]
    if len(set(normalized_fold_ids)) != len(normalized_fold_ids):
        raise ValueError(f"Duplicate OOF fold ids: {normalized_fold_ids}")
    if not all(bool(item.get("oof_heldout")) for item in metadata):
        raise ValueError("Every input must be marked as an OOF held-out cache.")

    candidate_hashes = {
        str(item.get("candidate_config_sha256", "")) for item in metadata
    }
    hidden_sizes = {int(item.get("hidden_size", -1)) for item in metadata}
    num_types = {int(item.get("num_types", -1)) for item in metadata}
    if len(candidate_hashes) != 1 or "" in candidate_hashes:
        raise ValueError("Fold caches use different candidate configurations.")
    if len(hidden_sizes) != 1 or len(num_types) != 1:
        raise ValueError("Fold caches use incompatible feature dimensions.")

    records: list[dict] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for path, payload in zip(input_paths, payloads):
        for record in payload["records"]:
            missing = missing_hierarchical_cache_fields(record)
            if missing:
                raise ValueError(f"Cache {path} lacks hierarchical fields {missing}.")
            record_id = str((record.get("metadata") or {}).get("record_id", ""))
            if not record_id:
                raise ValueError(f"Cache {path} contains a record without record_id.")
            if record_id in seen_ids:
                duplicate_ids.append(record_id)
            seen_ids.add(record_id)
            records.append(record)
    if duplicate_ids:
        raise ValueError(f"Duplicate record ids across folds: {duplicate_ids[:20]}")
    if expected_records is not None and len(records) != int(expected_records):
        raise ValueError(
            f"Expected {expected_records} merged records, found {len(records)}."
        )
    records.sort(key=_record_sort_key)

    correct = Counter()
    predicted = gold_count = 0
    for record in records:
        item_metadata = dict(record.get("metadata") or {})
        predictions = list(item_metadata.get("stage1_predictions") or [])
        gold = list(item_metadata.get("gold_entities") or [])
        matched = _match_counts(predictions, gold)
        correct.update(matched)
        predicted += len(predictions)
        gold_count += len(gold)
    bypass = {
        name: _f1(correct[name], predicted, gold_count)
        for name in ("span", "mner", "eeg", "gmner")
    }

    stage1_hashes = [
        str(item.get("stage1_checkpoint_sha256", "")) for item in metadata
    ]
    if any(not value for value in stage1_hashes):
        raise ValueError("Fold cache is missing a Stage1 checkpoint fingerprint.")
    ordered_folds = sorted(zip(normalized_fold_ids, stage1_hashes))
    composite_stage1_hash = stable_digest(ordered_folds)
    first = metadata[0]
    summary = {
        "records": len(records),
        "num_folds": len(input_paths),
        "fold_ids": sorted(normalized_fold_ids),
        "stage1_prediction_count": predicted,
        "gold_entity_count": gold_count,
        "stage1_bypass": bypass,
    }
    merged_metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "split": "train",
        "stage1_checkpoint": "out-of-fold composite",
        "stage1_checkpoint_sha256": composite_stage1_hash,
        "stage1_checkpoint_sha256s": [value for _, value in ordered_folds],
        "data_sources": [str(item.get("data_source", "")) for item in metadata],
        "data_source_sha256s": [
            str(item.get("data_source_sha256", "")) for item in metadata
        ],
        "candidate_config": first.get("candidate_config"),
        "candidate_config_sha256": next(iter(candidate_hashes)),
        "transition_sources": [item.get("transition_source") for item in metadata],
        "source2id": first.get("source2id"),
        "hidden_size": next(iter(hidden_sizes)),
        "num_types": next(iter(num_types)),
        "summary": summary,
        "oof": {
            "enabled": True,
            "num_folds": len(input_paths),
            "fold_ids": sorted(normalized_fold_ids),
            "fold_stage1_sha256": {
                str(fold): checksum for fold, checksum in ordered_folds
            },
            "records": len(records),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save({"metadata": merged_metadata, "records": records}, temporary)
    temporary.replace(output_path)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**summary, "output": str(output_path.resolve())}


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    result = merge_caches(
        [resolve(value, root) for value in args.inputs],
        resolve(args.output, root),
        expected_records=args.expected_records,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
