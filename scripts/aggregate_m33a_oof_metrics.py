"""Aggregate formal M3.3A train metrics from ten held-out OOF feature caches.

This entry point performs no model inference or training. It reconstructs the
frozen formal prediction from fields materialized by each held-out fold:

* ``deployment_span_mask`` selects the deployed entity spans;
* ``fixed_type_ids`` supplies the frozen coarse entity type;
* ``current_visible`` chooses Fine top-1 or the NULL region.

The source train file is read only to obtain the complete gold-entity
denominator and to verify exact 7000-record coverage. Span/type/region
correctness comes from the held-out cache's frozen gold matching masks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.constants import DEFAULT_LABEL2ID, normalize_bio_label
from gmner.data.null_release_oof_cache import (
    sha256_file,
    validate_fold_oof_payload,
    validate_full_chain_oof_payload,
)
from gmner.engine.utils import f1_counts
from gmner.utils.io import read_jsonl
from gmner.utils.metrics import extract_entities_from_word_labels
from scripts.convert_gmner_conll_to_jsonl import parse_conll


METRIC_NAMES = ("span", "entity", "eeg", "triple")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--feature-root",
        help="Directory containing fold0/.../fold9/heldout_features.pt.",
    )
    inputs.add_argument(
        "--cache",
        help="Merged full_chain_train_oof.pt produced by the strict merger.",
    )
    inputs.add_argument(
        "--inputs",
        nargs="+",
        help="Explicit heldout_features.pt paths, one for every fold.",
    )
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-folds", type=int, default=10)
    parser.add_argument("--expected-records", type=int, default=7000)
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _source_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".txt":
        return parse_conll(path, image_ext=".jpg")
    return read_jsonl(path)


def load_gold_entity_counts(
    source_path: Path,
    *,
    expected_records: int,
) -> dict[str, int]:
    """Return complete per-record gold counts from the coarse BIO labels."""

    records = _source_records(source_path)
    if len(records) != int(expected_records):
        raise ValueError(
            f"Expected {expected_records} source records, found {len(records)}."
        )
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    result: dict[str, int] = {}
    for position, record in enumerate(records):
        record_id = str(
            record.get("id", record.get("record_id", record.get("sample_id", position)))
        )
        if not record_id or record_id in result:
            raise ValueError(f"Source contains an empty or duplicate id: {record_id!r}.")
        raw_tags = list(record.get("ner_tags") or [])
        labels = [
            int(tag)
            if isinstance(tag, int)
            else DEFAULT_LABEL2ID.get(
                normalize_bio_label(str(tag)), DEFAULT_LABEL2ID["O"]
            )
            for tag in raw_tags
        ]
        entities = extract_entities_from_word_labels(
            labels,
            list(record.get("tokens") or []),
            id2label,
        )
        result[record_id] = len(entities)
    return result


def _batch_counts(batch: dict, gold_counts: dict[str, int]) -> Counter:
    expanded = dict(batch["expanded"])
    fine = dict(batch["fine_outputs"])
    hierarchy = dict(batch["hierarchy_outputs"])
    record_ids = [str(value) for value in batch["record_ids"]]

    span_mask = expanded["span_mask"].bool()
    selected = batch["deployment_span_mask"].bool()
    if selected.shape != span_mask.shape or (selected & ~span_mask).any():
        raise ValueError("Deployment span mask is not a subset of valid spans.")
    if (
        selected
        & expanded["span_source_ids"].long().ne(0)
    ).any():
        raise ValueError("Formal M3.3A OOF decode unexpectedly selected a non-Stage1 span.")

    fixed_types = hierarchy["fixed_type_ids"].long()
    fine_fixed_types = fine["fixed_type_ids"].long()
    if fixed_types.shape != selected.shape or fine_fixed_types.shape != selected.shape:
        raise ValueError("Fixed type ids do not align with deployment spans.")
    if (selected & fixed_types.ne(fine_fixed_types)).any():
        raise ValueError("Hierarchy and Fine fixed types disagree on a deployed span.")

    type_correct = (
        expanded["type_candidates"].long().eq(fixed_types.unsqueeze(-1))
        & expanded["gold_type_mask"].bool()
    ).any(dim=-1)

    candidate_mask = fine["candidate_mask"].bool()
    region_mask = expanded["region_mask"].bool()
    region_is_null = expanded["region_is_null"].bool()
    if region_is_null.sum(dim=-1).ne(1).any():
        raise ValueError("Every cached record must contain exactly one NULL region.")
    real_mask = (
        candidate_mask
        & region_mask[:, None, :]
        & ~region_is_null[:, None, :]
    )
    current_visible = batch["current_visible"].bool()
    if (selected & current_visible & ~real_mask.any(dim=-1)).any():
        raise ValueError("A deployed visible prediction has no valid real candidate.")
    fine_top1 = (
        fine["final_region_logits"]
        .float()
        .masked_fill(~real_mask, -1e4)
        .argmax(dim=-1)
    )
    null_index = region_is_null.float().argmax(dim=-1)[:, None].expand_as(fine_top1)
    current_region = torch.where(current_visible, fine_top1, null_index)
    region_correct = (
        expanded["gold_region_positive_mask"]
        .bool()
        .gather(-1, current_region.unsqueeze(-1))
        .squeeze(-1)
    )
    span_correct = expanded["gold_span_mask"].bool()

    missing_ids = [record_id for record_id in record_ids if record_id not in gold_counts]
    if missing_ids:
        raise ValueError(f"OOF cache contains ids absent from train source: {missing_ids[:5]}.")

    counts = Counter()
    counts["records"] = len(record_ids)
    counts["predicted"] = int(selected.sum().item())
    counts["gold"] = sum(int(gold_counts[record_id]) for record_id in record_ids)
    counts["span_correct"] = int((selected & span_correct).sum().item())
    counts["entity_correct"] = int(
        (selected & span_correct & type_correct).sum().item()
    )
    counts["eeg_correct"] = int(
        (selected & span_correct & region_correct).sum().item()
    )
    counts["triple_correct"] = int(
        (selected & span_correct & type_correct & region_correct).sum().item()
    )
    counts["formal_visible_predictions"] = int(
        (selected & current_visible).sum().item()
    )
    counts["formal_null_predictions"] = int(
        (selected & ~current_visible).sum().item()
    )
    return counts


def _metrics(counts: Counter) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        precision, recall, score = f1_counts(
            int(counts[f"{name}_correct"]),
            int(counts["predicted"]),
            int(counts["gold"]),
        )
        result[f"{name}_precision"] = precision
        result[f"{name}_recall"] = recall
        result[f"{name}_f1"] = score
    result["mner_precision"] = result["entity_precision"]
    result["mner_recall"] = result["entity_recall"]
    result["mner_f1"] = result["entity_f1"]
    result["gmner_score"] = result["triple_f1"]
    result["type_accuracy_given_span"] = counts["entity_correct"] / max(
        counts["span_correct"], 1
    )
    result["region_accuracy_given_span"] = counts["eeg_correct"] / max(
        counts["span_correct"], 1
    )
    result["gmner_accuracy_given_span"] = counts["triple_correct"] / max(
        counts["span_correct"], 1
    )
    result["formal_visible_prediction_ratio"] = counts[
        "formal_visible_predictions"
    ] / max(counts["predicted"], 1)
    result["formal_null_prediction_ratio"] = counts[
        "formal_null_predictions"
    ] / max(counts["predicted"], 1)
    return result


def aggregate_m33a_oof_batches(
    batches: Iterable[dict],
    gold_counts: dict[str, int],
    *,
    expected_folds: int,
    expected_records: int,
) -> dict:
    """Compute micro OOF metrics and fold-level diagnostics."""

    total = Counter()
    by_fold: dict[int, Counter] = {}
    seen_ids: set[str] = set()
    for batch in batches:
        fold_id = int(batch["fold_id"])
        ids = [str(value) for value in batch["record_ids"]]
        overlap = seen_ids.intersection(ids)
        if overlap:
            raise ValueError(f"OOF records appear more than once: {sorted(overlap)[:5]}.")
        seen_ids.update(ids)
        counts = _batch_counts(batch, gold_counts)
        total.update(counts)
        by_fold.setdefault(fold_id, Counter()).update(counts)

    expected_fold_ids = set(range(int(expected_folds)))
    if set(by_fold) != expected_fold_ids:
        raise ValueError(
            f"Expected OOF folds {sorted(expected_fold_ids)}, found {sorted(by_fold)}."
        )
    if len(seen_ids) != int(expected_records):
        raise ValueError(
            f"Expected {expected_records} unique OOF records, found {len(seen_ids)}."
        )
    source_ids = set(gold_counts)
    if seen_ids != source_ids:
        raise ValueError(
            "OOF record ids do not exactly equal source train ids: "
            f"missing={len(source_ids - seen_ids)}, extra={len(seen_ids - source_ids)}."
        )

    per_fold = []
    for fold_id in sorted(by_fold):
        counts = by_fold[fold_id]
        per_fold.append(
            {
                "fold_id": fold_id,
                "counts": {key: int(value) for key, value in sorted(counts.items())},
                "metrics": _metrics(counts),
            }
        )
    fold_statistics = {}
    for name in ("span_f1", "entity_f1", "eeg_f1", "triple_f1"):
        values = [float(item["metrics"][name]) for item in per_fold]
        fold_statistics[name] = {
            "mean": statistics.mean(values),
            "population_std": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "counts": {key: int(value) for key, value in sorted(total.items())},
        "metrics": _metrics(total),
        "per_fold": per_fold,
        "fold_statistics": fold_statistics,
        "record_ids": seen_ids,
    }


def _input_paths(args: argparse.Namespace, root: Path) -> tuple[list[Path], bool]:
    if args.feature_root:
        feature_root = resolve(args.feature_root, root)
        return (
            [
                feature_root / f"fold{fold_id}" / "heldout_features.pt"
                for fold_id in range(int(args.expected_folds))
            ],
            False,
        )
    if args.inputs:
        return ([resolve(value, root) for value in args.inputs], False)
    return ([resolve(args.cache, root)], True)


def _load_batches(
    paths: list[Path],
    *,
    merged: bool,
    expected_folds: int,
    expected_records: int,
) -> tuple[list[dict], list[dict]]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing OOF feature caches: {missing}.")
    batches: list[dict] = []
    provenance = []
    if merged:
        payload = _load_torch(paths[0])
        validated = validate_full_chain_oof_payload(
            payload,
            expected_num_folds=expected_folds,
            expected_records=expected_records,
            require_reliability=True,
        )
        batches.extend(validated["batches"])
        provenance.append(
            {
                "path": str(paths[0].resolve()),
                "sha256": sha256_file(paths[0]),
                "kind": "merged_full_chain_oof",
            }
        )
        return batches, provenance

    if len(paths) != int(expected_folds):
        raise ValueError(
            f"Expected {expected_folds} held-out fold caches, found {len(paths)}."
        )
    observed = set()
    for path in paths:
        payload = _load_torch(path)
        metadata = dict(payload.get("metadata") or {})
        fold_id = int(metadata.get("fold_id", -1))
        if fold_id in observed:
            raise ValueError(f"Duplicate fold id {fold_id} in OOF inputs.")
        observed.add(fold_id)
        if not bool(metadata.get("excluded_heldout")):
            raise ValueError(f"Fold {fold_id} is not marked excluded_heldout=true.")
        validated = validate_fold_oof_payload(
            payload,
            expected_fold_id=fold_id,
            expected_record_ids=[
                str(value) for value in metadata.get("heldout_record_ids") or []
            ],
            require_reliability=True,
        )
        batches.extend(validated["batches"])
        provenance.append(
            {
                "fold_id": fold_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "fold_proof_sha256": metadata.get("fold_proof_sha256"),
                "records": int(validated["records"]),
            }
        )
    if observed != set(range(int(expected_folds))):
        raise ValueError(f"Expected fold ids 0..{expected_folds - 1}, found {sorted(observed)}.")
    return batches, sorted(provenance, key=lambda item: int(item["fold_id"]))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source_path = resolve(args.source_file, root)
    output_path = resolve(args.output, root)
    paths, merged = _input_paths(args, root)
    gold_counts = load_gold_entity_counts(
        source_path,
        expected_records=args.expected_records,
    )
    batches, feature_provenance = _load_batches(
        paths,
        merged=merged,
        expected_folds=args.expected_folds,
        expected_records=args.expected_records,
    )
    aggregate = aggregate_m33a_oof_batches(
        batches,
        gold_counts,
        expected_folds=args.expected_folds,
        expected_records=args.expected_records,
    )
    report = {
        "format_version": 1,
        "kind": "m33a_full_chain_oof_train_evaluation",
        "split": "train_oof",
        "records": int(aggregate["counts"]["records"]),
        "folds": int(args.expected_folds),
        "counts": aggregate["counts"],
        "metrics": aggregate["metrics"],
        "per_fold": aggregate["per_fold"],
        "fold_statistics": aggregate["fold_statistics"],
        "provenance": {
            "full_chain_oof": True,
            "source_file": str(source_path.resolve()),
            "source_file_sha256": sha256_file(source_path),
            "source_role": "gold denominator and exact record-id audit only",
            "feature_inputs": feature_provenance,
            "prediction_source": (
                "cached deployment_span_mask + fixed_type_ids + "
                "current_visible + Fine top-1/NULL"
            ),
            "model_inference": False,
            "model_training": False,
            "fold_ensemble": False,
            "dev_accessed": False,
            "test_accessed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": report["records"],
                "folds": report["folds"],
                "predicted": report["counts"]["predicted"],
                "gold": report["counts"]["gold"],
                "span_f1": report["metrics"]["span_f1"],
                "mner_f1": report["metrics"]["mner_f1"],
                "eeg_f1": report["metrics"]["eeg_f1"],
                "gmner_f1": report["metrics"]["gmner_score"],
                "test_accessed": False,
                "output": str(output_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
