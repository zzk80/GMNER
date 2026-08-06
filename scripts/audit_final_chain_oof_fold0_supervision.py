#!/usr/bin/env python3
"""Attach fold supervision after sealing without mutating gold-free OOF rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID
from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.final_chain_oof_population_contract import (
    validate_final_chain_authorization,
)
from gmner.utils.metrics import extract_entities_from_word_labels


TYPE_NAMES = ("LOC", "PER", "ORG", "OTHER")
ID2LABEL = {value: key for key, value in DEFAULT_LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--materialization-report", required=True)
    parser.add_argument("--heldout-source", required=True)
    parser.add_argument("--output-sidecar", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_forbidden_path(path: Path) -> None:
    lowered = str(path).replace("\\", "/").casefold()
    if "/test" in lowered or "_test" in lowered or "/dev" in lowered or "_dev" in lowered:
        raise ValueError(f"Dev/Test path is forbidden: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def span_tuple(value: dict[str, Any]) -> tuple[int, int]:
    return int(value["start"]), int(value["end"])


def gold_entities(record: dict[str, Any]) -> list[tuple[int, int, int]]:
    entities = extract_entities_from_word_labels(
        [int(value) for value in record["ner_tags"]],
        [str(value) for value in record["tokens"]],
        ID2LABEL,
    )
    return [
        (
            int(entity["start"]),
            int(entity["end"]),
            int(ENTITY_TYPE2ID[str(entity["type"])]),
        )
        for entity in entities
    ]


def correct_count(
    predictions: Iterable[tuple[int, ...]], gold: Iterable[tuple[int, ...]]
) -> int:
    predicted_counts = Counter(predictions)
    gold_counts = Counter(gold)
    return sum(min(count, gold_counts[key]) for key, count in predicted_counts.items())


def matched_gold(
    predictions: Iterable[tuple[int, ...]], gold: Iterable[tuple[int, ...]]
) -> set[tuple[int, ...]]:
    predicted_counts = Counter(predictions)
    gold_counts = Counter(gold)
    return {
        key
        for key in gold_counts
        if min(predicted_counts[key], gold_counts[key]) > 0
    }


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return float(ordered[left])
    weight = position - left
    return float(ordered[left] * (1.0 - weight) + ordered[right] * weight)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values]
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("Non-finite observable value in supervision audit.")
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "q25": quantile(finite, 0.25),
        "median": median(finite) if finite else None,
        "mean": mean(finite) if finite else None,
        "q75": quantile(finite, 0.75),
        "max": max(finite) if finite else None,
    }


def supervise_record(
    row: dict[str, Any], source: dict[str, Any], *, fold_id: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gold_mner = gold_entities(source)
    gold_spans = [(start, end) for start, end, _ in gold_mner]
    gold_by_span = {(start, end): type_id for start, end, type_id in gold_mner}

    predictions = row["formal_predictions"]
    base_mner = [
        (*span_tuple(item["span"]), int(item["type_id"])) for item in predictions
    ]
    base_spans = [span_tuple(item["span"]) for item in predictions]
    base_mner_correct = correct_count(base_mner, gold_mner)
    base_span_correct = correct_count(base_spans, gold_spans)
    base_matched_mner = matched_gold(base_mner, gold_mner)

    b1_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        span = span_tuple(prediction["span"])
        exact = span in gold_by_span
        gold_type_id = gold_by_span.get(span)
        base_correct = bool(exact and int(prediction["type_id"]) == gold_type_id)
        b1_rows.append(
            {
                "prediction_id": prediction["prediction_id"],
                "exact_span": exact,
                "gold_type_id": gold_type_id,
                "base_type_id": int(prediction["type_id"]),
                "base_type_correct": base_correct if exact else None,
                "population_label": (
                    "base_correct" if base_correct else "base_wrong" if exact else "not_exact_span"
                ),
            }
        )

    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in row["r36_candidates"]["span_candidates"]
    }
    prediction_index = {
        prediction["prediction_id"]: index
        for index, prediction in enumerate(predictions)
    }
    a1_rows: list[dict[str, Any]] = []
    for action in row["replacement_actions"]:
        base_index = prediction_index[action["base_prediction_id"]]
        candidate = candidates[action["candidate_id"]]
        candidate_span = span_tuple(candidate["span"])
        candidate_mner = (*candidate_span, int(candidate["type_id"]))
        after_mner = list(base_mner)
        after_spans = list(base_spans)
        after_mner[base_index] = candidate_mner
        after_spans[base_index] = candidate_span
        after_mner_correct = correct_count(after_mner, gold_mner)
        after_span_correct = correct_count(after_spans, gold_spans)
        mner_delta = after_mner_correct - base_mner_correct
        span_delta = after_span_correct - base_span_correct
        lost_correct = sorted(base_matched_mner - matched_gold(after_mner, gold_mner))
        other_overlap = int(
            action["conflict_features"]["overlaps_other_formal_count"]
        )
        metric_outcome = (
            "positive" if mner_delta > 0 else "damaging" if mner_delta < 0 else "neutral"
        )
        protected_positive = (
            mner_delta > 0
            and span_delta >= 0
            and not lost_correct
            and other_overlap == 0
        )
        protected_damaging = (
            mner_delta < 0
            or span_delta < 0
            or bool(lost_correct)
            or other_overlap > 0
        )
        protected_label = (
            "positive"
            if protected_positive
            else "damaging"
            if protected_damaging
            else "neutral"
        )
        a1_rows.append(
            {
                "action_id": action["action_id"],
                "candidate_source": action["candidate_source"],
                "metric_outcome": metric_outcome,
                "protected_label": protected_label,
                "span_correct_delta": span_delta,
                "mner_correct_delta": mner_delta,
                "lost_correct_mner_entities": [list(value) for value in lost_correct],
                "overlaps_other_formal_count": other_overlap,
            }
        )

    supervision = {
        "kind": "final_chain_oof_record_supervision",
        "format_version": 1,
        "record_id": row["record_id"],
        "fold_id": int(fold_id),
        "source_row_sha256": canonical_sha256(row),
        "gold_entity_count": len(gold_mner),
        "base_span_correct_count": base_span_correct,
        "base_mner_correct_count": base_mner_correct,
        "b1_predictions": b1_rows,
        "a1_actions": a1_rows,
        "dev_accessed": False,
        "test_accessed": False,
    }
    return supervision, b1_rows, a1_rows


def main() -> None:
    args = parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "authorization": args.authorization,
            "rows": args.rows,
            "materialization_report": args.materialization_report,
            "heldout_source": args.heldout_source,
            "output_sidecar": args.output_sidecar,
            "output_report": args.output_report,
        }.items()
    }
    for name in ("rows", "materialization_report", "heldout_source"):
        reject_forbidden_path(paths[name])
    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    if authorization.get("kind") == "final_chain_oof_fold0_postseal_supervision_authorization":
        if authorization.get("status") != "AUTHORIZED" or args.fold_id != 0:
            raise RuntimeError("Fold-0 post-seal supervision audit is not authorized.")
    else:
        validate_final_chain_authorization(authorization, fold_id=args.fold_id)
    materialization = json.loads(
        paths["materialization_report"].read_text(encoding="utf-8")
    )
    rows_sha_before = sha256_file(paths["rows"])
    if materialization.get("status") != "PASSED" or materialization.get("rows_sha256") != rows_sha_before:
        raise RuntimeError("Sealed gold-free row hash does not match materialization report.")
    if any(
        materialization.get(key)
        for key in (
            "folds_1_9_accessed",
            "other_folds_accessed",
            "dev_accessed",
            "test_accessed",
        )
    ):
        raise RuntimeError("Materialization access lock is not clean.")

    rows = read_jsonl(paths["rows"])
    sources = read_jsonl(paths["heldout_source"])
    source_by_id = {str(record["id"]): record for record in sources}
    if len(rows) != 700 or len(source_by_id) != 700:
        raise RuntimeError("Fold-0 supervision audit requires exactly 700 records.")
    if [row["record_id"] for row in rows] != [str(record["id"]) for record in sources]:
        raise RuntimeError("Sealed rows and held-out source orders differ.")

    sidecars: list[dict[str, Any]] = []
    all_b1: list[dict[str, Any]] = []
    all_a1: list[dict[str, Any]] = []
    for row in rows:
        serialized = json.dumps(row, ensure_ascii=False).casefold()
        if '"supervision"' in serialized or '"gold' in serialized:
            raise RuntimeError("Gold or supervision was present before post-seal attachment.")
        sidecar, b1_rows, a1_rows = supervise_record(
            row, source_by_id[row["record_id"]], fold_id=args.fold_id
        )
        sidecars.append(sidecar)
        all_b1.extend(b1_rows)
        all_a1.extend(a1_rows)

    paths["output_sidecar"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["output_sidecar"].with_suffix(paths["output_sidecar"].suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sidecars:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(paths["output_sidecar"])
    rows_sha_after = sha256_file(paths["rows"])
    if rows_sha_after != rows_sha_before:
        raise RuntimeError("Sealed gold-free rows changed during supervision audit.")

    exact_b1 = [row for row in all_b1 if row["exact_span"]]
    wrong_b1 = [row for row in exact_b1 if row["population_label"] == "base_wrong"]
    confusion = Counter(
        f"{TYPE_NAMES[int(row['gold_type_id'])]}->{TYPE_NAMES[int(row['base_type_id'])]}"
        for row in wrong_b1
    )
    metric_counts = Counter(row["metric_outcome"] for row in all_a1)
    protected_counts = Counter(row["protected_label"] for row in all_a1)
    by_source: dict[str, Any] = {}
    for source in sorted({row["candidate_source"] for row in all_a1}):
        source_rows = [row for row in all_a1 if row["candidate_source"] == source]
        by_source[source] = {
            "count": len(source_rows),
            "metric_outcome": dict(sorted(Counter(row["metric_outcome"] for row in source_rows).items())),
            "protected_label": dict(sorted(Counter(row["protected_label"] for row in source_rows).items())),
        }
    action_by_id = {
        action["action_id"]: action
        for row in rows
        for action in row["replacement_actions"]
    }
    feature_distributions: dict[str, Any] = {}
    for label in ("positive", "damaging", "neutral"):
        labels = [row for row in all_a1 if row["protected_label"] == label]
        feature_distributions[label] = {
            "candidate_score": distribution(action_by_id[row["action_id"]]["candidate_score"] for row in labels),
            "base_candidate_margin": distribution(action_by_id[row["action_id"]]["base_candidate_margin"] for row in labels),
            "boundary_distance": distribution(action_by_id[row["action_id"]]["boundary_distance"] for row in labels),
        }

    report = {
        "kind": "final_chain_oof_fold_postseal_supervision_audit",
        "format_version": 1,
        "status": "DESCRIPTIVE_AUDIT_COMPLETE_METHOD_SIGNAL_NOT_EVALUATED",
        "fold_id": int(args.fold_id),
        "records": len(rows),
        "record_ids_sha256": stable_id_digest([row["record_id"] for row in rows]),
        "sealed_rows_sha256_before": rows_sha_before,
        "sealed_rows_sha256_after": rows_sha_after,
        "sealed_rows_unchanged": rows_sha_before == rows_sha_after,
        "sealed_discrete_replay_sha256": materialization["discrete_replay_sha256"],
        "sidecar": str(paths["output_sidecar"]),
        "sidecar_sha256": sha256_file(paths["output_sidecar"]),
        "b1": {
            "formal_predictions": len(all_b1),
            "exact_span_population": len(exact_b1),
            "base_correct": sum(row["population_label"] == "base_correct" for row in exact_b1),
            "base_wrong": len(wrong_b1),
            "not_exact_span": sum(row["population_label"] == "not_exact_span" for row in all_b1),
            "wrong_type_confusion": dict(sorted(confusion.items())),
        },
        "a1": {
            "actions": len(all_a1),
            "metric_outcome": dict(sorted(metric_counts.items())),
            "protected_label": dict(sorted(protected_counts.items())),
            "by_candidate_source": by_source,
            "observable_feature_distributions_by_protected_label": feature_distributions,
        },
        "method_signal_evaluated": False,
        "auroc_computed": False,
        "threshold_selected": False,
        "oracle_policy_computed": False,
        "training_run": False,
        "other_folds_accessed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    paths["output_report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
