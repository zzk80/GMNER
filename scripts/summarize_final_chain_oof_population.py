#!/usr/bin/env python3
"""Summarize the sealed ten-fold OOF population without model selection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from gmner.data.artifact_utils import sha256_file, stable_id_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold0-dir", default="knowledge/final_chain_oof/fold0_dry_run/fold0"
    )
    parser.add_argument(
        "--population-root", default="knowledge/final_chain_oof/population_folds1_9"
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median(ordered),
        "mean": mean(ordered),
        "max": ordered[-1],
    }


def softmax_statistics(logits: list[float]) -> tuple[float, float, float]:
    if len(logits) != 4 or not all(math.isfinite(float(value)) for value in logits):
        raise ValueError("B1 type logits must contain four finite values.")
    shifted = [math.exp(float(value) - max(logits)) for value in logits]
    total = sum(shifted)
    probabilities = sorted((value / total for value in shifted), reverse=True)
    entropy = -sum(value * math.log(max(value, 1e-30)) for value in probabilities)
    return probabilities[0], probabilities[0] - probabilities[1], entropy


def confidence_bins(values: Iterable[float]) -> dict[str, int]:
    counts = Counter()
    for value in values:
        if value < 0.6:
            counts["lt_0.6"] += 1
        elif value < 0.9:
            counts["0.6_to_0.9"] += 1
        else:
            counts["ge_0.9"] += 1
    return dict(sorted(counts.items()))


def margin_bins(values: Iterable[float]) -> dict[str, int]:
    counts = Counter()
    for value in values:
        if value <= 0.1:
            counts["le_0.1"] += 1
        elif value <= 0.5:
            counts["0.1_to_0.5"] += 1
        else:
            counts["gt_0.5"] += 1
    return dict(sorted(counts.items()))


def fold_paths(fold0_dir: Path, population_root: Path, fold_id: int) -> dict[str, Path]:
    directory = fold0_dir if fold_id == 0 else population_root / f"fold{fold_id}"
    return {
        "directory": directory,
        "rows": directory / "final_chain_oof_rows.jsonl",
        "sidecar": directory / f"fold{fold_id}_supervision.jsonl",
        "supervision": directory / f"fold{fold_id}_supervision_audit.json",
        "completion": directory / f"fold{fold_id}_completion_audit.json",
        "archive": directory / "fold_archive_manifest.json",
    }


def main() -> None:
    args = parse_args()
    fold0_dir = Path(args.fold0_dir).resolve()
    population_root = Path(args.population_root).resolve()
    all_ids: list[str] = []
    all_action_ids: set[str] = set()
    fold_reports: list[dict[str, Any]] = []
    confusion: Counter[str] = Counter()
    protected: Counter[str] = Counter()
    metric_outcomes: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    action_counts_per_prediction: list[int] = []
    wrong_confidence: list[float] = []
    wrong_margin: list[float] = []
    wrong_entropy: list[float] = []
    correct_confidence: list[float] = []
    correct_margin: list[float] = []
    correct_entropy: list[float] = []

    for fold_id in range(10):
        paths = fold_paths(fold0_dir, population_root, fold_id)
        required = ("rows", "sidecar", "supervision", "completion")
        missing = [name for name in required if not paths[name].is_file()]
        if missing:
            raise FileNotFoundError(f"Fold {fold_id} is incomplete: {missing}")
        if fold_id > 0 and not paths["archive"].is_file():
            raise FileNotFoundError(f"Fold {fold_id} was not cleaned and archived.")
        rows = read_jsonl(paths["rows"])
        sidecars = read_jsonl(paths["sidecar"])
        supervision = json.loads(paths["supervision"].read_text(encoding="utf-8"))
        completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
        if len(rows) != 700 or len(sidecars) != 700:
            raise RuntimeError(f"Fold {fold_id} does not contain 700 records.")
        if completion.get("status") != "PASSED" or supervision.get("sealed_rows_unchanged") is not True:
            raise RuntimeError(f"Fold {fold_id} failed its data-contract Gate.")
        row_by_id = {row["record_id"]: row for row in rows}
        sidecar_by_id = {row["record_id"]: row for row in sidecars}
        if set(row_by_id) != set(sidecar_by_id):
            raise RuntimeError(f"Fold {fold_id} row/sidecar IDs differ.")
        if set(all_ids) & set(row_by_id):
            raise RuntimeError("A held-out record appears in more than one fold.")
        all_ids.extend(row_by_id)
        b1_correct = b1_wrong = b1_nonexact = 0
        fold_action_counts: Counter[str] = Counter()
        for record_id, row in row_by_id.items():
            supervision_row = sidecar_by_id[record_id]
            predictions = {item["prediction_id"]: item for item in row["formal_predictions"]}
            counts = Counter(action["base_prediction_id"] for action in row["replacement_actions"])
            action_counts_per_prediction.extend(counts.get(prediction_id, 0) for prediction_id in predictions)
            for action in row["replacement_actions"]:
                action_id = str(action["action_id"])
                if action_id in all_action_ids:
                    raise RuntimeError("An action identity appears more than once.")
                all_action_ids.add(action_id)
            for item in supervision_row["b1_predictions"]:
                label = str(item["population_label"])
                if label == "not_exact_span":
                    b1_nonexact += 1
                    continue
                prediction = predictions[item["prediction_id"]]
                confidence, margin, entropy = softmax_statistics(prediction["type_logits"])
                if label == "base_wrong":
                    b1_wrong += 1
                    wrong_confidence.append(confidence)
                    wrong_margin.append(margin)
                    wrong_entropy.append(entropy)
                elif label == "base_correct":
                    b1_correct += 1
                    correct_confidence.append(confidence)
                    correct_margin.append(margin)
                    correct_entropy.append(entropy)
                else:
                    raise ValueError(f"Unknown B1 population label: {label}")
            for action in supervision_row["a1_actions"]:
                label = str(action["protected_label"])
                outcome = str(action["metric_outcome"])
                source = str(action["candidate_source"])
                protected[label] += 1
                metric_outcomes[outcome] += 1
                source_counts[source][label] += 1
                fold_action_counts[label] += 1
        confusion.update(supervision["b1"]["wrong_type_confusion"])
        exact = b1_correct + b1_wrong
        fold_reports.append(
            {
                "fold_id": fold_id,
                "records": len(rows),
                "formal_predictions": sum(len(row["formal_predictions"]) for row in rows),
                "b1_exact_span": exact,
                "b1_base_correct": b1_correct,
                "b1_base_wrong": b1_wrong,
                "b1_not_exact_span": b1_nonexact,
                "b1_wrong_rate": b1_wrong / exact if exact else None,
                "b1_wrong_type_confusion": dict(
                    sorted(supervision["b1"]["wrong_type_confusion"].items())
                ),
                "a1_actions": sum(fold_action_counts.values()),
                "a1_positive": fold_action_counts["positive"],
                "a1_neutral": fold_action_counts["neutral"],
                "a1_damaging": fold_action_counts["damaging"],
                "a1_by_candidate_source": supervision["a1"]["by_candidate_source"],
                "rows_sha256": sha256_file(paths["rows"]),
                "sidecar_sha256": sha256_file(paths["sidecar"]),
            }
        )

    if len(all_ids) != 7000 or len(set(all_ids)) != 7000:
        raise RuntimeError("Ten folds do not provide exactly 7000 unique Train records.")
    total_wrong = sum(item["b1_base_wrong"] for item in fold_reports)
    total_exact = sum(item["b1_exact_span"] for item in fold_reports)
    total_positive = protected["positive"]
    summary = {
        "kind": "final_chain_oof_ten_fold_descriptive_population_summary",
        "format_version": 1,
        "status": "DATA_DISTRIBUTION_AUDIT_COMPLETE_METHOD_TRAINING_LOCKED",
        "records": len(all_ids),
        "record_ids_sha256": stable_id_digest(all_ids),
        "folds": fold_reports,
        "b1": {
            "exact_span": total_exact,
            "base_correct": sum(item["b1_base_correct"] for item in fold_reports),
            "base_wrong": total_wrong,
            "not_exact_span": sum(item["b1_not_exact_span"] for item in fold_reports),
            "wrong_rate": total_wrong / total_exact if total_exact else None,
            "wrong_present_in_all_folds": all(item["b1_base_wrong"] > 0 for item in fold_reports),
            "max_fold_wrong_fraction": max(item["b1_base_wrong"] for item in fold_reports) / total_wrong if total_wrong else None,
            "wrong_type_confusion": dict(sorted(confusion.items())),
            "observable_type_statistics": {
                "base_wrong": {
                    "top1_confidence": distribution(wrong_confidence),
                    "top1_confidence_descriptive_bins": confidence_bins(wrong_confidence),
                    "top1_top2_margin": distribution(wrong_margin),
                    "top1_top2_margin_descriptive_bins": margin_bins(wrong_margin),
                    "entropy": distribution(wrong_entropy),
                },
                "base_correct": {
                    "top1_confidence": distribution(correct_confidence),
                    "top1_confidence_descriptive_bins": confidence_bins(correct_confidence),
                    "top1_top2_margin": distribution(correct_margin),
                    "top1_top2_margin_descriptive_bins": margin_bins(correct_margin),
                    "entropy": distribution(correct_entropy),
                },
            },
            "formal_dev_reference": {
                "exact_span": 2162,
                "base_wrong": 139,
                "wrong_rate": 139 / 2162,
                "used_for_selection": False,
            },
        },
        "a1": {
            "actions": sum(protected.values()),
            "protected_label": dict(sorted(protected.items())),
            "metric_outcome": dict(sorted(metric_outcomes.items())),
            "by_candidate_source": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(source_counts.items())
            },
            "positive_present_in_all_folds": all(item["a1_positive"] > 0 for item in fold_reports),
            "max_fold_positive_fraction": max(item["a1_positive"] for item in fold_reports) / total_positive if total_positive else None,
            "actions_per_base_prediction": distribution(action_counts_per_prediction),
            "hard_negative_definition": "protected damaging actions; neutral actions reported separately",
        },
        "method_signal_evaluated": False,
        "auroc_computed": False,
        "feature_selected": False,
        "threshold_selected": False,
        "calibration_run": False,
        "training_run": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Final-chain OOF Ten-fold Descriptive Summary",
        "",
        "This report describes the sealed OOF population. It does not train a model, compute AUROC, or select a threshold.",
        "",
        f"- Records: {summary['records']}",
        f"- B1 exact-span: {summary['b1']['exact_span']}",
        f"- B1 base-wrong: {summary['b1']['base_wrong']}",
        f"- B1 wrong rate: {summary['b1']['wrong_rate']:.6f}",
        f"- A1 actions: {summary['a1']['actions']}",
        f"- A1 positive / neutral / damaging: {protected['positive']} / {protected['neutral']} / {protected['damaging']}",
        "",
        "| Fold | B1 exact | B1 wrong | Wrong rate | A1 positive | A1 neutral | A1 damaging |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in fold_reports:
        markdown.append(
            f"| {item['fold_id']} | {item['b1_exact_span']} | {item['b1_base_wrong']} | "
            f"{item['b1_wrong_rate']:.6f} | {item['a1_positive']} | "
            f"{item['a1_neutral']} | {item['a1_damaging']} |"
        )
    markdown.extend(
        [
            "",
            "Method training, feature selection, calibration, Dev, and Test remain locked.",
            "",
        ]
    )
    output_md.write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
