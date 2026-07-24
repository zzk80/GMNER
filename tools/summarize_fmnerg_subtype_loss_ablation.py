"""Summarize three-loss, multi-seed FMNERG subtype sidecar ablations."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.evaluator import save_json_atomic


MODES = ("ce", "class_weighted", "effective_number")
METRIC_SOURCES = {
    "fine_mner_f1": ("metrics", "fine_mner_f1"),
    "fmnerg_f1": ("metrics", "fmnerg_f1"),
    "gold_span_accuracy": ("metrics", "subtype_accuracy_on_gold_spans"),
    "gold_span_macro_f1": (
        "metrics",
        "subtype_macro_f1_on_gold_spans",
    ),
    "parent_correct_accuracy": (
        "metrics",
        "parent_conditioned_subtype_accuracy",
    ),
    "high_frequency_accuracy": (
        "error",
        "per_frequency",
        "high",
        "subtype_accuracy_given_correct_parent",
    ),
    "low_frequency_accuracy": (
        "error",
        "per_frequency",
        "low",
        "subtype_accuracy_given_correct_parent",
    ),
    "per_accuracy": (
        "error",
        "per_parent",
        "PER",
        "subtype_accuracy_given_correct_parent",
    ),
    "org_accuracy": (
        "error",
        "per_parent",
        "ORG",
        "subtype_accuracy_given_correct_parent",
    ),
    "visible_fmnerg_recall": (
        "error",
        "per_visibility",
        "visible",
        "fmnerg_recall",
    ),
    "null_fmnerg_recall": (
        "error",
        "per_visibility",
        "null",
        "fmnerg_recall",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def nested(payload: dict, path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def load_run(root: Path, mode: str, seed: int) -> dict[str, Any]:
    run = root / f"{mode}_seed{seed}"
    metrics_path = run / "dev_metrics.json"
    error_path = run / "dev_error_analysis.json"
    if not metrics_path.is_file() or not error_path.is_file():
        raise FileNotFoundError(
            f"Incomplete subtype ablation run: {mode} seed={seed}."
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    error = json.loads(error_path.read_text(encoding="utf-8"))
    if (
        metrics.get("metadata", {}).get("test_accessed") is not False
        or error.get("metadata", {}).get("test_accessed") is not False
    ):
        raise ValueError(f"Test access detected in {mode} seed={seed}.")
    if metrics.get("metadata", {}).get("gmner_identity_exact") is not True:
        raise ValueError(f"GMNER identity failed in {mode} seed={seed}.")
    values = {
        name: float(
            nested(
                metrics["metrics"] if source[0] == "metrics" else error,
                source[1:],
            )
        )
        for name, source in METRIC_SOURCES.items()
    }
    return {
        "mode": mode,
        "seed": seed,
        "directory": str(run),
        "metrics_path": str(metrics_path),
        "error_path": str(error_path),
        "values": values,
        "records": list(metrics.get("records") or []),
        "per_subtype": dict(error["per_subtype"]),
        "checkpoint_epoch": int(metrics["metadata"]["checkpoint_epoch"]),
        "gmner_identity_exact": True,
    }


def summarize_values(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": statistics.mean(
                run["values"][metric] for run in runs
            ),
            "std": statistics.pstdev(
                run["values"][metric] for run in runs
            ),
            "min": min(run["values"][metric] for run in runs),
            "max": max(run["values"][metric] for run in runs),
        }
        for metric in METRIC_SOURCES
    }


def subtype_outcomes(run: dict[str, Any]) -> dict[tuple[str, int, int], bool]:
    output = {}
    for record in run["records"]:
        predictions = {
            tuple(map(int, prediction["span"])): prediction
            for prediction in record.get("predictions") or []
        }
        for target in record.get("gold_entities") or []:
            start, end = map(int, target["span"])
            prediction = predictions.get((start, end))
            key = (str(record["record_id"]), start, end)
            output[key] = bool(
                prediction is not None
                and int(prediction["subtype_id"]) == int(target["subtype_id"])
            )
    return output


def paired_head_tail_changes(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    base_outcomes = subtype_outcomes(baseline)
    candidate_outcomes = subtype_outcomes(candidate)
    if base_outcomes.keys() != candidate_outcomes.keys():
        raise ValueError("Ablation runs have different dev gold-span keys.")
    label_by_key = {}
    for record in baseline["records"]:
        for target in record.get("gold_entities") or []:
            start, end = map(int, target["span"])
            label_by_key[(str(record["record_id"]), start, end)] = str(
                target["subtype"]
            )
    counts = {
        "head_corrections": 0,
        "head_damages": 0,
        "tail_corrections": 0,
        "tail_damages": 0,
    }
    for key, base_correct in base_outcomes.items():
        candidate_correct = candidate_outcomes[key]
        band = baseline["per_subtype"][label_by_key[key]]["frequency_band"]
        prefix = "head" if band == "high" else "tail" if band == "low" else None
        if prefix is None:
            continue
        if not base_correct and candidate_correct:
            counts[f"{prefix}_corrections"] += 1
        elif base_correct and not candidate_correct:
            counts[f"{prefix}_damages"] += 1
    return {
        **{key: float(value) for key, value in counts.items()},
        "head_net": float(
            counts["head_corrections"] - counts["head_damages"]
        ),
        "tail_net": float(
            counts["tail_corrections"] - counts["tail_damages"]
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("Ablation requires at least three unique seeds.")
    runs = {
        mode: [load_run(root, mode, seed) for seed in seeds]
        for mode in MODES
    }
    summaries = {
        mode: summarize_values(mode_runs)
        for mode, mode_runs in runs.items()
    }
    baseline = summaries["ce"]
    comparisons = {}
    for mode in ("class_weighted", "effective_number"):
        deltas = {
            metric: (
                summaries[mode][metric]["mean"]
                - baseline[metric]["mean"]
            )
            for metric in METRIC_SOURCES
        }
        positive_seeds = sum(
            int(candidate["values"]["fmnerg_f1"] > base["values"]["fmnerg_f1"])
            for base, candidate in zip(runs["ce"], runs[mode])
        )
        paired = [
            paired_head_tail_changes(base, candidate)
            for base, candidate in zip(runs["ce"], runs[mode])
        ]
        paired_totals = {
            key: sum(item[key] for item in paired)
            for key in paired[0]
        }
        passes = {
            "fmnerg_delta_at_least_0.005": deltas["fmnerg_f1"] >= 0.005,
            "macro_f1_delta_at_least_0.015": (
                deltas["gold_span_macro_f1"] >= 0.015
            ),
            "low_frequency_accuracy_improved": (
                deltas["low_frequency_accuracy"] > 0
            ),
            "high_frequency_drop_within_0.02": (
                deltas["high_frequency_accuracy"] >= -0.02
            ),
            "all_seed_fmnerg_gain_positive": positive_seeds == len(seeds),
        }
        comparisons[mode] = {
            "mean_deltas_vs_ce": deltas,
            "positive_fmnerg_seed_count": positive_seeds,
            "seed_count": len(seeds),
            "paired_head_tail_changes": paired,
            "paired_head_tail_totals": paired_totals,
            "acceptance_checks": passes,
            "accepted": all(passes.values()),
        }

    result = {
        "metadata": {
            "kind": "fmnerg_subtype_loss_ablation_summary",
            "format_version": 1,
            "seeds": seeds,
            "modes": list(MODES),
            "selection_metric": "fmnerg_f1",
            "test_accessed": False,
        },
        "per_run": {
            mode: [
                {
                    key: value
                    for key, value in run.items()
                    if key not in {"records", "per_subtype"}
                }
                for run in mode_runs
            ]
            for mode, mode_runs in runs.items()
        },
        "aggregate": summaries,
        "comparisons": comparisons,
    }
    save_json_atomic(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
