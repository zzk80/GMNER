"""Summarize shared versus parent-specific FMNERG subtype heads."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from tools.summarize_fmnerg_subtype_loss_ablation import (
    paired_head_tail_changes,
)


ARCHITECTURES = ("shared_hard", "parent_specific_hard")
METRICS = (
    "fine_mner_f1",
    "fmnerg_f1",
    "subtype_accuracy_on_gold_spans",
    "subtype_macro_f1_on_gold_spans",
    "parent_conditioned_subtype_accuracy",
    "predicted_parent_subtype_accuracy_on_exact_predicted_spans",
    "gold_parent_subtype_accuracy_on_exact_predicted_spans",
    "coarse_wrong_exact_predicted_span_count",
    "coarse_wrong_gold_parent_subtype_correct",
    "coarse_wrong_gold_parent_subtype_recovery_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_run(root: Path, architecture: str, seed: int) -> dict[str, Any]:
    run_dir = root / f"{architecture}_seed{seed}"
    metrics_path = run_dir / "dev_metrics.json"
    error_path = run_dir / "dev_error_analysis.json"
    if not metrics_path.is_file() or not error_path.is_file():
        raise FileNotFoundError(
            f"Incomplete subtype head run: {architecture} seed={seed}."
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    error = json.loads(error_path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    if (
        metadata.get("test_accessed") is not False
        or error.get("metadata", {}).get("test_accessed") is not False
    ):
        raise ValueError(
            f"Test access detected in {architecture} seed={seed}."
        )
    if metadata.get("gmner_identity_exact") is not True:
        raise ValueError(
            f"GMNER identity failed in {architecture} seed={seed}."
        )
    actual_architecture = str(
        metadata.get("head_architecture", "shared_hard")
    )
    if actual_architecture != architecture:
        raise ValueError(
            f"Expected {architecture}, checkpoint reports {actual_architecture}."
        )
    metrics = dict(payload["metrics"])
    return {
        "architecture": architecture,
        "seed": seed,
        "directory": str(run_dir),
        "checkpoint_epoch": int(metadata["checkpoint_epoch"]),
        "model_parameter_count": int(metadata["model_parameter_count"]),
        "values": {name: float(metrics[name]) for name in METRICS},
        "records": list(payload.get("records") or []),
        "per_subtype": dict(error["per_subtype"]),
        "gmner_identity_exact": True,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model_parameter_count": {
            "mean": statistics.mean(
                run["model_parameter_count"] for run in runs
            ),
            "min": min(run["model_parameter_count"] for run in runs),
            "max": max(run["model_parameter_count"] for run in runs),
        },
        "metrics": {
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
            for metric in METRICS
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("Head ablation requires at least three unique seeds.")
    runs = {
        architecture: [
            load_run(root, architecture, seed)
            for seed in seeds
        ]
        for architecture in ARCHITECTURES
    }
    summaries = {
        architecture: aggregate(architecture_runs)
        for architecture, architecture_runs in runs.items()
    }
    baseline = summaries["shared_hard"]
    candidate = summaries["parent_specific_hard"]
    deltas = {
        metric: (
            candidate["metrics"][metric]["mean"]
            - baseline["metrics"][metric]["mean"]
        )
        for metric in METRICS
    }
    positive_seeds = sum(
        int(
            candidate_run["values"]["fmnerg_f1"]
            > baseline_run["values"]["fmnerg_f1"]
        )
        for baseline_run, candidate_run in zip(
            runs["shared_hard"],
            runs["parent_specific_hard"],
        )
    )
    paired = [
        paired_head_tail_changes(baseline_run, candidate_run)
        for baseline_run, candidate_run in zip(
            runs["shared_hard"],
            runs["parent_specific_hard"],
        )
    ]
    paired_totals = {
        key: sum(item[key] for item in paired)
        for key in paired[0]
    }
    parameter_ratio = (
        candidate["model_parameter_count"]["mean"]
        / baseline["model_parameter_count"]["mean"]
    )
    checks = {
        "fmnerg_delta_at_least_0.005": deltas["fmnerg_f1"] >= 0.005,
        "fine_mner_not_lower": deltas["fine_mner_f1"] >= 0.0,
        "macro_f1_improved": deltas["subtype_macro_f1_on_gold_spans"] > 0.0,
        "all_seed_fmnerg_gain_positive": positive_seeds == len(seeds),
        "parameter_count_within_2_percent": abs(parameter_ratio - 1.0) <= 0.02,
    }
    result = {
        "metadata": {
            "kind": "fmnerg_subtype_head_ablation_summary",
            "format_version": 1,
            "architectures": list(ARCHITECTURES),
            "seeds": seeds,
            "selection_metric": "fmnerg_f1",
            "loss_mode": "ce",
            "soft_prior_variants_omitted": True,
            "soft_prior_reason": (
                "With frozen coarse output and a hard final parent constraint, "
                "the parent log prior is constant across all legal subtypes."
            ),
            "test_accessed": False,
        },
        "per_run": {
            architecture: [
                {
                    key: value
                    for key, value in run.items()
                    if key not in {"records", "per_subtype"}
                }
                for run in architecture_runs
            ]
            for architecture, architecture_runs in runs.items()
        },
        "aggregate": summaries,
        "comparison": {
            "mean_deltas_parent_specific_vs_shared": deltas,
            "positive_fmnerg_seed_count": positive_seeds,
            "seed_count": len(seeds),
            "parameter_count_ratio": parameter_ratio,
            "paired_head_tail_changes": paired,
            "paired_head_tail_totals": paired_totals,
            "acceptance_checks": checks,
            "accepted": all(checks.values()),
        },
    }
    save_json_atomic(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
