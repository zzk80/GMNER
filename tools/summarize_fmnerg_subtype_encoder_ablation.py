"""Summarize frozen, last-four-layer, and full-encoder FMNERG results."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.evaluator import save_json_atomic


SCOPES = ("last4", "all")
METRICS = (
    "fine_mner_f1",
    "fmnerg_f1",
    "subtype_accuracy_on_gold_spans",
    "subtype_macro_f1_on_gold_spans",
    "parent_conditioned_subtype_accuracy",
    "gmner_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_run(root: Path, scope: str, seed: int) -> dict[str, Any]:
    path = root / f"{scope}_seed{seed}" / "train_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    expected_scope = "last_n" if scope == "last4" else "all"
    if metadata.get("encoder_scope") != expected_scope:
        raise ValueError(f"Encoder scope mismatch in {path}.")
    if metadata.get("gmner_identity_exact") is not True:
        raise ValueError(f"GMNER identity failed in {path}.")
    if metadata.get("formal_stage1_mutated") is not False:
        raise ValueError(f"Formal Stage1 mutation detected in {path}.")
    if metadata.get("test_accessed") is not False:
        raise ValueError(f"Test access detected in {path}.")
    metrics = dict(payload["metrics"])
    return {
        "scope": scope,
        "seed": seed,
        "path": str(path),
        "best_epoch": int(metadata["best_epoch"]),
        "trainable_parameters": int(metadata["trainable_parameters"]),
        "metrics": {name: float(metrics[name]) for name in METRICS},
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trainable_parameters": {
            "mean": statistics.mean(
                run["trainable_parameters"] for run in runs
            ),
            "min": min(run["trainable_parameters"] for run in runs),
            "max": max(run["trainable_parameters"] for run in runs),
        },
        "metrics": {
            metric: {
                "mean": statistics.mean(
                    run["metrics"][metric] for run in runs
                ),
                "std": statistics.pstdev(
                    run["metrics"][metric] for run in runs
                ),
                "min": min(run["metrics"][metric] for run in runs),
                "max": max(run["metrics"][metric] for run in runs),
            }
            for metric in METRICS
        },
    }


def load_frozen_f0_runs(
    root: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    output = []
    for seed in seeds:
        path = root / f"frozen_seed{seed}" / "dev_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata") or {})
        if metadata.get("test_accessed") is not False:
            raise ValueError(f"Test access detected in frozen F0 run: {path}.")
        if metadata.get("gmner_identity_exact") is not True:
            raise ValueError(f"Frozen F0 GMNER identity failed for seed {seed}.")
        metrics = dict(payload["metrics"])
        output.append(
            {
                "scope": "frozen",
                "seed": seed,
                "path": str(path),
                "best_epoch": int(metadata["checkpoint_epoch"]),
                "metrics": {
                    "fine_mner_f1": float(metrics["fine_mner_f1"]),
                    "fmnerg_f1": float(metrics["fmnerg_f1"]),
                    "subtype_accuracy_on_gold_spans": float(
                        metrics["subtype_accuracy_on_gold_spans"]
                    ),
                    "subtype_macro_f1_on_gold_spans": float(
                        metrics["subtype_macro_f1_on_gold_spans"]
                    ),
                    "parent_conditioned_subtype_accuracy": float(
                        metrics["parent_conditioned_subtype_accuracy"]
                    ),
                    "gmner_f1": float(metrics["gmner_f1"]),
                },
            }
        )
    return output


def aggregate_frozen(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metrics": {
            metric: {
                "mean": statistics.mean(
                    run["metrics"][metric] for run in runs
                ),
                "std": statistics.pstdev(
                    run["metrics"][metric] for run in runs
                ),
                "min": min(run["metrics"][metric] for run in runs),
                "max": max(run["metrics"][metric] for run in runs),
            }
            for metric in METRICS
        }
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("Encoder ablation requires at least three unique seeds.")
    f0_runs = load_frozen_f0_runs(root, seeds)
    runs = {
        scope: [load_run(root, scope, seed) for seed in seeds]
        for scope in SCOPES
    }
    summaries = {
        scope: aggregate(scope_runs)
        for scope, scope_runs in runs.items()
    }
    f0_summary = aggregate_frozen(f0_runs)
    baseline = {
        metric: f0_summary["metrics"][metric]["mean"]
        for metric in (
            "fine_mner_f1",
            "fmnerg_f1",
            "subtype_macro_f1_on_gold_spans",
        )
    }
    comparisons = {}
    for scope in SCOPES:
        summary = summaries[scope]["metrics"]
        deltas = {
            metric: summary[metric]["mean"] - baseline[metric]
            for metric in baseline
        }
        paired_fmnerg_deltas = {
            str(run["seed"]): (
                run["metrics"]["fmnerg_f1"]
                - next(
                    item["metrics"]["fmnerg_f1"]
                    for item in f0_runs
                    if item["seed"] == run["seed"]
                )
            )
            for run in runs[scope]
        }
        positive_seeds = sum(
            int(delta > 0.0) for delta in paired_fmnerg_deltas.values()
        )
        comparisons[scope] = {
            "mean_deltas_vs_frozen_f0": deltas,
            "paired_fmnerg_deltas_vs_f0": paired_fmnerg_deltas,
            "positive_fmnerg_seed_count": positive_seeds,
            "seed_count": len(seeds),
            "acceptance_checks": {
                "fmnerg_mean_delta_at_least_0.005": (
                    deltas["fmnerg_f1"] >= 0.005
                ),
                "fine_mner_mean_not_lower": deltas["fine_mner_f1"] >= 0.0,
                "all_paired_seed_fmnerg_gains_positive": (
                    positive_seeds == len(seeds)
                ),
                "gmner_identity_exact": all(
                    abs(run["metrics"]["gmner_f1"] - 0.621316) <= 5e-7
                    for run in runs[scope]
                ),
            },
        }
        comparisons[scope]["accepted"] = all(
            comparisons[scope]["acceptance_checks"].values()
        )
    winner = max(
        SCOPES,
        key=lambda scope: summaries[scope]["metrics"]["fmnerg_f1"]["mean"],
    )
    result = {
        "metadata": {
            "kind": "fmnerg_subtype_encoder_ablation_summary",
            "format_version": 1,
            "scopes": list(SCOPES),
            "seeds": seeds,
            "selection_metric": "dev_fmnerg_f1",
            "frozen_f0_reference": baseline,
            "formal_gmner_frozen": True,
            "test_accessed": False,
        },
        "per_run": {"frozen": f0_runs, **runs},
        "aggregate": {"frozen": f0_summary, **summaries},
        "comparison": comparisons,
        "best_scope_by_mean_dev_fmnerg": winner,
    }
    save_json_atomic(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
