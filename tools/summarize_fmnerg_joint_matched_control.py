"""Paired Dev comparison of J0 visual fusion and C1 text continuation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.evaluator import save_json_atomic


SEEDS = (41, 42, 43)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j0-root", required=True)
    parser.add_argument("--c1-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def _load_run(root: Path, seed: int, expected_mode: str) -> dict[str, Any]:
    path = root / f"seed{seed}" / "train_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload["metadata"])
    if metadata.get("experiment_mode") != expected_mode:
        raise ValueError(
            f"Seed {seed} expected {expected_mode}, found "
            f"{metadata.get('experiment_mode')!r}."
        )
    if metadata.get("test_accessed") is not False:
        raise ValueError(f"Seed {seed} {expected_mode} accessed Test.")
    if (
        metadata.get("formal_stage1_mutated") is not False
        or metadata.get("formal_region_mutated") is not False
        or metadata.get("gmner_identity_exact") is not True
    ):
        raise ValueError(
            f"Seed {seed} {expected_mode} changed the formal chain."
        )
    return {
        "seed": seed,
        "best_epoch": int(metadata["best_epoch"]),
        "metrics": {
            key: float(value)
            for key, value in dict(payload["metrics"]).items()
            if isinstance(value, (int, float))
        },
    }


def build_matched_summary(
    j0_runs: list[dict[str, Any]],
    c1_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if [run["seed"] for run in j0_runs] != list(SEEDS):
        raise ValueError("J0 runs must use seeds 41,42,43 in order.")
    if [run["seed"] for run in c1_runs] != list(SEEDS):
        raise ValueError("C1 runs must use seeds 41,42,43 in order.")
    paired = []
    for j0, c1 in zip(j0_runs, c1_runs):
        j0_metrics = j0["metrics"]
        c1_metrics = c1["metrics"]
        for key in (
            "initial_f2_fine_mner_f1",
            "initial_f2_fmnerg_f1",
            "coarse_mner_f1",
            "eeg_f1",
            "gmner_f1",
        ):
            if abs(j0_metrics[key] - c1_metrics[key]) > 1e-12:
                raise ValueError(
                    f"Seed {j0['seed']} has an unmatched {key} baseline."
                )
        paired.append(
            {
                "seed": int(j0["seed"]),
                "j0_best_epoch": int(j0["best_epoch"]),
                "c1_best_epoch": int(c1["best_epoch"]),
                "initial_f2_fmnerg_f1": j0_metrics[
                    "initial_f2_fmnerg_f1"
                ],
                "c1_fmnerg_f1": c1_metrics["fmnerg_f1"],
                "j0_fmnerg_f1": j0_metrics["fmnerg_f1"],
                "c1_delta_vs_initial_f2": (
                    c1_metrics["fmnerg_f1"]
                    - c1_metrics["initial_f2_fmnerg_f1"]
                ),
                "j0_delta_vs_initial_f2": (
                    j0_metrics["fmnerg_f1"]
                    - j0_metrics["initial_f2_fmnerg_f1"]
                ),
                "visual_fmnerg_delta_vs_c1": (
                    j0_metrics["fmnerg_f1"] - c1_metrics["fmnerg_f1"]
                ),
                "visual_fine_mner_delta_vs_c1": (
                    j0_metrics["fine_mner_f1"]
                    - c1_metrics["fine_mner_f1"]
                ),
                "coarse_mner_f1": j0_metrics["coarse_mner_f1"],
                "eeg_f1": j0_metrics["eeg_f1"],
                "gmner_f1": j0_metrics["gmner_f1"],
            }
        )
    aggregate_keys = (
        "initial_f2_fmnerg_f1",
        "c1_fmnerg_f1",
        "j0_fmnerg_f1",
        "c1_delta_vs_initial_f2",
        "j0_delta_vs_initial_f2",
        "visual_fmnerg_delta_vs_c1",
        "visual_fine_mner_delta_vs_c1",
        "coarse_mner_f1",
        "eeg_f1",
        "gmner_f1",
    )
    aggregate = {
        key: _aggregate([float(row[key]) for row in paired])
        for key in aggregate_keys
    }
    overall_positive = sum(
        int(row["j0_delta_vs_initial_f2"] > 0) for row in paired
    )
    visual_positive = sum(
        int(row["visual_fmnerg_delta_vs_c1"] > 0) for row in paired
    )
    coarse_identity = all(
        aggregate[key]["std"] <= 1e-12
        for key in ("coarse_mner_f1", "eeg_f1", "gmner_f1")
    )
    overall_accepted = (
        aggregate["j0_delta_vs_initial_f2"]["mean"] >= 0.005
        and overall_positive >= 2
        and coarse_identity
    )
    visual_accepted = (
        aggregate["visual_fmnerg_delta_vs_c1"]["mean"] >= 0.003
        and visual_positive >= 2
        and aggregate["visual_fine_mner_delta_vs_c1"]["mean"] >= 0.0
        and coarse_identity
    )
    return {
        "metadata": {
            "kind": "fmnerg_joint_j0_matched_control_dev_summary",
            "format_version": 1,
            "selection_source": "dev",
            "seeds": list(SEEDS),
            "report": "paired_mean_std",
            "select_best_seed": False,
            "formal_stage1_mutated": False,
            "formal_region_mutated": False,
            "test_accessed": False,
        },
        "per_seed": paired,
        "aggregate": aggregate,
        "acceptance": {
            "overall_scheme": {
                "mean_j0_delta_vs_initial_f2_at_least_0.005": (
                    aggregate["j0_delta_vs_initial_f2"]["mean"] >= 0.005
                ),
                "positive_seed_count_at_least_2": overall_positive >= 2,
                "mner_eeg_gmner_identity_exact": coarse_identity,
                "accepted": overall_accepted,
            },
            "visual_module": {
                "mean_j0_delta_vs_c1_at_least_0.003": (
                    aggregate["visual_fmnerg_delta_vs_c1"]["mean"]
                    >= 0.003
                ),
                "positive_seed_count_at_least_2": visual_positive >= 2,
                "fine_mner_mean_not_lower": (
                    aggregate["visual_fine_mner_delta_vs_c1"]["mean"]
                    >= 0.0
                ),
                "mner_eeg_gmner_identity_exact": coarse_identity,
                "accepted": visual_accepted,
            },
        },
    }


def main() -> None:
    args = parse_args()
    j0_root = Path(args.j0_root)
    c1_root = Path(args.c1_root)
    j0_runs = [
        _load_run(j0_root, seed, "visual_fusion") for seed in SEEDS
    ]
    c1_runs = [
        _load_run(c1_root, seed, "text_continuation") for seed in SEEDS
    ]
    result = build_matched_summary(j0_runs, c1_runs)
    save_json_atomic(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
