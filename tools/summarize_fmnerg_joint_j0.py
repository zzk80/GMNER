"""Aggregate the three preregistered J0 Dev runs without seed selection."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.evaluator import save_json_atomic


METRICS = (
    "fine_mner_f1",
    "fmnerg_f1",
    "j0_base_fine_mner_f1",
    "j0_base_fmnerg_f1",
    "j0_fmnerg_delta",
    "subtype_macro_f1_on_gold_spans",
    "j0_formal_prediction_changed_rate",
    "j0_subtype_net_corrections",
    "gmner_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if seeds != [41, 42, 43]:
        raise ValueError("J0 formal Dev comparison requires seeds 41,42,43.")
    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = root / f"seed{seed}" / "train_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(payload["metadata"])
        metrics = dict(payload["metrics"])
        if metadata.get("test_accessed") is not False:
            raise ValueError(f"J0 seed {seed} accessed Test data.")
        if (
            metadata.get("formal_stage1_mutated") is not False
            or metadata.get("formal_region_mutated") is not False
            or metadata.get("gmner_identity_exact") is not True
        ):
            raise ValueError(f"J0 seed {seed} changed the formal chain.")
        rows.append(
            {
                "seed": seed,
                "best_epoch": int(metadata["best_epoch"]),
                "metrics": {name: float(metrics[name]) for name in METRICS},
            }
        )
    result = {
        name: aggregate([row["metrics"][name] for row in rows])
        for name in METRICS
    }
    positive = sum(
        int(row["metrics"]["j0_fmnerg_delta"] > 0) for row in rows
    )
    accepted = (
        result["j0_fmnerg_delta"]["mean"] >= 0.005
        and positive >= 2
        and all(
            abs(
                row["metrics"]["gmner_f1"]
                - rows[0]["metrics"]["gmner_f1"]
            )
            <= 1e-12
            for row in rows
        )
    )
    payload = {
        "metadata": {
            "kind": "fmnerg_joint_j0_dev_summary",
            "format_version": 1,
            "selection_source": "dev",
            "seeds": seeds,
            "report": "mean_std",
            "select_best_seed": False,
            "formal_stage1_mutated": False,
            "formal_region_mutated": False,
            "test_accessed": False,
        },
        "per_seed": rows,
        "aggregate": result,
        "acceptance": {
            "mean_fmnerg_delta_at_least_0.005": (
                result["j0_fmnerg_delta"]["mean"] >= 0.005
            ),
            "positive_seed_count_at_least_2": positive >= 2,
            "gmner_identity_exact": True,
            "accepted": accepted,
        },
    }
    save_json_atomic(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
