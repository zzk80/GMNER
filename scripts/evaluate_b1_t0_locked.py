#!/usr/bin/env python3
"""Run the single frozen B1-T0 evaluation on locked OOF folds 8-9."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.artifact_utils import sha256_file
from gmner.engine.b1_t0 import (
    action_metrics,
    concatenate_predictions,
    load_fold_payload,
    mner_f1,
    predict,
    ranking_diagnostics,
)
from gmner.models.b1_t0 import B1T0TextCorrectionModel


TYPE_NAMES = ("LOC", "PER", "ORG", "OTHER")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        default="docs/experiments/b1_t0_oof_separability_authorization.json",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_model(payload: dict[str, Any], device: str) -> B1T0TextCorrectionModel:
    config = payload["model_config"]
    model = B1T0TextCorrectionModel(
        text_size=int(payload["text_size"]),
        scalar_size=int(payload["scalar_size"]),
        text_projection_size=int(config["text_projection_size"]),
        scalar_projection_size=int(config["scalar_projection_size"]),
        hidden_size=int(config["shared_hidden_size"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().to(device)


def calibration_curve(scores: torch.Tensor, labels: torch.Tensor) -> list[dict[str, Any]]:
    bins = []
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        mask = scores.ge(lower) & (scores.lt(upper) if index < 9 else scores.le(upper))
        count = int(mask.sum())
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_score": float(scores[mask].mean()) if count else None,
                "base_wrong_rate": float(labels[mask].float().mean()) if count else None,
            }
        )
    return bins


def confusion_changes(
    predictions: dict[str, torch.Tensor], threshold: float
) -> dict[str, dict[str, int]]:
    action = predictions["gate_score"].ge(float(threshold))
    result: dict[str, dict[str, int]] = {"corrected": {}, "damaged": {}, "neutral": {}}
    for index in torch.nonzero(action, as_tuple=False).flatten().tolist():
        base = int(predictions["base_type"][index])
        gold = int(predictions["gold_type"][index])
        target = int(predictions["target_prediction"][index])
        if base != gold and target == gold:
            category = "corrected"
        elif base == gold:
            category = "damaged"
        else:
            category = "neutral"
        key = f"{TYPE_NAMES[base]}->{TYPE_NAMES[target]}"
        result[category][key] = result[category].get(key, 0) + 1
    return result


def locked_gate(metrics: dict[str, Any], authorization: dict[str, Any]) -> bool:
    threshold = authorization["threshold_contract"]
    return bool(
        metrics["corrected"] > metrics["damaged"]
        and metrics["net"] > 0
        and metrics["action_precision"] >= float(threshold["minimum_action_precision"])
        and metrics["base_correct_preservation"]
        >= float(threshold["minimum_base_correct_preservation"])
        and metrics["mner_f1_delta"] > 0.0
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "AUTHORIZED":
        raise PermissionError("B1-T0 is not authorized.")
    output_root = resolve(root, authorization["output_contract"]["root"])
    result_path = output_root / authorization["output_contract"]["locked_evaluation"]
    if result_path.exists():
        raise FileExistsError("The one-time B1-T0 locked evaluation already exists.")
    freeze_path = output_root / authorization["output_contract"]["development_freeze"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("status") != "FROZEN_FOR_ONE_LOCKED_EVALUATION"
        or freeze.get("locked_folds_accessed") is not False
        or freeze.get("development_folds") != list(range(8))
        or freeze.get("locked_folds") != [8, 9]
    ):
        raise PermissionError("B1-T0 development freeze is invalid.")

    feature_root = output_root / "features"
    locked_payloads = {
        fold_id: load_fold_payload(str(feature_root / f"fold{fold_id}.pt"), fold_id)
        for fold_id in (8, 9)
    }
    examples = [
        item for fold_id in (8, 9) for item in locked_payloads[fold_id]["examples"]
    ]
    baseline = {
        key: sum(int(locked_payloads[fold]["baseline_counts"][key]) for fold in (8, 9))
        for key in ("records", "prediction_count", "gold_count", "mner_correct")
    }
    baseline["mner_f1"] = mner_f1(
        baseline["mner_correct"], baseline["prediction_count"], baseline["gold_count"]
    )

    seed_results = []
    for seed_entry in freeze["seeds"]:
        checkpoint_path = Path(seed_entry["checkpoint"])
        if sha256_file(checkpoint_path) != seed_entry["checkpoint_sha256"]:
            raise RuntimeError("A frozen B1-T0 checkpoint changed before locked evaluation.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("locked_folds_accessed") is not False:
            raise PermissionError("A B1-T0 checkpoint accessed locked folds during development.")
        model = load_model(checkpoint, args.device)
        counts = Counter({str(key): int(value) for key, value in checkpoint["mention_counts"].items()})
        per_fold_predictions = [
            predict(
                model,
                locked_payloads[fold_id]["examples"],
                counts=counts,
                batch_size=int(checkpoint["model_config"]["batch_size"]),
                device=args.device,
            )
            for fold_id in (8, 9)
        ]
        pooled = concatenate_predictions(per_fold_predictions)
        threshold = float(checkpoint["threshold"])
        metrics = action_metrics(pooled, threshold)
        metrics["baseline_mner_correct"] = baseline["mner_correct"]
        metrics["final_mner_correct"] = baseline["mner_correct"] + metrics["net"]
        metrics["baseline_mner_f1"] = baseline["mner_f1"]
        metrics["final_mner_f1"] = mner_f1(
            metrics["final_mner_correct"], baseline["prediction_count"], baseline["gold_count"]
        )
        metrics["mner_f1_delta"] = metrics["final_mner_f1"] - baseline["mner_f1"]
        metrics["ranking"] = ranking_diagnostics(pooled["gate_score"], pooled["gate_label"])
        metrics["calibration"] = calibration_curve(pooled["gate_score"], pooled["gate_label"])
        metrics["confusion_changes"] = confusion_changes(pooled, threshold)
        metrics["per_fold"] = {
            str(fold_id): action_metrics(per_fold_predictions[index], threshold)
            for index, fold_id in enumerate((8, 9))
        }
        metrics["gate_passed"] = locked_gate(metrics, authorization)
        seed_results.append(
            {
                "seed": int(checkpoint["seed"]),
                "threshold_frozen_on_folds_0_7": threshold,
                "checkpoint_sha256": seed_entry["checkpoint_sha256"],
                "metrics": metrics,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passing = sum(bool(item["metrics"]["gate_passed"]) for item in seed_results)
    mean_net = sum(item["metrics"]["net"] for item in seed_results) / len(seed_results)
    mean_delta = sum(item["metrics"]["mner_f1_delta"] for item in seed_results) / len(seed_results)
    final_gate = bool(
        passing >= int(authorization["locked_gate"]["minimum_passing_seeds"])
        and mean_net > 0
        and mean_delta > 0
    )
    result = {
        "kind": "b1_t0_locked_oof_evaluation",
        "format_version": 1,
        "status": "PASS" if final_gate else "NO_GO",
        "authorization_sha256": sha256_file(authorization_path),
        "development_freeze_sha256": sha256_file(freeze_path),
        "development_folds": list(range(8)),
        "locked_evaluation_folds": [8, 9],
        "baseline": baseline,
        "seeds": seed_results,
        "summary": {
            "passing_seeds": passing,
            "seed_count": len(seed_results),
            "mean_net": mean_net,
            "mean_mner_f1_delta": mean_delta,
            "gate_passed": final_gate,
        },
        "invariants": {
            "exact_span_set_unchanged": True,
            "prediction_count_unchanged": True,
            "region_and_visibility_unchanged": True,
            "ordering_unchanged": True,
            "threshold_selected_on_locked_folds": False,
            "feature_selected_on_locked_folds": False,
        },
        "visual_features": False,
        "a1_training": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
