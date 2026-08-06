#!/usr/bin/env python3
"""Run the one-time locked A1-T0 evaluation on folds 8-9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.artifact_utils import sha256_file
from gmner.engine.a1_t0 import (
    apply_frozen_utility,
    calibrated_probabilities,
    concatenate_folds,
    load_fold,
    load_frozen_protocol,
    predict_logits,
    ranking_diagnostics,
)
from gmner.models.a1_t0 import A1T0ActionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        default="docs/experiments/a1_t0_execution_authorization.json",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_model(checkpoint: dict[str, Any], device: str) -> A1T0ActionModel:
    config = checkpoint["model_config"]
    if checkpoint["final_epoch"] != 30 or checkpoint["class_order"] != ["FIX", "NEUTRAL", "DAMAGE"]:
        raise RuntimeError("A1-T0 checkpoint contract changed.")
    model = A1T0ActionModel(
        numeric_size=int(checkpoint["numeric_size"]),
        source_aware=bool(checkpoint["source_aware"]),
        projection_size=int(config["input_projection_size"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device)


def per_seed_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    folds = metrics["per_fold"]
    return bool(
        metrics["actions"] >= int(gate["minimum_pooled_actions"])
        and all(
            folds[str(fold)]["actions"] >= int(gate["minimum_actions_per_locked_fold"])
            for fold in (8, 9)
        )
        and all(folds[str(fold)]["net"] > 0 for fold in (8, 9))
        and metrics["corrected"] > metrics["damaged"]
        and metrics["action_precision"] >= float(gate["minimum_pooled_action_precision"])
        and metrics["formal_correct_preservation"] >= float(gate["minimum_formal_correct_preservation"])
        and metrics["net"] > 0
        and metrics["mner_f1_delta"] > 0
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "AUTHORIZED":
        raise PermissionError("A1-T0 locked evaluation is not authorized.")
    protocol = load_frozen_protocol(root, authorization)
    output_root = resolve(root, authorization["outputs"]["root"])
    result_path = output_root / authorization["outputs"]["locked_evaluation"]
    if result_path.exists():
        raise FileExistsError("A1-T0 locked evaluation has already run.")
    freeze_path = output_root / authorization["outputs"]["development_freeze"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("status") != "FROZEN_FOR_ONE_LOCKED_EVALUATION"
        or freeze.get("locked_folds_accessed") is not False
        or freeze.get("all_checkpoints_frozen_before_locked_evaluation") is not True
        or len(freeze.get("checkpoints", [])) != 6
    ):
        raise PermissionError("A1-T0 development freeze is invalid.")

    locked_payloads = {
        fold: load_fold(str(output_root / f"dataset/fold{fold}.pt"), fold)
        for fold in (8, 9)
    }
    locked_data = concatenate_folds([locked_payloads[8], locked_payloads[9]])
    seed_results = []
    by_seed_variant = {
        (int(item["seed"]), str(item["variant"])): item
        for item in freeze["checkpoints"]
    }
    for seed in protocol["model_contract"]["seeds"]:
        variant_results = {}
        for variant in ("formal_source_aware", "ablation_no_source"):
            entry = by_seed_variant[(int(seed), variant)]
            checkpoint_path = root / entry["checkpoint"]
            if sha256_file(checkpoint_path) != entry["checkpoint_sha256"]:
                raise RuntimeError("A frozen A1-T0 checkpoint changed.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if checkpoint.get("locked_folds_accessed") is not False:
                raise PermissionError("A checkpoint used locked folds during development.")
            model = load_model(checkpoint, args.device)
            logits = predict_logits(
                model,
                locked_data,
                mean=checkpoint["numeric_mean"],
                std=checkpoint["numeric_std"],
                batch_size=int(checkpoint["model_config"]["batch_size"]),
                device=args.device,
            )
            probabilities = calibrated_probabilities(logits, checkpoint["temperature"])
            selected, metrics = apply_frozen_utility(
                probabilities, locked_data, checkpoint["utility_selection"]
            )
            metrics["ranking"] = ranking_diagnostics(
                probabilities, locked_data["labels"]
            )
            metrics["base_predictions_considered"] = len(
                {item["base_prediction_id"] for item in locked_data["metadata"]}
            )
            metrics["gate_passed"] = (
                per_seed_gate(metrics, protocol["locked_evaluation_gate_per_seed"])
                if variant == "formal_source_aware"
                else None
            )
            variant_results[variant] = {
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "temperature": checkpoint["temperature"],
                "utility_selection": checkpoint["utility_selection"],
                "metrics": metrics,
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        seed_results.append({"seed": int(seed), "variants": variant_results})

    formal_metrics = [
        item["variants"]["formal_source_aware"]["metrics"] for item in seed_results
    ]
    passing = sum(bool(item["gate_passed"]) for item in formal_metrics)
    final_gate = bool(
        passing == 3
        and sum(item["net"] for item in formal_metrics) / 3 > 0
        and sum(item["mner_f1_delta"] for item in formal_metrics) / 3 > 0
    )
    result = {
        "kind": "a1_t0_locked_oof_evaluation",
        "format_version": 1,
        "status": "GO" if final_gate else "NO_GO",
        "protocol_commit": authorization["frozen_protocol_commit"],
        "authorization_sha256": sha256_file(authorization_path),
        "development_freeze_sha256": sha256_file(freeze_path),
        "development_folds": list(range(8)),
        "locked_evaluation_folds": [8, 9],
        "locked_evaluation_run_count": 1,
        "seeds": seed_results,
        "summary": {
            "passing_seeds": passing,
            "required_passing_seeds": 3,
            "mean_net": sum(item["net"] for item in formal_metrics) / 3,
            "mean_mner_f1_delta": sum(item["mner_f1_delta"] for item in formal_metrics) / 3,
            "gate_passed": final_gate,
        },
        "invariants": {
            "strict_population_286_of_31138": True,
            "prediction_count_identity": all(item["prediction_count_identity"] for item in formal_metrics),
            "type_identity": all(item["type_identity"] for item in formal_metrics),
            "region_null_identity": all(item["region_null_identity"] for item in formal_metrics),
            "locked_fold_model_selection": False,
            "locked_fold_calibration": False,
            "locked_fold_threshold_selection": False,
        },
        "latent_features": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
