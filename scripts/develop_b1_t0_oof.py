#!/usr/bin/env python3
"""Develop B1-T0 only on folds 0-7 and freeze models and thresholds."""

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
    freeze_threshold,
    load_fold_features,
    mention_counts,
    predict,
    ranking_diagnostics,
    scalar_vector,
    train_model,
)


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


def validate_authorization(payload: dict[str, Any]) -> None:
    if payload.get("status") != "AUTHORIZED":
        raise PermissionError("B1-T0 is not authorized.")
    if payload["partition_contract"]["development_folds"] != list(range(8)):
        raise PermissionError("B1-T0 development partition changed.")
    if payload["partition_contract"]["locked_evaluation_folds"] != [8, 9]:
        raise PermissionError("B1-T0 locked partition changed.")
    for key in ("dev_access", "test_access", "locked_fold_feature_selection", "locked_fold_threshold_selection"):
        if payload["forbidden"].get(key) is not True:
            raise PermissionError(f"Required B1-T0 lock is disabled: {key}")


def checkpoint_payload(
    model: torch.nn.Module,
    *,
    seed: int,
    counts: Counter[str],
    threshold: float,
    cv_metrics: dict[str, Any],
    config: dict[str, Any],
    text_size: int,
    scalar_size: int,
) -> dict[str, Any]:
    return {
        "kind": "b1_t0_frozen_development_checkpoint",
        "format_version": 1,
        "seed": int(seed),
        "development_folds": list(range(8)),
        "locked_folds_accessed": False,
        "threshold": float(threshold),
        "cv_metrics": cv_metrics,
        "mention_counts": dict(counts),
        "model_config": dict(config),
        "text_size": int(text_size),
        "scalar_size": int(scalar_size),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "dev_accessed": False,
        "test_accessed": False,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_authorization(authorization)
    output_root = resolve(root, authorization["output_contract"]["root"])
    feature_root = output_root / "features"
    development_root = output_root / "development"
    freeze_path = development_root / "development_freeze.json"
    if freeze_path.exists():
        raise FileExistsError("B1-T0 development is already frozen.")

    # This script intentionally constructs only fold0..fold7 paths. It never reads
    # the feature manifest or either locked feature payload.
    development_folds: dict[int, list[dict[str, Any]]] = {}
    feature_inputs = []
    for fold_id in range(8):
        path = feature_root / f"fold{fold_id}.pt"
        examples = load_fold_features(str(path), fold_id)
        development_folds[fold_id] = examples
        feature_inputs.append(
            {
                "fold_id": fold_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "examples": len(examples),
                "base_wrong": sum(bool(item["base_wrong"]) for item in examples),
            }
        )

    config = dict(authorization["model_contract"])
    threshold_contract = dict(authorization["threshold_contract"])
    seed_results = []
    development_root.mkdir(parents=True, exist_ok=True)
    for seed in config["seeds"]:
        cv_outputs = []
        cv_folds = []
        for heldout_fold in range(8):
            train_examples = [
                item
                for fold_id in range(8)
                if fold_id != heldout_fold
                for item in development_folds[fold_id]
            ]
            validation_examples = development_folds[heldout_fold]
            counts = mention_counts(train_examples)
            model = train_model(
                train_examples,
                counts=counts,
                config=config,
                seed=int(seed) * 100 + heldout_fold,
                device=args.device,
            )
            local = predict(
                model,
                validation_examples,
                counts=counts,
                batch_size=int(config["batch_size"]),
                device=args.device,
            )
            cv_outputs.append(local)
            cv_folds.append(
                {
                    "heldout_fold": heldout_fold,
                    "train_folds": [value for value in range(8) if value != heldout_fold],
                    "examples": len(validation_examples),
                    "base_wrong": int(local["gate_label"].sum()),
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pooled = concatenate_predictions(cv_outputs)
        threshold, cv_metrics = freeze_threshold(pooled, threshold_contract)
        cv_metrics["ranking"] = ranking_diagnostics(
            pooled["gate_score"], pooled["gate_label"]
        )
        cv_metrics["folds"] = cv_folds

        all_development = [
            item for fold_id in range(8) for item in development_folds[fold_id]
        ]
        final_counts = mention_counts(all_development)
        final_model = train_model(
            all_development,
            counts=final_counts,
            config=config,
            seed=int(seed),
            device=args.device,
        )
        scalar_size = len(scalar_vector(all_development[0], final_counts))
        text_size = int(all_development[0]["text_embedding"].numel())
        seed_dir = development_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_dir / "model.pt"
        torch.save(
            checkpoint_payload(
                final_model,
                seed=int(seed),
                counts=final_counts,
                threshold=threshold,
                cv_metrics=cv_metrics,
                config=config,
                text_size=text_size,
                scalar_size=scalar_size,
            ),
            checkpoint_path,
        )
        seed_results.append(
            {
                "seed": int(seed),
                "threshold": float(threshold),
                "development_metrics": cv_metrics,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    freeze = {
        "kind": "b1_t0_development_freeze",
        "format_version": 1,
        "status": "FROZEN_FOR_ONE_LOCKED_EVALUATION",
        "authorization": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "development_folds": list(range(8)),
        "locked_folds": [8, 9],
        "internal_validation": "leave_one_fold_out",
        "feature_inputs": feature_inputs,
        "seeds": seed_results,
        "locked_folds_accessed": False,
        "threshold_selected_on_locked_folds": False,
        "feature_selected_on_locked_folds": False,
        "visual_features": False,
        "a1_training": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
