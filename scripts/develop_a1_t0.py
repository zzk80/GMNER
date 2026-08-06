#!/usr/bin/env python3
"""Develop and freeze A1-T0 formal and no-source models on folds 0-7 only."""

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
    calibrated_probabilities,
    concatenate_folds,
    fit_temperature,
    load_fold,
    load_frozen_protocol,
    predict_logits,
    ranking_diagnostics,
    select_utility,
    train_model,
)


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


def checkpoint_payload(
    model: torch.nn.Module,
    *,
    seed: int,
    variant: str,
    source_aware: bool,
    mean: torch.Tensor,
    std: torch.Tensor,
    temperature: float,
    selection: dict[str, Any] | None,
    config: dict[str, Any],
    numeric_size: int,
    development_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "a1_t0_frozen_development_checkpoint",
        "format_version": 1,
        "seed": int(seed),
        "variant": variant,
        "source_aware": bool(source_aware),
        "class_order": ["FIX", "NEUTRAL", "DAMAGE"],
        "development_folds": list(range(8)),
        "locked_folds_accessed": False,
        "final_epoch": 30,
        "checkpoint_selection": "epoch_30_only",
        "temperature": float(temperature),
        "utility_selection": selection,
        "numeric_mean": mean,
        "numeric_std": std,
        "numeric_size": int(numeric_size),
        "model_config": dict(config),
        "development_metrics": development_metrics,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "dev_accessed": False,
        "test_accessed": False,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "AUTHORIZED":
        raise PermissionError("A1-T0 execution is not authorized.")
    protocol = load_frozen_protocol(root, authorization)
    if protocol["partition_contract"]["development_folds"] != list(range(8)):
        raise PermissionError("A1-T0 development partition changed.")
    if protocol["model_contract"]["epochs"] != 30:
        raise PermissionError("A1-T0 epoch contract changed.")
    output_root = resolve(root, authorization["outputs"]["root"])
    freeze_path = output_root / authorization["outputs"]["development_freeze"]
    if freeze_path.exists():
        raise FileExistsError("A1-T0 development is already frozen.")

    fold_payloads = {}
    fold_inputs = []
    for fold_id in range(8):
        path = output_root / f"dataset/fold{fold_id}.pt"
        payload = load_fold(str(path), fold_id)
        fold_payloads[fold_id] = payload
        fold_inputs.append(
            {
                "fold_id": fold_id,
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "actions": len(payload["labels"]),
            }
        )

    model_config = dict(protocol["model_contract"])
    calibration = dict(protocol["calibration_contract"])
    utility = dict(protocol["utility_contract"])
    gate = dict(protocol["development_feasibility_gate"])
    checkpoint_entries = []
    development_root = freeze_path.parent
    development_root.mkdir(parents=True, exist_ok=True)
    for seed in model_config["seeds"]:
        for variant, source_aware in (("formal_source_aware", True), ("ablation_no_source", False)):
            oof_logits = []
            oof_data_parts = []
            fold_diagnostics = []
            for heldout_fold in range(8):
                train_data = concatenate_folds(
                    fold_payloads[fold]
                    for fold in range(8)
                    if fold != heldout_fold
                )
                heldout_data = concatenate_folds([fold_payloads[heldout_fold]])
                model, mean, std = train_model(
                    train_data,
                    config=model_config,
                    seed=int(seed) * 100 + heldout_fold,
                    source_aware=source_aware,
                    device=args.device,
                )
                logits = predict_logits(
                    model,
                    heldout_data,
                    mean=mean,
                    std=std,
                    batch_size=int(model_config["batch_size"]),
                    device=args.device,
                )
                oof_logits.append(logits)
                oof_data_parts.append(heldout_data)
                fold_diagnostics.append(
                    {
                        "heldout_fold": heldout_fold,
                        "train_folds": [fold for fold in range(8) if fold != heldout_fold],
                        "actions": len(heldout_data["labels"]),
                        "checkpoint_epoch": 30,
                    }
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"seed={seed} variant={variant} heldout_fold={heldout_fold} complete",
                    flush=True,
                )

            pooled_data = concatenate_folds(oof_data_parts)
            pooled_logits = torch.cat(oof_logits)
            temperature = fit_temperature(
                pooled_logits,
                pooled_data["labels"],
                calibration["temperature_bounds"],
            )
            probabilities = calibrated_probabilities(pooled_logits, temperature)
            selection = select_utility(probabilities, pooled_data, utility, gate)
            diagnostics = ranking_diagnostics(probabilities, pooled_data["labels"])
            diagnostics["folds"] = fold_diagnostics
            diagnostics["development_gate_feasible"] = selection is not None
            diagnostics["selected_metrics"] = (
                selection["development_metrics"] if selection is not None else None
            )

            all_development = concatenate_folds(
                fold_payloads[fold] for fold in range(8)
            )
            final_model, final_mean, final_std = train_model(
                all_development,
                config=model_config,
                seed=int(seed),
                source_aware=source_aware,
                device=args.device,
            )
            checkpoint_dir = development_root / f"seed{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{variant}.pt"
            torch.save(
                checkpoint_payload(
                    final_model,
                    seed=int(seed),
                    variant=variant,
                    source_aware=source_aware,
                    mean=final_mean,
                    std=final_std,
                    temperature=temperature,
                    selection=selection,
                    config=model_config,
                    numeric_size=int(all_development["numeric_features"].shape[1]),
                    development_metrics=diagnostics,
                ),
                checkpoint_path,
            )
            checkpoint_entries.append(
                {
                    "seed": int(seed),
                    "variant": variant,
                    "source_aware": source_aware,
                    "checkpoint": str(checkpoint_path.relative_to(root)).replace("\\", "/"),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "temperature": temperature,
                    "utility_selection": selection,
                    "development_gate_feasible": selection is not None,
                }
            )
            del final_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(checkpoint_entries) != 6:
        raise RuntimeError("A1-T0 did not freeze all six checkpoints.")
    freeze = {
        "kind": "a1_t0_development_freeze",
        "format_version": 1,
        "status": "FROZEN_FOR_ONE_LOCKED_EVALUATION",
        "protocol_commit": authorization["frozen_protocol_commit"],
        "authorization_sha256": sha256_file(authorization_path),
        "development_folds": list(range(8)),
        "locked_folds": [8, 9],
        "fold_inputs": fold_inputs,
        "checkpoints": checkpoint_entries,
        "all_checkpoints_frozen_before_locked_evaluation": True,
        "epoch_30_only": True,
        "class_order": ["FIX", "NEUTRAL", "DAMAGE"],
        "quantile_interpolation": "linear",
        "duplicate_delta_candidates": "exact_float_deduplication",
        "execute_comparison": "strict_greater_than",
        "locked_folds_accessed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    freeze_path.write_bytes(
        (json.dumps(freeze, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
