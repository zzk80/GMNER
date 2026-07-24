"""Train the independent frozen-RoBERTa hierarchical subtype sidecar."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.config import load_sidecar_config
from sidecars.fmnerg_subtype.data import SubtypeFeatureDataset
from sidecars.fmnerg_subtype.evaluator import (
    evaluate_formal_predictions,
    evaluate_gold_spans,
    load_formal_predictions,
    save_json_atomic,
    validate_feature_contract,
    validate_expected_frozen_gmner,
)
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.losses import (
    LOSS_MODES,
    build_subtype_class_weights,
)
from sidecars.fmnerg_subtype.model import (
    HEAD_ARCHITECTURES,
    HierarchicalSubtypeSidecar,
)
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default=None)
    parser.add_argument(
        "--head-architecture",
        choices=HEAD_ARCHITECTURES,
        default=None,
    )
    parser.add_argument("--parent-hidden-size", type=int, default=None)
    parser.add_argument("--effective-number-beta", type=float, default=None)
    parser.add_argument(
        "--save-best-metric",
        choices=(
            "fine_mner_f1",
            "fmnerg_f1",
            "subtype_macro_f1_on_gold_spans",
        ),
        default=None,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_gold_hierarchy(
    dataset: SubtypeFeatureDataset,
    taxonomy: SubtypeTaxonomy,
) -> None:
    expected = torch.tensor(
        [taxonomy.parent_id(int(value)) for value in dataset.subtype_ids.tolist()],
        dtype=torch.long,
    )
    if not torch.equal(expected, dataset.coarse_type_ids):
        mismatch = torch.nonzero(
            expected.ne(dataset.coarse_type_ids),
            as_tuple=False,
        ).reshape(-1)
        raise ValueError(
            "Gold subtype hierarchy mismatch at feature rows "
            f"{mismatch[:20].tolist()}."
        )


def save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def log_line(handle, message: str) -> None:
    print(message, flush=True)
    handle.write(message + "\n")
    handle.flush()


def metric_tuple(
    metrics: dict[str, float],
    *,
    primary: str,
) -> tuple[float, float, float]:
    tie_breakers = {
        "fmnerg_f1": (
            "fine_mner_f1",
            "subtype_macro_f1_on_gold_spans",
        ),
        "fine_mner_f1": (
            "fmnerg_f1",
            "subtype_macro_f1_on_gold_spans",
        ),
        "subtype_macro_f1_on_gold_spans": (
            "fmnerg_f1",
            "fine_mner_f1",
        ),
    }
    secondary, tertiary = tie_breakers[primary]
    return (
        float(metrics[primary]),
        float(metrics[secondary]),
        float(metrics[tertiary]),
    )


def checkpoint_filename(metric: str) -> str:
    return {
        "fmnerg_f1": "best_fmnerg_model.pt",
        "fine_mner_f1": "best_fine_mner_model.pt",
        "subtype_macro_f1_on_gold_spans": "best_subtype_macro_model.pt",
    }[metric]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, root)
    config = load_sidecar_config(config_path)
    if args.loss_mode is not None:
        config.optim.loss_mode = args.loss_mode
    if args.head_architecture is not None:
        config.model.head_architecture = args.head_architecture
    if args.parent_hidden_size is not None:
        if args.parent_hidden_size <= 0:
            raise ValueError("--parent-hidden-size must be positive.")
        config.model.parent_hidden_size = args.parent_hidden_size
    if args.effective_number_beta is not None:
        if not 0 < args.effective_number_beta < 1:
            raise ValueError("--effective-number-beta must be in (0, 1).")
        config.optim.effective_number_beta = args.effective_number_beta
    if args.save_best_metric is not None:
        config.runtime.save_best_metric = args.save_best_metric
    taxonomy = SubtypeTaxonomy.from_file(resolve_path(config.taxonomy, root))
    seed = int(args.seed if args.seed is not None else config.runtime.seed)
    set_seed(seed)
    requested_device = args.device or config.runtime.device
    device = torch.device(
        requested_device
        if str(requested_device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    output_dir = resolve_path(
        args.output_dir or config.runtime.output_dir,
        root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_path(config.data.train_gold_features, root)
    dev_gold_path = resolve_path(config.data.dev_gold_features, root)
    dev_formal_path = resolve_path(config.data.dev_formal_features, root)
    formal_predictions_path = resolve_path(
        config.data.dev_formal_predictions,
        root,
    )
    train_dataset = SubtypeFeatureDataset.from_file(train_path)
    dev_gold_dataset = SubtypeFeatureDataset.from_file(dev_gold_path)
    dev_formal_dataset = SubtypeFeatureDataset.from_file(dev_formal_path)
    formal_payload = load_formal_predictions(
        formal_predictions_path,
        taxonomy=taxonomy,
    )
    validate_expected_frozen_gmner(
        formal_payload,
        expected=config.runtime.expected_dev_gmner_f1,
        tolerance=config.runtime.expected_dev_gmner_tolerance,
    )
    stage1_sha256 = validate_feature_contract(
        train_dataset,
        taxonomy=taxonomy,
        split="train",
        mode="gold",
        input_size=config.model.input_size,
    )
    validate_feature_contract(
        dev_gold_dataset,
        taxonomy=taxonomy,
        split="dev",
        mode="gold",
        input_size=config.model.input_size,
        expected_stage1_sha256=stage1_sha256,
    )
    validate_feature_contract(
        dev_formal_dataset,
        taxonomy=taxonomy,
        split="dev",
        mode="formal",
        input_size=config.model.input_size,
        expected_stage1_sha256=stage1_sha256,
    )
    validate_gold_hierarchy(train_dataset, taxonomy)
    validate_gold_hierarchy(dev_gold_dataset, taxonomy)
    class_weights, loss_report = build_subtype_class_weights(
        train_dataset.subtype_ids,
        taxonomy=taxonomy,
        mode=config.optim.loss_mode,
        effective_number_beta=config.optim.effective_number_beta,
        parent_normalize=config.optim.parent_normalize_class_weights,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    configured_checkpoint = resolve_path(
        config.frozen.stage1_checkpoint,
        root,
    )
    if sha256_file(configured_checkpoint) != stage1_sha256:
        raise ValueError(
            "Configured frozen Stage1 checkpoint differs from feature caches."
        )
    if (
        dev_formal_dataset.metadata.get("formal_predictions_sha256")
        != sha256_file(formal_predictions_path)
    ):
        raise ValueError(
            "Formal prediction artifact differs from the formal feature cache."
        )

    model = HierarchicalSubtypeSidecar(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        dropout=config.model.dropout,
        taxonomy=taxonomy,
        head_architecture=config.model.head_architecture,
        parent_hidden_size=config.model.parent_hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.optim.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != total_parameters:
        raise AssertionError("Subtype sidecar unexpectedly contains frozen parameters.")

    best_checkpoint = output_dir / "best_model.pt"
    tracked_metrics = {
        config.runtime.save_best_metric,
        "fmnerg_f1",
        "subtype_macro_f1_on_gold_spans",
    }
    metric_checkpoints = {
        metric: output_dir / checkpoint_filename(metric)
        for metric in tracked_metrics
    }
    history: list[dict[str, Any]] = []
    best_scores: dict[str, tuple[float, float, float] | None] = {
        metric: None for metric in tracked_metrics
    }
    best_epochs: dict[str, int] = {metric: -1 for metric in tracked_metrics}
    stale_epochs = 0
    feature_artifacts = {
        "train_gold": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
        },
        "dev_gold": {
            "path": str(dev_gold_path),
            "sha256": sha256_file(dev_gold_path),
        },
        "dev_formal": {
            "path": str(dev_formal_path),
            "sha256": sha256_file(dev_formal_path),
        },
        "dev_formal_predictions": {
            "path": str(formal_predictions_path),
            "sha256": sha256_file(formal_predictions_path),
        },
    }
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log_line(
            log,
            json.dumps(
                {
                    "event": "subtype_sidecar_start",
                    "device": str(device),
                    "seed": seed,
                    "train_examples": len(train_dataset),
                    "dev_gold_examples": len(dev_gold_dataset),
                    "dev_formal_examples": len(dev_formal_dataset),
                    "trainable_parameters": trainable_parameters,
                    "head_architecture": config.model.head_architecture,
                    "parent_hidden_size": model.parent_hidden_size,
                    "loss": loss_report,
                    "selection_metric": config.runtime.save_best_metric,
                    "base_model_loaded": False,
                    "test_accessed": False,
                },
                ensure_ascii=False,
            ),
        )
        for epoch in range(1, config.optim.num_epochs + 1):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for batch in loader:
                features = batch["features"].to(device)
                coarse_type_ids = batch["coarse_type_ids"].to(device)
                subtype_ids = batch["subtype_ids"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features, coarse_type_ids)["logits"]
                loss = F.cross_entropy(
                    logits,
                    subtype_ids,
                    weight=class_weights,
                )
                loss.backward()
                if config.optim.gradient_clip_norm > 0:
                    clip_grad_norm_(
                        model.parameters(),
                        config.optim.gradient_clip_norm,
                    )
                optimizer.step()
                count = int(subtype_ids.numel())
                total_loss += float(loss.detach().item()) * count
                total_examples += count

            gold_metrics = evaluate_gold_spans(
                model,
                dev_gold_dataset,
                taxonomy=taxonomy,
                batch_size=config.optim.batch_size,
                device=device,
            )
            formal_result = evaluate_formal_predictions(
                model,
                dev_formal_dataset,
                formal_payload,
                taxonomy=taxonomy,
                batch_size=config.optim.batch_size,
                device=device,
            )
            metrics = {
                "train_loss": total_loss / max(total_examples, 1),
                **gold_metrics,
                **formal_result["metrics"],
            }
            epoch_result = {
                "epoch": epoch,
                "metrics": metrics,
                "gmner_identity_exact": formal_result["metadata"][
                    "gmner_identity_exact"
                ],
            }
            history.append(epoch_result)
            log_line(
                log,
                json.dumps(epoch_result, ensure_ascii=False, sort_keys=True),
            )
            primary_improved = False
            for tracked_metric in sorted(tracked_metrics):
                score = metric_tuple(metrics, primary=tracked_metric)
                previous = best_scores[tracked_metric]
                if previous is not None and score <= previous:
                    continue
                best_scores[tracked_metric] = score
                best_epochs[tracked_metric] = epoch
                payload = {
                    "kind": "fmnerg_hierarchical_subtype_sidecar",
                    "format_version": 1,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "config_path": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "taxonomy": taxonomy.to_dict(),
                    "taxonomy_sha256": taxonomy.source_sha256,
                    "stage1_checkpoint_sha256": stage1_sha256,
                    "feature_artifacts": feature_artifacts,
                    "loss": loss_report,
                    "head_architecture": config.model.head_architecture,
                    "parent_hidden_size": model.parent_hidden_size,
                    "model_parameter_count": total_parameters,
                    "selection_metric": tracked_metric,
                    "selection_tuple": list(score),
                    "metrics": metrics,
                    "base_model_loaded": False,
                    "test_accessed": False,
                }
                save_checkpoint_atomic(
                    payload,
                    metric_checkpoints[tracked_metric],
                )
                if tracked_metric == config.runtime.save_best_metric:
                    save_checkpoint_atomic(payload, best_checkpoint)
                    primary_improved = True
            if primary_improved:
                stale_epochs = 0
            else:
                stale_epochs += 1
            save_json_atomic(
                {
                    "metadata": {
                        "kind": "fmnerg_subtype_sidecar_training_history",
                        "format_version": 1,
                        "test_accessed": False,
                    },
                    "history": history,
                },
                output_dir / "history.json",
            )
            if stale_epochs >= config.optim.early_stop_patience:
                log_line(log, f"Early stopping at epoch {epoch}.")
                break

    best_payload = torch.load(best_checkpoint, map_location="cpu")
    model.load_state_dict(best_payload["model_state_dict"])
    model.to(device).eval()
    final_gold = evaluate_gold_spans(
        model,
        dev_gold_dataset,
        taxonomy=taxonomy,
        batch_size=config.optim.batch_size,
        device=device,
        include_detailed=True,
    )
    final_formal = evaluate_formal_predictions(
        model,
        dev_formal_dataset,
        formal_payload,
        taxonomy=taxonomy,
        batch_size=config.optim.batch_size,
        device=device,
    )
    summary = {
        "metadata": {
            **final_formal["metadata"],
            "kind": "fmnerg_subtype_sidecar_training_summary",
            "format_version": 1,
            "best_epoch": best_epochs[config.runtime.save_best_metric],
            "best_epochs": best_epochs,
            "best_checkpoint": str(best_checkpoint),
            "metric_checkpoints": {
                metric: str(path)
                for metric, path in metric_checkpoints.items()
            },
            "selection_metric": config.runtime.save_best_metric,
            "loss": loss_report,
            "head_architecture": config.model.head_architecture,
            "parent_hidden_size": model.parent_hidden_size,
            "model_parameter_count": total_parameters,
            "base_model_loaded": False,
            "test_accessed": False,
        },
        "metrics": {
            **final_gold,
            **final_formal["metrics"],
        },
    }
    save_json_atomic(summary, output_dir / "train_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
