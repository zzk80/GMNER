"""Train J0 visual subtype fusion while keeping M3.3A outputs immutable."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_joint.config import load_joint_subtype_config
from sidecars.fmnerg_joint.data import (
    JointOnlineSubtypeCollator,
    load_joint_subtype_data,
)
from sidecars.fmnerg_joint.evaluator import (
    evaluate_joint_formal_predictions,
    evaluate_joint_gold_spans,
    joint_model_inputs,
    move_joint_batch,
)
from sidecars.fmnerg_joint.losses import j0_visual_fusion_loss
from sidecars.fmnerg_joint.model import (
    build_j0_optimizer_groups,
    build_j0_visual_subtype_model,
    j0_checkpoint_state,
    load_j0_checkpoint_state,
)
from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate artifacts and epoch-0 identity without training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def metric_tuple(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["fmnerg_f1"]),
        float(metrics["fine_mner_f1"]),
        float(metrics["subtype_macro_f1_on_gold_spans"]),
    )


def evaluate_dev(
    *,
    model,
    dev_gold_dataset,
    dev_formal_dataset,
    formal_payload,
    collator,
    taxonomy,
    batch_size: int,
    device: torch.device,
    include_detailed: bool = False,
) -> dict[str, Any]:
    gold = evaluate_joint_gold_spans(
        model,
        dev_gold_dataset,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=batch_size,
        device=device,
        include_detailed=include_detailed,
    )
    formal = evaluate_joint_formal_predictions(
        model,
        dev_formal_dataset,
        formal_payload,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=batch_size,
        device=device,
    )
    return {
        "metadata": formal["metadata"],
        "metrics": {**gold, **formal["metrics"]},
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, root)
    config = load_joint_subtype_config(config_path)
    if args.num_epochs is not None:
        if args.num_epochs <= 0:
            raise ValueError("--num-epochs must be positive.")
        config.optim.num_epochs = int(args.num_epochs)
    seed = int(args.seed if args.seed is not None else config.runtime.seed)
    config.runtime.seed = seed
    requested_device = args.device or config.runtime.device
    config.runtime.device = str(requested_device)
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
    config.runtime.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    taxonomy = SubtypeTaxonomy.from_file(resolve_path(config.taxonomy, root))
    (
        train_dataset,
        dev_gold_dataset,
        dev_formal_dataset,
        formal_payload,
        data_artifacts,
    ) = load_joint_subtype_data(
        config=config,
        taxonomy=taxonomy,
        root=root,
    )
    model, tokenizer, encoder_config, initialization = (
        build_j0_visual_subtype_model(
            config=config,
            taxonomy=taxonomy,
            root=root,
            device=device,
            seed=seed,
        )
    )
    max_length = int(
        initialization["subtype_encoder_initialization"]["max_length"]
    )
    collator = JointOnlineSubtypeCollator(
        tokenizer,
        max_length=max_length,
        region_feature_size=config.model.region_feature_size,
        geometry_size=config.model.geometry_size,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.optim.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        generator=generator,
    )
    groups, optimizer_report = build_j0_optimizer_groups(
        model,
        encoder_config=encoder_config,
        config=config,
    )
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=config.optim.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / config.optim.gradient_accumulation_steps
    )
    total_updates = max(1, updates_per_epoch * config.optim.num_epochs)
    warmup_updates = int(total_updates * config.optim.warmup_ratio)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )
    amp_enabled = bool(
        config.runtime.mixed_precision and device.type == "cuda"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    initial = evaluate_dev(
        model=model,
        dev_gold_dataset=dev_gold_dataset,
        dev_formal_dataset=dev_formal_dataset,
        formal_payload=formal_payload,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
    )
    initial_metrics = initial["metrics"]
    expected_gmner = float(config.runtime.expected_dev_gmner_f1)
    if (
        abs(float(initial_metrics["gmner_f1"]) - expected_gmner)
        > config.runtime.expected_dev_gmner_tolerance
    ):
        raise AssertionError(
            "J0 epoch 0 does not reproduce frozen M3.3A GMNER."
        )
    if float(initial_metrics["j0_formal_prediction_changed_count"]) != 0:
        raise AssertionError(
            "Zero-initialized J0 must exactly reproduce F2 subtype predictions."
        )
    expected_initial = config.runtime.expected_initial_fmnerg_f1
    if expected_initial is None:
        expected_initial = float(
            initialization["subtype_checkpoint_metrics"]["fmnerg_f1"]
        )
    if (
        abs(float(initial_metrics["fmnerg_f1"]) - float(expected_initial))
        > config.runtime.expected_initial_fmnerg_tolerance
    ):
        raise AssertionError(
            "J0 epoch 0 does not reproduce its paired F2 checkpoint."
        )
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "seed": seed,
                    "train_records": len(train_dataset),
                    "train_examples": len(train_dataset.examples),
                    "dev_formal_examples": len(dev_formal_dataset.examples),
                    "data_coverage": data_artifacts["coverage"],
                    "initial_fine_mner_f1": initial_metrics[
                        "fine_mner_f1"
                    ],
                    "initial_fmnerg_f1": initial_metrics["fmnerg_f1"],
                    "initial_gmner_f1": initial_metrics["gmner_f1"],
                    "formal_prediction_changed_count": initial_metrics[
                        "j0_formal_prediction_changed_count"
                    ],
                    "formal_stage1_mutated": False,
                    "formal_region_mutated": False,
                    "test_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = [
        {"epoch": 0, "metrics": initial_metrics}
    ]
    best_score = metric_tuple(initial_metrics)
    best_epoch = 0
    stale_epochs = 0

    def checkpoint_payload(
        epoch: int,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "kind": "fmnerg_joint_j0_visual_fusion",
            "format_version": 1,
            "epoch": int(epoch),
            "model": j0_checkpoint_state(model),
            "config": config.to_dict(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "taxonomy": taxonomy.to_dict(),
            "taxonomy_sha256": taxonomy.source_sha256,
            "initialization": initialization,
            "optimizer_groups": optimizer_report,
            "data_artifacts": data_artifacts,
            "selection_metric": "fmnerg_f1",
            "selection_tuple": list(metric_tuple(metrics)),
            "metrics": metrics,
            "formal_stage1_mutated": False,
            "formal_region_mutated": False,
            "test_accessed": False,
        }

    save_checkpoint_atomic(
        checkpoint_payload(0, initial_metrics),
        checkpoint_path,
    )
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        start = {
            "event": "fmnerg_joint_j0_start",
            "seed": seed,
            "device": str(device),
            "train_records": len(train_dataset),
            "train_examples": len(train_dataset.examples),
            "dev_formal_examples": len(dev_formal_dataset.examples),
            "optimizer_groups": optimizer_report,
            "initial_metrics": initial_metrics,
            "formal_stage1_mutated": False,
            "formal_region_mutated": False,
            "test_accessed": False,
        }
        line = json.dumps(start, ensure_ascii=False, sort_keys=True)
        print(line, flush=True)
        log.write(line + "\n")
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, config.optim.num_epochs + 1):
            model.train()
            total_loss = 0.0
            total_examples = 0
            component_totals = {
                "loss_fused": 0.0,
                "loss_text": 0.0,
                "loss_residual": 0.0,
            }
            for batch_index, raw_batch in enumerate(train_loader, start=1):
                batch = move_joint_batch(raw_batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    outputs = model(**joint_model_inputs(batch))
                    raw_loss, components = j0_visual_fusion_loss(
                        outputs,
                        batch["subtype_ids"],
                        config.loss,
                    )
                    loss = (
                        raw_loss
                        / config.optim.gradient_accumulation_steps
                    )
                scaler.scale(loss).backward()
                count = int(batch["subtype_ids"].numel())
                total_loss += float(raw_loss.detach().item()) * count
                total_examples += count
                for name, value in components.items():
                    component_totals[name] += float(value.item()) * count
                should_step = (
                    batch_index
                    % config.optim.gradient_accumulation_steps
                    == 0
                    or batch_index == len(train_loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    if config.optim.gradient_clip_norm > 0:
                        clip_grad_norm_(
                            trainable_parameters,
                            config.optim.gradient_clip_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            result = evaluate_dev(
                model=model,
                dev_gold_dataset=dev_gold_dataset,
                dev_formal_dataset=dev_formal_dataset,
                formal_payload=formal_payload,
                collator=collator,
                taxonomy=taxonomy,
                batch_size=config.optim.eval_batch_size,
                device=device,
            )
            metrics = result["metrics"]
            if (
                abs(float(metrics["gmner_f1"]) - expected_gmner)
                > config.runtime.expected_dev_gmner_tolerance
            ):
                raise AssertionError(
                    f"J0 changed frozen GMNER at epoch {epoch}."
                )
            metrics["train_loss"] = total_loss / max(total_examples, 1)
            for name, total in component_totals.items():
                metrics[f"train_{name}"] = total / max(total_examples, 1)
            record = {
                "epoch": epoch,
                "metrics": metrics,
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"])
                    for group in optimizer.param_groups
                },
            }
            history.append(record)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            score = metric_tuple(metrics)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint_atomic(
                    checkpoint_payload(epoch, metrics),
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
            save_json_atomic(
                {
                    "metadata": {
                        "kind": "fmnerg_joint_j0_training_history",
                        "format_version": 1,
                        "test_accessed": False,
                    },
                    "history": history,
                },
                output_dir / "history.json",
            )
            if stale_epochs >= config.optim.early_stop_patience:
                message = f"Early stopping at epoch {epoch}."
                print(message, flush=True)
                log.write(message + "\n")
                break

    best = torch.load(checkpoint_path, map_location="cpu")
    load_j0_checkpoint_state(model, best["model"])
    model.to(device).eval()
    final = evaluate_dev(
        model=model,
        dev_gold_dataset=dev_gold_dataset,
        dev_formal_dataset=dev_formal_dataset,
        formal_payload=formal_payload,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
        include_detailed=True,
    )
    summary = {
        "metadata": {
            "kind": "fmnerg_joint_j0_training_summary",
            "format_version": 1,
            "best_epoch": best_epoch,
            "best_checkpoint": str(checkpoint_path),
            "selection_metric": "fmnerg_f1",
            "initialization": initialization,
            "optimizer_groups": optimizer_report,
            "formal_stage1_mutated": False,
            "formal_region_mutated": False,
            "gmner_identity_exact": final["metadata"][
                "gmner_identity_exact"
            ],
            "test_accessed": False,
        },
        "metrics": final["metrics"],
    }
    save_json_atomic(summary, output_dir / "train_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
