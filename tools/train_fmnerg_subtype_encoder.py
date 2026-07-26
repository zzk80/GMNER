"""Fine-tune an isolated RoBERTa copy for hierarchical FMNERG subtypes."""

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
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.encoder_config import (
    load_subtype_encoder_config,
)
from sidecars.fmnerg_subtype.encoder_evaluator import (
    evaluate_online_formal_predictions,
    evaluate_online_gold_spans,
    move_online_batch,
    online_model_inputs,
)
from sidecars.fmnerg_subtype.encoder_model import (
    build_optimizer_groups,
    build_trainable_subtype_encoder,
    load_trainable_checkpoint_state,
    trainable_checkpoint_state,
)
from sidecars.fmnerg_subtype.encoder_runtime import (
    load_online_subtype_data,
)
from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.online_data import OnlineSubtypeCollator
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["fmnerg_f1"]),
        float(metrics["fine_mner_f1"]),
        float(metrics["subtype_macro_f1_on_gold_spans"]),
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    gold = evaluate_online_gold_spans(
        model,
        dev_gold_dataset,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=batch_size,
        device=device,
        include_detailed=include_detailed,
    )
    formal = evaluate_online_formal_predictions(
        model,
        dev_formal_dataset,
        formal_payload,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=batch_size,
        device=device,
    )
    return {**gold, **formal["metrics"]}, formal["metadata"]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, root)
    config = load_subtype_encoder_config(config_path)
    if args.num_epochs is not None:
        if args.num_epochs <= 0:
            raise ValueError("--num-epochs must be positive.")
        config.optim.num_epochs = int(args.num_epochs)
    seed = int(args.seed if args.seed is not None else config.runtime.seed)
    config.runtime.seed = seed
    set_seed(seed)
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
    stage1_checkpoint = resolve_path(
        config.initialization.stage1_checkpoint,
        root,
    )
    if output_dir.resolve() == stage1_checkpoint.parent.resolve():
        raise ValueError(
            "Subtype encoder output_dir cannot overwrite formal Stage1."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = SubtypeTaxonomy.from_file(resolve_path(config.taxonomy, root))
    (
        train_dataset,
        dev_gold_dataset,
        dev_formal_dataset,
        formal_payload,
        data_artifacts,
    ) = load_online_subtype_data(
        config=config,
        taxonomy=taxonomy,
        root=root,
    )
    model, tokenizer, initialization, trainability = (
        build_trainable_subtype_encoder(
            config=config,
            taxonomy=taxonomy,
            root=root,
            device=device,
        )
    )
    collator = OnlineSubtypeCollator(
        tokenizer,
        max_length=int(initialization["max_length"]),
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
    optimizer_groups, optimizer_report = build_optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
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
    total_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )

    initial_metrics, initial_identity = evaluate_dev(
        model=model,
        dev_gold_dataset=dev_gold_dataset,
        dev_formal_dataset=dev_formal_dataset,
        formal_payload=formal_payload,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
    )
    if not bool(initial_identity["gmner_identity_exact"]):
        raise AssertionError("Initial subtype encoder changed frozen GMNER.")

    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_score: tuple[float, float, float] | None = None
    best_epoch = 0
    stale_epochs = 0
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log_line(
            log,
            json.dumps(
                {
                    "event": "fmnerg_subtype_encoder_start",
                    "device": str(device),
                    "seed": seed,
                    "encoder_scope": config.model.encoder_scope,
                    "train_records": len(train_dataset),
                    "train_examples": len(train_dataset.examples),
                    "dev_gold_examples": len(dev_gold_dataset.examples),
                    "dev_formal_examples": len(dev_formal_dataset.examples),
                    "total_parameters": total_parameter_count,
                    "trainable_parameters": trainable_parameter_count,
                    "optimizer_groups": optimizer_report,
                    "initial_metrics": initial_metrics,
                    "gmner_identity_exact": True,
                    "formal_stage1_mutated": False,
                    "test_accessed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, config.optim.num_epochs + 1):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for batch_index, raw_batch in enumerate(train_loader, start=1):
                batch = move_online_batch(raw_batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    outputs = model(**online_model_inputs(batch))
                    raw_loss = F.cross_entropy(
                        outputs["logits"],
                        batch["subtype_ids"],
                    )
                    loss = (
                        raw_loss
                        / config.optim.gradient_accumulation_steps
                    )
                scaler.scale(loss).backward()
                count = int(batch["subtype_ids"].numel())
                total_loss += float(raw_loss.detach().item()) * count
                total_examples += count
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

            metrics, identity = evaluate_dev(
                model=model,
                dev_gold_dataset=dev_gold_dataset,
                dev_formal_dataset=dev_formal_dataset,
                formal_payload=formal_payload,
                collator=collator,
                taxonomy=taxonomy,
                batch_size=config.optim.eval_batch_size,
                device=device,
            )
            if not bool(identity["gmner_identity_exact"]):
                raise AssertionError(
                    f"Subtype encoder changed frozen GMNER at epoch {epoch}."
                )
            metrics["train_loss"] = total_loss / max(total_examples, 1)
            result = {
                "epoch": epoch,
                "metrics": metrics,
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"])
                    for group in optimizer.param_groups
                },
                "gmner_identity_exact": True,
            }
            history.append(result)
            log_line(log, json.dumps(result, ensure_ascii=False, sort_keys=True))
            score = metric_tuple(metrics)
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                checkpoint = {
                    "kind": "fmnerg_trainable_subtype_encoder",
                    "format_version": 1,
                    "epoch": epoch,
                    "model": trainable_checkpoint_state(model),
                    "config": config.to_dict(),
                    "config_path": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "taxonomy": taxonomy.to_dict(),
                    "taxonomy_sha256": taxonomy.source_sha256,
                    "initialization": initialization,
                    "trainability": trainability,
                    "optimizer_groups": optimizer_report,
                    "data_artifacts": data_artifacts,
                    "selection_metric": "fmnerg_f1",
                    "selection_tuple": list(score),
                    "metrics": metrics,
                    "formal_stage1_mutated": False,
                    "test_accessed": False,
                }
                save_checkpoint_atomic(checkpoint, checkpoint_path)
            else:
                stale_epochs += 1
            save_json_atomic(
                {
                    "metadata": {
                        "kind": "fmnerg_subtype_encoder_training_history",
                        "format_version": 1,
                        "encoder_scope": config.model.encoder_scope,
                        "test_accessed": False,
                    },
                    "history": history,
                },
                output_dir / "history.json",
            )
            if stale_epochs >= config.optim.early_stop_patience:
                log_line(log, f"Early stopping at epoch {epoch}.")
                break

    best_payload = torch.load(checkpoint_path, map_location="cpu")
    load_trainable_checkpoint_state(model, best_payload["model"])
    model.to(device).eval()
    final_metrics, final_identity = evaluate_dev(
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
            "kind": "fmnerg_subtype_encoder_training_summary",
            "format_version": 1,
            "best_epoch": best_epoch,
            "best_checkpoint": str(checkpoint_path),
            "selection_metric": "fmnerg_f1",
            "encoder_scope": config.model.encoder_scope,
            "trainability": trainability,
            "initialization": initialization,
            "optimizer_groups": optimizer_report,
            "total_parameters": total_parameter_count,
            "trainable_parameters": trainable_parameter_count,
            "gmner_identity_exact": final_identity["gmner_identity_exact"],
            "formal_stage1_mutated": False,
            "test_accessed": False,
        },
        "metrics": final_metrics,
    }
    save_json_atomic(summary, output_dir / "train_summary.json")
    save_json_atomic(
        {
            "metadata": {
                **final_identity,
                "kind": "fmnerg_subtype_encoder_dev_evaluation",
                "format_version": 1,
                "encoder_scope": config.model.encoder_scope,
                "test_accessed": False,
            },
            "metrics": final_metrics,
        },
        output_dir / "dev_metrics.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
