#!/usr/bin/env python3
"""Train independent DVH-Stage1 on Train and select on Dev only."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from gmner.config import load_config
from gmner.data.dvh_record_collator import DVHRecordCollator
from gmner.data.frozen_clip_cache import (
    DVHRecordDataset,
    FrozenClipFeatureStore,
    sha256_file,
)
from gmner.data.graph_builders import GraphBuilderConfig, TextGraphBuilder
from gmner.data.record_level_stage1_dataset import RecordLevelStage1Dataset
from gmner.data.tokenization import load_word_aligned_tokenizer
from gmner.engine.s3_stage1_evaluator import evaluate_s3_stage1
from gmner.engine.utils import move_batch_to_device
from gmner.losses.dvh_stage1_loss import compute_dvh_stage1_losses
from gmner.models.dvh_stage1 import DVHStage1
from gmner.utils.seed import set_seed
from scripts.train import build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-dev-records", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve(path: str, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def build_optimizer(model: DVHStage1, config) -> AdamW:
    groups = {
        "backbone": {
            "lr": float(config.optim.backbone_learning_rate),
            "params": [],
            "names": [],
        },
        "high": {
            "lr": float(config.optim.high_level_learning_rate),
            "params": [],
            "names": [],
        },
        "new": {
            "lr": float(config.optim.new_module_learning_rate),
            "params": [],
            "names": [],
        },
        "default": {
            "lr": float(config.optim.learning_rate),
            "params": [],
            "names": [],
        },
    }
    new_prefixes = (
        "boundary_head",
        "boundary_visual_residual",
        "span_type_head",
        "span_type_projection",
        "type_visual_residual",
        "grounding_visual_residual",
        "grounding_head",
        "type_queries",
    )
    high_prefixes = (
        "text_projector",
        "text_graph_encoder",
        "region_projector",
        "region_norm",
        "image_graph_encoder",
        "clip_global_projection",
        "clip_patch_projection",
        "boundary_patch_attention",
        "type_patch_attention",
        "alignment_",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("text_encoder.backbone"):
            key = "backbone"
        elif name.startswith(new_prefixes):
            key = "new"
        elif name.startswith(high_prefixes):
            key = "high"
        else:
            key = "default"
        groups[key]["params"].append(parameter)
        groups[key]["names"].append(name)

    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assigned = [
        id(parameter)
        for group in groups.values()
        for parameter in group["params"]
    ]
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise RuntimeError("DVH optimizer parameter assignment is not exact.")
    clip_encoder_names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("clip_encoder") or name.startswith("clip_model")
    ]
    if clip_encoder_names:
        raise RuntimeError("DVH model unexpectedly contains a CLIP encoder.")
    audit = {}
    optimizer_groups = []
    for name, group in groups.items():
        if not group["params"]:
            continue
        audit[name] = {
            "learning_rate": group["lr"],
            "parameter_tensors": len(group["params"]),
            "trainable_elements": sum(
                parameter.numel() for parameter in group["params"]
            ),
            "sample_names": group["names"][:5],
        }
        optimizer_groups.append(
            {
                "params": group["params"],
                "lr": group["lr"],
                "group_name": name,
            }
        )
    print(json.dumps({"optimizer_groups": audit}, indent=2))
    optimizer = AdamW(
        optimizer_groups,
        weight_decay=float(config.optim.weight_decay),
    )
    optimizer.dvh_group_audit = audit
    return optimizer


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    config = load_config(config_path)
    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError("--max-epochs must be positive")
        config.optim.num_epochs = args.max_epochs
    if args.output_dir is not None:
        config.runtime.output_dir = args.output_dir
    if args.device is not None:
        config.runtime.device = args.device
    if args.seed is not None:
        config.runtime.seed = int(args.seed)
    if config.runtime.init_checkpoint:
        raise ValueError("DVH independent training forbids init_checkpoint.")
    if not bool(config.model.dvh_enabled):
        raise ValueError("DVH configuration is not enabled.")
    set_seed(int(config.runtime.seed))
    device = torch.device(config.runtime.device)
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8"
    )

    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=config.data.use_dependency_graph,
            dependency_backend=config.data.dependency_backend,
            dependency_model=config.data.dependency_model,
            window_size=config.data.graph_window_size,
        )
    )
    expanded_train, expanded_dev, _, _ = build_datasets(
        config=config,
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        project_root=root,
        output_dir=output_dir / "converted",
        build_test=False,
    )
    record_train = RecordLevelStage1Dataset(expanded_train, split="train")
    record_dev = RecordLevelStage1Dataset(expanded_dev, split="dev")
    clip_root = resolve(config.data.frozen_clip_feature_dir, root)
    train_store = FrozenClipFeatureStore(
        clip_root / "train",
        expected_split="train",
        expected_kind=config.data.frozen_clip_cache_kind,
    )
    dev_store = FrozenClipFeatureStore(
        clip_root / "dev",
        expected_split="dev",
        expected_kind=config.data.frozen_clip_cache_kind,
    )
    train_data = DVHRecordDataset(
        record_train,
        train_store,
        shuffle_clip=bool(config.model.dvh_shuffle_clip),
        shuffle_seed=int(config.runtime.seed),
    )
    dev_data = DVHRecordDataset(
        record_dev,
        dev_store,
        shuffle_clip=bool(config.model.dvh_shuffle_clip),
        shuffle_seed=int(config.runtime.seed),
    )
    if args.max_train_records is not None:
        train_data = torch.utils.data.Subset(
            train_data, range(min(len(train_data), args.max_train_records))
        )
    if args.max_dev_records is not None:
        dev_data = torch.utils.data.Subset(
            dev_data, range(min(len(dev_data), args.max_dev_records))
        )
    collator = DVHRecordCollator(tokenizer)
    train_loader = DataLoader(
        train_data,
        batch_size=int(config.optim.batch_size),
        shuffle=True,
        num_workers=int(config.data.num_workers),
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=int(config.optim.batch_size),
        shuffle=False,
        num_workers=int(config.data.num_workers),
        collate_fn=collator,
    )

    model = DVHStage1(config).to(device)
    optimizer = build_optimizer(model, config)
    accumulation = max(1, int(config.optim.gradient_accumulation_steps))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, updates_per_epoch * int(config.optim.num_epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * float(config.optim.warmup_ratio)),
        num_training_steps=total_updates,
    )
    use_bf16 = bool(config.runtime.fp16) and device.type == "cuda"
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, int(config.optim.num_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        loader = tqdm(train_loader, desc=f"DVH {epoch}/{config.optim.num_epochs}")
        for step, raw_batch in enumerate(loader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(batch)
                losses = compute_dvh_stage1_losses(
                    model=model,
                    outputs=outputs,
                    batch=batch,
                )
                loss = losses["loss"] / accumulation
            loss.backward()
            if step % accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config.optim.gradient_clip_norm),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(losses["loss"].detach().item())
            loader.set_postfix(loss=f"{total_loss / step:.4f}")

        evaluation = evaluate_s3_stage1(
            model=model,
            dataloader=dev_loader,
            device=device,
        )
        metrics = evaluation["metrics"]
        score = float(metrics["gmner_score"])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "metrics": metrics,
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "kind": "dvh_frozen_clip_stage1",
                    "format_version": 1,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "config": asdict(config),
                    "metadata": model.checkpoint_metadata(),
                    "clip_train_manifest_sha256": sha256_file(
                        train_store.manifest_path
                    ),
                    "clip_dev_manifest_sha256": sha256_file(
                        dev_store.manifest_path
                    ),
                    "optimizer_group_audit": optimizer.dvh_group_audit,
                    "test_accessed": False,
                },
                output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
        patience = int(config.runtime.early_stopping_patience)
        if patience > 0 and stale_epochs >= patience:
            break

    summary = {
        "kind": "dvh_stage1_training_summary",
        "status": "COMPLETED",
        "best_epoch": best_epoch,
        "best_gmner_score": best_score,
        "history": history,
        "independent_training": True,
        "clip_fully_frozen": True,
        "old_checkpoint_used": False,
        "test_accessed": False,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
