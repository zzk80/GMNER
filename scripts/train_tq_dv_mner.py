#!/usr/bin/env python3
"""Train independent Type-Query Dual-Visual MNER on Train/Dev only."""

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
from gmner.data.frozen_clip_cache import (
    DVHRecordDataset,
    FrozenClipFeatureStore,
    sha256_file,
)
from gmner.data.graph_builders import GraphBuilderConfig, TextGraphBuilder
from gmner.data.record_level_stage1_dataset import RecordLevelStage1Dataset
from gmner.data.tokenization import load_word_aligned_tokenizer
from gmner.data.type_query_collator import TypeQueryRecordCollator
from gmner.engine.tq_mner_evaluator import evaluate_tq_mner
from gmner.engine.utils import move_batch_to_device
from gmner.losses.tq_dv_mner_loss import compute_tq_dv_mner_losses
from gmner.models.tq_dv_mner import TQDualVisualMNER
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


def build_optimizer(model: TQDualVisualMNER, config) -> AdamW:
    groups = {
        "backbone": {
            "lr": float(config.optim.backbone_learning_rate),
            "params": [],
            "names": [],
        },
        "visual": {
            "lr": float(config.optim.high_level_learning_rate),
            "params": [],
            "names": [],
        },
        "heads": {
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
    visual_prefixes = (
        "clip_global_projection",
        "clip_patch_projection",
        "region_projection",
        "query_visual_projection",
        "clip_retrieval",
        "region_retrieval",
        "visual_fusion",
    )
    head_prefixes = (
        "visual_residual",
        "existence_state",
        "existence_head",
        "existence_to_word",
        "word_norm",
        "start_head",
        "end_head",
        "span_start_projection",
        "span_end_projection",
        "qg_query_projection",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("text_encoder.backbone"):
            key = "backbone"
        elif name.startswith(visual_prefixes):
            key = "visual"
        elif name.startswith(head_prefixes):
            key = "heads"
        else:
            key = "default"
        groups[key]["params"].append(parameter)
        groups[key]["names"].append(name)

    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assigned = [
        id(parameter)
        for group in groups.values()
        for parameter in group["params"]
    ]
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise RuntimeError("TQ-DV optimizer parameter assignment is not exact.")
    if any(
        name.startswith(("clip_encoder", "clip_model"))
        for name, _ in model.named_parameters()
    ):
        raise RuntimeError("TQ-DV model must not contain a trainable CLIP encoder.")

    audit = {}
    optimizer_groups = []
    for group_name, group in groups.items():
        if not group["params"]:
            continue
        audit[group_name] = {
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
                "group_name": group_name,
            }
        )
    print(json.dumps({"optimizer_groups": audit}, indent=2))
    optimizer = AdamW(
        optimizer_groups, weight_decay=float(config.optim.weight_decay)
    )
    optimizer.tq_group_audit = audit
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
    _validate_protocol(config)

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
        shuffle_clip=bool(config.model.tq_shuffle_clip),
        shuffle_seed=int(config.runtime.seed),
    )
    dev_data = DVHRecordDataset(
        record_dev,
        dev_store,
        shuffle_clip=bool(config.model.tq_shuffle_clip),
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
    collator = TypeQueryRecordCollator(
        tokenizer,
        max_length=int(config.data.max_length),
        max_span_length=int(config.model.tq_max_span_length),
    )
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

    model = TQDualVisualMNER(config).to(device)
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
        model.set_visual_enabled(
            epoch > int(config.model.tq_visual_warmup_epochs)
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        loader = tqdm(
            train_loader, desc=f"TQ-DV {epoch}/{config.optim.num_epochs}"
        )
        for step, raw_batch in enumerate(loader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(batch)
                losses = compute_tq_dv_mner_losses(
                    model=model, outputs=outputs, batch=batch
                )
                loss = losses["loss"] / accumulation
            loss.backward()
            if step % accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.optim.gradient_clip_norm)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(losses["loss"].detach().item())
            loader.set_postfix(loss=f"{total_loss / step:.4f}")

        model.set_visual_enabled(True)
        evaluation = evaluate_tq_mner(
            model=model, dataloader=dev_loader, device=device
        )
        metrics = evaluation["metrics"]
        score = float(metrics["mner_score"])
        row = {
            "epoch": epoch,
            "visual_enabled": epoch > int(config.model.tq_visual_warmup_epochs),
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
                    "kind": "tq_dv_mner",
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
                    "optimizer_group_audit": optimizer.tq_group_audit,
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
        "kind": "tq_dv_mner_training_summary",
        "status": "COMPLETED",
        "best_epoch": best_epoch,
        "best_mner_score": best_score,
        "history": history,
        "independent_training": True,
        "clip_fully_frozen": True,
        "old_checkpoint_used": False,
        "checkpoint_selected_only_by_dev_mner": True,
        "test_accessed": False,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _validate_protocol(config) -> None:
    if config.runtime.init_checkpoint:
        raise ValueError("TQ-DV independent training forbids init_checkpoint.")
    if not bool(config.model.tq_enabled):
        raise ValueError("TQ-DV configuration is not enabled.")
    if str(config.data.test_file).strip():
        raise ValueError("TQ-DV Stage M forbids a configured Test file.")
    if str(config.runtime.save_best_metric) != "mner_score":
        raise ValueError("TQ-DV checkpoint selection must use mner_score only.")


if __name__ == "__main__":
    main()
