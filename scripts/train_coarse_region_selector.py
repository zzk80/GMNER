"""Train the recall-preserving coarse selector on expanded VinVL caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.coarse_region_selector_config import load_coarse_region_selector_config
from gmner.data import HierarchicalRecordCandidateCollator, RecordCandidateDataset
from gmner.engine.coarse_region_selector_evaluator import (
    evaluate_coarse_region_selector,
)
from gmner.losses.coarse_region_selector_loss import coarse_region_selector_loss
from gmner.models.coarse_region_selector import RecallPreservingCoarseSelector
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def make_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup = int(total_steps * max(0.0, min(float(warmup_ratio), 1.0)))

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(step, 1) / warmup
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))

    return LambdaLR(optimizer, factor)


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validate_caches(
    train: RecordCandidateDataset,
    dev: RecordCandidateDataset,
    *,
    expanded_budget: int,
    input_size: int,
) -> None:
    for dataset in (train, dev):
        spec = dict(dataset.metadata.get("candidate_config") or {})
        if int(spec.get("max_regions", 0)) != int(expanded_budget):
            raise ValueError(
                f"Cache {dataset.path} has max_regions={spec.get('max_regions')}; "
                f"expected {expanded_budget}."
            )
        hidden = int(dataset.metadata.get("hidden_size", input_size))
        if hidden != int(input_size):
            raise ValueError(
                f"Cache {dataset.path} hidden_size={hidden}; expected {input_size}."
            )
    for key in ("stage1_checkpoint_sha256", "candidate_config_sha256"):
        if str(train.metadata.get(key, "")) != str(dev.metadata.get(key, "")):
            raise ValueError(f"Train/dev expanded caches disagree on {key}.")


def compact_metrics(metrics: dict[str, float], primary: str, tie: str) -> str:
    keys = (
        primary,
        tie,
        "raw_detector_r16_recall",
        "raw_detector_r36_recall",
        "base_top16_recall_eligible",
        "learned_top16_recall_eligible",
        "union_base8_learned8_new_gold_promoted",
        "union_base8_learned8_gold_dropped",
        "union_base8_learned8_base_wrong_corrected",
        "loss",
    )
    return ", ".join(
        f"{key}={metrics[key]:.4f}" for key in keys if key in metrics
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_coarse_region_selector_config(args.config)
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.optim.num_epochs = max(1, int(args.num_epochs))
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("gmner.coarse_selector_train", output_dir / "train.log")
    set_seed(config.runtime.seed)

    datasets: dict[str, RecordCandidateDataset | Subset] = {
        "train": RecordCandidateDataset(resolve(config.data.train_cache, root)),
        "dev": RecordCandidateDataset(resolve(config.data.dev_cache, root)),
    }
    assert isinstance(datasets["train"], RecordCandidateDataset)
    assert isinstance(datasets["dev"], RecordCandidateDataset)
    _validate_caches(
        datasets["train"],
        datasets["dev"],
        expanded_budget=config.policy.expanded_budget,
        input_size=config.model.input_size,
    )
    train_metadata = dict(datasets["train"].metadata)
    dev_metadata = dict(datasets["dev"].metadata)
    if args.max_train_records is not None and args.max_train_records < len(
        datasets["train"]
    ):
        datasets["train"] = Subset(
            datasets["train"], range(max(1, int(args.max_train_records)))
        )

    collator = HierarchicalRecordCandidateCollator()
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.optim.batch_size,
            shuffle=name == "train",
            num_workers=config.data.num_workers,
            collate_fn=collator,
        )
        for name, dataset in datasets.items()
    }
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    model = RecallPreservingCoarseSelector(config.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    accumulation = max(1, int(config.optim.gradient_accumulation_steps))
    total_steps = math.ceil(len(loaders["train"]) / accumulation) * int(
        config.optim.num_epochs
    )
    scheduler = make_scheduler(optimizer, total_steps, config.optim.warmup_ratio)
    amp_enabled = bool(config.runtime.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loss_options = vars(config.loss).copy()
    loss_options["reference_budget"] = config.policy.final_budget
    evaluation_options = {
        "final_budget": config.policy.final_budget,
        "base_keep_values": config.policy.base_keep_values,
        "loss_options": loss_options,
    }
    primary = config.runtime.save_best_metric
    tie = config.runtime.save_best_tie_breaker
    logger.info(
        "Records train/dev: %d/%d; parameters=%d; device=%s",
        len(datasets["train"]),
        len(datasets["dev"]),
        sum(parameter.numel() for parameter in model.parameters()),
        device,
    )

    best_path = output_dir / "best_model.pt"
    history: list[dict] = []
    initial_metrics = evaluate_coarse_region_selector(
        model, loaders["dev"], device, **evaluation_options
    )
    if primary not in initial_metrics or tie not in initial_metrics:
        raise KeyError(f"Unknown checkpoint metrics: {primary}, {tie}")
    best_selection = (float(initial_metrics[primary]), float(initial_metrics[tie]))
    best_epoch = 0
    history.append({"epoch": 0, "dev": initial_metrics})
    atomic_save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "metrics": initial_metrics,
            "config": config.to_dict(),
            "stage1_checkpoint_sha256": dev_metadata.get(
                "stage1_checkpoint_sha256"
            ),
            "candidate_config_sha256": dev_metadata.get(
                "candidate_config_sha256"
            ),
        },
        best_path,
    )
    logger.info("Epoch 0 dev: %s", compact_metrics(initial_metrics, primary, tie))
    patience = 0

    for epoch in range(1, int(config.optim.num_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        finite_steps = 0
        progress = tqdm(
            loaders["train"],
            desc=f"Coarse selector {epoch}/{config.optim.num_epochs}",
        )
        for step, raw_batch in enumerate(progress, start=1):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(batch)
                losses = coarse_region_selector_loss(outputs, batch, **loss_options)
                loss = losses["loss"] / accumulation
            if not torch.isfinite(loss):
                logger.warning(
                    "Skipping non-finite loss at epoch=%d step=%d", epoch, step
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            running += float(losses["loss"].item())
            finite_steps += 1
            if step % accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.gradient_clip_norm
                )
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= before:
                    scheduler.step()
            if step % max(1, int(config.runtime.log_every_steps)) == 0:
                progress.set_postfix(
                    loss=f"{running / max(finite_steps, 1):.4f}"
                )

        dev_metrics = evaluate_coarse_region_selector(
            model, loaders["dev"], device, **evaluation_options
        )
        history.append({"epoch": epoch, "dev": dev_metrics})
        logger.info(
            "Epoch %d train_loss=%.4f; dev: %s",
            epoch,
            running / max(finite_steps, 1),
            compact_metrics(dev_metrics, primary, tie),
        )
        selection = (float(dev_metrics[primary]), float(dev_metrics[tie]))
        if selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            patience = 0
            atomic_save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": dev_metrics,
                    "config": config.to_dict(),
                    "stage1_checkpoint_sha256": dev_metadata.get(
                        "stage1_checkpoint_sha256"
                    ),
                    "candidate_config_sha256": dev_metadata.get(
                        "candidate_config_sha256"
                    ),
                },
                best_path,
            )
            logger.info(
                "New best epoch %d: %s=%.4f, %s=%.4f",
                epoch,
                primary,
                selection[0],
                tie,
                selection[1],
            )
        else:
            patience += 1
            if (
                config.runtime.early_stop_patience > 0
                and patience >= config.runtime.early_stop_patience
            ):
                logger.info("Early stopping at epoch %d", epoch)
                break

    report = {
        "best_epoch": best_epoch,
        "selection_metrics": [primary, tie],
        "best_selection": list(best_selection),
        "history": history,
        "train_cache": str(resolve(config.data.train_cache, root)),
        "dev_cache": str(resolve(config.data.dev_cache, root)),
        "train_stage1_checkpoint_sha256": train_metadata.get(
            "stage1_checkpoint_sha256"
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_selection": list(best_selection),
                "checkpoint": str(best_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
