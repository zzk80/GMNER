"""Train M3.3A while freezing the hierarchy and M3.2 fine adapter."""

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

from gmner.data import (
    PairedRecordCandidateCollator,
    PairedRecordCandidateDataset,
    RecordCandidateDataset,
)
from gmner.engine.evidence_visibility_evaluator import (
    evaluate_evidence_visibility,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.fine_grounding_adapter_config import (
    load_fine_grounding_adapter_config,
)
from gmner.losses.evidence_visibility_loss import evidence_visibility_loss
from gmner.models.evidence_visibility import RegionEvidenceVisibilityHead
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed
from scripts.train_fine_grounding_adapter import (
    atomic_save,
    decode_options,
    load_frozen_models,
    resolve,
    selection_key,
    validate_fingerprints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-dev-records", type=int, default=None)
    return parser.parse_args()


def make_scheduler(
    optimizer: AdamW, total_steps: int, warmup_ratio: float
) -> LambdaLR:
    warmup = int(total_steps * max(0.0, min(float(warmup_ratio), 1.0)))

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(step, 1) / warmup
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))

    return LambdaLR(optimizer, factor)


def load_frozen_chain(config, root: Path, device: torch.device):
    fine_config_path = resolve(config.frozen.fine_config, root)
    fine_config = load_fine_grounding_adapter_config(fine_config_path)
    (
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
    ) = load_frozen_models(fine_config, root, device)
    fine_checkpoint_path = resolve(config.frozen.fine_checkpoint, root)
    fine_checkpoint = torch.load(fine_checkpoint_path, map_location="cpu")
    fine_model.load_state_dict(fine_checkpoint["model_state_dict"])
    fine_model.to(device).eval()
    hierarchy.to(device).eval()
    for frozen_model in (fine_model, hierarchy):
        for parameter in frozen_model.parameters():
            parameter.requires_grad = False
    if int(config.model.input_size) != int(fine_config.model.hidden_size):
        raise ValueError(
            "Evidence model.input_size must match the M3.2 hidden size: "
            f"{config.model.input_size} != {fine_config.model.hidden_size}."
        )
    model = RegionEvidenceVisibilityHead(config.model).to(device)
    return (
        model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        fine_checkpoint,
    )


def compact_metrics(metrics: dict[str, float]) -> str:
    keys = (
        "gmner_score",
        "baseline_gmner_score",
        "gmner_delta",
        "eeg_f1",
        "visible_net_correction",
        "null_net_correction",
        "null_correct_preservation_rate",
        "fine_top1_correct_final_null_type_correct",
        "promoted_final_triple_correct",
        "loss",
    )
    return ", ".join(
        f"{key}={metrics[key]:.4f}" for key in keys if key in metrics
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_evidence_visibility_config(args.config)
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.optim.num_epochs = max(1, int(args.num_epochs))
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("gmner.evidence_visibility_train", output_dir / "train.log")
    set_seed(config.runtime.seed)

    datasets = {}
    for split in ("train", "dev"):
        formal = RecordCandidateDataset(
            resolve(getattr(config.data, f"formal_{split}_cache"), root)
        )
        expanded = RecordCandidateDataset(
            resolve(getattr(config.data, f"expanded_{split}_cache"), root)
        )
        datasets[split] = PairedRecordCandidateDataset(formal, expanded)
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        fine_checkpoint,
    ) = load_frozen_chain(config, root, device)
    for split in ("train", "dev"):
        validate_fingerprints(
            datasets[split],
            hierarchy_checkpoint=hierarchy_checkpoint,
            coarse_checkpoint=coarse_checkpoint,
            require_oof=(
                config.data.require_oof_train_cache and split == "train"
            ),
        )
    if args.max_train_records is not None:
        datasets["train"] = Subset(
            datasets["train"],
            range(min(max(1, args.max_train_records), len(datasets["train"]))),
        )
    if args.max_dev_records is not None:
        datasets["dev"] = Subset(
            datasets["dev"],
            range(min(max(1, args.max_dev_records), len(datasets["dev"]))),
        )
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.optim.batch_size,
            shuffle=split == "train",
            num_workers=config.data.num_workers,
            collate_fn=PairedRecordCandidateCollator(),
        )
        for split, dataset in datasets.items()
    }
    trainable = list(model.parameters())
    optimizer = AdamW(
        trainable,
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    accumulation = max(1, config.optim.gradient_accumulation_steps)
    total_steps = math.ceil(len(loaders["train"]) / accumulation) * int(
        config.optim.num_epochs
    )
    scheduler = make_scheduler(optimizer, total_steps, config.optim.warmup_ratio)
    amp_enabled = bool(config.runtime.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loss_options = vars(config.loss).copy()
    evaluation_options = {
        "decode_options": decode_options(hierarchy_config),
        "loss_options": loss_options,
    }
    logger.info(
        "Records train/dev=%d/%d; trainable=%d; frozen hierarchy/fine/coarse; device=%s",
        len(datasets["train"]),
        len(datasets["dev"]),
        sum(parameter.numel() for parameter in trainable),
        device,
    )

    primary = config.runtime.save_best_metric
    ties = list(config.runtime.save_best_tie_breakers)
    best_path = output_dir / "best_model.pt"
    history: list[dict] = []
    initial = evaluate_evidence_visibility(
        model,
        fine_model,
        hierarchy,
        loaders["dev"],
        device,
        **evaluation_options,
    )
    if abs(float(initial["gmner_delta"])) > 1e-12:
        raise RuntimeError(
            "Zero-initialized M3.3A must reproduce the frozen M3.2 baseline; "
            f"observed delta={initial['gmner_delta']}."
        )
    best_selection = selection_key(initial, primary, ties)
    best_epoch = 0
    history.append({"epoch": 0, "dev": initial})

    def checkpoint_payload(epoch: int, metrics: dict[str, float]) -> dict:
        return {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config.to_dict(),
            "fine_checkpoint_epoch": fine_checkpoint.get("epoch"),
            "fine_checkpoint_metrics": fine_checkpoint.get("metrics"),
            "hierarchy_stage1_checkpoint_sha256": hierarchy_checkpoint.get(
                "stage1_checkpoint_sha256"
            ),
            "formal_candidate_config_sha256": hierarchy_checkpoint.get(
                "candidate_config_sha256"
            ),
            "expanded_candidate_config_sha256": coarse_checkpoint.get(
                "candidate_config_sha256"
            ),
        }

    atomic_save(checkpoint_payload(0, initial), best_path)
    logger.info("Epoch 0 frozen M3.2 dev: %s", compact_metrics(initial))
    patience = 0

    region_decode_options = {
        key: value
        for key, value in evaluation_options["decode_options"].items()
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    for epoch in range(1, config.optim.num_epochs + 1):
        model.train()
        fine_model.eval()
        hierarchy.eval()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        finite_steps = 0
        progress = tqdm(
            loaders["train"],
            desc=f"Evidence visibility {epoch}/{config.optim.num_epochs}",
        )
        for step, raw_batch in enumerate(progress, start=1):
            paired = move_paired_record_batch(raw_batch, device)
            formal = paired["formal"]
            expanded = paired["expanded"]
            with torch.no_grad():
                baseline = frozen_hierarchical_context(
                    hierarchy,
                    formal,
                    expanded,
                    decode_options=region_decode_options,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    fine_outputs = fine_model(expanded)
            hierarchy_outputs = baseline["outputs"]
            decoded = baseline["decoded"]
            baseline_visible = baseline["visible_mask"]
            assert isinstance(hierarchy_outputs, dict)
            assert isinstance(decoded, dict)
            assert isinstance(baseline_visible, torch.Tensor)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(
                    fine_outputs,
                    hierarchy_outputs,
                    expanded,
                    baseline_visible_mask=baseline_visible,
                    base_is_null_mask=decoded["base_is_null"],
                )
                losses = evidence_visibility_loss(
                    outputs,
                    fine_outputs,
                    hierarchy_outputs,
                    expanded,
                    baseline_visible_mask=baseline_visible,
                    **loss_options,
                )
                loss = losses["loss"] / accumulation
            if not torch.isfinite(loss):
                logger.warning(
                    "Skipping non-finite loss epoch=%d step=%d", epoch, step
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            running += float(losses["loss"].item())
            finite_steps += 1
            if step % accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable, config.optim.gradient_clip_norm
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            if step % max(1, config.runtime.log_every_steps) == 0:
                progress.set_postfix(
                    loss=f"{running / max(finite_steps, 1):.4f}"
                )

        dev_metrics = evaluate_evidence_visibility(
            model,
            fine_model,
            hierarchy,
            loaders["dev"],
            device,
            **evaluation_options,
        )
        history.append({"epoch": epoch, "dev": dev_metrics})
        logger.info(
            "Epoch %d train_loss=%.4f; dev: %s",
            epoch,
            running / max(finite_steps, 1),
            compact_metrics(dev_metrics),
        )
        current = selection_key(dev_metrics, primary, ties)
        if current > best_selection:
            best_selection = current
            best_epoch = epoch
            patience = 0
            atomic_save(checkpoint_payload(epoch, dev_metrics), best_path)
            logger.info("New best epoch %d: %s", epoch, current)
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
        "selection_metrics": [primary, *ties],
        "best_selection": list(best_selection),
        "history": history,
        "test": None,
        "engineering_train_cache_is_oof": bool(
            config.data.require_oof_train_cache
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
                "test": None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
