"""Train the dev-only M3.4A reliability ablations over a frozen chain."""

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

from gmner.data import PairedRecordCandidateCollator, PairedRecordCandidateDataset
from gmner.data.record_candidate_dataset import RecordCandidateDataset
from gmner.data.siglip2_region_cache import (
    Siglip2PairedRecordCollator,
    Siglip2PairedRecordDataset,
    Siglip2RegionFeatureCache,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from gmner.engine.siglip2_region_reliability_evaluator import (
    evaluate_siglip2_region_reliability,
    frozen_current_visibility_context,
)
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.fine_grounding_adapter_config import (
    load_fine_grounding_adapter_config,
)
from gmner.losses.siglip2_region_reliability_loss import (
    siglip2_region_reliability_loss,
)
from gmner.models.siglip2_region_reliability import (
    Siglip2RegionReliabilityHead,
)
from gmner.models.evidence_visibility import RegionEvidenceVisibilityHead
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed
from scripts.train_fine_grounding_adapter import (
    atomic_save,
    decode_options,
    load_frozen_models,
    resolve,
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


def _paired_dataset(config, root: Path, split: str):
    formal = RecordCandidateDataset(
        resolve(getattr(config.data, f"formal_{split}_cache"), root)
    )
    expanded = RecordCandidateDataset(
        resolve(getattr(config.data, f"expanded_{split}_cache"), root)
    )
    paired = PairedRecordCandidateDataset(formal, expanded)
    if config.model.feature_mode == "vinvl_only":
        return paired, PairedRecordCandidateCollator()
    cache_path = getattr(config.data, f"siglip2_{split}_cache")
    siglip2 = Siglip2RegionFeatureCache(resolve(cache_path, root))
    return (
        Siglip2PairedRecordDataset(
            paired,
            siglip2,
            verify_file_hashes=config.data.verify_siglip2_cache_hashes,
        ),
        Siglip2PairedRecordCollator(),
    )


def _base_paired(dataset) -> PairedRecordCandidateDataset:
    return dataset.paired if isinstance(dataset, Siglip2PairedRecordDataset) else dataset


def _siglip2_dataset(dataset) -> Siglip2PairedRecordDataset | None:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset if isinstance(dataset, Siglip2PairedRecordDataset) else None


def load_frozen_reliability_chain(config, root: Path, device: torch.device):
    fine_config = load_fine_grounding_adapter_config(
        resolve(config.frozen.fine_config, root)
    )
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
    evidence_config_path = resolve(
        config.frozen.evidence_visibility_config, root
    )
    evidence_config = load_evidence_visibility_config(evidence_config_path)
    if resolve(evidence_config.frozen.fine_config, root) != resolve(
        config.frozen.fine_config, root
    ):
        raise ValueError(
            "M3.4A and Evidence Visibility must use the same Fine config."
        )
    if resolve(evidence_config.frozen.fine_checkpoint, root) != fine_checkpoint_path:
        raise ValueError(
            "M3.4A and Evidence Visibility must use the same Fine checkpoint."
        )
    evidence_checkpoint_path = resolve(
        config.frozen.evidence_visibility_checkpoint, root
    )
    evidence_checkpoint = torch.load(evidence_checkpoint_path, map_location="cpu")
    evidence_model = RegionEvidenceVisibilityHead(evidence_config.model)
    evidence_model.load_state_dict(evidence_checkpoint["model_state_dict"])
    evidence_model.to(device).eval()
    fine_model.to(device).eval()
    hierarchy.to(device).eval()
    for frozen_model in (evidence_model, fine_model, hierarchy):
        for parameter in frozen_model.parameters():
            parameter.requires_grad = False
    if int(config.model.input_size) != int(fine_config.model.hidden_size):
        raise ValueError(
            "Reliability model.input_size must match Fine Adapter hidden_size: "
            f"{config.model.input_size} != {fine_config.model.hidden_size}."
        )
    model = Siglip2RegionReliabilityHead(config.model).to(device)
    return (
        model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        fine_checkpoint,
        evidence_checkpoint,
    )


def compact_metrics(metrics: dict) -> str:
    keys = (
        "hard_ab_auc",
        "hard_ab_auprc",
        "hard_ab_best_balanced_accuracy",
        "hard_ab_brier",
        "hard_ab_ece",
        "risk_best_net_correction",
        "risk_best_null_preservation_rate",
        "risk_best_promoted_fix_count",
        "go_no_go",
        "loss",
    )
    return ", ".join(
        f"{key}={float(metrics[key]):.4f}" for key in keys if key in metrics
    )


def _selection(metrics: dict, kind: str) -> tuple[float, ...]:
    if kind == "ab":
        return (
            float(metrics["hard_ab_auc"]),
            float(metrics["hard_ab_auprc"]),
            float(metrics["hard_ab_best_balanced_accuracy"]),
        )
    if kind == "risk":
        return (
            float(metrics["risk_best_net_correction"]),
            float(metrics["risk_best_promoted_fix_count"]),
            float(metrics["risk_best_null_preservation_rate"]),
            float(metrics["hard_ab_auc"]),
        )
    if kind == "calibrated":
        return (
            -float(metrics["hard_ab_brier"]),
            -float(metrics["hard_ab_ece"]),
            float(metrics["hard_ab_auc"]),
        )
    raise ValueError(kind)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_siglip2_region_reliability_config(args.config)
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.optim.num_epochs = max(1, int(args.num_epochs))
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(
        "gmner.siglip2_reliability_train", output_dir / "train.log"
    )
    set_seed(config.runtime.seed)

    datasets = {}
    collators = {}
    for split in ("train", "dev"):
        datasets[split], collators[split] = _paired_dataset(config, root, split)
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        fine_checkpoint,
        evidence_checkpoint,
    ) = load_frozen_reliability_chain(config, root, device)
    for split in ("train", "dev"):
        validate_fingerprints(
            _base_paired(datasets[split]),
            hierarchy_checkpoint=hierarchy_checkpoint,
            coarse_checkpoint=coarse_checkpoint,
            require_oof=(
                config.data.require_oof_train_cache and split == "train"
            ),
        )
    siglip2_datasets = {
        split: _siglip2_dataset(dataset) for split, dataset in datasets.items()
    }
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
            collate_fn=collators[split],
        )
        for split, dataset in datasets.items()
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
        **vars(config.evaluation),
    }
    logger.info(
        "M3.4A mode=%s records train/dev=%d/%d trainable=%d device=%s",
        config.model.feature_mode,
        len(datasets["train"]),
        len(datasets["dev"]),
        sum(parameter.numel() for parameter in trainable),
        device,
    )

    paths = {
        "ab": output_dir / "best_ab_model.pt",
        "risk": output_dir / "best_risk_model.pt",
        "calibrated": output_dir / "best_calibrated_model.pt",
    }
    history = []

    def checkpoint_payload(epoch: int, metrics: dict) -> dict:
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config.to_dict(),
            "fine_checkpoint_epoch": fine_checkpoint.get("epoch"),
            "fine_checkpoint_metrics": fine_checkpoint.get("metrics"),
            "evidence_visibility_checkpoint_epoch": evidence_checkpoint.get(
                "epoch"
            ),
            "evidence_visibility_checkpoint_metrics": evidence_checkpoint.get(
                "metrics"
            ),
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
        if siglip2_datasets["train"] is not None:
            payload["siglip2_train_build_signature"] = siglip2_datasets[
                "train"
            ].siglip2.manifest.get("build_signature")
            payload["siglip2_dev_build_signature"] = siglip2_datasets[
                "dev"
            ].siglip2.manifest.get("build_signature")
        return payload

    initial = evaluate_siglip2_region_reliability(
        model,
        evidence_model,
        fine_model,
        hierarchy,
        loaders["dev"],
        device,
        **evaluation_options,
    )
    history.append({"epoch": 0, "dev": initial})
    best = {kind: _selection(initial, kind) for kind in paths}
    best_epochs = {kind: 0 for kind in paths}
    for kind, path in paths.items():
        atomic_save(checkpoint_payload(0, initial), path)
    logger.info("Epoch 0 dev: %s", compact_metrics(initial))

    patience = 0
    region_options = {
        key: value
        for key, value in evaluation_options["decode_options"].items()
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    for epoch in range(1, config.optim.num_epochs + 1):
        model.train()
        evidence_model.eval()
        fine_model.eval()
        hierarchy.eval()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        finite_steps = 0
        progress = tqdm(
            loaders["train"],
            desc=f"M3.4A {config.model.feature_mode} {epoch}/{config.optim.num_epochs}",
        )
        for step, raw_batch in enumerate(progress, start=1):
            paired = move_paired_record_batch(raw_batch, device)
            formal = paired["formal"]
            expanded = paired["expanded"]
            siglip2 = paired.get("siglip2")
            with torch.no_grad():
                baseline = frozen_hierarchical_context(
                    hierarchy,
                    formal,
                    expanded,
                    decode_options=region_options,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    fine_outputs = fine_model(expanded)
            hierarchy_outputs = baseline["outputs"]
            decoded = baseline["decoded"]
            hierarchy_visible = baseline["visible_mask"]
            assert isinstance(hierarchy_outputs, dict)
            assert isinstance(decoded, dict)
            assert isinstance(hierarchy_visible, torch.Tensor)
            hierarchy_outputs, _, baseline_visible = (
                frozen_current_visibility_context(
                    evidence_model,
                    fine_outputs,
                    hierarchy_outputs,
                    expanded,
                    hierarchy_visible_mask=hierarchy_visible,
                    base_is_null_mask=decoded["base_is_null"],
                    decode_options=evaluation_options["decode_options"],
                )
            )
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
                    siglip2_features=siglip2,
                )
                losses = siglip2_region_reliability_loss(
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
        dev_metrics = evaluate_siglip2_region_reliability(
            model,
            evidence_model,
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
        improved = False
        for kind, path in paths.items():
            current = _selection(dev_metrics, kind)
            if current > best[kind]:
                best[kind] = current
                best_epochs[kind] = epoch
                atomic_save(checkpoint_payload(epoch, dev_metrics), path)
                logger.info("New best %s epoch %d: %s", kind, epoch, current)
                improved = True
        if improved:
            patience = 0
        else:
            patience += 1
            if (
                config.runtime.early_stop_patience > 0
                and patience >= config.runtime.early_stop_patience
            ):
                logger.info("Early stopping at epoch %d", epoch)
                break

    report = {
        "feature_mode": config.model.feature_mode,
        "best_epochs": best_epochs,
        "best_selection": {key: list(value) for key, value in best.items()},
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
                "feature_mode": config.model.feature_mode,
                "best_epochs": best_epochs,
                "checkpoints": {key: str(path.resolve()) for key, path in paths.items()},
                "test": None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
