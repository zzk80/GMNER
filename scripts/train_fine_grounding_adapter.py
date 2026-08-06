"""Train the M3.2 correction-preservation fine grounding adapter."""

from __future__ import annotations

import argparse
import copy
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
from gmner.engine.fine_grounding_adapter_evaluator import (
    evaluate_fine_grounding_adapter,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from gmner.fine_grounding_adapter_config import (
    load_fine_grounding_adapter_config,
)
from gmner.hierarchical_record_verifier_config import (
    load_hierarchical_record_verifier_config,
)
from gmner.losses.fine_grounding_adapter_loss import (
    fine_grounding_adapter_loss,
)
from gmner.models.coarse_region_selector import (
    CoarseRegionSelectorConfig,
    RecallPreservingCoarseSelector,
)
from gmner.models.fine_grounding_adapter import (
    CorrectionPreservationGroundingAdapter,
    FineGroundingAdapterConfig,
)
from gmner.models.hierarchical_record_verifier import HierarchicalRecordVerifier
from gmner.models.protected_downstream import ProtectedFineResidual
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-dev-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--protected-teacher-checkpoint", default=None)
    parser.add_argument("--allow-protected-cache-transfer", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def make_scheduler(
    optimizer: AdamW, total_steps: int, warmup_ratio: float
) -> LambdaLR:
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


def decode_options(config) -> dict:
    decode = config.decode
    return {
        "entity_threshold": decode.entity_threshold,
        "decode_strategy": decode.strategy,
        "stage1_spans_only": decode.stage1_spans_only,
        "enable_visibility_correction": decode.enable_visibility_correction,
        "enable_region_override": decode.enable_region_override,
        "visible_from_null_threshold": decode.visible_from_null_threshold,
        "null_from_visible_threshold": decode.null_from_visible_threshold,
        "region_override_mode": decode.region_override_mode,
        "region_override_logit_margin": decode.region_override_logit_margin,
        "region_override_probability_margin": (
            decode.region_override_probability_margin
        ),
        "override_damage_cost": decode.override_damage_cost,
        "override_utility_threshold": decode.override_utility_threshold,
        "enable_action_controller": decode.enable_action_controller,
        "action_top_k": decode.action_top_k,
        "action_execution_margin": decode.action_execution_margin,
    }


def load_frozen_models(config, root: Path, device: torch.device):
    hierarchy_config = load_hierarchical_record_verifier_config(
        resolve(config.frozen.hierarchical_config, root)
    )
    hierarchy_checkpoint_path = resolve(
        config.frozen.hierarchical_checkpoint, root
    )
    hierarchy_checkpoint = torch.load(
        hierarchy_checkpoint_path, map_location="cpu"
    )
    hierarchy = HierarchicalRecordVerifier(hierarchy_config.model)
    hierarchy.load_state_dict(hierarchy_checkpoint["model_state_dict"])
    hierarchy.to(device).eval()
    for parameter in hierarchy.parameters():
        parameter.requires_grad = False

    coarse_checkpoint_path = resolve(config.frozen.coarse_checkpoint, root)
    coarse_checkpoint = torch.load(coarse_checkpoint_path, map_location="cpu")
    coarse_config = CoarseRegionSelectorConfig(
        **coarse_checkpoint["config"]["model"]
    )
    coarse = RecallPreservingCoarseSelector(coarse_config)
    coarse.load_state_dict(coarse_checkpoint["model_state_dict"])
    coarse.eval()
    for parameter in coarse.parameters():
        parameter.requires_grad = False
    model = CorrectionPreservationGroundingAdapter(config.model, coarse).to(device)
    return (
        model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
    )


def validate_fingerprints(
    paired: PairedRecordCandidateDataset,
    *,
    hierarchy_checkpoint: dict,
    coarse_checkpoint: dict,
    require_oof: bool,
) -> None:
    formal_metadata = paired.formal.metadata
    expanded_metadata = paired.expanded.metadata
    expected_hierarchy_stage1 = str(
        hierarchy_checkpoint.get("stage1_checkpoint_sha256") or ""
    )
    expected_hierarchy_candidates = str(
        hierarchy_checkpoint.get("candidate_config_sha256") or ""
    )
    expected_coarse_stage1 = str(
        coarse_checkpoint.get("stage1_checkpoint_sha256") or ""
    )
    expected_coarse_candidates = str(
        coarse_checkpoint.get("candidate_config_sha256") or ""
    )
    checks = (
        (
            str(formal_metadata.get("stage1_checkpoint_sha256") or ""),
            expected_hierarchy_stage1,
            "formal Stage1",
        ),
        (
            str(formal_metadata.get("candidate_config_sha256") or ""),
            expected_hierarchy_candidates,
            "formal candidate config",
        ),
        (
            str(expanded_metadata.get("stage1_checkpoint_sha256") or ""),
            expected_coarse_stage1,
            "expanded Stage1",
        ),
        (
            str(expanded_metadata.get("candidate_config_sha256") or ""),
            expected_coarse_candidates,
            "expanded candidate config",
        ),
    )
    for actual, expected, name in checks:
        if expected and actual != expected:
            raise ValueError(
                f"{name} fingerprint mismatch: expected {expected}, found {actual}."
            )
    if require_oof and not bool(formal_metadata.get("oof_heldout", False)):
        raise ValueError("Formal training requires an OOF train cache.")


def compact_metrics(metrics: dict[str, float]) -> str:
    keys = (
        "visible_net_correction",
        "gmner_score",
        "baseline_gmner_score",
        "gmner_delta",
        "base_wrong_corrected",
        "base_correct_damaged",
        "base_correct_preservation_rate",
        "promoted_gold_fine_correct",
        "promoted_gold_recovery_rate",
        "loss",
    )
    return ", ".join(
        f"{key}={metrics[key]:.4f}" for key in keys if key in metrics
    )


def selection_key(metrics: dict[str, float], primary: str, ties: list[str]) -> tuple:
    names = [primary, *ties]
    missing = [name for name in names if name not in metrics]
    if missing:
        raise KeyError(f"Unknown checkpoint metrics: {missing}")
    return tuple(float(metrics[name]) for name in names)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_fine_grounding_adapter_config(args.config)
    if args.seed is not None:
        config.runtime.seed = int(args.seed)
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.optim.num_epochs = max(1, int(args.num_epochs))
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("gmner.fine_grounding_train", output_dir / "train.log")
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
        if str(config.runtime.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
    ) = load_frozen_models(config, root, device)
    protected_mode = args.protected_teacher_checkpoint is not None
    teacher_checkpoint = None
    if protected_mode != bool(args.allow_protected_cache_transfer):
        raise ValueError(
            "Protected Fine training requires both --protected-teacher-checkpoint "
            "and --allow-protected-cache-transfer."
        )
    if protected_mode:
        teacher_checkpoint = torch.load(
            resolve(args.protected_teacher_checkpoint, root), map_location="cpu"
        )
        teacher_config = FineGroundingAdapterConfig(
            **teacher_checkpoint["config"]["model"]
        )
        teacher = CorrectionPreservationGroundingAdapter(
            teacher_config,
            copy.deepcopy(model.coarse_selector),
        )
        teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
        model = ProtectedFineResidual(teacher, model).to(device)
    else:
        validate_fingerprints(
            datasets["train"],
            hierarchy_checkpoint=hierarchy_checkpoint,
            coarse_checkpoint=coarse_checkpoint,
            require_oof=config.data.require_oof_train_cache,
        )
        validate_fingerprints(
            datasets["dev"],
            hierarchy_checkpoint=hierarchy_checkpoint,
            coarse_checkpoint=coarse_checkpoint,
            require_oof=False,
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
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
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
    loss_options["detector_reference_budget"] = (
        config.model.detector_reference_budget
    )
    evaluation_options = {
        "decode_options": decode_options(hierarchy_config),
        "loss_options": loss_options,
    }
    logger.info(
        "Records train/dev=%d/%d; trainable=%d; frozen hierarchy/coarse; device=%s",
        len(datasets["train"]),
        len(datasets["dev"]),
        sum(parameter.numel() for parameter in trainable),
        device,
    )
    for split, dataset in datasets.items():
        source = dataset.dataset if isinstance(dataset, Subset) else dataset
        summary = getattr(source, "alignment_summary", None)
        if summary:
            logger.info("%s span alignment: %s", split, summary)

    primary = config.runtime.save_best_metric
    ties = list(config.runtime.save_best_tie_breakers)
    best_path = output_dir / "best_model.pt"
    history: list[dict] = []
    initial = evaluate_fine_grounding_adapter(
        model, hierarchy, loaders["dev"], device, **evaluation_options
    )
    if protected_mode and abs(float(initial["gmner_delta"])) > 1e-12:
        raise RuntimeError(
            "Zero-initialized protected Fine residual must reproduce its "
            f"Teacher; observed delta={initial['gmner_delta']}."
        )
    best_selection = selection_key(initial, primary, ties)
    best_epoch = 0
    history.append({"epoch": 0, "dev": initial})
    atomic_save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "metrics": initial,
            "config": config.to_dict(),
            "hierarchy_stage1_checkpoint_sha256": hierarchy_checkpoint.get(
                "stage1_checkpoint_sha256"
            ),
            "formal_candidate_config_sha256": hierarchy_checkpoint.get(
                "candidate_config_sha256"
            ),
            "expanded_candidate_config_sha256": coarse_checkpoint.get(
                "candidate_config_sha256"
            ),
            "kind": (
                "protected_fine_residual" if protected_mode else "fine_grounding_adapter"
            ),
            "protected_cache_transfer": protected_mode,
            "protected_teacher_config": (
                teacher_checkpoint.get("config") if teacher_checkpoint else None
            ),
        },
        best_path,
    )
    logger.info("Epoch 0 prior-only dev: %s", compact_metrics(initial))
    patience = 0

    for epoch in range(1, config.optim.num_epochs + 1):
        model.train()
        hierarchy.eval()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        finite_steps = 0
        progress = tqdm(
            loaders["train"],
            desc=f"Fine grounding {epoch}/{config.optim.num_epochs}",
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
                    decode_options={
                        key: value
                        for key, value in evaluation_options[
                            "decode_options"
                        ].items()
                        if key
                        not in {
                            "entity_threshold",
                            "decode_strategy",
                            "stage1_spans_only",
                        }
                    },
                )
            baseline_indices = baseline["expanded_region_indices"]
            baseline_visible = baseline["visible_mask"]
            assert isinstance(baseline_indices, torch.Tensor)
            assert isinstance(baseline_visible, torch.Tensor)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(expanded)
                training_baseline_indices = outputs.get(
                    "protected_reference_region_index", baseline_indices
                ).long()
                losses = fine_grounding_adapter_loss(
                    outputs,
                    expanded,
                    baseline_region_indices=training_baseline_indices,
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

        dev_metrics = evaluate_fine_grounding_adapter(
            model, hierarchy, loaders["dev"], device, **evaluation_options
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
            atomic_save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": dev_metrics,
                    "config": config.to_dict(),
                    "hierarchy_stage1_checkpoint_sha256": hierarchy_checkpoint.get(
                        "stage1_checkpoint_sha256"
                    ),
                    "formal_candidate_config_sha256": hierarchy_checkpoint.get(
                        "candidate_config_sha256"
                    ),
                    "expanded_candidate_config_sha256": coarse_checkpoint.get(
                        "candidate_config_sha256"
                    ),
                    "kind": (
                        "protected_fine_residual"
                        if protected_mode
                        else "fine_grounding_adapter"
                    ),
                    "protected_cache_transfer": protected_mode,
                    "protected_teacher_config": (
                        teacher_checkpoint.get("config")
                        if teacher_checkpoint
                        else None
                    ),
                },
                best_path,
            )
            logger.info(
                "New best epoch %d: %s",
                epoch,
                "/".join(f"{value:.4f}" for value in current),
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
