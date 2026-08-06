"""Train the hierarchical frozen-Stage1 record verifier."""

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

from gmner.data import HierarchicalRecordCandidateCollator, RecordCandidateDataset
from gmner.data.hierarchical_record_candidate_collator import (
    missing_hierarchical_cache_fields,
)
from gmner.engine.hierarchical_record_verifier_evaluator import (
    evaluate_hierarchical_record_verifier,
)
from gmner.hierarchical_record_verifier_config import (
    load_hierarchical_record_verifier_config,
)
from gmner.losses.hierarchical_record_candidate_loss import (
    hierarchical_record_candidate_loss,
)
from gmner.models.hierarchical_record_verifier import HierarchicalRecordVerifier
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def make_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = int(total_steps * max(0.0, min(float(warmup_ratio), 1.0)))

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))

    return LambdaLR(optimizer, factor)


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def compact_metrics(metrics: dict[str, float]) -> str:
    keys = (
        "span_f1",
        "entity_f1",
        "eeg_f1",
        "gmner_score",
        "stage1_gold_span_accept_rate",
        "visibility_final_visible_recall",
        "region_override_count",
        "region_override_precision",
        "override_fix_count",
        "override_damage_count",
        "override_net_correction",
        "action_controller_executed_count",
        "action_controller_fix_count",
        "action_controller_damage_count",
        "action_controller_neutral_count",
        "action_controller_net_correction",
        "action_policy_fixable_top1_recall",
        "action_keep_correct_preservation_rate",
        "action_controller_cumulative_max_net_correction",
        "action_controller_cumulative_max_threshold",
        "visible_corrected",
        "visible_damaged",
        "null_corrected",
        "null_damaged",
        "net_corrections",
        "loss",
    )
    return ", ".join(
        f"{key}={metrics[key]:.4f}" for key in keys if key in metrics
    )


def checkpoint_selection_key(
    metrics: dict[str, float],
    primary: str,
    tie_breakers: list[str],
) -> tuple[float, ...]:
    """Build a deterministic lexicographic checkpoint objective."""

    names = [primary, *tie_breakers]
    missing = [name for name in names if name not in metrics]
    if missing:
        raise KeyError(f"Unknown checkpoint metrics: {missing}")
    return tuple(float(metrics[name]) for name in names)


def _loss_options(config, device: torch.device) -> dict:
    values = vars(config.loss).copy()
    values["source_weights"] = torch.tensor(
        values["source_weights"], device=device
    )
    values.update(
        {
            "action_top_k": config.decode.action_top_k,
            "action_enable_visibility_correction": (
                config.decode.enable_visibility_correction
            ),
            "action_visible_from_null_threshold": (
                config.decode.visible_from_null_threshold
            ),
            "action_null_from_visible_threshold": (
                config.decode.null_from_visible_threshold
            ),
        }
    )
    return values


def _evaluation_options(config) -> dict:
    return {
        "entity_threshold": config.decode.entity_threshold,
        "decode_strategy": config.decode.strategy,
        "stage1_spans_only": config.decode.stage1_spans_only,
        "enable_visibility_correction": config.decode.enable_visibility_correction,
        "enable_region_override": config.decode.enable_region_override,
        "visible_from_null_threshold": config.decode.visible_from_null_threshold,
        "null_from_visible_threshold": config.decode.null_from_visible_threshold,
        "region_override_mode": config.decode.region_override_mode,
        "region_override_logit_margin": config.decode.region_override_logit_margin,
        "region_override_probability_margin": config.decode.region_override_probability_margin,
        "override_damage_cost": config.decode.override_damage_cost,
        "override_utility_threshold": config.decode.override_utility_threshold,
        "include_override_risk_curve": config.decode.include_override_risk_curve,
        "enable_action_controller": config.decode.enable_action_controller,
        "action_top_k": config.decode.action_top_k,
        "action_execution_margin": config.decode.action_execution_margin,
        "include_action_risk_curve": config.decode.include_action_risk_curve,
    }


def _configured_dataset_paths(config) -> dict[str, str]:
    if config.runtime.evaluate_test_after_training and not config.data.test_cache:
        raise ValueError(
            "runtime.evaluate_test_after_training=true requires data.test_cache."
        )
    paths = {
        "train": config.data.train_cache,
        "dev": config.data.dev_cache,
    }
    if config.runtime.evaluate_test_after_training:
        paths["test"] = config.data.test_cache
    return paths


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_hierarchical_record_verifier_config(args.config)
    if config.decode.enable_action_controller:
        if not config.model.enable_action_controller:
            raise ValueError(
                "decode.enable_action_controller requires "
                "model.enable_action_controller=true."
            )
        if config.decode.enable_region_override:
            raise ValueError(
                "The unified action controller requires "
                "decode.enable_region_override=false."
            )
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.optim.num_epochs = max(1, int(args.num_epochs))
    if args.init_checkpoint:
        config.runtime.init_checkpoint = args.init_checkpoint
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("gmner.hierarchical_record_train", output_dir / "train.log")
    set_seed(config.runtime.seed)

    dataset_paths = _configured_dataset_paths(config)
    datasets = {
        name: RecordCandidateDataset(resolve(path, root))
        for name, path in dataset_paths.items()
    }
    for name, dataset in datasets.items():
        first_record = dataset.records[0] if dataset.records else {}
        missing = missing_hierarchical_cache_fields(first_record)
        if missing:
            raise ValueError(
                f"{name} cache {dataset.path} lacks hierarchical fields "
                f"{missing}; verify the config path and rebuild it with the updated "
                "build_record_candidate_cache.py."
            )
        if int(dataset.metadata.get("format_version", 0)) < 2:
            logger.warning(
                "%s cache metadata reports format_version=%s, but all hierarchical "
                "fields are present; accepting the cache by capability.",
                name,
                dataset.metadata.get("format_version"),
            )
        summary = dict(dataset.metadata.get("summary") or {})
        bypass = dict(summary.get("stage1_bypass") or {})
        logger.info(
            "%s cache bypass: MNER=%.4f EEG=%.4f GMNER=%.4f",
            name,
            float((bypass.get("mner") or {}).get("f1", 0.0)),
            float((bypass.get("eeg") or {}).get("f1", 0.0)),
            float((bypass.get("gmner") or {}).get("f1", 0.0)),
        )
    train_oof = dict(datasets["train"].metadata.get("oof") or {})
    if config.data.require_oof_train_cache and not bool(train_oof.get("enabled")):
        raise ValueError(
            "This configuration requires a merged OOF train cache. Build fold-level "
            "candidate caches and merge them before correction-head training."
        )
    evaluation_names = [name for name in ("dev", "test") if name in datasets]
    evaluation_stage1_hashes = {
        str(datasets[name].metadata.get("stage1_checkpoint_sha256", ""))
        for name in evaluation_names
    }
    if len(evaluation_stage1_hashes) != 1:
        raise ValueError("Evaluation caches do not share a Stage1 fingerprint.")
    evaluation_stage1_hash = next(iter(evaluation_stage1_hashes))
    if not config.data.require_oof_train_cache:
        train_stage1_hash = str(
            datasets["train"].metadata.get("stage1_checkpoint_sha256", "")
        )
        if train_stage1_hash != evaluation_stage1_hash:
            raise ValueError(
                "Hierarchical caches do not share a Stage1 fingerprint. A different "
                "train fingerprint is allowed only for an explicitly required OOF cache."
            )
    evaluation_candidate_hashes = {
        str(datasets[name].metadata.get("candidate_config_sha256", ""))
        for name in evaluation_names
    }
    if len(evaluation_candidate_hashes) != 1:
        raise ValueError("Evaluation caches do not share a candidate fingerprint.")
    evaluation_candidate_hash = next(iter(evaluation_candidate_hashes))
    train_candidate_hash = str(
        datasets["train"].metadata.get("candidate_config_sha256", "")
    )
    if train_candidate_hash != evaluation_candidate_hash:
        candidate_specs = []
        for dataset in datasets.values():
            spec = dict(dataset.metadata.get("candidate_config") or {})
            spec.pop("cache_format_version", None)
            candidate_specs.append(json.dumps(spec, sort_keys=True))
        if len(set(candidate_specs)) != 1:
            raise ValueError(
                "Hierarchical caches use different candidate configurations."
            )
        logger.warning(
            "Candidate fingerprints differ only by cache format metadata; "
            "accepting semantically identical candidate configurations."
        )
    cache_hidden = int(
        datasets["train"].metadata.get("hidden_size", config.model.input_size)
    )
    if cache_hidden != int(config.model.input_size):
        raise ValueError(
            f"Verifier input_size={config.model.input_size}, cache hidden_size={cache_hidden}."
        )
    if args.max_train_records is not None and args.max_train_records < len(datasets["train"]):
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
    model = HierarchicalRecordVerifier(config.model).to(device)
    if config.runtime.init_checkpoint:
        init_path = resolve(config.runtime.init_checkpoint, root)
        initial = torch.load(init_path, map_location="cpu")
        incompatible = model.load_state_dict(
            initial["model_state_dict"], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        optional_prefixes = (
            "override_utility_head.",
            "action_real_scalar_projection.",
            "action_real_head.",
            "action_null_head.",
        )
        allowed_missing = all(key.startswith(optional_prefixes) for key in missing)
        if unexpected or (missing and not allowed_missing):
            raise ValueError(
                "Initialization checkpoint is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
        logger.info(
            "Initialized from %s (missing=%d, unexpected=%d)",
            init_path,
            len(missing),
            len(unexpected),
        )
    if (
        config.runtime.train_override_utility_only
        and config.runtime.train_action_controller_only
    ):
        raise ValueError(
            "Only one optional correction head can be trained at a time."
        )
    if config.runtime.train_override_utility_only:
        if model.override_utility_head is None:
            raise ValueError(
                "train_override_utility_only requires model.enable_override_utility=true."
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.override_utility_head.parameters():
            parameter.requires_grad = True
    if config.runtime.train_action_controller_only:
        if model.action_real_head is None or model.action_null_head is None:
            raise ValueError(
                "train_action_controller_only requires "
                "model.enable_action_controller=true."
            )
        assert model.action_real_scalar_projection is not None
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in (
            model.action_real_scalar_projection,
            model.action_real_head,
            model.action_null_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after module freezing.")
    optimizer = AdamW(
        trainable_parameters,
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    accumulation = max(1, int(config.optim.gradient_accumulation_steps))
    total_steps = (
        math.ceil(len(loaders["train"]) / accumulation) * config.optim.num_epochs
    )
    scheduler = make_scheduler(optimizer, total_steps, config.optim.warmup_ratio)
    amp_enabled = bool(config.runtime.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loss_options = _loss_options(config, device)
    evaluation_options = _evaluation_options(config)
    record_counts = "/".join(
        f"{name}={len(datasets[name])}" for name in ("train", "dev", "test")
        if name in datasets
    )
    logger.info(
        "Records %s; parameters=%d; device=%s",
        record_counts,
        sum(parameter.numel() for parameter in model.parameters()),
        device,
    )
    logger.info(
        "Trainable parameters: %d/%d%s",
        sum(parameter.numel() for parameter in trainable_parameters),
        sum(parameter.numel() for parameter in model.parameters()),
        (
            " (override utility only)"
            if config.runtime.train_override_utility_only
            else " (action controller only)"
            if config.runtime.train_action_controller_only
            else ""
        ),
    )

    best_value = float("-inf")
    best_selection: tuple[float, ...] | None = None
    best_epoch = 0
    patience = 0
    best_path = output_dir / "best_model.pt"
    history: list[dict] = []
    if config.runtime.train_action_controller_only:
        initial_metrics = evaluate_hierarchical_record_verifier(
            model,
            loaders["dev"],
            device,
            loss_options=loss_options,
            **evaluation_options,
        )
        best_selection = checkpoint_selection_key(
            initial_metrics,
            config.runtime.save_best_metric,
            config.runtime.save_best_tie_breakers,
        )
        best_value = best_selection[0]
        history.append({"epoch": 0, "dev": initial_metrics})
        atomic_save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": 0,
                "metrics": initial_metrics,
                "config": config.to_dict(),
                "stage1_checkpoint_sha256": evaluation_stage1_hash,
                "train_oof": train_oof,
                "candidate_config_sha256": evaluation_candidate_hash,
            },
            best_path,
        )
        logger.info(
            "Epoch 0 safe KEEP baseline: %s",
            compact_metrics(initial_metrics),
        )
    for epoch in range(1, config.optim.num_epochs + 1):
        if config.runtime.train_override_utility_only:
            model.eval()
            assert model.override_utility_head is not None
            model.override_utility_head.train()
        elif config.runtime.train_action_controller_only:
            model.eval()
            assert model.action_real_scalar_projection is not None
            assert model.action_real_head is not None
            assert model.action_null_head is not None
            model.action_real_scalar_projection.train()
            model.action_real_head.train()
            model.action_null_head.train()
        else:
            model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        finite_steps = 0
        progress = tqdm(
            loaders["train"],
            desc=f"Hierarchical verifier {epoch}/{config.optim.num_epochs}",
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
                losses = hierarchical_record_candidate_loss(
                    outputs, batch, **loss_options
                )
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
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before_step:
                    scheduler.step()
            if step % max(1, config.runtime.log_every_steps) == 0:
                progress.set_postfix(loss=f"{running / max(finite_steps, 1):.4f}")

        logger.info(
            "Epoch %d training loss: %.4f", epoch, running / max(finite_steps, 1)
        )
        dev_metrics = evaluate_hierarchical_record_verifier(
            model,
            loaders["dev"],
            device,
            loss_options=loss_options,
            **evaluation_options,
        )
        logger.info("Epoch %d dev: %s", epoch, compact_metrics(dev_metrics))
        history.append({"epoch": epoch, "dev": dev_metrics})
        metric_name = config.runtime.save_best_metric
        current_selection = checkpoint_selection_key(
            dev_metrics,
            metric_name,
            config.runtime.save_best_tie_breakers,
        )
        current = current_selection[0]
        if best_selection is None or current_selection > best_selection:
            best_value = current
            best_selection = current_selection
            best_epoch = epoch
            patience = 0
            atomic_save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": dev_metrics,
                    "config": config.to_dict(),
                    "stage1_checkpoint_sha256": evaluation_stage1_hash,
                    "train_oof": train_oof,
                    "candidate_config_sha256": evaluation_candidate_hash,
                },
                best_path,
            )
            logger.info(
                "New best checkpoint at epoch %d with %s=%s",
                epoch,
                "/".join(
                    [metric_name, *config.runtime.save_best_tie_breakers]
                ),
                "/".join(f"{value:.4f}" for value in current_selection),
            )
        else:
            patience += 1
            if (
                config.runtime.early_stop_patience > 0
                and patience >= config.runtime.early_stop_patience
            ):
                logger.info("Early stopping at epoch %d", epoch)
                break

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_metrics = None
    if config.runtime.evaluate_test_after_training:
        if "test" not in loaders:
            raise RuntimeError("Test evaluation requested without a test dataloader.")
        test_metrics = evaluate_hierarchical_record_verifier(
            model,
            loaders["test"],
            device,
            loss_options=loss_options,
            **evaluation_options,
        )
        logger.info(
            "Best epoch: %d; test: %s", best_epoch, compact_metrics(test_metrics)
        )
    else:
        logger.info(
            "Best epoch: %d; skipped automatic test evaluation. Calibrate on dev "
            "before the one-time test run.",
            best_epoch,
        )
    report = {
        "best_epoch": best_epoch,
        "best_dev": best_value,
        "best_selection": list(best_selection or ()),
        "selection_metrics": [
            config.runtime.save_best_metric,
            *config.runtime.save_best_tie_breakers,
        ],
        "test": test_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"best_epoch": best_epoch, "test": test_metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
