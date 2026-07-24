"""Train M3.6A KEEP/TO_NULL/TO_VISIBLE over the frozen formal chain."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.engine.layered_action_verifier_evaluator import (
    evaluate_layered_action_verifier,
    frozen_layered_action_features,
)
from gmner.data.null_release_oof_cache import (
    load_full_chain_oof_cache,
    move_null_release_context_batch,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    move_paired_record_batch,
)
from gmner.layered_action_verifier_config import (
    load_layered_action_verifier_config,
)
from gmner.losses.layered_action_verifier_loss import (
    layered_action_verifier_loss,
)
from gmner.models.layered_action_verifier import (
    ACTION_MODE_NULL_RELEASE_ONLY,
    LayeredActionVerifier,
)
from gmner.models.null_release_verifier import NullReleaseVerifier
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed
from scripts.train_fine_grounding_adapter import (
    atomic_save,
    decode_options,
    resolve,
    validate_fingerprints,
)
from scripts.train_siglip2_region_reliability import (
    _base_paired,
    _paired_dataset,
    _siglip2_dataset,
    load_frozen_reliability_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-dev-records", type=int, default=None)
    return parser.parse_args()


def make_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup = int(total_steps * max(0.0, min(float(warmup_ratio), 1.0)))

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(step, 1) / warmup
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))

    return LambdaLR(optimizer, factor)


def _selection(metrics: dict, config) -> tuple[float, ...]:
    names = [
        config.runtime.save_best_metric,
        *config.runtime.save_best_tie_breakers,
    ]
    return tuple(float(metrics.get(name, float("-inf"))) for name in names)


def _compact(metrics: dict) -> str:
    names = (
        "gmner_score",
        "eeg_f1",
        "entity_f1",
        "gmner_net_correction",
        "to_null_net_correction",
        "to_visible_net_correction",
        "null_release_net_correction",
        "region_switch_net_correction",
        "keep_correct_preservation_rate",
        "layer1_accuracy",
        "layer2_top4_accuracy",
        "action_cumulative_max_net_correction",
        "loss",
    )
    return ", ".join(
        f"{name}={float(metrics[name]):.4f}" for name in names if name in metrics
    )


def _assert_epoch0_identity(metrics: dict, config, *, full_dev: bool) -> None:
    if float(metrics.get("epoch0_identity_pass", 0.0)) != 1.0:
        raise RuntimeError(
            "M3.6A epoch 0 changed the frozen deployment chain: "
            f"executed={metrics.get('executed_count')}, "
            f"changed={metrics.get('prediction_changed_count')}, "
            f"record_identity={metrics.get('record_prediction_identity_rate')}, "
            f"gmner_delta={metrics.get('triple_f1_delta')}."
        )
    expected = config.evaluation.expected_baseline_gmner
    if full_dev and expected is not None:
        actual = float(metrics["baseline_gmner_score"])
        tolerance = float(config.evaluation.expected_baseline_tolerance)
        if abs(actual - float(expected)) > tolerance:
            raise RuntimeError(
                "Frozen-chain dev baseline drifted before M3.6A training: "
                f"expected={expected:.6f}, actual={actual:.6f}."
            )


def _build_model(config):
    if config.action_mode == ACTION_MODE_NULL_RELEASE_ONLY:
        return NullReleaseVerifier(config)
    return LayeredActionVerifier(config)


def main(*, required_action_mode: str | None = None) -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_layered_action_verifier_config(args.config)
    if (
        required_action_mode is not None
        and config.model.action_mode != required_action_mode
    ):
        raise ValueError(
            f"This entry point requires model.action_mode={required_action_mode!r}, "
            f"found {config.model.action_mode!r}."
        )
    reliability_config_path = resolve(config.frozen.reliability_config, root)
    reliability_config = load_siglip2_region_reliability_config(reliability_config_path)
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.optim.num_epochs = max(1, int(args.num_epochs))
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("gmner.layered_action_train", output_dir / "train.log")
    set_seed(config.runtime.seed)

    full_chain_oof = None
    if config.oof.require_full_chain_oof:
        full_chain_oof = load_full_chain_oof_cache(
            resolve(config.oof.train_feature_cache, root),
            expected_num_folds=config.oof.expected_num_folds,
            expected_records=config.oof.expected_records,
            require_reliability=config.model.use_region_reliability,
        )

    datasets = {}
    collators = {}
    online_splits = ("dev",) if full_chain_oof is not None else ("train", "dev")
    for split in online_splits:
        datasets[split], collators[split] = _paired_dataset(
            reliability_config, root, split
        )
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    (
        reliability_model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        fine_checkpoint,
        evidence_checkpoint,
    ) = load_frozen_reliability_chain(reliability_config, root, device)
    reliability_checkpoint_path = resolve(config.frozen.reliability_checkpoint, root)
    reliability_checkpoint = torch.load(reliability_checkpoint_path, map_location="cpu")
    reliability_model.load_state_dict(reliability_checkpoint["model_state_dict"])
    reliability_model.to(device).eval()
    for frozen in (reliability_model, evidence_model, fine_model, hierarchy):
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad = False
    if int(config.model.input_size) != int(reliability_config.model.input_size):
        raise ValueError(
            "M3.6A model.input_size must match the frozen reliability/Fine "
            f"state size: {config.model.input_size} != "
            f"{reliability_config.model.input_size}."
        )
    if not config.model.use_region_reliability:
        logger.warning(
            "Region reliability is disabled; M3.6A will use only frozen Fine "
            "and Evidence Visibility features."
        )
    model = _build_model(config.model).to(device)

    for split in online_splits:
        validate_fingerprints(
            _base_paired(datasets[split]),
            hierarchy_checkpoint=hierarchy_checkpoint,
            coarse_checkpoint=coarse_checkpoint,
            require_oof=(
                reliability_config.data.require_oof_train_cache and split == "train"
            ),
        )
    siglip2_datasets = {
        split: _siglip2_dataset(dataset) for split, dataset in datasets.items()
    }
    full_dev = args.max_dev_records is None
    if args.max_train_records is not None and full_chain_oof is None:
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
            num_workers=reliability_config.data.num_workers,
            collate_fn=collators[split],
        )
        for split, dataset in datasets.items()
    }
    if full_chain_oof is not None:
        cached_batches = list(full_chain_oof["batches"])
        if args.max_train_records is not None:
            selected_batches = []
            selected_records = 0
            for batch in cached_batches:
                if selected_records >= max(1, args.max_train_records):
                    break
                selected_batches.append(batch)
                selected_records += len(batch["record_ids"])
            cached_batches = selected_batches
        loaders["train"] = cached_batches
        train_record_count = sum(len(batch["record_ids"]) for batch in cached_batches)
    else:
        train_record_count = len(datasets["train"])
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable,
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
    decode = decode_options(hierarchy_config)
    loss_options = vars(config.loss).copy()
    evaluation_options = {
        "decode_options": decode,
        "loss_options": loss_options,
        "execution_margin": config.evaluation.execution_margin,
        "include_risk_curve": config.evaluation.include_risk_curve,
        "identity_tolerance": config.evaluation.identity_tolerance,
        "minimum_keep_preservation_rate": (
            config.evaluation.minimum_keep_preservation_rate
        ),
        "minimum_net_correction": config.evaluation.minimum_net_correction,
    }
    logger.info(
        "M3.6A records train/dev=%d/%d trainable=%d feature_mode=%s "
        "action_mode=%s device=%s",
        train_record_count,
        len(datasets["dev"]),
        sum(parameter.numel() for parameter in trainable),
        reliability_config.model.feature_mode,
        config.model.action_mode,
        device,
    )
    if full_chain_oof is None:
        logger.warning(
            "M3.6A is using in-sample engineering train caches. Treat dev as "
            "an architecture check; formal claims require aligned OOF caches."
        )
    else:
        logger.info(
            "Validated 10-fold full-chain OOF cache: %s records=%d sha256=%s",
            full_chain_oof["path"],
            full_chain_oof["records"],
            full_chain_oof["sha256"],
        )

    def checkpoint_payload(epoch: int, metrics: dict) -> dict:
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config.to_dict(),
            "reliability_config": reliability_config.to_dict(),
            "reliability_checkpoint_epoch": reliability_checkpoint.get("epoch"),
            "fine_checkpoint_epoch": fine_checkpoint.get("epoch"),
            "evidence_visibility_checkpoint_epoch": evidence_checkpoint.get("epoch"),
            "hierarchy_stage1_checkpoint_sha256": hierarchy_checkpoint.get(
                "stage1_checkpoint_sha256"
            ),
            "formal_candidate_config_sha256": hierarchy_checkpoint.get(
                "candidate_config_sha256"
            ),
            "expanded_candidate_config_sha256": coarse_checkpoint.get(
                "candidate_config_sha256"
            ),
            "full_chain_oof": bool(full_chain_oof is not None),
        }
        if full_chain_oof is not None:
            payload["full_chain_oof_cache"] = str(full_chain_oof["path"].resolve())
            payload["full_chain_oof_cache_sha256"] = full_chain_oof["sha256"]
            payload["full_chain_oof_metadata"] = full_chain_oof["metadata"]
        elif siglip2_datasets["train"] is not None:
            payload["siglip2_train_build_signature"] = siglip2_datasets[
                "train"
            ].siglip2.manifest.get("build_signature")
            payload["siglip2_dev_build_signature"] = siglip2_datasets[
                "dev"
            ].siglip2.manifest.get("build_signature")
        return payload

    initial = evaluate_layered_action_verifier(
        model,
        reliability_model,
        evidence_model,
        fine_model,
        hierarchy,
        loaders["dev"],
        device,
        **evaluation_options,
    )
    _assert_epoch0_identity(initial, config, full_dev=full_dev)
    (output_dir / "epoch0_metrics.json").write_text(
        json.dumps({"metrics": initial}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best = _selection(initial, config)
    best_epoch = 0
    best_path = output_dir / "best_model.pt"
    atomic_save(checkpoint_payload(0, initial), best_path)
    history = [{"epoch": 0, "dev": initial}]
    logger.info("Epoch 0 identity passed; dev: %s", _compact(initial))

    patience = 0
    for epoch in range(1, config.optim.num_epochs + 1):
        model.train()
        for frozen in (reliability_model, evidence_model, fine_model, hierarchy):
            frozen.eval()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        finite_steps = 0
        if full_chain_oof is not None:
            train_batches = list(loaders["train"])
            random.Random(config.runtime.seed + epoch).shuffle(train_batches)
        else:
            train_batches = loaders["train"]
        progress = tqdm(
            train_batches,
            desc=f"M3.6A {epoch}/{config.optim.num_epochs}",
        )
        for step, raw_batch in enumerate(progress, start=1):
            if full_chain_oof is not None:
                context = move_null_release_context_batch(raw_batch, device)
            else:
                paired = move_paired_record_batch(raw_batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    context = frozen_layered_action_features(
                        reliability_model,
                        evidence_model,
                        fine_model,
                        hierarchy,
                        paired,
                        decode_options=decode,
                    )
            expanded = context["expanded"]
            hierarchy_outputs = context["hierarchy_outputs"]
            fine_outputs = context["fine_outputs"]
            assert isinstance(expanded, dict)
            assert isinstance(hierarchy_outputs, dict)
            assert isinstance(fine_outputs, dict)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(
                    fine_outputs,
                    hierarchy_outputs,
                    context["evidence_outputs"],
                    expanded,
                    current_visible_mask=context["current_visible"],
                    base_is_null_mask=context["base_is_null"],
                    reliability_outputs=context["reliability_outputs"],
                    deployment_span_mask=context["deployment_span_mask"],
                )
                losses = layered_action_verifier_loss(
                    outputs,
                    fine_outputs,
                    hierarchy_outputs,
                    expanded,
                    **loss_options,
                )
                loss = losses["loss"] / accumulation
            if not torch.isfinite(loss):
                logger.warning("Skipping non-finite loss epoch=%d step=%d", epoch, step)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            running += float(losses["loss"].item())
            finite_steps += 1
            if step % accumulation == 0 or step == len(train_batches):
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
                progress.set_postfix(loss=f"{running / max(finite_steps, 1):.4f}")

        dev_metrics = evaluate_layered_action_verifier(
            model,
            reliability_model,
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
            _compact(dev_metrics),
        )
        selection = _selection(dev_metrics, config)
        if selection > best:
            best = selection
            best_epoch = epoch
            atomic_save(checkpoint_payload(epoch, dev_metrics), best_path)
            logger.info("New best epoch %d: %s", epoch, selection)
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
        "best_epoch": best_epoch,
        "best_selection": list(best),
        "history": history,
        "test": None,
        "formal_test_frozen": True,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_checkpoint": str(best_path.resolve()),
                "test": None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
