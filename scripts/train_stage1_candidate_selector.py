"""Train the preregistered D1 Stage1 candidate selector on strict OOF data."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import source_tree_sha256
from gmner.data.null_release_oof_cache import sha256_file
from gmner.data.stage1_candidate_selector import (
    Stage1CandidateSelectorCollator,
    Stage1CandidateSelectorDataset,
)
from gmner.engine.stage1_candidate_selector_evaluator import (
    evaluate_stage1_candidate_selector,
)
from gmner.losses.stage1_candidate_selector_loss import (
    stage1_candidate_selector_loss,
)
from gmner.models.stage1_candidate_selector import Stage1CandidateSelector
from gmner.stage1_candidate_selector_config import (
    load_stage1_candidate_selector_config,
)
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate Phase 1 and epoch-0 identity, then exit without training.",
    )
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def make_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    warmup_steps = int(total_steps * max(0.0, min(float(warmup_ratio), 1.0)))

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        return max(
            0.0,
            (total_steps - step) / max(total_steps - warmup_steps, 1),
        )

    return LambdaLR(optimizer, factor)


def checkpoint_selection_key(
    metrics: dict[str, Any],
    primary: str,
    tie_breakers: list[str],
) -> tuple[float, ...]:
    names = [primary, *tie_breakers]
    missing = [name for name in names if name not in metrics]
    if missing:
        raise KeyError(f"Unknown checkpoint metrics: {missing}")
    return tuple(float(metrics[name]) for name in names)


def compact_metrics(metrics: dict[str, Any]) -> str:
    keys = (
        "span_f1",
        "span_f1_delta",
        "mner_f1",
        "mner_f1_delta",
        "eeg_f1",
        "eeg_f1_delta",
        "gmner_score",
        "gmner_f1_delta",
        "formal_gold_preservation_rate",
        "promoted_exact_span_precision",
        "nonformal_selected_count",
        "span_corrected",
        "span_damaged",
        "loss",
    )
    return ", ".join(
        f"{key}={float(metrics[key]):.4f}" for key in keys if key in metrics
    )


def validate_phase1_contract(
    *,
    root: Path,
    config_path: Path,
    config,
    train_dataset: Stage1CandidateSelectorDataset,
    dev_dataset: Stage1CandidateSelectorDataset,
) -> dict[str, Any]:
    audit_path = resolve(config.data.phase1_audit, root)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "VALID_AUDIT" or not audit.get("contract_passed"):
        raise ValueError("Phase 1 has not passed its VALID_AUDIT contract.")
    if not audit.get("selector_training_supervision_present"):
        raise ValueError("Phase 1 did not find selector training supervision.")
    if audit.get("test_accessed") is not False:
        raise ValueError("Phase 1 audit indicates Test access.")

    train = train_dataset.metadata
    dev = dev_dataset.metadata
    if train.get("test_accessed") is not False or dev.get("test_accessed") is not False:
        raise ValueError("A selector feature cache indicates Test access.")
    common_keys = (
        "candidate_config_sha256",
        "formal_source_id",
        "source2id",
        "hidden_size",
        "source_tree_sha256",
    )
    mismatched = [key for key in common_keys if train.get(key) != dev.get(key)]
    if mismatched:
        raise ValueError(f"Train/Dev selector contracts differ: {mismatched}.")
    if train.get("candidate_config_sha256") != audit.get(
        "candidate_config_sha256"
    ):
        raise ValueError("Phase 1 audit and feature cache contracts differ.")
    if int(train.get("hidden_size", -1)) != int(config.model.input_size):
        raise ValueError("Selector model input_size differs from the cache.")
    source2id = dict(train.get("source2id") or {})
    if len(source2id) != int(config.model.num_sources):
        raise ValueError("Selector model num_sources differs from the cache.")

    # Phase 1 intentionally preceded selector implementation. Keep the immutable
    # feature provenance separate from the code that consumes those features.
    return {
        "kind": "stage1_candidate_selector_training_protocol",
        "phase1_status": audit["status"],
        "phase1_audit": str(audit_path),
        "phase1_audit_sha256": sha256_file(audit_path),
        "train_cache": str(train_dataset.path),
        "train_cache_sha256": sha256_file(train_dataset.path),
        "dev_cache": str(dev_dataset.path),
        "dev_cache_sha256": sha256_file(dev_dataset.path),
        "candidate_config_sha256": str(train["candidate_config_sha256"]),
        "feature_source_tree_sha256": str(train["source_tree_sha256"]),
        "selector_source_tree_sha256": source_tree_sha256(root),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git_commit": git_commit(root),
        "seed": int(config.runtime.seed),
        "test_accessed": False,
    }


def gate0_report(
    baseline: dict[str, Any],
    epoch0: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "disabled_prediction_hash_matches_stage1": (
            baseline["prediction_sha256"]
            == baseline["stage1_prediction_sha256"]
        ),
        "epoch0_prediction_hash_matches_stage1": (
            epoch0["prediction_sha256"]
            == epoch0["stage1_prediction_sha256"]
        ),
        "epoch0_exact_record_identity": bool(
            epoch0["prediction_set_equal_to_stage1"]
        ),
        "epoch0_formal_count_unchanged": (
            int(epoch0["formal_selected_count"])
            == int(epoch0["stage1_prediction_count"])
        ),
        "epoch0_nonformal_selected_zero": (
            int(epoch0["nonformal_selected_count"]) == 0
        ),
        "epoch0_span_identity": epoch0["span_f1_delta"] == 0.0,
        "epoch0_mner_identity": epoch0["mner_f1_delta"] == 0.0,
        "epoch0_eeg_identity": epoch0["eeg_f1_delta"] == 0.0,
        "epoch0_gmner_identity": epoch0["gmner_f1_delta"] == 0.0,
        "test_accessed_false": (
            baseline["test_accessed"] is False
            and epoch0["test_accessed"] is False
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "baseline": baseline,
        "epoch0": epoch0,
        "test_accessed": False,
    }


def gate1_report(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "span_f1_delta_at_least_0.005": metrics["span_f1_delta"] >= 0.005,
        "mner_f1_delta_at_least_0.003": metrics["mner_f1_delta"] >= 0.003,
        "formal_gold_preservation_at_least_0.99": (
            metrics["formal_gold_preservation_rate"] >= 0.99
        ),
        "promoted_exact_span_precision_above_0.50": (
            metrics["promoted_exact_span_precision"] > 0.50
        ),
        "corrected_spans_exceed_damaged": (
            metrics["span_corrected"] > metrics["span_damaged"]
        ),
        "eeg_delta_at_least_minus_0.002": metrics["eeg_f1_delta"] >= -0.002,
        "gmner_delta_at_least_minus_0.002": (
            metrics["gmner_f1_delta"] >= -0.002
        ),
        "test_accessed_false": metrics["test_accessed"] is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "test_accessed": False,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    config = load_stage1_candidate_selector_config(config_path)
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.optim.num_epochs = max(1, int(args.num_epochs))
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(
        "gmner.stage1_candidate_selector_train",
        output_dir / "train.log",
    )
    set_seed(config.runtime.seed)

    train_dataset = Stage1CandidateSelectorDataset(
        resolve(config.data.train_cache, root),
        split="train",
    )
    dev_dataset = Stage1CandidateSelectorDataset(
        resolve(config.data.dev_cache, root),
        split="dev",
    )
    provenance = validate_phase1_contract(
        root=root,
        config_path=config_path,
        config=config,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
    )
    write_json(output_dir / "protocol_manifest.json", provenance)

    collator = Stage1CandidateSelectorCollator()
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.optim.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=collator,
        ),
        "dev": DataLoader(
            dev_dataset,
            batch_size=config.optim.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collator,
        ),
    }
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    model = Stage1CandidateSelector(config.model).to(device)
    loss_options = vars(config.loss).copy()
    baseline = evaluate_stage1_candidate_selector(
        model,
        loaders["dev"],
        device,
        threshold=config.decode.threshold,
        disabled=True,
        loss_options=loss_options,
    )
    epoch0 = evaluate_stage1_candidate_selector(
        model,
        loaders["dev"],
        device,
        threshold=config.decode.threshold,
        disabled=False,
        loss_options=loss_options,
    )
    gate0 = gate0_report(baseline, epoch0)
    write_json(output_dir / "gate0_preflight.json", gate0)
    logger.info("Gate 0: %s; %s", gate0["status"], compact_metrics(epoch0))
    if gate0["status"] != "PASS":
        raise RuntimeError("Stage1 selector Gate 0 exact identity failed.")
    if args.preflight:
        print(json.dumps(gate0, ensure_ascii=False, indent=2))
        return

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    accumulation = max(1, int(config.optim.gradient_accumulation_steps))
    total_steps = (
        math.ceil(len(loaders["train"]) / accumulation)
        * config.optim.num_epochs
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps,
        config.optim.warmup_ratio,
    )
    amp_enabled = bool(config.runtime.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_selection = checkpoint_selection_key(
        epoch0,
        config.runtime.save_best_metric,
        config.runtime.save_best_tie_breakers,
    )
    best_epoch = 0
    patience = 0
    best_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = [{"epoch": 0, "dev": epoch0}]
    atomic_save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "metrics": epoch0,
            "config": config.to_dict(),
            "provenance": provenance,
            "test_accessed": False,
        },
        best_path,
    )
    logger.info(
        "Records train=%d dev=%d; parameters=%d; device=%s",
        len(train_dataset),
        len(dev_dataset),
        sum(parameter.numel() for parameter in model.parameters()),
        device,
    )

    for epoch in range(1, config.optim.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {
            "loss": 0.0,
            "loss_entity": 0.0,
            "loss_overlap_margin": 0.0,
            "loss_residual": 0.0,
        }
        finite_steps = 0
        progress = tqdm(
            loaders["train"],
            desc=f"Stage1 selector {epoch}/{config.optim.num_epochs}",
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
                losses = stage1_candidate_selector_loss(
                    outputs,
                    batch,
                    **loss_options,
                )
                loss = losses["loss"] / accumulation
            if not torch.isfinite(loss):
                logger.warning(
                    "Skipping non-finite loss at epoch=%d step=%d",
                    epoch,
                    step,
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            finite_steps += 1
            for key in running:
                running[key] += float(losses[key].item())
            if step % accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    config.optim.gradient_clip_norm,
                )
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before_step:
                    scheduler.step()
            if step % max(1, config.runtime.log_every_steps) == 0:
                progress.set_postfix(
                    loss=f"{running['loss'] / max(finite_steps, 1):.4f}"
                )

        train_metrics = {
            key: value / max(finite_steps, 1) for key, value in running.items()
        }
        logger.info(
            "Epoch %d train: %s",
            epoch,
            ", ".join(
                f"{key}={value:.4f}" for key, value in train_metrics.items()
            ),
        )
        dev_metrics = evaluate_stage1_candidate_selector(
            model,
            loaders["dev"],
            device,
            threshold=config.decode.threshold,
            loss_options=loss_options,
        )
        logger.info("Epoch %d dev: %s", epoch, compact_metrics(dev_metrics))
        history.append(
            {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        )
        selection = checkpoint_selection_key(
            dev_metrics,
            config.runtime.save_best_metric,
            config.runtime.save_best_tie_breakers,
        )
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
                    "provenance": provenance,
                    "test_accessed": False,
                },
                best_path,
            )
            logger.info(
                "New best checkpoint at epoch %d: %s",
                epoch,
                "/".join(f"{value:.6f}" for value in selection),
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
    best_metrics = evaluate_stage1_candidate_selector(
        model,
        loaders["dev"],
        device,
        threshold=config.decode.threshold,
        loss_options=loss_options,
    )
    gate1 = gate1_report(best_metrics)
    report = {
        "kind": "stage1_candidate_selector_seed42_result",
        "best_epoch": best_epoch,
        "selection_metrics": [
            config.runtime.save_best_metric,
            *config.runtime.save_best_tie_breakers,
        ],
        "best_selection": list(best_selection),
        "baseline": baseline,
        "best_dev": best_metrics,
        "gate0": gate0,
        "gate1": gate1,
        "history": history,
        "provenance": provenance,
        "test_accessed": False,
    }
    write_json(output_dir / "train_summary.json", report)
    write_json(output_dir / "metrics.json", report)
    logger.info(
        "Best epoch=%d; Gate 1=%s; %s",
        best_epoch,
        gate1["status"],
        compact_metrics(best_metrics),
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "gate1": gate1,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
