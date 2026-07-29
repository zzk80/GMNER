"""Train the M3.3A-P3 conditional same-type region resolver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
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
from gmner.engine.same_type_region_resolver_evaluator import (
    evaluate_same_type_region_resolver,
    frozen_same_type_resolver_context,
)
from gmner.evidence_visibility_config import (
    load_evidence_visibility_config,
)
from gmner.losses.same_type_region_resolver_loss import (
    same_type_region_resolver_loss,
)
from gmner.models.same_type_region_resolver import (
    ConditionalSameTypeRegionResolver,
)
from gmner.same_type_region_resolver_config import (
    load_same_type_region_resolver_config,
)
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed
from scripts.train_evidence_visibility import (
    load_frozen_chain as load_evidence_chain,
)
from scripts.train_fine_grounding_adapter import (
    atomic_save,
    decode_options,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_scheduler(
    optimizer: AdamW, total_steps: int, warmup_ratio: float
) -> LambdaLR:
    warmup = int(total_steps * max(0.0, min(float(warmup_ratio), 1.0)))

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(step, 1) / warmup
        return max(
            0.0,
            (total_steps - step) / max(total_steps - warmup, 1),
        )

    return LambdaLR(optimizer, factor)


def load_frozen_chain(config, root: Path, device: torch.device):
    evidence_config_path = resolve(
        config.frozen.evidence_config, root
    )
    evidence_config = load_evidence_visibility_config(
        evidence_config_path
    )
    (
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        fine_checkpoint,
    ) = load_evidence_chain(evidence_config, root, device)
    evidence_checkpoint_path = resolve(
        config.frozen.evidence_checkpoint, root
    )
    evidence_checkpoint = torch.load(
        evidence_checkpoint_path, map_location="cpu"
    )
    evidence_model.load_state_dict(
        evidence_checkpoint["model_state_dict"]
    )
    frozen_models = (evidence_model, fine_model, hierarchy)
    for frozen_model in frozen_models:
        frozen_model.to(device).eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad = False
    if int(config.model.hidden_size) != int(
        evidence_config.model.input_size
    ):
        raise ValueError(
            "Resolver hidden size must match frozen Fine states: "
            f"{config.model.hidden_size} != "
            f"{evidence_config.model.input_size}."
        )
    model = ConditionalSameTypeRegionResolver(config.model).to(device)
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
        evidence_checkpoint_path,
    )


def compact_metrics(metrics: dict[str, float]) -> str:
    keys = (
        "gmner_score",
        "baseline_gmner_score",
        "gmner_delta",
        "eeg_f1",
        "gmner_corrected",
        "gmner_damaged",
        "gmner_net_correction",
        "override_count",
        "base_correct_trigger_preservation_rate",
        "loss",
    )
    return ", ".join(
        f"{key}={metrics[key]:.4f}"
        for key in keys
        if key in metrics
    )


def assert_epoch_zero_contract(
    metrics: dict[str, float],
    *,
    expected_gmner: float | None,
    tolerance: float,
) -> None:
    checks = {
        "expected_baseline_gmner": (
            expected_gmner is None
            or abs(
                float(metrics["baseline_gmner_score"])
                - float(expected_gmner)
            )
            <= float(tolerance)
        ),
        "zero_gmner_delta": abs(float(metrics["gmner_delta"]))
        <= 1e-12,
        "zero_span_delta": abs(float(metrics["span_f1_delta"]))
        <= 1e-12,
        "zero_entity_delta": abs(float(metrics["entity_f1_delta"]))
        <= 1e-12,
        "zero_override": int(metrics["override_count"]) == 0,
        "visibility_identity": int(
            metrics["visibility_changed_count"]
        )
        == 0,
        "selected_span_identity": int(
            metrics["selected_span_changed_count"]
        )
        == 0,
        "type_identity": int(metrics["type_changed_count"]) == 0,
        "non_trigger_identity": int(
            metrics["non_trigger_region_changed_count"]
        )
        == 0,
        "candidate_contract": int(
            metrics["candidate_contract_violation_count"]
        )
        == 0,
        "null_excluded": int(
            metrics["null_candidate_violation_count"]
        )
        == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Epoch-0 C1 contract failed: "
            f"{failed}; metrics={metrics}"
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    config = load_same_type_region_resolver_config(config_path)
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.optim.num_epochs = max(1, int(args.num_epochs))
    output_dir = resolve(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(
        "gmner.same_type_resolver_train", output_dir / "train.log"
    )
    set_seed(config.runtime.seed)

    datasets = {}
    for split in ("train", "dev"):
        formal = RecordCandidateDataset(
            resolve(
                getattr(config.data, f"formal_{split}_cache"), root
            )
        )
        expanded = RecordCandidateDataset(
            resolve(
                getattr(config.data, f"expanded_{split}_cache"), root
            )
        )
        datasets[split] = PairedRecordCandidateDataset(
            formal, expanded
        )
    if args.max_train_records is not None:
        datasets["train"] = Subset(
            datasets["train"],
            range(
                min(
                    max(1, args.max_train_records),
                    len(datasets["train"]),
                )
            ),
        )
    if args.max_dev_records is not None:
        datasets["dev"] = Subset(
            datasets["dev"],
            range(
                min(
                    max(1, args.max_dev_records),
                    len(datasets["dev"]),
                )
            ),
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
        evidence_checkpoint_path,
    ) = load_frozen_chain(config, root, device)
    for split in ("train", "dev"):
        paired = datasets[split]
        if isinstance(paired, Subset):
            paired = paired.dataset
        validate_fingerprints(
            paired,
            hierarchy_checkpoint=hierarchy_checkpoint,
            coarse_checkpoint=coarse_checkpoint,
            require_oof=(
                bool(config.data.require_oof_train_cache)
                and split == "train"
            ),
        )

    trainable = list(model.parameters())
    optimizer = AdamW(
        trainable,
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    accumulation = max(
        1, int(config.optim.gradient_accumulation_steps)
    )
    total_steps = (
        math.ceil(len(loaders["train"]) / accumulation)
        * int(config.optim.num_epochs)
    )
    scheduler = make_scheduler(
        optimizer, total_steps, config.optim.warmup_ratio
    )
    amp_enabled = bool(
        config.runtime.fp16 and device.type == "cuda"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    registered_decode = decode_options(hierarchy_config)
    loss_options = vars(config.loss).copy()
    evaluation_options = {
        "decode_options": registered_decode,
        "loss_options": loss_options,
        "enabled": True,
    }
    logger.info(
        "Records train/dev=%d/%d; trainable=%d; "
        "frozen hierarchy/coarse/fine/evidence; device=%s",
        len(datasets["train"]),
        len(datasets["dev"]),
        sum(parameter.numel() for parameter in trainable),
        device,
    )

    initial = evaluate_same_type_region_resolver(
        model,
        evidence_model,
        fine_model,
        hierarchy,
        loaders["dev"],
        device,
        **evaluation_options,
    )
    assert_epoch_zero_contract(
        initial,
        expected_gmner=(
            None
            if args.max_dev_records is not None
            else config.runtime.expected_dev_baseline_gmner
        ),
        tolerance=config.runtime.baseline_tolerance,
    )
    primary = config.runtime.save_best_metric
    ties = list(config.runtime.save_best_tie_breakers)
    best_selection = selection_key(initial, primary, ties)
    best_epoch = 0
    best_path = output_dir / "best_model.pt"
    history = [{"epoch": 0, "dev": initial}]

    def checkpoint_payload(
        epoch: int, metrics: dict[str, float]
    ) -> dict:
        return {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config.to_dict(),
            "config_sha256": sha256_file(config_path),
            "evidence_checkpoint_sha256": sha256_file(
                evidence_checkpoint_path
            ),
            "fine_checkpoint_epoch": fine_checkpoint.get("epoch"),
            "fine_checkpoint_metrics": fine_checkpoint.get("metrics"),
            "evidence_checkpoint_epoch": evidence_checkpoint.get(
                "epoch"
            ),
            "evidence_checkpoint_metrics": evidence_checkpoint.get(
                "metrics"
            ),
            "hierarchy_stage1_checkpoint_sha256": (
                hierarchy_checkpoint.get("stage1_checkpoint_sha256")
            ),
            "formal_candidate_config_sha256": (
                hierarchy_checkpoint.get("candidate_config_sha256")
            ),
            "expanded_candidate_config_sha256": (
                coarse_checkpoint.get("candidate_config_sha256")
            ),
            "test_accessed": False,
        }

    atomic_save(checkpoint_payload(0, initial), best_path)
    logger.info("Epoch 0 frozen M3.3A dev: %s", compact_metrics(initial))
    patience = 0

    for epoch in range(1, int(config.optim.num_epochs) + 1):
        model.train()
        evidence_model.eval()
        fine_model.eval()
        hierarchy.eval()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        finite_steps = 0
        train_counts = Counter()
        progress = tqdm(
            loaders["train"],
            desc=(
                "Same-type resolver "
                f"{epoch}/{config.optim.num_epochs}"
            ),
        )
        for step, raw_batch in enumerate(progress, start=1):
            paired = {
                branch: {
                    key: value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in values.items()
                }
                for branch, values in raw_batch.items()
            }
            formal = paired["formal"]
            expanded = paired["expanded"]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                context = frozen_same_type_resolver_context(
                    evidence_model,
                    fine_model,
                    hierarchy,
                    formal,
                    expanded,
                    decode_options=registered_decode,
                )
                outputs = model(
                    context["fine_outputs"],
                    expanded,
                    selected_span_mask=context[
                        "selected_span_mask"
                    ],
                    final_visible_mask=context[
                        "final_visible_mask"
                    ],
                )
                losses = same_type_region_resolver_loss(
                    outputs, expanded, **loss_options
                )
                loss = losses["loss"] / accumulation
            if not torch.isfinite(loss):
                logger.warning(
                    "Skipping non-finite loss epoch=%d step=%d",
                    epoch,
                    step,
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            running += float(losses["loss"].item())
            finite_steps += 1
            for key in (
                "trigger_count",
                "trigger_candidate_count",
                "valid_count",
                "correction_count",
                "preservation_count",
                "candidate_missing_count",
            ):
                train_counts[key] += int(losses[key].item())
            if (
                step % accumulation == 0
                or step == len(loaders["train"])
            ):
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
            if step % max(
                1, int(config.runtime.log_every_steps)
            ) == 0:
                progress.set_postfix(
                    loss=f"{running / max(finite_steps, 1):.4f}"
                )

        dev_metrics = evaluate_same_type_region_resolver(
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
            "Epoch %d train_loss=%.6f; "
            "train_trigger=%d train_valid=%d train_correction=%d "
            "train_preservation=%d train_candidate_missing=%d; "
            "dev: %s",
            epoch,
            running / max(finite_steps, 1),
            train_counts["trigger_count"],
            train_counts["valid_count"],
            train_counts["correction_count"],
            train_counts["preservation_count"],
            train_counts["candidate_missing_count"],
            compact_metrics(dev_metrics),
        )
        current = selection_key(dev_metrics, primary, ties)
        if current > best_selection:
            best_selection = current
            best_epoch = epoch
            patience = 0
            atomic_save(
                checkpoint_payload(epoch, dev_metrics), best_path
            )
            logger.info("New best epoch %d: %s", epoch, current)
        else:
            patience += 1
            if (
                int(config.runtime.early_stop_patience) > 0
                and patience
                >= int(config.runtime.early_stop_patience)
            ):
                logger.info("Early stopping at epoch %d", epoch)
                break

    report = {
        "best_epoch": best_epoch,
        "selection_metrics": [primary, *ties],
        "best_selection": list(best_selection),
        "history": history,
        "test": None,
        "test_accessed": False,
        "c2_rule": (
            "Run override_margin=0.2 only when C1 net is positive "
            "and base_correct_trigger_damaged > 5."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_selection": list(best_selection),
                "checkpoint": str(best_path.resolve()),
                "test": None,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
