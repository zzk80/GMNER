#!/usr/bin/env python3
"""Train the authorized TP M1 protected typed-BIO residual on Train/Dev only."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from gmner.config import load_config
from gmner.data.artifact_utils import sha256_file
from gmner.data.clip_r16_cache import ClipR16Cache
from gmner.engine.tp_visual_residual_evaluator import evaluate_tp_visual_stage1
from gmner.engine.utils import move_batch_to_device
from gmner.models.typed_bio_visual_residual import (
    JointTypedBIOVisualStage1,
    ProtectedTypedBIOVisualStage1,
    TypedBIOVisualResidual,
    TypedBIOVisualResidualConfig,
    load_clip_features_for_batch,
    restore_joint_student_state,
    trainable_parameter_report,
)
from gmner.tp.config import TPJointM1Config, load_tp_training_config
from gmner.tp.grounding_replay import GroundabilityPriorLookup
from gmner.tp.runtime import build_tp_runtime, resolve_path
from gmner.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate epoch-0 identity and one Train-only backward pass, then exit.",
    )
    return parser.parse_args()


def state_fingerprints(
    module: torch.nn.Module,
    included_names: set[str] | None = None,
) -> tuple[str, dict[str, str]]:
    state = {
        name: tensor
        for name, tensor in module.state_dict().items()
        if included_names is None or name in included_names
    }
    cuda_devices = {
        tensor.device
        for tensor in state.values()
        if tensor.device.type == "cuda"
    }
    for device in cuda_devices:
        torch.cuda.synchronize(device)
    digest = hashlib.sha256()
    tensor_digests: dict[str, str] = {}
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        parts = (
            name.encode("utf-8"),
            str(value.dtype).encode("ascii"),
            str(tuple(value.shape)).encode("ascii"),
            value.numpy().tobytes(),
        )
        tensor_digest = hashlib.sha256()
        for part in parts:
            digest.update(part)
            tensor_digest.update(part)
        tensor_digests[name] = tensor_digest.hexdigest()
    return digest.hexdigest(), tensor_digests


def state_digest(module: torch.nn.Module) -> str:
    return state_fingerprints(module)[0]


def assert_frozen_state_exact(
    *,
    module: torch.nn.Module,
    expected_digest: str,
    expected_tensor_digests: dict[str, str],
    output_dir: Path,
    variant: str,
    epoch: int,
    included_names: set[str] | None = None,
    failure_filename: str = "frozen_state_failure.json",
    failure_kind: str = "tp_frozen_stage1_integrity_failure",
) -> str:
    actual_digest, actual_tensor_digests = state_fingerprints(module, included_names)
    if actual_digest == expected_digest:
        return actual_digest

    parameter_names = {name for name, _ in module.named_parameters()}
    buffer_names = {name for name, _ in module.named_buffers()}
    changed = [
        {
            "name": name,
            "kind": (
                "parameter"
                if name in parameter_names
                else "buffer"
                if name in buffer_names
                else "state"
            ),
            "before_sha256": expected_tensor_digests.get(name),
            "after_sha256": actual_tensor_digests.get(name),
        }
        for name in sorted(set(expected_tensor_digests) | set(actual_tensor_digests))
        if expected_tensor_digests.get(name) != actual_tensor_digests.get(name)
    ]
    failure = {
        "kind": failure_kind,
        "variant": variant,
        "epoch": int(epoch),
        "frozen_base_digest_before": expected_digest,
        "frozen_base_digest_after": actual_digest,
        "changed_tensors": changed,
        "test_accessed": False,
    }
    (output_dir / failure_filename).write_text(
        json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
    )
    raise RuntimeError(
        f"Frozen formal Stage1 parameters changed during TP M1 epoch {epoch}."
    )


def compact_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"prediction_records", "record_metrics"}
    }


def validate_m0_5_report(path: Path, base_config: Path, checkpoint: Path, dev_cache: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("kind") != "tp_m0_5_reachability_oracle":
        raise ValueError("M1 requires the formal TP M0.5 report.")
    if report.get("gate_passed") is not True or report.get("test_accessed") is not False:
        raise ValueError("M1 remains locked because M0.5 did not pass safely.")
    expected = {
        "config_sha256": sha256_file(base_config),
        "checkpoint_sha256": sha256_file(checkpoint),
        "clip_manifest_sha256": sha256_file(dev_cache / "manifest.json"),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"M0.5 provenance mismatch for {key}.")
    rho = float(report["rho"])
    if not math.isfinite(rho) or rho <= 1e-6:
        raise ValueError("M0.5 rho is invalid.")
    return report


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment_path = resolve_path(args.config, root)
    experiment = load_tp_training_config(experiment_path)
    joint_training = isinstance(experiment, TPJointM1Config)
    set_seed(experiment.seed)
    output_dir = resolve_path(experiment.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = resolve_path(experiment.base_config, root)
    checkpoint_path = resolve_path(experiment.base_checkpoint, root)
    train_cache_path = resolve_path(experiment.train_clip_cache, root)
    dev_cache_path = resolve_path(experiment.dev_clip_cache, root)
    m0_5_path = resolve_path(experiment.m0_5_report, root)
    m0_5 = validate_m0_5_report(
        m0_5_path, base_config_path, checkpoint_path, dev_cache_path
    )
    rho = float(m0_5["rho"])
    base_config = load_config(base_config_path)
    base_config.data.expand_entities_for_grounding = False
    runtime = build_tp_runtime(
        config=base_config,
        checkpoint_path=checkpoint_path,
        project_root=root,
        cache_dir=output_dir / "dataset_cache",
        batch_size=experiment.batch_size,
        include_train=True,
        train_expand_entities_for_grounding=(
            joint_training and experiment.grounding_objective
        ),
    )
    device = torch.device(args.device)
    train_cache = ClipR16Cache(train_cache_path, expected_split="train")
    dev_cache = ClipR16Cache(dev_cache_path, expected_split="dev")
    train_cache.preload_all()
    dev_cache.preload_all()
    if train_cache.feature_dim != dev_cache.feature_dim:
        raise ValueError("Train/Dev CLIP feature dimensions differ.")
    for key in ("model", "preprocessing_sha256", "region_min_score", "formal_region_budget"):
        train_value = train_cache.manifest["metadata"].get(key)
        dev_value = dev_cache.manifest["metadata"].get(key)
        if train_value != dev_value:
            raise ValueError(f"Train/Dev CLIP cache contract mismatch for {key}.")
    residual_config = TypedBIOVisualResidualConfig(
        variant=experiment.variant,
        clip_feature_dim=train_cache.feature_dim,
        hidden_size=experiment.hidden_size,
        attention_heads=experiment.attention_heads,
        ffn_intermediate_size=experiment.ffn_intermediate_size,
        dropout=experiment.dropout,
        region_budget=train_cache.region_budget,
        rho=rho,
    )
    if joint_training:
        model = JointTypedBIOVisualStage1(
            runtime["model"],
            TypedBIOVisualResidual(residual_config),
            unfreeze_last_n_layers=experiment.unfreeze_last_n_layers,
            grounding_objective=experiment.grounding_objective,
            train_residual=experiment.train_residual,
        ).to(device)
    else:
        model = ProtectedTypedBIOVisualStage1(
            runtime["model"], TypedBIOVisualResidual(residual_config)
        ).to(device)
    initialization = None
    initialization_path = None
    if joint_training and experiment.initialization_checkpoint:
        initialization_path = resolve_path(experiment.initialization_checkpoint, root)
        initialization = torch.load(initialization_path, map_location="cpu")
        if (
            initialization.get("kind") != "tp_typed_bio_visual_residual"
            or initialization.get("training_mode") != "joint"
            or initialization.get("variant") != experiment.variant
            or initialization.get("test_accessed") is not False
        ):
            raise ValueError("J3 initialization is not a compatible Train/Dev-only J1 checkpoint.")
        if initialization.get("base_checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError("J3 initialization uses a different formal Stage1 checkpoint.")
        restore_joint_student_state(
            model.base_model,
            initialization.get("student_trainable_state_dict") or {},
        )
        model.residual.load_state_dict(initialization["residual_state_dict"])
        model.refresh_teacher_from_student()
    # The formal TP path always consumes precomputed R16 region features. The
    # checkpoint's pixel ResNet is therefore inactive and remains on CPU so an
    # unused parameter cannot occupy or drift in long-lived GPU storage.
    model.offload_unused_image_encoder()
    prior_lookup = GroundabilityPriorLookup(
        resolve_path(base_config.data.groundability_type_priors, root),
        resolve_path(base_config.data.groundability_mention_priors, root),
    )
    if joint_training:
        teacher_before, teacher_tensor_digests_before = state_fingerprints(
            model.teacher_model
        )
        teacher_residual_before, teacher_residual_tensor_digests_before = (
            state_fingerprints(model.teacher_residual)
        )
        trainable_student_names = {
            name for name, parameter in model.base_model.named_parameters() if parameter.requires_grad
        }
        frozen_student_names = set(model.base_model.state_dict()) - trainable_student_names
        frozen_before, frozen_tensor_digests_before = state_fingerprints(
            model.base_model, frozen_student_names
        )
        trainable_student_before, _ = state_fingerprints(
            model.base_model, trainable_student_names
        )
    else:
        teacher_before = None
        teacher_tensor_digests_before = None
        teacher_residual_before = None
        teacher_residual_tensor_digests_before = None
        trainable_student_names = set()
        frozen_student_names = set(model.base_model.state_dict())
        trainable_student_before = None
        frozen_before, frozen_tensor_digests_before = state_fingerprints(model.base_model)
    baseline = evaluate_tp_visual_stage1(
        model=model,
        dataloader=runtime["loaders"]["dev"],
        clip_cache=dev_cache,
        device=device,
        prior_lookup=prior_lookup,
    )
    if initialization is None:
        if any(
            float(baseline[name]) != float(baseline[f"base_{name}"])
            for name in ("span_f1", "mner_f1", "eeg_f1", "gmner_f1")
        ):
            raise RuntimeError("Epoch-0 TP residual does not exactly reproduce frozen Stage1.")
    else:
        for name in (
            "span_f1",
            "mner_f1",
            "eeg_f1",
            "gmner_f1",
            "span_correct",
            "mner_correct",
            "eeg_correct",
            "gmner_correct",
            "prediction_count",
            "gold_count",
        ):
            if float(baseline[name]) != float(initialization["metrics"][name]):
                raise RuntimeError(
                    f"J3 epoch-0 baseline does not exactly reproduce J1 for {name}."
                )
    if joint_training:
        grouped = model.parameter_groups()
        optimizer = AdamW(
            [
                {
                    "name": "backbone",
                    "params": grouped["backbone"],
                    "lr": experiment.backbone_learning_rate,
                },
                {
                    "name": "fusion",
                    "params": grouped["fusion"],
                    "lr": experiment.fusion_learning_rate,
                },
                {
                    "name": "residual",
                    "params": grouped["residual"],
                    "lr": experiment.learning_rate,
                },
            ],
            weight_decay=experiment.weight_decay,
        )
        trainable = [parameter for values in grouped.values() for parameter in values]
        optimizer_groups = {
            name: {
                "parameter_tensors": len(values),
                "trainable_elements": sum(parameter.numel() for parameter in values),
                "learning_rate": float(
                    experiment.backbone_learning_rate
                    if name == "backbone"
                    else experiment.fusion_learning_rate
                    if name == "fusion"
                    else experiment.learning_rate
                ),
            }
            for name, values in grouped.items()
        }
    else:
        trainable = [parameter for parameter in model.residual.parameters() if parameter.requires_grad]
        optimizer = AdamW(
            trainable,
            lr=experiment.learning_rate,
            weight_decay=experiment.weight_decay,
        )
        optimizer_groups = {
            "residual": {
                "parameter_tensors": len(trainable),
                "trainable_elements": sum(parameter.numel() for parameter in trainable),
                "learning_rate": float(experiment.learning_rate),
            }
        }
    if args.preflight:
        model.train()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and joint_training
            else contextlib.nullcontext()
        )
        losses = None
        scanned_batches = 0
        for raw_batch in runtime["loaders"]["train"]:
            scanned_batches += 1
            batch = move_batch_to_device(raw_batch, device)
            clip_batch = load_clip_features_for_batch(train_cache, batch, device)
            with autocast:
                outputs = model(batch, clip_batch)
                candidate_losses = model.compute_loss(
                    outputs,
                    batch,
                    lambda_preserve=experiment.lambda_preserve,
                    lambda_residual=experiment.lambda_residual,
                    temperature=experiment.distillation_temperature,
                    lambda_grounding_supervision=experiment.lambda_grounding_supervision,
                    lambda_grounding_preserve=experiment.lambda_grounding_preserve,
                    grounding_temperature=experiment.grounding_temperature,
                )
            losses = candidate_losses
            if (
                not experiment.grounding_objective
                or float(losses["grounding_supervision_count"].item()) > 0
            ):
                break
            if scanned_batches >= 64:
                break
        if losses is None:
            raise RuntimeError("J3 preflight could not read a training batch.")
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in trainable
                if parameter.grad is not None
            )
        )
        if not torch.isfinite(losses["loss"]) or not torch.isfinite(gradient_norm):
            raise RuntimeError("J3 preflight produced a non-finite loss or gradient.")
        if experiment.grounding_objective:
            if float(losses["grounding_supervision_count"].item()) <= 0:
                raise RuntimeError(
                    "J3 preflight found no valid grounding supervision row in 64 batches."
                )
            if float(losses["loss_grounding_supervision"].item()) <= 1e-8:
                raise RuntimeError(
                    "J3 grounding supervision remains numerically saturated."
                )
            if any(parameter.requires_grad for parameter in model.base_model.grounding_head.parameters()):
                raise RuntimeError("J3 must keep the formal grounding head frozen.")
        optimizer.zero_grad(set_to_none=True)
        report = {
            "kind": "tp_j3_grounding_preflight",
            "baseline": compact_metrics(baseline),
            "train_samples": len(runtime["datasets"]["train"]),
            "train_records": len(runtime["datasets"]["train"].records),
            "scanned_batches": scanned_batches,
            "losses": {
                name: float(value.detach().item()) for name, value in losses.items()
            },
            "trainable_gradient_norm": float(gradient_norm.item()),
            "grounding_head_frozen": True,
            "teacher_frozen": True,
            "test_accessed": False,
        }
        (output_dir / "preflight.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    total_steps = experiment.epochs * len(runtime["loaders"]["train"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * experiment.warmup_ratio),
        num_training_steps=total_steps,
    )
    use_amp = bool(device.type == "cuda" and (experiment.fp16 or joint_training))
    amp_dtype = (
        torch.bfloat16
        if joint_training and experiment.amp_dtype == "bfloat16"
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(use_amp and amp_dtype == torch.float16)
    )
    history = [{"epoch": 0, "metrics": compact_metrics(baseline)}]
    best_key = None
    best_epoch = None
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, experiment.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "loss_crf": 0.0,
            "loss_preserve": 0.0,
            "loss_residual": 0.0,
            "loss_grounding_supervision": 0.0,
            "loss_grounding_preserve": 0.0,
            "grounding_supervision_count": 0.0,
            "grounding_teacher_error_count": 0.0,
            "grounding_preservation_count": 0.0,
        }
        steps = 0
        progress = tqdm(runtime["loaders"]["train"], desc=f"TP {experiment.variant} {epoch}/{experiment.epochs}")
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            clip_batch = load_clip_features_for_batch(train_cache, batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else contextlib.nullcontext()
            )
            with autocast:
                outputs = model(batch, clip_batch)
                losses = model.compute_loss(
                    outputs,
                    batch,
                    lambda_preserve=experiment.lambda_preserve,
                    lambda_residual=experiment.lambda_residual,
                    temperature=experiment.distillation_temperature,
                    lambda_grounding_supervision=experiment.lambda_grounding_supervision,
                    lambda_grounding_preserve=experiment.lambda_grounding_preserve,
                    grounding_temperature=experiment.grounding_temperature,
                )
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, experiment.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for name in totals:
                totals[name] += float(losses[name].detach().item())
            steps += 1
            progress.set_postfix(loss=totals["loss"] / steps)
        metrics = evaluate_tp_visual_stage1(
            model=model,
            dataloader=runtime["loaders"]["dev"],
            clip_cache=dev_cache,
            device=device,
            prior_lookup=prior_lookup,
        )
        assert_frozen_state_exact(
            module=model.base_model,
            expected_digest=frozen_before,
            expected_tensor_digests=frozen_tensor_digests_before,
            output_dir=output_dir,
            variant=experiment.variant,
            epoch=epoch,
            included_names=frozen_student_names,
        )
        if joint_training:
            assert_frozen_state_exact(
                module=model.teacher_model,
                expected_digest=teacher_before,
                expected_tensor_digests=teacher_tensor_digests_before,
                output_dir=output_dir,
                variant=experiment.variant,
                epoch=epoch,
                failure_filename="teacher_state_failure.json",
                failure_kind="tp_joint_teacher_integrity_failure",
            )
            assert_frozen_state_exact(
                module=model.teacher_residual,
                expected_digest=teacher_residual_before,
                expected_tensor_digests=teacher_residual_tensor_digests_before,
                output_dir=output_dir,
                variant=experiment.variant,
                epoch=epoch,
                failure_filename="teacher_residual_state_failure.json",
                failure_kind="tp_joint_teacher_residual_integrity_failure",
            )
        training = {name: value / max(steps, 1) for name, value in totals.items()}
        history.append(
            {
                "epoch": epoch,
                "training": training,
                "metrics": compact_metrics(metrics),
                "parameters": trainable_parameter_report(model.residual),
            }
        )
        if float(metrics["eeg_f1"]) >= float(baseline["eeg_f1"]) - experiment.eeg_preservation_tolerance:
            key = (float(metrics["gmner_f1"]), float(metrics["mner_f1"]), -epoch)
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                checkpoint_payload = {
                        "kind": "tp_typed_bio_visual_residual",
                        "format_version": 1,
                        "variant": experiment.variant,
                        "epoch": epoch,
                        "residual_state_dict": model.residual.state_dict(),
                        "residual_config": asdict(residual_config),
                        "metrics": compact_metrics(metrics),
                        "baseline_metrics": compact_metrics(baseline),
                        "experiment_config_sha256": sha256_file(experiment_path),
                        "base_config_sha256": sha256_file(base_config_path),
                        "base_checkpoint_sha256": sha256_file(checkpoint_path),
                        "m0_5_report_sha256": sha256_file(m0_5_path),
                        "train_clip_manifest_sha256": sha256_file(train_cache_path / "manifest.json"),
                        "dev_clip_manifest_sha256": sha256_file(dev_cache_path / "manifest.json"),
                        "frozen_base_digest": frozen_before,
                        "training_mode": "joint" if joint_training else "protected",
                        "optimizer_groups": optimizer_groups,
                        "implementation_sha256": sha256_file(Path(__file__)),
                        "protocol_sha256": sha256_file(root / "TP.txt"),
                        "test_accessed": False,
                    }
                if initialization_path is not None:
                    checkpoint_payload["initialization_checkpoint_sha256"] = sha256_file(
                        initialization_path
                    )
                if joint_training:
                    checkpoint_payload["student_trainable_state_dict"] = {
                        name: parameter.detach().cpu()
                        for name, parameter in model.base_model.named_parameters()
                        if parameter.requires_grad
                    }
                    checkpoint_payload["teacher_frozen_digest"] = teacher_before
                torch.save(checkpoint_payload, best_path)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True), encoding="utf-8"
        )
    frozen_after = assert_frozen_state_exact(
        module=model.base_model,
        expected_digest=frozen_before,
        expected_tensor_digests=frozen_tensor_digests_before,
        output_dir=output_dir,
        variant=experiment.variant,
        epoch=experiment.epochs,
        included_names=frozen_student_names,
    )
    teacher_after = None
    student_trainable_changed = None
    if joint_training:
        teacher_after = assert_frozen_state_exact(
            module=model.teacher_model,
            expected_digest=teacher_before,
            expected_tensor_digests=teacher_tensor_digests_before,
            output_dir=output_dir,
            variant=experiment.variant,
            epoch=experiment.epochs,
            failure_filename="teacher_state_failure.json",
            failure_kind="tp_joint_teacher_integrity_failure",
        )
        teacher_residual_after = assert_frozen_state_exact(
            module=model.teacher_residual,
            expected_digest=teacher_residual_before,
            expected_tensor_digests=teacher_residual_tensor_digests_before,
            output_dir=output_dir,
            variant=experiment.variant,
            epoch=experiment.epochs,
            failure_filename="teacher_residual_state_failure.json",
            failure_kind="tp_joint_teacher_residual_integrity_failure",
        )
        trainable_student_after, _ = state_fingerprints(
            model.base_model, trainable_student_names
        )
        student_trainable_changed = trainable_student_after != trainable_student_before
        if not student_trainable_changed:
            raise RuntimeError("Joint TP student trainable parameters did not change.")
    report = {
        "kind": "tp_m1_training_summary",
        "variant": experiment.variant,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path) if best_epoch is not None else None,
        "method_status": "COMPLETED" if best_epoch is not None else "NO_GO_NO_SAFE_CHECKPOINT",
        "baseline": compact_metrics(baseline),
        "frozen_base_digest_before": frozen_before,
        "frozen_base_digest_after": frozen_after,
        "frozen_base_exact": frozen_before == frozen_after,
        "training_mode": "joint" if joint_training else "protected",
        "optimizer_groups": optimizer_groups,
        "teacher_frozen_exact": (
            (
                teacher_before == teacher_after
                and teacher_residual_before == teacher_residual_after
            )
            if joint_training
            else None
        ),
        "grounding_objective": (
            bool(experiment.grounding_objective) if joint_training else False
        ),
        "initialization_checkpoint": (
            str(initialization_path) if initialization_path is not None else None
        ),
        "student_trainable_changed": student_trainable_changed,
        "test_accessed": False,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
