"""Training and static gradient-scaling utilities for S3.1."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from gmner.engine.s3_stage1_evaluator import evaluate_s3_stage1
from gmner.engine.utils import move_batch_to_device
from gmner.losses.s3_stage1_loss import (
    S3LossWeights,
    compute_s3_stage1_losses,
)
from gmner.s3_config import S3Stage1Config, dump_s3_config


_TASKS = ("boundary", "type", "grounding", "alignment")
_STATE_KEYS = (
    "base_text_nodes",
    "text_graph_nodes",
    "fused_tokens",
    "fused_global",
    "alignment_score",
    "image_nodes",
)


def s3_loss_weights(config: S3Stage1Config) -> S3LossWeights:
    return S3LossWeights(
        boundary=float(config.loss.lambda_boundary),
        type=float(config.loss.lambda_type),
        grounding=float(config.loss.lambda_grounding),
        alignment=float(config.loss.lambda_alignment),
    )


def build_s3_optimizer(
    model,
    config: S3Stage1Config,
) -> AdamW:
    new_prefixes = ("boundary_head", "span_type_head")
    high_prefixes = ["aligner", "text_projector"]
    graph_layers = getattr(model.text_graph_encoder, "layers", None)
    if graph_layers is not None and len(graph_layers) > 0:
        high_prefixes.append(
            f"text_graph_encoder.layers.{len(graph_layers) - 1}"
        )
    backbone_prefixes = ("text_encoder.backbone",)
    grouped = {
        "new": {
            "params": [],
            "parameter_names": [],
            "lr": float(config.optim.new_module_learning_rate),
        },
        "high": {
            "params": [],
            "parameter_names": [],
            "lr": float(config.optim.high_level_learning_rate),
        },
        "backbone": {
            "params": [],
            "parameter_names": [],
            "lr": float(config.optim.backbone_learning_rate),
        },
        "default": {
            "params": [],
            "parameter_names": [],
            "lr": float(config.optim.learning_rate),
        },
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _matches_prefix(name, new_prefixes):
            key = "new"
        elif _matches_prefix(name, tuple(high_prefixes)):
            key = "high"
        elif _matches_prefix(name, backbone_prefixes):
            key = "backbone"
        else:
            key = "default"
        grouped[key]["params"].append(parameter)
        grouped[key]["parameter_names"].append(name)
    audit = _audit_s3_optimizer_assignments(
        model=model,
        grouped=grouped,
        config=config,
    )
    groups = []
    for key, value in grouped.items():
        if not value["params"]:
            continue
        groups.append(
            {
                "params": value["params"],
                "lr": value["lr"],
                "group_name": key,
            }
        )
    if not groups:
        raise ValueError("S3.1 optimizer has no trainable parameters.")
    optimizer = AdamW(
        groups,
        weight_decay=float(config.optim.weight_decay),
    )
    optimizer.s3_group_audit = audit
    _print_s3_optimizer_audit(audit)
    return optimizer


def build_s3_scheduler(
    optimizer: AdamW,
    *,
    loader_length: int,
    config: S3Stage1Config,
):
    from transformers import get_linear_schedule_with_warmup

    updates_per_epoch = math.ceil(
        loader_length
        / max(1, config.optim.gradient_accumulation_steps)
    )
    total_updates = max(1, updates_per_epoch * config.optim.num_epochs)
    warmup = int(total_updates * config.optim.warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_updates,
    )


@torch.no_grad()
def verify_student_backbone_initialization(
    *,
    student,
    wrapper,
    batch: dict[str, Any],
    device: torch.device,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    student.eval()
    wrapper.eval()
    moved = move_batch_to_device(batch, device)
    student_outputs = student.encode_records(
        moved,
        decode_boundary=False,
    )
    baseline_outputs = wrapper.encode_records(moved)
    errors = {}
    for key in _STATE_KEYS:
        left = student_outputs[key].float()
        right = baseline_outputs[key].float()
        errors[key] = (
            float((left - right).abs().max().item())
            if left.numel()
            else 0.0
        )
    return {
        "max_abs_error": errors,
        "tolerance": float(tolerance),
        "passed": all(value < tolerance for value in errors.values()),
    }


def run_s3_scaling_probe(
    *,
    model,
    train_loader,
    config: S3Stage1Config,
    device: torch.device,
    initialization: Any,
    s3_config_path: Path,
    initialization_check: dict[str, Any],
) -> dict[str, Any]:
    """Run the fixed 100-step Train-only probe and return static lambdas."""

    model.to(device).train()
    optimizer = build_s3_optimizer(model, config)
    optimizer_group_audit = optimizer.s3_group_audit
    scheduler = build_s3_scheduler(
        optimizer,
        loader_length=len(train_loader),
        config=config,
    )
    iterator = iter(train_loader)
    observations: list[dict[str, Any]] = []
    loss_totals: defaultdict[str, float] = defaultdict(float)
    denominator_totals: defaultdict[str, float] = defaultdict(float)
    equal_weights = S3LossWeights(1.0, 1.0, 1.0, 1.0)

    first_batch = _next_batch(iterator, train_loader)
    observations.append(
        _audit_probe_batch(
            model=model,
            raw_batch=first_batch,
            device=device,
            weights=equal_weights,
            label_smoothing=config.loss.label_smoothing,
            step=0,
        )
    )
    current_batch = first_batch
    for step in range(1, config.probe.steps + 1):
        model.train()
        batch = move_batch_to_device(current_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = compute_s3_stage1_losses(
            model=model,
            outputs=outputs,
            batch=batch,
            weights=equal_weights,
            label_smoothing=config.loss.label_smoothing,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config.optim.gradient_clip_norm),
        )
        optimizer.step()
        scheduler.step()
        for task in _TASKS:
            loss_totals[task] += float(
                losses[f"task_loss_{task}"].detach().item()
            )
        for key, value in losses.items():
            if key.startswith("denominator_"):
                denominator_totals[key] += float(value.detach().item())

        if (
            step % config.probe.audit_interval == 0
            or step == config.probe.steps
        ):
            observations.append(
                _audit_probe_batch(
                    model=model,
                    raw_batch=current_batch,
                    device=device,
                    weights=equal_weights,
                    label_smoothing=config.loss.label_smoothing,
                    step=step,
                )
            )
            print(
                f"S3.1 scaling probe: step={step}/"
                f"{config.probe.steps}",
                flush=True,
            )
        if step < config.probe.steps:
            try:
                current_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                current_batch = next(iterator)

    derived = derive_static_lambdas(
        observations,
        lambda_min=float(config.probe.lambda_min),
        lambda_max=float(config.probe.lambda_max),
        epsilon=float(config.probe.epsilon),
    )
    weighted_summary = summarize_weighted_gradient_ratios(
        observations,
        derived,
        epsilon=float(config.probe.epsilon),
    )
    return {
        "kind": "s3_1_train_only_scaling_probe",
        "format_version": 1,
        "scope": "train_only",
        "seed": int(config.runtime.seed),
        "steps": int(config.probe.steps),
        "audit_interval": int(config.probe.audit_interval),
        "audit_steps": [int(item["step"]) for item in observations],
        "gradient_observations": observations,
        "derived_lambdas": derived,
        "weighted_gradient_summary": weighted_summary,
        "mean_raw_losses": {
            task: loss_totals[task] / max(config.probe.steps, 1)
            for task in _TASKS
        },
        "denominator_totals": dict(denominator_totals),
        "initialization_check": initialization_check,
        "optimizer_group_audit": optimizer_group_audit,
        "formal_config_sha256": initialization.formal_config_sha256,
        "initialization_checkpoint_sha256": (
            initialization.checkpoint_sha256
        ),
        "s3_config_sha256": _file_sha256(s3_config_path),
        "probe_checkpoint_saved": False,
        "dev_accessed": False,
        "test_accessed": False,
    }


def load_and_apply_scaling_report(
    *,
    config: S3Stage1Config,
    report_path: Path,
    initialization: Any,
    s3_config_path: Path,
) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(
            f"S3.1 scaling report not found: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    optimizer_audit = dict(
        report.get("optimizer_group_audit") or {}
    )
    optimizer_checks = dict(optimizer_audit.get("checks") or {})
    checks = {
        "kind": report.get("kind")
        == "s3_1_train_only_scaling_probe",
        "train_only": report.get("scope") == "train_only",
        "steps": int(report.get("steps", -1)) == 100,
        "seed": int(report.get("seed", -1))
        == int(config.runtime.seed),
        "formal_config": report.get("formal_config_sha256")
        == initialization.formal_config_sha256,
        "checkpoint": report.get("initialization_checkpoint_sha256")
        == initialization.checkpoint_sha256,
        "s3_config": report.get("s3_config_sha256")
        == _file_sha256(s3_config_path),
        "dev_not_accessed": not bool(report.get("dev_accessed", True)),
        "test_not_accessed": not bool(
            report.get("test_accessed", True)
        ),
        "probe_not_saved": not bool(
            report.get("probe_checkpoint_saved", True)
        ),
        "optimizer_grouping": bool(optimizer_checks)
        and all(bool(value) for value in optimizer_checks.values()),
    }
    if not all(checks.values()):
        raise ValueError(
            f"S3.1 scaling report provenance failed: {checks}"
        )
    lambdas = dict(report.get("derived_lambdas") or {})
    for task in _TASKS:
        value = float(lambdas[task])
        setattr(config.loss, f"lambda_{task}", value)
    config.loss.scaling_report = str(report_path)
    return report


def derive_static_lambdas(
    observations: list[dict[str, Any]],
    *,
    lambda_min: float,
    lambda_max: float,
    epsilon: float,
) -> dict[str, float]:
    output = {"boundary": 1.0}
    for task in _TASKS[1:]:
        log_ratios = []
        for observation in observations:
            norms = observation["raw_gradient_norms"]
            for region in norms["boundary"]:
                reference = float(norms["boundary"][region])
                current = float(norms[task][region])
                if reference <= epsilon or current <= epsilon:
                    continue
                log_ratios.append(math.log(reference / current))
        if not log_ratios:
            raise ValueError(
                f"No finite gradient ratio is available for {task}."
            )
        value = math.exp(statistics.median(log_ratios))
        output[task] = float(
            min(max(value, lambda_min), lambda_max)
        )
    return output


def summarize_weighted_gradient_ratios(
    observations: list[dict[str, Any]],
    lambdas: dict[str, float],
    *,
    epsilon: float,
) -> dict[str, Any]:
    by_region: dict[str, list[float]] = defaultdict(list)
    weighted_observations = []
    for observation in observations:
        weighted = {}
        for task in _TASKS:
            weighted[task] = {
                region: float(value) * float(lambdas[task])
                for region, value in observation[
                    "raw_gradient_norms"
                ][task].items()
            }
        ratios = {}
        for region in next(iter(weighted.values())):
            values = [
                weighted[task][region]
                for task in _TASKS
                if weighted[task][region] > epsilon
            ]
            ratio = max(values) / min(values) if values else 0.0
            ratios[region] = ratio
            by_region[region].append(ratio)
        weighted_observations.append(
            {
                "step": observation["step"],
                "weighted_gradient_norms": weighted,
                "max_min_ratio_by_region": ratios,
            }
        )
    return {
        "observations": weighted_observations,
        "median_max_min_ratio_by_region": {
            region: float(statistics.median(values))
            for region, values in by_region.items()
        },
    }


def late_training_static_scaling_unresolved(
    audits: list[dict[str, Any]],
    *,
    threshold: float = 100.0,
) -> bool:
    """Detect persistent imbalance at epoch 1 and the selected checkpoint."""

    late_labels = {"epoch_1_end", "best_checkpoint"}
    late_audits = [
        audit for audit in audits if audit.get("label") in late_labels
    ]
    if {audit.get("label") for audit in late_audits} != late_labels:
        return False
    regions = {
        region
        for audit in late_audits
        for region in audit.get(
            "weighted_max_min_ratio_by_region", {}
        )
    }
    return any(
        all(
            float(
                audit["weighted_max_min_ratio_by_region"].get(
                    region, 0.0
                )
            )
            >= float(threshold)
            for audit in late_audits
        )
        for region in regions
    )


def audit_task_gradients(
    *,
    model,
    losses: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    regions = _gradient_parameter_regions(model)
    flattened = []
    slices = {}
    for region, parameters in regions.items():
        start = len(flattened)
        flattened.extend(parameters)
        slices[region] = slice(start, len(flattened))
    output: dict[str, dict[str, float]] = {}
    for task in _TASKS:
        loss = losses[f"task_loss_{task}"]
        gradients = torch.autograd.grad(
            loss,
            flattened,
            retain_graph=True,
            allow_unused=True,
        )
        task_norms = {}
        for region, region_slice in slices.items():
            squared = loss.new_zeros((), dtype=torch.float32)
            for gradient in gradients[region_slice]:
                if gradient is not None:
                    value = gradient.detach().float()
                    squared = squared + torch.sum(value * value)
            task_norms[region] = float(squared.sqrt().item())
        output[task] = task_norms
    return output


class S3Stage1Trainer:
    """Seed42 trainer with GMNER-only checkpoint selection."""

    def __init__(
        self,
        *,
        model,
        config: S3Stage1Config,
        train_loader,
        dev_loader,
        device: torch.device,
        output_dir: Path,
        initialization: Any,
        scaling_report: dict[str, Any],
        fixed_audit_batch: dict[str, Any],
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.output_dir = output_dir
        self.initialization = initialization
        self.scaling_report = scaling_report
        self.fixed_audit_batch = fixed_audit_batch
        self.weights = s3_loss_weights(config)
        self.optimizer = build_s3_optimizer(model, config)
        self.optimizer_group_audit = self.optimizer.s3_group_audit
        self.scheduler = build_s3_scheduler(
            self.optimizer,
            loader_length=len(train_loader),
            config=config,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(config.runtime.fp16 and device.type == "cuda")
        )
        self.best_metric = float("-inf")
        self.best_path = output_dir / "best_model.pt"
        self.global_update = 0
        self.gradient_audits: list[dict[str, Any]] = []
        self.clipping_steps = 0
        self.optimizer_steps = 0

    def train(self) -> tuple[Path, dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_s3_config(
            self.config,
            self.output_dir / "resolved_config.yaml",
        )
        patience = 0
        for epoch in range(1, self.config.optim.num_epochs + 1):
            train_metrics = self._train_epoch(epoch)
            dev_report = evaluate_s3_stage1(
                model=self.model,
                dataloader=self.dev_loader,
                device=self.device,
            )
            score = float(dev_report["metrics"]["gmner_score"])
            epoch_payload = {
                "epoch": epoch,
                "train": train_metrics,
                "dev": dev_report,
                "test_accessed": False,
            }
            (self.output_dir / f"epoch_{epoch:02d}.json").write_text(
                json.dumps(
                    epoch_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                "S3.1 epoch "
                f"{epoch}: span={dev_report['metrics']['span_f1']:.6f} "
                f"mner={dev_report['metrics']['mner_f1']:.6f} "
                f"eeg={dev_report['metrics']['eeg_f1']:.6f} "
                f"gmner={score:.6f}",
                flush=True,
            )
            if epoch == 1:
                self.gradient_audits.append(
                    self._fixed_gradient_audit("epoch_1_end")
                )
            if score > self.best_metric:
                self.best_metric = score
                patience = 0
                self._save_checkpoint(epoch, dev_report)
            else:
                patience += 1
            if (
                self.config.runtime.early_stopping_patience > 0
                and patience
                >= self.config.runtime.early_stopping_patience
            ):
                break
        payload = torch.load(self.best_path, map_location="cpu")
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.gradient_audits.append(
            self._fixed_gradient_audit("best_checkpoint")
        )
        gradient_report = {
            "kind": "s3_1_formal_gradient_audit",
            "format_version": 1,
            "audits": self.gradient_audits,
            "clipping_trigger_count": int(self.clipping_steps),
            "optimizer_step_count": int(self.optimizer_steps),
            "clipping_trigger_rate": (
                self.clipping_steps / max(self.optimizer_steps, 1)
            ),
            "static_lambdas": {
                task: float(getattr(self.weights, task))
                for task in _TASKS
            },
            "optimizer_group_audit": self.optimizer_group_audit,
            "test_accessed": False,
        }
        (self.output_dir / "gradient_audit.json").write_text(
            json.dumps(
                gradient_report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.best_path, gradient_report

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accumulation = max(
            1, self.config.optim.gradient_accumulation_steps
        )
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, raw_batch in enumerate(self.train_loader, start=1):
            batch = move_batch_to_device(raw_batch, self.device)
            with torch.cuda.amp.autocast(
                enabled=self.scaler.is_enabled()
            ):
                outputs = self.model(batch)
                losses = compute_s3_stage1_losses(
                    model=self.model,
                    outputs=outputs,
                    batch=batch,
                    weights=self.weights,
                    label_smoothing=self.config.loss.label_smoothing,
                )
                scaled_loss = losses["loss"] / accumulation
            self.scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % accumulation == 0
                or batch_index == len(self.train_loader)
            )
            if should_step:
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.config.optim.gradient_clip_norm),
                )
                if float(norm) > float(
                    self.config.optim.gradient_clip_norm
                ):
                    self.clipping_steps += 1
                previous_scale = float(self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skipped = (
                    self.scaler.is_enabled()
                    and float(self.scaler.get_scale()) < previous_scale
                )
                if not skipped:
                    self.scheduler.step()
                    self.optimizer_steps += 1
                    self.global_update += 1
                self.optimizer.zero_grad(set_to_none=True)
                if not skipped and self.global_update == 100:
                    self.gradient_audits.append(
                        self._fixed_gradient_audit("formal_step_100")
                    )
            for task in _TASKS:
                totals[f"loss_{task}"] += float(
                    losses[f"task_loss_{task}"].detach().item()
                )
            totals["loss"] += float(losses["loss"].detach().item())
            batches += 1
        metrics = {
            key: value / max(batches, 1)
            for key, value in totals.items()
        }
        metrics["epoch"] = float(epoch)
        return metrics

    def _fixed_gradient_audit(self, label: str) -> dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        batch = move_batch_to_device(
            self.fixed_audit_batch,
            self.device,
        )
        with torch.enable_grad():
            outputs = self.model(batch)
            losses = compute_s3_stage1_losses(
                model=self.model,
                outputs=outputs,
                batch=batch,
                weights=self.weights,
                label_smoothing=self.config.loss.label_smoothing,
            )
            raw = audit_task_gradients(model=self.model, losses=losses)
        self.model.train(was_training)
        weighted = {
            task: {
                region: value * float(getattr(self.weights, task))
                for region, value in raw[task].items()
            }
            for task in _TASKS
        }
        ratios = {}
        for region in next(iter(weighted.values())):
            values = [
                weighted[task][region]
                for task in _TASKS
                if weighted[task][region] > 0
            ]
            ratios[region] = (
                max(values) / min(values) if values else 0.0
            )
        return {
            "label": label,
            "global_update": int(self.global_update),
            "raw_gradient_norms": raw,
            "weighted_gradient_norms": weighted,
            "weighted_max_min_ratio_by_region": ratios,
        }

    def _save_checkpoint(
        self,
        epoch: int,
        dev_report: dict[str, Any],
    ) -> None:
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "metrics": dev_report["metrics"],
            "selection_metric": "stage1_dev_gmner",
            "selection_value": float(
                dev_report["metrics"]["gmner_score"]
            ),
            "seed": int(self.config.runtime.seed),
            "model_metadata": self.model.checkpoint_metadata(),
            "formal_config_sha256": (
                self.initialization.formal_config_sha256
            ),
            "initialization_checkpoint_sha256": (
                self.initialization.checkpoint_sha256
            ),
            "scaling_report_sha256": _file_sha256(
                self.config.loss.scaling_report
            ),
            "optimizer_group_audit": self.optimizer_group_audit,
            "test_accessed": False,
        }
        temporary = self.best_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(self.best_path)


def _audit_probe_batch(
    *,
    model,
    raw_batch: dict[str, Any],
    device: torch.device,
    weights: S3LossWeights,
    label_smoothing: float,
    step: int,
) -> dict[str, Any]:
    was_training = model.training
    model.train()
    batch = move_batch_to_device(raw_batch, device)
    outputs = model(batch)
    losses = compute_s3_stage1_losses(
        model=model,
        outputs=outputs,
        batch=batch,
        weights=weights,
        label_smoothing=label_smoothing,
    )
    norms = audit_task_gradients(model=model, losses=losses)
    model.train(was_training)
    return {
        "step": int(step),
        "raw_gradient_norms": norms,
        "raw_losses": {
            task: float(
                losses[f"task_loss_{task}"].detach().item()
            )
            for task in _TASKS
        },
        "denominators": {
            key: float(value.detach().item())
            for key, value in losses.items()
            if key.startswith("denominator_")
        },
    }


def _gradient_parameter_regions(model) -> dict[str, list[torch.nn.Parameter]]:
    backbone = model.text_encoder.backbone
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None or len(layers) <= 11:
        raise ValueError(
            "S3.1 scaling requires RoBERTa layers 0, 5, and 11."
        )
    regions = {
        "roberta_layer_0": list(layers[0].parameters()),
        "roberta_layer_5": list(layers[5].parameters()),
        "roberta_layer_11": list(layers[11].parameters()),
        "cross_modal_aligner": list(model.aligner.parameters()),
    }
    for name, parameters in regions.items():
        regions[name] = [
            parameter
            for parameter in parameters
            if parameter.requires_grad
        ]
        if not regions[name]:
            raise ValueError(
                f"S3.1 gradient region {name} has no trainable parameters."
            )
    return regions


def _audit_s3_optimizer_assignments(
    *,
    model,
    grouped: dict[str, dict[str, Any]],
    config: S3Stage1Config,
) -> dict[str, Any]:
    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    owners: dict[int, str] = {}
    duplicate_names: list[str] = []
    group_summaries = []
    for group_name, values in grouped.items():
        parameters = list(values["params"])
        names = list(values["parameter_names"])
        if len(parameters) != len(names):
            raise ValueError(
                f"S3.1 optimizer group {group_name} lost parameter names."
            )
        for name, parameter in zip(names, parameters):
            identity = id(parameter)
            if identity in owners:
                duplicate_names.append(name)
            owners[identity] = group_name
        group_summaries.append(
            {
                "group_name": group_name,
                "learning_rate": float(values["lr"]),
                "parameter_tensor_count": len(parameters),
                "trainable_element_count": int(
                    sum(parameter.numel() for parameter in parameters)
                ),
                "first_parameter_names": names[:5],
            }
        )

    missing_names = sorted(
        trainable[identity]
        for identity in set(trainable) - set(owners)
    )
    unexpected_ids = sorted(
        str(identity)
        for identity in set(owners) - set(trainable)
    )
    backbone_names = sorted(
        name
        for name in trainable.values()
        if _matches_prefix(name, ("text_encoder.backbone",))
    )
    wrong_backbone_group = sorted(
        name
        for identity, name in trainable.items()
        if _matches_prefix(name, ("text_encoder.backbone",))
        and owners.get(identity) != "backbone"
    )
    group_lr = {
        item["group_name"]: float(item["learning_rate"])
        for item in group_summaries
    }
    checks = {
        "every_trainable_parameter_assigned_once": (
            not missing_names
            and not unexpected_ids
            and not duplicate_names
            and len(owners) == len(trainable)
        ),
        "roberta_backbone_present": bool(backbone_names),
        "all_roberta_backbone_in_backbone_group": (
            bool(backbone_names) and not wrong_backbone_group
        ),
        "roberta_backbone_lr_matches_config": (
            group_lr.get("backbone")
            == float(config.optim.backbone_learning_rate)
        ),
    }
    report = {
        "kind": "s3_1_optimizer_group_audit",
        "format_version": 1,
        "groups": group_summaries,
        "checks": checks,
        "trainable_parameter_tensor_count": len(trainable),
        "trainable_element_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "missing_parameter_names": missing_names,
        "duplicate_parameter_names": sorted(duplicate_names),
        "unexpected_parameter_ids": unexpected_ids,
        "wrong_backbone_group": wrong_backbone_group,
    }
    if not all(checks.values()):
        raise ValueError(f"S3.1 optimizer grouping failed: {report}")
    return report


def _print_s3_optimizer_audit(report: dict[str, Any]) -> None:
    print("S3.1 optimizer groups:", flush=True)
    for group in report["groups"]:
        if not group["parameter_tensor_count"]:
            continue
        names = ", ".join(group["first_parameter_names"])
        print(
            f"  {group['group_name']}: "
            f"lr={group['learning_rate']:.2e}, "
            f"parameter_count={group['parameter_tensor_count']}, "
            f"trainable_elements={group['trainable_element_count']}, "
            f"first_parameters=[{names}]",
            flush=True,
        )


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in prefixes
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator)
    except StopIteration:
        return next(iter(loader))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
