"""Training loop implementation."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from gmner.config import GMNERConfig, dump_config
from gmner.engine.evaluator import evaluate_model
from gmner.engine.utils import move_batch_to_device
from gmner.models.gmner_model import GMNERModel
from gmner.utils.io import ensure_dir


class GMNERTrainer:
    def __init__(
        self,
        model: GMNERModel,
        config: GMNERConfig,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        num_labels: int,
        logger,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.num_labels = num_labels
        self.logger = logger

        self.device = self._resolve_device(config.runtime.device)
        self.model.to(self.device)

        self.gradual_unfreeze_enabled = bool(
            getattr(config.optim, "gradual_unfreeze_enabled", False)
        )
        self._high_level_prefixes = self._resolve_high_level_prefixes()
        self._bert_top_prefixes = self._resolve_bert_top_prefixes()
        if self.gradual_unfreeze_enabled:
            self._configure_trainable_for_epoch(1, log_changes=False)
        self.optimizer = AdamW(
            self._build_optimizer_groups(),
            lr=config.optim.learning_rate,
            weight_decay=config.optim.weight_decay,
        )

        total_updates = math.ceil(
            len(train_dataloader) / max(1, config.optim.gradient_accumulation_steps)
        ) * config.optim.num_epochs
        warmup_steps = int(total_updates * config.optim.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_updates),
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.runtime.fp16 and self.device.type == "cuda")

        self.output_dir = ensure_dir(config.runtime.output_dir)
        dump_config(config, self.output_dir / "resolved_config.yaml")

        self.best_metric_name = config.runtime.save_best_metric
        self.best_metric_value = float("-inf")
        self.best_checkpoint_path = self.output_dir / "best_model.pt"
        self.early_stopping_patience = max(0, int(getattr(config.runtime, "early_stopping_patience", 0)))

    def _resolve_high_level_prefixes(self) -> tuple[str, ...]:
        prefixes = ["aligner", "text_projector"]
        graph_layers = getattr(self.model.text_graph_encoder, "layers", None)
        if graph_layers is not None and len(graph_layers) > 0:
            prefixes.append(f"text_graph_encoder.layers.{len(graph_layers) - 1}")
        return tuple(prefixes)

    def _resolve_bert_top_prefixes(self) -> tuple[str, ...]:
        backbone = self.model.text_encoder.backbone
        candidates = [
            (getattr(backbone, "encoder", None), "text_encoder.backbone.encoder.layer"),
            (getattr(backbone, "transformer", None), "text_encoder.backbone.transformer.layer"),
        ]
        layers = None
        prefix = ""
        for container, candidate_prefix in candidates:
            candidate_layers = getattr(container, "layer", None) if container is not None else None
            if candidate_layers is not None:
                layers = candidate_layers
                prefix = candidate_prefix
                break
        if layers is None:
            return tuple()

        count = max(0, int(getattr(self.config.optim, "bert_unfreeze_last_n_layers", 4)))
        start = max(0, len(layers) - count)
        return tuple(f"{prefix}.{index}" for index in range(start, len(layers)))

    @staticmethod
    def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
        return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)

    def _configure_trainable_for_epoch(self, epoch: int, log_changes: bool = True) -> None:
        if not self.gradual_unfreeze_enabled:
            return

        high_epoch = max(1, int(getattr(self.config.optim, "gradual_unfreeze_high_epoch", 3)))
        bert_epoch = max(1, int(getattr(self.config.optim, "gradual_unfreeze_bert_epoch", 6)))
        enable_high = epoch >= high_epoch
        enable_bert = epoch >= bert_epoch
        changed: list[str] = []

        for name, parameter in self.model.named_parameters():
            desired = parameter.requires_grad
            if self._matches_prefix(name, self._high_level_prefixes):
                desired = enable_high
            if self._matches_prefix(name, self._bert_top_prefixes):
                desired = enable_bert
            if parameter.requires_grad != desired:
                parameter.requires_grad = desired
                changed.append(name)

        if log_changes and changed:
            trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
            total = sum(parameter.numel() for parameter in self.model.parameters())
            self.logger.info(
                "Gradual unfreeze epoch=%d high=%s bert_top=%s changed=%d trainable=%d/%d (%.2f%%)",
                epoch,
                enable_high,
                enable_bert,
                len(changed),
                trainable,
                total,
                100.0 * trainable / max(total, 1),
            )

    def _build_optimizer_groups(self) -> list[dict]:
        default_lr = float(self.config.optim.learning_rate)
        new_lr = float(getattr(self.config.optim, "new_module_learning_rate", default_lr))
        high_lr = float(getattr(self.config.optim, "high_level_learning_rate", default_lr))
        backbone_lr = float(getattr(self.config.optim, "backbone_learning_rate", default_lr))
        new_prefixes = (
            "joint_entity_adapter",
            "joint_type_region_verifier",
            "external_knowledge_bank",
            "multiscale_grounding_aligner",
        )

        groups = {
            "new": {"params": [], "lr": new_lr, "group_name": "new"},
            "high": {"params": [], "lr": high_lr, "group_name": "high"},
            "backbone": {"params": [], "lr": backbone_lr, "group_name": "backbone"},
            "default": {"params": [], "lr": default_lr, "group_name": "default"},
        }
        for name, parameter in self.model.named_parameters():
            future_candidate = (
                self.gradual_unfreeze_enabled
                and (
                    self._matches_prefix(name, self._high_level_prefixes)
                    or self._matches_prefix(name, self._bert_top_prefixes)
                )
            )
            if not parameter.requires_grad and not future_candidate:
                continue
            if self._matches_prefix(name, new_prefixes):
                groups["new"]["params"].append(parameter)
            elif self._matches_prefix(name, self._bert_top_prefixes):
                groups["backbone"]["params"].append(parameter)
            elif self._matches_prefix(name, self._high_level_prefixes):
                groups["high"]["params"].append(parameter)
            else:
                groups["default"]["params"].append(parameter)

        optimizer_groups = [group for group in groups.values() if group["params"]]
        if not optimizer_groups:
            raise ValueError("No parameters were selected for optimization.")
        self.logger.info(
            "Optimizer groups: %s",
            ", ".join(
                f"{group['group_name']}={sum(p.numel() for p in group['params'])}@{group['lr']:.2e}"
                for group in optimizer_groups
            ),
        )
        return optimizer_groups

    def _grad_norms(self) -> Dict[str, float]:
        graph_prefix = self._high_level_prefixes[-1:] if self._high_level_prefixes else tuple()
        bert_prefix = self._bert_top_prefixes[-1:] if self._bert_top_prefixes else tuple()
        tracked = {
            "grad_norm_jtrv": ("joint_type_region_verifier",),
            "grad_norm_entity_adapter": ("joint_entity_adapter",),
            "grad_norm_external_knowledge": ("external_knowledge_bank",),
            "grad_norm_multiscale_grounding": (
                "multiscale_grounding_aligner",
            ),
            "grad_norm_entity_projection": (
                "joint_entity_adapter",
                "joint_type_region_verifier.entity_projection",
                "joint_type_region_verifier.type_projection",
            ),
            "grad_norm_aligner": ("aligner",),
            "grad_norm_graph_high": graph_prefix,
            "grad_norm_bert_last_layer": bert_prefix,
        }
        totals = {key: 0.0 for key in tracked}
        nonfinite = {key: 0 for key in tracked}
        element_counts = {key: 0 for key in tracked}
        global_nonfinite = 0
        global_elements = 0
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            finite_mask = torch.isfinite(grad)
            parameter_nonfinite = int((~finite_mask).sum().item())
            parameter_elements = grad.numel()
            finite_grad = torch.where(finite_mask, grad, torch.zeros_like(grad)).double()
            parameter_squared_norm = float(torch.sum(finite_grad * finite_grad).item())
            global_nonfinite += parameter_nonfinite
            global_elements += parameter_elements
            for key, prefixes in tracked.items():
                if prefixes and self._matches_prefix(name, prefixes):
                    totals[key] += parameter_squared_norm
                    nonfinite[key] += parameter_nonfinite
                    element_counts[key] += parameter_elements

        metrics = {key: value ** 0.5 for key, value in totals.items()}
        for key in tracked:
            suffix = key.removeprefix("grad_norm_")
            metrics[f"grad_nonfinite_ratio_{suffix}"] = (
                nonfinite[key] / max(element_counts[key], 1)
            )
        metrics["grad_nonfinite_ratio_global"] = (
            global_nonfinite / max(global_elements, 1)
        )
        metrics["grad_nonfinite_update_rate"] = float(global_nonfinite > 0)
        return metrics

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        normalized = device_name.lower()
        if normalized.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_name)
        return torch.device("cpu")

    def _save_checkpoint(self, checkpoint_path: Path, epoch: int, metrics: Dict[str, float]) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            estimated_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.state_dict().values()
                if torch.is_tensor(parameter)
            )
            free_bytes = shutil.disk_usage(checkpoint_path.parent).free
            # torch.save writes a zip archive and needs headroom for the temp
            # file plus metadata. Failing early gives a useful error instead
            # of PyTorch's low-level inline_container write failure.
            required_bytes = int(estimated_bytes * 1.2) + 64 * 1024 * 1024
            if free_bytes < required_bytes:
                raise OSError(
                    "Not enough disk space to save checkpoint "
                    f"{checkpoint_path}: need about {required_bytes / (1024 ** 3):.2f} GiB, "
                    f"available {free_bytes / (1024 ** 3):.2f} GiB."
                )
            torch.save(payload, temporary_path)
            temporary_path.replace(checkpoint_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _log_metrics(self, prefix: str, metrics: Dict[str, float]) -> None:
        text = ", ".join([f"{key}={value:.4f}" for key, value in metrics.items()])
        self.logger.info("%s: %s", prefix, text)

    def _set_frozen_modules_eval(self) -> None:
        """Keep fully frozen top-level modules out of train-mode dropout."""

        for module in self.model.children():
            parameters = list(module.parameters())
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()

    def train(self) -> Path:
        global_step = 0
        grad_accum_steps = max(1, self.config.optim.gradient_accumulation_steps)
        self.optimizer.zero_grad(set_to_none=True)
        non_finite_steps = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.config.optim.num_epochs + 1):
            self._configure_trainable_for_epoch(epoch)
            self.model.train()
            if bool(getattr(self.config.runtime, "eval_frozen_modules", False)):
                self._set_frozen_modules_eval()
            running_loss = 0.0
            grad_norm_sums: Dict[str, float] = {}
            grad_norm_updates = 0
            injected_type_count = 0
            injected_region_count = 0
            injection_sample_count = 0
            perturbed_span_count = 0
            perturbation_sample_count = 0
            amp_update_attempts = 0
            amp_skipped_updates = 0

            progress = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}/{self.config.optim.num_epochs}",
                disable=not sys.stderr.isatty(),
                mininterval=5.0,
            )
            remainder = len(self.train_dataloader) % grad_accum_steps
            for step, batch in enumerate(progress, start=1):
                batch = move_batch_to_device(batch, self.device)
                is_last_step = step == len(self.train_dataloader)
                current_accum_steps = (
                    remainder
                    if remainder and step > len(self.train_dataloader) - remainder
                    else grad_accum_steps
                )

                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(batch)
                    loss = outputs["loss"] / current_accum_steps

                type_injected = outputs.get("joint_type_candidate_injected")
                region_injected = outputs.get("joint_region_candidate_injected")
                if isinstance(type_injected, torch.Tensor) and isinstance(
                    region_injected,
                    torch.Tensor,
                ):
                    injected_type_count += int(type_injected.detach().sum().item())
                    injected_region_count += int(region_injected.detach().sum().item())
                    injection_sample_count += int(type_injected.numel())
                span_perturbed = outputs.get("joint_span_perturbed")
                if isinstance(span_perturbed, torch.Tensor):
                    perturbed_span_count += int(span_perturbed.detach().sum().item())
                    perturbation_sample_count += int(span_perturbed.numel())

                if not torch.isfinite(loss):
                    non_finite_steps += 1
                    loss_parts = {
                        key: float(value.detach().cpu())
                        for key, value in outputs.items()
                        if key.startswith("loss_") and torch.is_tensor(value) and value.numel() == 1
                    }
                    if non_finite_steps <= 3 or step % max(1, self.config.runtime.log_every_steps) == 0:
                        self.logger.warning(
                            "Skipping non-finite loss at epoch=%d step=%d skipped=%d parts=%s",
                            epoch,
                            step,
                            non_finite_steps,
                            loss_parts,
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.scaler.scale(loss).backward()
                running_loss += outputs["loss"].item()

                if step % grad_accum_steps == 0 or is_last_step:
                    self.scaler.unscale_(self.optimizer)
                    grad_log_interval = max(1, int(self.config.runtime.log_every_steps))
                    should_log_gradients = (
                        (global_step + 1) % grad_log_interval == 0
                        or is_last_step
                    )
                    if (
                        bool(getattr(self.config.runtime, "log_grad_norms", False))
                        and should_log_gradients
                    ):
                        current_grad_norms = self._grad_norms()
                        for key, value in current_grad_norms.items():
                            grad_norm_sums[key] = grad_norm_sums.get(key, 0.0) + value
                        grad_norm_updates += 1
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optim.gradient_clip_norm,
                    )
                    scale_before_step = float(self.scaler.get_scale())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after_step = float(self.scaler.get_scale())
                    amp_update_attempts += 1
                    optimizer_step_skipped = (
                        self.scaler.is_enabled()
                        and scale_after_step < scale_before_step
                    )
                    amp_skipped_updates += int(optimizer_step_skipped)
                    self.optimizer.zero_grad(set_to_none=True)
                    if not optimizer_step_skipped:
                        self.scheduler.step()
                        global_step += 1

                    if (
                        not optimizer_step_skipped
                        and global_step % self.config.runtime.log_every_steps == 0
                    ):
                        lr = self.scheduler.get_last_lr()[0]
                        progress.set_postfix({"loss": f"{running_loss / step:.4f}", "lr": f"{lr:.2e}"})

            train_loss = running_loss / max(1, len(self.train_dataloader))
            self.logger.info("Epoch %d training loss: %.4f", epoch, train_loss)

            epoch_grad_norms: Dict[str, float] = {}
            if grad_norm_updates > 0:
                epoch_grad_norms = {
                    key: value / grad_norm_updates
                    for key, value in grad_norm_sums.items()
                }
                self._log_metrics(prefix=f"Epoch {epoch} gradients", metrics=epoch_grad_norms)

            train_diagnostics: Dict[str, float] = {}
            if injection_sample_count > 0:
                train_diagnostics = {
                    "train_joint_type_injection_rate": (
                        injected_type_count / injection_sample_count
                    ),
                    "train_joint_region_injection_rate": (
                        injected_region_count / injection_sample_count
                    ),
                }
            if perturbation_sample_count > 0:
                train_diagnostics["train_joint_span_perturbation_rate"] = (
                    perturbed_span_count / perturbation_sample_count
                )
            if amp_update_attempts > 0:
                train_diagnostics.update(
                    {
                        "train_amp_skipped_update_rate": (
                            amp_skipped_updates / amp_update_attempts
                        ),
                        "train_amp_scale": float(self.scaler.get_scale()),
                    }
                )
            if train_diagnostics:
                self._log_metrics(
                    prefix=f"Epoch {epoch} train diagnostics",
                    metrics=train_diagnostics,
                )

            dev_metrics = evaluate_model(
                model=self.model,
                dataloader=self.dev_dataloader,
                device=self.device,
            )
            dev_metrics.update(epoch_grad_norms)
            dev_metrics.update(train_diagnostics)
            self._log_metrics(prefix=f"Epoch {epoch} dev", metrics=dev_metrics)

            metric_value = dev_metrics.get(self.best_metric_name)
            if metric_value is None:
                metric_value = -dev_metrics.get("loss", 0.0)

            if metric_value > self.best_metric_value:
                self.best_metric_value = metric_value
                epochs_without_improvement = 0
                self._save_checkpoint(self.best_checkpoint_path, epoch=epoch, metrics=dev_metrics)
                self.logger.info(
                    "New best checkpoint saved at epoch %d with %s=%.4f",
                    epoch,
                    self.best_metric_name,
                    metric_value,
                )
            else:
                epochs_without_improvement += 1

            if self.config.runtime.save_latest_checkpoint:
                latest_checkpoint = self.output_dir / "latest_model.pt"
                self._save_checkpoint(latest_checkpoint, epoch=epoch, metrics=dev_metrics)

            if self.early_stopping_patience and epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(
                    "Early stopping at epoch %d after %d epochs without %s improvement.",
                    epoch,
                    epochs_without_improvement,
                    self.best_metric_name,
                )
                break

        summary = {
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "best_checkpoint": str(self.best_checkpoint_path),
        }
        with (self.output_dir / "train_summary.json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)

        return self.best_checkpoint_path
