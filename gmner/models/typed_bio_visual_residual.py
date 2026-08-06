"""Protected typed-BIO emission residual using frozen CLIP visual evidence."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from gmner.tp.interfaces import TPStage1Interfaces, extract_tp_stage1_interfaces
from gmner.losses.multitask import multi_positive_region_loss


TPVariant = Literal["a_text", "a1_global", "a2_r16"]
TP_VARIANTS = frozenset({"a_text", "a1_global", "a2_r16"})


@dataclass(frozen=True)
class TypedBIOVisualResidualConfig:
    variant: TPVariant
    clip_feature_dim: int = 512
    hidden_size: int = 768
    attention_heads: int = 8
    ffn_intermediate_size: int = 1536
    dropout: float = 0.1
    num_labels: int = 9
    region_budget: int = 16
    rho: float = 1.0

    def __post_init__(self) -> None:
        if self.variant not in TP_VARIANTS:
            raise ValueError(f"Unknown TP M1 variant: {self.variant}")
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads.")
        if self.num_labels != 9:
            raise ValueError("TP M1 is restricted to 9-class typed BIO.")
        if not torch.isfinite(torch.tensor(self.rho)) or self.rho <= 1e-6:
            raise ValueError("TP M1 requires a finite Train-only rho > 1e-6.")


class TypedBIOVisualResidual(nn.Module):
    """Single-layer [text; visual] attention with a bounded emission residual."""

    def __init__(self, config: TypedBIOVisualResidualConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.global_projection = nn.Sequential(
            nn.LayerNorm(config.clip_feature_dim),
            nn.Linear(config.clip_feature_dim, hidden),
        )
        self.region_projection = nn.Sequential(
            nn.LayerNorm(config.clip_feature_dim),
            nn.Linear(config.clip_feature_dim, hidden),
        )
        self.bbox_projection = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden),
            nn.GELU(),
        )
        self.score_projection = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, config.ffn_intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_intermediate_size, hidden),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.num_labels),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def normalized_geometry(boxes: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        if boxes.ndim != 3 or boxes.size(-1) != 4:
            raise ValueError("Expected region boxes [B,R,4].")
        if image_sizes.ndim != 2 or image_sizes.size(-1) != 2:
            raise ValueError("Expected image sizes [B,2] in [height,width] order.")
        height = image_sizes[:, 0].clamp_min(1.0).unsqueeze(1)
        width = image_sizes[:, 1].clamp_min(1.0).unsqueeze(1)
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        normalized = torch.stack(
            [x1 / width, y1 / height, x2 / width, y2 / height],
            dim=-1,
        ).clamp(0.0, 1.0)
        area = (
            (normalized[..., 2] - normalized[..., 0]).clamp_min(0.0)
            * (normalized[..., 3] - normalized[..., 1]).clamp_min(0.0)
        )
        return torch.cat([normalized, area.unsqueeze(-1)], dim=-1)

    def _visual_memory(
        self,
        *,
        global_features: torch.Tensor,
        region_features: torch.Tensor,
        region_boxes: torch.Tensor,
        region_scores: torch.Tensor,
        region_mask: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.variant == "a1_global":
            return self.global_projection(global_features).unsqueeze(1), torch.ones(
                (global_features.size(0), 1),
                dtype=torch.bool,
                device=global_features.device,
            )
        geometry = self.normalized_geometry(region_boxes, image_sizes)
        memory = (
            self.region_projection(region_features)
            + self.bbox_projection(geometry)
            + self.score_projection(region_scores.unsqueeze(-1).clamp(0.0, 1.0))
        )
        valid = region_mask.bool()
        if self.config.variant == "a_text":
            valid = torch.zeros_like(valid)
        return memory, valid

    def forward(
        self,
        *,
        base_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        global_features: torch.Tensor,
        region_features: torch.Tensor,
        region_boxes: torch.Tensor,
        region_scores: torch.Tensor,
        region_mask: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        visual, visual_valid = self._visual_memory(
            global_features=global_features,
            region_features=region_features,
            region_boxes=region_boxes,
            region_scores=region_scores,
            region_mask=region_mask,
            image_sizes=image_sizes,
        )
        memory = torch.cat([base_tokens, visual], dim=1)
        memory_valid = torch.cat([attention_mask.bool(), visual_valid], dim=1)
        attended, _ = self.cross_attention(
            query=base_tokens,
            key=memory,
            value=memory,
            key_padding_mask=~memory_valid,
            need_weights=False,
        )
        hidden = self.attention_norm(base_tokens + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        interaction = torch.cat(
            [
                base_tokens,
                hidden,
                base_tokens * hidden,
                (base_tokens - hidden).abs(),
            ],
            dim=-1,
        )
        raw_delta = self.residual_head(interaction)
        normalized_delta = torch.tanh(raw_delta)
        delta = float(self.config.rho) * normalized_delta
        delta = delta.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
        return {
            "raw_delta": raw_delta,
            "normalized_delta": normalized_delta,
            "delta_emissions": delta,
            "interaction_states": hidden,
            "visual_valid_mask": visual_valid,
        }

    def forward_text_only(
        self,
        *,
        base_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Replay A-text with the original fixed-size, fully masked visual slots."""
        if self.config.variant != "a_text":
            raise ValueError("Text-only replay is restricted to the A-text variant.")
        batch = base_tokens.size(0)
        budget = int(self.config.region_budget)
        clip_dim = int(self.config.clip_feature_dim)
        device = base_tokens.device
        dtype = base_tokens.dtype
        return self(
            base_tokens=base_tokens,
            attention_mask=attention_mask,
            global_features=torch.zeros((batch, clip_dim), device=device, dtype=dtype),
            region_features=torch.zeros(
                (batch, budget, clip_dim), device=device, dtype=dtype
            ),
            region_boxes=torch.zeros((batch, budget, 4), device=device, dtype=dtype),
            region_scores=torch.zeros((batch, budget), device=device, dtype=dtype),
            region_mask=torch.zeros((batch, budget), device=device, dtype=torch.bool),
            image_sizes=torch.ones((batch, 2), device=device, dtype=dtype),
        )


class ProtectedTypedBIOVisualStage1(nn.Module):
    """Frozen formal Stage1 plus the only trainable TP M1 residual branch."""

    def __init__(self, base_model: nn.Module, residual: TypedBIOVisualResidual) -> None:
        super().__init__()
        self.base_model = base_model
        self.residual = residual
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def offload_unused_image_encoder(self) -> None:
        """Keep the inactive pixel encoder out of GPU memory on the R16 path."""
        image_encoder = getattr(self.base_model, "image_encoder", None)
        if image_encoder is None:
            raise AttributeError("The formal Stage1 has no image_encoder to offload.")
        image_encoder.to(torch.device("cpu"))
        self._image_encoder_offloaded = True

    def forward(self, batch: dict[str, torch.Tensor], clip_batch: dict[str, torch.Tensor]):
        if batch.get("region_features") is None:
            raise ValueError(
                "Protected TP M1 requires formal R16 region_features; "
                "the pixel image encoder is not an authorized fallback."
            )
        self.base_model.eval()
        with torch.no_grad():
            base_batch = dict(batch)
            for supervision_key in (
                "ner_labels",
                "region_labels",
                "region_positive_mask",
                "region_iou_targets",
                "target_subtype_ids",
            ):
                base_batch.pop(supervision_key, None)
            base_outputs = self.base_model(base_batch)
            interfaces = extract_tp_stage1_interfaces(base_outputs)
        residual_outputs = self.residual(
            base_tokens=interfaces.mner_base_tokens,
            attention_mask=batch["attention_mask"],
            global_features=clip_batch["global_features"],
            region_features=clip_batch["region_features"],
            region_boxes=clip_batch["region_boxes"],
            region_scores=clip_batch["region_scores"],
            region_mask=clip_batch["region_mask"],
            image_sizes=clip_batch["image_sizes"],
        )
        corrected = interfaces.base_emissions + residual_outputs["delta_emissions"]
        return {
            "base_outputs": base_outputs,
            "interfaces": interfaces,
            "corrected_emissions": corrected,
            **residual_outputs,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        lambda_preserve: float = 1.0,
        lambda_residual: float = 0.01,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        corrected = outputs["corrected_emissions"]
        base = outputs["interfaces"].base_emissions
        crf_loss = self.base_model.ner_head.compute_loss(
            logits=corrected,
            labels=batch["ner_labels"],
            attention_mask=batch["attention_mask"],
            sample_weight=None,
            label_smoothing=0.0,
        )
        valid = batch["attention_mask"].bool()
        old_prob = torch.softmax(base / temperature, dim=-1)
        new_log_prob = torch.log_softmax(corrected / temperature, dim=-1)
        old_log_prob = torch.log(old_prob.clamp_min(1e-8))
        kl_token = (old_prob * (old_log_prob - new_log_prob)).sum(dim=-1)
        preserve = kl_token[valid].mean() if torch.any(valid) else corrected.sum() * 0.0
        normalized = outputs["normalized_delta"]
        residual_token = normalized.square().mean(dim=-1)
        residual_loss = (
            residual_token[valid].mean() if torch.any(valid) else corrected.sum() * 0.0
        )
        total = crf_loss + lambda_preserve * preserve + lambda_residual * residual_loss
        return {
            "loss": total,
            "loss_crf": crf_loss,
            "loss_preserve": preserve,
            "loss_residual": residual_loss,
        }


class JointTypedBIOVisualStage1(ProtectedTypedBIOVisualStage1):
    """A2 residual plus a narrowly unfrozen Stage1 student and frozen teacher."""

    def __init__(
        self,
        base_model: nn.Module,
        residual: TypedBIOVisualResidual,
        *,
        unfreeze_last_n_layers: int = 4,
        grounding_objective: bool = False,
        train_residual: bool = True,
    ) -> None:
        teacher_model = copy.deepcopy(base_model)
        super().__init__(base_model, residual)
        self.teacher_model = teacher_model
        self.teacher_residual = copy.deepcopy(residual)
        self.grounding_objective = bool(grounding_objective)
        self.train_residual = bool(train_residual)
        if not self.train_residual:
            for parameter in self.residual.parameters():
                parameter.requires_grad_(False)
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.teacher_residual.parameters():
            parameter.requires_grad_(False)
        self.teacher_model.eval()
        self.teacher_residual.eval()

        layers = self.base_model.text_encoder.backbone.encoder.layer
        if not 1 <= unfreeze_last_n_layers <= len(layers):
            raise ValueError("Invalid number of RoBERTa layers to unfreeze.")
        self._backbone_modules = list(layers[-unfreeze_last_n_layers:])
        self._fusion_modules = [
            self.base_model.text_graph_encoder.layers[-1],
            self.base_model.aligner,
            self.base_model.ner_head.classifier,
        ]
        text_projector = getattr(self.base_model, "text_projector", None)
        if text_projector is not None:
            self._fusion_modules.append(text_projector)
        for module in self._backbone_modules + self._fusion_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.teacher_model.eval()
        self.teacher_residual.eval()
        self.base_model.eval()
        self.residual.train(mode if self.train_residual else False)
        if mode:
            for module in self._backbone_modules + self._fusion_modules:
                module.train(True)
        return self

    def refresh_teacher_from_student(self) -> None:
        """Freeze the current Student as the J3 preservation reference."""
        self.teacher_model.load_state_dict(self.base_model.state_dict())
        self.teacher_residual.load_state_dict(self.residual.state_dict())
        for module in (self.teacher_model, self.teacher_residual):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

    def offload_unused_image_encoder(self) -> None:
        super().offload_unused_image_encoder()
        teacher_image_encoder = getattr(self.teacher_model, "image_encoder", None)
        if teacher_image_encoder is None:
            raise AttributeError("The formal teacher has no image_encoder to offload.")
        teacher_image_encoder.to(torch.device("cpu"))

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        backbone_ids = {
            id(parameter)
            for module in self._backbone_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        fusion_ids = {
            id(parameter)
            for module in self._fusion_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        residual_ids = {
            id(parameter) for parameter in self.residual.parameters() if parameter.requires_grad
        }
        if backbone_ids & fusion_ids or backbone_ids & residual_ids or fusion_ids & residual_ids:
            raise RuntimeError("Joint TP optimizer parameter groups overlap.")
        groups = {
            "backbone": [p for p in self.parameters() if id(p) in backbone_ids],
            "fusion": [p for p in self.parameters() if id(p) in fusion_ids],
            "residual": [p for p in self.parameters() if id(p) in residual_ids],
        }
        grouped = {id(parameter) for values in groups.values() for parameter in values}
        trainable = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if grouped != trainable:
            raise RuntimeError("Joint TP optimizer does not cover each trainable parameter exactly once.")
        return groups

    def forward(self, batch: dict[str, torch.Tensor], clip_batch: dict[str, torch.Tensor]):
        if batch.get("region_features") is None:
            raise ValueError("Joint TP requires formal R16 region_features.")
        base_batch = dict(batch)
        supervision_keys = ["ner_labels", "target_subtype_ids"]
        if not self.grounding_objective:
            supervision_keys.extend(
                ["region_labels", "region_positive_mask", "region_iou_targets"]
            )
        for key in supervision_keys:
            base_batch.pop(key, None)
        base_outputs = self.base_model(base_batch)
        interfaces = extract_tp_stage1_interfaces(base_outputs, detach=False)
        teacher_emissions = None
        teacher_grounding_logits = None
        if self.training:
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(base_batch)
                teacher_interfaces = extract_tp_stage1_interfaces(teacher_outputs)
                teacher_residual_outputs = self.teacher_residual(
                    base_tokens=teacher_interfaces.mner_base_tokens,
                    attention_mask=batch["attention_mask"],
                    global_features=clip_batch["global_features"],
                    region_features=clip_batch["region_features"],
                    region_boxes=clip_batch["region_boxes"],
                    region_scores=clip_batch["region_scores"],
                    region_mask=clip_batch["region_mask"],
                    image_sizes=clip_batch["image_sizes"],
                )
                teacher_emissions = (
                    teacher_interfaces.base_emissions
                    + teacher_residual_outputs["delta_emissions"]
                )
                teacher_grounding_logits = teacher_outputs.get("grounding_logits")
        residual_outputs = self.residual(
            base_tokens=interfaces.mner_base_tokens,
            attention_mask=batch["attention_mask"],
            global_features=clip_batch["global_features"],
            region_features=clip_batch["region_features"],
            region_boxes=clip_batch["region_boxes"],
            region_scores=clip_batch["region_scores"],
            region_mask=clip_batch["region_mask"],
            image_sizes=clip_batch["image_sizes"],
        )
        corrected = interfaces.base_emissions + residual_outputs["delta_emissions"]
        return {
            "base_outputs": base_outputs,
            "interfaces": interfaces,
            "teacher_emissions": teacher_emissions,
            "teacher_grounding_logits": teacher_grounding_logits,
            "corrected_emissions": corrected,
            **residual_outputs,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        lambda_preserve: float = 1.0,
        lambda_residual: float = 0.01,
        temperature: float = 1.0,
        lambda_grounding_supervision: float = 0.0,
        lambda_grounding_preserve: float = 0.0,
        grounding_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        teacher = outputs.get("teacher_emissions")
        if teacher is None:
            raise RuntimeError("Joint TP training requires frozen teacher emissions.")
        corrected = outputs["corrected_emissions"]
        crf_loss = self.base_model.ner_head.compute_loss(
            logits=corrected,
            labels=batch["ner_labels"],
            attention_mask=batch["attention_mask"],
            sample_weight=batch.get("ner_loss_weight"),
            label_smoothing=0.0,
        )
        valid = batch["attention_mask"].bool()
        teacher_prob = torch.softmax(teacher / temperature, dim=-1)
        student_log_prob = torch.log_softmax(corrected / temperature, dim=-1)
        teacher_log_prob = torch.log(teacher_prob.clamp_min(1e-8))
        kl_token = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1)
        preserve = kl_token[valid].mean() if torch.any(valid) else corrected.sum() * 0.0
        residual_token = outputs["normalized_delta"].square().mean(dim=-1)
        residual_loss = (
            residual_token[valid].mean() if torch.any(valid) else corrected.sum() * 0.0
        )
        grounding_supervision = corrected.sum() * 0.0
        grounding_preserve = corrected.sum() * 0.0
        grounding_supervision_count = corrected.new_zeros(())
        grounding_teacher_error_count = corrected.new_zeros(())
        grounding_preservation_count = corrected.new_zeros(())
        if self.grounding_objective:
            student_grounding = outputs["base_outputs"].get("grounding_logits")
            teacher_grounding = outputs.get("teacher_grounding_logits")
            if student_grounding is None or teacher_grounding is None:
                raise RuntimeError("J3 requires Student and Teacher grounding logits.")
            candidate_mask = batch["region_mask"].bool()
            positive_mask = batch["region_positive_mask"].bool() & candidate_mask
            valid_rows = positive_mask.any(dim=-1)
            teacher_prediction = teacher_grounding.masked_fill(
                ~candidate_mask, -1e4
            ).argmax(dim=-1)
            teacher_correct = positive_mask.gather(
                -1, teacher_prediction.unsqueeze(-1)
            ).squeeze(-1)
            preservation_rows = valid_rows & teacher_correct
            grounding_supervision = multi_positive_region_loss(
                logits=student_grounding / grounding_temperature,
                positive_mask=positive_mask,
                valid_mask=candidate_mask,
                sample_weight=valid_rows.to(student_grounding.dtype),
            )
            teacher_prob = torch.softmax(
                teacher_grounding.masked_fill(~candidate_mask, -1e4)
                / grounding_temperature,
                dim=-1,
            )
            student_log_prob = torch.log_softmax(
                student_grounding.masked_fill(~candidate_mask, -1e4)
                / grounding_temperature,
                dim=-1,
            )
            grounding_kl = (
                teacher_prob
                * (torch.log(teacher_prob.clamp_min(1e-8)) - student_log_prob)
            ).sum(dim=-1)
            grounding_preserve = (
                grounding_kl[preservation_rows].mean()
                if torch.any(preservation_rows)
                else student_grounding.sum() * 0.0
            )
            grounding_supervision_count = valid_rows.sum()
            grounding_teacher_error_count = (valid_rows & ~teacher_correct).sum()
            grounding_preservation_count = preservation_rows.sum()
        total = (
            crf_loss
            + lambda_preserve * preserve
            + lambda_residual * residual_loss
            + lambda_grounding_supervision * grounding_supervision
            + lambda_grounding_preserve * grounding_preserve
        )
        return {
            "loss": total,
            "loss_crf": crf_loss,
            "loss_preserve": preserve,
            "loss_residual": residual_loss,
            "loss_grounding_supervision": grounding_supervision,
            "loss_grounding_preserve": grounding_preserve,
            "grounding_supervision_count": grounding_supervision_count,
            "grounding_teacher_error_count": grounding_teacher_error_count,
            "grounding_preservation_count": grounding_preservation_count,
        }


def load_clip_features_for_batch(
    cache,
    batch: dict,
    device: torch.device,
    *,
    image_id_map: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    globals_: list[torch.Tensor] = []
    regions: list[torch.Tensor] = []
    boxes: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    sizes: list[torch.Tensor] = []
    budget = cache.region_budget
    for index, metadata in enumerate(batch["metadata"]):
        formal_image_id = str(metadata.get("image_id"))
        image_id = image_id_map.get(formal_image_id, formal_image_id) if image_id_map else formal_image_id
        paired = image_id == formal_image_id
        value = cache.get(
            image_id,
            expected_boxes=batch["region_boxes"][index, :budget] if paired else None,
            expected_valid_mask=batch["region_mask"][index, :budget] if paired else None,
        )
        globals_.append(value["global_feature"].float())
        regions.append(value["region_features"].float())
        boxes.append(value["region_boxes"].float())
        scores.append(value["region_detector_scores"].float())
        masks.append(value["region_valid_mask"].bool())
        sizes.append(value["image_size"].float())
    return {
        "global_features": torch.stack(globals_).to(device),
        "region_features": torch.stack(regions).to(device),
        "region_boxes": torch.stack(boxes).to(device),
        "region_scores": torch.stack(scores).to(device),
        "region_mask": torch.stack(masks).to(device),
        "image_sizes": torch.stack(sizes).to(device),
    }


def restore_joint_student_state(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    """Restore the explicitly saved trainable Student subset into a formal model."""
    parameters = dict(model.named_parameters())
    unknown = sorted(set(state) - set(parameters))
    if unknown:
        raise ValueError(f"Joint checkpoint contains unknown Student parameters: {unknown}")
    if not state:
        raise ValueError("Joint checkpoint has no Student trainable state.")
    with torch.no_grad():
        for name, value in state.items():
            target = parameters[name]
            if target.shape != value.shape:
                raise ValueError(f"Joint Student shape mismatch for {name}.")
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def trainable_parameter_report(module: nn.Module) -> dict[str, int]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    return {
        "declared_parameter_tensors": len(parameters),
        "trainable_elements": sum(parameter.numel() for parameter in parameters),
        "nonzero_gradient_tensors": sum(
            parameter.grad is not None and bool(torch.any(parameter.grad != 0).item())
            for parameter in parameters
        ),
    }
