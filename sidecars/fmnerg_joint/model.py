"""Zero-initialized visual residual over the accepted F2 subtype encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from sidecars.fmnerg_subtype.encoder_config import (
    SubtypeEncoderConfig,
    load_subtype_encoder_config,
)
from sidecars.fmnerg_subtype.encoder_model import (
    TrainableSubtypeEncoder,
    build_optimizer_groups,
    build_trainable_subtype_encoder,
    load_trainable_checkpoint_state,
    trainable_checkpoint_state,
)
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy

from .config import JointSubtypeConfig


class J0VisualSubtypeFusion(nn.Module):
    """Condition subtype logits on a fixed M3.3A region without changing it."""

    def __init__(
        self,
        *,
        text_encoder: TrainableSubtypeEncoder,
        taxonomy: SubtypeTaxonomy,
        text_feature_size: int,
        region_feature_size: int,
        geometry_size: int,
        hidden_size: int,
        dropout: float,
        residual_scale: float,
        experiment_mode: str,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.taxonomy = taxonomy
        self.residual_scale = float(residual_scale)
        self.experiment_mode = str(experiment_mode)
        if self.experiment_mode not in {
            "visual_fusion",
            "text_continuation",
        }:
            raise ValueError(
                f"Unknown J0 experiment mode: {self.experiment_mode!r}."
            )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_feature_size),
            nn.Linear(text_feature_size, hidden_size),
            nn.GELU(),
        )
        self.region_projection = nn.Sequential(
            nn.LayerNorm(region_feature_size),
            nn.Linear(region_feature_size, hidden_size),
            nn.GELU(),
        )
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(geometry_size + 3),
            nn.Linear(geometry_size + 3, hidden_size),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 5),
            nn.Linear(hidden_size * 5, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, taxonomy.num_subtypes),
        )
        nn.init.zeros_(self.fusion_head[-1].weight)
        nn.init.zeros_(self.fusion_head[-1].bias)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_record_indices: torch.Tensor,
        span_start_indices: torch.Tensor,
        span_end_indices: torch.Tensor,
        span_token_mask: torch.Tensor,
        coarse_type_ids: torch.Tensor,
        joint_region_features: torch.Tensor,
        joint_region_geometry: torch.Tensor,
        joint_detector_scores: torch.Tensor,
        joint_region_is_null: torch.Tensor,
        joint_visual_available: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            span_record_indices=span_record_indices,
            span_start_indices=span_start_indices,
            span_end_indices=span_end_indices,
            span_token_mask=span_token_mask,
            coarse_type_ids=coarse_type_ids,
        )
        text_state = self.text_projection(text_outputs["features"].float())
        available = joint_visual_available.bool().unsqueeze(-1)
        region_state = self.region_projection(joint_region_features.float())
        region_state = region_state * available.to(region_state.dtype)
        scalars = torch.cat(
            [
                joint_region_geometry.float(),
                joint_detector_scores.float().unsqueeze(-1),
                joint_region_is_null.float().unsqueeze(-1),
                joint_visual_available.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        scalar_state = self.scalar_projection(scalars)
        interaction = torch.cat(
            [
                text_state,
                region_state,
                text_state * region_state,
                (text_state - region_state).abs(),
                scalar_state,
            ],
            dim=-1,
        )
        computed_residual = self.fusion_head(interaction)
        if self.experiment_mode == "visual_fusion":
            raw_residual = computed_residual
        else:
            # C1 executes the same operations (including dropout/RNG use) but
            # blocks both the visual value and its gradient.
            raw_residual = computed_residual.detach() * 0.0
        bounded_residual = self.residual_scale * torch.tanh(raw_residual)
        raw_logits = text_outputs["raw_logits"].float() + bounded_residual
        logits = self.taxonomy.mask_logits(raw_logits, coarse_type_ids)
        return {
            "raw_logits": raw_logits,
            "logits": logits,
            "predicted_subtype_ids": logits.argmax(dim=-1),
            "base_raw_logits": text_outputs["raw_logits"],
            "base_logits": text_outputs["logits"],
            "base_predicted_subtype_ids": text_outputs[
                "predicted_subtype_ids"
            ],
            "text_features": text_outputs["features"],
            "raw_visual_residual_logits": raw_residual,
            "bounded_visual_residual_logits": bounded_residual,
            "formal_region_mutated": torch.zeros(
                (),
                dtype=torch.bool,
                device=logits.device,
            ),
        }


def build_j0_visual_subtype_model(
    *,
    config: JointSubtypeConfig,
    taxonomy: SubtypeTaxonomy,
    root: Path,
    device: torch.device,
    seed: int,
) -> tuple[
    J0VisualSubtypeFusion,
    Any,
    SubtypeEncoderConfig,
    dict[str, Any],
]:
    encoder_config_path = resolve_path(
        config.initialization.subtype_encoder_config,
        root,
    )
    encoder_config = load_subtype_encoder_config(encoder_config_path)
    if encoder_config.model.encoder_scope != "all":
        raise ValueError(
            "J0 must initialize from the accepted F2 all-layer encoder."
        )
    if encoder_config.model.input_size != config.model.text_feature_size:
        raise ValueError(
            "J0 text_feature_size differs from the F2 span representation."
        )
    text_encoder, tokenizer, initialization, trainability = (
        build_trainable_subtype_encoder(
            config=encoder_config,
            taxonomy=taxonomy,
            root=root,
            device=device,
        )
    )
    checkpoint_path = resolve_path(config.subtype_checkpoint(seed), root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("kind") != "fmnerg_trainable_subtype_encoder":
        raise ValueError("J0 initialization is not an F2 subtype checkpoint.")
    if checkpoint.get("test_accessed") is not False:
        raise ValueError("J0 F2 initialization accessed Test data.")
    if checkpoint.get("taxonomy_sha256") != taxonomy.source_sha256:
        raise ValueError("J0 F2 taxonomy fingerprint changed.")
    if checkpoint.get("config_sha256") != sha256_file(encoder_config_path):
        raise ValueError("J0 F2 encoder config fingerprint changed.")
    if dict(checkpoint.get("trainability") or {}) != trainability:
        raise ValueError("J0 F2 trainability contract changed.")
    load_trainable_checkpoint_state(text_encoder, checkpoint["model"])
    model = J0VisualSubtypeFusion(
        text_encoder=text_encoder,
        taxonomy=taxonomy,
        text_feature_size=config.model.text_feature_size,
        region_feature_size=config.model.region_feature_size,
        geometry_size=config.model.geometry_size,
        hidden_size=config.model.hidden_size,
        dropout=config.model.dropout,
        residual_scale=config.model.residual_scale,
        experiment_mode=config.model.experiment_mode,
    ).to(device)
    report = {
        "stage": "j0",
        "experiment_mode": config.model.experiment_mode,
        "subtype_encoder_config": str(encoder_config_path),
        "subtype_encoder_config_sha256": sha256_file(encoder_config_path),
        "subtype_checkpoint": str(checkpoint_path),
        "subtype_checkpoint_sha256": sha256_file(checkpoint_path),
        "subtype_checkpoint_epoch": int(checkpoint["epoch"]),
        "subtype_checkpoint_metrics": dict(checkpoint["metrics"]),
        "subtype_encoder_initialization": initialization,
        "subtype_encoder_trainability": trainability,
        "formal_stage1_mutated": False,
        "formal_region_mutated": False,
        "test_accessed": False,
    }
    return model, tokenizer, encoder_config, report


def build_j0_optimizer_groups(
    model: J0VisualSubtypeFusion,
    *,
    encoder_config: SubtypeEncoderConfig,
    config: JointSubtypeConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups, report = build_optimizer_groups(
        model.text_encoder,
        encoder_config,
    )
    fusion_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("text_encoder.") and parameter.requires_grad
    ]
    groups.append(
        {
            "params": fusion_parameters,
            "lr": config.optim.fusion_learning_rate,
            "group_name": "visual_fusion",
        }
    )
    identifiers = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("J0 optimizer parameter groups overlap.")
    report["visual_fusion"] = sum(
        parameter.numel() for parameter in fusion_parameters
    )
    return groups, report


def j0_checkpoint_state(
    model: J0VisualSubtypeFusion,
) -> dict[str, Any]:
    return {
        "text_encoder": trainable_checkpoint_state(model.text_encoder),
        "visual_fusion_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if not name.startswith("text_encoder.")
        },
    }


def load_j0_checkpoint_state(
    model: J0VisualSubtypeFusion,
    payload: dict[str, Any],
) -> None:
    load_trainable_checkpoint_state(
        model.text_encoder,
        payload["text_encoder"],
    )
    current = model.state_dict()
    stored = dict(payload["visual_fusion_state_dict"])
    expected = {
        name for name in current if not name.startswith("text_encoder.")
    }
    if set(stored) != expected:
        raise ValueError("J0 visual-fusion checkpoint keys changed.")
    current.update(stored)
    model.load_state_dict(current, strict=True)
