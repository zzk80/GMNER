"""Trainable RoBERTa-copy model for the isolated FMNERG subtype task."""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn

from .encoder_config import ENCODER_SCOPES, SubtypeEncoderConfig
from .model import HierarchicalSubtypeSidecar
from .taxonomy import SubtypeTaxonomy


LAYER_PATTERN = re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.")


def encoder_layers(backbone: nn.Module) -> nn.ModuleList:
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "Subtype encoder expects a Hugging Face backbone with "
            "encoder.layer."
        )
    return layers


def configure_backbone_trainability(
    backbone: nn.Module,
    *,
    scope: str,
    last_n_layers: int,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    if scope not in ENCODER_SCOPES:
        raise ValueError(f"Unknown encoder scope: {scope!r}.")
    layers = encoder_layers(backbone)
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    trainable_layer_indices: list[int] = []
    if scope == "all":
        for parameter in backbone.parameters():
            parameter.requires_grad = True
        trainable_layer_indices = list(range(len(layers)))
    elif scope == "last_n":
        count = min(int(last_n_layers), len(layers))
        if count <= 0:
            raise ValueError("last_n_layers must select at least one layer.")
        start = len(layers) - count
        for index in range(start, len(layers)):
            for parameter in layers[index].parameters():
                parameter.requires_grad = True
            trainable_layer_indices.append(index)

    if (
        gradient_checkpointing
        and scope != "frozen"
        and hasattr(backbone, "gradient_checkpointing_enable")
    ):
        backbone.gradient_checkpointing_enable()

    total = sum(parameter.numel() for parameter in backbone.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )
    return {
        "scope": scope,
        "num_encoder_layers": len(layers),
        "trainable_layer_indices": trainable_layer_indices,
        "backbone_parameters": total,
        "trainable_backbone_parameters": trainable,
        "gradient_checkpointing": bool(
            gradient_checkpointing and scope != "frozen"
        ),
    }


def pool_online_span_features(
    hidden_states: torch.Tensor,
    *,
    span_record_indices: torch.Tensor,
    span_start_indices: torch.Tensor,
    span_end_indices: torch.Tensor,
    span_token_mask: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError("Online hidden states must have shape [B, L, H].")
    count = int(span_record_indices.numel())
    if (
        span_start_indices.numel() != count
        or span_end_indices.numel() != count
        or span_token_mask.shape
        != (count, hidden_states.size(1))
    ):
        raise ValueError("Online subtype span pooling tensors are misaligned.")
    selected = hidden_states[span_record_indices]
    rows = torch.arange(count, device=hidden_states.device)
    first = selected[rows, span_start_indices]
    last = selected[rows, span_end_indices]
    weights = span_token_mask.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (selected * weights.unsqueeze(-1)).sum(dim=1) / denominator
    return torch.cat((first, last, mean), dim=-1)


class TrainableSubtypeEncoder(nn.Module):
    """An isolated text-backbone copy plus a hierarchy-masked subtype head."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        taxonomy: SubtypeTaxonomy,
        input_size: int,
        hidden_size: int,
        dropout: float,
        head_architecture: str,
        parent_hidden_size: int | None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.subtype_head = HierarchicalSubtypeSidecar(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            taxonomy=taxonomy,
            head_architecture=head_architecture,
            parent_hidden_size=parent_hidden_size,
        )

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
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        hidden = self.backbone(**model_inputs).last_hidden_state
        features = pool_online_span_features(
            hidden,
            span_record_indices=span_record_indices,
            span_start_indices=span_start_indices,
            span_end_indices=span_end_indices,
            span_token_mask=span_token_mask,
        )
        outputs = self.subtype_head(features, coarse_type_ids)
        outputs["features"] = features
        return outputs


def build_trainable_subtype_encoder(
    *,
    config: SubtypeEncoderConfig,
    taxonomy: SubtypeTaxonomy,
    root,
    device: torch.device,
) -> tuple[
    TrainableSubtypeEncoder,
    Any,
    dict[str, Any],
    dict[str, Any],
]:
    from .frozen_encoder import load_frozen_stage1_backbone

    backbone, tokenizer, initialization = load_frozen_stage1_backbone(
        stage1_config_path=config.initialization.stage1_config,
        stage1_checkpoint_path=config.initialization.stage1_checkpoint,
        root=root,
        device=device,
    )
    hidden_size = int(backbone.config.hidden_size)
    if int(config.model.input_size) != hidden_size * 3:
        raise ValueError(
            f"Configured input_size={config.model.input_size}, but the copied "
            f"Stage1 backbone requires {hidden_size * 3}."
        )
    trainability = configure_backbone_trainability(
        backbone,
        scope=config.model.encoder_scope,
        last_n_layers=config.model.unfreeze_last_n_layers,
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
    model = TrainableSubtypeEncoder(
        backbone=backbone,
        taxonomy=taxonomy,
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        dropout=config.model.dropout,
        head_architecture=config.model.head_architecture,
        parent_hidden_size=config.model.parent_hidden_size,
    ).to(device)
    initialization.update(
        {
            "copied_from_formal_stage1": True,
            "formal_stage1_mutated": False,
            "base_model_training": config.model.encoder_scope != "frozen",
            "base_model_requires_grad": (
                trainability["trainable_backbone_parameters"] > 0
            ),
        }
    )
    return model, tokenizer, initialization, trainability


def build_optimizer_groups(
    model: TrainableSubtypeEncoder,
    config: SubtypeEncoderConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    upper_start = (
        len(encoder_layers(model.backbone))
        - int(config.model.upper_layer_count)
    )
    lower_parameters: list[nn.Parameter] = []
    upper_parameters: list[nn.Parameter] = []
    for name, parameter in model.backbone.named_parameters():
        if not parameter.requires_grad:
            continue
        match = LAYER_PATTERN.search(name)
        layer_index = int(match.group(1)) if match is not None else -1
        if layer_index >= upper_start:
            upper_parameters.append(parameter)
        else:
            lower_parameters.append(parameter)
    head_parameters = [
        parameter
        for parameter in model.subtype_head.parameters()
        if parameter.requires_grad
    ]
    groups: list[dict[str, Any]] = []
    if lower_parameters:
        groups.append(
            {
                "params": lower_parameters,
                "lr": config.optim.backbone_lower_learning_rate,
                "group_name": "backbone_lower",
            }
        )
    if upper_parameters:
        groups.append(
            {
                "params": upper_parameters,
                "lr": config.optim.backbone_upper_learning_rate,
                "group_name": "backbone_upper",
            }
        )
    groups.append(
        {
            "params": head_parameters,
            "lr": config.optim.head_learning_rate,
            "group_name": "subtype_head",
        }
    )
    identifiers = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Subtype optimizer parameter groups overlap.")
    report = {
        str(group["group_name"]): sum(
            parameter.numel() for parameter in group["params"]
        )
        for group in groups
    }
    return groups, report


def trainable_checkpoint_state(
    model: TrainableSubtypeEncoder,
) -> dict[str, Any]:
    trainable_names = {
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    }
    backbone_state = {
        name: value.detach().cpu()
        for name, value in model.backbone.state_dict().items()
        if name in trainable_names
    }
    return {
        "backbone_trainable_names": sorted(trainable_names),
        "backbone_state_dict": backbone_state,
        "subtype_head_state_dict": {
            name: value.detach().cpu()
            for name, value in model.subtype_head.state_dict().items()
        },
    }


def load_trainable_checkpoint_state(
    model: TrainableSubtypeEncoder,
    payload: dict[str, Any],
) -> None:
    expected = {
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    }
    stored = set(payload["backbone_trainable_names"])
    if stored != expected:
        raise ValueError(
            "Subtype checkpoint trainable-backbone scope does not match config."
        )
    current = model.backbone.state_dict()
    for name, value in dict(payload["backbone_state_dict"]).items():
        if name not in current:
            raise ValueError(f"Unknown subtype backbone parameter: {name}")
        current[name] = value
    model.backbone.load_state_dict(current, strict=True)
    model.subtype_head.load_state_dict(
        payload["subtype_head_state_dict"],
        strict=True,
    )
