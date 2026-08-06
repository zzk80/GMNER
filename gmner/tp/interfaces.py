"""Frozen Stage1 interface contract used by TP experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TPStage1Interfaces:
    mner_base_tokens: torch.Tensor
    base_emissions: torch.Tensor
    grounding_tokens: torch.Tensor
    image_nodes: torch.Tensor
    image_mask: torch.Tensor


def extract_tp_stage1_interfaces(
    outputs: dict[str, torch.Tensor], *, detach: bool = True
) -> TPStage1Interfaces:
    required = {
        "fused_tokens",
        "ner_logits",
        "pre_prototype_fused_tokens",
        "image_nodes",
        "image_mask",
    }
    missing = required.difference(outputs)
    if missing:
        raise ValueError(f"Frozen Stage1 outputs are missing TP interfaces: {sorted(missing)}")
    fused = outputs["fused_tokens"]
    emissions = outputs["ner_logits"]
    grounding = outputs["pre_prototype_fused_tokens"]
    if fused.ndim != 3 or grounding.shape != fused.shape:
        raise ValueError("TP token-state interfaces must share [B,L,H] shape.")
    if emissions.shape[:2] != fused.shape[:2] or emissions.size(-1) != 9:
        raise ValueError("TP requires 9-class typed-BIO emissions aligned to base tokens.")
    if outputs["image_nodes"].ndim != 3 or outputs["image_mask"].ndim != 2:
        raise ValueError("TP grounding replay requires [B,R,H] nodes and [B,R] mask.")
    maybe_detach = (lambda value: value.detach()) if detach else (lambda value: value)
    return TPStage1Interfaces(
        mner_base_tokens=maybe_detach(fused),
        base_emissions=maybe_detach(emissions),
        grounding_tokens=maybe_detach(grounding),
        image_nodes=maybe_detach(outputs["image_nodes"]),
        image_mask=maybe_detach(outputs["image_mask"]),
    )


def interface_equivalence_errors(
    outputs: dict[str, torch.Tensor],
    interfaces: TPStage1Interfaces,
) -> dict[str, float]:
    pairs = {
        "mner_base_tokens": (outputs["fused_tokens"], interfaces.mner_base_tokens),
        "base_emissions": (outputs["ner_logits"], interfaces.base_emissions),
        "grounding_tokens": (
            outputs["pre_prototype_fused_tokens"],
            interfaces.grounding_tokens,
        ),
        "image_nodes": (outputs["image_nodes"], interfaces.image_nodes),
        "image_mask": (outputs["image_mask"], interfaces.image_mask),
    }
    return {
        name: float((left.detach().float() - right.detach().float()).abs().max().item())
        for name, (left, right) in pairs.items()
    }
