"""Preregistered loss weighting for the subtype sidecar ablation."""

from __future__ import annotations

import math
from typing import Any

import torch

from .taxonomy import SubtypeTaxonomy


LOSS_MODES = ("ce", "class_weighted", "effective_number")


def build_subtype_class_weights(
    subtype_ids: torch.Tensor,
    *,
    taxonomy: SubtypeTaxonomy,
    mode: str,
    effective_number_beta: float,
    parent_normalize: bool,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    mode = str(mode)
    if mode not in LOSS_MODES:
        raise ValueError(f"Unknown subtype loss mode: {mode!r}.")
    labels = torch.as_tensor(subtype_ids, dtype=torch.long).reshape(-1)
    if labels.numel() == 0:
        raise ValueError("Cannot build subtype weights from an empty dataset.")
    if torch.any((labels < 0) | (labels >= taxonomy.num_subtypes)):
        raise ValueError("Subtype labels contain invalid class ids.")
    counts = torch.bincount(
        labels,
        minlength=taxonomy.num_subtypes,
    ).to(torch.float64)
    if torch.any(counts <= 0):
        missing = torch.nonzero(counts <= 0, as_tuple=False).reshape(-1)
        raise ValueError(
            "Loss ablation requires all 51 training subtypes; missing ids "
            f"{missing.tolist()}."
        )

    weights: torch.Tensor | None
    if mode == "ce":
        weights = None
        report_weights = torch.ones_like(counts)
    elif mode == "class_weighted":
        weights = counts.rsqrt()
        report_weights = weights.clone()
    else:
        beta = float(effective_number_beta)
        if not 0 < beta < 1:
            raise ValueError("effective_number_beta must be in (0, 1).")
        denominator = -torch.expm1(counts * math.log(beta))
        weights = (1.0 - beta) / denominator
        report_weights = weights.clone()

    if mode != "ce" and parent_normalize:
        parent_ids = torch.tensor(taxonomy.parent_ids, dtype=torch.long)
        for parent_id in sorted(set(taxonomy.parent_ids)):
            mask = parent_ids.eq(parent_id)
            weights[mask] /= weights[mask].mean()
        report_weights = weights.clone()
    elif mode != "ce":
        weights /= weights.mean()
        report_weights = weights.clone()

    report = {
        "loss_mode": mode,
        "effective_number_beta": float(effective_number_beta),
        "parent_normalize_class_weights": bool(parent_normalize),
        "class_counts": {
            label: int(counts[index].item())
            for index, label in enumerate(taxonomy.labels)
        },
        "class_weights": {
            label: float(report_weights[index].item())
            for index, label in enumerate(taxonomy.labels)
        },
        "parent_weight_means": {
            parent: float(
                report_weights[
                    torch.tensor(taxonomy.parent_ids).eq(parent_id)
                ].mean().item()
            )
            for parent, parent_id in taxonomy.coarse_type_ids.items()
        },
    }
    return (
        weights.to(torch.float32) if weights is not None else None,
        report,
    )
