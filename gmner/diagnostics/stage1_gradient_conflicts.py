"""Layer-wise gradient conflict measurements for Stage1 task losses."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn


STAGE1_TASKS = ("ner", "grounding", "alignment")
STAGE1_TASK_PAIRS = (
    ("ner", "grounding"),
    ("ner", "alignment"),
    ("grounding", "alignment"),
)


def stable_probe_record_ids(
    record_ids: Iterable[str | int],
    *,
    count: int,
    seed: int,
) -> list[str]:
    """Select a deterministic, order-independent record probe."""

    unique_ids = {str(record_id) for record_id in record_ids}
    if count <= 0:
        raise ValueError("Probe record count must be positive.")
    if not unique_ids:
        raise ValueError("Cannot select a probe from an empty record set.")

    def rank(record_id: str) -> tuple[bytes, str]:
        payload = f"{int(seed)}\0{record_id}".encode("utf-8")
        return hashlib.sha256(payload).digest(), record_id

    return sorted(unique_ids, key=rank)[: min(int(count), len(unique_ids))]


def encoder_layer_parameter_groups(
    model: nn.Module,
    layer_indices: Sequence[int],
) -> dict[str, list[nn.Parameter]]:
    """Resolve selected shared text-encoder layers without name assumptions."""

    try:
        layers = model.text_encoder.backbone.encoder.layer
    except AttributeError as exc:
        raise ValueError(
            "Model does not expose text_encoder.backbone.encoder.layer."
        ) from exc

    layer_count = len(layers)
    if not layer_indices:
        raise ValueError("At least one encoder layer must be selected.")

    groups: dict[str, list[nn.Parameter]] = {}
    seen: set[int] = set()
    for raw_index in layer_indices:
        index = int(raw_index)
        if index < 0 or index >= layer_count:
            raise ValueError(
                f"Encoder layer {index} is outside [0, {layer_count - 1}]."
            )
        parameters = [
            parameter
            for parameter in layers[index].parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError(f"Encoder layer {index} has no trainable parameters.")
        duplicate = [parameter for parameter in parameters if id(parameter) in seen]
        if duplicate:
            raise ValueError("Encoder layer parameter groups overlap.")
        seen.update(id(parameter) for parameter in parameters)
        groups[f"layer_{index}"] = parameters
    return groups


def _valid_task_losses(
    task_losses: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    valid: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    for task_name in STAGE1_TASKS:
        loss = task_losses.get(task_name)
        if not isinstance(loss, torch.Tensor):
            skipped[task_name] = "missing"
        elif not loss.requires_grad:
            skipped[task_name] = "does_not_require_grad"
        elif loss.numel() != 1:
            skipped[task_name] = "not_scalar"
        elif not bool(torch.isfinite(loss.detach()).all().item()):
            skipped[task_name] = "nonfinite_loss"
        else:
            valid[task_name] = loss
    return valid, skipped


def compute_gradient_observation(
    task_losses: Mapping[str, torch.Tensor],
    layer_groups: Mapping[str, Sequence[nn.Parameter]],
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute one batch observation with three autograd traversals."""

    if not layer_groups:
        raise ValueError("Layer parameter groups cannot be empty.")

    parameters: list[nn.Parameter] = []
    parameter_positions: dict[int, int] = {}
    group_positions: dict[str, list[int]] = {}
    for layer_name, group in layer_groups.items():
        positions: list[int] = []
        for parameter in group:
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity not in parameter_positions:
                parameter_positions[identity] = len(parameters)
                parameters.append(parameter)
            positions.append(parameter_positions[identity])
        if not positions:
            raise ValueError(f"Layer group {layer_name!r} has no trainable parameters.")
        group_positions[str(layer_name)] = positions
    if not parameters:
        raise ValueError("No trainable parameters are available for diagnosis.")

    valid_losses, skipped_tasks = _valid_task_losses(task_losses)
    gradient_sets: dict[str, tuple[torch.Tensor | None, ...]] = {}
    ordered_tasks = [
        task_name for task_name in STAGE1_TASKS if task_name in valid_losses
    ]
    for task_index, task_name in enumerate(ordered_tasks):
        gradient_sets[task_name] = torch.autograd.grad(
            valid_losses[task_name],
            parameters,
            retain_graph=task_index < len(ordered_tasks) - 1,
            create_graph=False,
            allow_unused=True,
        )

    layers: dict[str, Any] = {}
    for layer_name, positions in group_positions.items():
        task_details: dict[str, dict[str, Any]] = {}
        for task_name in ordered_tasks:
            squared_norm = 0.0
            has_gradient = False
            finite = True
            for position in positions:
                gradient = gradient_sets[task_name][position]
                if gradient is None:
                    continue
                has_gradient = True
                value = gradient.detach().float()
                if not bool(torch.isfinite(value).all().item()):
                    finite = False
                    break
                squared_norm += float(torch.sum(value * value).item())
            norm = math.sqrt(max(squared_norm, 0.0)) if finite else None
            task_details[task_name] = {
                "has_gradient": has_gradient,
                "finite": finite,
                "norm": norm,
            }

        pair_details: dict[str, dict[str, float]] = {}
        for left_name, right_name in STAGE1_TASK_PAIRS:
            if left_name not in gradient_sets or right_name not in gradient_sets:
                continue
            left_info = task_details[left_name]
            right_info = task_details[right_name]
            left_norm = left_info["norm"]
            right_norm = right_info["norm"]
            if (
                not left_info["finite"]
                or not right_info["finite"]
                or not left_info["has_gradient"]
                or not right_info["has_gradient"]
                or left_norm is None
                or right_norm is None
                or left_norm <= epsilon
                or right_norm <= epsilon
            ):
                continue

            dot = 0.0
            for position in positions:
                left_gradient = gradient_sets[left_name][position]
                right_gradient = gradient_sets[right_name][position]
                if left_gradient is None or right_gradient is None:
                    continue
                dot += float(
                    torch.sum(
                        left_gradient.detach().float()
                        * right_gradient.detach().float()
                    ).item()
                )
            cosine = dot / max(left_norm * right_norm, epsilon)
            pair_details[f"{left_name}_vs_{right_name}"] = {
                "cosine": max(-1.0, min(1.0, float(cosine))),
                "left_norm": float(left_norm),
                "right_norm": float(right_norm),
                "norm_ratio": float(
                    max(left_norm, right_norm)
                    / max(min(left_norm, right_norm), epsilon)
                ),
            }

        layers[layer_name] = {
            "tasks": task_details,
            "pairs": pair_details,
        }

    return {
        "task_losses": {
            task_name: float(loss.detach().float().item())
            for task_name, loss in valid_losses.items()
        },
        "skipped_tasks": skipped_tasks,
        "layers": layers,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def aggregate_gradient_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    min_valid_batches: int = 10,
    negative_ratio_threshold: float = 0.30,
    strong_negative_threshold: float = -0.30,
    strong_negative_ratio_threshold: float = 0.10,
    median_norm_ratio_threshold: float = 3.0,
) -> dict[str, Any]:
    """Aggregate batch observations and apply the preregistered D0 gate."""

    if min_valid_batches <= 0:
        raise ValueError("min_valid_batches must be positive.")
    layer_names = sorted(
        {
            str(layer_name)
            for observation in observations
            for layer_name in dict(observation.get("layers", {}))
        }
    )
    layer_summaries: dict[str, Any] = {}
    comparable_sites: list[dict[str, Any]] = []
    significant_sites: list[dict[str, Any]] = []

    for layer_name in layer_names:
        pair_summaries: dict[str, Any] = {}
        for left_name, right_name in STAGE1_TASK_PAIRS:
            pair_name = f"{left_name}_vs_{right_name}"
            entries: list[Mapping[str, float]] = []
            for observation in observations:
                layer = dict(observation.get("layers", {})).get(layer_name, {})
                pair = dict(layer.get("pairs", {})).get(pair_name)
                if pair is not None:
                    entries.append(pair)

            cosines = [float(entry["cosine"]) for entry in entries]
            left_norms = [float(entry["left_norm"]) for entry in entries]
            right_norms = [float(entry["right_norm"]) for entry in entries]
            norm_ratios = [float(entry["norm_ratio"]) for entry in entries]
            valid_batches = len(entries)
            negative_ratio = (
                sum(value < 0.0 for value in cosines) / valid_batches
                if valid_batches
                else None
            )
            strong_negative_ratio = (
                sum(value < strong_negative_threshold for value in cosines)
                / valid_batches
                if valid_batches
                else None
            )
            median_norm_ratio = _median(norm_ratios)
            sufficient = valid_batches >= min_valid_batches
            significant = bool(
                sufficient
                and negative_ratio is not None
                and negative_ratio > negative_ratio_threshold
                and strong_negative_ratio is not None
                and strong_negative_ratio > strong_negative_ratio_threshold
                and median_norm_ratio is not None
                and median_norm_ratio <= median_norm_ratio_threshold
            )
            summary = {
                "valid_batches": valid_batches,
                "mean_cosine": _mean(cosines),
                "median_cosine": _median(cosines),
                "negative_ratio": negative_ratio,
                "strong_negative_ratio": strong_negative_ratio,
                "mean_left_norm": _mean(left_norms),
                "mean_right_norm": _mean(right_norms),
                "mean_norm_ratio": _mean(norm_ratios),
                "median_norm_ratio": median_norm_ratio,
                "sufficient_data": sufficient,
                "significant_conflict": significant,
            }
            pair_summaries[pair_name] = summary
            if sufficient:
                site = {
                    "layer": layer_name,
                    "pair": pair_name,
                    **summary,
                }
                comparable_sites.append(site)
                if significant:
                    significant_sites.append(site)
        layer_summaries[layer_name] = {"pairs": pair_summaries}

    most_conflicted = None
    if comparable_sites:
        most_conflicted = min(
            comparable_sites,
            key=lambda item: (
                float(item["mean_cosine"])
                if item["mean_cosine"] is not None
                else math.inf,
                -float(item["strong_negative_ratio"] or 0.0),
                -float(item["negative_ratio"] or 0.0),
            ),
        )

    if not comparable_sites:
        status = "insufficient_data"
    elif significant_sites:
        status = "significant_conflict"
    else:
        status = "no_significant_conflict"

    return {
        "observation_count": len(observations),
        "criteria": {
            "min_valid_batches": int(min_valid_batches),
            "negative_ratio_gt": float(negative_ratio_threshold),
            "strong_negative_cosine_lt": float(strong_negative_threshold),
            "strong_negative_ratio_gt": float(
                strong_negative_ratio_threshold
            ),
            "median_norm_ratio_lte": float(median_norm_ratio_threshold),
        },
        "layers": layer_summaries,
        "recommendation": {
            "status": status,
            "has_significant_conflict": bool(significant_sites),
            "recommend_d2": bool(significant_sites),
            "significant_sites": significant_sites,
            "most_conflicted_site": most_conflicted,
        },
    }
