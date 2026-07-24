"""Deployable-action supervision for the M3.6A layered verifier."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gmner.models.layered_action_verifier import (
    ACTION_KEEP,
    ACTION_TO_NULL,
    ACTION_TO_VISIBLE,
)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if mask.any() else _zero(values)


def _weighted_available(
    terms: list[tuple[torch.Tensor, float, bool]],
    reference: torch.Tensor,
) -> torch.Tensor:
    active = [
        (value, float(weight))
        for value, weight, present in terms
        if present and float(weight) > 0.0
    ]
    if not active:
        return _zero(reference)
    denominator = max(sum(weight for _, weight in active), 1e-8)
    return sum(value * weight for value, weight in active) / denominator


def layered_action_supervision(
    outputs: dict[str, torch.Tensor],
    fine_outputs: dict[str, torch.Tensor],
    hierarchy_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    stage1_spans_only: bool = True,
    require_correct_type: bool = True,
) -> dict[str, torch.Tensor]:
    """Assign labels only when the frozen Top-4 policy can realize them."""

    span_mask = batch["span_mask"].bool()
    deployable = span_mask & outputs.get(
        "policy_scope_mask", torch.ones_like(span_mask)
    ).bool()
    if stage1_spans_only:
        deployable = deployable & batch["span_source_ids"].long().eq(0)
    gold_span = batch["gold_span_mask"].bool()
    visibility_known = batch["visibility_targets"].float().ge(0.0)
    target_visible = batch["visibility_targets"].float().gt(0.5)

    fixed_types = hierarchy_outputs.get(
        "fixed_type_ids", fine_outputs["fixed_type_ids"]
    ).long()
    type_matches = batch["type_candidates"].long().eq(fixed_types.unsqueeze(-1))
    type_correct = (type_matches & batch["gold_type_mask"].bool()).any(dim=-1)
    type_eligible = (
        type_correct if require_correct_type else torch.ones_like(type_correct)
    )

    gold_positive = batch["gold_region_positive_mask"].bool()
    layer2_candidate = outputs["layer2_candidate_mask"].bool()
    layer2_positive = layer2_candidate & gold_positive
    has_layer2_positive = layer2_positive.any(dim=-1)
    has_any_visible_mapping = (
        fine_outputs["candidate_mask"].bool() & gold_positive
    ).any(dim=-1)
    current_visible = outputs["current_visible_mask"].bool()
    current_index = (
        outputs["current_region_indices"].long().clamp(0, gold_positive.size(-1) - 1)
    )
    current_real_correct = gold_positive.gather(
        -1, current_index.unsqueeze(-1)
    ).squeeze(-1)
    current_correct = torch.where(
        target_visible,
        current_visible & current_real_correct,
        ~current_visible,
    )

    base_eligible = deployable & gold_span & visibility_known & type_eligible
    attribution_known = ~target_visible | has_any_visible_mapping
    eligible = base_eligible & attribution_known
    keep = eligible & current_correct
    layer1_valid = outputs["layer1_valid_mask"].bool()
    to_null = (
        eligible
        & ~target_visible
        & current_visible
        & layer1_valid[..., ACTION_TO_NULL]
    )
    to_visible = (
        eligible
        & target_visible
        & ~current_correct
        & has_layer2_positive
        & layer1_valid[..., ACTION_TO_VISIBLE]
    )
    supervised = keep | to_null | to_visible

    labels = torch.full_like(fixed_types, ACTION_KEEP)
    labels = torch.where(to_null, torch.full_like(labels, ACTION_TO_NULL), labels)
    labels = torch.where(to_visible, torch.full_like(labels, ACTION_TO_VISIBLE), labels)
    # Wrong span/type, missing Top-4 positives, and uncertain mappings are not
    # action labels. They still protect the frozen deployed decision.
    preservation = deployable & ~to_null & ~to_visible
    excluded = deployable & ~supervised
    return {
        "deployable_mask": deployable,
        "eligible_mask": eligible,
        "type_correct_mask": type_correct,
        "target_visible_mask": target_visible,
        "current_correct_mask": current_correct,
        "layer2_positive_mask": layer2_positive,
        "has_layer2_positive_mask": has_layer2_positive,
        "keep_mask": keep,
        "to_null_mask": to_null,
        "to_visible_mask": to_visible,
        "actionable_mask": to_null | to_visible,
        "supervised_mask": supervised,
        "preservation_mask": preservation,
        "excluded_mask": excluded,
        "uncovered_visible_mask": (
            base_eligible & target_visible & ~has_layer2_positive
        ),
        "uncertain_mapping_mask": (
            base_eligible & target_visible & ~has_any_visible_mapping
        ),
        "layer1_labels": labels,
    }


def layered_action_verifier_loss(
    outputs: dict[str, torch.Tensor],
    fine_outputs: dict[str, torch.Tensor],
    hierarchy_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_layer1: float = 1.0,
    lambda_layer2: float = 1.0,
    lambda_keep_margin: float = 0.5,
    lambda_correction_margin: float = 0.5,
    lambda_preservation: float = 0.2,
    lambda_residual: float = 0.02,
    keep_margin: float = 0.5,
    correction_margin: float = 0.5,
    keep_group_weight: float = 0.20,
    to_null_group_weight: float = 0.40,
    to_visible_group_weight: float = 0.40,
    false_release_weight: float = 3.0,
    missed_release_weight: float = 1.0,
    stage1_spans_only: bool = True,
    require_correct_type: bool = True,
) -> dict[str, torch.Tensor]:
    supervision = layered_action_supervision(
        outputs,
        fine_outputs,
        hierarchy_outputs,
        batch,
        stage1_spans_only=stage1_spans_only,
        require_correct_type=require_correct_type,
    )
    layer1_logits = (
        outputs["layer1_logits"]
        .float()
        .masked_fill(~outputs["layer1_valid_mask"].bool(), -1e4)
    )
    layer1_terms = F.cross_entropy(
        layer1_logits.transpose(1, 2),
        supervision["layer1_labels"],
        reduction="none",
    )
    keep = supervision["keep_mask"]
    to_null = supervision["to_null_mask"]
    to_visible = supervision["to_visible_mask"]
    release_mode = "release_advantage_logits" in outputs
    release_negative = supervision["deployable_mask"] & ~to_visible
    if release_mode:
        advantage = outputs["release_advantage_logits"].float()
        positive_bce = F.softplus(-advantage)
        negative_bce = F.softplus(advantage)
        loss_layer1 = _weighted_available(
            [
                (
                    _mean(negative_bce, release_negative),
                    false_release_weight,
                    bool(release_negative.any()),
                ),
                (
                    _mean(positive_bce, to_visible),
                    missed_release_weight,
                    bool(to_visible.any()),
                ),
            ],
            advantage,
        )
    else:
        loss_layer1 = _weighted_available(
            [
                (_mean(layer1_terms, keep), keep_group_weight, bool(keep.any())),
                (
                    _mean(layer1_terms, to_null),
                    to_null_group_weight,
                    bool(to_null.any()),
                ),
                (
                    _mean(layer1_terms, to_visible),
                    to_visible_group_weight,
                    bool(to_visible.any()),
                ),
            ],
            layer1_logits,
        )

    layer2_scores = outputs["layer2_scores"].float()
    layer2_candidate = outputs["layer2_candidate_mask"].bool()
    layer2_positive = supervision["layer2_positive_mask"]
    log_denominator = torch.logsumexp(
        layer2_scores.masked_fill(~layer2_candidate, -1e4), dim=-1
    )
    log_positive = torch.logsumexp(
        layer2_scores.masked_fill(~layer2_positive, -1e4), dim=-1
    )
    loss_layer2 = _mean(log_denominator - log_positive, to_visible)

    keep_score = layer1_logits[..., ACTION_KEEP]
    non_keep = layer1_logits[..., 1:].max(dim=-1).values
    keep_terms = F.relu(float(keep_margin) - keep_score + non_keep)
    loss_keep_margin = _mean(
        keep_terms, release_negative if release_mode else keep
    )

    target_score = layer1_logits.gather(
        -1, supervision["layer1_labels"].unsqueeze(-1)
    ).squeeze(-1)
    correction_terms = F.relu(float(correction_margin) - target_score + keep_score)
    loss_correction_margin = _mean(correction_terms, supervision["actionable_mask"])

    keep_log_probability = F.log_softmax(layer1_logits, dim=-1)[..., ACTION_KEEP]
    loss_preservation = _mean(-keep_log_probability, supervision["preservation_mask"])
    residual = outputs["bounded_layer2_delta_logits"].float().pow(2)
    loss_residual = _mean(residual, layer2_candidate)

    loss = (
        float(lambda_layer1) * loss_layer1
        + float(lambda_layer2) * loss_layer2
        + float(lambda_keep_margin) * loss_keep_margin
        + float(lambda_correction_margin) * loss_correction_margin
        + float(lambda_preservation) * loss_preservation
        + float(lambda_residual) * loss_residual
    )
    return {
        "loss": loss,
        "loss_layer1": loss_layer1,
        "loss_layer2": loss_layer2,
        "loss_keep_margin": loss_keep_margin,
        "loss_correction_margin": loss_correction_margin,
        "loss_preservation": loss_preservation,
        "loss_residual": loss_residual,
        "deployable_count": supervision["deployable_mask"].sum(),
        "eligible_count": supervision["eligible_mask"].sum(),
        "supervised_count": supervision["supervised_mask"].sum(),
        "keep_count": keep.sum(),
        "to_null_count": to_null.sum(),
        "to_visible_count": to_visible.sum(),
        "release_positive_count": to_visible.sum(),
        "release_negative_count": release_negative.sum(),
        "preservation_count": supervision["preservation_mask"].sum(),
        "excluded_count": supervision["excluded_mask"].sum(),
        "uncovered_visible_count": supervision["uncovered_visible_mask"].sum(),
        "uncertain_mapping_count": supervision["uncertain_mapping_mask"].sum(),
    }
