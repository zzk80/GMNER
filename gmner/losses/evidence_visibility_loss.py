"""Action-aware objectives for the frozen-region evidence visibility head."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if mask.any() else _zero(values)


def _weighted_terms(
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


def evidence_visibility_supervision(
    outputs: dict[str, torch.Tensor],
    fine_outputs: dict[str, torch.Tensor],
    hierarchy_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    baseline_visible_mask: torch.Tensor,
    uncertainty_entropy_threshold: float = 0.65,
    uncertainty_margin_threshold: float = 0.08,
) -> dict[str, torch.Tensor]:
    eligible = (
        batch["span_mask"].bool()
        & batch["span_source_ids"].long().eq(0)
        & batch["gold_span_mask"].bool()
        & batch["visibility_targets"].float().ge(0.0)
    )
    fixed_slots = hierarchy_outputs.get("fixed_type_slots")
    if fixed_slots is None:
        fixed_types = hierarchy_outputs["fixed_type_ids"].long()
        fixed_slots = (
            batch["type_candidates"].long().eq(fixed_types.unsqueeze(-1))
        ).float().argmax(dim=-1)
    fixed_slots = fixed_slots.long().clamp(
        0, batch["gold_type_mask"].size(-1) - 1
    )
    type_correct = batch["gold_type_mask"].bool().gather(
        -1, fixed_slots.unsqueeze(-1)
    ).squeeze(-1)

    fine_indices = outputs["fine_top1_region_index"].long().clamp(
        0, batch["gold_region_positive_mask"].size(-1) - 1
    )
    fine_correct = batch["gold_region_positive_mask"].bool().gather(
        -1, fine_indices.unsqueeze(-1)
    ).squeeze(-1)
    has_candidate = fine_outputs["candidate_mask"].bool().any(dim=-1)
    target_visible = batch["visibility_targets"].float().gt(0.5)
    baseline_visible = baseline_visible_mask.bool()
    triple_eligible = eligible & type_correct & has_candidate

    visible_correction = (
        triple_eligible
        & target_visible
        & fine_correct
        & ~baseline_visible
    )
    visible_preservation = (
        triple_eligible
        & target_visible
        & fine_correct
        & baseline_visible
    )
    null_correction = (
        triple_eligible & ~target_visible & baseline_visible
    )
    # Preserve correct NULL decisions for every gold span, including a span
    # whose fixed type is wrong; changing it cannot improve the final triple.
    null_preservation = eligible & ~target_visible & ~baseline_visible

    uncertain = eligible & (
        outputs["fine_normalized_entropy"].float().ge(
            float(uncertainty_entropy_threshold)
        )
        | outputs["fine_probability_margin"].float().le(
            float(uncertainty_margin_threshold)
        )
        | ~outputs["prior_fine_agreement"].bool()
    )
    strong_action = visible_correction | null_correction
    keep = eligible & ~strong_action & (
        uncertain
        | ~fine_correct
        | visible_preservation
        | null_preservation
        | ~type_correct
    )
    return {
        "eligible_mask": eligible,
        "type_correct_mask": type_correct,
        "fine_correct_mask": fine_correct,
        "target_visible_mask": target_visible,
        "visible_correction_mask": visible_correction,
        "visible_preservation_mask": visible_preservation,
        "null_correction_mask": null_correction,
        "null_preservation_mask": null_preservation,
        "uncertain_mask": uncertain,
        "keep_mask": keep,
    }


def evidence_visibility_loss(
    outputs: dict[str, torch.Tensor],
    fine_outputs: dict[str, torch.Tensor],
    hierarchy_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    baseline_visible_mask: torch.Tensor,
    lambda_bce: float = 1.0,
    lambda_visible_correction: float = 1.0,
    lambda_null_preservation: float = 1.0,
    lambda_keep: float = 0.5,
    lambda_residual: float = 0.05,
    visible_correction_group_weight: float = 0.35,
    visible_preservation_group_weight: float = 0.15,
    null_correction_group_weight: float = 0.20,
    null_preservation_group_weight: float = 0.30,
    visible_margin_gamma: float = 1.0,
    uncertainty_entropy_threshold: float = 0.65,
    uncertainty_margin_threshold: float = 0.08,
) -> dict[str, torch.Tensor]:
    supervision = evidence_visibility_supervision(
        outputs,
        fine_outputs,
        hierarchy_outputs,
        batch,
        baseline_visible_mask=baseline_visible_mask,
        uncertainty_entropy_threshold=uncertainty_entropy_threshold,
        uncertainty_margin_threshold=uncertainty_margin_threshold,
    )
    logits = outputs["final_visibility_logits"].float()
    target_visible = supervision["target_visible_mask"].float()
    bce = F.binary_cross_entropy_with_logits(
        logits, target_visible, reduction="none"
    )
    visible_correction = supervision["visible_correction_mask"]
    visible_preservation = supervision["visible_preservation_mask"]
    null_correction = supervision["null_correction_mask"]
    null_preservation = supervision["null_preservation_mask"]
    loss_bce = _weighted_terms(
        [
            (
                _mean(bce, visible_correction),
                visible_correction_group_weight,
                bool(visible_correction.any()),
            ),
            (
                _mean(bce, visible_preservation),
                visible_preservation_group_weight,
                bool(visible_preservation.any()),
            ),
            (
                _mean(bce, null_correction),
                null_correction_group_weight,
                bool(null_correction.any()),
            ),
            (
                _mean(bce, null_preservation),
                null_preservation_group_weight,
                bool(null_preservation.any()),
            ),
        ],
        logits,
    )

    correction_weight = 1.0 + float(visible_margin_gamma) * outputs[
        "fine_probability_margin"
    ].float().detach()
    correction_terms = F.softplus(-logits) * correction_weight
    loss_visible_correction = _mean(
        correction_terms, visible_correction
    )
    loss_null_preservation = _mean(F.softplus(logits), null_preservation)

    base_probability = outputs["base_visibility_probability"].float().detach()
    final_probability = outputs["final_visibility_probability"].float()
    base_probability = base_probability.clamp(1e-5, 1.0 - 1e-5)
    final_probability = final_probability.clamp(1e-5, 1.0 - 1e-5)
    keep_kl = (
        base_probability
        * (base_probability.log() - final_probability.log())
        + (1.0 - base_probability)
        * (
            (1.0 - base_probability).log()
            - (1.0 - final_probability).log()
        )
    )
    loss_keep = _mean(keep_kl, supervision["keep_mask"])
    preservation = (
        visible_preservation
        | null_preservation
        | supervision["keep_mask"]
    )
    loss_residual = _mean(
        outputs["bounded_visibility_delta_logits"].float().pow(2),
        preservation,
    )
    loss = (
        float(lambda_bce) * loss_bce
        + float(lambda_visible_correction) * loss_visible_correction
        + float(lambda_null_preservation) * loss_null_preservation
        + float(lambda_keep) * loss_keep
        + float(lambda_residual) * loss_residual
    )
    return {
        "loss": loss,
        "loss_bce": loss_bce,
        "loss_visible_correction": loss_visible_correction,
        "loss_null_preservation": loss_null_preservation,
        "loss_keep": loss_keep,
        "loss_residual": loss_residual,
        "eligible_count": supervision["eligible_mask"].sum(),
        "type_correct_count": (
            supervision["eligible_mask"]
            & supervision["type_correct_mask"]
        ).sum(),
        "visible_correction_count": visible_correction.sum(),
        "visible_preservation_count": visible_preservation.sum(),
        "null_correction_count": null_correction.sum(),
        "null_preservation_count": null_preservation.sum(),
        "uncertain_count": supervision["uncertain_mask"].sum(),
        "keep_count": supervision["keep_mask"].sum(),
    }
