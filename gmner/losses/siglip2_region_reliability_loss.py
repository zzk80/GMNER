"""Span-balanced M3.4A supervision for absolute region reliability."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _masked_topk(
    scores: torch.Tensor, mask: torch.Tensor, k: int
) -> torch.Tensor:
    selected = torch.zeros_like(mask, dtype=torch.bool)
    usable = min(max(int(k), 0), scores.size(-1))
    if usable == 0:
        return selected
    safe = scores.float().masked_fill(~mask.bool(), -1e4)
    values, indices = safe.topk(usable, dim=-1)
    selected.scatter_(-1, indices, values.gt(-1000.0))
    return selected & mask.bool()


def _gather_mask(indices: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(valid, dtype=torch.bool)
    output.scatter_(-1, indices.long().unsqueeze(-1), True)
    return output & valid.bool()


def _span_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (values * mask.to(values.dtype)).sum(dim=-1)
    denominator = mask.sum(dim=-1).clamp_min(1).to(values.dtype)
    return numerator / denominator


def _grouped_mean(
    values: torch.Tensor,
    groups: list[tuple[torch.Tensor, float]],
) -> torch.Tensor:
    active = [
        (values[mask].mean(), float(weight))
        for mask, weight in groups
        if mask.any() and float(weight) > 0.0
    ]
    if not active:
        return _zero(values)
    denominator = max(sum(weight for _, weight in active), 1e-8)
    return sum(value * weight for value, weight in active) / denominator


def reliability_quality_target(
    iou: torch.Tensor,
    *,
    low_iou: float,
    positive_iou: float,
) -> torch.Tensor:
    scale = max(float(positive_iou) - float(low_iou), 1e-4)
    return ((iou.float() - float(low_iou)) / scale).clamp(0.0, 1.0)


def siglip2_region_reliability_supervision(
    outputs: dict[str, torch.Tensor],
    fine_outputs: dict[str, torch.Tensor],
    hierarchy_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    baseline_visible_mask: torch.Tensor,
    low_iou: float = 0.1,
    positive_iou: float = 0.5,
    hard_negative_count: int = 4,
    other_entity_negative_count: int = 2,
    compatibility_negative_count: int = 2,
) -> dict[str, torch.Tensor]:
    candidate = outputs["candidate_mask"].bool()
    positive = batch["gold_region_positive_mask"].bool() & candidate
    visible = batch["visibility_targets"].float().gt(0.5)
    fixed_types = hierarchy_outputs["fixed_type_ids"].long()
    type_slots = (
        batch["type_candidates"].long().eq(fixed_types.unsqueeze(-1))
    ).float().argmax(dim=-1)
    type_correct = batch["gold_type_mask"].bool().gather(
        -1, type_slots.unsqueeze(-1)
    ).squeeze(-1)
    eligible = (
        batch["span_mask"].bool()
        & batch["span_source_ids"].long().eq(0)
        & batch["gold_span_mask"].bool()
        & batch["visibility_targets"].float().ge(0.0)
        & type_correct
    )
    fine_top1 = outputs["fine_top1_region_index"].long()
    base_top1 = fine_outputs["base_log_prior"].float().argmax(dim=-1)
    coarse_top1 = fine_outputs["coarse_log_prior"].float().argmax(dim=-1)
    fine_top1_mask = _gather_mask(fine_top1, candidate)
    base_top1_mask = _gather_mask(base_top1, candidate)
    coarse_top1_mask = _gather_mask(coarse_top1, candidate)
    fine_correct = positive.gather(
        -1, fine_top1.unsqueeze(-1)
    ).squeeze(-1)
    has_positive = positive.any(dim=-1)

    wrong = candidate & ~positive
    fine_hard = _masked_topk(
        fine_outputs["final_region_logits"].float(), wrong, hard_negative_count
    )
    other_positive = (
        batch["gold_region_positive_mask"].bool().any(dim=1)[:, None, :]
        & ~batch["gold_region_positive_mask"].bool()
        & candidate
    )
    other_hard = _masked_topk(
        fine_outputs["final_region_logits"].float(),
        other_positive,
        other_entity_negative_count,
    )
    compatibility_hard = _masked_topk(
        fine_outputs["fixed_type_region_compatibility"].float(),
        wrong,
        compatibility_negative_count,
    )
    promoted_wrong = fine_outputs["promoted_candidate_mask"].bool() & wrong
    selected = (
        positive
        | fine_top1_mask
        | base_top1_mask
        | coarse_top1_mask
        | fine_hard
        | other_hard
        | compatibility_hard
        | promoted_wrong
    ) & candidate & eligible.unsqueeze(-1)

    quality = reliability_quality_target(
        batch["region_iou_targets"],
        low_iou=low_iou,
        positive_iou=positive_iou,
    )
    quality = quality * visible.unsqueeze(-1).to(quality.dtype)
    baseline_visible = baseline_visible_mask.bool()
    group_a = eligible & visible & ~baseline_visible & fine_correct
    group_b = eligible & visible & ~baseline_visible & ~fine_correct
    group_b_hard = group_b & has_positive
    group_b_uncovered = group_b & ~has_positive
    group_null = eligible & ~visible
    group_ordinary = (eligible & visible & baseline_visible) | group_b_uncovered
    return {
        "eligible_mask": eligible,
        "type_correct_mask": type_correct,
        "visible_mask": visible,
        "candidate_mask": candidate,
        "selected_candidate_mask": selected,
        "positive_mask": positive,
        "quality_target": quality,
        "fine_top1_mask": fine_top1_mask,
        "base_top1_mask": base_top1_mask,
        "coarse_top1_mask": coarse_top1_mask,
        "fine_hard_negative_mask": fine_hard,
        "other_entity_negative_mask": other_hard,
        "compatibility_negative_mask": compatibility_hard,
        "promoted_wrong_mask": promoted_wrong,
        "fine_correct_mask": fine_correct,
        "candidate_covered_mask": has_positive,
        "group_a_mask": group_a,
        "group_b_mask": group_b,
        "group_b_hard_mask": group_b_hard,
        "group_b_uncovered_mask": group_b_uncovered,
        "group_null_mask": group_null,
        "group_ordinary_mask": group_ordinary,
    }


def siglip2_region_reliability_loss(
    outputs: dict[str, torch.Tensor],
    fine_outputs: dict[str, torch.Tensor],
    hierarchy_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    baseline_visible_mask: torch.Tensor,
    lambda_quality_focal: float = 1.0,
    lambda_positive_max: float = 0.5,
    lambda_null_suppress: float = 0.5,
    lambda_rank: float = 0.5,
    lambda_brier: float = 0.2,
    lambda_hard_ab_bce: float = 1.0,
    lambda_hard_ab_rank: float = 0.5,
    quality_focal_gamma: float = 2.0,
    rank_margin: float = 0.5,
    hard_ab_rank_margin: float = 0.5,
    low_iou: float = 0.1,
    positive_iou: float = 0.5,
    hard_negative_count: int = 4,
    other_entity_negative_count: int = 2,
    compatibility_negative_count: int = 2,
    group_a_weight: float = 0.30,
    group_b_weight: float = 0.30,
    group_null_weight: float = 0.20,
    group_ordinary_weight: float = 0.20,
    positive_pair_weight: float = 2.0,
    high_score_negative_weight: float = 2.0,
    promoted_negative_weight: float = 1.5,
    other_entity_negative_weight: float = 1.5,
    compatibility_negative_weight: float = 1.5,
) -> dict[str, torch.Tensor]:
    supervision = siglip2_region_reliability_supervision(
        outputs,
        fine_outputs,
        hierarchy_outputs,
        batch,
        baseline_visible_mask=baseline_visible_mask,
        low_iou=low_iou,
        positive_iou=positive_iou,
        hard_negative_count=hard_negative_count,
        other_entity_negative_count=other_entity_negative_count,
        compatibility_negative_count=compatibility_negative_count,
    )
    logits = outputs["reliability_logits"].float()
    probability = outputs["reliability_probability"].float()
    target = supervision["quality_target"].float()
    selected = supervision["selected_candidate_mask"].bool()
    pair_weight = torch.ones_like(target)
    pair_weight = torch.where(
        supervision["positive_mask"],
        torch.full_like(pair_weight, float(positive_pair_weight)),
        pair_weight,
    )
    high_score_wrong = (
        supervision["fine_top1_mask"]
        | supervision["base_top1_mask"]
        | supervision["coarse_top1_mask"]
        | supervision["fine_hard_negative_mask"]
    ) & ~supervision["positive_mask"]
    pair_weight = torch.where(
        high_score_wrong,
        torch.full_like(pair_weight, float(high_score_negative_weight)),
        pair_weight,
    )
    pair_weight = torch.where(
        supervision["promoted_wrong_mask"],
        torch.full_like(pair_weight, float(promoted_negative_weight)),
        pair_weight,
    )
    pair_weight = torch.where(
        supervision["other_entity_negative_mask"],
        torch.full_like(pair_weight, float(other_entity_negative_weight)),
        pair_weight,
    )
    pair_weight = torch.where(
        supervision["compatibility_negative_mask"],
        torch.full_like(pair_weight, float(compatibility_negative_weight)),
        pair_weight,
    )
    focal = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ) * (target - probability).abs().pow(float(quality_focal_gamma))
    focal_span = _span_mean(focal * pair_weight, selected)
    brier_span = _span_mean((probability - target).pow(2), selected)
    groups = [
        (supervision["group_a_mask"], group_a_weight),
        (supervision["group_b_hard_mask"], group_b_weight),
        (supervision["group_null_mask"], group_null_weight),
        (supervision["group_ordinary_mask"], group_ordinary_weight),
    ]
    loss_quality = _grouped_mean(focal_span, groups)
    loss_brier = _grouped_mean(brier_span, groups)

    positive = supervision["positive_mask"]
    negative = selected & ~positive
    positive_probability = probability.masked_fill(~positive, -1.0).max(
        dim=-1
    ).values
    positive_valid = supervision["eligible_mask"] & positive.any(dim=-1)
    positive_max_terms = -positive_probability.clamp_min(1e-6).log()
    positive_groups = [
        (mask & positive_valid, weight)
        for mask, weight in (groups[:2] + groups[3:])
    ]
    loss_positive = _grouped_mean(positive_max_terms, positive_groups)

    max_probability = probability.masked_fill(~selected, -1.0).max(dim=-1).values
    null_mask = supervision["group_null_mask"] & selected.any(dim=-1)
    loss_null = (
        -(1.0 - max_probability[null_mask]).clamp_min(1e-6).log().mean()
        if null_mask.any()
        else _zero(logits)
    )
    positive_logits = logits.masked_fill(~positive, -1e4).max(dim=-1).values
    negative_logits = logits.masked_fill(~negative, -1e4).max(dim=-1).values
    rank_valid = positive_valid & negative.any(dim=-1)
    rank_terms = F.relu(float(rank_margin) - positive_logits + negative_logits)
    loss_rank = rank_terms[rank_valid].mean() if rank_valid.any() else _zero(logits)

    fine_top1_logits = logits.gather(
        -1, outputs["fine_top1_region_index"].long().unsqueeze(-1)
    ).squeeze(-1)
    group_a = supervision["group_a_mask"]
    group_b_hard = supervision["group_b_hard_mask"]
    hard_ab_bce = F.binary_cross_entropy_with_logits(
        fine_top1_logits,
        group_a.to(fine_top1_logits.dtype),
        reduction="none",
    )
    loss_hard_ab_bce = _grouped_mean(
        hard_ab_bce, [(group_a, 0.5), (group_b_hard, 0.5)]
    )
    if group_a.any() and group_b_hard.any():
        a_logits = fine_top1_logits[group_a]
        b_logits = fine_top1_logits[group_b_hard]
        loss_hard_ab_rank = F.softplus(
            float(hard_ab_rank_margin)
            - a_logits[:, None]
            + b_logits[None, :]
        ).mean()
    else:
        loss_hard_ab_rank = _zero(logits)
    loss = (
        float(lambda_quality_focal) * loss_quality
        + float(lambda_positive_max) * loss_positive
        + float(lambda_null_suppress) * loss_null
        + float(lambda_rank) * loss_rank
        + float(lambda_brier) * loss_brier
        + float(lambda_hard_ab_bce) * loss_hard_ab_bce
        + float(lambda_hard_ab_rank) * loss_hard_ab_rank
    )
    return {
        "loss": loss,
        "loss_quality_focal": loss_quality,
        "loss_positive_max": loss_positive,
        "loss_null_suppress": loss_null,
        "loss_rank": loss_rank,
        "loss_brier": loss_brier,
        "loss_hard_ab_bce": loss_hard_ab_bce,
        "loss_hard_ab_rank": loss_hard_ab_rank,
        "eligible_count": supervision["eligible_mask"].sum(),
        "selected_pair_count": selected.sum(),
        "positive_pair_count": (selected & positive).sum(),
        "hard_negative_pair_count": (selected & high_score_wrong).sum(),
        "other_entity_negative_count": (
            selected & supervision["other_entity_negative_mask"]
        ).sum(),
        "compatibility_negative_count": (
            selected & supervision["compatibility_negative_mask"]
        ).sum(),
        "promoted_wrong_count": (
            selected & supervision["promoted_wrong_mask"]
        ).sum(),
        "group_a_count": supervision["group_a_mask"].sum(),
        "group_b_count": supervision["group_b_mask"].sum(),
        "group_b_hard_count": supervision["group_b_hard_mask"].sum(),
        "group_b_uncovered_count": supervision["group_b_uncovered_mask"].sum(),
        "group_null_count": supervision["group_null_mask"].sum(),
        "group_ordinary_count": supervision["group_ordinary_mask"].sum(),
    }
