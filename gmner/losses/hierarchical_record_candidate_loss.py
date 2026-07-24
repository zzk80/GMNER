"""Factorized objectives for the hierarchical record verifier."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gmner.models.hierarchical_action_controller import (
    balanced_keep_regions,
    prepend_keep_score,
    union_topk_action_mask,
)


OVERRIDE_NEUTRAL = 0
OVERRIDE_FIX = 1
OVERRIDE_DAMAGE = 2


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def build_override_utility_targets(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    require_correct_type: bool = True,
    stage1_only: bool = True,
) -> dict[str, torch.Tensor]:
    """Label the relative value of replacing Stage1 with the raw ranker region."""

    span_mask = batch["span_mask"].bool()
    action_mask = span_mask
    if stage1_only:
        action_mask = action_mask & batch["span_source_ids"].long().eq(0)

    real_mask = outputs["real_region_mask"].bool()
    base_indices = outputs["base_region_indices"].long().clamp(
        0, real_mask.size(-1) - 1
    )
    proposed_indices = outputs["best_real_region_index"].long().clamp(
        0, real_mask.size(-1) - 1
    )
    base_is_real = real_mask.gather(
        -1, base_indices.unsqueeze(-1)
    ).squeeze(-1)
    proposed_is_real = real_mask.gather(
        -1, proposed_indices.unsqueeze(-1)
    ).squeeze(-1)
    action_mask = (
        action_mask
        & base_is_real
        & proposed_is_real
        & proposed_indices.ne(base_indices)
    )

    positives = batch["gold_region_positive_mask"].bool() & real_mask
    base_correct = positives.gather(
        -1, base_indices.unsqueeze(-1)
    ).squeeze(-1)
    proposed_correct = positives.gather(
        -1, proposed_indices.unsqueeze(-1)
    ).squeeze(-1)
    gold_visible = (
        batch["gold_span_mask"].bool()
        & batch["visibility_targets"].float().gt(0.5)
    )

    type_correct = batch["gold_span_mask"].bool()
    if require_correct_type:
        fixed_slots = outputs.get("fixed_type_slots")
        if fixed_slots is None:
            fixed_types = outputs["fixed_type_ids"].long()
            fixed_slots = (
                batch["type_candidates"].long().eq(fixed_types.unsqueeze(-1))
            ).float().argmax(dim=-1)
        fixed_slots = fixed_slots.long().clamp(
            0, batch["gold_type_mask"].size(-1) - 1
        )
        type_correct = batch["gold_type_mask"].bool().gather(
            -1, fixed_slots.unsqueeze(-1)
        ).squeeze(-1)

    triple_relevant = gold_visible & type_correct
    fix_mask = (
        action_mask & triple_relevant & ~base_correct & proposed_correct
    )
    damage_mask = (
        action_mask & triple_relevant & base_correct & ~proposed_correct
    )
    targets = torch.full_like(base_indices, OVERRIDE_NEUTRAL)
    targets = targets.masked_fill(fix_mask, OVERRIDE_FIX)
    targets = targets.masked_fill(damage_mask, OVERRIDE_DAMAGE)
    neutral_mask = action_mask & ~fix_mask & ~damage_mask
    return {
        "targets": targets,
        "valid_mask": action_mask,
        "fix_mask": fix_mask,
        "damage_mask": damage_mask,
        "neutral_mask": neutral_mask,
        "base_correct": base_correct,
        "proposed_correct": proposed_correct,
    }


def build_action_controller_targets(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    top_k: int = 4,
    enable_visibility_correction: bool = True,
    visible_from_null_threshold: float = 0.8,
    null_from_visible_threshold: float = 0.2,
    require_correct_type: bool = True,
    stage1_only: bool = True,
) -> dict[str, torch.Tensor]:
    """Label TO_NULL and fused TO_REAL actions relative to balanced KEEP."""

    span_mask = batch["span_mask"].bool() & batch["gold_span_mask"].bool()
    if stage1_only:
        span_mask &= batch["span_source_ids"].long().eq(0)
    if require_correct_type:
        fixed_slots = outputs["fixed_type_slots"].long().clamp(
            0, batch["gold_type_mask"].size(-1) - 1
        )
        type_correct = batch["gold_type_mask"].bool().gather(
            -1, fixed_slots.unsqueeze(-1)
        ).squeeze(-1)
        span_mask &= type_correct

    keep = balanced_keep_regions(
        outputs,
        batch,
        enable_visibility_correction=enable_visibility_correction,
        visible_from_null_threshold=visible_from_null_threshold,
        null_from_visible_threshold=null_from_visible_threshold,
    )
    keep_indices = keep["region_indices"].long().clamp(
        0, batch["region_mask"].size(-1) - 1
    )
    positives = batch["gold_region_positive_mask"].bool()
    keep_correct = positives.gather(
        -1, keep_indices.unsqueeze(-1)
    ).squeeze(-1) & span_mask

    real_candidate_mask = union_topk_action_mask(
        fused_logits=outputs["final_region_logits"],
        residual_logits=outputs["region_residual_logits"],
        base_logits=outputs["base_region_scores"],
        real_mask=outputs["real_region_mask"].bool(),
        keep_indices=keep_indices,
        top_k=top_k,
    ) & span_mask.unsqueeze(-1)
    real_correct = positives & real_candidate_mask
    real_fix = real_candidate_mask & ~keep_correct.unsqueeze(-1) & real_correct
    real_damage = real_candidate_mask & keep_correct.unsqueeze(-1) & ~real_correct
    real_targets = torch.full_like(real_candidate_mask, OVERRIDE_NEUTRAL, dtype=torch.long)
    real_targets = real_targets.masked_fill(real_fix, OVERRIDE_FIX)
    real_targets = real_targets.masked_fill(real_damage, OVERRIDE_DAMAGE)

    null_indices = keep["null_indices"].long().clamp(
        0, batch["region_mask"].size(-1) - 1
    )
    null_valid = (
        span_mask
        & keep["has_null_region"]
        & keep_indices.ne(null_indices)
    )
    null_correct = positives.gather(
        -1, null_indices.unsqueeze(-1)
    ).squeeze(-1)
    null_fix = null_valid & ~keep_correct & null_correct
    null_damage = null_valid & keep_correct & ~null_correct
    null_targets = torch.full_like(keep_indices, OVERRIDE_NEUTRAL)
    null_targets = null_targets.masked_fill(null_fix, OVERRIDE_FIX)
    null_targets = null_targets.masked_fill(null_damage, OVERRIDE_DAMAGE)

    fix_mask = torch.cat([null_fix.unsqueeze(-1), real_fix], dim=-1)
    damage_mask = torch.cat([null_damage.unsqueeze(-1), real_damage], dim=-1)
    valid_mask = torch.cat([null_valid.unsqueeze(-1), real_candidate_mask], dim=-1)
    targets = torch.cat([null_targets.unsqueeze(-1), real_targets], dim=-1)
    neutral_mask = valid_mask & ~fix_mask & ~damage_mask
    fixable_span_mask = fix_mask.any(dim=-1)
    keep_positive_mask = span_mask & ~fixable_span_mask
    safe_neutral_mask = neutral_mask & keep_correct.unsqueeze(-1)
    useless_neutral_mask = neutral_mask & ~keep_correct.unsqueeze(-1)
    return {
        "targets": targets,
        "valid_mask": valid_mask,
        "fix_mask": fix_mask,
        "damage_mask": damage_mask,
        "neutral_mask": neutral_mask,
        "span_mask": span_mask,
        "keep_indices": keep_indices,
        "keep_correct": keep_correct,
        "null_indices": null_indices,
        "null_valid_mask": null_valid,
        "real_candidate_mask": real_candidate_mask,
        "fixable_span_mask": fixable_span_mask,
        "keep_positive_mask": keep_positive_mask,
        "safe_neutral_mask": safe_neutral_mask,
        "useless_neutral_mask": useless_neutral_mask,
        "preserve_span_mask": keep_correct & damage_mask.any(dim=-1),
    }


def _topk_mask(
    scores: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Select online hard examples independently inside each span."""

    if top_k <= 0 or not mask.any():
        return torch.zeros_like(mask)
    count = min(int(top_k), scores.size(-1))
    indices = scores.float().masked_fill(~mask, -1e4).topk(count, dim=-1).indices
    selected = torch.zeros_like(mask)
    selected.scatter_(-1, indices, True)
    return selected & mask


def _group_balanced_mean(
    terms: torch.Tensor,
    groups: tuple[tuple[torch.Tensor, float], ...],
) -> torch.Tensor:
    """Give fixable/protection/ordinary spans explicit population mass."""

    numerator = _zero(terms)
    denominator = terms.new_zeros(())
    for mask, raw_weight in groups:
        weight = float(raw_weight)
        if weight <= 0.0 or not mask.any():
            continue
        numerator = numerator + weight * terms[mask].mean()
        denominator = denominator + weight
    return numerator / denominator.clamp_min(1e-8)


def _listwise_action_policy_losses(
    action_scores: torch.Tensor,
    action_info: dict[str, torch.Tensor],
    *,
    hard_damage_k: int,
    hard_neutral_k: int,
    fix_margin: float,
    damage_margin: float,
    neutral_margin: float,
    risk_damage_cost: float,
    risk_neutral_cost: float,
    fixable_group_weight: float,
    preserve_group_weight: float,
    ordinary_group_weight: float,
) -> dict[str, torch.Tensor]:
    """Rank KEEP and actions jointly, with multi-positive FIX supervision."""

    valid_actions = action_info["valid_mask"]
    valid_spans = valid_actions.any(dim=-1)
    fix_mask = action_info["fix_mask"]
    hard_damage = _topk_mask(
        action_scores, action_info["damage_mask"], top_k=hard_damage_k
    )
    hard_neutral = _topk_mask(
        action_scores, action_info["neutral_mask"], top_k=hard_neutral_k
    )
    mined_actions = fix_mask | hard_damage | hard_neutral

    policy_scores = prepend_keep_score(action_scores.float())
    policy_valid = torch.cat([valid_spans.unsqueeze(-1), mined_actions], dim=-1)
    policy_positive = torch.cat(
        [action_info["keep_positive_mask"].unsqueeze(-1), fix_mask], dim=-1
    ) & policy_valid
    denominator = torch.logsumexp(
        policy_scores.masked_fill(~policy_valid, -1e4), dim=-1
    )
    numerator = torch.logsumexp(
        policy_scores.masked_fill(~policy_positive, -1e4), dim=-1
    )
    listwise_terms = denominator - numerator

    fixable = valid_spans & action_info["fixable_span_mask"]
    preserve = valid_spans & ~fixable & action_info["preserve_span_mask"]
    ordinary = valid_spans & ~fixable & ~preserve
    loss_listwise = _group_balanced_mean(
        listwise_terms,
        (
            (fixable, fixable_group_weight),
            (preserve, preserve_group_weight),
            (ordinary, ordinary_group_weight),
        ),
    )

    full_policy_scores = prepend_keep_score(action_scores.float())
    full_policy_valid = torch.cat(
        [valid_spans.unsqueeze(-1), valid_actions], dim=-1
    )
    full_probabilities = F.softmax(
        full_policy_scores.masked_fill(~full_policy_valid, -1e4), dim=-1
    )
    action_utilities = (
        fix_mask.to(action_scores.dtype)
        - float(risk_damage_cost) * action_info["damage_mask"].to(action_scores.dtype)
        - float(risk_neutral_cost)
        * action_info["neutral_mask"].to(action_scores.dtype)
    )
    policy_utilities = torch.cat(
        [torch.zeros_like(action_utilities[..., :1]), action_utilities], dim=-1
    )
    expected_utility = (full_probabilities * policy_utilities).sum(dim=-1)
    best_utility = fixable.to(expected_utility.dtype)
    regret_terms = best_utility - expected_utility
    loss_expected_regret = _group_balanced_mean(
        regret_terms,
        (
            (fixable, fixable_group_weight),
            (preserve, preserve_group_weight),
            (ordinary, ordinary_group_weight),
        ),
    )

    loss_fix_margin = (
        F.relu(float(fix_margin) - action_scores[fix_mask]).mean()
        if fix_mask.any()
        else _zero(action_scores)
    )
    loss_damage_margin = (
        F.relu(float(damage_margin) + action_scores[hard_damage]).mean()
        if hard_damage.any()
        else _zero(action_scores)
    )
    loss_neutral_cost = (
        F.relu(float(neutral_margin) + action_scores[hard_neutral]).mean()
        if hard_neutral.any()
        else _zero(action_scores)
    )
    return {
        "loss_listwise": loss_listwise,
        "loss_expected_regret": loss_expected_regret,
        "loss_fix_margin": loss_fix_margin,
        "loss_damage_margin": loss_damage_margin,
        "loss_neutral_cost": loss_neutral_cost,
        "hard_damage_mask": hard_damage,
        "hard_neutral_mask": hard_neutral,
    }


def hierarchical_record_candidate_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_entity: float = 1.0,
    lambda_visibility: float = 1.0,
    lambda_region_multi_positive: float = 1.0,
    lambda_region_iou: float = 0.2,
    lambda_region_hard: float = 0.5,
    lambda_region_preserve: float = 0.5,
    lambda_override_utility: float = 0.0,
    lambda_action_listwise: float = 0.0,
    lambda_action_expected_regret: float = 0.0,
    lambda_action_fix_margin: float = 0.0,
    lambda_action_damage_margin: float = 0.0,
    lambda_action_neutral_cost: float = 0.0,
    entity_positive_weight: float = 2.0,
    visibility_positive_weight: float = 2.0,
    visibility_error_weight: float = 3.0,
    visibility_preserve_weight: float = 1.0,
    region_hard_margin: float = 0.2,
    region_preserve_margin: float = 0.2,
    iou_temperature: float = 0.1,
    override_utility_neutral_weight: float = 0.5,
    override_utility_fix_weight: float = 2.0,
    override_utility_damage_weight: float = 3.0,
    override_utility_require_correct_type: bool = True,
    override_utility_stage1_only: bool = True,
    action_top_k: int = 4,
    action_enable_visibility_correction: bool = True,
    action_visible_from_null_threshold: float = 0.8,
    action_null_from_visible_threshold: float = 0.2,
    action_fix_margin: float = 0.5,
    action_damage_margin: float = 0.5,
    action_neutral_margin: float = 0.05,
    action_risk_damage_cost: float = 3.0,
    action_risk_neutral_cost: float = 0.05,
    action_hard_damage_k: int = 3,
    action_hard_neutral_k: int = 2,
    action_fixable_group_weight: float = 0.5,
    action_preserve_group_weight: float = 0.25,
    action_ordinary_group_weight: float = 0.25,
    action_require_correct_type: bool = True,
    action_stage1_only: bool = True,
    source_weights: torch.Tensor | None = None,
    grounding_stage1_only: bool = True,
) -> dict[str, torch.Tensor]:
    """Train entity, visibility, and real-region factors without NULL competition."""

    span_mask = batch["span_mask"].bool()
    gold_span = batch["gold_span_mask"].bool()
    source_ids = batch["span_source_ids"].long()
    stage1_mask = source_ids.eq(0)

    entity_targets = gold_span.float()
    entity_weights = torch.where(
        gold_span,
        torch.full_like(entity_targets, float(entity_positive_weight)),
        torch.ones_like(entity_targets),
    )
    if source_weights is not None:
        source_weights = source_weights.to(
            device=entity_weights.device, dtype=entity_weights.dtype
        )
        safe_source_ids = source_ids.clamp(0, source_weights.numel() - 1)
        entity_weights = entity_weights * source_weights[safe_source_ids]
    entity_terms = F.binary_cross_entropy_with_logits(
        outputs["entityness_logits"], entity_targets, reduction="none"
    )
    loss_entity = (
        entity_terms * entity_weights * span_mask
    ).sum() / span_mask.sum().clamp_min(1)

    supervision_mask = gold_span
    if grounding_stage1_only:
        supervision_mask = supervision_mask & stage1_mask
    visibility_targets = batch["visibility_targets"].float()
    valid_visibility = supervision_mask & visibility_targets.ge(0.0)
    base_indices = batch["base_region_indices"].long().clamp(
        0, batch["region_mask"].size(-1) - 1
    )
    base_is_null = batch["region_is_null"].bool().gather(1, base_indices)
    base_visible = ~base_is_null
    target_visible = visibility_targets.gt(0.5)
    base_visibility_correct = base_visible.eq(target_visible)
    visibility_weights = torch.where(
        target_visible,
        torch.full_like(visibility_targets, float(visibility_positive_weight)),
        torch.ones_like(visibility_targets),
    )
    visibility_weights = visibility_weights * torch.where(
        base_visibility_correct,
        torch.full_like(visibility_targets, float(visibility_preserve_weight)),
        torch.full_like(visibility_targets, float(visibility_error_weight)),
    )
    visibility_terms = F.binary_cross_entropy_with_logits(
        outputs["visibility_logits"],
        visibility_targets.clamp(0.0, 1.0),
        reduction="none",
    )
    loss_visibility = (
        visibility_terms * visibility_weights * valid_visibility
    ).sum() / (
        visibility_weights * valid_visibility
    ).sum().clamp_min(1.0)

    logits = outputs["final_region_logits"]
    real_mask = outputs["real_region_mask"].bool()
    positives = batch["gold_region_positive_mask"].bool() & real_mask
    valid_region = (
        supervision_mask
        & target_visible
        & positives.any(dim=-1)
    )
    if valid_region.any():
        active_logits = logits[valid_region].masked_fill(
            ~real_mask[valid_region], -1e4
        )
        active_positives = positives[valid_region]
        loss_region_multi = (
            torch.logsumexp(active_logits, dim=-1)
            - torch.logsumexp(
                active_logits.masked_fill(~active_positives, -1e4), dim=-1
            )
        ).mean()

        quality = batch["region_iou_targets"].float()[valid_region].clamp(0.0, 1.0)
        temperature = max(float(iou_temperature), 1e-4)
        target_logits = (quality / temperature).masked_fill(
            ~real_mask[valid_region], -1e4
        )
        soft_targets = F.softmax(target_logits, dim=-1)
        log_probabilities = F.log_softmax(active_logits, dim=-1)
        loss_region_iou = -(soft_targets * log_probabilities).sum(dim=-1).mean()
    else:
        loss_region_multi = _zero(logits)
        loss_region_iou = _zero(logits)

    base_real = real_mask.gather(-1, base_indices.unsqueeze(-1)).squeeze(-1)
    base_positive = positives.gather(-1, base_indices.unsqueeze(-1)).squeeze(-1)
    hard_mask = valid_region & base_real & ~base_positive
    if hard_mask.any():
        positive_scores = logits.masked_fill(~positives, -1e4).max(dim=-1).values
        base_scores = logits.gather(-1, base_indices.unsqueeze(-1)).squeeze(-1)
        loss_region_hard = F.relu(
            float(region_hard_margin) - positive_scores[hard_mask] + base_scores[hard_mask]
        ).mean()
    else:
        loss_region_hard = _zero(logits)

    preserve_mask = valid_region & base_positive
    region_indices = torch.arange(logits.size(-1), device=logits.device)
    other_mask = real_mask & region_indices.view(1, 1, -1).ne(
        base_indices.unsqueeze(-1)
    )
    has_other = other_mask.any(dim=-1)
    preserve_mask = preserve_mask & has_other
    if preserve_mask.any():
        base_scores = logits.gather(-1, base_indices.unsqueeze(-1)).squeeze(-1)
        strongest_other = logits.masked_fill(~other_mask, -1e4).max(dim=-1).values
        loss_region_preserve = F.relu(
            float(region_preserve_margin)
            - base_scores[preserve_mask]
            + strongest_other[preserve_mask]
        ).mean()
    else:
        loss_region_preserve = _zero(logits)

    utility_logits = outputs.get("override_utility_logits")
    if utility_logits is None:
        if float(lambda_override_utility) != 0.0:
            raise ValueError(
                "lambda_override_utility is non-zero but the model utility head "
                "is disabled."
            )
        loss_override_utility = _zero(outputs["entityness_logits"])
        utility_info = {
            key: torch.zeros_like(span_mask)
            for key in ("valid_mask", "fix_mask", "damage_mask", "neutral_mask")
        }
    else:
        utility_info = build_override_utility_targets(
            outputs,
            batch,
            require_correct_type=override_utility_require_correct_type,
            stage1_only=override_utility_stage1_only,
        )
        utility_valid = utility_info["valid_mask"]
        if utility_valid.any():
            class_weights = utility_logits.new_tensor(
                [
                    float(override_utility_neutral_weight),
                    float(override_utility_fix_weight),
                    float(override_utility_damage_weight),
                ]
            )
            loss_override_utility = F.cross_entropy(
                utility_logits[utility_valid].float(),
                utility_info["targets"][utility_valid],
                weight=class_weights.float(),
            )
        else:
            loss_override_utility = _zero(utility_logits)

    action_real_scores = outputs.get("action_real_scores")
    action_null_scores = outputs.get("action_null_scores")
    if action_real_scores is None or action_null_scores is None:
        if any(
            float(value) != 0.0
            for value in (
                lambda_action_listwise,
                lambda_action_expected_regret,
                lambda_action_fix_margin,
                lambda_action_damage_margin,
                lambda_action_neutral_cost,
            )
        ):
            raise ValueError(
                "Action-controller loss is enabled but the model action heads "
                "are disabled."
            )
        action_reference = outputs["entityness_logits"]
        loss_action_listwise = _zero(action_reference)
        loss_action_expected_regret = _zero(action_reference)
        loss_action_fix_margin = _zero(action_reference)
        loss_action_damage_margin = _zero(action_reference)
        loss_action_neutral_cost = _zero(action_reference)
        hard_damage_mask = torch.zeros(
            *span_mask.shape, 1, dtype=torch.bool, device=span_mask.device
        )
        hard_neutral_mask = torch.zeros_like(hard_damage_mask)
        action_info = {
            key: torch.zeros_like(span_mask)
            for key in (
                "span_mask",
                "fixable_span_mask",
                "preserve_span_mask",
                "keep_positive_mask",
            )
        }
        action_info.update(
            {
                key: torch.zeros(
                    *span_mask.shape,
                    1,
                    dtype=torch.bool,
                    device=span_mask.device,
                )
                for key in ("valid_mask", "fix_mask", "damage_mask", "neutral_mask")
            }
        )
    else:
        action_info = build_action_controller_targets(
            outputs,
            batch,
            top_k=action_top_k,
            enable_visibility_correction=action_enable_visibility_correction,
            visible_from_null_threshold=action_visible_from_null_threshold,
            null_from_visible_threshold=action_null_from_visible_threshold,
            require_correct_type=action_require_correct_type,
            stage1_only=action_stage1_only,
        )
        action_scores = torch.cat(
            [action_null_scores.unsqueeze(-1), action_real_scores], dim=-1
        )
        action_losses = _listwise_action_policy_losses(
            action_scores,
            action_info,
            hard_damage_k=action_hard_damage_k,
            hard_neutral_k=action_hard_neutral_k,
            fix_margin=action_fix_margin,
            damage_margin=action_damage_margin,
            neutral_margin=action_neutral_margin,
            risk_damage_cost=action_risk_damage_cost,
            risk_neutral_cost=action_risk_neutral_cost,
            fixable_group_weight=action_fixable_group_weight,
            preserve_group_weight=action_preserve_group_weight,
            ordinary_group_weight=action_ordinary_group_weight,
        )
        loss_action_listwise = action_losses["loss_listwise"]
        loss_action_expected_regret = action_losses["loss_expected_regret"]
        loss_action_fix_margin = action_losses["loss_fix_margin"]
        loss_action_damage_margin = action_losses["loss_damage_margin"]
        loss_action_neutral_cost = action_losses["loss_neutral_cost"]
        hard_damage_mask = action_losses["hard_damage_mask"]
        hard_neutral_mask = action_losses["hard_neutral_mask"]

    total = (
        float(lambda_entity) * loss_entity
        + float(lambda_visibility) * loss_visibility
        + float(lambda_region_multi_positive) * loss_region_multi
        + float(lambda_region_iou) * loss_region_iou
        + float(lambda_region_hard) * loss_region_hard
        + float(lambda_region_preserve) * loss_region_preserve
        + float(lambda_override_utility) * loss_override_utility
        + float(lambda_action_listwise) * loss_action_listwise
        + float(lambda_action_expected_regret) * loss_action_expected_regret
        + float(lambda_action_fix_margin) * loss_action_fix_margin
        + float(lambda_action_damage_margin) * loss_action_damage_margin
        + float(lambda_action_neutral_cost) * loss_action_neutral_cost
    )
    return {
        "loss": total,
        "loss_entity": loss_entity,
        "loss_visibility": loss_visibility,
        "loss_region_multi": loss_region_multi,
        "loss_region_iou": loss_region_iou,
        "loss_region_hard": loss_region_hard,
        "loss_region_preserve": loss_region_preserve,
        "loss_override_utility": loss_override_utility,
        "loss_action_listwise": loss_action_listwise,
        "loss_action_expected_regret": loss_action_expected_regret,
        "loss_action_fix_margin": loss_action_fix_margin,
        "loss_action_damage_margin": loss_action_damage_margin,
        "loss_action_neutral_cost": loss_action_neutral_cost,
        "valid_visibility_spans": valid_visibility.sum(),
        "valid_region_spans": valid_region.sum(),
        "hard_region_spans": hard_mask.sum(),
        "preserve_region_spans": preserve_mask.sum(),
        "override_utility_spans": utility_info["valid_mask"].sum(),
        "override_fix_spans": utility_info["fix_mask"].sum(),
        "override_damage_spans": utility_info["damage_mask"].sum(),
        "override_neutral_spans": utility_info["neutral_mask"].sum(),
        "action_supervision_spans": action_info["span_mask"].sum(),
        "action_valid_spans": action_info["valid_mask"].any(dim=-1).sum(),
        "action_fixable_spans": action_info["fixable_span_mask"].sum(),
        "action_preserve_spans": action_info["preserve_span_mask"].sum(),
        "action_fix_count": action_info["fix_mask"].sum(),
        "action_damage_count": action_info["damage_mask"].sum(),
        "action_neutral_count": action_info["neutral_mask"].sum(),
        "action_hard_damage_count": hard_damage_mask.sum(),
        "action_hard_neutral_count": hard_neutral_mask.sum(),
    }
