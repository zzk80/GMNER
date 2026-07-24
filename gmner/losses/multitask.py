"""Multi-task losses used in GMNER training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gmner.constants import IGNORE_INDEX



def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid = labels != ignore_index
    if valid.sum().item() == 0:
        return logits.sum() * 0.0

    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    return F.cross_entropy(
        logits[valid],
        labels[valid],
        label_smoothing=label_smoothing,
    )


def weighted_masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Masked cross entropy with per-sample weights."""

    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("labels must be a 1D tensor with batch size entries")
    if sample_weight.ndim != 1 or sample_weight.size(0) != logits.size(0):
        raise ValueError("sample_weight must be a 1D tensor with batch size entries")

    valid = labels != ignore_index
    if valid.sum().item() == 0:
        return logits.sum() * 0.0

    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    losses = F.cross_entropy(
        logits[valid],
        labels[valid],
        label_smoothing=label_smoothing,
        reduction="none",
    )
    weights = sample_weight.to(device=logits.device, dtype=losses.dtype)[valid].clamp_min(0.0)
    weight_sum = weights.sum()
    if weight_sum.item() <= 0:
        return losses.mean()
    return (losses * weights).sum() / weight_sum


def hard_negative_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.2,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Penalize the hardest wrong region when it outranks the gold region.

    Cross entropy optimizes the whole region distribution. This loss adds a
    local ranking constraint: gold region score should be larger than the
    highest-scoring valid negative region by at least ``margin``.
    """

    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor of shape [batch, regions]")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("labels must be a 1D tensor with batch size entries")
    if valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must match logits shape")

    valid_labels = (
        (labels != ignore_index)
        & (labels >= 0)
        & (labels < logits.size(1))
    )
    if valid_labels.sum().item() == 0:
        return logits.sum() * 0.0

    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    row_ids = torch.arange(logits.size(0), device=logits.device)

    gold_valid = valid_labels & valid_mask[row_ids, labels.clamp_min(0).clamp_max(logits.size(1) - 1)]
    if gold_valid.sum().item() == 0:
        return logits.sum() * 0.0

    safe_labels = labels.clamp_min(0).clamp_max(logits.size(1) - 1)
    positive_scores = logits[row_ids, safe_labels]

    negative_mask = valid_mask.clone()
    negative_mask[row_ids, safe_labels] = False
    has_negative = negative_mask.any(dim=-1)
    active = gold_valid & has_negative
    if active.sum().item() == 0:
        return logits.sum() * 0.0

    negative_scores = logits.masked_fill(~negative_mask, -1e4)
    hardest_negative = negative_scores.max(dim=-1).values
    losses = F.relu(float(margin) + hardest_negative - positive_scores)
    return losses[active].mean()


def base_top1_hard_negative_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    base_logits: torch.Tensor,
    focus_mask: torch.Tensor | None = None,
    margin: float = 0.2,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Margin loss that only compares gold against Stage-1's wrong top-1 region."""

    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor of shape [batch, regions]")
    if base_logits.shape != logits.shape:
        raise ValueError("base_logits must match logits shape")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("labels must be a 1D tensor with batch size entries")
    if valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must match logits shape")

    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    base_logits = torch.nan_to_num(base_logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    safe_labels = labels.clamp_min(0).clamp_max(logits.size(1) - 1)
    row_ids = torch.arange(logits.size(0), device=logits.device)

    valid_labels = (
        (labels != ignore_index)
        & (labels >= 0)
        & (labels < logits.size(1))
        & valid_mask[row_ids, safe_labels]
    )
    if focus_mask is not None:
        valid_labels = valid_labels & focus_mask.to(device=logits.device, dtype=torch.bool)

    base_pred = base_logits.masked_fill(~valid_mask, -1e4).argmax(dim=-1)
    active = valid_labels & (base_pred != safe_labels) & valid_mask[row_ids, base_pred]
    if active.sum().item() == 0:
        return logits.sum() * 0.0

    gold_scores = logits[row_ids, safe_labels]
    base_wrong_scores = logits[row_ids, base_pred]
    losses = F.relu(float(margin) + base_wrong_scores - gold_scores)
    return losses[active].mean()


def multi_positive_region_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Region ranking loss with multiple acceptable positive boxes.

    ``positive_mask`` marks all region proposals that should be treated as
    correct for an entity. This is useful for VinVL proposals where several
    boxes can overlap the same XML object. Rows without any valid positive are
    ignored.
    """

    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor of shape [batch, regions]")
    if positive_mask.shape != logits.shape:
        raise ValueError("positive_mask must match logits shape")
    if valid_mask is not None and valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must match logits shape")
    if sample_weight is not None and (
        sample_weight.ndim != 1 or sample_weight.size(0) != logits.size(0)
    ):
        raise ValueError("sample_weight must be a 1D tensor with batch size entries")

    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    positives = positive_mask.to(device=logits.device, dtype=torch.bool)
    if valid_mask is None:
        valid = torch.ones_like(positives, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)

    positives = positives & valid
    active = positives.any(dim=-1)
    if active.sum().item() == 0:
        return logits.sum() * 0.0

    active_logits = logits[active].masked_fill(~valid[active], -1e4)
    active_positives = positives[active]
    log_denom = torch.logsumexp(active_logits, dim=-1)
    log_pos = torch.logsumexp(active_logits.masked_fill(~active_positives, -1e4), dim=-1)
    losses = -(log_pos - log_denom)
    if sample_weight is None:
        return losses.mean()
    weights = sample_weight.to(device=logits.device, dtype=losses.dtype)[active].clamp_min(0.0)
    if weights.sum().item() <= 0:
        return logits.sum() * 0.0
    return (losses * weights).sum() / weights.sum()


def joint_multi_positive_loss(
    joint_logits: torch.Tensor,
    target_type_ids: torch.Tensor,
    positive_region_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-positive NLL over flattened type-region candidates."""

    if joint_logits.ndim != 3:
        raise ValueError("joint_logits must have shape [batch, types, regions]")
    batch_size, num_types, num_regions = joint_logits.shape
    if target_type_ids.shape != (batch_size,):
        raise ValueError("target_type_ids must have shape [batch]")
    if positive_region_mask.shape != (batch_size, num_regions):
        raise ValueError("positive_region_mask must have shape [batch, regions]")
    if candidate_mask.shape != joint_logits.shape:
        raise ValueError("candidate_mask must match joint_logits")
    if sample_weight is not None and sample_weight.shape != (batch_size,):
        raise ValueError("sample_weight must have shape [batch]")

    scores = torch.nan_to_num(joint_logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    candidates = candidate_mask.to(device=scores.device, dtype=torch.bool)
    positive_regions = positive_region_mask.to(device=scores.device, dtype=torch.bool)
    target_types = target_type_ids.to(device=scores.device)
    safe_types = target_types.clamp(0, num_types - 1)
    type_match = torch.arange(num_types, device=scores.device).view(1, -1, 1)
    type_match = type_match == safe_types.view(-1, 1, 1)
    positives = candidates & type_match & positive_regions.unsqueeze(1)
    active = (
        (target_types >= 0)
        & (target_types < num_types)
        & candidates.flatten(1).any(dim=-1)
        & positives.flatten(1).any(dim=-1)
    )
    if not torch.any(active):
        return joint_logits.sum() * 0.0

    flat_scores = scores.flatten(1)
    flat_candidates = candidates.flatten(1)
    flat_positives = positives.flatten(1)
    log_denom = torch.logsumexp(
        flat_scores.masked_fill(~flat_candidates, -1e4),
        dim=-1,
    )
    log_positive = torch.logsumexp(
        flat_scores.masked_fill(~flat_positives, -1e4),
        dim=-1,
    )
    losses = -(log_positive - log_denom)
    if sample_weight is None:
        return losses[active].mean()
    weights = sample_weight.to(device=scores.device, dtype=losses.dtype)[active]
    weights = weights.clamp_min(0.0)
    if weights.sum().item() <= 0:
        return joint_logits.sum() * 0.0
    return (losses[active] * weights).sum() / weights.sum()


def joint_visibility_loss(
    visibility_logits: torch.Tensor,
    target_type_ids: torch.Tensor,
    positive_region_mask: torch.Tensor,
    null_index: int,
    visible_weight: float = 1.0,
    null_weight: float = 1.0,
) -> torch.Tensor:
    """Binary visible-versus-NULL loss for the joint verifier."""

    if visibility_logits.ndim == 2 and visibility_logits.size(-1) == 1:
        visibility_logits = visibility_logits.squeeze(-1)
    if visibility_logits.ndim != 1:
        raise ValueError("visibility_logits must have shape [batch]")
    if target_type_ids.shape != visibility_logits.shape:
        raise ValueError("target_type_ids must match visibility_logits")
    if positive_region_mask.ndim != 2 or (
        positive_region_mask.size(0) != visibility_logits.size(0)
    ):
        raise ValueError("positive_region_mask must have shape [batch, regions]")
    if not 0 <= int(null_index) < positive_region_mask.size(1):
        raise ValueError("null_index is outside positive_region_mask")

    positives = positive_region_mask.to(
        device=visibility_logits.device,
        dtype=torch.bool,
    )
    real_positives = positives.clone()
    real_positives[:, int(null_index)] = False
    visible_targets = real_positives.any(dim=-1)
    null_targets = positives[:, int(null_index)]
    active = (
        (target_type_ids.to(device=visibility_logits.device) >= 0)
        & (target_type_ids.to(device=visibility_logits.device) < 4)
        & (visible_targets | null_targets)
    )
    if not torch.any(active):
        return visibility_logits.sum() * 0.0

    logits = torch.nan_to_num(
        visibility_logits,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    targets = visible_targets.to(dtype=logits.dtype)
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(
        visible_targets,
        torch.full_like(losses, max(float(visible_weight), 0.0)),
        torch.full_like(losses, max(float(null_weight), 0.0)),
    )
    active_weights = weights[active]
    if active_weights.sum().item() <= 0:
        return visibility_logits.sum() * 0.0
    return (losses[active] * active_weights).sum() / active_weights.sum()


def joint_structured_margin_loss(
    joint_logits: torch.Tensor,
    target_type_ids: torch.Tensor,
    positive_region_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    base_type_logits: torch.Tensor,
    base_region_logits: torch.Tensor,
    margin: float = 0.2,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank a gold joint candidate above Stage1's strongest wrong candidate."""

    if joint_logits.ndim != 3:
        raise ValueError("joint_logits must have shape [batch, types, regions]")
    batch_size, num_types, num_regions = joint_logits.shape
    if target_type_ids.shape != (batch_size,):
        raise ValueError("target_type_ids must have shape [batch]")
    if positive_region_mask.shape != (batch_size, num_regions):
        raise ValueError("positive_region_mask must have shape [batch, regions]")
    if candidate_mask.shape != joint_logits.shape:
        raise ValueError("candidate_mask must match joint_logits")
    if base_type_logits.shape != (batch_size, num_types):
        raise ValueError("base_type_logits must have shape [batch, types]")
    if base_region_logits.shape != (batch_size, num_regions):
        raise ValueError("base_region_logits must have shape [batch, regions]")
    if sample_weight is not None and sample_weight.shape != (batch_size,):
        raise ValueError("sample_weight must have shape [batch]")

    scores = torch.nan_to_num(joint_logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    candidates = candidate_mask.to(device=scores.device, dtype=torch.bool)
    positive_regions = positive_region_mask.to(device=scores.device, dtype=torch.bool)
    target_types = target_type_ids.to(device=scores.device)
    safe_types = target_types.clamp(0, num_types - 1)
    type_match = torch.arange(num_types, device=scores.device).view(1, -1, 1)
    positives = (
        candidates
        & (type_match == safe_types.view(-1, 1, 1))
        & positive_regions.unsqueeze(1)
    )
    negatives = candidates & ~positives
    active = (
        (target_types >= 0)
        & (target_types < num_types)
        & positives.flatten(1).any(dim=-1)
        & negatives.flatten(1).any(dim=-1)
    )
    if not torch.any(active):
        return joint_logits.sum() * 0.0

    positive_scores = scores.masked_fill(~positives, -1e4).flatten(1).max(dim=-1).values
    base_scores = torch.nan_to_num(
        base_type_logits.to(dtype=scores.dtype),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    ).unsqueeze(-1) + torch.nan_to_num(
        base_region_logits.to(dtype=scores.dtype),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    ).unsqueeze(1)
    base_negative_index = (
        base_scores.masked_fill(~negatives, -1e4).flatten(1).argmax(dim=-1)
    )
    negative_scores = scores.flatten(1).gather(
        1,
        base_negative_index.unsqueeze(-1),
    ).squeeze(-1)
    losses = F.relu(max(float(margin), 0.0) + negative_scores - positive_scores)
    if sample_weight is None:
        return losses[active].mean()
    weights = sample_weight.to(device=scores.device, dtype=losses.dtype)[active]
    weights = weights.clamp_min(0.0)
    if weights.sum().item() <= 0:
        return joint_logits.sum() * 0.0
    return (losses[active] * weights).sum() / weights.sum()


def joint_teacher_kl_loss(
    joint_logits: torch.Tensor,
    base_joint_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve a confident Stage1 joint distribution on selected rows."""

    if joint_logits.shape != base_joint_logits.shape:
        raise ValueError("joint_logits and base_joint_logits must have matching shapes")
    if candidate_mask.shape != joint_logits.shape:
        raise ValueError("candidate_mask must match joint_logits")
    if active_mask.shape != (joint_logits.size(0),):
        raise ValueError("active_mask must have shape [batch]")

    candidates = candidate_mask.to(device=joint_logits.device, dtype=torch.bool).flatten(1)
    active = active_mask.to(device=joint_logits.device, dtype=torch.bool)
    active = active & candidates.any(dim=-1)
    if not torch.any(active):
        return joint_logits.sum() * 0.0

    student = torch.nan_to_num(
        joint_logits.float(),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    ).flatten(1)
    teacher = torch.nan_to_num(
        base_joint_logits.detach().float(),
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    ).flatten(1)
    student_log_probs = F.log_softmax(student.masked_fill(~candidates, -1e4), dim=-1)
    teacher_probs = F.softmax(teacher.masked_fill(~candidates, -1e4), dim=-1)
    losses = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)
    return losses[active].mean().to(dtype=joint_logits.dtype)


def iou_aware_region_ranking_loss(
    logits: torch.Tensor,
    iou_targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    margin: float = 0.2,
    min_iou_gap: float = 0.1,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank candidate regions according to their continuous IoU quality.

    Every valid pair whose IoU differs by at least ``min_iou_gap`` contributes
    a weighted soft margin loss. This retains information discarded by a
    single best-box label: a 0.70-IoU proposal should outrank a 0.45 proposal,
    which should in turn outrank an unrelated box. For an ungroundable entity,
    the dataset assigns IoU quality 1 to the NULL candidate, so the same loss
    also learns the visible/NULL ordering without a separate special case.
    """

    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor of shape [batch, regions]")
    if iou_targets.shape != logits.shape:
        raise ValueError("iou_targets must match logits shape")
    if valid_mask is not None and valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must match logits shape")
    if sample_weight is not None and (
        sample_weight.ndim != 1 or sample_weight.size(0) != logits.size(0)
    ):
        raise ValueError("sample_weight must be a 1D tensor with batch size entries")

    scores = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    quality = torch.nan_to_num(
        iou_targets.to(device=logits.device, dtype=logits.dtype),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    if valid_mask is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)

    quality_gap = quality.unsqueeze(2) - quality.unsqueeze(1)
    pair_mask = (
        valid.unsqueeze(2)
        & valid.unsqueeze(1)
        & (quality_gap >= max(float(min_iou_gap), 0.0))
        & (quality_gap > 0.0)
    )
    if not torch.any(pair_mask):
        return logits.sum() * 0.0

    score_gap = scores.unsqueeze(2) - scores.unsqueeze(1)
    pair_losses = F.softplus(float(margin) - score_gap)
    pair_weights = quality_gap.clamp_min(0.0)
    if sample_weight is not None:
        row_weights = sample_weight.to(
            device=logits.device,
            dtype=pair_weights.dtype,
        ).clamp_min(0.0)
        pair_weights = pair_weights * row_weights.view(-1, 1, 1)
    weighted_losses = pair_losses * pair_weights
    weight_sum = pair_weights[pair_mask].sum().clamp_min(1e-6)
    return weighted_losses[pair_mask].sum() / weight_sum



def alignment_objective(
    alignment_scores: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if alignment_scores.ndim != 2 or alignment_scores.size(0) != alignment_scores.size(1):
        raise ValueError("alignment_scores must be a square batch similarity matrix")
    if positive_mask is None:
        positive_mask = torch.eye(
            alignment_scores.size(0),
            dtype=torch.bool,
            device=alignment_scores.device,
        )
    else:
        positive_mask = positive_mask.to(device=alignment_scores.device, dtype=torch.bool)
        if positive_mask.shape != alignment_scores.shape:
            raise ValueError("positive_mask must match alignment_scores shape")

    def multi_positive_loss(scores: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(scores, dim=1)
        positive_log_probs = log_probs.masked_fill(~positives, float("-inf"))
        return -torch.logsumexp(positive_log_probs, dim=1).mean()

    text_to_image = multi_positive_loss(alignment_scores, positive_mask)
    image_to_text = multi_positive_loss(alignment_scores.transpose(0, 1), positive_mask.transpose(0, 1))
    return 0.5 * (text_to_image + image_to_text)
