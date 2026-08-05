"""Safety losses for protected region-driven MNER refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gmner.constants import DEFAULT_LABEL2ID, IGNORE_INDEX


BOUNDARY_GROUPS = (
    (DEFAULT_LABEL2ID["O"],),
    tuple(DEFAULT_LABEL2ID[f"B-{entity_type}"] for entity_type in ("PER", "LOC", "ORG", "OTHER")),
    tuple(DEFAULT_LABEL2ID[f"I-{entity_type}"] for entity_type in ("PER", "LOC", "ORG", "OTHER")),
)


def boundary_log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Collapse typed BIO logits into O/B/I log probabilities."""

    normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
    grouped = torch.stack(
        [torch.logsumexp(logits[..., list(indices)], dim=-1) for indices in BOUNDARY_GROUPS],
        dim=-1,
    )
    return grouped - normalizer


def boundary_preservation_kl(
    *,
    base_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distill the frozen O/B/I boundary distribution into refined logits."""

    teacher_log_probs = boundary_log_probabilities(base_logits.detach())
    student_log_probs = boundary_log_probabilities(refined_logits)
    terms = F.kl_div(
        student_log_probs,
        teacher_log_probs.exp(),
        reduction="none",
    ).sum(dim=-1)
    valid = attention_mask.bool()
    if labels is not None:
        valid = valid & labels.ne(IGNORE_INDEX)
    return (terms * valid.to(terms.dtype)).sum() / valid.sum().clamp_min(1).to(terms.dtype)


def protected_gate_penalty(
    *,
    token_gate: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    null_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Suppress visual writeback on O tokens and formal NULL entities."""

    penalized = attention_mask.bool() & labels.eq(DEFAULT_LABEL2ID["O"])
    if target_mask is not None and null_target is not None:
        penalized = penalized | (target_mask.bool() & null_target.bool().unsqueeze(-1))
    return (token_gate * penalized.to(token_gate.dtype)).sum() / penalized.sum().clamp_min(1).to(
        token_gate.dtype
    )


def protected_region_residual_l2(
    *,
    region_delta: torch.Tensor,
    real_region_mask: torch.Tensor,
) -> torch.Tensor:
    per_region = region_delta.float().pow(2).mean(dim=-1)
    mask = real_region_mask.bool()
    return (per_region * mask.to(per_region.dtype)).sum() / mask.sum().clamp_min(1).to(per_region.dtype)
