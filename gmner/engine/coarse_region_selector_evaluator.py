"""Evaluation for recall-preserving coarse region selection."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

from gmner.losses.coarse_region_selector_loss import (
    coarse_region_selector_loss,
    coarse_selector_supervision,
)
from gmner.models.coarse_region_selector import (
    masked_topk_mask,
    recall_preserving_union_mask,
)

from .utils import move_record_batch


def build_coarse_policy_masks(
    outputs: dict[str, torch.Tensor],
    *,
    final_budget: int,
    base_keep_values: list[int],
) -> dict[str, torch.Tensor]:
    real_mask = outputs["real_region_mask"].bool()
    coarse = outputs["coarse_logits"].float()
    base = outputs["base_region_scores"].float()
    rank = torch.arange(real_mask.size(-1), device=real_mask.device)
    detector_prefix = real_mask & rank.view(1, 1, -1).lt(int(final_budget))
    policies = {
        f"detector_top{int(final_budget)}": detector_prefix,
        f"base_top{int(final_budget)}": masked_topk_mask(
            base, real_mask, final_budget
        ),
        f"learned_top{int(final_budget)}": masked_topk_mask(
            coarse, real_mask, final_budget
        ),
    }
    for base_keep in sorted({int(value) for value in base_keep_values}):
        learned_keep = max(int(final_budget) - base_keep, 0)
        name = f"union_base{base_keep}_learned{learned_keep}"
        policies[name] = recall_preserving_union_mask(
            base_scores=base,
            learned_scores=coarse,
            valid_mask=real_mask,
            total_budget=final_budget,
            base_keep=base_keep,
        )
    return policies


@torch.no_grad()
def evaluate_coarse_region_selector(
    model: torch.nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    final_budget: int = 16,
    base_keep_values: list[int] | None = None,
    loss_options: dict | None = None,
) -> dict[str, float]:
    model.eval()
    base_keep_values = list(base_keep_values or [8, 10])
    loss_options = dict(loss_options or {})
    counts = Counter()
    sums = Counter()
    policy_counts: dict[str, Counter] = {}

    for raw_batch in dataloader:
        batch = move_record_batch(raw_batch, device)
        outputs = model(batch)
        losses = coarse_region_selector_loss(outputs, batch, **loss_options)
        supervision = coarse_selector_supervision(
            outputs,
            batch,
            reference_budget=final_budget,
        )
        policies = build_coarse_policy_masks(
            outputs,
            final_budget=final_budget,
            base_keep_values=base_keep_values,
        )
        batch_size = len(batch["metadata"])
        counts["records"] += batch_size
        for key in (
            "loss",
            "loss_multi_positive",
            "loss_iou",
            "loss_correction_margin",
            "loss_preservation_margin",
        ):
            sums[key] += float(losses[key].item()) * batch_size

        eligible = supervision["eligible_mask"]
        recoverable = supervision["valid_mask"]
        positives = supervision["positive_mask"]
        type_correct = supervision["fixed_type_correct_mask"] & eligible
        promotion = supervision["promotion_mask"]
        coverage_preservation = supervision["coverage_preservation_mask"]
        base_wrong = supervision["base_wrong_mask"]
        base_correct = supervision["base_correct_mask"]
        detector_name = f"detector_top{int(final_budget)}"
        detector_hit = (policies[detector_name] & positives).any(dim=-1)
        counts["eligible"] += int(eligible.sum().item())
        counts["recoverable"] += int(recoverable.sum().item())
        counts["eligible_type_correct"] += int(type_correct.sum().item())
        counts["promotion"] += int(promotion.sum().item())
        counts["coverage_preservation"] += int(
            coverage_preservation.sum().item()
        )
        counts["base_wrong"] += int(base_wrong.sum().item())
        counts["base_correct"] += int(base_correct.sum().item())

        for name, selected in policies.items():
            stats = policy_counts.setdefault(name, Counter())
            hit = (selected & positives).any(dim=-1)
            stats["covered_eligible"] += int((hit & eligible).sum().item())
            stats["covered_recoverable"] += int((hit & recoverable).sum().item())
            stats["covered_type_correct"] += int((hit & type_correct).sum().item())
            stats["baseline_covered"] += int((detector_hit & eligible).sum().item())
            stats["baseline_preserved"] += int(
                (detector_hit & hit & eligible).sum().item()
            )
            stats["promoted"] += int(
                (~detector_hit & hit & recoverable).sum().item()
            )
            stats["dropped"] += int(
                (detector_hit & ~hit & eligible).sum().item()
            )
            stats["base_wrong_corrected"] += int(
                (base_wrong & hit).sum().item()
            )
            stats["base_correct_dropped"] += int(
                (base_correct & ~hit).sum().item()
            )
            stats["base_correct_preserved"] += int(
                (base_correct & hit).sum().item()
            )
            stats["selected"] += int(selected[eligible].sum().item())

        for metadata in batch["metadata"]:
            null_index = int(metadata.get("null_region_index", -1))
            for gold in metadata.get("gold_entities") or []:
                if not bool(gold.get("visible", False)):
                    continue
                counts["visible_gold"] += 1
                positives_raw = {
                    int(index)
                    for index in gold.get("region_positive_indices") or []
                    if int(index) != null_index
                }
                counts["raw_r36_covered"] += int(bool(positives_raw))
                counts["raw_r16_covered"] += int(
                    any(index < int(final_budget) for index in positives_raw)
                )

    records = max(int(counts["records"]), 1)
    metrics: dict[str, float] = {
        key: sums[key] / records
        for key in (
            "loss",
            "loss_multi_positive",
            "loss_iou",
            "loss_correction_margin",
            "loss_preservation_margin",
        )
    }
    visible = max(int(counts["visible_gold"]), 1)
    eligible_count = max(int(counts["eligible"]), 1)
    recoverable_count = max(int(counts["recoverable"]), 1)
    type_correct_count = max(int(counts["eligible_type_correct"]), 1)
    metrics.update(
        {
            "visible_gold_count": float(counts["visible_gold"]),
            "raw_detector_r16_recall": counts["raw_r16_covered"] / visible,
            "raw_detector_r36_recall": counts["raw_r36_covered"] / visible,
            "selector_eligible_count": float(counts["eligible"]),
            "selector_recoverable_count": float(counts["recoverable"]),
            "selector_type_correct_count": float(counts["eligible_type_correct"]),
            "promotion_needed_count": float(counts["promotion"]),
            "coverage_preservation_count": float(
                counts["coverage_preservation"]
            ),
            "base_wrong_count": float(counts["base_wrong"]),
            "base_correct_count": float(counts["base_correct"]),
        }
    )
    for name, stats in policy_counts.items():
        baseline_covered = max(int(stats["baseline_covered"]), 1)
        preservation_count = max(int(counts["base_correct"]), 1)
        metrics.update(
            {
                f"{name}_recall_eligible": (
                    stats["covered_eligible"] / eligible_count
                ),
                f"{name}_recall_recoverable": (
                    stats["covered_recoverable"] / recoverable_count
                ),
                f"{name}_recall_type_correct": (
                    stats["covered_type_correct"] / type_correct_count
                ),
                f"{name}_top16_preservation": (
                    stats["baseline_preserved"] / baseline_covered
                ),
                f"{name}_new_gold_promoted": float(stats["promoted"]),
                f"{name}_gold_dropped": float(stats["dropped"]),
                f"{name}_base_wrong_corrected": float(
                    stats["base_wrong_corrected"]
                ),
                f"{name}_base_correct_preservation": (
                    stats["base_correct_preserved"] / preservation_count
                ),
                f"{name}_average_candidate_count": (
                    stats["selected"] / eligible_count
                ),
            }
        )
    return metrics
