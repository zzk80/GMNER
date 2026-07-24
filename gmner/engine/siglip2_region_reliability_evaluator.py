"""Dev-only diagnostics for M3.4A absolute region reliability."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import torch

from gmner.losses.siglip2_region_reliability_loss import (
    siglip2_region_reliability_loss,
    siglip2_region_reliability_supervision,
)
from gmner.models.evidence_visibility import decode_evidence_visibility

from .evidence_visibility_diagnostics import (
    best_binary_balanced_accuracy,
    binary_auc,
    binary_average_precision,
    binary_balanced_accuracy,
    binary_calibration_error,
)
from .fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from .utils import f1_counts, match_record_predictions


def _concat(values: list[torch.Tensor], *, dtype=None) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=dtype or torch.float32)
    return torch.cat([value.detach().cpu().reshape(-1) for value in values])


def _safe_metric(value: float, fallback: float = -1.0) -> float:
    return float(value) if math.isfinite(float(value)) else float(fallback)


@torch.no_grad()
def frozen_current_visibility_context(
    evidence_model: torch.nn.Module,
    fine_outputs: dict[str, torch.Tensor],
    hierarchy_outputs: dict[str, torch.Tensor],
    expanded_batch: dict,
    *,
    hierarchy_visible_mask: torch.Tensor,
    base_is_null_mask: torch.Tensor,
    decode_options: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Run the frozen M3.3 Evidence Visibility decision used by KEEP."""

    evidence_model.eval()
    evidence_outputs = evidence_model(
        fine_outputs,
        hierarchy_outputs,
        expanded_batch,
        baseline_visible_mask=hierarchy_visible_mask,
        base_is_null_mask=base_is_null_mask,
    )
    has_null = expanded_batch["region_is_null"].bool().any(dim=-1)[:, None]
    has_null = has_null.expand_as(hierarchy_visible_mask)
    current_visible = decode_evidence_visibility(
        evidence_outputs["final_visibility_probability"],
        base_is_null=base_is_null_mask,
        baseline_visible=hierarchy_visible_mask,
        has_real_candidate=evidence_outputs["fine_has_real_candidate"],
        has_null_region=has_null,
        span_mask=expanded_batch["span_mask"],
        visible_from_null_threshold=float(
            decode_options.get("visible_from_null_threshold", 0.8)
        ),
        null_from_visible_threshold=float(
            decode_options.get("null_from_visible_threshold", 0.2)
        ),
        enabled=bool(decode_options.get("enable_visibility_correction", True)),
    )
    current_outputs = dict(hierarchy_outputs)
    current_outputs["visibility_logits"] = evidence_outputs[
        "final_visibility_logits"
    ]
    current_outputs["visibility_probability"] = evidence_outputs[
        "final_visibility_probability"
    ]
    return current_outputs, evidence_outputs, current_visible


def reliability_risk_curve(
    scores: torch.Tensor,
    outcomes: torch.Tensor,
    null_actions: torch.Tensor,
    promoted_fixes: torch.Tensor,
    *,
    null_preservation_floor: float,
    baseline_correct: int,
    predicted: int,
    gold: int,
) -> dict[str, float | list[dict[str, float]]]:
    """Find the best score prefix under a hard gold-NULL preservation floor."""

    if scores.numel() == 0:
        return {
            "candidate_count": 0.0,
            "best_net_correction": 0.0,
            "best_action_count": 0.0,
            "best_threshold": 1.0,
            "best_fix_count": 0.0,
            "best_damage_count": 0.0,
            "best_neutral_count": 0.0,
            "best_promoted_fix_count": 0.0,
            "best_null_preservation_rate": 1.0,
            "estimated_gmner": f1_counts(
                baseline_correct, predicted, gold
            )[2],
            "curve": [],
        }
    order = torch.argsort(scores, descending=True)
    score = scores[order]
    outcome = outcomes[order]
    promoted = promoted_fixes[order]
    fix = outcome.eq(1).long().cumsum(0)
    damage = outcome.eq(-1).long().cumsum(0)
    neutral = outcome.eq(0).long().cumsum(0)
    promoted_fix = promoted.long().cumsum(0)
    total_null = max(int(null_actions.sum().item()), 1)
    preservation = 1.0 - damage.float() / total_null
    net = fix - damage
    allowed = preservation.ge(float(null_preservation_floor))
    best_index = -1
    best_net = 0
    for index in range(score.numel()):
        current = int(net[index].item())
        if bool(allowed[index].item()) and current > best_net:
            best_net = current
            best_index = index
    checkpoints = {
        min(max(int(score.numel() * fraction) - 1, 0), score.numel() - 1)
        for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
    }
    if best_index >= 0:
        checkpoints.add(best_index)
    curve = []
    for index in sorted(checkpoints):
        current_net = int(net[index].item())
        curve.append(
            {
                "action_count": float(index + 1),
                "threshold": float(score[index].item()),
                "fix": float(fix[index].item()),
                "damage": float(damage[index].item()),
                "neutral": float(neutral[index].item()),
                "net_correction": float(current_net),
                "null_preservation_rate": float(preservation[index].item()),
                "estimated_gmner": f1_counts(
                    baseline_correct + current_net, predicted, gold
                )[2],
            }
        )
    if best_index < 0:
        best = (0, 1.0, 0, 0, 0, 0, 1.0)
    else:
        best = (
            best_index + 1,
            float(score[best_index].item()),
            int(fix[best_index].item()),
            int(damage[best_index].item()),
            int(neutral[best_index].item()),
            int(promoted_fix[best_index].item()),
            float(preservation[best_index].item()),
        )
    count, threshold, fixes, damages, neutrals, promoted_count, preserved = best
    return {
        "candidate_count": float(score.numel()),
        "best_net_correction": float(best_net),
        "best_action_count": float(count),
        "best_threshold": float(threshold),
        "best_fix_count": float(fixes),
        "best_damage_count": float(damages),
        "best_neutral_count": float(neutrals),
        "best_promoted_fix_count": float(promoted_count),
        "best_null_preservation_rate": float(preserved),
        "estimated_gmner": f1_counts(
            baseline_correct + best_net, predicted, gold
        )[2],
        "curve": curve,
    }


@torch.no_grad()
def evaluate_siglip2_region_reliability(
    model: torch.nn.Module,
    evidence_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchy: torch.nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    decode_options: dict,
    loss_options: dict | None = None,
    reliability_threshold: float = 0.5,
    null_preservation_floor: float = 0.98,
    calibration_bins: int = 10,
    detector_reference_budget: int = 16,
    minimum_hard_ab_auc: float = 0.70,
    minimum_balanced_accuracy: float = 0.62,
    minimum_risk_net_correction: int = 15,
    minimum_promoted_fix_count: int = 1,
) -> dict[str, float | list[dict[str, float]]]:
    model.eval()
    evidence_model.eval()
    fine_model.eval()
    hierarchy.eval()
    loss_options = dict(loss_options or {})
    counts = Counter()
    sums = Counter()
    pair_scores: list[torch.Tensor] = []
    pair_labels: list[torch.Tensor] = []
    pair_targets: list[torch.Tensor] = []
    selected_pair_scores: list[torch.Tensor] = []
    selected_pair_labels: list[torch.Tensor] = []
    selected_pair_targets: list[torch.Tensor] = []
    hard_scores: list[torch.Tensor] = []
    hard_labels: list[torch.Tensor] = []
    risk_scores: list[torch.Tensor] = []
    risk_outcomes: list[torch.Tensor] = []
    risk_null: list[torch.Tensor] = []
    risk_promoted: list[torch.Tensor] = []

    entity_threshold = float(decode_options.get("entity_threshold", 0.0))
    decode_strategy = str(decode_options.get("decode_strategy", "interval"))
    stage1_spans_only = bool(decode_options.get("stage1_spans_only", True))
    region_options = {
        key: value
        for key, value in decode_options.items()
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    for raw_batch in dataloader:
        paired = move_paired_record_batch(raw_batch, device)
        formal = paired["formal"]
        expanded = paired["expanded"]
        siglip2 = paired.get("siglip2")
        baseline = frozen_hierarchical_context(
            hierarchy, formal, expanded, decode_options=region_options
        )
        hierarchy_outputs = baseline["outputs"]
        decoded = baseline["decoded"]
        hierarchy_visible = baseline["visible_mask"]
        assert isinstance(hierarchy_outputs, dict)
        assert isinstance(decoded, dict)
        assert isinstance(hierarchy_visible, torch.Tensor)
        fine_outputs = fine_model(expanded)
        hierarchy_outputs, _, baseline_visible = frozen_current_visibility_context(
            evidence_model,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            hierarchy_visible_mask=hierarchy_visible,
            base_is_null_mask=decoded["base_is_null"],
            decode_options=decode_options,
        )
        outputs = model(
            fine_outputs,
            hierarchy_outputs,
            expanded,
            baseline_visible_mask=baseline_visible,
            base_is_null_mask=decoded["base_is_null"],
            siglip2_features=siglip2,
        )
        losses = siglip2_region_reliability_loss(
            outputs,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            baseline_visible_mask=baseline_visible,
            **loss_options,
        )
        supervision = siglip2_region_reliability_supervision(
            outputs,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            baseline_visible_mask=baseline_visible,
            low_iou=float(loss_options.get("low_iou", 0.1)),
            positive_iou=float(loss_options.get("positive_iou", 0.5)),
            hard_negative_count=int(loss_options.get("hard_negative_count", 4)),
            other_entity_negative_count=int(
                loss_options.get("other_entity_negative_count", 2)
            ),
            compatibility_negative_count=int(
                loss_options.get("compatibility_negative_count", 2)
            ),
        )
        batch_size = len(formal["metadata"])
        counts["records"] += batch_size
        for key in (
            "loss",
            "loss_quality_focal",
            "loss_positive_max",
            "loss_null_suppress",
            "loss_rank",
            "loss_brier",
            "loss_hard_ab_bce",
            "loss_hard_ab_rank",
        ):
            sums[key] += float(losses[key].item()) * batch_size
        for key in (
            "eligible_count",
            "selected_pair_count",
            "positive_pair_count",
            "hard_negative_pair_count",
            "other_entity_negative_count",
            "compatibility_negative_count",
            "promoted_wrong_count",
            "group_a_count",
            "group_b_count",
            "group_b_hard_count",
            "group_b_uncovered_count",
            "group_null_count",
            "group_ordinary_count",
        ):
            counts[key] += int(losses[key].item())

        probability = outputs["reliability_probability"].float()
        all_pair = supervision["eligible_mask"].unsqueeze(-1) & outputs[
            "candidate_mask"
        ]
        selected_pair = supervision["selected_candidate_mask"]
        hard_target = supervision["positive_mask"]
        quality = supervision["quality_target"].float()
        pair_scores.append(probability[all_pair])
        pair_labels.append(hard_target[all_pair])
        pair_targets.append(quality[all_pair])
        selected_pair_scores.append(probability[selected_pair])
        selected_pair_labels.append(hard_target[selected_pair])
        selected_pair_targets.append(quality[selected_pair])

        selected_mask = torch.zeros_like(expanded["span_mask"], dtype=torch.bool)
        fine_index = outputs["fine_top1_region_index"].long()
        null_indices = torch.tensor(
            [
                int(metadata.get("null_region_index", -1))
                for metadata in expanded["metadata"]
            ],
            device=device,
            dtype=torch.long,
        )[:, None].expand_as(fine_index)
        baseline_indices = torch.where(baseline_visible, fine_index, null_indices)
        for row, metadata in enumerate(expanded["metadata"]):
            spans, selected = _selected_span_indices(
                hierarchy_outputs,
                formal,
                row,
                entity_threshold=entity_threshold,
                decode_strategy=decode_strategy,
                stage1_spans_only=stage1_spans_only,
            )
            if selected:
                selected_mask[row, torch.tensor(selected, device=device)] = True
            predictions = [
                {
                    "span": list(spans[span_index]),
                    "type_id": int(
                        hierarchy_outputs["fixed_type_ids"][row, span_index].item()
                    ),
                    "region_index": int(
                        baseline_indices[row, span_index].item()
                    ),
                }
                for span_index in selected
            ]
            gold_entities = list(metadata.get("gold_entities") or [])
            matches = match_record_predictions(predictions, gold_entities)
            counts["baseline_correct"] += len(matches["gmner"])
            counts["predicted"] += len(predictions)
            counts["gold"] += len(gold_entities)

        selected_gold = selected_mask & supervision["eligible_mask"]
        a_mask = selected_mask & supervision["group_a_mask"]
        hard_b_mask = (
            selected_mask
            & supervision["group_b_mask"]
            & supervision["candidate_covered_mask"]
        )
        top_score = outputs["fine_top1_reliability"].float()
        hard_scores.extend([top_score[a_mask], top_score[hard_b_mask]])
        hard_labels.extend(
            [
                torch.ones_like(top_score[a_mask], dtype=torch.bool),
                torch.zeros_like(top_score[hard_b_mask], dtype=torch.bool),
            ]
        )
        accepted = top_score.ge(float(reliability_threshold))
        visible_correct = (
            selected_gold
            & supervision["visible_mask"]
            & supervision["fine_correct_mask"]
        )
        visible_wrong = (
            selected_gold
            & supervision["visible_mask"]
            & ~supervision["fine_correct_mask"]
            & supervision["candidate_covered_mask"]
        )
        null_gold = selected_gold & ~supervision["visible_mask"]
        region_index = torch.arange(
            supervision["positive_mask"].size(-1), device=device
        ).view(1, 1, -1)
        detector_covered = (
            expanded["gold_region_positive_mask"].bool()
            & region_index.lt(int(detector_reference_budget))
        ).any(dim=-1)
        promoted_gold = supervision["candidate_covered_mask"] & ~detector_covered
        promoted_a = a_mask & promoted_gold
        for key, value in {
            "a_count": a_mask,
            "a_accepted": a_mask & accepted,
            "hard_b_count": hard_b_mask,
            "hard_b_rejected": hard_b_mask & ~accepted,
            "visible_correct_count": visible_correct,
            "visible_correct_accepted": visible_correct & accepted,
            "visible_wrong_count": visible_wrong,
            "visible_wrong_rejected": visible_wrong & ~accepted,
            "null_count": null_gold,
            "null_false_positive": null_gold & accepted,
            "promoted_a_count": promoted_a,
            "promoted_a_accepted": promoted_a & accepted,
        }.items():
            counts[key] += int(value.sum().item())
        positive_probability = probability.masked_fill(
            ~supervision["positive_mask"], -1.0
        ).max(dim=-1).values
        positive_span = selected_gold & supervision["candidate_covered_mask"]
        counts["positive_span_count"] += int(positive_span.sum().item())
        counts["positive_span_recalled"] += int(
            (
                positive_span
                & positive_probability.ge(float(reliability_threshold))
            ).sum().item()
        )
        if "siglip2_fine_top1_agreement" in outputs:
            valid = selected_mask & supervision["eligible_mask"]
            counts["siglip2_agreement_denominator"] += int(valid.sum().item())
            counts["siglip2_fine_agreement"] += int(
                (valid & outputs["siglip2_fine_top1_agreement"].bool()).sum().item()
            )
            counts["siglip2_four_way_agreement"] += int(
                (valid & outputs["siglip2_four_way_top1_agreement"].bool()).sum().item()
            )
        action_mask = (
            selected_mask
            & ~baseline_visible
            & outputs["candidate_mask"].any(dim=-1)
        )
        action_outcome = torch.zeros_like(top_score, dtype=torch.long)
        action_outcome = torch.where(
            a_mask, torch.ones_like(action_outcome), action_outcome
        )
        action_outcome = torch.where(
            null_gold & ~baseline_visible,
            -torch.ones_like(action_outcome),
            action_outcome,
        )
        risk_scores.append(top_score[action_mask])
        risk_outcomes.append(action_outcome[action_mask])
        risk_null.append((null_gold & ~baseline_visible)[action_mask])
        risk_promoted.append(promoted_a[action_mask])

    records = max(int(counts["records"]), 1)
    metrics: dict[str, float | list[dict[str, float]]] = {
        key: sums[key] / records
        for key in (
            "loss",
            "loss_quality_focal",
            "loss_positive_max",
            "loss_null_suppress",
            "loss_rank",
            "loss_brier",
            "loss_hard_ab_bce",
            "loss_hard_ab_rank",
        )
    }
    all_scores = _concat(pair_scores)
    all_labels = _concat(pair_labels, dtype=torch.bool).bool()
    all_targets = _concat(pair_targets)
    train_scores = _concat(selected_pair_scores)
    train_labels = _concat(selected_pair_labels, dtype=torch.bool).bool()
    train_targets = _concat(selected_pair_targets)
    hard_score = _concat(hard_scores)
    hard_label = _concat(hard_labels, dtype=torch.bool).bool()
    metrics.update(
        {
            "pair_auc": _safe_metric(binary_auc(all_scores, all_labels)),
            "pair_auprc": _safe_metric(
                binary_average_precision(all_scores, all_labels)
            ),
            "selected_pair_auc": _safe_metric(
                binary_auc(train_scores, train_labels)
            ),
            "selected_pair_auprc": _safe_metric(
                binary_average_precision(train_scores, train_labels)
            ),
            "pair_brier": (
                float((all_scores - all_targets).pow(2).mean().item())
                if all_scores.numel()
                else -1.0
            ),
            "selected_pair_brier": (
                float((train_scores - train_targets).pow(2).mean().item())
                if train_scores.numel()
                else -1.0
            ),
            "pair_ece": _safe_metric(
                binary_calibration_error(
                    all_scores, all_labels, bins=calibration_bins
                )
            ),
            "selected_pair_ece": _safe_metric(
                binary_calibration_error(
                    train_scores, train_labels, bins=calibration_bins
                )
            ),
        }
    )
    hard_auc = binary_auc(hard_score, hard_label)
    hard_ap = binary_average_precision(hard_score, hard_label)
    hard_ba = binary_balanced_accuracy(
        hard_score, hard_label, threshold=reliability_threshold
    )
    best_ba, best_threshold = best_binary_balanced_accuracy(
        hard_score, hard_label
    )
    metrics.update(
        {
            "hard_ab_auc": _safe_metric(hard_auc),
            "hard_ab_auprc": _safe_metric(hard_ap),
            "hard_ab_balanced_accuracy": _safe_metric(hard_ba),
            "hard_ab_best_balanced_accuracy": _safe_metric(best_ba),
            "hard_ab_best_threshold": _safe_metric(best_threshold, 0.5),
            "hard_ab_brier": (
                float((hard_score - hard_label.float()).pow(2).mean().item())
                if hard_score.numel()
                else -1.0
            ),
            "hard_ab_ece": _safe_metric(
                binary_calibration_error(
                    hard_score, hard_label, bins=calibration_bins
                )
            ),
        }
    )
    for key in (
        "eligible_count",
        "selected_pair_count",
        "positive_pair_count",
        "hard_negative_pair_count",
        "other_entity_negative_count",
        "compatibility_negative_count",
        "promoted_wrong_count",
        "group_a_count",
        "group_b_count",
        "group_b_hard_count",
        "group_b_uncovered_count",
        "group_null_count",
        "group_ordinary_count",
        "a_count",
        "a_accepted",
        "hard_b_count",
        "hard_b_rejected",
        "visible_correct_count",
        "visible_correct_accepted",
        "visible_wrong_count",
        "visible_wrong_rejected",
        "null_count",
        "null_false_positive",
        "promoted_a_count",
        "promoted_a_accepted",
        "positive_span_count",
        "positive_span_recalled",
    ):
        metrics[key] = float(counts[key])
    metrics["a_accept_rate"] = counts["a_accepted"] / max(counts["a_count"], 1)
    metrics["hard_b_reject_rate"] = counts["hard_b_rejected"] / max(
        counts["hard_b_count"], 1
    )
    metrics["fine_correct_top1_accept_rate"] = counts[
        "visible_correct_accepted"
    ] / max(counts["visible_correct_count"], 1)
    metrics["fine_wrong_top1_reject_rate"] = counts[
        "visible_wrong_rejected"
    ] / max(counts["visible_wrong_count"], 1)
    null_fpr = counts["null_false_positive"] / max(counts["null_count"], 1)
    metrics["null_high_reliability_false_positive_rate"] = null_fpr
    metrics["null_preservation_rate"] = 1.0 - null_fpr
    metrics["positive_region_recall"] = counts[
        "positive_span_recalled"
    ] / max(counts["positive_span_count"], 1)
    metrics["promoted_a_accept_rate"] = counts[
        "promoted_a_accepted"
    ] / max(counts["promoted_a_count"], 1)
    agreement_total = max(counts["siglip2_agreement_denominator"], 1)
    metrics["siglip2_fine_top1_agreement_rate"] = counts[
        "siglip2_fine_agreement"
    ] / agreement_total
    metrics["siglip2_four_way_top1_agreement_rate"] = counts[
        "siglip2_four_way_agreement"
    ] / agreement_total

    baseline_precision, baseline_recall, baseline_gmner = f1_counts(
        int(counts["baseline_correct"]),
        int(counts["predicted"]),
        int(counts["gold"]),
    )
    metrics["keep_triple_precision"] = baseline_precision
    metrics["keep_triple_recall"] = baseline_recall
    metrics["keep_gmner"] = baseline_gmner
    risk = reliability_risk_curve(
        _concat(risk_scores),
        _concat(risk_outcomes, dtype=torch.long).long(),
        _concat(risk_null, dtype=torch.bool).bool(),
        _concat(risk_promoted, dtype=torch.bool).bool(),
        null_preservation_floor=null_preservation_floor,
        baseline_correct=int(counts["baseline_correct"]),
        predicted=int(counts["predicted"]),
        gold=int(counts["gold"]),
    )
    for source, target in (
        ("candidate_count", "risk_candidate_count"),
        ("best_net_correction", "risk_best_net_correction"),
        ("best_action_count", "risk_best_action_count"),
        ("best_threshold", "risk_best_threshold"),
        ("best_fix_count", "risk_best_fix_count"),
        ("best_damage_count", "risk_best_damage_count"),
        ("best_neutral_count", "risk_best_neutral_count"),
        ("best_promoted_fix_count", "risk_best_promoted_fix_count"),
        ("best_null_preservation_rate", "risk_best_null_preservation_rate"),
        ("estimated_gmner", "risk_estimated_gmner"),
        ("curve", "risk_coverage_curve"),
    ):
        metrics[target] = risk[source]
    metrics["go_hard_ab_auc"] = float(
        metrics["hard_ab_auc"] >= float(minimum_hard_ab_auc)
    )
    metrics["go_hard_ab_balanced_accuracy"] = float(
        metrics["hard_ab_best_balanced_accuracy"]
        >= float(minimum_balanced_accuracy)
    )
    metrics["go_risk_net_correction"] = float(
        metrics["risk_best_net_correction"]
        >= float(minimum_risk_net_correction)
    )
    metrics["go_promoted_release"] = float(
        metrics["risk_best_promoted_fix_count"]
        >= float(minimum_promoted_fix_count)
    )
    metrics["go_no_go"] = float(
        metrics["go_hard_ab_auc"]
        and metrics["go_hard_ab_balanced_accuracy"]
        and metrics["go_risk_net_correction"]
        and metrics["go_promoted_release"]
    )
    return metrics
