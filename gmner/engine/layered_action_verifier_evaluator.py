"""Evaluation and frozen-chain features for the M3.6A action verifier."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

from gmner.losses.layered_action_verifier_loss import (
    layered_action_supervision,
    layered_action_verifier_loss,
)
from gmner.models.layered_action_verifier import (
    ACTION_MODE_FULL,
    ACTION_MODE_NULL_RELEASE_ONLY,
    ACTION_MODE_TO_NULL_ONLY,
    ACTION_MODE_TO_REAL_ONLY,
    ACTION_TO_NULL,
    ACTION_TO_VISIBLE,
    decode_layered_actions,
    fine_topk_action_indices,
)

from .fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from .siglip2_region_reliability_evaluator import (
    frozen_current_visibility_context,
)
from .utils import f1_counts, match_record_predictions


def _direct_triple_correct(
    span: tuple[int, int],
    type_id: int,
    region_index: int,
    gold: list[dict],
) -> bool:
    return any(
        tuple(target["span"]) == span
        and int(target["type_id"]) == type_id
        and region_index
        in {int(value) for value in target.get("region_positive_indices") or []}
        for target in gold
    )


def _action_outcome(before: bool, after: bool) -> int:
    if after and not before:
        return 1
    if before and not after:
        return -1
    return 0


def _record_has_candidate_collision(
    fine_outputs: dict[str, torch.Tensor],
    row: int,
    selected: list[int],
) -> bool:
    if len(selected) < 2:
        return False
    logits = fine_outputs["final_region_logits"][row, selected].float()
    mask = fine_outputs["candidate_mask"][row, selected].bool()
    count = min(2, logits.size(-1))
    top = logits.masked_fill(~mask, -1e4).topk(count, dim=-1).indices
    usage = Counter()
    for local_row, span_indices in enumerate(top.tolist()):
        for index in span_indices:
            if bool(mask[local_row, index].item()):
                usage[int(index)] += 1
    return any(value > 1 for value in usage.values())


def layered_action_risk_curve(
    actions: list[tuple[float, int]],
    *,
    baseline_correct: int,
    predicted: int,
    gold: int,
    include_curve: bool,
) -> dict[str, float | list[dict[str, float]]]:
    ordered = sorted(actions, key=lambda item: item[0], reverse=True)
    fix = damage = neutral = 0
    best_net = best_count = 0
    best_threshold = 0.0
    positive_prefixes = 0
    curve: list[dict[str, float]] = []
    denominator = max(predicted + gold, 1)
    for index, (score, outcome) in enumerate(ordered, start=1):
        fix += int(outcome == 1)
        damage += int(outcome == -1)
        neutral += int(outcome == 0)
        net = fix - damage
        positive_prefixes += int(net > 0)
        if net > best_net:
            best_net = net
            best_count = index
            best_threshold = float(score)
        if include_curve:
            curve.append(
                {
                    "action_count": float(index),
                    "score_threshold": float(score),
                    "fix": float(fix),
                    "damage": float(damage),
                    "neutral": float(neutral),
                    "net_correction": float(net),
                    "estimated_gmner": (
                        2.0 * max(baseline_correct + net, 0) / denominator
                    ),
                }
            )
    return {
        "candidate_count": float(len(ordered)),
        "cumulative_max_net_correction": float(best_net),
        "cumulative_max_count": float(best_count),
        "cumulative_max_threshold": float(best_threshold),
        "positive_prefix_count": float(positive_prefixes),
        "positive_prefix_rate": positive_prefixes / max(len(ordered), 1),
        "risk_coverage_curve": curve,
    }


@torch.no_grad()
def frozen_layered_action_features(
    reliability_model: torch.nn.Module,
    evidence_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchy: torch.nn.Module,
    paired: dict,
    *,
    decode_options: dict,
) -> dict[str, object]:
    """Run every frozen component up to the learnable M3.6A policy."""

    formal = paired["formal"]
    expanded = paired["expanded"]
    siglip2 = paired.get("siglip2")
    region_options = {
        key: value
        for key, value in decode_options.items()
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
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
    fine_real_mask = (
        fine_outputs["candidate_mask"].bool()
        & expanded["region_mask"].bool()[:, None, :]
        & ~expanded["region_is_null"].bool()[:, None, :]
    )
    fine_top4_indices, fine_top4_valid = fine_topk_action_indices(
        fine_outputs["final_region_logits"],
        fine_real_mask,
        top_k=4,
    )
    fine_outputs["fine_top4_indices"] = fine_top4_indices
    fine_outputs["fine_top4_valid_mask"] = fine_top4_valid
    current_hierarchy, evidence_outputs, current_visible = (
        frozen_current_visibility_context(
            evidence_model,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            hierarchy_visible_mask=hierarchy_visible,
            base_is_null_mask=decoded["base_is_null"],
            decode_options=decode_options,
        )
    )
    reliability_outputs = reliability_model(
        fine_outputs,
        current_hierarchy,
        expanded,
        baseline_visible_mask=current_visible,
        base_is_null_mask=decoded["base_is_null"],
        siglip2_features=siglip2,
    )
    deployment_span_mask = torch.zeros_like(
        expanded["span_mask"], dtype=torch.bool
    )
    entity_threshold = float(decode_options.get("entity_threshold", 0.0))
    decode_strategy = str(decode_options.get("decode_strategy", "interval"))
    stage1_spans_only = bool(decode_options.get("stage1_spans_only", True))
    for row in range(deployment_span_mask.size(0)):
        _, selected = _selected_span_indices(
            current_hierarchy,
            formal,
            row,
            entity_threshold=entity_threshold,
            decode_strategy=decode_strategy,
            stage1_spans_only=stage1_spans_only,
        )
        if selected:
            deployment_span_mask[
                row, torch.as_tensor(selected, device=deployment_span_mask.device)
            ] = True
    return {
        "formal": formal,
        "expanded": expanded,
        "hierarchy_outputs": current_hierarchy,
        "decoded": decoded,
        "fine_outputs": fine_outputs,
        "evidence_outputs": evidence_outputs,
        "reliability_outputs": reliability_outputs,
        "current_visible": current_visible,
        "base_is_null": decoded["base_is_null"],
        "deployment_span_mask": deployment_span_mask,
    }


@torch.no_grad()
def evaluate_layered_action_verifier(
    model: torch.nn.Module,
    reliability_model: torch.nn.Module,
    evidence_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchy: torch.nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    decode_options: dict,
    loss_options: dict | None = None,
    execution_margin: float = 0.0,
    include_risk_curve: bool = True,
    identity_tolerance: float = 1e-12,
    minimum_keep_preservation_rate: float = 0.97,
    minimum_net_correction: int = 10,
) -> dict[str, float | list[dict[str, float]]]:
    for frozen in (reliability_model, evidence_model, fine_model, hierarchy):
        frozen.eval()
    model.eval()
    loss_options = dict(loss_options or {})
    counts = Counter()
    sums = Counter()
    correct = {branch: Counter() for branch in ("baseline", "final")}
    risk_actions: list[tuple[float, int]] = []
    risk_to_null: list[tuple[float, int]] = []
    risk_to_visible: list[tuple[float, int]] = []
    risk_null_release: list[tuple[float, int]] = []
    risk_region_switch: list[tuple[float, int]] = []

    entity_threshold = float(decode_options.get("entity_threshold", 0.0))
    decode_strategy = str(decode_options.get("decode_strategy", "interval"))
    stage1_spans_only = bool(decode_options.get("stage1_spans_only", True))

    for raw_batch in dataloader:
        paired = move_paired_record_batch(raw_batch, device)
        context = frozen_layered_action_features(
            reliability_model,
            evidence_model,
            fine_model,
            hierarchy,
            paired,
            decode_options=decode_options,
        )
        formal = context["formal"]
        expanded = context["expanded"]
        hierarchy_outputs = context["hierarchy_outputs"]
        fine_outputs = context["fine_outputs"]
        assert isinstance(formal, dict)
        assert isinstance(expanded, dict)
        assert isinstance(hierarchy_outputs, dict)
        assert isinstance(fine_outputs, dict)
        outputs = model(
            fine_outputs,
            hierarchy_outputs,
            context["evidence_outputs"],
            expanded,
            current_visible_mask=context["current_visible"],
            base_is_null_mask=context["base_is_null"],
            reliability_outputs=context["reliability_outputs"],
            deployment_span_mask=context["deployment_span_mask"],
        )
        decoded_actions = decode_layered_actions(
            outputs, execution_margin=execution_margin
        )
        losses = layered_action_verifier_loss(
            outputs,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            **loss_options,
        )
        supervision = layered_action_supervision(
            outputs,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            stage1_spans_only=bool(loss_options.get("stage1_spans_only", True)),
            require_correct_type=bool(loss_options.get("require_correct_type", True)),
        )
        batch_size = len(formal["metadata"])
        counts["records"] += batch_size
        for key in (
            "loss",
            "loss_layer1",
            "loss_layer2",
            "loss_keep_margin",
            "loss_correction_margin",
            "loss_preservation",
            "loss_residual",
        ):
            sums[key] += float(losses[key].item()) * batch_size
        for key in (
            "deployable_count",
            "eligible_count",
            "supervised_count",
            "keep_count",
            "to_null_count",
            "to_visible_count",
            "preservation_count",
            "excluded_count",
            "uncovered_visible_count",
            "uncertain_mapping_count",
        ):
            counts[key] += int(losses[key].item())

        selected_mask = torch.zeros_like(expanded["span_mask"], dtype=torch.bool)
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
            collision = _record_has_candidate_collision(fine_outputs, row, selected)
            slice_name = "collision" if collision else "noncollision"
            gold = list(metadata.get("gold_entities") or [])
            predictions = {"baseline": [], "final": []}
            for span_index in selected:
                span = tuple(spans[span_index])
                type_id = int(
                    hierarchy_outputs["fixed_type_ids"][row, span_index].item()
                )
                baseline_region = int(
                    outputs["current_region_indices"][row, span_index].item()
                )
                final_region = int(
                    decoded_actions["selected_region_indices"][row, span_index].item()
                )
                for branch, region in (
                    ("baseline", baseline_region),
                    ("final", final_region),
                ):
                    predictions[branch].append(
                        {
                            "span": list(span),
                            "type_id": type_id,
                            "region_index": region,
                        }
                    )
                baseline_ok = _direct_triple_correct(
                    span, type_id, baseline_region, gold
                )
                final_ok = _direct_triple_correct(span, type_id, final_region, gold)
                action_id = int(
                    decoded_actions["selected_action_ids"][row, span_index].item()
                )
                was_visible = bool(
                    outputs["current_visible_mask"][row, span_index].item()
                )
                executed = bool(
                    decoded_actions["executed_mask"][row, span_index].item()
                )
                if baseline_ok:
                    counts["keep_correct_total"] += 1
                    counts["keep_correct_preserved"] += int(final_ok)
                counts["base_wrong_corrected"] += int(not baseline_ok and final_ok)
                if executed:
                    outcome = _action_outcome(baseline_ok, final_ok)
                    counts["executed_count"] += 1
                    counts[f"executed_{action_id}"] += 1
                    counts["executed_fix"] += int(outcome == 1)
                    counts["executed_damage"] += int(outcome == -1)
                    counts["executed_neutral"] += int(outcome == 0)
                    counts[f"executed_{action_id}_fix"] += int(outcome == 1)
                    counts[f"executed_{action_id}_damage"] += int(outcome == -1)
                    counts[f"executed_{action_id}_neutral"] += int(outcome == 0)
                    if action_id == ACTION_TO_VISIBLE:
                        transition = (
                            "region_switch" if was_visible else "null_release"
                        )
                        counts[f"{transition}_executed"] += 1
                        counts[f"{transition}_fix"] += int(outcome == 1)
                        counts[f"{transition}_damage"] += int(outcome == -1)
                        counts[f"{transition}_neutral"] += int(outcome == 0)
                    counts[f"{slice_name}_fix"] += int(outcome == 1)
                    counts[f"{slice_name}_damage"] += int(outcome == -1)
                counts["prediction_changed_count"] += int(
                    baseline_region != final_region
                )

                if bool(
                    decoded_actions["best_non_keep_valid_mask"][row, span_index].item()
                ):
                    risk_action_id = int(
                        decoded_actions["best_non_keep_action_ids"][
                            row, span_index
                        ].item()
                    )
                    risk_region = int(
                        decoded_actions["best_non_keep_region_indices"][
                            row, span_index
                        ].item()
                    )
                    risk_score = float(
                        decoded_actions["best_non_keep_advantage"][
                            row, span_index
                        ].item()
                    )
                    risk_ok = _direct_triple_correct(span, type_id, risk_region, gold)
                    risk_outcome = _action_outcome(baseline_ok, risk_ok)
                    item = (risk_score, risk_outcome)
                    risk_actions.append(item)
                    if risk_action_id == ACTION_TO_NULL:
                        risk_to_null.append(item)
                    elif risk_action_id == ACTION_TO_VISIBLE:
                        risk_to_visible.append(item)
                        if was_visible:
                            risk_region_switch.append(item)
                        else:
                            risk_null_release.append(item)

            matches = {
                branch: match_record_predictions(values, gold)
                for branch, values in predictions.items()
            }
            counts["baseline_predicted"] += len(predictions["baseline"])
            counts["final_predicted"] += len(predictions["final"])
            counts["gold"] += len(gold)
            counts["identity_records"] += int(
                predictions["baseline"] == predictions["final"]
            )
            for branch in predictions:
                for metric in ("span", "mner", "eeg", "gmner"):
                    correct[branch][metric] += len(matches[branch][metric])
            counts[f"{slice_name}_baseline_gmner"] += len(matches["baseline"]["gmner"])
            counts[f"{slice_name}_final_gmner"] += len(matches["final"]["gmner"])
            counts[f"{slice_name}_records"] += 1
            counts["gmner_corrected"] += len(
                matches["final"]["gmner"] - matches["baseline"]["gmner"]
            )
            counts["gmner_damaged"] += len(
                matches["baseline"]["gmner"] - matches["final"]["gmner"]
            )

        layer1_prediction = outputs["layer1_logits"].argmax(dim=-1)
        supervised_selected = selected_mask & supervision["supervised_mask"]
        counts["selected_supervised"] += int(supervised_selected.sum().item())
        counts["layer1_correct"] += int(
            (supervised_selected & layer1_prediction.eq(supervision["layer1_labels"]))
            .sum()
            .item()
        )
        visible_selected = selected_mask & supervision["to_visible_mask"]
        layer2_prediction = outputs["layer2_scores"].argmax(dim=-1)
        layer2_correct = (
            supervision["layer2_positive_mask"]
            .gather(-1, layer2_prediction.unsqueeze(-1))
            .squeeze(-1)
        )
        counts["selected_to_visible"] += int(visible_selected.sum().item())
        counts["layer2_correct"] += int(
            (visible_selected & layer2_correct).sum().item()
        )
        full_correct = layer1_prediction.eq(supervision["layer1_labels"])
        full_correct = full_correct & (~visible_selected | layer2_correct)
        counts["full_action_correct"] += int(
            (supervised_selected & full_correct).sum().item()
        )

    records = max(int(counts["records"]), 1)
    metrics: dict[str, float | list[dict[str, float]]] = {
        key: sums[key] / records
        for key in (
            "loss",
            "loss_layer1",
            "loss_layer2",
            "loss_keep_margin",
            "loss_correction_margin",
            "loss_preservation",
            "loss_residual",
        )
    }
    metric_names = {
        "span": "span",
        "mner": "entity",
        "eeg": "eeg",
        "gmner": "triple",
    }
    for metric, name in metric_names.items():
        for branch in ("baseline", "final"):
            precision, recall, f1 = f1_counts(
                int(correct[branch][metric]),
                int(counts[f"{branch}_predicted"]),
                int(counts["gold"]),
            )
            prefix = "" if branch == "final" else "baseline_"
            metrics[f"{prefix}{name}_precision"] = precision
            metrics[f"{prefix}{name}_recall"] = recall
            metrics[f"{prefix}{name}_f1"] = f1
    metrics["gmner_score"] = metrics["triple_f1"]
    metrics["baseline_gmner_score"] = metrics["baseline_triple_f1"]
    for name in ("span", "entity", "eeg", "triple"):
        metrics[f"{name}_f1_delta"] = float(metrics[f"{name}_f1"]) - float(
            metrics[f"baseline_{name}_f1"]
        )
    for key in (
        "deployable_count",
        "eligible_count",
        "supervised_count",
        "keep_count",
        "to_null_count",
        "to_visible_count",
        "preservation_count",
        "excluded_count",
        "uncovered_visible_count",
        "uncertain_mapping_count",
        "executed_count",
        "prediction_changed_count",
        "gmner_corrected",
        "gmner_damaged",
    ):
        metrics[key] = float(counts[key])
    metrics["prediction_count_delta"] = float(
        counts["final_predicted"] - counts["baseline_predicted"]
    )
    metrics["record_prediction_identity_rate"] = counts["identity_records"] / records
    metrics["gmner_net_correction"] = float(
        counts["gmner_corrected"] - counts["gmner_damaged"]
    )
    metrics["keep_correct_preservation_rate"] = counts["keep_correct_preserved"] / max(
        counts["keep_correct_total"], 1
    )
    metrics["base_wrong_final_correct"] = float(counts["base_wrong_corrected"])
    metrics["layer1_accuracy"] = counts["layer1_correct"] / max(
        counts["selected_supervised"], 1
    )
    metrics["layer2_top4_accuracy"] = counts["layer2_correct"] / max(
        counts["selected_to_visible"], 1
    )
    metrics["full_action_accuracy"] = counts["full_action_correct"] / max(
        counts["selected_supervised"], 1
    )
    for action_id, name in (
        (ACTION_TO_NULL, "to_null"),
        (ACTION_TO_VISIBLE, "to_visible"),
    ):
        fix = counts[f"executed_{action_id}_fix"]
        damage = counts[f"executed_{action_id}_damage"]
        neutral = counts[f"executed_{action_id}_neutral"]
        metrics[f"{name}_executed"] = float(counts[f"executed_{action_id}"])
        metrics[f"{name}_corrected"] = float(fix)
        metrics[f"{name}_damaged"] = float(damage)
        metrics[f"{name}_neutral"] = float(neutral)
        metrics[f"{name}_net_correction"] = float(fix - damage)
        metrics[f"{name}_action_precision"] = fix / max(fix + damage, 1)
    metrics["to_real_executed"] = metrics["to_visible_executed"]
    metrics["to_real_corrected"] = metrics["to_visible_corrected"]
    metrics["to_real_damaged"] = metrics["to_visible_damaged"]
    metrics["to_real_net_correction"] = metrics["to_visible_net_correction"]
    for transition in ("null_release", "region_switch"):
        fix = counts[f"{transition}_fix"]
        damage = counts[f"{transition}_damage"]
        neutral = counts[f"{transition}_neutral"]
        metrics[f"{transition}_executed"] = float(
            counts[f"{transition}_executed"]
        )
        metrics[f"{transition}_corrected"] = float(fix)
        metrics[f"{transition}_damaged"] = float(damage)
        metrics[f"{transition}_neutral"] = float(neutral)
        metrics[f"{transition}_net_correction"] = float(fix - damage)
        metrics[f"{transition}_action_precision"] = fix / max(fix + damage, 1)
    for slice_name in ("collision", "noncollision"):
        metrics[f"{slice_name}_record_count"] = float(counts[f"{slice_name}_records"])
        metrics[f"{slice_name}_net_correction"] = float(
            counts[f"{slice_name}_final_gmner"] - counts[f"{slice_name}_baseline_gmner"]
        )
        metrics[f"{slice_name}_executed_net_correction"] = float(
            counts[f"{slice_name}_fix"] - counts[f"{slice_name}_damage"]
        )

    for prefix, values in (
        ("action", risk_actions),
        ("to_null", risk_to_null),
        ("to_visible", risk_to_visible),
        ("null_release", risk_null_release),
        ("region_switch", risk_region_switch),
    ):
        risk = layered_action_risk_curve(
            values,
            baseline_correct=int(correct["baseline"]["gmner"]),
            predicted=int(counts["baseline_predicted"]),
            gold=int(counts["gold"]),
            include_curve=include_risk_curve,
        )
        for key, value in risk.items():
            metrics[f"{prefix}_{key}"] = value

    tolerance = float(identity_tolerance)
    exact_metric_identity = all(
        abs(float(metrics[f"{name}_f1_delta"])) <= tolerance
        for name in ("span", "entity", "eeg", "triple")
    )
    metrics["epoch0_identity_pass"] = float(
        counts["executed_count"] == 0
        and counts["prediction_changed_count"] == 0
        and counts["baseline_predicted"] == counts["final_predicted"]
        and counts["identity_records"] == records
        and exact_metric_identity
    )
    metrics["go_keep_preservation"] = float(
        metrics["keep_correct_preservation_rate"]
        >= float(minimum_keep_preservation_rate)
    )
    metrics["go_total_net_correction"] = float(
        metrics["gmner_net_correction"] >= float(minimum_net_correction)
    )
    metrics["go_to_null_nonnegative"] = float(
        metrics["to_null_net_correction"] >= 0
    )
    metrics["go_to_null_positive"] = float(metrics["to_null_net_correction"] > 0)
    metrics["go_to_real_positive"] = float(metrics["to_real_net_correction"] > 0)
    action_mode = getattr(model.config, "action_mode", ACTION_MODE_FULL)
    metrics["action_mode_full"] = float(action_mode == ACTION_MODE_FULL)
    metrics["action_mode_to_real_only"] = float(
        action_mode == ACTION_MODE_TO_REAL_ONLY
    )
    metrics["action_mode_to_null_only"] = float(
        action_mode == ACTION_MODE_TO_NULL_ONLY
    )
    metrics["action_mode_null_release_only"] = float(
        action_mode == ACTION_MODE_NULL_RELEASE_ONLY
    )
    branch_pass = (
        metrics["go_to_real_positive"]
        if action_mode
        in (ACTION_MODE_TO_REAL_ONLY, ACTION_MODE_NULL_RELEASE_ONLY)
        else metrics["go_to_null_positive"]
        if action_mode == ACTION_MODE_TO_NULL_ONLY
        else metrics["go_to_null_nonnegative"]
        and metrics["go_to_real_positive"]
    )
    metrics["go_no_go"] = float(
        metrics["go_keep_preservation"]
        and metrics["go_total_net_correction"]
        and branch_pass
    )
    return metrics
