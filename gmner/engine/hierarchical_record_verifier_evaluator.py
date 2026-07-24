"""Evaluation and safe decoding for the hierarchical record verifier."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

from gmner.losses.hierarchical_record_candidate_loss import (
    OVERRIDE_DAMAGE,
    OVERRIDE_FIX,
    OVERRIDE_NEUTRAL,
    build_action_controller_targets,
    build_override_utility_targets,
    hierarchical_record_candidate_loss,
)
from gmner.models.structured_interval_decoder import (
    greedy_interval_decode,
    weighted_interval_decode,
)
from gmner.models.hierarchical_action_controller import (
    balanced_keep_regions,
    union_topk_action_mask,
)

from .utils import f1_counts as _f1
from .utils import match_record_predictions as _match_indices
from .utils import move_record_batch as _move


def decode_hierarchical_regions(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    enable_visibility_correction: bool = True,
    enable_region_override: bool = True,
    visible_from_null_threshold: float = 0.7,
    null_from_visible_threshold: float = 0.2,
    region_override_mode: str = "margin",
    region_override_logit_margin: float = 0.2,
    region_override_probability_margin: float = 0.05,
    override_damage_cost: float = 3.0,
    override_utility_threshold: float = 0.0,
    enable_action_controller: bool = False,
    action_top_k: int = 4,
    action_execution_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Apply conservative visibility switches and real-region residual overrides."""

    if enable_action_controller and enable_region_override:
        raise ValueError(
            "The unified action controller and legacy region override are mutually "
            "exclusive."
        )

    keep = balanced_keep_regions(
        outputs,
        batch,
        enable_visibility_correction=enable_visibility_correction,
        visible_from_null_threshold=visible_from_null_threshold,
        null_from_visible_threshold=null_from_visible_threshold,
    )
    base = outputs["base_region_indices"].long()
    proposed = outputs["best_real_region_index"].long()
    final_logits = outputs["final_region_logits"].float()
    real_mask = outputs["real_region_mask"].bool()
    null_mask = batch["region_is_null"].bool()
    safe_base = base.clamp(0, null_mask.size(-1) - 1)
    base_is_null = keep["base_is_null"]
    null_indices = keep["null_indices"]
    visibility_probability = outputs["visibility_probability"].float()
    has_real_region = keep["has_real_region"]

    selected = keep["region_indices"].clone()
    null_to_visible = keep["null_to_visible"]
    visible_to_null = keep["visible_to_null"]

    pre_override_selected = selected.clone()
    probabilities = torch.softmax(final_logits.masked_fill(~real_mask, -1e4), dim=-1)
    top_count = min(2, probabilities.size(-1))
    top_probabilities = probabilities.topk(top_count, dim=-1).values
    probability_margin = (
        top_probabilities[..., 0] - top_probabilities[..., 1]
        if top_count == 2
        else top_probabilities[..., 0]
    )
    safe_proposed = proposed.clamp(0, final_logits.size(-1) - 1)
    safe_base_real = safe_base.clamp(0, final_logits.size(-1) - 1)
    proposed_score = final_logits.gather(-1, safe_proposed.unsqueeze(-1)).squeeze(-1)
    base_score = final_logits.gather(-1, safe_base_real.unsqueeze(-1)).squeeze(-1)
    score_improvement = proposed_score - base_score
    region_override = torch.zeros_like(base_is_null)
    override_utility_probabilities = outputs.get("override_utility_probabilities")
    if override_utility_probabilities is None:
        override_utility_probabilities = torch.zeros(
            *base.shape,
            3,
            dtype=final_logits.dtype,
            device=final_logits.device,
        )
        override_utility_probabilities[..., OVERRIDE_NEUTRAL] = 1.0
    override_expected_utility = (
        override_utility_probabilities[..., OVERRIDE_FIX]
        - float(override_damage_cost)
        * override_utility_probabilities[..., OVERRIDE_DAMAGE]
    )
    override_candidate = (
        ~base_is_null
        & ~visible_to_null
        & has_real_region
        & proposed.ne(base)
    )
    if enable_region_override:
        mode = str(region_override_mode).lower()
        if mode == "margin":
            region_override = (
                override_candidate
                & score_improvement.ge(float(region_override_logit_margin))
                & probability_margin.ge(float(region_override_probability_margin))
            )
        elif mode == "utility":
            if "override_utility_logits" not in outputs:
                raise ValueError(
                    "region_override_mode=utility requires an enabled utility head."
                )
            region_override = override_candidate & override_expected_utility.gt(
                float(override_utility_threshold)
            )
        elif mode == "always":
            region_override = override_candidate
        else:
            raise ValueError(f"Unknown region_override_mode: {region_override_mode}")
        selected = torch.where(region_override, proposed, selected)

    action_candidate_mask = torch.zeros_like(real_mask)
    action_null_valid = torch.zeros_like(base_is_null)
    action_controller_executed = torch.zeros_like(base_is_null)
    action_controller_selected_is_null = torch.zeros_like(base_is_null)
    action_controller_score = torch.full_like(
        visibility_probability, -1e4, dtype=torch.float32
    )
    action_controller_proposed_region = pre_override_selected.clone()
    action_controller_region = pre_override_selected.clone()
    if enable_action_controller:
        if "action_real_scores" not in outputs or "action_null_scores" not in outputs:
            raise ValueError(
                "enable_action_controller requires model.enable_action_controller=true."
            )
        action_candidate_mask = union_topk_action_mask(
            fused_logits=final_logits,
            residual_logits=outputs["region_residual_logits"],
            base_logits=outputs["base_region_scores"],
            real_mask=real_mask,
            keep_indices=pre_override_selected,
            top_k=action_top_k,
        )
        action_null_valid = (
            keep["has_null_region"]
            & pre_override_selected.ne(null_indices)
        )
        real_scores = outputs["action_real_scores"].float().masked_fill(
            ~action_candidate_mask, -1e4
        )
        null_scores = outputs["action_null_scores"].float().masked_fill(
            ~action_null_valid, -1e4
        )
        best_real_score, best_real_index = real_scores.max(dim=-1)
        action_controller_selected_is_null = null_scores.gt(best_real_score)
        action_controller_score = torch.maximum(null_scores, best_real_score)
        has_action = action_null_valid | action_candidate_mask.any(dim=-1)
        action_controller_executed = (
            has_action
            & action_controller_score.gt(float(action_execution_margin))
        )
        action_controller_proposed_region = torch.where(
            action_controller_selected_is_null,
            null_indices,
            best_real_index,
        )
        action_controller_region = torch.where(
            action_controller_executed,
            action_controller_proposed_region,
            pre_override_selected,
        )
        selected = action_controller_region

    final_is_null = null_mask.gather(
        1, selected.clamp(0, null_mask.size(-1) - 1)
    )
    return {
        "region_indices": selected,
        "pre_override_region_indices": pre_override_selected,
        "base_is_null": base_is_null,
        "final_is_null": final_is_null,
        "null_to_visible": null_to_visible,
        "visible_to_null": visible_to_null,
        "region_override": region_override,
        "region_override_candidate": override_candidate,
        "score_improvement": score_improvement,
        "probability_margin": probability_margin,
        "override_utility_probabilities": override_utility_probabilities,
        "override_expected_utility": override_expected_utility,
        "action_candidate_mask": action_candidate_mask,
        "action_null_valid": action_null_valid,
        "action_controller_executed": action_controller_executed,
        "action_controller_selected_is_null": action_controller_selected_is_null,
        "action_controller_score": action_controller_score,
        "action_controller_proposed_region_indices": (
            action_controller_proposed_region
        ),
        "action_controller_region_indices": action_controller_region,
    }


def _binary_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


@torch.no_grad()
def evaluate_hierarchical_record_verifier(
    model: torch.nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    entity_threshold: float = 0.0,
    decode_strategy: str = "interval",
    stage1_spans_only: bool = True,
    enable_visibility_correction: bool = True,
    enable_region_override: bool = True,
    visible_from_null_threshold: float = 0.7,
    null_from_visible_threshold: float = 0.2,
    region_override_mode: str = "margin",
    region_override_logit_margin: float = 0.2,
    region_override_probability_margin: float = 0.05,
    override_damage_cost: float = 3.0,
    override_utility_threshold: float = 0.0,
    include_override_risk_curve: bool = False,
    enable_action_controller: bool = False,
    action_top_k: int = 4,
    action_execution_margin: float = 0.0,
    include_action_risk_curve: bool = False,
    loss_options: dict | None = None,
) -> dict[str, float | list[dict[str, float]]]:
    """Evaluate factorized predictions against the exact Stage-1 bypass."""

    model.eval()
    loss_options = dict(loss_options or {})
    counts = Counter()
    sums = Counter()
    final_correct = Counter()
    base_correct = Counter()
    pre_override_correct = Counter()
    override_actions: list[tuple[float, int]] = []
    controller_actions: list[tuple[float, int]] = []

    for raw_batch in dataloader:
        batch = _move(raw_batch, device)
        outputs = model(batch)
        losses = hierarchical_record_candidate_loss(outputs, batch, **loss_options)
        utility_info = build_override_utility_targets(
            outputs,
            batch,
            require_correct_type=bool(
                loss_options.get("override_utility_require_correct_type", True)
            ),
            stage1_only=bool(
                loss_options.get("override_utility_stage1_only", True)
            ),
        )
        action_info = None
        if "action_real_scores" in outputs and "action_null_scores" in outputs:
            action_info = build_action_controller_targets(
                outputs,
                batch,
                top_k=action_top_k,
                enable_visibility_correction=enable_visibility_correction,
                visible_from_null_threshold=visible_from_null_threshold,
                null_from_visible_threshold=null_from_visible_threshold,
                require_correct_type=bool(
                    loss_options.get("action_require_correct_type", True)
                ),
                stage1_only=bool(loss_options.get("action_stage1_only", True)),
            )
        decoded_regions = decode_hierarchical_regions(
            outputs,
            batch,
            enable_visibility_correction=enable_visibility_correction,
            enable_region_override=enable_region_override,
            visible_from_null_threshold=visible_from_null_threshold,
            null_from_visible_threshold=null_from_visible_threshold,
            region_override_mode=region_override_mode,
            region_override_logit_margin=region_override_logit_margin,
            region_override_probability_margin=region_override_probability_margin,
            override_damage_cost=override_damage_cost,
            override_utility_threshold=override_utility_threshold,
            enable_action_controller=enable_action_controller,
            action_top_k=action_top_k,
            action_execution_margin=action_execution_margin,
        )
        batch_size = len(batch["metadata"])
        counts["records"] += batch_size
        for key in (
            "loss",
            "loss_entity",
            "loss_visibility",
            "loss_region_multi",
            "loss_region_iou",
            "loss_region_hard",
            "loss_region_preserve",
            "loss_override_utility",
            "loss_action_listwise",
            "loss_action_expected_regret",
            "loss_action_fix_margin",
            "loss_action_damage_margin",
            "loss_action_neutral_cost",
        ):
            sums[key] += float(losses[key].item()) * batch_size

        if action_info is not None:
            action_scores = torch.cat(
                [
                    outputs["action_null_scores"].unsqueeze(-1),
                    outputs["action_real_scores"],
                ],
                dim=-1,
            )
            action_valid = action_info["valid_mask"]
            best_scores, best_slots = action_scores.float().masked_fill(
                ~action_valid, -1e4
            ).max(dim=-1)
            best_labels = action_info["targets"].gather(
                -1, best_slots.unsqueeze(-1)
            ).squeeze(-1)
            has_action = action_valid.any(dim=-1)
            chooses_action = has_action & best_scores.gt(0.0)
            counts["action_label_span_count"] += int(
                action_info["span_mask"].sum().item()
            )
            counts["action_fixable_span_count"] += int(
                action_info["fixable_span_mask"].sum().item()
            )
            counts["action_preserve_span_count"] += int(
                action_info["preserve_span_mask"].sum().item()
            )
            counts["action_policy_keep_count"] += int(
                (has_action & ~chooses_action).sum().item()
            )
            counts["action_policy_execute_count"] += int(
                chooses_action.sum().item()
            )
            counts["action_policy_fixable_top1"] += int(
                (
                    action_info["fixable_span_mask"]
                    & chooses_action
                    & best_labels.eq(OVERRIDE_FIX)
                ).sum().item()
            )
            for class_id, class_name in (
                (OVERRIDE_NEUTRAL, "neutral"),
                (OVERRIDE_FIX, "fix"),
                (OVERRIDE_DAMAGE, "damage"),
            ):
                class_mask = action_valid & action_info["targets"].eq(class_id)
                class_count = int(class_mask.sum().item())
                counts[f"action_{class_name}_label_count"] += class_count
                sums[f"action_{class_name}_score_sum"] += float(
                    action_scores.float().masked_fill(~class_mask, 0.0).sum().item()
                )
                counts[f"action_policy_top1_{class_name}"] += int(
                    (chooses_action & best_labels.eq(class_id)).sum().item()
                )
            for neutral_name in ("safe_neutral", "useless_neutral"):
                neutral_mask = action_info[f"{neutral_name}_mask"]
                neutral_count = int(neutral_mask.sum().item())
                counts[f"action_{neutral_name}_label_count"] += neutral_count
                sums[f"action_{neutral_name}_score_sum"] += float(
                    action_scores.float().masked_fill(~neutral_mask, 0.0).sum().item()
                )

        if "override_utility_logits" in outputs:
            valid_utility = utility_info["valid_mask"]
            predicted_utility = outputs["override_utility_logits"].argmax(dim=-1)
            counts["override_utility_label_total"] += int(
                valid_utility.sum().item()
            )
            counts["override_utility_label_correct"] += int(
                (
                    predicted_utility.eq(utility_info["targets"])
                    & valid_utility
                ).sum().item()
            )
            for class_id, class_name in (
                (OVERRIDE_NEUTRAL, "neutral"),
                (OVERRIDE_FIX, "fix"),
                (OVERRIDE_DAMAGE, "damage"),
            ):
                class_mask = valid_utility & utility_info["targets"].eq(class_id)
                counts[f"override_utility_{class_name}_labels"] += int(
                    class_mask.sum().item()
                )
                counts[f"override_utility_{class_name}_correct"] += int(
                    (predicted_utility.eq(class_id) & class_mask).sum().item()
                )

        for row, metadata in enumerate(batch["metadata"]):
            span_count = int(batch["span_mask"][row].sum().item())
            spans = [
                tuple(map(int, value))
                for value in batch["span_candidates"][row, :span_count].tolist()
            ]
            source_ids = batch["span_source_ids"][row, :span_count]
            decode_mask = torch.ones(span_count, dtype=torch.bool, device=device)
            if stage1_spans_only:
                decode_mask &= source_ids.eq(0)
            utilities = outputs["decode_utility"][row, :span_count].float().clone()
            utilities = utilities.masked_fill(~decode_mask, -1e4)
            utility_values = utilities.tolist()
            accepted = [
                index
                for index, value in enumerate(utility_values)
                if bool(decode_mask[index].item()) and value > entity_threshold
            ]
            if decode_strategy == "greedy":
                selected = greedy_interval_decode(
                    spans, utility_values, threshold=entity_threshold
                )
            elif decode_strategy == "interval":
                selected = weighted_interval_decode(
                    spans, utility_values, threshold=entity_threshold
                )
            else:
                raise ValueError(f"Unknown decode strategy: {decode_strategy}")

            gold_span_mask = batch["gold_span_mask"][row, :span_count].bool()
            eligible_gold = gold_span_mask & decode_mask
            entity_predictions = utilities.gt(entity_threshold) & decode_mask
            counts["candidate_gold"] += int(eligible_gold.sum().item())
            counts["entityness_pred"] += int(entity_predictions.sum().item())
            counts["entityness_correct"] += int(
                (entity_predictions & eligible_gold).sum().item()
            )
            counts["candidate_non_gold"] += int((decode_mask & ~gold_span_mask).sum().item())
            counts["non_gold_rejected"] += int(
                (decode_mask & ~gold_span_mask & ~entity_predictions).sum().item()
            )
            counts["gold_accepted"] += sum(
                int(gold_span_mask[index].item()) for index in accepted
            )
            counts["accepted_before"] += len(accepted)
            counts["accepted_after"] += len(selected)
            counts["overlap_removed"] += len(accepted) - len(selected)
            counts["gold_overlap_removed"] += sum(
                int(gold_span_mask[index].item())
                for index in accepted
                if index not in selected
            )
            counts["all_reject_records"] += int(len(selected) == 0)

            predictions: list[dict] = []
            pre_override_predictions: list[dict] = []
            for span_index in selected:
                shared = {
                    "span": list(spans[span_index]),
                    "type_id": int(outputs["fixed_type_ids"][row, span_index].item()),
                    "candidate_index": span_index,
                }
                predictions.append(
                    {
                        **shared,
                        "region_index": int(
                            decoded_regions["region_indices"][row, span_index].item()
                        ),
                    }
                )
                pre_override_predictions.append(
                    {
                        **shared,
                        "region_index": int(
                            decoded_regions["pre_override_region_indices"][
                                row, span_index
                            ].item()
                        ),
                    }
                )
            gold = list(metadata.get("gold_entities") or [])
            base_predictions = list(metadata.get("stage1_predictions") or [])
            final_matches = _match_indices(predictions, gold)
            base_matches = _match_indices(base_predictions, gold)
            pre_override_matches = _match_indices(pre_override_predictions, gold)
            counts["final_predicted"] += len(predictions)
            counts["base_predicted"] += len(base_predictions)
            counts["gold"] += len(gold)
            for name in ("span", "mner", "eeg", "gmner"):
                final_correct[name] += len(final_matches[name])
                base_correct[name] += len(base_matches[name])
                pre_override_correct[name] += len(pre_override_matches[name])
            counts["base_wrong_final_correct"] += len(
                final_matches["gmner"] - base_matches["gmner"]
            )
            counts["base_correct_final_wrong"] += len(
                base_matches["gmner"] - final_matches["gmner"]
            )
            counts["action_keep_correct_final_correct"] += len(
                pre_override_matches["gmner"] & final_matches["gmner"]
            )
            counts["action_keep_correct_final_wrong"] += len(
                pre_override_matches["gmner"] - final_matches["gmner"]
            )
            counts["action_keep_wrong_final_correct"] += len(
                final_matches["gmner"] - pre_override_matches["gmner"]
            )
            for name in ("span", "mner", "eeg"):
                counts[f"{name}_corrected"] += len(
                    final_matches[name] - base_matches[name]
                )
                counts[f"{name}_damaged"] += len(
                    base_matches[name] - final_matches[name]
                )

            null_index = int(metadata.get("null_region_index", -1))
            for gold_index, target in enumerate(gold):
                positives = list(target.get("region_positive_indices") or [])
                if positives == [null_index]:
                    counts["null_corrected"] += int(
                        gold_index in final_matches["eeg"]
                        and gold_index not in base_matches["eeg"]
                    )
                    counts["null_damaged"] += int(
                        gold_index in base_matches["eeg"]
                        and gold_index not in final_matches["eeg"]
                    )
                elif positives:
                    counts["visible_corrected"] += int(
                        gold_index in final_matches["eeg"]
                        and gold_index not in base_matches["eeg"]
                    )
                    counts["visible_damaged"] += int(
                        gold_index in base_matches["eeg"]
                        and gold_index not in final_matches["eeg"]
                    )

            candidate_by_span = {span: index for index, span in enumerate(spans)}
            final_by_span = {tuple(item["span"]): item for item in predictions}
            base_by_span = {tuple(item["span"]): item for item in base_predictions}
            for target in gold:
                span = tuple(target["span"])
                candidate_index = candidate_by_span.get(span)
                if candidate_index is None or source_ids[candidate_index].item() != 0:
                    continue
                visibility_target = batch["visibility_targets"][row, candidate_index]
                if visibility_target < 0:
                    continue
                gold_visible = bool(visibility_target.item())
                raw_visible = bool(
                    outputs["visibility_probability"][row, candidate_index].item() >= 0.5
                )
                base_visible = not bool(
                    decoded_regions["base_is_null"][row, candidate_index].item()
                )
                final_visible = not bool(
                    decoded_regions["final_is_null"][row, candidate_index].item()
                )
                for prefix, predicted_visible in (
                    ("visibility_raw", raw_visible),
                    ("visibility_final", final_visible),
                ):
                    counts[f"{prefix}_tp"] += int(predicted_visible and gold_visible)
                    counts[f"{prefix}_fp"] += int(predicted_visible and not gold_visible)
                    counts[f"{prefix}_fn"] += int(not predicted_visible and gold_visible)
                    counts[f"{prefix}_tn"] += int(not predicted_visible and not gold_visible)
                if not base_visible and final_visible:
                    counts["gold_null_to_visible_switch"] += 1
                    counts["gold_null_to_visible_corrected"] += int(gold_visible)
                    counts["gold_null_to_visible_damaged"] += int(not gold_visible)
                if base_visible and not final_visible:
                    counts["gold_visible_to_null_switch"] += 1
                    counts["gold_visible_to_null_corrected"] += int(not gold_visible)
                    counts["gold_visible_to_null_damaged"] += int(gold_visible)

                positives = set(target.get("region_positive_indices") or [])
                real_positives = positives - {null_index}
                if not gold_visible or not real_positives:
                    continue
                base_region_index = int(
                    outputs["base_region_indices"][row, candidate_index].item()
                )
                ranker_region_index = int(
                    outputs["best_real_region_index"][row, candidate_index].item()
                )
                base_rank_correct = base_region_index in real_positives
                ranker_correct = ranker_region_index in real_positives
                counts["ranker_visible_total"] += 1
                counts["ranker_base_visible_correct"] += int(base_rank_correct)
                counts["ranker_raw_visible_correct"] += int(ranker_correct)
                counts["ranker_raw_changed"] += int(
                    ranker_region_index != base_region_index
                )
                counts["ranker_raw_corrected"] += int(
                    ranker_correct and not base_rank_correct
                )
                counts["ranker_raw_damaged"] += int(
                    base_rank_correct and not ranker_correct
                )
                base_prediction = base_by_span.get(span)
                final_prediction = final_by_span.get(span)
                if base_prediction is None:
                    continue
                counts["gold_in_candidate_visible"] += 1
                base_region_correct = int(base_prediction["region_index"]) in real_positives
                final_region_correct = bool(
                    final_prediction
                    and int(final_prediction["region_index"]) in real_positives
                )
                if base_region_correct:
                    counts["base_correct_visible"] += 1
                    counts["base_correct_visible_preserved"] += int(final_region_correct)
                else:
                    counts["base_wrong_visible"] += 1
                    counts["base_wrong_visible_corrected"] += int(final_region_correct)

            for prediction in predictions:
                span_index = int(prediction["candidate_index"])
                if bool(decoded_regions["null_to_visible"][row, span_index].item()):
                    counts["null_to_visible_switches"] += 1
                if bool(decoded_regions["visible_to_null"][row, span_index].item()):
                    counts["visible_to_null_switches"] += 1
                is_override_candidate = bool(
                    decoded_regions["region_override_candidate"][
                        row, span_index
                    ].item()
                )
                utility_label = int(
                    utility_info["targets"][row, span_index].item()
                )
                utility_name = {
                    OVERRIDE_NEUTRAL: "neutral",
                    OVERRIDE_FIX: "fix",
                    OVERRIDE_DAMAGE: "damage",
                }[utility_label]
                if is_override_candidate:
                    counts["override_candidate_count"] += 1
                    counts[f"deployment_raw_{utility_name}"] += 1
                    if "override_utility_logits" in outputs:
                        action_score = float(
                            decoded_regions["override_expected_utility"][
                                row, span_index
                            ].item()
                        )
                    else:
                        action_score = float(
                            decoded_regions["score_improvement"][
                                row, span_index
                            ].item()
                        )
                    override_actions.append((action_score, utility_label))
                if enable_action_controller and action_info is not None:
                    valid_actions = action_info["valid_mask"][row, span_index]
                    counts["action_selected_candidate_count"] += int(
                        valid_actions.sum().item()
                    )
                    counts["action_selected_fixable_spans"] += int(
                        action_info["fixable_span_mask"][row, span_index].item()
                    )
                    selected_region = int(
                        decoded_regions[
                            "action_controller_proposed_region_indices"
                        ][row, span_index].item()
                    )
                    selected_is_null = bool(
                        decoded_regions["action_controller_selected_is_null"][
                            row, span_index
                        ].item()
                    )
                    action_slot = 0 if selected_is_null else selected_region + 1
                    action_valid = bool(
                        0 <= action_slot < valid_actions.numel()
                        and valid_actions[action_slot].item()
                    )
                    controller_label = (
                        int(action_info["targets"][row, span_index, action_slot].item())
                        if action_valid
                        else OVERRIDE_NEUTRAL
                    )
                    controller_score = float(
                        decoded_regions["action_controller_score"][
                            row, span_index
                        ].item()
                    )
                    if bool(
                        decoded_regions["action_null_valid"][row, span_index].item()
                        or decoded_regions["action_candidate_mask"][
                            row, span_index
                        ].any().item()
                    ):
                        controller_actions.append(
                            (controller_score, controller_label)
                        )
                    if bool(
                        decoded_regions["action_controller_executed"][
                            row, span_index
                        ].item()
                    ):
                        label_name = {
                            OVERRIDE_NEUTRAL: "neutral",
                            OVERRIDE_FIX: "fix",
                            OVERRIDE_DAMAGE: "damage",
                        }[controller_label]
                        counts["action_controller_executed_count"] += 1
                        counts[f"action_controller_executed_{label_name}"] += 1
                        counts["action_controller_executed_to_null"] += int(
                            selected_is_null
                        )
                        counts["action_controller_executed_to_real"] += int(
                            not selected_is_null
                        )
                if not bool(decoded_regions["region_override"][row, span_index].item()):
                    continue
                counts["region_override_count"] += 1
                counts[f"override_executed_{utility_name}"] += 1
                target = next(
                    (item for item in gold if tuple(item["span"]) == tuple(prediction["span"])),
                    None,
                )
                if target is None or not target.get("visible"):
                    continue
                positives = set(target.get("region_positive_indices") or [])
                base_prediction = base_by_span.get(tuple(prediction["span"]))
                if base_prediction is None:
                    continue
                base_ok = int(base_prediction["region_index"]) in positives
                final_ok = int(prediction["region_index"]) in positives
                counts["region_override_corrected"] += int(final_ok and not base_ok)
                counts["region_override_damaged"] += int(base_ok and not final_ok)

    records = max(counts["records"], 1)
    metrics: dict[str, float | list[dict[str, float]]] = {
        key: sums[key] / records
        for key in (
            "loss",
            "loss_entity",
            "loss_visibility",
            "loss_region_multi",
            "loss_region_iou",
            "loss_region_hard",
            "loss_region_preserve",
            "loss_override_utility",
            "loss_action_listwise",
            "loss_action_expected_regret",
            "loss_action_fix_margin",
            "loss_action_damage_margin",
            "loss_action_neutral_cost",
        )
    }
    names = {"span": "span", "mner": "entity", "eeg": "eeg", "gmner": "triple"}
    for match_name, output_name in names.items():
        precision, recall, score = _f1(
            final_correct[match_name], counts["final_predicted"], counts["gold"]
        )
        metrics[f"{output_name}_precision"] = precision
        metrics[f"{output_name}_recall"] = recall
        metrics[f"{output_name}_f1"] = score
        base_precision, base_recall, base_score = _f1(
            base_correct[match_name], counts["base_predicted"], counts["gold"]
        )
        metrics[f"stage1_bypass_{output_name}_precision"] = base_precision
        metrics[f"stage1_bypass_{output_name}_recall"] = base_recall
        metrics[f"stage1_bypass_{output_name}_f1"] = base_score
        metrics[f"{output_name}_f1_delta"] = score - base_score
    metrics["gmner_score"] = metrics["triple_f1"]
    metrics["final_prediction_count"] = float(counts["final_predicted"])
    metrics["stage1_prediction_count"] = float(counts["base_predicted"])
    metrics["prediction_count_delta"] = float(
        counts["final_predicted"] - counts["base_predicted"]
    )
    pre_override_precision, pre_override_recall, pre_override_f1 = _f1(
        pre_override_correct["gmner"], counts["final_predicted"], counts["gold"]
    )
    metrics["pre_override_triple_precision"] = pre_override_precision
    metrics["pre_override_triple_recall"] = pre_override_recall
    metrics["pre_override_triple_f1"] = pre_override_f1

    utility_total = counts["override_utility_label_total"]
    metrics["override_utility_label_accuracy"] = (
        counts["override_utility_label_correct"] / max(utility_total, 1)
    )
    metrics["override_utility_label_count"] = float(utility_total)
    for class_name in ("neutral", "fix", "damage"):
        label_count = counts[f"override_utility_{class_name}_labels"]
        metrics[f"override_utility_{class_name}_label_count"] = float(label_count)
        metrics[f"override_utility_{class_name}_recall"] = (
            counts[f"override_utility_{class_name}_correct"]
            / max(label_count, 1)
        )

    entity_p, entity_r, entity_f = _f1(
        counts["entityness_correct"], counts["entityness_pred"], counts["candidate_gold"]
    )
    metrics.update(
        {
            "entityness_precision": entity_p,
            "entityness_recall": entity_r,
            "entityness_f1": entity_f,
            "stage1_gold_span_accept_rate": counts["gold_accepted"] / max(counts["candidate_gold"], 1),
            "stage1_non_gold_reject_rate": counts["non_gold_rejected"] / max(counts["candidate_non_gold"], 1),
            "accepted_spans_per_record_before": counts["accepted_before"] / records,
            "accepted_spans_per_record_after": counts["accepted_after"] / records,
            "all_reject_ratio": counts["all_reject_records"] / records,
            "overlap_conflicts_removed": float(counts["overlap_removed"]),
            "gold_spans_removed_by_overlap": float(counts["gold_overlap_removed"]),
            "base_wrong_final_correct": float(counts["base_wrong_final_correct"]),
            "base_correct_final_wrong": float(counts["base_correct_final_wrong"]),
            "net_corrections": float(counts["base_wrong_final_correct"] - counts["base_correct_final_wrong"]),
        }
    )
    for name in ("span", "mner", "eeg"):
        metrics[f"{name}_corrected"] = float(counts[f"{name}_corrected"])
        metrics[f"{name}_damaged"] = float(counts[f"{name}_damaged"])
    for name in ("null", "visible"):
        metrics[f"{name}_corrected"] = float(counts[f"{name}_corrected"])
        metrics[f"{name}_damaged"] = float(counts[f"{name}_damaged"])

    for prefix in ("visibility_raw", "visibility_final"):
        tp = counts[f"{prefix}_tp"]
        fp = counts[f"{prefix}_fp"]
        fn = counts[f"{prefix}_fn"]
        tn = counts[f"{prefix}_tn"]
        precision, recall, f1 = _binary_metrics(tp, fp, fn)
        null_precision, null_recall, null_f1 = _binary_metrics(tn, fn, fp)
        metrics[f"{prefix}_visible_precision"] = precision
        metrics[f"{prefix}_visible_recall"] = recall
        metrics[f"{prefix}_visible_f1"] = f1
        metrics[f"{prefix}_null_precision"] = null_precision
        metrics[f"{prefix}_null_recall"] = null_recall
        metrics[f"{prefix}_null_f1"] = null_f1
        metrics[f"{prefix}_visible_to_null"] = float(fn)
        metrics[f"{prefix}_null_to_visible"] = float(fp)

    metrics["null_to_visible_switches"] = float(counts["null_to_visible_switches"])
    metrics["visible_to_null_switches"] = float(counts["visible_to_null_switches"])
    for direction in ("null_to_visible", "visible_to_null"):
        switch_count = counts[f"gold_{direction}_switch"]
        corrected = counts[f"gold_{direction}_corrected"]
        damaged = counts[f"gold_{direction}_damaged"]
        metrics[f"{direction}_gold_switch_count"] = float(switch_count)
        metrics[f"{direction}_gold_corrected"] = float(corrected)
        metrics[f"{direction}_gold_damaged"] = float(damaged)
        metrics[f"{direction}_gold_switch_precision"] = corrected / max(
            switch_count, 1
        )
    metrics["ranker_base_visible_accuracy"] = (
        counts["ranker_base_visible_correct"]
        / max(counts["ranker_visible_total"], 1)
    )
    metrics["ranker_raw_visible_accuracy"] = (
        counts["ranker_raw_visible_correct"]
        / max(counts["ranker_visible_total"], 1)
    )
    metrics["ranker_raw_changed"] = float(counts["ranker_raw_changed"])
    metrics["ranker_raw_corrected"] = float(counts["ranker_raw_corrected"])
    metrics["ranker_raw_damaged"] = float(counts["ranker_raw_damaged"])
    metrics["ranker_raw_net_corrections"] = float(
        counts["ranker_raw_corrected"] - counts["ranker_raw_damaged"]
    )
    metrics["ranker_raw_change_precision"] = (
        counts["ranker_raw_corrected"] / max(counts["ranker_raw_changed"], 1)
    )
    metrics["ranker_raw_decisive_precision"] = (
        counts["ranker_raw_corrected"]
        / max(counts["ranker_raw_corrected"] + counts["ranker_raw_damaged"], 1)
    )
    metrics["region_override_count"] = float(counts["region_override_count"])
    metrics["region_override_corrected"] = float(counts["region_override_corrected"])
    metrics["region_override_damaged"] = float(counts["region_override_damaged"])
    metrics["region_override_precision"] = (
        counts["region_override_corrected"] / max(counts["region_override_count"], 1)
    )
    metrics["region_override_decisive_precision"] = (
        counts["region_override_corrected"]
        / max(counts["region_override_corrected"] + counts["region_override_damaged"], 1)
    )
    raw_fix = counts["deployment_raw_fix"]
    raw_damage = counts["deployment_raw_damage"]
    raw_neutral = counts["deployment_raw_neutral"]
    executed_fix = counts["override_executed_fix"]
    executed_damage = counts["override_executed_damage"]
    executed_neutral = counts["override_executed_neutral"]
    metrics["override_candidate_count"] = float(counts["override_candidate_count"])
    metrics["deployment_raw_fix_count"] = float(raw_fix)
    metrics["deployment_raw_damage_count"] = float(raw_damage)
    metrics["deployment_raw_neutral_count"] = float(raw_neutral)
    metrics["deployment_raw_action_precision"] = raw_fix / max(
        raw_fix + raw_damage, 1
    )
    metrics["deployment_raw_useful_coverage"] = float(raw_fix + raw_damage)
    metrics["deployment_raw_net_correction"] = float(raw_fix - raw_damage)
    metrics["override_executed_count"] = float(counts["region_override_count"])
    metrics["override_fix_count"] = float(executed_fix)
    metrics["override_damage_count"] = float(executed_damage)
    metrics["override_neutral_count"] = float(executed_neutral)
    metrics["override_action_precision"] = executed_fix / max(
        executed_fix + executed_damage, 1
    )
    metrics["override_useful_coverage"] = float(executed_fix + executed_damage)
    metrics["override_net_correction"] = float(executed_fix - executed_damage)

    cumulative_fix = cumulative_damage = cumulative_neutral = 0
    maximum_net = 0
    maximum_net_count = 0
    risk_curve: list[dict[str, float]] = []
    denominator = max(counts["final_predicted"] + counts["gold"], 1)
    pre_override_tp = int(pre_override_correct["gmner"])
    for action_index, (_, label) in enumerate(
        sorted(override_actions, key=lambda item: item[0], reverse=True), start=1
    ):
        cumulative_fix += int(label == OVERRIDE_FIX)
        cumulative_damage += int(label == OVERRIDE_DAMAGE)
        cumulative_neutral += int(label == OVERRIDE_NEUTRAL)
        cumulative_net = cumulative_fix - cumulative_damage
        if cumulative_net > maximum_net:
            maximum_net = cumulative_net
            maximum_net_count = action_index
        if include_override_risk_curve:
            risk_curve.append(
                {
                    "override_count": float(action_index),
                    "corrected": float(cumulative_fix),
                    "damaged": float(cumulative_damage),
                    "neutral": float(cumulative_neutral),
                    "action_precision": cumulative_fix
                    / max(cumulative_fix + cumulative_damage, 1),
                    "cumulative_net_correction": float(cumulative_net),
                    "estimated_gmner": 2.0
                    * max(pre_override_tp + cumulative_net, 0)
                    / denominator,
                }
            )
    metrics["override_cumulative_max_net_correction"] = float(maximum_net)
    metrics["override_cumulative_max_count"] = float(maximum_net_count)
    if include_override_risk_curve:
        metrics["override_risk_coverage_curve"] = risk_curve
    action_executed_fix = counts["action_controller_executed_fix"]
    action_executed_damage = counts["action_controller_executed_damage"]
    action_executed_neutral = counts["action_controller_executed_neutral"]
    action_executed_total = (
        action_executed_fix + action_executed_damage + action_executed_neutral
    )
    metrics["action_label_span_count"] = float(counts["action_label_span_count"])
    metrics["action_fixable_span_count"] = float(
        counts["action_fixable_span_count"]
    )
    metrics["action_preserve_span_count"] = float(
        counts["action_preserve_span_count"]
    )
    metrics["action_policy_keep_count"] = float(counts["action_policy_keep_count"])
    metrics["action_policy_execute_count"] = float(
        counts["action_policy_execute_count"]
    )
    metrics["action_policy_fixable_top1_recall"] = (
        counts["action_policy_fixable_top1"]
        / max(counts["action_fixable_span_count"], 1)
    )
    for class_name in ("fix", "damage", "neutral"):
        label_count = counts[f"action_{class_name}_label_count"]
        metrics[f"action_{class_name}_label_count"] = float(label_count)
        metrics[f"action_{class_name}_score_mean"] = (
            sums[f"action_{class_name}_score_sum"] / max(label_count, 1)
        )
        metrics[f"action_policy_top1_{class_name}_count"] = float(
            counts[f"action_policy_top1_{class_name}"]
        )
    for neutral_name in ("safe_neutral", "useless_neutral"):
        label_count = counts[f"action_{neutral_name}_label_count"]
        metrics[f"action_{neutral_name}_label_count"] = float(label_count)
        metrics[f"action_{neutral_name}_score_mean"] = (
            sums[f"action_{neutral_name}_score_sum"] / max(label_count, 1)
        )
    metrics["action_selected_candidate_count"] = float(
        counts["action_selected_candidate_count"]
    )
    metrics["action_selected_fixable_spans"] = float(
        counts["action_selected_fixable_spans"]
    )
    metrics["action_controller_executed_count"] = float(
        counts["action_controller_executed_count"]
    )
    metrics["action_controller_execution_margin"] = float(
        action_execution_margin
    )
    metrics["action_controller_fix_count"] = float(action_executed_fix)
    metrics["action_controller_damage_count"] = float(action_executed_damage)
    metrics["action_controller_neutral_count"] = float(action_executed_neutral)
    metrics["action_controller_net_correction"] = float(
        action_executed_fix - action_executed_damage
    )
    metrics["action_controller_action_precision"] = action_executed_fix / max(
        action_executed_fix + action_executed_damage, 1
    )
    metrics["action_controller_fix_rate_over_executed"] = (
        action_executed_fix / max(action_executed_total, 1)
    )
    metrics["action_controller_neutral_rate_over_executed"] = (
        action_executed_neutral / max(action_executed_total, 1)
    )
    metrics["action_controller_to_null_count"] = float(
        counts["action_controller_executed_to_null"]
    )
    metrics["action_controller_to_real_count"] = float(
        counts["action_controller_executed_to_real"]
    )
    action_keep_correct = (
        counts["action_keep_correct_final_correct"]
        + counts["action_keep_correct_final_wrong"]
    )
    metrics["action_keep_correct_preservation_rate"] = (
        counts["action_keep_correct_final_correct"] / max(action_keep_correct, 1)
    )
    metrics["action_keep_correct_damaged_count"] = float(
        counts["action_keep_correct_final_wrong"]
    )
    metrics["action_keep_wrong_corrected_count"] = float(
        counts["action_keep_wrong_final_correct"]
    )

    action_cumulative_fix = action_cumulative_damage = action_cumulative_neutral = 0
    action_maximum_net = 0
    action_maximum_count = 0
    action_maximum_threshold = 0.0
    action_risk_curve: list[dict[str, float]] = []
    for action_index, (score, label) in enumerate(
        sorted(controller_actions, key=lambda item: item[0], reverse=True), start=1
    ):
        action_cumulative_fix += int(label == OVERRIDE_FIX)
        action_cumulative_damage += int(label == OVERRIDE_DAMAGE)
        action_cumulative_neutral += int(label == OVERRIDE_NEUTRAL)
        action_net = action_cumulative_fix - action_cumulative_damage
        if action_net > action_maximum_net:
            action_maximum_net = action_net
            action_maximum_count = action_index
            action_maximum_threshold = float(score)
        if include_action_risk_curve:
            action_risk_curve.append(
                {
                    "action_count": float(action_index),
                    "score_threshold": float(score),
                    "fix": float(action_cumulative_fix),
                    "damage": float(action_cumulative_damage),
                    "neutral": float(action_cumulative_neutral),
                    "net_correction": float(action_net),
                    "estimated_gmner": 2.0
                    * max(pre_override_tp + action_net, 0)
                    / denominator,
                }
            )
    metrics["action_controller_cumulative_max_net_correction"] = float(
        action_maximum_net
    )
    metrics["action_controller_cumulative_max_count"] = float(
        action_maximum_count
    )
    metrics["action_controller_cumulative_max_threshold"] = float(
        action_maximum_threshold
    )
    if include_action_risk_curve:
        metrics["action_controller_risk_coverage_curve"] = action_risk_curve
    metrics["base_correct_region_preservation_rate"] = (
        counts["base_correct_visible_preserved"] / max(counts["base_correct_visible"], 1)
    )
    metrics["base_wrong_region_correction_rate"] = (
        counts["base_wrong_visible_corrected"] / max(counts["base_wrong_visible"], 1)
    )
    metrics["gold_in_candidate_visible_count"] = float(counts["gold_in_candidate_visible"])
    return metrics
