"""Evaluation for the conditional same-type Fine-region resolver."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

from gmner.losses.same_type_region_resolver_loss import (
    same_type_region_resolver_loss,
)
from gmner.models.evidence_visibility import decode_evidence_visibility

from .fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from .utils import f1_counts, match_record_predictions


def selected_span_mask(
    hierarchy_outputs: dict[str, torch.Tensor],
    formal_batch: dict,
    *,
    entity_threshold: float,
    decode_strategy: str,
    stage1_spans_only: bool,
) -> torch.Tensor:
    selected = torch.zeros_like(formal_batch["span_mask"]).bool()
    for row in range(selected.size(0)):
        _, indices = _selected_span_indices(
            hierarchy_outputs,
            formal_batch,
            row,
            entity_threshold=entity_threshold,
            decode_strategy=decode_strategy,
            stage1_spans_only=stage1_spans_only,
        )
        if indices:
            selected[row, indices] = True
    return selected


@torch.no_grad()
def frozen_same_type_resolver_context(
    evidence_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchical_model: torch.nn.Module,
    formal_batch: dict,
    expanded_batch: dict,
    *,
    decode_options: dict,
) -> dict:
    evidence_model.eval()
    fine_model.eval()
    hierarchical_model.eval()
    entity_threshold = float(
        decode_options.get("entity_threshold", 0.0)
    )
    decode_strategy = str(
        decode_options.get("decode_strategy", "interval")
    )
    stage1_spans_only = bool(
        decode_options.get("stage1_spans_only", True)
    )
    region_decode_options = {
        key: value
        for key, value in decode_options.items()
        if key
        not in {
            "entity_threshold",
            "decode_strategy",
            "stage1_spans_only",
        }
    }
    hierarchy = frozen_hierarchical_context(
        hierarchical_model,
        formal_batch,
        expanded_batch,
        decode_options=region_decode_options,
    )
    hierarchy_outputs = hierarchy["outputs"]
    decoded = hierarchy["decoded"]
    baseline_visible = hierarchy["visible_mask"]
    assert isinstance(hierarchy_outputs, dict)
    assert isinstance(decoded, dict)
    assert isinstance(baseline_visible, torch.Tensor)
    fine_outputs = fine_model(expanded_batch)
    evidence_outputs = evidence_model(
        fine_outputs,
        hierarchy_outputs,
        expanded_batch,
        baseline_visible_mask=baseline_visible,
        base_is_null_mask=decoded["base_is_null"],
    )
    has_null = expanded_batch["region_is_null"].bool().any(
        dim=-1
    )[:, None].expand_as(baseline_visible)
    final_visible = decode_evidence_visibility(
        evidence_outputs["final_visibility_probability"],
        base_is_null=decoded["base_is_null"].bool(),
        baseline_visible=baseline_visible,
        has_real_candidate=evidence_outputs[
            "fine_has_real_candidate"
        ],
        has_null_region=has_null,
        span_mask=expanded_batch["span_mask"],
        visible_from_null_threshold=float(
            decode_options.get("visible_from_null_threshold", 0.8)
        ),
        null_from_visible_threshold=float(
            decode_options.get("null_from_visible_threshold", 0.2)
        ),
        enabled=bool(
            decode_options.get("enable_visibility_correction", True)
        ),
    )
    selected = selected_span_mask(
        hierarchy_outputs,
        formal_batch,
        entity_threshold=entity_threshold,
        decode_strategy=decode_strategy,
        stage1_spans_only=stage1_spans_only,
    )
    return {
        "hierarchy": hierarchy,
        "hierarchy_outputs": hierarchy_outputs,
        "fine_outputs": fine_outputs,
        "evidence_outputs": evidence_outputs,
        "final_visible_mask": final_visible,
        "selected_span_mask": selected,
    }


@torch.no_grad()
def evaluate_same_type_region_resolver(
    model: torch.nn.Module,
    evidence_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchical_model: torch.nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    decode_options: dict,
    loss_options: dict | None = None,
    enabled: bool = True,
) -> dict[str, float]:
    model.eval()
    evidence_model.eval()
    fine_model.eval()
    hierarchical_model.eval()
    loss_options = dict(loss_options or {})
    counts = Counter()
    sums = Counter()
    correct = {
        branch: Counter() for branch in ("baseline", "final")
    }

    for raw_batch in dataloader:
        paired = move_paired_record_batch(raw_batch, device)
        formal = paired["formal"]
        expanded = paired["expanded"]
        context = frozen_same_type_resolver_context(
            evidence_model,
            fine_model,
            hierarchical_model,
            formal,
            expanded,
            decode_options=decode_options,
        )
        fine_outputs = context["fine_outputs"]
        final_visible = context["final_visible_mask"]
        selected_mask = context["selected_span_mask"]
        hierarchy_outputs = context["hierarchy_outputs"]
        outputs = model(
            fine_outputs,
            expanded,
            selected_span_mask=selected_mask,
            final_visible_mask=final_visible,
            enabled=enabled,
        )
        old_indices = outputs["old_top1_region_index"].long()
        evidence_indices = context["evidence_outputs"][
            "fine_top1_region_index"
        ].long()
        deployed_real = (
            context["selected_span_mask"].bool()
            & context["final_visible_mask"].bool()
        )
        top1_mismatch = old_indices.ne(evidence_indices)
        if bool((top1_mismatch & deployed_real).any()):
            raise RuntimeError(
                "Resolver base Top1 differs from frozen Evidence Top1 "
                "on a selected final-visible span."
            )
        counts["inactive_top1_mismatch_count"] += int(
            (top1_mismatch & ~deployed_real).sum().item()
        )
        losses = same_type_region_resolver_loss(
            outputs, expanded, **loss_options
        )
        batch_size = len(formal["metadata"])
        counts["records"] += batch_size
        for key in (
            "loss",
            "loss_correction",
            "loss_preserve_kl",
            "loss_preserve_margin",
            "loss_residual",
        ):
            sums[key] += float(losses[key].item()) * batch_size
        for key in (
            "trigger_count",
            "trigger_candidate_count",
            "valid_count",
            "correction_count",
            "preservation_count",
            "candidate_missing_count",
        ):
            counts[key] += int(losses[key].item())

        changed = outputs["resolved_region_index"].ne(old_indices)
        trigger = outputs["trigger_mask"].bool()
        counts["override_count"] += int(changed.sum().item())
        counts["non_trigger_region_changed_count"] += int(
            (changed & ~trigger).sum().item()
        )
        counts["candidate_contract_violation_count"] += int(
            outputs["candidate_contract_violation"].sum().item()
        )
        counts["null_candidate_violation_count"] += int(
            (
                outputs["resolver_candidate_mask"].bool()
                & expanded["region_is_null"].bool()[:, None, :]
            )
            .sum()
            .item()
        )
        delta = outputs["bounded_delta_logits"].float().abs()
        trigger_candidates = outputs[
            "trigger_candidate_mask"
        ].bool()
        sums["delta_abs"] += float(
            delta.masked_select(trigger_candidates).sum().item()
        )
        counts["delta_count"] += int(trigger_candidates.sum().item())

        null_indices = torch.tensor(
            [
                int(metadata.get("null_region_index", -1))
                for metadata in expanded["metadata"]
            ],
            device=device,
            dtype=torch.long,
        )[:, None]
        null_indices = null_indices.expand_as(old_indices)
        baseline_regions = torch.where(
            final_visible, old_indices, null_indices
        )
        final_regions = torch.where(
            final_visible,
            outputs["resolved_region_index"].long(),
            null_indices,
        )
        if not torch.equal(
            baseline_regions.masked_select(~trigger),
            final_regions.masked_select(~trigger),
        ):
            raise RuntimeError("Non-trigger region output changed.")

        for row, metadata in enumerate(expanded["metadata"]):
            span_count = int(formal["span_mask"][row].sum().item())
            spans = [
                tuple(map(int, value))
                for value in formal["span_candidates"][
                    row, :span_count
                ].tolist()
            ]
            selected = selected_mask[row, :span_count].nonzero(
                as_tuple=False
            ).flatten().tolist()
            predictions = {"baseline": [], "final": []}
            for span_index in selected:
                shared = {
                    "span": list(spans[span_index]),
                    "type_id": int(
                        hierarchy_outputs["fixed_type_ids"][
                            row, span_index
                        ].item()
                    ),
                    "candidate_index": int(span_index),
                }
                predictions["baseline"].append(
                    {
                        **shared,
                        "region_index": int(
                            baseline_regions[row, span_index].item()
                        ),
                    }
                )
                predictions["final"].append(
                    {
                        **shared,
                        "region_index": int(
                            final_regions[row, span_index].item()
                        ),
                    }
                )
            gold = list(metadata.get("gold_entities") or [])
            matches = {
                branch: match_record_predictions(values, gold)
                for branch, values in predictions.items()
            }
            counts["predicted"] += len(predictions["final"])
            counts["gold"] += len(gold)
            for branch in predictions:
                for metric in ("span", "mner", "eeg", "gmner"):
                    correct[branch][metric] += len(
                        matches[branch][metric]
                    )
            corrected = (
                matches["final"]["gmner"]
                - matches["baseline"]["gmner"]
            )
            damaged = (
                matches["baseline"]["gmner"]
                - matches["final"]["gmner"]
            )
            counts["gmner_corrected"] += len(corrected)
            counts["gmner_damaged"] += len(damaged)
            counts["eeg_corrected"] += len(
                matches["final"]["eeg"] - matches["baseline"]["eeg"]
            )
            counts["eeg_damaged"] += len(
                matches["baseline"]["eeg"] - matches["final"]["eeg"]
            )

            candidate_by_span = {
                span: index for index, span in enumerate(spans)
            }
            for gold_index, target in enumerate(gold):
                span_index = candidate_by_span.get(
                    tuple(target["span"])
                )
                if span_index is None or not bool(
                    trigger[row, span_index].item()
                ):
                    continue
                base_correct = (
                    gold_index in matches["baseline"]["gmner"]
                )
                final_correct = gold_index in matches["final"]["gmner"]
                counts["trigger_gold_count"] += 1
                counts["base_correct_trigger_count"] += int(
                    base_correct
                )
                counts[
                    "base_correct_trigger_preserved"
                ] += int(base_correct and final_correct)
                counts["base_correct_trigger_damaged"] += int(
                    base_correct and not final_correct
                )
                counts["base_wrong_trigger_count"] += int(
                    not base_correct
                )
                counts["base_wrong_trigger_corrected"] += int(
                    not base_correct and final_correct
                )

    records = max(int(counts["records"]), 1)
    metrics = {
        key: sums[key] / records
        for key in (
            "loss",
            "loss_correction",
            "loss_preserve_kl",
            "loss_preserve_margin",
            "loss_residual",
        )
    }
    predicted = int(counts["predicted"])
    gold_count = int(counts["gold"])
    names = {
        "span": "span",
        "mner": "entity",
        "eeg": "eeg",
        "gmner": "triple",
    }
    for metric, output_name in names.items():
        for branch in ("baseline", "final"):
            precision, recall, score = f1_counts(
                int(correct[branch][metric]), predicted, gold_count
            )
            prefix = "" if branch == "final" else "baseline_"
            metrics[f"{prefix}{output_name}_precision"] = precision
            metrics[f"{prefix}{output_name}_recall"] = recall
            metrics[f"{prefix}{output_name}_f1"] = score
    metrics["gmner_score"] = metrics["triple_f1"]
    metrics["baseline_gmner_score"] = metrics[
        "baseline_triple_f1"
    ]
    metrics["gmner_delta"] = (
        metrics["gmner_score"] - metrics["baseline_gmner_score"]
    )
    metrics["span_f1_delta"] = (
        metrics["span_f1"] - metrics["baseline_span_f1"]
    )
    metrics["entity_f1_delta"] = (
        metrics["entity_f1"] - metrics["baseline_entity_f1"]
    )
    metrics["eeg_delta"] = (
        metrics["eeg_f1"] - metrics["baseline_eeg_f1"]
    )
    count_keys = (
        "trigger_count",
        "trigger_candidate_count",
        "valid_count",
        "correction_count",
        "preservation_count",
        "candidate_missing_count",
        "inactive_top1_mismatch_count",
        "override_count",
        "non_trigger_region_changed_count",
        "candidate_contract_violation_count",
        "null_candidate_violation_count",
        "gmner_corrected",
        "gmner_damaged",
        "eeg_corrected",
        "eeg_damaged",
        "trigger_gold_count",
        "base_correct_trigger_count",
        "base_correct_trigger_preserved",
        "base_correct_trigger_damaged",
        "base_wrong_trigger_count",
        "base_wrong_trigger_corrected",
    )
    for key in count_keys:
        metrics[key] = float(counts[key])
    metrics["gmner_net_correction"] = float(
        counts["gmner_corrected"] - counts["gmner_damaged"]
    )
    metrics["eeg_net_correction"] = float(
        counts["eeg_corrected"] - counts["eeg_damaged"]
    )
    metrics["base_correct_trigger_preservation_rate"] = (
        counts["base_correct_trigger_preserved"]
        / max(int(counts["base_correct_trigger_count"]), 1)
    )
    metrics["base_wrong_trigger_correction_rate"] = (
        counts["base_wrong_trigger_corrected"]
        / max(int(counts["base_wrong_trigger_count"]), 1)
    )
    metrics["delta_abs_mean"] = sums["delta_abs"] / max(
        int(counts["delta_count"]), 1
    )
    metrics["resolver_enabled"] = float(bool(enabled))
    metrics["visibility_changed_count"] = 0.0
    metrics["selected_span_changed_count"] = 0.0
    metrics["type_changed_count"] = 0.0
    metrics["c2_preregistered_condition_met"] = float(
        counts["gmner_corrected"] > counts["gmner_damaged"]
        and counts["base_correct_trigger_damaged"] > 5
    )
    return metrics
