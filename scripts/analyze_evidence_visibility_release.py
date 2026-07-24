"""Diagnose M3.3A visibility release on dev without retraining or test access."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import (
    PairedRecordCandidateCollator,
    PairedRecordCandidateDataset,
    RecordCandidateDataset,
)
from gmner.engine.evidence_visibility_diagnostics import (
    binary_auc,
    distribution_summary,
    release_threshold_logits,
    stratified_linear_probe,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from gmner.engine.utils import f1_counts, match_record_predictions
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.models.evidence_visibility import (
    EVIDENCE_SCALAR_NAMES,
    decode_evidence_visibility,
)
from scripts.train_evidence_visibility import load_frozen_chain
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)


GROUP_A = "A_base_null_fine_correct"
GROUP_B = "B_base_null_fine_wrong"
GROUP_C = "C_base_visible_fine_correct"
GROUP_D = "D_base_visible_fine_wrong"
GROUP_NULL_KEEP = "E_gold_null_base_null"
GROUP_NULL_FIX = "F_gold_null_base_visible"
SOURCE_NAMES = {
    0: "base_only",
    1: "learned_only",
    2: "both",
    3: "padding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-epochs", type=int, default=200)
    return parser.parse_args()


def _safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    return value


def _group_feature_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0.0}
    feature_names = list(
        dict.fromkeys(
            [
                "base_visibility_probability",
                "final_visibility_probability",
                "actual_visibility_residual",
                "required_positive_residual",
                *EVIDENCE_SCALAR_NAMES,
            ]
        )
    )
    return {
        "count": float(len(rows)),
        "features": {
            name: distribution_summary(
                torch.tensor([float(row[name]) for row in rows])
            )
            for name in feature_names
        },
    }


def _probe_dataset(
    rows: list[dict],
    positives: set[str],
    negatives: set[str],
    *,
    require_covered_negatives: bool = False,
):
    selected = [
        row
        for row in rows
        if row["selected"]
        and row["group"] in positives | negatives
        and (
            not require_covered_negatives
            or row["group"] in positives
            or bool(row["candidate_covered"])
        )
    ]
    if not selected:
        return torch.empty(0, len(EVIDENCE_SCALAR_NAMES) + 4), torch.empty(
            0, dtype=torch.bool
        ), torch.empty(0)
    scalar = torch.tensor([row["scalar_features"] for row in selected])
    source = F.one_hot(
        torch.tensor([int(row["candidate_source_id"]) for row in selected]),
        num_classes=4,
    ).float()
    labels = torch.tensor(
        [row["group"] in positives for row in selected], dtype=torch.bool
    )
    current_score = torch.tensor(
        [float(row["actual_visibility_residual"]) for row in selected]
    )
    return torch.cat([scalar, source], dim=-1), labels, current_score


def _separability_report(
    rows: list[dict],
    *,
    positives: set[str],
    negatives: set[str],
    folds: int,
    epochs: int,
    require_covered_negatives: bool = False,
) -> dict:
    features, labels, current_score = _probe_dataset(
        rows,
        positives,
        negatives,
        require_covered_negatives=require_covered_negatives,
    )
    if labels.numel() == 0:
        return {"samples": 0.0}
    individual = {}
    for index, name in enumerate((*EVIDENCE_SCALAR_NAMES, *SOURCE_NAMES.values())):
        auc = binary_auc(features[:, index], labels)
        individual[name] = {
            "auc": auc,
            "orientation_free_auc": max(auc, 1.0 - auc)
            if math.isfinite(auc)
            else auc,
        }
    return {
        "positive_groups": sorted(positives),
        "negative_groups": sorted(negatives),
        "current_head_residual_auc": binary_auc(current_score, labels),
        "scalar_plus_source_linear_probe": stratified_linear_probe(
            features,
            labels,
            folds=folds,
            epochs=epochs,
        ),
        "individual_feature_auc": individual,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_evidence_visibility_config(args.config)
    if args.device:
        config.runtime.device = args.device
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    formal = RecordCandidateDataset(resolve(config.data.formal_dev_cache, root))
    expanded = RecordCandidateDataset(
        resolve(config.data.expanded_dev_cache, root)
    )
    paired = PairedRecordCandidateDataset(formal, expanded)
    (
        model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
    ) = load_frozen_chain(config, root, device)
    validate_fingerprints(
        paired,
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    checkpoint = torch.load(resolve(args.checkpoint, root), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    batch_size = int(args.batch_size or config.optim.batch_size)
    loader = DataLoader(
        paired,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=PairedRecordCandidateCollator(),
    )
    options = decode_options(hierarchy_config)
    entity_threshold = float(options["entity_threshold"])
    decode_strategy = str(options["decode_strategy"])
    stage1_spans_only = bool(options["stage1_spans_only"])
    region_options = {
        key: value
        for key, value in options.items()
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    visible_threshold = float(options["visible_from_null_threshold"])
    null_threshold = float(options["null_from_visible_threshold"])
    residual_limit = float(config.model.residual_scale)
    amp_enabled = bool(config.runtime.fp16 and device.type == "cuda")

    rows: list[dict] = []
    a_samples: list[dict] = []
    counts = Counter()
    group_counts = defaultdict(Counter)
    baseline_triple_correct = 0
    final_triple_correct = 0
    predicted_count = 0
    gold_count = 0

    model.eval()
    fine_model.eval()
    hierarchy.eval()
    with torch.no_grad():
        for raw_batch in loader:
            paired_batch = move_paired_record_batch(raw_batch, device)
            formal_batch = paired_batch["formal"]
            expanded_batch = paired_batch["expanded"]
            baseline = frozen_hierarchical_context(
                hierarchy,
                formal_batch,
                expanded_batch,
                decode_options=region_options,
            )
            hierarchy_outputs = baseline["outputs"]
            decoded = baseline["decoded"]
            baseline_visible = baseline["visible_mask"]
            assert isinstance(hierarchy_outputs, dict)
            assert isinstance(decoded, dict)
            assert isinstance(baseline_visible, torch.Tensor)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                fine_outputs = fine_model(expanded_batch)
                evidence = model(
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
                evidence["final_visibility_probability"],
                base_is_null=decoded["base_is_null"],
                baseline_visible=baseline_visible,
                has_real_candidate=evidence["fine_has_real_candidate"],
                has_null_region=has_null,
                span_mask=expanded_batch["span_mask"],
                visible_from_null_threshold=visible_threshold,
                null_from_visible_threshold=null_threshold,
                enabled=bool(options["enable_visibility_correction"]),
            )
            threshold_logits = release_threshold_logits(
                decoded["base_is_null"],
                visible_from_null_threshold=visible_threshold,
                null_from_visible_threshold=null_threshold,
            )
            required_residual = (
                threshold_logits - evidence["base_visibility_logits"].float()
            )
            fine_indices = evidence["fine_top1_region_index"].long()
            null_indices = torch.tensor(
                [
                    int(metadata.get("null_region_index", -1))
                    for metadata in expanded_batch["metadata"]
                ],
                device=device,
                dtype=torch.long,
            )[:, None].expand_as(fine_indices)
            baseline_indices = torch.where(
                baseline_visible, fine_indices, null_indices
            )
            final_indices = torch.where(
                final_visible, fine_indices, null_indices
            )

            for row_index, metadata in enumerate(expanded_batch["metadata"]):
                spans, selected = _selected_span_indices(
                    hierarchy_outputs,
                    formal_batch,
                    row_index,
                    entity_threshold=entity_threshold,
                    decode_strategy=decode_strategy,
                    stage1_spans_only=stage1_spans_only,
                )
                selected_set = set(selected)
                predictions = {"baseline": [], "final": []}
                for span_index in selected:
                    shared = {
                        "span": list(spans[span_index]),
                        "type_id": int(
                            hierarchy_outputs["fixed_type_ids"][
                                row_index, span_index
                            ].item()
                        ),
                    }
                    predictions["baseline"].append(
                        {
                            **shared,
                            "region_index": int(
                                baseline_indices[
                                    row_index, span_index
                                ].item()
                            ),
                        }
                    )
                    predictions["final"].append(
                        {
                            **shared,
                            "region_index": int(
                                final_indices[row_index, span_index].item()
                            ),
                        }
                    )
                gold = list(metadata.get("gold_entities") or [])
                matches = {
                    branch: match_record_predictions(values, gold)
                    for branch, values in predictions.items()
                }
                baseline_triple_correct += len(matches["baseline"]["gmner"])
                final_triple_correct += len(matches["final"]["gmner"])
                predicted_count += len(predictions["final"])
                gold_count += len(gold)

                candidate_by_span = {
                    span: index for index, span in enumerate(spans)
                }
                tokens = list(metadata.get("tokens") or [])
                null_index = int(metadata.get("null_region_index", -1))
                for gold_index, target in enumerate(gold):
                    span_index = candidate_by_span.get(tuple(target["span"]))
                    if span_index is None or int(
                        expanded_batch["span_source_ids"][
                            row_index, span_index
                        ].item()
                    ) != 0:
                        counts["gold_without_stage1_candidate"] += 1
                        continue
                    predicted_type = int(
                        hierarchy_outputs["fixed_type_ids"][
                            row_index, span_index
                        ].item()
                    )
                    type_correct = predicted_type == int(target["type_id"])
                    if not type_correct:
                        counts["type_wrong_excluded"] += 1
                        continue
                    selected_span = span_index in selected_set
                    target_visible = bool(target.get("visible", False))
                    old_visible = bool(
                        baseline_visible[row_index, span_index].item()
                    )
                    new_visible = bool(
                        final_visible[row_index, span_index].item()
                    )
                    positives = {
                        int(index)
                        for index in target.get("region_positive_indices") or []
                        if int(index) != null_index
                    }
                    fine_index = int(
                        fine_indices[row_index, span_index].item()
                    )
                    fine_correct = target_visible and fine_index in positives
                    candidate_covered = any(
                        bool(
                            fine_outputs["candidate_mask"][
                                row_index, span_index, index
                            ].item()
                        )
                        for index in positives
                    )
                    promoted = bool(
                        target_visible
                        and candidate_covered
                        and not any(index < 16 for index in positives)
                    )
                    origin_base_null = bool(
                        decoded["base_is_null"][row_index, span_index].item()
                    )
                    if target_visible:
                        if old_visible:
                            group = GROUP_C if fine_correct else GROUP_D
                        else:
                            group = GROUP_A if fine_correct else GROUP_B
                    else:
                        group = GROUP_NULL_FIX if old_visible else GROUP_NULL_KEEP
                    group_counts[group]["total"] += 1
                    group_counts[group]["selected"] += int(selected_span)
                    group_counts[group]["unselected"] += int(not selected_span)
                    if target_visible:
                        group_counts[group]["candidate_covered"] += int(
                            candidate_covered
                        )
                        group_counts[group]["candidate_uncovered"] += int(
                            not candidate_covered
                        )
                        group_counts[group]["promoted"] += int(promoted)
                    if not selected_span:
                        continue

                    scalar = evidence["evidence_scalar_features"][
                        row_index, span_index
                    ].float().cpu()
                    scalar_values = {
                        name: float(value)
                        for name, value in zip(
                            EVIDENCE_SCALAR_NAMES, scalar.tolist()
                        )
                    }
                    source_id = int(
                        evidence["fine_top1_source_id"][
                            row_index, span_index
                        ].item()
                    )
                    actual_residual = float(
                        evidence["bounded_visibility_delta_logits"][
                            row_index, span_index
                        ].item()
                    )
                    required = float(
                        required_residual[row_index, span_index].item()
                    )
                    row_payload = {
                        "group": group,
                        "selected": True,
                        "scalar_features": scalar.tolist(),
                        "candidate_source_id": source_id,
                        "candidate_covered": candidate_covered,
                        "promoted": promoted,
                        "base_visibility_probability": float(
                            evidence["base_visibility_probability"][
                                row_index, span_index
                            ].item()
                        ),
                        "final_visibility_probability": float(
                            evidence["final_visibility_probability"][
                                row_index, span_index
                            ].item()
                        ),
                        "actual_visibility_residual": actual_residual,
                        "required_positive_residual": required,
                        **scalar_values,
                    }
                    rows.append(row_payload)

                    if group == GROUP_A:
                        counts["a_released_visible"] += int(new_visible)
                        counts["a_remained_null"] += int(not new_visible)
                        counts["a_promoted"] += int(promoted)
                        counts["a_promoted_released"] += int(
                            promoted and new_visible
                        )
                        counts["a_origin_stage1_null"] += int(
                            origin_base_null
                        )
                        counts["a_origin_stage1_visible"] += int(
                            not origin_base_null
                        )
                        counts["a_reachable_by_current_bound"] += int(
                            required <= residual_limit
                        )
                        counts["a_unreachable_by_current_bound"] += int(
                            required > residual_limit
                        )
                        counts["a_promoted_reachable"] += int(
                            promoted and required <= residual_limit
                        )
                    elif group == GROUP_B:
                        counts["b_switched_visible"] += int(new_visible)
                    elif group == GROUP_NULL_KEEP:
                        counts["gold_null_keep_damaged"] += int(new_visible)
                    elif group == GROUP_NULL_FIX:
                        counts["gold_null_corrected"] += int(not new_visible)

                    if group == GROUP_A:
                        start, end = map(int, target["span"])
                        a_samples.append(
                            {
                                "record_id": str(metadata.get("record_id", "")),
                                "text": str(metadata.get("text", "")),
                                "span": [start, end],
                                "mention": str(
                                    target.get("text")
                                    or " ".join(tokens[start:end])
                                ),
                                "type_id": int(target["type_id"]),
                                "stage1_base_is_null": origin_base_null,
                                "release_threshold_probability": (
                                    visible_threshold
                                    if origin_base_null
                                    else null_threshold
                                ),
                                "base_visibility_logit": float(
                                    evidence["base_visibility_logits"][
                                        row_index, span_index
                                    ].item()
                                ),
                                "base_visibility_probability": row_payload[
                                    "base_visibility_probability"
                                ],
                                "final_visibility_probability": row_payload[
                                    "final_visibility_probability"
                                ],
                                "required_positive_residual": required,
                                "maximum_positive_residual": residual_limit,
                                "reachable_by_current_bound": bool(
                                    required <= residual_limit
                                ),
                                "actual_visibility_residual": actual_residual,
                                "released_visible": new_visible,
                                "fine_top1_region_index": fine_index,
                                "fine_top1_correct": True,
                                "final_region_correct": bool(new_visible),
                                "final_triple_correct": bool(
                                    gold_index in matches["final"]["gmner"]
                                ),
                                "fine_probability_margin": row_payload[
                                    "fine_probability_margin"
                                ],
                                "fine_normalized_entropy": row_payload[
                                    "fine_normalized_entropy"
                                ],
                                "candidate_source_id": source_id,
                                "candidate_source": SOURCE_NAMES.get(
                                    source_id, "unknown"
                                ),
                                "detector_confidence": row_payload[
                                    "detector_confidence"
                                ],
                                "type_object_compatibility": row_payload[
                                    "type_object_compatibility"
                                ],
                                "base_rank": row_payload["base_rank"],
                                "coarse_rank": row_payload["coarse_rank"],
                                "promoted": promoted,
                                "base_fine_agreement": bool(
                                    row_payload["base_fine_agreement"]
                                ),
                                "coarse_fine_agreement": bool(
                                    row_payload["coarse_fine_agreement"]
                                ),
                                "prior_fine_agreement": bool(
                                    row_payload["prior_fine_agreement"]
                                ),
                            }
                        )

    _, _, baseline_gmner = f1_counts(
        baseline_triple_correct, predicted_count, gold_count
    )
    _, _, final_gmner = f1_counts(
        final_triple_correct, predicted_count, gold_count
    )
    selected_group_counts = {
        group: int(values["selected"])
        for group, values in group_counts.items()
    }
    a_count = selected_group_counts.get(GROUP_A, 0)
    f_count = selected_group_counts.get(GROUP_NULL_FIX, 0)
    reachable_a = int(counts["a_reachable_by_current_bound"])
    _, _, residual_oracle_gmner = f1_counts(
        baseline_triple_correct + reachable_a,
        predicted_count,
        gold_count,
    )
    _, _, release_oracle_gmner = f1_counts(
        baseline_triple_correct + a_count,
        predicted_count,
        gold_count,
    )
    _, _, combined_oracle_gmner = f1_counts(
        baseline_triple_correct + a_count + f_count,
        predicted_count,
        gold_count,
    )
    selected_rows_by_group = {
        group: [
            row for row in rows if row["selected"] and row["group"] == group
        ]
        for group in group_counts
    }
    report = {
        "split": "dev",
        "checkpoint": str(resolve(args.checkpoint, root)),
        "baseline_and_current": {
            "predicted": predicted_count,
            "gold": gold_count,
            "baseline_triple_correct": baseline_triple_correct,
            "baseline_gmner": baseline_gmner,
            "current_triple_correct": final_triple_correct,
            "current_gmner": final_gmner,
            "current_net_correction": (
                final_triple_correct - baseline_triple_correct
            ),
        },
        "group_counts": {
            group: dict(values) for group, values in group_counts.items()
        },
        "release_behavior": {
            key: float(value) for key, value in counts.items()
        },
        "residual_reachability": {
            "residual_limit": residual_limit,
            "a_samples": a_count,
            "reachable": reachable_a,
            "unreachable": int(counts["a_unreachable_by_current_bound"]),
            "reachable_rate": reachable_a / max(a_count, 1),
            "current_residual_structure_oracle_net_correction": reachable_a,
            "current_residual_structure_oracle_gmner": residual_oracle_gmner,
        },
        "deployment_action_oracle": {
            "null_to_visible_fix_count": a_count,
            "to_null_fix_count": f_count,
            "null_to_visible_oracle_net_correction": a_count,
            "null_to_visible_oracle_gmner": release_oracle_gmner,
            "combined_visibility_oracle_net_correction": a_count + f_count,
            "combined_visibility_oracle_gmner": combined_oracle_gmner,
            "go_no_go": "high_potential" if a_count >= 10 else "stop",
        },
        "feature_separability": {
            "feature_names": [
                *EVIDENCE_SCALAR_NAMES,
                *SOURCE_NAMES.values(),
            ],
            "a_vs_b": _separability_report(
                rows,
                positives={GROUP_A},
                negatives={GROUP_B},
                folds=args.cv_folds,
                epochs=args.cv_epochs,
            ),
            "a_vs_b_candidate_covered": _separability_report(
                rows,
                positives={GROUP_A},
                negatives={GROUP_B},
                folds=args.cv_folds,
                epochs=args.cv_epochs,
                require_covered_negatives=True,
            ),
            "a_vs_false_rescue_pool": _separability_report(
                rows,
                positives={GROUP_A},
                negatives={GROUP_B, GROUP_NULL_KEEP},
                folds=args.cv_folds,
                epochs=args.cv_epochs,
            ),
        },
        "group_feature_distributions": {
            group: _group_feature_summary(group_rows)
            for group, group_rows in selected_rows_by_group.items()
        },
        "excluded": {
            "gold_without_stage1_candidate": int(
                counts["gold_without_stage1_candidate"]
            ),
            "type_wrong": int(counts["type_wrong_excluded"]),
        },
    }
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_safe_json(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sample_output = output.with_name(output.stem + ".a_samples.jsonl")
    with sample_output.open("w", encoding="utf-8") as stream:
        for sample in a_samples:
            stream.write(json.dumps(_safe_json(sample), ensure_ascii=False) + "\n")
    compact = {
        "baseline_gmner": baseline_gmner,
        "current_gmner": final_gmner,
        "groups": selected_group_counts,
        "a_released": int(counts["a_released_visible"]),
        "a_reachable": reachable_a,
        "release_oracle_net": a_count,
        "combined_oracle_net": a_count + f_count,
        "a_vs_b_auc": report["feature_separability"]["a_vs_b"].get(
            "scalar_plus_source_linear_probe", {}
        ).get("auc"),
        "output": str(output),
        "a_samples_output": str(sample_output),
    }
    print(json.dumps(_safe_json(compact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
