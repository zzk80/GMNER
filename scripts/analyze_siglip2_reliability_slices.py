"""Analyze M3.4A hard A/B and risk-tail errors on the dev split only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.constants import ID2ENTITY_TYPE
from gmner.engine.evidence_visibility_diagnostics import (
    best_binary_balanced_accuracy,
    binary_auc,
    binary_average_precision,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from gmner.engine.siglip2_region_reliability_evaluator import (
    frozen_current_visibility_context,
)
from gmner.losses.siglip2_region_reliability_loss import (
    siglip2_region_reliability_supervision,
)
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
from scripts.build_siglip2_region_cache import _source_records
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)
from scripts.train_siglip2_region_reliability import (
    _base_paired,
    _paired_dataset,
    load_frozen_reliability_chain,
)


SLICE_DIMENSIONS = (
    "entity_type",
    "person_scene",
    "object_class_relation",
    "box_scale",
    "fine_siglip2_agreement",
    "four_way_agreement",
    "candidate_origin",
    "gold_candidate_origin",
    "context_overlap",
)
HUMAN_OBJECTS = frozenset(
    {
        "person",
        "people",
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "kid",
        "baby",
        "player",
        "players",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-file",
        default="GMNER-main/Twitter10000_v2.0/txt_fine/dev.txt",
    )
    parser.add_argument(
        "--vinvl-dir",
        default="GMNER-main/Twitter10000_v2.0/VinVL",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--context-expansion", type=float, default=1.5)
    parser.add_argument("--context-overlap-threshold", type=float, default=0.3)
    parser.add_argument("--small-area-threshold", type=float, default=0.05)
    parser.add_argument("--large-area-threshold", type=float, default=0.25)
    parser.add_argument("--examples-per-outcome", type=int, default=20)
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def binary_slice_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one hard A/B slice without inventing one-class AUROC."""

    scores = torch.tensor([float(row["score"]) for row in rows])
    labels = torch.tensor([bool(row["label"]) for row in rows])
    positives = int(labels.sum().item())
    negatives = int(labels.numel() - positives)
    if positives and negatives:
        best_accuracy, best_threshold = best_binary_balanced_accuracy(
            scores, labels
        )
        auc = binary_auc(scores, labels)
        average_precision = binary_average_precision(scores, labels)
    else:
        best_accuracy = best_threshold = auc = average_precision = float("nan")
    return {
        "count": float(labels.numel()),
        "group_a": float(positives),
        "group_b_hard": float(negatives),
        "hard_ab_auc": _finite(auc),
        "hard_ab_auprc": _finite(average_precision),
        "best_balanced_accuracy": _finite(best_accuracy),
        "best_threshold": _finite(best_threshold),
        "group_a_score_mean": (
            float(scores[labels].mean().item()) if positives else None
        ),
        "group_b_score_mean": (
            float(scores[~labels].mean().item()) if negatives else None
        ),
    }


def risk_slice_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    outcomes = Counter(int(row["outcome"]) for row in rows)
    fixes = outcomes[1]
    damages = outcomes[-1]
    neutrals = outcomes[0]
    return {
        "executed": float(len(rows)),
        "fix": float(fixes),
        "damage": float(damages),
        "neutral": float(neutrals),
        "net_correction": float(fixes - damages),
        "action_precision": fixes / max(fixes + damages, 1),
        "fix_rate_over_all_actions": fixes / max(len(rows), 1),
    }


def summarize_slices(
    rows: list[dict[str, Any]],
    *,
    binary: bool,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for dimension in SLICE_DIMENSIONS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(dimension, "unknown"))].append(row)
        output[dimension] = {
            key: (
                binary_slice_summary(values)
                if binary
                else risk_slice_summary(values)
            )
            for key, values in sorted(groups.items())
        }
    return output


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(right - left, 0.0) * np.maximum(
        bottom - top, 0.0
    )
    box_area = max(float(box[2] - box[0]), 0.0) * max(
        float(box[3] - box[1]), 0.0
    )
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    return intersection / np.maximum(box_area + areas - intersection, 1e-8)


def align_vinvl_object_labels(
    candidate_boxes: torch.Tensor,
    candidate_mask: torch.Tensor,
    npz_path: Path,
) -> list[str]:
    """Align candidate boxes to trusted VinVL labels by box IoU."""

    labels = ["unknown"] * int(candidate_boxes.size(0))
    if not npz_path.exists():
        return labels
    with np.load(npz_path, allow_pickle=True) as data:
        detector_boxes = np.asarray(data["bounding_boxes"], dtype=np.float32)
        detector_labels = [str(value).strip().lower() for value in data["objects"]]
    boxes = candidate_boxes.detach().float().cpu().numpy()
    for index in torch.nonzero(candidate_mask.bool(), as_tuple=False).flatten().tolist():
        overlaps = _box_iou(boxes[index], detector_boxes)
        if overlaps.size:
            match = int(overlaps.argmax())
            if float(overlaps[match]) >= 0.5:
                labels[index] = detector_labels[match] or "unknown"
    return labels


def _expanded_square(box: torch.Tensor, expansion: float) -> torch.Tensor:
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    side = max(x2 - x1, y2 - y1) * max(float(expansion), 1.0)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    return torch.tensor(
        [
            center_x - side / 2,
            center_y - side / 2,
            center_x + side / 2,
            center_y + side / 2,
        ]
    )


def maximum_context_overlap(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    selected_index: int,
    *,
    expansion: float,
) -> float:
    indices = torch.nonzero(valid.bool(), as_tuple=False).flatten()
    indices = indices[indices.ne(int(selected_index))]
    if indices.numel() == 0:
        return 0.0
    context = _expanded_square(boxes[selected_index].float().cpu(), expansion)
    other = boxes[indices].float().cpu().numpy()
    return float(_box_iou(context.numpy(), other).max(initial=0.0))


def _area_bucket(area: float, *, small: float, large: float) -> str:
    if area < small:
        return "small"
    if area < large:
        return "medium"
    return "large"


def _record_dimensions(
    *,
    row: int,
    span_index: int,
    fine_index: int,
    expanded: dict[str, Any],
    fine_outputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    supervision: dict[str, torch.Tensor],
    object_labels: list[str],
    detector_reference_budget: int,
    context_expansion: float,
    context_overlap_threshold: float,
    small_area_threshold: float,
    large_area_threshold: float,
) -> dict[str, str]:
    fixed_type = int(
        expanded["fixed_type_ids"][row, span_index].detach().cpu().item()
    )
    real_mask = (
        expanded["region_mask"][row].bool()
        & ~expanded["region_is_null"][row].bool()
    )
    human_count = sum(
        object_labels[index] in HUMAN_OBJECTS
        for index in torch.nonzero(real_mask, as_tuple=False).flatten().tolist()
    )
    if human_count == 0:
        person_scene = "no_person"
    elif human_count == 1:
        person_scene = "single_person"
    else:
        person_scene = "multi_person"

    positive = supervision["positive_mask"][row, span_index].bool()
    positive_indices = torch.nonzero(positive, as_tuple=False).flatten().tolist()
    fine_label = (
        object_labels[fine_index]
        if 0 <= fine_index < len(object_labels)
        else "unknown"
    )
    gold_labels = {
        object_labels[index]
        for index in positive_indices
        if 0 <= index < len(object_labels) and object_labels[index] != "unknown"
    }
    if fine_label == "unknown" or not gold_labels:
        relation = "unknown"
    elif fine_label in gold_labels:
        relation = "same"
    else:
        relation = "different"

    geometry = expanded["region_geometry"][row, fine_index].float()
    area = max(float((geometry[2] - geometry[0]).item()), 0.0) * max(
        float((geometry[3] - geometry[1]).item()), 0.0
    )
    overlap = maximum_context_overlap(
        expanded["region_geometry"][row],
        real_mask,
        fine_index,
        expansion=context_expansion,
    )
    promoted = bool(
        fine_outputs["promoted_candidate_mask"][row, span_index, fine_index]
        .detach()
        .cpu()
        .item()
    )
    gold_promoted = bool(positive_indices) and not any(
        index < int(detector_reference_budget) for index in positive_indices
    )
    if "siglip2_fine_top1_agreement" in outputs:
        fine_agreement = bool(
            outputs["siglip2_fine_top1_agreement"][row, span_index]
            .detach()
            .cpu()
            .item()
        )
        four_way = bool(
            outputs["siglip2_four_way_top1_agreement"][row, span_index]
            .detach()
            .cpu()
            .item()
        )
        fine_agreement_name = "agree" if fine_agreement else "disagree"
        four_way_name = "agree" if four_way else "disagree"
    else:
        fine_agreement_name = four_way_name = "not_available"
    return {
        "entity_type": ID2ENTITY_TYPE.get(fixed_type, "OTHER"),
        "person_scene": person_scene,
        "object_class_relation": relation,
        "box_scale": _area_bucket(
            area, small=small_area_threshold, large=large_area_threshold
        ),
        "fine_siglip2_agreement": fine_agreement_name,
        "four_way_agreement": four_way_name,
        "candidate_origin": "promoted" if promoted else "original",
        "gold_candidate_origin": "promoted" if gold_promoted else "original",
        "context_overlap": (
            "high" if overlap >= context_overlap_threshold else "low"
        ),
    }


def _example(
    *,
    metadata: dict[str, Any],
    span: tuple[int, int],
    score: float,
    outcome: int,
    fine_index: int,
    object_label: str,
    dimensions: dict[str, str],
) -> dict[str, Any]:
    tokens = list(metadata.get("tokens") or [])
    start, end = span
    return {
        "record_id": str(metadata.get("record_id", "")),
        "text": str(metadata.get("text", "")),
        "mention": " ".join(tokens[start:end]),
        "span": [start, end],
        "score": score,
        "outcome": int(outcome),
        "fine_region_index": int(fine_index),
        "fine_object_label": object_label,
        **dimensions,
    }


@torch.no_grad()
def collect_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config = load_siglip2_region_reliability_config(args.config)
    if args.device:
        config.runtime.device = args.device
    if config.model.feature_mode == "vinvl_only":
        raise ValueError(
            "Slice diagnostics require siglip2_only or fusion so agreement "
            "features use the same cached SigLIP 2 evidence."
        )
    dataset, collator = _paired_dataset(config, root, "dev")
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
        evidence_checkpoint,
    ) = load_frozen_reliability_chain(config, root, device)
    validate_fingerprints(
        _base_paired(dataset),
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    checkpoint_path = resolve(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_mode = str(
        ((checkpoint.get("config") or {}).get("model") or {}).get(
            "feature_mode", ""
        )
    )
    if checkpoint_mode and checkpoint_mode != config.model.feature_mode:
        raise ValueError(
            f"Checkpoint mode {checkpoint_mode} != config mode "
            f"{config.model.feature_mode}."
        )
    expected_signature = checkpoint.get("siglip2_dev_build_signature")
    actual_signature = dataset.siglip2.manifest.get("build_signature")
    if expected_signature and expected_signature != actual_signature:
        raise ValueError("Checkpoint and dev SigLIP 2 cache signatures differ.")
    if checkpoint.get("evidence_visibility_checkpoint_epoch") != (
        evidence_checkpoint.get("epoch")
    ):
        raise ValueError("Checkpoint and Evidence Visibility epochs differ.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )

    source_path = resolve(args.source_file, root)
    sources = _source_records(source_path)
    source_by_id = {str(record.get("id")): record for record in sources}
    vinvl_dir = resolve(args.vinvl_dir, root)
    detector_budget = int(config.evaluation.detector_reference_budget)
    loss_options = vars(config.loss).copy()
    all_decode_options = decode_options(hierarchy_config)
    entity_threshold = float(all_decode_options.get("entity_threshold", 0.0))
    decode_strategy = str(all_decode_options.get("decode_strategy", "interval"))
    stage1_spans_only = bool(
        all_decode_options.get("stage1_spans_only", True)
    )
    region_options = {
        key: value
        for key, value in all_decode_options.items()
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    risk_threshold = (
        float(args.risk_threshold)
        if args.risk_threshold is not None
        else float((checkpoint.get("metrics") or {}).get("risk_best_threshold", 1.0))
    )
    hard_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    object_alignment = Counter()

    for raw_batch in loader:
        paired = move_paired_record_batch(raw_batch, device)
        formal = paired["formal"]
        expanded = paired["expanded"]
        baseline = frozen_hierarchical_context(
            hierarchy, formal, expanded, decode_options=region_options
        )
        hierarchy_outputs = baseline["outputs"]
        hierarchy_visible = baseline["visible_mask"]
        decoded = baseline["decoded"]
        assert isinstance(hierarchy_outputs, dict)
        assert isinstance(hierarchy_visible, torch.Tensor)
        assert isinstance(decoded, dict)
        fine_outputs = fine_model(expanded)
        hierarchy_outputs, _, baseline_visible = frozen_current_visibility_context(
            evidence_model,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            hierarchy_visible_mask=hierarchy_visible,
            base_is_null_mask=decoded["base_is_null"],
            decode_options=all_decode_options,
        )
        outputs = model(
            fine_outputs,
            hierarchy_outputs,
            expanded,
            baseline_visible_mask=baseline_visible,
            base_is_null_mask=decoded["base_is_null"],
            siglip2_features=paired["siglip2"],
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
        selected_mask = torch.zeros_like(expanded["span_mask"], dtype=torch.bool)
        spans_by_row: list[list[tuple[int, int]]] = []
        for row in range(len(formal["metadata"])):
            spans, selected = _selected_span_indices(
                hierarchy_outputs,
                formal,
                row,
                entity_threshold=entity_threshold,
                decode_strategy=decode_strategy,
                stage1_spans_only=stage1_spans_only,
            )
            spans_by_row.append(spans)
            if selected:
                selected_mask[row, torch.tensor(selected, device=device)] = True

        a_mask = selected_mask & supervision["group_a_mask"]
        b_mask = (
            selected_mask
            & supervision["group_b_mask"]
            & supervision["candidate_covered_mask"]
        )
        selected_gold = selected_mask & supervision["eligible_mask"]
        null_gold = selected_gold & ~supervision["visible_mask"]
        action_mask = (
            selected_mask
            & ~baseline_visible
            & outputs["candidate_mask"].any(dim=-1)
        )
        fine_indices = outputs["fine_top1_region_index"].long()
        top_scores = outputs["fine_top1_reliability"].float()

        for row, metadata in enumerate(expanded["metadata"]):
            record_id = str(metadata.get("record_id", ""))
            source = source_by_id.get(record_id) or {}
            image_id = Path(str(source.get("image", ""))).stem
            npz_path = vinvl_dir / f"{image_id}.jpg.npz"
            real_mask = (
                expanded["region_mask"][row].bool()
                & ~expanded["region_is_null"][row].bool()
            )
            object_labels = align_vinvl_object_labels(
                expanded["region_boxes"][row], real_mask, npz_path
            )
            object_alignment["records"] += 1
            object_alignment["known_regions"] += sum(
                label != "unknown" for label in object_labels
            )
            object_alignment["real_regions"] += int(real_mask.sum().item())
            span_count = int(expanded["span_mask"][row].sum().item())
            for span_index in range(span_count):
                is_hard = bool(
                    (a_mask[row, span_index] | b_mask[row, span_index]).item()
                )
                is_action = bool(action_mask[row, span_index].item())
                if not is_hard and not is_action:
                    continue
                fine_index = int(fine_indices[row, span_index].item())
                score = float(top_scores[row, span_index].item())
                dimensions = _record_dimensions(
                    row=row,
                    span_index=span_index,
                    fine_index=fine_index,
                    expanded=expanded,
                    fine_outputs=fine_outputs,
                    outputs=outputs,
                    supervision=supervision,
                    object_labels=object_labels,
                    detector_reference_budget=detector_budget,
                    context_expansion=float(args.context_expansion),
                    context_overlap_threshold=float(
                        args.context_overlap_threshold
                    ),
                    small_area_threshold=float(args.small_area_threshold),
                    large_area_threshold=float(args.large_area_threshold),
                )
                if is_hard:
                    hard_rows.append(
                        {
                            "score": score,
                            "label": bool(a_mask[row, span_index].item()),
                            **dimensions,
                        }
                    )
                if is_action:
                    outcome = 0
                    if bool(a_mask[row, span_index].item()):
                        outcome = 1
                    elif bool(
                        (null_gold[row, span_index] & ~baseline_visible[row, span_index]).item()
                    ):
                        outcome = -1
                    action = {
                        "score": score,
                        "outcome": outcome,
                        **dimensions,
                    }
                    if score >= risk_threshold:
                        risk_rows.append(action)
                        name = {1: "fix", -1: "damage", 0: "neutral"}[outcome]
                        examples[name].append(
                            _example(
                                metadata=metadata,
                                span=spans_by_row[row][span_index],
                                score=score,
                                outcome=outcome,
                                fine_index=fine_index,
                                object_label=(
                                    object_labels[fine_index]
                                    if 0 <= fine_index < len(object_labels)
                                    else "unknown"
                                ),
                                dimensions=dimensions,
                            )
                        )

    for values in examples.values():
        values.sort(key=lambda item: float(item["score"]), reverse=True)
        del values[max(int(args.examples_per_outcome), 0) :]
    global_hard = binary_slice_summary(hard_rows)
    global_risk = risk_slice_summary(risk_rows)
    expected_metrics = dict(checkpoint.get("metrics") or {})
    return {
        "protocol": {
            "split": "dev",
            "test_read": False,
            "config": str(resolve(args.config, root)),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "feature_mode": config.model.feature_mode,
            "evidence_visibility_checkpoint_epoch": evidence_checkpoint.get(
                "epoch"
            ),
            "siglip2_dev_build_signature": actual_signature,
            "risk_threshold": risk_threshold,
            "detector_reference_budget": detector_budget,
        },
        "definitions": {
            "hard_a": (
                "selected gold span/type, gold visible, current Evidence "
                "Visibility KEEP is NULL, Fine top1 is IoU-correct"
            ),
            "hard_b": (
                "same eligibility as hard A, but Fine top1 is wrong and a "
                "positive R36 candidate exists"
            ),
            "context_overlap": (
                "max IoU between the 1.5x square Fine-top1 context box and "
                f"another real candidate; high >= {args.context_overlap_threshold}"
            ),
            "box_scale": (
                "Fine-top1 normalized area: small < "
                f"{args.small_area_threshold}, medium < "
                f"{args.large_area_threshold}, otherwise large"
            ),
            "object_class_relation": (
                "VinVL object label of Fine top1 compared with labels of all "
                "IoU-positive candidates"
            ),
        },
        "global_hard_ab": global_hard,
        "checkpoint_metric_consistency": {
            "checkpoint_hard_ab_auc": expected_metrics.get("hard_ab_auc"),
            "recomputed_hard_ab_auc": global_hard["hard_ab_auc"],
            "checkpoint_risk_best_net_correction": expected_metrics.get(
                "risk_best_net_correction"
            ),
            "recomputed_risk_net_correction": global_risk["net_correction"],
        },
        "hard_ab_slices": summarize_slices(hard_rows, binary=True),
        "risk_tail": global_risk,
        "risk_tail_slices": summarize_slices(risk_rows, binary=False),
        "risk_tail_examples": dict(examples),
        "object_alignment": {
            **{key: float(value) for key, value in object_alignment.items()},
            "known_region_rate": object_alignment["known_regions"]
            / max(object_alignment["real_regions"], 1),
        },
    }


def main() -> None:
    args = parse_args()
    payload = collect_diagnostics(args)
    root = Path(__file__).resolve().parents[1]
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "global_hard_ab": payload["global_hard_ab"],
                "risk_tail": payload["risk_tail"],
                "object_alignment": payload["object_alignment"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
