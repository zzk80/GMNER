"""Dev-only evaluation for fixed-region J0 subtype fusion."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from sidecars.fmnerg_subtype.evaluator import evaluate_formal_subtype_ids
from sidecars.fmnerg_subtype.metrics import (
    subtype_classification_metrics,
    subtype_classification_report,
)
from sidecars.fmnerg_subtype.online_data import OnlineSubtypeRecordDataset
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy

from .data import JointOnlineSubtypeCollator
from .model import J0VisualSubtypeFusion


def move_joint_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def joint_model_inputs(
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    keys = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "span_record_indices",
        "span_start_indices",
        "span_end_indices",
        "span_token_mask",
        "coarse_type_ids",
        "joint_region_features",
        "joint_region_geometry",
        "joint_detector_scores",
        "joint_region_is_null",
        "joint_visual_available",
    )
    return {key: batch[key] for key in keys if key in batch}


@torch.inference_mode()
def predict_joint_subtypes(
    model: J0VisualSubtypeFusion,
    dataset: OnlineSubtypeRecordDataset,
    *,
    collator: JointOnlineSubtypeCollator,
    batch_size: int,
    device: torch.device,
    coarse_type_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    fused = [-1] * len(dataset.examples)
    base = [-1] * len(dataset.examples)
    residual_abs_sum = 0.0
    residual_count = 0
    override = None
    if coarse_type_ids is not None:
        override = torch.as_tensor(coarse_type_ids, dtype=torch.long)
        if override.ndim != 1 or override.numel() != len(dataset.examples):
            raise ValueError(
                "J0 parent override must align with every example."
            )
    for raw_batch in loader:
        batch = move_joint_batch(raw_batch, device)
        if override is not None:
            batch["coarse_type_ids"] = override[
                raw_batch["example_indices"].long()
            ].to(device)
        outputs = model(**joint_model_inputs(batch))
        fused_values = (
            outputs["predicted_subtype_ids"].detach().cpu().tolist()
        )
        base_values = (
            outputs["base_predicted_subtype_ids"].detach().cpu().tolist()
        )
        residual = outputs["bounded_visual_residual_logits"].detach()
        residual_abs_sum += float(residual.abs().sum().item())
        residual_count += int(residual.numel())
        for example_index, fused_id, base_id in zip(
            raw_batch["example_indices"].tolist(),
            fused_values,
            base_values,
        ):
            index = int(example_index)
            if fused[index] >= 0 or base[index] >= 0:
                raise ValueError(f"Duplicate J0 example index {index}.")
            fused[index] = int(fused_id)
            base[index] = int(base_id)
    if any(value < 0 for value in fused + base):
        raise RuntimeError("J0 subtype prediction coverage is incomplete.")
    changed = sum(
        int(fused_id != base_id)
        for fused_id, base_id in zip(fused, base)
    )
    return {
        "fused": fused,
        "base": base,
        "prediction_changed_count": changed,
        "prediction_changed_rate": changed / max(len(fused), 1),
        "visual_residual_abs_mean": (
            residual_abs_sum / max(residual_count, 1)
        ),
    }


@torch.inference_mode()
def evaluate_joint_gold_spans(
    model: J0VisualSubtypeFusion,
    dataset: OnlineSubtypeRecordDataset,
    *,
    collator: JointOnlineSubtypeCollator,
    taxonomy: SubtypeTaxonomy,
    batch_size: int,
    device: torch.device,
    include_detailed: bool = False,
) -> dict[str, Any]:
    predictions = predict_joint_subtypes(
        model,
        dataset,
        collator=collator,
        batch_size=batch_size,
        device=device,
    )
    gold = [int(value) for value in dataset.subtype_ids.tolist()]
    fused = subtype_classification_metrics(
        predictions["fused"],
        gold,
        num_classes=taxonomy.num_subtypes,
    )
    base = subtype_classification_metrics(
        predictions["base"],
        gold,
        num_classes=taxonomy.num_subtypes,
    )
    report = subtype_classification_report(
        predictions["fused"],
        gold,
        taxonomy=taxonomy,
    )
    metrics: dict[str, Any] = {
        "subtype_accuracy_on_gold_spans": fused["subtype_accuracy"],
        "subtype_micro_f1_on_gold_spans": fused["subtype_micro_f1"],
        "subtype_macro_f1_on_gold_spans": fused["subtype_macro_f1"],
        "j0_base_subtype_accuracy_on_gold_spans": base[
            "subtype_accuracy"
        ],
        "j0_base_subtype_macro_f1_on_gold_spans": base[
            "subtype_macro_f1"
        ],
        "j0_gold_prediction_changed_rate": predictions[
            "prediction_changed_rate"
        ],
        "j0_visual_residual_abs_mean_on_gold_spans": predictions[
            "visual_residual_abs_mean"
        ],
        "gold_span_examples": float(len(gold)),
    }
    for parent, score in report["parent_macro_f1"].items():
        metrics[f"subtype_{parent.lower()}_macro_f1_on_gold_spans"] = score
    if include_detailed:
        metrics["subtype_classification_report_on_gold_spans"] = report
    return metrics


@torch.inference_mode()
def evaluate_joint_formal_predictions(
    model: J0VisualSubtypeFusion,
    dataset: OnlineSubtypeRecordDataset,
    formal_payload: dict[str, Any],
    *,
    collator: JointOnlineSubtypeCollator,
    taxonomy: SubtypeTaxonomy,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    predicted = predict_joint_subtypes(
        model,
        dataset,
        collator=collator,
        batch_size=batch_size,
        device=device,
    )
    target_subtype_ids = dataset.subtype_ids.long()
    target_available = target_subtype_ids.ge(0)
    oracle_parent_ids = dataset.coarse_type_ids.clone().long()
    if target_available.any():
        oracle_parent_ids[target_available] = torch.tensor(
            [
                taxonomy.parent_id(int(value))
                for value in target_subtype_ids[target_available].tolist()
            ],
            dtype=torch.long,
        )
    oracle_predictions = predict_joint_subtypes(
        model,
        dataset,
        collator=collator,
        batch_size=batch_size,
        device=device,
        coarse_type_ids=oracle_parent_ids,
    )
    final = evaluate_formal_subtype_ids(
        subtype_predictions=predicted["fused"],
        gold_parent_predictions=oracle_predictions["fused"],
        target_subtype_ids=target_subtype_ids,
        predicted_parent_ids=dataset.coarse_type_ids,
        examples=dataset.examples,
        formal_payload=formal_payload,
        taxonomy=taxonomy,
    )
    base = evaluate_formal_subtype_ids(
        subtype_predictions=predicted["base"],
        gold_parent_predictions=oracle_predictions["base"],
        target_subtype_ids=target_subtype_ids,
        predicted_parent_ids=dataset.coarse_type_ids,
        examples=dataset.examples,
        formal_payload=formal_payload,
        taxonomy=taxonomy,
    )
    if (
        not final["metadata"]["gmner_identity_exact"]
        or not base["metadata"]["gmner_identity_exact"]
    ):
        raise AssertionError("J0 changed the frozen GMNER prediction chain.")
    corrected = damaged = 0
    for fused_id, base_id, target in zip(
        predicted["fused"],
        predicted["base"],
        target_subtype_ids.tolist(),
    ):
        if int(target) < 0:
            continue
        corrected += int(base_id != target and fused_id == target)
        damaged += int(base_id == target and fused_id != target)
    final_metrics = dict(final["metrics"])
    final_metrics.update(
        {
            "j0_current_text_fine_mner_f1": float(
                base["metrics"]["fine_mner_f1"]
            ),
            "j0_current_text_fmnerg_f1": float(
                base["metrics"]["fmnerg_f1"]
            ),
            "j0_visual_residual_fmnerg_delta": float(
                final["metrics"]["fmnerg_f1"]
                - base["metrics"]["fmnerg_f1"]
            ),
            "j0_formal_prediction_changed_count": float(
                predicted["prediction_changed_count"]
            ),
            "j0_formal_prediction_changed_rate": float(
                predicted["prediction_changed_rate"]
            ),
            "j0_subtype_corrected": float(corrected),
            "j0_subtype_damaged": float(damaged),
            "j0_subtype_net_corrections": float(corrected - damaged),
            "j0_visual_residual_abs_mean": float(
                predicted["visual_residual_abs_mean"]
            ),
        }
    )
    return {
        "metadata": {
            **final["metadata"],
            "kind": "fmnerg_joint_j0_dev_evaluation",
            "format_version": 1,
            "experiment_mode": model.experiment_mode,
            "formal_stage1_mutated": False,
            "formal_region_mutated": False,
            "test_accessed": False,
        },
        "metrics": final_metrics,
    }
