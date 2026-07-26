"""Dev-only evaluation for the trainable FMNERG encoder sidecar."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .encoder_model import TrainableSubtypeEncoder
from .evaluator import evaluate_formal_subtype_ids
from .metrics import (
    subtype_classification_metrics,
    subtype_classification_report,
)
from .online_data import OnlineSubtypeCollator, OnlineSubtypeRecordDataset
from .taxonomy import SubtypeTaxonomy


def move_online_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def online_model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    keys = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "span_record_indices",
        "span_start_indices",
        "span_end_indices",
        "span_token_mask",
        "coarse_type_ids",
    )
    return {
        key: batch[key]
        for key in keys
        if key in batch
    }


@torch.inference_mode()
def predict_online_subtypes(
    model: TrainableSubtypeEncoder,
    dataset: OnlineSubtypeRecordDataset,
    *,
    collator: OnlineSubtypeCollator,
    batch_size: int,
    device: torch.device,
    coarse_type_ids: torch.Tensor | None = None,
) -> list[int]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    predictions = [-1] * len(dataset.examples)
    override = None
    if coarse_type_ids is not None:
        override = torch.as_tensor(coarse_type_ids, dtype=torch.long)
        if override.ndim != 1 or override.numel() != len(dataset.examples):
            raise ValueError(
                "Online subtype parent override must align with examples."
            )
    for raw_batch in loader:
        batch = move_online_batch(raw_batch, device)
        if override is not None:
            batch["coarse_type_ids"] = override[
                raw_batch["example_indices"].long()
            ].to(device)
        outputs = model(**online_model_inputs(batch))
        values = outputs["predicted_subtype_ids"].detach().cpu().tolist()
        for example_index, subtype_id in zip(
            raw_batch["example_indices"].tolist(),
            values,
        ):
            if predictions[int(example_index)] >= 0:
                raise ValueError(
                    f"Duplicate online subtype example {example_index}."
                )
            predictions[int(example_index)] = int(subtype_id)
    if any(value < 0 for value in predictions):
        raise RuntimeError("Online subtype prediction coverage is incomplete.")
    return predictions


@torch.inference_mode()
def evaluate_online_gold_spans(
    model: TrainableSubtypeEncoder,
    dataset: OnlineSubtypeRecordDataset,
    *,
    collator: OnlineSubtypeCollator,
    taxonomy: SubtypeTaxonomy,
    batch_size: int,
    device: torch.device,
    include_detailed: bool = False,
) -> dict[str, Any]:
    predicted = predict_online_subtypes(
        model,
        dataset,
        collator=collator,
        batch_size=batch_size,
        device=device,
    )
    gold = [int(value) for value in dataset.subtype_ids.tolist()]
    raw = subtype_classification_metrics(
        predicted,
        gold,
        num_classes=taxonomy.num_subtypes,
    )
    report = subtype_classification_report(
        predicted,
        gold,
        taxonomy=taxonomy,
    )
    metrics: dict[str, Any] = {
        "subtype_accuracy_on_gold_spans": raw["subtype_accuracy"],
        "subtype_micro_f1_on_gold_spans": raw["subtype_micro_f1"],
        "subtype_macro_f1_on_gold_spans": raw["subtype_macro_f1"],
        "gold_span_examples": float(len(gold)),
    }
    for parent, score in report["parent_macro_f1"].items():
        metrics[f"subtype_{parent.lower()}_macro_f1_on_gold_spans"] = score
    if include_detailed:
        metrics["subtype_classification_report_on_gold_spans"] = report
    return metrics


@torch.inference_mode()
def evaluate_online_formal_predictions(
    model: TrainableSubtypeEncoder,
    dataset: OnlineSubtypeRecordDataset,
    formal_payload: dict[str, Any],
    *,
    collator: OnlineSubtypeCollator,
    taxonomy: SubtypeTaxonomy,
    batch_size: int,
    device: torch.device,
    include_records: bool = False,
) -> dict[str, Any]:
    subtype_predictions = predict_online_subtypes(
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
                taxonomy.parent_id(int(subtype_id))
                for subtype_id in target_subtype_ids[target_available].tolist()
            ],
            dtype=torch.long,
        )
    gold_parent_predictions = predict_online_subtypes(
        model,
        dataset,
        collator=collator,
        batch_size=batch_size,
        device=device,
        coarse_type_ids=oracle_parent_ids,
    )
    return evaluate_formal_subtype_ids(
        subtype_predictions=subtype_predictions,
        gold_parent_predictions=gold_parent_predictions,
        target_subtype_ids=target_subtype_ids,
        predicted_parent_ids=dataset.coarse_type_ids,
        examples=dataset.examples,
        formal_payload=formal_payload,
        taxonomy=taxonomy,
        include_records=include_records,
    )
