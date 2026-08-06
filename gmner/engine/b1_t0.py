"""Training and evaluation helpers for the B1-T0 OOF experiment."""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gmner.models.b1_t0 import B1T0TextCorrectionModel


def load_fold_payload(path: str, expected_fold: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if (
        payload.get("kind") != "b1_t0_frozen_text_fold_features"
        or int(payload.get("fold_id", -1)) != int(expected_fold)
        or payload.get("dev_accessed") is not False
        or payload.get("test_accessed") is not False
    ):
        raise ValueError(f"Invalid B1-T0 feature payload for fold {expected_fold}.")
    examples = list(payload.get("examples") or [])
    if any(int(item["fold_id"]) != int(expected_fold) for item in examples):
        raise ValueError("Feature payload contains another fold.")
    payload["examples"] = examples
    return payload


def load_fold_features(path: str, expected_fold: int) -> list[dict[str, Any]]:
    return load_fold_payload(path, expected_fold)["examples"]


def mention_counts(examples: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(str(item["mention"]) for item in examples)


def scalar_vector(item: dict[str, Any], counts: Counter[str]) -> list[float]:
    logits = [float(value) for value in item["type_logits"]]
    shifted = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0)
    probabilities = [float(value) for value in shifted.tolist()]
    ranked = sorted(range(4), key=lambda index: (-probabilities[index], index))
    base_type = int(item["base_type_id"])
    confidence = probabilities[ranked[0]]
    margin = probabilities[ranked[0]] - probabilities[ranked[1]]
    entropy = -sum(value * math.log(max(value, 1e-30)) for value in probabilities)
    base_one_hot = [float(index == base_type) for index in range(4)]
    top2_one_hot = [float(index == ranked[1]) for index in range(4)]
    frequency = int(counts.get(str(item["mention"]), 0))
    return [
        *logits,
        *probabilities,
        *base_one_hot,
        confidence,
        margin,
        entropy,
        float(item["span_base_score"]),
        float(item["span_word_length"]),
        float(item["span_character_length"]),
        float(item["uppercase_ratio"]),
        float(item["digit_ratio"]),
        math.log1p(frequency),
        float(frequency > 0),
        *top2_one_hot,
    ]


def tensors(
    examples: list[dict[str, Any]], counts: Counter[str]
) -> tuple[torch.Tensor, ...]:
    text = torch.stack([item["text_embedding"].float() for item in examples])
    scalar = torch.tensor(
        [scalar_vector(item, counts) for item in examples], dtype=torch.float32
    )
    gate = torch.tensor([float(item["base_wrong"]) for item in examples])
    target = torch.tensor([int(item["gold_type_id"]) for item in examples], dtype=torch.long)
    base = torch.tensor([int(item["base_type_id"]) for item in examples], dtype=torch.long)
    fold = torch.tensor([int(item["fold_id"]) for item in examples], dtype=torch.long)
    return text, scalar, gate, target, base, fold


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_model(
    examples: list[dict[str, Any]],
    *,
    counts: Counter[str],
    config: dict[str, Any],
    seed: int,
    device: str,
) -> B1T0TextCorrectionModel:
    set_seed(seed)
    text, scalar, gate, target, base, _ = tensors(examples, counts)
    model = B1T0TextCorrectionModel(
        text_size=int(text.shape[1]),
        scalar_size=int(scalar.shape[1]),
        text_projection_size=int(config["text_projection_size"]),
        scalar_projection_size=int(config["scalar_projection_size"]),
        hidden_size=int(config["shared_hidden_size"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    dataset = TensorDataset(text, scalar, gate, target, base)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    for _ in range(int(config["epochs"])):
        model.train()
        for text_batch, scalar_batch, gate_batch, target_batch, base_batch in loader:
            text_batch = text_batch.to(device)
            scalar_batch = scalar_batch.to(device)
            gate_batch = gate_batch.to(device)
            target_batch = target_batch.to(device)
            base_batch = base_batch.to(device)
            gate_logits, target_logits = model(text_batch, scalar_batch)
            gate_loss = nn.functional.binary_cross_entropy_with_logits(
                gate_logits, gate_batch
            )
            positive = gate_batch.gt(0.5)
            if positive.any():
                masked_target = target_logits[positive].clone()
                masked_target.scatter_(
                    1, base_batch[positive].unsqueeze(1), -1e4
                )
                target_loss = nn.functional.cross_entropy(
                    masked_target, target_batch[positive]
                )
            else:
                target_loss = gate_logits.sum() * 0.0
            loss = gate_loss + target_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: B1T0TextCorrectionModel,
    examples: list[dict[str, Any]],
    *,
    counts: Counter[str],
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    text, scalar, gate, target, base, fold = tensors(examples, counts)
    gate_scores = []
    target_predictions = []
    for offset in range(0, len(examples), int(batch_size)):
        gate_logits, target_logits = model(
            text[offset : offset + batch_size].to(device),
            scalar[offset : offset + batch_size].to(device),
        )
        local_base = base[offset : offset + batch_size].to(device)
        target_logits.scatter_(1, local_base.unsqueeze(1), -1e4)
        gate_scores.append(torch.sigmoid(gate_logits).cpu())
        target_predictions.append(target_logits.argmax(dim=-1).cpu())
    return {
        "gate_score": torch.cat(gate_scores),
        "target_prediction": torch.cat(target_predictions),
        "gate_label": gate,
        "gold_type": target,
        "base_type": base,
        "fold": fold,
    }


def action_metrics(predictions: dict[str, torch.Tensor], threshold: float) -> dict[str, Any]:
    scores = predictions["gate_score"]
    target = predictions["target_prediction"]
    gate_label = predictions["gate_label"].bool()
    gold = predictions["gold_type"]
    action = scores.ge(float(threshold)) if math.isfinite(threshold) else torch.zeros_like(gate_label)
    corrected = action & gate_label & target.eq(gold)
    damaged = action & ~gate_label
    neutral = action & gate_label & ~target.eq(gold)
    action_count = int(action.sum())
    corrected_count = int(corrected.sum())
    damaged_count = int(damaged.sum())
    base_wrong = int(gate_label.sum())
    base_correct = int((~gate_label).sum())
    return {
        "threshold": float(threshold),
        "examples": int(len(gate_label)),
        "base_wrong": base_wrong,
        "base_correct": base_correct,
        "actions": action_count,
        "corrected": corrected_count,
        "damaged": damaged_count,
        "neutral_actions": int(neutral.sum()),
        "net": corrected_count - damaged_count,
        "action_precision": corrected_count / action_count if action_count else 0.0,
        "correction_recall": corrected_count / base_wrong if base_wrong else 0.0,
        "base_correct_preservation": 1.0 - damaged_count / base_correct if base_correct else 1.0,
        "target_accuracy_on_base_wrong": float(target[gate_label].eq(gold[gate_label]).float().mean())
        if base_wrong
        else 0.0,
    }


def freeze_threshold(
    predictions: dict[str, torch.Tensor], contract: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    scores = predictions["gate_score"]
    candidates = sorted({float(value) for value in scores.tolist()}, reverse=True)
    passing: list[tuple[tuple[float, ...], float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = action_metrics(predictions, threshold)
        if (
            metrics["corrected"] > metrics["damaged"]
            and metrics["net"] > 0
            and metrics["action_precision"]
            >= float(contract["minimum_action_precision"])
            and metrics["base_correct_preservation"]
            >= float(contract["minimum_base_correct_preservation"])
        ):
            key = (
                float(metrics["net"]),
                float(metrics["action_precision"]),
                -float(metrics["actions"]),
                float(threshold),
            )
            passing.append((key, threshold, metrics))
    if not passing:
        return float("inf"), action_metrics(predictions, float("inf"))
    _, threshold, metrics = max(passing, key=lambda item: item[0])
    return threshold, metrics


def ranking_diagnostics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    scores = scores.float()
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return {"auroc": 0.0, "auprc": 0.0}
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order].float()
    tp = torch.cumsum(ranked, dim=0)
    fp = torch.cumsum(1.0 - ranked, dim=0)
    precision = tp / (tp + fp)
    recall = tp / positives
    previous_recall = torch.cat([torch.zeros(1), recall[:-1]])
    auprc = float(((recall - previous_recall) * precision).sum())
    positive_ranks = torch.nonzero(labels[torch.argsort(scores, stable=True)], as_tuple=False).flatten().float() + 1
    rank_sum = float(positive_ranks.sum())
    auroc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return {"auroc": float(auroc), "auprc": auprc}


def concatenate_predictions(
    predictions: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not predictions:
        raise ValueError("No B1-T0 predictions to concatenate.")
    keys = tuple(predictions[0])
    if any(tuple(item) != keys for item in predictions):
        raise ValueError("B1-T0 prediction schemas differ.")
    return {key: torch.cat([item[key] for item in predictions]) for key in keys}


def mner_f1(correct: int, predicted: int, gold: int) -> float:
    denominator = int(predicted) + int(gold)
    return 2.0 * int(correct) / denominator if denominator else 0.0
