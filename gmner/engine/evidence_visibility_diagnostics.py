"""Small deterministic diagnostics for M3.3 visibility evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def release_threshold_logits(
    base_is_null: torch.Tensor,
    *,
    visible_from_null_threshold: float,
    null_from_visible_threshold: float,
) -> torch.Tensor:
    """Return the logit boundary needed to emit a real region per span."""

    probability = torch.where(
        base_is_null.bool(),
        torch.full_like(
            base_is_null.float(), float(visible_from_null_threshold)
        ),
        torch.full_like(
            base_is_null.float(), float(null_from_visible_threshold)
        ),
    ).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probability)


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Pairwise AUROC with exact tie handling and bounded memory."""

    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    positive = score[label]
    negative = score[~label]
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    wins = 0.0
    comparisons = 0
    for start in range(0, positive.numel(), 512):
        block = positive[start : start + 512, None]
        difference = block - negative[None, :]
        wins += float(difference.gt(0).sum().item())
        wins += 0.5 * float(difference.eq(0).sum().item())
        comparisons += int(difference.numel())
    return wins / max(comparisons, 1)


def binary_average_precision(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Average precision with score ties evaluated as one threshold group."""

    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    positive_count = int(label.sum().item())
    if positive_count == 0:
        return float("nan")
    order = torch.argsort(score, descending=True)
    sorted_score = score[order]
    sorted_label = label[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < sorted_score.numel():
        end = start + 1
        while (
            end < sorted_score.numel()
            and sorted_score[end].item() == sorted_score[start].item()
        ):
            end += 1
        group_positive = int(sorted_label[start:end].sum().item())
        true_positive += group_positive
        false_positive += (end - start) - group_positive
        recall = true_positive / positive_count
        precision = true_positive / max(true_positive + false_positive, 1)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return average_precision


def binary_balanced_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    positive = label
    negative = ~label
    if not positive.any() or not negative.any():
        return float("nan")
    predicted = score.ge(float(threshold))
    sensitivity = predicted[positive].float().mean()
    specificity = (~predicted[negative]).float().mean()
    return float((0.5 * (sensitivity + specificity)).item())


def best_binary_balanced_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    if not label.any() or label.all():
        return float("nan"), float("nan")
    thresholds = torch.unique(score).sort(descending=True).values
    thresholds = torch.cat(
        [thresholds[:1] + 1e-6, thresholds], dim=0
    )
    best_score = -1.0
    best_threshold = 0.5
    for threshold in thresholds.tolist():
        value = binary_balanced_accuracy(
            score, label, threshold=float(threshold)
        )
        if value > best_score:
            best_score = value
            best_threshold = float(threshold)
    return best_score, best_threshold


def binary_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 10,
) -> float:
    probability = probabilities.detach().float().reshape(-1).cpu().clamp(0, 1)
    label = labels.detach().float().reshape(-1).cpu()
    if probability.numel() == 0:
        return float("nan")
    error = 0.0
    boundaries = torch.linspace(0.0, 1.0, max(int(bins), 1) + 1)
    for index in range(boundaries.numel() - 1):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = probability.ge(lower) & (
            probability.le(upper)
            if index == boundaries.numel() - 2
            else probability.lt(upper)
        )
        if not selected.any():
            continue
        weight = float(selected.float().mean().item())
        confidence = float(probability[selected].mean().item())
        accuracy = float(label[selected].mean().item())
        error += weight * abs(confidence - accuracy)
    return error


def distribution_summary(values: torch.Tensor) -> dict[str, float]:
    data = values.detach().float().reshape(-1).cpu()
    if data.numel() == 0:
        return {"count": 0.0}
    quantiles = torch.quantile(
        data, torch.tensor([0.1, 0.5, 0.9], dtype=data.dtype)
    )
    return {
        "count": float(data.numel()),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "max": float(data.max().item()),
    }


def _balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predicted = logits.ge(0.0)
    positive = labels.bool()
    true_positive_rate = (
        predicted[positive].float().mean().item() if positive.any() else 0.0
    )
    negative = ~positive
    true_negative_rate = (
        (~predicted[negative]).float().mean().item()
        if negative.any()
        else 0.0
    )
    return 0.5 * (true_positive_rate + true_negative_rate)


def stratified_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    folds: int = 5,
    seed: int = 42,
    epochs: int = 200,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-3,
) -> dict[str, float | dict[str, float]]:
    """Cross-validated linear separability probe; never used for deployment."""

    x = torch.nan_to_num(
        features.detach().float().cpu(), nan=0.0, posinf=20.0, neginf=-20.0
    )
    y = labels.detach().bool().reshape(-1).cpu()
    if x.ndim != 2 or x.size(0) != y.numel():
        raise ValueError("features must be [N, D] and align with labels.")
    positive_indices = torch.nonzero(y, as_tuple=False).squeeze(-1)
    negative_indices = torch.nonzero(~y, as_tuple=False).squeeze(-1)
    usable_folds = min(
        int(folds), int(positive_indices.numel()), int(negative_indices.numel())
    )
    if usable_folds < 2:
        return {
            "samples": float(y.numel()),
            "positive": float(positive_indices.numel()),
            "negative": float(negative_indices.numel()),
            "folds": float(usable_folds),
            "auc": float("nan"),
            "balanced_accuracy": float("nan"),
        }
    generator = torch.Generator().manual_seed(int(seed))
    positive_indices = positive_indices[
        torch.randperm(positive_indices.numel(), generator=generator)
    ]
    negative_indices = negative_indices[
        torch.randperm(negative_indices.numel(), generator=generator)
    ]
    fold_ids = torch.full((y.numel(),), -1, dtype=torch.long)
    fold_ids[positive_indices] = torch.arange(
        positive_indices.numel()
    ) % usable_folds
    fold_ids[negative_indices] = torch.arange(
        negative_indices.numel()
    ) % usable_folds
    out_of_fold = torch.zeros(y.numel(), dtype=torch.float32)

    for fold in range(usable_folds):
        train = fold_ids.ne(fold)
        validation = fold_ids.eq(fold)
        train_x = x[train]
        train_y = y[train].float()
        mean = train_x.mean(dim=0, keepdim=True)
        scale = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
        normalized_train = (train_x - mean) / scale
        normalized_validation = (x[validation] - mean) / scale
        classifier = nn.Linear(x.size(-1), 1)
        nn.init.zeros_(classifier.weight)
        nn.init.zeros_(classifier.bias)
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        positive_count = train_y.sum().clamp_min(1.0)
        negative_count = train_y.numel() - positive_count
        positive_weight = (negative_count / positive_count).detach()
        for _ in range(max(int(epochs), 1)):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(normalized_train).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                train_y,
                pos_weight=positive_weight,
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            out_of_fold[validation] = classifier(
                normalized_validation
            ).squeeze(-1)

    return {
        "samples": float(y.numel()),
        "positive": float(positive_indices.numel()),
        "negative": float(negative_indices.numel()),
        "folds": float(usable_folds),
        "auc": binary_auc(out_of_fold, y),
        "balanced_accuracy": _balanced_accuracy(out_of_fold, y),
        "positive_score": distribution_summary(out_of_fold[y]),
        "negative_score": distribution_summary(out_of_fold[~y]),
    }
