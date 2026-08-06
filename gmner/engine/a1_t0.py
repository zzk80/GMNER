"""Training, calibration, grouped selection, and evaluation for A1-T0."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gmner.models.a1_t0 import A1T0ActionModel, CLASS_ORDER, SOURCE_ORDER


def load_frozen_protocol(root: Path, authorization: dict[str, Any]) -> dict[str, Any]:
    """Load and verify the exact preregistered protocol Git blob."""
    commit = str(authorization["frozen_protocol_commit"])
    protocol_path = str(authorization["frozen_protocol"]).replace("\\", "/")
    content = subprocess.check_output(
        ["git", "show", f"{commit}:{protocol_path}"],
        cwd=root,
    )
    actual_sha256 = hashlib.sha256(content).hexdigest()
    expected_sha256 = str(authorization["frozen_protocol_git_blob_sha256"])
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Frozen A1-T0 protocol blob SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    return json.loads(content.decode("utf-8"))


def load_fold(path: str, fold_id: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if (
        payload.get("kind") != "a1_t0_strict_observable_tabular_dataset"
        or int(payload.get("fold_id", -1)) != int(fold_id)
        or tuple(payload.get("class_order", ())) != CLASS_ORDER
        or tuple(payload.get("source_order", ())) != SOURCE_ORDER
        or payload.get("dev_accessed") is not False
        or payload.get("test_accessed") is not False
    ):
        raise ValueError(f"Invalid A1-T0 fold payload: {fold_id}")
    if not torch.isfinite(payload["numeric_features"]).all():
        raise ValueError("A1-T0 fold contains non-finite features.")
    return payload


def concatenate_folds(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(payloads)
    if not items:
        raise ValueError("No A1-T0 folds supplied.")
    names = tuple(items[0]["numeric_feature_names"])
    if any(tuple(item["numeric_feature_names"]) != names for item in items):
        raise ValueError("A1-T0 feature schemas differ.")
    return {
        "numeric_feature_names": names,
        "numeric_features": torch.cat([item["numeric_features"] for item in items]),
        "source_ids": torch.cat([item["source_ids"] for item in items]),
        "labels": torch.cat([item["labels"] for item in items]),
        "metadata": [entry for item in items for entry in item["metadata"]],
        "record_contract": {
            key: value for item in items for key, value in item["record_contract"].items()
        },
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def fit_standardizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.double().mean(dim=0).float()
    std = features.double().std(dim=0, unbiased=False).float().clamp_min(1e-6)
    return mean, std


def source_one_hot(source_ids: torch.Tensor) -> torch.Tensor:
    return nn.functional.one_hot(source_ids.long(), num_classes=len(SOURCE_ORDER)).float()


def train_model(
    data: dict[str, Any],
    *,
    config: dict[str, Any],
    seed: int,
    source_aware: bool,
    device: str,
) -> tuple[A1T0ActionModel, torch.Tensor, torch.Tensor]:
    set_seed(seed)
    features = data["numeric_features"].float()
    mean, std = fit_standardizer(features)
    normalized = (features - mean) / std
    sources = source_one_hot(data["source_ids"])
    labels = data["labels"].long()
    model = A1T0ActionModel(
        numeric_size=int(features.shape[1]),
        source_aware=source_aware,
        projection_size=int(config["input_projection_size"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(normalized, sources, labels),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    for _epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for numeric, source, target in loader:
            logits = model(numeric.to(device), source.to(device))
            loss = nn.functional.cross_entropy(logits, target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
    model.eval()
    return model, mean, std


@torch.no_grad()
def predict_logits(
    model: A1T0ActionModel,
    data: dict[str, Any],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    numeric = (data["numeric_features"].float() - mean) / std
    sources = source_one_hot(data["source_ids"])
    outputs = []
    for offset in range(0, len(numeric), int(batch_size)):
        outputs.append(
            model(
                numeric[offset : offset + batch_size].to(device),
                sources[offset : offset + batch_size].to(device),
            ).cpu()
        )
    return torch.cat(outputs)


def nll_at_temperature(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    return float(
        nn.functional.cross_entropy(
            logits.double() / float(temperature), labels.long(), reduction="mean"
        )
    )


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor, bounds: list[float]
) -> float:
    """Deterministic golden-section minimization in log-temperature space."""
    lower, upper = math.log(float(bounds[0])), math.log(float(bounds[1]))
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    for _ in range(100):
        left_loss = nll_at_temperature(logits, labels, math.exp(left))
        right_loss = nll_at_temperature(logits, labels, math.exp(right))
        if left_loss <= right_loss:
            upper, right = right, left
            left = upper - ratio * (upper - lower)
        else:
            lower, left = left, right
            right = lower + ratio * (upper - lower)
    return math.exp((lower + upper) / 2.0)


def calibrated_probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits.double() / float(temperature), dim=-1).float()


def group_winners(
    probabilities: torch.Tensor,
    metadata: list[dict[str, Any]],
    lambda_damage: float,
    lambda_neutral: float,
) -> list[tuple[int, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[str(item["base_prediction_id"])].append(index)
    winners = []
    for group_id in sorted(groups):
        candidates = []
        for index in groups[group_id]:
            probability = probabilities[index]
            utility = float(
                probability[0]
                - float(lambda_damage) * probability[2]
                - float(lambda_neutral) * probability[1]
            )
            item = metadata[index]
            key = (
                utility,
                float(probability[0]),
                -float(probability[2]),
                float(item["candidate_score"]),
            )
            candidates.append((key, str(item["action_id"]), index, utility))
        candidates.sort(key=lambda item: (item[0], _reverse_lex(item[1])), reverse=True)
        _, _, index, utility = candidates[0]
        winners.append((index, utility))
    return winners


def _reverse_lex(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)


def quantile_deltas(utilities: list[float], quantiles: list[float]) -> list[float]:
    values = torch.tensor(utilities, dtype=torch.float64)
    candidates = {0.0}
    for quantile in quantiles:
        candidates.add(float(torch.quantile(values, float(quantile), interpolation="linear")))
    return sorted(candidates)


def selected_indices(winners: list[tuple[int, float]], delta: float | None) -> list[int]:
    if delta is None:
        return []
    return [index for index, utility in winners if float(utility) > float(delta)]


def action_summary(
    selected: list[int], data: dict[str, Any]
) -> dict[str, Any]:
    metadata = data["metadata"]
    labels = data["labels"]
    counts = Counter(CLASS_ORDER[int(labels[index])] for index in selected)
    per_fold = {}
    for fold in sorted({int(item["fold_id"]) for item in metadata}):
        local = [index for index in selected if int(metadata[index]["fold_id"]) == fold]
        local_counts = Counter(CLASS_ORDER[int(labels[index])] for index in local)
        per_fold[str(fold)] = {
            "actions": len(local),
            "corrected": local_counts["FIX"],
            "neutral": local_counts["NEUTRAL"],
            "damaged": local_counts["DAMAGE"],
            "net": local_counts["FIX"] - local_counts["DAMAGE"],
        }
    by_source = {}
    for source in SOURCE_ORDER:
        local = [index for index in selected if metadata[index]["candidate_source"] == source]
        local_counts = Counter(CLASS_ORDER[int(labels[index])] for index in local)
        by_source[source] = {
            "actions": len(local),
            "corrected": local_counts["FIX"],
            "neutral": local_counts["NEUTRAL"],
            "damaged": local_counts["DAMAGE"],
            "net": local_counts["FIX"] - local_counts["DAMAGE"],
            "action_precision": local_counts["FIX"] / len(local) if local else 0.0,
        }
    baseline_correct = sum(int(item["base_mner_correct"]) for item in data["record_contract"].values())
    lost_by_record: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    gained_by_record: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for index in selected:
        item = metadata[index]
        record_id = str(item["record_id"])
        for entity in item["lost_correct_mner_entities"]:
            lost_by_record[record_id].add(tuple(int(value) for value in entity))
        if int(item["mner_correct_delta"]) > 0:
            gained_by_record[record_id].add(
                (*[int(value) for value in item["candidate_span"]], int(item["candidate_type_id"]))
            )
    lost = sum(len(value) for value in lost_by_record.values())
    gained = sum(len(value) for value in gained_by_record.values())
    mner_delta = gained - lost
    prediction_count = sum(int(item["prediction_count"]) for item in data["record_contract"].values())
    gold_count = sum(int(item["gold_count"]) for item in data["record_contract"].values())
    denominator = prediction_count + gold_count
    baseline_f1 = 2.0 * baseline_correct / denominator if denominator else 0.0
    final_f1 = 2.0 * (baseline_correct + mner_delta) / denominator if denominator else 0.0
    conflicts, identity = final_set_invariants(selected, data)
    actions = len(selected)
    return {
        "actions": actions,
        "corrected": counts["FIX"],
        "neutral": counts["NEUTRAL"],
        "damaged": counts["DAMAGE"],
        "net": counts["FIX"] - counts["DAMAGE"],
        "action_precision": counts["FIX"] / actions if actions else 0.0,
        "formal_correct_preservation": 1.0 - lost / baseline_correct if baseline_correct else 1.0,
        "lost_correct_entities": lost,
        "gained_correct_entities": gained,
        "baseline_mner_correct": baseline_correct,
        "mner_correct_delta": mner_delta,
        "baseline_mner_f1": baseline_f1,
        "final_mner_f1": final_f1,
        "mner_f1_delta": final_f1 - baseline_f1,
        "per_fold": per_fold,
        "by_source": by_source,
        "record_conflict_count": conflicts,
        **identity,
    }


def final_set_invariants(selected: list[int], data: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    replacements = {data["metadata"][index]["base_prediction_id"]: data["metadata"][index] for index in selected}
    conflict_count = 0
    prediction_count_identity = True
    for record_id, contract in data["record_contract"].items():
        final = []
        for prediction in contract["formal_predictions"]:
            replacement = replacements.get(prediction["prediction_id"])
            if replacement is None:
                final.append((tuple(prediction["span"]), prediction["type_id"], prediction["region_candidate_id"]))
            else:
                final.append((tuple(replacement["candidate_span"]), replacement["candidate_type_id"], replacement["candidate_region_candidate_id"]))
        prediction_count_identity &= len(final) == len(contract["formal_predictions"])
        for left in range(len(final)):
            for right in range(left + 1, len(final)):
                a, b = final[left][0], final[right][0]
                if min(a[1], b[1]) > max(a[0], b[0]):
                    conflict_count += 1
    return conflict_count, {
        "prediction_count_identity": bool(prediction_count_identity),
        "type_identity": all(
            item["candidate_type_id"]
            == next(
                prediction["type_id"]
                for prediction in data["record_contract"][item["record_id"]]["formal_predictions"]
                if prediction["prediction_id"] == item["base_prediction_id"]
            )
            for item in (data["metadata"][index] for index in selected)
        ),
        "region_null_identity": all(
            item["candidate_region_candidate_id"]
            == next(
                prediction["region_candidate_id"]
                for prediction in data["record_contract"][item["record_id"]]["formal_predictions"]
                if prediction["prediction_id"] == item["base_prediction_id"]
            )
            for item in (data["metadata"][index] for index in selected)
        ),
    }


def development_feasible(summary: dict[str, Any], gate: dict[str, Any]) -> bool:
    folds = summary["per_fold"].values()
    return bool(
        summary["actions"] >= int(gate["minimum_pooled_actions"])
        and sum(item["actions"] > 0 for item in folds) >= int(gate["minimum_folds_with_actions"])
        and sum(item["net"] > 0 for item in folds) >= int(gate["minimum_folds_with_positive_net"])
        and summary["corrected"] > summary["damaged"]
        and summary["action_precision"] >= float(gate["minimum_pooled_action_precision"])
        and summary["formal_correct_preservation"] >= float(gate["minimum_formal_correct_preservation"])
        and summary["net"] > 0
    )


def select_utility(
    probabilities: torch.Tensor,
    data: dict[str, Any],
    utility_contract: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any] | None:
    passing = []
    for lambda_damage in utility_contract["lambda_damage_candidates"]:
        for lambda_neutral in utility_contract["lambda_neutral_candidates"]:
            if float(lambda_damage) <= float(lambda_neutral):
                continue
            winners = group_winners(
                probabilities, data["metadata"], lambda_damage, lambda_neutral
            )
            deltas = quantile_deltas(
                [utility for _, utility in winners],
                utility_contract["delta_quantiles_of_group_max_utility"],
            )
            for delta in deltas:
                selected = selected_indices(winners, delta)
                summary = action_summary(selected, data)
                if development_feasible(summary, gate):
                    positive_folds = sum(
                        item["net"] > 0 for item in summary["per_fold"].values()
                    )
                    key = (
                        positive_folds,
                        summary["net"],
                        summary["action_precision"],
                        -summary["actions"],
                        delta,
                        -float(lambda_damage),
                        -float(lambda_neutral),
                    )
                    passing.append(
                        (
                            key,
                            {
                                "lambda_damage": float(lambda_damage),
                                "lambda_neutral": float(lambda_neutral),
                                "delta": float(delta),
                                "development_metrics": summary,
                            },
                        )
                    )
    return max(passing, key=lambda item: item[0])[1] if passing else None


def apply_frozen_utility(
    probabilities: torch.Tensor,
    data: dict[str, Any],
    selection: dict[str, Any] | None,
) -> tuple[list[int], dict[str, Any]]:
    if selection is None:
        selected = []
    else:
        winners = group_winners(
            probabilities,
            data["metadata"],
            selection["lambda_damage"],
            selection["lambda_neutral"],
        )
        selected = selected_indices(winners, selection["delta"])
    return selected, action_summary(selected, data)


def ranking_diagnostics(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    scores = probabilities[:, 0]
    positives = labels.eq(0)
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = positives[order].float()
    count = int(positives.sum())
    if count:
        tp = ranked.cumsum(0)
        precision = tp / torch.arange(1, len(tp) + 1)
        recall = tp / count
        prior = torch.cat([torch.zeros(1), recall[:-1]])
        auprc = float(((recall - prior) * precision).sum())
    else:
        auprc = 0.0
    precision_at_k = {}
    for k in (5, 10, 20, 50):
        local = ranked[: min(k, len(ranked))]
        precision_at_k[str(k)] = float(local.mean()) if len(local) else 0.0
    return {"auprc": auprc, "precision_at_k": precision_at_k}
