"""Evaluation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from gmner.constants import DEFAULT_LABEL2ID, IGNORE_INDEX


def word_labels_from_subwords(
    subword_labels: List[int],
    word_ids: List[int | None],
    ignore_index: int = IGNORE_INDEX,
) -> List[int]:
    word_to_label: Dict[int, int] = {}
    for label, word_id in zip(subword_labels, word_ids):
        if word_id is None:
            continue
        if word_id in word_to_label:
            continue
        if label == ignore_index:
            continue
        word_to_label[word_id] = int(label)

    if not word_to_label:
        return []

    max_word = max(word_to_label.keys())
    return [word_to_label.get(idx, DEFAULT_LABEL2ID["O"]) for idx in range(max_word + 1)]


def extract_entities_from_word_labels(
    word_labels: List[int],
    tokens: List[str],
    id2label: Dict[int, str],
) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    start = None
    current_type = None
    max_len = min(len(tokens), len(word_labels))

    for idx in range(max_len):
        label = id2label.get(int(word_labels[idx]), "O")
        if label == "O":
            if start is not None:
                entities.append(
                    {
                        "start": start,
                        "end": idx,
                        "type": current_type or "",
                        "text": " ".join(tokens[start:idx]),
                    }
                )
                start = None
                current_type = None
            continue

        if label.startswith("B-"):
            if start is not None:
                entities.append(
                    {
                        "start": start,
                        "end": idx,
                        "type": current_type or "",
                        "text": " ".join(tokens[start:idx]),
                    }
                )
            start = idx
            current_type = label[2:]
            continue

        if label.startswith("I-"):
            ent_type = label[2:]
            if start is None or current_type != ent_type:
                start = idx
                current_type = ent_type
            continue

    if start is not None:
        entities.append(
            {
                "start": start,
                "end": max_len,
                "type": current_type or "",
                "text": " ".join(tokens[start:max_len]),
            }
        )

    return entities


def classification_metrics(preds: Iterable[int], labels: Iterable[int], num_classes: int) -> Dict[str, float]:
    preds = np.asarray(list(preds))
    labels = np.asarray(list(labels))
    valid = labels != IGNORE_INDEX
    preds = preds[valid]
    labels = labels[valid]

    if len(labels) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    accuracy = float((preds == labels).mean())

    f1_scores: List[float] = []
    for cls in range(num_classes):
        tp = np.logical_and(preds == cls, labels == cls).sum()
        fp = np.logical_and(preds == cls, labels != cls).sum()
        fn = np.logical_and(preds != cls, labels == cls).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(float(f1))

    macro_f1 = float(np.mean(f1_scores))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def token_micro_f1(preds: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    for pred_seq, label_seq in zip(preds, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label == IGNORE_INDEX:
                continue
            if pred == label and label != DEFAULT_LABEL2ID["O"]:
                tp += 1
            elif pred != label:
                if pred != DEFAULT_LABEL2ID["O"]:
                    fp += 1
                if label != DEFAULT_LABEL2ID["O"]:
                    fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"token_precision": precision, "token_recall": recall, "token_f1": f1}


def entity_micro_f1(preds: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    predicted = set()
    gold = set()

    for sequence_idx, (pred_seq, label_seq) in enumerate(zip(preds, labels)):
        tokens = [str(idx) for idx in range(max(len(pred_seq), len(label_seq)))]
        for entity in extract_entities_from_word_labels(pred_seq, tokens, id2label):
            predicted.add((sequence_idx, entity["start"], entity["end"], entity["type"]))
        for entity in extract_entities_from_word_labels(label_seq, tokens, id2label):
            gold.add((sequence_idx, entity["start"], entity["end"], entity["type"]))

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"entity_precision": precision, "entity_recall": recall, "entity_f1": f1}


def span_micro_f1(preds: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
    """Compute exact-boundary entity F1 while ignoring the entity type."""

    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    predicted = set()
    gold = set()

    for sequence_idx, (pred_seq, label_seq) in enumerate(zip(preds, labels)):
        tokens = [str(idx) for idx in range(max(len(pred_seq), len(label_seq)))]
        for entity in extract_entities_from_word_labels(pred_seq, tokens, id2label):
            predicted.add((sequence_idx, entity["start"], entity["end"]))
        for entity in extract_entities_from_word_labels(label_seq, tokens, id2label):
            gold.add((sequence_idx, entity["start"], entity["end"]))

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
    }


def grounding_accuracy(preds: List[int], labels: List[int]) -> Dict[str, float]:
    preds_arr = np.asarray(preds)
    labels_arr = np.asarray(labels)
    valid = labels_arr != IGNORE_INDEX
    if valid.sum() == 0:
        return {"grounding_accuracy": 0.0, "grounding_coverage": 0.0}

    correct = (preds_arr[valid] == labels_arr[valid]).sum()
    accuracy = float(correct / valid.sum())
    coverage = float(valid.sum() / len(labels_arr)) if len(labels_arr) > 0 else 0.0
    return {"grounding_accuracy": accuracy, "grounding_coverage": coverage}
