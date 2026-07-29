"""Read-only P0 diagnostics for the hierarchical joint Stage1 protocol."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from gmner.constants import (
    DEFAULT_LABEL2ID,
    ENTITY_TYPE2ID,
    ID2ENTITY_TYPE,
    normalize_bio_label,
)
from gmner.data.tokenization import encode_words_with_alignment
from gmner.utils.metrics import extract_entities_from_word_labels


S3_AUDIT_VERSION = 1
_ID2LABEL = {value: key for key, value in DEFAULT_LABEL2ID.items()}


def ensure_s3_audit_split(split: str) -> str:
    """Reject any P0 invocation outside the pre-registered Train/Dev scope."""

    normalized = str(split).strip().lower()
    if normalized not in {"train", "dev"}:
        raise ValueError("S3 P0 audits support only train and dev.")
    return normalized


def read_s3_source_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL or Twitter10000 CoNLL text without materializing a cache."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".txt":
        records: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    records = []
    tokens: list[str] = []
    tags: list[str] = []
    fine_tags: list[str] = []
    image_id: str | None = None

    def flush() -> None:
        if not image_id or not tokens:
            return
        item: dict[str, Any] = {
            "id": len(records),
            "tokens": list(tokens),
            "ner_tags": [
                DEFAULT_LABEL2ID.get(
                    normalize_bio_label(tag),
                    DEFAULT_LABEL2ID["O"],
                )
                for tag in tags
            ],
            "image": f"{image_id}.jpg",
        }
        if fine_tags and len(fine_tags) == len(tokens):
            item["fine_ner_tags"] = list(fine_tags)
        records.append(item)

    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                flush()
                tokens.clear()
                tags.clear()
                fine_tags.clear()
                image_id = None
            elif line.startswith("IMGID:"):
                if image_id and tokens:
                    flush()
                    tokens.clear()
                    tags.clear()
                    fine_tags.clear()
                image_id = line.split("IMGID:", 1)[1].strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[1] if len(parts) >= 3 else parts[-1])
                    if len(parts) >= 3:
                        fine_tags.append(parts[2])
    flush()
    return records


def _span(item: dict[str, Any]) -> tuple[int, int]:
    value = item.get("span")
    if value is None:
        value = (item.get("start"), item.get("end"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid entity span: {value!r}.")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start:
        raise ValueError(f"Invalid half-open entity span: {(start, end)!r}.")
    return start, end


def _type_id(item: dict[str, Any]) -> int:
    value = item.get("type_id")
    if value is not None:
        return int(value)
    name = str(item.get("type", "O")).upper()
    return int(ENTITY_TYPE2ID.get(name, ENTITY_TYPE2ID["O"]))


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = _overlap(left, right)
    union = (left[1] - left[0]) + (right[1] - right[0]) - intersection
    return intersection / max(union, 1)


def _length_bucket(span: tuple[int, int]) -> str:
    length = span[1] - span[0]
    return str(length) if length <= 3 else "4+"


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _counter_dict(value: Counter) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(value.items())}


def audit_boundary_type_errors(
    records: Iterable[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    """Decompose formal Stage1 boundary and coarse-type errors in word space."""

    split = ensure_s3_audit_split(split)
    counts = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_length: dict[str, Counter] = defaultdict(Counter)
    by_scene: dict[str, Counter] = defaultdict(Counter)
    start_shifts = Counter()
    end_shifts = Counter()
    overlap_iou_sum = 0.0
    record_count = 0

    for record in records:
        metadata = dict(record.get("metadata") or {})
        gold = list(metadata.get("gold_entities") or [])
        predictions = list(metadata.get("stage1_predictions") or [])
        record_count += 1
        counts["gold"] += len(gold)
        counts["predicted"] += len(predictions)
        scene = "single" if len(gold) <= 1 else "multi"
        gold_spans = {_span(item) for item in gold}

        prediction_by_span: dict[
            tuple[int, int], list[tuple[int, dict[str, Any]]]
        ] = defaultdict(list)
        for index, prediction in enumerate(predictions):
            prediction_by_span[_span(prediction)].append((index, prediction))

        matched_prediction_indices: set[int] = set()
        for gold_item in gold:
            gold_span = _span(gold_item)
            gold_type = _type_id(gold_item)
            type_name = ID2ENTITY_TYPE.get(gold_type, "O")
            length_bucket = _length_bucket(gold_span)
            exact = prediction_by_span.get(gold_span, [])
            if exact:
                typed = next(
                    (
                        item
                        for item in exact
                        if _type_id(item[1]) == gold_type
                    ),
                    None,
                )
                category = (
                    "boundary_and_type_correct"
                    if typed is not None
                    else "boundary_correct_type_wrong"
                )
                chosen_index, _ = typed or exact[0]
                matched_prediction_indices.add(chosen_index)
            else:
                overlaps = [
                    (index, prediction)
                    for index, prediction in enumerate(predictions)
                    if index not in matched_prediction_indices
                    and _overlap(gold_span, _span(prediction)) > 0
                ]
                if overlaps:
                    index, chosen = max(
                        overlaps,
                        key=lambda item: (
                            _iou(gold_span, _span(item[1])),
                            -abs(_span(item[1])[0] - gold_span[0]),
                            -abs(_span(item[1])[1] - gold_span[1]),
                        ),
                    )
                    matched_prediction_indices.add(index)
                    chosen_span = _span(chosen)
                    start_shifts[chosen_span[0] - gold_span[0]] += 1
                    end_shifts[chosen_span[1] - gold_span[1]] += 1
                    overlap_iou_sum += _iou(gold_span, chosen_span)
                    category = "overlapping_boundary_error"
                else:
                    category = "completely_missing"

            counts[category] += 1
            by_type[type_name][category] += 1
            by_type[type_name]["gold"] += 1
            by_length[length_bucket][category] += 1
            by_length[length_bucket]["gold"] += 1
            by_scene[scene][category] += 1
            by_scene[scene]["gold"] += 1

        counts["extra_predictions"] += sum(
            1
            for index, prediction in enumerate(predictions)
            if index not in matched_prediction_indices
            and _span(prediction) not in gold_spans
        )

    gold_count = int(counts["gold"])
    exact_boundary = int(
        counts["boundary_and_type_correct"]
        + counts["boundary_correct_type_wrong"]
    )
    return {
        "kind": "s3_p0_boundary_type_audit",
        "format_version": S3_AUDIT_VERSION,
        "split": split,
        "records": record_count,
        "counts": _counter_dict(counts),
        "rates_over_gold": {
            "exact_boundary": _ratio(exact_boundary, gold_count),
            "exact_boundary_and_type": _ratio(
                counts["boundary_and_type_correct"], gold_count
            ),
            "type_error_given_exact_boundary": _ratio(
                counts["boundary_correct_type_wrong"], exact_boundary
            ),
            "overlapping_boundary_error": _ratio(
                counts["overlapping_boundary_error"], gold_count
            ),
            "completely_missing": _ratio(
                counts["completely_missing"], gold_count
            ),
        },
        "overlap_error": {
            "mean_span_iou": _ratio(
                overlap_iou_sum, counts["overlapping_boundary_error"]
            ),
            "start_shift_histogram": _counter_dict(start_shifts),
            "end_shift_histogram": _counter_dict(end_shifts),
        },
        "by_type": {
            key: _counter_dict(value) for key, value in sorted(by_type.items())
        },
        "by_span_length": {
            key: _counter_dict(value)
            for key, value in sorted(by_length.items())
        },
        "by_record_entity_count": {
            key: _counter_dict(value)
            for key, value in sorted(by_scene.items())
        },
        "test_accessed": False,
    }


def audit_candidate_actionability(
    records: Iterable[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    """Measure exact/typed candidate coverage without fitting a selector."""

    split = ensure_s3_audit_split(split)
    counts = Counter()
    source_exact = Counter()
    source_typed = Counter()
    exact_ranks: list[int] = []
    typed_ranks: list[int] = []
    record_count = 0

    for record in records:
        metadata = dict(record.get("metadata") or {})
        gold = list(metadata.get("gold_entities") or [])
        candidates = record["span_candidates"]
        span_mask = record["span_mask"].bool()
        type_candidates = record["type_candidates"]
        formal_mask = record["formal_candidate_mask"].bool()
        sources = list(metadata.get("candidate_sources") or [])
        valid_indices = (
            span_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        )
        formal_spans = {
            tuple(int(value) for value in candidates[index].tolist())
            for index in valid_indices
            if bool(formal_mask[index].item())
        }
        record_count += 1
        counts["gold"] += len(gold)

        for gold_item in gold:
            gold_span = _span(gold_item)
            gold_type = _type_id(gold_item)
            exact_rows = [
                index
                for index in valid_indices
                if tuple(int(value) for value in candidates[index].tolist())
                == gold_span
            ]
            if not exact_rows:
                counts["candidate_missing"] += 1
                continue

            counts["exact_candidate_covered"] += 1
            first = exact_rows[0]
            exact_ranks.append(first + 1)
            source = (
                str(sources[first])
                if first < len(sources)
                else str(int(record["span_source_ids"][first].item()))
            )
            source_exact[source] += 1
            typed_rows = [
                index
                for index in exact_rows
                if bool(type_candidates[index].eq(gold_type).any().item())
            ]
            if typed_rows:
                counts["typed_exact_candidate_covered"] += 1
                typed_first = typed_rows[0]
                typed_ranks.append(typed_first + 1)
                typed_source = (
                    str(sources[typed_first])
                    if typed_first < len(sources)
                    else str(
                        int(record["span_source_ids"][typed_first].item())
                    )
                )
                source_typed[typed_source] += 1
            else:
                counts["type_missing_given_exact_span"] += 1

            if gold_span in formal_spans:
                counts["formal_exact_span"] += 1
            else:
                counts["recoverable_nonformal_exact_span"] += 1
                if typed_rows:
                    counts["recoverable_nonformal_typed_span"] += 1

    gold_count = int(counts["gold"])
    return {
        "kind": "s3_p0_candidate_actionability_audit",
        "format_version": S3_AUDIT_VERSION,
        "split": split,
        "records": record_count,
        "counts": _counter_dict(counts),
        "coverage": {
            "exact_candidate": _ratio(
                counts["exact_candidate_covered"], gold_count
            ),
            "typed_exact_candidate": _ratio(
                counts["typed_exact_candidate_covered"], gold_count
            ),
            "formal_exact_span": _ratio(
                counts["formal_exact_span"], gold_count
            ),
            "recoverable_nonformal_exact_span": _ratio(
                counts["recoverable_nonformal_exact_span"], gold_count
            ),
            "recoverable_nonformal_typed_span": _ratio(
                counts["recoverable_nonformal_typed_span"], gold_count
            ),
        },
        "candidate_rank": {
            "exact_mean": _ratio(sum(exact_ranks), len(exact_ranks)),
            "typed_exact_mean": _ratio(
                sum(typed_ranks), len(typed_ranks)
            ),
            "exact_max": max(exact_ranks, default=0),
            "typed_exact_max": max(typed_ranks, default=0),
        },
        "source": {
            "exact": _counter_dict(source_exact),
            "typed_exact": _counter_dict(source_typed),
        },
        "test_accessed": False,
    }


def _record_gold_entities(record: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = list(record.get("tokens") or [])
    tags = list(record.get("ner_tags") or [])
    normalized_ids: list[int] = []
    for tag in tags:
        if isinstance(tag, int):
            normalized_ids.append(int(tag))
        else:
            normalized_ids.append(
                int(DEFAULT_LABEL2ID.get(normalize_bio_label(str(tag)), 0))
            )
    return extract_entities_from_word_labels(
        normalized_ids,
        tokens,
        _ID2LABEL,
    )


def audit_truncation(
    records: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    max_length: int,
    split: str,
) -> dict[str, Any]:
    """Report partially and fully truncated words/entities at max_length."""

    split = ensure_s3_audit_split(split)
    counts = Counter()
    by_type = Counter()
    record_count = 0
    for record in records:
        tokens = list(record.get("tokens") or [])
        _, word_ids = encode_words_with_alignment(
            tokenizer,
            tokens,
            max_length=max_length,
        )
        encoded_word_ids = {
            int(word_id) for word_id in word_ids if word_id is not None
        }
        piece_count = Counter(
            int(word_id) for word_id in word_ids if word_id is not None
        )
        full_piece_count: dict[int, int] = {}
        for word_index, token in enumerate(tokens):
            _, single_word_ids = encode_words_with_alignment(
                tokenizer,
                [token],
                max_length=max(int(max_length), 4096),
            )
            full_piece_count[word_index] = sum(
                word_id is not None for word_id in single_word_ids
            )
        truncated_words = {
            index
            for index in range(len(tokens))
            if index not in encoded_word_ids
            or piece_count[index] < full_piece_count[index]
        }
        counts["words"] += len(tokens)
        counts["truncated_words"] += len(truncated_words)
        entities = _record_gold_entities(record)
        counts["gold_entities"] += len(entities)
        for entity in entities:
            span = (int(entity["start"]), int(entity["end"]))
            words = set(range(span[0], span[1]))
            present = words & encoded_word_ids
            if not present:
                category = "fully_truncated_entities"
            elif words & truncated_words or present != words:
                category = "partially_truncated_entities"
            else:
                category = "fully_encoded_entities"
            counts[category] += 1
            if category != "fully_encoded_entities":
                by_type[str(entity.get("type", "O"))] += 1
        record_count += 1

    return {
        "kind": "s3_p0_truncation_audit",
        "format_version": S3_AUDIT_VERSION,
        "split": split,
        "records": record_count,
        "max_length": int(max_length),
        "counts": _counter_dict(counts),
        "rates": {
            "truncated_words": _ratio(
                counts["truncated_words"], counts["words"]
            ),
            "partially_truncated_entities": _ratio(
                counts["partially_truncated_entities"],
                counts["gold_entities"],
            ),
            "fully_truncated_entities": _ratio(
                counts["fully_truncated_entities"],
                counts["gold_entities"],
            ),
        },
        "truncated_entities_by_type": _counter_dict(by_type),
        "test_accessed": False,
    }
