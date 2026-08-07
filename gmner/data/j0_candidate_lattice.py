"""Gold-free typed-span lattice construction and exact J0 Oracle helpers."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Iterable


TYPE_ORDER = ("LOC", "PER", "ORG", "OTHER")
SOURCE_PRIORITY = {
    "formal": 0,
    "stage1": 1,
    "viterbi": 2,
    "kbest": 3,
    "perturbation": 4,
    "unknown": 9,
}
DERIVED_FLOAT_DECIMALS = 12


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_identity(prefix: str, payload: dict[str, Any]) -> str:
    def reject_float(value: Any) -> None:
        if isinstance(value, float):
            raise TypeError("Floating values are forbidden in stable identities.")
        if isinstance(value, dict):
            for item in value.values():
                reject_float(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                reject_float(item)

    reject_float(payload)
    return f"{prefix}:{canonical_sha256(payload)}"


def finite(value: Any, trail: str = "value") -> int:
    count = 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {trail}: {value}")
        return 1
    if isinstance(value, dict):
        for key, item in value.items():
            count += finite(item, f"{trail}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            count += finite(item, f"{trail}[{index}]")
    return count


def contains_gold_or_supervision(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized == "supervision" or normalized.startswith("gold"):
                return True
            if contains_gold_or_supervision(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(contains_gold_or_supervision(item) for item in value)
    return False


def span_tuple(value: dict[str, Any] | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, dict):
        start, end = int(value["start"]), int(value["end"])
        if value.get("space") != "word_half_open":
            raise ValueError("J0 requires word-space half-open spans.")
    else:
        start, end = (int(item) for item in value)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid word-space span: {(start, end)}")
    return start, end


def span_object(span: tuple[int, int]) -> dict[str, Any]:
    start, end = span_tuple(span)
    return {"start": start, "end": end, "space": "word_half_open"}


def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0])


@lru_cache(maxsize=131072)
def _log_softmax_cached(values: tuple[float, ...]) -> tuple[float, ...]:
    if len(values) != len(TYPE_ORDER):
        raise ValueError("Typed-span evidence must contain four logits.")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Non-finite type logit.")
    with localcontext() as context:
        context.prec = 50
        decimal_values = [Decimal(repr(float(value))) for value in values]
        maximum = max(decimal_values)
        denominator = maximum + sum(
            (value - maximum).exp() for value in decimal_values
        ).ln()
        quantum = Decimal(1).scaleb(-DERIVED_FLOAT_DECIMALS)
        return tuple(
            float((value - denominator).quantize(quantum, rounding=ROUND_HALF_EVEN))
            for value in decimal_values
        )


def _log_softmax(values: list[float]) -> list[float]:
    return list(_log_softmax_cached(tuple(float(value) for value in values)))


def _semantic_candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        *span_tuple(candidate["span"]),
        int(candidate["type_id"]),
        str(candidate["source"]),
    )


def _validate_source_row(row: dict[str, Any]) -> dict[str, int | bool]:
    if row.get("kind") != "final_chain_oof_record":
        raise ValueError("Unexpected OOF row kind.")
    if row.get("heldout") is not True or row.get("test_accessed") is not False:
        raise PermissionError("J0 requires held-out Train rows with Test locked.")
    if contains_gold_or_supervision(row):
        raise PermissionError("Gold/supervision appeared in a J0 build input.")
    for prediction in row["formal_predictions"]:
        if prediction["observable_features"].get("type_order") != list(TYPE_ORDER):
            raise ValueError("Formal coarse-type order changed.")
        if not 0 <= int(prediction["type_id"]) < len(TYPE_ORDER):
            raise ValueError("Invalid formal coarse type.")
        span_tuple(prediction["span"])
    r16 = sorted(
        _semantic_candidate_key(item)
        for item in row["r16_candidates"]["span_candidates"]
    )
    r36 = sorted(
        _semantic_candidate_key(item)
        for item in row["r36_candidates"]["span_candidates"]
    )
    r16_counter: dict[tuple[Any, ...], int] = {}
    r36_counter: dict[tuple[Any, ...], int] = {}
    for key in r16:
        r16_counter[key] = r16_counter.get(key, 0) + 1
    for key in r36:
        r36_counter[key] = r36_counter.get(key, 0) + 1
    r16_only = sum(
        max(0, count - r36_counter.get(key, 0))
        for key, count in r16_counter.items()
    )
    r36_only = sum(
        max(0, count - r16_counter.get(key, 0))
        for key, count in r36_counter.items()
    )
    return {
        "r16_r36_semantic_match": r16_only == 0 and r36_only == 0,
        "r16_only_candidates": r16_only,
        "r36_only_candidates": r36_only,
    }


def _origin_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    logits = [float(value) for value in candidate["scores"]["type_logits"]]
    if len(logits) != len(TYPE_ORDER):
        raise ValueError("Candidate coarse-type evidence changed width.")
    return {
        "origin_kind": "r36_span_candidate",
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_source": str(candidate["source"]),
        "region_candidate_id": str(candidate["region_candidate_id"]),
        "span_base_score": float(candidate["scores"]["span_base_score"]),
        "type_logits": logits,
    }


def _origin_from_formal(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "origin_kind": "formal_prediction",
        "candidate_id": str(prediction["prediction_id"]),
        "candidate_source": "formal",
        "region_candidate_id": str(prediction["region_candidate_id"]),
        "span_base_score": float(
            prediction["observable_features"]["span_base_score"]
        ),
        "type_logits": [float(value) for value in prediction["type_logits"]],
    }


def _raw_hypothesis(
    *,
    record_id: str,
    group_id: str,
    operation: str,
    span: tuple[int, int],
    type_id: int,
    origin: dict[str, Any],
) -> dict[str, Any]:
    log_probability = _log_softmax(origin["type_logits"])[int(type_id)]
    with localcontext() as context:
        context.prec = 50
        quantum = Decimal(1).scaleb(-DERIVED_FLOAT_DECIMALS)
        typed_score = float(
            (
                Decimal(repr(float(origin["span_base_score"])))
                + Decimal(repr(log_probability))
            ).quantize(quantum, rounding=ROUND_HALF_EVEN)
        )
    source = str(origin["candidate_source"])
    return {
        "hypothesis_id": stable_identity(
            "hypothesis",
            {
                "kind": "typed_span_hypothesis",
                "record_id": record_id,
                "group_id": group_id,
                "operation": operation,
                "span": list(span),
                "type_id": int(type_id),
            },
        ),
        "operation": operation,
        "span": span_object(span),
        "type_id": int(type_id),
        "typed_score": typed_score,
        "span_base_score": float(origin["span_base_score"]),
        "type_log_probability": log_probability,
        "primary_source": source,
        "source_priority": int(SOURCE_PRIORITY.get(source, SOURCE_PRIORITY["unknown"])),
        "origins": [origin],
    }


def _hypothesis_rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
    span = span_tuple(item["span"])
    return (
        -float(item["typed_score"]),
        int(item["source_priority"]),
        span[0],
        span[1],
        int(item["type_id"]),
        str(item["hypothesis_id"]),
    )


def _deduplicate(
    raw: list[dict[str, Any]], *, control_key: tuple[int, int, int] | None
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for item in raw:
        span = span_tuple(item["span"])
        grouped.setdefault((*span, int(item["type_id"])), []).append(item)
    output = []
    for key, candidates in grouped.items():
        if key == control_key:
            continue
        ordered = sorted(candidates, key=_hypothesis_rank_key)
        winner = dict(ordered[0])
        origins = {
            canonical_sha256(origin): origin
            for candidate in candidates
            for origin in candidate["origins"]
        }
        winner["origins"] = [origins[key] for key in sorted(origins)]
        output.append(winner)
    output.sort(key=_hypothesis_rank_key)
    for rank, item in enumerate(output, start=1):
        item["rank"] = rank
    return output


def _connected_span_components(
    candidates: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not candidates:
        return []
    spans = [span_tuple(item["span"]) for item in candidates]
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if overlaps(spans[left], spans[right]):
                union(left, right)
    components: dict[int, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        components.setdefault(find(index), []).append(candidate)
    return sorted(
        components.values(),
        key=lambda items: sorted(span_tuple(item["span"]) for item in items),
    )


def build_lattice_record(row: dict[str, Any]) -> dict[str, Any]:
    """Build one deterministic typed-span lattice without reading supervision."""

    source_audit = _validate_source_row(row)
    record_id = str(row["record_id"])
    formal_predictions = list(row["formal_predictions"])
    formal_spans = [span_tuple(item["span"]) for item in formal_predictions]
    proposals = list(row["r36_candidates"]["span_candidates"])
    groups: list[dict[str, Any]] = []
    raw_total = 0

    for prediction in formal_predictions:
        base_span = span_tuple(prediction["span"])
        base_type = int(prediction["type_id"])
        group_id = stable_identity(
            "group",
            {
                "kind": "replacement_group",
                "record_id": record_id,
                "base_prediction_id": str(prediction["prediction_id"]),
            },
        )
        formal_origin = _origin_from_formal(prediction)
        control = _raw_hypothesis(
            record_id=record_id,
            group_id=group_id,
            operation="KEEP",
            span=base_span,
            type_id=base_type,
            origin=formal_origin,
        )
        raw: list[dict[str, Any]] = []
        for type_id in range(len(TYPE_ORDER)):
            raw.append(
                _raw_hypothesis(
                    record_id=record_id,
                    group_id=group_id,
                    operation="REPLACE",
                    span=base_span,
                    type_id=type_id,
                    origin=formal_origin,
                )
            )
        for candidate in proposals:
            candidate_span = span_tuple(candidate["span"])
            if not overlaps(base_span, candidate_span):
                continue
            origin = _origin_from_candidate(candidate)
            for type_id in range(len(TYPE_ORDER)):
                raw.append(
                    _raw_hypothesis(
                        record_id=record_id,
                        group_id=group_id,
                        operation="REPLACE",
                        span=candidate_span,
                        type_id=type_id,
                        origin=origin,
                    )
                )
        raw_total += len(raw)
        alternatives = _deduplicate(
            raw, control_key=(*base_span, base_type)
        )
        groups.append(
            {
                "group_id": group_id,
                "group_kind": "replacement",
                "base_prediction_id": str(prediction["prediction_id"]),
                "control": control,
                "raw_alternative_count": len(raw),
                "alternatives": alternatives,
            }
        )

    add_proposals = [
        candidate
        for candidate in proposals
        if not any(
            overlaps(span_tuple(candidate["span"]), formal_span)
            for formal_span in formal_spans
        )
    ]
    for component in _connected_span_components(add_proposals):
        component_spans = sorted({span_tuple(item["span"]) for item in component})
        group_id = stable_identity(
            "group",
            {
                "kind": "addition_group",
                "record_id": record_id,
                "component_spans": [list(span) for span in component_spans],
            },
        )
        control = {
            "hypothesis_id": stable_identity(
                "hypothesis",
                {
                    "kind": "typed_span_hypothesis",
                    "record_id": record_id,
                    "group_id": group_id,
                    "operation": "NONE",
                },
            ),
            "operation": "NONE",
            "span": None,
            "type_id": None,
            "typed_score": 0.0,
            "span_base_score": 0.0,
            "type_log_probability": 0.0,
            "primary_source": "formal",
            "source_priority": SOURCE_PRIORITY["formal"],
            "origins": [],
        }
        raw = []
        for candidate in component:
            candidate_span = span_tuple(candidate["span"])
            origin = _origin_from_candidate(candidate)
            for type_id in range(len(TYPE_ORDER)):
                raw.append(
                    _raw_hypothesis(
                        record_id=record_id,
                        group_id=group_id,
                        operation="ADD",
                        span=candidate_span,
                        type_id=type_id,
                        origin=origin,
                    )
                )
        raw_total += len(raw)
        groups.append(
            {
                "group_id": group_id,
                "group_kind": "addition",
                "base_prediction_id": None,
                "control": control,
                "raw_alternative_count": len(raw),
                "alternatives": _deduplicate(raw, control_key=None),
            }
        )

    groups.sort(
        key=lambda group: (
            0 if group["group_kind"] == "replacement" else 1,
            str(group["group_id"]),
        )
    )
    lattice = {
        "kind": "j0_gold_free_typed_span_lattice_record",
        "format_version": 1,
        "record_id": record_id,
        "fold_id": int(row["fold_id"]),
        "heldout": True,
        "dev_accessed": False,
        "test_accessed": False,
        "source_row_sha256": canonical_sha256(row),
        "type_order": list(TYPE_ORDER),
        "derived_float_decimals": DERIVED_FLOAT_DECIMALS,
        "candidate_source_audit": source_audit,
        "groups": groups,
        "counts": {
            "formal_predictions": len(formal_predictions),
            "replacement_groups": sum(
                group["group_kind"] == "replacement" for group in groups
            ),
            "addition_groups": sum(
                group["group_kind"] == "addition" for group in groups
            ),
            "raw_alternatives": raw_total,
            "deduplicated_alternatives": sum(
                len(group["alternatives"]) for group in groups
            ),
        },
    }
    finite(lattice)
    if contains_gold_or_supervision(lattice):
        raise AssertionError("J0 lattice construction leaked supervision.")
    return lattice


def budget_groups(
    lattice: dict[str, Any],
    *,
    top_k: int | None,
    max_record_alternatives: int | None,
) -> list[dict[str, Any]]:
    groups = []
    for group in lattice["groups"]:
        alternatives = list(group["alternatives"])
        if top_k is not None:
            alternatives = alternatives[: int(top_k)]
        groups.append({**group, "alternatives": alternatives})
    if max_record_alternatives is not None:
        ranked = sorted(
            (
                item
                for group in groups
                for item in group["alternatives"]
            ),
            key=_hypothesis_rank_key,
        )
        allowed = {
            item["hypothesis_id"]
            for item in ranked[: int(max_record_alternatives)]
        }
        groups = [
            {
                **group,
                "alternatives": [
                    item
                    for item in group["alternatives"]
                    if item["hypothesis_id"] in allowed
                ],
            }
            for group in groups
        ]
    return groups


def _typed_key(item: dict[str, Any]) -> tuple[int, int, int] | None:
    if item.get("span") is None:
        return None
    span = span_tuple(item["span"])
    return *span, int(item["type_id"])


def _unconstrained_oracle(
    groups: list[dict[str, Any]],
    gold: set[tuple[int, int, int]],
    *,
    max_additions: int | None,
) -> tuple[int, int, tuple[str, ...]]:
    gold_order = {key: index for index, key in enumerate(sorted(gold))}
    edges: list[tuple[str, list[tuple[int, str]]]] = []
    for group in groups:
        choices = [group["control"], *group["alternatives"]]
        candidates = {
            gold_order[key]: str(item["hypothesis_id"])
            for item in choices
            if (key := _typed_key(item)) in gold_order
        }
        edges.append(
            (
                str(group["group_kind"]),
                sorted((index, candidate_id) for index, candidate_id in candidates.items()),
            )
        )

    @lru_cache(maxsize=None)
    def solve(
        index: int, used_gold: int, additions: int
    ) -> tuple[int, int, tuple[str, ...]]:
        if index == len(edges):
            return 0, additions, ()
        kind, candidates = edges[index]
        best = solve(index + 1, used_gold, additions)
        for gold_index, hypothesis_id in candidates:
            if used_gold & (1 << gold_index):
                continue
            next_additions = additions + (kind == "addition")
            if max_additions is not None and next_additions > max_additions:
                continue
            tail_correct, tail_additions, tail_ids = solve(
                index + 1, used_gold | (1 << gold_index), next_additions
            )
            candidate = (tail_correct + 1, tail_additions, (hypothesis_id, *tail_ids))
            candidate_ids = tuple(sorted(candidate[2]))
            best_ids = tuple(sorted(best[2]))
            if (
                candidate[0] > best[0]
                or (candidate[0] == best[0] and candidate[1] < best[1])
                or (
                    candidate[:2] == best[:2]
                    and candidate_ids < best_ids
                )
            ):
                best = candidate
        return best

    return solve(0, 0, 0)


def _collapse_choices(
    choices: list[dict[str, Any]], gold: set[tuple[int, int, int]]
) -> list[dict[str, Any]]:
    by_span: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in choices:
        if item.get("span") is None:
            continue
        by_span.setdefault(span_tuple(item["span"]), []).append(item)
    collapsed = []
    for span, items in by_span.items():
        correct = [item for item in items if _typed_key(item) in gold]
        pool = correct or items
        winner = sorted(pool, key=_hypothesis_rank_key)[0]
        collapsed.append(winner)
    return sorted(
        collapsed,
        key=lambda item: (
            0 if _typed_key(item) in gold else 1,
            *_hypothesis_rank_key(item),
        ),
    )


def _addition_oracle(
    groups: list[dict[str, Any]],
    gold: set[tuple[int, int, int]],
    selected_spans: tuple[tuple[int, int], ...],
    *,
    max_additions: int | None,
) -> tuple[int, tuple[str, ...]]:
    gold_order = {key: index for index, key in enumerate(sorted(gold))}
    edges: list[list[tuple[int, str]]] = []
    for group in groups:
        candidates: dict[int, str] = {}
        for item in group["alternatives"]:
            key = _typed_key(item)
            if key not in gold:
                continue
            span = key[:2]
            if any(overlaps(span, selected) for selected in selected_spans):
                continue
            gold_index = gold_order[key]
            current = candidates.get(gold_index)
            if current is None or str(item["hypothesis_id"]) < current:
                candidates[gold_index] = str(item["hypothesis_id"])
        edges.append(sorted(candidates.items()))

    @lru_cache(maxsize=None)
    def solve(
        index: int, used_gold: int, selected_count: int
    ) -> tuple[int, tuple[str, ...]]:
        if index == len(edges):
            return 0, ()
        best = solve(index + 1, used_gold, selected_count)
        for gold_index, hypothesis_id in edges[index]:
            if used_gold & (1 << gold_index):
                continue
            if max_additions is not None and selected_count >= max_additions:
                continue
            tail_count, tail_ids = solve(
                index + 1, used_gold | (1 << gold_index), selected_count + 1
            )
            candidate = tail_count + 1, tuple(sorted((hypothesis_id, *tail_ids)))
            if candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] < best[1]
            ):
                best = candidate
        return best

    return solve(0, 0, 0)


def _constrained_oracle(
    groups: list[dict[str, Any]],
    gold: set[tuple[int, int, int]],
    *,
    max_additions: int | None,
) -> tuple[int, int, tuple[str, ...]]:
    replacement = [group for group in groups if group["group_kind"] == "replacement"]
    additions = [group for group in groups if group["group_kind"] == "addition"]
    choice_groups = [
        _collapse_choices([group["control"], *group["alternatives"]], gold)
        for group in replacement
    ]
    order = sorted(range(len(choice_groups)), key=lambda index: len(choice_groups[index]))
    ordered_groups = [choice_groups[index] for index in order]
    best: tuple[int, int, tuple[str, ...]] = (-1, 0, ())

    def visit(
        index: int,
        selected_spans: tuple[tuple[int, int], ...],
        correct_keys: frozenset[tuple[int, int, int]],
        selected_ids: tuple[str, ...],
    ) -> None:
        nonlocal best
        remaining = len(ordered_groups) - index
        addition_bound = len(additions) if max_additions is None else max_additions
        if len(correct_keys) + remaining + addition_bound < best[0]:
            return
        if index == len(ordered_groups):
            add_correct, add_ids = _addition_oracle(
                additions,
                gold,
                selected_spans,
                max_additions=max_additions,
            )
            candidate = (
                len(correct_keys) + add_correct,
                add_correct,
                tuple(sorted((*selected_ids, *add_ids))),
            )
            if (
                candidate[0] > best[0]
                or (candidate[0] == best[0] and candidate[1] < best[1])
                or (
                    candidate[:2] == best[:2]
                    and (not best[2] or candidate[2] < best[2])
                )
            ):
                best = candidate
            return
        for item in ordered_groups[index]:
            span = span_tuple(item["span"])
            if any(overlaps(span, selected) for selected in selected_spans):
                continue
            key = _typed_key(item)
            visit(
                index + 1,
                (*selected_spans, span),
                correct_keys | ({key} if key in gold else set()),
                (*selected_ids, str(item["hypothesis_id"])),
            )

    visit(0, (), frozenset(), ())
    if best[0] < 0:
        raise RuntimeError("No feasible constrained candidate set; KEEP should be feasible.")
    return best


def evaluate_oracle_stage(
    lattice: dict[str, Any],
    gold_entities: Iterable[tuple[int, int, int]],
    *,
    top_k: int | None,
    enforce_nonoverlap: bool,
    max_record_alternatives: int | None = None,
    max_additions: int | None = None,
) -> dict[str, Any]:
    gold = set(tuple(int(value) for value in item) for item in gold_entities)
    groups = budget_groups(
        lattice,
        top_k=top_k,
        max_record_alternatives=max_record_alternatives,
    )
    if enforce_nonoverlap:
        correct, additions, selected = _constrained_oracle(
            groups, gold, max_additions=max_additions
        )
    else:
        correct, additions, selected = _unconstrained_oracle(
            groups, gold, max_additions=max_additions
        )
    base_predictions = sum(group["group_kind"] == "replacement" for group in groups)
    predicted = base_predictions + additions
    return {
        "correct": int(correct),
        "predicted": int(predicted),
        "gold": len(gold),
        "additions": int(additions),
        "selected_hypothesis_ids": list(selected),
    }


def baseline_result(
    lattice: dict[str, Any], gold_entities: Iterable[tuple[int, int, int]]
) -> dict[str, int]:
    gold = set(tuple(int(value) for value in item) for item in gold_entities)
    controls = [
        group["control"]
        for group in lattice["groups"]
        if group["group_kind"] == "replacement"
    ]
    predicted_keys = [_typed_key(item) for item in controls]
    predicted_spans = {key[:2] for key in predicted_keys if key is not None}
    gold_spans = {key[:2] for key in gold}
    return {
        "correct": sum(key in gold for key in predicted_keys),
        "span_correct": len(predicted_spans & gold_spans),
        "predicted": len(predicted_keys),
        "gold": len(gold),
    }


def oracle_action_breakdown(
    lattice: dict[str, Any],
    gold_entities: Iterable[tuple[int, int, int]],
    selected_hypothesis_ids: Iterable[str],
) -> dict[str, int]:
    """Describe a constrained Oracle set without changing its decisions."""

    gold = set(tuple(int(value) for value in item) for item in gold_entities)
    selected = set(str(value) for value in selected_hypothesis_ids)
    counts: dict[str, int] = {
        "keep_selected": 0,
        "keep_correct": 0,
        "replacement_selected": 0,
        "replacement_type_only": 0,
        "replacement_boundary_only": 0,
        "replacement_boundary_and_type": 0,
        "replacement_corrected": 0,
        "replacement_damaged": 0,
        "replacement_neutral": 0,
        "add_selected": 0,
        "add_correct": 0,
    }
    for group in lattice["groups"]:
        choices = [group["control"], *group["alternatives"]]
        chosen = [item for item in choices if item["hypothesis_id"] in selected]
        if group["group_kind"] == "replacement":
            if len(chosen) != 1:
                raise RuntimeError("A constrained replacement group must select exactly one item.")
            item = chosen[0]
            control = group["control"]
            base_correct = _typed_key(control) in gold
            selected_correct = _typed_key(item) in gold
            if item["operation"] == "KEEP":
                counts["keep_selected"] += 1
                counts["keep_correct"] += int(base_correct)
                continue
            counts["replacement_selected"] += 1
            base_span = span_tuple(control["span"])
            selected_span = span_tuple(item["span"])
            same_span = base_span == selected_span
            same_type = int(control["type_id"]) == int(item["type_id"])
            if same_span and not same_type:
                counts["replacement_type_only"] += 1
            elif not same_span and same_type:
                counts["replacement_boundary_only"] += 1
            elif not same_span and not same_type:
                counts["replacement_boundary_and_type"] += 1
            else:
                raise RuntimeError("A REPLACE hypothesis duplicated KEEP semantics.")
            if selected_correct and not base_correct:
                counts["replacement_corrected"] += 1
            elif base_correct and not selected_correct:
                counts["replacement_damaged"] += 1
            else:
                counts["replacement_neutral"] += 1
            source_key = f"replacement_source_{item['primary_source']}"
            counts[source_key] = counts.get(source_key, 0) + 1
        else:
            if len(chosen) > 1:
                raise RuntimeError("An addition group selected more than one hypothesis.")
            if not chosen or chosen[0]["operation"] == "NONE":
                continue
            item = chosen[0]
            counts["add_selected"] += 1
            counts["add_correct"] += int(_typed_key(item) in gold)
            source_key = f"add_source_{item['primary_source']}"
            counts[source_key] = counts.get(source_key, 0) + 1
    counts["net_correct_contribution"] = (
        counts["replacement_corrected"]
        - counts["replacement_damaged"]
        + counts["add_correct"]
    )
    return counts
