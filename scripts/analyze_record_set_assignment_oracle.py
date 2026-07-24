"""Measure dev-only record-level region assignment potential for M3.5B.

The deployed span, type, reject, Fine, and Evidence Visibility decisions stay
frozen. The diagnostic compares a sharing-aware per-entity candidate oracle
with a strict capacity-1 real-region matching diagnostic. The former is the
upper bound for the same actions; their gap quantifies how unsafe an
unconditional one-to-one Hungarian assumption would be.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.engine.fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from gmner.engine.siglip2_region_reliability_evaluator import (
    frozen_current_visibility_context,
)
from gmner.engine.utils import f1_counts, match_record_predictions
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Use the VinVL-only M3.4 config; SigLIP caches are not loaded.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", default="1,2,4,8,16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def parse_top_k(value: str) -> tuple[int, ...]:
    budgets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not budgets or budgets[0] <= 0:
        raise ValueError("--top-k must contain positive integers.")
    return tuple(budgets)


def ranked_candidate_indices(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> list[int]:
    """Return valid real-region indices in descending score order."""

    if logits.ndim != 1 or candidate_mask.shape != logits.shape:
        raise ValueError("logits and candidate_mask must be aligned vectors.")
    valid = torch.nonzero(candidate_mask.bool(), as_tuple=False).flatten()
    if valid.numel() == 0:
        return []
    order = torch.argsort(logits.float()[valid], descending=True)
    return [int(value) for value in valid[order].tolist()]


def maximum_bipartite_matching(edges: list[set[int]]) -> int:
    """Maximum entity-to-real-region matching with capacity one per region."""

    owner: dict[int, int] = {}

    def augment(entity: int, seen: set[int]) -> bool:
        for region in sorted(edges[entity]):
            if region in seen:
                continue
            seen.add(region)
            previous = owner.get(region)
            if previous is None or augment(previous, seen):
                owner[region] = entity
                return True
        return False

    matched = 0
    for entity in sorted(range(len(edges)), key=lambda index: len(edges[index])):
        matched += int(augment(entity, set()))
    return matched


def record_oracle_at_k(items: list[dict[str, Any]], top_k: int) -> dict[str, int]:
    """Evaluate KEEP/TO_NULL/TO_REAL actions for one record.

    Items contain selected predictions whose span and type match gold. KEEP is
    always available. NULL is reusable; only the strict diagnostic gives real
    regions capacity one.
    """

    current_correct = sum(bool(item["current_correct"]) for item in items)
    null_items = [item for item in items if not bool(item["gold_visible"])]
    visible_items = [item for item in items if bool(item["gold_visible"])]
    edges: list[set[int]] = []
    to_real_fixes = 0
    collision_to_real_fixes = 0
    visible_candidate_covered = 0
    visible_oracle_correctable = 0
    for item in visible_items:
        proposed = set(item["ranked_candidates"][:top_k])
        positive = set(item["positive_regions"])
        proposed_correct = proposed & positive
        correct_actions = set(proposed_correct)
        current_region = item.get("current_region")
        if bool(item["current_correct"]) and current_region is not None:
            # KEEP remains available below the requested diagnostic top-k.
            correct_actions.add(int(current_region))
        edges.append(correct_actions)
        visible_candidate_covered += int(bool(proposed_correct))
        visible_oracle_correctable += int(bool(correct_actions))
        is_fix = not bool(item["current_correct"]) and bool(proposed_correct)
        to_real_fixes += int(is_fix)
        collision_to_real_fixes += int(
            is_fix and bool(item.get("current_collision", False))
        )

    to_null_fixes = sum(
        not bool(item["current_correct"]) for item in null_items
    )
    independent_correct = len(null_items) + visible_oracle_correctable
    strict_correct = len(null_items) + maximum_bipartite_matching(edges)

    current_real = [
        int(item["current_region"])
        for item in items
        if item.get("current_region") is not None
    ]
    region_counts = Counter(current_real)
    collision_entities = sum(
        count for count in region_counts.values() if count > 1
    )
    shared_positive_pairs = 0
    for left in range(len(visible_items)):
        for right in range(left + 1, len(visible_items)):
            shared_positive_pairs += int(
                bool(
                    set(visible_items[left]["positive_regions"])
                    & set(visible_items[right]["positive_regions"])
                )
            )

    if strict_correct > independent_correct:
        raise AssertionError("Strict matching exceeded the independent oracle.")
    return {
        "eligible_entities": len(items),
        "visible_gold_entities": len(visible_items),
        "null_gold_entities": len(null_items),
        "current_correct": current_correct,
        "independent_oracle_correct": independent_correct,
        "strict_capacity_oracle_correct": strict_correct,
        "to_null_fixes": to_null_fixes,
        "to_real_fixes": to_real_fixes,
        "collision_to_real_fixes": collision_to_real_fixes,
        "visible_candidate_covered": visible_candidate_covered,
        "visible_oracle_correctable": visible_oracle_correctable,
        "sharing_gap": independent_correct - strict_correct,
        "current_collision_regions": sum(
            count > 1 for count in region_counts.values()
        ),
        "current_collision_entities": collision_entities,
        "shared_positive_pairs": shared_positive_pairs,
    }


def _metric(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision, recall, score = f1_counts(correct, predicted, gold)
    return {
        "correct": float(correct),
        "predicted": float(predicted),
        "gold": float(gold),
        "precision": precision,
        "recall": recall,
        "f1": score,
    }


def _duplicate_region_count(indices: list[int]) -> tuple[int, int]:
    counts = Counter(indices)
    return (
        sum(count > 1 for count in counts.values()),
        sum(count for count in counts.values() if count > 1),
    )


def _gold_for_prediction(
    span: tuple[int, int],
    type_id: int,
    gold_entities: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    span_matches = [
        gold for gold in gold_entities if tuple(gold.get("span") or ()) == span
    ]
    if not span_matches:
        return "span_wrong", None
    for gold in span_matches:
        if int(gold.get("type_id", -1)) == int(type_id):
            return "eligible", gold
    return "type_wrong", None


def _compact_example(record: dict[str, Any], summary: dict[str, int]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "text": record["text"],
        "selected_predictions": record["selected_predictions"],
        "eligible_entities": summary["eligible_entities"],
        "current_correct": summary["current_correct"],
        "independent_oracle_correct": summary["independent_oracle_correct"],
        "strict_capacity_oracle_correct": summary[
            "strict_capacity_oracle_correct"
        ],
        "current_collision_entities": summary["current_collision_entities"],
        "sharing_gap": summary["sharing_gap"],
        "entities": [
            {
                "mention": item["mention"],
                "span": list(item["span"]),
                "type_id": item["type_id"],
                "gold_visible": item["gold_visible"],
                "current_region": item["current_region"],
                "current_correct": item["current_correct"],
                "positive_regions": sorted(item["positive_regions"]),
                "ranked_candidates": item["ranked_candidates"],
            }
            for item in record["items"]
        ],
    }


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config = load_siglip2_region_reliability_config(args.config)
    if config.model.feature_mode != "vinvl_only":
        raise ValueError(
            "M3.5B must use the VinVL-only reliability config; no SigLIP cache "
            "is needed for this frozen-chain oracle."
        )
    if args.device:
        config.runtime.device = args.device
    budgets = parse_top_k(args.top_k)
    dataset, collator = _paired_dataset(config, root, "dev")
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    (
        unused_reliability_model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
        evidence_checkpoint,
    ) = load_frozen_reliability_chain(config, root, device)
    del unused_reliability_model
    validate_fingerprints(
        _base_paired(dataset),
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )

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

    counts = Counter()
    baseline_correct = Counter()
    records: list[dict[str, Any]] = []
    for raw_batch in loader:
        paired = move_paired_record_batch(raw_batch, device)
        formal = paired["formal"]
        expanded = paired["expanded"]
        baseline = frozen_hierarchical_context(
            hierarchy, formal, expanded, decode_options=region_options
        )
        hierarchy_outputs = baseline["outputs"]
        decoded = baseline["decoded"]
        hierarchy_visible = baseline["visible_mask"]
        assert isinstance(hierarchy_outputs, dict)
        assert isinstance(decoded, dict)
        assert isinstance(hierarchy_visible, torch.Tensor)
        fine_outputs = fine_model(expanded)
        hierarchy_outputs, _, current_visible = frozen_current_visibility_context(
            evidence_model,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            hierarchy_visible_mask=hierarchy_visible,
            base_is_null_mask=decoded["base_is_null"],
            decode_options=all_decode_options,
        )
        fine_top1 = fine_outputs["final_region_logits"].argmax(dim=-1)

        for row, metadata in enumerate(expanded["metadata"]):
            spans, selected = _selected_span_indices(
                hierarchy_outputs,
                formal,
                row,
                entity_threshold=entity_threshold,
                decode_strategy=decode_strategy,
                stage1_spans_only=stage1_spans_only,
            )
            gold_entities = list(metadata.get("gold_entities") or [])
            null_index = int(metadata.get("null_region_index", -1))
            predictions = []
            items: list[dict[str, Any]] = []
            selected_real_regions: list[int] = []
            tokens = list(metadata.get("tokens") or [])
            for span_index in selected:
                span = spans[span_index]
                type_id = int(
                    hierarchy_outputs["fixed_type_ids"][row, span_index].item()
                )
                visible = bool(current_visible[row, span_index].item())
                region_index = (
                    int(fine_top1[row, span_index].item())
                    if visible
                    else null_index
                )
                predictions.append(
                    {
                        "span": list(span),
                        "type_id": type_id,
                        "region_index": region_index,
                    }
                )
                if visible:
                    selected_real_regions.append(region_index)
                status, gold = _gold_for_prediction(span, type_id, gold_entities)
                counts[status] += 1
                if gold is None:
                    continue
                positive_all = {
                    int(index)
                    for index in (gold.get("region_positive_indices") or [])
                }
                positive_real = {
                    index for index in positive_all if index != null_index
                }
                gold_visible = bool(gold.get("visible", False))
                current_correct = region_index in positive_all
                real_mask = (
                    fine_outputs["candidate_mask"][row, span_index].bool()
                    & expanded["region_mask"][row].bool()
                    & ~expanded["region_is_null"][row].bool()
                )
                ranked = ranked_candidate_indices(
                    fine_outputs["final_region_logits"][row, span_index],
                    real_mask,
                )
                if gold_visible:
                    counts["visible_eligible"] += 1
                    if not positive_real:
                        counts["visible_detector_missing"] += 1
                    elif not (positive_real & set(ranked)):
                        counts["visible_not_in_fine_candidates"] += 1
                else:
                    counts["null_eligible"] += 1
                start, end = span
                items.append(
                    {
                        "span": span,
                        "mention": " ".join(tokens[start:end]),
                        "type_id": type_id,
                        "gold_visible": gold_visible,
                        "positive_regions": positive_real,
                        "current_region": region_index if visible else None,
                        "current_correct": current_correct,
                        "ranked_candidates": ranked,
                    }
                )

            matches = match_record_predictions(predictions, gold_entities)
            for metric, matched in matches.items():
                baseline_correct[metric] += len(matched)
            counts["records"] += 1
            counts["predicted"] += len(predictions)
            counts["gold"] += len(gold_entities)
            counts["eligible_predictions"] += len(items)
            selected_spans = {spans[index] for index in selected}
            counts["gold_without_selected_span"] += sum(
                tuple(gold.get("span") or ()) not in selected_spans
                for gold in gold_entities
            )
            collision_regions, collision_entities = _duplicate_region_count(
                selected_real_regions
            )
            selected_region_counts = Counter(selected_real_regions)
            duplicate_regions = {
                region
                for region, count in selected_region_counts.items()
                if count > 1
            }
            for item in items:
                item["current_collision"] = (
                    item.get("current_region") in duplicate_regions
                )
            counts["all_selected_collision_records"] += int(
                collision_regions > 0
            )
            counts["all_selected_collision_regions"] += collision_regions
            counts["all_selected_collision_entities"] += collision_entities
            records.append(
                {
                    "record_id": str(metadata.get("record_id", "")),
                    "text": str(metadata.get("text", "")),
                    "selected_predictions": len(predictions),
                    "items": items,
                    "all_selected_current_collision": collision_regions > 0,
                }
            )

    baseline_metrics = {
        metric: _metric(
            int(baseline_correct[metric]),
            int(counts["predicted"]),
            int(counts["gold"]),
        )
        for metric in ("span", "mner", "eeg", "gmner")
    }
    direct_correct = sum(
        bool(item["current_correct"])
        for record in records
        for item in record["items"]
    )
    if direct_correct != int(baseline_correct["gmner"]):
        raise RuntimeError(
            "Direct eligible-item correctness does not reproduce baseline GMNER."
        )

    budget_results: dict[str, Any] = {}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for budget in budgets:
        aggregate = Counter()
        for record in records:
            summary = record_oracle_at_k(record["items"], budget)
            for key, value in summary.items():
                aggregate[key] += int(value)
            independent_gain = (
                summary["independent_oracle_correct"]
                - summary["current_correct"]
            )
            strict_gain = (
                summary["strict_capacity_oracle_correct"]
                - summary["current_correct"]
            )
            multi_entity = summary["eligible_entities"] >= 2
            collision = bool(record["all_selected_current_collision"])
            aggregate["multi_entity_record_count"] += int(multi_entity)
            aggregate["single_entity_record_count"] += int(
                summary["eligible_entities"] == 1
            )
            aggregate["records_with_independent_gain"] += int(
                independent_gain > 0
            )
            aggregate["records_with_strict_gain"] += int(strict_gain > 0)
            aggregate["records_with_sharing_gap"] += int(
                summary["sharing_gap"] > 0
            )
            prefix = "multi_entity" if multi_entity else "single_entity"
            aggregate[f"{prefix}_independent_net"] += independent_gain
            aggregate[f"{prefix}_strict_net"] += strict_gain
            if collision:
                aggregate["collision_record_count"] += 1
                aggregate["collision_independent_net"] += independent_gain
                aggregate["collision_strict_net"] += strict_gain
            elif multi_entity:
                aggregate["noncollision_multi_independent_net"] += independent_gain
                aggregate["noncollision_multi_strict_net"] += strict_gain
            if budget == budgets[-1]:
                if independent_gain > 0:
                    examples["recoverable"].append(
                        _compact_example(record, summary)
                    )
                if summary["sharing_gap"] > 0:
                    examples["strict_capacity_conflict"].append(
                        _compact_example(record, summary)
                    )
                if collision and independent_gain > 0:
                    examples["collision_recoverable"].append(
                        _compact_example(record, summary)
                    )

        independent_correct = int(aggregate["independent_oracle_correct"])
        strict_correct = int(aggregate["strict_capacity_oracle_correct"])
        current_correct = int(baseline_correct["gmner"])
        if independent_correct < current_correct:
            raise AssertionError("KEEP-aware independent oracle regressed baseline.")
        if strict_correct > independent_correct:
            raise AssertionError("Strict matching exceeded independent oracle.")
        budget_results[f"top{budget}"] = {
            "candidate_actions": {
                "to_null_fixes": float(aggregate["to_null_fixes"]),
                "to_real_fixes": float(aggregate["to_real_fixes"]),
                "visible_gold_entities": float(aggregate["visible_gold_entities"]),
                "visible_candidate_covered": float(
                    aggregate["visible_candidate_covered"]
                ),
                "visible_candidate_coverage": aggregate[
                    "visible_candidate_covered"
                ]
                / max(aggregate["visible_gold_entities"], 1),
                "visible_oracle_correctable": float(
                    aggregate["visible_oracle_correctable"]
                ),
                "collision_entity_to_real_fixes": float(
                    aggregate["collision_to_real_fixes"]
                ),
            },
            "sharing_aware_independent_oracle": {
                **_metric(
                    independent_correct,
                    int(counts["predicted"]),
                    int(counts["gold"]),
                ),
                "net_correction": float(independent_correct - current_correct),
                "records_with_gain": float(
                    aggregate["records_with_independent_gain"]
                ),
            },
            "strict_real_region_capacity_one": {
                **_metric(
                    strict_correct,
                    int(counts["predicted"]),
                    int(counts["gold"]),
                ),
                "net_correction": float(strict_correct - current_correct),
                "records_with_gain": float(aggregate["records_with_strict_gain"]),
                "sharing_gap_from_independent": float(
                    independent_correct - strict_correct
                ),
                "records_with_sharing_gap": float(
                    aggregate["records_with_sharing_gap"]
                ),
            },
            "where_the_gain_is": {
                "single_entity_independent_net": float(
                    aggregate["single_entity_independent_net"]
                ),
                "single_entity_record_count": float(
                    aggregate["single_entity_record_count"]
                ),
                "multi_entity_independent_net": float(
                    aggregate["multi_entity_independent_net"]
                ),
                "multi_entity_record_count": float(
                    aggregate["multi_entity_record_count"]
                ),
                "multi_entity_strict_net": float(
                    aggregate["multi_entity_strict_net"]
                ),
                "current_collision_record_count": float(
                    aggregate["collision_record_count"]
                ),
                "collision_record_independent_net": float(
                    aggregate["collision_independent_net"]
                ),
                "collision_record_strict_net": float(
                    aggregate["collision_strict_net"]
                ),
                "noncollision_multi_entity_independent_net": float(
                    aggregate["noncollision_multi_independent_net"]
                ),
                "noncollision_multi_entity_strict_net": float(
                    aggregate["noncollision_multi_strict_net"]
                ),
                "eligible_current_collision_regions": float(
                    aggregate["current_collision_regions"]
                ),
                "eligible_current_collision_entities": float(
                    aggregate["current_collision_entities"]
                ),
                "gold_positive_region_overlap_pairs": float(
                    aggregate["shared_positive_pairs"]
                ),
            },
        }

    limit = max(int(args.examples), 0)
    for rows in examples.values():
        rows.sort(
            key=lambda row: (
                row["independent_oracle_correct"] - row["current_correct"],
                row["sharing_gap"],
            ),
            reverse=True,
        )
        del rows[limit:]

    max_budget = budget_results[f"top{budgets[-1]}"]
    independent_net = int(
        max_budget["sharing_aware_independent_oracle"]["net_correction"]
    )
    collision_net = int(
        max_budget["where_the_gain_is"]["collision_record_independent_net"]
    )
    collision_entity_fixes = int(
        max_budget["candidate_actions"]["collision_entity_to_real_fixes"]
    )
    multi_net = int(
        max_budget["where_the_gain_is"]["multi_entity_independent_net"]
    )
    return {
        "protocol": {
            "milestone": "M3.5B",
            "split": "dev",
            "test_read": False,
            "config": str(resolve(args.config, root)),
            "candidate_space": "Fine R36 real candidates plus KEEP and TO_NULL",
            "top_k": list(budgets),
            "device": str(device),
            "evidence_visibility_checkpoint_epoch": evidence_checkpoint.get(
                "epoch"
            ),
        },
        "definitions": {
            "independent_oracle": (
                "Gold-aware KEEP/TO_NULL/TO_REAL choice per entity; real regions "
                "may be reused. This is the upper bound for the same actions."
            ),
            "strict_capacity_one": (
                "Gold-aware maximum bipartite matching for visible entities; "
                "each real region can be used once, while NULL remains reusable."
            ),
            "sharing_gap": (
                "Independent oracle correct minus strict matching correct. A "
                "positive value is evidence against unconditional Hungarian."
            ),
            "set_specific_signal": (
                "Candidate-oracle gain inside records whose deployed visible "
                "predictions currently collide on a real region."
            ),
        },
        "baseline": baseline_metrics,
        "data_summary": {
            "records": float(counts["records"]),
            "predictions": float(counts["predicted"]),
            "gold_entities": float(counts["gold"]),
            "eligible_correct_span_type_predictions": float(
                counts["eligible_predictions"]
            ),
            "predictions_without_gold_span": float(counts["span_wrong"]),
            "predictions_with_wrong_type": float(counts["type_wrong"]),
            "gold_without_selected_span": float(counts["gold_without_selected_span"]),
            "visible_eligible": float(counts["visible_eligible"]),
            "null_eligible": float(counts["null_eligible"]),
            "visible_detector_missing": float(
                counts["visible_detector_missing"]
            ),
            "visible_not_in_fine_candidates": float(
                counts["visible_not_in_fine_candidates"]
            ),
            "all_selected_collision_records": float(
                counts["all_selected_collision_records"]
            ),
            "all_selected_collision_regions": float(
                counts["all_selected_collision_regions"]
            ),
            "all_selected_collision_entities": float(
                counts["all_selected_collision_entities"]
            ),
        },
        "budgets": budget_results,
        "decision_evidence_at_max_k": {
            "independent_candidate_net": float(independent_net),
            "multi_entity_candidate_net": float(multi_net),
            "collision_record_candidate_net": float(collision_net),
            "collision_entity_to_real_fixes": float(collision_entity_fixes),
            "non_collision_or_single_entity_net": float(
                independent_net - collision_net
            ),
            "strict_capacity_sharing_gap": max_budget[
                "strict_real_region_capacity_one"
            ]["sharing_gap_from_independent"],
            "interpretation": (
                "A large independent net proves candidate coverage, not set "
                "reasoning. Record-level modeling is specifically supported "
                "only when multi-entity/collision gain is substantial and the "
                "strict sharing gap is understood. collision_entity_to_real_fixes "
                "is the narrower set-specific signal; collision-record net may "
                "also contain unrelated TO_NULL fixes."
            ),
        },
        "examples_at_max_k": dict(examples),
    }


def main() -> None:
    args = parse_args()
    result = analyze(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        "baseline_gmner": result["baseline"]["gmner"]["f1"],
        "data_summary": result["data_summary"],
        "budgets": {
            key: {
                "independent_net": value[
                    "sharing_aware_independent_oracle"
                ]["net_correction"],
                "strict_net": value["strict_real_region_capacity_one"][
                    "net_correction"
                ],
                "sharing_gap": value["strict_real_region_capacity_one"][
                    "sharing_gap_from_independent"
                ],
                "multi_entity_net": value["where_the_gain_is"][
                    "multi_entity_independent_net"
                ],
                "collision_net": value["where_the_gain_is"][
                    "collision_record_independent_net"
                ],
            }
            for key, value in result["budgets"].items()
        },
        "decision_evidence": result["decision_evidence_at_max_k"],
        "output": str(output),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
