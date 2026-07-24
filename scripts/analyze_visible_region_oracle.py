"""Diagnose VinVL proposal coverage and visible-region errors for the main chain."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import HierarchicalRecordCandidateCollator, RecordCandidateDataset
from gmner.engine.hierarchical_record_verifier_evaluator import (
    decode_hierarchical_regions,
)
from gmner.engine.utils import move_record_batch
from gmner.hierarchical_record_verifier_config import (
    load_hierarchical_record_verifier_config,
)
from gmner.models.hierarchical_record_verifier import HierarchicalRecordVerifier
from gmner.models.structured_interval_decoder import (
    greedy_interval_decode,
    weighted_interval_decode,
)


def parse_int_list(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of positive integers."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expanded-cache", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument(
        "--proposal-budgets",
        type=parse_int_list,
        default=parse_int_list("16,36"),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-error-examples", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required for test analysis after the design is frozen on dev.",
    )
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def proposal_covered(
    positive_indices: list[int] | tuple[int, ...],
    *,
    budget: int,
    null_index: int,
) -> bool:
    """Return raw proposal-set coverage, independent of model ranking scores."""

    return any(
        0 <= int(index) < int(budget) and int(index) != int(null_index)
        for index in positive_indices
    )


def classify_visible_case(
    *,
    span_present: bool,
    type_correct: bool,
    final_present: bool,
    final_correct: bool,
    formal_covered: bool,
    final_is_null: bool,
    base_correct: bool,
) -> str:
    """Assign one mutually exclusive outcome to a visible gold entity."""

    if not span_present:
        return "unactionable_span_missing"
    if not type_correct:
        return "unactionable_type_wrong"
    if final_correct:
        return "final_correct"
    if not formal_covered:
        return "A_region_missing_formal_budget"
    if not final_present:
        return "unactionable_final_span_removed"
    if final_is_null:
        return "D_visibility_false_null"
    if base_correct:
        return "C_verifier_real_region_damage"
    return "B_base_misrank_remaining"


def _record_id(record: dict) -> str:
    return str((record.get("metadata") or {}).get("record_id", ""))


def _candidate_spec(dataset: RecordCandidateDataset) -> dict:
    return dict(dataset.metadata.get("candidate_config") or {})


def _validate_cache_pair(
    formal: RecordCandidateDataset,
    expanded: RecordCandidateDataset,
    *,
    checkpoint: dict,
) -> tuple[int, int]:
    expected_stage1 = str(checkpoint.get("stage1_checkpoint_sha256") or "")
    formal_stage1 = str(formal.metadata.get("stage1_checkpoint_sha256") or "")
    expanded_stage1 = str(expanded.metadata.get("stage1_checkpoint_sha256") or "")
    if expected_stage1 and formal_stage1 != expected_stage1:
        raise ValueError("Formal cache does not match the verifier Stage1 checkpoint.")
    if formal_stage1 != expanded_stage1:
        raise ValueError("Formal and expanded caches use different Stage1 checkpoints.")

    formal_spec = _candidate_spec(formal)
    expanded_spec = _candidate_spec(expanded)
    ignored = {"max_regions"}
    mismatched = {
        key: (formal_spec.get(key), expanded_spec.get(key))
        for key in sorted((set(formal_spec) | set(expanded_spec)) - ignored)
        if formal_spec.get(key) != expanded_spec.get(key)
    }
    if mismatched:
        raise ValueError(
            "Expanded cache must differ only in max_regions; mismatches="
            f"{mismatched}"
        )

    formal_budget = int(formal_spec.get("max_regions", 0))
    expanded_budget = int(expanded_spec.get("max_regions", 0))
    if formal_budget <= 0 or expanded_budget <= formal_budget:
        raise ValueError(
            "Expected an expanded cache with a larger max_regions value; "
            f"formal={formal_budget}, expanded={expanded_budget}."
        )
    return formal_budget, expanded_budget


def _select_predictions(
    *,
    batch: dict,
    outputs: dict[str, torch.Tensor],
    decoded_regions: dict[str, torch.Tensor],
    row: int,
    entity_threshold: float,
    decode_strategy: str,
    stage1_spans_only: bool,
) -> list[dict]:
    span_count = int(batch["span_mask"][row].sum().item())
    spans = [
        tuple(map(int, value))
        for value in batch["span_candidates"][row, :span_count].tolist()
    ]
    utilities = outputs["decode_utility"][row, :span_count].float().clone()
    decode_mask = torch.ones(span_count, dtype=torch.bool, device=utilities.device)
    if stage1_spans_only:
        decode_mask &= batch["span_source_ids"][row, :span_count].eq(0)
    utilities = utilities.masked_fill(~decode_mask, -1e4)
    values = utilities.tolist()
    if decode_strategy == "greedy":
        selected = greedy_interval_decode(spans, values, threshold=entity_threshold)
    elif decode_strategy == "interval":
        selected = weighted_interval_decode(spans, values, threshold=entity_threshold)
    else:
        raise ValueError(f"Unknown decode strategy: {decode_strategy}")
    return [
        {
            "span": spans[index],
            "type_id": int(outputs["fixed_type_ids"][row, index].item()),
            "region_index": int(
                decoded_regions["region_indices"][row, index].item()
            ),
        }
        for index in selected
    ]


def _by_span(values: list[dict]) -> dict[tuple[int, int], dict]:
    return {
        tuple(map(int, value["span"])): value
        for value in values
    }


def _append_example(
    examples: dict[str, list[dict]],
    category: str,
    *,
    limit: int,
    metadata: dict,
    gold: dict,
    base: dict | None,
    final: dict | None,
    coverage: dict[int, bool],
) -> None:
    if limit <= 0 or len(examples[category]) >= limit:
        return
    examples[category].append(
        {
            "record_id": str(metadata.get("record_id", "")),
            "text": str(metadata.get("text", "")),
            "entity": {
                "span": list(gold["span"]),
                "text": str(gold.get("text", "")),
                "type_id": int(gold["type_id"]),
            },
            "base_region_index": None if base is None else int(base["region_index"]),
            "final_region_index": None if final is None else int(final["region_index"]),
            "proposal_coverage": {
                f"R{budget}": bool(covered)
                for budget, covered in coverage.items()
            },
        }
    )


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise ValueError("Refusing test oracle analysis without --allow-test.")

    root = Path(__file__).resolve().parents[1]
    config = load_hierarchical_record_verifier_config(args.config)
    formal_path = resolve(
        {
            "train": config.data.train_cache,
            "dev": config.data.dev_cache,
            "test": config.data.test_cache,
        }[args.split],
        root,
    )
    checkpoint_path = resolve(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    formal = RecordCandidateDataset(
        formal_path,
        expected_stage1_sha256=checkpoint.get("stage1_checkpoint_sha256"),
        expected_candidate_sha256=checkpoint.get("candidate_config_sha256"),
    )
    expanded = RecordCandidateDataset(resolve(args.expanded_cache, root))
    formal_budget, expanded_budget = _validate_cache_pair(
        formal, expanded, checkpoint=checkpoint
    )

    budgets = sorted(
        {
            min(int(value), expanded_budget)
            for value in args.proposal_budgets
            if int(value) > 0
        }
        | {formal_budget, expanded_budget}
    )
    expanded_by_id = {_record_id(record): record for record in expanded.records}
    record_limit = len(formal)
    if args.max_records is not None:
        record_limit = max(0, min(int(args.max_records), len(formal)))
    formal_ids = [
        _record_id(record) for record in formal.records[:record_limit]
    ]
    missing_ids = [record_id for record_id in formal_ids if record_id not in expanded_by_id]
    if missing_ids:
        raise ValueError(
            f"Expanded cache is missing {len(missing_ids)} formal records; "
            f"examples={missing_ids[:5]}"
        )

    selected_dataset = formal
    if record_limit < len(formal):
        selected_dataset = Subset(formal, range(record_limit))
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size or config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=HierarchicalRecordCandidateCollator(),
    )
    device_name = args.device or config.runtime.device
    device = torch.device(
        device_name
        if str(device_name).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    model = HierarchicalRecordVerifier(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    counts = Counter()
    coverage_counts = Counter()
    attribution = Counter()
    examples: dict[str, list[dict]] = {
        key: []
        for key in (
            "A_region_missing_formal_budget",
            "B_base_misrank_remaining",
            "C_verifier_real_region_damage",
            "D_visibility_false_null",
            "unactionable_span_missing",
            "unactionable_type_wrong",
            "unactionable_final_span_removed",
        )
    }
    decode = config.decode

    for raw_batch in loader:
        batch = move_record_batch(raw_batch, device)
        outputs = model(batch)
        decoded_regions = decode_hierarchical_regions(
            outputs,
            batch,
            enable_visibility_correction=decode.enable_visibility_correction,
            enable_region_override=decode.enable_region_override,
            visible_from_null_threshold=decode.visible_from_null_threshold,
            null_from_visible_threshold=decode.null_from_visible_threshold,
            region_override_mode=decode.region_override_mode,
            region_override_logit_margin=decode.region_override_logit_margin,
            region_override_probability_margin=(
                decode.region_override_probability_margin
            ),
            override_damage_cost=decode.override_damage_cost,
            override_utility_threshold=decode.override_utility_threshold,
            enable_action_controller=decode.enable_action_controller,
            action_top_k=decode.action_top_k,
            action_execution_margin=decode.action_execution_margin,
        )

        for row, metadata in enumerate(batch["metadata"]):
            record_id = str(metadata.get("record_id", ""))
            expanded_record = expanded_by_id[record_id]
            expanded_metadata = dict(expanded_record.get("metadata") or {})
            expanded_null = int(expanded_metadata.get("null_region_index", -1))
            expanded_gold = {
                (tuple(map(int, value["span"])), int(value["type_id"])): value
                for value in expanded_metadata.get("gold_entities") or []
            }
            base_by_span = _by_span(list(metadata.get("stage1_predictions") or []))
            final_values = _select_predictions(
                batch=batch,
                outputs=outputs,
                decoded_regions=decoded_regions,
                row=row,
                entity_threshold=decode.entity_threshold,
                decode_strategy=decode.strategy,
                stage1_spans_only=decode.stage1_spans_only,
            )
            final_by_span = _by_span(final_values)
            formal_null = int(metadata.get("null_region_index", -1))

            for gold in metadata.get("gold_entities") or []:
                if not bool(gold.get("visible", False)):
                    continue
                counts["visible_gold"] += 1
                span = tuple(map(int, gold["span"]))
                gold_type = int(gold["type_id"])
                expanded_target = expanded_gold.get((span, gold_type))
                if expanded_target is None:
                    raise ValueError(
                        f"Expanded cache lost gold entity {record_id}:{span}:{gold_type}."
                    )
                positive_expanded = list(
                    expanded_target.get("region_positive_indices") or []
                )
                coverage = {
                    budget: proposal_covered(
                        positive_expanded,
                        budget=budget,
                        null_index=expanded_null,
                    )
                    for budget in budgets
                }
                for budget, covered in coverage.items():
                    coverage_counts[f"R{budget}"] += int(covered)

                formal_covered = coverage[formal_budget]
                expanded_covered = coverage[expanded_budget]
                counts["newly_covered_by_expansion"] += int(
                    expanded_covered and not formal_covered
                )
                base = base_by_span.get(span)
                final = final_by_span.get(span)
                span_present = base is not None
                type_correct = bool(
                    base is not None and int(base["type_id"]) == gold_type
                )
                formal_positives = {
                    int(index)
                    for index in gold.get("region_positive_indices") or []
                    if int(index) != formal_null
                }
                base_correct = bool(
                    type_correct
                    and int(base["region_index"]) in formal_positives
                )
                final_present = final is not None
                final_type_correct = bool(
                    final is not None and int(final["type_id"]) == gold_type
                )
                final_correct = bool(
                    final_type_correct
                    and int(final["region_index"]) in formal_positives
                )
                final_is_null = bool(
                    final is not None and int(final["region_index"]) == formal_null
                )

                if type_correct:
                    counts["actionable_span_type"] += 1
                    counts["base_region_correct"] += int(base_correct)
                    counts["base_region_wrong"] += int(not base_correct)
                    counts["base_wrong_gold_in_formal"] += int(
                        not base_correct and formal_covered
                    )
                    counts["base_wrong_gold_only_in_expanded"] += int(
                        not base_correct and expanded_covered and not formal_covered
                    )
                    counts["verifier_corrected_base_wrong"] += int(
                        not base_correct and final_correct
                    )
                    counts["verifier_damaged_base_correct"] += int(
                        base_correct and not final_correct
                    )
                    counts["final_visible_correct"] += int(final_correct)
                    counts["final_false_null"] += int(final_is_null)
                    counts["final_false_null_gold_in_formal"] += int(
                        final_is_null and formal_covered
                    )

                category = classify_visible_case(
                    span_present=span_present,
                    type_correct=type_correct,
                    final_present=final_present,
                    final_correct=final_correct,
                    formal_covered=formal_covered,
                    final_is_null=final_is_null,
                    base_correct=base_correct,
                )
                attribution[category] += 1
                if category in examples:
                    _append_example(
                        examples,
                        category,
                        limit=max(0, int(args.max_error_examples)),
                        metadata=metadata,
                        gold=gold,
                        base=base,
                        final=final,
                        coverage=coverage,
                    )

    visible = int(counts["visible_gold"])
    region_recall = {
        key: {
            "covered": int(coverage_counts[key]),
            "visible_gold": visible,
            "recall": coverage_counts[key] / max(visible, 1),
        }
        for key in (f"R{budget}" for budget in budgets)
    }
    formal_recall = region_recall[f"R{formal_budget}"]["recall"]
    expanded_recall = region_recall[f"R{expanded_budget}"]["recall"]
    gain = float(expanded_recall) - float(formal_recall)
    recommendation = (
        "candidate_expansion_then_grounding_adapter"
        if gain >= 0.03
        else "grounding_adapter_first"
    )
    attribution_total = sum(attribution.values())
    result = {
        "split": args.split,
        "formal_cache": str(formal.path.resolve()),
        "expanded_cache": str(expanded.path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "formal_budget": formal_budget,
        "expanded_budget": expanded_budget,
        "visible_gold_count": visible,
        "region_proposal_recall": region_recall,
        "coverage_gain_expanded_minus_formal": gain,
        "newly_covered_by_expansion": int(counts["newly_covered_by_expansion"]),
        "actionability": {
            "span_type_correct_visible": int(counts["actionable_span_type"]),
            "base_region_correct": int(counts["base_region_correct"]),
            "base_region_wrong": int(counts["base_region_wrong"]),
            "base_wrong_gold_in_formal": int(
                counts["base_wrong_gold_in_formal"]
            ),
            "base_wrong_gold_only_in_expanded": int(
                counts["base_wrong_gold_only_in_expanded"]
            ),
            "verifier_corrected_base_wrong": int(
                counts["verifier_corrected_base_wrong"]
            ),
            "verifier_damaged_base_correct": int(
                counts["verifier_damaged_base_correct"]
            ),
            "final_visible_correct": int(counts["final_visible_correct"]),
            "final_false_null": int(counts["final_false_null"]),
            "final_false_null_gold_in_formal": int(
                counts["final_false_null_gold_in_formal"]
            ),
        },
        "visible_outcome_attribution": {
            "counts": dict(attribution),
            "total": int(attribution_total),
            "consistent_with_visible_gold": attribution_total == visible,
            "definitions": {
                "A_region_missing_formal_budget": (
                    "Gold box has no IoU-qualified proposal in the formal budget."
                ),
                "B_base_misrank_remaining": (
                    "Gold proposal exists, Stage1 is wrong, and the final real region "
                    "remains wrong."
                ),
                "C_verifier_real_region_damage": (
                    "Stage1 real region is correct but the verifier changes it to a "
                    "wrong real region."
                ),
                "D_visibility_false_null": (
                    "Gold is visible and covered, but final decoding selects NULL."
                ),
            },
        },
        "decision": {
            "three_point_gain_threshold": 0.03,
            "recommendation": recommendation,
            "reason": (
                f"R{expanded_budget}-R{formal_budget} proposal recall gain is "
                f"{gain:.4f}."
            ),
        },
        "error_examples": examples,
    }
    return result


def main() -> None:
    args = parse_args()
    result = analyze(args)
    root = Path(__file__).resolve().parents[1]
    output = (
        resolve(args.output, root)
        if args.output
        else resolve(
            load_hierarchical_record_verifier_config(args.config).runtime.output_dir,
            root,
        )
        / f"{args.split}_visible_region_oracle.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {key: value for key, value in result.items() if key != "error_examples"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"saved_to={output.resolve()}")


if __name__ == "__main__":
    main()
