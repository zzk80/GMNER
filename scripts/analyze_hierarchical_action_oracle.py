"""Measure KEEP/TO_NULL/TO_REAL action headroom on a frozen balanced verifier."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import HierarchicalRecordCandidateCollator, RecordCandidateDataset
from gmner.engine.hierarchical_record_verifier_evaluator import (
    decode_hierarchical_regions,
)
from gmner.engine.utils import match_record_predictions as _match_indices
from gmner.engine.utils import move_record_batch as _move
from gmner.hierarchical_record_verifier_config import (
    load_hierarchical_record_verifier_config,
)
from gmner.models.hierarchical_record_verifier import HierarchicalRecordVerifier
from gmner.models.structured_interval_decoder import (
    greedy_interval_decode,
    weighted_interval_decode,
)


ACTION_FAMILIES = ("residual", "fused", "base")
ACTION_LABELS = ("fix", "damage", "neutral")


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
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--top-k", type=parse_int_list, default=parse_int_list("1,2,4,8"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-error-examples", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required for test analysis after the action design is frozen on dev.",
    )
    return parser.parse_args()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def fixed_prediction_metrics(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "correct": float(correct),
        "predicted": float(predicted),
        "gold": float(gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def topk_real_indices(
    scores: torch.Tensor,
    real_mask: torch.Tensor,
    top_ks: list[int],
) -> dict[int, set[int]]:
    valid = torch.nonzero(real_mask.bool(), as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return {top_k: set() for top_k in top_ks}
    ordered = valid[scores.float()[valid].argsort(descending=True)].tolist()
    return {
        top_k: {int(index) for index in ordered[: min(top_k, len(ordered))]}
        for top_k in top_ks
    }


def action_label(
    *, keep_region: int, action_region: int, positive_regions: set[int]
) -> str:
    """Label an action by its direct change to final triple correctness."""

    keep_correct = keep_region in positive_regions
    action_correct = action_region in positive_regions
    if action_correct and not keep_correct:
        return "fix"
    if keep_correct and not action_correct:
        return "damage"
    return "neutral"


def summarize_action_labels(counter: Counter) -> dict[str, float]:
    total = sum(int(counter[label]) for label in ACTION_LABELS)
    decisive = int(counter["fix"]) + int(counter["damage"])
    return {
        "actions": float(total),
        "fix": float(counter["fix"]),
        "damage": float(counter["damage"]),
        "neutral": float(counter["neutral"]),
        "net_if_all_executed": float(counter["fix"] - counter["damage"]),
        "fix_rate_over_all_actions": counter["fix"] / max(total, 1),
        "fix_precision_excluding_neutral": counter["fix"] / max(decisive, 1),
    }


def summarize_action_cases(
    cases: list[dict],
    *,
    top_ks: list[int],
    keep_correct: int,
    predicted: int,
    gold: int,
) -> dict:
    to_null_fixes = sum(int(case["to_null"]) for case in cases)
    family_oracles: dict[str, dict[str, dict[str, float]]] = {}
    union_oracles: dict[str, dict[str, float]] = {}
    for family in ACTION_FAMILIES:
        family_oracles[family] = {}
        for top_k in top_ks:
            real_fixes = sum(
                int(case["family_hits"][family][top_k]) for case in cases
            )
            corrections = to_null_fixes + real_fixes
            family_oracles[family][f"top{top_k}"] = {
                "real_region_fixes": float(real_fixes),
                "to_null_fixes": float(to_null_fixes),
                "candidate_oracle_fix_count": float(corrections),
                "candidate_oracle_net_correction": float(corrections),
                "candidate_oracle_gmner": fixed_prediction_metrics(
                    keep_correct + corrections, predicted, gold
                )["f1"],
            }

    for top_k in top_ks:
        real_fixes = sum(
            int(
                any(
                    case["family_hits"][family][top_k]
                    for family in ACTION_FAMILIES
                )
            )
            for case in cases
        )
        corrections = to_null_fixes + real_fixes
        union_oracles[f"top{top_k}"] = {
            "real_region_fixes": float(real_fixes),
            "to_null_fixes": float(to_null_fixes),
            "candidate_oracle_fix_count": float(corrections),
            "candidate_oracle_net_correction": float(corrections),
            "candidate_oracle_gmner": fixed_prediction_metrics(
                keep_correct + corrections, predicted, gold
            )["f1"],
        }

    maximum_k = max(top_ks)
    family_membership = Counter()
    for case in cases:
        if case["to_null"]:
            continue
        members = tuple(
            family
            for family in ACTION_FAMILIES
            if case["family_hits"][family][maximum_k]
        )
        if members:
            family_membership["+".join(members)] += 1
    candidate_oracle = union_oracles[f"top{maximum_k}"]
    net = int(candidate_oracle["candidate_oracle_net_correction"])
    if net < 10:
        decision = "stop"
    elif net < 20:
        decision = "structure_validation_only"
    elif net <= 40:
        decision = "implement_controller"
    else:
        decision = "high_potential"
    return {
        "to_null_oracle": {
            "fix_count": float(to_null_fixes),
            "net_correction": float(to_null_fixes),
            "gmner": fixed_prediction_metrics(
                keep_correct + to_null_fixes, predicted, gold
            )["f1"],
        },
        "family_oracles": family_oracles,
        "union_oracles": union_oracles,
        "recoverable_family_overlap_at_max_k": dict(family_membership),
        "candidate_set_oracle": {
            **candidate_oracle,
            "unique_fixable_span_count": float(
                candidate_oracle["candidate_oracle_fix_count"]
            ),
            "oracle_selected_action_count": float(
                candidate_oracle["candidate_oracle_fix_count"]
            ),
            "oracle_final_gmner": float(
                candidate_oracle["candidate_oracle_gmner"]
            ),
            "top_k": float(maximum_k),
            "go_no_go": decision,
        },
    }


def append_example(
    examples: dict[str, list[dict]],
    category: str,
    *,
    limit: int,
    metadata: dict,
    target: dict,
    keep_region: int | None,
) -> None:
    if len(examples[category]) >= limit:
        return
    examples[category].append(
        {
            "record_id": str(metadata.get("record_id", "")),
            "text": str(metadata.get("text", "")),
            "gold": target,
            "keep_region_index": keep_region,
        }
    )


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "Test action-oracle analysis is locked. Develop on dev, then pass "
            "--allow-test only after the action space is frozen."
        )
    root = Path(__file__).resolve().parents[1]
    config = load_hierarchical_record_verifier_config(args.config)
    checkpoint_path = resolve(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cache_path = resolve(
        {
            "train": config.data.train_cache,
            "dev": config.data.dev_cache,
            "test": config.data.test_cache,
        }[args.split],
        root,
    )
    dataset = RecordCandidateDataset(
        cache_path,
        expected_stage1_sha256=(
            None
            if args.split == "train" and config.data.require_oof_train_cache
            else checkpoint.get("stage1_checkpoint_sha256")
        ),
        expected_candidate_sha256=(
            None
            if args.split == "train" and config.data.require_oof_train_cache
            else checkpoint.get("candidate_config_sha256")
        ),
    )
    if args.max_records is not None:
        dataset.records = dataset.records[: max(0, int(args.max_records))]
    loader = DataLoader(
        dataset,
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

    top_ks = sorted(set(args.top_k))
    counts = Counter()
    action_counts = {
        family: Counter() for family in (*ACTION_FAMILIES, "union")
    }
    action_labels = {
        "to_null": Counter(),
        **{
            family: {top_k: Counter() for top_k in top_ks}
            for family in (*ACTION_FAMILIES, "union")
        },
    }
    cases: list[dict] = []
    examples = {
        name: []
        for name in (
            "span_missing",
            "type_wrong",
            "region_missing",
            "region_not_proposed",
            "recoverable_to_null",
            "recoverable_residual",
        )
    }

    for raw_batch in loader:
        batch = _move(raw_batch, device)
        outputs = model(batch)
        decoded = decode_hierarchical_regions(
            outputs,
            batch,
            enable_visibility_correction=config.decode.enable_visibility_correction,
            enable_region_override=False,
            visible_from_null_threshold=config.decode.visible_from_null_threshold,
            null_from_visible_threshold=config.decode.null_from_visible_threshold,
            region_override_mode="margin",
        )
        for row, metadata in enumerate(batch["metadata"]):
            counts["records"] += 1
            span_count = int(batch["span_mask"][row].sum().item())
            spans = [
                tuple(map(int, value))
                for value in batch["span_candidates"][row, :span_count].tolist()
            ]
            source_ids = batch["span_source_ids"][row, :span_count]
            decode_mask = torch.ones(span_count, dtype=torch.bool, device=device)
            if config.decode.stage1_spans_only:
                decode_mask &= source_ids.eq(0)
            utilities = outputs["decode_utility"][row, :span_count].float().clone()
            utilities = utilities.masked_fill(~decode_mask, -1e4)
            utility_values = utilities.tolist()
            if config.decode.strategy == "greedy":
                selected = greedy_interval_decode(
                    spans, utility_values, threshold=config.decode.entity_threshold
                )
            elif config.decode.strategy == "interval":
                selected = weighted_interval_decode(
                    spans, utility_values, threshold=config.decode.entity_threshold
                )
            else:
                raise ValueError(f"Unknown decode strategy: {config.decode.strategy}")

            null_index = int(metadata.get("null_region_index", -1))
            candidate_sets: dict[int, dict[str, dict[int, set[int]]]] = {}
            predictions: list[dict] = []
            for span_index in selected:
                keep_region = int(
                    decoded["pre_override_region_indices"][row, span_index].item()
                )
                real_mask = outputs["real_region_mask"][row, span_index]
                families = {
                    "residual": topk_real_indices(
                        outputs["region_residual_logits"][row, span_index],
                        real_mask,
                        top_ks,
                    ),
                    "fused": topk_real_indices(
                        outputs["final_region_logits"][row, span_index],
                        real_mask,
                        top_ks,
                    ),
                    "base": topk_real_indices(
                        outputs["base_region_scores"][row, span_index],
                        real_mask,
                        top_ks,
                    ),
                }
                # KEEP is already a separate action. Preserve the original top-k
                # truncation, then remove its duplicate from every TO_REAL set.
                for family in ACTION_FAMILIES:
                    for top_k in top_ks:
                        families[family][top_k].discard(keep_region)
                families["union"] = {
                    top_k: set().union(
                        *(families[family][top_k] for family in ACTION_FAMILIES)
                    )
                    for top_k in top_ks
                }
                candidate_sets[span_index] = families
                for family in (*ACTION_FAMILIES, "union"):
                    for top_k in top_ks:
                        action_counts[family][top_k] += len(
                            families[family][top_k]
                        )
                predictions.append(
                    {
                        "span": list(spans[span_index]),
                        "type_id": int(
                            outputs["fixed_type_ids"][row, span_index].item()
                        ),
                        "region_index": keep_region,
                        "candidate_index": span_index,
                    }
                )

            gold = list(metadata.get("gold_entities") or [])
            matches = _match_indices(predictions, gold)
            counts["predicted"] += len(predictions)
            counts["gold"] += len(gold)
            counts["keep_correct"] += len(matches["gmner"])
            gold_spans = {tuple(target["span"]) for target in gold}
            counts["predictions_without_gold_span"] += sum(
                int(tuple(prediction["span"]) not in gold_spans)
                for prediction in predictions
            )
            prediction_by_span = {
                tuple(prediction["span"]): prediction for prediction in predictions
            }
            prediction_by_typed_span = {
                (tuple(prediction["span"]), int(prediction["type_id"])): prediction
                for prediction in predictions
            }

            for gold_index, target in enumerate(gold):
                span = tuple(target["span"])
                if gold_index not in matches["mner"]:
                    prediction = prediction_by_span.get(span)
                    if gold_index in matches["span"]:
                        counts["unactionable_type_wrong"] += 1
                        append_example(
                            examples,
                            "type_wrong",
                            limit=args.max_error_examples,
                            metadata=metadata,
                            target=target,
                            keep_region=(
                                None
                                if prediction is None
                                else int(prediction["region_index"])
                            ),
                        )
                        continue
                    counts["unactionable_span_missing"] += 1
                    append_example(
                        examples,
                        "span_missing",
                        limit=args.max_error_examples,
                        metadata=metadata,
                        target=target,
                        keep_region=None,
                    )
                    continue
                prediction = prediction_by_typed_span.get(
                    (span, int(target["type_id"]))
                )
                if prediction is None:
                    raise RuntimeError(
                        "MNER matching reported a typed span without a matching "
                        f"prediction in record {metadata.get('record_id', '')}."
                    )
                positives = set(target.get("region_positive_indices") or [])
                keep_region = int(prediction["region_index"])
                span_index = int(prediction["candidate_index"])
                counts["action_supervision_spans"] += 1
                if null_index >= 0 and null_index != keep_region:
                    action_labels["to_null"][
                        action_label(
                            keep_region=keep_region,
                            action_region=null_index,
                            positive_regions=positives,
                        )
                    ] += 1
                for family in (*ACTION_FAMILIES, "union"):
                    for top_k in top_ks:
                        for action_region in candidate_sets[span_index][family][top_k]:
                            action_labels[family][top_k][
                                action_label(
                                    keep_region=keep_region,
                                    action_region=action_region,
                                    positive_regions=positives,
                                )
                            ] += 1
                if keep_region in positives:
                    counts["keep_grounding_correct"] += 1
                    continue

                counts["grounding_action_errors"] += 1
                gold_visible = bool(target.get("visible"))
                family_hits = {
                    family: {
                        top_k: bool(
                            candidate_sets[span_index][family][top_k] & positives
                        )
                        for top_k in top_ks
                    }
                    for family in ACTION_FAMILIES
                }
                case = {
                    "to_null": bool(not gold_visible and null_index in positives),
                    "family_hits": family_hits,
                }
                cases.append(case)
                if case["to_null"]:
                    counts["recoverable_to_null"] += 1
                    append_example(
                        examples,
                        "recoverable_to_null",
                        limit=args.max_error_examples,
                        metadata=metadata,
                        target=target,
                        keep_region=keep_region,
                    )
                    continue

                real_positives = positives - {null_index}
                if gold_visible and not real_positives:
                    counts["unactionable_region_missing"] += 1
                    append_example(
                        examples,
                        "region_missing",
                        limit=args.max_error_examples,
                        metadata=metadata,
                        target=target,
                        keep_region=keep_region,
                    )
                    continue
                maximum_k = max(top_ks)
                recoverable = any(
                    family_hits[family][maximum_k] for family in ACTION_FAMILIES
                )
                if recoverable:
                    counts["recoverable_to_real"] += 1
                    if family_hits["residual"][maximum_k]:
                        append_example(
                            examples,
                            "recoverable_residual",
                            limit=args.max_error_examples,
                            metadata=metadata,
                            target=target,
                            keep_region=keep_region,
                        )
                else:
                    counts["unactionable_region_not_proposed"] += 1
                    append_example(
                        examples,
                        "region_not_proposed",
                        limit=args.max_error_examples,
                        metadata=metadata,
                        target=target,
                        keep_region=keep_region,
                    )

    keep_metrics = fixed_prediction_metrics(
        counts["keep_correct"], counts["predicted"], counts["gold"]
    )
    action_summary = summarize_action_cases(
        cases,
        top_ks=top_ks,
        keep_correct=counts["keep_correct"],
        predicted=counts["predicted"],
        gold=counts["gold"],
    )
    maximum_k = max(top_ks)
    candidate_oracle = action_summary["candidate_set_oracle"]
    unique_fixable = int(candidate_oracle["candidate_oracle_fix_count"])
    candidate_oracle.update(
        {
            "fix_action_count": float(
                action_labels["to_null"]["fix"]
                + action_labels["union"][maximum_k]["fix"]
            ),
            "unique_fixable_span_count": float(unique_fixable),
            "oracle_selected_action_count": float(unique_fixable),
            "oracle_final_gmner": float(
                candidate_oracle["candidate_oracle_gmner"]
            ),
        }
    )
    selected_count = max(counts["predicted"], 1)
    mean_action_counts = {
        family: {
            f"top{top_k}": action_counts[family][top_k] / selected_count
            for top_k in top_ks
        }
        for family in (*ACTION_FAMILIES, "union")
    }
    action_label_distribution = {
        "to_null": summarize_action_labels(action_labels["to_null"]),
        **{
            family: {
                f"top{top_k}": summarize_action_labels(
                    action_labels[family][top_k]
                )
                for top_k in top_ks
            }
            for family in (*ACTION_FAMILIES, "union")
        },
    }
    baseline_gold_errors = counts["gold"] - counts["keep_correct"]
    categorized_errors = (
        counts["unactionable_span_missing"]
        + counts["unactionable_type_wrong"]
        + counts["grounding_action_errors"]
    )
    if categorized_errors != baseline_gold_errors:
        raise RuntimeError(
            "Gold error attribution is inconsistent: "
            f"categorized={categorized_errors}, baseline={baseline_gold_errors}."
        )
    report = {
        "split": args.split,
        "checkpoint": str(checkpoint_path.resolve()),
        "cache": str(cache_path.resolve()),
        "top_k": top_ks,
        "action_space": {
            "keep": "balanced decode after Reject/Visibility and before region override",
            "to_null": "the cache NULL region",
            "to_real": ["residual-only top-k", "fused-logit top-k", "base top-k"],
            "top_k_semantics": (
                "truncate each original ranking first, then remove KEEP and union "
                "duplicate region indices"
            ),
            "fixed_span_and_type": True,
            "visibility_correction_enabled": bool(
                config.decode.enable_visibility_correction
            ),
            "visible_from_null_threshold": float(
                config.decode.visible_from_null_threshold
            ),
            "null_from_visible_threshold": float(
                config.decode.null_from_visible_threshold
            ),
            "region_override_enabled_for_keep": False,
        },
        "keep_baseline": keep_metrics,
        "error_breakdown": {
            "baseline_wrong_gold_triples": float(baseline_gold_errors),
            "baseline_incorrect_predictions": float(
                counts["predicted"] - counts["keep_correct"]
            ),
            "predictions_without_gold_span": float(
                counts["predictions_without_gold_span"]
            ),
            "unactionable_because_span_wrong": float(
                counts["unactionable_span_missing"]
            ),
            "unactionable_because_type_wrong": float(
                counts["unactionable_type_wrong"]
            ),
            "grounding_action_errors": float(counts["grounding_action_errors"]),
            "unactionable_because_region_missing": float(
                counts["unactionable_region_missing"]
            ),
            "unactionable_because_region_not_proposed": float(
                counts["unactionable_region_not_proposed"]
            ),
            "recoverable_by_to_null": float(counts["recoverable_to_null"]),
            "recoverable_by_to_real_union": float(counts["recoverable_to_real"]),
            "actionable_wrong_triples": float(
                candidate_oracle["candidate_oracle_fix_count"]
            ),
            "action_supervision_spans": float(counts["action_supervision_spans"]),
            "predictions_excluded_from_action_supervision": float(
                counts["predicted"] - counts["action_supervision_spans"]
            ),
        },
        "mean_unique_real_actions_per_prediction": mean_action_counts,
        "action_label_distribution": action_label_distribution,
        **action_summary,
        "error_examples": examples,
    }
    return report


def main() -> None:
    args = parse_args()
    report = analyze(args)
    root = Path(__file__).resolve().parents[1]
    config = load_hierarchical_record_verifier_config(args.config)
    output = (
        resolve(args.output, root)
        if args.output
        else resolve(config.runtime.output_dir, root)
        / f"{args.split}_hierarchical_action_oracle.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {
        "split": report["split"],
        "keep_gmner": report["keep_baseline"]["f1"],
        "error_breakdown": report["error_breakdown"],
        "oracle_net_correction": {
            "to_null": report["to_null_oracle"]["net_correction"],
            **{
                family: {
                    name: values["candidate_oracle_net_correction"]
                    for name, values in report["family_oracles"][family].items()
                }
                for family in ACTION_FAMILIES
            },
            "union": {
                name: values["candidate_oracle_net_correction"]
                for name, values in report["union_oracles"].items()
            },
        },
        "candidate_set_oracle": report["candidate_set_oracle"],
        "recoverable_family_overlap_at_max_k": report[
            "recoverable_family_overlap_at_max_k"
        ],
        "output": str(output.resolve()),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
