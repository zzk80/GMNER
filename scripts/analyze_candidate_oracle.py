"""Measure record-level Stage-1 span/type/region candidate ceilings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID, IGNORE_INDEX
from gmner.data import (
    GMNERCollator,
    MMNERJsonDataset,
    TextGraphBuilder,
    load_word_aligned_tokenizer,
    validate_model_input_length,
)
from gmner.data.graph_builders import GraphBuilderConfig
from gmner.engine import evaluate_model
from gmner.engine.utils import move_batch_to_device
from gmner.models import GMNERModel
from gmner.utils.candidate_decoding import (
    bio_constraint_masks,
    build_span_candidates,
    extract_crf_parameters,
    extract_word_spans,
    k_best_viterbi_decode,
    sequence_hamming_diversity,
)
from gmner.utils.io import maybe_convert_conll
from gmner.utils.metrics import (
    extract_entities_from_word_labels,
    word_labels_from_subwords,
)


def parse_int_list(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")
    return values


def region_candidate_covered(
    gt_boxes: Iterable[Iterable[float]],
    region_boxes: torch.Tensor,
    ranking_scores: torch.Tensor,
    region_mask: torch.Tensor,
    top_r: int,
    iou_threshold: float,
    has_null_region: bool,
) -> bool:
    """Return whether a gold region is present among top-R valid proposals."""

    gt_boxes = list(gt_boxes)
    if not gt_boxes:
        return bool(has_null_region and region_mask.numel() > 0 and region_mask[-1] > 0)

    valid = region_mask.to(dtype=torch.bool).clone()
    if has_null_region and valid.numel() > 0:
        valid[-1] = False
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        return False

    count = min(int(top_r), int(valid_indices.numel()))
    candidate_scores = ranking_scores[valid_indices]
    selected = valid_indices[candidate_scores.topk(count).indices]
    candidates = region_boxes[selected]
    gold = torch.tensor(gt_boxes, dtype=candidates.dtype, device=candidates.device)
    return bool((box_iou(gold, candidates) >= float(iou_threshold)).any().item())


def span_mask_from_words(
    start: int,
    end: int,
    word_ids: list[int | None],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    for token_index, word_id in enumerate(word_ids[: attention_mask.numel()]):
        if word_id is not None and start <= int(word_id) < end:
            mask[token_index] = 1.0
    return mask


def f1_from_counts(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def oracle_ceiling(
    correct: int,
    gold: int,
    stage1_predicted: int,
) -> dict[str, object]:
    """Report both a perfect-reject ceiling and fixed-size engineering F1."""

    fixed_correct = min(int(correct), int(stage1_predicted), int(gold))
    return {
        "covered": int(correct),
        "coverage": correct / max(gold, 1),
        "perfect_reject": {
            "interpretation": "Oracle emits exactly the covered gold candidates.",
            **f1_from_counts(correct, correct, gold),
            "predicted": int(correct),
        },
        "fixed_stage1_prediction_count": {
            "interpretation": "Candidate upper bound if prediction count stays fixed.",
            **f1_from_counts(fixed_correct, stage1_predicted, gold),
            "correct": fixed_correct,
            "predicted": int(stage1_predicted),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze record-level CRF k-best GMNER candidate ceilings"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--k-best", type=int, default=6)
    parser.add_argument("--max-span-candidates", type=int, default=12)
    parser.add_argument("--boundary-shift", type=int, default=1)
    parser.add_argument("--boundary-penalty", type=float, default=0.25)
    parser.add_argument("--max-span-length", type=int, default=10)
    parser.add_argument("--top-m-types", type=parse_int_list, default=parse_int_list("1,2,3"))
    parser.add_argument("--top-r-regions", type=parse_int_list, default=parse_int_list("1,5,10,16"))
    parser.add_argument(
        "--enforce-bio-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-stage1-evaluation", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-error-examples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def grounding_candidate_scores(
    *,
    model: GMNERModel,
    outputs: dict[str, torch.Tensor],
    batch: dict,
    index: int,
    target_mask: torch.Tensor,
    config,
) -> torch.Tensor:
    entity_states = outputs["pre_prototype_fused_tokens"][index : index + 1]
    query = (entity_states * target_mask.unsqueeze(-1)).sum(dim=1) / target_mask.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0)
    logits = model.grounding_head(
        query=query,
        image_nodes=outputs["image_nodes"][index : index + 1],
        image_mask=batch["region_mask"][index : index + 1],
    )[0]

    region_score_weight = float(getattr(config.model, "region_score_prior_weight", 0.0))
    if region_score_weight:
        detector_scores = batch["region_scores"][index].to(
            device=logits.device,
            dtype=logits.dtype,
        ).clamp(1e-4, 1.0)
        score_bias = torch.log(detector_scores) * region_score_weight
        if config.data.add_null_region and score_bias.numel() > 0:
            score_bias = score_bias.clone()
            score_bias[-1] = 0.0
        logits = logits + score_bias
    null_bias = float(getattr(config.model, "grounding_null_logit_bias", 0.0))
    if null_bias and config.data.add_null_region and logits.numel() > 0:
        logits = logits.clone()
        logits[-1] += null_bias
    return logits


def append_error_example(
    examples: dict[str, list[dict]],
    category: str,
    *,
    limit: int,
    record_id: str,
    text: str,
    gold_entity: dict,
    candidate_boundaries: list[tuple[int, int]],
) -> None:
    if len(examples[category]) >= limit:
        return
    examples[category].append(
        {
            "record_id": record_id,
            "text": text,
            "gold": gold_entity,
            "span_candidates": candidate_boundaries,
        }
    )


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    if config.data.semantic_prototype_path:
        config.data.semantic_prototype_path = str(
            resolve_path(config.data.semantic_prototype_path, project_root)
        )
    if config.data.external_knowledge_prototype_path:
        config.data.external_knowledge_prototype_path = str(
            resolve_path(config.data.external_knowledge_prototype_path, project_root)
        )

    device_name = args.device or config.runtime.device
    device = torch.device(
        device_name
        if device_name.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    backbone_config = AutoConfig.from_pretrained(config.model.text_model_name)
    validate_model_input_length(
        tokenizer,
        backbone_config,
        config.data.max_length,
    )
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=config.data.use_dependency_graph,
            dependency_backend=config.data.dependency_backend,
            dependency_model=config.data.dependency_model,
            window_size=config.data.graph_window_size,
        )
    )

    split_file = {
        "train": config.data.train_file,
        "dev": config.data.dev_file,
        "test": config.data.test_file,
    }[args.split]
    output_root = Path(config.runtime.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = maybe_convert_conll(resolve_path(split_file, project_root), output_root)
    dataset = MMNERJsonDataset(
        jsonl_path=str(data_path),
        image_dir=str(resolve_path(config.data.image_dir, project_root)),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=True,
        # Candidate analysis is record-level; one copy still carries every NER
        # label and every XML box required by predicted-span evaluation.
        expand_entities_for_grounding=False,
        image_feature_dir=str(resolve_path(config.data.image_feature_dir, project_root)),
        image_annotation_dir=str(resolve_path(config.data.image_annotation_dir, project_root)),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        region_min_score=config.data.region_min_score,
    )
    analysis_dataset = dataset
    if args.max_records is not None:
        limit = max(0, min(int(args.max_records), len(dataset)))
        analysis_dataset = Subset(dataset, range(limit))
    loader = DataLoader(
        analysis_dataset,
        batch_size=args.batch_size or config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=GMNERCollator(tokenizer=tokenizer),
    )

    model = GMNERModel(config=config, num_labels=9)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    stage1_metrics: dict[str, float] = {}
    if not args.skip_stage1_evaluation:
        evaluated = evaluate_model(model=model, dataloader=loader, device=device)
        stage1_keys = (
            "entity_precision",
            "entity_recall",
            "entity_f1",
            "gmner_score",
            "triple_precision",
            "triple_recall",
            "triple_f1",
            "triple_correct",
            "triple_predict",
            "triple_gold",
            "eeg_precision",
            "eeg_recall",
            "eeg_f1",
        )
        stage1_metrics = {
            key: float(evaluated[key]) for key in stage1_keys if key in evaluated
        }

    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    num_tags = len(id2label)
    transitions, start_transitions, end_transitions, transition_source = (
        extract_crf_parameters(
            model.ner_head.crf,
            num_tags,
            device=device,
            dtype=torch.float32,
        )
    )
    allowed_start = None
    allowed_transitions = None
    if args.enforce_bio_constraints:
        allowed_start, allowed_transitions = bio_constraint_masks(
            id2label,
            num_tags,
            device=device,
        )

    top_ms = sorted({min(value, 4) for value in args.top_m_types})
    top_rs = sorted({min(value, config.data.max_regions) for value in args.top_r_regions})
    grid = {
        f"M{top_m}_R{top_r}": 0 for top_m in top_ms for top_r in top_rs
    }
    type_counts = {top_m: 0 for top_m in top_ms}
    typed_span_counts = {top_m: 0 for top_m in top_ms}
    region_counts = {
        top_r: {"covered": 0, "visible_covered": 0, "null_covered": 0}
        for top_r in top_rs
    }
    error_attribution = Counter()
    examples = {
        "span_missing": [],
        "type_missing": [],
        "region_missing": [],
    }

    records = 0
    total_gold = 0
    total_predicted = 0
    visible_gold = 0
    null_gold = 0
    actual_span_covered = 0
    rank1_span_covered = 0
    raw_kbest_span_covered = 0
    final_span_covered = 0
    new_gold_from_later_sequences = 0
    new_gold_from_perturbation = 0
    gold_lost_by_truncation = 0
    stage1_gold_lost_by_final = 0
    top1_sequence_matches = 0
    returned_sequences = 0
    sequence_diversity_sum = 0.0
    raw_unique_span_sum = 0
    final_span_sum = 0
    source_counts = Counter()
    first_gold_rank = Counter()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        labels = batch["ner_labels"]
        predicted = model.ner_head.decode(
            outputs["ner_logits"],
            batch["attention_mask"],
            valid_mask=labels != IGNORE_INDEX,
        )

        for index, meta in enumerate(batch.get("metadata", [])):
            records += 1
            record_id = str(meta.get("record_id", meta.get("sample_id")))
            tokens = meta.get("tokens") or []
            word_ids = meta.get("word_ids") or []
            word_pred = word_labels_from_subwords(predicted[index].tolist(), word_ids)
            word_gold = word_labels_from_subwords(labels[index].tolist(), word_ids)
            pred_entities = extract_entities_from_word_labels(word_pred, tokens, id2label)
            gold_entities = extract_entities_from_word_labels(word_gold, tokens, id2label)
            total_predicted += len(pred_entities)
            total_gold += len(gold_entities)
            actual_spans = {
                (int(entity["start"]), int(entity["end"])) for entity in pred_entities
            }

            valid_mask = batch["attention_mask"][index].bool() & (
                labels[index] != IGNORE_INDEX
            )
            positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            compact_positions: list[int] = []
            compact_word_indices: list[int] = []
            seen_word_indices: set[int] = set()
            for position in positions.tolist():
                word_id = word_ids[position] if position < len(word_ids) else None
                if word_id is None or int(word_id) in seen_word_indices:
                    continue
                seen_word_indices.add(int(word_id))
                compact_positions.append(position)
                compact_word_indices.append(int(word_id))

            sequences = []
            if compact_positions:
                emissions = outputs["base_ner_logits"][
                    index,
                    torch.tensor(compact_positions, device=device),
                ].float()
                sequences = k_best_viterbi_decode(
                    emissions,
                    k=max(1, int(args.k_best)),
                    transitions=transitions,
                    start_transitions=start_transitions,
                    end_transitions=end_transitions,
                    allowed_start=allowed_start,
                    allowed_transitions=allowed_transitions,
                )
            returned_sequences += len(sequences)
            sequence_diversity_sum += sequence_hamming_diversity(sequences)

            if sequences:
                actual_compact = tuple(
                    int(predicted[index, position].item()) for position in compact_positions
                )
                top1_sequence_matches += int(sequences[0].tag_ids == actual_compact)

            sequence_span_sets: list[set[tuple[int, int]]] = []
            for sequence in sequences:
                sequence_span_sets.append(
                    {
                        (start, end)
                        for start, end, _ in extract_word_spans(
                            sequence.tag_ids,
                            compact_word_indices,
                            id2label,
                        )
                    }
                )
            rank1_spans = sequence_span_sets[0] if sequence_span_sets else set()
            raw_kbest_spans = set().union(*sequence_span_sets) if sequence_span_sets else set()
            raw_unique_span_sum += len(raw_kbest_spans)

            num_words = max(compact_word_indices, default=-1) + 1
            candidates = build_span_candidates(
                sequences,
                word_indices=compact_word_indices,
                id2label=id2label,
                num_words=num_words,
                max_candidates=max(1, int(args.max_span_candidates)),
                boundary_shift=max(0, int(args.boundary_shift)),
                boundary_penalty=float(args.boundary_penalty),
                max_span_length=max(1, int(args.max_span_length)),
                required_spans=sorted(actual_spans),
            )
            final_spans = {candidate.boundary for candidate in candidates}
            final_span_sum += len(candidates)
            source_counts.update(candidate.source for candidate in candidates)

            for gold_entity in gold_entities:
                span = (int(gold_entity["start"]), int(gold_entity["end"]))
                has_actual_span = span in actual_spans
                has_rank1_span = span in rank1_spans
                has_raw_span = span in raw_kbest_spans
                has_final_span = span in final_spans
                actual_span_covered += int(has_actual_span)
                rank1_span_covered += int(has_rank1_span)
                raw_kbest_span_covered += int(has_raw_span)
                final_span_covered += int(has_final_span)
                new_gold_from_later_sequences += int(not has_rank1_span and has_raw_span)
                new_gold_from_perturbation += int(not has_raw_span and has_final_span)
                gold_lost_by_truncation += int(has_raw_span and not has_final_span)
                stage1_gold_lost_by_final += int(has_actual_span and not has_final_span)
                for rank, spans in enumerate(sequence_span_sets, start=1):
                    if span in spans:
                        first_gold_rank[rank] += 1
                        break

                target_mask = span_mask_from_words(
                    start=span[0],
                    end=span[1],
                    word_ids=word_ids,
                    attention_mask=batch["attention_mask"][index],
                ).unsqueeze(0)
                type_logits = model._span_type_logits_from_ner(
                    outputs["base_ner_logits"][index : index + 1],
                    target_mask,
                )[0]
                gold_type = ENTITY_TYPE2ID[str(gold_entity["type"])]
                type_ranking = type_logits.argsort(descending=True).tolist()
                type_covered: dict[int, bool] = {}
                for top_m in top_ms:
                    type_covered[top_m] = gold_type in type_ranking[:top_m]
                    type_counts[top_m] += int(type_covered[top_m])
                    typed_span_counts[top_m] += int(
                        has_final_span and type_covered[top_m]
                    )

                region_logits = grounding_candidate_scores(
                    model=model,
                    outputs=outputs,
                    batch=batch,
                    index=index,
                    target_mask=target_mask,
                    config=config,
                )
                gold_name = str(gold_entity["text"]).strip().lower()
                gt_boxes = (meta.get("gt_boxes_by_name") or {}).get(gold_name, [])
                is_visible = bool(gt_boxes)
                visible_gold += int(is_visible)
                null_gold += int(not is_visible)
                region_covered: dict[int, bool] = {}
                for top_r in top_rs:
                    covered = region_candidate_covered(
                        gt_boxes=gt_boxes,
                        region_boxes=batch["region_boxes"][index],
                        ranking_scores=region_logits,
                        region_mask=batch["region_mask"][index],
                        top_r=top_r,
                        iou_threshold=config.data.grounding_iou_threshold,
                        has_null_region=config.data.add_null_region,
                    )
                    region_covered[top_r] = covered
                    region_counts[top_r]["covered"] += int(covered)
                    if is_visible:
                        region_counts[top_r]["visible_covered"] += int(covered)
                    else:
                        region_counts[top_r]["null_covered"] += int(covered)

                for top_m in top_ms:
                    for top_r in top_rs:
                        if has_final_span and type_covered[top_m] and region_covered[top_r]:
                            grid[f"M{top_m}_R{top_r}"] += 1

                max_type_ok = type_covered[max(top_ms)]
                max_region_ok = region_covered[max(top_rs)]
                if not has_final_span:
                    category = "span_missing"
                elif not max_type_ok:
                    category = "type_missing"
                elif not max_region_ok:
                    category = "region_missing"
                else:
                    category = "all_candidates_available"
                error_attribution[category] += 1
                if category in examples:
                    append_error_example(
                        examples,
                        category,
                        limit=max(0, int(args.max_error_examples)),
                        record_id=record_id,
                        text=str(meta.get("text") or " ".join(tokens)),
                        gold_entity=gold_entity,
                        candidate_boundaries=sorted(final_spans),
                    )

    stage1_predicted = int(
        round(stage1_metrics.get("triple_predict", float(total_predicted)))
    )
    crf_backend = (
        f"{type(model.ner_head.crf).__module__}.{type(model.ner_head.crf).__name__}"
        if model.ner_head.crf is not None
        else "none"
    )
    result: dict[str, object] = {
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "candidate_budget": {
            "k_best_sequences": int(args.k_best),
            "max_span_candidates": int(args.max_span_candidates),
            "top_m_types": top_ms,
            "top_r_regions": top_rs,
            "boundary_shift": int(args.boundary_shift),
            "boundary_penalty": float(args.boundary_penalty),
            "max_span_length": int(args.max_span_length),
        },
        "records": records,
        "gold_entities": total_gold,
        "visible_gold_entities": visible_gold,
        "null_gold_entities": null_gold,
        "stage1": {
            "metrics": stage1_metrics,
            "decoded_entities": total_predicted,
        },
        "crf_kbest": {
            "backend": crf_backend,
            "transition_source": transition_source,
            "bio_constraints": bool(args.enforce_bio_constraints),
            "average_returned_sequences": returned_sequences / max(records, 1),
            "sequence_shortfall_ratio": 1.0
            - returned_sequences / max(records * max(int(args.k_best), 1), 1),
            "top1_matches_stage1_decode_ratio": top1_sequence_matches
            / max(records, 1),
            "mean_pairwise_hamming_diversity": sequence_diversity_sum
            / max(records, 1),
            "average_raw_unique_spans": raw_unique_span_sum / max(records, 1),
            "average_final_span_candidates": final_span_sum / max(records, 1),
            "final_candidate_sources": dict(source_counts),
            "gold_span_first_sequence_rank": {
                str(rank): count for rank, count in sorted(first_gold_rank.items())
            },
        },
        "span_coverage": {
            "single_stage1_decode": actual_span_covered / max(total_gold, 1),
            "kbest_rank1": rank1_span_covered / max(total_gold, 1),
            "raw_kbest_union": raw_kbest_span_covered / max(total_gold, 1),
            "final_at_budget": final_span_covered / max(total_gold, 1),
            "new_gold_from_ranks_2_to_k": new_gold_from_later_sequences,
            "new_gold_from_boundary_perturbation": new_gold_from_perturbation,
            "gold_lost_by_candidate_truncation": gold_lost_by_truncation,
            "stage1_gold_lost_by_final_candidates": stage1_gold_lost_by_final,
        },
        "type_coverage": {
            f"M{top_m}": {
                "given_gold_span": type_counts[top_m] / max(total_gold, 1),
                "typed_span_recall": typed_span_counts[top_m] / max(total_gold, 1),
            }
            for top_m in top_ms
        },
        "region_coverage": {
            f"R{top_r}": {
                "overall": values["covered"] / max(total_gold, 1),
                "visible": values["visible_covered"] / max(visible_gold, 1),
                "null": values["null_covered"] / max(null_gold, 1),
            }
            for top_r, values in region_counts.items()
        },
        "error_attribution_at_max_budget": {
            "budget": {
                "spans": int(args.max_span_candidates),
                "types": max(top_ms),
                "regions": max(top_rs),
            },
            "counts": {
                key: int(error_attribution.get(key, 0))
                for key in (
                    "span_missing",
                    "type_missing",
                    "region_missing",
                    "all_candidates_available",
                )
            },
        },
        "oracle_ceiling": {
            "span": oracle_ceiling(
                final_span_covered,
                total_gold,
                stage1_predicted,
            ),
            "typed_span": {
                f"M{top_m}": oracle_ceiling(
                    typed_span_counts[top_m],
                    total_gold,
                    stage1_predicted,
                )
                for top_m in top_ms
            },
            "triple": {
                key: oracle_ceiling(correct, total_gold, stage1_predicted)
                for key, correct in grid.items()
            },
        },
        "error_examples": examples,
        "checkpoint_missing_keys": list(incompatible.missing_keys),
        "checkpoint_unexpected_keys": list(incompatible.unexpected_keys),
    }
    return result


if __name__ == "__main__":
    arguments = parse_args()
    report = analyze(arguments)
    output_path = Path(arguments.output) if arguments.output else Path(
        load_config(arguments.config).runtime.output_dir
    ) / f"{arguments.split}_record_candidate_oracle.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_to={output_path.resolve()}")
