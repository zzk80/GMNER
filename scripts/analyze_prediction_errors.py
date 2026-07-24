"""Analyze MNER / EEG / GMNER prediction errors by span, type, and region."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.constants import DEFAULT_LABEL2ID, IGNORE_INDEX
from gmner.data import GMNERCollator, MMNERJsonDataset, TextGraphBuilder
from gmner.data.graph_builders import GraphBuilderConfig
from gmner.engine.utils import move_batch_to_device
from gmner.knowledge.region_compatibility import compatibility_score
from gmner.models import GMNERModel
from gmner.models.common import masked_mean
from gmner.utils.io import ensure_dir, maybe_convert_conll
from gmner.utils.metrics import extract_entities_from_word_labels, word_labels_from_subwords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GMNER prediction errors")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args()


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def f1(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    score = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": score}


def region_is_correct(
    pred_region_index: int,
    gold_entity: dict[str, Any],
    gt_boxes_by_name: dict[str, list],
    region_boxes: torch.Tensor,
    no_region_index: int,
) -> bool:
    gold_name = str(gold_entity["text"]).strip().lower()
    gt_boxes = gt_boxes_by_name.get(gold_name, [])
    if not gt_boxes:
        return pred_region_index == no_region_index
    if pred_region_index == no_region_index:
        return False
    pred_box = region_boxes[pred_region_index].unsqueeze(0)
    gt_box_tensor = torch.tensor(gt_boxes, dtype=pred_box.dtype, device=pred_box.device)
    ious = box_iou(gt_box_tensor, pred_box).squeeze(1)
    return bool((ious > 0.5).any().item())


def region_debug_info(
    region_index: int,
    metadata: dict[str, Any],
    region_scores: torch.Tensor | None,
    region_boxes: torch.Tensor,
    gold_entity: dict[str, Any] | None,
    gt_boxes_by_name: dict[str, list],
    no_region_index: int,
) -> dict[str, Any]:
    labels = metadata.get("region_object_labels") or []
    attributes = metadata.get("region_object_attributes") or []
    item: dict[str, Any] = {
        "index": int(region_index),
        "is_null": int(region_index) == int(no_region_index),
    }
    if 0 <= region_index < len(labels):
        item["object"] = labels[region_index]
    if 0 <= region_index < len(attributes):
        item["attribute"] = attributes[region_index]
    if region_scores is not None and 0 <= region_index < region_scores.numel():
        item["detector_score"] = float(region_scores[region_index].item())
    if 0 <= region_index < region_boxes.size(0):
        item["box"] = [float(value) for value in region_boxes[region_index].detach().cpu().tolist()]

    if gold_entity is not None and not item["is_null"]:
        gold_name = str(gold_entity["text"]).strip().lower()
        gt_boxes = gt_boxes_by_name.get(gold_name, [])
        if gt_boxes:
            pred_box = region_boxes[region_index].unsqueeze(0)
            gt_box_tensor = torch.tensor(gt_boxes, dtype=pred_box.dtype, device=pred_box.device)
            ious = box_iou(gt_box_tensor, pred_box).squeeze(1)
            item["max_iou_with_gold"] = float(ious.max().item())
        else:
            item["max_iou_with_gold"] = None
    return item


@torch.no_grad()
def predict_region(
    model: GMNERModel,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    sample_index: int,
    pred_entity: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    metadata = batch["metadata"][sample_index]
    word_ids = metadata.get("word_ids") or []
    target_mask = torch.zeros_like(batch["attention_mask"][sample_index], dtype=torch.float32)
    for token_pos, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if int(pred_entity["start"]) <= word_id < int(pred_entity["end"]):
            target_mask[token_pos] = 1.0
    if target_mask.sum() == 0:
        target_mask = batch["attention_mask"][sample_index].float()

    source_tokens = outputs.get("pre_prototype_fused_tokens", outputs["fused_tokens"])
    entity_token_states = source_tokens[sample_index : sample_index + 1]
    entity_target_mask = target_mask.unsqueeze(0).to(
        device=entity_token_states.device,
        dtype=entity_token_states.dtype,
    )

    query = masked_mean(entity_token_states, entity_target_mask)
    reranker_query = model._entity_boundary_repr(entity_token_states, entity_target_mask)
    base_type_logits = model._span_type_logits_from_ner(
        outputs["base_ner_logits"][sample_index : sample_index + 1],
        entity_target_mask,
    )
    grounding_logits = model.grounding_head(
        query=query,
        image_nodes=outputs["image_nodes"][sample_index : sample_index + 1],
        image_mask=outputs["image_mask"][sample_index : sample_index + 1],
    )
    if (
        model.prototype_bank is not None
        and bool(
            getattr(
                model.config.model,
                "use_alignment_preserving_prototype_grounding",
                False,
            )
        )
    ):
        prototype_token_states, _ = model._apply_semantic_prototypes(
            token_states=entity_token_states,
            attention_mask=batch["attention_mask"][sample_index : sample_index + 1],
            target_masks=entity_target_mask,
        )
        prototype_query = masked_mean(prototype_token_states, entity_target_mask)
        prototype_grounding_logits = model.grounding_head(
            query=prototype_query,
            image_nodes=outputs["image_nodes"][sample_index : sample_index + 1],
            image_mask=outputs["image_mask"][sample_index : sample_index + 1],
        )
        grounding_logits = model._apply_alignment_preserving_grounding_delta(
            base_logits=grounding_logits,
            prototype_logits=prototype_grounding_logits,
            image_mask=outputs["image_mask"][sample_index : sample_index + 1],
        )
    mini_batch = {
        "metadata": [metadata],
    }
    region_scores_for_rerank = batch.get("region_scores")
    if region_scores_for_rerank is not None:
        mini_batch["region_scores"] = region_scores_for_rerank[sample_index : sample_index + 1]
    grounding_logits, _ = model._apply_grounding_reranker(
        logits=grounding_logits,
        entity_repr=reranker_query,
        image_nodes=outputs["image_nodes"][sample_index : sample_index + 1],
        image_mask=outputs["image_mask"][sample_index : sample_index + 1],
        batch=mini_batch,
        base_type_logits=base_type_logits,
        region_boxes=batch.get("region_boxes")[sample_index : sample_index + 1]
        if batch.get("region_boxes") is not None
        else None,
        image_sizes=batch.get("image_sizes")[sample_index : sample_index + 1]
        if batch.get("image_sizes") is not None
        else None,
    )
    region_scores = batch.get("region_scores")
    if region_scores is not None and model.config.model.region_score_prior_weight:
        scores = region_scores[sample_index].to(grounding_logits.dtype).clamp(1e-4, 1.0)
        score_bias = torch.log(scores) * model.config.model.region_score_prior_weight
        score_bias[-1] = 0.0
        grounding_logits = grounding_logits + score_bias.unsqueeze(0)
    null_logit_bias = float(getattr(model.config.model, "grounding_null_logit_bias", 0.0))
    if null_logit_bias and bool(getattr(model.config.data, "add_null_region", False)):
        grounding_logits = grounding_logits.clone()
        grounding_logits[:, -1] = grounding_logits[:, -1] + null_logit_bias
    compatibility_weight = float(getattr(model.config.model, "region_object_compatibility_weight", 0.0))
    if compatibility_weight:
        metadata = batch["metadata"][sample_index]
        labels = metadata.get("region_object_labels") or []
        attributes = metadata.get("region_object_attributes") or []
        compatibility_bias = torch.zeros_like(grounding_logits)
        region_count = min(len(labels), grounding_logits.size(1))
        if bool(getattr(model.config.data, "add_null_region", False)) and region_count == grounding_logits.size(1):
            region_count -= 1
        for region_idx in range(max(region_count, 0)):
            attribute = attributes[region_idx] if region_idx < len(attributes) else ""
            compatibility_bias[0, region_idx] = compatibility_score(
                pred_entity.get("type"),
                labels[region_idx],
                attribute,
            )
        grounding_logits = grounding_logits + compatibility_bias * compatibility_weight
    if model.entity_evidence_decoder is not None:
        base_type_logits = model._span_type_logits_from_ner(
            outputs["base_ner_logits"][sample_index : sample_index + 1],
            entity_target_mask,
        )
        context_repr = masked_mean(
            source_tokens[sample_index : sample_index + 1],
            batch["attention_mask"][sample_index : sample_index + 1].to(
                device=source_tokens.device,
                dtype=source_tokens.dtype,
            ),
        )
        evidence_batch = {"metadata": [metadata]}
        if region_scores is not None:
            evidence_batch["region_scores"] = region_scores[sample_index : sample_index + 1]
        grounding_logits = model.score_entity_evidence(
            entity_repr=query,
            context_repr=context_repr,
            image_nodes=outputs["image_nodes"][sample_index : sample_index + 1],
            image_mask=outputs["image_mask"][sample_index : sample_index + 1],
            base_grounding_logits=grounding_logits,
            base_type_logits=base_type_logits,
            batch=evidence_batch,
        )["grounding_logits"]
    top_k = min(5, grounding_logits.size(1))
    top_values, top_indices = torch.topk(grounding_logits.squeeze(0), k=top_k)
    metadata = batch["metadata"][sample_index]
    region_scores_for_sample = batch.get("region_scores")
    if region_scores_for_sample is not None:
        region_scores_for_sample = region_scores_for_sample[sample_index]
    top_regions = []
    no_region_index = grounding_logits.size(1) - 1
    for value, region_index in zip(top_values.tolist(), top_indices.tolist()):
        item = region_debug_info(
            region_index=region_index,
            metadata=metadata,
            region_scores=region_scores_for_sample,
            region_boxes=batch["region_boxes"][sample_index],
            gold_entity=None,
            gt_boxes_by_name={},
            no_region_index=no_region_index,
        )
        item["grounding_logit"] = float(value)
        top_regions.append(item)
    return int(top_indices[0].item()), top_regions


def build_dataset(config, tokenizer, graph_builder, project_root: Path, output_dir: Path, split: str):
    data_file = config.data.dev_file if split == "dev" else config.data.test_file
    data_path = maybe_convert_conll(resolve_path(data_file, project_root), output_dir)
    image_dir = resolve_path(config.data.image_dir, project_root)
    groundability_type_priors = (
        resolve_path(config.data.groundability_type_priors, project_root)
        if config.data.groundability_type_priors
        else None
    )
    groundability_mention_priors = (
        resolve_path(config.data.groundability_mention_priors, project_root)
        if config.data.groundability_mention_priors
        else None
    )
    return MMNERJsonDataset(
        jsonl_path=str(data_path),
        image_dir=str(image_dir),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=config.data.grounding_enabled,
        expand_entities_for_grounding=config.data.expand_entities_for_grounding,
        image_feature_dir=str(resolve_path(config.data.image_feature_dir, project_root)),
        image_annotation_dir=str(resolve_path(config.data.image_annotation_dir, project_root)),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=str(groundability_type_priors) if groundability_type_priors else None,
        groundability_mention_priors=str(groundability_mention_priors) if groundability_mention_priors else None,
        region_min_score=config.data.region_min_score,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    if config.data.semantic_prototype_path:
        config.data.semantic_prototype_path = str(resolve_path(config.data.semantic_prototype_path, project_root))
    if config.data.external_knowledge_prototype_path:
        config.data.external_knowledge_prototype_path = str(
            resolve_path(
                config.data.external_knowledge_prototype_path,
                project_root,
            )
        )

    output_path = (
        resolve_path(args.output, project_root)
        if args.output
        else Path(config.runtime.output_dir) / f"{args.split}_error_analysis.json"
    )
    output_dir = ensure_dir(output_path.parent)

    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name, use_fast=True)
    graph_builder_cfg = GraphBuilderConfig(
        use_dependency_graph=config.data.use_dependency_graph,
        dependency_backend=config.data.dependency_backend,
        dependency_model=config.data.dependency_model,
        window_size=config.data.graph_window_size,
    )
    graph_builder = TextGraphBuilder(graph_builder_cfg)
    dataset = build_dataset(config, tokenizer, graph_builder, project_root, output_dir, args.split)
    if config.model.use_subtype_auxiliary and config.model.num_subtypes <= 0:
        config.model.num_subtypes = len(getattr(dataset, "subtype_label2id", {}))
    dataloader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=GMNERCollator(tokenizer=tokenizer),
    )

    model = GMNERModel(config=config, num_labels=args.num_labels or 9)
    checkpoint = torch.load(resolve_path(args.checkpoint, project_root), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(device)
    model.eval()

    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    seen_records = set()
    counts = Counter()
    per_type = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        labels = batch["ner_labels"]
        pred_tokens = model.ner_head.decode(
            outputs["ner_logits"],
            batch["attention_mask"],
            valid_mask=labels != IGNORE_INDEX,
        )

        for idx, meta in enumerate(batch.get("metadata", [])):
            record_id = meta.get("sample_id")
            if record_id in seen_records:
                continue
            seen_records.add(record_id)

            tokens = meta.get("tokens") or []
            word_ids = meta.get("word_ids") or []
            gt_boxes_by_name = meta.get("gt_boxes_by_name") or {}
            word_pred = word_labels_from_subwords(pred_tokens[idx].tolist(), word_ids)
            word_gold = word_labels_from_subwords(labels[idx].tolist(), word_ids)
            pred_entities = extract_entities_from_word_labels(word_pred, tokens, id2label)
            gold_entities = extract_entities_from_word_labels(word_gold, tokens, id2label)

            counts["records"] += 1
            counts["pred_entities"] += len(pred_entities)
            counts["gold_entities"] += len(gold_entities)
            for gold in gold_entities:
                per_type[str(gold["type"])]["gold"] += 1
            for pred in pred_entities:
                per_type[str(pred["type"])]["pred"] += 1

            region_boxes = batch["region_boxes"][idx]
            no_region_index = int(region_boxes.size(0) - 1)
            matched_span = set()
            matched_mner = set()
            matched_eeg = set()
            matched_gmner = set()

            for pred in pred_entities:
                counts["pred_span_total"] += 1
                pred_span = (pred["start"], pred["end"])
                pred_region_index, top_regions = predict_region(model, outputs, batch, idx, pred)

                span_match_idx = None
                for gold_idx, gold in enumerate(gold_entities):
                    if gold_idx in matched_span:
                        continue
                    if (gold["start"], gold["end"]) == pred_span:
                        span_match_idx = gold_idx
                        break

                if span_match_idx is None:
                    counts["span_fp"] += 1
                    if len(examples["span_error"]) < args.max_examples:
                        examples["span_error"].append(
                            {"record_id": record_id, "text": " ".join(tokens), "pred": pred}
                        )
                    continue

                matched_span.add(span_match_idx)
                counts["span_correct"] += 1
                gold = gold_entities[span_match_idx]
                gold_type = str(gold["type"])
                pred_type = str(pred["type"])
                type_ok = pred_type == gold_type
                region_ok = region_is_correct(
                    pred_region_index,
                    gold,
                    gt_boxes_by_name,
                    region_boxes,
                    no_region_index,
                )

                per_type[gold_type]["span_correct"] += 1
                if type_ok:
                    counts["type_correct_given_span"] += 1
                    per_type[gold_type]["mner_correct"] += 1
                    matched_mner.add(span_match_idx)
                else:
                    counts["type_error_given_span"] += 1
                    pair_key = f"{gold_type}->{pred_type}"
                    counts[f"type_confusion:{pair_key}"] += 1
                    if len(examples["type_error"]) < args.max_examples:
                        examples["type_error"].append(
                            {
                                "record_id": record_id,
                                "text": " ".join(tokens),
                                "pred": pred,
                                "gold": gold,
                            }
                        )

                if region_ok:
                    counts["region_correct_given_span"] += 1
                    per_type[gold_type]["eeg_correct"] += 1
                    matched_eeg.add(span_match_idx)
                else:
                    counts["region_error_given_span"] += 1
                    has_gold_box = bool(gt_boxes_by_name.get(str(gold["text"]).strip().lower(), []))
                    pred_region_info = region_debug_info(
                        region_index=pred_region_index,
                        metadata=meta,
                        region_scores=batch.get("region_scores")[idx] if batch.get("region_scores") is not None else None,
                        region_boxes=region_boxes,
                        gold_entity=gold,
                        gt_boxes_by_name=gt_boxes_by_name,
                        no_region_index=no_region_index,
                    )
                    pred_object = str(pred_region_info.get("object", "NULL" if pred_region_info["is_null"] else ""))
                    pred_attribute = str(pred_region_info.get("attribute", ""))
                    counts[f"region_error_pred_object:{pred_object}"] += 1
                    if pred_attribute:
                        counts[f"region_error_pred_attribute:{pred_attribute}"] += 1
                    if has_gold_box and pred_region_index == no_region_index:
                        counts["region_visible_pred_null"] += 1
                    elif not has_gold_box and pred_region_index != no_region_index:
                        counts["region_invisible_pred_region"] += 1
                    else:
                        counts["region_wrong_box"] += 1
                    if len(examples["region_error"]) < args.max_examples:
                        gold_name = str(gold["text"]).strip().lower()
                        gt_boxes = gt_boxes_by_name.get(gold_name, [])
                        top_regions_with_iou = []
                        for item in top_regions:
                            enriched = dict(item)
                            if gt_boxes and not enriched.get("is_null", False):
                                box_tensor = torch.tensor(
                                    [enriched["box"]],
                                    dtype=region_boxes.dtype,
                                    device=region_boxes.device,
                                )
                                gt_box_tensor = torch.tensor(gt_boxes, dtype=region_boxes.dtype, device=region_boxes.device)
                                ious = box_iou(gt_box_tensor, box_tensor).squeeze(1)
                                enriched["max_iou_with_gold"] = float(ious.max().item())
                            else:
                                enriched["max_iou_with_gold"] = None
                            top_regions_with_iou.append(enriched)
                        examples["region_error"].append(
                            {
                                "record_id": record_id,
                                "text": " ".join(tokens),
                                "pred": pred,
                                "gold": gold,
                                "pred_region_index": pred_region_index,
                                "pred_region": pred_region_info,
                                "gold_has_box": has_gold_box,
                                "gold_boxes": gt_boxes,
                                "top_regions": top_regions_with_iou,
                            }
                        )

                if type_ok and region_ok:
                    counts["gmner_correct"] += 1
                    per_type[gold_type]["gmner_correct"] += 1
                    matched_gmner.add(span_match_idx)
                elif type_ok and not region_ok:
                    counts["gmner_lost_by_region"] += 1
                elif region_ok and not type_ok:
                    counts["gmner_lost_by_type"] += 1
                else:
                    counts["gmner_lost_by_type_and_region"] += 1

            counts["span_fn"] += len(gold_entities) - len(matched_span)
            counts["mner_fn"] += len(gold_entities) - len(matched_mner)
            counts["eeg_fn"] += len(gold_entities) - len(matched_eeg)
            counts["gmner_fn"] += len(gold_entities) - len(matched_gmner)

    span = f1(counts["span_correct"], counts["pred_entities"], counts["gold_entities"])
    mner = f1(counts["type_correct_given_span"], counts["pred_entities"], counts["gold_entities"])
    eeg = f1(counts["region_correct_given_span"], counts["pred_entities"], counts["gold_entities"])
    gmner = f1(counts["gmner_correct"], counts["pred_entities"], counts["gold_entities"])

    per_type_metrics = {}
    for entity_type, stat in sorted(per_type.items()):
        per_type_metrics[entity_type] = {
            "gold": int(stat["gold"]),
            "pred": int(stat["pred"]),
            "span_f1": f1(stat["span_correct"], stat["pred"], stat["gold"])["f1"],
            "mner_f1": f1(stat["mner_correct"], stat["pred"], stat["gold"])["f1"],
            "eeg_f1": f1(stat["eeg_correct"], stat["pred"], stat["gold"])["f1"],
            "gmner_f1": f1(stat["gmner_correct"], stat["pred"], stat["gold"])["f1"],
        }

    span_correct = max(counts["span_correct"], 1)
    analysis = {
        "summary": {
            "records": int(counts["records"]),
            "pred_entities": int(counts["pred_entities"]),
            "gold_entities": int(counts["gold_entities"]),
            "span": span,
            "mner": mner,
            "eeg": eeg,
            "gmner": gmner,
        },
        "conditional": {
            "type_accuracy_given_span": counts["type_correct_given_span"] / span_correct,
            "region_accuracy_given_span": counts["region_correct_given_span"] / span_correct,
            "gmner_accuracy_given_span": counts["gmner_correct"] / span_correct,
            "gmner_loss_region_only": counts["gmner_lost_by_region"] / span_correct,
            "gmner_loss_type_only": counts["gmner_lost_by_type"] / span_correct,
            "gmner_loss_type_and_region": counts["gmner_lost_by_type_and_region"] / span_correct,
        },
        "counts": {key: int(value) for key, value in counts.items()},
        "per_type": per_type_metrics,
        "top_type_confusions": {
            key.split("type_confusion:", 1)[1]: int(value)
            for key, value in counts.most_common()
            if key.startswith("type_confusion:")
        },
        "top_region_error_objects": {
            key.split("region_error_pred_object:", 1)[1]: int(value)
            for key, value in counts.most_common()
            if key.startswith("region_error_pred_object:")
        },
        "top_region_error_attributes": {
            key.split("region_error_pred_attribute:", 1)[1]: int(value)
            for key, value in counts.most_common()
            if key.startswith("region_error_pred_attribute:")
        },
        "examples": examples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(analysis, fp, ensure_ascii=False, indent=2)
    print(json.dumps(analysis["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(analysis["conditional"], ensure_ascii=False, indent=2))
    print(f"saved_to={output_path}")


if __name__ == "__main__":
    main()
