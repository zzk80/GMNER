"""Batch collation for multimodal data."""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from gmner.constants import IGNORE_INDEX


class GMNERCollator:
    """Pad token and precomputed region data for the formal GMNER path."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor | List[Dict]]:
        tokenizer_inputs: List[Dict] = []
        for feature in features:
            token_entry = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            if feature.get("token_type_ids") is not None:
                token_entry["token_type_ids"] = feature["token_type_ids"]
            tokenizer_inputs.append(token_entry)

        padded = self.tokenizer.pad(
            tokenizer_inputs,
            padding=True,
            return_tensors="pt",
        )

        batch_size, max_len = padded["input_ids"].shape
        target_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
        adjacency = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)

        has_cls_label = any("label" in feature for feature in features)
        cls_labels = torch.full((batch_size,), IGNORE_INDEX, dtype=torch.long)

        has_ner_label = any("ner_labels" in feature for feature in features)
        ner_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        ner_loss_weight = torch.ones((batch_size,), dtype=torch.float32)
        num_entities_in_record = torch.ones((batch_size,), dtype=torch.long)

        has_region_features = all("region_features" in feature for feature in features)
        if not has_region_features:
            raise ValueError(
                "GMNERCollator requires region_features for every sample. "
                "Raw-image ResNet collation has been removed."
            )
        region_features = None
        region_mask = None
        region_labels = None
        region_boxes = None
        region_scores = None
        region_positive_mask = None
        region_iou_targets = None
        image_sizes = None
        target_type_ids = None
        base_predicted_type_ids = None
        target_subtype_ids = None
        grounding_null_prior = None
        if has_region_features:
            max_regions = max(feature["region_features"].shape[0] for feature in features)
            feature_dim = features[0]["region_features"].shape[-1]
            region_features = torch.zeros((batch_size, max_regions, feature_dim), dtype=torch.float32)
            region_mask = torch.zeros((batch_size, max_regions), dtype=torch.float32)
            region_labels = torch.full((batch_size,), IGNORE_INDEX, dtype=torch.long)
            region_boxes = torch.zeros((batch_size, max_regions, 4), dtype=torch.float32)
            region_scores = torch.zeros((batch_size, max_regions), dtype=torch.float32)
            region_positive_mask = torch.zeros((batch_size, max_regions), dtype=torch.float32)
            region_iou_targets = torch.zeros((batch_size, max_regions), dtype=torch.float32)
            image_sizes = torch.zeros((batch_size, 2), dtype=torch.float32)
            target_type_ids = torch.full((batch_size,), 4, dtype=torch.long)
            base_predicted_type_ids = torch.full(
                (batch_size,),
                IGNORE_INDEX,
                dtype=torch.long,
            )
            target_subtype_ids = torch.full((batch_size,), IGNORE_INDEX, dtype=torch.long)
            grounding_null_prior = torch.full((batch_size,), 0.5, dtype=torch.float32)

        metadata: List[Dict] = []

        for batch_idx, feature in enumerate(features):
            feature_target_mask = torch.tensor(feature["target_mask"], dtype=torch.float32)
            valid_len = min(max_len, feature_target_mask.numel())
            target_mask[batch_idx, :valid_len] = feature_target_mask[:valid_len]

            feature_adjacency = feature["adjacency"]
            if not isinstance(feature_adjacency, torch.Tensor):
                feature_adjacency = torch.tensor(feature_adjacency, dtype=torch.float32)

            adj_len = min(max_len, feature_adjacency.size(0), feature_adjacency.size(1))
            adjacency[batch_idx, :adj_len, :adj_len] = feature_adjacency[:adj_len, :adj_len]

            if has_cls_label and "label" in feature:
                cls_labels[batch_idx] = int(feature["label"])

            if has_ner_label and "ner_labels" in feature:
                feature_ner_labels = torch.tensor(feature["ner_labels"], dtype=torch.long)
                ner_len = min(max_len, feature_ner_labels.numel())
                ner_labels[batch_idx, :ner_len] = feature_ner_labels[:ner_len]
                ner_loss_weight[batch_idx] = float(feature.get("ner_loss_weight", 1.0))
                num_entities_in_record[batch_idx] = int(
                    feature.get("num_entities_in_record", 1)
                )

            if has_region_features:
                feature_regions = torch.tensor(feature["region_features"], dtype=torch.float32)
                region_len = min(max_regions, feature_regions.size(0))
                region_features[batch_idx, :region_len] = feature_regions[:region_len]
                feature_mask = torch.tensor(feature.get("region_mask", [1.0] * region_len), dtype=torch.float32)
                region_mask[batch_idx, :region_len] = feature_mask[:region_len]
                if "region_labels" in feature:
                    region_labels[batch_idx] = int(feature["region_labels"])
                if "region_boxes" in feature:
                    feature_boxes = torch.tensor(feature["region_boxes"], dtype=torch.float32)
                    box_len = min(max_regions, feature_boxes.size(0))
                    region_boxes[batch_idx, :box_len] = feature_boxes[:box_len]
                if "region_scores" in feature:
                    feature_scores = torch.tensor(feature["region_scores"], dtype=torch.float32)
                    score_len = min(max_regions, feature_scores.numel())
                    region_scores[batch_idx, :score_len] = feature_scores[:score_len]
                if "region_positive_mask" in feature:
                    feature_positive_mask = torch.tensor(feature["region_positive_mask"], dtype=torch.float32)
                    positive_len = min(max_regions, feature_positive_mask.numel())
                    region_positive_mask[batch_idx, :positive_len] = feature_positive_mask[:positive_len]
                if "region_iou_targets" in feature:
                    feature_iou_targets = torch.tensor(
                        feature["region_iou_targets"],
                        dtype=torch.float32,
                    )
                    iou_len = min(max_regions, feature_iou_targets.numel())
                    region_iou_targets[batch_idx, :iou_len] = feature_iou_targets[:iou_len]
                if "image_size" in feature:
                    feature_image_size = torch.tensor(feature["image_size"], dtype=torch.float32)
                    if feature_image_size.numel() >= 2:
                        image_sizes[batch_idx] = feature_image_size[:2]
                if "target_type_id" in feature:
                    target_type_ids[batch_idx] = int(feature["target_type_id"])
                if "base_predicted_type_id" in feature:
                    base_predicted_type_ids[batch_idx] = int(
                        feature["base_predicted_type_id"]
                    )
                if "target_subtype_id" in feature:
                    target_subtype_ids[batch_idx] = int(feature["target_subtype_id"])
                if "grounding_null_prior" in feature:
                    grounding_null_prior[batch_idx] = float(feature["grounding_null_prior"])

            word_ids = feature.get("word_ids")
            if word_ids is None:
                word_ids = [None] * len(feature.get("input_ids", []))
            padded_word_ids = list(word_ids) + [None] * max(0, max_len - len(word_ids))
            metadata.append(
                {
                    "sample_id": feature.get("sample_id"),
                    "record_id": feature.get("record_id", feature.get("sample_id")),
                    "num_entities_in_record": feature.get("num_entities_in_record", 1),
                    "text": feature.get("text"),
                    "target": feature.get("target"),
                    "image_id": feature.get("image_id"),
                    "tokens": feature.get("tokens"),
                    "word_ids": padded_word_ids[:max_len],
                    "gt_boxes_by_name": feature.get("gt_boxes_by_name"),
                    "target_text": feature.get("target_text"),
                    "target_start": feature.get("target_start"),
                    "target_end": feature.get("target_end"),
                    "target_entity_type": feature.get("target_entity_type"),
                    "base_predicted_type": feature.get("base_predicted_type"),
                    "target_subtype": feature.get("target_subtype"),
                    "fine_ner_tags": feature.get("fine_ner_tags"),
                    "grounding_null_prior": feature.get("grounding_null_prior"),
                    "region_object_labels": feature.get("region_object_labels"),
                    "region_object_attributes": feature.get("region_object_attributes"),
                    "image_size": feature.get("image_size"),
                }
            )

        batch: Dict[str, torch.Tensor | List[Dict]] = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "target_mask": target_mask,
            "adjacency": adjacency,
            "metadata": metadata,
        }

        if "token_type_ids" in padded:
            batch["token_type_ids"] = padded["token_type_ids"]

        if has_cls_label:
            batch["labels"] = cls_labels
        if has_ner_label:
            batch["ner_labels"] = ner_labels
            batch["ner_loss_weight"] = ner_loss_weight
            batch["num_entities_in_record"] = num_entities_in_record

        if has_region_features:
            batch["region_features"] = region_features
            batch["region_mask"] = region_mask
            batch["region_labels"] = region_labels
            batch["region_boxes"] = region_boxes
            batch["region_scores"] = region_scores
            batch["region_positive_mask"] = region_positive_mask
            batch["region_iou_targets"] = region_iou_targets
            batch["image_sizes"] = image_sizes
            batch["target_type_ids"] = target_type_ids
            batch["base_predicted_type_ids"] = base_predicted_type_ids
            batch["target_subtype_ids"] = target_subtype_ids
            batch["grounding_null_prior"] = grounding_null_prior

        return batch
