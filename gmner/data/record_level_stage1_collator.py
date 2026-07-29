"""Dynamic padding for the S3 record-level Stage1 contract."""

from __future__ import annotations

from typing import Any

import torch

from gmner.constants import ENTITY_TYPE2ID, IGNORE_INDEX
from gmner.data.stage1_record_contract import validate_stage1_record


class RecordLevelStage1Collator:
    """Pad records while retaining entity and NULL axes explicitly."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not records:
            raise ValueError("Cannot collate an empty S3 batch.")
        records = [validate_stage1_record(record) for record in records]
        tokenizer_inputs = []
        for record in records:
            item = {
                "input_ids": record["input_ids"].tolist(),
                "attention_mask": record["attention_mask"].long().tolist(),
            }
            if record.get("token_type_ids") is not None:
                item["token_type_ids"] = record[
                    "token_type_ids"
                ].tolist()
            tokenizer_inputs.append(item)
        padded = self.tokenizer.pad(
            tokenizer_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch_size, max_subwords = padded["input_ids"].shape
        max_words = max(int(record["word_count"]) for record in records)
        max_entities = max(
            int(record["gold_spans"].size(0)) for record in records
        )
        max_regions = max(
            int(record["region_features"].size(0)) for record in records
        )
        region_dims = {
            int(record["region_features"].size(1)) for record in records
        }
        if len(region_dims) != 1:
            raise ValueError("S3 records use inconsistent region dimensions.")
        region_dim = next(iter(region_dims))

        batch: dict[str, Any] = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"].bool(),
            "adjacency": torch.zeros(
                batch_size,
                max_subwords,
                max_subwords,
                dtype=torch.float32,
            ),
            "word_count": torch.zeros(batch_size, dtype=torch.long),
            "first_subword_indices": torch.full(
                (batch_size, max_words), -1, dtype=torch.long
            ),
            "word_to_subword_start": torch.full(
                (batch_size, max_words), -1, dtype=torch.long
            ),
            "word_to_subword_end": torch.full(
                (batch_size, max_words), -1, dtype=torch.long
            ),
            "subword_to_word": torch.full(
                (batch_size, max_subwords), -1, dtype=torch.long
            ),
            "word_complete_mask": torch.zeros(
                batch_size, max_words, dtype=torch.bool
            ),
            "word_mask": torch.zeros(
                batch_size, max_words, dtype=torch.bool
            ),
            "typed_bio_labels": torch.full(
                (batch_size, max_words),
                IGNORE_INDEX,
                dtype=torch.long,
            ),
            "legacy_ner_labels": torch.full(
                (batch_size, max_subwords),
                IGNORE_INDEX,
                dtype=torch.long,
            ),
            "region_features": torch.zeros(
                batch_size,
                max_regions,
                region_dim,
                dtype=torch.float32,
            ),
            "region_boxes": torch.zeros(
                batch_size, max_regions, 4, dtype=torch.float32
            ),
            "region_mask": torch.zeros(
                batch_size, max_regions, dtype=torch.bool
            ),
            "region_scores": torch.zeros(
                batch_size, max_regions, dtype=torch.float32
            ),
            "null_region_index": torch.full(
                (batch_size,), -1, dtype=torch.long
            ),
            "region_is_null": torch.zeros(
                batch_size, max_regions, dtype=torch.bool
            ),
            "gold_spans": torch.zeros(
                batch_size, max_entities, 2, dtype=torch.long
            ),
            "gold_type_ids": torch.full(
                (batch_size, max_entities),
                ENTITY_TYPE2ID["O"],
                dtype=torch.long,
            ),
            "gold_entity_mask": torch.zeros(
                batch_size, max_entities, dtype=torch.bool
            ),
            "grounding_entity_mask": torch.zeros(
                batch_size, max_entities, dtype=torch.bool
            ),
            "type_entity_mask": torch.zeros(
                batch_size, max_entities, dtype=torch.bool
            ),
            "gold_subword_masks": torch.zeros(
                batch_size,
                max_entities,
                max_subwords,
                dtype=torch.bool,
            ),
            "gold_region_labels": torch.full(
                (batch_size, max_entities),
                IGNORE_INDEX,
                dtype=torch.long,
            ),
            "gold_region_positive_mask": torch.zeros(
                batch_size,
                max_entities,
                max_regions,
                dtype=torch.bool,
            ),
            "gold_region_iou_targets": torch.zeros(
                batch_size,
                max_entities,
                max_regions,
                dtype=torch.float32,
            ),
            "grounding_null_prior": torch.full(
                (batch_size, max_entities),
                0.5,
                dtype=torch.float32,
            ),
            "metadata": [],
        }
        if "token_type_ids" in padded:
            batch["token_type_ids"] = padded["token_type_ids"]

        word_keys = (
            "first_subword_indices",
            "word_to_subword_start",
            "word_to_subword_end",
            "word_complete_mask",
            "word_mask",
            "typed_bio_labels",
        )
        entity_keys = (
            "gold_type_ids",
            "gold_entity_mask",
            "grounding_entity_mask",
            "type_entity_mask",
            "gold_region_labels",
            "grounding_null_prior",
        )
        for row, record in enumerate(records):
            subwords = int(record["input_ids"].numel())
            words = int(record["word_count"])
            entities = int(record["gold_spans"].size(0))
            regions = int(record["region_features"].size(0))
            batch["adjacency"][
                row, :subwords, :subwords
            ] = record["adjacency"]
            batch["word_count"][row] = words
            for key in word_keys:
                batch[key][row, :words] = record[key]
            batch["subword_to_word"][
                row, :subwords
            ] = record["subword_to_word"]
            batch["legacy_ner_labels"][
                row, :subwords
            ] = record["legacy_ner_labels"]
            for key in (
                "region_features",
                "region_boxes",
                "region_mask",
                "region_scores",
                "region_is_null",
            ):
                batch[key][row, :regions] = record[key]
            batch["null_region_index"][row] = int(
                record["null_region_index"]
            )
            batch["gold_spans"][row, :entities] = record["gold_spans"]
            for key in entity_keys:
                batch[key][row, :entities] = record[key]
            batch["gold_subword_masks"][
                row, :entities, :subwords
            ] = record["gold_subword_masks"]
            batch["gold_region_positive_mask"][
                row, :entities, :regions
            ] = record["gold_region_positive_mask"]
            batch["gold_region_iou_targets"][
                row, :entities, :regions
            ] = record["gold_region_iou_targets"]
            batch["metadata"].append(dict(record["metadata"]))
        return batch
