from __future__ import annotations

import torch

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID, IGNORE_INDEX
from gmner.data.record_level_stage1_collator import (
    RecordLevelStage1Collator,
)
from gmner.data.record_level_stage1_dataset import (
    record_from_expanded_samples,
)
from gmner.data.stage1_record_contract import (
    build_word_subword_mapping,
    entity_encoding_mask,
    validate_stage1_record,
    word_spans_to_subword_masks,
)


class _PadTokenizer:
    is_fast = True

    class _Encoding(dict):
        def __init__(self, input_ids, word_ids):
            super().__init__(
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
            )
            self._word_ids = word_ids

        def word_ids(self):
            return self._word_ids

    def __call__(
        self,
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=256,
    ):
        del is_split_into_words, truncation
        pieces = []
        word_ids = []
        for index, word in enumerate(words):
            count = 2 if word == "split" else 1
            pieces.extend([index + 4] * count)
            word_ids.extend([index] * count)
        budget = max_length - 2
        return self._Encoding(
            [0] + pieces[:budget] + [2],
            [None] + word_ids[:budget] + [None],
        )

    def pad(self, inputs, padding=True, return_tensors="pt"):
        del padding, return_tensors
        max_length = max(len(item["input_ids"]) for item in inputs)
        output = {
            "input_ids": torch.zeros(
                len(inputs), max_length, dtype=torch.long
            ),
            "attention_mask": torch.zeros(
                len(inputs), max_length, dtype=torch.long
            ),
        }
        for row, item in enumerate(inputs):
            length = len(item["input_ids"])
            output["input_ids"][row, :length] = torch.tensor(
                item["input_ids"]
            )
            output["attention_mask"][row, :length] = torch.tensor(
                item["attention_mask"]
            )
        return output


def _record() -> dict:
    word_ids = [None, 0, 1, 1, 2, None]
    alignment = build_word_subword_mapping(
        word_ids,
        word_count=3,
        word_complete_mask=[True, True, True],
    )
    spans = torch.tensor([[0, 1], [1, 3]], dtype=torch.long)
    subword_masks = word_spans_to_subword_masks(
        spans,
        alignment["subword_to_word"],
    )
    entity_mask = entity_encoding_mask(
        spans,
        alignment["word_complete_mask"],
        subword_masks,
    )
    return {
        "record_id": "r0",
        "input_ids": torch.tensor([0, 4, 5, 6, 7, 2]),
        "attention_mask": torch.ones(6, dtype=torch.bool),
        "token_type_ids": None,
        "adjacency": torch.eye(6),
        "word_count": 3,
        **alignment,
        "typed_bio_labels": torch.tensor(
            [
                DEFAULT_LABEL2ID["B-PER"],
                DEFAULT_LABEL2ID["B-LOC"],
                DEFAULT_LABEL2ID["I-LOC"],
            ]
        ),
        "legacy_ner_labels": torch.tensor(
            [
                IGNORE_INDEX,
                DEFAULT_LABEL2ID["B-PER"],
                DEFAULT_LABEL2ID["B-LOC"],
                IGNORE_INDEX,
                DEFAULT_LABEL2ID["I-LOC"],
                IGNORE_INDEX,
            ]
        ),
        "region_features": torch.randn(3, 4),
        "region_boxes": torch.zeros(3, 4),
        "region_mask": torch.tensor([True, True, True]),
        "region_scores": torch.tensor([0.9, 1.0, 0.8]),
        "null_region_index": 1,
        "region_is_null": torch.tensor([False, True, False]),
        "gold_spans": spans,
        "gold_type_ids": torch.tensor(
            [ENTITY_TYPE2ID["PER"], ENTITY_TYPE2ID["LOC"]]
        ),
        "gold_entity_mask": entity_mask,
        "grounding_entity_mask": entity_mask.clone(),
        "type_entity_mask": entity_mask.clone(),
        "gold_subword_masks": subword_masks,
        "gold_region_labels": torch.tensor([0, 1]),
        "gold_region_positive_mask": torch.tensor(
            [[True, False, False], [False, True, False]]
        ),
        "gold_region_iou_targets": torch.tensor(
            [[0.8, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        "grounding_null_prior": torch.tensor([0.2, 0.8]),
        "metadata": {
            "record_id": "r0",
            "tokens": ["one", "two", "three"],
            "region_object_labels": ["person", "NULL", "street"],
            "region_object_attributes": ["", "", ""],
        },
    }


def test_record_contract_uses_explicit_nonterminal_null() -> None:
    record = validate_stage1_record(_record())
    assert record["null_region_index"] == 1
    assert record["region_is_null"].tolist() == [False, True, False]


def test_truncated_entity_is_masked_not_relabelled_as_null() -> None:
    alignment = build_word_subword_mapping(
        [None, 0, 1, None],
        word_count=3,
        word_complete_mask=[True, False, False],
    )
    spans = torch.tensor([[0, 1], [1, 3]])
    subword_masks = word_spans_to_subword_masks(
        spans,
        alignment["subword_to_word"],
    )
    valid = entity_encoding_mask(
        spans,
        alignment["word_complete_mask"],
        subword_masks,
    )
    assert valid.tolist() == [True, False]
    assert subword_masks[1].any()


def test_collator_preserves_entity_region_and_alignment_axes() -> None:
    batch = RecordLevelStage1Collator(_PadTokenizer())([_record()])
    assert batch["gold_subword_masks"].shape == (1, 2, 6)
    assert batch["gold_region_positive_mask"].shape == (1, 2, 3)
    assert batch["null_region_index"].tolist() == [1]
    assert batch["word_count"].tolist() == [3]


def test_contract_rejects_implicit_or_multiple_null_regions() -> None:
    record = _record()
    record["region_is_null"] = torch.tensor([False, False, True])
    try:
        validate_stage1_record(record)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("Expected explicit NULL contract failure.")


def test_partial_entity_masks_all_of_its_boundary_words() -> None:
    tokenizer = _PadTokenizer()
    common = {
        "record_id": "truncated",
        "sample_id": "truncated",
        "input_ids": [0, 4, 5, 2],
        "attention_mask": [1, 1, 1, 1],
        "word_ids": [None, 0, 1, None],
        "ner_labels": [
            IGNORE_INDEX,
            DEFAULT_LABEL2ID["B-PER"],
            DEFAULT_LABEL2ID["I-PER"],
            IGNORE_INDEX,
        ],
        "adjacency": torch.eye(4),
        "tokens": ["one", "split"],
        "region_features": torch.randn(3, 4),
        "region_boxes": torch.zeros(3, 4),
        "region_mask": torch.ones(3),
        "region_scores": torch.ones(3),
        "target_start": 0,
        "target_end": 2,
        "target_type_id": ENTITY_TYPE2ID["PER"],
        "target_text": "one split",
        "region_labels": 2,
        "region_positive_mask": torch.tensor([0.0, 0.0, 1.0]),
        "region_iou_targets": torch.tensor([0.0, 0.0, 1.0]),
        "grounding_null_prior": 0.8,
        "region_object_labels": ["person", "street"],
        "region_object_attributes": ["", ""],
    }
    record = record_from_expanded_samples(
        [common],
        tokenizer=tokenizer,
        max_regions=2,
        add_null_region=True,
    )
    assert record["gold_entity_mask"].tolist() == [False]
    assert record["word_mask"].tolist() == [False, False]
    assert record["typed_bio_labels"].tolist() == [
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
