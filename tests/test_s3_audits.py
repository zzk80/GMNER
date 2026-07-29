from __future__ import annotations

import torch

from gmner.constants import ENTITY_TYPE2ID
from gmner.diagnostics import (
    audit_boundary_type_errors,
    audit_candidate_actionability,
    audit_truncation,
    ensure_s3_audit_split,
)


def _candidate_record() -> dict:
    return {
        "span_candidates": torch.tensor([[0, 1], [2, 3], [3, 4]]),
        "span_mask": torch.tensor([True, True, True]),
        "span_source_ids": torch.tensor([0, 2, 3]),
        "type_candidates": torch.tensor([[1, 0], [2, 0], [3, 1]]),
        "formal_candidate_mask": torch.tensor([True, False, False]),
        "metadata": {
            "record_id": "r",
            "candidate_sources": ["stage1", "kbest", "perturbation"],
            "stage1_predictions": [
                {
                    "span": [0, 1],
                    "type_id": ENTITY_TYPE2ID["LOC"],
                    "region_index": 0,
                },
                {
                    "span": [1, 3],
                    "type_id": ENTITY_TYPE2ID["ORG"],
                    "region_index": 1,
                },
            ],
            "gold_entities": [
                {
                    "span": [0, 1],
                    "type_id": ENTITY_TYPE2ID["PER"],
                },
                {
                    "span": [2, 3],
                    "type_id": ENTITY_TYPE2ID["ORG"],
                },
            ],
        },
    }


class _Encoding(dict):
    def __init__(self, input_ids, word_ids):
        super().__init__(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
        )
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class _ToyTokenizer:
    is_fast = True

    def __call__(
        self,
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=4,
    ):
        del is_split_into_words, truncation
        pieces = []
        ids = []
        for index, word in enumerate(words):
            count = 2 if word == "split" else 1
            pieces.extend([10 + index] * count)
            ids.extend([index] * count)
        content_budget = max_length - 2
        return _Encoding(
            [0] + pieces[:content_budget] + [2],
            [None] + ids[:content_budget] + [None],
        )


def test_boundary_type_audit_separates_error_sources() -> None:
    report = audit_boundary_type_errors(
        [_candidate_record()],
        split="dev",
    )
    assert report["counts"]["boundary_correct_type_wrong"] == 1
    assert report["counts"]["overlapping_boundary_error"] == 1
    assert report["test_accessed"] is False


def test_candidate_actionability_uses_existing_candidates_only() -> None:
    report = audit_candidate_actionability(
        [_candidate_record()],
        split="dev",
    )
    assert report["counts"]["exact_candidate_covered"] == 2
    assert report["counts"]["typed_exact_candidate_covered"] == 2
    assert report["counts"]["recoverable_nonformal_typed_span"] == 1


def test_truncation_audit_detects_partial_word_entity() -> None:
    records = [
        {
            "tokens": ["one", "split"],
            "ner_tags": [0, 1],
        }
    ]
    report = audit_truncation(
        records,
        tokenizer=_ToyTokenizer(),
        max_length=4,
        split="train",
    )
    assert report["counts"]["partially_truncated_entities"] == 1
    assert report["test_accessed"] is False


def test_p0_rejects_test_scope() -> None:
    try:
        ensure_s3_audit_split("test")
    except ValueError as error:
        assert "train and dev" in str(error)
    else:
        raise AssertionError("P0 unexpectedly accepted Test.")
