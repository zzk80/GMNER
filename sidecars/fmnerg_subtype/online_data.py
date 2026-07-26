"""Online tokenization and span batching for trainable subtype encoders."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import Dataset

from gmner.constants import ENTITY_TYPE2ID

from .data import FineRecord
from .taxonomy import SubtypeTaxonomy


def gold_online_records(
    records: list[FineRecord],
    taxonomy: SubtypeTaxonomy,
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": record.record_id,
            "tokens": list(record.tokens),
            "spans": [
                {
                    "start": entity.start,
                    "end": entity.end,
                    "coarse_type_id": ENTITY_TYPE2ID[entity.coarse_type],
                    "subtype_id": taxonomy.subtype_id(entity.subtype),
                    "subtype": entity.subtype,
                    "target_available": True,
                }
                for entity in record.entities
            ],
        }
        for record in records
    ]


def formal_online_records(
    formal_payload: dict[str, Any],
    fine_records: list[FineRecord],
    taxonomy: SubtypeTaxonomy,
) -> list[dict[str, Any]]:
    tokens_by_id = {
        record.record_id: list(record.tokens) for record in fine_records
    }
    gold_by_id = {
        record.record_id: {
            (entity.start, entity.end): entity for entity in record.entities
        }
        for record in fine_records
    }
    output: list[dict[str, Any]] = []
    for record_index, record in enumerate(formal_payload["records"]):
        record_id = str(record["record_id"])
        if record_id not in tokens_by_id:
            raise ValueError(
                f"Formal record {record_id!r} is absent from the fine source."
            )
        spans = []
        for prediction_index, prediction in enumerate(
            record.get("predictions") or []
        ):
            start, end = map(int, prediction["span"])
            gold_entity = gold_by_id[record_id].get((start, end))
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "coarse_type_id": int(prediction["type_id"]),
                    "subtype_id": (
                        taxonomy.subtype_id(gold_entity.subtype)
                        if gold_entity is not None
                        else -100
                    ),
                    "subtype": (
                        gold_entity.subtype if gold_entity is not None else None
                    ),
                    "target_available": gold_entity is not None,
                    "record_index": record_index,
                    "prediction_index": prediction_index,
                }
            )
        output.append(
            {
                "record_id": record_id,
                "tokens": tokens_by_id[record_id],
                "spans": spans,
            }
        )
    return output


class OnlineSubtypeRecordDataset(Dataset):
    """Record-level text dataset with globally indexed span examples."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records: list[dict[str, Any]] = []
        self.examples: list[dict[str, Any]] = []
        for raw_record in records:
            record = copy.deepcopy(raw_record)
            record_id = str(record["record_id"])
            tokens = list(record["tokens"])
            prepared_spans = []
            for span in record.get("spans") or []:
                start = int(span["start"])
                end = int(span["end"])
                if start < 0 or end <= start or end > len(tokens):
                    raise ValueError(
                        f"Invalid subtype span [{start}, {end}) in "
                        f"record {record_id}."
                    )
                example_index = len(self.examples)
                prepared = {
                    **span,
                    "start": start,
                    "end": end,
                    "record_id": record_id,
                    "text": " ".join(tokens[start:end]),
                    "example_index": example_index,
                }
                prepared_spans.append(prepared)
                self.examples.append(dict(prepared))
            if prepared_spans:
                self.records.append(
                    {
                        "record_id": record_id,
                        "tokens": tokens,
                        "spans": prepared_spans,
                    }
                )
        self.coarse_type_ids = torch.tensor(
            [int(example["coarse_type_id"]) for example in self.examples],
            dtype=torch.long,
        )
        self.subtype_ids = torch.tensor(
            [int(example["subtype_id"]) for example in self.examples],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class OnlineSubtypeCollator:
    def __init__(self, tokenizer: Any, *, max_length: int) -> None:
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise ValueError("Online subtype training requires a fast tokenizer.")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        encoding = self.tokenizer(
            [list(record["tokens"]) for record in records],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        sequence_length = int(encoding["input_ids"].size(1))
        span_record_indices: list[int] = []
        span_start_indices: list[int] = []
        span_end_indices: list[int] = []
        span_token_masks: list[torch.Tensor] = []
        coarse_type_ids: list[int] = []
        subtype_ids: list[int] = []
        example_indices: list[int] = []
        examples: list[dict[str, Any]] = []

        for row, record in enumerate(records):
            word_ids = list(encoding.word_ids(batch_index=row))
            for span in record["spans"]:
                start = int(span["start"])
                end = int(span["end"])
                positions = [
                    index
                    for index, word_id in enumerate(word_ids)
                    if word_id is not None and start <= int(word_id) < end
                ]
                start_positions = [
                    index
                    for index, word_id in enumerate(word_ids)
                    if word_id is not None and int(word_id) == start
                ]
                end_positions = [
                    index
                    for index, word_id in enumerate(word_ids)
                    if word_id is not None and int(word_id) == end - 1
                ]
                if not positions or not start_positions or not end_positions:
                    raise ValueError(
                        f"Span [{start}, {end}) in record "
                        f"{record['record_id']} was truncated at "
                        f"max_length={self.max_length}."
                    )
                mask = torch.zeros(sequence_length, dtype=torch.bool)
                mask[positions] = True
                span_record_indices.append(row)
                span_start_indices.append(start_positions[0])
                span_end_indices.append(end_positions[-1])
                span_token_masks.append(mask)
                coarse_type_ids.append(int(span["coarse_type_id"]))
                subtype_ids.append(int(span["subtype_id"]))
                example_indices.append(int(span["example_index"]))
                examples.append(dict(span))

        if not examples:
            raise ValueError("Online subtype batch contains no spans.")
        model_inputs = {
            key: value
            for key, value in encoding.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        return {
            **model_inputs,
            "span_record_indices": torch.tensor(
                span_record_indices, dtype=torch.long
            ),
            "span_start_indices": torch.tensor(
                span_start_indices, dtype=torch.long
            ),
            "span_end_indices": torch.tensor(
                span_end_indices, dtype=torch.long
            ),
            "span_token_mask": torch.stack(span_token_masks),
            "coarse_type_ids": torch.tensor(
                coarse_type_ids, dtype=torch.long
            ),
            "subtype_ids": torch.tensor(subtype_ids, dtype=torch.long),
            "example_indices": torch.tensor(
                example_indices, dtype=torch.long
            ),
            "examples": examples,
        }
