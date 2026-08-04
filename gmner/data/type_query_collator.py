"""Type-query record collation for TQ-DV-MNER."""

from __future__ import annotations

from typing import Any

import torch

from gmner.data.dvh_record_collator import DVHRecordCollator


TYPE_QUERIES: tuple[dict[str, Any], ...] = (
    {
        "type_id": 0,
        "name": "LOC",
        "text": "Location: countries, cities, towns and geographic places.",
    },
    {
        "type_id": 1,
        "name": "PER",
        "text": "Person: people's names and fictional characters.",
    },
    {
        "type_id": 2,
        "name": "ORG",
        "text": "Organization: companies, teams, governments and institutions.",
    },
    {
        "type_id": 3,
        "name": "OTHER",
        "text": "Other: named entities that are not people, locations or organizations.",
    },
)


class TypeQueryRecordCollator(DVHRecordCollator):
    """Build four query-sentence encodings and exact span supervision."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        max_span_length: int,
    ) -> None:
        super().__init__(tokenizer)
        self.max_length = int(max_length)
        self.max_span_length = int(max_span_length)
        if self.max_length <= 0 or self.max_span_length <= 0:
            raise ValueError("Query and span lengths must be positive.")

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(records)
        batch_size = len(records)
        type_count = len(TYPE_QUERIES)
        max_words = int(batch["word_mask"].size(1))

        flat_encodings: list[dict[str, list[int]]] = []
        alignment_rows: list[dict[str, torch.Tensor]] = []
        for record in records:
            tokens = [str(token) for token in record["metadata"]["tokens"]]
            for query in TYPE_QUERIES:
                encoding = self.tokenizer(
                    str(query["text"]).split(),
                    tokens,
                    is_split_into_words=True,
                    truncation="only_second",
                    max_length=self.max_length,
                )
                sequence_ids = list(encoding.sequence_ids())
                word_ids = list(encoding.word_ids())
                if len(sequence_ids) != len(encoding["input_ids"]):
                    raise ValueError("Query tokenizer sequence alignment mismatch.")
                item = {
                    key: list(value)
                    for key, value in encoding.items()
                }
                flat_encodings.append(item)
                first = torch.full((max_words,), -1, dtype=torch.long)
                word_mask = torch.zeros(max_words, dtype=torch.bool)
                query_mask = torch.zeros(len(sequence_ids), dtype=torch.bool)
                sentence_mask = torch.zeros(len(sequence_ids), dtype=torch.bool)
                for position, (sequence_id, word_id) in enumerate(
                    zip(sequence_ids, word_ids)
                ):
                    if sequence_id == 0:
                        query_mask[position] = True
                    elif sequence_id == 1:
                        sentence_mask[position] = True
                        if word_id is not None and 0 <= int(word_id) < max_words:
                            index = int(word_id)
                            word_mask[index] = True
                            if first[index] < 0:
                                first[index] = position
                alignment_rows.append(
                    {
                        "first": first,
                        "word_mask": word_mask,
                        "query_mask": query_mask,
                        "sentence_mask": sentence_mask,
                    }
                )

        padded = self.tokenizer.pad(
            flat_encodings,
            padding=True,
            return_tensors="pt",
        )
        query_length = int(padded["input_ids"].size(1))
        shape = (batch_size, type_count, query_length)
        batch["query_input_ids"] = padded["input_ids"].reshape(shape)
        batch["query_attention_mask"] = padded["attention_mask"].bool().reshape(shape)
        if "token_type_ids" in padded:
            batch["query_token_type_ids"] = padded["token_type_ids"].reshape(shape)
        batch["query_type_ids"] = torch.arange(type_count).unsqueeze(0).expand(
            batch_size, -1
        )
        batch["query_first_subword_indices"] = torch.stack(
            [row["first"] for row in alignment_rows]
        ).reshape(batch_size, type_count, max_words)
        query_word_mask = torch.stack(
            [row["word_mask"] for row in alignment_rows]
        ).reshape(batch_size, type_count, max_words)
        query_word_mask &= batch["word_mask"].unsqueeze(1)
        batch["query_word_mask"] = query_word_mask
        batch["query_token_mask"] = _pad_masks(
            [row["query_mask"] for row in alignment_rows], query_length
        ).reshape(shape)
        batch["query_sentence_token_mask"] = _pad_masks(
            [row["sentence_mask"] for row in alignment_rows], query_length
        ).reshape(shape)

        starts = torch.zeros(batch_size, type_count, max_words)
        ends = torch.zeros_like(starts)
        positives = torch.zeros(
            batch_size, type_count, max_words, max_words
        )
        exists = torch.zeros(batch_size, type_count)
        region_positive = torch.zeros(
            batch_size,
            type_count,
            int(batch["region_mask"].size(1)),
            dtype=torch.bool,
        )
        for row in range(batch_size):
            entity_indices = torch.nonzero(
                batch["gold_entity_mask"][row], as_tuple=False
            ).squeeze(-1)
            for entity_index in entity_indices.tolist():
                type_id = int(batch["gold_type_ids"][row, entity_index].item())
                if not 0 <= type_id < type_count:
                    continue
                start, end = batch["gold_spans"][row, entity_index].tolist()
                end_index = int(end) - 1
                start = int(start)
                if not (
                    0 <= start <= end_index < max_words
                    and bool(query_word_mask[row, type_id, start])
                    and bool(query_word_mask[row, type_id, end_index])
                ):
                    continue
                exists[row, type_id] = 1.0
                starts[row, type_id, start] = 1.0
                ends[row, type_id, end_index] = 1.0
                positives[row, type_id, start, end_index] = 1.0
                region_positive[row, type_id] |= batch[
                    "gold_region_positive_mask"
                ][row, entity_index]
        batch["query_existence_targets"] = exists
        batch["query_start_targets"] = starts
        batch["query_end_targets"] = ends
        batch["query_span_positive_mask"] = positives.bool()
        batch["query_span_valid_mask"] = _span_valid_mask(
            query_word_mask,
            max_span_length=self.max_span_length,
        )
        region_positive &= ~batch["region_is_null"].unsqueeze(1)
        batch["query_region_positive_mask"] = region_positive
        batch["query_region_supervision_mask"] = region_positive.any(dim=-1)
        return batch


def _pad_masks(masks: list[torch.Tensor], length: int) -> torch.Tensor:
    result = torch.zeros(len(masks), length, dtype=torch.bool)
    for row, mask in enumerate(masks):
        result[row, : mask.numel()] = mask
    return result


def _span_valid_mask(
    word_mask: torch.Tensor,
    *,
    max_span_length: int,
) -> torch.Tensor:
    word_count = int(word_mask.size(-1))
    starts = torch.arange(word_count).view(1, 1, word_count, 1)
    ends = torch.arange(word_count).view(1, 1, 1, word_count)
    geometry = (ends >= starts) & ((ends - starts + 1) <= int(max_span_length))
    return (
        word_mask.unsqueeze(-1)
        & word_mask.unsqueeze(-2)
        & geometry.to(word_mask.device)
    )
