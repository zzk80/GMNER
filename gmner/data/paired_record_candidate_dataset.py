"""Aligned formal/expanded record-candidate caches for grounding experiments."""

from __future__ import annotations

import hashlib
from typing import Any
from collections import Counter

import torch
from torch.utils.data import Dataset

from .hierarchical_record_candidate_collator import (
    HierarchicalRecordCandidateCollator,
)
from .record_candidate_dataset import RecordCandidateDataset


def _record_id(record: dict[str, Any]) -> str:
    return str((record.get("metadata") or {}).get("record_id", ""))


def _candidate_spec(dataset: RecordCandidateDataset) -> dict[str, Any]:
    return dict(dataset.metadata.get("candidate_config") or {})


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SPAN_ALIGNED_KEYS = (
    "span_candidates",
    "span_mask",
    "span_features",
    "span_base_scores",
    "span_source_ids",
    "span_lengths",
    "type_candidates",
    "type_base_scores",
    "type_mask",
    "region_base_scores",
    "type_region_compatibility",
    "fixed_type_ids",
    "base_region_indices",
    "base_region_scores",
    "gold_span_mask",
    "gold_type_mask",
    "gold_region_positive_mask",
    "positive_triple_mask",
    "region_iou_targets",
    "visibility_targets",
)


def _span_boundaries(record: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        tuple(map(int, value))
        for value in torch.as_tensor(record["span_candidates"]).tolist()
    ]


def _align_expanded_record(
    formal: dict[str, Any],
    expanded: dict[str, Any],
    formal_to_expanded: list[int],
) -> dict[str, Any]:
    """Reorder R36 span rows to the formal R16 candidate order."""

    expanded_span_count = len(_span_boundaries(expanded))
    indices = torch.tensor(
        [value if value >= 0 else 0 for value in formal_to_expanded],
        dtype=torch.long,
    )
    matched = torch.tensor(
        [value >= 0 for value in formal_to_expanded], dtype=torch.bool
    )
    aligned = dict(expanded)
    for key in _SPAN_ALIGNED_KEYS:
        if key not in expanded:
            continue
        value = torch.as_tensor(expanded[key])
        if value.ndim == 0 or value.size(0) != expanded_span_count:
            raise ValueError(
                f"Expanded record {_record_id(expanded)} has invalid {key} shape."
            )
        aligned[key] = value.index_select(0, indices)

    # The formal hierarchy is authoritative for candidate identity, source and
    # type. Unmatched non-Stage1 candidates remain shape-aligned but masked out.
    aligned["span_candidates"] = torch.as_tensor(
        formal["span_candidates"]
    ).clone()
    aligned["span_source_ids"] = torch.as_tensor(
        formal["span_source_ids"]
    ).clone()
    aligned["fixed_type_ids"] = torch.as_tensor(
        formal["fixed_type_ids"]
    ).clone()
    aligned["span_mask"] = (
        torch.as_tensor(formal["span_mask"]).bool() & matched
    )
    metadata = dict(expanded.get("metadata") or {})
    metadata["formal_span_count"] = len(formal_to_expanded)
    metadata["formal_unmatched_non_stage1_count"] = int((~matched).sum().item())
    aligned["metadata"] = metadata
    return aligned


class PairedRecordCandidateDataset(Dataset):
    """Pair R16 and R36 caches while enforcing a shared frozen Stage1."""

    def __init__(
        self,
        formal: RecordCandidateDataset,
        expanded: RecordCandidateDataset,
    ) -> None:
        self.formal = formal
        self.expanded = expanded
        self._validate_metadata()
        expanded_by_id = {
            _record_id(record): index
            for index, record in enumerate(expanded.records)
        }
        if len(expanded_by_id) != len(expanded.records):
            raise ValueError("Expanded cache contains duplicate or empty record ids.")
        self.pairs: list[tuple[int, int]] = []
        self.span_mappings: list[list[int]] = []
        alignment = Counter()
        for formal_index, record in enumerate(formal.records):
            record_id = _record_id(record)
            if not record_id or record_id not in expanded_by_id:
                raise ValueError(
                    f"Expanded cache is missing formal record {record_id or '<empty>'}."
                )
            expanded_index = expanded_by_id[record_id]
            expanded_record = expanded.records[expanded_index]
            expanded_spans = _span_boundaries(expanded_record)
            expanded_span_map = {
                boundary: span_index
                for span_index, boundary in enumerate(expanded_spans)
            }
            if len(expanded_span_map) != len(expanded_spans):
                raise ValueError(
                    f"Expanded record {record_id} contains duplicate spans."
                )
            formal_spans = _span_boundaries(record)
            mapping = [expanded_span_map.get(boundary, -1) for boundary in formal_spans]
            formal_sources = torch.as_tensor(record["span_source_ids"]).long()
            formal_types = torch.as_tensor(record["fixed_type_ids"]).long()
            expanded_types = torch.as_tensor(
                expanded_record["fixed_type_ids"]
            ).long()
            for span_index, (source, expanded_span_index) in enumerate(
                zip(formal_sources.tolist(), mapping)
            ):
                alignment["formal_spans"] += 1
                alignment["matched_spans"] += int(expanded_span_index >= 0)
                if int(source) != 0:
                    continue
                alignment["formal_stage1_spans"] += 1
                if expanded_span_index < 0:
                    raise ValueError(
                        f"Expanded record {record_id} is missing formal Stage1 "
                        f"span {formal_spans[span_index]}."
                    )
                alignment["matched_stage1_spans"] += 1
                if int(formal_types[span_index].item()) != int(
                    expanded_types[expanded_span_index].item()
                ):
                    # R16 and R36 run the multimodal Stage1 with different
                    # region budgets, so a shared span can rarely change its
                    # decoded type. The formal R16 branch is authoritative and
                    # _align_expanded_record overwrites this value downstream.
                    alignment["stage1_type_mismatches"] += 1
            alignment["records"] += 1
            alignment["exact_span_tables"] += int(formal_spans == expanded_spans)
            self.pairs.append((formal_index, expanded_index))
            self.span_mappings.append(mapping)
        if len(self.pairs) != len(expanded.records):
            raise ValueError("Formal and expanded caches contain different record sets.")
        self.alignment_summary = dict(alignment)

    def _validate_metadata(self) -> None:
        formal_hash = str(
            self.formal.metadata.get("stage1_checkpoint_sha256") or ""
        )
        expanded_hash = str(
            self.expanded.metadata.get("stage1_checkpoint_sha256") or ""
        )
        if not formal_hash or formal_hash != expanded_hash:
            raise ValueError("Formal and expanded caches use different Stage1 models.")
        formal_spec = _candidate_spec(self.formal)
        expanded_spec = _candidate_spec(self.expanded)
        ignored = {"max_regions"}
        mismatches = {
            key: (formal_spec.get(key), expanded_spec.get(key))
            for key in sorted((set(formal_spec) | set(expanded_spec)) - ignored)
            if formal_spec.get(key) != expanded_spec.get(key)
        }
        if mismatches:
            raise ValueError(
                "Paired caches must differ only in max_regions; "
                f"mismatches={mismatches}."
            )
        anchor = dict(
            self.expanded.metadata.get("formal_anchor_cache") or {}
        )
        if anchor:
            expected_anchor_hash = str(anchor.get("sha256") or "")
            if (
                not expected_anchor_hash
                or not hasattr(self.formal, "path")
                or _sha256_file(self.formal.path) != expected_anchor_hash
            ):
                raise ValueError(
                    "Expanded cache formal-anchor fingerprint does not match "
                    "the paired formal cache."
                )
        formal_budget = int(formal_spec.get("max_regions", 0))
        expanded_budget = int(expanded_spec.get("max_regions", 0))
        if formal_budget <= 0 or expanded_budget <= formal_budget:
            raise ValueError(
                "Expected expanded max_regions to exceed formal max_regions; "
                f"formal={formal_budget}, expanded={expanded_budget}."
            )
        self.formal_budget = formal_budget
        self.expanded_budget = expanded_budget

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        formal_index, expanded_index = self.pairs[index]
        formal = self.formal[formal_index]
        expanded_raw = self.expanded[expanded_index]
        expanded = _align_expanded_record(
            formal, expanded_raw, self.span_mappings[index]
        )
        formal_null = int((formal.get("metadata") or {}).get("null_region_index", -1))
        expanded_null = int(
            (expanded_raw.get("metadata") or {}).get("null_region_index", -1)
        )
        if formal_null != self.formal_budget or expanded_null != self.expanded_budget:
            raise ValueError(
                f"Unexpected NULL indices for record {_record_id(formal)}: "
                f"formal={formal_null}, expanded={expanded_null}."
            )
        formal_boxes = torch.as_tensor(formal["region_boxes"])[
            : self.formal_budget
        ]
        expanded_boxes = torch.as_tensor(expanded_raw["region_boxes"])[
            : self.formal_budget
        ]
        if not torch.allclose(formal_boxes, expanded_boxes, atol=1e-5, rtol=1e-5):
            raise ValueError(
                f"R16 proposal prefix differs for record {_record_id(formal)}."
            )
        return {"formal": formal, "expanded": expanded}


class PairedRecordCandidateCollator:
    """Collate aligned formal and expanded records independently."""

    def __init__(self) -> None:
        self.collator = HierarchicalRecordCandidateCollator()

    def __call__(
        self, records: list[dict[str, dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        return {
            "formal": self.collator([record["formal"] for record in records]),
            "expanded": self.collator(
                [record["expanded"] for record in records]
            ),
        }
