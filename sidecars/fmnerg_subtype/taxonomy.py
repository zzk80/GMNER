"""Fixed fine-grained subtype taxonomy and hierarchy constraints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


EXPECTED_SUBTYPE_COUNT = 51


@dataclass(frozen=True)
class SubtypeTaxonomy:
    labels: tuple[str, ...]
    label2id: dict[str, int]
    coarse_type_ids: dict[str, int]
    parent_by_label: dict[str, str]
    parent_ids: tuple[int, ...]
    source_path: str
    source_sha256: str

    @classmethod
    def from_file(cls, path: str | Path) -> "SubtypeTaxonomy":
        source = Path(path).resolve()
        raw_bytes = source.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError("Unsupported subtype taxonomy format.")

        coarse_type_ids = {
            str(name): int(index)
            for name, index in dict(payload["coarse_type_ids"]).items()
        }
        if coarse_type_ids != {"LOC": 0, "PER": 1, "ORG": 2, "OTHER": 3}:
            raise ValueError(
                "Subtype taxonomy coarse ids must match the frozen GMNER chain."
            )
        parent_by_label = {
            str(label): str(parent)
            for label, parent in dict(payload["subtype_parents"]).items()
        }
        labels = tuple(sorted(parent_by_label))
        if len(labels) != EXPECTED_SUBTYPE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_SUBTYPE_COUNT} subtypes, found {len(labels)}."
            )
        invalid_parents = sorted(
            {
                parent
                for parent in parent_by_label.values()
                if parent not in coarse_type_ids
            }
        )
        if invalid_parents:
            raise ValueError(f"Unknown subtype parents: {invalid_parents}")
        label2id = {label: index for index, label in enumerate(labels)}
        parent_ids = tuple(
            coarse_type_ids[parent_by_label[label]] for label in labels
        )
        return cls(
            labels=labels,
            label2id=label2id,
            coarse_type_ids=coarse_type_ids,
            parent_by_label=parent_by_label,
            parent_ids=parent_ids,
            source_path=str(source),
            source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    @property
    def num_subtypes(self) -> int:
        return len(self.labels)

    def subtype_id(self, label: str) -> int:
        try:
            return self.label2id[str(label)]
        except KeyError as exc:
            raise ValueError(f"Unknown fine-grained subtype: {label!r}") from exc

    def parent_id(self, subtype_id: int) -> int:
        if subtype_id < 0 or subtype_id >= self.num_subtypes:
            raise ValueError(f"Invalid subtype id: {subtype_id}")
        return self.parent_ids[subtype_id]

    def allowed_mask(
        self,
        coarse_type_ids: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        coarse = coarse_type_ids.long()
        if device is not None:
            coarse = coarse.to(device)
        if torch.any((coarse < 0) | (coarse >= len(self.coarse_type_ids))):
            invalid = coarse[(coarse < 0) | (coarse >= len(self.coarse_type_ids))]
            raise ValueError(
                f"Invalid predicted coarse type ids: {invalid.detach().cpu().tolist()}"
            )
        parents = torch.tensor(
            self.parent_ids,
            dtype=torch.long,
            device=coarse.device,
        )
        return parents.unsqueeze(0).eq(coarse.reshape(-1, 1))

    def mask_logits(
        self,
        logits: torch.Tensor,
        coarse_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.size(-1) != self.num_subtypes:
            raise ValueError(
                "Subtype logits must have shape [batch, 51], found "
                f"{tuple(logits.shape)}."
            )
        mask = self.allowed_mask(coarse_type_ids, device=logits.device)
        if mask.shape != logits.shape:
            raise ValueError("Subtype hierarchy mask and logits are misaligned.")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def validate_labels(self, labels: Iterable[str], *, require_all: bool) -> None:
        observed = {str(label) for label in labels}
        unknown = sorted(observed - set(self.labels))
        if unknown:
            raise ValueError(f"Dataset contains unknown subtypes: {unknown}")
        if require_all:
            missing = sorted(set(self.labels) - observed)
            if missing:
                raise ValueError(f"Dataset is missing taxonomy subtypes: {missing}")

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "coarse_type_ids": dict(self.coarse_type_ids),
            "subtype_parents": dict(self.parent_by_label),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }
