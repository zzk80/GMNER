"""Fixed 51-class FMNERG taxonomy and hierarchy constraints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch


EXPECTED_SUBTYPE_COUNT = 51
EXPECTED_COARSE_TYPE_IDS = {"LOC": 0, "PER": 1, "ORG": 2, "OTHER": 3}


@dataclass(frozen=True)
class SubtypeTaxonomy:
    """Immutable subtype IDs and their four coarse parents."""

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
        if coarse_type_ids != EXPECTED_COARSE_TYPE_IDS:
            raise ValueError(
                "Subtype taxonomy coarse ids must match the frozen GMNER "
                f"schema {EXPECTED_COARSE_TYPE_IDS}."
            )

        parent_by_label = {
            str(label): str(parent)
            for label, parent in dict(payload["subtype_parents"]).items()
        }
        labels = tuple(sorted(parent_by_label))
        if len(labels) != EXPECTED_SUBTYPE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_SUBTYPE_COUNT} subtypes, found "
                f"{len(labels)}."
            )
        if len(set(labels)) != len(labels):
            raise ValueError("Subtype taxonomy contains duplicate labels.")

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
        if set(parent_ids) != set(coarse_type_ids.values()):
            raise ValueError("Every coarse parent must contain a subtype.")

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

    @property
    def num_parents(self) -> int:
        return len(self.coarse_type_ids)

    def subtype_id(self, label: str) -> int:
        try:
            return self.label2id[str(label)]
        except KeyError as exc:
            raise ValueError(
                f"Unknown fine-grained subtype: {label!r}"
            ) from exc

    def subtype_label(self, subtype_id: int) -> str:
        if subtype_id < 0 or subtype_id >= self.num_subtypes:
            raise ValueError(f"Invalid subtype id: {subtype_id}")
        return self.labels[subtype_id]

    def parent_id(self, subtype_id: int) -> int:
        if subtype_id < 0 or subtype_id >= self.num_subtypes:
            raise ValueError(f"Invalid subtype id: {subtype_id}")
        return self.parent_ids[subtype_id]

    def parent_name(self, subtype_id: int) -> str:
        parent_id = self.parent_id(subtype_id)
        return next(
            name
            for name, candidate_id in self.coarse_type_ids.items()
            if candidate_id == parent_id
        )

    def allowed_mask(
        self,
        coarse_type_ids: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        coarse = coarse_type_ids.long()
        if device is not None:
            coarse = coarse.to(device)
        invalid = (coarse < 0) | (coarse >= self.num_parents)
        if torch.any(invalid):
            raise ValueError(
                "Invalid predicted coarse type ids: "
                f"{coarse[invalid].detach().cpu().tolist()}"
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

    def validate_labels(
        self,
        labels: Iterable[str],
        *,
        require_all: bool,
    ) -> None:
        observed = {str(label) for label in labels}
        unknown = sorted(observed - set(self.labels))
        if unknown:
            raise ValueError(f"Dataset contains unknown subtypes: {unknown}")
        if require_all:
            missing = sorted(set(self.labels) - observed)
            if missing:
                raise ValueError(f"Dataset is missing taxonomy subtypes: {missing}")

    def validate_parent(self, subtype_id: int, parent_id: int) -> None:
        expected = self.parent_id(subtype_id)
        if expected != int(parent_id):
            raise ValueError(
                "Subtype parent mismatch: "
                f"subtype={self.subtype_label(subtype_id)!r}, "
                f"expected={expected}, found={parent_id}."
            )

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "coarse_type_ids": dict(self.coarse_type_ids),
            "subtype_parents": dict(self.parent_by_label),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }

    def fingerprint_metadata(self) -> dict[str, object]:
        return {
            "taxonomy_path": self.source_path,
            "taxonomy_sha256": self.source_sha256,
            "num_subtypes": self.num_subtypes,
            "num_parent_types": self.num_parents,
        }


def validate_taxonomy_fingerprint(
    metadata: Mapping[str, object],
    taxonomy: SubtypeTaxonomy,
    *,
    artifact_name: str,
) -> None:
    """Reject checkpoints or caches built with another taxonomy."""

    actual = str(metadata.get("taxonomy_sha256") or "")
    if actual != taxonomy.source_sha256:
        raise ValueError(
            f"{artifact_name} taxonomy SHA mismatch: expected "
            f"{taxonomy.source_sha256}, found {actual or '<missing>'}."
        )
    subtype_count = int(metadata.get("num_subtypes", -1))
    if subtype_count != taxonomy.num_subtypes:
        raise ValueError(
            f"{artifact_name} subtype count mismatch: expected "
            f"{taxonomy.num_subtypes}, found {subtype_count}."
        )


def bind_config_taxonomy_fingerprint(
    data_config: object,
    taxonomy: SubtypeTaxonomy,
) -> None:
    """Validate a preregistered SHA and persist it in the resolved config."""

    expected = str(
        getattr(data_config, "subtype_taxonomy_sha256", "") or ""
    )
    if expected and expected != taxonomy.source_sha256:
        raise ValueError(
            "Configured subtype taxonomy SHA mismatch: expected "
            f"{expected}, found {taxonomy.source_sha256}."
        )
    setattr(
        data_config,
        "subtype_taxonomy_sha256",
        taxonomy.source_sha256,
    )
