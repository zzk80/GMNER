"""Trainable sidecar that cannot write back into the frozen GMNER model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .taxonomy import SubtypeTaxonomy


HEAD_ARCHITECTURES = ("shared_hard", "parent_specific_hard")


def _build_classifier(
    *,
    input_size: int,
    hidden_size: int,
    output_size: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_size),
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden_size, output_size),
    )


class HierarchicalSubtypeSidecar(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        dropout: float,
        taxonomy: SubtypeTaxonomy,
        head_architecture: str = "shared_hard",
        parent_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.taxonomy = taxonomy
        self.head_architecture = str(head_architecture)
        if self.head_architecture not in HEAD_ARCHITECTURES:
            raise ValueError(
                f"Unknown subtype head architecture: {self.head_architecture!r}."
            )
        self.parent_subtype_ids = tuple(
            tuple(
                subtype_id
                for subtype_id, parent_id in enumerate(taxonomy.parent_ids)
                if parent_id == requested_parent
            )
            for requested_parent in range(len(taxonomy.coarse_type_ids))
        )
        if any(not subtype_ids for subtype_ids in self.parent_subtype_ids):
            raise ValueError("Every coarse parent must contain at least one subtype.")

        if self.head_architecture == "shared_hard":
            self.parent_hidden_size = None
            self.classifier = _build_classifier(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=taxonomy.num_subtypes,
                dropout=dropout,
            )
            self.parent_classifiers = None
        else:
            resolved_parent_hidden_size = int(
                parent_hidden_size
                if parent_hidden_size is not None
                else max(
                    1,
                    self.hidden_size // len(taxonomy.coarse_type_ids),
                )
            )
            if resolved_parent_hidden_size <= 0:
                raise ValueError("parent_hidden_size must be positive.")
            self.parent_hidden_size = resolved_parent_hidden_size
            self.classifier = None
            self.parent_classifiers = nn.ModuleList(
                [
                    _build_classifier(
                        input_size=self.input_size,
                        hidden_size=resolved_parent_hidden_size,
                        output_size=len(subtype_ids),
                        dropout=dropout,
                    )
                    for subtype_ids in self.parent_subtype_ids
                ]
            )

    def _raw_logits(self, features: torch.Tensor) -> torch.Tensor:
        float_features = features.float()
        if self.head_architecture == "shared_hard":
            if self.classifier is None:
                raise RuntimeError("Shared subtype classifier is missing.")
            return self.classifier(float_features)
        if self.parent_classifiers is None:
            raise RuntimeError("Parent-specific subtype classifiers are missing.")
        parent_logits = [
            classifier(float_features)
            for classifier in self.parent_classifiers
        ]
        local_positions = {
            subtype_id: (parent_id, local_index)
            for parent_id, subtype_ids in enumerate(self.parent_subtype_ids)
            for local_index, subtype_id in enumerate(subtype_ids)
        }
        return torch.stack(
            [
                parent_logits[parent_id][:, local_index]
                for subtype_id in range(self.taxonomy.num_subtypes)
                for parent_id, local_index in [local_positions[subtype_id]]
            ],
            dim=-1,
        )

    def forward(
        self,
        features: torch.Tensor,
        coarse_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.size(-1) != self.input_size:
            raise ValueError(
                f"Expected subtype features [batch, {self.input_size}], "
                f"found {tuple(features.shape)}."
            )
        raw_logits = self._raw_logits(features)
        logits = (
            self.taxonomy.mask_logits(raw_logits, coarse_type_ids)
            if coarse_type_ids is not None
            else raw_logits
        )
        return {
            "raw_logits": raw_logits,
            "logits": logits,
            "predicted_subtype_ids": logits.argmax(dim=-1),
        }
