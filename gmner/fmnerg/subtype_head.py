"""Text-only hierarchy-masked subtype head for Stage1-F."""

from __future__ import annotations

import torch
import torch.nn as nn

from gmner.fmnerg.taxonomy import SubtypeTaxonomy


HEAD_ARCHITECTURES = ("shared_hard", "parent_specific_hard")


def pool_span_boundary_mean(
    token_states: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool first, last, and mean subword states for each span."""

    if token_states.ndim != 3:
        raise ValueError("token_states must have shape [batch, length, hidden].")
    if target_mask.shape != token_states.shape[:2]:
        raise ValueError(
            "target_mask must align with token_states; found "
            f"{tuple(target_mask.shape)} and {tuple(token_states.shape)}."
        )
    mask = target_mask.to(device=token_states.device, dtype=torch.bool)
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("Every subtype span must contain at least one subword.")

    length = token_states.size(1)
    positions = torch.arange(length, device=token_states.device)
    first_indices = positions.unsqueeze(0).masked_fill(~mask, length).min(dim=-1).values
    last_indices = positions.unsqueeze(0).masked_fill(~mask, -1).max(dim=-1).values
    rows = torch.arange(token_states.size(0), device=token_states.device)
    first = token_states[rows, first_indices]
    last = token_states[rows, last_indices]
    weights = mask.to(dtype=token_states.dtype)
    mean = (token_states * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    return torch.cat((first, last, mean), dim=-1)


def _classifier(
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


class FineSubtypeHead(nn.Module):
    """Predict raw 51-way logits and apply a four-parent hard mask."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        dropout: float,
        taxonomy: SubtypeTaxonomy,
        architecture: str = "shared_hard",
        parent_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if architecture not in HEAD_ARCHITECTURES:
            raise ValueError(
                f"Unknown subtype head architecture: {architecture!r}."
            )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.architecture = str(architecture)
        self.num_subtypes = taxonomy.num_subtypes
        self.num_parents = taxonomy.num_parents
        self.register_buffer(
            "subtype_parent_ids",
            torch.tensor(taxonomy.parent_ids, dtype=torch.long),
            persistent=True,
        )
        self.parent_subtype_ids = tuple(
            tuple(
                subtype_id
                for subtype_id, parent_id in enumerate(taxonomy.parent_ids)
                if parent_id == requested_parent
            )
            for requested_parent in range(taxonomy.num_parents)
        )

        if self.architecture == "shared_hard":
            self.classifier = _classifier(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.num_subtypes,
                dropout=dropout,
            )
            self.parent_classifiers = None
        else:
            resolved_parent_hidden = int(
                parent_hidden_size
                if parent_hidden_size is not None
                else max(1, self.hidden_size // self.num_parents)
            )
            self.classifier = None
            self.parent_classifiers = nn.ModuleList(
                [
                    _classifier(
                        input_size=self.input_size,
                        hidden_size=resolved_parent_hidden,
                        output_size=len(subtype_ids),
                        dropout=dropout,
                    )
                    for subtype_ids in self.parent_subtype_ids
                ]
            )

    def raw_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.input_size:
            raise ValueError(
                f"Expected subtype features [batch, {self.input_size}], "
                f"found {tuple(features.shape)}."
            )
        float_features = features.float()
        if self.classifier is not None:
            return self.classifier(float_features)
        if self.parent_classifiers is None:
            raise RuntimeError("Parent-specific subtype classifiers are missing.")
        local_logits = [
            classifier(float_features)
            for classifier in self.parent_classifiers
        ]
        positions = {
            subtype_id: (parent_id, local_index)
            for parent_id, subtype_ids in enumerate(self.parent_subtype_ids)
            for local_index, subtype_id in enumerate(subtype_ids)
        }
        return torch.stack(
            [
                local_logits[parent_id][:, local_index]
                for subtype_id in range(self.num_subtypes)
                for parent_id, local_index in [positions[subtype_id]]
            ],
            dim=-1,
        )

    def mask_logits(
        self,
        raw_logits: torch.Tensor,
        parent_ids: torch.Tensor,
    ) -> torch.Tensor:
        if raw_logits.ndim != 2 or raw_logits.size(-1) != self.num_subtypes:
            raise ValueError("Raw subtype logits have an invalid shape.")
        parents = parent_ids.to(device=raw_logits.device, dtype=torch.long)
        invalid = (parents < 0) | (parents >= self.num_parents)
        if torch.any(invalid):
            raise ValueError(
                "Invalid subtype parent ids: "
                f"{parents[invalid].detach().cpu().tolist()}."
            )
        allowed = self.subtype_parent_ids.unsqueeze(0).eq(
            parents.reshape(-1, 1)
        )
        return raw_logits.masked_fill(
            ~allowed,
            torch.finfo(raw_logits.dtype).min,
        )

    def forward(
        self,
        features: torch.Tensor,
        parent_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw_logits = self.raw_logits(features)
        logits = self.mask_logits(raw_logits, parent_ids)
        return {
            "raw_logits": raw_logits,
            "logits": logits,
            "predicted_subtype_ids": logits.argmax(dim=-1),
        }
