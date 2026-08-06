"""Zero-initialized residual adaptation over a frozen downstream teacher."""

from __future__ import annotations

import torch
from torch import nn


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _detached(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in outputs.items()
    }


class ProtectedFineResidual(nn.Module):
    """Add a new bounded region residual to a frozen Fine teacher."""

    def __init__(self, teacher: nn.Module, residual: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher
        self.residual = residual
        _freeze(self.teacher)

    def train(self, mode: bool = True) -> "ProtectedFineResidual":
        super().train(mode)
        self.teacher.eval()
        self.residual.train(mode)
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.teacher.eval()
        with torch.no_grad():
            reference = self.teacher(batch)
        proposal = self.residual(batch)
        reference_mask = reference["candidate_mask"].bool()
        proposal_mask = proposal["candidate_mask"].bool()
        if not torch.equal(reference_mask, proposal_mask):
            raise RuntimeError("Protected Fine teacher/residual candidate masks differ.")
        delta = proposal["bounded_residual_logits"].float().masked_fill(
            ~reference_mask, 0.0
        )
        logits = (
            reference["final_region_logits"].float() + delta
        ).masked_fill(~reference_mask, -1e4)
        outputs = _detached(reference)
        outputs.update(
            {
                "protected_reference_region_index": reference[
                    "best_real_region_index"
                ].long().detach(),
                "protected_reference_region_logits": reference[
                    "final_region_logits"
                ].float().detach(),
                "fine_delta_logits": proposal["fine_delta_logits"],
                "bounded_residual_logits": delta,
                "final_region_logits": logits,
                "best_real_region_index": logits.argmax(dim=-1),
            }
        )
        return outputs


class ProtectedEvidenceResidual(nn.Module):
    """Add a new bounded visibility residual to a frozen Evidence teacher."""

    def __init__(self, teacher: nn.Module, residual: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher
        self.residual = residual
        _freeze(self.teacher)

    def train(self, mode: bool = True) -> "ProtectedEvidenceResidual":
        super().train(mode)
        self.teacher.eval()
        self.residual.train(mode)
        return self

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        self.teacher.eval()
        with torch.no_grad():
            reference = self.teacher(*args, **kwargs)
        proposal = self.residual(*args, **kwargs)
        delta = proposal["bounded_visibility_delta_logits"].float()
        reference_logits = reference["final_visibility_logits"].float().detach()
        logits = reference_logits + delta
        outputs = dict(proposal)
        outputs.update(
            {
                "protected_reference_visibility_logits": reference_logits,
                "protected_reference_visibility_probability": torch.sigmoid(
                    reference_logits
                ),
                "base_visibility_logits": reference_logits,
                "base_visibility_probability": torch.sigmoid(reference_logits),
                "bounded_visibility_delta_logits": delta,
                "final_visibility_logits": logits,
                "final_visibility_probability": torch.sigmoid(logits),
            }
        )
        return outputs
