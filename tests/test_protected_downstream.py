from __future__ import annotations

import torch
from torch import nn

from gmner.models.protected_downstream import (
    ProtectedEvidenceResidual,
    ProtectedFineResidual,
)


class _FineModule(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(delta))

    def forward(self, _batch):
        mask = torch.tensor([[[True, True, False]]])
        base = torch.tensor([[[2.0, 1.0, -1e4]]])
        delta = self.weight * torch.tensor([[[0.0, 1.0, 0.0]]])
        logits = (base + delta).masked_fill(~mask, -1e4)
        return {
            "candidate_mask": mask,
            "final_region_logits": logits,
            "best_real_region_index": logits.argmax(dim=-1),
            "fine_delta_logits": delta,
            "bounded_residual_logits": delta,
        }


class _EvidenceModule(nn.Module):
    def __init__(self, base: float, delta: float) -> None:
        super().__init__()
        self.base = float(base)
        self.weight = nn.Parameter(torch.tensor(delta))

    def forward(self, *_args, **_kwargs):
        base = self.weight.new_tensor([[self.base]])
        delta = self.weight.reshape(1, 1)
        logits = base + delta
        return {
            "bounded_visibility_delta_logits": delta,
            "final_visibility_logits": logits,
            "final_visibility_probability": torch.sigmoid(logits),
            "fine_has_real_candidate": torch.ones(1, 1, dtype=torch.bool),
        }


def test_protected_fine_epoch_zero_matches_teacher_and_keeps_gradient() -> None:
    model = ProtectedFineResidual(_FineModule(0.5), _FineModule(0.0))
    outputs = model({})
    assert torch.equal(
        outputs["best_real_region_index"],
        outputs["protected_reference_region_index"],
    )
    assert torch.allclose(
        outputs["final_region_logits"],
        outputs["protected_reference_region_logits"],
    )
    outputs["final_region_logits"][0, 0, 1].backward()
    assert model.residual.weight.grad is not None
    assert model.teacher.weight.grad is None


def test_protected_evidence_epoch_zero_matches_teacher_and_keeps_gradient() -> None:
    model = ProtectedEvidenceResidual(
        _EvidenceModule(base=0.25, delta=0.5),
        _EvidenceModule(base=99.0, delta=0.0),
    )
    outputs = model()
    assert torch.allclose(
        outputs["final_visibility_logits"],
        outputs["protected_reference_visibility_logits"],
    )
    outputs["final_visibility_logits"].sum().backward()
    assert model.residual.weight.grad is not None
    assert model.teacher.weight.grad is None
