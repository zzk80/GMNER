from pathlib import Path

import torch

from gmner.models.prototype_bank import SemanticPrototypeBank


def build_bank(path: Path, hidden_size: int = 6) -> None:
    torch.manual_seed(7)
    type_prototypes = torch.randn(4, hidden_size)
    subtype_prototypes = torch.randn(8, hidden_size)
    subtype_type_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    torch.save(
        {
            "type_prototypes": type_prototypes,
            "subtype_prototypes": subtype_prototypes,
            "subtype_type_ids": subtype_type_ids,
        },
        path,
    )


def test_prototype_bank_only_writes_to_target_span(tmp_path: Path):
    bank_path = tmp_path / "semantic_prototypes.pt"
    build_bank(bank_path)
    bank = SemanticPrototypeBank(str(bank_path), hidden_size=6, dropout=0.0)

    token_states = torch.randn(2, 5, 6)
    attention_mask = torch.ones(2, 5)
    target_mask = torch.tensor(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    outputs = bank(
        token_states=token_states,
        attention_mask=attention_mask,
        target_mask=target_mask,
        valid_entity_mask=torch.tensor([True, False]),
    )

    enhanced = outputs["enhanced_tokens"]
    assert torch.allclose(enhanced[0, 0], token_states[0, 0])
    assert torch.allclose(enhanced[0, 3:], token_states[0, 3:])
    assert torch.allclose(enhanced[1], token_states[1])
    assert outputs["prototype_gate"][1].item() == 0.0


def test_subtype_set_loss_is_finite(tmp_path: Path):
    bank_path = tmp_path / "semantic_prototypes.pt"
    build_bank(bank_path)
    bank = SemanticPrototypeBank(str(bank_path), hidden_size=6, dropout=0.0)

    subtype_scores = torch.randn(3, 8)
    target_type_ids = torch.tensor([0, 2, 4])
    loss = bank.subtype_set_loss(subtype_scores, target_type_ids)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_constant_gate_respects_valid_entity_mask(tmp_path: Path):
    bank_path = tmp_path / "semantic_prototypes.pt"
    build_bank(bank_path)
    bank = SemanticPrototypeBank(
        str(bank_path),
        hidden_size=6,
        dropout=0.0,
        gate_mode="constant",
        constant_gate=0.25,
        max_gate=0.25,
    )

    token_states = torch.randn(2, 4, 6)
    attention_mask = torch.ones(2, 4)
    target_mask = torch.tensor(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )

    outputs = bank(
        token_states=token_states,
        attention_mask=attention_mask,
        target_mask=target_mask,
        valid_entity_mask=torch.tensor([True, False]),
    )

    assert outputs["prototype_gate"][0].item() > 0.0
    assert outputs["prototype_gate"][0].item() <= 0.25
    assert outputs["prototype_gate"][1].item() == 0.0
