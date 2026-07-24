import torch

from gmner.models.entity_evidence_decoder import EntityEvidenceDecoder


def test_entity_evidence_decoder_outputs_joint_type_region_scores():
    decoder = EntityEvidenceDecoder(
        hidden_size=8,
        num_types=4,
        dropout=0.0,
        num_layers=1,
        num_heads=2,
        object_vocab_size=32,
        attr_vocab_size=64,
        label_embedding_dim=4,
    )
    entity_repr = torch.randn(2, 8)
    context_repr = torch.randn(2, 8)
    region_nodes = torch.randn(2, 3, 8)
    region_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    base_type_logits = torch.randn(2, 4)
    base_region_logits = torch.randn(2, 3).masked_fill(region_mask == 0, -1e4)

    outputs = decoder(
        entity_repr=entity_repr,
        context_repr=context_repr,
        region_nodes=region_nodes,
        region_mask=region_mask,
        base_type_logits=base_type_logits,
        base_region_logits=base_region_logits,
        region_scores=torch.ones(2, 3),
        metadata=[
            {"region_object_labels": ["person", "logo"], "region_object_attributes": ["standing", "red"]},
            {"region_object_labels": ["building", "ball"], "region_object_attributes": ["tall", "round"]},
        ],
    )

    assert outputs["type_logits"].shape == (2, 4)
    assert outputs["region_logits"].shape == (2, 3)
    assert outputs["joint_logits"].shape == (2, 4, 3)
    assert outputs["region_logits"][0, 2].item() < -9999


def test_entity_evidence_decoder_is_noop_at_initialization():
    decoder = EntityEvidenceDecoder(
        hidden_size=8,
        num_types=4,
        dropout=0.0,
        num_layers=1,
        num_heads=2,
        object_vocab_size=32,
        attr_vocab_size=64,
        label_embedding_dim=4,
    )
    entity_repr = torch.randn(2, 8)
    context_repr = torch.randn(2, 8)
    region_nodes = torch.randn(2, 3, 8)
    region_mask = torch.ones(2, 3)
    base_type_logits = torch.randn(2, 4)
    base_region_logits = torch.randn(2, 3)

    outputs = decoder(
        entity_repr=entity_repr,
        context_repr=context_repr,
        region_nodes=region_nodes,
        region_mask=region_mask,
        base_type_logits=base_type_logits,
        base_region_logits=base_region_logits,
    )

    assert torch.allclose(outputs["type_logits"], base_type_logits)
    assert torch.allclose(outputs["region_logits"], base_region_logits)
    assert torch.allclose(outputs["pair_scores"], torch.zeros_like(outputs["pair_scores"]))
