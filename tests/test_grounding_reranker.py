import torch

from gmner.models.grounding_reranker import PrototypeAwareGroundingReranker, stable_bucket


def test_stable_bucket_is_deterministic():
    assert stable_bucket("person", 128) == stable_bucket("person", 128)


def test_grounding_reranker_shape_and_mask():
    reranker = PrototypeAwareGroundingReranker(
        hidden_size=8,
        dropout=0.0,
        object_vocab_size=32,
        attr_vocab_size=64,
        label_embedding_dim=4,
    )
    entity_repr = torch.randn(2, 8)
    region_nodes = torch.randn(2, 3, 8)
    region_scores = torch.ones(2, 3)
    region_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    metadata = [
        {
            "region_object_labels": ["person", "logo", "NULL"],
            "region_object_attributes": ["standing person", "red logo", ""],
        },
        {
            "region_object_labels": ["building", "ball", "NULL"],
            "region_object_attributes": ["tall building", "brown ball", ""],
        },
    ]

    logits = reranker(
        entity_repr=entity_repr,
        region_nodes=region_nodes,
        region_scores=region_scores,
        region_mask=region_mask,
        metadata=metadata,
    )

    assert logits.shape == (2, 3)
    assert logits[0, 2].item() < -9999


def test_grounding_reranker_uses_type_box_and_gate_features():
    reranker = PrototypeAwareGroundingReranker(
        hidden_size=8,
        dropout=0.0,
        object_vocab_size=32,
        attr_vocab_size=64,
        label_embedding_dim=4,
        entity_input_dim=24,
        type_embedding_dim=4,
        rank_embedding_dim=3,
        max_regions=3,
        has_null_region=True,
    )
    entity_repr = torch.randn(2, 24)
    region_nodes = torch.randn(2, 3, 8)
    region_scores = torch.tensor([[0.9, 0.4, 1.0], [0.7, 0.2, 1.0]])
    region_mask = torch.ones(2, 3)
    base_type_logits = torch.tensor([[3.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, -1.0]])
    region_boxes = torch.tensor(
        [
            [[0.0, 0.0, 50.0, 50.0], [10.0, 10.0, 20.0, 20.0], [0.0, 0.0, 0.0, 0.0]],
            [[5.0, 5.0, 30.0, 40.0], [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    image_sizes = torch.tensor([[100.0, 100.0], [80.0, 120.0]])

    out = reranker(
        entity_repr=entity_repr,
        region_nodes=region_nodes,
        region_scores=region_scores,
        region_mask=region_mask,
        base_type_logits=base_type_logits,
        region_boxes=region_boxes,
        image_sizes=image_sizes,
        return_aux=True,
    )

    assert out["logits"].shape == (2, 3)
    assert out["visible_logit"].shape == (2,)
    gate = reranker.uncertainty_gate(
        base_logits=torch.randn(2, 3),
        rerank_logits=out["logits"],
        valid_mask=region_mask.bool(),
        base_type_logits=base_type_logits,
    )
    assert gate.shape == (2,)
    assert torch.all((gate >= 0.0) & (gate <= 1.0))
