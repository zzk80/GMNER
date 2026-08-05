from __future__ import annotations

import torch

from gmner.losses.protected_region_mner_loss import (
    boundary_preservation_kl,
    protected_gate_penalty,
    protected_region_residual_l2,
)
from gmner.models.protected_region_mner import (
    ProtectedBidirectionalAttention,
    ProtectedRegionSemanticAdapter,
    ProtectedVisualTypeHead,
    real_region_mask,
)


def _region_inputs(hidden_size: int = 16):
    raw = torch.randn(2, 4, 12)
    projected = torch.randn(2, 4, hidden_size)
    base = torch.randn(2, 4, hidden_size)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 1]], dtype=torch.float32)
    boxes = torch.tensor(
        [
            [[0, 0, 20, 20], [20, 20, 40, 40], [5, 5, 10, 10], [0, 0, 0, 0]],
            [[10, 10, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[0.9, 0.8, 0.4, 0.0], [0.7, 0.0, 0.0, 0.0]])
    sizes = torch.tensor([[100, 200], [100, 200]], dtype=torch.float32)
    return raw, projected, base, mask, boxes, scores, sizes


def test_real_region_mask_excludes_formal_null() -> None:
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.float32)
    result = real_region_mask(mask, has_null_region=True)
    assert result.tolist() == [[True, True, False, False]]


def test_region_adapter_is_exact_noop_and_preserves_null() -> None:
    raw, projected, base, mask, boxes, scores, sizes = _region_inputs()
    adapter = ProtectedRegionSemanticAdapter(
        region_feature_dim=12,
        hidden_size=16,
        bottleneck_size=8,
        gate_hidden_size=4,
        dropout=0.0,
        has_null_region=True,
    )
    adapter.eval()
    outputs = adapter(
        raw_region_features=raw,
        base_region_states=base,
        gate_region_states=projected,
        image_mask=mask,
        region_boxes=boxes,
        region_scores=scores,
        image_sizes=sizes,
    )
    torch.testing.assert_close(outputs["region_states"], base, rtol=0, atol=0)
    torch.testing.assert_close(outputs["region_delta"], torch.zeros_like(base), rtol=0, atol=0)
    assert torch.count_nonzero(outputs["region_gate"][:, -1]) == 0


def test_bidirectional_attention_is_exact_text_noop_and_excludes_null() -> None:
    torch.manual_seed(4)
    refiner = ProtectedBidirectionalAttention(
        hidden_size=16,
        num_heads=4,
        dropout=0.0,
        gate_hidden_size=8,
        gate_max=0.3,
        has_null_region=True,
    )
    refiner.eval()
    text = torch.randn(2, 5, 16)
    regions = torch.randn(2, 4, 16)
    text_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    image_mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 1]])
    outputs = refiner(
        base_text_states=text,
        semantic_region_states=regions,
        attention_mask=text_mask,
        image_mask=image_mask,
    )
    torch.testing.assert_close(outputs["refined_text_states"], text, rtol=0, atol=0)
    assert torch.count_nonzero(outputs["feedback_attention"][..., -1]) == 0
    assert torch.count_nonzero(outputs["refined_region_states"][:, -1]) == 0


def test_bidirectional_attention_handles_records_without_real_regions() -> None:
    refiner = ProtectedBidirectionalAttention(
        hidden_size=8,
        num_heads=2,
        dropout=0.0,
        gate_hidden_size=4,
        gate_max=0.3,
        has_null_region=True,
    )
    text = torch.randn(1, 3, 8)
    regions = torch.randn(1, 2, 8)
    outputs = refiner(
        base_text_states=text,
        semantic_region_states=regions,
        attention_mask=torch.ones(1, 3),
        image_mask=torch.tensor([[0, 1]]),
    )
    assert all(torch.isfinite(value).all() for value in outputs.values())
    torch.testing.assert_close(outputs["refined_text_states"], text, rtol=0, atol=0)
    assert torch.count_nonzero(outputs["token_gate"]) == 0


def test_feedback_output_projection_has_gradient_at_zero_init() -> None:
    refiner = ProtectedBidirectionalAttention(
        hidden_size=8,
        num_heads=2,
        dropout=0.0,
        gate_hidden_size=4,
        gate_max=0.3,
        has_null_region=True,
    )
    outputs = refiner(
        base_text_states=torch.randn(1, 3, 8),
        semantic_region_states=torch.randn(1, 3, 8),
        attention_mask=torch.ones(1, 3),
        image_mask=torch.ones(1, 3),
    )
    outputs["refined_text_states"].sum().backward()
    gradient = refiner.feedback_attention.out_proj.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_visual_type_loss_reaches_raw_region_adapter() -> None:
    raw, projected, base, mask, boxes, scores, sizes = _region_inputs()
    adapter = ProtectedRegionSemanticAdapter(
        region_feature_dim=12,
        hidden_size=16,
        bottleneck_size=8,
        gate_hidden_size=4,
        dropout=0.0,
        has_null_region=True,
    )
    region_outputs = adapter(
        raw_region_features=raw,
        base_region_states=base,
        gate_region_states=projected,
        image_mask=mask,
        region_boxes=boxes,
        region_scores=scores,
        image_sizes=sizes,
    )
    type_head = ProtectedVisualTypeHead(hidden_size=16, dropout=0.0)
    text = torch.randn(2, 5, 16)
    target_mask = torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32)
    attention = torch.zeros(2, 5, 4)
    attention[:, :, 0] = 1.0
    logits = type_head(
        text_states=text,
        region_states=region_outputs["region_states"],
        feedback_attention=attention,
        target_mask=target_mask,
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
    gradient = adapter.semantic_up.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_protection_losses_are_zero_for_noop_outputs() -> None:
    logits = torch.randn(2, 4, 9)
    labels = torch.tensor([[0, 1, 2, -100], [0, 0, 3, 4]])
    mask = labels.ne(-100)
    boundary = boundary_preservation_kl(
        base_logits=logits,
        refined_logits=logits,
        attention_mask=mask,
        labels=labels,
    )
    assert abs(float(boundary)) < 1e-7

    gate = torch.zeros(2, 4)
    gate_loss = protected_gate_penalty(
        token_gate=gate,
        labels=labels,
        attention_mask=mask,
    )
    assert float(gate_loss) == 0.0

    delta = torch.zeros(2, 3, 8)
    residual = protected_region_residual_l2(
        region_delta=delta,
        real_region_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
    )
    assert float(residual) == 0.0
