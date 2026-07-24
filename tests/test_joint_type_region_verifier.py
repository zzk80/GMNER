import torch
import torch.nn.functional as F

from gmner.models.joint_type_region_verifier import (
    JointEntityAdapter,
    JointTypeRegionVerifier,
    perturb_span_masks,
)


def _inputs(batch_size: int = 2, hidden_size: int = 8, num_regions: int = 4):
    return {
        "entity_repr": torch.randn(batch_size, hidden_size),
        "image_global_repr": torch.randn(batch_size, hidden_size),
        "region_nodes": torch.randn(batch_size, num_regions, hidden_size),
        "region_mask": torch.ones(batch_size, num_regions, dtype=torch.bool),
        "base_type_logits": torch.randn(batch_size, 4),
        "base_region_logits": torch.randn(batch_size, num_regions),
    }


def test_joint_entity_adapter_is_residual_noop_at_initialization():
    adapter = JointEntityAdapter(hidden_size=8, dropout=0.0)
    entity_repr = torch.randn(2, 8)

    adapted = adapter(
        entity_repr=entity_repr,
        boundary_repr=torch.randn(2, 24),
        context_repr=torch.randn(2, 8),
        image_global_repr=torch.randn(2, 8),
    )

    assert torch.allclose(adapted, entity_repr)


def test_joint_verifier_preserves_factorized_stage1_ranking_at_initialization():
    verifier = JointTypeRegionVerifier(
        hidden_size=8,
        interaction_hidden_size=12,
        dropout=0.0,
        visibility_weight=1.0,
        top_m_types=4,
        top_r_regions=0,
    )
    inputs = _inputs()
    inputs["region_mask"][0, 1] = False
    outputs = verifier(**inputs)

    assert outputs["joint_logits"].shape == (2, 4, 4)
    assert outputs["type_logits"].shape == (2, 4)
    assert outputs["region_logits"].shape == (2, 4)
    assert torch.equal(
        outputs["type_logits"].argmax(dim=-1),
        inputs["base_type_logits"].argmax(dim=-1),
    )
    expected_region_log_probs = F.log_softmax(
        inputs["base_region_logits"].masked_fill(~inputs["region_mask"], -1e4),
        dim=-1,
    )
    assert torch.allclose(
        outputs["region_logits"][inputs["region_mask"]],
        expected_region_log_probs[inputs["region_mask"]],
        atol=1e-5,
    )
    assert torch.equal(
        outputs["region_logits"].argmax(dim=-1),
        inputs["base_region_logits"].masked_fill(~inputs["region_mask"], -1e4).argmax(dim=-1),
    )
    assert outputs["joint_logits"][0, :, 1].max().item() < -9999
    assert torch.count_nonzero(outputs["interaction_logits"]).item() == 0


def test_joint_verifier_injects_missing_gold_candidates_for_training():
    verifier = JointTypeRegionVerifier(
        hidden_size=8,
        interaction_hidden_size=12,
        dropout=0.0,
        top_m_types=1,
        top_r_regions=1,
    )
    inputs = _inputs(batch_size=1)
    inputs["base_type_logits"] = torch.tensor([[5.0, 1.0, 0.0, -1.0]])
    inputs["base_region_logits"] = torch.tensor([[5.0, 1.0, 0.0, -1.0]])
    outputs = verifier(
        **inputs,
        force_type_ids=torch.tensor([2]),
        force_region_mask=torch.tensor([[False, False, True, False]]),
    )

    assert outputs["type_candidate_mask"][0].tolist() == [True, False, True, False]
    assert outputs["region_candidate_mask"][0].tolist() == [True, False, True, True]
    assert outputs["type_candidate_injected"].item() is True
    assert outputs["region_candidate_injected"].item() is True


def test_legacy_visibility_uses_original_additive_adjustment():
    verifier = JointTypeRegionVerifier(
        hidden_size=8,
        interaction_hidden_size=12,
        dropout=0.0,
        visibility_weight=1.0,
        visibility_logit_max=0.0,
        hierarchical_visibility=False,
    )
    with torch.no_grad():
        verifier.visibility_head[-1].bias.fill_(2.0)
    inputs = _inputs(batch_size=1)
    inputs["base_region_logits"] = torch.zeros(1, 4)

    outputs = verifier(**inputs)
    base = F.log_softmax(inputs["base_region_logits"], dim=-1)

    assert torch.allclose(outputs["region_logits"][0, :-1], base[0, :-1] + 1.0)
    assert torch.allclose(outputs["region_logits"][0, -1], base[0, -1] - 1.0)


def test_joint_verifier_bounds_learned_interaction_and_visibility_logits():
    verifier = JointTypeRegionVerifier(
        hidden_size=8,
        interaction_hidden_size=12,
        dropout=0.0,
        interaction_logit_max=5.0,
        visibility_logit_max=4.0,
        hierarchical_visibility=True,
    )
    with torch.no_grad():
        verifier.interaction[-1].bias.fill_(100.0)
        verifier.visibility_head[-1].bias.fill_(100.0)

    outputs = verifier(**_inputs())

    assert outputs["interaction_logits"].abs().max().item() <= 5.0
    assert outputs["visibility_residual_logits"].abs().max().item() <= 4.0
    assert (
        outputs["visibility_logits"] - outputs["base_visibility_logits"]
    ).abs().max().item() <= 4.0
    assert outputs["raw_interaction_logits"].max().item() > 5.0
    assert outputs["raw_visibility_logits"].max().item() > 4.0


def test_span_perturbation_changes_only_joint_boundary_and_keeps_overlap():
    target_mask = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    attention_mask = torch.ones_like(target_mask)
    metadata = [
        {
            "target_start": 1,
            "target_end": 2,
            "word_ids": [None, 0, 1, 2, None],
        }
    ]

    perturbed, changed = perturb_span_masks(
        target_mask=target_mask,
        attention_mask=attention_mask,
        metadata=metadata,
        probability=1.0,
        max_words=1,
    )

    assert changed.item() is True
    assert perturbed.sum().item() == 2.0
    assert bool((perturbed.bool() & target_mask.bool()).any().item())
    assert torch.equal(target_mask, torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]))
