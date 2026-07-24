from __future__ import annotations

import torch

from gmner.losses import (
    joint_multi_positive_loss,
    joint_structured_margin_loss,
    joint_teacher_kl_loss,
    joint_visibility_loss,
)


def _joint_fixture():
    candidate_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    target_type_ids = torch.tensor([1])
    positive_region_mask = torch.tensor([[False, True, False]])
    base_type_logits = torch.tensor([[2.0, 1.0]])
    base_region_logits = torch.tensor([[3.0, 1.0, 0.0]])
    return (
        candidate_mask,
        target_type_ids,
        positive_region_mask,
        base_type_logits,
        base_region_logits,
    )


def test_joint_multi_positive_loss_rewards_gold_type_region_scores():
    candidate_mask, target_types, positive_regions, _, _ = _joint_fixture()
    good = torch.zeros(1, 2, 3)
    bad = torch.zeros(1, 2, 3)
    good[0, 1, 1] = 4.0
    bad[0, 0, 0] = 4.0

    good_loss = joint_multi_positive_loss(
        good,
        target_types,
        positive_regions,
        candidate_mask,
    )
    bad_loss = joint_multi_positive_loss(
        bad,
        target_types,
        positive_regions,
        candidate_mask,
    )

    assert good_loss < bad_loss


def test_joint_visibility_loss_uses_positive_logit_for_visible_entities():
    logits = torch.tensor([4.0, -4.0])
    target_types = torch.tensor([0, 1])
    positives = torch.tensor(
        [
            [True, False, False],
            [False, False, True],
        ]
    )

    correct = joint_visibility_loss(
        logits,
        target_types,
        positives,
        null_index=2,
    )
    reversed_loss = joint_visibility_loss(
        -logits,
        target_types,
        positives,
        null_index=2,
    )

    assert correct < reversed_loss


def test_joint_structured_margin_targets_stage1_wrong_candidate():
    candidate_mask, target_types, positive_regions, base_types, base_regions = (
        _joint_fixture()
    )
    good = torch.zeros(1, 2, 3)
    bad = torch.zeros(1, 2, 3)
    good[0, 1, 1] = 2.0
    bad[0, 0, 0] = 2.0

    good_loss = joint_structured_margin_loss(
        good,
        target_types,
        positive_regions,
        candidate_mask,
        base_types,
        base_regions,
        margin=0.2,
    )
    bad_loss = joint_structured_margin_loss(
        bad,
        target_types,
        positive_regions,
        candidate_mask,
        base_types,
        base_regions,
        margin=0.2,
    )

    assert torch.isclose(good_loss, torch.tensor(0.0))
    assert bad_loss > good_loss


def test_joint_teacher_kl_is_zero_for_identical_distributions():
    base = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]])
    candidates = torch.ones_like(base, dtype=torch.bool)
    active = torch.tensor([True])

    same_loss = joint_teacher_kl_loss(base, base, candidates, active)
    shifted_loss = joint_teacher_kl_loss(
        torch.flip(base, dims=[-1]),
        base,
        candidates,
        active,
    )

    assert torch.abs(same_loss) < 1e-6
    assert shifted_loss > same_loss
