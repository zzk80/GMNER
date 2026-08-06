from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gmner.models.heads import GroundingHead
from gmner.tp.grounding_replay import (
    GroundabilityPriorLookup,
    apply_grounding_priors_with_stages,
    word_span_target_mask,
)
from gmner.tp.interfaces import extract_tp_stage1_interfaces, interface_equivalence_errors
from gmner.tp.reachability import (
    constrained_gold_reachability,
    estimate_sequence_radius,
    k_best_viterbi,
    sequence_score,
)


class TinyCrf(torch.nn.Module):
    def __init__(self, labels: int) -> None:
        super().__init__()
        self.start_transitions = torch.nn.Parameter(torch.zeros(labels))
        self.transitions = torch.nn.Parameter(torch.zeros(labels, labels))
        self.end_transitions = torch.nn.Parameter(torch.zeros(labels))


def test_tp_interface_contract_is_exact() -> None:
    outputs = {
        "fused_tokens": torch.randn(2, 4, 6),
        "ner_logits": torch.randn(2, 4, 9),
        "pre_prototype_fused_tokens": torch.randn(2, 4, 6),
        "image_nodes": torch.randn(2, 3, 6),
        "image_mask": torch.ones(2, 3),
    }
    interfaces = extract_tp_stage1_interfaces(outputs)
    assert max(interface_equivalence_errors(outputs, interfaces).values()) == 0.0


def test_kbest_score_and_reachability() -> None:
    crf = TinyCrf(3)
    emissions = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    best = k_best_viterbi(emissions, crf, k=2)
    assert best[0].labels == (0, 1)
    labels = torch.tensor(best[0].labels)
    assert best[0].score == pytest.approx(
        float(sequence_score(emissions, labels, crf).detach().item())
    )
    assert estimate_sequence_radius(emissions, crf) >= 0.0
    result = constrained_gold_reachability(emissions, labels, crf, rho=0.0)
    assert result["reachable"] is True


def test_reachability_rejects_unprotected_nonbest_gold() -> None:
    crf = TinyCrf(2)
    emissions = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
    gold = torch.tensor([1, 1])
    assert constrained_gold_reachability(emissions, gold, crf, rho=0.0)["reachable"] is False
    assert constrained_gold_reachability(emissions, gold, crf, rho=3.0)["reachable"] is True


def test_grounding_prior_stages_recompute_type_and_null() -> None:
    config = SimpleNamespace(
        data=SimpleNamespace(add_null_region=True),
        model=SimpleNamespace(
            grounding_null_prior_weight=1.0,
            grounding_null_logit_bias=0.25,
            region_score_prior_weight=0.1,
            region_object_compatibility_weight=0.2,
        ),
    )
    stages = apply_grounding_priors_with_stages(
        raw_logits=torch.zeros((1, 3)),
        image_mask=torch.ones((1, 3)),
        region_scores=torch.tensor([[0.5, 0.8, 1.0]]),
        region_labels=["person", "building", "NULL"],
        region_attributes=["", "", ""],
        entity_type_id=1,
        null_prior=0.8,
        config=config,
    )
    assert stages.after_entity_null_prior[0, -1] > 0
    assert stages.after_global_null_bias[0, -1] > stages.after_entity_null_prior[0, -1]
    assert stages.formal_logits.shape == (1, 3)


def test_word_span_mask_never_falls_back_to_full_record() -> None:
    attention = torch.ones(6)
    mask = word_span_target_mask([None, 0, 1, 1, 2, None], 1, 2, attention)
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="no aligned subwords"):
        word_span_target_mask([None] * 6, 1, 2, attention)


def test_prior_lookup_uses_mention_before_type(tmp_path) -> None:
    type_path = tmp_path / "type.jsonl"
    mention_path = tmp_path / "mention.jsonl"
    type_path.write_text('{"entity_type":"PER","null_prior":0.4}\n', encoding="utf-8")
    mention_path.write_text(
        '{"mention":"Ada Lovelace","entity_type":"PER","null_prior":0.1}\n',
        encoding="utf-8",
    )
    lookup = GroundabilityPriorLookup(type_path, mention_path)
    assert lookup.null_prior("ada   lovelace", "PER") == pytest.approx(0.1)
    assert lookup.null_prior("Grace Hopper", "PER") == pytest.approx(0.4)
