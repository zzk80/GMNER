from pathlib import Path

import torch

from gmner.config import GMNERConfig
from gmner.constants import IGNORE_INDEX
from gmner.knowledge.external_descriptions import (
    EXPLANATION_KINDS,
    EXTERNAL_SUBTYPE_EXPLANATIONS,
)
from gmner.models.external_knowledge import (
    ExternalKnowledgePrototypeBank,
    ExternalKnowledgeTypeArbiter,
)
from scripts.build_curated_external_knowledge import build_curated_records
from scripts.build_external_knowledge_bank import build_prototype_payload
from scripts.init_external_knowledge_schema import (
    build_seed_records,
    discover_subtypes,
)


def save_bank(path: Path) -> None:
    prototypes = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    torch.save(
        {
            "prototypes": prototypes,
            "prototype_type_ids": torch.tensor([0, 0, 1, 2, 3]),
            "prototype_subtype_ids": torch.tensor([0, 0, 1, 2, 3]),
            "subtype_type_ids": torch.tensor([0, 1, 2, 3]),
            "type_names": ["LOC", "PER", "ORG", "OTHER"],
            "subtype_names": ["city", "artist", "company", "work"],
        },
        path,
    )


def test_external_bank_retrieves_without_modifying_token_states(tmp_path: Path):
    bank_path = tmp_path / "subtype_prototypes.pt"
    save_bank(bank_path)
    bank = ExternalKnowledgePrototypeBank(
        str(bank_path),
        hidden_size=4,
        temperature=0.1,
        dropout=0.0,
    )
    token_states = torch.randn(2, 5, 4)
    original = token_states.clone()
    outputs = bank(
        token_states=token_states,
        attention_mask=torch.ones(2, 5),
        target_mask=torch.tensor(
            [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0]],
            dtype=torch.float32,
        ),
    )

    assert outputs["type_logits"].shape == (2, 4)
    assert outputs["subtype_logits"].shape == (2, 4)
    assert outputs["center_scores"].shape == (2, 5)
    assert torch.equal(token_states, original)
    assert not bank.prototypes.requires_grad
    assert all(
        name.startswith(("query_projection", "query_norm"))
        for name, _ in bank.named_parameters()
    )


def test_log_mean_exp_does_not_reward_duplicate_centers(tmp_path: Path):
    bank_path = tmp_path / "subtype_prototypes.pt"
    save_bank(bank_path)
    bank = ExternalKnowledgePrototypeBank(str(bank_path), hidden_size=4)

    scores = torch.tensor([[0.8, 0.8, 0.2]])
    grouped = bank._group_log_mean_exp(
        scores,
        group_ids=torch.tensor([0, 0, 1]),
        group_count=2,
    )

    assert torch.allclose(grouped, torch.tensor([[0.8, 0.2]]), atol=1e-6)


def test_outcome_arbiter_is_identity_when_type_predictions_agree():
    arbiter = ExternalKnowledgeTypeArbiter(
        num_types=4,
        dropout=0.0,
        initial_gate=0.9,
    )
    base_logits = torch.tensor([[4.0, 1.0, 0.0, -1.0]])
    knowledge_logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])

    outputs = arbiter(base_logits, knowledge_logits)

    assert outputs["type_disagreement"].tolist() == [False]
    assert outputs["type_gate"].tolist() == [0.0]
    assert torch.equal(outputs["adjusted_type_logits"], base_logits)


def test_outcome_arbiter_can_select_knowledge_on_disagreement():
    arbiter = ExternalKnowledgeTypeArbiter(
        num_types=4,
        dropout=0.0,
        initial_gate=0.5,
        strength=1.0,
    )
    with torch.no_grad():
        arbiter.gate_network[-1].weight.zero_()
        arbiter.gate_network[-1].bias.fill_(20.0)
    base_logits = torch.tensor([[4.0, 1.0, 0.0, -1.0]])
    knowledge_logits = torch.tensor([[0.0, 5.0, 1.0, -1.0]])

    outputs = arbiter(base_logits, knowledge_logits)

    assert outputs["type_disagreement"].tolist() == [True]
    assert outputs["type_gate"].item() > 0.99
    assert outputs["adjusted_type_logits"].argmax(dim=-1).tolist() == [1]


def test_outcome_arbiter_threshold_restores_base_at_inference():
    arbiter = ExternalKnowledgeTypeArbiter(
        num_types=4,
        dropout=0.0,
        initial_gate=0.25,
        inference_threshold=0.5,
    )
    arbiter.eval()
    with torch.no_grad():
        arbiter.gate_network[-1].weight.zero_()
    base_logits = torch.tensor([[4.0, 1.0, 0.0, -1.0]])
    knowledge_logits = torch.tensor([[0.0, 5.0, 1.0, -1.0]])

    outputs = arbiter(base_logits, knowledge_logits)

    assert 0.24 < outputs["type_gate_probability"].item() < 0.26
    assert outputs["type_gate"].item() == 0.0
    assert torch.equal(outputs["adjusted_type_logits"], base_logits)


def test_outcome_arbiter_uses_decoded_base_type_for_disagreement():
    arbiter = ExternalKnowledgeTypeArbiter(
        num_types=4,
        dropout=0.0,
        initial_gate=0.5,
    )
    base_logits = torch.tensor([[1.0, 4.0, 0.0, -1.0]])
    knowledge_logits = torch.tensor([[0.0, 5.0, 1.0, -1.0]])

    outputs = arbiter(
        base_logits,
        knowledge_logits,
        base_type_ids=torch.tensor([0]),
    )

    assert outputs["base_type_prediction"].tolist() == [0]
    assert outputs["knowledge_type_prediction"].tolist() == [1]
    assert outputs["type_disagreement"].tolist() == [True]


def test_outcome_loss_rewards_correction_and_preservation():
    base_logits = torch.tensor([[4.0, 1.0], [1.0, 4.0]])
    knowledge_logits = torch.tensor([[1.0, 4.0], [4.0, 1.0]])
    targets = torch.tensor([1, 1])
    disagreement = torch.tensor([True, True])

    correct_gate_loss = ExternalKnowledgeTypeArbiter.outcome_loss(
        gate_logits=torch.tensor([4.0, -4.0]),
        disagreement=disagreement,
        base_type_logits=base_logits,
        knowledge_type_logits=knowledge_logits,
        targets=targets,
    )
    reversed_gate_loss = ExternalKnowledgeTypeArbiter.outcome_loss(
        gate_logits=torch.tensor([-4.0, 4.0]),
        disagreement=disagreement,
        base_type_logits=base_logits,
        knowledge_type_logits=knowledge_logits,
        targets=targets,
    )

    assert correct_gate_loss < reversed_gate_loss


def test_subtype_targets_use_coarse_type_for_ambiguous_names(tmp_path: Path):
    path = tmp_path / "ambiguous.pt"
    torch.save(
        {
            "prototypes": torch.eye(4),
            "prototype_type_ids": torch.tensor([0, 1, 2, 3]),
            "prototype_subtype_ids": torch.tensor([0, 1, 2, 3]),
            "subtype_type_ids": torch.tensor([0, 1, 2, 3]),
            "type_names": ["LOC", "PER", "ORG", "OTHER"],
            "subtype_names": ["other", "person", "group", "other"],
        },
        path,
    )
    bank = ExternalKnowledgePrototypeBank(str(path), hidden_size=4)

    targets = bank.subtype_targets(
        ["other", "other", "missing"],
        device=torch.device("cpu"),
        coarse_type_ids=torch.tensor([0, 3, 1]),
    )

    assert targets.tolist() == [0, 3, IGNORE_INDEX]


def test_builder_creates_multiple_centers_and_consistent_types():
    records = [
        {
            "id": f"city:{index}",
            "coarse_type": "LOC",
            "fine_type": "city",
            "confidence": 1.0,
            "source": "test",
        }
        for index in range(4)
    ]
    records.append(
        {
            "id": "artist:0",
            "coarse_type": "PER",
            "fine_type": "artist",
            "confidence": 1.0,
            "source": "test",
        }
    )
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-0.9, -0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    payload = build_prototype_payload(
        records,
        embeddings,
        max_centers_per_subtype=2,
    )

    assert payload["prototypes"].shape == (3, 4)
    assert payload["subtype_names"] == ["city", "artist"]
    center_types = payload["subtype_type_ids"][payload["prototype_subtype_ids"]]
    assert torch.equal(center_types, payload["prototype_type_ids"])


def test_schema_initializer_reads_labels_without_copying_mentions(tmp_path: Path):
    source = tmp_path / "train.txt"
    source.write_text(
        "\n".join(
            [
                "IMGID:1",
                "Paris B-building B-city",
                "Alice B-person B-artist",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pairs = discover_subtypes(source)
    records = build_seed_records(pairs)

    assert pairs == [("LOC", "city"), ("PER", "artist")]
    assert all("Paris" not in record["text"] for record in records)
    assert all("Alice" not in record["text"] for record in records)


def test_external_knowledge_is_disabled_by_default():
    config = GMNERConfig()

    assert config.model.use_external_knowledge is False
    assert config.model.external_knowledge_type_prior_weight == 0.0
    assert config.model.external_knowledge_fusion_mode == "fixed"
    assert config.model.external_knowledge_arbiter_inference_threshold == 0.0
    assert config.loss.lambda_external_knowledge_type == 0.0
    assert config.loss.lambda_external_knowledge_subtype == 0.0
    assert config.loss.lambda_external_knowledge_arbiter == 0.0


def test_curated_descriptions_cover_all_51_subtypes_with_three_views():
    schema_pairs = sorted(EXTERNAL_SUBTYPE_EXPLANATIONS)
    records = build_curated_records(schema_pairs)

    assert len(schema_pairs) == 51
    assert len(records) == 51 * len(EXPLANATION_KINDS) == 153
    assert len({record["id"] for record in records}) == 153
    assert {record["explanation_kind"] for record in records} == set(
        EXPLANATION_KINDS
    )
    assert all(record["uses_dataset_mentions"] is False for record in records)
    assert all(record["review_status"] == "draft" for record in records)
