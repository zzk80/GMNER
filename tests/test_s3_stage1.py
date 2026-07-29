from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID, IGNORE_INDEX
from gmner.engine.s3_stage1_evaluator import evaluate_s3_stage1
from gmner.losses.s3_stage1_loss import (
    S3LossWeights,
    compute_s3_stage1_losses,
)
from gmner.models.heads import GroundingHead, TokenClassificationHead
from gmner.models.stage1 import (
    BOUNDARY_B,
    BOUNDARY_I,
    BOUNDARY_O,
    HierarchicalJointStage1,
    LegacyStage1RecordWrapper,
    SpanTypeHead,
    WordBoundaryCRF,
    boundary_tags_to_spans,
    typed_bio_to_boundary,
)


def test_typed_bio_collapses_to_boundary_ids() -> None:
    typed = torch.tensor(
        [
            [
                DEFAULT_LABEL2ID["O"],
                DEFAULT_LABEL2ID["B-PER"],
                DEFAULT_LABEL2ID["I-PER"],
                DEFAULT_LABEL2ID["B-LOC"],
                DEFAULT_LABEL2ID["I-OTHER"],
                IGNORE_INDEX,
            ]
        ]
    )
    assert torch.equal(
        typed_bio_to_boundary(typed),
        torch.tensor(
            [[
                BOUNDARY_O,
                BOUNDARY_B,
                BOUNDARY_I,
                BOUNDARY_B,
                BOUNDARY_I,
                IGNORE_INDEX,
            ]]
        ),
    )


def test_boundary_crf_enforces_start_and_o_to_i_constraints() -> None:
    crf = WordBoundaryCRF(hidden_size=2, dropout=0.0)
    emissions = torch.tensor(
        [[[8.0, 7.0, 100.0], [9.0, 1.0, 100.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True]])
    decoded = crf.decode(emissions, mask)
    assert decoded[0, 0].item() != BOUNDARY_I
    if decoded[0, 0].item() == BOUNDARY_O:
        assert decoded[0, 1].item() != BOUNDARY_I

    labels = torch.tensor([[BOUNDARY_B, BOUNDARY_I]])
    loss, denominator = crf.neg_log_likelihood(
        emissions,
        labels,
        mask,
    )
    assert denominator.item() == 2
    loss.backward()
    assert emissions.grad is not None
    assert emissions.grad.abs().sum().item() > 0


def test_boundary_crf_resets_at_truncation_gap() -> None:
    crf = WordBoundaryCRF(hidden_size=2, dropout=0.0)
    emissions = torch.tensor(
        [
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ]
    )
    decoded = crf.decode(
        emissions,
        torch.tensor([[True, False, True]]),
    )
    assert decoded[0, 0].item() == BOUNDARY_B
    assert decoded[0, 1].item() == IGNORE_INDEX
    assert decoded[0, 2].item() != BOUNDARY_I


def test_boundary_decode_produces_word_space_half_open_spans() -> None:
    tags = torch.tensor(
        [[BOUNDARY_O, BOUNDARY_B, BOUNDARY_I, BOUNDARY_O]]
    )
    spans, valid, rows = boundary_tags_to_spans(
        tags,
        torch.ones_like(tags, dtype=torch.bool),
    )
    assert rows == [[[1, 3]]]
    assert spans.tolist() == [[[1, 3]]]
    assert valid.tolist() == [[True]]


def test_span_type_head_uses_first_last_and_mean() -> None:
    head = SpanTypeHead(hidden_size=2, dropout=0.0)
    tokens = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]]]
    )
    mask = torch.tensor([[[True, True, False]]])
    pooled = head.pool(tokens, mask)
    assert torch.equal(
        pooled,
        torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0, 2.0, 3.0]]]
        ),
    )


def test_s3_losses_use_independent_denominators() -> None:
    boundary = WordBoundaryCRF(hidden_size=2, dropout=0.0)
    model = SimpleNamespace(boundary_head=boundary)
    outputs = {
        "boundary_emissions": torch.randn(
            2, 3, 3, requires_grad=True
        ),
        "gold_type_logits": torch.randn(
            2, 2, 4, requires_grad=True
        ),
        "grounding_formal_logits": torch.randn(
            2, 2, 3, requires_grad=True
        ),
        "alignment_score": torch.randn(
            2, 2, requires_grad=True
        ),
    }
    batch = {
        "typed_bio_labels": torch.tensor(
            [
                [
                    DEFAULT_LABEL2ID["B-PER"],
                    DEFAULT_LABEL2ID["I-PER"],
                    DEFAULT_LABEL2ID["O"],
                ],
                [
                    DEFAULT_LABEL2ID["O"],
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                ],
            ]
        ),
        "word_mask": torch.tensor(
            [[True, True, True], [True, False, False]]
        ),
        "gold_type_ids": torch.tensor(
            [
                [ENTITY_TYPE2ID["PER"], ENTITY_TYPE2ID["O"]],
                [ENTITY_TYPE2ID["O"], ENTITY_TYPE2ID["O"]],
            ]
        ),
        "type_entity_mask": torch.tensor(
            [[True, False], [False, False]]
        ),
        "gold_region_labels": torch.tensor([[0, 1], [2, 0]]),
        "grounding_entity_mask": torch.tensor(
            [[True, True], [False, False]]
        ),
    }
    losses = compute_s3_stage1_losses(
        model=model,
        outputs=outputs,
        batch=batch,
        weights=S3LossWeights(1.0, 1.0, 1.0, 1.0),
    )
    assert losses["denominator_boundary_words"].item() == 4
    assert losses["denominator_type_entities"].item() == 1
    assert losses["denominator_grounding_entities"].item() == 2
    assert losses["denominator_alignment_records"].item() == 2
    losses["loss"].backward()
    for key in (
        "boundary_emissions",
        "gold_type_logits",
        "grounding_formal_logits",
        "alignment_score",
    ):
        assert outputs[key].grad is not None


class _TextEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        del attention_mask, token_type_ids
        states = self.embedding(input_ids)
        return states, states[:, 0]


class _IdentityGraph(nn.Module):
    def forward(self, states, adjacency):
        del adjacency
        return states


class _Aligner(nn.Module):
    def forward(self, text_nodes, image_nodes, text_mask, image_mask):
        del text_mask, image_mask
        context = image_nodes.mean(dim=1, keepdim=True)
        fused = text_nodes + context
        return fused, fused.mean(dim=1), torch.eye(fused.size(0))


class _Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 4
        self.config = SimpleNamespace(
            data=SimpleNamespace(grounding_iou_threshold=0.5),
            model=SimpleNamespace(
                hidden_size=hidden,
                grounding_null_prior_weight=1.0,
                grounding_null_logit_bias=0.0,
                region_score_prior_weight=0.0,
                region_object_compatibility_weight=0.0,
            ),
        )
        self.text_encoder = _TextEncoder(hidden)
        self.text_projector = nn.Identity()
        self.text_graph_encoder = _IdentityGraph()
        self.region_projector = nn.Linear(hidden, hidden)
        self.region_norm = nn.Identity()
        self.image_graph_encoder = _IdentityGraph()
        self.aligner = _Aligner()
        self.ner_head = TokenClassificationHead(
            hidden,
            len(DEFAULT_LABEL2ID),
            dropout=0.0,
            use_crf=False,
        )
        self.grounding_head = GroundingHead(hidden)
        self.prototype_bank = None
        self.external_knowledge_bank = None
        self.grounding_reranker = None
        self.grounding_residual_adapter = None
        self.multiscale_grounding_aligner = None
        self.entity_evidence_decoder = None
        self.joint_type_region_verifier = None


def _record_batch() -> dict:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "word_count": torch.tensor([2]),
        "typed_bio_labels": torch.tensor(
            [[DEFAULT_LABEL2ID["B-PER"], DEFAULT_LABEL2ID["O"]]]
        ),
        "legacy_ner_labels": torch.tensor(
            [[IGNORE_INDEX, 0, 0, IGNORE_INDEX]]
        ),
        "adjacency": torch.eye(4).unsqueeze(0),
        "first_subword_indices": torch.tensor([[1, 2]]),
        "word_mask": torch.tensor([[True, True]]),
        "subword_to_word": torch.tensor([[-1, 0, 1, -1]]),
        "region_features": torch.randn(1, 3, 4),
        "region_boxes": torch.zeros(1, 3, 4),
        "region_mask": torch.ones(1, 3, dtype=torch.bool),
        "region_scores": torch.ones(1, 3),
        "null_region_index": torch.tensor([2]),
        "region_is_null": torch.tensor([[False, False, True]]),
        "gold_spans": torch.tensor([[[0, 1]]]),
        "gold_subword_masks": torch.tensor(
            [[[False, True, False, False]]]
        ),
        "gold_type_ids": torch.tensor([[ENTITY_TYPE2ID["PER"]]]),
        "gold_entity_mask": torch.tensor([[True]]),
        "type_entity_mask": torch.tensor([[True]]),
        "grounding_entity_mask": torch.tensor([[True]]),
        "gold_region_labels": torch.tensor([[2]]),
        "gold_region_positive_mask": torch.tensor(
            [[[False, False, True]]]
        ),
        "grounding_null_prior": torch.tensor([[0.5]]),
        "metadata": [
            {
                "record_id": "r0",
                "tokens": ["Alice", "runs"],
                "gt_boxes_by_name": {},
                "region_object_labels": ["person", "street"],
                "region_object_attributes": ["", ""],
            }
        ],
    }


def test_trainable_student_copies_but_does_not_mutate_wrapper() -> None:
    teacher = _Teacher().eval()
    student = HierarchicalJointStage1(
        teacher,
        boundary_dropout=0.0,
        type_dropout=0.0,
    ).eval()
    wrapper = LegacyStage1RecordWrapper(teacher).eval()
    batch = _record_batch()
    old = wrapper.encode_records(batch)
    new = student.encode_records(batch, decode_boundary=False)
    for key in (
        "base_text_nodes",
        "text_graph_nodes",
        "fused_tokens",
        "fused_global",
        "alignment_score",
        "image_nodes",
    ):
        assert torch.equal(old[key], new[key])
    assert all(parameter.requires_grad for parameter in student.parameters())
    try:
        wrapper.train()
    except RuntimeError:
        pass
    else:
        raise AssertionError("S3.0 wrapper became trainable.")


def test_s3_evaluator_runs_formal_boundary_type_grounding_chain() -> None:
    teacher = _Teacher().eval()
    student = HierarchicalJointStage1(
        teacher,
        boundary_dropout=0.0,
        type_dropout=0.0,
    ).eval()
    wrapper = LegacyStage1RecordWrapper(teacher).eval()
    report = evaluate_s3_stage1(
        model=student,
        dataloader=[_record_batch()],
        device=torch.device("cpu"),
        baseline_wrapper=wrapper,
    )
    assert report["scope"] == "dev"
    assert report["test_accessed"] is False
    assert report["metrics"]["gold_count"] == 1.0
    assert report["metrics"]["records"] == 1.0


def test_full_s3_student_has_trainable_boundary_type_and_grounding() -> None:
    student = HierarchicalJointStage1(
        _Teacher().eval(),
        boundary_dropout=0.0,
        type_dropout=0.0,
    ).train()
    batch = _record_batch()
    outputs = student(batch)
    losses = compute_s3_stage1_losses(
        model=student,
        outputs=outputs,
        batch=batch,
        weights=S3LossWeights(1.0, 1.0, 1.0, 1.0),
    )
    losses["loss"].backward()
    assert student.boundary_head.emission.weight.grad is not None
    assert student.span_type_head.classifier[-1].weight.grad is not None
    assert student.grounding_head.proj.weight.grad is not None
