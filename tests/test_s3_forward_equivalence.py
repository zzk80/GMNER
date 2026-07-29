from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID, IGNORE_INDEX
from gmner.engine.s3_forward_equivalence import (
    _grounding_numerical_gate,
    _match_formal_record_predictions,
    evaluate_s3_forward_equivalence,
)
from gmner.knowledge.region_compatibility import compatibility_score
from gmner.models.heads import GroundingHead, TokenClassificationHead
from gmner.models.stage1 import LegacyStage1RecordWrapper


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
    def forward(
        self,
        text_nodes,
        image_nodes,
        text_mask,
        image_mask,
    ):
        del text_mask, image_mask
        context = image_nodes.mean(dim=1, keepdim=True)
        fused = text_nodes + context
        return fused, fused.mean(dim=1), torch.eye(
            fused.size(0), device=fused.device
        )


class _Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 4
        self.config = SimpleNamespace(
            data=SimpleNamespace(
                grounding_iou_threshold=0.5,
                add_null_region=True,
            ),
            model=SimpleNamespace(
                grounding_null_prior_weight=1.0,
                grounding_null_logit_bias=0.0,
                region_score_prior_weight=0.1,
                region_object_compatibility_weight=0.2,
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

    def forward(self, batch):
        text, _ = self.text_encoder(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids"),
        )
        text = self.text_projector(text)
        text = self.text_graph_encoder(text, batch["adjacency"])
        image = self.region_norm(
            self.region_projector(batch["region_features"])
        )
        image = self.image_graph_encoder(image, torch.empty(0))
        fused, global_state, alignment = self.aligner(
            text,
            image,
            batch["attention_mask"].float(),
            batch["region_mask"],
        )
        logits = self.ner_head(fused)
        return {
            "base_text_nodes": text,
            "text_graph_nodes": text,
            "ner_logits": logits,
            "pre_prototype_fused_tokens": fused,
            "image_nodes": image,
            "image_mask": batch["region_mask"],
            "fused_global": global_state,
            "alignment_score": alignment,
        }

    def _apply_grounding_knowledge(
        self,
        logits,
        image_nodes,
        image_mask,
        batch,
        target_type_ids=None,
    ):
        del image_nodes
        output = logits.clone()
        prior = batch["grounding_null_prior"].to(output).clamp(
            1e-4, 1 - 1e-4
        )
        output[:, -1] += torch.log(prior / (1 - prior))
        detector = batch["region_scores"].to(output).clamp(
            1e-4, 1.0
        )
        detector_bias = 0.1 * detector.log()
        detector_bias[:, -1] = 0.0
        output += detector_bias
        for row, metadata in enumerate(batch["metadata"]):
            labels = metadata["region_object_labels"]
            attrs = metadata["region_object_attributes"]
            for region in range(min(len(labels), output.size(1) - 1)):
                output[row, region] += 0.2 * compatibility_score(
                    int(target_type_ids[row].item()),
                    labels[region],
                    attrs[region],
                )
        return output.masked_fill(~image_mask.bool(), -1e4)


def _batch() -> dict:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "adjacency": torch.eye(4).unsqueeze(0),
        "word_count": torch.tensor([2]),
        "first_subword_indices": torch.tensor([[1, 2]]),
        "word_to_subword_start": torch.tensor([[1, 2]]),
        "word_to_subword_end": torch.tensor([[2, 3]]),
        "subword_to_word": torch.tensor([[-1, 0, 1, -1]]),
        "word_complete_mask": torch.tensor([[True, True]]),
        "word_mask": torch.tensor([[True, True]]),
        "typed_bio_labels": torch.tensor(
            [[DEFAULT_LABEL2ID["B-PER"], DEFAULT_LABEL2ID["O"]]]
        ),
        "legacy_ner_labels": torch.tensor(
            [
                [
                    IGNORE_INDEX,
                    DEFAULT_LABEL2ID["B-PER"],
                    DEFAULT_LABEL2ID["O"],
                    IGNORE_INDEX,
                ]
            ]
        ),
        "region_features": torch.randn(1, 3, 4),
        "region_boxes": torch.zeros(1, 3, 4),
        "region_mask": torch.ones(1, 3, dtype=torch.bool),
        "region_scores": torch.tensor([[0.9, 0.7, 1.0]]),
        "null_region_index": torch.tensor([2]),
        "region_is_null": torch.tensor([[False, False, True]]),
        "gold_spans": torch.tensor([[[0, 1]]]),
        "gold_type_ids": torch.tensor([[ENTITY_TYPE2ID["PER"]]]),
        "gold_entity_mask": torch.tensor([[True]]),
        "grounding_entity_mask": torch.tensor([[True]]),
        "type_entity_mask": torch.tensor([[True]]),
        "gold_subword_masks": torch.tensor(
            [[[False, True, False, False]]]
        ),
        "gold_region_labels": torch.tensor([[0]]),
        "gold_region_positive_mask": torch.tensor(
            [[[True, False, False]]]
        ),
        "gold_region_iou_targets": torch.tensor(
            [[[1.0, 0.0, 0.0]]]
        ),
        "grounding_null_prior": torch.tensor([[0.2]]),
        "metadata": [
            {
                "record_id": "r0",
                "tokens": ["Alice", "runs"],
                "region_object_labels": ["person", "street"],
                "region_object_attributes": ["", ""],
            }
        ],
    }


def test_eval_wrapper_matches_scalar_teacher_and_digest() -> None:
    teacher = _Teacher().eval()
    wrapper = LegacyStage1RecordWrapper(teacher).eval()
    report = evaluate_s3_forward_equivalence(
        teacher=teacher,
        wrapper=wrapper,
        dataloader=[_batch()],
        device=torch.device("cpu"),
    )
    assert report["gate_passed"] is True
    assert report["original_numerical_gate_passed"] is True
    assert report["amended_numerical_gate_passed"] is True
    assert report["checks"]["prediction_set"] is True
    assert (
        report["max_abs_error"]["grounding"]["formal_logits"] < 1e-6
    )


def test_wrapper_cannot_enter_training_mode() -> None:
    wrapper = LegacyStage1RecordWrapper(_Teacher().eval()).eval()
    try:
        wrapper.train()
    except RuntimeError as error:
        assert "eval-only" in str(error)
    else:
        raise AssertionError("S3.0 wrapper unexpectedly entered train mode.")


def test_formal_metric_does_not_treat_candidate_missing_as_null() -> None:
    predictions = [
        {
            "span": [0, 1],
            "type_id": ENTITY_TYPE2ID["PER"],
            "region_index": 2,
        }
    ]
    gold = [
        {
            "span": [0, 1],
            "type_id": ENTITY_TYPE2ID["PER"],
            "text": "Alice",
            # The training target falls back to NULL because no candidate
            # reached the threshold. The paper metric still sees XML boxes.
            "region_positive_indices": [2],
        }
    ]
    matches = _match_formal_record_predictions(
        predictions=predictions,
        gold=gold,
        metadata={
            "gt_boxes_by_name": {
                "alice": [[0.0, 0.0, 10.0, 10.0]],
            }
        },
        region_boxes=torch.tensor(
            [
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        null_region_index=2,
    )
    assert len(matches["span"]) == 1
    assert len(matches["mner"]) == 1
    assert len(matches["eeg"]) == 0
    assert len(matches["gmner"]) == 0


def test_amended_grounding_gate_preserves_original_failure() -> None:
    errors = {
        "raw_logits": 2.288818359375e-5,
        "after_entity_null_prior": 2.288818359375e-5,
        "after_global_null_bias": 2.288818359375e-5,
        "after_detector_prior": 2.288818359375e-5,
        "after_compatibility_prior": 2.288818359375e-5,
        "formal_logits": 2.288818359375e-5,
    }
    assert _grounding_numerical_gate(errors, 1e-5) is False
    assert _grounding_numerical_gate(errors, 3e-5) is True
