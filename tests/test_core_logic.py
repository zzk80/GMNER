import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from gmner.losses.multitask import (
    alignment_objective,
    base_top1_hard_negative_margin_loss,
    hard_negative_margin_loss,
    iou_aware_region_ranking_loss,
    masked_cross_entropy,
    multi_positive_region_loss,
    weighted_masked_cross_entropy,
)
from gmner.models.heads import GroundingResidualAdapter, TokenClassificationHead
from gmner.models.multiscale_grounding import MultiScaleGroundingAligner
from gmner.utils.bio import entity_masks_from_bio, first_entity_mask_from_bio
from gmner.utils.io import maybe_convert_conll
from gmner.utils.metrics import entity_micro_f1, span_micro_f1, token_micro_f1


def test_token_f1_excludes_o_labels():
    metrics = token_micro_f1([[0, 0, 1]], [[0, 0, 2]])

    assert metrics["token_f1"] == 0.0


def test_entity_f1_requires_exact_span_and_type():
    exact = entity_micro_f1([[1, 2, 0]], [[1, 2, 0]])
    wrong_type = entity_micro_f1([[3, 4, 0]], [[1, 2, 0]])

    assert exact["entity_f1"] > 0.999
    assert wrong_type["entity_f1"] == 0.0


def test_span_f1_ignores_type_but_requires_exact_boundary():
    wrong_type = span_micro_f1([[3, 4, 0]], [[1, 2, 0]])
    wrong_boundary = span_micro_f1([[1, 0, 0]], [[1, 2, 0]])

    assert wrong_type["span_f1"] > 0.999
    assert wrong_boundary["span_f1"] == 0.0


def test_alignment_objective_prefers_matching_pairs():
    matching = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    mismatching = torch.tensor([[0.0, 10.0], [10.0, 0.0]])

    assert alignment_objective(matching) < alignment_objective(mismatching)


def test_alignment_objective_supports_duplicate_positive_pairs():
    scores = torch.tensor([[10.0, 9.0, 0.0], [9.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    positives = torch.tensor(
        [[True, True, False], [True, True, False], [False, False, True]]
    )

    assert alignment_objective(scores, positive_mask=positives) < 0.001


def test_hard_negative_margin_loss_penalizes_stronger_wrong_region():
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.2, 1.1, 0.0]])
    labels = torch.tensor([0, 0])
    valid_mask = torch.ones_like(logits, dtype=torch.bool)

    loss = hard_negative_margin_loss(
        logits=logits,
        labels=labels,
        valid_mask=valid_mask,
        margin=0.2,
    )

    assert torch.allclose(loss, torch.tensor(0.55))


def test_hard_negative_margin_loss_is_zero_when_gold_has_margin():
    logits = torch.tensor([[2.0, 1.0, -1.0]])
    labels = torch.tensor([0])
    valid_mask = torch.ones_like(logits, dtype=torch.bool)

    loss = hard_negative_margin_loss(
        logits=logits,
        labels=labels,
        valid_mask=valid_mask,
        margin=0.2,
    )

    assert loss.item() == 0.0


def test_weighted_masked_cross_entropy_emphasizes_selected_samples():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0])
    weights = torch.tensor([0.1, 3.0])

    weighted = weighted_masked_cross_entropy(logits, labels, weights)
    unweighted = masked_cross_entropy(logits, labels)

    assert weighted > unweighted


def test_base_top1_hard_negative_margin_loss_uses_base_wrong_region_only():
    logits = torch.tensor([[1.0, 1.1, 0.0], [0.2, 1.5, 0.0]])
    base_logits = torch.tensor([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0]])
    labels = torch.tensor([0, 0])
    valid_mask = torch.ones_like(logits, dtype=torch.bool)

    loss = base_top1_hard_negative_margin_loss(
        logits=logits,
        labels=labels,
        valid_mask=valid_mask,
        base_logits=base_logits,
        margin=0.2,
    )

    assert torch.allclose(loss, torch.tensor(0.3))


def test_multi_positive_region_loss_accepts_any_positive_region():
    logits = torch.tensor([[0.0, 3.0, 2.5], [2.0, 0.0, -1.0]])
    positives = torch.tensor([[False, True, True], [True, False, False]])
    valid_mask = torch.ones_like(positives)

    loss = multi_positive_region_loss(logits, positives, valid_mask)

    assert loss < F.cross_entropy(logits, torch.tensor([1, 0]))


def test_iou_aware_region_ranking_prefers_iou_ordered_scores():
    quality = torch.tensor([[1.0, 0.5, 0.0]])
    valid_mask = torch.ones_like(quality, dtype=torch.bool)
    ordered = torch.tensor([[3.0, 2.0, 0.0]])
    reversed_scores = torch.tensor([[0.0, 2.0, 3.0]])

    ordered_loss = iou_aware_region_ranking_loss(
        logits=ordered,
        iou_targets=quality,
        valid_mask=valid_mask,
        margin=0.2,
        min_iou_gap=0.1,
    )
    reversed_loss = iou_aware_region_ranking_loss(
        logits=reversed_scores,
        iou_targets=quality,
        valid_mask=valid_mask,
        margin=0.2,
        min_iou_gap=0.1,
    )

    assert ordered_loss < reversed_loss


def test_multiscale_grounding_aligner_masks_regions_and_preserves_shapes():
    aligner = MultiScaleGroundingAligner(
        hidden_size=8,
        projection_dim=4,
        dropout=0.0,
        has_null_region=True,
    )
    token_states = torch.randn(2, 5, 8)
    target_mask = torch.tensor(
        [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
        dtype=torch.float32,
    )
    image_nodes = torch.randn(2, 4, 8)
    image_mask = torch.tensor(
        [[1, 1, 0, 1], [1, 1, 1, 1]],
        dtype=torch.float32,
    )

    outputs = aligner(
        token_states=token_states,
        target_mask=target_mask,
        attention_mask=attention_mask,
        image_nodes=image_nodes,
        image_mask=image_mask,
    )

    assert outputs["token_region_logits"].shape == (2, 4)
    assert outputs["span_region_logits"].shape == (2, 4)
    assert outputs["sentence_image_scores"].shape == (2, 2)
    assert outputs["grounding_delta"].shape == (2, 4)
    assert outputs["token_region_logits"][0, 2].item() == -1e4
    assert outputs["grounding_delta"][0, 2].item() == 0.0
    assert outputs["residual_scale"].item() == 0.0


def test_multi_positive_region_loss_supports_visible_null_reweighting():
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    positives = torch.tensor([[True, False], [False, True]])
    visible_heavy = multi_positive_region_loss(
        logits,
        positives,
        sample_weight=torch.tensor([3.0, 1.0]),
    )
    null_heavy = multi_positive_region_loss(
        logits,
        positives,
        sample_weight=torch.tensor([1.0, 3.0]),
    )

    assert visible_heavy < null_heavy


def test_grounding_loss_can_ignore_invalid_region_labels():
    logits = torch.tensor([[2.0, 1.0, -1e4], [0.0, 1.0, -1e4]])
    labels = torch.tensor([0, 2])
    valid_region_mask = torch.tensor([[True, True, False], [True, True, False]])
    row_ids = torch.arange(labels.size(0))
    safe_labels = labels.clamp_min(0).clamp_max(logits.size(1) - 1)
    valid_labels = valid_region_mask[row_ids, safe_labels]
    filtered_labels = labels.masked_fill(~valid_labels, -100)

    loss = masked_cross_entropy(logits, filtered_labels)

    assert torch.allclose(loss, torch.nn.functional.cross_entropy(logits[:1], labels[:1]))


def test_grounding_loss_can_ignore_masked_gold_logits():
    logits = torch.tensor([[2.0, 1.0, -1e4], [0.0, -1e4, 2.0]])
    labels = torch.tensor([0, 1])
    valid_region_mask = torch.ones_like(logits, dtype=torch.bool)
    row_ids = torch.arange(labels.size(0))
    safe_labels = labels.clamp_min(0).clamp_max(logits.size(1) - 1)
    gold_logits = logits[row_ids, safe_labels]
    valid_labels = valid_region_mask[row_ids, safe_labels] & torch.isfinite(gold_logits) & (gold_logits > -1000.0)
    filtered_labels = labels.masked_fill(~valid_labels, -100)

    loss = masked_cross_entropy(logits, filtered_labels)

    assert torch.allclose(loss, torch.nn.functional.cross_entropy(logits[:1], labels[:1]))


def test_grounding_residual_adapter_is_noop_at_initialization():
    adapter = GroundingResidualAdapter(hidden_size=4, max_delta=0.5)
    query = torch.randn(2, 4)
    image_nodes = torch.randn(2, 3, 4)
    image_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    delta = adapter(query=query, image_nodes=image_nodes, image_mask=image_mask)

    assert torch.allclose(delta, torch.zeros_like(delta))


def test_token_loss_ignores_subwords_and_duplicate_samples():
    head = TokenClassificationHead(hidden_size=2, num_labels=2)
    logits = torch.tensor(
        [[[0.0, 2.0], [10.0, 0.0]], [[2.0, 0.0], [0.0, 10.0]]]
    )
    labels = torch.tensor([[1, -100], [0, -100]])
    attention_mask = torch.ones_like(labels)

    loss = head.compute_loss(
        logits,
        labels,
        attention_mask,
        sample_weight=torch.tensor([1.0, 0.0]),
    )

    assert torch.allclose(loss, F.cross_entropy(logits[0, :1], labels[0, :1]))


def test_fractional_sample_weight_is_not_renormalized_away():
    head = TokenClassificationHead(hidden_size=2, num_labels=2)
    logits = torch.tensor([[[0.0, 2.0]], [[0.0, 1.0]]])
    labels = torch.tensor([[1], [0]])
    attention_mask = torch.ones_like(labels)
    weights = torch.tensor([1.0, 0.5])

    loss = head.compute_loss(logits, labels, attention_mask, sample_weight=weights)
    sample_losses = F.cross_entropy(logits[:, 0], labels[:, 0], reduction="none")
    expected = (sample_losses[0] + 0.5 * sample_losses[1]) / 2.0

    assert torch.allclose(loss, expected)


def test_predicted_entity_mask_uses_first_bio_span():
    decoded = torch.tensor(
        [
            [0, 1, 2, 0, 3],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    valid_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [False, True, True, True, False],
        ],
        dtype=torch.bool,
    )

    mask, type_ids = first_entity_mask_from_bio(decoded, valid_mask)

    assert mask.tolist() == [[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]
    assert type_ids.tolist() == [1, 4]


def test_entity_masks_from_bio_keeps_all_spans():
    decoded = torch.tensor(
        [
            [0, 1, 2, 0, 3, 4],
            [0, 5, 6, 7, 8, 0],
        ],
        dtype=torch.long,
    )
    valid_mask = torch.ones_like(decoded, dtype=torch.bool)

    masks, type_ids, valid_entities = entity_masks_from_bio(decoded, valid_mask)

    assert masks.shape == (2, 2, 6)
    assert masks[0].tolist() == [
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ]
    assert masks[1].tolist() == [
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
    ]
    assert type_ids.tolist() == [[1, 0], [2, 3]]
    assert valid_entities.tolist() == [[True, True], [True, True]]


def test_dataset_reads_predicted_evidence_entities():
    pytest.importorskip("transformers")
    from gmner.data.mmner_dataset import MMNERJsonDataset

    dataset = object.__new__(MMNERJsonDataset)
    tokens = ["Golden", "State", "Warriors", "win"]
    record = {
        "evidence_entities": [
            {
                "start": 0,
                "end": 3,
                "text": "Golden State Warriors",
                "predicted_type": "OTHER",
                "target_type": "ORG",
                "target_subtype": "sports_team",
            }
        ]
    }

    entities = dataset._extract_evidence_entities(tokens, record)

    assert entities == [
        (0, 3, "Golden State Warriors", "ORG", "sports_team", False, "OTHER")
    ]


def test_dataset_region_targets_keep_continuous_iou_quality():
    pytest.importorskip("transformers")
    from gmner.data.mmner_dataset import MMNERJsonDataset

    dataset = object.__new__(MMNERJsonDataset)
    dataset.max_regions = 2
    dataset.add_null_region = True
    dataset.grounding_iou_threshold = 0.5
    candidate_boxes = np.asarray(
        [[0, 0, 10, 10], [0, 0, 5, 5], [0, 0, 0, 0]],
        dtype=np.float32,
    )
    candidate_mask = np.asarray([1, 1, 1], dtype=np.float32)

    label, positives, iou_targets = dataset._select_region_targets(
        entity_text="Alice",
        boxes_by_name={"alice": [[0, 0, 10, 10]]},
        candidate_boxes=candidate_boxes,
        candidate_mask=candidate_mask,
    )

    assert label == 0
    assert positives.tolist() == [1.0, 0.0, 0.0]
    assert iou_targets.tolist() == [1.0, 0.25, 0.0]


def test_maybe_convert_conll_rebuilds_stale_cache(tmp_path: Path):
    source = tmp_path / "test.txt"
    output_dir = tmp_path / "outputs"
    source.write_text("IMGID:img1\nAlice B-PER\n\n", encoding="utf-8")

    converted = maybe_convert_conll(source, output_dir)
    first = json.loads(converted.read_text(encoding="utf-8").strip())
    assert first["tokens"] == ["Alice"]

    source.write_text("IMGID:img1\nAlice B-PER\nSmith I-PER\n\n", encoding="utf-8")
    source.touch()
    converted = maybe_convert_conll(source, output_dir)
    second = json.loads(converted.read_text(encoding="utf-8").strip())
    assert second["tokens"] == ["Alice", "Smith"]
