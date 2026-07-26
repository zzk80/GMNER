"""Model evaluation loops."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gmner.engine.utils import f1_counts, move_batch_to_device
from gmner.knowledge.region_compatibility import compatibility_score
from gmner.models.gmner_model import GMNERModel
from gmner.constants import (
    DEFAULT_LABEL2ID,
    ENTITY_TYPE2ID,
    IGNORE_INDEX,
)
from gmner.fmnerg.metrics import (
    fine_entities_from_bio_tags,
    subtype_classification_metrics,
)
from gmner.models.common import masked_mean
from gmner.utils.metrics import (
    extract_entities_from_word_labels,
    entity_micro_f1,
    grounding_accuracy,
    token_micro_f1,
    word_labels_from_subwords,
)
from torchvision.ops import box_iou


@torch.no_grad()
def evaluate_model(
    model: GMNERModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0

    all_token_preds = []
    all_token_labels = []
    all_region_preds = []
    all_region_labels = []
    base_region_preds = []
    reranker_only_region_preds = []
    diagnostic_region_labels = []
    grounding_loss = 0.0
    grounding_steps = 0
    grounding_base_ce_loss = 0.0
    grounding_reranker_ce_loss = 0.0
    grounding_aux_steps = 0
    grounding_positive_correct = 0
    grounding_positive_total = 0
    visible_grounding_correct = 0
    visible_grounding_total = 0
    null_grounding_correct = 0
    null_grounding_total = 0
    grounding_pred_null = 0
    grounding_predicted_iou_sum = 0.0
    grounding_oracle_iou_sum = 0.0
    grounding_iou_count = 0
    visible_predicted_iou_sum = 0.0
    visible_oracle_iou_sum = 0.0
    visible_iou_count = 0
    multiscale_token_region_correct = 0
    multiscale_span_region_correct = 0
    multiscale_local_region_correct = 0
    multiscale_region_total = 0
    multiscale_token_visible_correct = 0
    multiscale_span_visible_correct = 0
    multiscale_local_visible_correct = 0
    multiscale_token_null_correct = 0
    multiscale_span_null_correct = 0
    multiscale_local_null_correct = 0
    multiscale_visible_total = 0
    multiscale_null_total = 0
    multiscale_residual_scale_sum = 0.0
    multiscale_residual_scale_count = 0
    multiscale_decision_total = 0
    multiscale_decision_changed = 0
    multiscale_base_correct_final_wrong = 0
    multiscale_base_wrong_final_correct = 0
    multiscale_predicted_span_total = 0
    multiscale_predicted_span_changed = 0
    multiscale_predicted_span_base_eeg_final_wrong = 0
    multiscale_predicted_span_base_wrong_final_eeg = 0
    multiscale_predicted_span_base_triple_final_wrong = 0
    multiscale_predicted_span_base_wrong_final_triple = 0
    joint_visibility_probability_sum = 0.0
    joint_base_visibility_probability_sum = 0.0
    joint_visibility_residual_abs_sum = 0.0
    joint_visibility_correct = 0
    joint_visibility_visible_correct = 0
    joint_visibility_visible_total = 0
    joint_visibility_null_correct = 0
    joint_visibility_null_total = 0
    joint_visibility_count = 0
    reranker_shift_total = 0
    reranker_changed = 0
    base_correct_final_correct = 0
    base_correct_final_wrong = 0
    base_wrong_final_correct = 0
    base_wrong_final_wrong = 0
    reranker_gate_sum = 0.0
    reranker_gate_sq_sum = 0.0
    reranker_gate_count = 0
    reranker_gate_min = float("inf")
    reranker_gate_max = float("-inf")
    reranker_base_tau_sum = 0.0
    reranker_tau_sum = 0.0
    logit_stats = {
        "base": [0.0, 0.0, 0],
        "reranker": [0.0, 0.0, 0],
        "final": [0.0, 0.0, 0],
    }
    reranker_only_pred_sum = 0.0
    reranker_only_pred_count = 0
    reranker_only_null_count = 0
    reranker_gold_null_count = 0
    prototype_entity_count = 0
    prototype_gate_sum = 0.0
    prototype_reliability_sum = 0.0
    prototype_ambiguity_sum = 0.0
    prototype_accept_count = 0
    external_knowledge_type_correct = 0
    external_knowledge_base_type_correct = 0
    external_knowledge_adjusted_type_correct = 0
    external_knowledge_type_count = 0
    external_knowledge_type_confidence_sum = 0.0
    external_knowledge_disagreement_count = 0
    external_knowledge_recoverable_count = 0
    external_knowledge_harmful_count = 0
    external_knowledge_oracle_type_correct = 0
    external_knowledge_adjusted_change_count = 0
    external_knowledge_intervention_count = 0
    external_knowledge_gate_sum = 0.0
    external_knowledge_disagreement_gate_sum = 0.0
    external_knowledge_recoverable_gate_sum = 0.0
    external_knowledge_harmful_gate_sum = 0.0
    external_knowledge_subtype_correct = 0
    external_knowledge_subtype_count = 0
    external_knowledge_subtype_total = 0
    external_predicted_span_count = 0
    external_predicted_span_base_type_correct = 0
    external_predicted_span_raw_type_correct = 0
    external_predicted_span_adjusted_type_correct = 0
    external_predicted_span_base_correct_adjusted_wrong = 0
    external_predicted_span_base_wrong_adjusted_correct = 0
    external_predicted_span_disagreement_count = 0
    external_predicted_span_recoverable_count = 0
    external_predicted_span_harmful_count = 0
    external_predicted_span_oracle_type_correct = 0
    external_predicted_span_adjusted_change_count = 0
    external_predicted_span_intervention_count = 0
    external_predicted_span_gate_sum = 0.0
    external_predicted_span_disagreement_gate_sum = 0.0
    external_predicted_span_recoverable_gate_sum = 0.0
    external_predicted_span_harmful_gate_sum = 0.0
    region_score_sum = 0.0
    region_score_max_sum = 0.0
    region_score_count = 0
    region_record_count = 0
    type_confidence_sum = 0.0
    type_correct_sum = 0
    type_nll_sum = 0.0
    type_count = 0
    ece_bins = 10
    ece_confidence_sums = [0.0 for _ in range(ece_bins)]
    ece_correct_sums = [0.0 for _ in range(ece_bins)]
    ece_counts = [0 for _ in range(ece_bins)]
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}

    triple_predict = 0
    triple_gold = 0
    triple_correct = 0
    eeg_predict = 0
    eeg_gold = 0
    eeg_correct = 0
    base_triple_correct = 0
    base_eeg_correct = 0
    reranker_only_triple_correct = 0
    reranker_only_eeg_correct = 0
    joint_mner_predict = 0
    joint_mner_gold = 0
    joint_mner_correct = 0
    joint_decision_total = 0
    joint_decision_changed = 0
    joint_base_correct_final_correct = 0
    joint_base_correct_final_wrong = 0
    joint_base_wrong_final_correct = 0
    joint_base_wrong_final_wrong = 0
    joint_enabled = model.joint_type_region_verifier is not None
    seen_records = set()
    seen_ner_records = set()
    fine_enabled = model.fine_subtype_head is not None
    fine_mner_correct = 0
    fmnerg_correct = 0
    fine_prediction_count = 0
    fine_gold_count = 0
    gold_span_subtype_predictions: list[int] = []
    gold_span_subtype_targets: list[int] = []
    parent_conditioned_subtype_correct = 0
    parent_conditioned_subtype_count = 0
    hierarchy_consistent_count = 0
    hierarchy_prediction_count = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)

        if (
            "joint_visibility_logits" in outputs
            and "joint_base_visibility_logits" in outputs
            and "region_positive_mask" in batch
        ):
            positive_regions = batch["region_positive_mask"].to(dtype=torch.bool)
            active_visibility = positive_regions.any(dim=-1)
            if torch.any(active_visibility):
                visible_target = ~positive_regions[:, -1]
                visibility_logits = outputs["joint_visibility_logits"].float()
                base_visibility_logits = outputs["joint_base_visibility_logits"].float()
                visibility_pred = visibility_logits >= 0
                active_count = int(active_visibility.sum().item())
                joint_visibility_count += active_count
                joint_visibility_probability_sum += float(
                    torch.sigmoid(visibility_logits[active_visibility]).sum().item()
                )
                joint_base_visibility_probability_sum += float(
                    torch.sigmoid(base_visibility_logits[active_visibility]).sum().item()
                )
                residual = outputs.get("joint_visibility_residual_logits")
                if isinstance(residual, torch.Tensor):
                    joint_visibility_residual_abs_sum += float(
                        residual.float()[active_visibility].abs().sum().item()
                    )
                joint_visibility_correct += int(
                    (visibility_pred[active_visibility] == visible_target[active_visibility])
                    .sum()
                    .item()
                )
                visible_active = active_visibility & visible_target
                null_active = active_visibility & ~visible_target
                joint_visibility_visible_total += int(visible_active.sum().item())
                joint_visibility_null_total += int(null_active.sum().item())
                joint_visibility_visible_correct += int(
                    (visibility_pred & visible_active).sum().item()
                )
                joint_visibility_null_correct += int(
                    ((~visibility_pred) & null_active).sum().item()
                )

        if "prototype_gate" in outputs:
            valid_entities = outputs.get("prototype_valid_mask")
            if valid_entities is None:
                valid_entities = batch.get("target_type_ids")
                if valid_entities is not None:
                    valid_entities = (valid_entities >= 0) & (valid_entities < 4)
            if valid_entities is None:
                valid_entities = torch.ones_like(outputs["prototype_gate"], dtype=torch.bool)
            if torch.any(valid_entities):
                gates = outputs["prototype_gate"][valid_entities]
                reliabilities = outputs["prototype_reliability"][valid_entities]
                ambiguities = outputs["ambiguity"][valid_entities]
                prototype_entity_count += int(valid_entities.sum().item())
                prototype_gate_sum += float(gates.sum().item())
                prototype_reliability_sum += float(reliabilities.sum().item())
                prototype_ambiguity_sum += float(ambiguities.sum().item())
                prototype_accept_count += int((gates >= 0.1).sum().item())

        if (
            "external_knowledge_type_logits" in outputs
            and "target_type_ids" in batch
        ):
            external_type_logits = outputs[
                "external_knowledge_type_logits"
            ].float()
            external_type_targets = batch["target_type_ids"]
            valid_external_types = (
                (external_type_targets >= 0)
                & (external_type_targets < external_type_logits.size(-1))
            )
            if torch.any(valid_external_types):
                external_type_probs = torch.softmax(
                    external_type_logits[valid_external_types],
                    dim=-1,
                )
                external_confidence, external_prediction = (
                    external_type_probs.max(dim=-1)
                )
                external_knowledge_type_count += int(
                    valid_external_types.sum().item()
                )
                external_knowledge_type_correct += int(
                    external_prediction.eq(
                        external_type_targets[valid_external_types]
                    ).sum().item()
                )
                external_knowledge_type_confidence_sum += float(
                    external_confidence.sum().item()
                )
                valid_targets = external_type_targets[valid_external_types]
                base_external_logits = outputs.get(
                    "external_knowledge_base_type_logits"
                )
                base_external_prediction = None
                recoverable = None
                harmful = None
                disagreement = None
                if isinstance(base_external_logits, torch.Tensor):
                    base_external_prediction = base_external_logits[
                        valid_external_types
                    ].argmax(dim=-1)
                    base_correct = base_external_prediction.eq(valid_targets)
                    knowledge_correct = external_prediction.eq(valid_targets)
                    disagreement = base_external_prediction.ne(external_prediction)
                    recoverable = (~base_correct) & knowledge_correct
                    harmful = base_correct & (~knowledge_correct)
                    external_knowledge_base_type_correct += int(
                        base_correct.sum().item()
                    )
                    external_knowledge_disagreement_count += int(
                        disagreement.sum().item()
                    )
                    external_knowledge_recoverable_count += int(
                        recoverable.sum().item()
                    )
                    external_knowledge_harmful_count += int(harmful.sum().item())
                    external_knowledge_oracle_type_correct += int(
                        (base_correct | knowledge_correct).sum().item()
                    )

                    gate = outputs.get(
                        "external_knowledge_type_gate_probability",
                        outputs.get("external_knowledge_type_gate"),
                    )
                    if isinstance(gate, torch.Tensor):
                        valid_gate = gate[valid_external_types].float()
                        external_knowledge_gate_sum += float(valid_gate.sum().item())
                        external_knowledge_disagreement_gate_sum += float(
                            valid_gate[disagreement].sum().item()
                        )
                        external_knowledge_recoverable_gate_sum += float(
                            valid_gate[recoverable].sum().item()
                        )
                        external_knowledge_harmful_gate_sum += float(
                            valid_gate[harmful].sum().item()
                        )
                    effective_gate = outputs.get("external_knowledge_type_gate")
                    if isinstance(effective_gate, torch.Tensor):
                        external_knowledge_intervention_count += int(
                            (
                                effective_gate[valid_external_types].float() > 0
                            ).sum().item()
                        )
                adjusted_logits = outputs.get(
                    "external_knowledge_adjusted_type_logits"
                )
                if isinstance(adjusted_logits, torch.Tensor):
                    adjusted_prediction = adjusted_logits[
                        valid_external_types
                    ].argmax(dim=-1)
                    external_knowledge_adjusted_type_correct += int(
                        adjusted_prediction.eq(
                            external_type_targets[valid_external_types]
                        ).sum().item()
                    )
                    if base_external_prediction is not None:
                        external_knowledge_adjusted_change_count += int(
                            adjusted_prediction.ne(base_external_prediction).sum().item()
                        )

            subtype_targets = outputs.get("external_knowledge_subtype_targets")
            subtype_logits = outputs.get("external_knowledge_subtype_logits")
            if isinstance(subtype_targets, torch.Tensor) and isinstance(
                subtype_logits,
                torch.Tensor,
            ):
                external_knowledge_subtype_total += int(subtype_targets.numel())
                valid_external_subtypes = (
                    (subtype_targets != IGNORE_INDEX)
                    & (subtype_targets >= 0)
                    & (subtype_targets < subtype_logits.size(-1))
                )
                if torch.any(valid_external_subtypes):
                    external_knowledge_subtype_count += int(
                        valid_external_subtypes.sum().item()
                    )
                    external_knowledge_subtype_correct += int(
                        subtype_logits[valid_external_subtypes]
                        .argmax(dim=-1)
                        .eq(subtype_targets[valid_external_subtypes])
                        .sum()
                        .item()
                    )

        if "calibrated_base_type_logits" in outputs and "target_type_ids" in batch:
            logits = outputs["calibrated_base_type_logits"]
            targets = batch["target_type_ids"]
            if logits.size(0) == targets.size(0):
                valid_type = (targets >= 0) & (targets < logits.size(-1))
                if torch.any(valid_type):
                    valid_logits = logits[valid_type]
                    valid_targets = targets[valid_type]
                    probs = F.softmax(valid_logits, dim=-1)
                    confidence, prediction = probs.max(dim=-1)
                    correct = prediction.eq(valid_targets)
                    type_confidence_sum += float(confidence.sum().item())
                    type_correct_sum += int(correct.sum().item())
                    type_count += int(valid_targets.numel())
                    type_nll_sum += float(
                        F.cross_entropy(valid_logits, valid_targets, reduction="sum").item()
                    )
                    for conf, is_correct in zip(confidence.tolist(), correct.tolist()):
                        bin_idx = min(int(conf * ece_bins), ece_bins - 1)
                        ece_confidence_sums[bin_idx] += float(conf)
                        ece_correct_sums[bin_idx] += float(is_correct)
                        ece_counts[bin_idx] += 1

        if "loss" in outputs:
            total_loss += outputs["loss"].item()
            steps += 1

        labels = batch["ner_labels"]
        pred_tokens = model.ner_head.decode(
            outputs["ner_logits"],
            batch["attention_mask"],
            valid_mask=labels != IGNORE_INDEX,
        )

        metadata = batch.get("metadata", [])
        for sequence_idx, (pred_seq, label_seq, mask_seq) in enumerate(
            zip(pred_tokens, labels, batch["attention_mask"])
        ):
            record_id = metadata[sequence_idx].get("sample_id") if sequence_idx < len(metadata) else None
            if record_id is not None and record_id in seen_ner_records:
                continue
            if record_id is not None:
                seen_ner_records.add(record_id)
            current_preds = []
            current_labels = []
            for pred, label, mask in zip(pred_seq.tolist(), label_seq.tolist(), mask_seq.tolist()):
                if mask == 0 or label == IGNORE_INDEX:
                    continue
                current_preds.append(pred)
                current_labels.append(label)
            all_token_preds.append(current_preds)
            all_token_labels.append(current_labels)

        if "region_labels" in batch and "grounding_logits" in outputs:
            region_pred = outputs["grounding_logits"].argmax(dim=-1)
            region_labels = batch["region_labels"]
            all_region_preds.extend(region_pred.tolist())
            all_region_labels.extend(region_labels.tolist())

            positive_regions = batch.get("region_positive_mask")
            if positive_regions is not None:
                positive_regions = positive_regions.to(dtype=torch.bool)
                active_positive = positive_regions.any(dim=-1)
                row_ids = torch.arange(region_pred.size(0), device=region_pred.device)
                positive_correct = positive_regions[row_ids, region_pred] & active_positive
                null_index = positive_regions.size(1) - 1
                gold_null = positive_regions[:, null_index] & active_positive
                gold_visible = active_positive & ~gold_null
                grounding_positive_correct += int(positive_correct.sum().item())
                grounding_positive_total += int(active_positive.sum().item())
                visible_grounding_correct += int((positive_correct & gold_visible).sum().item())
                visible_grounding_total += int(gold_visible.sum().item())
                null_grounding_correct += int((positive_correct & gold_null).sum().item())
                null_grounding_total += int(gold_null.sum().item())
                grounding_pred_null += int((region_pred.eq(null_index) & active_positive).sum().item())

                multiscale_base_logits = outputs.get(
                    "multiscale_base_grounding_logits"
                )
                if multiscale_base_logits is not None:
                    multiscale_base_pred = multiscale_base_logits.argmax(dim=-1)
                    multiscale_base_correct = (
                        positive_regions[row_ids, multiscale_base_pred]
                        & active_positive
                    )
                    multiscale_final_correct = positive_correct
                    multiscale_decision_total += int(active_positive.sum().item())
                    multiscale_decision_changed += int(
                        (
                            multiscale_base_pred.ne(region_pred)
                            & active_positive
                        ).sum().item()
                    )
                    multiscale_base_correct_final_wrong += int(
                        (
                            multiscale_base_correct
                            & ~multiscale_final_correct
                            & active_positive
                        ).sum().item()
                    )
                    multiscale_base_wrong_final_correct += int(
                        (
                            ~multiscale_base_correct
                            & multiscale_final_correct
                            & active_positive
                        ).sum().item()
                    )

                multiscale_keys = (
                    ("multiscale_token_region_logits", "token"),
                    ("multiscale_span_region_logits", "span"),
                    ("multiscale_local_region_logits", "local"),
                )
                multiscale_correct = {}
                multiscale_visible_correct = {}
                multiscale_null_correct = {}
                for output_key, scale_name in multiscale_keys:
                    scale_logits = outputs.get(output_key)
                    if scale_logits is None:
                        continue
                    scale_pred = scale_logits.argmax(dim=-1)
                    scale_correct = positive_regions[row_ids, scale_pred] & active_positive
                    multiscale_correct[scale_name] = int(scale_correct.sum().item())
                    multiscale_visible_correct[scale_name] = int(
                        (scale_correct & gold_visible).sum().item()
                    )
                    multiscale_null_correct[scale_name] = int(
                        (scale_correct & gold_null).sum().item()
                    )
                if multiscale_correct:
                    multiscale_token_region_correct += multiscale_correct.get("token", 0)
                    multiscale_span_region_correct += multiscale_correct.get("span", 0)
                    multiscale_local_region_correct += multiscale_correct.get("local", 0)
                    multiscale_token_visible_correct += multiscale_visible_correct.get("token", 0)
                    multiscale_span_visible_correct += multiscale_visible_correct.get("span", 0)
                    multiscale_local_visible_correct += multiscale_visible_correct.get("local", 0)
                    multiscale_token_null_correct += multiscale_null_correct.get("token", 0)
                    multiscale_span_null_correct += multiscale_null_correct.get("span", 0)
                    multiscale_local_null_correct += multiscale_null_correct.get("local", 0)
                    multiscale_region_total += int(active_positive.sum().item())
                    multiscale_visible_total += int(gold_visible.sum().item())
                    multiscale_null_total += int(gold_null.sum().item())

                residual_scale = outputs.get("multiscale_residual_scale")
                if residual_scale is not None and residual_scale.numel() > 0:
                    multiscale_residual_scale_sum += float(
                        residual_scale.detach().float().mean().item()
                    )
                    multiscale_residual_scale_count += 1

                region_iou_targets = batch.get("region_iou_targets")
                if region_iou_targets is not None:
                    iou_quality = region_iou_targets.to(dtype=torch.float32)
                    iou_valid = batch.get("region_mask")
                    if iou_valid is None:
                        iou_valid = torch.ones_like(iou_quality, dtype=torch.bool)
                    else:
                        iou_valid = iou_valid.to(dtype=torch.bool)
                    predicted_iou = iou_quality[row_ids, region_pred]
                    oracle_iou = iou_quality.masked_fill(~iou_valid, -1.0).max(dim=-1).values
                    grounding_predicted_iou_sum += float(
                        predicted_iou[active_positive].sum().item()
                    )
                    grounding_oracle_iou_sum += float(
                        oracle_iou[active_positive].sum().item()
                    )
                    grounding_iou_count += int(active_positive.sum().item())
                    visible_predicted_iou_sum += float(
                        predicted_iou[gold_visible].sum().item()
                    )
                    visible_oracle_iou_sum += float(
                        oracle_iou[gold_visible].sum().item()
                    )
                    visible_iou_count += int(gold_visible.sum().item())

            diagnostic_valid = region_labels != IGNORE_INDEX
            if "grounding_base_logits" in outputs:
                base_pred = outputs["grounding_base_logits"].argmax(dim=-1)
                base_region_preds.extend(base_pred[diagnostic_valid].tolist())
                diagnostic_region_labels.extend(region_labels[diagnostic_valid].tolist())
                final_pred = region_pred
                active = diagnostic_valid
                if torch.any(active):
                    positive_regions = batch.get("region_positive_mask")
                    if positive_regions is not None:
                        positive_regions = positive_regions.to(dtype=torch.bool)
                        active = active & positive_regions.any(dim=-1)
                        row_ids = torch.arange(region_labels.size(0), device=region_labels.device)
                        base_correct_all = positive_regions[row_ids, base_pred]
                        final_correct_all = positive_regions[row_ids, final_pred]
                        base_correct = base_correct_all[active]
                        final_correct = final_correct_all[active]
                    else:
                        base_correct = base_pred[active].eq(region_labels[active])
                        final_correct = final_pred[active].eq(region_labels[active])
                    reranker_shift_total += int(active.sum().item())
                    reranker_changed += int(base_pred[active].ne(final_pred[active]).sum().item())
                    base_correct_final_correct += int((base_correct & final_correct).sum().item())
                    base_correct_final_wrong += int((base_correct & ~final_correct).sum().item())
                    base_wrong_final_correct += int((~base_correct & final_correct).sum().item())
                    base_wrong_final_wrong += int((~base_correct & ~final_correct).sum().item())

            if "grounding_reranker_only_logits" in outputs:
                reranker_pred = outputs["grounding_reranker_only_logits"].argmax(dim=-1)
                reranker_only_region_preds.extend(reranker_pred[diagnostic_valid].tolist())
                active_reranker_pred = reranker_pred[diagnostic_valid]
                if active_reranker_pred.numel() > 0:
                    reranker_only_pred_sum += float(active_reranker_pred.float().sum().item())
                    reranker_only_pred_count += int(active_reranker_pred.numel())
                    if outputs["grounding_reranker_only_logits"].size(1) > 0:
                        null_idx = outputs["grounding_reranker_only_logits"].size(1) - 1
                        reranker_only_null_count += int(active_reranker_pred.eq(null_idx).sum().item())
                        reranker_gold_null_count += int(region_labels[diagnostic_valid].eq(null_idx).sum().item())

            if "grounding_rerank_gate" in outputs:
                gate = outputs["grounding_rerank_gate"].detach().float()
                if gate.numel() > 0:
                    reranker_gate_sum += float(gate.sum().item())
                    reranker_gate_sq_sum += float((gate * gate).sum().item())
                    reranker_gate_count += int(gate.numel())
                    reranker_gate_min = min(reranker_gate_min, float(gate.min().item()))
                    reranker_gate_max = max(reranker_gate_max, float(gate.max().item()))
            if "grounding_rerank_base_temperature" in outputs:
                reranker_base_tau_sum += float(outputs["grounding_rerank_base_temperature"].detach().float().sum().item())
            if "grounding_rerank_temperature" in outputs:
                reranker_tau_sum += float(outputs["grounding_rerank_temperature"].detach().float().sum().item())

            region_mask_for_stats = batch.get("region_mask")
            if region_mask_for_stats is not None:
                stat_mask = region_mask_for_stats.bool()
                for stat_name, output_key in [
                    ("base", "grounding_base_logits"),
                    ("reranker", "grounding_reranker_only_logits"),
                    ("final", "grounding_logits"),
                ]:
                    if output_key not in outputs:
                        continue
                    values = outputs[output_key].detach().float()[stat_mask]
                    values = values[torch.isfinite(values) & (values > -1000.0)]
                    if values.numel() == 0:
                        continue
                    logit_stats[stat_name][0] += float(values.sum().item())
                    logit_stats[stat_name][1] += float((values * values).sum().item())
                    logit_stats[stat_name][2] += int(values.numel())

            if "loss_grounding" in outputs:
                grounding_loss += float(outputs["loss_grounding"].item())
                grounding_steps += 1
            if "loss_grounding_base_ce" in outputs:
                grounding_base_ce_loss += float(outputs["loss_grounding_base_ce"].item())
                grounding_aux_steps += 1
            if "loss_grounding_reranker_ce" in outputs:
                grounding_reranker_ce_loss += float(outputs["loss_grounding_reranker_ce"].item())

            region_scores = batch.get("region_scores")
            region_mask = batch.get("region_mask")
            if region_scores is not None and region_mask is not None:
                valid_region_mask = region_mask.bool()
                if bool(getattr(model.config.data, "add_null_region", False)) and valid_region_mask.size(1) > 0:
                    valid_region_mask = valid_region_mask.clone()
                    valid_region_mask[:, -1] = False
                if torch.any(valid_region_mask):
                    valid_scores = region_scores[valid_region_mask]
                    region_score_sum += float(valid_scores.sum().item())
                    region_score_count += int(valid_scores.numel())
                    per_record_scores = region_scores.masked_fill(~valid_region_mask, -1.0).max(dim=-1).values
                    has_region = valid_region_mask.any(dim=-1)
                    region_score_max_sum += float(per_record_scores[has_region].sum().item())
                    region_record_count += int(has_region.sum().item())

        # Paper-level triple evaluation (entity, type, region).
        if "region_features" in batch and "region_boxes" in batch:
            metadata = batch.get("metadata", [])
            region_boxes = batch["region_boxes"]
            image_nodes = outputs.get("image_nodes")
            image_mask = outputs.get("image_mask")
            prototype_source_tokens = outputs.get("pre_prototype_fused_tokens", outputs.get("fused_tokens"))
            external_knowledge_source_tokens = outputs.get("base_text_nodes")
            multiscale_source_tokens = outputs.get("text_graph_nodes")
            if image_nodes is None or image_mask is None:
                continue

            for idx, meta in enumerate(metadata):
                record_id = meta.get("sample_id")
                if record_id in seen_records:
                    continue
                seen_records.add(record_id)

                tokens = meta.get("tokens") or []
                word_ids = meta.get("word_ids") or []
                gt_boxes_by_name = meta.get("gt_boxes_by_name") or {}

                subword_pred = pred_tokens[idx].tolist()
                subword_gold = labels[idx].tolist()
                word_pred = word_labels_from_subwords(subword_pred, word_ids)
                word_gold = word_labels_from_subwords(subword_gold, word_ids)

                pred_entities = extract_entities_from_word_labels(word_pred, tokens, id2label)
                gold_entities = extract_entities_from_word_labels(word_gold, tokens, id2label)
                gold_fine_entities: list[dict] = []
                if fine_enabled:
                    fine_tags = meta.get("fine_ner_tags")
                    if not isinstance(fine_tags, list):
                        raise ValueError(
                            "Stage1-F evaluation requires fine_ner_tags in "
                            f"record {record_id}."
                        )
                    gold_fine_entities = fine_entities_from_bio_tags(
                        tokens=tokens,
                        coarse_tags=word_gold,
                        fine_tags=fine_tags,
                        taxonomy=model.fine_subtype_taxonomy,
                        coarse_id2label=id2label,
                    )
                    fine_prediction_count += len(pred_entities)
                    fine_gold_count += len(gold_fine_entities)

                no_region_index = int(region_boxes.size(1) - 1)
                record_region_boxes = region_boxes[idx]
                record_image_nodes = image_nodes[idx : idx + 1]
                record_image_mask = image_mask[idx : idx + 1]

                matched = set()
                eeg_matched = set()
                base_matched = set()
                base_eeg_matched = set()
                reranker_only_matched = set()
                reranker_only_eeg_matched = set()
                joint_mner_matched = set()
                fine_mner_matched: set[int] = set()
                fmnerg_matched: set[int] = set()
                triple_predict += len(pred_entities)
                triple_gold += len(gold_entities)
                eeg_predict += len(pred_entities)
                eeg_gold += len(gold_entities)
                if joint_enabled:
                    joint_mner_predict += len(pred_entities)
                    joint_mner_gold += len(gold_entities)

                if fine_enabled:
                    subtype_text_states = outputs["base_text_nodes"][
                        idx : idx + 1
                    ]
                    for gold_fine in gold_fine_entities:
                        gold_target_mask = torch.zeros_like(
                            batch["attention_mask"][idx],
                            dtype=torch.float32,
                        )
                        for token_pos, word_id in enumerate(word_ids):
                            if word_id is None:
                                continue
                            if (
                                int(gold_fine["start"])
                                <= word_id
                                < int(gold_fine["end"])
                            ):
                                gold_target_mask[token_pos] = 1.0
                        if gold_target_mask.sum() == 0:
                            raise ValueError(
                                "Gold fine span has no aligned subwords in "
                                f"record {record_id}."
                            )
                        gold_subtype_outputs = model.score_fine_subtypes(
                            token_states=subtype_text_states,
                            target_mask=gold_target_mask.unsqueeze(0),
                            parent_ids=torch.tensor(
                                [int(gold_fine["type_id"])],
                                dtype=torch.long,
                                device=subtype_text_states.device,
                            ),
                        )
                        gold_span_subtype_predictions.append(
                            int(
                                gold_subtype_outputs[
                                    "predicted_subtype_ids"
                                ][0].item()
                            )
                        )
                        gold_span_subtype_targets.append(
                            int(gold_fine["subtype_id"])
                        )

                for pred_ent in pred_entities:
                    pred_span = (pred_ent["start"], pred_ent["end"])
                    pred_type = pred_ent["type"]
                    base_pred_type = pred_type

                    target_mask = torch.zeros_like(batch["attention_mask"][idx], dtype=torch.float32)
                    for token_pos, word_id in enumerate(word_ids):
                        if word_id is None:
                            continue
                        if pred_ent["start"] <= word_id < pred_ent["end"]:
                            target_mask[token_pos] = 1.0
                    if target_mask.sum() == 0:
                        target_mask = batch["attention_mask"][idx].float()

                    entity_token_states = prototype_source_tokens[idx : idx + 1]
                    entity_target_mask = target_mask.unsqueeze(0).to(
                        device=entity_token_states.device,
                        dtype=entity_token_states.dtype,
                    )

                    query = masked_mean(entity_token_states, entity_target_mask)
                    reranker_query = model._entity_boundary_repr(entity_token_states, entity_target_mask)
                    base_type_logits = model._span_type_logits_from_ner(
                        outputs["base_ner_logits"][idx : idx + 1],
                        entity_target_mask,
                    )
                    span_knowledge_outputs = None
                    external_fusion_mode = "none"
                    external_type_intervened = False
                    if (
                        model.external_knowledge_bank is not None
                        and external_knowledge_source_tokens is not None
                    ):
                        external_type_names = ["LOC", "PER", "ORG", "OTHER"]
                        decoded_base_type_id = torch.tensor(
                            [
                                external_type_names.index(base_pred_type)
                                if base_pred_type in external_type_names
                                else -1
                            ],
                            device=base_type_logits.device,
                            dtype=torch.long,
                        )
                        span_knowledge_outputs = model.score_external_knowledge(
                            token_states=external_knowledge_source_tokens[
                                idx : idx + 1
                            ],
                            attention_mask=batch["attention_mask"][idx : idx + 1],
                            target_mask=entity_target_mask,
                            base_type_logits=base_type_logits,
                            base_type_ids=decoded_base_type_id,
                        )
                        base_type_logits = span_knowledge_outputs[
                            "adjusted_type_logits"
                        ]
                        external_fusion_mode = str(
                            getattr(
                                model.config.model,
                                "external_knowledge_fusion_mode",
                                "fixed",
                            )
                        ).strip().lower()
                        fixed_prior_enabled = float(
                            getattr(
                                model.config.model,
                                "external_knowledge_type_prior_weight",
                                0.0,
                            )
                        ) != 0.0
                        effective_type_gate = span_knowledge_outputs.get("type_gate")
                        if external_fusion_mode == "outcome_arbiter":
                            external_type_intervened = bool(
                                isinstance(effective_type_gate, torch.Tensor)
                                and float(effective_type_gate.item()) > 0.0
                            )
                        else:
                            external_type_intervened = (
                                external_fusion_mode == "fixed" and fixed_prior_enabled
                            )
                        if external_type_intervened:
                            type_names = ["LOC", "PER", "ORG", "OTHER"]
                            knowledge_type_id = int(
                                base_type_logits.argmax(dim=-1).item()
                            )
                            if 0 <= knowledge_type_id < len(type_names):
                                pred_type = type_names[knowledge_type_id]
                    if span_knowledge_outputs is not None:
                        exact_span_gold = next(
                            (
                                gold_entity
                                for gold_entity in gold_entities
                                if (
                                    int(gold_entity["start"]),
                                    int(gold_entity["end"]),
                                )
                                == pred_span
                            ),
                            None,
                        )
                        if exact_span_gold is not None:
                            type_names = ["LOC", "PER", "ORG", "OTHER"]
                            gold_type = str(exact_span_gold["type"])
                            raw_type_id = int(
                                span_knowledge_outputs["type_logits"]
                                .argmax(dim=-1)
                                .item()
                            )
                            base_type_id = (
                                type_names.index(base_pred_type)
                                if base_pred_type in type_names
                                else -1
                            )
                            adjusted_type_id = int(
                                span_knowledge_outputs["adjusted_type_logits"]
                                .argmax(dim=-1)
                                .item()
                            )
                            raw_type = (
                                type_names[raw_type_id]
                                if 0 <= raw_type_id < len(type_names)
                                else ""
                            )
                            adjusted_type = base_pred_type
                            if external_type_intervened:
                                adjusted_type = (
                                    type_names[adjusted_type_id]
                                    if 0 <= adjusted_type_id < len(type_names)
                                    else ""
                                )
                            base_correct = base_pred_type == gold_type
                            raw_correct = raw_type == gold_type
                            adjusted_correct = adjusted_type == gold_type
                            disagreement = base_type_id != raw_type_id
                            recoverable = (not base_correct) and raw_correct
                            harmful = base_correct and (not raw_correct)
                            external_predicted_span_count += 1
                            external_predicted_span_base_type_correct += int(
                                base_correct
                            )
                            external_predicted_span_raw_type_correct += int(
                                raw_correct
                            )
                            external_predicted_span_adjusted_type_correct += int(
                                adjusted_correct
                            )
                            external_predicted_span_base_correct_adjusted_wrong += int(
                                base_correct and not adjusted_correct
                            )
                            external_predicted_span_base_wrong_adjusted_correct += int(
                                not base_correct and adjusted_correct
                            )
                            external_predicted_span_disagreement_count += int(
                                disagreement
                            )
                            external_predicted_span_recoverable_count += int(
                                recoverable
                            )
                            external_predicted_span_harmful_count += int(harmful)
                            external_predicted_span_oracle_type_correct += int(
                                base_correct or raw_correct
                            )
                            external_predicted_span_adjusted_change_count += int(
                                adjusted_type != base_pred_type
                            )
                            external_predicted_span_intervention_count += int(
                                external_type_intervened
                            )
                            type_gate = span_knowledge_outputs.get(
                                "type_gate_probability",
                                span_knowledge_outputs.get("type_gate"),
                            )
                            if isinstance(type_gate, torch.Tensor):
                                gate_value = float(type_gate.item())
                                external_predicted_span_gate_sum += gate_value
                                if disagreement:
                                    external_predicted_span_disagreement_gate_sum += (
                                        gate_value
                                    )
                                if recoverable:
                                    external_predicted_span_recoverable_gate_sum += gate_value
                                if harmful:
                                    external_predicted_span_harmful_gate_sum += gate_value
                    grounding_logits = model.grounding_head(
                        query=query,
                        image_nodes=record_image_nodes,
                        image_mask=record_image_mask,
                    )
                    multiscale_base_decode_logits = None
                    if (
                        model.multiscale_grounding_aligner is not None
                        and multiscale_source_tokens is not None
                    ):
                        multiscale_base_decode_logits = grounding_logits
                        span_multiscale_outputs = model.multiscale_grounding_aligner(
                            token_states=multiscale_source_tokens[idx : idx + 1],
                            target_mask=entity_target_mask,
                            attention_mask=batch["attention_mask"][idx : idx + 1],
                            image_nodes=record_image_nodes,
                            image_mask=record_image_mask,
                        )
                        multiscale_weight = float(
                            getattr(
                                model.config.model,
                                "multiscale_grounding_logit_weight",
                                0.0,
                            )
                        )
                        grounding_logits = (
                            grounding_logits
                            + multiscale_weight
                            * span_multiscale_outputs["residual_scale"]
                            * span_multiscale_outputs["grounding_delta"]
                        )
                    if (
                        model.prototype_bank is not None
                        and bool(
                            getattr(
                                model.config.model,
                                "use_alignment_preserving_prototype_grounding",
                                False,
                            )
                        )
                    ):
                        prototype_token_states, _ = model._apply_semantic_prototypes(
                            token_states=entity_token_states,
                            attention_mask=batch["attention_mask"][idx : idx + 1],
                            target_masks=entity_target_mask,
                        )
                        prototype_query = masked_mean(prototype_token_states, entity_target_mask)
                        prototype_grounding_logits = model.grounding_head(
                            query=prototype_query,
                            image_nodes=record_image_nodes,
                            image_mask=record_image_mask,
                        )
                        grounding_logits = model._apply_alignment_preserving_grounding_delta(
                            base_logits=grounding_logits,
                            prototype_logits=prototype_grounding_logits,
                            image_mask=record_image_mask,
                        )
                    mini_batch = {
                        "metadata": [meta],
                    }
                    region_scores_for_rerank = batch.get("region_scores")
                    if region_scores_for_rerank is not None:
                        mini_batch["region_scores"] = region_scores_for_rerank[idx : idx + 1]
                    image_sizes = batch.get("image_sizes")
                    base_decode_logits = grounding_logits
                    grounding_logits, reranker_only_logits = model._apply_grounding_reranker(
                        logits=grounding_logits,
                        entity_repr=reranker_query,
                        image_nodes=record_image_nodes,
                        image_mask=record_image_mask,
                        batch=mini_batch,
                        base_type_logits=base_type_logits,
                        region_boxes=record_region_boxes.unsqueeze(0),
                        image_sizes=image_sizes[idx : idx + 1] if image_sizes is not None else None,
                    )

                    def apply_decode_biases(logits: torch.Tensor) -> torch.Tensor:
                        biased_logits = logits
                        region_scores = batch.get("region_scores")
                        if region_scores is not None and model.config.model.region_score_prior_weight:
                            scores = region_scores[idx].to(biased_logits.dtype).clamp(1e-4, 1.0)
                            score_bias = torch.log(scores) * model.config.model.region_score_prior_weight
                            score_bias[-1] = 0.0
                            biased_logits = biased_logits + score_bias.unsqueeze(0)
                        null_logit_bias = float(getattr(model.config.model, "grounding_null_logit_bias", 0.0))
                        if null_logit_bias and bool(getattr(model.config.data, "add_null_region", False)):
                            biased_logits = biased_logits.clone()
                            biased_logits[:, -1] = biased_logits[:, -1] + null_logit_bias
                        compatibility_weight = float(
                            getattr(model.config.model, "region_object_compatibility_weight", 0.0)
                        )
                        if compatibility_weight:
                            labels_for_regions = meta.get("region_object_labels") or []
                            attributes_for_regions = meta.get("region_object_attributes") or []
                            compatibility_bias = torch.zeros_like(biased_logits)
                            region_count = min(len(labels_for_regions), biased_logits.size(1))
                            if bool(getattr(model.config.data, "add_null_region", False)) and region_count == biased_logits.size(1):
                                region_count -= 1
                            for region_idx in range(max(region_count, 0)):
                                attribute = (
                                    attributes_for_regions[region_idx]
                                    if region_idx < len(attributes_for_regions)
                                    else ""
                                )
                                compatibility_bias[0, region_idx] = compatibility_score(
                                    pred_type,
                                    labels_for_regions[region_idx],
                                    attribute,
                                )
                            biased_logits = biased_logits + compatibility_bias * compatibility_weight
                        return biased_logits

                    base_decode_logits = apply_decode_biases(base_decode_logits)
                    grounding_logits = apply_decode_biases(grounding_logits)
                    if multiscale_base_decode_logits is not None:
                        multiscale_base_decode_logits = apply_decode_biases(
                            multiscale_base_decode_logits
                        )
                    if reranker_only_logits is not None:
                        reranker_only_logits = reranker_only_logits.masked_fill(record_image_mask == 0, -1e4)
                    joint_context_repr = masked_mean(
                        prototype_source_tokens[idx : idx + 1],
                        batch["attention_mask"][idx : idx + 1].to(
                            device=prototype_source_tokens.device,
                            dtype=prototype_source_tokens.dtype,
                        ),
                    )
                    if model.entity_evidence_decoder is not None:
                        region_scores = batch.get("region_scores")
                        evidence_batch = {"metadata": [meta]}
                        if region_scores is not None:
                            evidence_batch["region_scores"] = region_scores[idx : idx + 1]
                        evidence_outputs = model.score_entity_evidence(
                            entity_repr=query,
                            context_repr=joint_context_repr,
                            image_nodes=record_image_nodes,
                            image_mask=record_image_mask,
                            base_grounding_logits=grounding_logits,
                            base_type_logits=base_type_logits,
                            batch=evidence_batch,
                        )
                        grounding_logits = evidence_outputs["grounding_logits"]
                        if bool(getattr(model.config.model, "evidence_use_type_for_eval", True)):
                            type_names = ["LOC", "PER", "ORG", "OTHER"]
                            pred_type_idx = int(evidence_outputs["type_logits"].argmax(dim=-1).item())
                            if 0 <= pred_type_idx < len(type_names):
                                pred_type = type_names[pred_type_idx]

                    if model.joint_type_region_verifier is not None:
                        joint_outputs = model.score_joint_type_region(
                            entity_repr=query,
                            boundary_repr=reranker_query,
                            context_repr=joint_context_repr,
                            image_nodes=record_image_nodes,
                            image_mask=record_image_mask,
                            base_type_logits=base_type_logits,
                            base_region_logits=grounding_logits,
                        )
                        grounding_logits = joint_outputs["region_logits"]
                        type_names = ["LOC", "PER", "ORG", "OTHER"]
                        pred_type_idx = int(joint_outputs["type_logits"].argmax(dim=-1).item())
                        if 0 <= pred_type_idx < len(type_names):
                            pred_type = type_names[pred_type_idx]

                        for gold_idx, gold_ent in enumerate(gold_entities):
                            if gold_idx in joint_mner_matched:
                                continue
                            gold_span = (gold_ent["start"], gold_ent["end"])
                            if gold_span == pred_span and gold_ent["type"] == pred_type:
                                joint_mner_correct += 1
                                joint_mner_matched.add(gold_idx)
                                break
                    pred_subtype_id = IGNORE_INDEX
                    if fine_enabled:
                        parent_id = ENTITY_TYPE2ID.get(
                            pred_type,
                            IGNORE_INDEX,
                        )
                        subtype_outputs = model.score_fine_subtypes(
                            token_states=subtype_text_states,
                            target_mask=entity_target_mask,
                            parent_ids=torch.tensor(
                                [parent_id],
                                dtype=torch.long,
                                device=subtype_text_states.device,
                            ),
                        )
                        pred_subtype_id = int(
                            subtype_outputs["predicted_subtype_ids"][
                                0
                            ].item()
                        )
                        hierarchy_prediction_count += 1
                        hierarchy_consistent_count += int(
                            model.fine_subtype_taxonomy.parent_id(
                                pred_subtype_id
                            )
                            == parent_id
                        )
                        for gold_fine in gold_fine_entities:
                            if (
                                tuple(gold_fine["span"]) == pred_span
                                and int(gold_fine["type_id"])
                                == parent_id
                            ):
                                parent_conditioned_subtype_count += 1
                                parent_conditioned_subtype_correct += int(
                                    pred_subtype_id
                                    == int(gold_fine["subtype_id"])
                                )
                                break
                    pred_region_index = int(grounding_logits.argmax(dim=-1).item())
                    base_pred_region_index = int(base_decode_logits.argmax(dim=-1).item())
                    multiscale_base_pred_region_index = (
                        int(multiscale_base_decode_logits.argmax(dim=-1).item())
                        if multiscale_base_decode_logits is not None
                        else pred_region_index
                    )
                    reranker_only_pred_region_index = (
                        int(reranker_only_logits.argmax(dim=-1).item())
                        if reranker_only_logits is not None
                        else base_pred_region_index
                    )

                    def region_matches(region_index: int, gold_entity: dict) -> bool:
                        gold_name = str(gold_entity["text"]).strip().lower()
                        gt_boxes = gt_boxes_by_name.get(gold_name, [])
                        if not gt_boxes:
                            return region_index == no_region_index
                        if region_index == no_region_index:
                            return False
                        pred_box = record_region_boxes[region_index].unsqueeze(0)
                        gt_box_tensor = torch.tensor(gt_boxes, dtype=pred_box.dtype, device=pred_box.device)
                        ious = box_iou(gt_box_tensor, pred_box).squeeze(1)
                        if fine_enabled:
                            threshold = float(
                                getattr(
                                    model.config.data,
                                    "grounding_iou_threshold",
                                    0.5,
                                )
                            )
                            return bool((ious >= threshold).any().item())
                        return bool((ious > 0.5).any().item())

                    def triple_is_correct(region_index: int, entity_type: str) -> bool:
                        for gold_ent in gold_entities:
                            gold_span = (gold_ent["start"], gold_ent["end"])
                            if gold_span != pred_span or gold_ent["type"] != entity_type:
                                continue
                            return region_matches(region_index, gold_ent)
                        return False

                    if fine_enabled:
                        for gold_idx, gold_fine in enumerate(
                            gold_fine_entities
                        ):
                            if (
                                tuple(gold_fine["span"]) != pred_span
                                or int(gold_fine["subtype_id"])
                                != pred_subtype_id
                            ):
                                continue
                            if gold_idx not in fine_mner_matched:
                                fine_mner_correct += 1
                                fine_mner_matched.add(gold_idx)
                            if (
                                gold_idx not in fmnerg_matched
                                and region_matches(
                                    pred_region_index,
                                    gold_fine,
                                )
                            ):
                                fmnerg_correct += 1
                                fmnerg_matched.add(gold_idx)
                            break

                    if multiscale_base_decode_logits is not None:
                        multiscale_base_eeg_correct = any(
                            (gold_ent["start"], gold_ent["end"]) == pred_span
                            and region_matches(
                                multiscale_base_pred_region_index,
                                gold_ent,
                            )
                            for gold_ent in gold_entities
                        )
                        multiscale_final_eeg_correct = any(
                            (gold_ent["start"], gold_ent["end"]) == pred_span
                            and region_matches(pred_region_index, gold_ent)
                            for gold_ent in gold_entities
                        )
                        multiscale_base_triple_correct = triple_is_correct(
                            multiscale_base_pred_region_index,
                            base_pred_type,
                        )
                        multiscale_final_triple_correct = triple_is_correct(
                            pred_region_index,
                            pred_type,
                        )
                        multiscale_predicted_span_total += 1
                        multiscale_predicted_span_changed += int(
                            multiscale_base_pred_region_index != pred_region_index
                        )
                        multiscale_predicted_span_base_eeg_final_wrong += int(
                            multiscale_base_eeg_correct
                            and not multiscale_final_eeg_correct
                        )
                        multiscale_predicted_span_base_wrong_final_eeg += int(
                            not multiscale_base_eeg_correct
                            and multiscale_final_eeg_correct
                        )
                        multiscale_predicted_span_base_triple_final_wrong += int(
                            multiscale_base_triple_correct
                            and not multiscale_final_triple_correct
                        )
                        multiscale_predicted_span_base_wrong_final_triple += int(
                            not multiscale_base_triple_correct
                            and multiscale_final_triple_correct
                        )

                    if joint_enabled:
                        base_tuple_correct = triple_is_correct(
                            base_pred_region_index,
                            base_pred_type,
                        )
                        final_tuple_correct = triple_is_correct(
                            pred_region_index,
                            pred_type,
                        )
                        joint_decision_total += 1
                        joint_decision_changed += int(
                            base_pred_region_index != pred_region_index
                            or base_pred_type != pred_type
                        )
                        joint_base_correct_final_correct += int(
                            base_tuple_correct and final_tuple_correct
                        )
                        joint_base_correct_final_wrong += int(
                            base_tuple_correct and not final_tuple_correct
                        )
                        joint_base_wrong_final_correct += int(
                            not base_tuple_correct and final_tuple_correct
                        )
                        joint_base_wrong_final_wrong += int(
                            not base_tuple_correct and not final_tuple_correct
                        )

                    def update_eeg_count(region_index: int, used: set[int]) -> bool:
                        for gold_idx, gold_ent in enumerate(gold_entities):
                            if gold_idx in used:
                                continue
                            gold_span = (gold_ent["start"], gold_ent["end"])
                            if gold_span != pred_span:
                                continue
                            if region_matches(region_index, gold_ent):
                                used.add(gold_idx)
                                return True
                            break
                        return False

                    def update_triple_count(
                        region_index: int,
                        entity_type: str,
                        used: set[int],
                    ) -> bool:
                        for gold_idx, gold_ent in enumerate(gold_entities):
                            if gold_idx in used:
                                continue
                            gold_span = (gold_ent["start"], gold_ent["end"])
                            gold_type = gold_ent["type"]
                            if gold_span != pred_span or gold_type != entity_type:
                                continue
                            if region_matches(region_index, gold_ent):
                                used.add(gold_idx)
                                return True
                            break
                        return False

                    if update_eeg_count(base_pred_region_index, base_eeg_matched):
                        base_eeg_correct += 1
                    if update_triple_count(base_pred_region_index, base_pred_type, base_matched):
                        base_triple_correct += 1
                    if update_eeg_count(reranker_only_pred_region_index, reranker_only_eeg_matched):
                        reranker_only_eeg_correct += 1
                    if update_triple_count(
                        reranker_only_pred_region_index,
                        base_pred_type,
                        reranker_only_matched,
                    ):
                        reranker_only_triple_correct += 1

                    for gold_idx, gold_ent in enumerate(gold_entities):
                        if gold_idx in eeg_matched:
                            continue
                        gold_span = (gold_ent["start"], gold_ent["end"])
                        if gold_span != pred_span:
                            continue

                        gold_name = gold_ent["text"].strip().lower()
                        gt_boxes = gt_boxes_by_name.get(gold_name, [])
                        if not gt_boxes:
                            region_ok = pred_region_index == no_region_index
                        else:
                            if pred_region_index == no_region_index:
                                region_ok = False
                            else:
                                pred_box = record_region_boxes[pred_region_index].unsqueeze(0)
                                gt_box_tensor = torch.tensor(gt_boxes, dtype=pred_box.dtype, device=pred_box.device)
                                ious = box_iou(gt_box_tensor, pred_box).squeeze(1)
                                region_ok = bool((ious > 0.5).any().item())

                        if region_ok:
                            eeg_correct += 1
                            eeg_matched.add(gold_idx)
                        break

                    for gold_idx, gold_ent in enumerate(gold_entities):
                        if gold_idx in matched:
                            continue
                        gold_span = (gold_ent["start"], gold_ent["end"])
                        gold_type = gold_ent["type"]
                        if gold_span != pred_span or gold_type != pred_type:
                            continue

                        gold_name = gold_ent["text"].strip().lower()
                        gt_boxes = gt_boxes_by_name.get(gold_name, [])
                        if not gt_boxes:
                            region_ok = pred_region_index == no_region_index
                        else:
                            if pred_region_index == no_region_index:
                                region_ok = False
                            else:
                                pred_box = record_region_boxes[pred_region_index].unsqueeze(0)
                                gt_box_tensor = torch.tensor(gt_boxes, dtype=pred_box.dtype, device=pred_box.device)
                                ious = box_iou(gt_box_tensor, pred_box).squeeze(1)
                                region_ok = bool((ious > 0.5).any().item())

                        if region_ok:
                            triple_correct += 1
                            matched.add(gold_idx)
                        break

    metrics = token_micro_f1(all_token_preds, all_token_labels)
    metrics.update(entity_micro_f1(all_token_preds, all_token_labels))
    if joint_enabled:
        metrics["base_entity_precision"] = metrics["entity_precision"]
        metrics["base_entity_recall"] = metrics["entity_recall"]
        metrics["base_entity_f1"] = metrics["entity_f1"]
        joint_precision = joint_mner_correct / max(joint_mner_predict, 1)
        joint_recall = joint_mner_correct / max(joint_mner_gold, 1)
        joint_f1 = 2 * joint_precision * joint_recall / max(
            joint_precision + joint_recall,
            1e-8,
        )
        metrics["entity_precision"] = joint_precision
        metrics["entity_recall"] = joint_recall
        metrics["entity_f1"] = joint_f1
        metrics["joint_entity_correct"] = float(joint_mner_correct)
        metrics["joint_entity_predict"] = float(joint_mner_predict)
        metrics["joint_entity_gold"] = float(joint_mner_gold)
    if all_region_labels:
        metrics.update(grounding_accuracy(all_region_preds, all_region_labels))
        if grounding_positive_total > 0:
            metrics["grounding_multi_positive_accuracy"] = (
                grounding_positive_correct / grounding_positive_total
            )
            metrics["visible_grounding_accuracy"] = (
                visible_grounding_correct / max(visible_grounding_total, 1)
            )
            metrics["null_grounding_accuracy"] = (
                null_grounding_correct / max(null_grounding_total, 1)
            )
            metrics["grounding_pred_null_ratio"] = (
                grounding_pred_null / grounding_positive_total
            )
            metrics["grounding_gold_null_ratio"] = (
                null_grounding_total / grounding_positive_total
            )
        if grounding_iou_count > 0:
            metrics["grounding_predicted_iou_mean"] = (
                grounding_predicted_iou_sum / grounding_iou_count
            )
            metrics["grounding_oracle_iou_mean"] = (
                grounding_oracle_iou_sum / grounding_iou_count
            )
        if visible_iou_count > 0:
            metrics["visible_predicted_iou_mean"] = (
                visible_predicted_iou_sum / visible_iou_count
            )
            metrics["visible_oracle_iou_mean"] = (
                visible_oracle_iou_sum / visible_iou_count
            )
        if multiscale_region_total > 0:
            metrics["multiscale_token_region_accuracy"] = (
                multiscale_token_region_correct / multiscale_region_total
            )
            metrics["multiscale_span_region_accuracy"] = (
                multiscale_span_region_correct / multiscale_region_total
            )
            metrics["multiscale_local_region_accuracy"] = (
                multiscale_local_region_correct / multiscale_region_total
            )
            metrics["multiscale_token_visible_accuracy"] = (
                multiscale_token_visible_correct / max(multiscale_visible_total, 1)
            )
            metrics["multiscale_span_visible_accuracy"] = (
                multiscale_span_visible_correct / max(multiscale_visible_total, 1)
            )
            metrics["multiscale_local_visible_accuracy"] = (
                multiscale_local_visible_correct / max(multiscale_visible_total, 1)
            )
            metrics["multiscale_token_null_accuracy"] = (
                multiscale_token_null_correct / max(multiscale_null_total, 1)
            )
            metrics["multiscale_span_null_accuracy"] = (
                multiscale_span_null_correct / max(multiscale_null_total, 1)
            )
            metrics["multiscale_local_null_accuracy"] = (
                multiscale_local_null_correct / max(multiscale_null_total, 1)
            )
        if multiscale_residual_scale_count > 0:
            metrics["multiscale_residual_scale"] = (
                multiscale_residual_scale_sum / multiscale_residual_scale_count
            )
        if multiscale_decision_total > 0:
            metrics["multiscale_prediction_changed_ratio"] = (
                multiscale_decision_changed / multiscale_decision_total
            )
            metrics["multiscale_base_correct_final_wrong"] = float(
                multiscale_base_correct_final_wrong
            )
            metrics["multiscale_base_wrong_final_correct"] = float(
                multiscale_base_wrong_final_correct
            )
            metrics["multiscale_net_corrections"] = float(
                multiscale_base_wrong_final_correct
                - multiscale_base_correct_final_wrong
            )
        if multiscale_predicted_span_total > 0:
            metrics["multiscale_predicted_span_changed_ratio"] = (
                multiscale_predicted_span_changed
                / multiscale_predicted_span_total
            )
            metrics["multiscale_predicted_span_base_eeg_final_wrong"] = float(
                multiscale_predicted_span_base_eeg_final_wrong
            )
            metrics["multiscale_predicted_span_base_wrong_final_eeg"] = float(
                multiscale_predicted_span_base_wrong_final_eeg
            )
            metrics["multiscale_predicted_span_eeg_net_corrections"] = float(
                multiscale_predicted_span_base_wrong_final_eeg
                - multiscale_predicted_span_base_eeg_final_wrong
            )
            metrics[
                "multiscale_predicted_span_base_triple_final_wrong"
            ] = float(multiscale_predicted_span_base_triple_final_wrong)
            metrics[
                "multiscale_predicted_span_base_wrong_final_triple"
            ] = float(multiscale_predicted_span_base_wrong_final_triple)
            metrics["multiscale_predicted_span_triple_net_corrections"] = float(
                multiscale_predicted_span_base_wrong_final_triple
                - multiscale_predicted_span_base_triple_final_wrong
            )
        metrics["grounding_loss"] = grounding_loss / max(grounding_steps, 1)
        if grounding_aux_steps > 0:
            metrics["reranker_base_ce_loss"] = grounding_base_ce_loss / max(grounding_aux_steps, 1)
            metrics["reranker_only_ce_loss"] = grounding_reranker_ce_loss / max(grounding_aux_steps, 1)
        if "entity_f1" in metrics:
            metrics["gmner_score"] = 0.5 * (metrics["entity_f1"] + metrics["grounding_accuracy"])
    if joint_visibility_count > 0:
        metrics["joint_visibility_probability_mean"] = (
            joint_visibility_probability_sum / joint_visibility_count
        )
        metrics["joint_base_visibility_probability_mean"] = (
            joint_base_visibility_probability_sum / joint_visibility_count
        )
        metrics["joint_visibility_residual_abs_mean"] = (
            joint_visibility_residual_abs_sum / joint_visibility_count
        )
        metrics["joint_visibility_accuracy"] = (
            joint_visibility_correct / joint_visibility_count
        )
        metrics["joint_visibility_visible_recall"] = (
            joint_visibility_visible_correct / max(joint_visibility_visible_total, 1)
        )
        metrics["joint_visibility_null_accuracy"] = (
            joint_visibility_null_correct / max(joint_visibility_null_total, 1)
        )

    if triple_gold > 0:
        precision = triple_correct / max(triple_predict, 1)
        recall = triple_correct / max(triple_gold, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics["triple_precision"] = precision
        metrics["triple_recall"] = recall
        metrics["triple_f1"] = f1
        metrics["triple_correct"] = float(triple_correct)
        metrics["triple_predict"] = float(triple_predict)
        metrics["triple_gold"] = float(triple_gold)
        metrics["gmner_score"] = f1
        base_precision = base_triple_correct / max(triple_predict, 1)
        base_recall = base_triple_correct / max(triple_gold, 1)
        base_f1 = 2 * base_precision * base_recall / max(base_precision + base_recall, 1e-8)
        reranker_precision = reranker_only_triple_correct / max(triple_predict, 1)
        reranker_recall = reranker_only_triple_correct / max(triple_gold, 1)
        reranker_f1 = 2 * reranker_precision * reranker_recall / max(reranker_precision + reranker_recall, 1e-8)
        metrics["reranker_base_triple_f1"] = base_f1
        metrics["reranker_base_triple_correct"] = float(base_triple_correct)
        metrics["reranker_only_triple_f1"] = reranker_f1
        metrics["reranker_only_triple_correct"] = float(reranker_only_triple_correct)
        if joint_decision_total > 0:
            metrics["joint_prediction_changed_ratio"] = (
                joint_decision_changed / joint_decision_total
            )
            metrics["joint_base_correct_final_correct"] = float(
                joint_base_correct_final_correct
            )
            metrics["joint_base_correct_final_wrong"] = float(
                joint_base_correct_final_wrong
            )
            metrics["joint_base_wrong_final_correct"] = float(
                joint_base_wrong_final_correct
            )
            metrics["joint_base_wrong_final_wrong"] = float(
                joint_base_wrong_final_wrong
            )
            metrics["joint_net_corrections"] = float(
                joint_base_wrong_final_correct - joint_base_correct_final_wrong
            )
    if eeg_gold > 0:
        eeg_precision = eeg_correct / max(eeg_predict, 1)
        eeg_recall = eeg_correct / max(eeg_gold, 1)
        eeg_f1 = 2 * eeg_precision * eeg_recall / max(eeg_precision + eeg_recall, 1e-8)
        metrics["eeg_precision"] = eeg_precision
        metrics["eeg_recall"] = eeg_recall
        metrics["eeg_f1"] = eeg_f1
        metrics["eeg_correct"] = float(eeg_correct)
        metrics["eeg_predict"] = float(eeg_predict)
        metrics["eeg_gold"] = float(eeg_gold)
        base_eeg_precision = base_eeg_correct / max(eeg_predict, 1)
        base_eeg_recall = base_eeg_correct / max(eeg_gold, 1)
        base_eeg_f1 = 2 * base_eeg_precision * base_eeg_recall / max(
            base_eeg_precision + base_eeg_recall,
            1e-8,
        )
        reranker_eeg_precision = reranker_only_eeg_correct / max(eeg_predict, 1)
        reranker_eeg_recall = reranker_only_eeg_correct / max(eeg_gold, 1)
        reranker_eeg_f1 = 2 * reranker_eeg_precision * reranker_eeg_recall / max(
            reranker_eeg_precision + reranker_eeg_recall,
            1e-8,
        )
        metrics["reranker_base_eeg_f1"] = base_eeg_f1
        metrics["reranker_base_eeg_correct"] = float(base_eeg_correct)
        metrics["reranker_only_eeg_f1"] = reranker_eeg_f1
        metrics["reranker_only_eeg_correct"] = float(reranker_only_eeg_correct)
    if prototype_entity_count > 0:
        metrics["prototype_gate_mean"] = prototype_gate_sum / prototype_entity_count
        metrics["prototype_reliability_mean"] = prototype_reliability_sum / prototype_entity_count
        metrics["prototype_ambiguity_mean"] = prototype_ambiguity_sum / prototype_entity_count
        metrics["prototype_acceptance_rate"] = prototype_accept_count / prototype_entity_count
    if external_knowledge_type_count > 0:
        metrics["external_knowledge_base_type_accuracy"] = (
            external_knowledge_base_type_correct / external_knowledge_type_count
        )
        metrics["external_knowledge_type_accuracy"] = (
            external_knowledge_type_correct / external_knowledge_type_count
        )
        metrics["external_knowledge_adjusted_type_accuracy"] = (
            external_knowledge_adjusted_type_correct
            / external_knowledge_type_count
        )
        metrics["external_knowledge_type_confidence_mean"] = (
            external_knowledge_type_confidence_sum
            / external_knowledge_type_count
        )
        metrics["external_knowledge_disagreement_rate"] = (
            external_knowledge_disagreement_count / external_knowledge_type_count
        )
        metrics["external_knowledge_recoverable_count"] = float(
            external_knowledge_recoverable_count
        )
        metrics["external_knowledge_harmful_count"] = float(
            external_knowledge_harmful_count
        )
        metrics["external_knowledge_oracle_type_accuracy"] = (
            external_knowledge_oracle_type_correct / external_knowledge_type_count
        )
        metrics["external_knowledge_adjusted_change_rate"] = (
            external_knowledge_adjusted_change_count / external_knowledge_type_count
        )
        metrics["external_knowledge_intervention_rate"] = (
            external_knowledge_intervention_count / external_knowledge_type_count
        )
        metrics["external_knowledge_gate_mean"] = (
            external_knowledge_gate_sum / external_knowledge_type_count
        )
        metrics["external_knowledge_disagreement_gate_mean"] = (
            external_knowledge_disagreement_gate_sum
            / max(external_knowledge_disagreement_count, 1)
        )
        metrics["external_knowledge_recoverable_gate_mean"] = (
            external_knowledge_recoverable_gate_sum
            / max(external_knowledge_recoverable_count, 1)
        )
        metrics["external_knowledge_harmful_gate_mean"] = (
            external_knowledge_harmful_gate_sum
            / max(external_knowledge_harmful_count, 1)
        )
    if external_knowledge_subtype_total > 0:
        metrics["external_knowledge_subtype_coverage"] = (
            external_knowledge_subtype_count
            / external_knowledge_subtype_total
        )
    if external_knowledge_subtype_count > 0:
        metrics["external_knowledge_subtype_accuracy"] = (
            external_knowledge_subtype_correct
            / external_knowledge_subtype_count
        )
    if external_predicted_span_count > 0:
        metrics["external_predicted_span_count"] = float(
            external_predicted_span_count
        )
        metrics["external_predicted_span_base_type_accuracy"] = (
            external_predicted_span_base_type_correct
            / external_predicted_span_count
        )
        metrics["external_predicted_span_raw_type_accuracy"] = (
            external_predicted_span_raw_type_correct
            / external_predicted_span_count
        )
        metrics["external_predicted_span_adjusted_type_accuracy"] = (
            external_predicted_span_adjusted_type_correct
            / external_predicted_span_count
        )
        metrics["external_base_correct_adjusted_wrong"] = float(
            external_predicted_span_base_correct_adjusted_wrong
        )
        metrics["external_base_wrong_adjusted_correct"] = float(
            external_predicted_span_base_wrong_adjusted_correct
        )
        metrics["external_net_type_corrections"] = float(
            external_predicted_span_base_wrong_adjusted_correct
            - external_predicted_span_base_correct_adjusted_wrong
        )
        metrics["external_predicted_span_disagreement_rate"] = (
            external_predicted_span_disagreement_count
            / external_predicted_span_count
        )
        metrics["external_predicted_span_recoverable_count"] = float(
            external_predicted_span_recoverable_count
        )
        metrics["external_predicted_span_harmful_count"] = float(
            external_predicted_span_harmful_count
        )
        metrics["external_predicted_span_oracle_type_accuracy"] = (
            external_predicted_span_oracle_type_correct
            / external_predicted_span_count
        )
        metrics["external_predicted_span_adjusted_change_rate"] = (
            external_predicted_span_adjusted_change_count
            / external_predicted_span_count
        )
        metrics["external_predicted_span_intervention_rate"] = (
            external_predicted_span_intervention_count
            / external_predicted_span_count
        )
        metrics["external_predicted_span_gate_mean"] = (
            external_predicted_span_gate_sum / external_predicted_span_count
        )
        metrics["external_predicted_span_disagreement_gate_mean"] = (
            external_predicted_span_disagreement_gate_sum
            / max(external_predicted_span_disagreement_count, 1)
        )
        metrics["external_predicted_span_recoverable_gate_mean"] = (
            external_predicted_span_recoverable_gate_sum
            / max(external_predicted_span_recoverable_count, 1)
        )
        metrics["external_predicted_span_harmful_gate_mean"] = (
            external_predicted_span_harmful_gate_sum
            / max(external_predicted_span_harmful_count, 1)
        )
    if region_score_count > 0:
        metrics["region_score_mean"] = region_score_sum / region_score_count
        metrics["region_score_max_mean"] = region_score_max_sum / max(region_record_count, 1)
    if diagnostic_region_labels:
        if base_region_preds:
            base_metrics = grounding_accuracy(base_region_preds, diagnostic_region_labels)
            metrics["reranker_base_grounding_accuracy"] = base_metrics["grounding_accuracy"]
            metrics["reranker_base_grounding_coverage"] = base_metrics["grounding_coverage"]
        if reranker_only_region_preds:
            reranker_metrics = grounding_accuracy(reranker_only_region_preds, diagnostic_region_labels)
            metrics["reranker_only_grounding_accuracy"] = reranker_metrics["grounding_accuracy"]
            metrics["reranker_only_grounding_coverage"] = reranker_metrics["grounding_coverage"]
    if reranker_shift_total > 0:
        metrics["reranker_prediction_changed_ratio"] = reranker_changed / reranker_shift_total
        metrics["reranker_base_correct_final_correct"] = float(base_correct_final_correct)
        metrics["reranker_base_correct_final_wrong"] = float(base_correct_final_wrong)
        metrics["reranker_base_wrong_final_correct"] = float(base_wrong_final_correct)
        metrics["reranker_base_wrong_final_wrong"] = float(base_wrong_final_wrong)
        metrics["reranker_net_corrections"] = float(base_wrong_final_correct - base_correct_final_wrong)
    if reranker_gate_count > 0:
        gate_mean = reranker_gate_sum / reranker_gate_count
        gate_var = max(reranker_gate_sq_sum / reranker_gate_count - gate_mean * gate_mean, 0.0)
        metrics["reranker_gate_mean"] = gate_mean
        metrics["reranker_gate_std"] = gate_var ** 0.5
        metrics["reranker_gate_min"] = reranker_gate_min
        metrics["reranker_gate_max"] = reranker_gate_max
        metrics["reranker_tau_base"] = reranker_base_tau_sum / reranker_gate_count
        metrics["reranker_tau"] = reranker_tau_sum / reranker_gate_count
    if reranker_only_pred_count > 0:
        metrics["reranker_only_pred_index_mean"] = reranker_only_pred_sum / reranker_only_pred_count
        metrics["reranker_only_null_ratio"] = reranker_only_null_count / reranker_only_pred_count
        metrics["reranker_gold_null_ratio"] = reranker_gold_null_count / reranker_only_pred_count
    for stat_name, (value_sum, sq_sum, count) in logit_stats.items():
        if count <= 0:
            continue
        mean = value_sum / count
        var = max(sq_sum / count - mean * mean, 0.0)
        metrics[f"reranker_{stat_name}_logit_mean"] = mean
        metrics[f"reranker_{stat_name}_logit_std"] = var ** 0.5
    if type_count > 0:
        ece = 0.0
        for confidence_sum, correct_sum, count in zip(
            ece_confidence_sums,
            ece_correct_sums,
            ece_counts,
        ):
            if count == 0:
                continue
            avg_confidence = confidence_sum / count
            avg_accuracy = correct_sum / count
            ece += (count / type_count) * abs(avg_confidence - avg_accuracy)
        metrics["type_confidence_mean"] = type_confidence_sum / type_count
        metrics["type_accuracy"] = type_correct_sum / type_count
        metrics["type_nll"] = type_nll_sum / type_count
        metrics["type_ece"] = ece
    if fine_enabled:
        fine_precision, fine_recall, fine_f1 = f1_counts(
            fine_mner_correct,
            fine_prediction_count,
            fine_gold_count,
        )
        fmnerg_precision, fmnerg_recall, fmnerg_f1 = f1_counts(
            fmnerg_correct,
            fine_prediction_count,
            fine_gold_count,
        )
        metrics.update(
            {
                "fine_mner_precision": fine_precision,
                "fine_mner_recall": fine_recall,
                "fine_mner_f1": fine_f1,
                "fine_mner_correct": float(fine_mner_correct),
                "fmnerg_precision": fmnerg_precision,
                "fmnerg_recall": fmnerg_recall,
                "fmnerg_f1": fmnerg_f1,
                "fmnerg_correct": float(fmnerg_correct),
                "fmnerg_score": fmnerg_f1,
                "fine_prediction_count": float(fine_prediction_count),
                "fine_gold_count": float(fine_gold_count),
                "parent_conditioned_subtype_accuracy": (
                    parent_conditioned_subtype_correct
                    / max(parent_conditioned_subtype_count, 1)
                ),
                "parent_conditioned_subtype_count": float(
                    parent_conditioned_subtype_count
                ),
                "hierarchy_consistency": (
                    hierarchy_consistent_count
                    / max(hierarchy_prediction_count, 1)
                ),
            }
        )
        if fmnerg_f1 > fine_f1 + 1e-12:
            raise AssertionError("FMNERG F1 exceeds Fine MNER F1.")
        if "eeg_f1" in metrics and fmnerg_f1 > metrics["eeg_f1"] + 1e-12:
            raise AssertionError("FMNERG F1 exceeds EEG F1.")
        gold_span_metrics = subtype_classification_metrics(
            gold_span_subtype_predictions,
            gold_span_subtype_targets,
            num_classes=model.fine_subtype_head.num_subtypes,
        )
        metrics.update(
            {
                "gold_span_subtype_accuracy": gold_span_metrics[
                    "subtype_accuracy"
                ],
                "gold_span_subtype_micro_f1": gold_span_metrics[
                    "subtype_micro_f1"
                ],
                "gold_span_subtype_macro_f1": gold_span_metrics[
                    "subtype_macro_f1"
                ],
            }
        )
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics
