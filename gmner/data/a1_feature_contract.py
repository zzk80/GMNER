"""Gold-free feature and invariant contract for the A1-0 audit."""

from __future__ import annotations

from typing import Any


FEATURE_STATUSES = (
    "DIRECTLY_AVAILABLE",
    "DERIVABLE_GOLD_FREE",
    "REQUIRES_REMATERIALIZATION",
    "FORBIDDEN",
    "MISSING",
    "SEMANTICALLY_UNSTABLE",
)


def strict_replacement_scope(
    action: dict[str, Any], base_prediction: dict[str, Any]
) -> bool:
    """Return whether an action changes only the word-space boundary."""
    return bool(
        int(action["candidate_type_id"]) == int(base_prediction["type_id"])
        and str(action["observable_features"]["candidate_region_candidate_id"])
        == str(base_prediction["region_candidate_id"])
        and action["conflict_features"]["would_preserve_prediction_count"] is True
    )


def feature_registry() -> list[dict[str, Any]]:
    direct = [
        ("candidate_source", "replacement_actions[].candidate_source", "categorical"),
        ("candidate_score", "replacement_actions[].candidate_score", "float"),
        ("base_candidate_margin", "replacement_actions[].base_candidate_margin", "float"),
        ("boundary_distance", "replacement_actions[].boundary_distance", "integer"),
        ("candidate_type_id", "replacement_actions[].candidate_type_id", "integer"),
        ("overlap_words_with_base", "replacement_actions[].conflict_features.overlap_words_with_base", "integer"),
        ("overlaps_other_formal_count", "replacement_actions[].conflict_features.overlaps_other_formal_count", "integer"),
        ("would_preserve_prediction_count", "replacement_actions[].conflict_features.would_preserve_prediction_count", "boolean"),
        ("base_span", "replacement_actions[].observable_features.base_span", "integer_pair"),
        ("candidate_span", "replacement_actions[].observable_features.candidate_span", "integer_pair"),
        ("candidate_region_score", "replacement_actions[].observable_features.candidate_region_score", "float"),
        ("base_type_id", "formal_predictions[prediction_id].type_id", "integer"),
        ("base_type_logits", "formal_predictions[prediction_id].type_logits", "float_vector_4"),
        ("base_span_score", "formal_predictions[prediction_id].observable_features.span_base_score", "float"),
        ("base_is_null", "formal_predictions[prediction_id].observable_features.base_is_null", "boolean"),
        ("final_visible", "formal_predictions[prediction_id].observable_features.final_visible", "boolean"),
        ("fine_region_logit", "formal_predictions[prediction_id].observable_features.fine_region_logit", "float"),
        ("base_region_is_null", "formal_predictions[prediction_id].region_is_null", "boolean"),
        ("candidate_type_logits", "r36_candidates.span_candidates[candidate_id].scores.type_logits", "float_vector_4"),
        ("candidate_detector_score", "r36_candidates.region_candidates[region_candidate_id].detector_score", "float"),
    ]
    derived = [
        ("candidate_in_r16", "candidate_id in r16_candidates.span_candidates", "boolean"),
        ("same_type_as_base", "candidate_type_id == base_type_id", "boolean"),
        ("same_region_as_base", "candidate_region_candidate_id == base_region_candidate_id", "boolean"),
        ("strict_a1_scope_eligible", "same_type_as_base and same_region_as_base and preserves_count", "boolean"),
        ("base_span_length", "base_span.end-base_span.start", "integer"),
        ("candidate_span_length", "candidate_span.end-candidate_span.start", "integer"),
        ("span_length_delta", "candidate_span_length-base_span_length", "integer"),
        ("left_boundary_shift", "candidate_start-base_start", "integer"),
        ("right_boundary_shift", "candidate_end-base_end", "integer"),
        ("base_type_confidence", "softmax(base_type_logits).max", "float"),
        ("base_type_margin", "softmax(base_type_logits).top1-top2", "float"),
        ("base_type_entropy", "entropy(softmax(base_type_logits))", "float"),
        ("candidate_type_confidence", "softmax(candidate_type_logits).max", "float"),
        ("candidate_type_margin", "softmax(candidate_type_logits).top1-top2", "float"),
        ("candidate_type_entropy", "entropy(softmax(candidate_type_logits))", "float"),
        ("actions_in_base_group", "count(actions grouped by base_prediction_id)", "integer"),
        ("actions_from_same_source_in_group", "count(group actions with candidate_source)", "integer"),
        ("candidate_score_rank_in_group", "descending rank(candidate_score, action_id tie-break)", "integer"),
        ("candidate_score_gap_to_group_best", "group_max_candidate_score-candidate_score", "float"),
    ]
    entries = []
    for name, source, dtype in direct:
        entries.append(
            {
                "feature_name": name,
                "source_path": "sealed_gold_free_rows",
                "source_field": source,
                "availability": "DIRECTLY_AVAILABLE",
                "expected_dtype": dtype,
                "gold_free": True,
                "sealed_before_supervision": True,
                "deterministic": True,
                "requires_rematerialization": False,
                "authorized_for_a1": True,
                "reason": "Explicit sealed observable field or deterministic identity join.",
            }
        )
    for name, source, dtype in derived:
        entries.append(
            {
                "feature_name": name,
                "source_path": "sealed_gold_free_rows",
                "source_field": source,
                "availability": "DERIVABLE_GOLD_FREE",
                "expected_dtype": dtype,
                "gold_free": True,
                "sealed_before_supervision": True,
                "deterministic": True,
                "requires_rematerialization": False,
                "authorized_for_a1": True,
                "reason": "Frozen pure function of sealed observable fields.",
            }
        )
    invariant_only = {
        "would_preserve_prediction_count",
        "same_type_as_base",
        "same_region_as_base",
        "strict_a1_scope_eligible",
    }
    for item in entries:
        if item["feature_name"] in invariant_only:
            item["authorized_for_a1"] = False
            item["reason"] = "Used to define and verify the action space, not as a model input."
    entries.extend(
        [
            {
                "feature_name": "raw_region_candidate_id",
                "source_path": "sealed_gold_free_rows",
                "source_field": "region_candidate_id",
                "availability": "DIRECTLY_AVAILABLE",
                "expected_dtype": "stable_identity",
                "gold_free": True,
                "sealed_before_supervision": True,
                "deterministic": True,
                "requires_rematerialization": False,
                "authorized_for_a1": False,
                "reason": "May be used for joins and invariants, not as a high-cardinality model input.",
            },
            {
                "feature_name": "compressed_chain_state_arrays",
                "source_path": "sealed_gold_free_rows",
                "source_field": "chain_state.* arrays",
                "availability": "SEMANTICALLY_UNSTABLE",
                "expected_dtype": "record_array",
                "gold_free": True,
                "sealed_before_supervision": True,
                "deterministic": True,
                "requires_rematerialization": False,
                "authorized_for_a1": False,
                "reason": "The schema does not freeze an identity join from every array position to prediction/action IDs.",
            },
        ]
    )
    for name in (
        "base_latent_state_z_b",
        "candidate_latent_state_z_c",
        "span_pooled_roberta_state",
        "record_context_embedding",
        "candidate_conditioned_interaction_state",
        "full_hierarchical_fine_evidence_hidden_states",
    ):
        entries.append(
            {
                "feature_name": name,
                "source_path": None,
                "source_field": None,
                "availability": "REQUIRES_REMATERIALIZATION",
                "expected_dtype": "float_vector",
                "gold_free": True,
                "sealed_before_supervision": False,
                "deterministic": "REQUIRES_NEW_PROOF",
                "requires_rematerialization": True,
                "authorized_for_a1": False,
                "reason": "Not present in sealed rows; requires a separately authorized fold-specific replay.",
            }
        )
    entries.append(
        {
            "feature_name": "record_text_tokens_or_mentions",
            "source_path": None,
            "source_field": None,
            "availability": "MISSING",
            "expected_dtype": "text",
            "gold_free": True,
            "sealed_before_supervision": False,
            "deterministic": False,
            "requires_rematerialization": True,
            "authorized_for_a1": False,
            "reason": "The sealed row stores word-space coordinates but not source tokens.",
        }
    )
    for name in (
        "protected_label",
        "metric_outcome",
        "gold_boundary_distance",
        "candidate_is_exact_gold",
        "gold_type",
        "gold_entity_count",
        "dev_manual_review_label",
        "gold_filtered_candidate_rank",
    ):
        entries.append(
            {
                "feature_name": name,
                "source_path": "supervision_sidecar_or_dev_annotation",
                "source_field": name,
                "availability": "FORBIDDEN",
                "expected_dtype": "label",
                "gold_free": False,
                "sealed_before_supervision": False,
                "deterministic": True,
                "requires_rematerialization": False,
                "authorized_for_a1": False,
                "reason": "Gold-defined supervision may only provide the training target.",
            }
        )
    return entries
