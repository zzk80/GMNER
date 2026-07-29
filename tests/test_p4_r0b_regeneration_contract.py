from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from gmner.data.null_release_oof_cache import (
    NULL_RELEASE_OOF_FORMAT_VERSION,
    NULL_RELEASE_OOF_KIND,
    stable_id_digest,
)
from gmner.data.p4_r0b_regeneration_contract import (
    P4_R0B_ARTIFACT_IDENTITY,
    P4_R0B_M33A_CACHE_KIND,
    P4_R0B_M33A_CACHE_VERSION,
    canonical_formal_triple_digest,
    compare_compact_semantics,
    regeneration_metadata,
    validate_fold_cleanup_path,
    validate_r0b_preregistration,
)
from scripts.run_null_release_full_chain_oof_fold import _candidate_command


AUTHORIZATION_SHA256 = "a" * 64
EXPERIMENT_ID = "p4_r0b_test"


def _batch() -> dict:
    candidate_mask = torch.ones(1, 1, 4, dtype=torch.bool)
    fine = {
        "candidate_mask": candidate_mask,
        "final_region_logits": torch.tensor([[[4.0, 3.0, 2.0, 1.0]]]),
        "fine_top4_indices": torch.tensor([[[0, 1, 2, 3]]]),
        "fine_top4_valid_mask": torch.ones(1, 1, 4, dtype=torch.bool),
        "span_grounding_state": torch.zeros(1, 1, 2),
        "region_grounding_state": torch.zeros(1, 4, 2),
        "type_grounding_state": torch.zeros(1, 1, 2),
        "candidate_source_ids": torch.zeros(1, 1, 4, dtype=torch.long),
        "base_log_prior": torch.zeros(1, 1, 4),
        "coarse_log_prior": torch.zeros(1, 1, 4),
        "base_rank": torch.zeros(1, 1, 4),
        "coarse_rank": torch.zeros(1, 1, 4),
        "detector_rank": torch.zeros(1, 1, 4),
        "fixed_type_region_compatibility": torch.zeros(1, 1, 4),
        "promoted_candidate_mask": torch.zeros(1, 1, 4, dtype=torch.bool),
        "fixed_type_ids": torch.ones(1, 1, dtype=torch.long),
    }
    expanded = {
        "span_mask": torch.ones(1, 1, dtype=torch.bool),
        "span_source_ids": torch.zeros(1, 1, dtype=torch.long),
        "gold_span_mask": torch.ones(1, 1, dtype=torch.bool),
        "visibility_targets": torch.ones(1, 1),
        "type_candidates": torch.ones(1, 1, dtype=torch.long),
        "gold_type_mask": torch.ones(1, 1, dtype=torch.bool),
        "gold_region_positive_mask": torch.tensor(
            [[[True, False, False, False]]]
        ),
        "region_mask": torch.ones(1, 4, dtype=torch.bool),
        "region_is_null": torch.tensor([[False, False, False, True]]),
        "region_detector_scores": torch.tensor([[0.9, 0.8, 0.7, 1.0]]),
    }
    return {
        "fold_id": 0,
        "record_ids": ["record-0"],
        "fine_outputs": fine,
        "hierarchy_outputs": {
            "fixed_type_ids": torch.ones(1, 1, dtype=torch.long)
        },
        "evidence_outputs": {
            "evidence_scalar_features": torch.zeros(1, 1, 2)
        },
        "expanded": expanded,
        "reliability_outputs": {
            "reliability_probability": torch.zeros(1, 1)
        },
        "current_visible": torch.ones(1, 1, dtype=torch.bool),
        "base_is_null": torch.zeros(1, 1, dtype=torch.bool),
        "deployment_span_mask": torch.ones(1, 1, dtype=torch.bool),
    }


def _payload(*, regenerated: bool) -> dict:
    metadata = {
        "format_version": (
            P4_R0B_M33A_CACHE_VERSION
            if regenerated
            else NULL_RELEASE_OOF_FORMAT_VERSION
        ),
        "kind": (
            P4_R0B_M33A_CACHE_KIND
            if regenerated
            else NULL_RELEASE_OOF_KIND
        ),
        "full_chain_oof": True,
        "fold_id": 0,
        "num_folds": 10,
        "records": 1,
        "record_ids_sha256": stable_id_digest(["record-0"]),
        "heldout_record_ids": ["record-0"],
    }
    if regenerated:
        metadata.update(
            regeneration_metadata(
                authorization_sha256=AUTHORIZATION_SHA256,
                fold_id=0,
                experiment_id=EXPERIMENT_ID,
            )
        )
        metadata.update(
            {
                "siglip2_included": False,
                "reliability_included": False,
                "null_release_included": False,
            }
        )
    else:
        metadata["includes_reliability"] = True
    batch = _batch()
    if regenerated:
        batch.pop("evidence_outputs")
        batch.pop("reliability_outputs")
    return {"metadata": metadata, "batches": [batch]}


def test_repository_preregistration_keeps_all_downstream_scopes_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "docs"
        / "experiments"
        / "p4_r0_b_full_chain_oof_regeneration_preregistration.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_r0b_preregistration(payload)
    assert payload["authorization"]["execution_folds"] == list(range(8))
    assert payload["authorization"]["p4_oracle"] is False
    assert payload["authorization"]["p4_1"] is False
    assert payload["authorization"]["test_access"] is False
    assert payload["chain_contract"]["siglip2"] is False
    assert payload["chain_contract"]["fusion_reliability"] is False
    assert payload["chain_contract"]["null_release"] is False


def test_regeneration_identity_rejects_locked_fold() -> None:
    with pytest.raises(PermissionError, match="folds 8-9"):
        regeneration_metadata(
            authorization_sha256=AUTHORIZATION_SHA256,
            fold_id=8,
            experiment_id=EXPERIMENT_ID,
        )


def test_semantic_comparison_passes_exact_payload_and_detects_change() -> None:
    reference = _payload(regenerated=False)
    regenerated = _payload(regenerated=True)
    report = compare_compact_semantics(
        reference,
        regenerated,
        fold_id=0,
        authorization_sha256=AUTHORIZATION_SHA256,
        experiment_id=EXPERIMENT_ID,
    )
    assert report["gate_passed"] is True

    changed = copy.deepcopy(regenerated)
    changed["batches"][0]["expanded"]["region_detector_scores"][0, 0] = 0.5
    report = compare_compact_semantics(
        reference,
        changed,
        fold_id=0,
        authorization_sha256=AUTHORIZATION_SHA256,
        experiment_id=EXPERIMENT_ID,
    )
    assert report["gate_passed"] is False
    detector = report["semantic_fields"]["expanded.region_detector_scores"]
    assert detector["exact_records"] == 0


def test_canonical_formal_digest_uses_regenerated_r16_coordinates() -> None:
    compact = _payload(regenerated=True)
    r16 = {
        "metadata": regeneration_metadata(
            authorization_sha256=AUTHORIZATION_SHA256,
            fold_id=0,
            experiment_id=EXPERIMENT_ID,
        ),
        "records": [
            {
                "metadata": {"record_id": "record-0"},
                "span_candidates": torch.tensor([[2, 4]]),
                "span_source_ids": torch.tensor([0]),
            }
        ],
    }
    first = canonical_formal_triple_digest(
        r16,
        compact,
        fold_id=0,
        authorization_sha256=AUTHORIZATION_SHA256,
        experiment_id=EXPERIMENT_ID,
    )
    second = canonical_formal_triple_digest(
        r16,
        compact,
        fold_id=0,
        authorization_sha256=AUTHORIZATION_SHA256,
        experiment_id=EXPERIMENT_ID,
    )
    assert first == second
    assert first["records"] == 1
    assert first["predictions"] == 1
    assert len(first["canonical_formal_triple_sha256"]) == 64


def test_cleanup_is_limited_to_child_of_authorized_fold(tmp_path: Path) -> None:
    allowed = tmp_path / "work"
    target = allowed / "fold3" / "candidates"
    assert validate_fold_cleanup_path(
        target, allowed_root=allowed, fold_id=3
    ) == target.resolve()
    with pytest.raises(ValueError, match="outside"):
        validate_fold_cleanup_path(
            tmp_path / "legacy" / "fold3" / "candidates",
            allowed_root=allowed,
            fold_id=3,
        )
    with pytest.raises(ValueError, match="retained fold root"):
        validate_fold_cleanup_path(
            allowed / "fold3",
            allowed_root=allowed,
            fold_id=3,
        )


def test_candidate_command_propagates_regeneration_identity(
    tmp_path: Path,
) -> None:
    identity = regeneration_metadata(
        authorization_sha256=AUTHORIZATION_SHA256,
        fold_id=0,
        experiment_id=EXPERIMENT_ID,
    )
    command = _candidate_command(
        python="python",
        root=tmp_path,
        config=tmp_path / "config.yaml",
        checkpoint=tmp_path / "best_model.pt",
        source=tmp_path / "heldout.jsonl",
        output=tmp_path / "heldout_r16.pt",
        max_regions=16,
        fold_id=0,
        split="train",
        batch_size=8,
        device="cuda",
        regeneration=identity,
    )
    assert command[command.index("--artifact-identity") + 1] == (
        P4_R0B_ARTIFACT_IDENTITY
    )
    assert command[
        command.index("--regeneration-authorization-sha256") + 1
    ] == AUTHORIZATION_SHA256
