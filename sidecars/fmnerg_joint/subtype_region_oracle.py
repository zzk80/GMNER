"""Deterministic visual-prototype probe for subtype-region evidence."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


@dataclass(frozen=True)
class VisualPrototypeBank:
    """Train-only subtype centroids in the frozen R36 feature space."""

    prototypes: torch.Tensor
    counts: torch.Tensor

    @property
    def available(self) -> torch.Tensor:
        return self.counts.gt(0)


def build_visual_prototype_bank(
    entity_features: Iterable[tuple[int, torch.Tensor]],
    *,
    num_subtypes: int,
    feature_size: int,
) -> VisualPrototypeBank:
    sums = torch.zeros(num_subtypes, feature_size, dtype=torch.float32)
    counts = torch.zeros(num_subtypes, dtype=torch.long)
    for raw_subtype_id, raw_feature in entity_features:
        subtype_id = int(raw_subtype_id)
        if subtype_id < 0 or subtype_id >= num_subtypes:
            raise ValueError(f"Invalid visual prototype subtype id: {subtype_id}.")
        feature = torch.as_tensor(raw_feature, dtype=torch.float32).reshape(-1)
        if feature.numel() != feature_size:
            raise ValueError(
                f"Expected visual feature size {feature_size}, "
                f"found {feature.numel()}."
            )
        if not torch.isfinite(feature).all() or float(feature.norm().item()) <= 1e-8:
            continue
        sums[subtype_id] += F.normalize(feature, dim=0)
        counts[subtype_id] += 1
    prototypes = torch.zeros_like(sums)
    available = counts.gt(0)
    if available.any():
        prototypes[available] = F.normalize(sums[available], dim=-1)
    return VisualPrototypeBank(prototypes=prototypes, counts=counts)


def probe_region_feature(
    feature: torch.Tensor,
    *,
    bank: VisualPrototypeBank,
    taxonomy: SubtypeTaxonomy,
    parent_id: int,
    gold_subtype_id: int,
    predicted_subtype_id: int | None,
) -> dict[str, Any]:
    vector = torch.as_tensor(feature, dtype=torch.float32).reshape(-1)
    if (
        vector.numel() != bank.prototypes.size(1)
        or not torch.isfinite(vector).all()
        or float(vector.norm().item()) <= 1e-8
    ):
        return {"available": False, "reason": "invalid_region_feature"}
    gold_id = int(gold_subtype_id)
    if not bool(bank.available[gold_id].item()):
        return {"available": False, "reason": "gold_prototype_missing"}
    parent_mask = torch.tensor(
        [
            taxonomy.parent_id(subtype_id) == int(parent_id)
            for subtype_id in range(taxonomy.num_subtypes)
        ],
        dtype=torch.bool,
    )
    candidate_mask = parent_mask & bank.available
    if int(candidate_mask.sum().item()) < 2:
        return {"available": False, "reason": "sibling_prototypes_missing"}
    scores = bank.prototypes @ F.normalize(vector, dim=0)
    gold_score = float(scores[gold_id].item())
    competitors = candidate_mask.clone()
    competitors[gold_id] = False
    best_other_score, best_other_id = scores.masked_fill(
        ~competitors, -torch.inf
    ).max(dim=0)
    sibling_margin = gold_score - float(best_other_score.item())
    output: dict[str, Any] = {
        "available": True,
        "gold_score": gold_score,
        "best_other_score": float(best_other_score.item()),
        "best_other_subtype_id": int(best_other_id.item()),
        "sibling_margin": sibling_margin,
        "gold_beats_all_siblings": sibling_margin > 0.0,
        "available_sibling_prototypes": int(candidate_mask.sum().item()),
        "total_sibling_subtypes": int(parent_mask.sum().item()),
    }
    if (
        predicted_subtype_id is not None
        and int(predicted_subtype_id) != gold_id
    ):
        predicted_id = int(predicted_subtype_id)
        if (
            predicted_id < 0
            or predicted_id >= taxonomy.num_subtypes
            or taxonomy.parent_id(predicted_id) != int(parent_id)
        ):
            raise ValueError("Predicted subtype is outside the fixed parent.")
        if bool(bank.available[predicted_id].item()):
            predicted_score = float(scores[predicted_id].item())
            pairwise_margin = gold_score - predicted_score
            output.update(
                {
                    "predicted_score": predicted_score,
                    "pairwise_margin": pairwise_margin,
                    "gold_beats_predicted": pairwise_margin > 0.0,
                }
            )
        else:
            output["pairwise_reason"] = "predicted_prototype_missing"
    return output


def _best_probe(
    probes: list[tuple[int, dict[str, Any]]],
    margin_name: str,
) -> tuple[int, dict[str, Any]] | None:
    available = [
        (region_index, probe)
        for region_index, probe in probes
        if probe.get("available") and margin_name in probe
    ]
    if not available:
        return None
    return max(available, key=lambda item: float(item[1][margin_name]))


def probe_positive_candidate_set(
    *,
    candidate_indices: Iterable[int],
    positive_region_indices: set[int],
    region_features: torch.Tensor,
    bank: VisualPrototypeBank,
    taxonomy: SubtypeTaxonomy,
    parent_id: int,
    gold_subtype_id: int,
    predicted_subtype_id: int | None,
) -> dict[str, Any]:
    ordered: list[int] = []
    observed: set[int] = set()
    for raw_index in candidate_indices:
        index = int(raw_index)
        if index in observed:
            continue
        observed.add(index)
        ordered.append(index)
    positives = [index for index in ordered if index in positive_region_indices]
    probes = [
        (
            index,
            probe_region_feature(
                region_features[index],
                bank=bank,
                taxonomy=taxonomy,
                parent_id=parent_id,
                gold_subtype_id=gold_subtype_id,
                predicted_subtype_id=predicted_subtype_id,
            ),
        )
        for index in positives
    ]
    sibling = _best_probe(probes, "sibling_margin")
    pairwise = _best_probe(probes, "pairwise_margin")
    return {
        "candidate_count": len(ordered),
        "gold_positive_candidate_count": len(positives),
        "gold_positive_candidate_indices": positives,
        "scorable_positive_count": sum(
            int(probe.get("available", False)) for _, probe in probes
        ),
        "best_sibling_region_index": (
            sibling[0] if sibling is not None else None
        ),
        "best_sibling_margin": (
            float(sibling[1]["sibling_margin"])
            if sibling is not None
            else None
        ),
        "gold_beats_all_siblings": bool(
            sibling is not None
            and float(sibling[1]["sibling_margin"]) > 0.0
        ),
        "best_pairwise_region_index": (
            pairwise[0] if pairwise is not None else None
        ),
        "best_pairwise_margin": (
            float(pairwise[1]["pairwise_margin"])
            if pairwise is not None
            else None
        ),
        "gold_beats_predicted": bool(
            pairwise is not None
            and float(pairwise[1]["pairwise_margin"]) > 0.0
        ),
    }


def analyze_visible_error(
    *,
    formal_region_index: int,
    fine_ranked_region_indices: list[int],
    positive_region_indices: set[int],
    all_real_region_indices: list[int],
    region_features: torch.Tensor,
    bank: VisualPrototypeBank,
    taxonomy: SubtypeTaxonomy,
    parent_id: int,
    gold_subtype_id: int,
    predicted_subtype_id: int,
    top_ks: list[int],
) -> dict[str, Any]:
    if int(formal_region_index) not in positive_region_indices:
        raise ValueError(
            "GMNER-correct visible prediction must select a gold-positive region."
        )
    ordered = [int(formal_region_index)]
    ordered.extend(
        index
        for index in map(int, fine_ranked_region_indices)
        if index != int(formal_region_index)
    )
    formal_probe = probe_region_feature(
        region_features[int(formal_region_index)],
        bank=bank,
        taxonomy=taxonomy,
        parent_id=parent_id,
        gold_subtype_id=gold_subtype_id,
        predicted_subtype_id=predicted_subtype_id,
    )
    top_k = {
        str(value): probe_positive_candidate_set(
            candidate_indices=ordered[: int(value)],
            positive_region_indices=positive_region_indices,
            region_features=region_features,
            bank=bank,
            taxonomy=taxonomy,
            parent_id=parent_id,
            gold_subtype_id=gold_subtype_id,
            predicted_subtype_id=predicted_subtype_id,
        )
        for value in top_ks
    }
    full = probe_positive_candidate_set(
        candidate_indices=all_real_region_indices,
        positive_region_indices=positive_region_indices,
        region_features=region_features,
        bank=bank,
        taxonomy=taxonomy,
        parent_id=parent_id,
        gold_subtype_id=gold_subtype_id,
        predicted_subtype_id=predicted_subtype_id,
    )
    return {
        "formal_region": formal_probe,
        "top_k": top_k,
        "full_r36_positive_oracle": full,
    }


def _distribution(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0.0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": float(len(finite)),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _summarize_candidate_probe(
    probes: list[dict[str, Any]],
    *,
    visible_error_count: int,
    formal_probes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pairwise = [
        probe
        for probe in probes
        if probe.get("best_pairwise_margin") is not None
    ]
    sibling = [
        probe
        for probe in probes
        if probe.get("best_sibling_margin") is not None
    ]
    result: dict[str, Any] = {
        "entities": float(len(probes)),
        "pairwise_scorable": float(len(pairwise)),
        "pairwise_support_count": float(
            sum(int(probe["gold_beats_predicted"]) for probe in pairwise)
        ),
        "pairwise_support_rate_among_scorable": (
            sum(int(probe["gold_beats_predicted"]) for probe in pairwise)
            / max(len(pairwise), 1)
        ),
        "pairwise_support_rate_among_visible_errors": (
            sum(int(probe["gold_beats_predicted"]) for probe in pairwise)
            / max(int(visible_error_count), 1)
        ),
        "sibling_scorable": float(len(sibling)),
        "sibling_top1_count": float(
            sum(int(probe["gold_beats_all_siblings"]) for probe in sibling)
        ),
        "sibling_top1_rate_among_scorable": (
            sum(int(probe["gold_beats_all_siblings"]) for probe in sibling)
            / max(len(sibling), 1)
        ),
        "sibling_top1_rate_among_visible_errors": (
            sum(int(probe["gold_beats_all_siblings"]) for probe in sibling)
            / max(int(visible_error_count), 1)
        ),
        "pairwise_margin": _distribution(
            [float(probe["best_pairwise_margin"]) for probe in pairwise]
        ),
        "sibling_margin": _distribution(
            [float(probe["best_sibling_margin"]) for probe in sibling]
        ),
        "gold_positive_candidates": _distribution(
            [
                float(probe["gold_positive_candidate_count"])
                for probe in probes
            ]
        ),
    }
    if formal_probes is not None:
        pairwise_recovered = sibling_recovered = 0
        for formal, candidate in zip(formal_probes, probes):
            pairwise_recovered += int(
                formal.get("pairwise_margin") is not None
                and float(formal["pairwise_margin"]) <= 0.0
                and candidate.get("best_pairwise_margin") is not None
                and float(candidate["best_pairwise_margin"]) > 0.0
            )
            sibling_recovered += int(
                formal.get("sibling_margin") is not None
                and float(formal["sibling_margin"]) <= 0.0
                and candidate.get("best_sibling_margin") is not None
                and float(candidate["best_sibling_margin"]) > 0.0
            )
        result.update(
            {
                "incremental_pairwise_recovery_over_formal": float(
                    pairwise_recovered
                ),
                "incremental_pairwise_recovery_rate_over_visible_errors": (
                    pairwise_recovered / max(int(visible_error_count), 1)
                ),
                "incremental_sibling_recovery_over_formal": float(
                    sibling_recovered
                ),
                "incremental_sibling_recovery_rate_over_visible_errors": (
                    sibling_recovered / max(int(visible_error_count), 1)
                ),
            }
        )
    return result


def summarize_seed_rows(
    rows: list[dict[str, Any]],
    *,
    top_ks: list[int],
    formal_prediction_count: int,
    gmner_correct_count: int,
) -> dict[str, Any]:
    visible = [row for row in rows if row["visibility"] == "visible"]
    null = [row for row in rows if row["visibility"] == "null"]
    formal_probes = [
        dict(row["visual_evidence"]["formal_region"]) for row in visible
    ]
    formal_pairwise = [
        probe for probe in formal_probes if probe.get("pairwise_margin") is not None
    ]
    formal_sibling = [
        probe for probe in formal_probes if probe.get("sibling_margin") is not None
    ]
    formal_summary = {
        "entities": float(len(visible)),
        "pairwise_scorable": float(len(formal_pairwise)),
        "pairwise_support_count": float(
            sum(
                int(float(probe["pairwise_margin"]) > 0.0)
                for probe in formal_pairwise
            )
        ),
        "pairwise_support_rate_among_scorable": (
            sum(
                int(float(probe["pairwise_margin"]) > 0.0)
                for probe in formal_pairwise
            )
            / max(len(formal_pairwise), 1)
        ),
        "pairwise_support_rate_among_visible_errors": (
            sum(
                int(float(probe["pairwise_margin"]) > 0.0)
                for probe in formal_pairwise
            )
            / max(len(visible), 1)
        ),
        "sibling_scorable": float(len(formal_sibling)),
        "sibling_top1_count": float(
            sum(
                int(float(probe["sibling_margin"]) > 0.0)
                for probe in formal_sibling
            )
        ),
        "sibling_top1_rate_among_scorable": (
            sum(
                int(float(probe["sibling_margin"]) > 0.0)
                for probe in formal_sibling
            )
            / max(len(formal_sibling), 1)
        ),
        "sibling_top1_rate_among_visible_errors": (
            sum(
                int(float(probe["sibling_margin"]) > 0.0)
                for probe in formal_sibling
            )
            / max(len(visible), 1)
        ),
        "pairwise_margin": _distribution(
            [float(probe["pairwise_margin"]) for probe in formal_pairwise]
        ),
        "sibling_margin": _distribution(
            [float(probe["sibling_margin"]) for probe in formal_sibling]
        ),
    }
    by_parent = Counter(str(row["coarse_type"]) for row in rows)
    confusions = Counter(
        f"{row['predicted_subtype']}->{row['gold_subtype']}" for row in rows
    )
    return {
        "formal_predictions": float(formal_prediction_count),
        "gmner_correct_predictions": float(gmner_correct_count),
        "subtype_wrong_given_gmner_correct": float(len(rows)),
        "visible_subtype_errors": float(len(visible)),
        "null_subtype_errors": float(len(null)),
        "visible_error_rate": len(visible) / max(len(rows), 1),
        "errors_by_parent": dict(by_parent),
        "top_confusions": dict(confusions.most_common(25)),
        "formal_region_probe": formal_summary,
        "fine_top_k_positive_oracle": {
            str(value): _summarize_candidate_probe(
                [
                    dict(row["visual_evidence"]["top_k"][str(value)])
                    for row in visible
                ],
                visible_error_count=len(visible),
                formal_probes=formal_probes,
            )
            for value in top_ks
        },
        "full_r36_positive_oracle": _summarize_candidate_probe(
            [
                dict(row["visual_evidence"]["full_r36_positive_oracle"])
                for row in visible
            ],
            visible_error_count=len(visible),
            formal_probes=formal_probes,
        ),
    }
