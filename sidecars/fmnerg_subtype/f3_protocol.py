"""Machine-readable contracts for the controlled FMNERG F3-P1 study."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

import yaml


class F3ProtocolError(ValueError):
    """Raised when an F3 artifact violates the preregistered protocol."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_f3_p1_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path)
    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    if raw.get("kind") != "fmnerg_subtype_f3_p1_protocol":
        raise F3ProtocolError("Not an FMNERG F3-P1 protocol.")
    if int(raw.get("format_version", -1)) != 1:
        raise F3ProtocolError("Unsupported F3-P1 protocol format.")

    candidates = list(raw.get("candidates") or [])
    candidate_ids = [str(item.get("id", "")) for item in candidates]
    if len(candidates) != 6 or any(not value for value in candidate_ids):
        raise F3ProtocolError("F3-P1 requires exactly six named candidates.")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise F3ProtocolError("F3-P1 candidate ids must be unique.")
    expected_changes = {
        ("subtype_head", 0.5),
        ("subtype_head", 2.0),
        ("backbone_upper", 0.5),
        ("backbone_upper", 2.0),
        ("backbone_lower", 0.5),
        ("backbone_lower", 2.0),
    }
    actual_changes = {
        (str(item.get("changed_group")), float(item.get("multiplier", 0.0)))
        for item in candidates
    }
    if actual_changes != expected_changes:
        raise F3ProtocolError(
            "F3-P1 must contain the six preregistered one-group LR changes."
        )

    screen = dict(raw.get("screen") or {})
    conservative_order = [
        str(value) for value in screen.get("conservative_order") or []
    ]
    if set(conservative_order) != set(candidate_ids):
        raise F3ProtocolError(
            "Conservative tie order must contain every candidate exactly once."
        )
    if int(screen.get("seed", -1)) != 42:
        raise F3ProtocolError("F3-P1 screening seed must remain 42.")
    if float(screen.get("minimum_paired_delta", -1.0)) <= 0:
        raise F3ProtocolError("Screening delta must be positive.")
    if float(screen.get("one_triple_tolerance", -1.0)) <= 0:
        raise F3ProtocolError("Tie tolerance must be positive.")
    if screen.get("advance_exactly_one_winner") is not True:
        raise F3ProtocolError("F3-P1 must advance exactly one winner.")

    baseline = dict(raw.get("baseline") or {})
    per_seed = {
        int(seed): float(value)
        for seed, value in dict(
            baseline.get("per_seed_fmnerg") or {}
        ).items()
    }
    final_gate = dict(raw.get("final_gate") or {})
    seeds = [int(value) for value in final_gate.get("seeds") or []]
    if seeds != [41, 42, 43] or set(per_seed) != set(seeds):
        raise F3ProtocolError("F3-P1 paired seeds must remain 41/42/43.")
    computed_mean = statistics.mean(per_seed[seed] for seed in seeds)
    if abs(computed_mean - float(baseline.get("mean_fmnerg"))) > 1e-12:
        raise F3ProtocolError("Stored F2 mean disagrees with per-seed baselines.")
    if float(final_gate.get("minimum_mean_paired_delta", -1.0)) <= 0:
        raise F3ProtocolError("Final mean delta must be positive.")
    if final_gate.get("require_every_paired_delta_positive") is not True:
        raise F3ProtocolError(
            "Every paired F3-P1 FMNERG delta must remain positive."
        )
    if (
        float(final_gate.get("maximum_paired_delta_population_std", -1.0))
        <= 0
    ):
        raise F3ProtocolError("Final paired-delta std limit must be positive.")
    if final_gate.get("stop_model_selection_on_pass") is not True:
        raise F3ProtocolError("F3 must stop model selection after a pass.")
    if dict(raw.get("test_contract") or {}).get(
        "enabled_during_p1"
    ) is not False:
        raise F3ProtocolError("Test must remain disabled during F3-P1.")

    raw["_protocol_path"] = str(protocol_path)
    raw["_protocol_sha256"] = sha256_file(protocol_path)
    raw["_candidate_by_id"] = {
        candidate_id: candidate
        for candidate_id, candidate in zip(candidate_ids, candidates)
    }
    raw["_baseline_by_seed"] = per_seed
    return raw


def select_seed42_winner(
    fmnerg_by_candidate: Mapping[str, float],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Select at most one candidate under the frozen screen rule."""

    candidate_by_id = dict(protocol["_candidate_by_id"])
    provided = set(fmnerg_by_candidate)
    expected = set(candidate_by_id)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise F3ProtocolError(
            f"Screen candidate mismatch; missing={missing}, extra={extra}."
        )
    screen = dict(protocol["screen"])
    seed = int(screen["seed"])
    baseline = float(protocol["_baseline_by_seed"][seed])
    minimum_delta = float(screen["minimum_paired_delta"])
    tolerance = float(screen["one_triple_tolerance"])
    order = {
        candidate_id: index
        for index, candidate_id in enumerate(
            screen["conservative_order"]
        )
    }

    rows = []
    for candidate_id in candidate_by_id:
        score = float(fmnerg_by_candidate[candidate_id])
        delta = score - baseline
        rows.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "fmnerg_f1": score,
                "baseline_fmnerg_f1": baseline,
                "paired_delta": delta,
                "passes_screen_delta": delta >= minimum_delta,
                "conservative_rank": order[candidate_id],
            }
        )
    passing = [row for row in rows if row["passes_screen_delta"]]
    if not passing:
        winner_id = None
        tied_ids: list[str] = []
        reason = "no_candidate_reached_minimum_delta"
    else:
        top_delta = max(float(row["paired_delta"]) for row in passing)
        tied = [
            row
            for row in passing
            if top_delta - float(row["paired_delta"]) <= tolerance
        ]
        winner = min(tied, key=lambda row: int(row["conservative_rank"]))
        winner_id = str(winner["candidate_id"])
        tied_ids = [
            str(row["candidate_id"])
            for row in sorted(
                tied, key=lambda row: int(row["conservative_rank"])
            )
        ]
        reason = (
            "conservative_tie_break"
            if len(tied) > 1
            else "largest_paired_delta"
        )

    return {
        "screen_seed": seed,
        "minimum_paired_delta": minimum_delta,
        "one_triple_tolerance": tolerance,
        "candidates": rows,
        "passing_candidate_count": len(passing),
        "winner_id": winner_id,
        "near_tied_candidate_ids": tied_ids,
        "selection_reason": reason,
        "advance_to_confirmation": winner_id is not None,
    }


def evaluate_three_seed_gate(
    *,
    winner_id: str,
    fmnerg_by_seed: Mapping[int, float],
    run_contract_by_seed: Mapping[int, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the preregistered paired three-seed F3-P1 gate."""

    final_gate = dict(protocol["final_gate"])
    seeds = [int(value) for value in final_gate["seeds"]]
    scores = {int(seed): float(value) for seed, value in fmnerg_by_seed.items()}
    if set(scores) != set(seeds):
        raise F3ProtocolError("Final Gate requires exactly seeds 41/42/43.")
    if set(run_contract_by_seed) != set(seeds):
        raise F3ProtocolError("Final Gate is missing run-contract metadata.")
    if winner_id not in protocol["_candidate_by_id"]:
        raise F3ProtocolError(f"Unknown winner id: {winner_id}.")

    baseline_by_seed = dict(protocol["_baseline_by_seed"])
    deltas = {
        seed: scores[seed] - float(baseline_by_seed[seed])
        for seed in seeds
    }
    mean_score = statistics.mean(scores[seed] for seed in seeds)
    mean_delta = statistics.mean(deltas[seed] for seed in seeds)
    delta_std = statistics.pstdev(deltas[seed] for seed in seeds)
    expected_gmner = float(final_gate["expected_dev_gmner_f1"])
    gmner_tolerance = float(final_gate["expected_dev_gmner_tolerance"])

    identity_ok = True
    contract_rows: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        contract = dict(run_contract_by_seed[seed])
        row_ok = (
            contract.get("gmner_identity_exact") is True
            and contract.get("formal_stage1_mutated") is False
            and contract.get("test_accessed") is False
            and abs(float(contract["gmner_f1"]) - expected_gmner)
            <= gmner_tolerance
        )
        identity_ok = identity_ok and row_ok
        contract_rows[str(seed)] = {**contract, "passed": row_ok}

    checks = {
        "mean_delta_at_least_minimum": (
            mean_delta
            >= float(final_gate["minimum_mean_paired_delta"])
        ),
        "every_paired_delta_positive": all(
            deltas[seed] > 0.0 for seed in seeds
        ),
        "paired_delta_population_std_within_limit": (
            delta_std
            <= float(final_gate["maximum_paired_delta_population_std"])
        ),
        "gmner_identity_exact": identity_ok,
    }
    passed = all(checks.values())
    return {
        "winner_id": winner_id,
        "seeds": seeds,
        "fmnerg_by_seed": {
            str(seed): scores[seed] for seed in seeds
        },
        "baseline_fmnerg_by_seed": {
            str(seed): baseline_by_seed[seed] for seed in seeds
        },
        "paired_deltas": {
            str(seed): deltas[seed] for seed in seeds
        },
        "mean_fmnerg": mean_score,
        "mean_paired_delta": mean_delta,
        "paired_delta_population_std": delta_std,
        "run_contract": contract_rows,
        "checks": checks,
        "passed": passed,
        "decision": (
            "freeze_f3_and_stop_model_selection"
            if passed
            else "p1_no_go_continue_to_p2"
        ),
    }


def load_training_summary(
    path: str | Path,
    *,
    expected_seed: int,
    expected_config_sha256: str,
    expected_gmner_f1: float | None = None,
    expected_gmner_tolerance: float = 0.0,
) -> dict[str, Any]:
    summary_path = Path(path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("kind") != "fmnerg_subtype_encoder_training_summary":
        raise F3ProtocolError(f"Unexpected summary kind in {summary_path}.")
    if metadata.get("encoder_scope") != "all":
        raise F3ProtocolError(f"F3-P1 run is not full-encoder: {summary_path}.")
    if int(metadata.get("seed", -1)) != int(expected_seed):
        raise F3ProtocolError(f"Seed mismatch in {summary_path}.")
    if metadata.get("config_sha256") != expected_config_sha256:
        raise F3ProtocolError(f"Config fingerprint mismatch in {summary_path}.")
    if metadata.get("gmner_identity_exact") is not True:
        raise F3ProtocolError(f"GMNER identity failed in {summary_path}.")
    if metadata.get("formal_stage1_mutated") is not False:
        raise F3ProtocolError(f"Formal Stage1 mutation in {summary_path}.")
    if metadata.get("test_accessed") is not False:
        raise F3ProtocolError(f"Test access detected in {summary_path}.")
    metrics = dict(payload.get("metrics") or {})
    gmner_f1 = float(metrics["gmner_f1"])
    if (
        expected_gmner_f1 is not None
        and abs(gmner_f1 - expected_gmner_f1)
        > expected_gmner_tolerance
    ):
        raise F3ProtocolError(f"GMNER metric drift in {summary_path}.")
    return {
        "path": str(summary_path),
        "seed": int(expected_seed),
        "fmnerg_f1": float(metrics["fmnerg_f1"]),
        "fine_mner_f1": float(metrics["fine_mner_f1"]),
        "gmner_f1": gmner_f1,
        "gmner_identity_exact": True,
        "formal_stage1_mutated": False,
        "test_accessed": False,
        "best_epoch": int(metadata["best_epoch"]),
        "config_sha256": str(metadata["config_sha256"]),
    }
