"""Summarize the preregistered FMNERG F3-P1 learning-rate study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from sidecars.fmnerg_subtype.f3_protocol import (
    F3ProtocolError,
    evaluate_three_seed_gate,
    load_f3_p1_protocol,
    load_training_summary,
    select_seed42_winner,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="sidecars/fmnerg_subtype/f3_p1_protocol.yaml",
    )
    parser.add_argument(
        "--root",
        default="outputs/fmnerg_subtype_f3_p1",
        help="Root containing <candidate>/seed<seed>/train_summary.json.",
    )
    parser.add_argument("--stage", choices=("screen", "final"), required=True)
    parser.add_argument("--screen-summary", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve_from_repo(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_candidate_run(
    *,
    repo_root: Path,
    output_root: Path,
    candidate: dict[str, Any],
    seed: int,
    expected_gmner_f1: float,
    expected_gmner_tolerance: float,
) -> dict[str, Any]:
    config_path = resolve_from_repo(candidate["config"], repo_root)
    if not config_path.is_file():
        raise F3ProtocolError(f"Missing candidate config: {config_path}.")
    config_sha256 = sha256_file(config_path)
    summary_path = (
        output_root
        / str(candidate["id"])
        / f"seed{seed}"
        / "train_summary.json"
    )
    if not summary_path.is_file():
        raise F3ProtocolError(f"Missing training summary: {summary_path}.")
    run = load_training_summary(
        summary_path,
        expected_seed=seed,
        expected_config_sha256=config_sha256,
        expected_gmner_f1=expected_gmner_f1,
        expected_gmner_tolerance=expected_gmner_tolerance,
    )
    return {
        **run,
        "candidate_id": str(candidate["id"]),
        "candidate_config": str(config_path),
        "candidate_config_sha256": config_sha256,
    }


def summarize_screen(
    *,
    protocol: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    seed = int(protocol["screen"]["seed"])
    runs = [
        load_candidate_run(
            repo_root=repo_root,
            output_root=output_root,
            candidate=dict(candidate),
            seed=seed,
            expected_gmner_f1=float(
                protocol["final_gate"]["expected_dev_gmner_f1"]
            ),
            expected_gmner_tolerance=float(
                protocol["final_gate"]["expected_dev_gmner_tolerance"]
            ),
        )
        for candidate in protocol["candidates"]
    ]
    selection = select_seed42_winner(
        {
            str(run["candidate_id"]): float(run["fmnerg_f1"])
            for run in runs
        },
        protocol,
    )
    winner_id = selection["winner_id"]
    winner = (
        next(
            run for run in runs if run["candidate_id"] == winner_id
        )
        if winner_id is not None
        else None
    )
    return {
        "metadata": {
            "kind": "fmnerg_subtype_f3_p1_screen_summary",
            "format_version": 1,
            "protocol_path": protocol["_protocol_path"],
            "protocol_sha256": protocol["_protocol_sha256"],
            "selection_source": "dev",
            "selection_metric": "fmnerg_f1",
            "screen_seed": seed,
            "formal_gmner_frozen": True,
            "test_accessed": False,
        },
        "runs": runs,
        "selection": selection,
        "winner_id": winner_id,
        "winner_config": (
            winner["candidate_config"] if winner is not None else None
        ),
        "winner_config_sha256": (
            winner["candidate_config_sha256"]
            if winner is not None
            else None
        ),
    }


def load_screen_summary(
    path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("kind") != "fmnerg_subtype_f3_p1_screen_summary":
        raise F3ProtocolError(f"Unexpected screen summary kind: {path}.")
    if metadata.get("protocol_sha256") != protocol["_protocol_sha256"]:
        raise F3ProtocolError(f"Screen protocol fingerprint mismatch: {path}.")
    if metadata.get("test_accessed") is not False:
        raise F3ProtocolError(f"Test access detected in screen summary: {path}.")
    winner_id = payload.get("winner_id")
    if not winner_id:
        raise F3ProtocolError(
            "No Seed42 winner advanced; final confirmation is forbidden."
        )
    if winner_id not in protocol["_candidate_by_id"]:
        raise F3ProtocolError(f"Unknown screen winner: {winner_id}.")
    return payload


def summarize_final(
    *,
    protocol: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    screen_summary_path: Path,
) -> dict[str, Any]:
    screen = load_screen_summary(screen_summary_path, protocol)
    winner_id = str(screen["winner_id"])
    candidate = dict(protocol["_candidate_by_id"][winner_id])
    seeds = [int(seed) for seed in protocol["final_gate"]["seeds"]]
    runs = [
        load_candidate_run(
            repo_root=repo_root,
            output_root=output_root,
            candidate=candidate,
            seed=seed,
            expected_gmner_f1=float(
                protocol["final_gate"]["expected_dev_gmner_f1"]
            ),
            expected_gmner_tolerance=float(
                protocol["final_gate"]["expected_dev_gmner_tolerance"]
            ),
        )
        for seed in seeds
    ]
    gate = evaluate_three_seed_gate(
        winner_id=winner_id,
        fmnerg_by_seed={
            int(run["seed"]): float(run["fmnerg_f1"]) for run in runs
        },
        run_contract_by_seed={
            int(run["seed"]): {
                "gmner_f1": float(run["gmner_f1"]),
                "gmner_identity_exact": run["gmner_identity_exact"],
                "formal_stage1_mutated": run["formal_stage1_mutated"],
                "test_accessed": run["test_accessed"],
            }
            for run in runs
        },
        protocol=protocol,
    )
    return {
        "metadata": {
            "kind": "fmnerg_subtype_f3_p1_final_summary",
            "format_version": 1,
            "protocol_path": protocol["_protocol_path"],
            "protocol_sha256": protocol["_protocol_sha256"],
            "screen_summary": str(screen_summary_path),
            "screen_summary_sha256": sha256_file(screen_summary_path),
            "selection_source": "dev",
            "selection_metric": "fmnerg_f1",
            "formal_gmner_frozen": True,
            "test_accessed": False,
        },
        "winner_id": winner_id,
        "winner_config": str(
            resolve_from_repo(candidate["config"], repo_root)
        ),
        "runs": runs,
        "gate": gate,
        "passed": gate["passed"],
        "decision": gate["decision"],
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = resolve_from_repo(args.protocol, repo_root)
    output_root = resolve_from_repo(args.root, repo_root)
    output_path = resolve_from_repo(args.output, repo_root)
    protocol = load_f3_p1_protocol(protocol_path)

    if args.stage == "screen":
        if args.screen_summary is not None:
            raise ValueError("--screen-summary is only valid for --stage final.")
        result = summarize_screen(
            protocol=protocol,
            repo_root=repo_root,
            output_root=output_root,
        )
    else:
        if args.screen_summary is None:
            raise ValueError("--screen-summary is required for --stage final.")
        result = summarize_final(
            protocol=protocol,
            repo_root=repo_root,
            output_root=output_root,
            screen_summary_path=resolve_from_repo(
                args.screen_summary,
                repo_root,
            ),
        )

    save_json_atomic(result, output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
