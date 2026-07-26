"""Run the sealed, one-time, three-seed FMNERG subtype Test evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.data import read_fine_conll
from sidecars.fmnerg_subtype.encoder_config import (
    load_subtype_encoder_config,
)
from sidecars.fmnerg_subtype.encoder_evaluator import (
    evaluate_online_formal_predictions,
    evaluate_online_gold_spans,
)
from sidecars.fmnerg_subtype.encoder_model import (
    build_trainable_subtype_encoder,
    load_trainable_checkpoint_state,
)
from sidecars.fmnerg_subtype.encoder_runtime import (
    validate_online_gold_hierarchy,
)
from sidecars.fmnerg_subtype.evaluator import (
    load_formal_predictions,
    save_json_atomic,
)
from sidecars.fmnerg_subtype.final_test import (
    FINAL_TEST_SEEDS,
    aggregate_seed_metrics,
    load_final_test_protocol,
    validate_artifact,
    validate_dev_acceptance,
)
from sidecars.fmnerg_subtype.formal_chain import (
    export_evidence_visibility_predictions,
    save_formal_predictions,
)
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.online_data import (
    OnlineSubtypeCollator,
    OnlineSubtypeRecordDataset,
    formal_online_records,
    gold_online_records,
)
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


AGGREGATE_METRICS = (
    "fine_mner_f1",
    "fmnerg_f1",
    "subtype_accuracy_on_gold_spans",
    "subtype_macro_f1_on_gold_spans",
    "parent_conditioned_subtype_accuracy",
    "coarse_mner_f1",
    "eeg_f1",
    "gmner_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=(
            "sidecars/fmnerg_subtype/"
            "roberta128_encoder_final_test.yaml"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate only non-Test artifacts without sealing Test access.",
    )
    parser.add_argument(
        "--resume-sealed",
        action="store_true",
        help="Resume the same immutable run after an infrastructure failure.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def verify_checkpoint(
    *,
    specification: dict[str, Any],
    checkpoint: dict[str, Any],
    config_sha256: str,
    stage1_sha256: str,
) -> None:
    seed = int(specification["seed"])
    if checkpoint.get("kind") != "fmnerg_trainable_subtype_encoder":
        raise ValueError(f"Seed {seed} checkpoint kind is invalid.")
    if checkpoint.get("test_accessed") is not False:
        raise ValueError(f"Seed {seed} checkpoint already accessed Test.")
    if checkpoint.get("formal_stage1_mutated") is not False:
        raise ValueError(f"Seed {seed} mutated the formal Stage1 model.")
    if checkpoint.get("selection_metric") != "fmnerg_f1":
        raise ValueError(f"Seed {seed} was not selected by Dev FMNERG.")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError(f"Seed {seed} encoder config hash changed.")
    stored_config = dict(checkpoint.get("config") or {})
    if stored_config.get("model", {}).get("encoder_scope") != "all":
        raise ValueError(f"Seed {seed} is not a full-encoder checkpoint.")
    if int(stored_config.get("runtime", {}).get("seed", -1)) != seed:
        raise ValueError(f"Seed {seed} checkpoint seed metadata changed.")
    initialization = dict(checkpoint.get("initialization") or {})
    if initialization.get("stage1_checkpoint_sha256") != stage1_sha256:
        raise ValueError(f"Seed {seed} Stage1 initialization changed.")


def assert_expected_main_chain(
    metrics: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    tolerance = float(expected["tolerance"])
    for key in ("coarse_mner_f1", "eeg_f1", "gmner_f1"):
        actual = float(metrics[key])
        target = float(expected[key])
        if abs(actual - target) > tolerance:
            raise AssertionError(
                f"Frozen Test {key} changed: expected={target}, actual={actual}."
            )


def main() -> None:
    args = parse_args()
    if args.preflight and args.resume_sealed:
        raise ValueError("--preflight and --resume-sealed are mutually exclusive.")
    root = Path(__file__).resolve().parents[1]
    protocol = load_final_test_protocol(args.protocol, root)
    output_dir = resolve_path(protocol["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = output_dir / ".lock"
    try:
        os.mkdir(lock_dir)
    except FileExistsError as exc:
        raise RuntimeError("Final FMNERG Test evaluation is already running.") from exc

    seal_path = output_dir / "test_access_seal.json"
    summary_path = output_dir / "final_test_summary.json"
    try:
        if summary_path.exists():
            raise RuntimeError(
                "Final FMNERG Test is already complete; rerun is forbidden."
            )
        current_commit = git_commit(root)
        non_test_paths = {
            name: validate_artifact(specification, root)
            for name, specification in protocol["artifacts"].items()
        }
        dev_summary = validate_dev_acceptance(
            non_test_paths["dev_summary"],
            protocol,
        )
        encoder_config_sha = str(
            protocol["artifacts"]["encoder_config"]["sha256"]
        )
        stage1_sha = str(
            protocol["artifacts"]["stage1_checkpoint"]["sha256"]
        )
        for specification in protocol["checkpoints"]:
            checkpoint_path = validate_artifact(specification, root)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            verify_checkpoint(
                specification=specification,
                checkpoint=checkpoint,
                config_sha256=encoder_config_sha,
                stage1_sha256=stage1_sha,
            )
        if args.preflight:
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "protocol_sha256": protocol["_protocol_sha256"],
                        "code_commit": current_commit,
                        "dev_winner": dev_summary[
                            "best_scope_by_mean_dev_fmnerg"
                        ],
                        "seeds": list(FINAL_TEST_SEEDS),
                        "test_accessed": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if seal_path.exists():
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            if not args.resume_sealed:
                raise RuntimeError(
                    "Test access was already sealed. Use --resume-sealed only "
                    "to finish the same immutable run after a failure."
                )
            if seal.get("status") != "started":
                raise RuntimeError("Completed final Test cannot be resumed.")
            if seal.get("protocol_sha256") != protocol["_protocol_sha256"]:
                raise ValueError("Final Test protocol changed after sealing.")
            if seal.get("code_commit") != current_commit:
                raise ValueError("Code commit changed after final Test sealing.")
        else:
            if args.resume_sealed:
                raise RuntimeError("No sealed final Test run exists to resume.")
            seal = {
                "kind": "fmnerg_subtype_final_test_access_seal",
                "format_version": 1,
                "status": "started",
                "started_at": utc_now(),
                "protocol": protocol["_protocol_path"],
                "protocol_sha256": protocol["_protocol_sha256"],
                "code_commit": current_commit,
                "method_selected_on": "dev",
                "dev_winner": dev_summary[
                    "best_scope_by_mean_dev_fmnerg"
                ],
                "seeds": list(FINAL_TEST_SEEDS),
                "report": "mean_std",
                "select_best_seed_on_test": False,
                "architecture_and_hyperparameters_frozen": True,
                "test_accessed": True,
                "test_access_count": 1,
            }
            save_json_atomic(seal, seal_path)

        test_paths = {
            name: validate_artifact(specification, root)
            for name, specification in protocol["test_data"].items()
        }
        taxonomy_path = validate_artifact(
            protocol["artifacts"]["taxonomy"],
            root,
        )
        taxonomy = SubtypeTaxonomy.from_file(taxonomy_path)
        formal_path = output_dir / "test_formal_predictions.json"
        if formal_path.exists():
            formal_payload = load_formal_predictions(
                formal_path,
                taxonomy=taxonomy,
                expected_split="test",
            )
            if formal_payload["metadata"].get("split") != "test":
                raise ValueError("Stored final formal predictions are not Test.")
            if formal_payload["metadata"].get("test_accessed") is not True:
                raise ValueError("Stored final Test predictions lack access mark.")
        else:
            formal_payload = export_evidence_visibility_predictions(
                root=root,
                taxonomy=taxonomy,
                source_file=test_paths["source"],
                evidence_config_path=protocol["artifacts"]["evidence_config"][
                    "path"
                ],
                evidence_checkpoint_path=protocol["artifacts"][
                    "evidence_checkpoint"
                ]["path"],
                formal_cache_path=test_paths["formal_cache"],
                expanded_cache_path=test_paths["expanded_cache"],
                device=torch.device(
                    args.device
                    if args.device.startswith("cuda")
                    and torch.cuda.is_available()
                    else "cpu"
                ),
                split="test",
            )
            assert_expected_main_chain(
                formal_payload["metadata"]["coarse_metrics"],
                protocol["expected_test_main_chain"],
            )
            save_formal_predictions(formal_payload, formal_path)

        assert_expected_main_chain(
            formal_payload["metadata"]["coarse_metrics"],
            protocol["expected_test_main_chain"],
        )
        fine_records = read_fine_conll(
            test_paths["source"],
            taxonomy,
            require_all_subtypes=True,
        )
        gold_dataset = OnlineSubtypeRecordDataset(
            gold_online_records(fine_records, taxonomy)
        )
        formal_dataset = OnlineSubtypeRecordDataset(
            formal_online_records(formal_payload, fine_records, taxonomy)
        )
        validate_online_gold_hierarchy(gold_dataset, taxonomy)
        expected_predictions = sum(
            len(record.get("predictions") or [])
            for record in formal_payload["records"]
        )
        if len(formal_dataset.examples) != expected_predictions:
            raise ValueError("Final Test formal prediction coverage is incomplete.")

        encoder_config_path = validate_artifact(
            protocol["artifacts"]["encoder_config"],
            root,
        )
        config = load_subtype_encoder_config(encoder_config_path)
        device = torch.device(
            args.device
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        rows: list[dict[str, Any]] = []
        for specification in protocol["checkpoints"]:
            seed = int(specification["seed"])
            checkpoint_path = validate_artifact(specification, root)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            verify_checkpoint(
                specification=specification,
                checkpoint=checkpoint,
                config_sha256=str(
                    protocol["artifacts"]["encoder_config"]["sha256"]
                ),
                stage1_sha256=str(
                    protocol["artifacts"]["stage1_checkpoint"]["sha256"]
                ),
            )
            model, tokenizer, initialization, trainability = (
                build_trainable_subtype_encoder(
                    config=config,
                    taxonomy=taxonomy,
                    root=root,
                    device=device,
                )
            )
            if dict(checkpoint.get("trainability") or {}) != trainability:
                raise ValueError(f"Seed {seed} trainability contract changed.")
            load_trainable_checkpoint_state(model, checkpoint["model"])
            model.to(device).eval()
            collator = OnlineSubtypeCollator(
                tokenizer,
                max_length=int(initialization["max_length"]),
            )
            gold_metrics = evaluate_online_gold_spans(
                model,
                gold_dataset,
                collator=collator,
                taxonomy=taxonomy,
                batch_size=config.optim.eval_batch_size,
                device=device,
                include_detailed=True,
            )
            formal = evaluate_online_formal_predictions(
                model,
                formal_dataset,
                formal_payload,
                collator=collator,
                taxonomy=taxonomy,
                batch_size=config.optim.eval_batch_size,
                device=device,
            )
            if formal["metadata"]["gmner_identity_exact"] is not True:
                raise AssertionError(f"Seed {seed} changed formal Test GMNER.")
            metrics = {**gold_metrics, **formal["metrics"]}
            assert_expected_main_chain(
                metrics,
                protocol["expected_test_main_chain"],
            )
            result = {
                "metadata": {
                    **formal["metadata"],
                    "kind": "fmnerg_subtype_encoder_final_test_seed",
                    "format_version": 1,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "encoder_scope": "all",
                    "selection_source": "dev",
                    "select_best_seed_on_test": False,
                    "formal_stage1_mutated": False,
                    "test_accessed": True,
                    "test_access_count": 1,
                },
                "metrics": metrics,
            }
            seed_path = output_dir / f"seed{seed}_test_metrics.json"
            save_json_atomic(result, seed_path)
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "metrics": {
                        name: float(metrics[name])
                        for name in AGGREGATE_METRICS
                    },
                }
            )
            del model, checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        metric_rows = [row["metrics"] for row in rows]
        aggregate = aggregate_seed_metrics(metric_rows, AGGREGATE_METRICS)
        fixed_main_chain = formal_payload["metadata"]["coarse_metrics"]
        summary = {
            "metadata": {
                "kind": "fmnerg_subtype_encoder_final_test_summary",
                "format_version": 1,
                "protocol": protocol["_protocol_path"],
                "protocol_sha256": protocol["_protocol_sha256"],
                "code_commit": current_commit,
                "method": protocol["method"],
                "selection_source": "dev",
                "seeds": list(FINAL_TEST_SEEDS),
                "report": "mean_std",
                "select_best_seed_on_test": False,
                "formal_stage1_mutated": False,
                "gmner_identity_exact": True,
                "test_accessed": True,
                "test_access_count": 1,
                "formal_predictions": str(formal_path),
                "formal_predictions_sha256": sha256_file(formal_path),
            },
            "fixed_main_chain_metrics": fixed_main_chain,
            "per_seed": rows,
            "aggregate": aggregate,
        }
        save_json_atomic(summary, summary_path)
        seal.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "formal_predictions_sha256": sha256_file(formal_path),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
            }
        )
        save_json_atomic(seal, seal_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
