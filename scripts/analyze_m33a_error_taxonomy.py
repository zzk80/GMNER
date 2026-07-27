"""Run the frozen M3.3A Dev-only error taxonomy diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import (
    PairedRecordCandidateCollator,
    PairedRecordCandidateDataset,
    RecordCandidateDataset,
)
from gmner.engine.evidence_visibility_diagnostics import (
    M33A_ERROR_TAXONOMY_VERSION,
    collect_m33a_error_records,
    summarize_error_taxonomy,
)
from gmner.engine.evidence_visibility_evaluator import (
    evaluate_evidence_visibility,
)
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.fine_grounding_adapter_config import (
    load_fine_grounding_adapter_config,
)
from scripts.train_evidence_visibility import load_frozen_chain
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--formal-cache", required=True)
    parser.add_argument("--expanded-cache", required=True)
    parser.add_argument(
        "--split",
        choices=("dev",),
        default="dev",
        help="This diagnostic is deliberately restricted to Dev.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_dirty(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    temporary.replace(path)


def require_dev_cache(dataset: RecordCandidateDataset, role: str) -> None:
    split = str(dataset.metadata.get("split") or "").lower()
    if split != "dev":
        raise ValueError(
            f"{role} cache must declare split=dev; found {split or '<missing>'}."
        )


def main() -> None:
    args = parse_args()
    if args.split != "dev":
        raise ValueError("M3.3A error taxonomy is Dev-only.")
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    evidence_checkpoint_path = resolve(args.checkpoint, root)
    formal_cache_path = resolve(args.formal_cache, root)
    expanded_cache_path = resolve(args.expanded_cache, root)
    output_dir = resolve(args.output_dir, root)

    config = load_evidence_visibility_config(config_path)
    if args.device:
        config.runtime.device = args.device
    formal = RecordCandidateDataset(formal_cache_path)
    expanded = RecordCandidateDataset(expanded_cache_path)
    require_dev_cache(formal, "Formal")
    require_dev_cache(expanded, "Expanded")
    paired = PairedRecordCandidateDataset(formal, expanded)
    if paired.formal_budget != 16 or paired.expanded_budget != 36:
        raise ValueError(
            "The registered M3.3A protocol requires R16/R36 budgets; "
            f"found R{paired.formal_budget}/R{paired.expanded_budget}."
        )

    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
    ) = load_frozen_chain(config, root, device)
    validate_fingerprints(
        paired,
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    checkpoint = torch.load(evidence_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    loader = DataLoader(
        paired,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=PairedRecordCandidateCollator(),
    )
    registered_decode_options = decode_options(hierarchy_config)
    formal_metrics = evaluate_evidence_visibility(
        model,
        fine_model,
        hierarchy,
        loader,
        device,
        decode_options=registered_decode_options,
        loss_options=vars(config.loss).copy(),
    )
    records = collect_m33a_error_records(
        model,
        fine_model,
        hierarchy,
        loader,
        device,
        decode_options=registered_decode_options,
        formal_budget=paired.formal_budget,
        expanded_budget=paired.expanded_budget,
    )
    if len(records) != len(paired):
        raise RuntimeError(
            f"Collected {len(records)} records for a {len(paired)} record cache."
        )
    summary = summarize_error_taxonomy(
        records,
        formal_metrics=formal_metrics,
        tolerance=5e-6,
    )
    summary["formal_evaluator_metrics"] = {
        "mner_f1": float(formal_metrics["entity_f1"]),
        "eeg_f1": float(formal_metrics["eeg_f1"]),
        "gmner_f1": float(formal_metrics["gmner_score"]),
    }

    fine_config_path = resolve(config.frozen.fine_config, root)
    fine_config = load_fine_grounding_adapter_config(fine_config_path)
    hierarchy_checkpoint_path = resolve(
        fine_config.frozen.hierarchical_checkpoint, root
    )
    coarse_checkpoint_path = resolve(
        fine_config.frozen.coarse_checkpoint, root
    )
    fine_checkpoint_path = resolve(config.frozen.fine_checkpoint, root)
    diagnostic_module_path = (
        root / "gmner" / "engine" / "evidence_visibility_diagnostics.py"
    )
    diagnostic_script_path = Path(__file__).resolve()
    diagnostic_test_path = root / "tests" / "test_m33a_error_taxonomy.py"
    protocol = {
        "git_commit": git_commit(root),
        "git_dirty": git_is_dirty(root),
        "date": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "test_accessed": False,
        "diagnostic_module_path": str(diagnostic_module_path),
        "diagnostic_module_sha256": sha256_file(
            diagnostic_module_path
        ),
        "diagnostic_script_path": str(diagnostic_script_path),
        "diagnostic_script_sha256": sha256_file(
            diagnostic_script_path
        ),
        "diagnostic_test_path": str(diagnostic_test_path),
        "diagnostic_test_sha256": sha256_file(diagnostic_test_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "evidence_checkpoint_path": str(
            evidence_checkpoint_path.resolve()
        ),
        "evidence_checkpoint_sha256": sha256_file(
            evidence_checkpoint_path
        ),
        "hierarchy_checkpoint_path": str(
            hierarchy_checkpoint_path.resolve()
        ),
        "hierarchy_checkpoint_sha256": sha256_file(
            hierarchy_checkpoint_path
        ),
        "coarse_checkpoint_path": str(coarse_checkpoint_path.resolve()),
        "coarse_checkpoint_sha256": sha256_file(
            coarse_checkpoint_path
        ),
        "fine_checkpoint_path": str(fine_checkpoint_path.resolve()),
        "fine_checkpoint_sha256": sha256_file(fine_checkpoint_path),
        "stage1_checkpoint_sha256": str(
            formal.metadata.get("stage1_checkpoint_sha256") or ""
        ),
        "formal_cache_path": str(formal_cache_path.resolve()),
        "formal_cache_sha256": sha256_file(formal_cache_path),
        "expanded_cache_path": str(expanded_cache_path.resolve()),
        "expanded_cache_sha256": sha256_file(expanded_cache_path),
        "formal_budget": int(paired.formal_budget),
        "expanded_budget": int(paired.expanded_budget),
        "decode_options": registered_decode_options,
        "visibility_thresholds": {
            "visible_from_null_threshold": float(
                registered_decode_options["visible_from_null_threshold"]
            ),
            "null_from_visible_threshold": float(
                registered_decode_options["null_from_visible_threshold"]
            ),
        },
        "taxonomy_version": M33A_ERROR_TAXONOMY_VERSION,
        "bootstrap_used": False,
    }

    if not all(
        (
            summary["verification"]["formal_metrics_reproduced"],
            summary["verification"]["gold_accounting_passed"],
            summary["verification"]["prediction_accounting_passed"],
            summary["verification"]["test_accessed"] is False,
            protocol["test_accessed"] is False,
        )
    ):
        raise RuntimeError("M3.3A diagnostic Gate failed before output.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "records": len(records),
                "output_dir": str(output_dir.resolve()),
                "overall_metrics": summary["overall_metrics"],
                "gold_failure_distribution": summary[
                    "gold_failure_distribution"
                ],
                "prediction_error_distribution": summary[
                    "prediction_error_distribution"
                ],
                "assignment_analysis": summary["assignment_analysis"],
                "verification": summary["verification"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
