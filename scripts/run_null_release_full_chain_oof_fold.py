"""Run one auditable full-chain OOF fold for the NULL Release verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    FULL_CHAIN_PIPELINE_KIND,
    FULL_CHAIN_PIPELINE_VERSION,
    atomic_write_json,
    fold_from_manifest,
    source_tree_sha256,
    validate_fold_manifest,
    validate_pipeline_manifest,
)
from gmner.data.null_release_oof_cache import (
    sha256_file,
    validate_fold_oof_payload,
)
from gmner.data.p4_r0b_regeneration_contract import (
    P4_R0B_ARTIFACT_IDENTITY,
    P4_R0B_EXECUTION_FOLDS,
    P4_R0B_M33A_REQUIRED_STAGES,
    P4_R0B_M33A_SUPERVISED_STAGES,
    file_bundle_sha256,
    regeneration_metadata,
    validate_r0b_preregistration,
    validate_regeneration_metadata,
    validate_m33a_formal_oof_payload,
)
from gmner.utils.io import read_jsonl


SUPERVISED_STAGES = {
    "stage1",
    "hierarchical",
    "coarse",
    "fine",
    "evidence",
    "reliability",
}
STAGE_ORDER = (
    "stage1",
    "candidate_caches",
    "hierarchical",
    "coarse",
    "fine",
    "evidence",
    "siglip2_caches",
    "reliability",
    "formal_materialize",
    "materialize",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-config",
        default="configs/fmnerg_twitter10000_stage1.yaml",
    )
    parser.add_argument(
        "--hierarchical-config",
        default="configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml",
    )
    parser.add_argument(
        "--coarse-config",
        default="configs/fmnerg_twitter10000_coarse_selector.yaml",
    )
    parser.add_argument(
        "--fine-config",
        default="configs/fmnerg_twitter10000_fine_grounding_adapter.yaml",
    )
    parser.add_argument(
        "--evidence-config",
        default="configs/fmnerg_twitter10000_evidence_visibility.yaml",
    )
    parser.add_argument(
        "--reliability-config",
        default="configs/fmnerg_twitter10000_siglip2_reliability_fusion.yaml",
    )
    parser.add_argument(
        "--release-config",
        default="configs/fmnerg_twitter10000_null_release_verifier.yaml",
    )
    parser.add_argument(
        "--fold-summary",
        default="knowledge/null_release_oof/roberta128/folds/fold_summary.json",
    )
    parser.add_argument(
        "--work-root",
        default="knowledge/null_release_oof/roberta128",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/null_release_oof/roberta128",
    )
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--allow-nonzero-fold", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--siglip2-batch-size", type=int, default=32)
    parser.add_argument("--siglip2-shard-size", type=int, default=128)
    parser.add_argument(
        "--siglip2-model",
        default="/home/zzk/gmner/siglip2-base-patch16-224",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rebuild-fold-manifest", action="store_true")
    parser.add_argument(
        "--adopt-source-revision",
        action="store_true",
        help=(
            "One-time recovery for an unsealed fold after a verified code fix. "
            "Preserves the validated Stage1 artifact and invalidates candidate "
            "caches plus every downstream stage."
        ),
    )
    parser.add_argument(
        "--source-revision-reason",
        default=None,
        help="Required audit note for --adopt-source-revision.",
    )
    parser.add_argument(
        "--source-revision-invalidate-from",
        choices=STAGE_ORDER[1:],
        default="candidate_caches",
        help=(
            "First stage invalidated by --adopt-source-revision. Every earlier "
            "stage must have complete, unchanged provenance."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default=None)
    parser.add_argument(
        "--regeneration-authorization",
        default=None,
        help=(
            "Enable the separately authorized P4-R0-B regeneration contract. "
            "This mode is limited to folds 0-7 and independent output roots."
        ),
    )
    parser.add_argument(
        "--recover-completed-stage1-sigsegv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Accept a Stage1 SIGSEGV only when the checkpoint and complete "
            "train_summary.json already exist and pass validation."
        ),
    )
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def descriptor(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Expected pipeline artifact does not exist: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def command_sha256(commands: list[list[str]]) -> str:
    value = json.dumps(commands, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_stage_artifacts(stage: dict, groups: tuple[str, ...]) -> None:
    for group in groups:
        artifacts = list(stage.get(group) or [])
        if not artifacts:
            raise ValueError(f"Preserved Stage1 has no {group} provenance.")
        for artifact in artifacts:
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
                raise ValueError(
                    f"Preserved Stage1 {group} artifact is missing or changed: {path}"
                )


def adopt_source_revision(
    *,
    manifest_path: Path,
    pipeline_path: Path,
    new_source_sha256: str,
    reason: str,
    invalidate_from: str = "candidate_caches",
) -> None:
    """Rebase an unsealed failed fold while preserving verified prior stages."""

    if not reason.strip():
        raise ValueError("--source-revision-reason must be non-empty.")
    if not manifest_path.is_file() or not pipeline_path.is_file():
        raise FileNotFoundError(
            "Source revision adoption requires existing fold and pipeline manifests."
        )
    manifest_file_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    old_source = str(manifest.get("source_tree_sha256") or "")
    if old_source == new_source_sha256:
        print("[source-revision] already current; no migration needed", flush=True)
        return
    if not old_source or pipeline.get("source_tree_sha256") != old_source:
        raise ValueError("Fold and pipeline source revisions are inconsistent.")
    if pipeline.get("fold_manifest_sha256") != manifest_file_sha256:
        raise ValueError("Pipeline does not reference the current fold manifest.")
    if pipeline.get("sealed") or pipeline.get("test_accessed") is not False:
        raise ValueError("Only an unsealed, test-free pipeline can be migrated.")
    if invalidate_from not in STAGE_ORDER[1:]:
        raise ValueError(f"Invalid source-revision boundary: {invalidate_from}")
    boundary = STAGE_ORDER.index(invalidate_from)
    preserved_names = list(STAGE_ORDER[:boundary])
    existing_stages = dict(pipeline.get("stages") or {})
    preserved_stages: dict[str, dict] = {}
    for name in preserved_names:
        stage = dict(existing_stages.get(name) or {})
        if stage.get("status") != "complete" or stage.get("test_accessed") is not False:
            raise ValueError(
                f"A complete, test-free {name} stage is required for migration."
            )
        _verify_stage_artifacts(stage, ("inputs", "outputs"))
        if name in SUPERVISED_STAGES:
            for role in ("config", "checkpoint"):
                artifact = dict(stage.get(role) or {})
                path = Path(str(artifact.get("path", "")))
                if (
                    not path.is_file()
                    or sha256_file(path) != artifact.get("sha256")
                ):
                    raise ValueError(
                        f"Preserved {name} {role} artifact changed: {path}"
                    )
        preserved_stages[name] = stage

    revision = {
        "adopted_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_source_tree_sha256": old_source,
        "source_tree_sha256": new_source_sha256,
        "reason": reason.strip(),
        "preserved_stages": preserved_names,
        "invalidated_from": invalidate_from,
        "test_accessed": False,
    }
    manifest.setdefault("source_revision_history", []).append(revision)
    manifest["source_tree_sha256"] = new_source_sha256
    atomic_write_json(manifest_path, manifest)

    pipeline.setdefault("source_revision_history", []).append(revision)
    pipeline["source_tree_sha256"] = new_source_sha256
    pipeline["fold_manifest_sha256"] = sha256_file(manifest_path)
    pipeline["stages"] = preserved_stages
    pipeline["sealed"] = False
    pipeline["test_accessed"] = False
    atomic_write_json(pipeline_path, pipeline)
    print(
        f"[source-revision] preserved {', '.join(preserved_names)}; "
        f"invalidated {invalidate_from} and all downstream stages",
        flush=True,
    )


def _record_ids(payload: dict) -> list[str]:
    return [
        str((record.get("metadata") or {}).get("record_id", ""))
        for record in payload.get("records") or []
    ]


def _cache_metadata(path: Path) -> tuple[dict, list[str]]:
    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload.get("metadata") or {})
    ids = _record_ids(payload)
    del payload
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Candidate cache has missing or duplicate record ids: {path}")
    return metadata, ids


def validate_candidate_caches(
    paths: dict[str, Path],
    *,
    fold: dict,
    train_record_ids: list[str] | None = None,
    selection_record_ids: list[str] | None = None,
    stage1_checkpoint: Path,
    regeneration: dict | None = None,
) -> None:
    expected_stage1 = sha256_file(stage1_checkpoint)
    loaded = {name: _cache_metadata(path) for name, path in paths.items()}
    for name, (metadata, _) in loaded.items():
        if metadata.get("stage1_checkpoint_sha256") != expected_stage1:
            raise ValueError(f"{name} cache uses another Stage1 checkpoint.")
        if regeneration:
            validate_regeneration_metadata(
                metadata,
                authorization_sha256=str(
                    regeneration["regeneration_authorization_sha256"]
                ),
                fold_id=int(regeneration["regeneration_fold_id"]),
                experiment_id=str(
                    regeneration["regeneration_experiment_id"]
                ),
            )
        candidate = dict(metadata.get("candidate_config") or {})
        if bool(candidate.get("inject_gold_types")):
            raise ValueError(f"{name} cache illegally injects gold types.")
        expected_regions = 36 if name.endswith("r36") else 16
        if int(candidate.get("max_regions", -1)) != expected_regions:
            raise ValueError(
                f"{name} cache has max_regions={candidate.get('max_regions')}, "
                f"expected {expected_regions}."
            )
    for prefix, expected_ids in (
        ("train", list(train_record_ids or fold["train_record_ids"])),
        ("dev", list(selection_record_ids or [])),
        ("heldout", list(fold["heldout_record_ids"])),
    ):
        if loaded[f"{prefix}_r16"][1] != expected_ids:
            raise ValueError(f"{prefix} R16 cache record order differs from the manifest.")
        if loaded[f"{prefix}_r36"][1] != expected_ids:
            raise ValueError(f"{prefix} R36 cache record order differs from the manifest.")
    for name in ("heldout_r16", "heldout_r36"):
        metadata = loaded[name][0]
        if not bool(metadata.get("oof_heldout")):
            raise ValueError(f"{name} cache is not marked oof_heldout.")
        if int(metadata.get("oof_fold_id", -1)) != int(fold["fold"]):
            raise ValueError(f"{name} cache has the wrong OOF fold id.")
    for name in ("train_r16", "train_r36", "dev_r16", "dev_r36"):
        if bool(loaded[name][0].get("oof_heldout")):
            raise ValueError(f"{name} cache must not be marked held-out.")
    for prefix in ("train", "dev", "heldout"):
        anchor = dict(
            loaded[f"{prefix}_r36"][0].get("formal_anchor_cache") or {}
        )
        expected_anchor = sha256_file(paths[f"{prefix}_r16"])
        if anchor.get("sha256") != expected_anchor:
            raise ValueError(
                f"{prefix} R36 cache is not anchored to its formal R16 cache."
            )


def validate_siglip2_caches(
    paths: dict[str, Path], candidate_paths: dict[str, Path]
) -> None:
    pairs = {
        "train": ("train_r16", "train_r36"),
        "dev": ("dev_r16", "dev_r36"),
        "heldout": ("heldout_r16", "heldout_r36"),
    }
    for split, manifest_path in paths.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        formal_name, expanded_name = pairs[split]
        if manifest.get("formal_cache_sha256") != sha256_file(
            candidate_paths[formal_name]
        ):
            raise ValueError(f"SigLIP2 {split} formal cache fingerprint mismatch.")
        if manifest.get("expanded_cache_sha256") != sha256_file(
            candidate_paths[expanded_name]
        ):
            raise ValueError(f"SigLIP2 {split} expanded cache fingerprint mismatch.")
        ids = [str(item.get("record_id", "")) for item in manifest.get("records") or []]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"SigLIP2 {split} manifest has invalid record ids.")
        _, candidate_ids = _cache_metadata(candidate_paths[expanded_name])
        if ids != candidate_ids:
            raise ValueError(f"SigLIP2 {split} record order differs from candidates.")


def _no_test_contract(config: dict, *, allow_disabled_stage1_key: bool = False) -> None:
    def visit(value, trail: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                next_trail = (*trail, str(key))
                key_tokens = str(key).lower().split("_")
                if "test" in key_tokens:
                    allowed = (
                        allow_disabled_stage1_key
                        and next_trail == ("data", "test_file")
                        and item == "__OOF_TEST_DISABLED__"
                    )
                    allowed = allowed or (
                        next_trail == ("runtime", "evaluate_test_after_training")
                        and item is False
                    )
                    if not allowed:
                        raise ValueError(
                            "Full-chain OOF config contains a test field: "
                            + ".".join(next_trail)
                        )
                visit(item, next_trail)
        elif isinstance(value, list):
            for item in value:
                visit(item, trail)

    visit(config)


class FoldPipeline:
    def __init__(
        self,
        *,
        root: Path,
        manifest_path: Path,
        manifest: dict,
        fold: dict,
        pipeline_path: Path,
        resume: bool,
        dry_run: bool,
        regeneration: dict | None = None,
        recover_completed_stage1_sigsegv: bool = False,
        required_pipeline_stages: tuple[str, ...] | None = None,
        supervised_pipeline_stages: tuple[str, ...] | None = None,
    ) -> None:
        self.root = root
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.fold = fold
        self.pipeline_path = pipeline_path
        self.resume = resume
        self.dry_run = dry_run
        self.regeneration = dict(regeneration or {})
        self.recover_completed_stage1_sigsegv = bool(
            recover_completed_stage1_sigsegv
        )
        self.required_pipeline_stages = required_pipeline_stages
        self.supervised_pipeline_stages = supervised_pipeline_stages
        if pipeline_path.exists():
            self.payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
            if int(self.payload.get("fold_id", -1)) != int(fold["fold"]):
                raise ValueError("Existing pipeline manifest belongs to another fold.")
            if self.payload.get("fold_manifest_sha256") != sha256_file(manifest_path):
                raise ValueError("Existing pipeline uses another fold manifest.")
            if dict(self.payload.get("regeneration") or {}) != self.regeneration:
                raise ValueError(
                    "Existing pipeline uses another regeneration identity."
                )
        else:
            self.payload = {
                "format_version": FULL_CHAIN_PIPELINE_VERSION,
                "kind": FULL_CHAIN_PIPELINE_KIND,
                "fold_id": int(fold["fold"]),
                "num_folds": int(manifest["num_folds"]),
                "fold_manifest": str(manifest_path),
                "fold_manifest_sha256": sha256_file(manifest_path),
                "source_tree_sha256": manifest["source_tree_sha256"],
                "train_record_ids_sha256": fold["train_record_ids_sha256"],
                "heldout_record_ids_sha256": fold["heldout_record_ids_sha256"],
                "test_accessed": False,
                "sealed": False,
                "stages": {},
            }
            if self.regeneration:
                self.payload["regeneration"] = self.regeneration
            if not dry_run:
                self.save()
        current_source = source_tree_sha256(root)
        if current_source != manifest["source_tree_sha256"]:
            raise ValueError(
                "Source tree changed after the fold manifest was created. "
                "Rebuild the manifest before formal OOF execution."
            )

    def save(self) -> None:
        atomic_write_json(self.pipeline_path, self.payload)

    @staticmethod
    def _outputs_valid(stage: dict, outputs: list[Path]) -> bool:
        expected = {
            str(Path(item["path"]).resolve()): item["sha256"]
            for item in stage.get("outputs") or []
        }
        if set(expected) != {str(path.resolve()) for path in outputs}:
            return False
        return all(
            path.is_file() and sha256_file(path) == expected[str(path.resolve())]
            for path in outputs
        )

    def run(
        self,
        name: str,
        commands: list[list[str]],
        outputs: list[Path],
        *,
        config_path: Path | None = None,
        checkpoint_path: Path | None = None,
        inputs: list[Path] | None = None,
        validator: Callable[[], None] | None = None,
    ) -> None:
        digest = command_sha256(commands)
        existing = dict((self.payload.get("stages") or {}).get(name) or {})
        current_inputs = inputs or []
        descriptors_valid = self._outputs_valid(existing, outputs) and self._outputs_valid(
            {"outputs": existing.get("inputs") or []}, current_inputs
        )
        if config_path is not None:
            config_artifact = dict(existing.get("config") or {})
            descriptors_valid = descriptors_valid and (
                config_path.is_file()
                and config_artifact.get("path") == str(config_path.resolve())
                and config_artifact.get("sha256") == sha256_file(config_path)
            )
        if (
            self.resume
            and existing.get("status") == "complete"
            and existing.get("command_sha256") == digest
            and descriptors_valid
        ):
            if validator is not None:
                validator()
            print(f"[resume] {name}: verified and skipped", flush=True)
            return
        if bool(self.payload.get("sealed")):
            raise ValueError(
                f"Cannot rebuild invalid stage {name!r} after the pipeline is sealed."
            )
        print(f"[{name}]", flush=True)
        for command in commands:
            print("+ " + subprocess.list2cmdline(command), flush=True)
        if self.dry_run:
            return
        stages = self.payload.setdefault("stages", {})
        stages[name] = {
            "status": "running",
            "command_sha256": digest,
            "test_accessed": False,
        }
        self.save()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.root)
        environment["GMNER_FULL_CHAIN_OOF"] = "1"
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            recovered_sigsegv = False
            for command in commands:
                command_started_ns = time.time_ns()
                try:
                    subprocess.run(
                        command,
                        cwd=self.root,
                        env=environment,
                        check=True,
                    )
                except subprocess.CalledProcessError as error:
                    recoverable = (
                        name == "stage1"
                        and self.recover_completed_stage1_sigsegv
                        and int(error.returncode) in {-11, 139}
                        and all(path.is_file() for path in outputs)
                        and all(
                            path.stat().st_mtime_ns >= command_started_ns
                            for path in outputs
                        )
                    )
                    if not recoverable:
                        raise
                    recovered_sigsegv = True
                    print(
                        "[stage1] accepted post-completion SIGSEGV after "
                        "validating all required outputs",
                        flush=True,
                    )
            missing = [path for path in outputs if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Stage {name} did not create: {missing}")
            if validator is not None:
                validator()
        except Exception:
            stages[name]["status"] = "failed"
            self.save()
            raise
        stage = {
            "status": "complete",
            "command_sha256": digest,
            "test_accessed": False,
            "outputs": [descriptor(path) for path in outputs],
            "inputs": [descriptor(path) for path in current_inputs],
        }
        if name in SUPERVISED_STAGES:
            if config_path is None or checkpoint_path is None:
                raise ValueError(f"Supervised stage {name} lacks provenance artifacts.")
            stage.update(
                {
                    "heldout_excluded": True,
                    "train_record_ids_sha256": self.fold[
                        "train_record_ids_sha256"
                    ],
                    "config": descriptor(config_path),
                    "checkpoint": descriptor(checkpoint_path),
                }
            )
        if recovered_sigsegv:
            stage["post_completion_sigsegv_recovered"] = True
        stages[name] = stage
        self.save()

    def seal(self) -> None:
        if self.dry_run:
            return
        self.payload["sealed"] = True
        self.payload["test_accessed"] = False
        self.save()
        validate_pipeline_manifest(
            self.pipeline_path,
            fold_manifest=self.manifest,
            fold_id=int(self.fold["fold"]),
            required_stages=self.required_pipeline_stages,
            supervised_stages=self.supervised_pipeline_stages,
        )


def _candidate_command(
    *,
    python: str,
    root: Path,
    config: Path,
    checkpoint: Path,
    source: Path,
    output: Path,
    max_regions: int,
    fold_id: int | None,
    split: str,
    batch_size: int,
    device: str,
    formal_anchor_cache: Path | None = None,
    regeneration: dict | None = None,
) -> list[str]:
    command = [
        python,
        "-u",
        str(root / "scripts" / "build_record_candidate_cache.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--split",
        split,
        "--input-file",
        str(source),
        "--output",
        str(output),
        "--k-best",
        "6",
        "--max-span-candidates",
        "12",
        "--top-m-types",
        "3",
        "--boundary-shift",
        "1",
        "--boundary-penalty",
        "0.25",
        "--max-regions",
        str(max_regions),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    if fold_id is not None:
        command.extend(["--oof-fold-id", str(fold_id)])
    if formal_anchor_cache is not None:
        command.extend(
            ["--formal-anchor-cache", str(formal_anchor_cache)]
        )
    if regeneration:
        command.extend(
            [
                "--artifact-identity",
                str(regeneration["artifact_identity"]),
                "--regeneration-authorization-sha256",
                str(regeneration["regeneration_authorization_sha256"]),
                "--regeneration-fold-id",
                str(regeneration["regeneration_fold_id"]),
                "--regeneration-experiment-id",
                str(regeneration["regeneration_experiment_id"]),
            ]
        )
    return command


def _validate_stage1_completion(output_dir: Path, checkpoint: Path) -> None:
    summary_path = output_dir / "train_summary.json"
    if not checkpoint.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Stage1 completion requires best_model.pt and train_summary.json."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("best_metric_name"):
        raise ValueError("Stage1 train summary has no best metric.")
    if summary.get("best_epoch") is None:
        raise ValueError("Stage1 train summary has no best epoch.")
    reported = Path(str(summary.get("best_checkpoint", "")))
    if not reported.is_absolute():
        reported = (Path.cwd() / reported).resolve()
    if reported != checkpoint.resolve():
        raise ValueError("Stage1 train summary names another best checkpoint.")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    regeneration: dict | None = None
    authorization: dict | None = None
    authorization_path: Path | None = None
    if args.regeneration_authorization:
        authorization_path = resolve(args.regeneration_authorization, root)
        authorization = json.loads(
            authorization_path.read_text(encoding="utf-8")
        )
        validate_r0b_preregistration(authorization)
        regeneration = regeneration_metadata(
            authorization_sha256=sha256_file(authorization_path),
            fold_id=args.fold_id,
            experiment_id=str(authorization["experiment_id"]),
        )
        if args.seed != int(authorization["source_contract"]["seed"]):
            raise ValueError("R0-B seed differs from the preregistered seed.")
        if args.rebuild_fold_manifest:
            raise PermissionError(
                "R0-B fold manifests must be prepared by the read-only "
                "preflight; the fold runner cannot rebuild them."
            )
    if args.fold_id not in range(10):
        raise ValueError("Formal full-chain OOF fold id must be in 0..9.")
    if regeneration and args.fold_id not in P4_R0B_EXECUTION_FOLDS:
        raise PermissionError("P4-R0-B execution is limited to folds 0-7.")
    if args.fold_id != 0 and not args.allow_nonzero_fold:
        raise ValueError(
            "Fold 0 must pass end-to-end validation first. Use "
            "--allow-nonzero-fold only after that review."
        )
    python = sys.executable
    stage1_template_path = resolve(args.stage1_config, root)
    manifest_path = resolve(args.fold_summary, root)
    work_root = resolve(args.work_root, root)
    output_root = resolve(args.output_root, root)
    if authorization:
        storage = dict(authorization["storage_contract"])
        expected_work_root = resolve(storage["work_root"], root)
        expected_output_root = resolve(storage["output_root"], root)
        legacy_root = resolve(storage["legacy_evidence_root"], root)
        if work_root != expected_work_root or output_root != expected_output_root:
            raise ValueError(
                "R0-B work/output roots differ from the preregistration."
            )
        if work_root == legacy_root or output_root == legacy_root:
            raise ValueError("R0-B cannot write into the legacy evidence root.")
        try:
            manifest_path.relative_to(work_root)
        except ValueError as error:
            raise ValueError(
                "R0-B fold summary must be stored under its independent work root."
            ) from error
    fold_work = work_root / f"fold{args.fold_id}"
    pipeline_path = fold_work / "pipeline_manifest.json"
    if args.rebuild_fold_manifest or not manifest_path.exists():
        command = [
            python,
            str(root / "scripts" / "build_evidence_folds.py"),
            "--config",
            str(stage1_template_path),
            "--output-dir",
            str(manifest_path.parent),
            "--num-folds",
            "10",
            "--seed",
            str(args.seed),
        ]
        if args.rebuild_fold_manifest:
            command.append("--force")
        print("+ " + subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            if not manifest_path.exists():
                return
        else:
            subprocess.run(command, cwd=root, check=True)
    if args.adopt_source_revision:
        if args.dry_run:
            raise ValueError("--adopt-source-revision cannot be used with --dry-run.")
        adopt_source_revision(
            manifest_path=manifest_path,
            pipeline_path=pipeline_path,
            new_source_sha256=source_tree_sha256(root),
            reason=str(args.source_revision_reason or ""),
            invalidate_from=args.source_revision_invalidate_from,
        )
    manifest = validate_fold_manifest(
        manifest_path,
        expected_num_folds=10,
        verify_fold_ids=(
            P4_R0B_EXECUTION_FOLDS if regeneration else None
        ),
    )
    if regeneration:
        manifest_regeneration = dict(manifest.get("regeneration") or {})
        implementation = dict(
            manifest_regeneration.get("implementation_fingerprints") or {}
        )
        implementation_files = [
            Path(str(item["path"]))
            for item in implementation.get("files") or []
        ]
        if (
            manifest_regeneration.get("artifact_identity")
            != P4_R0B_ARTIFACT_IDENTITY
            or manifest_regeneration.get("regeneration_authorization_sha256")
            != regeneration["regeneration_authorization_sha256"]
            or manifest_regeneration.get("regeneration_experiment_id")
            != regeneration["regeneration_experiment_id"]
            or tuple(manifest_regeneration.get("execution_folds") or ())
            != P4_R0B_EXECUTION_FOLDS
            or dict(manifest_regeneration.get("chain_contract") or {})
            != dict(authorization["chain_contract"])
            or not implementation_files
            or file_bundle_sha256(implementation_files).get("sha256")
            != implementation.get("sha256")
        ):
            raise ValueError(
                "R0-B fold summary does not carry the authorized identity."
            )
    fold = fold_from_manifest(manifest, args.fold_id)

    fold_output = output_root / f"fold{args.fold_id}"
    config_dir = fold_work / "configs"
    candidate_dir = fold_work / "candidates"
    siglip2_root = fold_work / "siglip2"
    directories = [config_dir, candidate_dir, fold_output]
    if not regeneration:
        directories.append(siglip2_root)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    stage1_template = load_yaml(stage1_template_path)
    hierarchy_template = load_yaml(resolve(args.hierarchical_config, root))
    coarse_template = load_yaml(resolve(args.coarse_config, root))
    fine_template = load_yaml(resolve(args.fine_config, root))
    evidence_template = load_yaml(resolve(args.evidence_config, root))
    reliability_template = (
        None
        if regeneration
        else load_yaml(resolve(args.reliability_config, root))
    )
    release_template = (
        None
        if regeneration
        else load_yaml(resolve(args.release_config, root))
    )

    d0_path = fold_work / "d0_preflight.json"
    if not d0_path.is_file():
        raise FileNotFoundError(
            "Fold-0 D0 preflight must pass before the chain can start."
        )
    d0 = json.loads(d0_path.read_text(encoding="utf-8"))
    if (
        d0.get("status") != "PASSED"
        or d0.get("fold0_execution_authorized") is not True
        or d0.get("official_dev_accessed") is not False
        or d0.get("test_accessed") is not False
        or d0.get("fold_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise PermissionError("Fold-0 D0 preflight proof is invalid.")
    train_file = Path(d0["fit_file"]).resolve()
    dev_file = Path(d0["selection_file"]).resolve()
    train_record_ids = [
        str(item.get("id", (item.get("metadata") or {}).get("record_id", "")))
        for item in read_jsonl(train_file)
    ]
    selection_record_ids = [
        str(item.get("id", (item.get("metadata") or {}).get("record_id", "")))
        for item in read_jsonl(dev_file)
    ]
    if len(train_record_ids) != 5600 or len(selection_record_ids) != 700:
        raise ValueError("D0 nested checkpoint-selection split changed.")
    heldout_file = Path(fold["heldout_file"]).resolve()
    image_dir = resolve(stage1_template["data"]["image_dir"], root)

    candidate_paths = {
        f"{split}_r{regions}": candidate_dir / f"{split}_r{regions}.pt"
        for split in ("train", "dev", "heldout")
        for regions in (16, 36)
    }
    model_dirs = {
        name: fold_output / name
        for name in (
            "stage1",
            "hierarchical",
            "coarse",
            "fine",
            "evidence",
            "reliability",
        )
    }
    checkpoints = {
        name: model_dirs[name] / (
            "best_ab_model.pt" if name == "reliability" else "best_model.pt"
        )
        for name in model_dirs
    }
    config_names = [
        "stage1",
        "hierarchical",
        "coarse",
        "fine",
        "evidence",
    ]
    config_names.extend(
        ["evidence_heldout"]
        if regeneration
        else [
            "reliability",
            "reliability_heldout",
            "release_materialize",
        ]
    )
    config_paths = {
        name: config_dir / f"{name}.yaml" for name in config_names
    }

    stage1 = deepcopy(stage1_template)
    stage1["data"]["train_file"] = str(train_file)
    stage1["data"]["dev_file"] = str(dev_file)
    stage1["data"]["test_file"] = "__OOF_TEST_DISABLED__"
    stage1["runtime"]["output_dir"] = str(model_dirs["stage1"])
    stage1["runtime"]["seed"] = args.seed
    _no_test_contract(stage1, allow_disabled_stage1_key=True)
    write_yaml(config_paths["stage1"], stage1)

    hierarchy = deepcopy(hierarchy_template)
    hierarchy["data"].update(
        {
            "train_cache": str(candidate_paths["train_r16"]),
            "dev_cache": str(candidate_paths["dev_r16"]),
            "require_oof_train_cache": False,
        }
    )
    hierarchy["data"].pop("test_cache", None)
    hierarchy["runtime"]["output_dir"] = str(model_dirs["hierarchical"])
    hierarchy["runtime"]["seed"] = args.seed
    hierarchy["runtime"]["evaluate_test_after_training"] = False
    _no_test_contract(hierarchy)
    write_yaml(config_paths["hierarchical"], hierarchy)

    coarse = deepcopy(coarse_template)
    coarse["data"].update(
        {
            "train_cache": str(candidate_paths["train_r36"]),
            "dev_cache": str(candidate_paths["dev_r36"]),
        }
    )
    coarse["runtime"]["output_dir"] = str(model_dirs["coarse"])
    coarse["runtime"]["seed"] = args.seed
    _no_test_contract(coarse)
    write_yaml(config_paths["coarse"], coarse)

    fine = deepcopy(fine_template)
    fine["data"].update(
        {
            "formal_train_cache": str(candidate_paths["train_r16"]),
            "expanded_train_cache": str(candidate_paths["train_r36"]),
            "formal_dev_cache": str(candidate_paths["dev_r16"]),
            "expanded_dev_cache": str(candidate_paths["dev_r36"]),
            "require_oof_train_cache": False,
        }
    )
    fine["frozen"].update(
        {
            "hierarchical_config": str(config_paths["hierarchical"]),
            "hierarchical_checkpoint": str(checkpoints["hierarchical"]),
            "coarse_checkpoint": str(checkpoints["coarse"]),
        }
    )
    fine["runtime"]["output_dir"] = str(model_dirs["fine"])
    fine["runtime"]["seed"] = args.seed
    _no_test_contract(fine)
    write_yaml(config_paths["fine"], fine)

    evidence = deepcopy(evidence_template)
    evidence["data"].update(deepcopy(fine["data"]))
    evidence["frozen"].update(
        {
            "fine_config": str(config_paths["fine"]),
            "fine_checkpoint": str(checkpoints["fine"]),
        }
    )
    evidence["runtime"]["output_dir"] = str(model_dirs["evidence"])
    evidence["runtime"]["seed"] = args.seed
    _no_test_contract(evidence)
    write_yaml(config_paths["evidence"], evidence)

    if regeneration:
        evidence_heldout = deepcopy(evidence)
        evidence_heldout["data"].update(
            {
                "formal_train_cache": str(candidate_paths["heldout_r16"]),
                "expanded_train_cache": str(candidate_paths["heldout_r36"]),
                "require_oof_train_cache": True,
            }
        )
        _no_test_contract(evidence_heldout)
        write_yaml(config_paths["evidence_heldout"], evidence_heldout)
    else:
        assert reliability_template is not None
        assert release_template is not None
        siglip2_dirs = {
            split: siglip2_root / split
            for split in ("train", "dev", "heldout")
        }
        siglip2_manifests = {
            split: path / "manifest.json"
            for split, path in siglip2_dirs.items()
        }
        reliability = deepcopy(reliability_template)
        reliability["data"].update(
            {
                **deepcopy(fine["data"]),
                "siglip2_train_cache": str(siglip2_dirs["train"]),
                "siglip2_dev_cache": str(siglip2_dirs["dev"]),
                "verify_siglip2_cache_hashes": True,
            }
        )
        reliability["frozen"].update(
            {
                "fine_config": str(config_paths["fine"]),
                "fine_checkpoint": str(checkpoints["fine"]),
                "evidence_visibility_config": str(config_paths["evidence"]),
                "evidence_visibility_checkpoint": str(checkpoints["evidence"]),
            }
        )
        reliability["runtime"]["output_dir"] = str(
            model_dirs["reliability"]
        )
        reliability["runtime"]["seed"] = args.seed
        _no_test_contract(reliability)
        write_yaml(config_paths["reliability"], reliability)

        reliability_heldout = deepcopy(reliability)
        reliability_heldout["data"].update(
            {
                "formal_train_cache": str(candidate_paths["heldout_r16"]),
                "expanded_train_cache": str(candidate_paths["heldout_r36"]),
                "siglip2_train_cache": str(siglip2_dirs["heldout"]),
                "require_oof_train_cache": True,
            }
        )
        _no_test_contract(reliability_heldout)
        write_yaml(
            config_paths["reliability_heldout"], reliability_heldout
        )

        release = deepcopy(release_template)
        release["frozen"].update(
            {
                "reliability_config": str(
                    config_paths["reliability_heldout"]
                ),
                "reliability_checkpoint": str(checkpoints["reliability"]),
            }
        )
        release["oof"]["train_feature_cache"] = str(
            work_root / "full_chain_train_oof.pt"
        )
        release["runtime"]["output_dir"] = str(
            fold_output / "release_materialize"
        )
        _no_test_contract(release)
        write_yaml(config_paths["release_materialize"], release)

    pipeline = FoldPipeline(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        fold=fold,
        pipeline_path=pipeline_path,
        resume=args.resume,
        dry_run=args.dry_run,
        regeneration=regeneration,
        recover_completed_stage1_sigsegv=(
            args.recover_completed_stage1_sigsegv
        ),
        required_pipeline_stages=(
            P4_R0B_M33A_REQUIRED_STAGES if regeneration else None
        ),
        supervised_pipeline_stages=(
            P4_R0B_M33A_SUPERVISED_STAGES if regeneration else None
        ),
    )
    if regeneration and not args.dry_run:
        pipeline.payload["nested_checkpoint_selection"] = {
            "outer_train_record_ids_sha256": d0["outer_train_ids_sha256"],
            "fit_record_ids_sha256": d0["fit_ids_sha256"],
            "selection_record_ids_sha256": d0["selection_ids_sha256"],
            "fit_records": len(train_record_ids),
            "selection_records": len(selection_record_ids),
            "official_dev_accessed": False,
            "heldout_excluded": True,
        }
        pipeline.save()

    stage1_command = [
        python,
        "-u",
        str(root / "scripts" / "train.py"),
        "--config",
        str(config_paths["stage1"]),
        "--skip-test-evaluation",
    ]
    stage1_outputs = [checkpoints["stage1"]]
    if regeneration:
        stage1_outputs.append(model_dirs["stage1"] / "train_summary.json")
    pipeline.run(
        "stage1",
        [stage1_command],
        stage1_outputs,
        config_path=config_paths["stage1"],
        checkpoint_path=checkpoints["stage1"],
        inputs=[train_file, dev_file],
        validator=(
            lambda: _validate_stage1_completion(
                model_dirs["stage1"], checkpoints["stage1"]
            )
            if regeneration
            else None
        ),
    )
    if args.stop_after == "stage1":
        return

    candidate_commands = []
    for split, source, fold_marker in (
        ("train", train_file, None),
        ("dev", dev_file, None),
        ("heldout", heldout_file, args.fold_id),
    ):
        for regions in (16, 36):
            candidate_commands.append(
                _candidate_command(
                    python=python,
                    root=root,
                    config=config_paths["stage1"],
                    checkpoint=checkpoints["stage1"],
                    source=source,
                    output=candidate_paths[f"{split}_r{regions}"],
                    max_regions=regions,
                    fold_id=fold_marker,
                    split="dev" if split == "dev" else "train",
                    batch_size=args.inference_batch_size,
                    device=args.device,
                    formal_anchor_cache=(
                        candidate_paths[f"{split}_r16"]
                        if regions == 36
                        else None
                    ),
                    regeneration=regeneration,
                )
            )
    pipeline.run(
        "candidate_caches",
        candidate_commands,
        list(candidate_paths.values()),
        inputs=[checkpoints["stage1"], train_file, dev_file, heldout_file],
        validator=lambda: validate_candidate_caches(
            candidate_paths,
            fold=fold,
            train_record_ids=train_record_ids,
            selection_record_ids=selection_record_ids,
            stage1_checkpoint=checkpoints["stage1"],
            regeneration=regeneration,
        ),
    )
    if args.stop_after == "candidate_caches":
        return

    for name, script, dependencies in (
        (
            "hierarchical",
            "train_hierarchical_record_verifier.py",
            [candidate_paths["train_r16"], candidate_paths["dev_r16"]],
        ),
        (
            "coarse",
            "train_coarse_region_selector.py",
            [candidate_paths["train_r36"], candidate_paths["dev_r36"]],
        ),
        (
            "fine",
            "train_fine_grounding_adapter.py",
            [
                checkpoints["hierarchical"],
                checkpoints["coarse"],
                candidate_paths["train_r16"],
                candidate_paths["train_r36"],
            ],
        ),
        (
            "evidence",
            "train_evidence_visibility.py",
            [checkpoints["fine"], candidate_paths["train_r16"]],
        ),
    ):
        pipeline.run(
            name,
            [
                [
                    python,
                    "-u",
                    str(root / "scripts" / script),
                    "--config",
                    str(config_paths[name]),
                ]
            ],
            [checkpoints[name]],
            config_path=config_paths[name],
            checkpoint_path=checkpoints[name],
            inputs=dependencies,
        )
        if args.stop_after == name:
            return

    if regeneration:
        formal_state_path = fold_work / "m33a_formal_state.pt"
        formal_materialize_command = [
            python,
            "-u",
            str(root / "scripts" / "build_p4_r0b_m33a_formal_oof.py"),
            "--config",
            str(config_paths["evidence_heldout"]),
            "--checkpoint",
            str(checkpoints["evidence"]),
            "--fold-summary",
            str(manifest_path),
            "--pipeline-manifest",
            str(pipeline_path),
            "--fold-id",
            str(args.fold_id),
            "--output",
            str(formal_state_path),
            "--batch-size",
            str(args.inference_batch_size),
            "--device",
            args.device,
        ]

        def validate_formal_state() -> None:
            payload = torch.load(formal_state_path, map_location="cpu")
            validate_m33a_formal_oof_payload(
                payload,
                expected_fold_id=args.fold_id,
                expected_record_ids=list(fold["heldout_record_ids"]),
            )
            validate_regeneration_metadata(
                dict(payload["metadata"]),
                authorization_sha256=str(
                    regeneration["regeneration_authorization_sha256"]
                ),
                fold_id=args.fold_id,
                experiment_id=str(
                    regeneration["regeneration_experiment_id"]
                ),
            )

        pipeline.run(
            "formal_materialize",
            [formal_materialize_command],
            [formal_state_path],
            inputs=[
                checkpoints["hierarchical"],
                checkpoints["coarse"],
                checkpoints["fine"],
                checkpoints["evidence"],
                candidate_paths["heldout_r16"],
                candidate_paths["heldout_r36"],
                config_paths["evidence_heldout"],
            ],
            validator=validate_formal_state,
        )
        if args.stop_after == "formal_materialize":
            return
        if args.dry_run:
            return
        pipeline.seal()
        payload = torch.load(formal_state_path, map_location="cpu")
        validated = validate_m33a_formal_oof_payload(
            payload,
            expected_fold_id=args.fold_id,
            expected_record_ids=list(fold["heldout_record_ids"]),
        )
        print(
            json.dumps(
                {
                    "fold_id": args.fold_id,
                    "records": validated["records"],
                    "pipeline_manifest": str(pipeline_path),
                    "formal_state": str(formal_state_path),
                    "siglip2_included": False,
                    "reliability_included": False,
                    "test_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    siglip2_commands = []
    for split, source in (
        ("train", train_file),
        ("dev", dev_file),
        ("heldout", heldout_file),
    ):
        siglip2_commands.append(
            [
                python,
                "-u",
                str(root / "scripts" / "build_siglip2_region_cache.py"),
                "--formal-cache",
                str(candidate_paths[f"{split}_r16"]),
                "--expanded-cache",
                str(candidate_paths[f"{split}_r36"]),
                "--source-file",
                str(source),
                "--image-dir",
                str(image_dir),
                "--model-name",
                str(resolve(args.siglip2_model, root)),
                "--output-dir",
                str(siglip2_dirs[split]),
                "--split",
                "dev" if split == "dev" else "train",
                "--context-expansion",
                "1.5",
                "--batch-size",
                str(args.siglip2_batch_size),
                "--shard-size",
                str(args.siglip2_shard_size),
                "--fp16",
                "--resume",
                "--device",
                args.device,
            ]
        )
    pipeline.run(
        "siglip2_caches",
        siglip2_commands,
        list(siglip2_manifests.values()),
        inputs=list(candidate_paths.values()),
        validator=lambda: validate_siglip2_caches(
            siglip2_manifests, candidate_paths
        ),
    )
    if args.stop_after == "siglip2_caches":
        return

    reliability_command = [
        python,
        "-u",
        str(root / "scripts" / "train_siglip2_region_reliability.py"),
        "--config",
        str(config_paths["reliability"]),
    ]
    pipeline.run(
        "reliability",
        [reliability_command],
        [checkpoints["reliability"]],
        config_path=config_paths["reliability"],
        checkpoint_path=checkpoints["reliability"],
        inputs=[
            checkpoints["fine"],
            checkpoints["evidence"],
            siglip2_manifests["train"],
            siglip2_manifests["dev"],
        ],
    )
    if args.stop_after == "reliability":
        return

    if args.dry_run:
        print("[materialize] skipped in dry-run until supervised stages exist")
        return
    pipeline.seal()
    proof_path = fold_work / "fold_proof.json"
    feature_path = fold_work / "heldout_features.pt"
    proof_command = [
        python,
        str(root / "scripts" / "create_null_release_fold_proof.py"),
        "--config",
        str(config_paths["release_materialize"]),
        "--fold-summary",
        str(manifest_path),
        "--pipeline-manifest",
        str(pipeline_path),
        "--fold-id",
        str(args.fold_id),
        "--output",
        str(proof_path),
    ]
    feature_command = [
        python,
        "-u",
        str(root / "scripts" / "build_null_release_oof_features.py"),
        "--config",
        str(config_paths["release_materialize"]),
        "--fold-proof",
        str(proof_path),
        "--output",
        str(feature_path),
        "--batch-size",
        str(args.inference_batch_size),
        "--device",
        args.device,
    ]
    print("+ " + subprocess.list2cmdline(proof_command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment["GMNER_FULL_CHAIN_OOF"] = "1"
    subprocess.run(proof_command, cwd=root, env=environment, check=True)
    rebuild_features = True
    if args.resume and feature_path.is_file():
        existing_payload = torch.load(feature_path, map_location="cpu")
        existing_metadata = dict(existing_payload.get("metadata") or {})
        try:
            validate_fold_oof_payload(
                existing_payload,
                expected_fold_id=args.fold_id,
                expected_record_ids=list(fold["heldout_record_ids"]),
                require_reliability=True,
            )
            rebuild_features = (
                existing_metadata.get("fold_proof_sha256") != sha256_file(proof_path)
            )
        except ValueError:
            rebuild_features = True
        if not rebuild_features:
            print(f"[resume] verified existing {feature_path}", flush=True)
    if rebuild_features:
        command = feature_command
        print("+ " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)
    payload = torch.load(feature_path, map_location="cpu")
    validated = validate_fold_oof_payload(
        payload,
        expected_fold_id=args.fold_id,
        expected_record_ids=list(fold["heldout_record_ids"]),
        require_reliability=True,
    )
    if regeneration:
        validate_regeneration_metadata(
            dict(payload["metadata"]),
            authorization_sha256=str(
                regeneration["regeneration_authorization_sha256"]
            ),
            fold_id=args.fold_id,
            experiment_id=str(
                regeneration["regeneration_experiment_id"]
            ),
        )
    print(
        json.dumps(
            {
                "fold_id": args.fold_id,
                "records": validated["records"],
                "pipeline_manifest": str(pipeline_path),
                "fold_proof": str(proof_path),
                "features": str(feature_path),
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
