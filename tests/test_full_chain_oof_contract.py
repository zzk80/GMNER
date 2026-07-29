from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gmner.data.full_chain_oof_contract import (
    FULL_CHAIN_PIPELINE_KIND,
    FULL_CHAIN_PIPELINE_VERSION,
    REQUIRED_PIPELINE_STAGES,
    SUPERVISED_PIPELINE_STAGES,
    fold_from_manifest,
    validate_fold_manifest,
    validate_pipeline_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file
from gmner.coarse_region_selector_config import load_coarse_region_selector_config
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.fine_grounding_adapter_config import load_fine_grounding_adapter_config
from gmner.hierarchical_record_verifier_config import (
    load_hierarchical_record_verifier_config,
)
from gmner.layered_action_verifier_config import (
    load_layered_action_verifier_config,
)
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
from scripts.build_evidence_folds import main as build_folds
from scripts.run_null_release_full_chain_oof_fold import (
    _candidate_command,
    _no_test_contract,
    adopt_source_revision,
    main as run_fold,
)


def _write_records(path: Path, count: int = 20) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index in range(count):
            stream.write(
                json.dumps(
                    {
                        "id": str(index),
                        "tokens": [f"token-{index}"],
                        "ner_tags": [0],
                        "image": f"{index}.jpg",
                    }
                )
                + "\n"
            )


def _build_manifest(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    source = tmp_path / "train.jsonl"
    _write_records(source)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"data": {"train_file": str(source)}}),
        encoding="utf-8",
    )
    output = tmp_path / "folds"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_evidence_folds.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--num-folds",
            "10",
            "--seed",
            "42",
        ],
    )
    build_folds()
    path = output / "fold_summary.json"
    return path, validate_fold_manifest(path, expected_num_folds=10)


def test_ten_fold_manifest_is_complete_disjoint_and_reproducible(
    tmp_path, monkeypatch
) -> None:
    path, manifest = _build_manifest(tmp_path, monkeypatch)

    assert manifest["test_accessed"] is False
    assert manifest["records"] == 20
    assert len(manifest["folds"]) == 10
    heldout = [
        value
        for fold in manifest["folds"]
        for value in fold["heldout_record_ids"]
    ]
    assert set(heldout) == set(manifest["record_ids"])
    assert len(heldout) == len(set(heldout))
    assert validate_fold_manifest(path, expected_num_folds=10) == manifest


def test_manifest_can_verify_only_authorized_fold_files(
    tmp_path, monkeypatch
) -> None:
    path, manifest = _build_manifest(tmp_path, monkeypatch)
    for fold_id in (8, 9):
        fold = fold_from_manifest(manifest, fold_id)
        Path(fold["train_file"]).unlink()
        Path(fold["heldout_file"]).unlink()

    validated = validate_fold_manifest(
        path,
        expected_num_folds=10,
        verify_fold_ids=range(8),
    )
    assert validated["records"] == 20
    with pytest.raises(FileNotFoundError):
        validate_fold_manifest(path, expected_num_folds=10)


def test_pipeline_contract_requires_all_supervised_fold_specific_stages(
    tmp_path, monkeypatch
) -> None:
    _, manifest = _build_manifest(tmp_path, monkeypatch)
    fold = fold_from_manifest(manifest, 0)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    item = {"path": str(artifact), "sha256": sha256_file(artifact)}
    stages = {}
    for name in REQUIRED_PIPELINE_STAGES:
        stage = {
            "status": "complete",
            "test_accessed": False,
            "inputs": [item],
            "outputs": [item],
        }
        if name in SUPERVISED_PIPELINE_STAGES:
            stage.update(
                {
                    "heldout_excluded": True,
                    "train_record_ids_sha256": fold[
                        "train_record_ids_sha256"
                    ],
                    "config": item,
                    "checkpoint": item,
                }
            )
        stages[name] = stage
    pipeline = {
        "format_version": FULL_CHAIN_PIPELINE_VERSION,
        "kind": FULL_CHAIN_PIPELINE_KIND,
        "fold_id": 0,
        "source_tree_sha256": manifest["source_tree_sha256"],
        "train_record_ids_sha256": fold["train_record_ids_sha256"],
        "heldout_record_ids_sha256": fold["heldout_record_ids_sha256"],
        "test_accessed": False,
        "stages": stages,
    }
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(pipeline), encoding="utf-8")

    assert validate_pipeline_manifest(
        path, fold_manifest=manifest, fold_id=0
    ) == pipeline
    pipeline["stages"]["fine"]["heldout_excluded"] = False
    path.write_text(json.dumps(pipeline), encoding="utf-8")
    with pytest.raises(ValueError, match="heldout exclusion"):
        validate_pipeline_manifest(path, fold_manifest=manifest, fold_id=0)


def test_generated_oof_configs_reject_test_inputs() -> None:
    _no_test_contract(
        {
            "data": {"test_file": "__OOF_TEST_DISABLED__"},
            "runtime": {"evaluate_test_after_training": False},
        },
        allow_disabled_stage1_key=True,
    )
    with pytest.raises(ValueError, match="test field"):
        _no_test_contract({"data": {"test_cache": "secret-test.pt"}})


def test_expanded_candidate_command_uses_formal_anchor(tmp_path) -> None:
    formal = tmp_path / "formal.pt"
    command = _candidate_command(
        python="python",
        root=tmp_path,
        config=tmp_path / "config.yaml",
        checkpoint=tmp_path / "model.pt",
        source=tmp_path / "train.jsonl",
        output=tmp_path / "expanded.pt",
        max_regions=36,
        fold_id=0,
        split="train",
        batch_size=8,
        device="cuda",
        formal_anchor_cache=formal,
    )

    anchor_index = command.index("--formal-anchor-cache")
    assert command[anchor_index + 1] == str(formal)


def test_source_revision_preserves_only_verified_stage1(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"stage1")
    descriptor = {"path": str(artifact), "sha256": sha256_file(artifact)}
    manifest_path = tmp_path / "fold_summary.json"
    manifest_path.write_text(
        json.dumps({"source_tree_sha256": "old"}),
        encoding="utf-8",
    )
    stage1 = {
        "status": "complete",
        "test_accessed": False,
        "inputs": [descriptor],
        "outputs": [descriptor],
        "config": descriptor,
        "checkpoint": descriptor,
    }
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps(
            {
                "source_tree_sha256": "old",
                "fold_manifest_sha256": sha256_file(manifest_path),
                "test_accessed": False,
                "sealed": False,
                "stages": {
                    "stage1": stage1,
                    "candidate_caches": {"status": "complete"},
                    "fine": {"status": "failed"},
                },
            }
        ),
        encoding="utf-8",
    )

    adopt_source_revision(
        manifest_path=manifest_path,
        pipeline_path=pipeline_path,
        new_source_sha256="new",
        reason="anchor expanded candidates to formal Stage1",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert manifest["source_tree_sha256"] == "new"
    assert pipeline["source_tree_sha256"] == "new"
    assert list(pipeline["stages"]) == ["stage1"]
    assert pipeline["source_revision_history"][-1]["invalidated_from"] == (
        "candidate_caches"
    )


def test_source_revision_can_invalidate_only_siglip_and_downstream(
    tmp_path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified")
    item = {"path": str(artifact), "sha256": sha256_file(artifact)}
    manifest_path = tmp_path / "fold_summary.json"
    manifest_path.write_text(
        json.dumps({"source_tree_sha256": "old"}),
        encoding="utf-8",
    )
    stages = {}
    for name in (
        "stage1",
        "candidate_caches",
        "hierarchical",
        "coarse",
        "fine",
        "evidence",
    ):
        stage = {
            "status": "complete",
            "test_accessed": False,
            "inputs": [item],
            "outputs": [item],
        }
        if name in SUPERVISED_PIPELINE_STAGES:
            stage.update({"config": item, "checkpoint": item})
        stages[name] = stage
    stages["siglip2_caches"] = {"status": "failed"}
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        json.dumps(
            {
                "source_tree_sha256": "old",
                "fold_manifest_sha256": sha256_file(manifest_path),
                "test_accessed": False,
                "sealed": False,
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )

    adopt_source_revision(
        manifest_path=manifest_path,
        pipeline_path=pipeline_path,
        new_source_sha256="new",
        reason="separate SigLIP2 tokenizer and image processor",
        invalidate_from="siglip2_caches",
    )

    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert list(pipeline["stages"]) == [
        "stage1",
        "candidate_caches",
        "hierarchical",
        "coarse",
        "fine",
        "evidence",
    ]


def test_fold0_orchestrator_dry_run_stops_before_training(
    tmp_path, monkeypatch
) -> None:
    manifest_path, _ = _build_manifest(tmp_path, monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_null_release_full_chain_oof_fold.py",
            "--stage1-config",
            str(root / "configs" / "fmnerg_twitter10000_stage1.yaml"),
            "--fold-summary",
            str(manifest_path),
            "--work-root",
            str(tmp_path / "work"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--dry-run",
            "--stop-after",
            "stage1",
        ],
    )

    run_fold()

    generated = yaml.safe_load(
        (tmp_path / "work" / "fold0" / "configs" / "stage1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert generated["data"]["test_file"] == "__OOF_TEST_DISABLED__"
    generated_root = tmp_path / "work" / "fold0" / "configs"
    hierarchy = load_hierarchical_record_verifier_config(
        generated_root / "hierarchical.yaml"
    )
    assert hierarchy.data.test_cache is None
    assert hierarchy.runtime.evaluate_test_after_training is False
    load_coarse_region_selector_config(generated_root / "coarse.yaml")
    load_fine_grounding_adapter_config(generated_root / "fine.yaml")
    load_evidence_visibility_config(generated_root / "evidence.yaml")
    load_siglip2_region_reliability_config(generated_root / "reliability.yaml")
    heldout = load_siglip2_region_reliability_config(
        generated_root / "reliability_heldout.yaml"
    )
    assert heldout.data.require_oof_train_cache is True
    load_layered_action_verifier_config(
        generated_root / "release_materialize.yaml"
    )
