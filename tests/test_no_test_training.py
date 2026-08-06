from __future__ import annotations

from types import SimpleNamespace

import pytest

from gmner.hierarchical_record_verifier_config import (
    HierarchicalRecordDataConfig,
    HierarchicalRecordTrainingConfig,
)
from scripts.train_hierarchical_record_verifier import _configured_dataset_paths


def test_stage1_skip_test_does_not_resolve_or_build_test(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("transformers")
    import scripts.train as stage1_train

    converted: list[str] = []
    built: list[str] = []

    def fake_convert(path, _output_dir):
        converted.append(path.name)
        return path

    class FakeDataset:
        def __init__(self, *, jsonl_path: str, **_kwargs) -> None:
            built.append(jsonl_path)

    monkeypatch.setattr(stage1_train, "maybe_convert_conll", fake_convert)
    monkeypatch.setattr(stage1_train, "MMNERJsonDataset", FakeDataset)
    data = SimpleNamespace(
        train_file="train.jsonl",
        dev_file="dev.jsonl",
        test_file="must_not_be_read.jsonl",
        image_dir="images",
        image_feature_dir=None,
        image_annotation_dir=None,
        groundability_type_priors=None,
        groundability_mention_priors=None,
        grounding_enabled=False,
        expand_entities_for_grounding=False,
        max_length=128,
        max_regions=16,
        grounding_iou_threshold=0.5,
        add_null_region=True,
        region_min_score=0.0,
    )
    config = SimpleNamespace(
        data=data,
        model=SimpleNamespace(region_feature_dim=2048),
    )

    train, dev, test, labels = stage1_train.build_datasets(
        config=config,
        tokenizer=object(),
        graph_builder=object(),
        project_root=tmp_path,
        output_dir=tmp_path,
        build_test=False,
    )

    assert train is not None and dev is not None
    assert test is None
    assert labels == 9
    assert converted == ["train.jsonl", "dev.jsonl"]
    assert all("must_not_be_read" not in path for path in built)


def test_hierarchical_no_test_mode_omits_test_cache() -> None:
    config = HierarchicalRecordTrainingConfig(
        data=HierarchicalRecordDataConfig(
            train_cache="train.pt",
            dev_cache="dev.pt",
        )
    )
    config.runtime.evaluate_test_after_training = False

    assert _configured_dataset_paths(config) == {
        "train": "train.pt",
        "dev": "dev.pt",
    }


def test_hierarchical_no_test_mode_ignores_configured_test_cache() -> None:
    config = HierarchicalRecordTrainingConfig(
        data=HierarchicalRecordDataConfig(
            train_cache="train.pt",
            dev_cache="dev.pt",
            test_cache="must_not_be_read.pt",
        )
    )
    config.runtime.evaluate_test_after_training = False

    assert _configured_dataset_paths(config) == {
        "train": "train.pt",
        "dev": "dev.pt",
    }


def test_hierarchical_test_mode_requires_test_cache() -> None:
    config = HierarchicalRecordTrainingConfig(
        data=HierarchicalRecordDataConfig(
            train_cache="train.pt",
            dev_cache="dev.pt",
        )
    )
    config.runtime.evaluate_test_after_training = True

    with pytest.raises(ValueError, match="requires data.test_cache"):
        _configured_dataset_paths(config)
