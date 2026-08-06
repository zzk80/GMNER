from __future__ import annotations

import json

import pytest
import torch

from gmner.data.artifact_utils import sha256_file
from gmner.data.clip_r16_cache import (
    ClipR16Cache,
    model_artifact_fingerprint,
    require_train_or_dev_split,
    validate_clip_r16_manifest,
    write_clip_r16_cache,
)


def _record(image_id: str, offset: float = 0.0) -> dict:
    boxes = torch.zeros((16, 4), dtype=torch.float32)
    boxes[0] = torch.tensor([1.0 + offset, 2.0, 10.0, 12.0])
    valid = torch.zeros((16,), dtype=torch.bool)
    valid[0] = True
    scores = torch.zeros((16,), dtype=torch.float32)
    scores[0] = 0.75
    regions = torch.zeros((16, 8), dtype=torch.float16)
    regions[0] = torch.arange(8, dtype=torch.float16)
    return {
        "record_ids": [image_id],
        "image_id": image_id,
        "global_feature": torch.ones((8,), dtype=torch.float16),
        "region_features": regions,
        "region_boxes": boxes,
        "region_valid_mask": valid,
        "region_detector_scores": scores,
    }


def test_tp_cache_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="train/dev"):
        require_train_or_dev_split("test")
    with pytest.raises(FileNotFoundError, match="local frozen CLIP"):
        model_artifact_fingerprint("openai/clip-vit-base-patch32")


def test_tp_cache_round_trip_and_r16_identity(tmp_path) -> None:
    manifest_path = write_clip_r16_cache(
        output_dir=tmp_path,
        split="train",
        records=[_record("a"), _record("b", 1.0)],
        shard_size=1,
        metadata={"null_included": False},
    )
    manifest = validate_clip_r16_manifest(manifest_path, expected_split="train")
    assert manifest["test_accessed"] is False
    assert manifest["region_budget"] == 16
    assert len(manifest["shards"]) == 2

    cache = ClipR16Cache(tmp_path, expected_split="train", max_open_shards=1)
    expected = _record("a")
    loaded = cache.get(
        "a",
        expected_boxes=expected["region_boxes"],
        expected_valid_mask=expected["region_valid_mask"],
    )
    assert loaded["region_features"].dtype == torch.float16
    assert loaded["region_valid_mask"].sum().item() == 1
    assert loaded["region_features"].shape == (16, 8)

    drifted = expected["region_boxes"].clone()
    drifted[0, 0] += 1
    with pytest.raises(ValueError, match="box order drift"):
        cache.get("a", expected_boxes=drifted)


def test_tp_cache_preload_retains_every_shard(tmp_path) -> None:
    write_clip_r16_cache(
        output_dir=tmp_path,
        split="train",
        records=[_record("a"), _record("b", 1.0)],
        shard_size=1,
        metadata={"null_included": False},
    )
    cache = ClipR16Cache(tmp_path, expected_split="train", max_open_shards=1)

    assert cache.preload_all() == 2
    assert set(cache._shards) == set(cache.manifest["shards"])
    assert cache.get("a")["image_id"] == "a"
    assert cache.get("b")["image_id"] == "b"


def test_tp_cache_detects_shard_tampering(tmp_path) -> None:
    manifest_path = write_clip_r16_cache(
        output_dir=tmp_path,
        split="dev",
        records=[_record("a")],
        shard_size=4,
        metadata={},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_name = next(iter(manifest["shards"]))
    shard_path = tmp_path / shard_name
    original_sha = sha256_file(shard_path)
    shard_path.write_bytes(shard_path.read_bytes() + b"tamper")
    assert sha256_file(shard_path) != original_sha
    cache = ClipR16Cache(tmp_path, expected_split="dev")
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.get("a")
