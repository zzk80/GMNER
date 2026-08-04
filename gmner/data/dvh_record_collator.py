"""Record-level collation with frozen CLIP global and patch features."""

from __future__ import annotations

from typing import Any

import torch

from gmner.data.record_level_stage1_collator import RecordLevelStage1Collator


class DVHRecordCollator(RecordLevelStage1Collator):
    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(records)
        if not records:
            return batch
        feature_dims = {
            int(torch.as_tensor(record["clip_global_feature"]).numel())
            for record in records
        }
        if len(feature_dims) != 1:
            raise ValueError("DVH records use inconsistent CLIP dimensions.")
        feature_dim = next(iter(feature_dims))
        max_patches = max(
            int(torch.as_tensor(record["clip_patch_features"]).size(0))
            for record in records
        )
        batch_size = len(records)
        global_features = torch.zeros(
            batch_size, feature_dim, dtype=torch.float32
        )
        patch_features = torch.zeros(
            batch_size, max_patches, feature_dim, dtype=torch.float32
        )
        patch_mask = torch.zeros(
            batch_size, max_patches, dtype=torch.bool
        )
        for row, record in enumerate(records):
            global_feature = torch.as_tensor(
                record["clip_global_feature"], dtype=torch.float32
            )
            patches = torch.as_tensor(
                record["clip_patch_features"], dtype=torch.float32
            )
            mask = torch.as_tensor(
                record["clip_patch_mask"], dtype=torch.bool
            )
            if global_feature.shape != (feature_dim,):
                raise ValueError("DVH global CLIP feature shape mismatch.")
            if patches.ndim != 2 or patches.size(1) != feature_dim:
                raise ValueError("DVH patch CLIP feature shape mismatch.")
            if mask.shape != (patches.size(0),):
                raise ValueError("DVH CLIP patch mask shape mismatch.")
            count = int(patches.size(0))
            global_features[row] = global_feature
            patch_features[row, :count] = patches
            patch_mask[row, :count] = mask
        batch["clip_global_features"] = global_features
        batch["clip_patch_features"] = patch_features
        batch["clip_patch_mask"] = patch_mask
        return batch
