from __future__ import annotations

import torch

from gmner.data.record_candidate_dataset import RecordCandidateDataset
from scripts.merge_oof_record_candidate_caches import merge_caches

from test_hierarchical_record_verifier import _record


def _write_fold(path, *, fold: int, record_id: str) -> None:
    record = _record()
    record["metadata"]["record_id"] = record_id
    payload = {
        "metadata": {
            "format_version": 2,
            "split": "train",
            "oof_fold_id": fold,
            "oof_heldout": True,
            "stage1_checkpoint_sha256": f"stage1-{fold}",
            "data_source": f"heldout-{fold}.jsonl",
            "data_source_sha256": f"data-{fold}",
            "candidate_config": {"k_best": 6},
            "candidate_config_sha256": "candidate",
            "transition_source": "crf",
            "source2id": {"stage1": 0},
            "hidden_size": 8,
            "num_types": 4,
        },
        "records": [record],
    }
    torch.save(payload, path)


def test_merge_oof_candidate_caches_marks_and_sorts_records(tmp_path) -> None:
    fold1 = tmp_path / "fold1.pt"
    fold0 = tmp_path / "fold0.pt"
    output = tmp_path / "merged.pt"
    _write_fold(fold1, fold=1, record_id="11")
    _write_fold(fold0, fold=0, record_id="2")

    result = merge_caches([fold1, fold0], output, expected_records=2)
    dataset = RecordCandidateDataset(output)

    assert result["records"] == 2
    assert dataset.metadata["oof"]["enabled"] is True
    assert dataset.metadata["oof"]["num_folds"] == 2
    assert [record["metadata"]["record_id"] for record in dataset.records] == [
        "2",
        "11",
    ]
