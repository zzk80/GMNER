"""Dependency-light data helpers for the formal FMNERG workflow."""

from __future__ import annotations

from typing import Any


def first_record_indices(dataset: Any, max_records: int) -> list[int]:
    """Select complete entity-expanded samples for the first N records."""

    if max_records <= 0:
        raise ValueError("--max-records must be positive.")
    selected: list[int] = []
    seen: set[str] = set()
    for index, sample in enumerate(dataset.samples):
        record_id = str(sample["record_id"])
        if record_id not in seen:
            if len(seen) >= max_records:
                break
            seen.add(record_id)
        selected.append(index)
    return selected
