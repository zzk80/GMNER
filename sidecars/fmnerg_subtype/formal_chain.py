"""Read-only export of the frozen Evidence Visibility prediction chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gmner.data import (
    PairedRecordCandidateCollator,
    PairedRecordCandidateDataset,
    RecordCandidateDataset,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.models.evidence_visibility import decode_evidence_visibility
from scripts.train_evidence_visibility import load_frozen_chain
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)

from .data import fine_gold_by_record, read_fine_conll
from .io import sha256_file
from .metrics import (
    canonical_coarse_prediction_sha256,
    coarse_end_to_end_metrics,
)
from .taxonomy import SubtypeTaxonomy


@torch.inference_mode()
def export_evidence_visibility_predictions(
    *,
    root: Path,
    taxonomy: SubtypeTaxonomy,
    source_file: str | Path,
    evidence_config_path: str | Path,
    evidence_checkpoint_path: str | Path,
    formal_cache_path: str | Path,
    expanded_cache_path: str | Path,
    device: torch.device,
    batch_size: int | None = None,
) -> dict[str, Any]:
    config_path = resolve(evidence_config_path, root)
    checkpoint_path = resolve(evidence_checkpoint_path, root)
    formal_path = resolve(formal_cache_path, root)
    expanded_path = resolve(expanded_cache_path, root)
    source_path = resolve(source_file, root)
    config = load_evidence_visibility_config(config_path)
    config.runtime.device = str(device)

    formal_dataset = RecordCandidateDataset(formal_path)
    expanded_dataset = RecordCandidateDataset(expanded_path)
    paired = PairedRecordCandidateDataset(formal_dataset, expanded_dataset)
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    fine_model.eval()
    hierarchy.eval()
    for frozen in (model, fine_model, hierarchy):
        for parameter in frozen.parameters():
            parameter.requires_grad = False

    loader = DataLoader(
        paired,
        batch_size=int(batch_size or config.optim.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=PairedRecordCandidateCollator(),
    )
    options = decode_options(hierarchy_config)
    entity_threshold = float(options.get("entity_threshold", 0.0))
    decode_strategy = str(options.get("decode_strategy", "interval"))
    stage1_spans_only = bool(options.get("stage1_spans_only", True))
    region_options = {
        key: value
        for key, value in options.items()
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    visibility_enabled = bool(
        options.get("enable_visibility_correction", True)
    )
    visible_threshold = float(
        options.get("visible_from_null_threshold", 0.8)
    )
    null_threshold = float(
        options.get("null_from_visible_threshold", 0.2)
    )

    fine_records = read_fine_conll(
        source_path,
        taxonomy,
        require_all_subtypes=True,
    )
    fine_by_record = fine_gold_by_record(fine_records, taxonomy)
    tokens_by_record = {
        record.record_id: list(record.tokens) for record in fine_records
    }
    records: list[dict[str, Any]] = []
    observed_ids: set[str] = set()

    for raw_batch in tqdm(loader, desc="Exporting frozen formal predictions"):
        paired_batch = move_paired_record_batch(raw_batch, device)
        formal = paired_batch["formal"]
        expanded = paired_batch["expanded"]
        baseline = frozen_hierarchical_context(
            hierarchy,
            formal,
            expanded,
            decode_options=region_options,
        )
        hierarchy_outputs = baseline["outputs"]
        decoded = baseline["decoded"]
        baseline_visible = baseline["visible_mask"]
        if not isinstance(hierarchy_outputs, dict) or not isinstance(decoded, dict):
            raise TypeError("Frozen hierarchy returned invalid outputs.")
        if not isinstance(baseline_visible, torch.Tensor):
            raise TypeError("Frozen hierarchy visibility mask is missing.")

        fine_outputs = fine_model(expanded)
        evidence_outputs = model(
            fine_outputs,
            hierarchy_outputs,
            expanded,
            baseline_visible_mask=baseline_visible,
            base_is_null_mask=decoded["base_is_null"].bool(),
        )
        has_null = expanded["region_is_null"].bool().any(dim=-1)[:, None]
        has_null = has_null.expand_as(baseline_visible)
        final_visible = decode_evidence_visibility(
            evidence_outputs["final_visibility_probability"],
            base_is_null=decoded["base_is_null"].bool(),
            baseline_visible=baseline_visible,
            has_real_candidate=evidence_outputs["fine_has_real_candidate"],
            has_null_region=has_null,
            span_mask=expanded["span_mask"],
            visible_from_null_threshold=visible_threshold,
            null_from_visible_threshold=null_threshold,
            enabled=visibility_enabled,
        )
        fine_indices = evidence_outputs["fine_top1_region_index"].long()
        expanded_null = torch.tensor(
            [
                int(metadata.get("null_region_index", -1))
                for metadata in expanded["metadata"]
            ],
            device=device,
            dtype=torch.long,
        )[:, None].expand_as(fine_indices)
        final_indices = torch.where(
            final_visible,
            fine_indices,
            expanded_null,
        )

        for row, metadata in enumerate(expanded["metadata"]):
            record_id = str(metadata.get("record_id", ""))
            if not record_id or record_id in observed_ids:
                raise ValueError(
                    f"Missing or duplicate formal record id: {record_id!r}"
                )
            if record_id not in fine_by_record:
                raise ValueError(
                    f"Formal cache record {record_id!r} is absent from fine source."
                )
            observed_ids.add(record_id)
            spans, selected = _selected_span_indices(
                hierarchy_outputs,
                formal,
                row,
                entity_threshold=entity_threshold,
                decode_strategy=decode_strategy,
                stage1_spans_only=stage1_spans_only,
            )
            predictions = [
                {
                    "span": list(spans[span_index]),
                    "type_id": int(
                        hierarchy_outputs["fixed_type_ids"][
                            row, span_index
                        ].item()
                    ),
                    "region_index": int(
                        final_indices[row, span_index].item()
                    ),
                }
                for span_index in selected
            ]
            gold_entities = []
            raw_gold = fine_by_record[record_id]
            for target in list(metadata.get("gold_entities") or []):
                span = tuple(map(int, target["span"]))
                fine_target = raw_gold.get(span)
                if fine_target is None:
                    raise ValueError(
                        f"Fine label missing for record={record_id} span={span}."
                    )
                if int(target["type_id"]) != int(
                    fine_target["coarse_type_id"]
                ):
                    raise ValueError(
                        f"Coarse/fine parent mismatch for record={record_id} "
                        f"span={span}."
                    )
                gold_entities.append(
                    {
                        **target,
                        "span": list(span),
                        "type_id": int(target["type_id"]),
                        "subtype": str(fine_target["subtype"]),
                        "subtype_id": int(fine_target["subtype_id"]),
                        "region_positive_indices": [
                            int(value)
                            for value in target.get(
                                "region_positive_indices"
                            )
                            or []
                        ],
                    }
                )
            if len(gold_entities) != len(raw_gold):
                raise ValueError(
                    f"Formal cache gold count differs from fine source for "
                    f"record {record_id}: cache={len(gold_entities)} "
                    f"fine={len(raw_gold)}."
                )
            records.append(
                {
                    "record_id": record_id,
                    "tokens": tokens_by_record[record_id],
                    "predictions": predictions,
                    "gold_entities": gold_entities,
                }
            )

    if len(records) != len(fine_records):
        raise ValueError(
            f"Formal export contains {len(records)} records; "
            f"fine source contains {len(fine_records)}."
        )
    coarse_metrics = coarse_end_to_end_metrics(records)
    prediction_sha = canonical_coarse_prediction_sha256(records)
    return {
        "metadata": {
            "kind": "fmnerg_frozen_formal_predictions",
            "format_version": 1,
            "split": "dev",
            "records": len(records),
            "predictions": int(coarse_metrics["predicted"]),
            "gold": int(coarse_metrics["gold"]),
            "coarse_prediction_sha256": prediction_sha,
            "source_file": str(source_path),
            "source_sha256": sha256_file(source_path),
            "taxonomy_sha256": taxonomy.source_sha256,
            "evidence_config": str(config_path),
            "evidence_config_sha256": sha256_file(config_path),
            "evidence_checkpoint": str(checkpoint_path),
            "evidence_checkpoint_sha256": sha256_file(checkpoint_path),
            "formal_cache": str(formal_path),
            "formal_cache_sha256": sha256_file(formal_path),
            "expanded_cache": str(expanded_path),
            "expanded_cache_sha256": sha256_file(expanded_path),
            "coarse_metrics": coarse_metrics,
            "test_accessed": False,
        },
        "records": records,
    }


def save_formal_predictions(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
