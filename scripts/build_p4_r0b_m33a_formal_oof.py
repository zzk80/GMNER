"""Materialize one regenerated heldout fold of minimal M3.3A formal state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import (
    PairedRecordCandidateCollator,
    PairedRecordCandidateDataset,
    RecordCandidateDataset,
)
from gmner.data.full_chain_oof_contract import (
    fold_from_manifest,
    validate_fold_manifest,
    validate_pipeline_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from gmner.data.p4_r0b_regeneration_contract import (
    P4_R0B_EXECUTION_FOLDS,
    P4_R0B_M33A_CACHE_KIND,
    P4_R0B_M33A_CACHE_VERSION,
    P4_R0B_M33A_REQUIRED_STAGES,
    P4_R0B_M33A_SUPERVISED_STAGES,
    pack_m33a_formal_batch,
    validate_m33a_formal_oof_payload,
    validate_regeneration_metadata,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    move_paired_record_batch,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    frozen_hierarchical_context,
)
from gmner.models.evidence_visibility import decode_evidence_visibility
from gmner.evidence_visibility_config import load_evidence_visibility_config
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
    parser.add_argument("--fold-summary", required=True)
    parser.add_argument("--pipeline-manifest", required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _record_ids(dataset: PairedRecordCandidateDataset) -> list[str]:
    return [
        str(dict(record.get("metadata") or {}).get("record_id", ""))
        for record in dataset.formal.records
    ]


def fine_topk_action_indices(
    logits: torch.Tensor, real_mask: torch.Tensor, *, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != real_mask.shape:
        raise ValueError("logits and real_mask must have identical shapes.")
    count = min(max(int(top_k), 0), logits.size(-1))
    indices = logits.float().masked_fill(~real_mask.bool(), -1e4).topk(
        count, dim=-1
    ).indices
    valid = real_mask.bool().gather(-1, indices)
    if count < int(top_k):
        padding = int(top_k) - count
        indices = F.pad(indices, (0, padding), value=0)
        valid = F.pad(valid, (0, padding), value=False)
    return indices, valid


@torch.no_grad()
def frozen_formal_context(
    evidence_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchical_model: torch.nn.Module,
    formal_batch: dict,
    expanded_batch: dict,
    *,
    decode_options: dict,
) -> dict:
    region_options = {
        key: value
        for key, value in decode_options.items()
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    hierarchy = frozen_hierarchical_context(
        hierarchical_model,
        formal_batch,
        expanded_batch,
        decode_options=region_options,
    )
    hierarchy_outputs = hierarchy["outputs"]
    decoded = hierarchy["decoded"]
    baseline_visible = hierarchy["visible_mask"]
    fine_outputs = fine_model(expanded_batch)
    evidence_outputs = evidence_model(
        fine_outputs,
        hierarchy_outputs,
        expanded_batch,
        baseline_visible_mask=baseline_visible,
        base_is_null_mask=decoded["base_is_null"],
    )
    has_null = expanded_batch["region_is_null"].bool().any(dim=-1)[:, None]
    has_null = has_null.expand_as(baseline_visible)
    final_visible = decode_evidence_visibility(
        evidence_outputs["final_visibility_probability"],
        base_is_null=decoded["base_is_null"].bool(),
        baseline_visible=baseline_visible,
        has_real_candidate=evidence_outputs["fine_has_real_candidate"],
        has_null_region=has_null,
        span_mask=expanded_batch["span_mask"],
        visible_from_null_threshold=float(
            decode_options.get("visible_from_null_threshold", 0.8)
        ),
        null_from_visible_threshold=float(
            decode_options.get("null_from_visible_threshold", 0.2)
        ),
        enabled=bool(decode_options.get("enable_visibility_correction", True)),
    )
    return {
        "hierarchy": hierarchy,
        "hierarchy_outputs": hierarchy_outputs,
        "fine_outputs": fine_outputs,
        "evidence_outputs": evidence_outputs,
        "final_visible_mask": final_visible,
    }


def _deployment_span_mask(
    hierarchy_outputs: dict,
    formal: dict,
    expanded: dict,
    *,
    options: dict,
) -> torch.Tensor:
    selected = torch.zeros_like(expanded["span_mask"]).bool()
    for row in range(selected.size(0)):
        _, indices = _selected_span_indices(
            hierarchy_outputs,
            formal,
            row,
            entity_threshold=float(options.get("entity_threshold", 0.0)),
            decode_strategy=str(options.get("decode_strategy", "interval")),
            stage1_spans_only=bool(options.get("stage1_spans_only", True)),
        )
        if indices:
            selected[
                row,
                torch.as_tensor(indices, device=selected.device),
            ] = True
    return selected


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.fold_id not in P4_R0B_EXECUTION_FOLDS:
        raise PermissionError("M3.3A regeneration is limited to folds 0-7.")
    config_path = resolve(args.config, root)
    checkpoint_path = resolve(args.checkpoint, root)
    manifest_path = resolve(args.fold_summary, root)
    pipeline_path = resolve(args.pipeline_manifest, root)
    manifest = validate_fold_manifest(
        manifest_path,
        expected_num_folds=10,
        verify_fold_ids=P4_R0B_EXECUTION_FOLDS,
    )
    fold = fold_from_manifest(manifest, args.fold_id)
    validate_pipeline_manifest(
        pipeline_path,
        fold_manifest=manifest,
        fold_id=args.fold_id,
        required_stages=P4_R0B_M33A_REQUIRED_STAGES[:-1],
        supervised_stages=P4_R0B_M33A_SUPERVISED_STAGES,
    )
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    regeneration = dict(pipeline.get("regeneration") or {})

    config = load_evidence_visibility_config(config_path)
    formal_path = resolve(config.data.formal_train_cache, root)
    expanded_path = resolve(config.data.expanded_train_cache, root)
    formal = RecordCandidateDataset(formal_path)
    expanded = RecordCandidateDataset(expanded_path)
    dataset = PairedRecordCandidateDataset(formal, expanded)
    actual_ids = _record_ids(dataset)
    expected_ids = [str(value) for value in fold["heldout_record_ids"]]
    if actual_ids != expected_ids:
        raise ValueError("Heldout M3.3A candidate order differs from the fold.")
    for metadata in (formal.metadata, expanded.metadata):
        validate_regeneration_metadata(
            dict(metadata),
            authorization_sha256=str(
                regeneration["regeneration_authorization_sha256"]
            ),
            fold_id=args.fold_id,
            experiment_id=str(
                regeneration["regeneration_experiment_id"]
            ),
        )

    device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    (
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
    ) = load_frozen_chain(config, root, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    evidence_model.load_state_dict(checkpoint["model_state_dict"])
    evidence_model.to(device).eval()
    for model in (evidence_model, fine_model, hierarchy):
        model.eval()
    validate_fingerprints(
        dataset,
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=True,
    )

    options = decode_options(hierarchy_config)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=PairedRecordCandidateCollator(),
    )
    batches = []
    cached_ids = []
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for raw_batch in tqdm(
            loader,
            desc=f"M3.3A formal OOF fold {args.fold_id}",
        ):
            paired = move_paired_record_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                context = frozen_formal_context(
                    evidence_model,
                    fine_model,
                    hierarchy,
                    paired["formal"],
                    paired["expanded"],
                    decode_options=options,
                )
            fine_outputs = dict(context["fine_outputs"])
            real_mask = (
                fine_outputs["candidate_mask"].bool()
                & paired["expanded"]["region_mask"].bool()[:, None, :]
                & ~paired["expanded"]["region_is_null"].bool()[:, None, :]
            )
            top4_indices, top4_valid = fine_topk_action_indices(
                fine_outputs["final_region_logits"],
                real_mask,
                top_k=4,
            )
            fine_outputs["fine_top4_indices"] = top4_indices
            fine_outputs["fine_top4_valid_mask"] = top4_valid
            deployment = _deployment_span_mask(
                context["hierarchy_outputs"],
                paired["formal"],
                paired["expanded"],
                options=options,
            )
            packed = pack_m33a_formal_batch(
                {
                    "expanded": paired["expanded"],
                    "fine_outputs": fine_outputs,
                    "hierarchy_outputs": context["hierarchy_outputs"],
                    "current_visible": context["final_visible_mask"],
                    "base_is_null": context["hierarchy"]["decoded"][
                        "base_is_null"
                    ],
                    "deployment_span_mask": deployment,
                },
                fold_id=args.fold_id,
            )
            cached_ids.extend(packed["record_ids"])
            batches.append(packed)
    if cached_ids != expected_ids:
        raise RuntimeError("Materialized M3.3A record order changed.")

    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "kind": P4_R0B_M33A_CACHE_KIND,
            "format_version": P4_R0B_M33A_CACHE_VERSION,
            "fold_id": args.fold_id,
            "num_folds": 10,
            "records": len(cached_ids),
            "record_ids_sha256": stable_id_digest(cached_ids),
            "heldout_record_ids": expected_ids,
            "full_chain_oof": True,
            **regeneration,
            "artifact_sha256": {
                "config": sha256_file(config_path),
                "checkpoint": sha256_file(checkpoint_path),
                "formal_cache": sha256_file(formal_path),
                "expanded_cache": sha256_file(expanded_path),
                "pipeline_manifest": sha256_file(pipeline_path),
            },
            "siglip2_included": False,
            "reliability_included": False,
            "null_release_included": False,
            "p4_attached": False,
            "test_accessed": False,
        },
        "batches": batches,
    }
    validate_m33a_formal_oof_payload(
        payload,
        expected_fold_id=args.fold_id,
        expected_record_ids=expected_ids,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "fold_id": args.fold_id,
                "records": len(cached_ids),
                "output": str(output),
                "siglip2_included": False,
                "reliability_included": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
