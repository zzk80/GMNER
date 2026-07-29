"""Materialize one held-out fold of full-chain NULL Release features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.null_release_oof_cache import (
    NULL_RELEASE_OOF_FORMAT_VERSION,
    NULL_RELEASE_OOF_KIND,
    pack_null_release_context_batch,
    sha256_file,
    stable_id_digest,
    validate_fold_oof_payload,
)
from gmner.engine.fine_grounding_adapter_evaluator import move_paired_record_batch
from gmner.engine.layered_action_verifier_evaluator import (
    frozen_layered_action_features,
)
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.fine_grounding_adapter_config import (
    load_fine_grounding_adapter_config,
)
from gmner.layered_action_verifier_config import (
    load_layered_action_verifier_config,
)
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)
from scripts.train_siglip2_region_reliability import (
    _base_paired,
    _paired_dataset,
    load_frozen_reliability_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold-proof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def _ids(values) -> list[str]:
    result = [str(value) for value in values]
    if any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError("Fold proof contains empty or duplicate record ids.")
    return result


def _artifact_paths(config_path: Path, root: Path) -> dict[str, Path]:
    action = load_layered_action_verifier_config(config_path)
    reliability_config_path = resolve(action.frozen.reliability_config, root)
    reliability = load_siglip2_region_reliability_config(reliability_config_path)
    fine_config_path = resolve(reliability.frozen.fine_config, root)
    fine = load_fine_grounding_adapter_config(fine_config_path)
    evidence_config_path = resolve(
        reliability.frozen.evidence_visibility_config, root
    )
    evidence = load_evidence_visibility_config(evidence_config_path)
    siglip2_path = resolve(reliability.data.siglip2_train_cache, root)
    siglip2_manifest = siglip2_path / "manifest.json" if siglip2_path.is_dir() else siglip2_path
    return {
        "action_config": config_path,
        "reliability_config": reliability_config_path,
        "reliability_checkpoint": resolve(action.frozen.reliability_checkpoint, root),
        "fine_config": fine_config_path,
        "fine_checkpoint": resolve(reliability.frozen.fine_checkpoint, root),
        "evidence_config": evidence_config_path,
        "evidence_checkpoint": resolve(
            reliability.frozen.evidence_visibility_checkpoint, root
        ),
        "hierarchical_config": resolve(fine.frozen.hierarchical_config, root),
        "hierarchical_checkpoint": resolve(
            fine.frozen.hierarchical_checkpoint, root
        ),
        "coarse_checkpoint": resolve(fine.frozen.coarse_checkpoint, root),
        "formal_cache": resolve(reliability.data.formal_train_cache, root),
        "expanded_cache": resolve(reliability.data.expanded_train_cache, root),
        "siglip2_manifest": siglip2_manifest,
        "evidence_fine_checkpoint": resolve(evidence.frozen.fine_checkpoint, root),
    }


def _validate_proof(proof: dict, artifacts: dict[str, Path]) -> tuple[int, list[str], list[str]]:
    fold_id = int(proof.get("fold_id", -1))
    if int(proof.get("num_folds", -1)) != 10 or fold_id not in range(10):
        raise ValueError("NULL Release formal OOF requires fold ids 0..9 of 10.")
    if not bool(proof.get("excluded_heldout")):
        raise ValueError("Fold proof must assert excluded_heldout=true.")
    training_ids = _ids(proof.get("training_record_ids") or [])
    heldout_ids = _ids(proof.get("heldout_record_ids") or [])
    if set(training_ids) & set(heldout_ids):
        raise ValueError("Fold proof training and held-out ids overlap.")
    for role in ("fold_summary", "pipeline_manifest"):
        path = Path(str(proof.get(role, "")))
        if not path.exists():
            raise FileNotFoundError(f"Fold proof is missing {role}: {path}")
        if sha256_file(path) != proof.get(f"{role}_sha256"):
            raise ValueError(f"Fold proof {role} hash mismatch.")
    expected_hashes = dict(proof.get("artifact_sha256") or {})
    if set(expected_hashes) != set(artifacts):
        missing = sorted(set(artifacts) - set(expected_hashes))
        extra = sorted(set(expected_hashes) - set(artifacts))
        raise ValueError(f"Fold proof artifact set mismatch: missing={missing}, extra={extra}.")
    for name, path in artifacts.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing fold artifact {name}: {path}")
        actual = sha256_file(path)
        if actual != str(expected_hashes[name]):
            raise ValueError(f"Fold artifact hash mismatch for {name}: {path}")
    return fold_id, training_ids, heldout_ids


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    config = load_layered_action_verifier_config(config_path)
    proof_path = resolve(args.fold_proof, root)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    artifacts = _artifact_paths(config_path, root)
    fold_id, training_ids, heldout_ids = _validate_proof(proof, artifacts)

    reliability_config = load_siglip2_region_reliability_config(
        resolve(config.frozen.reliability_config, root)
    )
    dataset, collator = _paired_dataset(reliability_config, root, "train")
    base = _base_paired(dataset)
    regeneration_keys = (
        "artifact_identity",
        "regeneration_authorization_sha256",
        "regeneration_fold_id",
        "regeneration_experiment_id",
    )
    formal_regeneration = {
        key: base.formal.metadata.get(key) for key in regeneration_keys
    }
    expanded_regeneration = {
        key: base.expanded.metadata.get(key) for key in regeneration_keys
    }
    has_regeneration_identity = any(
        value is not None for value in formal_regeneration.values()
    ) or any(value is not None for value in expanded_regeneration.values())
    if has_regeneration_identity:
        if any(value is None for value in formal_regeneration.values()):
            raise ValueError(
                "Formal candidate cache has incomplete regeneration identity."
            )
        if formal_regeneration != expanded_regeneration:
            raise ValueError(
                "Formal and expanded candidate-cache regeneration identities differ."
            )
    formal_fold = int(base.formal.metadata.get("oof_fold_id", -1))
    expanded_fold = int(base.expanded.metadata.get("oof_fold_id", -1))
    if not bool(base.formal.metadata.get("oof_heldout")) or not bool(
        base.expanded.metadata.get("oof_heldout")
    ):
        raise ValueError("Both fold candidate caches must be marked oof_heldout.")
    if formal_fold != fold_id or expanded_fold != fold_id:
        raise ValueError(
            f"Candidate-cache fold mismatch: proof={fold_id}, "
            f"formal={formal_fold}, expanded={expanded_fold}."
        )
    actual_ids = [
        str((record.get("metadata") or {}).get("record_id", ""))
        for record in base.formal.records
    ]
    if set(actual_ids) != set(heldout_ids):
        raise ValueError("Fold proof held-out ids do not match candidate-cache ids.")
    if args.max_records is not None:
        raise ValueError(
            "Partial feature caches are not valid formal OOF inputs; omit --max-records."
        )

    device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    (
        reliability_model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
        _,
    ) = load_frozen_reliability_chain(reliability_config, root, device)
    reliability_checkpoint = torch.load(
        resolve(config.frozen.reliability_checkpoint, root), map_location="cpu"
    )
    reliability_model.load_state_dict(reliability_checkpoint["model_state_dict"])
    reliability_model.to(device).eval()
    for model in (reliability_model, evidence_model, fine_model, hierarchy):
        model.eval()
    validate_fingerprints(
        base,
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=reliability_config.data.num_workers,
        collate_fn=collator,
    )
    amp_enabled = device.type == "cuda"
    batches = []
    cached_ids: list[str] = []
    with torch.no_grad():
        for raw_batch in tqdm(loader, desc=f"Full-chain OOF fold {fold_id}"):
            paired = move_paired_record_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                context = frozen_layered_action_features(
                    reliability_model,
                    evidence_model,
                    fine_model,
                    hierarchy,
                    paired,
                    decode_options=decode_options(hierarchy_config),
                )
            packed = pack_null_release_context_batch(context, fold_id)
            cached_ids.extend(packed["record_ids"])
            batches.append(packed)
    if set(cached_ids) != set(heldout_ids):
        raise RuntimeError("Materialized feature ids differ from held-out fold ids.")

    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = {
            "format_version": NULL_RELEASE_OOF_FORMAT_VERSION,
            "kind": NULL_RELEASE_OOF_KIND,
            "full_chain_oof": True,
            "fold_id": fold_id,
            "num_folds": 10,
            "records": len(cached_ids),
            "record_ids_sha256": stable_id_digest(cached_ids),
            "training_record_ids": training_ids,
            "heldout_record_ids": heldout_ids,
            "excluded_heldout": True,
            "includes_reliability": True,
            "fold_proof": str(proof_path.resolve()),
            "fold_proof_sha256": sha256_file(proof_path),
            "artifact_sha256": {
                name: sha256_file(path) for name, path in artifacts.items()
            },
    }
    if has_regeneration_identity:
        payload_metadata.update(formal_regeneration)
    payload = {
        "metadata": payload_metadata,
        "batches": batches,
    }
    validate_fold_oof_payload(
        payload,
        expected_fold_id=fold_id,
        expected_record_ids=heldout_ids,
        require_reliability=True,
    )
    torch.save(payload, output)
    print(
        json.dumps(
            {
                "fold_id": fold_id,
                "records": len(cached_ids),
                "batches": len(batches),
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
