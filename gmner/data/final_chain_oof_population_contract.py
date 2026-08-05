"""Authorization helpers for the complete final-chain OOF population."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FOLD0_KIND = "final_chain_oof_fold0_dry_run_preregistration"
POPULATION_KIND = "final_chain_oof_folds1_9_population_authorization"


@dataclass(frozen=True)
class FinalChainExecutionContract:
    artifact_identity: str
    experiment_id: str
    execution_folds: tuple[int, ...]
    seed: int


def validate_final_chain_authorization(
    payload: dict[str, Any], *, fold_id: int
) -> FinalChainExecutionContract:
    kind = str(payload.get("kind", ""))
    if kind == FOLD0_KIND:
        if payload.get("status") != "AUTHORIZED_NOT_STARTED":
            raise PermissionError("Fold-0 dry-run authorization is not active.")
        authorization = dict(payload.get("authorization") or {})
        if authorization.get("fold0_execution") is not True or int(fold_id) != 0:
            raise PermissionError("Fold-0 authorization cannot execute this fold.")
        execution_folds = (0,)
        artifact_identity = "FINAL_CHAIN_OOF_FOLD0_DRY_RUN"
    elif kind == POPULATION_KIND:
        if payload.get("status") != "AUTHORIZED":
            raise PermissionError("Folds 1-9 population is not authorized.")
        execution_folds = tuple(int(value) for value in payload.get("execution_folds", ()))
        if execution_folds != tuple(range(1, 10)) or int(fold_id) not in execution_folds:
            raise PermissionError("Population authorization is limited to folds 1-9.")
        forbidden = dict(payload.get("forbidden") or {})
        required_locks = (
            "b1_a1_training",
            "auroc_feature_selection",
            "threshold_or_calibration",
            "dev_access",
            "test_access",
        )
        if any(forbidden.get(key) is not True for key in required_locks):
            raise PermissionError("A population training/access lock is not active.")
        artifact_identity = str(payload.get("artifact_identity", ""))
        if artifact_identity != "FINAL_CHAIN_OOF_POPULATION_FOLDS1_9":
            raise ValueError("Unexpected final-chain population identity.")
    else:
        raise ValueError(f"Unsupported final-chain authorization kind: {kind}")

    source = dict(payload.get("source_contract") or {})
    if int(source.get("seed", -1)) != 42 or int(source.get("num_folds", -1)) != 10:
        raise ValueError("Final-chain fold/seed contract changed.")
    if source.get("official_dev_access") is not False or source.get("test_access") is not False:
        raise PermissionError("Official Dev/Test must remain disabled.")
    chain = dict(payload.get("chain_contract") or {})
    if (
        chain.get("identity") != "M3.3A_FORMAL_BEST_CHAIN"
        or chain.get("siglip2") is not False
        or chain.get("fusion_reliability") is not False
        or chain.get("null_release") is not False
        or chain.get("clip", False) is not False
        or chain.get("fmnerg_subtype", False) is not False
    ):
        raise PermissionError("Only the formal M3.3A chain is authorized.")
    if kind == POPULATION_KIND:
        allowed = dict(payload.get("allowed") or {})
        required_outputs = (
            "generate_gold_free_rows",
            "deterministic_replay",
            "postseal_supervision_sidecar",
            "descriptive_distribution_summary",
            "sequential_fold_cleanup",
        )
        if any(allowed.get(key) is not True for key in required_outputs):
            raise PermissionError("The folds 1-9 output contract is incomplete.")
    return FinalChainExecutionContract(
        artifact_identity=artifact_identity,
        experiment_id=str(payload["experiment_id"]),
        execution_folds=execution_folds,
        seed=int(source["seed"]),
    )


def regeneration_metadata_for_contract(
    contract: FinalChainExecutionContract,
    *,
    authorization_sha256: str,
    fold_id: int,
) -> dict[str, Any]:
    if int(fold_id) not in contract.execution_folds:
        raise PermissionError("Fold is outside the authorized execution set.")
    if len(str(authorization_sha256)) != 64:
        raise ValueError("Invalid authorization SHA256.")
    return {
        "artifact_identity": contract.artifact_identity,
        "regeneration_authorization_sha256": str(authorization_sha256),
        "regeneration_fold_id": int(fold_id),
        "regeneration_experiment_id": contract.experiment_id,
        "execution_folds": list(contract.execution_folds),
    }


def validate_dynamic_regeneration_metadata(
    metadata: dict[str, Any],
    *,
    artifact_identity: str,
    authorization_sha256: str,
    fold_id: int,
    experiment_id: str,
) -> None:
    expected = {
        "artifact_identity": str(artifact_identity),
        "regeneration_authorization_sha256": str(authorization_sha256),
        "regeneration_fold_id": int(fold_id),
        "regeneration_experiment_id": str(experiment_id),
    }
    observed = {key: metadata.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"Regeneration identity differs: {observed} != {expected}.")
