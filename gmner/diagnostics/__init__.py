"""Read-only diagnostics for GMNER experiments."""

from .s3_audits import (
    audit_boundary_type_errors,
    audit_candidate_actionability,
    audit_truncation,
    ensure_s3_audit_split,
    read_s3_source_records,
)
from .stage1_gradient_conflicts import (
    STAGE1_TASK_PAIRS,
    STAGE1_TASKS,
    aggregate_gradient_observations,
    compute_gradient_observation,
    encoder_layer_parameter_groups,
    stable_probe_record_ids,
)

__all__ = [
    "STAGE1_TASK_PAIRS",
    "STAGE1_TASKS",
    "aggregate_gradient_observations",
    "audit_boundary_type_errors",
    "audit_candidate_actionability",
    "audit_truncation",
    "compute_gradient_observation",
    "encoder_layer_parameter_groups",
    "ensure_s3_audit_split",
    "read_s3_source_records",
    "stable_probe_record_ids",
]
