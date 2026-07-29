"""Read-only diagnostics for GMNER experiments."""

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
    "compute_gradient_observation",
    "encoder_layer_parameter_groups",
    "stable_probe_record_ids",
]
