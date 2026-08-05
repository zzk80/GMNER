"""Loss modules for GMNER."""

from .multitask import (
    alignment_objective,
    base_top1_hard_negative_margin_loss,
    hard_negative_margin_loss,
    iou_aware_region_ranking_loss,
    joint_multi_positive_loss,
    joint_structured_margin_loss,
    joint_teacher_kl_loss,
    joint_visibility_loss,
    masked_cross_entropy,
    multi_positive_region_loss,
    weighted_masked_cross_entropy,
)
from .hierarchical_record_candidate_loss import hierarchical_record_candidate_loss
from .coarse_region_selector_loss import (
    coarse_region_selector_loss,
    coarse_selector_supervision,
)
from .fine_grounding_adapter_loss import (
    fine_grounding_adapter_loss,
    fine_grounding_supervision,
)
from .evidence_visibility_loss import (
    evidence_visibility_loss,
    evidence_visibility_supervision,
)
from .protected_region_mner_loss import (
    boundary_log_probabilities,
    boundary_preservation_kl,
    protected_gate_penalty,
    protected_region_residual_l2,
)

__all__ = [
    "alignment_objective",
    "base_top1_hard_negative_margin_loss",
    "hard_negative_margin_loss",
    "iou_aware_region_ranking_loss",
    "joint_multi_positive_loss",
    "joint_structured_margin_loss",
    "joint_teacher_kl_loss",
    "joint_visibility_loss",
    "masked_cross_entropy",
    "multi_positive_region_loss",
    "hierarchical_record_candidate_loss",
    "coarse_region_selector_loss",
    "coarse_selector_supervision",
    "fine_grounding_adapter_loss",
    "fine_grounding_supervision",
    "evidence_visibility_loss",
    "evidence_visibility_supervision",
    "weighted_masked_cross_entropy",
    "boundary_log_probabilities",
    "boundary_preservation_kl",
    "protected_gate_penalty",
    "protected_region_residual_l2",
]
