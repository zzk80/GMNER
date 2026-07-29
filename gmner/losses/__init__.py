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
from .siglip2_region_reliability_loss import (
    siglip2_region_reliability_loss,
    siglip2_region_reliability_supervision,
)
from .layered_action_verifier_loss import (
    layered_action_supervision,
    layered_action_verifier_loss,
)
from .same_type_region_resolver_loss import (
    same_type_region_resolver_loss,
    same_type_region_supervision,
)
from .stage1_candidate_selector_loss import (
    stage1_candidate_selector_loss,
    stage1_candidate_selector_supervision,
)
from .s3_stage1_loss import (
    S3LossWeights,
    compute_s3_stage1_losses,
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
    "siglip2_region_reliability_loss",
    "siglip2_region_reliability_supervision",
    "layered_action_verifier_loss",
    "layered_action_supervision",
    "same_type_region_resolver_loss",
    "same_type_region_supervision",
    "stage1_candidate_selector_loss",
    "stage1_candidate_selector_supervision",
    "S3LossWeights",
    "compute_s3_stage1_losses",
    "weighted_masked_cross_entropy",
]
