"""Model modules for GMNER."""

__all__ = [
    "GMNERModel",
    "JointEntityAdapter",
    "JointTypeRegionVerifier",
    "HierarchicalRecordVerifierConfig",
    "HierarchicalRecordVerifier",
    "CoarseRegionSelectorConfig",
    "RecallPreservingCoarseSelector",
    "FineGroundingAdapterConfig",
    "CorrectionPreservationGroundingAdapter",
    "EvidenceVisibilityHeadConfig",
    "RegionEvidenceVisibilityHead",
    "TypedBIOVisualResidualConfig",
    "TypedBIOVisualResidual",
    "ProtectedTypedBIOVisualStage1",
]


def __getattr__(name):
    if name == "GMNERModel":
        from .gmner_model import GMNERModel

        return GMNERModel
    if name in {"JointEntityAdapter", "JointTypeRegionVerifier"}:
        from .joint_type_region_verifier import JointEntityAdapter, JointTypeRegionVerifier

        return {
            "JointEntityAdapter": JointEntityAdapter,
            "JointTypeRegionVerifier": JointTypeRegionVerifier,
        }[name]
    if name in {"HierarchicalRecordVerifierConfig", "HierarchicalRecordVerifier"}:
        from .hierarchical_record_verifier import (
            HierarchicalRecordVerifier,
            HierarchicalRecordVerifierConfig,
        )

        return {
            "HierarchicalRecordVerifierConfig": HierarchicalRecordVerifierConfig,
            "HierarchicalRecordVerifier": HierarchicalRecordVerifier,
        }[name]
    if name in {"CoarseRegionSelectorConfig", "RecallPreservingCoarseSelector"}:
        from .coarse_region_selector import (
            CoarseRegionSelectorConfig,
            RecallPreservingCoarseSelector,
        )

        return {
            "CoarseRegionSelectorConfig": CoarseRegionSelectorConfig,
            "RecallPreservingCoarseSelector": RecallPreservingCoarseSelector,
        }[name]
    if name in {
        "FineGroundingAdapterConfig",
        "CorrectionPreservationGroundingAdapter",
    }:
        from .fine_grounding_adapter import (
            CorrectionPreservationGroundingAdapter,
            FineGroundingAdapterConfig,
        )

        return {
            "FineGroundingAdapterConfig": FineGroundingAdapterConfig,
            "CorrectionPreservationGroundingAdapter": (
                CorrectionPreservationGroundingAdapter
            ),
        }[name]
    if name in {
        "EvidenceVisibilityHeadConfig",
        "RegionEvidenceVisibilityHead",
    }:
        from .evidence_visibility import (
            EvidenceVisibilityHeadConfig,
            RegionEvidenceVisibilityHead,
        )

        return {
            "EvidenceVisibilityHeadConfig": EvidenceVisibilityHeadConfig,
            "RegionEvidenceVisibilityHead": RegionEvidenceVisibilityHead,
        }[name]
    if name in {
        "TypedBIOVisualResidualConfig",
        "TypedBIOVisualResidual",
        "ProtectedTypedBIOVisualStage1",
    }:
        from .typed_bio_visual_residual import (
            ProtectedTypedBIOVisualStage1,
            TypedBIOVisualResidual,
            TypedBIOVisualResidualConfig,
        )

        return {
            "TypedBIOVisualResidualConfig": TypedBIOVisualResidualConfig,
            "TypedBIOVisualResidual": TypedBIOVisualResidual,
            "ProtectedTypedBIOVisualStage1": ProtectedTypedBIOVisualStage1,
        }[name]
    raise AttributeError(name)
