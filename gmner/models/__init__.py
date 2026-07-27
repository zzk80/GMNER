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
    "Siglip2RegionReliabilityHeadConfig",
    "Siglip2RegionReliabilityHead",
    "LayeredActionVerifierConfig",
    "LayeredActionVerifier",
    "NullReleaseVerifier",
    "SameTypeRegionResolverConfig",
    "ConditionalSameTypeRegionResolver",
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
        "Siglip2RegionReliabilityHeadConfig",
        "Siglip2RegionReliabilityHead",
    }:
        from .siglip2_region_reliability import (
            Siglip2RegionReliabilityHead,
            Siglip2RegionReliabilityHeadConfig,
        )

        return {
            "Siglip2RegionReliabilityHeadConfig": (
                Siglip2RegionReliabilityHeadConfig
            ),
            "Siglip2RegionReliabilityHead": Siglip2RegionReliabilityHead,
        }[name]
    if name in {"LayeredActionVerifierConfig", "LayeredActionVerifier"}:
        from .layered_action_verifier import (
            LayeredActionVerifier,
            LayeredActionVerifierConfig,
        )

        return {
            "LayeredActionVerifierConfig": LayeredActionVerifierConfig,
            "LayeredActionVerifier": LayeredActionVerifier,
        }[name]
    if name == "NullReleaseVerifier":
        from .null_release_verifier import NullReleaseVerifier

        return NullReleaseVerifier
    if name in {
        "SameTypeRegionResolverConfig",
        "ConditionalSameTypeRegionResolver",
    }:
        from .same_type_region_resolver import (
            ConditionalSameTypeRegionResolver,
            SameTypeRegionResolverConfig,
        )

        return {
            "SameTypeRegionResolverConfig": (
                SameTypeRegionResolverConfig
            ),
            "ConditionalSameTypeRegionResolver": (
                ConditionalSameTypeRegionResolver
            ),
        }[name]
    raise AttributeError(name)
