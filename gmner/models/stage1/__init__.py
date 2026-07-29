"""S3 record-level wrappers around the frozen legacy Stage1."""

from .legacy_stage1_wrapper import LegacyStage1RecordWrapper
from .record_grounding import (
    apply_record_grounding_knowledge,
    vectorized_legacy_grounding,
)

__all__ = [
    "LegacyStage1RecordWrapper",
    "apply_record_grounding_knowledge",
    "vectorized_legacy_grounding",
]
