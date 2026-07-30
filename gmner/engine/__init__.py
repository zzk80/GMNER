"""Training and evaluation engines exposed through lazy imports."""

__all__ = [
    "evaluate_model",
    "GMNERTrainer",
    "evaluate_hierarchical_record_verifier",
    "evaluate_coarse_region_selector",
    "evaluate_fine_grounding_adapter",
    "evaluate_evidence_visibility",
]


def __getattr__(name):
    if name == "evaluate_model":
        from .evaluator import evaluate_model

        return evaluate_model
    if name == "GMNERTrainer":
        from .trainer import GMNERTrainer

        return GMNERTrainer
    if name == "evaluate_hierarchical_record_verifier":
        from .hierarchical_record_verifier_evaluator import (
            evaluate_hierarchical_record_verifier,
        )

        return evaluate_hierarchical_record_verifier
    if name == "evaluate_coarse_region_selector":
        from .coarse_region_selector_evaluator import (
            evaluate_coarse_region_selector,
        )

        return evaluate_coarse_region_selector
    if name == "evaluate_fine_grounding_adapter":
        from .fine_grounding_adapter_evaluator import (
            evaluate_fine_grounding_adapter,
        )

        return evaluate_fine_grounding_adapter
    if name == "evaluate_evidence_visibility":
        from .evidence_visibility_evaluator import (
            evaluate_evidence_visibility,
        )

        return evaluate_evidence_visibility
    raise AttributeError(name)
