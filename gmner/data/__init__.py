"""Data modules for GMNER, exposed through dependency-light lazy imports."""

__all__ = [
    "GMNERCollator",
    "TextGraphBuilder",
    "MMNERJsonDataset",
    "RecordCandidateCollator",
    "HierarchicalRecordCandidateCollator",
    "RecordCandidateDataset",
    "PairedRecordCandidateDataset",
    "PairedRecordCandidateCollator",
    "Siglip2RegionFeatureCache",
    "Siglip2PairedRecordDataset",
    "Siglip2PairedRecordCollator",
    "encode_words_with_alignment",
    "infer_model_input_limit",
    "load_word_aligned_tokenizer",
    "validate_model_input_length",
]


def __getattr__(name):
    if name == "GMNERCollator":
        from .collator import GMNERCollator

        return GMNERCollator
    if name == "TextGraphBuilder":
        from .graph_builders import TextGraphBuilder

        return TextGraphBuilder
    if name == "MMNERJsonDataset":
        from .mmner_dataset import MMNERJsonDataset

        return MMNERJsonDataset
    if name == "RecordCandidateCollator":
        from .record_candidate_collator import RecordCandidateCollator

        return RecordCandidateCollator
    if name == "HierarchicalRecordCandidateCollator":
        from .hierarchical_record_candidate_collator import (
            HierarchicalRecordCandidateCollator,
        )

        return HierarchicalRecordCandidateCollator
    if name == "RecordCandidateDataset":
        from .record_candidate_dataset import RecordCandidateDataset

        return RecordCandidateDataset
    if name in {"PairedRecordCandidateDataset", "PairedRecordCandidateCollator"}:
        from .paired_record_candidate_dataset import (
            PairedRecordCandidateCollator,
            PairedRecordCandidateDataset,
        )

        return {
            "PairedRecordCandidateDataset": PairedRecordCandidateDataset,
            "PairedRecordCandidateCollator": PairedRecordCandidateCollator,
        }[name]
    if name in {
        "Siglip2RegionFeatureCache",
        "Siglip2PairedRecordDataset",
        "Siglip2PairedRecordCollator",
    }:
        from .siglip2_region_cache import (
            Siglip2PairedRecordCollator,
            Siglip2PairedRecordDataset,
            Siglip2RegionFeatureCache,
        )

        return {
            "Siglip2RegionFeatureCache": Siglip2RegionFeatureCache,
            "Siglip2PairedRecordDataset": Siglip2PairedRecordDataset,
            "Siglip2PairedRecordCollator": Siglip2PairedRecordCollator,
        }[name]
    if name in {
        "encode_words_with_alignment",
        "infer_model_input_limit",
        "load_word_aligned_tokenizer",
        "validate_model_input_length",
    }:
        from .tokenization import (
            encode_words_with_alignment,
            infer_model_input_limit,
            load_word_aligned_tokenizer,
            validate_model_input_length,
        )

        return {
            "encode_words_with_alignment": encode_words_with_alignment,
            "infer_model_input_limit": infer_model_input_limit,
            "load_word_aligned_tokenizer": load_word_aligned_tokenizer,
            "validate_model_input_length": validate_model_input_length,
        }[name]
    raise AttributeError(name)
