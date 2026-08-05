"""Configuration dataclasses and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    task_type: str = "multimodal_ner"
    dataset_name: str = "twitter10000"
    train_file: str = "data/mmner/train.jsonl"
    dev_file: str = "data/mmner/dev.jsonl"
    test_file: str = "data/mmner/test.jsonl"
    image_dir: str = "data/mmner/images"
    image_feature_dir: str = "data/mmner/vinvl"
    image_annotation_dir: str = "data/mmner/xml"
    max_length: int = 192
    num_workers: int = 0
    use_dependency_graph: bool = False
    dependency_backend: str = "spacy"
    dependency_model: str = "en_core_web_sm"
    graph_window_size: int = 2
    max_regions: int = 16
    grounding_iou_threshold: float = 0.5
    add_null_region: bool = True
    grounding_enabled: bool = True
    expand_entities_for_grounding: bool = True
    semantic_prototype_path: str = "knowledge/semantic/semantic_prototypes.pt"
    external_knowledge_prototype_path: str = (
        "knowledge/external/subtype_prototypes.pt"
    )
    groundability_type_priors: str = "knowledge/grounding/groundability_by_type.jsonl"
    groundability_mention_priors: str = "knowledge/grounding/groundability_by_mention_type.jsonl"
    region_min_score: float = 0.0
    label_schema: str = "coarse"
    subtype_taxonomy: str = ""
    subtype_taxonomy_sha256: str = ""
    frozen_clip_feature_dir: str = ""
    frozen_clip_cache_kind: str = "dvh_frozen_clip_cache"


@dataclass
class ModelConfig:
    text_model_name: str = "bert-base-multilingual-cased"
    hidden_size: int = 768
    projection_dim: int = 768
    dropout: float = 0.1
    graph_layers: int = 2
    graph_dropout: float = 0.1
    cross_attention_heads: int = 8
    use_crf: bool = False
    region_feature_dim: int = 2048
    grounding_null_prior_weight: float = 1.0
    grounding_null_logit_bias: float = 0.0
    region_score_prior_weight: float = 0.1
    region_object_compatibility_weight: float = 0.0
    use_grounding_reranker: bool = False
    grounding_reranker_weight: float = 0.1
    grounding_reranker_confidence_threshold: float = 0.85
    grounding_reranker_confidence_floor: float = 0.50
    grounding_reranker_max_delta: float = 0.50
    grounding_reranker_object_vocab_size: int = 2048
    grounding_reranker_attr_vocab_size: int = 4096
    grounding_reranker_label_dim: int = 64
    grounding_reranker_type_dim: int = 64
    grounding_reranker_rank_dim: int = 16
    grounding_reranker_base_temperature: float = 1.0
    grounding_reranker_temperature: float = 1.0
    grounding_reranker_use_uncertainty_gate: bool = True
    grounding_reranker_center_logits: bool = True
    grounding_reranker_gate_min: float = 0.0
    grounding_reranker_gate_max: float = 1.0
    grounding_reranker_use_null_visibility: bool = True
    grounding_reranker_fusion_mode: str = "gated"
    grounding_reranker_null_logit_bias: float = 0.0
    grounding_reranker_use_bilinear: bool = True
    grounding_reranker_use_label_features: bool = True
    grounding_reranker_use_score_features: bool = True
    grounding_reranker_use_rank_features: bool = True
    use_grounding_residual_adapter: bool = False
    grounding_adapter_max_delta: float = 0.5
    use_multiscale_grounding: bool = False
    multiscale_projection_dim: int = 256
    multiscale_local_temperature: float = 0.1
    multiscale_global_temperature: float = 0.07
    multiscale_token_pool_temperature: float = 0.1
    multiscale_grounding_logit_weight: float = 0.0
    multiscale_grounding_delta_max: float = 1.0
    multiscale_residual_initial_scale: float = 0.0
    multiscale_residual_scale_max: float = 1.0
    use_entity_evidence_decoder: bool = False
    evidence_decoder_layers: int = 1
    evidence_decoder_heads: int = 4
    evidence_region_logit_weight: float = 0.5
    evidence_joint_region_weight: float = 0.2
    evidence_delta_max: float = 1.0
    evidence_pair_score_max: float = 5.0
    evidence_use_type_for_eval: bool = True
    use_subtype_auxiliary: bool = False
    num_subtypes: int = 0
    subtype_contrastive_temperature: float = 0.1
    use_fine_subtype_head: bool = False
    fine_subtype_hidden_size: int = 768
    fine_subtype_head_architecture: str = "shared_hard"
    fine_subtype_parent_hidden_size: int = 192
    fine_subtype_input_source: str = "text_only"
    use_semantic_prototypes: bool = False
    prototype_type_score_weight: float = 1.0
    prototype_subtype_score_weight: float = 1.0
    prototype_retrieval_temperature: float = 0.1
    prototype_reliability_margin: float = 0.1
    prototype_reliability_score: float = 0.2
    prototype_reliability_temperature: float = 0.1
    prototype_span_source: str = "gold"
    prototype_type_temperature: float = 1.0
    prototype_gate_mode: str = "entropy"
    prototype_constant_gate: float = 0.2
    prototype_max_gate: float = 1.0
    prototype_type_refinement_weight: float = 0.0
    prototype_writeback_to_tokens: bool = True
    prototype_type_prior_weight: float = 0.0
    prototype_type_prior_detach: bool = True
    use_alignment_preserving_prototype_grounding: bool = False
    prototype_grounding_delta_weight: float = 0.2
    prototype_grounding_delta_max: float = 0.5
    prototype_grounding_preservation_margin: float = 0.0
    use_external_knowledge: bool = False
    external_knowledge_temperature: float = 0.1
    external_knowledge_query_dropout: float = 0.1
    external_knowledge_type_prior_weight: float = 0.0
    external_knowledge_type_prior_max_delta: float = 1.0
    external_knowledge_type_prior_detach: bool = False
    external_knowledge_fusion_mode: str = "fixed"
    external_knowledge_arbiter_hidden_size: int = 32
    external_knowledge_arbiter_dropout: float = 0.1
    external_knowledge_arbiter_initial_gate: float = 0.05
    external_knowledge_arbiter_strength: float = 1.0
    external_knowledge_arbiter_base_temperature: float = 1.0
    external_knowledge_arbiter_knowledge_temperature: float = 1.0
    external_knowledge_arbiter_detach_base: bool = True
    external_knowledge_arbiter_inference_threshold: float = 0.0
    use_joint_type_region_verifier: bool = False
    joint_verifier_hidden_size: int = 256
    joint_verifier_dropout: float = 0.1
    joint_verifier_type_temperature: float = 1.0
    joint_verifier_region_temperature: float = 1.0
    joint_verifier_base_type_weight: float = 1.0
    joint_verifier_base_region_weight: float = 1.0
    joint_verifier_interaction_weight: float = 1.0
    joint_verifier_visibility_weight: float = 1.0
    joint_verifier_interaction_logit_max: float = 0.0
    joint_verifier_visibility_logit_max: float = 0.0
    joint_verifier_hierarchical_visibility: bool = False
    joint_verifier_top_m_types: int = 4
    joint_verifier_top_r_regions: int = 0
    joint_span_perturbation_probability: float = 0.0
    joint_span_perturbation_max_words: int = 1
    dvh_enabled: bool = False
    dvh_use_clip: bool = True
    dvh_use_vinvl: bool = True
    dvh_shuffle_clip: bool = False
    dvh_clip_feature_dim: int = 768
    dvh_clip_patch_grid_size: int = 7
    dvh_type_query_count: int = 4
    dvh_gate_initial_bias: float = -2.0
    dvh_boundary_visual: bool = True
    dvh_type_visual: bool = True
    dvh_grounding_visual: bool = True
    tq_enabled: bool = False
    tq_use_clip: bool = True
    tq_use_vinvl: bool = True
    tq_shuffle_clip: bool = False
    tq_visual_dim: int = 256
    tq_clip_feature_dim: int = 768
    tq_type_count: int = 4
    tq_gate_initial_bias: float = -2.0
    tq_max_span_length: int = 10
    tq_decode_top_k_per_type: int = 32
    tq_existence_threshold: float = 0.5
    tq_span_score_threshold: float = 0.0
    tq_existence_score_weight: float = 0.5
    tq_visual_warmup_epochs: int = 3
    use_protected_region_mner: bool = False
    protected_region_bottleneck_size: int = 512
    protected_region_gate_hidden_size: int = 128
    protected_region_dropout: float = 0.1
    protected_mner_attention_heads: int = 8
    protected_mner_attention_dropout: float = 0.1
    protected_mner_gate_hidden_size: int = 128
    protected_mner_gate_max: float = 0.3
    protected_mner_exclude_null: bool = True
    protected_mner_grounding_use_refined_text: bool = False
    protected_mner_grounding_use_refined_regions: bool = False


@dataclass
class OptimConfig:
    batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    new_module_learning_rate: float = 1e-4
    high_level_learning_rate: float = 1e-5
    backbone_learning_rate: float = 3e-6
    gradual_unfreeze_enabled: bool = False
    gradual_unfreeze_high_epoch: int = 3
    gradual_unfreeze_bert_epoch: int = 6
    bert_unfreeze_last_n_layers: int = 4
    subtype_learning_rate: float = 1e-4
    backbone_lower_learning_rate: float = 1e-6
    backbone_upper_learning_rate: float = 5e-6
    backbone_upper_layer_count: int = 4


@dataclass
class LossConfig:
    label_smoothing: float = 0.0
    lambda_ner: float = 1.0
    lambda_grounding: float = 1.0
    lambda_alignment: float = 0.1
    lambda_type_prototype: float = 0.1
    lambda_subtype_prototype: float = 0.05
    lambda_subtype_auxiliary: float = 0.0
    lambda_subtype_contrastive: float = 0.0
    lambda_fine_subtype: float = 0.0
    lambda_external_knowledge_type: float = 0.0
    lambda_external_knowledge_subtype: float = 0.0
    lambda_external_knowledge_arbiter: float = 0.0
    lambda_external_knowledge_fusion: float = 0.0
    external_knowledge_arbiter_positive_weight: float = 1.0
    lambda_grounding_preservation: float = 0.0
    lambda_grounding_hard_negative: float = 0.0
    grounding_hard_negative_margin: float = 0.2
    lambda_grounding_multi_positive: float = 0.0
    lambda_token_region_contrastive: float = 0.0
    lambda_span_region_contrastive: float = 0.0
    lambda_sentence_image_contrastive: float = 0.0
    lambda_iou_ranking: float = 0.0
    iou_ranking_margin: float = 0.2
    iou_ranking_min_gap: float = 0.1
    iou_ranking_score_source: str = "grounding"
    multiscale_visible_sample_weight: float = 1.0
    multiscale_null_sample_weight: float = 1.0
    lambda_grounding_reranker_aux: float = 0.0
    grounding_base_error_positive_weight: float = 1.0
    grounding_base_correct_weight: float = 1.0
    grounding_base_default_weight: float = 1.0
    grounding_type_confidence_threshold: float = 0.0
    lambda_base_top1_hard_negative: float = 0.0
    lambda_evidence_type: float = 0.0
    lambda_evidence_joint: float = 0.0
    lambda_joint_type_region: float = 0.0
    lambda_joint_visibility: float = 0.0
    lambda_joint_hard_negative: float = 0.0
    joint_hard_negative_margin: float = 0.2
    lambda_joint_preserve: float = 0.0
    joint_preserve_margin_threshold: float = 1.0
    joint_preserve_evidence_threshold: float = 0.1
    lambda_joint_representation: float = 0.0
    joint_visible_sample_weight: float = 1.0
    joint_null_sample_weight: float = 1.0
    lambda_boundary: float = 1.0
    lambda_type: float = 1.0
    lambda_gate_regularization: float = 0.01
    lambda_tq_existence: float = 0.5
    lambda_tq_start: float = 1.0
    lambda_tq_end: float = 1.0
    lambda_tq_span_match: float = 1.0
    lambda_tq_visual_alignment: float = 0.1
    lambda_tq_gate_regularization: float = 0.01
    tq_start_positive_weight: float = 5.0
    tq_end_positive_weight: float = 5.0
    tq_span_positive_weight: float = 20.0
    lambda_protected_boundary_preserve: float = 0.0
    lambda_protected_visual_type: float = 0.0
    lambda_protected_visual_gate: float = 0.0
    lambda_protected_region_residual: float = 0.0


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = "outputs/gmner_twitter10000"
    init_checkpoint: str = ""
    trainable_modules: str = "all"
    save_best_metric: str = "gmner_score"
    save_latest_checkpoint: bool = False
    early_stopping_patience: int = 0
    eval_frozen_modules: bool = False
    log_every_steps: int = 20
    log_grad_norms: bool = False
    task_gradient_cosine_interval: int = 0


@dataclass
class GMNERConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _update_dataclass(instance: Any, updates: Dict[str, Any]) -> Any:
    annotations = getattr(instance, "__annotations__", {})
    for key, value in updates.items():
        if hasattr(instance, key):
            expected_type = annotations.get(key)
            setattr(instance, key, _coerce_value(value, expected_type))
    return instance


def _coerce_value(value: Any, expected_type: Any) -> Any:
    if expected_type is None or value is None:
        return value

    if isinstance(expected_type, str):
        type_name = expected_type.strip().lower()
        if type_name in {"bool", "builtins.bool"}:
            expected_type = bool
        elif type_name in {"int", "builtins.int"}:
            expected_type = int
        elif type_name in {"float", "builtins.float"}:
            expected_type = float
        elif type_name in {"str", "builtins.str"}:
            expected_type = str

    try:
        if expected_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y", "on"}:
                    return True
                if lowered in {"false", "0", "no", "n", "off"}:
                    return False
            return bool(value)

        if expected_type is int:
            if isinstance(value, int):
                return value
            return int(float(value))

        if expected_type is float:
            if isinstance(value, float):
                return value
            return float(value)

        if expected_type is str:
            return str(value)

    except Exception:
        return value

    return value


def load_config(config_path: str | Path) -> GMNERConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fp:
        raw_cfg = yaml.safe_load(fp) or {}

    config = GMNERConfig()
    if "data" in raw_cfg:
        _update_dataclass(config.data, raw_cfg["data"])
    if "model" in raw_cfg:
        _update_dataclass(config.model, raw_cfg["model"])
    if "optim" in raw_cfg:
        _update_dataclass(config.optim, raw_cfg["optim"])
    if "loss" in raw_cfg:
        _update_dataclass(config.loss, raw_cfg["loss"])
    if "runtime" in raw_cfg:
        _update_dataclass(config.runtime, raw_cfg["runtime"])

    return config


def dump_config(config: GMNERConfig, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": vars(config.data),
        "model": vars(config.model),
        "optim": vars(config.optim),
        "loss": vars(config.loss),
        "runtime": vars(config.runtime),
    }
    with save_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False)
