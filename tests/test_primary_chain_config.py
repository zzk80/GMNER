from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def test_primary_stage1_uses_roberta_128() -> None:
    config = _load("configs/fmnerg_twitter10000_stage1.yaml")

    assert config["model"]["text_model_name"].endswith("/roberta-base")
    assert config["data"]["max_length"] == 128
    assert config["runtime"]["output_dir"] == "outputs/fmnerg_stage1_roberta128"


def test_hierarchical_primary_chain_uses_isolated_roberta_caches() -> None:
    config = _load("configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml")

    for key in ("train_cache", "dev_cache", "test_cache"):
        assert "/roberta128/" in config["data"][key]
    assert config["runtime"]["output_dir"] == (
        "outputs/fmnerg_roberta128_hierarchical_record_verifier"
    )
    assert config["runtime"]["evaluate_test_after_training"] is False


def test_coarse_selector_is_dev_gated_and_uses_expanded_roberta_caches() -> None:
    config = _load("configs/fmnerg_twitter10000_coarse_selector.yaml")

    assert "/roberta128/" in config["data"]["train_cache"]
    assert "/roberta128/" in config["data"]["dev_cache"]
    assert config["data"]["train_cache"].endswith("_r36.pt")
    assert config["data"]["dev_cache"].endswith("_r36.pt")
    assert "test_cache" not in config["data"]
    assert config["policy"] == {
        "expanded_budget": 36,
        "final_budget": 16,
        "base_keep_values": [8, 10],
    }
    assert config["runtime"]["save_best_metric"] == (
        "union_base8_learned8_recall_eligible"
    )


def test_fine_adapter_freezes_primary_chain_and_has_no_test_cache() -> None:
    config = _load("configs/fmnerg_twitter10000_fine_grounding_adapter.yaml")

    assert config["model"]["final_budget"] == 16
    assert config["model"]["base_keep"] == 8
    assert config["model"]["base_prior_weight"] > config["model"][
        "coarse_prior_weight"
    ]
    assert config["data"]["expanded_train_cache"].endswith("_r36.pt")
    assert config["data"]["expanded_dev_cache"].endswith("_r36.pt")
    assert not any("test" in key for key in config["data"])
    assert config["frozen"]["hierarchical_checkpoint"].endswith(
        "fmnerg_roberta128_hierarchical_record_verifier/best_model.pt"
    )
    assert config["frozen"]["coarse_checkpoint"].endswith(
        "fmnerg_roberta128_coarse_selector/best_model.pt"
    )
    assert config["runtime"]["save_best_metric"] == "gmner_score"
    assert config["runtime"]["save_best_tie_breakers"][0] == (
        "visible_net_correction"
    )


def test_evidence_visibility_freezes_m32_and_is_dev_gated() -> None:
    config = _load("configs/fmnerg_twitter10000_evidence_visibility.yaml")

    assert not any("test" in key for key in config["data"])
    assert config["frozen"]["fine_checkpoint"].endswith(
        "fmnerg_roberta128_fine_grounding_adapter/best_model.pt"
    )
    assert config["model"]["input_size"] == 256
    assert config["runtime"]["save_best_metric"] == "gmner_score"
    assert config["runtime"]["save_best_tie_breakers"][0] == (
        "null_correct_preservation_rate"
    )
