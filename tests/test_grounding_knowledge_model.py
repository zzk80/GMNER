import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from gmner.data.mmner_dataset import MMNERJsonDataset
from gmner.data.graph_builders import GraphBuilderConfig, TextGraphBuilder


class DummyEncoding(dict):
    def __init__(self, tokens):
        super().__init__(
            {
                "input_ids": list(range(len(tokens) + 2)),
                "attention_mask": [1] * (len(tokens) + 2),
            }
        )
        self._word_ids = [None] + list(range(len(tokens))) + [None]

    def word_ids(self):
        return self._word_ids


class DummyTokenizer:
    is_fast = True

    def __call__(self, tokens, **kwargs):
        return DummyEncoding(tokens)


def test_dataset_attaches_groundability_prior(tmp_path: Path):
    data_path = tmp_path / "train.jsonl"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    prior_dir = tmp_path / "grounding"
    prior_dir.mkdir()

    data_path.write_text(
        json.dumps(
            {
                "id": 0,
                "tokens": ["White", "House"],
                "ner_tags": [3, 4],
                "image": "img.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (prior_dir / "groundability_by_type.jsonl").write_text(
        json.dumps({"entity_type": "LOC", "null_prior": 0.9}) + "\n",
        encoding="utf-8",
    )
    (prior_dir / "groundability_by_mention_type.jsonl").write_text(
        json.dumps({"mention": "white house", "entity_type": "LOC", "null_prior": 0.2}) + "\n",
        encoding="utf-8",
    )

    dataset = MMNERJsonDataset(
        jsonl_path=str(data_path),
        image_dir=str(image_dir),
        tokenizer=DummyTokenizer(),
        graph_builder=TextGraphBuilder(GraphBuilderConfig()),
        grounding_enabled=True,
        groundability_type_priors=str(prior_dir / "groundability_by_type.jsonl"),
        groundability_mention_priors=str(prior_dir / "groundability_by_mention_type.jsonl"),
    )

    sample = dataset[0]
    assert sample["target_entity_type"] == "LOC"
    assert sample["target_type_id"] == 0
    assert sample["grounding_null_prior"] == 0.2
