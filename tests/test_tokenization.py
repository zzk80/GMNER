from __future__ import annotations

from types import SimpleNamespace

import pytest

from gmner.data.tokenization import (
    encode_words_with_alignment,
    infer_model_input_limit,
    validate_model_input_length,
)


class FakeSlowTokenizer:
    is_fast = False
    unk_token_id = 99
    model_input_names = ["input_ids", "attention_mask"]
    model_max_length = int(1e30)

    def tokenize(self, word):
        pieces = {
            "alpha": ["a", "lpha"],
            "beta": ["beta"],
            "gamma": ["g", "amma"],
            "empty": [],
        }
        return pieces[word]

    def convert_tokens_to_ids(self, pieces):
        vocabulary = {"a": 1, "lpha": 2, "beta": 3, "g": 4, "amma": 5}
        return [vocabulary[piece] for piece in pieces]

    def num_special_tokens_to_add(self, pair=False):
        assert not pair
        return 2

    def prepare_for_model(self, token_ids, **kwargs):
        assert kwargs["return_special_tokens_mask"]
        return {
            "input_ids": [10, *token_ids, 11],
            "attention_mask": [1] * (len(token_ids) + 2),
            "special_tokens_mask": [1, *([0] * len(token_ids)), 1],
        }


class FakeFastEncoding(dict):
    def word_ids(self):
        return [None, 0, 0, 1, None]


class FakeFastTokenizer:
    is_fast = True

    def __call__(self, words, **kwargs):
        assert words == ["alpha", "beta"]
        assert kwargs == {
            "is_split_into_words": True,
            "truncation": True,
            "max_length": 8,
        }
        return FakeFastEncoding(
            input_ids=[10, 1, 2, 3, 11],
            attention_mask=[1, 1, 1, 1, 1],
        )


def test_slow_tokenizer_preserves_word_ids_and_whole_length_budget():
    encoding, word_ids = encode_words_with_alignment(
        FakeSlowTokenizer(),
        ["alpha", "beta", "gamma"],
        max_length=5,
    )

    assert encoding["input_ids"] == [10, 1, 2, 3, 11]
    assert encoding["attention_mask"] == [1, 1, 1, 1, 1]
    assert word_ids == [None, 0, 0, 1, None]


def test_slow_tokenizer_uses_unknown_token_for_empty_word():
    encoding, word_ids = encode_words_with_alignment(
        FakeSlowTokenizer(),
        ["empty"],
        max_length=4,
    )

    assert encoding["input_ids"] == [10, 99, 11]
    assert word_ids == [None, 0, None]


def test_fast_tokenizer_keeps_native_word_alignment():
    encoding, word_ids = encode_words_with_alignment(
        FakeFastTokenizer(),
        ["alpha", "beta"],
        max_length=8,
    )

    assert encoding["input_ids"] == [10, 1, 2, 3, 11]
    assert word_ids == [None, 0, 0, 1, None]


def test_roberta_position_offset_limits_bertweet_to_128():
    tokenizer = SimpleNamespace(model_max_length=int(1e30))
    config = SimpleNamespace(
        model_type="roberta",
        max_position_embeddings=130,
        pad_token_id=1,
        _name_or_path="bertweet-base",
    )

    assert infer_model_input_limit(tokenizer, config) == 128
    assert validate_model_input_length(tokenizer, config, 128) == 128
    with pytest.raises(ValueError, match="max_length=192"):
        validate_model_input_length(tokenizer, config, 192)


def test_explicit_tokenizer_limit_takes_precedence():
    tokenizer = SimpleNamespace(model_max_length=96)
    config = SimpleNamespace(
        model_type="roberta",
        max_position_embeddings=514,
        pad_token_id=1,
    )

    assert infer_model_input_limit(tokenizer, config) == 96
