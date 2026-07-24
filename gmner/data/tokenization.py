"""Tokenizer-independent word to subword alignment."""

from __future__ import annotations

from typing import Any, Optional, Sequence


def load_word_aligned_tokenizer(model_name: str, **kwargs: Any) -> Any:
    """Load a tokenizer configured for pre-tokenized word sequences."""

    from transformers import AutoTokenizer

    load_kwargs = dict(kwargs)
    load_kwargs["use_fast"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)

    # Byte-level BPE fast tokenizers such as RoBERTa must be initialized this
    # way before ``is_split_into_words=True`` can be used.
    if bool(getattr(tokenizer, "is_fast", False)) and (
        getattr(tokenizer, "add_prefix_space", None) is False
    ):
        load_kwargs["add_prefix_space"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)

    return tokenizer


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _slow_tokenize_words(
    tokenizer: Any,
    words: Sequence[str],
    max_length: int,
) -> tuple[dict[str, list[int]], list[Optional[int]]]:
    """Tokenize pre-split words with a slow tokenizer and retain alignment."""

    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_budget = int(max_length) - special_count
    if content_budget < 1:
        raise ValueError(
            f"max_length={max_length} leaves no content tokens after "
            f"{special_count} special tokens."
        )

    content_ids: list[int] = []
    content_word_ids: list[int] = []
    for word_index, word in enumerate(words):
        pieces = tokenizer.tokenize(str(word))
        if pieces:
            piece_ids = tokenizer.convert_tokens_to_ids(pieces)
            if isinstance(piece_ids, int):
                piece_ids = [piece_ids]
            else:
                piece_ids = _as_list(piece_ids)
        else:
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            if unk_token_id is None:
                raise ValueError(
                    f"Tokenizer produced no pieces for word {word!r} and has no unk token."
                )
            piece_ids = [int(unk_token_id)]

        remaining = content_budget - len(content_ids)
        if remaining <= 0:
            break
        selected_ids = piece_ids[:remaining]
        content_ids.extend(int(piece_id) for piece_id in selected_ids)
        content_word_ids.extend([word_index] * len(selected_ids))
        if len(selected_ids) < len(piece_ids):
            break

    model_input_names = set(getattr(tokenizer, "model_input_names", []))
    prepared = tokenizer.prepare_for_model(
        content_ids,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids="token_type_ids" in model_input_names,
        return_special_tokens_mask=True,
    )
    special_mask = _as_list(prepared.pop("special_tokens_mask"))
    encoding = {key: _as_list(value) for key, value in prepared.items()}
    input_ids = encoding["input_ids"]
    if len(input_ids) != len(special_mask):
        raise ValueError("Tokenizer returned inconsistent special-token metadata.")

    word_ids: list[Optional[int]] = []
    content_index = 0
    for is_special in special_mask:
        if is_special:
            word_ids.append(None)
        else:
            if content_index >= len(content_word_ids):
                raise ValueError("Tokenizer inserted an untracked non-special token.")
            word_ids.append(content_word_ids[content_index])
            content_index += 1

    if content_index != len(content_word_ids):
        raise ValueError("Tokenizer dropped content tokens while adding special tokens.")
    if len(input_ids) > max_length:
        raise ValueError(
            f"Aligned encoding length {len(input_ids)} exceeds max_length={max_length}."
        )
    return encoding, word_ids


def encode_words_with_alignment(
    tokenizer: Any,
    words: Sequence[str],
    max_length: int,
) -> tuple[dict[str, list[int]], list[Optional[int]]]:
    """Encode pre-split words and return input fields plus source word ids."""

    if bool(getattr(tokenizer, "is_fast", False)):
        batch_encoding = tokenizer(
            list(words),
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        word_ids = list(batch_encoding.word_ids())
        encoding = {key: _as_list(value) for key, value in batch_encoding.items()}
        if len(encoding["input_ids"]) != len(word_ids):
            raise ValueError("Fast tokenizer returned inconsistent word alignment.")
        return encoding, word_ids

    return _slow_tokenize_words(tokenizer, words, max_length)


def infer_model_input_limit(tokenizer: Any, model_config: Any) -> Optional[int]:
    """Infer the usable sequence length without trusting sentinel tokenizer values."""

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        return tokenizer_limit

    position_count = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(position_count, int) or position_count <= 0:
        return None
    if str(getattr(model_config, "model_type", "")).lower() == "roberta":
        padding_offset = int(getattr(model_config, "pad_token_id", 1)) + 1
        return position_count - padding_offset
    return position_count


def validate_model_input_length(
    tokenizer: Any,
    model_config: Any,
    max_length: int,
) -> Optional[int]:
    """Reject a configured length that exceeds the text backbone capacity."""

    model_limit = infer_model_input_limit(tokenizer, model_config)
    if model_limit is not None and int(max_length) > model_limit:
        raise ValueError(
            f"Configured max_length={max_length} exceeds the usable input limit "
            f"{model_limit} for {getattr(model_config, '_name_or_path', 'text backbone')}."
        )
    return model_limit
