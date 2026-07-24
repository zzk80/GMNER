"""Hugging Face Hub runtime defaults."""

from __future__ import annotations

import os


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def configure_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
