"""Evaluate the NULL-to-visible release verifier on dev only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.models.layered_action_verifier import ACTION_MODE_NULL_RELEASE_ONLY
from scripts.evaluate_layered_action_verifier import main


if __name__ == "__main__":
    main(required_action_mode=ACTION_MODE_NULL_RELEASE_ONLY)
