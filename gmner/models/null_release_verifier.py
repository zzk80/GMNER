"""High-precision NULL-to-visible release verifier."""

from __future__ import annotations

from .layered_action_verifier import (
    ACTION_MODE_NULL_RELEASE_ONLY,
    LayeredActionVerifier,
    LayeredActionVerifierConfig,
)


class NullReleaseVerifier(LayeredActionVerifier):
    """Expose the narrowed M3.6A-r2 policy as a dedicated model."""

    def __init__(self, config: LayeredActionVerifierConfig) -> None:
        if config.action_mode != ACTION_MODE_NULL_RELEASE_ONLY:
            raise ValueError(
                "NullReleaseVerifier requires action_mode='null_release_only'."
            )
        super().__init__(config)
