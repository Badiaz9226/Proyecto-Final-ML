"""Transformaciones pequenas y serializables usadas por el pipeline."""

from __future__ import annotations

from typing import Any


def split_pipe_tokens(value: Any) -> list[str]:
    """Separate normalized multi-value fields stored with a pipe delimiter."""

    return [token.strip() for token in str(value).split("|") if token.strip()]

