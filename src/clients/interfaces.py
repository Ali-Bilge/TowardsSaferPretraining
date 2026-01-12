"""
Lightweight interfaces for safety clients/adapters.

These are intentionally minimal for a research reproduction repo:
- keep typing helpful without forcing large abstractions
- standardize the common `predict(text) -> HarmLabel` shape used by baselines
"""

from __future__ import annotations

from typing import Protocol

from ..utils.taxonomy import HarmLabel


class TextSafetyClient(Protocol):
    """Adapter that can classify a single text into the repo's HarmLabel taxonomy."""

    def predict(self, text: str) -> HarmLabel:  # pragma: no cover
        ...

