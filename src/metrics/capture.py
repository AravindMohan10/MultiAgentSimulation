"""Capture outcome metrics."""

from __future__ import annotations

from typing import Optional


def capture_rate(result: str) -> float:
    return 1.0 if result == "hero_captured" else 0.0


def capture_time(result: str, capture_step_index: Optional[int]) -> Optional[int]:
    if result == "hero_captured" and capture_step_index is not None:
        return int(capture_step_index)
    return None
