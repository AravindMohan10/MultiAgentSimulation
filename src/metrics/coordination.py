"""Redundancy / overlap heuristic for multi-villain pursuit."""

from __future__ import annotations

from typing import Any, List, Mapping


def redundancy_score(step_logs: List[Mapping[str, Any]]) -> float:
    """Higher when villains cluster (redundant coverage). Crude proxy from pairwise distance variance."""
    if len(step_logs) < 2:
        return 0.0
    import math

    vals: list[float] = []
    for s in step_logs:
        vp = s.get("villain_positions") or {}
        if "villain_1" not in vp or "villain_2" not in vp:
            continue
        a, b = vp["villain_1"], vp["villain_2"]
        vals.append(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    # Low variance => villains stay similar distance => higher redundancy
    return float(1.0 / (1.0 + var))
