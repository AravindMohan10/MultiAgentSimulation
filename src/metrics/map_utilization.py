"""Map structure metrics (chokepoints, spokes)."""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Sequence


def chokepoint_proximity_score(
    step_logs: List[Mapping[str, Any]],
    chokepoints: Sequence[tuple[float, float]],
    villain_id: str,
) -> float | None:
    if not step_logs or not chokepoints:
        return None
    dists: list[float] = []
    for s in step_logs:
        vp = s.get("villain_positions") or {}
        if villain_id not in vp:
            continue
        vx, vy = float(vp[villain_id][0]), float(vp[villain_id][1])
        best = min(math.hypot(vx - cx, vy - cy) for cx, cy in chokepoints)
        dists.append(best)
    if not dists:
        return None
    # Closer to chokepoints => higher score (invert mean distance)
    m = sum(dists) / len(dists)
    return float(1.0 / (1.0 + m))


def spoke_coverage_score(
    step_logs: List[Mapping[str, Any]],
    chokepoints: Sequence[tuple[float, float]],
    villain_ids: Sequence[str],
) -> float | None:
    if not step_logs or not chokepoints:
        return None
    scores = []
    for vid in villain_ids:
        c = chokepoint_proximity_score(step_logs, chokepoints, vid)
        if c is not None:
            scores.append(c)
    if not scores:
        return None
    return float(sum(scores) / len(scores))
