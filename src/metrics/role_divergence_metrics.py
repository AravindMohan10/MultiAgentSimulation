"""
Within-episode role divergence: how villain policies diverge over time in one run.

Complements cross-episode ``role_divergence`` (entropy over all intents) in ``efficiency.py``.

Note: module named ``role_divergence_metrics`` to avoid shadowing the
``role_divergence`` function exported from ``efficiency.py`` via ``metrics.__init__``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence


def _intent_for(step: Mapping[str, Any], agent_id: str) -> str:
    for p in step.get("per_agent") or []:
        if p.get("agent_id") == agent_id:
            return str(p.get("intent") or "").strip()
    return ""


def _dominant_intent(intents: List[str]) -> str:
    intents = [i for i in intents if i]
    if not intents:
        return ""
    return Counter(intents).most_common(1)[0][0]


def _villain_distance(step: Mapping[str, Any], a: str, b: str) -> float | None:
    vp = step.get("villain_positions") or {}
    if a not in vp or b not in vp:
        return None
    pa, pb = vp[a], vp[b]
    dx = float(pa[0]) - float(pb[0])
    dy = float(pa[1]) - float(pb[1])
    return math.sqrt(dx * dx + dy * dy)


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den < 1e-18:
        return 0.0
    return float(num / den)


def within_episode_role_divergence(
    episode_steps: Sequence[Mapping[str, Any]],
    window_size: int = 10,
    agent_ids: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Computes role divergence in rolling windows across the episode.

    Parameters
    ----------
    episode_steps:
        Ordered step records (e.g. JSONL dicts or ``asdict(StepLogEntry)``).
    window_size:
        Number of consecutive steps per window.
    agent_ids:
        Villain ids to compare (default ``villain_1``, ``villain_2``).

    Returns
    -------
    dict
        ``windows`` list, ``divergence_trend`` (slope of overlap vs window index;
        negative => overlap decreases over time = more specialization),
        ``peak_divergence_step`` (start step of window with minimum overlap),
        ``final_window_intent_overlap`` (overlap of last window).

    Single-villain episodes return empty windows and zero trend (no pairwise comparison).
    """
    if agent_ids is None:
        agent_ids = ["villain_1", "villain_2"]
    if len(agent_ids) < 2:
        return {
            "windows": [],
            "divergence_trend": 0.0,
            "peak_divergence_step": 0,
            "final_window_intent_overlap": None,
        }

    v1, v2 = agent_ids[0], agent_ids[1]
    steps: List[Mapping[str, Any]] = list(episode_steps)
    n = len(steps)
    if n == 0:
        return {
            "windows": [],
            "divergence_trend": 0.0,
            "peak_divergence_step": 0,
            "final_window_intent_overlap": None,
        }

    w = max(1, int(window_size))
    windows_out: List[Dict[str, Any]] = []

    if n <= w:
        ranges = [(0, n - 1, steps)]
    else:
        ranges = []
        for start in range(0, n - w + 1):
            end = start + w - 1
            ranges.append((start, end, steps[start : end + 1]))

    for step_start, step_end, chunk in ranges:
        intents_1 = [_intent_for(s, v1) for s in chunk]
        intents_2 = [_intent_for(s, v2) for s in chunk]
        d1 = _dominant_intent(intents_1)
        d2 = _dominant_intent(intents_2)
        if d1 and d2:
            intent_overlap = 1.0 if d1 == d2 else 0.0
        else:
            intent_overlap = 0.0

        dists: List[float] = []
        for s in chunk:
            d = _villain_distance(s, v1, v2)
            if d is not None:
                dists.append(d)
        spatial_separation = float(sum(dists) / len(dists)) if dists else 0.0

        windows_out.append(
            {
                "step_start": int(step_start),
                "step_end": int(step_end),
                "intent_overlap": float(intent_overlap),
                "villain_1_dominant_intent": d1,
                "villain_2_dominant_intent": d2,
                "spatial_separation": spatial_separation,
            }
        )

    overlaps = [float(w["intent_overlap"]) for w in windows_out]
    if len(windows_out) >= 2:
        xs = [float(i) for i in range(len(windows_out))]
        divergence_trend = _linear_slope(xs, overlaps)
    elif len(windows_out) == 1:
        divergence_trend = 0.0
    else:
        divergence_trend = 0.0

    peak_divergence_step = 0
    if windows_out:
        min_i = min(range(len(windows_out)), key=lambda i: overlaps[i])
        peak_divergence_step = int(windows_out[min_i]["step_start"])

    final_overlap: float | None
    if windows_out:
        final_overlap = float(windows_out[-1]["intent_overlap"])
    else:
        final_overlap = None

    return {
        "windows": windows_out,
        "divergence_trend": float(divergence_trend),
        "peak_divergence_step": int(peak_divergence_step),
        "final_window_intent_overlap": final_overlap,
    }
