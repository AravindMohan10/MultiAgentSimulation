"""
Spontaneous role separation: distinct villain roles without explicit prompting.

Uses intents + movement geometry relative to the hero (per-step logs).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence


def _angle_deg_between(vx: float, vy: float, wx: float, wy: float) -> float:
    n1 = math.hypot(vx, vy)
    n2 = math.hypot(wx, wy)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = max(-1.0, min(1.0, (vx * wx + vy * wy) / (n1 * n2)))
    return math.degrees(math.acos(c))


def _classify_role(
    intent: str,
    *,
    toward_hero_deg: float | None,
) -> str:
    """Map intent + geometry to PURSUER / INTERCEPTOR / SEARCHER / OTHER."""
    i = (intent or "").strip().lower()

    if i in ("explore_area", "search_systematic"):
        return "SEARCHER"

    if i in ("cut_off", "cut_off_escape", "hold_chokepoint"):
        return "INTERCEPTOR"

    if i in ("pursue_target", "pursue_memory"):
        if toward_hero_deg is not None and toward_hero_deg < 30.0:
            return "PURSUER"
        return "INTERCEPTOR"

    if i in ("regroup", "signal_teammates", "bait", "hold_position"):
        return "OTHER"

    return "OTHER"


def _per_step_row(step: Mapping[str, Any], agent_id: str) -> Mapping[str, Any] | None:
    for p in step.get("per_agent") or []:
        if p.get("agent_id") == agent_id:
            return p
    return None


def _hero_xy_for_step(step: Mapping[str, Any]) -> tuple[float, float] | None:
    """
    Hero position for this step only (same snapshot as per_agent rows).

    Prefer hero_1's actual_position from per_agent so geometry matches each
    villain row; fall back to top-level hero_position.
    """
    for p in step.get("per_agent") or []:
        if p.get("agent_id") == "hero_1":
            ap = p.get("actual_position")
            if isinstance(ap, (list, tuple)) and len(ap) >= 2:
                try:
                    return float(ap[0]), float(ap[1])
                except (TypeError, ValueError):
                    pass
            break
    hp = step.get("hero_position")
    if hp and isinstance(hp, (list, tuple)) and len(hp) >= 2:
        try:
            return float(hp[0]), float(hp[1])
        except (TypeError, ValueError):
            pass
    return None


def _dominant_role_for_window(
    chunk: Sequence[Mapping[str, Any]],
    villain_id: str,
) -> str:
    roles: List[str] = []
    for s in chunk:
        row = _per_step_row(s, villain_id)
        if row is None:
            continue
        intent = str(row.get("intent") or "")
        mv = row.get("movement")
        hero_xy = _hero_xy_for_step(s)
        vp = row.get("actual_position")
        if not (
            isinstance(vp, (list, tuple))
            and len(vp) >= 2
        ):
            vp = (s.get("villain_positions") or {}).get(villain_id)
        toward_deg: float | None = None
        if (
            hero_xy is not None
            and vp
            and isinstance(mv, (list, tuple))
            and len(mv) >= 2
            and len(vp) >= 2
        ):
            vx, vy = float(mv[0]), float(mv[1])
            hx = float(hero_xy[0]) - float(vp[0])
            hy = float(hero_xy[1]) - float(vp[1])
            toward_deg = _angle_deg_between(vx, vy, hx, hy)
        roles.append(_classify_role(intent, toward_hero_deg=toward_deg))

    if not roles:
        return "OTHER"
    return Counter(roles).most_common(1)[0][0]


def compute_role_separation(
    episode_steps: Sequence[Mapping[str, Any]],
    window: int = 10,
) -> Dict[str, Any]:
    """
    Measures whether villains spontaneously adopt distinct roles WITHOUT being prompted.

    Roles:
      PURSUER: intent in pursue_target / pursue_memory AND moving ~toward hero (<30°)
      INTERCEPTOR: cut_off / cut_off_escape OR not pure pursuer geometry
      SEARCHER: explore_area / search_systematic

    For each ``window``-step window, classify each villain's dominant role.

    Returns
    -------
    dict
        role_per_villain_per_window, divergence_score, spontaneous_divergence_steps
    """
    steps = list(episode_steps)
    n = len(steps)
    w = max(1, int(window))
    agent_ids = ["villain_1", "villain_2"]

    role_per_villain_per_window: List[Dict[str, Any]] = []

    if n == 0:
        return {
            "role_per_villain_per_window": [],
            "divergence_score": 0.0,
            "spontaneous_divergence_steps": [],
            "spontaneous_divergence_fraction": 0.0,
        }

    ranges: List[tuple[int, int, List[Mapping[str, Any]]]]
    if n <= w:
        ranges = [(0, n - 1, steps)]
    else:
        ranges = []
        for start in range(0, n - w + 1):
            end = start + w - 1
            ranges.append((start, end, steps[start : end + 1]))

    diverged_windows = 0
    total_windows = 0
    spontaneous_steps: List[int] = []

    for step_start, step_end, chunk in ranges:
        roles_map: Dict[str, str] = {}
        for vid in agent_ids:
            # Skip if villain absent (single-villain baseline)
            if not any(
                _per_step_row(s, vid) is not None for s in chunk
            ):
                continue
            roles_map[vid] = _dominant_role_for_window(chunk, vid)

        if len(roles_map) < 2:
            role_per_villain_per_window.append(
                {
                    "step_start": step_start,
                    "step_end": step_end,
                    "roles": roles_map,
                    "diverged": False,
                }
            )
            continue

        r1 = roles_map.get("villain_1", "OTHER")
        r2 = roles_map.get("villain_2", "OTHER")
        both_present = "villain_1" in roles_map and "villain_2" in roles_map
        diverged = bool(both_present and r1 != r2)

        total_windows += 1
        if diverged:
            diverged_windows += 1
            for si in range(step_start, step_end + 1):
                spontaneous_steps.append(si)

        role_per_villain_per_window.append(
            {
                "step_start": step_start,
                "step_end": step_end,
                "roles": roles_map,
                "diverged": diverged,
            }
        )

    divergence_score = (
        float(diverged_windows) / float(total_windows) if total_windows > 0 else 0.0
    )
    uniq_steps = sorted(set(spontaneous_steps))
    spontaneous_fraction = (
        float(len(uniq_steps)) / float(n) if n > 0 else 0.0
    )

    return {
        "role_per_villain_per_window": role_per_villain_per_window,
        "divergence_score": float(divergence_score),
        "spontaneous_divergence_steps": uniq_steps,
        "spontaneous_divergence_fraction": float(spontaneous_fraction),
    }
