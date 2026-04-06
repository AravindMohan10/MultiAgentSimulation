"""
Superadditivity: 2-villain team vs twice single-villain performance.

Uses first-contact or capture metrics aligned by seed.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _fc_any(ep: Mapping[str, Any]) -> Optional[float]:
    v = ep.get("first_contact_step_any")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _capture_rate(eps: Sequence[Mapping[str, Any]]) -> float:
    if not eps:
        return 0.0
    cap = sum(1 for e in eps if (e.get("outcome") or "") == "hero_captured")
    return float(cap) / float(len(eps))


def compute_superadditivity(
    episodes_2v: Sequence[Mapping[str, Any]],
    episodes_1v: Sequence[Mapping[str, Any]],
    performance_metric: str = "first_contact_any",
) -> Dict[str, Any]:
    """
    Compare 2-villain vs single-villain episodes (matched by ``seed`` when present).

    For first-contact (lower is better for villains — sooner detection)::

      superadditivity_index = mean(1v_first_contact) / (2 * mean(2v_first_contact))

    Interpretation: if additive expectation is ``mean_1v / 2`` parallel search and
    ``mean_2v`` is better (smaller), index > 1 indicates superadditive coordination.

    Parameters
    ----------
    performance_metric:
        ``first_contact_any`` (default) or ``first_contact_v2`` for v2-specific detection.
    """
    key_fc = "first_contact_step_any"
    if performance_metric == "first_contact_v2":
        key_fc = "first_contact_step_v2"

    def _fc(ep: Mapping[str, Any]) -> Optional[float]:
        v = ep.get(key_fc)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    by_seed_2: Dict[Any, List[float]] = {}
    for e in episodes_2v:
        s = e.get("seed")
        fc = _fc(e)
        if fc is None:
            continue
        by_seed_2.setdefault(s, []).append(fc)

    by_seed_1: Dict[Any, List[float]] = {}
    for e in episodes_1v:
        s = e.get("seed")
        fc = _fc(e)
        if fc is None:
            continue
        by_seed_1.setdefault(s, []).append(fc)

    seeds = sorted(set(by_seed_2.keys()) & set(by_seed_1.keys()))
    paired_2: List[float] = []
    paired_1: List[float] = []
    per_seed_index: Dict[str, float] = {}

    for s in seeds:
        m2 = mean(by_seed_2[s]) if by_seed_2[s] else None
        m1 = mean(by_seed_1[s]) if by_seed_1[s] else None
        if m2 is None or m1 is None or m2 <= 0:
            continue
        idx = (m1 / 2.0) / m2
        paired_2.append(m2)
        paired_1.append(m1)
        per_seed_index[str(s)] = float(idx)

    mean_2v = mean(paired_2) if paired_2 else None
    mean_1v = mean(paired_1) if paired_1 else None

    superadditivity_index: Optional[float] = None
    if mean_2v is not None and mean_1v is not None and mean_2v > 0:
        superadditivity_index = float((mean_1v / 2.0) / mean_2v)

    is_superadditive = (
        superadditivity_index is not None and superadditivity_index > 1.0
    )

    # Effective detection radius: caller can pass precomputed radii per episode
    radii: List[float] = []
    sight = None
    for e in episodes_2v:
        r = e.get("effective_detection_radius")
        if r is not None:
            try:
                radii.append(float(r))
            except (TypeError, ValueError):
                pass
        sr = e.get("villain_hero_sight_radius")
        if sr is not None and sight is None:
            try:
                sight = float(sr)
            except (TypeError, ValueError):
                pass

    eff_mean = mean(radii) if radii else None
    individual_sight = sight if sight is not None else 20.0
    detection_radius_ratio = (
        float(eff_mean / individual_sight)
        if eff_mean is not None and individual_sight > 0
        else None
    )

    cr2 = _capture_rate(episodes_2v)
    cr1 = _capture_rate(episodes_1v)
    capture_superadditivity = None
    if cr1 > 1e-9:
        capture_superadditivity = float(cr2 / cr1)

    return {
        "superadditivity_index": superadditivity_index,
        "is_superadditive": bool(is_superadditive),
        "mean_2v_first_contact": float(mean_2v) if mean_2v is not None else None,
        "mean_1v_first_contact": float(mean_1v) if mean_1v is not None else None,
        "per_seed_superadditivity_index": per_seed_index,
        "effective_detection_radius_mean": float(eff_mean) if eff_mean is not None else None,
        "individual_sight_radius": float(individual_sight),
        "detection_radius_ratio": detection_radius_ratio,
        "capture_rate_2v": float(cr2),
        "capture_rate_1v": float(cr1),
        "capture_superadditivity": capture_superadditivity,
    }


def effective_v2_detection_radius_at_contact(
    episode_steps: Sequence[Mapping[str, Any]],
    sight_radius: float = 20.0,
) -> Optional[float]:
    """Distance |v2 - hero| at first step where v2 has ``hero_truly_visible`` True."""
    for s in episode_steps:
        r2 = None
        for p in s.get("per_agent") or []:
            if p.get("agent_id") == "villain_2" and p.get("hero_truly_visible") is True:
                r2 = p
                break
        if r2 is None:
            continue
        vp = (s.get("villain_positions") or {}).get("villain_2")
        hp = s.get("hero_position")
        if not vp or not hp:
            return None
        return float(
            math.hypot(float(vp[0]) - float(hp[0]), float(vp[1]) - float(hp[1]))
        )
    return None
