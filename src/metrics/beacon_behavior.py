"""
Beacon navigation metric: villain_2 targets near villain_1 while blind to the hero.

Emergence signal: coordination proxy without explicit instruction.

Theory-of-mind: beacon should correlate with villain_1 seeing the hero (information available).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence


def _hypot(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def _row(step: Mapping[str, Any], agent_id: str) -> Mapping[str, Any] | None:
    for p in step.get("per_agent") or []:
        if p.get("agent_id") == agent_id:
            return p
    return None


def _v2_hero_dist(s: Mapping[str, Any]) -> float | None:
    vp2 = (s.get("villain_positions") or {}).get("villain_2")
    hp = s.get("hero_position")
    if not vp2 or not hp:
        return None
    return _hypot(float(vp2[0]), float(vp2[1]), float(hp[0]), float(hp[1]))


def detect_beacon_behavior(
    episode_steps: Sequence[Mapping[str, Any]],
    threshold_dist: float = 8.0,
    min_duration: int = 5,
    window: int = 5,
) -> Dict[str, Any]:
    """
    Detects when villain_2 uses villain_1 as a navigation beacon toward the hero.

    Per step where villain_2 ``hero_truly_visible=False`` and ``llm_target_position`` exists:

      - ``dist_target_to_v1`` = distance(v2_target, v1 actual position)
      - ``dist_target_to_hero`` = distance(v2_target, hero position)

    Raw beacon step when::

      dist_target_to_v1 < threshold_dist AND dist_target_to_hero > threshold_dist

    Only contiguous runs of raw beacon steps with length >= ``min_duration`` count.

    Parameters
    ----------
    window:
        Reserved for future rolling smoothing.
    """
    _ = window
    min_dur = max(1, int(min_duration))
    thr = float(threshold_dist)

    raw_flags: List[tuple[int, bool]] = []
    for step in episode_steps:
        si = int(step.get("step_index", -1))
        v2_row = _row(step, "villain_2")
        if v2_row is None:
            raw_flags.append((si, False))
            continue
        if v2_row.get("hero_truly_visible") is not False:
            raw_flags.append((si, False))
            continue
        tp = v2_row.get("llm_target_position")
        if not tp or len(tp) < 2:
            raw_flags.append((si, False))
            continue
        vp = step.get("villain_positions") or {}
        if "villain_1" not in vp or "villain_2" not in vp:
            raw_flags.append((si, False))
            continue
        hero = step.get("hero_position")
        if not hero or len(hero) < 2:
            raw_flags.append((si, False))
            continue

        v1 = vp["villain_1"]
        tx, ty = float(tp[0]), float(tp[1])
        d_v1 = _hypot(tx, ty, float(v1[0]), float(v1[1]))
        d_hero = _hypot(tx, ty, float(hero[0]), float(hero[1]))
        raw = bool(d_v1 < thr and d_hero > thr)
        raw_flags.append((si, raw))

    # Contiguous True runs
    runs: List[List[int]] = []
    cur: List[int] = []
    for si, flag in raw_flags:
        if flag:
            cur.append(si)
        else:
            if cur:
                runs.append(cur)
                cur = []
    if cur:
        runs.append(cur)

    valid_runs = [r for r in runs if len(r) >= min_dur]
    beacon_steps: List[int] = []
    for r in valid_runs:
        beacon_steps.extend(sorted(r))
    beacon_steps = sorted(set(beacon_steps))

    beacon_detected = len(valid_runs) > 0
    beacon_duration = len(beacon_steps)
    beacon_onset_step: int | None = int(min(beacon_steps)) if beacon_steps else None
    beacon_end_step: int | None = int(max(beacon_steps)) if beacon_steps else None

    # v1 sees hero during beacon steps
    v1_ok = 0
    v1_tot = 0
    for step in episode_steps:
        si = int(step.get("step_index", -1))
        if si not in beacon_steps:
            continue
        r1 = _row(step, "villain_1")
        if r1 is None:
            continue
        v1_tot += 1
        if r1.get("hero_truly_visible") is True:
            v1_ok += 1
    v1_visible_during_beacon = float(v1_ok) / float(v1_tot) if v1_tot > 0 else 0.0

    theory_of_mind_score = float(beacon_duration) * v1_visible_during_beacon

    # Spatial gain: pre-beacon v2–hero distance vs end of beacon vs start
    spatial_gain = 0.0
    if beacon_onset_step is not None and beacon_end_step is not None:
        def _step_at(idx: int) -> Mapping[str, Any] | None:
            for s in episode_steps:
                if int(s.get("step_index", -999999)) == idx:
                    return s
            return None

        pre_idx = beacon_onset_step - 1
        s_pre = _step_at(pre_idx)
        s_start = _step_at(beacon_onset_step)
        s_end = _step_at(beacon_end_step)
        d_pre = _v2_hero_dist(s_pre) if s_pre else None
        d_start = _v2_hero_dist(s_start) if s_start else None
        d_end = _v2_hero_dist(s_end) if s_end else None
        if d_pre is not None and d_end is not None:
            # Positive => closer to hero from pre-beacon to end of beacon
            spatial_gain = float(d_pre - d_end)
        elif d_start is not None and d_end is not None:
            spatial_gain = float(d_start - d_end)

    return {
        "beacon_detected": beacon_detected,
        "beacon_steps": beacon_steps,
        "beacon_duration": int(beacon_duration),
        "beacon_onset_step": beacon_onset_step,
        "beacon_end_step": beacon_end_step,
        "v1_visible_during_beacon": v1_visible_during_beacon,
        "spatial_gain": float(spatial_gain),
        "theory_of_mind_score": float(theory_of_mind_score),
        # Back-compat keys for older callers
        "beacon_start_step": beacon_onset_step,
        "beacon_spatial_gain": float(spatial_gain),
    }
