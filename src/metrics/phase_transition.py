"""
Phase-transition proxy: mutual information jump in villain movement coupling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _movement_bin(mv: Sequence[float]) -> int:
    """8-bin direction from movement vector (dx, dy)."""
    if not mv or len(mv) < 2:
        return 0
    dx, dy = float(mv[0]), float(mv[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0
    ang = math.atan2(dy, dx)  # -pi..pi
    # Map to [0, 2pi)
    if ang < 0:
        ang += 2 * math.pi
    b = int(ang / (2 * math.pi / 8.0)) % 8
    return b


def _mi_from_joint(counts: List[List[float]], n: int) -> float:
    """MI in bits from 8x8 joint counts."""
    if n <= 0:
        return 0.0
    mi = 0.0
    for i in range(8):
        for j in range(8):
            pij = counts[i][j] / n
            if pij <= 0:
                continue
            pi = sum(counts[i][k] for k in range(8)) / n
            pj = sum(counts[k][j] for k in range(8)) / n
            if pi <= 0 or pj <= 0:
                continue
            mi += pij * math.log2(pij / (pi * pj))
    return float(mi)


def mi_for_window(steps: Sequence[Mapping[str, Any]]) -> float:
    if len(steps) < 2:
        return 0.0
    counts = [[0.0 for _ in range(8)] for _ in range(8)]
    n = 0
    for s in steps:
        m1 = m2 = None
        for p in s.get("per_agent") or []:
            aid = p.get("agent_id")
            mv = p.get("actual_movement") or p.get("movement")
            if aid == "villain_1" and isinstance(mv, (list, tuple)) and len(mv) >= 2:
                m1 = mv
            if aid == "villain_2" and isinstance(mv, (list, tuple)) and len(mv) >= 2:
                m2 = mv
        if m1 is None or m2 is None:
            continue
        b1 = _movement_bin(m1)
        b2 = _movement_bin(m2)
        counts[b1][b2] += 1.0
        n += 1
    if n == 0:
        return 0.0
    return _mi_from_joint(counts, n)


def detect_phase_transition(
    episode_steps: Sequence[Mapping[str, Any]],
    window_size: int = 10,
    mi_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Sliding-window mutual information between villain movement bins.
    Detects first sustained rise above ``mi_threshold``.
    """
    steps = list(episode_steps)
    n = len(steps)
    w = max(2, int(window_size))
    if n < w:
        return {
            "transition_detected": False,
            "transition_step": None,
            "pre_transition_mi": 0.0,
            "post_transition_mi": 0.0,
            "sharpness": 0.0,
            "is_sharp_transition": False,
            "mi_per_window": [],
            "coordination_sustained": False,
        }

    mi_windows: List[Tuple[int, float]] = []
    for start in range(0, n - w + 1):
        chunk = steps[start : start + w]
        mi_val = mi_for_window(chunk)
        step_idx = int(chunk[0].get("step_index", start))
        mi_windows.append((step_idx, mi_val))

    mi_per_window = [m for _, m in mi_windows]

    transition_step: int | None = None
    high_run = 0
    for i, (si, mi_val) in enumerate(mi_windows):
        if mi_val >= mi_threshold:
            high_run += 1
            if high_run >= 3 and transition_step is None:
                transition_step = int(mi_windows[i - 2][0])
        else:
            high_run = 0

    transition_detected = transition_step is not None

    pre_mi = 0.0
    post_mi = 0.0
    if transition_step is not None:
        pre_vals = [m for si, m in mi_windows if si < transition_step][-10:]
        post_vals = [m for si, m in mi_windows if si >= transition_step][:10]
        if pre_vals:
            pre_mi = float(sum(pre_vals) / len(pre_vals))
        if post_vals:
            post_mi = float(sum(post_vals) / len(post_vals))

    sharpness = (post_mi / pre_mi) if pre_mi > 1e-9 else float("inf") if post_mi > 0 else 0.0
    is_sharp = sharpness > 3.0 if math.isfinite(sharpness) else False

    sustained = False
    if transition_step is not None and mi_windows:
        tail = [m for si, m in mi_windows if si >= transition_step]
        sustained = len(tail) >= 3 and sum(1 for x in tail if x >= mi_threshold * 0.8) >= len(tail) * 0.6

    return {
        "transition_detected": bool(transition_detected),
        "transition_step": transition_step,
        "pre_transition_mi": float(pre_mi),
        "post_transition_mi": float(post_mi),
        "sharpness": float(sharpness) if math.isfinite(sharpness) else 999.0,
        "is_sharp_transition": bool(is_sharp),
        "mi_per_window": mi_per_window,
        "coordination_sustained": bool(sustained),
    }
