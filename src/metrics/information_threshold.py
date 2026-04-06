"""
Information threshold: relate beacon onset to steps_since_hero_seen for villain_2.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence


def analyze_information_threshold(
    episode_steps: Sequence[Mapping[str, Any]],
    beacon_result: Mapping[str, Any],
    multi_episode_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    At ``beacon_onset_step``, read villain_2 ``steps_since_hero_seen`` (blindness duration).

    ``multi_episode_thresholds`` can be passed from batch analysis to assess consistency (CV).
    """
    onset = beacon_result.get("beacon_onset_step")
    threshold_at_beacon_onset: Optional[int] = None
    pre_beacon_blind = 0

    if onset is not None:
        for s in episode_steps:
            if int(s.get("step_index", -1)) != int(onset):
                continue
            for p in s.get("per_agent") or []:
                if p.get("agent_id") == "villain_2":
                    ssh = p.get("steps_since_hero_seen")
                    if ssh is not None:
                        try:
                            threshold_at_beacon_onset = int(ssh)
                        except (TypeError, ValueError):
                            pass
                    break
            break

        # Steps v2 was blind before onset (approx: onset value of steps_since)
        if threshold_at_beacon_onset is not None:
            pre_beacon_blind = int(threshold_at_beacon_onset)

    threshold_mean: Optional[float] = None
    threshold_std: Optional[float] = None
    threshold_cv: Optional[float] = None
    is_consistent_across_seeds = False

    if multi_episode_thresholds and len(multi_episode_thresholds) >= 2:
        vals = [float(x) for x in multi_episode_thresholds if x is not None]
        if vals:
            threshold_mean = float(mean(vals))
            threshold_std = float(pstdev(vals)) if len(vals) > 1 else 0.0
            if threshold_mean and abs(threshold_mean) > 1e-9:
                threshold_cv = float(threshold_std / abs(threshold_mean))
                is_consistent_across_seeds = bool(threshold_cv < 0.3)

    rational_switch_estimate = threshold_at_beacon_onset

    return {
        "threshold_at_beacon_onset": threshold_at_beacon_onset,
        "pre_beacon_steps_blind": int(pre_beacon_blind),
        "is_consistent_across_seeds": is_consistent_across_seeds,
        "threshold_mean": threshold_mean,
        "threshold_std": threshold_std,
        "threshold_cv": threshold_cv,
        "rational_switch_estimate": rational_switch_estimate,
    }
