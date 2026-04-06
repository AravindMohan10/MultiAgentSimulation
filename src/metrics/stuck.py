"""Stuck rate per agent from step logs."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


def stuck_rate_per_agent(step_logs: List[Mapping[str, Any]]) -> Dict[str, float]:
    stuck: Dict[str, int] = {}
    total: Dict[str, int] = {}
    for s in step_logs:
        for p in s.get("per_agent") or []:
            aid = str(p.get("agent_id") or "")
            if not aid:
                continue
            total[aid] = total.get(aid, 0) + 1
            if p.get("stuck_this_step"):
                stuck[aid] = stuck.get(aid, 0) + 1
    out: Dict[str, float] = {}
    for aid, t in total.items():
        out[aid] = float(stuck.get(aid, 0)) / float(t) if t else 0.0
    return out
