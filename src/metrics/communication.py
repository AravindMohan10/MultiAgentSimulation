"""Message utilization from step logs."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


def message_utilization_score(step_logs: List[Mapping[str, Any]]) -> float:
    if not step_logs:
        return 0.0
    total_msgs = 0
    for s in step_logs:
        for p in s.get("per_agent") or []:
            total_msgs += int(p.get("messages_sent") or 0)
    return float(total_msgs) / float(len(step_logs))
