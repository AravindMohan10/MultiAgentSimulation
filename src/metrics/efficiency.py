from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List


def role_divergence(step_logs: List[Dict]) -> Dict[str, float]:
    intents: List[str] = []
    for step in step_logs:
        for p in step.get("per_agent", []):
            intent = p.get("intent")
            if intent:
                intents.append(intent)
    if not intents:
        return {"unique_intents": 0.0, "entropy": 0.0}
    c = Counter(intents)
    total = float(sum(c.values()))
    ent = 0.0
    for v in c.values():
        p = v / total
        ent -= p * math.log(max(p, 1e-12), 2)
    return {"unique_intents": float(len(c)), "entropy": float(ent)}


def path_efficiency(step_logs: List[Dict]) -> float:
    """
    Direct/actual path ratio for villains relative to final hero position.
    Lower => more wandering.
    """
    if len(step_logs) < 2:
        return 1.0
    final_hero = step_logs[-1].get("hero_position")
    if not final_hero:
        return 1.0
    villain_ids = list(step_logs[-1].get("villain_positions", {}).keys())
    ratios: List[float] = []
    for vid in villain_ids:
        coords = []
        for s in step_logs:
            vp = s.get("villain_positions", {})
            if vid in vp:
                coords.append(vp[vid])
        if len(coords) < 2:
            continue
        direct = math.dist(coords[0][:2], final_hero[:2])
        actual = 0.0
        for i in range(1, len(coords)):
            actual += math.dist(coords[i - 1][:2], coords[i][:2])
        if actual > 0:
            ratios.append(float(direct / actual))
    if not ratios:
        return 1.0
    return float(sum(ratios) / len(ratios))

