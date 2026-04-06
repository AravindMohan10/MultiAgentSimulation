from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_trajectories(step_logs: List[Dict], output_path: str | Path, mark_capture: bool = True) -> None:
    output_path = Path(output_path)
    hero_xy = []
    villain_xy: Dict[str, List[List[float]]] = {}
    for s in step_logs:
        hp = s.get("hero_position")
        if hp:
            hero_xy.append(hp[:2])
        for vid, pos in s.get("villain_positions", {}).items():
            villain_xy.setdefault(vid, []).append(pos[:2])

    plt.figure(figsize=(6, 6))
    if hero_xy:
        plt.plot([p[0] for p in hero_xy], [p[1] for p in hero_xy], color="blue", label="hero")
    colors = ["red", "orange", "purple", "brown", "green"]
    for i, (vid, pts) in enumerate(sorted(villain_xy.items())):
        c = colors[i % len(colors)]
        plt.plot([p[0] for p in pts], [p[1] for p in pts], color=c, label=vid)
    if mark_capture and hero_xy:
        plt.scatter([hero_xy[-1][0]], [hero_xy[-1][1]], c="black", s=20, marker="x")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Agent trajectories")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

