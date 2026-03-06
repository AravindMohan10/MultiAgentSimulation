#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.map_validator import MapComplexityValidator
from src.core.models import EnvironmentConfig, MapTemplate
from src.core.engine import SimulationEngine


def _fmt_mean_std(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f}\u00b1{std:.{digits}f}"


def _summary_row(res: Dict[str, float]) -> Dict[str, str]:
    return {
        "Map Type": str(res["map_type"]),
        "Free Space": _fmt_mean_std(res["free_space_ratio_mean"], res["free_space_ratio_std"], 2),
        "Bottlenecks": _fmt_mean_std(res["bottleneck_count_mean"], res["bottleneck_count_std"], 0),
        "Detour Ratio": _fmt_mean_std(res["detour_ratio_mean"], res["detour_ratio_std"], 2),
        "Complexity": _fmt_mean_std(res["complexity_score_mean"], res["complexity_score_std"], 2),
        "Valid": f'{int(res["valid_count"])}/{int(res["n_seeds"])}',
    }


def _print_table(rows: List[Dict[str, str]]) -> None:
    headers = ["Map Type", "Free Space", "Bottlenecks", "Detour Ratio", "Complexity", "Valid"]
    widths = {h: max(len(h), max(len(r[h]) for r in rows)) for h in headers}
    head = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-|-".join("-" * widths[h] for h in headers)
    print(head)
    print(sep)
    for r in rows:
        print(" | ".join(r[h].ljust(widths[h]) for h in headers))


def _save_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Map Type", "Free Space", "Bottlenecks", "Detour Ratio", "Complexity", "Valid"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _plot_examples(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    map_types = [MapTemplate.OPEN, MapTemplate.CHOKEPOINT, MapTemplate.STANDARD_MAZE]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mt in zip(axes, map_types):
        env = EnvironmentConfig(
            world_size=(100.0, 100.0),
            max_steps=10,
            seed=0,
            map_template=mt,
            obstacle_radius=1.5,
        )
        eng = SimulationEngine(env, agent_configs=[])
        obstacles = eng._raw_obstacles_for_template(env)
        xs = [float(o.position.x) for o in obstacles]
        ys = [float(o.position.y) for o in obstacles]
        ss = [max(8.0, (float(o.radius) * 6.0) ** 2) for o in obstacles]
        ax.scatter(xs, ys, s=ss, alpha=0.7)
        cps = env.chokepoint_positions or []
        if cps:
            cpx = [float(x) for x, _ in cps]
            cpy = [float(y) for _, y in cps]
            ax.scatter(cpx, cpy, s=42, marker="x")
        ax.set_title(mt.value)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    n_seeds = 10
    map_types = ["open", "chokepoint", "standard_maze"]
    results = [MapComplexityValidator.validate_across_seeds(mt, n_seeds=n_seeds) for mt in map_types]
    rows = [_summary_row(r) for r in results]
    _print_table(rows)
    _save_csv(rows, Path("map_validation_results.csv"))
    _plot_examples(Path("map_validation_examples.png"))

    for r in results:
        if r["verdict"] != "PASS":
            print(f'Validation failed for {r["map_type"]}: {r["valid_count"]}/{r["n_seeds"]} valid seeds')
            raise SystemExit(2)


if __name__ == "__main__":
    main()

