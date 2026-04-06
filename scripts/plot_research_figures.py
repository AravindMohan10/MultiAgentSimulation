#!/usr/bin/env python3
"""
Build CSV + PNG figures from (1) results/**/_summary.json and (2) results/aggregates/*.json.

Usage (repo root):
  PYTHONPATH=. python scripts/plot_research_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def load_summaries(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(results_root.rglob("*_summary.json")):
        if "aggregates" in p.parts:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "file": str(p.relative_to(results_root)),
                "map_template": d.get("map_template"),
                "outcome": d.get("outcome"),
                "steps": d.get("steps"),
                "seed": d.get("seed"),
                "prompt_version": d.get("prompt_version"),
            }
        )
    return pd.DataFrame(rows)


def plot_pilot_snapshot(snapshot_path: Path, out_dir: Path) -> None:
    d = json.loads(snapshot_path.read_text(encoding="utf-8"))
    by_map = d["by_map"]
    maps = list(by_map.keys())
    esc = [by_map[m]["hero_escaped"] for m in maps]
    cap = [by_map[m]["hero_captured"] for m in maps]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(maps))
    ax.bar(x, esc, label="hero escaped", color="#4c72b0")
    ax.bar(x, cap, bottom=esc, label="hero captured", color="#c44e52")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("_", "\n") for m in maps], fontsize=8)
    ax.set_ylabel("Episodes (count)")
    ax.set_title("20-episode pilot: outcome by map (5 seeds each)")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "pilot_20ep_outcomes_by_map.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("Wrote", p)


def plot_repo_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        print("No summary rows; skip repo summaries plot.")
        return
    sub = df.dropna(subset=["map_template", "steps"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    g = sub.groupby("map_template")["steps"].mean().sort_index()
    g.plot(kind="bar", ax=ax, color="#55a868")
    ax.set_ylabel("Mean steps (logged)")
    ax.set_title("On-disk result summaries: mean steps by map")
    ax.set_xticklabels([str(x).replace("_", "\n") for x in g.index], rotation=0, fontsize=8)
    fig.tight_layout()
    p = out_dir / "repo_summaries_steps_by_map.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("Wrote", p)

    csv_path = out_dir / "repo_summaries_table.csv"
    df.to_csv(csv_path, index=False)
    print("Wrote", csv_path)


def main() -> None:
    results = _ROOT / "results"
    out_dir = results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = results / "aggregates" / "pilot_20ep_snapshot.json"
    if snap.is_file():
        plot_pilot_snapshot(snap, out_dir)

    df = load_summaries(results)
    plot_repo_summaries(df, out_dir)


if __name__ == "__main__":
    main()
