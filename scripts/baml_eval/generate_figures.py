#!/usr/bin/env python3
"""Generate LLM output quality charts for one phase (before or after)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_metric_bars(summary: dict, out_dir: Path, phase: str) -> None:
    metrics = [
        ("parse_success_rate", "Parse success"),
        ("strict_json_parse_rate", "Strict JSON parse"),
        ("repair_parse_rate", "Repair pipeline parse"),
        ("raw_target_rate", "Raw target emitted"),
        ("kept_target_rate", "Target kept (post-validate)"),
        ("target_dropped_rate", "Target dropped"),
        ("intent_in_vocab_rate", "Intent in vocabulary"),
        ("fallback_rate", "Fallback rate"),
    ]
    labels, vals = [], []
    for key, label in metrics:
        v = summary.get(key)
        if v is not None:
            labels.append(label)
            vals.append(float(v) * 100.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4c72b0" if phase == "before" else "#55a868"] * len(vals)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("% of agent-steps")
    ax.set_title(f"LLM output quality — {phase} ({summary.get('parser', '?')})")
    ax.set_xlim(0, 105)
    fig.tight_layout()
    p = out_dir / f"metrics_overview_{phase}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def plot_by_map(summary: dict, out_dir: Path, phase: str) -> None:
    by_map = summary.get("by_map") or {}
    if not by_map:
        return
    maps = list(by_map.keys())
    x = np.arange(len(maps))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (key, label, color) in enumerate(
        [
            ("parse_success_rate", "Parse success", "#4c72b0"),
            ("raw_target_rate", "Raw target", "#dd8452"),
            ("target_dropped_rate", "Target dropped", "#c44e52"),
        ]
    ):
        vals = [float(by_map[m].get(key, 0)) * 100 for m in maps]
        ax.bar(x + (i - 1) * w, vals, width=w, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in maps], fontsize=9)
    ax.set_ylabel("%")
    ax.set_title(f"By map — {phase}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    p = out_dir / f"metrics_by_map_{phase}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def plot_by_agent(summary: dict, out_dir: Path, phase: str) -> None:
    by_agent = summary.get("by_agent") or {}
    if not by_agent:
        return
    agents = list(by_agent.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agents))
    success = [float(by_agent[a].get("parse_success_rate", 0)) * 100 for a in agents]
    dropped = [float(by_agent[a].get("target_dropped_rate", 0)) * 100 for a in agents]
    ax.bar(x - 0.2, success, 0.35, label="Parse success", color="#4c72b0")
    ax.bar(x + 0.2, dropped, 0.35, label="Target dropped", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("%")
    ax.set_title(f"By agent — {phase}")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"metrics_by_agent_{phase}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def plot_target_funnel(summary: dict, out_dir: Path, phase: str) -> None:
    raw = float(summary.get("raw_target_rate", 0)) * 100
    kept = float(summary.get("kept_target_rate", 0)) * 100
    dropped = float(summary.get("target_dropped_rate", 0)) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    stages = ["Raw target\nemitted", "Kept after\nvalidation", "Dropped\n(raw only)"]
    vals = [raw, kept, dropped]
    colors = ["#8da0cb", "#66c2a5", "#fc8d62"]
    ax.bar(stages, vals, color=colors)
    ax.set_ylabel("% of agent-steps")
    ax.set_title(f"Target funnel — {phase}")
    ax.set_ylim(0, max(105, max(vals) * 1.1))
    fig.tight_layout()
    p = out_dir / f"target_funnel_{phase}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def plot_intent_distribution(summary: dict, out_dir: Path, phase: str) -> None:
    top = summary.get("top_raw_intents") or []
    if not top:
        return
    labels = [t[0][:20] for t in top[:12]]
    counts = [t[1] for t in top[:12]]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    ax.barh(y, counts, color="#8172b3")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Count (agent-steps)")
    ax.set_title(f"Top llm_raw_intent — {phase}")
    fig.tight_layout()
    p = out_dir / f"intent_raw_top_{phase}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def plot_confidence_hist(csv_path: Path, out_dir: Path, phase: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "confidence" not in df.columns:
        return
    conf = df["confidence"].dropna()
    if conf.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(conf, bins=20, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("llm_confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence distribution — {phase}")
    fig.tight_layout()
    p = out_dir / f"confidence_hist_{phase}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def plot_comparison(before_s: dict, after_s: dict, out_dir: Path) -> None:
    keys = [
        ("parse_success_rate", "Parse success"),
        ("raw_target_rate", "Raw target"),
        ("target_dropped_rate", "Target dropped"),
        ("intent_in_vocab_rate", "Intent in vocab"),
        ("fallback_rate", "Fallback"),
    ]
    labels = [k[1] for k in keys]
    before_v = [float(before_s.get(k[0], 0)) * 100 for k in keys]
    after_v = [float(after_s.get(k[0], 0)) * 100 for k in keys]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, before_v, w, label="Before (Pydantic)", color="#4c72b0")
    ax.bar(x + w / 2, after_v, w, label="After (BAML)", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("% of agent-steps")
    ax.set_title("LLM output quality: before vs after BAML")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    p = out_dir / "comparison_before_after.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["before", "after", "comparison"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--figures-dir", type=Path, required=True)
    p.add_argument("--before-json", type=Path, default=None)
    p.add_argument("--after-json", type=Path, default=None)
    args = p.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "comparison":
        if not args.before_json or not args.after_json:
            raise SystemExit("comparison requires --before-json and --after-json")
        plot_comparison(
            _load_summary(args.before_json),
            _load_summary(args.after_json),
            args.figures_dir,
        )
        return

    summary_path = args.data_dir / f"llm_quality_summary_{args.phase}.json"
    csv_path = args.data_dir / f"llm_quality_per_step_{args.phase}.csv"
    summary = _load_summary(summary_path)

    print(f"Figures for {args.phase}:")
    plot_metric_bars(summary, args.figures_dir, args.phase)
    plot_by_map(summary, args.figures_dir, args.phase)
    plot_by_agent(summary, args.figures_dir, args.phase)
    plot_target_funnel(summary, args.figures_dir, args.phase)
    plot_intent_distribution(summary, args.figures_dir, args.phase)
    plot_confidence_hist(csv_path, args.figures_dir, args.phase)


if __name__ == "__main__":
    main()
