#!/usr/bin/env python3
"""Render system design diagrams (before/after LLM boundary) to PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, xy, w, h, text, fc, ec="#333333"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def draw_before(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Before: Pydantic + manual JSON pipeline", fontsize=14, fontweight="bold")

    _box(ax, (0.05, 0.72), 0.22, 0.12, "Observation\n(prompts.py)", "#e8f4fc")
    _box(ax, (0.32, 0.72), 0.18, 0.12, "Groq API\n(raw text)", "#fff3cd")
    _box(ax, (0.54, 0.72), 0.18, 0.12, "Fence strip\nregex {…}", "#f8d7da")
    _box(ax, (0.76, 0.72), 0.18, 0.12, "json.loads\n+ expr fold", "#f8d7da")

    _box(ax, (0.20, 0.48), 0.25, 0.12, "Pydantic\nLLMActionOutput", "#d4edda")
    _box(ax, (0.50, 0.48), 0.22, 0.12, "_drop_invalid\n_llm_target", "#f8d7da")
    _box(ax, (0.76, 0.48), 0.18, 0.12, "Movement\npipelines", "#e2e3e5")

    _box(ax, (0.32, 0.22), 0.36, 0.12, "PhysicsEngine → step JSONL logs", "#cfe2ff")

    arrows = [
        ((0.27, 0.78), (0.32, 0.78)),
        ((0.50, 0.78), (0.54, 0.78)),
        ((0.72, 0.78), (0.76, 0.78)),
        ((0.85, 0.72), (0.33, 0.60)),
        ((0.45, 0.48), (0.50, 0.54)),
        ((0.72, 0.48), (0.76, 0.54)),
        ((0.50, 0.22), (0.50, 0.34)),
    ]
    for a, b in arrows:
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="->",
                mutation_scale=12,
                color="#555",
                linewidth=1.2,
            )
        )

    ax.text(
        0.5,
        0.06,
        "Failure modes: parse errors → fallback; dropped targets → null TGV; intent normalization table",
        ha="center",
        fontsize=9,
        style="italic",
        color="#666",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_after(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("After: BAML structured boundary (same prompts)", fontsize=14, fontweight="bold")

    _box(ax, (0.05, 0.72), 0.22, 0.12, "Observation\n(prompts.py)", "#e8f4fc")
    _box(ax, (0.32, 0.72), 0.28, 0.12, "BAML ChooseAgentAction\n(SAP + schema + retries)", "#d1ecf1")
    _box(ax, (0.65, 0.72), 0.28, 0.12, "LLMActionOutput\n(generated types)", "#d4edda")

    _box(ax, (0.50, 0.48), 0.22, 0.12, "_drop_invalid\n(same policy)", "#f8d7da")
    _box(ax, (0.76, 0.48), 0.18, 0.12, "Movement\npipelines", "#e2e3e5")
    _box(ax, (0.32, 0.22), 0.36, 0.12, "PhysicsEngine → step JSONL logs", "#cfe2ff")

    for a, b in [
        ((0.27, 0.78), (0.32, 0.78)),
        ((0.60, 0.78), (0.65, 0.78)),
        ((0.79, 0.72), (0.61, 0.60)),
        ((0.61, 0.48), (0.76, 0.54)),
        ((0.50, 0.22), (0.50, 0.34)),
    ]:
        ax.add_patch(
            FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=12, color="#555", linewidth=1.2)
        )

    ax.text(
        0.5,
        0.06,
        "Same prompts & sim; BAML owns HTTP+parse. Playground tests without full episodes.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#666",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["before", "after"], required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "before":
        draw_before(args.out)
    else:
        draw_after(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
