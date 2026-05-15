#!/usr/bin/env python3
"""Write FINDINGS.md comparing before/after LLM output quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()

    root = args.output_root
    before = json.loads((root / "before" / "data" / "llm_quality_summary_before.json").read_text())
    after = json.loads((root / "after" / "data" / "llm_quality_summary_after.json").read_text())

    lines = [
        "# BAML Evaluation — MultiAgentSimulation",
        "",
        "## Setup",
        "",
        f"- **Before:** `LLM_OUTPUT_PARSER=pydantic` (manual JSON + Pydantic in `llm_agent.py`)",
        f"- **After:** `LLM_OUTPUT_PARSER=baml` (BAML `ChooseAgentAction`, same `prompts.py` text)",
        f"- **Agent-steps analyzed:** before={before.get('n_agent_steps')}, after={after.get('n_agent_steps')}",
        f"- **Episodes:** {before.get('n_episodes')} per phase",
        "",
        "## LLM output quality (primary)",
        "",
        "| Metric | Before (Pydantic) | After (BAML) | Δ |",
        "|--------|-------------------|--------------|---|",
    ]

    metrics = [
        ("parse_success_rate", "Parse success"),
        ("strict_json_parse_rate", "Strict JSON (no repair)"),
        ("repair_parse_rate", "Repair pipeline parse"),
        ("raw_target_rate", "Raw target emitted"),
        ("kept_target_rate", "Target kept"),
        ("target_dropped_rate", "Target dropped"),
        ("intent_in_vocab_rate", "Intent in vocabulary"),
        ("fallback_rate", "Fallback rate"),
        ("parse_error_rate", "Parse error rate"),
    ]

    for key, label in metrics:
        bv = float(before.get(key, 0))
        av = float(after.get(key, 0))
        delta = av - bv
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {_pct(bv)} | {_pct(av)} | {sign}{100*delta:.1f}pp |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "### Before (`output_Baml/before/`)",
            "- `data/logs/` — episode JSONL from pydantic parser",
            "- `data/llm_quality_summary_before.json` — aggregate metrics",
            "- `data/llm_quality_per_step_before.csv` — per agent-step rows",
            "- `figures/` — bar charts, funnel, intent distribution, system design",
            "",
            "### After (`output_Baml/after/`)",
            "- Same layout for BAML parser",
            "",
            "### Comparison",
            "- `comparison/figures/comparison_before_after.png`",
            "",
            "## Research payoff (secondary)",
            "",
            "Cleaner structured outputs mean step logs reflect model proposals rather than",
            "parser fallbacks or silent target drops — improving trust in intent timelines and",
            "target geometry (TGV) analyses without changing simulation physics.",
            "",
        ]
    )

    out = root / "FINDINGS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
