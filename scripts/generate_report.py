#!/usr/bin/env python3
"""
Markdown reports from batch analysis rows (Phase 1 hypothesis template).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping


def _verdict_h1(rows: List[Dict[str, Any]]) -> tuple[str, str]:
    two_v = [r for r in rows if int(r.get("num_villains") or 0) == 2]
    if not two_v:
        return "FAILED", "No 2-villain episodes in batch."
    detected = sum(1 for r in two_v if r.get("beacon_detected"))
    ratios = [float(r.get("v1_visible_during_beacon") or 0) for r in two_v if r.get("beacon_detected")]
    mean_vis = mean(ratios) if ratios else 0.0
    ok = detected >= 3 and mean_vis >= 0.8
    if ok:
        return "CONFIRMED", f"Beacon in {detected}/{len(two_v)} 2v episodes; mean v1 visible during beacon {mean_vis:.2%}."
    if detected >= 1:
        return "PARTIAL", f"Beacon in {detected}/{len(two_v)}; v1 visible mean {mean_vis:.2%}."
    return "FAILED", "No beacon episodes detected under current thresholds."


def _verdict_h2(aggregate: Mapping[str, Any], rows: List[Dict[str, Any]]) -> tuple[str, str]:
    idxs = [
        float(r["superadditivity_index"])
        for r in rows
        if r.get("superadditivity_index") is not None and int(r.get("num_villains") or 0) == 2
    ]
    glob = aggregate.get("superadditivity_index")
    passed = sum(1 for i in idxs if i > 1.0) if idxs else 0
    if idxs and passed >= 3:
        return "CONFIRMED", f"Superadditivity index >1 in {passed}/{len(idxs)} paired seeds; global={glob}."
    if glob and float(glob) > 1.0:
        return "PARTIAL", f"Mean index={glob}."
    return "FAILED", "Superadditivity not sustained across seeds."


def _verdict_h3(rows: List[Dict[str, Any]]) -> tuple[str, str]:
    two_v = [r for r in rows if int(r.get("num_villains") or 0) == 2]
    det = sum(1 for r in two_v if r.get("phase_transition_detected"))
    sharp = sum(1 for r in two_v if r.get("is_sharp_transition"))
    if not two_v:
        return "N/A", "No 2v episodes."
    return (
        "PARTIAL" if det else "FAILED",
        f"Transition detected {det}/{len(two_v)}; sharp {sharp}/{len(two_v)}.",
    )


def _verdict_h4(rows: List[Dict[str, Any]]) -> tuple[str, str]:
    th = [r.get("information_threshold") for r in rows if r.get("information_threshold") is not None]
    if len(th) < 2:
        return "PARTIAL", "Not enough threshold samples."
    vals = [float(x) for x in th]
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    cv = sd / m if m else 999.0
    cons = cv < 0.3
    return ("CONFIRMED" if cons else "PARTIAL"), f"threshold mean={m:.2f}, CV={cv:.2f}"


def write_phase_report(
    *,
    log_dir: Path,
    output_dir: Path,
    phase: int,
    rows: List[Dict[str, Any]],
    aggregate: Mapping[str, Any],
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    h1_v, h1_r = _verdict_h1(rows)
    h2_v, h2_r = _verdict_h2(aggregate, rows)
    h3_v, h3_r = _verdict_h3(rows)
    h4_v, h4_r = _verdict_h4(rows)

    lines: List[str] = [
        f"# Phase {phase} Results Report",
        "",
        f"Generated: {ts}",
        f"Episodes: {len(rows)}",
        f"Log directory: `{log_dir}`",
        "",
        "## H1: Beacon Behavior",
        "",
        h1_r,
        "",
        "### Seed-by-seed (2v)",
        "",
        "| seed | beacon | duration | onset | v1_visible | ToM_score |",
        "|------|--------|----------|-------|------------|-----------|",
    ]

    for r in sorted(
        [x for x in rows if int(x.get("num_villains") or 0) == 2],
        key=lambda x: (x.get("seed") is None, x.get("seed")),
    ):
        lines.append(
            f"| {r.get('seed')} | {r.get('beacon_detected')} | {r.get('beacon_duration')} | "
            f"{r.get('beacon_onset_step')} | {r.get('v1_visible_during_beacon')} | "
            f"{r.get('theory_of_mind_score')} |"
        )

    lines.extend(
        [
            "",
            f"**H1 VERDICT: {h1_v}**",
            "",
            "## H2: Superadditivity",
            "",
            str(aggregate),
            "",
            f"**H2 VERDICT: {h2_v}** — {h2_r}",
            "",
            "## H3: Phase Transition (MI proxy)",
            "",
            h3_r,
            "",
            f"**H3 VERDICT: {h3_v}**",
            "",
            "## H4: Information Threshold",
            "",
            h4_r,
            "",
            f"**H4 VERDICT: {h4_v}**",
            "",
            "## OVERALL PHASE 1",
            "",
            f"Foundation (H1+H2): {h1_v}/{h2_v}",
            "",
            "## Unexpected observations",
            "",
            "- (Add manual notes after review.)",
            "",
            "## Episodes with errors",
            "",
        ]
    )

    errs = [r for r in rows if (r.get("outcome") or "") == "error"]
    lines.append(f"Count: {len(errs)}")
    for r in errs:
        lines.append(f"- {r.get('episode_id')}")

    lines.extend(["", "## RAW EPISODE SUMMARIES (JSON)", "", "```json"])
    lines.append(json.dumps(rows, indent=2, ensure_ascii=False))
    lines.append("```")

    out = output_dir / f"phase{phase}_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    print("Use: python scripts/analyze_batch.py --log-dir ... --output-dir ...")
