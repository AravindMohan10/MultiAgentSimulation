#!/usr/bin/env python3
"""
Reads episode ``*_summary.json`` + ``*_steps.jsonl`` from ``--log-dir`` and produces:

1. ``results_summary.csv`` — one row per episode (wide metrics)
2. ``phase{N}_report.md`` — human-readable report (``--phase``)

Usage:
  PYTHONPATH=. python scripts/analyze_batch.py \\
    --log-dir logs_phase1 --output-dir results/phase1 --phase 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import (  # noqa: E402
    analyze_information_threshold,
    compute_role_separation,
    compute_superadditivity,
    detect_beacon_behavior,
    detect_phase_transition,
    effective_v2_detection_radius_at_contact,
    role_divergence,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_manifest(log_dir: Path, episode_id: str) -> Dict[str, Any]:
    mp = log_dir / f"{episode_id}_manifest.json"
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _stuck_totals(steps: List[Dict[str, Any]]) -> tuple[int, int]:
    v1 = v2 = 0
    for s in steps:
        for p in s.get("per_agent") or []:
            if not p.get("stuck_this_step"):
                continue
            aid = p.get("agent_id")
            if aid == "villain_1":
                v1 += 1
            elif aid == "villain_2":
                v2 += 1
    return v1, v2


def _row_from_episode(log_dir: Path, summary_path: Path) -> Optional[Dict[str, Any]]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    episode_id = summary.get("episode_id") or summary_path.stem.replace("_summary", "")
    steps_path = log_dir / f"{episode_id}_steps.jsonl"
    steps = _read_jsonl(steps_path)
    manifest = _load_manifest(log_dir, str(episode_id))

    beacon = detect_beacon_behavior(steps)
    role_sep = compute_role_separation(steps)
    div = role_divergence(steps)
    phase_tr = detect_phase_transition(steps)
    info_thr = analyze_information_threshold(steps, beacon)

    sight_r = float(summary.get("villain_hero_sight_radius") or manifest.get("regime_env", {}).get("villain_hero_sight_radius") or 20.0)
    eff_rad = effective_v2_detection_radius_at_contact(steps, sight_r)

    s1, s2 = _stuck_totals(steps)

    regime = summary.get("regime") or manifest.get("regime") or manifest.get("constraint")
    capture_step = summary.get("capture_step")
    outcome = summary.get("outcome")
    steps_to_capture = capture_step if outcome == "hero_captured" else None

    row: Dict[str, Any] = {
        "episode_id": episode_id,
        "prompt_version": summary.get("prompt_version"),
        "regime": regime,
        "map_template": summary.get("map_template"),
        "spawn_mode": summary.get("spawn_mode"),
        "seed": summary.get("seed"),
        "num_villains": summary.get("num_villains"),
        "outcome": outcome,
        "capture_step": capture_step,
        "steps_to_capture": steps_to_capture,
        "first_contact_step_v1": summary.get("first_contact_step_v1"),
        "first_contact_step_v2": summary.get("first_contact_step_v2"),
        "first_contact_step_any": summary.get("first_contact_step_any"),
        "villain_1_initial_dist": summary.get("villain_1_initial_dist"),
        "villain_2_initial_dist": summary.get("villain_2_initial_dist"),
        "beacon_detected": beacon.get("beacon_detected"),
        "beacon_duration": beacon.get("beacon_duration"),
        "beacon_onset_step": beacon.get("beacon_onset_step"),
        "beacon_end_step": beacon.get("beacon_end_step"),
        "v1_visible_during_beacon": beacon.get("v1_visible_during_beacon"),
        "beacon_spatial_gain": beacon.get("spatial_gain"),
        "theory_of_mind_score": beacon.get("theory_of_mind_score"),
        "superadditivity_index": None,
        "effective_detection_radius": eff_rad,
        "detection_radius_ratio": (eff_rad / sight_r) if eff_rad is not None and sight_r > 0 else None,
        "phase_transition_detected": phase_tr.get("transition_detected"),
        "transition_step": phase_tr.get("transition_step"),
        "transition_sharpness": phase_tr.get("sharpness"),
        "is_sharp_transition": phase_tr.get("is_sharp_transition"),
        "information_threshold": info_thr.get("threshold_at_beacon_onset"),
        "threshold_cv": info_thr.get("threshold_cv"),
        "role_divergence_score": div.get("entropy"),
        "peak_divergence_step": summary.get("peak_divergence_step"),
        "divergence_trend": summary.get("divergence_trend"),
        "spontaneous_divergence_fraction": role_sep.get("spontaneous_divergence_fraction"),
        "hero_oscillation_escape_triggered": summary.get("hero_oscillation_escape_triggered"),
        "hero_oscillation_escape_step": summary.get("hero_oscillation_escape_step"),
        "total_fallback_steps_v1": summary.get("v1_used_fallback_count"),
        "total_fallback_steps_v2": summary.get("v2_used_fallback_count"),
        "total_stuck_steps_v1": s1,
        "total_stuck_steps_v2": s2,
        "villain_hero_sight_radius": sight_r,
    }
    return row


def _aggregate_superadditivity(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r2 = [r for r in rows if int(r.get("num_villains") or 0) == 2]
    r1 = [r for r in rows if int(r.get("num_villains") or 0) == 1]
    for r in r2:
        if r.get("effective_detection_radius") is not None:
            r["villain_hero_sight_radius"] = r.get("villain_hero_sight_radius")
    return compute_superadditivity(r2, r1, performance_metric="first_contact_any")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch analysis: CSV + phase report.")
    p.add_argument("--log-dir", type=Path, required=True, help="Directory with *_summary.json")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <log-dir>)",
    )
    p.add_argument("--phase", type=int, default=1, help="Phase number for report filename")
    p.add_argument(
        "--csv-name",
        type=str,
        default="results_summary.csv",
        help="CSV filename inside output-dir",
    )
    args = p.parse_args()

    log_dir = args.log_dir
    out_dir = args.output_dir or log_dir
    if not log_dir.is_dir():
        print(f"Not a directory: {log_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = sorted(log_dir.glob("*_summary.json"))
    if not summaries:
        print(f"No *_summary.json in {log_dir}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for sp in summaries:
        r = _row_from_episode(log_dir, sp)
        if r:
            rows.append(r)

    if not rows:
        print("No valid rows.", file=sys.stderr)
        sys.exit(1)

    agg = _aggregate_superadditivity(rows)
    per = agg.get("per_seed_superadditivity_index") or {}
    for r in rows:
        if int(r.get("num_villains") or 0) != 2:
            continue
        seed = r.get("seed")
        if seed is None:
            continue
        idx = per.get(str(seed))
        if idx is not None:
            r["superadditivity_index"] = idx

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name

    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"Wrote {len(rows)} rows to {csv_path}")

    _gp = Path(__file__).resolve().parent / "generate_report.py"
    spec = importlib.util.spec_from_file_location("generate_report", _gp)
    if spec is None or spec.loader is None:
        print("Could not load generate_report.py", file=sys.stderr)
        return
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    gr.write_phase_report(
        log_dir=log_dir,
        output_dir=out_dir,
        phase=args.phase,
        rows=rows,
        aggregate=agg,
    )


if __name__ == "__main__":
    main()
