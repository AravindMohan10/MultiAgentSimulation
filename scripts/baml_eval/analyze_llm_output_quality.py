#!/usr/bin/env python3
"""
Extract per-step and aggregate LLM output quality metrics from episode JSONL logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.schema import ALLOWED_INTENT_VALUES

try:
    from src.agents.llm_agent import (
        _extract_json_candidate,
        _fold_numeric_expressions_in_json_text,
        _parse_llm_output,
    )
except ImportError:
    _extract_json_candidate = None  # type: ignore


def _tp_ok(v: Any) -> bool:
    return v is not None and isinstance(v, (list, tuple)) and len(v) >= 2


def _movement_mag(mv: Any) -> float:
    if not isinstance(mv, (list, tuple)) or len(mv) < 2:
        return 0.0
    return math.hypot(float(mv[0]), float(mv[1]))


def _strict_json_parse_ok(raw: Optional[str]) -> Optional[bool]:
    """True/False if raw parses with json.loads only; None if no raw text."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        import json as _json

        text = raw.strip()
        if _extract_json_candidate is not None:
            text = _extract_json_candidate(text)
        _json.loads(text)
        return True
    except Exception:
        return False


def _repair_parse_ok(raw: Optional[str]) -> Optional[bool]:
    if not raw or _parse_llm_output is None:
        return None
    try:
        _parse_llm_output(raw)
        return True
    except Exception:
        return False


def _analyze_agent_row(
    episode_id: str,
    map_template: str,
    step_index: int,
    agent: Dict[str, Any],
    parser: str,
) -> Dict[str, Any]:
    raw = agent.get("raw_llm_response")
    intent = agent.get("intent")
    raw_intent = agent.get("llm_raw_intent")
    ltp = agent.get("llm_target_position")
    raw_tp = agent.get("llm_raw_target_position")
    used_fb = bool(agent.get("used_fallback"))
    fb_reason = agent.get("fallback_reason") or ""

    has_target = _tp_ok(ltp)
    has_raw_target = _tp_ok(raw_tp)
    target_dropped = has_raw_target and not has_target
    intent_in_vocab = intent in ALLOWED_INTENT_VALUES if intent else False
    intent_mismatch = (
        raw_intent is not None
        and intent is not None
        and str(raw_intent).strip().lower() != str(intent).strip().lower()
    )

    mv = agent.get("movement_vector") or agent.get("movement")
    near_zero_move = _movement_mag(mv) < 1e-6 and not has_target

    code_bug_fallback = "raw_response' is not defined" in fb_reason

    return {
        "episode_id": episode_id,
        "map_template": map_template,
        "step_index": step_index,
        "agent_id": agent.get("agent_id"),
        "role": agent.get("role"),
        "llm_parser": parser,
        "used_fallback": used_fb,
        "fallback_reason": fb_reason,
        "code_bug_fallback": code_bug_fallback,
        "parse_error": "parse_or_validation_error" in fb_reason,
        "timeout": fb_reason.startswith("timeout"),
        "strict_json_parse_ok": _strict_json_parse_ok(raw),
        "repair_parse_ok": _repair_parse_ok(raw),
        "has_intent": intent is not None,
        "intent_in_vocab": intent_in_vocab,
        "intent_mismatch": intent_mismatch,
        "intent": intent,
        "llm_raw_intent": raw_intent,
        "has_llm_target": has_target,
        "has_raw_target": has_raw_target,
        "target_dropped": target_dropped,
        "target_out_of_bounds": bool(agent.get("target_out_of_bounds")),
        "target_in_obstacle": agent.get("target_in_obstacle"),
        "target_bfs_reachable": agent.get("target_bfs_reachable"),
        "near_zero_movement": near_zero_move,
        "confidence": agent.get("llm_confidence"),
    }


def _load_parser_from_summaries(log_dir: Path) -> str:
    for p in sorted(log_dir.glob("*_summary.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            if s.get("llm_parser"):
                return str(s["llm_parser"])
        except Exception:
            pass
    return "unknown"


def analyze_log_dir(log_dir: Path, parser: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for steps_path in sorted(log_dir.glob("*_steps.jsonl")):
        episode_id = steps_path.stem.replace("_steps", "")
        map_template = "unknown"
        parts = episode_id.split("_")
        for m in ("open", "chokepoint", "standard_maze"):
            if m in parts:
                map_template = m
                break

        for line in steps_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            si = int(step.get("step_index", 0))
            for agent in step.get("per_agent") or []:
                row = _analyze_agent_row(episode_id, map_template, si, agent, parser)
                rows.append(row)
                by_map[map_template].append(row)

    def _rate(subset: List[Dict[str, Any]], key: str) -> float:
        if not subset:
            return 0.0
        return sum(1 for r in subset if r.get(key)) / len(subset)

    def _rate_nonnull(subset: List[Dict[str, Any]], key: str) -> float:
        vals = [r.get(key) for r in subset if r.get(key) is not None]
        if not vals:
            return 0.0
        return sum(1 for v in vals if v) / len(vals)

    summary: Dict[str, Any] = {
        "parser": parser,
        "n_agent_steps": len(rows),
        "n_episodes": len(list(log_dir.glob("*_steps.jsonl"))),
        "parse_success_rate": 1.0 - _rate(rows, "used_fallback"),
        "fallback_rate": _rate(rows, "used_fallback"),
        "parse_error_rate": _rate(rows, "parse_error"),
        "unexpected_error_rate": (
            sum(
                1
                for r in rows
                if str(r.get("fallback_reason") or "").startswith("unexpected_error")
            )
            / max(len(rows), 1)
        ),
        "code_bug_fallback_rate": _rate(rows, "code_bug_fallback"),
        "timeout_rate": _rate(rows, "timeout"),
        "strict_json_parse_rate": _rate_nonnull(rows, "strict_json_parse_ok"),
        "repair_parse_rate": _rate_nonnull(rows, "repair_parse_ok"),
        "intent_in_vocab_rate": _rate(rows, "intent_in_vocab"),
        "intent_mismatch_rate": _rate(rows, "intent_mismatch"),
        "raw_target_rate": _rate(rows, "has_raw_target"),
        "kept_target_rate": _rate(rows, "has_llm_target"),
        "target_dropped_rate": _rate(rows, "target_dropped"),
        "target_out_of_bounds_rate": _rate(rows, "target_out_of_bounds"),
        "near_zero_movement_rate": _rate(rows, "near_zero_movement"),
        "by_map": {},
        "by_agent": {},
        "top_raw_intents": Counter(
            str(r.get("llm_raw_intent") or "") for r in rows if r.get("llm_raw_intent")
        ).most_common(25),
        "top_intents": Counter(str(r.get("intent") or "") for r in rows).most_common(25),
    }

    for m, subset in by_map.items():
        summary["by_map"][m] = {
            "n": len(subset),
            "parse_success_rate": 1.0 - _rate(subset, "used_fallback"),
            "raw_target_rate": _rate(subset, "has_raw_target"),
            "kept_target_rate": _rate(subset, "has_llm_target"),
            "target_dropped_rate": _rate(subset, "target_dropped"),
            "intent_in_vocab_rate": _rate(subset, "intent_in_vocab"),
        }

    for aid in ("hero_1", "villain_1", "villain_2"):
        subset = [r for r in rows if r.get("agent_id") == aid]
        if subset:
            summary["by_agent"][aid] = {
                "n": len(subset),
                "parse_success_rate": 1.0 - _rate(subset, "used_fallback"),
                "raw_target_rate": _rate(subset, "has_raw_target"),
                "target_dropped_rate": _rate(subset, "target_dropped"),
            }

    return rows, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--parser", type=str, default="")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    parser = args.parser or _load_parser_from_summaries(args.log_dir)
    rows, summary = analyze_log_dir(args.log_dir, parser)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out_csv} ({len(rows)} rows)")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
