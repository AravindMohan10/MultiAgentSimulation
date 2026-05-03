#!/usr/bin/env python3
"""
Export a finished episode log into a compact JSON file for the 3D web viewer.

Reads *_steps.jsonl + *_summary.json + *_config.json, reconstructs obstacles
via engine reset (same seed/map), strips prompts from frames.

Usage:
    PYTHONPATH=. python scripts/export_episode_for_viewer.py \\
        --log-dir logs_self_steer \\
        --episode-id self_steer_scattered_nv2_seed0 \\
        --out viewer/public/episodes/self_steer_scattered_nv2_seed0.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.engine import SimulationEngine
from src.core.models import AgentConfig, EnvironmentConfig


def _latest_episode_id(log_dir: Path) -> str:
    files = sorted(log_dir.glob("*_steps.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"No *_steps.jsonl under {log_dir}")
    return files[0].name.removesuffix("_steps.jsonl")


def _parse_obstacles_from_prompt(prompt: str) -> List[Dict[str, float]]:
    """Fallback: parse obstacle tuples from map block in a logged prompt."""
    m = re.search(r"Obstacles:\s*(.+?)(?:\n\n|\n\[OUTPUT\])", prompt, re.DOTALL)
    if not m:
        return []
    block = m.group(1).strip()
    if block.lower() == "none":
        return []
    out: List[Dict[str, float]] = []
    for part in block.split(";"):
        part = part.strip()
        mm = re.match(r"\(([-\d.]+),([-\d.]+),r=([-\d.]+)\)", part)
        if mm:
            out.append({"x": float(mm[1]), "y": float(mm[2]), "r": float(mm[3])})
    return out


def _obstacles_from_engine(env: EnvironmentConfig, agent_cfgs: List[AgentConfig]) -> List[Dict[str, float]]:
    engine = SimulationEngine(env, agent_cfgs)
    ws = engine.reset()
    return [
        {"x": float(o.position.x), "y": float(o.position.y), "r": float(o.radius)}
        for o in ws.obstacles
    ]


def _slim_agent(row: Dict[str, Any]) -> Dict[str, Any]:
    target = row.get("llm_target_position")
    return {
        "id": row.get("agent_id"),
        "role": row.get("role"),
        "x": float(row["actual_position"][0]),
        "y": float(row["actual_position"][1]),
        "intent": row.get("intent"),
        "movementSource": row.get("movement_source"),
        "target": [float(target[0]), float(target[1])] if target else None,
        "messagesSent": int(row.get("messages_sent") or 0),
        "messagesReceived": int(row.get("messages_received") or 0),
        "usedFallback": bool(row.get("used_fallback")),
    }


def export_episode(
    log_dir: Path,
    episode_id: str,
    out_path: Path,
) -> Dict[str, Any]:
    steps_path = log_dir / f"{episode_id}_steps.jsonl"
    summary_path = log_dir / f"{episode_id}_summary.json"
    config_path = log_dir / f"{episode_id}_config.json"

    if not steps_path.is_file():
        raise FileNotFoundError(steps_path)

    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    config_blob: Dict[str, Any] = {}
    if config_path.is_file():
        config_blob = json.loads(config_path.read_text(encoding="utf-8"))

    frames: List[Dict[str, Any]] = []
    first_prompt = ""
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        per_agent = row.get("per_agent") or []
        if not first_prompt and per_agent:
            first_prompt = str(per_agent[0].get("prompt") or "")
        frames.append(
            {
                "step": int(row.get("step_index", 0)),
                "time": float(row.get("time", 0)),
                "heroPosition": row.get("hero_position"),
                "agents": [_slim_agent(a) for a in per_agent],
            }
        )

    env_data = config_blob.get("environment") or config_blob.get("episode", {}).get("environment") or {}
    world_size = env_data.get("world_size") or [160.0, 160.0]
    capture_radius = float(
        config_blob.get("episode", {}).get("capture_radius")
        or env_data.get("capture_radius")
        or summary.get("capture_radius")
        or 2.0
    )

    obstacles: List[Dict[str, float]] = []
    agent_cfgs: List[AgentConfig] = []
    try:
        if env_data:
            env = EnvironmentConfig.model_validate(env_data)
            raw_agents = config_blob.get("agent_configs") or config_blob.get("agents") or []
            agent_cfgs = [AgentConfig.model_validate(a) for a in raw_agents]
            obstacles = _obstacles_from_engine(env, agent_cfgs)
    except Exception:
        obstacles = _parse_obstacles_from_prompt(first_prompt)

    payload: Dict[str, Any] = {
        "episodeId": episode_id,
        "worldSize": [float(world_size[0]), float(world_size[1])],
        "captureRadius": capture_radius,
        "mapTemplate": summary.get("map_template") or env_data.get("map_template"),
        "seed": summary.get("seed") or env_data.get("seed"),
        "outcome": summary.get("outcome"),
        "steps": summary.get("steps") or len(frames),
        "winnerTeam": summary.get("winner_team"),
        "obstacles": obstacles,
        "frames": frames,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, default=Path("logs_self_steer"))
    p.add_argument("--episode-id", default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: viewer/public/episodes/<id>.json)",
    )
    args = p.parse_args()

    eid = args.episode_id or _latest_episode_id(args.log_dir)
    out = args.out or Path("viewer/public/episodes") / f"{eid}.json"
    payload = export_episode(args.log_dir, eid, out)
    print(f"Exported {len(payload['frames'])} frames, {len(payload['obstacles'])} obstacles → {out.resolve()}")


if __name__ == "__main__":
    main()
