#!/usr/bin/env python3
"""
Run a single pursuit–evasion episode with Gemini-backed agents.

Prerequisites:
  pip install -r requirements.txt
  export GEMINI_API_KEY=...

From the repo root::

  PYTHONPATH=. python scripts/run_episode_gemini.py
  PYTHONPATH=. python scripts/run_episode_gemini.py --log-dir ./runs/demo1
  PYTHONPATH=. python scripts/run_episode_gemini.py --no-log

Detailed per-step data (positions, each agent's intent, movement, comms, fallbacks)
is written when you pass ``--log-dir`` (default: ``./logs``):

  <log_dir>/<episode_id>_config.json   — environment + agent configs
  <log_dir>/<episode_id>_steps.jsonl   — one JSON object per timestep

Inspect examples::

  head -1 logs/gemini_demo_steps.jsonl | python -m json.tool
  # villains' last step intents
  tail -1 logs/gemini_demo_steps.jsonl | python -c "import json,sys; d=json.load(sys.stdin); print(d['per_agent'])"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.clients import build_default_gemini_clients
from src.core.models import AgentConfig, AgentType, EnvironmentConfig
from src.experiments.runner import EpisodeConfig, run_episode


def main() -> None:
    p = argparse.ArgumentParser(description="Run one Gemini episode with optional file logs.")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for *_config.json and *_steps.jsonl (default: ./logs)",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write log files; only print EpisodeOutcome.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Environment max_steps (default: 50)",
    )
    args = p.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    env = EnvironmentConfig(
        world_size=(80.0, 80.0),
        max_steps=args.max_steps,
        obstacle_density=0.08,
    )
    agents = [
        AgentConfig(
            id="hero_1",
            agent_type=AgentType.HERO,
            strategy_mode="llm",
            model_backend="gemini-flash",
            max_speed=1.2,
            vision_radius=14.0,
        ),
        AgentConfig(
            id="villain_1",
            agent_type=AgentType.VILLAIN,
            strategy_mode="llm",
            model_backend="gemini-flash",
            max_speed=1.0,
            vision_radius=12.0,
        ),
        AgentConfig(
            id="villain_2",
            agent_type=AgentType.VILLAIN,
            strategy_mode="llm",
            model_backend="gemini-flash",
            max_speed=1.0,
            vision_radius=12.0,
        ),
    ]
    episode_id = "gemini_demo"
    cfg = EpisodeConfig(
        episode_id=episode_id,
        environment=env,
        agent_configs=agents,
        llm_timeout_seconds=45.0,
    )
    clients = build_default_gemini_clients()
    log_dir = None if args.no_log else args.log_dir
    out = run_episode(cfg, clients, log_dir=log_dir)
    print(out)

    if log_dir is not None:
        log_dir = log_dir.resolve()
        steps_path = log_dir / f"{episode_id}_steps.jsonl"
        config_path = log_dir / f"{episode_id}_config.json"
        print("\n--- Detailed logs ---")
        print(f"  Config:  {config_path}")
        print(f"  Steps:   {steps_path}  (one JSON line per step)")
        print("\nEach step line includes:")
        print("  step_index, time, hero_position, villain_positions,")
        print("  per_agent[]: agent_id, role, intent, movement, message (if any), used_fallback, fallback_reason")


if __name__ == "__main__":
    main()
