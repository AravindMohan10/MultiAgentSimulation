#!/usr/bin/env python3
"""
Run one episode per map template (open, chokepoint, standard_maze) with LLM agents
and optional live Pygame visualization.

Uses the same 100×100 world and obstacle_radius=1.5 as map validation.

Examples::

  export GROQ_API_KEY=...
  PYTHONPATH=. python scripts/run_episode_llm_maps_demo.py
  PYTHONPATH=. python scripts/run_episode_llm_maps_demo.py --maps chokepoint --max-steps 80
  PYTHONPATH=. python scripts/run_episode_llm_maps_demo.py --no-viz --no-log   # headless smoke test

Requires: pygame for visualization (pip install pygame).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.env_loader import load_local_env

load_local_env(repo_root=Path(_ROOT))

from src.agents.clients import build_default_groq_clients
from src.core.models import AgentConfig, AgentType, EnvironmentConfig, MapTemplate
from src.experiments.runner import EpisodeConfig, run_episode


# Scaled from R1 for 100×100 worlds (validation geometry).
_PRESET = {
    "env": {
        "visibility_radius": 50.0,
        "message_delay_steps": 0,
        "message_budget_per_agent": None,
        "observation_noise_std": 0.0,
        "villain_hero_sight_radius": 18.0,
    },
    "agent": {"hero_vision_radius": 50.0, "villain_vision_radius": 47.0},
}


def _agents(prompt_version: str = "V2_GUIDED") -> list[AgentConfig]:
    h = _PRESET["agent"]["hero_vision_radius"]
    v = _PRESET["agent"]["villain_vision_radius"]
    return [
        AgentConfig(
            id="hero_1",
            agent_type=AgentType.HERO,
            strategy_mode="llm",
            model_backend="groq",
            max_speed=1.2,
            vision_radius=h,
            prompt_version=prompt_version,
        ),
        AgentConfig(
            id="villain_1",
            agent_type=AgentType.VILLAIN,
            strategy_mode="llm",
            model_backend="groq",
            max_speed=1.0,
            vision_radius=v,
            prompt_version=prompt_version,
        ),
        AgentConfig(
            id="villain_2",
            agent_type=AgentType.VILLAIN,
            strategy_mode="llm",
            model_backend="groq",
            max_speed=1.0,
            vision_radius=v,
            prompt_version=prompt_version,
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="LLM episodes with live viz for validated map templates.")
    p.add_argument(
        "--maps",
        nargs="+",
        default=["open", "chokepoint", "standard_maze"],
        choices=["open", "chokepoint", "standard_maze"],
        help="Map templates to run in order.",
    )
    p.add_argument("--seed", type=int, default=0, help="Environment seed.")
    p.add_argument("--max-steps", type=int, default=120, help="Max steps per episode.")
    p.add_argument("--prompt-version", default="V2_GUIDED", choices=["V0_BASELINE", "V1_COMMUNICATION", "V2_GUIDED"])
    p.add_argument("--log-dir", type=Path, default=Path("logs_maps_demo"))
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-viz", action="store_true", help="Headless (no Pygame window).")
    p.add_argument("--show-vision", action="store_true", help="Draw vision disks in Pygame.")
    p.add_argument(
        "--wait-between",
        action="store_true",
        help="Wait for Enter between episodes (TTY only).",
    )
    args = p.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY for Groq LLM agents.", file=sys.stderr)
        sys.exit(1)

    clients = build_default_groq_clients()
    agents = _agents(args.prompt_version)

    for i, mt in enumerate(args.maps):
        print("\n" + "=" * 72)
        print(f" MAP: {mt}  ({i + 1}/{len(args.maps)})")
        print("=" * 72 + "\n")

        env = EnvironmentConfig(
            world_size=(100.0, 100.0),
            max_steps=args.max_steps,
            obstacle_density=0.08,
            seed=args.seed,
            map_template=MapTemplate(mt),
            obstacle_radius=1.5,
            regime_name="maps_demo_R1_scaled_100",
            **_PRESET["env"],
        )

        episode_id = f"maps_demo_{mt}_seed{args.seed}"
        cfg = EpisodeConfig(
            episode_id=episode_id,
            environment=env,
            agent_configs=agents,
            llm_timeout_seconds=45.0,
        )

        log_dir = None if args.no_log else args.log_dir
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        renderer = None
        if not args.no_viz:
            try:
                from src.viz.pygame_renderer import PygameRenderer
            except ImportError as e:
                print("Install pygame for visualization: pip install pygame", file=sys.stderr)
                raise SystemExit(2) from e
            renderer = PygameRenderer(
                env,
                window_size=(900, 900),
                fps_cap=30,
                show_vision=bool(args.show_vision),
            )

        out = run_episode(cfg, clients, log_dir=log_dir, renderer=renderer, stream_logs=True)
        print(out)

        if args.wait_between and i + 1 < len(args.maps) and sys.stdin.isatty():
            input("Press Enter for next map… ")

    print("\nDone.")


if __name__ == "__main__":
    main()
