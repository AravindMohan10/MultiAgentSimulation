#!/usr/bin/env python3
"""
Greedy deterministic pursuers vs LLM hero — baseline for paper comparison.

Villains use GreedyPursuerAgent (no API). Hero uses the same LLM setup as maps_demo.
"""

from __future__ import annotations

import argparse
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
            strategy_mode="greedy_baseline",
            model_backend=None,
            max_speed=1.0,
            vision_radius=v,
            prompt_version=prompt_version,
            disable_messages=True,
            use_auto_coord_message=False,
        ),
        AgentConfig(
            id="villain_2",
            agent_type=AgentType.VILLAIN,
            strategy_mode="greedy_baseline",
            model_backend=None,
            max_speed=1.0,
            vision_radius=v,
            prompt_version=prompt_version,
            disable_messages=True,
            use_auto_coord_message=False,
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Greedy baseline pursuers + LLM hero.")
    p.add_argument(
        "--maps",
        nargs="+",
        default=["open", "chokepoint", "standard_maze"],
        choices=["open", "chokepoint", "standard_maze"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--prompt-version", default="V2_GUIDED", choices=["V0_BASELINE", "V1_COMMUNICATION", "V2_GUIDED"])
    p.add_argument("--log-dir", type=Path, default=Path("logs_greedy_baseline"))
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    args = p.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY for the LLM hero.", file=sys.stderr)
        sys.exit(1)

    print(
        "Greedy baseline: villain agents use GreedyPursuerAgent (no LLM / no API calls). "
        "Only hero_1 calls Groq.",
        flush=True,
    )

    clients = build_default_groq_clients()
    agents = _agents(args.prompt_version)

    for i, mt in enumerate(args.maps):
        print("\n" + "=" * 72)
        print(f" MAP: {mt}  ({i + 1}/{len(args.maps)})  greedy baseline")
        print("=" * 72 + "\n")

        env = EnvironmentConfig(
            world_size=(100.0, 100.0),
            max_steps=args.max_steps,
            obstacle_density=0.08,
            seed=args.seed,
            map_template=MapTemplate(mt),
            obstacle_radius=1.5,
            regime_name="greedy_baseline_R1_scaled_100",
            **_PRESET["env"],
        )

        episode_id = f"greedy_baseline_{mt}_seed{args.seed}"
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
                show_vision=False,
            )

        out = run_episode(
            cfg,
            clients,
            log_dir=log_dir,
            renderer=renderer,
            stream_logs=True,
            summary_extra={"agent_type": "greedy_baseline"},
        )
        print(out)

    print("\nDone.")


if __name__ == "__main__":
    main()
