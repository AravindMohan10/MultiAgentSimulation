#!/usr/bin/env python3
"""
Run LLM episodes for BAML evaluation (before=pydantic, after=baml).

Logs -> output_Baml/{phase}/data/logs/
Summary tag: llm_parser in episode summary JSON.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.env_loader import load_local_env

load_local_env(repo_root=_ROOT)

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
    p = argparse.ArgumentParser(description="Run BAML eval episode batch.")
    p.add_argument(
        "--phase",
        required=True,
        choices=["before", "after"],
        help="before: LLM_OUTPUT_PARSER=pydantic; after: baml",
    )
    p.add_argument(
        "--maps",
        nargs="+",
        default=["open", "chokepoint", "standard_maze"],
        choices=["open", "chokepoint", "standard_maze"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--prompt-version", default="V2_GUIDED")
    p.add_argument(
        "--output-root",
        type=Path,
        default=_ROOT / "output_Baml",
    )
    args = p.parse_args()

    parser_mode = "pydantic" if args.phase == "before" else "baml"
    os.environ["LLM_OUTPUT_PARSER"] = parser_mode

    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY.", file=sys.stderr)
        sys.exit(1)

    log_dir = args.output_root / args.phase / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    clients = build_default_groq_clients()
    agents = _agents(args.prompt_version)

    print(f"Phase={args.phase}  parser={parser_mode}  log_dir={log_dir}")

    for i, mt in enumerate(args.maps):
        print(f"\n--- map {mt} ({i + 1}/{len(args.maps)}) ---")
        env = EnvironmentConfig(
            world_size=(100.0, 100.0),
            max_steps=args.max_steps,
            obstacle_density=0.08,
            seed=args.seed,
            map_template=MapTemplate(mt),
            obstacle_radius=1.5,
            regime_name=f"baml_eval_{args.phase}",
            **_PRESET["env"],
        )
        episode_id = f"baml_eval_{args.phase}_{mt}_seed{args.seed}"
        cfg = EpisodeConfig(
            episode_id=episode_id,
            environment=env,
            agent_configs=agents,
            llm_timeout_seconds=45.0,
        )
        out = run_episode(
            cfg,
            clients,
            log_dir=log_dir,
            summary_extra={
                "llm_parser": parser_mode,
                "baml_eval_phase": args.phase,
            },
        )
        print(out)

    meta_path = args.output_root / args.phase / "data" / "run_metadata.txt"
    meta_path.write_text(
        f"phase={args.phase}\nparser={parser_mode}\nmaps={args.maps}\n"
        f"seed={args.seed}\nmax_steps={args.max_steps}\nlog_dir={log_dir}\n",
        encoding="utf-8",
    )
    print(f"\nWrote {meta_path}")


if __name__ == "__main__":
    main()
