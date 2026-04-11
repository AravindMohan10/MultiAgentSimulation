#!/usr/bin/env python3
"""
Self-steer LLM episode runner (parallel track to run_episode_groq.py).

Minimal facts-only prompts, recent[] memory in user JSON, no auto-coord messages.
Logs to logs_self_steer/ by default.
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

from src.agents.self_steer_agent import SelfSteerAgent
from src.core.models import AgentConfig, AgentType, EnvironmentConfig, MapTemplate
from src.experiments.runner import EpisodeConfig, run_episode

_REGIMES = {
    "R1": {
        "env": {
            "visibility_radius": 80.0,
            "message_delay_steps": 0,
            "message_budget_per_agent": None,
            "observation_noise_std": 0.0,
            "villain_hero_sight_radius": 20.0,
        },
        "agent": {"hero_vision_radius": 80.0, "villain_vision_radius": 75.0},
    },
}


def _constraint_config(name: str) -> dict:
    alias = {"C0": "R1", "C1": "R2", "C2": "R3"}.get(name, name)
    preset = _REGIMES.get(alias)
    if preset is None:
        preset = _REGIMES["R1"]
    return preset


def _agent_configs(
    *,
    hero_vision_radius: float,
    villain_vision_radius: float,
    num_villains: int = 2,
) -> list[AgentConfig]:
    nv = max(1, min(2, int(num_villains)))
    agents = [
        AgentConfig(
            id="hero_1",
            agent_type=AgentType.HERO,
            strategy_mode="self_steer",
            model_backend="groq",
            max_speed=1.2,
            vision_radius=hero_vision_radius,
            prompt_version="SELF_STEER",
            disable_messages=True,
            disable_memory=False,
            disable_guidance=True,
            use_auto_coord_message=False,
        ),
        AgentConfig(
            id="villain_1",
            agent_type=AgentType.VILLAIN,
            strategy_mode="self_steer",
            model_backend="groq",
            max_speed=1.0,
            vision_radius=villain_vision_radius,
            prompt_version="SELF_STEER",
            disable_messages=False,
            disable_memory=False,
            disable_guidance=True,
            use_auto_coord_message=False,
            communication_enabled=True,
        ),
    ]
    if nv >= 2:
        agents.append(
            AgentConfig(
                id="villain_2",
                agent_type=AgentType.VILLAIN,
                strategy_mode="self_steer",
                model_backend="groq",
                max_speed=1.0,
                vision_radius=villain_vision_radius,
                prompt_version="SELF_STEER",
                disable_messages=False,
                disable_memory=False,
                disable_guidance=True,
                use_auto_coord_message=False,
                communication_enabled=True,
            )
        )
    return agents


def main() -> None:
    p = argparse.ArgumentParser(description="Self-steer LLM episode (minimal prompts + recent memory).")
    p.add_argument("--map-template", default="scattered", choices=[m.value for m in MapTemplate])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--constraint", default="R1", choices=["R1", "R2", "R3", "C0", "C1", "C2"])
    p.add_argument("--capture-radius", type=float, default=2.0)
    p.add_argument("--log-dir", type=Path, default=Path("logs_self_steer"))
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--num-villains", type=int, default=2, choices=[1, 2])
    p.add_argument("--recent-limit", type=int, default=6)
    p.add_argument("--step-sleep", type=float, default=0.0, help="Seconds between LLM calls (TPM pacing).")
    p.add_argument("--no-baml", action="store_true", help="Use Groq + pydantic parse instead of BAML.")
    args = p.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    preset = _constraint_config(args.constraint)
    env = EnvironmentConfig(
        world_size=(160.0, 160.0),
        max_steps=args.max_steps,
        obstacle_density=0.08,
        seed=args.seed,
        map_template=MapTemplate(args.map_template),
        capture_radius=float(args.capture_radius),
        **preset["env"],
        regime_name="self_steer_R1",
    )
    agent_cfgs = _agent_configs(
        hero_vision_radius=preset["agent"]["hero_vision_radius"],
        villain_vision_radius=preset["agent"]["villain_vision_radius"],
        num_villains=args.num_villains,
    )

    episode_id = (
        f"self_steer_{args.map_template}_nv{args.num_villains}_seed{args.seed}"
    )
    episode_cfg = EpisodeConfig(
        episode_id=episode_id,
        environment=env,
        agent_configs=agent_cfgs,
        capture_radius=float(args.capture_radius),
        llm_timeout_seconds=45.0,
    )

    agents = {
        cfg.id: SelfSteerAgent(
            cfg,
            environment_config=env,
            capture_radius=float(args.capture_radius),
            recent_limit=args.recent_limit,
            step_sleep_seconds=args.step_sleep,
            use_baml=not args.no_baml,
        )
        for cfg in agent_cfgs
    }

    log_dir = None if args.no_log else args.log_dir
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_type": "self_steer",
            "episode_id": episode_id,
            "prompt_tier": "MINIMAL_SELF_STEER",
            "use_baml": not args.no_baml,
            "use_auto_coord_message": False,
            "recent_limit": args.recent_limit,
            "model": os.environ.get("SELF_STEER_MODEL", "llama-3.3-70b-versatile"),
            "map_template": args.map_template,
            "seed": args.seed,
            "max_steps": args.max_steps,
        }
        (log_dir / f"{episode_id}_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    renderer = None
    if not args.no_viz:
        try:
            from src.viz.pygame_renderer import PygameRenderer

            renderer = PygameRenderer(
                env,
                window_size=(1024, 1024),
                fps_cap=10,
                show_vision=False,
            )
        except ImportError as exc:
            print(f"Pygame unavailable ({exc}); running headless.", file=sys.stderr)

    print(
        f"Self-steer run: {episode_id}  map={args.map_template}  steps={args.max_steps}  "
        f"baml={not args.no_baml}  recent_limit={args.recent_limit}",
        flush=True,
    )

    outcome = run_episode(
        episode_cfg,
        llm_clients={},
        log_dir=log_dir,
        summary_extra={
            "prompt_tier": "MINIMAL_SELF_STEER",
            "llm_parser": "baml" if not args.no_baml else "pydantic",
            "use_auto_coord_message": False,
        },
        renderer=renderer,
        agents=agents,
    )
    print(f"Outcome: {outcome.result}  steps={outcome.steps}  winner={outcome.winner_team}")
    if log_dir is not None:
        print(f"Logs: {log_dir.resolve()}")


if __name__ == "__main__":
    main()
