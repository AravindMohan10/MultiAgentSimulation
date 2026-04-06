#!/usr/bin/env python3
"""
Visualize procedural maps in Pygame with rule-based agents only (no LLM, no API keys).

Modes:
  static   — show initial world after reset (obstacles + spawns); close window or Esc to exit.
  run      — play a full episode with deterministic RuleBasedAgent policies.

Examples:
  PYTHONPATH=. python scripts/viz_maps_rule_based.py --map-template hub_and_spokes --mode static
  PYTHONPATH=. python scripts/viz_maps_rule_based.py --map-template asymmetric_labyrinth --mode run --seed 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.engine import SimulationEngine
from src.core.models import AgentConfig, AgentType, EnvironmentConfig, MapTemplate
from src.experiments.runner import EpisodeConfig, run_episode
from src.viz.pygame_renderer import PygameRenderer

# Match scripts/run_episode_groq.py regime R1 (interaction-rich visibility).
_R1_ENV_EXTRA = {
    "visibility_radius": 80.0,
    "message_delay_steps": 0,
    "message_budget_per_agent": None,
    "observation_noise_std": 0.0,
    "villain_hero_sight_radius": 20.0,
}
_VISION = {"hero": 80.0, "villain": 75.0}


def _make_rule_agents(num_villains: int) -> list[AgentConfig]:
    nv = max(1, min(2, num_villains))
    agents: list[AgentConfig] = [
        AgentConfig(
            id="hero_1",
            agent_type=AgentType.HERO,
            strategy_mode="rule_based",
            max_speed=1.2,
            vision_radius=_VISION["hero"],
            prompt_version="V2_GUIDED",
            use_auto_coord_message=False,
        ),
        AgentConfig(
            id="villain_1",
            agent_type=AgentType.VILLAIN,
            strategy_mode="rule_based",
            max_speed=1.0,
            vision_radius=_VISION["villain"],
            prompt_version="V2_GUIDED",
            use_auto_coord_message=False,
        ),
    ]
    if nv >= 2:
        agents.append(
            AgentConfig(
                id="villain_2",
                agent_type=AgentType.VILLAIN,
                strategy_mode="rule_based",
                max_speed=1.0,
                vision_radius=_VISION["villain"],
                prompt_version="V2_GUIDED",
                use_auto_coord_message=False,
            )
        )
    return agents


def _build_env(args: argparse.Namespace) -> EnvironmentConfig:
    return EnvironmentConfig(
        world_size=(160.0, 160.0),
        max_steps=max(1, int(args.max_steps)),
        obstacle_density=float(args.obstacle_density),
        seed=int(args.seed),
        map_template=MapTemplate(args.map_template),
        gradient_max_density=float(args.gradient_max_density),
        num_villains=int(args.num_villains),
        spawn_mode=str(args.spawn_mode),
        asymmetric_close_distance=float(args.close_distance),
        asymmetric_far_distance=float(args.far_distance),
        regime_name="viz_R1",
        **_R1_ENV_EXTRA,
    )


def run_static(args: argparse.Namespace) -> None:
    env = _build_env(args)
    agent_cfgs = _make_rule_agents(args.num_villains)
    engine = SimulationEngine(env, agent_cfgs)
    world_state = engine.reset()

    renderer = PygameRenderer(
        env,
        window_size=(int(args.window), int(args.window)),
        fps_cap=int(args.fps),
        show_vision=bool(args.show_vision),
    )
    renderer.init()
    caption = f"Map preview (static) — {args.map_template} seed={args.seed}"
    try:
        import pygame

        pygame.display.set_caption(caption)
    except Exception:
        pass

    print(caption)
    print("Close the window or press Esc to quit.")
    try:
        while renderer.handle_events():
            renderer.render(world_state)
    finally:
        renderer.close()


def run_episode_viz(args: argparse.Namespace) -> None:
    env = _build_env(args)
    agents = _make_rule_agents(args.num_villains)
    episode_id = f"viz_rule_{args.map_template}_seed{args.seed}"
    cfg = EpisodeConfig(episode_id=episode_id, environment=env, agent_configs=agents, capture_radius=2.0)

    log_dir = None if args.no_log else Path(args.log_dir)
    renderer = PygameRenderer(
        env,
        window_size=(int(args.window), int(args.window)),
        fps_cap=int(args.fps),
        show_vision=bool(args.show_vision),
    )
    out = run_episode(cfg, {}, log_dir=log_dir, renderer=renderer, stream_logs=True)
    print(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Pygame map viz — rule-based only, no LLM.")
    p.add_argument(
        "--mode",
        choices=("static", "run"),
        default="static",
        help="static = initial layout only; run = full episode with rule-based agents.",
    )
    p.add_argument(
        "--map-template",
        choices=["scattered", "hub_and_spokes", "asymmetric_labyrinth", "gradient"],
        default="hub_and_spokes",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200, help="Episode length when mode=run.")
    p.add_argument("--obstacle-density", type=float, default=0.08, help="SCATTERED / GRADIENT baseline.")
    p.add_argument("--gradient-max-density", type=float, default=0.25)
    p.add_argument("--num-villains", type=int, default=2, choices=[1, 2])
    p.add_argument("--spawn-mode", choices=["asymmetric", "random"], default="asymmetric")
    p.add_argument("--close-distance", type=float, default=12.0)
    p.add_argument("--far-distance", type=float, default=60.0)
    p.add_argument("--window", type=int, default=1024, help="Square window size in pixels.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--show-vision", action="store_true", help="Draw approximate vision disks.")
    p.add_argument("--no-log", action="store_true", help="When mode=run, do not write logs/ files.")
    p.add_argument("--log-dir", type=Path, default=Path("logs_viz_rule"))
    args = p.parse_args()

    if args.mode == "static":
        run_static(args)
    else:
        run_episode_viz(args)


if __name__ == "__main__":
    main()
