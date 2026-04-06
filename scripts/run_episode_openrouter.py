#!/usr/bin/env python3
"""Run single or batched OpenRouter experiments for research."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.clients import build_default_openrouter_clients
from src.core.models import AgentConfig, AgentType, EnvironmentConfig
from src.metrics import (
    capture_rate,
    capture_time,
    message_utilization_score,
    path_efficiency,
    redundancy_score,
    role_divergence,
)
from src.experiments.runner import EpisodeConfig, run_episode


# Three research regimes (not arbitrary C0/C1/C2).
# Keep agent vision constant so the regime differences come only from the environment
# (`visibility_radius`, message delay, and observation noise).
_REGIMES = {
    # R1 — Interaction-rich baseline: see far, no delay/noise → coordination measurable.
    "R1": {
        "env": {
            "visibility_radius": 80.0,
            "message_delay_steps": 0,
            "message_budget_per_agent": None,
            "observation_noise_std": 0.0,
        },
        "agent": {"hero_vision_radius": 80.0, "villain_vision_radius": 75.0},
    },
    # R2 — Interaction + uncertainty (primary “C1” target): partial overlap, delay, noise.
    "R2": {
        "env": {
            "visibility_radius": 40.0,
            "message_delay_steps": 1,
            "message_budget_per_agent": None,
            "observation_noise_std": 0.2,
        },
        "agent": {"hero_vision_radius": 80.0, "villain_vision_radius": 75.0},
    },
    # R3 — Sparse + heavy uncertainty: rare line-of-sight, delayed comms, noise, budget.
    "R3": {
        "env": {
            "visibility_radius": 22.0,
            "message_delay_steps": 2,
            "message_budget_per_agent": 25,
            "observation_noise_std": 0.45,
        },
        "agent": {"hero_vision_radius": 80.0, "villain_vision_radius": 75.0},
    },
}


def _constraint_config(name: str) -> dict:
    alias = {"C0": "R1", "C1": "R2", "C2": "R3"}.get(name, name)
    preset = _REGIMES.get(alias)
    if preset is None or alias not in ("R1", "R2", "R3"):
        raise ValueError(
            f"Unknown constraint preset: {name!r}. Use R1, R2, R3 (or legacy C0/C1/C2)."
        )
    return preset


def _read_steps(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _episode_metrics(outcome, step_logs: list[dict]) -> dict:
    div = role_divergence(step_logs)
    return {
        "capture_time": capture_time(outcome.result, outcome.capture_step_index),
        "capture_rate": capture_rate(outcome.result),
        "redundancy_score": redundancy_score(step_logs),
        "message_utilization": message_utilization_score(step_logs),
        "role_divergence_unique_intents": div["unique_intents"],
        "role_divergence_entropy": div["entropy"],
        "path_efficiency": path_efficiency(step_logs),
    }


def _make_agents(
    prompt_version: str,
    *,
    hero_vision_radius: float,
    villain_vision_radius: float,
    disable_messages: bool,
    disable_memory: bool,
    disable_guidance: bool,
):
    return [
        AgentConfig(
            id="hero_1",
            agent_type=AgentType.HERO,
            strategy_mode="llm",
            model_backend="openrouter-flash",
            max_speed=1.2,
            vision_radius=hero_vision_radius,
            prompt_version=prompt_version,
            disable_messages=disable_messages,
            disable_memory=disable_memory,
            disable_guidance=disable_guidance,
        ),
        AgentConfig(
            id="villain_1",
            agent_type=AgentType.VILLAIN,
            strategy_mode="llm",
            model_backend="openrouter-flash",
            max_speed=1.0,
            vision_radius=villain_vision_radius,
            prompt_version=prompt_version,
            disable_messages=disable_messages,
            disable_memory=disable_memory,
            disable_guidance=disable_guidance,
        ),
        AgentConfig(
            id="villain_2",
            agent_type=AgentType.VILLAIN,
            strategy_mode="llm",
            model_backend="openrouter-flash",
            max_speed=1.0,
            vision_radius=villain_vision_radius,
            prompt_version=prompt_version,
            disable_messages=disable_messages,
            disable_memory=disable_memory,
            disable_guidance=disable_guidance,
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="OpenRouter experiment runner.")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs_openrouter"),
        help="Single-episode log directory.",
    )
    p.add_argument("--experiments-dir", type=Path, default=Path("logs_experiments"))
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write log files.",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Do not open a Pygame window (headless).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Environment max_steps (default: 50)",
    )
    p.add_argument("--prompt-version", default="V2_GUIDED", choices=["V0_BASELINE", "V1_COMMUNICATION", "V2_GUIDED"])
    p.add_argument(
        "--constraint",
        default="R1",
        choices=["R1", "R2", "R3", "C0", "C1", "C2"],
        help="R1=interaction-rich; R2=interaction+uncertainty; R3=sparse+uncertainty. C0–C2 alias R1–R3.",
    )
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--batch", action="store_true", help="Run full VxC grid experiments.")
    p.add_argument("--disable-messages", action="store_true")
    p.add_argument("--disable-memory", action="store_true")
    p.add_argument("--disable-guidance", action="store_true")
    args = p.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    clients = build_default_openrouter_clients()

    if not args.batch:
        preset = _constraint_config(args.constraint)
        env = EnvironmentConfig(
            world_size=(80.0, 80.0),
            max_steps=args.max_steps,
            obstacle_density=0.08,
            seed=0,
            **preset["env"],
        )
        agents = _make_agents(
            args.prompt_version,
            hero_vision_radius=preset["agent"]["hero_vision_radius"],
            villain_vision_radius=preset["agent"]["villain_vision_radius"],
            disable_messages=args.disable_messages,
            disable_memory=args.disable_memory,
            disable_guidance=args.disable_guidance,
        )
        episode_id = "openrouter_stepflash_demo"
        cfg = EpisodeConfig(
            episode_id=episode_id,
            environment=env,
            agent_configs=agents,
            llm_timeout_seconds=45.0,
        )
        log_dir = None if args.no_log else args.log_dir
        renderer = None
        if not args.no_viz:
            from src.viz.pygame_renderer import PygameRenderer

            renderer = PygameRenderer(env, window_size=(800, 800), fps_cap=30)
        out = run_episode(cfg, clients, log_dir=log_dir, renderer=renderer, stream_logs=True)
        print(out)
        return

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    per_episode_path = args.experiments_dir / "per_episode_metrics.jsonl"
    all_rows: list[dict] = []
    prompt_versions = ["V0_BASELINE", "V1_COMMUNICATION", "V2_GUIDED"]
    constraints = ["R1", "R2", "R3"]
    with per_episode_path.open("w", encoding="utf-8") as f:
        for pv in prompt_versions:
            for cn in constraints:
                preset = _constraint_config(cn)
                for epi in range(args.num_episodes):
                    eid = f"{pv}_{cn}_{epi:03d}"
                    env = EnvironmentConfig(
                        world_size=(80.0, 80.0),
                        max_steps=args.max_steps,
                        obstacle_density=0.08,
                        seed=epi,
                        **preset["env"],
                    )
                    agents = _make_agents(
                        pv,
                        hero_vision_radius=preset["agent"]["hero_vision_radius"],
                        villain_vision_radius=preset["agent"]["villain_vision_radius"],
                        disable_messages=args.disable_messages,
                        disable_memory=args.disable_memory,
                        disable_guidance=args.disable_guidance,
                    )
                    cfg = EpisodeConfig(
                        episode_id=eid,
                        environment=env,
                        agent_configs=agents,
                        llm_timeout_seconds=45.0,
                    )
                    run_dir = args.experiments_dir / "episodes"
                    out = run_episode(cfg, clients, log_dir=run_dir, renderer=None, stream_logs=True)
                    steps = _read_steps(run_dir / f"{eid}_steps.jsonl")
                    m = _episode_metrics(out, steps)
                    row = {
                        "episode_id": eid,
                        "prompt_version": pv,
                        "constraint": cn,
                        "outcome": out.result,
                        "winner_team": out.winner_team,
                        **m,
                    }
                    all_rows.append(row)
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                    f.flush()

    summary: dict = {"num_rows": len(all_rows), "groups": {}}
    for pv in prompt_versions:
        for cn in constraints:
            rows = [r for r in all_rows if r["prompt_version"] == pv and r["constraint"] == cn]
            if not rows:
                continue
            key = f"{pv}__{cn}"
            summary["groups"][key] = {
                "episodes": len(rows),
                "capture_rate_mean": mean(r["capture_rate"] for r in rows),
                "redundancy_score_mean": mean(r["redundancy_score"] for r in rows),
                "message_utilization_mean": mean(r["message_utilization"] for r in rows),
                "role_divergence_entropy_mean": mean(r["role_divergence_entropy"] for r in rows),
                "path_efficiency_mean": mean(r["path_efficiency"] for r in rows),
            }
    (args.experiments_dir / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {args.experiments_dir / 'per_episode_metrics.jsonl'}")
    print(f"Saved: {args.experiments_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
