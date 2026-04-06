#!/usr/bin/env python3
"""Run single or batched Groq experiments for research."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.env_loader import load_local_env

load_local_env(repo_root=Path(_ROOT))

from src.agents.clients import build_default_groq_clients
from src.core.models import AgentConfig, AgentType, EnvironmentConfig, MapTemplate
from src.metrics import (
    capture_rate,
    capture_time,
    chokepoint_proximity_score,
    compute_role_separation,
    detect_beacon_behavior,
    message_utilization_score,
    path_efficiency,
    redundancy_score,
    role_divergence,
    spoke_coverage_score,
    stuck_rate_per_agent,
    within_episode_role_divergence,
)
from src.experiments.runner import EpisodeConfig, run_episode


# Regimes calibrated for a 160×160 world (diagonal ≈ 226). Sight radii are a deliberate
# fraction of map width (160) for partial observability studies:
#   R1: villain↔hero "prompt visibility" radius 20 → 20/160 = 12.5% of width (interaction-rich).
#   R2: radius 15 → 9.4% width; observation_noise_std=0.2 (uncertainty).
#   R3: radius 10 → 6.25% width; observation_noise_std=0.45 (sparse + noisy).
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
    "R2": {
        "env": {
            "visibility_radius": 40.0,
            "message_delay_steps": 1,
            "message_budget_per_agent": None,
            "observation_noise_std": 0.2,
            "villain_hero_sight_radius": 15.0,
        },
        "agent": {"hero_vision_radius": 80.0, "villain_vision_radius": 75.0},
    },
    "R3": {
        "env": {
            "visibility_radius": 22.0,
            "message_delay_steps": 2,
            "message_budget_per_agent": 25,
            "observation_noise_std": 0.45,
            "villain_hero_sight_radius": 10.0,
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


def _fallback_counts(step_logs: list[dict]) -> dict[str, int]:
    """
    Episode-level fallback/error statistics derived from per-step per-agent logs.

    We count steps (not agents) so a single API failure doesn't over-weight a timestep.
    """

    fallback_steps = 0
    timeout_steps = 0
    invalid_output_steps = 0

    for step in step_logs:
        step_used_fallback = False
        step_timeout = False
        step_invalid = False

        for p in step.get("per_agent", []):
            if not p.get("used_fallback"):
                continue

            step_used_fallback = True
            intent = (p.get("intent") or "").lower()
            fr = (p.get("fallback_reason") or "").lower()

            # LLMAgent uses "timeout" intent for timeouts; everything else is treated as invalid output.
            if intent == "timeout" or fr.startswith("timeout") or "exceeded timeout" in fr or "timeout:" in fr:
                step_timeout = True
            else:
                step_invalid = True

        if step_used_fallback:
            fallback_steps += 1
        if step_timeout:
            timeout_steps += 1
        if step_invalid:
            invalid_output_steps += 1

    return {
        "num_fallback_steps": fallback_steps,
        "num_timeout_steps": timeout_steps,
        "num_invalid_outputs": invalid_output_steps,
    }


def _episode_metrics(
    outcome,
    step_logs: list[dict],
    *,
    map_template: str = "scattered",
    chokepoints: list[tuple[float, float]] | None = None,
) -> dict:
    div = role_divergence(step_logs)
    fc = _fallback_counts(step_logs)
    sr = stuck_rate_per_agent(step_logs)
    cps = chokepoints or []
    v1_cp = chokepoint_proximity_score(step_logs, cps, "villain_1") if cps else None
    v2_cp = chokepoint_proximity_score(step_logs, cps, "villain_2") if cps else None
    spoke: float | None = None
    if map_template == MapTemplate.HUB_AND_SPOKES.value and cps:
        spoke = spoke_coverage_score(step_logs, cps, ["villain_1", "villain_2"])
    wed = getattr(outcome, "within_episode_divergence", None)
    if wed is None:
        wed = within_episode_role_divergence(step_logs)
    beacon = detect_beacon_behavior(step_logs)
    role_sep = compute_role_separation(step_logs)
    return {
        "capture_time": capture_time(outcome.result, outcome.capture_step_index),
        "capture_rate": capture_rate(outcome.result),
        "redundancy_score": redundancy_score(step_logs),
        "message_utilization": message_utilization_score(step_logs),
        "role_divergence_unique_intents": div["unique_intents"],
        "role_divergence_entropy": div["entropy"],
        "path_efficiency": path_efficiency(step_logs),
        "stuck_rate_per_agent": sr,
        "map_template": map_template,
        "villain_1_chokepoint_proximity": v1_cp,
        "villain_2_chokepoint_proximity": v2_cp,
        "spoke_coverage_score": spoke,
        "divergence_trend": wed.get("divergence_trend"),
        "peak_divergence_step": wed.get("peak_divergence_step"),
        "final_window_intent_overlap": wed.get("final_window_intent_overlap"),
        "beacon_detected": beacon.get("beacon_detected"),
        "beacon_duration": beacon.get("beacon_duration"),
        "beacon_onset_step": beacon.get("beacon_onset_step"),
        "beacon_start_step": beacon.get("beacon_start_step"),
        "beacon_spatial_gain": beacon.get("beacon_spatial_gain"),
        "v1_visible_during_beacon": beacon.get("v1_visible_during_beacon"),
        "theory_of_mind_score": beacon.get("theory_of_mind_score"),
        "role_divergence_score": div.get("entropy"),
        "spontaneous_divergence_fraction": role_sep.get("spontaneous_divergence_fraction"),
        **fc,
    }


def _make_agents(
    prompt_version: str,
    *,
    hero_vision_radius: float,
    villain_vision_radius: float,
    disable_messages: bool,
    disable_memory: bool,
    disable_guidance: bool,
    hero_strategy: str = "llm",
    villain_strategy: str = "llm",
    num_villains: int = 2,
) -> list[AgentConfig]:
    hs = str(hero_strategy).lower()
    vs = str(villain_strategy).lower()
    hero_mode = "rule_based" if hs == "rule_based" else "llm"
    villain_mode = "rule_based" if vs == "rule_based" else "llm"
    nv = max(1, min(2, int(num_villains)))
    agents: list[AgentConfig] = [
        AgentConfig(
            id="hero_1",
            agent_type=AgentType.HERO,
            strategy_mode=hero_mode,
            model_backend=None if hero_mode == "rule_based" else "groq",
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
            strategy_mode=villain_mode,
            model_backend=None if villain_mode == "rule_based" else "groq",
            max_speed=1.0,
            vision_radius=villain_vision_radius,
            prompt_version=prompt_version,
            disable_messages=disable_messages,
            disable_memory=disable_memory,
            disable_guidance=disable_guidance,
        ),
    ]
    if nv >= 2:
        agents.append(
            AgentConfig(
                id="villain_2",
                agent_type=AgentType.VILLAIN,
                strategy_mode=villain_mode,
                model_backend=None if villain_mode == "rule_based" else "groq",
                max_speed=1.0,
                vision_radius=villain_vision_radius,
                prompt_version=prompt_version,
                disable_messages=disable_messages,
                disable_memory=disable_memory,
                disable_guidance=disable_guidance,
            )
        )
    return agents


def main() -> None:
    p = argparse.ArgumentParser(description="Groq experiment runner.")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs_groq"),
        help="Single-episode log directory.",
    )
    p.add_argument("--experiments-dir", type=Path, default=Path("logs_experiments"))
    p.add_argument("--no-log", action="store_true", help="Do not write log files.")
    p.add_argument("--no-viz", action="store_true", help="Headless (no Pygame window).")
    p.add_argument(
        "--show-vision",
        action="store_true",
        help="In Pygame, draw faint vision disks around agents (single-episode mode only).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Max steps per episode. 150+ recommended for 160x160 world.",
    )
    p.add_argument(
        "--map-template",
        type=str,
        choices=[
            "scattered",
            "hub_and_spokes",
            "asymmetric_labyrinth",
            "gradient",
        ],
        default="scattered",
        help=(
            "Map topology. scattered=control. hub_and_spokes=communication test. "
            "asymmetric_labyrinth=role divergence. gradient=sensitivity."
        ),
    )
    p.add_argument(
        "--gradient-max-density",
        type=float,
        default=0.25,
        help="Right-edge density for gradient map (GRADIENT template only).",
    )
    p.add_argument(
        "--fast-batch",
        action="store_true",
        help=(
            "Batch: reduced sweep — seeds [0,1,2] and villain_strategy=llm only "
            "(3×3×3×3 = 81 episodes). Full batch uses all seeds and llm+rule_based villains."
        ),
    )
    p.add_argument(
        "--episodes-per-condition",
        type=int,
        default=5,
        help="In batch mode: number of random seeds per condition (default 5 → seeds 0..4).",
    )
    p.add_argument(
        "--prompt-version",
        default="V2_GUIDED",
        choices=["V0_BASELINE", "V1_COMMUNICATION", "V2_GUIDED"],
    )
    p.add_argument(
        "--constraint",
        default="R1",
        choices=["R1", "R2", "R3", "C0", "C1", "C2"],
        help="R1=interaction-rich; R2=interaction+uncertainty; R3=sparse+uncertainty. C0–C2 alias R1–R3.",
    )
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--batch", action="store_true", help="Run full VxC grid experiments.")
    p.add_argument(
        "--villains-know-start-positions",
        action="store_true",
        help="Inject one-time initial messages so villains know each other's starting positions in first observation.",
    )
    p.add_argument("--disable-messages", action="store_true")
    p.add_argument("--disable-memory", action="store_true")
    p.add_argument("--disable-guidance", action="store_true")
    p.add_argument(
        "--hero-strategy",
        choices=["llm", "rule_based"],
        default="llm",
        help="Hero policy: Groq LLM or deterministic rule_based baseline.",
    )
    p.add_argument(
        "--villain-strategy",
        choices=["llm", "rule_based"],
        default="llm",
        help="Both villains' policy: Groq LLM or deterministic rule_based baseline.",
    )
    p.add_argument(
        "--num-villains",
        type=int,
        default=2,
        choices=[1, 2],
        help="Single-villain baseline (1: only villain_1 / close spawn) or full team (2).",
    )
    p.add_argument(
        "--spawn-mode",
        type=str,
        choices=["asymmetric", "random"],
        default="asymmetric",
        help="Villain placement: asymmetric (v1 close, v2 far) or random.",
    )
    p.add_argument(
        "--close-distance",
        type=float,
        default=12.0,
        help="Asymmetric mode: distance from hero to villain_1.",
    )
    p.add_argument(
        "--far-distance",
        type=float,
        default=60.0,
        help="Asymmetric mode: distance from hero to villain_2.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment RNG seed (map / spawns).",
    )
    args = p.parse_args()

    needs_llm = not (
        args.hero_strategy == "rule_based" and args.villain_strategy == "rule_based"
    )
    if needs_llm and not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    clients = build_default_groq_clients() if needs_llm else {}

    if not args.batch:
        preset = _constraint_config(args.constraint)
        env = EnvironmentConfig(
            world_size=(160.0, 160.0),
            max_steps=args.max_steps,
            obstacle_density=0.08,
            seed=args.seed,
            map_template=MapTemplate(args.map_template),
            gradient_max_density=args.gradient_max_density,
            **preset["env"],
            inject_villain_start_positions=args.villains_know_start_positions,
            num_villains=args.num_villains,
            spawn_mode=args.spawn_mode,
            asymmetric_close_distance=args.close_distance,
            asymmetric_far_distance=args.far_distance,
            regime_name=str(args.constraint),
        )
        agents = _make_agents(
            args.prompt_version,
            hero_vision_radius=preset["agent"]["hero_vision_radius"],
            villain_vision_radius=preset["agent"]["villain_vision_radius"],
            disable_messages=args.disable_messages,
            disable_memory=args.disable_memory,
            disable_guidance=args.disable_guidance,
            hero_strategy=args.hero_strategy,
            villain_strategy=args.villain_strategy,
            num_villains=args.num_villains,
        )
        # Include seed so multi-seed loops do not overwrite the same *_summary.json / *_steps.jsonl.
        episode_id = (
            f"groq_stepflash_demo_{args.prompt_version}_{args.constraint}_{args.map_template}"
            f"_nv{args.num_villains}_H{args.hero_strategy}_V{args.villain_strategy}"
            f"_seed{args.seed}"
        )
        cfg = EpisodeConfig(
            episode_id=episode_id,
            environment=env,
            agent_configs=agents,
            llm_timeout_seconds=45.0,
        )
        log_dir = None if args.no_log else args.log_dir
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            # Save the exact run manifest for reproducibility.
            manifest = {
                "run_type": "groq_stepflash_demo",
                "episode_id": episode_id,
                "model_backend": "groq",
                "model": os.environ.get("GROQ_MODEL", "llama3-70b-8192"),
                "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.1")),
                "top_p": 0.9,
                "max_tokens": int(os.environ.get("GROQ_MAX_TOKENS", "256")),
                "prompt_version": args.prompt_version,
                "constraint": args.constraint,
                "regime": args.constraint,
                "max_steps": args.max_steps,
                "world_size": [160.0, 160.0],
                "map_template": args.map_template,
                "gradient_max_density": args.gradient_max_density,
                "regime_env": preset["env"],
                "disable_messages": args.disable_messages,
                "disable_memory": args.disable_memory,
                "disable_guidance": args.disable_guidance,
                "hero_strategy": args.hero_strategy,
                "villain_strategy": args.villain_strategy,
                "num_villains": args.num_villains,
                "spawn_mode": args.spawn_mode,
                "asymmetric_close_distance": args.close_distance,
                "asymmetric_far_distance": args.far_distance,
                "seed": args.seed,
            }
            (log_dir / f"{episode_id}_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        renderer = None
        if not args.no_viz:
            from src.viz.pygame_renderer import PygameRenderer

            renderer = PygameRenderer(
                env,
                window_size=(1024, 1024),
                fps_cap=30,
                show_vision=bool(args.show_vision),
            )
        out = run_episode(cfg, clients, log_dir=log_dir, renderer=renderer, stream_logs=True)
        print(out)
        if log_dir is not None:
            steps = _read_steps(log_dir / f"{episode_id}_steps.jsonl")
            m = _episode_metrics(
                out,
                steps,
                map_template=env.map_template.value,
                chokepoints=env.chokepoint_positions,
            )
            print("SINGLE_EPISODE_METRICS:", json.dumps(m, separators=(",", ":"), ensure_ascii=False))
            print(
                "PHASE1_BEACON: "
                f"beacon_detected={m.get('beacon_detected')} "
                f"beacon_onset_step={m.get('beacon_onset_step')} "
                f"v1_visible_during_beacon={m.get('v1_visible_during_beacon')} "
                f"theory_of_mind_score={m.get('theory_of_mind_score')}"
            )
            slope = float(m.get("divergence_trend") or 0.0)
            print(
                f"Role divergence trend: {slope:.3f} (negative = agents specializing)"
            )
            if steps:
                last = steps[-1]
                actions_preview = [
                    {
                        "agent_id": p.get("agent_id"),
                        "intent": p.get("intent"),
                        "movement": p.get("movement"),
                        "used_fallback": p.get("used_fallback"),
                        "fallback_reason": p.get("fallback_reason"),
                    }
                    for p in last.get("per_agent", [])
                ]
                print("LAST_STEP_ACTIONS_PREVIEW:", json.dumps(actions_preview, separators=(",", ":"), ensure_ascii=False))
        return

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    per_episode_path = args.experiments_dir / "per_episode_metrics.jsonl"
    all_rows: list[dict] = []
    prompt_versions = ["V0_BASELINE", "V1_COMMUNICATION", "V2_GUIDED"]
    constraints = ["R1", "R2", "R3"]
    # Research topology aliases (publication grid): map to engine MapTemplate strings.
    topology_map: dict[str, str] = {
        "open_field": "scattered",
        "corridor": "asymmetric_labyrinth",
        "rooms": "hub_and_spokes",
    }
    topologies = list(topology_map.keys())

    if args.fast_batch:
        seed_list = [0, 1, 2]
        villain_strategies_batch = ["llm"]
    else:
        seed_list = list(range(max(1, int(args.episodes_per_condition))))
        villain_strategies_batch = ["llm", "rule_based"]

    sleep_seconds = float(os.environ.get("GROQ_BATCH_SLEEP_SECONDS", "0.75"))

    model = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
    temperature = float(os.environ.get("GROQ_TEMPERATURE", "0.1"))
    top_p = 0.9
    max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", "256"))

    total_conditions = (
        len(prompt_versions)
        * len(constraints)
        * len(topologies)
        * len(villain_strategies_batch)
        * len(seed_list)
    )

    condition_grid = {
        "prompt_versions": prompt_versions,
        "regimes": constraints,
        "map_topologies": topologies,
        "villain_strategies": villain_strategies_batch,
        "seeds": seed_list,
        "hero_strategy_fixed": args.hero_strategy,
    }

    (args.experiments_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "run_type": "groq_stepflash_demo",
                "model_backend": "groq",
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "condition_grid": condition_grid,
                "total_conditions": total_conditions,
                "llm_driven_fraction_threshold": 0.7,
                "topology_to_map_template": topology_map,
                "fast_batch": args.fast_batch,
                "episodes_per_condition": len(seed_list),
                "max_steps": args.max_steps,
                "world_size": [160.0, 160.0],
                "gradient_max_density": args.gradient_max_density,
                "sleep_seconds": sleep_seconds,
                "disable_messages": args.disable_messages,
                "disable_memory": args.disable_memory,
                "disable_guidance": args.disable_guidance,
                "hero_strategy": args.hero_strategy,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    run_dir = args.experiments_dir / "episodes"
    run_dir.mkdir(parents=True, exist_ok=True)

    with per_episode_path.open("w", encoding="utf-8") as f:
        for pv in prompt_versions:
            for cn in constraints:
                preset = _constraint_config(cn)
                for topo in topologies:
                    mt = topology_map[topo]
                    for vs in villain_strategies_batch:
                        for seed in seed_list:
                            eid = f"{pv}_{cn}_{topo}_{vs}_{seed:03d}"
                            env = EnvironmentConfig(
                                world_size=(160.0, 160.0),
                                max_steps=args.max_steps,
                                obstacle_density=0.08,
                                seed=seed,
                                map_template=MapTemplate(mt),
                                gradient_max_density=args.gradient_max_density,
                                **preset["env"],
                                inject_villain_start_positions=args.villains_know_start_positions,
                                num_villains=2,
                                regime_name=str(cn),
                            )
                            agents = _make_agents(
                                pv,
                                hero_vision_radius=preset["agent"]["hero_vision_radius"],
                                villain_vision_radius=preset["agent"]["villain_vision_radius"],
                                disable_messages=args.disable_messages,
                                disable_memory=args.disable_memory,
                                disable_guidance=args.disable_guidance,
                                hero_strategy=args.hero_strategy,
                                villain_strategy=vs,
                                num_villains=2,
                            )
                            batch_needs_llm = not (
                                args.hero_strategy == "rule_based" and vs == "rule_based"
                            )
                            batch_clients = build_default_groq_clients() if batch_needs_llm else {}
                            if batch_needs_llm and not os.environ.get("GROQ_API_KEY"):
                                print("Set GROQ_API_KEY first.", file=sys.stderr)
                                sys.exit(1)

                            cfg = EpisodeConfig(
                                episode_id=eid,
                                environment=env,
                                agent_configs=agents,
                                llm_timeout_seconds=45.0,
                            )
                            out = run_episode(
                                cfg, batch_clients, log_dir=run_dir, renderer=None, stream_logs=True
                            )
                            steps = _read_steps(run_dir / f"{eid}_steps.jsonl")
                            m = _episode_metrics(
                                out,
                                steps,
                                map_template=env.map_template.value,
                                chokepoints=env.chokepoint_positions,
                            )
                            cfg_path = run_dir / f"{eid}_config.json"
                            low_quality = False
                            llm_driven_frac: dict = {}
                            if cfg_path.exists():
                                try:
                                    cfg_obj = json.loads(cfg_path.read_text(encoding="utf-8"))
                                    llm_driven_frac = cfg_obj.get("llm_driven_step_fraction") or {}
                                    vals = [float(v) for v in llm_driven_frac.values() if v is not None]
                                    if vals and min(vals) < 0.7:
                                        low_quality = True
                                except Exception:
                                    low_quality = False

                            row = {
                                "episode_id": eid,
                                "prompt_version": pv,
                                "constraint": cn,
                                "map_topology": topo,
                                "map_template": mt,
                                "villain_strategy": vs,
                                "seed": seed,
                                "outcome": out.result,
                                "winner_team": out.winner_team,
                                "steps": out.steps,
                                "capture_step_index": out.capture_step_index,
                                "disable_messages": args.disable_messages,
                                "disable_memory": args.disable_memory,
                                "disable_guidance": args.disable_guidance,
                                "low_llm_driven_quality": low_quality,
                                "llm_driven_step_fraction": llm_driven_frac,
                                **m,
                            }
                            all_rows.append(row)
                            f.write(json.dumps(row, separators=(",", ":")) + "\n")
                            f.flush()
                            time.sleep(sleep_seconds)

    summary: dict = {"num_rows": len(all_rows), "groups": {}}
    for pv in prompt_versions:
        for cn in constraints:
            for topo in topologies:
                for vs in villain_strategies_batch:
                    rows = [
                        r
                        for r in all_rows
                        if r["prompt_version"] == pv
                        and r["constraint"] == cn
                        and r.get("map_topology") == topo
                        and r.get("villain_strategy") == vs
                    ]
                    if not rows:
                        continue
                    key = f"{pv}__{cn}__{topo}__{vs}"
                    episodes = len(rows)
                    captures = sum(1 for r in rows if r["outcome"] == "hero_captured")
                    non_capture_episodes = sum(
                        1 for r in rows if r["outcome"] in ("hero_escaped", "time_limit")
                    )
                    errors = sum(1 for r in rows if r["outcome"] == "error")

                    fallback_episodes = sum(1 for r in rows if r.get("num_fallback_steps", 0) > 0)
                    invalid_output_episodes = sum(
                        1 for r in rows if r.get("num_invalid_outputs", 0) > 0
                    )

                    capture_times = [r["capture_time"] for r in rows if r.get("capture_time") is not None]
                    capture_time_mean_when_captured = mean(capture_times) if capture_times else None
                    capture_time_median_when_captured = median(capture_times) if capture_times else None

                    summary["groups"][key] = {
                        "episodes": episodes,
                        "captures": captures,
                        "non_capture_episodes": non_capture_episodes,
                        "errors": errors,
                        "fallback_episodes": fallback_episodes,
                        "invalid_output_episodes": invalid_output_episodes,
                        "capture_rate_mean": mean(r["capture_rate"] for r in rows),
                        "capture_time_mean_when_captured": capture_time_mean_when_captured,
                        "capture_time_median_when_captured": capture_time_median_when_captured,
                        "num_fallback_steps_mean": mean(r["num_fallback_steps"] for r in rows),
                        "num_timeout_steps_mean": mean(r["num_timeout_steps"] for r in rows),
                        "num_invalid_outputs_mean": mean(r["num_invalid_outputs"] for r in rows),
                        "redundancy_score_mean": mean(r["redundancy_score"] for r in rows),
                        "message_utilization_mean": mean(r["message_utilization"] for r in rows),
                        "role_divergence_entropy_mean": mean(r["role_divergence_entropy"] for r in rows),
                        "path_efficiency_mean": mean(r["path_efficiency"] for r in rows),
                        "divergence_trend_mean": mean(
                            float(r.get("divergence_trend") or 0.0) for r in rows
                        ),
                    }
    (args.experiments_dir / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {args.experiments_dir / 'per_episode_metrics.jsonl'}")
    print(f"Saved: {args.experiments_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()

