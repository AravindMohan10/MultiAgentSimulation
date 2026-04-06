#!/usr/bin/env python3
"""
Cheap multi-map pilot (Groq LLM) with maximum logging for review before big seed sweeps.

Writes under --output-dir (default: logs_pilot/pilot_<timestamp>/):
  pilot_session.json      — host, git, env snapshot (secrets redacted), run parameters
  pilot_episodes.jsonl    — one JSON object per episode (outcome + key metrics + paths)
  pilot_aggregate.json    — summary table + pass/fail quality hints
  per episode (same as run_episode):
    <episode_id>_config.json, _steps.jsonl, _summary.json
    <episode_id>_manifest.json — Groq + CLI knobs
    <episode_id>_metrics.json  — full metric dict from run_episode_groq._episode_metrics

Optional: --run-analyze  runs scripts/analyze_batch.py on this directory when done.

Usage:
  # Option A: export in shell   Option B: cp .env.example .env and set GROQ_API_KEY there
  PYTHONPATH=. python scripts/run_pilot.py --max-steps 60 --seeds 0
  PYTHONPATH=. python scripts/run_pilot.py --maps scattered,hub_and_spokes --seeds 0,1 --no-viz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.env_loader import load_local_env

load_local_env(repo_root=_ROOT)


def _load_run_episode_groq():
    path = _ROOT / "scripts" / "run_episode_groq.py"
    spec = importlib.util.spec_from_file_location("run_episode_groq", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _groq_env_for_log() -> Dict[str, Any]:
    """Relevant env vars; API key redacted."""
    out: Dict[str, Any] = {}
    for k, v in os.environ.items():
        if not k.startswith("GROQ_"):
            continue
        if k == "GROQ_API_KEY":
            out[k] = "***set***" if v else ""
        else:
            out[k] = v
    return out


def _run_analyze_batch(log_dir: Path) -> None:
    ab = _ROOT / "scripts" / "analyze_batch.py"
    if not ab.is_file():
        print("analyze_batch.py not found; skip.", file=sys.stderr)
        return
    try:
        subprocess.run(
            [sys.executable, str(ab), "--log-dir", str(log_dir), "--output-dir", str(log_dir), "--phase", "1"],
            check=False,
            cwd=str(_ROOT),
            env={**os.environ, "PYTHONPATH": str(_ROOT)},
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"analyze_batch failed: {e}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot sweep: few seeds, full logging.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Log directory (default: logs_pilot/pilot_<UTC timestamp>)",
    )
    p.add_argument(
        "--maps",
        type=str,
        default="scattered,hub_and_spokes,asymmetric_labyrinth,gradient",
        help="Comma-separated map templates.",
    )
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated integers.")
    p.add_argument("--constraint", type=str, default="R1", choices=["R1", "R2", "R3", "C0", "C1", "C2"])
    p.add_argument("--prompt-version", type=str, default="V2_GUIDED", choices=["V0_BASELINE", "V1_COMMUNICATION", "V2_GUIDED"])
    p.add_argument("--max-steps", type=int, default=60, help="Short pilot horizon (save tokens).")
    p.add_argument("--gradient-max-density", type=float, default=0.25)
    p.add_argument("--num-villains", type=int, default=2, choices=[1, 2])
    p.add_argument("--spawn-mode", type=str, default="asymmetric", choices=["asymmetric", "random"])
    p.add_argument("--close-distance", type=float, default=12.0)
    p.add_argument("--far-distance", type=float, default=60.0)
    p.add_argument("--disable-messages", action="store_true")
    p.add_argument("--disable-memory", action="store_true")
    p.add_argument("--disable-guidance", action="store_true")
    p.add_argument("--villains-know-start-positions", action="store_true")
    p.add_argument("--sleep-seconds", type=float, default=None, help="Pause between episodes (rate limit). Default: env GROQ_BATCH_SLEEP_SECONDS or 0.5")
    p.add_argument(
        "--viz",
        action="store_true",
        help="Enable pygame (default: headless). With many episodes, each run opens a window sequentially.",
    )
    p.add_argument("--run-analyze", action="store_true", help="Run analyze_batch.py on output dir after sweep.")
    p.add_argument("--llm-timeout", type=float, default=45.0)
    args = p.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY before running the pilot.", file=sys.stderr)
        sys.exit(1)

    groq = _load_run_episode_groq()
    _constraint_config = groq._constraint_config
    _make_agents = groq._make_agents
    _read_steps = groq._read_steps
    _episode_metrics = groq._episode_metrics

    from src.agents.clients import build_default_groq_clients
    from src.core.models import EnvironmentConfig, MapTemplate
    from src.experiments.runner import EpisodeConfig, run_episode

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = _ROOT / "logs_pilot" / f"pilot_{ts}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    preset = _constraint_config(args.constraint)
    sleep_s = args.sleep_seconds
    if sleep_s is None:
        sleep_s = float(os.environ.get("GROQ_BATCH_SLEEP_SECONDS", "0.5"))

    session: Dict[str, Any] = {
        "schema": "pilot_session_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "groq_env": _groq_env_for_log(),
        "argv": sys.argv,
        "parameters": {
            "maps": maps,
            "seeds": seeds,
            "constraint": args.constraint,
            "prompt_version": args.prompt_version,
            "max_steps": args.max_steps,
            "gradient_max_density": args.gradient_max_density,
            "num_villains": args.num_villains,
            "spawn_mode": args.spawn_mode,
            "asymmetric_close_distance": args.close_distance,
            "asymmetric_far_distance": args.far_distance,
            "disable_messages": args.disable_messages,
            "disable_memory": args.disable_memory,
            "disable_guidance": args.disable_guidance,
            "inject_villain_start_positions": args.villains_know_start_positions,
            "sleep_seconds_between_episodes": sleep_s,
            "llm_timeout_seconds": args.llm_timeout,
            "regime_preset": preset,
        },
        "artifacts": {
            "per_episode": [
                "{episode_id}_config.json",
                "{episode_id}_steps.jsonl",
                "{episode_id}_summary.json",
                "{episode_id}_manifest.json",
                "{episode_id}_metrics.json",
            ],
            "session": "pilot_session.json",
            "episodes_log": "pilot_episodes.jsonl",
            "aggregate": "pilot_aggregate.json",
        },
    }
    (out_dir / "pilot_session.json").write_text(
        json.dumps(session, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    clients = build_default_groq_clients()
    episode_rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    use_viz = bool(args.viz)
    renderer_mod = None
    if use_viz:
        from src.viz.pygame_renderer import PygameRenderer

    total = len(maps) * len(seeds)
    n_done = 0

    for map_name in maps:
        mt = MapTemplate(map_name)
        for seed in seeds:
            n_done += 1
            episode_id = f"pilot_{ts}_{map_name}_{args.constraint}_seed{seed}"
            env = EnvironmentConfig(
                world_size=(160.0, 160.0),
                max_steps=int(args.max_steps),
                obstacle_density=0.08,
                seed=seed,
                map_template=mt,
                gradient_max_density=float(args.gradient_max_density),
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
                hero_strategy="llm",
                villain_strategy="llm",
                num_villains=args.num_villains,
            )
            cfg = EpisodeConfig(
                episode_id=episode_id,
                environment=env,
                agent_configs=agents,
                llm_timeout_seconds=float(args.llm_timeout),
            )

            manifest = {
                "run_type": "pilot",
                "episode_id": episode_id,
                "pilot_batch": f"{n_done}/{total}",
                "model_backend": "groq",
                "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.1")),
                "max_tokens": int(os.environ.get("GROQ_MAX_TOKENS", "256")),
                "prompt_version": args.prompt_version,
                "constraint": args.constraint,
                "max_steps": args.max_steps,
                "map_template": map_name,
                "gradient_max_density": args.gradient_max_density,
                "regime_env": preset["env"],
                "seed": seed,
                "full_environment": env.model_dump(mode="json"),
                "agent_configs": [a.model_dump(mode="json") for a in agents],
            }
            (out_dir / f"{episode_id}_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            renderer = None
            if use_viz:
                renderer = PygameRenderer(env, window_size=(1024, 1024), fps_cap=30)

            t0 = time.perf_counter()
            try:
                outcome = run_episode(cfg, clients, log_dir=out_dir, renderer=renderer, stream_logs=True)
            except Exception as e:
                failures.append(f"{episode_id}: {e!r}")
                row = {
                    "episode_id": episode_id,
                    "map_template": map_name,
                    "seed": seed,
                    "ok": False,
                    "error": repr(e),
                }
                episode_rows.append(row)
                with (out_dir / "pilot_episodes.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            elapsed = time.perf_counter() - t0

            steps_path = out_dir / f"{episode_id}_steps.jsonl"
            steps = _read_steps(steps_path)
            metrics = _episode_metrics(
                outcome,
                steps,
                map_template=env.map_template.value,
                chokepoints=env.chokepoint_positions,
            )
            (out_dir / f"{episode_id}_metrics.json").write_text(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "outcome": outcome.result,
                        "steps": outcome.steps,
                        "capture_step_index": outcome.capture_step_index,
                        "winner_team": outcome.winner_team,
                        "within_episode_divergence": outcome.within_episode_divergence,
                        "metrics": metrics,
                        "elapsed_seconds": round(elapsed, 3),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            cfg_path = out_dir / f"{episode_id}_config.json"
            llm_frac: Dict[str, Any] = {}
            if cfg_path.exists():
                try:
                    cfg_obj = json.loads(cfg_path.read_text(encoding="utf-8"))
                    llm_frac = cfg_obj.get("llm_driven_step_fraction") or {}
                except json.JSONDecodeError:
                    pass

            low_quality = False
            vals = [float(v) for v in llm_frac.values() if v is not None]
            if vals and min(vals) < 0.7:
                low_quality = True

            row = {
                "episode_id": episode_id,
                "map_template": map_name,
                "seed": seed,
                "ok": True,
                "outcome": outcome.result,
                "steps": outcome.steps,
                "elapsed_seconds": round(elapsed, 3),
                "num_fallback_steps": metrics.get("num_fallback_steps"),
                "num_timeout_steps": metrics.get("num_timeout_steps"),
                "capture_time": metrics.get("capture_time"),
                "low_llm_driven_quality": low_quality,
                "llm_driven_step_fraction": llm_frac,
                "paths": {
                    "summary": f"{episode_id}_summary.json",
                    "steps": f"{episode_id}_steps.jsonl",
                    "metrics": f"{episode_id}_metrics.json",
                },
            }
            episode_rows.append(row)
            with (out_dir / "pilot_episodes.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"[{n_done}/{total}] {episode_id} -> {outcome.result} ({elapsed:.1f}s)", flush=True)

            if n_done < total and sleep_s > 0:
                time.sleep(sleep_s)

    # Aggregate
    ok_rows = [r for r in episode_rows if r.get("ok")]
    agg: Dict[str, Any] = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "total_episodes": len(episode_rows),
        "successful": len(ok_rows),
        "failures": failures,
        "episodes": episode_rows,
        "hints": {
            "check_pilot_episodes_jsonl": "Per-episode quick view; includes low_llm_driven_quality when min fraction < 0.7",
            "check_metrics_json": "Full metric bundle per episode",
            "if_high_fallback": "Inspect *_steps.jsonl for used_fallback and fallback_reason",
        },
    }
    (out_dir / "pilot_aggregate.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Logs: {out_dir}")
    print(f"  pilot_session.json  pilot_episodes.jsonl  pilot_aggregate.json")
    if failures:
        print(f"Failures: {failures}", file=sys.stderr)

    if args.run_analyze:
        print("Running analyze_batch.py ...")
        _run_analyze_batch(out_dir)


if __name__ == "__main__":
    main()
