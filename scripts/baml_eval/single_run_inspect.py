#!/usr/bin/env python3
"""
Single-episode LLM I/O inspector.

Runs ONE episode on a chosen map template (default: scattered) and writes a
clean, structured view of:

  - the SYSTEM + USER prompt sent to the LLM (raw input)
  - the raw LLM response (raw output)
  - the parsed action (intent / target / movement)
  - validation flags (fallback, target dropped, etc.)

Outputs:
  output_Baml/single_run/<phase>_<map>_seed<seed>/
    episode_steps.jsonl    raw per-step structured rows
    inspect.md             human-readable markdown, one section per (step, agent)
    summary.json           episode outcome + counts

Usage:
    PYTHONPATH=. python scripts/baml_eval/single_run_inspect.py \
        --map scattered --max-steps 5 --parser baml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import indent

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.env_loader import load_local_env

load_local_env(repo_root=_ROOT)

from src.agents.clients import build_default_groq_clients
from src.core.models import AgentConfig, AgentType, EnvironmentConfig, MapTemplate
from src.experiments.runner import EpisodeConfig, run_episode


def _agents(prompt_version: str) -> list[AgentConfig]:
    return [
        AgentConfig(
            id="hero_1",
            agent_type=AgentType.HERO,
            strategy_mode="llm",
            model_backend="groq",
            max_speed=1.2,
            vision_radius=50.0,
            prompt_version=prompt_version,
        ),
        AgentConfig(
            id="villain_1",
            agent_type=AgentType.VILLAIN,
            strategy_mode="llm",
            model_backend="groq",
            max_speed=1.0,
            vision_radius=47.0,
            prompt_version=prompt_version,
        ),
        AgentConfig(
            id="villain_2",
            agent_type=AgentType.VILLAIN,
            strategy_mode="llm",
            model_backend="groq",
            max_speed=1.0,
            vision_radius=47.0,
            prompt_version=prompt_version,
        ),
    ]


def _split_prompt(combined: str | None) -> tuple[str, str]:
    """
    `LLMAgent._build_combined_prompt` joins SYSTEM and USER blocks with the
    headers below. Split them back out so the markdown is easy to skim.
    """
    if not combined:
        return "", ""
    sys_marker = "SYSTEM PROMPT\n-------------\n"
    usr_marker = "\n\nUSER PROMPT\n-----------\n"
    if sys_marker in combined and usr_marker in combined:
        body = combined.split(sys_marker, 1)[1]
        sys_part, usr_part = body.split(usr_marker, 1)
        return sys_part.strip(), usr_part.strip()
    return "", combined.strip()


def _pretty_json(text: str | None) -> str:
    if not text:
        return "(no raw response)"
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return text


def _md_code(label: str, body: str, lang: str = "") -> str:
    return f"#### {label}\n\n```{lang}\n{body}\n```\n"


def _write_inspect_md(log_dir: Path, episode_id: str, out_path: Path) -> None:
    steps_path = log_dir / f"{episode_id}_steps.jsonl"
    summary_path = log_dir / f"{episode_id}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append(f"# Episode inspect — `{episode_id}`\n")
    lines.append(
        f"- Outcome: **{summary.get('outcome')}**  |  steps: **{summary.get('steps')}**  "
        f"|  winner: **{summary.get('winner_team')}**\n"
    )
    lines.append(f"- Parser: `{summary.get('llm_parser', 'pydantic')}`  |  map: `{summary.get('map_template')}`\n")
    lines.append("\n---\n")

    for raw_line in steps_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        step = json.loads(raw_line)
        si = step.get("step_index")
        lines.append(f"\n## Step {si}\n")
        for agent in step.get("per_agent") or []:
            aid = agent.get("agent_id")
            role = agent.get("role")
            lines.append(f"\n### {aid}  ({role})\n")

            sys_p, usr_p = _split_prompt(agent.get("prompt"))
            if sys_p or usr_p:
                lines.append(_md_code("System prompt (raw input)", sys_p or "(empty)", "text"))
                lines.append(_md_code("User prompt (raw input)", usr_p or "(empty)", "text"))
            else:
                lines.append("_(no prompt captured — older log?)_\n")

            raw_resp = agent.get("raw_llm_response")
            lines.append(_md_code("Raw LLM response (raw output)", _pretty_json(raw_resp), "json"))

            parsed = {
                "intent": agent.get("intent"),
                "llm_raw_intent": agent.get("llm_raw_intent"),
                "llm_target_position (kept)": agent.get("llm_target_position"),
                "llm_raw_target_position (proposed)": agent.get("llm_raw_target_position"),
                "movement": agent.get("movement"),
                "movement_source": agent.get("movement_source"),
                "llm_confidence": agent.get("llm_confidence"),
                "used_fallback": agent.get("used_fallback"),
                "fallback_reason": agent.get("fallback_reason"),
                "target_out_of_bounds": agent.get("target_out_of_bounds"),
                "target_in_obstacle": agent.get("target_in_obstacle"),
            }
            lines.append(
                _md_code(
                    "Parsed action (after BAML/pydantic + sim validation)",
                    json.dumps(parsed, indent=2, ensure_ascii=False),
                    "json",
                )
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect raw LLM I/O for one episode.")
    p.add_argument("--map", default="scattered", choices=[m.value for m in MapTemplate])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80, help="Default 80 (paper protocol)")
    p.add_argument("--parser", choices=["pydantic", "baml"], default="baml")
    p.add_argument("--prompt-version", default="V2_GUIDED")
    p.add_argument("--out-root", type=Path, default=_ROOT / "output_Baml" / "single_run")
    p.add_argument("--render", action="store_true", help="Open pygame window during episode")
    p.add_argument("--fps", type=int, default=10, help="Pygame fps cap (with LLM, low is fine)")
    p.add_argument("--show-vision", action="store_true", help="Draw villain sight cones")
    args = p.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY in your environment / .env", file=sys.stderr)
        sys.exit(1)

    os.environ["LLM_OUTPUT_PARSER"] = args.parser

    run_dir = args.out_root / f"{args.parser}_{args.map}_seed{args.seed}"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = EnvironmentConfig(
        world_size=(100.0, 100.0),
        max_steps=args.max_steps,
        obstacle_density=0.08,
        seed=args.seed,
        map_template=MapTemplate(args.map),
        obstacle_radius=1.5,
        visibility_radius=50.0,
        message_delay_steps=0,
        message_budget_per_agent=None,
        observation_noise_std=0.0,
        villain_hero_sight_radius=18.0,
        regime_name=f"inspect_{args.parser}",
    )
    episode_id = f"inspect_{args.parser}_{args.map}_seed{args.seed}"
    cfg = EpisodeConfig(
        episode_id=episode_id,
        environment=env,
        agent_configs=_agents(args.prompt_version),
        llm_timeout_seconds=45.0,
    )

    renderer = None
    if args.render:
        from src.viz.pygame_renderer import PygameRenderer

        renderer = PygameRenderer(
            env,
            window_size=(1024, 1024),
            fps_cap=args.fps,
            show_vision=args.show_vision,
        )

    print(
        f"Running {episode_id}  parser={args.parser}  map={args.map}  steps={args.max_steps}"
        f"  render={'on' if renderer else 'off'}"
    )
    outcome = run_episode(
        cfg,
        build_default_groq_clients(),
        log_dir=log_dir,
        summary_extra={"llm_parser": args.parser},
        renderer=renderer,
        stream_logs=True,
    )
    print("Outcome:", outcome)

    md_path = run_dir / "inspect.md"
    _write_inspect_md(log_dir, episode_id, md_path)
    print(f"\nWrote {md_path}")
    print(f"Raw JSONL: {log_dir}/{episode_id}_steps.jsonl")
    print(f"Summary:   {log_dir}/{episode_id}_summary.json")


if __name__ == "__main__":
    main()
