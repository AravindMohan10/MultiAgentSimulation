#!/usr/bin/env python3
"""
Standalone LLM observation probe — no simulation imports.

Reads villain_2 rows from finished episode logs, replays the raw observation JSON
with a minimal free-text Groq prompt, and writes probe_results.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from groq import Groq

USER_PROMPT_MARKER = "USER PROMPT\n-----------\n"

SYSTEM_PROMPT = (
    "You are an entity in a 2D world. You have a position, you can observe nearby "
    "things, and you may have received messages from others."
)

USER_PROMPT_TEMPLATE = """Here is what you currently observe:

{obs_json}

In plain English: what is happening around you right now? What do you understand about the situation? What would you do, and why? Think freely. No format required."""

PROBES = [
    {
        "map": "scattered",
        "file": "logs_groq/groq_stepflash_demo_V2_GUIDED_R1_scattered_nv2_Hllm_Vllm_seed0_steps.jsonl",
        "steps": [2, 5, 35, 79],
    },
    {
        "map": "hub_and_spokes",
        "file": "logs_groq/groq_stepflash_demo_V2_GUIDED_R1_hub_and_spokes_nv2_Hllm_Vllm_seed0_steps.jsonl",
        "steps": [2, 5, 35, 50],
    },
]

OUTPUT_MD = Path("probe_results.md")
GROQ_MODEL = "llama-3.3-70b-versatile"


def _load_step_row(steps_path: Path, step_index: int) -> dict[str, Any] | None:
    if not steps_path.is_file():
        raise FileNotFoundError(f"Steps file not found: {steps_path}")
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("step_index", -1)) == step_index:
            return row
    return None


def _find_villain_2(per_agent: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in per_agent:
        if entry.get("agent_id") == "villain_2":
            return entry
    return None


def _extract_obs_from_prompt(prompt: str | None) -> dict[str, Any]:
    if not prompt:
        raise ValueError("missing prompt on per_agent row")
    if USER_PROMPT_MARKER not in prompt:
        raise ValueError("prompt missing USER PROMPT marker")
    user_body = prompt.split(USER_PROMPT_MARKER, 1)[1].strip()
    payload = json.loads(user_body)
    obs = payload.get("obs")
    if not isinstance(obs, dict):
        raise ValueError("user prompt JSON has no 'obs' object")
    return obs


def _format_messages(obs: dict[str, Any]) -> str:
    msgs = obs.get("msgs") or []
    if not msgs:
        return "(none)"
    parts: list[str] = []
    for i, m in enumerate(msgs):
        tp = m.get("tp") if isinstance(m, dict) else None
        conf = m.get("c") if isinstance(m, dict) else None
        if isinstance(tp, (list, tuple)) and len(tp) >= 2:
            parts.append(f"msg[{i}] tp=[{tp[0]}, {tp[1]}] confidence={conf}")
        else:
            parts.append(f"msg[{i}] {m!r}")
    return "; ".join(parts)


def _pretty_raw_response(raw: str | None) -> str:
    if not raw:
        return "(no raw response)"
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return raw


def _hero_visible_label(entry: dict[str, Any]) -> str:
    v = entry.get("hero_truly_visible")
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return "unknown"


def _position_from_obs(obs: dict[str, Any]) -> tuple[float, float] | None:
    self_pos = obs.get("self")
    if isinstance(self_pos, (list, tuple)) and len(self_pos) >= 2:
        return float(self_pos[0]), float(self_pos[1])
    return None


def _call_groq(client: Groq, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return (response.choices[0].message.content or "").strip()


def _format_probe_section(
    *,
    map_name: str,
    step_index: int,
    obs: dict[str, Any],
    entry: dict[str, Any],
    free_text: str,
) -> str:
    pos = _position_from_obs(obs)
    pos_str = f"({pos[0]}, {pos[1]})" if pos else "(unknown)"
    msgs_n = int(entry.get("messages_received") or 0)
    steps_since = entry.get("steps_since_hero_seen")
    target = entry.get("llm_target_position")
    target_str = json.dumps(target) if target is not None else "null"

    lines = [
        f"## [{map_name}] — Step {step_index} — villain_2",
        "",
        "### Context",
        f"- Position: {pos_str}",
        f"- Messages received: {msgs_n}",
        f"- Message content: {_format_messages(obs)}",
        f"- Steps since hero seen: {steps_since}",
        f"- Hero visible: {_hero_visible_label(entry)}",
        "",
        "### What the constrained run produced",
        f"- intent: {entry.get('intent')}",
        f"- target: {target_str}",
        f"- movement_source: {entry.get('movement_source')}",
        "- raw response:",
        "```json",
        _pretty_raw_response(entry.get("raw_llm_response")),
        "```",
        "",
        "### Free text LLM response",
        free_text or "(empty response)",
        "",
        "### Analysis notes",
        "- Does it mention hero movement or just position?",
        "- Does it treat coordinates as stale or current?",
        "- Does it model hero as moving agent?",
        "- Does it mention teammate or coordination spontaneously?",
        "- Does it match or contradict the constrained run behavior?",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY in the environment.", file=sys.stderr)
        return 1

    client = Groq(api_key=api_key)
    md_parts: list[str] = [
        "# LLM observation probe results",
        "",
        f"Model: `{GROQ_MODEL}`  |  temperature: `0.3`  |  max_tokens: `600`",
        "",
    ]

    for probe in PROBES:
        map_name = probe["map"]
        steps_path = Path(probe["file"])
        for step_index in probe["steps"]:
            label = f"[{map_name}] step {step_index}"
            print(f"\n{'=' * 72}\nProbing {label} ...", flush=True)

            try:
                row = _load_step_row(steps_path, step_index)
                if row is None:
                    raise ValueError(f"step_index {step_index} not found in {steps_path}")

                entry = _find_villain_2(row.get("per_agent") or [])
                if entry is None:
                    raise ValueError("villain_2 not found in per_agent")

                obs = _extract_obs_from_prompt(entry.get("prompt"))
                obs_json = json.dumps(obs, indent=2, ensure_ascii=False)
                user_prompt = USER_PROMPT_TEMPLATE.format(obs_json=obs_json)

                free_text = _call_groq(client, user_prompt)
                section = _format_probe_section(
                    map_name=map_name,
                    step_index=step_index,
                    obs=obs,
                    entry=entry,
                    free_text=free_text,
                )
                md_parts.append(section)
                print(section, flush=True)
                print(f"OK {label}", flush=True)

            except Exception as exc:
                err_block = (
                    f"## [{map_name}] — Step {step_index} — villain_2\n\n"
                    f"**ERROR:** `{exc}`\n\n---\n\n"
                )
                md_parts.append(err_block)
                print(err_block, flush=True)
                print(f"ERROR {label}: {exc}", file=sys.stderr, flush=True)

    OUTPUT_MD.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\nWrote {OUTPUT_MD.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
