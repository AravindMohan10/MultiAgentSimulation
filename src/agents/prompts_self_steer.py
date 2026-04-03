"""
Minimal prompts for the self-steer agent track.

Facts-only system prompt (static map/obstacles once). Per-step user JSON with
delta observation plus a short recent[] memory of the agent's own past actions.
No tactical MUST rules or map hints in user messages.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import (
    AgentConfig,
    AgentState,
    AgentType,
    EnvironmentConfig,
    Message,
    Obstacle,
    Observation,
)


def _sort_agent_states(agents: List[AgentState]) -> List[AgentState]:
    return sorted(agents, key=lambda a: a.id)


def _sort_messages(messages: List[Message]) -> List[Message]:
    return sorted(
        messages,
        key=lambda m: (
            m.sender_id,
            tuple(m.recipient_ids or []),
            tuple(m.payload),
            m.channel or "",
        ),
    )


def _hero_visible_to_villain(observation: Observation) -> tuple[bool, Optional[Tuple[float, float]]]:
    sx = float(observation.self_state.position.x)
    sy = float(observation.self_state.position.y)
    sight_r2 = float(observation.villain_hero_sight_radius) ** 2
    for a in observation.visible_agents:
        if a.agent_type == AgentType.HERO and a.alive:
            dx = float(a.position.x) - sx
            dy = float(a.position.y) - sy
            if (dx * dx + dy * dy) <= sight_r2:
                return True, (float(a.position.x), float(a.position.y))
            return False, None
    return False, None


def _compress_visible_for_prompt(
    observation: Observation,
    *,
    is_villain: bool,
) -> List[Dict[str, Any]]:
    hero_visible = False
    if is_villain:
        hero_visible, _ = _hero_visible_to_villain(observation)

    visible = observation.visible_agents
    if is_villain and not hero_visible:
        visible = [a for a in visible if a.agent_type != AgentType.HERO]

    sx = float(observation.self_state.position.x)
    sy = float(observation.self_state.position.y)
    out: List[Dict[str, Any]] = []
    for a in _sort_agent_states(visible):
        ox = float(a.position.x)
        oy = float(a.position.y)
        dist = math.hypot(ox - sx, oy - sy)
        out.append(
            {
                "id": a.id,
                "r": a.agent_type.value,
                "p": [round(ox, 2), round(oy, 2)],
                "dist": round(dist, 2),
            }
        )
    return out


def _compress_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in _sort_messages(messages):
        payload = m.payload or []
        tp_x = float(payload[0]) if len(payload) >= 1 else 0.0
        tp_y = float(payload[1]) if len(payload) >= 2 else 0.0
        conf = float(payload[2]) if len(payload) >= 3 else 0.0
        rows.append({"tp": [tp_x, tp_y], "c": conf})
    return rows


def format_recent_line(
    *,
    step_index: int,
    intent: str,
    target: Optional[Tuple[float, float]],
    pos: Optional[Tuple[float, float]] = None,
    dist: Optional[float] = None,
    msg: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    tgt_part = "tgt=null"
    if target is not None:
        tgt_part = f"tgt=({target[0]:.1f},{target[1]:.1f})"

    pos_part = ""
    if pos is not None:
        pos_part = f" pos=({pos[0]:.1f},{pos[1]:.1f})"

    dist_part = f" dist={dist:.1f}" if dist is not None else " dist=null"

    msg_part = f" msg={msg}" if msg else " msg=none"

    note_part = ""
    if note and note.strip():
        note_part = f" note={note.strip()[:30]}"

    return f"[s{step_index}] intent={intent} {tgt_part}{pos_part}{dist_part}{msg_part}{note_part}"


def build_static_map_block(
    observation: Observation,
    env_config: EnvironmentConfig,
) -> str:
    w, h = float(env_config.world_size[0]), float(env_config.world_size[1])
    mt = observation.map_template or env_config.map_template.value
    lines = [
        f"Template: {mt}",
        f"World size: {w:g} x {h:g}",
    ]
    cp = observation.chokepoint_positions or env_config.chokepoint_positions or []
    if cp:
        cp_str = ", ".join(
            f"({float(p[0]):.1f},{float(p[1]):.1f})" for p in cp[:12]
        )
        lines.append(f"Chokepoints: {cp_str}")
    else:
        lines.append("Chokepoints: none")

    obs_lines: List[str] = []
    for o in observation.world_obstacles[:40]:
        ox = float(o.position.x)
        oy = float(o.position.y)
        r = float(o.radius)
        obs_lines.append(f"({ox:.1f},{oy:.1f},r={r:.1f})")
    if obs_lines:
        lines.append("Obstacles: " + "; ".join(obs_lines))
    else:
        lines.append("Obstacles: none")
    return "\n".join(lines)


def build_system_prompt(
    config: AgentConfig,
    env_config: EnvironmentConfig,
    *,
    capture_radius: float,
    map_block: str,
) -> str:
    role = "HERO" if config.agent_type == AgentType.HERO else "VILLAIN"
    if config.agent_type == AgentType.HERO:
        objective = (
            "You are the hero. Avoid being caught by villains for the full episode."
        )
    else:
        objective = "You are a villain. Coordinate with teammates to catch the hero."

    w, h = float(env_config.world_size[0]), float(env_config.world_size[1])
    ms = float(config.max_speed or 1.0)

    comm = ""
    if config.agent_type == AgentType.VILLAIN and config.communication_enabled:
        comm = (
            "\n[COMMUNICATION]\n"
            "You may send an optional message: "
            '{"message": {"payload": [hero_x, hero_y, confidence, self_x, self_y], "channel": "coord"}}\n'
            "Or shorthand list [hero_x, hero_y, confidence, self_x, self_y]. "
            "Only send when you choose to share information.\n"
        )

    return (
        f"[ROLE]\nYou are {role}. {objective}\n\n"
        f"[WORLD]\n"
        f"Size: {w:g}x{h:g}. Capture radius: {capture_radius:g}. Max speed: {ms:g}.\n"
        f"Keep target_position inside [{1:g}, {w - 1:g}] x [{1:g}, {h - 1:g}].\n\n"
        f"[MAP]\n{map_block}\n"
        f"{comm}\n"
        "[OUTPUT]\n"
        "Each step reply with one JSON object only:\n"
        '{"intent": str, "target_position": [x, y], "message": object or list or null, '
        '"note": str or null}\n'
        "intent is a short snake_case label you choose (e.g. pursue_target, flee_threat, "
        "search_systematic, hold_chokepoint).\n"
        '"note": str or null — a private one-sentence reminder visible only to you on your next step, if you want to leave one.\n'
    )


def build_delta_obs(
    observation: Observation,
    config: AgentConfig,
    *,
    steps_since_hero_seen: int,
    last_seen_hero: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    is_villain = config.agent_type == AgentType.VILLAIN
    hero_visible = False
    if is_villain:
        hero_visible, hero_pos = _hero_visible_to_villain(observation)
        if hero_visible and hero_pos is not None:
            last_seen = hero_pos
        else:
            last_seen = last_seen_hero
    else:
        last_seen = None
        hero_visible = any(
            a.agent_type == AgentType.VILLAIN and a.alive for a in observation.visible_agents
        )

    payload: Dict[str, Any] = {
        "step": int(observation.step_index),
        "pos": [
            round(float(observation.self_state.position.x), 3),
            round(float(observation.self_state.position.y), 3),
        ],
        "vis": _compress_visible_for_prompt(observation, is_villain=is_villain),
        "msgs": _compress_messages(observation.incoming_messages),
        "feedback": observation.last_action_feedback,
    }
    if is_villain:
        payload["hero_visible"] = bool(hero_visible)
        payload["steps_since_hero_seen"] = int(steps_since_hero_seen)
        if last_seen is not None:
            payload["last_seen_hero"] = [round(last_seen[0], 3), round(last_seen[1], 3)]
        else:
            payload["last_seen_hero"] = None
    else:
        payload["villains_visible"] = sum(
            1 for a in observation.visible_agents if a.agent_type == AgentType.VILLAIN
        )
    return payload


def build_user_prompt(
    observation: Observation,
    config: AgentConfig,
    *,
    steps_since_hero_seen: int,
    last_seen_hero: Optional[Tuple[float, float]],
    recent_lines: List[str],
) -> str:
    obs = build_delta_obs(
        observation,
        config,
        steps_since_hero_seen=steps_since_hero_seen,
        last_seen_hero=last_seen_hero,
    )
    user_payload: Dict[str, Any] = {"obs": obs}
    if recent_lines:
        user_payload["recent"] = recent_lines
    return json.dumps(user_payload, separators=(",", ":"), ensure_ascii=False)


def build_combined_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "SYSTEM PROMPT\n"
        "-------------\n"
        f"{system_prompt}\n\n"
        "USER PROMPT\n"
        "-----------\n"
        f"{user_prompt}"
    )
