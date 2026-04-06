"""
Prompt building for isolated LLM agents.

This module turns an agent's own Observation plus its private AgentSession into
neutral, role-based prompts. The design goals are:

- fairness: same structure for hero and villain agents
- isolation: never include another agent's private context
- determinism: stable serialization so runs are reproducible
- future-proofing: observation and actions stay 3D-compatible even when the
  current environment is only 2D

No persona language is used here. The only role-specific difference is the
agent's objective (hero avoids capture, villain captures the hero).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import (
    AgentConfig,
    AgentState,
    AgentType,
    EnvironmentConfig,
    MapTemplate,
    Message,
    Obstacle,
    Observation,
)
from .schema import ALLOWED_INTENT_VALUES
from .session import AgentSession


@dataclass(slots=True)
class PromptBundle:
    """Container for the two prompts we send to the LLM backend."""

    system_prompt: str
    user_prompt: str


def _dump_model(model: Any) -> Dict[str, Any]:
    """
    Serialize a pydantic model into JSON-compatible primitives.

    We prefer mode='json' for stable output, but fall back to the default dump
    if the runtime pydantic version or model type does not support it.
    """
    if hasattr(model, "model_dump"):
        try:
            return model.model_dump(mode="json")
        except TypeError:
            return model.model_dump()
    if isinstance(model, dict):
        return dict(model)
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def _sort_agent_states(agents: List[AgentState]) -> List[AgentState]:
    """Sort visible agents for deterministic prompt ordering."""
    return sorted(agents, key=lambda agent: agent.id)


def _sort_messages(messages: List[Message]) -> List[Message]:
    """Sort incoming messages for deterministic prompt ordering."""
    return sorted(
        messages,
        key=lambda message: (
            message.sender_id,
            tuple(message.recipient_ids or []),
            tuple(message.payload),
            message.channel or "",
        ),
    )


def _segment_intersects_circle(
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    cx: float,
    cy: float,
    circle_radius: float,
) -> bool:
    """True if segment (p1,p2) intersects the closed disk centered at (cx,cy)."""
    dx = p2x - p1x
    dy = p2y - p1y
    fx = p1x - cx
    fy = p1y - cy
    a = dx * dx + dy * dy
    if a < 1e-18:
        return fx * fx + fy * fy <= circle_radius * circle_radius
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - circle_radius * circle_radius
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    sqrt_disc = math.sqrt(max(0.0, disc))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    def _seg_hit(t: float) -> bool:
        return 0.0 <= t <= 1.0

    return _seg_hit(t1) or _seg_hit(t2)


def _label_chokepoint(index: int, map_template: str) -> str:
    if map_template == MapTemplate.HUB_AND_SPOKES.value:
        return f"spoke_{index}_entrance"
    if map_template == MapTemplate.ASYMMETRIC_LABYRINTH.value:
        return "bridge"
    return f"chokepoint_{index}"


def _get_nearby_obstacle_info(
    agent_pos: Tuple[float, float],
    obstacles: List[Obstacle],
    hero_last_known_pos: Optional[Tuple[float, float]],
    chokepoint_positions: List[Tuple[float, float]],
    *,
    sight_radius: float = 20.0,
    max_obstacles: int = 6,
) -> List[Dict[str, Any]]:
    ax, ay = float(agent_pos[0]), float(agent_pos[1])
    nearby: List[Dict[str, Any]] = []
    for obs in obstacles:
        ox, oy = float(obs.position.x), float(obs.position.y)
        dist = math.hypot(ax - ox, ay - oy)
        if dist > sight_radius:
            continue
        blocks_los = False
        if hero_last_known_pos is not None:
            hx, hy = float(hero_last_known_pos[0]), float(hero_last_known_pos[1])
            blocks_los = _segment_intersects_circle(
                ax, ay, hx, hy, ox, oy, float(obs.radius)
            )
        near_chokepoint = any(
            math.hypot(ox - float(cp[0]), oy - float(cp[1])) < 8.0
            for cp in chokepoint_positions
        )
        nearby.append(
            {
                "pos": [round(ox, 1), round(oy, 1)],
                "dist": round(dist, 1),
                "blocks_los_to_hero": blocks_los,
                "near_chokepoint": near_chokepoint,
            }
        )
    nearby.sort(key=lambda x: float(x["dist"]))
    return nearby[:max_obstacles]


def serialize_observation(
    observation: Observation,
    *,
    hero_last_known_pos: Optional[Tuple[float, float]] = None,
    villain_behavior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert an Observation into a compact, deterministic JSON payload.

    Only data visible to the agent is included here. This keeps the prompt
    fair and prevents leakage of global or other-agent private state.
    """
    # Minimize prompt bloat: keep only decision-relevant state.
    # - Use 2D positions only (z is always 0 in this environment).
    # - Summarize visible agents as {pos, role}.
    # - Compress incoming messages into {target_pos, confidence}.
    sight_r = float(observation.villain_hero_sight_radius)

    is_villain = observation.self_state.agent_type == AgentType.VILLAIN

    # Partial observability: villains only treat the hero as "visible" when
    # the hero is within villain_hero_sight_radius (from EnvironmentConfig).
    # When not visible, we remove hero position from the prompt payload.
    hero_visible = False
    if is_villain:
        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)
        for a in observation.visible_agents:
            if a.agent_type == AgentType.HERO and a.alive:
                dx = float(a.position.x) - sx
                dy = float(a.position.y) - sy
                hero_visible = (dx * dx + dy * dy) <= (sight_r * sight_r)
                break

    # Filter what goes into the prompt.
    visible_for_prompt = observation.visible_agents
    if is_villain and not hero_visible:
        visible_for_prompt = [a for a in observation.visible_agents if a.agent_type != AgentType.HERO]
    def _pos2(agent_state: AgentState) -> list[float]:
        return [float(agent_state.position.x), float(agent_state.position.y)]

    def _compress_visible(agent_state: AgentState) -> Dict[str, Any]:
        return {"p": _pos2(agent_state), "r": agent_state.agent_type.value}

    def _compress_message(m: Message) -> Dict[str, Any]:
        # Protocol expected by existing coordinated comms:
        # payload = [target_x, target_y, confidence, self_x, self_y]
        payload = m.payload or []
        tp_x = float(payload[0]) if len(payload) >= 1 else 0.0
        tp_y = float(payload[1]) if len(payload) >= 2 else 0.0
        conf = float(payload[2]) if len(payload) >= 3 else 0.0

        return {"tp": [tp_x, tp_y], "c": conf}

    ax = float(observation.self_state.position.x)
    ay = float(observation.self_state.position.y)
    cp_list = list(observation.chokepoint_positions or [])
    nearby = _get_nearby_obstacle_info(
        (ax, ay),
        list(observation.world_obstacles),
        hero_last_known_pos,
        cp_list,
        sight_radius=20.0,
        max_obstacles=6,
    )
    mt = observation.map_template
    map_context = {
        "template": mt,
        "chokepoints": [
            {"pos": [float(cp[0]), float(cp[1])], "label": _label_chokepoint(i, mt)}
            for i, cp in enumerate(cp_list)
        ],
    }

    payload: Dict[str, Any] = {
        "si": int(observation.step_index),
        "self": [ax, ay],
        "vis": [_compress_visible(a) for a in _sort_agent_states(visible_for_prompt)],
        "msgs": [_compress_message(m) for m in _sort_messages(observation.incoming_messages)],
        "nearby_obstacles": nearby,
        "map_context": map_context,
        # Dynamic partial observability hint for the model.
        "hero_visible": bool(hero_visible),
        "hero_vision_instruction": (
            "You can see the hero. Use their position to pursue or intercept."
            if hero_visible
            else "You cannot see the hero. Explore the environment to locate them."
        )
        if is_villain
        else "",
    }
    if villain_behavior:
        payload["villain_behavior"] = villain_behavior
    return payload


def serialize_session(session: AgentSession, limit: int | None = None) -> Dict[str, Any]:
    """
    Serialize only the private context for this one agent.

    We intentionally omit raw prompt text and raw responses from the prompt
    input to reduce prompt bloat and to keep the model focused on summarized
    private memory and recent structured turns.
    """
    # Keep only the minimal private context that can help recover from partial
    # observability and short comms delays.
    if session.config.disable_memory:
        return {"m": "", "lv": None}

    mem = (session.memory_summary or "").strip()
    # Bound memory_summary length to keep prompts stable across episodes.
    if len(mem) > 120:
        mem = mem[-120:]

    lv = session.last_valid_action
    last_action = None
    if lv is not None:
        last_action = {
            "move": [float(lv.movement.x), float(lv.movement.y), float(lv.movement.z)],
            "intent": lv.intent,
        }
    return {"m": mem, "lv": last_action}


def _role_objective(config: AgentConfig) -> str:
    """Neutral objective text based only on the agent type."""
    if config.agent_type.value == "hero":
        return "maximize survival time and avoid capture"
    return "capture the hero"


def _v2_map_guidance(config: AgentConfig, env: EnvironmentConfig) -> str:
    """Extra map-aware tactics for V2_GUIDED on structured maps."""
    if env.map_template == MapTemplate.HUB_AND_SPOKES:
        if config.agent_type == AgentType.VILLAIN:
            return (
                "\nMAP AWARENESS — HUB AND SPOKES:\n"
                "The map has a central open hub with spoke corridors radiating outward.\n"
                "Hero can enter any spoke and immediately break your line of sight.\n"
                "CRITICAL: You and your teammate cannot cover all spoke entrances alone.\n"
                "Use communication to tell your teammate which spoke the hero entered.\n"
                "Optimal split: one villain guards hub center, one pursues into spoke.\n"
                "If hero is not visible and no message received, move to nearest "
                "spoke entrance and wait — do not wander randomly.\n"
            )
        return (
            "\nMAP AWARENESS — HUB AND SPOKES:\n"
            "You can use spoke corridors to break villain line of sight instantly.\n"
            "Entering a spoke forces pursuing villains to commit to that spoke.\n"
            "If a villain guards hub center, use a far spoke to maximize distance.\n"
            "Do not stay in hub — you are visible from all directions there.\n"
        )
    if env.map_template == MapTemplate.ASYMMETRIC_LABYRINTH:
        if config.agent_type == AgentType.VILLAIN:
            return (
                "\nMAP AWARENESS — ASYMMETRIC LABYRINTH:\n"
                "Left half is open terrain (fast movement).\n"
                "Right half is dense labyrinth (slow movement, many hiding spots).\n"
                "The two halves connect through a single bridge chokepoint.\n"
                "Optimal split: one villain patrols bridge to block crossing, "
                "one pursues hero in whichever zone hero is currently in.\n"
                "If hero crosses to labyrinth side, the bridge-guardian villain "
                "should follow and swap roles with pursuing villain.\n"
            )
        return (
            "\nMAP AWARENESS — ASYMMETRIC LABYRINTH:\n"
            "The labyrinth side gives you hiding opportunities but slows movement.\n"
            "The bridge is a chokepoint — if a villain guards it, you are trapped on one side.\n"
            "Monitor bridge position before committing to a side.\n"
            "Open side: you move faster but are more visible.\n"
            "Labyrinth side: slower but harder to chase.\n"
        )
    return ""


def build_system_prompt(
    config: AgentConfig,
    environment_config: EnvironmentConfig | None = None,
) -> str:
    """
    Build a neutral system prompt shared across all agents except role objective.

    No persona language is used. The prompt only establishes the task, the
    output format, and the validation constraints.
    """
    role_label = "HERO" if config.agent_type.value == "hero" else "VILLAIN"
    base = (
        f"You are the {role_label}.\n"
        "You are an intelligent agent in a multi-agent pursuit-evasion environment.\n"
        "\n"
        "Your role determines your goal:\n"
        "\t• HERO: Avoid capture. Continuously move away from the closest villain. You MUST move every step.\n"
        "\t• VILLAIN: Work with other villains to capture the hero by predicting the hero’s movement and intercepting rather than directly chasing.\n"
        "\n"
        "STRICT RULES:\n"
        "\t• Provide target_position [x,y] whenever possible; the simulator moves you toward it.\n"
        "\t• If you use legacy movement instead, it MUST NOT be [0,0,0] and magnitude should be significant.\n"
        "\t• Every step MUST result in an adaptive intent/target based on positions.\n"
        "\t• Standing still is NEVER allowed (always choose a meaningful target or direction).\n"
        "\t• If you are a villain:\n"
        "\t\t• One agent should pursue directly\n"
        "\t\t• One agent should attempt to cut off escape paths\n"
        "\t\t• Do NOT repeat identical movement patterns every step.\n"
        "\n"
        "SPATIAL REASONING:\n"
        "\t• Use positions to decide direction\n"
        "\t• Move AWAY (hero) or INTERCEPT (villains), not randomly\n"
        "\n"
        "Output ONLY valid JSON (one object):\n"
        "{\n"
        '  "intent": "string",\n'
        '  "target_position": [float, float],\n'
        '  "target_description": "string or null",\n'
        '  "confidence": float,\n'
        '  "movement": [float, float, float] or null\n'
        "}\n"
        "\n"
        "Rules:\n"
        "\t• Prefer target_position [x, y] in world coordinates (movement toward that point is executed by the simulator).\n"
        "\t• movement is deprecated: include only if you cannot specify a target; otherwise set movement to null.\n"
        "\t• confidence is in [0, 1].\n"
        "\n"
        "No explanations. No extra text.\n"
    )
    if config.agent_type == AgentType.VILLAIN and not config.disable_messages:
        base += (
            "\nCOMMUNICATION (VILLAIN only):\n"
            "If you send a teammate message, use this exact JSON shape (field \"message\" is an object):\n"
            '  "message": {"payload": [hero_x, hero_y, confidence, self_x, self_y], "channel": "coord"}\n'
            "• payload: exactly five numbers — hero world position (or 0,0 if unseen), confidence 0..1, your position.\n"
            "You may also use the shorthand list form [hero_x, hero_y, confidence, self_x, self_y] as \"message\" "
            "(the simulator accepts both).\n"
        )
    if (
        config.prompt_version == "V2_GUIDED"
        and not config.disable_guidance
        and environment_config is not None
        and environment_config.map_template
        in (MapTemplate.HUB_AND_SPOKES, MapTemplate.ASYMMETRIC_LABYRINTH)
    ):
        base += _v2_map_guidance(config, environment_config)
    if (
        config.prompt_version == "V2_GUIDED"
        and not config.disable_guidance
        and config.agent_type == AgentType.HERO
    ):
        base += (
            "\nMULTI-VILLAIN MODELING (THEORY OF MIND):\n"
            "You can infer villain intentions from their movement patterns.\n"
            "Use the structured field obs.villain_behavior[*].inferred_intent to anticipate strategy.\n"
            "If one villain is flanking and one approaching, move perpendicular to the flanker's "
            "recent movement vector when possible to break a pincer.\n"
        )
    return base


def build_user_prompt(
    observation: Observation,
    session: AgentSession,
    *,
    history_limit: int | None = None,
    hero_last_known_pos: Optional[Tuple[float, float]] = None,
    villain_behavior: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the agent-specific user prompt.

    This prompt contains only the agent's own observation plus that agent's
    private session context. It never includes another agent's private history.
    """
    # Plain JSON context only (no markdown, no wrapper text).
    payload = {
        "obs": serialize_observation(
            observation,
            hero_last_known_pos=hero_last_known_pos,
            villain_behavior=villain_behavior,
        ),
        "priv": serialize_session(session, limit=history_limit),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def build_prompt_bundle(
    observation: Observation,
    session: AgentSession,
    *,
    history_limit: int | None = None,
    environment_config: EnvironmentConfig | None = None,
    hero_last_known_pos: Optional[Tuple[float, float]] = None,
    villain_behavior: Optional[Dict[str, Any]] = None,
) -> PromptBundle:
    """Convenience helper that returns both prompts together."""
    return PromptBundle(
        system_prompt=build_system_prompt(session.config, environment_config),
        user_prompt=build_user_prompt(
            observation,
            session,
            history_limit=history_limit,
            hero_last_known_pos=hero_last_known_pos,
            villain_behavior=villain_behavior,
        ),
    )

