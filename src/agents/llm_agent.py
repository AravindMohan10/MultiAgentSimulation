"""
LLM-backed agent implementation.

This agent is designed for the research use case where each agent is an
independent LLM instance with its own private session/history and strict
prompt/output validation.

Key properties:
- isolation: each agent keeps its own AgentSession
- fairness: prompt structure comes from prompts.py and is identical across roles
- robustness: JSON extraction, Pydantic validation, timeout, and retries
- smoothness: per-call timeout prevents a single slow model response from
  stalling the agent indefinitely
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from ..core.models import Action, AgentConfig, AgentType, Message, Observation, Vec3
from .base import BaseAgent
from .prompts import (
    PromptBundle,
    build_prompt_bundle,
    build_system_prompt,
    serialize_observation,
    serialize_session,
)
from .schema import LLMActionOutput, llm_action_to_action
from .session import AgentSession


@runtime_checkable
class LLMClient(Protocol):
    """
    Minimal client contract for an LLM backend.

    Implementations can be synchronous or asynchronous. The agent will accept
    either a method named `complete(system_prompt, user_prompt)` or a plain
    callable with the same signature.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> Any:
        ...


@dataclass(slots=True)
class LLMCallResult:
    """Internal helper for carrying an LLM call result or an exception."""

    value: Optional[str] = None
    error: Optional[BaseException] = None


def _call_client(client: Any, system_prompt: str, user_prompt: str) -> str:
    """
    Call a client object or plain callable and normalize async/sync responses.
    """
    if hasattr(client, "complete") and callable(getattr(client, "complete")):
        response = client.complete(system_prompt, user_prompt)
    else:
        response = client(system_prompt, user_prompt)

    if inspect.isawaitable(response):
        response = asyncio.run(response)

    if not isinstance(response, str):
        raise TypeError(
            f"LLM client returned {type(response)!r}, expected str."
        )
    return response


def _run_with_timeout(
    fn: Callable[[], str],
    timeout_seconds: float,
) -> str:
    """
    Run a blocking LLM call in a daemon thread and wait at most timeout_seconds.

    We cannot forcibly kill Python threads, so timeout means we stop waiting and
    return control to the caller. The thread is daemonized so it cannot block
    process shutdown.
    """
    result_queue: Queue[LLMCallResult] = Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(LLMCallResult(value=fn()))
        except BaseException as exc:  # pragma: no cover - defensive boundary
            result_queue.put(LLMCallResult(error=exc))

    thread = threading.Thread(
        target=worker,
        name="llm-agent-call",
        daemon=True,
    )
    thread.start()

    try:
        result = result_queue.get(timeout=timeout_seconds)
    except Empty as exc:
        raise TimeoutError(
            f"LLM call exceeded timeout of {timeout_seconds:.3f} seconds."
        ) from exc

    if result.error is not None:
        raise result.error
    if result.value is None:
        raise ValueError("LLM call returned no value.")
    return result.value


def _build_combined_prompt(bundle: PromptBundle) -> str:
    """Store the exact prompt context sent to the model for auditability."""
    return (
        "SYSTEM PROMPT\n"
        "-------------\n"
        f"{bundle.system_prompt}\n\n"
        "USER PROMPT\n"
        "-----------\n"
        f"{bundle.user_prompt}"
    )


def _extract_json_candidate(raw_text: str) -> str:
    """
    Extract a likely JSON object from an LLM response.

    The model is instructed to return JSON only, but we still defensively accept
    code fences or extra text around the JSON object.
    """
    text = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match is not None:
        text = fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM output does not contain a JSON object.")
    return text[start : end + 1]


def _fold_numeric_expressions_in_json_text(s: str) -> str:
    """
    LLMs sometimes emit invalid JSON like [146.1 + 0.25, 155.0 + 1] instead of computed floats.
    Repeatedly fold binary sub-expressions until stable so json.loads succeeds.
    """
    num = r"\d+\.?\d*(?:[eE][+-]?\d+)?"
    pat = re.compile(rf"({num})\s*([+\-*/])\s*({num})")

    def _fold_once(text: str) -> str:
        def repl(m: Any) -> str:
            a, op, b = m.group(1), m.group(2), m.group(3)
            try:
                va, vb = float(a), float(b)
                if op == "+":
                    r = va + vb
                elif op == "-":
                    r = va - vb
                elif op == "*":
                    r = va * vb
                else:
                    r = va / vb if abs(vb) > 1e-18 else 0.0
                if math.isfinite(r):
                    return str(r)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
            return m.group(0)

        return pat.sub(repl, text)

    prev = None
    out = s
    safety = 0
    while prev != out and safety < 256:
        prev = out
        out = _fold_once(out)
        safety += 1
    return out


def _parse_llm_output(raw_text: str) -> LLMActionOutput:
    """
    Parse raw model text into the strict Pydantic schema.
    """
    candidate = _fold_numeric_expressions_in_json_text(_extract_json_candidate(raw_text))
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("LLM output JSON must be an object.")
    return LLMActionOutput.model_validate(payload)


def _parse_llm_output_with_raw(raw_text: str) -> tuple[LLMActionOutput, Optional[str]]:
    """Parse LLM JSON and preserve the pre-normalization intent string for logging."""
    candidate = _fold_numeric_expressions_in_json_text(_extract_json_candidate(raw_text))
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("LLM output JSON must be an object.")
    raw_intent: Optional[str] = None
    if "intent" in payload:
        raw_intent = str(payload.get("intent"))
    return LLMActionOutput.model_validate(payload), raw_intent


_FALLBACK_DIRS: list[tuple[float, float]] = [
    (1.0, 0.0),
    (0.70710678, 0.70710678),
    (0.0, 1.0),
    (-0.70710678, 0.70710678),
    (-1.0, 0.0),
    (-0.70710678, -0.70710678),
    (0.0, -1.0),
    (0.70710678, -0.70710678),
]

def _stable_seed(agent_id: str) -> int:
    """Deterministic, non-hash-based seed for exploration direction."""
    return sum(ord(c) for c in agent_id)


def _exploration_move(session: AgentSession, step_index: int) -> Vec3:
    """Deterministic lightweight exploration to avoid [0,0,0] deadlocks."""
    max_speed = float(getattr(session.config, "max_speed", 1.0) or 1.0)
    scale = min(1.0, max_speed) if max_speed > 0 else 1.0
    idx = (_stable_seed(session.agent_id) + int(step_index)) % len(_FALLBACK_DIRS)
    dx, dy = _FALLBACK_DIRS[idx]
    return Vec3(x=dx * scale, y=dy * scale, z=0.0)


# Reject targets in a 1-unit margin from world edges so agents don't chase boundary points.
_TARGET_POSITION_BOUNDARY_BUFFER = 1.0
# Targets closer than this to current position are treated as non-goals (e.g. LLM echoing own pos).
_MEANINGFUL_TARGET_MIN_SEPARATION = 5.0


def _is_valid_world_target(
    target_pos: tuple[float, float],
    world_size: tuple[float, float],
    *,
    boundary_buffer: float = _TARGET_POSITION_BOUNDARY_BUFFER,
) -> bool:
    """
    Reject [0,0] (common LLM default) and targets outside the safe interior.

    Valid region: ``[boundary_buffer, world_size - boundary_buffer]`` on each axis
    (inclusive), so agents are not steered to edges or slightly outside the map.
    """
    x, y = float(target_pos[0]), float(target_pos[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    if x == 0.0 and y == 0.0:
        return False
    w, h = float(world_size[0]), float(world_size[1])
    b = max(0.0, float(boundary_buffer))
    # Degenerate world: no interior
    if w <= 2.0 * b or h <= 2.0 * b:
        return False
    if x < b or x > w - b or y < b or y > h - b:
        return False
    return True


def _is_meaningful_target(
    current_pos: tuple[float, float],
    target_pos: tuple[float, float],
    *,
    min_separation: float = _MEANINGFUL_TARGET_MIN_SEPARATION,
) -> bool:
    """
    True if the target is far enough from the agent's current position to be a real goal.

    If ``dist(current, target) < min_separation`` (default 5.0), returns False so we
    strip ``target_position`` (e.g. villain LLM echoing own coordinates).
    """
    cx, cy = float(current_pos[0]), float(current_pos[1])
    tx, ty = float(target_pos[0]), float(target_pos[1])
    if not all(math.isfinite(v) for v in (cx, cy, tx, ty)):
        return False
    d = math.hypot(tx - cx, ty - cy)
    if not math.isfinite(d):
        return False
    ms = float(min_separation)
    return d >= ms


def validate_target_position(tp: Any) -> Optional[list[float]]:
    """
    Normalize Groq JSON ``target_position`` to ``[x, y]`` or ``None`` if invalid.

    Used by ``GroqClient`` after parsing; rejects non-length-2 arrays, ``[0,0]``,
    non-finite values, and coordinates outside the buffered interior of the default
    160×160 world (see ``_TARGET_POSITION_BOUNDARY_BUFFER``).
    """
    if tp is None:
        return None
    if not isinstance(tp, (list, tuple)) or len(tp) != 2:
        return None
    try:
        x, y = float(tp[0]), float(tp[1])
    except (TypeError, ValueError):
        return None
    if not _is_valid_world_target((x, y), (160.0, 160.0)):
        return None
    return [x, y]


def _compute_movement_from_target(
    cur_x: float,
    cur_y: float,
    target_x: float,
    target_y: float,
    max_speed: float,
) -> Vec3:
    """Pure geometry: unit direction from current position to target, scaled to max_speed."""
    dx = float(target_x) - float(cur_x)
    dy = float(target_y) - float(cur_y)
    n = math.hypot(dx, dy)
    eps = 1e-9
    if not math.isfinite(n) or n < eps:
        return Vec3(x=0.0, y=0.0, z=0.0)
    ms = float(max_speed) if math.isfinite(float(max_speed)) else 1.0
    if ms <= 0.0:
        return Vec3(x=0.0, y=0.0, z=0.0)
    return Vec3(x=(dx / n) * ms, y=(dy / n) * ms, z=0.0)


def _apply_boundary_constraint(
    movement: Vec3,
    pos_x: float,
    pos_y: float,
    world_w: float,
    world_h: float,
    margin: float = 3.0,
) -> Vec3:
    """
    Hard post-constraint: adjust displacement so the next position stays inside
    [margin, world_size-margin] on each axis (clamped end position, then delta).
    """
    m = max(0.0, float(margin))
    w = float(world_w)
    h = float(world_h)
    min_x = m
    min_y = m
    max_x = max(min_x, w - m)
    max_y = max(min_y, h - m)
    nx = float(pos_x) + float(movement.x)
    ny = float(pos_y) + float(movement.y)
    nx2 = max(min_x, min(max_x, nx))
    ny2 = max(min_y, min(max_y, ny))
    return Vec3(x=nx2 - float(pos_x), y=ny2 - float(pos_y), z=0.0)


def _fallback_action(session: AgentSession, step_index: int, intent: str) -> Action:
    return Action(
        movement=_exploration_move(session, step_index),
        message=None,
        intent=intent,
        movement_source="fallback_explore",
    )


def _fallback_from_last_valid(
    session: AgentSession,
    observation: Observation,
    intent: str,
) -> Action:
    """
    Fallback policy:
    - If last_valid_action is non-trivial, reuse it (continuation).
    - Else, try a deterministic target-aware move using visible agents.
    - Else, use deterministic lightweight exploration.
    """
    last = session.last_valid_action
    if last is not None:
        dx = float(last.movement.x)
        dy = float(last.movement.y)
        if dx * dx + dy * dy > 1e-8:
            return Action(
                movement=last.movement,
                message=None,
                intent=LLMActionOutput.normalize_intent(intent),
                movement_source="fallback_last_valid",
            )

    # Target-aware fallback (deterministic if visibility is identical).
    self_x = float(observation.self_state.position.x)
    self_y = float(observation.self_state.position.y)

    if session.config.agent_type == AgentType.HERO:
        target_type = AgentType.VILLAIN
        away = True
    else:
        target_type = AgentType.HERO
        away = False

    sight_r = float(observation.villain_hero_sight_radius)
    sight_r2 = sight_r * sight_r
    candidates = [
        a
        for a in observation.visible_agents
        if a.agent_type == target_type
        and a.alive
        and (
            # For partial observability: villains may only target the hero
            # when the hero is within villain_hero_sight_radius (from env).
            not (session.config.agent_type == AgentType.VILLAIN and target_type == AgentType.HERO)
            or (
                (float(a.position.x) - self_x) ** 2
                + (float(a.position.y) - self_y) ** 2
                <= sight_r2
            )
        )
    ]
    if candidates:
        def _dist2(a: Any) -> float:
            tx = float(a.position.x)
            ty = float(a.position.y)
            return (tx - self_x) ** 2 + (ty - self_y) ** 2

        candidates_sorted = sorted(candidates, key=lambda a: (_dist2(a), a.id))
        tgt = candidates_sorted[0]
        tx = float(tgt.position.x)
        ty = float(tgt.position.y)

        vx = (self_x - tx) if away else (tx - self_x)
        vy = (self_y - ty) if away else (ty - self_y)
        v_len2 = vx * vx + vy * vy
        if v_len2 > 1e-8:
            v_len = math.sqrt(v_len2)
            v_unit_x = vx / v_len
            v_unit_y = vy / v_len
            max_speed = float(getattr(session.config, "max_speed", 1.0) or 1.0)
            scale = min(1.0, max_speed) if max_speed > 0 else 1.0
            return Action(
                movement=Vec3(x=v_unit_x * scale, y=v_unit_y * scale, z=0.0),
                message=None,
                intent=LLMActionOutput.normalize_intent(intent),
                movement_source="fallback_last_valid",
            )

    return _fallback_action(session, observation.step_index, intent)


class LLMAgent(BaseAgent):
    """
    LLM-backed agent with private session state.

    Each instance is intended to be used for exactly one agent id. The session
    keeps prompt history and memory isolated so no information leaks across
    agents even when they share the same API key/provider.
    """

    def __init__(
        self,
        config: AgentConfig,
        client: Any,
        *,
        session: Optional[AgentSession] = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        history_limit: Optional[int] = None,
        environment_config: Optional[EnvironmentConfig] = None,
    ) -> None:
        super().__init__(config)
        self._client = client
        self._environment_config = environment_config
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._max_retries = max(0, int(max_retries))
        self._output_parser = os.environ.get("LLM_OUTPUT_PARSER", "pydantic").strip().lower()

        if session is None:
            self._session = AgentSession(
                agent_id=config.id,
                config=config,
                history_limit=history_limit if history_limit is not None else 8,
            )
        else:
            if session.agent_id != config.id:
                raise ValueError(
                    "AgentSession agent_id must match the AgentConfig id."
                )
            self._session = session
            if history_limit is not None:
                self._session.history_limit = max(1, int(history_limit))

        # Movement stability guard: if the final movement direction becomes
        # near-constant for many consecutive steps, temporarily ignore the
        # LLM direction and use pure geometric pressure toward the hero.
        self._prev_final_dir_unit: Optional[tuple[float, float]] = None
        self._constant_dir_steps: int = 0

        # Short-term memory (villains only):
        # used to persist pursuit after losing sight of the hero, and to drive
        # structured exploration when the hero has been unseen for a while.
        self.last_seen_hero_position: Optional[tuple[float, float]] = None
        self.steps_since_seen: int = 999

        # Hero-only: track villain motion for naive "theory of mind" scaffolding.
        self._villain_movement_history: dict[str, list[list[float]]] = {}
        self._villain_inferred_intent: dict[str, str] = {}

        # Hero-only: recent LLM target points (target-cluster oscillation; separate from position).
        self._hero_llm_targets: list[tuple[float, float]] = []
        # Hero-only: recent actual positions — tight cluster => physical oscillation / stuck jiggle.
        self._hero_pos_history: list[tuple[float, float]] = []
        # Hero-only: short stuck-recovery nudge (8-dir, away from nearest villain) for exactly 5 steps.
        self._hero_stuck_recovery_remaining: int = 0
        self._hero_stuck_recovery_unit: tuple[float, float] = (1.0, 0.0)

    @property
    def session(self) -> AgentSession:
        """Private session for this agent only."""
        return self._session

    def reset(self) -> None:
        """Reset the private per-agent runtime state between episodes."""
        self._session.reset()
        self._prev_final_dir_unit = None
        self._constant_dir_steps = 0

        self.last_seen_hero_position = None
        self.steps_since_seen = 999

        self._villain_movement_history.clear()
        self._villain_inferred_intent.clear()
        self._hero_llm_targets.clear()
        self._hero_pos_history.clear()
        self._hero_stuck_recovery_remaining = 0
        self._hero_stuck_recovery_unit = (1.0, 0.0)

    def _update_hero_villain_tracking(self, observation: Observation) -> None:
        """Hero only: keep short position history for visible villains."""
        if self.config.agent_type != AgentType.HERO:
            return
        hx = float(observation.self_state.position.x)
        hy = float(observation.self_state.position.y)
        for a in observation.visible_agents:
            if a.agent_type != AgentType.VILLAIN or not a.alive:
                continue
            vid = a.id
            px = float(a.position.x)
            py = float(a.position.y)
            hist = self._villain_movement_history.setdefault(vid, [])
            hist.append([px, py])
            if len(hist) > 5:
                del hist[: len(hist) - 5]

            # Naive intent from last movement vs direction to hero.
            tohx = hx - px
            tohy = hy - py
            thn = math.hypot(tohx, tohy)
            if thn < 1e-12:
                self._villain_inferred_intent[vid] = "unknown"
                continue
            tox, toy = tohx / thn, tohy / thn
            if len(hist) < 2:
                self._villain_inferred_intent[vid] = "unknown"
                continue
            p0x, p0y = hist[-2][0], hist[-2][1]
            mvx = px - p0x
            mvy = py - p0y
            mvn = math.hypot(mvx, mvy)
            if mvn < 0.1:
                self._villain_inferred_intent[vid] = "stationary"
                continue
            mux, muy = mvx / mvn, mvy / mvn
            dot = mux * tox + muy * toy
            cross = abs(mux * (-toy) + muy * tox)
            if dot > 0.7:
                self._villain_inferred_intent[vid] = "approaching"
            elif dot > 0.3 and cross > 0.5:
                self._villain_inferred_intent[vid] = "flanking"
            else:
                self._villain_inferred_intent[vid] = "unknown"

    def _hero_villain_behavior_payload(self, observation: Observation) -> dict[str, Any]:
        """Structured villain behavior summary for hero prompts (Change 9)."""
        hx = float(observation.self_state.position.x)
        hy = float(observation.self_state.position.y)
        out: dict[str, Any] = {}
        for a in observation.visible_agents:
            if a.agent_type != AgentType.VILLAIN or not a.alive:
                continue
            vid = a.id
            px = float(a.position.x)
            py = float(a.position.y)
            tohx = hx - px
            tohy = hy - py
            thn = math.hypot(tohx, tohy)
            tox, toy = (1.0, 0.0) if thn < 1e-12 else (tohx / thn, tohy / thn)
            hist = self._villain_movement_history.get(vid, [])
            hist_len = len(hist)
            approaching_speed = 0.0
            if hist_len >= 2:
                p0x, p0y = hist[-2][0], hist[-2][1]
                mvx = px - p0x
                mvy = py - p0y
                mvn = math.hypot(mvx, mvy)
                if mvn > 1e-12:
                    mux, muy = mvx / mvn, mvy / mvn
                    approaching_speed = float(max(0.0, mux * tox + muy * toy) * mvn)
            out[vid] = {
                "inferred_intent": self._villain_inferred_intent.get(vid, "unknown"),
                "approaching_speed": round(approaching_speed, 4),
                "last_known_pos": [round(px, 3), round(py, 3)],
                "steps_tracked": min(5, hist_len),
            }
        return out

    def _make_prompt_bundle(self, observation: Observation) -> PromptBundle:
        # For HERO we keep the existing prompt structure unchanged.
        if self.config.agent_type != AgentType.VILLAIN:
            vb = None
            if self.config.agent_type == AgentType.HERO:
                vb = self._hero_villain_behavior_payload(observation)
            return build_prompt_bundle(
                observation,
                self._session,
                history_limit=self._session.history_limit,
                environment_config=self._environment_config,
                villain_behavior=vb,
            )

        # For VILLAIN we inject short-term memory + partial observability
        # instructions about whether the hero is visible.
        obs_payload = serialize_observation(observation)
        priv_payload = serialize_session(self._session, limit=self._session.history_limit)

        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)
        sight_r = float(observation.villain_hero_sight_radius)
        sight_r2 = sight_r * sight_r

        hero_visible = False
        hx = 0.0
        hy = 0.0
        for a in observation.visible_agents:
            if a.agent_type == AgentType.HERO and a.alive:
                dx = float(a.position.x) - sx
                dy = float(a.position.y) - sy
                if (dx * dx + dy * dy) <= sight_r2:
                    hero_visible = True
                    hx = float(a.position.x)
                    hy = float(a.position.y)
                break

        if hero_visible:
            vision_note = (
                f"You can see the hero at ({hx:.3f}, {hy:.3f}). Pursue or intercept."
            )
        else:
            if self.last_seen_hero_position is not None:
                lsx, lsy = self.last_seen_hero_position
            else:
                lsx, lsy = 0.0, 0.0
            vision_note = (
                "You cannot see the hero. "
                f"Last seen position: ({lsx:.3f}, {lsy:.3f}). "
                f"Steps since last seen: {self.steps_since_seen}. "
                "Continue searching intelligently. Maintain direction if already searching."
            )

        # Override the villain's partial observability instruction for this step.
        obs_payload["hero_vision_instruction"] = vision_note

        user_payload = {"obs": obs_payload, "priv": priv_payload}
        user_prompt = json.dumps(user_payload, separators=(",", ":"), ensure_ascii=False)
        system_prompt = build_system_prompt(self.config, self._environment_config)
        return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt)

    def _villain_hero_visible_and_pos(
        self, observation: Observation
    ) -> tuple[bool, Optional[tuple[float, float]]]:
        """Hero is only considered "visible" within villain_hero_sight_radius (env)."""
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

    def _villain_movement_from_llm_action(
        self,
        observation: Observation,
        action: Action,
    ) -> Action:
        """
        Villains: execute LLM-chosen target (or legacy vector) with geometry + hard boundary clamp.
        Does not override LLM intent.
        """
        if self._environment_config is not None:
            world_w = float(self._environment_config.world_size[0])
            world_h = float(self._environment_config.world_size[1])
        else:
            world_w, world_h = 80.0, 80.0
        max_speed = float(getattr(self.config, "max_speed", 1.0) or 1.0)
        px = float(observation.self_state.position.x)
        py = float(observation.self_state.position.y)

        src = action.movement_source
        if src in (
            "fallback_explore",
            "fallback_last_valid",
            "stuck_recovery",
            "stuck_halted",
        ):
            mv = action.movement
            mv2 = _apply_boundary_constraint(mv, px, py, world_w, world_h)
            new_src = src
            if (mv2.x != mv.x or mv2.y != mv.y) and src != "stuck_halted":
                new_src = "boundary_override"
            md = dict(action.movement_debug or {})
            md["final_movement"] = [float(mv2.x), float(mv2.y), 0.0]
            return action.model_copy(
                update={"movement": mv2, "movement_source": new_src, "movement_debug": md}
            )

        if action.llm_target_position is not None:
            tx, ty = action.llm_target_position
            mv = _compute_movement_from_target(px, py, tx, ty, max_speed)
            new_src = "llm_target"
        else:
            dx = float(action.movement.x)
            dy = float(action.movement.y)
            n = math.hypot(dx, dy)
            if n < 1e-9:
                mv = _exploration_move(self._session, observation.step_index)
                new_src = "fallback_explore"
            else:
                if n > max_speed and max_speed > 0:
                    s = max_speed / n
                    dx *= s
                    dy *= s
                mv = Vec3(x=dx, y=dy, z=0.0)
                new_src = "llm_vector_legacy"

        mv2 = _apply_boundary_constraint(mv, px, py, world_w, world_h)
        if mv2.x != mv.x or mv2.y != mv.y:
            new_src = "boundary_override"

        md = dict(action.movement_debug or {})
        md["final_movement"] = [float(mv2.x), float(mv2.y), 0.0]
        return action.model_copy(
            update={"movement": mv2, "movement_source": new_src, "movement_debug": md}
        )

    # Stuck recovery (replaces long-range quadrant "oscillation escape"): strict position-only gate.
    _HERO_STUCK_MIN_STEP = 40
    _HERO_POSITION_OSCILLATION_COUNT = 20
    _HERO_POSITION_OSCILLATION_RADIUS = 4.0
    _HERO_STUCK_RECOVERY_STEPS = 5
    # Cardinal + diagonal unit directions (axis and diagonals normalized).
    _STUCK_RECOVERY_DIRECTIONS: tuple[tuple[float, float], ...] = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7071067811865475, 0.7071067811865475),
        (-0.7071067811865475, 0.7071067811865475),
        (0.7071067811865475, -0.7071067811865475),
        (-0.7071067811865475, -0.7071067811865475),
    )

    def _detect_hero_position_oscillation(self) -> bool:
        """True if last N hero positions all lie within a small disk (pairwise max distance < R)."""
        n = self._HERO_POSITION_OSCILLATION_COUNT
        r = float(self._HERO_POSITION_OSCILLATION_RADIUS)
        if len(self._hero_pos_history) < n:
            return False
        last = self._hero_pos_history[-n:]
        max_d = 0.0
        for i in range(len(last)):
            for j in range(i + 1, len(last)):
                d = math.hypot(last[i][0] - last[j][0], last[i][1] - last[j][1])
                max_d = max(max_d, d)
        return max_d < r

    def _detect_oscillation(self, observation: Observation) -> bool:
        """Strict stuck check: only after min step; last 20 positions within 4.0 unit disk."""
        if int(getattr(observation, "step_index", 0)) < self._HERO_STUCK_MIN_STEP:
            return False
        return self._detect_hero_position_oscillation()

    def _best_stuck_recovery_direction(self, observation: Observation) -> tuple[float, float]:
        """
        Among 8 axis/diagonal directions, pick the unit vector that maximizes distance to the
        **nearest** visible villain after one max-speed step (greedy away from closest threat).
        """
        hx = float(observation.self_state.position.x)
        hy = float(observation.self_state.position.y)
        ms = float(getattr(self.config, "max_speed", 1.0) or 1.0)
        villains: list[tuple[float, float]] = []
        for a in observation.visible_agents:
            if a.agent_type == AgentType.VILLAIN and a.alive:
                villains.append((float(a.position.x), float(a.position.y)))
        if not villains:
            return (1.0, 0.0)
        best_u = (1.0, 0.0)
        best_score = -1.0
        for ux, uy in self._STUCK_RECOVERY_DIRECTIONS:
            nx = hx + ux * ms
            ny = hy + uy * ms
            nearest = min(math.hypot(nx - vx, ny - vy) for vx, vy in villains)
            if nearest > best_score:
                best_score = nearest
                best_u = (ux, uy)
        return best_u

    def _hero_movement_pipeline(
        self,
        observation: Observation,
        action: Action,
    ) -> Action:
        """
        HERO only: stabilize LLM movement while preserving intent.
        - Always 2D (z=0)
        - Blend LLM direction with direction_to_target (away from nearest villain)
        - Boundary awareness to prevent edge sliding
        - Degenerate motion fallback + stuck/constant-direction reset
        - Stuck recovery: if genuinely frozen (step>=40, last 20 positions within 4 units), run
          exactly 5 steps of axis/diagonal nudge in the direction that maximizes distance to the
          nearest visible villain (no long-range quadrant teleport).
        """
        max_speed = float(getattr(self.config, "max_speed", 1.0) or 1.0)
        cur_x = float(observation.self_state.position.x)
        cur_y = float(observation.self_state.position.y)
        if self._environment_config is not None:
            world_w = float(self._environment_config.world_size[0])
            world_h = float(self._environment_config.world_size[1])
        else:
            world_w, world_h = 80.0, 80.0

        step_idx = int(getattr(observation, "step_index", 0))
        if self._hero_stuck_recovery_remaining == 0 and self._detect_oscillation(observation):
            self._hero_stuck_recovery_unit = self._best_stuck_recovery_direction(observation)
            self._hero_stuck_recovery_remaining = self._HERO_STUCK_RECOVERY_STEPS
            self._hero_llm_targets.clear()
            self._hero_pos_history.clear()

        if self._hero_stuck_recovery_remaining > 0:
            first_recovery_step = self._hero_stuck_recovery_remaining == self._HERO_STUCK_RECOVERY_STEPS
            ux, uy = self._hero_stuck_recovery_unit
            mv = Vec3(x=ux * max_speed, y=uy * max_speed, z=0.0)
            self._hero_stuck_recovery_remaining -= 1
            mv2 = _apply_boundary_constraint(mv, cur_x, cur_y, world_w, world_h)
            md = dict(action.movement_debug or {})
            md["final_movement"] = [float(mv2.x), float(mv2.y), 0.0]
            if first_recovery_step:
                md["oscillation_escape_triggered"] = True
                md["stuck_recovery_nudge"] = True
            return action.model_copy(
                update={
                    "movement": mv2,
                    "movement_source": "stuck_recovery_nudge",
                    "movement_debug": md,
                }
            )

        eps = 1e-6
        boundary_margin = 5.0
        boundary_push_strength = 0.5

        debug_enabled = os.environ.get("LLM_MOVEMENT_DEBUG", "").strip() in {"1", "true", "True", "yes"}

        def _norm2(x: float, y: float) -> float:
            return math.sqrt(x * x + y * y)

        def _unit2(x: float, y: float, fallback_x: float, fallback_y: float) -> tuple[float, float]:
            n = _norm2(x, y)
            if not math.isfinite(n) or n < eps:
                nf = _norm2(fallback_x, fallback_y)
                if not math.isfinite(nf) or nf < eps:
                    return 1.0, 0.0
                return fallback_x / nf, fallback_y / nf
            return x / n, y / n

        # -------------------------
        # direction_to_target
        # -------------------------
        direction_to_target_x = 0.0
        direction_to_target_y = 0.0
        has_target = False

        # Hero: move away from closest visible villain.
        best = None
        best_d2 = None
        for a in observation.visible_agents:
            if a.agent_type == AgentType.VILLAIN and a.alive:
                dxv = float(a.position.x) - cur_x
                dyv = float(a.position.y) - cur_y
                d2 = dxv * dxv + dyv * dyv
                if best is None or d2 < (best_d2 or float("inf")):
                    best = a
                    best_d2 = d2
        if best is not None:
            direction_to_target_x = cur_x - float(best.position.x)
            direction_to_target_y = cur_y - float(best.position.y)
            has_target = True

        # Normalize target direction (fallback to LLM direction later).
        # We'll compute llm direction first, then use it as fallback.

        # -------------------------
        # llm_direction (2D)
        # -------------------------
        llm_dx = float(action.movement.x)
        llm_dy = float(action.movement.y)
        llm_norm = _norm2(llm_dx, llm_dy)

        if not math.isfinite(llm_norm) or llm_norm < eps:
            # Replace near-zero with direction_to_target (spec).
            if has_target:
                llm_dx, llm_dy = direction_to_target_x, direction_to_target_y
                llm_norm = _norm2(llm_dx, llm_dy)
            if not math.isfinite(llm_norm) or llm_norm < eps:
                llm_dx, llm_dy = 1.0, 0.0
                llm_norm = 1.0

        llm_dir_x = llm_dx / llm_norm
        llm_dir_y = llm_dy / llm_norm

        # direction_to_target unit
        if has_target:
            d_unit_x, d_unit_y = _unit2(
                direction_to_target_x, direction_to_target_y, llm_dir_x, llm_dir_y
            )
        else:
            d_unit_x, d_unit_y = llm_dir_x, llm_dir_y

        # -------------------------
        # Blend: 0.7*llm + 0.3*target
        # -------------------------
        blend_x = 0.7 * llm_dir_x + 0.3 * d_unit_x
        blend_y = 0.7 * llm_dir_y + 0.3 * d_unit_y
        final_dir_norm = _norm2(blend_x, blend_y)

        if not math.isfinite(final_dir_norm) or final_dir_norm < 1e-3:
            # Prevent stuck/degenerate motion.
            final_dir_x, final_dir_y = d_unit_x, d_unit_y
        else:
            final_dir_x, final_dir_y = blend_x / final_dir_norm, blend_y / final_dir_norm

        # -------------------------
        # Boundary awareness
        # -------------------------
        if self._environment_config is not None:
            world_w = float(self._environment_config.world_size[0])
            world_h = float(self._environment_config.world_size[1])
        else:
            world_w, world_h = 80.0, 80.0

        near_edge = (
            cur_x < boundary_margin
            or cur_x > (world_w - boundary_margin)
            or cur_y < boundary_margin
            or cur_y > (world_h - boundary_margin)
        )
        if near_edge:
            center_x = world_w * 0.5
            center_y = world_h * 0.5
            push_x = center_x - cur_x
            push_y = center_y - cur_y
            push_x, push_y = _unit2(push_x, push_y, final_dir_x, final_dir_y)
            fin2_x = final_dir_x + boundary_push_strength * push_x
            fin2_y = final_dir_y + boundary_push_strength * push_y
            fin2_norm = _norm2(fin2_x, fin2_y)
            if math.isfinite(fin2_norm) and fin2_norm >= eps:
                final_dir_x, final_dir_y = fin2_x / fin2_norm, fin2_y / fin2_norm

        # -------------------------
        # Degenerate safeguard
        # -------------------------
        if _norm2(final_dir_x, final_dir_y) < 1e-3:
            final_dir_x, final_dir_y = d_unit_x, d_unit_y

        # -------------------------
        # Constant-direction stuck fix
        # -------------------------
        stuck_fix_triggered = False
        if self._prev_final_dir_unit is not None:
            prev_x, prev_y = self._prev_final_dir_unit
            cos_sim = float(final_dir_x * prev_x + final_dir_y * prev_y)
            if cos_sim > 0.99:
                self._constant_dir_steps += 1
            else:
                self._constant_dir_steps = 0
        else:
            self._constant_dir_steps = 0

        if self._constant_dir_steps > 5:
            stuck_fix_triggered = True
            final_dir_x, final_dir_y = d_unit_x, d_unit_y
            self._constant_dir_steps = 0

        self._prev_final_dir_unit = (float(final_dir_x), float(final_dir_y))

        # -------------------------
        # Speed scaling + force z=0
        # -------------------------
        final_movement_x = float(final_dir_x) * max_speed
        final_movement_y = float(final_dir_y) * max_speed

        movement_debug: dict[str, Any] = {
            "stuck_fix_triggered": bool(stuck_fix_triggered),
        }
        if debug_enabled:
            movement_debug.update(
                {
                    "llm_direction": [float(llm_dir_x), float(llm_dir_y), 0.0],
                    "direction_to_target": [
                        float(d_unit_x),
                        float(d_unit_y),
                        0.0,
                    ],
                    "final_dir": [float(final_dir_x), float(final_dir_y), 0.0],
                }
            )

        return action.model_copy(
            update={
                "movement": Vec3(x=final_movement_x, y=final_movement_y, z=0.0),
                "movement_debug": movement_debug,
            }
        )

    def _invoke(self, bundle: PromptBundle) -> str:
        """
        Call the LLM backend with a per-call timeout.

        This method is intentionally isolated so the caller can retry on
        timeouts, parse failures, or validation failures.
        """
        return _run_with_timeout(
            lambda: _call_client(self._client, bundle.system_prompt, bundle.user_prompt),
            self._timeout_seconds,
        )

    def _call_and_parse(self, bundle: PromptBundle) -> tuple[LLMActionOutput, str, Optional[str]]:
        """
        Invoke LLM and parse to LLMActionOutput.

        LLM_OUTPUT_PARSER=pydantic (default): Groq client + manual JSON repair.
        LLM_OUTPUT_PARSER=baml: BAML ChooseAgentAction (same system/user prompts).
        """
        if self._output_parser == "baml":
            from .baml_parser import invoke_baml_choose_action

            def _baml_call() -> tuple[LLMActionOutput, str]:
                return invoke_baml_choose_action(bundle.system_prompt, bundle.user_prompt)

            parsed, raw_text = _run_with_timeout(_baml_call, self._timeout_seconds)
            return parsed, raw_text, parsed.intent

        raw_response = self._invoke(bundle)
        parsed, raw_intent_str = _parse_llm_output_with_raw(raw_response)
        return parsed, raw_response, raw_intent_str

    def _drop_invalid_llm_target(self, observation: Observation, action: Action) -> Action:
        """
        Strip bogus LLM ``target_position`` before movement execution:

        - Out of buffered world bounds (see ``_is_valid_world_target``)
        - Too close to current position (see ``_is_meaningful_target``, default 5.0)
        """
        tp = action.llm_target_position
        if tp is None:
            return action
        if self._environment_config is not None:
            ws = (
                float(self._environment_config.world_size[0]),
                float(self._environment_config.world_size[1]),
            )
        else:
            ws = (160.0, 160.0)
        tx, ty = float(tp[0]), float(tp[1])
        if not _is_valid_world_target((tx, ty), ws):
            return action.model_copy(update={"llm_target_position": None})
        cur = (
            float(observation.self_state.position.x),
            float(observation.self_state.position.y),
        )
        if not _is_meaningful_target(cur, (tx, ty)):
            return action.model_copy(update={"llm_target_position": None})
        return action

    def step(self, observation: Observation) -> Action:
        """
        Produce one validated action for the given observation.

        Retry policy:
        - try the LLM call up to max_retries + 1 times
        - each call has a per-call timeout
        - parse and validate the response into Action
        - only fall back to a zero-action at the very end
        """
        hero_visible = False
        hero_pos: Optional[tuple[float, float]] = None
        if self.config.agent_type == AgentType.VILLAIN:
            hero_visible, hero_pos = self._villain_hero_visible_and_pos(observation)
            if hero_visible and hero_pos is not None:
                self.last_seen_hero_position = hero_pos
                self.steps_since_seen = 0
            else:
                self.steps_since_seen += 1

        bundle = self._make_prompt_bundle(observation)
        combined_prompt = _build_combined_prompt(bundle)
        last_error: Optional[str] = None
        last_raw_response: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            try:
                parsed, last_raw_response, raw_intent_str = self._call_and_parse(bundle)
                action = llm_action_to_action(self.id, parsed)
                action = action.model_copy(
                    update={"llm_raw_target_position": action.llm_target_position}
                )
                action = self._drop_invalid_llm_target(observation, action)
                if self.config.disable_messages:
                    action = action.model_copy(update={"message": None})
                action = self._maybe_attach_coordination_message(observation, action)

                if self.config.agent_type == AgentType.HERO:
                    hpx = float(observation.self_state.position.x)
                    hpy = float(observation.self_state.position.y)
                    self._hero_pos_history.append((hpx, hpy))
                    # Keep enough history for 20-position window (+ margin).
                    if len(self._hero_pos_history) > 44:
                        self._hero_pos_history.pop(0)
                    if action.llm_target_position is not None:
                        tp = action.llm_target_position
                        self._hero_llm_targets.append((float(tp[0]), float(tp[1])))
                        if len(self._hero_llm_targets) > 20:
                            self._hero_llm_targets.pop(0)
                        px = float(observation.self_state.position.x)
                        py = float(observation.self_state.position.y)
                        tx, ty = action.llm_target_position
                        ms = float(self.config.max_speed or 1.0)
                        pre = _compute_movement_from_target(px, py, tx, ty, ms)
                        action = action.model_copy(update={"movement": pre})
                    action = self._hero_movement_pipeline(observation, action)
                else:
                    action = self._villain_movement_from_llm_action(observation, action)

                self._session.record_turn(
                    step_index=observation.step_index,
                    time=observation.time,
                    observation=observation,
                    prompt=combined_prompt,
                    raw_response=last_raw_response,
                    action=action,
                    valid=True,
                    error=None,
                    llm_raw_intent=raw_intent_str,
                )
                return action

            except TimeoutError as exc:
                last_error = f"timeout: {exc}"
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = f"parse_or_validation_error: {exc}"
            except Exception as exc:  # pragma: no cover - defensive boundary
                last_error = f"unexpected_error: {type(exc).__name__}: {exc}"

            if attempt < self._max_retries:
                continue

        fallback_intent = (
            "timeout" if last_error and last_error.startswith("timeout") else "invalid_output"
        )
        # Research note: if we have no valid LLM output and we're a villain,
        # label the fallback intent as `pursue_target` to keep analytics stable.
        if (
            self.config.agent_type == AgentType.VILLAIN
            and fallback_intent == "invalid_output"
        ):
            fallback_intent = "pursue_target"
        fallback_action = _fallback_from_last_valid(self._session, observation, fallback_intent)
        fallback_action = self._maybe_attach_coordination_message(observation, fallback_action)
        if self.config.agent_type == AgentType.HERO:
            fallback_action = self._hero_movement_pipeline(observation, fallback_action)
        else:
            fallback_action = self._villain_movement_from_llm_action(observation, fallback_action)

        self._session.record_turn(
            step_index=observation.step_index,
            time=observation.time,
            observation=observation,
            prompt=combined_prompt,
            raw_response=last_raw_response,
            action=fallback_action,
            valid=False,
            error=last_error,
            llm_raw_intent=None,
        )
        return fallback_action

    def _maybe_attach_coordination_message(self, observation: Observation, action: Action) -> Action:
        """
        Lightweight coordination moved OUT of the prompt:
        - Villains broadcast a compact numeric message when comms are enabled
          and use_auto_coord_message is True (see AgentConfig).
        - Format payload: [hero_x, hero_y, hero_conf, self_x, self_y]
        - If hero not visible: hero_conf=0 and hero_x/hero_y are 0.
        """
        if self.config.agent_type != AgentType.VILLAIN:
            return action
        if self.config.disable_messages:
            return action.model_copy(update={"message": None})
        if not self.config.communication_enabled:
            return action
        if not self.config.use_auto_coord_message:
            return action
        if action.message is not None:
            return action

        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)
        sight_r2 = float(observation.villain_hero_sight_radius) ** 2

        hero_x = 0.0
        hero_y = 0.0
        hero_conf = 0.0
        for other in observation.visible_agents:
            if other.agent_type == AgentType.HERO and other.alive:
                dx = float(other.position.x) - sx
                dy = float(other.position.y) - sy
                if (dx * dx + dy * dy) <= sight_r2:
                    hero_x = float(other.position.x)
                    hero_y = float(other.position.y)
                    hero_conf = 1.0
                # If the hero exists but is outside our partial observability
                # radius, we keep hero_conf=0 and hero_x/y=0.
                break

        msg = Message(
            sender_id=self.id,
            recipient_ids=None,  # broadcast/team routing handled by router
            payload=[hero_x, hero_y, hero_conf, sx, sy],
            channel="coord",
        )
        return action.model_copy(update={"message": msg})

