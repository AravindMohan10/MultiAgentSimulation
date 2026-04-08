"""
Self-steer LLM agent: minimal prompts, recent-memory in user JSON, optional BAML parse.

Parallel track to LLMAgent — does not replace the steered V2_GUIDED pipeline.
No auto-coord messages; fallbacks use exploration only (no scripted pursue).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..core.models import (
    Action,
    AgentConfig,
    AgentType,
    EnvironmentConfig,
    Message,
    Observation,
    Vec3,
)
from .base import BaseAgent
from .llm_agent import (
    _apply_boundary_constraint,
    _compute_movement_from_target,
    _exploration_move,
    _is_meaningful_target,
    _is_valid_world_target,
    _parse_llm_output_with_raw,
    _run_with_timeout,
)
from .baml_parser import extract_baml_raw_output
from .prompts_self_steer import (
    build_combined_prompt,
    build_static_map_block,
    build_system_prompt,
    build_user_prompt,
    format_recent_line,
)
from .schema import LLMActionOutput, llm_action_to_action
from .session import AgentSession


@dataclass(slots=True)
class _RecentRecord:
    step_index: int
    intent: str
    target: Optional[Tuple[float, float]]
    pos: Optional[Tuple[float, float]] = None
    dist: Optional[float] = None
    msg: Optional[str] = None
    note: Optional[str] = None


class SelfSteerAgent(BaseAgent):
    """
    Minimal-prompt agent with short-term memory via ``recent[]`` in the user JSON.

    Compatible with runner logging (``.session``, ``steps_since_seen`` for villains).
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        environment_config: EnvironmentConfig,
        capture_radius: float = 2.0,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        recent_limit: int = 6,
        step_sleep_seconds: float = 0.0,
        use_baml: Optional[bool] = None,
    ) -> None:
        super().__init__(config)
        self._environment_config = environment_config
        self._capture_radius = float(capture_radius)
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._max_retries = max(0, int(max_retries))
        self._recent_limit = max(1, int(recent_limit))
        self._step_sleep = max(0.0, float(step_sleep_seconds))
        if use_baml is None:
            use_baml = os.environ.get("SELF_STEER_USE_BAML", "1").strip().lower() in (
                "1",
                "true",
                "yes",
            )
        self._use_baml = bool(use_baml)

        self._session = AgentSession(agent_id=config.id, config=config)
        self._system_prompt: str = ""
        self._map_block: str = ""
        self._initialized = False
        self._recent: List[_RecentRecord] = []

        self.last_seen_hero_position: Optional[Tuple[float, float]] = None
        self.steps_since_seen: int = 0

    @property
    def session(self) -> AgentSession:
        return self._session

    def reset_episode(self) -> None:
        self._session.reset()
        self._system_prompt = ""
        self._map_block = ""
        self._initialized = False
        self._recent.clear()
        self.last_seen_hero_position = None
        self.steps_since_seen = 0

    def _ensure_initialized(self, observation: Observation) -> None:
        if self._initialized:
            return
        self._map_block = build_static_map_block(observation, self._environment_config)
        self._system_prompt = build_system_prompt(
            self.config,
            self._environment_config,
            capture_radius=self._capture_radius,
            map_block=self._map_block,
        )
        self._initialized = True

    def _update_villain_sight(self, observation: Observation) -> None:
        if self.config.agent_type != AgentType.VILLAIN:
            return
        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)
        sight_r2 = float(observation.villain_hero_sight_radius) ** 2
        for a in observation.visible_agents:
            if a.agent_type == AgentType.HERO and a.alive:
                dx = float(a.position.x) - sx
                dy = float(a.position.y) - sy
                if (dx * dx + dy * dy) <= sight_r2:
                    self.last_seen_hero_position = (
                        float(a.position.x),
                        float(a.position.y),
                    )
                    self.steps_since_seen = 0
                    return
        self.steps_since_seen += 1

    def _recent_lines(self) -> List[str]:
        return [
            format_recent_line(
                step_index=r.step_index,
                intent=r.intent,
                target=r.target,
                pos=r.pos,
                dist=r.dist,
                msg=r.msg,
                note=r.note,
            )
            for r in self._recent[-self._recent_limit :]
        ]

    def _build_bundle(self, observation: Observation) -> tuple[str, str, str]:
        user_prompt = build_user_prompt(
            observation,
            self.config,
            steps_since_hero_seen=self.steps_since_seen,
            last_seen_hero=self.last_seen_hero_position,
            recent_lines=self._recent_lines(),
        )
        combined = build_combined_prompt(self._system_prompt, user_prompt)
        return self._system_prompt, user_prompt, combined

    def _call_llm(self, system_prompt: str, user_prompt: str) -> tuple[LLMActionOutput, str, Optional[str]]:
        if self._use_baml:
            from .baml_parser import invoke_baml_choose_action

            def _baml() -> tuple[LLMActionOutput, str]:
                return invoke_baml_choose_action(system_prompt, user_prompt)

            parsed, raw = _run_with_timeout(_baml, self._timeout_seconds)
            return parsed, raw, parsed.intent

        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        model = os.environ.get("SELF_STEER_MODEL", "llama-3.3-70b-versatile")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(os.environ.get("SELF_STEER_TEMPERATURE", "0.3")),
            max_tokens=int(os.environ.get("SELF_STEER_MAX_TOKENS", "400")),
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed, raw_intent = _parse_llm_output_with_raw(raw)
        return parsed, raw, raw_intent

    def _extract_note(self, raw: str) -> Optional[str]:
        try:
            candidate = raw
            if "{" in raw:
                start = raw.find("{")
                end = raw.rfind("}")
                if end > start:
                    candidate = raw[start : end + 1]
            data = json.loads(candidate)
            if isinstance(data, dict):
                note = data.get("note")
                if isinstance(note, str) and note.strip():
                    return note.strip()[:120]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return None

    def _append_recent(
        self, step_index: int, action: Action, note: Optional[str], observation: "Observation"
    ) -> None:
        tp: Optional[Tuple[float, float]] = None
        if action.llm_target_position is not None:
            tp = (float(action.llm_target_position[0]), float(action.llm_target_position[1]))

        pos: Optional[Tuple[float, float]] = (
            round(float(observation.self_state.position.x), 1),
            round(float(observation.self_state.position.y), 1),
        )

        dist: Optional[float] = None
        if observation.visible_agents:
            sx = float(observation.self_state.position.x)
            sy = float(observation.self_state.position.y)
            nearest = min(
                math.hypot(float(a.position.x) - sx, float(a.position.y) - sy)
                for a in observation.visible_agents
                if a.alive
            ) if any(a.alive for a in observation.visible_agents) else None
            if nearest is not None:
                dist = round(nearest, 1)

        msg: Optional[str] = None
        best_conf = 0.0
        for m in (observation.incoming_messages or []):
            payload = getattr(m, "payload", None) or []
            if len(payload) >= 3:
                c = float(payload[2])
                if c > best_conf:
                    best_conf = c
                    msg = f"[{float(payload[0]):.1f},{float(payload[1]):.1f}]c={c:.1f}"

        self._recent.append(
            _RecentRecord(
                step_index=step_index,
                intent=str(action.intent or "unknown"),
                target=tp,
                pos=pos,
                dist=dist,
                msg=msg,
                note=note,
            )
        )
        if len(self._recent) > self._recent_limit * 2:
            self._recent = self._recent[-self._recent_limit * 2 :]

    def _drop_invalid_target(self, observation: Observation, action: Action) -> Action:
        tp = action.llm_target_position
        if tp is None:
            return action
        ws = (
            float(self._environment_config.world_size[0]),
            float(self._environment_config.world_size[1]),
        )
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

    def _apply_movement(self, observation: Observation, action: Action) -> Action:
        """Target-based movement with boundary clamp (shared logic with LLMAgent villains)."""
        world_w = float(self._environment_config.world_size[0])
        world_h = float(self._environment_config.world_size[1])
        max_speed = float(self.config.max_speed or 1.0)
        px = float(observation.self_state.position.x)
        py = float(observation.self_state.position.y)

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
        actually_clamped = abs(mv2.x - mv.x) > 1e-6 or abs(mv2.y - mv.y) > 1e-6
        if actually_clamped:
            new_src = "boundary_override"

        md = dict(action.movement_debug or {})
        md["final_movement"] = [float(mv2.x), float(mv2.y), 0.0]
        return action.model_copy(
            update={"movement": mv2, "movement_source": new_src, "movement_debug": md}
        )

    def _fallback_action(self, observation: Observation, intent: str) -> Action:
        return Action(
            movement=_exploration_move(self._session, observation.step_index),
            message=None,
            intent=intent,
            movement_source="fallback_explore",
        )

    def step(self, observation: Observation) -> Action:
        self._ensure_initialized(observation)
        self._update_villain_sight(observation)

        system_prompt, user_prompt, combined_prompt = self._build_bundle(observation)
        last_error: Optional[str] = None
        last_raw: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            try:
                parsed, last_raw, raw_intent = self._call_llm(system_prompt, user_prompt)
                action = llm_action_to_action(self.id, parsed)
                action = action.model_copy(
                    update={"llm_raw_target_position": action.llm_target_position}
                )
                if self.config.disable_messages:
                    action = action.model_copy(update={"message": None})

                action = self._drop_invalid_target(observation, action)
                action = self._apply_movement(observation, action)

                note = self._extract_note(last_raw or "")
                self._append_recent(observation.step_index, action, note, observation)

                self._session.record_turn(
                    step_index=observation.step_index,
                    time=observation.time,
                    observation=observation,
                    prompt=combined_prompt,
                    raw_response=last_raw,
                    action=action,
                    valid=True,
                    error=None,
                    llm_raw_intent=raw_intent,
                )
                if self._step_sleep > 0:
                    time.sleep(self._step_sleep)
                return action

            except TimeoutError as exc:
                last_error = f"timeout: {exc}"
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = f"parse_or_validation_error: {exc}"
            except Exception as exc:
                last_error = f"unexpected_error: {type(exc).__name__}: {exc}"
                if not last_raw:
                    last_raw = extract_baml_raw_output(exc)

            if attempt < self._max_retries:
                continue

        fallback_intent = (
            "timeout" if last_error and last_error.startswith("timeout") else "invalid_output"
        )
        fallback = self._fallback_action(observation, fallback_intent)
        self._session.record_turn(
            step_index=observation.step_index,
            time=observation.time,
            observation=observation,
            prompt=combined_prompt,
            raw_response=last_raw,
            action=fallback,
            valid=False,
            error=last_error,
            llm_raw_intent=None,
        )
        if self._step_sleep > 0:
            time.sleep(self._step_sleep)
        return fallback
