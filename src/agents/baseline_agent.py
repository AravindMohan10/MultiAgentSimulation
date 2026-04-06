"""
Deterministic rule-based agent for ablation vs LLM policies.

Implements simple heuristics so experiments can compare LLM behavior against
a fixed baseline without API calls.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

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
from .schema import LLMActionOutput


def _clamp_pos(x: float, y: float, w: float, h: float, margin: float = 1.0) -> Tuple[float, float]:
    return (
        max(margin, min(w - margin, x)),
        max(margin, min(h - margin, y)),
    )


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 0.0, 0.0
    return dx / n, dy / n


def _movement_toward(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    max_speed: float,
) -> Vec3:
    ux, uy = _unit(tx - sx, ty - sy)
    return Vec3(x=ux * max_speed, y=uy * max_speed, z=0.0)


def _movement_away(
    sx: float,
    sy: float,
    ox: float,
    oy: float,
    max_speed: float,
) -> Vec3:
    ux, uy = _unit(sx - ox, sy - oy)
    return Vec3(x=ux * max_speed, y=uy * max_speed, z=0.0)


class RuleBasedAgent(BaseAgent):
    """
    Deterministic baseline for ablation comparison.

    HERO strategy:
      - Move away from nearest visible villain
      - If no villain visible: move toward world center
      - Never sends messages

    VILLAIN strategy:
      - If hero visible within villain_hero_sight_radius: move toward hero,
        share hero position with teammate via message
      - If hero not visible but have memory (last_seen_hero_pos): move toward last known position
      - If no memory: move in expanding spiral pattern from spawn
      - No coordination beyond position sharing
    """

    def __init__(self, agent_config: AgentConfig, env_config: EnvironmentConfig) -> None:
        super().__init__(agent_config)
        self.env_config = env_config
        self.last_seen_hero_pos: Optional[Tuple[float, float]] = None
        self.steps_since_seen: int = 0
        self.spawn_position: Optional[Tuple[float, float]] = None
        self.spiral_step: int = 0
        self._last_action: Optional[Action] = None

    def _world_wh(self) -> Tuple[float, float]:
        w, h = self.env_config.world_size
        return float(w), float(h)

    def _hero_visible(
        self, observation: Observation
    ) -> tuple[bool, Optional[Tuple[float, float]]]:
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

    def _nearest_visible_villain(self, observation: Observation) -> Optional[Tuple[float, float, float]]:
        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)
        best: Optional[Tuple[float, float, float]] = None
        for a in observation.visible_agents:
            if a.agent_type == AgentType.VILLAIN and a.alive:
                vx = float(a.position.x)
                vy = float(a.position.y)
                d2 = (vx - sx) ** 2 + (vy - sy) ** 2
                if best is None or d2 < best[2]:
                    best = (vx, vy, d2)
        if best is None:
            return None
        return (best[0], best[1], best[2])

    def _spiral_target(self) -> Tuple[float, float]:
        assert self.spawn_position is not None
        sx0, sy0 = self.spawn_position
        w, h = self._world_wh()
        t = float(self.spiral_step)
        # Archimedean spiral: radius grows with step index, angle advances slowly.
        a, b = 0.8, 0.35
        r = min(a + b * t, 0.45 * min(w, h))
        theta = t * 0.45
        tx = sx0 + r * math.cos(theta)
        ty = sy0 + r * math.sin(theta)
        margin = 2.0
        return _clamp_pos(tx, ty, w, h, margin=margin)

    def _coord_message(
        self,
        observation: Observation,
        hero_x: float,
        hero_y: float,
    ) -> Optional[Message]:
        if self.config.disable_messages or not self.config.communication_enabled:
            return None
        # Other villain id (team of two villains in default setup).
        teammates = []
        if self.id == "villain_1":
            teammates = ["villain_2"]
        elif self.id == "villain_2":
            teammates = ["villain_1"]
        else:
            for aid in ("villain_1", "villain_2", "hero_1"):
                if aid != self.id:
                    teammates.append(aid)
                    break
        if not teammates:
            return None
        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)
        payload = [hero_x, hero_y, 1.0, sx, sy]
        return Message(
            sender_id=self.id,
            recipient_ids=teammates,
            payload=payload,
            channel="coord",
        )

    def _hero_step(self, observation: Observation) -> Action:
        w, h = self._world_wh()
        cx, cy = w / 2.0, h / 2.0
        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)
        max_speed = float(self.config.max_speed or 1.0)

        nv = self._nearest_visible_villain(observation)
        if nv is not None:
            vx, vy, _ = nv
            mv = _movement_away(sx, sy, vx, vy, max_speed)
            intent = "flee_visible_villain"
        else:
            mv = _movement_toward(sx, sy, cx, cy, max_speed)
            intent = "go_world_center"

        act = Action(
            movement=mv,
            message=None,
            intent=LLMActionOutput.normalize_intent(intent),
            movement_source="rule_based",
            llm_target_position=None,
        )
        return act

    def _villain_step(self, observation: Observation) -> Action:
        max_speed = float(self.config.max_speed or 1.0)
        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)

        visible, hero_pos = self._hero_visible(observation)
        msg: Optional[Message] = None

        if visible and hero_pos is not None:
            hx, hy = hero_pos
            self.last_seen_hero_pos = (hx, hy)
            self.steps_since_seen = 0
            mv = _movement_toward(sx, sy, hx, hy, max_speed)
            msg = self._coord_message(observation, hx, hy)
            intent = "pursue_visible_hero"
            self.spiral_step += 1
        else:
            if self.last_seen_hero_pos is not None:
                self.steps_since_seen += 1
                tx, ty = self.last_seen_hero_pos
                mv = _movement_toward(sx, sy, tx, ty, max_speed)
                intent = "go_last_seen_hero"
            else:
                tx, ty = self._spiral_target()
                mv = _movement_toward(sx, sy, tx, ty, max_speed)
                intent = "spiral_search"
            self.spiral_step += 1

        act = Action(
            movement=mv,
            message=msg,
            intent=LLMActionOutput.normalize_intent(intent),
            movement_source="rule_based",
            llm_target_position=None,
        )
        return act

    def step(self, observation: Observation) -> Action:
        if self.spawn_position is None:
            self.spawn_position = (
                float(observation.self_state.position.x),
                float(observation.self_state.position.y),
            )

        if self.config.agent_type == AgentType.HERO:
            out = self._hero_step(observation)
        else:
            out = self._villain_step(observation)

        self._last_action = out
        return out
