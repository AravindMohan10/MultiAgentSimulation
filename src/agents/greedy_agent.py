"""
Deterministic greedy pursuer: no LLM, no API. Uses only Observation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from ..core.models import Action, AgentConfig, AgentType, EnvironmentConfig, Observation, Vec3
from .base import BaseAgent


class GreedyPursuerAgent(BaseAgent):
    """
    Villain policy: if hero is in ``visible_agents``, move at max_speed toward the
    hero's current position; else move toward last known hero position, or map center.
    """

    def __init__(self, agent_config: AgentConfig, env_config: EnvironmentConfig) -> None:
        super().__init__(agent_config)
        self._env_config = env_config
        self._last_known_hero: Optional[Tuple[float, float]] = None
        self.steps_since_seen = 0

    def step(self, observation: Observation) -> Action:
        if self.config.agent_type != AgentType.VILLAIN:
            return Action(
                movement=Vec3(x=0.0, y=0.0, z=0.0),
                intent="hold_position",
                movement_source="greedy_baseline",
            )

        ms = float(self.config.max_speed or 1.0)
        sx = float(observation.self_state.position.x)
        sy = float(observation.self_state.position.y)

        hero_visible_pos: Optional[Tuple[float, float]] = None
        for a in observation.visible_agents:
            if a.agent_type == AgentType.HERO and a.alive:
                hero_visible_pos = (float(a.position.x), float(a.position.y))
                break

        if hero_visible_pos is not None:
            self._last_known_hero = hero_visible_pos
            self.steps_since_seen = 0
            tx, ty = hero_visible_pos
        else:
            self.steps_since_seen += 1
            if self._last_known_hero is not None:
                tx, ty = self._last_known_hero
            else:
                w, h = float(self._env_config.world_size[0]), float(self._env_config.world_size[1])
                tx, ty = w * 0.5, h * 0.5

        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy)
        if dist < 1e-12:
            return Action(
                movement=Vec3(x=0.0, y=0.0, z=0.0),
                intent="pursue_target",
                movement_source="greedy_baseline",
            )

        ux, uy = dx / dist, dy / dist
        return Action(
            movement=Vec3(x=ux * ms, y=uy * ms, z=0.0),
            intent="pursue_target",
            movement_source="greedy_baseline",
        )
