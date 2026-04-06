"""
Simulation engine: central loop and sub-systems.

This is the **only** place that owns and updates WorldState. It implements the
snapshot-update cycle:

  world_state_t → PerceptionEngine (observations) → [external: agent.step(obs)]
  → actions → CommunicationRouter (buffer messages) → PhysicsEngine (apply)
  → world_state_t+1

Components in this file:
- **PerceptionEngine**: builds a filtered Observation per agent from WorldState.
- **CommunicationRouter**: buffers messages from actions and delivers to teammates
  at the next step (team-only by default).
- **PhysicsEngine**: validates and applies actions (movement, bounds) to produce
  the next WorldState.
- **SimulationEngine**: holds configs, current WorldState, RNG; exposes reset()
  and step(actions); uses the three above.

Agents are **not** in this file. The runner creates agents, gets observations from
the engine, calls agent.step(observation), then passes the resulting actions
into engine.step(actions). So the engine never imports or holds agent instances.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

from .models import (
    Action,
    AgentConfig,
    AgentState,
    AgentType,
    EnvironmentConfig,
    MapTemplate,
    Message,
    Obstacle,
    Observation,
    TerrainInfo,
    Vec3,
    WeatherInfo,
    WorldState,
)


def _dist2(a: Vec3, b: Vec3) -> float:
    """Squared Euclidean distance in the plane (x, y)."""
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(_dist2(a, b))


def _point_in_obstacle(px: float, py: float, obstacle: Obstacle) -> bool:
    """True if (px, py) is inside the obstacle (circle)."""
    d2 = (px - obstacle.position.x) ** 2 + (py - obstacle.position.y) ** 2
    return d2 <= (obstacle.radius ** 2)


def _near_obstacle(px: float, py: float, obstacle: Obstacle, buffer: float = 3.0) -> bool:
    """True if (px, py) lies within obstacle radius plus safety buffer (v2 spawn validation only)."""
    d2 = (px - obstacle.position.x) ** 2 + (py - obstacle.position.y) ** 2
    return d2 <= (obstacle.radius + buffer) ** 2


def _generate_scattered_obstacles(
    rng: random.Random,
    world_size: Tuple[float, float],
    obstacle_density: float,
    obstacle_radius: float,
) -> List[Obstacle]:
    """
    SCATTERED template: procedural obstacles from density. Reproducible with same seed.
    Density scales with area: more obstacles on larger maps.
    """
    w, h = world_size
    if w <= 0 or h <= 0 or obstacle_radius <= 0 or obstacle_density <= 0:
        return []
    area = w * h
    # Scale count with area and density; cap so we don't create huge lists
    num = int(obstacle_density * area / 80.0)
    num = max(0, min(num, 2000))
    margin = max(obstacle_radius, 1.0)
    x_min = margin
    x_max = max(x_min, w - margin)
    y_min = margin
    y_max = max(y_min, h - margin)
    obstacles: List[Obstacle] = []
    for _ in range(num):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        obstacles.append(
            Obstacle(
                position=Vec3(x=x, y=y, z=0.0),
                radius=obstacle_radius,
            )
        )
    return obstacles


def _apply_boundary(
    x: float, y: float, w: float, h: float, mode: str
) -> Tuple[float, float]:
    """Apply boundary mode: hard (clamp), wrap (toroidal), or bounce (reflect)."""
    if mode == "hard":
        return (max(0.0, min(w, x)), max(0.0, min(h, y)))
    if mode == "wrap":
        xw = x % w if w > 0 else 0.0
        yw = y % h if h > 0 else 0.0
        if xw < 0:
            xw += w
        if yw < 0:
            yw += h
        return (xw, yw)
    if mode == "bounce":
        nx, ny = x, y
        if nx < 0:
            nx = -nx
        if nx > w:
            nx = 2 * w - nx
        if ny < 0:
            ny = -ny
        if ny > h:
            ny = 2 * h - ny
        return (max(0.0, min(w, nx)), max(0.0, min(h, ny)))
    return (max(0.0, min(w, x)), max(0.0, min(h, y)))


class PerceptionEngine:
    """
    Builds filtered observations for each agent from a WorldState snapshot.

    Each agent receives only:
    - Its own state (copy).
    - Other agents within its vision radius (after weather modifier).
    - Local terrain/weather (currently global; can be refined to local later).
    - Incoming messages from the communication router (passed in).
    """

    def __init__(
        self,
        agent_configs: List[AgentConfig],
        base_visibility_radius: float,
        weather_visibility_modifier: float = 1.0,
        global_visibility_radius: float | None = None,
        observation_noise_std: float = 0.0,
        *,
        env_config: EnvironmentConfig,
        villain_hero_sight_radius: float = 15.0,
        rng: random.Random | None = None,
    ) -> None:
        self._agent_configs = {c.id: c for c in agent_configs}
        self._base_visibility = base_visibility_radius
        self._visibility_mod = weather_visibility_modifier
        self._global_visibility_radius = global_visibility_radius
        self._noise_std = max(0.0, float(observation_noise_std))
        self._villain_hero_sight_radius = max(0.0, float(villain_hero_sight_radius))
        self._env_config = env_config
        self._rng = rng or random.Random(0)

    def build_observations(
        self,
        world_state: WorldState,
        incoming_messages: Dict[str, List[Message]],
    ) -> Dict[str, Observation]:
        """One Observation per agent that is alive in world_state."""
        out: Dict[str, Observation] = {}
        effective_visibility = self._base_visibility * self._visibility_mod

        for agent_id, state in world_state.agents.items():
            if not state.alive:
                continue
            config = self._agent_configs.get(agent_id)
            vision_radius = (
                config.vision_radius if config else 10.0
            )
            vision_radius = min(vision_radius, effective_visibility)
            if self._global_visibility_radius is not None:
                vision_radius = min(vision_radius, self._global_visibility_radius)

            visible: List[AgentState] = []
            for other_id, other in world_state.agents.items():
                if other_id == agent_id or not other.alive:
                    continue
                if _dist(state.position, other.position) <= vision_radius:
                    obs_other = other.model_copy(deep=True)
                    if self._noise_std > 0:
                        obs_other.position.x += self._rng.gauss(0.0, self._noise_std)
                        obs_other.position.y += self._rng.gauss(0.0, self._noise_std)
                    visible.append(obs_other)

            cp = self._env_config.chokepoint_positions
            obs = Observation(
                self_state=state.model_copy(deep=True),
                visible_agents=visible,
                local_terrain=world_state.terrain.model_copy(deep=True),
                local_weather=world_state.weather.model_copy(deep=True),
                incoming_messages=incoming_messages.get(agent_id, []),
                time=world_state.time,
                step_index=world_state.step_index,
                villain_hero_sight_radius=self._villain_hero_sight_radius,
                world_obstacles=[o.model_copy(deep=True) for o in world_state.obstacles],
                map_template=self._env_config.map_template.value,
                chokepoint_positions=list(cp) if cp else None,
            )
            out[agent_id] = obs
        return out


class CommunicationRouter:
    """
    Environment-mediated communication.

    - Accepts messages emitted in actions at step t.
    - Stores them and delivers to eligible agents at step t+1.
    - Eligibility: team-only (same team_id). If AgentState has no team_id,
      we use agent_type (all villains share a team for now).
    """

    def __init__(
        self,
        agent_configs: List[AgentConfig],
        *,
        message_delay_steps: int = 0,
        message_budget_per_agent: int | None = None,
    ) -> None:
        self._configs = {c.id: c for c in agent_configs}
        self._pending: List[tuple[int, Message]] = []
        self._message_delay_steps = max(0, int(message_delay_steps))
        self._message_budget_per_agent = message_budget_per_agent
        self._sent_count: Dict[str, int] = {}

    def reset(self) -> None:
        self._pending.clear()
        self._sent_count.clear()

    def submit(self, message: Message, current_step_index: int) -> None:
        """Called by the engine when an action contains a message."""
        sender_cfg = self._configs.get(message.sender_id)
        if sender_cfg and sender_cfg.disable_messages:
            return
        if self._message_budget_per_agent is not None:
            used = self._sent_count.get(message.sender_id, 0)
            if used >= self._message_budget_per_agent:
                return
            self._sent_count[message.sender_id] = used + 1
        deliver_step = current_step_index + 1 + self._message_delay_steps
        self._pending.append((deliver_step, message))

    def deliver(self, world_state: WorldState, current_step_index: int) -> Dict[str, List[Message]]:
        """
        Compute who receives what for the next observation cycle.
        Returns dict: agent_id -> list of Message to include in their Observation.
        """
        result: Dict[str, List[Message]] = {}
        remaining: List[tuple[int, Message]] = []
        for deliver_step, msg in self._pending:
            if deliver_step > current_step_index:
                remaining.append((deliver_step, msg))
                continue
            sender_state = world_state.agents.get(msg.sender_id)
            if not sender_state or not sender_state.alive:
                continue
            sender_team = getattr(sender_state, "team_id", None)
            if sender_team is None:
                sender_team = sender_state.agent_type.value

            for agent_id, state in world_state.agents.items():
                if not state.alive or agent_id == msg.sender_id:
                    continue
                cfg = self._configs.get(agent_id)
                if cfg and (not cfg.communication_enabled or cfg.disable_messages):
                    continue
                other_team = getattr(state, "team_id", None)
                if other_team is None:
                    other_team = state.agent_type.value
                if sender_team != other_team:
                    continue
                if msg.recipient_ids is not None and len(msg.recipient_ids) > 0:
                    if agent_id not in msg.recipient_ids:
                        continue
                result.setdefault(agent_id, []).append(msg)
        self._pending = remaining
        return result


class PhysicsEngine:
    """
    Applies validated actions to WorldState to produce the next state.

    - Movement: position += movement (clamped by max_speed per agent).
    - Obstacle collision: block movement if new position is inside an obstacle.
    - Boundary: hard (clamp), wrap (toroidal), or bounce (reflect) per config.
    - Time and step_index advanced.
    - Does not yet handle capture or terrain movement cost.
    """

    def __init__(
        self,
        agent_configs: List[AgentConfig],
        world_size: Tuple[float, float],
        decision_dt: float,
        boundary_mode: str = "hard",
        *,
        rng: random.Random,
    ) -> None:
        self._configs = {c.id: c for c in agent_configs}
        self._world_size = world_size
        self._decision_dt = decision_dt
        self._boundary_mode = boundary_mode
        self._rng = rng

    def apply(
        self,
        world_state: WorldState,
        actions: Dict[str, Action],
    ) -> WorldState:
        """Produce a new WorldState from the current one and the actions."""
        w, h = self._world_size
        new_agents: Dict[str, AgentState] = {}
        obstacles = world_state.obstacles

        # Small epsilon for "didn't move".
        eps2 = 1e-12
        verbose_block_print = os.environ.get("MOVEMENT_DEBUG_VERBOSE", "").strip() in {"1", "true", "True", "yes"}
        wall_thresh = max(0.75, 0.02 * float(min(w, h)))

        def _normalize_scale(dx: float, dy: float, mag: float) -> tuple[float, float]:
            norm = math.sqrt(dx * dx + dy * dy)
            if norm <= 1e-12:
                return 0.0, 0.0
            if mag <= 0.0:
                return 0.0, 0.0
            scale = mag / norm
            return dx * scale, dy * scale

        def _boundary_adjust(
            dx_in: float,
            dy_in: float,
            x0: float,
            y0: float,
            mag: float,
        ) -> tuple[float, float, bool]:
            """
            Make a movement vector safe for boundaries by (1) reflecting the blocked axis
            and/or zeroing/projection and (2) biasing inward when near a wall.
            Returns: (dx_adj, dy_adj, hit_boundary)
            """
            if self._boundary_mode == "wrap":
                dx_adj, dy_adj = _normalize_scale(dx_in, dy_in, mag)
                return dx_adj, dy_adj, False

            dx = dx_in
            dy = dy_in
            hit_boundary = False

            # Predict next position; if out of bounds, reflect blocked axis.
            nx = x0 + dx
            ny = y0 + dy

            if self._boundary_mode in {"hard", "bounce"}:
                if nx < 0.0:
                    hit_boundary = True
                    if dx < 0.0:
                        # Damped reflection: reduce tangential energy.
                        dx = -dx * 0.5
                    if x0 + dx < 0.0:
                        # Still outside: project exactly onto the wall.
                        dx = -x0
                elif nx > w:
                    hit_boundary = True
                    if dx > 0.0:
                        dx = -dx * 0.5
                    if x0 + dx > w:
                        dx = w - x0

                if ny < 0.0:
                    hit_boundary = True
                    if dy < 0.0:
                        dy = -dy * 0.5
                    if y0 + dy < 0.0:
                        dy = -y0
                elif ny > h:
                    hit_boundary = True
                    if dy > 0.0:
                        dy = -dy * 0.5
                    if y0 + dy > h:
                        dy = h - y0

                # Gentle wall bias: add a small inward component when near edges
                # to reduce boundary-sliding / reflection loops.
                # (We keep it subtle because we normalize afterward.)
                bias_strength = 0.06  # fraction of intended magnitude
                if x0 < wall_thresh:
                    hit_boundary = True
                    if dx < 0.0:
                        dx = dx * 0.5
                    dx += mag * bias_strength
                if (w - x0) < wall_thresh:
                    hit_boundary = True
                    if dx > 0.0:
                        dx = dx * 0.5
                    dx -= mag * bias_strength

                if y0 < wall_thresh:
                    hit_boundary = True
                    if dy < 0.0:
                        dy = dy * 0.5
                    dy += mag * bias_strength
                if (h - y0) < wall_thresh:
                    hit_boundary = True
                    if dy > 0.0:
                        dy = dy * 0.5
                    dy -= mag * bias_strength

            dx_adj, dy_adj = _normalize_scale(dx, dy, mag)
            return dx_adj, dy_adj, hit_boundary

        def _in_obstacle(px: float, py: float) -> bool:
            return any(_point_in_obstacle(px, py, obs) for obs in obstacles)

        for agent_id, state in world_state.agents.items():
            if not state.alive:
                new_agents[agent_id] = state.model_copy(deep=True)
                continue

            action = actions.get(agent_id) or Action()
            config = self._configs.get(agent_id)
            max_speed = float(config.max_speed if config else 1.0)

            start_x = float(state.position.x)
            start_y = float(state.position.y)

            # Consecutive "stuck" tracking (no random exploration injection).
            stuck_steps_before = int(getattr(state, "stuck_steps", 0))
            total_stuck_before = int(getattr(state, "total_stuck_steps", 0))

            # Intended movement magnitude (after clamping to max_speed).
            dx_from_action = float(action.movement.x)
            dy_from_action = float(action.movement.y)
            len_from_action = math.sqrt(dx_from_action * dx_from_action + dy_from_action * dy_from_action)
            if len_from_action > max_speed and len_from_action > 0.0:
                scale = max_speed / len_from_action
                dx_from_action *= scale
                dy_from_action *= scale
                len_from_action = max_speed
            intended_mag = float(len_from_action)

            # Always use the LLM/policy movement vector (no injected exploration directions).
            dx0 = dx_from_action
            dy0 = dy_from_action

            # --- Attempt 1: boundary-aware adjustment + obstacle check ---
            dx_adj, dy_adj, hit_boundary = _boundary_adjust(dx0, dy0, start_x, start_y, intended_mag)
            cand_x = start_x + dx_adj
            cand_y = start_y + dy_adj

            if _in_obstacle(cand_x, cand_y):
                pos_x1, pos_y1 = start_x, start_y
                dx_applied1, dy_applied1 = 0.0, 0.0
            else:
                pos_x1, pos_y1 = _apply_boundary(cand_x, cand_y, w, h, self._boundary_mode)
                dx_applied1 = pos_x1 - start_x
                dy_applied1 = pos_y1 - start_y

            delta2_1 = dx_applied1 * dx_applied1 + dy_applied1 * dy_applied1
            blocked_initial = delta2_1 <= eps2

            action_debug = getattr(action, "movement_debug", None) or {}
            boundary_adjustment_applied = bool(hit_boundary)
            debug = {
                **action_debug,
                "blocked_movement": blocked_initial,
                "unstuck_triggered": False,
                "hit_boundary": hit_boundary,
                "boundary_adjustment_applied": boundary_adjustment_applied,
                "adjusted_movement": [dx_adj, dy_adj],
            }

            pos_x_final, pos_y_final = pos_x1, pos_y1
            dx_applied_final, dy_applied_final = dx_applied1, dy_applied1
            blocked_final = blocked_initial
            dx_adj_final, dy_adj_final = dx_adj, dy_adj
            hit_boundary_final = hit_boundary

            # Sustained block: halt in place (no physics fallbacks that could look like policy).
            stuck_halt = bool(blocked_initial and stuck_steps_before >= 3)

            # --- Fallback recovery only before sustained stuck threshold ---
            if blocked_initial and not stuck_halt:
                if verbose_block_print:
                    print(f"[movement] blocked agent={agent_id} step={world_state.step_index} initial=({dx_adj:.3f},{dy_adj:.3f})")

                # Determine a non-zero base direction for perturb/perp.
                base_dx = dx0
                base_dy = dy0
                base_len = math.sqrt(base_dx * base_dx + base_dy * base_dy)
                if base_len <= 1e-12:
                    # No direction at all; create a direction for fallback.
                    theta = self._rng.uniform(0.0, 2.0 * math.pi)
                    base_dx = math.cos(theta)
                    base_dy = math.sin(theta)
                    base_len = 1.0

                base_dx_unit = base_dx / base_len
                base_dy_unit = base_dy / base_len

                # Keep magnitude the same as the intended movement.
                if intended_mag <= 1e-12:
                    intended_mag = max_speed * 0.5

                base_angle = math.atan2(base_dy_unit, base_dx_unit)
                strategies = ["perturb", "perp"]
                if self._rng.random() < 0.5:
                    strategies = ["perp", "perturb"]

                # Try up to 2 directions (covers the ±15-30deg perturb OR perpendicular).
                for strat in strategies:
                    if strat == "perturb":
                        off_deg = float(self._rng.uniform(15.0, 30.0))
                        off = math.radians(off_deg)
                        sign = -1.0 if self._rng.random() < 0.5 else 1.0
                        ang = base_angle + sign * off
                        dx_fb = math.cos(ang) * intended_mag
                        dy_fb = math.sin(ang) * intended_mag
                    else:
                        # Perpendicular direction to the original vector.
                        perp_x = -base_dy_unit
                        perp_y = base_dx_unit
                        sign = -1.0 if self._rng.random() < 0.5 else 1.0
                        dx_fb = perp_x * sign * intended_mag
                        dy_fb = perp_y * sign * intended_mag

                    dx_adj2, dy_adj2, hit_boundary2 = _boundary_adjust(dx_fb, dy_fb, start_x, start_y, intended_mag)
                    cand_x2 = start_x + dx_adj2
                    cand_y2 = start_y + dy_adj2

                    if _in_obstacle(cand_x2, cand_y2):
                        pos_x2, pos_y2 = start_x, start_y
                        dx_applied2, dy_applied2 = 0.0, 0.0
                    else:
                        pos_x2, pos_y2 = _apply_boundary(cand_x2, cand_y2, w, h, self._boundary_mode)
                        dx_applied2 = pos_x2 - start_x
                        dy_applied2 = pos_y2 - start_y

                    delta2_2 = dx_applied2 * dx_applied2 + dy_applied2 * dy_applied2
                    blocked2 = delta2_2 <= eps2

                    # Update debug with the last adjustment that was actually tried.
                    dx_adj_final, dy_adj_final = dx_adj2, dy_adj2
                    hit_boundary_final = hit_boundary2
                    pos_x_final, pos_y_final = pos_x2, pos_y2
                    dx_applied_final, dy_applied_final = dx_applied2, dy_applied2
                    blocked_final = blocked2

                    if not blocked2:
                        break

                debug["hit_boundary"] = hit_boundary_final
                debug["boundary_adjustment_applied"] = bool(hit_boundary_final)
                debug["adjusted_movement"] = [dx_adj_final, dy_adj_final]
            elif stuck_halt:
                pos_x_final, pos_y_final = start_x, start_y
                dx_applied_final, dy_applied_final = 0.0, 0.0
                blocked_final = True
                debug["blocked_movement"] = True
                debug["movement_source"] = "stuck_halted"
                debug["hit_boundary"] = hit_boundary_final
                debug["boundary_adjustment_applied"] = bool(hit_boundary_final)
                debug["adjusted_movement"] = [0.0, 0.0]

            # Update stuck counters based on final movement.
            if blocked_final:
                stuck_steps_new = stuck_steps_before + 1
            else:
                stuck_steps_new = 0

            stuck_this_step = bool(blocked_final)
            total_stuck_new = total_stuck_before + (1 if stuck_this_step else 0)

            if not stuck_halt:
                debug["movement_source"] = getattr(action, "movement_source", "llm_target")
            debug["actual_movement"] = [dx_applied_final, dy_applied_final, 0.0]
            debug["blocked_movement"] = blocked_final

            new_pos = Vec3(x=pos_x_final, y=pos_y_final, z=state.position.z)
            new_vel = Vec3(
                x=dx_applied_final / self._decision_dt if self._decision_dt > 0 else 0.0,
                y=dy_applied_final / self._decision_dt if self._decision_dt > 0 else 0.0,
                z=0.0,
            )

            new_agents[agent_id] = AgentState(
                id=state.id,
                agent_type=state.agent_type,
                position=new_pos,
                velocity=new_vel,
                orientation=state.orientation,
                alive=state.alive,
                stuck_steps=stuck_steps_new,
                unstuck_steps_remaining=0,
                stuck_this_step=stuck_this_step,
                total_stuck_steps=total_stuck_new,
                last_movement_debug=debug,
            )

        return WorldState(
            time=world_state.time + self._decision_dt,
            step_index=world_state.step_index + 1,
            agents=new_agents,
            terrain=world_state.terrain.model_copy(deep=True),
            weather=world_state.weather.model_copy(deep=True),
            obstacles=world_state.obstacles,
        )


class SimulationEngine:
    """
    Central simulation controller.

    - Owns the current WorldState.
    - Uses PerceptionEngine, CommunicationRouter, and PhysicsEngine.
    - reset(): create initial world (hero + 4 villains, simple placement).
    - step(actions): snapshot → deliver messages → build observations (for
      external use; we don't call agents here) → apply actions → new state.
    - get_observations(): build observations from current state and last
      delivery (so the runner can get obs, call agents, then step(actions)).
    """

    def __init__(
        self,
        env_config: EnvironmentConfig,
        agent_configs: List[AgentConfig],
    ) -> None:
        self._env_config = env_config
        self._agent_configs = agent_configs
        self._rng = random.Random(env_config.seed)

        self._perception = PerceptionEngine(
            agent_configs,
            env_config.base_visibility_radius,
            weather_visibility_modifier=1.0,
            global_visibility_radius=env_config.visibility_radius,
            observation_noise_std=env_config.observation_noise_std,
            env_config=env_config,
            villain_hero_sight_radius=env_config.villain_hero_sight_radius,
            rng=self._rng,
        )
        self._comm = CommunicationRouter(
            agent_configs,
            message_delay_steps=env_config.message_delay_steps,
            message_budget_per_agent=env_config.message_budget_per_agent,
        )
        # Separate RNG stream so physics-only randomness does not perturb
        # obstacle generation / perception noise streams.
        self._physics_rng = random.Random(env_config.seed + 1337)
        self._physics = PhysicsEngine(
            agent_configs,
            tuple(env_config.world_size),
            env_config.decision_dt,
            boundary_mode=env_config.boundary_mode,
            rng=self._physics_rng,
        )

        self._world_state: WorldState = WorldState()
        self._last_delivery: Dict[str, List[Message]] = {}
        # Asymmetric spawn: angle from hero→villain_1 so villain_2 can be placed opposite.
        self._asymmetric_v1_base_angle: Optional[float] = None

    def _gen_hub_and_spokes(self, config: EnvironmentConfig) -> List[Obstacle]:
        obstacles: List[Obstacle] = []
        w, h = config.world_size
        cx, cy = w / 2.0, h / 2.0
        hub_radius = 25.0
        spoke_width = 6.0
        spoke_count = 6
        r = config.obstacle_radius
        spacing = r * 2.0
        spoke_angles = [(2 * math.pi * i / spoke_count) for i in range(spoke_count)]
        chokepoints: List[Tuple[float, float]] = []

        x = r
        while x < w - r:
            y = r
            while y < h - r:
                dist_from_center = math.hypot(x - cx, y - cy)
                if dist_from_center < hub_radius:
                    y += spacing
                    continue
                dx = x - cx
                dy = y - cy
                in_spoke = False
                for angle in spoke_angles:
                    sx = math.cos(angle)
                    sy = math.sin(angle)
                    px = -math.sin(angle)
                    py = math.cos(angle)
                    along = dx * sx + dy * sy
                    perp = dx * px + dy * py
                    if along > hub_radius - 2.0 and abs(perp) < spoke_width / 2.0:
                        in_spoke = True
                        break
                if not in_spoke:
                    obstacles.append(Obstacle(position=Vec3(x=x, y=y, z=0.0), radius=r))
                y += spacing
            x += spacing

        for angle in spoke_angles:
            choke_x = cx + math.cos(angle) * hub_radius
            choke_y = cy + math.sin(angle) * hub_radius
            chokepoints.append((choke_x, choke_y))

        self._env_config.chokepoint_positions = chokepoints
        return obstacles

    def _gen_asymmetric_labyrinth(self, config: EnvironmentConfig) -> List[Obstacle]:
        rng = random.Random(config.seed)
        obstacles: List[Obstacle] = []
        w, h = config.world_size
        r = config.obstacle_radius
        spacing = r * 2.0
        mid_x = w / 2.0
        bridge_y = rng.uniform(h * 0.3, h * 0.7)
        bridge_width = 8.0
        chokepoints = [(mid_x, bridge_y)]

        y = r
        while y < h - r:
            if abs(y - bridge_y) > bridge_width / 2.0:
                obstacles.append(Obstacle(position=Vec3(x=mid_x, y=y, z=0.0), radius=r))
            y += spacing

        open_density = 0.04
        open_area = max(0.0, (mid_x - 10.0)) * h
        open_count = int(open_area * open_density / (math.pi * r * r)) if r > 0 else 0

        placed = 0
        attempts = 0
        while placed < open_count and attempts < open_count * 10:
            x = rng.uniform(r + 2, max(r + 3, mid_x - r - 2))
            yy = rng.uniform(r + 2, h - r - 2)
            obstacles.append(Obstacle(position=Vec3(x=x, y=yy, z=0.0), radius=r))
            placed += 1
            attempts += 1

        grid_step = r * 3.5
        x = mid_x + grid_step
        while x < w - grid_step:
            yy = grid_step
            while yy < h - grid_step:
                if rng.random() > 0.25:
                    obstacles.append(Obstacle(position=Vec3(x=x, y=yy, z=0.0), radius=r))
                yy += grid_step
            x += grid_step

        self._env_config.chokepoint_positions = chokepoints
        return obstacles

    def _gen_gradient(self, config: EnvironmentConfig) -> List[Obstacle]:
        rng = random.Random(config.seed)
        obstacles: List[Obstacle] = []
        w, h = config.world_size
        r = config.obstacle_radius
        min_density = config.obstacle_density
        max_density = config.gradient_max_density
        col_width = r * 4.0
        x = r + 2
        while x < w - r - 2:
            t = x / w if w > 0 else 0.0
            col_density = min_density + t * (max_density - min_density)
            col_area = col_width * h
            denom = math.pi * r * r if r > 0 else 1.0
            expected = col_area * col_density / denom
            count = int(expected)
            if rng.random() < (expected - count):
                count += 1
            for _ in range(count):
                yy = rng.uniform(r + 2, h - r - 2)
                obstacles.append(Obstacle(position=Vec3(x=x, y=yy, z=0.0), radius=r))
            x += col_width
        self._env_config.chokepoint_positions = None
        return obstacles

    def _filter_spawn_unsafe(
        self, obstacles: List[Obstacle], spawn_positions: List[Vec3]
    ) -> List[Obstacle]:
        safe_radius = self._env_config.obstacle_radius * 4.0
        out: List[Obstacle] = []
        for obs in obstacles:
            ox, oy = float(obs.position.x), float(obs.position.y)
            too_close = False
            for sp in spawn_positions:
                if math.hypot(ox - sp.x, oy - sp.y) < safe_radius:
                    too_close = True
                    break
            if not too_close:
                out.append(obs)
        return out

    def _generate_obstacles(self, spawn_positions: List[Vec3]) -> List[Obstacle]:
        cfg = self._env_config
        cfg.chokepoint_positions = None
        template = cfg.map_template
        if template == MapTemplate.HUB_AND_SPOKES:
            raw = self._gen_hub_and_spokes(cfg)
        elif template == MapTemplate.ASYMMETRIC_LABYRINTH:
            raw = self._gen_asymmetric_labyrinth(cfg)
        elif template == MapTemplate.GRADIENT:
            raw = self._gen_gradient(cfg)
        else:
            raw = _generate_scattered_obstacles(
                rng=random.Random(cfg.seed),
                world_size=tuple(cfg.world_size),
                obstacle_density=cfg.obstacle_density,
                obstacle_radius=cfg.obstacle_radius,
            )
            self._env_config.chokepoint_positions = None
        return self._filter_spawn_unsafe(raw, spawn_positions)

    def _raw_obstacles_for_template(self, cfg: EnvironmentConfig) -> List[Obstacle]:
        """Procedural obstacles for the current map template (unfiltered)."""
        template = cfg.map_template
        if template == MapTemplate.HUB_AND_SPOKES:
            return self._gen_hub_and_spokes(cfg)
        if template == MapTemplate.ASYMMETRIC_LABYRINTH:
            return self._gen_asymmetric_labyrinth(cfg)
        if template == MapTemplate.GRADIENT:
            return self._gen_gradient(cfg)
        return _generate_scattered_obstacles(
            rng=random.Random(cfg.seed),
            world_size=tuple(cfg.world_size),
            obstacle_density=cfg.obstacle_density,
            obstacle_radius=cfg.obstacle_radius,
        )

    def _v2_not_inside_filtered_obstacles(
        self, px: float, py: float, obstacles: List[Obstacle]
    ) -> bool:
        """True if (px,py) is outside every obstacle disk plus 3-unit buffer (post filter)."""
        return not any(_near_obstacle(px, py, o) for o in obstacles)

    def _nearest_open_near_map_center(
        self,
        hero_xy: tuple[float, float],
        v1_xy: tuple[float, float],
        raw: List[Obstacle],
        w: float,
        h: float,
        margin: float,
    ) -> tuple[float, float]:
        """
        Search outward from map center for a point not inside any obstacle after
        hero+v1+candidate filtering (same as final placement rules).
        """
        cx, cy = w / 2.0, h / 2.0
        hx, hy = float(hero_xy[0]), float(hero_xy[1])
        vx, vy = float(v1_xy[0]), float(v1_xy[1])
        hero_v = Vec3(x=hx, y=hy, z=0.0)
        v1_v = Vec3(x=vx, y=vy, z=0.0)
        max_r = max(w, h) * 0.55
        ring_step = 5.0
        r = 0.0
        while r <= max_r:
            n_angles = max(36, int(2 * math.pi * r / ring_step) if r > 1e-6 else 1)
            for k in range(n_angles):
                theta = 2.0 * math.pi * k / float(n_angles)
                px = cx + r * math.cos(theta)
                py = cy + r * math.sin(theta)
                px = max(margin, min(w - margin, px))
                py = max(margin, min(h - margin, py))
                v2_v = Vec3(x=px, y=py, z=0.0)
                obs_f = self._filter_spawn_unsafe(raw, [hero_v, v1_v, v2_v])
                if self._v2_not_inside_filtered_obstacles(px, py, obs_f):
                    return (px, py)
            r += ring_step
        return (
            max(margin, min(w - margin, cx)),
            max(margin, min(h - margin, cy)),
        )

    def _place_asymmetric_v2_structured_map(
        self,
        hero_xy: tuple[float, float],
        v1_xy: tuple[float, float],
        rng: random.Random,
    ) -> tuple[float, float]:
        """
        Chicken-and-egg fix: template obstacles are deterministic, so generate them
        temporarily with candidate hero/v1/v2 spawns, filter like ``_generate_obstacles``,
        and reject v2 positions that remain inside a wall disk.

        Search order: configured ``far`` distance with 50 angles (7.2° steps), then
        distances 55, 50, 45, 40 (deduped). Fallback: nearest open point to map center.
        """
        _ = rng  # reserved for future jitter; search is deterministic per geometry
        cfg = self._env_config
        w, h = float(cfg.world_size[0]), float(cfg.world_size[1])
        margin = 10.0
        hx, hy = float(hero_xy[0]), float(hero_xy[1])
        vx, vy = float(v1_xy[0]), float(v1_xy[1])
        raw = self._raw_obstacles_for_template(cfg)
        hero_v = Vec3(x=hx, y=hy, z=0.0)
        v1_v = Vec3(x=vx, y=vy, z=0.0)

        def _filtered_for_v2(px: float, py: float) -> List[Obstacle]:
            v2_v = Vec3(x=px, y=py, z=0.0)
            return self._filter_spawn_unsafe(raw, [hero_v, v1_v, v2_v])

        # hub_and_spokes-specific override:
        # place villain_2 inside open hub (radius 25 around map center),
        # while keeping at least 30 units from hero.
        if cfg.map_template == MapTemplate.HUB_AND_SPOKES:
            cx, cy = w / 2.0, h / 2.0
            hub_radius = 25.0
            min_hero_dist = 30.0
            for _ in range(200):
                theta = rng.uniform(0.0, 2.0 * math.pi)
                r = hub_radius * math.sqrt(rng.random())
                px = cx + r * math.cos(theta)
                py = cy + r * math.sin(theta)
                px = max(margin, min(w - margin, px))
                py = max(margin, min(h - margin, py))
                if math.hypot(px - hx, py - hy) < min_hero_dist:
                    continue
                obs_f = _filtered_for_v2(px, py)
                if self._v2_not_inside_filtered_obstacles(px, py, obs_f):
                    return (px, py)
            # Deterministic fallback if random hub sampling fails.
            return self._nearest_open_near_map_center(hero_xy, v1_xy, raw, w, h, margin)

        far_cfg = float(getattr(cfg, "asymmetric_far_distance", 60.0))
        distance_candidates: list[float] = []
        seen_d: set[float] = set()
        for d in (far_cfg, 55.0, 50.0, 45.0, 40.0):
            if d > 0 and d not in seen_d:
                seen_d.add(d)
                distance_candidates.append(d)

        for dist in distance_candidates:
            for k in range(50):
                theta = 2.0 * math.pi * k / 50.0
                px = hx + dist * math.cos(theta)
                py = hy + dist * math.sin(theta)
                px = max(margin, min(w - margin, px))
                py = max(margin, min(h - margin, py))
                obs_f = _filtered_for_v2(px, py)
                if self._v2_not_inside_filtered_obstacles(px, py, obs_f):
                    return (px, py)

        return self._nearest_open_near_map_center(hero_xy, v1_xy, raw, w, h, margin)

    def _spawn_villain_near_hero(
        self,
        hero_pos: tuple[float, float],
        villain_index: int,
        max_radius: float,
        min_radius: float = 20.0,
        rng: random.Random | None = None,
        existing_spawns: list[tuple[float, float]] | None = None,
    ) -> tuple[float, float]:
        cfg = self._env_config
        if rng is None:
            rng = self._rng
        existing = list(existing_spawns or [])
        hero_x, hero_y = float(hero_pos[0]), float(hero_pos[1])
        w, h = float(cfg.world_size[0]), float(cfg.world_size[1])
        margin = 10.0
        world_w, world_h = w, h

        if getattr(cfg, "spawn_mode", "random") == "asymmetric":
            close = float(getattr(cfg, "asymmetric_close_distance", 12.0))
            far = float(getattr(cfg, "asymmetric_far_distance", 60.0))
            if villain_index == 0:
                ang = rng.uniform(0.0, 2.0 * math.pi)
                self._asymmetric_v1_base_angle = ang
                x = hero_x + close * math.cos(ang)
                y = hero_y + close * math.sin(ang)
                return (
                    max(margin, min(w - margin, x)),
                    max(margin, min(h - margin, y)),
                )
            if villain_index == 1:
                base = self._asymmetric_v1_base_angle
                if base is None:
                    base = rng.uniform(0.0, 2.0 * math.pi)
                v2_angle = base + math.pi + rng.uniform(-math.pi / 4, math.pi / 4)
                x = hero_x + far * math.cos(v2_angle)
                y = hero_y + far * math.sin(v2_angle)
                return (
                    max(margin, min(w - margin, x)),
                    max(margin, min(h - margin, y)),
                )

        base_angle = rng.uniform(0.0, 2.0 * math.pi)
        max_tries = 50
        for t in range(max_tries):
            radius = rng.uniform(float(min_radius), float(max_radius))
            angle = base_angle + rng.uniform(-0.8, 0.8) + t * 0.27
            x = hero_x + radius * math.cos(angle)
            y = hero_y + radius * math.sin(angle)
            x = max(margin, min(world_w - margin, x))
            y = max(margin, min(world_h - margin, y))
            too_close = any(math.hypot(x - sx, y - sy) < 15.0 for sx, sy in existing)
            if not too_close:
                return (x, y)
        radius = (float(min_radius) + float(max_radius)) / 2.0
        x = max(margin, min(world_w - margin, hero_x + radius * math.cos(base_angle)))
        y = max(margin, min(world_h - margin, hero_y + radius * math.sin(base_angle)))
        return (x, y)

    def reset(self) -> WorldState:
        """Create and set initial WorldState; return it."""
        w, h = self._env_config.world_size
        agents: Dict[str, AgentState] = {}
        self._asymmetric_v1_base_angle = None

        hero_cfgs = [c for c in self._agent_configs if c.agent_type == AgentType.HERO]
        villain_cfgs = [c for c in self._agent_configs if c.agent_type == AgentType.VILLAIN]

        hero_pos: Optional[tuple[float, float]] = None
        if hero_cfgs:
            c = hero_cfgs[0]
            # Spawn near center with bounded random offset.
            cx, cy = float(w) * 0.5, float(h) * 0.5
            offset_x = float(self._rng.uniform(-10.0, 10.0))
            offset_y = float(self._rng.uniform(-10.0, 10.0))
            margin = 5.0
            hx = max(margin, min(float(w) - margin, cx + offset_x))
            hy = max(margin, min(float(h) - margin, cy + offset_y))
            hero_pos = (hx, hy)
            agents[c.id] = AgentState(
                id=c.id,
                agent_type=AgentType.HERO,
                position=Vec3(x=hx, y=hy, z=0.0),
                velocity=Vec3(x=0.0, y=0.0, z=0.0),
                orientation=0.0,
                alive=True,
            )
        villain_positions: list[tuple[float, float]] = []
        spawn_radius = float(getattr(self._env_config, "villain_spawn_radius", 50.0) or 50.0)
        cfg = self._env_config
        structured_v2_maps = cfg.map_template in (
            MapTemplate.HUB_AND_SPOKES,
            MapTemplate.ASYMMETRIC_LABYRINTH,
            MapTemplate.GRADIENT,
        )
        for i, c in enumerate(villain_cfgs):
            if hero_pos is not None and spawn_radius > 0.0:
                use_structured_v2 = (
                    i == 1
                    and len(villain_cfgs) >= 2
                    and getattr(cfg, "spawn_mode", "random") == "asymmetric"
                    and structured_v2_maps
                )
                if use_structured_v2 and villain_positions:
                    x, y = self._place_asymmetric_v2_structured_map(
                        hero_pos, villain_positions[0], self._rng
                    )
                else:
                    x, y = self._spawn_villain_near_hero(
                        hero_pos=hero_pos,
                        villain_index=i,
                        max_radius=spawn_radius,
                        min_radius=20.0,
                        rng=self._rng,
                        existing_spawns=villain_positions,
                    )
            else:
                margin = 10.0
                x = self._rng.uniform(margin, float(w) - margin)
                y = self._rng.uniform(margin, float(h) - margin)

            agents[c.id] = AgentState(
                id=c.id,
                agent_type=AgentType.VILLAIN,
                position=Vec3(x=x, y=y, z=0.0),
                velocity=Vec3(x=0.0, y=0.0, z=0.0),
                orientation=0.0,
                alive=True,
            )
            villain_positions.append((float(x), float(y)))

        spawn_positions = [a.position for a in agents.values()]
        obstacles = self._generate_obstacles(spawn_positions)

        self._world_state = WorldState(
            time=0.0,
            step_index=0,
            agents=agents,
            terrain=TerrainInfo(
                terrain_type=self._env_config.terrain_type,
                slope_steepness=self._env_config.slope_steepness,
            ),
            weather=WeatherInfo(
                weather_type=self._env_config.weather_type,
                visibility_modifier=1.0,
            ),
            obstacles=obstacles,
        )
        self._last_delivery = {}
        self._comm.reset()

        if self._env_config.inject_villain_start_positions:
            # One-time conditioning: each villain receives the other villain’s
            # exact starting position as an "incoming message" in the first
            # observation cycle (before any agent actions happen).
            villains = [
                a for a in self._world_state.agents.values() if a.alive and a.agent_type == AgentType.VILLAIN
            ]
            if len(villains) >= 2:
                delivery: Dict[str, List[Message]] = {}
                for recipient in villains:
                    msgs: List[Message] = []
                    for sender in villains:
                        if sender.id == recipient.id:
                            continue
                        payload = [
                            float(sender.position.x),
                            float(sender.position.y),
                            1.0,  # confidence
                            float(recipient.position.x),
                            float(recipient.position.y),
                        ]
                        msgs.append(
                            Message(
                                sender_id=sender.id,
                                recipient_ids=[recipient.id],
                                payload=payload,
                                channel="initial_villain_pos",
                            )
                        )
                    delivery[recipient.id] = msgs
                self._last_delivery = delivery

        return self._world_state.model_copy(deep=True)

    def get_observations(self) -> Dict[str, Observation]:
        """
        Build observations for the current world state using last delivered
        messages. Call this before step(actions) so the runner can pass
        observations to agents and collect actions.
        """
        return self._perception.build_observations(
            self._world_state,
            self._last_delivery,
        )

    def step(self, actions: Dict[str, Action]) -> WorldState:
        """
        Snapshot-update cycle:
        1. Submit messages from actions to CommunicationRouter.
        2. Deliver messages for next observation (stored for get_observations).
        3. Apply physics (movement, bounds) → new WorldState.
        4. Store and return new state.
        """
        for agent_id, action in actions.items():
            if action.message is not None:
                self._comm.submit(action.message, self._world_state.step_index)

        self._last_delivery = self._comm.deliver(self._world_state, self._world_state.step_index)
        self._world_state = self._physics.apply(self._world_state, actions)
        return self._world_state.model_copy(deep=True)

    @property
    def world_state(self) -> WorldState:
        """Current world state (read-only snapshot)."""
        return self._world_state.model_copy(deep=True)

    @property
    def env_config(self) -> EnvironmentConfig:
        return self._env_config

    @property
    def agent_configs(self) -> List[AgentConfig]:
        return list(self._agent_configs)

    def message_sent_counts(self) -> Dict[str, int]:
        """Cumulative messages submitted per agent this episode (for logging / budgets)."""
        return dict(self._comm._sent_count)

    def message_budget_remaining(self, agent_id: str) -> Optional[int]:
        """Remaining sends for agent_id if a per-agent budget is configured; else None."""
        cap = self._env_config.message_budget_per_agent
        if cap is None:
            return None
        used = int(self._comm._sent_count.get(agent_id, 0))
        return max(0, int(cap) - used)
