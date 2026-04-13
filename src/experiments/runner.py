from __future__ import annotations

import functools
import json
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from ..core.engine import SimulationEngine
from ..core.map_validator import MapComplexityValidator, _build_occupancy_grid
from ..core.models import Action, AgentConfig, AgentType, EnvironmentConfig, Observation, Obstacle, Vec3, WorldState
from ..agents.base import Agent
from ..agents.baseline_agent import RuleBasedAgent
from ..agents.factory import create_agent, LLMClient
from ..agents.greedy_agent import GreedyPursuerAgent
from ..agents.llm_agent import LLMAgent
from ..agents.self_steer_agent import SelfSteerAgent
from ..agents.schema import ALLOWED_INTENT_VALUES, LLMActionOutput
from ..metrics.role_divergence_metrics import within_episode_role_divergence

_logger = logging.getLogger(__name__)


@dataclass
class EpisodeConfig:
    episode_id: str
    environment: EnvironmentConfig
    agent_configs: List[AgentConfig]
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 2
    history_limit: int = 8
    capture_radius: float = 2.0


@dataclass
class EpisodeOutcome:
    episode_id: str
    result: Literal["hero_captured", "hero_escaped", "time_limit", "error"]
    steps: int
    capture_step_index: Optional[int] = None
    winner_team: Optional[str] = None
    error_message: Optional[str] = None
    within_episode_divergence: Optional[Dict[str, Any]] = None


@dataclass
class StepLogEntry:
    episode_id: str
    step_index: int
    time: float
    hero_position: Optional[List[float]]
    villain_positions: Dict[str, List[float]]
    per_agent: List[Dict]


def _extract_positions(world_state: WorldState) -> tuple[Optional[List[float]], Dict[str, List[float]]]:
    hero_pos: Optional[List[float]] = None
    villains: Dict[str, List[float]] = {}
    for a in world_state.agents.values():
        if not a.alive:
            continue
        if a.agent_type.value == "hero":
            hero_pos = [a.position.x, a.position.y, a.position.z]
        elif a.agent_type.value == "villain":
            villains[a.id] = [a.position.x, a.position.y, a.position.z]
    return hero_pos, villains


def _check_capture_or_done(
    world_state: WorldState,
    capture_radius: float,
    max_steps: int,
    episode_id: str,
) -> Optional[EpisodeOutcome]:
    hero = None
    villains = []
    for a in world_state.agents.values():
        if not a.alive:
            continue
        if a.agent_type.value == "hero":
            hero = a
        elif a.agent_type.value == "villain":
            villains.append(a)

    if hero is None:
        return EpisodeOutcome(
            episode_id=episode_id,
            result="hero_captured",
            steps=world_state.step_index,
            capture_step_index=world_state.step_index,
            winner_team="villains",
        )

    r2 = capture_radius * capture_radius
    for v in villains:
        dx = v.position.x - hero.position.x
        dy = v.position.y - hero.position.y
        if dx * dx + dy * dy <= r2:
            return EpisodeOutcome(
                episode_id=episode_id,
                result="hero_captured",
                steps=world_state.step_index,
                capture_step_index=world_state.step_index,
                winner_team="villains",
            )

    if world_state.step_index >= max_steps:
        return EpisodeOutcome(
            episode_id=episode_id,
            result="hero_escaped",
            steps=world_state.step_index,
            capture_step_index=None,
            winner_team="heroes",
        )

    return None


def _parallel_agent_step(
    agents: Dict[str, Agent],
    observations,
) -> Dict[str, Action]:
    actions: Dict[str, Action] = {}

    def safe_step(aid: str, agent: Agent) -> Action:
        try:
            return agent.step(observations[aid])
        except Exception:
            return Action(
                movement=Vec3(x=0.0, y=0.0, z=0.0),
                message=None,
                intent=LLMActionOutput.normalize_intent("unexpected_error"),
                movement_source="fallback_explore",
            )

    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {
            executor.submit(safe_step, aid, agent): aid
            for aid, agent in agents.items()
        }
        for future in as_completed(futures):
            aid = futures[future]
            actions[aid] = future.result()
    return actions


def _warn_intent_vocabulary(per_agent: List[Dict]) -> None:
    for p in per_agent:
        intent = p.get("intent")
        if intent is None:
            continue
        if intent not in ALLOWED_INTENT_VALUES:
            _logger.warning("Step log intent not in ALLOWED_INTENT_VALUES: %r", intent)


def _villain_visibility_flags(observation: Observation) -> tuple[bool, bool]:
    sx = float(observation.self_state.position.x)
    sy = float(observation.self_state.position.y)
    sight_r2 = float(observation.villain_hero_sight_radius) ** 2
    hero_engine = False
    hero_truly = False
    for a in observation.visible_agents:
        if a.agent_type == AgentType.HERO and a.alive:
            hero_engine = True
            dx = float(a.position.x) - sx
            dy = float(a.position.y) - sy
            if (dx * dx + dy * dy) <= sight_r2:
                hero_truly = True
            break
    return hero_truly, hero_engine


def _angle_deg_intended_vs_actual(ivx: float, ivy: float, avx: float, avy: float) -> Optional[float]:
    """Angle (degrees) between intended vector (toward LLM target from pre-step position) and actual displacement."""
    li = math.hypot(ivx, ivy)
    la = math.hypot(avx, avy)
    if li < 1e-9 or la < 1e-9:
        return None
    dot = ivx * avx + ivy * avy
    c = max(-1.0, min(1.0, dot / (li * la)))
    return float(math.degrees(math.acos(c)))


def _apply_pid_and_angle_metrics(row: Dict[str, Any], actual_movement: List[float]) -> None:
    """Populate pid and intent_execution_angle_delta; null llm_target_position => null metrics."""
    ax, ay = float(row["actual_position"][0]), float(row["actual_position"][1])
    mx, my = float(actual_movement[0]), float(actual_movement[1])
    sx, sy = ax - mx, ay - my
    tp = row.get("llm_target_position")
    if tp is not None and isinstance(tp, (list, tuple)) and len(tp) >= 2:
        tx, ty = float(tp[0]), float(tp[1])
        row["pid"] = math.hypot(tx - ax, ty - ay)
    else:
        row["pid"] = None

    raw = row.get("raw_llm_movement_vector")
    if (
        raw is not None
        and isinstance(raw, (list, tuple))
        and len(raw) >= 2
        and tp is not None
        and isinstance(tp, (list, tuple))
        and len(tp) >= 2
    ):
        tx, ty = float(tp[0]), float(tp[1])
        rx, ry = float(raw[0]), float(raw[1])
        row["intent_execution_angle_delta"] = _angle_deg_intended_vs_actual(tx - sx, ty - sy, rx, ry)
    else:
        row["intent_execution_angle_delta"] = None


_RES_GEOM = 2.0

_NULL_TARGET_GEOM: Dict[str, Any] = {
    "target_in_obstacle": None,
    "target_bfs_reachable": None,
    "target_bfs_path_length": None,
    "target_straight_distance": None,
    "target_detour_ratio": None,
}


def _obstacle_signature(obstacles: Sequence[Obstacle]) -> Tuple[Tuple[float, float, float], ...]:
    obs_list = list(obstacles)
    obs_list.sort(key=lambda o: (float(o.position.x), float(o.position.y), float(o.radius)))
    return tuple((float(o.position.x), float(o.position.y), float(o.radius)) for o in obs_list)


@functools.lru_cache(maxsize=64)
def _cached_vertex_blocked_grid(
    world_w: float,
    world_h: float,
    obs_sig: Tuple[Tuple[float, float, float], ...],
) -> np.ndarray:
    obstacles = [
        Obstacle(position=Vec3(x=a, y=b, z=0.0), radius=r)
        for a, b, r in obs_sig
    ]
    return _build_occupancy_grid(
        obstacles,
        (world_w, world_h),
        _RES_GEOM,
        agent_radius=0.0,
        sampling="vertex",
    )


def _walkable_vertex_grid(obstacles: Sequence[Obstacle], world_size: Tuple[float, float]) -> np.ndarray:
    ws = (float(world_size[0]), float(world_size[1]))
    sig = _obstacle_signature(obstacles)
    blocked = _cached_vertex_blocked_grid(ws[0], ws[1], sig)
    return np.logical_not(blocked)


def _world_xy_to_grid_indices_vertex(
    x: float,
    y: float,
    world_size: Tuple[float, float],
    nx: int,
    ny: int,
) -> Tuple[int, int]:
    w, h = float(world_size[0]), float(world_size[1])
    xw = min(w, max(0.0, x))
    yw = min(h, max(0.0, y))
    ix = int(xw / _RES_GEOM)
    iy = int(yw / _RES_GEOM)
    return max(0, min(nx - 1, ix)), max(0, min(ny - 1, iy))


def _snap_to_nearest_free_cell(
    ix: int,
    iy: int,
    walkable: np.ndarray,
) -> Optional[Tuple[int, int]]:
    nx, ny = walkable.shape
    if walkable[ix, iy]:
        return (ix, iy)
    q: deque[Tuple[int, int]] = deque([(ix, iy)])
    seen = {(ix, iy)}
    while q:
        cx, cy = q.popleft()
        if walkable[cx, cy]:
            return (cx, cy)
        for dx, dy in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, -1),
        ):
            nx2, ny2 = cx + dx, cy + dy
            if nx2 < 0 or ny2 < 0 or nx2 >= nx or ny2 >= ny:
                continue
            if (nx2, ny2) in seen:
                continue
            seen.add((nx2, ny2))
            q.append((nx2, ny2))
    return None


def _tp_pair_ok(tp: Any) -> bool:
    return tp is not None and isinstance(tp, (list, tuple)) and len(tp) >= 2


def _raw_target_out_of_bounds(raw_tp: Any, world_size: Tuple[float, float]) -> bool:
    """True if raw target exists and lies outside the closed world rectangle [0, w] × [0, h]."""
    if not _tp_pair_ok(raw_tp):
        return False
    rx, ry = float(raw_tp[0]), float(raw_tp[1])
    ww, wh = float(world_size[0]), float(world_size[1])
    return not (0.0 <= rx <= ww and 0.0 <= ry <= wh)


def _target_point_in_obstacle(tx: float, ty: float, obstacles: Sequence[Obstacle]) -> bool:
    for obs in obstacles:
        ox = float(obs.position.x)
        oy = float(obs.position.y)
        r = float(obs.radius)
        if r <= 0.0:
            continue
        if math.hypot(tx - ox, ty - oy) <= r:
            return True
    return False


def _compute_target_geometry_metrics(
    llm_target_position: Any,
    llm_raw_target_position: Any,
    agent_start_pos: Tuple[float, float],
    obstacles: Sequence[Obstacle],
    world_size: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Spatial validity on the 2.0 vertex BFS grid (validator-aligned).
    Uses ``llm_target_position`` when set; otherwise ``llm_raw_target_position`` (pre-drop).
    Raw-only targets outside the world rectangle get partial metrics per spec (no BFS).
    """
    tgv_tp = llm_target_position if _tp_pair_ok(llm_target_position) else llm_raw_target_position
    if not _tp_pair_ok(tgv_tp):
        return dict(_NULL_TARGET_GEOM)

    if (
        not _tp_pair_ok(llm_target_position)
        and _tp_pair_ok(llm_raw_target_position)
        and _raw_target_out_of_bounds(llm_raw_target_position, world_size)
    ):
        return {
            "target_in_obstacle": False,
            "target_bfs_reachable": False,
            "target_bfs_path_length": None,
            "target_straight_distance": None,
            "target_detour_ratio": None,
        }

    tx = float(tgv_tp[0])
    ty = float(tgv_tp[1])
    sx, sy = float(agent_start_pos[0]), float(agent_start_pos[1])

    in_obs = _target_point_in_obstacle(tx, ty, obstacles)
    straight = math.hypot(tx - sx, ty - sy)

    walkable = _walkable_vertex_grid(obstacles, world_size)
    nx, ny = walkable.shape

    six, siy = _world_xy_to_grid_indices_vertex(sx, sy, world_size, nx, ny)
    gix, giy = _world_xy_to_grid_indices_vertex(tx, ty, world_size, nx, ny)

    start_cell = _snap_to_nearest_free_cell(six, siy, walkable)
    goal_cell = _snap_to_nearest_free_cell(gix, giy, walkable)

    bfs_len: Optional[float] = None
    reachable = False
    if start_cell is not None and goal_cell is not None:
        bfs_len = MapComplexityValidator._bfs_path_len_numpy(
            start_cell, goal_cell, walkable, _RES_GEOM
        )
        reachable = bfs_len is not None

    # BFS at 2.0-unit resolution can underestimate path length for short distances due to grid quantization; clamp to theoretical minimum of 1.0.
    detour: Optional[float] = None
    if bfs_len is not None and straight > 1e-6:
        detour = max(1.0, float(bfs_len / straight))

    return {
        "target_in_obstacle": in_obs,
        "target_bfs_reachable": reachable,
        "target_bfs_path_length": float(bfs_len) if bfs_len is not None else None,
        "target_straight_distance": float(straight),
        "target_detour_ratio": detour,
    }


def _build_step_log(
    episode_id: str,
    world_state: WorldState,
    agents: Dict[str, Agent],
    actions: Dict[str, Action],
    observations: Dict[str, Observation],
    engine: SimulationEngine,
) -> StepLogEntry:
    hero_pos, villains = _extract_positions(world_state)
    per_agent: List[Dict] = []
    ws = tuple(float(x) for x in engine.env_config.world_size)
    obs_list = list(world_state.obstacles)

    for aid, agent in agents.items():
        st = world_state.agents.get(aid)
        obs = observations.get(aid)
        act = actions.get(aid)
        if st is None or not st.alive:
            continue

        md = getattr(st, "last_movement_debug", None) or {}
        am = md.get("actual_movement")
        if isinstance(am, (list, tuple)) and len(am) >= 2:
            actual_movement = [float(am[0]), float(am[1])]
        else:
            actual_movement = [0.0, 0.0]

        boundary_hit = bool(md.get("hit_boundary", False))
        blocked = bool(md.get("blocked_movement", False))
        obstacle_collision = bool(blocked) and not boundary_hit

        msgs_in = len(obs.incoming_messages) if obs is not None else 0
        msgs_out = 1 if (act is not None and act.message is not None) else 0
        budget_left = engine.message_budget_remaining(aid)

        row: Dict[str, Any] = {
            "agent_id": aid,
            "role": agent.config.agent_type.value,
            "actual_position": [float(st.position.x), float(st.position.y)],
            "actual_movement": actual_movement,
            "raw_llm_movement_vector": md.get("raw_llm_movement"),
            "stuck_this_step": bool(getattr(st, "stuck_this_step", False)),
            "boundary_hit": boundary_hit,
            "obstacle_collision": obstacle_collision,
            "messages_sent": int(msgs_out),
            "messages_received": int(msgs_in),
            "message_budget_remaining": budget_left,
        }

        if isinstance(agent, (LLMAgent, SelfSteerAgent)):
            session = agent.session
            last_turn = session.recent_turns(1)[0] if session.turn_history else None
            if last_turn is not None and last_turn.action is not None:
                action = last_turn.action
                row.update(
                    {
                        "intent": action.intent,
                        "llm_raw_intent": getattr(last_turn, "llm_raw_intent", None),
                        "llm_target_position": (
                            [float(action.llm_target_position[0]), float(action.llm_target_position[1])]
                            if action.llm_target_position is not None
                            else None
                        ),
                        "llm_raw_target_position": (
                            [float(action.llm_raw_target_position[0]), float(action.llm_raw_target_position[1])]
                            if getattr(action, "llm_raw_target_position", None) is not None
                            else None
                        ),
                        "raw_llm_response": last_turn.raw_response,
                        "prompt": last_turn.prompt,
                        "llm_confidence": float(getattr(action, "llm_confidence", 1.0)),
                        "movement_source": getattr(action, "movement_source", None),
                        "movement": [action.movement.x, action.movement.y, action.movement.z],
                        "movement_vector": [float(action.movement.x), float(action.movement.y)],
                        "actual_movement_vector": [float(actual_movement[0]), float(actual_movement[1])],
                        "movement_debug": getattr(action, "movement_debug", None) or {},
                        "used_fallback": not last_turn.valid,
                        "fallback_reason": last_turn.error,
                    }
                )
                _apply_pid_and_angle_metrics(row, actual_movement)
                row["target_out_of_bounds"] = _raw_target_out_of_bounds(row.get("llm_raw_target_position"), ws)
                start_xy = (
                    float(row["actual_position"][0]) - float(actual_movement[0]),
                    float(row["actual_position"][1]) - float(actual_movement[1]),
                )
                row.update(
                    _compute_target_geometry_metrics(
                        row.get("llm_target_position"),
                        row.get("llm_raw_target_position"),
                        start_xy,
                        obs_list,
                        ws,
                    )
                )
                if agent.config.agent_type == AgentType.VILLAIN and obs is not None:
                    ht, he = _villain_visibility_flags(obs)
                    row["hero_truly_visible"] = ht
                    row["hero_in_engine_obs"] = he
                    row["steps_since_hero_seen"] = int(getattr(agent, "steps_since_seen", 0))
                else:
                    row["hero_truly_visible"] = None
                    row["hero_in_engine_obs"] = None
                    row["steps_since_hero_seen"] = None
                per_agent.append(row)
        elif isinstance(agent, GreedyPursuerAgent):
            if act is not None:
                row.update(
                    {
                        "intent": act.intent,
                        "llm_raw_intent": None,
                        "llm_target_position": None,
                        "llm_raw_target_position": None,
                        "raw_llm_response": None,
                        "llm_confidence": 1.0,
                        "movement_source": getattr(act, "movement_source", "greedy_baseline"),
                        "movement": [act.movement.x, act.movement.y, act.movement.z],
                        "movement_vector": [float(act.movement.x), float(act.movement.y)],
                        "actual_movement_vector": [float(actual_movement[0]), float(actual_movement[1])],
                        "movement_debug": {},
                        "used_fallback": False,
                        "fallback_reason": None,
                    }
                )
                _apply_pid_and_angle_metrics(row, actual_movement)
                row["target_out_of_bounds"] = False
                start_xy = (
                    float(row["actual_position"][0]) - float(actual_movement[0]),
                    float(row["actual_position"][1]) - float(actual_movement[1]),
                )
                row.update(
                    _compute_target_geometry_metrics(
                        None,
                        None,
                        start_xy,
                        obs_list,
                        ws,
                    )
                )
                if agent.config.agent_type == AgentType.VILLAIN and obs is not None:
                    ht, he = _villain_visibility_flags(obs)
                    row["hero_truly_visible"] = ht
                    row["hero_in_engine_obs"] = he
                    row["steps_since_hero_seen"] = int(getattr(agent, "steps_since_seen", 0))
                else:
                    row["hero_truly_visible"] = None
                    row["hero_in_engine_obs"] = None
                    row["steps_since_hero_seen"] = None
                per_agent.append(row)
        elif isinstance(agent, RuleBasedAgent):
            la = getattr(agent, "_last_action", None)
            if la is not None:
                row.update(
                    {
                        "intent": la.intent,
                        "llm_raw_intent": None,
                        "llm_target_position": None,
                        "llm_raw_target_position": None,
                        "raw_llm_response": None,
                        "llm_confidence": float(getattr(la, "llm_confidence", 1.0)),
                        "movement_source": la.movement_source,
                        "movement": [la.movement.x, la.movement.y, la.movement.z],
                        "movement_vector": [float(la.movement.x), float(la.movement.y)],
                        "actual_movement_vector": [float(actual_movement[0]), float(actual_movement[1])],
                        "movement_debug": getattr(la, "movement_debug", None) or {},
                        "used_fallback": False,
                        "fallback_reason": None,
                    }
                )
                _apply_pid_and_angle_metrics(row, actual_movement)
                row["target_out_of_bounds"] = False
                start_xy = (
                    float(row["actual_position"][0]) - float(actual_movement[0]),
                    float(row["actual_position"][1]) - float(actual_movement[1]),
                )
                row.update(
                    _compute_target_geometry_metrics(
                        row.get("llm_target_position"),
                        row.get("llm_raw_target_position"),
                        start_xy,
                        obs_list,
                        ws,
                    )
                )
                if agent.config.agent_type == AgentType.VILLAIN and obs is not None:
                    ht, he = _villain_visibility_flags(obs)
                    row["hero_truly_visible"] = ht
                    row["hero_in_engine_obs"] = he
                    row["steps_since_hero_seen"] = int(getattr(agent, "steps_since_seen", 0))
                else:
                    row["hero_truly_visible"] = None
                    row["hero_in_engine_obs"] = None
                    row["steps_since_hero_seen"] = None
                per_agent.append(row)

    _warn_intent_vocabulary(per_agent)

    return StepLogEntry(
        episode_id=episode_id,
        step_index=world_state.step_index,
        time=world_state.time,
        hero_position=hero_pos,
        villain_positions=villains,
        per_agent=per_agent,
    )


# Steps where movement still reflects a successful LLM plan (incl. post-clamp geometry).
_LLM_POLICY_MOVEMENT_SOURCES = frozenset(
    {
        "llm_target",
        "llm_vector_legacy",
        "boundary_override",  # target from LLM; world clamp adjusted displacement
    }
)


def _llm_driven_step_fraction(step_logs: List[StepLogEntry], agent_ids: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for aid in agent_ids:
        driven = 0
        total = 0
        for s in step_logs:
            for p in s.per_agent:
                if p.get("agent_id") != aid:
                    continue
                total += 1
                ms = (p.get("movement_source") or "").lower()
                if ms in _LLM_POLICY_MOVEMENT_SOURCES:
                    driven += 1
        out[aid] = float(driven) / float(total) if total else 0.0
    return out


def _episode_config_jsonable(episode_config: EpisodeConfig) -> Dict[str, Any]:
    """EpisodeConfig embeds Pydantic models; dataclasses.asdict() is not JSON-safe."""
    return {
        "episode_id": episode_config.episode_id,
        "llm_timeout_seconds": episode_config.llm_timeout_seconds,
        "llm_max_retries": episode_config.llm_max_retries,
        "history_limit": episode_config.history_limit,
        "capture_radius": episode_config.capture_radius,
        "environment": episode_config.environment.model_dump(mode="json"),
        "agent_configs": [c.model_dump(mode="json") for c in episode_config.agent_configs],
    }


def _episode_summary_extras(
    episode_config: EpisodeConfig,
    step_logs: List[StepLogEntry],
    world_state: WorldState,
) -> Dict[str, Any]:
    agent_ids = [c.id for c in episode_config.agent_configs]
    stuck_map = {
        aid: int(getattr(st, "total_stuck_steps", 0))
        for aid, st in world_state.agents.items()
    }
    # Per-agent llm_control_rate: fraction of steps where the executed movement
    # was driven by the LLM's structured target (movement_source == "llm_target").
    # Any behavioral claim only holds on these steps; non-LLM steps fall back to
    # heuristic / nudge / random movement and should not be attributed to the LLM.
    agent_metrics: Dict[str, Dict[str, float]] = {}
    for aid in agent_ids:
        total = 0
        llm_steps = 0
        for s in step_logs:
            for p in s.per_agent or []:
                if p.get("agent_id") != aid:
                    continue
                total += 1
                if (p.get("movement_source") or "") == "llm_target":
                    llm_steps += 1
        rate = float(llm_steps) / float(total) if total > 0 else 0.0
        agent_metrics[aid] = {"llm_control_rate": round(rate, 4)}
    return {
        "map_template": episode_config.environment.map_template.value,
        "villain_hero_sight_radius": float(episode_config.environment.villain_hero_sight_radius),
        "llm_driven_step_fraction": _llm_driven_step_fraction(step_logs, agent_ids),
        "total_stuck_steps_per_agent": stuck_map,
        "agent_metrics": agent_metrics,
    }


def _first_contact_steps_from_logs(step_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    v1: int | None = None
    v2: int | None = None
    for s in step_logs:
        si = s.get("step_index")
        for p in s.get("per_agent") or []:
            if p.get("agent_id") == "villain_1" and p.get("hero_truly_visible") is True and v1 is None:
                v1 = int(si) if si is not None else 0
            if p.get("agent_id") == "villain_2" and p.get("hero_truly_visible") is True and v2 is None:
                v2 = int(si) if si is not None else 0
    any_list = [x for x in (v1, v2) if x is not None]
    any_c = min(any_list) if any_list else None
    return {
        "first_contact_step_v1": v1,
        "first_contact_step_v2": v2,
        "first_contact_step_any": any_c,
    }


def _spawn_xy_from_per_agent_row(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Pre-step / spawn position: end-of-step position minus displacement that step."""
    ap = row.get("actual_position")
    am = row.get("actual_movement")
    if not isinstance(ap, (list, tuple)) or len(ap) < 2:
        return None
    if not isinstance(am, (list, tuple)) or len(am) < 2:
        return None
    return (float(ap[0]) - float(am[0]), float(ap[1]) - float(am[1]))


def _initial_villain_hero_dists(step_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Distances at **spawn** (t=0), not after the first policy step, so baselines
    with different villain policies stay comparable for the same seed/map.
    """
    if not step_logs:
        return {"villain_1_initial_dist": None, "villain_2_initial_dist": None}
    s0 = step_logs[0]
    out: Dict[str, Any] = {"villain_1_initial_dist": None, "villain_2_initial_dist": None}

    hero_spawn: Optional[Tuple[float, float]] = None
    villain_spawn: Dict[str, Tuple[float, float]] = {}
    for p in s0.get("per_agent") or []:
        aid = p.get("agent_id")
        xy = _spawn_xy_from_per_agent_row(p)
        if xy is None:
            continue
        if aid == "hero_1":
            hero_spawn = xy
        elif aid in ("villain_1", "villain_2"):
            villain_spawn[aid] = xy

    if hero_spawn is None:
        hp = s0.get("hero_position")
        if isinstance(hp, (list, tuple)) and len(hp) >= 2:
            hero_spawn = (float(hp[0]), float(hp[1]))
    if hero_spawn is None:
        return out

    hx, hy = hero_spawn[0], hero_spawn[1]
    if not villain_spawn:
        vp = s0.get("villain_positions") or {}
        for vid, key in (("villain_1", "villain_1_initial_dist"), ("villain_2", "villain_2_initial_dist")):
            if vid in vp:
                vx, vy = float(vp[vid][0]), float(vp[vid][1])
                out[key] = float(math.hypot(vx - hx, vy - hy))
        return out

    for vid, key in (("villain_1", "villain_1_initial_dist"), ("villain_2", "villain_2_initial_dist")):
        sp = villain_spawn.get(vid)
        if sp is not None:
            out[key] = float(math.hypot(sp[0] - hx, sp[1] - hy))
    return out


def _fallback_counts_per_villain(step_logs: List[Dict[str, Any]]) -> Dict[str, int]:
    c1 = 0
    c2 = 0
    for s in step_logs:
        for p in s.get("per_agent") or []:
            if not p.get("used_fallback"):
                continue
            aid = p.get("agent_id")
            if aid == "villain_1":
                c1 += 1
            elif aid == "villain_2":
                c2 += 1
    return {"v1_used_fallback_count": c1, "v2_used_fallback_count": c2}


def _hero_oscillation_escape_stats(step_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    triggered = False
    first_step: int | None = None
    for s in step_logs:
        si = s.get("step_index")
        for p in s.get("per_agent") or []:
            if p.get("agent_id") != "hero_1":
                continue
            md = p.get("movement_debug") or {}
            src = (p.get("movement_source") or "").lower()
            if (
                md.get("oscillation_escape_triggered")
                or src == "oscillation_escape"
                or src == "stuck_recovery_nudge"
            ):
                triggered = True
                if first_step is None and si is not None:
                    first_step = int(si)
                break
    return {
        "hero_oscillation_escape_triggered": triggered,
        "hero_oscillation_escape_step": first_step,
    }


def _episode_summary_payload(
    episode_config: EpisodeConfig,
    outcome: EpisodeOutcome,
    step_logs: List[StepLogEntry],
    *,
    summary_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = [asdict(s) for s in step_logs]
    fc = _first_contact_steps_from_logs(raw)
    dists = _initial_villain_hero_dists(raw)
    fb = _fallback_counts_per_villain(raw)
    osc = _hero_oscillation_escape_stats(raw)
    env = episode_config.environment
    prompt_version = None
    for c in episode_config.agent_configs:
        prompt_version = getattr(c, "prompt_version", None)
        break
    wed = outcome.within_episode_divergence or {}

    payload: Dict[str, Any] = {
        "episode_id": episode_config.episode_id,
        "outcome": outcome.result,
        "steps": outcome.steps,
        "capture_step": outcome.capture_step_index,
        "winner_team": outcome.winner_team,
        "prompt_version": prompt_version,
        "map_template": env.map_template.value,
        "spawn_mode": env.spawn_mode,
        "seed": env.seed,
        "num_villains": env.num_villains,
        "asymmetric_close_distance": env.asymmetric_close_distance,
        "asymmetric_far_distance": env.asymmetric_far_distance,
        "villain_hero_sight_radius": float(env.villain_hero_sight_radius),
        "observation_noise_std": float(env.observation_noise_std),
        "regime": env.regime_name,
        "divergence_trend": wed.get("divergence_trend"),
        "peak_divergence_step": wed.get("peak_divergence_step"),
        **fc,
        **dists,
        **fb,
        **osc,
    }
    if summary_extra:
        payload.update(summary_extra)
    return payload


def run_episode(
    episode_config: EpisodeConfig,
    llm_clients: Dict[str, LLMClient],
    *,
    log_dir: Optional[Path] = None,
    summary_extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> EpisodeOutcome:
    renderer = kwargs.get("renderer")
    stream_logs = kwargs.get("stream_logs", False)
    _ = stream_logs
    # Mirror EpisodeConfig.capture_radius onto the EnvironmentConfig so the
    # hero system prompt (which only sees env_cfg) can quote it dynamically.
    try:
        episode_config.environment.capture_radius = float(episode_config.capture_radius)
    except Exception:
        pass

    engine = SimulationEngine(
        episode_config.environment,
        episode_config.agent_configs,
    )
    world_state = engine.reset()

    env_cfg = episode_config.environment

    prebuilt = kwargs.get("agents")
    if prebuilt is not None:
        agents = prebuilt
        for agent in agents.values():
            reset = getattr(agent, "reset_episode", None)
            if callable(reset):
                reset()
    else:
        agents = {
            cfg.id: create_agent(
                cfg,
                llm_clients,
                default_client_name=cfg.model_backend,
                timeout_seconds=episode_config.llm_timeout_seconds,
                max_retries=episode_config.llm_max_retries,
                history_limit=episode_config.history_limit,
                environment_config=env_cfg,
            )
            for cfg in episode_config.agent_configs
        }

    step_logs: List[StepLogEntry] = []
    outcome: Optional[EpisodeOutcome] = None
    max_steps = episode_config.environment.max_steps
    pending_feedback: Dict[str, Optional[str]] = {}

    try:
        while True:
            observations = engine.get_observations()
            for aid, obs in observations.items():
                fb = pending_feedback.get(aid)
                if fb:
                    obs.last_action_feedback = fb
            actions = _parallel_agent_step(agents, observations)
            world_state = engine.step(actions)

            outcome = _check_capture_or_done(
                world_state,
                episode_config.capture_radius,
                max_steps,
                episode_config.episode_id,
            )

            step_entry = _build_step_log(
                episode_config.episode_id,
                world_state,
                agents,
                actions,
                observations,
                engine,
            )
            step_logs.append(step_entry)
            # Build feedback for next step from this step's per-agent rows.
            next_feedback: Dict[str, Optional[str]] = {}
            for row in step_entry.per_agent or []:
                aid = row.get("agent_id")
                if not aid:
                    continue
                in_obs = bool(row.get("target_in_obstacle")) if row.get("target_in_obstacle") is not None else False
                oob = bool(row.get("target_out_of_bounds")) if row.get("target_out_of_bounds") is not None else False
                if in_obs or oob:
                    next_feedback[aid] = (
                        "WARNING: your previous target was invalid "
                        "(in obstacle / out of bounds). Choose a different target."
                    )
                else:
                    next_feedback[aid] = None
            pending_feedback = next_feedback

            if renderer is not None:
                if not renderer.handle_events():
                    outcome = EpisodeOutcome(
                        episode_id=episode_config.episode_id,
                        result="error",
                        steps=world_state.step_index,
                        capture_step_index=None,
                        winner_team=None,
                        error_message="pygame_window_closed",
                    )
                    break
                renderer.render(world_state)

            if outcome is not None:
                break
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

    assert outcome is not None
    wed = within_episode_role_divergence([asdict(s) for s in step_logs])
    outcome = EpisodeOutcome(
        episode_id=outcome.episode_id,
        result=outcome.result,
        steps=outcome.steps,
        capture_step_index=outcome.capture_step_index,
        winner_team=outcome.winner_team,
        error_message=outcome.error_message,
        within_episode_divergence=wed,
    )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_extras = _episode_summary_extras(episode_config, step_logs, world_state)
        (log_dir / f"{episode_config.episode_id}_config.json").write_text(
            json_dumps(
                {
                    "episode": _episode_config_jsonable(episode_config),
                    "environment": episode_config.environment.model_dump(mode="json"),
                    "agents": [cfg.model_dump(mode="json") for cfg in episode_config.agent_configs],
                    **summary_extras,
                }
            ),
            encoding="utf-8",
        )
        (log_dir / f"{episode_config.episode_id}_steps.jsonl").write_text(
            "\n".join(json_dumps(asdict(s)) for s in step_logs),
            encoding="utf-8",
        )
        (log_dir / f"{episode_config.episode_id}_summary.json").write_text(
            json.dumps(
                _episode_summary_payload(
                    episode_config, outcome, step_logs, summary_extra=summary_extra
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return outcome


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
