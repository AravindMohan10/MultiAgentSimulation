from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import math
from typing import Any, Dict, List, Literal, Optional

from ..core.engine import SimulationEngine
from ..core.models import Action, AgentConfig, AgentType, EnvironmentConfig, Observation, Vec3, WorldState
from ..agents.base import Agent
from ..agents.baseline_agent import RuleBasedAgent
from ..agents.factory import create_agent, LLMClient
from ..agents.llm_agent import LLMAgent
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
            "stuck_this_step": bool(getattr(st, "stuck_this_step", False)),
            "boundary_hit": boundary_hit,
            "obstacle_collision": obstacle_collision,
            "messages_sent": int(msgs_out),
            "messages_received": int(msgs_in),
            "message_budget_remaining": budget_left,
        }

        if isinstance(agent, LLMAgent):
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
                        "llm_confidence": float(getattr(action, "llm_confidence", 1.0)),
                        "movement_source": getattr(action, "movement_source", None),
                        "movement": [action.movement.x, action.movement.y, action.movement.z],
                        "movement_debug": getattr(action, "movement_debug", None) or {},
                        "used_fallback": not last_turn.valid,
                        "fallback_reason": last_turn.error,
                    }
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
                        "llm_confidence": float(getattr(la, "llm_confidence", 1.0)),
                        "movement_source": la.movement_source,
                        "movement": [la.movement.x, la.movement.y, la.movement.z],
                        "used_fallback": False,
                        "fallback_reason": None,
                    }
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
    return {
        "map_template": episode_config.environment.map_template.value,
        "villain_hero_sight_radius": float(episode_config.environment.villain_hero_sight_radius),
        "llm_driven_step_fraction": _llm_driven_step_fraction(step_logs, agent_ids),
        "total_stuck_steps_per_agent": stuck_map,
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


def _initial_villain_hero_dists(step_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not step_logs:
        return {"villain_1_initial_dist": None, "villain_2_initial_dist": None}
    s0 = step_logs[0]
    hp = s0.get("hero_position")
    vp = s0.get("villain_positions") or {}
    out: Dict[str, Any] = {"villain_1_initial_dist": None, "villain_2_initial_dist": None}
    if not hp or len(hp) < 2:
        return out
    hx, hy = float(hp[0]), float(hp[1])
    for vid, key in (("villain_1", "villain_1_initial_dist"), ("villain_2", "villain_2_initial_dist")):
        if vid in vp:
            vx, vy = float(vp[vid][0]), float(vp[vid][1])
            out[key] = float(math.hypot(vx - hx, vy - hy))
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
    return {
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


def run_episode(
    episode_config: EpisodeConfig,
    llm_clients: Dict[str, LLMClient],
    *,
    log_dir: Optional[Path] = None,
    **kwargs: Any,
) -> EpisodeOutcome:
    renderer = kwargs.get("renderer")
    stream_logs = kwargs.get("stream_logs", False)
    _ = stream_logs
    engine = SimulationEngine(
        episode_config.environment,
        episode_config.agent_configs,
    )
    world_state = engine.reset()

    env_cfg = episode_config.environment

    agents: Dict[str, Agent] = {
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

    try:
        while True:
            observations = engine.get_observations()
            actions = _parallel_agent_step(agents, observations)
            world_state = engine.step(actions)

            outcome = _check_capture_or_done(
                world_state,
                episode_config.capture_radius,
                max_steps,
                episode_config.episode_id,
            )

            step_logs.append(
                _build_step_log(
                    episode_config.episode_id,
                    world_state,
                    agents,
                    actions,
                    observations,
                    engine,
                )
            )

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
            json.dumps(_episode_summary_payload(episode_config, outcome, step_logs), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return outcome


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
