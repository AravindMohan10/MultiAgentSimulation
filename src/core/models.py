"""
Core data models and configuration for the multi-agent pursuit–evasion simulation.

This module defines:
- Config objects for agents and the environment.
- World state representations (agents, terrain, weather, obstacles, time).
- Observation, action, and message types used between the simulation core and agents.

These models are designed to be:
- **Config-driven**: easy to construct from YAML/JSON using pydantic.
- **Renderer-agnostic**: `WorldState` can be consumed by any renderer (Pygame, Three.js, etc.).
- **Agent-agnostic**: agents interact only via `Observation` and `Action`, not raw `WorldState`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class Vec3(BaseModel):
    """3D vector for positions and velocities (z=0 for initial 2D worlds)."""

    x: float
    y: float
    z: float = 0.0


class AgentType(str, Enum):
    """Categorization of agents in the pursuit–evasion task."""

    HERO = "hero"
    VILLAIN = "villain"


class MapTemplate(str, Enum):
    """
    Procedural map topology for research experiments.

    SCATTERED is the backward-compatible control (random low-density obstacles).
    Other templates create structured coordination / role-specialization pressure.
    """

    SCATTERED = "scattered"
    HUB_AND_SPOKES = "hub_and_spokes"
    ASYMMETRIC_LABYRINTH = "asymmetric_labyrinth"
    GRADIENT = "gradient"


class AgentConfig(BaseModel):
    """
    Configuration for a single agent.

    This is intended to be created from experiment configs, and passed into
    the simulation engine when constructing agents.
    """

    id: str = Field(..., description="Unique identifier for the agent.")
    agent_type: AgentType = Field(..., description="Role of the agent (hero or villain).")
    max_speed: float = Field(
        1.0,
        ge=0.0,
        description="Maximum movement speed per physics timestep.",
    )
    vision_radius: float = Field(
        10.0,
        ge=0.0,
        description="Vision radius used by the perception engine.",
    )
    communication_enabled: bool = Field(
        True, description="Whether this agent can send/receive messages."
    )
    use_auto_coord_message: bool = Field(
        True,
        description=(
            "When True (default), villain agents receive engine-built coordination "
            "messages if the LLM sends none. When False, only LLM-emitted messages are used."
        ),
    )
    strategy_mode: str = Field(
        "heuristic",
        description="High-level strategy mode label (e.g. 'heuristic', 'llm', 'rl').",
    )
    model_backend: Optional[str] = Field(
        None,
        description=(
            "Identifier for the underlying model backend, e.g. "
            "'local-7b', 'gpt-4o', 'gemini', etc., if applicable."
        ),
    )
    prompt_version: Literal["V0_BASELINE", "V1_COMMUNICATION", "V2_GUIDED"] = Field(
        "V2_GUIDED",
        description="Prompt policy variant used for controlled experiments.",
    )
    disable_messages: bool = Field(
        False,
        description="Ablation flag: when true, this agent neither sends nor receives messages.",
    )
    disable_memory: bool = Field(
        False,
        description="Ablation flag: when true, private memory context is not provided to prompts.",
    )
    disable_guidance: bool = Field(
        False,
        description="Ablation flag: when true, role/policy guidance is minimized in the system prompt.",
    )


class EnvironmentConfig(BaseModel):
    """
    Configuration for a single simulation environment instance.

    This captures all environment-level parameters needed for reproducible runs.
    """

    world_size: Tuple[float, float] = Field(
        (160.0, 160.0),
        description="Width and height of the 2D world in world units.",
    )
    physics_dt: float = Field(
        0.1,
        gt=0.0,
        description="Physics timestep size in seconds.",
    )
    decision_dt: float = Field(
        0.5,
        gt=0.0,
        description="Decision timestep size in seconds (agent actions).",
    )
    max_steps: int = Field(
        150,
        gt=0,
        description="Maximum number of decision steps per episode.",
    )
    terrain_type: str = Field(
        "flat",
        description="High-level terrain type (e.g. 'flat', 'hilly', 'urban').",
    )
    map_template: MapTemplate = Field(
        MapTemplate.SCATTERED,
        description="Procedural map topology; obstacle_density applies only to SCATTERED.",
    )
    obstacle_density: float = Field(
        0.08,
        ge=0.0,
        le=1.0,
        description="Used only for SCATTERED and GRADIENT (left-edge baseline); ignored for structured maps.",
    )
    obstacle_radius: float = Field(
        1.5,
        ge=0.0,
        description="Default radius for procedurally generated obstacles.",
    )
    boundary_mode: Literal["hard", "wrap", "bounce"] = Field(
        "hard",
        description=(
            "How agents interact with world boundaries: "
            "hard = clamp inside [0,w]x[0,h], wrap = toroidal, bounce = reflect."
        ),
    )
    weather_type: str = Field(
        "clear",
        description="High-level weather condition (e.g. 'clear', 'rain', 'fog').",
    )
    base_visibility_radius: float = Field(
        20.0,
        ge=0.0,
        description="Baseline visibility radius before weather modifiers.",
    )
    slope_steepness: float = Field(
        0.0,
        ge=0.0,
        description="High-level control of terrain slope / steepness.",
    )
    seed: int = Field(
        0,
        description="Random seed for environment generation and reproducibility.",
    )
    visibility_radius: Optional[float] = Field(
        None,
        ge=0.0,
        description="Optional global cap on visibility radius for all agents.",
    )
    message_delay_steps: int = Field(
        0,
        ge=0,
        description="Symmetric message delivery delay in decision steps.",
    )
    message_budget_per_agent: Optional[int] = Field(
        None,
        ge=0,
        description="Optional max number of sent messages per agent per episode.",
    )
    observation_noise_std: float = Field(
        0.0,
        ge=0.0,
        description="Gaussian stddev added to observed visible-agent positions.",
    )

    # Research ablation/conditioning: give villains one-time initial knowledge
    # of each other’s starting positions (without requiring early visibility).
    inject_villain_start_positions: bool = Field(
        False,
        description="If true, inject initial villain-to-villain position messages into the first observation.",
    )
    villain_hero_sight_radius: float = Field(
        15.0,
        ge=0.0,
        description=(
            "Radius within which villains treat the hero as prompt-visible (narrow sight). "
            "Engine perception may use a larger effective vision radius; this value is the "
            "single source of truth for prompt/filter semantics and logging."
        ),
    )
    spawn_mode: str = Field(
        "asymmetric",
        description=(
            'Controls villain spawn positioning. "asymmetric" = villain_1 close (12 units), '
            'villain_2 far (60 units); "random" = existing behavior.'
        ),
    )
    asymmetric_close_distance: float = Field(
        12.0,
        ge=0.0,
        description="Asymmetric mode: villain_1 distance from hero spawn.",
    )
    asymmetric_far_distance: float = Field(
        60.0,
        ge=0.0,
        description="Asymmetric mode: villain_2 distance from hero spawn.",
    )

    villain_spawn_radius: float = Field(
        50.0,
        ge=0.0,
        description=(
            "Max distance from hero spawn where villains can appear. "
            "Used to ensure encounters occur within typical episode lengths on 160x160 worlds."
        ),
    )
    chokepoint_positions: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Auto-populated by map generators for hub/labyrinth maps; used in prompts and metrics.",
    )
    gradient_max_density: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Right-edge obstacle density for GRADIENT template.",
    )
    num_villains: int = Field(
        2,
        ge=1,
        le=2,
        description=(
            "How many villains participate (drives agent list in experiments). "
            "1 = single-villain baseline; 2 = full pursuit team. Engine uses agent_configs; "
            "this field documents the condition for manifests."
        ),
    )
    regime_name: Optional[str] = Field(
        None,
        description="Optional experiment regime label (e.g. R1, R2, R3) for summary JSON / analysis.",
    )


class AgentState(BaseModel):
    """Dynamic state of a single agent at a given simulation time."""

    id: str
    agent_type: AgentType
    position: Vec3
    velocity: Vec3 = Field(default_factory=lambda: Vec3(x=0.0, y=0.0, z=0.0))
    orientation: float = Field(
        0.0,
        description="Orientation angle in radians, e.g. heading direction in the plane.",
    )
    alive: bool = Field(
        True,
        description="Whether the agent is active in the simulation (not captured / removed).",
    )

    # Physics stability / debug instrumentation.
    # These fields are updated by PhysicsEngine each step.
    stuck_steps: int = Field(
        0,
        ge=0,
        description="Consecutive steps where the agent failed to change position.",
    )
    unstuck_steps_remaining: int = Field(
        0,
        ge=0,
        description="Legacy field; exploration injection is disabled (kept for schema compatibility).",
    )
    stuck_this_step: bool = Field(
        False,
        description="True if the agent did not move this decision step (physics outcome).",
    )
    total_stuck_steps: int = Field(
        0,
        ge=0,
        description="Cumulative count of stuck steps in the current episode.",
    )
    last_movement_debug: Dict[str, Any] = Field(
        default_factory=lambda: {
            "blocked_movement": False,
            "unstuck_triggered": False,
            "hit_boundary": False,
            "boundary_adjustment_applied": False,
            "adjusted_movement": [0.0, 0.0],
            "movement_source": "llm_target",
            "actual_movement": [0.0, 0.0, 0.0],
            # Geometry/LLM debugging (populated by LLMAgent, merged by PhysicsEngine).
            "direction_to_hero": [0.0, 0.0, 0.0],
            "llm_vector": [0.0, 0.0, 0.0],
            "final_movement": [0.0, 0.0, 0.0],
            "stuck_fix_triggered": False,
        },
        description="Per-step debug info produced by the physics movement resolver.",
    )


class TerrainInfo(BaseModel):
    """
    Coarse representation of terrain parameters.

    This can be extended later to include heightmaps or grids. For now, it
    captures global descriptors and can be augmented by local samples.
    """

    terrain_type: str
    slope_steepness: float = 0.0


class WeatherInfo(BaseModel):
    """High-level weather descriptor affecting visibility and movement."""

    weather_type: str
    visibility_modifier: float = Field(
        1.0,
        ge=0.0,
        description="Multiplier applied to base visibility radius (e.g. <1 for fog).",
    )


class Obstacle(BaseModel):
    """Simple obstacle representation located in the world."""

    position: Vec3
    radius: float = Field(
        1.0,
        ge=0.0,
        description="Effective radius used for collision / line-of-sight checks.",
    )


class WorldState(BaseModel):
    """
    Complete simulation world state at a single time.

    This is owned and mutated exclusively by the SimulationEngine. Agents never
    see or modify this object directly; they only receive filtered Observations.
    """

    time: float = 0.0
    step_index: int = 0
    agents: Dict[str, AgentState] = Field(
        default_factory=dict,
        description="Mapping from agent id to current agent state.",
    )
    terrain: TerrainInfo = Field(
        default_factory=lambda: TerrainInfo(terrain_type="flat", slope_steepness=0.0)
    )
    weather: WeatherInfo = Field(
        default_factory=lambda: WeatherInfo(weather_type="clear", visibility_modifier=1.0)
    )
    obstacles: List[Obstacle] = Field(default_factory=list)


class Message(BaseModel):
    """
    Structured communication payload emitted by an agent.

    The payload is intentionally numeric / structured to support analysis of
    emergent communication without relying on free-form text.
    """

    sender_id: str
    # None or empty list can be interpreted by the communication router as
    # broadcast or team-based routing depending on config.
    recipient_ids: Optional[List[str]] = None
    payload: List[float] = Field(
        default_factory=list,
        description="Numeric payload interpreted according to experiment protocol.",
    )
    channel: Optional[str] = Field(
        None,
        description="Optional symbolic channel/tag to categorize the message.",
    )


class Observation(BaseModel):
    """
    Filtered snapshot of the world available to a single agent at decision time.

    Constructed by the PerceptionEngine from a WorldState snapshot. Agents
    should base their decisions solely on this structure, not on WorldState.
    """

    self_state: AgentState
    visible_agents: List[AgentState] = Field(
        default_factory=list,
        description="Other agents within vision / communication range.",
    )
    local_terrain: TerrainInfo
    local_weather: WeatherInfo
    incoming_messages: List[Message] = Field(default_factory=list)
    time: float
    step_index: int
    villain_hero_sight_radius: float = Field(
        15.0,
        ge=0.0,
        description=(
            "Copied from EnvironmentConfig: villains use this radius to decide if the hero "
            "counts as visible for prompts and policy (narrow sight vs engine visibility)."
        ),
    )
    world_obstacles: List[Obstacle] = Field(
        default_factory=list,
        description="Full world obstacle set for map-aware prompts (same for all agents).",
    )
    map_template: str = Field(
        "scattered",
        description="Map template id (EnvironmentConfig.map_template).",
    )
    chokepoint_positions: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Strategic chokepoints from map generator (if any).",
    )


class Action(BaseModel):
    """
    Structured action emitted by an agent for a single decision timestep.

    The SimulationEngine (via a PhysicsEngine) is responsible for validating and
    applying these actions to advance the WorldState.
    """

    movement: Vec3 = Field(
        default_factory=lambda: Vec3(x=0.0, y=0.0, z=0.0),
        description="Desired movement vector in world units per decision step.",
    )
    message: Optional[Message] = Field(
        None,
        description="Optional message to emit via the communication router.",
    )
    intent: Optional[str] = Field(
        None,
        description=(
            "Optional high-level intent label, useful for logging / analysis "
            "(e.g. 'pursue_hero', 'flee', 'regroup')."
        ),
    )

    # Research traceability: how movement was produced (filter non-LLM steps in analysis).
    movement_source: str = Field(
        default="llm_target",
        description=(
            "One of: llm_target, llm_vector_legacy, fallback_explore, fallback_last_valid, "
            "stuck_recovery, stuck_halted, boundary_override, rule_based, "
            "oscillation_escape, structured_search_fallback."
        ),
    )
    llm_target_position: Optional[Tuple[float, float]] = Field(
        default=None,
        description="World [x, y] target from the LLM when using target-based policy.",
    )
    target_description: Optional[str] = Field(
        default=None,
        description="Optional LLM label for the chosen target.",
    )
    llm_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="LLM-reported confidence in its target/intent (0..1).",
    )

    movement_debug: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional per-step movement debug payload (kept structured for analysis).",
    )
