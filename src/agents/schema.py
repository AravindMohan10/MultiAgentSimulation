"""
LLM action output schema and validation.

This module defines the exact shape of JSON that LLMs must return, and
converts validated output into the simulation's Action and Message types.
Intent is a fixed vocabulary for consistent logging and analysis.

Research (Change 1): prefer target_position [x, y] world coords + intent;
legacy movement [dx, dy, dz] remains supported for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from ..core.models import Action

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Fixed vocabulary for intent: used for logging/analysis only, not for physics.
AllowedIntent = Literal[
    "pursue_target",
    "pursue_memory",
    "flee_threat",
    "hold_position",
    "explore_area",
    "search_systematic",
    "regroup",
    "signal_teammates",
    "cut_off",
    "hold_chokepoint",
    "bait",
    "unknown",
    "invalid_output",
    "timeout",
    "unexpected_error",
]

ALLOWED_INTENT_VALUES: tuple[str, ...] = (
    "pursue_target",
    "pursue_memory",
    "flee_threat",
    "hold_position",
    "explore_area",
    "search_systematic",
    "regroup",
    "signal_teammates",
    "cut_off",
    "hold_chokepoint",
    "bait",
    "unknown",
    "invalid_output",
    "timeout",
    "unexpected_error",
)


class LLMMessagePayload(BaseModel):
    """Message payload as returned by the LLM; converted to Message for the sim."""

    payload: List[float] = Field(default_factory=list, max_length=16)
    channel: Optional[str] = None
    recipients: Optional[List[str]] = None

    @field_validator("payload", mode="before")
    @classmethod
    def coerce_payload(cls, v: object) -> List[float]:
        if not isinstance(v, list):
            return []
        out: List[float] = []
        for i, x in enumerate(v):
            if i >= 16:
                break
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                out.append(0.0)
        return out


def normalize_intent_string(v: object) -> str:
    """
    Normalize free-form intent to ALLOWED_INTENT_VALUES.
    Use this anywhere intent is assigned so logs stay schema-consistent.
    """
    if not isinstance(v, str) or not v.strip():
        return "unknown"
    s = v.strip().lower()
    if "move" in s and "away" in s:
        return "flee_threat"
    if s in ("move_away", "retreat", "withdraw"):
        return "flee_threat"
    if s in ("search", "scan", "scout"):
        return "explore_area"
    if s in ("search_around", "search_area", "patrol"):
        return "explore_area"
    if s in ("search_around_last_seen", "search_last_seen", "last_seen_search") or "search_around_last_seen" in s:
        return "pursue_memory"
    # Hero / free-form phrases models often emit (must be before generic checks)
    if "move" in s and "away" in s:
        return "flee_threat"
    if s in ("move_away", "retreat", "withdraw"):
        return "flee_threat"
    if s in ("search", "scan", "scout"):
        return "explore_area"
    if s in ("search_around", "search_area", "patrol"):
        return "explore_area"
    if "search_around_last_seen" in s or s in ("search_around_last_seen", "search_last_seen", "last_seen_search"):
        return "pursue_memory"
    if s.startswith(("evade", "escape")):
        return "flee_threat"
    if s.startswith(("intercept", "cut_off", "cutoff", "pursue")) and "memory" not in s:
        return "pursue_target"
    if s.startswith(("explore",)):
        return "explore_area"
    if "memory" in s or s in ("pursue_memory", "chase_memory", "last_known"):
        return "pursue_memory"
    if s.startswith(("search", "systematic")) or s in ("search_systematic", "systematic_search"):
        return "search_systematic"
    if s in ("hold_chokepoint", "chokepoint", "hold_choke"):
        return "hold_chokepoint"
    if s in ("bait", "decoy"):
        return "bait"

    if s in ("evade", "evade_villain", "evade_villains"):
        return "flee_threat"
    if s in ("pursue", "pursue_target"):
        return "pursue_target"
    if s in ("pursue_hero", "pursue_the_hero", "pursuehero"):
        return "pursue_target"
    if s in ("intercept", "cut_off", "cutoff"):
        return "cut_off"
    if s in ("cut_off_escape", "cutoff_escape", "cut_off_path", "cutoff_path"):
        return "cut_off"
    if s in ("intercept_escape_path", "intercept_escape_paths"):
        return "cut_off"
    if s in ("intercept_hero", "intercept_heroes", "intercept_hero_path"):
        return "pursue_target"
    if s in ("explore", "explore_area"):
        return "explore_area"
    if s in ALLOWED_INTENT_VALUES:
        return s

    # Rule-based baseline / runner aliases → canonical vocabulary
    if s in ("flee_visible_villain", "evade_visible", "flee_villain"):
        return "flee_threat"
    if s in ("go_world_center", "center", "move_center"):
        return "explore_area"
    if s in ("pursue_visible_hero", "pursue_hero"):
        return "pursue_target"
    if s in ("go_last_seen_hero", "last_known_pursuit"):
        return "pursue_memory"
    if s in ("spiral_search", "spiral"):
        return "search_systematic"
    if s == "unexpected_error":
        return "unexpected_error"
    return "unknown"


class LLMActionOutput(BaseModel):
    """
    LLM response: high-level intent plus optional world target or legacy movement.

    Preferred: target_position [x, y], intent, confidence, target_description.
    Legacy: movement [dx, dy, dz] when target_position is absent.
    """

    model_config = ConfigDict(extra="ignore")

    intent: str = "unknown"
    target_position: Optional[List[float]] = Field(
        default=None,
        description="World coordinates [x, y] the agent wants to move toward.",
    )
    target_description: Optional[str] = Field(
        default=None,
        description='Optional label, e.g. "last known hero position".',
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    movement: Optional[List[float]] = Field(
        default=None,
        max_length=3,
        description="Deprecated: raw [dx, dy, dz]; used when target_position is not set.",
    )
    message: Optional[LLMMessagePayload] = None

    @field_validator("message", mode="before")
    @classmethod
    def coerce_message(cls, v: object) -> object:
        """
        Models often emit ``message`` as a raw list (coord protocol) instead of
        ``{"payload": [...], ...}``. Accept both.
        """
        if v is None:
            return None
        if isinstance(v, LLMMessagePayload):
            return v
        if isinstance(v, dict):
            return v
        if isinstance(v, list):
            return {"payload": v, "channel": "coord"}
        return v

    @field_validator("target_position", mode="before")
    @classmethod
    def coerce_target_position(cls, v: object) -> Optional[List[float]]:
        if v is None:
            return None
        if not isinstance(v, list) or len(v) < 2:
            return None
        try:
            return [float(v[0]), float(v[1])]
        except (TypeError, ValueError, IndexError):
            return None

    @field_validator("movement", mode="before")
    @classmethod
    def movement_optional(cls, v: object) -> Optional[List[float]]:
        if v is None:
            return None
        if not isinstance(v, list):
            return None
        out: List[float] = []
        for i in range(3):
            try:
                out.append(float(v[i]) if i < len(v) else 0.0)
            except (TypeError, ValueError, IndexError):
                out.append(0.0)
        return out

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, v: object) -> str:
        """Map free-form intent strings to the schema vocabulary; safe for post-processing overrides."""
        return normalize_intent_string(v)

    @model_validator(mode="after")
    def ensure_movement_or_target(self) -> "LLMActionOutput":
        """Allow neither only if both will be handled as degenerate downstream."""
        tp = self.target_position
        has_tp = tp is not None and len(tp) >= 2
        if not has_tp and self.movement is None:
            object.__setattr__(self, "movement", [0.0, 0.0, 0.0])
        return self


def llm_action_to_action(agent_id: str, raw: LLMActionOutput) -> Action:
    """
    Convert validated LLM output into the simulation's Action (and optional Message).

    Sets movement_source:
    - llm_target when target_position is present (movement may be zero until executor)
    - llm_vector_legacy when only legacy movement is used
    """
    from ..core.models import Action, Message, Vec3

    tp = raw.target_position
    has_tp = tp is not None and len(tp) >= 2

    message: Optional[Message] = None
    if raw.message is not None and raw.message.payload is not None:
        message = Message(
            sender_id=agent_id,
            recipient_ids=raw.message.recipients,
            payload=raw.message.payload,
            channel=raw.message.channel,
        )

    if has_tp:
        ltp: Optional[Tuple[float, float]] = (float(tp[0]), float(tp[1]))
        movement = Vec3(x=0.0, y=0.0, z=0.0)
        movement_source = "llm_target"
    else:
        ltp = None
        m = raw.movement or [0.0, 0.0, 0.0]
        movement = Vec3(
            x=m[0] if len(m) > 0 else 0.0,
            y=m[1] if len(m) > 1 else 0.0,
            z=m[2] if len(m) > 2 else 0.0,
        )
        movement_source = "llm_vector_legacy"

    return Action(
        movement=movement,
        message=message,
        intent=raw.intent,
        movement_source=movement_source,
        llm_target_position=ltp,
        target_description=raw.target_description,
        llm_confidence=float(raw.confidence),
    )
