"""
Per-agent runtime session and private memory.

This module exists to prevent information leakage between agents.

Each agent gets its own AgentSession instance, and that session owns:
- prompt history
- raw LLM responses
- validated actions
- memory summary
- any other private per-agent runtime state

Nothing in here should ever be shared across agents.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.models import Action, AgentConfig, Observation


@dataclass(slots=True)
class AgentTurnRecord:
    """
    One private step of an agent's runtime history.

    This is intentionally agent-local. The runner or agent wrapper can store the
    original observation, prompt, response, and validated action for debugging,
    replay, and memory summarization.
    """

    step_index: int
    time: float
    observation: Optional[Observation] = None
    prompt: str = ""
    raw_response: Optional[str] = None
    action: Optional[Action] = None
    valid: bool = True
    error: Optional[str] = None
    # Raw intent string from the LLM JSON before schema normalization (research traceability).
    llm_raw_intent: Optional[str] = None


@dataclass(slots=True)
class AgentSession:
    """
    Per-agent isolated runtime state.

    This object should never be shared across agents. It is the boundary that
    keeps prompt history and memory private for each LLM agent.
    """

    agent_id: str
    config: AgentConfig
    history_limit: int = 8
    memory_summary: str = ""
    turn_history: List[AgentTurnRecord] = field(default_factory=list)
    last_prompt: Optional[str] = None
    last_raw_response: Optional[str] = None
    last_valid_action: Optional[Action] = None
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        # Keep the history window usable even if a caller passes 0 or a negative.
        self.history_limit = max(1, int(self.history_limit))

    @property
    def role(self) -> str:
        """Convenience role label derived from the agent config."""
        return self.config.agent_type.value

    def reset(self) -> None:
        """
        Clear all private runtime state.

        Use this at the start of a new episode so no history leaks across runs.
        """
        self.memory_summary = ""
        self.turn_history.clear()
        self.last_prompt = None
        self.last_raw_response = None
        self.last_valid_action = None
        self.last_error = None

    def set_memory_summary(self, summary: str) -> None:
        """
        Update the compact agent-private memory summary.

        This is where a future summarizer can store a compressed summary of the
        agent's own past interactions without exposing other agents' context.
        """
        self.memory_summary = summary.strip()

    def record_turn(
        self,
        *,
        step_index: int,
        time: float,
        observation: Optional[Observation] = None,
        prompt: str = "",
        raw_response: Optional[str] = None,
        action: Optional[Action] = None,
        valid: bool = True,
        error: Optional[str] = None,
        llm_raw_intent: Optional[str] = None,
    ) -> AgentTurnRecord:
        """
        Store one private step of the agent runtime.

        Deep copies are used for Observation and Action so later mutation in the
        simulation or validation pipeline cannot affect historical records.
        """
        record = AgentTurnRecord(
            step_index=step_index,
            time=time,
            observation=deepcopy(observation) if observation is not None else None,
            prompt=prompt,
            raw_response=raw_response,
            action=deepcopy(action) if action is not None else None,
            valid=valid,
            error=error,
            llm_raw_intent=llm_raw_intent,
        )
        self.turn_history.append(record)
        self.last_prompt = prompt
        self.last_raw_response = raw_response
        self.last_valid_action = deepcopy(action) if action is not None else None
        self.last_error = error
        self._trim_history()
        return record

    def recent_turns(self, limit: Optional[int] = None) -> List[AgentTurnRecord]:
        """Return the most recent private turns for this agent only."""
        if limit is None:
            limit = self.history_limit
        limit = max(1, int(limit))
        return self.turn_history[-limit:]

    def context_snapshot(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Serializable snapshot of agent-private context.

        This is intended for prompt building later, so the prompt builder can
        choose to include a compact summary of the agent's own history.
        """
        turns = []
        for turn in self.recent_turns(limit):
            turns.append(
                {
                    "step_index": turn.step_index,
                    "time": turn.time,
                    "prompt": turn.prompt,
                    "raw_response": turn.raw_response,
                    "valid": turn.valid,
                    "error": turn.error,
                    "action": turn.action.model_dump() if turn.action is not None else None,
                }
            )

        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "memory_summary": self.memory_summary,
            "history_limit": self.history_limit,
            "last_error": self.last_error,
            "last_valid_action": (
                self.last_valid_action.model_dump()
                if self.last_valid_action is not None
                else None
            ),
            "turn_history": turns,
        }

    def _trim_history(self) -> None:
        """Keep only the newest history window."""
        if len(self.turn_history) <= self.history_limit:
            return
        del self.turn_history[: len(self.turn_history) - self.history_limit]

