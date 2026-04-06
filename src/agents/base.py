"""
Base agent interface for the pursuit-evasion simulation.

This file defines the smallest possible contract that the runner needs:

- every agent must accept an Observation
- every agent must return an Action

Keeping this contract tiny lets us swap implementations freely:
LLM agents, heuristic agents, mocks for tests, or future policies all work as
long as they implement the same step(observation) -> Action interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.models import Action, AgentConfig, Observation


@runtime_checkable
class Agent(Protocol):
    """
    Minimal runtime contract for any agent implementation.

    The simulation never gives agents direct access to WorldState.
    Agents only receive filtered Observation objects and must respond with a
    structured Action.
    """

    id: str
    config: AgentConfig

    def step(self, observation: Observation) -> Action:
        """Produce one action for the current decision timestep."""
        ...


class BaseAgent:
    """
    Optional convenience base class.

    Concrete agents can inherit this to get shared config/id storage without
    repeating boilerplate. Subclasses still must implement step().
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.id = config.id

    def step(self, observation: Observation) -> Action:
        raise NotImplementedError("Subclasses must implement step(observation).")

