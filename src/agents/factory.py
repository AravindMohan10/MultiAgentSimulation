"""
Agent factory: build concrete agent instances from AgentConfig.

This module is the single place that knows how to map:
- strategy_mode
- model_backend
- agent_type (hero / villain)

onto a concrete Agent implementation.
"""

from __future__ import annotations

from typing import Dict

from ..core.models import AgentConfig, EnvironmentConfig
from .base import Agent
from .baseline_agent import RuleBasedAgent
from .llm_agent import LLMClient, LLMAgent


def create_agent(
    config: AgentConfig,
    llm_clients: Dict[str, LLMClient],
    *,
    default_client_name: str | None = None,
    timeout_seconds: float = 20.0,
    max_retries: int = 2,
    history_limit: int = 8,
    environment_config: EnvironmentConfig | None = None,
) -> Agent:
    """
    Create a concrete Agent instance from an AgentConfig.

    Parameters
    ----------
    config:
        Configuration for this agent (id, type, strategy_mode, model_backend, etc.).
    llm_clients:
        Mapping from backend name to LLMClient instance.
    default_client_name:
        Name of the default backend to use if config.model_backend is None.
    environment_config:
        Required for ``strategy_mode="rule_based"`` (world bounds, sight radius semantics).
    """
    strategy = (config.strategy_mode or "").lower()

    if strategy == "rule_based":
        if environment_config is None:
            raise ValueError(
                f"Agent {config.id!r} uses strategy_mode='rule_based' but environment_config "
                "was not provided to create_agent()."
            )
        return RuleBasedAgent(agent_config=config, env_config=environment_config)

    # LLM-backed agents (current primary use case).
    if strategy in ("", "llm"):
        backend_name = config.model_backend or default_client_name
        if not backend_name:
            raise ValueError(
                f"Agent {config.id!r} has strategy_mode={strategy!r} but no "
                "model_backend and no default_client_name was provided."
            )
        try:
            client = llm_clients[backend_name]
        except KeyError as exc:
            raise ValueError(
                f"LLM backend {backend_name!r} not found in llm_clients "
                f"for agent {config.id!r}."
            ) from exc

        return LLMAgent(
            config=config,
            client=client,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            history_limit=history_limit,
            environment_config=environment_config,
        )

    raise ValueError(
        f"Unsupported strategy_mode {config.strategy_mode!r} for agent {config.id!r}."
    )
