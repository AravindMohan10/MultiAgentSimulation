"""
Bridge BAML-generated types to LLMActionOutput / Action pipeline.

Used when LLM_OUTPUT_PARSER=baml. Prompts still come from prompts.py;
only parsing + LLM HTTP is delegated to BAML.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import re

from .schema import LLMActionOutput, LLMMessagePayload, normalize_intent_string


def _pascal_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _intent_to_str(intent: Any) -> str:
    if intent is None:
        return "unknown"
    raw = str(intent.value if hasattr(intent, "value") else intent)
    if "_" in raw:
        return normalize_intent_string(raw)
    return normalize_intent_string(_pascal_to_snake(raw))


def _optional_pair(coords: Any) -> Optional[List[float]]:
    if coords is None:
        return None
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            return [float(coords[0]), float(coords[1])]
        except (TypeError, ValueError):
            return None
    return None


def _optional_movement(mv: Any) -> Optional[List[float]]:
    if mv is None:
        return None
    if not isinstance(mv, (list, tuple)):
        return None
    out: List[float] = []
    for i in range(3):
        try:
            out.append(float(mv[i]) if i < len(mv) else 0.0)
        except (TypeError, ValueError, IndexError):
            out.append(0.0)
    return out


def baml_action_to_llm_output(baml_action: Any) -> LLMActionOutput:
    """Convert generated AgentAction to pydantic LLMActionOutput."""
    msg_out: Optional[LLMMessagePayload] = None
    raw_msg = getattr(baml_action, "message", None)
    if raw_msg is not None:
        payload = list(getattr(raw_msg, "payload", None) or [])
        channel = getattr(raw_msg, "channel", None)
        msg_out = LLMMessagePayload(payload=payload, channel=channel, recipients=None)

    return LLMActionOutput(
        intent=_intent_to_str(getattr(baml_action, "intent", None)),
        target_position=_optional_pair(getattr(baml_action, "target_position", None)),
        target_description=getattr(baml_action, "target_description", None),
        confidence=float(getattr(baml_action, "confidence", 1.0) or 1.0),
        movement=_optional_movement(getattr(baml_action, "movement", None)),
        message=msg_out,
    )


def invoke_baml_choose_action(system_prompt: str, user_prompt: str) -> Tuple[LLMActionOutput, str]:
    """
    Call BAML ChooseAgentAction; return (LLMActionOutput, synthetic raw_response for logs).
    """
    from baml_client.baml_client.sync_client import b

    result = b.ChooseAgentAction(system_prompt, user_prompt)
    out = baml_action_to_llm_output(result)
    raw_intent = _intent_to_str(getattr(result, "intent", None))
    # Serialize parsed object for raw_llm_response field (audit trail).
    try:
        raw_json = result.model_dump_json() if hasattr(result, "model_dump_json") else str(result)
    except Exception:
        raw_json = str(result)
    return out, raw_json
