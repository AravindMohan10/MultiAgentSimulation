"""
Bridge BAML-generated types to LLMActionOutput / Action pipeline.

Used when LLM_OUTPUT_PARSER=baml. Prompts still come from prompts.py;
only parsing + LLM HTTP is delegated to BAML.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

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


def extract_baml_raw_output(exc: BaseException) -> Optional[str]:
    """
    Recover the model's raw text from a BAML validation error.

    Self-steer prompts ask for snake_case intents (``pursue_memory``) while the
    BAML enum expects PascalCase (``PursueMemory``). The LLM often follows the
    prompt; this helper lets us fall back to pydantic parsing on that text.
    """
    raw = getattr(exc, "raw_output", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    text = str(exc)
    for marker in ("Raw Response:", "raw_output="):
        idx = text.find(marker)
        if idx < 0:
            continue
        chunk = text[idx + len(marker) :].strip()
        if chunk.startswith("```"):
            end_fence = chunk.find("```", 3)
            if end_fence > 0:
                inner = chunk[3:end_fence].strip()
                if inner.lower().startswith("json"):
                    inner = inner[4:].strip()
                if inner:
                    return inner
        if chunk.startswith("{") or chunk.startswith("["):
            return chunk.split(", prompt=")[0].strip().rstrip("`")

    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "{" in text:
        from .llm_agent import _extract_json_candidate

        try:
            return _extract_json_candidate(text)
        except (ValueError, TypeError):
            pass
    return None


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


def invoke_baml_choose_action(
    system_prompt: str,
    user_prompt: str,
    *,
    pydantic_fallback: bool = True,
) -> Tuple[LLMActionOutput, str]:
    """
    Call BAML ChooseAgentAction; return (LLMActionOutput, raw_response for logs).

    When ``pydantic_fallback`` is True (default), BAML enum mismatches (e.g.
    snake_case intent from self-steer prompts) recover via pydantic parsing of
    the model's raw JSON instead of failing the step.
    """
    from baml_client.baml_client.sync_client import b

    try:
        result = b.ChooseAgentAction(system_prompt, user_prompt)
    except Exception as exc:
        if not pydantic_fallback:
            raise
        raw_text = extract_baml_raw_output(exc)
        if not raw_text:
            raise
        from .llm_agent import _parse_llm_output_with_raw

        parsed, _ = _parse_llm_output_with_raw(raw_text)
        return parsed, raw_text.strip()

    out = baml_action_to_llm_output(result)
    try:
        raw_json = result.model_dump_json() if hasattr(result, "model_dump_json") else str(result)
    except Exception:
        raw_json = str(result)
    return out, raw_json
