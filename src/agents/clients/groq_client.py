"""
Groq client for the same LLMClient contract used by LLMAgent.

Uses Groq's OpenAI-compatible chat-completions endpoint:
https://api.groq.com/openai/v1/chat/completions
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ..llm_agent import LLMClient, validate_target_position


GROQ_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions").rstrip("/")


def sanitize_llm_json(raw: str) -> str:
    """
    Evaluate arithmetic expressions embedded in JSON by the LLM.

    Handles patterns like:
      {"target_position": [148.18 + 0.70, 119.23 + 0.70]}

    Replaces simple binary +/- float expressions with a computed float.
    """

    if not isinstance(raw, str) or not raw:
        return raw

    def eval_expr(match: re.Match[str]) -> str:
        expr = match.group(0)
        try:
            # Matches are constrained to floats + operators only.
            result = eval(expr, {"__builtins__": {}})  # noqa: S307 - constrained by regex
            return str(round(float(result), 8))
        except Exception:
            return expr

    # Match: (optional sign) float, operator (+/-), (optional sign) float.
    # Supports decimals without leading zeros like `.5`.
    pattern = (
        r"-?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+\-]?\d+)?"
        r"\s*[\+\-]\s*"
        r"-?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+\-]?\d+)?"
    )
    return re.sub(pattern, eval_expr, raw)


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:2000]
    except Exception:
        return ""


def _post_chat_completion(
    url: str,
    api_key: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 60.0,
) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "MultiAgentSimulation/GroqClient (urllib)",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = _http_error_body(exc)
        msg = f"HTTP {exc.code} {exc.reason}"
        if body:
            msg = f"{msg}: {body}"
        raise RuntimeError(f"Groq request failed: {msg}") from exc
    except Exception as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc


def _extract_text(choice: Any) -> str:
    message = getattr(choice, "message", None) or choice.get("message") if isinstance(choice, dict) else None
    if not message:
        return ""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    # Some models may return the "useful" text in reasoning-related fields.
    if isinstance(message, dict):
        for k in ("reasoning", "reasoning_details"):
            v = message.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if v is not None and not isinstance(v, str):
                # Best-effort stringification for nested objects.
                sv = str(v).strip()
                if sv:
                    return sv

    return str(content or "").strip()


class GroqClient:
    """Simple HTTP client that satisfies the LLMClient protocol."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
        top_p: float | None = None,
        request_timeout_seconds: float | None = None,
        request_retries: int | None = None,
    ) -> None:
        # Groq has deprecated `llama3-70b-8192` (recommended replacement: `llama-3.3-70b-versatile`).
        self._model_name = model_name or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        # Env var overrides let experiments record exact settings in manifests.
        self._temperature = float(os.environ.get("GROQ_TEMPERATURE", str(temperature)))
        self._max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", str(max_tokens)))
        self._top_p = float(top_p if top_p is not None else os.environ.get("GROQ_TOP_P", "0.9"))
        self._timeout_seconds = float(
            request_timeout_seconds
            if request_timeout_seconds is not None
            else os.environ.get("GROQ_TIMEOUT_SECONDS", "60")
        )
        self._request_retries = int(
            request_retries if request_retries is not None else os.environ.get("GROQ_REQUEST_RETRIES", "2")
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Export it or pass api_key= to GroqClient.")

        want_debug = os.environ.get("GROQ_DEBUG_RAW_OUTPUT", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        base: Dict[str, Any] = {
            "model": self._model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            # Keep stable sampling params for research reproducibility.
            "top_p": 0.9,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        want_json_mode = os.environ.get("GROQ_JSON_MODE", "1").strip().lower() in ("1", "true", "yes", "on")
        payloads: list[Dict[str, Any]] = []
        if want_json_mode:
            payloads.append({**base, "response_format": {"type": "json_object"}})
        payloads.append(dict(base))

        last_err: Optional[RuntimeError] = None
        for payload in payloads:
            # Per-payload: try twice before moving on.
            for attempt in range(2):
                try:
                    raw = _post_chat_completion(GROQ_URL, self._api_key, payload, timeout=self._timeout_seconds)
                    obj = json.loads(raw)
                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not choices:
                        raise RuntimeError(f"Groq returned no choices: {obj}")
                    text = _extract_text(choices[0])
                    if not text:
                        raise RuntimeError(f"Groq returned empty message content: {obj}")

                    if want_debug:
                        print("RAW LLM OUTPUT:", text)

                    # Defensive parsing: trim to the first "{" and the last "}".
                    # This makes us resilient to code fences or pre/postamble text.
                    s = text.strip()
                    start = s.find("{")
                    end = s.rfind("}")
                    if start == -1 or end == -1 or end <= start:
                        raise RuntimeError(f"Groq output does not contain a JSON object: {text[:200]}")
                    candidate = s[start : end + 1]
                    try:
                        sanitized = sanitize_llm_json(candidate)
                        parsed = json.loads(sanitized)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"Groq action output is not valid JSON: {exc}") from exc

                    if not isinstance(parsed, dict):
                        raise RuntimeError("Groq action JSON is not an object.")
                    if "intent" not in parsed:
                        raise RuntimeError("Groq action JSON missing required key: intent.")
                    if "target_position" not in parsed and "movement" not in parsed:
                        raise RuntimeError(
                            "Groq action JSON must include target_position [x,y] and/or movement [dx,dy,dz]."
                        )
                    if "target_position" in parsed:
                        parsed["target_position"] = validate_target_position(parsed.get("target_position"))

                    # Return a clean, JSON-parseable candidate string for LLMAgent.
                    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
                except RuntimeError as exc:
                    last_err = exc
                    err_s = str(exc).lower()

                    # If json-mode is rejected, stop trying this payload and fall back to non-json.
                    if (
                        want_json_mode
                        and "response_format" in payload
                        and any(t in err_s for t in ("400", "422", "invalid", "not supported", "unsupported"))
                    ):
                        break

                    if attempt < 1:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    raise

        assert last_err is not None
        raise last_err


def _extract_json_candidate(text: str) -> str:
    s = (text or "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"Groq output does not contain a JSON object: {text[:200]}")
    return s[start : end + 1]


def build_default_groq_clients(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, LLMClient]:
    key = api_key or os.environ.get("GROQ_API_KEY")
    m = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    client: LLMClient = GroqClient(m, api_key=key)  # type: ignore[assignment]
    return {
        "groq": client,
    }

