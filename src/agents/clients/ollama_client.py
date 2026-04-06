"""
Ollama client: OpenAI-compatible chat-completions at local Ollama.

Drop-in for the same LLMClient contract as GroqClient:
http://localhost:11434/v1/chat/completions
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ..llm_agent import LLMClient, validate_target_position
from .groq_client import _extract_text, sanitize_llm_json

OLLAMA_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434/v1/chat/completions"
).rstrip("/")

_JSON_ONLY_SUFFIX = "Respond with valid JSON only, no other text."


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:2000]
    except Exception:
        return ""


def _post_ollama_chat_completion(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 60.0,
) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "MultiAgentSimulation/OllamaClient (urllib)",
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
        raise RuntimeError(f"Ollama request failed: {msg}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


class OllamaClient:
    """HTTP client for local Ollama; satisfies the LLMClient protocol."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 256,
        top_p: float | None = None,
        request_timeout_seconds: float | None = None,
        request_retries: int | None = None,
    ) -> None:
        self._model_name = model_name or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self._temperature = float(os.environ.get("OLLAMA_TEMPERATURE", str(temperature)))
        self._max_tokens = int(os.environ.get("OLLAMA_MAX_TOKENS", str(max_tokens)))
        self._top_p = float(top_p if top_p is not None else os.environ.get("OLLAMA_TOP_P", "0.9"))
        self._timeout_seconds = float(
            request_timeout_seconds
            if request_timeout_seconds is not None
            else os.environ.get("OLLAMA_TIMEOUT_SECONDS", "60")
        )
        self._request_retries = int(
            request_retries if request_retries is not None else os.environ.get("OLLAMA_REQUEST_RETRIES", "2")
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        want_debug = os.environ.get("OLLAMA_DEBUG_RAW_OUTPUT", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        system_with_hint = (system_prompt or "").rstrip() + "\n\n" + _JSON_ONLY_SUFFIX

        base: Dict[str, Any] = {
            "model": self._model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": 0.9,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_with_hint},
                {"role": "user", "content": user_prompt},
            ],
        }

        payloads: list[Dict[str, Any]] = [base]

        last_err: Optional[RuntimeError] = None
        for payload in payloads:
            for attempt in range(2):
                try:
                    raw = _post_ollama_chat_completion(OLLAMA_URL, payload, timeout=self._timeout_seconds)
                    obj = json.loads(raw)
                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not choices:
                        raise RuntimeError(f"Ollama returned no choices: {obj}")
                    text = _extract_text(choices[0])
                    if not text:
                        raise RuntimeError(f"Ollama returned empty message content: {obj}")

                    if want_debug:
                        print("RAW LLM OUTPUT:", text)

                    s = text.strip()
                    start = s.find("{")
                    end = s.rfind("}")
                    if start == -1 or end == -1 or end <= start:
                        raise RuntimeError(f"Ollama output does not contain a JSON object: {text[:200]}")
                    candidate = s[start : end + 1]
                    try:
                        sanitized = sanitize_llm_json(candidate)
                        parsed = json.loads(sanitized)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"Ollama action output is not valid JSON: {exc}") from exc

                    if not isinstance(parsed, dict):
                        raise RuntimeError("Ollama action JSON is not an object.")
                    if "intent" not in parsed:
                        raise RuntimeError("Ollama action JSON missing required key: intent.")
                    if "target_position" not in parsed and "movement" not in parsed:
                        raise RuntimeError(
                            "Ollama action JSON must include target_position [x,y] and/or movement [dx,dy,dz]."
                        )
                    if "target_position" in parsed:
                        parsed["target_position"] = validate_target_position(parsed.get("target_position"))

                    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
                except RuntimeError as exc:
                    last_err = exc
                    if attempt < 1:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    raise

        assert last_err is not None
        raise last_err


def build_default_ollama_clients(
    model: Optional[str] = None,
) -> Dict[str, LLMClient]:
    m = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
    client: LLMClient = OllamaClient(m)  # type: ignore[assignment]
    return {
        "ollama": client,
    }
