"""
OpenRouter client using the chat-completions HTTP API.

Environment:
- ``OPENROUTER_API_KEY`` — required.
- ``OPENROUTER_MODEL`` — optional, default ``openrouter/free`` (OpenRouter free-model router).
- ``OPENROUTER_BASE_URL`` — optional, default ``https://openrouter.ai/api/v1/chat/completions``.
- ``OPENROUTER_JSON_MODE`` — optional ``0/1``; when ``1`` (default), request ``response_format`` JSON mode
  and automatically retry without it if the gateway returns HTTP 4xx (some free models reject JSON mode).
- ``OPENROUTER_MAX_TOKENS`` — optional, default ``256`` (some routed free models use a
  ``reasoning`` channel; too-low caps yield ``content: null`` and ``finish_reason: length``).
- ``OPENROUTER_PROVIDER_ORDER`` — optional comma-separated provider order, e.g.
  ``open-inference``. When set, request includes provider routing preferences.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import json
import urllib.error
import urllib.request

from ..llm_agent import LLMClient


# Full URL to POST chat completions (override only if OpenRouter changes routing).
OPENROUTER_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
).rstrip("/")


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
        # OpenRouter accepts HTTP-Referer; some stacks also expect standard Referer.
        "HTTP-Referer": "https://github.com/aravind-mohan/multi-agent-sim",
        "Referer": "https://github.com/aravind-mohan/multi-agent-sim",
        "X-Title": "multi-agent-pursuit-evasion-sim",
        "User-Agent": "MultiAgentSimulation/OpenRouterClient (urllib)",
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
        raise RuntimeError(f"OpenRouter request failed: {msg}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


def _provider_pref_for_model(model_name: str) -> Optional[Dict[str, Any]]:
    # Allow explicit override via env:
    #   OPENROUTER_PROVIDER_ORDER="open-inference,..." 
    order_env = os.environ.get("OPENROUTER_PROVIDER_ORDER", "").strip()
    if order_env:
        order = [p.strip() for p in order_env.split(",") if p.strip()]
        if order:
            return {"order": order, "allow_fallbacks": True}

    # OpenRouter free GPT-OSS routes are commonly served by open-inference.
    if model_name.startswith("openai/gpt-oss-"):
        return {"order": ["open-inference"], "allow_fallbacks": True}
    return None


def _json_mode_retryable(err_text: str) -> bool:
    e = err_text.lower()
    return any(
        token in e
        for token in (
            "http 400",
            "http 404",
            "http 422",
            "not support",
            "unsupported",
            "invalid_request",
        )
    )


def _available_providers_from_error(err_text: str) -> list[str]:
    """
    Parse OpenRouter error text and extract metadata.available_providers when present.
    """
    # Error text usually ends with a JSON object after "HTTP ...: ".
    start = err_text.find("{")
    end = err_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        obj = json.loads(err_text[start : end + 1])
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    err = obj.get("error")
    if not isinstance(err, dict):
        return []
    meta = err.get("metadata")
    if not isinstance(meta, dict):
        return []
    providers = meta.get("available_providers")
    if not isinstance(providers, list):
        return []
    out: list[str] = []
    for p in providers:
        if isinstance(p, str) and p.strip():
            out.append(p.strip())
    return out


def _parse_completion_response(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenRouter returned non-JSON response: {raw[:200]}") from exc

    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {obj}")

    text = _extract_text(choices[0])
    if not text:
        raise RuntimeError(f"OpenRouter returned empty message content: {obj}")
    return text


def _extract_text(choice: Any) -> str:
    """
    Extract text from an OpenRouter chat choice.

    The SDK-normalized response typically has:
        choices[0].message.content  (string or list of segments).
    """
    if choice is None:
        return ""
    message = getattr(choice, "message", None) or choice.get("message") if isinstance(choice, dict) else None
    if message is None:
        return ""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text") or part.get("content")
                if t:
                    parts.append(str(t))
        if parts:
            return "\n".join(parts).strip()

    # Providers that emit chain-of-thought in ``reasoning`` / ``reasoning_details`` with ``content`` null
    # (e.g. some Nvidia / routed free models when max_tokens is tight or reasoning isn't excluded upstream).
    if isinstance(message, dict):
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
        details = message.get("reasoning_details")
        if isinstance(details, list):
            chunks: list[str] = []
            for block in details:
                if isinstance(block, dict):
                    t = block.get("text")
                    if t:
                        chunks.append(str(t))
            if chunks:
                return "\n".join(chunks).strip()
    return ""


class OpenRouterClient:
    """Simple HTTP client that satisfies the ``LLMClient`` protocol."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
    ) -> None:
        self._model_name = model_name or os.environ.get("OPENROUTER_MODEL", "openrouter/free")
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it or pass api_key= to OpenRouterClient."
            )

        max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "256"))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Optional: OpenAI-style JSON mode. Some free / older models reject this and may
        # surface as HTTP 4xx (including 404 on some gateways). Retry without it.
        want_json_mode = os.environ.get("OPENROUTER_JSON_MODE", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        base: Dict[str, Any] = {
            "model": self._model_name,
            "temperature": self._temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            # Prefer assistant JSON in ``message.content``; avoid content=null + reasoning-only.
            "reasoning": {"exclude": True},
        }
        provider_pref = _provider_pref_for_model(self._model_name)
        if provider_pref is not None:
            base["provider"] = provider_pref

        payloads: list[Dict[str, Any]] = []
        if want_json_mode:
            payloads.append({**base, "response_format": {"type": "json_object"}})
        payloads.append(dict(base))

        last_err: RuntimeError | None = None
        for payload in payloads:
            try:
                raw = _post_chat_completion(OPENROUTER_URL, self._api_key, payload)
                return _parse_completion_response(raw)
            except RuntimeError as exc:
                last_err = exc
                err_s = str(exc)
                if want_json_mode and "response_format" in payload and _json_mode_retryable(str(exc)):
                    continue
                # If OpenRouter reports provider mismatch (e.g. requested openai while model
                # is available on venice/open-inference), retry once with available_providers.
                providers = _available_providers_from_error(err_s)
                if providers:
                    retry_payload = dict(payload)
                    retry_payload["provider"] = {"order": providers, "allow_fallbacks": True}
                    raw = _post_chat_completion(OPENROUTER_URL, self._api_key, retry_payload)
                    return _parse_completion_response(raw)
                raise

        if last_err is not None:
            raise last_err
        raise RuntimeError("OpenRouter request failed: no attempts made.")


def build_default_openrouter_clients(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, LLMClient]:
    """
    Ready-to-use ``llm_clients`` mapping for :func:`run_episode`.

    Use ``AgentConfig.model_backend`` of ``\"openrouter-flash\"`` (or ``\"openrouter\"``)
    to route agents to this client.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    m = model or os.environ.get("OPENROUTER_MODEL", "openrouter/free")
    client: LLMClient = OpenRouterClient(m, api_key=key)  # type: ignore[assignment]
    return {
        "openrouter-flash": client,
        "openrouter": client,
    }

