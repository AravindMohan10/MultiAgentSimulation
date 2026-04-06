"""
Google Gemini client (AI Studio / google-generativeai).

Set GEMINI_API_KEY in the environment. Optional GEMINI_MODEL (default gemini-2.0-flash).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.agents.llm_agent import LLMClient


def _extract_text(response: Any) -> str:
    """Best-effort text from Gemini response (handles blocks / empty)."""
    try:
        if hasattr(response, "text") and response.text:
            return response.text.strip()
    except Exception:
        pass
    parts: list[str] = []
    if getattr(response, "candidates", None):
        for c in response.candidates:
            content = getattr(c, "content", None)
            if not content:
                continue
            for p in getattr(content, "parts", []) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
    return "\n".join(parts).strip()


class GeminiClient(LLMClient):
    """Single-turn completion via Gemini REST/SDK."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> None:
        self._model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it or pass api_key= to GeminiClient."
            )
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(
            self._model_name,
            generation_config={
                "temperature": self._temperature,
                "max_output_tokens": self._max_output_tokens,
            },
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_model()
        assert self._model is not None
        # Single string: many models follow system-like first block if we prefix clearly
        combined = (
            f"[SYSTEM — follow exactly; output only valid JSON as instructed]\n{system_prompt}\n\n"
            f"[USER]\n{user_prompt}"
        )
        response = self._model.generate_content(combined)
        text = _extract_text(response)
        if not text:
            finish = getattr(getattr(response, "candidates", [None])[0] or None, "finish_reason", None)
            raise RuntimeError(f"Gemini returned empty text (finish_reason={finish})")
        return text


def build_default_gemini_clients(
    api_key: Optional[str] = None,
    model_flash: Optional[str] = None,
) -> Dict[str, LLMClient]:
    """
    Map model_backend keys to clients. Use model_backend \"gemini-flash\" in AgentConfig.

    Example:
        llm_clients = build_default_gemini_clients()
        run_episode(..., llm_clients=llm_clients)
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    flash = model_flash or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    return {
        "gemini-flash": GeminiClient(flash, api_key=key),
        "gemini": GeminiClient(flash, api_key=key),
    }
