"""
Load repo-root ``.env`` into the process environment (optional ``python-dotenv``).

Does nothing if ``.env`` is missing or ``python-dotenv`` is not installed.
Shell exports (``export GROQ_API_KEY=...``) still take precedence for already-set vars
unless you use ``override=True`` (not used here).
"""

from __future__ import annotations

from pathlib import Path


def load_local_env(*, repo_root: Path | None = None) -> None:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)
