#!/usr/bin/env python3
"""
Turn a finished episode's logs into the same ``inspect.md`` we use for groq runs.

Reads ``<log_dir>/<episode_id>_steps.jsonl`` + ``<log_dir>/<episode_id>_summary.json``
and writes ``<log_dir>/<out_name>``. If ``--episode-id`` is omitted, the most
recently modified ``*_steps.jsonl`` in ``--log-dir`` is used.

Usage:
    PYTHONPATH=. python scripts/run_episode_groq.py --map-template scattered --max-steps 80 --seed 0
    PYTHONPATH=. python scripts/dump_inspect_md.py --log-dir logs_groq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.baml_eval.single_run_inspect import _write_inspect_md


def _latest_episode_id(log_dir: Path) -> str:
    steps_files = sorted(
        log_dir.glob("*_steps.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not steps_files:
        raise SystemExit(f"No *_steps.jsonl found under {log_dir}")
    return steps_files[0].name.removesuffix("_steps.jsonl")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, default=Path("logs_groq"))
    p.add_argument(
        "--episode-id",
        default=None,
        help="Episode id (filename prefix before _steps.jsonl). Defaults to newest in --log-dir.",
    )
    p.add_argument("--out-name", default="inspect.md")
    args = p.parse_args()

    log_dir: Path = args.log_dir
    if not log_dir.is_dir():
        raise SystemExit(f"--log-dir does not exist: {log_dir}")

    episode_id = args.episode_id or _latest_episode_id(log_dir)
    out_path = log_dir / args.out_name

    _write_inspect_md(log_dir, episode_id, out_path)
    print(f"wrote {out_path}  (episode_id={episode_id})")


if __name__ == "__main__":
    main()
