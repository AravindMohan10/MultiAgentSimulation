#!/usr/bin/env python3
"""
Full BAML evaluation pipeline:

1. before: run episodes (pydantic) + analyze + figures + system diagram
2. baml-cli generate (after only, if client missing)
3. after: run episodes (baml) + analyze + figures + system diagram
4. comparison figure + README summary
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PY = sys.executable


def _run(cmd: list[str], env: dict | None = None) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(_ROOT), check=True, env=env)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=_ROOT / "output_Baml")
    p.add_argument("--maps", nargs="+", default=["open", "chokepoint", "standard_maze"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--skip-before", action="store_true")
    p.add_argument("--skip-after", action="store_true")
    p.add_argument("--skip-runs", action="store_true", help="Only analyze/figure existing logs")
    args = p.parse_args()

    root = args.output_root
    eval_py = _ROOT / "scripts" / "baml_eval"

    for phase in ("before", "after"):
        (root / phase / "data").mkdir(parents=True, exist_ok=True)
        (root / phase / "figures").mkdir(parents=True, exist_ok=True)
        (root / phase / "docs").mkdir(parents=True, exist_ok=True)

    if not args.skip_runs and not args.skip_before:
        _run(
            [
                _PY,
                str(eval_py / "run_experiment_batch.py"),
                "--phase",
                "before",
                "--maps",
                *args.maps,
                "--seed",
                str(args.seed),
                "--max-steps",
                str(args.max_steps),
                "--output-root",
                str(root),
            ]
        )

    # Analyze before
    before_log = root / "before" / "data" / "logs"
    if before_log.exists() and list(before_log.glob("*_steps.jsonl")):
        _run(
            [
                _PY,
                str(eval_py / "analyze_llm_output_quality.py"),
                "--log-dir",
                str(before_log),
                "--parser",
                "pydantic",
                "--out-csv",
                str(root / "before" / "data" / "llm_quality_per_step_before.csv"),
                "--out-json",
                str(root / "before" / "data" / "llm_quality_summary_before.json"),
            ]
        )
        _run(
            [
                _PY,
                str(eval_py / "generate_figures.py"),
                "--phase",
                "before",
                "--data-dir",
                str(root / "before" / "data"),
                "--figures-dir",
                str(root / "before" / "figures"),
            ]
        )
        _run(
            [
                _PY,
                str(eval_py / "generate_system_design.py"),
                "--phase",
                "before",
                "--out",
                str(root / "before" / "figures" / "system_design_before.png"),
            ]
        )

    baml_client = _ROOT / "baml_client"
    if not args.skip_after and not baml_client.exists():
        print("\nGenerating BAML client...")
        _run([_PY, "-m", "pip", "install", "baml-py", "-q"])
        _run(["baml-cli", "generate"], env=None)

    if not args.skip_runs and not args.skip_after:
        _run(
            [
                _PY,
                str(eval_py / "run_experiment_batch.py"),
                "--phase",
                "after",
                "--maps",
                *args.maps,
                "--seed",
                str(args.seed),
                "--max-steps",
                str(args.max_steps),
                "--output-root",
                str(root),
            ]
        )

    after_log = root / "after" / "data" / "logs"
    if after_log.exists() and list(after_log.glob("*_steps.jsonl")):
        _run(
            [
                _PY,
                str(eval_py / "analyze_llm_output_quality.py"),
                "--log-dir",
                str(after_log),
                "--parser",
                "baml",
                "--out-csv",
                str(root / "after" / "data" / "llm_quality_per_step_after.csv"),
                "--out-json",
                str(root / "after" / "data" / "llm_quality_summary_after.json"),
            ]
        )
        _run(
            [
                _PY,
                str(eval_py / "generate_figures.py"),
                "--phase",
                "after",
                "--data-dir",
                str(root / "after" / "data"),
                "--figures-dir",
                str(root / "after" / "figures"),
            ]
        )
        _run(
            [
                _PY,
                str(eval_py / "generate_system_design.py"),
                "--phase",
                "after",
                "--out",
                str(root / "after" / "figures" / "system_design_after.png"),
            ]
        )

    bj = root / "before" / "data" / "llm_quality_summary_before.json"
    aj = root / "after" / "data" / "llm_quality_summary_after.json"
    if bj.exists() and aj.exists():
        comp_dir = root / "comparison" / "figures"
        comp_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                _PY,
                str(eval_py / "generate_figures.py"),
                "--phase",
                "comparison",
                "--data-dir",
                str(root),
                "--figures-dir",
                str(comp_dir),
                "--before-json",
                str(bj),
                "--after-json",
                str(aj),
            ]
        )
        _run([_PY, str(eval_py / "write_report.py"), "--output-root", str(root)])

    print(f"\nDone. Artifacts under {root}")


if __name__ == "__main__":
    main()
