#!/usr/bin/env bash
# Remove generated logs and Python caches from the repo working tree.
# Run from repo root: bash scripts/clean_ephemeral.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Repo: $ROOT"
echo "Removing: logs_*/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/ (if present)"
shopt -s nullglob
for d in logs_*; do rm -rf "$d"; done
shopt -u nullglob
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
echo "Done."
