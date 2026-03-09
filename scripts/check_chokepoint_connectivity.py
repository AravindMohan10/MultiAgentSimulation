#!/usr/bin/env python3
"""
One-off BFS connectivity spot-check for chokepoint maps across seeds.

Uses the same 2.0-resolution occupancy grid and BFS rules as map_validator
(diagonal corner-cutting disallowed). Samples random pairs of free vertices until
n_pairs successes or failure (no path).
"""

from __future__ import annotations

import os
import random
import sys
from typing import List, Set, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.map_validator import MapComplexityValidator, _build_pair_grids


def _reachable_component(
    walkable: np.ndarray, start: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """All vertex indices reachable from start via cardinal + diagonal (with gate)."""
    from collections import deque

    sx, sy = start
    nx, ny = walkable.shape
    if not walkable[sx, sy]:
        return set()
    seen: Set[Tuple[int, int]] = {start}
    q: deque[Tuple[int, int]] = deque([start])
    moves = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    while q:
        cx, cy = q.popleft()
        for dx, dy in moves:
            nx2, ny2 = cx + dx, cy + dy
            if nx2 < 0 or ny2 < 0 or nx2 >= nx or ny2 >= ny:
                continue
            if not walkable[nx2, ny2]:
                continue
            if dx != 0 and dy != 0:
                if not walkable[cx + dx, cy] or not walkable[cx, cy + dy]:
                    continue
            if (nx2, ny2) not in seen:
                seen.add((nx2, ny2))
                q.append((nx2, ny2))
    return seen


def _path_exists(
    walkable: np.ndarray,
    s: Tuple[int, int],
    g: Tuple[int, int],
) -> bool:
    if not walkable[s[0], s[1]] or not walkable[g[0], g[1]]:
        return False
    if s == g:
        return True
    return g in _reachable_component(walkable, s)


def run_seed(
    seed: int,
    n_pairs: int,
    rng: random.Random,
) -> Tuple[bool, int, int]:
    """Returns (all_connected, free_cells, pairs_checked)."""
    obstacles, world_size, _ = MapComplexityValidator._generate_map_for_seed(
        "chokepoint", seed
    )
    # Match validator obstacle stamping (default agent_radius on disks).
    _, g2 = _build_pair_grids(obstacles, world_size)
    walkable = ~g2

    free_cells: List[Tuple[int, int]] = [
        (ix, iy)
        for ix in range(walkable.shape[0])
        for iy in range(walkable.shape[1])
        if walkable[ix, iy]
    ]
    if len(free_cells) < 2:
        return False, len(free_cells), 0

    # Strong check: entire walkable graph should be one component.
    comp = _reachable_component(walkable, free_cells[0])
    if len(comp) != len(free_cells):
        return False, len(free_cells), 0

    checked = 0
    for _ in range(n_pairs):
        a = free_cells[rng.randrange(len(free_cells))]
        b = free_cells[rng.randrange(len(free_cells))]
        if a == b:
            continue
        checked += 1
        if not _path_exists(walkable, a, b):
            return False, len(free_cells), checked

    return True, len(free_cells), checked


def main() -> None:
    n_seeds = 10
    n_pairs_per_seed = 80
    rng = random.Random(12345)
    print(
        "Chokepoint connectivity (2.0 grid, validator BFS rules)\n"
        f"  Seeds: 0–{n_seeds - 1}, random pairs per seed: {n_pairs_per_seed}\n"
    )
    all_ok = True
    for seed in range(n_seeds):
        ok, nfree, checked = run_seed(seed, n_pairs_per_seed, rng)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(
            f"  seed {seed:2d}: {status}  free_vertices={nfree:5d}  pairs_checked={checked:3d}"
        )
    print()
    if all_ok:
        print("All seeds: single connected component + sampled pairs connected.")
    else:
        print("FAILED: disconnected regions or missing path for sampled pair.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
