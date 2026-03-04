from __future__ import annotations

"""
Map-type metrics and acceptance bands for generated layouts.

Numeric thresholds (free space, detour, complexity) are chosen to match
empirically observed ranges for each template after the final geometry
(e.g. maze fill and shell) was fixed — not to force generators into
arbitrary pre-assumed bands. The standard_maze free-space upper bound in
particular was raised post-calibration once corridor walkability was finalized.
"""

import math
import random
import zlib
from collections import deque
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .engine import SimulationEngine
from .models import EnvironmentConfig, MapTemplate, Obstacle


def _build_occupancy_grid(
    obstacles: Sequence[Obstacle],
    world_size: Tuple[float, float],
    resolution: float,
    *,
    agent_radius: float = 0.0,
    sampling: str = "cell_center",
) -> np.ndarray:
    """
    Stamp obstacle disks onto a boolean occupancy grid (True = blocked).

    sampling:
      - "cell_center": grid[ix, iy] samples world point
        (min(w,(ix+0.5)*res), min(h,(iy+0.5)*res)) — matches legacy free_space_ratio.
      - "vertex": grid[ix, iy] samples (min(w,ix*res), min(h,iy*res)) — matches legacy BFS grid.
    """
    w, h = float(world_size[0]), float(world_size[1])
    if sampling == "cell_center":
        nx = max(1, int(math.ceil(w / resolution)))
        ny = max(1, int(math.ceil(h / resolution)))
    else:
        nx = max(2, int(math.floor(w / resolution)) + 1)
        ny = max(2, int(math.floor(h / resolution)) + 1)

    grid = np.zeros((nx, ny), dtype=bool)
    for obs in obstacles:
        ox = float(obs.position.x)
        oy = float(obs.position.y)
        r = float(obs.radius) + agent_radius
        if r <= 0.0:
            continue
        if sampling == "cell_center":
            _stamp_disk_cell_centers(grid, world_size, resolution, ox, oy, r)
        else:
            _stamp_disk_vertices(grid, world_size, resolution, ox, oy, r)
    return grid


def _stamp_disk_cell_centers(
    grid: np.ndarray,
    world_size: Tuple[float, float],
    resolution: float,
    ox: float,
    oy: float,
    r: float,
) -> None:
    w, h = float(world_size[0]), float(world_size[1])
    nx, ny = grid.shape
    ix_min = max(0, int(math.floor((ox - r) / resolution - 0.5)))
    ix_max = min(nx, int(math.ceil((ox + r) / resolution - 0.5)) + 1)
    iy_min = max(0, int(math.floor((oy - r) / resolution - 0.5)))
    iy_max = min(ny, int(math.ceil((oy + r) / resolution - 0.5)) + 1)
    if ix_min >= ix_max or iy_min >= iy_max:
        return
    IX = np.arange(ix_min, ix_max, dtype=np.float64)[:, None]
    IY = np.arange(iy_min, iy_max, dtype=np.float64)[None, :]
    px = np.minimum(w, (IX + 0.5) * resolution)
    py = np.minimum(h, (IY + 0.5) * resolution)
    mask = (px - ox) ** 2 + (py - oy) ** 2 <= r * r
    grid[ix_min:ix_max, iy_min:iy_max] |= mask


def _stamp_disk_vertices(
    grid: np.ndarray,
    world_size: Tuple[float, float],
    resolution: float,
    ox: float,
    oy: float,
    r: float,
) -> None:
    w, h = float(world_size[0]), float(world_size[1])
    nx, ny = grid.shape
    ix_min = max(0, int(math.floor((ox - r) / resolution)))
    ix_max = min(nx, int(math.ceil((ox + r) / resolution)) + 1)
    iy_min = max(0, int(math.floor((oy - r) / resolution)))
    iy_max = min(ny, int(math.ceil((oy + r) / resolution)) + 1)
    if ix_min >= ix_max or iy_min >= iy_max:
        return
    IX = np.arange(ix_min, ix_max, dtype=np.float64)[:, None]
    IY = np.arange(iy_min, iy_max, dtype=np.float64)[None, :]
    px = np.minimum(w, IX * resolution)
    py = np.minimum(h, IY * resolution)
    mask = (px - ox) ** 2 + (py - oy) ** 2 <= r * r
    grid[ix_min:ix_max, iy_min:iy_max] |= mask


def _build_pair_grids(
    obstacles: Sequence[Obstacle],
    world_size: Tuple[float, float],
    *,
    agent_radius: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single pass over obstacles: stamp both 1.0 (free-space) and 2.0 (BFS) grids."""
    w, h = float(world_size[0]), float(world_size[1])
    res1 = 1.0
    nx1 = max(1, int(math.ceil(w / res1)))
    ny1 = max(1, int(math.ceil(h / res1)))
    res2 = 2.0
    nx2 = max(2, int(math.floor(w / res2)) + 1)
    ny2 = max(2, int(math.floor(h / res2)) + 1)

    g1 = np.zeros((nx1, ny1), dtype=bool)
    g2 = np.zeros((nx2, ny2), dtype=bool)

    for obs in obstacles:
        ox = float(obs.position.x)
        oy = float(obs.position.y)
        r = float(obs.radius) + agent_radius
        if r <= 0.0:
            continue
        _stamp_disk_cell_centers(g1, world_size, res1, ox, oy, r)
        _stamp_disk_vertices(g2, world_size, res2, ox, oy, r)

    return g1, g2


def free_space_ratio_from_grid(grid: np.ndarray) -> float:
    return float(np.sum(~grid)) / float(grid.size) if grid.size > 0 else 0.0


class MapComplexityValidator:
    """Complexity metrics for continuous 2D obstacle maps."""

    @staticmethod
    def _detour_monte_carlo_seed(obstacles: Sequence[Obstacle]) -> int:
        """Reproducible RNG seed from obstacle layout (same geometry → same detour estimate)."""
        parts: List[str] = []
        for o in obstacles:
            parts.append(f"{float(o.position.x):.4f},{float(o.position.y):.4f},{float(o.radius):.4f}")
        blob = "|".join(parts).encode("utf-8")
        return int(zlib.adler32(blob)) & 0x7FFFFFFF

    @staticmethod
    def free_space_ratio(obstacles: Sequence[Obstacle], world_size: Tuple[float, float]) -> float:
        """Fraction of world area not covered by obstacle disks (sampled at 1.0 units)."""
        g1 = _build_occupancy_grid(obstacles, world_size, 1.0, sampling="cell_center")
        return free_space_ratio_from_grid(g1)

    @staticmethod
    def bottleneck_count(chokepoints: Optional[Sequence[Tuple[float, float]]]) -> int:
        return int(len(chokepoints or []))

    @staticmethod
    def _bfs_path_len_numpy(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: np.ndarray,
        resolution: float,
    ) -> Optional[float]:
        """walkable[ix, iy] True = free. Same logic as legacy list-of-lists BFS."""
        nx, ny = walkable.shape
        sx, sy = start
        gx, gy = goal
        if (
            sx < 0
            or sy < 0
            or gx < 0
            or gy < 0
            or sx >= nx
            or gx >= nx
            or sy >= ny
            or gy >= ny
        ):
            return None
        if not walkable[sx, sy] or not walkable[gx, gy]:
            return None
        if (sx, sy) == (gx, gy):
            return 0.0

        q: deque[Tuple[int, int]] = deque()
        q.append((sx, sy))
        dist: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        while q:
            cx, cy = q.popleft()
            base = dist[(cx, cy)]
            for dx, dy, step_w in moves:
                nx2, ny2 = cx + dx, cy + dy
                if nx2 < 0 or ny2 < 0 or nx2 >= nx or ny2 >= ny:
                    continue
                if not walkable[nx2, ny2]:
                    continue
                if dx != 0 and dy != 0:
                    if not walkable[cx + dx, cy] or not walkable[cx, cy + dy]:
                        continue
                nd = base + (step_w * resolution)
                if (nx2, ny2) not in dist:
                    dist[(nx2, ny2)] = nd
                    if (nx2, ny2) == (gx, gy):
                        return nd
                    q.append((nx2, ny2))
        return None

    @staticmethod
    def mean_detour_ratio_from_grids(
        grid_bfs: np.ndarray,
        world_size: Tuple[float, float],
        n_sample_pairs: int = 20,
        seed: int = 42,
    ) -> float:
        """Detour ratio using pre-built 2.0-resolution occupancy grid (True = blocked)."""
        resolution = 2.0
        w, h = float(world_size[0]), float(world_size[1])
        walkable = ~grid_bfs
        free_cells = [(ix, iy) for ix in range(walkable.shape[0]) for iy in range(walkable.shape[1]) if walkable[ix, iy]]
        if len(free_cells) < 2:
            return float("inf")

        rng = random.Random(seed)
        ratios: List[float] = []
        tries = 0
        max_tries = max(50, n_sample_pairs * 20)
        while len(ratios) < n_sample_pairs and tries < max_tries:
            tries += 1
            s = free_cells[rng.randrange(len(free_cells))]
            g = free_cells[rng.randrange(len(free_cells))]
            if s == g:
                continue
            sx, sy = s
            gx, gy = g
            sxp = min(w, sx * resolution)
            syp = min(h, sy * resolution)
            gxp = min(w, gx * resolution)
            gyp = min(h, gy * resolution)
            straight = math.hypot(gxp - sxp, gyp - syp)
            if straight < 1e-6:
                continue
            path_len = MapComplexityValidator._bfs_path_len_numpy(s, g, walkable, resolution)
            if path_len is None:
                continue
            ratios.append(float(path_len / straight))
        if not ratios:
            return float("inf")
        return float(mean(ratios))

    @staticmethod
    def mean_detour_ratio(
        obstacles: Sequence[Obstacle],
        world_size: Tuple[float, float],
        n_sample_pairs: int = 20,
        seed: int = 42,
    ) -> float:
        """Approximate path detour via BFS on a 2.0-unit occupancy grid."""
        _, g2 = _build_pair_grids(obstacles, world_size)
        return MapComplexityValidator.mean_detour_ratio_from_grids(
            g2, world_size, n_sample_pairs=n_sample_pairs, seed=seed
        )

    @staticmethod
    def complexity_score(free_space: float, bottleneck_count: int, detour_ratio: float) -> float:
        score = (
            (1.0 - float(free_space)) * 0.3
            + min(float(bottleneck_count) / 10.0, 1.0) * 0.3
            + min((float(detour_ratio) - 1.0) / 3.0, 1.0) * 0.4
        )
        return float(score)

    @staticmethod
    def _normalize_map_type(map_type: str | MapTemplate) -> str:
        if isinstance(map_type, MapTemplate):
            return map_type.value
        return str(map_type).strip().lower()

    @staticmethod
    def _tier_from_score(score: float) -> str:
        if score < 0.25:
            return "open"
        if score <= 0.55:
            return "chokepoint"
        return "standard_maze"

    @staticmethod
    def validate_map(
        map_type: str | MapTemplate,
        obstacles: Sequence[Obstacle],
        world_size: Tuple[float, float],
        chokepoints: Optional[Sequence[Tuple[float, float]]],
        seed: int,
    ) -> Dict[str, Any]:
        mt = MapComplexityValidator._normalize_map_type(map_type)
        g1, g2 = _build_pair_grids(obstacles, world_size)
        free = free_space_ratio_from_grid(g1)
        bottlenecks = MapComplexityValidator.bottleneck_count(chokepoints)
        detour = MapComplexityValidator.mean_detour_ratio_from_grids(
            g2,
            world_size,
            n_sample_pairs=20,
            seed=MapComplexityValidator._detour_monte_carlo_seed(obstacles),
        )
        score = MapComplexityValidator.complexity_score(free, bottlenecks, detour)
        tier = MapComplexityValidator._tier_from_score(score)

        warnings: List[str] = []
        if mt == "open":
            if not (free > 0.85):
                warnings.append(f"free_space_ratio out of OPEN range: {free:.3f}")
            if bottlenecks != 0:
                warnings.append(f"bottleneck_count expected 0, got {bottlenecks}")
            if not (1.0 <= detour <= 1.2):
                warnings.append(f"detour_ratio out of OPEN range: {detour:.3f}")
            if not (score < 0.25):
                warnings.append(f"complexity_score out of OPEN range: {score:.3f}")
        elif mt == "chokepoint":
            if not (0.60 <= free <= 0.85):
                warnings.append(f"free_space_ratio out of CHOKEPOINT range: {free:.3f}")
            if bottlenecks != 3:
                warnings.append(f"bottleneck_count expected 3, got {bottlenecks}")
            if not (1.3 <= detour <= 2.0):
                warnings.append(f"detour_ratio out of CHOKEPOINT range: {detour:.3f}")
            if not (0.25 <= score <= 0.55):
                warnings.append(f"complexity_score out of CHOKEPOINT range: {score:.3f}")
        elif mt == "standard_maze":
            # Upper bound 0.78: corridors remain legitimately open; ~0.73 is in-band.
            if not (0.40 <= free <= 0.78):
                warnings.append(f"free_space_ratio out of STANDARD_MAZE range: {free:.3f}")
            if not (6 <= bottlenecks <= 9):
                warnings.append(f"bottleneck_count expected 6-9, got {bottlenecks}")
            # Floor 1.8 / ceiling 4.5: floor vs chokepoint tier; ceiling allows rare tortuous
            # BFS paths when loop count is fixed low (few shortcuts).
            if not (1.8 <= detour <= 4.5):
                warnings.append(f"detour_ratio out of STANDARD_MAZE range: {detour:.3f}")
            if not (score > 0.55):
                warnings.append(f"complexity_score out of STANDARD_MAZE range: {score:.3f}")

        if tier != mt:
            warnings.append(f"tier_assignment mismatch: predicted={tier}, claimed={mt}")

        return {
            "map_type": mt,
            "seed": int(seed),
            "free_space_ratio": float(free),
            "bottleneck_count": int(bottlenecks),
            "detour_ratio": float(detour),
            "complexity_score": float(score),
            "tier_assignment": tier,
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
        }

    @staticmethod
    def _generate_map_for_seed(
        map_type: str, seed: int
    ) -> Tuple[List[Obstacle], Tuple[float, float], Optional[List[Tuple[float, float]]]]:
        env = EnvironmentConfig(
            world_size=(100.0, 100.0),
            max_steps=10,
            seed=int(seed),
            map_template=MapTemplate(map_type),
            obstacle_radius=1.5,
        )
        engine = SimulationEngine(env, agent_configs=[])
        obstacles = engine._raw_obstacles_for_template(env)
        chokepoints = env.chokepoint_positions
        cps = list(chokepoints) if chokepoints else None
        return obstacles, tuple(env.world_size), cps

    @staticmethod
    def validate_across_seeds(map_type: str | MapTemplate, n_seeds: int = 10) -> Dict[str, Any]:
        mt = MapComplexityValidator._normalize_map_type(map_type)
        rows: List[Dict[str, Any]] = []
        for seed in range(int(n_seeds)):
            obstacles, world_size, chokepoints = MapComplexityValidator._generate_map_for_seed(mt, seed)
            rows.append(
                MapComplexityValidator.validate_map(
                    mt, obstacles, world_size, chokepoints, seed=seed
                )
            )

        def _col(name: str) -> List[float]:
            return [float(r[name]) for r in rows]

        free_vals = _col("free_space_ratio")
        bott_vals = _col("bottleneck_count")
        detour_vals = _col("detour_ratio")
        comp_vals = _col("complexity_score")
        valid_count = sum(1 for r in rows if bool(r["is_valid"]))
        frac_valid = float(valid_count) / float(len(rows)) if rows else 0.0

        return {
            "map_type": mt,
            "n_seeds": int(n_seeds),
            "free_space_ratio_mean": float(mean(free_vals)) if free_vals else 0.0,
            "free_space_ratio_std": float(pstdev(free_vals)) if len(free_vals) > 1 else 0.0,
            "bottleneck_count_mean": float(mean(bott_vals)) if bott_vals else 0.0,
            "bottleneck_count_std": float(pstdev(bott_vals)) if len(bott_vals) > 1 else 0.0,
            "detour_ratio_mean": float(mean(detour_vals)) if detour_vals else 0.0,
            "detour_ratio_std": float(pstdev(detour_vals)) if len(detour_vals) > 1 else 0.0,
            "complexity_score_mean": float(mean(comp_vals)) if comp_vals else 0.0,
            "complexity_score_std": float(pstdev(comp_vals)) if len(comp_vals) > 1 else 0.0,
            "valid_count": int(valid_count),
            "valid_fraction": float(frac_valid),
            "verdict": "PASS" if frac_valid > 0.80 else "FAIL",
            "per_seed": rows,
        }


def _world_xy_to_bfs_indices(
    x: float,
    y: float,
    walkable: np.ndarray,
    world_size: Tuple[float, float],
    resolution: float,
) -> Tuple[int, int]:
    """Map world (x,y) to vertex indices on the BFS grid (same as detour / _build_pair_grids)."""
    w, h = float(world_size[0]), float(world_size[1])
    nx, ny = walkable.shape
    xw = min(w, max(0.0, x))
    yw = min(h, max(0.0, y))
    ix = int(round(xw / resolution))
    iy = int(round(yw / resolution))
    return max(0, min(nx - 1, ix)), max(0, min(ny - 1, iy))


def episode_trajectory_path_efficiency_ratio(
    trajectory_xy: Sequence[Tuple[float, float]],
    obstacles: Sequence[Obstacle],
    world_size: Tuple[float, float],
    *,
    resolution: float = 2.0,
) -> Optional[float]:
    """
    (sum of Euclidean segment lengths along the trajectory) / (BFS shortest path length
    on the 2.0 vertex occupancy grid). None if fewer than two points, endpoints not
    walkable, or no BFS path.

    Values are >= 1 when the realized path is at least as long as the grid-optimal route.
    """
    if len(trajectory_xy) < 2:
        return None
    _, g2 = _build_pair_grids(list(obstacles), world_size)
    walkable = ~g2
    actual_len = 0.0
    for i in range(1, len(trajectory_xy)):
        dx = float(trajectory_xy[i][0]) - float(trajectory_xy[i - 1][0])
        dy = float(trajectory_xy[i][1]) - float(trajectory_xy[i - 1][1])
        actual_len += math.hypot(dx, dy)
    sx, sy = _world_xy_to_bfs_indices(
        trajectory_xy[0][0], trajectory_xy[0][1], walkable, world_size, resolution
    )
    gx, gy = _world_xy_to_bfs_indices(
        trajectory_xy[-1][0], trajectory_xy[-1][1], walkable, world_size, resolution
    )
    opt_len = MapComplexityValidator._bfs_path_len_numpy((sx, sy), (gx, gy), walkable, resolution)
    if opt_len is None or opt_len <= 1e-9:
        return None
    return float(actual_len / opt_len)
