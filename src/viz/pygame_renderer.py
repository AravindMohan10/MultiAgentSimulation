"""
2D Pygame visualization: read-only view of WorldState.

Maps any world_size and obstacle layout from the simulation core onto a fixed
window. Does not modify physics or agents. Suitable for live runs or replay.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..core.models import AgentType, EnvironmentConfig, WorldState

try:
    import pygame
except ImportError as exc:  # pragma: no cover
    pygame = None  # type: ignore
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None


class PygameRenderer:
    """
    Render WorldState to a Pygame window.

    - World coordinates: x in [0, world_width], y in [0, world_height] (e.g. 160x160 research maps).
    - Screen: linear scale to window (y flipped so +y is up visually).
    """

    def __init__(
        self,
        env_config: EnvironmentConfig,
        window_size: Tuple[int, int] = (1024, 1024),
        *,
        fps_cap: int = 30,
        show_vision: bool = False,
    ) -> None:
        if pygame is None:
            raise ImportError(
                "pygame is required for PygameRenderer. Install with: pip install pygame"
            ) from _PYGAME_IMPORT_ERROR

        self._env = env_config
        self._ww, self._wh = float(env_config.world_size[0]), float(env_config.world_size[1])
        self._win_w, self._win_h = int(window_size[0]), int(window_size[1])
        self._fps_cap = max(1, int(fps_cap))
        self._show_vision = show_vision

        self._screen: Any = None
        self._clock: Any = None
        self._initialized = False

        # Colors (RGB)
        self._color_bg_flat = (235, 240, 230)
        self._color_bg_hilly = (210, 220, 200)
        self._color_bg_urban = (180, 180, 185)
        self._color_obstacle = (90, 90, 95)
        self._color_boundary = (40, 40, 45)
        self._color_hero = (50, 120, 220)
        self._color_villain = (220, 70, 50)
        self._color_vision = (100, 150, 255, 40)

    def _scale(self) -> Tuple[float, float]:
        if self._ww <= 0 or self._wh <= 0:
            return 1.0, 1.0
        return self._win_w / self._ww, self._win_h / self._wh

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx, sy = self._scale()
        px = int(wx * sx)
        py = int(self._win_h - wy * sy)
        return px, py

    def _radius_screen(self, world_radius: float) -> int:
        sx, sy = self._scale()
        r = max(world_radius * min(sx, sy), 2.0)
        return int(r)

    def _terrain_color(self, world_state: WorldState) -> Tuple[int, int, int]:
        t = (world_state.terrain.terrain_type or self._env.terrain_type or "flat").lower()
        if t == "hilly":
            return self._color_bg_hilly
        if t == "urban":
            return self._color_bg_urban
        return self._color_bg_flat

    def init(self) -> None:
        if self._initialized:
            return
        pygame.init()
        pygame.display.set_caption("Multi-agent pursuit–evasion")
        self._screen = pygame.display.set_mode((self._win_w, self._win_h))
        self._clock = pygame.time.Clock()
        self._initialized = True

    def handle_events(self) -> bool:
        """
        Process pygame events. Return False if the user closed the window.
        """
        if not self._initialized or pygame is None:
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render(self, world_state: WorldState) -> None:
        if not self._initialized or self._screen is None:
            self.init()

        assert self._screen is not None
        screen = self._screen
        screen.fill(self._terrain_color(world_state))

        # World boundary rectangle
        corners = [
            self.world_to_screen(0, 0),
            self.world_to_screen(self._ww, 0),
            self.world_to_screen(self._ww, self._wh),
            self.world_to_screen(0, self._wh),
        ]
        pygame.draw.lines(screen, self._color_boundary, True, corners, 2)

        # Obstacles
        for obs in world_state.obstacles:
            cx, cy = self.world_to_screen(obs.position.x, obs.position.y)
            r = self._radius_screen(obs.radius)
            pygame.draw.circle(screen, self._color_obstacle, (cx, cy), r)

        # Agents (vision optional)
        for agent in world_state.agents.values():
            if not agent.alive:
                continue
            px, py = self.world_to_screen(agent.position.x, agent.position.y)
            if self._show_vision:
                # Approximate vision: use env base_visibility * weather modifier
                vis = self._env.base_visibility_radius * world_state.weather.visibility_modifier
                vr = self._radius_screen(vis)
                vis_surf = pygame.Surface((vr * 2, vr * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    vis_surf,
                    self._color_vision,
                    (vr, vr),
                    vr,
                )
                screen.blit(vis_surf, (px - vr, py - vr))

            r_agent = max(6, self._radius_screen(0.8))
            if agent.agent_type == AgentType.HERO:
                color = self._color_hero
            else:
                color = self._color_villain
            pygame.draw.circle(screen, color, (px, py), r_agent)

        # HUD: step and time
        font = pygame.font.Font(None, 22)
        hud = f"t={world_state.time:.1f}  step={world_state.step_index}"
        surf = font.render(hud, True, (20, 20, 20))
        screen.blit(surf, (8, 8))

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self._fps_cap)

    def close(self) -> None:
        if self._initialized and pygame is not None:
            pygame.quit()
        self._initialized = False
        self._screen = None
        self._clock = None
