"""
Visualization package.

Intended contents:
- A 2D visualization frontend that reads world state or logs and renders the simulation.
- Adapters or APIs designed so that a future 3D renderer (e.g., Three.js or a game engine)
  can consume the same world state representation.

See `pygame_renderer.PygameRenderer` for a 2D live view of `WorldState`.
"""

from .pygame_renderer import PygameRenderer

__all__ = ["PygameRenderer"]

