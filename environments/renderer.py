"""Pygame renderer for visualizing GridWorld episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from environments.objects import CellType


@dataclass(frozen=True, slots=True)
class RendererConfig:
    """Display configuration for the Pygame renderer."""

    cell_size_px: int = 48
    border_px: int = 1
    caption: str = "GridWorld Prototype"


class GridWorldRenderer:
    """Render integer-grid observations in a Pygame window."""

    def __init__(self, grid_shape: tuple[int, int], config: RendererConfig | None = None) -> None:
        """Create a renderer for a specific grid shape.

        Args:
            grid_shape: Grid shape in ``(rows, cols)`` format.
            config: Optional renderer configuration.
        """

        self._rows, self._cols = grid_shape
        self._config = config or RendererConfig()
        self._window_size = (
            self._cols * self._config.cell_size_px,
            self._rows * self._config.cell_size_px,
        )

        self._surface: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None

    def initialize(self) -> None:
        """Initialize the Pygame display resources."""

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption(self._config.caption)
        self._surface = pygame.display.set_mode(self._window_size)
        self._font = pygame.font.Font(None, 24)

    def close(self) -> None:
        """Dispose Pygame resources."""

        pygame.quit()

    def draw(self, observation: np.ndarray, hud_lines: tuple[str, ...] = ()) -> None:
        """Draw one frame.

        Args:
            observation: Integer grid observation.
            hud_lines: Optional lines of text rendered on top.
        """

        if self._surface is None:
            raise RuntimeError("Renderer is not initialized. Call initialize() first.")

        color_map: dict[CellType, tuple[int, int, int]] = {
            CellType.EMPTY: (245, 245, 245),
            CellType.START: (80, 120, 220),
            CellType.GOAL: (80, 200, 120),
            CellType.AGENT: (245, 160, 45),
            # special cells
            CellType.WALL: (40, 40, 40),
            CellType.HAZARD: (220, 80, 80),
            CellType.SLIPPERY: (90, 210, 240),
        }

        for row in range(self._rows):
            for col in range(self._cols):
                value = CellType(int(observation[row, col]))
                rect = pygame.Rect(
                    col * self._config.cell_size_px,
                    row * self._config.cell_size_px,
                    self._config.cell_size_px,
                    self._config.cell_size_px,
                )
                pygame.draw.rect(self._surface, color_map[value], rect)
                pygame.draw.rect(self._surface, (180, 180, 180), rect, self._config.border_px)

        if self._font is not None and hud_lines:
            for index, line in enumerate(hud_lines):
                text_surface = self._font.render(line, True, (20, 20, 20))
                self._surface.blit(text_surface, (8, 8 + index * 22))

        pygame.display.flip()

    @staticmethod
    def key_to_action(key: int) -> int | None:
        """Map arrow and WASD keys to action indices.

        Args:
            key: Pygame key constant.

        Returns:
            Action index for environment step or ``None`` if unmapped.
        """

        mapping: dict[int, int] = {
            pygame.K_UP: 0,
            pygame.K_w: 0,
            pygame.K_DOWN: 1,
            pygame.K_s: 1,
            pygame.K_LEFT: 2,
            pygame.K_a: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_SPACE: 4,
        }
        return mapping.get(key)

    @staticmethod
    def pump_events() -> list[pygame.event.Event]:
        """Return the currently pending Pygame events."""

        return pygame.event.get()

    @staticmethod
    def tick(clock: pygame.time.Clock, fps: int) -> None:
        """Advance frame timing with a fixed FPS cap."""

        clock.tick(fps)
