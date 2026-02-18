"""Typed objects and constants used by the GridWorld environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CellType(IntEnum):
    """Integer encodings used in the environment observation grid."""

    EMPTY = 0
    START = 1
    GOAL = 2
    AGENT = 3
    WALL = 4
    HAZARD = 5
    SLIPPERY = 6


@dataclass(frozen=True, slots=True)
class Position:
    """2D grid coordinate in ``(row, col)`` format."""

    row: int
    col: int


def manhattan_distance(a: Position, b: Position) -> int:
    """Return Manhattan distance between two positions.

    Args:
        a: First position.
        b: Second position.

    Returns:
        Number of grid steps between ``a`` and ``b``.
    """

    return abs(a.row - b.row) + abs(a.col - b.col)
