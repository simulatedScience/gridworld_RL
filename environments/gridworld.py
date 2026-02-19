"""Gymnasium-compatible GridWorld environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.objects import CellType, Position


class Action(IntEnum):
    """Discrete action indices for movement in the grid."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


@dataclass(frozen=True, slots=True)
class GridWorldConfig:
    """Configuration for constructing a :class:`GridWorldEnv`.

    The observation remains an integer grid for the prototype stage.
    A future migration to float channels can be done by changing only
    :meth:`GridWorldEnv._get_observation` and ``observation_space``.
    """

    width: int
    height: int
    start_positions: tuple[Position, ...]  # at least one required; chosen randomly on reset
    goal_positions: tuple[Position, ...]   # at least one required; any reached position terminates the episode
    walls: tuple[Position, ...] = ()
    hazards: tuple[Position, ...] = ()
    slippery_tiles: tuple[Position, ...] = ()
    slip_probability: float = 0.35
    max_steps: int = 200
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    hazard_penalty: float = -1.0

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> GridWorldConfig:
        """Load a configuration from a JSON file.

        Args:
            file_path: Path to a JSON configuration file.

        Returns:
            Parsed :class:`GridWorldConfig`.
        """

        path = Path(file_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        if "start_positions" in payload:
            start_positions = tuple(Position(*p) for p in payload["start_positions"])
        else:
            raise ValueError("Config must contain 'start_positions'.")

        if "goal_positions" in payload:
            goal_positions = tuple(Position(*p) for p in payload["goal_positions"])
        else:
            raise ValueError("Config must contain 'goal_positions'.")

        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            start_positions=start_positions,
            goal_positions=goal_positions,
            walls=tuple(Position(*coords) for coords in payload.get("walls", [])),
            hazards=tuple(Position(*coords) for coords in payload.get("hazards", [])),
            slippery_tiles=tuple(Position(*coords) for coords in payload.get("slippery_tiles", [])),
            slip_probability=float(payload.get("slip_probability", 0.35)),
            max_steps=int(payload.get("max_steps", 200)),
            step_penalty=float(payload.get("step_penalty", -0.01)),
            goal_reward=float(payload.get("goal_reward", 1.0)),
            hazard_penalty=float(payload.get("hazard_penalty", -1.0)),
        )


class GridWorldEnv(gym.Env[np.ndarray, int]):
    """A 2D GridWorld environment with optional stochastic slippery tiles.

    Observation:
        Integer grid with shape ``(height, width)`` and values from
        :class:`environments.objects.CellType`.

    Actions:
        Discrete action space with indices from :class:`Action`.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: GridWorldConfig,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the GridWorld environment.

        Args:
            config: Environment configuration.
            render_mode: Optional Gymnasium render mode.
        """

        super().__init__()
        self.config = config
        self.render_mode = render_mode

        self.action_space: spaces.Discrete = spaces.Discrete(len(Action))
        max_cell_value = max(int(cell) for cell in CellType)
        self.observation_space: spaces.Box = spaces.Box(
            low=int(CellType.EMPTY),
            high=max_cell_value,
            shape=(config.height, config.width),
            dtype=np.int32,
        )

        self._step_count: int = 0
        self._agent_pos: Position = config.start_positions[0]  # overwritten properly in reset()
        self._goal_positions: frozenset[Position] = frozenset(config.goal_positions)

        self._walls: set[Position] = set(config.walls)
        self._hazards: set[Position] = set(config.hazards)
        self._slippery_tiles: set[Position] = set(config.slippery_tiles)

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration consistency and bounds."""

        def in_bounds(pos: Position) -> bool:
            return 0 <= pos.row < self.config.height and 0 <= pos.col < self.config.width

        if self.config.width <= 0 or self.config.height <= 0:
            raise ValueError("Grid dimensions must be positive integers.")

        if not self.config.start_positions:
            raise ValueError("At least one start position is required.")

        if not self.config.goal_positions:
            raise ValueError("At least one goal position is required.")

        for pos in self.config.start_positions:
            if not in_bounds(pos):
                raise ValueError(f"Start position {pos} is outside the grid bounds.")

        for pos in self.config.goal_positions:
            if not in_bounds(pos):
                raise ValueError(f"Goal position {pos} is outside the grid bounds.")

        if not 0.0 <= self.config.slip_probability <= 1.0:
            raise ValueError("slip_probability must be in [0.0, 1.0].")

        for pos in (*self._walls, *self._hazards, *self._slippery_tiles):
            if not in_bounds(pos):
                raise ValueError(f"Object position {pos} is outside the grid bounds.")

        start_set = frozenset(self.config.start_positions)
        goal_set = frozenset(self.config.goal_positions)

        if start_set & goal_set:
            raise ValueError("Start and goal positions cannot overlap.")
        if start_set & self._walls:
            raise ValueError("Start position(s) cannot be placed on walls.")
        if goal_set & self._walls:
            raise ValueError("Goal position(s) cannot be placed on walls.")
        if start_set & self._hazards:
            raise ValueError("Start position(s) cannot be placed on hazards.")
        if goal_set & self._hazards:
            raise ValueError("Goal position(s) cannot be placed on hazards.")
        if start_set & self._slippery_tiles:
            raise ValueError("Start position(s) cannot be placed on slippery tiles.")
        if goal_set & self._slippery_tiles:
            raise ValueError("Goal position(s) cannot be placed on slippery tiles.")
        if self._walls & self._slippery_tiles:
            raise ValueError("A tile cannot be both a wall and a slippery tile.")

    @property
    def agent_pos(self) -> Position:
        """Current agent position in ``(row, col)`` format."""

        return self._agent_pos

    @property
    def step_count(self) -> int:
        """Number of actions taken in the current episode."""

        return self._step_count

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset episode state and return the initial observation.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional reset options (unused in prototype).

        Returns:
            Tuple of initial observation and info dictionary.
        """

        super().reset(seed=seed)
        _ = options

        self._step_count = 0
        starts = self.config.start_positions
        if len(starts) > 1:
            assert self.np_random is not None
            self._agent_pos = starts[int(self.np_random.integers(0, len(starts)))]
        else:
            self._agent_pos = starts[0]

        observation = self._get_observation()
        info: dict[str, Any] = {
            "agent_pos": (self._agent_pos.row, self._agent_pos.col),
            "step_count": self._step_count,
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one environment step.

        Args:
            action: Action index in :class:`Action`.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """

        requested_action = Action(action)
        effective_action, slipped = self._sample_effective_action(requested_action)
        move = self._action_to_delta(effective_action)
        candidate = Position(self._agent_pos.row + move[0], self._agent_pos.col + move[1])

        moved = False
        if self._is_in_bounds(candidate) and candidate not in self._walls:
            self._agent_pos = candidate
            moved = True

        self._step_count += 1

        reward = self.config.step_penalty
        terminated = False

        if self._agent_pos in self._goal_positions:
            reward += self.config.goal_reward
            terminated = True
        elif self._agent_pos in self._hazards:
            reward += self.config.hazard_penalty
            terminated = True

        truncated = self._step_count >= self.config.max_steps

        observation = self._get_observation()
        info: dict[str, Any] = {
            "agent_pos": (self._agent_pos.row, self._agent_pos.col),
            "moved": moved,
            "requested_action": int(requested_action),
            "effective_action": int(effective_action),
            "slipped": slipped,
            "on_slippery_tile": self._agent_pos in self._slippery_tiles,
            "step_count": self._step_count,
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the current state into an RGB array.

        Returns:
            RGB image with shape ``(height, width, 3)``.
        """

        color_map: dict[CellType, tuple[int, int, int]] = {
            CellType.EMPTY: (245, 245, 245),
            CellType.WALL: (40, 40, 40),
            CellType.GOAL: (80, 200, 120),
            CellType.HAZARD: (220, 80, 80),
            CellType.START: (80, 120, 220),
            CellType.SLIPPERY: (90, 210, 240),
            CellType.AGENT: (245, 160, 45),
        }

        obs = self._get_observation()
        image = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

        for row in range(self.config.height):
            for col in range(self.config.width):
                cell_value = CellType(int(obs[row, col]))
                image[row, col] = np.array(color_map[cell_value], dtype=np.uint8)

        return image

    def get_static_layout(self) -> np.ndarray:
        """Return a grid with static objects only (without agent overlay)."""

        layout = np.full((self.config.height, self.config.width), int(CellType.EMPTY), dtype=np.int32)

        for pos in self._walls:
            layout[pos.row, pos.col] = int(CellType.WALL)

        for pos in self._hazards:
            layout[pos.row, pos.col] = int(CellType.HAZARD)

        for pos in self._slippery_tiles:
            layout[pos.row, pos.col] = int(CellType.SLIPPERY)

        for pos in self.config.start_positions:
            layout[pos.row, pos.col] = int(CellType.START)
        for pos in self.config.goal_positions:
            layout[pos.row, pos.col] = int(CellType.GOAL)
        return layout

    def _get_observation(self) -> np.ndarray:
        """Build and return the current integer-grid observation."""

        observation = self.get_static_layout()
        observation[self._agent_pos.row, self._agent_pos.col] = int(CellType.AGENT)
        return observation

    def _is_in_bounds(self, pos: Position) -> bool:
        """Check whether a position is inside the grid boundaries."""

        return 0 <= pos.row < self.config.height and 0 <= pos.col < self.config.width

    @staticmethod
    def _action_to_delta(action: Action) -> tuple[int, int]:
        """Map an action index to a grid coordinate delta."""

        deltas: dict[Action, tuple[int, int]] = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
            Action.STAY: (0, 0),
        }
        return deltas[action]

    def _sample_effective_action(self, requested_action: Action) -> tuple[Action, bool]:
        """Sample the action that will be executed after slippery-tile effects.

        Slipping is only applied when the agent starts the step on a slippery tile and
        the requested action is a movement action (not ``STAY``).
        """

        if requested_action == Action.STAY:
            return requested_action, False

        if self._agent_pos not in self._slippery_tiles:
            return requested_action, False

        assert self.np_random is not None
        if float(self.np_random.random()) >= self.config.slip_probability:
            return requested_action, False

        alternatives = [
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]
        alternatives.remove(requested_action)
        sampled_index = int(self.np_random.integers(low=0, high=len(alternatives)))
        return alternatives[sampled_index], True
