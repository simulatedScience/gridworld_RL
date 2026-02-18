"""Smoke tests for the Stage-1 GridWorld environment."""

from __future__ import annotations

import unittest

import numpy as np

from environments.gridworld import Action, GridWorldConfig, GridWorldEnv
from environments.objects import Position


class TestGridWorldEnv(unittest.TestCase):
    """Validate basic Gymnasium behavior for the prototype environment."""

    def setUp(self) -> None:
        """Create a simple deterministic test environment."""

        self.config = GridWorldConfig(
            width=5,
            height=4,
            start_positions=(Position(1, 1),),
            goal_positions=(Position(1, 3),),
            walls=(Position(0, 1),),
            hazards=(Position(3, 3),),
            slippery_tiles=(Position(1, 2),),
            slip_probability=0.0,
            max_steps=20,
        )
        self.env = GridWorldEnv(config=self.config)

    def test_reset_returns_valid_observation(self) -> None:
        """Reset returns a correctly-shaped integer observation grid."""

        observation, info = self.env.reset(seed=42)

        self.assertEqual(observation.shape, (4, 5))
        self.assertEqual(observation.dtype, np.int32)
        self.assertIn("agent_pos", info)
        self.assertIn("step_count", info)
        self.assertEqual(info["step_count"], 0)

    def test_wall_blocks_movement(self) -> None:
        """Moving into a wall should keep the agent in place."""

        self.env.reset(seed=0)
        observation, reward, terminated, truncated, info = self.env.step(Action.UP)

        _ = observation
        self.assertFalse(info["moved"])
        self.assertEqual(info["agent_pos"], (1, 1))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, 0.0)

    def test_reaching_goal_terminates_episode(self) -> None:
        """Stepping onto the goal should terminate the episode with positive reward."""

        self.env.reset(seed=0)
        self.env.step(Action.RIGHT)
        observation, reward, terminated, truncated, info = self.env.step(Action.RIGHT)

        _ = observation
        _ = info
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertGreater(reward, 0.0)

    def test_slippery_tile_can_change_action(self) -> None:
        """On slippery tiles with p=1, movement action should be redirected."""

        config = GridWorldConfig(
            width=6,
            height=4,
            start_positions=(Position(1, 1),),
            goal_positions=(Position(1, 5),),
            slippery_tiles=(Position(1, 2),),
            slip_probability=1.0,
            max_steps=20,
        )
        env = GridWorldEnv(config=config)

        env.reset(seed=123)
        env.step(Action.RIGHT)
        _, _, _, _, info = env.step(Action.RIGHT)

        self.assertTrue(bool(info["slipped"]))
        self.assertEqual(int(info["requested_action"]), int(Action.RIGHT))
        self.assertNotEqual(int(info["effective_action"]), int(Action.RIGHT))


if __name__ == "__main__":
    unittest.main()
