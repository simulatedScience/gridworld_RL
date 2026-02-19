"""Manual control runner for the GridWorld prototype.

Controls:
    - Arrow keys / WASD: move
    - Space: stay in place
    - R: reset episode
    - Esc or window close: quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from experiment_logging.logger import EpisodeSummary, JsonRunLogger
from environments.gridworld import GridWorldConfig, GridWorldEnv
from environments.renderer import GridWorldRenderer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with config path and fps.
    """

    parser = argparse.ArgumentParser(description="Run GridWorld in manual-control mode.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/env/default.json"),
        help="Path to environment JSON configuration.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Display frame rate.")
    return parser.parse_args()


def main(args = None) -> None:
    """Run the interactive manual-control loop."""

    args = parse_args() if args is None else args
    config = GridWorldConfig.from_json_file(args.config)
    env = GridWorldEnv(config=config)
    logger = JsonRunLogger(
        run_name="manual_control",
        output_dir="experiments/logs/manual_control",
        metadata={
            "config_path": str(args.config),
            "fps": args.fps,
            "slip_probability": config.slip_probability,
            "step_penalty": config.step_penalty,
            "goal_reward": config.goal_reward,
            "hazard_penalty": config.hazard_penalty,
            "max_steps": config.max_steps,
        },
    )

    observation, info = env.reset(seed=0)

    renderer = GridWorldRenderer(grid_shape=(config.height, config.width))
    renderer.initialize()

    clock = pygame.time.Clock()
    running = True
    episode_return = 0.0
    episode_index = 0
    slip_count = 0
    trajectory: list[tuple[int, int]] = []

    try:
        while running:
            hud = (
                f"steps={info.get('step_count', 0)}",
                f"return={episode_return:.3f}",
                f"slips={slip_count}",
                "R=reset, ESC=quit",
            )
            renderer.draw(observation, hud_lines=hud)

            for event in renderer.pump_events():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if event.key == pygame.K_r:
                    agent_pos_raw = info.get("agent_pos", (0, 0))
                    final_position = (int(agent_pos_raw[0]), int(agent_pos_raw[1]))
                    logger.log_episode(
                        EpisodeSummary(
                            episode_index=episode_index,
                            steps=int(info.get("step_count", 0)),
                            total_return=float(episode_return),
                            terminated=False,
                            truncated=False,
                            reached_goal=False,
                            hazard_hit=False,
                            slip_count=slip_count,
                            final_position=final_position,
                            trajectory=tuple(trajectory),
                        )
                    )
                    episode_index += 1
                    observation, info = env.reset()
                    episode_return = 0.0
                    slip_count = 0
                    trajectory = []
                    continue

                action = renderer.key_to_action(event.key)
                if action is None:
                    continue

                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                trajectory.append(
                    (
                        int(info.get("requested_action", action)),
                        int(info.get("effective_action", action)),
                    )
                )
                if bool(info.get("slipped", False)):
                    slip_count += 1

                if terminated or truncated:
                    _goal_tuples = {(p.row, p.col) for p in config.goal_positions}
                    reached_goal = info.get("agent_pos") in _goal_tuples
                    hazard_hit = bool(not reached_goal)
                    agent_pos_raw = info.get("agent_pos", (0, 0))
                    final_position = (int(agent_pos_raw[0]), int(agent_pos_raw[1]))
                    logger.log_episode(
                        EpisodeSummary(
                            episode_index=episode_index,
                            steps=int(info.get("step_count", 0)),
                            total_return=float(episode_return),
                            terminated=terminated,
                            truncated=truncated,
                            reached_goal=reached_goal,
                            hazard_hit=hazard_hit,
                            slip_count=slip_count,
                            final_position=final_position,
                            trajectory=tuple(trajectory),
                        )
                    )
                    episode_index += 1
                    observation, info = env.reset()
                    episode_return = 0.0
                    slip_count = 0
                    trajectory = []

            renderer.tick(clock, fps=args.fps)
    finally:
        logger.close()
        print(f"Saved logs to: {logger.log_path}")
        renderer.close()


if __name__ == "__main__":
    # in-file settings
    
    main()
