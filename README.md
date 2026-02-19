# Plan
* create a customizable gridworld RL environment
  * save starting positions
  * modular reward functions
  * easily add new special objects
  * visualization
  * parallelization support for sb3 or sbx

* create framework to document and compare agents and reward functions
  * extensive logging of agent behaviour and performance
  * performance measures independent of agents 
    * training steps to reach >`threshold` % success rate
    * number of steps in episode to reach goal
    * rate of exploration
  * easily modify and document agent architecture and hyperparameters
  * visualization of agent behaviour and its evolution during training
    * arrows showing the policy, animated to show policy evolution during training
    * heatmaps of state visitation frequency, reward distribution, etc.

I am particularly interested in exploring ways to make exploration more efficient.

## Experimental Goals

We want to investigate how various design decisions affect the learning process and performance of RL agents in our gridworld environment. Options to test include:

| **Algorithm** | **Backbone** | **Exploration** | **Reset strategy** | **Reward function** |
|---|---|---|---|---|
| DQN | MLP  | ε-greedy constant | Standard | Sparse |
| PPO | LSTM | ε-greedy decay    | Replay (fraction f) | Dense |
| SAC | GRU  | Softmax (τ)       | Memory-preserve | Step penalty |
| DreamerV3  | | | Standard | Curiosity |
| | | | | Composite |

See [implementation_outline.md](implementation_outline.md) for the full plan.

## Environment Prototype

Current prototype uses a `gymnasium` environment in `environments/gridworld.py` and a `pygame` manual controller in `manual_control.py`.

### Core Elements

- **Cell types:** Empty, Start, Goal, Agent, Wall, Hazard, Slippery
- **Actions:** Up, Down, Left, Right, Stay
- **Episode end:** reaching goal, stepping on hazard, or `max_steps` truncation
- **Observation:** integer grid (`np.int32`) with one value per tile

### Slippery Tile Mechanics

- Slippery tiles are configured via `slippery_tiles` in the JSON config.
- If the agent starts a step on a slippery tile, the requested movement action is replaced with a random alternative with probability `slip_probability`.
- Slip metadata is returned in `info`:
  - `requested_action`
  - `effective_action`
  - `slipped`
  - `on_slippery_tile`

### Default Map

`configs/env/default.json` is a 22×16 map containing three goals:
- the closest goal is at the end of a narrow path
- the second closest goal is in a more open area accessible via slippery tiles
- the third goal is farthest away but reachable via a wide corridor without slippery tiles but a few hazards along the sides

### Manual Controller

Run the environment with keyboard controls:

```powershell
.\.venv\Scripts\python.exe manual_control.py --config configs/env/default.json
```

Controls:

- Arrow keys / WASD: move
- Space: stay
- R: reset current episode
- ESC: quit

### Logging

- Manual runs are logged to JSONL files under `experiments/logs/`.
- Logger module: `experiment_logging/logger.py`.
- A run writes `run_started`, `episode_finished`, and `run_finished` events.
