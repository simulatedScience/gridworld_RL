# Gridworld RL — Implementation Outline

## Contents
1. [Experimental Matrix](#1-experimental-matrix) — the headline goal
2. [Research Axes](#2-research-axes) — what we vary and why
3. [Implementation Phases](#3-implementation-phases) — build order
4. [Project Structure](#4-project-structure) — files and directories
5. [Module Notes](#5-module-notes) — key design decisions

---

## 1. Experimental Matrix

Each experiment run is one combination of choices from the five axes below.
Runs are fully specified by a config file in `configs/experiments/`.

| Axis | Options |
|---|---|
| **Algorithm** | DQN · PPO · SAC · DreamerV3 |
| **Backbone** | MLP · LSTM · GRU |
| **Exploration** | ε-greedy constant · ε-greedy decay · Softmax (τ) · Random |
| **Reset strategy** | Standard · Replay (fraction f) · Memory-preserve |
| **Reward function** | Sparse · Dense · Step penalty · Curiosity · Composite |

> The axes are independent — any combination is a valid experiment.  
> Start with DQN + MLP + ε-greedy decay + Standard + Sparse as the baseline.

---

## 2. Research Axes

### Algorithms
| | Algorithm | Type | Notes |
|---|---|---|---|
| 1 | **DQN** | off-policy, value-based | Baseline. Replay buffer, target network, double-DQN option. |
| 2 | **PPO** | on-policy, policy gradient | Strong general baseline. Clipped surrogate, GAE. |
| 3 | **SAC** | off-policy, actor-critic | Entropy regularization, handles stochastic policies well. |
| 4 | **DreamerV3** | model-based | World model + imagination rollouts. Most complex, highest potential. |

### Network Backbones
| Backbone | Memory | Notes |
|---|---|---|
| MLP | None | Feedforward; Markov assumption. |
| LSTM | Yes | Gated recurrence; better for partial observability and long-range deps. |
| GRU | Yes | Fewer parameters than LSTM; often similar performance. |

All backbones share the same `RNNEncoder` API (`reset_hidden()`, `detach_hidden()`), so agents are backbone-agnostic.

### Exploration Strategies
| Strategy | Key parameters | Notes |
|---|---|---|
| ε-greedy (constant) | `epsilon` | Simple baseline. |
| ε-greedy (decay) | `ε_start`, `ε_end`, `decay_type` ∈ {linear, exp, step}, `decay_steps` | Most practical default. |
| Softmax / stochastic | `temperature τ` | Samples ∝ softmax(logits / τ); exploits relative confidence. |
| Random | — | Pure random rollout; lower bound on exploration quality. |

Strategies are decoupled from agents — any strategy pairs with any agent.

### Reset Strategies
| Strategy | Env state | RNN hidden state | Notes |
|---|---|---|---|
| **Standard** | Start state | Full reset | Baseline. |
| **Replay** | Previously visited state | Full reset | Fraction f ∈ (0,1) of episodes use a state sampled from a ring buffer; sampling can be uniform or recency-weighted. |
| **Memory-preserve** | Start state | **No reset** | Tests whether persistent memory aids learning. |

### Reward Functions
| File | Description |
|---|---|
| `sparse.py` | +1 on goal, 0 otherwise |
| `dense.py` | Distance-based shaping |
| `step_penalty.py` | Small negative reward per step |
| `curiosity.py` | Exploration bonus (count-based or RND) |
| `composite.py` | Weighted sum of any of the above |

---

## 3. Implementation Phases

### Phase 1 — Core Infrastructure
- [ ] `gridworld.py` with Gym interface and object system
- [ ] `renderer.py` (Matplotlib static, Pygame live)
- [ ] `reward_functions/` base + sparse + dense
- [ ] `logger.py` + `metrics.py`

### Phase 2 — Agents & Networks
- [ ] `mlp.py`, `rnn.py`
- [ ] `dqn_agent.py`
- [ ] `ppo_agent.py`
- [ ] Exploration strategies (`epsilon_greedy.py`, `stochastic_policy.py`)
- [ ] `train.py` + `evaluate.py`

### Phase 3 — Advanced Agents
- [ ] `sac_agent.py`
- [ ] `dreamer_agent.py` + `world_model.py`

### Phase 4 — Reset Strategies & Experiments
- [ ] `replay_reset.py` + `memory_preserve_reset.py`
- [ ] Full experimental sweep
- [ ] `visualizations.py` (heatmaps, animated policy arrows)

### Phase 5 — Analysis & Documentation
- [ ] Comparative plots across all axes
- [ ] Hyperparameter sensitivity analysis
- [ ] Final write-up

---

## 4. Project Structure

```
gridworld_RL/
│
├── environments/
│   ├── __init__.py
│   ├── gridworld.py              # Core environment (gym-compatible)
│   │                             #   - grid layout, objects, agent position
│   │                             #   - save/load starting positions
│   │                             #   - parallelization support (sb3/sbx VecEnv)
│   ├── objects.py                # Modular special-object definitions
│   │                             #   (walls, goals, hazards, keys, doors, …)
│   ├── renderer.py               # Pygame / Matplotlib renderer
│   │                             #   - step-by-step and live-training views
│   └── reward_functions/
│       ├── __init__.py
│       ├── base.py               # Abstract RewardFunction interface
│       ├── sparse.py             # +1 on goal, 0 otherwise
│       ├── dense.py              # Distance-based shaping
│       ├── step_penalty.py       # Small penalty per step
│       ├── curiosity.py          # Exploration bonus (count-based / RND)
│       └── composite.py          # Weighted sum of multiple functions
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py             # Abstract Agent interface
│   │                             #   act(), train_step(), save(), load()
│   ├── dqn_agent.py              # Deep Q-Network
│   ├── ppo_agent.py              # Proximal Policy Optimization
│   ├── sac_agent.py              # Soft Actor-Critic
│   └── dreamer_agent.py          # DreamerV3 (world-model based)
│
├── networks/
│   ├── __init__.py
│   ├── mlp.py                    # Standard MLP encoder / heads
│   ├── rnn.py                    # LSTM and GRU wrappers
│   │                             #   - configurable hidden size & layers
│   │                             #   - hidden-state management API
│   └── world_model.py            # Recurrent State-Space Model (for Dreamer)
│                                 #   - encoder, RSSM, decoder, reward head
│
├── exploration/
│   ├── __init__.py
│   ├── base.py                   # Abstract ExplorationStrategy interface
│   ├── epsilon_greedy.py         # ε-greedy with constant or scheduled ε
│   │                             #   - linear / exponential / step decay
│   └── stochastic_policy.py      # Softmax sampling over Q-values / logits
│                                 #   (temperature parameter τ)
│
├── reset_strategies/
│   ├── __init__.py
│   ├── base.py                   # Abstract ResetStrategy interface
│   ├── standard_reset.py         # Always reset env + memory to start state
│   ├── replay_reset.py           # Reset to a previously visited state
│   │                             #   - configurable fraction of episodes
│   │                             #   - state buffer with sampling policy
│   └── memory_preserve_reset.py  # Reset env grid but keep RNN hidden states
│
├── training/
│   ├── train.py                  # Main training loop
│   │                             #   - wires env / agent / exploration / reset
│   │                             #   - calls logger at configurable intervals
│   ├── evaluate.py               # Evaluation / test script
│   │                             #   - greedy policy rollouts
│   │                             #   - computes all performance metrics
│   └── hyperparams.py            # Default hyperparameter sets per algorithm
│
├── logging/
│   ├── __init__.py
│   ├── logger.py                 # Experiment logger (TensorBoard / W&B / CSV)
│   ├── metrics.py                # Algorithm-agnostic performance metrics
│   │                             #   - steps to reach >threshold% success rate
│   │                             #   - episode length to goal
│   │                             #   - exploration / state-coverage rate
│   └── visualizations.py         # Post-hoc analysis plots
│                                 #   - policy-arrow overlays
│                                 #   - animated policy evolution (GIF/video)
│                                 #   - state-visitation heatmaps
│                                 #   - reward distribution maps
│
├── configs/
│   ├── env/                      # YAML/JSON environment configs
│   ├── agents/                   # Per-algorithm hyperparameter configs
│   └── experiments/              # Full experiment run configs
│                                 #   (env + agent + exploration + reset + seed)
│
├── experiments/                  # Saved results, logs, plots per run
├── checkpoints/                  # Model weight snapshots
├── tests/                        # Unit & integration tests
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_reset_strategies.py
│   └── test_metrics.py
│
├── implementation_outline.md     # This file
└── README.md
```

---

## 5. Module Notes

**`gridworld.py`** — Gym-compatible `step()` / `reset()` / `render()`; configurable grid, objects, agents; start-state snapshots; `VecEnv` wrapper for sb3/sbx.

**`renderer.py`** — Pygame for live training, Matplotlib for static analysis; receives a renderer-agnostic state dict.

**`reward_functions/`** — All functions are stateless and receive `(state, action, next_state, info)`. `composite.py` takes a weighted sum. Plugged in via config at env creation.

**`rnn.py`** — Unified `RNNEncoder` switchable between LSTM and GRU. Exposes `reset_hidden()` / `detach_hidden()`. Used by all recurrent agents and by the Dreamer RSSM.

**`replay_reset.py`** — Maintains a ring buffer of `(env_snapshot, step)` tuples. Sampling policy: uniform or recency-weighted.

**`visualizations.py`** — Policy-arrow overlays; animated policy evolution (GIF/video); state-visitation heatmaps; reward distribution maps.
