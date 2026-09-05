# Stalingrad 1942 — RL Architecture & Training Reference

Technical reference for the Reinforcement Learning system, environment encodings,
neural network models, CleanRL training pipeline, and web management hub.

---

## 1. Environment & State Representation (`rl/stalingrad_env.py`)

The environment cleanly wraps the pure Python `Game` engine. Observations are
canonical spatial tensors of shape `(15, 10, 12)` preserving the 2D grid:

| Channel | Contents | Values |
|---|---|---|
| `0` | Ruins terrain (`#`) | `1.0` if ruin, `0.0` otherwise |
| `1` | Volga river impassable tiles (`~`) | `1.0` if river, `0.0` otherwise |
| `2` | Ferry crossing tiles (`F`) | `1.0` if ferry, `0.0` otherwise |
| `3` | Objective locations (5 fixed tiles) | `1.0` on objective tile, `0.0` elsewhere |
| `4` | Sticky objective control | `+1.0` (friendly), `-1.0` (enemy), `0.0` (neutral) |
| `5` | Friendly Rifle Squads | Normalized HP (`hp / 5.0`) at unit tile |
| `6` | Friendly Sniper Teams | Normalized HP (`hp / 4.0`) at unit tile |
| `7` | Friendly Tanks | Normalized HP (`hp / 6.0`) at unit tile |
| `8` | Enemy Rifle Squads | Normalized HP (`hp / 5.0`) at unit tile |
| `9` | Enemy Sniper Teams | Normalized HP (`hp / 4.0`) at unit tile |
| `10` | Enemy Tanks | Normalized HP (`hp / 6.0`) at unit tile |
| `11` | Friendly move readiness | `1.0` at tile if alive and not `moved` |
| `12` | Friendly attack readiness | `1.0` at tile if alive and not `attacked` |
| `13` | Turn progression | Global float: `round / 12.0` |
| `14` | Perspective indicator | `1.0` for Axis, `0.0` for Soviet |

---

## 2. Action Space & Invalid Action Masking

To avoid combinatorial explosion across multi-unit turns, each step represents a
single micro-action from an 897-element discrete space:

$$\text{Action Space Size} = 1 + (\text{NUM\_UNIT\_SLOTS} \times 128) = 1 + (7 \times 128) = 897$$

* **Action `0`**: `END_TURN` (always legal, mask = 1).
* **For friendly unit slot $u \in [0..6]$**:
  * Offset base: $1 + u \times 128$
  * `base + 0`: **Pass / Hold Unit** (marks unit done for turn).
  * `base + 1 .. base + 120`: **Move Unit** to tile $(x, y)$ ($tile\_idx = y \times 12 + x$).
  * `base + 121 .. base + 127`: **Attack Enemy** slot $e \in [0..6]$.

### Action Masking
`get_action_mask(game, team)` queries `game.legal_moves()` and `game.attackable()`.
Illegal actions receive $-\infty$ logits inside the PyTorch `CategoricalMasked`
distribution, guaranteeing zero invalid action attempts during sampling.

---

## 3. Actor-Critic Neural Network (`rl/models.py`)

* **Input**: `(batch_size, 15, 10, 12)`
* **Backbone**:
  * Initial Conv: $15 \rightarrow 64$ channels ($3\times 3$, padding 1) + ReLU
  * Residual Block 1: 2 conv layers ($64 \rightarrow 64$, $3\times 3$, residual skip)
  * Residual Block 2: 2 conv layers ($64 \rightarrow 64$, $3\times 3$, residual skip)
  * Flattened Dense: $64 \times 10 \times 12 = 7680 \rightarrow 256$ + ReLU
* **Heads**:
  * **Actor**: Linear($256 \rightarrow 897$) masked policy logits
  * **Critic**: Linear($256 \rightarrow 1$) scalar value estimate $V(s)$

---

## 4. Training Pipeline (`rl/train_ppo.py`)

* **Framework**: CleanRL-style single-file Maskable PPO.
* **Algorithm**: Actor-Critic with Generalized Advantage Estimation ($\lambda=0.95, \gamma=0.99$).
* **Self-Play**: Shared network evaluates from the perspective of the active team.
* **Evaluation**: Evaluates 10 games vs the heuristic AI in `ai.py` every $N$ steps.
* **Telemetry**: Atomically writes live progress, SPS, losses, and win rate to `checkpoints/train_status.json`.

---

## 5. Web Training Manager (`rl/train_manager.py`)

* **Process Isolation**: Launches `train_ppo.py` as an independent background subprocess using `C:\Anaconda\envs\stalingrad-rl\python.exe`.
* **Non-Blocking**: Keeps Flask and frontend UI responsive at all times.
* **Model Serving**: Caches the active `RLAgent` instance for zero-latency in-game moves.

### HTTP Endpoints
* `GET /api/training/status`: Returns current status, step, SPS, losses, available checkpoints, and checkpoint_steps completed.
* `POST /api/training/start`: Starts training with `{ total_timesteps, num_envs, learning_rate }`.
* `POST /api/training/stop`: Gracefully terminates the running process.
* `POST /api/training/select_model`: Selects active checkpoint for gameplay `{ model }`.
* `POST /api/set_ai_type`: Configures `{ axis: "heuristic"|"rl", soviet: "heuristic"|"rl" }`.
