# Task-Space RL for Robotic Manipulation

A reinforcement learning system for robotic pick-and-lift manipulation using the Franka Emika Panda arm in MuJoCo simulation. This project trains and compares three RL algorithms — **PPO**, **SAC**, and **TD3** — on a 5-object pick-and-lift task using task-space (Cartesian) control with Inverse Kinematics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Environment](#environment)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward Function](#reward-function)
  - [Curriculum Learning](#curriculum-learning)
- [Training](#training)
- [Visualisation](#visualisation)
- [Evaluation](#evaluation)
- [Algorithms](#algorithms)
- [Design Decisions](#design-decisions)
- [Results](#results)

---

## Project Overview

This project investigates whether task-space control (where the agent outputs Cartesian end-effector movements rather than raw joint angles) improves the sample efficiency and success rate of RL-based robotic manipulation. A custom MuJoCo environment is built around the Franka Emika Panda arm from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), and three algorithms are compared:

- **PPO** (Proximal Policy Optimisation) — on-policy
- **SAC** (Soft Actor-Critic) — off-policy
- **TD3** (Twin Delayed DDPG) — off-policy

The agent must pick up one of five objects (cube, sphere, cylinder, box, cone) from a table and lift it above a height threshold, maintaining the lift for 10 consecutive steps to count as a success.

---

## Folder Structure

```
Final-Year-Project-Runs/
│
├── envs/
│   └── panda_pick_env.py        # Custom Gymnasium environment
│
├── franka_emika_panda/          # MuJoCo Menagerie Panda model
│   ├── scene.xml                # Main scene file (includes panda.xml)
│   ├── panda.xml                # Panda arm + hand (updated with finger pads)
│   └── assets/                  # Mesh files (.stl, .obj)
│
├── logs/                        # TensorBoard logs (auto-created)
│   ├── panda_ppo/
│   ├── panda_sac/
│   └── panda_td3/
│
├── models/                      # Saved models (auto-created)
│   ├── panda_ppo/
│   │   ├── best_model.zip
│   │   ├── final_model.zip
│   │   └── vec_normalize.pkl
│   ├── panda_sac/
│   └── panda_td3/
│
├── results/                     # Evaluation outputs (auto-created)
│   ├── metrics_summary.csv
│   ├── graph1_training_reward.png
│   ├── graph2_success_rate.png
│   ├── graph3_episode_length.png
│   └── graph4_learning_curve.png
│
├── inverse_kinematics.py        # IK solver (taken from Google Deepmind's dm_control repository)
├── train.py                     # Training script (PPO / SAC / TD3)
├── visualise.py                 # Load trained model and render
├── evaluate.py                  # Compare algorithms, generate graphs + CSV
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python 3.10+
- Windows / Linux / macOS
- Anaconda/ Miniconda (recommended)

### Steps

```bash
# 1. Clone or download this repository
cd Final-Year-Project-Runs

# 2. Create a conda environment
conda create -n my_env python=3.10
conda activate my+env

# 3. Install dependencies
pip install -r requirements.txt
```


### Verify installation

```bash
python envs/panda_pick_env.py
```

You should see a sanity check run 20 steps and print obs shape `(42,)` with no errors.

---

## Environment

**File:** `envs/panda_pick_env.py`

The environment wraps the Franka Emika Panda arm + Franka Hand from MuJoCo Menagerie. At each episode, one of five objects is randomly placed on a table in front of the robot. The agent must reach, grasp, and lift the object above the lift height threshold for 10 consecutive steps.

### Observation Space

42-dimensional privileged state vector (no camera required):

| Index | Size | Description |
|-------|------|-------------|
| 0:7 | 7 | Arm joint positions (rad) |
| 7:14 | 7 | Arm joint velocities (rad/s) |
| 14:15 | 1 | Gripper jaw separation from `qpos` (0=closed, 1=open) |
| 15:16 | 1 | Gripper openness from `qpos` (0=closed, 1=open) |
| 16:19 | 3 | End-effector position (x, y, z) |
| 19:22 | 3 | Object position (x, y, z) |
| 22:25 | 3 | Relative vector: `obj_pos − ee_pos` |
| 25:29 | 4 | Object quaternion (w, qx, qy, qz) |
| 29:32 | 3 | Target lift position (x, y, z) |
| 32:37 | 5 | Object type one-hot (cube/sphere/cylinder/box/cone) |
| 37:38 | 1 | `is_grasped` flag (contact-based) |
| 38:39 | 1 | Curriculum stage (0–4) |
| 39:41 | 2 | Individual finger contact flags (left, right) |
| 41:42 | 1 | Height progress [0, 1] |

> **Privileged observation**: Object position, quaternion, and target position are read directly from the simulator. On a real robot, these would require perception (camera + pose estimation). This approach is standard in simulation-to-real research — train with privileged info, then distil to sensor-based inputs.

### Action Space

4-dimensional continuous action in `[-1, 1]`:

| Index | Meaning | Effect |
|-------|---------|--------|
| 0 | dx | Move EE left/right (±5 cm/step) |
| 1 | dy | Move EE forward/back (±5 cm/step) |
| 2 | dz | Move EE up/down (±5 cm/step) |
| 3 | gripper | +1 = open, −1 = close |

The agent outputs Cartesian displacements, not joint angles. An **Inverse Kinematics (IK)** solver converts the target EE position into 7 joint angle targets, which are sent to MuJoCo's position actuators.

### Reward Function

Reward function method was adopted from [ggando's github repo](https://github.com/ggand0/pick-101) on Rl training for SO-101 robot manipulation 

Dense shaped reward guiding the agent through four phases:
| Phase | Component | Value |
|-------|-----------|-------|
| Reach | `1 − tanh(10 × dist(ee, obj))` | [0, 1] |
| Reach | Push-down penalty if `obj_z < table + 5mm` | `−depth × 50` |
| Grasp | Gripper-close shaping (proximity-weighted) | [0, 0.15] |
| Grasp | Per-step grasp bonus (contact-based) | +0.25 |
| Grasp | Drop penalty (grasp lost this step) | −2.0 |
| Lift | Continuous lift progress (gated on grasp) | [0, 3.0] |
| Lift | Binary lift bonus (`obj_z > table + 2cm`) | +1.0 |
| Lift | Target height bonus (`obj_z ≥ threshold`) | +1.0 |
| Lift | Target pull toward goal (gated on grasp) | ≤ 0 |
| All | Action rate penalty: `−0.01 × ‖Δaction‖²` | ≤ 0 |
| **Success** | Held lift for `hold_steps=10` steps | **+10.0** |
| Failure | Object fell off table or arm wandered >80cm | −50.0 |

> Key design: success requires **holding** the lift for 10 consecutive steps, not just momentarily crossing the threshold. This prevents lucky collision bounces from being counted as successes.

### Curriculum Learning

Five-stage reverse curriculum (starts easy, works toward the full task):

| Stage | Description | EE Start | Gripper |
|-------|-------------|----------|---------|
| 4 | Object pre-grasped at half lift height | At lift | Closed |
| 3 | EE at grasp height, fingers on object | At object | Closing |
| 2 | EE at grasp height, gripper open | At object | Open |
| 1 | EE above object, gripper open | 5cm above | Open |
| 0 | Full task from home pose | Home | Open |

The agent starts at stage 4 (easiest). When rolling success rate exceeds **70% over 20 episodes**, it advances to the next harder stage. This is handled automatically — no manual intervention needed.

---

## Training

**File:** `train.py`

### Commands

```bash
# Train PPO (default)
python train.py --algo PPO --timesteps 1000000 --exp-name panda_ppo

# Train SAC (off-policy — use n-envs 1)
python train.py --algo SAC --timesteps 1000000 --exp-name panda_sac --n-envs 1

# Train TD3 (off-policy — use n-envs 1)
python train.py --algo TD3 --timesteps 1000000 --exp-name panda_td3 --n-envs 1
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | `PPO` | Algorithm: `PPO`, `SAC`, or `TD3` |
| `--timesteps` | `500000` | Total environment steps |
| `--n-envs` | `4` | Parallel environments (use `1` for SAC/TD3) |
| `--exp-name` | `panda_ppo` | Name for logs and model output folders |
| `--seed` | `42` | Random seed |
| `--no-normalize` | `False` | Disable VecNormalize if set |

### Outputs

```
logs/<exp_name>/          ← TensorBoard logs
models/<exp_name>/
  ├── best_model.zip      ← best checkpoint by eval reward
  ├── final_model.zip     ← model at end of training
  └── vec_normalize.pkl   ← normalisation statistics
```

### Monitor training

```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

---

## Visualisation

**File:** `visualise.py`

Loads a trained model and renders it in the MuJoCo interactive viewer.

```bash
# Standard
python visualise.py --model models/panda_ppo/best_model.zip \
                    --stats models/panda_ppo/vec_normalize.pkl

# Fix the object type
python visualise.py --model models/panda_ppo/best_model.zip \
                    --stats models/panda_ppo/vec_normalize.pkl \
                    --object cube

# Run exactly 10 episodes
python visualise.py --model models/panda_ppo/best_model.zip \
                    --stats models/panda_ppo/vec_normalize.pkl \
                    --episodes 10
```

The terminal prints a results line per episode:

```
Episode    1  |  object=cube       |  reward= +42.31  |  steps= 134  |  success=✓
Episode    2  |  object=sphere     |  reward=  -1.20  |  steps= 200  |  success=✗
```

---

## Evaluation

**File:** `evaluate.py`

Evaluates all three trained models and generates comparison graphs and a CSV summary.

```bash
# Evaluate all three algorithms (30 episodes each)
python evaluate.py

# More episodes for more accurate metrics
python evaluate.py --n-episodes 50

# Evaluate specific algorithms only
python evaluate.py --algos PPO SAC
```

### Outputs

| File | Description |
|------|-------------|
| `results/metrics_summary.csv` | Full metrics table for all algorithms |
| `results/graph1_training_reward.png` | Training reward vs timesteps (with std bands) |
| `results/graph2_success_rate.png` | Overall + per-object success rate bar charts |
| `results/graph3_episode_length.png` | Average episode length comparison |
| `results/graph4_learning_curve.png` | Success rate vs training timesteps |

### Metrics

| # | Metric | How measured |
|---|--------|-------------|
| 1 | Mean episode reward | Average total reward across eval episodes |
| 2 | Success rate | % of episodes where lift was held ≥10 steps |
| 3 | Average episode length | Mean steps per episode (lower = faster) |
| 4 | Learning speed | Timestep of first consistent success (from `evaluations.npz`) |
| 5 | Stability | Std deviation of episode rewards |

---

## Algorithms

### PPO — Proximal Policy Optimisation

On-policy algorithm. Collects rollouts across 4 parallel environments, then updates the policy in batches. Uses a clipping mechanism (`clip_range=0.2`) to prevent destructively large updates.

- Naturally handles continuous action spaces
- Action is output directly — no discretisation needed
- Slower sample efficiency but very stable training

### SAC — Soft Actor-Critic

Off-policy algorithm with a replay buffer. Maximises both reward and entropy (randomness) simultaneously, which encourages thorough exploration.

- Typically fastest to learn on manipulation tasks
- Automatic entropy coefficient tuning (`ent_coef='auto'`)

### TD3 — Twin Delayed DDPG

Off-policy algorithm. Uses two critic networks (to reduce overestimation bias) and delayed policy updates. Adds noise during training for exploration.

- More conservative than SAC, often more stable
- Sensitive to hyperparameters

---

## Design Decisions

### Why task-space control?

Pick-and-lift is fundamentally a Cartesian task — the agent needs to move a point in 3D space to another point. Training PPO to learn raw joint-space control requires learning the full forward kinematics mapping, which is nonlinear and varies with arm configuration. Task-space control reduces the action space from 8D (7 joints + gripper) to 4D (dx, dy, dz, gripper), making exploration far more efficient.

### Why IK and not Jacobian?

The Jacobian pseudoinverse is an approximation that becomes unstable near kinematic singularities. Full iterative IK (used here, from the `inverse_kinematics.py` solver) finds exact solutions and handles edge cases more robustly. The tradeoff is slightly higher compute per step, which is negligible at training scale.

### Why privileged observation?

Using ground-truth object position and orientation from the simulator removes the need for a perception system (camera + pose estimation), which is a separate research problem. This approach is standard in manipulation RL research: train a strong policy with privileged information, then distil it to sensor observations if needed.

### Why contact-based grasp detection?

Simple proximity-based grasping (EE within X cm of object) is easily exploited — the agent learns to hover near the object without actually grasping. Using MuJoCo contact data to require simultaneous contact from both finger pads ensures the object is physically between the fingers before any grasp bonus is awarded.

---

## Results

*Results will be updated after training completes.*

| Algorithm | Success Rate | Mean Reward | Learning Speed | Stability |
|-----------|-------------|-------------|----------------|-----------|
| PPO | — | — | — | — |
| SAC | — | — | — | — |
| TD3 | — | — | — | — |

Training curves and per-object success breakdowns are available in `results/` after running `evaluate.py`.

---

## References

- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- JericLew/Push_MuJoCo: https://github.com/JericLew/Push_MuJoCo
- Google Deepmind's dm_control (Inverse Kinematics Implementation): https://github.com/DLR-RM/stable-baselines3
- ggand0/pick-101: https://github.com/ggand0/pick-101
- ggand0 blog (which I found first before his GitHub):https://ggando.com/blog/so101-rl-lift/ 