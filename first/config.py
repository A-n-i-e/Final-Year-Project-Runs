"""
config.py — Central configuration for Sawyer PPO pick-and-place.
Tune these values to balance training stability and sample efficiency.
"""

# ─── Environment ────────────────────────────────────────────────────────────
ENV_CONFIG = {
    "max_episode_steps": 200,       # Steps before forced reset
    "n_objects": 5,                 # Number of objects to pick and place
    "goal_threshold": 0.05,         # Distance (m) to count as "placed"
    "grasp_threshold": 0.03,        # Distance (m) to count as "grasped"
    "lift_height": 0.15,            # Required lift height (m)
    "render_mode": None,            # "human" for live view, None for headless
    "object_shapes": ["cube", "sphere", "cylinder", "box", "cone"],
}

# ─── PPO Hyperparameters ─────────────────────────────────────────────────────
# These defaults work well for continuous robotic control.
# Increase n_steps * n_envs if you see high variance in rewards.
PPO_CONFIG = {
    "learning_rate": 3e-4,          # Adam LR; reduce to 1e-4 if unstable
    "n_steps": 2048,                # Steps per rollout per env
    "batch_size": 64,               # Mini-batch size; must divide n_steps * n_envs
    "n_epochs": 10,                 # Gradient update passes per rollout
    "gamma": 0.99,                  # Discount factor
    "gae_lambda": 0.95,             # GAE smoothing (bias-variance tradeoff)
    "clip_range": 0.2,              # PPO clip epsilon
    "clip_range_vf": None,          # Value function clip (None = disabled)
    "ent_coef": 0.01,               # Entropy bonus (encourages exploration)
    "vf_coef": 0.5,                 # Value function loss coefficient
    "max_grad_norm": 0.5,           # Gradient clipping for stability
    "use_sde": True,                # State-Dependent Exploration (better for robotics)
    "sde_sample_freq": 4,           # Re-sample noise every N steps
    "policy_kwargs": {
        "net_arch": [256, 256],     # Shared MLP layers for actor + critic
        "log_std_init": -2,         # Initial log std for action distribution
    },
}

# ─── Training ────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "total_timesteps": 2000000,   # Total env steps to train
    "n_envs": 8,                    # Parallel environments (increase for faster data)
    "eval_freq": 20000,            # Evaluate every N steps
    "eval_episodes": 10,            # Episodes per evaluation run
    "save_freq": 50000,            # Checkpoint every N steps
    "log_dir": "./logs/",
    "model_dir": "./models/",
    "tensorboard_log": "./tb_logs/",
    "seed": 42,
}

# ─── Reward Shaping ──────────────────────────────────────────────────────────
# Each coefficient scales its reward component. 
# Start conservative; increase success bonus once reaching is reliable.
REWARD_CONFIG = {
    "reach_coeff": 1.0,             # Reward for moving EE toward object
    "grasp_coeff": 5.0,             # Reward for successfully grasping
    "lift_coeff": 3.0,              # Reward for lifting the object
    "place_coeff": 10.0,            # Reward for placing at goal
    "success_bonus": 50.0,          # Bonus per completed object
    "all_done_bonus": 100.0,        # Bonus for clearing all 5 objects
    "time_penalty": -0.01,          # Per-step penalty to encourage speed
    "drop_penalty": -2.0,           # Penalty for dropping a held object
    "collision_penalty": -1.0,      # Penalty for self-collision or table hit
}
