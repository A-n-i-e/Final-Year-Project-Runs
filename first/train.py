"""
train.py — Main PPO training entry point.

Run:
    python train.py                     # Full training
    python train.py --resume ./models/  # Resume from checkpoint
    python train.py --debug             # 1 env, fewer steps, verbose

Best practices implemented:
  - SubprocVecEnv for true parallelism (separate processes, no GIL)
  - VecNormalize to normalize observations and returns (critical for stability)
  - EvalCallback for periodic model saving based on best eval reward
  - TensorBoard logging for all metrics
  - Seeded randomness for reproducibility
"""

import os
import sys
import argparse
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecNormalize, DummyVecEnv
)
from stable_baselines3.common.callbacks import (
    CallbackList, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from first.envs.sawyer_env import SawyerPickPlaceEnv
from first.callbacks.training_callbacks import (
    SuccessRateCallback, RewardPlotCallback, CurriculumCallback
)
from first.config import PPO_CONFIG, TRAIN_CONFIG, ENV_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sawyer PPO pick-and-place")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 env, fewer steps")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (start with 1 object)")
    parser.add_argument("--total-steps", type=int,
                        default=TRAIN_CONFIG["total_timesteps"])
    return parser.parse_args()


def make_env(rank: int, seed: int = 0, render: bool = False):
    """Factory function for creating a monitored env (one per parallel worker)."""
    def _init():
        env = SawyerPickPlaceEnv(
            render_mode="human" if render else None,
            n_objects=1 if args.curriculum else ENV_CONFIG["n_objects"],
            max_episode_steps=ENV_CONFIG["max_episode_steps"],
        )
        env = Monitor(env, filename=os.path.join(TRAIN_CONFIG["log_dir"], f"env_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


def build_callbacks(eval_env, model_dir: str, log_dir: str, use_curriculum: bool):
    """Assemble the callback list for training."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Periodically evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=TRAIN_CONFIG["eval_freq"],
        n_eval_episodes=TRAIN_CONFIG["eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Save checkpoints so you can resume or roll back
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAIN_CONFIG["save_freq"],
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="sawyer_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,  # Save normalization stats too!
        verbose=1,
    )

    # Custom metrics logging
    success_callback = SuccessRateCallback(eval_freq=5000, verbose=1)
    plot_callback = RewardPlotCallback(plot_freq=20_000, log_dir=log_dir, verbose=1)

    callbacks = [eval_callback, checkpoint_callback, success_callback, plot_callback]

    if use_curriculum:
        callbacks.append(CurriculumCallback(success_threshold=0.8, verbose=1))

    return CallbackList(callbacks)


def main():
    global args
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed = TRAIN_CONFIG["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_envs = 1 if args.debug else TRAIN_CONFIG["n_envs"]
    total_steps = 50_000 if args.debug else args.total_steps

    print(f"╔══════════════════════════════════════╗")
    print(f"║    Sawyer PPO Pick-and-Place         ║")
    print(f"║    Envs: {n_envs:<5}  Steps: {total_steps:>10,}  ║")
    print(f"╚══════════════════════════════════════╝\n")

    # ── Vectorised training environments ─────────────────────────────────────
    # SubprocVecEnv = each env in its own OS process (bypasses Python GIL)
    # DummyVecEnv   = all envs in one process (simpler, good for debugging)
    VecEnvClass = DummyVecEnv if args.debug else SubprocVecEnv
    train_env = VecEnvClass([make_env(i, seed) for i in range(n_envs)])

    # VecNormalize: running mean/std normalization for obs and rewards.
    # CRITICAL: this dramatically improves PPO stability for continuous control.
    train_env = VecNormalize(
        train_env,
        norm_obs=True,          # Normalize observations
        norm_reward=True,       # Normalize rewards (use gamma=0.99)
        clip_obs=10.0,          # Clip normalized obs to [-10, 10]
        clip_reward=10.0,       # Clip normalized rewards
        gamma=PPO_CONFIG["gamma"],
    )

    # ── Evaluation environment (single, unnormalized for true metrics) ───────
    eval_env = DummyVecEnv([make_env(0, seed + 1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)  # Don't update stats during eval

    # ── Build or load model ──────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            device="auto",
            tensorboard_log=TRAIN_CONFIG["tensorboard_log"],
        )
        # Also load normalization stats
        vecnorm_path = args.resume.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            train_env = VecNormalize.load(vecnorm_path, train_env)
            print(f"Loaded VecNormalize stats from {vecnorm_path}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            **PPO_CONFIG,
            verbose=1,
            seed=seed,
            device="auto",
            tensorboard_log=TRAIN_CONFIG["tensorboard_log"],
        )

    print(f"\nPolicy network:\n{model.policy}\n")
    print(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}\n")

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = build_callbacks(
        eval_env=eval_env,
        model_dir=TRAIN_CONFIG["model_dir"],
        log_dir=TRAIN_CONFIG["log_dir"],
        use_curriculum=args.curriculum,
    )

    # ── Training ─────────────────────────────────────────────────────────────
    print("Starting training... Monitor with:")
    print(f"  tensorboard --logdir {TRAIN_CONFIG['tensorboard_log']}\n")

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            reset_num_timesteps=not bool(args.resume),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving checkpoint...")

    # ── Save final model ─────────────────────────────────────────────────────
    final_path = os.path.join(TRAIN_CONFIG["model_dir"], "sawyer_ppo_final")
    model.save(final_path)
    train_env.save(final_path + "_vecnormalize.pkl")
    print(f"\nSaved final model → {final_path}.zip")
    print(f"Saved normalization stats → {final_path}_vecnormalize.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
