"""
train.py
========
Train a PPO agent on the Panda pick-and-lift environment using
Stable Baselines3.

Usage
-----
    python train.py                        # train with defaults
    python train.py --timesteps 1000000    # longer run
    python train.py --exp-name my_run      # custom experiment name

Outputs
-------
    logs/<exp_name>/                       # TensorBoard logs
    models/<exp_name>/best_model.zip       # best checkpoint (by mean reward)
    models/<exp_name>/final_model.zip      # model at end of training
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from envs.panda_pick_env import PandaPickEnv


# ---------------------------------------------------------------------------
# Custom callback: print a short progress line every N rollouts
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    def __init__(self, print_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.print_every:
            self._last_print = self.num_timesteps

            # Pull recent episode stats from the rollout buffer infos
            ep_rews = self.model.ep_info_buffer
            if ep_rews:
                mean_rew = np.mean([ep["r"] for ep in ep_rews])
                mean_len = np.mean([ep["l"] for ep in ep_rews])
                print(
                    f"  timesteps={self.num_timesteps:>8,} | "
                    f"mean_reward={mean_rew:+.2f} | "
                    f"mean_ep_len={mean_len:.0f}"
                )
        return True   # returning False would stop training


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on PandaPickEnv")
    parser.add_argument("--timesteps",   type=int,   default=500_000,
                        help="Total environment timesteps to train for")
    parser.add_argument("--n-envs",      type=int,   default=4,
                        help="Number of parallel environments")
    parser.add_argument("--exp-name",    type=str,   default="panda_ppo",
                        help="Experiment name (used for log/model dirs)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no-normalize",action="store_true",
                        help="Disable VecNormalize (obs + reward normalisation)")
    return parser.parse_args()


def main():
    args = parse_args()

    log_dir   = os.path.join("logs",   args.exp_name)
    model_dir = os.path.join("models", args.exp_name)
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Experiment : {args.exp_name}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  Envs       : {args.n_envs}")
    print(f"  Seed       : {args.seed}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Training environment  (vectorised, optionally normalised)
    # ------------------------------------------------------------------
    train_env = make_vec_env(
        PandaPickEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"max_episode_steps": 200, "render_mode": None},
    )

    if not args.no_normalize:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

    # ------------------------------------------------------------------
    # Evaluation environment  (single env, same normalisation stats)
    # ------------------------------------------------------------------
    eval_env = make_vec_env(
        PandaPickEnv,
        n_envs=1,
        seed=args.seed + 999,
        env_kwargs={"max_episode_steps": 200, "render_mode": None},
    )

    if not args.no_normalize:
        # Share the running stats from the training env
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,   # don't normalise rewards during eval
            clip_obs=10.0,
            training=False,      # don't update stats during eval
        )
        # Copy current stats over (they'll be updated as training proceeds
        # via the EvalCallback syncing the vec_normalize wrapper)
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms

    # ------------------------------------------------------------------
    # PPO model
    # ------------------------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,           # steps per env per rollout
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # small entropy bonus — helps exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        verbose=0,
        seed=args.seed,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=model_dir,
        name_prefix="checkpoint",
    )

    progress_callback = ProgressCallback(print_every=10_000)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\nStarting training...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback],
        tb_log_name=args.exp_name,
        reset_num_timesteps=True,
    )

    # ------------------------------------------------------------------
    # Save final model (and normalisation stats if used)
    # ------------------------------------------------------------------
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")

    if not args.no_normalize:
        stats_path = os.path.join(model_dir, "vec_normalize.pkl")
        train_env.save(stats_path)
        print(f"Normalisation stats saved → {stats_path}")

    train_env.close()
    eval_env.close()

    print("\nDone! To view training curves:")
    print(f"  tensorboard --logdir {log_dir}")
    print("\nTo visualise the trained agent:")
    print(f"  python visualise.py --model {model_dir}/best_model.zip")
    if not args.no_normalize:
        print(f"             --stats {model_dir}/vec_normalize.pkl")


if __name__ == "__main__":
    main()