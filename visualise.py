"""
visualise.py
============
Load a trained PPO model and watch it run in the MuJoCo interactive viewer.

Usage
-----
    # With VecNormalize stats (if you trained with normalisation on)
    python visualise.py --model models/panda_ppo/best_model.zip \
                        --stats models/panda_ppo/vec_normalize.pkl

    # Without normalisation stats
    python visualise.py --model models/panda_ppo/best_model.zip

    # Fix the object type instead of randomising each episode
    python visualise.py --model models/panda_ppo/best_model.zip \
                        --object cube

    # Run a fixed number of episodes then exit
    python visualise.py --model models/panda_ppo/best_model.zip \
                        --episodes 5
"""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.panda_pick_env import PandaPickEnv


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise trained PPO agent")
    parser.add_argument("--model",    type=str, required=True,
                        help="Path to .zip model file")
    parser.add_argument("--stats",    type=str, default=None,
                        help="Path to vec_normalize.pkl (if training used normalisation)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes to run (0 = run forever)")
    parser.add_argument("--object",   type=str, default=None,
                        choices=PandaPickEnv.OBJECT_NAMES,
                        help="Fix the object type (default: random each episode)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions (default: True)")
    parser.add_argument("--step-delay", type=float, default=0.02,
                        help="Seconds to sleep between steps (slow down for viewing)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Fixed-object wrapper (optional)
# ---------------------------------------------------------------------------

class FixedObjectEnv(PandaPickEnv):
    """Thin wrapper that always spawns the same object type."""
    def __init__(self, object_name: str, **kwargs):
        super().__init__(**kwargs)
        self._fixed_object = object_name

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # If the randomised object doesn't match, reload with the fixed one
        if self.current_object_name != self._fixed_object:
            self.current_object_name = self._fixed_object
            self.current_object_idx  = self.OBJECT_NAMES.index(self._fixed_object)
            self._load_model(self._fixed_object)
            import mujoco
            mujoco.mj_resetData(self.model, self.data)
            # Re-randomise position
            obj_x = float(self.np_random.uniform(0.3, 0.55))
            obj_y = float(self.np_random.uniform(-0.2, 0.2))
            self.data.qpos[self._obj_qpos_start + 0] = obj_x
            self.data.qpos[self._obj_qpos_start + 1] = obj_y
            self.data.qpos[self._obj_qpos_start + 2] = 0.03
            self.data.qpos[self._obj_qpos_start + 3] = 1.0
            self.data.qpos[self._obj_qpos_start + 4] = 0.0
            self.data.qpos[self._obj_qpos_start + 5] = 0.0
            self.data.qpos[self._obj_qpos_start + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            obs = self._get_obs()
        return obs, {"object_name": self._fixed_object}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print(f"  Model    : {args.model}")
    print(f"  Stats    : {args.stats or 'none (no normalisation)'}")
    print(f"  Object   : {args.object or 'random'}")
    print(f"  Episodes : {'∞' if args.episodes == 0 else args.episodes}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Build the raw env
    # ------------------------------------------------------------------
    if args.object:
        env_cls = lambda: FixedObjectEnv(args.object, max_episode_steps=200, render_mode="human")
    else:
        env_cls = lambda: PandaPickEnv(max_episode_steps=200, render_mode="human")


    raw_env = DummyVecEnv([env_cls])

    # ------------------------------------------------------------------
    # Wrap with VecNormalize if stats were saved during training
    # ------------------------------------------------------------------
    if args.stats:
        print(f"\nLoading normalisation stats from {args.stats}...")
        vec_env = VecNormalize.load(args.stats, raw_env)
        vec_env.training = False      # freeze stats — do NOT update them
        vec_env.norm_reward = False   # no need to normalise reward at eval
    else:
        vec_env = raw_env

    # ------------------------------------------------------------------
    # Load the PPO model
    # ------------------------------------------------------------------
    print(f"Loading model from {args.model}...\n")
    model = PPO.load(args.model, env=vec_env)

    # ------------------------------------------------------------------
    # Run episodes
    # ------------------------------------------------------------------
    ep          = 0
    total_steps = 0
    successes   = 0

    print("Controls: close the MuJoCo viewer window to stop.\n")

    while True:
        ep += 1
        obs   = vec_env.reset()
        done  = False
        ep_reward   = 0.0
        ep_steps    = 0
        ep_success  = False

        # Pull the underlying env to read object name
        underlying = vec_env.envs[0] if hasattr(vec_env, "envs") else None

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = vec_env.step(action)

            ep_reward += float(reward[0])
            ep_steps  += 1

            if info[0].get("is_success", False):
                ep_success = True

            # Honour the step delay (slow down for human viewing)
            if args.step_delay > 0:
                time.sleep(args.step_delay)

            if done[0]:
                break

        successes   += int(ep_success)
        total_steps += ep_steps
        obj_name = info[0].get("object_name", "?")

        print(
            f"Episode {ep:>4}  |  object={obj_name:<10}  |  "
            f"reward={ep_reward:+8.2f}  |  steps={ep_steps:>4}  |  "
            f"success={'✓' if ep_success else '✗'}  |  "
            f"success_rate={successes/ep:.1%}"
        )

        if args.episodes > 0 and ep >= args.episodes:
            break

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Episodes run   : {ep}")
    print(f"  Total steps    : {total_steps:,}")
    print(f"  Successes      : {successes} / {ep} ({successes/ep:.1%})")
    print("=" * 60)

    vec_env.close()


if __name__ == "__main__":
    main()