"""
visualise.py
============
Visualize a trained agent on the Panda pick-and-lift environmentand watch it run in the MuJoCo interactive viewer.

Usage
-----
    # Recommended (auto path based on algorithm)
    For PPO:
    python visualise.py --algo PPO

    For SAC:
    python visualise.py --algo SAC

    For TD3:
    python visualise.py --algo TD3

    # With fixed object and limited episodes
    python visualise.py --algo SAC --object cube --episodes 10
"""

import argparse
import time
import mujoco

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.panda_pick_env import PandaPickEnv


algo_dict = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise trained Panda Emika Franka agent")
    parser.add_argument("--algo", type=str, default="PPO", choices=list(algo_dict.keys()),
                        help="Algorithm to use for training and visualisation (default: PPO)")
    parser.add_argument("--model",    type=str, default=None,
                        help="Full path to .zip model file (overrides auto-path based on algo)")
    parser.add_argument("--stats",    type=str, default=None,
                        help="Path to vec_normalize.pkl")
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
# Fixed Object Wrapper
# ---------------------------------------------------------------------------
class FixedObjectEnv(PandaPickEnv):
    def __init__(self, object_name: str, **kwargs):
        super().__init__(**kwargs)
        self._fixed_object = object_name

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.current_object_name != self._fixed_object:
            self.current_object_name = self._fixed_object
            # Force reload the desired object
            self.current_object_idx = self.OBJECT_NAMES.index(self._fixed_object)
            self._load_model(self._fixed_object)

            # Reset simulation
            mujoco.mj_resetData(self.model, self.data)

            # Randomize position slightly
            obj_x = float(self.np_random.uniform(0.3, 0.55))
            obj_y = float(self.np_random.uniform(-0.2, 0.2))
            self.data.qpos[self._obj_qpos_start:self._obj_qpos_start+7] = [obj_x, obj_y, 0.03, 1, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)
            obs = self._get_obs()
        return obs, info


# Main

def main():
    args = parse_args()

    # Auto path building based on folder structure if user didn't specify model
    if args.model is None:
        base = f"models/panda_{args.algo.lower()}"
        args.model = f"{base}/best_model.zip"
        if args.stats is None:
            args.stats = f"{base}/vec_normalize.pkl"

    print("=" * 60)
    print(f"  Algorithm : {args.algo}")
    print(f"  Model    : {args.model}")
    print(f"  Stats    : {args.stats or 'none (no normalisation)'}")
    print(f"  Object   : {args.object or 'random'}")
    print(f"  Episodes : {'∞' if args.episodes == 0 else args.episodes}")
    print("=" * 60)


   # Create environment
    if args.object:
        env_cls = lambda: FixedObjectEnv(args.object, max_episode_steps=200, render_mode="human")
    else:
        env_cls = lambda: PandaPickEnv(max_episode_steps=200, render_mode="human")


    vec_env = DummyVecEnv([env_cls])

    # Load normalisation stats
    if args.stats:
        print(f"\nLoading normalisation stats from {args.stats}...")
        vec_env = VecNormalize.load(args.stats, vec_env)
        vec_env.training = False      # freeze stats — do NOT update them
        vec_env.norm_reward = False   # no need to normalise reward at eval
    else:
        print("\nWarning: No normalisation stats provided, running without normalisation.")

    # Load model with correct algorithm
    AlgoClass = algo_dict[args.algo]
    model = AlgoClass.load(args.model, env=vec_env)

    print(f"\nStarting visualization... (Close MuJoCo window to stop)\n")



    # Run episodes
    ep          = 0
    total_steps = 0
    successes   = 0

    while True:
        ep += 1
        obs   = vec_env.reset()
        done  = False
        ep_reward   = 0.0
        ep_steps    = 0
        ep_success  = False


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


    # Summary
    print("\n" + "=" * 60)
    print(f"  Episodes run   : {ep}")
    print(f"  Total steps    : {total_steps:,}")
    print(f"  Successes      : {successes} / {ep} ({successes/ep:.1%})")
    print("=" * 60)

    vec_env.close()


if __name__ == "__main__":
    main()


